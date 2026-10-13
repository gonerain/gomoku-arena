"""引擎子进程管理：stdin/stdout 收发 NDJSON，stderr 直通日志。

平台与引擎的全部接口就是本模块：一个 JSON 一行，stdin 收、stdout 发。
引擎的调试输出必须走 stderr，平台会原样转发到自己的 stderr，不参与协议解析。

健壮性：超时、崩溃、非法输出、进程树清理都在这里处理。
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time

try:  # psutil 可选：作为进程树清理的兜底手段之一
    import psutil
except Exception:  # pragma: no cover
    psutil = None

# Windows 创建标志。CREATE_SUSPENDED 不在 subprocess 的公开常量里，用原始值。
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000


class EngineError(Exception):
    """引擎侧故障，导致该方判负。code 直接写进棋谱的 result.reason。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class EngineProc:
    """一个引擎进程。生命周期：start() -> send/recv 若干轮 -> close()。"""

    def __init__(self, spec: dict, label: str):
        self.spec = spec
        self.label = label
        self.cmd = spec["cmd"]
        self.cwd = spec.get("cwd")
        self.env_extra = spec.get("env") or {}
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._closed = False
        self._job = None

    # ---------- 生命周期 ----------

    def start(self) -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.update(self.env_extra)

        # 先建 Job Object（kill-on-close），并让子进程挂起启动：
        # 这样"加入 Job"发生在子进程执行第一条指令之前，它之后派生的任何后代
        # 都必然落在同一个 Job 里，关掉 Job 句柄即可被操作系统整棵端掉。
        job = _create_kill_on_close_job()
        creationflags = 0
        if job is not None and os.name == "nt":
            # 挂起启动 + 先入 Job 再恢复，杜绝"子进程抢在入 Job 前又派生后代"的竞态
            creationflags = CREATE_SUSPENDED | CREATE_NO_WINDOW
        elif os.name == "nt":
            creationflags = CREATE_NO_WINDOW

        try:
            self.proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise EngineError("START_FAILED", f"无法启动引擎 {self.label}: {exc}") from exc

        if job is not None:
            self._job = job
            if not _assign_to_job(job, self.proc):
                sys.stderr.write(
                    f"[engine_host] 警告: {self.label} 无法加入 Job Object，"
                    f"进程树清理将退化为 best-effort\n"
                )
            _resume_process(self.proc)

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)  # 流结束：进程已退出

    def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            sys.stderr.write(f"[{self.label}] {line.rstrip()}\n")
        sys.stderr.flush()

    def close(self) -> None:
        """关闭 stdin 并清理整棵进程树。无论如何都必须走到清理，不能被异常打断。"""
        if self._closed:
            return
        self._closed = True
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        _kill_tree(proc, job=self._job)
        _close_handle(self._job)
        self._job = None

    # ---------- 收发 ----------

    def send(self, frame: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise EngineError("CRASH", f"{self.label} 进程不存在")
        if proc.poll() is not None:
            raise EngineError("CRASH", f"{self.label} 已退出 (code={proc.returncode})")
        try:
            proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineError("CRASH", f"{self.label} 管道写入失败: {exc}") from exc

    def recv(self, deadline_ms: int) -> dict:
        """在 deadline_ms 内读取一个合法 JSON 帧。超时/崩溃/脏输出都抛 EngineError。

        判定规则：stdout 出现 EOF = 进程树已终止（含已崩溃）= CRASH；
        未在时限内读到帧 = TIMEOUT，附带一次进程状态探测，便于事后定位原因。
        """
        deadline = time.monotonic() + deadline_ms / 1000.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError("TIMEOUT", f"{self.label} 超过 {deadline_ms}ms 未落子")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise EngineError("TIMEOUT", f"{self.label} 超过 {deadline_ms}ms 未落子{self._exit_note()}") from None
            if line is None:
                code = self.proc.returncode if self.proc else None
                raise EngineError("CRASH", f"{self.label} 进程已退出 (code={code})")
            line = line.strip()
            if not line:
                continue  # 空行忽略
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EngineError(
                    "BAD_OUTPUT", f"{self.label} stdout 非 JSON: {line[:120]!r} ({exc.msg})"
                ) from exc
            if not isinstance(frame, dict) or "type" not in frame:
                raise EngineError("BAD_OUTPUT", f"{self.label} 缺少 type 字段: {line[:120]!r}")
            return frame

    def _exit_note(self) -> str:
        """超时时探测一次子进程状态，只为让错误信息更有诊断价值。"""
        proc = self.proc
        if proc is None:
            return ""
        try:
            code = proc.poll()
        except Exception:
            return ""
        if code is None:
            return ""
        return f"（子进程已退出 code={code}，可能是它派生的后代仍持有 stdout 管道）"


def _kill_tree(proc: subprocess.Popen, job=None, grace: float = 0.3) -> None:
    """确保进程及其所有后代都被结束。

    引擎完全可能派生子进程（搜索 worker、DataLoader 等），只杀父进程会留下孤儿
    进程吃 CPU。清理力度按可靠性从高到低排：

      1. Job Object（Windows，kill-on-close）—— 操作系统级保证整棵树的成员一起死，
         不管后代是否已被 reparent。这是唯一能可靠处理"父进程先退出、孙进程被
        重新挂到别的父进程下"这种情形的手段。
      2. psutil 递归枚举 —— 父进程仍存活时有效。
      3. taskkill /F /T（Windows）、killpg（POSIX）—— 兜底。
    """
    # 1) 先给 grace 秒体面退出（stdin 已关闭，正常引擎会自己收尾）
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass

    # 2) Job Object：关句柄即整棵树终结
    if job is not None:
        _close_handle(job)
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _safe_kill(proc)
        return

    # 3) psutil 递归（父进程还活着才有意义）
    if psutil is not None and proc.poll() is None:
        try:
            parent = psutil.Process(proc.pid)
            victims = parent.children(recursive=True) + [parent]
            for victim in victims:
                _safe_kill(victim)
            _gone, alive = psutil.wait_procs(victims, timeout=1.0)
            for victim in alive:
                _safe_kill(victim)
            if alive:
                psutil.wait_procs(alive, timeout=1.0)
            return
        except Exception:
            pass

    # 4) 系统工具兜底
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
    else:  # pragma: no cover - 仅在 POSIX 上执行
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except OSError:
            _safe_kill(proc)

    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _safe_kill(proc)


def _safe_kill(target) -> None:
    try:
        target.kill()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Windows Job Object：kill-on-close
#
# 为什么需要它：引擎可能派生子进程，而父进程一旦退出，后代会被操作系统 reparent，
# 此时无论按父 pid 递归枚举还是 taskkill /T 都找不到它们。Job Object 是内核级的
# 进程分组，把子进程放进 Job 后，关闭 Job 句柄会让组内所有进程（含所有后代）一起
# 终结 —— 与它们被 reparent 到谁名下无关。
# ---------------------------------------------------------------------------

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_THREAD_SUSPEND_RESUME = 0x0002
_TH32CS_SNAPTHREAD = 0x00000004
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_JOB_AVAILABLE = hasattr(ctypes, "WinDLL")


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JobExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _ThreadEntry32(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ThreadID", ctypes.c_uint32),
        ("th32OwnerProcessID", ctypes.c_uint32),
        ("tpBasePri", ctypes.c_int32),
        ("tpDeltaPri", ctypes.c_int32),
        ("dwFlags", ctypes.c_uint32),
    ]


def _create_kill_on_close_job():
    """建一个"句柄关闭即杀死组内所有进程"的 Job。失败返回 None，退回 best-effort。"""
    if not _JOB_AVAILABLE or os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _JobExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            ctypes.c_void_p(job),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(ctypes.c_void_p(job))
            return None
        return job
    except Exception:
        return None


def _assign_to_job(job, proc: subprocess.Popen) -> bool:
    if not _JOB_AVAILABLE or os.name != "nt":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        handle = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, proc.pid)
        if not handle:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(handle)))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    except Exception:
        return False


def _resume_process(proc: subprocess.Popen) -> None:
    """恢复被 CREATE_SUSPENDED 挂起的进程。

    挂起进程此刻只有一个线程，用 Toolhelp32 枚举它的线程 id，再 OpenThread 恢复。
    （subprocess 不暴露线程句柄，所以只能这么找。）
    """
    if not _JOB_AVAILABLE or os.name != "nt":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        kernel32.Thread32First.restype = ctypes.c_int
        kernel32.Thread32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ThreadEntry32)]
        kernel32.Thread32Next.restype = ctypes.c_int
        kernel32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ThreadEntry32)]
        kernel32.OpenThread.restype = ctypes.c_void_p
        kernel32.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.ResumeThread.restype = ctypes.c_uint32
        kernel32.ResumeThread.argtypes = [ctypes.c_void_p]

        snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
        if snapshot == _INVALID_HANDLE_VALUE or not snapshot:
            sys.stderr.write("[engine_host] 警告: 无法创建线程快照，子进程可能仍处于挂起\n")
            return
        try:
            entry = _ThreadEntry32()
            entry.dwSize = ctypes.sizeof(_ThreadEntry32)
            ok = kernel32.Thread32First(ctypes.c_void_p(snapshot), ctypes.byref(entry))
            while ok:
                if entry.th32OwnerProcessID == proc.pid:
                    thread = kernel32.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                    if thread:
                        kernel32.ResumeThread(ctypes.c_void_p(thread))
                        kernel32.CloseHandle(ctypes.c_void_p(thread))
                    return
                ok = kernel32.Thread32Next(ctypes.c_void_p(snapshot), ctypes.byref(entry))
            sys.stderr.write(f"[engine_host] 警告: 未找到 pid={proc.pid} 的线程，无法恢复运行\n")
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(snapshot))
    except Exception as exc:
        sys.stderr.write(f"[engine_host] 警告: 恢复挂起进程失败: {exc}\n")


def _close_handle(handle) -> None:
    if not handle or not _JOB_AVAILABLE or os.name != "nt":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle(ctypes.c_void_p(handle))
    except Exception:
        pass
