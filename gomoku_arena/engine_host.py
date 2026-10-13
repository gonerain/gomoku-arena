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
        self._job: JobHandle | None = None
        self._job_assigned = False
        self._exit_code: int | None = None

    # ---------- 生命周期 ----------

    @property
    def exit_code(self) -> int | None:
        """子进程退出码；仍在运行或尚未观测到退出时为 None。"""
        if self._exit_code is None and self.proc is not None:
            self._exit_code = self._reap()
        return self._exit_code

    def _reap(self) -> int | None:
        """回收子进程并取回退出码。

        收到 stdout EOF 的瞬间 Popen.returncode 往往还是 None（读端先看到管道关闭，
        进程退出状态还没被回收）。不补一次 wait，判负记录里就只能写"退出码未知"，
        丢掉"退出码 2 = 模型加载失败"这类关键信息。
        """
        proc = self.proc
        if proc is None:
            return None
        try:
            return proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return proc.returncode
        except Exception:
            return proc.returncode

    def start(self) -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.update(self.env_extra)

        # Windows：先建 kill-on-close 的 Job，再以挂起态启动子进程，入 Job 后才恢复运行。
        # 这样"加入 Job"发生在子进程执行第一条指令之前，它之后派生的任何后代都必然
        # 落在同一个 Job 里。Job 不可用时退回普通启动 + best-effort 清理。
        job = _create_kill_on_close_job()
        is_windows = os.name == "nt"
        creationflags = CREATE_NO_WINDOW if is_windows else 0
        suspended = job is not None and is_windows
        if suspended:
            creationflags |= CREATE_SUSPENDED

        kwargs = {}
        if not is_windows:
            # POSIX：让子进程自成会话/进程组，killpg 才不会波及裁判自身
            kwargs["start_new_session"] = True

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
                **kwargs,
            )
        except OSError as exc:
            if job is not None:
                job.close()
            raise EngineError("START_FAILED", f"无法启动引擎 {self.label}: {exc}") from exc

        if suspended:
            self._finish_suspended_start(job)
        elif job is not None:
            job.close()  # 没用上就立刻归还，避免句柄泄漏

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _finish_suspended_start(self, job: "JobHandle") -> None:
        """把挂起的子进程放进 Job 并恢复运行；任一步失败都不能留下永久挂起的进程。"""
        assert self.proc is not None
        if _assign_to_job(job, self.proc):
            self._job = job
            self._job_assigned = True
        else:
            # 入 Job 失败：这个 Job 对清理已无意义，立刻关闭归还，
            # 并退回 psutil / taskkill 的 best-effort 路径。
            job.close()
            sys.stderr.write(
                f"[engine_host] 警告: {self.label} 无法加入 Job Object，"
                f"进程树清理退化为 best-effort\n"
            )

        if not _resume_process(self.proc):
            # 恢复失败会让子进程永远挂起，必须就地终结并向上报错
            _safe_kill(self.proc)
            _kill_tree(self.proc, job=self._job if self._job_assigned else None)
            raise EngineError(
                "START_FAILED", f"{self.label} 无法从挂起态恢复，已终止该进程"
            )

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
        """关闭 stdin 并清理整棵进程树。清理必须无条件执行，不能被异常打断。"""
        if self._closed:
            return
        self._closed = True
        proc = self.proc
        job = self._job if self._job_assigned else None
        if proc is None:
            if job is not None:
                job.close()
            self._job = None
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        _kill_tree(proc, job=job)
        if self._job is not None:
            # _kill_tree 已负责关闭入 Job 的句柄；未入 Job 的句柄已在上一步归还
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

        判定规则：stdout 出现 EOF = 进程树已终止（含崩溃、含管道被后代关闭）= CRASH；
        未在时限内读到帧 = TIMEOUT，附带一次进程状态探测以便事后定位。
        """
        deadline = time.monotonic() + deadline_ms / 1000.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError(
                    "TIMEOUT", f"{self.label} 超过 {deadline_ms}ms 未落子{self._exit_note()}"
                ) from None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise EngineError(
                    "TIMEOUT", f"{self.label} 超过 {deadline_ms}ms 未落子{self._exit_note()}"
                ) from None
            if line is None:
                self._exit_code = self._reap()
                raise EngineError("CRASH", f"{self.label} {self._exit_message(self._exit_code)}")
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

    def _exit_message(self, code: int | None) -> str:
        """把退出码翻译成明确的判负原因，而不是笼统的"进程已退出"。"""
        if code is None:
            return "进程已退出（stdout 管道已关闭，可能已被系统回收）"
        if code == 0:
            return "进程已退出（退出码 0：正常退出但没有给出着法）"
        return f"进程异常退出（退出码 {code}）"

    def _exit_note(self) -> str:
        """超时时探测一次子进程状态，只为让错误信息更有诊断价值。"""
        code = self._exit_code if self._exit_code is not None else (
            self.proc.returncode if self.proc else None
        )
        if code is None:
            return ""
        return f"（子进程已退出{'，退出码 ' + str(code) if code is not None else ''}，" \
               f"可能是它派生的后代仍持有 stdout 管道）"


def _kill_tree(proc: subprocess.Popen, job: "JobHandle | None" = None, grace: float = 0.3) -> None:
    """确保进程及其所有后代都被结束。

    引擎完全可能派生子进程（搜索 worker、DataLoader 等），只杀父进程会留下孤儿
    进程吃 CPU。清理力度按可靠性从高到低排：

      1. Job Object（Windows，kill-on-close）—— 内核级保证整棵树一起终结，
         不管后代是否已被 reparent。这是唯一能可靠处理"父进程先退出、孙进程
         被重新挂到别的父进程下"这种情形的手段。仅在成功入 Job 时才传 job。
      2. psutil 递归枚举 —— 父进程仍存活时有效。
      3. taskkill /F /T（Windows）、killpg（POSIX，子进程自成进程组）—— 兜底。
    """
    # 1) 先给 grace 秒体面退出（stdin 已关闭，正常引擎会自己收尾）
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass

    # 2) Job Object：关句柄即整棵树终结（同一声明周期内重复调用安全，只关一次）
    if job is not None:
        job.close()

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
        _kill_process_group(proc)

    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _safe_kill(proc)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """POSIX 兜底：只杀子进程自己的进程组。

    必须先确认子进程确实是组长（pgid == pid）。否则 killpg 会打到裁判所在的
    进程组，把裁判自己杀掉 —— 这比漏杀更严重。
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        _safe_kill(proc)
        return
    if pgid != proc.pid:
        sys.stderr.write(
            f"[engine_host] 警告: pid={proc.pid} 不是进程组组长 (pgid={pgid})，"
            f"跳过 killpg 以免误杀裁判，改为只杀该进程\n"
        )
        _safe_kill(proc)
        return
    try:
        os.killpg(pgid, 9)
    except OSError:
        _safe_kill(proc)


def _safe_kill(target) -> None:
    try:
        target.kill()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Windows 句柄所有权
#
# 为什么不用"全局已关闭句柄值集合"：Windows 的句柄值会被系统复用，一旦某个值被
# 回收后又分配给新句柄，按值记账就会误判"已经关过了"从而漏关，或对已复用的句柄
# 重复关闭。正确做法是让每个句柄对象的生命周期自带 closed 标记，谁创建谁负责关一次。
# ---------------------------------------------------------------------------

class JobHandle:
    """一个 kill-on-close 的 Job Object 句柄，自带"只关一次"的生命周期标记。"""

    __slots__ = ("value", "foreign_function", "_closed", "_lock")

    def __init__(self, value: int, foreign_function=None):
        self.value = int(value)
        # 持有 WinDLL 的引用，防止 ctypes 把 ForeignFunction 回收掉导致句柄失效
        self.foreign_function = foreign_function
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> bool:
        """关闭句柄。重复调用是安全的，第二次不再触碰内核对象。"""
        with self._lock:
            if self._closed:
                return False
            self._closed = True
        if not _JOB_AVAILABLE or os.name != "nt":
            return True
        try:
            kernel32 = self.foreign_function or ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle(ctypes.c_void_p(self.value))
        except Exception:
            pass
        return True

    def __enter__(self) -> "JobHandle":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


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


def _create_kill_on_close_job() -> "JobHandle | None":
    """建一个"句柄关闭即杀死组内所有进程"的 Job。失败返回 None，退回 best-effort。"""
    if not _JOB_AVAILABLE or os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        raw = kernel32.CreateJobObjectW(None, None)
        if not raw:
            return None
        handle = JobHandle(raw, foreign_function=kernel32)
        info = _JobExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            ctypes.c_void_p(handle.value),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            handle.close()  # 配置失败：立刻归还这个刚创建的 Job
            return None
        return handle
    except Exception:
        return None


def _assign_to_job(job: "JobHandle", proc: subprocess.Popen) -> bool:
    if not _JOB_AVAILABLE or os.name != "nt" or job is None or job.closed:
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
            for attempt in (1, 2):  # 极小概率的瞬时失败，重试一次即可
                if kernel32.AssignProcessToJobObject(
                    ctypes.c_void_p(job.value), ctypes.c_void_p(handle)
                ):
                    return True
                if attempt == 1:
                    time.sleep(0.05)
            return False
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    except Exception:
        return False


def _resume_process(proc: subprocess.Popen) -> bool:
    """恢复被 CREATE_SUSPENDED 挂起的进程，返回是否成功。

    挂起进程此刻只有一个线程，用 Toolhelp32 枚举它的线程 id，再 OpenThread 恢复。
    （subprocess 不暴露线程句柄，所以只能这么找。）
    """
    if not _JOB_AVAILABLE or os.name != "nt":
        return False
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
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            sys.stderr.write("[engine_host] 警告: 无法创建线程快照，子进程可能仍处于挂起\n")
            return False
        resumed = False
        try:
            entry = _ThreadEntry32()
            entry.dwSize = ctypes.sizeof(_ThreadEntry32)
            ok = kernel32.Thread32First(ctypes.c_void_p(snapshot), ctypes.byref(entry))
            while ok:
                if entry.th32OwnerProcessID == proc.pid:
                    thread = kernel32.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                    if thread:
                        try:
                            # ResumeThread 返回 0xFFFFFFFF 表示失败
                            resumed = kernel32.ResumeThread(ctypes.c_void_p(thread)) != 0xFFFFFFFF
                        finally:
                            kernel32.CloseHandle(ctypes.c_void_p(thread))
                    break
                ok = kernel32.Thread32Next(ctypes.c_void_p(snapshot), ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(snapshot))
        if not resumed:
            sys.stderr.write(f"[engine_host] 警告: 未能恢复 pid={proc.pid} 的线程\n")
        return resumed
    except Exception as exc:
        sys.stderr.write(f"[engine_host] 警告: 恢复挂起进程失败: {exc}\n")
        return False
