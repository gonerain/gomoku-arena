"""单局对局编排：启动两个引擎进程，跑完整一局，产出棋谱 dict。

裁判是唯一权威：落子合法性、单步超时、崩溃、脏输出全部由这里判定。
引擎协议（全部 4 种帧）见 rules.PROTOCOL。

身份模型（三层分离，避免"同一个适配器加载不同权重"时混淆）：
    participant 参赛方  —— 统计与棋谱的唯一标识（如 dsh / codex）
    engine      引擎程序 —— 说协议的适配器进程（DSH 与 Codex 可以用同一个）
    checkpoint  模型权重 —— 同一适配器加载的不同权重
DSH 与 Codex 完全可以共用 `trained` 适配器，只要 participant 与 checkpoint 不同。

超时归责原则（公平性）：
  * 单步超时 = 该方自己的问题，判该方负（reason=timeout）。
  * 整局墙钟上限是"平台为了防止对局无限拖下去"而设的护栏，双方共享同一份时间，
    因此它被触发时不能归责于当前落子方 —— 判为无胜负中止（reason=game_timeout）。
  * 握手阶段的超时同样无法可靠归责（可能是引擎加载慢，也可能是它挂了），
    一律判为 aborted，不计入任何一方的败绩。
  * 单方启动失败是明确的例外：那是该方进程根本没起来，归责于该方。
"""

from __future__ import annotations

import time
import traceback
import uuid
from datetime import datetime, timezone

from .engine_host import EngineError, EngineProc
from .rules import BOARD_SIZE, COLOR_CODE, Board, other

# 归因到落败方的错误码（其余视为平台侧故障或不归责的中止）
PLAYER_FAULTS = {
    "TIMEOUT",
    "CRASH",
    "BAD_OUTPUT",
    "ILLEGAL_MOVE",
    "PROTOCOL",
    "RESIGN",
    "START_FAILED",
}

DEFAULT_CHECKPOINT = "none"


class Participant:
    """一个参赛方：谁在下棋，用哪个引擎程序，加载哪份权重。"""

    def __init__(
        self,
        pid: str,
        name: str | None = None,
        engine: str | None = None,
        version: str = "v0",
        checkpoint: str | None = None,
        meta: dict | None = None,
        spec: dict | None = None,
    ):
        if not pid:
            raise ValueError("参赛方必须有唯一 id")
        self.pid = pid
        self.name = name or pid
        self.engine = engine or pid          # 引擎程序标识；可选集合见 server.ENGINES
        self.version = version
        self.checkpoint = checkpoint or DEFAULT_CHECKPOINT
        self.meta = dict(meta or {})
        self.spec = dict(spec or {})
        self.spec.setdefault("name", self.name)
        self.spec.setdefault("version", self.version)

    def for_color(self, color: str) -> dict:
        """该参赛方执某色时的棋谱记录片段。"""
        return {
            "pid": self.pid,
            "name": self.name,
            "engine": self.engine,
            "version": self.version,
            "color": color,
            "checkpoint": self.checkpoint,
            "meta": dict(self.meta),
        }

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "name": self.name,
            "engine": self.engine,
            "version": self.version,
            "checkpoint": self.checkpoint,
            "meta": dict(self.meta),
        }


def participant_from_spec(spec: dict, color: str | None = None, pid: str | None = None) -> Participant:
    """把旧的"裸引擎规格"包装成参赛方，保持向后兼容。"""
    meta = dict(spec.get("meta") or {})
    return Participant(
        pid=pid or spec.get("name") or "engine",
        name=spec.get("name") or pid or "engine",
        engine=spec.get("engine") or spec.get("name") or pid or "engine",
        version=str(spec.get("version") or "v0"),
        checkpoint=spec.get("checkpoint") or meta.get("checkpoint"),
        meta=meta,
        spec=spec,
    )


def check_distinct(black: Participant, white: Participant) -> None:
    """两份配置必须真的不同，否则"两个模型对战"就是自欺欺人。

    只有**非基线**引擎（即真正的模型接入点）才强制权重必须不同：
    基线引擎（random/greedy）之间同配置对局是有意义的自检，不拦。
    """
    if black.pid == white.pid:
        raise ValueError(
            f"黑方与白方使用了相同的参赛方 id {black.pid!r}；"
            f"DSH 与 Codex 必须用不同的 pid 区分身份"
        )
    if black.engine != white.engine or black.checkpoint == DEFAULT_CHECKPOINT:
        return
    if black.checkpoint == white.checkpoint and black.version == white.version:
        raise ValueError(
            f"双方用同一个引擎加载了同一份权重（{black.checkpoint}）且版本相同："
            f"这不会产生有意义的对比结果。请为双方指定各自的 checkpoint"
        )


class MatchConfig:
    def __init__(
        self,
        seed: int | None = None,
        board_size: int = BOARD_SIZE,
        move_timeout_ms: int = 1000,
        game_timeout_ms: int = 120_000,
    ):
        if board_size != BOARD_SIZE:
            raise ValueError(
                f"棋盘尺寸固定为 {BOARD_SIZE}x{BOARD_SIZE}（规则见 rules.py 与 README），"
                f"收到 {board_size}"
            )
        self.seed = seed if seed is not None else int(time.time()) % 1_000_000
        self.board_size = board_size
        self.move_timeout_ms = move_timeout_ms
        self.game_timeout_ms = game_timeout_ms


class Match:
    def __init__(
        self,
        black,
        white,
        config: MatchConfig | None = None,
        on_progress=None,
        batch: dict | None = None,
    ):
        self.config = config or MatchConfig()
        self.participants = {
            "black": black if isinstance(black, Participant) else participant_from_spec(black, "black"),
            "white": white if isinstance(white, Participant) else participant_from_spec(white, "white"),
        }
        check_distinct(self.participants["black"], self.participants["white"])
        self.on_progress = on_progress
        self.batch = batch  # 批量对战的归属信息，会写进棋谱
        self.procs: dict[str, EngineProc] = {}
        # 棋谱身份以参赛方为唯一标识；引擎自报信息另存，便于发现"注册名与自报名不一致"
        self.identity = {
            color: p.for_color(color) for color, p in self.participants.items()
        }
        self.game_id = f"g-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"

    # ---------- 主流程 ----------

    def play(self) -> dict:
        board = Board(self.config.board_size)
        moves: list[dict] = []
        result: dict = {"winner": None, "reason": "aborted", "line": None}
        error: dict | None = None
        started = time.monotonic()
        global_deadline = started + self.config.game_timeout_ms / 1000.0
        turn_color = "black"

        try:
            self._handshake()
            if self.on_progress:
                self.on_progress(moves, turn_color)  # 握手完成：让外部拿到引擎自报版本

            while True:
                if time.monotonic() >= global_deadline:
                    # 平台护栏触发，双方共同承担，不判任何一方负
                    result = {"winner": None, "reason": "game_timeout", "line": None}
                    error = {
                        "color": None,
                        "reason": "GAME_TIMEOUT",
                        "message": self._global_timeout_message(),
                    }
                    break
                try:
                    mv, ms = self._ask(turn_color, board, moves, global_deadline)
                except EngineError as exc:
                    if exc.code == "GAME_TIMEOUT":
                        # 平台护栏触发：无胜负中止，谁都不判负
                        result = {"winner": None, "reason": "game_timeout", "line": None}
                        error = {"color": None, "reason": exc.code, "message": exc.message}
                        break
                    color = getattr(exc, "color", None) or turn_color
                    result = {
                        "winner": other(color),
                        "reason": exc.code.lower(),
                        "line": None,
                    }
                    error = {"color": color, "reason": exc.code, "message": exc.message}
                    break

                moves.append(
                    {"n": len(moves) + 1, "color": turn_color, "x": mv[0], "y": mv[1], "ms": ms}
                )
                board.place(mv[0], mv[1], COLOR_CODE[turn_color])

                line = board.win_line(mv[0], mv[1], COLOR_CODE[turn_color])
                if line:
                    result = {"winner": turn_color, "reason": "five", "line": line}
                    break
                if board.is_full():
                    result = {"winner": None, "reason": "draw", "line": None}
                    break
                turn_color = other(turn_color)
                if self.on_progress:
                    self.on_progress(moves, turn_color)

        except EngineError as exc:
            color = getattr(exc, "color", None)
            if color and exc.code in PLAYER_FAULTS:
                result = {"winner": other(color), "reason": exc.code.lower(), "line": None}
                error = {"color": color, "reason": exc.code, "message": exc.message}
            else:
                # 无法可靠归责（例如握手阶段超时）：中止而不判负
                result = {"winner": None, "reason": "aborted", "line": None}
                error = {"color": color, "reason": exc.code, "message": exc.message}
        except Exception as exc:  # 平台自身故障：留痕，不静默
            traceback.print_exc()
            result = {"winner": None, "reason": "platform_error", "line": None}
            error = {"color": None, "reason": type(exc).__name__, "message": str(exc)}
        finally:
            self._notify_gameover(result)
            for proc in self.procs.values():
                proc.close()

        record = {
            "v": 2,
            "game_id": self.game_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "board_size": self.config.board_size,
            "seed": self.config.seed,
            "move_timeout_ms": self.config.move_timeout_ms,
            "game_timeout_ms": self.config.game_timeout_ms,
            "participants": [p.to_dict() | {"color": color}
                             for color, p in self.participants.items()],
            "black": self.identity["black"],
            "white": self.identity["white"],
            "moves": moves,
            "result": result,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        if error:
            record["error"] = error
        if self.batch:
            record["batch"] = self.batch
        return record

    # ---------- 内部步骤 ----------

    def _handshake(self) -> None:
        for color in ("black", "white"):
            proc = EngineProc(self.participants[color].spec, color)
            self.procs[color] = proc
            try:
                proc.start()
                proc.send(
                    {
                        "type": "init",
                        "game_id": self.game_id,
                        "board_size": self.config.board_size,
                        "color": color,
                        "seed": self.config.seed,
                    }
                )
            except EngineError as exc:
                raise _blame(exc, color) from exc

        for color in ("black", "white"):
            try:
                frame = self.procs[color].recv(self._handshake_timeout_ms())
            except EngineError as exc:
                # 无论是超时还是崩溃，责任方都是"正在等 ready 的那一方"
                raise _blame(exc, color) from exc
            if frame.get("type") != "ready":
                raise _blame(
                    EngineError("PROTOCOL", f"握手期望 ready，收到 {frame.get('type')!r}"), color
                )
            info = self.identity[color]
            spec_meta = dict(self.participants[color].meta)
            self_meta = frame.get("meta") if isinstance(frame.get("meta"), dict) else {}
            # 引擎自报的 meta 与平台声明合并；平台声明的参赛身份字段优先
            info["meta"] = {**spec_meta, **self_meta}
            info["checkpoint"] = str(info["meta"].get("checkpoint") or info["checkpoint"])
            info["reported"] = {
                "name": str(frame.get("name") or ""),
                "version": str(frame.get("version") or ""),
            }
            if info["reported"]["name"] and info["reported"]["name"] != info["name"]:
                info["name_mismatch"] = info["reported"]["name"]

    def _ask(
        self, color: str, board: Board, moves: list[dict], global_deadline: float
    ) -> tuple[tuple[int, int], int]:
        """要一步棋。

        等待时长取"单步时限"与"全局墙钟剩余"的较小值 —— 否则单步时限长的引擎会把
        已经耗尽的全局墙钟又拖长一整个单步时限。若先到期的是全局墙钟，则判为
        无胜负中止（谁都没违规），而不是让当前落子方背锅。
        """
        proc = self.procs[color]
        payload = [[m["color"], m["x"], m["y"]] for m in moves]
        remaining_ms = int((global_deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            raise EngineError("GAME_TIMEOUT", self._global_timeout_message())
        limit_ms = min(self.config.move_timeout_ms, remaining_ms)

        sent = time.monotonic()
        try:
            proc.send({"type": "turn", "moves": payload, "deadline_ms": limit_ms})
            frame = proc.recv(limit_ms)
        except EngineError as exc:
            if exc.code == "TIMEOUT" and limit_ms < self.config.move_timeout_ms:
                # 等待被全局墙钟截断，而不是该方思考超时
                raise EngineError("GAME_TIMEOUT", self._global_timeout_message()) from exc
            raise _blame(exc, color) from exc
        elapsed_ms = int((time.monotonic() - sent) * 1000)

        ftype = frame.get("type")
        if ftype == "resign":
            raise _blame(EngineError("RESIGN", f"{color} 主动认输"), color)
        if ftype != "move":
            raise _blame(EngineError("PROTOCOL", f"{color} 回合期望 move，收到 {ftype!r}"), color)

        x, y = frame.get("x"), frame.get("y")
        for value in (x, y):
            if not isinstance(value, int) or isinstance(value, bool):
                raise _blame(
                    EngineError("BAD_OUTPUT", f"{color} 返回非法坐标: x={x!r} y={y!r}"), color
                )
        if not board.in_bounds(x, y):
            raise _blame(
                EngineError(
                    "ILLEGAL_MOVE",
                    f"{color} 坐标越界: ({x},{y})，合法范围 0..{board.size - 1}",
                ),
                color,
            )
        if not board.is_empty(x, y):
            raise _blame(EngineError("ILLEGAL_MOVE", f"{color} 落在已占点: ({x},{y})"), color)
        return (x, y), elapsed_ms

    def _global_timeout_message(self) -> str:
        return (
            f"整局墙钟上限 {self.config.game_timeout_ms}ms 用完，"
            f"该上限由双方共享，不归责于任何一方"
        )

    def _handshake_timeout_ms(self) -> int:
        # 模型加载可能较慢，握手给 30s（或整局墙钟，取小）
        return max(1, min(30_000, self.config.game_timeout_ms))

    def _notify_gameover(self, result: dict) -> None:
        for proc in self.procs.values():
            if proc.proc is not None and proc.proc.poll() is None:
                try:
                    proc.send({"type": "gameover", **result})
                except EngineError:
                    pass


def _blame(exc: EngineError, color: str) -> EngineError:
    """把责任方贴到异常上，供上层归因判负。"""
    exc.color = color  # type: ignore[attr-defined]
    return exc
