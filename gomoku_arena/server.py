"""HTTP 服务：裁判调度 + 批量对战 + 浏览器实时棋盘 + 比赛记录。

零外部依赖，只用标准库 http.server。前端用轮询读取对局状态，一步棋的思考时间
就在百毫秒级以上，轮询的开销和延迟可以忽略。

身份：参赛方(pid) → 引擎程序(engine) → 模型权重(checkpoint) 三层分离。
DSH 与 Codex 可以共用 trained 适配器，只要 pid 与 checkpoint 不同。

接口：
  GET  /                     实时棋盘页面
  GET  /api/board            棋盘尺寸与规则摘要
  GET  /api/engines          可用引擎程序（含 baseline / 默认 checkpoint 标识）
  GET  /api/models           可用模型文件
  GET  /api/games?limit=N    最近对局列表（不含着法）
  GET  /api/stats            完整统计（覆盖内存中的全部棋谱）
  GET  /api/games/<id>       单局完整棋谱（回放用）
  GET  /api/state            实时状态（进行中的对局，含批量归属）
  POST /api/matches          开始一局
  POST /api/batches          开始一批（逐局交换先手）
  GET  /api/batches/<id>     批量进度与分色统计
  DELETE /api/batches/<id>   中止批量对战
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .batch import Batch
from .match import DEFAULT_CHECKPOINT, Match, MatchConfig, Participant, participant_from_spec
from .rules import BOARD_SIZE

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
RECORDS_PATH = ROOT / "records" / "games.jsonl"
MODELS_DIR = ROOT / "checkpoints"

DEFAULT_MOVE_TIMEOUT_MS = 1000
DEFAULT_GAME_TIMEOUT_MS = 120_000
MAX_BATCH_GAMES = 500
# 内存中保留的棋谱条数上限：统计用它计算，超出的旧记录只留在 JSONL 里，
# 避免长时间连续对局把内存撑大（文件仍是完整记录）。
MAX_IN_MEMORY_RECORDS = 5000

RULE_SUMMARY = (
    f"{BOARD_SIZE}×{BOARD_SIZE} · 黑先 · ≥5 连（横/竖/斜）获胜 · 双方无禁手 · 无开局规则 · 满盘平局"
)


def _engine_spec(engine: str, name: str, version: str, meta: dict, env: dict | None = None) -> dict:
    return {
        "engine": engine,
        "name": name,
        "version": version,
        "cmd": [sys.executable, str(ROOT / "engines" / engine / "engine.py")],
        "meta": meta,
        "env": env or {},
        "checkpoint": (env or {}).get("GOMOKU_MODEL") or DEFAULT_CHECKPOINT,
    }


# 可用引擎程序。加一个 Python 适配器 = 加一条记录，不需要动其他地方。
ENGINES: dict[str, dict] = {
    "random": _engine_spec(
        "py_random", "random", "v0",
        {"algo": "uniform", "baseline": True, "desc": "随机落子：接口最小实现，不是 RL 模型"},
    ),
    "greedy": _engine_spec(
        "py_greedy", "greedy", "v0",
        {"algo": "heuristic-1ply", "baseline": True,
         "desc": "一步贪心基线，不是 RL 模型；用来判断训练是否真的有效"},
    ),
    "trained": _engine_spec(
        "py_trained", "trained", "v0",
        {"algo": "rl-checkpoint", "baseline": False,
         "desc": "RL 模型接入点；必须通过 checkpoint 指到真实模型文件"},
        {"GOMOKU_MODEL": os.environ.get("GOMOKU_MODEL", "")},
    ),
    # 仅用于验收：加载 tests/models/*/meta.json 这类测试权重，验证"同一适配器分别加载
    # 两份权重"的身份不混淆。它自报 meta.baseline=true，不冒充 RL 模型，也不是比赛用引擎。
    "ckpt_test": _engine_spec(
        "py_ckpt_test", "ckpt-test", "t0",
        {"algo": "test-weight-hash", "baseline": True,
         "desc": "测试用权重驱动策略（非 RL），仅用于验收双模型身份"},
        {"GOMOKU_MODEL": "", "GOMOKU_TEST_MOVE_DELAY_MS": "0"},
    ),
    # 仅用于验收：故意派生孙进程后崩溃，用来验证崩溃判负与进程树清理。禁止用于正式比赛。
    "test_crash": _engine_spec(
        "test_crash", "test-crash", "v0",
        {"algo": "crash", "baseline": True, "test_only": True,
         "desc": "故障注入引擎（必崩），仅用于验收清理逻辑"},
    ),
}


def launch_spec(spec: dict) -> dict:
    """把引擎程序规格变成可启动的子进程规格：强制无缓冲 + UTF-8，否则 stdout 会阻塞。"""
    out = dict(spec)
    env = dict(out.get("env") or {})
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    out["env"] = env
    return out


def participant_for(
    engine: str,
    pid: str | None = None,
    name: str | None = None,
    version: str | None = None,
    checkpoint: str | None = None,
    meta: dict | None = None,
    move_delay_ms: int | None = None,
) -> Participant:
    """按"引擎程序 + 参赛方身份 + 权重"组装一个参赛方。

    engine 可以是注册表里的名字（random/greedy/trained/...），
    也可以是一条外部引擎命令路径（首个 token 是 .py 脚本或可执行文件）。
    """
    if engine not in ENGINES:
        from .cli import build_external_participant

        # 展示名交给 build_external_participant 按 name/pid 推导，
        # 不要用命令里的解释器名（python.exe → "python"）
        return build_external_participant(
            engine, pid=pid, checkpoint=checkpoint, version=version,
            name=name, move_delay_ms=move_delay_ms,
        )
    base = ENGINES[engine]
    resolved_pid = pid or name or base["name"]
    merged_meta = {**(base.get("meta") or {}), **(meta or {})}
    resolved_version = version or base.get("version") or "v0"
    resolved_ckpt = checkpoint or base.get("checkpoint") or DEFAULT_CHECKPOINT

    spec = dict(base)
    env = dict(spec.get("env") or {})
    if resolved_ckpt and resolved_ckpt != DEFAULT_CHECKPOINT:
        env["GOMOKU_MODEL"] = resolved_ckpt
    elif engine == "trained":
        # 未指定 checkpoint 时不要继承环境里可能存在的值，避免"以为在用 A 模型"
        env["GOMOKU_MODEL"] = ""
    # 测试适配器支持人为步间延迟，便于验收观战采样（正式引擎不需要）
    if engine == "ckpt_test" and move_delay_ms:
        env["GOMOKU_TEST_MOVE_DELAY_MS"] = str(int(move_delay_ms))
    spec["env"] = env
    spec["name"] = name or resolved_pid
    spec["version"] = resolved_version
    spec["checkpoint"] = resolved_ckpt
    if resolved_ckpt != DEFAULT_CHECKPOINT:
        merged_meta["checkpoint"] = resolved_ckpt

    return Participant(
        pid=resolved_pid,
        name=name or resolved_pid,
        engine=engine,
        version=resolved_version,
        checkpoint=resolved_ckpt,
        meta=merged_meta,
        spec=launch_spec(spec),
    )


def resolve_participants(body: dict) -> tuple[Participant, Participant]:
    """从请求体解析双方参赛方。

    新格式（推荐，允许同引擎不同权重）：
        {"participants": {"black": {"engine":"trained","pid":"dsh","checkpoint":"..","version":"v3"},
                          "white": {"engine":"trained","pid":"codex","checkpoint":"..","version":"v7"}}}
    旧格式（仍支持）：
        {"black": "greedy", "white": "random", "checkpoints": {...}}
    """
    raw = body.get("participants")
    if isinstance(raw, dict) and any(k in raw for k in ("black", "white")):
        return (_parse_side(raw.get("black"), "black", "random"),
                _parse_side(raw.get("white"), "white", "greedy"))

    checkpoints = body.get("checkpoints") if isinstance(body.get("checkpoints"), dict) else {}
    black = body.get("black", "random")
    white = body.get("white", "greedy")
    return (
        participant_for(black, checkpoint=checkpoints.get(black)),
        participant_for(white, checkpoint=checkpoints.get(white)),
    )


def _parse_side(raw, color: str, default_engine: str) -> Participant:
    if raw is None:
        return participant_for(default_engine)
    if isinstance(raw, str):
        return participant_for(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"{color} 参赛方格式错误，应为对象或字符串")
    engine = raw.get("engine") or raw.get("engine_key") or raw.get("program")
    if not engine:
        raise ValueError(f"{color} 参赛方缺少 engine 字段（引擎程序标识）")
    return participant_for(
        engine,
        pid=raw.get("pid") or raw.get("id"),
        name=raw.get("name"),
        version=raw.get("version"),
        checkpoint=raw.get("checkpoint"),
        meta=raw.get("meta") if isinstance(raw.get("meta"), dict) else None,
        move_delay_ms=_optional_int(raw.get("move_delay_ms")),
    )


def _optional_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Store:
    """比赛记录：追加式 JSONL，每局一行。启动时读回，用于列表、回放与统计。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: deque[dict] = deque(maxlen=MAX_IN_MEMORY_RECORDS)
        self.total_written = 0
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                    self.total_written += 1
                except json.JSONDecodeError:
                    print(f"[store] 跳过损坏记录 {self.path}:{lineno}", file=sys.stderr)

    def append(self, record: dict) -> None:
        with self._lock:
            self.records.append(record)
            self.total_written += 1
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_summaries(self, limit: int) -> list[dict]:
        """最近对局的轻量列表：不含着法（回放时再单独取完整棋谱）。"""
        with self._lock:
            picked = list(self.records)[-limit:]
        return [_summary_of(rec) for rec in reversed(picked)]  # 最新在前

    def get(self, game_id: str) -> dict | None:
        with self._lock:
            for rec in reversed(self.records):
                if rec.get("game_id") == game_id:
                    return rec
        # 内存里被淘汰的旧对局：回到文件里找
        return self._scan_file(game_id)

    def _scan_file(self, game_id: str) -> dict | None:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or game_id not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("game_id") == game_id:
                    return rec
        return None

    def stats(self) -> dict:
        """完整统计：覆盖内存中保留的全部棋谱（内存上限见 MAX_IN_MEMORY_RECORDS）。"""
        with self._lock:
            snapshot = list(self.records)
        out = aggregate_stats(snapshot)
        out["records_on_disk"] = self.total_written
        out["records_in_memory"] = len(snapshot)
        return out


def aggregate_stats(records: list[dict]) -> dict:
    """按参赛方 pid 聚合，绝不按可能重复的引擎名称聚合。"""
    per_participant: dict[str, dict] = {}
    black_wins = white_wins = draws = 0
    abnormal = 0
    for rec in records:
        result = rec.get("result", {})
        reason = result.get("reason")
        winner_color = result.get("winner")
        if winner_color == "black":
            black_wins += 1
        elif winner_color == "white":
            white_wins += 1
        elif reason == "draw":
            draws += 1
        if reason not in ("five", "draw"):
            abnormal += 1
        for color in ("black", "white"):
            info = rec[color]
            pid = info.get("pid") or info["name"]
            entry = per_participant.setdefault(
                pid,
                {
                    "pid": pid,
                    "name": info.get("name", pid),
                    "engine": info.get("engine", "?"),
                    "version": info.get("version", "v0"),
                    "checkpoint": info.get("checkpoint", DEFAULT_CHECKPOINT),
                    "games": 0, "win": 0, "loss": 0, "draw": 0,
                    "timeout": 0, "illegal_move": 0, "crash": 0, "bad_output": 0,
                    "game_timeout": 0, "aborted": 0, "platform_error": 0,
                },
            )
            entry["games"] += 1
            if winner_color == color:
                entry["win"] += 1
            elif winner_color is not None:
                entry["loss"] += 1
                if reason in entry:
                    entry[reason] += 1
            elif reason in entry:
                entry[reason] += 1
    for entry in per_participant.values():
        games = entry["games"]
        entry["win_rate"] = round(entry["win"] / games, 4) if games else None
    return {
        "total_games": len(records),
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "abnormal": abnormal,
        "participants": sorted(per_participant.values(), key=lambda e: (-e["games"], e["pid"])),
    }


def _summary_of(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k != "moves"}
    out["move_count"] = len(rec.get("moves", []))
    out["status"] = "finished"  # 能进棋谱说明已结束
    out["running"] = False
    return out


class Runner:
    """在后台线程里跑单局或批量对战；把进行中的状态实时供给前端。"""

    def __init__(self, store: Store):
        self.store = store
        self.games: dict[str, dict] = {}
        self.batches: dict[str, dict] = {}
        self._batch_objects: dict[str, Batch] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # ---------- 对局的注册与执行（单局与批量共用同一条路径） ----------

    def _register(self, black: Participant, white: Participant, config: MatchConfig,
                  batch_info: dict | None = None) -> tuple[Match, dict]:
        """创建一个 Match 并立刻登记进实时状态，使浏览器能逐手观战。"""
        snapshot = {
            "game_id": None,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "board_size": config.board_size,
            "status": "running",
            "seed": config.seed,
            "move_timeout_ms": config.move_timeout_ms,
            "game_timeout_ms": config.game_timeout_ms,
            "black": black.for_color("black"),
            "white": white.for_color("white"),
            "participants": [black.to_dict() | {"color": "black"},
                             white.to_dict() | {"color": "white"}],
            "moves": [],
            "move_count": 0,
            "turn": "black",
            "result": None,
            "error": None,
            "duration_ms": 0,
        }
        if batch_info:
            snapshot["batch"] = batch_info
        match = Match(
            black, white, config,
            on_progress=lambda moves, turn: self._progress(snapshot, moves, turn),
            batch=batch_info,
        )
        snapshot["game_id"] = match.game_id
        with self._lock:
            self.games[match.game_id] = snapshot
        return match, snapshot

    def _progress(self, snapshot: dict, moves: list[dict], turn: str | None) -> None:
        # 直接引用活跃着法列表，不做拷贝：
        # 拷贝会让快照停在拷贝那一刻，于是整局过程中浏览器始终看到 0 手，
        # 直到结束时才一次性补上全部着法（实时观战就失效了）。
        # 快照与对局共享同一列表是安全的：结束时 _finish 会换成棋谱里的新列表。
        with self._lock:
            snapshot["moves"] = moves
            snapshot["move_count"] = len(moves)
            if turn:
                snapshot["turn"] = turn

    def _finish(self, match: Match, snapshot: dict) -> dict:
        try:
            record = match.play()
        except Exception as exc:  # 兜底：任何未预期异常都不能让这一局凭空消失
            import traceback

            traceback.print_exc()
            record = _failed_record(match, snapshot, exc)
        with self._lock:
            snapshot.update({k: v for k, v in record.items() if k != "moves"})
            # 换成棋谱里的新列表，切断与对局内部活跃列表的共享
            snapshot["moves"] = record.get("moves", [])
            snapshot["move_count"] = len(record.get("moves", []))
            snapshot["turn"] = None
            snapshot["status"] = "finished"
            snapshot["error"] = record.get("error")
        if not match.batch:
            self.store.append(record)  # 批量对局由 Batch.on_record 落盘，避免重复写入
        return record

    # ---------- 单局 ----------

    def start_match(self, black: Participant, white: Participant, config: MatchConfig) -> dict:
        match, snapshot = self._register(black, white, config)
        thread = threading.Thread(target=self._finish, args=(match, snapshot), daemon=True)
        thread.start()
        self._threads.append(thread)
        return _effective(snapshot)

    # ---------- 批量 ----------

    def start_batch(self, black: Participant, white: Participant, games: int,
                    config: MatchConfig) -> dict:
        batch = Batch(
            black, white, games, config,
            match_factory=lambda b, w, cfg, info: self._register(b, w, cfg, info)[0],
            on_update=self._store_batch_summary,
            on_record=self.store.append,
        )
        with self._lock:
            self._batch_objects[batch.batch_id] = batch
            self.batches[batch.batch_id] = batch.summary()
        thread = threading.Thread(target=self._run_batch, args=(batch,), daemon=True)
        thread.start()
        self._threads.append(thread)
        return self.batches[batch.batch_id]

    def _store_batch_summary(self, summary: dict) -> None:
        with self._lock:
            self.batches[summary["batch_id"]] = summary

    def _run_batch(self, batch: Batch) -> None:
        try:
            summary = batch.run()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            summary = batch.summary()
            summary["status"] = "error"
            summary["error"] = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self.batches[batch.batch_id] = summary

    def cancel_batch(self, batch_id: str) -> bool:
        with self._lock:
            batch = self._batch_objects.get(batch_id)
        if batch is None:
            return False
        batch.cancel()
        return True

    def get_batch(self, batch_id: str) -> dict | None:
        with self._lock:
            batch = self._batch_objects.get(batch_id)
            cached = self.batches.get(batch_id)
            if batch is not None and cached and cached.get("status") == "running":
                fresh = batch.summary()
                fresh["status"] = "running"
                self.batches[batch_id] = fresh
                return fresh
            return cached

    def list_batches(self) -> list[dict]:
        with self._lock:
            return [
                {k: v for k, v in item.items() if k != "games"}
                for item in self.batches.values()
            ]

    # ---------- 查询 ----------

    def current(self) -> dict | None:
        """给前端：优先返回进行中的对局，否则返回最后一局。"""
        with self._lock:
            running = [g for g in self.games.values() if g["status"] == "running"]
            if running:
                return _effective(running[-1])
        recents = self.store.list_summaries(1)
        return _effective(recents[0]) if recents else None

    def get(self, game_id: str) -> dict | None:
        with self._lock:
            live = self.games.get(game_id)
            if live and live["status"] == "running":
                return _effective(live)  # 进行中：用实时快照
        record = self.store.get(game_id)
        return _effective(record) if record else None


def _failed_record(match: Match, snapshot: dict, exc: Exception) -> dict:
    return {
        "v": 2,
        "game_id": match.game_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "board_size": snapshot["board_size"],
        "seed": snapshot["seed"],
        "participants": snapshot.get("participants", []),
        "black": snapshot["black"],
        "white": snapshot["white"],
        "moves": snapshot["moves"],
        "result": {"winner": None, "reason": "platform_error", "line": None},
        "error": {"color": None, "reason": type(exc).__name__, "message": str(exc)},
        "duration_ms": 0,
        "batch": snapshot.get("batch"),
    }


def _effective(snapshot: dict | None) -> dict | None:
    """统一对外状态的形态：补上 status 与 running。"""
    if snapshot is None:
        return None
    out = dict(snapshot)
    out.setdefault("status", "finished")
    out["running"] = out["status"] == "running"
    out.setdefault("move_count", len(out.get("moves", [])))
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "GomokuArena"
    protocol_version = "HTTP/1.1"

    @property
    def store(self) -> Store:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def runner(self) -> Runner:
        return self.server.runner  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/", "/index.html"):
                return self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            if route == "/api/board":
                return self._send_json(
                    {"board_size": BOARD_SIZE, "rule": RULE_SUMMARY,
                     "rule_id": "gomoku-freestyle-15"}
                )
            if route == "/api/engines":
                return self._send_json(
                    [
                        {
                            "key": key,
                            "name": spec["name"],
                            "version": spec["version"],
                            "meta": spec["meta"],
                            "baseline": bool((spec.get("meta") or {}).get("baseline")),
                            "checkpoint": str(spec.get("checkpoint") or DEFAULT_CHECKPOINT),
                        }
                        for key, spec in ENGINES.items()
                    ]
                )
            if route == "/api/models":
                return self._send_json(_list_models())
            if route == "/api/games":
                params = parse_qs(parsed.query)
                limit = _clamp(_int(params.get("limit", ["30"])[0], 30), 1, 500)
                return self._send_json(self.store.list_summaries(limit))
            if route == "/api/stats":
                return self._send_json(self.store.stats())
            if route == "/api/batches":
                return self._send_json(self.runner.list_batches())
            if route.startswith("/api/batches/"):
                batch_id = route[len("/api/batches/"):]
                summary = self.runner.get_batch(batch_id)
                if summary is None:
                    return self._send_json({"error": f"未找到批量对战 {batch_id}"}, 404)
                return self._send_json(summary)
            if route.startswith("/api/games/"):
                game_id = route[len("/api/games/"):]
                record = self.runner.get(game_id) or self.store.get(game_id)
                if record is None:
                    return self._send_json({"error": f"未找到对局 {game_id}"}, 404)
                return self._send_json(record)
            if route == "/api/current":
                return self._send_json(self.runner.current())
            if route == "/api/state":
                game_id = parse_qs(parsed.query).get("game_id", [None])[0]
                snapshot = self.runner.get(game_id) if game_id else self.runner.current()
                return self._send_json(snapshot)
            if route == "/favicon.ico":
                return self._send_bytes(b"", "image/x-icon")
            return self._send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route not in ("/api/matches", "/api/batches"):
            return self._send_json({"error": "not found"}, 404)
        try:
            body = self._read_json()
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)

        try:
            black, white = resolve_participants(body)
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)

        board_size = _int(body.get("board_size"), BOARD_SIZE)
        if board_size != BOARD_SIZE:
            return self._send_json(
                {"error": f"棋盘尺寸固定为 {BOARD_SIZE}x{BOARD_SIZE}（见 README 规则），"
                          f"不接受 board_size={board_size}"}, 400
            )
        move_timeout_ms = _clamp(_int(body.get("move_timeout_ms"), DEFAULT_MOVE_TIMEOUT_MS), 50, 600_000)
        game_timeout_ms = _clamp(
            _int(body.get("game_timeout_ms"), DEFAULT_GAME_TIMEOUT_MS), 1_000, 3_600_000
        )
        seed = _optional_seed(body.get("seed"))
        try:
            config = MatchConfig(
                seed=seed, board_size=board_size,
                move_timeout_ms=move_timeout_ms, game_timeout_ms=game_timeout_ms,
            )
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)

        if route == "/api/matches":
            try:
                snapshot = self.runner.start_match(black, white, config)
            except ValueError as exc:
                return self._send_json({"error": str(exc)}, 400)
            print(
                f"[match] {snapshot['game_id']} "
                f"{black.pid}({black.engine}/{black.checkpoint}) vs "
                f"{white.pid}({white.engine}/{white.checkpoint})",
                file=sys.stderr,
            )
            return self._send_json(snapshot, 202)

        games = _clamp(_int(body.get("games"), 10), 1, MAX_BATCH_GAMES)
        try:
            summary = self.runner.start_batch(black, white, games, config)
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)
        print(
            f"[batch] {summary['batch_id']} {black.pid}({black.checkpoint}) vs "
            f"{white.pid}({white.checkpoint}) 共 {games} 局（交换先手）",
            file=sys.stderr,
        )
        return self._send_json(summary, 202)

    def do_DELETE(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route.startswith("/api/batches/"):
            batch_id = route[len("/api/batches/"):]
            if not self.runner.cancel_batch(batch_id):
                return self._send_json({"error": f"未找到批量对战 {batch_id}"}, 404)
            return self._send_json({"batch_id": batch_id, "status": "cancelling"})
        return self._send_json({"error": "not found"}, 404)

    # ---------- 工具 ----------

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 1_000_000:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_bytes(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            return self._send_json({"error": f"缺少文件 {path}"}, 404)
        self._send_bytes(path.read_bytes(), ctype)

    def log_message(self, fmt: str, *args) -> None:  # 静音逐请求日志
        return


def _list_models() -> list[dict]:
    if not MODELS_DIR.is_dir():
        return []
    out = []
    for path in sorted(MODELS_DIR.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            out.append({"path": str(path), "name": path.name, "size": path.stat().st_size})
    return out


def _optional_seed(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gomoku_arena.server", description="五子棋 RL 对抗平台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--records", type=Path, default=RECORDS_PATH)
    parser.add_argument("--autostart", metavar="BLACK:WHITE", default=None,
                        help="启动后立刻开一局，例如 --autostart greedy:random")
    args = parser.parse_args(argv)

    store = Store(args.records)
    runner = Runner(store)

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        # 端口被占用时静默失败会非常难查（旧实例仍在应答，看起来像"改动没生效"）
        print(
            f"无法监听 {args.host}:{args.port} —— {exc}\n"
            f"该端口通常已被另一个平台实例占用。处理方法：\n"
            f"  1) 结束占用进程: netstat -ano | findstr :{args.port}  →  taskkill /F /PID <pid>\n"
            f"  2) 或换端口启动: --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1
    httpd.daemon_threads = True
    httpd.store = store  # type: ignore[attr-defined]
    httpd.runner = runner  # type: ignore[attr-defined]

    url = f"http://{args.host}:{args.port}"
    print(f"五子棋对抗平台已启动: {url}")
    print(f"规则: {RULE_SUMMARY}")
    print(f"棋谱记录: {args.records}（已有 {store.total_written} 局）")
    print("Ctrl+C 停止")

    if args.autostart:
        parts = args.autostart.split(":")
        try:
            black = participant_for(parts[0])
            white = participant_for(parts[1] if len(parts) > 1 and parts[1] else "greedy")
            runner.start_match(
                black, white,
                MatchConfig(board_size=BOARD_SIZE, move_timeout_ms=DEFAULT_MOVE_TIMEOUT_MS,
                            game_timeout_ms=DEFAULT_GAME_TIMEOUT_MS),
            )
        except ValueError as exc:
            print(f"无法自动开局: {exc}", file=sys.stderr)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
