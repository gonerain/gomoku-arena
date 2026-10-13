"""HTTP 服务：裁判调度 + 批量对战 + 浏览器实时棋盘 + 比赛记录。

零外部依赖，只用标准库 http.server。前端用轮询读取对局状态，一步棋的思考时间
就在百毫秒级以上，轮询的开销和延迟可以忽略。

接口：
  GET  /                     实时棋盘页面
  GET  /api/engines          可用引擎（含 baseline / checkpoint 标识）
  GET  /api/models           可用模型文件
  GET  /api/games?limit=N    最近对局列表（不含着法，避免列表接口体积失控）
  GET  /api/stats            完整统计（全部棋谱，不是"最近 N 局"）
  GET  /api/games/<id>       单局完整棋谱（回放用）
  GET  /api/state            实时状态（进行中的对局）
  POST /api/matches          开始一局
  POST /api/batches          开始一批（逐局交换先手）
  GET  /api/batches/<id>     批量对战进度与分色统计
  DELETE /api/batches/<id>   中止批量对战
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .batch import Batch
from .match import Match, MatchConfig
from .rules import BOARD_SIZE

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
RECORDS_PATH = ROOT / "records" / "games.jsonl"
MODELS_DIR = ROOT / "checkpoints"

DEFAULT_MOVE_TIMEOUT_MS = 1000
DEFAULT_GAME_TIMEOUT_MS = 120_000
MAX_BATCH_GAMES = 500

# 规则单一定义，前端与日志都从这里取，避免三处各写一份而漂移
RULE_SUMMARY = (
    f"{BOARD_SIZE}×{BOARD_SIZE} · 黑先 · ≥5 连（横/竖/斜）获胜 · 双方无禁手 · 无开局规则 · 满盘平局"
)


def _engine_spec(script: str, name: str, version: str, meta: dict, env: dict | None = None) -> dict:
    return {
        "name": name,
        "version": version,
        "cmd": [sys.executable, str(ROOT / "engines" / script / "engine.py")],
        "meta": meta,
        "env": env or {},
        "checkpoint": (env or {}).get("GOMOKU_MODEL", "none"),
    }


# 可用引擎。加一个 Python 引擎 = 加一条记录，不需要动其他地方。
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
}


def launch_spec(spec: dict) -> dict:
    """把引擎变成可启动的子进程规格：强制无缓冲 + UTF-8，否则 stdout 会阻塞。"""
    out = dict(spec)
    env = dict(out.get("env") or {})
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    out["env"] = env
    return out


def with_checkpoint(spec: dict, checkpoint: str | None) -> dict:
    """把某个引擎实例绑定到指定 checkpoint（不修改全局注册表）。"""
    if not checkpoint:
        return spec
    out = dict(spec)
    env = dict(out.get("env") or {})
    env["GOMOKU_MODEL"] = checkpoint
    out["env"] = env
    out["checkpoint"] = checkpoint
    out["meta"] = {**(out.get("meta") or {}), "checkpoint": checkpoint}
    return out


class Store:
    """比赛记录：追加式 JSONL，每局一行。启动时读回，用于列表、回放与统计。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
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
                except json.JSONDecodeError:
                    print(f"[store] 跳过损坏记录 {self.path}:{lineno}", file=sys.stderr)

    def append(self, record: dict) -> None:
        with self._lock:
            self.records.append(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_summaries(self, limit: int) -> list[dict]:
        """最近对局的轻量列表：不含着法（回放时再单独取完整棋谱）。"""
        with self._lock:
            picked = self.records[-limit:]
        out = []
        for rec in reversed(picked):  # 最新在前
            out.append(_summary_of(rec))
        return out

    def get(self, game_id: str) -> dict | None:
        with self._lock:
            for rec in reversed(self.records):
                if rec.get("game_id") == game_id:
                    return rec
        return None

    def stats(self) -> dict:
        """完整统计：覆盖全部棋谱，不是"最近 N 局"。"""
        with self._lock:
            snapshot = list(self.records)
        return aggregate_stats(snapshot)


def aggregate_stats(records: list[dict]) -> dict:
    per_engine: dict[str, dict] = {}
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
            name = rec[color]["name"]
            entry = per_engine.setdefault(
                name,
                {
                    "name": name,
                    "version": rec[color].get("version", "v0"),
                    "checkpoint": rec[color].get("checkpoint", "none"),
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
            elif reason == "draw":
                entry["draw"] += 1
            elif reason in entry:
                entry[reason] += 1
    for entry in per_engine.values():
        games = entry["games"]
        entry["win_rate"] = round(entry["win"] / games, 4) if games else None
    return {
        "total_games": len(records),
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "abnormal": abnormal,
        "engines": sorted(per_engine.values(), key=lambda e: (-e["games"], e["name"])),
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

    # ---------- 单局 ----------

    def start_match(self, black: str, white: str, board_size: int, seed: int | None,
                    move_timeout_ms: int, game_timeout_ms: int,
                    checkpoints: dict | None = None) -> dict:
        checkpoints = checkpoints or {}

        def spec_factory(key: str) -> dict:
            return with_checkpoint(launch_spec(ENGINES[key]), checkpoints.get(key))

        config = MatchConfig(
            seed=seed,
            board_size=board_size,
            move_timeout_ms=move_timeout_ms,
            game_timeout_ms=game_timeout_ms,
        )
        black_spec = spec_factory(black)
        white_spec = spec_factory(white)
        snapshot = {
            "game_id": None,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "board_size": board_size,
            "status": "running",
            "seed": seed,
            "move_timeout_ms": move_timeout_ms,
            "game_timeout_ms": game_timeout_ms,
            "black": _player_stub(black_spec, "black"),
            "white": _player_stub(white_spec, "white"),
            "moves": [],
            "move_count": 0,
            "turn": "black",
            "result": None,
            "error": None,
            "duration_ms": 0,
        }
        match = Match(
            black_spec, white_spec, config,
            on_progress=lambda moves, turn: self._progress(snapshot, moves, turn),
        )
        snapshot["game_id"] = match.game_id
        with self._lock:
            self.games[match.game_id] = snapshot

        thread = threading.Thread(target=self._run_match, args=(match, snapshot), daemon=True)
        thread.start()
        self._threads.append(thread)
        return _effective(snapshot)

    def _progress(self, snapshot: dict, moves: list[dict], turn: str | None) -> None:
        with self._lock:
            snapshot["moves"] = list(moves)
            snapshot["move_count"] = len(moves)
            if turn:
                snapshot["turn"] = turn

    def _run_match(self, match: Match, snapshot: dict, batch_id: str | None = None) -> None:
        try:
            record = match.play()
        except Exception as exc:  # 兜底：任何未预期异常都不能让这一局凭空消失
            import traceback

            traceback.print_exc()
            record = _failed_record(match, snapshot, exc)
        with self._lock:
            snapshot.update({k: v for k, v in record.items() if k != "moves"})
            snapshot["moves"] = record.get("moves", [])
            snapshot["move_count"] = len(record.get("moves", []))
            snapshot["turn"] = None
            snapshot["status"] = "finished"
            snapshot["error"] = record.get("error")
        self.store.append(record)

    # ---------- 批量 ----------

    def start_batch(self, black: str, white: str, games: int, board_size: int,
                    seed: int | None, move_timeout_ms: int, game_timeout_ms: int,
                    checkpoints: dict | None = None) -> dict:
        checkpoints = checkpoints or {}

        def spec_factory(key: str) -> dict:
            return with_checkpoint(launch_spec(ENGINES[key]), checkpoints.get(key))

        config = MatchConfig(
            seed=seed,
            board_size=board_size,
            move_timeout_ms=move_timeout_ms,
            game_timeout_ms=game_timeout_ms,
        )
        batch = Batch(black, white, games, config, spec_factory, self.store.append)
        with self._lock:
            self._batch_objects[batch.batch_id] = batch
            self.batches[batch.batch_id] = batch.summary()
        thread = threading.Thread(target=self._run_batch, args=(batch,), daemon=True)
        thread.start()
        self._threads.append(thread)
        return self.batches[batch.batch_id]

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
            if batch is not None and batch_id in self.batches:
                cached = self.batches[batch_id]
                if cached.get("status") == "running":
                    self.batches[batch_id] = {**batch.summary(), "status": "running",
                                              "requested": batch.games}
            return self.batches.get(batch_id)

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
        return _effective(record) if record else None  # 已结束或重启后：用棋谱原文


def _failed_record(match: Match, snapshot: dict, exc: Exception) -> dict:
    return {
        "v": 1,
        "game_id": match.game_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "board_size": snapshot["board_size"],
        "seed": snapshot["seed"],
        "black": snapshot["black"],
        "white": snapshot["white"],
        "moves": snapshot["moves"],
        "result": {"winner": None, "reason": "platform_error", "line": None},
        "error": {"color": None, "reason": type(exc).__name__, "message": str(exc)},
        "duration_ms": 0,
    }


def _effective(snapshot: dict | None) -> dict | None:
    """统一对外状态的形态：补上 status 与 running。

    进行中的实时快照自带 status；棋谱原文（来自 JSONL）没有，需要按"已结束"补齐，
    这样前端和 API 使用者面对的状态字段永远是一致的。
    """
    if snapshot is None:
        return None
    out = dict(snapshot)
    out.setdefault("status", "finished")
    out["running"] = out["status"] == "running"
    out.setdefault("move_count", len(out.get("moves", [])))
    return out


def _player_stub(spec: dict, color: str) -> dict:
    return {
        "name": spec.get("name", "unknown"),
        "version": spec.get("version", "v0"),
        "color": color,
        "meta": spec.get("meta", {}),
        "checkpoint": str(spec.get("checkpoint") or "none"),
    }


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
            if route == "/api/engines":
                return self._send_json(
                    [
                        {
                            "key": key,
                            "name": spec["name"],
                            "version": spec["version"],
                            "meta": spec["meta"],
                            "baseline": bool((spec.get("meta") or {}).get("baseline")),
                            "checkpoint": str(spec.get("checkpoint") or "none"),
                        }
                        for key, spec in ENGINES.items()
                    ]
                )
            if route == "/api/board":
                return self._send_json(
                    {
                        "board_size": BOARD_SIZE,
                        "rule": RULE_SUMMARY,
                        "rule_id": "gomoku-freestyle-15",
                    }
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

        black = body.get("black", "random")
        white = body.get("white", "greedy")
        for key, value in (("black", black), ("white", white)):
            if value not in ENGINES:
                return self._send_json(
                    {"error": f"{key} 引擎不存在: {value!r}，可选 {sorted(ENGINES)}"}, 400
                )

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
        checkpoints = _parse_checkpoints(body.get("checkpoints"), (black, white))

        if route == "/api/matches":
            snapshot = self.runner.start_match(
                black, white, board_size, seed, move_timeout_ms, game_timeout_ms, checkpoints
            )
            print(
                f"[match] {snapshot['game_id']} {black}(黑) vs {white}(白) "
                f"每步{move_timeout_ms}ms 整局{game_timeout_ms}ms",
                file=sys.stderr,
            )
            return self._send_json(snapshot, 202)

        games = _clamp(_int(body.get("games"), 10), 1, MAX_BATCH_GAMES)
        summary = self.runner.start_batch(
            black, white, games, board_size, seed, move_timeout_ms, game_timeout_ms, checkpoints
        )
        print(
            f"[batch] {summary['batch_id']} {black} vs {white} 共 {games} 局（交换先手）",
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
        if path.is_file():
            out.append({"path": str(path), "name": path.name, "size": path.stat().st_size})
    return out


def _parse_checkpoints(raw, engine_keys) -> dict:
    if not isinstance(raw, dict):
        return {}
    return {key: str(raw[key]) for key in engine_keys if raw.get(key)}


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
                        help="启动后立刻开一局，例如 --autostart random:greedy")
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
    print(f"棋谱记录: {args.records}（已有 {len(store.records)} 局）")
    print("Ctrl+C 停止")

    if args.autostart:
        parts = args.autostart.split(":")
        black = parts[0]
        white = parts[1] if len(parts) > 1 and parts[1] else "greedy"
        if black in ENGINES and white in ENGINES:
            runner.start_match(
                black, white, BOARD_SIZE, None,
                DEFAULT_MOVE_TIMEOUT_MS, DEFAULT_GAME_TIMEOUT_MS,
            )
        else:
            print(f"未知引擎: {args.autostart}，可选 {sorted(ENGINES)}", file=sys.stderr)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
