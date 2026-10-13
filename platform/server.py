"""HTTP 服务：裁判调度 + 浏览器实时棋盘 + 比赛记录。

零外部依赖，只用标准库 http.server。前端用 400ms 轮询读取对局状态，
一步棋的思考时间就在百毫秒级以上，轮询的开销和延迟可以忽略。

接口：
  GET  /                     实时棋盘页面
  GET  /api/engines          可用引擎
  GET  /api/games?limit=N    对局列表（含末尾若干手，够画棋盘）
  GET  /api/games/<id>       单局完整棋谱
  POST /api/matches          开始一局
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):  # 支持 python platform/server.py 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from platform.match import Match, MatchConfig
    from platform.rules import BOARD_SIZE
else:
    from .match import Match, MatchConfig
    from .rules import BOARD_SIZE

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
RECORDS_PATH = ROOT / "records" / "games.jsonl"
EMPTY_SPEC = {"name": "empty", "version": "v0", "cmd": [], "meta": {"algo": "none"}}

# 可用引擎。加一个 Python 引擎 = 加一条记录，不需要动其他地方。
ENGINES: dict[str, dict] = {
    "random": {
        "name": "random",
        "version": "v0",
        "cmd": [sys.executable, str(ROOT / "engines" / "py_random" / "engine.py")],
        "meta": {"algo": "uniform", "desc": "随机落子，接口最小实现"},
    },
    "greedy": {
        "name": "greedy",
        "version": "v0",
        "cmd": [sys.executable, str(ROOT / "engines" / "py_greedy" / "engine.py")],
        "meta": {"algo": "heuristic-1ply", "desc": "一步贪心：先赢、再堵、否则取邻居"},
    },
    "trained": {
        "name": "trained",
        "version": "v0",
        "cmd": [sys.executable, str(ROOT / "engines" / "py_trained" / "engine.py")],
        "meta": {"algo": "rl-checkpoint", "desc": "接入你自己的 RL 模型（见该文件 TODO）"},
    },
}

MAX_LIST_MOVES = 400


def launch_spec(spec: dict) -> dict:
    """把引擎变成可启动的子进程规格：强制无缓冲 + UTF-8，否则 stdout 会阻塞。"""
    out = dict(spec)
    env = dict(out.get("env") or {})
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    out["env"] = env
    return out


class Store:
    """比赛记录：追加式 JSONL，每局一行。启动时读回，用于列表与回放。"""

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
        with self._lock:
            picked = self.records[-limit:]
        out = []
        for rec in reversed(picked):  # 最新在前
            item = {k: v for k, v in rec.items() if k != "moves"}
            item["move_count"] = len(rec.get("moves", []))
            item["moves"] = rec.get("moves", [])[-MAX_LIST_MOVES:]
            item["status"] = "finished"  # 能进棋谱说明已结束
            item["running"] = False
            out.append(item)
        return out

    def get(self, game_id: str) -> dict | None:
        with self._lock:
            for rec in reversed(self.records):
                if rec.get("game_id") == game_id:
                    return rec
        return None


class Runner:
    """在后台线程里跑一局；把进行中的状态实时供给前端。"""

    def __init__(self, store: Store):
        self.store = store
        self.games: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self, black: str, white: str, board_size: int, seed: int | None,
              move_timeout_ms: int, game_timeout_ms: int) -> dict:
        snapshot = {
            "game_id": None,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "board_size": board_size,
            "status": "running",
            "seed": seed,
            "move_timeout_ms": move_timeout_ms,
            "black": _player_stub(ENGINES[black], black, "black"),
            "white": _player_stub(ENGINES[white], white, "white"),
            "moves": [],
            "move_count": 0,
            "turn": "black",
            "result": None,
            "error": None,
            "duration_ms": 0,
        }
        config = MatchConfig(
            seed=seed,
            board_size=board_size,
            move_timeout_ms=move_timeout_ms,
            game_timeout_ms=game_timeout_ms,
        )
        match = Match(
            launch_spec(ENGINES[black]),
            launch_spec(ENGINES[white]),
            config,
            on_progress=lambda moves, turn: self._progress(snapshot, moves, turn),
        )
        snapshot["game_id"] = match.game_id
        with self._lock:
            self.games[match.game_id] = snapshot

        thread = threading.Thread(target=self._run, args=(match, snapshot), daemon=True)
        thread.start()
        self._threads.append(thread)
        return _effective(snapshot)

    def _progress(self, snapshot: dict, moves: list[dict], turn: str | None) -> None:
        with self._lock:
            snapshot["moves"] = list(moves)
            snapshot["move_count"] = len(moves)
            if turn:
                snapshot["turn"] = turn

    def _run(self, match: Match, snapshot: dict) -> None:
        try:
            record = match.play()
        except Exception as exc:  # 兜底：任何未预期异常都不能让这一局凭空消失
            import traceback

            traceback.print_exc()
            record = {
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
        with self._lock:
            snapshot.update({k: v for k, v in record.items() if k != "moves"})
            snapshot["moves"] = record.get("moves", [])
            snapshot["move_count"] = len(record.get("moves", []))
            snapshot["turn"] = None
            snapshot["status"] = "finished"
            snapshot["error"] = record.get("error")
        self.store.append(record)

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


def _player_stub(spec: dict, key: str, color: str) -> dict:
    return {
        "name": spec.get("name", key),
        "version": spec.get("version", "v0"),
        "color": color,
        "meta": spec.get("meta", {}),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "GomokuArena"
    protocol_version = "HTTP/1.1"

    # 依赖由 main() 注入到 server 上
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
                        {"key": key, "name": spec["name"], "version": spec["version"], "meta": spec["meta"]}
                        for key, spec in ENGINES.items()
                    ]
                )
            if route == "/api/games":
                params = parse_qs(parsed.query)
                limit = _clamp(_int(params.get("limit", ["20"])[0], 20), 1, 200)
                return self._send_json(self.store.list_summaries(limit))
            if route.startswith("/api/games/"):
                game_id = route[len("/api/games/"):]
                record = self.store.get(game_id)
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
        if urlparse(self.path).path != "/api/matches":
            return self._send_json({"error": "not found"}, 404)
        try:
            body = self._read_json()
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)

        black = body.get("black", "random")
        white = body.get("white", "greedy")
        for key, value in (("black", black), ("white", white)):
            if value not in ENGINES:
                return self._send_json({"error": f"{key} 引擎不存在: {value!r}，可选 {sorted(ENGINES)}"}, 400)

        board_size = _clamp(_int(body.get("board_size"), BOARD_SIZE), 5, 25)
        move_timeout_ms = _clamp(_int(body.get("move_timeout_ms"), 1000), 50, 600_000)
        game_timeout_ms = _clamp(_int(body.get("game_timeout_ms"), 120_000), 1_000, 3_600_000)
        seed = body.get("seed")
        seed = int(seed) if isinstance(seed, (int, str)) and str(seed).lstrip("-").isdigit() else None

        snapshot = self.runner.start(
            black, white, board_size, seed, move_timeout_ms, game_timeout_ms
        )
        print(
            f"[match] {snapshot['game_id']} {black}(黑) vs {white}(白) "
            f"每步{move_timeout_ms}ms 整局{game_timeout_ms}ms",
            file=sys.stderr,
        )
        return self._send_json(snapshot, 202)

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


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="platform.server", description="五子棋 RL 对抗平台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--records", type=Path, default=RECORDS_PATH)
    parser.add_argument("--autostart", metavar="BLACK:WHITE", default=None,
                        help="启动后立刻开一局，例如 --autostart random:greedy")
    args = parser.parse_args(argv)

    mimetypes.add_type("text/javascript", ".js")
    store = Store(args.records)
    runner = Runner(store)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.store = store  # type: ignore[attr-defined]
    httpd.runner = runner  # type: ignore[attr-defined]

    url = f"http://{args.host}:{args.port}"
    print(f"五子棋对抗平台已启动: {url}")
    print(f"棋谱记录: {args.records}（已有 {len(store.records)} 局）")
    print("Ctrl+C 停止")

    if args.autostart:
        parts = args.autostart.split(":")
        black = parts[0]
        white = parts[1] if len(parts) > 1 and parts[1] else "greedy"
        runner.start(black, white, BOARD_SIZE, None, 1000, 120_000)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
