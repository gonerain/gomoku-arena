"""人类玩家适配器：把"人"接成平台的一个引擎进程。

用法（平台会像启动其他引擎一样启动它）：

    GOMOKU_HUMAN_PID=you GOMOKU_PLATFORM_URL=http://127.0.0.1:8090 \
        python engines/py_human/engine.py

它遵守与其他引擎完全相同的 NDJSON 协议（stdin 收、stdout 发），只是"想棋"这一步
改为等待人类通过平台接口落子：

    轮到我时 → 轮询 GET  {平台}/api/human/{pid}/{game_id}/move
    人类在浏览器点棋盘（或 POST 同一路径）→ 拿到坐标 → 作为 move 帧发出

这样裁判、计时、棋谱、实时观战全部复用既有路径，人类与 AI 走的是同一条链路。
超时由平台判定（人类若超时会像引擎一样判负），所以可以放心设置每步时限。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # engines/

from engine_loop import Context, GameView, run  # noqa: E402

PLATFORM_URL = (os.environ.get("GOMOKU_PLATFORM_URL") or "http://127.0.0.1:8090").rstrip("/")
HUMAN_PID = os.environ.get("GOMOKU_HUMAN_PID") or "human"
POLL_INTERVAL_S = float(os.environ.get("GOMOKU_HUMAN_POLL_MS", "300") or 300) / 1000.0

stderr = sys.stderr


def _fail(reason: str) -> None:
    stderr.write(f"[human] 无法作为人类玩家运行：{reason}\n")
    stderr.flush()
    raise SystemExit(2)


def _fetch(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_move(game_id: str, ctx: Context, deadline_s: float) -> tuple[int, int]:
    """一直等到人类落子，或平台宣告对局结束。

    刻意**不**按 deadline 自己超时退出：裁判才是计时的权威。若适配器在时限到达时
    自己 exit，平台只会看到"进程异常退出"，于是把一次普通的思考超时记成 crash（判罚
    理由错误）；而不退出的话，平台会在超时后自己判 timeout 并终结这个进程 —— 语义才正确。

    deadline_s 只用于提示文案与轮询节奏。
    """
    url = f"{PLATFORM_URL}/api/human/{ctx.color}/{game_id}/move"
    state_url = f"{PLATFORM_URL}/api/state?game_id={game_id}"
    announced = False
    while True:
        try:
            payload = _fetch(url, timeout=3.0)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 409):
                # 平台已结束该对局（或对局不存在）：确认一次后干净退出，
                # 不把正常收尾报成故障。
                if _game_over(state_url):
                    stderr.write("[human] 平台已结束该对局，适配器退出\n")
                    stderr.flush()
                    raise SystemExit(0)
            time.sleep(POLL_INTERVAL_S)
            continue
        except Exception:
            time.sleep(POLL_INTERVAL_S)
            continue

        if payload.get("move"):
            x, y = payload["move"]
            stderr.write(f"[human] 收到落子 x={x} y={y}\n")
            stderr.flush()
            return int(x), int(y)

        if not announced:
            stderr.write(
                f"[human] 轮到我落子 —— 请在浏览器点棋盘，或执行：\n"
                f"        curl -X POST {url} -H \"Content-Type: application/json\" "
                f"-d '{{\"x\":7,\"y\":7}}'\n"
            )
            stderr.flush()
            announced = True
        time.sleep(POLL_INTERVAL_S)


def _game_over(state_url: str) -> bool:
    """对局是否已被平台判定结束。取不到状态时按"仍在进行"处理，避免误退出。"""
    try:
        state = _fetch(state_url, timeout=3.0)
    except Exception:
        return False
    if not state:
        return False
    return not state.get("running", False)


CURRENT = {"game_id": None, "deadline_ms": 0}


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    if not CURRENT["game_id"]:
        _fail("尚未收到 init 帧，无法确定对局 id")
    return wait_for_move(CURRENT["game_id"], ctx, CURRENT["deadline_ms"] / 1000.0)


def main() -> None:
    """自己实现主循环：需要在 init 里拿到 game_id，并在 turn 里拿到 deadline。"""
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    if init.get("type") != "init":
        _fail(f"期望 init 帧，收到 {init.get('type')!r}")
    CURRENT["game_id"] = init.get("game_id")
    ctx = Context(init)

    print(json.dumps({
        "type": "ready",
        "name": f"human:{HUMAN_PID}",
        "version": "v0",
        "meta": {"algo": "human", "baseline": True, "desc": "人类玩家（浏览器点棋盘落子）"},
    }, ensure_ascii=False), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        frame = json.loads(line)
        ftype = frame.get("type")
        if ftype == "gameover":
            stderr.write(f"[human] 对局结束：{frame.get('winner')} / {frame.get('reason')}\n")
            stderr.flush()
            return
        if ftype != "turn":
            continue
        CURRENT["deadline_ms"] = int(frame.get("deadline_ms") or 1000)
        view = GameView(ctx.board_size, [[c, x, y] for c, x, y in frame.get("moves", [])])
        x, y = choose(view, ctx)
        print(json.dumps({"type": "move", "x": x, "y": y}), flush=True)


if __name__ == "__main__":
    main()
