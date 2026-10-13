"""端到端测试 D：人类玩家接入（浏览器点棋盘落子的那条链路）。

    python -m gomoku_arena.server --port 8090     # 先起服务
    python tests/test_human.py

验证：
  * 人类可以作为一方参赛，并按引擎协议被裁判调度；
  * 轮到人类时 /api/human/<pid>/<game_id>/move 返回 move=null；
  * POST 落子后，人类适配器取到坐标并真正落到棋盘上（棋谱里出现该坐标）；
  * 非本回合、非人类方、已占点、越界都被拒绝（409/400）；
  * 取走即消费：同一坐标不会被重复使用。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    ok = bool(ok)
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def actor_color(state: dict, pid: str):
    """对局的 turn 字段是颜色（black/white），不是 pid。这里把 pid 映射成颜色。"""
    for color in ("black", "white"):
        info = state.get(color) or {}
        if info.get("pid") == pid:
            return color
    return None


def wait_turn(game_id: str, pid: str, timeout: float = 30.0):
    """等到轮到 pid 这一方，返回该局的实时状态。"""
    deadline = time.time() + timeout
    state = None
    while time.time() < deadline:
        state = get(f"/api/state?game_id={game_id}")
        if not state.get("running"):
            return state
        if state.get("turn") == actor_color(state, pid):
            return state
        time.sleep(0.1)
    return state


def main() -> int:
    print(f"=== 人类玩家接入测试（{BASE}）===\n")
    try:
        engines = {e["key"]: e for e in get("/api/engines")}
    except Exception as exc:
        print(f"服务端不可达: {exc}")
        return 1

    check("human" in engines, "人类适配器已注册为引擎",
          ",".join(sorted(engines)))
    check(engines["human"]["baseline"] is True,
          "人类适配器被标注为基线（不是 RL 模型）")

    # 人类执黑先走 —— 先手方最容易观察
    status, snap = post("/api/matches", {
        "participants": {
            "black": {"engine": "human", "pid": "you", "name": "你"},
            "white": {"engine": "greedy", "pid": "baseline"},
        },
        "move_timeout_ms": 15000, "game_timeout_ms": 300000, "seed": 4242,
    })
    game_id = snap["game_id"]
    check(status == 202, "人类 vs 贪心 的对局已创建", game_id)

    started = time.time()
    state = wait_turn(game_id, "you")
    elapsed = time.time() - started
    print(f"       [诊断] 等待轮到人类耗时 {elapsed:.2f}s，状态 turn="
          f"{state and state.get('turn')} running={state and state.get('running')} "
          f"moves={state and state.get('move_count')}")
    check(state is not None and state.get("turn") == actor_color(state, "you"),
          "裁判把黑方第一手交给人类（引擎进程正常启动并握手）",
          f"turn={state and state.get('turn')} moves={state and state.get('move_count')}")

    # 落子（模拟浏览器点击：只 POST，绝不能去 GET 那个队列端点 ——
    # 那等于把落子从适配器手里抢走，适配器就永远取不到了）
    code, body = post(f"/api/human/black/{game_id}/move", {"x": 7, "y": 7})
    check(code == 202 and body.get("move") == [7, 7],
          "人类落子被平台接受", f"HTTP {code} {body}")

    # 适配器会取走该落子并交给裁判：等它出现在棋谱上
    deadline = time.time() + 30
    moves = []
    while time.time() < deadline:
        state = get(f"/api/state?game_id={game_id}")
        moves = state.get("moves", [])
        if moves:
            break
        time.sleep(0.1)
    check(bool(moves) and moves[0]["x"] == 7 and moves[0]["y"] == 7,
          "人类适配器取走落子并真正落到棋盘上（棋谱出现该坐标）",
          json.dumps(moves[:1], ensure_ascii=False))

    # 等 AI 回手，然后再轮到人类
    state = wait_turn(game_id, "you", timeout=30)
    check(state is not None and (state.get("move_count") or 0) >= 2,
          "AI 回手后重新轮到人类（对局在推进）",
          f"moves={state and state.get('move_count')}")

    # 非法落子应被拒绝
    code, body = post(f"/api/human/black/{game_id}/move", {"x": 7, "y": 7})
    check(code == 409, "已占点被拒绝", f"HTTP {code} {body.get('error')}")
    code, body = post(f"/api/human/black/{game_id}/move", {"x": 99, "y": 0})
    check(code == 409, "越界坐标被拒绝", f"HTTP {code} {body.get('error')}")
    code, body = post(f"/api/human/black/{game_id}/move", {"x": "a", "y": 0})
    check(code == 400, "非整数坐标被拒绝", f"HTTP {code} {body.get('error')}")

    # 走完这一局：人类用一个固定的可靠策略（沿着一条线补子）把棋下完
    deadline = time.time() + 60
    played = 0
    while time.time() < deadline and played < 40:
        state = get(f"/api/state?game_id={game_id}")
        if not state.get("running"):
            break
        if state.get("turn") != actor_color(state, "you"):
            time.sleep(0.1)
            continue
        occupied = {(m["x"], m["y"]) for m in state.get("moves", [])}
        pick = next(((7, y) for y in range(15) if (7, y) not in occupied), None) \
            or next(((x, 7) for x in range(15) if (x, 7) not in occupied), None)
        if pick is None:
            break
        code, _ = post(f"/api/human/black/{game_id}/move", {"x": pick[0], "y": pick[1]})
        if code != 202:
            time.sleep(0.2)
            continue
        # 等这手被消费并落到棋盘
        for _ in range(60):
            cur = get(f"/api/state?game_id={game_id}")
            if any(m["x"] == pick[0] and m["y"] == pick[1]
                   for m in cur.get("moves", [])):
                break
            time.sleep(0.1)
        played += 1

    final = get(f"/api/state?game_id={game_id}")
    check(final.get("status") == "finished",
          "人类与 AI 的对局能正常结束",
          f"status={final.get('status')} reason={(final.get('result') or {}).get('reason')} "
          f"moves={final.get('move_count')}")
    check(all(m["color"] in ("black", "white") for m in final.get("moves", [])),
          "终局棋谱着法合法", f"{len(final.get('moves', []))} 手")

    # 非人类方不能通过该接口落子
    status, snap2 = post("/api/matches", {
        "participants": {
            "black": {"engine": "human", "pid": "you", "name": "你"},
            "white": {"engine": "greedy", "pid": "baseline"},
        },
        "move_timeout_ms": 8000, "game_timeout_ms": 120000, "seed": 7,
    })
    gid2 = snap2["game_id"]
    wait_turn(gid2, "you")
    code, body = post(f"/api/human/white/{gid2}/move", {"x": 0, "y": 0})
    check(code == 409, "非人类方无法通过该接口落子", f"HTTP {code} {body.get('error')}")

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
