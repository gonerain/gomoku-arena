"""命令行跑单局（不开浏览器时用）。

    python -m gomoku_arena.cli --black random --white greedy
    python -m gomoku_arena.cli --black trained --black-pid dsh \
        --black-checkpoint checkpoints/dsh_v1.pt --white greedy --board
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .match import Match, MatchConfig, Participant, participant_from_spec
from .rules import BOARD_SIZE, PROTOCOL, replay
from .server import ENGINES, Store, launch_spec, participant_for

REASON_TEXT = {
    "five": "五连获胜",
    "draw": "棋盘已满，平局",
    "timeout": "单步超时判负",
    "game_timeout": "整局墙钟上限用完，无胜负中止（不归责任何一方）",
    "crash": "进程崩溃判负",
    "illegal_move": "非法落子判负",
    "bad_output": "输出非协议帧，判负",
    "protocol": "协议违规判负",
    "resign": "主动认输",
    "start_failed": "引擎启动失败",
    "aborted": "对局中止（无法可靠归责）",
}


def build_external_participant(
    command: str,
    pid: str | None = None,
    checkpoint: str | None = None,
    version: str | None = None,
    name: str | None = None,
    move_delay_ms: int | None = None,
) -> Participant:
    """把一条外部引擎命令变成参赛方。

    两种写法都支持：
        "D:\\Code\\my_engine\\engine.py"                 → 用当前解释器执行该脚本
        "C:\\Python313\\python.exe D:\\...\\engine.py"    → 原样使用给定解释器
    外部引擎常常装在别的 Python 环境里，后者是必需的。
    """
    parts = command.split()
    first = Path(parts[0])
    if not first.exists():
        raise ValueError(
            f"既不是注册表引擎也不是有效路径: {command!r}；"
            f"可选脚本/可执行文件路径，或注册表引擎名"
        )

    # 展示名优先用调用方给的 name/pid；脚本名（常是 engine.py）只作最后兜底
    label = name or pid or (Path(parts[1]).stem if len(parts) > 1 else first.stem)
    if first.suffix.lower() == ".py":
        cmd = [sys.executable, str(first), *parts[1:]]
    else:
        cmd = [str(first), *parts[1:]]

    # checkpoint 必须写进环境变量，否则引擎收不到权重（外部适配器同样依赖 GOMOKU_MODEL）
    spec = {
        "engine": label,
        "name": label,
        "version": version or "v0",
        "cmd": cmd,
        "meta": {"desc": "外部引擎"},
        "checkpoint": checkpoint or "none",
        "env": {},
    }
    if checkpoint:
        spec["env"]["GOMOKU_MODEL"] = checkpoint
        spec["meta"]["checkpoint"] = checkpoint
    if move_delay_ms:
        spec["env"]["GOMOKU_TEST_MOVE_DELAY_MS"] = str(int(move_delay_ms))

    return participant_from_spec(launch_spec(spec), pid=pid)


def build_participant(
    key: str,
    pid: str | None = None,
    checkpoint: str | None = None,
    version: str | None = None,
) -> Participant:
    """注册表引擎走 participant_for；否则按外部命令处理。"""
    if key in ENGINES:
        return participant_for(key, pid=pid, version=version, checkpoint=checkpoint)
    return build_external_participant(key, pid=pid, checkpoint=checkpoint, version=version)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gomoku_arena.cli", description="五子棋对抗：命令行跑单局")
    parser.add_argument("--black", default="random", help="黑方引擎（名称或 .py 路径）")
    parser.add_argument("--white", default="greedy", help="白方引擎（名称或 .py 路径）")
    parser.add_argument("--black-pid", default=None, help="黑方参赛方 id（同一引擎不同权重时必须给）")
    parser.add_argument("--white-pid", default=None, help="白方参赛方 id")
    parser.add_argument("--black-checkpoint", default=None, help="黑方模型 checkpoint（仅 trained 引擎）")
    parser.add_argument("--white-checkpoint", default=None, help="白方模型 checkpoint（仅 trained 引擎）")
    parser.add_argument("--black-version", default=None, help="黑方模型版本")
    parser.add_argument("--white-version", default=None, help="白方模型版本")
    parser.add_argument("--move-timeout", type=int, default=1000, help="每步时限 ms")
    parser.add_argument("--game-timeout", type=int, default=120_000, help="整局墙钟上限 ms")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认按时间取）")
    parser.add_argument("--moves", action="store_true", help="打印逐手记录")
    parser.add_argument("--board", action="store_true", help="打印终局棋盘")
    parser.add_argument("--save", action="store_true", help="写入 records/games.jsonl")
    parser.add_argument("--protocol", action="store_true", help="打印引擎协议")
    args = parser.parse_args(argv)

    if args.protocol:
        print(PROTOCOL)
        return 0

    config = MatchConfig(
        seed=args.seed,
        board_size=BOARD_SIZE,
        move_timeout_ms=args.move_timeout,
        game_timeout_ms=args.game_timeout,
    )
    try:
        black = build_participant(
            args.black, args.black_pid, args.black_checkpoint, args.black_version
        )
        white = build_participant(
            args.white, args.white_pid, args.white_checkpoint, args.white_version
        )
        match = Match(black, white, config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    record = match.play()

    if args.moves or args.board:
        print(f"{'手':>4}  {'方':<5} {'坐标':<6} {'耗时':>7}")
        for m in record["moves"]:
            print(f"{m['n']:>4}  {m['color']:<5} "
                  f"{chr(ord('A') + m['x'])}{m['y'] + 1:<5} {m['ms']:>5}ms")

    if args.board:
        board = replay([[m["color"], m["x"], m["y"]] for m in record["moves"]], record["board_size"])
        print()
        print(board.as_text())
        print("  X=黑 O=白 .=空（自上而下为第 15..1 行）")

    res = record["result"]
    winner = res.get("winner")
    label = f"{winner} ({record[winner]['name']} {record[winner]['version']})" if winner else "无胜者"
    print()
    print(f"对局 {record['game_id']}  黑 {record['black']['name']} vs 白 {record['white']['name']}")
    print(f"结果: {label} · {REASON_TEXT.get(res.get('reason'), res.get('reason'))}")
    if record.get("error"):
        err = record["error"]
        print(f"异常: [{err.get('color')}] {err.get('reason')}: {err.get('message')}")
    print(f"共 {len(record['moves'])} 手，耗时 {record['duration_ms']}ms")

    if args.save:
        from pathlib import Path

        store = Store(Path("records/games.jsonl"))
        store.append(record)
        print(f"已写入 {store.path}（共 {len(store.records)} 局）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
