"""命令行批量对战：连续 N 局，逐局交换先后手，输出分色统计。

    # 方式一：命令行直接指定（两边可以是同一个适配器、不同权重）
    python -m gomoku_arena.batch_cli --black-engine trained --black-pid dsh \
        --black-checkpoint checkpoints/dsh_ppo_v3.pt --black-version v3 \
        --white-engine trained --white-pid codex \
        --white-checkpoint checkpoints/codex_az_v7.pt --white-version v7 \
        --games 20 --save

    # 方式二：配置文件（推荐，便于复现与固化"某一届比赛"的双方）
    python -m gomoku_arena.batch_cli --config matchups/dsh_vs_codex.json --games 20 --save

    # 基线自检（非 RL）
    python -m gomoku_arena.batch_cli --black-engine random --white-engine greedy --games 10

DSH 与 Codex 各自训练完一个模型后，用这个命令跑若干局即可得到可信的对比数字：
交换先手消除了先手优势，统计以参赛方 pid 为键，不会把不同模型版本的成绩混在一起。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import Batch
from .match import Match, MatchConfig, Participant
from .rules import BOARD_SIZE
from .server import RECORDS_PATH, Store, load_participants, participant_for

CONFIG_TEMPLATE = {
    "_comment": "参赛方配置：engine=引擎程序(适配器或含解释器的完整命令)，pid=参赛方唯一标识，checkpoint=模型权重",
    "black": {"engine": "trained", "pid": "dsh", "name": "DSH",
              "version": "v3", "checkpoint": "checkpoints/dsh_ppo_v3.pt",
              "meta": {"algo": "ppo", "episodes": 12000, "budget": "2e6 steps"}},
    "white": {"engine": "trained", "pid": "codex", "name": "Codex",
              "version": "v7", "checkpoint": "checkpoints/codex_az_v7.pt",
              "meta": {"algo": "alphazero", "sims": 200}},
}


def participant_from_config(raw, fallback_engine: str) -> Participant:
    """兼容旧调用点：转发到 server.participant_from_config。"""
    from .server import participant_from_config as _impl

    return _impl(raw, fallback_engine)


def build_participants(args) -> tuple[Participant, Participant]:
    if args.config:
        try:
            return load_participants(args.config)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    black = participant_for(
        args.black_engine, pid=args.black_pid, name=args.black_name,
        version=args.black_version, checkpoint=args.black_checkpoint,
    )
    white = participant_for(
        args.white_engine, pid=args.white_pid, name=args.white_name,
        version=args.white_version, checkpoint=args.white_checkpoint,
    )
    return black, white


def _rate(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def _print_report(summary: dict, black: Participant, white: Participant) -> None:
    print()
    print(f"批量对战 {summary['batch_id']}")
    print(f"  黑方(首局执黑) : {black.name} [{black.pid}] 引擎={black.engine} "
          f"版本={black.version} 权重={black.checkpoint}")
    print(f"  白方(首局执白) : {white.name} [{white.pid}] 引擎={white.engine} "
          f"版本={white.version} 权重={white.checkpoint}")
    print(f"请求 {summary['requested']} 局，完成 {summary['played']} 局，"
          f"用时 {summary.get('duration_ms')}ms，状态 {summary['status']}")
    print()
    header = (f"{'参赛方':<14}{'引擎':<10}{'执子':<6}{'局':>4}{'胜':>5}{'负':>5}{'平':>5}"
              f"{'超时':>6}{'非法':>6}{'崩溃':>6}{'墙钟超时':>9}{'胜率':>8}")
    print(header)
    print("-" * 96)
    for pid in (black.pid, white.pid):
        data = summary["sides"].get(pid)
        if not data:
            continue
        for key, label in (("as_black", "黑"), ("as_white", "白")):
            side = data[key]
            print(f"{pid:<14}{data['engine']:<10}{label:<6}{side['games']:>4}{side['win']:>5}"
                  f"{side['loss']:>5}{side['draw']:>5}{side['timeout']:>6}"
                  f"{side['illegal_move']:>6}{side['crash']:>6}{side['game_timeout']:>9}"
                  f"{_rate(side['win'] / side['games'] if side['games'] else None):>8}")
    print("-" * 96)
    for pid in (black.pid, white.pid):
        total = summary["totals"].get(pid)
        if not total:
            continue
        print(f"{pid:<14}{total['engine']:<10}{'合计':<6}{total['games']:>4}{total['win']:>5}"
              f"{total['loss']:>5}{total['draw']:>5}{total['timeout']:>6}"
              f"{total['illegal_move']:>6}{total['crash']:>6}{total['game_timeout']:>9}"
              f"{_rate(total['win_rate']):>8}")
    print()
    for pid in (black.pid, white.pid):
        total = summary["totals"].get(pid)
        if not total:
            continue
        print(f"{pid} [{total['checkpoint']}]: 胜率 {_rate(total['win_rate'])}  "
              f"负率 {_rate(total['loss_rate'])}  平局率 {_rate(total['draw_rate'])}")
    print()
    print("逐局结果：")
    for game in summary["games"]:
        index = game.get("batch_index") or "-"
        winner = f"{game['winner']} [{game['winner_pid']}]" if game["winner"] else "无胜者"
        print(f"  #{index:<3} {game['moves']:>3}手  "
              f"{game['black']}[{game['black_pid']}](黑) vs "
              f"{game['white']}[{game['white_pid']}](白)  → {winner}  [{game['reason']}]"
              f"  {game['game_id']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="gomoku_arena.batch_cli", description="五子棋批量对战（交换先手，分色统计）"
    )
    parser.add_argument("--config", default=None, help="参赛方配置文件（JSON）")
    parser.add_argument("--print-config-template", action="store_true",
                        help="打印配置文件模板后退出")
    parser.add_argument("--games", type=int, default=10, help="总对局数（逐局交换先手）")

    parser.add_argument("--black-engine", default="trained", help="黑方引擎程序（适配器）")
    parser.add_argument("--black-pid", default=None, help="黑方参赛方唯一标识，如 dsh")
    parser.add_argument("--black-name", default=None, help="黑方展示名")
    parser.add_argument("--black-version", default=None, help="黑方模型版本")
    parser.add_argument("--black-checkpoint", default=None, help="黑方模型权重路径")

    parser.add_argument("--white-engine", default="trained", help="白方引擎程序（适配器）")
    parser.add_argument("--white-pid", default=None, help="白方参赛方唯一标识，如 codex")
    parser.add_argument("--white-name", default=None, help="白方展示名")
    parser.add_argument("--white-version", default=None, help="白方模型版本")
    parser.add_argument("--white-checkpoint", default=None, help="白方模型权重路径")

    parser.add_argument("--move-timeout", type=int, default=1000, help="每步时限 ms")
    parser.add_argument("--game-timeout", type=int, default=120_000, help="整局墙钟上限 ms")
    parser.add_argument("--seed", type=int, default=None, help="起始随机种子（逐局 +1）")
    parser.add_argument("--save", action="store_true", help="写入 records/games.jsonl")
    parser.add_argument("--records", type=Path, default=RECORDS_PATH, help="棋谱文件路径")
    parser.add_argument("--summary-json", type=Path, default=None, help="把统计结果写成 JSON 文件")
    args = parser.parse_args(argv)

    if args.print_config_template:
        print(json.dumps(CONFIG_TEMPLATE, ensure_ascii=False, indent=2))
        return 0
    if args.games < 1:
        raise SystemExit("--games 至少为 1")

    black, white = build_participants(args)
    base_config = MatchConfig(
        seed=args.seed,
        board_size=BOARD_SIZE,
        move_timeout_ms=args.move_timeout,
        game_timeout_ms=args.game_timeout,
    )

    store = Store(args.records)
    on_record = store.append if args.save else None

    def match_factory(b: Participant, w: Participant, config: MatchConfig, batch_info: dict) -> Match:
        return Match(b, w, config, batch=batch_info)

    try:
        batch = Batch(black, white, args.games, base_config, match_factory, on_record=on_record)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        summary = batch.run()
    except KeyboardInterrupt:
        print("\n已中断；本批已完成的棋谱仍然有效", file=sys.stderr)
        summary = batch.summary()

    _print_report(summary, black, white)

    if args.save:
        print(f"\n棋谱已写入 {store.path}（文件内共 {store.total_written} 局）")
    if args.summary_json:
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"统计已写入 {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
