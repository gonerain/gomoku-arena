"""命令行批量对战：连续 N 局，逐局交换先后手，输出分色统计。

    python -m gomoku_arena.batch_cli --black random --white greedy --games 20

DSH 与 Codex 各自训练完一个模型后，用这个命令跑若干局即可得到可信的对比数字：
交换先手消除了先手优势，分色统计能看出某一方是否依赖执黑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .batch import Batch
from .cli import build_spec
from .match import MatchConfig
from .rules import BOARD_SIZE
from .server import RECORDS_PATH, Store, launch_spec, with_checkpoint


def _print_report(summary: dict, black: str, white: str) -> None:
    print()
    print(f"批量对战 {summary['batch_id']}  {black} vs {white}")
    print(f"请求 {summary['requested']} 局，完成 {summary['played']} 局，"
          f"用时 {summary.get('duration_ms')}ms，状态 {summary['status']}")
    print()
    header = f"{'引擎':<16}{'执子':<8}{'局':>4}{'胜':>5}{'负':>5}{'平':>5}{'超时':>6}{'非法':>6}{'崩溃':>6}{'墙钟超时':>9}{'胜率':>8}"
    print(header)
    print("-" * 100)
    for name in (black, white):
        data = summary["sides"][name]
        for key, label in (("as_black", "黑"), ("as_white", "白")):
            side = data[key]
            rate = f"{side['win'] / side['games']:.1%}" if side["games"] else "-"
            print(f"{name:<16}{label:<8}{side['games']:>4}{side['win']:>5}{side['loss']:>5}"
                  f"{side['draw']:>5}{side['timeout']:>6}{side['illegal_move']:>6}"
                  f"{side['crash']:>6}{side['game_timeout']:>9}{rate:>8}")
    print("-" * 100)
    for name in (black, white):
        total = summary["totals"][name]
        rate = f"{total['win_rate']:.1%}" if total["win_rate"] is not None else "-"
        print(f"{name:<16}{'合计':<8}{total['games']:>4}{total['win']:>5}{total['loss']:>5}"
              f"{total['draw']:>5}{total['timeout']:>6}{total['illegal_move']:>6}"
              f"{total['crash']:>6}{total['game_timeout']:>9}{rate:>8}")
    print()
    for name in (black, white):
        total = summary["totals"][name]
        print(f"{name}: 胜率 {total['win_rate']:.1%}  负率 {total['loss_rate']:.1%}  "
              f"平局率 {total['draw_rate']:.1%}")
    print()
    print("逐局结果：")
    for game in summary["games"]:
        winner = game["winner"] or "无胜者"
        print(f"  #{game['moves']:>3}手  {game['black']}(黑) vs {game['white']}(白)  "
              f"→ {winner}  [{game['reason']}]  {game['game_id']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="gomoku_arena.batch_cli", description="五子棋批量对战（交换先手）"
    )
    parser.add_argument("--black", default="random", help="先执黑的引擎")
    parser.add_argument("--white", default="greedy", help="先执白的引擎")
    parser.add_argument("--games", type=int, default=10, help="总对局数（逐局交换先手）")
    parser.add_argument("--black-checkpoint", default=None, help="黑方模型 checkpoint")
    parser.add_argument("--white-checkpoint", default=None, help="白方模型 checkpoint")
    parser.add_argument("--move-timeout", type=int, default=1000, help="每步时限 ms")
    parser.add_argument("--game-timeout", type=int, default=120_000, help="整局墙钟上限 ms")
    parser.add_argument("--seed", type=int, default=None, help="起始随机种子（逐局 +1）")
    parser.add_argument("--save", action="store_true", help="写入 records/games.jsonl")
    parser.add_argument("--records", type=Path, default=RECORDS_PATH, help="棋谱文件路径")
    parser.add_argument("--summary-json", type=Path, default=None, help="把统计结果写成 JSON 文件")
    args = parser.parse_args(argv)

    if args.games < 1:
        raise SystemExit("--games 至少为 1")

    base_config = MatchConfig(
        seed=args.seed,
        board_size=BOARD_SIZE,
        move_timeout_ms=args.move_timeout,
        game_timeout_ms=args.game_timeout,
    )
    checkpoints = {args.black: args.black_checkpoint, args.white: args.white_checkpoint}

    def spec_factory(key: str) -> dict:
        return with_checkpoint(launch_spec(build_spec(key)), checkpoints.get(key))

    store = Store(args.records)
    on_record = store.append if args.save else (lambda rec: None)

    batch = Batch(args.black, args.white, args.games, base_config, spec_factory, on_record)
    summary = batch.run()
    _print_report(summary, args.black, args.white)

    if args.save:
        print(f"棋谱已写入 {store.path}（共 {len(store.records)} 局）")
    if args.summary_json:
        import json

        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"统计已写入 {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
