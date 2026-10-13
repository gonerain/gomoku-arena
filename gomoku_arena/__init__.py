"""五子棋 RL 竞技场 —— 供两个（强化学习）程序对弈并观察训练效果。

模块：
  rules.py        棋盘与规则判定（规则在此冻结：15x15，黑先，>=5 连获胜，双方无禁手）
  match.py        单局编排（裁判主体），NDJSON 引擎协议
  engine_host.py  引擎子进程、协议收发、超时/崩溃/进程树清理
  server.py       HTTP 服务、比赛记录、批量对战调度
  batch.py        批量对战与分色统计
  cli.py          命令行跑单局
  batch_cli.py    命令行跑批量对战

注意：本包原名 platform，与 Python 标准库 platform 同名会造成导入冲突，故改名。
启动方式：python -m gomoku_arena.server
"""

__all__ = ["rules", "match", "engine_host", "server", "batch", "cli", "batch_cli"]
