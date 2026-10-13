"""五子棋对抗平台 —— 供两个（强化学习）程序对弈并观察训练效果。

四个核心模块：
  rules.py        棋盘与规则判定（裁判的一半）
  match.py        单局编排（裁判的另一半）
  engine_host.py  引擎子进程与 NDJSON 协议
  server.py       HTTP 服务 + 浏览器实时棋盘 + 比赛记录

规则见 README：15x15，黑先，>=5 连获胜，双方无禁手，无开局规则。
"""

__all__ = ["rules", "match", "engine_host", "server"]
