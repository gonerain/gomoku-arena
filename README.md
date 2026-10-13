# 五子棋 RL 竞技场

让两个（强化学习）程序互相对弈、并在浏览器里实时观察训练效果的最小平台。

平台**只做四件事**：当裁判、给引擎一个极简的 Python 接口、在浏览器里实时画棋盘、把每局
棋谱和结果落成 JSON 文件。它不实现任何训练算法，也不限制你用 PPO、AlphaZero、DQN 还是
进化策略 —— 训练完全解耦，接入方式就是"提供一个会说协议的子进程"。

```
┌──────────────┐   stdin/stdout JSON   ┌─────────────────┐   stdin/stdout JSON   ┌──────────────┐
│  DSH 的模型   │ ◀───────────────────▶ │   裁判 (平台)    │ ◀───────────────────▶ │ Codex 的模型  │
└──────────────┘                       └────────┬────────┘                       └──────────────┘
                                                │ 轮询 /api/state
                                                ▼
                                    浏览器实时棋盘 + 比赛记录
                                                │
                                                ▼
                                     records/games.jsonl
```

## 规则（已冻结）

> **15×15；黑先；双方轮流落一子于空点；先形成 ≥5 连（横/竖/斜）者胜；双方均无禁手；无开局规则；棋盘满且无胜者判平局。**

这不是"简化版"，而是与连珠（Renju）的规则差异决定的。已核对 RIF 官方规则原文
（[International Rules of Renju](https://mail.renju.net/rifrules/)）：

| | Renju（连珠） | 本平台（五子棋 / free-style） |
|---|---|---|
| 黑方胜 | **仅恰好 5 连**，黑长连（6+）**不算胜** | ≥5 连即胜 |
| 白方胜 | 5 连及长连都算胜（9.1） | ≥5 连即胜 |
| 黑方禁手 | **有**：长连(9.2a)、双四(9.2b)、双三(9.2c) 判白胜 | **无** |
| 开局规则 | 12.2–12.8：13 种开局型 + 第五手黑提两案白择一 | 无，自由开局 |
| 首手限制 | 黑首手必须天元（4.2） | 无 |

**因此本平台不实现禁手系统。** 这不只是省事：连珠的禁手判定（尤其 9.3 的递归"合法活四"）
本身存在实现歧义，会把噪声引入胜负信号；而在无禁手规则下，胜负完全确定，胜率可以直接
当作可信的训练指标。

一个术语提醒：部分文献里的 "Standard Gomoku" 指**恰好 5 连 + 黑方长连判负**，与
free-style 不同。本平台采用上表冻结的那一套，不要与那个变体混淆。

## 快速开始

零外部依赖，只需 Python 3.10+（本机为 3.13）。

```bash
# 1) 起服务 + 自动开一局
python -m platform.server --port 8090 --autostart greedy:random

# 2) 打开浏览器
#    http://127.0.0.1:8090
```

页面里可以：实时看棋盘（落子带手数、最后一手高亮、终局五连描边）、点「开始对局」换引擎/时限/
随机种子、看比赛记录列表、拖动进度条回放任意一局、看总战绩。

命令行跑单局（不开浏览器）：

```bash
python -m platform.cli --black random --white greedy --seed 42 --board
python -m platform.cli --black greedy --white "engines/py_trained/engine.py" --moves
python -m platform.cli --protocol          # 打印引擎协议
```

## 引擎协议（全部内容）

**stdin 收 JSON，stdout 发 JSON，每行一个对象；调试日志一律走 stderr。** 整局只有 4 种帧：

```
平台 → 引擎  {"type":"init","game_id":..,"board_size":15,"color":"black","seed":..}
引擎 → 平台  {"type":"ready","name":..,"version":..,"meta":{..}}
平台 → 引擎  {"type":"turn","moves":[["black",7,7],["white",7,8]],"deadline_ms":1000}
引擎 → 平台  {"type":"move","x":8,"y":8}
平台 → 引擎  {"type":"gameover","winner":"black","reason":"five"}
引擎 → 平台  {"type":"resign"}            （可选：主动认输）
```

- **坐标**：`x` = 列 0–14，`y` = 行 0–14，原点在左下角。仅显示用：`x`→`A..O`，`y`→`1..15`，天元 `H8`。
- **每回合收到完整 `moves` 历史**，所以引擎不需要维护任何跨回合状态 —— 每步都是纯函数式的
  `观测 → 动作`，这对 RL 侧最省事。
- `ready` 里的 **`meta` 原样存进棋谱**。DSH 可以写 `{"algo":"ppo","episodes":12000}`，
  Codex 可以写 `{"algo":"alphazero","sims":200}` —— 平台只存不解释，这是与训练系统解耦的具体做法。
- 引擎写 stdout 的任何非协议内容都会被视为脏输出并判负，所以**不要用 `print` 打日志**，用 `sys.stderr`。

判负规则（裁判是唯一权威）：

| 情况 | 结果 |
|---|---|
| 超过 `deadline_ms` 未落子 | 该方判负，`reason=timeout` |
| 进程退出 / 崩溃 | 该方判负，`reason=crash` |
| 坐标越界、落在已占点、非整数 | 该方判负，`reason=illegal_move` |
| stdout 不是合法协议帧 | 该方判负，`reason=bad_output` |
| 对局尚未结束就发 `resign` | 该方判负，`reason=resign` |

## 接入你自己的 RL 模型

照抄 `engines/py_random/engine.py`（最小实现）或 `engines/py_trained/engine.py`（RL 模板）：

```python
import sys
from pathlib import Path
sys.path.insert(0, "engines")              # 或把 engine_loop.py 复制到你自己的目录
from engine_loop import run

def choose(view, ctx):                     # view: 只读棋盘视图；ctx: 对局元信息（含 seed）
    ...
    return (x, y)

run(choose, name="dsh-ppo", version="v3", meta={"algo": "ppo", "episodes": 12000})
```

`py_trained/engine.py` 里给了观测编码的默认约定（`state_tensor`，`(4,15,15)` float32：
己方/对方/上一手/当前执子方）和动作掩码（`legal_mask`）。**训练侧只要对齐这个编码，就能把
训练时的网络直接搬来推理**；想换成别的编码，改那两个函数即可，平台侧一行都不用动。

配置模型版本：用环境变量指定 checkpoint，然后在 `platform/server.py` 的 `ENGINES` 里加一条：

```python
"dsh-v3": {
    "name": "dsh-ppo", "version": "v3",
    "cmd": [sys.executable, str(ROOT / "engines/py_trained/engine.py")],
    "env": {"GOMOKU_MODEL": "checkpoints/dsh_v3.pt"},
    "meta": {"algo": "ppo", "episodes": 12000, "budget": "2e6 steps"},
},
```

## 棋谱记录

`records/games.jsonl`，追加式，**每局一行**，纯 JSON 不需要数据库：

```json
{"v":1,"game_id":"g-20261013-134754-590d","ts":"2026-10-13T13:47:54+00:00","board_size":15,
 "seed":899274,"move_timeout_ms":1000,
 "black":{"name":"greedy","version":"v0","color":"black","meta":{"algo":"heuristic-1ply"}},
 "white":{"name":"random","version":"v0","color":"white","meta":{"algo":"uniform"}},
 "moves":[{"n":1,"color":"black","x":7,"y":7,"ms":0}, ...],
 "result":{"winner":"black","reason":"five","line":[[7,7],[8,8],[9,9],[10,10],[11,11]]},
 "duration_ms":176}
```

研究训练效果需要的字段都在里面：**参赛模型版本**（`name`/`version`/`meta`）、**先后手**
（`black`/`white` 及其 `color`）、**对局结果**（`winner`/`reason`/`line`）、**训练预算**（`meta` 里
你自己写的字段）、以及 `seed`（配合确定性引擎可复现整局）。异常结束的对局会多一个 `error`
字段，说明是哪一方、什么原因。

统计示例：

```bash
python -c "
import json, collections
recs = [json.loads(l) for l in open('records/games.jsonl', encoding='utf-8')]
w = collections.Counter((r['black']['name'], r['white']['name'], r['result']['winner']) for r in recs)
print(f'总局数 {len(recs)}')
for k, v in w.items(): print(k, v)
"
```

## 项目结构

```
platform/
├─ platform/                  ← 平台本体，四个模块对应四件核心功能
│  ├─ rules.py                棋盘、落子合法性、五连/平局判定
│  ├─ match.py                单局编排（裁判主体）+ 协议常量 PROTOCOL
│  ├─ engine_host.py          引擎子进程、NDJSON 收发、超时/崩溃/进程树清理
│  ├─ server.py               HTTP 服务、比赛记录、进行中对局的实时状态
│  └─ cli.py                  命令行跑单局
├─ web/index.html             实时棋盘 + 记录列表 + 回放（零依赖，原生 Canvas）
├─ engines/
│  ├─ engine_loop.py          引擎侧协议实现（写引擎时复制它）
│  ├─ py_random/              最小实现，兼作协议自检
│  ├─ py_greedy/              一步贪心，RL 训练的有意义基线
│  ├─ py_trained/             ← RL 接入模板
│  └─ test_slow|test_crash|test_illegal/   健壮性测试用的"坏引擎"
├─ tests/
│  ├─ test_robustness.py      超时/崩溃/非法落子/进程树清理
│  └─ test_api.py             浏览器实际会走的全部接口路径
└─ records/games.jsonl        棋谱
```

## 测试

```bash
python tests/test_robustness.py     # 10 项：判负归因 + 进程树清理
python -m platform.server --port 8090 &
python tests/test_api.py            # 10 项：接口与回放
```

`test_robustness.py` 里最关键的一项是**进程树清理**：测试引擎会派生一个 600 秒的孙进程再崩溃，
平台必须在判负的同时把孙进程也收拾掉。实现方式是 Windows **Job Object（kill-on-close）**，并把
子进程以挂起态启动、先入 Job 再恢复 —— 这样它之后派生的任何后代都必然在同一个 Job 里，
关掉 Job 句柄即被内核整棵终结。之所以不能只靠 `taskkill /T` 或按父 pid 递归枚举：父进程一旦
退出，后代会被操作系统 reparent，那时无论哪种方式都找不到它们了。

## HTTP 接口

```
GET  /                      实时棋盘页面
GET  /api/engines           可用引擎
GET  /api/games?limit=N     对局列表（含末尾若干手，够画棋盘）
GET  /api/games/<id>        单局完整棋谱
GET  /api/state?game_id=X   实时状态（进行中返回实时快照，已结束返回棋谱）
POST /api/matches           开始一局
     {"black":"greedy","white":"random","move_timeout_ms":1000,
      "game_timeout_ms":120000,"seed":42}
```

返回的状态里 `running` 表示是否进行中，字段形态在"进行中"与"已结束"两种情况下是统一的。

## 已知边界

- **浏览器用 400ms 轮询**而非 WebSocket。一步棋的思考时间在百毫秒级以上，轮询的延迟和开销
  可以忽略；等真需要更高频率时再换。
- **只在本机 spawn 子进程**，不做远程/容器引擎接入。
- **胜负判定为 ≥5 连**（无禁手），见上文规则表。
- Windows 控制台可能把中文显示成乱码（GBK 代码页），但 `games.jsonl` 与接口一律是正确的 UTF-8。
