# 五子棋 RL 竞技场

让两个（强化学习）程序互相对弈、并在浏览器里实时观察训练效果的平台。

平台**只做四件事**：当裁判、给引擎一个极简的 Python 接口、在浏览器里实时画棋盘、把每局
棋谱和结果落成 JSON 文件。它不实现任何训练算法，也不限制你用 PPO、AlphaZero、DQN 还是
进化策略 —— 训练完全解耦，接入方式就是"提供一个会说协议的子进程"。

```
┌──────────────┐   stdin/stdout JSON   ┌─────────────────┐   stdin/stdout JSON   ┌──────────────┐
│  DSH 的模型   │ ◀───────────────────▶ │   裁判 (平台)    │ ◀───────────────────▶ │ Codex 的模型  │
└──────────────┘                       └────────┬────────┘                       └──────────────┘
                                                │ 轮询 /api/state
                                                ▼
                                    浏览器实时棋盘 + 比赛记录 + 批量对战统计
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

规则的单一定义在 `gomoku_arena/rules.py`（`BOARD_SIZE`、`win_line`、`PROTOCOL`），
裁判、引擎与浏览器界面都从同一处取用；`/api/board` 会把规则摘要下发给前端显示，
`MatchConfig` 与 `POST /api/matches` 都会拒绝非 15×15 的棋盘尺寸。

一个术语提醒：部分文献里的 "Standard Gomoku" 指**恰好 5 连 + 黑方长连判负**，与
free-style 不同。本平台采用上表冻结的那一套，不要与那个变体混淆。

## 快速开始

零外部依赖，只需 Python 3.10+（本机为 3.13）。

```bash
# 起服务
python -m gomoku_arena.server --port 8090
# 打开浏览器
#   http://127.0.0.1:8090
```

页面里可以：实时看棋盘（落子带手数、最后一手高亮、终局五连描边）、开单局、开批量对战、
看最近对局列表、拖动进度条回放任意一局、看覆盖全部棋谱的完整统计。
**历史回放与实时对局互不干扰** —— 回放期间实时对局仍在后台推进，不会被轮询覆盖。

命令行跑单局：

```bash
python -m gomoku_arena.cli --black random --white greedy --seed 42 --board
python -m gomoku_arena.cli --protocol          # 打印引擎协议
```

## 批量对战（DSH vs Codex）

连续 N 局、**逐局交换先后手**，并分别统计双方执黑/执白的成绩。交换先手是为了把先手优势
从胜负里剥离出来，否则单局结果没有统计意义。

```bash
python -m gomoku_arena.batch_cli --black random --white greedy --games 20 --save

# 指定模型版本（checkpoint 会写进棋谱与统计）
python -m gomoku_arena.batch_cli --black trained --white trained --games 20 \
  --black-checkpoint checkpoints/dsh_ppo_v3.pt \
  --white-checkpoint checkpoints/codex_az_v7.pt --save
```

输出示例：

```
批量对战 b-20261013-141534-f4ee  random vs greedy
请求 8 局，完成 8 局，用时 1392ms，状态 finished

引擎              执子     局    胜    负    平   超时   非法   崩溃  墙钟超时     胜率
random          黑        4    0    4    0    0    0    0        0    0.0%
random          白        4    0    4    0    0    0    0        0    0.0%
greedy          黑        4    4    0    0    0    0    0        0  100.0%
greedy          白        4    4    0    0    0    0    0        0  100.0%
```

浏览器里也能开批量对战（「批量（交换先手）」按钮），面板会实时显示进度与分色统计，
并可随时中止。

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

### 判罚与归责

| 情况 | 结果 |
|---|---|
| 单步超过 `deadline_ms` 未落子 | 该方判负，`reason=timeout` |
| 进程退出 / 崩溃 | 该方判负，`reason=crash`（附退出码，`退出码 2` 通常意味着模型加载失败） |
| 坐标越界、落在已占点、非整数 | 该方判负，`reason=illegal_move` |
| stdout 不是合法协议帧 | 该方判负，`reason=bad_output` |
| 回合里回非 `move` 帧 | 该方判负，`reason=protocol` |
| 主动发 `resign` | 该方判负，`reason=resign` |
| 可执行文件/工作目录不存在 | 该方判负，`reason=start_failed` |
| **整局墙钟上限用完** | **无胜负中止，`reason=game_timeout`，不归责任何一方** |
| 握手阶段超时或异常 | 归责到"正在等 ready 的那一方"，判其负 |

整局墙钟上限（`game_timeout_ms`）是平台为了防止对局无限拖下去而设的护栏，双方共享同一份
时间，因此它被触发时**不能**归责于当前落子方。实现上，每一步的等待时长取"单步时限"与
"全局墙钟剩余"的较小值 —— 否则单步时限很长的引擎会把已经耗尽的全局墙钟又拖长一整个单步时限。

## 接入你自己的 RL 模型

照抄 `engines/py_random/engine.py`（最小实现）或 `engines/py_trained/engine.py`（接入点）：

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

### checkpoint 诚实性（重要）

平台**绝不让未训练的策略伪装成 RL 模型**。`py_trained` 的两种模式：

| 场景 | 行为 |
|---|---|
| **未指定 checkpoint**（`GOMOKU_MODEL` 为空） | 进入演示模式：用启发式走子，自报 `meta.baseline=true`、`algo=heuristic-demo`。前端会标注「基线·非RL」。仅用于跑通流程。 |
| **指定了 checkpoint，但文件不存在** | 打印原因并以**退出码 2** 结束 → 平台判该方 `start_failed` 负，0 手。 |
| **checkpoint 存在，但 `_load()` 未实现**（仍是模板占位） | 同样以退出码 2 结束，明确告诉你"推理未实现"，而不是偷偷退回启发式。 |
| **推理失败 / 返回非法坐标** | 该方判负，原因写明是推理问题。 |

配置模型版本：用环境变量指定 checkpoint，然后在 `gomoku_arena/server.py` 的 `ENGINES` 里配置，
或直接在 API / CLI 里用 `checkpoints` 参数指定（不会修改全局注册表）：

```python
"dsh-v3": {
    "name": "dsh-ppo", "version": "v3",
    "cmd": [sys.executable, str(ROOT / "engines/py_trained/engine.py")],
    "env": {"GOMOKU_MODEL": "checkpoints/dsh_v3.pt"},
    "meta": {"algo": "ppo", "episodes": 12000, "budget": "2e6 steps", "baseline": False},
},
```

## 棋谱记录

`records/games.jsonl`，追加式，**每局一行**，纯 JSON 不需要数据库：

```json
{"v":1,"game_id":"g-20261013-134754-590d","ts":"2026-10-13T13:47:54+00:00","board_size":15,
 "seed":899274,"move_timeout_ms":1000,"game_timeout_ms":120000,
 "black":{"name":"greedy","version":"v0","color":"black","checkpoint":"none",
          "meta":{"algo":"heuristic-1ply","baseline":true}},
 "white":{"name":"trained","version":"v0","color":"white","checkpoint":"checkpoints/dsh_ppo_v3.pt",
          "meta":{"algo":"rl-checkpoint","baseline":false}},
 "moves":[{"n":1,"color":"black","x":7,"y":7,"ms":0}, ...],
 "result":{"winner":"black","reason":"five","line":[[7,7],[8,8],[9,9],[10,10],[11,11]]},
 "batch":{"batch_id":"b-20261013-143000-ab12","index":3,"of":20},
 "duration_ms":176}
```

研究训练效果需要的字段都在里面：**参赛模型版本**（`name`/`version`/`meta`）、**checkpoint 标识**
（`checkpoint`）、**先后手**（`black`/`white` 及其 `color`）、**对局结果**（`winner`/`reason`/`line`）、
**训练预算**（`meta` 里你自己写的字段）、**批量归属**（`batch`）、以及 `seed`（配合确定性引擎可复现整局）。
异常结束的对局会多一个 `error` 字段，说明是哪一方、什么原因。

## 项目结构

```
gomoku_arena/                ← 平台本体（原 platform/，与标准库 platform 同名会造成导入冲突）
├─ rules.py                  棋盘、落子合法性、五连/平局判定、协议常量、规则单一定义
├─ match.py                  单局编排（裁判主体）+ 超时归责规则
├─ engine_host.py            引擎子进程、NDJSON 收发、进程树清理、退出码归因
├─ server.py                 HTTP 服务、比赛记录、完整统计、批量对战调度
├─ batch.py                  批量对战与分色统计
├─ cli.py                    命令行跑单局
└─ batch_cli.py              命令行跑批量对战
web/index.html               实时棋盘 + 记录列表 + 回放 + 批量面板（原生 Canvas，零依赖）
engines/
├─ engine_loop.py            引擎侧协议实现（写引擎时复制它）
├─ py_random/                最小实现，兼作协议自检
├─ py_greedy/                一步贪心，RL 训练的有意义基线
├─ py_trained/               ← RL 接入点（含 checkpoint 诚实性检查）
└─ test_slow|test_crash|test_illegal/   健壮性测试用的"坏引擎"
tests/
├─ test_rules.py             规则单元测试（37 项）
├─ test_robustness.py        判罚归责 / 进程树清理 / checkpoint 诚实性（27 项）
└─ test_api.py               HTTP 接口、回放、批量统计（30 项）
records/games.jsonl          棋谱
```

## 测试

```bash
python tests/test_rules.py          # 规则：横竖斜/边界/长连/未胜/非法/平局
python tests/test_robustness.py     # 判罚归责 + 进程树清理 + checkpoint 诚实性
python -m gomoku_arena.server --port 8090 &
python tests/test_api.py            # 接口、回放、批量统计与交换先手
```

`test_robustness.py` 里最关键的一项是**进程树清理**：测试引擎会派生一个 600 秒的孙进程再崩溃，
平台必须在判负的同时把孙进程也收拾掉。实现方式是 Windows **Job Object（kill-on-close）**，并把
子进程以挂起态启动、先入 Job 再恢复 —— 这样它之后派生的任何后代都必然在同一个 Job 里，
关掉 Job 句柄即被内核整棵终结。之所以不能只靠 `taskkill /T` 或按父 pid 递归枚举：父进程一旦
退出，后代会被操作系统 reparent，那时无论哪种方式都找不到它们了。POSIX 下则是让子进程
`start_new_session=True` 自成进程组，并且只有在确认它是组长时才 `killpg`，避免误杀裁判。

## HTTP 接口

```
GET  /                      实时棋盘页面
GET  /api/board             棋盘尺寸与规则摘要（前端据此配置棋盘，避免前后端不一致）
GET  /api/engines           可用引擎（含 baseline / checkpoint 标识）
GET  /api/models            可用模型文件
GET  /api/games?limit=N     最近对局列表（不含着法，回放时单独取）
GET  /api/stats             完整统计（覆盖全部棋谱）
GET  /api/games/<id>        单局完整棋谱（回放用）
GET  /api/state?game_id=X   实时状态（进行中返回实时快照，已结束返回棋谱）
POST /api/matches           开始一局
     {"black":"greedy","white":"random","move_timeout_ms":1000,
      "game_timeout_ms":120000,"seed":42,"checkpoints":{"white":"checkpoints/x.pt"}}
POST /api/batches           开始一批（逐局交换先手）
     {"black":"greedy","white":"random","games":20, ...同上}
GET  /api/batches/<id>      批量进度与分色统计
DELETE /api/batches/<id>    中止批量对战
```

返回的状态里 `running` 表示是否进行中，字段形态在"进行中"与"已结束"两种情况下是统一的。

## 已知边界

- **浏览器用轮询**而非 WebSocket。一步棋的思考时间在百毫秒级以上，轮询的延迟和开销可以忽略；
  历史回放期间轮询自动降频到 5s，但仍持续，以便实时对局不中断。
- **只在本机 spawn 子进程**，不做远程/容器引擎接入。
- **批量对战是顺序执行的**（一局接一局），不做并行；也没有 Elo / 数据库 / 淘汰赛。
- 棋盘尺寸固定 15×15，不做可配置（避免前后端与协议三处各写一份）。
- **POSIX 下的进程树清理逻辑未在本机验证**（当前环境为 Windows，走 Job Object 路径）。
- Windows 控制台可能把中文显示成乱码（GBK 代码页），但 `games.jsonl` 与接口一律是正确的 UTF-8。
