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

## 身份模型（三层分离）

DSH 与 Codex 各自训练、各自加载权重，但可以**共用同一个 Python 适配器**。为此平台把三件事分开：

| 层 | 字段 | 含义 |
|---|---|---|
| 参赛方 | `pid`（如 `dsh` / `codex`） | 统计与棋谱的**唯一标识**，必须双方不同 |
| 引擎程序 | `engine`（如 `trained`） | 说 NDJSON 协议的适配器，双方可以相同 |
| 模型权重 | `checkpoint` + `version` | 该参赛方加载的具体模型文件与版本 |

规则：

- **权重跟随参赛方，不跟随颜色**。批量对战逐局交换先手，`dsh` 执黑与执白时加载的都是它自己的权重。
- **统计以 `pid` 为键**，绝不用引擎自报名称。即使两边的引擎进程都自称 `trained`，成绩也不会混。
- 同一 `pid` 在不同局里出现不同 `checkpoint` 时，统计会给出 `mixed_checkpoints` 标记，便于发现配置错误。
- 两方 `pid` 相同、或同一引擎同一权重且版本相同，创建对局会**直接报 400**，避免"自己打自己"被当成有效对比。
- 引擎自报名称与平台注册名不一致时，棋谱里会有 `name_mismatch` 字段。

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

**批量对战的实时观战**：批量进行时，浏览器会逐手显示**当前那一局**（标题栏标注批次与
"第 N/M 局"），每局结束后自动切到下一局，右侧实时面板同步显示双方比分。
**历史回放与实时观战互不干扰** —— 点开历史棋谱后画面锁定在该局，轮询降频但仍继续，
实时对局不会被覆盖；点「回到实时」即可恢复。

命令行跑单局：

```bash
python -m gomoku_arena.cli --black random --white greedy --seed 42 --board
python -m gomoku_arena.cli --protocol          # 打印引擎协议
```

## 批量对战（DSH vs Codex）

连续 N 局、**逐局交换先后手**，并分别统计双方执黑/执白的成绩。交换先手是为了把先手优势
从胜负里剥离出来，否则单局结果没有统计意义。

```bash
# 命令行直接指定：两边同一个适配器，各自的权重与身份
python -m gomoku_arena.batch_cli --black-engine trained --black-pid dsh \
    --black-checkpoint checkpoints/dsh_ppo_v3.pt --black-version v3 \
    --white-engine trained --white-pid codex \
    --white-checkpoint checkpoints/codex_az_v7.pt --white-version v7 \
    --games 20 --save

# 配置文件方式（推荐：把"某一届比赛"的双方固化下来）
python -m gomoku_arena.batch_cli --config matchups/dsh_vs_codex.json --games 20 --save
python -m gomoku_arena.batch_cli --print-config-template   # 打印配置模板

# 基线自检（非 RL，用来确认链路通畅）
python -m gomoku_arena.batch_cli --black-engine random --white-engine greedy --games 10
```

输出示例：

```
批量对战 b-20261013-144138-bd4f
  黑方(首局执黑) : DSH [dsh] 引擎=ckpt_test 版本=ta1 权重=.../model_a/meta.json
  白方(首局执白) : Codex [codex] 引擎=ckpt_test 版本=tb2 权重=.../model_b/meta.json
请求 20 局，完成 20 局，用时 8421ms，状态 finished

参赛方        引擎      执子    局    胜    负    平  超时  非法  崩溃 墙钟超时    胜率
dsh           ckpt_test 黑      10   10    0    0    0    0    0       0  100.0%
dsh           ckpt_test 白      10    0   10    0    0    0    0       0    0.0%
codex         ckpt_test 黑      10   10    0    0    0    0    0       0  100.0%
codex         ckpt_test 白      10    0   10    0    0    0    0       0    0.0%
```

浏览器里也能开批量对战，「批量（交换先手）」按钮旁边可分别填两边的**引擎、身份 id 与权重路径**；
面板实时显示进度、当前局 ID 与分色统计，并可随时中止。

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
    ...                                    # view.in_bounds/is_empty/neighbors/free_cells/would_win
    return (x, y)

run(choose, name="dsh-ppo", version="v3", meta={"algo": "ppo", "episodes": 12000})
```

`py_trained/engine.py` 里给了观测编码的默认约定（`state_tensor`，`(4,15,15)` float32：
己方/对方/上一手/当前执子方）和动作掩码（`legal_mask`）。**训练侧只要对齐这个编码，就能把
训练时的网络直接搬来推理**；想换成别的编码，改那两个函数即可，平台侧一行都不用动。

DSH 与 Codex 的推荐做法：各自 fork 一份 `py_trained/engine.py`（或共用同一份），分别实现
`_load()` 与 `_infer()`，然后在 `matchups/` 里各写一份配置指向各自的 checkpoint。

### checkpoint 诚实性（重要）

平台**绝不让未训练的策略伪装成 RL 模型**。`py_trained` 的两种模式：

| 场景 | 行为 |
|---|---|
| **未指定 checkpoint**（`GOMOKU_MODEL` 为空） | 进入演示模式：用启发式走子，自报 `meta.baseline=true`、`algo=heuristic-demo`。前端标注「基线·非RL」。仅用于跑通流程。 |
| **指定了 checkpoint，但文件不存在** | 打印原因并以**退出码 2** 结束 → 平台判该方 `start_failed` 负，0 手，判负原因写明"退出码 2"。 |
| **checkpoint 存在，但 `_infer()` 未实现**（仍是模板占位） | 同样以退出码 2 结束，明确告诉你"推理未实现"，而不是偷偷退回启发式。检测方式是按源码定位 `_infer` 的函数体，因此"就地改模板"也能被识别。 |
| **推理失败 / 返回越界或已占点坐标** | 该方判负，原因写明是推理问题。 |

平台侧没有任何"允许占位实现"的开关，也没有静默回退路径。

配置方式有两种，都不修改全局注册表。**外部引擎可以直接写完整命令**（常用于引擎装在别的
Python 环境里，或引擎代码不在本仓库）：

```bash
# ① CLI：注册表引擎 + 各自权重
python -m gomoku_arena.batch_cli --black-engine trained --black-pid dsh \
    --black-checkpoint checkpoints/dsh_ppo_v3.pt --white-engine trained --white-pid codex \
    --white-checkpoint checkpoints/codex_az_v7.pt --games 20 --save

# ② CLI：外部引擎（完整命令含解释器；checkpoint 会被注入 GOMOKU_MODEL）
python -m gomoku_arena.cli \
    --black "C:\Python313\python.exe D:\Code\dsh_gomoku\arena\engines\py_trained\engine.py" \
    --black-pid dsh --black-version v1 \
    --black-checkpoint "D:\Code\dsh_gomoku\runs\az\checkpoints\chosen.pt" \
    --white greedy --white-pid baseline --move-timeout 2000 --board

# ③ JSON 配置（推荐，见 matchups/ 下的实例；engine 字段同样支持完整命令）
python -m gomoku_arena.batch_cli --config matchups/dsh_vs_codex.json --games 20 --save
```

engine 字段的两种写法：

| 写法 | 含义 |
|---|---|
| `"trained"` | 注册表里的引擎程序（`random`/`greedy`/`trained`/`ckpt_test`/`test_crash`） |
| `"C:\\Python313\\python.exe D:\\...\\engine.py"` | 外部引擎的完整命令，原样使用给定解释器 |
| `"D:\\...\\engine.py"` | 外部脚本，用平台自身的解释器执行 |

无论哪种写法，`checkpoint` 都会被注入到引擎进程的 `GOMOKU_MODEL` 环境变量。

```json
{
  "black": {"engine": "trained", "pid": "dsh", "name": "DSH", "version": "v3",
            "checkpoint": "checkpoints/dsh_ppo_v3.pt",
            "meta": {"algo": "ppo", "episodes": 12000, "budget": "2e6 steps"}},
  "white": {"engine": "trained", "pid": "codex", "name": "Codex", "version": "v7",
            "checkpoint": "checkpoints/codex_az_v7.pt",
            "meta": {"algo": "alphazero", "sims": 200}}
}
```

HTTP 接口同构（`participants.black` / `participants.white`）：

```bash
curl -X POST http://127.0.0.1:8090/api/batches -H "Content-Type: application/json" -d '{
  "participants": {
    "black": {"engine":"trained","pid":"dsh","version":"v3","checkpoint":"checkpoints/dsh_ppo_v3.pt"},
    "white": {"engine":"trained","pid":"codex","version":"v7","checkpoint":"checkpoints/codex_az_v7.pt"}
  }, "games": 20, "move_timeout_ms": 2000}'
```

## 棋谱记录

`records/games.jsonl`，追加式，**每局一行**，纯 JSON 不需要数据库：

```json
{"v":2,"game_id":"g-20261013-134754-590d","ts":"2026-10-13T13:47:54+00:00","board_size":15,
 "seed":899274,"move_timeout_ms":1000,"game_timeout_ms":120000,
 "participants":[{"pid":"dsh","name":"DSH","engine":"trained","version":"v3","color":"black",
                  "checkpoint":"checkpoints/dsh_ppo_v3.pt","meta":{"algo":"ppo","episodes":12000}},
                 {"pid":"codex","name":"Codex","engine":"trained","version":"v7","color":"white",
                  "checkpoint":"checkpoints/codex_az_v7.pt","meta":{"algo":"alphazero"}}],
 "black":{"pid":"dsh","name":"DSH","engine":"trained","version":"v3","color":"black",
          "checkpoint":"checkpoints/dsh_ppo_v3.pt","meta":{"algo":"ppo"},
          "reported":{"name":"dsh-ppo","version":"v3"}},
 "white":{"pid":"codex","name":"Codex","engine":"trained","version":"v7","color":"white",
          "checkpoint":"checkpoints/codex_az_v7.pt","meta":{"algo":"alphazero"},
          "reported":{"name":"codex-az","version":"v7"}},
 "moves":[{"n":1,"color":"black","x":7,"y":7,"ms":0}, ...],
 "result":{"winner":"black","reason":"five","line":[[7,7],[8,8],[9,9],[10,10],[11,11]]},
 "batch":{"batch_id":"b-20261013-143000-ab12","index":3,"of":20},
 "duration_ms":176}
```

研究训练效果需要的字段都在里面：**参赛方身份**（`participants` 与 `black`/`white` 的 `pid`）、
**模型版本与权重**（`version`、`checkpoint`）、**引擎自报信息**（`reported`）、
**先后手**（`color`）、**对局结果**（`winner`/`reason`/`line`）、**训练预算**（`meta` 里你自己写的
字段）、**批量归属**（`batch`）、以及 `seed`。异常结束的对局多一个 `error` 字段。

`v` 字段是棋谱格式版本：`1` 为早期的"引擎名即身份"格式，`2` 起使用参赛方 `pid`。

## 项目结构

```
gomoku_arena/                ← 平台本体（原 platform/，与标准库 platform 同名会造成导入冲突）
├─ rules.py                  棋盘、落子合法性、五连/平局判定、协议常量、规则单一定义
├─ match.py                  单局编排（裁判主体）、参赛方身份模型、超时归责规则
├─ engine_host.py            引擎子进程、NDJSON 收发、Job Object 句柄所有权、进程树清理
├─ server.py                 HTTP 服务、比赛记录、完整统计、批量对战调度与实时观战
├─ batch.py                  批量对战、逐局交换先手、按参赛方分色统计
├─ cli.py                    命令行跑单局
└─ batch_cli.py              命令行跑批量对战（支持 --config）
web/index.html               实时棋盘 + 记录列表 + 回放 + 批量观战面板（原生 Canvas，零依赖）
matchups/
└─ dsh_vs_codex.example.json 参赛方配置模板（复制后填真实 checkpoint）
engines/
├─ engine_loop.py            引擎侧协议实现（写引擎时复制它）
├─ py_random/                最小实现，兼作协议自检（基线）
├─ py_greedy/                一步贪心，RL 训练的有意义基线（基线）
├─ py_trained/               ← RL 接入点（含 checkpoint 诚实性检查，无静默回退）
├─ py_ckpt_test/             测试用权重驱动策略（验收双模型身份，非 RL）
└─ test_slow|test_crash|test_illegal/   故障注入引擎（验收判罚与进程树清理）
tests/
├─ test_rules.py             规则单元测试
├─ test_robustness.py        判罚归责 / 句柄所有权 / 进程树清理 / checkpoint 诚实性
├─ test_identity.py          参赛方身份、统计不混淆、推理接口
├─ test_api.py               HTTP 接口、回放、批量统计
├─ test_dual_models.py       E2E-A：两份不同权重跑 20 局
├─ test_stability.py         E2E-B：100 局连续稳定性与资源泄漏监测
├─ test_batch_spectate.py    E2E-C：批量实时观战与回放隔离
└─ models/model_a|model_b/   测试权重（与 checkpoints/ 的正式模型严格分离）
records/games.jsonl          棋谱
```

## 测试

```bash
python tests/test_rules.py            # 规则：横竖斜/边界/长连/未胜/非法/平局
python tests/test_identity.py         # 身份三层分离、统计不混淆、推理接口
python tests/test_robustness.py       # 判罚归责 + 句柄所有权 + 进程树清理 + checkpoint 诚实性
python -m gomoku_arena.server --port 8090 &
python tests/test_api.py              # 接口、回放、批量统计与交换先手
# 端到端（各自拉起独立端口的平台实例，互不干扰）
python tests/test_dual_models.py      # A：两份不同模型权重，20 局
python tests/test_stability.py        # B：100 局稳定性 + 资源泄漏 + 崩溃清理
python tests/test_batch_spectate.py   # C：批量实时观战 + 回放隔离
```

`test_robustness.py` 里最关键的一项是**进程树清理**：测试引擎会派生一个 600 秒的孙进程再崩溃，
平台必须在判负的同时把孙进程也收拾掉。实现方式是 Windows **Job Object（kill-on-close）**，并把
子进程以挂起态启动、先入 Job 再恢复 —— 这样它之后派生的任何后代都必然在同一个 Job 里，
关掉 Job 句柄即被内核整棵终结。之所以不能只靠 `taskkill /T` 或按父 pid 递归枚举：父进程一旦
退出，后代会被操作系统 reparent，那时无论哪种方式都找不到它们了。

句柄管理用 `JobHandle` 对象自带的所有权标记，而不是"全局已关闭句柄值集合"：Windows 的句柄值
会被系统复用（测试里连续 6 次创建 Job 拿到的是同一个值），按值记账会误判"已经关过了"从而漏关。
测试会实际制造句柄值复用并检查每个 Job 都被各自关闭。POSIX 下则是让子进程
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
     {"participants":{"black":{"engine":"trained","pid":"dsh","checkpoint":"checkpoints/a.pt",
                               "version":"v3","move_delay_ms":0},
                      "white":{"engine":"trained","pid":"codex","checkpoint":"checkpoints/b.pt"}},
      "move_timeout_ms":1000,"game_timeout_ms":120000,"seed":42}
     （兼容旧格式 {"black":"greedy","white":"random","checkpoints":{...}}）
POST /api/batches           开始一批（逐局交换先手），请求体同上外加 "games": 20
GET  /api/batches/<id>      批量进度与分色统计（current_game_id 指向当前观战局）
DELETE /api/batches/<id>    中止批量对战
```

返回的状态里 `running` 表示是否进行中，字段形态在"进行中"与"已结束"两种情况下是统一的；
批量进行中的对局会带 `batch:{batch_id,index,of}`，前端据此显示"第 N/M 局"。

## 已知边界

- **浏览器用轮询**而非 WebSocket。一步棋的思考时间在百毫秒级以上，轮询的延迟和开销可以忽略；
  历史回放期间轮询自动降频到 5s，但仍持续，以便实时对局不中断。
- **观战窗口取决于引擎速度**：若引擎每步只要 1ms，整局会在两次轮询之间结束，浏览器只能看到
  结果而看不到逐手过程。真实模型推理远慢于此；验收测试用 `move_delay_ms` 人为放慢测试适配器
  来构造可观察窗口（该字段只对测试适配器 `ckpt_test` 生效）。
- **只在本机 spawn 子进程**，不做远程/容器引擎接入。
- **批量对战是顺序执行的**（一局接一局），不做并行；也没有 Elo / 数据库 / 淘汰赛。
- 棋盘尺寸固定 15×15，不做可配置（避免前后端与协议三处各写一份）。
- 内存中最多保留 `MAX_IN_MEMORY_RECORDS`（5000）局用于统计；更早的对局仍在 `games.jsonl` 里，
  `/api/stats` 会同时报告 `records_on_disk` 与 `records_in_memory`。
- **POSIX 下的进程树清理逻辑未在本机验证**（当前环境为 Windows，走 Job Object 路径）。
- **浏览器 UI 未经真实浏览器自动化测试**：Canvas 绘制与交互由 HTML 结构检查 + 全套 HTTP 接口
  行为覆盖，未用浏览器驱动点击验证。
- Windows 控制台可能把中文显示成乱码（GBK 代码页），但 `games.jsonl` 与接口一律是正确的 UTF-8。
