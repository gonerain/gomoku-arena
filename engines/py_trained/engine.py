"""RL 模型接入点 —— 平台与训练系统之间唯一的耦合点。

平台不关心你用什么算法（PPO / AlphaZero / DQN / 进化策略都行），
它只要求这个进程做到两件事：
    1. 从 stdin 收 {"type":"turn","moves":[...]}；
    2. 往 stdout 发 {"type":"move","x":..,"y":..}。
于是 DSH 和 Codex 可以各自独立训练，训练代码、框架、checkpoint 格式互不干涉，
最后都以"一个能说协议的子进程"的形态进入竞技场。

重要：本引擎绝不用启发式策略冒充 RL 模型。
  * 没有指定 checkpoint（GOMOKU_MODEL 为空）时，本进程会带
    meta.baseline=true 启动，明确标注"这不是 RL 模型，只是在演示协议"，
    此时它用 py_greedy 的启发式走子，仅用于跑通流程。
  * 指定了 checkpoint 后，如果文件不存在、加载失败或推理失败，本进程会
    打印原因并以退出码 2 结束，不会静默退回启发式继续比赛。
    平台会据此判该方 start_failed 负 —— 这是刻意的：宁可比赛失败，
    也不能让"未训练的模型"伪装成训练结果进入统计。

接入步骤（训练侧）：
  1. 实现 _load()：把 checkpoint 读成一个可调用对象；
  2. 实现 _infer()：用 state_tensor() 编码观测、用 legal_mask() 屏蔽非法点、返回 (x, y)；
  3. 训练侧的观测编码必须与 state_tensor() 一致，否则推理结果无意义。

用法：
    GOMOKU_MODEL=checkpoints/ppo_v3.pt python -m ...            # 真实模型
    python engines/py_trained/engine.py                         # 演示模式（非 RL）
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import NoReturn

ENGINES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINES_DIR))

from engine_loop import Context, GameView, run  # noqa: E402

EXIT_MODEL_FAILED = 2

CHECKPOINT = os.environ.get("GOMOKU_MODEL") or ""
# 仅在自检/演示时使用：允许占位 _infer() 参与，不用于任何真实比赛
ALLOW_STUB = os.environ.get("GOMOKU_ALLOW_STUB") == "1"

stderr = sys.stderr


def _fail(reason: str) -> NoReturn:
    """模型相关的失败必须显式退出，绝不退回启发式。"""
    stderr.write(
        f"[trained] 无法作为 RL 模型运行：{reason}\n"
        f"[trained] 该进程将以退出码 {EXIT_MODEL_FAILED} 结束，平台会判其 start_failed 负。\n"
        f"[trained] 如果只是想跑通协议，请改用引擎 'greedy'（它是启发式基线，不冒充 RL 模型）。\n"
    )
    stderr.flush()
    raise SystemExit(EXIT_MODEL_FAILED)


def _load_greedy_fallback():
    """演示模式用的启发式走子。

    用文件路径加载而不是 import，因为目录名 py_greedy 不是合法模块名。
    只有在没有 checkpoint 的演示模式下才会用到它。
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "_gomoku_greedy", ENGINES_DIR / "py_greedy" / "engine.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module.choose
    except Exception as exc:
        _fail(f"演示模式加载启发式基线失败: {type(exc).__name__}: {exc}")


def state_tensor(view: GameView, ctx: Context):
    """观测编码约定（供训练侧对齐）。

    返回 shape = (4, N, N) 的 float32 数组，坐标系与协议一致：
        [0] 己方棋子
        [1] 对方棋子
        [2] 上一手落点
        [3] 当前执子方指示位（该自己走时为全 1）

    动作空间 = N*N 个格点，索引 = y * N + x。
    """
    import numpy as np

    n = view.board_size
    tensor = np.zeros((4, n, n), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            cell = view.grid[y][x]
            if cell == ctx.code:
                tensor[0, y, x] = 1.0
            elif cell == ctx.opponent_code:
                tensor[1, y, x] = 1.0
    last = view.last_move
    if last:
        tensor[2, last[1], last[0]] = 1.0
    tensor[3, :, :] = 1.0
    return tensor


def legal_mask(view: GameView):
    """动作掩码：空点为 1、已占点为 0（已展平为 N*N）。训练与推理都该用上它。"""
    import numpy as np

    n = view.board_size
    mask = np.zeros((n, n), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            if view.grid[y][x] == 0:
                mask[y, x] = 1.0
    return mask.reshape(-1)


class Policy:
    """把训练好的模型包成"输入棋盘、输出落子"的东西。"""

    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self.demo_mode = not checkpoint
        self.model = None
        if self.demo_mode:
            self._greedy = _load_greedy_fallback()
            return
        path = Path(checkpoint)
        if not path.is_file():
            _fail(f"checkpoint 不存在或不是文件: {checkpoint}")
        self.model = self._load(str(path))
        if self.model is None:
            _fail(f"_load() 未能加载模型: {checkpoint}（返回了 None）")
        if not ALLOW_STUB and not type(self)._infer_implemented():
            _fail(
                "_infer() 尚未实现（仍是接入模板的占位实现）。"
                "请在 engines/py_trained/engine.py 中实现 _load() 与 _infer() 后再指定 checkpoint。"
            )

    @classmethod
    def _infer_implemented(cls) -> bool:
        return cls._infer is not Policy._infer

    def _load(self, path: str):
        """TODO(训练侧)：加载模型并返回一个可调用对象。

        示例（PyTorch）：
            import torch
            model = MyNet(board_size=15)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model
        """
        return None

    def _infer(self, view: GameView, ctx: Context):
        """TODO(训练侧)：推理并返回 (x, y)；返回 None 视为失败。

        示例（PyTorch）：
            import torch
            with torch.no_grad():
                logits = self.model(torch.from_numpy(state_tensor(view, ctx))[None])[0]
                logits = logits.masked_fill(torch.from_numpy(legal_mask(view) == 0), -1e9)
                action = int(logits.argmax(-1))
            n = view.board_size
            return (action % n, action // n)
        """
        return None

    def act(self, view: GameView, ctx: Context) -> tuple[int, int]:
        if self.demo_mode:
            return self._greedy(view, ctx)
        try:
            move = self._infer(view, ctx)
        except Exception as exc:
            _fail(f"推理失败: {type(exc).__name__}: {exc}")
        if move is None:
            _fail("推理返回了 None（_infer() 未正确实现）")
        x, y = move
        if not isinstance(x, int) or not isinstance(y, int) or isinstance(x, bool) or isinstance(y, bool):
            _fail(f"推理返回了非整数坐标: {move!r}")
        if not view.in_bounds(x, y):
            _fail(f"推理返回了越界坐标: ({x},{y})")
        if view.grid[y][x] != 0:
            _fail(f"推理返回了已占点: ({x},{y})")
        return (x, y)


_POLICY = Policy(CHECKPOINT)

if _POLICY.demo_mode:
    stderr.write(
        "[trained] 演示模式：未指定 GOMOKU_MODEL，本进程使用启发式走子，"
        "meta.baseline=true —— 这不是 RL 模型，请不要把它的成绩当成训练结果。\n"
    )
    META = {"algo": "heuristic-demo", "baseline": True, "checkpoint": "none",
            "desc": "未指定 checkpoint 的演示模式，非 RL 模型"}
else:
    META = {"algo": "rl-checkpoint", "baseline": False, "checkpoint": CHECKPOINT,
            "desc": f"已加载 checkpoint: {CHECKPOINT}"}


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    return _POLICY.act(view, ctx)


if __name__ == "__main__":
    run(choose, name="trained", version="v0", meta=META)
