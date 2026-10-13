"""RL 模型接入模板 —— 这是平台与训练系统之间唯一的耦合点。

平台不关心你用什么算法（PPO / AlphaZero / DQN / 进化策略都行），
它只要求这个进程做到两件事：
    1. 从 stdin 收 {"type":"turn","moves":[...]}；
    2. 往 stdout 发 {"type":"move","x":..,"y":..}。
于是 DSH 和 Codex 可以各自独立训练，训练代码、框架、checkpoint 格式互不干涉，
最后都以"一个能说协议的子进程"的形态进入竞技场。

state_tensor() 给了观测编码的一个默认约定。训练侧只要用同一个编码，
就能把训练时的网络直接搬来推理；想换编码就改这个函数，平台侧不用动。

用法：
    python engines/py_trained/engine.py                     # 无 checkpoint 时退化为贪心
    GOMOKU_MODEL=path/to/model.pt python engines/py_trained/engine.py
"""

import importlib.util
import os
import sys
from pathlib import Path

ENGINES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINES_DIR))

from engine_loop import Context, GameView, run  # noqa: E402

# checkpoint 用环境变量指定，方便在 server.py 的 ENGINES 里按模型版本配置
CHECKPOINT = os.environ.get("GOMOKU_MODEL") or None


def _load_greedy_fallback():
    """没有模型时退回贪心基线，保证这个文件开箱即可运行、协议立刻可验证。

    用文件路径加载而不是 import，因为目录名 py_greedy 不是合法模块名。
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "_gomoku_greedy", ENGINES_DIR / "py_greedy" / "engine.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module.choose
    except Exception as exc:  # pragma: no cover
        print(f"[trained] 贪心基线加载失败: {exc}", file=sys.stderr)
        return None


greedy_choose = _load_greedy_fallback()


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

    def __init__(self, checkpoint: str | None):
        self.checkpoint = checkpoint
        self.model = None
        if checkpoint:
            path = Path(checkpoint)
            if path.exists():
                self.model = self._load(str(path))
            else:
                print(f"[trained] checkpoint 不存在: {checkpoint}，退化为贪心", file=sys.stderr)

    def _load(self, path: str):
        """TODO(训练侧)：在这里加载你自己的模型，返回一个可调用的对象。

        示例（PyTorch）：
            import torch
            model = MyNet(board_size=15)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model
        """
        print(f"[trained] _load() 尚未实现，忽略 {path}，退化为贪心", file=sys.stderr)
        return None

    def _infer(self, view: GameView, ctx: Context):
        """TODO(训练侧)：在这里做推理，返回 (x, y)。返回 None 则退回贪心。

        示例（PyTorch）：
            import torch
            with torch.no_grad():
                logits = self.model(torch.from_numpy(state_tensor(view, ctx))[None])[0]
                logits = logits.masked_fill(
                    torch.from_numpy(legal_mask(view) == 0), -1e9
                )
                action = int(logits.argmax(-1))
            n = view.board_size
            return (action % n, action // n)
        """
        return None

    def act(self, view: GameView, ctx: Context) -> tuple[int, int]:
        if self.model is not None:
            move = self._infer(view, ctx)
            if move is not None:
                return move
        if greedy_choose is None:
            raise RuntimeError("既没有模型，也没有可用的贪心基线")
        return greedy_choose(view, ctx)


_POLICY = Policy(CHECKPOINT)


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    return _POLICY.act(view, ctx)


if __name__ == "__main__":
    run(
        choose,
        name="trained",
        version="v0",
        meta={"algo": "rl-checkpoint", "checkpoint": CHECKPOINT or "none"},
    )
