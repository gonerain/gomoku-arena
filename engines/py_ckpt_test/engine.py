"""测试用模型引擎：真正做一次模型推理，但**不是** RL 模型。

用途：验证"同一个适配器分别加载两份不同权重"这条真实通路，而不依赖任何训练算法。
它刻意做成一个权重驱动的确定性策略：

    meta.json  {"name": "...", "version": "...", "policy_version": 1, "bias": 0, "prefer_center": true}

推理过程（每一步都真的读取权重、真的计算）：
    1. 从权重里取出 policy_version / bias；
    2. 对每个候选空点按权重打分：分数 = version*(x+1) + bias*(y+1) + 中心偏好；
    3. 取分数最高者。

因为它把 policy_version 直接参与打分，所以两份权重（version 不同）会产生**可观测的
不同着法序列**——这正是"两个模型没有串味"的证据。

身份标注：meta.baseline = true、algo = "test-weight-hash"，明确声明它不是经过 RL 训练的
模型，平台与界面都会把它显示为基线。

    GOMOKU_MODEL=tests/models/model_a/meta.json python engines/py_ckpt_test/engine.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # engines/

from engine_loop import Context, GameView, run  # noqa: E402

CHECKPOINT = os.environ.get("GOMOKU_MODEL") or ""
# 仅测试用：人为增加每步耗时，让观战采样有可观察的窗口（真实模型不需要）
MOVE_DELAY_MS = int(os.environ.get("GOMOKU_TEST_MOVE_DELAY_MS", "0") or 0)


def _fail(reason: str) -> None:
    sys.stderr.write(f"[ckpt-test] 无法加载权重：{reason}\n")
    sys.stderr.flush()
    raise SystemExit(2)


def load_weights(path: str) -> dict:
    if not path:
        _fail("未指定 GOMOKU_MODEL")
    file = Path(path)
    if not file.is_file():
        _fail(f"权重文件不存在: {path}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"权重不是合法 JSON: {exc}")
    for key in ("policy_version", "bias"):
        if not isinstance(data.get(key), int):
            _fail(f"权重缺少整数字段 {key!r}: {data!r}")
    return data


WEIGHTS = load_weights(CHECKPOINT)
CALLS = 0


def score(x: int, y: int, board_size: int, weights: dict) -> int:
    center = board_size // 2
    bias = weights["bias"]
    value = weights["policy_version"] * (x + 1) * 7 + bias * (y + 1) * 3
    if weights.get("prefer_center"):
        value -= (abs(x - center) + abs(y - center)) * 2
    return value


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    """真实推理：读权重、算分数、选点。没有任何启发式棋力，但确实用到了模型权重。"""
    global CALLS
    CALLS += 1
    if MOVE_DELAY_MS:
        import time

        time.sleep(MOVE_DELAY_MS / 1000.0)  # 仅测试用：放慢以便观战采样
    candidates = view.neighbors(2) or view.free_cells()
    best = max(candidates, key=lambda p: (score(p[0], p[1], view.board_size, WEIGHTS), -p[0], -p[1]))
    print(
        f"[ckpt-test] call={CALLS} policy_version={WEIGHTS['policy_version']} "
        f"bias={WEIGHTS['bias']} -> {best}",
        file=sys.stderr,
    )
    return best


if __name__ == "__main__":
    meta = {
        "algo": "test-weight-hash",
        "baseline": True,  # 明确声明：不是 RL 模型
        "checkpoint": CHECKPOINT or "none",
        "policy_version": WEIGHTS["policy_version"],
        "bias": WEIGHTS["bias"],
        "desc": "测试用权重驱动策略，用于验证双模型身份不混淆；非 RL 模型",
    }
    run(choose, name=WEIGHTS.get("name", "ckpt-test"),
        version=str(WEIGHTS.get("version", "t0")), meta=meta)
