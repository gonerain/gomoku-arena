"""身份模型与推理接口测试（不需要服务端）。

    python tests/test_identity.py

覆盖本轮 P0 的三类问题：
  * 参赛方身份 / 引擎程序 / 模型权重 三层分离，同名引擎不同权重不再互相覆盖；
  * 统计以参赛方 pid 为键，不同模型版本不混合；
  * py_trained 的 _infer_implemented() 检测正确，GameView.in_bounds() 存在，
    且不加入任何静默启发式回退。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engines"))

from gomoku_arena.batch import Batch, summarise  # noqa: E402
from gomoku_arena.match import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    Match,
    MatchConfig,
    Participant,
    check_distinct,
)
from gomoku_arena.server import ENGINES, participant_for  # noqa: E402

results: list[tuple[bool, str, str]] = []
not_verified: list[str] = []

MODEL_A = ROOT / "tests" / "models" / "model_a" / "meta.json"
MODEL_B = ROOT / "tests" / "models" / "model_b" / "meta.json"


def check(ok: bool, name: str, detail: str = "") -> None:
    ok = bool(ok)
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：inspect.getsource 依赖模块可解析，否则会抛 OSError
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------- 参赛方身份 ----------

def test_participant_separates_engine_and_checkpoint() -> None:
    dsh = participant_for("trained", pid="dsh", name="DSH", version="v3",
                          checkpoint=str(MODEL_A))
    codex = participant_for("trained", pid="codex", name="Codex", version="v7",
                            checkpoint=str(MODEL_B))
    check(dsh.engine == codex.engine == "trained",
          "双方可以是同一个引擎适配器", f"engine={dsh.engine}")
    check(dsh.checkpoint != codex.checkpoint,
          "双方的 checkpoint 各自独立，未被覆盖",
          f"{Path(dsh.checkpoint).parent.name} vs {Path(codex.checkpoint).parent.name}")
    check(dsh.pid != codex.pid, "双方 pid 不同", f"{dsh.pid} vs {codex.pid}")
    check(dsh.spec["env"]["GOMOKU_MODEL"] == str(MODEL_A)
          and codex.spec["env"]["GOMOKU_MODEL"] == str(MODEL_B),
          "各自启动环境变量里的权重路径正确",
          f"{dsh.spec['env']['GOMOKU_MODEL']} / {codex.spec['env']['GOMOKU_MODEL']}")


def test_checkpoint_follows_participant_not_color() -> None:
    """交换先后手时，权重必须跟着参赛方走，不能跟颜色绑定。"""
    dsh = participant_for("trained", pid="dsh", version="v3", checkpoint=str(MODEL_A))
    codex = participant_for("trained", pid="codex", version="v7", checkpoint=str(MODEL_B))

    seen: list[tuple] = []

    class FakeMatch:
        def __init__(self, black, white, config, batch=None, **kwargs):
            seen.append((black.pid, black.checkpoint, white.pid, white.checkpoint))
            self.game_id = f"g-{len(seen)}"
            self.batch = batch

        def play(self):
            return {
                "game_id": self.game_id, "moves": [], "duration_ms": 0,
                "black": {"pid": "dsh", "name": "dsh", "checkpoint": "x"},
                "white": {"pid": "codex", "name": "codex", "checkpoint": "y"},
                "result": {"winner": "black", "reason": "five", "line": None},
            }

    batch = Batch(dsh, codex, 2, MatchConfig(),
                  lambda b, w, cfg, info: FakeMatch(b, w, cfg, info))
    batch.run()
    check(len(seen) == 2, "批量对战编排了 2 局", f"{len(seen)} 局")
    check(seen[0] == ("dsh", str(MODEL_A), "codex", str(MODEL_B)),
          "第 1 局：dsh 执黑带 A 权重，codex 执白带 B 权重", str(seen[0]))


def test_checkpoint_follows_participant_on_swap() -> None:
    dsh = participant_for("trained", pid="dsh", version="v3", checkpoint=str(MODEL_A))
    codex = participant_for("trained", pid="codex", version="v7", checkpoint=str(MODEL_B))
    seen: list[tuple] = []

    class FakeMatch:
        def __init__(self, black, white, config, batch=None, **kwargs):
            seen.append((black.pid, black.checkpoint, white.pid, white.checkpoint))
            self.game_id = f"g-{len(seen)}"
            self.batch = batch

        def play(self):
            return {
                "game_id": self.game_id, "moves": [], "duration_ms": 0,
                "black": {"pid": "dsh", "name": "dsh", "checkpoint": "x"},
                "white": {"pid": "codex", "name": "codex", "checkpoint": "y"},
                "result": {"winner": "black", "reason": "five", "line": None},
            }

    batch = Batch(dsh, codex, 4, MatchConfig(),
                  lambda b, w, cfg, info: FakeMatch(b, w, cfg, info))
    batch.run()

    check(seen[0][0] == "dsh" and seen[0][1] == str(MODEL_A),
          "第 1 局 dsh 执黑且带自己的权重", str(seen[0]))
    check(seen[1][0] == "codex" and seen[1][1] == str(MODEL_B),
          "第 2 局交换后 codex 执黑仍带自己的权重", str(seen[1]))
    check(seen[1][2] == "dsh" and seen[1][3] == str(MODEL_A),
          "第 2 局 dsh 转为执白，权重依然是 A", str(seen[1]))


def test_same_participant_rejected() -> None:
    a = participant_for("trained", pid="dsh", checkpoint=str(MODEL_A))
    b = participant_for("trained", pid="dsh", checkpoint=str(MODEL_B))
    try:
        check_distinct(a, b)
        check(False, "相同 pid 的双方被拒绝", "竟然通过了")
    except ValueError as exc:
        check(True, "相同 pid 的双方被拒绝", str(exc)[:60])

    c = participant_for("trained", pid="x", checkpoint=str(MODEL_A))
    d = participant_for("trained", pid="y", checkpoint=str(MODEL_A))
    try:
        check_distinct(c, d)
        check(False, "模型接入点同一权重、无版本差异时被拒绝", "竟然通过了")
    except ValueError as exc:
        check(True, "模型接入点同一权重、无版本差异时被拒绝", str(exc)[:60])

    # 基线引擎之间同配置对局是有意义的自检，不应被拦
    try:
        check_distinct(participant_for("greedy", pid="b1"), participant_for("greedy", pid="b2"))
        check(True, "基线引擎之间同配置自检不被拦截")
    except ValueError as exc:
        check(False, "基线引擎之间同配置自检不被拦截", str(exc)[:60])


def test_no_checkpoint_leaks_from_environment() -> None:
    """不给 checkpoint 时，不能继承环境里遗留的 GOMOKU_MODEL。"""
    import os

    old = os.environ.get("GOMOKU_MODEL")
    os.environ["GOMOKU_MODEL"] = "checkpoints/should-not-be-used.pt"
    try:
        p = participant_for("trained", pid="fresh")
        check(p.checkpoint == DEFAULT_CHECKPOINT and p.spec["env"]["GOMOKU_MODEL"] == "",
              "未指定 checkpoint 时不会继承环境变量", f"checkpoint={p.checkpoint}")
    finally:
        if old is None:
            os.environ.pop("GOMOKU_MODEL", None)
        else:
            os.environ["GOMOKU_MODEL"] = old


# ---------- 统计身份 ----------

def _record(game_id, black_pid, black_name, black_ckpt, white_pid, white_name, white_ckpt,
            winner, reason="five"):
    return {
        "game_id": game_id,
        "duration_ms": 1,
        "moves": [{"n": 1, "color": "black", "x": 7, "y": 7, "ms": 0}],
        "black": {"pid": black_pid, "name": black_name, "engine": "trained",
                  "version": "v1", "checkpoint": black_ckpt},
        "white": {"pid": white_pid, "name": white_name, "engine": "trained",
                  "version": "v2", "checkpoint": white_ckpt},
        "result": {"winner": winner, "reason": reason, "line": None},
    }


def test_stats_keyed_by_pid_not_name() -> None:
    """两个参赛方自报相同引擎名时，统计仍必须分开。"""
    records = [
        _record("g1", "dsh", "same-name", "a.pt", "codex", "same-name", "b.pt", "black"),
        _record("g2", "codex", "same-name", "b.pt", "dsh", "same-name", "a.pt", "white"),
    ]
    summary = summarise("b-test", "dsh", "codex", records, finished=True)
    check(set(summary["totals"]) == {"dsh", "codex"},
          "统计以参赛方 pid 为键，即使引擎自报名称相同也不混淆",
          f"keys={sorted(summary['totals'])}")
    check(summary["totals"]["dsh"]["win"] == 2 and summary["totals"]["codex"]["loss"] == 2,
          "自报同名时胜负归属仍然正确",
          f"dsh={summary['totals']['dsh']['win']}胜 codex={summary['totals']['codex']['loss']}负")
    check(summary["sides"]["dsh"]["checkpoint"] == "a.pt"
          and summary["sides"]["codex"]["checkpoint"] == "b.pt",
          "checkpoint 跟随参赛方记录", f"{summary['sides']['dsh']['checkpoint']}")


def test_different_versions_not_mixed() -> None:
    records = [
        _record("g1", "dsh", "DSH", "v3.pt", "codex", "Codex", "v7.pt", "black"),
        _record("g2", "dsh-v4", "DSH", "v4.pt", "codex", "Codex", "v7.pt", "white"),
    ]
    summary = summarise("b-test", "dsh", "codex", records, finished=True)
    check("dsh" in summary["totals"] and "dsh-v4" in summary["totals"],
          "不同模型版本（不同 pid）的成绩分开统计",
          f"keys={sorted(summary['totals'])}")
    check(summary["totals"]["dsh"]["checkpoint"] == "v3.pt",
          "版本 v3 的 checkpoint 未被 v4 覆盖",
          summary["totals"]["dsh"]["checkpoint"])


def test_mixed_checkpoint_detected() -> None:
    """同一个 pid 在不同局里带了不同权重 → 必须被标记出来。"""
    records = [
        _record("g1", "dsh", "DSH", "v3.pt", "codex", "Codex", "v7.pt", "black"),
        _record("g2", "dsh", "DSH", "v9.pt", "codex", "Codex", "v7.pt", "white"),
    ]
    summary = summarise("b-test", "dsh", "codex", records, finished=True)
    mixed = summary["sides"]["dsh"].get("mixed_checkpoints")
    check("v9.pt" in (mixed or []),
          "同一 pid 混用不同权重时被显式标记", f"mixed={mixed}")
    check(not summary["sides"]["codex"].get("mixed_checkpoints"),
          "未混用的参赛方不会被误标", str(summary["sides"]["codex"].get("mixed_checkpoints")))


def test_infer_implemented_detection() -> None:
    module = load_module(ROOT / "engines" / "py_trained" / "engine.py", "_pt_template")
    check(module.Policy._infer_implemented() is False,
          "模板占位时 _infer_implemented() 返回 False（旧实现在此永远为 True）")


def test_infer_implemented_accepts_real_code() -> None:
    """把 _infer 的函数体换成真实实现后，必须被识别为已实现。"""
    source = (ROOT / "engines" / "py_trained" / "engine.py").read_text(encoding="utf-8")
    # 用真实实现替换整个 _infer 函数体（按函数边界切分，避免匹配到其它文本）
    head, marker, _tail = source.partition("    def _infer(self, view: GameView, ctx: Context):")
    check(bool(marker), "定位到模板里的 _infer 定义")
    real_body = (
        marker + "\n"
        '        """已实现（测试注入）。"""\n'
        "        logits = self.model(view, ctx)\n"
        "        n = view.board_size\n"
        "        action = int(logits)\n"
        "        return (action % n, action // n)\n"
    )
    # 保留 _infer 之后的其余代码（act 与模块尾部）
    rest = _tail.split("\n    def act(", 1)
    rebuilt = head + real_body + "\n    def act(" + rest[1]

    # 生成文件必须与真实引擎同目录：引擎内部按自身路径推导 engines/ 位置
    engine_dir = ROOT / "engines" / "py_trained"
    path = engine_dir / "_generated_test_engine.py"
    try:
        path.write_text(rebuilt, encoding="utf-8")
        module = load_module(path, "_pt_real")
        check(module.Policy._infer_implemented() is True,
              "真实实现的 _infer 被识别为已实现")
    finally:
        path.unlink(missing_ok=True)


def test_gameview_in_bounds() -> None:
    from engine_loop import GameView

    view = GameView(15, [["black", 7, 7]])
    check(view.in_bounds(0, 0) and view.in_bounds(14, 14), "GameView.in_bounds 接受边界内坐标")
    check(not view.in_bounds(-1, 0) and not view.in_bounds(0, 15),
          "GameView.in_bounds 拒绝越界坐标（真实推理落点校验不再报错）")
    check(view.is_empty(0, 0) and not view.is_empty(7, 7),
          "GameView.is_empty 正确反映占位情况")


def test_external_command_paths() -> None:
    """外部引擎要以完整命令形式接入（常装在别的 Python 环境里）。

    回归点：platform 曾给外部路径又套了一层 sys.executable，变成
    `python.exe python.exe engine.py`；也曾忘记把 checkpoint 写进 GOMOKU_MODEL。
    """
    from gomoku_arena.cli import build_external_participant, build_participant
    from gomoku_arena.server import participant_for

    interpreter = sys.executable
    script = ROOT / "engines" / "py_random" / "engine.py"
    command = f"{interpreter} {script}"
    checkpoint = str(MODEL_A)

    via_cli = build_participant(command, "dsh", checkpoint, "v1")
    check(via_cli.spec["cmd"] == [interpreter, str(script)],
          "完整命令不再被重复套一层解释器", str(via_cli.spec["cmd"]))
    check(via_cli.spec["env"].get("GOMOKU_MODEL") == checkpoint,
          "完整命令形式也会把 checkpoint 注入 GOMOKU_MODEL")
    check(via_cli.pid == "dsh" and via_cli.name == "dsh",
          "完整命令形式的身份显示为参赛方 id", f"{via_cli.pid}/{via_cli.name}")

    via_config = participant_for(command, pid="codex", version="v9", checkpoint=checkpoint)
    check(via_config.spec["cmd"] == [interpreter, str(script)]
          and via_config.spec["env"].get("GOMOKU_MODEL") == checkpoint,
          "配置文件路径（participant_for）同样支持完整命令",
          f"cmd={via_config.spec['cmd']}")

    via_bare = build_external_participant(str(script), pid="bare")
    check(via_bare.spec["cmd"] == [sys.executable, str(script)],
          "只给 .py 路径时用当前解释器执行", str(via_bare.spec["cmd"]))


def test_no_silent_fallback_in_trained() -> None:
    """py_trained 必须在权重失败时显式退出，不能静默退回启发式。"""
    source = (ROOT / "engines" / "py_trained" / "engine.py").read_text(encoding="utf-8")
    check("greedy_choose(view, ctx)" not in source,
          "py_trained 中不存在把启发式当默认策略的回退分支")
    check("_fail(" in source and "EXIT_MODEL_FAILED" in source,
          "py_trained 使用显式失败路径（退出码 2 + 原因）")
    check("ALLOW_STUB" not in source,
          "不存在绕过模型检查的开关（避免把未实现当真模型跑）")


def main() -> int:
    print("=== 身份模型与推理接口测试 ===\n")
    for fn in (
        test_participant_separates_engine_and_checkpoint,
        test_checkpoint_follows_participant_not_color,
        test_checkpoint_follows_participant_on_swap,
        test_same_participant_rejected,
        test_no_checkpoint_leaks_from_environment,
        test_stats_keyed_by_pid_not_name,
        test_different_versions_not_mixed,
        test_mixed_checkpoint_detected,
        test_infer_implemented_detection,
        test_infer_implemented_accepts_real_code,
        test_gameview_in_bounds,
        test_external_command_paths,
        test_no_silent_fallback_in_trained,
    ):
        try:
            fn()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            check(False, f"{fn.__name__} 执行异常", f"{type(exc).__name__}: {exc}")

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
