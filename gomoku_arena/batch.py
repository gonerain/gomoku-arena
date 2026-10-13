"""批量对战：同一对参赛方连续下 N 局，逐局交换先后手，并分色统计成绩。

只做最低限度的事：不引入 Elo、不建数据库、不做淘汰赛。
它解决的问题是"单局结果没有统计意义"——交换先手才能把先手优势从胜负里剥离出来。

身份规则（重要）：统计与报表一律以**参赛方 pid** 为键，而不是引擎名称。
DSH 与 Codex 可以共用同一个适配器程序（engine），只要 pid 与 checkpoint 不同；
交换先手时 checkpoint 跟随参赛方移动，绝不跟颜色绑定。
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from .match import Match, MatchConfig, Participant, check_distinct

# 归到某一方名下的败因；game_timeout / aborted / platform_error 不计入任何一方的败绩
LOSS_REASONS = ("timeout", "crash", "illegal_move", "bad_output", "protocol", "resign", "start_failed")


def new_batch_id() -> str:
    return f"b-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def _empty_side() -> dict:
    return {
        "games": 0,
        "win": 0,
        "loss": 0,
        "draw": 0,
        "timeout": 0,
        "illegal_move": 0,
        "crash": 0,
        "bad_output": 0,
        "protocol": 0,
        "resign": 0,
        "start_failed": 0,
        "game_timeout": 0,
        "aborted": 0,
        "platform_error": 0,
    }


def summarise(batch_id: str, black_pid: str, white_pid: str, records: list[dict],
              finished: bool, cancelled: bool = False) -> dict:
    """按执黑/执白分别统计，并给出总胜率。records 顺序即对局顺序。

    键是参赛方 pid；展示名与 checkpoint 从棋谱的参与者身份里取，避免"引擎自报名称"
    与"平台注册名称"不一致时统计错位。
    """
    sides: dict[str, dict] = {}

    def ensure(pid: str, info: dict) -> dict:
        if pid not in sides:
            sides[pid] = {
                "pid": pid,
                "name": info.get("name") or pid,
                "engine": info.get("engine") or pid,
                "version": info.get("version") or "v0",
                "checkpoint": info.get("checkpoint") or "none",
                "as_black": _empty_side(),
                "as_white": _empty_side(),
                "mixed_checkpoints": set(),
            }
        else:
            # 同一 pid 在不同局里若出现不同 checkpoint，说明身份被复用了，必须暴露出来
            entry = sides[pid]
            seen_ckpt = info.get("checkpoint")
            if seen_ckpt and seen_ckpt not in ("none", entry["checkpoint"]):
                entry["mixed_checkpoints"].add(seen_ckpt)
        return sides[pid]

    order: list[dict] = []
    for rec in records:
        result = rec.get("result", {})
        reason = result.get("reason")
        winner_color = result.get("winner")
        colors = {"black": rec["black"], "white": rec["white"]}
        for color in ("black", "white"):
            ensure(colors[color]["pid"], colors[color])

        winner_pid = colors[winner_color]["pid"] if winner_color else None
        order.append(
            {
                "game_id": rec.get("game_id"),
                "black": colors["black"]["name"],
                "black_pid": colors["black"]["pid"],
                "black_checkpoint": colors["black"].get("checkpoint", "none"),
                "white": colors["white"]["name"],
                "white_pid": colors["white"]["pid"],
                "white_checkpoint": colors["white"].get("checkpoint", "none"),
                "winner": colors[winner_color]["name"] if winner_color else None,
                "winner_pid": winner_pid,
                "winner_color": winner_color,
                "reason": reason,
                "moves": len(rec.get("moves", [])),
                "duration_ms": rec.get("duration_ms"),
                "batch_index": (rec.get("batch") or {}).get("index"),
            }
        )
        for color in ("black", "white"):
            info = colors[color]
            key = "as_black" if color == "black" else "as_white"
            side = sides[info["pid"]][key]
            side["games"] += 1
            if winner_color == color:
                side["win"] += 1
            elif winner_color is not None:
                side["loss"] += 1
                if reason in side:
                    side[reason] += 1
            elif reason == "draw":
                side["draw"] += 1
            elif reason in side:
                side[reason] += 1

    totals = {}
    for pid, data in sides.items():
        black_side, white_side = data["as_black"], data["as_white"]
        games = black_side["games"] + white_side["games"]
        win = black_side["win"] + white_side["win"]
        loss = black_side["loss"] + white_side["loss"]
        draw = black_side["draw"] + white_side["draw"]
        neutral = {
            key: black_side[key] + white_side[key]
            for key in ("timeout", "illegal_move", "crash", "bad_output", "protocol",
                        "resign", "start_failed", "game_timeout", "aborted", "platform_error")
        }
        totals[pid] = {
            "pid": pid,
            "name": data["name"],
            "engine": data["engine"],
            "version": data["version"],
            "checkpoint": data["checkpoint"],
            "games": games,
            "win": win,
            "loss": loss,
            "draw": draw,
            **neutral,
            "win_rate": round(win / games, 4) if games else None,
            "loss_rate": round(loss / games, 4) if games else None,
            "draw_rate": round(draw / games, 4) if games else None,
        }

    for data in sides.values():
        # set 不是 JSON 可序列化的，统一转成排序后的列表
        mixed = data.get("mixed_checkpoints")
        data["mixed_checkpoints"] = sorted(mixed) if mixed else []

    return {
        "batch_id": batch_id,
        "status": "cancelled" if cancelled else ("finished" if finished else "running"),
        "participants": {"black": black_pid, "white": white_pid},
        "requested": None,
        "played": len(records),
        "sides": sides,
        "totals": totals,
        "games": order,
        "started_at": None,
        "duration_ms": None,
    }


class Batch:
    """一串对局的编排者。逐局交换先手，逐局落盘，并在平台上注册为可实时观战的对局。"""

    def __init__(self, black: Participant, white: Participant, games: int, config: MatchConfig,
                 match_factory, on_update=None, on_record=None):
        check_distinct(black, white)
        self.batch_id = new_batch_id()
        self.black = black
        self.white = white
        self.games = max(1, int(games))
        self.config = config
        self.match_factory = match_factory  # (black_participant, white_participant, config, batch) -> Match
        self.on_update = on_update
        self.on_record = on_record           # 每局一结束就落盘，中途中断也不丢已完成的对局
        self.records: list[dict] = []
        self.cancelled = False
        self.started = time.monotonic()
        self.current_game_id: str | None = None
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def cancel(self) -> None:
        self.cancelled = True

    def _persist(self, record: dict) -> None:
        if not self.on_record:
            return
        try:
            self.on_record(record)
        except Exception as exc:  # 落盘失败不能中断整批对战，但必须留痕
            import sys

            sys.stderr.write(f"[batch] 写入棋谱失败: {exc}\n")

    def run(self) -> dict:
        for index in range(self.games):
            if self.cancelled:
                break
            # 逐局交换先后手：偶数局 black 参赛方执黑，奇数局它执白。
            # checkpoint 与身份跟着参赛方走，不跟颜色走。
            if index % 2 == 0:
                black_participant, white_participant = self.black, self.white
            else:
                black_participant, white_participant = self.white, self.black

            config = MatchConfig(
                seed=self.config.seed + index if self.config.seed is not None else None,
                board_size=self.config.board_size,
                move_timeout_ms=self.config.move_timeout_ms,
                game_timeout_ms=self.config.game_timeout_ms,
            )
            batch_info = {"batch_id": self.batch_id, "index": index + 1, "of": self.games}
            match = self.match_factory(black_participant, white_participant, config, batch_info)
            self.current_game_id = match.game_id
            if self.on_update:
                self.on_update(self.summary())
            record = match.play()
            self.records.append(record)
            self._persist(record)
            if self.on_update:
                self.on_update(self.summary())
        self.current_game_id = None

        summary = self.summary()
        summary["duration_ms"] = int((time.monotonic() - self.started) * 1000)
        return summary

    def summary(self) -> dict:
        out = summarise(
            self.batch_id, self.black.pid, self.white.pid,
            self.records, finished=len(self.records) >= self.games or self.cancelled,
            cancelled=self.cancelled,
        )
        out["requested"] = self.games
        out["started_at"] = self.started_at
        out["current_game_id"] = self.current_game_id
        out["roster"] = {"black": self.black.to_dict(), "white": self.white.to_dict()}
        return out
