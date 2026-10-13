"""批量对战：同一对引擎连续下 N 局，逐局交换先后手，并分色统计成绩。

只做最低限度的事：不引入 Elo、不建数据库、不做淘汰赛。
它解决的问题是"单局结果没有统计意义"——交换先手才能把先手优势从胜负里剥离出来。
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from .match import Match, MatchConfig

# 归到某一方名下的败因；game_timeout / aborted / platform_error 不计入任何一方的败绩
LOSS_REASONS = ("timeout", "crash", "illegal_move", "bad_output", "protocol", "resign", "start_failed")

# 棋谱里 result.reason -> 统计分类
REASON_CLASS = {
    "five": "win",
    "draw": "draw",
    "game_timeout": "game_timeout",
    "platform_error": "platform_error",
    "aborted": "aborted",
}


def new_batch_id() -> str:
    return f"b-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def summarise(batch_id: str, black_engine: str, white_engine: str,
              records: list[dict], finished: bool, cancelled: bool = False) -> dict:
    """按执黑/执白分别统计，并给出总胜率。records 顺序即对局顺序。"""
    sides = {
        black_engine: {"as_black": _empty_side(), "as_white": _empty_side()},
        white_engine: {"as_black": _empty_side(), "as_white": _empty_side()},
    }
    order: list[dict] = []
    for rec in records:
        result = rec.get("result", {})
        reason = result.get("reason")
        winner_color = result.get("winner")
        names = {"black": rec["black"]["name"], "white": rec["white"]["name"]}
        order.append(
            {
                "game_id": rec.get("game_id"),
                "black": names["black"],
                "white": names["white"],
                "winner": names[winner_color] if winner_color else None,
                "winner_color": winner_color,
                "reason": reason,
                "moves": len(rec.get("moves", [])),
                "duration_ms": rec.get("duration_ms"),
                "black_checkpoint": rec["black"].get("checkpoint", "none"),
                "white_checkpoint": rec["white"].get("checkpoint", "none"),
            }
        )
        for color in ("black", "white"):
            name = names[color]
            key = "as_black" if color == "black" else "as_white"
            side = sides[name][key]
            side["games"] += 1
            if winner_color == color:
                side["win"] += 1
            elif winner_color is not None:
                side["loss"] += 1
                if reason in side:
                    side[reason] += 1
            elif reason == "draw":
                side["draw"] += 1
            elif reason == "game_timeout":
                side["game_timeout"] += 1
            elif reason in ("aborted", "platform_error"):
                side[reason] += 1

    totals = {}
    for name, data in sides.items():
        black_side, white_side = data["as_black"], data["as_white"]
        games = black_side["games"] + white_side["games"]
        win = black_side["win"] + white_side["win"]
        loss = black_side["loss"] + white_side["loss"]
        draw = black_side["draw"] + white_side["draw"]
        # 这些分类计入哪一方都不公平，但仍要能被看到，故逐项合计
        neutral = {
            key: black_side[key] + white_side[key]
            for key in ("timeout", "illegal_move", "crash", "bad_output", "protocol",
                        "resign", "start_failed", "game_timeout", "aborted", "platform_error")
        }
        totals[name] = {
            "games": games,
            "win": win,
            "loss": loss,
            "draw": draw,
            **neutral,
            "win_rate": round(win / games, 4) if games else None,
            "loss_rate": round(loss / games, 4) if games else None,
            "draw_rate": round(draw / games, 4) if games else None,
        }

    return {
        "batch_id": batch_id,
        "status": "cancelled" if cancelled else ("finished" if finished else "running"),
        "engines": {"black": black_engine, "white": white_engine},
        "requested": None,
        "played": len(records),
        "sides": sides,
        "totals": totals,
        "games": order,
        "started_at": None,
        "duration_ms": None,
    }


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


class Batch:
    """一串对局的编排者。逐局交换先手，逐局落盘。"""

    def __init__(self, black_engine: str, white_engine: str, games: int, config: MatchConfig,
                 spec_factory, on_record, on_update=None):
        self.batch_id = new_batch_id()
        self.black_engine = black_engine
        self.white_engine = white_engine
        self.games = max(1, int(games))
        self.config = config
        self.spec_factory = spec_factory
        self.on_record = on_record
        self.on_update = on_update
        self.records: list[dict] = []
        self.cancelled = False
        self.started = time.monotonic()

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> dict:
        for index in range(self.games):
            if self.cancelled:
                break
            # 逐局交换先后手：偶数局 DSH 执黑，奇数局 DSH(即 black_engine) 执白
            if index % 2 == 0:
                black_name, white_name = self.black_engine, self.white_engine
            else:
                black_name, white_name = self.white_engine, self.black_engine

            config = MatchConfig(
                seed=self.config.seed + index,
                board_size=self.config.board_size,
                move_timeout_ms=self.config.move_timeout_ms,
                game_timeout_ms=self.config.game_timeout_ms,
            )
            match = Match(
                self.spec_factory(black_name),
                self.spec_factory(white_name),
                config,
                batch={"batch_id": self.batch_id, "index": index + 1, "of": self.games},
            )
            record = match.play()
            self.records.append(record)
            self._persist(record)
            if self.on_update:
                self.on_update(self.summary())

        summary = self.summary()
        summary["duration_ms"] = int((time.monotonic() - self.started) * 1000)
        return summary

    def _persist(self, record: dict) -> None:
        try:
            self.on_record(record)
        except Exception as exc:  # 落盘失败不能中断整批对战，但必须留痕
            import sys

            sys.stderr.write(f"[batch] 写入棋谱失败: {exc}\n")

    def summary(self) -> dict:
        out = summarise(
            self.batch_id, self.black_engine, self.white_engine,
            self.records, finished=len(self.records) >= self.games or self.cancelled,
            cancelled=self.cancelled,
        )
        out["requested"] = self.games
        out["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return out
