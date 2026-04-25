from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Block3Config:
    fix_sec: float = 0.50
    stim_sec: float = 2.00
    response_max_sec: float = 1.20
    iti_sec: float = 0.35
    countdown_sec: int = 3

    load_low: tuple[int, int] = (3, 4)
    load_med: tuple[int, int] = (5, 6)
    load_high: tuple[int, int] = (7, 8)

    practice_trials: int = 6
    default_main_trials: int = 30

    margin_x: tuple[float, float] = (0.12, 0.88)
    margin_y: tuple[float, float] = (0.18, 0.82)
    min_dist: float = 0.10


class Block3WorkingMemoryTask:
    def __init__(self, cfg: Optional[Block3Config] = None):
        self.cfg = cfg or Block3Config()
        self.reset()

    def reset(self):
        self.phase = "IDLE"
        self.practice_trials = self.cfg.practice_trials
        self.main_trials = self.cfg.default_main_trials

        self.practice_seq: List[Dict[str, Any]] = []
        self.main_seq: List[Dict[str, Any]] = []

        self.current_trial_kind = "practice"
        self.trial_index = -1
        self.trial_stage = "IDLE"   # FIX / STIM / QUESTION / ITI
        self.stage_t0 = None
        self.question_t0 = None

        self.current_trial = None
        self.finished = False

        self.response_key = ""
        self.response_valid = False
        self.response_captured = False
        self.pressed = False
        self.rt_ms = float("nan")

        self.correct = 0
        self.incorrect_key = 0
        self.omission = 0

        self.behavior_rows: List[Dict[str, Any]] = []
        self.marker_rows: List[Dict[str, Any]] = []

        self.block_t0 = None
        self.paused_at = None
        self.current_row: Dict[str, Any] = {}

    def set_main_trials(self, n_trials: int):
        self.main_trials = int(max(1, n_trials))

    def set_practice_trials(self, n_trials: int):
        self.practice_trials = int(max(0, n_trials))

    def start(self):
        self.reset()
        self.phase = "INSTRUCTIONS"
        self.practice_seq = self.generate_practice_trials(self.practice_trials)
        self.main_seq = self.generate_main_trials(self.main_trials)

    def pause(self):
        if self.paused_at is None:
            self.paused_at = time.perf_counter()

    def resume(self):
        if self.paused_at is None:
            return
        dt = time.perf_counter() - self.paused_at
        if self.stage_t0 is not None:
            self.stage_t0 += dt
        if self.question_t0 is not None:
            self.question_t0 += dt
        if self.block_t0 is not None:
            self.block_t0 += dt
        self.paused_at = None

    def rel_time(self) -> float:
        if self.block_t0 is None:
            return float("nan")
        return time.perf_counter() - self.block_t0

    def start_main_clock(self):
        self.block_t0 = time.perf_counter()

    def add_marker(self, event: str, trial_index: int | None, value: str = ""):
        self.marker_rows.append({
            "time_s": self.rel_time(),
            "event": event,
            "trial_index": -1 if trial_index is None else trial_index,
            "phase": self.current_trial_kind,
            "value": value,
        })

    def instruction_text(self) -> str:
        return (
            "BLOQUE 3 - MEMORIA OPERATIVA\n\n"
            "Aparecerán varios números entre 1 y 9.\n"
            "Algunos pueden repetirse.\n\n"
            "Debes decidir si el número que apareció más veces fue:\n\n"
            "1 = IMPAR\n"
            "2 = PAR\n\n"
            "Pulsa ESPACIO para continuar."
        )

    def practice_end_text(self) -> str:
        return "Práctica terminada.\n\nPulsa ESPACIO para continuar."

    def finished_text(self) -> str:
        return "Bloque terminado."

    def begin_practice(self):
        self.phase = "PRACTICE"
        self.current_trial_kind = "practice"
        self.trial_index = -1
        self.correct = 0
        self.incorrect_key = 0
        self.omission = 0
        self._advance_to_next_trial()

    def begin_countdown(self):
        self.phase = "COUNTDOWN"
        self.stage_t0 = time.perf_counter()

    def begin_main(self):
        self.phase = "MAIN"
        self.current_trial_kind = "main"
        self.trial_index = -1
        self.correct = 0
        self.incorrect_key = 0
        self.omission = 0
        self.start_main_clock()
        self.add_marker("BLOCK3_START", None, "")
        self._advance_to_next_trial()

    def handle_key(self, key: str):
        k = key.lower().strip()

        if self.phase == "INSTRUCTIONS" and k == "space":
            if self.practice_trials > 0:
                self.begin_practice()
            else:
                self.begin_countdown()
            return

        if self.phase == "PRACTICE_END" and k == "space":
            self.begin_countdown()
            return

        if self.trial_stage == "QUESTION" and not self.response_captured:
            if k in ("1", "2", "numpad1", "numpad2"):
                if k == "numpad1":
                    k = "1"
                elif k == "numpad2":
                    k = "2"
                self.response_key = k
                self.response_valid = True
                self.response_captured = True
                self.pressed = True
                self.rt_ms = (time.perf_counter() - self.question_t0) * 1000.0

    def generate_practice_trials(self, n: int) -> List[Dict[str, Any]]:
        levels = ["bajo", "medio", "alto"]
        out = []
        for i in range(n):
            out.append(self.generate_trial(levels[i % 3]))
        random.shuffle(out)
        return out

    def generate_main_trials(self, n: int) -> List[Dict[str, Any]]:
        levels = [["bajo", "medio", "alto"][i % 3] for i in range(n)]
        random.shuffle(levels)
        return [self.generate_trial(lv) for lv in levels]

    def generate_trial(self, level: str) -> Dict[str, Any]:
        if level == "bajo":
            n_stim = random.randint(*self.cfg.load_low)
        elif level == "medio":
            n_stim = random.randint(*self.cfg.load_med)
        else:
            n_stim = random.randint(*self.cfg.load_high)

        while True:
            nums = [random.randint(1, 9) for _ in range(n_stim)]
            counts = {k: nums.count(k) for k in range(1, 10)}
            maxf = max(counts.values())
            winners = [k for k, v in counts.items() if v == maxf]
            if len(winners) == 1:
                target = winners[0]
                break

        answer = "2" if target % 2 == 0 else "1"
        parity = "par" if target % 2 == 0 else "impar"
        pos = self.generate_positions(n_stim)

        return {
            "level": level,
            "n_stimuli": n_stim,
            "numbers": nums,
            "target_number": target,
            "target_frequency": maxf,
            "target_parity": parity,
            "correct_answer": answer,
            "positions": pos,
        }

    def generate_positions(self, n: int):
        for _ in range(1000):
            pos = []
            ok = True
            for _j in range(n):
                placed = False
                for _ in range(500):
                    x = random.uniform(*self.cfg.margin_x)
                    y = random.uniform(*self.cfg.margin_y)
                    if not pos:
                        pos.append((x, y))
                        placed = True
                        break
                    d_ok = all((((px - x) ** 2 + (py - y) ** 2) ** 0.5) >= self.cfg.min_dist for px, py in pos)
                    if d_ok:
                        pos.append((x, y))
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok:
                return pos

        return [
            (random.uniform(*self.cfg.margin_x), random.uniform(*self.cfg.margin_y))
            for _ in range(n)
        ]

    def _advance_to_next_trial(self):
        seq = self.practice_seq if self.current_trial_kind == "practice" else self.main_seq
        self.trial_index += 1

        if self.trial_index >= len(seq):
            if self.current_trial_kind == "practice":
                self.phase = "PRACTICE_END"
            else:
                self.phase = "FINISHED"
                self.finished = True
                self.add_marker("BLOCK3_END", None, "")
            return

        self.current_trial = seq[self.trial_index]
        self.trial_stage = "FIX"
        self.stage_t0 = time.perf_counter()
        self.question_t0 = None

        self.response_key = ""
        self.response_valid = False
        self.response_captured = False
        self.pressed = False
        self.rt_ms = float("nan")

        self.current_row = {
            "phase": self.current_trial_kind,
            "trial_index": self.trial_index + 1,
            "load_level": self.current_trial["level"],
            "n_stimuli": self.current_trial["n_stimuli"],
            "stimuli_numbers": str(self.current_trial["numbers"]),
            "positions": str(self.current_trial["positions"]),
            "target_number": self.current_trial["target_number"],
            "target_frequency": self.current_trial["target_frequency"],
            "target_parity": self.current_trial["target_parity"],
            "correct_answer": self.current_trial["correct_answer"],
            "response_key": "",
            "pressed": False,
            "response_valid": False,
            "outcome": "",
            "is_correct": False,
            "is_incorrect_key": False,
            "is_omission": False,
            "rt_ms": float("nan"),
            "t_fix_onset_s": self.rel_time(),
            "t_stim_onset_s": float("nan"),
            "t_stim_offset_s": float("nan"),
            "t_question_onset_s": float("nan"),
            "t_response_s": float("nan"),
            "t_trial_end_s": float("nan"),
        }

    def update(self) -> Dict[str, Any]:
        now = time.perf_counter()

        if self.phase == "INSTRUCTIONS":
            return {"screen": "instruction", "text": self.instruction_text()}

        if self.phase == "PRACTICE_END":
            return {"screen": "instruction", "text": self.practice_end_text()}

        if self.phase == "COUNTDOWN":
            elapsed = now - self.stage_t0
            if elapsed >= self.cfg.countdown_sec:
                self.begin_main()
                return self.update()
            return {"screen": "stimulus", "text": str(max(1, self.cfg.countdown_sec - int(elapsed)))}

        if self.phase == "FINISHED":
            return {"screen": "instruction", "text": self.finished_text()}

        if self.phase not in ("PRACTICE", "MAIN"):
            return {"screen": "blank", "text": ""}

        elapsed = now - self.stage_t0

        if self.trial_stage == "FIX":
            if elapsed >= self.cfg.fix_sec:
                self.trial_stage = "STIM"
                self.stage_t0 = now
                self.current_row["t_stim_onset_s"] = self.rel_time()
            return {"screen": "fixation", "text": "+"}

        if self.trial_stage == "STIM":
            if elapsed >= self.cfg.stim_sec:
                self.trial_stage = "QUESTION"
                self.stage_t0 = now
                self.question_t0 = now
                self.current_row["t_stim_offset_s"] = self.rel_time()
                self.current_row["t_question_onset_s"] = self.rel_time()
                return {
                    "screen": "instruction",
                    "text": "¿El número que apareció más veces fue?\n\n1 = IMPAR\n2 = PAR"
                }

            items = []
            for num, (x, y) in zip(self.current_trial["numbers"], self.current_trial["positions"]):
                items.append({"text": str(num), "x": x, "y": y})
            return {"screen": "multi_stimuli", "items": items}

        if self.trial_stage == "QUESTION":
            if self.response_captured or (now - self.question_t0) >= self.cfg.response_max_sec:
                if not self.pressed:
                    outcome = "omission"
                    is_correct = False
                    self.omission += 1
                    incorrect_key = False
                elif self.response_key == self.current_trial["correct_answer"]:
                    outcome = "correct"
                    is_correct = True
                    incorrect_key = False
                    self.correct += 1
                else:
                    outcome = "incorrect_key"
                    is_correct = False
                    incorrect_key = True
                    self.incorrect_key += 1

                self.current_row["response_key"] = self.response_key
                self.current_row["pressed"] = self.pressed
                self.current_row["response_valid"] = self.response_valid
                self.current_row["outcome"] = outcome
                self.current_row["is_correct"] = is_correct
                self.current_row["is_incorrect_key"] = incorrect_key
                self.current_row["is_omission"] = (outcome == "omission")
                self.current_row["rt_ms"] = self.rt_ms
                self.current_row["t_response_s"] = self.rel_time() if self.pressed else float("nan")

                self.trial_stage = "ITI"
                self.stage_t0 = now
                self.current_row["t_trial_end_s"] = self.rel_time() + self.cfg.iti_sec
                self.behavior_rows.append(self.current_row)
                return {"screen": "blank", "text": ""}

            return {
                "screen": "instruction",
                "text": "¿El número que apareció más veces fue?\n\n1 = IMPAR\n2 = PAR"
            }

        if self.trial_stage == "ITI":
            if elapsed >= self.cfg.iti_sec:
                self._advance_to_next_trial()
                return self.update()
            return {"screen": "blank", "text": ""}

        return {"screen": "blank", "text": ""}

    def get_operator_status(self) -> Dict[str, Any]:
        total = len(self.main_seq) if self.phase == "MAIN" else len(self.practice_seq)
        current = max(0, self.trial_index + 1)
        return {
            "phase": self.phase,
            "trial_current": current,
            "trial_total": total,
            "load_level": "" if self.current_trial is None else self.current_trial["level"],
            "correct": self.correct,
            "incorrect_key": self.incorrect_key,
            "omission": self.omission,
        }

    def get_summary_rows(self) -> List[Dict[str, Any]]:
        main_rows = [r for r in self.behavior_rows if r["phase"] == "main"]
        n = len(main_rows)
        n_correct = sum(1 for r in main_rows if r["is_correct"])
        n_incorrect = sum(1 for r in main_rows if r["is_incorrect_key"])
        n_omission = sum(1 for r in main_rows if r["is_omission"])
        acc = 100.0 * n_correct / max(n, 1)
        rt_vals = [r["rt_ms"] for r in main_rows if r["is_correct"] and r["rt_ms"] == r["rt_ms"]]
        rt_mean = sum(rt_vals) / len(rt_vals) if rt_vals else float("nan")

        return [
            {"metric": "n_main_trials", "value": n},
            {"metric": "correct", "value": n_correct},
            {"metric": "incorrect_key", "value": n_incorrect},
            {"metric": "omission", "value": n_omission},
            {"metric": "accuracy_percent", "value": acc},
            {"metric": "mean_rt_ms_correct", "value": rt_mean},
        ]