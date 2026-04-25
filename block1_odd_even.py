from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Block1Config:
    fix_sec: float = 0.5
    stim_sec: float = 0.5
    response_extra_sec: float = 0.7
    iti_min_sec: float = 0.8
    iti_max_sec: float = 1.2
    practice_trials: int = 10
    default_main_trials: int = 180
    countdown_sec: int = 3

    @property
    def response_total_sec(self) -> float:
        return self.stim_sec + self.response_extra_sec


class Block1OddEvenTask:
    def __init__(self, cfg: Optional[Block1Config] = None):
        self.cfg = cfg or Block1Config()
        self.reset()

    def reset(self):
        self.phase = "IDLE"
        self.practice_trials = self.cfg.practice_trials
        self.main_trials = self.cfg.default_main_trials

        self.sequence_practice: List[int] = []
        self.sequence_main: List[int] = []

        self.trial_index = -1
        self.trial_stage = "IDLE"
        self.stage_t0 = None

        self.current_digit: Optional[int] = None
        self.current_trial_kind = "practice"

        self.accept_response = False
        self.response_captured = False
        self.response_key = ""
        self.response_label = "none"
        self.rt_sec = float("nan")
        self.stim_t0 = None

        self.finished = False
        self.correct_count = 0
        self.incorrect_count = 0
        self.omission_count = 0

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
        self.sequence_practice = self.generate_sequence_balanced(self.practice_trials)
        self.sequence_main = self.generate_sequence_balanced(self.main_trials)

    def pause(self):
        if self.paused_at is None:
            self.paused_at = time.perf_counter()

    def resume(self):
        if self.paused_at is None:
            return
        dt = time.perf_counter() - self.paused_at
        if self.stage_t0 is not None:
            self.stage_t0 += dt
        if self.stim_t0 is not None:
            self.stim_t0 += dt
        if self.block_t0 is not None:
            self.block_t0 += dt
        self.paused_at = None

    def start_main_clock(self):
        self.block_t0 = time.perf_counter()

    def rel_time(self) -> float:
        if self.block_t0 is None:
            return float("nan")
        return time.perf_counter() - self.block_t0

    def add_marker(self, event: str, trial_index: Optional[int], value: str = ""):
        self.marker_rows.append({
            "time_s": self.rel_time(),
            "event": event,
            "trial_index": trial_index if trial_index is not None else -1,
            "value": value,
        })

    def generate_sequence_balanced(self, n: int) -> List[int]:
        base_digits = list(range(1, 10))
        seq = [random.choice(base_digits) for _ in range(n)]

        for i in range(3, n):
            p1 = seq[i - 1] % 2 == 0
            p2 = seq[i - 2] % 2 == 0
            p3 = seq[i - 3] % 2 == 0
            if p1 == p2 == p3:
                need_even = not p1
                candidates = [d for d in base_digits if (d % 2 == 0) == need_even]
                seq[i] = random.choice(candidates)

        return seq

    def instruction_text(self) -> str:
        return (
            "BLOQUE 1 - PARIDAD\n\n"
            "Vas a ver números del 1 al 9.\n"
            "Tu tarea es decidir si el número es IMPAR o PAR.\n\n"
            "1 = IMPAR\n"
            "2 = PAR\n\n"
            "Pulsa ESPACIO para continuar."
        )

    def practice_end_text(self) -> str:
        return "Práctica terminada.\n\nPulsa ESPACIO para continuar."

    def begin_practice(self):
        self.phase = "PRACTICE"
        self.current_trial_kind = "practice"
        self.trial_index = -1
        self._advance_to_next_trial()

    def begin_countdown(self):
        self.phase = "COUNTDOWN"
        self.stage_t0 = time.perf_counter()

    def begin_main(self):
        self.phase = "MAIN"
        self.current_trial_kind = "main"
        self.trial_index = -1
        self.correct_count = 0
        self.incorrect_count = 0
        self.omission_count = 0
        self.start_main_clock()
        self.add_marker("BLOCK1_START", None, "")
        self._advance_to_next_trial()

    def handle_key(self, key: str):
        k = key.lower()

        if self.phase == "INSTRUCTIONS" and k == "space":
            if self.practice_trials > 0:
                self.begin_practice()
            else:
                self.begin_countdown()
            return

        if self.phase == "PRACTICE_END" and k == "space":
            self.begin_countdown()
            return

        if self.accept_response and not self.response_captured:
            if k in ("1", "numpad1"):
                self.response_captured = True
                self.response_key = "1"
                self.response_label = "odd"
                self.rt_sec = time.perf_counter() - self.stim_t0
            elif k in ("2", "numpad2"):
                self.response_captured = True
                self.response_key = "2"
                self.response_label = "even"
                self.rt_sec = time.perf_counter() - self.stim_t0

    def _advance_to_next_trial(self):
        seq = self.sequence_practice if self.current_trial_kind == "practice" else self.sequence_main

        self.trial_index += 1
        if self.trial_index >= len(seq):
            if self.current_trial_kind == "practice":
                self.phase = "PRACTICE_END"
            else:
                self.phase = "FINISHED"
                self.finished = True
                self.add_marker("BLOCK1_END", None, "")
            return

        self.current_digit = seq[self.trial_index]
        self.trial_stage = "FIX"
        self.stage_t0 = time.perf_counter()

        self.accept_response = False
        self.response_captured = False
        self.response_key = ""
        self.response_label = "none"
        self.rt_sec = float("nan")
        self.stim_t0 = None

        self.current_row = {
            "phase": self.current_trial_kind,
            "trial_index": self.trial_index + 1,
            "stimulus": str(self.current_digit),
            "stimulus_numeric": int(self.current_digit),
            "condition": "even" if self.current_digit % 2 == 0 else "odd",
            "correct_answer": "even" if self.current_digit % 2 == 0 else "odd",
            "response_key": "none",
            "response_label": "none",
            "response_type": "none",
            "is_correct": False,
            "was_omission": False,
            "iti_ms": 0.0,
            "rt_ms": float("nan"),
            "t_fix_onset_s": self.rel_time(),
            "t_stim_onset_s": float("nan"),
            "t_stim_offset_s": float("nan"),
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
            return {"screen": "instruction", "text": "Bloque terminado."}

        if self.phase not in ("PRACTICE", "MAIN"):
            return {"screen": "blank", "text": ""}

        elapsed = now - self.stage_t0

        if self.trial_stage == "FIX":
            if elapsed >= self.cfg.fix_sec:
                self.trial_stage = "STIM"
                self.stage_t0 = now
                self.stim_t0 = now
                self.accept_response = True
                self.current_row["t_stim_onset_s"] = self.rel_time()
            return {"screen": "fixation", "text": "+"}

        if self.trial_stage == "STIM":
            if elapsed >= self.cfg.stim_sec:
                self.trial_stage = "RESP"
                self.stage_t0 = now
                self.current_row["t_stim_offset_s"] = self.rel_time()
                return {"screen": "blank", "text": ""}
            return {"screen": "stimulus", "text": str(self.current_digit)}

        if self.trial_stage == "RESP":
            total_from_stim = now - self.stim_t0
            if self.response_captured or total_from_stim >= self.cfg.response_total_sec:
                self.accept_response = False

                if self.response_captured:
                    self.current_row["response_key"] = self.response_key
                    self.current_row["response_label"] = self.response_label
                    self.current_row["t_response_s"] = self.rel_time()
                    self.current_row["rt_ms"] = self.rt_sec * 1000.0
                    is_correct = self.response_label == self.current_row["correct_answer"]
                    self.current_row["is_correct"] = bool(is_correct)
                    self.current_row["response_type"] = "correct" if is_correct else "incorrect"
                    if self.current_trial_kind == "main":
                        if is_correct:
                            self.correct_count += 1
                        else:
                            self.incorrect_count += 1
                else:
                    self.current_row["response_type"] = "omission"
                    self.current_row["was_omission"] = True
                    if self.current_trial_kind == "main":
                        self.omission_count += 1

                self.trial_stage = "ITI"
                self.stage_t0 = now
                self.current_iti = random.uniform(self.cfg.iti_min_sec, self.cfg.iti_max_sec)
                self.current_row["iti_ms"] = self.current_iti * 1000.0
                self.current_row["t_trial_end_s"] = self.rel_time()
                self.behavior_rows.append(self.current_row)
                return {"screen": "blank", "text": ""}

            return {"screen": "blank", "text": ""}

        if self.trial_stage == "ITI":
            if elapsed >= self.current_iti:
                self._advance_to_next_trial()
                return self.update()
            return {"screen": "blank", "text": ""}

        return {"screen": "blank", "text": ""}

    def get_operator_status(self) -> Dict[str, Any]:
        total = len(self.sequence_main) if self.phase == "MAIN" else len(self.sequence_practice)
        current = max(0, self.trial_index + 1)
        return {
            "phase": self.phase,
            "trial_current": current,
            "trial_total": total,
            "correct": self.correct_count,
            "incorrect": self.incorrect_count,
            "omission": self.omission_count,
        }

    def get_summary_rows(self) -> List[Dict[str, Any]]:
        main_rows = [r for r in self.behavior_rows if r["phase"] == "main"]
        n_main = len(main_rows)
        correct = sum(1 for r in main_rows if r["is_correct"])
        incorrect = sum(1 for r in main_rows if r["response_type"] == "incorrect")
        omissions = sum(1 for r in main_rows if r["was_omission"])
        acc = 100.0 * correct / max(n_main, 1)
        rt_vals = [r["rt_ms"] for r in main_rows if math.isfinite(r["rt_ms"])]
        rt_mean = sum(rt_vals) / len(rt_vals) if rt_vals else float("nan")
        rt_sd = (sum((x - rt_mean) ** 2 for x in rt_vals) / len(rt_vals)) ** 0.5 if len(rt_vals) > 1 else float("nan")
        return [
            {"metric": "n_main_trials", "value": n_main},
            {"metric": "correct", "value": correct},
            {"metric": "incorrect", "value": incorrect},
            {"metric": "omissions", "value": omissions},
            {"metric": "accuracy_percent", "value": acc},
            {"metric": "rt_mean_ms", "value": rt_mean},
            {"metric": "rt_sd_ms", "value": rt_sd},
        ]