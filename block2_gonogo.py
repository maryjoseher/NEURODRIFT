from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Block2Config:
    go_char: str = "X"
    nogo_char: str = "Y"
    go_probability: float = 0.80

    fix_sec: float = 0.4
    stim_min_sec: float = 0.3
    stim_max_sec: float = 0.5
    response_window_sec: float = 1.2
    iti_min_sec: float = 0.9
    iti_max_sec: float = 1.3

    practice_trials: int = 30
    default_main_trials: int = 300
    countdown_sec: int = 3

    response_key: str = "space"


class Block2GoNoGoTask:
    def __init__(self, cfg: Optional[Block2Config] = None):
        self.cfg = cfg or Block2Config()
        self.reset()

    def reset(self):
        self.phase = "IDLE"  # IDLE / INSTRUCTIONS / PRACTICE / PRACTICE_END / COUNTDOWN / MAIN / FINISHED
        self.main_trials = self.cfg.default_main_trials
        self.practice_trials = self.cfg.practice_trials

        self.sequence_practice: List[Dict[str, Any]] = []
        self.sequence_main: List[Dict[str, Any]] = []

        self.current_trial_kind = "practice"
        self.trial_index = -1
        self.trial_stage = "IDLE"  # FIX / STIM / ITI
        self.stage_t0 = None

        self.current_stim = ""
        self.current_trial_type = ""
        self.current_is_go = False
        self.current_stim_duration_sec = 0.0
        self.current_iti_sec = 0.0

        self.accept_response = False
        self.response_captured = False
        self.response_key = ""
        self.response_valid = False
        self.pressed = False
        self.t_response_s = float("nan")
        self.rt_ms = float("nan")

        self.countdown_value = self.cfg.countdown_sec
        self.finished = False

        self.correct = 0
        self.incorrect = 0
        self.hit = 0
        self.miss = 0
        self.correct_rejection = 0
        self.commission_error = 0
        self.invalid_key = 0

        self.behavior_rows: List[Dict[str, Any]] = []
        self.marker_rows: List[Dict[str, Any]] = []

        self.block_t0 = None
        self.t_block_start_s = float("nan")
        self.t_block_end_s = float("nan")

        self.current_row: Dict[str, Any] = {}
        self.stim_abs_t0 = None

    # =========================================================
    # CONFIG
    # =========================================================
    def set_main_trials(self, n_trials: int):
        self.main_trials = int(max(1, n_trials))

    def set_practice_trials(self, n_trials: int):
        self.practice_trials = int(max(0, n_trials))

    # =========================================================
    # INICIO
    # =========================================================
    def start(self):
        self.reset()
        self.phase = "INSTRUCTIONS"
        self.sequence_practice = self.generate_trials(self.practice_trials)
        self.sequence_main = self.generate_trials(self.main_trials)

    def rel_time(self) -> float:
        if self.block_t0 is None:
            return float("nan")
        return time.perf_counter() - self.block_t0

    def start_main_clock(self):
        self.block_t0 = time.perf_counter()
        self.t_block_start_s = 0.0

    def add_marker(self, event: str, trial_index: Optional[int], phase: str, value: str = ""):
        self.marker_rows.append({
            "time_s": self.rel_time(),
            "event": event,
            "trial_index": trial_index if trial_index is not None else -1,
            "phase": phase,
            "value": value,
        })

    # =========================================================
    # SECUENCIA
    # =========================================================
    def generate_trials(self, n: int) -> List[Dict[str, Any]]:
        seq = []
        for _ in range(n):
            is_go = random.random() < self.cfg.go_probability
            stim = self.cfg.go_char if is_go else self.cfg.nogo_char
            seq.append({
                "stimulus": stim,
                "trial_type": "GO" if is_go else "NOGO",
                "is_go": is_go,
            })
        return seq

    # =========================================================
    # TEXTOS
    # =========================================================
    def instruction_text(self) -> str:
        return (
            f"BLOQUE 2 - GO / NO-GO\n\n"
            f"Presiona ESPACIO cuando veas '{self.cfg.go_char}'.\n"
            f"NO presiones nada cuando veas '{self.cfg.nogo_char}'.\n\n"
            f"Primero harás una práctica.\n"
            f"Responde lo más rápido y preciso posible.\n\n"
            f"Pulsa ESPACIO para continuar."
        )

    def practice_end_text(self) -> str:
        return (
            "Práctica terminada.\n\n"
            "Pulsa ESPACIO para comenzar la cuenta regresiva."
        )

    def finished_text(self) -> str:
        return "Bloque terminado."

    # =========================================================
    # FASES
    # =========================================================
    def begin_practice(self):
        self.reset_counts()
        self.phase = "PRACTICE"
        self.current_trial_kind = "practice"
        self.trial_index = -1
        self._advance_to_next_trial()

    def begin_countdown(self):
        self.phase = "COUNTDOWN"
        self.stage_t0 = time.perf_counter()

    def begin_main(self):
        self.reset_counts()
        self.phase = "MAIN"
        self.current_trial_kind = "main"
        self.trial_index = -1
        self.start_main_clock()
        self.add_marker("BLOCK2_START", None, "main", "")
        self._advance_to_next_trial()

    def reset_counts(self):
        self.correct = 0
        self.incorrect = 0
        self.hit = 0
        self.miss = 0
        self.correct_rejection = 0
        self.commission_error = 0
        self.invalid_key = 0

    # =========================================================
    # TECLAS
    # =========================================================
    def handle_key(self, key: str):
        key = key.lower().strip()

        if self.phase == "INSTRUCTIONS" and key == "space":
            if self.practice_trials > 0:
                self.begin_practice()
            else:
                self.begin_countdown()
            return

        if self.phase == "PRACTICE_END" and key == "space":
            self.begin_countdown()
            return

        if self.accept_response and not self.response_captured:
            self.pressed = True
            self.response_key = key
            self.t_response_s = self.rel_time()
            self.rt_ms = (time.perf_counter() - self.stim_abs_t0) * 1000.0
            self.response_valid = (key == self.cfg.response_key)
            self.response_captured = True

    # =========================================================
    # TRIALS
    # =========================================================
    def _advance_to_next_trial(self):
        seq = self.sequence_practice if self.current_trial_kind == "practice" else self.sequence_main

        self.trial_index += 1
        if self.trial_index >= len(seq):
            if self.current_trial_kind == "practice":
                self.phase = "PRACTICE_END"
            else:
                self.phase = "FINISHED"
                self.finished = True
                self.t_block_end_s = self.rel_time()
                self.add_marker("BLOCK2_END", None, "main", "")
            return

        trial = seq[self.trial_index]
        self.current_stim = trial["stimulus"]
        self.current_trial_type = trial["trial_type"]
        self.current_is_go = trial["is_go"]
        self.current_stim_duration_sec = random.uniform(self.cfg.stim_min_sec, self.cfg.stim_max_sec)
        self.current_iti_sec = random.uniform(self.cfg.iti_min_sec, self.cfg.iti_max_sec)

        self.trial_stage = "FIX"
        self.stage_t0 = time.perf_counter()

        self.accept_response = False
        self.response_captured = False
        self.response_key = ""
        self.response_valid = False
        self.pressed = False
        self.t_response_s = float("nan")
        self.rt_ms = float("nan")
        self.stim_abs_t0 = None

        self.current_row = {
            "trial_index": self.trial_index + 1,
            "phase": self.current_trial_kind,
            "stimulus": self.current_stim,
            "trial_type": self.current_trial_type,
            "is_go": bool(self.current_is_go),
            "correct_answer": "space" if self.current_is_go else "none",
            "response_key": "",
            "pressed": False,
            "response_valid": False,
            "outcome": "",
            "is_correct": False,
            "rt_ms": float("nan"),
            "stim_duration_ms": self.current_stim_duration_sec * 1000.0,
            "iti_duration_ms": self.current_iti_sec * 1000.0,
            "t_fix_onset_s": self.rel_time(),
            "t_stim_onset_s": float("nan"),
            "t_stim_offset_s": float("nan"),
            "t_response_s": float("nan"),
            "t_iti_onset_s": float("nan"),
            "t_trial_end_s": float("nan"),
            "t_block_start_s": self.t_block_start_s,
            "t_block_end_s": float("nan"),
            "trial_duration_ms": float("nan"),
        }

        if self.current_trial_kind == "main":
            self.add_marker("TRIAL_START", self.trial_index + 1, "main", "")

    # =========================================================
    # UPDATE
    # =========================================================
    def update(self) -> Dict[str, Any]:
        now = time.perf_counter()

        if self.phase == "INSTRUCTIONS":
            return {"screen": "instruction", "text": self.instruction_text()}

        if self.phase == "PRACTICE_END":
            return {"screen": "instruction", "text": self.practice_end_text()}

        if self.phase == "COUNTDOWN":
            elapsed = now - self.stage_t0
            remaining = self.cfg.countdown_sec - int(elapsed)

            if elapsed >= self.cfg.countdown_sec:
                self.begin_main()
                return self.update()

            return {"screen": "stimulus", "text": str(max(1, remaining))}

        if self.phase == "FINISHED":
            return {"screen": "instruction", "text": self.finished_text()}

        if self.phase not in ("PRACTICE", "MAIN"):
            return {"screen": "blank", "text": ""}

        elapsed = now - self.stage_t0

        if self.trial_stage == "FIX":
            if elapsed >= self.cfg.fix_sec:
                self.trial_stage = "STIM"
                self.stage_t0 = now
                self.stim_abs_t0 = now
                self.current_row["t_stim_onset_s"] = self.rel_time()
                self.accept_response = True

                if self.current_trial_kind == "main":
                    self.add_marker("STIM_ON", self.trial_index + 1, "main", self.current_stim)

            return {"screen": "fixation", "text": "+"}

        if self.trial_stage == "STIM":
            if elapsed >= self.current_stim_duration_sec:
                self.trial_stage = "RESP"
                self.stage_t0 = now
                self.current_row["t_stim_offset_s"] = self.rel_time()
                if self.current_trial_kind == "main":
                    self.add_marker("STIM_OFF", self.trial_index + 1, "main", "")
                return {"screen": "blank", "text": ""}

            return {"screen": "stimulus", "text": self.current_stim}

        if self.trial_stage == "RESP":
            time_since_stim = now - self.stim_abs_t0
            if self.response_captured or time_since_stim >= self.cfg.response_window_sec:
                self.accept_response = False
                self.trial_stage = "ITI"
                self.stage_t0 = now
                self.current_row["t_iti_onset_s"] = self.rel_time()

                outcome, is_correct = self._resolve_outcome()

                self.current_row["response_key"] = self.response_key
                self.current_row["pressed"] = bool(self.pressed)
                self.current_row["response_valid"] = bool(self.response_valid)
                self.current_row["outcome"] = outcome
                self.current_row["is_correct"] = bool(is_correct)
                self.current_row["rt_ms"] = self.rt_ms
                self.current_row["t_response_s"] = self.t_response_s
                self.current_row["t_trial_end_s"] = self.rel_time() + self.current_iti_sec
                self.current_row["trial_duration_ms"] = (
                    self.current_row["t_trial_end_s"] - self.current_row["t_fix_onset_s"]
                ) * 1000.0

                if self.current_trial_kind == "main":
                    if self.pressed:
                        self.add_marker("RESPONSE", self.trial_index + 1, "main", self.response_key)
                    self.add_marker("ITI_ON", self.trial_index + 1, "main", "")
                    self.add_marker("TRIAL_END", self.trial_index + 1, "main", outcome)

                self.behavior_rows.append(self.current_row)
                return {"screen": "blank", "text": ""}

            return {"screen": "blank", "text": ""}

        if self.trial_stage == "ITI":
            if elapsed >= self.current_iti_sec:
                self._advance_to_next_trial()
                return self.update()
            return {"screen": "blank", "text": ""}

        return {"screen": "blank", "text": ""}

    # =========================================================
    # RESULTADO DEL TRIAL
    # =========================================================
    def _resolve_outcome(self):
        if self.current_is_go:
            if not self.pressed:
                outcome = "miss"
                is_correct = False
                self.miss += 1
                self.incorrect += 1
            elif self.response_valid:
                outcome = "hit"
                is_correct = True
                self.hit += 1
                self.correct += 1
            else:
                outcome = "invalid_key"
                is_correct = False
                self.invalid_key += 1
                self.incorrect += 1
        else:
            if not self.pressed:
                outcome = "correct_rejection"
                is_correct = True
                self.correct_rejection += 1
                self.correct += 1
            elif self.response_valid:
                outcome = "commission_error"
                is_correct = False
                self.commission_error += 1
                self.incorrect += 1
            else:
                outcome = "invalid_key"
                is_correct = False
                self.invalid_key += 1
                self.incorrect += 1

        return outcome, is_correct

    # =========================================================
    # ESTADO PARA OPERADOR
    # =========================================================
    def get_operator_status(self) -> Dict[str, Any]:
        total = len(self.sequence_main) if self.phase == "MAIN" else len(self.sequence_practice)
        current = max(0, self.trial_index + 1)

        return {
            "phase": self.phase,
            "trial_current": current,
            "trial_total": total,
            "stimulus": self.current_stim,
            "trial_type": self.current_trial_type,
            "response": self.response_key,
            "last_rt_ms": self.rt_ms,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "hit": self.hit,
            "miss": self.miss,
            "correct_rejection": self.correct_rejection,
            "commission_error": self.commission_error,
            "invalid_key": self.invalid_key,
        }

    # =========================================================
    # RESUMEN
    # =========================================================
    def get_summary_rows(self) -> List[Dict[str, Any]]:
        main_rows = [r for r in self.behavior_rows if r["phase"] == "main"]
        n_main = len(main_rows)
        n_go = sum(1 for r in main_rows if r["trial_type"] == "GO")
        n_nogo = sum(1 for r in main_rows if r["trial_type"] == "NOGO")

        hits = sum(1 for r in main_rows if r["outcome"] == "hit")
        misses = sum(1 for r in main_rows if r["outcome"] == "miss")
        correct_rej = sum(1 for r in main_rows if r["outcome"] == "correct_rejection")
        commission_err = sum(1 for r in main_rows if r["outcome"] == "commission_error")
        invalid_key = sum(1 for r in main_rows if r["outcome"] == "invalid_key")

        acc = 100.0 * sum(1 for r in main_rows if r["is_correct"]) / max(n_main, 1)

        go_rt_vals = [r["rt_ms"] for r in main_rows if r["outcome"] == "hit" and r["rt_ms"] == r["rt_ms"]]
        if len(go_rt_vals) > 0:
            rt_mean = sum(go_rt_vals) / len(go_rt_vals)
            rt_sd = (sum((x - rt_mean) ** 2 for x in go_rt_vals) / len(go_rt_vals)) ** 0.5
        else:
            rt_mean = float("nan")
            rt_sd = float("nan")

        omission_rate = 100.0 * misses / max(n_go, 1)
        commission_rate = 100.0 * commission_err / max(n_nogo, 1)

        return [
            {"metric": "n_main_trials", "value": n_main},
            {"metric": "n_go_trials", "value": n_go},
            {"metric": "n_nogo_trials", "value": n_nogo},
            {"metric": "hits_go", "value": hits},
            {"metric": "misses_go", "value": misses},
            {"metric": "correct_rejections_nogo", "value": correct_rej},
            {"metric": "commission_errors_nogo", "value": commission_err},
            {"metric": "invalid_key_trials", "value": invalid_key},
            {"metric": "accuracy_percent", "value": acc},
            {"metric": "go_rt_mean_ms", "value": rt_mean},
            {"metric": "go_rt_sd_ms", "value": rt_sd},
            {"metric": "omission_rate_percent", "value": omission_rate},
            {"metric": "commission_rate_percent", "value": commission_rate},
            {"metric": "t_block_end_s", "value": self.t_block_end_s},
        ]