from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class SessionAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.session_started = False
        self.session_id = ""
        self.subject_id = ""
        self.operator = ""
        self.record_name = ""

        self.block_order: List[str] = []
        self.events_global: List[Dict[str, Any]] = []

        self.all_ocular_rows: List[Dict[str, Any]] = []
        self.all_trials_long: List[Dict[str, Any]] = []
        self.blocks_summary_rows: List[Dict[str, Any]] = []

        # NUEVO
        self.all_physio_feature_rows: List[Dict[str, Any]] = []
        self.all_eda_event_rows: List[Dict[str, Any]] = []

        self.blocks_data: Dict[str, Dict[str, Any]] = {}

    def start_session(self, session_id: str, subject_id: str, operator: str, record_name: str):
        self.reset()
        self.session_started = True
        self.session_id = session_id
        self.subject_id = subject_id
        self.operator = operator
        self.record_name = record_name

    def ensure_block(self, block_name: str):
        if block_name not in self.blocks_data:
            self.blocks_data[block_name] = {
                "ocular_rows": [],
                "behavior_rows": [],
                "markers_rows": [],
                "summary_rows": [],
                # NUEVO
                "physio_feature_rows": [],
                "eda_event_rows": [],
            }
            self.block_order.append(block_name)

    def append_global_event(self, event_row: Dict[str, Any]):
        self.events_global.append(dict(event_row))

    def append_block_data(
        self,
        block_name: str,
        ocular_rows: Optional[List[Dict[str, Any]]] = None,
        behavior_rows: Optional[List[Dict[str, Any]]] = None,
        markers_rows: Optional[List[Dict[str, Any]]] = None,
        summary_rows: Optional[List[Dict[str, Any]]] = None,
    ):
        self.ensure_block(block_name)
        bd = self.blocks_data[block_name]

        if ocular_rows:
            rows = [dict(r) for r in ocular_rows]
            bd["ocular_rows"].extend(rows)
            self.all_ocular_rows.extend(rows)

        if behavior_rows:
            bd["behavior_rows"].extend([dict(r) for r in behavior_rows])

        if markers_rows:
            bd["markers_rows"].extend([dict(r) for r in markers_rows])

        if summary_rows:
            bd["summary_rows"].extend([dict(r) for r in summary_rows])

    # NUEVO
    def append_physio_block_data(
        self,
        block_name: str,
        physio_feature_rows: list[dict] | None = None,
        eda_event_rows: list[dict] | None = None,
    ):
        self.ensure_block(block_name)
        bd = self.blocks_data[block_name]

        if physio_feature_rows:
            rows = [dict(r) for r in physio_feature_rows]
            bd["physio_feature_rows"].extend(rows)
            self.all_physio_feature_rows.extend(rows)

        if eda_event_rows:
            rows = [dict(r) for r in eda_event_rows]
            bd["eda_event_rows"].extend(rows)
            self.all_eda_event_rows.extend(rows)

    def append_trials_long(self, trial_rows: List[Dict[str, Any]]):
        if trial_rows:
            self.all_trials_long.extend([dict(r) for r in trial_rows])

    def add_block_summary(self, summary_row: Dict[str, Any]):
        self.blocks_summary_rows.append(dict(summary_row))

    def build_trials_long_from_behavior(self):
        out = []
        for block_name, bd in self.blocks_data.items():
            for row in bd["behavior_rows"]:
                rr = dict(row)
                rr["block_name"] = block_name
                out.append(rr)
        self.all_trials_long = out

    def summarize_ocular_block(self, block_name: str) -> Dict[str, Any]:
        self.ensure_block(block_name)
        rows = self.blocks_data[block_name]["ocular_rows"]

        if not rows:
            return {
                "session_id": self.session_id,
                "subject_id": self.subject_id,
                "block_name": block_name,
                "duration_s": np.nan,
                "valid_pct": np.nan,
                "blink_count": np.nan,
                "blink_per_min": np.nan,
                "perclos_mean": np.nan,
                "var_iris_mean": np.nan,
                "drift_mean": np.nan,
                "focus_pct": np.nan,
                "distraction_pct": np.nan,
                "fatigue_pct": np.nan,
                "invalid_pct": np.nan,
            }

        df = pd.DataFrame(rows)

        duration_s = float(df["time_s"].max() - df["time_s"].min()) if "time_s" in df else np.nan
        valid_pct = float(df["valid_pct"].mean()) if "valid_pct" in df else np.nan
        blink_count = int(df["blink_count_accum"].max()) if "blink_count_accum" in df else np.nan
        blink_per_min = float((blink_count / duration_s) * 60.0) if duration_s and duration_s > 0 else np.nan
        perclos_mean = float(df["perclos"].mean()) if "perclos" in df else np.nan
        var_iris_mean = float(df["var_iris_clean"].mean()) if "var_iris_clean" in df else np.nan
        drift_mean = float(df["drift"].mean()) if "drift" in df else np.nan

        if "state" in df:
            focus_pct = 100.0 * float((df["state"] == "FOCO").mean())
            distraction_pct = 100.0 * float((df["state"] == "DISTRACCION").mean())
            fatigue_pct = 100.0 * float((df["state"] == "FATIGA").mean())
            invalid_pct = 100.0 * float((df["state"] == "NO_CONFIABLE").mean())
        else:
            focus_pct = distraction_pct = fatigue_pct = invalid_pct = np.nan

        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "block_name": block_name,
            "duration_s": duration_s,
            "valid_pct": valid_pct,
            "blink_count": blink_count,
            "blink_per_min": blink_per_min,
            "perclos_mean": perclos_mean,
            "var_iris_mean": var_iris_mean,
            "drift_mean": drift_mean,
            "focus_pct": focus_pct,
            "distraction_pct": distraction_pct,
            "fatigue_pct": fatigue_pct,
            "invalid_pct": invalid_pct,
        }