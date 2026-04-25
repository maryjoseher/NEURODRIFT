from __future__ import annotations

import json
import os
import shutil
from typing import Dict, Any, Iterable

import pandas as pd

from config import AppConfig
from session_accumulator import SessionAccumulator


class SessionManager:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.reset_runtime()

    def reset_runtime(self):
        self.subject_id = "S01"
        self.operator = "Investigador"
        self.record_name = "Sesion_Atencional_01"
        self.session_id = ""
        self.events = []

    def start(self, subject_id: str, operator: str, record_name: str, session_id: str) -> None:
        self.subject_id = subject_id.strip() or "S01"
        self.operator = operator.strip() or "Investigador"
        self.record_name = record_name.strip() or "Sesion_Atencional_01"
        self.session_id = session_id
        self.events = []

    def add_event(self, time_s: float, event_name: str) -> None:
        self.events.append({
            "session_id": self.session_id,
            "time_s": float(time_s),
            "event_name": event_name,
        })

    def save_full_session(
        self,
        acc: SessionAccumulator,
        baseline: Dict[str, Any],
        base_dir: str | None = None,
        extra_files: Iterable[str] | None = None,
    ) -> str:
        root_dir = base_dir if base_dir else self.cfg.output_dir
        os.makedirs(root_dir, exist_ok=True)

        subject_dir = os.path.join(root_dir, self._sanitize(acc.subject_id))
        os.makedirs(subject_dir, exist_ok=True)

        session_folder_name = f"{self._sanitize(acc.record_name)}_{acc.session_id}"
        session_dir = os.path.join(subject_dir, session_folder_name)
        os.makedirs(session_dir, exist_ok=True)

        meta = {
            "session_id": acc.session_id,
            "subject_id": acc.subject_id,
            "operator": acc.operator,
            "record_name": acc.record_name,
            "block_order": acc.block_order,
            "baseline": baseline,
            "output_dir": session_dir,
        }

        with open(os.path.join(session_dir, "session_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        pd.DataFrame(acc.events_global).to_csv(os.path.join(session_dir, "events.csv"), index=False)
        pd.DataFrame(acc.all_ocular_rows).to_csv(os.path.join(session_dir, "ocular_all_samples.csv"), index=False)
        pd.DataFrame(acc.all_trials_long).to_csv(os.path.join(session_dir, "trials_long.csv"), index=False)
        pd.DataFrame(acc.blocks_summary_rows).to_csv(os.path.join(session_dir, "blocks_summary.csv"), index=False)

        if getattr(acc, "all_physio_feature_rows", None):
            pd.DataFrame(acc.all_physio_feature_rows).to_csv(os.path.join(session_dir, "physio_features_all.csv"), index=False)
        if getattr(acc, "all_eda_event_rows", None):
            pd.DataFrame(acc.all_eda_event_rows).to_csv(os.path.join(session_dir, "eda_events_all.csv"), index=False)

        for block_name, bd in acc.blocks_data.items():
            block_dir = os.path.join(session_dir, self._normalize_block_folder(block_name))
            os.makedirs(block_dir, exist_ok=True)
            if bd["ocular_rows"]:
                pd.DataFrame(bd["ocular_rows"]).to_csv(os.path.join(block_dir, "ocular_samples.csv"), index=False)
            if bd["behavior_rows"]:
                pd.DataFrame(bd["behavior_rows"]).to_csv(os.path.join(block_dir, "behavior.csv"), index=False)
            if bd["markers_rows"]:
                pd.DataFrame(bd["markers_rows"]).to_csv(os.path.join(block_dir, "markers.csv"), index=False)
            if bd["summary_rows"]:
                pd.DataFrame(bd["summary_rows"]).to_csv(os.path.join(block_dir, "summary.csv"), index=False)
            if bd.get("physio_feature_rows"):
                pd.DataFrame(bd["physio_feature_rows"]).to_csv(os.path.join(block_dir, "physio_features.csv"), index=False)
            if bd.get("eda_event_rows"):
                pd.DataFrame(bd["eda_event_rows"]).to_csv(os.path.join(block_dir, "eda_events.csv"), index=False)

        if extra_files:
            raw_dir = os.path.join(session_dir, "raw_physio")
            os.makedirs(raw_dir, exist_ok=True)
            for src in extra_files:
                if not src or not os.path.isfile(src):
                    continue
                dst = os.path.join(raw_dir, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)

        return session_dir

    def _normalize_block_folder(self, name: str) -> str:
        name = name.strip().lower().replace(" ", "_").replace("ó", "o").replace("í", "i")
        return self._sanitize(name)

    def _sanitize(self, txt: str) -> str:
        txt = str(txt).strip()
        keep = []
        for ch in txt:
            if ch.isalnum() or ch in ("_", "-", "."):
                keep.append(ch)
            elif ch.isspace():
                keep.append("_")
        out = "".join(keep)
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("._") or "unnamed"
