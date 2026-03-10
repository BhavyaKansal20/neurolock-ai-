"""
NeuroLock AI v2 — Classroom Session Manager
Tracks emotions per student across before/during/after class phases.
Generates engagement reports with progress insights.
"""

import os
import json
import uuid
import base64
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict
from typing import Optional

from utils import database as db

EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Emotions that indicate understanding/engagement
POSITIVE_EMOTIONS  = {'happy', 'surprised'}
ENGAGED_EMOTIONS   = {'happy', 'surprised', 'neutral'}
CONFUSED_EMOTIONS  = {'fearful', 'disgusted', 'sad', 'angry'}

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports', 'reports')


class ClassroomSession:
    """
    Manages a full classroom analysis session with 3 phases:
    - BEFORE:  Baseline emotion capture
    - DURING:  Active learning tracking
    - AFTER:   Post-lesson comprehension assessment
    """

    PHASES = ['before', 'during', 'after']

    def __init__(self, name: str, teacher: str = '', subject: str = '',
                 location: str = ''):
        self.session_id  = str(uuid.uuid4())[:12]
        self.name        = name
        self.teacher     = teacher
        self.subject     = subject
        self.location    = location
        self.phase       = 'before'
        self.active      = True
        self.start_time  = datetime.now()

        # In-memory buffer: student_id → {phase → [emotion_dicts]}
        self.buffer = defaultdict(lambda: defaultdict(list))
        # Student snapshots for report: student_id → base64 image
        self.snapshots: dict = {}

        db.init_db()
        db.create_session(
            self.session_id, name, teacher, subject, location
        )
        print(f"  Session started: {self.session_id} | {name}")

    def set_phase(self, phase: str):
        if phase not in self.PHASES:
            raise ValueError(f"Phase must be one of {self.PHASES}")
        self.phase = phase
        db.update_session_phase(self.session_id, phase)
        print(f"  Phase → {phase.upper()}")

    def log(self, emotion_result: dict, student_id: str = 'unknown',
            student_name: str = 'Unknown'):
        """Log one emotion reading for a student in current phase."""
        if not self.active:
            return

        self.buffer[student_id][self.phase].append({
            'dominant':   emotion_result['dominant'],
            'confidence': emotion_result['confidence'],
            'emotions':   emotion_result['emotions'],
            'timestamp':  datetime.now().isoformat(),
        })

        db.log_emotion(
            session_id   = self.session_id,
            phase        = self.phase,
            dominant     = emotion_result['dominant'],
            confidence   = emotion_result['confidence'],
            emotions_dict= emotion_result['emotions'],
            student_id   = student_id,
            student_name = student_name,
        )

    def save_snapshot(self, student_id: str, face_bgr: np.ndarray):
        """Save a face snapshot for the report."""
        _, buf = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.snapshots[student_id] = base64.b64encode(buf).decode()

    def end(self) -> dict:
        """End session and generate full report."""
        self.active = False
        db.end_session(self.session_id)
        report = self._generate_report()
        db.save_report(self.session_id, report)

        # Save to file
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = f"report_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(REPORTS_DIR, fname), 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Session ended. Report saved: {fname}")
        return report

    def _generate_report(self) -> dict:
        session_meta = {
            'session_id': self.session_id,
            'name':       self.name,
            'teacher':    self.teacher,
            'subject':    self.subject,
            'location':   self.location,
            'start_time': self.start_time.isoformat(),
            'end_time':   datetime.now().isoformat(),
            'phase':      self.phase,
        }

        student_reports = {}
        all_student_ids = list(self.buffer.keys())

        for sid in all_student_ids:
            student_reports[sid] = self._student_report(sid)

        # Class-level summary
        class_summary = self._class_summary(student_reports)

        return {
            'meta':             session_meta,
            'class_summary':    class_summary,
            'students':         student_reports,
            'generated_at':     datetime.now().isoformat(),
        }

    def _student_report(self, student_id: str) -> dict:
        data = self.buffer[student_id]

        phase_summaries = {}
        for phase in self.PHASES:
            logs = data.get(phase, [])
            if not logs:
                phase_summaries[phase] = None
                continue
            phase_summaries[phase] = self._summarize_logs(logs)

        # Engagement score: compare before vs during vs after
        engagement = self._compute_engagement(phase_summaries)
        comprehension = self._compute_comprehension(phase_summaries)
        trend = self._compute_trend(phase_summaries)

        return {
            'student_id':    student_id,
            'snapshot_b64':  self.snapshots.get(student_id),
            'phases':        phase_summaries,
            'engagement':    engagement,
            'comprehension': comprehension,
            'trend':         trend,
            'total_readings':sum(len(data.get(p, [])) for p in self.PHASES),
            'recommendation': self._recommend(engagement, comprehension),
        }

    def _summarize_logs(self, logs: list) -> dict:
        emotions_agg = defaultdict(list)
        for log in logs:
            for emo, val in log['emotions'].items():
                emotions_agg[emo].append(val)

        avg_emotions = {emo: round(float(np.mean(vals)), 4)
                        for emo, vals in emotions_agg.items()}

        dominants = [l['dominant'] for l in logs]
        dominant_counts = {e: dominants.count(e) for e in set(dominants)}
        top_dominant = max(dominant_counts, key=dominant_counts.get)

        positive_pct = round(
            sum(1 for d in dominants if d in POSITIVE_EMOTIONS) / len(dominants) * 100, 1
        )
        engaged_pct = round(
            sum(1 for d in dominants if d in ENGAGED_EMOTIONS) / len(dominants) * 100, 1
        )

        return {
            'avg_emotions':    avg_emotions,
            'top_dominant':    top_dominant,
            'dominant_dist':   dominant_counts,
            'positive_pct':    positive_pct,
            'engaged_pct':     engaged_pct,
            'reading_count':   len(logs),
            'timeline':        [{'t': l['timestamp'], 'd': l['dominant'],
                                  'c': l['confidence']} for l in logs[-50:]],
        }

    def _compute_engagement(self, phases: dict) -> dict:
        scores = {}
        for phase, summary in phases.items():
            if summary:
                scores[phase] = summary['engaged_pct']
            else:
                scores[phase] = None

        # Overall: weighted average (during matters most)
        weights = {'before': 0.2, 'during': 0.5, 'after': 0.3}
        total, weight_sum = 0, 0
        for p, w in weights.items():
            if scores.get(p) is not None:
                total += scores[p] * w
                weight_sum += w
        overall = round(total / weight_sum, 1) if weight_sum > 0 else 0

        return {'by_phase': scores, 'overall': overall,
                'level': _score_label(overall)}

    def _compute_comprehension(self, phases: dict) -> dict:
        """
        Comprehension: compare AFTER vs BEFORE.
        If after has more positive emotions than before → understood.
        """
        before = phases.get('before')
        after  = phases.get('after')
        during = phases.get('during')

        if not before and not after and not during:
            return {'score': 0, 'level': 'insufficient_data'}

        after_pos  = (after['positive_pct']  if after  else 0)
        before_pos = (before['positive_pct'] if before else 0)
        during_eng = (during['engaged_pct']  if during else 0)

        delta = after_pos - before_pos
        score = round(min(100, max(0, 50 + delta + (during_eng * 0.3))), 1)

        return {
            'score': score,
            'level': _score_label(score),
            'delta_positive': round(delta, 1),
        }

    def _compute_trend(self, phases: dict) -> str:
        before = phases.get('before')
        after  = phases.get('after')
        if not before or not after:
            return 'insufficient_data'
        delta = after['positive_pct'] - before['positive_pct']
        if delta > 10:   return 'improved'
        if delta < -10:  return 'declined'
        return 'stable'

    def _recommend(self, engagement: dict, comprehension: dict) -> str:
        eng = engagement.get('overall', 0)
        comp = comprehension.get('score', 0)
        if eng >= 70 and comp >= 70:
            return "Excellent engagement and comprehension. Student understood the topic well."
        if eng >= 70 and comp < 50:
            return "Student was engaged but may need revision. Consider a follow-up quiz."
        if eng < 50 and comp >= 60:
            return "Moderate engagement but showed understanding. Could benefit from interactive methods."
        if eng < 40:
            return "Low engagement detected. Student may need individual attention or different teaching approach."
        return "Adequate performance. Monitor in next session for trends."

    def _class_summary(self, student_reports: dict) -> dict:
        if not student_reports:
            return {}

        total = len(student_reports)
        engagements = [r['engagement']['overall'] for r in student_reports.values()]
        comprehensions = [r['comprehension']['score'] for r in student_reports.values()]
        trends = [r['trend'] for r in student_reports.values()]

        return {
            'total_students':       total,
            'avg_engagement':       round(np.mean(engagements), 1) if engagements else 0,
            'avg_comprehension':    round(np.mean(comprehensions), 1) if comprehensions else 0,
            'trend_improved':       trends.count('improved'),
            'trend_stable':         trends.count('stable'),
            'trend_declined':       trends.count('declined'),
            'highly_engaged':       sum(1 for e in engagements if e >= 70),
            'needs_attention':      sum(1 for e in engagements if e < 40),
            'class_level':          _score_label(np.mean(engagements) if engagements else 0),
        }

    def get_live_stats(self) -> dict:
        """Quick stats for real-time dashboard."""
        student_count = len(self.buffer)
        phase_counts = defaultdict(int)
        for sid, phases in self.buffer.items():
            for phase, logs in phases.items():
                phase_counts[phase] += len(logs)
        return {
            'session_id':    self.session_id,
            'phase':         self.phase,
            'student_count': student_count,
            'log_counts':    dict(phase_counts),
            'active':        self.active,
        }


def _score_label(score: float) -> str:
    if score >= 80: return 'excellent'
    if score >= 65: return 'good'
    if score >= 45: return 'moderate'
    if score >= 25: return 'low'
    return 'very_low'
