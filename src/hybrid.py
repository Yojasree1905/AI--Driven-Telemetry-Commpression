"""The hybrid detection engine.

Three estimators, one decision path:

    detector  p(attack)          -> is it hostile?
    classifier argmax p(family)  -> what kind of attack?
    novelty   percentile         -> have we ever seen anything like this?

The rule below is deliberately not a majority vote. A vote treats the three
models as interchangeable opinions, but they answer different questions and
have very different reliability. The supervised detector decides *whether*
something is an attack; the novelty estimator only gets authority over flows
the detector cleared, which is precisely the zero-day case.

Both ``train_models.py`` and the UI call ``ThreatEngine.score`` so evaluation
numbers and screen output can never disagree.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ATTACK_PLAYBOOK,
    DEFAULT_ALERT_BUDGET,
    DEFAULT_FAMILY_CONFIDENCE_FLOOR,
    DEFAULT_NOVELTY_QUANTILE,
    ENGINE_CONFIG_PATH,
    STATUS_ATTACK,
    STATUS_NORMAL,
    STATUS_ZERO_DAY,
)
from src.preprocessing import align_features

UNKNOWN_FAMILY = 'Unknown'


class ThreatEngine:
    """Scores raw network flows and returns analyst-ready verdicts."""

    def __init__(self, preprocessor, detector, classifier, novelty, config,
                 reference_forest=None):
        self.preprocessor = preprocessor
        self.detector = detector
        self.classifier = classifier
        self.novelty = novelty
        self.reference_forest = reference_forest
        self.config = config or {}

        self.classes = list(self.config.get('attack_classes', []))
        self.threshold = float(self.config.get('detector_threshold', 0.5))
        self.novelty_threshold = float(self.config.get('novelty_threshold', np.inf))
        self.family_floor = float(
            self.config.get('family_confidence_floor', DEFAULT_FAMILY_CONFIDENCE_FLOOR)
        )
        # Empirical benign novelty scores, used to turn a raw Isolation Forest
        # score into an interpretable "more unusual than N% of benign traffic".
        self._novelty_ref = np.asarray(self.config.get('novelty_reference', []), dtype=float)
        self._novelty_ref.sort()

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls):
        """Load the engine from disk, or raise a clear error if untrained."""
        from src.config import (
            CLASSIFIER_PATH, DETECTOR_PATH, NOVELTY_PATH, PREPROCESSOR_PATH, RF_PATH,
        )

        required = {
            'preprocessor': PREPROCESSOR_PATH,
            'detector': DETECTOR_PATH,
            'classifier': CLASSIFIER_PATH,
            'novelty': NOVELTY_PATH,
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f'Missing trained artifacts: {", ".join(missing)}. Run `python train_models.py` first.'
            )

        config = {}
        if ENGINE_CONFIG_PATH.exists():
            config = json.loads(ENGINE_CONFIG_PATH.read_text(encoding='utf-8'))

        return cls(
            preprocessor=joblib.load(PREPROCESSOR_PATH),
            detector=joblib.load(DETECTOR_PATH),
            classifier=joblib.load(CLASSIFIER_PATH),
            novelty=joblib.load(NOVELTY_PATH),
            reference_forest=joblib.load(RF_PATH) if RF_PATH.exists() else None,
            config=config,
        )

    # -- internals ----------------------------------------------------------
    def transform(self, df):
        return self.preprocessor.transform(align_features(df))

    def _novelty_percentile(self, scores):
        """Where each score falls within the benign reference distribution."""
        if self._novelty_ref.size == 0:
            return np.zeros_like(scores, dtype=float)
        return np.searchsorted(self._novelty_ref, scores, side='right') / self._novelty_ref.size

    # -- scoring ------------------------------------------------------------
    def score_matrix(self, X):
        """Score an already-transformed matrix. Returns a raw signal dict."""
        p_attack = self.detector.predict_proba(X)[:, 1]

        family_proba = self.classifier.predict_proba(X)
        family_idx = family_proba.argmax(axis=1)
        family_conf = family_proba.max(axis=1)

        raw_novelty = -self.novelty.score_samples(X)
        novelty_pct = self._novelty_percentile(raw_novelty)

        return {
            'p_attack': p_attack,
            'family_proba': family_proba,
            'family_idx': family_idx,
            'family_conf': family_conf,
            'novelty': raw_novelty,
            'novelty_pct': novelty_pct,
        }

    def decide(self, signals):
        """Apply the decision rule to raw signals. Returns (status, attack_type)."""
        p = signals['p_attack']
        novelty = signals['novelty']
        conf = signals['family_conf']
        idx = signals['family_idx']

        is_attack = p >= self.threshold
        is_novel = novelty >= self.novelty_threshold

        status = np.where(is_attack, STATUS_ATTACK,
                          np.where(is_novel, STATUS_ZERO_DAY, STATUS_NORMAL))

        names = np.array([self.classes[i] if i < len(self.classes) else UNKNOWN_FAMILY
                          for i in idx], dtype=object)

        attack_type = np.full(len(p), '—', dtype=object)
        # A confidently-detected attack the family model cannot place, on traffic
        # that is also structurally unfamiliar, is an unnamed family — not a
        # forced guess into the closest known bucket.
        named = is_attack & ((conf >= self.family_floor) | ~is_novel)
        attack_type[named] = names[named]
        attack_type[is_attack & ~named] = UNKNOWN_FAMILY
        attack_type[status == STATUS_ZERO_DAY] = UNKNOWN_FAMILY

        return status, attack_type

    def score(self, df, top_k=3):
        """Score raw flow records.

        Returns a DataFrame indexed like ``df`` with one row per flow, plus the
        transformed matrix so callers can feed SHAP without transforming twice.
        """
        X = self.transform(df)
        signals = self.score_matrix(X)
        status, attack_type = self.decide(signals)

        p = signals['p_attack']
        novelty_pct = signals['novelty_pct']

        # Risk blends how confident the supervised detector is with how far the
        # flow sits outside the benign baseline. Both terms are in [0, 1], so
        # the score is directly interpretable and never saturates at 100.
        risk = 100.0 * (0.70 * p + 0.30 * novelty_pct)
        risk = np.where(status == STATUS_ZERO_DAY, np.maximum(risk, 55.0), risk)

        severity = [
            ATTACK_PLAYBOOK.get(a, {}).get('severity', 'Info') if s != STATUS_NORMAL else 'Info'
            for s, a in zip(status, attack_type)
        ]
        action = [
            ATTACK_PLAYBOOK.get(a, {}).get('action', 'No action required.')
            if s != STATUS_NORMAL else 'No action required — flow matches the authorised baseline.'
            for s, a in zip(status, attack_type)
        ]

        # Distance from the operating point, expressed as an analyst-facing band.
        margin = np.abs(p - self.threshold)
        band = np.where(margin >= 0.30, 'High', np.where(margin >= 0.12, 'Medium', 'Low'))

        result = pd.DataFrame({
            'Status': status,
            'Attack Type': attack_type,
            'Severity': severity,
            'Risk Score': np.round(risk, 1),
            'Detector Confidence': np.round(p * 100, 1),
            'Family Confidence': np.round(signals['family_conf'] * 100, 1),
            'Novelty Percentile': np.round(novelty_pct * 100, 1),
            'Novelty Score': np.round(signals['novelty'], 4),
            'Decision Confidence': band,
            'Recommended Action': action,
        }, index=df.index if len(df.index) == len(status) else None)

        # Family confidence is meaningless on flows we did not call an attack.
        result.loc[result['Status'] == STATUS_NORMAL, 'Family Confidence'] = np.nan

        if top_k:
            proba = signals['family_proba']
            order = np.argsort(-proba, axis=1)[:, :top_k]
            alternatives = []
            for row_i, cols in enumerate(order):
                if status[row_i] == STATUS_NORMAL:
                    alternatives.append('—')
                    continue
                alternatives.append(', '.join(
                    f'{self.classes[c]} {proba[row_i, c] * 100:.0f}%' for c in cols
                    if c < len(self.classes)
                ))
            result['Family Ranking'] = alternatives

        if self.reference_forest is not None:
            rf_p = self.reference_forest.predict_proba(X)[:, 1]
            result['RF Second Opinion'] = np.where(rf_p >= 0.5, 'Attack', 'Normal')
            result['RF Confidence'] = np.round(rf_p * 100, 1)

        return result, X


def build_engine_config(classes, threshold, novelty_threshold, novelty_reference,
                        alert_budget=DEFAULT_ALERT_BUDGET,
                        novelty_quantile=DEFAULT_NOVELTY_QUANTILE,
                        family_floor=DEFAULT_FAMILY_CONFIDENCE_FLOOR,
                        extra=None):
    """Serialisable engine calibration written alongside the model artifacts."""
    reference = np.asarray(novelty_reference, dtype=float)
    # Subsample the reference distribution: 4k quantile points reproduce the
    # percentile lookup to well under a tenth of a percent at a fraction of the
    # file size.
    if reference.size > 4000:
        reference = np.quantile(reference, np.linspace(0, 1, 4000))
    config = {
        'attack_classes': list(classes),
        'detector_threshold': float(threshold),
        'novelty_threshold': float(novelty_threshold),
        'novelty_quantile': float(novelty_quantile),
        'alert_budget': float(alert_budget),
        'family_confidence_floor': float(family_floor),
        'novelty_reference': [float(v) for v in np.sort(reference)],
    }
    if extra:
        config.update(extra)
    return config


# ---------------------------------------------------------------------------
# Backwards-compatible helpers for older notebooks
# ---------------------------------------------------------------------------
def hybrid_predict(detector, classifier, novelty, X, threshold=0.5):
    """Binary attack prediction for an already-transformed matrix."""
    return (detector.predict_proba(X)[:, 1] >= threshold).astype(int)


def hybrid_predict_records(xgb_model, rf_model, iso_model, preprocessor, df):
    """Return the calibrated result columns consumed by the Streamlit UI."""
    X = preprocessor.transform(align_features(df))
    xgb_probability = xgb_model.predict_proba(X)[:, 1]
    rf_probability = rf_model.predict_proba(X)[:, 1]
    
    # Supervised attack signal requires strong signature consensus across both models
    is_known_attack = (xgb_probability >= 0.72) & (rf_probability >= 0.65)
    
    # Isolation Forest structural anomaly detection
    iso_decision = -iso_model.decision_function(X)
    isolation_anomaly = (iso_model.predict(X) == -1) | (iso_decision >= -0.01)
    
    # Zero-Day anomaly: cleared by supervised signature models, but flagged as structural anomaly
    is_zeroday = (~is_known_attack) & isolation_anomaly
    
    status = np.where(
        is_known_attack,
        'Attack',
        np.where(is_zeroday, 'Zero-Day Anomaly', 'Normal')
    )
    
    avg_prob = (xgb_probability + rf_probability) / 2.0
    normal_risk = np.round(np.clip(avg_prob * 25.0, 5.0, 22.0), 1)
    attack_risk = np.round(np.maximum(xgb_probability, rf_probability) * 100.0, 1)
    zeroday_risk = np.round(np.clip(72.0 + iso_decision * 140.0, 70.0, 95.0), 1)
    
    risk = np.where(
        status == 'Attack',
        attack_risk,
        np.where(status == 'Zero-Day Anomaly', zeroday_risk, normal_risk)
    )

    result = pd.DataFrame({
        'Isolation Forest': np.where(isolation_anomaly, 'Anomaly', 'Normal'),
        'Random Forest': np.where(rf_probability >= 0.5, 'Attack', 'Normal'),
        'XGBoost': np.where(xgb_probability >= 0.5, 'Attack', 'Normal'),
        'Hybrid Verdict': status,
        'Risk Score (%)': risk,
        'Status': status,
    }, index=df.index)
    return result, X
