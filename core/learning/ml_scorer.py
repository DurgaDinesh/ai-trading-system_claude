"""
Phase 2 Self-Learning: XGBoost signal quality classifier.
Activates after MIN_TRADES paper trades.
Blends ML probability with rule-based composite score.
"""

import os
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import yaml

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from database.trade_journal import journal, get_db
from database.models import Trade, TradeStatus, MLModelMetric, TradingMode

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))["ml_scorer"]

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "signal_classifier.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "composite_score", "confluence_count", "vix_at_signal",
    "pcr_at_signal", "news_sentiment", "atr_at_entry",
    "hour_of_day", "day_of_week", "global_score",
    "ema_signal", "rsi_signal", "macd_signal",
]


class MLScorer:
    """XGBoost classifier for signal quality prediction."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._is_active = False
        self._n_trades_trained_on = 0          # size of the training sample window (metrics only)
        self._lifetime_trades_at_last_train = 0  # total lifetime trades as of last training (retrain cadence)
        self._load_if_exists()

    def _load_if_exists(self):
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                self._model = joblib.load(MODEL_PATH)
                self._scaler = joblib.load(SCALER_PATH)
                self._is_active = True
                logger.info("ml_model_loaded", path=str(MODEL_PATH))
            except Exception as e:
                logger.warning("ml_model_load_failed", error=str(e))

    @property
    def is_active(self) -> bool:
        return self._is_active and self._model is not None

    def should_train(self) -> bool:
        """Check if we have enough trades to train/retrain."""
        stats = journal.get_performance_stats(mode=TradingMode.PAPER)
        n = stats.get("total_trades", 0)
        min_n = _cfg["activate_after_n_trades"]
        retrain_n = _cfg["retrain_every_n_trades"]
        if n < min_n:
            return False
        if not self.is_active:
            return True
        # Cadence is measured against lifetime trade count, not the size of
        # the (window-capped) training sample — otherwise once lifetime
        # trades exceed what feature_window_days holds, this gap grows
        # unbounded and retraining fires on every single trade close.
        return (n - self._lifetime_trades_at_last_train) >= retrain_n

    def _build_feature_df(self) -> Optional[pd.DataFrame]:
        """Extract features and labels from closed paper trades."""
        cutoff = datetime.utcnow() - timedelta(days=_cfg["feature_window_days"])
        with get_db() as db:
            trades = (
                db.query(Trade)
                .filter(
                    Trade.status == TradeStatus.CLOSED,
                    Trade.mode == TradingMode.PAPER,
                    Trade.created_at >= cutoff,
                )
                .all()
            )

        if len(trades) < _cfg["min_train_samples"]:
            return None

        rows = []
        for t in trades:
            ctx = t.global_context or {}
            indicators = t.indicators_triggered or []
            row = {
                "composite_score": t.composite_score or 50,
                "confluence_count": t.confluence_count or 0,
                "vix_at_signal": t.vix_at_signal or 15,
                "pcr_at_signal": t.pcr_at_signal or 1.0,
                "news_sentiment": t.news_sentiment or 0,
                "atr_at_entry": t.atr_at_entry or 0,
                "hour_of_day": t.signal_time.hour if t.signal_time else 10,
                "day_of_week": t.signal_time.weekday() if t.signal_time else 0,
                "global_score": ctx.get("global_score", 0),
                "ema_signal": 1 if "EMA_STACK" in indicators else 0,
                "rsi_signal": 1 if "RSI_CROSSOVER" in indicators else 0,
                "macd_signal": 1 if "MACD_CROSSOVER" in indicators else 0,
                "label": 1 if (t.net_pnl or 0) > 0 else 0,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def train(self):
        """Train (or retrain) the XGBoost classifier on paper trade history."""
        if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("ml_dependencies_not_installed", xgboost=XGB_AVAILABLE, sklearn=SKLEARN_AVAILABLE)
            return

        df = self._build_feature_df()
        if df is None:
            logger.info("ml_training_skipped_insufficient_data")
            return

        X = df[FEATURE_COLUMNS].values
        y = df["label"].values

        # Train into LOCAL variables first — self._model/self._scaler must
        # only be updated together, once training fully succeeds. Otherwise
        # a mid-training failure (e.g. XGBClassifier choking on a single-class
        # y_train, plausible with lopsided early results) leaves the old
        # model paired with a scaler already re-fit to the new distribution.
        try:
            stratify = y if len(set(y)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )

            new_scaler = StandardScaler()
            X_train_sc = new_scaler.fit_transform(X_train)
            X_test_sc = new_scaler.transform(X_test)

            new_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            new_model.fit(X_train_sc, y_train, eval_set=[(X_test_sc, y_test)], verbose=False)

            y_pred = new_model.predict(X_test_sc)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        except Exception as e:
            logger.error("ml_training_failed", error=str(e))
            return   # self._model / self._scaler are untouched — no mismatch

        logger.info("ml_model_trained", accuracy=acc, precision=prec, recall=rec, f1=f1, n=len(df))

        # Commit the new model+scaler together now that training succeeded
        self._model = new_model
        self._scaler = new_scaler
        self._is_active = True
        self._n_trades_trained_on = len(df)
        lifetime_stats = journal.get_performance_stats(mode=TradingMode.PAPER)
        self._lifetime_trades_at_last_train = lifetime_stats.get("total_trades", 0)

        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)

        # Save metrics to DB
        feat_imp = dict(zip(FEATURE_COLUMNS, new_model.feature_importances_.tolist()))
        with get_db() as db:
            db.add(MLModelMetric(
                model_type="xgboost",
                n_samples=len(df),
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                feature_importances=feat_imp,
                model_path=str(MODEL_PATH),
            ))

    def predict_win_probability(self, features: dict) -> float:
        """
        Return probability (0–1) that a signal will win.
        Returns 0.5 (neutral) if model not active.
        """
        if not self.is_active:
            return 0.5

        row = [features.get(col, 0) for col in FEATURE_COLUMNS]
        X = self._scaler.transform([row])
        prob = float(self._model.predict_proba(X)[0][1])
        return round(prob, 3)

    def blend_score(self, rule_based_score: float, features: dict) -> float:
        """
        Blend ML probability with rule-based composite score.
        config: ml_scorer.confidence_blend_weight = 0.4 (40% ML, 60% rule-based)
        """
        if not self.is_active:
            return rule_based_score

        ml_prob = self.predict_win_probability(features)
        ml_score = ml_prob * 100
        blend = _cfg["confidence_blend_weight"]
        blended = (blend * ml_score) + ((1 - blend) * rule_based_score)
        logger.debug("score_blended", rule=rule_based_score, ml=ml_score, blended=round(blended, 1))
        return round(blended, 1)


ml_scorer = MLScorer()
