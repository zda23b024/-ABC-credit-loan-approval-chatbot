"""Shared, validated loading and inference for the saved ABC Credit bundle."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

from train_credit_model import make_features

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / 'ABC_Credit' / 'results_v6_local' / 'abc_credit_model_bundle.joblib'

# These versions match the dependencies used by the saved scikit-learn pipeline
# and LightGBM estimator. Keep chatbot/requirements.txt aligned with this map.
MODEL_DEPENDENCIES = {
    'joblib': '1.4.2',
    'numpy': '2.1.3',
    'pandas': '2.2.3',
    'scipy': '1.16.3',
    'scikit-learn': '1.5.2',
    'lightgbm': '4.7.0',
}


def check_model_dependencies() -> None:
    """Fail with an actionable message before unpickling on incompatible versions."""
    mismatches = []
    for package, expected in MODEL_DEPENDENCIES.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = 'not installed'
        if actual != expected:
            mismatches.append(f'{package}=={expected} required (found {actual})')
    if mismatches:
        raise RuntimeError(
            'The saved ABC Credit model requires the pinned runtime dependencies. '
            'Install chatbot/requirements.txt with Python 3.12. Mismatches: '
            + '; '.join(mismatches)
        )


@lru_cache(maxsize=4)
def _load_model_bundle_cached(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f'Saved ABC Credit model bundle not found: {path}')
    check_model_dependencies()
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f'Could not load the saved ABC Credit model at {path}. '
            'Confirm that the pinned dependencies in chatbot/requirements.txt are installed.'
        ) from exc

    required = {'model', 'calibrator', 'features', 'threshold_declined', 'positive_class'}
    missing = sorted(required.difference(bundle)) if isinstance(bundle, dict) else sorted(required)
    if missing:
        raise RuntimeError(f'Model bundle is missing required fields: {", ".join(missing)}')
    if not callable(getattr(bundle['model'], 'predict_proba', None)):
        raise RuntimeError('Saved model does not provide predict_proba().')
    if bundle['positive_class'] != 'DECLINED':
        raise RuntimeError('Saved model positive_class must be DECLINED.')
    features = bundle['features']
    if not isinstance(features, (list, tuple)) or not features:
        raise RuntimeError('Saved model feature list is empty or invalid.')
    try:
        threshold = float(bundle['threshold_declined'])
    except (TypeError, ValueError) as exc:
        raise RuntimeError('Saved model threshold is invalid.') from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RuntimeError('Saved model threshold must be a finite probability from 0 to 1.')
    classes = list(getattr(bundle['model'], 'classes_', []))
    if 1 not in classes:
        raise RuntimeError(f'Saved model classes do not contain the DECLINED label 1: {classes!r}')
    calibrator = bundle['calibrator']
    if calibrator is not None and not callable(getattr(calibrator, 'predict_proba', None)):
        raise RuntimeError('Saved probability calibrator does not provide predict_proba().')
    return bundle


def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and validate the persisted pipeline, calibrator, and threshold."""
    return _load_model_bundle_cached(str(Path(model_path).resolve()))


def predict_applicants(applicants, bundle: dict[str, Any] | None = None,
                       model_path: str | Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """Return calibrated probabilities and class using the bundle threshold.

    This function is the single scoring path used by both the chatbot API and
    the standalone inference helper. It predicts historical decisions only.
    """
    loaded = bundle if bundle is not None else load_model_bundle(model_path)
    x = make_features(pd.DataFrame(applicants), loaded['features'])
    model = loaded['model']
    classes = list(model.classes_)
    decline_index = classes.index(1)
    raw = np.asarray(model.predict_proba(x), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(classes):
        raise RuntimeError('Saved model returned an invalid probability matrix.')
    decline_probability = raw[:, decline_index]
    calibrator = loaded.get('calibrator')
    if calibrator is not None:
        calibrated = np.asarray(calibrator.predict_proba(
            logit(np.clip(decline_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        ), dtype=float)
        calibrator_classes = list(getattr(calibrator, 'classes_', []))
        if calibrated.ndim != 2 or 1 not in calibrator_classes:
            raise RuntimeError('Saved calibrator returned an invalid probability matrix.')
        decline_probability = calibrated[:, calibrator_classes.index(1)]
    if not np.all(np.isfinite(decline_probability)):
        raise RuntimeError('Model returned a non-finite probability.')
    if np.any((decline_probability < 0) | (decline_probability > 1)):
        raise RuntimeError('Model returned a probability outside [0, 1].')
    threshold = float(loaded['threshold_declined'])
    return pd.DataFrame({
        'decline_probability': decline_probability,
        'approval_probability': 1.0 - decline_probability,
        'Decision': np.where(decline_probability >= threshold, 'DECLINED', 'APPROVED'),
    })
