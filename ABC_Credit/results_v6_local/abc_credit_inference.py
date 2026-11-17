"""Command-line friendly helper for the locally trained ABC Credit model."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_model_runtime import DEFAULT_MODEL_PATH, predict_applicants as _predict_applicants


def predict_applicants(applicants, model_path=None):
    """Return decline/approval probabilities and the selected class.

    The model predicts historical application decisions, not default risk.
    """
    return _predict_applicants(
        applicants,
        model_path=Path(model_path) if model_path else DEFAULT_MODEL_PATH,
    )
