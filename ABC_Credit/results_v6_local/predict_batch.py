"""Score an unseen CSV with the saved ABC Credit model bundle.

Input rows must contain the bundle's raw model features and an ``ID`` column.
Extra columns (for example Pincode, Gender, or a historical Decision label) are
ignored. The output is deliberately limited to ID, probabilities, and the
thresholded historical decision.

Example from the project root:
    python ABC_Credit/results_v6_local/predict_batch.py new_applications.csv predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_model_runtime import DEFAULT_MODEL_PATH, load_model_bundle, predict_applicants


def predict_csv(input_path: str | Path, output_path: str | Path,
                model_path: str | Path | None = None) -> pd.DataFrame:
    """Score a CSV and write a minimal result file while preserving every ID.

    Missing feature cells and unseen categorical values are handled by the
    exact preprocessing path stored by ``credit_model_runtime``. Missing
    feature columns are rejected because silently inventing values can change
    predictions.
    """
    source = Path(input_path)
    target = Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError('Input and output paths must be different.')
    if not source.is_file():
        raise FileNotFoundError(f'Input CSV not found: {source}')

    # Keep IDs as strings so values such as 000123 are not changed by parsing.
    frame = pd.read_csv(source, dtype={'ID': 'string'}, low_memory=False)
    if frame.empty:
        raise ValueError('Input CSV contains no application rows.')
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f'Input CSV has duplicate column names: {duplicates}')
    if 'ID' not in frame:
        raise ValueError("Input CSV must contain an 'ID' column to preserve application IDs.")

    bundle = load_model_bundle(model_path or DEFAULT_MODEL_PATH)
    missing = [feature for feature in bundle['features'] if feature not in frame.columns]
    if missing:
        raise ValueError('Input CSV is missing required model feature columns: ' + ', '.join(missing))
    predictions = predict_applicants(frame, bundle=bundle)
    output = pd.DataFrame({
        'ID': frame['ID'],
        'decline_probability': predictions['decline_probability'].to_numpy(),
        'approval_probability': predictions['approval_probability'].to_numpy(),
        'Decision': predictions['Decision'].to_numpy(),
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path, help='Unseen CSV with ID and saved-model raw feature columns')
    parser.add_argument('output_csv', type=Path, help='Destination CSV for IDs, probabilities, and decisions')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH, help='Saved model bundle path')
    args = parser.parse_args(argv)
    try:
        output = predict_csv(args.input_csv, args.output_csv, args.model)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(f'Scored {len(output)} rows with the saved bundle; wrote {args.output_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
