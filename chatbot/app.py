"""Local FastAPI assistant for the trained ABC Credit decision model.

This assistant reproduces historical application decisions. It is not a
repayment/default model or a production credit decision service.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4
import json
import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from credit_model_runtime import load_model_bundle, predict_applicants

ROOT = Path(__file__).resolve().parents[1]
CHATBOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'ABC_Credit' / 'results_v6_local' / 'abc_credit_model_bundle.joblib'
CATALOG_PATH = CHATBOT_DIR / 'catalog.json'
AUDIT_PATH = Path(os.environ.get(
    'ABC_CREDIT_AUDIT_LOG', CHATBOT_DIR / 'data' / 'audit' / 'applications.jsonl'))

if not MODEL_PATH.exists():
    raise RuntimeError(f'Trained model bundle not found: {MODEL_PATH}')
if not CATALOG_PATH.exists():
    raise RuntimeError(f'Question catalog not found: {CATALOG_PATH}. Run python chatbot/build_catalog.py.')

with CATALOG_PATH.open(encoding='utf-8') as handle:
    CATALOG = json.load(handle)
BUNDLE = load_model_bundle(MODEL_PATH)

PRODUCT_LABELS = {'MC': 'Motorcycle', 'SC': 'Scooter', 'MO': 'Moped', 'EB': 'Electric two-wheeler'}
EMPLOYMENT_LABELS = {
    'SAL': 'Salaried', 'SEP': 'Self-employed', 'AGR': 'Agriculture',
    'NREGI': 'Non-regular income', 'STU': 'Student', 'NPP': 'Non-permanent probationary',
    'PEN': 'Pensioner', 'NONEARNMEM': 'Non-earning family member',
}
ACTIVE_LOAN_VALUES = {True: 'PAST_LOANS_ACTIVE', False: 'NO_PAST_LOANS'}

AUDIT_LOCK = Lock()
LOGGER = logging.getLogger('abc_credit_chatbot')

app = FastAPI(
    title='ABC Credit Application Assistant',
    description='ABC Credit loan application assistant.',
    version='1.0.0',
    docs_url=None,
    redoc_url=None,
)
app.mount('/static', StaticFiles(directory=CHATBOT_DIR / 'static'), name='static')


class ApplicationAnswers(BaseModel):
    """Validated answers collected by the browser chat."""
    model_config = ConfigDict(protected_namespaces=(), extra='forbid')
    age: int = Field(ge=18, le=60)
    employment_type: str = Field(min_length=1, max_length=24)
    monthly_net_income: float = Field(ge=0, le=10_000_000, allow_inf_nan=False)
    vehicle_type: str = Field(min_length=1, max_length=8)
    loan_amount: float = Field(gt=0, le=50_000_000, allow_inf_nan=False)
    vehicle_price: float = Field(gt=0, le=100_000_000, allow_inf_nan=False)
    has_active_loan: bool
    consent_to_log: bool


def build_model_row(answers: ApplicationAnswers) -> tuple[dict, dict]:
    """Map the seven collected answers to model inputs and derived values."""
    ltv = 100.0 * answers.loan_amount / answers.vehicle_price
    if not np.isfinite(ltv) or ltv < 0.01 or ltv > 150:
        raise HTTPException(status_code=422, detail='Loan amount to vehicle price must produce an LTV between 0.01% and 150%.')

    provided_features = {
        'Employment_Type': answers.employment_type,
        'Age': answers.age,
        'Product_Code': answers.vehicle_type,
        'Loan_Amount': answers.loan_amount,
        'LTV': ltv,
        'Net_salary': answers.monthly_net_income,
        'PAST_LOANS_ACTIVE': ACTIVE_LOAN_VALUES[answers.has_active_loan],
    }
    expected = list(BUNDLE['features'])
    ordered_provided = [feature for feature in expected if feature in provided_features]
    if ordered_provided != list(provided_features):
        raise RuntimeError(
            'Chatbot feature mapping does not match the saved model. '
            f'Expected provided features in model order {ordered_provided!r}; '
            f'got {list(provided_features)!r}.'
        )
    model_row = {feature: provided_features.get(feature) for feature in expected}
    derived = {
        'Age': answers.age,
        'Net_salary': answers.monthly_net_income,
        'Loan_Amount': answers.loan_amount,
        'Vehicle_Price': answers.vehicle_price,
        'LTV': ltv,
    }
    return model_row, derived


def _append_audit(record: dict) -> None:
    """Append one pseudonymous application record; never include IP or direct IDs."""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    with AUDIT_LOCK:
        with AUDIT_PATH.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(line + '\n')


def _model_prediction(row: dict) -> tuple[float, str]:
    prediction = predict_applicants([row], bundle=BUNDLE).iloc[0]
    return float(prediction['decline_probability']), str(prediction['Decision'])


@app.get('/', include_in_schema=False)
def home():
    return FileResponse(CHATBOT_DIR / 'static' / 'index.html')


@app.get('/api/health')
def health():
    return {'status': 'ready', 'model_name': BUNDLE.get('model_name'),
            'feature_set': BUNDLE.get('feature_set'), 'threshold': BUNDLE['threshold_declined']}


@app.get('/api/catalog')
def catalog():
    """Return category choices only; this file contains no applicant records."""
    return {
        'employment_types': [{'value': x, 'label': EMPLOYMENT_LABELS.get(x, x)}
                             for x in CATALOG['Employment_Type']],
        'vehicle_types': [{'value': code, 'label': PRODUCT_LABELS.get(code, code)}
                          for code in CATALOG['Product_Code']],
    }


@app.post('/api/decision')
def decision(answers: ApplicationAnswers):
    if not answers.consent_to_log:
        raise HTTPException(status_code=422, detail='Please acknowledge the audit-log notice to continue.')
    allowed = {
        'employment_type': set(CATALOG['Employment_Type']),
        'vehicle_type': set(CATALOG['Product_Code']),
    }
    for field, choices in allowed.items():
        if getattr(answers, field) not in choices:
            raise HTTPException(status_code=422, detail=f'Choose a valid {field.replace("_", " ")}.')
    model_row, derived = build_model_row(answers)
    age_value = derived['Age']
    income_value = derived['Net_salary']
    loan_value = derived['Loan_Amount']
    vehicle_price_value = derived['Vehicle_Price']
    ltv = derived['LTV']
    started = perf_counter()
    try:
        decline_probability, predicted_decision = _model_prediction(model_row)
    except Exception as exc:
        # Keep applicant data and model internals out of the returned error.
        LOGGER.exception('Model inference failed')
        raise HTTPException(status_code=503, detail='The model is temporarily unavailable. Please try again later.') from exc
    latency_ms = round((perf_counter() - started) * 1000, 2)

    # Data-quality notice when applicant inputs are far outside training support.
    warnings = []
    if loan_value > 286_847:
        warnings.append('Requested amount is above the maximum observed in training data; treat the score cautiously.')
    if income_value > 980_000:
        warnings.append('Income is above the maximum observed in training data; treat the score cautiously.')
    warnings.append(
        'The saved model also uses qualification, residence type, area tier, vehicle make, '
        'model, and variant. These were not asked and were imputed; treat this estimate cautiously.'
    )

    session_id = str(uuid4())
    result = {
        'session_id': session_id,
        'decision': predicted_decision,
        'decline_probability': decline_probability,
        'approval_probability': 1-decline_probability,
        'threshold_declined': float(BUNDLE['threshold_declined']),
        'model_name': BUNDLE.get('model_name'),
        'latency_ms': latency_ms,
        'warnings': warnings,
        'message': ('The model classifies this application with the historical DECLINED label.'
                    if predicted_decision == 'DECLINED' else
                    'The model classifies this application with the historical APPROVED label.'),
    }
    audit = {
        'event': 'model_decision',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'session_id': session_id,
        'model_version': BUNDLE.get('version', 'unknown'),
        'model_name': BUNDLE.get('model_name'),
        'feature_set': BUNDLE.get('feature_set'),
        'answers': {
            'age': answers.age,
            'employment_type': answers.employment_type,
            'monthly_net_income': answers.monthly_net_income,
            'vehicle_type': answers.vehicle_type,
            'loan_amount': answers.loan_amount,
            'vehicle_price': answers.vehicle_price,
            'has_active_loan': answers.has_active_loan,
        },
        'model_features': model_row,
        'derived': {'age_representative_value': age_value,
                    'monthly_net_income_representative_value': income_value,
                    'loan_amount_representative_value': loan_value,
                    'vehicle_price_representative_value': vehicle_price_value,
                    'ltv_percent': ltv},
        'score': {'decline_probability': decline_probability,
                  'approval_probability': 1-decline_probability,
                  'decline_threshold': float(BUNDLE['threshold_declined']),
                  'decision': predicted_decision},
        'latency_ms': latency_ms,
    }
    try:
        _append_audit(audit)
    except OSError:
        LOGGER.exception('Audit log write failed')
        raise HTTPException(status_code=503, detail='Could not record this application. No assessment was returned.')
    return result


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('chatbot.app:app', host='127.0.0.1', port=8000, reload=False, access_log=False)
