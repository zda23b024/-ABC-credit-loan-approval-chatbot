"""Local FastAPI assistant for the trained ABC Credit decision model.

This assistant reproduces historical application decisions. It is not a
repayment/default model or a production credit decision service.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
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
HOSTED_DEMO_DEPLOYMENT = (
    os.environ.get('VERCEL') == '1'
    or bool(os.environ.get('RAILWAY_ENVIRONMENT_NAME'))
)

if not MODEL_PATH.exists():
    raise RuntimeError(f'Trained model bundle not found: {MODEL_PATH}')
if not CATALOG_PATH.exists():
    raise RuntimeError(f'Question catalog not found: {CATALOG_PATH}. Run python chatbot/build_catalog.py.')

with CATALOG_PATH.open(encoding='utf-8') as handle:
    CATALOG = json.load(handle)
BUNDLE = load_model_bundle(MODEL_PATH)

RESIDENCE_LABELS = {'O': 'Owned', 'R': 'Rented', 'L': 'Leased'}
PRODUCT_LABELS = {'MC': 'Motorcycle', 'SC': 'Scooter', 'MO': 'Moped', 'EB': 'Electric two-wheeler'}
EMPLOYMENT_LABELS = {
    'SAL': 'Salaried', 'SEP': 'Self-employed', 'AGR': 'Agriculture',
    'NREGI': 'Non-regular income', 'STU': 'Student', 'NPP': 'Non-permanent probationary',
    'PEN': 'Pensioner', 'NONEARNMEM': 'Non-earning family member',
}
QUALIFICATION_LABELS = {
    'SSC': 'Secondary school (SSC)', 'HSC': 'Higher secondary (HSC)',
    'UG': 'Undergraduate', 'GRAD': 'Graduate', 'PG': 'Postgraduate', 'OTHERS': 'Other',
}
TIER_LABELS = {
    '01 Megapolis (A)': 'Megapolis A', '02 Megapolis (B)': 'Megapolis B',
    '03 Large Metro': 'Large metro', '04 Metro': 'Metro', '05 Urban': 'Urban',
    '06 Semi-Urban': 'Semi-urban', '07 Rural': 'Rural',
}
AGE_REPRESENTATIVES = {'18-25': 22, '26-35': 30, '36-45': 40, '46-60': 53}
INCOME_REPRESENTATIVES = {
    '0': 0, 'under_15000': 10000, '15000_24999': 20000,
    '25000_29999': 27000, '30000_36999': 33000, '37000_49999': 43000,
    '50000_84999': 67500, '85000_plus': 85000,
}
LOAN_REPRESENTATIVES = {
    'under_50000': 40000, '50000_74999': 62500, '75000_99999': 87500,
    '100000_124999': 112500, '125000_149999': 137500,
    '150000_199999': 175000, '200000_plus': 225000,
}
VEHICLE_PRICE_REPRESENTATIVES = {
    'under_75000': 65000, '75000_94999': 85000, '95000_114999': 105000,
    '115000_129999': 122500, '130000_154999': 142500,
    '155000_174999': 165000, '175000_plus': 200000,
}
ACTIVE_LOAN_VALUES = {True: 'PAST_LOANS_ACTIVE', False: 'NO_PAST_LOANS'}

VEHICLES = {row['description']: row for row in CATALOG['vehicles']}
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
    model_config = ConfigDict(protected_namespaces=())
    age_band: str | None = Field(default=None, max_length=20)
    age_value: int | None = Field(default=None, ge=18, le=60)
    qualification: str = Field(min_length=1, max_length=20)
    employment_type: str = Field(min_length=1, max_length=24)
    income_band: str | None = Field(default=None, max_length=24)
    monthly_net_income: float | None = Field(default=None, ge=0, le=10_000_000, allow_inf_nan=False)
    residence: Literal['O', 'R', 'L']
    area_tier: str = Field(min_length=1, max_length=32)
    vehicle_type: str = Field(min_length=1, max_length=8)
    model_description: str = Field(min_length=1, max_length=200)
    loan_band: str | None = Field(default=None, max_length=24)
    loan_amount: float | None = Field(default=None, gt=0, le=50_000_000, allow_inf_nan=False)
    vehicle_price_band: str | None = Field(default=None, max_length=24)
    vehicle_price: float | None = Field(default=None, gt=0, le=100_000_000, allow_inf_nan=False)
    has_active_loan: bool
    consent_to_log: bool


def build_model_row(answers: ApplicationAnswers) -> tuple[dict, dict]:
    """Map the 11 chatbot answers to the exact raw fields used by the bundle.

    Range selections intentionally use documented representative values. Exact
    typed numbers are retained. Vehicle price is used only to calculate LTV.
    """
    age_value = answers.age_value if answers.age_value is not None else AGE_REPRESENTATIVES[answers.age_band]
    income_value = (answers.monthly_net_income if answers.monthly_net_income is not None
                    else INCOME_REPRESENTATIVES[answers.income_band])
    loan_value = answers.loan_amount if answers.loan_amount is not None else LOAN_REPRESENTATIVES[answers.loan_band]
    vehicle_price_value = (answers.vehicle_price if answers.vehicle_price is not None
                           else VEHICLE_PRICE_REPRESENTATIVES[answers.vehicle_price_band])
    ltv = 100.0 * loan_value / vehicle_price_value
    if not np.isfinite(ltv) or ltv < 0.01 or ltv > 150:
        raise HTTPException(status_code=422, detail='Loan amount to vehicle price must produce an LTV between 0.01% and 150%.')

    vehicle = VEHICLES.get(answers.model_description)
    if vehicle is None or vehicle['product_code'] != answers.vehicle_type:
        raise HTTPException(status_code=422, detail='Choose a vehicle model from the supplied list.')

    model_row = {
        'Qualifications': answers.qualification,
        'Employment_Type': answers.employment_type,
        'Age': age_value,
        'Product_Code': vehicle['product_code'],
        'Loan_Amount': loan_value,
        'LTV': ltv,
        'Resident_Type': answers.residence,
        'Net_salary': income_value,
        'Final_Tier': answers.area_tier,
        'Make_Code': vehicle['make_code'],
        'Model_Description': vehicle['description'],
        'Model_Variant': vehicle['model_variant'],
        'PAST_LOANS_ACTIVE': ACTIVE_LOAN_VALUES[answers.has_active_loan],
    }
    expected = list(BUNDLE['features'])
    if list(model_row) != expected:
        raise RuntimeError(
            'Chatbot feature mapping does not match the saved model. '
            f'Expected {expected!r}; got {list(model_row)!r}.'
        )
    derived = {
        'Age': age_value,
        'Net_salary': income_value,
        'Loan_Amount': loan_value,
        'Vehicle_Price': vehicle_price_value,
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
            'feature_set': BUNDLE.get('feature_set'), 'threshold': BUNDLE['threshold_declined'],
            'deployment_mode': 'synthetic_demo' if HOSTED_DEMO_DEPLOYMENT else 'local',
            'persistent_audit_logging': not HOSTED_DEMO_DEPLOYMENT}


@app.get('/api/catalog')
def catalog():
    """Return category choices only; this file contains no applicant records."""
    return {
        'qualifications': [{'value': x, 'label': QUALIFICATION_LABELS.get(x, x)}
                           for x in CATALOG['Qualifications']],
        'employment_types': [{'value': x, 'label': EMPLOYMENT_LABELS.get(x, x)}
                             for x in CATALOG['Employment_Type']],
        'residences': [{'value': code, 'label': RESIDENCE_LABELS.get(code, code)}
                       for code in CATALOG['Resident_Type']],
        'area_tiers': [{'value': x, 'label': TIER_LABELS.get(x, x)} for x in CATALOG['Final_Tier']],
        'vehicle_types': [{'value': code, 'label': PRODUCT_LABELS.get(code, code)}
                          for code in CATALOG['Product_Code']],
        'representative_values': {
            'age_band': AGE_REPRESENTATIVES,
            'income_band': INCOME_REPRESENTATIVES,
            'loan_band': LOAN_REPRESENTATIVES,
            'vehicle_price_band': VEHICLE_PRICE_REPRESENTATIVES,
        },
        'vehicles': [
            {'value': x['description'], 'label': x['description'],
             'description': x['description'], 'product_code': x['product_code'],
             'make_code': x['make_code']}
            for x in CATALOG['vehicles']
        ],
    }


@app.post('/api/decision')
def decision(answers: ApplicationAnswers):
    if not answers.consent_to_log:
        raise HTTPException(status_code=422, detail='Please acknowledge the audit-log notice to continue.')
    allowed = {
        'qualification': set(CATALOG['Qualifications']),
        'employment_type': set(CATALOG['Employment_Type']),
        'area_tier': set(CATALOG['Final_Tier']),
        'vehicle_type': set(CATALOG['Product_Code']),
    }
    for field, choices in allowed.items():
        if getattr(answers, field) not in choices:
            raise HTTPException(status_code=422, detail=f'Choose a valid {field.replace("_", " ")}.')
    range_choices = {
        'age_band': set(AGE_REPRESENTATIVES),
        'income_band': set(INCOME_REPRESENTATIVES),
        'loan_band': set(LOAN_REPRESENTATIVES),
        'vehicle_price_band': set(VEHICLE_PRICE_REPRESENTATIVES),
    }
    for band_field, allowed_values in range_choices.items():
        value = getattr(answers, band_field)
        if value is not None and value not in allowed_values:
            raise HTTPException(status_code=422, detail=f'Choose a valid {band_field.replace("_", " ")}.')
    pairs = [
        ('age_band', answers.age_band, 'age_value', answers.age_value),
        ('income_band', answers.income_band, 'monthly_net_income', answers.monthly_net_income),
        ('loan_band', answers.loan_band, 'loan_amount', answers.loan_amount),
        ('vehicle_price_band', answers.vehicle_price_band, 'vehicle_price', answers.vehicle_price),
    ]
    for band_name, band_value, exact_name, exact_value in pairs:
        if (band_value is None) == (exact_value is None):
            raise HTTPException(status_code=422, detail=f'Provide one answer for {band_name.replace("_", " ")}.')
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
    if answers.age_band:
        warnings.append(f'Age range represented as {age_value} years for this estimate.')
    if answers.income_band:
        warnings.append(f'Income range represented as ₹{income_value:,.0f} per month for this estimate.')
    if answers.loan_band:
        warnings.append(f'Loan range represented as ₹{loan_value:,.0f} for this estimate.')
    if answers.vehicle_price_band:
        warnings.append(f'Vehicle-price range represented as ₹{vehicle_price_value:,.0f} to calculate LTV.')

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
            'age_band': answers.age_band, 'age_value': answers.age_value,
            'qualification': answers.qualification,
            'employment_type': answers.employment_type,
            'income_band': answers.income_band, 'monthly_net_income': answers.monthly_net_income,
            'residence': answers.residence,
            'area_tier': answers.area_tier,
            'vehicle_type': answers.vehicle_type,
            'model_description': answers.model_description,
            'loan_band': answers.loan_band, 'loan_amount': answers.loan_amount,
            'vehicle_price_band': answers.vehicle_price_band, 'vehicle_price': answers.vehicle_price,
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
    if not HOSTED_DEMO_DEPLOYMENT:
        try:
            _append_audit(audit)
        except OSError:
            LOGGER.exception('Audit log write failed')
            raise HTTPException(status_code=503, detail='Could not record this application. No assessment was returned.')
    result['persistent_audit_logging'] = not HOSTED_DEMO_DEPLOYMENT
    result['deployment_mode'] = 'synthetic_demo' if HOSTED_DEMO_DEPLOYMENT else 'local'
    return result


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('chatbot.app:app', host='127.0.0.1', port=8000, reload=False, access_log=False)
