# ABC Credit application assistant

A local, browser-based chatbot that calls the saved model bundle at
`ABC_Credit/results_v6_local/abc_credit_model_bundle.joblib`.

## Scope

The model predicts whether an application resembles historical **APPROVED** or
**DECLINED** decisions. The training labels are not repayment or default
outcomes. The UI states this before collecting answers and again with the
result. This is an academic prototype, not an operational lending service.

The chatbot asks exactly seven questions: age, employment status, exact
monthly take-home income, two-wheeler type (motorcycle, scooter, moped, or
electric), exact requested loan amount, exact vehicle price, and whether there
is an existing active loan. Applicants can go back between questions and edit
any answer in the seven-answer review before confirming submission. The exact
loan and vehicle prices determine `LTV = 100 × loan amount / vehicle price`.
Numeric answers are used as entered; no ranges or representative amounts are
used. Gender, Pincode, name, and contact details are not collected.

The existing saved model still uses qualification, residence type, area tier,
vehicle make, exact model, and model variant in addition to the answers above.
The shorter form does not ask those questions: their model inputs are left
missing and passed through the saved preprocessing pipeline's imputers and
unknown-category handling. The model was not retrained for this questionnaire,
so estimates may be less reliable; the UI discloses this with each result.
`Final_Tier` is location-related and remains a model input despite not being
collected. Review its availability and proxy-discrimination risks before any
use beyond this academic prototype.

The binary active-loan answer maps `Yes` to `PAST_LOANS_ACTIVE` and `No` to
`NO_PAST_LOANS`, matching the training categories. The assignment describes
this as whether a prior loan is currently active, while the negative category
literally says no past loans; the source data has no separate category for
previously settled loans. Confirm that definition before using the question
outside this project.

## Run locally on Windows

The saved model uses a scikit-learn pipeline and LightGBM estimator serialized
with joblib. Use Python 3.12 and the exact dependency versions in
`chatbot/requirements.txt`; the application checks the model-critical versions
before loading the bundle and returns a clear startup error if they do not match.
Model loading, preprocessing, probability calibration, and thresholding are
shared by the chatbot and standalone inference helper.

From the project root:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r chatbot\requirements.txt
python chatbot\build_catalog.py
python -m uvicorn chatbot.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open <http://127.0.0.1:8000>. The model and API run locally. The UI uses system
fonts and does not load third-party scripts or fonts.

To verify loading, helper/API prediction consistency, and audit output, run:

```powershell
python -m unittest discover -s tests -v
node --check chatbot\static\app.js
node tests\test_chatbot_ui_flow.js
```

The JavaScript harness exercises all seven answers, back navigation, answer
edits, LTV recalculation, and the confirmation step. Python tests cover
saved-model inference with omitted inputs, API prediction, threshold behavior,
audit records, and batch scoring.

## Batch scoring for a new CSV

Run the saved model against an unseen CSV from the project root:

```powershell
python ABC_Credit\results_v6_local\predict_batch.py new_applications.csv predictions.csv
```

The input must include `ID` and every raw feature named by the saved bundle.
Extra columns, including historical `Decision`, `Pincode`, and `Gender`, are
ignored. Missing feature cells and previously unseen categories go through the
saved preprocessing pipeline; missing feature columns are rejected so the
script never invents answers. The output contains the original IDs plus decline
probability, approval probability, and the historical `APPROVED`/`DECLINED`
classification. It omits all other input fields from the output.

Batch scoring uses the same calibrated probability and saved threshold as the
chatbot. It does not retrain the model. Validate output before operational use;
the decision label reflects past application decisions, not repayment ability.

## Flow and audit record

The API validates age, category values, exact amount fields, and the derived
LTV before it invokes the saved preprocessing and model pipeline. It returns
the model probabilities, the bundle's decline threshold, the selected
historical-decision label, model latency, and a pseudonymous session reference.

After consent, each completed submission is appended to
`chatbot/data/audit/applications.jsonl`. The record contains the seven answers,
model-ready feature values (including missing inputs for unasked features),
score, threshold, decision, and latency for project audit. The chatbot does
not collect names, phone numbers, exact postal codes, account IDs, or IP
addresses. Audit rows still contain sensitive financial and demographic
answers: keep the local folder access-controlled, do not share the log, and
delete it when it is no longer needed. The audit directory is excluded from
Git by the project `.gitignore`.

## Production boundary

This local demo has no account system, encrypted database, formal retention
workflow, institution review, or public deployment hardening. Do not expose it
to the internet or use it to determine real applicants' access to credit.
Real use would require repayment-outcome labels, independent temporal and
holdout evaluation, fairness and compliance review, secure storage and access
controls, and a human decision process.
