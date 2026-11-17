# ABC Credit application assistant

A local, browser-based chatbot that calls the saved model bundle at
`ABC_Credit/results_v6_local/abc_credit_model_bundle.joblib`.

## Scope

The model predicts whether an application resembles historical **APPROVED** or
**DECLINED** decisions. The training labels are not repayment or default
outcomes. The UI states this before collecting answers and again with the
result. This is an academic prototype, not an operational lending service.

The chatbot asks eleven short prompts: age range, highest qualification,
employment or income type, monthly net-income range, residence status, broad
area tier, vehicle type, exact vehicle model, loan-amount range, vehicle-price
range, and whether there is an active prior loan. Each answer stays selected in
the chat composer until the applicant presses **Send**. A text field stays at
the bottom of the chat for typed answers; suggestions appear above it and fill
the draft when tapped. Applicants can go back between questions or edit any
answer in the 11-answer review before confirming submission. Changing vehicle
type clears the previously selected model, and changing loan amount asks for
vehicle price again so calculated LTV uses the current answers. The vehicle
catalog filters model choices by selected vehicle type and maps the model to product,
make, and variant. Numeric ranges map to representative values, while typed
numeric answers are passed as entered; LTV is
calculated as `100 × representative loan amount / representative vehicle
price`. The range-based result is approximate. Gender, Pincode, and ID are not
collected; the saved model does not use those fields. The saved model does use
`Final_Tier`, so the UI asks for a broad area tier without an
exact address. `Final_Tier` is location-related and should be checked for
application-time availability and proxy discrimination; the assignment notebook
also flags it for review.

Typed numeric answers pass through unchanged. Range choices use the server's
single set of representative values, and the result identifies each value used;
vehicle price is only used to derive `LTV = 100 × loan amount / vehicle price`.
The binary prior-loan answer maps `Yes` to `PAST_LOANS_ACTIVE` and `No` to
`NO_PAST_LOANS`, matching the training categories. The assignment describes
this field as whether a prior loan is currently active, while the negative
category name literally says no past loans; the source data has no separate
category for previously settled loans. Confirm that definition before using the
question outside this project.

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

The JavaScript harness exercises all 11 answers, back navigation, answer edits,
dependent vehicle/model and loan/price updates, LTV recalculation, and the
confirmation step. Python tests cover saved-model inference, API prediction,
threshold behavior, audit records, and batch scoring.

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

The API validates age, category values, amount fields, vehicle model, and the
derived LTV before it invokes the saved preprocessing and model pipeline. It
returns the model probabilities, the bundle's decline threshold, the selected
historical-decision label, model latency, and a pseudonymous session reference.

After consent, each completed submission is appended to
`chatbot/data/audit/applications.jsonl`. The record contains the answers,
model-ready feature values, score, threshold, decision, and latency for project
audit. The chatbot does not collect names, phone numbers, exact postal codes,
account IDs, or IP addresses. Audit rows still contain sensitive financial and
demographic answers: keep the local folder access-controlled, do not share the
log, and delete it when it is no longer needed. The audit directory is excluded
from Git by the project `.gitignore`.

## Production boundary

This local demo has no account system, encrypted database, formal retention
workflow, institution review, or public deployment hardening. Do not expose it
to the internet or use it to determine real applicants' access to credit.
Real use would require repayment-outcome labels, independent temporal and
holdout evaluation, fairness and compliance review, secure storage and access
controls, and a human decision process.
