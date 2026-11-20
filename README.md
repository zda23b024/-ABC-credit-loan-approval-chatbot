# ABC Credit Loan Approval Chatbot

Academic project containing the training data and notebook, reproducible
training and inference code, a saved LightGBM model, a local FastAPI chatbot,
tests, and the final academic report.

## Project folders

| Path | Contents |
| --- | --- |
| `chatbot/` | Browser interface, API, choice catalog, dependency list, and run instructions. |
| `ABC_Credit/results_v6_local/` | Saved model bundle, internal test metrics, validation comparisons, inference helper, and batch predictor. |
| `tests/` | Model/API, batch scoring, and chatbot answer-edit flow checks. |
| `ABC_Credit_Model_Training.ipynb` | Training and exploration notebook. |
| `training_data.csv` | Supplied labeled training data. |
| `train_credit_model.py` | Reproducible cleaning, feature, training, calibration, and evaluation script. |
| `credit_model_runtime.py` | Shared saved-model loading and prediction code. |
| `MODEL_CARD.md` | Responsible AI model card, including current chatbot-input and subgroup limitations. |
| `ABC_Credit_Overleaf_Report.pdf` | Academic report. |

## Run locally

Use Python 3.12 and dependencies in `chatbot/requirements.txt`. From this
folder in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r chatbot\requirements.txt
python chatbot\build_catalog.py
python -m uvicorn chatbot.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

Then open <http://127.0.0.1:8000>. The chatbot collects seven answers: age,
employment status, exact monthly take-home income, two-wheeler type, exact
requested loan amount, exact vehicle price, and existing active-loan status.
Applicants can go back or edit answers and review them before submission. The
exact amounts determine LTV. It uses the saved model bundle, preprocessing
pipeline, and threshold; it does not retrain at application time.

The saved model also uses qualification, residence type, area tier, vehicle
make, model, and variant. Since the seven-question form does not collect these,
the API leaves them missing for the saved preprocessing pipeline to impute and
discloses this limitation with each assessment. The model was not retrained for
the shorter questionnaire.

## Verify the project

```powershell
python -m unittest discover -s tests -v
node --check chatbot\static\app.js
node tests\test_chatbot_ui_flow.js
```

## Results and scope

The saved model's 16,098-record internal test accuracy is 90.88%, compared with
94.73% for an always-APPROVED baseline. DECLINED precision is 20.10% and recall
is 24.50%. These metrics measure agreement with historical APPROVED/DECLINED
decisions, not repayment or default. The assignment's separate blind holdout
was not available. Proposed completion and response targets are X = 3 minutes
and Y = 2 seconds; repeated browser timing results remain pending and are not
claimed as met.

Keep `chatbot/data/audit/` out of submission archives: local JSONL audit rows
contain sensitive financial and demographic answers. The model bundle and
source code are preserved as supplied in this project.
