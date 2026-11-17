# ABC Credit Loan Approval Chatbot

Academic project containing reproducible training and inference code, a saved
LightGBM model, the FastAPI chatbot, tests, and the final academic report. The
source dataset and training notebook remain local and are intentionally ignored
by Git to avoid publishing applicant-level data or notebook outputs.

## Project folders

| Path | Contents |
| --- | --- |
| `chatbot/` | Browser interface, API, choice catalog, dependency list, and run instructions. |
| `ABC_Credit/results_v6_local/` | Saved model bundle, internal test metrics, validation comparisons, inference helper, and batch predictor. |
| `tests/` | Model/API, batch scoring, and chatbot answer-edit flow checks. |
| `ABC_Credit_Model_Training.ipynb` | Local training and exploration notebook (not pushed). |
| `training_data.csv` | Supplied labeled source data (local only; not pushed). |
| `train_credit_model.py` | Reproducible cleaning, feature, training, calibration, and evaluation script. |
| `credit_model_runtime.py` | Shared saved-model loading and prediction code. |
| `ABC_Credit_Final_Academic_Report_Updated.pdf` | Final report with verified internal results and stated limitations. |

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

Then open <http://127.0.0.1:8000>. The chatbot collects 11 answers, lets the
applicant go back or edit answers, reviews all answers before submission, and
recalculates dependent values such as LTV. It uses the saved model bundle,
preprocessing pipeline, and threshold; it does not retrain at application time.

## Verify the project

```powershell
python -m unittest discover -s tests -v
node --check chatbot\static\app.js
node tests\test_chatbot_ui_flow.js
```

## Deployment

The pinned LightGBM Linux wheel requires the system library `libgomp.so.1`.
Vercel's standard Python Function does not provide a supported system-package
installation step, so that backend cannot load the model there. Docker is not
used. The prepared alternative is Railway's native Railpack builder, configured
in `railpack.json` to install `libgomp1` and serve the same FastAPI UI and API.
See [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md) and
[VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md).

The hosted version is synthetic-data only and does not persist audit records.

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
