# Railway deployment without a Dockerfile

Railway's Railpack build configuration adds Debian's `libgomp1` runtime package
and runs the existing FastAPI application. The same app serves the current
HTML/CSS/JavaScript at `/` and its API at `/api/*`; no UI rewrite, separate
frontend service, CORS setup, or model change is needed.

## Deploy from GitHub

1. In Railway, create a project and choose **Deploy from GitHub Repo**.
2. Select `zda23b024/-ABC-credit-loan-approval-chatbot` and branch `main`.
3. Set the service root directory to `/` (the repository root). Railway should
   detect the Python provider and read `railpack.json`.
4. Confirm the deployment configuration uses Python 3.12, installs the runtime
   APT package `libgomp1`, and starts with
   `uvicorn chatbot.app:app --host 0.0.0.0 --port $PORT`.
5. Generate a Railway domain and check `/`, `/api/health`, and `/api/catalog`.
   Health should report `deployment_mode: synthetic_demo` and
   `persistent_audit_logging: false`.
6. Test only with fictional applicant answers.

Before pushing the Railpack migration, pause or disconnect the existing Vercel
Git integration. Vercel's standard Python runtime will still fail on the missing
system library; the Railway configuration does not change what that runtime
provides.

No secrets or custom environment variables are required. Railway sets
`RAILWAY_ENVIRONMENT_NAME`; the app uses that signal to disable local JSONL
logging and mark hosted assessment as synthetic-only. The service listens on
Railway's assigned `$PORT`.

## Linux verification

`.github/workflows/linux-smoke.yml` runs on GitHub's Ubuntu 22.04 runner. It
installs `libgomp1` through apt, installs the exact pinned requirements, checks
`import lightgbm`, loads the saved model and threshold, and runs the full Python
and JavaScript tests. This is the deployment-equivalent native Linux check; no
Docker commands or Dockerfile are used.

Run equivalent checks on an Ubuntu 22.04 host with Python 3.12 and Node.js:

```bash
sudo apt-get update
sudo apt-get install --yes libgomp1
python -m pip install -r requirements.txt
python -c "import lightgbm; print(lightgbm.__version__)"
python -c "from credit_model_runtime import load_model_bundle; print(load_model_bundle()['threshold_declined'])"
python -m unittest discover -s tests -v
node --check chatbot/static/app.js
node tests/test_chatbot_ui_flow.js
```

The trained LightGBM model bundle, preprocessing, calibrator, and threshold
`0.13995426939656921` are unchanged. No source dataset or applicant audit data
is required in the deployment package. Hosted results are not persistently
logged, and the model predicts historical application labels rather than
repayment outcomes.
