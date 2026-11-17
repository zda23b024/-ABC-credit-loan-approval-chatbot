# Vercel deployment guide

This project deploys the existing FastAPI application as a single Vercel Python
Function. The existing HTML/CSS/JavaScript is served by that same application;
`/api/catalog`, `/api/health`, and `/api/decision` stay on the same origin, so no
CORS configuration is needed. The trained model, calibrator, preprocessing
pipeline, and decision threshold are unchanged.

## Connect GitHub to Vercel

1. Put this project in a **private GitHub repository**. Keep `.gitignore` and
   `.vercelignore`; do not commit local audit rows or real applicant data.
2. In Vercel, choose **Add New → Project**, connect GitHub, and import the repo.
3. Set the **Root Directory** to the repository root. Leave the framework
   preset on **FastAPI** (or the detected Python preset); do not set a separate
   build/output directory.
4. Deploy. Vercel reads `.python-version` (3.12), root `requirements.txt`,
   `vercel.json`, and root `app.py`.
5. Check the deployment at `/`, `/api/health`, and `/api/catalog`. The health
   response should say `deployment_mode: synthetic_demo` and
   `persistent_audit_logging: false`.
6. Keep the deployment as a private/preview deployment or enable Vercel
   Deployment Protection if available. Do not invite real applicants or use it
   for lending decisions.

The Git integration will deploy preview deployments for branches and deploy
production from the configured production branch (normally `main`). Review each
preview before promoting it.

## Environment variables

No custom environment variables are required. Vercel supplies `VERCEL=1`, which
selects the hosted synthetic-demo behavior and prevents JSONL file writes.

`ABC_CREDIT_AUDIT_LOG` is optional for **local runs only**; it changes the local
JSONL path. Do not set it on Vercel expecting persistence. No database or other
persistent audit service is configured, so hosted submissions are not logged.

## Hosted data and privacy boundary

The hosted page clearly requires fictional answers and the API does not write
applicant data to disk. Use synthetic examples only. The standard answer fields
are constrained to the supplied choices and numeric ranges. A public deployment
can still receive requests and expose predictions, so keep it access-protected.
The `.vercelignore` file excludes source training data, local audit logs,
notebooks, tests, reports, and non-runtime evaluation outputs. The saved model
is bundled server-side for inference and is not published as a static asset.

The application is still an academic demonstration of historical APPROVED and
DECLINED labels. It is not a repayment/default model or a production lending
decision system. Production use requires appropriate identity/access controls,
privacy review, compliant retention and durable audit storage, fairness and
model validation, rate limiting/abuse controls, human review, and legal approval.

## Runtime and verification

The bundle requires exact model dependency versions in root `requirements.txt`.
Vercel's Python 3.12 runtime is selected, and `vercel.json` explicitly includes
the model bundle, question catalog, and browser assets. The function is limited
to 30 seconds. The model file is under 1 MB; the dependency stack is the larger
bundle component and should be checked in the Vercel build log. Vercel documents
a 500 MB standard uncompressed Python function limit, 2 GB default memory on
Hobby, and a read-only filesystem apart from temporary scratch storage.

Run locally before pushing:

```powershell
python -m pip install -r requirements.txt
python -m unittest tests.test_vercel_deployment -v
python -m unittest discover -s tests -v
node --check chatbot\static\app.js
node tests\test_chatbot_ui_flow.js
```

For an actual Vercel build smoke test, install the Vercel CLI and run `vercel
 dev` from the repository root. A local ASGI test validates the FastAPI entry
point and same-origin paths; it does not replace a real Vercel deployment.
