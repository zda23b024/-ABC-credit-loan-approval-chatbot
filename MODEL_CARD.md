# ABC Credit Historical Decision Classifier — Model Card

**Prepared:** 2026-11-20  
**Project status:** Academic prototype; not approved for operational lending  
**Model version:** `local-retraining-1`  
**Saved model:** `full_smooth_w1` (`feature_set: full`)

This card follows the Model Card fields shown in the Responsible AI lecture:
Model Details, Intended Use, Factors, Metrics, Evaluation Data, Training Data,
Ethical Considerations, and Caveats & Recommendations. It also documents
subgroup and environmental evaluation limits.

## Model Details

- **Task:** Binary classification of historical application decisions:
  `APPROVED` or `DECLINED`.
- **Model type:** LightGBM classifier in a scikit-learn preprocessing
  pipeline, followed by logistic probability calibration and a fixed decision
  threshold.
- **Saved bundle version:** `local-retraining-1`.
- **Positive class:** `DECLINED`.
- **Output:** Calibrated decline probability and a historical-decision label.
  The label is `DECLINED` when probability is at least `0.1399542694`;
  otherwise it is `APPROVED`.
- **Developer/owner:** ABC Credit academic project. Individual developer and
  support contact are not identified in the project materials.
- **Implementation:** Local FastAPI service and browser chatbot. The bundle,
  training code, and results are in this repository.

### Training algorithm and parameters

The selected model is `full_smooth_w1`: 650 LightGBM estimators, learning rate
0.03, 12 leaves, 180 minimum child samples, maximum depth -1, column sampling
0.85, L2 regularization 12, L1 regularization 1, and declined-class weight 1.
The random seed is 42.

Numeric features use median imputation with missingness indicators. Categorical
features use one-hot encoding, ignore unseen categories, and group categories
with fewer than 30 training examples. A logistic regression calibrator
(`C=10`, maximum 500 iterations) is fit on the calibration partition. The
decline threshold is selected on validation data to maximize decline recall
subject to declined precision of at least 20%.

## Intended Use

- **Intended use:** Coursework, research, and controlled demonstration of
  historical-decision classification and responsible model reporting.
- **Intended users:** Students and researchers inspecting the model and its
  limitations.
- **Not intended for:** Real credit eligibility, underwriting, pricing,
  collections, repayment/default prediction, or use as the sole basis for any
  decision affecting an applicant. It does not estimate a person's ability or
  willingness to repay.

The target is a past institutional decision, not a repayment outcome. A
prediction that resembles a historical `APPROVED` or `DECLINED` label is not a
recommendation, guarantee, or offer.

## Factors

### Original model factors

The saved bundle expects these 13 input fields:

`Qualifications`, `Employment_Type`, `Age`, `Product_Code`, `Loan_Amount`,
`LTV`, `Resident_Type`, `Net_salary`, `Final_Tier`, `Make_Code`,
`Model_Description`, `Model_Variant`, and `PAST_LOANS_ACTIVE`.

The training code excludes `ID`, `Pincode`, and `Gender` from prediction.
Excluding those columns does not establish that the model is fair: `Age`,
employment and financial variables, and the location-related `Final_Tier` can
still reflect or proxy group differences.

### Current chatbot factors

The current chatbot asks exactly seven questions: age, employment status,
exact monthly take-home income, two-wheeler type, exact requested loan amount,
exact vehicle price, and whether an existing loan is active. It derives
`LTV = 100 × loan amount / vehicle price`.

To keep the existing saved model, the chatbot leaves these six model inputs
missing: `Qualifications`, `Resident_Type`, `Final_Tier`, `Make_Code`,
`Model_Description`, and `Model_Variant`. The saved preprocessing pipeline
handles them as missing/unseen values. The model was **not retrained** for this
seven-answer input pattern. Each chatbot result discloses this limitation.

## Metrics

The metrics below use the original fixed threshold (`0.1399542694`) and the
same held-out internal test records. The seven-answer figures are a post-hoc
replay with the six unasked model inputs masked; they are not results from a
newly trained model or an independent validation set.

| Metric | Current chatbot: seven-answer replay | Archived full-input model |
|---|---:|---:|
| Test records | 16,098 | 16,098 |
| Actual `DECLINED` rate | 5.27% | 5.27% |
| Accuracy | 91.82% | 90.88% |
| Always-`APPROVED` accuracy baseline | 94.73% | 94.73% |
| `DECLINED` precision | 19.21% | 20.10% |
| `DECLINED` recall | 17.20% | 24.50% |
| PR-AUC | 0.1463 | 0.1524 |
| ROC-AUC | 0.7530 | 0.7608 |
| Brier score | 0.04757 | 0.04734 |

For the seven-answer replay, the confusion counts are 14,635 true approvals,
614 approved applications flagged as declined, 703 declines missed, and 146
declines flagged. Accuracy is below the always-approve baseline; it should not
be presented alone as evidence of usefulness. The decline precision in this
replay is also below the 20% validation constraint used to select the original
threshold.

## Evaluation Data

The project creates a stratified random split with seed 42: 70% training data,
15% validation data, and 15% held-out internal test data. The validation
partition is split again for probability calibration and model/threshold
selection. Invalid-target rows are dropped; labeled ages outside 18–60 are
excluded; invalid loan amount, LTV, and income values are treated as missing.
Missing ages are retained.

The seven-answer replay uses only `Employment_Type`, `Age`, `Product_Code`,
`Loan_Amount`, `LTV`, `Net_salary`, and `PAST_LOANS_ACTIVE` when scoring the
same 16,098 test records. The dataset does not provide a separate customer
vehicle-price field for this replay, so it uses recorded `LTV`; this does not
fully reproduce the chatbot's collection process.

The assignment's separate blind holdout was not available. No temporal,
external-population, or prospective evaluation is reported. The data source's
collection period, sampling process, and decision-labeling policy are not
documented in the supplied materials.

## Training Data

- **Source:** The project-provided `training_data.csv`; its upstream source and
  collection methodology are unspecified.
- **Size:** 108,423 raw rows; 107,317 labeled records remain after the training
  script's target and age filters.
- **Target:** Historical `Decision` values, `APPROVED` or `DECLINED`.
- **Class balance:** The internal test split has 5.27% `DECLINED` records and
  94.73% `APPROVED` records.
- **Preprocessing:** Text categories are stripped and standard missing-value
  spellings are normalized. Ages outside 18–60 are excluded. Nonpositive loan
  amounts, LTV outside 0.01–150%, and negative income are set missing. The
  fitted pipeline imputes numeric values and encodes categories as described
  under Model Details.

The labels describe previous decisions and may contain historical policy,
measurement, or societal biases. They are not ground-truth repayment outcomes.

## Quantitative Analyses

The following are exploratory subgroup results on the internal test split for
the **seven-answer replay**. `Gender` and `Final_Tier` are used only to group
test outcomes; neither is sent to the seven-answer predictor. `Age` is an input.
False-decline rate is the share of actually approved records classified as
declined within that group. Percentages are rounded to two decimals.

| Grouping factor | Group | n | Actual decline rate | Decline precision | Decline recall | False-decline rate |
|---|---|---:|---:|---:|---:|---:|
| Gender | Female | 3,814 | 5.69% | 16.02% | 17.05% | 5.39% |
| Gender | Male | 12,284 | 5.15% | 20.61% | 17.25% | 3.61% |
| Age | 18–25 | 4,169 | 6.07% | 20.49% | 29.64% | 7.43% |
| Age | 26–35 | 5,358 | 5.43% | 17.48% | 17.18% | 4.66% |
| Age | 36–45 | 3,853 | 4.88% | 18.68% | 9.04% | 2.02% |
| Age | 46–60 | 2,715 | 4.27% | 23.53% | 3.45% | 0.50% |
| Area tier | Megapolis (A) | 102 | 0.98% | 0.00% | 0.00% | 0.99% |
| Area tier | Megapolis (B) | 537 | 4.47% | 17.65% | 12.50% | 2.73% |
| Area tier | Large Metro | 201 | 3.98% | 33.33% | 12.50% | 1.04% |
| Area tier | Metro | 410 | 4.88% | 0.00% | 0.00% | 2.82% |
| Area tier | Urban | 914 | 5.80% | 26.47% | 16.98% | 2.90% |
| Area tier | Semi-Urban | 3,165 | 5.43% | 22.45% | 19.19% | 3.81% |
| Area tier | Rural | 10,766 | 5.30% | 18.28% | 17.51% | 4.39% |

These differences warrant further review; they do not prove discrimination or
fairness. Some area-tier groups are small, and no confidence intervals,
significance tests, or subgroup calibration analysis were performed. The
supplied test split contains only Female/Male gender categories; results for
other or unrecorded identities cannot be inferred. Three records have missing
age and three have missing area tier and are not shown in those group tables.

No evaluation was performed across time periods, institutions, regions outside
the supplied data, device/network conditions, or changing lending policies.

## Ethical Considerations

- **Fairness and bias:** Historical decisions can reproduce prior biases.
  Gender and Pincode are excluded from prediction, but this alone does not
  remove bias. Age, financial factors, employment, and geographic proxies
  require review. The subgroup gaps above are a warning, not a fairness
  certification.
- **Transparency:** Clearly identify the predicted label as a historical
  decision, disclose the imputed inputs, and avoid presenting the score as
  repayment risk or creditworthiness.
- **Privacy:** Age, employment, income, loan amount, vehicle price, and active
  loan status are sensitive. Local audit records contain these answers and
  scores; access must be restricted and records retained only as necessary.
- **Human impact:** A false decline can deny or delay access unfairly; a
  historical `APPROVED` label is not evidence that an applicant can repay.
  Outputs must not replace a lawful, accountable human process.

## Caveats & Recommendations

1. **Do not use the seven-answer replay for real credit decisions.** The model
   was trained on 13 features but the chatbot supplies only seven; six are
   imputed. The archived full-input test metrics do not validate this changed
   input pathway.
2. Before any operational consideration, train and evaluate a model designed
   for the actual seven collected inputs, using repayment outcomes where
   appropriate, leakage-safe temporal/external holdouts, and a predeclared
   decision policy.
3. Repeat subgroup evaluation with adequate sample sizes and uncertainty
   intervals. Review false-decline rates, recall, precision, calibration, and
   selection rates by relevant demographic and proxy groups; investigate
   disparities rather than assuming a single metric establishes fairness.
4. Document source provenance, population coverage, collection period, missing
   data, label policy, intended users, and known limitations before submission
   or release.
5. Keep the system local and demonstrative. Do not expose it publicly or use it
   to determine applicants' access to credit.

## References and Project Artifacts

- Mitchell et al., “Model Cards for Model Reporting,” FAT* 2019 (the model-card
  framework cited in the Responsible AI lecture).
- `train_credit_model.py` — training, preprocessing, split, and threshold policy.
- `credit_model_runtime.py` — saved-bundle loading and inference path.
- `ABC_Credit/results_v6_local/final_test_metrics.json` — archived full-input
  internal test metrics.
- `training_data.csv` — project-provided labeled data.
