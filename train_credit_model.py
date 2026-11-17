"""Reproducible local training run for the ABC Credit project.

Predicts historical APPROVED/DECLINED decisions. It does not predict repayment
or default. Run: python train_credit_model.py
"""
from pathlib import Path
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    precision_recall_curve, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "training_data.csv"
OUT = ROOT / "ABC_Credit" / "results_v6_local"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_DECLINED_PRECISION = .20

FULL = ['Qualifications','Employment_Type','Age','Product_Code','Loan_Amount',
        'LTV','Resident_Type','Net_salary','Final_Tier','Make_Code',
        'Model_Description','Model_Variant','PAST_LOANS_ACTIVE']
COMPACT = ['Employment_Type','Age','Product_Code','Loan_Amount','LTV',
           'Net_salary','PAST_LOANS_ACTIVE','Resident_Type']
CORE = ['Employment_Type','Age','Loan_Amount','LTV','Net_salary','PAST_LOANS_ACTIVE']
NO_VEHICLE = [c for c in FULL if c not in ['Make_Code','Model_Description','Model_Variant']]
SETS = {'full': FULL, 'compact': COMPACT, 'core': CORE, 'no_vehicle': NO_VEHICLE}
NUMERIC = ('Age','Loan_Amount','LTV','Net_salary')


def make_features(frame, base_columns):
    x = pd.DataFrame(frame).reindex(columns=base_columns).copy()
    for c in NUMERIC:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors='coerce')
            if c == 'Age': x.loc[~x[c].between(18,60),c] = np.nan
            elif c == 'Loan_Amount': x.loc[x[c] <= 0,c] = np.nan
            elif c == 'LTV': x.loc[~x[c].between(.01,150),c] = np.nan
            elif c == 'Net_salary': x.loc[x[c] < 0,c] = np.nan
    for c in NUMERIC:
        if c in x: x[c+'_missing'] = x[c].isna().astype('int8')
    if 'Loan_Amount' in x: x['log_loan_amount'] = np.log1p(x.Loan_Amount.clip(lower=0))
    if 'Net_salary' in x: x['log_net_salary'] = np.log1p(x.Net_salary.clip(lower=0))
    if {'Loan_Amount','Net_salary'}.issubset(x.columns):
        x['loan_to_income'] = (x.Loan_Amount/x.Net_salary.replace(0,np.nan)).clip(upper=100)
        x['annual_income_to_loan'] = (12*x.Net_salary/x.Loan_Amount.replace(0,np.nan)).clip(upper=10000)
    if {'Loan_Amount','LTV'}.issubset(x.columns):
        x['estimated_vehicle_price'] = (100*x.Loan_Amount/x.LTV.replace(0,np.nan)).clip(upper=1e7)
    if 'Age' in x:
        x['age_group'] = pd.cut(x.Age,bins=[17,25,35,45,60],labels=['18-25','26-35','36-45','46-60']).astype('string').fillna('__MISSING__').astype(str)
    if 'Employment_Type' in x and 'Net_salary' in x:
        x['employment_income_group'] = x['Employment_Type'].astype('string').fillna('__MISSING__').astype(str)+'_'+pd.cut(x.Net_salary,bins=[-1,10000,20000,40000,float('inf')],labels=['low','mid','high','very_high']).astype('string').fillna('__MISSING__').astype(str)
    numeric = set(NUMERIC)|{'loan_to_income','annual_income_to_loan','estimated_vehicle_price','log_loan_amount','log_net_salary'}|{c+'_missing' for c in NUMERIC}
    for c in x.columns:
        if c not in numeric:
            x[c] = x[c].astype('string').str.strip().fillna('__MISSING__').astype(str)
            x[c] = x[c].replace({'':'__MISSING__','nan':'__MISSING__','None':'__MISSING__','<NA>':'__MISSING__'})
    return x


def main():
    if not CSV.exists(): raise FileNotFoundError(f'Missing input file: {CSV}')
    expected = ['ID','Qualifications','Employment_Type','Age','Pincode','Gender','Product_Code',
        'Loan_Amount','LTV','Resident_Type','Net_salary','Final_Tier','Make_Code',
        'Model_Description','Model_Variant','PAST_LOANS_ACTIVE','Decision']
    df = pd.read_csv(CSV, low_memory=False).reindex(columns=expected)
    for c in df.select_dtypes(include=['object','string']).columns:
        df[c] = df[c].astype('string').str.strip().replace(['','nan','None','NULL','N/A'],pd.NA)
    df.Decision = df.Decision.str.upper()
    df = df.loc[df.Decision.isin(['APPROVED','DECLINED'])].copy()
    for c in NUMERIC: df[c] = pd.to_numeric(df[c], errors='coerce')
    invalid_age = df.Age.notna() & ~df.Age.between(18,60)
    df = df.loc[~invalid_age].copy()
    for col,bad in [('Loan_Amount',df.Loan_Amount<=0),('LTV',~df.LTV.between(.01,150)),('Net_salary',df.Net_salary<0)]:
        df.loc[bad & df[col].notna(),col] = np.nan

    train, remain = train_test_split(df,test_size=.30,stratify=df.Decision,random_state=SEED)
    validation, test = train_test_split(remain,test_size=.50,stratify=remain.Decision,random_state=SEED)
    cal, select = train_test_split(validation,test_size=.60,stratify=validation.Decision,random_state=SEED)
    ys = [(part.Decision=='DECLINED').astype(int).to_numpy() for part in (train,cal,select,test)]
    y_train,y_cal,y_sel,y_test = ys
    experiments = [
        # Each tuple is (name, feature set, leaves, min child, L2, declined-class weight).
        ('full_regularized_w1','full',24,100,8,1),
        ('full_regularized_w2','full',24,100,8,2),
        ('full_regularized_w4','full',24,100,8,4),
        ('full_regularized_w8','full',24,100,8,8),
        ('full_regularized_w18','full',24,100,8,18),  # approximate inverse class frequency
        ('full_smooth_w1','full',12,180,12,1),
        ('compact_regularized_w1','compact',24,100,8,1),
        ('compact_regularized_w2','compact',24,100,8,2),
        ('compact_regularized_w4','compact',24,100,8,4),
        ('compact_regularized_w8','compact',24,100,8,8),
        ('compact_regularized_w18','compact',24,100,8,18),
        ('compact_smooth_w1','compact',12,180,12,1),
        ('core_regularized_w1','core',24,100,8,1),
        ('no_vehicle_regularized_w1','no_vehicle',24,100,8,1)]
    results=[]; fitted={}; probabilities={}; timings={}
    for name,fs,leaves,min_child,reg,weight in experiments:
        xtr=make_features(train,SETS[fs]); xcal=make_features(cal,SETS[fs]); xsel=make_features(select,SETS[fs])
        nums=xtr.select_dtypes(include='number').columns.tolist(); cats=[c for c in xtr if c not in nums]
        enc=ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=True),nums),
            ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=30),cats)],sparse_threshold=1)
        model=Pipeline([('prep',enc),('model',lgb.LGBMClassifier(n_estimators=650,learning_rate=.03,
            num_leaves=leaves,min_child_samples=min_child,max_depth=-1,colsample_bytree=.85,
            reg_lambda=reg,reg_alpha=1,scale_pos_weight=weight,verbosity=-1,n_jobs=-1,random_state=SEED))])
        start=time.perf_counter(); model.fit(xtr,y_train)
        pcal=np.asarray(model.predict_proba(xcal)[:,1]); psel=np.asarray(model.predict_proba(xsel)[:,1])
        timings[name]=time.perf_counter()-start; fitted[name]=model; probabilities[name]=(pcal,psel,fs,weight)
        results.append({'model':name,'feature_set':fs,'declined_class_weight':weight,'pr_auc':average_precision_score(y_sel,psel),
            'roc_auc':roc_auc_score(y_sel,psel),'brier_uncalibrated':brier_score_loss(y_sel,psel),
            'fit_seconds':timings[name]})
        print(f'Trained {name} in {timings[name]:.1f}s; validation PR-AUC={results[-1]["pr_auc"]:.4f}')
    comparison=pd.DataFrame(results).sort_values('pr_auc',ascending=False)
    comparison.to_csv(OUT/'model_comparison_validation.csv',index=False)

    threshold_rows=[]; calibrators={}
    for candidate in comparison.model:
        pcal,psel,fs,weight=probabilities[candidate]
        cal_model=LogisticRegression(C=10,max_iter=500,random_state=SEED)
        cal_model.fit(logit(np.clip(pcal,1e-6,1-1e-6)).reshape(-1,1),y_cal)
        if cal_model.coef_[0,0] <= 0: cal_model=None
        calibrators[candidate]=cal_model
        p=cal_model.predict_proba(logit(np.clip(psel,1e-6,1-1e-6)).reshape(-1,1))[:,1] if cal_model is not None else psel
        prec_curve, rec_curve, candidates = precision_recall_curve(y_sel,p)
        positives=int(y_sel.sum()); negatives=len(y_sel)-positives
        pr=average_precision_score(y_sel,p); auc=roc_auc_score(y_sel,p); brier=brier_score_loss(y_sel,p)
        for t,prec,rec in zip(candidates,prec_curve[:-1],rec_curve[:-1]):
            tp=rec*positives; fp=(tp/prec-tp) if prec>0 else 0.0
            fp=float(np.clip(fp,0,negatives)); tn=negatives-fp
            f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
            threshold_rows.append({'model':candidate,'feature_set':fs,'declined_class_weight':weight,'threshold':float(t),
                'accuracy':(tn+tp)/len(y_sel),'declined_precision':float(prec),
                'declined_recall':float(rec),'declined_f1':float(f1),
                'approved_recall':tn/negatives,'false_decline_rate':fp/negatives,
                'pr_auc':pr,'roc_auc':auc,'brier':brier})
    thresholds=pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT/'threshold_sweep_validation.csv',index=False)
    feasible=thresholds.loc[(thresholds.declined_precision>=MIN_DECLINED_PRECISION)&(thresholds.declined_recall>0)]
    if feasible.empty: selected=thresholds.loc[thresholds.declined_recall>0].sort_values(['declined_recall','declined_precision','pr_auc'],ascending=False).iloc[0]
    else: selected=feasible.sort_values(['declined_recall','declined_precision','pr_auc'],ascending=False).iloc[0]
    name=selected.model; fs=selected.feature_set; threshold=float(selected.threshold)
    model=fitted[name]; cal_model=calibrators[name]
    selected_weight=int(selected.declined_class_weight)
    xtest=make_features(test,SETS[fs]); raw_p=model.predict_proba(xtest)[:,1]
    p=cal_model.predict_proba(logit(np.clip(raw_p,1e-6,1-1e-6)).reshape(-1,1))[:,1] if cal_model is not None else raw_p
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    all_approve=int(np.sum(y_test==0)); base_accuracy=all_approve/len(y_test)
    metrics={'model':name,'feature_set':fs,'declined_class_weight':selected_weight,'threshold':threshold,
        'threshold_policy':f'Validation max declined recall subject to declined precision >= {MIN_DECLINED_PRECISION:.0%}',
        'test_n':len(test),'test_declined_rate':float(y_test.mean()),
        'accuracy':accuracy_score(y_test,pred),'approve_all_baseline_accuracy':base_accuracy,
        'declined_precision':precision_score(y_test,pred,zero_division=0),
        'declined_recall':recall_score(y_test,pred,zero_division=0),
        'declined_f1':f1_score(y_test,pred,zero_division=0),
        'approved_precision':precision_score(y_test,pred,pos_label=0,zero_division=0),
        'approved_recall':tn/(tn+fp),'false_decline_rate':fp/(tn+fp),
        'pr_auc':average_precision_score(y_test,p),'roc_auc':roc_auc_score(y_test,p),
        'brier':brier_score_loss(y_test,p),'tn_approved_as_approved':int(tn),
        'fp_approved_flagged_declined':int(fp),'fn_declined_missed':int(fn),
        'tp_declined_flagged':int(tp),'total_actual_declines':int(y_test.sum()),
        'fit_seconds':timings[name]}
    (OUT/'final_test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    pd.DataFrame([metrics]).to_csv(OUT/'final_test_metrics.csv',index=False)
    pd.DataFrame({'actual':np.where(y_test==1,'DECLINED','APPROVED'),'decline_probability':p,
        'predicted_decision':np.where(pred==1,'DECLINED','APPROVED')}).to_csv(OUT/'internal_test_predictions.csv',index=False)
    joblib.dump({'model':model,'calibrator':cal_model,'model_name':name,'features':SETS[fs],
        'feature_set':fs,'threshold_declined':threshold,'declined_class_weight':selected_weight,
        'positive_class':'DECLINED','seed':SEED,
        'version':'local-retraining-1'},OUT/'abc_credit_model_bundle.joblib')
    print('\nValidation model ranking:')
    print(comparison.to_string(index=False))
    print('\nHeld-out internal test metrics:')
    print(json.dumps(metrics,indent=2))
    print(f'\nArtifacts: {OUT}')

if __name__ == '__main__': main()
