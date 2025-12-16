
import pandas as pd

def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'DAYS_BIRTH' in d.columns: d['AGE_YEARS'] = (-d['DAYS_BIRTH']/365.25).astype('float')
    if 'DAYS_EMPLOYED' in d.columns: d['EMPLOYED_YEARS'] = (-d['DAYS_EMPLOYED']/365.25).astype('float')
    if {'AMT_CREDIT','AMT_INCOME_TOTAL'}.issubset(d.columns): d['CREDIT_TO_INCOME'] = d['AMT_CREDIT']/d['AMT_INCOME_TOTAL'].replace(0, pd.NA)
    if {'AMT_ANNUITY','AMT_INCOME_TOTAL'}.issubset(d.columns): d['ANNUITY_TO_INCOME'] = d['AMT_ANNUITY']/d['AMT_INCOME_TOTAL'].replace(0, pd.NA)
    if {'AMT_GOODS_PRICE','AMT_INCOME_TOTAL'}.issubset(d.columns): d['GOODS_TO_INCOME'] = d['AMT_GOODS_PRICE']/d['AMT_INCOME_TOTAL'].replace(0, pd.NA)
    if {'AMT_CREDIT','AMT_GOODS_PRICE'}.issubset(d.columns): d['CREDIT_TO_GOODS'] = d['AMT_CREDIT']/d['AMT_GOODS_PRICE'].replace(0, pd.NA)
    ext = [c for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in d.columns]
    if ext:
        d['EXT_SOURCE_SUM'] = d[ext].sum(axis=1)
        d['EXT_SOURCE_MEAN'] = d[ext].mean(axis=1)
    return d

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if {'CREDIT_TO_INCOME','BUREAU_AMT_CREDIT_SUM_mean'}.issubset(d.columns):
        d['CRED_INCOME_VS_BUREAU'] = d['CREDIT_TO_INCOME'] / d['BUREAU_AMT_CREDIT_SUM_mean'].replace(0, pd.NA)
    if {'AGE_YEARS','BUREAU_CREDIT_DAY_OVERDUE_mean'}.issubset(d.columns):
        d['OVERDUE_PER_YEAR'] = d['BUREAU_CREDIT_DAY_OVERDUE_mean'] / d['AGE_YEARS'].replace(0, pd.NA)
    return d

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_application_features(df)
    d = add_interactions(d)
    return d
