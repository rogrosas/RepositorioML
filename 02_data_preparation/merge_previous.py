
import pandas as pd

def _agg_previous_application(previous: pd.DataFrame) -> pd.DataFrame:
    agg = {
        'AMT_APPLICATION':['mean','median','min','max','sum'],
        'AMT_CREDIT':['mean','median','min','max','sum'],
        'AMT_GOODS_PRICE':['mean','median','min','max','sum']
    }
    agg = {k:v for k,v in agg.items() if k in previous.columns}
    base = previous.groupby('SK_ID_CURR').size().rename('PREV_COUNT')
    prev_num = previous.groupby('SK_ID_CURR').agg(agg) if agg else pd.DataFrame(index=previous['SK_ID_CURR'].unique())
    if len(prev_num.columns):
        prev_num.columns = ["PREV_" + "_".join(c) for c in prev_num.columns]
    parts = [base, prev_num]
    if 'NAME_CONTRACT_STATUS' in previous.columns:
        parts.append(previous['NAME_CONTRACT_STATUS'].eq('Approved').groupby(previous['SK_ID_CURR']).sum().rename('PREV_APPROVED_COUNT'))
        parts.append(previous['NAME_CONTRACT_STATUS'].eq('Refused').groupby(previous['SK_ID_CURR']).sum().rename('PREV_REFUSED_COUNT'))
    diversity = []
    for cat in ['NAME_CONTRACT_TYPE','CHANNEL_TYPE','NAME_YIELD_GROUP','PRODUCT_COMBINATION']:
        if cat in previous.columns:
            diversity.append(previous.groupby('SK_ID_CURR')[cat].nunique().rename(f"PREV_{cat}_NUNIQUE"))
    if diversity:
        parts.append(pd.concat(diversity, axis=1))
    return pd.concat(parts, axis=1)

def _agg_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    agg = {}
    if 'MONTHS_BALANCE' in pos.columns: agg['MONTHS_BALANCE'] = ['count','min','max','mean']
    if 'SK_DPD' in pos.columns: agg['SK_DPD'] = ['mean','max']
    if 'SK_DPD_DEF' in pos.columns: agg['SK_DPD_DEF'] = ['mean','max']
    g = pos.groupby('SK_ID_PREV').agg(agg) if agg else pd.DataFrame(index=pos['SK_ID_PREV'].unique())
    if len(g.columns): g.columns = ["POS_" + "_".join(c) for c in g.columns]
    if 'NAME_CONTRACT_STATUS' in pos.columns:
        g = g.join(pos.groupby('SK_ID_PREV')['NAME_CONTRACT_STATUS'].nunique().rename('POS_STATUS_NUNIQUE'), how='left')
    return g

def _agg_installments(inst: pd.DataFrame) -> pd.DataFrame:
    df = inst.copy()
    if {'AMT_PAYMENT','AMT_INSTALMENT'}.issubset(df.columns):
        df['PAYMENT_DIFF'] = df['AMT_PAYMENT'] - df['AMT_INSTALMENT']
        df['PAYMENT_RATIO'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT'].replace(0, pd.NA)
    if {'DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT'}.issubset(df.columns):
        df['DAYS_DIFF'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    agg = {
        'PAYMENT_DIFF':['mean','min','max'],
        'PAYMENT_RATIO':['mean'],
        'DAYS_DIFF':['mean','min','max'],
        'AMT_PAYMENT':['mean','sum'],
        'AMT_INSTALMENT':['mean','sum']
    }
    agg = {k:v for k,v in agg.items() if k in df.columns}
    g = df.groupby('SK_ID_PREV').agg(agg) if agg else pd.DataFrame(index=df['SK_ID_PREV'].unique())
    if len(g.columns): g.columns = ["INST_" + "_".join(c) for c in g.columns]
    return g

def _agg_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    df = cc.copy()
    if {'AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL'}.issubset(df.columns):
        df['LIMIT_UTILIZATION'] = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, pd.NA)
    agg = {
        'AMT_BALANCE':['mean','max'],
        'AMT_CREDIT_LIMIT_ACTUAL':['mean','max'],
        'LIMIT_UTILIZATION':['mean','max'],
        'MONTHS_BALANCE':['count','min','max','mean']
    }
    agg = {k:v for k,v in agg.items() if k in df.columns}
    g = df.groupby('SK_ID_PREV').agg(agg) if agg else pd.DataFrame(index=df['SK_ID_PREV'].unique())
    if len(g.columns): g.columns = ["CC_" + "_".join(c) for c in g.columns]
    return g

def agregar_previous_features(previous: pd.DataFrame, pos_cash=None, installments=None, credit_card=None) -> pd.DataFrame:
    prev_by_client = _agg_previous_application(previous)
    parts = []
    if pos_cash is not None and len(pos_cash): parts.append(_agg_pos_cash(pos_cash))
    if installments is not None and len(installments): parts.append(_agg_installments(installments))
    if credit_card is not None and len(credit_card): parts.append(_agg_credit_card(credit_card))
    if parts:
        by_prev = parts[0]
        for p in parts[1:]: by_prev = by_prev.join(p, how='outer')
        map_prev = previous[['SK_ID_PREV','SK_ID_CURR']].drop_duplicates()
        by_prev_client = by_prev.join(map_prev.set_index('SK_ID_PREV'), how='left')
        by_client = by_prev_client.groupby('SK_ID_CURR').agg('mean')
        by_client.columns = ["PREV_BEHAV_" + c for c in by_client.columns]
        final = prev_by_client.join(by_client, how='left')
    else:
        final = prev_by_client
    return final.sort_index()

def guardar_parquet(df, path: str):
    df.to_parquet(path, index=True)
