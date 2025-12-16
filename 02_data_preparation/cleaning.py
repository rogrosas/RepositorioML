
import pandas as pd

def drop_high_missing(df, threshold: float = 0.80):
    pct = df.isnull().mean(); keep = pct[pct <= threshold].index
    return df[keep].copy()

def simple_impute(df):
    df = df.copy()
    num = df.select_dtypes(include=['number']).columns
    cat = df.select_dtypes(include=['object','category']).columns
    for c in num:
        if not df[c].isnull().all():
            df[c] = df[c].fillna(df[c].median())
    for c in cat:
        m = df[c].mode(dropna=True)
        df[c] = df[c].fillna(m.iloc[0] if len(m) else 'Unknown')
    return df

def basic_clean_application(app):
    df = drop_high_missing(app, 0.90)
    df = simple_impute(df)
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, pd.NA)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())
    return df

def downcast_numeric(df):
    """Reduce uso de memoria downcasteando tipos numéricos de forma segura."""
    result = df.copy()
    for col in result.select_dtypes(include=['float64']).columns:
        result[col] = pd.to_numeric(result[col], downcast='float')
    for col in result.select_dtypes(include=['int64']).columns:
        result[col] = pd.to_numeric(result[col], downcast='integer')
    return result
