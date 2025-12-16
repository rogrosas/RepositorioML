
# 02_data_preparation/merge_bureau.py

import pandas as pd

def agregar_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    # --- Agregaciones directas de bureau a nivel cliente ---
    agg_bureau = {}
    for col in [
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_ANNUITY",
        "CREDIT_DAY_OVERDUE", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE",
    ]:
        if col in bureau.columns:
            agg_bureau[col] = ["mean", "sum", "max"] if col != "DAYS_CREDIT" else ["mean", "min", "max"]

    bureau_num = (
        bureau.groupby("SK_ID_CURR").agg(agg_bureau)
        if agg_bureau else pd.DataFrame(index=bureau["SK_ID_CURR"].unique())
    )
    if len(bureau_num.columns):
        bureau_num.columns = ["BUREAU_" + "_".join(col) for col in bureau_num.columns]

    # Diversidad y conteos (categorical-safe)
    diversity_parts = []
    for cat in ["CREDIT_TYPE", "CREDIT_ACTIVE"]:
        if cat in bureau.columns:
            s = bureau.groupby("SK_ID_CURR")[cat].nunique().rename(f"BUREAU_{cat}_NUNIQUE")
            diversity_parts.append(s)
    counts_total = bureau.groupby("SK_ID_CURR").size().rename("BUREAU_LINES_COUNT")

    bureau_agg = pd.concat(
        [bureau_num, counts_total, *diversity_parts] if diversity_parts else [bureau_num, counts_total],
        axis=1
    )

    # --- Agregaciones de bureau_balance al nivel de SK_ID_BUREAU ---
    bb_agg = {}
    if "MONTHS_BALANCE" in bureau_balance.columns:
        bb_agg["MONTHS_BALANCE"] = ["count", "min", "max", "mean"]

    bb_num = (
        bureau_balance.groupby("SK_ID_BUREAU").agg(bb_agg)
        if bb_agg else pd.DataFrame(index=bureau_balance["SK_ID_BUREAU"].unique())
    )
    if len(bb_num.columns):
        bb_num.columns = ["BB_" + "_".join(col) for col in bb_num.columns]

    if "STATUS" in bureau_balance.columns:
        status_nunique = bureau_balance.groupby("SK_ID_BUREAU")["STATUS"].nunique().rename("BB_STATUS_NUNIQUE")
        bb_num = bb_num.join(status_nunique, how="left")

    # Merge con bureau por SK_ID_BUREAU
    bureau_full = bureau.merge(bb_num, on="SK_ID_BUREAU", how="left")

    # 🚀 Pro-mediar SOLO columnas numéricas
    num_cols = bureau_full.select_dtypes(include="number").columns
    if len(bureau_full):
        bureau_full_agg = bureau_full.groupby("SK_ID_CURR")[num_cols].mean()
        bureau_full_agg.columns = ["BUREAU_BAL_" + c for c in bureau_full_agg.columns]
    else:
        bureau_full_agg = pd.DataFrame(index=bureau["SK_ID_CURR"].unique())

    # Unión final
    final = bureau_agg.join(bureau_full_agg, how="left")
    final = final.sort_index()
    return final
