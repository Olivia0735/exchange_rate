import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_pickle("SAR_USD_clean.pkl")

    # -------------------------------
    # 1. Fix Date column if missing
    # -------------------------------
    if "Date" not in df.columns:
        df = df.reset_index()

    # After reset_index, find date column
    for col in df.columns:
        if "date" in col.lower():
            df.rename(columns={col: "Date"}, inplace=True)
            break

    # -------------------------------
    # 2. Fix SAR column name
    # -------------------------------
    target_col = None
    for col in df.columns:
        if "sar" in col.lower():
            target_col = col
            break

    if target_col is None:
        st.error("❌ SAR column not found")
        st.stop()

    # -------------------------------
    # 3. Final cleaning
    # -------------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = df.rename(columns={target_col: "SAR"})

    return df
