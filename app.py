import streamlit as st
import pandas as pd
import pickle
import base64
import os
import requests

# ---------------- BACKGROUND ----------------
def add_bg():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#fbc2eb,#a6c1ee);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOADERS ----------------
def load_model():
    try:
        with open("ipl_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

def load_data():
    try:
        return pd.read_csv("ipl_data.csv")
    except:
        return None

# ---------------- MAIN ----------------
def main():
    st.set_page_config(page_title="IPL Predictor", layout="wide")
    add_bg()

    st.title("🏏 IPL Player Performance Predictor")

    # -------- API BUTTON --------
    if st.button("Load Player Data", key="btn_load"):
        url = "https://api.cricapi.com/v1/players?apikey=7874ce44-b5d6-4220-ae95-4c9bfe42559b&offset=0"
        try:
            r = requests.get(url)
            data = r.json()

            if "data" in data:
                p = data["data"][0]
                st.subheader("Live Player Info")
                st.write("Name:", p.get("name"))
                st.write("Country:", p.get("country"))
                st.write("Role:", p.get("role"))
            else:
                st.warning("No data")

        except:
            st.error("API Error")

    # -------- LOAD FILES --------
    model = load_model()
    df = load_data()

    col1, col2 = st.columns(2)

    # -------- INPUT --------
    with col1:
        matches = st.number_input("Matches", 0.0)
        runs = st.number_input("Runs", 0.0)
        strike_rate = st.number_input("Strike Rate", 0.0)
        average = st.number_input("Average", 0.0)
        wickets = st.number_input("Wickets", 0.0)
        economy = st.number_input("Economy", 0.0)

        # -------- PREDICT --------
        if st.button("Predict", key="btn_predict"):
            if model is None:
                st.error("Model missing")
            else:
                X = pd.DataFrame([[matches, runs, strike_rate, average, wickets, economy]],
                                 columns=['matches','runs','strike_rate','average','wickets','economy'])
                try:
                    pred = model.predict(X)
                    st.success(f"Prediction: {pred[0]:.2f}")
                except Exception as e:
                    st.error(str(e))

        # -------- RESET --------
        if st.button("Reset", key="btn_reset"):
            st.rerun()

    # -------- DATA --------
    with col2:
        if df is not None:
            st.subheader("Dataset")
            st.dataframe(df.head())
        else:
            st.info("CSV file not found")

# -------- RUN --------
if __name__ == "__main__":
    main()
