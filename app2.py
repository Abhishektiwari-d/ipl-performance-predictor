# app.py
import streamlit as st
import pandas as pd
import pickle
import base64
import sys, os
import requests

# ----------------- helpers -----------------

def resource_path(relative):
    return os.path.join(os.path.abspath("."), relative)

def add_gradient_background():
    st.markdown("""
    <style>
    .stApp {
      background: linear-gradient(135deg, #fbc2eb 0%, #a18cd1 50%, #89f7fe 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def add_kohli_overlay(image_file):
    try:
        with open(image_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="display:flex;justify-content:center;">
              <img src="data:image/jpeg;base64,{b64}" width="60%" style="opacity:0.2;">
            </div>
            """, unsafe_allow_html=True)
    except:
        pass

def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None

def load_dataset(path):
    try:
        return pd.read_csv(path)
    except:
        return None

# ----------------- MAIN -----------------

def main():
    st.set_page_config(page_title="IPL Predictor", layout="wide")
    add_gradient_background()

    st.title("🏏 IPL Player Performance Predictor")

    # -------- API BUTTON --------
    if st.button("Load Player Data", key="btn_load"):
        try:
            url = "https://api.cricapi.com/v1/players?apikey=7874ce44-b5d6-4220-ae95-4c9bfe42559b&offset=0"
            res = requests.get(url, timeout=10)
            data = res.json()

            if "data" in data:
                player = data["data"][0]
                st.success("Live Data Loaded ✅")
                st.write("Name:", player.get("name"))
                st.write("Country:", player.get("country"))
                st.write("Role:", player.get("role"))
            else:
                st.warning("No data found")

        except:
            st.error("API Failed ❌")

    # -------- LOAD FILES --------
    model = load_model("ipl_model.pkl")
    df = load_dataset("ipl_data.csv")

    add_kohli_overlay("kohli.jpg")

    # -------- INPUT --------
    st.subheader("Enter Player Stats")

    matches = st.number_input("Matches", 0.0)
    runs = st.number_input("Runs", 0.0)
    strike_rate = st.number_input("Strike Rate", 0.0)
    average = st.number_input("Average", 0.0)
    wickets = st.number_input("Wickets", 0.0)
    economy = st.number_input("Economy", 0.0)

    # -------- PREDICT --------
    if st.button("Predict", key="btn_predict"):
        if model is None:
            st.error("Model not loaded ❌")
        else:
            X = pd.DataFrame([[matches, runs, strike_rate, average, wickets, economy]],
                             columns=['matches','runs','strike_rate','average','wickets','economy'])
            try:
                pred = model.predict(X)
                st.success(f"Prediction: {pred[0]:.2f}")
            except:
                st.error("Prediction Error ❌")

    # -------- RESET --------
    if st.button("Clear / Reset", key="btn_reset"):
        st.rerun()

    # -------- DATA --------
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

# RUN APP
main()
