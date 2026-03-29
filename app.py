import streamlit as st
import pandas as pd
import pickle
import base64
import sys, os
import requests

# ------------------ helpers ------------------

def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    return os.path.join(base, relative)

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
        with open(resource_path(image_file), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="text-align:center;">
            <img src="data:image/jpeg;base64,{b64}" width="80%" style="opacity:0.2"/>
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

# ------------------ main ------------------

def main():
    st.set_page_config(page_title="IPL Predictor", layout="wide")
    add_gradient_background()

    st.title("🏏 IPL Player Performance Predictor")

    # ---------- API ----------
    if st.button("Load Player Data", key="btn_load"):
        API_KEY = "7874ce44-b5d6-4220-ae95-4c9bfe42559b"
        url = f"https://api.cricapi.com/v1/players?apikey={API_KEY}&offset=0"

        try:
            res = requests.get(url, timeout=10)
            data = res.json()

            if "data" in data and len(data["data"]) > 0:
                p = data["data"][0]
                st.subheader("Live Player Info")
                st.write("Name:", p.get("name"))
                st.write("Country:", p.get("country"))
                st.write("Role:", p.get("role"))
            else:
                st.warning("No data found")

        except:
            st.error("API Error")

    # ---------- load ----------
    model = load_model("ipl_model.pkl")
    df = load_dataset("ipl_data.csv")

    add_kohli_overlay("kohli.jpg")

    col1, col2 = st.columns(2)

    # ---------- INPUT ----------
    with col1:
        matches = st.number_input("Matches", 0.0)
        runs = st.number_input("Runs", 0.0)
        strike_rate = st.number_input("Strike Rate", 0.0)
        average = st.number_input("Average", 0.0)
        wickets = st.number_input("Wickets", 0.0)
        economy = st.number_input("Economy", 0.0)

        # ✅ PREDICT BUTTON
        if st.button("Predict", key="btn_predict"):
            if model is None:
                st.error("Model not found")
            else:
                X = pd.DataFrame([[matches, runs, strike_rate, average, wickets, economy]],
                                 columns=['matches','runs','strike_rate','average','wickets','economy'])
                try:
                    pred = model.predict(X)
                    st.success(f"Prediction: {pred[0]:.2f}")
                except Exception as e:
                    st.error(str(e))

        # ✅ RESET BUTTON
        if st.button("Reset", key="btn_reset"):
            st.rerun()

    # ---------- DATA ----------
    with col2:
        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
        else:
            st.info("No dataset found")

# ✅ ONLY HERE CALL MAIN
if __name__ == "__main__":
    main()
