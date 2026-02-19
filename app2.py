# app.py
import streamlit as st
import pandas as pd
import pickle
import base64
import sys, os
import io
import requests

# helper to support running inside PyInstaller exe or normal folder
def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    return os.path.join(base, relative)

# add a colorful gradient background (CSS)
def add_gradient_background():
    css = """
    <style>
    .stApp {
      background: linear-gradient(135deg, #fbc2eb 0%, #a18cd1 50%, #89f7fe 100%);
      background-attachment: fixed;
    }
    /* Make the main container slightly transparent so background shows */
    .css-1d391kg { background-color: rgba(255,255,255,0.85); } /* wrapper */
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# add Kohli overlay image (center, semi-transparent)
def add_kohli_overlay(image_file, width_pct=80, opacity=0.18):
    path = resource_path(image_file)
    try:
        with open(path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            img_html = f"""
            <div style="display:flex;justify-content:center;align-items:center;pointer-events:none;">
              <img src="data:image/jpeg;base64,{b64}" style="width:{width_pct}%;opacity:{opacity};filter: blur(0.4px); border-radius:8px;" />
            </div>
            """
            st.markdown(img_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning("Background image not found or couldn't load: " + str(e))

# load dataset (if exists)
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

# load model
def load_model(path):
    try:
        with open(path,"rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error("Model file not found or couldn't load: " + str(e))
        return None

# main UI
def main():
    st.set_page_config(page_title="IPL Player Performance Predictor", layout="wide")
    add_gradient_background()

    # top title area
    st.title("🏏 IPL Player Performance Predictor App")
    st.write("Choose a player from dataset (if available) OR enter stats manually, then click Predict.")

    if st.button("Load Player Data"):

       API_KEY = "59046beb-2468-437b-9945-a3240e7a4337"

       url = f"https://api.cricketdata.org/v1/players?apikey={API_KEY}&search=Virat Kohli"

       try:
           response = requests.get(url, timeout=10)
           data = response.json()

           if "data" in data and len(data["data"]) > 0:
               player = data["data"][0]

               st.subheader("Live Player Info")
               st.write("Name:", player.get("name"))
               st.write("Country:", player.get("country"))
               st.write("Role:", player.get("role"))

           else:
               st.warning("No data found")

       except requests.exceptions.RequestException:
           st.error("⚠️ API connection failed.")
           
       if __name__== "__main__":
           main()


    # load resources using resource_path
    model = load_model(resource_path("ipl_model.pkl"))
    df = load_dataset(resource_path("ipl_data.csv"))

    # add kohli overlay (below title)
    add_kohli_overlay("kohli.jpg", width_pct=90, opacity=0.18)

    # layout columns
    left, right = st.columns([1.2, 1])

    # Session defaults (so widgets show dataset values when player chosen)
    if 'matches' not in st.session_state:
        st.session_state.matches = 0.0
        st.session_state.runs = 0.0
        st.session_state.strike_rate = 0.0
        st.session_state.average = 0.0
        st.session_state.wickets = 0.0
        st.session_state.economy = 0.0

    with left:
        # Player selector (if dataset available)
        selected_player = None
        if isinstance(df, pd.DataFrame):
            # try to detect a name column
            name_col = None
            for c in df.columns:
                if 'name' in c.lower():
                    name_col = c
                    break
            if name_col:
                names = sorted(df[name_col].astype(str).tolist())
                names.insert(0, "--- Select player from dataset ---")
                selected_player = st.selectbox("Player name (from CSV)", names, index=0)
                if selected_player and selected_player != names[0]:
                    # find the row
                    row = df[df[name_col].astype(str) == str(selected_player)]
                    if not row.empty:
                        row = row.iloc[0]
                        # set session defaults BEFORE widgets are created
                        st.session_state.matches = float(row.get('matches', 0))
                        st.session_state.runs = float(row.get('runs', 0))
                        st.session_state.strike_rate = float(row.get('strike_rate', 0))
                        st.session_state.average = float(row.get('average', 0))
                        st.session_state.wickets = float(row.get('wickets', 0))
                        st.session_state.economy = float(row.get('economy', 0))
                    else:
                        st.info("Selected player not found in CSV row.")
            else:
                st.info("No player name column found in CSV. You can still enter stats manually.")
        else:
            st.info("No dataset CSV loaded. Put 'ipl_data.csv' in the app folder to enable player selector.")

        st.markdown("### Enter stats (or leave from dataset):")
        matches = st.number_input("Matches Played", min_value=0.0, step=1.0, value=float(st.session_state.matches))
        runs = st.number_input("Total Runs", min_value=0.0, step=1.0, value=float(st.session_state.runs))
        strike_rate = st.number_input("Strike Rate", min_value=0.0, step=0.1, value=float(st.session_state.strike_rate))
        average = st.number_input("Batting Average", min_value=0.0, step=0.1, value=float(st.session_state.average))
        wickets = st.number_input("Wickets Taken", min_value=0.0, step=1.0, value=float(st.session_state.wickets))
        economy = st.number_input("Economy Rate", min_value=0.0, step=0.1, value=float(st.session_state.economy))

        # predict button
        if st.button("Predict Performance Rating"):
            # avoid predicting when all zero
            if matches == 0 and runs == 0 and strike_rate == 0 and average == 0 and wickets == 0 and economy == 0:
                st.warning("Please enter player's stats before predicting.")
            else:
                if model is None:
                    st.error("Model not loaded. Cannot predict.")
                else:
                    # prepare features as DataFrame with column names your model expects
                    X = pd.DataFrame([[matches, runs, strike_rate, average, wickets, economy]],
                                     columns=['matches','runs','strike_rate','average','wickets','economy'])
                    try:
                        pred = model.predict(X)
                        st.success(f"Predicted Performance Rating: {pred[0]:.2f}")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

    with right:
        st.markdown("## Player quick info")
        if isinstance(df, pd.DataFrame) and selected_player and selected_player != "--- Select player from dataset ---":
            # display selected player's row
            name_col = [c for c in df.columns if 'name' in c.lower()]
            name_col = name_col[0] if name_col else df.columns[0]
            row = df[df[name_col].astype(str) == str(selected_player)]
            if not row.empty:
                info = row.iloc[0].to_dict()
                st.write("**Name:**", selected_player)
                st.write("**performance_rating:**", int(info.get('performance_rating', 0)))
                st.table(pd.DataFrame([info]))
            else:
                st.info("No row data found for this player.")
        else:
            st.write("Select a player from left (if dataset loaded) or input stats and predict.")

    # show dataset summary at bottom (optional)
    st.markdown("---")
    if isinstance(df, pd.DataFrame):
        st.write("Dataset summary (if loaded)")
        st.dataframe(df.head(10))
    else:
        st.info("No dataset file (ipl_data.csv) present in folder.")

if __name__ == "__main__":

  









