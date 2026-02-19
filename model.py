import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Step 1: Load dataset
df = pd.read_csv("ipl_data.csv")

# Step 2: Select features (X) and target (y)
X = df[['matches', 'runs', 'strike_rate', 'average', 'wickets', 'economy']]
y = df['performance_rating']

# Step 3: Split dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the trained model so Streamlit can use it
pickle.dump(model, open("ipl_model.pkl", "wb"))

print("Model training completed! File saved as ipl_model.pkl")