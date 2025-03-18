import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ Load dataset
file_path = r"C:\Users\valli\Downloads\KILL\KILL\MentalHealthSurvey.csv"  # Update if needed
df = pd.read_csv(file_path)

# ✅ Normalize column names (Lowercase + Trim Spaces)
df.columns = df.columns.str.lower().str.strip()

# ✅ Print column names for debugging
print("Dataset Columns:", df.columns)

# ✅ Define features & target
features = ["age", "cgpa", "average_sleep", "academic_workload", "academic_pressure", "financial_concerns", "social_relationships"]
target = "depression"  # Update if predicting something else

# ✅ Ensure required columns exist
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    raise KeyError(f"Missing columns in dataset: {missing_cols}")

# ✅ Function to convert range-like values into numerical values
def convert_range_to_avg(value):
    if isinstance(value, str):
        value = value.replace("hrs", "").strip()  # Remove text like 'hrs'
        if '-' in value:  # Handle range values (e.g., "4-6")
            nums = list(map(float, value.split('-')))
            return sum(nums) / len(nums)  # Take average
        try:
            return float(value)  # Convert if single number
        except ValueError:
            return np.nan  # If conversion fails, return NaN
    return value

# ✅ Apply conversion to `cgpa` and `average_sleep`
df["cgpa"] = df["cgpa"].apply(convert_range_to_avg)
df["average_sleep"] = df["average_sleep"].apply(convert_range_to_avg)

# ✅ Handle missing values
df = df.dropna()

# ✅ Encode categorical data
if "gender" in df.columns:
    df["gender"] = LabelEncoder().fit_transform(df["gender"])  # Convert Male/Female to 0/1
    features.append("gender")  # Include in features if relevant

# ✅ Extract feature matrix & target
X = df[features]
y = df[target]

# ✅ Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save model & scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model training complete. Model saved as 'model.pkl'.")




