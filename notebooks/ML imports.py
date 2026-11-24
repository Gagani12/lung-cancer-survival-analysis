
# 1. ML IMPORTS & DATA PREPARATION

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Regression Models (for Survival_Months prediction)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)


# Feature Engineering: Pack-Years (standard clinical metric for smoking exposure)
df["Pack_Years"] = (df["Cigarettes_Per_Day"] / 20.0) * df["Years_Smoking"]

# Feature Engineering: Total Symptom Count
symptom_cols = [
    "Coughing", "Shortness_of_Breath", "Chest_Pain", "Coughing_Blood",
    "Fatigue", "Weight_Loss", "Wheezing", "Recurrent_Infections",
    "Swallowing_Difficulty", "Finger_Clubbing"
]
df["Total_Symptoms"] = (df[symptom_cols] == "Yes").sum(axis=1)

# Define Target (Classification: Survived) and drop leakage/ID columns
drop_cols = ["Patient_ID", "Diagnosis_Date", "Survived", "Survival_Months", "Cluster", "Diagnosis_Year"]
X = df.drop(columns=drop_cols)
y = df["Survived"].map({"Yes": 1, "No": 0})

# Identify numerical and categorical columns dynamically
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Train-Test Split (Stratified to preserve the 62.2% / 37.8% class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing Pipeline (prevents data leakage)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

print(f"Training Set Shape: {X_train.shape} | Testing Set Shape: {X_test.shape}")
print(f"Numerical Features ({len(num_cols)}): {num_cols}")
print(f"Categorical Features ({len(cat_cols)}): {len(cat_cols)} columns")