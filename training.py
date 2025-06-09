import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
import joblib

# Load data
df = pd.read_csv("data.csv")

# Features and target
X = df.drop("Laptop_Model", axis=1)
y = df["Laptop_Model"]

# Encode target variable (Laptop_Model)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Preprocessing pipeline
numeric_features = ["RAM", "Storage"]
categorical_features = ["Brand", "Processor"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Hyperparameter grid
param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [5, 10, None]
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Cross-validation strategy
cv_split = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

# GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=cv_split, n_jobs=-1)
grid.fit(X_train, y_train)

# Save the model
joblib.dump(grid.best_estimator_, "laptop_model.pkl")

print("Model trained and saved successfully.")
