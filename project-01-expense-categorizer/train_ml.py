import random
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# 1. Generate a 300-row realistic dataset and save to expenses.csv
random.seed(42)
np.random.seed(42)

templates = {
    "Food": [
        ("swiggy bangalore", 220, 650),
        ("zomato order", 250, 750),
        ("third wave coffee", 180, 420),
        ("swiggy instamart groceries", 450, 1400),
        ("amazon fresh groceries", 600, 1800),
        ("dmart supermarket groceries", 900, 2800),
        ("blinkit daily essentials", 250, 900),
        ("starbucks coffee latte", 280, 520),
        ("blue tokai cold brew", 200, 450),
        ("haldirams snacks order", 180, 600),
    ],
    "Transport": [
        ("uber auto koramangala", 90, 260),
        ("metro smart card recharge", 200, 800),
        ("rapido bike ride", 60, 180),
        ("rapido auto ride", 95, 250),
        ("uber cab airport", 850, 1750),
        ("irctc train ticket booking", 350, 1800),
        ("redbus intercity ticket", 500, 1600),
        ("ola cab city ride", 140, 450),
        ("namma yatri auto ride", 85, 240),
    ],
    "Utilities": [
        ("airtel fiber bill", 899, 1399),
        ("bescom electricity bill", 900, 2200),
        ("jio prepaid recharge", 239, 799),
        ("netflix subscription", 199, 649),
        ("airtel postpaid bill", 499, 999),
        ("bengaluru water board bill", 250, 700),
        ("tata play dth recharge", 250, 550),
        ("act fibernet broadband bill", 850, 1350),
        ("spotify duo plan", 149, 199),
    ],
    "Shopping": [
        ("amazon prime shopping", 799, 3500),
        ("myntra fashion haul", 900, 3200),
        ("flipkart electronics cable", 350, 1500),
        ("zara store purchase", 1990, 5500),
        ("uniqlo tshirts shopping", 1290, 3490),
        ("ikea home organizer", 800, 4200),
        ("nykaa skincare order", 600, 2200),
        ("croma usb hub adapter", 500, 1800),
        ("decathlon running shoes", 1800, 4800),
    ],
    "Health": [
        ("apollo pharmacy meds", 200, 950),
        ("cultfit monthly pass", 1500, 2600),
        ("apollo diagnostics blood test", 700, 1800),
        ("1mg pharmacy refill", 300, 1100),
        ("practo doctor consultation", 500, 1000),
        ("pharmeasy lab checkup", 800, 2200),
        ("anytime fitness membership", 1800, 2800),
        ("max hospital dental visit", 600, 1600),
    ],
}

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
rows = []

for _ in range(295):
    cat = random.choice(list(templates.keys()))
    merchant, low, high = random.choice(templates[cat])
    amount = int(np.random.uniform(low, high))
    day = random.choice(days)
    rows.append((merchant, amount, day, cat))

# Add 5 obvious high-amount anomalies
anomalies_seed = [
    ("unknown merchant pos", 48500, "Wednesday", "Shopping"),
    ("swiggy order", 12500, "Tuesday", "Food"),
    ("luxury watch boutique", 64000, "Sunday", "Shopping"),
    ("emergency hospital admission", 38000, "Thursday", "Health"),
    ("uber cab airport", 15000, "Friday", "Transport"),
]
rows.extend(anomalies_seed)

df = pd.DataFrame(rows, columns=["merchant", "amount", "day_of_week", "category"])
df.to_csv("expenses.csv", index=False)

# 2. Split into 75% train and 25% test
X = df[["merchant", "amount", "day_of_week"]]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# 3. Turn text, amount, and day into numbers
log_amount_pipe = Pipeline(
    steps=[
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scale", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "merchant_tfidf",
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True),
            "merchant",
        ),
        ("amount_log_scaled", log_amount_pipe, ["amount"]),
        ("day_ohe", OneHotEncoder(handle_unknown="ignore"), ["day_of_week"]),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# 4. Train and test
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"Total Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Test Accuracy: {(preds == y_test).mean():.2%}\n")
print("Classification Report:")
print(classification_report(y_test, preds, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

# 5. Flag big spend spikes per category
df["cat_mean"] = df.groupby("category")["amount"].transform("mean")
df["cat_std"] = df.groupby("category")["amount"].transform("std").replace(0, 1)
df["z_score"] = (df["amount"] - df["cat_mean"]) / df["cat_std"]

big_spikes = df[df["z_score"].abs() > 2.0]
print("\nBig Spend Anomalies (|z_score| > 2.0):")
print(big_spikes[["merchant", "amount", "category", "z_score"]].round(2))