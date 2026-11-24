import pandas as pd

df = pd.read_csv("expenses.csv")


def predict_rule(row):
    text = str(row["merchant"]).lower()
    if any(k in text for k in ["swiggy", "zomato", "coffee", "groceries"]):
        return "Food"
    elif any(k in text for k in ["uber", "metro", "rapido", "cab"]):
        return "Transport"
    elif any(k in text for k in ["bill", "recharge", "fiber", "netflix"]):
        return "Utilities"
    elif any(k in text for k in ["pharmacy", "cultfit", "meds"]):
        return "Health"
    elif any(k in text for k in ["amazon", "myntra", "flipkart", "zara", "pos"]):
        return "Shopping"
    return "Other"


df["rule_pred"] = df.apply(predict_rule, axis=1)
accuracy = (df["rule_pred"] == df["category"]).mean()

print(f"Rule Baseline Accuracy: {accuracy:.2%}")
print("\nRows where rule missed or fell into Other:")
print(df[df["rule_pred"] != df["category"]][["merchant", "amount", "category", "rule_pred"]])