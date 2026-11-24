# Project 01: Expense Categorizer & Anomaly Check

- **50 rows:** RandomForestClassifier, LogisticRegression, and LinearSVC each achieved 46.15% test accuracy (37 train / 13 test).
- **300 rows:** Expanding to 300 rows (225 train / 75 test) raised LogisticRegression test accuracy to 100.00% on the synthetic test set.
- **Anomaly check:** Z-scores flagged large spend spikes (such as 64000, 48500, 38000, 15000, and 12500).
