# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import
 (   classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("risk_factors_cervical_cancer.csv")
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)
target_cols = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
feature_cols = [c for c in df.columns if c not in target_cols]
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Naive Bayes": GaussianNB()
}

def evaluate_models(X, y, smote=False):
    results = []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42, stratify=y
    )

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append([name, acc, prec, rec, f1])
        print(f"\n{name} Confusion Matrix ({'After' if smote else 'Before'} SMOTE):")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models["Random Forest"]),
            ('svm', models["SVM"]),
            ('nb', models["Naive Bayes"])
        ],
        voting='hard'
    )
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)

    acc = accuracy_score(y_test, y_pred_ens)
    prec = precision_score(y_test, y_pred_ens, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred_ens, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_ens, average='macro', zero_division=0)

    results.append(["Voting Ensemble", acc, prec, rec, f1])
    print(f"\nVoting Ensemble Confusion Matrix ({'After' if smote else 'Before'} SMOTE):")
    print(confusion_matrix(y_test, y_pred_ens))
    print(classification_report(y_test, y_pred_ens))

    return pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
before_results, after_results = [], []
before_counts, after_counts = {}, {}

for target in target_cols:
    print(f"\n=== {target} =")
    X = df[feature_cols]
    y = df[target]
    before_counts[target] = y.value_counts()
    before_df = evaluate_models(X, y, smote=False)
    before_df["Target"] = target
    before_results.append(before_df)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    after_counts[target] = y_res.value_counts()
    after_df = evaluate_models(X_res, y_res, smote=True)
    after_df["Target"] = target
    after_results.append(after_df)

before_final = pd.concat(before_results, ignore_index=True)
after_final = pd.concat(after_results, ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(
    x=list(before_counts.keys()),
    y=[v[1] / v.sum() if 1 in v.index else 0 for v in before_counts.values()],
    ax=axes[0], palette="Blues_r"
)
axes[0].set_title("Minority Class Ratio (Before SMOTE)")
axes[0].set_ylabel("Proportion of Positive Cases")

sns.barplot(
    x=list(after_counts.keys()),
    y=[v[1] / v.sum() if 1 in v.index else 0 for v in after_counts.values()],
    ax=axes[1], palette="Greens"
)
axes[1].set_title("Minority Class Ratio (After SMOTE)")
axes[1].set_ylabel("Proportion of Positive Cases")

plt.tight_layout()
plt.show()

print("\n---  Performance Before SMOTE ---")
print(before_final)
print("\n--- Performance After SMOTE ---")
print(after_final)

best_before = before_final.loc[before_final.groupby("Target")["Accuracy"].idxmax()]
best_after = after_final.loc[after_final.groupby("Target")["Accuracy"].idxmax()]

plt.figure(figsize=(10,6))
plt.plot(best_before["Target"], best_before["Accuracy"], marker='o', label='Before SMOTE')
plt.plot(best_after["Target"], best_after["Accuracy"], marker='s', label='After SMOTE')
plt.title("Highest Model Accuracy per Target (Before vs After SMOTE)")
plt.xlabel("Target")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
