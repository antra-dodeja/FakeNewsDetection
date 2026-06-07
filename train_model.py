"""
train_model.py — Model Training Script
========================================
Run this ONCE to train and save the Fake News Detection model.

Usage:
    python train_model.py

Dataset Required:
    Download from Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    Place Fake.csv and True.csv in the same folder as this script.

Output:
    model.pkl        — Trained classifier (Logistic Regression or Naive Bayes)
    vectorizer.pkl   — Fitted TF-IDF vectorizer
    results/         — Evaluation plots (confusion matrix, ROC curve, feature importance)
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from utils import preprocess_text


# ── Helpers ────────────────────────────────────────────────────────────────────
def print_header(text):
    print("\n" + "=" * 58)
    print(f"  {text}")
    print("=" * 58)

def print_step(n, total, text):
    print(f"\n  [{n}/{total}] {text}")

def remove_location_tag(text):
    """Remove 'CITY (Source) -' patterns that may bias the model."""
    return re.sub(r'^[A-Z\s,]+\(.*?\)\s*-\s*', '', str(text))


# ── Main ───────────────────────────────────────────────────────────────────────
print_header("FAKE NEWS DETECTION — MODEL TRAINING")

# ── Step 1: Load Dataset ───────────────────────────────────────────────────────
print_step(1, 6, "Loading dataset...")

for fname in ["Fake.csv", "True.csv"]:
    if not os.path.exists(fname):
        print(f"\n  ❌ File not found: {fname}")
        print("  Please download from:")
        print("  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("  and place both Fake.csv and True.csv here.\n")
        exit(1)

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0   # 0 = Fake
true_df["label"] = 1   # 1 = Real

print(f"     Fake articles loaded : {len(fake_df):,}")
print(f"     Real articles loaded : {len(true_df):,}")


# ── Step 2: Prepare Data ───────────────────────────────────────────────────────
print_step(2, 6, "Preparing and combining data...")

# Combine title + body (remove location tags from body)
fake_df["content"] = (
    fake_df["title"].fillna("") + " " +
    fake_df["text"].fillna("").apply(remove_location_tag)
)
true_df["content"] = (
    true_df["title"].fillna("") + " " +
    true_df["text"].fillna("").apply(remove_location_tag)
)

df = pd.concat([fake_df[["content", "label"]], true_df[["content", "label"]]])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"     Total samples        : {len(df):,}")
print(f"     Fake (label=0)       : {(df['label']==0).sum():,}")
print(f"     Real (label=1)       : {(df['label']==1).sum():,}")


# ── Step 3: Preprocess Text ────────────────────────────────────────────────────
print_step(3, 6, "Preprocessing text — this may take 1–2 minutes...")
df["content"] = df["content"].apply(preprocess_text)
print("     ✓ Done.")


# ── Step 4: TF-IDF Vectorization ──────────────────────────────────────────────
print_step(4, 6, "Extracting TF-IDF features...")

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000,    # Top 10,000 words
    ngram_range=(1, 2),    # Unigrams + bigrams
    sublinear_tf=True,     # Apply log normalization
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"     Training samples : {X_train_tfidf.shape[0]:,}")
print(f"     Testing  samples : {X_test_tfidf.shape[0]:,}")
print(f"     Feature count    : {X_train_tfidf.shape[1]:,}")


# ── Step 5: Train & Compare Models ────────────────────────────────────────────
print_step(5, 6, "Training Logistic Regression and Naive Bayes...")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
lr_acc   = accuracy_score(y_test, lr_preds)

# Naive Bayes
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)
nb_acc   = accuracy_score(y_test, nb_preds)

print(f"\n     Logistic Regression accuracy : {lr_acc * 100:.2f}%")
print(f"     Naive Bayes accuracy         : {nb_acc * 100:.2f}%")

# Select best
if lr_acc >= nb_acc:
    best_model = lr_model
    best_preds = lr_preds
    best_name  = "Logistic Regression"
else:
    best_model = nb_model
    best_preds = nb_preds
    best_name  = "Naive Bayes"

print(f"\n     ✅ Best model selected: {best_name} ({max(lr_acc, nb_acc)*100:.2f}%)")

# Detailed classification report
print("\n" + "-" * 45)
print("  Classification Report:")
print("-" * 45)
print(classification_report(y_test, best_preds, target_names=["Fake", "Real"]))


# ── Step 6: Save Evaluation Plots ─────────────────────────────────────────────
print_step(6, 6, "Generating evaluation plots...")

os.makedirs("results", exist_ok=True)

DARK_BG    = "#0a0a0f"
CARD_BG    = "#12121a"
BORDER     = "#1e1e2e"
ACCENT     = "#818cf8"
REAL_COL   = "#10b981"
FAKE_COL   = "#ef4444"
TEXT_COL   = "#e2e8f0"
MUTED      = "#475569"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "text.color":       TEXT_COL,
    "axes.labelcolor":  TEXT_COL,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "axes.edgecolor":   BORDER,
    "grid.color":       BORDER,
    "font.family":      "monospace",
})

# --- 1. Confusion Matrix ---
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues", alpha=0.8)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Fake", "Real"], fontsize=12)
ax.set_yticklabels(["Fake", "Real"], fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, pad=14, color=ACCENT)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if cm[i, j] > cm.max() * 0.5 else MUTED)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved: results/confusion_matrix.png")

# --- 2. ROC Curve ---
proba_test = best_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"ROC AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, linestyle="--", label="Random (AUC = 0.50)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(f"ROC Curve — {best_name}", fontsize=12, pad=14, color=ACCENT)
ax.legend(fontsize=10, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"     Saved: results/roc_curve.png  (AUC = {roc_auc:.4f})")

# --- 3. Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(5.5, 4))
models = ["Logistic\nRegression", "Naive\nBayes"]
accs   = [lr_acc * 100, nb_acc * 100]
colors = [ACCENT, REAL_COL]
bars   = ax.bar(models, accs, color=colors, width=0.45, alpha=0.85, edgecolor=BORDER)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{acc:.2f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT_COL)
ax.set_ylim(85, 100)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Accuracy Comparison", fontsize=12, pad=14, color=ACCENT)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved: results/model_comparison.png")

# --- 4. Top Features ---
if hasattr(best_model, "coef_"):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = best_model.coef_[0]
    top_n = 15
    top_real_idx = np.argsort(coefs)[-top_n:]
    top_fake_idx = np.argsort(coefs)[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    for ax, idx, color, title in [
        (ax1, top_fake_idx, FAKE_COL, "Top Features → FAKE"),
        (ax2, top_real_idx[::-1], REAL_COL, "Top Features → REAL"),
    ]:
        ax.set_facecolor(CARD_BG)
        ax.barh(feature_names[idx], np.abs(coefs[idx]), color=color, alpha=0.8, edgecolor=BORDER)
        ax.set_title(title, color=ACCENT, fontsize=11, pad=10)
        ax.set_xlabel("Coefficient Magnitude", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Most Influential Features (TF-IDF Coefficients)", fontsize=12, color=TEXT_COL, y=1.01)
    plt.tight_layout()
    plt.savefig("results/top_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: results/top_features.png")


# ── Save Model Files ───────────────────────────────────────────────────────────
print("\n  Saving model files...")
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("  ✅ Saved: model.pkl")
print("  ✅ Saved: vectorizer.pkl")

print_header("TRAINING COMPLETE")
print(f"  Best model   : {best_name}")
print(f"  Accuracy     : {max(lr_acc, nb_acc)*100:.2f}%")
print(f"  ROC AUC      : {roc_auc:.4f}")
print(f"  Plots saved  : results/")
print("\n  To launch the app, run:")
print("  streamlit run app.py\n")
