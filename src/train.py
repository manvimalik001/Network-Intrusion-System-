"""
train.py - Multi-class Network Intrusion Detection (NSL-KDD)

Classes: normal, DoS, Probe, R2L, U2R
Trains on KDDTrain+.txt, tests on KDDTest+.txt.
Compares XGBoost vs Random Forest on per-class recall and macro F1,
then saves the best model + everything the Flask app needs.

Expected layout (project root = one level above src/):
    KDDTrain+.txt
    KDDTest+.txt
    src/train.py
Outputs:
    models/  -> model.pkl, scaler.pkl, encoders.pkl, features.pkl, labels.pkl
    reports/ -> comparison.csv, confusion_<model>.png
"""
from io import StringIO
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                     # save plots to files; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "KDDTrain+.txt"
TEST_PATH = ROOT / "KDDTest+.txt"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Columns and label mapping
# ----------------------------------------------------------------------------
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack',
]
CAT_COLS = ['protocol_type', 'service', 'flag']

CLASSES = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']   # index = class id

ATTACK_MAP = {
    'normal': 'normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'worm': 'DoS', 'mailbomb': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe',
    'portsweep': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L',
    'named': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_nsl_kdd(path: Path) -> pd.DataFrame:
    """Read an NSL-KDD file (plain CSV or ARFF) into a labelled DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        lines = f.readlines()

    # ARFF files have an "@data" line; plain text files do not
    start = next(
        (i + 1 for i, line in enumerate(lines) if line.strip().lower() == "@data"),
        0,
    )
    df = pd.read_csv(StringIO("".join(lines[start:])), header=None)

    # NSL-KDD has an extra 'level' column at the end; drop it
    df = df.iloc[:, :len(COLUMNS)]
    df.columns = COLUMNS

    # Map raw attack names to the 5 classes
    df['label'] = df['attack'].str.strip().str.lower().map(ATTACK_MAP)
    unmapped = df.loc[df['label'].isnull(), 'attack'].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Attack names missing from ATTACK_MAP: {list(unmapped)}")

    return df.drop(columns=['attack'])


print("Loading data...")
train_df = load_nsl_kdd(TRAIN_PATH)
test_df = load_nsl_kdd(TEST_PATH)
print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
print("\nTrain class counts:\n", train_df['label'].value_counts())
print("\nTest class counts:\n", test_df['label'].value_counts())

# ----------------------------------------------------------------------------
# Encode categorical features (one encoder per column, saved for Flask)
# ----------------------------------------------------------------------------
encoders = {}
for col in CAT_COLS:
    enc = LabelEncoder()
    # Fit on train + test so values that only appear in the test set don't crash
    enc.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
    encoders[col] = enc
    train_df[col] = enc.transform(train_df[col].astype(str))
    test_df[col] = enc.transform(test_df[col].astype(str))

# Encode the target as 0..4 using the fixed CLASSES order
class_to_id = {name: i for i, name in enumerate(CLASSES)}
y_train = train_df['label'].map(class_to_id).values
y_test = test_df['label'].map(class_to_id).values

FEATURES = [c for c in COLUMNS if c != 'attack']
X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

# Scale (not needed by tree models, but keeps the Flask pipeline simple/consistent)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ----------------------------------------------------------------------------
# Class imbalance: U2R and R2L are rare, so weight samples by class
# ----------------------------------------------------------------------------
sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
models = {
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
}

results = []
fitted = {}

for name, model in models.items():
    print(f"\n===== Training {name} =====")
    if name == "XGBoost":
        model.fit(X_train_s, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train_s, y_train)       # RF uses class_weight="balanced"
    fitted[name] = model

    y_pred = model.predict(X_test_s)

    print(classification_report(
        y_test, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    ))

    # Per-class recall = share of each attack type that was actually caught
    recalls = recall_score(
        y_test, y_pred,
        labels=list(range(len(CLASSES))),
        average=None,
        zero_division=0,
    )
    row = {"model": name, "accuracy": float((y_pred == y_test).mean())}
    for cls, r in zip(CLASSES, recalls):
        row[f"recall_{cls}"] = round(float(r), 4)
    row["macro_f1"] = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    row["accuracy"] = round(row["accuracy"], 4)
    results.append(row)

    # Confusion matrix normalised by row (each cell = recall for that true class)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix (counts; colour = row-normalised)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_{name}.png", dpi=150)
    plt.close()

# ----------------------------------------------------------------------------
# Comparison table
# ----------------------------------------------------------------------------
comparison = pd.DataFrame(results).set_index("model")
comparison.to_csv(REPORTS_DIR / "comparison.csv")
print("\n===== Model comparison (test set) =====")
print(comparison.T)

# Pick the best model by macro F1 (treats all 5 classes equally)
best_name = comparison["macro_f1"].idxmax()
best_model = fitted[best_name]
print(f"\nBest model by macro F1: {best_name}")

# ----------------------------------------------------------------------------
# Save everything the Flask app needs
# ----------------------------------------------------------------------------
joblib.dump(best_model, MODELS_DIR / "model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
joblib.dump(FEATURES, MODELS_DIR / "features.pkl")
joblib.dump(np.array(CLASSES), MODELS_DIR / "labels.pkl")
print(f"Saved model files to: {MODELS_DIR}")
print(f"Saved reports to: {REPORTS_DIR}")