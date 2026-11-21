import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
plt.rcParams["figure.figsize"] = (10, 6)

from io import StringIO          # FIXED: this import was missing
from pathlib import Path

# KDDTrain+.txt sits in the project root (one level above src/)
DATA_PATH = Path(__file__).resolve().parent.parent / "KDDTrain+.txt"

with open(DATA_PATH, "r") as f:
    lines = f.readlines()

# If the file has an ARFF header, start after "@data"; otherwise read everything
data_start = next(
    (i + 1 for i, line in enumerate(lines) if line.strip().lower() == "@data"),
    0,
)

data_str = "".join(lines[data_start:])
df_0 = pd.read_csv(StringIO(data_str), header=None)

df = df_0.copy()
print(df.head())

columns = ([
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'attack',
])
print("df shape:", df.shape)
print("columns list length:", len(columns))

# NSL-KDD has an extra 'level' column at the end; drop it
if df.shape[1] == len(columns) + 1:
    df = df.iloc[:, :len(columns)]

df.columns = columns

df.info()

print(df.isnull().sum())


def unique_values(df, columns):
    """Prints unique values and their counts for specific columns in the DataFrame."""
    for column_name in columns:
        print(f"Column: {column_name}\n{'-'*30}")
        unique_vals = df[column_name].unique()
        value_counts = df[column_name].value_counts()
        print(f"Unique Values ({len(unique_vals)}): {unique_vals}\n")
        print(f"Value Counts:\n{value_counts}\n{'='*40}\n")


print(df.duplicated().sum())

print(df.shape)

df.select_dtypes(include='number').plot(kind='box', subplots=True, layout=(8, 5), figsize=(20, 40))
plt.show()

print(df.columns.tolist())
attack_n = []
for i in df['attack']:
    if i == 'normal':
        attack_n.append("normal")
    else:
        attack_n.append("attack")
df['attack'] = attack_n
print(df['attack'].value_counts())

print(df['attack'].unique())

df.hist(bins=43, figsize=(20, 30))
plt.show()

plt.figure(figsize=(16, 4))
sns.countplot(x='attack', data=df, hue='protocol_type')
plt.xticks(rotation=45)
plt.title('Attack Counts over Protocol Types', fontdict={'fontsize': 16})
plt.show()

print(df["protocol_type"].value_counts(normalize=True))

plt.figure(figsize=(20, 8))
ax = sns.countplot(x='service', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.xlabel('Service')
plt.ylabel('Count')
plt.title('Count of Services')
plt.grid(True)
plt.show()

plt.figure(figsize=(20, 8))
ax = sns.countplot(x='service', hue='attack', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.xlabel('Service')
plt.ylabel('Count')
plt.title('Distribution of Attacks by Service')
plt.legend(title='Attack Type')
plt.grid(True)
plt.show()

g = sns.displot(
    data=df,
    x="duration",
    hue="flag",
    kind="kde",
    height=6,
    multiple="fill",
    clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)
g.figure.suptitle('Kernel Density Estimate (KDE) Plot of Duration by Flag')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='is_guest_login', hue='attack', data=df, palette='Set2')
plt.xlabel('Is Guest Login')
plt.ylabel('Count')
plt.title('Distribution of Attack Types by Guest Login')
plt.legend(title='Attack Type')
plt.grid(True)
plt.show()

cat_features = df.select_dtypes(include='object').columns
print(cat_features)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

clm = ['protocol_type', 'service', 'flag']
for x in clm:
    df[x] = le.fit_transform(df[x])
df['attack'] = df['attack'].map({'normal': 0, 'attack': 1})
label_names = np.array(['normal', 'attack'])

from sklearn.model_selection import train_test_split

X = df.drop(["attack"], axis=1)
y = df["attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=43, stratify=y
)

train_index = X_train.columns
print(train_index)

from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = train_index
print(mutual_info.sort_values(ascending=False))

mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 5))
plt.tight_layout()
plt.show()

from sklearn.feature_selection import SelectKBest
Select_features = SelectKBest(mutual_info_classif, k=15)
Select_features.fit(X_train, y_train)
selected_columns = list(train_index[Select_features.get_support()])
print(selected_columns)

X_train = X_train[selected_columns]
X_test = X_test[selected_columns]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

XGBoost_model = XGBClassifier(random_state=42)
Logistic_model = LogisticRegression(random_state=42, max_iter=1000)

XGBoost = XGBoost_model.fit(X_train, y_train)

Logistic = Logistic_model.fit(X_train, y_train)

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    recall_score, precision_score, f1_score, roc_auc_score,
)


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))


print("\n===== Logistic Regression =====")
eval_metric(Logistic_model, X_train, y_train, X_test, y_test)

print("\n===== XGBoost (default) =====")
eval_metric(XGBoost_model, X_train, y_train, X_test, y_test)

# NOTE: this grid is 108 combinations x 3 folds and can take several minutes.
# For a quick test, reduce each list to 1-2 values.
param_grid = {
    "n_estimators": [50, 100, 128],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.5, 0.8],
    "colsample_bytree": [0.5, 0.8],
}

from sklearn.model_selection import GridSearchCV

XGB_model = XGBClassifier(random_state=42)

XGB_grid_model = GridSearchCV(
    XGB_model,
    param_grid,
    scoring="f1",
    cv=3,
    n_jobs=1,          # line 276: was -1, which crashes on Windows + Python 3.13
    return_train_score=True,
).fit(X_train, y_train)

print("Best CV F1:", XGB_grid_model.best_score_)
print("Best params:", XGB_grid_model.best_params_)

XGB_model = XGBClassifier(random_state=42, **XGB_grid_model.best_params_)

XGB_model.fit(X_train, y_train)

y_pred = XGB_model.predict(X_test)
y_pred_proba = XGB_model.predict_proba(X_test)

xgb_f1 = f1_score(y_test, y_pred)
xgb_recall = recall_score(y_test, y_pred)
xgb_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print("F1:", xgb_f1)
print("Recall:", xgb_recall)
print("ROC AUC:", xgb_auc)

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(XGB_model, X_test, y_test)
plt.show()

print("\n===== XGBoost (tuned) =====")
eval_metric(XGB_model, X_train, y_train, X_test, y_test)

model = XGB_model

feats = pd.DataFrame(index=selected_columns, data=model.feature_importances_, columns=['XGB_importance'])
ada_imp_feats = feats.sort_values("XGB_importance", ascending=False)
print(ada_imp_feats)

y_pred_string = label_names[y_pred]
print(y_pred_string)

plt.figure(figsize=(10, 6))
sns.countplot(x=y_pred_string, hue=y_pred_string, palette="pastel", legend=False)

plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.title("Distribution of Attack Types")

plt.show()