# Network Intrusion Detection System

A multi-class classifier that labels network connections as **normal** or one of four attack categories (**DoS, Probe, R2L, U2R**). It is trained on the NSL-KDD dataset (a cleaned version of KDD Cup 1999), compares XGBoost against Random Forest using per-class recall and macro F1, and is served through a small Flask web app.

![App screenshot](screenshots/app.png)

## Dataset

NSL-KDD, using the official split:

| Split | Records |
|---|---|
| `KDDTrain+.txt` | 125,973 |
| `KDDTest+.txt` | 22,544 |

Each record has 41 features describing a network connection. Raw attack names are grouped into 5 classes:

| Class | Train | Test |
|---|---|---|
| normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,460 |
| Probe | 11,656 | 2,421 |
| R2L | 995 | 2,885 |
| U2R | 52 | 67 |

## Approach

1. Map raw attack names to the 5 classes.
2. Label-encode `protocol_type`, `service` and `flag`; scale features with `StandardScaler`.
3. Handle class imbalance with balanced sample weights (XGBoost) and `class_weight="balanced"` (Random Forest).
4. Train both models on the training set and evaluate on the official test set.
5. Save the best model (by macro F1) with its scaler, encoders, feature list and labels.

## Results (official KDDTest+ set)

| Metric | XGBoost | Random Forest |
|---|---|---|
| Accuracy | 0.7796 | 0.7464 |
| Macro F1 | 0.5980 | 0.5087 |
| Recall: normal | 0.9702 | 0.9716 |
| Recall: DoS | 0.8008 | 0.7721 |
| Recall: Probe | 0.7679 | 0.6646 |
| Recall: R2L | 0.1064 | 0.0059 |
| Recall: U2R | 0.2090 | 0.0896 |

XGBoost was selected. Confusion matrices are in `reports/`.

## Limitations

- **R2L and U2R are detected poorly.** The training set has only 995 R2L and 52 U2R records, and the test set contains attack types that never appear in training. This is a known property of NSL-KDD. In the web app, a `guess_passwd` (R2L) record can be classified as normal with 100% confidence.
- The test set was also used to choose between the two models. A fair next step is a separate validation split from the training data.
- The model works on pre-extracted connection features, not raw packets.

## Project structure

```
app.py                Flask app
src/train.py          trains and compares models, saves artifacts
src/predict.py        loads artifacts and classifies one record
templates/index.html  web page
models/               saved model, scaler, encoders, features, labels
reports/              comparison.csv and confusion matrices
```

## Run it

```bash
pip install -r requirements.txt
# place KDDTrain+.txt and KDDTest+.txt in the project root
python src/train.py      # trains and saves models/ and reports/
python app.py            # open http://127.0.0.1:5000
```

In the app, paste a 41-value record or click **Load random test record**, then **Classify record**.

## Tech

Python, pandas, scikit-learn, XGBoost, Flask.