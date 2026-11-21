"""app.py - Flask app serving the intrusion detection model."""
import random
import os 
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from src.predict import FEATURES, predict, record_from_csv_line
app = Flask(__name__)
TEST_FILE = Path(__file__).resolve().parent / "KDDTest+.txt"


@app.route("/")
def index():
    return render_template("index.html", n_features=len(FEATURES))


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or {}
    try:
        if "record" in data:                      # one raw CSV line
            record = record_from_csv_line(data["record"])
        else:                                     # JSON with feature names
            record = data
        return jsonify(predict(record))
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/sample")
def sample():
    """Return a random record from the test file with its true attack name."""
    with open(TEST_FILE) as f:
        lines = f.readlines()
    line = random.choice(lines).strip()
    parts = line.split(",")
    return jsonify({"record": ",".join(parts[:41]), "true_attack": parts[41]})


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)