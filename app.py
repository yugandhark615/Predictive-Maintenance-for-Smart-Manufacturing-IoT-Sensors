from flask import Flask, render_template, request
import joblib, numpy as np, pandas as pd
from io import StringIO

app = Flask(__name__)
m = joblib.load("model.pkl")
clf = m["binary_model"]; type_model = m["type_model"]; FEATURES = m["features"]

# Safe operating limits per sensor (used for both the manual-mode risk gauges
# and the rule-based dataset evaluation below).
LIMITS = {
    "Temperature_K":    {"label": "Temperature",      "unit": "K",   "low": 340,  "high": 350},
    "Rotational Speed": {"label": "Rotational Speed",  "unit": "RPM", "low": 1900, "high": 2200},
    "Torque":           {"label": "Torque",            "unit": "Nm",  "low": 55,   "high": 65},
    "Tool Wear":        {"label": "Tool Wear",         "unit": "min", "low": 140,  "high": 180},
}

# Which failure category a breached sensor points to (matches the trained type model's classes).
FAILURE_TYPE_MAP = {
    "Temperature_K": "Heat Dissipation Failure",
    "Rotational Speed": "Power Failure",
    "Torque": "Power Failure",
    "Tool Wear": "Tool Wear Failure",
}

DEFAULTS = {"feature_stats": [], "row_errors": [], "row_errors_total": 0, "total_rows": 0,
            "series": {}, "importance": [], "risks": {}}


def level(v, low, high):
    return "Low" if v <= low else "Medium" if v <= high else "High"


def risks(values_by_feature):
    """values_by_feature: {feature_key: value} -- a single reading (manual mode)
    or the worst-case (max) reading seen across an uploaded dataset."""
    out = {}
    for f, spec in LIMITS.items():
        val = float(values_by_feature[f])
        out[f"{spec['label']} ({spec['unit']})"] = {
            "value": round(val, 1),
            "level": level(val, spec["low"], spec["high"]),
            "pct": round(val / spec["high"] * 100, 1),
        }
    return out


def importance(v):
    x = np.array([[v[f] for f in FEATURES]], float)
    base = clf.predict_proba(x)[0][1]
    ref = {"Temperature_K": 332.4, "Rotational Speed": 1531.5, "Torque": 45, "Tool Wear": 116}
    s = []
    for f in FEATURES:
        xx = x.copy(); xx[0, FEATURES.index(f)] = ref[f]
        s.append((f, abs(base - clf.predict_proba(xx)[0][1])))
    total = sum(x[1] for x in s)
    return sorted([(f, round(x / total * 100, 2)) for f, x in s], key=lambda x: x[1], reverse=True) if total else s


def classify(v):
    """Manual Prediction mode -- unchanged: trained Random Forest model on a single reading."""
    x = np.array([[v[f] for f in FEATURES]], float)
    fail = int(clf.predict(x)[0])
    p = round(float(clf.predict_proba(x)[0][1]) * 100, 2)
    typ = str(type_model.predict(x)[0]) if fail else "No Failure"
    return {
        "result": "Machine Failure Risk Detected" if fail else "Machine Operating Normally",
        "status": "danger" if fail else "safe",
        "probability": p,
        "failure_type": typ,
        "recommendation": f"Predicted failure type: {typ}. Schedule inspection/maintenance." if fail else
                           "No immediate maintenance action is indicated by the model.",
        "risks": risks(v),
        "importance": importance(v),
    }


def clean(df):
    aliases = {"temperature": "Temperature_K", "temperature_k": "Temperature_K", "temp": "Temperature_K",
               "rotational_speed": "Rotational Speed", "rotational speed": "Rotational Speed",
               "rpm": "Rotational Speed", "speed": "Rotational Speed",
               "torque_nm": "Torque", "torque": "Torque",
               "tool_wear": "Tool Wear", "tool wear": "Tool Wear", "wear": "Tool Wear"}
    df = df.rename(columns={c: aliases.get(c.strip().lower(), c) for c in df.columns})
    missing = [f for f in FEATURES if f not in df.columns]
    if missing: raise ValueError("Missing columns: " + ", ".join(missing))
    time = [c for c in df if c.strip().lower() in ["timestamp", "time", "date", "datetime"]]
    if time:
        d = pd.to_datetime(df[time[0]], errors="coerce")
        if d.notna().all(): df = df.assign(_t=d).sort_values("_t").drop(columns="_t")
    df = df[FEATURES].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if len(df) < 1: raise ValueError("No valid numeric sensor rows were found in the uploaded data.")
    return df


def evaluate_dataset(df):
    """Sensor Log Analysis mode. No forecasting: every row of the uploaded dataset is
    checked against the same manual-style safe limits used in Manual Prediction. Any single
    reading that crosses a sensor's limit raises an error against that specific feature, and
    -- because one failing parameter is enough to take the machine down -- the whole batch is
    reported as a failure."""
    total = len(df)
    feature_stats = {}
    worst_case = {}
    any_breach = False

    for f, spec in LIMITS.items():
        vals = df[f].astype(float)
        max_v = float(vals.max()); max_row = int(vals.idxmax()) + 1
        worst_case[f] = max_v
        breach_rows = (df.index[vals > spec["high"]] + 1).tolist()
        feature_stats[f] = {
            "key": f, "label": spec["label"], "unit": spec["unit"],
            "max_value": round(max_v, 2), "max_row": max_row,
            "threshold": spec["high"],
            "level": level(max_v, spec["low"], spec["high"]),
            "breach_count": len(breach_rows),
            "breach_pct": round(len(breach_rows) / total * 100, 1) if total else 0,
            "status": "Exceeded" if breach_rows else "Within Limits",
        }
        if breach_rows: any_breach = True

    row_errors = []
    for i, row in df.iterrows():
        errs = []
        for f, spec in LIMITS.items():
            val = float(row[f])
            if val > spec["high"]:
                errs.append({"feature": spec["label"], "value": round(val, 2),
                             "threshold": spec["high"], "unit": spec["unit"]})
        if errs:
            row_errors.append({"row": i + 1, "errors": errs})

    triggered = [f for f, s in feature_stats.items() if s["breach_count"] > 0]
    types = sorted({FAILURE_TYPE_MAP[f] for f in triggered})
    failure_type = " + ".join(types) if types else "No Failure"
    failure_pct = round(len(row_errors) / total * 100, 1) if total else 0

    if any_breach:
        bad = ", ".join(feature_stats[f]["label"] for f in triggered)
        recommendation = (f"{len(row_errors)} of {total} readings exceeded a safe operating limit. "
                           f"A single out-of-range reading is treated as a fault, so the machine is "
                           f"flagged as failing overall. Inspect: {bad}.")
    else:
        recommendation = f"All {total} readings across every parameter stayed within safe operating limits."

    return {
        "result": "Machine Failure Risk Detected" if any_breach else "Machine Operating Normally",
        "status": "danger" if any_breach else "safe",
        "probability": failure_pct,
        "failure_type": failure_type,
        "recommendation": recommendation,
        "risks": risks(worst_case),
        "feature_stats": list(feature_stats.values()),
        "row_errors": row_errors[:100],
        "row_errors_total": len(row_errors),
        "total_rows": total,
        "series": {f: df[f].round(2).tolist() for f in FEATURES},
    }


@app.route("/")
def home():
    return render_template("index.html", mode="manual", **DEFAULTS)


@app.route("/predict", methods=["POST"])
def predict():
    mode = request.form.get("mode", "manual")
    try:
        if mode == "manual":
            v = {"Temperature_K": float(request.form["temperature"]),
                 "Rotational Speed": float(request.form["speed"]),
                 "Torque": float(request.form["torque"]),
                 "Tool Wear": float(request.form["wear"])}
            return render_template("index.html", mode=mode, **{**DEFAULTS, **classify(v)})

        raw = request.form.get("timeseries_csv", "").strip()
        file = request.files.get("timeseries_file")
        if file and file.filename: raw = file.read().decode("utf-8-sig")
        if not raw: raise ValueError("Please paste CSV data or upload a CSV file.")
        df = clean(pd.read_csv(StringIO(raw)))
        return render_template("index.html", mode=mode, **{**DEFAULTS, **evaluate_dataset(df)})
    except Exception as e:
        return render_template("index.html", mode=mode, result="Input Error", status="danger",
                                probability=0, failure_type="-", recommendation=str(e), **DEFAULTS)


if __name__ == "__main__":
    app.run(debug=True)
