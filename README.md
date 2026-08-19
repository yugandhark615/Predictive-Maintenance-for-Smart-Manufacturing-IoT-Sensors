# Predictive Maintenance — IoT Sensor Fault Monitor

Flask project with two modes:

1. **Manual Prediction** — enter a single Temperature / RPM / Torque / Tool Wear reading and
   run it through the trained Random Forest failure-classification model (`model.pkl`).
2. **Sensor Log Analysis** — paste or upload a CSV log. Every row is checked against the same
   safe operating limits used in Manual Prediction (no forecasting, no ML model): any single
   reading that crosses its limit raises an error against that specific sensor, and — because
   one failing parameter is enough to take the machine down — the whole log is reported as a
   failure.

Safe limits used for the rule-based check:

| Sensor | Limit |
|---|---|
| Temperature | 350 K |
| Rotational Speed | 2200 RPM |
| Torque | 65 Nm |
| Tool Wear | 180 min |

The result page shows:
- Overall Failure / Normal verdict and failure type
- Sensor status strip (Low / Medium / High per sensor)
- Sensor reading profile chart
- Manual mode: ML feature-importance chart
- Sensor Log Analysis mode: per-sensor threshold breach rate, per-sensor trend chart with the
  limit line marked, and a table of every flagged row with the specific sensor(s) and values
  that breached their limit

## Run

Windows: double-click `run_project.bat`

Or:
```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

The existing `model.pkl` is used for Manual Prediction, so no retraining is needed.
