"""Quick check: does a generated Dataset 2 contain a usable outage warning signal?

Run it on the QUICK_TEST output (500 sites, 6 months) before generating the full data.
It is a small, fast version of the main notebook, not a replacement for it.

Usage:
    python check_signal.py <dataset2_folder> [<outage_causes.csv>]

Example:
    python check_signal.py "D:\\Dataset\\Uganda_Cell_Telecom_Dataset\\Dataset_2_Cell_KPI_Outage_v2_s50_quick" ^
                           "D:\\Dataset\\Uganda_Cell_Telecom_Dataset\\Dataset_2_Cell_KPI_Outage_v2_s50_quick_outage_causes.csv"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = pd.Timedelta(hours=12)
KPIS = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Load", "Alarm_Count"]
ALERT_SHARE = 0.01  # recall when the top 1% riskiest rows raise an alert

folder = Path(sys.argv[1])
causes_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

# ------------------------------------------------------------------ load
files = sorted(folder.glob("*.csv"))
if not files:
    sys.exit(f"No CSV files in {folder}")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="ISO8601")
print(f"Rows: {len(df):,} from {len(files)} files | cells: {df.Cell_ID.nunique():,}")

kpi = df[df.Record_Type == "KPI"].sort_values(["Cell_ID", "Timestamp"]).reset_index(drop=True)
out = df[df.Record_Type == "OUTAGE"][["Cell_ID", "Timestamp"]].rename(columns={"Timestamp": "Outage_Start"})
print(f"KPI rows: {len(kpi):,} | outages: {len(out):,}")

causes = None
if causes_path is not None and causes_path.exists():
    causes = pd.read_csv(causes_path, parse_dates=["Cell_Down_Time"])
    out = out.merge(causes[["Cell_ID", "Cell_Down_Time", "Cause"]],
                    left_on=["Cell_ID", "Outage_Start"], right_on=["Cell_ID", "Cell_Down_Time"], how="left").drop(columns="Cell_Down_Time")
    print("Outages by cause:", out.Cause.value_counts().to_dict())

# ------------------------------------------------------------------ label: new outage start in (t, t + 12 h]
kpi = kpi.sort_values("Timestamp")
nxt = pd.merge_asof(kpi[["Timestamp", "Cell_ID"]].reset_index(),
                    out.sort_values("Outage_Start").rename(columns={"Outage_Start": "Next_Outage"}).assign(Timestamp=lambda x: x.Next_Outage),
                    on="Timestamp", by="Cell_ID", direction="forward", allow_exact_matches=False)
nxt = nxt.set_index("index").sort_index()
kpi = kpi.sort_index()
kpi["Next_Outage"] = nxt["Next_Outage"]
kpi["Next_Cause"] = nxt["Cause"] if "Cause" in nxt else np.nan
kpi["y"] = ((kpi.Next_Outage - kpi.Timestamp) <= HORIZON).astype(int)
kpi = kpi[kpi.Timestamp <= kpi.Timestamp.max() - HORIZON]

# ------------------------------------------------------------------ past-only features
kpi = kpi.sort_values(["Cell_ID", "Timestamp"]).reset_index(drop=True)
kpi["Utilization"] = kpi.Load / kpi.Capacity * 100
feats = []
g = kpi.groupby("Cell_ID", sort=False)
for c in KPIS + ["Utilization"]:
    kpi[f"{c}_lag1"] = g[c].shift(1)
    kpi[f"{c}_lag2"] = g[c].shift(2)
    kpi[f"{c}_diff1"] = kpi[c] - kpi[f"{c}_lag1"]
    kpi[f"{c}_mean3"] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    feats += [c, f"{c}_lag1", f"{c}_lag2", f"{c}_diff1", f"{c}_mean3"]
kpi = kpi.dropna(subset=feats)

# ------------------------------------------------------------------ chronological split: last third = test
t0, t1 = kpi.Timestamp.min(), kpi.Timestamp.max()
cut = t0 + (t1 - t0) * 2 / 3
train, test = kpi[kpi.Timestamp < cut], kpi[kpi.Timestamp >= cut]
print(f"\nTrain {len(train):,} rows ({train.y.mean():.3%} positive) | test {len(test):,} rows ({test.y.mean():.3%} positive), from {cut:%Y-%m-%d}")

# training: all positives + 5% of negatives (as in the notebook)
tr = pd.concat([train[train.y == 1], train[train.y == 0].sample(frac=0.05, random_state=42)])

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=42),
}
prev = test.y.mean()
rows = []
for name, m in models.items():
    m.fit(tr[feats], tr.y)
    p = m.predict_proba(test[feats])[:, 1]
    ap = average_precision_score(test.y, p)
    k = max(1, int(len(test) * ALERT_SHARE))
    alert = np.zeros(len(test), bool)
    alert[np.argsort(-p)[:k]] = True
    row = {"Model": name, "PR-AUC": round(ap, 4), "Random guess": round(prev, 4), "Lift": round(ap / prev, 2),
           f"Recall @ top {ALERT_SHARE:.0%}": round(alert[test.y.values == 1].mean(), 3),
           f"Precision @ top {ALERT_SHARE:.0%}": round(test.y.values[alert].mean(), 3)}
    if test.Next_Cause.notna().any():
        for cause in ["degradation", "sudden", "site_power"]:
            mask = (test.y.values == 1) & (test.Next_Cause.values == cause)
            if mask.any():
                row[f"Recall {cause}"] = round(alert[mask].mean(), 3)
    rows.append(row)
res = pd.DataFrame(rows)
print("\nModel results on the test period (Lift = PR-AUC / random guess; 1.0 = no signal):")
print(res.to_string(index=False))

# ------------------------------------------------------------------ pre-outage test, by cause
kpi_idx = kpi.set_index(["Cell_ID", "Timestamp"]).sort_index()
recs = []
for o in out.itertuples():
    try:
        cell = kpi_idx.loc[o.Cell_ID]
    except KeyError:
        continue
    pre = cell.loc[o.Outage_Start - HORIZON: o.Outage_Start - pd.Timedelta(seconds=1), KPIS]
    ctl = cell.loc[o.Outage_Start - HORIZON - pd.Timedelta(days=7): o.Outage_Start - pd.Timedelta(days=7, seconds=1), KPIS]
    if len(pre) and len(ctl):
        d = (pre.mean() > ctl.mean()).astype(float)
        d["Cause"] = getattr(o, "Cause", "all")
        recs.append(d)
if recs:
    pre_tab = pd.DataFrame(recs)
    print("\nShare of outages with a HIGHER value in the 12 h before vs 7 days earlier (50% = no signal):")
    print((pre_tab.groupby("Cause")[KPIS].mean() * 100).round(1).to_string())

res.to_csv(folder.parent / (folder.name + "_signal_check.csv"), index=False)
print(f"\nSaved: {folder.parent / (folder.name + '_signal_check.csv')}")
