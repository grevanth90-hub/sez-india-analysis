#!/usr/bin/env python3

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# ============================================================
# PATHS
# ============================================================

BASE = Path("/Users/ginkanagarevanthreddy/Downloads")
RAW = BASE / "sez_project" / "data" / "raw"
OUT = BASE / "sez_project" / "elite_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DISTRICTS_PATH = BASE / "india_district.geojson"
SEZ_PATH = BASE / "sez_data.csv"

RASTER_FILES = {
    2000: RAW / "harmonized_2000.tif",
    2004: RAW / "harmonized_2004.tif",
    2008: RAW / "harmonized_2008.tif",
    2010: RAW / "harmonized_2010.tif",
    2012: BASE / "viirs_2012.tif",
    2016: BASE / "viirs_2016.tif",
    2020: BASE / "viirs_2020.tif",
    2024: BASE / "viirs_2024.tif",
}

# ============================================================
# HELPERS
# ============================================================

def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def safe_sample(src, x, y):
    try:
        val = list(src.sample([(x, y)]))[0][0]
        if pd.isna(val) or val < 0:
            return 0.0
        return float(val)
    except Exception:
        return 0.0

def winsorize_series(s, lower=0.01, upper=0.99):
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def pct_effect_from_log(beta):
    return np.exp(beta) - 1

# ============================================================
# LOAD DISTRICTS
# ============================================================

print("Loading districts...")
districts = gpd.read_file(DISTRICTS_PATH)
districts = districts[["NAME_1", "NAME_2", "geometry"]].copy()
districts.columns = ["state", "district", "geometry"]
districts = districts.to_crs("EPSG:4326")

districts["district"] = districts["district"].astype(str).str.strip()
districts["state"] = districts["state"].astype(str).str.strip()

districts["district_clean"] = districts["district"].str.lower().str.strip()
districts["state_clean"] = districts["state"].str.lower().str.strip()

districts["centroid"] = districts.geometry.representative_point()
districts["district_id"] = np.arange(len(districts))

# ============================================================
# LOAD SEZ DATA
# ============================================================

print("Loading SEZ data...")
sez = pd.read_csv(SEZ_PATH)

sez["district"] = sez["district"].astype(str).str.strip()
sez["state"] = sez["state"].astype(str).str.strip()
sez["district_clean"] = sez["district"].str.lower().str.strip()
sez["state_clean"] = sez["state"].str.lower().str.strip()

# earliest treatment year per district
sez = (
    sez.groupby(["district_clean", "state_clean"], as_index=False)["notification_year"]
    .min()
)

# ============================================================
# EXTRACT NIGHTLIGHTS
# ============================================================

print("Extracting nightlights...")
panel_parts = []

for year, raster_path in RASTER_FILES.items():
    print(f"Processing {year}: {raster_path}")
    if not raster_path.exists():
        raise FileNotFoundError(f"Missing raster: {raster_path}")

    with rasterio.open(raster_path) as src:
        rows = []
        for _, row in districts.iterrows():
            x = row["centroid"].x
            y = row["centroid"].y
            val = safe_sample(src, x, y)

            rows.append({
                "district_id": row["district_id"],
                "district": row["district"],
                "state": row["state"],
                "district_clean": row["district_clean"],
                "state_clean": row["state_clean"],
                "year": year,
                "nightlights": val
            })

    tmp = pd.DataFrame(rows)
    print(tmp["nightlights"].describe())
    panel_parts.append(tmp)

df = pd.concat(panel_parts, ignore_index=True)

# ============================================================
# MERGE TREATMENT
# ============================================================

print("Merging SEZ timing...")
df = df.merge(
    sez,
    on=["district_clean", "state_clean"],
    how="left"
)

df["treated"] = df["notification_year"].notna().astype(int)
df["post"] = np.where(
    df["notification_year"].notna() & (df["year"] >= df["notification_year"]),
    1,
    0
)
df["did"] = df["treated"] * df["post"]

df["log_nightlights"] = np.log1p(df["nightlights"])
df["nightlights_winsor"] = winsorize_series(df["nightlights"])
df["log_nightlights_winsor"] = np.log1p(df["nightlights_winsor"])

# numeric time trend
year_map = {y: i for i, y in enumerate(sorted(df["year"].unique()))}
df["t"] = df["year"].map(year_map)

# state-specific trend variable
df["state_trend_group"] = df["state"]

print("\nData check:")
print(df[["district", "state", "year", "nightlights", "notification_year", "treated", "post", "did"]].head(12))

print("\nOverall nightlights summary:")
print(df["nightlights"].describe())

print("\nTreatment counts:")
print(df["treated"].value_counts())

# ============================================================
# MATCHING STEP
# Use pre-treatment baseline characteristics
# ============================================================

print("\nPreparing matched sample...")

pre_df = df[df["year"] == 2000].copy()

match_data = pre_df[[
    "district_id", "district", "state", "treated",
    "nightlights", "log_nightlights"
]].copy()

# add 2004 baseline too
pre_2004 = df[df["year"] == 2004][["district_id", "nightlights", "log_nightlights"]].copy()
pre_2004.columns = ["district_id", "nightlights_2004", "log_nightlights_2004"]

match_data = match_data.merge(pre_2004, on="district_id", how="left")
match_data["nightlights_2004"] = match_data["nightlights_2004"].fillna(0)
match_data["log_nightlights_2004"] = match_data["log_nightlights_2004"].fillna(0)

X = match_data[["log_nightlights", "log_nightlights_2004"]].fillna(0)
y = match_data["treated"]

ps_model = LogisticRegression(max_iter=2000)
ps_model.fit(X, y)
match_data["pscore"] = ps_model.predict_proba(X)[:, 1]

treated_units = match_data[match_data["treated"] == 1].copy()
control_units = match_data[match_data["treated"] == 0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_units[["pscore"]])

distances, indices = nn.kneighbors(treated_units[["pscore"]])

matched_controls = control_units.iloc[indices.flatten()].copy()
matched_controls = matched_controls.assign(match_weight=1.0)

treated_units = treated_units.copy()
treated_units["match_weight"] = 1.0

matched_ids = pd.concat([
    treated_units[["district_id", "match_weight"]],
    matched_controls[["district_id", "match_weight"]]
], ignore_index=True)

matched_ids["matched"] = 1
matched_ids = matched_ids.drop_duplicates(subset=["district_id"])

df = df.merge(
    matched_ids[["district_id", "matched", "match_weight"]],
    on="district_id",
    how="left"
)

df["matched"] = df["matched"].fillna(0)
df["match_weight"] = df["match_weight"].fillna(0)

matched_panel = df[df["matched"] == 1].copy()

print("Matched districts:", matched_panel["district_id"].nunique())
print("Matched observations:", len(matched_panel))

# ============================================================
# MAIN REGRESSION BATTERY
# ============================================================

print("\nRunning regression battery...")

reg_rows = []

specs = [
    {
        "name": "twfe_log_main",
        "formula": "log_nightlights ~ did + C(district_id) + C(year)",
        "data": df
    },
    {
        "name": "twfe_log_winsor",
        "formula": "log_nightlights_winsor ~ did + C(district_id) + C(year)",
        "data": df
    },
    {
        "name": "twfe_levels_winsor",
        "formula": "nightlights_winsor ~ did + C(district_id) + C(year)",
        "data": df
    },
    {
        "name": "twfe_log_state_trend",
        "formula": "log_nightlights ~ did + C(district_id) + C(year) + C(state):t",
        "data": df
    },
    {
        "name": "matched_twfe_log",
        "formula": "log_nightlights ~ did + C(district_id) + C(year)",
        "data": matched_panel
    }
]

for spec in specs:
    print("Running", spec["name"], "...")
    model = smf.ols(spec["formula"], data=spec["data"]).fit(
        cov_type="cluster",
        cov_kwds={"groups": spec["data"]["district_id"]}
    )

    coef = model.params.get("did", np.nan)
    se = model.bse.get("did", np.nan)
    p = model.pvalues.get("did", np.nan)

    reg_rows.append({
        "model": spec["name"],
        "coef_did": coef,
        "std_err_did": se,
        "p_value_did": p,
        "stars": stars(p),
        "pct_effect": pct_effect_from_log(coef) if "log" in spec["name"] and pd.notna(coef) else np.nan,
        "nobs": model.nobs,
        "r2": model.rsquared
    })

    full_res = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values
    })
    full_res.to_csv(OUT / f"{spec['name']}_full_results.csv", index=False)

reg_table = pd.DataFrame(reg_rows)
reg_table.to_csv(OUT / "regression_summary_table.csv", index=False)

# ============================================================
# PLACEBO TEST
# fake treatment starts in 2004 for actually treated districts
# ============================================================

print("Running placebo test...")
df["placebo_post_2004"] = (df["year"] >= 2004).astype(int)
df["placebo_did_2004"] = df["treated"] * df["placebo_post_2004"]

placebo_model = smf.ols(
    "log_nightlights ~ placebo_did_2004 + C(district_id) + C(year)",
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["district_id"]}
)

placebo_out = pd.DataFrame([{
    "coef_placebo": placebo_model.params.get("placebo_did_2004", np.nan),
    "std_err": placebo_model.bse.get("placebo_did_2004", np.nan),
    "p_value": placebo_model.pvalues.get("placebo_did_2004", np.nan),
    "stars": stars(placebo_model.pvalues.get("placebo_did_2004", np.nan)),
    "nobs": placebo_model.nobs,
    "r2": placebo_model.rsquared
}])
placebo_out.to_csv(OUT / "placebo_test_2004.csv", index=False)

# ============================================================
# PRETREND CHECK
# only pre-treatment years; interacted linear trend
# ============================================================

print("Running pretrend check...")
pretrend_df = df[
    (df["year"] < df["notification_year"]) | (df["treated"] == 0)
].copy()

pretrend_df["treated_trend"] = pretrend_df["treated"] * pretrend_df["t"]

pretrend_model = smf.ols(
    "log_nightlights ~ treated_trend + C(district_id) + C(year)",
    data=pretrend_df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": pretrend_df["district_id"]}
)

pretrend_out = pd.DataFrame([{
    "coef_treated_x_trend": pretrend_model.params.get("treated_trend", np.nan),
    "std_err": pretrend_model.bse.get("treated_trend", np.nan),
    "p_value": pretrend_model.pvalues.get("treated_trend", np.nan),
    "stars": stars(pretrend_model.pvalues.get("treated_trend", np.nan)),
    "nobs": pretrend_model.nobs,
    "r2": pretrend_model.rsquared
}])
pretrend_out.to_csv(OUT / "pretrend_check.csv", index=False)

# ============================================================
# EVENT STUDY
# create safe variable names
# omitted event time = -1
# ============================================================

print("Running event study...")
event_df = df[df["treated"] == 1].copy()
event_df["event_time"] = event_df["year"] - event_df["notification_year"]

event_df = event_df[(event_df["event_time"] >= -8) & (event_df["event_time"] <= 8)].copy()

event_names = []
for k in range(-8, 9):
    if k == -1:
        continue
    if k < 0:
        v = f"event_m{abs(k)}"
    else:
        v = f"event_{k}"
    event_df[v] = (event_df["event_time"] == k).astype(int)
    event_names.append(v)

if len(event_df) > 0 and len(event_names) > 0:
    rhs = " + ".join(event_names) + " + C(district_id) + C(year)"
    event_formula = "log_nightlights ~ " + rhs

    event_model = smf.ols(event_formula, data=event_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": event_df["district_id"]}
    )

    event_results = []
    for k in range(-8, 9):
        if k == -1:
            continue
        name = f"event_m{abs(k)}" if k < 0 else f"event_{k}"
        p = event_model.pvalues.get(name, np.nan)
        event_results.append({
            "event_time": k,
            "coef": event_model.params.get(name, np.nan),
            "std_err": event_model.bse.get(name, np.nan),
            "p_value": p,
            "stars": stars(p)
        })

    event_table = pd.DataFrame(event_results)
else:
    event_table = pd.DataFrame(columns=["event_time", "coef", "std_err", "p_value", "stars"])

event_table.to_csv(OUT / "event_study.csv", index=False)

# ============================================================
# TREATED VS CONTROL MEANS
# ============================================================

means_table = (
    df.groupby(["year", "treated"])["nightlights"]
    .mean()
    .reset_index()
    .pivot(index="year", columns="treated", values="nightlights")
    .reset_index()
)

means_table.columns = ["year", "control_mean", "treated_mean"]
means_table["difference"] = means_table["treated_mean"] - means_table["control_mean"]
means_table.to_csv(OUT / "treated_control_means.csv", index=False)

# ============================================================
# SAVE PANEL
# ============================================================

df.to_csv(OUT / "panel_full.csv", index=False)

# ============================================================
# WRITE BRIEF RESULTS FILE
# ============================================================

with open(OUT / "results_brief.txt", "w") as f:
    f.write("SEZ PROJECT: ELITE EMPIRICAL OUTPUT\n")
    f.write("=" * 70 + "\n\n")

    f.write("MAIN REGRESSIONS\n")
    f.write("-" * 70 + "\n")
    for _, row in reg_table.iterrows():
        f.write(f"Model: {row['model']}\n")
        f.write(f"  DID coef:   {row['coef_did']:.6f}{row['stars']}\n")
        f.write(f"  Std. err:   {row['std_err_did']:.6f}\n")
        f.write(f"  P-value:    {row['p_value_did']:.6f}\n")
        if pd.notna(row["pct_effect"]):
            f.write(f"  Pct effect: {100 * row['pct_effect']:.2f}%\n")
        f.write(f"  N:          {row['nobs']}\n")
        f.write(f"  R2:         {row['r2']:.4f}\n\n")

    f.write("PLACEBO TEST (fake post = 2004)\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"coef={placebo_out.loc[0, 'coef_placebo']:.6f}{placebo_out.loc[0, 'stars']}, "
        f"se={placebo_out.loc[0, 'std_err']:.6f}, "
        f"p={placebo_out.loc[0, 'p_value']:.6f}\n\n"
    )

    f.write("PRETREND CHECK\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"coef={pretrend_out.loc[0, 'coef_treated_x_trend']:.6f}{pretrend_out.loc[0, 'stars']}, "
        f"se={pretrend_out.loc[0, 'std_err']:.6f}, "
        f"p={pretrend_out.loc[0, 'p_value']:.6f}\n\n"
    )

    f.write("EVENT STUDY (omitted event time = -1)\n")
    f.write("-" * 70 + "\n")
    if len(event_table) > 0:
        for _, r in event_table.iterrows():
            f.write(
                f"event_time={int(r['event_time']):>2}: "
                f"coef={r['coef']:.6f}{r['stars']}, "
                f"se={r['std_err']:.6f}, "
                f"p={r['p_value']:.6f}\n"
            )
    else:
        f.write("No event-study estimates produced.\n")

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MAIN RESULTS")
print("=" * 80)
print(reg_table)

print("\nPLACEBO TEST")
print(placebo_out)

print("\nPRETREND CHECK")
print(pretrend_out)

print("\nEVENT STUDY")
print(event_table if len(event_table) > 0 else "No event-study output.")

print("\nSaved files:")
for p in sorted(OUT.glob("*")):
    print(p)

print("\nDONE.")
