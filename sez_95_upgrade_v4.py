#!/usr/bin/env python3
# ============================================================
# SEZ  UPGRADE V4 (FIXED EVENT STUDY)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path("/Users/ginkanagarevanthreddy/Downloads")
RAW = BASE / "sez_project/data/raw"
OUT = BASE / "sez_project/outputs_v4"
OUT.mkdir(parents=True, exist_ok=True)

raster_files = {
    2000: RAW / "harmonized_2000.tif",
    2004: RAW / "harmonized_2004.tif",
    2008: RAW / "harmonized_2008.tif",
    2010: RAW / "harmonized_2010.tif",
    2012: BASE / "viirs_2012.tif",
    2016: BASE / "viirs_2016.tif",
    2020: BASE / "viirs_2020.tif",
    2024: BASE / "viirs_2024.tif",
}

districts_path = BASE / "india_district.geojson"
sez_path = BASE / "sez_data.csv"

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

def winsorize_series(s, lower=0.01, upper=0.99):
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)

def pct_effect(beta):
    if pd.isna(beta):
        return np.nan
    return np.exp(beta) - 1

# ============================================================
# LOAD DISTRICTS
# ============================================================

print("Loading districts...")
districts = gpd.read_file(districts_path)
districts = districts[["NAME_1", "NAME_2", "geometry"]].copy()
districts.columns = ["state", "district", "geometry"]
districts = districts.to_crs("EPSG:4326")
districts["centroid"] = districts.geometry.representative_point()
districts["district_id"] = np.arange(len(districts))

# ============================================================
# LOAD SEZ DATA
# ============================================================

print("Loading SEZ data...")
sez = pd.read_csv(sez_path)
sez = sez.groupby(["district", "state"], as_index=False)["notification_year"].min()

# ============================================================
# EXTRACT NIGHTLIGHTS
# ============================================================

print("Extracting nightlights...")
panel_data = []

for year, raster_path in raster_files.items():
    print(f"Processing {year}: {raster_path}")
    with rasterio.open(raster_path) as src:
        samples = []
        for _, row in districts.iterrows():
            x, y = row["centroid"].x, row["centroid"].y
            try:
                val = list(src.sample([(x, y)]))[0][0]
                if pd.isna(val) or val < 0:
                    val = 0.0
            except Exception:
                val = 0.0

            samples.append({
                "district": row["district"],
                "state": row["state"],
                "district_id": row["district_id"],
                "year": year,
                "nightlights": float(val)
            })

    tmp = pd.DataFrame(samples)
    print(tmp["nightlights"].describe())
    panel_data.append(tmp)

df = pd.concat(panel_data, ignore_index=True)
df["log_nightlights"] = np.log1p(df["nightlights"])
df["nightlights_winsor"] = winsorize_series(df["nightlights"])
df["log_nightlights_winsor"] = np.log1p(df["nightlights_winsor"])

# ============================================================
# MERGE SEZ TIMING
# ============================================================

print("Merging SEZ timing...")
df = df.merge(sez, on=["district", "state"], how="left")

df["treated"] = np.where(df["notification_year"].notna(), 1, 0)
df["post"] = np.where(
    df["notification_year"].notna() & (df["year"] >= df["notification_year"]),
    1,
    0
)
df["did"] = df["treated"] * df["post"]

print("\nData check:")
print(df[["district", "state", "year", "nightlights", "notification_year", "treated", "post", "did"]].head(12))

print("\nOverall nightlights summary:")
print(df["nightlights"].describe())

print("\nTreatment counts:")
print(df["treated"].value_counts())

# ============================================================
# REGRESSION BATTERY
# ============================================================

print("\nRunning regression battery...")

results = []

# 1. Main TWFE log
m1 = smf.ols(
    "log_nightlights ~ did + C(district_id) + C(year)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["district_id"]})

results.append({
    "model": "twfe_log_main",
    "coef_did": m1.params.get("did", np.nan),
    "std_err_did": m1.bse.get("did", np.nan),
    "p_value_did": m1.pvalues.get("did", np.nan),
    "stars": stars(m1.pvalues.get("did", np.nan)),
    "pct_effect": pct_effect(m1.params.get("did", np.nan)),
    "nobs": m1.nobs,
    "r2": m1.rsquared
})

# 2. Winsorized log
m2 = smf.ols(
    "log_nightlights_winsor ~ did + C(district_id) + C(year)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["district_id"]})

results.append({
    "model": "twfe_log_winsor",
    "coef_did": m2.params.get("did", np.nan),
    "std_err_did": m2.bse.get("did", np.nan),
    "p_value_did": m2.pvalues.get("did", np.nan),
    "stars": stars(m2.pvalues.get("did", np.nan)),
    "pct_effect": pct_effect(m2.params.get("did", np.nan)),
    "nobs": m2.nobs,
    "r2": m2.rsquared
})

# 3. Levels winsor
m3 = smf.ols(
    "nightlights_winsor ~ did + C(district_id) + C(year)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["district_id"]})

results.append({
    "model": "twfe_levels_winsor",
    "coef_did": m3.params.get("did", np.nan),
    "std_err_did": m3.bse.get("did", np.nan),
    "p_value_did": m3.pvalues.get("did", np.nan),
    "stars": stars(m3.pvalues.get("did", np.nan)),
    "pct_effect": np.nan,
    "nobs": m3.nobs,
    "r2": m3.rsquared
})

# 4. State-year FE
df["state_year"] = df["state"].astype(str) + "_" + df["year"].astype(str)

m4 = smf.ols(
    "log_nightlights ~ did + C(district_id) + C(state_year)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["district_id"]})

results.append({
    "model": "twfe_log_stateyear",
    "coef_did": m4.params.get("did", np.nan),
    "std_err_did": m4.bse.get("did", np.nan),
    "p_value_did": m4.pvalues.get("did", np.nan),
    "stars": stars(m4.pvalues.get("did", np.nan)),
    "pct_effect": pct_effect(m4.params.get("did", np.nan)),
    "nobs": m4.nobs,
    "r2": m4.rsquared
})

reg_table = pd.DataFrame(results)

# ============================================================
# PLACEBO TEST
# ============================================================

print("Running placebo test...")
df["fake_post_2004"] = (df["year"] >= 2004).astype(int)
df["placebo"] = df["treated"] * df["fake_post_2004"]

placebo_model = smf.ols(
    "log_nightlights ~ placebo + C(district_id) + C(year)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["district_id"]})

placebo_out = pd.DataFrame([{
    "coef_placebo": placebo_model.params.get("placebo", np.nan),
    "std_err": placebo_model.bse.get("placebo", np.nan),
    "p_value": placebo_model.pvalues.get("placebo", np.nan),
    "stars": stars(placebo_model.pvalues.get("placebo", np.nan)),
    "nobs": placebo_model.nobs,
    "r2": placebo_model.rsquared
}])

# ============================================================
# PRETREND CHECK
# ============================================================

print("Running pretrend check...")
pre_df = df[
    (df["treated"] == 0) |
    ((df["treated"] == 1) & (df["year"] < df["notification_year"]))
].copy()

pre_df["time_trend"] = pre_df["year"] - pre_df["year"].min()
pre_df["treated_x_trend"] = pre_df["treated"] * pre_df["time_trend"]

pretrend_model = smf.ols(
    "log_nightlights ~ treated_x_trend + C(district_id) + C(year)",
    data=pre_df
).fit(cov_type="cluster", cov_kwds={"groups": pre_df["district_id"]})

pretrend_out = pd.DataFrame([{
    "coef_treated_x_trend": pretrend_model.params.get("treated_x_trend", np.nan),
    "std_err": pretrend_model.bse.get("treated_x_trend", np.nan),
    "p_value": pretrend_model.pvalues.get("treated_x_trend", np.nan),
    "stars": stars(pretrend_model.pvalues.get("treated_x_trend", np.nan)),
    "nobs": pretrend_model.nobs,
    "r2": pretrend_model.rsquared
}])

# ============================================================
# EVENT STUDY
# ============================================================

print("Running event study...")

event_df = df[df["treated"] == 1].copy()
event_df["event_time"] = event_df["year"] - event_df["notification_year"]
event_df = event_df[(event_df["event_time"] >= -8) & (event_df["event_time"] <= 8)].copy()

event_table = pd.DataFrame(columns=["event_time", "coef", "std_err", "p_value", "stars"])

if len(event_df) > 0:
    event_df["event_time_str"] = event_df["event_time"].astype(int).astype(str)
    event_df = event_df[event_df["event_time_str"] != "-1"].copy()

    if event_df["event_time_str"].nunique() >= 2:
        event_model = smf.ols(
            "log_nightlights ~ C(event_time_str) + C(district_id) + C(year)",
            data=event_df
        ).fit(cov_type="cluster", cov_kwds={"groups": event_df["district_id"]})

        rows = []
        for term in event_model.params.index:
            if term.startswith("C(event_time_str)"):
                label = term.split("T.")[-1].replace("]", "")
                try:
                    et = int(label)
                except Exception:
                    continue
                pval = event_model.pvalues.get(term, np.nan)
                rows.append({
                    "event_time": et,
                    "coef": event_model.params.get(term, np.nan),
                    "std_err": event_model.bse.get(term, np.nan),
                    "p_value": pval,
                    "stars": stars(pval)
                })

        event_table = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

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

# ============================================================
# SAVE OUTPUTS
# ============================================================

df.to_csv(OUT / "panel_full.csv", index=False)
reg_table.to_csv(OUT / "regression_summary_table.csv", index=False)
placebo_out.to_csv(OUT / "placebo_test_2004.csv", index=False)
pretrend_out.to_csv(OUT / "pretrend_check.csv", index=False)
event_table.to_csv(OUT / "event_study.csv", index=False)
means_table.to_csv(OUT / "treated_control_means.csv", index=False)

pd.DataFrame({
    "term": m1.params.index,
    "coef": m1.params.values,
    "std_err": m1.bse.values,
    "p_value": m1.pvalues.values
}).to_csv(OUT / "twfe_log_main_full_results.csv", index=False)

pd.DataFrame({
    "term": m2.params.index,
    "coef": m2.params.values,
    "std_err": m2.bse.values,
    "p_value": m2.pvalues.values
}).to_csv(OUT / "twfe_log_winsor_full_results.csv", index=False)

pd.DataFrame({
    "term": m3.params.index,
    "coef": m3.params.values,
    "std_err": m3.bse.values,
    "p_value": m3.pvalues.values
}).to_csv(OUT / "twfe_levels_winsor_full_results.csv", index=False)

pd.DataFrame({
    "term": m4.params.index,
    "coef": m4.params.values,
    "std_err": m4.bse.values,
    "p_value": m4.pvalues.values
}).to_csv(OUT / "twfe_log_stateyear_full_results.csv", index=False)

with open(OUT / "results_brief.txt", "w") as f:
    f.write("SEZ PROJECT: MAIN RESULTS\n")
    f.write("=" * 70 + "\n\n")

    for _, row in reg_table.iterrows():
        f.write(f"Model: {row['model']}\n")
        f.write(f"  DID coef:   {row['coef_did']:.4f}{row['stars']}\n")
        f.write(f"  Std. err:   {row['std_err_did']:.4f}\n")
        f.write(f"  P-value:    {row['p_value_did']:.6f}\n")
        if not pd.isna(row["pct_effect"]):
            f.write(f"  Pct effect: {100 * row['pct_effect']:.2f}%\n")
        f.write(f"  N:          {row['nobs']:.0f}\n")
        f.write(f"  R2:         {row['r2']:.4f}\n\n")

    f.write("PLACEBO TEST (fake post = 2004)\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"coef={placebo_out.loc[0, 'coef_placebo']:.4f}{placebo_out.loc[0, 'stars']}, "
        f"se={placebo_out.loc[0, 'std_err']:.4f}, "
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
                f"coef={r['coef']:.4f}{r['stars']}, "
                f"se={r['std_err']:.4f}, "
                f"p={r['p_value']:.6f}\n"
            )
    else:
        f.write("No event-study estimates produced.\n")

# ============================================================
# CONSOLE SUMMARY
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
if len(event_table) > 0:
    print(event_table)
else:
    print("No event-study output.")

print("\nSaved files:")
for p in sorted(OUT.glob("*")):
    print(p)

print("\nDONE.")
