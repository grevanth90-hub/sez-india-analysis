from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import statsmodels.formula.api as smf

# ============================================================
# PATHS
# ============================================================
BASE = Path("/Users/ginkanagarevanthreddy/Downloads")
RAW = BASE / "sez_project" / "data" / "raw"
OUT = BASE / "sez_project" / "outputs_v2"
OUT.mkdir(parents=True, exist_ok=True)

districts_path = BASE / "india_district.geojson"
sez_path = BASE / "sez_data.csv"

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

# ============================================================
# HELPERS
# ============================================================
def starify(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def pct_effect(beta):
    return np.exp(beta) - 1

def winsorize_series(s, lower=0.01, upper=0.99):
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

# ============================================================
# LOAD DISTRICTS
# ============================================================
print("Loading districts...")
districts = gpd.read_file(districts_path)
districts = districts[["NAME_1", "NAME_2", "geometry"]].copy()
districts.columns = ["state", "district", "geometry"]
districts["state"] = districts["state"].apply(safe_text)
districts["district"] = districts["district"].apply(safe_text)
districts = districts.to_crs("EPSG:4326")
districts["point"] = districts.geometry.representative_point()

# ============================================================
# LOAD SEZ DATA
# ============================================================
print("Loading SEZ data...")
sez = pd.read_csv(sez_path)
sez["district"] = sez["district"].apply(safe_text)
sez["state"] = sez["state"].apply(safe_text)
sez["notification_year"] = pd.to_numeric(sez["notification_year"], errors="coerce")

sez = (
    sez.dropna(subset=["district", "state", "notification_year"])
       .groupby(["district", "state"], as_index=False)["notification_year"]
       .min()
)

# ============================================================
# EXTRACT NIGHTLIGHTS
# ============================================================
print("Extracting nightlights...")
panel_parts = []

for year, raster_path in raster_files.items():
    print(f"Processing {year}: {raster_path}")
    if not raster_path.exists():
        raise FileNotFoundError(f"Missing raster: {raster_path}")

    with rasterio.open(raster_path) as src:
        rows = []
        for _, row in districts.iterrows():
            x = row["point"].x
            y = row["point"].y
            try:
                val = list(src.sample([(x, y)]))[0][0]
                if pd.isna(val) or val < 0:
                    val = 0.0
            except Exception:
                val = 0.0

            rows.append({
                "district": row["district"],
                "state": row["state"],
                "year": year,
                "nightlights": float(val),
            })

        temp = pd.DataFrame(rows)
        print(temp["nightlights"].describe())
        panel_parts.append(temp)

panel = pd.concat(panel_parts, ignore_index=True)

# ============================================================
# MERGE SEZ TIMING
# ============================================================
print("Merging SEZ timing...")
panel = panel.merge(sez, on=["district", "state"], how="left")

panel["treated"] = np.where(panel["notification_year"].notna(), 1, 0)
panel["post"] = np.where(
    panel["notification_year"].notna() & (panel["year"] >= panel["notification_year"]),
    1,
    0
)
panel["did"] = panel["treated"] * panel["post"]

panel["log_nightlights"] = np.log1p(panel["nightlights"])
panel["nightlights_w"] = winsorize_series(panel["nightlights"], 0.01, 0.99)
panel["log_nightlights_w"] = np.log1p(panel["nightlights_w"])

# ============================================================
# BASIC CHECKS
# ============================================================
print("\nData check:")
print(panel[["district", "state", "year", "nightlights", "notification_year", "treated", "post", "did"]].head(12))

print("\nOverall nightlights summary:")
print(panel["nightlights"].describe())

print("\nTreatment counts:")
print(panel["treated"].value_counts(dropna=False))

# ============================================================
# TREATED VS CONTROL MEANS
# ============================================================
means_table = (
    panel.groupby(["year", "treated"])["nightlights"]
         .mean()
         .reset_index()
         .pivot(index="year", columns="treated", values="nightlights")
         .reset_index()
)
means_table.columns = ["year", "control_mean", "treated_mean"]
means_table["difference"] = means_table["treated_mean"] - means_table["control_mean"]
means_table.to_csv(OUT / "treated_control_means.csv", index=False)

# ============================================================
# REGRESSION BATTERY
# ============================================================
print("\nRunning regression battery...")

specs = {
    "twfe_log_main": "log_nightlights ~ did + C(district) + C(year)",
    "twfe_log_winsor": "log_nightlights_w ~ did + C(district) + C(year)",
    "twfe_levels_winsor": "nightlights_w ~ did + C(district) + C(year)",
    "twfe_log_stateyear": "log_nightlights ~ did + C(district) + C(state):C(year)",
}

results_list = []

for model_name, formula in specs.items():
    print(f"Running {model_name} ...")
    reg = smf.ols(formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["district"]}
    )

    beta = reg.params.get("did", np.nan)
    se = reg.bse.get("did", np.nan)
    pval = reg.pvalues.get("did", np.nan)

    results_list.append({
        "model": model_name,
        "coef_did": beta,
        "std_err_did": se,
        "p_value_did": pval,
        "stars": starify(pval),
        "pct_effect": pct_effect(beta) if pd.notna(beta) else np.nan,
        "nobs": reg.nobs,
        "r2": reg.rsquared,
    })

    coef_table = pd.DataFrame({
        "term": reg.params.index,
        "coef": reg.params.values,
        "std_err": reg.bse.values,
        "p_value": reg.pvalues.values,
    })
    coef_table.to_csv(OUT / f"{model_name}_full_results.csv", index=False)

reg_table = pd.DataFrame(results_list)
reg_table.to_csv(OUT / "regression_summary_table.csv", index=False)

# ============================================================
# PLACEBO TEST
# ============================================================
print("Running placebo test...")

panel_placebo = panel.copy()
panel_placebo["post_placebo_2004"] = np.where(panel_placebo["year"] >= 2004, 1, 0)
panel_placebo["did_placebo_2004"] = panel_placebo["treated"] * panel_placebo["post_placebo_2004"]

placebo_model = smf.ols(
    "log_nightlights ~ did_placebo_2004 + C(district) + C(year)",
    data=panel_placebo
).fit(
    cov_type="cluster",
    cov_kwds={"groups": panel_placebo["district"]}
)

placebo_out = pd.DataFrame([{
    "coef_placebo": placebo_model.params.get("did_placebo_2004", np.nan),
    "std_err": placebo_model.bse.get("did_placebo_2004", np.nan),
    "p_value": placebo_model.pvalues.get("did_placebo_2004", np.nan),
    "stars": starify(placebo_model.pvalues.get("did_placebo_2004", np.nan)),
    "nobs": placebo_model.nobs,
    "r2": placebo_model.rsquared,
}])
placebo_out.to_csv(OUT / "placebo_test_2004.csv", index=False)

# ============================================================
# PRETREND CHECK
# ============================================================
print("Running pretrend check...")

pre_df = panel[panel["year"].isin([2000, 2004, 2008, 2010])].copy()
pre_df["treated_x_trend"] = pre_df["treated"] * pre_df["year"]

pre_model = smf.ols(
    "log_nightlights ~ treated_x_trend + C(district) + C(year)",
    data=pre_df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": pre_df["district"]}
)

pretrend_out = pd.DataFrame([{
    "coef_treated_x_trend": pre_model.params.get("treated_x_trend", np.nan),
    "std_err": pre_model.bse.get("treated_x_trend", np.nan),
    "p_value": pre_model.pvalues.get("treated_x_trend", np.nan),
    "stars": starify(pre_model.pvalues.get("treated_x_trend", np.nan)),
    "nobs": pre_model.nobs,
    "r2": pre_model.rsquared,
}])
pretrend_out.to_csv(OUT / "pretrend_check.csv", index=False)

# ============================================================
# EVENT STUDY
# ============================================================
print("Running event study...")

event_df = panel.copy()
event_df["event_time"] = event_df["year"] - event_df["notification_year"]
event_df = event_df[event_df["treated"] == 1].copy()
event_df = event_df[event_df["event_time"].between(-8, 8)].copy()
event_df = event_df[event_df["event_time"] != -1].copy()

event_table = pd.DataFrame()

if len(event_df) > 0 and event_df["event_time"].nunique() > 1:
    event_model = smf.ols(
        "log_nightlights ~ C(event_time) + C(district) + C(year)",
        data=event_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": event_df["district"]}
    )

    rows = []
    for term in event_model.params.index:
        if "C(event_time)" in term:
            label = term.split("[T.")[-1].replace("]", "")
            try:
                ev = float(label)
            except ValueError:
                continue

            rows.append({
                "event_time": ev,
                "coef": event_model.params[term],
                "std_err": event_model.bse.get(term, np.nan),
                "p_value": event_model.pvalues.get(term, np.nan),
                "stars": starify(event_model.pvalues.get(term, np.nan)),
            })

    event_table = pd.DataFrame(rows).sort_values("event_time")

event_table.to_csv(OUT / "event_study.csv", index=False)

# ============================================================
# SAVE PANEL
# ============================================================
panel.to_csv(OUT / "panel_full.csv", index=False)

# ============================================================
# WRITE BRIEF
# ============================================================
with open(OUT / "results_brief.txt", "w") as f:
    f.write("SEZ PROJECT: MAIN RESULTS\n")
    f.write("=" * 70 + "\n\n")

    for _, row in reg_table.iterrows():
        f.write(f"Model: {row['model']}\n")
        f.write(f"  DID coef:   {row['coef_did']:.4f}{row['stars']}\n")
        f.write(f"  Std. err:   {row['std_err_did']:.4f}\n")
        f.write(f"  P-value:    {row['p_value_did']:.6f}\n")
        f.write(f"  Pct effect: {100 * row['pct_effect']:.2f}%\n")
        f.write(f"  N:          {int(row['nobs'])}\n")
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

    f.write("EVENT STUDY\n")
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
