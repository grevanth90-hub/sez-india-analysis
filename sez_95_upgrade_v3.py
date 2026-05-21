# ============================================================
# SEZ PROJECT — UPGRADE V3
# Stronger identification with district trends
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE = Path("/Users/ginkanagarevanthreddy/Downloads")
RAW = BASE / "sez_project" / "data" / "raw"
OUT = BASE / "sez_project" / "outputs_v3"
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
def star(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def pct_effect(beta):
    return np.exp(beta) - 1

def winsorize_series(s, lower=0.01, upper=0.99):
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def run_model(name, formula, data, cluster_col):
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col]}
    )
    return {
        "name": name,
        "model": model
    }

def extract_main_result(model_obj, varname="did"):
    model = model_obj["model"]
    if varname not in model.params.index:
        return {
            "model": model_obj["name"],
            "coef_did": np.nan,
            "std_err_did": np.nan,
            "p_value_did": np.nan,
            "stars": "",
            "pct_effect": np.nan,
            "nobs": model.nobs,
            "r2": model.rsquared
        }
    b = model.params[varname]
    se = model.bse[varname]
    p = model.pvalues[varname]
    return {
        "model": model_obj["name"],
        "coef_did": b,
        "std_err_did": se,
        "p_value_did": p,
        "stars": star(p),
        "pct_effect": pct_effect(b),
        "nobs": model.nobs,
        "r2": model.rsquared
    }

# ============================================================
# LOAD DISTRICTS
# ============================================================
print("Loading districts...")
districts = gpd.read_file(DISTRICTS_PATH)
districts = districts[["NAME_1", "NAME_2", "geometry"]].copy()
districts.columns = ["state", "district", "geometry"]
districts = districts.to_crs("EPSG:4326")

# representative point is safer than centroid for irregular polygons
districts["sample_point"] = districts.geometry.representative_point()

# clean text
districts["district"] = districts["district"].astype(str).str.strip()
districts["state"] = districts["state"].astype(str).str.strip()

# district id for FE/trends
districts["district_id"] = districts["state"] + "___" + districts["district"]

# ============================================================
# LOAD SEZ DATA
# ============================================================
print("Loading SEZ data...")
sez = pd.read_csv(SEZ_PATH)
sez["district"] = sez["district"].astype(str).str.strip()
sez["state"] = sez["state"].astype(str).str.strip()

# earliest notification year by district-state
sez = (
    sez.groupby(["district", "state"], as_index=False)["notification_year"]
    .min()
)

# ============================================================
# EXTRACT NIGHTLIGHTS BY REPRESENTATIVE POINT
# ============================================================
print("Extracting nightlights...")
panel_parts = []

for year, raster_path in RASTER_FILES.items():
    print(f"Processing {year}: {raster_path}")
    if not raster_path.exists():
        raise FileNotFoundError(f"Missing raster: {raster_path}")

    rows = []
    with rasterio.open(raster_path) as src:
        for _, row in districts.iterrows():
            x = row["sample_point"].x
            y = row["sample_point"].y

            try:
                val = list(src.sample([(x, y)]))[0][0]
                if pd.isna(val) or val < 0:
                    val = 0.0
            except Exception:
                val = 0.0

            rows.append({
                "district": row["district"],
                "state": row["state"],
                "district_id": row["district_id"],
                "year": year,
                "nightlights": float(val),
            })

    tmp = pd.DataFrame(rows)
    print(tmp["nightlights"].describe())
    panel_parts.append(tmp)

panel = pd.concat(panel_parts, ignore_index=True)

# ============================================================
# MERGE TREATMENT
# ============================================================
print("Merging SEZ timing...")
panel = panel.merge(sez, on=["district", "state"], how="left")

panel["treated"] = panel["notification_year"].notna().astype(int)
panel["post"] = np.where(
    panel["notification_year"].notna() & (panel["year"] >= panel["notification_year"]),
    1,
    0
)
panel["did"] = panel["treated"] * panel["post"]

# time index for trends
panel["t"] = panel["year"] - panel["year"].min()

# transformed outcomes
panel["log_nightlights"] = np.log1p(panel["nightlights"])
panel["nightlights_w"] = winsorize_series(panel["nightlights"])
panel["log_nightlights_w"] = np.log1p(panel["nightlights_w"])

# event time
panel["event_time"] = panel["year"] - panel["notification_year"]

print("\nData check:")
print(panel[[
    "district", "state", "year", "nightlights",
    "notification_year", "treated", "post", "did"
]].head(12))

print("\nOverall nightlights summary:")
print(panel["nightlights"].describe())

print("\nTreatment counts:")
print(panel["treated"].value_counts(dropna=False))

# ============================================================
# BASELINE REGRESSIONS
# ============================================================
print("\nRunning regression battery...")

models = []

# 1. Main TWFE log
models.append(run_model(
    "twfe_log_main",
    "log_nightlights ~ did + C(district_id) + C(year)",
    panel,
    "district_id"
))

# 2. Winsorized log
models.append(run_model(
    "twfe_log_winsor",
    "log_nightlights_w ~ did + C(district_id) + C(year)",
    panel,
    "district_id"
))

# 3. Levels winsor
models.append(run_model(
    "twfe_levels_winsor",
    "nightlights_w ~ did + C(district_id) + C(year)",
    panel,
    "district_id"
))

# 4. State-year FE
models.append(run_model(
    "twfe_log_stateyear",
    "log_nightlights ~ did + C(district_id) + C(state):C(year)",
    panel,
    "district_id"
))

# 5. District linear trends
models.append(run_model(
    "twfe_log_district_trends",
    "log_nightlights ~ did + C(district_id) + C(year) + C(district_id):t",
    panel,
    "district_id"
))

# 6. State-year FE + district trends
models.append(run_model(
    "twfe_log_stateyear_dtrends",
    "log_nightlights ~ did + C(district_id) + C(state):C(year) + C(district_id):t",
    panel,
    "district_id"
))

reg_table = pd.DataFrame([extract_main_result(m, "did") for m in models])
reg_table.to_csv(OUT / "regression_summary_table.csv", index=False)

# save full coefficient tables
for m in models:
    model = m["model"]
    out_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values
    })
    out_df.to_csv(OUT / f"{m['name']}_full_results.csv", index=False)

# ============================================================
# PLACEBO TEST
# fake treatment post begins in 2004
# ============================================================
print("Running placebo test...")
panel["post_placebo_2004"] = (panel["year"] >= 2004).astype(int)
panel["did_placebo_2004"] = panel["treated"] * panel["post_placebo_2004"]

placebo_model = smf.ols(
    "log_nightlights ~ did_placebo_2004 + C(district_id) + C(year)",
    data=panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": panel["district_id"]}
)

placebo_out = pd.DataFrame([{
    "coef_placebo": placebo_model.params.get("did_placebo_2004", np.nan),
    "std_err": placebo_model.bse.get("did_placebo_2004", np.nan),
    "p_value": placebo_model.pvalues.get("did_placebo_2004", np.nan),
    "stars": star(placebo_model.pvalues.get("did_placebo_2004", np.nan)),
    "nobs": placebo_model.nobs,
    "r2": placebo_model.rsquared
}])
placebo_out.to_csv(OUT / "placebo_test_2004.csv", index=False)

# ============================================================
# PRETREND CHECK
# only pre-2012 years, treated x linear trend
# ============================================================
print("Running pretrend check...")
pre_df = panel[panel["year"] <= 2010].copy()
pre_df["treated_x_t"] = pre_df["treated"] * pre_df["t"]

pretrend_model = smf.ols(
    "log_nightlights ~ treated_x_t + C(district_id) + C(year)",
    data=pre_df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": pre_df["district_id"]}
)

pretrend_out = pd.DataFrame([{
    "coef_treated_x_trend": pretrend_model.params.get("treated_x_t", np.nan),
    "std_err": pretrend_model.bse.get("treated_x_t", np.nan),
    "p_value": pretrend_model.pvalues.get("treated_x_t", np.nan),
    "stars": star(pretrend_model.pvalues.get("treated_x_t", np.nan)),
    "nobs": pretrend_model.nobs,
    "r2": pretrend_model.rsquared
}])
pretrend_out.to_csv(OUT / "pretrend_check.csv", index=False)

# ============================================================
# EVENT STUDY
# omit event_time = -1
# keep reasonable window
# ============================================================
print("Running event study...")

event_df = panel[panel["treated"] == 1].copy()
event_df = event_df[event_df["event_time"].between(-8, 8)].copy()
event_df = event_df[event_df["event_time"] != -1].copy()

event_table = pd.DataFrame()

if len(event_df) > 0 and event_df["event_time"].nunique() > 1:
    # create categorical labels manually
    for k in sorted(event_df["event_time"].dropna().unique()):
        if k == -1:
            continue
        event_df[f"event_{int(k)}"] = (event_df["event_time"] == k).astype(int)

    event_terms = " + ".join(
        [f"event_{int(k)}" for k in sorted(event_df["event_time"].dropna().unique()) if k != -1]
    )

    event_formula = f"log_nightlights ~ {event_terms} + C(district_id) + C(year)"
    event_model = smf.ols(event_formula, data=event_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": event_df['district_id']}
    )

    rows = []
    for k in sorted(event_df["event_time"].dropna().unique()):
        if k == -1:
            continue
        term = f"event_{int(k)}"
        rows.append({
            "event_time": k,
            "coef": event_model.params.get(term, np.nan),
            "std_err": event_model.bse.get(term, np.nan),
            "p_value": event_model.pvalues.get(term, np.nan),
            "stars": star(event_model.pvalues.get(term, np.nan))
        })

    event_table = pd.DataFrame(rows)

event_table.to_csv(OUT / "event_study.csv", index=False)

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

if 0 in means_table.columns and 1 in means_table.columns:
    means_table = means_table.rename(columns={0: "control_mean", 1: "treated_mean"})
else:
    if "control_mean" not in means_table.columns:
        means_table["control_mean"] = np.nan
    if "treated_mean" not in means_table.columns:
        means_table["treated_mean"] = np.nan

means_table["difference"] = means_table["treated_mean"] - means_table["control_mean"]
means_table.to_csv(OUT / "treated_control_means.csv", index=False)

# ============================================================
# SAVE PANEL
# ============================================================
panel.to_csv(OUT / "panel_full.csv", index=False)

# ============================================================
# PAPER-READY TEXT OUTPUT
# ============================================================
with open(OUT / "results_brief.txt", "w") as f:
    f.write("SEZ PROJECT: MAIN RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("BASELINE + ROBUSTNESS REGRESSIONS\n")
    f.write("-" * 70 + "\n")
    for _, row in reg_table.iterrows():
        f.write(f"Model: {row['model']}\n")
        f.write(f"  DID coef:   {row['coef_did']:.4f}{row['stars']}\n")
        f.write(f"  Std. err:   {row['std_err_did']:.4f}\n")
        f.write(f"  P-value:    {row['p_value_did']:.6f}\n")
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
