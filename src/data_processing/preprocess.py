"""
=============================================================
  Indian Railway Delay Data - Cleaning & Preprocessing
=============================================================
Inputs:
  - 02501.csv          : Station-level delay statistics (20 delay files)
  - schedules.csv      : Train schedules with station lists
  - Train_List.csv     : Master list of trains with type
  - EXP-TRAINS.json    : Express train routes & timings
  - PASS-TRAINS.json   : Passenger train routes & timings
  - SF-TRAINS.json     : Superfast train routes & timings

Outputs (saved to ./preprocessed/):
  - delay_clean.csv
  - schedules_clean.csv
  - train_list_clean.csv
  - train_routes_clean.csv   (merged from 3 JSON files)
  - master_merged.csv        (all sources joined)
  - preprocessing_report.txt
"""

import os
import json
import re
import ast
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

SCHED_CSV = f"{BASE}/schedule/schedules.csv"
TRAIN_CSV = f"{BASE}/train_list/Train_List.csv"

EXP_JSON  = f"{BASE}/trains/EXP-TRAINS.json"
PASS_JSON = f"{BASE}/trains/PASS-TRAINS.json"
SF_JSON   = f"{BASE}/trains/SF-TRAINS.json"
report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def strip_str_cols(df):
    """Strip whitespace from all string columns."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def standardize_station_code(code):
    """Uppercase and strip station codes."""
    if pd.isna(code):
        return np.nan
    return str(code).strip().upper()


def parse_time_to_minutes(t):
    """Convert 'HH:MM' → total minutes from midnight. Returns NaN for Source/Destination."""
    if pd.isna(t) or str(t).strip().lower() in ("source", "destination", ""):
        return np.nan
    try:
        h, m = str(t).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return np.nan


def parse_distance_km(d):
    """Convert '25 kms' → 25 (int). Returns NaN on failure."""
    if pd.isna(d):
        return np.nan
    try:
        return int(str(d).replace("kms", "").replace("km", "").strip())
    except Exception:
        return np.nan


def running_days_str(days_dict):
    """Convert {'MON': True, 'TUE': False, ...} → 'MON,WED,FRI'."""
    if not isinstance(days_dict, dict):
        return ""
    return ",".join(k for k, v in days_dict.items() if v)


# # ══════════════════════════════════════════════════════════════════════════════
# # 1. DELAY DATA  (02501.csv)
# # ══════════════════════════════════════════════════════════════════════════════
# log("\n" + "="*60)
# log("1. DELAY DATA (02501.csv)")
# log("="*60)

# delay = pd.read_csv(DELAY_CSV)
# log(f"  Raw shape: {delay.shape}")
# log(f"  Columns  : {list(delay.columns)}")

# # --- Rename columns for consistency ---
# delay.columns = [
#     "station_code",
#     "station_name",
#     "avg_delay_min",
#     "pct_right_time",
#     "pct_slight_delay",
#     "pct_significant_delay",
#     "pct_cancelled_unknown",
# ]

# # --- Strip whitespace ---
# delay = strip_str_cols(delay)

# # --- Standardize station codes ---
# delay["station_code"] = delay["station_code"].apply(standardize_station_code)
# delay["station_name"] = delay["station_name"].str.title()

# # --- Validate percentage columns (should sum to ~100) ---
# pct_cols = ["pct_right_time", "pct_slight_delay", "pct_significant_delay", "pct_cancelled_unknown"]
# delay["pct_total_check"] = delay[pct_cols].sum(axis=1).round(2)

# # --- Derive delay severity label ---
# def delay_category(row):
#     if row["avg_delay_min"] <= 15:
#         return "On Time"
#     elif row["avg_delay_min"] <= 60:
#         return "Slight Delay"
#     else:
#         return "Significantly Delayed"

# delay["delay_category"] = delay.apply(delay_category, axis=1)

# # --- Null check ---
# nulls = delay.isnull().sum()
# log(f"  Nulls after clean:\n{nulls[nulls>0].to_string() if nulls.sum() > 0 else '  None'}")
# log(f"  Clean shape: {delay.shape}")
# log(f"  Sample:\n{delay.head(3).to_string()}")

# delay.to_csv(f"{OUT_DIR}/delay_clean.csv", index=False)
# log(f"  ✓ Saved → preprocessed/delay_clean.csv")



# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN LIST  (Train_List.csv)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("2. TRAIN LIST (Train_List.csv)")
log("="*60)

train_list = pd.read_csv(TRAIN_CSV)
log(f"  Raw shape: {train_list.shape}")

# --- Rename ---
train_list.columns = ["train_number", "train_name", "from_station", "to_station", "train_type"]

# --- Strip ---
train_list = strip_str_cols(train_list)

# --- Train number as zero-padded 5-digit string (matches JSON keys) ---
train_list["train_number_str"] = train_list["train_number"].astype(str).str.zfill(5)

# --- Standardize station codes ---
train_list["from_station"] = train_list["from_station"].apply(standardize_station_code)
train_list["to_station"]   = train_list["to_station"].apply(standardize_station_code)

# --- Standardize train type ---
train_list["train_type"] = train_list["train_type"].str.strip().str.title()

# --- Duplicates ---
dups = train_list.duplicated(subset="train_number").sum()
log(f"  Duplicate train numbers: {dups}")
train_list = train_list.drop_duplicates(subset="train_number")

log(f"  Train type distribution:\n{train_list['train_type'].value_counts().to_string()}")
log(f"  Clean shape: {train_list.shape}")

train_list.to_csv(f"{OUT_DIR}/train_list_clean.csv", index=False)
#log(f"  ✓ Saved → preprocessed/train_list_clean.csv")
log(f"  ✓ Saved → data/processed/train_list_clean.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SCHEDULES  (schedules.csv)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("3. SCHEDULES (schedules.csv)")
log("="*60)

sched = pd.read_csv(SCHED_CSV)
log(f"  Raw shape: {sched.shape}")

# --- Rename days columns ---
sched = sched.rename(columns={
    "trainNumber"     : "train_number",
    "trainName"       : "train_name",
    "stationFrom"     : "from_station",
    "stationTo"       : "to_station",
    "trainRunsOnMon"  : "runs_mon",
    "trainRunsOnTue"  : "runs_tue",
    "trainRunsOnWed"  : "runs_wed",
    "trainRunsOnThu"  : "runs_thu",
    "trainRunsOnFri"  : "runs_fri",
    "trainRunsOnSat"  : "runs_sat",
    "trainRunsOnSun"  : "runs_sun",
    "timeStamp"       : "timestamp",
    "stationList"     : "station_list_raw",
})

# --- Strip strings ---
sched = strip_str_cols(sched)

# --- Train number as zero-padded string ---
sched["train_number_str"] = sched["train_number"].astype(str).str.zfill(5)

# --- Parse 'Y'/'N' day flags to bool ---
day_cols = ["runs_mon","runs_tue","runs_wed","runs_thu","runs_fri","runs_sat","runs_sun"]
for col in day_cols:
    sched[col] = sched[col].map({"Y": True, "N": False, True: True, False: False}).fillna(False)

# --- Build comma-separated running days string ---
day_map = dict(zip(day_cols, ["MON","TUE","WED","THU","FRI","SAT","SUN"]))
def running_days_from_cols(row):
    return ",".join(day_map[c] for c in day_cols if row[c])
sched["running_days"] = sched.apply(running_days_from_cols, axis=1)

# --- Parse station_list (stored as JSON string) ---
def safe_parse_station_list(raw):
    if pd.isna(raw):
        return []
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []

sched["station_list"] = sched["station_list_raw"].apply(safe_parse_station_list)
sched["num_stations"]  = sched["station_list"].apply(len)

# --- Parse timestamp ---
sched["timestamp"] = pd.to_datetime(sched["timestamp"], errors="coerce")

# --- Standardize station codes ---
sched["from_station"] = sched["from_station"].apply(standardize_station_code)
sched["to_station"]   = sched["to_station"].apply(standardize_station_code)

# --- Drop raw stationList column (already parsed) ---
sched_clean = sched.drop(columns=["station_list_raw", "station_list"])

# --- Null check ---
nulls = sched_clean.isnull().sum()
log(f"  Nulls after clean:\n{nulls[nulls>0].to_string() if nulls.sum() > 0 else '  None'}")
log(f"  Clean shape: {sched_clean.shape}")

sched_clean.to_csv(f"{OUT_DIR}/schedules_clean.csv", index=False)
#log(f"  ✓ Saved → preprocessed/schedules_clean.csv")
log(f"  ✓ Saved → data/processed/schedules_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN ROUTES  (3 JSON files: EXP, PASS, SF)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("4. TRAIN ROUTES (EXP / PASS / SF JSON files)")
log("="*60)

def load_json_trains(path, train_category):
    with open(path) as f:
        data = json.load(f)

    records = []
    for train in data:
        train_num  = str(train.get("trainNumber", "")).strip().zfill(5)
        train_name = str(train.get("trainName", "")).strip().title()
        route      = str(train.get("route", "")).strip()
        running    = running_days_str(train.get("runningDays", {}))

        for stop in train.get("trainRoute", []):
            station_raw = str(stop.get("stationName", ""))
            # Extract code from "STATION NAME - CODE" pattern
            if " - " in station_raw:
                parts = station_raw.rsplit(" - ", 1)
                s_name = parts[0].strip().title()
                s_code = parts[1].strip().upper()
            else:
                s_name = station_raw.strip().title()
                s_code = np.nan

            arrives_raw = stop.get("arrives", "")
            departs_raw = stop.get("departs", "")

            # Determine stop type
            if str(arrives_raw).strip().lower() == "source":
                stop_type = "Source"
            elif str(departs_raw).strip().lower() == "destination":
                stop_type = "Destination"
            else:
                stop_type = "Intermediate"

            records.append({
                "train_number"     : train_num,
                "train_name"       : train_name,
                "train_category"   : train_category,
                "route"            : route,
                "running_days"     : running,
                "stop_no"          : int(stop.get("sno", 0)),
                "station_code"     : s_code,
                "station_name"     : s_name,
                "arrives"          : arrives_raw if stop_type != "Source"      else "Source",
                "departs"          : departs_raw if stop_type != "Destination" else "Destination",
                "arrives_min"      : parse_time_to_minutes(arrives_raw),
                "departs_min"      : parse_time_to_minutes(departs_raw),
                "distance_km"      : parse_distance_km(stop.get("distance", "")),
                "day"              : int(stop.get("day", 1)),
                "stop_type"        : stop_type,
            })

    return pd.DataFrame(records)


exp_df  = load_json_trains(EXP_JSON,  "Express")
pass_df = load_json_trains(PASS_JSON, "Passenger")
sf_df   = load_json_trains(SF_JSON,   "Superfast")

routes = pd.concat([exp_df, pass_df, sf_df], ignore_index=True)
log(f"  Express trains : {exp_df['train_number'].nunique():>5} trains, {len(exp_df):>7} stops")
log(f"  Passenger trains: {pass_df['train_number'].nunique():>4} trains, {len(pass_df):>7} stops")
log(f"  Superfast trains: {sf_df['train_number'].nunique():>4} trains, {len(sf_df):>7} stops")
log(f"  Combined total  : {routes['train_number'].nunique():>4} trains, {len(routes):>7} stops")

# --- Derived: halt duration (departs - arrives) ---
routes["halt_min"] = (routes["departs_min"] - routes["arrives_min"]).clip(lower=0)

# --- Null check ---
nulls = routes.isnull().sum()
log(f"  Nulls (NaN expected for Source/Dest times):\n{nulls[nulls>0].to_string()}")

routes.to_csv(f"{OUT_DIR}/train_routes_clean.csv", index=False)
#log(f"  ✓ Saved → preprocessed/train_routes_clean.csv")
log(f"  ✓ Saved → data/processed/train_routes_clean.csv")



#-----------------------------

delay = pd.read_csv("data/processed/merged_delay.csv")

#---------------------------------------



# ══════════════════════════════════════════════════════════════════════════════
# 5. MASTER MERGED TABLE
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("5. MASTER MERGED TABLE")
log("="*60)

# Build per-train summary from routes
route_summary = (
    routes.groupby("train_number")
    .agg(
        total_stops    = ("stop_no", "max"),
        total_distance = ("distance_km", "max"),
        running_days   = ("running_days", "first"),
        train_category = ("train_category", "first"),
    )
    .reset_index()
)

# Merge: train_list + route_summary
master = train_list.merge(route_summary, left_on="train_number_str", right_on="train_number", how="left")

# Fix column naming after merge
if "train_number_x" in master.columns:
    master = master.rename(columns={"train_number_x": "train_number"})
    if "train_number_y" in master.columns:
        master = master.drop(columns=["train_number_y"])

# Merge with schedules (drop duplicates per train)
sched_mini = sched_clean[["train_number_str","num_stations","timestamp"]].drop_duplicates(subset="train_number_str")
master = master.merge(sched_mini, on="train_number_str", how="left")

# Merge station-level delay stats onto from_station
master = master.merge(
    delay[["station_code","avg_delay_min","delay_category"]].rename(
        columns={"station_code":"from_station","avg_delay_min":"origin_avg_delay_min","delay_category":"origin_delay_category"}),
    on="from_station", how="left"
)

log(f"  Master table shape: {master.shape}")
log(f"  Columns: {list(master.columns)}")
nulls = master.isnull().sum()
log(f"  Nulls:\n{nulls[nulls>0].to_string() if nulls.sum() > 0 else '  None'}")

master.to_csv(f"{OUT_DIR}/master_merged.csv", index=False)
#log(f"  ✓ Saved → preprocessed/master_merged.csv")
log(f"  ✓ Saved → data/processed/master_merged.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PREPROCESSING REPORT
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("SUMMARY")
log("="*60)
log(f"  merged_delay.csv      → {delay.shape[0]} rows × {delay.shape[1]} cols")
log(f"  train_list_clean.csv → {train_list.shape[0]} rows × {train_list.shape[1]} cols")
log(f"  schedules_clean.csv  → {sched_clean.shape[0]} rows × {sched_clean.shape[1]} cols")
log(f"  train_routes_clean.csv → {routes.shape[0]} rows × {routes.shape[1]} cols")
log(f"  master_merged.csv    → {master.shape[0]} rows × {master.shape[1]} cols")

#with open(f"{OUT_DIR}/preprocessing_report.txt", "w") as f:
with open(f"{OUT_DIR}/preprocessing_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n✅ All preprocessing complete. Files saved to:", OUT_DIR)