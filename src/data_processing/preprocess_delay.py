import pandas as pd
import os
import numpy as np

folder_path = "data/raw/delay/"
output_path = "data/processed/merged_delay.csv"

all_files = os.listdir(folder_path)

df_list = []

print("🔄 Loading all delay files...")

for file in all_files:
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

delay = pd.concat(df_list, ignore_index=True)

print(f"✅ Total rows after merge: {delay.shape}")

delay.columns = [
    "station_code",
    "station_name",
    "avg_delay_min",
    "pct_right_time",
    "pct_slight_delay",
    "pct_significant_delay",
    "pct_cancelled_unknown",
]

delay["station_code"] = delay["station_code"].astype(str).str.strip().str.upper()
delay["station_name"] = delay["station_name"].astype(str).str.strip().str.title()

delay.drop_duplicates(inplace=True)

def delay_category(x):
    if x <= 15:
        return "On Time"
    elif x <= 60:
        return "Slight Delay"
    else:
        return "High Delay"

delay["delay_category"] = delay["avg_delay_min"].apply(delay_category)

delay.to_csv(output_path, index=False)

print("✅ Delay data merged + cleaned successfully!")