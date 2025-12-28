import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates

# === CONFIGURATION ===
SITES = ['Queens']
YEARS = [2018]
FREQ = 28.0
BASE_DIR = "/Volumes/500GB_Temp_Storage/NYSM_Data"
OUTPUT_DIR = f"{SITES[0]}_{YEARS[0]}_{FREQ}"
FIGSIZE = (14, 4)

'''
BASE_DIR = "/Volumes/500GB_Temp_Storage/NYSM_Data"
'''

RFI_THRESHOLD = 350
VAPOR_ZERO_DURATION_MAX = 2
VAPOR_NEARBY_NONZERO_WINDOW = 5
HIGH_VAPOR_THRESHOLD = 9.0
HIGH_LIQUID_THRESHOLD = 10.0
LIQUID_NEARBY_NONZERO_WINDOW = 5

YLIM_RANGES = {
    "Voltage": (None, None),
    "Brightness Temperature": (None, None),
    "Integrated Vapor": (None, None),
    "Integrated Liquid": (None, None),
    "Cloud Base": (None, None),
    "Data Quality": (None, None)
}

PLOT_CONFIGS = {
    "Voltage": ("lv0", FREQ, "voltages", "Voltage (V)"),
    "Brightness Temperature": ("lv1", FREQ, "brightness", "Brightness (K)"),
    "Integrated Vapor": ("lv2", "Int. Vapor(cm)", "values", "Vapor (cm)"),
    "Integrated Liquid": ("lv2", "Int. Liquid(mm)", "values", "Liquid (mm)"),
    "Cloud Base": ("lv2", "Cloud Base(km)", "values", "Cloud Base (km)"),
    "Data Quality": ("lv2", "DataQuality", "values", "Data Quality")
}

def load_data(site, year, level, key, subkey):
    prefix = f"{key}_{subkey}" if level != "lv2" else key
    dir_path = os.path.join(BASE_DIR, str(year), site, level)
    dates_path = os.path.join(dir_path, f"{prefix}_dates.npy")
    values_path = os.path.join(dir_path, f"{prefix}_values.npy")
    if not os.path.exists(dates_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"Missing {dates_path} or {values_path}")
    dates = np.load(dates_path, allow_pickle=True)
    values = np.load(values_path, allow_pickle=True)
    return pd.to_datetime(dates), values

def get_segments(df, condition_col):
    df["group"] = (df[condition_col] != df[condition_col].shift()).cumsum()
    return [(g["t"].iloc[0], g["t"].iloc[-1]) for _, g in df[df[condition_col]].groupby("group")]

def get_rfi_segments(time, values):
    df = pd.DataFrame({"t": time, "v": values}).dropna()
    df["rfi"] = df["v"] > RFI_THRESHOLD
    segments = get_segments(df, "rfi")
    return segments, df[df["rfi"]]["t"].dt.date.value_counts().to_dict()

def get_vapor_drop_segments(time, values):
    df = pd.DataFrame({"t": time, "v": values}).dropna()
    df["is_zero"] = np.isclose(df["v"], 0, atol=1e-4)
    df["group"] = (df["is_zero"] != df["is_zero"].shift()).cumsum()
    segments = []
    dates = []
    for _, g in df[df["is_zero"]].groupby("group"):
        start = g["t"].iloc[0]
        end = g["t"].iloc[-1]
        duration = (end - start)
        if duration <= pd.Timedelta(minutes=VAPOR_ZERO_DURATION_MAX):
            window_start = start - pd.Timedelta(minutes=VAPOR_NEARBY_NONZERO_WINDOW)
            window_end = end + pd.Timedelta(minutes=VAPOR_NEARBY_NONZERO_WINDOW)
            in_window = df[(df["t"] >= window_start) & (df["t"] <= window_end)]
            in_window_outside = in_window[(in_window["t"] < start) | (in_window["t"] > end)]
            if not (in_window_outside["is_zero"]).any() and (in_window_outside["v"] >= 0.1).all():
                seg = (start - pd.Timedelta(minutes=1), end + pd.Timedelta(minutes=1))
                segments.append(seg)
                dates.append(seg[0].date())
    return segments, pd.Series(dates).value_counts().to_dict()

def get_high_vapor_segments(time, values):
    df = pd.DataFrame({"t": time, "v": values}).dropna()
    df["high"] = df["v"] > HIGH_VAPOR_THRESHOLD
    segments = get_segments(df, "high")
    return segments, df[df["high"]]["t"].dt.date.value_counts().to_dict()

def get_high_liquid_segments(time, values):
    df = pd.DataFrame({"t": time, "v": values}).dropna()
    df["high"] = df["v"] > HIGH_LIQUID_THRESHOLD
    df["group"] = (df["high"] != df["high"].shift()).cumsum()
    segments = []
    dates = []
    for _, g in df[df["high"]].groupby("group"):
        start = g["t"].iloc[0]
        end = g["t"].iloc[-1]
        window_start = start - pd.Timedelta(minutes=LIQUID_NEARBY_NONZERO_WINDOW)
        window_end = end + pd.Timedelta(minutes=LIQUID_NEARBY_NONZERO_WINDOW)
        in_window = df[(df["t"] >= window_start) & (df["t"] <= window_end)]
        in_window_outside = in_window[(in_window["t"] < start) | (in_window["t"] > end)]
        if (in_window_outside["v"] > 0.1).any():
            seg = (start - pd.Timedelta(minutes=1), end + pd.Timedelta(minutes=1))
            segments.append(seg)
            dates.append(seg[0].date())
    return segments, pd.Series(dates).value_counts().to_dict()

def get_negative_bt_segments(time, values):
    df = pd.DataFrame({"t": time, "v": values}).dropna()
    df["neg"] = df["v"] < 0
    segments = get_segments(df, "neg")
    return segments, df[df["neg"]]["t"].dt.date.value_counts().to_dict()

def plot_with_segments(time, values, rfi_segments, vapor_segments, high_segments, blue_segments, purple_segments, ylabel, ylim, save_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(time, values, color="black", linewidth=0.6, label="Data", zorder=1)

    for segs, color, label, z in [
        (purple_segments, "purple", "Liquid > 10 mm", 1),
        (vapor_segments, "orange", "Vapor Drop", 2),
        (high_segments, "green", "Vapor > 9 cm", 3),
        (rfi_segments, "red", "Brightness > 350K", 4),
        (blue_segments, "deepskyblue", "Brightness < 0", 6)
    ]:
        first = True
        for start, end in segs:
            if start == end:
                start -= pd.Timedelta(minutes=7.5)
                end   += pd.Timedelta(minutes=7.5)
            mask = (time >= start) & (time <= end)
            if mask.any():
                ax.plot(time[mask], values[mask], color=color, linewidth=2.0,
                        label=label if first else "", zorder=z)
                first = False

    if ylim is not None and any(y is not None for y in ylim):
        ax.set_ylim(*ylim)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.set_xlabel("Time", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === MAIN LOOP with sub-year ranges ===

print(YEARS)
for year in tqdm(YEARS, desc="Processing years"):
    SUBYEAR_RANGES = [("Jan-Dec", pd.to_datetime(f"{year}-01-01"), pd.to_datetime(f"{year}-12-31 23:59:59"))] 

    '''
    [
        ("Jan-Feb", pd.to_datetime(f"{year}-01-01"), pd.to_datetime(f"{year}-02-28 23:59:59")),
        ("Mar-Apr", pd.to_datetime(f"{year}-03-01"), pd.to_datetime(f"{year}-04-30 23:59:59")),
        ("May-Jun", pd.to_datetime(f"{year}-05-01"), pd.to_datetime(f"{year}-06-30 23:59:59")),
        ("Jul-Aug", pd.to_datetime(f"{year}-07-01"), pd.to_datetime(f"{year}-08-31 23:59:59")),
        ("Sep-Oct", pd.to_datetime(f"{year}-09-01"), pd.to_datetime(f"{year}-10-31 23:59:59")),
        ("Jan-Dec", pd.to_datetime(f"{year}-01-01"), pd.to_datetime(f"{year}-12-31 23:59:59"))
    ]
    '''

    for site in tqdm(SITES, leave=False, desc=f"Sites for {year}"):

        try:
            # Load raw data once per site/year
            bt_time, bt_values = load_data(site, year, "lv1", FREQ, "brightness")
            vap_time, vap_values = load_data(site, year, "lv2", "Int. Vapor(cm)", "values")
            liq_time, liq_values = load_data(site, year, "lv2", "Int. Liquid(mm)", "values")

            # Detect segments once per site/year
            rfi_segments, rfi_daily = get_rfi_segments(bt_time, bt_values)
            vapor_segments, vapor_daily = get_vapor_drop_segments(vap_time, vap_values)
            high_vapor_segments, high_vapor_daily = get_high_vapor_segments(vap_time, vap_values)
            blue_segments, blue_daily = get_negative_bt_segments(bt_time, bt_values)
            high_liquid_segments, high_liquid_daily = get_high_liquid_segments(liq_time, liq_values)
            
            for subrange_name, ZOOM_START, ZOOM_END in SUBYEAR_RANGES:
                save_dir = os.path.join(OUTPUT_DIR, str(year), site, subrange_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # Prepare daily summary CSV
                all_days = pd.date_range(ZOOM_START, ZOOM_END, freq='D').date
                daily_df = pd.DataFrame({'day': all_days})
                for name, daily in [
                    ('brightness_gt_350K', rfi_daily),
                    ('brightness_lt_0', blue_daily),
                    ('vapor_eq_0', vapor_daily),
                    ('vapor_gt_9cm', high_vapor_daily),
                    ('liquid_gt_10mm', high_liquid_daily)
                ]:
                    daily_df[name] = daily_df['day'].map(daily).fillna(0).astype(int)
                daily_df.to_csv(os.path.join(save_dir, f"{site}_{year}_{subrange_name}_daily_summary.csv"), index=False)
                
                # Plot all measurements for this subrange
                for title, (level, key, subkey, ylabel) in PLOT_CONFIGS.items():
                    print('\nMarker 4')
                    try:
                        print(site)
                        print(year)
                        print(level)
                        print(key)
                        print(subkey)
                        t, v = load_data(site, year, level, key, subkey)
                        t = pd.Series(t)
                        v = pd.Series(v)
                        mask_zoom = (t >= ZOOM_START) & (t <= ZOOM_END)
                        t_zoomed = t[mask_zoom].reset_index(drop=True)
                        v_zoomed = v[mask_zoom].reset_index(drop=True)

                        ylim = YLIM_RANGES.get(title, (None, None))
                        save_path = os.path.join(save_dir, f"{site}_{year}_{subrange_name}_{title.replace(' ', '_')}.png")

                        print(save_path)
                        
                        plot_with_segments(
                            t_zoomed, v_zoomed, rfi_segments, vapor_segments,
                            high_vapor_segments, blue_segments, high_liquid_segments,
                            ylabel, ylim, save_path
                        )
                    except Exception as e:
                        print(f"[{site} {year} {subrange_name}] Error plotting {title}: {e}")
        except Exception as e:
            print(f"[{site} {year}] Skipped due to error: {e}")

