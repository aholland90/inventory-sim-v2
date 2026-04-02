"""
Min/Max Inventory Simulator
============================
A Streamlit app that simulates inventory performance using historical
consumption data and recommends optimized Min/Max settings.

Modes:
  - Static       : Use existing Min/Max from Min_Max_Store_SKU
  - DOH          : Days-on-Hand formula  (ADD × Min/Max DOH)
  - Service Level: Statistical safety stock + lead-time demand
"""

import io
import warnings
from math import ceil, sqrt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Min/Max Inventory Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────

def load_excel(file) -> dict:
    """Load every sheet from the uploaded Excel file into a dict of DataFrames."""
    xl = pd.ExcelFile(file)
    return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}


REQUIRED_SHEETS = {
    "Consumed": ["ConDate", "StoreNum", "PartNum", "Consumption"],
    "Part_List": ["PartNum", "PartDesc"],
    "Store_List": ["StoreNum", "StoreDesc"],
    "Store_Order_Schedule": ["StoreNum", "DayOfWeek", "OrderYes"],
    "Info_Input": [],
}

def validate_sheets(sheets: dict) -> tuple:
    """Return (is_valid, error_list)."""
    errors = []
    for sheet, cols in REQUIRED_SHEETS.items():
        if sheet not in sheets:
            errors.append(f"Missing required sheet: **{sheet}**")
            continue
        for col in cols:
            if col not in sheets[sheet].columns:
                errors.append(f"Sheet '{sheet}' is missing column: **{col}**")
    return len(errors) == 0, errors


def parse_info_input(df: pd.DataFrame) -> dict:
    """
    Parse the Info_Input sheet.
    The sheet has documentation rows at the top; the actual settings
    begin at the row where the first cell equals 'Setting'.
    """
    params = {}
    header_idx = None
    for i, row in df.iterrows():
        if str(row.iloc[0]).strip() == "Setting":
            header_idx = i
            break
    if header_idx is None:
        return params
    data = df.iloc[header_idx + 1 :].reset_index(drop=True)
    for _, row in data.iterrows():
        setting = str(row.iloc[0]).strip()
        value = row.iloc[1]
        if setting and setting.lower() != "nan":
            params[setting] = value
    return params


def safe_float(params: dict, key: str, default: float) -> float:
    """Safely coerce a param value to float, returning default on failure."""
    val = params.get(key, default)
    try:
        s = str(val).strip().lower()
        if s in ("nan", "", "none", "calculated from store order schedule"):
            return default
        return float(s)
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────
# SECTION 2 — DEMAND CALCULATION
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def calculate_demand(
    consumed_json: str,
    data_window: int,
    variability_window: int,
) -> pd.DataFrame:
    """
    For every (StoreNum, PartNum) pair compute:
      - AvgDailyDemand  : mean daily consumption over last `data_window` days
      - DemandStdDev    : std dev over last `variability_window` days
    Missing dates are treated as 0 consumption.
    Accepts serialised JSON to support st.cache_data.
    """
    consumed_df = pd.read_json(io.StringIO(consumed_json))
    consumed_df["ConDate"] = pd.to_datetime(consumed_df["ConDate"], unit="ms")

    max_date = consumed_df["ConDate"].max()
    window_start = max_date - pd.Timedelta(days=data_window - 1)
    var_start = max_date - pd.Timedelta(days=variability_window - 1)
    date_range = pd.date_range(start=window_start, end=max_date, freq="D")

    combos = consumed_df[["StoreNum", "PartNum"]].drop_duplicates()
    records = []

    for _, combo in combos.iterrows():
        store, part = combo["StoreNum"], combo["PartNum"]
        mask = (consumed_df["StoreNum"] == store) & (consumed_df["PartNum"] == part)
        hist = (
            consumed_df[mask]
            .set_index("ConDate")["Consumption"]
            .reindex(date_range, fill_value=0)
        )
        add = hist.mean()
        var_hist = hist[hist.index >= var_start]
        std_dev = float(var_hist.std()) if len(var_hist) > 1 else 0.0
        if np.isnan(std_dev):
            std_dev = 0.0
        records.append(
            {
                "StoreNum": store,
                "PartNum": part,
                "AvgDailyDemand": add,
                "DemandStdDev": std_dev,
                "DataDays": len(hist),
            }
        )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# SECTION 3 — Z-SCORE HELPER
# ─────────────────────────────────────────────────────────────

_Z_TABLE = {90.0: 1.28, 95.0: 1.65, 97.5: 1.96, 99.0: 2.33}

def get_z_score(service_level_pct: float) -> float:
    if service_level_pct in _Z_TABLE:
        return _Z_TABLE[service_level_pct]
    return float(norm.ppf(service_level_pct / 100.0))


# ─────────────────────────────────────────────────────────────
# SECTION 4 — MIN/MAX CALCULATION
# ─────────────────────────────────────────────────────────────

def calculate_min_max(
    demand_df: pd.DataFrame,
    mode: str,
    params: dict,
    min_max_df: pd.DataFrame | None,
    part_list_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with RecommendedMin, RecommendedMax, and supporting
    columns for every (StoreNum, PartNum).

    Parameters
    ----------
    demand_df   : output of calculate_demand()
    mode        : 'Static' | 'DOH' | 'Service Level'
    params      : parsed Info_Input dict (possibly overridden by UI)
    min_max_df  : Min_Max_Store_SKU sheet (optional)
    part_list_df: Part_List sheet (for battery detection)
    """
    lead_time    = safe_float(params, "Lead Time (Days)", 2.0)
    cycle_days   = safe_float(params, "Cycle Days (Order Frequency Target)", 4.0)
    min_doh      = safe_float(params, "Min DOH (Days on Hand)", 14.0)
    max_doh      = safe_float(params, "Max DOH (Days on Hand)", 28.0)
    service_lvl  = safe_float(params, "Service Level Target (%)", 95.0)
    safety_buf   = safe_float(params, "Safety Buffer (Days of Demand)", 1.0)
    prot_reg     = safe_float(params, "Protection Days - Regular SKU", 4.0)
    prot_bat     = safe_float(params, "Protection Days - Battery SKU", 6.0)
    order_mult_v = params.get("Order Multiple", None)
    try:
        order_multiple = int(order_mult_v) if order_mult_v and float(order_mult_v) > 0 else None
    except (TypeError, ValueError):
        order_multiple = None

    z = get_z_score(service_lvl)
    result = demand_df.copy()

    # Battery detection
    if part_list_df is not None:
        pl = part_list_df.drop_duplicates("PartNum")[["PartNum", "PartDesc"]]
        result = result.merge(pl, on="PartNum", how="left")
        result["IsBattery"] = result["PartDesc"].str.lower().str.contains("battery", na=False)
    else:
        result["IsBattery"] = False
        result["PartDesc"] = ""

    result["ProtectionDays"]  = result["IsBattery"].map({True: prot_bat, False: prot_reg})
    result["SafetyBuffer"]    = safety_buf
    result["LeadTimeDays"]    = lead_time
    result["CycleDays"]       = cycle_days
    result["ServiceLevelTarget"] = service_lvl

    # ── Static ──────────────────────────────────────────────
    if mode == "Static":
        result["SafetyStock"] = 0.0
        result["CalculationMode"] = "Static"
        if min_max_df is not None:
            result = result.merge(
                min_max_df[["StoreNum", "PartNum", "Min", "Max"]],
                on=["StoreNum", "PartNum"],
                how="left",
            )
            result["RecommendedMin"] = result["Min"].fillna(0).round().astype(int)
            result["RecommendedMax"] = result["Max"].fillna(0).round().astype(int)
            result.rename(columns={"Min": "CurrentMin", "Max": "CurrentMax"}, inplace=True)
        else:
            result["CurrentMin"]    = np.nan
            result["CurrentMax"]    = np.nan
            result["RecommendedMin"] = 0
            result["RecommendedMax"] = 0

    # ── DOH ─────────────────────────────────────────────────
    elif mode == "DOH":
        result["SafetyStock"]    = 0.0
        result["CalculationMode"] = "DOH"
        result["RecommendedMin"] = (result["AvgDailyDemand"] * min_doh).round().astype(int)
        result["RecommendedMax"] = (result["AvgDailyDemand"] * max_doh).round().astype(int)
        result["RecommendedMax"] = result[["RecommendedMin", "RecommendedMax"]].max(axis=1)

        if order_multiple:
            result["RecommendedMax"] = (
                np.ceil(result["RecommendedMax"] / order_multiple) * order_multiple
            ).astype(int)

        if min_max_df is not None:
            result = result.merge(
                min_max_df[["StoreNum", "PartNum", "Min", "Max"]],
                on=["StoreNum", "PartNum"],
                how="left",
            )
            result.rename(columns={"Min": "CurrentMin", "Max": "CurrentMax"}, inplace=True)
        else:
            result["CurrentMin"] = np.nan
            result["CurrentMax"] = np.nan

    # ── Service Level ────────────────────────────────────────
    elif mode == "Service Level":
        result["SafetyStock"] = (
            z * result["DemandStdDev"] * np.sqrt(lead_time)
        ).round()
        result["CalculationMode"] = "Service Level"
        result["RecommendedMin"]  = (
            result["AvgDailyDemand"] * lead_time + result["SafetyStock"]
        ).round().astype(int)
        result["RecommendedMax"]  = (
            result["RecommendedMin"] + result["AvgDailyDemand"] * cycle_days
        ).round().astype(int)
        result["RecommendedMax"]  = result[["RecommendedMin", "RecommendedMax"]].max(axis=1)

        if order_multiple:
            result["RecommendedMax"] = (
                np.ceil(result["RecommendedMax"] / order_multiple) * order_multiple
            ).astype(int)

        if min_max_df is not None:
            result = result.merge(
                min_max_df[["StoreNum", "PartNum", "Min", "Max"]],
                on=["StoreNum", "PartNum"],
                how="left",
            )
            result.rename(columns={"Min": "CurrentMin", "Max": "CurrentMax"}, inplace=True)
        else:
            result["CurrentMin"] = np.nan
            result["CurrentMax"] = np.nan

    # Ensure non-negative
    result["RecommendedMin"] = result["RecommendedMin"].clip(lower=0)
    result["RecommendedMax"] = result["RecommendedMax"].clip(lower=0)
    return result


# ─────────────────────────────────────────────────────────────
# SECTION 5 — SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_simulation(
    consumed_json: str,
    min_max_json: str,
    order_schedule_json: str,
    mode: str,
    lead_time: int,
    order_multiple: int | None,
    sim_start_str: str,
    sim_end_str: str,
) -> pd.DataFrame:
    """
    Core daily inventory simulation engine.

    For every (StoreNum, PartNum) and every date in the simulation window:
      1. Receive any orders due today
      2. Subtract consumption  → track MissedRepairs
      3. If InventoryOH <= Min AND store can order → place order (arrives in lead_time days)

    Returns Sim_Data DataFrame.
    """
    consumed_df      = pd.read_json(io.StringIO(consumed_json))
    min_max_calc     = pd.read_json(io.StringIO(min_max_json))
    order_schedule   = pd.read_json(io.StringIO(order_schedule_json))

    consumed_df["ConDate"] = pd.to_datetime(consumed_df["ConDate"], unit="ms")
    for df in [min_max_calc, order_schedule]:
        for col in df.select_dtypes("datetime64").columns:
            df[col] = pd.to_datetime(df[col])

    sim_start = pd.Timestamp(sim_start_str)
    sim_end   = pd.Timestamp(sim_end_str)
    all_dates = pd.date_range(start=sim_start, end=sim_end, freq="D")

    # Order-allowed lookup: {(StoreNum, DayOfWeek): bool}
    order_ok = {
        (int(r.StoreNum), int(r.DayOfWeek)): bool(r.OrderYes)
        for _, r in order_schedule.iterrows()
    }

    # Consumption lookup: {(StoreNum, PartNum, date): qty}
    consumed_df["_key"] = list(
        zip(consumed_df["StoreNum"], consumed_df["PartNum"], consumed_df["ConDate"])
    )
    consumption_lkp = consumed_df.set_index("_key")["Consumption"].to_dict()

    # Min/Max lookup: {(StoreNum, PartNum): (min_val, max_val)}
    # For Static mode we use CurrentMin/CurrentMax, otherwise Recommended
    if mode == "Static":
        min_col, max_col = "CurrentMin", "CurrentMax"
        # Fall back to Recommended if Current is missing
    else:
        min_col, max_col = "RecommendedMin", "RecommendedMax"

    mm_lkp = {}
    for _, r in min_max_calc.iterrows():
        k = (int(r.StoreNum), int(r.PartNum))
        if mode == "Static":
            try:
                mn = 0 if pd.isna(r.get("CurrentMin")) else int(r["CurrentMin"])
                mx = 0 if pd.isna(r.get("CurrentMax")) else int(r["CurrentMax"])
            except Exception:
                mn, mx = int(r.get("RecommendedMin", 0)), int(r.get("RecommendedMax", 0))
        else:
            mn = int(r.get("RecommendedMin", 0))
            mx = int(r.get("RecommendedMax", 0))
        mm_lkp[k] = (mn, mx)

    combos = list(mm_lkp.keys())
    records = []

    for store, part in combos:
        mn, mx = mm_lkp[(store, part)]
        inv_oh = mx  # start at Max
        pending: dict = {}  # {receipt_date: qty}

        for date in all_dates:
            dow = date.isoweekday()  # 1=Mon … 7=Sun

            # Receive
            received = pending.pop(date, 0)
            inv_oh  += received

            # Consume
            consumption = consumption_lkp.get((store, part, date), 0)
            if inv_oh >= consumption:
                missed = 0
                inv_oh -= consumption
            else:
                missed  = consumption - inv_oh
                inv_oh  = 0

            # Order
            ordered   = 0
            can_order = order_ok.get((store, dow), False)
            if inv_oh <= mn and can_order and mx > inv_oh:
                qty = mx - inv_oh
                if order_multiple and order_multiple > 0:
                    qty = int(ceil(qty / order_multiple) * order_multiple)
                ordered = max(0, qty)
                receipt_date = date + pd.Timedelta(days=lead_time)
                pending[receipt_date] = pending.get(receipt_date, 0) + ordered

            records.append(
                {
                    "ConDate":      date,
                    "StoreNum":     store,
                    "PartNum":      part,
                    "Consumption":  consumption,
                    "InventoryOH":  inv_oh,
                    "Ordered":      ordered,
                    "Received":     received,
                    "MissedRepairs": missed,
                }
            )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# SECTION 6 — RECOMMENDATION GENERATION
# ─────────────────────────────────────────────────────────────

def generate_recommendations(
    sim_data: pd.DataFrame,
    min_max_calc: pd.DataFrame,
    store_list: pd.DataFrame,
    part_list: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Build the Recommended_Min_Max output table."""
    service_lvl_tgt = safe_float(params, "Service Level Target (%)", 95.0) / 100.0
    total_sim_days  = sim_data["ConDate"].nunique() or 1

    # Aggregate simulation results
    agg = (
        sim_data.groupby(["StoreNum", "PartNum"])
        .agg(
            TotalDemand       =("Consumption",   "sum"),
            TotalMissedRepairs=("MissedRepairs",  "sum"),
            AvgInventory      =("InventoryOH",    "mean"),
            TotalOrders       =("Ordered",        lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    agg["FillRate"] = np.where(
        agg["TotalDemand"] > 0,
        (agg["TotalDemand"] - agg["TotalMissedRepairs"]) / agg["TotalDemand"],
        1.0,
    ).clip(0, 1)

    result = min_max_calc.merge(agg, on=["StoreNum", "PartNum"], how="left")

    # Descriptions
    result = result.merge(
        store_list.drop_duplicates("StoreNum")[["StoreNum", "StoreDesc"]],
        on="StoreNum", how="left",
    )
    if "PartDesc" not in result.columns:
        result = result.merge(
            part_list.drop_duplicates("PartNum")[["PartNum", "PartDesc"]],
            on="PartNum", how="left",
        )

    # Recommendation reasons
    def reason(row) -> str:
        add       = float(row.get("AvgDailyDemand", 0) or 0)
        demand    = float(row.get("TotalDemand",    0) or 0)
        fill      = float(row.get("FillRate",       1) or 1)
        avg_inv   = float(row.get("AvgInventory",   0) or 0)
        n_orders  = float(row.get("TotalOrders",    0) or 0)
        has_cur   = not pd.isna(row.get("CurrentMin", np.nan))

        if demand == 0 or add < 0.01:
            return "Low demand item; review before applying recommendation"

        msgs = []
        if not has_cur:
            msgs.append("No current Min/Max found; recommendation based on demand history")
        if fill < service_lvl_tgt:
            msgs.append("Increase Min to reduce missed repairs")
        if add > 0 and avg_inv > add * 30:
            msgs.append("Lower Max to reduce excess inventory")
        if total_sim_days > 0 and n_orders > total_sim_days / 3:
            msgs.append("Increase Max to reduce order frequency")
        return "; ".join(msgs) if msgs else "Current settings appear sufficient"

    result["RecommendationReason"] = result.apply(reason, axis=1)

    col_order = [
        "StoreNum", "PartNum", "PartDesc", "StoreDesc",
        "CurrentMin", "CurrentMax", "RecommendedMin", "RecommendedMax",
        "CalculationMode", "AvgDailyDemand", "DemandStdDev", "SafetyStock",
        "LeadTimeDays", "CycleDays", "ServiceLevelTarget",
        "TotalDemand", "TotalMissedRepairs", "FillRate",
        "AvgInventory", "TotalOrders", "RecommendationReason",
    ]
    return result[[c for c in col_order if c in result.columns]]


# ─────────────────────────────────────────────────────────────
# SECTION 7 — SUMMARY METRICS
# ─────────────────────────────────────────────────────────────

def compute_summary(
    sim_data: pd.DataFrame,
    recommendations: pd.DataFrame,
    params: dict,
) -> dict:
    total_demand = sim_data["Consumption"].sum()
    total_missed = sim_data["MissedRepairs"].sum()
    fill_rate    = (
        (total_demand - total_missed) / total_demand if total_demand > 0 else 1.0
    )
    total_orders = (sim_data["Ordered"] > 0).sum()
    avg_inventory = sim_data["InventoryOH"].mean()

    low_velocity = changed = 0
    if recommendations is not None and len(recommendations):
        low_velocity = recommendations["RecommendationReason"].str.contains(
            "Low demand", na=False
        ).sum()
        has_cur = recommendations["CurrentMin"].notna()
        if has_cur.any():
            sub = recommendations[has_cur].copy()
            sub["_cur_min"] = pd.to_numeric(sub["CurrentMin"], errors="coerce").fillna(0).astype(int)
            sub["_cur_max"] = pd.to_numeric(sub["CurrentMax"], errors="coerce").fillna(0).astype(int)
            changed = int(
                ((sub["RecommendedMin"] != sub["_cur_min"]) |
                 (sub["RecommendedMax"] != sub["_cur_max"])).sum()
            )

    return {
        "total_demand":     int(total_demand),
        "total_missed":     int(total_missed),
        "fill_rate":        float(fill_rate),
        "total_orders":     int(total_orders),
        "avg_inventory":    float(avg_inventory),
        "low_velocity":     int(low_velocity),
        "changed_count":    int(changed),
    }


# ─────────────────────────────────────────────────────────────
# SECTION 8 — STREAMLIT UI
# ─────────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main():
    # ── Header ──────────────────────────────────────────────
    st.title("📦 Min/Max Inventory Simulator")
    st.caption(
        "Simulate inventory performance against historical demand and generate "
        "optimized Min/Max recommendations.  "
        "*All values are simulated recommendations — not final production settings.*"
    )

    # ── Sidebar — File Upload ────────────────────────────────
    st.sidebar.header("⚙️  Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Inventory Excel File",
        type=["xlsx", "xls"],
        help="Requires: Consumed, Part_List, Store_List, Store_Order_Schedule, Info_Input",
    )

    if uploaded_file is None:
        st.info("👈  Upload your Excel inventory file in the sidebar to get started.")
        with st.expander("Required sheet structure"):
            st.markdown(
                """
| Sheet | Key Columns |
|---|---|
| **Consumed** | ConDate, StoreNum, PartNum, Consumption |
| **Part_List** | PartNum, PartDesc |
| **Store_List** | StoreNum, StoreDesc |
| **Store_Order_Schedule** | StoreNum, DayOfWeek, OrderYes |
| **Info_Input** | Setting / Value rows |
| **Min_Max_Store_SKU** *(optional)* | StoreNum, PartNum, Min, Max |
"""
            )
        return

    # ── Load & Validate ──────────────────────────────────────
    with st.spinner("Loading workbook…"):
        sheets = load_excel(uploaded_file)

    valid, errors = validate_sheets(sheets)
    if not valid:
        for e in errors:
            st.error(e)
        return

    st.sidebar.success(f"✅  {len(sheets)} sheets loaded")

    # ── Parse Sheets ─────────────────────────────────────────
    consumed_df = sheets["Consumed"].copy()
    consumed_df["ConDate"]     = pd.to_datetime(consumed_df["ConDate"])
    consumed_df["Consumption"] = pd.to_numeric(consumed_df["Consumption"], errors="coerce").fillna(0)

    part_list_df  = sheets["Part_List"].drop_duplicates("PartNum")
    store_list_df = sheets["Store_List"].drop_duplicates("StoreNum")
    order_sched   = sheets["Store_Order_Schedule"]
    min_max_df    = sheets.get("Min_Max_Store_SKU")
    params        = parse_info_input(sheets.get("Info_Input", pd.DataFrame()))

    min_date = consumed_df["ConDate"].min().date()
    max_date = consumed_df["ConDate"].max().date()

    # ── Sidebar — Mode & Parameters ──────────────────────────
    st.sidebar.subheader("Simulation Mode")
    mode = st.sidebar.selectbox(
        "Mode",
        ["Static", "DOH", "Service Level"],
        help=(
            "**Static** — use existing Min/Max values as-is\n"
            "**DOH** — compute Min/Max from Days-on-Hand targets\n"
            "**Service Level** — statistical safety stock approach"
        ),
    )

    st.sidebar.subheader("Core Parameters")
    data_window = int(
        st.sidebar.number_input(
            "Data Window (Days)", 7, 365,
            value=int(safe_float(params, "Data Window (Days)", 30)),
        )
    )
    lead_time = int(
        st.sidebar.number_input(
            "Lead Time (Days)", 0, 30,
            value=int(safe_float(params, "Lead Time (Days)", 2)),
        )
    )
    cycle_days = int(
        st.sidebar.number_input(
            "Cycle Days (Order Frequency)", 1, 90,
            value=int(safe_float(params, "Cycle Days (Order Frequency Target)", 4)),
        )
    )

    if mode == "DOH":
        st.sidebar.subheader("DOH Parameters")
        min_doh = int(
            st.sidebar.number_input("Min DOH", 1, 180,
                value=int(safe_float(params, "Min DOH (Days on Hand)", 14)))
        )
        max_doh = int(
            st.sidebar.number_input("Max DOH", 1, 365,
                value=int(safe_float(params, "Max DOH (Days on Hand)", 28)))
        )
        params["Min DOH (Days on Hand)"] = min_doh
        params["Max DOH (Days on Hand)"] = max_doh

    if mode == "Service Level":
        st.sidebar.subheader("Service Level Parameters")
        service_lvl = st.sidebar.selectbox(
            "Service Level Target (%)", [90, 95, 97.5, 99], index=1
        )
        variability_window = int(
            st.sidebar.number_input(
                "Variability Window (Days)", 7, 365,
                value=int(safe_float(params, "Demand Variability Window (Days)", 30)),
            )
        )
        params["Service Level Target (%)"]        = service_lvl
        params["Demand Variability Window (Days)"] = variability_window
    else:
        variability_window = data_window
        if "Service Level Target (%)" not in params:
            params["Service Level Target (%)"] = 95.0

    st.sidebar.subheader("Protection & Safety")
    prot_regular = int(
        st.sidebar.number_input("Protection Days (Regular)", 0, 30,
            value=int(safe_float(params, "Protection Days - Regular SKU", 4)))
    )
    prot_battery = int(
        st.sidebar.number_input("Protection Days (Battery)", 0, 30,
            value=int(safe_float(params, "Protection Days - Battery SKU", 6)))
    )
    safety_buf = st.sidebar.number_input(
        "Safety Buffer (Days)", 0.0, 10.0, step=0.5,
        value=float(safe_float(params, "Safety Buffer (Days of Demand)", 1)),
    )
    order_mult_ui = st.sidebar.number_input(
        "Order Multiple (0 = off)", 0, 100,
        value=0,
    )

    # Propagate UI overrides to params
    params.update({
        "Data Window (Days)":               data_window,
        "Lead Time (Days)":                 lead_time,
        "Cycle Days (Order Frequency Target)": cycle_days,
        "Protection Days - Regular SKU":    prot_regular,
        "Protection Days - Battery SKU":    prot_battery,
        "Safety Buffer (Days of Demand)":   safety_buf,
        "Order Multiple":                   order_mult_ui if order_mult_ui > 0 else None,
    })
    order_multiple = int(order_mult_ui) if order_mult_ui > 0 else None

    # ── Sidebar — Date Range ─────────────────────────────────
    st.sidebar.subheader("Simulation Date Range")
    sim_start = st.sidebar.date_input("Start Date", value=min_date,
                                       min_value=min_date, max_value=max_date)
    sim_end   = st.sidebar.date_input("End Date",   value=max_date,
                                       min_value=min_date, max_value=max_date)

    # ── Run Button ───────────────────────────────────────────
    run_btn = st.sidebar.button("▶  Run Simulation", type="primary", use_container_width=True)

    # ── Pre-Run Overview ─────────────────────────────────────
    if not run_btn:
        st.subheader("📊 Data Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Stores",           consumed_df["StoreNum"].nunique())
        c2.metric("Unique SKUs",       consumed_df["PartNum"].nunique())
        c3.metric("Consumption Rows",  f"{len(consumed_df):,}")
        c4.metric("Earliest Date",     str(min_date))
        c5.metric("Latest Date",       str(max_date))

        if min_max_df is not None:
            st.metric("Active Min/Max Rules", len(min_max_df))

        st.subheader("📋 Parameters from Excel")
        if params:
            df_p = pd.DataFrame(
                [(k, v) for k, v in params.items() if v is not None],
                columns=["Setting", "Value"],
            )
            st.dataframe(df_p, use_container_width=True, hide_index=True)
        return

    # ════════════════════════════════════════════════════════
    # RUN PIPELINE
    # ════════════════════════════════════════════════════════
    with st.status("Running simulation pipeline…", expanded=True) as status:

        st.write("⚙️  Calculating demand metrics…")
        demand_df = calculate_demand(
            consumed_df.to_json(), data_window, variability_window
        )

        st.write("📐  Computing Min/Max values…")
        min_max_calc = calculate_min_max(
            demand_df, mode, params, min_max_df, part_list_df
        )

        st.write("🔄  Running day-by-day inventory simulation…")
        sim_data = run_simulation(
            consumed_df.to_json(),
            min_max_calc.to_json(),
            order_sched.to_json(),
            mode,
            lead_time,
            order_multiple,
            str(sim_start),
            str(sim_end),
        )

        st.write("🎯  Generating recommendations…")
        recommendations = generate_recommendations(
            sim_data, min_max_calc, store_list_df, part_list_df, params
        )

        summary = compute_summary(sim_data, recommendations, params)
        status.update(label="✅  Simulation complete!", state="complete")

    # ── Summary KPIs ─────────────────────────────────────────
    st.subheader("📈 Summary Metrics")
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Fill Rate",          f"{summary['fill_rate']:.1%}")
    k2.metric("Total Demand",       f"{summary['total_demand']:,}")
    k3.metric("Total Missed",       f"{summary['total_missed']:,}")
    k4.metric("Orders Placed",      f"{summary['total_orders']:,}")
    k5.metric("Avg Inventory",      f"{summary['avg_inventory']:.1f}")
    k6.metric("Recs w/ Changes",    f"{summary['changed_count']:,}")
    k7.metric("Low-Velocity Items", f"{summary['low_velocity']:,}")

    # ── Filters ──────────────────────────────────────────────
    st.subheader("🔍 Filters")
    fc1, fc2, fc3, fc4 = st.columns(4)

    store_names = store_list_df.set_index("StoreNum")["StoreDesc"].to_dict()
    stores_in   = sorted(sim_data["StoreNum"].unique())
    store_opts  = ["All"] + [f"{s} – {store_names.get(s, s)}" for s in stores_in]
    sel_store   = fc1.selectbox("Store", store_opts)

    part_names  = part_list_df.set_index("PartNum")["PartDesc"].to_dict()
    parts_in    = sorted(sim_data["PartNum"].unique())
    part_opts   = ["All"] + [f"{p} – {str(part_names.get(p, p))[:35]}" for p in parts_in]
    sel_part    = fc2.selectbox("Part", part_opts)

    only_changes  = fc3.checkbox("Recommendations w/ changes only")
    only_low_vel  = fc4.checkbox("Low-velocity items only")

    # Apply filters
    def filter_store_part(df):
        out = df.copy()
        if sel_store != "All":
            s = int(sel_store.split(" – ")[0])
            out = out[out["StoreNum"] == s]
        if sel_part != "All":
            p = int(sel_part.split(" – ")[0])
            out = out[out["PartNum"] == p]
        return out

    sim_filt = filter_store_part(sim_data)
    rec_filt = filter_store_part(recommendations)

    if only_changes and len(rec_filt):
        has_cur = rec_filt["CurrentMin"].notna()
        rec_filt = rec_filt[
            has_cur & (
                (rec_filt["RecommendedMin"] != pd.to_numeric(rec_filt["CurrentMin"], errors="coerce").fillna(-1).astype(int)) |
                (rec_filt["RecommendedMax"] != pd.to_numeric(rec_filt["CurrentMax"], errors="coerce").fillna(-1).astype(int))
            )
        ]

    if only_low_vel and len(rec_filt):
        rec_filt = rec_filt[rec_filt["RecommendationReason"].str.contains("Low demand", na=False)]

    # ════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════
    tab_charts, tab_sim, tab_rec, tab_mm = st.tabs(
        ["📊 Charts", "📋 Sim Data", "🎯 Recommendations", "🔢 Min/Max Calc"]
    )

    # ── Tab: Charts ──────────────────────────────────────────
    with tab_charts:

        # Inventory over time
        if len(sim_filt):
            inv_ts = sim_filt.groupby("ConDate")["InventoryOH"].mean().reset_index()
            fig = px.line(
                inv_ts, x="ConDate", y="InventoryOH",
                title="Average Inventory on Hand Over Time",
                labels={"ConDate": "Date", "InventoryOH": "Avg Inventory (units)"},
            )
            fig.update_traces(line_color="#1f77b4")
            st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)

        with col_l:
            # Missed repairs over time
            if len(sim_filt):
                miss_ts = sim_filt.groupby("ConDate")["MissedRepairs"].sum().reset_index()
                fig2 = px.bar(
                    miss_ts, x="ConDate", y="MissedRepairs",
                    title="Daily Missed Repairs",
                    labels={"ConDate": "Date", "MissedRepairs": "Missed Repairs"},
                    color_discrete_sequence=["#d62728"],
                )
                st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            # Fill rate by store
            fr_store = (
                sim_data.groupby("StoreNum")
                .agg(demand=("Consumption", "sum"), missed=("MissedRepairs", "sum"))
                .reset_index()
            )
            fr_store["FillRate"] = (
                (fr_store["demand"] - fr_store["missed"]) /
                fr_store["demand"].clip(lower=1)
            )
            fr_store["Label"] = fr_store["StoreNum"].map(store_names).fillna(
                fr_store["StoreNum"].astype(str)
            )
            fig3 = px.bar(
                fr_store, x="Label", y="FillRate",
                title="Fill Rate by Store",
                color="FillRate",
                color_continuous_scale="RdYlGn",
                range_color=[0.7, 1.0],
                labels={"Label": "Store", "FillRate": "Fill Rate"},
            )
            fig3.update_layout(yaxis_tickformat=".1%")
            st.plotly_chart(fig3, use_container_width=True)

        # Top SKUs with missed repairs
        top_miss = (
            sim_data.groupby(["StoreNum", "PartNum"])["MissedRepairs"]
            .sum()
            .reset_index()
            .query("MissedRepairs > 0")
            .sort_values("MissedRepairs", ascending=False)
            .head(20)
        )
        if len(top_miss):
            top_miss["PartDesc"] = top_miss["PartNum"].map(part_names)
            top_miss["Label"] = (
                top_miss["PartDesc"].fillna(top_miss["PartNum"].astype(str)).str[:35]
                + "  (Store "
                + top_miss["StoreNum"].astype(str)
                + ")"
            )
            fig4 = px.bar(
                top_miss, x="MissedRepairs", y="Label", orientation="h",
                title="Top 20 SKUs by Total Missed Repairs",
                color="MissedRepairs",
                color_continuous_scale="Reds",
                labels={"Label": "", "MissedRepairs": "Total Missed Repairs"},
            )
            fig4.update_layout(height=max(350, len(top_miss) * 28), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.success("🎉  No missed repairs in the simulation period!")

        # Orders per store
        ord_store = (
            sim_data.groupby("StoreNum")
            .agg(TotalOrders=("Ordered", lambda x: (x > 0).sum()))
            .reset_index()
        )
        ord_store["Label"] = ord_store["StoreNum"].map(store_names).fillna(
            ord_store["StoreNum"].astype(str)
        )
        fig5 = px.bar(
            ord_store, x="Label", y="TotalOrders",
            title="Total Order Events by Store",
            labels={"Label": "Store", "TotalOrders": "Orders"},
            color_discrete_sequence=["#2ca02c"],
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── Tab: Sim Data ─────────────────────────────────────────
    with tab_sim:
        st.markdown(f"**{len(sim_filt):,} rows** — simulation output")
        st.dataframe(sim_filt.round(2), use_container_width=True, height=420, hide_index=True)
        st.download_button(
            "⬇️  Download Sim_Data.csv",
            to_csv_bytes(sim_filt),
            "Sim_Data.csv",
            "text/csv",
        )

    # ── Tab: Recommendations ─────────────────────────────────
    with tab_rec:
        st.markdown(f"**{len(rec_filt):,} SKU/Store combinations**")

        # Colour-hint columns for display
        disp = rec_filt.copy()
        for col in ["FillRate", "AvgDailyDemand", "AvgInventory"]:
            if col in disp.columns:
                disp[col] = disp[col].round(3)

        st.dataframe(disp, use_container_width=True, height=420, hide_index=True)

        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button(
            "⬇️  Download Recommended_Min_Max.csv",
            to_csv_bytes(rec_filt),
            "Recommended_Min_Max.csv",
            "text/csv",
        )

        # Reason summary chart
        if len(rec_filt) and "RecommendationReason" in rec_filt.columns:
            st.subheader("Recommendation Reason Breakdown")
            reason_cts = (
                rec_filt["RecommendationReason"]
                .value_counts()
                .reset_index()
            )
            reason_cts.columns = ["Reason", "Count"]
            fig_r = px.bar(
                reason_cts, x="Count", y="Reason", orientation="h",
                title="SKUs by Recommendation Reason",
                color_discrete_sequence=["#9467bd"],
            )
            fig_r.update_layout(
                height=max(250, len(reason_cts) * 38),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_r, use_container_width=True)

    # ── Tab: Min/Max Calc ────────────────────────────────────
    with tab_mm:
        st.markdown("Calculated demand stats and proposed Min/Max values per SKU/Store.")

        display_cols = [
            "StoreNum", "PartNum", "PartDesc",
            "AvgDailyDemand", "DemandStdDev", "SafetyStock",
            "RecommendedMin", "RecommendedMax",
            "CurrentMin", "CurrentMax",
            "CalculationMode",
        ]
        mm_disp = filter_store_part(min_max_calc)
        mm_disp = mm_disp[[c for c in display_cols if c in mm_disp.columns]]

        st.dataframe(mm_disp.round(4), use_container_width=True, height=420, hide_index=True)
        st.download_button(
            "⬇️  Download MinMax_Calc.csv",
            to_csv_bytes(mm_disp),
            "MinMax_Calc.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()
