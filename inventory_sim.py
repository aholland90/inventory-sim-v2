"""
Inventory Simulation Application
==================================
Simulates store-level inventory over time using:
  • Consumption data        (uploaded by user)
  • Min/Max settings        (editable inside the app)
  • Store order schedules   (editable inside the app)

Outputs Sim_Data format:
  ConDate | StoreNum | PartNum | Consumption | InventoryOH | Ordered | Received | MissedRepairs

Assumptions & Defaults (configurable in sidebar):
  • Lead time              : 1 day  (orders placed today arrive tomorrow)
  • Starting inventory     : Max value for each Store/Part
  • Missing Min/Max        : that Store/Part is skipped with a warning
  • No order days selected : that store never orders (it is warned)
  • Consumption applied    : before order-placement check each day
  • DayOfWeek convention   : 1=Sun  2=Mon  3=Tue  4=Wed  5=Thu  6=Fri  7=Sat
"""

import io
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Simulation",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand palette & global CSS ────────────────────────────────────
PALETTE = {
    "purple":     "#8223D2",
    "blue":       "#6B8BFF",
    "green":      "#6EFAC3",
    "lime":       "#D2FA46",
    "dark":       "#333F48",
    "gray":       "#A5AAAF",
    "light_gray": "#E6E6EB",
    "indigo":     "#6469E1",
    "lavender":   "#E0E1F9",
    "mint":       "#E2FEF3",
    "lilac":      "#E6D2F7",
}

st.markdown(
    """
    <style>
    /* Multiselect tag/bubble color */
    span[data-baseweb="tag"] {
        background-color: #E6D2F7 !important;
        color: #333F48 !important;
    }
    span[data-baseweb="tag"] span { color: #333F48 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────
# Our day numbering: 1=Sun … 7=Sat
DAY_MAP = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}

REQUIRED_COLUMNS = {
    "All_Dates":           [],          # date column auto-detected
    "Consumed":            ["StoreNum", "Consumption"],
    "Min_Max_Store_SKU":   ["StoreNum", "PartNum", "Min", "Max"],
    "Store_List":          ["StoreNum", "StoreDesc"],
    "Store_Order_Schedule":["StoreNum", "DayOfWeek", "OrderYes"],
}

# Alternative column names accepted (mapped → canonical name)
COLUMN_ALIASES = {
    "ConDate":     ["ConDate", "Date", "Con Date", "con_date", "condate",
                    "OrderDate", "order_date", "CONDATE", "DATE"],
    "StoreNum":    ["StoreNum", "Store Num", "Store_Num", "StoreNumber",
                    "Store Number", "store_num", "STORENUM"],
    "PartNum":     ["PartNum", "Part Num", "Part_Num", "PartNumber",
                    "Part Number", "part_num", "PARTNUM", "SKU"],
    "Consumption": ["Consumption", "consumption", "CONSUMPTION", "Qty",
                    "quantity", "Amount"],
    "StoreDesc":   ["StoreDesc", "Store Desc", "Store_Desc", "StoreName",
                    "Store Name", "store_desc", "STOREDESC"],
    "DayOfWeek":   ["DayOfWeek", "Day Of Week", "Day_Of_Week", "day_of_week",
                    "DOW", "DAYOFWEEK"],
    "OrderYes":    ["OrderYes", "Order Yes", "Order_Yes", "order_yes",
                    "ORDERYES", "CanOrder", "can_order"],
    "Min":         ["Min", "min", "MIN", "Minimum", "minimum"],
    "Max":         ["Max", "max", "MAX", "Maximum", "maximum"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using COLUMN_ALIASES."""
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col in aliases and col != canonical:
                rename[col] = canonical
    return df.rename(columns=rename)


def detect_date_column(df: pd.DataFrame) -> str | None:
    """Return the first column that looks like a date column."""
    date_hints = COLUMN_ALIASES["ConDate"]
    for col in df.columns:
        if col in date_hints:
            return col
    # Fallback: first column that can be parsed as dates
    for col in df.columns:
        try:
            pd.to_datetime(df[col].dropna().head(5))
            return col
        except Exception:
            pass
    return None

# ──────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────
_STATE_DEFAULTS = {
    "all_dates":           None,   # DataFrame
    "consumed":            None,   # DataFrame
    "min_max":             None,   # DataFrame (editable)
    "store_list":          None,   # DataFrame
    "order_schedule":      None,   # DataFrame (editable)
    "sim_results":         None,   # DataFrame — output
    "low_velocity_skus":   None,   # set of (StoreNum, PartNum) tuples
    "lead_time":           1,
    "start_inv":           "max",  # "max" | "zero"
    "low_vel_threshold":   0.1,    # ADD below this = low velocity
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ──────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def pandas_dow_to_ours(ts: pd.Timestamp) -> int:
    """
    Convert a pandas Timestamp to our day-of-week convention.
    pandas dayofweek: 0=Mon … 6=Sun
    ours:             1=Sun  2=Mon … 7=Sat
    Formula: (pandas_dow + 1) % 7 + 1
    """
    return (ts.dayofweek + 1) % 7 + 1


def parse_excel(file) -> dict[str, pd.DataFrame]:
    """Read every sheet in an uploaded Excel file."""
    xl = pd.ExcelFile(file)
    return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}


def validate_df(df: pd.DataFrame, required_cols: list[str], name: str) -> tuple[bool, str]:
    """Return (ok, error_message)."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"**{name}** is missing columns: {', '.join(missing)}"
    return True, ""


def coerce_order_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DayOfWeek"] = pd.to_numeric(df["DayOfWeek"], errors="coerce").fillna(0).astype(int)
    # Coerce OrderYes — accept 1/0, True/False, "yes"/"no", "1"/"0"
    raw = df["OrderYes"].astype(str).str.strip().str.lower()
    df["OrderYes"] = raw.isin(["1", "true", "yes"])
    return df


def store_name(num, store_list_df):
    """Return StoreDesc for a given StoreNum, or the number as a string."""
    if store_list_df is None:
        return str(num)
    row = store_list_df.loc[store_list_df["StoreNum"] == num, "StoreDesc"]
    return row.values[0] if len(row) else str(num)


def velocity_tag(store_num, part_num) -> str:
    """Return 'Low' or 'High' for a given Store×Part using session state."""
    lv = st.session_state.get("low_velocity_skus") or set()
    return "Low" if (store_num, part_num) in lv else "High"


def int_opts(series: pd.Series) -> list[int]:
    """Return sorted unique integer values from a series (drops NaN)."""
    return sorted(int(v) for v in series.dropna().unique())


def vel_filter_widget(key: str) -> str:
    """Render a Velocity selectbox and return the selection."""
    return st.selectbox("Velocity", ["All", "High", "Low"], key=key)


def load_tab(df: pd.DataFrame, name: str) -> bool:
    """Validate, coerce, and store a tab in session state. Returns success."""
    df = df.copy()
    df = normalize_columns(df)   # remap aliases → canonical names

    if name == "All_Dates":
        # Accept any recognisable date column
        date_col = detect_date_column(df)
        if date_col is None:
            st.warning(
                f"**All_Dates**: could not find a date column. "
                f"Expected a column named one of: {', '.join(COLUMN_ALIASES['ConDate'])}. "
                f"Columns found: {', '.join(df.columns)}"
            )
            return False
        df = df.rename(columns={date_col: "ConDate"})
        df["ConDate"] = pd.to_datetime(df["ConDate"])
        st.session_state.all_dates = df[["ConDate"]]
        return True

    req = REQUIRED_COLUMNS.get(name, [])
    ok, err = validate_df(df, req, name)
    if not ok:
        st.warning(err)
        return False

    if name == "Consumed":
        date_col = detect_date_column(df)
        if date_col and date_col != "ConDate":
            df = df.rename(columns={date_col: "ConDate"})
        df["ConDate"] = pd.to_datetime(df["ConDate"])
        df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce").fillna(0)
        st.session_state.consumed = df

    elif name == "Min_Max_Store_SKU":
        df["Min"] = pd.to_numeric(df["Min"], errors="coerce").fillna(0)
        df["Max"] = pd.to_numeric(df["Max"], errors="coerce").fillna(0)
        st.session_state.min_max = df

    elif name == "Store_List":
        st.session_state.store_list = df

    elif name == "Store_Order_Schedule":
        df = coerce_order_schedule(df)
        st.session_state.order_schedule = df

    return True


# ──────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────────

def run_simulation(
    all_dates_df:      pd.DataFrame,
    consumed_df:       pd.DataFrame,
    min_max_df:        pd.DataFrame,
    order_schedule_df: pd.DataFrame,
    lead_time:         int  = 1,
    start_inv:         str  = "max",
) -> pd.DataFrame:
    """
    Core simulation loop.

    Date spine: unique dates from Consumed (left table) left-joined to
    All_Dates. This means every date with consumption is simulated; dates
    in All_Dates that have no consumption rows are excluded unless they
    fall within the consumed date range (in which case they appear with
    zero consumption).

    Steps per (StoreNum, PartNum) per date:
      1. Receive any in-transit orders due today.
      2. Apply consumption; record MissedRepairs if stock is insufficient.
      3. Check if today is an allowed order day for this store.
      4. If InventoryOH ≤ Min → place an order for (Max − InventoryOH) units.
      5. Schedule receipt of that order `lead_time` days later.

    Returns DataFrame in Sim_Data format.
    """

    # ── Consumption lookup ────────────────────────────────────────
    cons = consumed_df.copy()
    cons["ConDate"] = pd.to_datetime(cons["ConDate"])

    # ── Master calendar ───────────────────────────────────────────
    # All_Dates drives the calendar, but clipped to the date range
    # covered by the Consumed data (e.g. last 90 days).
    # This prevents simulating years of history when consumption only
    # covers a recent window.
    c_min = cons["ConDate"].min()
    c_max = cons["ConDate"].max()

    if all_dates_df is not None:
        all_d = pd.to_datetime(all_dates_df["ConDate"]).drop_duplicates()
        dates = (
            all_d[(all_d >= c_min) & (all_d <= c_max)]
            .sort_values()
            .values
        )
        if len(dates) == 0:
            # Fallback: no overlap — just use consumed dates directly
            dates = cons["ConDate"].drop_duplicates().sort_values().values
    else:
        dates = cons["ConDate"].drop_duplicates().sort_values().values

    has_part = "PartNum" in cons.columns

    if has_part:
        cons_idx = (
            cons.groupby(["ConDate", "StoreNum", "PartNum"])["Consumption"]
            .sum()
        )
    else:
        cons_idx = (
            cons.groupby(["ConDate", "StoreNum"])["Consumption"]
            .sum()
        )

    # ── Order-schedule lookup: store → set of allowed day-of-week ─
    order_allowed: dict[object, set[int]] = {}
    for _, row in order_schedule_df.iterrows():
        s = row["StoreNum"]
        if s not in order_allowed:
            order_allowed[s] = set()
        if row["OrderYes"]:
            order_allowed[s].add(int(row["DayOfWeek"]))

    # ── Store×Part combinations — Consumed is the driver ──────────
    # All SKUs that appear in Consumed are included. Min/Max is
    # left-joined: SKUs without Min/Max are simulated but never order.
    if has_part:
        consumed_combos = cons[["StoreNum", "PartNum"]].drop_duplicates()
    else:
        # No PartNum in consumed — pair each store with its parts from Min/Max
        consumed_combos = (
            cons[["StoreNum"]].drop_duplicates()
            .merge(min_max_df[["StoreNum", "PartNum"]].drop_duplicates(),
                   on="StoreNum", how="inner")
        )

    combos_df = consumed_combos.merge(
        min_max_df[["StoreNum", "PartNum", "Min", "Max"]],
        on=["StoreNum", "PartNum"], how="left",
    )

    results: list[dict] = []

    for _, combo_row in combos_df.iterrows():
        store_num = combo_row["StoreNum"]
        part_num  = combo_row["PartNum"]
        has_mm    = pd.notna(combo_row.get("Min")) and pd.notna(combo_row.get("Max"))
        min_val   = float(combo_row["Min"]) if has_mm else 0.0
        max_val   = float(combo_row["Max"]) if has_mm else 0.0

        # Starting inventory — only meaningful when Min/Max exists
        inv_oh: float = max_val if (start_inv == "max" and has_mm) else 0.0

        # Orders in transit: {date_key → float}
        transit: dict = {}

        allowed_days = order_allowed.get(store_num, set())

        for date in dates:
            ts = pd.Timestamp(date)
            date_key = ts.normalize()  # strip time component

            # 1. Receive in-transit orders
            received_today: float = transit.pop(date_key, 0.0)
            inv_oh += received_today

            # 2. Consumption for today
            if has_part:
                consumption: float = float(cons_idx.get((date_key, store_num, part_num), 0.0))
            else:
                consumption: float = float(cons_idx.get((date_key, store_num), 0.0))

            # Apply consumption; cap at available inventory
            if consumption > inv_oh:
                missed: float = consumption - inv_oh
                inv_oh = 0.0
            else:
                missed: float = 0.0
                inv_oh -= consumption

            # 3. Order check — only when Min/Max is defined
            our_dow = pandas_dow_to_ours(ts)
            ordered_today: float = 0.0

            if has_mm and our_dow in allowed_days and inv_oh <= min_val:
                order_qty = max_val - inv_oh
                if order_qty > 0:
                    ordered_today = order_qty
                    receive_date = (ts + timedelta(days=max(lead_time, 0))).normalize()
                    transit[receive_date] = transit.get(receive_date, 0.0) + order_qty

            results.append({
                "ConDate":      date_key,
                "StoreNum":     store_num,
                "PartNum":      part_num,
                "Consumption":  round(consumption, 4),
                "InventoryOH":  round(inv_oh, 4),
                "Ordered":      round(ordered_today, 4),
                "Received":     round(received_today, 4),
                "MissedRepairs":round(missed, 4),
                "HasMinMax":    has_mm,
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────
# SIDEBAR — Global Settings
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Simulation Settings")

    st.session_state.lead_time = st.number_input(
        "Lead Time (days)",
        min_value=0, max_value=365,
        value=st.session_state.lead_time,
        step=1,
        help="Days between order placement and receipt. 0 = same-day receipt.",
    )

    st.session_state.start_inv = st.selectbox(
        "Starting Inventory",
        options=["max", "zero"],
        index=0 if st.session_state.start_inv == "max" else 1,
        format_func=lambda x: "Start at Max (recommended)" if x == "max" else "Start at Zero",
        help="Initial on-hand inventory at the first simulation date.",
    )

    st.session_state.low_vel_threshold = st.number_input(
        "Low-Velocity Threshold (avg units/day)",
        min_value=0.0, max_value=10.0,
        value=float(st.session_state.low_vel_threshold),
        step=0.05, format="%.2f",
        help="SKUs with average daily demand below this value are flagged as low velocity.",
    )

    st.divider()
    with st.expander("📋 Built-in Assumptions"):
        st.caption("• Orders placed when inventory ≤ Min on an allowed order day")
        st.caption("• Order qty = Max − current inventory")
        st.caption("• Consumption applied before the order check")
        st.caption("• Store/Parts without Min/Max are excluded")
        st.caption("• Stores with no order days selected never order")
        st.caption("• DayOfWeek: 1=Sun  2=Mon  …  7=Sat")

    st.divider()
    if st.button("🔄 Clear All Data", use_container_width=True):
        for k in _STATE_DEFAULTS:
            st.session_state[k] = _STATE_DEFAULTS[k]
        st.rerun()


# ──────────────────────────────────────────────────────────────────
# APP HEADER
# ──────────────────────────────────────────────────────────────────
st.title("📦 Inventory Simulation")
st.caption(
    "Upload your data → edit Min/Max and order schedules → run the simulation → explore results."
)

tab_upload, tab_minmax, tab_schedule, tab_results, tab_summary, tab_recommend = st.tabs([
    "📁  Data Upload",
    "⚖️  Min / Max Editor",
    "📅  Order Schedule",
    "▶️  Simulation Results",
    "📊  Summary & Charts",
    "🎯  Min/Max Recommendations",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — DATA UPLOAD
# ══════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("Upload Input Data")
    st.caption("Upload a single Excel workbook containing all required sheets.")

    excel_file = st.file_uploader(
        "Upload Excel (.xlsx)",
        type=["xlsx", "xls"],
        key="xl_upload",
    )

    if excel_file:
        # Only re-process when the file actually changes (name + size fingerprint).
        # Without this guard, Streamlit re-fires the uploader on every widget
        # interaction, which clears sim_results and resets the app.
        file_id = f"{excel_file.name}_{excel_file.size}"
        if st.session_state.get("_last_xl_id") != file_id:
            st.session_state["_last_xl_id"] = file_id
            with st.spinner("Reading workbook…"):
                sheets = parse_excel(excel_file)
                loaded, skipped = [], []
                for tab_name in REQUIRED_COLUMNS:
                    if tab_name in sheets:
                        ok = load_tab(sheets[tab_name], tab_name)
                        (loaded if ok else skipped).append(tab_name)
                    else:
                        st.warning(f"Sheet **{tab_name}** not found in the uploaded file.")
                        skipped.append(tab_name)
                if loaded:
                    st.success(f"✅ Loaded: {', '.join(loaded)}")
                if skipped:
                    st.warning(f"⚠️ Not loaded: {', '.join(skipped)}")
                st.session_state.sim_results = None  # invalidate on new file only

    # ── Status / Preview ──────────────────────────────────────────
    st.divider()
    st.subheader("Loaded Data Status")

    data_state = {
        "All_Dates":            st.session_state.all_dates,
        "Consumed":             st.session_state.consumed,
        "Min_Max_Store_SKU":    st.session_state.min_max,
        "Store_List":           st.session_state.store_list,
        "Store_Order_Schedule": st.session_state.order_schedule,
    }

    stat_cols = st.columns(5)
    for i, (name, df) in enumerate(data_state.items()):
        with stat_cols[i]:
            if df is not None:
                st.success(f"✅ **{name}**\n{len(df):,} rows")
            else:
                st.error(f"❌ **{name}**\nNot loaded")

    st.divider()
    st.subheader("Data Previews")
    prev_tabs = st.tabs(list(data_state.keys()))
    for i, (name, df) in enumerate(data_state.items()):
        with prev_tabs[i]:
            if df is not None:
                st.dataframe(df.head(100), use_container_width=True)
            else:
                st.info(f"No data loaded for **{name}** yet.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — MIN/MAX EDITOR
# ══════════════════════════════════════════════════════════════════
with tab_minmax:
    st.header("Min / Max Editor")
    st.caption(
        "Edit Min and Max inventory levels for each Store × Part combination. "
        "Saving changes will invalidate the simulation (re-run required)."
    )

    if st.session_state.min_max is None:
        st.warning("Upload **Min_Max_Store_SKU** data first (Data Upload tab).")
    else:
        mm = st.session_state.min_max.copy()
        sl = st.session_state.store_list

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            s_opts_mm = int_opts(mm["StoreNum"])
            sel_store = st.multiselect("Store (leave blank = all)", s_opts_mm, key="mm_sf")
        with f2:
            p_opts_mm = int_opts(mm["PartNum"])
            sel_part  = st.multiselect("Part (leave blank = all)", p_opts_mm, key="mm_pf")
        with f3:
            mm_vel = vel_filter_widget("mm_vel")

        filtered_mm = mm.copy()
        if sel_store:
            filtered_mm = filtered_mm[filtered_mm["StoreNum"].astype(int).isin(sel_store)]
        if sel_part:
            filtered_mm = filtered_mm[filtered_mm["PartNum"].astype(int).isin(sel_part)]
        if mm_vel != "All" and st.session_state.low_velocity_skus:
            lv = st.session_state.low_velocity_skus
            mask = filtered_mm.apply(
                lambda r: velocity_tag(r["StoreNum"], r["PartNum"]) == mm_vel, axis=1
            )
            filtered_mm = filtered_mm[mask]

        # Merge store description for display
        display_mm = filtered_mm.copy()
        if sl is not None:
            display_mm = display_mm.merge(sl[["StoreNum", "StoreDesc"]], on="StoreNum", how="left")
        col_order = [c for c in ["StoreNum", "StoreDesc", "PartNum", "Min", "Max"] if c in display_mm.columns]
        display_mm = display_mm[col_order]

        st.info(f"Showing **{len(display_mm):,}** records. Click any Min or Max cell to edit.")

        edited_mm = st.data_editor(
            display_mm,
            use_container_width=True,
            num_rows="dynamic",
            key="mm_editor",
            column_config={
                "StoreNum":  st.column_config.TextColumn("Store #",    disabled=True),
                "StoreDesc": st.column_config.TextColumn("Store Name", disabled=True),
                "PartNum":   st.column_config.TextColumn("Part #",     disabled=True),
                "Min": st.column_config.NumberColumn("Min", min_value=0, step=1, format="%d"),
                "Max": st.column_config.NumberColumn("Max", min_value=0, step=1, format="%d"),
            },
        )

        if st.button("💾  Save Min/Max Changes", type="primary"):
            # Write edits back into the master DataFrame
            edit_core = edited_mm[["StoreNum", "PartNum", "Min", "Max"]].copy()
            master = st.session_state.min_max.set_index(["StoreNum", "PartNum"])
            updates = edit_core.set_index(["StoreNum", "PartNum"])
            master.update(updates)
            st.session_state.min_max = master.reset_index()
            st.session_state.sim_results = None
            st.success("✅ Min/Max updated! Re-run the simulation to see new results.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ORDER SCHEDULE EDITOR
# ══════════════════════════════════════════════════════════════════
with tab_schedule:
    st.header("Order Schedule Editor")
    st.caption(
        "Check a box to allow orders on that day. Uncheck to block orders. "
        "DayOfWeek convention: **1=Sun  2=Mon  3=Tue  4=Wed  5=Thu  6=Fri  7=Sat**"
    )

    if st.session_state.order_schedule is None:
        st.warning("Upload **Store_Order_Schedule** data first (Data Upload tab).")
    else:
        sch = st.session_state.order_schedule.copy()
        sl  = st.session_state.store_list

        # Pivot to wide: one row per store, one bool column per day
        pivot = (
            sch.pivot_table(
                index="StoreNum",
                columns="DayOfWeek",
                values="OrderYes",
                aggfunc="first",
            )
            .reindex(columns=range(1, 8), fill_value=False)
            .fillna(False)
            .astype(bool)
        )
        pivot.columns = [DAY_MAP[c] for c in pivot.columns]
        pivot = pivot.reset_index()

        # Add store name
        if sl is not None:
            pivot = pivot.merge(sl[["StoreNum", "StoreDesc"]], on="StoreNum", how="left")
        else:
            pivot["StoreDesc"] = pivot["StoreNum"].astype(str)

        # Weekly avg orders from simulation (if available)
        sim = st.session_state.sim_results
        if sim is not None:
            n_weeks = max((sim["ConDate"].max() - sim["ConDate"].min()).days / 7, 1)
            weekly_avg = (
                sim[sim["Ordered"] > 0]
                .groupby("StoreNum")["Ordered"]
                .sum()
                .div(n_weeks)
                .round(1)
                .rename("AvgWeeklyOrders")
                .reset_index()
            )
            pivot = pivot.merge(weekly_avg, on="StoreNum", how="left")
            pivot["AvgWeeklyOrders"] = pivot["AvgWeeklyOrders"].fillna(0)

        day_cols = list(DAY_MAP.values())
        col_order = ["StoreNum", "StoreDesc"]
        if "AvgWeeklyOrders" in pivot.columns:
            col_order.append("AvgWeeklyOrders")
        col_order += day_cols
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        st.info("Each row is one store. Check the days orders are allowed.")

        edited_sch = st.data_editor(
            pivot,
            use_container_width=True,
            key="sch_editor",
            column_config={
                "StoreNum":        st.column_config.TextColumn("Store #",             disabled=True),
                "StoreDesc":       st.column_config.TextColumn("Store Name",          disabled=True),
                "AvgWeeklyOrders": st.column_config.NumberColumn("Avg Weekly Orders", disabled=True, format="%.1f"),
                **{day: st.column_config.CheckboxColumn(day) for day in day_cols},
            },
        )

        if st.button("💾  Save Schedule Changes", type="primary"):
            day_name_to_num = {v: k for k, v in DAY_MAP.items()}
            rows = []
            for _, row in edited_sch.iterrows():
                for day_name, day_num in day_name_to_num.items():
                    rows.append({
                        "StoreNum":  row["StoreNum"],
                        "DayOfWeek": day_num,
                        "OrderYes":  bool(row.get(day_name, False)),
                    })
            st.session_state.order_schedule = pd.DataFrame(rows)
            st.session_state.sim_results = None
            st.success("✅ Order schedule updated! Re-run the simulation to see new results.")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — SIMULATION RESULTS
# ══════════════════════════════════════════════════════════════════
with tab_results:
    st.header("Simulation Results")

    # All_Dates is optional — the simulation derives its date spine from Consumed.
    needed = {
        "Consumed":             st.session_state.consumed,
        "Min_Max_Store_SKU":    st.session_state.min_max,
        "Store_Order_Schedule": st.session_state.order_schedule,
    }
    missing = [k for k, v in needed.items() if v is None]

    if missing:
        st.warning(f"Please upload the following data first: **{', '.join(missing)}**")
    else:
        run_col, info_col = st.columns([1, 3])
        with run_col:
            run_btn = st.button("▶️  Run Simulation", type="primary", use_container_width=True)
        with info_col:
            if st.session_state.sim_results is not None:
                n = len(st.session_state.sim_results)
                st.info(f"Last run produced **{n:,}** rows. Re-run to apply any changes.")

        if run_btn:
            with st.spinner("Running simulation…"):
                try:
                    results = run_simulation(
                        st.session_state.all_dates,
                        st.session_state.consumed,
                        st.session_state.min_max,
                        st.session_state.order_schedule,
                        lead_time=st.session_state.lead_time,
                        start_inv=st.session_state.start_inv,
                    )
                    st.session_state.sim_results = results

                    # ── Low-velocity flagging ─────────────────────
                    lv_thresh = st.session_state.low_vel_threshold
                    n_dates   = results["ConDate"].nunique()
                    add_per_sku = (
                        results.groupby(["StoreNum", "PartNum"])["Consumption"]
                        .sum()
                        .div(max(n_dates, 1))
                    )
                    lv_skus = set(
                        add_per_sku[add_per_sku < lv_thresh].index.tolist()
                    )
                    st.session_state.low_velocity_skus = lv_skus

                    st.success(
                        f"✅ Simulation complete! {len(results):,} rows — "
                        f"{len(lv_skus):,} low-velocity SKUs flagged."
                    )
                except Exception as exc:
                    st.error(f"Simulation error: {exc}")

        # ── Results display ───────────────────────────────────────
        if st.session_state.sim_results is not None:
            res = st.session_state.sim_results.copy()
            sl  = st.session_state.store_list
            if sl is not None:
                res = res.merge(sl[["StoreNum", "StoreDesc"]], on="StoreNum", how="left")

            # ── Tag Velocity on the full result set first ─────────
            lv_skus = st.session_state.low_velocity_skus or set()
            res["Velocity"] = res.apply(
                lambda r: "Low" if (r["StoreNum"], r["PartNum"]) in lv_skus else "High",
                axis=1,
            )

            st.divider()
            st.subheader("Filter Results")
            fc1, fc2, fc3 = st.columns(3)

            with fc1:
                # Cast to int so multiselect shows whole numbers
                s_opts = sorted(res["StoreNum"].dropna().astype(int).unique().tolist())
                s_sel  = st.multiselect("Store (leave blank = all)", s_opts, key="res_s")
            with fc2:
                p_opts = sorted(res["PartNum"].dropna().astype(int).unique().tolist())
                p_sel  = st.multiselect("Part (leave blank = all)", p_opts, key="res_p")
            with fc3:
                vel_sel = st.selectbox(
                    "Velocity",
                    options=["All", "High", "Low"],
                    key="res_vel",
                )

            filtered = res.copy()
            # Coerce to int for comparison to match the int options above
            if s_sel:
                filtered = filtered[filtered["StoreNum"].astype(int).isin(s_sel)]
            if p_sel:
                filtered = filtered[filtered["PartNum"].astype(int).isin(p_sel)]
            if vel_sel != "All":
                filtered = filtered[filtered["Velocity"] == vel_sel]

            # ── Filter-responsive KPI metrics ─────────────────────
            st.divider()
            f_cons   = filtered["Consumption"].sum()
            f_missed = filtered["MissedRepairs"].sum()
            f_fill   = (1 - f_missed / f_cons) * 100 if f_cons > 0 else 100.0
            f_avg_oh = filtered["InventoryOH"].mean()
            f_order_events = int((filtered["Ordered"] > 0).sum())
            f_skus_ordered = filtered[filtered["Ordered"] > 0]["PartNum"].nunique()

            f_lv = int(
                filtered[filtered["Velocity"] == "Low"][["StoreNum", "PartNum"]]
                .drop_duplicates()
                .shape[0]
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Demand",       f"{f_cons:,.0f}")
            m2.metric("Missed Repairs",     f"{f_missed:,.0f}")
            m3.metric("Fill Rate",          f"{f_fill:.1f}%")
            m4.metric("Avg Inventory OH",   f"{f_avg_oh:.1f}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Total Order Events", f"{f_order_events:,}")
            m6.metric("SKUs with Orders",   f"{f_skus_ordered:,}")
            m7.metric("Low-Velocity Flagged", f"{f_lv:,}")
            m8.metric("SKUs w/o Min/Max",
                      str(int(filtered[~filtered["HasMinMax"]]["PartNum"].nunique()))
                      if "HasMinMax" in filtered.columns else "—")

            # Friendly column order
            show_cols = ["ConDate", "StoreNum"]
            if "StoreDesc" in filtered.columns:
                show_cols.append("StoreDesc")
            show_cols += ["PartNum", "Velocity", "Consumption", "InventoryOH", "Ordered",
                          "Received", "MissedRepairs", "HasMinMax"]
            filtered_show = filtered[[c for c in show_cols if c in filtered.columns]]

            st.divider()
            st.subheader(f"Results — {len(filtered_show):,} rows")
            st.dataframe(
                filtered_show,
                use_container_width=True,
                column_config={
                    "ConDate":       st.column_config.DateColumn("Date"),
                    "Velocity":      st.column_config.TextColumn("Velocity"),
                    "InventoryOH":   st.column_config.NumberColumn("Inv. On Hand",  format="%.2f"),
                    "Consumption":   st.column_config.NumberColumn("Consumption",    format="%.2f"),
                    "Ordered":       st.column_config.NumberColumn("Ordered",        format="%.2f"),
                    "Received":      st.column_config.NumberColumn("Received",       format="%.2f"),
                    "MissedRepairs": st.column_config.NumberColumn("Missed Repairs", format="%.2f"),
                    "HasMinMax":     st.column_config.CheckboxColumn("Has Min/Max",  disabled=True),
                },
            )

            # ── Export ────────────────────────────────────────────
            st.divider()
            st.subheader("Export")
            ex1, ex2 = st.columns(2)

            with ex1:
                st.download_button(
                    "⬇️  Download CSV",
                    data=filtered_show.to_csv(index=False),
                    file_name="sim_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with ex2:
                xl_buf = io.BytesIO()
                with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                    filtered_show.to_excel(writer, index=False, sheet_name="Sim_Data")
                xl_buf.seek(0)
                st.download_button(
                    "⬇️  Download Excel",
                    data=xl_buf.getvalue(),
                    file_name="sim_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════
# TAB 5 — SUMMARY & CHARTS
# ══════════════════════════════════════════════════════════════════
with tab_summary:
    st.header("Summary & Charts")

    if st.session_state.sim_results is None:
        st.info("Run the simulation first (Simulation Results tab).")
    else:
        res = st.session_state.sim_results.copy()
        sl  = st.session_state.store_list
        if sl is not None:
            res = res.merge(sl[["StoreNum", "StoreDesc"]], on="StoreNum", how="left")

        # Tag velocity
        lv_s = st.session_state.low_velocity_skus or set()
        res["Velocity"] = res.apply(
            lambda r: "Low" if (r["StoreNum"], r["PartNum"]) in lv_s else "High", axis=1
        )

        # ── Summary-level velocity filter ─────────────────────────
        sv1, sv2, sv3 = st.columns(3)
        with sv1:
            sum_s_sel = st.multiselect(
                "Store (leave blank = all)", int_opts(res["StoreNum"]), key="sum_s"
            )
        with sv2:
            sum_p_sel = st.multiselect(
                "Part (leave blank = all)", int_opts(res["PartNum"]), key="sum_p"
            )
        with sv3:
            sum_vel = vel_filter_widget("sum_vel")

        res_f = res.copy()
        if sum_s_sel:
            res_f = res_f[res_f["StoreNum"].astype(int).isin(sum_s_sel)]
        if sum_p_sel:
            res_f = res_f[res_f["PartNum"].astype(int).isin(sum_p_sel)]
        if sum_vel != "All":
            res_f = res_f[res_f["Velocity"] == sum_vel]

        label_col = "StoreDesc" if "StoreDesc" in res_f.columns else "StoreNum"

        # ── KPI Cards ─────────────────────────────────────────────
        total_cons   = res_f["Consumption"].sum()
        total_ord    = res_f["Ordered"].sum()
        total_rec    = res_f["Received"].sum()
        total_missed = res_f["MissedRepairs"].sum()
        fill_rate    = (1 - total_missed / total_cons) * 100 if total_cons > 0 else 100.0

        st.subheader("📊 Key Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Consumption",         f"{total_cons:,.1f}")
        k2.metric("Total Ordered",             f"{total_ord:,.1f}")
        k3.metric("Total Received",            f"{total_rec:,.1f}")
        k4.metric("Missed Repairs",            f"{total_missed:,.1f}")
        k5.metric("Service Level / Fill Rate", f"{fill_rate:.1f}%")

        st.divider()

        # ── Store-level summary ───────────────────────────────────
        store_agg = (
            res_f.groupby([label_col], as_index=False)
            .agg(
                Consumption   =("Consumption",   "sum"),
                Ordered       =("Ordered",       "sum"),
                MissedRepairs =("MissedRepairs", "sum"),
            )
        )
        store_agg["FillRate%"] = (
            (1 - store_agg["MissedRepairs"] / store_agg["Consumption"].clip(lower=1e-9)) * 100
        ).round(1)

        ch1, ch2 = st.columns(2)

        with ch1:
            st.subheader("Missed Repairs by Store")
            fig_miss = px.bar(
                store_agg.sort_values("MissedRepairs", ascending=False),
                x=label_col, y="MissedRepairs",
                color_discrete_sequence=[PALETTE["purple"]],
                labels={"MissedRepairs": "Missed Repairs", label_col: "Store"},
            )
            fig_miss.update_traces(marker_line_color=PALETTE["dark"], marker_line_width=1)
            st.plotly_chart(fig_miss, use_container_width=True)

        with ch2:
            st.subheader("Fill Rate by Store")
            fig_fill = px.bar(
                store_agg.sort_values("FillRate%"),
                x=label_col, y="FillRate%",
                color_discrete_sequence=[PALETTE["green"]],
                labels={"FillRate%": "Fill Rate (%)", label_col: "Store"},
                range_y=[0, 105],
            )
            fig_fill.update_traces(marker_line_color=PALETTE["dark"], marker_line_width=1)
            st.plotly_chart(fig_fill, use_container_width=True)

        # ── Inventory Over Time ───────────────────────────────────
        st.divider()
        st.subheader("Inventory Over Time — Store × Part Detail")

        ic1, ic2 = st.columns(2)
        with ic1:
            chart_store = st.selectbox(
                "Select Store",
                int_opts(res_f["StoreNum"]),
                key="ch_store",
            )
        with ic2:
            chart_part = st.selectbox(
                "Select Part",
                int_opts(res_f["PartNum"]),
                key="ch_part",
            )

        chart_data = res_f[
            (res_f["StoreNum"] == chart_store) &
            (res_f["PartNum"]  == chart_part)
        ].sort_values("ConDate")

        if len(chart_data) == 0:
            st.info("No simulation data for the selected Store / Part combination.")
        else:
            mm_row = st.session_state.min_max[
                (st.session_state.min_max["StoreNum"] == chart_store) &
                (st.session_state.min_max["PartNum"]  == chart_part)
            ]

            fig_inv = go.Figure()

            fig_inv.add_trace(go.Scatter(
                x=chart_data["ConDate"], y=chart_data["InventoryOH"],
                mode="lines", name="Inventory On Hand",
                line=dict(color=PALETTE["blue"], width=2),
            ))
            fig_inv.add_trace(go.Bar(
                x=chart_data["ConDate"], y=chart_data["Consumption"],
                name="Consumption",
                marker_color=PALETTE["lime"],
            ))
            fig_inv.add_trace(go.Bar(
                x=chart_data["ConDate"], y=chart_data["MissedRepairs"],
                name="Missed Repairs",
                marker_color=PALETTE["purple"],
            ))
            fig_inv.add_trace(go.Scatter(
                x=chart_data["ConDate"], y=chart_data["Ordered"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=9, color=PALETTE["green"]),
                name="Orders Placed",
            ))

            if len(mm_row) > 0:
                mn = mm_row["Min"].values[0]
                mx = mm_row["Max"].values[0]
                fig_inv.add_hline(y=mn, line_dash="dash", line_color="red",
                                  annotation_text=f"Min ({mn:g})", annotation_position="top left")
                fig_inv.add_hline(y=mx, line_dash="dash", line_color="green",
                                  annotation_text=f"Max ({mx:g})", annotation_position="top left")

            fig_inv.update_layout(
                title=f"Store {chart_store}  |  Part {chart_part}",
                xaxis_title="Date",
                yaxis_title="Units",
                barmode="overlay",
                height=430,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_inv, use_container_width=True)

        # ── Orders by Day of Week ─────────────────────────────────
        st.divider()
        st.subheader("Orders by Day of Week")
        st.caption(
            "Aggregated across all weeks in the simulation. "
            "Shows total parts quantity ordered on each day of the week."
        )

        # Build a working orders-only frame with calendar fields
        DOW_ORDER  = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        ord_detail = res_f[res_f["Ordered"] > 0].copy()
        ord_detail["DayOfWeek"] = ord_detail["ConDate"].apply(
            lambda t: DOW_ORDER[pandas_dow_to_ours(t) - 1]   # 1=Sun→index 0
        )
        ord_detail["WeekStart"] = ord_detail["ConDate"].dt.to_period("W").apply(
            lambda p: p.start_time.date()
        )
        ord_detail["MonthName"] = ord_detail["ConDate"].dt.strftime("%b")
        ord_detail["MonthSort"] = ord_detail["ConDate"].dt.month   # for ordering

        # ── Bar chart: total qty ordered by day of week ───────────
        dow_agg = (
            ord_detail.groupby("DayOfWeek", as_index=False)["Ordered"].sum()
        )
        # Enforce Sun–Sat order
        dow_agg["_sort"] = dow_agg["DayOfWeek"].map({d: i for i, d in enumerate(DOW_ORDER)})
        dow_agg = dow_agg.sort_values("_sort").drop(columns="_sort")

        fig_dow = px.bar(
            dow_agg,
            x="DayOfWeek", y="Ordered",
            category_orders={"DayOfWeek": DOW_ORDER},
            labels={"DayOfWeek": "Day of Week", "Ordered": "Total Parts Qty Ordered"},
            title="Total Parts Ordered by Day of Week",
            color_discrete_sequence=[PALETTE["indigo"]],
        )
        fig_dow.update_traces(marker_line_color=PALETTE["dark"], marker_line_width=1)
        fig_dow.update_layout(xaxis_title="Day of Week")
        st.plotly_chart(fig_dow, use_container_width=True)

        # ── Pivot: Week (rows) × Day of Week (columns) ───────────
        st.subheader("By Week × Day of Week")
        week_dow = (
            ord_detail.groupby(["WeekStart", "DayOfWeek"])["Ordered"]
            .sum()
            .unstack("DayOfWeek")
            .reindex(columns=DOW_ORDER)
            .fillna(0)
            .reset_index()
        )
        week_dow.columns.name = None
        week_dow = week_dow.rename(columns={"WeekStart": "Week Starting"})
        # Row totals
        week_dow["Total"] = week_dow[DOW_ORDER].sum(axis=1)

        st.dataframe(
            week_dow,
            use_container_width=True,
            column_config={
                "Week Starting": st.column_config.DateColumn("Week Starting"),
                **{d: st.column_config.NumberColumn(d, format="%.1f") for d in DOW_ORDER},
                "Total": st.column_config.NumberColumn("Total", format="%.1f"),
            },
        )

        # ── Pivot: Month (rows) × Day of Week (columns) ──────────
        st.subheader("By Month × Day of Week")
        # One row per month — aggregate totals
        month_dow = (
            ord_detail.groupby(["MonthSort", "MonthName", "DayOfWeek"])["Ordered"]
            .sum()
            .reset_index()
            .pivot_table(index=["MonthSort", "MonthName"], columns="DayOfWeek",
                         values="Ordered", aggfunc="sum")
            .reindex(columns=DOW_ORDER)
            .fillna(0)
            .reset_index()
            .sort_values("MonthSort")
            .drop(columns="MonthSort")
        )
        month_dow.columns.name = None
        month_dow = month_dow.rename(columns={"MonthName": "Month"})
        month_dow["Total"] = month_dow[DOW_ORDER].sum(axis=1)

        st.dataframe(
            month_dow,
            use_container_width=True,
            column_config={
                "Month": st.column_config.TextColumn("Month"),
                **{d: st.column_config.NumberColumn(d, format="%.1f") for d in DOW_ORDER},
                "Total": st.column_config.NumberColumn("Total", format="%.1f"),
            },
        )

        # ── Pivot: Store (rows) × Day of Week (columns) ──────────
        st.divider()
        st.subheader("By Store × Day of Week")
        store_dow = (
            ord_detail.groupby(["StoreNum", "DayOfWeek"])["Ordered"]
            .sum()
            .reset_index()
            .pivot_table(index="StoreNum", columns="DayOfWeek",
                         values="Ordered", aggfunc="sum")
            .reindex(columns=DOW_ORDER)
            .fillna(0)
            .reset_index()
        )
        store_dow.columns.name = None
        store_dow["Total"] = store_dow[DOW_ORDER].sum(axis=1)
        # Optionally join store description
        if sl is not None:
            store_dow = store_dow.merge(sl[["StoreNum", "StoreDesc"]], on="StoreNum", how="left")
            front_cols = ["StoreNum", "StoreDesc"]
        else:
            front_cols = ["StoreNum"]
        store_dow = store_dow[front_cols + [c for c in DOW_ORDER if c in store_dow.columns] + ["Total"]]

        st.dataframe(
            store_dow,
            use_container_width=True,
            column_config={
                "StoreNum":  st.column_config.NumberColumn("Store #", format="%d"),
                "StoreDesc": st.column_config.TextColumn("Store"),
                **{d: st.column_config.NumberColumn(d, format="%.1f") for d in DOW_ORDER},
                "Total": st.column_config.NumberColumn("Total", format="%.1f"),
            },
        )

        # ── Store × Part summary table ────────────────────────────
        st.divider()
        st.subheader("Summary by Store & Part")

        part_agg = (
            res_f.groupby(["StoreNum", "PartNum"], as_index=False)
            .agg(
                TotalConsumption=("Consumption",   "sum"),
                TotalOrdered    =("Ordered",       "sum"),
                TotalReceived   =("Received",      "sum"),
                TotalMissed     =("MissedRepairs", "sum"),
            )
        )
        part_agg["FillRate%"] = (
            (1 - part_agg["TotalMissed"] / part_agg["TotalConsumption"].clip(lower=1e-9)) * 100
        ).round(1)

        if sl is not None:
            part_agg = part_agg.merge(sl[["StoreNum", "StoreDesc"]], on="StoreNum", how="left")

        col_ord = [c for c in ["StoreNum", "StoreDesc", "PartNum",
                                "TotalConsumption", "TotalOrdered", "TotalReceived",
                                "TotalMissed", "FillRate%"] if c in part_agg.columns]
        st.dataframe(part_agg[col_ord], use_container_width=True)



# ══════════════════════════════════════════════════════════════════
# TAB 6 — MIN/MAX RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
with tab_recommend:
    st.header("Min/Max Recommendations")
    st.caption(
        "Compares your current Min/Max settings against data-driven recommendations "
        "calculated from actual consumption history. Recommendations account for "
        "lead time and the average number of days between order opportunities."
    )

    needs_consumed  = st.session_state.consumed  is None
    needs_schedule  = st.session_state.order_schedule is None

    if needs_consumed or needs_schedule:
        missing_tabs = []
        if needs_consumed: missing_tabs.append("Consumed")
        if needs_schedule: missing_tabs.append("Store_Order_Schedule")
        st.warning(f"Please upload the following data first: **{', '.join(missing_tabs)}**")
    else:
        consumed_df   = st.session_state.consumed.copy()
        mm_df         = st.session_state.min_max.copy() if st.session_state.min_max is not None else None
        sch_df        = st.session_state.order_schedule.copy()
        sl            = st.session_state.store_list
        lead_time_val = st.session_state.lead_time

        # ── Sidebar params for recommendation ────────────────────
        st.sidebar.divider()
        st.sidebar.subheader("Recommendation Settings")
        target_fill_pct = st.sidebar.slider(
            "Target Fill Rate (%)",
            min_value=50, max_value=99, value=95, step=1,
            help="Desired service level. Drives safety-stock calculation "
                 "via the normal z-score. Higher = more safety stock = higher Min.",
        )

        # ── Average daily demand + std dev per Store × Part ──────
        consumed_df["ConDate"] = pd.to_datetime(consumed_df["ConDate"])
        has_part_col = "PartNum" in consumed_df.columns

        if has_part_col:
            grp = consumed_df.groupby(["StoreNum", "PartNum"])
            add_df = grp["Consumption"].agg(
                TotalCons="sum",
                StdDev="std",
                ActiveDays="count",
            ).reset_index()
        else:
            grp = consumed_df.groupby("StoreNum")
            add_store = grp["Consumption"].agg(
                TotalCons="sum", StdDev="std", ActiveDays="count"
            ).reset_index()
            parts_per_store = mm_df[["StoreNum", "PartNum"]].drop_duplicates() \
                if mm_df is not None else \
                consumed_df[["StoreNum"]].assign(PartNum=0).drop_duplicates()
            n_parts = parts_per_store.groupby("StoreNum")["PartNum"].count().rename("NParts")
            add_df = parts_per_store.merge(
                add_store.merge(n_parts, on="StoreNum"), on="StoreNum", how="left"
            )
            add_df["TotalCons"] /= add_df["NParts"].clip(lower=1)
            add_df["StdDev"]    /= add_df["NParts"].clip(lower=1)
            add_df = add_df[["StoreNum", "PartNum", "TotalCons", "StdDev", "ActiveDays"]]

        add_df["ADD"]    = (add_df["TotalCons"] / add_df["ActiveDays"].clip(lower=1)).round(4)
        add_df["StdDev"] = add_df["StdDev"].fillna(0)

        # ── Average order cycle ───────────────────────────────────
        order_days_count = (
            sch_df[sch_df["OrderYes"]]
            .groupby("StoreNum")["DayOfWeek"].nunique()
            .rename("OrderDaysPerWeek").reset_index()
        )
        order_days_count["AvgCycleDays"] = (
            7 / order_days_count["OrderDaysPerWeek"].clip(lower=1)
        ).round(1)

        # ── Build base: ALL consumed SKUs, left-join Min/Max ──────
        if has_part_col:
            base_combos = consumed_df[["StoreNum", "PartNum"]].drop_duplicates()
        else:
            base_combos = add_df[["StoreNum", "PartNum"]].drop_duplicates()

        mm_ref = mm_df[["StoreNum", "PartNum", "Min", "Max"]].copy() \
                 if mm_df is not None else \
                 pd.DataFrame(columns=["StoreNum", "PartNum", "Min", "Max"])

        rec = base_combos.merge(mm_ref, on=["StoreNum", "PartNum"], how="left")
        rec = rec.rename(columns={"Min": "CurrentMin", "Max": "CurrentMax"})
        rec["CurrentMin"] = rec["CurrentMin"].fillna(0).astype(int)
        rec["CurrentMax"] = rec["CurrentMax"].fillna(0).astype(int)
        rec = rec.merge(add_df[["StoreNum", "PartNum", "ADD", "StdDev"]],
                        on=["StoreNum", "PartNum"], how="left")
        rec = rec.merge(order_days_count[["StoreNum", "AvgCycleDays"]], on="StoreNum", how="left")
        rec["ADD"]          = rec["ADD"].fillna(0)
        rec["StdDev"]       = rec["StdDev"].fillna(0)
        rec["AvgCycleDays"] = rec["AvgCycleDays"].fillna(7)

        # ── Current fill% from simulation results ─────────────────
        sim = st.session_state.sim_results
        if sim is not None:
            sim_fill = (
                sim.groupby(["StoreNum", "PartNum"])
                .apply(lambda g: (
                    (1 - g["MissedRepairs"].sum() / g["Consumption"].sum()) * 100
                    if g["Consumption"].sum() > 0 else 100.0
                ))
                .rename("CurrentFill%")
                .reset_index()
            )
            rec = rec.merge(sim_fill, on=["StoreNum", "PartNum"], how="left")
            rec["CurrentFill%"] = rec["CurrentFill%"].fillna(100.0).round(1)
        else:
            rec["CurrentFill%"] = float("nan")

        # ── Recommended Min/Max via z-score safety stock ──────────
        # Safety stock = Z × σ × √(lead_time)
        # RecMin = ADD × lead_time + safety_stock
        # RecMax = RecMin + ADD × avg_cycle_days
        z_score = float(_norm.ppf(target_fill_pct / 100.0))
        rec["SafetyStock"] = np.ceil(
            z_score * rec["StdDev"] * np.sqrt(max(lead_time_val, 1))
        ).clip(lower=0)
        rec["RecMin"] = np.ceil(
            rec["ADD"] * lead_time_val + rec["SafetyStock"]
        ).astype(int).clip(lower=0)
        rec["RecMax"] = np.ceil(
            rec["RecMin"] + rec["ADD"] * rec["AvgCycleDays"]
        ).astype(int)
        rec["RecMax"] = rec[["RecMin", "RecMax"]].max(axis=1) + 1

        # ── Cap recommendations when current fill already meets target ─
        # If the SKU is already hitting the target fill rate, don't recommend
        # increasing Min or Max above what's currently working.
        already_meeting = rec["CurrentFill%"] >= target_fill_pct
        rec.loc[already_meeting, "RecMin"] = rec.loc[already_meeting, [
            "RecMin", "CurrentMin"]].min(axis=1)
        rec.loc[already_meeting, "RecMax"] = rec.loc[already_meeting, [
            "RecMax", "CurrentMax"]].min(axis=1)

        # ── Recommended fill% (re-use z → pct) ───────────────────
        rec["RecFill%"] = target_fill_pct

        # Delta columns
        rec["ΔMin"] = rec["RecMin"] - rec["CurrentMin"]
        rec["ΔMax"] = rec["RecMax"] - rec["CurrentMax"]

        # ── Velocity tag ──────────────────────────────────────────
        lv_r = st.session_state.low_velocity_skus or set()
        rec["Velocity"] = rec.apply(
            lambda r: "Low" if (r["StoreNum"], r["PartNum"]) in lv_r else "High", axis=1
        )

        # ── Comments ──────────────────────────────────────────────
        def make_comment(row):
            if row["CurrentMin"] == 0 and row["CurrentMax"] == 0:
                return "⚠️ No Min/Max set — please review"
            if row["ADD"] == 0:
                return "🔇 No consumption recorded — consider removing SKU"
            lv_thresh = st.session_state.low_vel_threshold
            if row["ADD"] < lv_thresh:
                return "🐢 Low-velocity SKU — verify still active"
            current_fill = row["CurrentFill%"]
            target_fill  = row["RecFill%"]
            delta_min = row["ΔMin"]
            delta_max = row["ΔMax"]
            pct_min = abs(delta_min) / max(row["CurrentMin"], 1)
            pct_max = abs(delta_max) / max(row["CurrentMax"], 1)
            # Currently meeting or exceeding target fill rate
            if current_fill >= target_fill:
                # Only suggest reducing if BOTH rec Min and rec Max are lower — consistent direction
                if delta_min < 0 and delta_max < 0 and pct_min > 0.2:
                    return "📉 Levels above target — consider reducing to free up capital"
                return "✅ Min/Max levels are appropriate"
            # Currently below target fill rate — recommend increases
            if delta_min > 0 and pct_min > 0.2:
                return "📈 Min too low — recommend increasing to avoid stockouts"
            if delta_max > 0 and pct_max > 0.2:
                return "📦 Max too low — recommend increasing order-up-to level"
            return "✅ Min/Max levels are appropriate"

        rec["Comments"] = rec.apply(make_comment, axis=1)

        # Add store description
        if sl is not None:
            rec = rec.merge(sl[["StoreNum", "StoreDesc"]], on="StoreNum", how="left")

        # ── Filters ───────────────────────────────────────────────
        rf1, rf2, rf3 = st.columns(3)
        with rf1:
            r_s_sel = st.multiselect("Store (leave blank = all)",
                                     int_opts(rec["StoreNum"]), key="rec_s")
        with rf2:
            r_p_sel = st.multiselect("Part (leave blank = all)",
                                     int_opts(rec["PartNum"]),  key="rec_p")
        with rf3:
            rec_vel = vel_filter_widget("rec_vel")

        rec_filtered = rec.copy()
        if r_s_sel:
            rec_filtered = rec_filtered[rec_filtered["StoreNum"].astype(int).isin(r_s_sel)]
        if r_p_sel:
            rec_filtered = rec_filtered[rec_filtered["PartNum"].astype(int).isin(r_p_sel)]
        if rec_vel != "All":
            rec_filtered = rec_filtered[rec_filtered["Velocity"] == rec_vel]

        # ── Display columns ───────────────────────────────────────
        disp_cols = [c for c in
            ["StoreNum", "StoreDesc", "PartNum", "Velocity", "ADD",
             "CurrentMin", "RecMin", "ΔMin",
             "CurrentMax", "RecMax", "ΔMax",
             "CurrentFill%", "RecFill%",
             "Comments"]
            if c in rec_filtered.columns]

        st.info(
            f"Showing **{len(rec_filtered):,}** records. "
            f"Target fill rate: **{target_fill_pct}%** (z={z_score:.2f}). "
            f"RecMin = ADD × lead time + safety stock.  RecMax = RecMin + ADD × avg order cycle."
        )

        st.dataframe(
            rec_filtered[disp_cols],
            use_container_width=True,
            column_config={
                "StoreNum":    st.column_config.NumberColumn("Store #",          format="%d"),
                "StoreDesc":   st.column_config.TextColumn("Store Name"),
                "PartNum":     st.column_config.NumberColumn("Part #",           format="%d"),
                "Velocity":    st.column_config.TextColumn("Velocity"),
                "ADD":         st.column_config.NumberColumn("Avg Daily Demand", format="%.3f"),
                "CurrentMin":  st.column_config.NumberColumn("Current Min",      format="%d"),
                "RecMin":      st.column_config.NumberColumn("Rec. Min",         format="%d"),
                "ΔMin":        st.column_config.NumberColumn("Δ Min",            format="%+d"),
                "CurrentMax":  st.column_config.NumberColumn("Current Max",      format="%d"),
                "RecMax":      st.column_config.NumberColumn("Rec. Max",         format="%d"),
                "ΔMax":        st.column_config.NumberColumn("Δ Max",            format="%+d"),
                "CurrentFill%":st.column_config.NumberColumn("Current Fill %",   format="%.1f"),
                "RecFill%":    st.column_config.NumberColumn("Target Fill %",    format="%d"),
                "Comments":    st.column_config.TextColumn("Comments",           width="large"),
            },
        )

        st.divider()

        # ── Apply recommendations button ──────────────────────────
        apply_col, _ = st.columns([1, 2])
        with apply_col:
            if st.button("⚡  Apply Recommended Values to Simulation", type="primary",
                         use_container_width=True):
                updated = st.session_state.min_max.set_index(["StoreNum", "PartNum"])
                apply_src = rec.set_index(["StoreNum", "PartNum"])
                updated["Min"] = apply_src["RecMin"]
                updated["Max"] = apply_src["RecMax"]
                st.session_state.min_max = updated.reset_index()
                st.session_state.sim_results = None
                st.success(
                    "✅ Recommended Min/Max values applied! "
                    "Go to the **Min/Max Editor** to review, or **Run Simulation** to see the impact."
                )

        # ── Delta distribution chart ──────────────────────────────
        st.divider()
        st.subheader("Δ Min and Δ Max Distribution")
        ch_dc1, ch_dc2 = st.columns(2)

        with ch_dc1:
            fig_dmin = px.histogram(
                rec_filtered, x="ΔMin", nbins=20,
                title="Recommended Min − Current Min",
                labels={"ΔMin": "Δ Min (units)"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_dmin.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_dmin, use_container_width=True)

        with ch_dc2:
            fig_dmax = px.histogram(
                rec_filtered, x="ΔMax", nbins=20,
                title="Recommended Max − Current Max",
                labels={"ΔMax": "Δ Max (units)"},
                color_discrete_sequence=["#ff7f0e"],
            )
            fig_dmax.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_dmax, use_container_width=True)

        # ── Export recommendations ────────────────────────────────
        st.divider()
        xl_rec = io.BytesIO()
        with pd.ExcelWriter(xl_rec, engine="openpyxl") as writer:
            rec_filtered[disp_cols].to_excel(writer, index=False, sheet_name="Recommendations")

        xl_rec.seek(0)
        st.download_button(
            "⬇️  Download Recommendations (Excel)",
            data=xl_rec.getvalue(),
            file_name="minmax_recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
