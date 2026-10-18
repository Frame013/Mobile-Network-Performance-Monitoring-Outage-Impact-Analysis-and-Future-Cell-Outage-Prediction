# ================================================================
# DATASET 2 GENERATOR - VERSION 2 (KPI-linked outages)
# ================================================================
#
# v2 CHANGES (search for "v2" to find every change):
#   1. Outages can now depend on the KPIs. A share of outages
#      (SIGNAL_STRENGTH) is preceded by a "degradation" period of
#      6-24 h in which packet loss, latency and alarms rise and
#      throughput falls. Some degradations recover without an outage
#      (false precursors). Overloaded cells degrade more often.
#   2. The other outages are sudden (no warning), as with power or
#      fiber cuts. Part of them are site-wide power failures that take
#      down every cell of a site at the same time.
#   3. Outage start times are spread over the whole 3-hour KPI interval
#      (version 1 only used the first hour).
#   4. Outage durations are long-tailed (log-normal) instead of uniform.
#   5. The true cause of every outage is written to a separate file,
#      outage_causes.csv. It is NOT part of Dataset 2 and must not be
#      used as a model feature; it is only for evaluation.
#   6. QUICK_TEST mode: 500 sites and 6 months, for fast checks.
#
# SIGNAL_STRENGTH = 0 gives version-1-like data with no warning signal.
# ================================================================
#
# Project:
# Mobile Network KPI Analysis & Performance Monitoring System
#
# Country:
# Uganda
#
# Dataset 1:
# Dataset_1_Cell_Information.csv
#
# Dataset 2:
# Cell-level KPI + OUTAGE + RECOVERY records
#
# Period:
# 2024-01-01 to 2025-12-31
#
# KPI interval:
# Every 3 hours
#
# Record types:
#   KPI       = normal KPI measurement
#   OUTAGE    = cell outage event
#   RECOVERY  = cell recovery event
#
# FINAL DESIGN:
#   - 6,000 sites
#   - 22,900 cells
#   - 4G/3G/2G on all sites
#   - 5G on 4,900 sites
#   - Cell-based outage
#   - Same-site neighbors allowed
#   - Same-district external neighbors allowed
#   - NO cross-district neighbors
#   - Neighbor_Cell_ID uses comma separator
#   - No KPI while a cell is down
#   - Historical outage count increases after recovery
#   - Neighbor cells are affected by outages
#   - Load may exceed Capacity
#   - Availability is not stored
#   - Utilization is not stored
#   - Outage_Duration is not stored
#   - Outage_Next_12h is not stored
#   - Continuous state across all 24 months
#   - Daily writing to reduce memory usage
#
# ================================================================


import os
import math  # v2
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ================================================================
# 1. RANDOM SEED
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ================================================================
# 2. FILE PATHS
# ================================================================

INPUT_FILE = (
    r"C:\Dataset\Uganda_Cell_Telecom_Dataset"
    r"\Dataset_1_Cell_Information.csv"
)

OUTPUT_DIR = (
    r"C:\Dataset\Uganda_Cell_Telecom_Dataset"
    r"\Dataset_2_Cell_KPI_Outage"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 3. SIMULATION PERIOD
# ================================================================

START_DATE = datetime(
    2024, 1, 1, 0, 0, 0
)

END_DATE = datetime(
    2025, 12, 31, 23, 59, 59
)


# ================================================================
# 4. KPI INTERVAL
# ================================================================

KPI_INTERVAL_HOURS = 3


# ================================================================
# 5. EXPECTED DATASET 1 SIZE
# ================================================================

EXPECTED_SITES = 6000
EXPECTED_CELLS = 22900


# ================================================================
# 6. OUTAGE SETTINGS
# ================================================================

# Synthetic base probability of a new cell outage.
BASE_OUTAGE_PROBABILITY = 0.00020

# Maximum number of newly generated outages in one hour.
MAX_NEW_OUTAGES_PER_HOUR = 15

# Outage duration.
MIN_OUTAGE_DURATION_MIN = 30
MAX_OUTAGE_DURATION_MIN = 240


# ================================================================
# 6B. v2 SIGNAL SETTINGS
# ================================================================

# Share of outages that are preceded by a degradation period.
# 0.0 = no warning signal (like version 1); try 0.25, 0.5, 0.75.
SIGNAL_STRENGTH = 0.5

# How strongly degradation changes the KPIs (1.0 = default effect).
DEGRADE_EFFECT = 1.0

# Length of the degradation period before the outage, in hours.
DEGRADE_HOURS_MIN = 6
DEGRADE_HOURS_MAX = 24

# Share of degradation periods that recover WITHOUT an outage.
DEGRADE_FALSE_ALARM_SHARE = 0.30

# Extra degradation risk for overloaded cells:
# risk x (1 + GAIN * (utilization - 100) / 100) when utilization > 100%.
CONGESTION_HAZARD_GAIN = 1.0

# Share of sudden outages that are site-wide power failures.
SITE_POWER_SHARE = 0.30

# Outage start time: anywhere in the 3-hour KPI interval (True)
# or only in the first hour, as in version 1 (False).
START_ANYWHERE_IN_INTERVAL = True

# Outage duration model: "lognormal" (long-tailed) or "uniform" (version 1).
DURATION_MODEL = "lognormal"
DURATION_MEDIAN_MIN = {
    "degradation": 120,   # hardware fault or overload: repair needed
    "sudden": 75,         # software crash, short fiber or link fault
    "site_power": 180,    # power failure: generator or grid restore
}
DURATION_SIGMA = 0.7
DURATION_MIN_MIN = 15
DURATION_MAX_MIN = 24 * 60

# Quick test: 500 random sites, January-June 2024 only.
QUICK_TEST = False
SITE_LIMIT = None

if QUICK_TEST:
    SITE_LIMIT = 500
    END_DATE = datetime(2024, 6, 30, 23, 59, 59)

# Write version 2 to its own folder so version 1 is not overwritten.
# Folder name includes the signal strength, e.g. ..._v2_s50 or ..._v2_s50_quick
OUTPUT_DIR = (
    OUTPUT_DIR
    + f"_v2_s{round(SIGNAL_STRENGTH * 100):02d}"
    + ("_quick" if QUICK_TEST else "")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 7. TECHNOLOGY PARAMETERS
# ================================================================
#
# Synthetic ranges used to create different KPI behavior
# for different mobile technologies.
#
# These are NOT claimed to be real operator measurements.
#
# ================================================================

TECH_PARAMS = {

    "2G": {

        "capacity_min": 12,
        "capacity_max": 25,

        "users_min": 20,
        "users_max": 90,

        "traffic_min": 0.5,
        "traffic_max": 3.0,

        "throughput_min": 0.15,
        "throughput_max": 0.80,

        "latency_min": 80,
        "latency_max": 160,

        "packet_loss_min": 0.20,
        "packet_loss_max": 1.50,

        "alarm_probability": 0.015,
    },


    "3G": {

        "capacity_min": 30,
        "capacity_max": 70,

        "users_min": 25,
        "users_max": 120,

        "traffic_min": 1.0,
        "traffic_max": 8.0,

        "throughput_min": 1.0,
        "throughput_max": 5.0,

        "latency_min": 45,
        "latency_max": 100,

        "packet_loss_min": 0.10,
        "packet_loss_max": 1.00,

        "alarm_probability": 0.012,
    },


    "4G": {

        "capacity_min": 80,
        "capacity_max": 180,

        "users_min": 40,
        "users_max": 250,

        "traffic_min": 4.0,
        "traffic_max": 30.0,

        "throughput_min": 10.0,
        "throughput_max": 80.0,

        "latency_min": 20,
        "latency_max": 55,

        "packet_loss_min": 0.05,
        "packet_loss_max": 0.70,

        "alarm_probability": 0.010,
    },


    "5G": {

        "capacity_min": 150,
        "capacity_max": 400,

        "users_min": 50,
        "users_max": 350,

        "traffic_min": 8.0,
        "traffic_max": 70.0,

        "throughput_min": 30.0,
        "throughput_max": 250.0,

        "latency_min": 8,
        "latency_max": 30,

        "packet_loss_min": 0.02,
        "packet_loss_max": 0.40,

        "alarm_probability": 0.008,
    },
}


# ================================================================
# 8. TECHNOLOGY OUTAGE FACTORS
# ================================================================

TECH_OUTAGE_FACTOR = {

    "2G": 1.05,
    "3G": 1.00,
    "4G": 1.00,
    "5G": 0.90,

}


# ================================================================
# 9. DAILY USAGE FACTOR
# ================================================================

def get_daily_factor(hour):

    # 00:00 - 05:59
    if 0 <= hour < 6:
        return 0.55

    # 06:00 - 08:59
    elif 6 <= hour < 9:
        return 0.80

    # 09:00 - 16:59
    elif 9 <= hour < 17:
        return 1.00

    # 17:00 - 21:59
    elif 17 <= hour < 22:
        return 1.20

    # 22:00 - 23:59
    else:
        return 0.85


# ================================================================
# 10. SEASONAL FACTOR
# ================================================================

def get_seasonal_factor(month):

    # Summer
    if month in [2, 3, 4, 5]:
        return 1.00

    # Rainy
    elif month in [6, 7, 8, 9]:
        return 1.12

    # Winter
    else:
        return 1.05


# ================================================================
# 11. CLIP FUNCTION
# ================================================================

def clip(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            value,
            maximum
        )
    )


# ================================================================
# 12. ALARM COUNT
# ================================================================

def generate_alarm_count(
    base_probability,
    extra_factor=1.0
):

    probability = (
        base_probability
        * extra_factor
    )

    probability = clip(
        probability,
        0.0,
        0.50
    )

    if random.random() < probability:

        count = np.random.poisson(
            1.2
        )

        return max(
            1,
            int(count)
        )

    return 0


# ================================================================
# 13. LOAD DATASET 1
# ================================================================

print()
print("=" * 70)
print("LOADING DATASET 1")
print("=" * 70)

df1 = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: {len(df1):,}"
)


# ================================================================
# 14. REQUIRED COLUMNS
# ================================================================

required_columns = [

    "Site_ID",
    "Site_Name",

    "Cell_ID",
    "Cell_Name",

    "Technology",

    "Region",
    "District",

    "Neighbor_Cell_ID",

]


missing_columns = [

    column

    for column in required_columns

    if column not in df1.columns

]


if missing_columns:

    raise ValueError(

        "Dataset 1 is missing these columns:\n"
        + "\n".join(
            missing_columns
        )

    )


# ================================================================
# 14B. v2 OPTIONAL SITE SAMPLE (QUICK_TEST)
# ================================================================

if SITE_LIMIT is not None:

    _site_rng = random.Random(SEED)  # separate generator: main sequence unchanged

    _keep_sites = sorted(
        _site_rng.sample(sorted(df1["Site_ID"].unique()), SITE_LIMIT)
    )

    df1 = df1[df1["Site_ID"].isin(_keep_sites)].reset_index(drop=True)

    print(f"QUICK TEST: using {SITE_LIMIT} sites ({len(df1):,} cells)")


# ================================================================
# 15. DATASET 1 VALIDATION
# ================================================================

site_count = (
    df1["Site_ID"]
    .nunique()
)

cell_count = (
    df1["Cell_ID"]
    .nunique()
)


print(
    f"Unique sites: {site_count:,}"
)

print(
    f"Unique cells: {cell_count:,}"
)


if SITE_LIMIT is None and site_count != EXPECTED_SITES:  # v2

    raise ValueError(

        f"Expected {EXPECTED_SITES:,} sites "
        f"but found {site_count:,}."

    )


if SITE_LIMIT is None and cell_count != EXPECTED_CELLS:  # v2

    raise ValueError(

        f"Expected {EXPECTED_CELLS:,} cells "
        f"but found {cell_count:,}."

    )


# ================================================================
# 16. CLEAN DATA TYPES
# ================================================================

string_columns = [

    "Site_ID",
    "Site_Name",

    "Cell_ID",
    "Cell_Name",

    "Technology",

    "Region",
    "District",

]


for column in string_columns:

    df1[column] = (

        df1[column]
        .fillna("")
        .astype(str)

    )


df1["Neighbor_Cell_ID"] = (

    df1["Neighbor_Cell_ID"]
    .fillna("")
    .astype(str)

)


# ================================================================
# 17. CELL LIST
# ================================================================

cell_ids = (
    df1["Cell_ID"]
    .tolist()
)

valid_cell_ids = set(
    cell_ids
)


# ================================================================
# 18. CELL INFORMATION LOOKUP
# ================================================================

cell_info = {}


for row in df1.itertuples(
    index=False
):

    cell_info[
        row.Cell_ID
    ] = {

        "Site_ID":
            row.Site_ID,

        "Site_Name":
            row.Site_Name,

        "Cell_Name":
            row.Cell_Name,

        "Technology":
            row.Technology,

        "Region":
            row.Region,

        "District":
            row.District,

    }


# ================================================================
# 19. BUILD NEIGHBOR MAP
# ================================================================
#
# Your actual Dataset 1 has:
#
# Neighbor_Cell_ID =
#
# UG-MBR-0212-S03,UG-MBR-1255-S03,...
#
# Therefore we use:
#
# split(",")
#
# ================================================================

print()
print("=" * 70)
print("BUILDING NEIGHBOR MAP")
print("=" * 70)


neighbor_map = {}


for row in df1.itertuples(
    index=False
):

    cell_id = row.Cell_ID

    own_district = row.District

    neighbors = []


    raw_neighbors = (
        row.Neighbor_Cell_ID
    )


    if raw_neighbors:

        parts = (
            raw_neighbors
            .split(",")
        )


        for neighbor in parts:

            neighbor = (
                neighbor.strip()
            )


            if not neighbor:
                continue


            # No self-neighbor.
            if neighbor == cell_id:
                continue


            # Neighbor must exist.
            if neighbor not in valid_cell_ids:
                continue


            # ----------------------------------------------------
            # FINAL RULE:
            #
            # External neighbors must be in the SAME DISTRICT.
            # ----------------------------------------------------

            neighbor_district = (
                cell_info[
                    neighbor
                ]["District"]
            )


            if (
                neighbor_district
                != own_district
            ):

                continue


            # No duplicate neighbors.
            if neighbor not in neighbors:

                neighbors.append(
                    neighbor
                )


    neighbor_map[
        cell_id
    ] = neighbors


# ================================================================
# 20. VALIDATE NEIGHBOR MAP
# ================================================================

print(
    "Validating neighbor relationships..."
)


invalid_neighbors = 0

cross_district_neighbors = 0

self_neighbors = 0

duplicate_neighbors = 0


for cell_id, neighbors in (
    neighbor_map.items()
):

    own_district = (
        cell_info[
            cell_id
        ]["District"]
    )


    if (
        len(neighbors)
        != len(set(neighbors))
    ):

        duplicate_neighbors += 1


    for neighbor in neighbors:

        if neighbor == cell_id:

            self_neighbors += 1
            invalid_neighbors += 1


        elif neighbor not in valid_cell_ids:

            invalid_neighbors += 1


        elif (
            cell_info[
                neighbor
            ]["District"]
            != own_district
        ):

            cross_district_neighbors += 1
            invalid_neighbors += 1


if invalid_neighbors > 0:

    raise ValueError(

        "Neighbor validation failed.\n"
        f"Invalid: {invalid_neighbors:,}\n"
        f"Cross-district: {cross_district_neighbors:,}\n"
        f"Self-neighbor: {self_neighbors:,}\n"
        f"Duplicates: {duplicate_neighbors:,}"

    )


neighbor_counts = [

    len(
        neighbor_map[
            cell_id
        ]
    )

    for cell_id in cell_ids

]


print(
    f"Minimum neighbors: "
    f"{min(neighbor_counts)}"
)

print(
    f"Maximum neighbors: "
    f"{max(neighbor_counts)}"
)

print(
    f"Average neighbors: "
    f"{np.mean(neighbor_counts):.2f}"
)


# ================================================================
# 21. CREATE CELL BASELINE PARAMETERS
# ================================================================

print()
print("=" * 70)
print("CREATING CELL BASELINE PARAMETERS")
print("=" * 70)


cell_params = {}


for row in df1.itertuples(
    index=False
):

    technology = row.Technology


    if technology not in TECH_PARAMS:

        raise ValueError(

            f"Unknown technology "
            f"{technology} for "
            f"{row.Cell_ID}"

        )


    params = (
        TECH_PARAMS[
            technology
        ]
    )


    cell_params[
        row.Cell_ID
    ] = {

        "capacity":
            random.uniform(

                params[
                    "capacity_min"
                ],

                params[
                    "capacity_max"
                ]

            ),

        "users_base":
            random.uniform(

                params[
                    "users_min"
                ],

                params[
                    "users_max"
                ]

            ),

        "traffic_base":
            random.uniform(

                params[
                    "traffic_min"
                ],

                params[
                    "traffic_max"
                ]

            ),

        "throughput_base":
            random.uniform(

                params[
                    "throughput_min"
                ],

                params[
                    "throughput_max"
                ]

            ),

        "latency_base":
            random.uniform(

                params[
                    "latency_min"
                ],

                params[
                    "latency_max"
                ]

            ),

        "packet_loss_base":
            random.uniform(

                params[
                    "packet_loss_min"
                ],

                params[
                    "packet_loss_max"
                ]

            ),

        "alarm_probability":
            params[
                "alarm_probability"
            ],

    }


# ================================================================
# 22. HISTORICAL OUTAGE COUNT
# ================================================================
#
# Only COMPLETED historical outages are counted.
#
# Example:
#
# Before outage:
#     0
#
# OUTAGE:
#     0
#
# RECOVERY:
#     0
#
# Next KPI:
#     1
#
# ================================================================

previous_outage_count = {

    cell_id: 0

    for cell_id in cell_ids

}


# ================================================================
# 23. ACTIVE OUTAGE STATE
# ================================================================

active_outages = {}


# ================================================================
# 23B. v2 DEGRADATION STATE, SITE LOOKUP, CAUSE LOG
# ================================================================

# cell_id -> {"start": datetime, "end": datetime, "will_fail": bool}
degrading = {}

# Last utilization (%) of each cell, used for the congestion risk.
last_utilization = {}

# Site_ID -> list of its cells (for site-wide power failures).
site_cells = {}

for _cell in cell_ids:
    site_cells.setdefault(cell_info[_cell]["Site_ID"], []).append(_cell)

site_ids = sorted(site_cells)

# True cause of every outage (written to outage_causes.csv).
outage_causes = []


# ================================================================
# 24. OUTAGE PROBABILITY
# ================================================================

def calculate_outage_probability(
    timestamp,
    technology
):

    daily_factor = (
        get_daily_factor(
            timestamp.hour
        )
    )


    seasonal_factor = (
        get_seasonal_factor(
            timestamp.month
        )
    )


    technology_factor = (
        TECH_OUTAGE_FACTOR.get(
            technology,
            1.0
        )
    )


    probability = (

        BASE_OUTAGE_PROBABILITY

        * daily_factor

        * seasonal_factor

        * technology_factor

    )


    return probability


# ================================================================
# 25. GENERATE OUTAGE EVENTS  (v2: rewritten)
# ================================================================

def sample_duration_minutes(cause):

    if DURATION_MODEL == "uniform":

        return random.randint(MIN_OUTAGE_DURATION_MIN, MAX_OUTAGE_DURATION_MIN)

    minutes = DURATION_MEDIAN_MIN[cause] * math.exp(random.gauss(0.0, DURATION_SIGMA))

    return int(clip(minutes, DURATION_MIN_MIN, DURATION_MAX_MIN))


def random_time_between(window_start, window_end):

    total_seconds = int((window_end - window_start).total_seconds())

    if total_seconds <= 0:

        return None

    return window_start + timedelta(seconds=random.randint(0, total_seconds - 1))


def congestion_risk(cell_id):

    utilization = last_utilization.get(cell_id)

    if utilization is None or utilization <= 100:

        return 1.0

    return 1.0 + CONGESTION_HAZARD_GAIN * (utilization - 100.0) / 100.0


def degradation_severity(cell_id, timestamp):

    # 0 = healthy, rising linearly to 1 at the end of the degradation period.

    state = degrading.get(cell_id)

    if state is None or timestamp < state["start"]:

        return 0.0

    span = (state["end"] - state["start"]).total_seconds()

    if span <= 0:

        return 1.0

    return clip((timestamp - state["start"]).total_seconds() / span, 0.0, 1.0)


def generate_outage_events(
    window_start,
    window_end
):

    events = []

    taken = set()

    # ------------------------------------------------------------
    # 1. Degradation periods that end in this window
    # ------------------------------------------------------------

    for cell_id, state in list(degrading.items()):

        if state["end"] >= window_end:

            continue

        del degrading[cell_id]

        if (
            state["will_fail"]
            and state["end"] >= window_start
            and cell_id not in active_outages
        ):

            outage_time = state["end"]

            events.append((
                cell_id,
                outage_time,
                outage_time + timedelta(minutes=sample_duration_minutes("degradation")),
                "degradation",
            ))

            taken.add(cell_id)

    # ------------------------------------------------------------
    # 2. New degradation periods (the warning signal)
    # ------------------------------------------------------------

    available_cells = [
        cell_id
        for cell_id in cell_ids
        if cell_id not in active_outages
        and cell_id not in degrading
        and cell_id not in taken
    ]

    for cell_id in available_cells:

        probability = (
            calculate_outage_probability(window_start, cell_info[cell_id]["Technology"])
            * SIGNAL_STRENGTH
            * congestion_risk(cell_id)
            / (1.0 - DEGRADE_FALSE_ALARM_SHARE)
        )

        if random.random() < probability:

            hours = random.uniform(DEGRADE_HOURS_MIN, DEGRADE_HOURS_MAX)

            degrading[cell_id] = {
                "start": window_start,
                "end": window_start + timedelta(hours=hours),
                "will_fail": random.random() >= DEGRADE_FALSE_ALARM_SHARE,
            }

    # ------------------------------------------------------------
    # 3. Sudden single-cell outages (no warning)
    # ------------------------------------------------------------

    sudden_cells = [
        cell_id
        for cell_id in cell_ids
        if cell_id not in active_outages and cell_id not in taken
    ]

    random.shuffle(sudden_cells)

    for cell_id in sudden_cells:

        probability = (
            calculate_outage_probability(window_start, cell_info[cell_id]["Technology"])
            * (1.0 - SIGNAL_STRENGTH)
            * (1.0 - SITE_POWER_SHARE)
        )

        if random.random() < probability:

            outage_time = random_time_between(window_start, window_end)

            if outage_time is None:

                continue

            events.append((
                cell_id,
                outage_time,
                outage_time + timedelta(minutes=sample_duration_minutes("sudden")),
                "sudden",
            ))

            taken.add(cell_id)

            degrading.pop(cell_id, None)

            if len(events) >= MAX_NEW_OUTAGES_PER_HOUR:

                return events

    # ------------------------------------------------------------
    # 4. Site-wide power failures (all cells of a site, no warning)
    #    One draw per site, scaled so the share of power-failure
    #    EVENTS matches SITE_POWER_SHARE.
    # ------------------------------------------------------------

    for site_id in site_ids:

        cells = [
            cell_id
            for cell_id in site_cells[site_id]
            if cell_id not in active_outages and cell_id not in taken
        ]

        if not cells:

            continue

        probability = (
            calculate_outage_probability(window_start, "3G")
            * (1.0 - SIGNAL_STRENGTH)
            * SITE_POWER_SHARE
        )

        if random.random() < probability:

            outage_time = random_time_between(window_start, window_end)

            if outage_time is None:

                continue

            recovery_time = outage_time + timedelta(minutes=sample_duration_minutes("site_power"))

            for cell_id in cells:

                events.append((cell_id, outage_time, recovery_time, "site_power"))

                taken.add(cell_id)

                degrading.pop(cell_id, None)

    return events


# ================================================================
# 26. CHECK WHETHER CELL IS DOWN
# ================================================================

def is_cell_down(
    cell_id,
    timestamp
):

    if cell_id not in active_outages:

        return False


    outage = (
        active_outages[
            cell_id
        ]
    )


    return (

        outage[
            "down_time"
        ]

        <= timestamp

        < outage[
            "up_time"
        ]

    )


# ================================================================
# 27. CONGESTION EFFECT
# ================================================================

def congestion_effect(
    utilization
):

    # ------------------------------------------------------------
    # <= 70%
    # ------------------------------------------------------------

    if utilization <= 70:

        return {

            "throughput_factor":
                1.00,

            "latency_multiplier":
                1.00,

            "packet_loss_add":
                0.00,

            "alarm_factor":
                1.00,

        }


    # ------------------------------------------------------------
    # 70-90%
    # ------------------------------------------------------------

    elif utilization <= 90:

        x = (
            utilization - 70
        ) / 20


        return {

            "throughput_factor":
                1.00
                - 0.10 * x,

            "latency_multiplier":
                1.00
                + 0.25 * x,

            "packet_loss_add":
                0.15 * x,

            "alarm_factor":
                1.20,

        }


    # ------------------------------------------------------------
    # 90-100%
    # ------------------------------------------------------------

    elif utilization <= 100:

        x = (
            utilization - 90
        ) / 10


        return {

            "throughput_factor":
                0.90
                - 0.15 * x,

            "latency_multiplier":
                1.25
                + 0.35 * x,

            "packet_loss_add":
                0.15
                + 0.35 * x,

            "alarm_factor":
                1.50,

        }


    # ------------------------------------------------------------
    # >100%
    # ------------------------------------------------------------

    else:

        overload = min(

            (
                utilization - 100
            ) / 50,

            2.0

        )


        return {

            "throughput_factor":
                max(

                    0.75
                    - 0.20 * overload,

                    0.25

                ),

            "latency_multiplier":
                1.60
                + 0.80 * overload,

            "packet_loss_add":
                0.50
                + 0.80 * overload,

            "alarm_factor":
                2.00,

        }


# ================================================================
# 28. GENERATE BASE KPI
# ================================================================

def generate_base_kpi(
    cell_id,
    timestamp
):

    params = (
        cell_params[
            cell_id
        ]
    )


    daily_factor = (
        get_daily_factor(
            timestamp.hour
        )
    )


    seasonal_factor = (
        get_seasonal_factor(
            timestamp.month
        )
    )


    # ------------------------------------------------------------
    # Connected Users
    # ------------------------------------------------------------

    users = (

        params[
            "users_base"
        ]

        * daily_factor

        * seasonal_factor

    )


    users *= np.random.normal(
        1.0,
        0.08
    )


    users = max(
        users,
        1.0
    )


    # ------------------------------------------------------------
    # Traffic
    # ------------------------------------------------------------

    traffic = (

        params[
            "traffic_base"
        ]

        * daily_factor

        * seasonal_factor

    )


    traffic *= np.random.normal(
        1.0,
        0.10
    )


    traffic = max(
        traffic,
        0.01
    )


    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    load_ratio = (

        0.40

        + 0.003 * users

        + 0.02 * traffic

    )


    load = (

        params[
            "capacity"
        ]

        * load_ratio

    )


    load *= np.random.normal(
        1.0,
        0.07
    )


    load = max(
        load,
        0.1
    )


    return {

        "Traffic":
            traffic,

        "Connected_Users":
            users,

        "Load":
            load,

    }


# ================================================================
# 29. CALCULATE NEIGHBOR IMPACT
# ================================================================

def calculate_neighbor_impact(
    base_metrics,
    down_cells
):

    impact = {}


    for down_cell in down_cells:

        neighbors = (
            neighbor_map[
                down_cell
            ]
        )


        if not neighbors:

            continue


        source = (
            base_metrics.get(
                down_cell
            )
        )


        if source is None:

            continue


        # Only available neighboring cells
        # can receive redistributed demand.

        available_neighbors = [

            neighbor

            for neighbor in neighbors

            if neighbor
            not in down_cells

        ]


        if not available_neighbors:

            continue


        # --------------------------------------------------------
        # Random distribution weights
        # --------------------------------------------------------

        raw_weights = np.random.uniform(

            0.5,
            1.5,

            size=len(
                available_neighbors
            )

        )


        weights = (

            raw_weights
            / raw_weights.sum()

        )


        # 70-95% of demand is redistributed.
        redistribution_factor = (
            random.uniform(
                0.70,
                0.95
            )
        )


        for neighbor, weight in zip(

            available_neighbors,
            weights

        ):

            if neighbor not in impact:

                impact[
                    neighbor
                ] = {

                    "users_add":
                        0.0,

                    "traffic_add":
                        0.0,

                    "load_add":
                        0.0,

                }


            impact[
                neighbor
            ][
                "users_add"
            ] += (

                source[
                    "Connected_Users"
                ]

                * redistribution_factor

                * weight

            )


            impact[
                neighbor
            ][
                "traffic_add"
            ] += (

                source[
                    "Traffic"
                ]

                * redistribution_factor

                * weight

            )


            impact[
                neighbor
            ][
                "load_add"
            ] += (

                source[
                    "Load"
                ]

                * redistribution_factor

                * weight

            )


    return impact


# ================================================================
# 30. GENERATE FINAL KPI
# ================================================================

def generate_final_kpi(
    cell_id,
    timestamp,
    neighbor_impact
):

    params = (
        cell_params[
            cell_id
        ]
    )


    base = (
        generate_base_kpi(
            cell_id,
            timestamp
        )
    )


    users = (
        base[
            "Connected_Users"
        ]
    )


    traffic = (
        base[
            "Traffic"
        ]
    )


    load = (
        base[
            "Load"
        ]
    )


    # ------------------------------------------------------------
    # Add neighbor outage impact
    # ------------------------------------------------------------

    if cell_id in neighbor_impact:

        impact = (
            neighbor_impact[
                cell_id
            ]
        )


        users += (
            impact[
                "users_add"
            ]
        )


        traffic += (
            impact[
                "traffic_add"
            ]
        )


        load += (
            impact[
                "load_add"
            ]
        )


    # ------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------

    capacity = (
        params[
            "capacity"
        ]
    )


    # ------------------------------------------------------------
    # Utilization
    #
    # Derived:
    #
    # Load / Capacity * 100
    #
    # It is intentionally NOT stored in Dataset 2.
    # ------------------------------------------------------------

    utilization = (

        load
        / capacity
        * 100

    )


    # ------------------------------------------------------------
    # Congestion
    # ------------------------------------------------------------

    congestion = (
        congestion_effect(
            utilization
        )
    )


    # ------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------

    daily_factor = (
        get_daily_factor(
            timestamp.hour
        )
    )


    throughput = (

        params[
            "throughput_base"
        ]

        * daily_factor

        * congestion[
            "throughput_factor"
        ]

    )


    # Additional congestion caused by neighbor outage.
    if cell_id in neighbor_impact:

        added_load = (
            neighbor_impact[
                cell_id
            ][
                "load_add"
            ]
        )


        if capacity > 0:

            impact_ratio = (
                added_load
                / capacity
            )


            throughput *= max(

                0.70,

                1.0
                - 0.20
                * impact_ratio

            )


    throughput *= np.random.normal(
        1.0,
        0.08
    )


    throughput = max(
        throughput,
        0.01
    )


    # ------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------

    latency = (

        params[
            "latency_base"
        ]

        * congestion[
            "latency_multiplier"
        ]

    )


    if cell_id in neighbor_impact:

        latency *= 1.08


    latency *= np.random.normal(
        1.0,
        0.06
    )


    latency = max(
        latency,
        1.0
    )


    # ------------------------------------------------------------
    # Packet Loss
    # ------------------------------------------------------------

    packet_loss = (

        params[
            "packet_loss_base"
        ]

        + congestion[
            "packet_loss_add"
        ]

    )


    if cell_id in neighbor_impact:

        packet_loss += 0.05


    packet_loss *= np.random.normal(
        1.0,
        0.10
    )


    packet_loss = max(
        packet_loss,
        0.0
    )


    # ------------------------------------------------------------
    # Alarm Count
    # ------------------------------------------------------------

    alarm_factor = (
        congestion[
            "alarm_factor"
        ]
    )


    if cell_id in neighbor_impact:

        alarm_factor *= 1.15


    # ------------------------------------------------------------
    # v2: degradation before an outage (the warning signal)
    # ------------------------------------------------------------

    severity = degradation_severity(cell_id, timestamp) * DEGRADE_EFFECT

    if severity > 0:

        throughput = max(throughput * max(1.0 - 0.35 * severity, 0.10), 0.01)

        latency *= 1.0 + 0.40 * severity

        packet_loss = packet_loss * (1.0 + 1.0 * severity) + 0.20 * severity

        alarm_factor *= 1.0 + 20.0 * severity

    last_utilization[cell_id] = utilization  # v2: for congestion risk


    alarm_count = (
        generate_alarm_count(

            params[
                "alarm_probability"
            ],

            alarm_factor

        )
    )


    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------

    return {

        "Traffic":
            round(
                traffic,
                3
            ),

        "Connected_Users":
            int(
                round(
                    users
                )
            ),

        "Throughput":
            round(
                throughput,
                3
            ),

        "Latency":
            round(
                latency,
                3
            ),

        "Packet_Loss":
            round(
                packet_loss,
                3
            ),

        "Load":
            round(
                load,
                3
            ),

        "Capacity":
            round(
                capacity,
                3
            ),

        "Alarm_Count":
            int(
                alarm_count
            ),

    }


# ================================================================
# 31. OUTPUT COLUMNS
# ================================================================

OUTPUT_COLUMNS = [

    "Timestamp",

    "Site_ID",
    "Cell_ID",

    "Record_Type",
    "Status",

    "Traffic",
    "Connected_Users",

    "Throughput",
    "Latency",
    "Packet_Loss",

    "Load",
    "Capacity",

    "Alarm_Count",

    "Previous_Outage_Count",

    "Cell_Down_Time",
    "Cell_Up_Time",

]


# ================================================================
# 32. WRITE DAILY RECORDS
# ================================================================

def write_daily_records(
    records,
    output_file
):

    if not records:

        return


    df_day = pd.DataFrame(
        records,
        columns=OUTPUT_COLUMNS
    )


    # ------------------------------------------------------------
    # Datetime conversion
    # ------------------------------------------------------------

    df_day["Timestamp"] = pd.to_datetime(
        df_day["Timestamp"]
    )


    df_day["Cell_Down_Time"] = pd.to_datetime(
        df_day["Cell_Down_Time"]
    )


    df_day["Cell_Up_Time"] = pd.to_datetime(
        df_day["Cell_Up_Time"]
    )


    # ------------------------------------------------------------
    # Randomize rows within identical timestamp
    #
    # Timestamp order remains chronological.
    # ------------------------------------------------------------

    df_day["_random_order"] = np.random.random(
        len(df_day)
    )


    df_day = (
        df_day
        .sort_values(
            by=[
                "Timestamp",
                "_random_order"
            ],
            kind="mergesort"
        )
    )


    df_day.drop(
        columns=[
            "_random_order"
        ],
        inplace=True
    )


    # ------------------------------------------------------------
    # Append to monthly file
    # ------------------------------------------------------------

    file_exists = os.path.exists(
        output_file
    )


    df_day.to_csv(

        output_file,

        mode="a",

        header=not file_exists,

        index=False

    )


# ================================================================
# 33. PROCESS ONE DAY
# ================================================================

def process_day(
    day_start,
    day_end,
    output_file
):

    daily_records = []


    timestamp = day_start


    while timestamp <= day_end:

        # ========================================================
        # 1. PROCESS RECOVERIES
        # ========================================================

        recovery_cells = []


        for cell_id, outage in list(
            active_outages.items()
        ):

            recovery_time = (
                outage[
                    "up_time"
                ]
            )


            if (
                recovery_time
                <= timestamp
            ):

                recovery_cells.append(
                    cell_id
                )


        # --------------------------------------------------------
        # Write recovery records
        # --------------------------------------------------------

        for cell_id in recovery_cells:

            outage = (
                active_outages[
                    cell_id
                ]
            )


            down_time = (
                outage[
                    "down_time"
                ]
            )


            recovery_time = (
                outage[
                    "up_time"
                ]
            )


            daily_records.append({

                "Timestamp":
                    recovery_time,

                "Site_ID":
                    cell_info[
                        cell_id
                    ][
                        "Site_ID"
                    ],

                "Cell_ID":
                    cell_id,

                "Record_Type":
                    "RECOVERY",

                "Status":
                    "UP",

                "Traffic":
                    np.nan,

                "Connected_Users":
                    np.nan,

                "Throughput":
                    np.nan,

                "Latency":
                    np.nan,

                "Packet_Loss":
                    np.nan,

                "Load":
                    np.nan,

                "Capacity":
                    np.nan,

                "Alarm_Count":
                    np.nan,

                "Previous_Outage_Count":
                    previous_outage_count[
                        cell_id
                    ],

                "Cell_Down_Time":
                    down_time,

                "Cell_Up_Time":
                    recovery_time,

            })


            # ----------------------------------------------------
            # Increase historical outage count AFTER recovery.
            # ----------------------------------------------------

            previous_outage_count[
                cell_id
            ] += 1


            del active_outages[
                cell_id
            ]


        # ========================================================
        # 2. GENERATE NEW OUTAGES
        # ========================================================

        hour_start = timestamp


        hour_end = (
            timestamp
            + timedelta(
                hours=KPI_INTERVAL_HOURS if START_ANYWHERE_IN_INTERVAL else 1  # v2
            )
        )


        if hour_end > day_end:

            hour_end = day_end


        new_outages = (
            generate_outage_events(

                hour_start,

                hour_end

            )
        )


        # --------------------------------------------------------
        # Add newly failed cells to active outage state.
        # --------------------------------------------------------

        for (

            cell_id,
            outage_time,
            recovery_time,
            cause  # v2

        ) in new_outages:

            if cell_id in active_outages:

                continue


            active_outages[
                cell_id
            ] = {

                "down_time":
                    outage_time,

                "up_time":
                    recovery_time,

            }

            # v2: remember the true cause (evaluation only).
            outage_causes.append({
                "Site_ID": cell_info[cell_id]["Site_ID"],
                "Cell_ID": cell_id,
                "Cell_Down_Time": outage_time,
                "Cell_Up_Time": recovery_time,
                "Cause": cause,
            })


        # ========================================================
        # 3. FIND CELLS CURRENTLY DOWN
        # ========================================================

        down_cells = []


        for cell_id in cell_ids:

            if is_cell_down(

                cell_id,

                timestamp

            ):

                down_cells.append(
                    cell_id
                )


        down_cell_set = set(
            down_cells
        )


        # ========================================================
        # 4. GENERATE BASE KPI
        # ========================================================

        base_metrics = {}


        # Available cells.
        for cell_id in cell_ids:

            if cell_id in down_cell_set:

                continue


            base_metrics[
                cell_id
            ] = generate_base_kpi(

                cell_id,

                timestamp

            )


        # --------------------------------------------------------
        # Generate hypothetical baseline demand for down cells.
        #
        # These values are ONLY used to calculate the demand
        # redistributed to neighbors.
        #
        # They are NOT written to Dataset 2.
        # --------------------------------------------------------

        for cell_id in down_cells:

            base_metrics[
                cell_id
            ] = generate_base_kpi(

                cell_id,

                timestamp

            )


        # ========================================================
        # 5. CALCULATE OUTAGE IMPACT ON NEIGHBORS
        # ========================================================

        neighbor_impact = (
            calculate_neighbor_impact(

                base_metrics,

                down_cells

            )
        )


        # ========================================================
        # 6. WRITE NORMAL KPI RECORDS
        # ========================================================

        for cell_id in cell_ids:

            # ----------------------------------------------------
            # No KPI while cell is down.
            # ----------------------------------------------------

            if cell_id in down_cell_set:

                continue


            metrics = (
                generate_final_kpi(

                    cell_id,

                    timestamp,

                    neighbor_impact

                )
            )


            daily_records.append({

                "Timestamp":
                    timestamp,

                "Site_ID":
                    cell_info[
                        cell_id
                    ][
                        "Site_ID"
                    ],

                "Cell_ID":
                    cell_id,

                "Record_Type":
                    "KPI",

                "Status":
                    "UP",

                "Traffic":
                    metrics[
                        "Traffic"
                    ],

                "Connected_Users":
                    metrics[
                        "Connected_Users"
                    ],

                "Throughput":
                    metrics[
                        "Throughput"
                    ],

                "Latency":
                    metrics[
                        "Latency"
                    ],

                "Packet_Loss":
                    metrics[
                        "Packet_Loss"
                    ],

                "Load":
                    metrics[
                        "Load"
                    ],

                "Capacity":
                    metrics[
                        "Capacity"
                    ],

                "Alarm_Count":
                    metrics[
                        "Alarm_Count"
                    ],

                "Previous_Outage_Count":
                    previous_outage_count[
                        cell_id
                    ],

                "Cell_Down_Time":
                    pd.NaT,

                "Cell_Up_Time":
                    pd.NaT,

            })


        # ========================================================
        # 7. WRITE OUTAGE RECORDS
        # ========================================================

        for (

            cell_id,
            outage_time,
            recovery_time,
            cause  # v2

        ) in new_outages:

            # Event must belong to this day.

            if not (

                day_start
                <= outage_time
                <= day_end

            ):

                continue


            daily_records.append({

                "Timestamp":
                    outage_time,

                "Site_ID":
                    cell_info[
                        cell_id
                    ][
                        "Site_ID"
                    ],

                "Cell_ID":
                    cell_id,

                "Record_Type":
                    "OUTAGE",

                "Status":
                    "DOWN",

                "Traffic":
                    np.nan,

                "Connected_Users":
                    np.nan,

                "Throughput":
                    np.nan,

                "Latency":
                    np.nan,

                "Packet_Loss":
                    np.nan,

                "Load":
                    np.nan,

                "Capacity":
                    np.nan,

                "Alarm_Count":
                    np.nan,

                # Current outage is NOT counted.
                "Previous_Outage_Count":
                    previous_outage_count[
                        cell_id
                    ],

                "Cell_Down_Time":
                    outage_time,

                "Cell_Up_Time":
                    pd.NaT,

            })


        # ========================================================
        # 8. NEXT KPI TIMESTAMP
        # ========================================================

        timestamp += timedelta(
            hours=KPI_INTERVAL_HOURS
        )


    # ============================================================
    # 9. WRITE DAY
    # ============================================================

    write_daily_records(

        daily_records,

        output_file

    )


    # ============================================================
    # 10. DAILY STATISTICS
    # ============================================================

    if not daily_records:

        return {

            "records": 0,
            "kpi": 0,
            "outage": 0,
            "recovery": 0,

        }


    record_types = [

        record[
            "Record_Type"
        ]

        for record in daily_records

    ]


    return {

        "records":
            len(daily_records),

        "kpi":
            record_types.count(
                "KPI"
            ),

        "outage":
            record_types.count(
                "OUTAGE"
            ),

        "recovery":
            record_types.count(
                "RECOVERY"
            ),

    }


# ================================================================
# 34. PROCESS ONE MONTH
# ================================================================

def process_month(
    year,
    month
):

    print()
    print("=" * 70)
    print(
        f"PROCESSING {year}-{month:02d}"
    )
    print("=" * 70)


    month_start = datetime(

        year,
        month,
        1,
        0,
        0,
        0

    )


    if month == 12:

        next_month = datetime(

            year + 1,
            1,
            1,
            0,
            0,
            0

        )

    else:

        next_month = datetime(

            year,
            month + 1,
            1,
            0,
            0,
            0

        )


    month_end = (
        next_month
        - timedelta(
            seconds=1
        )
    )


    # Respect overall simulation range.

    month_start = max(

        month_start,

        START_DATE

    )


    month_end = min(

        month_end,

        END_DATE

    )


    output_file = os.path.join(

        OUTPUT_DIR,

        f"{year}_{month:02d}.csv"

    )


    # ------------------------------------------------------------
    # Delete previous file if it exists.
    #
    # This makes re-running the script safe.
    # ------------------------------------------------------------

    if os.path.exists(
        output_file
    ):

        os.remove(
            output_file
        )


    # ------------------------------------------------------------
    # Month statistics
    # ------------------------------------------------------------

    month_records = 0
    month_kpi = 0
    month_outage = 0
    month_recovery = 0


    # ------------------------------------------------------------
    # Process each day
    # ------------------------------------------------------------

    day_start = month_start


    while day_start <= month_end:

        day_end = (

            day_start
            + timedelta(
                days=1
            )
            - timedelta(
                seconds=1
            )

        )


        if day_end > month_end:

            day_end = month_end


        print(

            f"  Day: "
            f"{day_start.strftime('%Y-%m-%d')}"

        )


        stats = process_day(

            day_start,

            day_end,

            output_file

        )


        month_records += (
            stats[
                "records"
            ]
        )


        month_kpi += (
            stats[
                "kpi"
            ]
        )


        month_outage += (
            stats[
                "outage"
            ]
        )


        month_recovery += (
            stats[
                "recovery"
            ]
        )


        day_start += timedelta(
            days=1
        )


    # ============================================================
    # MONTH SUMMARY
    # ============================================================

    print()
    print(
        f"Completed {year}-{month:02d}"
    )


    print(
        f"  Total records : "
        f"{month_records:,}"
    )


    print(
        f"  KPI records   : "
        f"{month_kpi:,}"
    )


    print(
        f"  OUTAGE records: "
        f"{month_outage:,}"
    )


    print(
        f"  RECOVERY      : "
        f"{month_recovery:,}"
    )


    print(
        f"  Active outages: "
        f"{len(active_outages):,}"
    )


    print(
        f"  Output file   : "
        f"{output_file}"
    )


# ================================================================
# 35. START
# ================================================================

print()
print("=" * 70)
print("UGANDA DATASET 2 GENERATION")
print("=" * 70)

print(
    f"Simulation start : {START_DATE}"
)

print(
    f"Simulation end   : {END_DATE}"
)

print(
    f"Sites            : {site_count:,}"
)

print(
    f"Cells            : {cell_count:,}"
)

print(
    f"KPI interval     : "
    f"{KPI_INTERVAL_HOURS} hours"
)

print()
print(
    "Generation mode:"
)

print(
    "  24 monthly CSV files"
)

print(
    "  Daily writing to reduce memory usage"
)

print()
print(
    "Output folder:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)


# ================================================================
# 36. GENERATE 24 MONTHS
# ================================================================

current_year = START_DATE.year
current_month = START_DATE.month


while True:

    current_month_start = datetime(

        current_year,

        current_month,

        1,

        0,
        0,
        0

    )


    if current_month_start > END_DATE:

        break


    process_month(

        current_year,

        current_month

    )


    # Move to next month.

    if current_month == 12:

        current_year += 1

        current_month = 1

    else:

        current_month += 1


# ================================================================
# 37. FINAL FILE VALIDATION
# ================================================================

print()
print("=" * 70)
print("FINAL FILE VALIDATION")
print("=" * 70)


monthly_files = sorted(

    [

        filename

        for filename
        in os.listdir(
            OUTPUT_DIR
        )

        if filename.endswith(
            ".csv"
        )

    ]

)


print(
    f"CSV files found: "
    f"{len(monthly_files)}"
)


if len(monthly_files) == 24:

    print(
        "SUCCESS: 24 monthly files created."
    )

else:

    print(
        "WARNING: Expected 24 monthly files."
    )


print()
print("Generated files:")


for filename in monthly_files:

    print(
        f"  {filename}"
    )


# ================================================================
# 38. FINAL STATE
# ================================================================

print()
print("=" * 70)
print("FINAL SIMULATION STATE")
print("=" * 70)


print(
    f"Cells tracked: "
    f"{len(previous_outage_count):,}"
)


print(
    f"Active outages at end: "
    f"{len(active_outages):,}"
)


# v2: true outage causes, for evaluation only (NOT a model feature).
causes_file = os.path.join(os.path.dirname(OUTPUT_DIR), os.path.basename(OUTPUT_DIR) + "_outage_causes.csv")

pd.DataFrame(outage_causes).to_csv(causes_file, index=False)

print(f"Outage causes written to: {causes_file}")

print(pd.DataFrame(outage_causes)["Cause"].value_counts().to_string())


print()
print(
    "DATASET 2 GENERATION COMPLETED SUCCESSFULLY."
)

print()
print(
    "Output folder:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)