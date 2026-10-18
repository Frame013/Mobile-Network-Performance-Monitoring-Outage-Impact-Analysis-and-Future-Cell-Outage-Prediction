# Data

The data files are **not stored in this repository**. They are too large for GitHub (100 MB per-file limit), and all of them are synthetic and can be regenerated.

Expected files (in one folder; set this folder in the notebooks):

| File | Content | Size |
|---|---|---|
| `Dataset_1_Cell_Information_V3.csv` | 22,900 cells: Site_ID, Cell_ID, Technology, Region, District, Neighbor_Cell_ID | small |
| `Dataset_2_Marge.csv` | 133.9 M rows: KPI records every 3 h + OUTAGE / RECOVERY events, Jan 2024 – Dec 2025 | ~15 GB |

Dataset 2 columns: `Timestamp, Site_ID, Cell_ID, Record_Type, Status, Traffic, Connected_Users, Throughput, Latency, Packet_Loss, Load, Capacity, Alarm_Count, Previous_Outage_Count, Cell_Down_Time, Cell_Up_Time`

Files created while running the notebooks (also not stored in Git):

- `Dataset_2_Cell_KPI_Outage/KPI_Model_Data.csv` and `KPI_Feature_Data.csv` (Sections 6.3–6.4, ~62 GB)

To regenerate the data, see the generator scripts in `scripts/`.
