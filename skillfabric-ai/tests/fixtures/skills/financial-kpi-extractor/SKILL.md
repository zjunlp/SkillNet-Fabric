---
name: financial-kpi-extractor
description: Extract financial KPI values from CSV tables.
---

# Financial KPI Extractor

Use this after `pdf-table-parser` has produced `.csv` tables.
Read revenue, margin, and cash flow values, then output `kpi.json`.

## Workflow

1. Load CSV tables.
2. Extract financial metrics.
3. Write JSON metrics.
