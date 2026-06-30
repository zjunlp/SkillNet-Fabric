---
name: financial-kpi-extractor
description: Extract revenue, margin, cash flow, and other KPIs from structured financial tables.
---

# financial-kpi-extractor

Use this skill after `pdf-table-parser` has produced CSV tables from a financial
report.

## Inputs

- CSV tables from a financial report.

## Outputs

- KPI JSON with normalized metric names, values, units, and source rows.

## Notes

This skill depends on clean structured tables.
