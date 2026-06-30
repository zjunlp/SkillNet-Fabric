---
name: pdf-table-parser
description: Extract tables from PDF files and save structured CSV output.
allowed-tools:
  - Bash(python:*)
---

# PDF Table Parser

Use pdfplumber to extract tables from `.pdf` documents.
This skill produces `.csv` files for downstream analysis.

## Workflow

1. Parse PDF pages.
2. Extract tabular rows.
3. Validate column headers.
