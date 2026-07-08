from __future__ import annotations

from skillfabric.registry.models import SkillNode


class FixtureInterfaceExtractor:
    """Explicit fixture contracts for tests that need graph edges."""

    model_id = "fixture-interface"

    def extract(self, skill: SkillNode) -> dict[str, object]:
        payload: dict[str, object] = {
            "capability_summary": skill.description,
            "when_to_use": skill.description,
            "requires": [],
            "produces": [],
            "uses_tools": [],
        }
        fields = {
            "skill:pdf-table-parser": {
                "produces": [("csv", "artifact", 0.95, 11, "This skill produces `.csv` files for downstream analysis.")],
                "uses_tools": [("pdfplumber", "tool", 0.9, 10, "Use pdfplumber to extract tables from `.pdf` documents.")],
            },
            "skill:financial-kpi-extractor": {
                "requires": [("csv", "artifact", 0.95, 6, "Use this after `pdf-table-parser` has produced `.csv` tables.")],
                "produces": [("json", "artifact", 0.95, 7, "Read revenue, margin, and cash flow values, then output `kpi.json`.")],
            },
            "skill:report-writer": {
                "requires": [("json", "artifact", 0.95, 6, "Use KPI JSON and chart artifacts to compose a final `.md` report.")],
                "produces": [("markdown", "artifact", 0.9, 6, "Use KPI JSON and chart artifacts to compose a final `.md` report.")],
            },
            "skill:webshop-product-search": {
                "produces": [("candidate_products", "data", 0.9, 6, "The search results feed downstream product evaluation.")],
            },
            "skill:webshop-product-evaluator": {
                "requires": [("candidate_products", "data", 0.9, 6, "Use after `webshop-product-search` returns candidates.")],
            },
        }.get(skill.id, {})
        for group, rows in fields.items():
            payload[group] = [
                {
                    "name": name,
                    "kind": kind,
                    "confidence": confidence,
                    "evidence": [{"skill": skill.id, "line": line, "text": text}],
                }
                for name, kind, confidence, line, text in rows
            ]
        return payload
