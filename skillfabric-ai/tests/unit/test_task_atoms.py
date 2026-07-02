from __future__ import annotations

import json
import unittest

from skillfabric.router.atomizer import build_task_atomizer_messages, parse_task_atomizer_response
from skillfabric.router.task_atoms import (
    TASK_ATOMS_SCHEMA_VERSION,
    TaskAtom,
    TaskDecomposition,
    validate_task_decomposition,
)


class TaskAtomsTests(unittest.TestCase):
    def test_validate_task_decomposition_accepts_minimal_atoms(self) -> None:
        query = "Research Acme and Beta, capture homepage screenshots, and write competitor_report.docx."

        result = validate_task_decomposition(
            {
                "schema_version": TASK_ATOMS_SCHEMA_VERSION,
                "atoms": [
                    {
                        "id": "a1",
                        "kind": "action",
                        "text": "research Acme and Beta",
                        "evidence": "Research Acme and Beta",
                        "required": True,
                        "depends_on": [],
                    },
                    {
                        "id": "a2",
                        "kind": "artifact",
                        "text": "capture homepage screenshots",
                        "evidence": "capture homepage screenshots",
                        "required": True,
                        "depends_on": ["a1"],
                    },
                    {
                        "id": "a3",
                        "kind": "artifact",
                        "text": "write competitor_report.docx",
                        "evidence": "write competitor_report.docx",
                        "required": True,
                        "depends_on": ["a1", "a2"],
                    },
                ],
            },
            query=query,
        )

        self.assertEqual([atom.id for atom in result.atoms], ["a1", "a2", "a3"])
        self.assertEqual(result.atoms[1].depends_on, ["a1"])

    def test_validate_task_decomposition_rejects_old_router_fields_and_extra_keys(self) -> None:
        query = "Create a report."
        payload = {
            "schema_version": TASK_ATOMS_SCHEMA_VERSION,
            "atoms": [
                {
                    "id": "a1",
                    "kind": "action",
                    "text": "create report",
                    "evidence": "Create a report",
                    "required": True,
                    "depends_on": [],
                    "intent": "document_creation",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "forbidden field"):
            validate_task_decomposition(payload, query=query)

    def test_validate_task_decomposition_rejects_invalid_schema_values(self) -> None:
        valid = {
            "schema_version": TASK_ATOMS_SCHEMA_VERSION,
            "atoms": [
                {
                    "id": "a1",
                    "kind": "action",
                    "text": "create report",
                    "evidence": "Create a report",
                    "required": True,
                    "depends_on": [],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unknown task atom kind"):
            validate_task_decomposition(
                {**valid, "atoms": [{**valid["atoms"][0], "kind": "intent"}]},
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "evidence is required"):
            validate_task_decomposition(
                {**valid, "atoms": [{**valid["atoms"][0], "evidence": ""}]},
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "missing required field"):
            validate_task_decomposition(
                {"atoms": valid["atoms"]},
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "missing required field"):
            validate_task_decomposition(
                {
                    **valid,
                    "atoms": [
                        {
                            key: value
                            for key, value in valid["atoms"][0].items()
                            if key != "depends_on"
                        }
                    ],
                },
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "more than 12 atoms"):
            validate_task_decomposition(
                {**valid, "atoms": [{**valid["atoms"][0], "id": f"a{index}"} for index in range(13)]},
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "more than 12 atoms"):
            validate_task_decomposition(
                TaskDecomposition(
                    atoms=[
                        TaskAtom(
                            id=f"a{index}",
                            kind="action",
                            text="create report",
                            evidence="Create a report",
                        )
                        for index in range(13)
                    ]
                ),
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "depends on unknown"):
            validate_task_decomposition(
                {**valid, "atoms": [{**valid["atoms"][0], "depends_on": ["missing"]}]},
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "must not mention skill ids"):
            validate_task_decomposition(
                {**valid, "atoms": [{**valid["atoms"][0], "text": "use skill:docx"}]},
                query="Create a report.",
            )
        with self.assertRaisesRegex(ValueError, "not quoted from query"):
            validate_task_decomposition(valid, query="Different query.")

    def test_parse_task_atomizer_response_validates_json_payload(self) -> None:
        query = "Create index.html and summary.docx."
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "schema_version": TASK_ATOMS_SCHEMA_VERSION,
                                "atoms": [
                                    {
                                        "id": "a1",
                                        "kind": "artifact",
                                        "text": "create index.html",
                                        "evidence": "index.html",
                                        "required": True,
                                        "depends_on": [],
                                    },
                                    {
                                        "id": "a2",
                                        "kind": "artifact",
                                        "text": "create summary.docx",
                                        "evidence": "summary.docx",
                                        "required": True,
                                        "depends_on": [],
                                    },
                                ],
                            }
                        )
                    }
                }
            ]
        }

        result = parse_task_atomizer_response(response, query=query)

        self.assertEqual([atom.evidence for atom in result.atoms], ["index.html", "summary.docx"])

    def test_prompt_contract_forbids_skill_selection_and_old_labels(self) -> None:
        messages = build_task_atomizer_messages("Make a report.")
        text = "\n".join(message["content"] for message in messages)

        self.assertIn("You never select skills", text)
        self.assertIn("Do not output skill_id", text)
        self.assertIn("Do not map vague words to file formats", text)
        self.assertIn('"kind"', text)
        self.assertIn('"action"', text)
        self.assertIn('"artifact"', text)
        self.assertIn('"constraint"', text)

    def test_dataclass_payload_still_validates(self) -> None:
        result = validate_task_decomposition(
            TaskDecomposition(
                atoms=[
                    TaskAtom(
                        id="a1",
                        kind="constraint",
                        text="keep the final report under 2 pages",
                        evidence="under 2 pages",
                    )
                ]
            ),
            query="Write a report under 2 pages.",
        )

        self.assertEqual(result.atoms[0].kind, "constraint")


if __name__ == "__main__":
    unittest.main()
