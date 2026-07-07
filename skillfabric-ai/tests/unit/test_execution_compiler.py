from __future__ import annotations

import unittest

from skillfabric.compiled_graph.canonicalization.models import (
    CanonicalAssignment,
    CanonicalizationBuild,
    CanonicalObject,
    RejectedCanonicalTerm,
)
from skillfabric.compiled_graph.execution.compiler import (
    compile_execution_graph,
    execution_index_from_validation_records,
)
from skillfabric.compiled_graph.execution.models import (
    ExecutionEdge,
    ExecutionEvidence,
    ExecutionValidationRecord,
)
from skillfabric.compiled_graph.interface.models import (
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
)


def _interface(
    skill_id: str,
    *,
    requires: list[InterfaceField] | None = None,
    produces: list[InterfaceField] | None = None,
) -> SkillInterface:
    return SkillInterface(
        skill_id=skill_id,
        content_hash=f"hash-{skill_id}",
        capability_summary=f"{skill_id} summary",
        requires=requires or [],
        produces=produces or [],
    )


def _field(skill_id: str, name: str, kind: str) -> InterfaceField:
    return InterfaceField(
        name=name,
        kind=kind,
        confidence=0.9,
        evidence=[InterfaceEvidence(skill=skill_id, line=2, text=f"{skill_id} mentions {name}.")],
    )


def _build_canonicalization(canonical_id: str, assignments: list[tuple[str, str, InterfaceField]]) -> CanonicalizationBuild:
    object_type, _, name = canonical_id.partition(":")
    return CanonicalizationBuild(
        objects=[
            CanonicalObject(
                canonical_id=canonical_id,
                name=name,
                type=object_type,
                promoted=True,
                confidence=0.9,
            )
        ],
        assignments=[
            CanonicalAssignment(
                raw_key="|".join([skill_id, role, field.name.lower(), field.kind.lower()]),
                skill_id=skill_id,
                role=role,
                raw_name=field.name,
                raw_kind=field.kind,
                canonical_id=canonical_id,
                confidence=0.9,
            )
            for skill_id, role, field in assignments
        ],
    )


class ExecutionCompilerTests(unittest.TestCase):
    def test_without_canonicalization_assignments_emits_warning_and_no_candidates(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "DOCX document", "artifact")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "docx_file", "artifact")])

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
        )

        self.assertFalse(hasattr(compiled, "artifact_nodes"))
        self.assertFalse(hasattr(compiled, "skill_artifact_edges"))
        self.assertEqual(compiled.candidates, [])
        self.assertIn("canonicalization assignments", compiled.warnings[0])
        self.assertEqual(compiled.execution_index, [])

    def test_outputs_and_inputs_generate_candidate_from_canonical_assignments(self) -> None:
        produced = _field("skill:producer", "DOCX document", "artifact")
        consumed = _field("skill:consumer", "docx_file", "artifact")
        producer = _interface("skill:producer", produces=[produced])
        consumer = _interface("skill:consumer", requires=[consumed])
        canonicalization = _build_canonicalization(
            "artifact:docx_document",
            [
                ("skill:producer", "produces", produced),
                ("skill:consumer", "requires", consumed),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(len(compiled.candidates), 1)
        self.assertEqual(compiled.candidates[0].source_skill, "skill:producer")
        self.assertEqual(compiled.candidates[0].target_skill, "skill:consumer")
        self.assertEqual(compiled.candidates[0].flow_type, "artifact_flow")
        self.assertEqual(compiled.candidates[0].matched_name, "docx_document")
        self.assertEqual(compiled.execution_index, [])

    def test_broad_context_object_does_not_generate_execution_candidates(self) -> None:
        produced = _field("skill:producer", "source_data", "data")
        consumed = _field("skill:consumer", "source_data", "data")
        producer = _interface("skill:producer", produces=[produced])
        consumer = _interface("skill:consumer", requires=[consumed])
        canonicalization = _build_canonicalization(
            "data:source_data",
            [
                ("skill:producer", "produces", produced),
                ("skill:consumer", "requires", consumed),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(compiled.candidates, [])

    def test_duplicate_compatibility_candidates_are_merged_before_validation(self) -> None:
        producer = _interface(
            "skill:producer",
            produces=[
                _field("skill:producer", "PDF document", "artifact"),
                _field("skill:producer", "PDF file", "artifact"),
            ],
        )
        consumer = _interface(
            "skill:consumer",
            requires=[_field("skill:consumer", "PDF document", "artifact")],
        )
        canonicalization = _build_canonicalization(
            "artifact:pdf_document",
            [
                ("skill:producer", "produces", producer.produces[0]),
                ("skill:producer", "produces", producer.produces[1]),
                ("skill:consumer", "requires", consumer.requires[0]),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(len(compiled.candidates), 1)
        self.assertEqual(compiled.candidates[0].source_skill, "skill:producer")
        self.assertEqual(compiled.candidates[0].target_skill, "skill:consumer")
        self.assertEqual(compiled.candidates[0].matched_name, "pdf_document")
        self.assertEqual(len(compiled.candidates[0].evidence), 3)

    def test_duplicate_fields_do_not_trip_bucket_limit_for_one_pair(self) -> None:
        producer = _interface(
            "skill:producer",
            produces=[
                _field("skill:producer", "PDF document", "artifact"),
                _field("skill:producer", "PDF file", "artifact"),
                _field("skill:producer", "PDF report", "artifact"),
            ],
        )
        consumer = _interface(
            "skill:consumer",
            requires=[_field("skill:consumer", "PDF document", "artifact")],
        )
        canonicalization = _build_canonicalization(
            "artifact:pdf_document",
            [
                ("skill:producer", "produces", producer.produces[0]),
                ("skill:producer", "produces", producer.produces[1]),
                ("skill:producer", "produces", producer.produces[2]),
                ("skill:consumer", "requires", consumer.requires[0]),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=1,
            canonicalization=canonicalization,
        )

        self.assertEqual(len(compiled.candidates), 1)
        self.assertEqual(compiled.candidates[0].matched_name, "pdf_document")

    def test_reusable_postconditions_and_preconditions_generate_state_candidate(self) -> None:
        enabler = _interface(
            "skill:login",
            produces=[_field("skill:login", "authenticated session", "state")],
        )
        requiring = _interface(
            "skill:purchase",
            requires=[_field("skill:purchase", "authenticated session", "state")],
        )
        canonicalization = _build_canonicalization(
            "state:authenticated_session",
            [
                ("skill:login", "produces", enabler.produces[0]),
                ("skill:purchase", "requires", requiring.requires[0]),
            ],
        )

        compiled = compile_execution_graph(
            {enabler.skill_id: enabler, requiring.skill_id: requiring},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertFalse(hasattr(compiled, "scenario_nodes"))
        self.assertFalse(hasattr(compiled, "skill_scenario_edges"))
        self.assertEqual(compiled.candidates[0].flow_type, "scenario_transition")
        self.assertEqual(compiled.candidates[0].matched_name, "authenticated_session")
        self.assertEqual(compiled.execution_index, [])

    def test_belief_state_does_not_generate_world_state_execution_candidate(self) -> None:
        planner = _interface(
            "skill:goal-interpreter",
            produces=[_field("skill:goal-interpreter", "object_permanence_state", "belief_state")],
        )
        cleaner = _interface(
            "skill:clean-object",
            requires=[_field("skill:clean-object", "object_in_inventory", "world_state")],
        )

        compiled = compile_execution_graph(
            {planner.skill_id: planner, cleaner.skill_id: cleaner},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])
        self.assertTrue(any(node.kind == "belief_state" for node in compiled.raw_scenario_nodes))

    def test_execution_index_uses_only_accepted_validation_records(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "DOCX document", "artifact")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "docx_file", "artifact")])
        canonicalization = _build_canonicalization(
            "artifact:docx_document",
            [
                ("skill:producer", "produces", producer.produces[0]),
                ("skill:consumer", "requires", consumer.requires[0]),
            ],
        )
        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )
        candidate = compiled.candidates[0]
        accepted = ExecutionValidationRecord(
            candidate=candidate,
            raw_output={"accepted": True},
            normalized={
                "accepted": True,
                "flow_type": "artifact_flow",
                "projected_edge_type": "depend_on",
                "confidence": 0.91,
                "reason": "Producer emits a DOCX document consumed downstream.",
            },
            accepted=True,
            rejection_reason="",
            flow_edge=ExecutionEdge(
                source=candidate.source_skill,
                target=candidate.target_skill,
                type="artifact_flow",
                confidence=0.91,
                evidence=candidate.evidence,
                reason="Producer emits a DOCX document consumed downstream.",
                metadata=candidate.metadata,
            ),
        )
        rejected = ExecutionValidationRecord(
            candidate=candidate,
            raw_output={"accepted": False},
            normalized={"accepted": False},
            accepted=False,
            rejection_reason="accepted is false",
        )

        records = execution_index_from_validation_records([accepted, rejected])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].relation_type, "artifact_compatibility")
        self.assertEqual(records[0].canonical_object, "docx_document")
        self.assertEqual(records[0].projected_edge_type, "depend_on")

    def test_execution_index_deduplicates_same_compatibility_record(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "DOCX document", "artifact")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "docx_file", "artifact")])
        canonicalization = _build_canonicalization(
            "artifact:docx_document",
            [
                ("skill:producer", "produces", producer.produces[0]),
                ("skill:consumer", "requires", consumer.requires[0]),
            ],
        )
        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )
        candidate = compiled.candidates[0]
        first = ExecutionValidationRecord(
            candidate=candidate,
            raw_output={"accepted": True},
            normalized={
                "accepted": True,
                "flow_type": "artifact_flow",
                "projected_edge_type": "depend_on",
                "confidence": 0.91,
                "reason": "Producer emits a DOCX document.",
            },
            accepted=True,
            rejection_reason="",
            flow_edge=ExecutionEdge(
                source=candidate.source_skill,
                target=candidate.target_skill,
                type="artifact_flow",
                confidence=0.91,
                evidence=candidate.evidence,
                reason="Producer emits a DOCX document.",
                metadata=candidate.metadata,
            ),
        )
        second = ExecutionValidationRecord(
            candidate=candidate,
            raw_output={"accepted": True},
            normalized={
                "accepted": True,
                "flow_type": "artifact_flow",
                "projected_edge_type": "depend_on",
                "confidence": 0.96,
                "reason": "Consumer needs that DOCX document downstream.",
            },
            accepted=True,
            rejection_reason="",
            flow_edge=ExecutionEdge(
                source=candidate.source_skill,
                target=candidate.target_skill,
                type="artifact_flow",
                confidence=0.96,
                evidence=[
                    *candidate.evidence,
                    ExecutionEvidence(skill="skill:consumer", line=8, text="Consumes the produced DOCX."),
                ],
                reason="Consumer needs that DOCX document downstream.",
                metadata=candidate.metadata,
            ),
        )

        records = execution_index_from_validation_records([first, second])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].confidence, 0.96)
        self.assertIn("Producer emits a DOCX document.", records[0].reason)
        self.assertIn("Consumer needs that DOCX document downstream.", records[0].reason)
        self.assertEqual(len(records[0].evidence), len({item.key for item in records[0].evidence}))

    def test_local_conditions_do_not_generate_global_scenario_records(self) -> None:
        only_enabler = _interface(
            "skill:algorithmic-art",
            produces=[_field("skill:algorithmic-art", "algorithmic philosophy created", "state")],
        )

        compiled = compile_execution_graph({only_enabler.skill_id: only_enabler})

        self.assertFalse(hasattr(compiled, "scenario_nodes"))
        self.assertFalse(hasattr(compiled, "skill_scenario_edges"))
        self.assertEqual(compiled.candidates, [])
        self.assertEqual(compiled.execution_index, [])

    def test_large_candidate_bucket_is_skipped(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "shared artifact", "artifact")])
        consumers = {
            f"skill:consumer-{index}": _interface(
                f"skill:consumer-{index}",
                requires=[_field(f"skill:consumer-{index}", "shared artifact", "artifact")],
            )
            for index in range(3)
        }
        interfaces = {producer.skill_id: producer, **consumers}
        canonicalization = _build_canonicalization(
            "artifact:shared_artifact",
            [
                ("skill:producer", "produces", producer.produces[0]),
                *[
                    (skill_id, "requires", interface.requires[0])
                    for skill_id, interface in consumers.items()
                ],
            ],
        )

        compiled = compile_execution_graph(interfaces, bucket_limit=2, canonicalization=canonicalization)

        self.assertEqual(compiled.candidates, [])

    def test_output_destination_requirement_does_not_create_artifact_flow(self) -> None:
        producer = _interface(
            "skill:markdown-producer",
            produces=[_field("skill:markdown-producer", "analysis_md", "artifact")],
        )
        converter = _interface(
            "skill:converter",
            requires=[_field("skill:converter", "markdown_output_destination", "artifact")],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, converter.skill_id: converter},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])

    def test_markdown_output_target_requirement_does_not_create_artifact_flow(self) -> None:
        producer = _interface(
            "skill:markdown-producer",
            produces=[_field("skill:markdown-producer", "analysis_md", "artifact")],
        )
        converter = _interface(
            "skill:markitdown",
            requires=[
                InterfaceField(
                    name="markdown_output_target",
                    kind="artifact",
                    confidence=0.95,
                    evidence=[
                        InterfaceEvidence(
                            skill="skill:markitdown",
                            line=97,
                            text="markitdown document.pdf -o output.md",
                        )
                    ],
                )
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, converter.skill_id: converter},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])

    def test_showcase_pdf_requirement_does_not_create_artifact_flow(self) -> None:
        producer = _interface(
            "skill:pdf-producer",
            produces=[_field("skill:pdf-producer", "new_pdf", "artifact")],
        )
        style_tool = _interface(
            "skill:style-tool",
            requires=[
                InterfaceField(
                    name="theme_showcase_pdf_available",
                    kind="artifact",
                    confidence=0.9,
                    evidence=[
                        InterfaceEvidence(
                            skill="skill:style-tool",
                            line=23,
                            text=(
                                "Show the theme showcase: Display the `theme-showcase.pdf` file "
                                "to allow users to see all available themes visually."
                            ),
                        )
                    ],
                )
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, style_tool.skill_id: style_tool},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])

    def test_optional_edit_mode_image_requirement_does_not_create_artifact_flow(self) -> None:
        producer = _interface(
            "skill:image-producer",
            produces=[_field("skill:image-producer", "generated_image_png", "artifact")],
        )
        editor = _interface(
            "skill:image-editor",
            requires=[
                InterfaceField(
                    name="image_file_for_editing",
                    kind="artifact",
                    confidence=0.92,
                    evidence=[
                        InterfaceEvidence(
                            skill="skill:image-editor",
                            line=113,
                            text="- `--input` or `-i`: Input image path for editing (enables edit mode)",
                        )
                    ],
                )
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, editor.skill_id: editor},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])

    def test_user_uploaded_image_requirement_does_not_create_artifact_flow(self) -> None:
        producer = _interface(
            "skill:image-producer",
            produces=[_field("skill:image-producer", "generated_image_png", "artifact")],
        )
        gif_builder = _interface(
            "skill:gif-builder",
            requires=[
                InterfaceField(
                    name="optional_user_uploaded_image",
                    kind="artifact",
                    confidence=0.91,
                    evidence=[
                        InterfaceEvidence(
                            skill="skill:gif-builder",
                            line=48,
                            text="If a user uploads an image, use PIL to load and work with it.",
                        )
                    ],
                )
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, gif_builder.skill_id: gif_builder},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])

    def test_required_image_handoff_still_creates_artifact_flow(self) -> None:
        producer = _interface(
            "skill:image-producer",
            produces=[_field("skill:image-producer", "generated_image_png", "artifact")],
        )
        consumer = _interface(
            "skill:image-consumer",
            requires=[
                InterfaceField(
                    name="rendered_image_file",
                    kind="artifact",
                    confidence=0.92,
                    evidence=[
                        InterfaceEvidence(
                            skill="skill:image-consumer",
                            line=12,
                            text="Consumes the rendered image file from the previous rendering step.",
                        )
                    ],
                )
            ],
        )
        canonicalization = _build_canonicalization(
            "artifact:image_asset",
            [
                ("skill:image-producer", "produces", producer.produces[0]),
                ("skill:image-consumer", "requires", consumer.requires[0]),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(len(compiled.candidates), 1)
        self.assertEqual(compiled.candidates[0].matched_name, "image_asset")

    def test_local_reference_markdown_requirement_does_not_create_artifact_flow(self) -> None:
        producer = _interface(
            "skill:outline-producer",
            produces=[_field("skill:outline-producer", "selected_outline_md", "artifact")],
        )
        presenter = _interface(
            "skill:presentation-builder",
            requires=[
                InterfaceField(
                    name="html2pptx_md",
                    kind="artifact",
                    confidence=0.95,
                    evidence=[
                        InterfaceEvidence(
                            skill="skill:presentation-builder",
                            line=4,
                            text="Read html2pptx.md completely for syntax and best practices before creating slides.",
                        )
                    ],
                )
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, presenter.skill_id: presenter},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])

    def test_local_template_html_requirement_does_not_create_artifact_flow(self) -> None:
        producer = _interface(
            "skill:html-producer",
            produces=[_field("skill:html-producer", "bundle_html", "artifact")],
        )
        consumer = _interface(
            "skill:template-consumer",
            requires=[
                InterfaceField(
                    name="template_viewer_html",
                    kind="artifact",
                    confidence=0.95,
                    evidence=[
                        InterfaceEvidence(
                            skill="skill:template-consumer",
                            line=390,
                            text="- **templates/viewer.html**: REQUIRED STARTING POINT for all HTML artifacts.",
                        )
                    ],
                )
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
        )

        self.assertEqual(compiled.candidates, [])

    def test_input_markdown_requirement_still_creates_artifact_flow(self) -> None:
        producer = _interface(
            "skill:markdown-producer",
            produces=[_field("skill:markdown-producer", "analysis_md", "artifact")],
        )
        consumer = _interface(
            "skill:markdown-consumer",
            requires=[_field("skill:markdown-consumer", "input_markdown_document", "artifact")],
        )
        canonicalization = _build_canonicalization(
            "artifact:markdown_document",
            [
                ("skill:markdown-producer", "produces", producer.produces[0]),
                ("skill:markdown-consumer", "requires", consumer.requires[0]),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(len(compiled.candidates), 1)
        self.assertEqual(compiled.candidates[0].matched_name, "markdown_document")

    def test_non_promoted_canonical_terms_do_not_fall_back_to_format_ontology(self) -> None:
        producer = _interface(
            "skill:image-producer",
            produces=[_field("skill:image-producer", "generated_image_png", "artifact")],
        )
        consumer = _interface(
            "skill:image-consumer",
            requires=[_field("skill:image-consumer", "rendered_image_file", "artifact")],
        )
        canonicalization = CanonicalizationBuild(
            objects=[
                CanonicalObject(
                    canonical_id="artifact:generated_image_file",
                    name="generated_image_file",
                    type="artifact",
                    produced_by=["skill:image-producer"],
                    promoted=False,
                    confidence=0.98,
                ),
                CanonicalObject(
                    canonical_id="artifact:rendered_image_file",
                    name="rendered_image_file",
                    type="artifact",
                    required_by=["skill:image-consumer"],
                    promoted=False,
                    confidence=0.98,
                ),
            ],
            assignments=[
                CanonicalAssignment(
                    raw_key="skill:image-producer|produces|generated_image_png|artifact",
                    skill_id="skill:image-producer",
                    role="produces",
                    raw_name="generated_image_png",
                    raw_kind="artifact",
                    canonical_id="artifact:generated_image_file",
                ),
                CanonicalAssignment(
                    raw_key="skill:image-consumer|requires|rendered_image_file|artifact",
                    skill_id="skill:image-consumer",
                    role="requires",
                    raw_name="rendered_image_file",
                    raw_kind="artifact",
                    canonical_id="artifact:rendered_image_file",
                ),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(compiled.candidates, [])

    def test_spreadsheet_deliverable_aliases_require_accepted_canonical_assignment(self) -> None:
        producer = _interface(
            "skill:financial-analysis",
            produces=[_field("skill:financial-analysis", "formatted_excel_report", "artifact")],
        )
        consumer = _interface(
            "skill:xlsx",
            requires=[_field("skill:xlsx", "workbook_or_tabular_data", "artifact")],
        )
        canonicalization = CanonicalizationBuild(
            objects=[
                CanonicalObject(
                    canonical_id="artifact:formatted_excel_report",
                    name="formatted_excel_report",
                    type="artifact",
                    produced_by=["skill:financial-analysis"],
                    promoted=False,
                    confidence=0.98,
                ),
                CanonicalObject(
                    canonical_id="artifact:workbook_or_tabular_data",
                    name="workbook_or_tabular_data",
                    type="artifact",
                    required_by=["skill:xlsx"],
                    promoted=False,
                    confidence=0.98,
                ),
            ],
            assignments=[
                CanonicalAssignment(
                    raw_key="skill:financial-analysis|produces|formatted_excel_report|artifact",
                    skill_id="skill:financial-analysis",
                    role="produces",
                    raw_name="formatted_excel_report",
                    raw_kind="artifact",
                    canonical_id="artifact:formatted_excel_report",
                ),
                CanonicalAssignment(
                    raw_key="skill:xlsx|requires|workbook_or_tabular_data|artifact",
                    skill_id="skill:xlsx",
                    role="requires",
                    raw_name="workbook_or_tabular_data",
                    raw_kind="artifact",
                    canonical_id="artifact:workbook_or_tabular_data",
                ),
            ],
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(compiled.candidates, [])

        accepted = _build_canonicalization(
            "artifact:spreadsheet_table",
            [
                ("skill:financial-analysis", "produces", producer.produces[0]),
                ("skill:xlsx", "requires", consumer.requires[0]),
            ],
        )
        accepted_compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=accepted,
        )

        self.assertEqual(len(accepted_compiled.candidates), 1)
        self.assertEqual(accepted_compiled.candidates[0].matched_name, "spreadsheet_table")
        self.assertEqual(accepted_compiled.candidates[0].source_skill, "skill:financial-analysis")
        self.assertEqual(accepted_compiled.candidates[0].target_skill, "skill:xlsx")

    def test_canonicalization_rejected_terms_do_not_fall_back_to_format_ontology(self) -> None:
        producer = _interface(
            "skill:image-producer",
            produces=[_field("skill:image-producer", "generated_image_png", "artifact")],
        )
        consumer = _interface(
            "skill:image-consumer",
            requires=[_field("skill:image-consumer", "rendered_image_file", "artifact")],
        )
        canonicalization = CanonicalizationBuild(
            rejected_terms=[
                RejectedCanonicalTerm(
                    raw_key="skill:image-consumer|requires|rendered_image_file|artifact",
                    skill_id="skill:image-consumer",
                    role="requires",
                    raw_name="rendered_image_file",
                    raw_kind="artifact",
                    reason="local_only",
                )
            ]
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            bucket_limit=100,
            canonicalization=canonicalization,
        )

        self.assertEqual(compiled.candidates, [])


if __name__ == "__main__":
    unittest.main()
