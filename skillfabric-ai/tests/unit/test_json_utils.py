from __future__ import annotations

import unittest

from skillfabric.runtime.json_utils import extract_response_text, parse_json_response


class JsonUtilsTests(unittest.TestCase):
    def test_extracts_text_from_responses_api_output(self) -> None:
        response = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": '{"accepted":true}'},
                    ]
                }
            ]
        }

        self.assertEqual(extract_response_text(response), '{"accepted":true}')

    def test_extracts_text_from_anthropic_content_blocks(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"accepted":true}'},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(extract_response_text(response), '{"accepted":true}')

    def test_empty_top_level_content_does_not_hide_chat_completion_text(self) -> None:
        response = {
            "content": None,
            "choices": [{"message": {"content": '{"accepted":true}'}}],
        }

        self.assertEqual(extract_response_text(response), '{"accepted":true}')

    def test_non_text_blocks_do_not_override_anthropic_text_output(self) -> None:
        response = {
            "content": [
                {"type": "thinking", "thinking": '{"accepted":false}'},
                {"type": "tool_use", "input": {"accepted": False}},
                {"type": "text", "text": '{"accepted":true}'},
            ]
        }

        self.assertEqual(parse_json_response(response), {"accepted": True})

    def test_extracts_all_chat_completion_choices_for_usage_accounting(self) -> None:
        response = {
            "choices": [
                {"message": {"content": "first"}},
                {"message": {"content": "second"}},
            ]
        }

        self.assertEqual(extract_response_text(response), "first\nsecond")

    def test_parses_first_json_object_without_greedy_brace_matching(self) -> None:
        response = 'Result: {"accepted":true}\nTrailing example: {"accepted":false}'

        self.assertEqual(parse_json_response(response), {"accepted": True})

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_json_response("[1, 2, 3]")


if __name__ == "__main__":
    unittest.main()
