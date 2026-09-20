"""Provider interface + the grounded-extraction prompt.

The prompt is short on purpose. Constraints that can be checked in code are NOT asked
for in prose — they are enforced by validate.py and deeplinks.py. Guide §7.5:
"Asking an LLM to respect word count constraints in natural language is unreliable.
Enforce programmatic validation, trimming, and correction loops in the application layer."
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from ..ir import Extraction

SYSTEM_PROMPT = """You convert one Samsung customer-care reference article into a structured troubleshooting plan.

ABSOLUTE RULES
1. EVIDENCE ONLY. Every action and every step must come from the reference text supplied in this request. If the reference text does not support a step, do not write it.
2. DO NOT INVENT STEPS. You are not recalling Samsung knowledge from training; you are restructuring the text in front of you.
3. DO NOT INVENT URLS OR DEEPLINKS. Never output http, https, www, a markdown link, or any bixby:// URI. Deeplinks are attached by a separate system; there is no field for them here.
4. PRESERVE MEANING. Keep the source's wording and intent. Do not soften a destructive operation or generalise a specific screen.
5. RETURN SOURCE SPANS. For each step, return [start, end) character offsets into the reference text that support it.
6. OUTPUT ONLY the required JSON object. No prose, no markdown fence, no preamble.
7. If the reference text contains no viable solution for the complaint, return an empty actions list.
8. BE EXHAUSTIVE. The reference text is organised into distinct troubleshooting sections (often numbered or headed). Emit ONE action per section that describes a user-performable step, in the order the sections appear. Do not skip a section because it seems minor, applies to an edge case, or is not a Settings screen — a step to try a different charger, check for software updates, or contact support is just as required as an "auto" one. Only skip a section if it truly contains no user-performable step (e.g., pure background explanation).

STRUCTURE
- One action = one physical screen or feature. Several taps on the SAME screen belong to ONE action, not several.
- NEVER combine a manual/physical step (cleaning, removing an accessory, handling hardware) with a Settings-screen step (opening Settings, tapping a toggle) in the same action, even if the source text discusses them in the same paragraph. They are different actions with different category_hint values — split them.
- If one screen supports two distinct operations that the text describes separately (for example an enable path and a disable path under different conditions), emit them as two step_groups under ONE action.
- Each step is one physical interaction, written as an imperative.
- category_hint: "auto" for a Settings screen the user can be taken to; "critical" for disruptive or irreversible operations (factory reset, restart, firmware update, safe mode); "manual" for physical interventions (cleaning, removing an accessory, replacing hardware, contacting support).
- description: begins "It will" and states the concrete benefit.
- title: 2 to 3 words naming the core issue, sentence case.
- goal_topic: 1 to 3 words naming the topic, Title Case.
- query_variations: 8 to 10 distinct paraphrases of the USER COMPLAINT across registers: formal, casual, keyword-only, frustrated, typo-inclusive.
"""

USER_TEMPLATE = """USER COMPLAINT:
{query}

REFERENCE TEXT (the only permitted source of actions and steps):
<<<
{evidence}
>>>

Return the JSON object now."""

# JSON schema handed to providers that support structured output.
EXTRACTION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal_topic": {"type": "string"},
        "title": {"type": "string"},
        "query_variations": {"type": "array", "items": {"type": "string"}},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action_name": {"type": "string"},
                    "description": {"type": "string"},
                    "category_hint": {"type": "string", "enum": ["auto", "manual", "critical"]},
                    "source_span": {
                        "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                    },
                    "step_groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "source_span": {
                                                "type": "array", "items": {"type": "integer"},
                                                "minItems": 2, "maxItems": 2,
                                            },
                                        },
                                        "required": ["text"],
                                    },
                                }
                            },
                            "required": ["steps"],
                        },
                    },
                },
                "required": ["action_name", "description", "step_groups"],
            },
        },
    },
    "required": ["goal_topic", "title", "actions", "query_variations"],
}


class LLMProvider(abc.ABC):
    """All providers are temperature 0 and return an Extraction. No provider ever
    sees the deeplink catalog."""

    name: str = "base"
    model: Optional[str] = None

    @abc.abstractmethod
    def extract(self, query: str, evidence: str) -> Extraction:
        ...

    def last_usage(self) -> Dict[str, float]:
        """Token/cost accounting for meta.cost_usd. Zero for offline providers."""
        return {"prompt_tokens": 0.0, "completion_tokens": 0.0, "cost_usd": 0.0}
