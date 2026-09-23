"""Rule validators + Samsung schema validation (contract §3.2, §3.3).

Split into REPAIR (deterministic fixes we are entitled to make, e.g. trimming a
description to the word window) and ASSERT (hard gates that must hold or the response is
refused). Guide §7.5 is explicit that word-count constraints must be enforced in the
application layer rather than requested in a prompt.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .. import config
from ..schema_samsung import ContextDeeplinkResponse
from ..text import is_title_case, to_title_case, word_count

# Guide §4.2.1 — absolute prohibition on web URLs, including markdown links.
URL_PATTERN = re.compile(r"https?://|www\.|\[[^\]]*\]\([^)]*\)|\bmailto:", re.I)


class ValidationError(Exception):
    pass


# ----------------------------------------------------------------- repair
def repair_description(text: str) -> str:
    """Force `It will` + 5-7 words. A5/A4: the window counts words AFTER the prefix.

    Trimming never leaves a dangling word. The word cap is Samsung's and stays, but a
    sentence cut at exactly seven words produced customer-facing cards reading "It will
    remove physical obstructions that may interfere with". After the cut, trailing
    conjunctions, prepositions and articles are dropped until the phrase closes or the
    five-word floor is reached. The prompt now asks for a complete 5-7 word phrase in the
    first place (see llm/base.py); this is the guarantee for when it does not comply.
    """
    body = text.strip()
    if body.lower().startswith(config.DESCRIPTION_PREFIX.lower()):
        body = body[len(config.DESCRIPTION_PREFIX):].strip()
    body = body.rstrip(".")
    words = _drop_dangling([w for w in body.split() if w])
    if len(words) > config.DESCRIPTION_MAX_WORDS:
        words = _drop_dangling(words[: config.DESCRIPTION_MAX_WORDS])
    # Too short: complete the phrase rather than repeat a filler token. Padding used to
    # append "issue" per missing word, which turned a model's trailing "...obstructions
    # that" into "...obstructions that issue".
    completion = config.DESCRIPTION_COMPLETION
    pad = 0
    while len(words) < config.DESCRIPTION_MIN_WORDS:
        words.append(completion[pad % len(completion)])
        pad += 1
    if len(words) > config.DESCRIPTION_MAX_WORDS:
        words = _drop_dangling(words[: config.DESCRIPTION_MAX_WORDS])
    return f"{config.DESCRIPTION_PREFIX} " + " ".join(words)


def _drop_dangling(words: List[str]) -> List[str]:
    """Remove trailing words a phrase may not end on. Never empties the list."""
    out = list(words)
    while len(out) > 1 and out[-1].strip(".,;:").lower() in config.DESCRIPTION_DANGLING_WORDS:
        out.pop()
    return out


def repair_title(text: str) -> str:
    """2-3 words, sentence case."""
    words = [w for w in text.strip().rstrip(".").split() if w][: config.TITLE_MAX_WORDS]
    while len(words) < config.TITLE_MIN_WORDS:
        words.append("issue")
    joined = " ".join(words)
    return joined[0].upper() + joined[1:].lower()


def repair_action_name(text: str) -> str:
    return to_title_case(text.strip().rstrip("."))


def build_goal(topic: str) -> str:
    topic = " ".join(w.capitalize() for w in topic.strip().split()[:3]) or "Device"
    return config.GOAL_TEMPLATE.format(topic=topic)


def scrub_urls(value: Any) -> Any:
    """Strip URL-shaped substrings anywhere in the payload before the assert gate.

    Belt and braces: the prompt forbids them, this removes them, and assert_no_urls then
    fails the request if anything survived. Models inject these from pretraining memory
    (guide §7.3), so a single layer is not enough.
    """
    if isinstance(value, str):
        return re.sub(r"\s+", " ", URL_PATTERN.sub("", value)).strip()
    if isinstance(value, list):
        return [scrub_urls(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_urls(v) for k, v in value.items()}
    return value


# ----------------------------------------------------------------- assert
def assert_no_urls(payload: Dict[str, Any]) -> None:
    import json

    hit = URL_PATTERN.search(json.dumps(payload))
    if hit:
        raise ValidationError(f"URL leak in response: {hit.group(0)!r}")


def assert_rules(response: Dict[str, Any]) -> List[str]:
    """Every graded field rule from guide §4.1. Returns the list of checks that ran."""
    checks: List[str] = []
    for goal in response.get("contexts", []):
        if not re.fullmatch(r"Follow these steps to perform this .+ (Troubleshooting|Configuration)",
                            goal["goal"]):
            raise ValidationError(f"goal template violated: {goal['goal']!r}")
        if not config.TITLE_MIN_WORDS <= word_count(goal["title"]) <= config.TITLE_MAX_WORDS:
            raise ValidationError(f"title must be 2-3 words: {goal['title']!r}")
        if not 0.0 <= goal["score"] <= 1.0:
            raise ValidationError(f"score out of range: {goal['score']}")
        checks += ["goal_template", "title_words", "score_range"]

        categories = [a["category"] for a in goal["actions"]]
        if "critical" in categories:
            first = categories.index("critical")
            if any(c != "critical" for c in categories[first:]):
                raise ValidationError("critical actions must be ordered last")
            checks.append("critical_last")

        for action in goal["actions"]:
            desc = action["description"]
            if not desc.startswith(config.DESCRIPTION_PREFIX):
                raise ValidationError(f"description must start 'It will': {desc!r}")
            body_words = word_count(desc) - word_count(config.DESCRIPTION_PREFIX)
            if not config.DESCRIPTION_MIN_WORDS <= body_words <= config.DESCRIPTION_MAX_WORDS:
                raise ValidationError(f"description body must be 5-7 words ({body_words}): {desc!r}")
            if not is_title_case(action["actionName"]):
                raise ValidationError(f"actionName must be Title Case: {action['actionName']!r}")
            if action["category"] == "manual":
                for group in action["stepGroups"]:
                    if group.get("actionableDeeplink") or group.get("validationDeeplink"):
                        raise ValidationError(
                            f"manual action carries a deeplink: {action['actionName']!r}")
            checks += ["description_prefix", "description_words", "action_name_case", "manual_null"]
    return checks


def assert_variations(variations: List[str]) -> None:
    if not config.VARIATIONS_MIN <= len(variations) <= config.VARIATIONS_MAX:
        raise ValidationError(
            f"query_variations must number {config.VARIATIONS_MIN}-{config.VARIATIONS_MAX}, "
            f"got {len(variations)}")
    if len({v.strip().lower() for v in variations}) != len(variations):
        raise ValidationError("query_variations must be distinct")


def validate_response(response: Dict[str, Any]) -> ContextDeeplinkResponse:
    """The hard gate: the response object must satisfy Samsung's unmodified schema."""
    return ContextDeeplinkResponse(**response)
