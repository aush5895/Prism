"""Article fit: does the supplied article cover what the customer actually described?

NOT PART OF THE GRADED RESPONSE. The result rides in `meta.article_fit`, the debug view
and guided mode. `response` is exactly what it was before this module existed.

WHY IT EXISTS
-------------
The engine builds its plan from the article supplied with the request (C1). Samsung's own
kit pairs some complaints with the wrong article; Phase 0 §4.3 lists four, among them a
screen-flashing complaint supplied with an email-server article. Every step extracted from
such an article is faithfully grounded, and useless. A lexical signal was measured and
could not separate the mismatched pairings (LIMITATIONS §2.6).

So the one extraction call also lists each problem the customer described and says
whether the article addresses it, quoting the sentence that does. The model is still a
parser here, not an authority: a "covered" claim is believed only when its quote is found
in the article. A model that says yes without a quote, or quotes something that is not
there, is recorded as an unverified claim and counted as not covered.

    full      every described problem is covered, with a verified quote
    partial   some are, some are not
    none      none are
    unknown   the provider reported no problems (offline stub, older recordings)

"unknown" is never shown to the customer as a verdict either way.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .. import config
from ..ir import Extraction
from . import spans
from .validate import URL_PATTERN

FULL, PARTIAL, NONE, UNKNOWN = "full", "partial", "none", "unknown"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", URL_PATTERN.sub("", text or "")).strip()


def _find(quote: str, article: str):
    """Where a claimed quote is in the article, or None. Verbatim first; then, since
    models normalise whitespace and punctuation, a located span that supports nearly all
    of the quote's subject words."""
    quote = quote.strip()
    if not quote or not article:
        return None
    at = article.find(quote)
    if at >= 0:
        return at, at + len(quote), "verbatim"
    squashed = re.sub(r"\s+", " ", quote)
    start, end, _provenance, confidence = spans.locate(squashed, article)
    if start is not None and confidence >= config.ARTICLE_FIT_MIN_SUPPORT:
        return start, end, "located"
    return None


def assess(extraction: Extraction, article: str) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    for item in extraction.complaint_issues[: config.ARTICLE_FIT_MAX_ISSUES]:
        name = _clean(item.issue)
        if not name:
            continue
        found = _find(item.evidence, article) if item.covered else None
        issues.append({
            "issue": name,
            "claimed_covered": bool(item.covered),
            "covered": found is not None,
            # None when nothing was claimed; False is a claim the article did not back.
            "quote_found": (found is not None) if item.covered else None,
            "evidence": article[found[0]:found[1]] if found else "",
            "span": [found[0], found[1]] if found else None,
            "match": found[2] if found else None,
        })
    total = len(issues)
    covered = sum(1 for i in issues if i["covered"])
    if total == 0:
        fit = UNKNOWN
    elif covered == total:
        fit = FULL
    elif covered == 0:
        fit = NONE
    else:
        fit = PARTIAL
    return {
        "fit": fit,
        "covered": covered,
        "total": total,
        "unverified_claims": sum(1 for i in issues if i["quote_found"] is False),
        "issues": issues,
    }


def uncovered(fit: Dict[str, Any] | None) -> List[str]:
    """The problems the article does not cover, for the customer and the agent."""
    if not fit or fit.get("fit") in (None, UNKNOWN):
        return []
    return [i["issue"] for i in fit.get("issues", []) if not i["covered"]]
