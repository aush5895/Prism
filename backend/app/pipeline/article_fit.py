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
parser here, not an authority. Each problem ends in one of three states:

    covered       the model said yes AND its quote is a whole sentence of the article
    not_covered   the model said no
    unverified    the model said yes but the quote is not a whole sentence of the article
                  (invented, a fragment, a single word, or carrying a URL)

An unverified claim is NOT evidence that the article misses the problem: the model may be
right and have misquoted. So it is never shown as "not covered", and it keeps the fit off
"none" and "full". The fit over all problems:

    full      every problem is covered
    none      every problem is not_covered
    partial   anything else with at least one judged problem
    unknown   no problems reported (offline stub, older recordings, a cache hit) or every
              claim unverified

What verification proves is narrow: the quoted words are in the article. It does not
prove the sentence addresses the problem; a model can quote a real sentence about
something else. See LIMITATIONS §2.6.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .. import config
from ..ir import Extraction
from .validate import URL_PATTERN

FULL, PARTIAL, NONE, UNKNOWN = "full", "partial", "none", "unknown"
COVERED, NOT_COVERED, UNVERIFIED = "covered", "not_covered", "unverified"

# Characters a model routinely rewrites when "copying": curly quotes and dashes.
_EQUIVALENT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                             "\u2013": "-", "\u2014": "-", "\u00a0": " "})
_END_MARKS = ".!?"
# What may sit immediately before a sentence start: nothing, a line break, the end of the
# previous sentence, or a list marker ("1.", "-", "\u2022").
_START_AFTER = re.compile(r"(?:^|[\n.!?:;\u2022*\-)]|\b\d+[.)])[ \t\"'\u201c]*$")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", URL_PATTERN.sub("", text or "")).strip()


def _normalise(text: str):
    """Lower-cased text with whitespace runs collapsed and quotes/dashes unified, plus a
    map from each normalised character back to its offset in the original."""
    out, where, prev_space = [], [], False
    for i, ch in enumerate((text or "").translate(_EQUIVALENT)):
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch.lower())
            prev_space = False
        where.append(i)
    return "".join(out), where


def _at_sentence_start(article: str, at: int) -> bool:
    return bool(_START_AFTER.search(article[max(0, at - 12):at]))


def _at_sentence_end(article: str, end: int) -> bool:
    rest = article[end:end + 3]
    return (article[end - 1:end] in _END_MARKS
            or not rest.strip(" \t")
            or rest.lstrip(" \t")[:1] in _END_MARKS + "\n")


def _find(quote: str, article: str):
    """Where a claimed quote sits in the article as a whole sentence, or None.

    Accepted: the quote verbatim, or equal to the article after collapsing whitespace,
    unifying curly quotes and dashes and ignoring case and a final full stop. Refused: a
    quote shorter than a real sentence ("Settings", "."), one that starts or stops
    mid-sentence, and any quote carrying a URL. Word overlap alone is never enough: the
    earlier version accepted a located span, and an opposite-meaning sentence sharing the
    same words passed.
    """
    quote = (quote or "").strip()
    if (not quote or not article or URL_PATTERN.search(quote)
            or len(quote) < config.GUIDED_MIN_QUOTE_CHARS):
        return None
    at = article.find(quote)
    if at >= 0:
        span, how = (at, at + len(quote)), "verbatim"
    else:
        norm_article, where = _normalise(article)
        norm_quote, _ = _normalise(quote)
        norm_quote = norm_quote.rstrip(_END_MARKS + " ")
        pos = norm_article.find(norm_quote) if norm_quote else -1
        if pos < 0:
            return None
        span, how = (where[pos], where[pos + len(norm_quote) - 1] + 1), "normalised"
    start, end = span
    # Take in the sentence's own full stop when the quote left it off.
    if end < len(article) and article[end] in _END_MARKS:
        end += 1
    if not (_at_sentence_start(article, start) and _at_sentence_end(article, end)):
        return None
    if URL_PATTERN.search(article[start:end]):
        return None
    return start, end, how


def assess(extraction: Extraction, article: str) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    for item in extraction.complaint_issues[: config.ARTICLE_FIT_MAX_ISSUES]:
        name = _clean(item.issue)
        if not name:
            continue
        found = _find(item.evidence, article) if item.covered else None
        status = COVERED if found else (UNVERIFIED if item.covered else NOT_COVERED)
        issues.append({
            "issue": name,
            "status": status,
            "claimed_covered": bool(item.covered),
            "covered": status == COVERED,
            # None when nothing was claimed; False is a claim the article did not back.
            "quote_found": (found is not None) if item.covered else None,
            "evidence": article[found[0]:found[1]] if found else "",
            "span": [found[0], found[1]] if found else None,
            "match": found[2] if found else None,
        })
    counts = {k: sum(1 for i in issues if i["status"] == k)
              for k in (COVERED, NOT_COVERED, UNVERIFIED)}
    total = len(issues)
    if total == 0 or counts[UNVERIFIED] == total:
        fit = UNKNOWN
    elif counts[COVERED] == total:
        fit = FULL
    elif counts[NOT_COVERED] == total:
        fit = NONE
    else:
        fit = PARTIAL
    return {
        "fit": fit,
        "covered": counts[COVERED],
        "not_covered": counts[NOT_COVERED],
        "total": total,
        "unverified_claims": counts[UNVERIFIED],
        "issues": issues,
    }


def unknown(reason: str) -> Dict[str, Any]:
    """A fit that states no verdict, with the reason it cannot."""
    return {"fit": UNKNOWN, "covered": 0, "not_covered": 0, "total": 0,
            "unverified_claims": 0, "issues": [], "reason": reason}


def status_of(issue: Dict[str, Any]) -> str:
    """An issue's state; derived for fits recorded before `status` existed."""
    if issue.get("status"):
        return issue["status"]
    if issue.get("covered"):
        return COVERED
    return UNVERIFIED if issue.get("quote_found") is False else NOT_COVERED


def _named(fit: Dict[str, Any] | None, status: str) -> List[str]:
    return [i["issue"] for i in (fit or {}).get("issues", []) if status_of(i) == status]


def uncovered(fit: Dict[str, Any] | None) -> List[str]:
    """Problems the model judged the article does not cover. Never an unverified claim,
    and never anything from an unknown fit."""
    if not fit or fit.get("fit") in (None, UNKNOWN):
        return []
    return _named(fit, NOT_COVERED)


def unverified(fit: Dict[str, Any] | None) -> List[str]:
    """Problems the model said were covered but could not back with a real sentence. Kept
    even when the fit is unknown (every claim unverified): the agent should know."""
    return _named(fit, UNVERIFIED)
