"""[0] Query enrichment (contract §7 stage 1).

Normalizes a colloquial complaint into slots + a canonical technical query. The canonical
form is the cache key, so two phrasings of the same issue must normalize to the same
string — that is the whole point (guide §7.1: exact-string cache keying is pitfall #1).
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from ..ir import EnrichedQuery
from ..text import content, tokens
from .deeplinks import load_lexicons

# Device mention, generic across Samsung's naming schemes. Masked models ("S***** Ultra")
# are matched too, since input fixtures contain them.
_DEVICE = re.compile(
    r"\b((?:samsung\s+)?(?:galaxy\s+)?"
    r"(?:z\s+)?(?:flip|fold|note|tab)?\s*"
    r"[a-z]?[\*\d]{1,6}[a-z0-9/]*"
    r"(?:\s+(?:ultra|plus|pro|fe|5g|4g))?)\b",
    re.I,
)
_MODEL_HINT = re.compile(r"(galaxy|samsung|flip|fold|note|tab)", re.I)


def _device(raw: str) -> Optional[str]:
    best: Optional[str] = None
    for m in _DEVICE.finditer(raw):
        cand = " ".join(m.group(1).split())
        if not _MODEL_HINT.search(cand) or len(cand) < 4:
            continue
        if best is None or len(cand) > len(best):
            best = cand
    return best.title() if best else None


def _domain(raw: str, lexicons: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    toks = set(tokens(raw))
    scored = {
        name: sum(1 for kw in words if kw in toks)
        for name, words in lexicons["domains"].items()
    }
    top = max(scored, key=lambda k: scored[k])
    return top if scored[top] else None


def _symptoms(raw: str, lexicons) -> List[str]:
    toks = tokens(raw)
    vocab = {kw for words in lexicons["domains"].values() for kw in words}
    return [t for i, t in enumerate(toks) if t in vocab and t not in toks[:i]]


def enrich(raw: str) -> EnrichedQuery:
    lex = load_lexicons()
    device = _device(raw)
    domain = _domain(raw, lex)
    symptoms = _symptoms(raw, lex)

    # Canonical query: order-stable, deduplicated content tokens. Deliberately drops
    # filler so paraphrases converge on one key.
    seen, canon = set(), []
    for t in content(raw):
        if t not in seen:
            seen.add(t)
            canon.append(t)
    canonical = " ".join(canon)

    return EnrichedQuery(
        raw=raw,
        canonical=canonical,
        device=device,
        domain=domain,
        symptoms=symptoms,
        cache_key=hashlib.sha1(canonical.encode()).hexdigest(),
    )
