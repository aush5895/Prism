"""Central configuration. Every tunable lives here — no magic numbers in pipeline code.

Contract refs (docs/API_CONTRACT.md): thresholds in §5, score weights in §3.4,
tier table in §6.1. Nothing in this file is scenario-specific (directive 17).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load the repo-root .env BEFORE any os.getenv below. Without this, nothing reads .env at
# all: every value here came from the real process environment, so a configured
# GEMINI_API_KEY sitting in .env was invisible and `--provider gemini` failed with "not
# set" unless the operator had exported it by hand.
#
# override=False so a real environment variable still wins over the file — CI and
# container runs set variables directly and must not be overridden by a stray local .env.
# The file stays gitignored; only .env.example is committed.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv absent, environment-only operation
    pass

DATA_DIR = Path(os.getenv("PRISM_DATA_DIR", REPO_ROOT / "data"))
LEXICON_DIR = DATA_DIR / "lexicons"

DEEPLINKS_PATH = DATA_DIR / "deeplinks.json"
SIIS_PATH = DATA_DIR / "siis_responses.json"
SCHEMA_PATH = DATA_DIR / "schema.py"

# Samsung's schema.py is immutable. Asserted by tests/test_contract.py.
SCHEMA_SHA256 = "649440e0309b25dc5363039fc28dd7f21b71177779b909def8757b41e76ebf6b"

# ---- resolver gates (contract §5) ----
CONCEPT_COVERAGE_MIN = float(os.getenv("PRISM_CONCEPT_COVERAGE_MIN", "0.60"))  # gate [4]
MARGIN_DELTA = float(os.getenv("PRISM_MARGIN_DELTA", "0.08"))                 # gate [5]
CANDIDATE_POOL = int(os.getenv("PRISM_CANDIDATE_POOL", "40"))                 # gate [1]

# Gate [3] reads its scope qualifier from these catalog fields only. Measured: 42 of the
# 578 entries carry a qualifier ("talkback", "cover screen", "auto") in `description`
# that never appears in `message` — e.g. DL-0330 "View Speak usage hints" is described as
# a TalkBack screen. Including `description` therefore rejects correct candidates whose
# own message a step legitimately names. `message` is the candidate's claim about what it
# targets; `description` is prose about where it lives.
SCOPE_FIELDS = tuple(os.getenv("PRISM_SCOPE_FIELDS", "message").split(","))

# ---- score formula (contract §3.4) ----
SCORE_W_SPAN = 0.40
SCORE_W_DEEPLINK = 0.30
SCORE_W_EVIDENCE = 0.30

# ---- output rule constraints (Theme 2 guide §4.1) ----
DESCRIPTION_PREFIX = "It will"
DESCRIPTION_MIN_WORDS = 5   # counted AFTER the prefix (contract ambiguity A4)
DESCRIPTION_MAX_WORDS = 7
TITLE_MIN_WORDS = 2
TITLE_MAX_WORDS = 3
GOAL_TEMPLATE = "Follow these steps to perform this {topic} Troubleshooting"
VARIATIONS_MIN = 8
VARIATIONS_MAX = 10

DUMMY_DEEPLINK = "bixby://dummy_positive"
TOGGLE_TYPES = {"onURL", "offURL", "updateURL"}  # tier 1 vs tier 2 (contract §6.1)

# ---- LLM ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
# gemini-2.0-flash was retired for new API keys (live 404 confirmed 2026-09-21: Google's
# own error names the replacement). gemini-3.1-flash-lite is the current default: no
# thinking-token overhead for a deterministic parsing task, cheapest per-token rate,
# and no transient 503s observed in testing (unlike gemini-3.6-flash, the heavier
# successor, which is still a valid override via LLM_MODEL).
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
LLM_TEMPERATURE = 0.0
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # never a literal key

# ---- cache (contract §7 stage 8): precision over recall ----
# Threshold fitted on a seeded half of the supplied rows and reported on the held-out
# half by evaluation/run_eval.py, under a rule declared before the split: zero false
# positives first, hit rate second. See docs/metrics.md §4.
CACHE_SIMILARITY_MIN = float(os.getenv("PRISM_CACHE_SIMILARITY_MIN", "0.60"))
CACHE_SLOT_GUARD = True
CACHE_ENABLED = os.getenv("PRISM_CACHE_ENABLED", "1") not in ("0", "false", "False")
CACHE_MAX_ENTRIES = int(os.getenv("PRISM_CACHE_MAX_ENTRIES", "5000"))
CACHE_HIT_LOG_MAX = int(os.getenv("PRISM_CACHE_HIT_LOG_MAX", "1000"))
CACHE_EMBED_MODEL = os.getenv("PRISM_CACHE_EMBED_MODEL", "all-MiniLM-L6-v2")
CACHE_TFIDF_COMPONENTS = int(os.getenv("PRISM_CACHE_TFIDF_COMPONENTS", "128"))
