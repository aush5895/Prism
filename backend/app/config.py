"""Central configuration. Every tunable lives here — no magic numbers in pipeline code.

Contract refs (docs/API_CONTRACT.md): thresholds in §5, score weights in §3.4,
tier table in §6.1. Nothing in this file is scenario-specific (directive 17).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = 0.0
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # never a literal key

# ---- cache (contract §7 stage 8): precision over recall ----
CACHE_SIMILARITY_MIN = float(os.getenv("PRISM_CACHE_SIMILARITY_MIN", "0.92"))
CACHE_SLOT_GUARD = True
