"""
src/retrieval/query_analyzer.py
────────────────────────────────
4-way query classifier. Pure regex + heuristics. No API call. No ML model.

QueryType:
  PAGE_METADATA  — "page 25 of the printer manual"  → ChunkStore scan, 0 embeddings
  EXACT_CODE     — "CF289A", "Error E42", "P0342"   → BM25 primary, dense backup
  VAGUE          — "cartridge error" (short, no verb, no code) → HyDE expansion
  NORMAL         — everything else                   → direct embedding
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    PAGE_METADATA = "page_metadata"
    EXACT_CODE    = "exact_code"
    VAGUE         = "vague"
    NORMAL        = "normal"


@dataclass
class ParsedQuery:
    query_type:  QueryType
    raw_query:   str
    page:        int | None        = None   # PAGE_METADATA only
    manual_name: str | None        = None   # if a manual was detected
    use_hyde:    bool              = False  # set by caller using config.use_hyde


# ---------------------------------------------------------------------------
# Manual name aliases → internal manual_name keys
# ---------------------------------------------------------------------------
_MANUAL_ALIASES: dict[str, str] = {
    "printer":    "printer_manual",
    "hp printer": "printer_manual",
    "laserjet":   "printer_manual",
    "haier":      "haier_ac_manual",
    "air conditioner": "haier_ac_manual",
    "ac":         "haier_ac_manual",
    "atlas":      "atlas_copco_manual",
    "atlas copco":"atlas_copco_manual",
    "copco":      "atlas_copco_manual",
    "cummins":    "cummins_generator_manual",
    "generator":  "cummins_generator_manual",
    "goulds":     "goulds_pump_manual",
    "pump":       "goulds_pump_manual",
    "hospira":    "hospira_infusion_manual",
    "infusion":   "hospira_infusion_manual",
    "hyundai":    "hyundai_cnc_manual",
    "cnc":        "hyundai_cnc_manual",
    "merrychef":  "merrychef_oven_manual",
    "oven":       "merrychef_oven_manual",
    "nellcor":    "nellcor_monitor_manual",
    "monitor":    "nellcor_monitor_manual",
    "ups":        "ups_manual",
    "whirlpool":  "whirlpool_dishwasher_manual",
    "dishwasher": "whirlpool_dishwasher_manual",
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Page patterns: "page 3", "pg 3", "pages 5-6", "page number 3"
_PAGE_RE = re.compile(
    r'\bpage[s]?\s*(?:number\s*)?(\d+)(?:\s*[-–]\s*\d+)?\b'
    r'|\bpg\.?\s*(\d+)\b',
    re.IGNORECASE,
)

# Exact code patterns (part numbers, error codes, model numbers)
# CF289A, P0342, Error E01, E42, 06-2148-00, HP-CF289A
_CODE_RE = re.compile(
    r'\b[A-Z]{1,4}[-]?\d{2,}[A-Z0-9]*\b'           # CF289A, P0342, HP-CF289A
    r'|\b[Ee]rror[-\s]?[A-Z0-9]+\b'                 # Error E01, Error-42
    r'|\b[Ee]\d{2,3}\b'                              # E42, E101
    r'|\b\d{2,}-\d{2,}-\d{2,}\b'                    # 06-2148-00 (part numbers)
    r'|\b[A-Z]\d{4,}\b',                             # A12345 style model numbers
)

# Strong question/action verbs that indicate a normal (non-vague) semantic query
_VERB_RE = re.compile(
    r'\b(how|what|where|when|why|which|who'
    r'|replace|remove|install|insert|set|reset|check|fix|clean'
    r'|open|close|press|connect|disconnect|configure|adjust|calibrate'
    r'|start|stop|turn|switch|load|unload|feed|refill|drain'
    r'|is|are|does|do|can|will|should|would|could'
    r'|explain|describe|show|list|tell)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_query(query: str) -> ParsedQuery:
    """
    Classify query into one of four types.

    Priority order (strict):
      1. PAGE_METADATA  (explicit page number)
      2. EXACT_CODE     (part number / error code)
      3. VAGUE          (short, no verb, no code)
      4. NORMAL         (default)
    """
    q = query.strip()

    # ── Step 1: detect manual name ─────────────────────────────────────────
    manual_name = _detect_manual(q)

    # ── Step 2: detect page number ─────────────────────────────────────────
    page_match = _PAGE_RE.search(q)
    if page_match:
        page_num = int(page_match.group(1) or page_match.group(2))
        return ParsedQuery(
            query_type=QueryType.PAGE_METADATA,
            raw_query=q,
            page=page_num,
            manual_name=manual_name,
        )

    # ── Step 3: detect exact code ──────────────────────────────────────────
    if _CODE_RE.search(q):
        return ParsedQuery(
            query_type=QueryType.EXACT_CODE,
            raw_query=q,
            manual_name=manual_name,
        )

    # ── Step 4: vague vs normal ────────────────────────────────────────────
    words = q.split()
    has_verb = bool(_VERB_RE.search(q))
    is_vague = len(words) <= 4 and not has_verb

    return ParsedQuery(
        query_type=QueryType.VAGUE if is_vague else QueryType.NORMAL,
        raw_query=q,
        manual_name=manual_name,
        use_hyde=is_vague,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_manual(query: str) -> str | None:
    """Return internal manual_name if a known alias appears in the query."""
    q_lower = query.lower()
    # Try longer aliases first (avoid "ac" matching inside "each")
    for alias in sorted(_MANUAL_ALIASES, key=len, reverse=True):
        # word-boundary check
        pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
        if re.search(pattern, q_lower):
            return _MANUAL_ALIASES[alias]
    return None
