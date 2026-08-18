"""Deterministic pattern library for input, scope and injection screening.

All detection here is regex and set membership — no model call. That is the
point: the decision to block must not depend on an LLM, because an LLM
classifier is itself a target of the injection it is meant to detect. An
optional model-based second opinion exists, but only for the *inconclusive*
middle ground, and it can never overturn a deterministic block.

Patterns are grouped by what they indicate, so an audit record names a category
rather than a raw regex.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# --------------------------------------------------------------------------
# Character-level hygiene
# --------------------------------------------------------------------------

#: Zero-width and bidi-override characters. Used to hide text from a human
#: reviewer while keeping it visible to the model. Written as escapes so the
#: pattern stays reviewable in a diff.
INVISIBLE_CHARS: Final = re.compile(
    "["
    "­"  # soft hyphen
    "​-‏"  # zero-width space .. right-to-left mark
    "‪-‮"  # bidi embedding / override
    "⁠-⁤"  # word joiner .. invisible plus
    "⁪-⁯"  # deprecated format characters
    "﻿"  # byte order mark
    "]"
)

#: C0/C1 control characters except tab/newline/carriage return.
CONTROL_CHARS: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

#: Long unbroken token — a marker of encoded payloads rather than prose.
LONG_TOKEN: Final = re.compile(r"\S{200,}")

#: Base64-ish blob long enough to hide an instruction in.
ENCODED_BLOB: Final = re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})")

# --------------------------------------------------------------------------
# Prompt injection / system manipulation
# --------------------------------------------------------------------------

INSTRUCTION_OVERRIDE: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?i)\b(?:ignore|disregard|forget|discard|override|bypass|skip)\b[^.\n]{0,40}"
        r"\b(?:previous|prior|earlier|above|all|any|initial|original|system|these|your)\b"
        r"[^.\n]{0,30}\b(?:instruction|instructions|prompt|prompts|rule|rules|"
        r"direction|directions|context|guideline|guidelines|constraint|constraints|task)\b"
    ),
    re.compile(r"(?i)\bforget\s+(?:everything|all)\b"),
    re.compile(r"(?i)\bstart\s+over\s+(?:and|then)\b[^.\n]{0,30}\binstead\b"),
    re.compile(
        r"(?i)\b(?:new|updated|revised|real|actual|true)\s+"
        r"(?:instruction|instructions|system\s+prompt|directive|directives|rules?)\b\s*[:\-]"
    ),
    re.compile(r"(?i)\byou\s+(?:are|must)\s+now\b[^.\n]{0,40}\b(?:instead|not)\b"),
    re.compile(r"(?i)\bfrom\s+now\s+on\b[^.\n]{0,40}\bignore\b"),
    re.compile(r"(?i)\bdo\s+not\s+follow\b[^.\n]{0,30}\b(?:system|previous|prior)\b"),
    re.compile(r"(?i)\bstop\s+being\b[^.\n]{0,30}\b(?:assistant|agent|shopping)\b"),
    re.compile(r"(?i)\b(?:ignore|skip)\s+the\s+laptop\s+task\b"),
)

SYSTEM_PROMPT_EXTRACTION: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?i)\b(?:reveal|show|print|output|repeat|display|expose|dump|disclose|"
        r"reproduce|echo|leak|share|tell\s+me|give\s+me|what\s+(?:is|are|was|were))\b"
        r"[^.\n]{0,50}\b(?:system\s+prompt|system\s+message|system\s+instruction\w*|"
        r"initial\s+prompt|original\s+prompt|your\s+prompt|your\s+instruction\w*|"
        r"your\s+rules|your\s+configuration|your\s+guidelines|prompt\s+template)\b"
    ),
    re.compile(
        r"(?i)\b(?:repeat|print|output|echo)\b[^.\n]{0,30}"
        r"\b(?:everything|all)\b[^.\n]{0,30}\babove\b"
    ),
    re.compile(r"(?i)\bwhat\s+were\s+you\s+told\b"),
    re.compile(r"(?i)\byour\s+(?:exact|verbatim|full|complete|raw)\s+(?:prompt|instructions?)\b"),
    re.compile(r"(?i)\b(?:list|enumerate|describe)\s+your\s+(?:tools?|functions?|schemas?)\b"),
    re.compile(r"(?i)\bverbatim\b[^.\n]{0,30}\b(?:prompt|instructions?)\b"),
)

ROLE_SPOOFING: Final[tuple[re.Pattern[str], ...]] = (
    # Chat-template delimiters appearing in user text.
    re.compile(r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>"),
    re.compile(r"\[/?INST\]|<<SYS>>|\[/?SYS\]"),
    re.compile(r"(?i)</?(?:system|assistant|human|user)(?:_instructions?|_prompt)?\s*>"),
    # Line-leading role labels — an attempt to fake a new conversation turn.
    re.compile(r"(?im)^\s*(?:system|assistant|developer)\s*:\s*\S"),
    re.compile(r"(?i)```\s*(?:system|assistant)\b"),
    re.compile(r"(?i)\bhuman\s*:\s*.{0,80}\bassistant\s*:"),
)

JAILBREAK: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\b(?:developer|debug|god|admin|root|maintenance|unrestricted)\s+mode\b"),
    re.compile(r"(?i)\bjailbreak\b|\bDAN\s+mode\b|\bdo\s+anything\s+now\b"),
    re.compile(r"(?i)\bpretend\s+(?:you|to\s+be)\b[^.\n]{0,40}\bno\s+(?:rules|restrictions|limits)\b"),
    re.compile(r"(?i)\bwithout\s+(?:any\s+)?(?:restrictions|limitations|filters|guardrails|rules)\b"),
    re.compile(r"(?i)\byou\s+have\s+no\s+(?:rules|restrictions|guidelines|filters)\b"),
    re.compile(r"(?i)\bact\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?an?\s+unfiltered\b"),
    re.compile(r"(?i)\b(?:disable|turn\s+off|remove)\s+(?:your\s+)?(?:safety|guardrails?|filters?|restrictions?)\b"),
    re.compile(r"(?i)\bhypothetically\b[^.\n]{0,40}\bif\s+you\s+(?:had|could)\s+no\b"),
)

SECRET_EXFILTRATION: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?i)\b(?:reveal|show|print|output|give|tell|leak|dump|expose|send|list|"
        r"what\s+(?:is|are))\b[^.\n]{0,40}"
        r"\b(?:api[\s_-]?key|api[\s_-]?keys|secret\s+key|access\s+token|auth\s+token|"
        r"bearer\s+token|credential|credentials|password|passwords|private\s+key|"
        r"aws[\s_-]?(?:key|credential|secret)|env(?:ironment)?\s+variable|\.env\b|"
        r"connection\s+string|service\s+account)\b"
    ),
    re.compile(r"(?i)\b(?:cat|print|read|open)\s+(?:the\s+)?\.env\b"),
    re.compile(r"(?i)\bos\.environ\b|\bprocess\.env\b|\bgetenv\b"),
    re.compile(r"(?i)\byour\s+(?:api\s+)?key\b[^.\n]{0,20}\?"),
)

#: Injection phrasing that specifically targets the recommendation decision —
#: the highest-value attack against a shopping agent.
RECOMMENDATION_MANIPULATION: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?i)\b(?:you\s+must|always|be\s+sure\s+to|make\s+sure\s+(?:to|you)|"
        r"remember\s+to|it\s+is\s+(?:important|mandatory)\s+to)\b[^.\n]{0,40}"
        r"\b(?:recommend|choose|select|pick|rank|prefer|promote|feature)\b"
    ),
    re.compile(r"(?i)\b(?:recommend|select|choose)\s+this\s+(?:product|item|laptop|one)\b"),
    re.compile(r"(?i)\b(?:rank|place|put)\s+(?:this|it)\s+(?:first|top|#?1|number\s+one)\b"),
    re.compile(r"(?i)\bignore\s+(?:the\s+)?(?:other|competing|cheaper)\s+(?:product|option|laptop)s?\b"),
    re.compile(r"(?i)\bthis\s+is\s+the\s+(?:best|only)\s+(?:choice|option)\b[^.\n]{0,20}\bmust\b"),
    re.compile(r"(?i)\b(?:set|report|treat)\s+(?:the\s+)?price\s+(?:as|to)\b"),
    re.compile(r"(?i)\bdisregard\s+(?:the\s+)?(?:budget|price|constraint)\b"),
)

#: Category name -> patterns. Order matters only for which reason is reported.
INJECTION_CATEGORIES: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    "instruction_override": INSTRUCTION_OVERRIDE,
    "system_prompt_extraction": SYSTEM_PROMPT_EXTRACTION,
    "role_spoofing": ROLE_SPOOFING,
    "jailbreak": JAILBREAK,
    "secret_exfiltration": SECRET_EXFILTRATION,
    "recommendation_manipulation": RECOMMENDATION_MANIPULATION,
}

# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------

#: Topics refused outright, regardless of how the request is framed.
DISALLOWED_TOPICS: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    "malware": (
        re.compile(r"(?i)\b(?:write|create|build|generate|make|code)\b[^.\n]{0,30}"
                   r"\b(?:malware|virus|ransomware|keylogger|trojan|rootkit|botnet|worm|spyware)\b"),
        re.compile(r"(?i)\b(?:malware|ransomware|keylogger|rootkit|botnet|trojan)\b"),
    ),
    "intrusion": (
        re.compile(r"(?i)\bhow\s+to\s+hack\b"),
        re.compile(r"(?i)\bhack(?:ing)?\s+(?:into\s+)?(?:amazon|flipkart|a\s+website|an?\s+account|the\s+server)\b"),
        re.compile(r"(?i)\b(?:exploit|breach|penetrate|compromise)\b[^.\n]{0,25}"
                   r"\b(?:server|database|account|system|website|network)\b"),
        re.compile(r"(?i)\b(?:sql\s+injection|xss|csrf|privilege\s+escalation|reverse\s+shell)\b"),
        re.compile(r"(?i)\b(?:ddos|denial\s+of\s+service)\b"),
        re.compile(r"(?i)\bbrute[\s-]?forc\w+\b"),
        re.compile(r"(?i)\b(?:crack|bypass|steal)\b[^.\n]{0,25}\b(?:password|login|auth|2fa|otp)\b"),
    ),
    "fraud": (
        re.compile(r"(?i)\b(?:fake|forge|forged|counterfeit|stolen)\b[^.\n]{0,25}"
                   r"\b(?:invoice|receipt|review|reviews|card|payment|coupon|identity)\b"),
        re.compile(r"(?i)\b(?:card|credit\s+card)\s+(?:generator|number\s+generator)\b"),
        re.compile(r"(?i)\bcarding\b|\bcvv\s+shop\b"),
        re.compile(r"(?i)\b(?:scam|defraud|launder)\b[^.\n]{0,25}\b(?:seller|buyer|marketplace|money)\b"),
        re.compile(r"(?i)\bhow\s+to\s+(?:get|obtain)\b[^.\n]{0,20}\bfor\s+free\b[^.\n]{0,20}\bwithout\s+paying\b"),
    ),
    "harm": (
        re.compile(r"(?i)\b(?:weapon|explosive|bomb|poison|drug\s+synthesis)\b"),
        re.compile(r"(?i)\bhow\s+to\s+(?:kill|harm|hurt)\b"),
    ),
    "unrelated_professional_advice": (
        re.compile(r"(?i)\b(?:medical|legal|tax|investment|financial)\s+advice\b"),
        re.compile(r"(?i)\b(?:diagnose|prescribe)\b[^.\n]{0,20}\b(?:symptom|illness|condition)\b"),
        # Financial-instrument requests. "stock" is deliberately NOT matched on
        # its own: "is it in stock" is core shopping vocabulary, so only
        # finance-specific constructions count.
        re.compile(r"(?i)\bstock\s+market\b|\bshare\s+market\b"),
        re.compile(r"(?i)\bstocks?\s+to\s+(?:buy|invest|pick)\b"),
        re.compile(r"(?i)\b(?:buy|invest\s+in|recommend|suggest|pick)\s+(?:me\s+)?"
                   r"(?:a|some|any)?\s*(?:stocks?|shares?|crypto\w*|bitcoin|mutual\s+funds?)\b"),
        re.compile(r"(?i)\b(?:which|what)\s+(?:stock|crypto\w*|share|mutual\s+fund)s?\b"),
        re.compile(r"(?i)\b(?:crypto\w*|bitcoin|ethereum|mutual\s+funds?|sip)\b"
                   r"[^.\n]{0,25}\b(?:buy|invest|portfolio|returns?)\b"),
    ),
}

#: Vocabulary that marks a request as being about laptop shopping.
ON_TOPIC_TERMS: Final[frozenset[str]] = frozenset(
    {
        # devices
        "laptop", "laptops", "notebook", "ultrabook", "macbook", "chromebook",
        "computer", "pc", "device", "machine", "workstation",
        # brands
        "dell", "hp", "lenovo", "asus", "acer", "apple", "msi", "samsung", "lg",
        "microsoft", "surface", "thinkpad", "ideapad", "vivobook", "zenbook",
        "inspiron", "latitude", "pavilion", "victus", "omen", "nitro", "aspire",
        "legion", "rog", "tuf", "swift", "spectre", "envy", "xps",
        # specs
        "ram", "memory", "gb", "tb", "ssd", "hdd", "storage", "processor", "cpu",
        "gpu", "graphics", "rtx", "gtx", "radeon", "intel", "amd", "ryzen", "core",
        "i3", "i5", "i7", "i9", "m1", "m2", "m3", "m4", "snapdragon",
        "screen", "display", "inch", "inches", "resolution", "oled", "ips",
        "refresh", "hz", "battery", "weight", "kg", "lightweight", "portable",
        "keyboard", "trackpad", "webcam", "thunderbolt", "usb", "hdmi", "port",
        "windows", "macos", "linux", "chromeos", "os",
        "touchscreen", "convertible", "2-in-1",
        # shopping
        "budget", "price", "prices", "pricing", "cost", "cheap", "cheaper",
        "cheapest", "expensive", "affordable", "under", "below", "within",
        "buy", "purchase", "order", "shop", "shopping", "deal", "deals",
        "discount", "discounts", "offer", "offers", "sale", "cashback",
        "coupon", "exchange", "emi", "warranty", "delivery",
        "amazon", "flipkart", "marketplace", "seller", "stock", "available",
        "compare", "comparison", "recommend", "recommendation", "suggest",
        "best", "good", "option", "options", "alternative", "review", "rating",
        "rupees", "inr", "usd", "dollar", "dollars", "lakh", "k",
        # use cases
        "gaming", "game", "games", "student", "college", "school", "study",
        "office", "work", "business", "productivity", "coding", "programming",
        "development", "developer", "engineering", "software",
        "data", "science", "ml", "ai", "editing", "video", "photo", "design",
        "rendering", "cad", "streaming", "travel", "home",
    }
)

#: Answers to clarifying questions that are legitimately terse.
_AFFIRMATIVE: Final[frozenset[str]] = frozenset(
    {"yes", "yeah", "yep", "yup", "no", "nope", "none", "sure", "ok", "okay",
     "correct", "right", "wrong", "any", "either", "both", "neither", "skip"}
)


def normalise_for_matching(text: str) -> str:
    """Canonical form used for pattern matching.

    Unicode-normalises (so ``ｉｇｎｏｒｅ`` and homoglyph tricks collapse to ASCII
    where possible), strips invisible characters, and flattens runs of
    punctuation used to break up keywords (``i-g-n-o-r-e``).
    """
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE_CHARS.sub("", text)
    # Collapse single-letter separations: "i.g.n.o.r.e" -> "ignore". Restricted
    # to letters so numeric values ("8.5 inch") are left intact.
    text = re.sub(r"(?<=\b[a-zA-Z])[.\-_*]+(?=[a-zA-Z]\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def matching_variants(text: str) -> tuple[str, ...]:
    """Every form the text must be screened against.

    Invisible characters are an evasion in two opposite directions: inserted
    *between* words (``ignore<ZWSP>all<ZWSP>previous``) they defeat matching if
    simply deleted, and inserted *inside* a word (``ig<ZWSP>nore``) they defeat
    matching if replaced by a space. Screening both forms closes both, at the
    cost of running each pattern twice.
    """
    variants = [
        normalise_for_matching(text),
        # Invisibles replaced by a space rather than deleted.
        normalise_for_matching(INVISIBLE_CHARS.sub(" ", text)),
        # Control characters removed — catches "ig\x00nore all previous ...".
        normalise_for_matching(CONTROL_CHARS.sub("", text)),
    ]
    unique: list[str] = []
    for variant in variants:
        if variant not in unique:
            unique.append(variant)
    return tuple(unique)


def has_letters(text: str) -> bool:
    return any(char.isalpha() for char in text)


def tokenise(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_terse_answer(text: str) -> bool:
    """Whether the text looks like a short answer to a clarifying question."""
    stripped = text.strip().lower().rstrip(".!")
    if not stripped:
        return False
    tokens = tokenise(stripped)
    if tokens & _AFFIRMATIVE:
        return True
    # Bare numbers / amounts: "80000", "80k", "16gb", "1.5 lakh", "₹75,000"
    return bool(re.fullmatch(r"[₹$]?\s*[\d,.]+\s*(?:k|gb|tb|lakh|inches?|kg|hours?|hrs?)?", stripped))
