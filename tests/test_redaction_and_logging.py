"""Secret/PII redaction and log hygiene."""

from __future__ import annotations

import json
import logging

import pytest

from laptop_agent.security.logging import RedactingJsonFormatter
from laptop_agent.security.redaction import MASK, SecretRedactor


@pytest.fixture
def redactor() -> SecretRedactor:
    return SecretRedactor(extra_literals=["literal-process-secret-value"])


SECRETS = [
    "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz123456",
    "lsv2_pt_abcdef1234567890abcdef1234567890",
    "AKIAIOSFODNN7EXAMPLE",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
    "literal-process-secret-value",
]


@pytest.mark.parametrize("secret", SECRETS)
def test_secrets_are_masked(redactor, secret):
    assert secret not in redactor.redact_text(f"the value is {secret} ok")


def test_card_numbers_are_masked(redactor):
    assert "4111" not in redactor.redact_text("card 4111 1111 1111 1111")
    assert "1234" not in redactor.redact_text("My HDFC card ends in 1234")
    assert "9876" not in redactor.redact_text("last 4 digits are 9876")


def test_card_eligibility_survives_redaction(redactor):
    """Only the digits go. The bank name is legitimately needed for offers."""
    redacted = redactor.redact_text("My HDFC card ends in 1234, budget 80000")
    assert "HDFC" in redacted
    assert "1234" not in redacted
    # A budget is not PII and must not be destroyed.
    assert "80000" in redacted


def test_pii_is_masked(redactor):
    text = "reach me at anand@example.com or 9876543210, PAN ABCDE1234F"
    redacted = redactor.redact_text(text)
    for value in ("anand@example.com", "9876543210", "ABCDE1234F"):
        assert value not in redacted


def test_legitimate_laptop_text_is_untouched(redactor):
    """False positives here would corrupt real requirements."""
    for text in (
        "I want 16GB RAM and 512GB SSD under 80000 rupees",
        "RTX 4060 with 165Hz display, 1.5 kg",
        "order 123456789012345678901 tracking",  # long but not a valid card
    ):
        assert redactor.redact_text(text) == text


def test_sensitive_keys_are_dropped_by_name(redactor):
    payload = {
        "api_key": "anything",
        "Authorization": "Bearer xyz",
        "cvv": "123",
        "nested": {"aws_secret_access_key": "s3cret", "ram_gb": 16},
    }
    redacted = redactor.redact(payload)
    assert redacted["api_key"] == MASK
    assert redacted["Authorization"] == MASK
    assert redacted["cvv"] == MASK
    assert redacted["nested"]["aws_secret_access_key"] == MASK
    assert redacted["nested"]["ram_gb"] == 16


def test_redaction_is_depth_bounded(redactor):
    deep: dict = {"level": 0}
    node = deep
    for index in range(1, 40):
        node["child"] = {"level": index}
        node = node["child"]
    # Must terminate rather than recurse without bound.
    assert redactor.redact(deep) is not None


def test_log_formatter_redacts_message_and_extras():
    formatter = RedactingJsonFormatter(SecretRedactor())
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="key is sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx", args=(), exc_info=None,
    )
    record.api_key = "sk-ant-api03-ZZZZZZZZZZZZZZZZZZZZZZZZ"
    record.session_id = "sess_abc"
    payload = json.loads(formatter.format(record))
    assert "sk-ant" not in json.dumps(payload)
    assert payload["api_key"] == MASK
    assert payload["session_id"] == "sess_abc"


def test_log_formatter_omits_locals_on_exception():
    formatter = RedactingJsonFormatter(SecretRedactor())
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert payload["error_type"] == "ValueError"
    # No traceback text, which can carry local variable values.
    assert "Traceback" not in json.dumps(payload)


def test_purchase_profile_cannot_hold_a_card_number():
    from laptop_agent.domain.requirements import PurchaseProfile

    fields = set(PurchaseProfile.model_fields)
    for forbidden in ("card_number", "card", "cvv", "expiry", "pan", "account_number"):
        assert forbidden not in fields
    # And unexpected fields are refused outright.
    with pytest.raises(Exception):
        PurchaseProfile(card_number="4111111111111111")


# ---------------------------------------------------------------------------
# Regression: real key shapes, not synthetic ones.
#
# The original patterns were tested against invented keys with no internal
# underscore, and passed. A real LangSmith key is
# lsv2_pt_<32 hex>_<10 hex> — and because "_" is a word character, the trailing
# \b never matched after the first segment, so the ENTIRE key was emitted
# unmasked. The same flaw hid a second bug: \b before "api_key" does not match
# inside "LANGSMITH_API_KEY", so the key=value rule missed it too.
#
# Keys below are structurally identical to real ones but not real.
# ---------------------------------------------------------------------------

# Non-hex filler on purpose. The property under test is the *internal
# underscore* and the segment lengths, not the alphabet — and hex-shaped
# fixtures are realistic enough that GitHub push protection classifies them as
# live credentials and refuses the push. Keeping them obviously synthetic tests
# the same bug without tripping a scanner or teaching anyone to click through it.
_LS_SEG_A = "notARealKeyNotARealKeyNotARealK"
_LS_SEG_B = "synthetic1"

REALISTIC_KEYS = [
    # LangSmith: two underscore-separated segments after the prefix.
    f"lsv2_pt_{_LS_SEG_A}_{_LS_SEG_B}",
    f"lsv2_sk_{_LS_SEG_A}_{_LS_SEG_B}",
    # OpenRouter: hyphenated prefix then a long hex run.
    "sk-or-v1-" + "9f8e7d6c" * 8,
    # Anthropic.
    "sk-ant-api03-" + "AbCdEfGh" * 4,
    # SerpApi: bare 64-char hex with no prefix at all.
    # (Caught by the literal list from Settings, not by shape — see below.)
]


@pytest.mark.parametrize("key", REALISTIC_KEYS)
def test_realistic_key_shapes_are_fully_masked(redactor, key):
    for template in ("{k}", "key is {k} ok", "LANGSMITH_API_KEY={k}", '"api_key": "{k}"'):
        out = redactor.redact_text(template.format(k=key))
        assert key not in out, f"whole key survived in {template!r}"
        # No fragment longer than the prefix may survive either.
        for fragment in key.replace("-", "_").split("_"):
            if len(fragment) >= 8:
                assert fragment not in out, f"fragment {fragment!r} survived in {template!r}"


@pytest.mark.parametrize(
    "name",
    [
        "LANGSMITH_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "x-api-key",
        "X_API_TOKEN",
        "db_password",
    ],
)
def test_prefixed_env_var_names_are_matched(redactor, name):
    """A bare \\b before "api_key" never matches inside "LANGSMITH_API_KEY"."""
    out = redactor.redact_text(f"{name}=s3cr3tvalue123456")
    assert "s3cr3tvalue123456" not in out


def test_bare_hex_secret_is_masked_via_configured_literals():
    """A 64-char hex SerpApi key has no distinguishing shape.

    Shape-based patterns cannot catch it without also masking order numbers and
    hashes, so it is covered by seeding the redactor with the process's own
    configured secrets instead.
    """
    from laptop_agent.security.redaction import SecretRedactor

    key = "5785e8f0" * 8
    assert key not in SecretRedactor(extra_literals=[key]).redact_text(f"SERPAPI_KEY={key}")
    # And the key=value rule catches it even without the literal.
    assert key not in SecretRedactor().redact_text(f"SERPAPI_KEY={key}")


def test_legitimate_text_survives_the_broadened_patterns(redactor):
    """The widened name pattern must not start eating ordinary prose."""
    for text in (
        "I want 16GB RAM and 512GB SSD under 80000 rupees",
        "the token bucket rate limiter allows 20 requests",
        "my budget is 80000",
    ):
        assert redactor.redact_text(text) == text
