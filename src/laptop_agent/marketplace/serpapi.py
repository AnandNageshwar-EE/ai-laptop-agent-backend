"""Live marketplace data via SerpApi.

Two engines, because the two marketplaces are not equally reachable:

* **Amazon.in** — SerpApi's ``amazon`` engine with ``amazon_domain=amazon.in``
  returns real ASINs, real prices and real ``/dp/<ASIN>`` links.
* **Flipkart** — there is no Flipkart engine, and the obvious substitute does
  not work. ``google_shopping`` *does* return Flipkart rows with correct prices,
  but their only link is a Google catalog page, so there is no ``flipkart.com``
  URL for the provenance check to verify. Those rows are therefore useless here.

  What works is the ``google`` engine with a ``site:flipkart.com`` query: organic
  results carry real ``flipkart.com/<slug>/p/<id>`` product URLs, and roughly two
  thirds include the price in the snippet. Rows without a parseable price are
  dropped — a listing whose price cannot be established is of no use to an agent
  whose entire job is comparing what you would pay.

This class produces **raw payloads only**, in the same envelope shape the fixture
transport produces. Every field then passes through
:class:`~laptop_agent.guardrails.tool_output.MarketplaceResponseValidator`
unchanged — which is the point: real third-party data is exactly what the
validation layer was built for, and none of it is trusted here.

Failure is never fatal. A missing key, a timeout, a rate limit or a malformed
response returns an empty result set; the graph already tolerates one provider
returning nothing.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from urllib.parse import urlparse

from ..domain.enums import Marketplace
from ..guardrails.tool_input import SearchProductsRequest
from ..security.logging import get_logger
from .spec_parser import parse_specs

_logger = get_logger("laptop_agent.serpapi")

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

#: Hosts a result must be on to be attributed to a marketplace. The
#: tool-output validator enforces this again; this is an early filter to avoid
#: shipping obvious rejects downstream.
_HOST_HINTS: dict[Marketplace, tuple[str, ...]] = {
    Marketplace.AMAZON: ("amazon.in", "amazon.com"),
    Marketplace.FLIPKART: ("flipkart.com",),
}

#: Prices arrive as display strings such as "₹56,490" or "Rs. 56,490.00".
_PRICE = re.compile(r"[\d,]+(?:\.\d{1,2})?")

#: A URL is only a usable product link if it points at a product *page*.
#: Being on the right host is not enough: a sponsored ``/sspa/click`` redirect
#: lives on amazon.in but resolves through an ad tracker, and surfacing it as
#: "the product link" would be wrong even though the host check passes.
_PRODUCT_PATHS: dict[Marketplace, tuple[str, ...]] = {
    Marketplace.AMAZON: ("/dp/", "/gp/product/"),
    Marketplace.FLIPKART: ("/p/",),
}
_REDIRECT_PATHS: tuple[str, ...] = ("/sspa/", "/gp/slredirect", "/url?", "/aclk")

#: Accessories and non-laptops that pollute a laptop search.
_NOT_A_LAPTOP = re.compile(
    r"\b(?:sleeve|case|cover|bag|backpack|skin|sticker|screen\s+guard|"
    r"charger|adapter|battery\s+for|keyboard\s+cover|stand|cooling\s+pad|"
    r"docking\s+station|ram\s+module|ssd\s+upgrade|hard\s+drive|mouse)\b",
    re.IGNORECASE,
)


class SerpApiTransport:
    """Fetches raw listing payloads from SerpApi."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 20.0,
        country: str = "in",
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._country = country

    # ------------------------------------------------------------------

    def fetch_products(
        self, marketplace: Marketplace, request: SearchProductsRequest
    ) -> dict[str, Any]:
        """Return a raw, unvalidated product payload."""
        try:
            if marketplace is Marketplace.AMAZON:
                raw = self._call(
                    {
                        "engine": "amazon",
                        "amazon_domain": "amazon.in",
                        "k": request.query,
                    }
                )
                products = self._map_amazon(raw, request.max_results)
            else:
                raw = self._call(
                    {
                        "engine": "google",
                        "q": f"site:flipkart.com {request.query}",
                        "gl": self._country,
                        "hl": "en",
                        "num": "20",
                    }
                )
                products = self._map_flipkart_organic(raw, request.max_results)
        except Exception as exc:
            # Observability, not propagation: one provider failing must not fail
            # the search.
            _logger.warning(
                "serpapi.fetch_failed",
                extra={
                    "marketplace": marketplace.value,
                    "error_type": type(exc).__name__,
                },
            )
            return {"marketplace": marketplace.value, "source": "serpapi", "products": []}

        return {
            "marketplace": marketplace.value,
            "source": "serpapi",
            "products": products,
        }

    # ------------------------------------------------------------------

    def _call(self, params: dict[str, str]) -> dict[str, Any]:
        query = urllib.parse.urlencode({**params, "api_key": self._api_key})
        url = f"{SERPAPI_ENDPOINT}?{query}"
        request = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": "ai-laptop-agent/1.0"}
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("SerpApi returned an unexpected top-level type")
        if payload.get("error"):
            # Never log the message verbatim — it can echo the request, which
            # contains the API key.
            raise RuntimeError("SerpApi reported an error")
        return payload

    # ------------------------------------------------------------------

    def _map_amazon(self, raw: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        results = raw.get("organic_results")
        if not isinstance(results, list):
            return []

        products: list[dict[str, Any]] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            asin = _clean_id(row.get("asin"))
            title = _text(row.get("title"))
            if not asin or not title or _NOT_A_LAPTOP.search(title):
                continue

            amount = _price(row.get("price"), row.get("extracted_price"))
            if amount is None:
                continue

            # Always canonicalise. A live result's link carries the search
            # session (`/ref=sr_1_17?dib=...`), and sponsored rows are ad
            # redirects; neither belongs in a recommendation. The ASIN is the
            # stable identity, so the URL is rebuilt from it.
            link = f"https://www.amazon.in/dp/{asin}"

            products.append(
                _payload(
                    product_id=f"AMZ-{asin}",
                    title=title,
                    url=link,
                    amount=amount,
                    mrp=_price(row.get("old_price"), row.get("extracted_old_price")),
                    rating=_rating(row.get("rating")),
                    rating_count=_count(row.get("reviews")),
                )
            )
            if len(products) >= limit:
                break
        return products

    def _map_flipkart_organic(self, raw: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        """Map organic Flipkart results, keeping only rows we can price."""
        results = raw.get("organic_results")
        if not isinstance(results, list):
            return []

        products: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in results:
            if not isinstance(row, dict):
                continue

            link = _text(row.get("link"))
            if not _is_product_url(link, Marketplace.FLIPKART):
                # Category and listing pages are not products.
                continue

            title = _clean_title(_text(row.get("title")))
            if not title or _NOT_A_LAPTOP.search(title):
                continue

            # Price appears in the snippet or the rich snippet, not a dedicated
            # field. No price means the row cannot be compared, so it is dropped.
            haystack = " ".join(
                (
                    _text(row.get("snippet")),
                    json.dumps(row.get("rich_snippet") or {}, ensure_ascii=False),
                    json.dumps(row.get("snippet_highlighted_words") or [], ensure_ascii=False),
                )
            )
            amount = _rupees(haystack)
            if amount is None:
                continue

            identifier = _clean_id(_flipkart_id(link))
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)

            products.append(
                _payload(
                    product_id=f"FK-{identifier}",
                    # The URL slug carries the specs Flipkart states, and the
                    # snippet carries the rest. Both are untrusted text parsed by
                    # regex only.
                    title=title,
                    url=link,
                    amount=amount,
                    mrp=None,
                    rating=_rating_from_text(haystack),
                    rating_count=0,
                    extra_spec_text=f"{_slug_words(link)} {haystack}",
                )
            )
            if len(products) >= limit:
                break
        return products


# ---------------------------------------------------------------------------
# mapping helpers
# ---------------------------------------------------------------------------


def _payload(
    *,
    product_id: str,
    title: str,
    url: str,
    amount: str,
    mrp: str | None,
    rating: float | None,
    rating_count: int,
    extra_spec_text: str = "",
) -> dict[str, Any]:
    """Build the raw envelope. Specs are parsed from the text, never invented."""
    specs = parse_specs(title, extra_spec_text)
    return {
        "product_id": product_id,
        "title": title[:300],
        "brand": title.strip().split(" ", 1)[0][:32],
        "url": url,
        "price": {"amount": amount, "currency": "INR"},
        **({"mrp": {"amount": mrp, "currency": "INR"}} if mrp else {}),
        "rating": rating,
        "rating_count": rating_count,
        "in_stock": True,
        "description": "",
        # Dumped rather than passed as an object: the validator re-parses this
        # like any other untrusted field.
        "specs": specs.model_dump(exclude_none=True),
    }


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_title(title: str) -> str:
    """Reduce a search-engine page title to the product name.

    Marketplace page titles are wrapped for SEO — "Buy X ... - Flipkart.com" —
    and that wrapping is not part of the product's name.
    """
    title = re.sub(r"(?i)^\s*buy\s+", "", title)
    title = re.sub(r"(?i)\s*[-|]\s*(?:Buy\s+)?(?:Flipkart|Amazon)\.?\w*\s*$", "", title)
    title = re.sub(r"(?i)\s*\|\s*Price in India.*$", "", title)
    title = re.sub(r"(?i)\s+online\s+at\s+(?:best\s+)?price.*$", "", title)
    return title.strip(" -|")


def _slug_words(url: str) -> str:
    """The URL slug, which on Flipkart spells out the specifications."""
    path = urlparse(url).path
    slug = path.split("/p/")[0].strip("/").split("/")[-1]
    return slug.replace("-", " ")


def _rupees(text: str) -> str | None:
    """First rupee amount plausible as a laptop price."""
    for match in re.finditer(r"(?:₹|\bRs\.?\s?)\s?([\d,]{4,})", text):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if 5_000 <= value <= 1_000_000:
            return f"{value:.2f}"
    return None


def _rating_from_text(text: str) -> float | None:
    """A rating stated as "4.3 out of 5" or "Rating: 4.3"."""
    match = re.search(r"\b([0-5](?:\.\d)?)\s*(?:out of 5|/\s*5|stars?)\b", text, re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if 0 <= value <= 5 else None


def _clean_id(value: Any) -> str:
    """Reduce an external identifier to the characters the domain pattern allows."""
    text = _text(value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", text)
    return cleaned[:56] if len(cleaned) >= 4 else ""


def _flipkart_id(url: str) -> str:
    """Extract Flipkart's ``pid`` or path slug id from a product URL."""
    parsed = urlparse(url)
    pid = urllib.parse.parse_qs(parsed.query).get("pid", [""])[0]
    if pid:
        return pid
    match = re.search(r"/p/(itm[\w]+)", parsed.path)
    return match.group(1) if match else ""


def _price(display: Any, extracted: Any) -> str | None:
    """Prefer the numeric field; fall back to parsing the display string."""
    if isinstance(extracted, (int, float)) and extracted > 0:
        return f"{float(extracted):.2f}"
    match = _PRICE.search(_text(display))
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return f"{value:.2f}" if value > 0 else None


def _rating(value: Any) -> float | None:
    if isinstance(value, (int, float)) and 0 <= value <= 5:
        return float(value)
    return None


def _count(value: Any) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    digits = re.sub(r"[^\d]", "", _text(value))
    return int(digits) if digits else 0


def _host_allowed(url: str, marketplace: Marketplace) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return any(host.endswith(hint) for hint in _HOST_HINTS[marketplace])


def _is_product_url(url: str, marketplace: Marketplace) -> bool:
    """Right host, product path, and not a redirect."""
    if not _host_allowed(url, marketplace):
        return False
    parsed = urlparse(url)
    full = f"{parsed.path}?{parsed.query}"
    if any(marker in full for marker in _REDIRECT_PATHS):
        return False
    return any(marker in parsed.path for marker in _PRODUCT_PATHS[marketplace])
