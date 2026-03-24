from __future__ import annotations

import html
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
USER_AGENT = "CarPriceDemo/1.0 (used-car recommendation demo)"


def _fetch_json(url: str) -> dict | None:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _thumbnail_from_page(page: dict) -> str | None:
    thumb = page.get("thumbnail") or {}
    url = thumb.get("url")
    if not url:
        return None
    if url.startswith("//"):
        url = "https:" + url
    return url


def fetch_image_bytes(url: str) -> bytes | None:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/*",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.read()
    except Exception:
        return None


def get_vehicle_image(title: str, typical_year: int | None = None) -> str | None:
    """
    Fetch a representative vehicle image from Wikipedia search results.

    The function tries a year-specific query first, then falls back to the
    make/model title alone. It returns the first thumbnail URL it finds.
    """
    queries = []
    if typical_year:
        queries.append("{} {}".format(typical_year, title))
    queries.append(title)

    for query in queries:
        search_url = "{}?{}".format(
            WIKIPEDIA_SEARCH_URL,
            urlencode({"q": query, "limit": 10}),
        )
        payload = _fetch_json(search_url)
        if not payload:
            continue

        pages = payload.get("pages") or []
        for page in pages:
            thumb_url = _thumbnail_from_page(page)
            if thumb_url:
                return thumb_url

    return None


def render_vehicle_card_html(
    image_url: str | None,
    title: str,
    body_text: list[str],
    confidence: str,
) -> str:
    safe_title = html.escape(title)
    safe_confidence = html.escape(confidence)
    safe_lines = "".join(
        '<p class="car-card-meta">{}</p>'.format(html.escape(line)) for line in body_text
    )

    image_html = (
        '<img src="{}" alt="{}" class="car-card-image" />'.format(html.escape(image_url), safe_title)
        if image_url
        else '<div class="car-card-image car-card-placeholder">Image unavailable</div>'
    )

    return """
    <div class="car-card">
      <div class="car-card-media">{}</div>
      <div class="car-card-content">
        <div class="car-card-header">
          <h4>{}</h4>
          <span class="car-card-badge">{}</span>
        </div>
        {}
      </div>
    </div>
    """.format(image_html, safe_title, safe_confidence, safe_lines)
