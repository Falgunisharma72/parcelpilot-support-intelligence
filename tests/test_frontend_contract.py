"""Static contract between index.html, app.js and styles.css.

There is no browser in this environment, so the UI cannot be exercised. These
checks cover the failure modes that a page-load would have caught immediately
and that nothing else in the suite would: a script reaching for an element that
does not exist, a class applied with no rule behind it, and a template
referencing a script that was never written.

This is not a substitute for looking at the page. It is the part of "looking at
the page" that can be automated, and it already caught a live bug - app.js
guarded `$("#send")` with a truthiness check, so a missing id failed silently
and the send button never disabled while a turn was streaming.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
HTML = (STATIC / "index.html").read_text()
JS = (STATIC / "app.js").read_text()
CSS = (STATIC / "styles.css").read_text()

HTML_IDS = set(re.findall(r'id="([^"]+)"', HTML))
JS_IDS = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', JS))
CSS_CLASSES = set(re.findall(r"\.([a-z][A-Za-z0-9_-]*)", CSS))
JS_CLASSES = {
    cls
    for group in re.findall(r'el\("[a-z]+",\s*"([^"]+)"', JS)
    for cls in group.split()
    if cls and "$" not in cls
}


def test_every_element_the_script_looks_up_exists():
    missing = sorted(JS_IDS - HTML_IDS)
    assert not missing, f"app.js references ids absent from index.html: {missing}"


def test_every_declared_id_is_actually_used():
    """A stale id is dead markup, and usually the leftover of a rename that only
    got applied on one side."""
    unused = sorted(HTML_IDS - JS_IDS)
    assert not unused, f"index.html declares ids nothing uses: {unused}"


def test_every_class_the_script_applies_has_a_rule():
    unstyled = sorted(JS_CLASSES - CSS_CLASSES)
    assert not unstyled, f"app.js applies classes with no CSS rule: {unstyled}"


@pytest.mark.parametrize("asset", ["/static/styles.css", "/static/app.js"])
def test_referenced_assets_exist(asset):
    assert asset in HTML
    assert (STATIC / asset.split("/")[-1]).exists()


def test_model_output_is_never_inserted_as_html():
    """Model text goes through text nodes only. An innerHTML assignment on
    streamed content would be a cross-site scripting hole fed by whatever a
    ticket description happens to contain."""
    assignments = re.findall(r"(\w+)\.innerHTML\s*=\s*([^;]+);", JS)
    for target, value in assignments:
        assert value.strip() in ('""', "''"), (
            f"{target}.innerHTML is assigned {value.strip()!r}; only clearing is "
            "allowed - render text with textContent or createTextNode")


def test_the_composer_is_disabled_while_a_turn_streams():
    assert "setBusy(true)" in JS and "setBusy(false)" in JS
    assert "button.disabled = busy" in JS
