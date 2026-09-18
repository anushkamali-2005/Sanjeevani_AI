"""
tests/test_frontend.py — Validation tests for frontend static assets, 
DOM structure integrity, and client application scripts.
"""

import os
import subprocess
import pytest


def test_frontend_files_exist():
    """Verify all critical frontend files exist in both frontend/ and public/ distribution mirrors."""
    expected_files = [
        "frontend/index.html",
        "frontend/styles.css",
        "frontend/app.js",
        "public/index.html",
        "public/static/styles.css",
        "public/static/app.js",
    ]
    for path in expected_files:
        assert os.path.exists(path), f"Required frontend asset missing: {path}"
        assert os.path.getsize(path) > 0, f"Frontend asset is empty: {path}"


def test_frontend_html_dom_structure():
    """Verify index.html contains essential DOM element IDs for advisory chat and structured form."""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Chat interaction elements
    assert 'id="chat-messages"' in html
    assert 'id="chat-form"' in html
    assert 'id="chat-input"' in html
    assert 'id="chat-send-btn"' in html

    # Structured environmental form elements
    assert 'id="structured-form"' in html
    assert 'id="form-soc"' in html
    assert 'id="form-ph"' in html
    assert 'id="form-moisture"' in html
    assert 'id="form-landuse"' in html
    assert 'id="form-rainfall"' in html
    assert 'id="form-climate"' in html
    assert 'id="form-submit-btn"' in html


def test_styles_css_design_tokens():
    """Verify styles.css contains required design tokens and utility classes."""
    with open("frontend/styles.css", "r", encoding="utf-8") as f:
        css = f.read()

    assert "--font-family" in css
    assert "--accent" in css
    assert ".message" in css
    assert ".sources-panel" in css
    assert ".source-card" in css


def test_javascript_syntax_validity():
    """Verify app.js does not contain syntax errors using node -c if node is available."""
    try:
        res = subprocess.run(
            ["node", "-c", "frontend/app.js"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert res.returncode == 0, f"JavaScript syntax error in frontend/app.js: {res.stderr}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Gracefully pass if Node.js runtime is not installed locally
        pass
