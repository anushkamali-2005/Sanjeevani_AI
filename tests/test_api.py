"""
tests/test_api.py — End-to-end API regression prevention test suite using FastAPI TestClient.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))
from backend.app.main import app
from backend.app.conversation import _sessions


client = TestClient(app)


def test_different_messages_produce_different_responses():
    """
    API End-to-End Test: Send two clearly different messages in the SAME session
    via POST /chat and assert the response text is NOT identical.
    """
    # Turn 1: Greeting
    res1 = client.post("/chat", json={"session_id": None, "message": "hlo"})
    assert res1.status_code == 200
    data1 = res1.json()
    session_id = data1["session_id"]
    assert session_id is not None
    assert "Welcome to **Darukaa.Earth**" in data1["response_text"]

    # Turn 2: Unrelated free-text question using the same session_id
    res2 = client.post("/chat", json={"session_id": session_id, "message": "tell me about kharghar navi mumbai"})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["session_id"] == session_id

    # Responses must be completely distinct
    assert data1["response_text"] != data2["response_text"], "CRITICAL: Consecutive API calls produced identical responses!"
    assert len(data2["response_text"]) > 20, "Response text is unexpectedly short"
    assert "Welcome to **Darukaa.Earth**" not in data2["response_text"]


def test_session_id_persists_across_messages():
    """
    API End-to-End Test: Send two messages with the same session_id, assert the second
    response reflects awareness of the first message rather than resetting.
    """
    # Message 1
    res1 = client.post("/chat", json={"session_id": None, "message": "I farm monoculture wheat."})
    assert res1.status_code == 200
    data1 = res1.json()
    session_id = data1["session_id"]

    # Message 2
    res2 = client.post("/chat", json={"session_id": session_id, "message": "Our rainfall is low."})
    assert res2.status_code == 200
    data2 = res2.json()

    # Verify session slots accumulated in the backend store
    assert session_id in _sessions
    stored_slots = _sessions[session_id]["slots"]
    assert stored_slots.get("land_use_type") is not None, "Slot from message 1 lost in session"
    assert stored_slots.get("rainfall_pattern") == "low", "Slot from message 2 not added to session"


def test_message_actually_reaches_backend():
    """
    End-to-end TestClient test: Sends a message containing a known extractable slot value
    ('rainfall is low'), and asserts that value appears in the session's stored slots afterward.
    Hard proof the pipeline from HTTP request to slot extraction actually works.
    """
    unique_session = "verify_pipeline_reach"
    _sessions.pop(unique_session, None)

    res = client.post("/chat", json={"session_id": unique_session, "message": "Our rainfall is low in this district."})
    assert res.status_code == 200
    data = res.json()

    assert data["session_id"] == unique_session
    # Confirm it extracted and stored in session slots
    assert unique_session in _sessions, "Session was not stored in backend"
    stored_slots = _sessions[unique_session]["slots"]
    assert stored_slots.get("rainfall_pattern") == "low", f"Expected 'low' rainfall, got: {stored_slots.get('rainfall_pattern')}"
    # Confirm it also returned in the response slots object
    assert data["slots"]["rainfall_pattern"] == "low"


if __name__ == "__main__":
    test_different_messages_produce_different_responses()
    test_session_id_persists_across_messages()
    test_message_actually_reaches_backend()
    print("All API regression tests passed successfully!")
