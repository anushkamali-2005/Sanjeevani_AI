"""
tests/test_conversation.py — Regression prevention test suite for conversation engine.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from backend.app.conversation import process_message, _sessions


def test_different_messages_produce_different_responses():
    """
    Send two clearly different messages (one greeting, one unrelated free-text question)
    in the SAME session and assert the response text/content is NOT identical.
    """
    session_id = "test_diff_responses"
    _sessions.pop(session_id, None)

    # Message 1: Greeting
    r1 = process_message(session_id, "hlo")
    assert r1.session_id == session_id
    assert "Welcome to **Darukaa.Earth**" in r1.response_text

    # Message 2: Unrelated free-text question without environmental slots
    r2 = process_message(session_id, "tell me about kharghar navi mumbai")
    assert r2.session_id == session_id
    
    # Assert response texts are strictly NOT identical
    assert r1.response_text != r2.response_text, "CRITICAL: Consecutive messages produced identical responses!"
    assert "kharghar" in r2.response_text.lower(), "Expected response to acknowledge the user's inquiry"
    assert "Welcome to **Darukaa.Earth**" not in r2.response_text, "Welcome message should not repeat on follow-up question"


def test_session_id_persists_across_messages():
    """
    Send two messages with the same session_id, assert the second response reflects
    awareness of the first message (e.g. slot count increases, or the response references prior context)
    rather than resetting.
    """
    session_id = "test_persistence"
    _sessions.pop(session_id, None)

    # Message 1: Introduce land use
    r1 = process_message(session_id, "I farm monoculture wheat on 20 hectares.")
    assert r1.session_id == session_id
    assert r1.slots.land_use_type is not None or "monoculture" in str(r1.slots.land_use_type).lower()

    # Message 2: Add rainfall information
    r2 = process_message(session_id, "Our rainfall is low and mostly rainfed.")
    assert r2.session_id == session_id

    # The session must retain land_use_type from Message 1 AND have rainfall_pattern from Message 2
    session_slots = _sessions[session_id]["slots"]
    assert session_slots.get("land_use_type") is not None, "Prior slot 'land_use_type' was lost across turns!"
    assert session_slots.get("rainfall_pattern") == "low", "New slot 'rainfall_pattern' was not added!"
    assert len([k for k, v in session_slots.items() if v is not None]) >= 2, "Slot count did not accumulate across turns!"


if __name__ == "__main__":
    test_different_messages_produce_different_responses()
    test_session_id_persists_across_messages()
    print("All conversation regression tests passed successfully!")
