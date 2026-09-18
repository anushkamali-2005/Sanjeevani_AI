"""
Test script for conversational intelligence hardening in Darukaa.Earth.
Tests:
1. Contradiction / correction handling
2. Partial / ambiguous input (greetings / help)
3. Early recommendation requests (provisional low confidence)
4. Multi-variable single message slot extraction
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
from backend.app.conversation import process_message, _sessions


def test_case_1_contradiction():
    print("\n--- TEST CASE 1: Contradiction / Slot Correction ---")
    session_id = "test_contradiction"
    _sessions.pop(session_id, None)

    # Turn 1: Set initial parameters
    r1 = process_message(session_id, "I have a wheat farm in Rajasthan with low rainfall and 0.3% soil carbon.")
    print(f"Turn 1 Response: {r1.response_text[:120]}...")
    assert r1.slots.rainfall_pattern == "low", f"Expected low rainfall, got {r1.slots.rainfall_pattern}"
    assert r1.recommendation is not None, "Expected initial recommendation"

    # Turn 2: User corrects rainfall
    r2 = process_message(session_id, "Actually scratch that, rainfall is moderate here due to a new canal, not low.")
    print(f"Turn 2 Response: {r2.response_text[:150]}...")
    print(f"Updated Slot Notice: {r2.slot_updated}")
    assert "moderate" in str(r2.slots.rainfall_pattern).lower(), f"Expected moderate rainfall, got {r2.slots.rainfall_pattern}"
    assert r2.slot_updated is not None or "updating" in r2.response_text.lower(), "Expected explicit update notice in response"
    assert r2.recommendation is not None, "Expected revised recommendation"
    print("[PASS] Case 1: Contradiction handled and recommendation revised explicitly.")


def test_case_2_ambiguous_greeting():
    print("\n--- TEST CASE 2: Partial / Ambiguous Input ---")
    session_id = "test_greeting"
    _sessions.pop(session_id, None)

    r = process_message(session_id, "hello! what can you do?")
    print(f"Response: {r.response_text[:180]}...")
    assert "Welcome to **Darukaa.Earth**" in r.response_text or "biodiversity" in r.response_text.lower(), "Expected onboarding message"
    assert r.recommendation is None, "Should not return a recommendation for a pure greeting"
    print("[PASS] Case 2: Greeting routed to helpful guidance without error.")


def test_case_3_early_recommendation():
    print("\n--- TEST CASE 3: Early Recommendation Request ---")
    session_id = "test_early"
    _sessions.pop(session_id, None)

    # Provide only 1-2 variables but explicitly ask what to do
    r = process_message(session_id, "I have monoculture wheat in a semi-arid zone. What should I do?")
    print(f"Response: {r.response_text[:180]}...")
    assert r.recommendation is not None, "Expected provisional recommendation"
    assert r.recommendation.confidence.value == "low", f"Expected low confidence for provisional rec, got {r.recommendation.confidence}"
    assert len(r.missing_slots) > 0, "Expected missing mandatory slots to be requested"
    print("[PASS] Case 3: Provisional low-confidence recommendation generated with missing slot request.")


def test_case_4_multivariable_single_message():
    print("\n--- TEST CASE 4: Multi-Variable Single Message ---")
    session_id = "test_multivar"
    _sessions.pop(session_id, None)

    message = "I manage a 50 hectare monoculture wheat farm in Rajasthan with 0.3 percent soil carbon, low rainfall, dry soil moisture, and low species richness."
    r = process_message(session_id, message)
    print(f"Response: {r.response_text[:150]}...")
    print(f"Extracted Slots: {r.slots.model_dump(exclude_none=True)}")

    filled = [k for k, v in r.slots.model_dump(exclude_none=True).items() if k not in ["geo_lat", "geo_lon"]]
    print(f"Total slots filled in single turn: {len(filled)}")
    assert len(filled) >= 4, f"Expected at least 4 slots extracted, got {len(filled)}"
    assert r.recommendation is not None, "Expected full recommendation"
    assert r.recommendation.overlap_count >= 2, f"Expected overlap_count >= 2, got {r.recommendation.overlap_count}"
    print("[PASS] Case 4: 4+ variables extracted simultaneously in a single turn.")


def test_case_5_different_messages_different_responses():
    print("\n--- TEST CASE 5: Different Messages Produce Different Responses ---")
    session_id = "test_diff_responses"
    _sessions.pop(session_id, None)

    # Message 1: Greeting
    r1 = process_message(session_id, "hlo")
    print(f"Turn 1 (hlo): {r1.response_text[:80]}...")
    assert "Welcome to **Darukaa.Earth**" in r1.response_text

    # Message 2: Free-form non-slot query
    r2 = process_message(session_id, "tell me about kharghar navi mumbai")
    print(f"Turn 2 (kharghar navi mumbai): {r2.response_text[:80]}...")

    assert r1.response_text != r2.response_text, "CRITICAL: Consecutive messages produced identical responses!"
    assert "kharghar navi mumbai" in r2.response_text.lower()
    assert "Welcome to **Darukaa.Earth**" not in r2.response_text
    print("[PASS] Case 5: Different messages produced completely distinct, context-aware responses.")


def test_case_6_session_persistence():
    print("\n--- TEST CASE 6: Session State Persistence Across Turns ---")
    session_id = "test_persistence"
    _sessions.pop(session_id, None)

    # Message 1
    r1 = process_message(session_id, "I farm monoculture wheat.")
    assert _sessions[session_id]["slots"].get("land_use_type") is not None

    # Message 2
    r2 = process_message(session_id, "Rainfall is low.")
    assert _sessions[session_id]["slots"].get("land_use_type") is not None, "Prior slot lost"
    assert _sessions[session_id]["slots"].get("rainfall_pattern") == "low", "New slot missing"
    print("[PASS] Case 6: Session state accumulated correctly across turns.")


if __name__ == "__main__":
    print("Running Conversation Intelligence Hardening Test Suite...")
    try:
        test_case_1_contradiction()
        test_case_2_ambiguous_greeting()
        test_case_3_early_recommendation()
        test_case_4_multivariable_single_message()
        test_case_5_different_messages_different_responses()
        test_case_6_session_persistence()
        print("\n==========================================")
        print("ALL 6 CONVERSATION HARDENING TESTS PASSED!")
        print("==========================================")
    except Exception as e:
        print(f"\n[FAIL] Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
