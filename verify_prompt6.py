"""
Verification script for Prompt 6 checks 2 and 3.
Uses FastAPI TestClient to test /chat in-process.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("=" * 70)
print("  PROMPT 6 VERIFICATION TEST")
print("=" * 70)

# ---------------------------------------------------------------------------
# Check 2: /chat asking about soil carbon in a specific Indian state
# ---------------------------------------------------------------------------
print("\n[*] RUNNING CHECK 2: Soil carbon in an Indian state (Maharashtra)...")
res2 = client.post(
    "/chat",
    json={"session_id": "verify_check2", "message": "How to improve soil carbon in Maharashtra with low rainfall?"}
)

assert res2.status_code == 200, f"Check 2 failed with status {res2.status_code}: {res2.text}"
data2 = res2.json()

print(f"Response Status: {res2.status_code}")
print(f"Response Recommendation: {data2.get('recommendation', {}).get('recommendation') if data2.get('recommendation') else 'None'}")
print(f"Total Sources Returned: {len(data2.get('sources', []))}")

sources2 = data2.get("sources", [])
doc_sources = [s for s in sources2 if s.get("type") == "document"]
csv_sources = [s for s in sources2 if s.get("type") == "structured_data"]
web_sources = [s for s in sources2 if s.get("type") == "web_search"]

print(f"  - Document sources (IPCC): {len(doc_sources)}")
for d in doc_sources[:3]:
    print(f"    • [{d.get('title')}] Page {d.get('page')}: \"{d.get('excerpt', '')[:90]}...\"")

print(f"  - Structured CSV sources: {len(csv_sources)}")
for c in csv_sources:
    print(f"    • [{c.get('file')}]: {c.get('detail')}")

print(f"  - Web sources (should be 0): {len(web_sources)}")

assert len(doc_sources) >= 1, "CHECK 2 FAILED: Expected at least one document citation"
assert doc_sources[0].get("page") is not None, "CHECK 2 FAILED: Document source must have page number"
assert len(csv_sources) >= 1, "CHECK 2 FAILED: Expected at least one structured_data CSV citation"
assert len(web_sources) == 0, "CHECK 2 FAILED: Web search should NOT trigger for normal ecological query"
print("\n[PASSED] CHECK 2: Found real IPCC document+page citations AND structured CSV dataset citations!")

# ---------------------------------------------------------------------------
# Check 3: /chat with obviously time-sensitive phrase ("what's the weather like today near Kharghar")
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("[*] RUNNING CHECK 3: Time-sensitive live query ('what's the weather like today near Kharghar')...")
res3 = client.post(
    "/chat",
    json={"session_id": "verify_check3", "message": "what's the weather like today near Kharghar"}
)

assert res3.status_code == 200, f"Check 3 failed with status {res3.status_code}: {res3.text}"
data3 = res3.json()

sources3 = data3.get("sources", [])
web_sources3 = [s for s in sources3 if s.get("type") == "web_search"]

print(f"Response Status: {res3.status_code}")
print(f"Total Sources Returned: {len(sources3)}")
print(f"Web Sources Count: {len(web_sources3)}")

for w in web_sources3:
    print(f"  • [web_search] {w.get('title')}")
    print(f"    URL: {w.get('url')}")
    print(f"    Retrieved: {w.get('retrieved_at')}")

assert len(web_sources3) >= 1, "CHECK 3 FAILED: Expected at least one web_search source type"
assert all(w.get("url") and w.get("url").startswith("http") for w in web_sources3), "CHECK 3 FAILED: Web sources must have real URLs"
print("\n[PASSED] CHECK 3: Web search triggered with real URLs for time-sensitive query!")

print("\n" + "=" * 70)
print("  ALL PROMPT 6 CHECKS VERIFIED SUCCESSFULLY!")
print("=" * 70)
