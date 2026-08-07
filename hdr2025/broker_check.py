"""
Is the broker reachable and is the key good? One call, or none at all.

    python3 -u broker_check.py

Distinguishes the three ways a run can come back with every reader unavailable:
a key that never reached the shell, a container with no route to the broker,
and a key the broker rejects. Costs at most one completion.
"""
import os
import socket
import sys
import urllib.request

HOST, PORT = "80.151.131.52", 9839
BASE = f"http://{HOST}:{PORT}/v1"

key = os.environ.get("BROKER_API_KEY") or os.environ.get("OPENAI_API_KEY")
print(f"1. BROKER_API_KEY set : {bool(key)}"
      + (f"  (len {len(key)}, starts {key[:4]!r})" if key else "  <-- THIS IS THE PROBLEM"))
if key and key != key.strip():
    print("   WARNING: the key has leading/trailing whitespace")

print(f"2. TCP to {HOST}:{PORT} ... ", end="", flush=True)
try:
    with socket.create_connection((HOST, PORT), timeout=10):
        print("open")
except OSError as e:
    print(f"UNREACHABLE ({type(e).__name__}: {e})  <-- THIS IS THE PROBLEM")
    sys.exit(1)

print("3. GET /v1/usage ... ", end="", flush=True)
req = urllib.request.Request(f"{BASE}/usage",
                             headers={"Authorization": f"Bearer {key or 'none'}"})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print(f"{r.status}  {r.read()[:200].decode('utf-8', 'replace')}")
except Exception as e:                                        # noqa: BLE001
    body = getattr(e, "read", lambda: b"")()[:200].decode("utf-8", "replace")
    print(f"{type(e).__name__}: {e}  {body}  <-- key rejected" if "401" in str(e)
          else f"{type(e).__name__}: {e}  {body}")

if not key:
    sys.exit("\nNo key: nothing else to test.")

print("4. one real completion ... ", end="", flush=True)
try:
    from openai import OpenAI
    c = OpenAI(base_url=BASE, api_key=key.strip(), timeout=60, max_retries=0)
    r = c.chat.completions.create(model="ministral-3b-2512",
                                  messages=[{"role": "user", "content": "Reply with OK."}])
    print(f"OK -- {r.choices[0].message.content!r}")
    print("\nThe broker works from here. The failed run did not have this environment.")
except Exception as e:                                        # noqa: BLE001
    print(f"{type(e).__name__}: {str(e)[:300]}")
    print("\nThis is the exception the readers hit.")
