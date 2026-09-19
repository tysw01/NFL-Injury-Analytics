# tests/run_tests.py
# ------------------
# Quick smoke-test for the /api/model/predict endpoint.
#
# Runs a fixed set of named test cases against the live API and prints whether
# each prediction returned a valid play probability.  This is a manual test —
# start the Flask server (python injuryAPI.py) before running this script.
#
# Usage:
#   python tests/run_tests.py
#
# Expected output for a passing run:
#   OK  Mahomes - Ankle/Questionable: play=0.72  games_out=0
#   OK  Hurts - Concussion/Out: play=0.12  games_out=2
#   ...

import urllib.request, json

# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry is (label, request_body).  The label is only for display; the
# body dict is sent as JSON to the prediction endpoint.
TESTS = [
    ("Mahomes - Ankle/Questionable",  {"player_name":"Patrick Mahomes","injury_detail":"Ankle","report_status":"Questionable"}),
    ("Hurts - Concussion/Out",        {"player_name":"Jalen Hurts","injury_detail":"Concussion","report_status":"Out"}),
    ("Barkley - Knee/Out",            {"player_name":"Saquon Barkley","injury_detail":"Knee","report_status":"Out"}),
    ("Kelce - Knee/Limited",          {"player_name":"Travis Kelce","injury_detail":"Knee","report_status":"Limited"}),
    ("Jefferson - Hamstring/Question",{"player_name":"Justin Jefferson","injury_detail":"Hamstring","report_status":"Questionable"}),
    ("Generic - Hamstring/Doubtful",  {"player_name":"John Generic","injury_detail":"Hamstring","report_status":"Doubtful"}),
]

# URL of the prediction endpoint (Flask must be running locally on port 5001)
URL = "http://127.0.0.1:5001/api/model/predict"

# ── Run each test case ────────────────────────────────────────────────────────
for label, body in TESTS:
    # Encode the request payload as UTF-8 JSON bytes
    data = json.dumps(body).encode()

    # Build an HTTP POST request with the JSON body
    req = urllib.request.Request(
        URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        # Send the request and read the JSON response
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.load(resp)

        # Navigate the nested response structure
        ro = d.get("responseObject") or {}     # The main payload object
        ng = ro.get("next_game") or {}          # Next-game prediction block

        # Extract the key metrics we care about
        prob = ng.get("play_probability")       # Probability the player actually plays (0-1)
        # projected_games_out can live at the top of responseObject OR inside return_timeline
        out  = ro.get("projected_games_out") or (ro.get("return_timeline") or {}).get("projected_games_out")
        msg  = d.get("message", "")

        # Print a concise summary line
        if prob is not None:
            print(f"OK  {label}: play={prob:.1%}  games_out={out}")
        else:
            # Prediction returned something unexpected — show a truncated message
            print(f"UNK {label}: {msg[:80]} | ro_keys={list(ro.keys())}")

    except Exception as e:
        # Network error, timeout, or JSON parse failure
        print(f"EXC {label}: {e}")
