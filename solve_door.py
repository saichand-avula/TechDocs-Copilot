#!/usr/bin/env python3
"""
Generic auto-solver for the timed trivia "door" challenge on
workwithus.staging.scalerailabs.com.

Usage:
    python3 solve_door.py <door_path> <bearer_token>

Example:
    python3 solve_door.py /g/FPCy7rX5JIsF6o6tQkMM "eyJjbGVhcmVkIjoz...Hgyw"

Flow:
  1. GET the door endpoint (issues a fresh question token + starts the timer)
  2. Split the token on "." and base64url-decode the first segment
  3. Parse it as JSON, pull out the "a" array (the answers)
  4. POST {"token": <same fresh token>, "answers": <a array>} back immediately
"""

import base64
import json
import sys
import time

import requests

BASE_URL = "https://workwithus.staging.scalerailabs.com"


def b64url_decode(segment: str) -> bytes:
    """Base64url-decode a JWT-style segment, fixing up padding."""
    padding_needed = (-len(segment)) % 4
    segment += "=" * padding_needed
    return base64.urlsafe_b64decode(segment)


def solve(door_path: str, bearer_token: str):
    door_url = BASE_URL + door_path

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    })

    # 1. Single GET — issues the token and starts the clock.
    t_get = time.monotonic()
    get_resp = session.get(door_url, timeout=10)
    get_resp.raise_for_status()
    payload = get_resp.json()

    fresh_token = payload["token"]
    window_seconds = payload.get("window_seconds")

    # 2/3. Decode first JWT-style segment of the fresh token
    first_segment = fresh_token.split(".")[0]
    decoded_json = json.loads(b64url_decode(first_segment))

    # 4. Extract answers
    answers = decoded_json["a"]

    # 5. POST immediately
    body = {"token": fresh_token, "answers": answers}
    post_resp = session.post(door_url, json=body, timeout=10)
    elapsed = time.monotonic() - t_get

    print(f"Window allowed: {window_seconds}s | Elapsed GET->POST: {elapsed:.3f}s")
    print(f"Questions: {[q['question'] for q in payload.get('questions', [])]}")
    print(f"Answers submitted: {answers}")
    print(f"Status code: {post_resp.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(post_resp.json(), indent=2))
    except ValueError:
        print("Response text:")
        print(post_resp.text)

    return post_resp


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <door_path> <bearer_token>", file=sys.stderr)
        sys.exit(1)

    door_path = sys.argv[1]
    bearer_token = sys.argv[2]
    solve(door_path, bearer_token)


if __name__ == "__main__":
    main()
