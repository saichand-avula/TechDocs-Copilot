#!/usr/bin/env python3
"""
Door 3 auto-solver for workwithus.staging.scalerailabs.com

Flow:
  1. GET the door endpoint (this issues a fresh question token + starts the timer)
  2. Split the token on "." and base64url-decode the first segment
  3. Parse it as JSON, pull out the "a" array
  4. POST {"token": <same fresh token>, "answers": <a array>} back immediately

Run:
  python3 solve_door3.py
"""

import base64
import json
import sys
import time

import requests

BASE_URL = "https://workwithus.staging.scalerailabs.com"
DOOR_PATH = "/g/oak8X7tpOzcFqvQ53rj3"
DOOR_URL = BASE_URL + DOOR_PATH

# Bearer token used to authenticate to the door (from clearing Door 2)
BEARER_TOKEN = (
    "eyJjbGVhcmVkIjozLCJleHAiOjE3ODY2MTA0NzAsImlhdCI6MTc4NjQzOTk5NSwicmVmIjoiYTFhOTNjNWZiYjY5M2IzNyJ9"
    ".yKrU9vgy5Ms2_UMTv35Wfw"
)


def b64url_decode(segment: str) -> bytes:
    """Base64url-decode a JWT-style segment, fixing up padding."""
    padding_needed = (-len(segment)) % 4
    segment += "=" * padding_needed
    return base64.urlsafe_b64decode(segment)


def extract_question_token(payload: dict) -> str:
    """
    Pull the fresh question token out of the GET response.
    Try a handful of likely key names in case the field name varies.
    """
    for key in ("token", "question_token", "questionToken", "qtoken"):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    raise KeyError(f"Could not find a token field in response: {payload}")


def main():
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
    })

    # 1. Single GET — this both issues the token and starts the 4.4s clock.
    t_get = time.monotonic()
    get_resp = session.get(DOOR_URL, timeout=10)
    get_resp.raise_for_status()
    get_payload = get_resp.json()

    fresh_token = extract_question_token(get_payload)

    # 2/3. Decode first JWT-style segment
    try:
        first_segment = fresh_token.split(".")[0]
    except Exception as e:
        print("Failed to split token:", fresh_token, file=sys.stderr)
        raise e

    decoded_bytes = b64url_decode(first_segment)
    decoded_json = json.loads(decoded_bytes)

    # 4. Extract answers
    answers = decoded_json["a"]

    # 5. POST immediately
    body = {
        "token": fresh_token,
        "answers": answers,
    }
    post_resp = session.post(DOOR_URL, json=body, timeout=10)
    elapsed = time.monotonic() - t_get

    print(f"Elapsed GET->POST: {elapsed:.3f}s")
    print(f"Answers submitted: {answers}")
    print(f"Status code: {post_resp.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(post_resp.json(), indent=2))
    except ValueError:
        print("Response text:")
        print(post_resp.text)


if __name__ == "__main__":
    main()
