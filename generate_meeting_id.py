"""Utility to create a VideoSDK meeting room via REST API.

Reads `VIDEOSDK_AUTH_TOKEN` from `.env` and creates a room using
`https://api.videosdk.live/v2/rooms`. Prints status code and response JSON.

Usage:
    python generate_meeting_id.py
"""

import requests
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://api.videosdk.live/v2/rooms"

token = os.getenv("VIDEOSDK_AUTH_TOKEN")
if not token:
    raise RuntimeError(
        "Missing VIDEOSDK_AUTH_TOKEN in environment. Create a .env with "
        "VIDEOSDK_AUTH_TOKEN=<your_jwt_token> (see README)."
    )

headers = {
    'Authorization': token,
    'Content-Type': 'application/json'
}

# You can remove customRoomId to let the API auto-generate one.
response = requests.request("POST", url, json = {
    "customRoomId" : "aaa-bbb-ccc"
}, headers = headers)

print(response.status_code)
print(response.text)