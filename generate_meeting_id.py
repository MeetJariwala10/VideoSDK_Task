import requests
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://api.videosdk.live/v2/rooms"

token = os.getenv("VIDEOSDK_AUTH_TOKEN")
if not token:
    raise RuntimeError("Missing VIDEOSDK_TOKEN in environment. Create a .env with VIDEOSDK_TOKEN=<your_jwt_token>.")

headers = {
    'Authorization': token,
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, json = {
    "customRoomId" : "aaa-bbb-ccc"
}, headers = headers)

print(response.status_code)
print(response.text)