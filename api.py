
# ---------- Payload ----------


import requests
import json

# ---------- Configuration ----------
URL = "https://api.runpod.ai/v2/yqe4icw0r14oo4/run"

# ---------- Payload ----------
payload = {
    "input": {
        "data": [
            {
                "user_id": "test_user",
                "fitcheck_id": "test_fitcheck",
                "profile_image_url": "https://thumbs.dreamstime.com/b/handsome-african-american-male-fashion-model-full-length-portrait-posing-isolated-white-background-46340658.jpg",
                "garment_url": "https://fitcheck-assets-maskfix.s3.eu-north-1.amazonaws.com/test/garment_only/WhatsApp+Image+2025-12-03+at+18.52.47+(1).jpeg",
                "cloth_type": "lower"
            }
        ],
        "num_inference_steps": 50,
        "guidance_scale": 2.5,
        "seed": 42
    }
}
# ---------- Headers ----------
# Postman may work without explicit auth if you already have a session,
# but usually RunPod requires Authorization or session cookie.
# If it works without auth, you can leave headers minimal:
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# ---------- Make POST request ----------
try:
    response = requests.post(URL, headers=headers, json=payload, timeout=30)

    print("Status code:", response.status_code)

    try:
        result = response.json()
        print("Response JSON:")
        print(json.dumps(result, indent=2))
    except ValueError:
        print("Response is not JSON. Raw response text:")
        print(response.text)

    if response.status_code == 401:
        print("Unauthorized. Postman may be sending a session or auth token automatically.")
except requests.RequestException as e:
    print("Request failed:", e)
