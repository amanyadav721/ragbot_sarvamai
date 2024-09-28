import requests

url = "https://api.sarvam.ai/translate"

payload = {
    "input": "Who are you",
    "source_language_code": "en-IN",
    "target_language_code": "hi-IN",
    "speaker_gender": "Male",
    "mode": "formal",
    "model": "mayura:v1",
    "enable_preprocessing": True
}
headers = {"Content-Type": "application/json", 'API-Subscription-Key': '5a6816b0-7121-48ca-a8e2-f9e7cf06d4c0'}

response = requests.request("POST", url, json=payload, headers=headers)
response = response.json().get("translated_text")

print(response)