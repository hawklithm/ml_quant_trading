#!/usr/bin/env python3
"""Try raw requests to debug"""
import json, os, requests

auth_path = os.path.expanduser("~/.hermes/auth.json")
with open(auth_path) as f:
    auth_data = json.load(f)
cp = auth_data.get("credential_pool", {})

for cred in cp.get("custom:liantong", []):
    ak = cred.get("access_token", "")
    bu = cred.get("base_url", "")
    if ak and ak != "***" and bu:
        for p in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                  "ALL_PROXY", "all_proxy"):
            os.environ.pop(p, None)
        
        url = bu.rstrip("/") + "/chat/completions"
        payload = {
            "model": "DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "say hi in 2 words"}],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        r = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {ak}",
            "Content-Type": "application/json"
        }, timeout=15)
        
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"Raw response keys: {list(data.keys())}")
            choice = data["choices"][0]
            msg = choice.get("message", {})
            print(f"message keys: {list(msg.keys())}")
            print(f"content: {repr(msg.get('content'))}")
            print(f"reasoning: {repr(msg.get('reasoning'))}")
            print(f"tool_calls: {msg.get('tool_calls', [])}")
        else:
            print(f"Error: {r.text[:200]}")
        break
