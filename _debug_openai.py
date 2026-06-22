#!/usr/bin/env python3
"""Debug OpenAI init step by step"""
import json, os

# Read liantong cred from auth.json
auth_path = os.path.expanduser("~/.hermes/auth.json")
with open(auth_path) as f:
    auth_data = json.load(f)

cp = auth_data.get("credential_pool", {})
for cred in cp.get("custom:liantong", []):
    ak = cred.get("access_token", "")
    bu = cred.get("base_url", "")
    if ak and ak != "***" and bu:
        print(f"key_len={len(ak)} bu={bu}")
        print(f"key_valid: {bool(ak)}")

        from openai import OpenAI
        try:
            client = OpenAI(base_url=bu, api_key=ak, timeout=10)
            print("OpenAI() created OK")
            print(f"base_url={client.base_url}")
            
            # Quick test call
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "say hi"}],
                temperature=0.1,
                max_tokens=10
            )
            print(f"API call OK: {resp.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
        break
