#!/usr/bin/env python3
"""Debug LLM client - no secrets in output"""
import json, os, sys

# Strategy 1: auth.json
auth_path = os.path.expanduser("~/.hermes/auth.json")
found_api_key = False
if os.path.exists(auth_path):
    with open(auth_path) as f:
        auth_data = json.load(f)
    cp = auth_data.get("credential_pool", {})
    providers_found = list(cp.keys())
    print(f"Step1: auth.json providers = {providers_found}")
    
    for pk in ["custom:liantong", "custom:aigw-gzgy2.cucloud.cn:8443"]:
        if pk in cp:
            for cred in cp[pk]:
                ak = cred.get("access_token", "")
                bu = cred.get("base_url", "")
                ok = bool(ak and ak != "***" and bu)
                print(f"  {pk}: ak_exists={bool(ak)} ak_not_starred={ak != '***'} bu_exists={bool(bu)} valid={ok}")
                if ok:
                    found_api_key = True
                    print(f"  -> VALID, bu={bu}")
        else:
            print(f"  {pk}: NOT FOUND")
else:
    print("Step1: auth.json NOT FOUND")

if not found_api_key:
    print("Step2: trying config.yaml...")
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw = f.read()
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped.startswith("api_key:") and "auxiliary" not in stripped and "custom_providers" not in stripped:
                v = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                print(f"  api_key line: has_value={bool(v)} starred={v == '***'} len={len(v)}")
                if v and v != "***":
                    found_api_key = True
            if stripped.startswith("base_url:") and "auxiliary" not in stripped and "custom_providers" not in stripped:
                v = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                print(f"  base_url line: exists={bool(v)} val={v[:50]}")
    else:
        print("Step2: config.yaml NOT FOUND")

print(f"\nFinal: found_api_key={found_api_key}")
