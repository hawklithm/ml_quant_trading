#!/usr/bin/env python3
import json

path = '/home/hawky/.hermes/cron/jobs.json'
with open(path) as f:
    data = json.load(f)

for job in data['jobs']:
    if job.get('id') == 'f4543edfb5fd':
        job['prompt'] = '运行量化选股美股开市前预测。执行命令：\n```bash\ncd /home/hawky/projects/quant-trading && source .venv/bin/activate && NO_PROXY="*" unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY && timeout 300 python3 cron_market_job.py --market US --mode pre\n```\n将输出结果发送给用户。'
        print(f'Updated prompt for job: {job["name"]}')
        break

with open(path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Done')
