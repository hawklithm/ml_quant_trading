#!/usr/bin/env python3
"""Run sentiment fusion on existing predictions."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

state_file = os.path.expanduser('~/.cache/hermes-quant/market_jobs/us_state.json')
with open(state_file) as f:
    state = json.load(f)

predictions = state.get('last_predictions', [])
if not predictions:
    print('No predictions found')
    sys.exit(1)

tickers = [p['ticker'] for p in predictions]
print(f'Running sentiment analysis for {len(tickers)} stocks...')

try:
    from finbert_sentiment import build_sentiment_factors, sentiment_boost, print_sentiment_report
    
    sentiment_factors = build_sentiment_factors(tickers)
    
    event_stocks = []
    fused_count = 0
    for p in predictions:
        t = p['ticker']
        sf = sentiment_factors.get(t, {})
        if sf and sf.get('news_count', 0) > 0:
            original = p['score']
            fused, adj, evt_adj = sentiment_boost(original, sf)
            p['score'] = round(fused, 4)
            p['sentiment_adj'] = round(adj, 4)
            p['event_adj'] = round(evt_adj, 4)
            fused_count += 1
            if sf.get('events'):
                event_stocks.append((t, sf['event_labels'], sf.get('event_discount', 1.0)))
    
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    print(f'\n情绪融合完成: {fused_count} 只有新闻情绪, {len(event_stocks)} 只有异常事件')
    if event_stocks:
        print(f'\n异常事件预警 ({len(event_stocks)} 只):')
        for t, labels, discount in event_stocks:
            print(f'    [RED] {t}: {" + ".join(labels)} (折扣 {discount:.3f})')
    else:
        print('\n无异常事件检测到')
    
    # Save fused state
    state['last_predictions'] = predictions
    state['last_pre_time'] = '22:19 + sentiment'
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    # Print final sorted ranking
    print(f'\n=== 情绪融合后排名 ===')
    print(f'{"#":>3} {"代码":>8} {"评分":>7} {"方向":>6} {"情绪调整":>8} {"事件调整":>8}')
    print('-' * 50)
    for i, p in enumerate(predictions[:10]):
        sa = p.get('sentiment_adj', 0)
        ea = p.get('event_adj', 0)
        print(f'{i+1:>3} {p["ticker"]:>8} {p["score"]:>7.4f} {p["direction"]:>6} {sa:>+8.4f} {ea:>+8.4f}')
    
except ImportError as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f'Sentiment analysis error: {e}')
    import traceback
    traceback.print_exc()
