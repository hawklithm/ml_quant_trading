#!/usr/bin/env python3
import yfinance as yf, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

spy = yf.download('SPY', period='1y', auto_adjust=True, progress=False)
if isinstance(spy.columns, pd.MultiIndex): spy.columns=[c[0] for c in spy.columns]
spy_r = spy['Close'].pct_change().dropna()
spy_vol = float(spy_r.tail(21).std() * np.sqrt(252) * 100)
spy_ret = (spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1) * 100
print(f'SPY 年度收益: {spy_ret:+.1f}% | 波动率: {spy_vol:.0f}%')
print()

# H股
tickers = ['0700.HK','9988.HK','9999.HK','1810.HK','3690.HK',
           '0941.HK','0883.HK','0388.HK','0005.HK','1299.HK',
           '2269.HK','2382.HK']
names = {'0700.HK':'腾讯','9988.HK':'阿里','9999.HK':'网易','1810.HK':'小米',
         '3690.HK':'美团','0941.HK':'中移动','0883.HK':'中海油',
         '0388.HK':'港交所','0005.HK':'汇丰','1299.HK':'友���',
         '2269.HK':'药明','2382.HK':'舜宇'}

results = []
for t in tickers:
    try:
        d = yf.download(t, period='6mo', auto_adjust=True, progress=False)
        if d.empty: continue
        if isinstance(d.columns, pd.MultiIndex): d.columns=[c[0] for c in d.columns]
        c = d['Close'].values
        if len(c) < 20: continue
        r60 = (c[-1]/c[-61]-1)*100 if len(c)>=61 else 0
        r20 = (c[-1]/c[-21]-1)*100 if len(c)>=21 else 0
        sc = pd.Series(c); delta = sc.diff()
        g = delta.where(delta>0,0).rolling(14).mean()
        l = -delta.where(delta<0,0).rolling(14).mean()
        rsi = float((100-100/(1+g/l)).iloc[-1])
        sma20 = np.mean(c[-20:]); sma50 = np.mean(c[-50:]) if len(c)>=50 else sma20
        v20 = (c[-1]/sma20-1)*100; v50 = (c[-1]/sma50-1)*100
        sr = pd.Series(c).pct_change().dropna()
        vol = float(sr.tail(21).std()*np.sqrt(252)*100)
        h52=np.max(c); fh=(c[-1]/h52-1)*100; fl=(c[-1]/np.min(c)-1)*100
        s = 0
        if v20>0 and v50>0: s+=3
        elif v20>0: s+=2
        if r60>0: s+=2
        elif r60>-5: s+=1
        else: s -= 0
        if -30<fh<-10: s+=2
        elif fh<-30: s+=1
        if 30<rsi<60: s+=1
        elif rsi<=30: s+=2
        if vol<30: s+=2
        elif vol<40: s+=1
        if v20>0: s+=2
        if v50>0: s+=2
        elif v50>-3: s+=1
        results.append({'t':t,'n':names.get(t,''),'p':round(c[-1],3),'r60':round(r60,1),'rsi':round(rsi,1),'v20':round(v20,1),'v50':round(v50,1),'vol':round(vol,1),'fh':round(fh,1),'s':s})
    except Exception as e:
        pass

results.sort(key=lambda x:x['s'], reverse=True)
print('=== 港股推荐排名 ===')
print('#  名称    代码        现价     60日%    RSI   vs20    vs50    波动   距高    分')
print('-'*85)
for i,r in enumerate(results):
    print(f"{i+1:<3} {r['n']:<6} {r['t']:<12} {r['p']:<8} {r['r60']:<+8.1f} {r['rsi']:<6.1f} {r['v20']:<+8.1f} {r['v50']:<+8.1f} {r['vol']:<6.1f} {r['fh']:<+7.1f} {r['s']:<4}")

print()
print('=== 推荐评语 ===')
for i,r in enumerate(results[:6]):
    n=r['n'] or r['t']
    notes=[]
    if r['v20']>0 and r['v50']>0: notes.append('均线多头')
    elif r['v20']>0: notes.append('短线企稳')
    if r['rsi']<35: notes.append('超跌区')
    if r['fh']<-30 and r['fh']>-50: notes.append(f'超跌{r["fh"]:.0f}%')
    elif r['fh']>=-10: notes.append('接近高点')
    if r['vol']<30: notes.append('低波动')
    if not notes: notes.append('中性')
    act = '*** 推荐' if r['s']>=6 else '** 观察' if r['s']>=4 else '* 观望' if r['s']>=2 else 'X 回避'
    print(f'{act} {n}({r["t"]}) 分{r["s"]}: {", ".join(notes)}')

# 美股扫描
print()
print('=== 美股扫描 ===')
us = ['NVDA','META','MSFT','AMZN','GOOGL','AVGO','PLTR','SOFI','COIN',
      'MSTR','LLY','COST','JPM','V','CAT','WMT','XOM','CVX']
usr = []
for t in us:
    try:
        d = yf.download(t, period='6mo', auto_adjust=True, progress=False)
        if d.empty: continue
        if isinstance(d.columns, pd.MultiIndex): d.columns=[c[0] for c in d.columns]
        c = d['Close'].values
        if len(c)<20: continue
        r60 = (c[-1]/c[-61]-1)*100 if len(c)>=61 else 0
        sc = pd.Series(c); delta=sc.diff()
        g=delta.where(delta>0,0).rolling(14).mean()
        l=-delta.where(delta<0,0).rolling(14).mean()
        rsi = float((100-100/(1+g/l)).iloc[-1])
        sma20=np.mean(c[-20:]); sma50=np.mean(c[-50:]) if len(c)>=50 else sma20
        v20=(c[-1]/sma20-1)*100; v50=(c[-1]/sma50-1)*100
        sr=pd.Series(c).pct_change().dropna()
        vol=float(sr.tail(21).std()*np.sqrt(252)*100)
        fh=(c[-1]/np.max(c)-1)*100
        s=0
        if v20>0 and v50>0: s+=3
        if r60>0: s+=2
        if 30<rsi<60: s+=1
        if vol<30: s+=2
        elif vol<40: s+=1
        usr.append({'t':t,'p':round(c[-1],2),'r60':round(r60,1),'rsi':round(rsi,1),'v20':round(v20,1),'v50':round(v50,1),'vol':round(vol,1),'fh':round(fh,1),'s':s})
    except:
        pass
usr.sort(key=lambda x:x['s'], reverse=True)

print('#  代码    现价      60日%    RSI    vs20    vs50    波动   距高    分')
print('-'*75)
for i,r in enumerate(usr[:10]):
    print(f"{i+1:<3} {r['t']:<8} {r['p']:<10} {r['r60']:<+8.1f} {r['rsi']:<6.1f} {r['v20']:<+8.1f} {r['v50']:<+8.1f} {r['vol']:<6.1f} {r['fh']:<+7.1f} {r['s']:<4}")
