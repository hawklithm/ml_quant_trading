import yfinance as yf, warnings
warnings.filterwarnings('ignore')

# Test AAPL
tk = yf.Ticker('AAPL')
df = tk.history(period='3mo')
if df.empty:
    print('Empty dataframe')
else:
    print(f'Rows: {len(df)}, Cols: {list(df.columns)}')
    print(f'Index: {df.index[0]} ... {df.index[-1]}')
    tz = getattr(df.index, 'tz', None)
    print(f'Index tz: {tz}')
