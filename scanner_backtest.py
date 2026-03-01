import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
EXCHANGE = "multi"
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]  # Reducido para mayor probabilidad
LOOKBACK_CANDLES = 300
TP_PCT = 0.02
SL_PCT = 0.01

ASSETS = ["BTC", "ETH", "SOL"]
BASE_SYMBOLS = ["BTC", "ETH", "SOL"]

HEADERS = {'User-Agent': 'Mozilla/5.0'}
DATA_CACHE = {}

CATEGORY_MAP = {
    '1m': 'Scalp', '3m': 'Scalp', '5m': 'Scalp', '15m': 'Scalp',
    '30m': 'Intraday', '1h': 'Intraday', '4h': 'Intraday',
    '8h': 'Swing', '12h': 'Swing', '1d': 'Swing'
}

# ============================================================
# FUNCIONES DE DESCARGA (KuCoin / Crypto.com / CoinGecko)
# ============================================================
def fetch_klines_kucoin(symbol, timeframe, limit=500):
    tf_map = {'1m':'1min','3m':'3min','5m':'5min','15m':'15min','30m':'30min',
              '1h':'1hour','2h':'2hour','4h':'4hour','6h':'6hour','8h':'8hour',
              '12h':'12hour','1d':'1day','1w':'1week'}
    interval = tf_map.get(timeframe)
    if not interval: return None
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {'symbol': symbol, 'type': interval, 'limit': limit}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
        if data['code'] != '200000': return None
        df = pd.DataFrame(data['data'], columns=['timestamp','open','close','high','low','volume','turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        df.set_index('timestamp', inplace=True)
        for col in ['open','high','low','close','volume']: df[col] = df[col].astype(float)
        df = df[['open','high','low','close','volume']].sort_index()
        print(f"   📥 KuCoin {symbol} {timeframe}: {len(df)} velas")
        return df
    except Exception as e:
        print(f"   ⚠️ KuCoin error {symbol}: {e}")
        return None

def fetch_klines_cryptocom(symbol, timeframe, limit=500):
    tf_map = {'1m':'1m','5m':'5m','15m':'15m','30m':'30m',
              '1h':'1h','2h':'2h','4h':'4h','6h':'6h','8h':'8h','12h':'12h','1d':'1d','1w':'1w'}
    interval = tf_map.get(timeframe)
    if not interval: return None
    url = "https://api.crypto.com/exchange/v1/public/get-candlestick"
    params = {'instrument_name': symbol, 'timeframe': interval}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
        if data['code'] != 0: return None
        df = pd.DataFrame(data['result']['data'])
        for col in ['t','o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        df = df[['open','high','low','close','volume']].sort_index()
        print(f"   📥 Crypto.com {symbol} {timeframe}: {len(df)} velas")
        return df
    except Exception as e:
        print(f"   ⚠️ Crypto.com error {symbol}: {e}")
        return None

def fetch_klines_coingecko(coin_id, timeframe):
    days = 2 if timeframe in ['1m','5m','15m','30m','1h'] else 30
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {'vs_currency':'usd','days':days}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['volume'] = 0
        print(f"   📥 CoinGecko {coin_id}: {len(df)} velas OHLC")
        return df
    except Exception as e:
        print(f"   ⚠️ CoinGecko error {coin_id}: {e}")
        return None

def fetch_klines(symbol, timeframe):
    key = f"{symbol}_{timeframe}"
    if key in DATA_CACHE: return DATA_CACHE[key]

    kucoin_sym = symbol.replace('_','-') if '_' in symbol else f"{symbol}-USDT"
    cryptocom_sym = symbol if '_' in symbol else f"{symbol}_USDT"

    df = fetch_klines_kucoin(kucoin_sym, timeframe, limit=LOOKBACK_CANDLES)
    if df is not None and len(df)>50: 
        DATA_CACHE[key]=df
        return df
    df = fetch_klines_cryptocom(cryptocom_sym, timeframe, limit=LOOKBACK_CANDLES)
    if df is not None and len(df)>50: 
        DATA_CACHE[key]=df
        return df

    coingecko_ids={'BTC':'bitcoin','ETH':'ethereum','SOL':'solana'}
    if symbol in coingecko_ids:
        df = fetch_klines_coingecko(coingecko_ids[symbol], timeframe)
        if df is not None and len(df)>20: 
            DATA_CACHE[key]=df
            return df

    print(f"   ⚠️ No se pudieron obtener datos para {symbol} {timeframe}")
    return None

# ============================================================
# NORMALIZACIÓN Y TENSIÓN
# ============================================================
def normalize(series, window=50):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)

def tension_235(series):
    ema2 = series.ewm(span=2).mean()
    ema3 = series.ewm(span=3).mean()
    ema5 = series.ewm(span=5).mean()
    return (ema2-ema3).abs() + (ema3-ema5).abs()

def compute_edge_pct(price, tension, k, quantile=0.85):
    # Asegurar que price y tension tengan el mismo índice
    common_idx = price.index.intersection(tension.index)
    price = price.loc[common_idx]
    tension = tension.loc[common_idx]
    
    mask = tension > tension.quantile(quantile)
    if mask.sum() == 0:
        return np.nan, np.nan
    future_ret = (price.shift(-k) - price) / price
    # Alinear future_ret con el índice de mask (ya tienen el mismo índice)
    future_ret = future_ret.loc[common_idx]  # aunque shift introduce NaN al final
    edge = future_ret[mask].mean()
    hitrate = (future_ret[mask] > 0).mean()
    return edge, hitrate

def compute_pidelta(price, window=20):
    returns = price.pct_change().fillna(0)
    P_struct = returns.rolling(window).mean()
    P_hist = returns.ewm(span=window).mean()
    return P_struct-P_hist

def compute_corr(pidelta1, pidelta2):
    return pidelta1.corr(pidelta2) if len(pidelta1)>1 else 0

# ============================================================
# ANALISIS POR SIMBOLO/TIMEFRAME
# ============================================================
def analyze_symbol_tf(symbol, tf):
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < 50: 
        print(f"   ⚠️ {symbol} {tf}: menos de 50 velas")
        return None, None

    price = df['close'].dropna()
    if len(price) < 50:
        print(f"   ⚠️ {symbol} {tf}: precio con pocos datos")
        return None, None
        
    volume = df['volume'].fillna(0)
    S = normalize(price)
    T = tension_235(S).dropna()
    if len(T) < 30:
        print(f"   ⚠️ {symbol} {tf}: tensión insuficiente ({len(T)} puntos)")
        return None, None
        
    pidelta = compute_pidelta(price)

    k_values=[1,2,3,5,8,13,21]
    best_score=-np.inf
    best_result=None

    for k in k_values:
        edge_pct, hitrate = compute_edge_pct(price, T, k)
        if np.isnan(edge_pct):
            continue
            
        # Monte Carlo simplificado
        mc_edges = []
        for _ in range(20):  # reducido para velocidad
            perm = np.random.permutation(S.dropna().values)
            T_mc = tension_235(pd.Series(perm, index=S.dropna().index))
            mc_e, _ = compute_edge_pct(price, T_mc, k)
            if not np.isnan(mc_e):
                mc_edges.append(mc_e)
        if len(mc_edges) < 5:
            continue
        mc_mean = np.nanmean(mc_edges)
        mc_std = np.nanstd(mc_edges)
        Z = (edge_pct - mc_mean) / (mc_std + 1e-9)

        # Tiempo medio hasta alcanzar el edge (target porcentual)
        mask = T > T.quantile(0.85)
        idxs = np.where(mask)[0]
        target = edge_pct
        tau_list = []
        for idx in idxs:
            if idx + 50 >= len(price): continue
            base = price.iloc[idx]
            target_price = base * (1 + target)
            for i in range(1, 50):
                if idx+i >= len(price): break
                if target>0 and price.iloc[idx+i] >= target_price:
                    tau_list.append(i)
                    break
                elif target<0 and price.iloc[idx+i] <= target_price:
                    tau_list.append(i)
                    break
        tau = np.nanmean(tau_list) if tau_list else 100.0  # si no hay, asignamos un valor grande

        score = abs(Z) * abs(edge_pct) / tau
        if score > best_score:
            best_score = score
            best_result = {
                'Symbol':symbol, 'TF':tf, 'Categoria':CATEGORY_MAP.get(tf,'Otro'),
                'k':k, 'Edge_%':edge_pct, 'HitRate':hitrate, 'Z':Z, 'Tau':tau, 'Score':score,
                'Precio_actual':price.iloc[-1], 'TP_est':price.iloc[-1]*(1+TP_PCT),
                'SL_est':price.iloc[-1]*(1-SL_PCT),
                'PIDelta_mean':pidelta.mean(), 'PIDelta_std':pidelta.std(),
                'Volume_mean':volume.mean(), 'Volume_std':volume.std()
            }
    return best_result, pidelta

# ============================================================
# ESCANEO POR TIMEFRAME
# ============================================================
def scan_timeframe(tf):
    results = []
    base_pideltas = {}
    # Obtener PIDelta de activos base para correlación
    for base in BASE_SYMBOLS:
        _, pid = analyze_symbol_tf(base, tf)
        if pid is not None: 
            base_pideltas[base]=pid
    for asset in ASSETS:
        print(f"🔍 Analizando {asset} {tf}...")
        sys.stdout.flush()
        result, pid = analyze_symbol_tf(asset, tf)
        if result:
            for base in BASE_SYMBOLS:
                if base in base_pideltas and pid is not None:
                    common_idx = pid.index.intersection(base_pideltas[base].index)
                    corr = compute_corr(pid.loc[common_idx], base_pideltas[base].loc[common_idx]) if len(common_idx)>5 else np.nan
                    result[f'Corr_{base}'] = corr
                else:
                    result[f'Corr_{base}'] = np.nan
            results.append(result)
            print(f"   ✅ {asset} {tf} | Edge: {result['Edge_%']:.4%} | HitRate: {result['HitRate']:.2%} | Score: {result['Score']:.4f}")
        else:
            print(f"   ⚠️ {asset} {tf}: no se obtuvieron resultados")
        time.sleep(0.2)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.to_csv(f"escaneo_{tf}.txt", index=False, sep='\t')
    return df_results

# ============================================================
# MAIN
# ============================================================
if __name__=="__main__":
    print("="*70)
    print("🚀 BACKTEST ESTADÍSTICO DE TENSIÓN 2-3-5 (Edge porcentual)")
    print("="*70)
    sys.stdout.flush()
    all_results = []
    for tf in TIMEFRAMES:
        print(f"\n⏰ Timeframe: {tf} ({CATEGORY_MAP.get(tf,'Otro')})")
        df_tf = scan_timeframe(tf)
        if not df_tf.empty:
            all_results.append(df_tf)
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        cols = ['Symbol','TF','Categoria','k','Edge_%','HitRate','Z','Tau','Score',
                'Precio_actual','TP_est','SL_est','PIDelta_mean','PIDelta_std',
                'Volume_mean','Volume_std','Corr_BTC','Corr_ETH','Corr_SOL']
        combined = combined[[c for c in cols if c in combined.columns]]
        combined.to_csv("backtest_completo.txt", index=False, sep='\t')
        print("\n✅ Archivo combinado generado: 'backtest_completo.txt'")
    else:
        pd.DataFrame(columns=['Symbol','TF','Edge_%','HitRate','Z','Score'])\
          .to_csv("backtest_completo.txt", index=False, sep='\t')
        print("⚠️ No se generaron datos. Archivo vacío creado.")
    print("\n✅ Proceso completado.")
