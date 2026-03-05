import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime
import warnings
import sys

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================

TIMEFRAMES = ["1m","3m","5m","15m","30m","1h","4h"]
LOOKBACK_CANDLES = 300

TP_PCT = 0.02
SL_PCT = 0.01

ASSETS = ["BTC","ETH","SOL"]
BASE_SYMBOLS = ["BTC","ETH","SOL"]

HEADERS = {"User-Agent":"Mozilla/5.0"}

DATA_CACHE = {}

CATEGORY_MAP = {
"1m":"Scalp",
"3m":"Scalp",
"5m":"Scalp",
"15m":"Scalp",
"30m":"Intraday",
"1h":"Intraday",
"4h":"Intraday",
"1d":"Swing"
}

# ============================================================
# UTILIDADES
# ============================================================

def log(msg):
    print(f"[{datetime.utcnow()}] {msg}")
    sys.stdout.flush()

def print_block(title,data):

    print("\n"+"="*70)
    print(title)
    print("="*70)

    if isinstance(data,dict):
        print(json.dumps(data,indent=4,default=str))
    else:
        print(data)

    print("="*70+"\n")


# ============================================================
# DESCARGA DATOS
# ============================================================

def fetch_klines_kucoin(symbol,tf,limit=500):

    tf_map={
    "1m":"1min","3m":"3min","5m":"5min","15m":"15min",
    "30m":"30min","1h":"1hour","4h":"4hour","1d":"1day"
    }

    interval=tf_map.get(tf)

    if not interval:
        return None

    url="https://api.kucoin.com/api/v1/market/candles"

    params={
    "symbol":symbol,
    "type":interval,
    "limit":limit
    }

    try:

        r=requests.get(url,params=params,headers=HEADERS,timeout=10)

        if r.status_code!=200:
            return None

        data=r.json()

        if data["code"]!="200000":
            return None

        df=pd.DataFrame(
        data["data"],
        columns=["timestamp","open","close","high","low","volume","turnover"]
        )

        df["timestamp"]=pd.to_datetime(df["timestamp"].astype(float),unit="s")

        df.set_index("timestamp",inplace=True)

        df=df.astype(float)

        df=df[["open","high","low","close","volume"]]

        df=df.sort_index()

        log(f"DATA KuCoin {symbol} {tf} candles={len(df)}")

        return df

    except Exception as e:

        log(f"ERROR KuCoin {symbol} {e}")

        return None


def fetch_klines(symbol,tf):

    key=f"{symbol}_{tf}"

    if key in DATA_CACHE:
        return DATA_CACHE[key]

    kucoin_symbol=f"{symbol}-USDT"

    df=fetch_klines_kucoin(kucoin_symbol,tf,LOOKBACK_CANDLES)

    if df is not None and len(df)>50:

        DATA_CACHE[key]=df

        return df

    log(f"NO DATA {symbol} {tf}")

    return None


# ============================================================
# INDICADORES
# ============================================================

def normalize(series,window=50):

    mean=series.rolling(window).mean()
    std=series.rolling(window).std()

    return (series-mean)/(std+1e-9)


def tension_235(series):

    ema2=series.ewm(span=2).mean()
    ema3=series.ewm(span=3).mean()
    ema5=series.ewm(span=5).mean()

    return (ema2-ema3).abs()+(ema3-ema5).abs()


def compute_edge(price,tension,k):

    idx=price.index.intersection(tension.index)

    price=price.loc[idx]
    tension=tension.loc[idx]

    mask=tension>tension.quantile(0.85)

    if mask.sum()==0:
        return np.nan,np.nan

    future=(price.shift(-k)-price)/price

    edge=future[mask].mean()

    hitrate=(future[mask]>0).mean()

    return edge,hitrate


def compute_pidelta(price):

    r=price.pct_change().fillna(0)

    struct=r.rolling(20).mean()

    hist=r.ewm(span=20).mean()

    return struct-hist


# ============================================================
# ANALISIS
# ============================================================

def analyze(symbol,tf):

    df=fetch_klines(symbol,tf)

    if df is None or len(df)<50:

        log(f"SKIP {symbol} {tf}")

        return None,None

    price=df["close"]

    volume=df["volume"]

    S=normalize(price)

    T=tension_235(S).dropna()

    if len(T)<30:
        return None,None

    pidelta=compute_pidelta(price)

    k_values=[1,2,3,5,8,13]

    best=None
    best_score=-1

    for k in k_values:

        edge,hit=compute_edge(price,T,k)

        if np.isnan(edge):
            continue

        mc=[]

        base=S.dropna()

        for _ in range(20):

            perm=np.random.permutation(base.values)

            t_mc=tension_235(pd.Series(perm,index=base.index))

            e,_=compute_edge(price,t_mc,k)

            if not np.isnan(e):
                mc.append(e)

        if len(mc)<5:
            continue

        mc_mean=np.mean(mc)
        mc_std=np.std(mc)

        Z=(edge-mc_mean)/(mc_std+1e-9)

        tau=10

        score=abs(Z)*abs(edge)/(tau+1e-9)

        if score>best_score:

            best_score=score

            best={
            "Symbol":symbol,
            "TF":tf,
            "Categoria":CATEGORY_MAP.get(tf,"Other"),
            "k":k,
            "Edge":float(edge),
            "HitRate":float(hit),
            "Zscore":float(Z),
            "Score":float(score),
            "Price":float(price.iloc[-1]),
            "TP":float(price.iloc[-1]*(1+TP_PCT)),
            "SL":float(price.iloc[-1]*(1-SL_PCT)),
            "VolumeMean":float(volume.mean()),
            "VolumeSTD":float(volume.std())
            }

    return best,pidelta


# ============================================================
# ESCANEO
# ============================================================

def scan_tf(tf):

    log(f"SCAN TF {tf}")

    results=[]

    base_pidelta={}

    for base in BASE_SYMBOLS:

        r,p=analyze(base,tf)

        if p is not None:
            base_pidelta[base]=p

    for asset in ASSETS:

        res,pid=analyze(asset,tf)

        if res:

            for base in BASE_SYMBOLS:

                if base in base_pidelta and pid is not None:

                    idx=pid.index.intersection(base_pidelta[base].index)

                    if len(idx)>10:

                        corr=pid.loc[idx].corr(base_pidelta[base].loc[idx])

                    else:

                        corr=np.nan

                else:

                    corr=np.nan

                res[f"Corr_{base}"]=float(corr)

            results.append(res)

            print_block("RESULT",res)

        time.sleep(0.2)

    return results


# ============================================================
# MAIN
# ============================================================

if __name__=="__main__":

    print_block("START BACKTEST","TENSION 2-3-5")

    all_results=[]

    for tf in TIMEFRAMES:

        r=scan_tf(tf)

        all_results.extend(r)

    if len(all_results)>0:

        df=pd.DataFrame(all_results)

        df.to_csv("backtest_completo.txt",sep="\t",index=False)

        print_block("SUMMARY",df.describe())

    else:

        log("NO RESULTS")

    log("FINISHED")
