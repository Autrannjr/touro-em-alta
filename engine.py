"""Motor do screener de abertura: baixa dados, pontua e monta o plano de cada operação.

Importante: isto é uma ferramenta de triagem. Ela ordena ativos por um conjunto de critérios
técnicos; não prevê o mercado e não garante resultado.
"""
from __future__ import annotations

import datetime as dt
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = ZoneInfo("America/Sao_Paulo")
HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_CSV = os.path.join(HERE, "universe.csv")

# liquidez mínima (média de 20 dias, em R$ por dia)
MIN_FIN_B3 = 5_000_000
MIN_FIN_BDR = 1_000_000

# pregão B3 (ações): 10:00 a 17:55 -> 475 minutos
SESSION_START = dt.time(10, 0)
SESSION_MIN = 475


# ----------------------------------------------------------------------------- indicadores
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------------------------------------------------------- dados
def load_universe(path: str = UNIVERSE_CSV) -> pd.DataFrame:
    u = pd.read_csv(path, dtype=str).fillna("")
    u["ticker"] = u["ticker"].str.strip().str.upper()
    u["tipo"] = u["tipo"].str.strip().str.upper()
    u["subjacente"] = u["subjacente"].str.strip().str.upper()
    return u[u["ticker"] != ""].reset_index(drop=True)


def yahoo_symbol(ticker: str, tipo: str) -> str:
    return ticker + ".SA"


def download(symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """Baixa candles diários do Yahoo Finance. Retorna {simbolo: df(open,high,low,close,volume)}."""
    import yfinance as yf  # importado aqui para o motor poder ser testado sem internet

    symbols = sorted(set(symbols))
    out: dict[str, pd.DataFrame] = {}
    # em lotes para não estourar limite do Yahoo
    for i in range(0, len(symbols), 40):
        chunk = symbols[i : i + 40]
        raw = yf.download(
            chunk,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        for s in chunk:
            try:
                df = raw[s] if len(chunk) > 1 else raw
                df = df.dropna(subset=["Close"]).rename(columns=str.lower)
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                if len(df) >= 60:
                    out[s] = df
            except Exception:
                continue
    return out


# ----------------------------------------------------------------------------- pontuação
def _trend_points(c: pd.Series) -> tuple[float, list[str]]:
    e9, e21, e50 = ema(c, 9).iloc[-1], ema(c, 21).iloc[-1], ema(c, 50).iloc[-1]
    last = c.iloc[-1]
    pts, why = 0.0, []
    if last > e21:
        pts += 8
    if e9 > e21:
        pts += 6
    if e21 > e50:
        pts += 6
    if last > e9:
        pts += 5
    if pts >= 20:
        why.append("tendência de alta (preço acima das médias 9/21/50)")
    elif pts <= 6:
        why.append("tendência fraca/baixa")
    return pts, why


def session_fraction(now: dt.datetime) -> float:
    start = dt.datetime.combine(now.date(), SESSION_START, tzinfo=TZ)
    mins = (now - start).total_seconds() / 60
    return float(min(max(mins / SESSION_MIN, 0.05), 1.0))


def analyze(
    df: pd.DataFrame,
    now: dt.datetime,
    under_df: pd.DataFrame | None = None,
) -> dict | None:
    """Pontua um ativo. `df` = candles do ativo negociado na B3; `under_df` = candles da ação-mãe (BDR)."""
    if df is None or len(df) < 60:
        return None
    d = df.copy()
    has_today = d.index[-1].date() == now.date()
    hist = d.iloc[:-1] if has_today else d
    if len(hist) < 55:
        return None

    last = d.iloc[-1]
    prev = hist.iloc[-1]
    price = float(last["close"])

    c = hist["close"]
    atr_v = float(atr(hist).iloc[-1])
    atr_pct = atr_v / float(prev["close"])
    rsi_v = float(rsi(c).iloc[-1])
    avg_vol = float(hist["volume"].tail(20).mean())
    avg_fin = float((hist["close"] * hist["volume"]).tail(20).mean())
    hi20 = float(hist["high"].tail(20).max())
    ret5 = float(c.iloc[-1] / c.iloc[-6] - 1)

    gap = float(last["open"] / prev["close"] - 1) if has_today else None

    if has_today and last["volume"] > 0 and avg_vol > 0:
        rvol = float(last["volume"] / (avg_vol * session_fraction(now)))
    else:
        rvol = float(prev["volume"] / avg_vol) if avg_vol > 0 else 0.0

    why: list[str] = []

    # tendência (25)
    t_pts, t_why = _trend_points(c)
    under_ret1 = None
    if under_df is not None and len(under_df) >= 60:
        u_c = under_df["close"]
        u_pts, _ = _trend_points(u_c)
        t_pts = (t_pts + u_pts) / 2
        under_ret1 = float(u_c.iloc[-1] / u_c.iloc[-2] - 1)
    why += t_why

    # momento (15)
    if 50 <= rsi_v <= 68:
        m_pts = 15
        why.append(f"RSI saudável ({rsi_v:.0f})")
    elif 45 <= rsi_v < 50 or 68 < rsi_v <= 75:
        m_pts = 9
    elif 40 <= rsi_v < 45:
        m_pts = 4
    else:
        m_pts = 0
        if rsi_v > 75:
            why.append(f"RSI esticado ({rsi_v:.0f}), risco de devolução")
    if ret5 < -0.03:
        m_pts = min(m_pts, 3)

    # volume relativo (20)
    if rvol >= 2:
        v_pts = 20
    elif rvol >= 1.5:
        v_pts = 16
    elif rvol >= 1.2:
        v_pts = 11
    elif rvol >= 1.0:
        v_pts = 6
    else:
        v_pts = 2
    if rvol >= 1.5:
        why.append(f"volume {rvol:.1f}x acima do normal")

    # gap (15)
    if gap is None:
        g_pts = 7
        why.append("pré-abertura: ainda sem gap (reavalie após a abertura)")
    else:
        g = gap * 100
        if 0.3 <= g <= 3:
            g_pts = 15
            why.append(f"gap de alta {g:+.1f}%")
        elif 0 < g < 0.3:
            g_pts = 8
        elif 3 < g <= 5:
            g_pts = 7
            why.append(f"gap grande ({g:+.1f}%), cuidado com exaustão")
        elif g > 5:
            g_pts = 2
            why.append(f"gap muito grande ({g:+.1f}%), alto risco de devolução")
        elif -0.5 <= g <= 0:
            g_pts = 5
        else:
            g_pts = 0
            why.append(f"gap de baixa {g:+.1f}%")

    # volatilidade diária (15)
    ap = atr_pct * 100
    if 1.5 <= ap <= 4:
        a_pts = 15
    elif 1.0 <= ap < 1.5 or 4 < ap <= 6:
        a_pts = 9
    else:
        a_pts = 3
    if ap < 1.0:
        why.append(f"pouca volatilidade (ATR {ap:.1f}%)")

    # proximidade da máxima de 20 dias (10)
    prox = price / hi20
    if prox >= 0.99:
        b_pts = 10
        why.append("perto/acima da máxima de 20 dias (rompimento)")
    elif prox >= 0.97:
        b_pts = 7
    elif prox >= 0.94:
        b_pts = 4
    else:
        b_pts = 0

    score = t_pts + m_pts + v_pts + g_pts + a_pts + b_pts

    return {
        "price": price,
        "gap": gap,
        "rvol": rvol,
        "atr": atr_v,
        "atr_pct": atr_pct,
        "rsi": rsi_v,
        "ret5": ret5,
        "avg_fin": avg_fin,
        "under_ret1": under_ret1,
        "score": float(round(score, 1)),
        "why": why,
        "has_today": has_today,
    }


def trade_plan(
    price: float,
    atr_v: float,
    capital: float,
    risk_pct: float,
    sl_mult: float = 0.6,
    tp_mult: float = 0.9,
) -> dict:
    """Entrada de referência = preço atual. Stop e alvo em múltiplos do ATR diário."""
    entry = round(price, 2)
    stop = round(entry - sl_mult * atr_v, 2)
    target = round(entry + tp_mult * atr_v, 2)
    risk_ps = max(entry - stop, 0.01)
    risk_brl = capital * risk_pct / 100.0
    qty = math.floor(risk_brl / risk_ps)
    qty = min(qty, math.floor(capital / entry)) if entry > 0 else 0
    if qty >= 100:
        qty = qty // 100 * 100  # lote padrão
    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "qty": int(max(qty, 0)),
        "risk_brl": round(qty * risk_ps, 2) if qty > 0 else 0.0,
        "rr": round((target - entry) / risk_ps, 2),
    }


# ----------------------------------------------------------------------------- orquestração
def run_scan(
    capital: float = 10_000.0,
    risk_pct: float = 1.0,
    top_n: int = 5,
    include_bdr: bool = True,
    sl_mult: float = 0.6,
    tp_mult: float = 0.9,
    now: dt.datetime | None = None,
    downloader=download,
    universe: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    now = now or dt.datetime.now(TZ)
    u = universe if universe is not None else load_universe()
    if not include_bdr:
        u = u[u["tipo"] == "B3"]

    symbols = [yahoo_symbol(t, k) for t, k in zip(u["ticker"], u["tipo"])]
    symbols += [s for s in u["subjacente"] if s]
    data = downloader(symbols)

    rows = []
    missing, illiquid = [], []
    for _, r in u.iterrows():
        sym = yahoo_symbol(r["ticker"], r["tipo"])
        df = data.get(sym)
        if df is None:
            missing.append(r["ticker"])
            continue
        under = data.get(r["subjacente"]) if r["subjacente"] else None
        res = analyze(df, now, under)
        if res is None:
            missing.append(r["ticker"])
            continue
        min_fin = MIN_FIN_BDR if r["tipo"] == "BDR" else MIN_FIN_B3
        if res["avg_fin"] < min_fin:
            illiquid.append(r["ticker"])
            continue
        plan = trade_plan(res["price"], res["atr"], capital, risk_pct, sl_mult, tp_mult)
        rows.append({"ticker": r["ticker"], "tipo": r["tipo"], "subjacente": r["subjacente"], **res, **plan})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        out.index = out.index + 1
        out = out.head(top_n)

    info = {
        "now": now,
        "missing": missing,
        "illiquid": illiquid,
        "scanned": len(rows) + len(illiquid),
        "has_today": bool(rows and rows[0]["has_today"]),
    }
    return out, info


def headlines(ticker: str, tipo: str, subjacente: str, n: int = 3) -> list[str]:
    """Manchetes recentes (melhor esforço; o Yahoo nem sempre devolve notícias de ações da B3)."""
    try:
        import yfinance as yf

        sym = subjacente if tipo == "BDR" and subjacente else ticker + ".SA"
        items = yf.Ticker(sym).news or []
        titles = []
        for it in items[: n * 2]:
            t = it.get("title") or (it.get("content") or {}).get("title")
            if t:
                titles.append(t)
        return titles[:n]
    except Exception:
        return []
