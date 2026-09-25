from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone

# ============================================================
# COMMODITY ROTATION BOT v1.2C
# LONG ONLY - NO SHORT - NO LEVERAGE - NO AUTOMATIC ORDERS
# ============================================================

app = FastAPI(
    title="Commodity Rotation Bot",
    description="Long-only commodity ETF/ETP rotation system",
    version="1.2C",
)

MODEL_NAME = "COMMODITY-ROTATION-V1.2C"

ASSETS = {
    "GLD": {"name": "Gold", "tr_name": "Altın", "group": "METALS"},
    "SLV": {"name": "Silver", "tr_name": "Gümüş", "group": "METALS"},
    "CPER": {"name": "Copper", "tr_name": "Bakır", "group": "METALS"},
    "USO": {"name": "Crude Oil", "tr_name": "Petrol", "group": "ENERGY"},
    "UNG": {"name": "Natural Gas", "tr_name": "Doğal Gaz", "group": "ENERGY"},
    "DBA": {"name": "Agriculture Basket", "tr_name": "Tarım Sepeti", "group": "AGRICULTURE"},
    "WEAT": {"name": "Wheat", "tr_name": "Buğday", "group": "AGRICULTURE"},
    "CORN": {"name": "Corn", "tr_name": "Mısır", "group": "AGRICULTURE"},
    "SOYB": {"name": "Soybeans", "tr_name": "Soya", "group": "AGRICULTURE"},
    "DBC": {"name": "Broad Commodities", "tr_name": "Geniş Emtia Sepeti", "group": "BROAD_COMMODITY"},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_float(value, digits=4):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    except Exception:
        return None


def safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


# ============================================================
# DATA
# ============================================================

def download_history(symbol, period="10y"):
    symbol = symbol.upper()

    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"{symbol} için veri alınamadı.")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        except Exception:
            df.columns = [
                x[0] if isinstance(x, tuple) else x
                for x in df.columns
            ]

    required = ["Open", "High", "Low", "Close", "Volume"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"{symbol}: {column} bulunamadı.")

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if len(df) < 220:
        raise ValueError(f"{symbol}: yeterli tarihsel veri yok.")

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(100)


def calculate_atr(df, period=14):
    previous_close = df["Close"].shift(1)

    tr1 = (df["High"] - df["Low"]).abs()
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_indicators(df):
    df = df.copy()
    close = df["Close"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA100"] = close.ewm(span=100, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    df["MOM20"] = close.pct_change(20) * 100
    df["MOM60"] = close.pct_change(60) * 100
    df["MOM120"] = close.pct_change(120) * 100

    df["RSI14"] = calculate_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    df["ATR14"] = calculate_atr(df, 14)
    df["ATR_PCT"] = df["ATR14"] / close * 100

    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VOLUME_RATIO"] = df["Volume"] / df["VOL20"]

    df["DIST_EMA20"] = (close / df["EMA20"] - 1) * 100
    df["DIST_EMA50"] = (close / df["EMA50"] - 1) * 100
    df["DIST_EMA200"] = (close / df["EMA200"] - 1) * 100

    high252 = close.rolling(252).max()
    df["DRAWDOWN_252"] = (close / high252 - 1) * 100

    return df


# ============================================================
# SCORE
# ============================================================

def calculate_score(row):
    score = 0
    reasons = []
    warnings = []

    close = row["Close"]
    ema20 = row["EMA20"]
    ema50 = row["EMA50"]
    ema200 = row["EMA200"]

    mom20 = row["MOM20"]
    mom60 = row["MOM60"]
    mom120 = row["MOM120"]

    rsi = row["RSI14"]
    atr_pct = row["ATR_PCT"]

    macd = row["MACD"]
    macd_signal = row["MACD_SIGNAL"]
    macd_hist = row["MACD_HIST"]

    # TREND - 40
    if close > ema200:
        score += 15
        reasons.append("Fiyat EMA200 üzerinde")
    else:
        warnings.append("Fiyat EMA200 altında")

    if ema50 > ema200:
        score += 10
        reasons.append("EMA50 > EMA200")
    else:
        warnings.append("EMA50 EMA200 üzerinde değil")

    if ema20 > ema50:
        score += 10
        reasons.append("EMA20 > EMA50")
    else:
        warnings.append("Kısa vadeli trend zayıf")

    if close > ema20:
        score += 5
        reasons.append("Fiyat EMA20 üzerinde")

    # MOMENTUM - 30
    if mom20 > 0:
        score += 5
        reasons.append("20 günlük momentum pozitif")

    if mom60 > 0:
        score += 10
        reasons.append("60 günlük momentum pozitif")
    else:
        warnings.append("60 günlük momentum negatif")

    if mom120 > 0:
        score += 10
        reasons.append("120 günlük momentum pozitif")
    else:
        warnings.append("120 günlük momentum negatif")

    if mom20 > 0 and mom60 > 0 and mom120 > 0:
        score += 5
        reasons.append("Momentum zaman dilimleri uyumlu")

    # MACD - 10
    if macd > macd_signal:
        score += 5
        reasons.append("MACD signal üzerinde")
    else:
        warnings.append("MACD kısa vadeli momentum zayıflıyor")

    if macd_hist > 0:
        score += 5
        reasons.append("MACD histogram pozitif")

    # RSI - 10
    if 45 <= rsi <= 65:
        score += 10
        reasons.append("RSI sağlıklı momentum bölgesinde")

    elif 40 <= rsi < 45:
        score += 5

    elif 65 < rsi <= 70:
        score += 5
        warnings.append("RSI yükselmiş durumda")

    elif rsi > 70:
        warnings.append("RSI aşırı alım bölgesinde")

    else:
        warnings.append("RSI zayıf")

    # VOLATILITY - 10
    if atr_pct <= 3:
        score += 10
        reasons.append("Volatilite kontrollü")

    elif atr_pct <= 5:
        score += 5
        reasons.append("Volatilite orta düzeyde")

    else:
        warnings.append("Volatilite yüksek")

    score = max(0, min(int(score), 100))

    return score, reasons, warnings


def determine_signal(row, score):
    close = row["Close"]
    ema20 = row["EMA20"]
    ema50 = row["EMA50"]
    ema200 = row["EMA200"]

    mom60 = row["MOM60"]
    mom120 = row["MOM120"]
    rsi = row["RSI14"]

    long_term_trend = (
        close > ema200
        and ema50 > ema200
    )

    medium_term_trend = ema20 > ema50

    momentum_ok = (
        mom60 > 0
        and mom120 > 0
    )

    rsi_ok = 40 <= rsi <= 70

    if (
        score >= 75
        and long_term_trend
        and medium_term_trend
        and momentum_ok
        and rsi_ok
    ):
        return "AL_ADAYI"

    if score >= 60 and long_term_trend:
        return "IZLE"

    return "BEKLE"


# ============================================================
# SINGLE ANALYSIS
# ============================================================

def analyze_symbol(symbol, period="10y"):
    symbol = symbol.upper()

    if symbol not in ASSETS:
        raise ValueError(f"{symbol} ürün listesinde yok.")

    df = download_history(symbol, period)
    df = calculate_indicators(df)

    clean = df.dropna().copy()

    if clean.empty:
        raise ValueError(f"{symbol}: gösterge üretilemedi.")

    row = clean.iloc[-1]

    score, reasons, warnings = calculate_score(row)
    signal = determine_signal(row, score)

    asset = ASSETS[symbol]

    return {
        "model": MODEL_NAME,
        "symbol": symbol,
        "name": asset["name"],
        "tr_name": asset["tr_name"],
        "group": asset["group"],
        "date": str(clean.index[-1].date()),
        "close": safe_float(row["Close"], 2),
        "signal": signal,
        "score": score,

        "trend": {
            "ema20": safe_float(row["EMA20"], 2),
            "ema50": safe_float(row["EMA50"], 2),
            "ema100": safe_float(row["EMA100"], 2),
            "ema200": safe_float(row["EMA200"], 2),
            "distance_ema20_pct": safe_float(row["DIST_EMA20"], 2),
            "distance_ema50_pct": safe_float(row["DIST_EMA50"], 2),
            "distance_ema200_pct": safe_float(row["DIST_EMA200"], 2),
        },

        "momentum": {
            "20d_pct": safe_float(row["MOM20"], 2),
            "60d_pct": safe_float(row["MOM60"], 2),
            "120d_pct": safe_float(row["MOM120"], 2),
        },

        "rsi14": safe_float(row["RSI14"], 2),

        "macd": {
            "line": safe_float(row["MACD"], 4),
            "signal": safe_float(row["MACD_SIGNAL"], 4),
            "histogram": safe_float(row["MACD_HIST"], 4),
        },

        "risk": {
            "atr14": safe_float(row["ATR14"], 4),
            "atr_pct": safe_float(row["ATR_PCT"], 2),
            "drawdown_1y_pct": safe_float(row["DRAWDOWN_252"], 2),
        },

        "volume": {
            "current": safe_int(row["Volume"]),
            "average_20d": safe_int(row["VOL20"]),
            "ratio": safe_float(row["VOLUME_RATIO"], 2),
        },

        "reasons": reasons,
        "warnings": warnings,
        "generated_at_utc": utc_now(),
    }


# ============================================================
# SCAN
# ============================================================

def scan_all_assets():
    results = []
    errors = []

    for symbol in ASSETS:
        try:
            results.append(
                analyze_symbol(symbol, "10y")
            )
        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "error": str(exc),
            })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )

    for index, item in enumerate(results, start=1):
        item["overall_rank"] = index

    return {
        "model": MODEL_NAME,
        "generated_at_utc": utc_now(),
        "asset_count": len(results),
        "buy_candidate_count": sum(
            x["signal"] == "AL_ADAYI"
            for x in results
        ),
        "watch_candidate_count": sum(
            x["signal"] == "IZLE"
            for x in results
        ),
        "wait_candidate_count": sum(
            x["signal"] == "BEKLE"
            for x in results
        ),
        "ranking": results,
        "groups": {
            "metals": [
                x for x in results
                if x["group"] == "METALS"
            ],
            "energy": [
                x for x in results
                if x["group"] == "ENERGY"
            ],
            "agriculture": [
                x for x in results
                if x["group"] == "AGRICULTURE"
            ],
            "broad_commodity_reference": [
                x for x in results
                if x["group"] == "BROAD_COMMODITY"
            ],
        },
        "errors": errors,
    }


# ============================================================
# V1.2A SINGLE-ASSET BACKTEST
# ============================================================

def run_backtest(
    symbol,
    years=10,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
):
    symbol = symbol.upper()

    if symbol not in ASSETS:
        raise ValueError(f"{symbol} ürün listesinde yok.")

    df = download_history(symbol, f"{years}y")
    df = calculate_indicators(df)
    df = df.dropna().copy()

    if len(df) < 250:
        raise ValueError("Backtest için yeterli veri yok.")

    scores = []
    signals = []

    for _, row in df.iterrows():
        score, _, _ = calculate_score(row)
        scores.append(score)
        signals.append(
            determine_signal(row, score)
        )

    df["SCORE"] = scores
    df["MODEL_SIGNAL"] = signals

    cost = transaction_cost_pct / 100

    in_position = False
    entry_price = None
    entry_date = None
    entry_score_value = None

    trades = []
    equity = 1.0

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_row = df.iloc[i + 1]
        next_date = df.index[i + 1]

        score = int(current["SCORE"])
        signal = current["MODEL_SIGNAL"]
        next_open = float(next_row["Open"])

        if not in_position:

            if (
                score >= entry_score
                and signal == "AL_ADAYI"
            ):
                entry_price = next_open * (1 + cost)
                entry_date = next_date
                entry_score_value = score
                in_position = True

        else:

            close = float(current["Close"])
            ema20 = float(current["EMA20"])
            ema50 = float(current["EMA50"])
            ema100 = float(current["EMA100"])
            ema200 = float(current["EMA200"])

            exit_condition = (
                score <= exit_score
                or ema20 < ema50
                or close < ema100
                or ema50 < ema200
            )

            if exit_condition:
                exit_price = next_open * (1 - cost)

                trade_return = (
                    exit_price / entry_price - 1
                )

                equity *= 1 + trade_return

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(next_date.date()),
                    "entry_price": safe_float(entry_price, 2),
                    "exit_price": safe_float(exit_price, 2),
                    "entry_score": entry_score_value,
                    "exit_score": score,
                    "return_pct": safe_float(
                        trade_return * 100,
                        2,
                    ),
                    "holding_days": (
                        next_date - entry_date
                    ).days,
                })

                in_position = False
                entry_price = None
                entry_date = None
                entry_score_value = None

    if in_position:
        final = df.iloc[-1]
        final_date = df.index[-1]

        exit_price = (
            float(final["Close"])
            * (1 - cost)
        )

        trade_return = (
            exit_price / entry_price - 1
        )

        equity *= 1 + trade_return

        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(final_date.date()),
            "entry_price": safe_float(entry_price, 2),
            "exit_price": safe_float(exit_price, 2),
            "entry_score": entry_score_value,
            "exit_score": safe_int(final["SCORE"]),
            "return_pct": safe_float(
                trade_return * 100,
                2,
            ),
            "holding_days": (
                final_date - entry_date
            ).days,
            "forced_exit_at_test_end": True,
        })

    returns = [
        x["return_pct"]
        for x in trades
        if x["return_pct"] is not None
    ]

    winners = [x for x in returns if x > 0]
    losers = [x for x in returns if x <= 0]

    trade_count = len(trades)

    if trade_count:
        win_rate = (
            len(winners)
            / trade_count
            * 100
        )

        avg_trade = np.mean(returns)
        median_trade = np.median(returns)

        avg_holding = np.mean([
            x["holding_days"]
            for x in trades
        ])

    else:
        win_rate = 0
        avg_trade = 0
        median_trade = 0
        avg_holding = 0

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0

    total_return = (
        equity - 1
    ) * 100

    first_open = float(df.iloc[0]["Open"])
    last_close = float(df.iloc[-1]["Close"])

    buy_hold_return = (
        last_close / first_open - 1
    ) * 100

    running_equity = 1.0
    peak = 1.0
    max_drawdown = 0

    for trade in trades:
        running_equity *= (
            1
            + trade["return_pct"] / 100
        )

        peak = max(peak, running_equity)

        drawdown = (
            running_equity / peak - 1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    start_date = df.index[0]
    end_date = df.index[-1]

    elapsed_years = (
        (end_date - start_date).days
        / 365.25
    )

    if elapsed_years > 0 and equity > 0:
        cagr = (
            equity ** (1 / elapsed_years)
            - 1
        ) * 100
    else:
        cagr = None

    return {
        "model": MODEL_NAME,
        "symbol": symbol,
        "commodity": ASSETS[symbol]["tr_name"],
        "group": ASSETS[symbol]["group"],

        "test_period": {
            "start": str(start_date.date()),
            "end": str(end_date.date()),
            "years_requested": years,
        },

        "settings": {
            "entry_score": entry_score,
            "exit_score": exit_score,
            "transaction_cost_pct_each_side":
                transaction_cost_pct,
            "execution":
                "Signal at close, next trading day open",
            "exit_model":
                "V1.2A_SLOW_EXIT",
            "exit_rules": [
                "score <= exit_score",
                "EMA20 < EMA50",
                "Close < EMA100",
                "EMA50 < EMA200",
            ],
            "long_only": True,
            "short": False,
            "leverage": False,
        },

        "performance": {
            "trade_count": trade_count,
            "win_rate_pct": safe_float(win_rate, 2),
            "average_trade_pct": safe_float(avg_trade, 2),
            "median_trade_pct": safe_float(median_trade, 2),
            "average_holding_days":
                safe_float(avg_holding, 1),
            "profit_factor":
                safe_float(profit_factor, 3),
            "strategy_total_return_pct":
                safe_float(total_return, 2),
            "strategy_cagr_pct":
                safe_float(cagr, 2),
            "max_drawdown_trade_level_pct":
                safe_float(max_drawdown, 2),
            "buy_hold_return_pct":
                safe_float(buy_hold_return, 2),
        },

        "trades": trades,
        "generated_at_utc": utc_now(),
    }


def run_backtest_all(
    years=10,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
):
    results = []
    errors = []

    for symbol in ASSETS:
        try:
            bt = run_backtest(
                symbol=symbol,
                years=years,
                entry_score=entry_score,
                exit_score=exit_score,
                transaction_cost_pct=
                    transaction_cost_pct,
            )

            p = bt["performance"]

            results.append({
                "symbol": symbol,
                "commodity":
                    ASSETS[symbol]["tr_name"],
                "group":
                    ASSETS[symbol]["group"],
                "trade_count":
                    p["trade_count"],
                "win_rate_pct":
                    p["win_rate_pct"],
                "profit_factor":
                    p["profit_factor"],
                "average_trade_pct":
                    p["average_trade_pct"],
                "median_trade_pct":
                    p["median_trade_pct"],
                "average_holding_days":
                    p["average_holding_days"],
                "strategy_total_return_pct":
                    p["strategy_total_return_pct"],
                "strategy_cagr_pct":
                    p["strategy_cagr_pct"],
                "max_drawdown_pct":
                    p["max_drawdown_trade_level_pct"],
                "buy_hold_return_pct":
                    p["buy_hold_return_pct"],
            })

        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "error": str(exc),
            })

    def pf_sort_value(item):
        pf = item["profit_factor"]
        return -999999 if pf is None else pf

    results = sorted(
        results,
        key=pf_sort_value,
        reverse=True,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):
        item["backtest_rank"] = index

    profitable_count = sum(
        1
        for item in results
        if (
            item["strategy_total_return_pct"]
            is not None
            and
            item["strategy_total_return_pct"] > 0
        )
    )

    pf_above_one_count = sum(
        1
        for item in results
        if (
            item["profit_factor"]
            is not None
            and
            item["profit_factor"] > 1
        )
    )

    beat_buy_hold_count = sum(
        1
        for item in results
        if (
            item["strategy_total_return_pct"]
            is not None
            and
            item["buy_hold_return_pct"]
            is not None
            and
            item["strategy_total_return_pct"]
            >
            item["buy_hold_return_pct"]
        )
    )

    return {
        "model": MODEL_NAME,
        "test": "BACKTEST_ALL",
        "generated_at_utc": utc_now(),

        "settings": {
            "years": years,
            "entry_score": entry_score,
            "exit_score": exit_score,
            "transaction_cost_pct_each_side":
                transaction_cost_pct,
            "execution":
                "Signal at close, next trading day open",
            "exit_model":
                "V1.2A_SLOW_EXIT",
            "long_only": True,
            "short": False,
            "leverage": False,
        },

        "summary": {
            "asset_count_tested":
                len(results),
            "error_count":
                len(errors),
            "profitable_strategy_count":
                profitable_count,
            "profit_factor_above_1_count":
                pf_above_one_count,
            "strategy_beats_buy_hold_count":
                beat_buy_hold_count,
        },

        "results": results,
        "errors": errors,
    }


# ============================================================
# V1.2C STICKY PORTFOLIO ROTATION
# ============================================================

def prepare_rotation_data(years=10):
    prepared = {}
    errors = []

    for symbol in ASSETS:
        try:
            df = download_history(
                symbol,
                f"{years}y",
            )

            df = calculate_indicators(df)
            df = df.dropna().copy()

            scores = []
            signals = []

            for _, row in df.iterrows():
                score, _, _ = calculate_score(row)

                scores.append(score)
                signals.append(
                    determine_signal(
                        row,
                        score,
                    )
                )

            df["SCORE"] = scores
            df["MODEL_SIGNAL"] = signals

            prepared[symbol] = df

        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "error": str(exc),
            })

    if not prepared:
        raise ValueError(
            "Rotation backtest için veri hazırlanamadı."
        )

    common_dates = None

    for df in prepared.values():
        dates = set(df.index)

        if common_dates is None:
            common_dates = dates
        else:
            common_dates = (
                common_dates.intersection(dates)
            )

    common_dates = sorted(common_dates)

    if len(common_dates) < 250:
        raise ValueError(
            "Rotation backtest için yeterli ortak tarih yok."
        )

    return prepared, common_dates, errors


def rotation_candidates(
    prepared,
    signal_date,
    entry_score,
):
    candidates = []

    for symbol, df in prepared.items():

        if signal_date not in df.index:
            continue

        row = df.loc[signal_date]

        score = int(row["SCORE"])
        signal = row["MODEL_SIGNAL"]

        if (
            score >= entry_score
            and signal == "AL_ADAYI"
        ):
            candidates.append({
                "symbol": symbol,
                "score": score,
                "mom120":
                    float(row["MOM120"]),
                "mom60":
                    float(row["MOM60"]),
                "mom20":
                    float(row["MOM20"]),
            })

    candidates.sort(
        key=lambda x: (
            x["score"],
            x["mom120"],
            x["mom60"],
            x["mom20"],
        ),
        reverse=True,
    )

    return candidates


def rotation_exit_required(
    row,
    exit_score,
):
    return (
        int(row["SCORE"]) <= exit_score
        or
        float(row["EMA20"])
        <
        float(row["EMA50"])
        or
        float(row["Close"])
        <
        float(row["EMA100"])
        or
        float(row["EMA50"])
        <
        float(row["EMA200"])
    )


def run_rotation_variant(
    prepared,
    common_dates,
    top_n,
    entry_score,
    exit_score,
    transaction_cost_pct,
):
    """
    V1.2C STICKY ROTATION

    Temel fark:
    - Mevcut pozisyon, başka bir emtia daha yüksek skora çıktı diye satılmaz.
    - Pozisyon yalnızca V1.2A slow-exit koşulu oluşursa kapanır.
    - Boşalan slot, o günkü en güçlü uygun AL_ADAYI ile doldurulur.
    - Sinyal kapanışta, işlem bir sonraki işlem günü açılışında uygulanır.
    """
    cost = transaction_cost_pct / 100

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    holdings = {}
    trades = []
    rebalances = []
    equity_curve = []

    cash_days = 0
    invested_days = 0
    rotation_count = 0
    entry_count = 0
    exit_count = 0

    for i in range(len(common_dates) - 1):
        signal_date = common_dates[i]
        next_date = common_dates[i + 1]

        # CURRENT OPEN -> NEXT OPEN MARK TO MARKET
        if holdings:
            # Gerçekte portföy o anda kaç slot doluysa sermaye o pozisyonlara
            # eşit dağılmış kabul edilir. Top-2'de tek pozisyon varsa %50 değil,
            # mevcut portföy sermayesinin tamamı o pozisyonda kabul edilmez;
            # strateji slot mantığını korumak için her slot 1/top_n ağırlıktadır.
            weight = 1.0 / top_n
            daily_return = 0.0

            for symbol in holdings:
                df = prepared[symbol]
                current_open = float(df.loc[signal_date, "Open"])
                next_open = float(df.loc[next_date, "Open"])

                if current_open > 0:
                    daily_return += (next_open / current_open - 1) * weight

            equity *= 1 + daily_return

        # CLOSE SIGNAL: önce sadece gerçek exit koşullarını kontrol et
        old_symbols = list(holdings.keys())
        exiting = []

        for symbol in old_symbols:
            row = prepared[symbol].loc[signal_date]
            if rotation_exit_required(row, exit_score):
                exiting.append(symbol)

        survivors = [s for s in old_symbols if s not in exiting]

        # Yeni aday sıralaması yalnızca BOŞ SLOT doldurmak için kullanılır
        candidates = rotation_candidates(
            prepared,
            signal_date,
            entry_score,
        )

        candidate_symbols = [x["symbol"] for x in candidates]
        entering = []
        next_symbols = list(survivors)

        free_slots = max(0, top_n - len(next_symbols))

        if free_slots > 0:
            for symbol in candidate_symbols:
                if symbol in next_symbols:
                    continue
                entering.append(symbol)
                next_symbols.append(symbol)
                if len(entering) >= free_slots:
                    break

        # COSTS: çıkan ve giren her taraf için bir işlem maliyeti
        sides = len(exiting) + len(entering)
        if sides:
            equity *= 1 - cost * sides / top_n

        # CLOSE TRADES
        for symbol in exiting:
            old = holdings[symbol]
            entry_date = old["entry_date"]
            entry_open = float(prepared[symbol].loc[entry_date, "Open"])
            exit_open = float(prepared[symbol].loc[next_date, "Open"])

            gross = exit_open / entry_open - 1
            net = (1 + gross) * (1 - cost) * (1 - cost) - 1

            trades.append({
                "symbol": symbol,
                "entry_date": str(entry_date.date()),
                "exit_date": str(next_date.date()),
                "entry_score": old["entry_score"],
                "exit_score": safe_int(prepared[symbol].loc[signal_date, "SCORE"]),
                "return_pct": safe_float(net * 100, 2),
                "holding_days": (next_date - entry_date).days,
                "exit_reason": "V1.2A_SLOW_EXIT",
            })
            exit_count += 1

        # BUILD NEXT HOLDINGS
        new_holdings = {}

        for symbol in survivors:
            new_holdings[symbol] = holdings[symbol]

        for symbol in entering:
            score = int(prepared[symbol].loc[signal_date, "SCORE"])
            new_holdings[symbol] = {
                "entry_date": next_date,
                "entry_score": score,
            }
            entry_count += 1

        # Rotation = aynı rebalance gününde en az bir çıkış ve en az bir giriş
        if exiting and entering:
            rotation_count += 1

        if sides:
            rebalances.append({
                "signal_date": str(signal_date.date()),
                "execution_date": str(next_date.date()),
                "from": old_symbols if old_symbols else ["CASH"],
                "to": next_symbols if next_symbols else ["CASH"],
                "exiting": exiting,
                "entering": entering,
                "survivors": survivors,
                "transaction_sides": sides,
            })

        holdings = new_holdings

        if holdings:
            invested_days += 1
        else:
            cash_days += 1

        peak = max(peak, equity)
        drawdown = (equity / peak - 1) * 100
        max_drawdown = min(max_drawdown, drawdown)

        equity_curve.append({
            "date": str(next_date.date()),
            "equity": safe_float(equity, 6),
            "drawdown_pct": safe_float(drawdown, 2),
            "holdings": list(holdings.keys()) if holdings else ["CASH"],
        })

    # FINAL CLOSE
    final_date = common_dates[-1]

    if holdings:
        weight = 1.0 / top_n
        final_day_return = 0.0

        for symbol in holdings:
            df = prepared[symbol]
            final_open = float(df.loc[final_date, "Open"])
            final_close = float(df.loc[final_date, "Close"])
            if final_open > 0:
                final_day_return += (final_close / final_open - 1) * weight

        equity *= 1 + final_day_return
        equity *= 1 - cost * len(holdings) / top_n

        for symbol, old in holdings.items():
            entry_date = old["entry_date"]
            entry_open = float(prepared[symbol].loc[entry_date, "Open"])
            final_close = float(prepared[symbol].loc[final_date, "Close"])

            gross = final_close / entry_open - 1
            net = (1 + gross) * (1 - cost) * (1 - cost) - 1

            trades.append({
                "symbol": symbol,
                "entry_date": str(entry_date.date()),
                "exit_date": str(final_date.date()),
                "entry_score": old["entry_score"],
                "exit_score": safe_int(prepared[symbol].loc[final_date, "SCORE"]),
                "return_pct": safe_float(net * 100, 2),
                "holding_days": (final_date - entry_date).days,
                "forced_exit_at_test_end": True,
            })
            exit_count += 1

        peak = max(peak, equity)
        drawdown = (equity / peak - 1) * 100
        max_drawdown = min(max_drawdown, drawdown)

    returns = [x["return_pct"] for x in trades if x["return_pct"] is not None]
    winners = [x for x in returns if x > 0]
    losers = [x for x in returns if x <= 0]

    if returns:
        win_rate = len(winners) / len(returns) * 100
        avg_trade = np.mean(returns)
        median_trade = np.median(returns)
        avg_holding = np.mean([x["holding_days"] for x in trades])
    else:
        win_rate = 0
        avg_trade = 0
        median_trade = 0
        avg_holding = 0

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0

    start_date = common_dates[0]
    end_date = common_dates[-1]
    elapsed_years = (end_date - start_date).days / 365.25

    total_return = (equity - 1) * 100
    if elapsed_years > 0 and equity > 0:
        cagr = (equity ** (1 / elapsed_years) - 1) * 100
    else:
        cagr = None

    active_days = cash_days + invested_days
    cash_pct = cash_days / active_days * 100 if active_days else 0
    invested_pct = invested_days / active_days * 100 if active_days else 0

    return {
        "portfolio": f"TOP_{top_n}",
        "slots": top_n,
        "weighting": "EQUAL_WEIGHT",
        "rotation_model": "V1.2C_STICKY_ROTATION",
        "performance": {
            "strategy_total_return_pct": safe_float(total_return, 2),
            "strategy_cagr_pct": safe_float(cagr, 2),
            "max_drawdown_daily_pct": safe_float(max_drawdown, 2),
            "trade_count": len(trades),
            "win_rate_pct": safe_float(win_rate, 2),
            "profit_factor": safe_float(profit_factor, 3),
            "average_trade_pct": safe_float(avg_trade, 2),
            "median_trade_pct": safe_float(median_trade, 2),
            "average_holding_days": safe_float(avg_holding, 1),
            "entry_count": entry_count,
            "exit_count": exit_count,
            "rotation_count": rotation_count,
            "cash_days": cash_days,
            "cash_time_pct": safe_float(cash_pct, 2),
            "invested_time_pct": safe_float(invested_pct, 2),
            "final_equity": safe_float(equity, 6),
        },
        "trades": trades,
        "rebalances": rebalances,
        "equity_curve": equity_curve,
    }


def run_rotation_backtest(
    years=10,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
):
    prepared, common_dates, errors = (
        prepare_rotation_data(years)
    )

    top1 = run_rotation_variant(
        prepared=prepared,
        common_dates=common_dates,
        top_n=1,
        entry_score=entry_score,
        exit_score=exit_score,
        transaction_cost_pct=
            transaction_cost_pct,
    )

    top2 = run_rotation_variant(
        prepared=prepared,
        common_dates=common_dates,
        top_n=2,
        entry_score=entry_score,
        exit_score=exit_score,
        transaction_cost_pct=
            transaction_cost_pct,
    )

    dbc_return = None

    if "DBC" in prepared:
        dbc = prepared["DBC"]

        first_date = common_dates[0]
        last_date = common_dates[-1]

        first_open = float(
            dbc.loc[
                first_date,
                "Open",
            ]
        )

        last_close = float(
            dbc.loc[
                last_date,
                "Close",
            ]
        )

        if first_open > 0:
            dbc_return = (
                last_close
                /
                first_open
                - 1
            ) * 100

    return {
        "model": MODEL_NAME,

        "test":
            "PORTFOLIO_ROTATION_BACKTEST",

        "generated_at_utc":
            utc_now(),

        "test_period": {
            "start":
                str(common_dates[0].date()),

            "end":
                str(common_dates[-1].date()),

            "years_requested":
                years,

            "common_trading_days":
                len(common_dates),
        },

        "settings": {
            "universe":
                list(ASSETS.keys()),

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

            "transaction_cost_pct_each_side":
                transaction_cost_pct,

            "ranking":
                "SCORE, then MOM120, MOM60, MOM20",

            "eligibility":
                "AL_ADAYI and score >= entry_score",

            "portfolio_variants": [
                "TOP_1",
                "TOP_2_EQUAL_WEIGHT",
            ],

            "cash_rule":
                "Eligible candidate yoksa CASH",

            "execution":
                "Signal at close, rebalance next trading day open",

            "exit_model":
                "V1.2A_SLOW_EXIT",

            "rotation_model":
                "V1.2C_STICKY_ROTATION",

            "holding_rule":
                "Mevcut pozisyon sadece slow-exit ile kapanir; ranking sadece bos slot doldurur",

            "long_only":
                True,

            "short":
                False,

            "leverage":
                False,

            "automatic_orders":
                False,
        },

        "benchmark_reference": {
            "symbol":
                "DBC",

            "buy_hold_return_pct":
                safe_float(
                    dbc_return,
                    2,
                ),
        },

        "top_1":
            top1,

        "top_2":
            top2,

        "data_errors":
            errors,
    }


# ============================================================
# API
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "project": "Commodity Rotation Bot",
        "model": MODEL_NAME,
        "mode": "LONG_ONLY_MANUAL_EXECUTION",
        "asset_count": len(ASSETS),
        "tracked_assets": list(ASSETS.keys()),

        "endpoints": {
            "health":
                "/health",

            "assets":
                "/assets",

            "scan":
                "/scan",

            "rotation_backtest":
                "/rotation-backtest?years=10",

            "backtest_all":
                "/backtest-all?years=10",

            "analyze_gold":
                "/analyze/GLD",

            "analyze_copper":
                "/analyze/CPER",

            "backtest_gold":
                "/backtest/GLD?years=10",

            "backtest_copper":
                "/backtest/CPER?years=10",

            "dashboard":
                "/dashboard",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "asset_count": len(ASSETS),
        "time_utc": utc_now(),
    }


@app.get("/assets")
def assets():
    return {
        "model": MODEL_NAME,
        "count": len(ASSETS),
        "assets": ASSETS,
    }


@app.get("/analyze/{symbol}")
def analyze(symbol: str):
    try:
        return analyze_symbol(symbol)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/scan")
def scan():
    return scan_all_assets()


# IMPORTANT:
# STATIC ROUTES MUST BE ABOVE /backtest/{symbol}

@app.get("/rotation-backtest")
def rotation_backtest(
    years: int = Query(
        10,
        ge=2,
        le=20,
    ),
    entry_score: int = Query(
        75,
        ge=50,
        le=100,
    ),
    exit_score: int = Query(
        50,
        ge=0,
        le=80,
    ),
    transaction_cost_pct: float = Query(
        0.10,
        ge=0,
        le=2,
    ),
):
    try:
        return run_rotation_backtest(
            years=years,
            entry_score=entry_score,
            exit_score=exit_score,
            transaction_cost_pct=
                transaction_cost_pct,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/backtest-all")
def backtest_all(
    years: int = Query(
        10,
        ge=2,
        le=20,
    ),
    entry_score: int = Query(
        75,
        ge=50,
        le=100,
    ),
    exit_score: int = Query(
        50,
        ge=0,
        le=80,
    ),
    transaction_cost_pct: float = Query(
        0.10,
        ge=0,
        le=2,
    ),
):
    try:
        return run_backtest_all(
            years=years,
            entry_score=entry_score,
            exit_score=exit_score,
            transaction_cost_pct=
                transaction_cost_pct,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/backtest/{symbol}")
def backtest(
    symbol: str,
    years: int = Query(
        10,
        ge=2,
        le=20,
    ),
    entry_score: int = Query(
        75,
        ge=50,
        le=100,
    ),
    exit_score: int = Query(
        50,
        ge=0,
        le=80,
    ),
    transaction_cost_pct: float = Query(
        0.10,
        ge=0,
        le=2,
    ),
):
    try:
        return run_backtest(
            symbol,
            years,
            entry_score,
            exit_score,
            transaction_cost_pct,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/dashboard",
    response_class=HTMLResponse,
)
def dashboard():

    html = """
<!DOCTYPE html>
<html>

<head>
<meta charset="UTF-8">
<title>Commodity Rotation Bot</title>

<style>
body {
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 20px;
    background: #f4f4f4;
}

.card {
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 10px;
}

a {
    display: block;
    margin: 8px 0;
    font-size: 17px;
}
</style>

</head>

<body>

<h1>Commodity Rotation Bot v1.2C</h1>

<p>
Long-only commodity ETF/ETP scanner
</p>

<p>
V1.2C - Sticky Portfolio Rotation
</p>


<div class="card">

<h2>PORTFOLIO ROTATION</h2>

<a href="/rotation-backtest?years=10">
Top-1 + Top-2 Rotation - 10 Yıl
</a>

<a href="/rotation-backtest?years=5">
Top-1 + Top-2 Rotation - 5 Yıl
</a>

<a href="/rotation-backtest?years=3">
Top-1 + Top-2 Rotation - 3 Yıl
</a>

</div>


<div class="card">

<h2>TARAMA</h2>

<a href="/scan">
Tüm Emtiaları Tara
</a>

</div>


<div class="card">

<h2>BACKTEST ALL</h2>

<a href="/backtest-all?years=10">
10 Emtia - 10 Yıllık Toplu Backtest
</a>

</div>


<div class="card">

<h2>METALLER</h2>

<a href="/analyze/GLD">
Altın - GLD
</a>

<a href="/analyze/SLV">
Gümüş - SLV
</a>

<a href="/analyze/CPER">
Bakır - CPER
</a>

</div>


<div class="card">

<h2>ENERJİ</h2>

<a href="/analyze/USO">
Petrol - USO
</a>

<a href="/analyze/UNG">
Doğal Gaz - UNG
</a>

</div>


<div class="card">

<h2>TARIM</h2>

<a href="/analyze/DBA">
Tarım Sepeti - DBA
</a>

<a href="/analyze/WEAT">
Buğday - WEAT
</a>

<a href="/analyze/CORN">
Mısır - CORN
</a>

<a href="/analyze/SOYB">
Soya - SOYB
</a>

</div>


<div class="card">

<h2>GENEL</h2>

<a href="/analyze/DBC">
Geniş Emtia Sepeti - DBC
</a>

</div>


<div class="card">

<h2>TEKLİ BACKTEST</h2>

<a href="/backtest/GLD?years=10">
Altın 10 yıl
</a>

<a href="/backtest/SLV?years=10">
Gümüş 10 yıl
</a>

<a href="/backtest/CPER?years=10">
Bakır 10 yıl
</a>

<a href="/backtest/USO?years=10">
Petrol 10 yıl
</a>

<a href="/backtest/UNG?years=10">
Doğal Gaz 10 yıl
</a>

<a href="/backtest/DBA?years=10">
Tarım Sepeti 10 yıl
</a>

<a href="/backtest/WEAT?years=10">
Buğday 10 yıl
</a>

<a href="/backtest/CORN?years=10">
Mısır 10 yıl
</a>

<a href="/backtest/SOYB?years=10">
Soya 10 yıl
</a>

<a href="/backtest/DBC?years=10">
Geniş Emtia Sepeti 10 yıl
</a>

</div>

</body>
</html>
"""

    return HTMLResponse(
        content=html
    )
