from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone


# ============================================================
# COMMODITY ROTATION BOT
# Version: 1.2A
#
# LONG ONLY
# NO SHORT
# NO LEVERAGE
# NO AUTOMATIC ORDERS
#
# v1.2A:
# Entry sistemi v1.1 ile aynı.
# Score sistemi v1.1 ile aynı.
# Sadece backtest EXIT mekanizması yavaşlatıldı.
#
# EXIT:
# score <= exit_score
# OR EMA20 < EMA50
# OR Close < EMA100
# OR EMA50 < EMA200
# ============================================================


app = FastAPI(
    title="Commodity Rotation Bot",
    description="Long-only commodity ETF/ETP rotation system",
    version="1.2A",
)

MODEL_NAME = "COMMODITY-ROTATION-V1.2A"


# ============================================================
# ASSETS
# ============================================================

ASSETS = {

    "GLD": {
        "name": "Gold",
        "tr_name": "Altın",
        "group": "METALS",
    },

    "SLV": {
        "name": "Silver",
        "tr_name": "Gümüş",
        "group": "METALS",
    },

    "CPER": {
        "name": "Copper",
        "tr_name": "Bakır",
        "group": "METALS",
    },

    "USO": {
        "name": "Crude Oil",
        "tr_name": "Petrol",
        "group": "ENERGY",
    },

    "UNG": {
        "name": "Natural Gas",
        "tr_name": "Doğal Gaz",
        "group": "ENERGY",
    },

    "DBA": {
        "name": "Agriculture Basket",
        "tr_name": "Tarım Sepeti",
        "group": "AGRICULTURE",
    },

    "WEAT": {
        "name": "Wheat",
        "tr_name": "Buğday",
        "group": "AGRICULTURE",
    },

    "CORN": {
        "name": "Corn",
        "tr_name": "Mısır",
        "group": "AGRICULTURE",
    },

    "SOYB": {
        "name": "Soybeans",
        "tr_name": "Soya",
        "group": "AGRICULTURE",
    },

    "DBC": {
        "name": "Broad Commodities",
        "tr_name": "Geniş Emtia Sepeti",
        "group": "BROAD_COMMODITY",
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

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

        raise ValueError(
            f"{symbol} için veri alınamadı."
        )

    if isinstance(df.columns, pd.MultiIndex):

        try:

            df = df.xs(
                symbol,
                axis=1,
                level=-1,
                drop_level=True,
            )

        except Exception:

            df.columns = [
                x[0] if isinstance(x, tuple) else x
                for x in df.columns
            ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"{symbol}: {column} bulunamadı."
            )

    df = df[required].copy()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    if len(df) < 220:

        raise ValueError(
            f"{symbol}: yeterli tarihsel veri yok."
        )

    return df


# ============================================================
# RSI
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

    rs = (
        avg_gain
        /
        avg_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi.fillna(100)


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    previous_close = (
        df["Close"].shift(1)
    )

    tr1 = (
        df["High"]
        -
        df["Low"]
    ).abs()

    tr2 = (
        df["High"]
        -
        previous_close
    ).abs()

    tr3 = (
        df["Low"]
        -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]

    # TREND

    df["EMA20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    df["EMA100"] = close.ewm(
        span=100,
        adjust=False,
    ).mean()

    df["EMA200"] = close.ewm(
        span=200,
        adjust=False,
    ).mean()

    # MOMENTUM

    df["MOM20"] = (
        close.pct_change(20) * 100
    )

    df["MOM60"] = (
        close.pct_change(60) * 100
    )

    df["MOM120"] = (
        close.pct_change(120) * 100
    )

    # RSI

    df["RSI14"] = calculate_rsi(
        close,
        14,
    )

    # MACD

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"]
        -
        df["MACD_SIGNAL"]
    )

    # ATR

    df["ATR14"] = calculate_atr(
        df,
        14,
    )

    df["ATR_PCT"] = (
        df["ATR14"]
        /
        close
        *
        100
    )

    # VOLUME

    df["VOL20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["VOLUME_RATIO"] = (
        df["Volume"]
        /
        df["VOL20"]
    )

    # DISTANCE TO EMA

    df["DIST_EMA20"] = (
        (
            close
            /
            df["EMA20"]
        )
        - 1
    ) * 100

    df["DIST_EMA50"] = (
        (
            close
            /
            df["EMA50"]
        )
        - 1
    ) * 100

    df["DIST_EMA200"] = (
        (
            close
            /
            df["EMA200"]
        )
        - 1
    ) * 100

    # 1 YEAR DRAWDOWN

    high252 = (
        close
        .rolling(252)
        .max()
    )

    df["DRAWDOWN_252"] = (
        (
            close
            /
            high252
        )
        - 1
    ) * 100

    return df


# ============================================================
# SCORE ENGINE
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

    # ========================================================
    # TREND - MAX 40
    # ========================================================

    if close > ema200:

        score += 15

        reasons.append(
            "Fiyat EMA200 üzerinde"
        )

    else:

        warnings.append(
            "Fiyat EMA200 altında"
        )

    if ema50 > ema200:

        score += 10

        reasons.append(
            "EMA50 > EMA200"
        )

    else:

        warnings.append(
            "EMA50 EMA200 üzerinde değil"
        )

    if ema20 > ema50:

        score += 10

        reasons.append(
            "EMA20 > EMA50"
        )

    else:

        warnings.append(
            "Kısa vadeli trend zayıf"
        )

    if close > ema20:

        score += 5

        reasons.append(
            "Fiyat EMA20 üzerinde"
        )

    # ========================================================
    # MOMENTUM - MAX 30
    # ========================================================

    if mom20 > 0:

        score += 5

        reasons.append(
            "20 günlük momentum pozitif"
        )

    if mom60 > 0:

        score += 10

        reasons.append(
            "60 günlük momentum pozitif"
        )

    else:

        warnings.append(
            "60 günlük momentum negatif"
        )

    if mom120 > 0:

        score += 10

        reasons.append(
            "120 günlük momentum pozitif"
        )

    else:

        warnings.append(
            "120 günlük momentum negatif"
        )

    if (
        mom20 > 0
        and mom60 > 0
        and mom120 > 0
    ):

        score += 5

        reasons.append(
            "Momentum zaman dilimleri uyumlu"
        )

    # ========================================================
    # MACD - MAX 10
    # ========================================================

    if macd > macd_signal:

        score += 5

        reasons.append(
            "MACD signal üzerinde"
        )

    else:

        warnings.append(
            "MACD kısa vadeli momentum zayıflıyor"
        )

    if macd_hist > 0:

        score += 5

        reasons.append(
            "MACD histogram pozitif"
        )

    # ========================================================
    # RSI - MAX 10
    # ========================================================

    if 45 <= rsi <= 65:

        score += 10

        reasons.append(
            "RSI sağlıklı momentum bölgesinde"
        )

    elif 40 <= rsi < 45:

        score += 5

    elif 65 < rsi <= 70:

        score += 5

        warnings.append(
            "RSI yükselmiş durumda"
        )

    elif rsi > 70:

        warnings.append(
            "RSI aşırı alım bölgesinde"
        )

    else:

        warnings.append(
            "RSI zayıf"
        )

    # ========================================================
    # VOLATILITY - MAX 10
    # ========================================================

    if atr_pct <= 3:

        score += 10

        reasons.append(
            "Volatilite kontrollü"
        )

    elif atr_pct <= 5:

        score += 5

        reasons.append(
            "Volatilite orta düzeyde"
        )

    else:

        warnings.append(
            "Volatilite yüksek"
        )

    score = max(
        0,
        min(
            int(score),
            100,
        ),
    )

    return (
        score,
        reasons,
        warnings,
    )


# ============================================================
# SIGNAL
# ============================================================

def determine_signal(
    row,
    score,
):

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

    medium_term_trend = (
        ema20 > ema50
    )

    momentum_ok = (
        mom60 > 0
        and mom120 > 0
    )

    rsi_ok = (
        40 <= rsi <= 70
    )

    if (
        score >= 75
        and long_term_trend
        and medium_term_trend
        and momentum_ok
        and rsi_ok
    ):

        return "AL_ADAYI"

    if (
        score >= 60
        and long_term_trend
    ):

        return "IZLE"

    return "BEKLE"


# ============================================================
# ANALYZE SINGLE ASSET
# ============================================================

def analyze_symbol(
    symbol,
    period="10y",
):

    symbol = symbol.upper()

    if symbol not in ASSETS:

        raise ValueError(
            f"{symbol} ürün listesinde yok."
        )

    df = download_history(
        symbol,
        period,
    )

    df = calculate_indicators(
        df
    )

    clean = df.dropna().copy()

    if clean.empty:

        raise ValueError(
            f"{symbol}: gösterge üretilemedi."
        )

    row = clean.iloc[-1]

    score, reasons, warnings = (
        calculate_score(row)
    )

    signal = determine_signal(
        row,
        score,
    )

    asset = ASSETS[symbol]

    return {

        "model": MODEL_NAME,

        "symbol": symbol,

        "name": asset["name"],

        "tr_name": asset["tr_name"],

        "group": asset["group"],

        "date": str(
            clean.index[-1].date()
        ),

        "close": safe_float(
            row["Close"],
            2,
        ),

        "signal": signal,

        "score": score,

        "trend": {

            "ema20": safe_float(
                row["EMA20"],
                2,
            ),

            "ema50": safe_float(
                row["EMA50"],
                2,
            ),

            "ema100": safe_float(
                row["EMA100"],
                2,
            ),

            "ema200": safe_float(
                row["EMA200"],
                2,
            ),

            "distance_ema20_pct":
                safe_float(
                    row["DIST_EMA20"],
                    2,
                ),

            "distance_ema50_pct":
                safe_float(
                    row["DIST_EMA50"],
                    2,
                ),

            "distance_ema200_pct":
                safe_float(
                    row["DIST_EMA200"],
                    2,
                ),
        },

        "momentum": {

            "20d_pct": safe_float(
                row["MOM20"],
                2,
            ),

            "60d_pct": safe_float(
                row["MOM60"],
                2,
            ),

            "120d_pct": safe_float(
                row["MOM120"],
                2,
            ),
        },

        "rsi14": safe_float(
            row["RSI14"],
            2,
        ),

        "macd": {

            "line": safe_float(
                row["MACD"],
                4,
            ),

            "signal": safe_float(
                row["MACD_SIGNAL"],
                4,
            ),

            "histogram": safe_float(
                row["MACD_HIST"],
                4,
            ),
        },

        "risk": {

            "atr14": safe_float(
                row["ATR14"],
                4,
            ),

            "atr_pct": safe_float(
                row["ATR_PCT"],
                2,
            ),

            "drawdown_1y_pct":
                safe_float(
                    row["DRAWDOWN_252"],
                    2,
                ),
        },

        "volume": {

            "current": safe_int(
                row["Volume"]
            ),

            "average_20d": safe_int(
                row["VOL20"]
            ),

            "ratio": safe_float(
                row["VOLUME_RATIO"],
                2,
            ),
        },

        "reasons": reasons,

        "warnings": warnings,

        "generated_at_utc":
            utc_now(),
    }


# ============================================================
# SCANNER
# ============================================================

def scan_all_assets():

    results = []

    errors = []

    for symbol in ASSETS:

        try:

            result = analyze_symbol(
                symbol,
                "10y",
            )

            results.append(
                result
            )

        except Exception as exc:

            errors.append(
                {
                    "symbol": symbol,
                    "error": str(exc),
                }
            )

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        item["overall_rank"] = index

    metals = [
        x for x in results
        if x["group"] == "METALS"
    ]

    energy = [
        x for x in results
        if x["group"] == "ENERGY"
    ]

    agriculture = [
        x for x in results
        if x["group"] == "AGRICULTURE"
    ]

    broad = [
        x for x in results
        if x["group"] == "BROAD_COMMODITY"
    ]

    buy_candidates = [
        x for x in results
        if x["signal"] == "AL_ADAYI"
    ]

    watch_candidates = [
        x for x in results
        if x["signal"] == "IZLE"
    ]

    wait_candidates = [
        x for x in results
        if x["signal"] == "BEKLE"
    ]

    return {

        "model": MODEL_NAME,

        "generated_at_utc":
            utc_now(),

        "asset_count":
            len(results),

        "buy_candidate_count":
            len(buy_candidates),

        "watch_candidate_count":
            len(watch_candidates),

        "wait_candidate_count":
            len(wait_candidates),

        "ranking":
            results,

        "groups": {

            "metals":
                metals,

            "energy":
                energy,

            "agriculture":
                agriculture,

            "broad_commodity_reference":
                broad,
        },

        "errors":
            errors,
    }


# ============================================================
# BACKTEST
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

        raise ValueError(
            f"{symbol} ürün listesinde yok."
        )

    period = f"{years}y"

    df = download_history(
        symbol,
        period,
    )

    df = calculate_indicators(
        df
    )

    df = df.dropna().copy()

    if len(df) < 250:

        raise ValueError(
            "Backtest için yeterli veri yok."
        )

    scores = []

    signals = []

    for _, row in df.iterrows():

        score, _, _ = (
            calculate_score(row)
        )

        signal = (
            determine_signal(
                row,
                score,
            )
        )

        scores.append(score)

        signals.append(signal)

    df["SCORE"] = scores

    df["MODEL_SIGNAL"] = signals

    cost = (
        transaction_cost_pct
        /
        100
    )

    in_position = False

    entry_price = None

    entry_date = None

    entry_score_value = None

    trades = []

    equity = 1.0

    for i in range(
        len(df) - 1
    ):

        current = df.iloc[i]

        next_row = df.iloc[i + 1]

        next_date = df.index[i + 1]

        score = int(
            current["SCORE"]
        )

        signal = current[
            "MODEL_SIGNAL"
        ]

        next_open = float(
            next_row["Open"]
        )

        # ====================================================
        # ENTRY
        # ====================================================

        if not in_position:

            if (
                score >= entry_score
                and
                signal == "AL_ADAYI"
            ):

                entry_price = (
                    next_open
                    *
                    (1 + cost)
                )

                entry_date = next_date

                entry_score_value = score

                in_position = True

        # ====================================================
        # EXIT - V1.2A
        # ====================================================

        else:

            close = float(
                current["Close"]
            )

            ema20 = float(
                current["EMA20"]
            )

            ema50 = float(
                current["EMA50"]
            )

            ema100 = float(
                current["EMA100"]
            )

            ema200 = float(
                current["EMA200"]
            )

            exit_condition = (

                score <= exit_score

                or

                ema20 < ema50

                or

                close < ema100

                or

                ema50 < ema200
            )

            if exit_condition:

                exit_price = (
                    next_open
                    *
                    (1 - cost)
                )

                trade_return = (
                    (
                        exit_price
                        /
                        entry_price
                    )
                    - 1
                )

                equity *= (
                    1 + trade_return
                )

                holding_days = (
                    next_date
                    -
                    entry_date
                ).days

                trades.append(
                    {

                        "entry_date":
                            str(
                                entry_date.date()
                            ),

                        "exit_date":
                            str(
                                next_date.date()
                            ),

                        "entry_price":
                            safe_float(
                                entry_price,
                                2,
                            ),

                        "exit_price":
                            safe_float(
                                exit_price,
                                2,
                            ),

                        "entry_score":
                            entry_score_value,

                        "exit_score":
                            score,

                        "return_pct":
                            safe_float(
                                trade_return
                                *
                                100,
                                2,
                            ),

                        "holding_days":
                            holding_days,
                    }
                )

                in_position = False

                entry_price = None

                entry_date = None

                entry_score_value = None

    # ========================================================
    # CLOSE FINAL OPEN TRADE
    # ========================================================

    if in_position:

        final = df.iloc[-1]

        final_date = df.index[-1]

        exit_price = (
            float(final["Close"])
            *
            (1 - cost)
        )

        trade_return = (
            (
                exit_price
                /
                entry_price
            )
            - 1
        )

        equity *= (
            1 + trade_return
        )

        trades.append(
            {

                "entry_date":
                    str(
                        entry_date.date()
                    ),

                "exit_date":
                    str(
                        final_date.date()
                    ),

                "entry_price":
                    safe_float(
                        entry_price,
                        2,
                    ),

                "exit_price":
                    safe_float(
                        exit_price,
                        2,
                    ),

                "entry_score":
                    entry_score_value,

                "exit_score":
                    safe_int(
                        final["SCORE"]
                    ),

                "return_pct":
                    safe_float(
                        trade_return * 100,
                        2,
                    ),

                "holding_days":
                    (
                        final_date
                        -
                        entry_date
                    ).days,

                "forced_exit_at_test_end":
                    True,
            }
        )

    returns = [

        x["return_pct"]

        for x in trades

        if x["return_pct"] is not None
    ]

    winners = [
        x for x in returns
        if x > 0
    ]

    losers = [
        x for x in returns
        if x <= 0
    ]

    trade_count = len(
        trades
    )

    if trade_count:

        win_rate = (
            len(winners)
            /
            trade_count
            *
            100
        )

        avg_trade = np.mean(
            returns
        )

        median_trade = np.median(
            returns
        )

        avg_holding = np.mean(
            [
                x["holding_days"]
                for x in trades
            ]
        )

    else:

        win_rate = 0

        avg_trade = 0

        median_trade = 0

        avg_holding = 0

    gross_profit = sum(
        winners
    )

    gross_loss = abs(
        sum(losers)
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        profit_factor = None

    else:

        profit_factor = 0

    total_return = (
        equity - 1
    ) * 100

    first_open = float(
        df.iloc[0]["Open"]
    )

    last_close = float(
        df.iloc[-1]["Close"]
    )

    buy_hold_return = (
        (
            last_close
            /
            first_open
        )
        - 1
    ) * 100

    # ========================================================
    # TRADE-LEVEL DRAWDOWN
    # ========================================================

    running_equity = 1.0

    peak = 1.0

    max_drawdown = 0

    for trade in trades:

        running_equity *= (
            1
            +
            trade["return_pct"]
            /
            100
        )

        peak = max(
            peak,
            running_equity,
        )

        drawdown = (
            (
                running_equity
                /
                peak
            )
            - 1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    start_date = df.index[0]

    end_date = df.index[-1]

    elapsed_years = (
        (
            end_date
            -
            start_date
        ).days
        /
        365.25
    )

    if (
        elapsed_years > 0
        and equity > 0
    ):

        cagr = (
            equity
            **
            (
                1
                /
                elapsed_years
            )
            - 1
        ) * 100

    else:

        cagr = None

    return {

        "model":
            MODEL_NAME,

        "symbol":
            symbol,

        "commodity":
            ASSETS[symbol][
                "tr_name"
            ],

        "group":
            ASSETS[symbol][
                "group"
            ],

        "test_period": {

            "start":
                str(
                    start_date.date()
                ),

            "end":
                str(
                    end_date.date()
                ),

            "years_requested":
                years,
        },

        "settings": {

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

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

            "long_only":
                True,

            "short":
                False,

            "leverage":
                False,
        },

        "performance": {

            "trade_count":
                trade_count,

            "win_rate_pct":
                safe_float(
                    win_rate,
                    2,
                ),

            "average_trade_pct":
                safe_float(
                    avg_trade,
                    2,
                ),

            "median_trade_pct":
                safe_float(
                    median_trade,
                    2,
                ),

            "average_holding_days":
                safe_float(
                    avg_holding,
                    1,
                ),

            "profit_factor":
                safe_float(
                    profit_factor,
                    3,
                ),

            "strategy_total_return_pct":
                safe_float(
                    total_return,
                    2,
                ),

            "strategy_cagr_pct":
                safe_float(
                    cagr,
                    2,
                ),

            "max_drawdown_trade_level_pct":
                safe_float(
                    max_drawdown,
                    2,
                ),

            "buy_hold_return_pct":
                safe_float(
                    buy_hold_return,
                    2,
                ),
        },

        "trades":
            trades,

        "generated_at_utc":
            utc_now(),
    }


# ============================================================
# BACKTEST ALL
# ============================================================

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
                transaction_cost_pct=transaction_cost_pct,
            )

            performance = bt[
                "performance"
            ]

            results.append(
                {

                    "symbol":
                        symbol,

                    "commodity":
                        ASSETS[symbol][
                            "tr_name"
                        ],

                    "group":
                        ASSETS[symbol][
                            "group"
                        ],

                    "trade_count":
                        performance[
                            "trade_count"
                        ],

                    "win_rate_pct":
                        performance[
                            "win_rate_pct"
                        ],

                    "profit_factor":
                        performance[
                            "profit_factor"
                        ],

                    "average_trade_pct":
                        performance[
                            "average_trade_pct"
                        ],

                    "median_trade_pct":
                        performance[
                            "median_trade_pct"
                        ],

                    "average_holding_days":
                        performance[
                            "average_holding_days"
                        ],

                    "strategy_total_return_pct":
                        performance[
                            "strategy_total_return_pct"
                        ],

                    "strategy_cagr_pct":
                        performance[
                            "strategy_cagr_pct"
                        ],

                    "max_drawdown_pct":
                        performance[
                            "max_drawdown_trade_level_pct"
                        ],

                    "buy_hold_return_pct":
                        performance[
                            "buy_hold_return_pct"
                        ],
                }
            )

        except Exception as exc:

            errors.append(
                {
                    "symbol": symbol,
                    "error": str(exc),
                }
            )

    # Sadece raporlama kolaylığı için PF sırası

    def pf_sort_value(item):

        pf = item[
            "profit_factor"
        ]

        if pf is None:

            return -999999

        return pf

    results = sorted(
        results,
        key=pf_sort_value,
        reverse=True,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        item[
            "backtest_rank"
        ] = index

    profitable_count = sum(
        1
        for item in results
        if (
            item[
                "strategy_total_return_pct"
            ]
            is not None
            and
            item[
                "strategy_total_return_pct"
            ] > 0
        )
    )

    pf_above_one_count = sum(
        1
        for item in results
        if (
            item[
                "profit_factor"
            ]
            is not None
            and
            item[
                "profit_factor"
            ] > 1
        )
    )

    beat_buy_hold_count = sum(
        1
        for item in results
        if (
            item[
                "strategy_total_return_pct"
            ]
            is not None
            and
            item[
                "buy_hold_return_pct"
            ]
            is not None
            and
            item[
                "strategy_total_return_pct"
            ]
            >
            item[
                "buy_hold_return_pct"
            ]
        )
    )

    return {

        "model":
            MODEL_NAME,

        "test":
            "BACKTEST_ALL",

        "generated_at_utc":
            utc_now(),

        "settings": {

            "years":
                years,

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

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

            "long_only":
                True,

            "short":
                False,

            "leverage":
                False,
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

        "results":
            results,

        "errors":
            errors,
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "online",

        "project":
            "Commodity Rotation Bot",

        "model":
            MODEL_NAME,

        "mode":
            "LONG_ONLY_MANUAL_EXECUTION",

        "asset_count":
            len(ASSETS),

        "tracked_assets":
            list(
                ASSETS.keys()
            ),

        "endpoints": {

            "health":
                "/health",

            "assets":
                "/assets",

            "scan":
                "/scan",

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

        "status":
            "ok",

        "model":
            MODEL_NAME,

        "asset_count":
            len(ASSETS),

        "time_utc":
            utc_now(),
    }


@app.get("/assets")
def assets():

    return {

        "model":
            MODEL_NAME,

        "count":
            len(ASSETS),

        "assets":
            ASSETS,
    }


@app.get("/analyze/{symbol}")
def analyze(symbol: str):

    try:

        return analyze_symbol(
            symbol
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/scan")
def scan():

    return scan_all_assets()


# ============================================================
# IMPORTANT:
# /backtest-all MUST BE ABOVE /backtest/{symbol}
# ============================================================

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
            transaction_cost_pct=transaction_cost_pct,
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

<h1>
Commodity Rotation Bot v1.2A
</h1>

<p>
Long-only commodity ETF/ETP scanner
</p>

<p>
V1.2A - Slow Exit Backtest
</p>


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
