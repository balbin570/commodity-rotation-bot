from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone

# ============================================================
# COMMODITY ROTATION BOT v1.2D
# LONG ONLY - NO SHORT - NO LEVERAGE - NO AUTOMATIC ORDERS
#
# V1.2D:
# 1) Exact requested-year window
# 2) Frozen Top-2 Sticky chronological OOS / walk-forward
#
# V1.2C STRATEGY RULES ARE FROZEN
# ============================================================

app = FastAPI(
    title="Commodity Rotation Bot",
    description="Long-only commodity ETF/ETP rotation system",
    version="1.2D",
)

MODEL_NAME = "COMMODITY-ROTATION-V1.2D"

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
# HELPERS
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
                x[0]
                if isinstance(x, tuple)
                else x
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

    rs = (
        avg_gain
        /
        avg_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi.fillna(100)


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


def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]

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

    df["MOM20"] = (
        close.pct_change(20)
        * 100
    )

    df["MOM60"] = (
        close.pct_change(60)
        * 100
    )

    df["MOM120"] = (
        close.pct_change(120)
        * 100
    )

    df["RSI14"] = calculate_rsi(
        close,
        14,
    )

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

    df["ATR14"] = calculate_atr(
        df,
        14,
    )

    df["ATR_PCT"] = (
        df["ATR14"]
        /
        close
        * 100
    )

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

    df["DIST_EMA20"] = (
        close
        /
        df["EMA20"]
        - 1
    ) * 100

    df["DIST_EMA50"] = (
        close
        /
        df["EMA50"]
        - 1
    ) * 100

    df["DIST_EMA200"] = (
        close
        /
        df["EMA200"]
        - 1
    ) * 100

    high252 = (
        close
        .rolling(252)
        .max()
    )

    df["DRAWDOWN_252"] = (
        close
        /
        high252
        - 1
    ) * 100

    return df


# ============================================================
# SCORE
# FROZEN V1.2C
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

    # ----------------------------------------
    # TREND = 40
    # ----------------------------------------

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

    # ----------------------------------------
    # MOMENTUM = 30
    # ----------------------------------------

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
        and
        mom60 > 0
        and
        mom120 > 0
    ):

        score += 5

        reasons.append(
            "Momentum zaman dilimleri uyumlu"
        )

    # ----------------------------------------
    # MACD = 10
    # ----------------------------------------

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

    # ----------------------------------------
    # RSI = 10
    # ----------------------------------------

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

    # ----------------------------------------
    # VOLATILITY = 10
    # ----------------------------------------

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
        and
        ema50 > ema200
    )

    medium_term_trend = (
        ema20 > ema50
    )

    momentum_ok = (
        mom60 > 0
        and
        mom120 > 0
    )

    rsi_ok = (
        40 <= rsi <= 70
    )

    if (
        score >= 75
        and
        long_term_trend
        and
        medium_term_trend
        and
        momentum_ok
        and
        rsi_ok
    ):

        return "AL_ADAYI"

    if (
        score >= 60
        and
        long_term_trend
    ):

        return "IZLE"

    return "BEKLE"


# ============================================================
# SINGLE ANALYSIS
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

    clean = (
        df
        .dropna()
        .copy()
    )

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

        "model":
            MODEL_NAME,

        "symbol":
            symbol,

        "name":
            asset["name"],

        "tr_name":
            asset["tr_name"],

        "group":
            asset["group"],

        "date":
            str(
                clean.index[-1].date()
            ),

        "close":
            safe_float(
                row["Close"],
                2,
            ),

        "signal":
            signal,

        "score":
            score,

        "trend": {

            "ema20":
                safe_float(
                    row["EMA20"],
                    2,
                ),

            "ema50":
                safe_float(
                    row["EMA50"],
                    2,
                ),

            "ema100":
                safe_float(
                    row["EMA100"],
                    2,
                ),

            "ema200":
                safe_float(
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

            "20d_pct":
                safe_float(
                    row["MOM20"],
                    2,
                ),

            "60d_pct":
                safe_float(
                    row["MOM60"],
                    2,
                ),

            "120d_pct":
                safe_float(
                    row["MOM120"],
                    2,
                ),
        },

        "rsi14":
            safe_float(
                row["RSI14"],
                2,
            ),

        "macd": {

            "line":
                safe_float(
                    row["MACD"],
                    4,
                ),

            "signal":
                safe_float(
                    row["MACD_SIGNAL"],
                    4,
                ),

            "histogram":
                safe_float(
                    row["MACD_HIST"],
                    4,
                ),
        },

        "risk": {

            "atr14":
                safe_float(
                    row["ATR14"],
                    4,
                ),

            "atr_pct":
                safe_float(
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

            "current":
                safe_int(
                    row["Volume"]
                ),

            "average_20d":
                safe_int(
                    row["VOL20"]
                ),

            "ratio":
                safe_float(
                    row["VOLUME_RATIO"],
                    2,
                ),
        },

        "reasons":
            reasons,

        "warnings":
            warnings,

        "generated_at_utc":
            utc_now(),
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
                analyze_symbol(
                    symbol,
                    "10y",
                )
            )

        except Exception as exc:

            errors.append({
                "symbol":
                    symbol,

                "error":
                    str(exc),
            })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        item["overall_rank"] = (
            index
        )

    return {

        "model":
            MODEL_NAME,

        "generated_at_utc":
            utc_now(),

        "asset_count":
            len(results),

        "buy_candidate_count":
            sum(
                x["signal"]
                ==
                "AL_ADAYI"
                for x in results
            ),

        "watch_candidate_count":
            sum(
                x["signal"]
                ==
                "IZLE"
                for x in results
            ),

        "wait_candidate_count":
            sum(
                x["signal"]
                ==
                "BEKLE"
                for x in results
            ),

        "ranking":
            results,

        "errors":
            errors,
    }


# ============================================================
# V1.2A SINGLE-ASSET BACKTEST
# UNCHANGED
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

    df = download_history(
        symbol,
        f"{years}y",
    )

    df = calculate_indicators(
        df
    )

    df = (
        df
        .dropna()
        .copy()
    )

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

        scores.append(
            score
        )

        signals.append(
            determine_signal(
                row,
                score,
            )
        )

    df["SCORE"] = scores

    df["MODEL_SIGNAL"] = (
        signals
    )

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

        next_row = (
            df.iloc[i + 1]
        )

        next_date = (
            df.index[i + 1]
        )

        score = int(
            current["SCORE"]
        )

        signal = (
            current["MODEL_SIGNAL"]
        )

        next_open = float(
            next_row["Open"]
        )

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

                entry_date = (
                    next_date
                )

                entry_score_value = (
                    score
                )

                in_position = True

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
                    exit_price
                    /
                    entry_price
                    - 1
                )

                equity *= (
                    1 + trade_return
                )

                trades.append({

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
                        (
                            next_date
                            -
                            entry_date
                        ).days,
                })

                in_position = False

                entry_price = None
                entry_date = None
                entry_score_value = None

    if in_position:

        final = (
            df.iloc[-1]
        )

        final_date = (
            df.index[-1]
        )

        exit_price = (
            float(
                final["Close"]
            )
            *
            (1 - cost)
        )

        trade_return = (
            exit_price
            /
            entry_price
            - 1
        )

        equity *= (
            1 + trade_return
        )

        trades.append({

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
                    trade_return
                    *
                    100,
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
        })

    returns = [

        x["return_pct"]

        for x in trades

        if (
            x["return_pct"]
            is not None
        )
    ]

    winners = [
        x
        for x in returns
        if x > 0
    ]

    losers = [
        x
        for x in returns
        if x <= 0
    ]

    trade_count = (
        len(trades)
    )

    if trade_count:

        win_rate = (
            len(winners)
            /
            trade_count
            *
            100
        )

        avg_trade = (
            np.mean(returns)
        )

        median_trade = (
            np.median(returns)
        )

        avg_holding = (
            np.mean([
                x["holding_days"]
                for x in trades
            ])
        )

    else:

        win_rate = 0
        avg_trade = 0
        median_trade = 0
        avg_holding = 0

    gross_profit = (
        sum(winners)
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
        last_close
        /
        first_open
        - 1
    ) * 100

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
            running_equity
            /
            peak
            - 1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    start_date = (
        df.index[0]
    )

    end_date = (
        df.index[-1]
    )

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
        and
        equity > 0
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
            ASSETS[symbol]["tr_name"],

        "group":
            ASSETS[symbol]["group"],

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

                "symbol":
                    symbol,

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

                "average_holding_days":
                    p["average_holding_days"],

                "strategy_total_return_pct":
                    p["strategy_total_return_pct"],

                "strategy_cagr_pct":
                    p["strategy_cagr_pct"],

                "max_drawdown_pct":
                    p[
                        "max_drawdown_trade_level_pct"
                    ],

                "buy_hold_return_pct":
                    p["buy_hold_return_pct"],
            })

        except Exception as exc:

            errors.append({

                "symbol":
                    symbol,

                "error":
                    str(exc),
            })

    results = sorted(
        results,
        key=lambda x: (
            -999999
            if x["profit_factor"] is None
            else x["profit_factor"]
        ),
        reverse=True,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        item["backtest_rank"] = (
            index
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

            "exit_model":
                "V1.2A_SLOW_EXIT",
        },

        "results":
            results,

        "errors":
            errors,
    }


# ============================================================
# ROTATION DATA
# V1.2D FIX:
# EXTRA DATA FOR INDICATOR WARM-UP
# THEN EXACT REQUESTED YEAR WINDOW
# ============================================================

def prepare_rotation_data(
    years=10,
):

    prepared = {}
    errors = []

    # Example:
    # years=3 -> download 5 years
    # indicators mature on older data
    # actual test is then cut to exactly 3 years

    download_years = min(
        int(years) + 2,
        20,
    )

    for symbol in ASSETS:

        try:

            df = download_history(
                symbol,
                f"{download_years}y",
            )

            df = calculate_indicators(
                df
            )

            df = (
                df
                .dropna()
                .copy()
            )

            scores = []
            signals = []

            for _, row in df.iterrows():

                score, _, _ = (
                    calculate_score(row)
                )

                scores.append(
                    score
                )

                signals.append(
                    determine_signal(
                        row,
                        score,
                    )
                )

            df["SCORE"] = (
                scores
            )

            df["MODEL_SIGNAL"] = (
                signals
            )

            prepared[symbol] = (
                df
            )

        except Exception as exc:

            errors.append({

                "symbol":
                    symbol,

                "error":
                    str(exc),
            })

    if not prepared:

        raise ValueError(
            "Rotation backtest için veri hazırlanamadı."
        )

    common_dates = None

    for df in prepared.values():

        dates = set(
            df.index
        )

        if common_dates is None:

            common_dates = (
                dates
            )

        else:

            common_dates = (
                common_dates
                .intersection(
                    dates
                )
            )

    common_dates = sorted(
        common_dates
    )

    if not common_dates:

        raise ValueError(
            "Ortak işlem tarihi bulunamadı."
        )

    # ----------------------------------------
    # EXACT TEST WINDOW
    # ----------------------------------------

    end_date = pd.Timestamp(
        common_dates[-1]
    )

    start_cutoff = (
        end_date
        -
        pd.DateOffset(
            years=int(years)
        )
    )

    common_dates = [

        d

        for d in common_dates

        if (
            pd.Timestamp(d)
            >=
            start_cutoff
        )
    ]

    if len(common_dates) < 250:

        raise ValueError(
            "Rotation backtest için yeterli ortak tarih yok."
        )

    return (
        prepared,
        common_dates,
        errors,
    )


# ============================================================
# ROTATION CANDIDATES
# FROZEN
# ============================================================

def rotation_candidates(
    prepared,
    signal_date,
    entry_score,
):

    candidates = []

    for symbol, df in (
        prepared.items()
    ):

        if signal_date not in df.index:
            continue

        row = df.loc[
            signal_date
        ]

        score = int(
            row["SCORE"]
        )

        signal = (
            row["MODEL_SIGNAL"]
        )

        if (
            score >= entry_score
            and
            signal == "AL_ADAYI"
        ):

            candidates.append({

                "symbol":
                    symbol,

                "score":
                    score,

                "mom120":
                    float(
                        row["MOM120"]
                    ),

                "mom60":
                    float(
                        row["MOM60"]
                    ),

                "mom20":
                    float(
                        row["MOM20"]
                    ),
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


# ============================================================
# SLOW EXIT
# FROZEN V1.2A / V1.2C
# ============================================================

def rotation_exit_required(
    row,
    exit_score,
):

    return (

        int(
            row["SCORE"]
        )
        <=
        exit_score

        or

        float(
            row["EMA20"]
        )
        <
        float(
            row["EMA50"]
        )

        or

        float(
            row["Close"]
        )
        <
        float(
            row["EMA100"]
        )

        or

        float(
            row["EMA50"]
        )
        <
        float(
            row["EMA200"]
        )
    )


# ============================================================
# STICKY ROTATION
# FROZEN V1.2C
# ============================================================

def run_rotation_variant(
    prepared,
    common_dates,
    top_n,
    entry_score,
    exit_score,
    transaction_cost_pct,
):

    cost = (
        transaction_cost_pct
        /
        100
    )

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

    for i in range(
        len(common_dates) - 1
    ):

        signal_date = (
            common_dates[i]
        )

        next_date = (
            common_dates[i + 1]
        )

        # ------------------------------------
        # MARK TO MARKET
        # OPEN -> NEXT OPEN
        # ------------------------------------

        if holdings:

            weight = (
                1.0 / top_n
            )

            daily_return = 0.0

            for symbol in holdings:

                df = (
                    prepared[symbol]
                )

                current_open = float(
                    df.loc[
                        signal_date,
                        "Open",
                    ]
                )

                next_open = float(
                    df.loc[
                        next_date,
                        "Open",
                    ]
                )

                if current_open > 0:

                    daily_return += (
                        (
                            next_open
                            /
                            current_open
                            - 1
                        )
                        *
                        weight
                    )

            equity *= (
                1 + daily_return
            )

        # ------------------------------------
        # EXIT CHECK
        # ------------------------------------

        old_symbols = list(
            holdings.keys()
        )

        exiting = []

        for symbol in old_symbols:

            row = (
                prepared[symbol]
                .loc[signal_date]
            )

            if rotation_exit_required(
                row,
                exit_score,
            ):

                exiting.append(
                    symbol
                )

        survivors = [

            symbol

            for symbol in old_symbols

            if (
                symbol
                not in exiting
            )
        ]

        # ------------------------------------
        # RANKING ONLY FILLS EMPTY SLOTS
        # ------------------------------------

        candidates = (
            rotation_candidates(
                prepared,
                signal_date,
                entry_score,
            )
        )

        candidate_symbols = [

            x["symbol"]

            for x in candidates
        ]

        entering = []

        next_symbols = list(
            survivors
        )

        free_slots = max(
            0,
            top_n
            -
            len(next_symbols),
        )

        if free_slots > 0:

            for symbol in candidate_symbols:

                if symbol in next_symbols:
                    continue

                entering.append(
                    symbol
                )

                next_symbols.append(
                    symbol
                )

                if (
                    len(entering)
                    >=
                    free_slots
                ):
                    break

        # ------------------------------------
        # TRANSACTION COST
        # ------------------------------------

        sides = (
            len(exiting)
            +
            len(entering)
        )

        if sides:

            equity *= (
                1
                -
                cost
                *
                sides
                /
                top_n
            )

        # ------------------------------------
        # CLOSE TRADES
        # ------------------------------------

        for symbol in exiting:

            old = (
                holdings[symbol]
            )

            entry_date = (
                old["entry_date"]
            )

            entry_open = float(

                prepared[symbol]
                .loc[
                    entry_date,
                    "Open",
                ]
            )

            exit_open = float(

                prepared[symbol]
                .loc[
                    next_date,
                    "Open",
                ]
            )

            gross = (
                exit_open
                /
                entry_open
                - 1
            )

            net = (
                (1 + gross)
                *
                (1 - cost)
                *
                (1 - cost)
                - 1
            )

            trades.append({

                "symbol":
                    symbol,

                "entry_date":
                    str(
                        entry_date.date()
                    ),

                "exit_date":
                    str(
                        next_date.date()
                    ),

                "entry_score":
                    old["entry_score"],

                "exit_score":
                    safe_int(
                        prepared[symbol]
                        .loc[
                            signal_date,
                            "SCORE",
                        ]
                    ),

                "return_pct":
                    safe_float(
                        net * 100,
                        2,
                    ),

                "holding_days":
                    (
                        next_date
                        -
                        entry_date
                    ).days,

                "exit_reason":
                    "V1.2A_SLOW_EXIT",
            })

            exit_count += 1

        # ------------------------------------
        # BUILD NEXT HOLDINGS
        # ------------------------------------

        new_holdings = {}

        for symbol in survivors:

            new_holdings[symbol] = (
                holdings[symbol]
            )

        for symbol in entering:

            score = int(

                prepared[symbol]
                .loc[
                    signal_date,
                    "SCORE",
                ]
            )

            new_holdings[symbol] = {

                "entry_date":
                    next_date,

                "entry_score":
                    score,
            }

            entry_count += 1

        if (
            exiting
            and
            entering
        ):

            rotation_count += 1

        if sides:

            rebalances.append({

                "signal_date":
                    str(
                        signal_date.date()
                    ),

                "execution_date":
                    str(
                        next_date.date()
                    ),

                "from":
                    (
                        old_symbols
                        if old_symbols
                        else ["CASH"]
                    ),

                "to":
                    (
                        next_symbols
                        if next_symbols
                        else ["CASH"]
                    ),

                "exiting":
                    exiting,

                "entering":
                    entering,

                "survivors":
                    survivors,

                "transaction_sides":
                    sides,
            })

        holdings = (
            new_holdings
        )

        if holdings:

            invested_days += 1

        else:

            cash_days += 1

        peak = max(
            peak,
            equity,
        )

        drawdown = (
            equity
            /
            peak
            - 1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

        equity_curve.append({

            "date":
                str(
                    next_date.date()
                ),

            "equity":
                safe_float(
                    equity,
                    6,
                ),

            "drawdown_pct":
                safe_float(
                    drawdown,
                    2,
                ),

            "holdings":
                (
                    list(
                        holdings.keys()
                    )
                    if holdings
                    else ["CASH"]
                ),
        })

    # ----------------------------------------
    # FINAL CLOSE
    # ----------------------------------------

    final_date = (
        common_dates[-1]
    )

    if holdings:

        weight = (
            1.0 / top_n
        )

        final_day_return = 0.0

        for symbol in holdings:

            df = (
                prepared[symbol]
            )

            final_open = float(
                df.loc[
                    final_date,
                    "Open",
                ]
            )

            final_close = float(
                df.loc[
                    final_date,
                    "Close",
                ]
            )

            if final_open > 0:

                final_day_return += (
                    (
                        final_close
                        /
                        final_open
                        - 1
                    )
                    *
                    weight
                )

        equity *= (
            1 + final_day_return
        )

        equity *= (
            1
            -
            cost
            *
            len(holdings)
            /
            top_n
        )

        for symbol, old in (
            holdings.items()
        ):

            entry_date = (
                old["entry_date"]
            )

            entry_open = float(

                prepared[symbol]
                .loc[
                    entry_date,
                    "Open",
                ]
            )

            final_close = float(

                prepared[symbol]
                .loc[
                    final_date,
                    "Close",
                ]
            )

            gross = (
                final_close
                /
                entry_open
                - 1
            )

            net = (
                (1 + gross)
                *
                (1 - cost)
                *
                (1 - cost)
                - 1
            )

            trades.append({

                "symbol":
                    symbol,

                "entry_date":
                    str(
                        entry_date.date()
                    ),

                "exit_date":
                    str(
                        final_date.date()
                    ),

                "entry_score":
                    old["entry_score"],

                "exit_score":
                    safe_int(
                        prepared[symbol]
                        .loc[
                            final_date,
                            "SCORE",
                        ]
                    ),

                "return_pct":
                    safe_float(
                        net * 100,
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
            })

            exit_count += 1

        peak = max(
            peak,
            equity,
        )

        drawdown = (
            equity
            /
            peak
            - 1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    # ----------------------------------------
    # PERFORMANCE
    # ----------------------------------------

    returns = [

        x["return_pct"]

        for x in trades

        if (
            x["return_pct"]
            is not None
        )
    ]

    winners = [
        x
        for x in returns
        if x > 0
    ]

    losers = [
        x
        for x in returns
        if x <= 0
    ]

    if returns:

        win_rate = (
            len(winners)
            /
            len(returns)
            *
            100
        )

        avg_trade = (
            np.mean(returns)
        )

        median_trade = (
            np.median(returns)
        )

        avg_holding = (
            np.mean([
                x["holding_days"]
                for x in trades
            ])
        )

    else:

        win_rate = 0
        avg_trade = 0
        median_trade = 0
        avg_holding = 0

    gross_profit = (
        sum(winners)
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

    start_date = (
        common_dates[0]
    )

    end_date = (
        common_dates[-1]
    )

    elapsed_years = (
        (
            end_date
            -
            start_date
        ).days
        /
        365.25
    )

    total_return = (
        equity - 1
    ) * 100

    if (
        elapsed_years > 0
        and
        equity > 0
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

    active_days = (
        cash_days
        +
        invested_days
    )

    cash_pct = (
        cash_days
        /
        active_days
        *
        100
        if active_days
        else 0
    )

    invested_pct = (
        invested_days
        /
        active_days
        *
        100
        if active_days
        else 0
    )

    return {

        "portfolio":
            f"TOP_{top_n}",

        "slots":
            top_n,

        "weighting":
            "EQUAL_WEIGHT",

        "rotation_model":
            "V1.2C_STICKY_ROTATION",

        "performance": {

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

            "max_drawdown_daily_pct":
                safe_float(
                    max_drawdown,
                    2,
                ),

            "trade_count":
                len(trades),

            "win_rate_pct":
                safe_float(
                    win_rate,
                    2,
                ),

            "profit_factor":
                safe_float(
                    profit_factor,
                    3,
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

            "entry_count":
                entry_count,

            "exit_count":
                exit_count,

            "rotation_count":
                rotation_count,

            "cash_days":
                cash_days,

            "cash_time_pct":
                safe_float(
                    cash_pct,
                    2,
                ),

            "invested_time_pct":
                safe_float(
                    invested_pct,
                    2,
                ),

            "final_equity":
                safe_float(
                    equity,
                    6,
                ),
        },

        "trades":
            trades,

        "rebalances":
            rebalances,

        "equity_curve":
            equity_curve,
    }


# ============================================================
# NORMAL ROTATION BACKTEST
# ============================================================

def run_rotation_backtest(
    years=10,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
):

    (
        prepared,
        common_dates,
        errors,
    ) = prepare_rotation_data(
        years
    )

    top1 = run_rotation_variant(

        prepared=
            prepared,

        common_dates=
            common_dates,

        top_n=
            1,

        entry_score=
            entry_score,

        exit_score=
            exit_score,

        transaction_cost_pct=
            transaction_cost_pct,
    )

    top2 = run_rotation_variant(

        prepared=
            prepared,

        common_dates=
            common_dates,

        top_n=
            2,

        entry_score=
            entry_score,

        exit_score=
            exit_score,

        transaction_cost_pct=
            transaction_cost_pct,
    )

    dbc_return = None

    if "DBC" in prepared:

        dbc = (
            prepared["DBC"]
        )

        first_date = (
            common_dates[0]
        )

        last_date = (
            common_dates[-1]
        )

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

        "model":
            MODEL_NAME,

        "test":
            "PORTFOLIO_ROTATION_BACKTEST",

        "generated_at_utc":
            utc_now(),

        "test_period": {

            "start":
                str(
                    common_dates[0].date()
                ),

            "end":
                str(
                    common_dates[-1].date()
                ),

            "years_requested":
                years,

            "common_trading_days":
                len(common_dates),
        },

        "settings": {

            "universe":
                list(
                    ASSETS.keys()
                ),

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
# V1.2D
# CHRONOLOGICAL OOS / WALK-FORWARD
# TOP-2 STICKY ONLY
# ============================================================

def compact_rotation_result(
    result,
):

    return {

        "portfolio":
            result["portfolio"],

        "slots":
            result["slots"],

        "weighting":
            result["weighting"],

        "rotation_model":
            result["rotation_model"],

        "performance":
            result["performance"],

        "trades":
            result["trades"],

        "rebalances":
            result["rebalances"],
    }


def run_oos_walk_forward(
    years=10,
    folds=4,
    initial_train_pct=50,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
):

    (
        prepared,
        common_dates,
        errors,
    ) = prepare_rotation_data(
        years
    )

    n = len(
        common_dates
    )

    if n < 500:

        raise ValueError(
            "OOS/walk-forward için yeterli ortak tarih yok."
        )

    folds = int(
        folds
    )

    if folds < 2:

        raise ValueError(
            "folds en az 2 olmalı."
        )

    initial_train_pct = int(
        initial_train_pct
    )

    if not (
        30
        <=
        initial_train_pct
        <=
        80
    ):

        raise ValueError(
            "initial_train_pct 30 ile 80 arasında olmalı."
        )

    train_end = int(
        n
        *
        initial_train_pct
        /
        100
    )

    train_end = max(
        250,
        train_end,
    )

    minimum_oos_days = (
        folds * 40
    )

    if (
        n - train_end
        <
        minimum_oos_days
    ):

        train_end = (
            n
            -
            minimum_oos_days
        )

    if train_end < 250:

        raise ValueError(
            "Train/OOS bölünmesi için yeterli veri yok."
        )

    remaining = (
        n - train_end
    )

    if (
        remaining
        <
        minimum_oos_days
    ):

        raise ValueError(
            "Seçilen fold sayısı için OOS dönemi çok kısa."
        )

    base = (
        remaining
        //
        folds
    )

    extra = (
        remaining
        %
        folds
    )

    fold_results = []

    cursor = (
        train_end
    )

    compounded_equity = 1.0

    total_trade_count = 0
    total_rotation_count = 0

    for fold_no in range(
        1,
        folds + 1,
    ):

        fold_len = (
            base
            +
            (
                1
                if fold_no <= extra
                else 0
            )
        )

        start_idx = (
            cursor
        )

        end_idx = min(
            n,
            cursor + fold_len,
        )

        # One prior trading day is included
        # only to create the signal that executes
        # on the first actual OOS trading day.

        slice_start = max(
            0,
            start_idx - 1,
        )

        fold_dates = (
            common_dates[
                slice_start:end_idx
            ]
        )

        if len(fold_dates) < 2:
            break

        result = run_rotation_variant(

            prepared=
                prepared,

            common_dates=
                fold_dates,

            top_n=
                2,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
                transaction_cost_pct,
        )

        performance = (
            result["performance"]
        )

        fold_equity = (
            performance[
                "final_equity"
            ]
        )

        if fold_equity is None:
            fold_equity = 1.0

        fold_equity = float(
            fold_equity
        )

        compounded_equity *= (
            fold_equity
        )

        total_trade_count += int(
            performance[
                "trade_count"
            ]
            or 0
        )

        total_rotation_count += int(
            performance[
                "rotation_count"
            ]
            or 0
        )

        fold_results.append({

            "fold":
                fold_no,

            "train_period": {

                "start":
                    str(
                        common_dates[0].date()
                    ),

                "end":
                    str(
                        common_dates[
                            start_idx - 1
                        ].date()
                    ),

                "trading_days":
                    start_idx,
            },

            "oos_period": {

                "start":
                    str(
                        common_dates[
                            start_idx
                        ].date()
                    ),

                "end":
                    str(
                        common_dates[
                            end_idx - 1
                        ].date()
                    ),

                "trading_days":
                    (
                        end_idx
                        -
                        start_idx
                    ),
            },

            "result":
                compact_rotation_result(
                    result
                ),
        })

        cursor = (
            end_idx
        )

    if not fold_results:

        raise ValueError(
            "OOS fold üretilemedi."
        )

    oos_start = pd.Timestamp(
        fold_results[0][
            "oos_period"
        ]["start"]
    )

    oos_end = pd.Timestamp(
        fold_results[-1][
            "oos_period"
        ]["end"]
    )

    elapsed_years = (
        (
            oos_end
            -
            oos_start
        ).days
        /
        365.25
    )

    compounded_return = (
        compounded_equity
        - 1
    ) * 100

    if (
        elapsed_years > 0
        and
        compounded_equity > 0
    ):

        compounded_cagr = (

            compounded_equity
            **
            (
                1
                /
                elapsed_years
            )
            - 1

        ) * 100

    else:

        compounded_cagr = None

    positive_folds = sum(

        1

        for fold in fold_results

        if (
            fold[
                "result"
            ][
                "performance"
            ][
                "strategy_total_return_pct"
            ]
            is not None

            and

            fold[
                "result"
            ][
                "performance"
            ][
                "strategy_total_return_pct"
            ]
            > 0
        )
    )

    pf_above_one_folds = sum(

        1

        for fold in fold_results

        if (
            fold[
                "result"
            ][
                "performance"
            ][
                "profit_factor"
            ]
            is not None

            and

            fold[
                "result"
            ][
                "performance"
            ][
                "profit_factor"
            ]
            > 1
        )
    )

    return {

        "model":
            MODEL_NAME,

        "test":
            "TOP2_STICKY_CHRONOLOGICAL_OOS_WALK_FORWARD",

        "generated_at_utc":
            utc_now(),

        "full_period": {

            "start":
                str(
                    common_dates[0].date()
                ),

            "end":
                str(
                    common_dates[-1].date()
                ),

            "years_requested":
                years,

            "common_trading_days":
                n,
        },

        "frozen_settings": {

            "portfolio":
                "TOP_2_EQUAL_WEIGHT",

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

            "exit_model":
                "V1.2A_SLOW_EXIT",

            "rotation_model":
                "V1.2C_STICKY_ROTATION",

            "execution":
                "Signal at close, next trading day open",

            "parameter_optimization":
                False,

            "long_only":
                True,

            "short":
                False,

            "leverage":
                False,

            "automatic_orders":
                False,
        },

        "walk_forward": {

            "initial_train_pct":
                initial_train_pct,

            "fold_count":
                len(
                    fold_results
                ),

            "fold_rule":
                "Expanding chronological history; each OOS fold starts from CASH; no fitting or parameter changes",
        },

        "oos_summary": {

            "oos_start":
                str(
                    oos_start.date()
                ),

            "oos_end":
                str(
                    oos_end.date()
                ),

            "positive_fold_count":
                positive_folds,

            "profit_factor_above_1_fold_count":
                pf_above_one_folds,

            "total_fold_count":
                len(
                    fold_results
                ),

            "compounded_oos_return_pct":
                safe_float(
                    compounded_return,
                    2,
                ),

            "compounded_oos_cagr_pct":
                safe_float(
                    compounded_cagr,
                    2,
                ),

            "compounded_final_equity":
                safe_float(
                    compounded_equity,
                    6,
                ),

            "trade_count":
                total_trade_count,

            "rotation_count":
                total_rotation_count,
        },

        "folds":
            fold_results,

        "data_errors":
            errors,
    }


# ============================================================
# API
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

            "rotation_backtest":
                "/rotation-backtest?years=10",

            "oos_walk_forward":
                "/oos-walk-forward?years=10&folds=4&initial_train_pct=50",

            "backtest_all":
                "/backtest-all?years=10",

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
def analyze(
    symbol: str,
):

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

    return (
        scan_all_assets()
    )


# ============================================================
# STATIC ROUTES BEFORE /backtest/{symbol}
# ============================================================

@app.get("/rotation-backtest")
def rotation_backtest(

    years: int = Query(
        10,
        ge=2,
        le=18,
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

            years=
                years,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
                transaction_cost_pct,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/oos-walk-forward")
def oos_walk_forward(

    years: int = Query(
        10,
        ge=3,
        le=18,
    ),

    folds: int = Query(
        4,
        ge=2,
        le=8,
    ),

    initial_train_pct: int = Query(
        50,
        ge=30,
        le=80,
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

        return run_oos_walk_forward(

            years=
                years,

            folds=
                folds,

            initial_train_pct=
                initial_train_pct,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

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

            years=
                years,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

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

            symbol=
                symbol,

            years=
                years,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
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

<title>
Commodity Rotation Bot
</title>

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
Commodity Rotation Bot v1.2D
</h1>

<p>
Frozen v1.2C Sticky Rotation
</p>

<p>
Exact Years + Top-2 OOS / Walk-Forward
</p>


<div class="card">

<h2>
ROTATION BACKTEST
</h2>

<a href="/rotation-backtest?years=10">
10 Yıl
</a>

<a href="/rotation-backtest?years=5">
5 Yıl
</a>

<a href="/rotation-backtest?years=3">
3 Yıl
</a>

</div>


<div class="card">

<h2>
OOS / WALK-FORWARD
</h2>

<a href="/oos-walk-forward?years=10&folds=4&initial_train_pct=50">
Top-2 Sticky - 10 Yıl - 4 Fold
</a>

</div>


<div class="card">

<h2>
TARAMA
</h2>

<a href="/scan">
Tüm Emtiaları Tara
</a>

</div>


<div class="card">

<h2>
BACKTEST ALL
</h2>

<a href="/backtest-all?years=10">
10 Emtia - 10 Yıl
</a>

</div>


<div class="card">

<h2>
METALLER
</h2>

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

<h2>
ENERJİ
</h2>

<a href="/analyze/USO">
Petrol - USO
</a>

<a href="/analyze/UNG">
Doğal Gaz - UNG
</a>

</div>


<div class="card">

<h2>
TARIM
</h2>

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

<h2>
GENEL
</h2>

<a href="/analyze/DBC">
Geniş Emtia Sepeti - DBC
</a>

</div>


<div class="card">

<h2>
TEKLİ BACKTEST
</h2>

<a href="/backtest/GLD?years=10">
GLD
</a>

<a href="/backtest/SLV?years=10">
SLV
</a>

<a href="/backtest/CPER?years=10">
CPER
</a>

<a href="/backtest/USO?years=10">
USO
</a>

<a href="/backtest/UNG?years=10">
UNG
</a>

<a href="/backtest/DBA?years=10">
DBA
</a>

<a href="/backtest/WEAT?years=10">
WEAT
</a>

<a href="/backtest/CORN?years=10">
CORN
</a>

<a href="/backtest/SOYB?years=10">
SOYB
</a>

<a href="/backtest/DBC?years=10">
DBC
</a>

</div>


</body>

</html>
"""

    return HTMLResponse(
        content=html
    )

# ============================================================
# V1.3 EXPERIMENT
# 3-DAY ENTRY CONFIRMATION
#
# IMPORTANT:
# - V1.2D BASELINE IS NOT CHANGED
# - SCORE MODEL IS NOT CHANGED
# - ENTRY SCORE = 75
# - EXIT SCORE = 50
# - V1.2A SLOW EXIT IS NOT CHANGED
# - V1.2C STICKY ROTATION IS NOT CHANGED
# - ASSET UNIVERSE IS NOT CHANGED
# - RANKING IS NOT CHANGED
# - COSTS ARE NOT CHANGED
# - SIGNAL CLOSE -> EXECUTION NEXT OPEN
#
# ONLY EXPERIMENTAL CHANGE:
# A NEW POSITION CAN BE OPENED ONLY IF THE ASSET
# HAS BEEN "AL_ADAYI" FOR 3 CONSECUTIVE TRADING DAYS.
#
# EXISTING HOLDINGS DO NOT NEED TO REMAIN CONFIRMED.
# THEY CONTINUE UNTIL THE NORMAL V1.2A SLOW EXIT.
# ============================================================


V13_MODEL_NAME = "COMMODITY-ROTATION-V1.3-EXPERIMENT-CONFIRM3"

V13_CONFIRMATION_DAYS = 3


# ============================================================
# BUILD V1.3 CONFIRMED DATA
# ============================================================

def build_v13_confirmed_prepared(
    prepared,
    confirmation_days=V13_CONFIRMATION_DAYS,
):

    confirmed_prepared = {}

    confirmation_days = int(
        confirmation_days
    )

    if confirmation_days < 1:

        raise ValueError(
            "confirmation_days en az 1 olmalı."
        )

    for symbol, original_df in prepared.items():

        df = original_df.copy()

        # ----------------------------------------
        # ORIGINAL V1.2D AL_ADAYI STATUS
        # ----------------------------------------

        original_buy_candidate = (
            df["MODEL_SIGNAL"]
            .eq("AL_ADAYI")
            .astype(int)
        )

        # ----------------------------------------
        # CONSECUTIVE CONFIRMATION
        #
        # Example confirmation_days = 3:
        #
        # Day 1 AL_ADAYI
        # Day 2 AL_ADAYI
        # Day 3 AL_ADAYI
        #
        # -> Day 3 is confirmed.
        #
        # No future information is used.
        # ----------------------------------------

        rolling_confirmation = (
            original_buy_candidate
            .rolling(
                window=confirmation_days,
                min_periods=confirmation_days,
            )
            .sum()
        )

        confirmed = (
            rolling_confirmation
            >=
            confirmation_days
        )

        # ----------------------------------------
        # SAVE ORIGINAL SIGNAL FOR AUDIT
        # ----------------------------------------

        df["V12D_ORIGINAL_SIGNAL"] = (
            df["MODEL_SIGNAL"]
        )

        df["V13_CONFIRMATION_DAYS"] = (
            confirmation_days
        )

        df["V13_ENTRY_CONFIRMED"] = (
            confirmed
        )

        # ----------------------------------------
        # IMPORTANT
        #
        # run_rotation_variant() uses MODEL_SIGNAL
        # only for NEW ENTRY candidate selection.
        #
        # Exit logic does NOT depend on MODEL_SIGNAL.
        # Exit remains:
        #
        # score <= exit_score
        # OR EMA20 < EMA50
        # OR Close < EMA100
        # OR EMA50 < EMA200
        #
        # Therefore we can safely mask AL_ADAYI
        # for unconfirmed entry candidates without
        # changing the frozen exit model.
        # ----------------------------------------

        df.loc[
            ~confirmed,
            "MODEL_SIGNAL",
        ] = "BEKLE"

        # Confirmed rows keep their original signal.
        # Since confirmed requires AL_ADAYI on all
        # confirmation days, current row is AL_ADAYI.

        confirmed_prepared[symbol] = df

    return confirmed_prepared


# ============================================================
# V1.3 NORMAL ROTATION BACKTEST
# TOP-1 + TOP-2
# ============================================================

def run_v13_rotation_backtest(
    years=10,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
    confirmation_days=V13_CONFIRMATION_DAYS,
):

    (
        prepared,
        common_dates,
        errors,
    ) = prepare_rotation_data(
        years
    )

    confirmed_prepared = (
        build_v13_confirmed_prepared(
            prepared=
                prepared,

            confirmation_days=
                confirmation_days,
        )
    )

    top1 = run_rotation_variant(

        prepared=
            confirmed_prepared,

        common_dates=
            common_dates,

        top_n=
            1,

        entry_score=
            entry_score,

        exit_score=
            exit_score,

        transaction_cost_pct=
            transaction_cost_pct,
    )

    top2 = run_rotation_variant(

        prepared=
            confirmed_prepared,

        common_dates=
            common_dates,

        top_n=
            2,

        entry_score=
            entry_score,

        exit_score=
            exit_score,

        transaction_cost_pct=
            transaction_cost_pct,
    )

    # ----------------------------------------
    # DBC BUY & HOLD BENCHMARK
    # SAME PERIOD
    # ----------------------------------------

    dbc_return = None

    if "DBC" in prepared:

        dbc = prepared["DBC"]

        first_date = (
            common_dates[0]
        )

        last_date = (
            common_dates[-1]
        )

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

        "model":
            V13_MODEL_NAME,

        "test":
            "V1.3_CONFIRM3_ROTATION_BACKTEST",

        "generated_at_utc":
            utc_now(),

        "test_period": {

            "start":
                str(
                    common_dates[0].date()
                ),

            "end":
                str(
                    common_dates[-1].date()
                ),

            "years_requested":
                years,

            "common_trading_days":
                len(
                    common_dates
                ),
        },

        "experimental_change": {

            "name":
                "ENTRY_CONFIRMATION",

            "confirmation_days":
                confirmation_days,

            "rule":
                (
                    "New entry requires AL_ADAYI on "
                    f"{confirmation_days} consecutive closes"
                ),

            "applies_to":
                "NEW_ENTRIES_ONLY",

            "existing_holdings":
                "UNCHANGED_UNTIL_V1.2A_SLOW_EXIT",
        },

        "frozen_settings": {

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

            "transaction_cost_pct_each_side":
                transaction_cost_pct,

            "ranking":
                "SCORE, then MOM120, MOM60, MOM20",

            "base_entry_eligibility":
                "AL_ADAYI and score >= entry_score",

            "additional_v13_entry_filter":
                (
                    f"AL_ADAYI for {confirmation_days} "
                    "consecutive closes"
                ),

            "exit_model":
                "V1.2A_SLOW_EXIT",

            "rotation_model":
                "V1.2C_STICKY_ROTATION",

            "execution":
                "Signal at close, next trading day open",

            "parameter_optimization":
                False,

            "long_only":
                True,

            "short":
                False,

            "leverage":
                False,

            "automatic_orders":
                False,
        },

        "benchmark": {

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
# V1.3 CHRONOLOGICAL OOS
# TOP-2 STICKY
# SAME SPLITS AS V1.2D
# ============================================================

def run_v13_oos_walk_forward(
    years=10,
    folds=4,
    initial_train_pct=50,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
    confirmation_days=V13_CONFIRMATION_DAYS,
):

    (
        prepared,
        common_dates,
        errors,
    ) = prepare_rotation_data(
        years
    )

    confirmed_prepared = (
        build_v13_confirmed_prepared(
            prepared=
                prepared,

            confirmation_days=
                confirmation_days,
        )
    )

    n = len(
        common_dates
    )

    if n < 500:

        raise ValueError(
            "OOS/walk-forward için yeterli ortak tarih yok."
        )

    folds = int(
        folds
    )

    if folds < 2:

        raise ValueError(
            "folds en az 2 olmalı."
        )

    initial_train_pct = int(
        initial_train_pct
    )

    if not (
        30
        <=
        initial_train_pct
        <=
        80
    ):

        raise ValueError(
            "initial_train_pct 30 ile 80 arasında olmalı."
        )

    train_end = int(
        n
        *
        initial_train_pct
        /
        100
    )

    train_end = max(
        250,
        train_end,
    )

    minimum_oos_days = (
        folds * 40
    )

    if (
        n - train_end
        <
        minimum_oos_days
    ):

        train_end = (
            n
            -
            minimum_oos_days
        )

    if train_end < 250:

        raise ValueError(
            "Train/OOS bölünmesi için yeterli veri yok."
        )

    remaining = (
        n
        -
        train_end
    )

    if (
        remaining
        <
        minimum_oos_days
    ):

        raise ValueError(
            "Seçilen fold sayısı için OOS dönemi çok kısa."
        )

    base = (
        remaining
        //
        folds
    )

    extra = (
        remaining
        %
        folds
    )

    fold_results = []

    cursor = (
        train_end
    )

    compounded_equity = 1.0

    total_trade_count = 0
    total_rotation_count = 0

    # ----------------------------------------
    # FOLDS
    # ----------------------------------------

    for fold_no in range(
        1,
        folds + 1,
    ):

        fold_len = (
            base
            +
            (
                1
                if fold_no <= extra
                else 0
            )
        )

        start_idx = (
            cursor
        )

        end_idx = min(
            n,
            cursor + fold_len,
        )

        # ----------------------------------------
        # IMPORTANT:
        #
        # Same methodology as V1.2D:
        # one prior common trading day is included
        # so a signal can execute on first OOS day.
        #
        # The 3-day confirmation itself is already
        # calculated from the full chronological
        # indicator history BEFORE this slicing.
        #
        # Therefore the filter can legitimately use
        # the two preceding closes without using
        # future information.
        # ----------------------------------------

        slice_start = max(
            0,
            start_idx - 1,
        )

        fold_dates = (
            common_dates[
                slice_start:end_idx
            ]
        )

        if len(
            fold_dates
        ) < 2:

            break

        result = run_rotation_variant(

            prepared=
                confirmed_prepared,

            common_dates=
                fold_dates,

            top_n=
                2,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
                transaction_cost_pct,
        )

        performance = (
            result[
                "performance"
            ]
        )

        fold_equity = (
            performance[
                "final_equity"
            ]
        )

        if fold_equity is None:

            fold_equity = 1.0

        fold_equity = float(
            fold_equity
        )

        compounded_equity *= (
            fold_equity
        )

        total_trade_count += int(
            performance[
                "trade_count"
            ]
            or 0
        )

        total_rotation_count += int(
            performance[
                "rotation_count"
            ]
            or 0
        )

        fold_results.append({

            "fold":
                fold_no,

            "train_period": {

                "start":
                    str(
                        common_dates[
                            0
                        ].date()
                    ),

                "end":
                    str(
                        common_dates[
                            start_idx - 1
                        ].date()
                    ),

                "trading_days":
                    start_idx,
            },

            "oos_period": {

                "start":
                    str(
                        common_dates[
                            start_idx
                        ].date()
                    ),

                "end":
                    str(
                        common_dates[
                            end_idx - 1
                        ].date()
                    ),

                "trading_days":
                    (
                        end_idx
                        -
                        start_idx
                    ),
            },

            "result":
                compact_rotation_result(
                    result
                ),
        })

        cursor = (
            end_idx
        )

    if not fold_results:

        raise ValueError(
            "V1.3 OOS fold üretilemedi."
        )

    # ----------------------------------------
    # OOS SUMMARY
    # ----------------------------------------

    oos_start = pd.Timestamp(
        fold_results[
            0
        ][
            "oos_period"
        ][
            "start"
        ]
    )

    oos_end = pd.Timestamp(
        fold_results[
            -1
        ][
            "oos_period"
        ][
            "end"
        ]
    )

    elapsed_years = (
        (
            oos_end
            -
            oos_start
        ).days
        /
        365.25
    )

    compounded_return = (
        compounded_equity
        -
        1
    ) * 100

    if (
        elapsed_years > 0
        and
        compounded_equity > 0
    ):

        compounded_cagr = (
            compounded_equity
            **
            (
                1
                /
                elapsed_years
            )
            -
            1
        ) * 100

    else:

        compounded_cagr = None

    positive_folds = sum(

        1

        for fold in fold_results

        if (
            fold[
                "result"
            ][
                "performance"
            ][
                "strategy_total_return_pct"
            ]
            is not None

            and

            fold[
                "result"
            ][
                "performance"
            ][
                "strategy_total_return_pct"
            ]
            > 0
        )
    )

    pf_above_one_folds = sum(

        1

        for fold in fold_results

        if (
            fold[
                "result"
            ][
                "performance"
            ][
                "profit_factor"
            ]
            is not None

            and

            fold[
                "result"
            ][
                "performance"
            ][
                "profit_factor"
            ]
            > 1
        )
    )

    return {

        "model":
            V13_MODEL_NAME,

        "test":
            "TOP2_STICKY_CONFIRM3_CHRONOLOGICAL_OOS",

        "generated_at_utc":
            utc_now(),

        "full_period": {

            "start":
                str(
                    common_dates[
                        0
                    ].date()
                ),

            "end":
                str(
                    common_dates[
                        -1
                    ].date()
                ),

            "years_requested":
                years,

            "common_trading_days":
                n,
        },

        "experimental_change": {

            "confirmation_days":
                confirmation_days,

            "rule":
                (
                    "New position requires AL_ADAYI "
                    f"for {confirmation_days} consecutive closes"
                ),

            "applies_only_to":
                "NEW_POSITION_ENTRY",

            "exit_rules_changed":
                False,

            "score_rules_changed":
                False,

            "asset_universe_changed":
                False,

            "ranking_changed":
                False,
        },

        "frozen_settings": {

            "portfolio":
                "TOP_2_EQUAL_WEIGHT",

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

            "transaction_cost_pct_each_side":
                transaction_cost_pct,

            "ranking":
                "SCORE, then MOM120, MOM60, MOM20",

            "base_eligibility":
                "AL_ADAYI and score >= entry_score",

            "confirmation":
                (
                    f"{confirmation_days} consecutive "
                    "AL_ADAYI closes"
                ),

            "exit_model":
                "V1.2A_SLOW_EXIT",

            "rotation_model":
                "V1.2C_STICKY_ROTATION",

            "execution":
                "Signal at close, next trading day open",

            "parameter_optimization":
                False,

            "long_only":
                True,

            "short":
                False,

            "leverage":
                False,

            "automatic_orders":
                False,
        },

        "walk_forward": {

            "initial_train_pct":
                initial_train_pct,

            "fold_count":
                len(
                    fold_results
                ),

            "fold_rule":
                (
                    "Same chronological folds as V1.2D; "
                    "each fold starts from CASH; "
                    "no fitting or parameter optimization"
                ),
        },

        "oos_summary": {

            "oos_start":
                str(
                    oos_start.date()
                ),

            "oos_end":
                str(
                    oos_end.date()
                ),

            "positive_fold_count":
                positive_folds,

            "profit_factor_above_1_fold_count":
                pf_above_one_folds,

            "total_fold_count":
                len(
                    fold_results
                ),

            "compounded_oos_return_pct":
                safe_float(
                    compounded_return,
                    2,
                ),

            "compounded_oos_cagr_pct":
                safe_float(
                    compounded_cagr,
                    2,
                ),

            "compounded_final_equity":
                safe_float(
                    compounded_equity,
                    6,
                ),

            "trade_count":
                total_trade_count,

            "rotation_count":
                total_rotation_count,
        },

        "folds":
            fold_results,

        "data_errors":
            errors,
    }


# ============================================================
# V1.2D vs V1.3 COMPARISON
# SAME DATA / SAME FOLDS
# ============================================================

def run_v13_comparison(
    years=10,
    folds=4,
    initial_train_pct=50,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
    confirmation_days=V13_CONFIRMATION_DAYS,
):

    baseline = run_oos_walk_forward(

        years=
            years,

        folds=
            folds,

        initial_train_pct=
            initial_train_pct,

        entry_score=
            entry_score,

        exit_score=
            exit_score,

        transaction_cost_pct=
            transaction_cost_pct,
    )

    candidate = run_v13_oos_walk_forward(

        years=
            years,

        folds=
            folds,

        initial_train_pct=
            initial_train_pct,

        entry_score=
            entry_score,

        exit_score=
            exit_score,

        transaction_cost_pct=
            transaction_cost_pct,

        confirmation_days=
            confirmation_days,
    )

    baseline_summary = (
        baseline[
            "oos_summary"
        ]
    )

    candidate_summary = (
        candidate[
            "oos_summary"
        ]
    )

    fold_comparison = []

    max_folds = min(
        len(
            baseline[
                "folds"
            ]
        ),
        len(
            candidate[
                "folds"
            ]
        ),
    )

    for i in range(
        max_folds
    ):

        base_fold = (
            baseline[
                "folds"
            ][i]
        )

        cand_fold = (
            candidate[
                "folds"
            ][i]
        )

        bp = (
            base_fold[
                "result"
            ][
                "performance"
            ]
        )

        cp = (
            cand_fold[
                "result"
            ][
                "performance"
            ]
        )

        base_return = (
            bp[
                "strategy_total_return_pct"
            ]
        )

        cand_return = (
            cp[
                "strategy_total_return_pct"
            ]
        )

        return_difference = None

        if (
            base_return is not None
            and
            cand_return is not None
        ):

            return_difference = (
                float(
                    cand_return
                )
                -
                float(
                    base_return
                )
            )

        fold_comparison.append({

            "fold":
                i + 1,

            "oos_period":
                base_fold[
                    "oos_period"
                ],

            "v12d": {

                "return_pct":
                    base_return,

                "max_drawdown_pct":
                    bp[
                        "max_drawdown_daily_pct"
                    ],

                "profit_factor":
                    bp[
                        "profit_factor"
                    ],

                "trade_count":
                    bp[
                        "trade_count"
                    ],

                "win_rate_pct":
                    bp[
                        "win_rate_pct"
                    ],

                "average_holding_days":
                    bp[
                        "average_holding_days"
                    ],
            },

            "v13_confirm3": {

                "return_pct":
                    cand_return,

                "max_drawdown_pct":
                    cp[
                        "max_drawdown_daily_pct"
                    ],

                "profit_factor":
                    cp[
                        "profit_factor"
                    ],

                "trade_count":
                    cp[
                        "trade_count"
                    ],

                "win_rate_pct":
                    cp[
                        "win_rate_pct"
                    ],

                "average_holding_days":
                    cp[
                        "average_holding_days"
                    ],
            },

            "v13_minus_v12d_return_pct_points":
                safe_float(
                    return_difference,
                    2,
                ),
        })

    baseline_return = (
        baseline_summary[
            "compounded_oos_return_pct"
        ]
    )

    candidate_return = (
        candidate_summary[
            "compounded_oos_return_pct"
        ]
    )

    aggregate_difference = None

    if (
        baseline_return is not None
        and
        candidate_return is not None
    ):

        aggregate_difference = (
            float(
                candidate_return
            )
            -
            float(
                baseline_return
            )
        )

    return {

        "experiment":
            "V1.2D_BASELINE_VS_V1.3_CONFIRM3",

        "generated_at_utc":
            utc_now(),

        "methodology_note":
            (
                "V1.3 was designed after inspection of prior V1.2D "
                "results. Therefore these historical folds are a "
                "research robustness comparison, not pristine unseen "
                "validation for V1.3."
            ),

        "only_rule_change":
            (
                f"V1.3 requires {confirmation_days} consecutive "
                "AL_ADAYI closes before filling an empty portfolio slot."
            ),

        "unchanged": [

            "Asset universe",

            "Score calculation",

            "Entry score threshold",

            "V1.2A slow exit",

            "V1.2C sticky holding logic",

            "Top-2 equal weight",

            "Ranking SCORE/MOM120/MOM60/MOM20",

            "Transaction costs",

            "Close signal / next-open execution",

            "Long only",

            "No leverage",

            "No automatic orders",
        ],

        "v12d_baseline_summary":
            baseline_summary,

        "v13_candidate_summary":
            candidate_summary,

        "aggregate_return_difference_pct_points":
            safe_float(
                aggregate_difference,
                2,
            ),

        "fold_comparison":
            fold_comparison,

        "v12d_full_result":
            baseline,

        "v13_full_result":
            candidate,
    }


# ============================================================
# V1.3 API ROUTES
# ============================================================

@app.get(
    "/v13-rotation-backtest"
)
def v13_rotation_backtest_endpoint(

    years: int = Query(
        10,
        ge=3,
        le=15,
    ),

    entry_score: int = Query(
        75,
        ge=0,
        le=100,
    ),

    exit_score: int = Query(
        50,
        ge=0,
        le=100,
    ),

    transaction_cost_pct: float = Query(
        0.10,
        ge=0,
        le=5,
    ),
):

    try:

        return run_v13_rotation_backtest(

            years=
                years,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
                transaction_cost_pct,

            confirmation_days=
                V13_CONFIRMATION_DAYS,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get(
    "/v13-oos-walk-forward"
)
def v13_oos_walk_forward_endpoint(

    years: int = Query(
        10,
        ge=5,
        le=15,
    ),

    folds: int = Query(
        4,
        ge=2,
        le=8,
    ),

    initial_train_pct: int = Query(
        50,
        ge=30,
        le=80,
    ),

    entry_score: int = Query(
        75,
        ge=0,
        le=100,
    ),

    exit_score: int = Query(
        50,
        ge=0,
        le=100,
    ),

    transaction_cost_pct: float = Query(
        0.10,
        ge=0,
        le=5,
    ),
):

    try:

        return run_v13_oos_walk_forward(

            years=
                years,

            folds=
                folds,

            initial_train_pct=
                initial_train_pct,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
                transaction_cost_pct,

            confirmation_days=
                V13_CONFIRMATION_DAYS,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get(
    "/v13-compare"
)
def v13_compare_endpoint(

    years: int = Query(
        10,
        ge=5,
        le=15,
    ),

    folds: int = Query(
        4,
        ge=2,
        le=8,
    ),

    initial_train_pct: int = Query(
        50,
        ge=30,
        le=80,
    ),

    entry_score: int = Query(
        75,
        ge=0,
        le=100,
    ),

    exit_score: int = Query(
        50,
        ge=0,
        le=100,
    ),

    transaction_cost_pct: float = Query(
        0.10,
        ge=0,
        le=5,
    ),
):

    try:

        return run_v13_comparison(

            years=
                years,

            folds=
                folds,

            initial_train_pct=
                initial_train_pct,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
                transaction_cost_pct,

            confirmation_days=
                V13_CONFIRMATION_DAYS,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

# ============================================================
# V1.4 EXPERIMENT
# GLOBAL BREADTH REGIME FILTER
#
# FROZEN BASE:
#   V1.3 CONFIRM3
#
# ONLY NEW EXPERIMENTAL CHANGE:
#   New entries are allowed only when:
#       at least 5 of 10 assets have Close > EMA200
#
# IMPORTANT:
# - Existing holdings are NOT affected by breadth
# - Existing holdings exit only with V1.2A slow exit
# - CONFIRM3 remains unchanged
# - Entry score remains 75
# - Exit score remains 50
# - Ranking remains SCORE, MOM120, MOM60, MOM20
# - No asset exclusions
# - No optimization
# - Long only
# - No leverage
# - No automatic orders
# ============================================================


V14_MODEL_NAME = "COMMODITY-ROTATION-V1.4-EXPERIMENT-BREADTH5"
V14_CONFIRMATION_DAYS = 3
V14_BREADTH_MIN = 5


# ============================================================
# HELPER
# BASE AL_ADAYI CHECK
# ============================================================

def v14_is_al_adayi(
    prepared,
    symbol,
    signal_date,
    entry_score=75,
):
    if symbol not in prepared:
        return False

    df = prepared[symbol]

    if signal_date not in df.index:
        return False

    row = df.loc[signal_date]

    try:
        score = int(row["SCORE"])
        signal = str(row["MODEL_SIGNAL"])
    except Exception:
        return False

    return (
        score >= entry_score
        and
        signal == "AL_ADAYI"
    )


# ============================================================
# HELPER
# CONFIRM3
#
# Candidate must be AL_ADAYI on the current close
# AND previous 2 consecutive trading closes.
# ============================================================

def v14_confirmed_al_adayi(
    prepared,
    symbol,
    signal_date,
    entry_score=75,
    confirmation_days=3,
):
    if symbol not in prepared:
        return False

    df = prepared[symbol]

    if signal_date not in df.index:
        return False

    try:
        pos = df.index.get_loc(signal_date)
    except Exception:
        return False

    # Defensive handling in case index lookup returns slice/array
    if not isinstance(pos, (int, np.integer)):
        try:
            pos = int(np.where(df.index == signal_date)[0][-1])
        except Exception:
            return False

    if pos < confirmation_days - 1:
        return False

    dates_to_check = df.index[
        pos - confirmation_days + 1:
        pos + 1
    ]

    if len(dates_to_check) != confirmation_days:
        return False

    for dt in dates_to_check:

        if not v14_is_al_adayi(
            prepared=prepared,
            symbol=symbol,
            signal_date=dt,
            entry_score=entry_score,
        ):
            return False

    return True


# ============================================================
# GLOBAL BREADTH
#
# Breadth = number of assets with Close > EMA200
#
# IMPORTANT:
# This is calculated across the whole frozen universe.
# It is NOT asset-specific.
# ============================================================

def v14_market_breadth(
    prepared,
    signal_date,
):
    above_ema200 = 0
    available = 0
    details = {}

    for symbol, df in prepared.items():

        if signal_date not in df.index:
            continue

        row = df.loc[signal_date]

        try:
            close = float(row["Close"])
            ema200 = float(row["EMA200"])
        except Exception:
            continue

        if (
            np.isnan(close)
            or
            np.isnan(ema200)
        ):
            continue

        available += 1

        is_above = (
            close > ema200
        )

        if is_above:
            above_ema200 += 1

        details[symbol] = {
            "close": safe_float(close, 4),
            "ema200": safe_float(ema200, 4),
            "above_ema200": bool(is_above),
        }

    return {
        "above_ema200": above_ema200,
        "available_assets": available,
        "total_universe": len(ASSETS),
        "breadth_pct_of_available": safe_float(
            (
                above_ema200
                /
                available
                *
                100
            )
            if available
            else 0,
            2,
        ),
        "details": details,
    }


# ============================================================
# V1.3 BASELINE CANDIDATES
#
# CONFIRM3 only.
# No breadth filter.
#
# Used here so v1.3 and v1.4 are compared with exactly
# the same portfolio engine and same data window.
# ============================================================

def v13_baseline_candidates(
    prepared,
    signal_date,
    entry_score=75,
):
    candidates = []

    for symbol, df in prepared.items():

        if signal_date not in df.index:
            continue

        if not v14_confirmed_al_adayi(
            prepared=prepared,
            symbol=symbol,
            signal_date=signal_date,
            entry_score=entry_score,
            confirmation_days=V14_CONFIRMATION_DAYS,
        ):
            continue

        row = df.loc[signal_date]

        candidates.append({
            "symbol": symbol,
            "score": int(row["SCORE"]),
            "mom120": float(row["MOM120"]),
            "mom60": float(row["MOM60"]),
            "mom20": float(row["MOM20"]),
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


# ============================================================
# V1.4 CANDIDATES
#
# CONFIRM3
# +
# GLOBAL BREADTH >= 5
# ============================================================

def v14_candidates(
    prepared,
    signal_date,
    entry_score=75,
    breadth_min=5,
):
    breadth = v14_market_breadth(
        prepared,
        signal_date,
    )

    # --------------------------------------------------------
    # GLOBAL REGIME GATE
    # --------------------------------------------------------

    if (
        breadth["above_ema200"]
        <
        breadth_min
    ):
        return [], breadth

    candidates = []

    for symbol, df in prepared.items():

        if signal_date not in df.index:
            continue

        # ----------------------------------------------------
        # V1.3 CONFIRM3
        # ----------------------------------------------------

        if not v14_confirmed_al_adayi(
            prepared=prepared,
            symbol=symbol,
            signal_date=signal_date,
            entry_score=entry_score,
            confirmation_days=V14_CONFIRMATION_DAYS,
        ):
            continue

        row = df.loc[signal_date]

        candidates.append({
            "symbol": symbol,
            "score": int(row["SCORE"]),
            "mom120": float(row["MOM120"]),
            "mom60": float(row["MOM60"]),
            "mom20": float(row["MOM20"]),
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

    return candidates, breadth


# ============================================================
# GENERIC V1.3 / V1.4 TOP-N STICKY ENGINE
#
# mode:
#   "V13" = CONFIRM3
#   "V14" = CONFIRM3 + BREADTH >= 5
#
# Existing holdings:
#   NEVER sold because breadth falls below 5.
#   NEVER sold because another candidate ranks higher.
#
# They exit ONLY via frozen V1.2A slow exit.
# ============================================================

def run_v14_rotation_variant(
    prepared,
    common_dates,
    top_n=2,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
    mode="V14",
    breadth_min=5,
):
    if len(common_dates) < 2:
        raise ValueError(
            "Rotation backtest için yeterli tarih yok."
        )

    cost = (
        transaction_cost_pct
        /
        100
    )

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

    blocked_entry_days = 0
    breadth_pass_days = 0
    breadth_fail_days = 0

    for i in range(
        len(common_dates) - 1
    ):
        signal_date = common_dates[i]
        next_date = common_dates[i + 1]

        # ====================================================
        # 1. MARK TO MARKET
        # OPEN -> NEXT OPEN
        # ====================================================

        if holdings:

            weight = (
                1.0
                /
                top_n
            )

            daily_return = 0.0

            for symbol in holdings:

                df = prepared[symbol]

                current_open = float(
                    df.loc[
                        signal_date,
                        "Open",
                    ]
                )

                next_open = float(
                    df.loc[
                        next_date,
                        "Open",
                    ]
                )

                if current_open > 0:

                    daily_return += (
                        (
                            next_open
                            /
                            current_open
                            -
                            1
                        )
                        *
                        weight
                    )

            equity *= (
                1
                +
                daily_return
            )

        # ====================================================
        # 2. EXIT CHECK
        # FROZEN V1.2A
        # ====================================================

        old_symbols = list(
            holdings.keys()
        )

        exiting = []

        for symbol in old_symbols:

            row = (
                prepared[symbol]
                .loc[signal_date]
            )

            if rotation_exit_required(
                row,
                exit_score,
            ):
                exiting.append(
                    symbol
                )

        survivors = [
            symbol
            for symbol in old_symbols
            if symbol not in exiting
        ]

        # ====================================================
        # 3. DETERMINE EMPTY SLOTS
        # ====================================================

        next_symbols = list(
            survivors
        )

        free_slots = max(
            0,
            top_n
            -
            len(next_symbols),
        )

        entering = []

        breadth = v14_market_breadth(
            prepared,
            signal_date,
        )

        # ====================================================
        # 4. NEW ENTRY LOGIC
        #
        # IMPORTANT:
        # breadth only controls EMPTY SLOT filling.
        # It does NOT touch survivors.
        # ====================================================

        if free_slots > 0:

            if mode == "V13":

                candidates = (
                    v13_baseline_candidates(
                        prepared=prepared,
                        signal_date=signal_date,
                        entry_score=entry_score,
                    )
                )

            else:

                candidates, breadth = (
                    v14_candidates(
                        prepared=prepared,
                        signal_date=signal_date,
                        entry_score=entry_score,
                        breadth_min=breadth_min,
                    )
                )

            candidate_symbols = [
                x["symbol"]
                for x in candidates
            ]

            # -----------------------------------------------
            # Diagnostic counters
            # -----------------------------------------------

            if mode == "V14":

                if (
                    breadth["above_ema200"]
                    >=
                    breadth_min
                ):
                    breadth_pass_days += 1

                else:
                    breadth_fail_days += 1

                    # Was there actually a CONFIRM3 candidate
                    # that would have entered under v1.3?
                    baseline_candidates = (
                        v13_baseline_candidates(
                            prepared=prepared,
                            signal_date=signal_date,
                            entry_score=entry_score,
                        )
                    )

                    baseline_available = [
                        x
                        for x in baseline_candidates
                        if x["symbol"] not in next_symbols
                    ]

                    if baseline_available:
                        blocked_entry_days += 1

            # -----------------------------------------------
            # Sticky fill
            # -----------------------------------------------

            for symbol in candidate_symbols:

                if symbol in next_symbols:
                    continue

                entering.append(
                    symbol
                )

                next_symbols.append(
                    symbol
                )

                if (
                    len(entering)
                    >=
                    free_slots
                ):
                    break

        # ====================================================
        # 5. TRANSACTION COST
        # ====================================================

        sides = (
            len(exiting)
            +
            len(entering)
        )

        if sides:

            equity *= (
                1
                -
                cost
                *
                sides
                /
                top_n
            )

        # ====================================================
        # 6. CLOSE EXITING TRADES
        # ====================================================

        for symbol in exiting:

            old = holdings[symbol]

            entry_date = (
                old["entry_date"]
            )

            entry_open = float(
                prepared[symbol]
                .loc[
                    entry_date,
                    "Open",
                ]
            )

            exit_open = float(
                prepared[symbol]
                .loc[
                    next_date,
                    "Open",
                ]
            )

            gross = (
                exit_open
                /
                entry_open
                -
                1
            )

            net = (
                (1 + gross)
                *
                (1 - cost)
                *
                (1 - cost)
                -
                1
            )

            trades.append({
                "symbol":
                    symbol,

                "entry_date":
                    str(
                        entry_date.date()
                    ),

                "exit_date":
                    str(
                        next_date.date()
                    ),

                "entry_score":
                    old["entry_score"],

                "exit_score":
                    safe_int(
                        prepared[symbol]
                        .loc[
                            signal_date,
                            "SCORE",
                        ]
                    ),

                "return_pct":
                    safe_float(
                        net * 100,
                        2,
                    ),

                "holding_days":
                    (
                        next_date
                        -
                        entry_date
                    ).days,

                "exit_reason":
                    "V1.2A_SLOW_EXIT",

                "entry_breadth":
                    old.get(
                        "entry_breadth"
                    ),
            })

            exit_count += 1

        # ====================================================
        # 7. BUILD NEXT HOLDINGS
        # ====================================================

        new_holdings = {}

        for symbol in survivors:

            new_holdings[symbol] = (
                holdings[symbol]
            )

        for symbol in entering:

            score = int(
                prepared[symbol]
                .loc[
                    signal_date,
                    "SCORE",
                ]
            )

            new_holdings[symbol] = {
                "entry_date":
                    next_date,

                "entry_score":
                    score,

                "entry_breadth":
                    breadth[
                        "above_ema200"
                    ],
            }

            entry_count += 1

        if (
            exiting
            and
            entering
        ):
            rotation_count += 1

        # ====================================================
        # 8. REBALANCE LOG
        # ====================================================

        if sides:

            rebalances.append({
                "signal_date":
                    str(
                        signal_date.date()
                    ),

                "execution_date":
                    str(
                        next_date.date()
                    ),

                "from":
                    (
                        old_symbols
                        if old_symbols
                        else ["CASH"]
                    ),

                "to":
                    (
                        next_symbols
                        if next_symbols
                        else ["CASH"]
                    ),

                "exiting":
                    exiting,

                "entering":
                    entering,

                "survivors":
                    survivors,

                "transaction_sides":
                    sides,

                "breadth_above_ema200":
                    breadth[
                        "above_ema200"
                    ],

                "breadth_available":
                    breadth[
                        "available_assets"
                    ],

                "breadth_gate":
                    (
                        "PASS"
                        if (
                            breadth[
                                "above_ema200"
                            ]
                            >=
                            breadth_min
                        )
                        else
                        "BLOCK"
                    ),
            })

        holdings = new_holdings

        # ====================================================
        # 9. CASH / INVESTED
        # ====================================================

        if holdings:
            invested_days += 1
        else:
            cash_days += 1

        # ====================================================
        # 10. DAILY DRAWDOWN
        # ====================================================

        peak = max(
            peak,
            equity,
        )

        drawdown = (
            equity
            /
            peak
            -
            1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

        equity_curve.append({
            "date":
                str(
                    next_date.date()
                ),

            "equity":
                safe_float(
                    equity,
                    6,
                ),

            "drawdown_pct":
                safe_float(
                    drawdown,
                    2,
                ),

            "holdings":
                (
                    list(
                        holdings.keys()
                    )
                    if holdings
                    else ["CASH"]
                ),

            "breadth_above_ema200":
                breadth[
                    "above_ema200"
                ],
        })

    # ========================================================
    # 11. FORCED FINAL CLOSE
    # ========================================================

    final_date = common_dates[-1]

    if holdings:

        weight = (
            1.0
            /
            top_n
        )

        final_day_return = 0.0

        for symbol in holdings:

            df = prepared[symbol]

            final_open = float(
                df.loc[
                    final_date,
                    "Open",
                ]
            )

            final_close = float(
                df.loc[
                    final_date,
                    "Close",
                ]
            )

            if final_open > 0:

                final_day_return += (
                    (
                        final_close
                        /
                        final_open
                        -
                        1
                    )
                    *
                    weight
                )

        equity *= (
            1
            +
            final_day_return
        )

        equity *= (
            1
            -
            cost
            *
            len(holdings)
            /
            top_n
        )

        for symbol, old in holdings.items():

            entry_date = (
                old["entry_date"]
            )

            entry_open = float(
                prepared[symbol]
                .loc[
                    entry_date,
                    "Open",
                ]
            )

            final_close = float(
                prepared[symbol]
                .loc[
                    final_date,
                    "Close",
                ]
            )

            gross = (
                final_close
                /
                entry_open
                -
                1
            )

            net = (
                (1 + gross)
                *
                (1 - cost)
                *
                (1 - cost)
                -
                1
            )

            trades.append({
                "symbol":
                    symbol,

                "entry_date":
                    str(
                        entry_date.date()
                    ),

                "exit_date":
                    str(
                        final_date.date()
                    ),

                "entry_score":
                    old["entry_score"],

                "exit_score":
                    safe_int(
                        prepared[symbol]
                        .loc[
                            final_date,
                            "SCORE",
                        ]
                    ),

                "return_pct":
                    safe_float(
                        net * 100,
                        2,
                    ),

                "holding_days":
                    (
                        final_date
                        -
                        entry_date
                    ).days,

                "entry_breadth":
                    old.get(
                        "entry_breadth"
                    ),

                "forced_exit_at_test_end":
                    True,
            })

            exit_count += 1

        peak = max(
            peak,
            equity,
        )

        drawdown = (
            equity
            /
            peak
            -
            1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    # ========================================================
    # 12. PERFORMANCE
    # ========================================================

    returns = [
        x["return_pct"]
        for x in trades
        if x["return_pct"] is not None
    ]

    winners = [
        x
        for x in returns
        if x > 0
    ]

    losers = [
        x
        for x in returns
        if x <= 0
    ]

    if returns:

        win_rate = (
            len(winners)
            /
            len(returns)
            *
            100
        )

        avg_trade = float(
            np.mean(returns)
        )

        median_trade = float(
            np.median(returns)
        )

        avg_holding = float(
            np.mean([
                x["holding_days"]
                for x in trades
            ])
        )

    else:

        win_rate = 0.0
        avg_trade = 0.0
        median_trade = 0.0
        avg_holding = 0.0

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

        profit_factor = 0.0

    start_date = common_dates[0]
    end_date = common_dates[-1]

    elapsed_years = (
        (
            end_date
            -
            start_date
        ).days
        /
        365.25
    )

    total_return = (
        equity
        -
        1
    ) * 100

    if (
        elapsed_years > 0
        and
        equity > 0
    ):

        cagr = (
            equity
            **
            (
                1
                /
                elapsed_years
            )
            -
            1
        ) * 100

    else:

        cagr = None

    active_days = (
        cash_days
        +
        invested_days
    )

    cash_pct = (
        cash_days
        /
        active_days
        *
        100
        if active_days
        else 0
    )

    invested_pct = (
        invested_days
        /
        active_days
        *
        100
        if active_days
        else 0
    )

    return {
        "portfolio":
            f"TOP_{top_n}",

        "slots":
            top_n,

        "weighting":
            "EQUAL_WEIGHT",

        "rotation_model":
            (
                "V1.3_CONFIRM3"
                if mode == "V13"
                else
                "V1.4_CONFIRM3_PLUS_BREADTH5"
            ),

        "performance": {
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

            "max_drawdown_daily_pct":
                safe_float(
                    max_drawdown,
                    2,
                ),

            "trade_count":
                len(trades),

            "win_rate_pct":
                safe_float(
                    win_rate,
                    2,
                ),

            "profit_factor":
                safe_float(
                    profit_factor,
                    3,
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

            "entry_count":
                entry_count,

            "exit_count":
                exit_count,

            "rotation_count":
                rotation_count,

            "cash_days":
                cash_days,

            "cash_time_pct":
                safe_float(
                    cash_pct,
                    2,
                ),

            "invested_time_pct":
                safe_float(
                    invested_pct,
                    2,
                ),

            "final_equity":
                safe_float(
                    equity,
                    6,
                ),
        },

        "breadth_diagnostics": {
            "breadth_min":
                breadth_min,

            "breadth_definition":
                "Number of assets with Close > EMA200",

            "blocked_entry_days":
                (
                    blocked_entry_days
                    if mode == "V14"
                    else 0
                ),

            "breadth_pass_empty_slot_days":
                (
                    breadth_pass_days
                    if mode == "V14"
                    else None
                ),

            "breadth_fail_empty_slot_days":
                (
                    breadth_fail_days
                    if mode == "V14"
                    else None
                ),
        },

        "trades":
            trades,

        "rebalances":
            rebalances,

        "equity_curve":
            equity_curve,
    }


# ============================================================
# BENCHMARK
# ============================================================

def v14_dbc_benchmark(
    prepared,
    common_dates,
):
    if (
        "DBC" not in prepared
        or
        not common_dates
    ):
        return None

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

    if first_open <= 0:
        return None

    return (
        last_close
        /
        first_open
        -
        1
    ) * 100


# ============================================================
# COMPACT RESULT
# Used for OOS output so JSON is not unnecessarily huge.
# ============================================================

def v14_compact_result(
    result,
):
    return {
        "portfolio":
            result["portfolio"],

        "rotation_model":
            result[
                "rotation_model"
            ],

        "performance":
            result[
                "performance"
            ],

        "breadth_diagnostics":
            result.get(
                "breadth_diagnostics"
            ),

        "trades":
            result[
                "trades"
            ],

        "rebalances":
            result[
                "rebalances"
            ],
    }


# ============================================================
# 10-YEAR / N-YEAR CONTINUOUS COMPARISON
#
# V1.3 baseline
# versus
# V1.4 breadth5
# ============================================================

def run_v14_comparison(
    years=10,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
    breadth_min=5,
):
    (
        prepared,
        common_dates,
        errors,
    ) = prepare_rotation_data(
        years
    )

    if len(common_dates) < 2:
        raise ValueError(
            "Yeterli ortak trading date yok."
        )

    # --------------------------------------------------------
    # SAME DATA
    # SAME PORTFOLIO ENGINE
    # ONLY DIFFERENCE = BREADTH FILTER
    # --------------------------------------------------------

    v13 = run_v14_rotation_variant(
        prepared=prepared,
        common_dates=common_dates,
        top_n=2,
        entry_score=entry_score,
        exit_score=exit_score,
        transaction_cost_pct=transaction_cost_pct,
        mode="V13",
        breadth_min=breadth_min,
    )

    v14 = run_v14_rotation_variant(
        prepared=prepared,
        common_dates=common_dates,
        top_n=2,
        entry_score=entry_score,
        exit_score=exit_score,
        transaction_cost_pct=transaction_cost_pct,
        mode="V14",
        breadth_min=breadth_min,
    )

    dbc_return = v14_dbc_benchmark(
        prepared,
        common_dates,
    )

    p13 = v13["performance"]
    p14 = v14["performance"]

    return {
        "experiment":
            "V1.3_CONFIRM3_VS_V1.4_BREADTH5",

        "generated_at_utc":
            utc_now(),

        "test_period": {
            "start":
                str(
                    common_dates[0].date()
                ),

            "end":
                str(
                    common_dates[-1].date()
                ),

            "years_requested":
                years,

            "common_trading_days":
                len(common_dates),
        },

        "frozen_settings": {
            "portfolio":
                "TOP_2_EQUAL_WEIGHT",

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

            "confirmation_days":
                V14_CONFIRMATION_DAYS,

            "transaction_cost_pct_each_side":
                transaction_cost_pct,

            "ranking":
                "SCORE, MOM120, MOM60, MOM20",

            "base_entry":
                "AL_ADAYI and score >= 75 for 3 consecutive closes",

            "exit":
                "V1.2A_SLOW_EXIT",

            "rotation":
                "V1.2C_STICKY_ROTATION",

            "execution":
                "Signal at close, execute next trading day open",

            "long_only":
                True,

            "short":
                False,

            "leverage":
                False,

            "automatic_orders":
                False,
        },

        "experimental_change": {
            "name":
                "GLOBAL_BREADTH_REGIME_FILTER",

            "definition":
                "Number of universe assets with Close > EMA200",

            "minimum_required":
                breadth_min,

            "universe_size":
                len(ASSETS),

            "rule":
                f"New entries allowed only when breadth >= {breadth_min}/{len(ASSETS)}",

            "applies_to":
                "NEW ENTRIES ONLY",

            "existing_holdings":
                "UNCHANGED; exit only by V1.2A slow exit",

            "parameter_optimization":
                False,
        },

        "benchmark": {
            "symbol":
                "DBC",

            "buy_hold_return_pct":
                safe_float(
                    dbc_return,
                    2,
                ),
        },

        "summary_comparison": {
            "v13_return_pct":
                p13[
                    "strategy_total_return_pct"
                ],

            "v14_return_pct":
                p14[
                    "strategy_total_return_pct"
                ],

            "return_difference_pct_points":
                safe_float(
                    (
                        p14[
                            "strategy_total_return_pct"
                        ]
                        -
                        p13[
                            "strategy_total_return_pct"
                        ]
                    ),
                    2,
                ),

            "v13_cagr_pct":
                p13[
                    "strategy_cagr_pct"
                ],

            "v14_cagr_pct":
                p14[
                    "strategy_cagr_pct"
                ],

            "v13_max_dd_pct":
                p13[
                    "max_drawdown_daily_pct"
                ],

            "v14_max_dd_pct":
                p14[
                    "max_drawdown_daily_pct"
                ],

            "v13_profit_factor":
                p13[
                    "profit_factor"
                ],

            "v14_profit_factor":
                p14[
                    "profit_factor"
                ],

            "v13_trade_count":
                p13[
                    "trade_count"
                ],

            "v14_trade_count":
                p14[
                    "trade_count"
                ],

            "v13_cash_pct":
                p13[
                    "cash_time_pct"
                ],

            "v14_cash_pct":
                p14[
                    "cash_time_pct"
                ],

            "v14_blocked_entry_days":
                v14[
                    "breadth_diagnostics"
                ][
                    "blocked_entry_days"
                ],
        },

        "v13_baseline":
            v13,

        "v14_experiment":
            v14,

        "data_errors":
            errors,
    }


# ============================================================
# CHRONOLOGICAL OOS COMPARISON
#
# IMPORTANT:
# - No fitting
# - No optimization
# - Same folds for V1.3 and V1.4
# - Each fold starts from CASH
# - One previous trading day is supplied so the signal can
#   execute at the first OOS open.
# ============================================================

def run_v14_oos_comparison(
    years=10,
    folds=4,
    initial_train_pct=50,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
    breadth_min=5,
):
    (
        prepared,
        common_dates,
        errors,
    ) = prepare_rotation_data(
        years
    )

    n = len(common_dates)

    if n < 500:
        raise ValueError(
            "OOS için yeterli ortak tarih yok."
        )

    folds = int(folds)

    if folds < 2:
        raise ValueError(
            "folds en az 2 olmalı."
        )

    initial_train_pct = int(
        initial_train_pct
    )

    if not (
        30
        <=
        initial_train_pct
        <=
        80
    ):
        raise ValueError(
            "initial_train_pct 30-80 arasında olmalı."
        )

    train_end = int(
        n
        *
        initial_train_pct
        /
        100
    )

    train_end = max(
        250,
        train_end,
    )

    minimum_oos_days = (
        folds
        *
        40
    )

    if (
        n
        -
        train_end
        <
        minimum_oos_days
    ):
        train_end = (
            n
            -
            minimum_oos_days
        )

    if train_end < 250:
        raise ValueError(
            "Train/OOS bölünmesi için yeterli veri yok."
        )

    remaining = (
        n
        -
        train_end
    )

    if remaining < folds:
        raise ValueError(
            "OOS fold oluşturmak için yeterli veri yok."
        )

    # --------------------------------------------------------
    # BUILD FOLD SIZES
    # --------------------------------------------------------

    base_fold_size = (
        remaining
        //
        folds
    )

    remainder = (
        remaining
        %
        folds
    )

    fold_sizes = []

    for fold_idx in range(folds):

        size = base_fold_size

        if fold_idx < remainder:
            size += 1

        fold_sizes.append(
            size
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    fold_results = []

    v13_compounded_equity = 1.0
    v14_compounded_equity = 1.0

    v13_positive_folds = 0
    v14_positive_folds = 0

    v13_pf_gt_1_folds = 0
    v14_pf_gt_1_folds = 0

    v13_total_trades = 0
    v14_total_trades = 0

    v13_total_rotations = 0
    v14_total_rotations = 0

    cursor = train_end

    for fold_number in range(
        1,
        folds + 1
    ):
        fold_size = (
            fold_sizes[
                fold_number - 1
            ]
        )

        oos_start_idx = cursor

        oos_end_idx = min(
            n,
            cursor
            +
            fold_size,
        )

        if (
            oos_end_idx
            <=
            oos_start_idx
        ):
            break

        # ----------------------------------------------------
        # Include ONE prior date only to generate signal
        # for first OOS trading day.
        #
        # Portfolio starts CASH.
        # ----------------------------------------------------

        slice_start_idx = max(
            0,
            oos_start_idx - 1,
        )

        fold_dates = (
            common_dates[
                slice_start_idx:
                oos_end_idx
            ]
        )

        actual_oos_dates = (
            common_dates[
                oos_start_idx:
                oos_end_idx
            ]
        )

        if len(fold_dates) < 2:
            cursor = oos_end_idx
            continue

        # ----------------------------------------------------
        # V1.3 BASELINE
        # ----------------------------------------------------

        v13 = run_v14_rotation_variant(
            prepared=prepared,
            common_dates=fold_dates,
            top_n=2,
            entry_score=entry_score,
            exit_score=exit_score,
            transaction_cost_pct=transaction_cost_pct,
            mode="V13",
            breadth_min=breadth_min,
        )

        # ----------------------------------------------------
        # V1.4 EXPERIMENT
        # ----------------------------------------------------

        v14 = run_v14_rotation_variant(
            prepared=prepared,
            common_dates=fold_dates,
            top_n=2,
            entry_score=entry_score,
            exit_score=exit_score,
            transaction_cost_pct=transaction_cost_pct,
            mode="V14",
            breadth_min=breadth_min,
        )

        p13 = v13["performance"]
        p14 = v14["performance"]

        r13 = (
            p13[
                "strategy_total_return_pct"
            ]
            /
            100
        )

        r14 = (
            p14[
                "strategy_total_return_pct"
            ]
            /
            100
        )

        v13_compounded_equity *= (
            1 + r13
        )

        v14_compounded_equity *= (
            1 + r14
        )

        if r13 > 0:
            v13_positive_folds += 1

        if r14 > 0:
            v14_positive_folds += 1

        pf13 = p13[
            "profit_factor"
        ]

        pf14 = p14[
            "profit_factor"
        ]

        if (
            pf13 is not None
            and
            pf13 > 1
        ):
            v13_pf_gt_1_folds += 1

        if (
            pf14 is not None
            and
            pf14 > 1
        ):
            v14_pf_gt_1_folds += 1

        v13_total_trades += (
            p13[
                "trade_count"
            ]
        )

        v14_total_trades += (
            p14[
                "trade_count"
            ]
        )

        v13_total_rotations += (
            p13[
                "rotation_count"
            ]
        )

        v14_total_rotations += (
            p14[
                "rotation_count"
            ]
        )

        fold_results.append({
            "fold":
                fold_number,

            "train_period": {
                "start":
                    str(
                        common_dates[0].date()
                    ),

                "end":
                    str(
                        common_dates[
                            oos_start_idx - 1
                        ].date()
                    ),

                "trading_days":
                    oos_start_idx,
            },

            "oos_period": {
                "start":
                    str(
                        actual_oos_dates[0].date()
                    ),

                "end":
                    str(
                        actual_oos_dates[-1].date()
                    ),

                "trading_days":
                    len(
                        actual_oos_dates
                    ),
            },

            "v13_baseline":
                v14_compact_result(
                    v13
                ),

            "v14_experiment":
                v14_compact_result(
                    v14
                ),

            "comparison": {
                "v13_return_pct":
                    p13[
                        "strategy_total_return_pct"
                    ],

                "v14_return_pct":
                    p14[
                        "strategy_total_return_pct"
                    ],

                "return_difference_pct_points":
                    safe_float(
                        (
                            p14[
                                "strategy_total_return_pct"
                            ]
                            -
                            p13[
                                "strategy_total_return_pct"
                            ]
                        ),
                        2,
                    ),

                "v13_max_dd_pct":
                    p13[
                        "max_drawdown_daily_pct"
                    ],

                "v14_max_dd_pct":
                    p14[
                        "max_drawdown_daily_pct"
                    ],

                "v13_profit_factor":
                    p13[
                        "profit_factor"
                    ],

                "v14_profit_factor":
                    p14[
                        "profit_factor"
                    ],

                "v13_trade_count":
                    p13[
                        "trade_count"
                    ],

                "v14_trade_count":
                    p14[
                        "trade_count"
                    ],

                "v14_blocked_entry_days":
                    v14[
                        "breadth_diagnostics"
                    ][
                        "blocked_entry_days"
                    ],
            },
        })

        cursor = oos_end_idx

    # ========================================================
    # AGGREGATE OOS
    # ========================================================

    actual_fold_count = len(
        fold_results
    )

    if actual_fold_count == 0:
        raise ValueError(
            "OOS fold üretilemedi."
        )

    first_oos_date = (
        common_dates[
            train_end
        ]
    )

    last_oos_date = (
        common_dates[-1]
    )

    oos_elapsed_years = (
        (
            last_oos_date
            -
            first_oos_date
        ).days
        /
        365.25
    )

    v13_compounded_return = (
        v13_compounded_equity
        -
        1
    ) * 100

    v14_compounded_return = (
        v14_compounded_equity
        -
        1
    ) * 100

    if oos_elapsed_years > 0:

        v13_cagr = (
            v13_compounded_equity
            **
            (
                1
                /
                oos_elapsed_years
            )
            -
            1
        ) * 100

        v14_cagr = (
            v14_compounded_equity
            **
            (
                1
                /
                oos_elapsed_years
            )
            -
            1
        ) * 100

    else:

        v13_cagr = None
        v14_cagr = None

    return {
        "experiment":
            "V1.3_CONFIRM3_VS_V1.4_BREADTH5_OOS",

        "generated_at_utc":
            utc_now(),

        "full_period": {
            "start":
                str(
                    common_dates[0].date()
                ),

            "end":
                str(
                    common_dates[-1].date()
                ),

            "years_requested":
                years,

            "common_trading_days":
                n,
        },

        "walk_forward": {
            "initial_train_pct":
                initial_train_pct,

            "requested_folds":
                folds,

            "actual_folds":
                actual_fold_count,

            "rule":
                (
                    "Expanding chronological history; "
                    "each OOS fold starts from CASH; "
                    "no fitting or parameter changes"
                ),

            "oos_start":
                str(
                    first_oos_date.date()
                ),

            "oos_end":
                str(
                    last_oos_date.date()
                ),
        },

        "frozen_settings": {
            "portfolio":
                "TOP_2_EQUAL_WEIGHT",

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

            "confirmation_days":
                V14_CONFIRMATION_DAYS,

            "transaction_cost_pct_each_side":
                transaction_cost_pct,

            "ranking":
                "SCORE, MOM120, MOM60, MOM20",

            "exit":
                "V1.2A_SLOW_EXIT",

            "sticky_rotation":
                True,

            "signal_execution":
                "Signal close -> next trading day open",

            "parameter_optimization":
                False,
        },

        "experimental_change": {
            "v13":
                "CONFIRM3 only",

            "v14":
                (
                    "CONFIRM3 + global breadth >= "
                    f"{breadth_min}/{len(ASSETS)}"
                ),

            "breadth_definition":
                "Universe assets with Close > EMA200",

            "applies_to":
                "NEW ENTRIES ONLY",

            "existing_positions":
                "Not affected by breadth",
        },

        "aggregate_oos": {
            "v13": {
                "positive_fold_count":
                    v13_positive_folds,

                "pf_gt_1_fold_count":
                    v13_pf_gt_1_folds,

                "compounded_return_pct":
                    safe_float(
                        v13_compounded_return,
                        2,
                    ),

                "compounded_cagr_pct":
                    safe_float(
                        v13_cagr,
                        2,
                    ),

                "final_equity":
                    safe_float(
                        v13_compounded_equity,
                        6,
                    ),

                "trade_count":
                    v13_total_trades,

                "rotation_count":
                    v13_total_rotations,
            },

            "v14": {
                "positive_fold_count":
                    v14_positive_folds,

                "pf_gt_1_fold_count":
                    v14_pf_gt_1_folds,

                "compounded_return_pct":
                    safe_float(
                        v14_compounded_return,
                        2,
                    ),

                "compounded_cagr_pct":
                    safe_float(
                        v14_cagr,
                        2,
                    ),

                "final_equity":
                    safe_float(
                        v14_compounded_equity,
                        6,
                    ),

                "trade_count":
                    v14_total_trades,

                "rotation_count":
                    v14_total_rotations,
            },

            "difference": {
                "return_pct_points":
                    safe_float(
                        (
                            v14_compounded_return
                            -
                            v13_compounded_return
                        ),
                        2,
                    ),

                "cagr_pct_points":
                    safe_float(
                        (
                            v14_cagr
                            -
                            v13_cagr
                        )
                        if (
                            v14_cagr is not None
                            and
                            v13_cagr is not None
                        )
                        else None,
                        2,
                    ),

                "positive_folds_change":
                    (
                        v14_positive_folds
                        -
                        v13_positive_folds
                    ),

                "pf_gt_1_folds_change":
                    (
                        v14_pf_gt_1_folds
                        -
                        v13_pf_gt_1_folds
                    ),

                "trade_count_change":
                    (
                        v14_total_trades
                        -
                        v13_total_trades
                    ),
            },
        },

        "folds":
            fold_results,

        "data_errors":
            errors,

        "research_warning":
            (
                "V1.4 breadth filter was designed after reviewing "
                "earlier V1.2D/V1.3 results. Therefore these historical "
                "folds are research evidence, not pristine untouched OOS."
            ),
    }


# ============================================================
# FASTAPI ENDPOINT
# CONTINUOUS 10Y COMPARISON
#
# Example:
# /v14-comparison?years=10
# ============================================================

@app.get("/v14-comparison")
def v14_comparison_endpoint(
    years: int = 10,
):
    return run_v14_comparison(
        years=years,
        entry_score=75,
        exit_score=50,
        transaction_cost_pct=0.10,
        breadth_min=5,
    )


# ============================================================
# FASTAPI ENDPOINT
# SAME 4-FOLD OOS COMPARISON
#
# Example:
# /v14-oos?years=10&folds=4&initial_train_pct=50
# ============================================================

@app.get("/v14-oos")
def v14_oos_endpoint(
    years: int = 10,
    folds: int = 4,
    initial_train_pct: int = 50,
):
    return run_v14_oos_comparison(
        years=years,
        folds=folds,
        initial_train_pct=initial_train_pct,
        entry_score=75,
        exit_score=50,
        transaction_cost_pct=0.10,
        breadth_min=5,
    )

# ============================================================
# V1.3 DIAGNOSTIC ANALYSIS
#
# PURPOSE:
# - DO NOT change strategy
# - DO NOT change entries/exits
# - DO NOT optimize parameters
#
# Analyze entry characteristics of V1.3 CONFIRM3 trades.
#
# Main goal:
# Compare WINNERS vs LOSERS
# especially inside chronological OOS Fold-2.
# ============================================================


V13_DIAGNOSTIC_NAME = "V1.3-CONFIRM3-ENTRY-DIAGNOSTIC"


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def diag_float(value, digits=4):

    try:

        value = float(value)

        if np.isnan(value):
            return None

        if np.isinf(value):
            return None

        return round(
            value,
            digits,
        )

    except Exception:

        return None


def diag_mean(values):

    clean = [
        float(x)
        for x in values
        if x is not None
        and not pd.isna(x)
    ]

    if not clean:
        return None

    return round(
        float(
            np.mean(clean)
        ),
        4,
    )


def diag_median(values):

    clean = [
        float(x)
        for x in values
        if x is not None
        and not pd.isna(x)
    ]

    if not clean:
        return None

    return round(
        float(
            np.median(clean)
        ),
        4,
    )


# ============================================================
# ENTRY FEATURE EXTRACTOR
#
# IMPORTANT:
# trade["entry_date"] is the execution day.
#
# Our system:
# signal at CLOSE
# execution NEXT OPEN
#
# Therefore entry indicators MUST come from the previous
# common trading day.
#
# This prevents accidental look-ahead.
# ============================================================

def diagnostic_entry_features(
    prepared,
    common_dates,
    trade,
):

    symbol = trade["symbol"]

    if symbol not in prepared:
        return None

    df = prepared[symbol]

    entry_date = pd.Timestamp(
        trade["entry_date"]
    )

    # --------------------------------------------------------
    # Find execution date inside common trading calendar
    # --------------------------------------------------------

    try:

        execution_pos = (
            common_dates.index(
                entry_date
            )
        )

    except ValueError:

        # Timestamp timezone / normalization fallback

        execution_pos = None

        for i, dt in enumerate(
            common_dates
        ):

            if (
                pd.Timestamp(dt).date()
                ==
                entry_date.date()
            ):

                execution_pos = i
                break

        if execution_pos is None:
            return None

    # Need previous trading day for signal
    if execution_pos <= 0:
        return None

    signal_date = (
        common_dates[
            execution_pos - 1
        ]
    )

    if signal_date not in df.index:
        return None

    row = df.loc[
        signal_date
    ]

    # --------------------------------------------------------
    # RAW VALUES
    # --------------------------------------------------------

    close = float(
        row["Close"]
    )

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    ema100 = float(
        row["EMA100"]
    )

    ema200 = float(
        row["EMA200"]
    )

    # --------------------------------------------------------
    # EMA SPREADS
    #
    # These are diagnostic only.
    # They are NOT new trading rules.
    # --------------------------------------------------------

    ema20_vs_ema50_pct = (
        (
            ema20
            /
            ema50
        )
        -
        1
    ) * 100

    ema50_vs_ema200_pct = (
        (
            ema50
            /
            ema200
        )
        -
        1
    ) * 100

    ema20_vs_ema200_pct = (
        (
            ema20
            /
            ema200
        )
        -
        1
    ) * 100

    # EMA100 distance is not stored directly in base code,
    # so calculate it here.

    dist_ema100_pct = (
        (
            close
            /
            ema100
        )
        -
        1
    ) * 100

    trade_return = diag_float(
        trade.get(
            "return_pct"
        ),
        4,
    )

    if (
        trade_return is not None
        and
        trade_return > 0
    ):

        outcome = "WINNER"

    else:

        outcome = "LOSER"

    return {

        "symbol":
            symbol,

        "signal_date":
            str(
                pd.Timestamp(
                    signal_date
                ).date()
            ),

        "entry_date":
            str(
                entry_date.date()
            ),

        "exit_date":
            trade.get(
                "exit_date"
            ),

        "outcome":
            outcome,

        "return_pct":
            trade_return,

        "holding_days":
            trade.get(
                "holding_days"
            ),

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        "score":
            diag_float(
                row["SCORE"],
                2,
            ),

        # ----------------------------------------------------
        # MOMENTUM
        # ----------------------------------------------------

        "mom20":
            diag_float(
                row["MOM20"],
                4,
            ),

        "mom60":
            diag_float(
                row["MOM60"],
                4,
            ),

        "mom120":
            diag_float(
                row["MOM120"],
                4,
            ),

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        "rsi14":
            diag_float(
                row["RSI14"],
                4,
            ),

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        "macd":
            diag_float(
                row["MACD"],
                6,
            ),

        "macd_signal":
            diag_float(
                row["MACD_SIGNAL"],
                6,
            ),

        "macd_hist":
            diag_float(
                row["MACD_HIST"],
                6,
            ),

        # ----------------------------------------------------
        # VOLATILITY
        # ----------------------------------------------------

        "atr14":
            diag_float(
                row["ATR14"],
                6,
            ),

        "atr_pct":
            diag_float(
                row["ATR_PCT"],
                4,
            ),

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        "volume_ratio":
            diag_float(
                row["VOLUME_RATIO"],
                4,
            ),

        # ----------------------------------------------------
        # PRICE / EMA DISTANCES
        # ----------------------------------------------------

        "dist_ema20_pct":
            diag_float(
                row["DIST_EMA20"],
                4,
            ),

        "dist_ema50_pct":
            diag_float(
                row["DIST_EMA50"],
                4,
            ),

        "dist_ema100_pct":
            diag_float(
                dist_ema100_pct,
                4,
            ),

        "dist_ema200_pct":
            diag_float(
                row["DIST_EMA200"],
                4,
            ),

        # ----------------------------------------------------
        # EMA STRUCTURE
        # ----------------------------------------------------

        "ema20_vs_ema50_pct":
            diag_float(
                ema20_vs_ema50_pct,
                4,
            ),

        "ema50_vs_ema200_pct":
            diag_float(
                ema50_vs_ema200_pct,
                4,
            ),

        "ema20_vs_ema200_pct":
            diag_float(
                ema20_vs_ema200_pct,
                4,
            ),

        # ----------------------------------------------------
        # 1Y DRAWDOWN
        # ----------------------------------------------------

        "drawdown_252_pct":
            diag_float(
                row["DRAWDOWN_252"],
                4,
            ),

        # ----------------------------------------------------
        # EXISTING DIAGNOSTIC
        # ----------------------------------------------------

        "entry_breadth":
            trade.get(
                "entry_breadth"
            ),
    }


# ============================================================
# GROUP SUMMARY
# ============================================================

DIAGNOSTIC_FIELDS = [

    "score",

    "mom20",
    "mom60",
    "mom120",

    "rsi14",

    "macd_hist",

    "atr_pct",

    "volume_ratio",

    "dist_ema20_pct",
    "dist_ema50_pct",
    "dist_ema100_pct",
    "dist_ema200_pct",

    "ema20_vs_ema50_pct",
    "ema50_vs_ema200_pct",
    "ema20_vs_ema200_pct",

    "drawdown_252_pct",

    "entry_breadth",

    "holding_days",

    "return_pct",
]


def diagnostic_group_summary(
    trades,
):

    summary = {
        "count":
            len(trades),
    }

    if not trades:
        return summary

    for field in DIAGNOSTIC_FIELDS:

        values = [
            x.get(field)
            for x in trades
        ]

        summary[field] = {

            "mean":
                diag_mean(
                    values
                ),

            "median":
                diag_median(
                    values
                ),
        }

    return summary


# ============================================================
# WINNER VS LOSER DIFFERENCE
#
# Positive value:
# winner mean > loser mean
#
# Negative value:
# winner mean < loser mean
#
# IMPORTANT:
# This is descriptive only.
# It does NOT create a threshold.
# ============================================================

def diagnostic_winner_loser_difference(
    winners,
    losers,
):

    result = {}

    for field in DIAGNOSTIC_FIELDS:

        winner_values = [
            x.get(field)
            for x in winners
        ]

        loser_values = [
            x.get(field)
            for x in losers
        ]

        winner_mean = (
            diag_mean(
                winner_values
            )
        )

        loser_mean = (
            diag_mean(
                loser_values
            )
        )

        if (
            winner_mean is not None
            and
            loser_mean is not None
        ):

            difference = round(
                winner_mean
                -
                loser_mean,
                4,
            )

        else:

            difference = None

        result[field] = {

            "winner_mean":
                winner_mean,

            "loser_mean":
                loser_mean,

            "winner_minus_loser":
                difference,
        }

    return result


# ============================================================
# SYMBOL SUMMARY
#
# Diagnostic only.
# We are NOT using this to exclude assets.
# ============================================================

def diagnostic_symbol_summary(
    trades,
):

    symbols = {}

    for trade in trades:

        symbol = trade["symbol"]

        if symbol not in symbols:

            symbols[symbol] = {
                "trade_count": 0,
                "winner_count": 0,
                "loser_count": 0,
                "returns": [],
            }

        item = symbols[
            symbol
        ]

        item[
            "trade_count"
        ] += 1

        return_pct = (
            trade.get(
                "return_pct"
            )
        )

        if return_pct is not None:

            item[
                "returns"
            ].append(
                return_pct
            )

            if return_pct > 0:

                item[
                    "winner_count"
                ] += 1

            else:

                item[
                    "loser_count"
                ] += 1

    output = {}

    for symbol, item in symbols.items():

        returns = (
            item.pop(
                "returns"
            )
        )

        output[
            symbol
        ] = {

            **item,

            "win_rate_pct":
                round(
                    (
                        item[
                            "winner_count"
                        ]
                        /
                        item[
                            "trade_count"
                        ]
                        *
                        100
                    ),
                    2,
                )
                if item[
                    "trade_count"
                ]
                else 0,

            "average_return_pct":
                diag_mean(
                    returns
                ),

            "median_return_pct":
                diag_median(
                    returns
                ),

            "total_trade_return_pct":
                diag_float(
                    sum(
                        returns
                    ),
                    4,
                )
                if returns
                else 0,
        }

    return output


# ============================================================
# ANALYZE ONE FOLD
# ============================================================

def analyze_v13_diagnostic_fold(
    prepared,
    full_common_dates,
    fold_dates,
    fold_number,
    actual_oos_dates,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
):

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Use V1.3 baseline logic already used in V1.4 comparison.
    #
    # mode="V13" means:
    # CONFIRM3
    # no breadth gate
    # --------------------------------------------------------

    result = (
        run_v14_rotation_variant(
            prepared=prepared,
            common_dates=fold_dates,
            top_n=2,
            entry_score=entry_score,
            exit_score=exit_score,
            transaction_cost_pct=transaction_cost_pct,
            mode="V13",
            breadth_min=5,
        )
    )

    enriched = []

    for trade in result["trades"]:

        features = (
            diagnostic_entry_features(
                prepared=prepared,
                common_dates=full_common_dates,
                trade=trade,
            )
        )

        if features is not None:

            enriched.append(
                features
            )

    winners = [
        x
        for x in enriched
        if x[
            "outcome"
        ] == "WINNER"
    ]

    losers = [
        x
        for x in enriched
        if x[
            "outcome"
        ] == "LOSER"
    ]

    return {

        "fold":
            fold_number,

        "oos_period": {

            "start":
                str(
                    actual_oos_dates[
                        0
                    ].date()
                ),

            "end":
                str(
                    actual_oos_dates[
                        -1
                    ].date()
                ),

            "trading_days":
                len(
                    actual_oos_dates
                ),
        },

        "performance":
            result[
                "performance"
            ],

        "diagnostic_trade_count":
            len(
                enriched
            ),

        "winner_count":
            len(
                winners
            ),

        "loser_count":
            len(
                losers
            ),

        "winner_summary":
            diagnostic_group_summary(
                winners
            ),

        "loser_summary":
            diagnostic_group_summary(
                losers
            ),

        "winner_vs_loser":
            diagnostic_winner_loser_difference(
                winners,
                losers,
            ),

        "symbol_summary":
            diagnostic_symbol_summary(
                enriched
            ),

        "trades":
            enriched,
    }


# ============================================================
# ALL 4 OOS FOLDS
#
# Same chronological structure:
#
# years = 10
# initial train = 50%
# folds = 4
#
# No fitting.
# No optimization.
# Each OOS fold starts CASH.
# ============================================================

def run_v13_entry_diagnostic(
    years=10,
    folds=4,
    initial_train_pct=50,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
):

    (
        prepared,
        common_dates,
        errors,
    ) = prepare_rotation_data(
        years
    )

    n = len(
        common_dates
    )

    if n < 500:

        raise ValueError(
            "Diagnostic OOS için yeterli veri yok."
        )

    folds = int(
        folds
    )

    if folds < 2:

        raise ValueError(
            "folds en az 2 olmalı."
        )

    initial_train_pct = int(
        initial_train_pct
    )

    if not (
        30
        <=
        initial_train_pct
        <=
        80
    ):

        raise ValueError(
            "initial_train_pct 30-80 arasında olmalı."
        )

    train_end = int(
        n
        *
        initial_train_pct
        /
        100
    )

    train_end = max(
        250,
        train_end,
    )

    minimum_oos_days = (
        folds
        *
        40
    )

    if (
        n
        -
        train_end
        <
        minimum_oos_days
    ):

        train_end = (
            n
            -
            minimum_oos_days
        )

    if train_end < 250:

        raise ValueError(
            "Train/OOS bölünmesi için yeterli veri yok."
        )

    remaining = (
        n
        -
        train_end
    )

    base_fold_size = (
        remaining
        //
        folds
    )

    remainder = (
        remaining
        %
        folds
    )

    fold_sizes = []

    for fold_idx in range(
        folds
    ):

        size = (
            base_fold_size
        )

        if fold_idx < remainder:

            size += 1

        fold_sizes.append(
            size
        )

    fold_results = []

    cursor = (
        train_end
    )

    for fold_number in range(
        1,
        folds + 1
    ):

        fold_size = (
            fold_sizes[
                fold_number - 1
            ]
        )

        oos_start_idx = (
            cursor
        )

        oos_end_idx = min(
            n,
            cursor
            +
            fold_size,
        )

        if (
            oos_end_idx
            <=
            oos_start_idx
        ):

            break

        # ----------------------------------------------------
        # One previous trading day included so the signal
        # can execute at first OOS open.
        # ----------------------------------------------------

        slice_start_idx = max(
            0,
            oos_start_idx - 1,
        )

        fold_dates = (
            common_dates[
                slice_start_idx:
                oos_end_idx
            ]
        )

        actual_oos_dates = (
            common_dates[
                oos_start_idx:
                oos_end_idx
            ]
        )

        if len(
            actual_oos_dates
        ) == 0:

            cursor = (
                oos_end_idx
            )

            continue

        fold_result = (
            analyze_v13_diagnostic_fold(
                prepared=prepared,
                full_common_dates=common_dates,
                fold_dates=fold_dates,
                fold_number=fold_number,
                actual_oos_dates=actual_oos_dates,
                entry_score=entry_score,
                exit_score=exit_score,
                transaction_cost_pct=transaction_cost_pct,
            )
        )

        fold_results.append(
            fold_result
        )

        cursor = (
            oos_end_idx
        )

    # ========================================================
    # CROSS-FOLD SUMMARY
    #
    # Helps determine whether a feature seen in bad Fold-2
    # also exists in successful folds.
    # ========================================================

    cross_fold = {}

    for field in DIAGNOSTIC_FIELDS:

        cross_fold[
            field
        ] = []

        for fold in fold_results:

            comparison = (
                fold[
                    "winner_vs_loser"
                ].get(
                    field,
                    {}
                )
            )

            cross_fold[
                field
            ].append({

                "fold":
                    fold[
                        "fold"
                    ],

                "winner_mean":
                    comparison.get(
                        "winner_mean"
                    ),

                "loser_mean":
                    comparison.get(
                        "loser_mean"
                    ),

                "winner_minus_loser":
                    comparison.get(
                        "winner_minus_loser"
                    ),
            })

    return {

        "diagnostic":
            V13_DIAGNOSTIC_NAME,

        "generated_at_utc":
            utc_now(),

        "purpose":
            (
                "Describe entry characteristics of "
                "V1.3 CONFIRM3 winners and losers. "
                "No strategy rules are changed."
            ),

        "methodology": {

            "model":
                "V1.3_CONFIRM3",

            "portfolio":
                "TOP_2_EQUAL_WEIGHT",

            "years":
                years,

            "folds":
                folds,

            "initial_train_pct":
                initial_train_pct,

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

            "transaction_cost_pct_each_side":
                transaction_cost_pct,

            "entry_features_from":
                (
                    "Signal close immediately before "
                    "next-open execution"
                ),

            "parameter_optimization":
                False,

            "new_trading_filter":
                False,

            "strategy_modified":
                False,
        },

        "full_period": {

            "start":
                str(
                    common_dates[
                        0
                    ].date()
                ),

            "end":
                str(
                    common_dates[
                        -1
                    ].date()
                ),

            "common_trading_days":
                n,
        },

        "folds":
            fold_results,

        "cross_fold_feature_comparison":
            cross_fold,

        "data_errors":
            errors,

        "interpretation_warning":
            (
                "This endpoint is descriptive research only. "
                "Do not convert a Fold-2 difference directly "
                "into a new threshold without checking whether "
                "the same relationship is stable across the "
                "other folds."
            ),
    }


# ============================================================
# FOLD-2 ONLY
#
# Smaller output.
# Start with this endpoint.
# ============================================================

@app.get(
    "/v13-diagnostic-fold2"
)
def v13_diagnostic_fold2_endpoint():

    result = (
        run_v13_entry_diagnostic(
            years=10,
            folds=4,
            initial_train_pct=50,
            entry_score=75,
            exit_score=50,
            transaction_cost_pct=0.10,
        )
    )

    fold2 = None

    for fold in result[
        "folds"
    ]:

        if fold[
            "fold"
        ] == 2:

            fold2 = fold
            break

    if fold2 is None:

        raise ValueError(
            "Fold-2 bulunamadı."
        )

    return {

        "diagnostic":
            result[
                "diagnostic"
            ],

        "methodology":
            result[
                "methodology"
            ],

        "fold2":
            fold2,

        "interpretation_warning":
            result[
                "interpretation_warning"
            ],
    }


# ============================================================
# ALL FOLDS
#
# Run AFTER Fold-2 result is checked.
# ============================================================

@app.get(
    "/v13-diagnostic-all-folds"
)
def v13_diagnostic_all_folds_endpoint():

    return (
        run_v13_entry_diagnostic(
            years=10,
            folds=4,
            initial_train_pct=50,
            entry_score=75,
            exit_score=50,
            transaction_cost_pct=0.10,
        )
    )

# ============================================================
# COMMODITY DATA HISTORY DIAGNOSTIC
#
# PURPOSE:
# - Find earliest available history for every asset
# - Find common start date
# - Check whether pre-2016 independent testing is possible
#
# NO STRATEGY RULES ARE CHANGED.
# ============================================================


@app.get("/data-history")
def commodity_data_history():

    history_results = []

    valid_series = {}

    # --------------------------------------------------------
    # Use the SAME ASSET universe already defined in main.py
    # --------------------------------------------------------

    for symbol in ASSETS.keys():

        try:

            # ------------------------------------------------
            # Download maximum available history
            # ------------------------------------------------

            df = yf.download(
                symbol,
                period="max",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:

                history_results.append({
                    "symbol": symbol,
                    "status": "NO_DATA",
                })

                continue

            # ------------------------------------------------
            # Handle possible yfinance MultiIndex columns
            # ------------------------------------------------

            if isinstance(
                df.columns,
                pd.MultiIndex,
            ):

                df.columns = [
                    col[0]
                    if isinstance(col, tuple)
                    else col
                    for col in df.columns
                ]

            # ------------------------------------------------
            # Keep valid close prices only
            # ------------------------------------------------

            if "Close" not in df.columns:

                history_results.append({
                    "symbol": symbol,
                    "status": "NO_CLOSE_COLUMN",
                })

                continue

            df = df.dropna(
                subset=["Close"]
            )

            if df.empty:

                history_results.append({
                    "symbol": symbol,
                    "status": "NO_VALID_CLOSE",
                })

                continue

            # ------------------------------------------------
            # Normalize index
            # ------------------------------------------------

            df.index = pd.to_datetime(
                df.index
            )

            # Remove timezone if present

            try:

                if df.index.tz is not None:

                    df.index = (
                        df.index.tz_localize(
                            None
                        )
                    )

            except Exception:

                pass

            first_date = (
                df.index.min()
            )

            last_date = (
                df.index.max()
            )

            trading_days = len(
                df
            )

            years_available = (
                (
                    last_date
                    -
                    first_date
                ).days
                /
                365.25
            )

            valid_series[
                symbol
            ] = df

            # ------------------------------------------------
            # Count observations before 2016
            # ------------------------------------------------

            pre_2016 = df[
                df.index
                <
                pd.Timestamp(
                    "2016-01-01"
                )
            ]

            pre_2016_days = len(
                pre_2016
            )

            # ------------------------------------------------
            # Count observations before Sep-2016
            #
            # Current 10y test starts around Sep-2016.
            # ------------------------------------------------

            pre_current_test = df[
                df.index
                <
                pd.Timestamp(
                    "2016-09-26"
                )
            ]

            pre_current_test_days = len(
                pre_current_test
            )

            history_results.append({

                "symbol":
                    symbol,

                "status":
                    "OK",

                "first_date":
                    str(
                        first_date.date()
                    ),

                "last_date":
                    str(
                        last_date.date()
                    ),

                "trading_days":
                    trading_days,

                "years_available":
                    round(
                        years_available,
                        2,
                    ),

                "pre_2016_trading_days":
                    pre_2016_days,

                "pre_2016_09_26_trading_days":
                    pre_current_test_days,
            })

        except Exception as e:

            history_results.append({

                "symbol":
                    symbol,

                "status":
                    "ERROR",

                "error":
                    str(e),
            })

    # ========================================================
    # COMMON CALENDAR
    # ========================================================

    if not valid_series:

        return {

            "status":
                "NO_VALID_DATA",

            "assets":
                history_results,
        }

    # --------------------------------------------------------
    # Earliest date where EVERY asset exists
    # --------------------------------------------------------

    first_dates = {

        symbol:
            df.index.min()

        for symbol, df
        in valid_series.items()
    }

    last_dates = {

        symbol:
            df.index.max()

        for symbol, df
        in valid_series.items()
    }

    common_start = max(
        first_dates.values()
    )

    common_end = min(
        last_dates.values()
    )

    # --------------------------------------------------------
    # Exact intersection of trading dates
    # --------------------------------------------------------

    common_index = None

    for symbol, df in valid_series.items():

        idx = pd.DatetimeIndex(
            df.index
        )

        if common_index is None:

            common_index = idx

        else:

            common_index = (
                common_index.intersection(
                    idx
                )
            )

    common_index = (
        common_index.sort_values()
    )

    if len(
        common_index
    ) > 0:

        exact_common_first = (
            common_index[
                0
            ]
        )

        exact_common_last = (
            common_index[
                -1
            ]
        )

    else:

        exact_common_first = None
        exact_common_last = None

    # ========================================================
    # PRE-2016 COMMON DATA
    # ========================================================

    if common_index is not None:

        pre_2016_common = (
            common_index[
                common_index
                <
                pd.Timestamp(
                    "2016-01-01"
                )
            ]
        )

        pre_current_test_common = (
            common_index[
                common_index
                <
                pd.Timestamp(
                    "2016-09-26"
                )
            ]
        )

    else:

        pre_2016_common = []
        pre_current_test_common = []

    # ========================================================
    # ASSET THAT LIMITS HISTORY
    # ========================================================

    limiting_symbol = max(
        first_dates,
        key=first_dates.get,
    )

    limiting_date = (
        first_dates[
            limiting_symbol
        ]
    )

    # ========================================================
    # SIMPLE RESEARCH FEASIBILITY
    #
    # We are NOT creating a strategy rule here.
    #
    # Rough interpretation:
    #
    # >= 750 common trading days before current test
    #    ~ 3 years
    #
    # >= 1250
    #    ~ 5 years
    # ========================================================

    pre_test_days = len(
        pre_current_test_common
    )

    if pre_test_days >= 1250:

        feasibility = (
            "STRONG_PRE_2016_TEST_WINDOW"
        )

    elif pre_test_days >= 750:

        feasibility = (
            "USABLE_PRE_2016_TEST_WINDOW"
        )

    elif pre_test_days >= 500:

        feasibility = (
            "LIMITED_PRE_2016_TEST_WINDOW"
        )

    else:

        feasibility = (
            "INSUFFICIENT_PRE_2016_COMMON_HISTORY"
        )

    # ========================================================
    # SORT ASSETS BY START DATE
    # ========================================================

    history_results = sorted(
        history_results,
        key=lambda x: (
            x.get(
                "first_date",
                "9999-12-31",
            )
        ),
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "diagnostic":
            "COMMODITY-DATA-HISTORY-V1",

        "purpose":
            (
                "Check whether an independent historical "
                "period exists before the current "
                "2016-09-26 onward research window."
            ),

        "strategy_modified":
            False,

        "asset_count_expected":
            len(
                ASSETS
            ),

        "asset_count_with_data":
            len(
                valid_series
            ),

        "assets":
            history_results,

        "common_history": {

            "common_start_based_on_asset_availability":
                str(
                    common_start.date()
                ),

            "common_end_based_on_asset_availability":
                str(
                    common_end.date()
                ),

            "exact_common_first_trading_day":
                (
                    str(
                        exact_common_first.date()
                    )
                    if exact_common_first is not None
                    else None
                ),

            "exact_common_last_trading_day":
                (
                    str(
                        exact_common_last.date()
                    )
                    if exact_common_last is not None
                    else None
                ),

            "exact_common_trading_days":
                len(
                    common_index
                ),

            "common_days_before_2016_01_01":
                len(
                    pre_2016_common
                ),

            "common_days_before_current_test_start":
                pre_test_days,
        },

        "history_limiting_asset": {

            "symbol":
                limiting_symbol,

            "first_date":
                str(
                    limiting_date.date()
                ),
        },

        "pre_2016_research_feasibility":
            feasibility,

        "important_note":
            (
                "No V1.3 or V1.5 trading rule was executed. "
                "This endpoint only examines raw historical "
                "data availability."
            ),
    }

# ============================================================
# V1.3 PRE-2016 HISTORICAL HOLDOUT TEST
# ============================================================
#
# PURPOSE
# -------
# Test the frozen V1.3 CONFIRM3 strategy on historical data
# BEFORE the existing 2016-09-26 -> 2026-09-25 research window.
#
# IMPORTANT
# ---------
# - NO parameter optimization
# - NO V1.4 breadth filter
# - NO V1.5 overextension filter
# - SAME score model
# - SAME slow exit
# - SAME sticky Top-2 portfolio
# - SAME 3 consecutive AL_ADAYI confirmation
# - SAME next-open execution
# - SAME 0.10% transaction cost each side
#
# This is a HOLDOUT VALIDATION TEST.
# ============================================================


def v13_pre2016_prepare_data():

    prepared = {}
    errors = []

    # --------------------------------------------------------
    # Download MAX history.
    #
    # Indicators are calculated BEFORE the holdout window
    # is selected.
    #
    # This avoids artificially resetting EMA/MOM indicators
    # at the beginning of the holdout.
    # --------------------------------------------------------

    for symbol in ASSETS:

        try:

            df = yf.download(
                symbol,
                period="max",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:

                errors.append({
                    "symbol": symbol,
                    "error": "NO_DATA",
                })

                continue

            # ------------------------------------------------
            # yfinance can return MultiIndex columns
            # ------------------------------------------------

            if isinstance(
                df.columns,
                pd.MultiIndex,
            ):

                df.columns = [

                    col[0]
                    if isinstance(
                        col,
                        tuple,
                    )
                    else col

                    for col in df.columns
                ]

            # ------------------------------------------------
            # Normalize index
            # ------------------------------------------------

            df.index = pd.to_datetime(
                df.index
            )

            try:

                if df.index.tz is not None:

                    df.index = (
                        df.index.tz_localize(
                            None
                        )
                    )

            except Exception:

                pass

            df = df.sort_index()

            # ------------------------------------------------
            # Existing frozen indicator engine
            # ------------------------------------------------

            df = calculate_indicators(
                df
            )

            # ------------------------------------------------
            # Do NOT drop early rows until indicators exist.
            # Same principle as existing engine.
            # ------------------------------------------------

            df = df.dropna().copy()

            if df.empty:

                errors.append({
                    "symbol": symbol,
                    "error":
                        "NO_ROWS_AFTER_INDICATORS",
                })

                continue

            # ------------------------------------------------
            # Frozen score + signal model
            # ------------------------------------------------

            scores = []
            signals = []

            for _, row in df.iterrows():

                score, _, _ = (
                    calculate_score(
                        row
                    )
                )

                scores.append(
                    score
                )

                signals.append(

                    determine_signal(
                        row,
                        score,
                    )
                )

            df["SCORE"] = scores

            df["MODEL_SIGNAL"] = (
                signals
            )

            prepared[symbol] = df

        except Exception as exc:

            errors.append({
                "symbol":
                    symbol,

                "error":
                    str(exc),
            })

    if len(prepared) != len(ASSETS):

        raise ValueError(
            "Holdout testi için 10 varlığın tamamı "
            "hazırlanamadı. "
            f"Hazırlanan={len(prepared)}, "
            f"Beklenen={len(ASSETS)}, "
            f"Hatalar={errors}"
        )

    # ========================================================
    # EXACT COMMON CALENDAR
    # ========================================================

    common_dates = None

    for symbol, df in prepared.items():

        dates = set(
            df.index
        )

        if common_dates is None:

            common_dates = dates

        else:

            common_dates = (
                common_dates.intersection(
                    dates
                )
            )

    common_dates = sorted(
        common_dates
    )

    if not common_dates:

        raise ValueError(
            "Ortak işlem tarihi bulunamadı."
        )

    # ========================================================
    # HOLDOUT WINDOW
    #
    # Existing research begins:
    #
    # 2016-09-26
    #
    # Therefore holdout must END before that date.
    # ========================================================

    research_start = pd.Timestamp(
        "2016-09-26"
    )

    holdout_dates = [

        d

        for d in common_dates

        if pd.Timestamp(d)
        <
        research_start
    ]

    if len(holdout_dates) < 500:

        raise ValueError(
            "Pre-2016 holdout için yeterli "
            "ortak işlem günü bulunamadı."
        )

    return (
        prepared,
        holdout_dates,
        errors,
    )


# ============================================================
# V1.3 CONFIRM3 CANDIDATES
# ============================================================

def v13_pre2016_confirmed_candidates(
    prepared,
    common_dates,
    signal_index,
    entry_score=75,
    confirmation_days=3,
):

    candidates = []

    # --------------------------------------------------------
    # Need current day + previous confirmation days
    # --------------------------------------------------------

    if (
        signal_index
        <
        confirmation_days - 1
    ):

        return candidates

    signal_date = (
        common_dates[
            signal_index
        ]
    )

    confirmation_dates = (

        common_dates[
            signal_index
            -
            confirmation_days
            +
            1
            :
            signal_index
            +
            1
        ]
    )

    for symbol, df in (
        prepared.items()
    ):

        confirmed = True

        # ----------------------------------------------------
        # Frozen V1.3 rule:
        #
        # AL_ADAYI + score >= 75
        # for THREE consecutive closes
        # ----------------------------------------------------

        for d in confirmation_dates:

            if d not in df.index:

                confirmed = False
                break

            row = df.loc[d]

            if (
                int(
                    row["SCORE"]
                )
                <
                entry_score
            ):

                confirmed = False
                break

            if (
                row["MODEL_SIGNAL"]
                !=
                "AL_ADAYI"
            ):

                confirmed = False
                break

        if not confirmed:

            continue

        row = df.loc[
            signal_date
        ]

        candidates.append({

            "symbol":
                symbol,

            "score":
                int(
                    row["SCORE"]
                ),

            "mom120":
                float(
                    row["MOM120"]
                ),

            "mom60":
                float(
                    row["MOM60"]
                ),

            "mom20":
                float(
                    row["MOM20"]
                ),
        })

    # --------------------------------------------------------
    # Frozen ranking
    # --------------------------------------------------------

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


# ============================================================
# V1.3 PRE-2016 TOP-2 STICKY ENGINE
# ============================================================

def run_v13_pre2016_holdout_engine(
    prepared,
    common_dates,
    entry_score=75,
    exit_score=50,
    transaction_cost_pct=0.10,
    confirmation_days=3,
):

    top_n = 2

    cost = (
        transaction_cost_pct
        /
        100
    )

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

    # ========================================================
    # LOOP
    #
    # Signal at today's CLOSE.
    # Execution at NEXT trading day's OPEN.
    # ========================================================

    for i in range(
        len(common_dates) - 1
    ):

        signal_date = (
            common_dates[i]
        )

        next_date = (
            common_dates[i + 1]
        )

        # ====================================================
        # OPEN -> NEXT OPEN MARK TO MARKET
        # ====================================================

        if holdings:

            weight = (
                1.0
                /
                top_n
            )

            daily_return = 0.0

            for symbol in list(
                holdings.keys()
            ):

                df = (
                    prepared[symbol]
                )

                current_open = float(

                    df.loc[
                        signal_date,
                        "Open",
                    ]
                )

                next_open = float(

                    df.loc[
                        next_date,
                        "Open",
                    ]
                )

                if current_open > 0:

                    daily_return += (

                        (
                            next_open
                            /
                            current_open
                            -
                            1
                        )

                        *

                        weight
                    )

            equity *= (
                1
                +
                daily_return
            )

        # ====================================================
        # HOLDING STATE FOR THIS DAY
        # ====================================================

        if holdings:

            invested_days += 1

        else:

            cash_days += 1

        # ====================================================
        # FROZEN SLOW EXIT
        # ====================================================

        old_symbols = list(
            holdings.keys()
        )

        exiting = []

        for symbol in old_symbols:

            row = (
                prepared[symbol]
                .loc[
                    signal_date
                ]
            )

            if rotation_exit_required(
                row,
                exit_score,
            ):

                exiting.append(
                    symbol
                )

        survivors = [

            symbol

            for symbol in old_symbols

            if symbol not in exiting
        ]

        # ====================================================
        # V1.3 CONFIRM3 RANKING
        #
        # Ranking ONLY fills empty slots.
        #
        # Existing survivor is NOT displaced by a stronger
        # ranked commodity.
        # ====================================================

        candidates = (
            v13_pre2016_confirmed_candidates(
                prepared=
                    prepared,

                common_dates=
                    common_dates,

                signal_index=
                    i,

                entry_score=
                    entry_score,

                confirmation_days=
                    confirmation_days,
            )
        )

        candidate_symbols = [

            x["symbol"]

            for x in candidates
        ]

        next_symbols = list(
            survivors
        )

        entering = []

        free_slots = max(

            0,

            top_n
            -
            len(
                next_symbols
            ),
        )

        if free_slots > 0:

            for symbol in (
                candidate_symbols
            ):

                if symbol in next_symbols:
                    continue

                entering.append(
                    symbol
                )

                next_symbols.append(
                    symbol
                )

                if (
                    len(entering)
                    >=
                    free_slots
                ):
                    break

        # ====================================================
        # EXECUTE EXITS AT NEXT OPEN
        # ====================================================

        for symbol in exiting:

            if symbol not in holdings:
                continue

            df = (
                prepared[symbol]
            )

            exit_price = float(

                df.loc[
                    next_date,
                    "Open",
                ]
            )

            position = (
                holdings[symbol]
            )

            entry_price = float(
                position[
                    "entry_price"
                ]
            )

            entry_date = (
                position[
                    "entry_date"
                ]
            )

            # -----------------------------------------------
            # Trade return including BOTH transaction costs
            # -----------------------------------------------

            gross_return = (
                exit_price
                /
                entry_price
                -
                1
            )

            net_return = (

                (
                    1
                    +
                    gross_return
                )

                *

                (
                    1
                    -
                    cost
                )

                /

                (
                    1
                    +
                    cost
                )

                -
                1
            )

            holding_days = (

                pd.Timestamp(
                    next_date
                )

                -

                pd.Timestamp(
                    entry_date
                )

            ).days

            trades.append({

                "symbol":
                    symbol,

                "entry_date":
                    str(
                        pd.Timestamp(
                            entry_date
                        ).date()
                    ),

                "exit_date":
                    str(
                        pd.Timestamp(
                            next_date
                        ).date()
                    ),

                "entry_price":
                    safe_float(
                        entry_price,
                        4,
                    ),

                "exit_price":
                    safe_float(
                        exit_price,
                        4,
                    ),

                "return_pct":
                    safe_float(
                        net_return
                        *
                        100,
                        2,
                    ),

                "holding_days":
                    holding_days,

                "exit_reason":
                    "V1.2A_SLOW_EXIT",
            })

            # -----------------------------------------------
            # Apply exit transaction cost to portfolio equity
            # Weight = 1 / Top2
            # -----------------------------------------------

            equity *= (

                1
                -
                (
                    cost
                    /
                    top_n
                )
            )

            del holdings[
                symbol
            ]

            exit_count += 1

        # ====================================================
        # EXECUTE ENTRIES AT NEXT OPEN
        # ====================================================

        for symbol in entering:

            if symbol in holdings:
                continue

            df = (
                prepared[symbol]
            )

            entry_price = float(

                df.loc[
                    next_date,
                    "Open",
                ]
            )

            if entry_price <= 0:
                continue

            holdings[
                symbol
            ] = {

                "entry_date":
                    next_date,

                "entry_price":
                    entry_price,

                "signal_date":
                    signal_date,
            }

            # -----------------------------------------------
            # Apply entry transaction cost
            # -----------------------------------------------

            equity *= (

                1
                -
                (
                    cost
                    /
                    top_n
                )
            )

            entry_count += 1

        # ====================================================
        # ROTATION COUNT
        # ====================================================

        if (
            exiting
            and
            entering
        ):

            rotation_count += 1

        if (
            exiting
            or
            entering
        ):

            rebalances.append({

                "signal_date":
                    str(
                        pd.Timestamp(
                            signal_date
                        ).date()
                    ),

                "execution_date":
                    str(
                        pd.Timestamp(
                            next_date
                        ).date()
                    ),

                "exiting":
                    exiting,

                "entering":
                    entering,

                "holdings_after":
                    list(
                        holdings.keys()
                    ),
            })

        # ====================================================
        # DAILY DRAWDOWN
        # ====================================================

        if equity > peak:

            peak = equity

        if peak > 0:

            drawdown = (

                equity
                /
                peak
                -
                1
            )

            max_drawdown = min(

                max_drawdown,

                drawdown,
            )

        equity_curve.append({

            "date":
                str(
                    pd.Timestamp(
                        next_date
                    ).date()
                ),

            "equity":
                safe_float(
                    equity,
                    6,
                ),

            "drawdown_pct":
                safe_float(
                    drawdown
                    *
                    100
                    if peak > 0
                    else 0,
                    2,
                ),

            "holdings":
                list(
                    holdings.keys()
                ),
        })

    # ========================================================
    # FORCE CLOSE REMAINING POSITIONS
    #
    # Same principle as historical backtests:
    # close remaining positions on final available close.
    # ========================================================

    final_date = (
        common_dates[-1]
    )

    for symbol in list(
        holdings.keys()
    ):

        df = (
            prepared[symbol]
        )

        final_price = float(

            df.loc[
                final_date,
                "Close",
            ]
        )

        position = (
            holdings[symbol]
        )

        entry_price = float(
            position[
                "entry_price"
            ]
        )

        entry_date = (
            position[
                "entry_date"
            ]
        )

        gross_return = (

            final_price
            /
            entry_price
            -
            1
        )

        net_return = (

            (
                1
                +
                gross_return
            )

            *

            (
                1
                -
                cost
            )

            /

            (
                1
                +
                cost
            )

            -
            1
        )

        holding_days = (

            pd.Timestamp(
                final_date
            )

            -

            pd.Timestamp(
                entry_date
            )

        ).days

        trades.append({

            "symbol":
                symbol,

            "entry_date":
                str(
                    pd.Timestamp(
                        entry_date
                    ).date()
                ),

            "exit_date":
                str(
                    pd.Timestamp(
                        final_date
                    ).date()
                ),

            "entry_price":
                safe_float(
                    entry_price,
                    4,
                ),

            "exit_price":
                safe_float(
                    final_price,
                    4,
                ),

            "return_pct":
                safe_float(
                    net_return
                    *
                    100,
                    2,
                ),

            "holding_days":
                holding_days,

            "exit_reason":
                "FORCED_FINAL_CLOSE",
        })

        # ----------------------------------------------------
        # Mark final open -> close for remaining holding
        # ----------------------------------------------------

        final_open = float(

            df.loc[
                final_date,
                "Open",
            ]
        )

        if final_open > 0:

            final_intraday_return = (

                final_price
                /
                final_open
                -
                1
            )

            equity *= (

                1

                +

                (
                    final_intraday_return
                    /
                    top_n
                )
            )

        equity *= (

            1
            -
            (
                cost
                /
                top_n
            )
        )

        exit_count += 1

        del holdings[
            symbol
        ]

    # ========================================================
    # FINAL DRAWDOWN UPDATE
    # ========================================================

    if equity > peak:

        peak = equity

    if peak > 0:

        final_drawdown = (

            equity
            /
            peak
            -
            1
        )

        max_drawdown = min(

            max_drawdown,

            final_drawdown,
        )

    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    trade_returns = [

        float(
            x["return_pct"]
        )

        for x in trades

        if x.get(
            "return_pct"
        )
        is not None
    ]

    winning_returns = [

        x

        for x in trade_returns

        if x > 0
    ]

    losing_returns = [

        x

        for x in trade_returns

        if x < 0
    ]

    trade_count = len(
        trade_returns
    )

    win_rate = (

        len(
            winning_returns
        )

        /
        trade_count

        *
        100

        if trade_count
        else 0
    )

    average_trade = (

        float(
            np.mean(
                trade_returns
            )
        )

        if trade_returns
        else 0
    )

    median_trade = (

        float(
            np.median(
                trade_returns
            )
        )

        if trade_returns
        else 0
    )

    holding_values = [

        float(
            x["holding_days"]
        )

        for x in trades
    ]

    average_holding = (

        float(
            np.mean(
                holding_values
            )
        )

        if holding_values
        else 0
    )

    gross_profit = sum(
        winning_returns
    )

    gross_loss = abs(
        sum(
            losing_returns
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        profit_factor = 999.0

    else:

        profit_factor = 0.0

    # ========================================================
    # PERFORMANCE
    # ========================================================

    total_return = (

        equity
        -
        1
    ) * 100

    start_date = pd.Timestamp(
        common_dates[0]
    )

    end_date = pd.Timestamp(
        common_dates[-1]
    )

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
        and
        equity > 0
    ):

        cagr = (

            equity
            **
            (
                1
                /
                elapsed_years
            )

            -
            1

        ) * 100

    else:

        cagr = None

    active_days = (

        cash_days
        +
        invested_days
    )

    cash_pct = (

        cash_days
        /
        active_days
        *
        100

        if active_days
        else 0
    )

    invested_pct = (

        invested_days
        /
        active_days
        *
        100

        if active_days
        else 0
    )

    return {

        "portfolio":
            "TOP_2_EQUAL_WEIGHT",

        "rotation_model":
            "V1.3_CONFIRM3_STICKY",

        "performance": {

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

            "max_drawdown_daily_pct":
                safe_float(
                    max_drawdown
                    *
                    100,
                    2,
                ),

            "trade_count":
                trade_count,

            "win_rate_pct":
                safe_float(
                    win_rate,
                    2,
                ),

            "profit_factor":
                safe_float(
                    profit_factor,
                    3,
                ),

            "average_trade_pct":
                safe_float(
                    average_trade,
                    2,
                ),

            "median_trade_pct":
                safe_float(
                    median_trade,
                    2,
                ),

            "average_holding_days":
                safe_float(
                    average_holding,
                    1,
                ),

            "entry_count":
                entry_count,

            "exit_count":
                exit_count,

            "rotation_count":
                rotation_count,

            "cash_days":
                cash_days,

            "cash_time_pct":
                safe_float(
                    cash_pct,
                    2,
                ),

            "invested_time_pct":
                safe_float(
                    invested_pct,
                    2,
                ),

            "final_equity":
                safe_float(
                    equity,
                    6,
                ),
        },

        "trades":
            trades,

        "rebalances":
            rebalances,

        "equity_curve":
            equity_curve,
    }


# ============================================================
# V1.3 PRE-2016 HOLDOUT
# ============================================================

def run_v13_pre2016_holdout():

    entry_score = 75
    exit_score = 50
    confirmation_days = 3
    transaction_cost_pct = 0.10

    (
        prepared,
        holdout_dates,
        errors,
    ) = (
        v13_pre2016_prepare_data()
    )

    # ========================================================
    # RUN FROZEN TOP-2 V1.3
    # ========================================================

    result = (
        run_v13_pre2016_holdout_engine(

            prepared=
                prepared,

            common_dates=
                holdout_dates,

            entry_score=
                entry_score,

            exit_score=
                exit_score,

            transaction_cost_pct=
                transaction_cost_pct,

            confirmation_days=
                confirmation_days,
        )
    )

    # ========================================================
    # DBC BUY & HOLD BENCHMARK
    # ========================================================

    dbc_return = None

    if "DBC" in prepared:

        dbc = (
            prepared["DBC"]
        )

        first_date = (
            holdout_dates[0]
        )

        last_date = (
            holdout_dates[-1]
        )

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
                -
                1

            ) * 100

    # ========================================================
    # SYMBOL TRADE SUMMARY
    # ========================================================

    symbol_summary = {}

    for trade in result["trades"]:

        symbol = (
            trade["symbol"]
        )

        if symbol not in symbol_summary:

            symbol_summary[
                symbol
            ] = {

                "trade_count":
                    0,

                "winner_count":
                    0,

                "loser_count":
                    0,

                "total_trade_return_pct":
                    0.0,
            }

        item = (
            symbol_summary[
                symbol
            ]
        )

        item[
            "trade_count"
        ] += 1

        trade_return = float(
            trade[
                "return_pct"
            ]
        )

        item[
            "total_trade_return_pct"
        ] += (
            trade_return
        )

        if trade_return > 0:

            item[
                "winner_count"
            ] += 1

        elif trade_return < 0:

            item[
                "loser_count"
            ] += 1

    for symbol, item in (
        symbol_summary.items()
    ):

        count = (
            item[
                "trade_count"
            ]
        )

        winners = (
            item[
                "winner_count"
            ]
        )

        item[
            "win_rate_pct"
        ] = (

            safe_float(
                winners
                /
                count
                *
                100,
                2,
            )

            if count
            else 0
        )

        item[
            "total_trade_return_pct"
        ] = safe_float(

            item[
                "total_trade_return_pct"
            ],

            2,
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "test":
            "V1.3_PRE_2016_HISTORICAL_HOLDOUT",

        "model":
            "V1.3_CONFIRM3",

        "generated_at_utc":
            utc_now(),

        "purpose":
            (
                "Evaluate frozen V1.3 CONFIRM3 on "
                "historical data before the existing "
                "2016-09-26 research window."
            ),

        "methodology": {

            "parameter_optimization":
                False,

            "strategy_modified":
                False,

            "v14_breadth_filter":
                False,

            "v15_overextension_filter":
                False,

            "portfolio":
                "TOP_2_EQUAL_WEIGHT",

            "sticky_rotation":
                True,

            "entry_score":
                entry_score,

            "exit_score":
                exit_score,

            "confirmation_days":
                confirmation_days,

            "transaction_cost_pct_each_side":
                transaction_cost_pct,

            "ranking":
                "SCORE, MOM120, MOM60, MOM20",

            "entry_rule":
                (
                    "AL_ADAYI and score >= 75 "
                    "for 3 consecutive closes"
                ),

            "exit_rule":
                (
                    "score <= 50 OR EMA20 < EMA50 "
                    "OR Close < EMA100 "
                    "OR EMA50 < EMA200"
                ),

            "holding_rule":
                (
                    "Existing holding remains until "
                    "its own slow-exit. Ranking only "
                    "fills empty slots."
                ),

            "execution":
                (
                    "Signal at close, execution at "
                    "next trading day open"
                ),

            "long_only":
                True,

            "leverage":
                False,

            "automatic_orders":
                False,
        },

        "holdout_period": {

            "start":
                str(
                    pd.Timestamp(
                        holdout_dates[0]
                    ).date()
                ),

            "end":
                str(
                    pd.Timestamp(
                        holdout_dates[-1]
                    ).date()
                ),

            "common_trading_days":
                len(
                    holdout_dates
                ),

            "research_window_starts":
                "2016-09-26",
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

        "performance":
            result[
                "performance"
            ],

        "symbol_summary":
            symbol_summary,

        "trades":
            result[
                "trades"
            ],

        "rebalances":
            result[
                "rebalances"
            ],

        "equity_curve":
            result[
                "equity_curve"
            ],

        "data_errors":
            errors,

        "interpretation_warning":
            (
                "This period is being used as a historical "
                "holdout for V1.3. Do not optimize V1.3 "
                "parameters using this result."
            ),
    }


# ============================================================
# API ENDPOINT
# ============================================================

@app.get("/v13-pre2016-holdout")
def v13_pre2016_holdout_endpoint():

    try:

        return (
            run_v13_pre2016_holdout()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
