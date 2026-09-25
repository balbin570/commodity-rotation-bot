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
