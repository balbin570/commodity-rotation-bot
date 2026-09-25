from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np
import math
import traceback
from datetime import datetime, timezone


# ============================================================
# COMMODITY ROTATION BOT
# Version: 1.0
#
# Amaç:
# - Kaldıraç yok
# - Short yok
# - Futures işlemi yok
# - Sadece ABD borsasında işlem gören ETF/ETP'leri takip eder
# - İşlemleri otomatik yapmaz
# - Sadece analiz ve sinyal üretir
#
# İlk sürüm:
# GLD  = Gold
# SLV  = Silver
# USO  = Oil
# DBC  = Broad Commodities
# ============================================================


app = FastAPI(
    title="Commodity Rotation Bot",
    description="Long-only commodity ETF trend and rotation signal system",
    version="1.0.0",
)


MODEL_NAME = "COMMODITY-ROTATION-V1"


# ============================================================
# TAKİP EDİLECEK ÜRÜNLER
# ============================================================

ASSETS = {
    "GLD": {
        "name": "Gold",
        "tr_name": "Altın",
        "category": "PRECIOUS_METALS",
    },
    "SLV": {
        "name": "Silver",
        "tr_name": "Gümüş",
        "category": "PRECIOUS_METALS",
    },
    "USO": {
        "name": "Oil",
        "tr_name": "Petrol",
        "category": "ENERGY",
    },
    "DBC": {
        "name": "Broad Commodities",
        "tr_name": "Geniş Emtia Sepeti",
        "category": "BROAD_COMMODITY",
    },
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def safe_float(value, digits=4):
    """
    numpy/pandas sayılarını JSON uyumlu Python float'a çevirir.
    NaN ve infinity değerlerini None yapar.
    """
    try:
        if value is None:
            return None

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return round(value, digits)

    except Exception:
        return None


def safe_int(value):
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return int(value)

    except Exception:
        return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# VERİ ÇEKME
# ============================================================

def download_history(
    symbol: str,
    period: str = "10y",
    interval: str = "1d",
):
    """
    Yahoo Finance üzerinden tarihsel veri çeker.

    auto_adjust=True:
    Bölünme ve temettü etkilerinin fiyat serisine yansıtılması açısından
    backtestlerde daha kullanışlı bir seri sağlar.
    """

    symbol = symbol.upper()

    try:

        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            raise ValueError(f"{symbol} için veri alınamadı.")

        # yfinance bazı sürümlerde MultiIndex döndürebiliyor.
        if isinstance(df.columns, pd.MultiIndex):

            if symbol in df.columns.get_level_values(-1):
                try:
                    df = df.xs(
                        symbol,
                        axis=1,
                        level=-1,
                        drop_level=True,
                    )
                except Exception:
                    pass

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]

        df = df.copy()

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"{symbol}: {column} kolonu bulunamadı."
                )

        df = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ].copy()

        for column in df.columns:
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
                f"{symbol}: teknik analiz için yeterli veri yok. "
                f"Satır sayısı: {len(df)}"
            )

        return df

    except Exception as exc:
        raise ValueError(
            f"{symbol} veri hatası: {str(exc)}"
        )


# ============================================================
# TEKNİK GÖSTERGELER
# ============================================================

def calculate_rsi(
    close: pd.Series,
    period: int = 14,
):
    """
    Wilder benzeri exponential RSI.
    """

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

    rsi = 100 - (
        100 / (1 + rs)
    )

    # Hiç kayıp olmayan çok güçlü serilerde RSI 100 kabul edilebilir.
    rsi = rsi.fillna(100)

    return rsi


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
):
    """
    Average True Range.
    """

    high_low = (
        df["High"] - df["Low"]
    ).abs()

    high_close = (
        df["High"] - df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def calculate_indicators(
    df: pd.DataFrame,
):
    """
    Commodity Rotation v1 göstergeleri.
    """

    df = df.copy()

    close = df["Close"]

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["MOM20"] = (
        close.pct_change(20) * 100
    )

    df["MOM60"] = (
        close.pct_change(60) * 100
    )

    df["MOM120"] = (
        close.pct_change(120) * 100
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI14"] = calculate_rsi(
        close,
        14,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["MACD"] = (
        close.ewm(
            span=12,
            adjust=False,
        ).mean()
        -
        close.ewm(
            span=26,
            adjust=False,
        ).mean()
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

    # --------------------------------------------------------
    # ATR / VOLATILITY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DISTANCE FROM EMA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    rolling_high = (
        close
        .rolling(252)
        .max()
    )

    df["DRAWDOWN_252"] = (
        (
            close
            /
            rolling_high
        )
        - 1
    ) * 100

    return df


# ============================================================
# PUANLAMA MODELİ
# ============================================================

def calculate_score(row):
    """
    Maksimum 100 puan.

    V1 bilinçli olarak basit ve açıklanabilir tutuluyor.

    TREND          40
    MOMENTUM       30
    MACD           10
    RSI            10
    RISK/VOL       10
    -----------------
    TOTAL         100
    """

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

    macd = row["MACD"]
    macd_signal = row["MACD_SIGNAL"]
    macd_hist = row["MACD_HIST"]

    atr_pct = row["ATR_PCT"]

    # ========================================================
    # 1. TREND
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
    # 2. MOMENTUM
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
    # 3. MACD
    # ========================================================

    if macd > macd_signal:
        score += 5
        reasons.append(
            "MACD signal üzerinde"
        )

    if macd_hist > 0:
        score += 5
        reasons.append(
            "MACD histogram pozitif"
        )

    # ========================================================
    # 4. RSI
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

    elif rsi < 40:
        warnings.append(
            "RSI zayıf"
        )

    # ========================================================
    # 5. VOLATILITY
    # ========================================================

    if atr_pct <= 3:
        score += 10
        reasons.append(
            "Volatilite kontrollü"
        )

    elif atr_pct <= 5:
        score += 5

    else:
        warnings.append(
            "Volatilite yüksek"
        )

    score = max(
        0,
        min(
            100,
            int(score),
        ),
    )

    return (
        score,
        reasons,
        warnings,
    )


# ============================================================
# SİNYAL MOTORU
# ============================================================

def determine_signal(
    row,
    score,
):
    """
    İlk versiyon sinyal mantığı.

    AL_ADAYI:
    Güçlü trend + momentum + yeterli puan.

    IZLE:
    Yapı fena değil ancak giriş için yeterince güçlü değil.

    BEKLE:
    Trend filtresi geçilmemiş.
    """

    close = row["Close"]

    ema20 = row["EMA20"]
    ema50 = row["EMA50"]
    ema200 = row["EMA200"]

    rsi = row["RSI14"]

    mom60 = row["MOM60"]
    mom120 = row["MOM120"]

    long_term_trend = (
        close > ema200
        and ema50 > ema200
    )

    medium_trend = (
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
        and medium_trend
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
# TEK SEMBOL ANALİZİ
# ============================================================

def analyze_symbol(
    symbol: str,
    period: str = "10y",
):
    symbol = symbol.upper()

    if symbol not in ASSETS:
        raise ValueError(
            f"{symbol} V1 ürün listesinde yok."
        )

    df = download_history(
        symbol=symbol,
        period=period,
    )

    df = calculate_indicators(
        df
    )

    clean = df.dropna().copy()

    if clean.empty:
        raise ValueError(
            f"{symbol}: göstergeler hesaplanamadı."
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

    result = {
        "model": MODEL_NAME,
        "symbol": symbol,
        "name": asset["name"],
        "tr_name": asset["tr_name"],
        "category": asset["category"],

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
            "distance_ema20_pct": safe_float(
                row["DIST_EMA20"],
                2,
            ),
            "distance_ema50_pct": safe_float(
                row["DIST_EMA50"],
                2,
            ),
            "distance_ema200_pct": safe_float(
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
            "drawdown_1y_pct": safe_float(
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

        "generated_at_utc": utc_now(),
    }

    return result


# ============================================================
# TÜM EMTİALARI TARA
# ============================================================

def scan_all_assets():
    results = []

    errors = []

    for symbol in ASSETS.keys():

        try:

            result = analyze_symbol(
                symbol=symbol,
                period="10y",
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

    buy_candidates = [
        item
        for item in results
        if item["signal"] == "AL_ADAYI"
    ]

    watch_candidates = [
        item
        for item in results
        if item["signal"] == "IZLE"
    ]

    return {
        "model": MODEL_NAME,
        "generated_at_utc": utc_now(),
        "asset_count": len(results),
        "buy_candidate_count": len(
            buy_candidates
        ),
        "watch_candidate_count": len(
            watch_candidates
        ),
        "results": results,
        "errors": errors,
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    symbol: str,
    years: int = 10,
    entry_score: int = 75,
    exit_score: int = 50,
    transaction_cost_pct: float = 0.10,
):
    """
    Basit long-only backtest.

    ÖNEMLİ:
    Sinyal kapanış verisiyle oluşur.
    İşlem bir sonraki günün OPEN fiyatından yapılır.

    Böylece aynı kapanış fiyatını kullanarak geleceği görme
    hatasını azaltmaya çalışıyoruz.

    transaction_cost_pct:
    Tek yön toplam işlem maliyeti varsayımı.
    Örnek 0.10 = %0.10.
    """

    symbol = symbol.upper()

    if symbol not in ASSETS:
        raise ValueError(
            f"{symbol} V1 ürün listesinde yok."
        )

    if years < 2:
        years = 2

    if years > 20:
        years = 20

    period = f"{years}y"

    df = download_history(
        symbol=symbol,
        period=period,
    )

    df = calculate_indicators(
        df
    )

    df = df.dropna().copy()

    if len(df) < 250:
        raise ValueError(
            "Backtest için yeterli veri yok."
        )

    # Her gün için puan ve sinyal hesapla.
    scores = []
    signals = []

    for _, row in df.iterrows():

        score, _, _ = (
            calculate_score(row)
        )

        signal = determine_signal(
            row,
            score,
        )

        scores.append(
            score
        )

        signals.append(
            signal
        )

    df["SCORE"] = scores
    df["MODEL_SIGNAL"] = signals

    # ========================================================
    # BACKTEST STATE
    # ========================================================

    in_position = False

    entry_price = None
    entry_date = None
    entry_score_value = None

    trades = []

    equity = 1.0
    equity_curve = []

    cost_fraction = (
        transaction_cost_pct
        / 100
    )

    # Sinyal i gününün kapanışında oluşuyor.
    # Emir i+1 OPEN'da uygulanıyor.
    for i in range(
        0,
        len(df) - 1,
    ):

        current = df.iloc[i]

        next_row = df.iloc[i + 1]

        current_date = df.index[i]

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

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        if not in_position:

            entry_condition = (
                score >= entry_score
                and signal == "AL_ADAYI"
            )

            if entry_condition:

                in_position = True

                entry_price = (
                    next_open
                    *
                    (
                        1
                        +
                        cost_fraction
                    )
                )

                entry_date = next_date

                entry_score_value = score

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        else:

            close = float(
                current["Close"]
            )

            ema50 = float(
                current["EMA50"]
            )

            ema200 = float(
                current["EMA200"]
            )

            exit_condition = (
                score <= exit_score
                or close < ema50
                or ema50 < ema200
            )

            if exit_condition:

                exit_price = (
                    next_open
                    *
                    (
                        1
                        -
                        cost_fraction
                    )
                )

                gross_return = (
                    (
                        exit_price
                        /
                        entry_price
                    )
                    - 1
                )

                gross_return_pct = (
                    gross_return
                    * 100
                )

                holding_days = (
                    next_date
                    -
                    entry_date
                ).days

                equity *= (
                    1
                    +
                    gross_return
                )

                trades.append(
                    {
                        "entry_date": str(
                            entry_date.date()
                        ),
                        "exit_date": str(
                            next_date.date()
                        ),
                        "entry_price": safe_float(
                            entry_price,
                            2,
                        ),
                        "exit_price": safe_float(
                            exit_price,
                            2,
                        ),
                        "entry_score": (
                            entry_score_value
                        ),
                        "exit_score": score,
                        "return_pct": safe_float(
                            gross_return_pct,
                            2,
                        ),
                        "holding_days": (
                            holding_days
                        ),
                    }
                )

                in_position = False
                entry_price = None
                entry_date = None
                entry_score_value = None

        equity_curve.append(
            {
                "date": str(
                    current_date.date()
                ),
                "equity": equity,
            }
        )

    # ========================================================
    # AÇIK POZİSYONU TEST SONUNDA KAPAT
    # ========================================================

    if in_position:

        final_row = df.iloc[-1]

        final_date = df.index[-1]

        final_close = float(
            final_row["Close"]
        )

        exit_price = (
            final_close
            *
            (
                1
                -
                cost_fraction
            )
        )

        gross_return = (
            (
                exit_price
                /
                entry_price
            )
            - 1
        )

        gross_return_pct = (
            gross_return
            * 100
        )

        holding_days = (
            final_date
            -
            entry_date
        ).days

        equity *= (
            1
            +
            gross_return
        )

        trades.append(
            {
                "entry_date": str(
                    entry_date.date()
                ),
                "exit_date": str(
                    final_date.date()
                ),
                "entry_price": safe_float(
                    entry_price,
                    2,
                ),
                "exit_price": safe_float(
                    exit_price,
                    2,
                ),
                "entry_score": (
                    entry_score_value
                ),
                "exit_score": safe_int(
                    final_row["SCORE"]
                ),
                "return_pct": safe_float(
                    gross_return_pct,
                    2,
                ),
                "holding_days": (
                    holding_days
                ),
                "forced_exit_at_test_end": True,
            }
        )

    # ========================================================
    # İSTATİSTİKLER
    # ========================================================

    returns = [
        trade["return_pct"]
        for trade in trades
        if trade["return_pct"] is not None
    ]

    winners = [
        value
        for value in returns
        if value > 0
    ]

    losers = [
        value
        for value in returns
        if value <= 0
    ]

    trade_count = len(
        trades
    )

    if trade_count > 0:

        win_rate = (
            len(winners)
            /
            trade_count
            *
            100
        )

        avg_trade = float(
            np.mean(
                returns
            )
        )

        median_trade = float(
            np.median(
                returns
            )
        )

        avg_holding = float(
            np.mean(
                [
                    trade[
                        "holding_days"
                    ]
                    for trade in trades
                ]
            )
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
        sum(
            losers
        )
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

    strategy_return_pct = (
        equity - 1
    ) * 100

    # --------------------------------------------------------
    # BUY & HOLD
    # --------------------------------------------------------

    first_open = float(
        df.iloc[0]["Open"]
    )

    last_close = float(
        df.iloc[-1]["Close"]
    )

    buy_hold_return_pct = (
        (
            last_close
            /
            first_open
        )
        - 1
    ) * 100

    # --------------------------------------------------------
    # MAX DRAWDOWN
    #
    # Burada trade-level equity üzerinden kaba ilk ölçüm.
    # V2'de günlük mark-to-market equity curve kullanacağız.
    # --------------------------------------------------------

    running_equity = 1.0

    peak_equity = 1.0

    max_drawdown = 0.0

    for trade in trades:

        r = (
            trade["return_pct"]
            /
            100
        )

        running_equity *= (
            1 + r
        )

        peak_equity = max(
            peak_equity,
            running_equity,
        )

        drawdown = (
            (
                running_equity
                /
                peak_equity
            )
            - 1
        ) * 100

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    start_date = df.index[0]

    end_date = df.index[-1]

    elapsed_days = (
        end_date
        -
        start_date
    ).days

    elapsed_years = (
        elapsed_days
        /
        365.25
    )

    if (
        elapsed_years > 0
        and equity > 0
    ):

        cagr = (
            (
                equity
                **
                (
                    1
                    /
                    elapsed_years
                )
            )
            - 1
        ) * 100

    else:
        cagr = None

    return {
        "model": MODEL_NAME,
        "symbol": symbol,
        "name": ASSETS[
            symbol
        ]["name"],

        "test_period": {
            "start": str(
                start_date.date()
            ),
            "end": str(
                end_date.date()
            ),
            "years_requested": years,
        },

        "settings": {
            "entry_score": entry_score,
            "exit_score": exit_score,
            "transaction_cost_pct_each_side": (
                transaction_cost_pct
            ),
            "execution": (
                "Signal at close, "
                "execution at next trading day open"
            ),
            "long_only": True,
            "short": False,
            "leverage": False,
        },

        "performance": {
            "trade_count": trade_count,
            "win_rate_pct": safe_float(
                win_rate,
                2,
            ),
            "average_trade_pct": safe_float(
                avg_trade,
                2,
            ),
            "median_trade_pct": safe_float(
                median_trade,
                2,
            ),
            "average_holding_days": safe_float(
                avg_holding,
                1,
            ),
            "profit_factor": safe_float(
                profit_factor,
                3,
            ),
            "strategy_total_return_pct": safe_float(
                strategy_return_pct,
                2,
            ),
            "strategy_cagr_pct": safe_float(
                cagr,
                2,
            ),
            "max_drawdown_trade_level_pct": safe_float(
                max_drawdown,
                2,
            ),
            "buy_hold_return_pct": safe_float(
                buy_hold_return_pct,
                2,
            ),
        },

        "trades": trades,

        "generated_at_utc": utc_now(),
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
        "tracked_assets": list(
            ASSETS.keys()
        ),
        "endpoints": {
            "health": "/health",
            "assets": "/assets",
            "analyze_example": "/analyze/GLD",
            "scan": "/scan",
            "backtest_example": (
                "/backtest/GLD?years=10"
            ),
            "dashboard": "/dashboard",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "time_utc": utc_now(),
    }


@app.get("/assets")
def assets():
    return {
        "model": MODEL_NAME,
        "count": len(
            ASSETS
        ),
        "assets": ASSETS,
    }


@app.get("/analyze/{symbol}")
def analyze(
    symbol: str,
):
    try:

        return analyze_symbol(
            symbol=symbol,
            period="10y",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/scan")
def scan():
    try:

        return scan_all_assets()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/backtest/{symbol}")
def backtest(
    symbol: str,

    years: int = Query(
        default=10,
        ge=2,
        le=20,
    ),

    entry_score: int = Query(
        default=75,
        ge=50,
        le=100,
    ),

    exit_score: int = Query(
        default=50,
        ge=0,
        le=80,
    ),

    transaction_cost_pct: float = Query(
        default=0.10,
        ge=0,
        le=2,
    ),
):
    try:

        return run_backtest(
            symbol=symbol,
            years=years,
            entry_score=entry_score,
            exit_score=exit_score,
            transaction_cost_pct=(
                transaction_cost_pct
            ),
        )

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# BASİT DASHBOARD
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
        <title>Commodity Rotation Bot</title>

        <meta charset="UTF-8">

        <style>

            body {
                font-family: Arial, sans-serif;
                max-width: 1100px;
                margin: 40px auto;
                padding: 20px;
                background: #f5f5f5;
            }

            h1 {
                margin-bottom: 5px;
            }

            .subtitle {
                color: #666;
                margin-bottom: 30px;
            }

            .warning {
                background: #fff3cd;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 25px;
            }

            .links {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
            }

            a {
                display: block;
                margin: 10px 0;
                font-size: 18px;
            }

            code {
                background: #eee;
                padding: 3px 6px;
                border-radius: 4px;
            }

        </style>

    </head>

    <body>

        <h1>
            Commodity Rotation Bot v1.0
        </h1>

        <div class="subtitle">
            Long-only commodity ETF signal system
        </div>

        <div class="warning">

            Bu sistem otomatik işlem yapmaz.

            <br><br>

            Short, futures veya kaldıraç kullanmaz.

            <br><br>

            İlk amaç:
            algoritmayı tarihsel ve forward veride test etmektir.

        </div>

        <div class="links">

            <h2>Endpoints</h2>

            <a href="/health">
                Health Check
            </a>

            <a href="/assets">
                Takip Edilen Ürünler
            </a>

            <a href="/analyze/GLD">
                GLD Analizi
            </a>

            <a href="/analyze/SLV">
                SLV Analizi
            </a>

            <a href="/analyze/USO">
                USO Analizi
            </a>

            <a href="/analyze/DBC">
                DBC Analizi
            </a>

            <a href="/scan">
                Tümünü Tara
            </a>

            <a href="/backtest/GLD?years=10">
                GLD 10 Yıllık Backtest
            </a>

            <a href="/backtest/SLV?years=10">
                SLV 10 Yıllık Backtest
            </a>

            <a href="/backtest/USO?years=10">
                USO 10 Yıllık Backtest
            </a>

            <a href="/backtest/DBC?years=10">
                DBC 10 Yıllık Backtest
            </a>

        </div>

        <div class="links">

            <h2>Model</h2>

            <p>
                <strong>
                    COMMODITY-ROTATION-V1
                </strong>
            </p>

            <p>
                Trend:
                EMA20 / EMA50 / EMA200
            </p>

            <p>
                Momentum:
                20 / 60 / 120 gün
            </p>

            <p>
                Ek göstergeler:
                RSI14, MACD, ATR
            </p>

            <p>
                Sinyaller:
                <code>AL_ADAYI</code>,
                <code>IZLE</code>,
                <code>BEKLE</code>
            </p>

        </div>

    </body>
    </html>
    """

    return HTMLResponse(
        content=html
    )
