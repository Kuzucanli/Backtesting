# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 15:03:20 2025

@author: Asus
"""
from numpy import fabs as npfabs
import pandas as pd
import numpy as np
from sys import float_info as sflt


def psar(high, low, close=None, af=None, max_af=None, offset=0, **kwargs):
    """Indicator: Parabolic Stop and Reverse (PSAR)"""
    # Validate Arguments
    high = (high)
    low = (low)
    af = float(af) if af and af > 0 else 0.02
    max_af = float(max_af) if max_af and max_af > 0 else 0.2
    offset = (offset)

    # Initialize
    m = high.shape[0]
    af0 = af
    bullish = True
    high_point = high.iloc[0]
    low_point = low.iloc[0]

    if close is not None:
        close = (close)
        sar = close.copy()
    else:
        sar = low.copy()

    long = pd.Series(np.nan, index=sar.index)
    short = long.copy()
    reversal = pd.Series(False, index=sar.index)
    _af = long.copy()
    _af.iloc[0:2] = af0

    # Calculate Result
    for i in range(2, m):
        reverse = False
        _af.iloc[i] = af

        if bullish:
            sar.iloc[i] = sar.iloc[i - 1] + af * (high_point - sar.iloc[i - 1])

            if low.iloc[i] < sar.iloc[i]:
                bullish, reverse, af = False, True, af0
                sar.iloc[i] = high_point
                low_point = low.iloc[i]
        else:
            sar.iloc[i] = sar.iloc[i - 1] + af * (low_point - sar.iloc[i - 1])

            if high.iloc[i] > sar.iloc[i]:
                bullish, reverse, af = True, True, af0
                sar.iloc[i] = low_point
                high_point = high.iloc[i]

        reversal.iloc[i] = reverse

        if not reverse:
            if bullish:
                if high.iloc[i] > high_point:
                    high_point = high.iloc[i]
                    af = min(af + af0, max_af)
                if low.iloc[i - 1] < sar.iloc[i]:
                    sar.iloc[i] = low.iloc[i - 1]
                if low.iloc[i - 2] < sar.iloc[i]:
                    sar.iloc[i] = low.iloc[i - 2]
            else:
                if low.iloc[i] < low_point:
                    low_point = low.iloc[i]
                    af = min(af + af0, max_af)
                if high.iloc[i - 1] > sar.iloc[i]:
                    sar.iloc[i] = high.iloc[i - 1]
                if high.iloc[i - 2] > sar.iloc[i]:
                    sar.iloc[i] = high.iloc[i - 2]

        if bullish:
            long.iloc[i] = sar.iloc[i]
        else:
            short.iloc[i] = sar.iloc[i]

    # Offset
    if offset != 0:
        _af = _af.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)
        reversal = reversal.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        _af.fillna(kwargs["fillna"], inplace=True)
        long.fillna(kwargs["fillna"], inplace=True)
        short.fillna(kwargs["fillna"], inplace=True)
        reversal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        _af.fillna(method=kwargs["fill_method"], inplace=True)
        long.fillna(method=kwargs["fill_method"], inplace=True)
        short.fillna(method=kwargs["fill_method"], inplace=True)
        reversal.fillna(method=kwargs["fill_method"], inplace=True)

    # Prepare DataFrame to return
    _params = f"_{af0}_{max_af}"
    data = {
        f"PSARl{_params}": long,
        f"PSARs{_params}": short,
        f"PSARaf{_params}": _af,
        f"PSARr{_params}": reversal,
    }
    psardf = pd.DataFrame(data)
    psardf.name = f"PSAR{_params}"
    psardf.category = long.category = short.category = "trend"

    return psardf



def sma(close, length=None, offset=0, **kwargs):
    """Indicator: Simple Moving Average (SMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    close = close
    offset = offset

    if close is None: return

    # Calculate Result
    sma = close.rolling(length, min_periods=min_periods).mean()

    # Offset
    if offset != 0:
        sma = sma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        sma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        sma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    sma.name = f"SMA_{length}"
    sma.category = "overlap"

    return sma

def ema(close, length=None, offset=0, **kwargs):
    """Indicator: Exponential Moving Average (EMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    adjust = kwargs.pop("adjust", False)
    sma = kwargs.pop("sma", True)
    close = close
    offset = offset

    if close is None: return

    # Calculate Result
    if sma:
        close = close.copy()
        sma_nth = close[0:length].mean()
        close[:length - 1] = np.nan
        close.iloc[length - 1] = sma_nth
    ema = close.ewm(span=length, adjust=adjust).mean()

    # Offset
    if offset != 0:
        ema = ema.shift(offset)

    # Name & Category
    ema.name = f"EMA_{length}"
    ema.category = "overlap"

    return ema


def alma(close, length=None, sigma=None, distribution_offset=None, offset=0, **kwargs):
    """Indicator: Arnaud Legoux Moving Average (ALMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    sigma = float(sigma) if sigma and sigma > 0 else 6.0
    distribution_offset = float(distribution_offset) if distribution_offset and distribution_offset > 0 else 0.85
    close =close
    offset = offset

    if close is None: return

    # Pre-Calculations
    m = distribution_offset * (length - 1)
    s = length / sigma
    wtd = list(range(length))
    for i in range(0, length):
        wtd[i] = np.exp(-1 * ((i - m) * (i - m)) / (2 * s * s))

    # Calculate Result
    result = [np.nan for _ in range(0, length - 1)] + [0]
    for i in range(length, close.size):
        window_sum = 0
        cum_sum = 0
        for j in range(0, length):
            # wtd = math.exp(-1 * ((j - m) * (j - m)) / (2 * s * s))        # moved to pre-calc for efficiency
            window_sum = window_sum + wtd[j] * close.iloc[i - j]
            cum_sum = cum_sum + wtd[j]

        almean = window_sum / cum_sum
        result.append(np.nan) if i == length else result.append(almean)

    alma = pd.Series(result, index=close.index)

    # Offset
    if offset != 0:
        alma = alma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        alma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        alma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    alma.name = f"ALMA_{length}_{sigma}_{distribution_offset}"
    alma.category = "overlap"

    return alma


def macd(close, fast=None, slow=None, signal=None, offset=0, **kwargs):
    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    if slow < fast:
        fast, slow = slow, fast
    close = close
    offset = (offset)

    if close is None: return

    # Calculate Result
    fastma = ema(close, length=fast)
    slowma = ema(close, length=slow)

    macd = fastma - slowma
    signalma = ema(close=macd.loc[macd.first_valid_index():,], length=signal)
    histogram = macd - signalma

    # Offset
    if offset != 0:
        macd = macd.shift(offset)
        histogram = histogram.shift(offset)
        signalma = signalma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        macd.fillna(kwargs["fillna"], inplace=True)
        histogram.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        macd.fillna(method=kwargs["fill_method"], inplace=True)
        histogram.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _props = f"_{fast}_{slow}_{signal}"
    macd.name = f"MACD{_props}"
    histogram.name = f"MACDh{_props}"
    signalma.name = f"MACDs{_props}"
    macd.category = histogram.category = signalma.category = "momentum"

    # Prepare DataFrame to return
    data = {macd.name: macd, histogram.name: histogram, signalma.name: signalma}
    df = pd.DataFrame(data)
    df.name = f"MACD{_props}"
    df.category = macd.category

    
    return df


def dema(close, length=None, offset=0, **kwargs):
    """Indicator: Double Exponential Moving Average (DEMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = close
    offset = offset

    if close is None: return

    # Calculate Result
    ema1 = ema(close=close, length=length)
    ema2 = ema(close=ema1, length=length)
    dema = 2 * ema1 - ema2

    # Offset
    if offset != 0:
        dema = dema.shift(offset)

    # Name & Category
    dema.name = f"DEMA_{length}"
    dema.category = "overlap"

    return dema



def calculate_atr(data, period=14):
    """
    ATR (Average True Range) hesaplar.
    Args:
        data (pd.DataFrame): 'High', 'Low', 'Close' sütunlarını içeren veri seti.
        period (int): ATR için kullanılacak dönem sayısı (varsayılan: 14).
    Returns:
        pd.Series: ATR değerleri, ilk (period-1) değer NaN olabilir.
    """
    # True Range (TR) hesapla
    data['prev_close'] = data['Close'].shift(1)
    data['high_low'] = data['High'] - data['Low']
    data['high_prev_close'] = np.abs(data['High'] - data['prev_close'])
    data['low_prev_close'] = np.abs(data['Low'] - data['prev_close'])
    
    # TR = Max(High-Low, |High-Prev Close|, |Low-Prev Close|)
    data['true_range'] = data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    
    # İlk ATR: period dönemlik TR ortalaması
    atr = data['true_range'].rolling(window=period).mean()
    
    # Wilder'ın yumuşatma yöntemini uygula (isteğe bağlı, pandas_ta ile uyum için)
    for i in range(period, len(data)):
        atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + data['true_range'].iloc[i]) / period
    
    return atr



def mom(close, length=None, offset=None, **kwargs):
    """Indicator: Momentum (MOM)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = close
    offset = 0

    if close is None: return

    # Calculate Result
    mom = close.diff(length)

    # Offset
    if offset != 0:
        mom = mom.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        mom.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        mom.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    mom.name = f"MOM_{length}"
    mom.category = "momentum"

    return mom


def rma(close, length=None, offset=None, **kwargs):
    """Indicator: wildeR's Moving Average (RMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    alpha = (1.0 / length) if length > 0 else 0.5
    close =close
    offset =0

    if close is None: return

    # Calculate Result
    rma = close.ewm(alpha=alpha, min_periods=length).mean()

    # Offset
    if offset != 0:
        rma = rma.shift(offset)

    # Name & Category
    rma.name = f"RMA_{length}"
    rma.category = "overlap"

    return rma

def t3(close, length=None, a=None, offset=None, **kwargs):
    """Indicator: T3"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    a = float(a) if a and a > 0 and a < 1 else 0.7
    close = close
    offset = 0

    if close is None: return

    # Calculate Result
    c1 = -a * a**2
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = a**3 + 3 * a**2 + 3 * a + 1

    e1 = close.ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    e4 = e3.ewm(span=length, adjust=False).mean()
    e5 = e4.ewm(span=length, adjust=False).mean()
    e6 = e5.ewm(span=length, adjust=False).mean()
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # Offset
    if offset != 0:
        t3 = t3.shift(offset)

    # Name & Category
    t3.name = f"T3_{length}_{a}"
    t3.category = "overlap"

    return t3


def midprice(high, low, length=None, offset=0, **kwargs):
    """Indicator: Midprice"""
    # Validate arguments
    length = int(length) if length and length > 0 else 2
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    _length = max(length, min_periods)
    high = high
    low = low
    offset = (offset)

    if high is None or low is None: return

    # Calculate Result
    lowest_low = low.rolling(length, min_periods=min_periods).min()
    highest_high = high.rolling(length, min_periods=min_periods).max()
    midprice = 0.5 * (lowest_low + highest_high)

    # Offset
    if offset != 0:
        midprice = midprice.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        midprice.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        midprice.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    midprice.name = f"MIDPRICE_{length}"
    midprice.category = "overlap"

    return midprice

def ichimoku(high, low, close, tenkan=None, kijun=None, senkou=None, offset=0, **kwargs):
    """Indicator: Ichimoku Kinkō Hyō (Ichimoku)"""
    tenkan = int(tenkan) if tenkan and tenkan > 0 else 9
    kijun = int(kijun) if kijun and kijun > 0 else 26
    senkou = int(senkou) if senkou and senkou > 0 else 52
    _length = max(tenkan, kijun, senkou)
    high = high
    low = low
    close = close
    offset = (offset)

    if high is None or low is None or close is None: return None, None

    # Calculate Result
    tenkan_sen = midprice(high=high, low=low, length=tenkan)
    kijun_sen = midprice(high=high, low=low, length=kijun)
    span_a = 0.5 * (tenkan_sen + kijun_sen)
    span_b = midprice(high=high, low=low, length=senkou)

    # Copy Span A and B values before their shift
    _span_a = span_a[-kijun:].copy()
    _span_b = span_b[-kijun:].copy()

    span_a = span_a.shift(kijun)
    span_b = span_b.shift(kijun)
    chikou_span = close.shift(-kijun)

    # Offset
    if offset != 0:
        tenkan_sen = tenkan_sen.shift(offset)
        kijun_sen = kijun_sen.shift(offset)
        span_a = span_a.shift(offset)
        span_b = span_b.shift(offset)
        chikou_span = chikou_span.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        span_a.fillna(kwargs["fillna"], inplace=True)
        span_b.fillna(kwargs["fillna"], inplace=True)
        chikou_span.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        span_a.fillna(method=kwargs["fill_method"], inplace=True)
        span_b.fillna(method=kwargs["fill_method"], inplace=True)
        chikou_span.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    span_a.name = f"ISA_{tenkan}"
    span_b.name = f"ISB_{kijun}"
    tenkan_sen.name = f"ITS_{tenkan}"
    kijun_sen.name = f"IKS_{kijun}"
    chikou_span.name = f"ICS_{kijun}"

    chikou_span.category = kijun_sen.category = tenkan_sen.category = "trend"
    span_b.category = span_a.category = chikou_span

    # Prepare Ichimoku DataFrame
    data = {
        span_a.name: span_a,
        span_b.name: span_b,
        tenkan_sen.name: tenkan_sen,
        kijun_sen.name: kijun_sen,
        chikou_span.name: chikou_span,
    }
    ichimokudf = pd.DataFrame(data)
    ichimokudf.name = f"ICHIMOKU_{tenkan}_{kijun}_{senkou}"
    ichimokudf.category = "overlap"

    # Prepare Span DataFrame
    last = close.index[-1]
    if close.index.dtype == "int64":
        ext_index = pd.RangeIndex(start=last + 1, stop=last + kijun + 1)
        spandf = pd.DataFrame(index=ext_index, columns=[span_a.name, span_b.name])
        _span_a.index = _span_b.index = ext_index
    else:
        df_freq = close.index.value_counts().mode()[0]
        tdelta = pd.Timedelta(df_freq, unit="d")
        new_dt = pd.date_range(start=last + tdelta, periods=kijun, freq="B")
        spandf = pd.DataFrame(index=new_dt, columns=[span_a.name, span_b.name])
        _span_a.index = _span_b.index = new_dt

    spandf[span_a.name] = _span_a
    spandf[span_b.name] = _span_b
    spandf.name = f"ICHISPAN_{tenkan}_{kijun}"
    spandf.category = "overlap"

    return ichimokudf, spandf



def rsi(close, length=14, scalar=None, drift=None, offset=None, **kwargs):
    """Indicator: Relative Strength Index (RSI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    scalar = float(scalar) if scalar else 100
    close = close
    drift = 1
    offset = 0
          
    if close is None: return

    # Calculate Result
    negative = close.diff(drift)
    positive = negative.copy()

    positive[positive < 0] = 0  # Make negatives 0 for the postive series
    negative[negative > 0] = 0  # Make postives 0 for the negative series

    positive_avg = rma(positive, length=length)
    negative_avg = rma(negative, length=length)

    rsi = scalar * positive_avg / (positive_avg + negative_avg.abs())

    # Offset
    if offset != 0:
        rsi = rsi.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        rsi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rsi.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    rsi.name = f"RSI_{length}"
    rsi.category = "momentum"

    return rsi


def supertrend(data, length=None, multiplier=None, offset=None, **kwargs):
    
    """Indicator: Supertrend"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 7
    multiplier = float(multiplier) if multiplier and multiplier > 0 else 3.0
    high = data['High']
    low = data['Low']
    close = data['Close']
    offset =offset

    if high is None or low is None or close is None: return

    # Calculate Results
    m = close.size
    dir_, trend = [1] * m, [0] * m
    long, short = [np.nan] * m, [np.nan] * m

    hl2_ = (high+low)/2
    matr = multiplier * calculate_atr(data, length)
    upperband = hl2_ + matr
    lowerband = hl2_ - matr

    for i in range(1, m):
        if close.iloc[i] > upperband.iloc[i - 1]:
            dir_[i] = 1
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if dir_[i] < 0 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lowerband.iloc[i]
        else:
            trend[i] = short[i] = upperband.iloc[i]

    # Prepare DataFrame to return
    _props = f"_{length}_{multiplier}"
    df = pd.DataFrame({
            f"SUPERT{_props}": trend,
            f"SUPERTd{_props}": dir_,
            f"SUPERTl{_props}": long,
            f"SUPERTs{_props}": short,
        }, index=close.index)

    df.name = f"SUPERT{_props}"
    df.category = "overlap"

    # Apply offset if needed
    """if offset != 0:
        df = df.shift(offset)"""

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)

    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    return df


def ha(open_, high, low, close, offset=None, **kwargs):
    """Heikin Ashi Candles

    Heikin Ashi is a candlestick charting technique used to smooth price data and identify trends.
    It calculates modified open, high, low, and close prices based on the input OHLC data.

    Calculation:
        HA_close = (open + high + low + close) / 4
        HA_open[0] = (open[0] + close[0]) / 2
        HA_open[i] = (HA_open[i-1] + HA_close[i-1]) / 2 for i > 0
        HA_high = max(HA_open, HA_high, HA_close)
        HA_low = min(HA_open, HA_low, HA_close)

    Args:
        open_ (pd.Series): Series of opening prices
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of closing prices
        offset (int): Number of periods to shift the result. Default: 0

    Kwargs:
        fillna (value, optional): Value to fill NaNs with
        fill_method (str, optional): Method to fill NaNs (e.g., 'ffill', 'bfill')

    Returns:
        pd.DataFrame: DataFrame with HA_open, HA_high, HA_low, HA_close columns
    """
    # Validate Arguments
    def verify_series(obj):
        if obj is None or not isinstance(obj, pd.Series):
            raise ValueError("Input must be a pandas Series")
        return obj

    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)

    if not (open_.index.equals(high.index) and high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("All input series must have the same index.")

    offset = int(offset) if offset is not None else 0

    # Calculate Result
    m = close.size
    df = pd.DataFrame(index=close.index)
    df["HA_close"] = 0.25 * (open_ + high + low + close)
    df["HA_open"] = np.nan
    df["HA_open"].iloc[0] = 0.5 * (open_.iloc[0] + close.iloc[0])

    # Calculate HA_open for subsequent rows
    for i in range(1, m):
        df["HA_open"].iloc[i] = 0.5 * (df["HA_open"].iloc[i - 1] + df["HA_close"].iloc[i - 1])

    df["HA_high"] = high
    df["HA_low"] = low
    df["HA_high"] = df[["HA_open", "HA_high", "HA_close"]].max(axis=1)
    df["HA_low"] = df[["HA_open", "HA_low", "HA_close"]].min(axis=1)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize
    df.name = "Heikin-Ashi"
    df.category = "candles"
    return df


def cci(high, low, close, length=None, c=None, offset=None, **kwargs):
    """Indicator: Commodity Channel Index (CCI)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    c = float(c) if c and c > 0 else 0.015
    high = high
    low = low
    close = close
    offset = offset

    if high is None or low is None or close is None: return
    # Calculate Result
    typical_price =  (high + low + close) / 3
    mean_typical_price = sma(typical_price, length=length)
    mad_typical_price = close.rolling(length).apply(lambda series: npfabs(series - series.mean()).mean(), raw=True)

    cci = typical_price - mean_typical_price
    cci /= c * mad_typical_price

    # Offset
    if offset != 0:
        cci = cci.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        cci.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cci.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    cci.name = f"CCI_{length}_{c}"
    cci.category = "momentum"

    return cci
import pandas as pd
import numpy as np

def non_zero_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Returns the difference of two series and adds epsilon to any zero values. This occurs commonly in crypto data when 'high' = 'low'."""
    diff = high - low
    if diff.eq(0).any().any():
        diff += np.finfo(float).eps  # Use np.finfo(float).eps instead of sflt.epsilon
    return diff

def stoch(high, low, close, k=None, d=None, smooth_k=None, offset=None, **kwargs):
    """Indicator: Stochastic Oscillator (STOCH)"""
    # Validate arguments
    k = int(k) if k and k > 0 else 14
    d = int(d) if d and d > 0 else 3
    smooth_k = int(smooth_k) if smooth_k and smooth_k > 0 else 3
    offset = int(offset) if offset is not None and offset != 0 else 0  # Validate offset
    _length = max(k, d, smooth_k)

    if high is None or low is None or close is None:
        return

    # Calculate Result
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()

    stoch = 100 * (close - lowest_low)
    stoch /= non_zero_range(highest_high, lowest_low)

    stoch_k = sma(stoch, length=smooth_k)
    stoch_d = sma(stoch_k, length=d)

    # Offset
    if offset != 0:
        stoch_k = stoch_k.shift(offset)
        stoch_d = stoch_d.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        stoch_k.fillna(kwargs["fillna"], inplace=True)
        stoch_d.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stoch_k.fillna(method=kwargs["fill_method"], inplace=True)
        stoch_d.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _name = "STOCH"
    _props = f"_{k}_{d}_{smooth_k}"
    stoch_k.name = f"{_name}k{_props}"
    stoch_d.name = f"{_name}d{_props}"
    stoch_k.category = stoch_d.category = "momentum"

    # Prepare DataFrame to return
    data = {stoch_k.name: stoch_k, stoch_d.name: stoch_d}
    df = pd.DataFrame(data)
    df.name = f"{_name}{_props}"
    df.category = stoch_k.category

    return df

def mad(close, length=None, offset=None, **kwargs):
    """Indicator: Mean Absolute Deviation"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    close = close
    offset = offset

    if close is None: return

    # Calculate Result
    def mad_(series):
        """Mean Absolute Deviation"""
        return npfabs(series - series.mean()).mean()

    mad = close.rolling(length, min_periods=min_periods).apply(mad_, raw=True)

    # Offset
    if offset != 0:
        mad = mad.shift(offset)

    # Name & Category
    mad.name = f"MAD_{length}"
    mad.category = "statistics"

    return mad


def kama(price, length=21, fast_sc=0.666, slow_sc=0.0645):
    """
    Pine Script tarzında Kaufman Hareketli Ortalama (KAMA) hesaplar.
    
    Parametreler:
    price: Fiyat verileri (pandas Series veya numpy array)
    length: Verimlilik oranı için bakılacak dönem (varsayılan: 21)
    fast_sc: Hızlı hareketli ortalama sabiti (varsayılan: 0.666)
    slow_sc: Yavaş hareketli ortalama sabiti (varsayılan: 0.0645)
    
    Dönen değer:
    KAMA değerlerini içeren pandas Series
    """
    # Fiyat verilerini numpy array'e çevir
    price = np.array(price)
    
    # Gürültü (xvnoise) hesaplama
    xvnoise = np.abs(price[1:] - price[:-1])
    
    # Sinyal ve gürültü hesaplama
    nsignal = np.abs(price[length:] - price[:-length])
    nnoise = np.zeros(len(price) - length)
    for i in range(len(nnoise)):
        nnoise[i] = np.sum(xvnoise[i:i+length])
    
    # Verimlilik oranı (Efficiency Ratio - ER)
    nefratio = np.where(nnoise != 0, nsignal / nnoise, 0)
    
    # Yumuşatma katsayısı (Smoothing Constant - SC)
    nsmooth = np.power(nefratio * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA hesaplama
    kama = np.zeros_like(price)
    kama[:length] = price[:length]  # İlk dönem için fiyatları kullan
    
    for i in range(length, len(price)):
        kama[i] = kama[i-1] + nsmooth[i-length] * (price[i] - kama[i-1])
    
    # NaN değerleri temizle ve pandas Series olarak döndür
    kama = pd.Series(kama, index=price.index if isinstance(price, pd.Series) else None)
    return kama


def bband(length=20,ma_Type:str=None,src:pd.Series=None,mult:float = 2,offset:int=0) -> tuple:
    """
    Inputs: 
        length = int
        ma_type = str
        src = pd.Series
        mult = float
        offset = int, default 0 
    Output:
        basis = float
        lower_band = float
        upper_band = float
    """
    if ma_Type is None:
        raise
    if not isinstance(src, pd.Series):
        raise
    try:
        if ma_Type=='SMA':
            ma = sma(src,length=length,offset=offset)
        if ma_Type == 'EMA':
            ma = ema(src, length=length,offset=offset)
        basis = ma
        dev = mult* src.rolling(length).std(ddof=0)
        upper = basis + dev
        lower = basis - dev
        return basis,upper,lower
    except:
        raise
        