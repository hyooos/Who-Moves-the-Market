import numpy as np
import pandas as pd

from . import config


def robust_zscore(series: pd.Series, window: int = 60, baseline_mask: pd.Series = None) -> pd.Series:
    baseline = series.where(baseline_mask) if baseline_mask is not None else series
    median = baseline.rolling(window, min_periods=20).median()
    mad = (baseline - median).abs().rolling(window, min_periods=20).median()
    mad_scaled = mad * 1.4826
    return (series - median) / mad_scaled.replace(0, np.nan)


def _event_exclusion_mask(df: pd.DataFrame, exclude_dates_by_ticker: dict = None) -> pd.Series:
    if not exclude_dates_by_ticker:
        return pd.Series(True, index=df.index)
    excluded = set(pd.to_datetime(exclude_dates_by_ticker.get(df["ticker"].iloc[0], [])).normalize())
    if not excluded:
        return pd.Series(True, index=df.index)
    return ~df["date"].isin(excluded)


def compute_price_features(prices: pd.DataFrame, exclude_dates_by_ticker: dict = None) -> pd.DataFrame:
    frames = []
    returns = prices.pivot(index="date", columns="ticker", values="close").pct_change()

    for ticker, df in prices.groupby("ticker"):
        df = df.sort_values("date").copy()
        market_ticker = config.MARKET_PROXY.get(ticker, "SPY")
        df["stock_return"] = df["close"].pct_change()
        if market_ticker == ticker:
            # SPY(또는 자기 자신을 시장 프록시로 갖는 티커)는 벤치마크로 삼을 상위 지수가
            # 없다. market_return을 자기 자신으로 두면 abnormal_return이 항상 0이 되어버리므로
            # (실제로 이 버그로 SPY 이벤트 58건 전부 abnormal_return=0이 찍혔었음), 이 경우엔
            # 벤치마크 차감 없이 원 수익률 자체를 쓴다.
            df["market_return"] = 0.0
        else:
            df["market_return"] = returns[market_ticker].reindex(df["date"]).to_numpy()
        df["ar_market"] = df["stock_return"] - df["market_return"]

        peers = config.PEER_TICKERS.get(ticker)
        if peers:
            peer_avg = returns[peers].mean(axis=1).reindex(df["date"]).to_numpy()
            df["abnormal_return"] = df["stock_return"] - peer_avg
        else:
            df["abnormal_return"] = df["ar_market"]

        df["volatility"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
        df["log_volume"] = np.log1p(df["volume"])
        df["log_volatility"] = np.log1p(df["volatility"])
        baseline_mask = _event_exclusion_mask(df, exclude_dates_by_ticker)
        df["baseline_excluded"] = ~baseline_mask
        df["z_volume"] = robust_zscore(df["log_volume"], baseline_mask=baseline_mask)
        df["z_volatility"] = robust_zscore(df["log_volatility"], baseline_mask=baseline_mask)
        df["z_ar"] = robust_zscore(df["abnormal_return"], baseline_mask=baseline_mask)
        df["impact_score"] = (
            df["z_ar"].abs()
            + df["z_volume"].clip(lower=0)
            + df["z_volatility"].clip(lower=0)
        )
        df["direction"] = np.sign(df["abnormal_return"])
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
