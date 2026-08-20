from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if "Ticker" in df.columns.names:
        df = df.stack(level="Ticker").reset_index()
    else:
        df = df.stack(level=-1).reset_index()
    return df


def download_daily_prices(
    tickers: Optional[list] = None,
    start: str = config.PRICE_START,
    end: str = config.PRICE_END,
) -> pd.DataFrame:
    import yfinance as yf

    tickers = tickers or config.TICKERS
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("yfinance가 빈 가격 데이터를 반환했습니다.")
    raw = _flatten_yfinance_columns(raw)
    rename = {
        "Date": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    raw = raw.rename(columns=rename)
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"다운로드한 가격 데이터에 필요한 컬럼이 없습니다: {missing}")
    prices = raw[required].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.dropna(subset=["close"]).sort_values(["ticker", "date"])
    return prices


def _normalize_price_cache(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path)
    rename = {
        "Date": "date",
        "Datetime": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    prices = prices.rename(columns={k: v for k, v in rename.items() if k in prices.columns})
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in prices.columns]
    if missing:
        raise ValueError(f"{path} 가격 캐시에 필요한 컬럼이 없습니다: {missing}")
    prices = prices[required].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices[prices["ticker"].isin(config.TICKERS)]
    return prices.dropna(subset=["close"]).sort_values(["ticker", "date"])


def load_or_download_daily_prices(cache_path: Optional[Path] = None) -> pd.DataFrame:
    cache_path = cache_path or (config.INTERIM_DIR / "daily_prices.csv")
    if cache_path.exists():
        prices = pd.read_csv(cache_path, parse_dates=["date"])
        return prices
    try:
        prices = download_daily_prices()
    except Exception as exc:
        if config.PRICE_CACHE_CSV.exists():
            print(f"[가격] yfinance 수집 실패: {exc}")
            print(f"[가격] 수동 캐시를 사용합니다: {config.PRICE_CACHE_CSV}")
            prices = _normalize_price_cache(config.PRICE_CACHE_CSV)
        else:
            raise RuntimeError(
                "가격 데이터를 가져올 수 없습니다. 인터넷 연결을 확인하거나 "
                f"{config.PRICE_CACHE_CSV}에 date,ticker,open,high,low,close,volume 형식의 CSV를 준비해주세요."
            ) from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path, index=False)
    return prices


def load_intraday_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "Datetime": "datetime",
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Ticker": "ticker",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = ["datetime", "ticker", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"장중 가격 CSV에 필요한 컬럼이 없습니다: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["ticker", "datetime"])
