from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANUAL_DIR = DATA_DIR / "manual"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
PRICE_CACHE_CSV = MANUAL_DIR / "price_cache.csv"

FREEZE_DATE = "2026-08-11"
TRACK1_START = "2023-01-01"
TRACK1_END = "2025-04-13"
PRICE_START = "2022-11-01"
# Track1(이벤트 스터디)은 TRACK1_END까지만 쓰지만, Track2(케이스 스터디, 예: 2025년 6월
# Trump-Musk 결별)는 그 이후 날짜의 가격 반응이 필요하므로 가격 다운로드 범위는 더 넓게 둔다.
PRICE_END = "2026-08-20"
MUSK_TWITTER_ACQUISITION = "2022-10-27"

TICKERS = ["TSLA", "SPY", "QQQ", "GM", "F", "RIVN"]
PEER_TICKERS = {"TSLA": ["GM", "F", "RIVN"]}
MARKET_PROXY = {"TSLA": "QQQ", "QQQ": "SPY", "SPY": "SPY"}

TRUMP_PRESIDENT_START = "2025-01-20"

# 원본 Musk/Trump CSV의 게시 시각은 UTC(+00:00)입니다. 분석 화면과 거래일
# 정렬에는 미국 동부시간을 사용합니다. 정규장은 09:30~16:00으로 두되,
# 조기폐장일은 별도 거래소 캘린더가 필요한 후속 보완 항목입니다.
SOURCE_TIMEZONE = "UTC"
MARKET_TIMEZONE = "America/New_York"
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Track 2 수동 사건 CSV에서 timezone offset이 생략된 시각은 미국 동부시간으로
# 입력했다고 해석합니다. offset이 포함된 값은 해당 offset을 그대로 존중합니다.
TRACK2_DEFAULT_TIMEZONE = MARKET_TIMEZONE

OUTPUT_FOLDERS = [
    INTERIM_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]


def ensure_output_folders() -> None:
    for folder in OUTPUT_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)
