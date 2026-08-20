import re
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

TEXT_COLUMNS = ["text", "tweet", "content", "body", "message", "full_text"]
DATE_COLUMNS = ["posted_at", "date", "datetime", "created_at", "timestamp", "time"]
LIKE_COLUMNS = ["likes", "favorite_count", "favorites", "like_count"]
RETWEET_COLUMNS = ["retweets", "retweet_count", "reposts", "share_count"]
REPLY_COLUMNS = ["replies", "reply_count", "comments", "comment_count"]
PLATFORM_COLUMNS = ["platform", "source", "site"]
IS_REPLY_COLUMNS = ["is_reply", "isreply"]
REPLY_TO_USERNAME_COLUMNS = ["in_reply_to_username", "inreplytousername"]
DELETED_COLUMNS = ["deleted_flag", "is_deleted"]

# 답글 필터링 예외 계정: 짧은 답글이어도 정책적으로 의미 있다고 EDA에서 확인된 계정.
POLICY_REPLY_ACCOUNTS = {"cb_doge", "dogeofficialceo"}
MIN_REPLY_WORD_COUNT = 5

# musk_quote_tweets.csv는 all_musk_posts.csv에 있는 인용 트윗의 "원본 인용 대상 상세정보"를
# 덧붙인 파일이라, id 기준으로 all_musk_posts.csv와 100% 겹칩니다(EDA에서 확인됨).
# 별도 소스로 읽으면 같은 게시물이 두 번 카운팅되므로 인물 파일 자동 인식에서 제외합니다.
DUPLICATE_SOURCE_FILES = {"musk_quote_tweets.csv"}


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"^RT\s+", "", text)
    text = re.sub(r"[^\w\s.,!?%$'-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_first_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    normalized_to_original = {}
    for col in df.columns:
        normalized_to_original.setdefault(_normalize_column_name(col), col)
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized_to_original:
            return normalized_to_original[key]
    return None


def infer_person(path: Path) -> str:
    name = path.name.lower()
    if "musk" in name or "elon" in name:
        return "Musk"
    if "trump" in name or "donald" in name:
        return "Trump"
    raise ValueError(f"파일명에서 인물을 추정할 수 없습니다: {path.name}")


def infer_platform(row: pd.Series) -> str:
    if row["person"] == "Musk":
        return "X"
    raw_platform = str(row.get("platform_raw", "")).lower()
    if "truth" in raw_platform:
        return "Truth Social"
    if raw_platform in {"x", "twitter"} or "twitter" in raw_platform:
        return "X"
    posted_at = row["posted_at"]
    if posted_at < pd.Timestamp("2021-01-08"):
        return "X"
    if posted_at < pd.Timestamp("2023-08-24"):
        return "Truth Social"
    return "Unknown"


def assign_trump_role(row: pd.Series) -> Optional[str]:
    if row["person"] != "Trump":
        return None
    cutoff = pd.Timestamp(config.TRUMP_PRESIDENT_START)
    return "president" if row["posted_at"] >= cutoff else "candidate_or_citizen"


def normalize_post_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    text_col = find_first_column(df, TEXT_COLUMNS)
    date_col = find_first_column(df, DATE_COLUMNS)
    if text_col is None or date_col is None:
        raise ValueError(
            f"{path.name}에는 텍스트 컬럼 {TEXT_COLUMNS} 중 하나와 날짜 컬럼 {DATE_COLUMNS} 중 하나가 필요합니다. "
            f"현재 컬럼: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["posted_at"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    out["person"] = infer_person(path)
    out["text_raw"] = df[text_col].astype(str)
    out["text_clean"] = out["text_raw"].map(clean_text)

    platform_col = find_first_column(df, PLATFORM_COLUMNS)
    out["platform_raw"] = df[platform_col].astype(str) if platform_col else ""

    for target, candidates in {
        "likes": LIKE_COLUMNS,
        "retweets": RETWEET_COLUMNS,
        "replies": REPLY_COLUMNS,
    }.items():
        col = find_first_column(df, candidates)
        # 결측을 0으로 채우지 않습니다: "반응이 0"과 "값을 못 가져옴"은 다른 의미이고,
        # 0으로 채우면 H4(engagement 상관) 검정이 왜곡됩니다(Musk EDA에서 확인된 원칙).
        # 컬럼 자체가 없는 파일만 0으로 취급합니다(그 지표를 애초에 안 실었다는 뜻이므로).
        out[target] = pd.to_numeric(df[col], errors="coerce") if col else 0.0

    # Trump가 X로 복귀한 2023-08-24 이후 Twitter 게시물의 repost_count는 결측이 0으로
    # 잘못 기록돼 있음(Trump EDA에서 확인, Track1 구간 내 Twitter 536행 전부 0). 실제 값이
    # 아니라 수집 결함이므로 0이 아니라 NaN으로 되돌려 engagement 계산에서 빠지게 합니다.
    platform_raw_lower = out["platform_raw"].str.lower()
    is_trump_broken_retweets = (
        (infer_person(path) == "Trump")
        & platform_raw_lower.str.contains("twitter", na=False)
        & (out["posted_at"] >= pd.Timestamp("2023-08-24"))
    )
    out.loc[is_trump_broken_retweets, "retweets"] = pd.NA

    out["engagement"] = out["likes"] + out["retweets"] + out["replies"]
    out = out.dropna(subset=["posted_at", "text_clean"])
    out["platform"] = out.apply(infer_platform, axis=1)
    out["trump_role"] = out.apply(assign_trump_role, axis=1)

    deleted_col = find_first_column(df, DELETED_COLUMNS)
    out["is_deleted"] = df.loc[out.index, deleted_col].fillna(False).astype(bool) if deleted_col else False

    is_reply_col = find_first_column(df, IS_REPLY_COLUMNS)
    reply_to_col = find_first_column(df, REPLY_TO_USERNAME_COLUMNS)
    if is_reply_col is not None:
        is_reply = df.loc[out.index, is_reply_col].fillna(False).astype(bool)
        reply_to_username = (
            df.loc[out.index, reply_to_col].fillna("").astype(str).str.lower()
            if reply_to_col is not None
            else pd.Series("", index=out.index)
        )
        word_count = out["text_clean"].str.split().str.len().fillna(0)
        # EDA 결론: 답글의 71%가 짧은 감탄사형이라 5단어 미만 답글은 노이즈로 보고 제외하되,
        # DOGE 관련 계정(cb_doge, dogeofficialceo)에 대한 답글은 길이와 상관없이 정책적으로
        # 의미가 있다고 판단해 항상 포함한다.
        is_policy_reply = reply_to_username.isin(POLICY_REPLY_ACCOUNTS)
        drop_short_reply = is_reply & (word_count < MIN_REPLY_WORD_COUNT) & ~is_policy_reply
        dropped = int(drop_short_reply.sum())
        if dropped:
            print(f"[게시물] {path.name}: 짧은 감탄사형 답글 {dropped}건 제외 (5단어 미만, 정책 계정 제외)")
        out = out[~drop_short_reply]

    return out


def load_all_posts(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    person_files = [
        path for path in files
        if any(token in path.name.lower() for token in ["musk", "elon", "trump", "donald"])
        and path.name not in DUPLICATE_SOURCE_FILES
    ]
    ignored = [path.name for path in files if path not in person_files]
    for name in ignored:
        if name in DUPLICATE_SOURCE_FILES:
            print(f"[게시물] {name}는 다른 파일과 중복이라 건너뜀 (id 기준 100% 겹침)")
        else:
            print(f"[게시물] 인물 인식 불가로 건너뜀: {name}")
    if not person_files:
        raise FileNotFoundError(
            f"{raw_dir}에서 Musk/Trump CSV를 찾지 못했습니다. Kaggle 파일을 data/raw/에 먼저 넣어주세요."
        )
    frames = []
    found_people = set()
    for path in person_files:
        person = infer_person(path)
        try:
            frame = normalize_post_file(path)
        except ValueError as exc:
            print(f"[게시물] {path.name} 스키마 인식 실패로 건너뜀: {exc}")
            continue
        print(f"[게시물] {path.name} -> {person} ({len(frame)}행)")
        frames.append(frame)
        found_people.add(person)
    if not frames:
        raise ValueError("모든 CSV의 스키마 인식에 실패했습니다. data/raw/ 파일 컬럼을 확인하세요.")
    missing = {"Musk", "Trump"} - found_people
    if missing:
        print(f"[게시물] 경고: 다음 인물의 CSV가 없습니다: {sorted(missing)}")
    posts = pd.concat(frames, ignore_index=True)
    posts = posts.sort_values("posted_at").reset_index(drop=True)
    return posts
