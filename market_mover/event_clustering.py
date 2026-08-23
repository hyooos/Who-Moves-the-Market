import json

import pandas as pd


DEFAULT_CLUSTER_HOURS = 6.0
CLUSTER_GROUP_COLUMNS = ["person", "ticker", "topic", "event_date"]


def _json_values(series: pd.Series) -> str:
    values = []
    for value in series.tolist():
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            values.append(None)
        else:
            values.append(str(value))
    return json.dumps(values, ensure_ascii=False)


def _fixed_window_labels(timestamps: pd.Series, window_hours: float) -> list[int]:
    """첫 게시물 기준 고정 창으로 cluster 번호를 만듭니다.

    직전 게시물과의 간격만 보면 5시간 간격 게시물이 계속 이어질 때 하나의 사건이
    며칠까지 늘어나는 chaining 문제가 생깁니다. 그래서 각 cluster의 첫 게시물로부터
    ``window_hours`` 이내인 게시물만 같은 사건으로 묶습니다.
    """
    labels = []
    cluster_no = -1
    cluster_start = None
    for timestamp in timestamps:
        if cluster_start is None or (timestamp - cluster_start).total_seconds() > window_hours * 3600:
            cluster_no += 1
            cluster_start = timestamp
        labels.append(cluster_no)
    return labels


def cluster_daily_events(events: pd.DataFrame, window_hours: float = DEFAULT_CLUSTER_HOURS) -> pd.DataFrame:
    """같은 화자·ticker·topic·반응 거래일의 연속 게시물을 하나의 사건으로 묶습니다.

    일봉 가격은 같은 ticker·event_date에 하나뿐이므로 같은 캠페인의 연속 글을 각각
    독립 관측치로 세면 pseudo-replication이 생깁니다. cluster는 원문을 버리지 않고
    JSON 목록 컬럼에 모든 게시물 ID·시각·본문·URL을 보존합니다.

    ``window_hours <= 0``이면 묶지 않고 모든 게시물을 singleton 사건으로 남겨
    민감도 비교에 사용할 수 있습니다.
    """
    if events.empty:
        return events.copy()

    required = set(CLUSTER_GROUP_COLUMNS + ["event_id", "post_id", "posted_at"])
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"이벤트 clustering에 필요한 컬럼이 없습니다: {missing}")

    out = events.copy()
    time_source = "posted_at_utc" if "posted_at_utc" in out.columns else "posted_at"
    out["_cluster_time"] = pd.to_datetime(out[time_source], errors="coerce", utc=True)
    if out["_cluster_time"].isna().any():
        raise ValueError("게시 시각이 없는 이벤트가 있어 clustering할 수 없습니다.")
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    out = out.sort_values(CLUSTER_GROUP_COLUMNS + ["_cluster_time", "event_id"]).copy()
    if window_hours <= 0:
        out["_cluster_no"] = out.groupby(CLUSTER_GROUP_COLUMNS, dropna=False).cumcount()
    else:
        pieces = []
        for _, group in out.groupby(CLUSTER_GROUP_COLUMNS, dropna=False, sort=False):
            group = group.copy()
            group["_cluster_no"] = _fixed_window_labels(group["_cluster_time"], float(window_hours))
            pieces.append(group)
        out = pd.concat(pieces, ignore_index=False)

    cluster_keys = CLUSTER_GROUP_COLUMNS + ["_cluster_no"]
    records = []
    for cluster_index, (_, members) in enumerate(
        out.groupby(cluster_keys, dropna=False, sort=True), start=1
    ):
        members = members.sort_values(["_cluster_time", "event_id"])
        representative = members.iloc[0]
        record = representative.drop(labels=["_cluster_time", "_cluster_no"]).to_dict()
        record["representative_event_id"] = representative["event_id"]
        record["event_id"] = f"tk1c_{cluster_index:06d}"
        record["cluster_size"] = int(len(members))
        record["is_clustered_event"] = bool(len(members) > 1)
        record["cluster_window_hours"] = float(window_hours)
        record["cluster_start_utc"] = members["_cluster_time"].iloc[0]
        record["cluster_end_utc"] = members["_cluster_time"].iloc[-1]
        record["cluster_duration_minutes"] = float(
            (members["_cluster_time"].iloc[-1] - members["_cluster_time"].iloc[0]).total_seconds() / 60
        )
        if "engagement" in members.columns:
            record["engagement"] = members["engagement"].sum(min_count=1)
        record["member_event_ids_json"] = _json_values(members["event_id"])
        record["member_post_ids_json"] = _json_values(members["post_id"])
        for source, target in [
            ("source_post_id", "member_source_post_ids_json"),
            ("source_url", "member_source_urls_json"),
            ("posted_at_et", "member_posted_at_et_json"),
            ("text_raw", "member_texts_raw_json"),
            ("text_clean", "member_texts_clean_json"),
            ("market_session", "member_market_sessions_json"),
        ]:
            if source in members.columns:
                record[target] = _json_values(members[source])
        if "text_raw" in members.columns:
            record["cluster_text_raw"] = "\n\n".join(members["text_raw"].fillna("").astype(str))
        record["cluster_text_clean"] = "\n\n".join(members["text_clean"].fillna("").astype(str))
        records.append(record)

    clustered = pd.DataFrame(records)
    print(
        f"[클러스터] {len(events)}개 게시물 -> {len(clustered)}개 사건 "
        f"(같은 화자/ticker/topic/거래일, 첫 글 기준 {window_hours:g}시간 창)"
    )
    return clustered.reset_index(drop=True)
