import json
import os
from pathlib import Path

import pandas as pd
import requests

# 인물 이름은 화면 전체에서 같은 한국어 표기를 사용한다.
_PERSON_KO = {"Musk": "일론 머스크", "Trump": "도널드 트럼프"}
DEFAULT_LLM_MODELS = {
    "gemini": "gemini-1.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b",
}


def fetch_article_markdown(url: str, timeout: int = 20) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    reader_url = f"https://r.jina.ai/{url.strip()}"
    response = requests.get(reader_url, timeout=timeout)
    response.raise_for_status()
    return response.text


def build_price_context(event: dict) -> dict:
    return {
        "abnormal_return": event.get("abnormal_return"),
        "impact_score": event.get("impact_score"),
        "volume_z": event.get("z_volume"),
        "contamination_level": event.get("contamination_level") or "Track 2: 통계 오염 분류 미적용",
    }


def fallback_narrative(event: dict, reason: str) -> str:
    description = event.get("description") or event.get("text_clean") or "설명 없음"
    return (
        f"판단보류: 자동 내러티브 생성에 필요한 근거가 충분하지 않습니다. "
        f"수동 설명은 다음과 같습니다: {description} "
        f"자동 생성 생략 사유: {reason}"
    )


def build_narrative_prompt(event: dict, article_md: str, price_context: dict) -> str:
    ar = price_context.get("abnormal_return")
    ar_text = "계산 불가" if pd.isna(ar) else f"{ar:.2%}"
    impact = price_context.get("impact_score")
    impact_text = "계산 불가" if pd.isna(impact) else f"{impact:.2f}"
    person_ko = _PERSON_KO.get(str(event.get("person")), str(event.get("person")))

    return f"""다음은 SNS 게시물 또는 공개 발언과 그 직후 시장 반응에 대한 정보입니다.

이벤트 정보:
- 인물: {person_ko}
- 게시/발언 시각(미국 동부시간 우선): {event.get('posted_at_et') or event.get('posted_at')}
- 주제: {event.get('topic')}
- 종목/지수: {event.get('ticker')}
- 수동 설명: {event.get('description') or event.get('text_clean') or ''}

관련 뉴스 기사 발췌:
{article_md[:3000]}

가격 반응:
- 초과수익률: {ar_text}
- 영향 점수: {impact_text}
- 오염 수준: {price_context.get('contamination_level')}

위 정보를 바탕으로, 이 이벤트 직후 관측된 시장 반응을 한국어 2~3문장으로 요약하세요.
반드시 지킬 것:
- 오직 한국어로만 답할 것. 중국어, 영어 문장이나 단어를 절대 섞지 말 것 (Respond in Korean only. Do not mix in Chinese or English sentences.)
- 인물 이름은 반드시 "{person_ko}"라고만 쓸 것("무스크"처럼 다르게 표기하지 말 것). 이 이벤트의 주체는 반드시 "{person_ko}"이다. 기사에 다른 인물이 함께 언급되더라도 절대 혼동하지 말고, 발언/행동의 주어는 항상 "{person_ko}"으로 쓸 것
- "이 발언이 하락을 야기했다"처럼 인과관계를 단정하지 말 것
- "직후 이례적인 반응이 관측됐다"처럼 관찰 진술로 표현할 것
- 기사 근거가 부족하거나 가격 맥락이 약하면 "판단보류"라고 명시할 것
- 과장된 투자 조언을 하지 말 것

한국어 답변:"""


def generate_with_gemini(prompt: str, model_name: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수가 없습니다.")

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai 패키지가 설치되어 있지 않습니다.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_with_groq(prompt: str, model_name: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경변수가 없습니다.")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": "너는 한국어 금융 이벤트 스터디 보조 분석가다. 인과관계를 단정하지 않는다."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


_CHINESE_MARKERS = ("的", "了", "是", "，", "他", "这", "在", "和", "对")


def _looks_like_chinese(text: str) -> bool:
    if sum(text.count(marker) for marker in _CHINESE_MARKERS) >= 2:
        return True
    # 응답 전체가 중국어가 아니라 한두 단어만 섞여 들어온 경우도 잡는다 — 한국어는
    # 한자를 거의 안 쓰므로, CJK 한자(U+4E00~U+9FFF)가 조금이라도 있으면 의심한다.
    han_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
    return han_chars >= 2


def _call_ollama(prompt: str, model_name: str, base_url: str) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model_name,
            "system": "너는 한국어로만 답하는 금융 이벤트 분석가다. 중국어, 영어 단어를 섞지 말고 자연스러운 한국어 문장으로만 답한다.",
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_with_ollama(prompt: str, model_name: str, max_attempts: int = 3) -> str:
    # Qwen 계열 소형 모델은 system 프롬프트로 한국어를 지정해도 가끔 중국어로 통째로
    # 답하는 경우가 관측됨(예: tk2_003). 감지되면 같은 요청을 재시도한다.
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    last_response = ""
    for _ in range(max_attempts):
        last_response = _call_ollama(prompt, model_name, base_url)
        if not _looks_like_chinese(last_response):
            return last_response
    raise RuntimeError(f"{max_attempts}회 재시도에도 중국어가 섞여서 반환됨: {last_response[:80]}")


def generate_text(prompt: str, provider: str = "ollama", model_name: str = None) -> str:
    """대시보드와 batch pipeline이 같은 LLM provider 선택 로직을 사용하게 합니다."""
    provider = (provider or "none").lower()
    model_name = model_name or DEFAULT_LLM_MODELS.get(provider)
    if provider == "gemini":
        return generate_with_gemini(prompt, model_name)
    if provider == "groq":
        return generate_with_groq(prompt, model_name)
    if provider == "ollama":
        return generate_with_ollama(prompt, model_name)
    if provider == "none":
        raise RuntimeError("LLM provider가 꺼져 있습니다.")
    raise ValueError(f"지원하지 않는 LLM provider입니다: {provider}")


def generate_case_narrative(
    event: dict,
    article_md: str,
    price_context: dict,
    provider: str = "gemini",
    model_name: str = None,
) -> str:
    prompt = build_narrative_prompt(event, article_md, price_context)
    provider = provider.lower()
    try:
        if provider == "none":
            return fallback_narrative(event, "LLM provider가 none으로 설정되었습니다.")
        return generate_text(prompt, provider=provider, model_name=model_name)
    except Exception as exc:
        return fallback_narrative(event, f"{provider} 생성 실패: {exc}")


def build_case_narratives(
    events: pd.DataFrame,
    output_path: Path,
    provider: str = "gemini",
    model_name: str = None,
) -> pd.DataFrame:
    track2 = events[events["track"].eq("track2_manual")].copy()
    if track2.empty:
        narratives = pd.DataFrame(columns=["event_id", "narrative"])
        narratives.to_csv(output_path, index=False)
        return narratives

    rows = []
    for _, row in track2.iterrows():
        event = row.to_dict()
        try:
            article_md = fetch_article_markdown(event.get("source_url", ""))
        except Exception as exc:
            article_md = ""
            fetch_error = f"기사 수집 실패: {exc}"
        else:
            fetch_error = ""

        if not article_md:
            narrative = fallback_narrative(event, fetch_error or "source_url이 비어 있거나 기사 본문을 가져오지 못했습니다.")
        else:
            narrative = generate_case_narrative(
                event,
                article_md,
                build_price_context(event),
                provider=provider,
                model_name=model_name,
            )
        rows.append({"event_id": event.get("event_id"), "narrative": narrative, "narrative_provider": provider})

    narratives = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    narratives.to_csv(output_path, index=False)
    return narratives


def _size_label(pct) -> str:
    """백분위 숫자를 사람이 읽는 쉬운 말로 바꾼다. LLM 프롬프트에 '백분위'라는
    단어 자체를 안 주기 위해 파이썬에서 미리 문구로 변환해둔다."""
    if pct is None:
        return "비교할 과거 사례가 부족해 판단하기 어려운"
    if pct >= 0.9:
        return "이 종목에서 보기 드물게 큰"
    if pct >= 0.7:
        return "평소보다 꽤 큰"
    if pct >= 0.3:
        return "평소와 비슷한"
    if pct >= 0.1:
        return "평소보다 작은"
    return "이 종목에서 보기 드물게 작은"


def build_dashboard_commentary_prompt(event: dict, ticker_stats: dict, content_text: str = None) -> str:
    """대시보드에서 클릭한 이벤트 하나에 대한 짧은 해설 프롬프트.

    이 프로젝트는 §5-3에서 topic 내용이 반응 크기를 거의 설명하지 못하고(topic
    순수 추가 설명력 0.3%p), 종목(ticker) 자체의 변동성이 대부분을 설명한다는 걸
    이미 확인했다. §5-3-3의 예측 실험(out-of-sample R²)도 같은 결론을 재확인했다.
    그래서 이 프롬프트는 "이 발언이 크게 움직일 것이다"·"다음엔 이렇게 대응하라" 같은
    예측·조언을 절대 만들지 말고, 이 종목/topic의 과거 CLEAN 표본 통계를 있는
    그대로(회고적으로만) 설명하도록 강하게 제약한다.

    content_text가 주어지면(Track2 기사 원문 등) 요약 대상은 반드시 그 텍스트로
    한정한다 — 요약 문단이 메타데이터가 아니라 실제 원문 내용을 요약하게 만들기
    위함이다.
    """
    ar = event.get("abnormal_return")
    ar_text = "계산 불가" if ar is None or pd.isna(ar) else f"{ar:.2%}"
    pct = ticker_stats.get("percentile")
    pct_text = "계산 불가" if pct is None else f"{pct*100:.0f}%"
    body = content_text or event.get("text_clean") or event.get("description") or ""
    # "Musk"를 LLM이 스스로 한글로 옮기게 두면 "무스크"처럼 잘못 표기하는 경우가 있어서,
    # 표준 한글 표기를 미리 정해서 넣어준다.
    person_ko = _PERSON_KO.get(str(event.get("person")), str(event.get("person")))
    topic_mean = ticker_stats.get("topic_mean_abs_ar")
    contamination = event.get("contamination_level")
    # "CLEAN"은 텍스트 정제 여부가 아니라 "다른 요인과 안 섞여서 원인을 특정할 수 있는가"를
    # 뜻하는 방법론 용어라(contamination.py, §4-2), LLM에는 그 의미를 그대로 문장으로 풀어서
    # 준다 — 그래야 LLM도 "CLEAN"이라는 단어를 답변에 그대로 안 쓴다.
    _CONTAM_DESC = {
        "CLEAN": "다른 요인과 섞이지 않아 이 게시물 하나의 영향으로 볼 수 있는 경우",
        "MINOR": "다른 요인과 약간 섞여 있어 이 게시물 하나의 영향이라고 단정하기 조심스러운 경우",
        "MAJOR": "다른 요인(같은 날 다른 게시물, 매크로 이벤트, 시장 전체 충격 등)과 많이 섞여 있어 이 게시물 하나의 영향이라고 보기 어려운 경우",
    }
    contamination_text = (
        "2025년 4월 이후 뉴스로 정리한 사건(다른 요인과 겹쳤는지는 개별 확인 필요)"
        if contamination is None or (isinstance(contamination, float) and pd.isna(contamination))
        else _CONTAM_DESC.get(str(contamination), str(contamination))
    )

    size_label = _size_label(pct)

    return f"""다음은 대시보드에서 사용자가 클릭한 게시물/기사 1건과, 그 종목의 과거 가격 반응 정보입니다.
읽는 사람은 통계 용어를 모르는 일반 사용자입니다. "CLEAN 표본", "백분위", "impact_score"
같은 통계·분석 전문용어는 **절대 쓰지 말고**, 아래 문구를 그대로 참고해서 쉬운 말로 풀어 쓰세요.

게시물 메타데이터(요약 대상 아님, 분석에만 참고):
- 인물: {person_ko}(반드시 이 표기 그대로 쓸 것 — "무스크"처럼 다르게 쓰지 말 것)
- 게시 시각(미국 동부시간 우선): {event.get('posted_at_et') or event.get('posted_at')}
- 규칙 기반 분류 topic: {event.get('topic')}
- 매핑된 종목: {event.get('ticker')}
- 이 게시물 다음 거래일의 실제 주가 변화: {ar_text}
- 오염 수준(다중게시/매크로/시장충격 여부): {contamination_text}

요약해야 할 실제 원문/기사 본문(이것만 요약할 것):
{body[:2500]}

이 종목의 평소 가격 변화 정보(이 프로젝트가 실제로 계산한 값, 추측 아님 — 이 문구를 쉬운 말로 풀어서 쓸 것):
- 이 종목은 평소(과거 비슷한 사례들에서) 하루에 보통 {ticker_stats.get('mean_abs_ar')} 정도 움직인다.
- 이번 반응은 이 종목이 과거에 보인 반응들과 비교하면 "{size_label}" 편이다.
- 같은 topic({event.get('topic')})의 게시물들은 평소 {topic_mean if topic_mean else '비교할 만한 사례가 적어 계산 불가'} 정도 움직였다.

이 정보를 바탕으로 아래 **두 단락**으로 한국어 해설을 쓰세요.

1문단(요약, 2~3문장): 위 "요약해야 할 실제 원문/기사 본문"의 내용만 요약할 것 — 메타데이터나 통계를 요약에 넣지 말 것.
2문단(분석, 3~4문장): 다음을 순서대로, 쉬운 말로 담을 것 — 반드시 지킬 것:
(a) 이 게시물 다음 거래일에 실제로 주가가 얼마나 움직였는지({ar_text}) 사실로 서술.
(b) 이 종목이 평소 보이는 움직임과 비교했을 때 이번이 "{size_label}" 편이라는 것을 자연스러운 문장으로 서술(예: "이 종목은 평소 하루에 X% 정도 움직이는데, 이번엔 그보다 [훨씬 크게/비슷하게/작게] 움직였다").
(c) 오염 수준이 CLEAN이 아니면 "이 시기에 다른 사건도 같이 있었을 수 있어 이 게시물 하나 때문이라고 단정하기 어렵다"는 점을 쉬운 말로 명시.
(d) 마지막 한 문장에는 "이 프로젝트가 확인한 바로는 무슨 내용의 글인지보다 어떤 종목에 관한 글인지가 반응 크기에 더 큰 영향을 준다"는 취지를 쉬운 말로 반드시 넣을 것.

절대 금지:
- "CLEAN 표본", "백분위", "impact_score", "영향 점수" 같은 통계 용어를 그대로 쓰는 것. 반드시 일상 언어로 풀어 쓸 것.
- "이 발언 때문에 반응이 컸다/작았다"처럼 내용이 반응 크기를 결정한다는 인과관계를 단정하는 것.
- "앞으로 이런 글이 올라오면 이렇게 대응하라/투자하라" 같은 조언이나 행동 지시. 과거 사실을 설명하는 것은 되지만, 미래 행동을 지시하는 조언은 금지.
- "이 게시물이 앞으로 시장을 얼마나 움직일지" 같은 예측.
- 중국어를 섞는 것. 오직 한국어로만 답할 것.

두 문단을 줄바꿈으로 구분해서, 요약 문단 앞에 "요약: ", 분석 문단 앞에 "분석: "을 붙여서 출력하세요.

한국어 해설:"""


def generate_event_commentary(
    event: dict,
    ticker_stats: dict,
    content_text: str = None,
    provider: str = "ollama",
    model_name: str = None,
) -> str:
    prompt = build_dashboard_commentary_prompt(event, ticker_stats, content_text=content_text)
    try:
        return generate_text(prompt, provider=provider, model_name=model_name)
    except Exception as exc:
        return f"판단보류: AI 요약을 생성하지 못했습니다({exc}). 왼쪽의 AI 요약 설정을 확인하세요."


def build_translation_prompt(text: str) -> str:
    return f"""다음 영어(또는 외국어) 기사 본문을 자연스러운 한국어로 번역하세요.
요약하지 말고 원문의 내용을 빠짐없이 번역할 것. 마크다운 링크나 이미지 문법은 그대로 두되
본문 텍스트만 한국어로 옮길 것. 번역문 외에 다른 설명은 붙이지 말 것.
인물 이름 표기: "Elon Musk"/"Musk"는 "일론 머스크"로, "Donald Trump"/"Trump"는
"도널드 트럼프"로 표기할 것.

원문:
{text[:4000]}

한국어 번역:"""


def translate_to_korean(
    text: str,
    provider: str = "ollama",
    model_name: str = None,
) -> str:
    if not text or not text.strip():
        return text
    prompt = build_translation_prompt(text)
    try:
        return generate_text(prompt, provider=provider, model_name=model_name)
    except Exception as exc:
        return f"(번역 실패: {exc})\n\n{text}"


def build_ask_data_prompt(question: str, context: dict) -> str:
    return f"""당신은 "SNS 발언과 주가 반응" 프로젝트 대시보드에 붙어있는 질의응답
도우미입니다. 아래 "제공된 데이터"에 있는 내용만으로 답하세요. 데이터에 없는 내용은 추측하지
말고 "제공된 데이터로는 답할 수 없습니다"라고 답하세요.

제공된 데이터(이 프로젝트가 실제로 계산한 값):
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}

사용자 질문: {question}

반드시 지킬 것:
- 오직 한국어로만 답할 것.
- 위 데이터에 있는 사실만 사용할 것 — 없는 내용을 지어내지 말 것.
- "CLEAN 표본", "백분위", "impact_score" 같은 통계 용어 대신 쉬운 말로 풀어서 답할 것.
- 미래 예측이나 투자 조언을 하지 말 것 — 이 프로젝트는 회고적 통계 도구다.
- 인물 이름은 "일론 머스크", "도널드 트럼프"로 표기할 것.
- 질문에 적용된 필터와 표본 수를 먼저 확인하고, 평균과 중앙값을 혼동하지 말 것.
- 상위 사건을 언급할 때는 제공된 사건 번호와 날짜를 함께 쓸 것.
- 4~8문장 이내로 답하되 비교 질문이면 두 집단을 모두 설명할 것.

답변:"""


def answer_data_question(
    question: str,
    context: dict,
    provider: str = "ollama",
    model_name: str = None,
) -> str:
    prompt = build_ask_data_prompt(question, context)
    try:
        return generate_text(prompt, provider=provider, model_name=model_name)
    except Exception as exc:
        return f"판단보류: AI 답변을 생성하지 못했습니다({exc}). 왼쪽의 AI 요약 설정을 확인하세요."
