import os
from pathlib import Path

import pandas as pd
import requests


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

    return f"""다음은 SNS 게시물 또는 공개 발언과 그 직후 시장 반응에 대한 정보입니다.

이벤트 정보:
- 인물: {event.get('person')}
- 게시/발언 시각: {event.get('posted_at')}
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
- 이 이벤트의 주체는 반드시 "{event.get('person')}"이다. 기사에 다른 인물이 함께 언급되더라도 절대 혼동하지 말고, 발언/행동의 주어는 항상 "{event.get('person')}"으로 쓸 것
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
    return sum(text.count(marker) for marker in _CHINESE_MARKERS) >= 2


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


def generate_case_narrative(
    event: dict,
    article_md: str,
    price_context: dict,
    provider: str = "gemini",
    model_name: str = None,
) -> str:
    prompt = build_narrative_prompt(event, article_md, price_context)
    provider = provider.lower()
    defaults = {
        "gemini": "gemini-1.5-flash",
        "groq": "llama-3.3-70b-versatile",
        "ollama": "qwen2.5:7b",
    }
    model_name = model_name or defaults.get(provider)
    try:
        if provider == "gemini":
            return generate_with_gemini(prompt, model_name)
        if provider == "groq":
            return generate_with_groq(prompt, model_name)
        if provider == "ollama":
            return generate_with_ollama(prompt, model_name)
        if provider == "none":
            return fallback_narrative(event, "LLM provider가 none으로 설정되었습니다.")
        return fallback_narrative(event, f"지원하지 않는 LLM provider입니다: {provider}")
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


def build_dashboard_commentary_prompt(event: dict, ticker_stats: dict) -> str:
    """대시보드에서 클릭한 이벤트 하나에 대한 짧은 해설 프롬프트.

    이 프로젝트는 §5-3에서 topic 내용이 반응 크기를 거의 설명하지 못하고(topic
    순수 추가 설명력 0.3%p), 종목(ticker) 자체의 변동성이 대부분을 설명한다는 걸
    이미 확인했다. 그래서 이 프롬프트는 "이 발언이 크게 움직일 것이다" 같은 예측을
    절대 만들지 말고, 이 종목의 과거 CLEAN 표본 통계를 있는 그대로 설명하도록
    강하게 제약한다.
    """
    ar = event.get("abnormal_return")
    ar_text = "계산 불가" if ar is None or pd.isna(ar) else f"{ar:.2%}"
    pct = ticker_stats.get("percentile")
    pct_text = "계산 불가" if pct is None else f"{pct*100:.0f}%"

    return f"""다음은 대시보드에서 사용자가 클릭한 SNS 게시물 1건과, 그 종목의 과거 통계입니다.

게시물 정보:
- 인물: {event.get('person')}
- 게시 시각: {event.get('posted_at')}
- 규칙 기반 분류 topic: {event.get('topic')}
- 매핑된 종목: {event.get('ticker')}
- 원문/설명: {str(event.get('text_clean') or event.get('description') or '')[:500]}
- 이 게시물의 초과수익률: {ar_text}
- 오염 수준(다중게시/매크로/시장충격 여부): {event.get('contamination_level')}

이 종목의 과거 CLEAN 표본 통계(이 프로젝트가 실제로 계산한 값):
- CLEAN 표본 수: {ticker_stats.get('n')}
- 평균 절대 초과수익률: {ticker_stats.get('mean_abs_ar')}
- 이 게시물의 영향점수가 이 종목 CLEAN 표본에서 차지하는 백분위: {pct_text}

이 정보를 바탕으로 아래 **두 단락**으로 한국어 해설을 쓰세요.

1문단(요약, 1문장): 이 게시물이 무슨 내용인지 먼저 요약할 것.
2문단(분석, 2~3문장): 그 다음 통계 기반 해설을 쓸 것. 반드시 지킬 것:
- 오직 한국어로만 답할 것. 중국어를 섞지 말 것.
- "이 발언 때문에 반응이 컸다/작았다"처럼 내용이 반응 크기를 결정한다는 인과관계를 절대 단정하지 말 것 — 이 프로젝트는 topic 내용보다 종목 자체의 변동성이 반응 크기를 더 크게 설명한다는 걸 통계로 확인했다(topic이 설명하는 몫은 0.3%포인트뿐).
- "이 게시물이 앞으로 시장을 얼마나 움직일지" 같은 예측은 절대 하지 말 것 — 이 대시보드는 예측 도구가 아니라 과거 통계를 보여주는 회고적 도구다.
- 대신 "이 종목은 과거 CLEAN 표본에서 평균 X% 반응을 보였고, 이 게시물은 그중 Y번째 백분위에 해당한다" 같은 사실 기술 위주로 쓸 것.
- 오염 수준이 CLEAN이 아니면 "다른 요인과 섞여 있어 이 게시물 하나의 영향으로 단정하기 어렵다"는 점을 명시할 것.
- 과장된 투자 조언을 하지 말 것.

두 문단을 줄바꿈으로 구분해서, 요약 문단 앞에 "요약: ", 분석 문단 앞에 "분석: "을 붙여서 출력하세요.

한국어 해설:"""


def generate_event_commentary(event: dict, ticker_stats: dict, model_name: str = "qwen2.5:7b") -> str:
    prompt = build_dashboard_commentary_prompt(event, ticker_stats)
    try:
        return generate_with_ollama(prompt, model_name)
    except Exception as exc:
        return f"판단보류: 로컬 LLM(Ollama) 호출에 실패했습니다({exc}). Ollama가 실행 중인지 확인하세요."
