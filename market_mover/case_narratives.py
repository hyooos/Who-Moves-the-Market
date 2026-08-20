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
- "이 발언이 하락을 야기했다"처럼 인과관계를 단정하지 말 것
- "직후 이례적인 반응이 관측됐다"처럼 관찰 진술로 표현할 것
- 기사 근거가 부족하거나 가격 맥락이 약하면 "판단보류"라고 명시할 것
- 과장된 투자 조언을 하지 말 것
"""


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


def generate_with_ollama(prompt: str, model_name: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


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
        "ollama": "llama3.1:8b",
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
