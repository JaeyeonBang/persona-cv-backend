import os
import json
import httpx
from typing import AsyncIterator


ANSWER_LENGTH_MAP = {
    "short": "2~3문장으로 간결하게",
    "medium": "5~7문장으로 적당하게",
    "long": "충분히 자세하게 (10문장 이상)",
}

SPEECH_STYLE_MAP = {
    "formal": "격식체 (합니다/입니다)",
    "casual": "반말 (해/야)",
}

QUESTION_STYLE_MAP = {
    "free": "자유로운 대화처럼",
    "interview": "면접관의 질문에 답하듯 구체적이고 근거를 들어서",
    "chat": "친구와 가볍게 대화하듯",
}

PRESET_TONE_MAP = {
    "professional": "전문적이고 논리적인 어조",
    "friendly": "친근하고 따뜻한 어조",
    "challenger": "도전적이고 자신감 있는 어조",
}


def build_system_prompt(user: dict, config: dict, context: str, search_results: list = []) -> str:
    """페르소나 + Interviewer 설정을 바탕으로 시스템 프롬프트를 생성합니다."""
    pc = user.get("persona_config", {})
    lang = "한국어" if config.get("language", "ko") == "ko" else "English"
    tone = PRESET_TONE_MAP.get(pc.get("preset", "professional"), "전문적이고 논리적인 어조")
    speech = SPEECH_STYLE_MAP.get(config.get("speechStyle", "formal"), "격식체 (합니다/입니다)")
    length = ANSWER_LENGTH_MAP.get(config.get("answerLength", "medium"), "5~7문장으로 적당하게")
    style = QUESTION_STYLE_MAP.get(config.get("questionStyle", "free"), "자유로운 대화처럼")
    custom_prompt = pc.get("custom_prompt")

    lines = [
        f"당신은 {user['name']}입니다.",
        f"직책: {user['title']}",
        f"소개: {user['bio']}",
        "",
        "## 말투 & 스타일",
        f"- 기본 어조: {tone}",
    ]
    if custom_prompt:
        lines.append(f"- 추가 지시: {custom_prompt}")
    lines += [
        f"- 답변 길이: {length}",
        f"- 말투: {speech}",
        f"- 질문 스타일: {style}",
        f"- 답변 언어: {lang}",
        "",
        (
            f"## 참고 자료 (관련 문서에서 발췌)\n{context}\n"
            if context
            else "## 참고 자료\n(등록된 문서가 없습니다. 일반 지식으로 답변하세요.)\n"
        ),
        "## 지시사항",
        f"- 항상 {user['name']} 본인의 입장에서 1인칭으로 답변하세요.",
        "- 자료에 없는 내용은 지어내지 말고 모른다고 하세요.",
        "- 마크다운 문법은 사용하지 마세요.",
    ]

    relevant = [r for r in search_results if r.similarity >= 0.25]
    if relevant:
        lines += [
            "",
            "## 인용 지침",
            "- 참고 자료를 바탕으로 답변한 문장 끝에 [1], [2] 형식으로 출처 번호를 붙이세요.",
            "- 예시: '저는 Kakao에서 3년간 근무했습니다. [1]'",
            "- 출처 번호는 위 참고 자료의 [1], [2], [3] 번호와 동일하게 사용하세요.",
            "- 자료를 사용하지 않은 일반적인 문장에는 번호를 붙이지 마세요.",
        ]

    return "\n".join(lines)


async def stream_chat(system_prompt: str, question: str) -> AsyncIterator[str]:
    """OpenRouter SSE 스트리밍 응답을 텍스트 청크로 yield합니다."""
    api_key = os.environ["OPENROUTER_API_KEY"]
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("OPENROUTER_MODEL", "z-ai/glm-4.7-flash")

    payload = {
        "model": model,
        "stream": True,
        "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(payload),
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for raw_chunk in response.aiter_bytes():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    trimmed = line.strip()
                    if not trimmed.startswith("data:"):
                        continue
                    data = trimmed[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data)
                        choices = parsed.get("choices", [])
                        if not choices:
                            continue
                        text = choices[0].get("delta", {}).get("content")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        pass
