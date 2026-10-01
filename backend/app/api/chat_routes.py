"""
POST /api/v1/chat — document-aware chat using OpenAI.

The client sends the masked text as context and the conversation history.
We never send the original (unmasked) PII to OpenAI — only the masked version
with placeholders like {Person_1}, {TC_No_1}, etc.
"""

import os

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

router = APIRouter()

SYSTEM_PROMPT_TEMPLATE = """Sen bir belge analiz asistanısın. Aşağıdaki maskelenmiş belge sana bağlam olarak verilmiştir.
Belgede kişisel veriler {{Person_1}}, {{TC_No_1}} gibi yer tutucularla maskelenmiştir — bu yer tutucuların asıl değerlerini bilmiyorsun ve tahmin etmemelisin.
Kullanıcının sorularını yalnızca maskelenmiş belgedeki bilgilere dayanarak yanıtla.
Kısa, öz ve Türkçe cevaplar ver.

MASKELENMIŞ BELGE:
MASKED_TEXT_PLACEHOLDER"""


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    masked_text: str
    messages: list[ChatMessage]
    model: str = "gpt-4o-mini"


class ChatResponse(BaseModel):
    reply: str


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    client = AsyncOpenAI(api_key=api_key)

    system = SYSTEM_PROMPT_TEMPLATE.replace("MASKED_TEXT_PLACEHOLDER", request.masked_text[:12_000])

    messages = [{"role": "system", "content": system}]
    for m in request.messages:
        if m.role not in ("user", "assistant"):
            continue
        messages.append({"role": m.role, "content": m.content})

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=1024,
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    reply = response.choices[0].message.content or ""
    return ChatResponse(reply=reply)
