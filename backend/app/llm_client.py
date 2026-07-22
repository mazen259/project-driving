"""
llm_client.py
--------------
Thin wrapper around the Groq API (groq SDK).
Keeps all "talking to the model" logic in one place so main.py and
rag.py don't need to know anything about prompts or SDK details.
"""

import os
from typing import List

from groq import Groq
from .rag import Chunk

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client = None


def _get_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "or environment variables before calling the chat endpoint."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في تعليم السواقة في مصر. مهمتك:
- الرد على أسئلة المستخدمين حول: استخراج رخصة القيادة، إجراءات الاختبار النظري والعملي،
  مدارس/مدربي السواقة المتاحين ومناطقهم وأرقام هواتفهم، وأسعار صيانة السيارات.
- اعتمد فقط على المعلومات الموجودة في "السياق" أدناه. لا تختلق معلومات غير موجودة فيه.
- إذا لم تجد إجابة كافية في السياق، وضّح للمستخدم أن هذه المعلومة غير متوفرة لديك حالياً.
- اكتب الإجابة بالعربية الفصحى البسيطة، بشكل واضح ومنظم (نقاط مرقمة عند الحاجة).
- إذا كانت الإجابة تتضمن مدرسة/مدرب سواقة، اذكر الاسم ورقم الهاتف والمنطقة بوضوح.
"""


def build_context(chunks: List[Chunk]) -> str:
    if not chunks:
        return "لا يوجد سياق متاح لهذا السؤال."
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"{i}) {c.text}")
    return "\n".join(lines)


def generate_answer(user_message: str, chunks: List[Chunk]) -> str:
    client = _get_client()
    context = build_context(chunks)

    user_content = (
        f"### السياق (بيانات مسترجعة من قاعدة المعرفة):\n{context}\n\n"
        f"### سؤال المستخدم:\n{user_message}"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
