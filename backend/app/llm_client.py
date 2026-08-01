"""
llm_client.py
--------------
Thin wrapper around the Groq API (groq SDK).
Keeps all "talking to the model" logic in one place so main.py and
rag.py don't need to know anything about prompts or SDK details.

Supports multi-turn conversation: the frontend can send prior
messages so the assistant remembers context across turns, instead of
treating every question as brand new.
"""

import os
from typing import List, Dict

from groq import Groq
from .rag import Chunk

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# llama-3.3-70b-versatile: strong general-purpose model, good Arabic
# quality, large context window. Override with GROQ_MODEL in .env if
# you want to try a different one available on Groq.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_HISTORY_TURNS = 6  # keep the last N user/assistant pairs only

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


SYSTEM_PROMPT = """أنت مساعد ذكي محترف متخصص في تعليم السواقة في مصر. مهمتك الرد على أسئلة
المستخدمين حول: استخراج رخصة القيادة، إجراءات الاختبار النظري والعملي، مدارس/مدربي
السواقة المتاحين ومناطقهم وأرقام هواتفهم، وأسعار صيانة السيارات.

قواعد المحتوى:
- اعتمد فقط على المعلومات الموجودة في "السياق" المرفق مع كل سؤال. لا تختلق معلومات غير موجودة فيه.
- "السياق" هو مادة خام قد تحتوي على تفاصيل أكثر مما يحتاجه السؤال. لا تنسخه أو تعيد
  سرده كما هو؛ افهمه ثم لخّصه واستخرج منه فقط ما يجيب على سؤال المستخدم تحديدًا.
- إذا لم تجد إجابة كافية في السياق، وضّح للمستخدم أن هذه المعلومة غير متوفرة لديك حالياً، واقترح
  إعادة صياغة السؤال أو تحديد المنطقة/نوع السيارة بدقة أكبر.

قواعد الأسلوب والتنسيق (مهمة جدًا):
- اجعل الإجابة مختصرة ومباشرة قدر الإمكان. ابدأ بجملة أو سطر واحد يجاوب على صلب السؤال
  مباشرة، ثم أضف التفاصيل الداعمة فقط إذا لزم الأمر.
- لا تكتب أكثر من 5-6 نقاط في الإجابة الواحدة إلا إذا طلب المستخدم تفصيلاً كاملاً صراحة.
- استخدم تنسيق Markdown بسيط: **نص عريض** للكلمات المفتاحية والأرقام والأسماء المهمة،
  وقوائم نقطية أو مرقمة (- أو 1.) عند سرد خطوات أو خيارات متعددة فقط.
- لا تستخدم عناوين Markdown (# أو ##) ولا تستخدم فقرات طويلة متلاحقة.
- إذا كانت الإجابة تتضمن مدرسة/مدرب سواقة، اذكر بشكل مختصر: **الاسم** والمنطقة —
  سطر واحد لكل مدرسة، بدون إعادة كل تفاصيل السياق. اذكر رقم الهاتف فقط إذا كان
  موجودًا فعليًا في السياق؛ لا تخترع رقم هاتف غير موجود ولا تقل إنه "غير متوفر"
  إلا لو المستخدم سأل عنه صراحة.
- اكتب الإجابة بالعربية الفصحى البسيطة أو باللهجة المصرية الواضحة حسب أسلوب سؤال المستخدم، بدون حشو أو تكرار.
- استخدم سياق المحادثة السابقة لفهم الأسئلة المتابعة (مثل "وإيه بعد كده؟" أو "وفي منطقة تانية؟").
"""


def build_context(chunks: List[Chunk]) -> str:
    if not chunks:
        return "لا يوجد سياق متاح لهذا السؤال."
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"{i}) {c.text}")
    return "\n".join(lines)


def generate_answer(
    user_message: str,
    chunks: List[Chunk],
    history: List[Dict[str, str]] = None,
) -> str:
    """history: list of {"role": "user"|"assistant", "content": "..."}
    from earlier turns in the same conversation (most recent last)."""
    client = _get_client()
    context = build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        trimmed = history[-(MAX_HISTORY_TURNS * 2):]
        for turn in trimmed:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_content = (
        f"### السياق (بيانات مسترجعة من قاعدة المعرفة):\n{context}\n\n"
        f"### سؤال المستخدم:\n{user_message}"
    )
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content