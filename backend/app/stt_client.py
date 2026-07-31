"""
stt_client.py
--------------
Thin wrapper around faster-whisper (local speech-to-text, لا يحتاج API
خارجي). بيحوّل ملف صوتي (رسالة صوتية) لنص، عشان النص ده يتمرر بعد كده
لنفس الـ RAG pipeline بتاع الرسائل النصية العادية (rag.py + llm_client.py).

نفس فلسفة llm_client.py: كل التفاصيل الخاصة بالموديل نفسه (تحميله،
إعداداته) متجمعة هنا، وmain.py مش محتاج يعرف حاجة عن faster-whisper.
"""

import os
from typing import Optional

from faster_whisper import WhisperModel

# ---- إعدادات الموديل (تقدر تتحكم فيها من backend/.env) ----
# tiny/base/small: أسرع وأخف، مناسبة لتجربة سريعة أو سيرفر بإمكانيات محدودة.
# medium/large-v3: أدق (خصوصًا مع اللهجة المصرية) بس أبطأ وأتقل على الذاكرة.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" أو "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
# لو عايز الموديل يكتشف اللغة تلقائيًا سيب WHISPER_LANGUAGE فاضية في .env
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ar")

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """
    بيحمّل موديل faster-whisper مرة واحدة بس (lazy singleton) ويعيد
    استخدامه لكل الطلبات اللي بعد كده — تحميل الموديل هو الجزء البطيء،
    مش عملية التفريغ (transcription) نفسها.
    """
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe_audio(file_path: str) -> str:
    """
    بياخد مسار ملف صوتي على الديسك (wav, mp3, m4a, ogg, webm ...) ويرجع
    النص المستخرج منه كـ string واحد.

    بيرمي أي Exception زي ما هي لو الملف تالف أو الموديل فشل يحمّل، عشان
    main.py هو اللي يقرر يترجمها لـ HTTPException مناسبة للمستخدم.
    """
    model = _get_model()
    segments, _info = model.transcribe(
        file_path,
        language=WHISPER_LANGUAGE or None,
        vad_filter=True,  # يشيل فترات الصمت -> نتيجة أدق وأسرع
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text)
    return text.strip()
