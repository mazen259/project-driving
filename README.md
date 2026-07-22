# مساعد تعليم السواقة (Driving School RAG Assistant)

مشروع RAG (Retrieval-Augmented Generation) كامل يجمع بين:
- قاعدة معرفة عن إجراءات استخراج رخصة القيادة في مصر
- قائمة مدارس/مدربين السواقة (الاسم، الهاتف، المنطقة، المحافظة)
- بيانات أسعار صيانة السيارات

ويستخدم **Groq API** لتوليد إجابات دقيقة بالعربية بناءً على البيانات
المسترجعة فقط، مع واجهة ويب (frontend) متصلة عبر **FastAPI**.

## هيكل المشروع

```
driving-rag/
├── backend/
│   ├── app/
│   │   ├── main.py            # تطبيق FastAPI (API + تشغيل الواجهة)
│   │   ├── rag.py             # فهرسة TF-IDF والبحث الدلالي
│   │   ├── llm_client.py       # الاتصال بـ Groq API
│   │   ├── data_loader.py     # قراءة وتحويل ملف البيانات الخام
│   │   └── __init__.py
│   ├── data/
│   │   └── all_data.txt       # البيانات الخام (رخص + مدارس + صيانة)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html              # واجهة الدردشة (Dark theme, RTL)
```

## طريقة التشغيل

### 1. تجهيز البيئة

```bash
cd backend
python3 -m venv venv
source venv/bin/activate      # على Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. إضافة مفتاح Groq API

```bash
cp .env.example .env
```

ثم افتح ملف `.env` وضع مفتاحك الحقيقي (يمكن الحصول عليه مجاناً من
https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_...
```

### 3. تشغيل السيرفر

```bash
uvicorn app.main:app --reload --port 8000
```

ثم افتح المتصفح على: **http://localhost:8000**

الواجهة الأمامية (frontend) والـ backend يعملان معاً من نفس السيرفر — لا حاجة
لأي إعداد إضافي.

## كيف يعمل RAG هنا

1. **الفهرسة (Indexing):** عند تشغيل السيرفر، يقرأ `data_loader.py` ملف
   `all_data.txt` ويحوّله إلى ثلاث مجموعات بيانات منظمة (إجراءات، مدارس،
   صيانة). ثم يبني `rag.py` فهرس TF-IDF عليها.
2. **الاسترجاع (Retrieval):** عند وصول سؤال المستخدم، يبحث الفهرس عن أقرب
   6 مقاطع نصية للسؤال (باستخدام تشابه الجيب التمام / cosine similarity).
3. **التوليد (Generation):** يتم إرسال المقاطع المسترجعة + سؤال المستخدم إلى
   Groq (نموذج Llama 3.3)، مع تعليمات صارمة بالاعتماد فقط على هذا السياق، فيولد إجابة نهائية
   بالعربية.

## نقاط الـ API الرئيسية

| Method | Endpoint | الوظيفة |
|---|---|---|
| POST | `/api/chat` | إرسال سؤال والحصول على إجابة RAG + Gemini |
| GET | `/api/schools?area=..&governorate=..` | فلترة مدارس السواقة مباشرة |
| GET | `/api/maintenance?engine_cc=..&service_type=..&city=..` | فلترة أسعار الصيانة مباشرة |
| GET | `/api/health` | التأكد أن السيرفر وقاعدة المعرفة تعمل |

## أفكار للتطوير لاحقاً

- استبدال TF-IDF بـ embeddings حقيقية لدقة أعلى في البحث الدلالي.
- إضافة قاعدة بيانات حقيقية (PostgreSQL/SQLite) بدل قراءة الملف كل مرة.
- إضافة تسجيل دخول وتقييم للمدربين من المستخدمين.
- نشر المشروع على منصة مثل Render أو Railway أو Fly.io.
