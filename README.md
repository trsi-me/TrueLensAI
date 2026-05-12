# نظام TrueLens AI — نظام كشف المحتوى المزيف بالذكاء الاصطناعي

## فهرس المحتويات

1. [مقدمة وفكرة المشروع](#1-مقدمة-وفكرة-المشروع)
2. [المتطلبات وبيئة التشغيل](#2-المتطلبات-وبيئة-التشغيل)
3. [كيفية تشغيل المشروع](#3-كيفية-تشغيل-المشروع)  
   - [3.1 التنزيل اليدوي للداتاسيتات](#31-التنزيل-اليدوي-للداتاسيتات)
4. [بنية المشروع والملفات](#4-بنية-المشروع-والملفات)
5. [قاعدة البيانات](#5-قاعدة-البيانات)
6. [الداتاسيت المستخدمة](#6-الداتاسيت-المستخدمة)
7. [النماذج والخوارزميات](#7-النماذج-والخوارزميات)
8. [واجهات برمجة التطبيق (API Endpoints)](#8-واجهات-برمجة-التطبيق-api-endpoints)
9. [المصطلحات البرمجية](#9-المصطلحات-البرمجية)  
   - [9.1 معجم تقني (طرفية، Git، GitHub، Git LFS، pkl، دوال…)](#section-glossary-tech)
10. [الأسئلة الشائعة والمشكلات](#10-الأسئلة-الشائعة-والمشكلات)
11. [نتائج التدريب (تشغيل فعلي)](#11-نتائج-التدريب-تشغيل-فعلي)
12. [شرح الكود والمقتطفات](#12-شرح-الكود-والمقتطفات)  
    (يشمل 12.1–12.6 نقطة الدخول والإعدادات والقاعدة والنماذج ومسار النص؛ و12.7–12.20 حفظ النتائج، الصورة، الفيديو، السجل، الكواشف، الأدوات، التدريب، التنزيل، والجدول المرجعي.)
13. [الحسابات الافتراضية واللوحة والإدارة](#section-accounts-dashboard-admin)
14. [النتائج غير المؤكدة والتقرير الكامل وELA وmetadata الصوت](#section-uncertain-report-ela)
15. [نشر Render — النماذج والذاكرة وما لا يُرفع](#section-render-models)

---

## 1. مقدمة وفكرة المشروع

### 1.1 المشكلة التي يحلها المشروع

تنتشر الأخبار والصور ومقاطع الفيديو المزيفة أو المولدة بالذكاء الاصطناعي بسرعة، مما يصعّب على المستخدم تمييز المحتوى الموثوق دون أدوات مساعدة.

### 1.2 الحل المقترح

TrueLens AI تطبيق ويب يعمل محلياً (دون اتصال إلزامي أثناء التشغيل) يقدّم:

- تقديراً تقريبياً لمصداقية **نص الخبر** من أنماط **الصياغة والأسلوب والمفردات** (مثل تدريب WELFake)، مع مؤشرات إرشادية — **وليس** تحققاً آلياً من صحة كل جملة مقابل الإنترنت.
- كشفاً تقريبياً للصور المزيفة أو المولدة.
- تحليلاً للفيديو عبر استخراج إطارات وتطبيق نموذج الصور.

### 1.3 أهمية المشروع

يقلل الاعتماد على خدمات سحابية أثناء التحليل، ويحفظ الخصوصية عند معالجة الملفات على الجهاز، ويوضّح نسبة ثقة ونتيجة مفهومة بالعربية.

---

## 2. المتطلبات وبيئة التشغيل

### 2.1 متطلبات النظام

- نظام تشغيل يدعم Python 3.10.6.
- ذاكرة وصول عشوائي كافية لتشغيل TensorFlow وتدريب النماذج (يُفضّل 8 جيجابايت أو أكثر للتدريب).
- مساحة قرص للداتاسيتات والنماذج المحفوظة.

### 2.2 تثبيت Python 3.10.6

1. حمّل المثبت الرسمي من الرابط التالي:  
   `https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe`
2. شغّل الملف `python-3.10.6-amd64.exe`.
3. في أول شاشة، فعّل خيار **Add Python 3.10 to PATH** ثم اختر **Install Now**.
4. بعد انتهاء التثبيت، افتح موجه أوامر جديداً واكتب:  
   `python --version`  
   يجب أن يظهر: `Python 3.10.6`.
5. (وصف بصري نصي) شاشة التثبيت تعرض شعار Python ومعلومات المسار؛ تأكد من عدم إغلاق النافذة قبل اكتمال شريط التقدم.

### 2.3 تثبيت المتطلبات

1. انتقل إلى مجلد المشروع:  
   `D:\VSCode\Projects\TrueLensAI`
2. ثبّت الحزم:  
   `python -m pip install --upgrade pip`  
   `pip install -r requirements.txt`

---

## 3. كيفية تشغيل المشروع

### 3.1 التنزيل اليدوي للداتاسيتات

لا تحتاج `kaggle.json` ولا سطر أوامر تنزيل إذا نزّلت الملفات من الموقع بنفسك.

**متطلب:** حساب مجاني على Kaggle وتسجيل الدخول في المتصفح.

**هيكل مجلد `datasate` الذي يعتمد عليه المشروع (كما في CIFAKE بعد الفك):**

```
datasate/
  WELFake_Dataset.csv     ← نصوص الأخبار (تدريب نموذج النص)
  archive/
    train/
      FAKE/
      REAL/
    test/                  ← اختياري: يُستخدم لتقييم النموذج بعد التدريب
      FAKE/
      REAL/
```

التدريب يقرأ من `archive/train`. إن وُجد `archive/test` يطبع السكربت دقة اختبار منفصلة بعد الحفظ. إذا لم يوجد `train`، يُستخدم محتوى `archive` مباشرة إن كان فيه مجلدان للفئتين.

#### أ) WELFake (نصوص — تدريب نموذج الأخبار)

1. افتح صفحة المجموعة:  
   `https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification`
2. نزّل الأرشيف وفكّ الضغط واستخرج ملف الـ CSV الرئيسي.
3. انسخه أو سمّه **`WELFake_Dataset.csv`** وضعه مباشرة في:  
   `D:\VSCode\Projects\TrueLensAI\datasate\WELFake_Dataset.csv`  
   (سكربت التدريب يقرأ هذا المسار تلقائياً؛ وإن استخدمت اسماً آخر مرّر `--csv`.)

#### ب) CIFAKE (صور — تدريب نموذج الصور)

1. افتح صفحة المجموعة:  
   `https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images`
2. نزّل الأرشيف وفكّ الضغط داخل المجلد:  
   `D:\VSCode\Projects\TrueLensAI\datasate\archive`  
   بحيث يصبح إما `archive\train\REAL` و`archive\train\FAKE` أو `archive\REAL` و`archive\FAKE`.

#### ج) بديل (اختياري): التنزيل من الطرفية

إذا رغبت بالتنزيل عبر `python scripts/download_dataset.py welfake` أو `cifake`، يجب إعداد ملف `kaggle.json` في `C:\Users\<اسمك>\.kaggle\` أو متغيرات البيئة `KAGGLE_USERNAME` و`KAGGLE_KEY`. إن تعذّر ذلك، استخدم القسم **3.1 أ–ب** أعلاه.

---

### 3.2 الخطوة الأولى: تدريب نموذج الأخبار

1. تأكد من وجود `datasate\WELFake_Dataset.csv` (انظر **3.1**)، ثم من جذر المشروع:  
   `python scripts/train_text_model.py`  
   أو إذا كان الملف باسم آخر:  
   `python scripts/train_text_model.py --csv D:\مسار\كامل\إلى\الملف.csv`
2. يجب أن يُنشأ:  
   `ml_models/saved_models/text_model.pkl`  
   `ml_models/saved_models/tfidf_vectorizer.pkl`

### 3.3 الخطوة الثانية: تدريب نموذج الصور

1. بعد تجهيز `datasate\archive` كما في **3.1**، شغّل من جذر المشروع (المسار الافتراضي هو `archive` أو `archive\train`):  
   `python scripts/train_image_model.py`  
   أو إذا كان ترتيب المجلدات مختلفاً:  
   `python scripts/train_image_model.py --data_dir D:\مسار\إلى\مجلد_فيه_REAL_و_FAKE`
2. يُحفظ النموذج في:  
   `ml_models/saved_models/image_model.h5`

### 3.4 الخطوة الثالثة: تشغيل التطبيق

من جذر المشروع:

`python app.py`

### 3.5 فتح الموقع في المتصفح

افتح المتصفح وانتقل إلى:  
`http://127.0.0.1:5000/`

---

## 4. بنية المشروع والملفات

- `app.py` — إنشاء تطبيق Flask وتسجيل المسارات وتهيئة قاعدة البيانات وتحميل النماذج.
- `config.py` — مسارات النماذج والمجلدات والحدود.
- `datasate/` — بيانات التدريب: `WELFake_Dataset.csv` للنصوص، و`archive/` للصور (`DATASET_ROOT` في `config.py`).
- `database/` — تهيئة SQLite وجداول `detection_history` و`app_stats` ودوال الحفظ والقراءة.
- `ml_models/` — كاشف النص والصورة والفيديو ومحمّل النماذج.
- `routes/` — صفحات وواجهات JSON للكشف والسجل **ومسارات المصادقة (`auth`) واللوحة (`dashboard`) والإدارة (`admin`)**.
- `utils/` — رفع الملفات والتحقق من المدخلات **وملحقات العرض (نطاق Uncertain، ELA، ffprobe، صلاحيات الوصول)**.
- `static/` — أنماط CSS وسكربتات JS.
- `templates/` — قوالب HTML.
- `assets/uploads/` — ملفات مرفوعة مؤقتاً.
- `scripts/` — تنزيل الداتاسيت وتدريب النماذج.

---

## 5. قاعدة البيانات

### 5.1 شرح مفهوم SQLite للمبتدئين

SQLite قاعدة بيانات ملف واحد تُخزَّن على القرص دون خادم منفصل؛ مناسبة للتطبيقات المحلية والصغيرة.

### 5.2 هيكل الجداول

- `detection_history`: سجل الفحوصات المحفوظة يدوياً من الواجهة (نوع، ملخص، نتيجة، ثقة، نموذج، وقت، اسم ملف، تاريخ).
- `app_stats`: عدادات `total_scans` و`fake_detected` و`real_detected`.
- **(إضافة لاحقة)** جدول **`users`**: حسابات المستخدمين (بريد، كلمة مرور مشفّرة، اسم عرض اختياري، `is_admin`، تاريخ الإنشاء). لا يُنشأ مستخدم تلقائياً إلا إذا ضبطت **`TRUELENS_DEFAULT_ADMIN_EMAIL`** و **`TRUELENS_DEFAULT_ADMIN_PASSWORD`** معاً في البيئة (انظر [القسم 13](#section-accounts-dashboard-admin))؛ وإلا التسجيل من الواجهة أو يدوياً في قاعدة البيانات.
- **(إضافة لاحقة)** عمود **`user_id`** في `detection_history`: يربط السجل بالمستخدم عند الحفظ؛ السجلات القديمة بدون مستخدم تبقى كما هي حتى يُحفظ سجل جديد بعد تسجيل الدخول.
- **(إضافة لاحقة)** عداد **`uncertain_detected`** في `app_stats`: يزيد عندما تكون نتيجة الفحص المعروضة **Uncertain** (نطاق ثقة قابل للضبط، انظر [القسم 14](#section-uncertain-report-ela)).

### 5.3 العلاقات بين الجداول

لا توجد علاقات مفاتيح أجنبية بين الجدولين؛ الإحصائيات مستقلة عن صفوف السجل لكنها تُحدَّث عند إتمام فحص ناجح من واجهات الكشف.

**ملاحظة إضافية:** بين `users` و`detection_history` العلاقة منطقياً **واحد إلى كثير** (مستخدم واحد، عدة سجلات) عبر `user_id` عند الحفظ بعد تسجيل الدخول.

---

## 6. الداتاسيت المستخدمة

### 6.1 WELFake Dataset (لكشف الأخبار)

- مصدر: منصة Kaggle، مجموعة **Fake News Classification (WELFake)**.
- المحتوى: نصوص أخبار مع تصنيف ثنائي (مزيف/حقيقي).
- التنزيل: يدوياً من صفحة المجموعة بعد تسجيل الدخول (انظر القسم 3.1)، أو عبر السكربت إن وُجدت اعتمادات API.
- الحجم يختلف حسب النسخة؛ ضع ملف CSV باسم `WELFake_Dataset.csv` داخل `datasate/`.
- الاستخدام: تدريب `TF-IDF` مع مصنف تعلم آلي وحفظ النموذج والمحوّل بصيغة `joblib`.

### 6.2 CIFAKE Dataset (لكشف الصور)

- مصدر: Kaggle، مجموعة **CIFAKE: Real and AI-Generated Synthetic Images**.
- المحتوى: صور حقيقية ومولدة ضمن مجلدات تدريب منظمة.
- التنزيل: يدوياً من صفحة المجموعة (القسم 3.1)، ثم فك الصور تحت `datasate/archive`.
- الحجم كبير نسبياً.
- الاستخدام: تدريب EfficientNetB0 مع تعلم انتقالي على مجلد `train` الذي يحتوي `REAL` و`FAKE`.

---

## 7. النماذج والخوارزميات

### 7.1 شرح مفهوم Machine Learning للمبتدئين

التعلم الآلي يعني تدريب نموذج رياضي على أمثلة (مدخلات ونتائج معروفة) ليتعلم قاعدة تقريبية يتوقع بها النتائج لمدخلات جديدة.

### 7.2 نموذج TF-IDF — شرح مفصل مع كود

TF-IDF يحوّل النص إلى أرقام تعكس أهمية الكلمات في المستند مقارنة بمجموعة النصوص. في المشروع يُستخدم داخل أنبوب تدريب مع مصنف:

```python
TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
```

### 7.3 خوارزمية Passive Aggressive Classifier — شرح مبسط

مصنف خطي يُحدّث أوزاناً بسرعة مناسبة للنصوص الكبيرة؛ مع `loss='log_loss'` يمكن الحصول على احتمالات للثقة.

### 7.4 نموذج EfficientNetB0 — شرح Transfer Learning

يُحمّل EfficientNetB0 مسبق التدريب على ImageNet، تُجمَّد الطبقات أولاً ثم تُدرَّب طبقات علوية على صور حقيقية/مزيفة، ثم يُجرى ضبط دقيق اختياري.

### 7.5 كيف يكشف النظام الفيديو (frame analysis)

يُستخرج عدد محدود من الإطارات موزّعة على طول الفيديو، ويُطبَّق نموذج الصور على كل إطار، ثم يُحسب متوسط احتمال التزييف لاتخاذ قرار إجمالي مع انحراف معياري مساعد للثقة.

---

## 8. واجهات برمجة التطبيق (API Endpoints)

| الطريقة | المسار | المدخلات | المخرجات |
|--------|--------|----------|----------|
| GET | `/` | — | صفحة رئيسية HTML |
| GET | `/about` | — | صفحة عن المشروع |
| GET | `/detect/text` | — | صفحة كشف نص |
| POST | `/detect/text/analyze` | JSON: `text` | JSON موحّد مع النتيجة والثقة؛ عند تسجيل الدخول قد يتضمّن `history_id` بعد الحفظ التلقائي في السجل |
| POST | `/detect/text/save` | JSON للنتيجة | حفظ يدوي إضافي (اختياري؛ التحليل الناجح يُحفظ تلقائياً عند تسجيل الدخول ما لم يُضبط `TRUELENS_DISABLE_AUTO_HISTORY`) |
| GET | `/detect/image` | — | صفحة كشف صورة |
| POST | `/detect/image/analyze` | `multipart/form-data` ملف | JSON بالنتيجة؛ قد يتضمّن `history_id` عند الحفظ التلقائي |
| POST | `/detect/image/save` | JSON | حفظ يدوي إضافي في السجل |
| GET | `/detect/video` | — | صفحة كشف فيديو |
| POST | `/detect/video/analyze` | `multipart/form-data` | JSON بالنتيجة وعدد الإطارات؛ قد يتضمّن `history_id` عند الحفظ التلقائي |
| POST | `/detect/video/save` | JSON | حفظ يدوي إضافي في السجل |
| GET | `/history/` | — | صفحة السجل |
| GET | `/history/data` | — | JSON بقائمة السجلات |
| DELETE | `/history/delete/<id>` | — | حذف سجل |
| DELETE | `/history/clear` | — | مسح السجل كله |
| GET | `/auth/login` ، `/auth/register` ، `/auth/account` | نماذج HTML | تسجيل الدخول والتسجيل وإدارة الحساب |
| POST | `/auth/logout` | — | تسجيل الخروج |
| GET | `/dashboard` | يتطلب جلسة مسجّل | لوحة ملخص للمستخدم (عدد السجلات المحفوظة حسب النوع) |
| GET | `/report` | بعد تحليل ناجح يُملأ من الجلسة | صفحة تقرير نصي للتحليل الأخير (بدون صورة ELA الثقيلة في الجلسة) |
| GET | `/admin/` ، `/admin/detections` | يتطلب `is_admin` | لوحة إدارية: مستخدمون وسجل عام (للمشرف فقط) |

صيغة الاستجابة الموحّدة:  
`{"success": true/false, "data": {...}, "error": null أو نص الخطأ}`

**تحديثات لاحقة:** استجابات `analyze` للنص/الصورة/الفيديو قد تتضمن `model_label` و`explanations` و`ela_image` (JPEG) و`audio_summary` (فيديو). مسارات **`/detect/*/save`** تتطلب **جلسة مسجّل** وتقبل التسمية **`Uncertain`** أيضاً.

---

## 9. المصطلحات البرمجية

| المصطلح | المعنى باختصار |
|--------|----------------|
| Flask | إطار ويب بلغة بايثون لتوجيه الطلبات وعرض القوالب |
| SQLite | قاعدة بيانات ملفية خفيفة |
| TF-IDF | تمثيل نصي عددي للكلمات |
| Neural Network | شبكة طبقات للتعلم من البيانات |
| Transfer Learning | الاستفادة من نموذج مدرب مسبقاً على مهمة جديدة |
| Deepfake | محتوى مرئي مزيف أو مولد يشبه الحقيقي |
| Pipeline | سلسلة خطوات (مثلاً تحويل نص ثم تصنيف) |
| Joblib | تنسيق حفظ وتحميل النماذج في بايثون |
| OpenCV | مكتبة لمعالجة الفيديو واستخراج الإطارات |

<a id="section-glossary-tech"></a>

### 9.1 معجم مصطلحات تقنية شائعة (أدوات، مستودع، ملفات، برمجة)

| المصطلح | شرح مبسّط |
|---------|-----------|
| **الطرفية (Terminal)** | نافذة نصية تُنفَّذ فيها أوامر النظام أو المترجم (مثل `python`، `pip`) بدون واجهة رسومية؛ على ويندوز غالباً **PowerShell** أو **CMD**، وعلى لينكس/ماك **Terminal**. |
| **Git** | نظام **تحكم بالإصدارات** محلياً على جهازك: يتتبّع تغييرات الملفات، الفروع، والدمج، ويُعدّ الريبو جاهزاً للمزامنة مع خادم بعيد. |
| **GitHub (قيت هاب)** | خدمة **استضافة مستودعات Git** على الإنترنت (مشاركة الكود، Issues، Actions للبناء التلقائي، Releases لرفع مرفقات كبيرة مثل النماذج). المشروع **ليس** GitHub نفسه — Git أداة محلية، GitHub منصة. |
| **Git LFS (Large File Storage)** | امتداد لـ Git يخزّن **ملفات كبيرة** (مثلاً نماذج `.h5`) كمراجع منفصلة بدل دمجها التاريخي الثقيل في كل clone؛ مفيد عند الحاجة لرفع أصول ضخمة مع الريبو. |
| **ملف `.pkl` (Pickle)** | تنسيق **تسلسل بايثون** (pickle/joblib) لحفظ كائنات جاهزة مثل مصنّف تعلم آلة بعد التدريب؛ يُحمّل لاحقاً بنفس إصدار المكتبات قدر الإمكان. **لا** تفتح ملفات `.pkl` من مصادر غير موثوقة (قد تنفّذ كوداً). |
| **دوال (Functions)** | وحدات مسماة في البرنامج تقبل **مدخلات** وتعيد **مخرجات** (أو تنفّذ أثراً جانبياً)، لتنظيم المنطق وإعادة الاستخدام؛ في بايثون تُعرَّف بـ `def اسم_الدالة(...):`. |
| **متغيرات البيئة (Environment variables)** | أزواج `اسم=قيمة` يقرأها البرنامج عند التشغيل (مثل `TRUELENS_DEFAULT_ADMIN_EMAIL`) دون تثبيت الأسرار داخل الكود. |
| **HTTPS / الاستضافة (Render، إلخ)** | **HTTPS** يشفّر الاتصال بين المتصفح والخادم؛ **Render** مثال على خدمة نشر تطبيقات مع تعيين أسرار من لوحة التحكم. |

---

## 10. الأسئلة الشائعة والمشكلات

**لا تظهر نتائج النص أو تظهر رسالة غياب النموذج**  
تأكد من وجود `text_model.pkl` و`tfidf_vectorizer.pkl` بعد تشغيل `train_text_model.py`.

**تحليل النص سريع جداً أو يظهر دائماً «Real» لنصوص مزيفة واضحة**  
مسار TF‑IDF خفيف وغالباً يكتمل بأقل من 200ms — هذا طبيعي. إن كان التصنيف يخالف التوقع دائماً، قد يكون تدريبك يستخدم ترميز تسميات مختلف عن WELFake الافتراضي (0=مزيف، 1=حقيقي): التطبيق يفعّل افتراضياً **`TRUELENS_TEXT_AUTO_FAKE_CLASS=1`** ليستنتج فئة «المزيف» من نصّين مرجعيين عند أول طلب؛ لتعطيله استخدم **`TRUELENS_TEXT_AUTO_FAKE_CLASS=0`** وحدد **`TRUELENS_TEXT_FAKE_CLASS=1`** أو **`TRUELENS_TEXT_INVERT=1`** يدوياً. إن كانت دقة التدريب منخفضة (~50٪) فالنموذج قريب من التخمين العشوائي — أعد التدريب أو راجع عمود التسميات في CSV.  
يُمزج أيضاً (عند وجود **`predict_proba`**) احتمال النموذج مع **إشارة أسلوب صارخ/كليكبيت** عبر **`TRUELENS_TEXT_HEURISTIC_WEIGHT`** (افتراضي حوالي `0.52`)؛ لتعطيل المزج استخدم **`TRUELENS_TEXT_HEURISTIC_WEIGHT=0`**. عند عدة كلمات عاطفية والنموذج لا يزال يميل للحقيقي، **`TRUELENS_TEXT_STRONG_HEURISTIC=1`** (افتراضي) يعزّز الاحتمال قليلاً؛ عطّله بـ **`0`** إن لم ترد هذا السلوك.

**تعذّر تحميل نموذج الصور**  
تأكد من مسار `image_model.h5` وأن TensorFlow متوافق مع بايثون 3.10.

**الرسالة: Image model is not available رغم وجود الملف**  
تحقق مما يلي:
1. **المسار المتوقع:** الافتراضي `ml_models/saved_models/image_model.h5` (أو `image_model.tflite`). إن وضعت الملف تحت `ml_models/image_model.h5` أو `ml/image_model.h5` فالكود يبحث هناك أيضاً بعد التحديث.
2. **مؤشر Git LFS:** إن كان الملف **صغيراً جداً** (مئات البايت) ويبدأ بـ `version https://git-lfs.github.com` فهذا **مؤشر LFS** وليس النموذج — نفّذ `git lfs pull` أو انسخ الملف الثقيل يدوياً.
3. **`TRUELENS_IMAGE_ONLY_TFLITE`:** إن كانت مفعّلة والملف `.h5` فقط، لن يُحمّل الـ `.h5` — عطّل المتغير أو وفّر `image_model.tflite`.
4. **`requirements-app.txt`:** بيئة بلا TensorFlow كامل لا تفتح `.h5` — إمّا ثبّت `requirements.txt` للتطوير أو استخدم **TFLite** مع `image_model.tflite`.
5. **توقيت التحميل:** النماذج تُحمَّل في خيط خلفي؛ أول طلب تحليل **ينتظر** انتهاء التحميل (حتى `TRUELENS_MODEL_LOAD_TIMEOUT_SEC`، افتراضي 600 ث). للتحميل المتزامن عند الإقلاع: **`TRUELENS_EAGER_LOAD_MODELS=1`**.
6. **مسار مخصّص:** **`TRUELENS_IMAGE_MODEL_PATH`** = المسار الكامل لملف `.h5` أو `.tflite` إن أردت وضع النموذج خارج المشروع.

**صيغ الصور في الواجهة**  
التحقق من الامتداد يعتمد على المجموعة **`ALLOWED_IMAGE_EXTENSIONS`** في `config.py` (تشمل PNG، JPEG بأشكاله، WebP، GIF، BMP، TIFF، HEIC، ICO، PSD، JPEG2000، DDS، PCX، TGA، Netpbm، وغيرها مما يدعمه Pillow). ملفات **SVG** (متجه فقط) غير مدعومة للتحليل بالنموذج الحالي. صيغ **HEIC/HEIF** قد تحتاج مكتبات نظام أو `pillow-heif` على بعض الخوادم؛ إن فشل الفتح ستظهر رسالة خطأ من الخادم.

**رسالة «Image analysis failed: File type not allowed» مع ملف PNG/JPEG صحيح**  
كان استخراج الامتداد يعتمد على `secure_filename` الذي **يحذف الأحرف غير اللاتينية** من اسم الملف؛ فأسماء عربية أو بأحرف يونيكود قد تُفقد وتُرفض رغم أن الصيغة مدعومة. الإصلاح: أخذ الامتداد من **الاسم الأصلي** وحفظ نسخة على القرص باسم `uuid.امتداد` فقط. إن استمر الخطأ، تأكد أن الامتداد الحقيقي مسموح (وليس مثلاً `image.png.exe`).

**تعذّر التنزيل عبر الطرفية أو Kaggle API**  
استخدم التنزيل اليدوي من المتصفح كما في القسم [3.1](#31-التنزيل-اليدوي-للداتاسيتات)؛ لا حاجة لـ `kaggle.json` لتلك الطريقة.

**النماذج غير مرفوعة على GitHub وحجّة Render / السيرفر تطلب ملفات `saved_models`**  
المستودع يتجنّب تضمين `.h5`/`.pkl` لضخامتها. الخيارات: درّب محلياً كما في أقسام [3.2](#32-الخطوة-الأولى-تدريب-نموذج-الأخبار) و [3.3](#33-الخطوة-الثانية-تدريب-نموذج-الصور)، أو ارفع النماذج كـ Release/ملف عام واضبط متغيرات البيئة ثم خلال البناء يشغَّل السكربت `scripts/fetch_pretrained_models.py`:
- **`PRETRAINED_MODELS_BASE_URL`** — عنوان أساس يُكمَّل بأسماء الملفات الثلاثة (`image_model.h5`، `text_model.pkl`، `tfidf_vectorizer.pkl`).
- أو عناوين لكل ملف: **`IMAGE_MODEL_DOWNLOAD_URL`**، **`TEXT_MODEL_DOWNLOAD_URL`**، **`TFIDF_VECTORIZER_DOWNLOAD_URL`**.
- على خط البناء الذي يعتمد تنزيل النماذج: **`PRETRAINED_FETCH_REQUIRED=true`** لتفشل البناء إن لم يُعرَف أي رابط تنزيل.
- **`PRETRAINED_FETCH_FORCE=true`** إعادة تنزيل حتى لو الملفات موجودة؛ **`PRETRAINED_FETCH_TIMEOUT_SECONDS`** زمن انتظار أطول تنزيل (افتراضي 7200).

**نفاد الذاكرة على Render (مثلاً Ran out of memory 512MB) أو رفع ملفات غير لازمة**  
راجع القسم **[15 — نشر Render](#section-render-models)** أدناه: `requirements-app`، `image_model.tflite`، تنزيل النماذج بالروابط أثناء البناء، وما لا يُرفع مع Git.

**Permission denied عند قراءة `WELFake_Dataset.csv`**  
أغلق الملف إن كان مفتوحاً في Excel أو أي محرر، ثم أعد تشغيل السكربت. أو انسخ الملف إلى مجلد آخر واستخدم `python scripts/train_text_model.py --csv مسار\النسخة.csv`.

**الفيديو بطيء جداً**  
قلل طول الملف أو راقب عدد الإطارات في الإعدادات (`MAX_FRAMES_PER_VIDEO`).

**الشعار لا يظهر**  
ضع الملف `Logo.png` داخل `assets/images/` بالمسار المذكور في القالب.


---

## 11. نتائج التدريب (تشغيل فعلي)

سجلٌّ لأحد التشغيلات المحلية (Python 3.10، TensorFlow/Keras). المسارات التالية نسبةً لجذر المشروع `TrueLensAI/`.

### 11.1 نموذج النص (WELFake + TF-IDF + Passive Aggressive)

| الملف | المسار |
|--------|--------|
| المصنف المحفوظ | `ml_models/saved_models/text_model.pkl` |
| محوّل TF-IDF | `ml_models/saved_models/tfidf_vectorizer.pkl` |

### 11.2 نموذج الصور (CIFAKE — EfficientNetB0)

**البيانات:** وُجد **20 000** صورة ضمن **فئتين** (`REAL` / `FAKE`) أثناء التدريب.

**تدريب Keras (سطر التقدّم):**

| المقياس | القيمة |
|---------|--------|
| الخطوات | `2500/2500` |
| الزمن الإجمالي | ~5570 ثانية |
| متوسط الزمن لكل خطوة | ~2 ثانية/خطوة |
| `loss` | 0.6931 |
| `accuracy` | 0.5000 |
| `val_loss` | 0.6931 |
| `val_accuracy` | 0.5000 |

**النموذج المحفوظ:** `ml_models/saved_models/image_model.h5`

**تقييم على مجموعة اختبار محجوزة** (`datasate/archive/test` إن وُجدت):

| المقياس | القيمة |
|---------|--------|
| `loss` | 0.6931 |
| `accuracy` | 0.5000 |

### 11.3 ملاحظات من التشغيل

- طباعة السكربت: **تحذير** عندما تكون `val_accuracy` أقل من **90٪** — يُنصح بزيادة البيانات أو عدد الـ epochs أو مراجعة إعدادات التدريب.
- **تنبيه Keras:** عند الحفظ بصيغة HDF5 (`.h5`) قد يظهر تحذير بأن الصيغة legacy؛ البديل الموصى به هو الصيغة الأصلية لـ Keras مثل `my_model.keras` عند التحديث لاحقاً.
- دقة **0.50** و`loss` قريبة من **ln(2) ≈ 0.693** تشير غالباً إلى أن النموذج لا يتفوق على التخمين العشوائي على هذه الدفعة؛ يستحق التحقق من توازن الفئات، مسارات الصور، ومعدّل التعلم قبل الاعتماد على النتيجة في الإنتاج.

---

## 12. شرح الكود والمقتطفات

يوضّح هذا القسم تدفّق البرنامج من الإقلاع إلى معالجة طلب، مع مقتطفات من الملفات نفسها وبنفس أسلوب الشرح السابق (عربية تقنية، نقاط واضحة، وأكواد داخل كتل `python`).

### 12.1 نقطة الدخول — `app.py`

يُنشأ كائن Flask، تُحمَّل إعدادات `Config`، تُنشأ المجلدات اللازمة، تُهيَّأ قاعدة البيانات داخل سياق التطبيق، ثم تُحمَّل النماذج مرة واحدة، وتُسجَّل الـ blueprints للصفحات الرئيسية و`/detect` و`/history`، مع مسار ثابت لملفات `assets/`.

```python
def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object("config.Config")

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, "ml_models", "saved_models"), exist_ok=True)

    with app.app_context():
        init_db()

    init_models()

    app.register_blueprint(main_bp)
    app.register_blueprint(text_bp, url_prefix="/detect")
    app.register_blueprint(image_bp, url_prefix="/detect")
    app.register_blueprint(video_bp, url_prefix="/detect")
    app.register_blueprint(history_bp, url_prefix="/history")

    @app.route("/assets/<path:filename>")
    def assets_files(filename):
        return send_from_directory(os.path.join(Config.BASE_DIR, "assets"), filename)

    return app
```

**ملخص سريع:** `init_db()` يضمن وجود الجداول؛ `init_models()` يحاول تحميل ملفات `.pkl` و`.h5` إن وُجدت؛ التشغيل المباشر `python app.py` يستدعي `create_app()` ثم `run` على المنفذ 5000.

### 12.2 الإعدادات والمسارات — `config.py`

الفئة `Config` تجمع مسارات مطلقة مبنية على `BASE_DIR`: بيانات التدريب تحت `datasate/`، قاعدة SQLite تحت `database/truelens.db`، الرفع تحت `assets/uploads`، وحدود أحجام الملفات والامتدادات المسموحة، ومسارات النماذج المحفوظة.

```python
class Config:
    SECRET_KEY = "truelens-ai-secret-2024"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_ROOT = os.path.join(BASE_DIR, "datasate")
    WELFAKE_CSV_PATH = os.path.join(DATASET_ROOT, "WELFake_Dataset.csv")
    IMAGE_ARCHIVE_DIR = os.path.join(DATASET_ROOT, "archive")
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "truelens.db")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "assets", "uploads")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    TEXT_MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "text_model.pkl")
    TFIDF_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "tfidf_vectorizer.pkl")
    IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "image_model.h5")
    MAX_FRAMES_PER_VIDEO = 30
```

**ملخص سريع:** أي تعديل لموقع الداتا أو النماذج يمرّ عبر هذه الفئة لتبقى بقية الملفات متسقة.

### 12.3 الصفحة الرئيسية والإحصائيات — `routes/main_routes.py`

الصفحة `/` تقرأ العدادات من قاعدة البيانات؛ إذا كان `total_scans` صفراً تُعرض أرقام افتراضية للواجهة فقط (`stats_estimate` يُستخدم في القالب لإظهار ملاحظة «أرقام إرشادية»)، وإذا وُجدت فحوصات فعلية تُعرض القيم الحقيقية.

```python
_PLACEHOLDER = {
    "total_scans": 1248,
    "fake_detected": 612,
    "real_detected": 636,
}

def _home_stats_for_template():
    raw = get_stats()
    total = int(raw.get("total_scans") or 0)
    if total > 0:
        return {
            "total_scans": total,
            "fake_detected": int(raw.get("fake_detected") or 0),
            "real_detected": int(raw.get("real_detected") or 0),
            "stats_estimate": False,
        }
    return {
        "total_scans": _PLACEHOLDER["total_scans"],
        "fake_detected": _PLACEHOLDER["fake_detected"],
        "real_detected": _PLACEHOLDER["real_detected"],
        "stats_estimate": True,
    }

@main_bp.route("/")
def index():
    stats = _home_stats_for_template()
    return render_template("index.html", stats=stats)
```

**ملخص سريع:** مسار `/about` يعرض قالب `about.html` دون منطق إضافي.

### 12.4 قاعدة البيانات — `database/db_handler.py`

`init_db()` ينشئ جدول `detection_history` لسجل الحفظ اليدوي من الواجهة، وجدول `app_stats` بمفاتيح `total_scans` و`fake_detected` و`real_detected` مع `INSERT OR IGNORE` لتهيئة الصفوف مرة واحدة. `get_stats()` يجمع القيم في قاموس؛ `increment_stat` يزيد العداد الآمن حسب المفتاح.

```python
def init_db() -> None:
    with _cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS detection_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "detection_type TEXT NOT NULL,"
            "input_summary TEXT,"
            "result_label TEXT NOT NULL,"
            "confidence_score REAL NOT NULL,"
            "model_used TEXT,"
            "processing_time_ms INTEGER,"
            "file_name TEXT,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS app_stats ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "stat_key TEXT UNIQUE NOT NULL,"
            "stat_value INTEGER DEFAULT 0,"
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO app_stats (stat_key, stat_value) VALUES ('total_scans', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO app_stats (stat_key, stat_value) VALUES ('fake_detected', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO app_stats (stat_key, stat_value) VALUES ('real_detected', 0)"
        )

def get_stats() -> Dict[str, int]:
    with _cursor() as cur:
        cur.execute("SELECT stat_key, stat_value FROM app_stats")
        rows = cur.fetchall()
    out = {"total_scans": 0, "fake_detected": 0, "real_detected": 0}
    for r in rows:
        k = r["stat_key"]
        if k in out:
            out[k] = int(r["stat_value"])
    return out
```

### 12.5 تحميل النماذج عند الإقلاع — `ml_models/model_loader.py`

`init_models()` يتحقق من وجود الملفات على القرص قبل إنشاء كاشف النص أو الصورة؛ كاشف الفيديو يعتمد على نجاح تحميل نموذج الصور.

```python
def init_models() -> None:
    global _text_detector, _image_detector, _video_detector
    from ml_models.text_detector import TextDetector
    from ml_models.image_detector import ImageDetector
    from ml_models.video_detector import VideoDetector

    _text_detector = None
    if os.path.isfile(Config.TEXT_MODEL_PATH) and os.path.isfile(Config.TFIDF_PATH):
        try:
            _text_detector = TextDetector(Config.TEXT_MODEL_PATH, Config.TFIDF_PATH)
        except Exception:
            _text_detector = None

    _image_detector = None
    if os.path.isfile(Config.IMAGE_MODEL_PATH):
        try:
            _image_detector = ImageDetector(Config.IMAGE_MODEL_PATH)
        except Exception:
            _image_detector = None

    _video_detector = None
    if _image_detector is not None and _image_detector.is_loaded():
        _video_detector = VideoDetector(_image_detector)
```

**ملخص سريع:** إن فُقد ملف نموذج، يبقى الكاشف `None` وتُعاد رسالة خطأ واضحة من المسارات عند المحاولة.

### 12.6 مثال مسار تحليل النص — `routes/text_routes.py`

المسار `POST /detect/text/analyze` يستخرج JSON أو النموذج، يتحقق من المدخلات، يستدعي `det.predict`، يحدّث الإحصائيات عند نجاح التنبؤ، ويُرجع الاستجابة الموحّدة `success` / `data` / `error`.

```python
@text_bp.route("/text/analyze", methods=["POST"])
def analyze_text():
    det = get_text_detector()
    if det is None or not det.is_loaded():
        return _json(
            False,
            None,
            "نموذج النص غير متوفر. درّب النموذج وضع الملفات في مجلد ml_models/saved_models.",
        )
    payload = request.get_json(silent=True) or {}
    text = payload.get("text") or request.form.get("text") or ""
    ok, err = validate_text_input(text)
    if not ok:
        return _json(False, None, err)
    t0 = time.perf_counter()
    try:
        out = det.predict(text)
    except Exception as e:
        return _json(False, None, f"فشل التنبؤ: {str(e)}")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    summary = clip_for_summary(text, 200)
    increment_stat("total_scans")
    increment_stat("fake_detected" if out["label"] == "مزيف" else "real_detected")
    return _json(
        True,
        {
            "label": out["label"],
            "confidence": out["confidence"],
            "model": out.get("model"),
            "processing_time_ms": elapsed_ms,
            "input_summary": summary,
        },
        None,
    )
```

**ملخص سريع:** بعد كل تحليل ناجح من `/detect/*/analyze` ووجود جلسة مسجّلة، يُستدعى `save_detection` تلقائياً (`utils/history_auto.py`) فيُعاد `history_id` في JSON. مسارات **`/detect/*/save`** تبقى لحفظ **نسخة إضافية** أو عند تعطيل الحفظ التلقائي بـ **`TRUELENS_DISABLE_AUTO_HISTORY=1`**.

### 12.7 حفظ نتيجة النص — `routes/text_routes.py`

يستقبل JSON من الواجهة بعد عرض النتيجة؛ يتحقق من التسمية والثقة ثم يستدعي `save_detection` لنوع `text` دون اسم ملف.

```python
@text_bp.route("/text/save", methods=["POST"])
def save_text_result():
    payload = request.get_json(silent=True) or {}
    label = payload.get("label")
    confidence = payload.get("confidence")
    summary = payload.get("input_summary") or ""
    model = payload.get("model")
    time_ms = payload.get("processing_time_ms", 0)
    if label not in ("مزيف", "حقيقي"):
        return _json(False, None, "نتيجة غير صالحة.")
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return _json(False, None, "قيمة ثقة غير صالحة.")
    rid = save_detection(
        "text",
        summary,
        label,
        conf,
        model,
        int(time_ms),
        None,
    )
    return _json(True, {"id": rid}, None)
```

### 12.8 كشف الصورة — `routes/image_routes.py`

يتحقق من نموذج الصور، يستقبل `multipart`، يتحقق من الامتداد والحجم، يحفظ الملف مؤقتاً عبر `save_upload_temp`، يستدعي `det.predict(path)`، يحذف الملف في `finally`، ثم يحدّث الإحصائيات. مسار `/image/save` يشبه حفظ النص مع `file_name`.

```python
path, safe_name, _ext = save_upload_temp(f, Config.ALLOWED_IMAGE_EXTENSIONS)
t0 = time.perf_counter()
out = det.predict(path)
elapsed_ms = int((time.perf_counter() - t0) * 1000)
# ... finally: remove_file_safe(path)
increment_stat("total_scans")
increment_stat("fake_detected" if out["label"] == "مزيف" else "real_detected")
```

### 12.9 كشف الفيديو — `routes/video_routes.py`

يعتمد على `get_video_detector()` و`get_image_detector()` معاً؛ يحفظ الملف المرفوع، يستدعي `vdet.predict(path)` الذي يستخرج الإطارات داخلياً، ويُرجع للواجهة `frames_analyzed` و`std` إضافةً إلى الحقول المعتادة.

```python
vdet = get_video_detector()
img_det = get_image_detector()
if img_det is None or not img_det.is_loaded() or vdet is None:
    return _json(False, None, "نموذج الفيديو غير متوفر (يعتمد على نموذج الصور). ...")
# ...
out = vdet.predict(path)
```

### 12.10 السجل وواجهاته — `routes/history_routes.py`

صفحة HTML للسجل، و`GET /history/data` يعيد JSON بقائمة السجلات، وحذف سجل أو مسح كامل عبر `DELETE`.

```python
@history_bp.route("/data")
def history_data():
    rows = get_all_history(200)
    return _json(True, {"records": rows}, None)

@history_bp.route("/delete/<int:record_id>", methods=["DELETE"])
def delete_record(record_id: int):
    if delete_history_record(record_id):
        return _json(True, {"deleted": record_id}, None)
    return _json(False, None, "لم يُعثر على السجل.")
```

### 12.11 حذف السجل من قاعدة البيانات — `database/db_handler.py`

```python
def delete_history_record(record_id: int) -> bool:
    with _cursor() as cur:
        cur.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
        return cur.rowcount > 0

def clear_all_history() -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM detection_history")
```

### 12.12 كاشف النص أثناء التشغيل — `ml_models/text_detector.py`

يحمّل المصنف والمحوّل بـ `joblib`، ينظّف النص (`clean_text`) متوافقاً مع التدريب، يحوّل النص إلى متجه TF-IDF، يتنبأ بالفئة، ويستخرج الثقة من `predict_proba` أو `decision_function` عند توفرهما، ويُخرج التسمية بالعربية.

```python
def predict(self, text: str) -> dict:
    cleaned = self.clean_text(text)
    if not cleaned:
        return {"label": "مزيف", "confidence": 0.5, "model": self._model_name}
    X = self._vectorizer.transform([cleaned])
    pred = int(self._model.predict(X)[0])
    # ... confidence من proba أو sigmoid على decision_function
    label_ar = "حقيقي" if pred == 1 else "مزيف"
    return {"label": label_ar, "confidence": min(1.0, max(0.0, confidence)), "model": self._model_name}
```

**ملاحظة:** تطابق تسمية التدريب (0/1) مع «مزيف/حقيقي» يفترض أن النموذج المدرَّب يستخدم نفس الترميز.

### 12.13 كاشف الصورة — `ml_models/image_detector.py`

يحمّل نموذج Keras، يصغّر الصورة إلى 224×224، يطبّع القيم إلى [0,1]، ويفسّر خرج الـ sigmoid كاحتمال «حقيقي» ثم يشتق احتمال التزييف للعرض.

```python
raw = float(self.model.predict(batch, verbose=0)[0][0])
# CIFAKE: FAKE=0, REAL=1; sigmoid output is P(real), app uses p_fake = 1 - raw
p_fake = 1.0 - raw
label = "مزيف" if p_fake >= 0.5 else "حقيقي"
confidence = p_fake if label == "مزيف" else (1.0 - p_fake)
```

### 12.14 كاشف الفيديو — `ml_models/video_detector.py`

يختار حتى `MAX_FRAMES_PER_VIDEO` إطاراً موزّعاً على طول المقطع، يكتب كل إطار مؤقتاً كملف PNG، يمرّ على `image_detector.predict` ليجمع احتمالات التزييف، يحذف الملفات في `finally`، ثم يجمع المتوسط والانحراف المعياري ويقرّر التسمية.

```python
def predict(self, video_path: str) -> dict:
    frames, _duration = self.extract_frames(video_path, Config.MAX_FRAMES_PER_VIDEO)
    if not frames:
        return {"label": "مزيف", "confidence": 0.5, "frames_analyzed": 0, ...}
    scores = []
    try:
        for fp in frames:
            out = self.image_detector.predict(fp)
            fake_prob = (
                out["confidence"] if out["label"] == "مزيف" else (1.0 - out["confidence"])
            )
            scores.append(fake_prob)
    finally:
        for fp in frames:
            if os.path.isfile(fp):
                os.remove(fp)
    mean_fake = float(np.mean(scores)) if scores else 0.5
    # ...
```

### 12.15 الأدوات المساعدة — `utils/`

**التحقق (`validators.py`):** حدود طول **نص الخبر** (20–50000 حرف) مع رسائل توجّه المستخدم للصق مقال كامل؛ امتدادات الصور والفيديو، وأحجام من `Config`.

```python
def validate_text_input(text: str) -> tuple:
    if text is None:
        return False, "Paste the news article to analyze (headline + lead paragraph is best)."
    s = text.strip()
    if len(s) < 20:
        return False, "Please paste at least 20 characters of the news piece (headline and first sentences work best)."
    if len(s) > 50000:
        return False, "Article is too long (maximum 50000 characters)."
    return True, ""
```

**المعالجة المسبقة للنص (`preprocessor.py`):** تطبيع المسافات وقص الملخص للسجل.

```python
def clip_for_summary(text: str, max_len: int = 200) -> str:
    t = normalize_whitespace(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"
```

**الرفع (`file_handler.py`):** مجلد آمن؛ يُستخرج الامتداد من اسم الملف الأصلي (يدعم أسماء عربية/يونيكود)، ويُحفظ على القرص كـ **`{uuid}.{ext}`** لتفادي أحرف غير آمنة في المسارات؛ الاسم المعروض في الواجهة يبقى من `basename` الأصلي (مقصوصاً).

```python
def save_upload_temp(file_storage, allowed_ext) -> tuple:
    ensure_upload_dir()
    base = os.path.basename((file_storage.filename or "").replace("\\", "/"))
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    if ext not in allowed_ext:
        raise ValueError("File type not allowed.")
    path = os.path.join(Config.UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file_storage.save(path)
    return path, base[:200], ext
```

### 12.16 ثوابت السجل — `database/models.py`

تعرّف `DetectionRecord` و`DETECTION_TYPES` و`LABEL_FAKE` / `LABEL_REAL` لاستخدام اختياري وتوحيد القيم النصية.

```python
DETECTION_TYPES = ("text", "image", "video")
LABEL_FAKE = "مزيف"
LABEL_REAL = "حقيقي"
```

### 12.17 تدريب نموذج النص — `scripts/train_text_model.py`

يحلّ مسار CSV (بما فيه `WELFake_Dataset.csv` أو بحث glob)، يقرأ البيانات مع معالجة أخطاء القراءة على ويندوز، يبني أنبوب `TfidfVectorizer` + `PassiveAggressiveClassifier`، يقسّم train/test، يحفظ المصنف والمحوّل منفصلين ليتوافقا مع `TextDetector` الذي يحمّلهما كملفين.

```python
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), ...)),
        ("clf", PassiveAggressiveClassifier(loss="hinge", max_iter=2000, ...)),
    ]
)
pipeline.fit(X_train, y_train)
joblib.dump(pipeline.named_steps["clf"], OUT_MODEL)
joblib.dump(pipeline.named_steps["tfidf"], OUT_VEC)
```

### 12.18 تدريب نموذج الصور — `scripts/train_image_model.py`

يبني EfficientNetB0 مجمّداً ثم طبقات علوية، يدرّب بـ `ImageDataGenerator` مع `validation_split`، يستخدم EarlyStopping، ثم يفك التجميد جزئياً لجولة إضافية، يحفظ `image_model.h5`، ويُقيّم مجلد `archive/test` إن وُجد.

```python
def build_model(img_size: int = 224) -> keras.Model:
    base = EfficientNetB0(include_top=False, weights="imagenet", ...)
    base.trainable = False
    # ... GlobalAveragePooling2D، Dense، sigmoid
```

### 12.19 تنزيل الداتا من Kaggle — `scripts/download_dataset.py`

يتحقق من `kaggle.json` أو متغيرات البيئة، يستورد واجهة Kaggle، وينزّل مجموعات WELFake أو CIFAKE إلى تحت `datasate/`.

```python
def _has_kaggle_credentials():
    if os.path.isfile(_kaggle_json_path()):
        return True
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    return bool(user and key)
```

### 12.20 جدول مرجعي شامل للملفات المهمة

| الموضوع | الملف والعناصر الرئيسية |
|---------|-------------------------|
| إقلاع التطبيق والمسارات الثابتة | `app.py` — `create_app`؛ `config.py` — `Config` |
| الصفحة الرئيسية والعن المشروع | `routes/main_routes.py` — `index`, `about`, `_PLACEHOLDER` |
| تحميل النماذج مرة واحدة | `ml_models/model_loader.py` — `init_models`, `get_*_detector` |
| قاعدة البيانات | `database/db_handler.py` — `init_db`, `get_stats`, `increment_stat`, `save_detection`, `delete_history_record`, `clear_all_history` |
| ثوابت السجل | `database/models.py` — `DETECTION_TYPES`, التسميات |
| كشف النص (مسار وتحليل وحفظ) | `routes/text_routes.py` — `analyze_text`, `save_text_result` |
| كشف الصورة والفيديو | `routes/image_routes.py`, `routes/video_routes.py` |
| السجل في الواجهة | `routes/history_routes.py` — `history_data`, `delete_record`, `clear_history` |
| منطق التنبؤ | `ml_models/text_detector.py`, `image_detector.py`, `video_detector.py` |
| تحقق ورفع | `utils/validators.py`, `utils/file_handler.py`, `utils/preprocessor.py` |
| التدريب والتنزيل | `scripts/train_text_model.py`, `train_image_model.py`, `download_dataset.py` |

بهذا يغطي القسم 12 المسارات من الواجهة إلى قاعدة البيانات والنماذج وسكربتات التدريب، مع مقتطفات تربط الشرح بالملفات الفعلية وبنفس أسلوب المستند السابق.

---

<a id="section-accounts-dashboard-admin"></a>

## 13. الحسابات الافتراضية واللوحة والإدارة

<a id="section-default-admin-env"></a>

### 13.1 حساب مشرف جاهز (بدون أسرار داخل Git)

**لا يوجد** بريد/كلمة مرور ثابتة مكتوبة في الكود أو README. إن أردت **مشرفاً يُنشأ تلقائياً عند كل إقلاع** (بعد `init_db`)، عيّن **معاً** في البيئة (ملف `.env` محلي أو Render → Environment):

| المتغير | الغرض |
|---------|--------|
| `TRUELENS_DEFAULT_ADMIN_EMAIL` | البريد (يُطبَّع صغيراً) |
| `TRUELENS_DEFAULT_ADMIN_PASSWORD` | كلمة المرور الأولية (تُخزَّن مشفّرة في SQLite) |
| `TRUELENS_DEFAULT_ADMIN_DISPLAY_NAME` | اختياري؛ الافتراضي `Administrator` |

**السلوك:** إن لم يكن هناك مستخدم بهذا البريد، يُنشأ حساب **`is_admin = 1`**. إن وُجد المستخدم مسبقاً ولم يكن مشرفاً، يُرفع إلى مشرف **دون تغيير كلمة المرور**. إن كان مشرفاً مسبقاً، لا يُعدّل شيء.

**أمان:** لا تلصق قيماً حقيقية في README أو Git؛ غيّر كلمة المرور بعد أول دخول إن رغبت، وأزل المتغيرات من البيئة إن لم تعد تحتاج إعادة الزرع.

بدون هذه المتغيرات: أنشئ حساباً من **Sign up** (`/auth/register`) كأي مستخدم عادي، أو استخدم الطرق في الفقرة التالية.

### 13.2 ترقية مستخدم إلى «مشرف» (Admin)

- **`TRUELENS_DEFAULT_ADMIN_EMAIL`** + **`TRUELENS_DEFAULT_ADMIN_PASSWORD`**: كما في [13.1](#section-default-admin-env) — زرع تلقائي عند الإقلاع.
- **`TRUELENS_ADMIN_EMAIL`**: اضبطها في بيئة التشغيل (مثلاً Render → Environment) إلى **البريد بالضبط** الذي ستسجّل به حسابك. عند **أول** تسجيل ناجح بهذا البريد يُضاف المستخدم مع **`is_admin = 1`** ويظهر رابط **Admin** في القائمة.
- **`TRUELENS_BOOTSTRAP_ADMIN`**: إذا وضعتها `1` أو `true` وكان **عدد المستخدمين في قاعدة البيانات صفراً** عند التسجيل، يُمنح **أول** مستخدم يُسجَّل صلاحية مشرف (مفيد لأول نشر قبل وجود أي حساب). **يُنصح بإزالة القيمة أو تعطيلها بعد إنشاء المشرف** حتى لا يُمنح كل مستخدم جديد صلاحية بالخطأ إذا أُعيدت قاعدة بيانات فارغة.

لا تضع كلمات مرور حقيقية في المستودع؛ استخدم بريداً قوياً وكلمة مرور قوية وخزّن الأسرار في متغيرات البيئة على الخادم فقط.

### 13.3 ما الذي أضيف للواجهة؟

- **تسجيل / دخول / خروج / حسابي** (`/auth/...`) مع تغيير كلمة المرور واسم العرض.
- **لوحة Dashboard** (`/dashboard`): ملخص السجلات **المحفوظة** لحسابك (بما فيها المحفوظة تلقائياً بعد كل تحليل ناجح عند تسجيل الدخول؛ **التحليل** نفسه متاح بدون تسجيل).
- **صفحة تقرير** (`/report`): تفتح بعد التحليل من زر **Open full report** (تعتمد على الجلسة بعد طلب `analyze`؛ لا تُخزَّن صور ELA كبيرة في الكوكي).
- **Admin** (`/admin/` و `/admin/detections`): للمشرف فقط — قائمة مستخدمين وآخر سجلات فحص على مستوى النظام.

### 13.4 أسرار الجلسة على الإنتاج

- **`SECRET_KEY`**: عيّن قيمة عشوائية طويلة على Render (أو أي استضافة).
- **`SESSION_COOKIE_SECURE`**: عيّن `1` أو `true` عند العمل على **HTTPS** حتى لا تُرسل كوكي الجلسة على اتصال غير مشفّر.

---

<a id="section-uncertain-report-ela"></a>

## 14. النتائج غير المؤكدة والتقرير الكامل وELA وmetadata الصوت

### 14.1 تصنيف Uncertain (غير مؤكد)

إذا وقعت **ثقة النموذج** ضمن نطاق «منطقة رمادية»، تُعرض النتيجة **Uncertain** بدلاً من Fake/Real فقط، مع الإبقاء على **`model_label`** في استجابة JSON (القرار الخام للنموذج). النطاق الافتراضي قابل للضبط:

- **`TRUELENS_UNCERTAIN_LOW`** (افتراضي تقريباً `0.42`)
- **`TRUELENS_UNCERTAIN_HIGH`** (افتراضي تقريباً `0.58`)

أو عبر `config.py` (`UNCERTAIN_CONFIDENCE_LOW` / `HIGH`) عند التشغيل المحلي.

### 14.2 المؤشرات التوضيحية (Indicators)

بعد كل تحليل ناجح تُرجع الواجهة قائمة **`explanations`** (نصوص إرشادية: أسلوب الصياغة للنص، وشرح النموذج للصورة/الفيديو، إلخ). هذه **مكمّلات** للثقة وليست بديلاً عن التحقق من المصدر.

### 14.3 خريطة ELA للصور JPEG

لملفات **`.jpg` / `.jpeg`** فقط، يُحسب تمثيل **Error Level Analysis** ويُعرض في الصفحة كصورة مساعدة للتفرسة البصرية على مناطق ضغط غير طبيعية. الصيغ الأخرى لا تُولّد ELA (لتوفير الوقت والذاكرة).

### 14.4 الفيديو و«الصوت»

- التحليل الأساسي للفيديو ما زال **على الإطارات** باستخدام نموذج الصور (مثل السابق).
- إن وُجد **`ffprobe`** (ضمن حزمة **FFmpeg**) في `PATH` على السيرفر، يُسجَّل **وجود مسار صوتي أو لا** في الحاوية (`audio_summary` + تفاصيل في JSON). **لا يوجد** حالياً نموذج ذكاء اصطناعي للتحقق من صوت التزييف؛ الغرض إبلاغ المستخدم وتوثيق البنية التقنية كما في متطلبات التقرير.

---

<a id="section-render-models"></a>

## 15. نشر Render — النماذج والذاكرة وما لا يُرفع مع المستودع

الهدف: **ألا ترفع مع Git/Render** ملفات ضخمة أو غير لازمة للتشغيل، وتفادي **نفاد الذاكرة (مثلاً حد 512MB)** بسبب تحميل TensorFlow كامل أو نموذج `.h5` ثقيل أثناء التشغيل.

### 15.1 ما الذي لا يُرفع على Git (ومن ثم لا يُعاد بناؤه بلا داعٍ)؟

احرص على أن **لا تُلحق** بالمستودع (أو تُستثنى بـ `.gitignore`):

| نوع | أمثلة | السبب |
|-----|--------|--------|
| بيانات تدريب | مجلد `datasate/` بالكامل أو أرشيفات Kaggle | حجم كبير جداً؛ التدريب يكون محلياً ثم تُرفع **النماذج فقط** إن احتجت للسحابة |
| نماذج ثقيلة داخل Git | `*.h5`، `*.pkl`، `*.tflite` كملفات ثنائية داخل الفرع | تضخيم الريبو؛ استخدم **Release** أو تخزيناً عاماً + روابط تنزيل (انظر أدناه) |
| بيئة بايثون | `.venv/`، `venv/`، `__pycache__/` | تُعاد توليدها على الخادم |
| رفع المستخدمين | محتويات `assets/uploads/` | مؤقت وخصوصية |

**ما يُرفع مع الكود:** السكربتات، القوالب، `requirements-app.txt`، إعدادات البناء التي **تنزّل** النماذج أثناء البناء فقط.

### 15.2 حلّ مشكلة النماذج على Render بدون رفع ملفات زائدة

1. **درّب محلياً** (القسم [3.2](#32-الخطوة-الأولى-تدريب-نموذج-الأخبار) و [3.3](#33-الخطوة-الثانية-تدريب-نموذج-الصور)) ثم صدّر للإنتاج:
   - للصور: يُفضّل وجود **`image_model.tflite`** (أخف من تحميل TensorFlow كامل مع `.h5`). يوجد في المشروع مسار تصدير TFLite (انظر `scripts/export_image_model_tflite.py` ووثائق السكربتات إن وُجدت).
2. **ارفع ملفات النماذج فقط** إلى مكان يعطي **رابط HTTPS مباشر** للتحميل:
   - **GitHub Release** (مرفقات Assets) — شائع ومجاني للمستودعات العامة.
   - أو تخزين سحابي آخر برابط عام (R2، إلخ) حسب تفضيلك.
3. في **Render → Environment** عيّن مثلاً:
   - **`PRETRAINED_MODELS_BASE_URL`** = عنوان أساس ينتهي بمسار يُسمى بعده الملفات (`text_model.pkl`، `tfidf_vectorizer.pkl`، `image_model.tflite` أو `image_model.h5`)؛  
     **أو** روابط لكل ملف: **`TEXT_MODEL_DOWNLOAD_URL`**، **`TFIDF_VECTORIZER_DOWNLOAD_URL`**، **`IMAGE_MODEL_TFLITE_DOWNLOAD_URL`** (يُفضّل TFLite للإنتاج).
4. في **Build Command** تأكد من تشغيل **`scripts/fetch_pretrained_models.py`** (غالباً عبر `scripts/render_build.sh` إن كنت تستخدمه) حتى تُنزَل الملفات إلى `ml_models/saved_models/` **أثناء البناء** وليس كجزء من git.
5. **اعتماد التشغيل:** على Linux في الإنتاج استخدم **`requirements-app.txt`** (يحتوي `tflite-runtime` ولا يسحب TensorFlow كاملاً). تجنّب `requirements.txt` الكامل مع TensorFlow على خطة ذاكرة 512MB إن أمكن.
6. **ذاكرة التشغيل:** فعّل **`TRUELENS_IMAGE_ONLY_TFLITE=1`** إن أردت عدم محاولة تحميل `.h5`. استخدم **عامل Gunicorn واحد** (`web: gunicorn ...`) على الخطة المجانية لتجنّب تكرار النموذج في الذاكرة لكل عامل.
7. **اختياري:** **`TRUELENS_EAGER_LOAD_MODELS=1`** إن احتجت تحميلاً متزامناً عند الإقلاع (للتشخيص؛ قد يطيل بدء الحاوية).

### 15.3 استدلال عبر API (خادم ML منفصل — بلا نماذج على خادم الويب)

إذا أردت أن **لا تُرفع ولا تُحمَّل** ملفات النماذج على خدمة Render نفسها (لا `.pkl` ولا TensorFlow على الويب):

1. انشر **خادماً ثانياً** (Fly.io، Railway، VPS، أو خدمة Python أخرى على Render بذاكرة أوضح) وضع النماذج على **قرص ذلك الخادم فقط** تحت `ml_models/saved_models/` (أو عيّن **`TRUELENS_IMAGE_MODEL_PATH`** لمسار كامل).
2. على عامل الاستدلال: ثبّت **`pip install -r requirements-ml-inference.txt`** إن احتجت **`image_model.h5`** مع TensorFlow؛ وإن كان لديك **`image_model.tflite` فقط** يمكن الاكتفاء بـ **`requirements-app.txt`** على Linux بعد نسخ الملفات.
3. شغّل الخادم من جذر المشروع، مثلاً:  
   `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 services.ml_inference_server:app`
4. على **تطبيق TrueLens (Render)** عيّن:
   - **`TRUELENS_ML_API_BASE_URL`**: عنوان HTTPS لخادم الاستدلال (مثل `https://ml-worker.example.com`).
   - **`TRUELENS_ML_API_KEY`**: نفس السرّ على الخادمين؛ الطلبات ترسل **`Authorization: Bearer <المفتاح>`**. بدون مفتاح يبقى الخادم مفتوحاً — **للاختبار أو شبكة خاصة فقط**.
   - اختياري **`TRUELENS_ML_API_TIMEOUT_SEC`** (افتراضي `120`) لرفع الصور أو الفيديو البطيء.
5. **الفيديو:** ما زال يعمل عبر عدة طلبات صورة (إطاراً إطاراً) — قد يكون أبطأ عبر الإنترنت.

### 15.4 ربط «النماذج على السحابة» بمشكلة الذاكرة

وضع الملفات على رابط تنزيل **لا يقلل** ذاكرة التشغيل أثناء الاستدلال بنفسه؛ يقلل حجم الريبو ويحل مشكلة Git LFS. **تخفيف الذاكرة** يأتي من **TFLite + `tflite-runtime` + عدم تحميل TensorFlow كامل** ومن حد أعمال متزامنة معقول.

### 15.5 ملخص متغيرات بيئة مفيدة (جدول مرجعي سريع)

| المتغير | الغرض |
|---------|--------|
| `SECRET_KEY` | توقيع جلسات Flask |
| `SESSION_COOKIE_SECURE` | `1` على HTTPS |
| `PRETRAINED_MODELS_BASE_URL` | قاعدة URL تُكمَّل بأسماء الملفات (`text_model.pkl`، …) |
| `TEXT_MODEL_DOWNLOAD_URL` / `TFIDF_VECTORIZER_DOWNLOAD_URL` | روابط مباشرة لملفات النص |
| `IMAGE_MODEL_TFLITE_DOWNLOAD_URL` / `IMAGE_MODEL_DOWNLOAD_URL` | يُفضّل **TFLite** للإنتاج؛ `.h5` أثقل ويحتاج TensorFlow كاملاً |
| `PRETRAINED_FETCH_REQUIRED` | فشل البناء إن لم تُضبط الروابط |
| `TRUELENS_ML_API_BASE_URL` | عنوان خادم استدلال منفصل؛ عند التعيين لا يُحمّل النص/الصورة محلياً على خادم الويب |
| `TRUELENS_ML_API_KEY` | سرّ مشترك بين الويب والعامل؛ ترويسة `Authorization: Bearer …` |
| `TRUELENS_ML_API_TIMEOUT_SEC` | مهلة HTTP للاستدلال عن بُعد (افتراضي 120 ث) |
| `TRUELENS_DEFAULT_ADMIN_EMAIL` + `TRUELENS_DEFAULT_ADMIN_PASSWORD` | زرع مشرف عند الإقلاع (اختياري `TRUELENS_DEFAULT_ADMIN_DISPLAY_NAME`) |
| `TRUELENS_ADMIN_EMAIL` | أول تسجيل بهذا البريد = مشرف |
| `TRUELENS_BOOTSTRAP_ADMIN` | أول مستخدم في DB فارغة = مشرف (استخدم بحذر) |
| `TRUELENS_DISABLE_AUTO_HISTORY` | `1` لتعطيل الحفظ التلقائي في السجل بعد التحليل (الاعتماد على زر Save فقط) |
| `TRUELENS_TEXT_AUTO_FAKE_CLASS` | `1` (افتراضي) يستنتج أي رقم فئة يعني «مزيف» عبر نصّين مرجعيين عند أول تحليل؛ `0`/`off` لتعطيل والاعتماد على `TRUELENS_TEXT_FAKE_CLASS` فقط |
| `TRUELENS_TEXT_FAKE_CLASS` | `0` أو `1` يدوياً (يتجاوز الاستنتاج التلقائي عند التعيين) |
| `TRUELENS_TEXT_INVERT` | `1` يعكس احتمال الفئات الثنائية (تشخيص عند اختلاف التسميات عن WELFake) |
| `TRUELENS_TEXT_HEURISTIC_WEIGHT` | وزن (0–1) يمزج احتمال «مزيف» من النموذج مع درجة كلمات صارخة؛ `0` يعطّل المزج؛ افتراضي تقريباً `0.52` |
| `TRUELENS_TEXT_STRONG_HEURISTIC` | `1` (افتراضي) يعزّز النتيجة عند أربع إصابات أسلوبية أو أكثر إذا كان احتمال النموذج للمزيف ما زال أقل من 55٪؛ `0`/`off` لتعطيل |
| `TRUELENS_UNCERTAIN_LOW` / `HIGH` | نطاق عرض **Uncertain** |
| `TRUELENS_IMAGE_ONLY_TFLITE` | إجبار مسار TFLite فقط |
| `TRUELENS_EAGER_LOAD_MODELS` | تحميل النماذج عند الإقلاع مباشرة |
| `TRUELENS_IMAGE_MODEL_PATH` | مسار كامل لـ `image_model.h5` أو `.tflite` خارج المسارات الافتراضية |
| `TRUELENS_MODEL_LOAD_TIMEOUT_SEC` | أقصى انتظار (ثوانٍ) لانتهاء تحميل النماذج في الخيط قبل أول طلب تحليل (افتراضي 600) |

بهذا يبقى المستودع خفيفاً، ويمكن لـ Render إما تنزيل النماذج مرة عند البناء أو توجيه الاستدلال إلى **خادم ML منفصل** (القسم **15.3** أعلاه) بحيث لا تُحمَّل الأوزان على خادم الويب، والتشغيل يبقى ضمن حدود الذاكرة قدر الإمكان دون رفع داتاسيتات أو مكتبات زائدة مع الكود.

