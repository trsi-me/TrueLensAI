# -*- coding: utf-8 -*-
# Shared sensational / clickbait token lists for heuristics + optional blend with ML score.

_EMOTION_WORDS_EN = (
    "shocking",
    "urgent",
    "miracle",
    "guaranteed",
    "destroy",
    "secret",
    "they don't want",
    "you won't believe",
    "breaking",
    "immediately",
    "catastrophe",
    "horrible",
)
_EMOTION_WORDS_AR = (
    "صادم",
    "فضيحة",
    "مؤامرة",
    "سر خطير",
    "لن تصدق",
    "كارثة",
    "فورا",
    "عاجل",
    "حقيقة صادمة",
    "الحقيقة المخفية",
)


def emotion_hit_count(text: str) -> int:
    t = (text or "").lower()
    hits_en = sum(1 for w in _EMOTION_WORDS_EN if w in t)
    hits_ar = sum(1 for w in _EMOTION_WORDS_AR if w in (text or ""))
    return hits_en + hits_ar
