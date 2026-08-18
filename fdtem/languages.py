"""Language codes shared by every experiment."""
from __future__ import annotations

# Languages
# ════════════════════════════════════════════════════════════════════════════
# Map ISO-639-1 → FLORES-200 code. The core set mirrors the Bio-MQM training
# pairs (en/de/es/fr/ru/zh) so the analysis lines up with the fine-tuning; the
# extended set adds typological + script diversity (incl. low-resource swh).
FLORES_CODE = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "tr": "tur_Latn", "vi": "vie_Latn",
    "sw": "swh_Latn", "el": "ell_Grek", "th": "tha_Thai", "bg": "bul_Cyrl",
    "ur": "urd_Arab",
}
CORE_LANGS = ["en", "de", "es", "fr", "ru", "zh"]

# Languages with no whitespace word segmentation — perturbations operate on
# characters instead of space-delimited tokens for these.
NO_SPACE_LANGS = {"zh", "ja", "th"}


