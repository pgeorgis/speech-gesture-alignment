from HanTa import HanoverTagger as ht

from constants import (ASR_TEXT_KEY, ASR_TIMED_RESULTS_KEY, POS_TAG_KEY,
                       TOKEN_KEY)

POS_TAGGERS = {
    "de": ht.HanoverTagger('morphmodel_ger.pgz'),
    "en": ht.HanoverTagger('morphmodel_en.pgz'),
}

def get_pos_tagger(language: str):
    """Return the POS tagger for a specific language."""
    if language not in POS_TAGGERS:
        supported_language_codes = ", ".join(sorted(list(POS_TAGGERS.keys())))
        raise ValueError(f"Invalid language '{language}' provided. Supported language codes: {supported_language_codes}")
    return POS_TAGGERS[language]


def tokenize(text: str):
    """Perform basic tokenization by lowercasing and splitting by whitespace."""
    tokens = text.lower().split()
    return tokens


def pos_tag_text(text: str, language: str):
    """Add part-of-speech (POS) tags to words of text."""
    pos_tagger = get_pos_tagger(language)
    return pos_tagger.tag_sent(tokenize(text))


def pos_tag_asr_results(asr_results: dict, language: str):
    """Add part-of-speech (POS) annotation to ASR results json."""
    text = asr_results[ASR_TEXT_KEY]
    pos_tagged = pos_tag_text(text, language=language)
    for (word, _, pos_tag), word_entry in zip(pos_tagged, asr_results[ASR_TIMED_RESULTS_KEY]):
        assert word == word_entry[TOKEN_KEY]
        word_entry[POS_TAG_KEY] = pos_tag
    return asr_results
