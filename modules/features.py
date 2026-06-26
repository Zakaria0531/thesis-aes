"""Linguistic feature extraction (surface, readability, coherence, syntactic) for AES essays."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import textstat
from lexicalrichness import LexicalRichness

logger = logging.getLogger(__name__)

RANDOM_SEED  = 42
SPACY_MODEL  = "en_core_web_sm"
MATTR_WINDOW = 50

LEVEL_MAP: dict[str, int] = {
    "original": 0,
    "light":    1,
    "medium":   2,
    "heavy":    3,
}

_CONNECTIVE_PHRASES: list[str] = [
    "for example", "for instance", "in contrast", "in addition",
    "in conclusion", "on the other hand", "as a result", "in fact",
    "in other words", "to summarize", "that is to say",
]
_CONNECTIVE_WORDS: list[str] = [
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "additionally", "meanwhile", "although", "because",
    "since", "thus", "hence", "whereas", "nonetheless", "besides",
    "accordingly", "subsequently", "likewise", "conversely", "instead",
    "otherwise", "similarly", "specifically", "notably", "indeed",
    "certainly", "undoubtedly",
]

_CONNECTIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _CONNECTIVE_PHRASES + _CONNECTIVE_WORDS) + r")\b",
    re.IGNORECASE,
)


def load_spacy_model() -> spacy.language.Language:
    """Load en_core_web_sm with NER disabled."""
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    logger.info("Loaded spaCy model '%s'", SPACY_MODEL)
    return nlp


def extract_surface(text: str, doc: spacy.tokens.Doc) -> dict:
    """Return surface features: counts, word length, lexical diversity (MATTR/MTLD/HDD), and POS ratios."""
    tokens   = [t for t in doc if not t.is_space]
    n_tokens = max(len(tokens), 1)
    sentences = list(doc.sents)
    n_sents   = max(len(sentences), 1)

    avg_sent_len = sum(len([t for t in s if not t.is_space]) for s in sentences) / n_sents

    alpha_tokens = [t for t in tokens if t.is_alpha]
    avg_word_len = (
        sum(len(t.text) for t in alpha_tokens) / len(alpha_tokens)
        if alpha_tokens else float("nan")
    )

    try:
        lex    = LexicalRichness(text)
        window = min(MATTR_WINDOW, max(lex.words, 1))
        mattr  = lex.mattr(window_size=window)
        mtld   = lex.mtld()
        try:
            hdd = lex.hdd()
        except Exception:
            hdd = float("nan")
    except (ZeroDivisionError, ValueError):
        mattr = mtld = hdd = float("nan")

    _TARGET_POS = {"NOUN", "VERB", "ADJ", "ADV"}
    pos_counts: dict[str, int] = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "other": 0}
    for token in tokens:
        pos_counts[token.pos_ if token.pos_ in _TARGET_POS else "other"] += 1

    return {
        "avg_sent_len":    avg_sent_len,
        "word_count":      len(tokens),
        "sentence_count":  len(sentences),
        "avg_word_len":    avg_word_len,
        "mattr":           mattr,
        "mtld":            mtld,
        "hdd":             hdd,
        "pos_noun_ratio":  pos_counts["NOUN"]  / n_tokens,
        "pos_verb_ratio":  pos_counts["VERB"]  / n_tokens,
        "pos_adj_ratio":   pos_counts["ADJ"]   / n_tokens,
        "pos_adv_ratio":   pos_counts["ADV"]   / n_tokens,
        "pos_other_ratio": pos_counts["other"] / n_tokens,
    }


def extract_readability(text: str) -> dict:
    """Return seven grade-level readability indices via textstat."""
    return {
        "flesch_kincaid_grade":         textstat.flesch_kincaid_grade(text),
        "coleman_liau_index":           textstat.coleman_liau_index(text),
        "gunning_fog":                  textstat.gunning_fog(text),
        "smog_index":                   textstat.smog_index(text),
        "automated_readability_index":  textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "linsear_write_formula":        textstat.linsear_write_formula(text),
    }


def extract_coherence(doc: spacy.tokens.Doc) -> dict:
    """Return connective frequency per sentence and mean adjacent-sentence Jaccard overlap of content lemmas."""
    sentences = list(doc.sents)
    n_sents   = max(len(sentences), 1)

    connective_freq = len(_CONNECTIVE_RE.findall(doc.text)) / n_sents

    overlaps: list[float] = []
    for sent_a, sent_b in zip(sentences[:-1], sentences[1:]):
        lemmas_a = {t.lemma_.lower() for t in sent_a if t.is_alpha and not t.is_stop}
        lemmas_b = {t.lemma_.lower() for t in sent_b if t.is_alpha and not t.is_stop}
        union = lemmas_a | lemmas_b
        if union:
            overlaps.append(len(lemmas_a & lemmas_b) / len(union))

    return {
        "connective_freq":     connective_freq,
        "avg_lexical_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def extract_syntactic(doc: spacy.tokens.Doc) -> dict:
    """Return dependency-parse features: tree depth, subordinate clause ratio, passive ratio, NP modifier count, pronoun density."""
    sentences = list(doc.sents)
    n_sents   = max(len(sentences), 1)
    tokens    = [t for t in doc if not t.is_space]
    n_tokens  = max(len(tokens), 1)

    def _tree_depth(token: spacy.tokens.Token) -> int:
        children = list(token.children)
        return 0 if not children else 1 + max(_tree_depth(c) for c in children)

    mean_dep_tree_depth = float(np.mean([_tree_depth(s.root) for s in sentences])) if sentences else 0.0

    subordinate_clause_ratio = sum(1 for t in tokens if t.dep_ in {"advcl", "relcl"}) / n_sents

    passive_ratio = sum(
        1 for s in sentences if any(t.dep_ in {"nsubjpass", "auxpass"} for t in s)
    ) / n_sents

    noun_chunks = list(doc.noun_chunks)
    mean_noun_phrase_modifiers = (
        float(np.mean([sum(1 for t in chunk if t != chunk.root) for chunk in noun_chunks]))
        if noun_chunks else 0.0
    )

    pronoun_density = sum(1 for t in tokens if t.pos_ == "PRON") / n_tokens

    return {
        "mean_dep_tree_depth":        mean_dep_tree_depth,
        "subordinate_clause_ratio":   subordinate_clause_ratio,
        "passive_ratio":              passive_ratio,
        "mean_noun_phrase_modifiers": mean_noun_phrase_modifiers,
        "pronoun_density":            pronoun_density,
    }


def featurize_essay(text: str, doc: spacy.tokens.Doc) -> dict:
    return {
        **extract_surface(text, doc),
        **extract_readability(text),
        **extract_coherence(doc),
        **extract_syntactic(doc),
    }


def get_features(text: str, nlp: spacy.language.Language) -> dict:
    return featurize_essay(text, nlp(text))


def load_essays(data_dir: Path) -> pd.DataFrame:
    """Walk data_dir and return a DataFrame of all essays across prompts and intervention levels."""
    records: list[dict] = []

    for prompt_dir in sorted(data_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue

        scores_path = prompt_dir / "scores.csv"
        if not scores_path.exists():
            logger.warning("No scores.csv in '%s' — skipping", prompt_dir.name)
            continue

        score_lookup: dict[str, float] = (
            pd.read_csv(scores_path).set_index("essay_id")["score"].to_dict()
        )

        for level_name, level_int in LEVEL_MAP.items():
            level_dir = prompt_dir / level_name
            if not level_dir.exists():
                continue

            for txt_path in sorted(level_dir.glob("*.txt")):
                essay_id = txt_path.stem
                score    = score_lookup.get(essay_id)
                if score is None:
                    logger.warning("essay_id '%s' missing from scores.csv (%s) — skipping", essay_id, prompt_dir.name)
                    continue
                records.append({
                    "essay_id":    essay_id,
                    "prompt_name": prompt_dir.name,
                    "level":       level_int,
                    "score":       float(score),
                    "text":        txt_path.read_text(encoding="utf-8"),
                })

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d essays (%d prompts, levels: %s)",
        len(df),
        df["prompt_name"].nunique() if not df.empty else 0,
        sorted(df["level"].unique().tolist()) if not df.empty else [],
    )
    return df
