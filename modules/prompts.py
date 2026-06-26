"""System prompts for GPT-4o-mini intervention levels L1 (light) through L3 (heavy)."""

PROMPTS: dict[str, str] = {
    "light": (
        "Fix only surface-level errors: spelling mistakes, "
        "grammar errors, and punctuation issues in the student essay below. "
        "Do not change word choice, sentence structure, or any ideas. "
        "Return only the corrected essay text with no commentary."
    ),
    "medium": (
        "Improve the student essay below by: "
        "(1) correcting all spelling, grammar, and punctuation errors, "
        "(2) improving sentence clarity and variety where needed, and "
        "(3) replacing weak or imprecise word choices with stronger alternatives. "
        "Do not add new arguments, examples, or ideas, and do not reorganize paragraphs. "
        "Return only the improved essay text with no commentary."
    ),
    "heavy": (
        "Substantially improve the student "
        "essay below while preserving its original arguments, and examples. "
        "Make comprehensive improvements to: (1) grammar, spelling, and punctuation, "
        "(2) vocabulary and academic register, (3) sentence variety and fluency, "
        "(4) paragraph structure and internal coherence, and (5) logical flow and "
        "transitions between ideas. The essay should sound like an advanced, polished "
        "version of the original student's essay, and not a replacement. "
        "Return only the revised essay text with no commentary."
    ),
}

LEVELS: list[str] = ["light", "medium", "heavy"]
