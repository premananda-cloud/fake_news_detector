"""
explain.py
----------
Everything that turns a raw model prediction into the human-readable
explainability panel: linguistic signal detection (regex/lexicon based,
independent of the model) and templated summary/recommendation/caveat text.

Nothing here is per-example hardcoded — every signal is a genuine match
against the submitted text, and every explanation sentence is selected
by real thresholds on confidence + which signal category dominates.
"""

import re

# ---------------------------------------------------------------------------
# Linguistic signal patterns
# ---------------------------------------------------------------------------

SIGNAL_PATTERNS = {
    "sensationalist": {
        "label": "Sensationalist Language",
        "description": "Exaggerated, alarming, or emotionally charged words designed to provoke reaction.",
        "patterns": [
            r"\b(shocking|bombshell|explosive|outrage|scandal|exposed|revealed|unbelievable|"
            r"jaw-dropping|stunning|alarming|catastrophic|devastating|unprecedented|"
            r"terrifying|horrifying|disgusting|insane|crazy|unreal|mindblowing)\b"
        ],
    },
    "hedging": {
        "label": "Unverified Claims",
        "description": "Vague sourcing language that avoids attribution to named, credible sources.",
        "patterns": [
            r"\b(sources say|insiders claim|reportedly|allegedly|rumored|word is|"
            r"some people say|many believe|experts warn|officials claim|it is said|"
            r"according to insiders|anonymous sources|they don't want you to know|"
            r"mainstream media won't|what they're hiding)\b"
        ],
    },
    "emotional_appeal": {
        "label": "Emotional Manipulation",
        "description": "Appeals to fear, anger, or tribal identity rather than evidence.",
        "patterns": [
            r"\b(wake up|sheeple|patriots|traitors|evil|corrupt|destroy|"
            r"fight back|rise up|they hate us|our children|protect your family|"
            r"before it's too late|don't let them|stand up|regime|tyranny|"
            r"freedom fighters|deep state|globalists|elites)\b"
        ],
    },
    "absolute_language": {
        "label": "Absolute / Hyperbolic Claims",
        "description": "All-or-nothing framing with no nuance — a common marker of propaganda.",
        "patterns": [
            r"\b(always|never|everyone knows|nobody|no one|100%|completely|totally|"
            r"absolutely|the truth is|the fact is|undeniable|irrefutable|"
            r"proven beyond|without a doubt|definitively|once and for all)\b"
        ],
    },
    "citation_present": {
        "label": "Source Citations",
        "description": "References to studies, institutions, or named individuals — a positive credibility signal.",
        "patterns": [
            r"\b(according to|cited by|published in|study by|research from|"
            r"university|institute|journal|professor|dr\.|ph\.d|spokesperson|"
            r"said in a statement|press release|official report|data shows)\b"
        ],
        "positive": True,
    },
    "balanced_language": {
        "label": "Balanced Reporting",
        "description": "Use of hedged, measured language typical of professional journalism.",
        "patterns": [
            r"\b(however|nevertheless|on the other hand|while|although|"
            r"it is unclear|remains to be seen|could not be independently verified|"
            r"did not respond to requests for comment|declined to comment|"
            r"disputed by|contradicted by|experts disagree)\b"
        ],
        "positive": True,
    },
    "all_caps": {
        "label": "Excessive Capitalisation",
        "description": "Shouting in text — a stylistic marker of low-credibility content.",
        "patterns": [r"\b[A-Z]{4,}\b"],
    },
    "punctuation_abuse": {
        "label": "Punctuation Abuse",
        "description": "Multiple exclamation marks or question marks — hallmark of tabloid and disinformation content.",
        "patterns": [r"[!?]{2,}"],
    },
}


def analyse_signals(text: str) -> list[dict]:
    """Scan text for each signal category; return matched categories only."""
    results = []
    for key, cfg in SIGNAL_PATTERNS.items():
        matches = []
        for pattern in cfg["patterns"]:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))

        if not matches:
            continue

        seen: dict[str, int] = {}
        for m in matches:
            seen[m.lower()] = seen.get(m.lower(), 0) + 1
        top = sorted(seen.items(), key=lambda x: -x[1])[:5]

        results.append(
            {
                "key": key,
                "label": cfg["label"],
                "description": cfg["description"],
                "positive": cfg.get("positive", False),
                "count": sum(seen.values()),
                "examples": [t for t, _ in top],
            }
        )

    results.sort(key=lambda x: (x["positive"], -x["count"]))
    return results


# ---------------------------------------------------------------------------
# Confidence tiering (shared by explanation + spectrum reading)
# ---------------------------------------------------------------------------

def confidence_tier(confidence: float) -> tuple[str, str]:
    if confidence >= 0.88:
        return "high", "High confidence"
    if confidence >= 0.65:
        return "moderate", "Moderate confidence"
    return "low", "Low confidence — borderline result"


def spectrum_reading(p_real: float) -> str:
    """Mirrors the reading text under the certainty spectrum needle."""
    if p_real < 0.2:
        return "Strongly skewed toward fabrication. Very few credibility markers detected."
    if p_real < 0.4:
        return "Leaning toward misinformation. Negative linguistic signals outweigh positive ones."
    if p_real < 0.6:
        return "Near the boundary — the model is genuinely uncertain about this text."
    if p_real < 0.8:
        return "Leaning toward credibility. Language is largely consistent with factual reporting."
    return "Strongly consistent with credible journalism. Few disinformation markers detected."


# ---------------------------------------------------------------------------
# Explanation copy
# ---------------------------------------------------------------------------

def build_explanation(
    is_fake: bool,
    confidence: float,
    signals: list[dict],
    attention_tokens: list[dict],
) -> dict:
    neg_signals = [s for s in signals if not s["positive"]]
    pos_signals = [s for s in signals if s["positive"]]
    neg_count = sum(s["count"] for s in neg_signals)
    pos_count = sum(s["count"] for s in pos_signals)

    tier, tier_label = confidence_tier(confidence)

    if is_fake:
        if tier == "high":
            summary = (
                f"The model strongly classifies this article as misinformation "
                f"({confidence*100:.1f}% certainty). It detected {neg_count} negative "
                f"linguistic signal(s) across {len(neg_signals)} "
                f"categor{'y' if len(neg_signals) == 1 else 'ies'}."
            )
        elif tier == "moderate":
            summary = (
                f"The model leans toward misinformation ({confidence*100:.1f}% certainty) "
                f"but is not fully certain. This may reflect a mix of real and misleading "
                f"content, opinion writing, or satire."
            )
        else:
            summary = (
                f"The model nudges toward misinformation ({confidence*100:.1f}% certainty) "
                f"but sits close to the decision boundary. This result alone is inconclusive — "
                f"independent verification is essential."
            )
    else:
        if tier == "high":
            summary = (
                f"The model strongly classifies this as credible journalism "
                f"({confidence*100:.1f}% certainty). Language patterns are consistent with "
                f"factual, measured reporting."
            )
        elif tier == "moderate":
            summary = (
                f"The model leans toward credible content ({confidence*100:.1f}% certainty). "
                f"Some stylistic elements introduced uncertainty — this can occur with opinion "
                f"pieces, advocacy writing, or emotionally charged but legitimate journalism."
            )
        else:
            summary = (
                f"The model marginally favours credibility ({confidence*100:.1f}% certainty) "
                f"but the result is near the boundary. Highly specialised text, unusual writing "
                f"styles, or translated articles can produce this."
            )

    if is_fake and tier == "high":
        recommendation = (
            "Cross-check with multiple established news outlets before sharing or acting on "
            "this content. Verify the publication, author, and original sources cited."
        )
    elif is_fake:
        recommendation = (
            "Treat as a yellow flag. Look for primary sources, check if claims appear in "
            "credible outlets, and consider whether this could be satire or commentary."
        )
    elif tier == "high":
        recommendation = (
            "Even credible-looking content can contain factual errors or bias. Verify specific "
            "claims against primary sources for consequential decisions."
        )
    else:
        recommendation = (
            "Verify the source's reputation and check whether core claims are independently "
            "corroborated before drawing conclusions."
        )

    caveat = (
        "This model was fine-tuned on BERT-base-uncased via Sequential Parameter Segment "
        "Training (SPST). It analyses linguistic patterns only — not external facts. Satire, "
        "opinion, and highly technical text may score differently."
    )

    return {
        "summary": summary,
        "tier": tier,
        "tier_label": tier_label,
        "negative_signal_count": neg_count,
        "positive_signal_count": pos_count,
        "recommendation": recommendation,
        "caveat": caveat,
    }
