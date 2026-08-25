"""
components.py
--------------
render_results(data) -> HTML string

This is a direct port of the original Veritas front-end's renderResults()
JS function into Python. It produces one self-contained HTML fragment for
a gr.HTML() output component, using the exact class names from theme.css
so the ported CSS applies unchanged.

Note on motion: the original used requestAnimationFrame to grow bars from
0% on each result. Script execution inside dynamically-injected HTML isn't
guaranteed across Gradio versions, so bars/needle are rendered at their
final position directly (safe), with a CSS fade/slide-in on the whole
results block for the "reveal" feel instead of per-element growth.
"""

import html


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def render_results(data: dict) -> str:
    is_fake = data["is_fake"]
    confidence = data["confidence"]
    p_fake = data["probability"]["fake"]
    p_real = data["probability"]["real"]
    expl = data.get("explanation", {})
    signals = data.get("signals", [])
    attention = data.get("attention", [])
    preview = data.get("extracted_text_preview")
    filename = data.get("filename")

    verdict_class = "fake" if is_fake else "real"
    verdict_word = "Fake" if is_fake else "Real"
    verdict_headline = "Likely Misinformation" if is_fake else "Likely Credible"
    tier_label = expl.get("tier_label", "")

    # -- signals grid ----------------------------------------------------
    if not signals:
        signals_html = '<div class="no-signals">No notable linguistic signals detected.</div>'
    else:
        cards = []
        for sig in signals:
            polarity = "positive" if sig["positive"] else "negative"
            tags = "".join(
                f'<span class="signal-tag">{_esc(e)}</span>' for e in sig["examples"]
            )
            cards.append(
                f"""
            <div class="signal-item {polarity}">
              <div class="signal-header">
                <span class="signal-label {polarity}">{_esc(sig['label'])}</span>
                <span class="signal-count">{sig['count']}&times;</span>
              </div>
              <div class="signal-desc">{_esc(sig['description'])}</div>
              <div class="signal-examples">{tags}</div>
            </div>"""
            )
        signals_html = "".join(cards)

    # -- token cloud -------------------------------------------------------
    token_spans = []
    for t in attention:
        tok = t["token"][2:] if t["token"].startswith("##") else t["token"]
        score = t["score"]
        alpha = 0.15 + score * 0.8
        if is_fake:
            border = f"rgba(176,48,32,{alpha:.3f})"
            bg = f"rgba(176,48,32,{alpha * 0.18:.3f})"
            color = "var(--red)" if score > 0.55 else "var(--ink-soft)"
        else:
            border = f"rgba(26,107,60,{alpha:.3f})"
            bg = f"rgba(26,107,60,{alpha * 0.18:.3f})"
            color = "var(--green)" if score > 0.55 else "var(--ink-soft)"
        token_spans.append(
            f'<span class="token" title="Attention score: {score}" '
            f'style="border-color:{border};background:{bg};color:{color};">'
            f"{_esc(tok)}</span>"
        )
    token_cloud_html = "".join(token_spans) if token_spans else (
        '<div class="no-signals">No attention data available.</div>'
    )

    # -- preview block -------------------------------------------------------
    if preview:
        file_label = f'<div class="preview-file-label">File: {_esc(filename)}</div>' if filename else ""
        preview_html = f"""
    <div class="preview-block" style="display:block">
      <div class="card">
        <div class="card-head"><div class="card-eyebrow">Extracted Text Preview</div></div>
        <div class="card-body">
          {file_label}
          <div class="preview-quote">{_esc(preview)}&hellip;</div>
        </div>
      </div>
    </div>"""
    else:
        preview_html = ""

    spectrum_text = data.get("spectrum_reading", "")

    return f"""
<div id="results" class="visible">

  <div class="verdict-banner">
    <div class="verdict-accent {verdict_class}"></div>
    <div class="verdict-main">
      <div class="verdict-stamp show {verdict_class}">{verdict_word}</div>
      <div class="verdict-copy">
        <div class="verdict-headline {verdict_class}">{verdict_headline}</div>
        <div class="verdict-subline">{_esc(tier_label)} &middot; {_pct(confidence)} certainty</div>
      </div>
    </div>
    <div class="verdict-confidence">
      <div class="conf-value {verdict_class}">{_pct(confidence)}</div>
      <div class="conf-label">Confidence</div>
    </div>
  </div>

  <div class="explanation-card">
    <div class="card-head"><div class="card-eyebrow">Step 02 &middot; Analysis Explanation</div></div>
    <div class="card-body">
      <div class="explanation-summary">{_esc(expl.get('summary', ''))}</div>
      <div class="explanation-recommendation {verdict_class}">{_esc(expl.get('recommendation', ''))}</div>
      <div class="explanation-caveat">{_esc(expl.get('caveat', ''))}</div>
    </div>
  </div>

  <div class="results-grid">
    <div class="card">
      <div class="card-head"><div class="card-eyebrow">Model Probabilities</div></div>
      <div class="card-body">
        <div class="prob-row">
          <div class="prob-label">
            <span class="prob-label-text">Misinformation</span>
            <span class="prob-label-pct fake">{_pct(p_fake)}</span>
          </div>
          <div class="prob-track"><div class="prob-fill fake" style="width:{_pct(p_fake)}"></div></div>
        </div>
        <div class="prob-row">
          <div class="prob-label">
            <span class="prob-label-text">Credible</span>
            <span class="prob-label-pct real">{_pct(p_real)}</span>
          </div>
          <div class="prob-track"><div class="prob-fill real" style="width:{_pct(p_real)}"></div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><div class="card-eyebrow">Certainty Spectrum</div></div>
      <div class="card-body">
        <div class="spectrum-track">
          <div class="spectrum-needle" style="left:{p_real * 100:.1f}%"></div>
        </div>
        <div class="spectrum-labels">
          <span>Clearly Fake</span><span>Borderline</span><span>Clearly Real</span>
        </div>
        <div class="spectrum-reading">{_esc(spectrum_text)}</div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px">
    <div class="card-head"><div class="card-eyebrow">Step 03 &middot; Linguistic Signal Breakdown</div></div>
    <div class="card-body">
      <div class="signals-grid">{signals_html}</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px">
    <div class="card-head"><div class="card-eyebrow">Step 04 &middot; Model Attention &mdash; Key Tokens</div></div>
    <div class="card-body">
      <div class="token-cloud">{token_cloud_html}</div>
      <div class="token-legend">
        Colour intensity reflects relative attention weight assigned by the model's [CLS] token.
        These are the words most influencing the final prediction. Darker = higher weight.
      </div>
    </div>
  </div>

  {preview_html}
</div>
"""


def render_error(message: str) -> str:
    return f'<div class="error-box visible">{_esc(message)}</div>'


def render_empty() -> str:
    """Shown before the first analysis has run."""
    return '<div id="results"></div>'
