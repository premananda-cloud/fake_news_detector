"""
app.py
------
Veritas — Fake News Detector, as a Gradio app.

Run with:
    python -m gradio_ui.app
from the project root (so the relative `model/` path resolves correctly).
"""

from datetime import datetime
from pathlib import Path

import gradio as gr

from . import inference, extract, explain, components

_HERE = Path(__file__).resolve().parent
CSS = (_HERE / "theme.css").read_text(encoding="utf-8")

MAX_PREVIEW_CHARS = 300


def _header_html() -> str:
    now = datetime.now()
    dateline = (
        now.strftime("%A, %B %-d, %Y")
        + "  ·  Veritas Integrity Engine  ·  BERT-base-uncased  ·  SPST fine-tuned"
    )
    return f"""
<header class="vt-header">
  <div class="vt-header-inner">
    <div class="vt-masthead-brand">
      <div class="vt-masthead-title"><em>Veritas</em></div>
      <div class="vt-masthead-rule-v"></div>
      <div class="vt-masthead-meta">
        <div class="vt-masthead-meta-line">News Integrity Engine</div>
        <div class="vt-masthead-meta-line"><strong>BERT</strong> &middot; SPST fine-tuned &middot; Attention-based explainer</div>
      </div>
    </div>
    <div class="vt-header-spacer"></div>
    <div class="vt-header-badge"><div class="vt-badge-dot"></div>Model online</div>
  </div>
</header>
<div class="vt-dateline-bar">{dateline}</div>
"""


def _section_head_html() -> str:
    return """
<div class="vt-section-head">
  <div class="vt-section-eyebrow">Step 01<span class="vt-section-eyebrow-sep">&middot;</span></div>
  <div class="vt-section-title-sm">Submit Article for Analysis</div>
</div>
"""


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

TAB_ORDER = ["text", "url", "file"]


def _on_tab_select(evt: gr.SelectData):
    idx = evt.index if evt.index is not None else 0
    return TAB_ORDER[idx] if idx < len(TAB_ORDER) else "text"


def _char_count(text: str) -> str:
    n = len(text or "")
    return f'<div class="vt-char-count">{n:,} character{"" if n == 1 else "s"}</div>'


def run_analysis(text: str, url: str, file_path: str, active_tab: str):
    preview = None
    filename = None
    try:
        if active_tab == "url":
            article_text = extract.fetch_url_text(url)
            preview = article_text[:MAX_PREVIEW_CHARS]
        elif active_tab == "file":
            article_text, filename = extract.extract_file_text(file_path)
            preview = article_text[:MAX_PREVIEW_CHARS]
        else:
            article_text = text

        article_text = extract.validate_text(article_text)

        detector = inference.get_detector()
        result = detector.analyze(article_text)
        result["signals"] = explain.analyse_signals(article_text)
        result["explanation"] = explain.build_explanation(
            is_fake=result["is_fake"],
            confidence=result["confidence"],
            signals=result["signals"],
            attention_tokens=result["attention"],
        )
        result["spectrum_reading"] = explain.spectrum_reading(result["probability"]["real"])
        if preview:
            result["extracted_text_preview"] = preview
        if filename:
            result["filename"] = filename

        return components.render_results(result), ""

    except extract.ExtractionError as e:
        return components.render_empty(), components.render_error(str(e))
    except Exception as e:  # noqa: BLE001 — surface unexpected errors in-UI too
        return components.render_empty(), components.render_error(f"Unexpected error: {e}")


def clear_all():
    return "", "", None, components.render_empty(), ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Veritas — Fake News Detector") as demo:
        active_tab = gr.State("text")

        with gr.Column(elem_id="veritas-app"):
            gr.HTML(_header_html())

            with gr.Column(elem_id="vt-container", elem_classes="vt-container"):
                gr.HTML(_section_head_html())

                with gr.Column(elem_classes="vt-input-body"):
                    with gr.Tabs() as tabs:
                        with gr.Tab("Paste Text"):
                            text_input = gr.Textbox(
                                lines=8,
                                max_lines=14,
                                placeholder="Paste article text here — headline, body copy, or both…",
                                elem_id="text-input",
                                show_label=False,
                            )
                            char_count = gr.HTML(_char_count(""))
                        with gr.Tab("URL"):
                            url_input = gr.Textbox(
                                placeholder="https://example.com/article-to-check",
                                elem_id="url-input",
                                show_label=False,
                            )
                        with gr.Tab("Upload File"):
                            file_input = gr.File(
                                file_types=[".txt", ".pdf"],
                                elem_id="file-drop",
                                elem_classes="vt-file-drop",
                                label="Drop a .txt or .pdf here, or click to browse",
                            )

                    with gr.Row(elem_classes="vt-action-row"):
                        analyze_btn = gr.Button("Analyse Article", elem_id="analyze-btn")
                        clear_btn = gr.Button("✕ Clear", elem_id="clear-btn")

                    error_box = gr.HTML("")

                results_html = gr.HTML(components.render_empty())

        # -- wiring ---------------------------------------------------------
        tabs.select(fn=_on_tab_select, outputs=active_tab)
        text_input.change(fn=_char_count, inputs=text_input, outputs=char_count)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[text_input, url_input, file_input, active_tab],
            outputs=[results_html, error_box],
        )
        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, url_input, file_input, results_html, error_box],
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
