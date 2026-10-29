#!/usr/bin/env python3
"""
sorting-morphogenesis/build_substack_article_html.py —
Build the museum-style, one-click-copy preview
`docs/substack_levin_latent_space_article.html` from
`docs/SUBSTACK_LEVIN_LATENT_SPACE_ARTICLE.md`.

Design follows `thompson/build_aristotle_1913_html.py` (the annotated 1913
"Aristotle as a Biologist" reader): cream paper on a dark navy ground, Georgia
serif body, numbered Plates with sans-serif provenance captions, sans-serif
"wall labels" under each gallery heading, and blue-bordered curator cards for
the verdicts.

Everything that carries the museum feel is expressed through *structure* that
survives pasting into Substack (headings, italic paragraphs, blockquotes,
images followed by italic captions). The CSS only dresses that structure for
the GitHub Pages preview.

Markdown conventions recognised:
  * `![](url)` followed by a paragraph starting `*Plate N — ...*`  -> figure + plate caption
  * a paragraph starting `*Wall label · ...*`                       -> wall label
  * a blockquote starting `**Curator’s verdict`                     -> curator card
  * the first italic paragraph after the H1                         -> subtitle
  * an italic paragraph starting `*A museum-style walkthrough`      -> frontispiece note

Usage:
  venv/bin/python sorting-morphogenesis/build_substack_article_html.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    {
        "md": ROOT / "docs" / "SUBSTACK_LEVIN_LATENT_SPACE_ARTICLE.md",
        "html": ROOT / "docs" / "substack_levin_latent_space_article.html",
        "label": "Main article",
        "peer_href": "substack_levin_latent_space_claims.html",
        "peer_label": "Claims Walkthrough",
        "src_href": "SUBSTACK_LEVIN_LATENT_SPACE_ARTICLE.md",
    },
    {
        "md": ROOT / "docs" / "SUBSTACK_LEVIN_LATENT_SPACE_CLAIMS.md",
        "html": ROOT / "docs" / "substack_levin_latent_space_claims.html",
        "label": "Claims walkthrough",
        "peer_href": "substack_levin_latent_space_article.html",
        "peer_label": "Main Article",
        "src_href": "SUBSTACK_LEVIN_LATENT_SPACE_CLAIMS.md",
    },
]

CSS = r"""
    :root {
      --ink: #1e293b;
      --paper: #fdfbf7;
      --muted: #475569;
      --rule: #cbd5e1;
      --accent: #0284c7;
      --accent-soft: #f0f9ff;
      --verdict-ink: #0f172a;
    }
    body {
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: var(--ink);
      font-family: "Georgia", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      line-height: 1.76;
      font-size: 18px;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 100;
      background: #0b1325;
      color: #f8fafc;
      padding: 12px 24px;
      border-bottom: 2px solid #38bdf8;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
    }
    .topbar a {
      color: #7dd3fc;
      text-decoration: none;
      font-weight: 600;
      margin-left: 14px;
    }
    .topbar a:hover { text-decoration: underline; }
    .copy-btn {
      background: #22c55e;
      color: #052e16;
      border: none;
      border-radius: 6px;
      padding: 9px 18px;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .copy-btn:hover { background: #4ade80; transform: translateY(-1px); }
    main.reader {
      max-width: 900px;
      margin: 32px auto 80px auto;
      background: var(--paper);
      padding: 52px 60px;
      border-radius: 12px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
    }
    @media (max-width: 768px) {
      main.reader { padding: 28px 22px; margin: 12px; }
      .topbar { flex-wrap: wrap; }
    }
    h1 {
      font-size: 2.1rem;
      line-height: 1.25;
      color: #0f172a;
      margin-top: 0;
      margin-bottom: 10px;
    }
    p.subtitle {
      font-size: 1.08rem;
      color: #334155;
      margin-top: 0;
    }
    p.frontispiece {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 0.9rem;
      color: var(--muted);
      border-top: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      padding: 10px 0;
      text-align: left;
    }
    h2 {
      font-size: 1.45rem;
      line-height: 1.3;
      color: #0f172a;
      border-bottom: 2px solid var(--rule);
      padding-bottom: 6px;
      margin-top: 48px;
    }
    h3 {
      font-size: 1.15rem;
      color: #0f172a;
      margin-top: 36px;
    }
    p { margin: 16px 0; text-align: justify; }
    p.wall-label {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 0.84rem;
      line-height: 1.55;
      color: var(--muted);
      letter-spacing: 0.01em;
      text-align: left;
      background: #f8fafc;
      border: 1px solid var(--rule);
      border-left: 4px solid #64748b;
      padding: 10px 14px;
      border-radius: 4px;
      margin: -4px 0 22px 0;
    }
    figure.plate-figure {
      margin: 28px 0 10px 0;
      text-align: center;
    }
    figure.plate-figure img {
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--rule);
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
    }
    p.plate-caption {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 0.86rem;
      line-height: 1.5;
      color: var(--muted);
      text-align: left;
      background: #f1f5f9;
      padding: 10px 14px;
      border-radius: 6px;
      margin-top: 6px;
      margin-bottom: 28px;
    }
    /* In Markdown the caption is italic and titles/variables are "un-italicised";
       in the museum preview we invert that: upright sans caption, italic titles. */
    p.plate-caption { font-style: italic; }
    p.plate-caption em { font-style: normal; }
    p.plate-caption code { font-style: normal; }
    p.wall-label, p.frontispiece { font-style: italic; }
    p.wall-label em, p.frontispiece em { font-style: normal; }
    blockquote.curator-card {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      background: var(--accent-soft);
      border-left: 5px solid var(--accent);
      border-radius: 6px;
      padding: 14px 20px;
      margin: 24px 0;
      color: var(--verdict-ink);
      font-size: 0.95rem;
      line-height: 1.62;
      box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);
    }
    blockquote.curator-card p { margin: 0; text-align: left; }
    blockquote.source-quote {
      margin: 20px 36px;
      padding-left: 18px;
      border-left: 3px solid #94a3b8;
      color: #334155;
    }
    blockquote.source-quote p { text-align: left; }
    hr { border: none; border-top: 1px solid var(--rule); margin: 36px 0; }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: #e2e8f0;
      color: #0f172a;
      padding: 1px 5px;
      border-radius: 4px;
      font-size: 0.84em;
    }
    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 14px 18px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 0.82rem;
      line-height: 1.45;
    }
    pre code { background: transparent; color: inherit; padding: 0; font-size: inherit; }
    a { color: var(--accent); text-decoration: underline; }
    ul, ol { padding-left: 22px; }
    li { margin-bottom: 8px; }
    li p { margin: 6px 0; text-align: left; }
"""

SCRIPT = r"""
    function copyForSubstack(btn) {
      const el = document.getElementById("article-content");
      const range = document.createRange();
      range.selectNodeContents(el);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      try {
        document.execCommand("copy");
        btn.textContent = "✔ Copied Rich HTML + Images! Now Paste (Cmd+V) in Substack";
        btn.style.background = "#38bdf8";
        setTimeout(() => {
          btn.textContent = "📋 Copy Entire Article for Substack";
          btn.style.background = "#22c55e";
        }, 4000);
      } catch (err) {
        alert("Press Cmd+C now to copy the selected article!");
      }
    }
"""


def slugify_heading(text: str) -> str:
    plain = re.sub(r"<[^>]+>", "", text)
    m = re.match(r"Claim (\d+)", plain)
    if m:
        return f"claim-{m.group(1)}"
    if plain.startswith("Claims 5"):
        return "claims-5-7"
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
    return slug or "section"


def add_heading_ids(body: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tag, inner = match.group(1), match.group(2)
        return f'<{tag} id="{slugify_heading(inner)}">{inner}</{tag}>'

    return re.sub(r"<(h[1-3])>(.*?)</\1>", repl, body, flags=re.S)


def render_body(md_text: str) -> str:
    # Substack does not render inline LaTeX; show the one formula as code.
    md_text = re.sub(r"\$\\mathcal\{B\}_3 = \\\{L_1, D_1, D_2\\\}\$", "`B_3 = {L_1, D_1, D_2}`", md_text)
    assert "$" not in md_text, "unexpected remaining inline math"

    md = MarkdownIt("commonmark", {"html": False, "typographer": False})
    body = md.render(md_text)
    body = add_heading_ids(body)

    # External links open in a new tab.
    body = re.sub(r'<a href="(https?://[^"]+)">', r'<a href="\1" target="_blank" rel="noopener">', body)

    # Images -> plate figures. In the Markdown the caption follows the image on the
    # next line without a blank line, so CommonMark puts both in one paragraph;
    # split the image out into its own <figure> and leave the caption as a <p>.
    body = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)" ?/?>\n',
        r'<figure class="plate-figure"><img src="\1" alt="\2" loading="lazy" /></figure>\n<p>',
        body,
    )
    body = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)" ?/?></p>',
        r'<figure class="plate-figure"><img src="\1" alt="\2" loading="lazy" /></figure>',
        body,
    )
    # Plate captions (italic paragraph starting "Plate N —").
    body = re.sub(
        r'<p><em>(Plate [IVXLC]+ — .*?)</p>',
        r'<p class="plate-caption"><em>\1</p>',
        body,
        flags=re.S,
    )
    # Wall labels.
    body = re.sub(
        r'<p><em>(Wall label · .*?)</p>',
        r'<p class="wall-label"><em>\1</p>',
        body,
        flags=re.S,
    )
    # Frontispiece note.
    body = re.sub(
        r'<p><em>(A museum-style walkthrough.*?)</p>',
        r'<p class="frontispiece"><em>\1</p>',
        body,
        flags=re.S,
    )
    # Subtitle: first italic paragraph after the H1.
    body = re.sub(
        r'(</h1>\n)<p><em>',
        r'\1<p class="subtitle"><em>',
        body,
        count=1,
    )
    # Curator cards vs. source quotations.
    body = re.sub(
        r'<blockquote>\n<p><strong>Verdict',
        r'<blockquote class="curator-card">\n<p><strong>Verdict',
        body,
    )
    body = body.replace("<blockquote>\n", '<blockquote class="source-quote">\n')

    # XHTML-style void tags, for consistency with the 1913 reader.
    body = body.replace("<hr>", "<hr />").replace("<br>", "<br />")
    return body


def write_page(page: dict) -> None:
    src = page["md"].read_text(encoding="utf-8")
    body = render_body(src)

    h1 = re.search(r"^# (.+)$", src, flags=re.M).group(1)
    h1_plain = html.escape(re.sub(r"[*`]", "", h1))

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Substack Museum Edition — {h1_plain}</title>
  <style>{CSS}  </style>
</head>
<body>
  <div class="topbar">
    <div><strong>Project Delphi</strong> · {page["label"]} — <em>{h1_plain}</em></div>
    <div>
      <button class="copy-btn" onclick="copyForSubstack(this)">📋 Copy Entire Article for Substack</button>
      <a href="{page["peer_href"]}">{page["peer_label"]}</a>
      <a href="{page["src_href"]}">Markdown Source</a>
      <a href="thompson_1913_aristotle_as_a_biologist.html">1913 Aristotle Reader</a>
    </div>
  </div>
  <main class="reader" id="article-content">
{body}  </main>
  <script>{SCRIPT}  </script>
</body>
</html>
"""
    page["html"].write_text(doc, encoding="utf-8")
    print(f"Saved {page['html']} ({len(doc):,} bytes)")
    for token in ("**", "](http"):
        print(f"  leftover {token!r}: {body.count(token)}")
    print(f"  plates: {body.count('class=\"plate-figure\"')} figures / {body.count('class=\"plate-caption\"')} captions")
    print(f"  wall labels: {body.count('class=\"wall-label\"')}, curator cards: {body.count('class=\"curator-card\"')}")


def main() -> None:
    for page in PAGES:
        write_page(page)


if __name__ == "__main__":
    main()
