"""Presentation helpers for the Streamlit dashboard."""

from __future__ import annotations

from html import escape

from perplexity_at_home.dashboard.models import DashboardThreadRecord

__all__ = [
    "build_mermaid_embed",
    "format_thread_label",
]


def format_thread_label(thread: DashboardThreadRecord) -> str:
    """Return the user-facing label for a thread selector option."""
    if thread.last_summary:
        return f"{thread.display_label} • {thread.last_summary[:48]}"
    return thread.display_label


def build_mermaid_embed(
    mermaid_text: str,
    *,
    title: str,
    subtitle: str,
) -> str:
    """Return embeddable HTML for an interactive Mermaid workflow panel."""
    escaped_title = escape(title)
    escaped_subtitle = escape(subtitle)
    escaped_mermaid = escape(mermaid_text)
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      body {{
        margin: 0;
        background:
          radial-gradient(circle at top left, rgba(243, 244, 246, 0.95), transparent 30%),
          linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
        color: #0f172a;
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .shell {{
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      }}
      .header {{
        margin-bottom: 0.85rem;
      }}
      .title {{
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
      }}
      .subtitle {{
        color: #475569;
        font-size: 0.9rem;
      }}
      .viewport {{
        min-height: 420px;
        border: 1px solid #dbeafe;
        border-radius: 16px;
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%);
        overflow: auto;
        padding: 0.5rem;
      }}
      .mermaid {{
        min-width: 680px;
      }}
      @media (max-width: 900px) {{
        .viewport {{
          min-height: 320px;
        }}
        .mermaid {{
          min-width: 520px;
        }}
      }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{
        startOnLoad: true,
        securityLevel: "loose",
        theme: "base",
        themeVariables: {{
          primaryColor: "#ffffff",
          primaryBorderColor: "#0f766e",
          primaryTextColor: "#0f172a",
          secondaryColor: "#ecfeff",
          tertiaryColor: "#eff6ff",
          lineColor: "#0f766e",
          background: "#ffffff",
          mainBkg: "#ffffff",
          fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        }}
      }});
    </script>
  </head>
  <body>
    <div class="shell">
      <div class="header">
        <div class="title">{escaped_title}</div>
        <div class="subtitle">{escaped_subtitle}</div>
      </div>
      <div class="viewport">
        <pre class="mermaid">{escaped_mermaid}</pre>
      </div>
    </div>
  </body>
</html>
""".strip()
