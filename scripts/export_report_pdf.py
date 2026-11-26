"""Render the final Markdown report as a self-contained PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown


REPORT_CSS = """
@page {
  size: A4;
  margin: 18mm 17mm 20mm;
  @bottom-center {
    content: counter(page);
    color: #64748b;
    font-size: 8pt;
  }
}
body {
  color: #172033;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 9.2pt;
  line-height: 1.45;
}
h1, h2, h3 { color: #12213d; page-break-after: avoid; }
h1 { font-size: 21pt; text-align: center; margin: 12mm 0 8mm; }
h2 {
  border-bottom: 1px solid #cbd5e1;
  font-size: 14pt;
  margin-top: 8mm;
  padding-bottom: 1.5mm;
}
h3 { font-size: 11pt; margin-top: 5mm; }
p { margin: 2.2mm 0; text-align: justify; }
li { margin-bottom: 1mm; }
table {
  border-collapse: collapse;
  font-size: 7.5pt;
  margin: 4mm auto;
  page-break-inside: avoid;
  width: 100%;
}
th, td {
  border: 0.4pt solid #94a3b8;
  padding: 1.4mm;
  text-align: left;
}
th { background: #eaf0f8; }
img {
  display: block;
  height: auto;
  margin: 4mm auto;
  max-height: 185mm;
  max-width: 100%;
  page-break-inside: avoid;
}
blockquote {
  border-left: 2.5mm solid #93a9c7;
  color: #334155;
  margin: 3mm 0;
  padding: 1mm 4mm;
}
code {
  background: #eef2f7;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8pt;
  padding: 0.2mm 0.7mm;
}
pre {
  background: #eef2f7;
  overflow-wrap: anywhere;
  padding: 3mm;
  white-space: pre-wrap;
}
a { color: #245ea8; text-decoration: none; }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/final_report.md"),
        help="Markdown report path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/final_report.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()

    try:
        from weasyprint import CSS as WeasyCSS
        from weasyprint import HTML
    except ImportError as exc:
        raise SystemExit(
            "PDF export requires WeasyPrint: python -m pip install weasyprint"
        ) from exc

    source = args.input.resolve()
    output = args.output.resolve()
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Sycophancy Benchmark Final Report</title></head>"
        f"<body>{body}</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(source.parent)).write_pdf(
        output,
        stylesheets=[WeasyCSS(string=REPORT_CSS)],
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
