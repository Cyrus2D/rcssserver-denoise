#!/usr/bin/env python3
"""Render an Exp4 Markdown report, embedding local SVG plots in the HTML."""
import argparse
from pathlib import Path
import re
import markdown

STYLE = """
body{margin:0;background:#eef2f7;color:#172b4d;font:16px/1.6 system-ui,sans-serif}
main{max-width:1080px;margin:35px auto;background:white;padding:42px 48px;border-radius:14px}
h1{font-size:32px;line-height:1.2}h2{margin-top:46px;border-top:1px solid #dce4ed;padding-top:22px}h3{margin-top:30px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th{background:#edf3f8;text-align:left}
td,th{padding:11px 12px;border-bottom:1px solid #dce4ed}td:last-child{text-align:right}
code{overflow-wrap:anywhere;font-size:.87em;background:#edf3f8;padding:2px 4px;border-radius:3px}
pre{background:#edf3f8;padding:18px;overflow:auto}pre code{white-space:pre;overflow-wrap:normal}
svg{max-width:100%;height:auto;margin:22px 0;border:1px solid #dce4ed;border-radius:8px}
a{color:#075ea8}strong{font-weight:700}
@media(max-width:700px){main{padding:22px 15px;margin:0}table{font-size:12px;display:block;overflow:auto}}
@media print{main{margin:0;padding:0}h2{break-after:avoid}table,svg{break-inside:avoid}}
"""

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    args = p.parse_args()
    body = markdown.markdown(args.source.read_text(), extensions=["tables", "fenced_code"])
    def embed(match):
        source = re.search(r'src="([^"]+)"', match[0])
        path = args.source.parent / source[1] if source else None
        return path.read_text() if path and path.suffix == ".svg" and path.is_file() else match[0]
    body = re.sub(r"<img\b[^>]*>", embed, body)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Exp4 · Live data report</title><style>' + STYLE + "</style><main>" + body + "</main></html>")
    output = args.source.with_suffix(".html")
    output.write_text(page)
    print(output)

if __name__ == "__main__":
    main()
