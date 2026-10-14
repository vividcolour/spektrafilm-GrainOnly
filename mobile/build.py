#!/usr/bin/env python3
"""Assemble Spektra Pocket into single-file pages.

  dist/spektra-pocket.html  page body for publishing as a claude.ai Artifact
  dist/index.html           standalone page: open it in any phone browser

Usage: python3 mobile/build.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def main():
    app = read("app.html")
    data = read("spektra_data.js")
    engine = read("engine.js")
    body = app.replace("<!--DATA-->", f'<script id="spektra-data">\n{data}</script>').replace(
        "<!--ENGINE-->", f'<script id="spektra-engine">\n{engine}</script>')
    os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
    with open(os.path.join(HERE, "dist", "spektra-pocket.html"), "w", encoding="utf-8") as f:
        f.write(body)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<meta name="theme-color" content="#12100e">'
            '<style>*{box-sizing:border-box}:root{padding-top:env(safe-area-inset-top,0px);'
            'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}[hidden]{display:none!important}</style>'
            '</head><body>')
    with open(os.path.join(HERE, "dist", "index.html"), "w", encoding="utf-8") as f:
        f.write(head + body + "</body></html>")
    print("wrote dist/spektra-pocket.html and dist/index.html",
          f"({os.path.getsize(os.path.join(HERE, 'dist', 'index.html')) // 1024} KB)")


if __name__ == "__main__":
    main()
