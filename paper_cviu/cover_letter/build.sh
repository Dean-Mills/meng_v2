#!/usr/bin/env bash
# Agent build for the cover letter. Writes ONLY into build_agent/ so Dean's IDE build of this directory
# (latexmk on save) is never touched. Run alone - not alongside other shell commands.
set -u
cd "$(dirname "$0")"
mkdir -p build_agent
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_agent cover_letter.tex > /dev/null 2>&1 || { echo "pdflatex FAILED"; grep -n -A3 '^!' build_agent/cover_letter.log | head -20; exit 1; }
echo "pages: $(grep -o 'Output written on .*' build_agent/cover_letter.log)"
echo "errors: $(grep -c '^!' build_agent/cover_letter.log); overfull: $(grep -c 'Overfull' build_agent/cover_letter.log)"
