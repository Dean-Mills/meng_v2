#!/usr/bin/env bash
# Agent build for the CVIU paper. Writes ONLY into build_agent/ so Dean's IDE build of this directory
# (latexmk on save) is never touched. Run alone - not alongside other shell commands (see
# docs/dissertation_revision_guide.md, "NOTE (agent workflow)").
set -u
cd "$(dirname "$0")"
mkdir -p build_agent
run_tex() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_agent main.tex > /dev/null 2>&1; }
run_tex || { echo "pdflatex pass 1 FAILED"; grep -n -A3 '^!' build_agent/main.log | head -40; exit 1; }
( cd build_agent && BIBINPUTS="..:" BSTINPUTS="..:" bibtex main > bibtex.out 2>&1 ); BIB=$?
run_tex && run_tex || { echo "pdflatex pass 2/3 FAILED"; grep -n -A3 '^!' build_agent/main.log | head -40; exit 1; }
echo "bibtex exit $BIB; $(grep -c -i 'warning' build_agent/bibtex.out) bibtex warnings"
grep -i -E 'warning--|error' build_agent/bibtex.out | head -10
echo "pages: $(grep -o 'Output written on .*' build_agent/main.log)"
echo "errors: $(grep -c '^!' build_agent/main.log); undefined refs/cites: $(grep -c -E 'Reference .* undefined|Citation .* undefined' build_agent/main.log); overfull: $(grep -c 'Overfull' build_agent/main.log)"
grep -E 'LaTeX Warning|Package .* Warning' build_agent/main.log | grep -v -E 'There were undefined|Rerun|Label\(s\) may have changed' | sort | uniq -c | sort -rn | head -12
