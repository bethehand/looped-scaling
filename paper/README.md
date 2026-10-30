# Paper: What a Loop Buys

*What a Loop Buys: Parameter Efficiency on Text, Extendable Depth on a Serial Task*, the version of 2026-10-30, rewritten for readability; no result changed, and one statement about seed noise was
corrected after the rewrite (first draft 2026-10-17; the deviation log lists what changed).
**[Read the PDF](paper.pdf)**. The technical report in [docs/report.md](../docs/report.md) covers the same study
in more detail; the paper reorganises it around one question and adds a review of related work.

## Build

The paper uses the TMLR LaTeX template and builds with [Tectonic](https://tectonic-typesetting.github.io/), a
self-contained LaTeX engine (`brew install tectonic`; any TeX Live with `pdflatex` and `bibtex` works too).

    cd paper
    mkdir -p build                                       # build/ is not tracked; Tectonic does not create it
    ../code/.venv/bin/python scripts/make_figures.py     # figures from code/results/ (prints the plotted numbers)
    tectonic -X compile preprint.tex --outdir build      # with author and repository link -> build/preprint.pdf (= paper.pdf)
    tectonic -X compile main.tex --outdir build          # anonymous submission version -> build/main.pdf

## Layout

| | |
|---|---|
| `main.tex` | preamble; builds the anonymous version by default |
| `preprint.tex` | defines `\PREPRINT` and inputs `main.tex`: author block and repository link |
| `sections/` | abstract, sections 1–8 and the appendix |
| `references.bib` | generated from arXiv metadata; venues only where the arXiv page states one |
| `figures/` | made by `scripts/make_figures.py` from the committed results (PDF for the paper, PNG previews) |
| `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` | the TMLR style files, Apache License 2.0 (`TMLR_STYLE_LICENSE`) |

Every number in the paper comes from `code/results/`; the figure script prints the numbers it plots.
