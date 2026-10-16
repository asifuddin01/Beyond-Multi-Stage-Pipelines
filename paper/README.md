# Paper

`main.tex` is the IEEE conference-format (IEEEtran) source of the arXiv preprint;
`main.pdf` is the compiled 8-page paper.

```bash
pdflatex main.tex && pdflatex main.tex
```

* `figures/` — figures used by the paper (PDF).
* `figures_src/architecture.tex` — TikZ source of Fig. 1 (`pdflatex architecture.tex`).
* The data figures (Figs. 2–4) are produced by `scripts/make_paper_figures.py` from `results/`.
* The bibliography is embedded in `main.tex`, so no BibTeX run is needed.
