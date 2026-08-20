# Springer LNCS LaTeX Research Manuscript

## Overview
This directory contains the self-contained Springer LNCS LaTeX project for the research paper:

> **"Robust Human Activity Recognition through Federated Learning with Differential Privacy: A Comparison of Baseline and Centralized Models"**  
> *Authors:* Jonath Jimmi, Nishanth Shet, Usha Moorthy  
> *Affiliation:* Manipal Institute of Technology Bengaluru

---

## Directory Structure

```text
SPRINGER_LATEX/
├── main.tex            # Master LaTeX entry point
├── main.pdf            # Compiled PDF manuscript (12 pages)
├── sections/           # Modular document sections
│   ├── 00_metadata.tex # Title, authors, abstract, keywords
│   ├── 01_introduction.tex
│   ├── 02_dataset.tex
│   ├── 03_eda.tex
│   ├── 04_models.tex
│   ├── 05_metrics.tex
│   ├── 06_results.tex
│   ├── 07_xai.tex
│   └── 08_conclusion.tex
├── figures/            # 12 Figure PNG assets (fig1.png - fig10b.png)
├── styles/             # Package inclusions and macro definitions
│   ├── packages.tex
│   ├── macros.tex
│   └── commands.tex
├── references.bib      # BibTeX bibliography file
├── llncs.cls           # Official Springer LNCS Document Class v2.24
└── splncs04.bst        # Official Springer LNCS BibTeX Style
```

---

## Compilation Instructions

To compile the manuscript using `pdflatex` and `bibtex`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
