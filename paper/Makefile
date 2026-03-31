TEX = pdflatex -interaction=nonstopmode
BIB = bibtex
SRC = main
PY = python

all: figures $(SRC).pdf

figures:
	$(PY) -c "import os; os.makedirs('figures', exist_ok=True)"
	$(PY) ../scripts/gen_sls_figure.py --output figures/sls_kernel.pdf
	cd ../docs && $(TEX) pipeline.tex
	$(PY) -c "import shutil; shutil.copy('../docs/pipeline.pdf', 'figures/pipeline.pdf')"

$(SRC).pdf: $(SRC).tex references.bib checklist.tex neurips_2026.sty
	$(TEX) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	$(PY) -c "import os; files=['$(SRC).aux','$(SRC).bbl','$(SRC).blg','$(SRC).log','$(SRC).out','$(SRC).pdf','$(SRC).fls','$(SRC).fdb_latexmk','$(SRC).synctex.gz','figures/sls_kernel.pdf','figures/pipeline.pdf']; [os.remove(f) for f in files if os.path.exists(f)]"

.PHONY: all clean figures
