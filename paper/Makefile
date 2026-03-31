TEX = pdflatex -interaction=nonstopmode
BIB = bibtex
SRC = main

all: figures $(SRC).pdf

figures:
	mkdir -p figures
	python ../scripts/gen_sls_figure.py --output figures/sls_kernel.pdf
	cd ../docs && $(TEX) pipeline.tex && cp pipeline.pdf ../paper/figures/pipeline.pdf

$(SRC).pdf: $(SRC).tex references.bib checklist.tex neurips_2026.sty
	$(TEX) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f $(SRC).{aux,bbl,blg,log,out,pdf,synctex.gz,fls,fdb_latexmk}
	rm -f figures/sls_kernel.pdf figures/pipeline.pdf

.PHONY: all clean figures
