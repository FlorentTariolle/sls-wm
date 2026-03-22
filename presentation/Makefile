TEX = pdflatex
SRC = main.tex
PDF = main.pdf

all: $(PDF)

$(PDF): $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

.PHONY: all clean
