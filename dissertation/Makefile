##
##  Department of Electrical, Electronic and Computer Engineering
##  MEng Dissertation / PhD Thesis - Makefile.
##  Copyright (C) 2011-2017 University of Pretoria.
##

all: diss_thesis.pdf

diss_thesis.pdf: *.tex *.cls
	pdflatex -shell-escape diss_thesis
	bibtex   diss_thesis
	pdflatex -shell-escape diss_thesis
	pdflatex -shell-escape diss_thesis

clean:
	rm -f *.aux *.log *.bbl *.blg *.toc
	rm -f *converted-to.pdf diss_thesis.pdf

## End of File.
