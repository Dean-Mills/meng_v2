##
##  Department of Electrical, Electronic and Computer Engineering
##  MEng / PhD Research Proposal - Makefile
##  Copyright (C) 2011-2016 University of Pretoria.
##

all: research_proposal.pdf

research_proposal.pdf: *.tex *.cls
	pdflatex -shell-escape research_proposal
	-bibtex   research_proposal
	pdflatex -shell-escape research_proposal
	pdflatex -shell-escape research_proposal

clean:
	rm -f *.aux *.log *.bbl *.blg
	rm -f *converted-to.pdf research_proposal.pdf

## End of File
