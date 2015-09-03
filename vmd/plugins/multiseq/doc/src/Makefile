NAME := mseq_manual

default: pdf

all:	pdf html

clean:
	rm -f `find . -name "*.aux"` $(NAME).out $(NAME).bbl $(NAME).blg $(NAME).lof $(NAME).log $(NAME).lot $(NAME).toc $(NAME).pdf

#	bibtex $(NAME)
pdf:
	pdflatex $(NAME)
	pdflatex $(NAME)
#	pdflatex $(NAME)

html: html/index.html

html/index.html: $(NAME).tex $(NAME).aux
	mkdir -p html
	cp -r icons html/
	cp $(NAME).tex index.tex
	cp $(NAME).aux index.aux
	latex2html index
	rm index.tex index.aux
    
$(NAME).aux:
	latex $(NAME)
