all: pbctools.pdf index.html

# generate the PDF docs via pdflatex
pbctools.pdf: pbctools.tex
	pdflatex pbctools.tex
	pdflatex pbctools.tex

TEX4HTTEMPFILES=pbctools.4ct pbctools.4tc pbctools.out \
	pbctools.lg pbctools.aux pbctools.dvi pbctools.idv \
	pbctools.tmp pbctools.toc pbctools.xref texput.log

# generate the HTML docs via tex4ht
pbctools.html: pbctools.tex pbctools.cfg
	htlatex pbctools.tex "pbctools"
	-rm $(TEX4HTTEMPFILES)
	ls -l pbctools.html

# generate the VMD plugin page via perl
index.html: pbctools.html pbctools.pl
	perl pbctools.pl pbctools.html > index.html

CLEANFILES= make.log pbctools.log \
	$(TEX4HTTEMPFILES)
clean:
	-rm $(CLEANFILES)
