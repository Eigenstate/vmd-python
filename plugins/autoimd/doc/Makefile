TEXFILES = ug.tex ug_title.tex ug_text.tex ug_macros.tex
#PICS = autoimd-GUI autoimd-diagram simsettings-GUI
PDFPICS = autoimd-GUI.pdf autoimd-diagram.pdf simsettings-GUI.pdf
PSPICS = autoimd-GUI.eps autoimd-diagram.eps simsettings-GUI.eps


ps: $(TEXFILES) $(PSPICS)
	latex ug
	latex ug
	dvips ug -o
        
pdf: clean $(TEXFILES) $(PDFPICS) 
	pdflatex ug
	pdflatex ug

html: ps
	-rm -rf ug
	latex2html -split 1 -toc_depth 2 -notop_navigation -nofootnode -noinfo\
		-t "AutoIMD User's Guide" \
		-address "vmd@ks.uiuc.edu" \
		-up_url "http://www.ks.uiuc.edu/Research/vmd/" \
		-up_title "Main VMD Page" \
	        -image_type gif \
		ug
	tar -cvf - ug | gzip > autoimd.html.tar.gz

clean:
	rm -f ug.aux ug_macros.aux ug_text.aux ug_title.aux

.SUFFIXES : .eps .pdf .gif

.gif.pdf: 
	convert $< $@
        
.gif.eps:
	convert $< $@

