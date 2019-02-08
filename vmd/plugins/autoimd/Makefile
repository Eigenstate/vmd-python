.SILENT:

AUTOIMDFILES = autoimd.tcl autoimd-settings.tcl autoimd-api.tcl autoimd-gui.tcl pkgIndex.tcl autoimd-template.namd doc/ug.pdf namdrun.tcl
AUTOIMDVERSION = 1.7
DIR = $(PLUGINDIR)/noarch/tcl/autoimd$(AUTOIMDVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:


distrib: pdf
	@echo "Copying autoimd $(AUTOIMDVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp -f $(AUTOIMDFILES) $(DIR) 

pdf: doc/ug.pdf

doc/ug.pdf: doc/ug.tex doc/ug_title.tex doc/ug_text.tex doc/ug_macros.tex 
	cd doc; make pdf

html:
	cd doc; make html
