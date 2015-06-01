
TCLFILES = colormap_vmd.tcl \
	export_options.tcl get_input.tcl import_options.tcl multiseq.tcl \
	phylotree_options.tcl pkgIndex.tcl plot.tcl select_contactshell.tcl \
	select_nrset.tcl select_residues.tcl stamp_options.tcl find.tcl

PKGVERSION = 2.1
DIR = $(PLUGINDIR)/noarch/tcl/multiseq$(PKGVERSION)
VERSIONFILES = versions*

#
# Rules
#
bins:
win32bins:
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying multiseq $(PKGVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(TCLFILES) $(DIR) 
	cp $(VERSIONFILES) $(DIR)
	
