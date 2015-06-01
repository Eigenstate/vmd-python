.SILENT:

VMFILES = pbcbox.tcl pbcgui.tcl pbcjoin.tcl pbcset.tcl pbctools.tcl \
	pbcunwrap.tcl pbcwrap.tcl pkgIndex.tcl

VMVERSION = 2.8
DIR = $(PLUGINDIR)/noarch/tcl/pbctools$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying pbctools $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
