.SILENT:

VMFILES = inorganicbuilder.tcl pkgIndex.tcl
VMVERSION = 0.1
DIR = $(PLUGINDIR)/noarch/tcl/inorganicbuilder$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying inorganicbuilder $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 
	cp -r materials $(DIR)
	
