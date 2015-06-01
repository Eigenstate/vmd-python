.SILENT:

VMFILES = trunctraj.tcl pkgIndex.tcl
VMVERSION = 1.5
DIR = $(PLUGINDIR)/noarch/tcl/trunctraj$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying trunctraj $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 
	mkdir -p $(DIR)/toppar
	mkdir -p $(DIR)/toppar/non_charmm
	mkdir -p $(DIR)/toppar/stream
	cp toppar/*.inp toppar/*.prm toppar/*.rtf    $(DIR)/toppar
	cp toppar/non_charmm/*.inp                   $(DIR)/toppar/non_charmm/
	cp toppar/stream/*.str                       $(DIR)/toppar/stream/
	
