.SILENT:

VMFILES = bdtk.tcl bdtk_gui.tcl bdtk_procs.tcl pkgIndex.tcl
VMVERSION = 1.0
DIR = $(PLUGINDIR)/noarch/tcl/bdtk$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying bdtk $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 
	cp -r database $(DIR)
	
