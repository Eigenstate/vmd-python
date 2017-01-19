.SILENT:

VMFILES = pkgIndex.tcl centerMolecule.tcl vmdlite_analysis.tcl \
	vmdlite_graphics.tcl vmdlite_guiProcs.tcl vmdlite_main.tcl \
	vmdlite_sandbox.tcl vmdlite_system.tcl vmdlite.tcl vmd.rc

VMVERSION = 1.1
DIR = $(PLUGINDIR)/noarch/tcl/vmdlite$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying vmdlite $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 
	cp -r sandboxMolecules $(DIR)
	
