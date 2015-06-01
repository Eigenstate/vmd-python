.SILENT:

VMFILES = molefacture.tcl molefacture_builder.tcl \
	molefacture_edit.tcl molefacture_geometry.tcl molefacture_gui.tcl \
	molefacture_internals.tcl molefacture_state.tcl pkgIndex.tcl
VMVERSION = 1.3
DIR = $(PLUGINDIR)/noarch/tcl/molefacture$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying molefacture $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 
	cp -r lib $(DIR)
