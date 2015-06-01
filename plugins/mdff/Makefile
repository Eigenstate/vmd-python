.SILENT:

VMFILES = mdff.tcl mdff_check.tcl mdff_map.tcl mdff_sim.tcl mdff_correlation.tcl mdff_setup.tcl pkgIndex.tcl mdff_template.namd mdff_tmp.tcl xmdff_phenix.tcl xmdff_template.namd mdff_gui.tcl

VMVERSION = 0.5
DIR = $(PLUGINDIR)/noarch/tcl/mdff$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying mdff $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
