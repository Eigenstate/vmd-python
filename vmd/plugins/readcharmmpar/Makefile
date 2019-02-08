.SILENT:

VMFILES = readcharmmpar.tcl par_all27_prot_lipid_na.inp pkgIndex.tcl par_amber2charmm.inp par_all36_lipid.prm toppar_all36_carb_glycopeptide.str toppar_water_ions_namd.str par_all22_prot.prm  par_all36_carb.prm  par_all36_cgenff.prm  par_all36_na.prm  par_all36_prot.prm
VMVERSION = 1.4
DIR = $(PLUGINDIR)/noarch/tcl/readcharmmpar$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying readcharmmpar $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
