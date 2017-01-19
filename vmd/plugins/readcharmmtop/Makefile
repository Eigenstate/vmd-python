.SILENT:

VMFILES = readcharmmtop.tcl pkgIndex.tcl top_all27_prot_lipid_na.inp top_all27_hybrid.inp top_amber2charmm.inp top_all36_lipid.rtf toppar_all36_carb_glycopeptide.str  toppar_water_ions_namd.str top_all22_prot.rtf  top_all36_carb.rtf  top_all36_cgenff.rtf top_all36_na.rtf top_all36_prot.rtf top_all36_hybrid.inp
VMVERSION = 1.2
DIR = $(PLUGINDIR)/noarch/tcl/readcharmmtop$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying readcharmmtop $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
