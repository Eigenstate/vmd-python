.SILENT:

VMFILES = pkgIndex.tcl torsionplot.tcl AMAN_12_AMAN.dx AMAN_14_BGLN.dx BGLN_14_BMAN.dx BGLN_or_AMAN_14_BGLN.dx MAN_13_AMAN.dx NGLB_no_FUC.dx rama-general.dx rama-ileval.dx rama-transpro.dx AMAN_12_BGLN.dx AMAN_16_AMAN.dx BGLN_16_AFUC.dx BMAN_16_AMAN.dx NGLB_FUC.dx rama-cispro.dx rama-gly.dx rama-prepro-noGP.dx

VMVERSION = 1.1
DIR = $(PLUGINDIR)/noarch/tcl/torsionplot$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying torsionplot $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
