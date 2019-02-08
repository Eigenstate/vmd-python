.SILENT:

VMFILES = pkgIndex.tcl fftk.tcl fftk_BondAngleOpt.tcl fftk_BuildPar.tcl \
	fftk_ChargeOpt.tcl fftk_ChargeOpt_ESP.tcl fftk_Configuration.tcl \
        fftk_DihOpt.tcl fftk_GenBonded.tcl \
	fftk_GenDihScan.tcl fftk_GenZMatrix.tcl fftk_GeomOpt.tcl \
	fftk_guiInterface.tcl fftk_guiProcs.tcl fftk_SharedFcns.tcl \
	fftk_distort.tcl
 
VMVERSION = 1.1
DIR = $(PLUGINDIR)/noarch/tcl/fftk$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying fftk $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
