.SILENT:

VMFILES = networkView.tcl networkViewGui.tcl communityNetwork.tcl suboptimalPaths.tcl networkSetup.tcl adjacencyMatrix.tcl pkgIndex.tcl

VMVERSION = 1.41
DIR = $(PLUGINDIR)/noarch/tcl/networkview$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying networkview $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

