.SILENT:

VMFILES = cggui.tcl cgtools.tcl cgnetworking.tcl pkgIndex.tcl \
          protein.cgc water.cgc rbcg-2007.par rbcg-2007.top
VMVERSION = 1.2
DIR = $(PLUGINDIR)/noarch/tcl/cgtools$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying cgtools $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
