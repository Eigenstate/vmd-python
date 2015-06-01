.SILENT:

VMFILES = qmtool.tcl qmtool_analysis.tcl qmtool_atomedit.tcl \
          qmtool_aux.tcl qmtool_charges.tcl qmtool_intcoor.tcl \
          qmtool_readwrite.tcl qmtool_setup.tcl pkgIndex.tcl

VMVERSION = 1.3
DIR = $(PLUGINDIR)/noarch/tcl/qmtool$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying qmtool $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
