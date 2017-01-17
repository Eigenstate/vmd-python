.SILENT:

VMFILES = CHANGES.txt COPYRIGHT.txt README.txt \
	pkgIndex.tcl tablelistPublic.tcl tablelist.tcl tablelist_tile.tcl

VMFILES2 = scripts/mwutil.tcl scripts/pencil.cur scripts/repair.tcl \
	scripts/tablelistBind.tcl scripts/tablelistConfig.tcl \
	scripts/tablelistEdit.tcl scripts/tablelistImages.tcl \
	scripts/tablelistMove.tcl scripts/tablelistSort.tcl \
	scripts/tablelistThemes.tcl scripts/tablelistUtil.tcl \
	scripts/tablelistWidget.tcl scripts/tclIndex

VMVERSION = 5.13
DIR = $(PLUGINDIR)/noarch/tcl/tablelist$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying tablelist $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR)/scripts
	cp $(VMFILES) $(DIR) 
	cp $(VMFILES2) $(DIR)/scripts

	
