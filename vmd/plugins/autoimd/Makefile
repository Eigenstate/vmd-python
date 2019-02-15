.SILENT:

AUTOIMDFILES = autoimd.tcl autoimd-settings.tcl autoimd-api.tcl autoimd-gui.tcl pkgIndex.tcl autoimd-template.namd namdrun.tcl
AUTOIMDVERSION = 1.7
DIR = $(PLUGINDIR)/noarch/tcl/autoimd$(AUTOIMDVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:


distrib:
	@echo "Copying autoimd $(AUTOIMDVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp -f $(AUTOIMDFILES) $(DIR) 

