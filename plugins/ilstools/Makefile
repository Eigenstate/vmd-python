.SILENT:

VMFILES = ilstools.tcl pkgIndex.tcl oxygen.xyz nitricoxide.xyz \
            carbonmonoxide.xyz carbondioxide.xyz ethene.xyz methane.xyz \
            nitrousoxide.xyz nitrogendioxide.xyz nitrogen.xyz \
            acetylene.xyz hydrogen.xyz
VMVERSION = 1.5

DIR = $(PLUGINDIR)/noarch/tcl/ilstools$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib: 
	@echo "Copying ilstools $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp -rf $(VMFILES) $(DIR) 
