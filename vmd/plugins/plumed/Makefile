.SILENT:

VMFILES = vmdplumed.tcl pkgIndex.tcl \
	  templates_list_v1.tcl templates_list_v2.tcl templates_list_vmdcv.tcl
VMVERSION = 2.7
DIR = $(PLUGINDIR)/noarch/tcl/plumed$(VMVERSION)


bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:


# The first targets are used by VMD's builds.
distrib: 
	@echo "Copying plumed $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 



# export
# .PHONY:
# autogen: 
#	make -f maintainer/Makefile


