.SILENT:

VMFILES = membrane.tcl pkgIndex.tcl \
	popc_box.pdb popc_box.psf \
	popc36_box.pdb popc36_box.psf \
	pope36_box.pdb pope36_box.psf \
	pope_box.pdb pope_box.psf \
	top_all27_prot_lipid.inp

VMVERSION = 1.2
DIR = $(PLUGINDIR)/noarch/tcl/membrane$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying membrane $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
