.SILENT:

VMFILES = paratool.tcl paratool_atomedit.tcl paratool_aux.tcl \
          paratool_respcharges.tcl paratool_charmmcharges.tcl \
          paratool_intcoor.tcl paratool_lists.tcl \
          paratool_readwrite.tcl paratool_tmcomplex.tcl pkgIndex.tcl \
          paratool_components.tcl paratool_topology.tcl \
	  paratool_parameters.tcl paratool_potscan.tcl paratool_refinement.tcl \
	  paratool_energies.tcl paratool_hessian.tcl
 
VMVERSION = 1.5
DIR = $(PLUGINDIR)/noarch/tcl/paratool$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying paratool $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
