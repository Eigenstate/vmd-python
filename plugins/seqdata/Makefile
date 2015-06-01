
VMFILES = pkgIndex.tcl seqdata.tcl seqdata_aln.tcl \
	seqdata_astral.tcl seqdata_enzyme.tcl seqdata_fasta.tcl \
	seqdata_genbank.tcl seqdata_nex.tcl seqdata_notes.tcl \
	seqdata_pdb.tcl seqdata_pir.tcl seqdata_phy.tcl seqdata_scop.tcl seqdata_swissprot.tcl \
	seqdata_taxonomy.tcl seqdata_temperature.tcl seqdata_vmd.tcl
VMVERSION = 1.1
DIR = $(PLUGINDIR)/noarch/tcl/seqdata$(VMVERSION)

#
# Rules
#
bins:
win32bins:
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying seqdata $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

