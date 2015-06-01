
VMFILES = clustalw_preferences.tcl colormap_default.tcl consensus_builder.tcl customize_groups.tcl \
	get_groupname.tcl metric_conservation.tcl metric_element.tcl metric_entropy.tcl metric_import.tcl metric_insertions.tcl metric_mutualinfo.tcl \
	metric_percentid.tcl metric_selection.tcl metric_signatures.tcl metric_similarity.tcl metric_type.tcl \
	pkgIndex.tcl removegaps_options.tcl seqedit.tcl seqedit_preview.tcl \
	seqedit_widget.tcl seqedit_widget_edit.tcl seqedit_widget_select.tcl \
	seqedit_widget_statusbar.tcl taxonomy_options.tcl
BLOSUMFILES = blosum*
VMVERSION = 1.1
DIR = $(PLUGINDIR)/noarch/tcl/seqedit$(VMVERSION)

#
# Rules
#
bins:
win32bins:
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying seqedit $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 
	mkdir -p $(DIR)/blosum 
	cd blosum; cp $(BLOSUMFILES) $(DIR)/blosum 

	
