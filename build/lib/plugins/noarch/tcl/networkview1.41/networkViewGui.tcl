############################################################################
#cr
#cr            (C) Copyright 1995-2013 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: networkViewGui.tcl,v 1.7 2014/02/20 20:16:08 kvandivo Exp $
#

package provide networkview 1.41

namespace eval ::NetworkView {

    #namespace import ::NetworkView::*

    variable w ""   ;# handle to window
    #variable networkCount 0    ;# whether base network data is loaded or not
    variable selectionType "atomselection"  ;# type of node selection used: atomselection, community, suboptimalpath
    variable atomSelection "all"  ;# selection string for specifying a set of nodes
    variable communityId ""    ;# community ID for specifying a set of nodes
    variable communityList [list ]  ;# list of all communities
    variable suboptimalPathId ""  ;# suboptimalPath ID for specifying a set of nodes
    variable suboptimalPathList [list ] ;# list of all suboptimal path numbers for a given source and target
    variable actionType "activate"      ;# action to be performed: activate, deactivate, colorid, colorstyle
    variable networkFragment "network"  ;# piece of the network that will be affected: network, nodes, edges
    variable colorId "blue"         ;# global color selection
    variable colorStyle "weight"    ;# style for edge colors
    variable communityColor "blue"  ;# color for community coloring
    variable interfaceColor "red"   ;# color for community coloring
    variable loadValueType "suboptPathCount"  ;# data type to load into the value field
    variable scaleFactor "1.0"  ;# default scaling factor
    variable elementSize 1        ;# generic size for setting global node and edge sizes
    variable nodeRadiusStyle "global"   ;# style for node radii
    variable edgeRadiusStyle "global"   ;# style for edge radii
    #variable currentMolid [molinfo top] ;# molid for currently active structure (in NetworkViewGui)
    variable currentMolid ""            ;# molid for currently active structure (in NetworkViewGui)
    variable currentGuiNetworkId ""     ;# guiNetworkId for currently active network (in NetworkView)

    variable guiNetworkIds              ;# array from (molid,guiNetworkId) to global networkIds
    array set guiNetworkIds {}

}


# Public proc to start up NetworkViewGui
proc networkviewgui {} {

    if {[catch {
        ::NetworkView::printReference
        ::NetworkView::createNetworkViewGui
        ::NetworkView::showNetworkViewGui
    } errorMessage] != 0} {
        global errorInfo
        set callStack [list "NetworkViewGui Error) " $errorInfo]
        puts [join $callStack "\n"]
        tk_messageBox -type ok -icon error -parent $w -title "Error" -message "NetworkViewGui failed to start with the following message:\n\n$errorMessage"
        return
    }

    return $::NetworkView::w
}

namespace eval ::NetworkView {

    # Print the paper reference for protein structure network analysis
    proc printReference {} {
	puts "NetworkView Reference) In any publication of scientific results based completely or"
	puts "NetworkView Reference) in part on the use of NetworkView, please reference:"
	puts "NetworkView Reference) Anurag Sethi, John Eargle, Alexis Black, and Zaida Luthey-"
	puts "NetworkView Reference) Schulten. Dynamical Networks in tRNA:Protein Complexes."
	puts "NetworkView Reference) PNAS. 2009."
	
	return
    }
    
    # Call getOpenFile dialog to load network data.
    proc loadNetwork { } {

	variable networkCount
	variable currentMolid
	variable guiNetworkIds

	set filename [tk_getOpenFile -filetypes {{{Network Data} {.dat}} {{All Files} *}} -title "Load Network"]
	set success 0

	if {$filename != ""} {
      	    set success [networkView $currentMolid $filename]
	    if {$success == 1} {
		set currentMolidNetworkCount [llength [array names guiNetworkIds "$currentMolid,*"]]
		set guiNetworkIds($currentMolid,$currentMolidNetworkCount) [expr $networkCount - 1]
	    }
	}
	
	return
    }

    # Call getOpenFile dialog to load values into a network.
    proc loadValueData { } {

      variable networkCount
      variable currentMolid
      variable guiNetworkIds

      set filename [tk_getOpenFile -filetypes {{{Value Data} {.dat}} {{All Files} *}} -title "Load Value Data"]
      set success 0

      if {$filename != ""} {
         set success [readValueDataFile $filename]
      }
   
      return
    }

    # Call getOpenFile dialog to load community data.
    proc loadCommunity { } {

	set filename [tk_getOpenFile -filetypes {{{Community Data} {.out}} {{All Files} *}} -title "Load Communities"]
	if {$filename != "" && $::NetworkView::networkCount != 0} {
	    readCommunityFile $filename
	}
	
	return
    }
    
    # Call getOpenFile dialog to load suboptimal path data.
    proc loadSuboptimalPath { } {
	
	set filename [tk_getOpenFile -filetypes {{{Suboptimal Path Data} {.out}} {{All Files} *}} -title "Load Suboptimal Paths"]
	if {$filename != "" && $::NetworkView::networkCount != 0} {
	    readSuboptimalPathFile $filename
	}
	
	return
    }
    
    # Set up the NetworkView Tk GUI objects.
    proc createNetworkViewGui { } {
	
	variable w
	variable networkCount
	variable communityLoaded
	
	set w [toplevel ".networkviewgui"]
	wm title $w "NetworkView"
	wm minsize $w 300 300
	
	frame $w.top
	
	createMenu
	
	createSimpleGui

	trace add variable ::NetworkView::suboptimalPathLoaded write "::NetworkView::suboptimalPathDataUpdated"

	# Update GUI if network information is already loaded
	if {[molinfo num] > 0} {
	    moleculeDataUpdated dummy dummy dummy
	}
	if {$networkCount > 0} {
	    networkDataUpdated dummy dummy dummy
	}
	if {[array size communityLoaded] > 0} {
	    communityDataUpdated dummy dummy dummy
	}
	if {[array size suboptimalPathLoaded] > 0} {
	    suboptimalPathDataUpdated dummy dummy dummy
	}
	
	#frame $w.top.pickdisplay
	#pack $w.top.pickdisplay -fill both -expand 1 ;# -padx {0 1} -pady {0 1}

        pack $w.top -fill both -expand 1

	return
    }

    # Create the menu items for the main NetworkViewGui window
    proc createMenu { } {
	
	variable w

	# Create menubar
	frame $w.top.menubar -relief raised -bd 2
	pack $w.top.menubar -padx 1 -fill x -side top

	# File menu
	menubutton $w.top.menubar.file -text "File" -underline 0 -menu $w.top.menubar.file.menu
        $w.top.menubar.file config -width 5
        pack $w.top.menubar.file -side left
        menu $w.top.menubar.file.menu -tearoff no
        $w.top.menubar.file.menu add command -label "Load Network..." \
	    -command {::NetworkView::loadNetwork}
        $w.top.menubar.file.menu add command -label "Load Value Data..." \
       -command {::NetworkView::loadValueData}
        $w.top.menubar.file.menu add command -label "Load Community Data..." \
	    -command {::NetworkView::loadCommunity}
        $w.top.menubar.file.menu add command -label "Load Suboptimal Path Data..." \
	    -command {::NetworkView::loadSuboptimalPath}
        #$w.top.menubar.file.menu add command -label "Save Session..." \
	\#    -command "puts \"Save Session...\""
        $w.top.menubar.file.menu add separator
        $w.top.menubar.file.menu add command -label "Write GDF File..." \
	    -command {::NetworkView::writeGdfFile}
        $w.top.menubar.file.menu add separator
        $w.top.menubar.file.menu add command -label "Quit NetworkView" \
	    -command {menu networkview off}
	return
    }

    # Create the Simple GUI layout
    proc createSimpleGui { } {

	variable w
	variable selectionType
	variable atomSelection
	variable communityList
	variable suboptimalPathId
	variable suboptimalPathList
	variable actionType
	variable networkFragment
	variable colorId
	variable colorStyle
	variable communityColor
	variable interfaceColor
	variable loadValueType
	variable elementSize
	variable nodeRadiusStyle
	variable edgeRadiusStyle
	variable currentNetwork
	variable molids
	variable currentMolid

#	puts ">createSimpleGui"

	setupVmdEventListeners

	set areCommunitiesLoaded [array get ::NetworkView::communityLoaded "$::NetworkView::currentNetwork"]
	if {[llength $areCommunitiesLoaded] == 0} {
	    communityDataUpdated name1 name2 op
	}

	set tempSuboptimalPathIds [list ]
	foreach {suboptKey totalPathCount} [array get ::NetworkView::suboptimalPaths "$::NetworkView::currentNetwork*,totalPathCount"] {
#	    puts "suboptKey($suboptKey) = $totalPathCount"
	    regexp {\d+,(\d+),(\d+),totalPathCount} $suboptKey matchVar source target
	    lappend tempSuboptimalPathIds "$source.$target"
	}
	set tempSuboptimalPathIds [lsort -real $tempSuboptimalPathIds]
	set suboptimalPathIds [list ]
	foreach suboptId $tempSuboptimalPathIds {
	    lappend suboptimalPathIds [regsub {\.} $suboptId ","]
	}
	if {[llength $suboptimalPathIds] > 0} {
	    set suboptimalPathId [lindex $suboptimalPathIds 0]
	}
	if {$suboptimalPathId != ""} {
	    for {set i 0} {$i < $::NetworkView::suboptimalPaths($::NetworkView::currentNetwork,$suboptimalPathId,totalPathCount)} {incr i} {
		lappend suboptimalPathList $i
	    }
	}

	# Set up display parameters
	labelframe $w.top.display -bd 2 -relief ridge -text "Display Parameters"	
	frame $w.top.display.options

	label $w.top.display.options.molid_lbl -text "Molecule ID"
	menubutton $w.top.display.options.molid -relief raised -bd 2 \
	    -textvariable ::NetworkView::currentMolid -direction flush \
	    -menu $w.top.display.options.molid.menu
	menu $w.top.display.options.molid.menu -tearoff no
	foreach molid [molinfo list] {
	    $w.top.display.options.molid.menu add radiobutton \
		-variable ::NetworkView::currentMolid -value $molid \
		-label $molid -command {::NetworkView::currentMolidChosen}
	}

	label $w.top.display.options.networkid_lbl -text "Network ID"
	menubutton $w.top.display.options.networkid -relief raised -bd 2 \
	    -textvariable ::NetworkView::currentGuiNetworkId -direction flush \
	    -menu $w.top.display.options.networkid.menu
	menu $w.top.display.options.networkid.menu -tearoff no
	foreach {key networkId} [array get guiNetworkIds "$currentMolid,*"] {
	    # Get guiNetworkId; networkId
	    regexp {$currentMolid,(\d+)} $key matchVar guiNetworkId
#	    puts "key: $key, currentMolid: $currentMolid, guiNetworkId: $guiNetworkId, networkId: $networkId"
	    $w.top.display.options.networkid.menu add radiobutton \
		-variable ::NetworkView::currentNetwork -value $networkId \
		-label $guiNetworkId -command {::NetworkView::currentNetworkIdChosen}
	}

	label $w.top.display.options.networkfragment_lbl -text "Network Fragment"
	menubutton $w.top.display.options.networkfragment -relief raised -bd 2 \
	    -textvariable ::NetworkView::networkFragment -direction flush \
	    -menu $w.top.display.options.networkfragment.menu
	menu $w.top.display.options.networkfragment.menu -tearoff no
	$w.top.display.options.networkfragment.menu add radiobutton \
	    -variable ::NetworkView::networkFragment -value "network" -label "network"
	$w.top.display.options.networkfragment.menu add radiobutton \
	    -variable ::NetworkView::networkFragment -value "nodes" -label "nodes"
	$w.top.display.options.networkfragment.menu add radiobutton \
	    -variable ::NetworkView::networkFragment -value "edges" -label "edges"

	label $w.top.display.options.nodesize_lbl -text "Node Size"
	menubutton $w.top.display.options.nodesize -relief raised -bd 2 \
	    -textvariable ::NetworkView::nodeRadiusStyle -direction flush \
	    -menu $w.top.display.options.nodesize.menu
	menu $w.top.display.options.nodesize.menu -tearoff no
	$w.top.display.options.nodesize.menu add radiobutton \
	    -variable ::NetworkView::nodeRadiusStyle -value "global" -label "global"
	$w.top.display.options.nodesize.menu add radiobutton \
	    -variable ::NetworkView::nodeRadiusStyle -value "weight" -label "weight"
	#$w.top.display.options.nodesize.menu add radiobutton \
	\#    -variable ::NetworkView::nodeRadiusStyle -value "correlation" -label "correlation"
	#$w.top.display.options.nodesize.menu add radiobutton \
	\#    -variable ::NetworkView::nodeRadiusStyle -value "betweenness" -label "betweenness"
	$w.top.display.options.nodesize.menu add radiobutton \
	    -variable ::NetworkView::nodeRadiusStyle -value "value" -label "value"

	label $w.top.display.options.edgesize_lbl -text "Edge Size"
	menubutton $w.top.display.options.edgesize -relief raised -bd 2 \
	    -textvariable ::NetworkView::edgeRadiusStyle -direction flush \
	    -menu $w.top.display.options.edgesize.menu
	menu $w.top.display.options.edgesize.menu -tearoff no
	$w.top.display.options.edgesize.menu add radiobutton \
	    -variable ::NetworkView::edgeRadiusStyle -value "global" -label "global"
	$w.top.display.options.edgesize.menu add radiobutton \
	    -variable ::NetworkView::edgeRadiusStyle -value "weight" -label "weight"
	#$w.top.display.options.edgesize.menu add radiobutton \
	\#    -variable ::NetworkView::edgeRadiusStyle -value "correlation" -label "correlation"
	#$w.top.display.options.edgesize.menu add radiobutton \
	\#    -variable ::NetworkView::edgeRadiusStyle -value "betweenness" -label "betweenness"
	$w.top.display.options.edgesize.menu add radiobutton \
	    -variable ::NetworkView::edgeRadiusStyle -value "value" -label "value"

	label $w.top.display.options.nodescale_lbl -text "Global Node Scale"
	menubutton $w.top.display.options.nodescale -relief raised -bd 2 \
	    -textvariable ::NetworkView::sphereRadius -direction flush \
	    -menu $w.top.display.options.nodescale.menu
	menu $w.top.display.options.nodescale.menu -tearoff no
	$w.top.display.options.nodescale.menu add radiobutton \
	    -variable ::NetworkView::sphereRadius -value "0" -label "0"
	$w.top.display.options.nodescale.menu add radiobutton \
	    -variable ::NetworkView::sphereRadius -value "0.2" -label "0.2"
	$w.top.display.options.nodescale.menu add radiobutton \
	    -variable ::NetworkView::sphereRadius -value "0.4" -label "0.4"
	$w.top.display.options.nodescale.menu add radiobutton \
	    -variable ::NetworkView::sphereRadius -value "0.6" -label "0.6"
	$w.top.display.options.nodescale.menu add radiobutton \
	    -variable ::NetworkView::sphereRadius -value "0.8" -label "0.8"
	$w.top.display.options.nodescale.menu add radiobutton \
	    -variable ::NetworkView::sphereRadius -value "1" -label "1"

	label $w.top.display.options.edgescale_lbl -text "Global Edge Scale"
	menubutton $w.top.display.options.edgescale -relief raised -bd 2 \
	    -textvariable ::NetworkView::cylinderRadius -direction flush \
	    -menu $w.top.display.options.edgescale.menu
	menu $w.top.display.options.edgescale.menu -tearoff no
	$w.top.display.options.edgescale.menu add radiobutton \
	    -variable ::NetworkView::cylinderRadius -value "0" -label "0"
	$w.top.display.options.edgescale.menu add radiobutton \
	    -variable ::NetworkView::cylinderRadius -value "0.2" -label "0.2"
	$w.top.display.options.edgescale.menu add radiobutton \
	    -variable ::NetworkView::cylinderRadius -value "0.4" -label "0.4"
	$w.top.display.options.edgescale.menu add radiobutton \
	    -variable ::NetworkView::cylinderRadius -value "0.6" -label "0.6"
	$w.top.display.options.edgescale.menu add radiobutton \
	    -variable ::NetworkView::cylinderRadius -value "0.8" -label "0.8"
	$w.top.display.options.edgescale.menu add radiobutton \
	    -variable ::NetworkView::cylinderRadius -value "1" -label "1"

	label $w.top.display.options.noderes_lbl -text "Node Resolution"
	menubutton $w.top.display.options.noderes -relief raised -bd 2 \
	    -textvariable ::NetworkView::sphereResolution -direction flush \
	    -menu $w.top.display.options.noderes.menu
	menu $w.top.display.options.noderes.menu -tearoff no
	$w.top.display.options.noderes.menu add radiobutton \
	    -variable ::NetworkView::sphereResolution -value 4 -label "4"
	$w.top.display.options.noderes.menu add radiobutton \
	    -variable ::NetworkView::sphereResolution -value 8 -label "8"
	$w.top.display.options.noderes.menu add radiobutton \
	    -variable ::NetworkView::sphereResolution -value 12 -label "12"
	$w.top.display.options.noderes.menu add radiobutton \
	    -variable ::NetworkView::sphereResolution -value 16 -label "16"
	$w.top.display.options.noderes.menu add radiobutton \
	    -variable ::NetworkView::sphereResolution -value 20 -label "20"
	
	label $w.top.display.options.edgeres_lbl -text "Edge Resolution"
	menubutton $w.top.display.options.edgeres -relief raised -bd 2 \
	    -textvariable ::NetworkView::cylinderResolution -direction flush \
	    -menu $w.top.display.options.edgeres.menu
	menu $w.top.display.options.edgeres.menu -tearoff no
	$w.top.display.options.edgeres.menu add radiobutton \
	    -variable ::NetworkView::cylinderResolution -value 4 -label "4"
	$w.top.display.options.edgeres.menu add radiobutton \
	    -variable ::NetworkView::cylinderResolution -value 8 -label "8"
	$w.top.display.options.edgeres.menu add radiobutton \
	    -variable ::NetworkView::cylinderResolution -value 12 -label "12"
	$w.top.display.options.edgeres.menu add radiobutton \
	    -variable ::NetworkView::cylinderResolution -value 16 -label "16"
	$w.top.display.options.edgeres.menu add radiobutton \
	    -variable ::NetworkView::cylinderResolution -value 20 -label "20"

	grid $w.top.display.options.molid_lbl -column 1 -row 1 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.molid     -column 2 -row 1 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.networkid_lbl -column 3 -row 1 -sticky w -pady 2 -padx 5
	grid $w.top.display.options.networkid     -column 4 -row 1 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.networkfragment_lbl -column 1 -row 2 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.networkfragment     -column 2 -row 2 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.nodesize_lbl -column 1 -row 3 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.nodesize     -column 2 -row 3 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.edgesize_lbl -column 3 -row 3 -sticky w -pady 2 -padx 5
	grid $w.top.display.options.edgesize     -column 4 -row 3 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.nodescale_lbl -column 1 -row 4 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.nodescale     -column 2 -row 4 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.edgescale_lbl -column 3 -row 4 -sticky w -pady 2 -padx 5
	grid $w.top.display.options.edgescale     -column 4 -row 4 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.noderes_lbl -column 1 -row 5 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.noderes     -column 2 -row 5 -sticky w -pady 2 -padx 2
	grid $w.top.display.options.edgeres_lbl -column 3 -row 5 -sticky w -pady 2 -padx 5
	grid $w.top.display.options.edgeres     -column 4 -row 5 -sticky w -pady 2 -padx 2

	pack $w.top.display.options -padx 3 -anchor e -side left
	pack $w.top.display -pady 5 -padx 3 -ipady 1m -fill x -anchor w -side top

	# Set up node selections
	labelframe $w.top.select -bd 2 -relief ridge -text "Node Selection"

	frame $w.top.select.type

	radiobutton $w.top.select.type.atomselection_rb -text "Atom Selection" \
	    -variable ::NetworkView::selectionType -relief flat -value atomselection \
	    ;#-command "puts \"atomselection\""
	entry $w.top.select.type.atomselection -textvariable ::NetworkView::atomSelection -width 40 -bg white

	radiobutton $w.top.select.type.community_rb -text "Community" \
	    -variable ::NetworkView::selectionType -relief flat -value community \
	    ;#-command "puts \"community\""
	frame $w.top.select.type.communitysel_fr
	listbox $w.top.select.type.communitysel_fr.lb -height 5 \
	    -listvariable ::NetworkView::communityList \
	    -selectmode multiple -yscrollcommand "$w.top.select.type.communitysel_fr.scr set" \
	    -exportselection no -bg white
	scrollbar $w.top.select.type.communitysel_fr.scr -command "$w.top.select.type.communitysel_fr.lb yview"
	button $w.top.select.type.community_all -text "All" -command {::NetworkView::communityBulkSelect all}
	button $w.top.select.type.community_none -text "None" -command {::NetworkView::communityBulkSelect none}

	radiobutton $w.top.select.type.suboptimalpath_rb -text "Suboptimal Path" \
	    -variable ::NetworkView::selectionType -relief flat -value suboptimalpath \
	    ;#-command "puts \"suboptimalpath\""
	menubutton $w.top.select.type.suboptimalpath -relief raised -bd 2 \
	    -textvariable ::NetworkView::suboptimalPathId -direction flush \
	    -menu $w.top.select.type.suboptimalpath.menu
	menu $w.top.select.type.suboptimalpath.menu -tearoff no
	foreach suboptimalPath $suboptimalPathIds {
	    $w.top.select.type.suboptimalpath.menu add radiobutton \
		-variable ::NetworkView::suboptimalPathId -value $suboptimalPath \
		-label $suboptimalPath -command {::NetworkView::suboptimalPathSetChosen}
	}
	frame $w.top.select.type.suboptimalpathsel_fr
	listbox $w.top.select.type.suboptimalpathsel_fr.lb -height 5 \
	    -listvariable ::NetworkView::suboptimalPathList \
	    -selectmode multiple -yscrollcommand "$w.top.select.type.suboptimalpathsel_fr.scr set" \
	    -exportselection no -bg white
	scrollbar $w.top.select.type.suboptimalpathsel_fr.scr -command "$w.top.select.type.suboptimalpathsel_fr.lb yview"
	button $w.top.select.type.suboptimalpath_all -text "All" -command {::NetworkView::suboptimalPathBulkSelect all}
	button $w.top.select.type.suboptimalpath_none -text "None" -command {::NetworkView::suboptimalPathBulkSelect none}

	radiobutton $w.top.select.type.criticalnode_rb -text "Critical Node" \
	    -variable ::NetworkView::selectionType -relief flat -value criticalnode \
	    ;#-command "puts \"criticalnode\""

	grid $w.top.select.type.atomselection_rb -column 1 -row 1 -sticky w
	grid $w.top.select.type.atomselection -column 2 -columnspan 3 -row 1 -sticky w
	grid $w.top.select.type.community_rb -column 1 -row 2 -sticky w
	grid $w.top.select.type.communitysel_fr.lb -column 1 -row 1 -sticky w
	grid $w.top.select.type.communitysel_fr.scr -column 2 -row 1 -sticky nsw
	grid $w.top.select.type.communitysel_fr -column 2 -row 2 -sticky w
	grid $w.top.select.type.community_all -column 3 -row 2 -sticky w
	grid $w.top.select.type.community_none -column 4 -row 2 -sticky w
	grid $w.top.select.type.suboptimalpath_rb -column 1 -row 3 -sticky w
	grid $w.top.select.type.suboptimalpath -column 2 -row 3 -sticky w
	grid $w.top.select.type.suboptimalpathsel_fr.lb -column 1 -row 1 -sticky w
	grid $w.top.select.type.suboptimalpathsel_fr.scr -column 2 -row 1 -sticky nsw
	grid $w.top.select.type.suboptimalpathsel_fr -column 2 -row 4 -sticky w
	grid $w.top.select.type.suboptimalpath_all -column 3 -row 4 -sticky w
	grid $w.top.select.type.suboptimalpath_none -column 4 -row 4 -sticky w
	grid $w.top.select.type.criticalnode_rb -column 1 -row 5 -sticky w
	
	pack $w.top.select.type -padx 3 -anchor e -side left
	pack $w.top.select -side top -pady 5 -padx 3 -ipady 1m -fill x -anchor w	

	# Set up actions
	labelframe $w.top.action -bd 2 -relief ridge -text "Action"
	
	frame $w.top.action.type
		
	radiobutton $w.top.action.type.activate_rb -text Activate -variable ::NetworkView::actionType \
	    -relief flat -value activate ;#-command "puts \"activate\""
	radiobutton $w.top.action.type.deactivate_rb -text Deactivate -variable ::NetworkView::actionType \
	    -relief flat -value deactivate ;#-command "puts \"deactivate\""

	radiobutton $w.top.action.type.colorid_rb -text "Color ID" -variable ::NetworkView::actionType \
	    -relief flat -value colorid ;#-command "puts \"colorID\""
	menubutton $w.top.action.type.colorid -relief raised -bd 2 \
	    -textvariable ::NetworkView::colorId -direction flush \
	    -menu $w.top.action.type.colorid.menu
	menu $w.top.action.type.colorid.menu -tearoff no
	addMenuColors $w.top.action.type.colorid.menu "::NetworkView::colorId"

	radiobutton $w.top.action.type.colorbulk_rb -text "Color Bulk" -variable ::NetworkView::actionType \
	    -relief flat -value colorbulk ;#-command "puts \"colorBulk\""
	menubutton $w.top.action.type.colorstyle -relief raised -bd 2 \
	    -textvariable ::NetworkView::colorStyle -direction flush \
	    -menu $w.top.action.type.colorstyle.menu
	menu $w.top.action.type.colorstyle.menu -tearoff no
	$w.top.action.type.colorstyle.menu add radiobutton \
	    -variable ::NetworkView::colorStyle -value "weight" -label "weight"
	#$w.top.action.type.colorstyle.menu add radiobutton \
	\#    -variable ::NetworkView::colorStyle -value "betweenness" -label "betweenness"
	$w.top.action.type.colorstyle.menu add radiobutton \
	    -variable ::NetworkView::colorStyle -value "value" -label "value"

	radiobutton $w.top.action.type.colorcommunities_rb -text "Color Communities" -variable ::NetworkView::actionType \
	    -relief flat -value colorcommunities ;#-command "puts \"colorCommunities\""
	#label $w.top.action.type.communitycolor_lbl -text "Community"
	#menubutton $w.top.action.type.communitycolor -relief raised -bd 2 \
	\#    -textvariable ::NetworkView::communityColor -direction flush \
	 \#   -menu $w.top.action.type.communitycolor.menu
	#menu $w.top.action.type.communitycolor.menu -tearoff no
	#addMenuColors $w.top.action.type.communitycolor.menu "::NetworkView::communityColor"

	#label $w.top.action.type.interfacecolor_lbl -text "Interface"
	#menubutton $w.top.action.type.interfacecolor -relief raised -bd 2 \
	\#    -textvariable ::NetworkView::interfaceColor -direction flush \
	 \#   -menu $w.top.action.type.interfacecolor.menu
	#menu $w.top.action.type.interfacecolor.menu -tearoff no
	#addMenuColors $w.top.action.type.interfacecolor.menu "::NetworkView::interfaceColor"

	radiobutton $w.top.action.type.loadvalue_rb -text "Load into Value" -variable ::NetworkView::actionType \
	    -relief flat -value loadvalue ;#-command "puts \"loadValue\""
	menubutton $w.top.action.type.loadvalue -relief raised -bd 2 \
	    -textvariable ::NetworkView::loadValueType -direction flush \
	    -menu $w.top.action.type.loadvalue.menu
	menu $w.top.action.type.loadvalue.menu -tearoff no
	$w.top.action.type.loadvalue.menu add radiobutton \
	    -variable ::NetworkView::loadValueType -value "suboptPathCount" -label "suboptimal path count"

	label $w.top.action.type.scale_lbl -text "Scale Factor:"
   entry $w.top.action.type.scaleFactor -width 5 -textvariable [namespace current]::scaleFactor   

	button $w.top.action.type.apply -text "Apply" -command ::NetworkView::executeAction

	grid $w.top.action.type.activate_rb -column 1 -row 1 -sticky w
	grid $w.top.action.type.deactivate_rb -column 1 -row 2 -sticky w
	grid $w.top.action.type.colorid_rb -column 1 -row 3 -sticky w
	grid $w.top.action.type.colorid -column 2 -row 3 -sticky w
	grid $w.top.action.type.colorbulk_rb -column 1 -row 4 -sticky w
	grid $w.top.action.type.colorstyle -column 2 -row 4 -sticky w
	grid $w.top.action.type.colorcommunities_rb -column 1 -row 5 -sticky w
	grid $w.top.action.type.loadvalue_rb -column 1 -row 7 -sticky w
	grid $w.top.action.type.loadvalue -column 2 -row 7 -sticky w
	grid $w.top.action.type.scale_lbl -column 3 -row 7 -sticky e
	grid $w.top.action.type.scaleFactor -column 4 -row 7 -sticky w
	grid $w.top.action.type.apply -column 1 -row 8 -sticky w
	
	pack $w.top.action.type -padx 3 -anchor e -side left
	pack $w.top.action -side top -pady 5 -padx 3 -ipady 1m -fill x -anchor w
	
	# Set up buttons
	#button $w.top.apply -text "Apply" -command ::NetworkView::executeAction
	button $w.top.draw -text "Draw" -command ::NetworkView::executeDraw

	#pack $w.top.apply -side left -anchor w
	pack $w.top.draw -pady 5

#	puts "<createSimpleGui"

	return
    }

    # This method sets up the VMD event listeners.
    proc setupVmdEventListeners { } {
	
        global vmd_frame
        global vmd_initialize_structure
        global vmd_trajectory_read
        global vmd_quit
        global vmd_pick_atom
		
	# Update the GUI when new data is loaded
	trace add variable vmd_initialize_structure write "::NetworkView::moleculeDataUpdated"
	#trace add variable ::NetworkView::networkCount write "::NetworkView::networkDataUpdated"
	trace add variable ::NetworkView::guiNetworkIds write "::NetworkView::networkDataUpdated"
	trace add variable ::NetworkView::communityLoaded write "::NetworkView::communityDataUpdated"
	trace add variable ::NetworkView::suboptimalPathLoaded write "::NetworkView::suboptimalPathDataUpdated"

	# Debugging for listboxes
	#trace add variable ::NetworkView::communityList write "::NetworkView::communityListOutput"
	#trace add variable ::NetworkView::suboptimalPathList write "::NetworkView::suboptimalPathListOutput"
	
	return
    }


    # Add menu colors
    # @param menu
    # @param menuVariable
    proc addMenuColors { menu menuVariable } {

	$menu add radiobutton -variable $menuVariable -value "blue" -label "blue"
 	$menu add radiobutton -variable $menuVariable -value "red" -label "red"
 	$menu add radiobutton -variable $menuVariable -value "gray" -label "gray"
 	$menu add radiobutton -variable $menuVariable -value "orange" -label "orange"
 	$menu add radiobutton -variable $menuVariable -value "yellow" -label "yellow"
 	$menu add radiobutton -variable $menuVariable -value "tan" -label "tan"
 	$menu add radiobutton -variable $menuVariable -value "silver" -label "silver"
 	$menu add radiobutton -variable $menuVariable -value "green" -label "green"
 	$menu add radiobutton -variable $menuVariable -value "white" -label "white"
 	$menu add radiobutton -variable $menuVariable -value "pink" -label "pink"
 	$menu add radiobutton -variable $menuVariable -value "cyan" -label "cyan"
 	$menu add radiobutton -variable $menuVariable -value "purple" -label "purple"
 	$menu add radiobutton -variable $menuVariable -value "lime" -label "lime"
 	$menu add radiobutton -variable $menuVariable -value "mauve" -label "mauve"
 	$menu add radiobutton -variable $menuVariable -value "ochre" -label "ochre"
 	$menu add radiobutton -variable $menuVariable -value "iceblue" -label "iceblue"
 	$menu add radiobutton -variable $menuVariable -value "black" -label "black"

	return
    }


    # Debugging proc for communityList listbox
    # @param name1
    # @param namd2
    # @param op
    proc communityListOutput { name1 name2 op } {
	variable communityList
	puts -nonewline "--communityListOutput: "
	puts $communityList

	return
    }

    # Debugging proc for suboptimalPathList listbox
    # @param name1
    # @param namd2
    # @param op
    proc suboptimalPathListOutput { name1 name2 op } {
	variable suboptimalPathList
	puts -nonewline "--suboptimalPathListOutput: "
	puts $suboptimalPathList

	return
    }

    
    # Make NetworkViewGui visible
    proc showNetworkViewGui { } {

	return
    }

    # Update the GUI when new molecule is loaded
    # @param name1 dummy parameter sent back by Tk
    # @param namd2 dummy parameter sent back by Tk
    # @param op dummy parameter sent back by Tk
    proc moleculeDataUpdated { name1 name2 op } {

	variable currentMolid

	if {$currentMolid == "" &&
	    [llength [molinfo list]] > 0} {
	    set currentMolid [lindex [molinfo list] 0]
	}

	if {$currentMolid != "" &&
	    [llength [molinfo list]] == 0} {
	    set currentMolid ""
	}
	
	rebuildMolidMenu
	
	return
    }

    # Update the GUI when new network is loaded
    # @param name1 dummy parameter sent back by Tk
    # @param namd2 dummy parameter sent back by Tk
    # @param op dummy parameter sent back by Tk
    proc networkDataUpdated { name1 name2 op } {

	variable currentNetwork
	variable currentMolid
	variable currentGuiNetworkId
	variable guiNetworkIds
	
	# XXX - need to make sure that procs using currentNetwork do nothing when it is set to -1
	# if there are no networks loaded for this molecule, set currentNetwork to -1
	#set sortedGuiNetworkIds {}
	foreach {key networkId} [array get guiNetworkIds "$currentMolid,*"] {
	    # Get guiNetworkId; networkId is a global index
	    if {$networkId == $currentNetwork &&
		[regexp {\d+,(\d+)} $key matchVar guiNetworkId]} {
;#		puts "  key: $key; currentMolid: $currentMolid; guiNetworkId: $guiNetworkId; networkId: $networkId"
		set currentGuiNetworkId $guiNetworkId
		break
	    } else {
		puts "  NO MATCH!!!"
		puts "  key: $key; currentMolid: $currentMolid; networkId: $networkId"
	    }
	}	    
	rebuildNetworkIdMenu
	
	return
    }

    # Update the GUI when community data is updated
    # @param name1 dummy parameter sent back by Tk
    # @param namd2 dummy parameter sent back by Tk
    # @param op dummy parameter sent back by Tk
    proc communityDataUpdated { name1 name2 op } {
	
#	puts ">communityDataUpdated"
	rebuildCommunityPicklist

#	puts "<communityDataUpdated"
	return
    }

    #
    proc rebuildCommunityPicklist { } {
	variable w
	variable communityList

	set communityList [list ]
	foreach {communityKey active} [array get ::NetworkView::communityNodes "$::NetworkView::currentNetwork,*,active"] {
	    set communityKey
#	    puts $communityKey
	    regexp {(\d+),active} $communityKey matchVar community
	    lappend communityList [expr $community]
	}
	set communityList [lsort -integer $communityList]

	return
    }

    # Bulk selection for the community picklist
    # @param selection VMD atomselection
    proc communityBulkSelect { selection } {
	
	variable w
	variable communityList

	if {$selection == "all"} {
	    $w.top.select.type.communitysel_fr.lb selection set 0 end
	} elseif {$selection == "none"} {
	    $w.top.select.type.communitysel_fr.lb selection clear 0 end
	} else {
	    puts "Error: communityBulkSelect - invalid argument: $selection"
	}

	return
    }
        
    # Update the GUI when suboptimal path data is updated
    # @param name1 dummy parameter sent back by Tk
    # @param name2 dummy parameter sent back by Tk
    # @param op dummy parameter sent back by Tk
    proc suboptimalPathDataUpdated { name1 name2 op } {
	
#	puts ">suboptimalPathDataUpdated"

	rebuildSuboptimalPathMenu

#	puts "<suboptimalPathDataUpdated"
	return
    }

    # 
    proc rebuildSuboptimalPathMenu { } {

	variable w
	variable suboptimalPathId
	
	
	set tempSuboptimalPathIds [list ]
	foreach {suboptKey totalPathCount} [array get ::NetworkView::suboptimalPaths "$::NetworkView::currentNetwork,*,totalPathCount"] {
#	    puts "suboptKey: $suboptKey"
	    regexp {\d+,(\d+),(\d+),totalPathCount} $suboptKey matchVar source target
	    lappend tempSuboptimalPathIds "$source.$target"
	}
	set tempSuboptimalPathIds [lsort -real $tempSuboptimalPathIds]
	set suboptimalPathIds [list ]
	foreach suboptId $tempSuboptimalPathIds {
	    lappend suboptimalPathIds [regsub {\.} $suboptId ","]
	}
	if {[llength $suboptimalPathIds] > 0} {
	    set suboptimalPathId [lindex $suboptimalPathIds 0]
	} else {
	    set suboptimalPathId ""
	}
	
	$w.top.select.type.suboptimalpath.menu delete 0 end
	foreach suboptimalPath $suboptimalPathIds {
	    $w.top.select.type.suboptimalpath.menu add radiobutton \
		-variable ::NetworkView::suboptimalPathId -value $suboptimalPath \
		-label $suboptimalPath -command {::NetworkView::suboptimalPathSetChosen}
	}

	suboptimalPathSetChosen
	
	return

    }
    
    # Update the GUI when suboptimal path set is chosen
    proc suboptimalPathSetChosen { } {
	
	variable w
	variable suboptimalPathId
	variable suboptimalPathList
	
	set suboptimalPathList [list ]
	
	if {$suboptimalPathId != ""} {
	    for {set i 0} {$i < $::NetworkView::suboptimalPaths($::NetworkView::currentNetwork,$suboptimalPathId,totalPathCount)} {incr i} {
		lappend suboptimalPathList $i
	    }
	}
	
	return
    }

    # Bulk selection for the suboptimalPath picklist
    # @param selection VMD atomselection
    proc suboptimalPathBulkSelect { selection } {
	
	variable w
	#variable suboptimalPathList

	if {$selection == "all"} {
	    $w.top.select.type.suboptimalpathsel_fr.lb selection set 0 end
	} elseif {$selection == "none"} {
	    $w.top.select.type.suboptimalpathsel_fr.lb selection clear 0 end
	} else {
	    puts "Error: suboptimalPathBulkSelect - invalid argument: $selection"
	}

	return
    }
        
    # Write a gdf file (2D network) after getting a filename
    proc writeGdfFile { } {
	
	variable currentNetwork
	variable communityLoaded
	
	set filename [tk_getSaveFile -filetypes {{{Network File} {.gdf}} {{All Files} *}} -title "Save GDF File"]
	set areCommunitiesLoaded [array get ::NetworkView::communityLoaded "$currentNetwork"]
	if {[llength $areCommunitiesLoaded] != 0 && $filename != ""} {
#	    puts "communityLoaded: $communityLoaded($currentNetwork)"
	    writeCommunityGdf $filename
	}

	#if {$communityLoaded($currentNetwork) == 1 && $filename != ""} {
	#}

	return
    }

    # Retrieve selections from a list box
    # @param listbox Listbox
    proc getSelections { listbox } {

	set selections [list ]
	
	for {set i 0} {$i < [$listbox size]} {incr i} {
	    if {[$listbox selection includes $i]} {
		lappend selections $i
	    }
	}
	
	return $selections
    }

    # Update the GUI when currentMolid is chosen
    proc currentMolidChosen { } {
	
	#variable w
	variable currentMolid
	variable currentGuiNetworkId
	variable guiNetworkIds
	
	# XXX - need to make sure that procs using currentNetwork do nothing when it is set to -1
	# if there are no networks loaded for this molecule, set currentNetwork to -1
	if {[llength [array get guiNetworkIds "$currentMolid,*"]] == 0} {
	    #setCurrentNetwork -1
	    set currentGuiNetworkId ""
	} else {
	    set sortedGuiNetworkIds {}
	    foreach {key networkId} [array get guiNetworkIds "$currentMolid,*"] {
		# Get guiNetworkId; networkId is a global index
		if {[regexp {\d+,(\d+)} $key matchVar guiNetworkId]} {
#		    puts "  key: $key; currentMolid: $currentMolid; guiNetworkId: $guiNetworkId; networkId: $networkId"
		    lappend sortedGuiNetworkIds $guiNetworkId
		} else {
		    puts "  NO MATCH!!!"
		    puts "  key: $key; currentMolid: $currentMolid; networkId: $networkId"
		}
	    }	    
	    set currentGuiNetworkId [lindex [lsort -integer $sortedGuiNetworkIds] 0]
	}

	currentNetworkIdChosen
	rebuildNetworkIdMenu

	return
    }

    # Rebuild molid menu
    proc rebuildMolidMenu { } {

	variable w

	$w.top.display.options.molid.menu delete 0 end
	foreach molid [molinfo list] {
	    $w.top.display.options.molid.menu add radiobutton \
		-variable ::NetworkView::currentMolid -value $molid \
		-label $molid -command {::NetworkView::currentMolidChosen}
	}

	return
    }

    # Update the GUI when guiNetworkId is chosen
    proc currentNetworkIdChosen { } {
	
	#variable w
	variable currentMolid
	variable currentGuiNetworkId
	variable guiNetworkIds
	variable suboptimalPathId

	if {[array name guiNetworkIds "$currentMolid,$currentGuiNetworkId"] == ""} {
	    setCurrentNetwork -1
	} else {
	    setCurrentNetwork $guiNetworkIds($currentMolid,$currentGuiNetworkId)
	}
	rebuildCommunityPicklist
	rebuildSuboptimalPathMenu
	
	return
    }

    # Rebuild networkId menu
    proc rebuildNetworkIdMenu { } {

	variable w
	variable currentNetwork
	variable currentMolid
	variable currentGuiNetworkId
	variable guiNetworkIds

#	puts ">rebuildNetworkIdMenu"
	set guiNetworkId -1
	
	$w.top.display.options.networkid.menu delete 0 end
	#puts [array get guiNetworkIds "$currentMolid,*"]
	set idList [list ]
	foreach {key networkId} [array get guiNetworkIds "$currentMolid,*"] {
	    # Get guiNetworkId; networkId is a global index
	    if {[regexp {\d+,(\d+)} $key matchVar guiNetworkId]} {
#		puts "  key: $key; currentMolid: $currentMolid; guiNetworkId: $guiNetworkId; networkId: $networkId"
		lappend idList $networkId
	    } else {
		puts "  NO MATCH!!!"
		puts "  key: $key; currentMolid: $currentMolid; networkId: $networkId"
	    }
	}

	foreach id [lsort -integer $idList] {
	    $w.top.display.options.networkid.menu add radiobutton \
		-variable ::NetworkView::currentGuiNetworkId -value $id \
		-label $id -command {::NetworkView::currentNetworkIdChosen}	    
	}

#	puts "<rebuildNetworkIdMenu"
	return
    }

    # Execute the selected action
    proc executeAction { } {

	variable w
	variable selectionType
	variable atomSelection
	variable suboptimalPathId
	variable actionType
	variable networkFragment
	variable colorId
	variable colorStyle
	variable communityColor
	variable communityList
	variable interfaceColor
	variable loadValueType
	variable elementSize
	#variable edgeRadiusStyle

	set command ""

#	puts ""
#	puts "---PARAMS---"
#	puts "actionType: $actionType"
#	puts "selectionType: $selectionType"

	# First, handle the cases that don't depend on node/edge selections
	if {$actionType == "colorbulk"} {
#	    puts "colorStyle: $colorStyle"
	    colorBy $colorStyle
	#} elseif {$actionType == "setsize"} {
	#    if {$networkFragment == "nodes"} {
	#	puts "setSphereRadius $elementSize"
	#	setSphereRadius $elementSize
	#    } elseif {$networkFragment == "edges"} {
	#	puts "setCylinderRadius $elementSize"
	#	setCylinderRadius $elementSize
	#    } else {
	#	puts "setSphereRadius $elementSize"
	#	puts "setCylinderRadius $elementSize"
	#	setSphereRadius $elementSize
	#	setCylinderRadius $elementSize
	#    }	    
	} elseif {$actionType == "colorcommunities"} {
	    if {$selectionType == "community"} {
		executeCommunity
	    } else {
#		puts "colorCommunities $communityColor $interfaceColor"
		#colorCommunities $communityColor $interfaceColor
		foreach community $communityList {
		    colorCommunity $community $community
		    #puts "community: $community"
		}
	    }
	} elseif {$selectionType == "atomselection"} {
	    executeAtomselection
	} elseif {$selectionType == "community"} {
	    executeCommunity
	} elseif {$selectionType == "suboptimalpath"} {
	    executeSuboptimalPath
	} elseif {$selectionType == "criticalnode"} {
	    executeCriticalNode
	}

	# Debugging proc
	printCurrentParams
	
	return
    }

    
    # Perform action on an atomselection
    proc executeAtomselection { } {

	variable atomSelection
	variable actionType
	variable networkFragment
	variable colorId
	variable colorStyle
	#variable w
	#variable selectionType
	#variable suboptimalPathId
	#variable loadValueType
	#variable elementSize
	#variable edgeRadiusStyle	

	if {$actionType == "activate"} {
	    if {$networkFragment == "nodes"} {
#		puts "activateNodeSelection $atomSelection"
		activateNodeSelection $atomSelection
	    } elseif {$networkFragment == "edges"} {
#		puts "activateEdgeSelection $atomSelection"
		activateEdgeSelection $atomSelection
	    } else {
#		puts "activateNodeSelection $atomSelection"
#		puts "activateEdgeSelection $atomSelection"		
		activateNodeSelection $atomSelection
		activateEdgeSelection $atomSelection
	    }
	} elseif {$actionType == "deactivate"} {
	    if {$networkFragment == "nodes"} {
#		puts "deactivateNodeSelection $atomSelection"
	        deactivateNodeSelection $atomSelection
	    } elseif {$networkFragment == "edges"} {
#		puts "deactivateEdgeSelection $atomSelection"
		deactivateEdgeSelection $atomSelection
	    } else {
#		puts "deactivateNodeSelection $atomSelection"
#		puts "deactivateEdgeSelection $atomSelection"		
		deactivateNodeSelection $atomSelection
		deactivateEdgeSelection $atomSelection
	    }
	} elseif {$actionType == "colorid"} {
	    if {$networkFragment == "nodes"} {
#		puts "colorNodeSelection $atomSelection $colorId"
		colorNodeSelection $atomSelection $colorId
	    } elseif {$networkFragment == "edges"} {
#		puts "colorEdgeSelection $atomSelection $colorId"
		colorEdgeSelection $atomSelection $colorId
	    } else {
#		puts "colorNodeSelection $atomSelection $colorId"
#		puts "colorEdgeSelection $atomSelection $colorId"
		colorNodeSelection $atomSelection $colorId
		colorEdgeSelection $atomSelection $colorId
	    }
	}

	return
    }


    # Perform action on a community selection
    proc executeCommunity { } {

	variable w
	variable actionType
	variable colorId
	variable colorStyle
	variable communityColor
	variable interfaceColor

	if {$actionType == "activate"} {
	    foreach community [getSelections $w.top.select.type.communitysel_fr.lb] {
#		puts "activateCommunity $community"
		activateCommunity $community
	    }
	} elseif {$actionType == "deactivate"} {
	    foreach community [getSelections $w.top.select.type.communitysel_fr.lb] {
#		puts "deactivateCommunity $community"
		deactivateCommunity $community
	    }
	} elseif {$actionType == "colorid"} {
	    foreach community [getSelections $w.top.select.type.communitysel_fr.lb] {
#		puts "colorCommunity $community $colorId"
		colorCommunity $community $colorId
	    }
	} elseif {$actionType == "colorcommunities"} {
	    #puts "colorCommunities $communityColor $interfaceColor [getSelections $w.top.select.type.communitysel_fr.lb]"
	    #colorCommunities $communityColor $interfaceColor [getSelections $w.top.select.type.communitysel_fr.lb]
	    foreach community [getSelections $w.top.select.type.communitysel_fr.lb] {
		colorCommunity $community $community
		#puts "community: $community"
	    }
	}

	return
    }


    # Perform action on a suboptimal path selection
    proc executeSuboptimalPath { } {

	variable w
	variable suboptimalPathId
	variable actionType
	variable colorId
	variable colorStyle
	variable loadValueType
	variable scaleFactor

	if {$actionType == "activate"} {
	    regexp {(\d+),(\d+)} $suboptimalPathId matchVar source target
	    foreach path [getSelections $w.top.select.type.suboptimalpathsel_fr.lb] {
#		puts "activateSuboptimalPath $source $target $path"
		activateSuboptimalPath $source $target $path
	    }
	} elseif {$actionType == "deactivate"} {
	    regexp {(\d+),(\d+)} $suboptimalPathId matchVar source target
	    foreach path [getSelections $w.top.select.type.suboptimalpathsel_fr.lb] {
#		puts "deactivateSuboptimalPath $source $target $path"
		deactivateSuboptimalPath $source $target $path
	    }
	} elseif {$actionType == "colorid"} {
	    regexp {(\d+),(\d+)} $suboptimalPathId matchVar source target
	    foreach path [getSelections $w.top.select.type.suboptimalpathsel_fr.lb] {
#		puts "colorSuboptimalPath $source $target $path $colorId"
		colorSuboptimalPath $source $target $path $colorId
	    }
	} elseif {$actionType == "loadvalue"} {
	    regexp {(\d+),(\d+)} $suboptimalPathId matchVar source target
	    puts "countSuboptimalPathsPerEdge $scaleFactor"
#	    puts "countSuboptimalPathsPerEdge $source $target"
	    countSuboptimalPathsPerEdge $source $target $scaleFactor
	}

	return
    }


    # Perform action on a suboptimal path selection
    proc executeCriticalNode { } {

	variable actionType
	variable colorId
	variable colorStyle

	if {$actionType == "activate"} {
#	    puts "activateCriticalNodes"
	    activateCriticalNodes
	} elseif {$actionType == "deactivate"} {
#	    puts "deactivateCriticalNodes"
	    deactivateCriticalNodes
	} elseif {$actionType == "colorid"} {
#	    puts "colorCriticalNodes $colorId"
	    colorCriticalNodes $colorId
	}

	return
    }


    # Draw the network, nodes, or edges
    proc executeDraw { } {
	
	variable networkFragment
	variable nodeRadiusStyle
	variable edgeRadiusStyle
	
	if {$networkFragment == "nodes"} {
	    set tme1 [clock seconds]
	    drawNodes
	    set tme2 [clock seconds]
	    set TME [expr $tme2 - $tme1]
#	    puts "DBG executeDraw drawNodes took $TME seconds"
	} elseif {$networkFragment == "edges"} {
	    set tme1 [clock seconds]
	    drawEdges $edgeRadiusStyle
	    set tme2 [clock seconds]
	    set TME [expr $tme2 - $tme1]
#	    puts "DBG executeDraw drawEdges took $TME seconds"
	} else {
	    set tme1 [clock seconds]
	    drawNetwork $edgeRadiusStyle
	    set tme2 [clock seconds]
	    set TME [expr $tme2 - $tme1]
#	    puts "DBG executeDraw drawNetwork took $TME seconds"
	}

	return
    }


    # Print out the relevant information for executing the selected action
    proc printCurrentParams { } {
	
	variable selectionType
	variable atomSelection
	variable communityId
	variable suboptimalPathId
	variable actionType
	variable networkFragment
	variable colorId
	variable colorStyle
	variable communityColor
	variable interfaceColor
	variable loadValueType
	variable elementSize
	variable nodeRadiusStyle
	variable edgeRadiusStyle
	variable scaleFactor

	puts ""
	puts "---PARAMS---"
	puts "selectionType: $selectionType"
	if {$selectionType == "atomselection"} {
	    puts "  atomSelection: $atomSelection"
	} elseif {$selectionType == "community"} {
	    puts "  communityId: $communityId"
	} elseif {$selectionType == "suboptimalpath"} {
	    puts "  suboptimalPathId: $suboptimalPathId"
	    #puts "  suboptimalPaths: [getSelectedPaths]"
	}

	puts "actionType: $actionType"
	if {$actionType == "activate"} {
	    puts "  activate"
	} elseif {$actionType == "deactivate"} {
	    puts "  deactivate"
	} elseif {$actionType == "colorid"} {
	    puts "  color ID: $colorId"
	} elseif {$actionType == "colorbulk"} {
	    puts "  colorStyle: $colorStyle"
	} elseif {$actionType == "colorcommunity"} {
	    puts "  communityColor: $communityColor"
	    puts "  interfaceColor: $interfaceColor"
	} elseif {$actionType == "loadvalue"} {
	    puts "  load value type: $loadValueType"
	    puts "  load value scaleFactor: $scaleFactor"
	} elseif {$actionType == "setsize"} {
	    puts "  set size: $elementSize"
	}

	puts "networkFragment: $networkFragment"

	if {$networkFragment == "nodes"} {
	    puts "nodeRadiusStyle: $nodeRadiusStyle"
	} elseif {$networkFragment == "edges"} {
	    puts "edgeRadiusStyle: $edgeRadiusStyle"
	} else {
	    puts "nodeRadiusStyle: $nodeRadiusStyle"
	    puts "edgeRadiusStyle: $edgeRadiusStyle"
	}
	
	return
    }

}
