#
# $Id: fftk_guiProcs.tcl,v 1.41 2016/05/31 21:21:23 mayne Exp $
#

#======================================================
#   ffTK GUI PROCS
#======================================================

#------------------------------------------------------
# GENERAL
#------------------------------------------------------
proc ::ForceFieldToolKit::gui::init {} {

    # BuildPar Tab Setup
    # Initialize BuildPar namespace
    ::ForceFieldToolKit::BuildPar::init
    # Initialize BuildPar GUI settings
    set ::ForceFieldToolKit::gui::bparIdMissingAnalyzeMolID -1
    set ::ForceFieldToolKit::gui::bparVDWInputParFile {}
    set ::ForceFieldToolKit::gui::bparVDWele {}
    set ::ForceFieldToolKit::gui::bparVDWparSet {}
    set ::ForceFieldToolKit::gui::bparVDWtvNodeIDs {}
    set ::ForceFieldToolKit::gui::bparVDWrefComment {}
    set ::ForceFieldToolKit::gui::bparCGenFFMolID -1
    array set ::ForceFieldToolKit::gui::bparCGenFFTvSort {bonds -1 angles -1 dihedrals -1 impropers -1}

    # GeomOpt Tab Setup
    # Initialize GeomOpt namespace
    ::ForceFieldToolKit::GeomOpt::init
    # Initialize GeomOpt GUI settings

    # GenZMatrix Tab Setup
    # Initialize GenZMatrix namespace
    ::ForceFieldToolKit::GenZMatrix::init
    # Initialize GenZMatrix GUI settings
    set ::ForceFieldToolKit::gui::gzmAtomLabels {}
    set ::ForceFieldToolKit::gui::gzmVizSpheresDon {}
    set ::ForceFieldToolKit::gui::gzmVizSpheresAcc {}
    set ::ForceFieldToolKit::gui::gzmVizSpheresBoth {}
    set ::ForceFieldToolKit::gui::gzmCOMfiles {}
    set ::ForceFieldToolKit::gui::gzmLOGfiles {}

    # ChargeOpt Tab Setup
    # Initialize ChargeOpt namespace
    ::ForceFieldToolKit::ChargeOpt::init
    # Initialize ChargeOpt GUI Settings
    set ::ForceFieldToolKit::gui::coptAtomLabel "None"
    set ::ForceFieldToolKit::gui::coptAtomLabelInd {}
    set ::ForceFieldToolKit::gui::coptPSFNewPath ""
    set ::ForceFieldToolKit::gui::coptNetCharge 0
    set ::ForceFieldToolKit::gui::coptOvrCharge ""
    set ::ForceFieldToolKit::gui::coptPsfCharge ""
    set ::ForceFieldToolKit::gui::coptPrevLogFile ""
    set ::ForceFieldToolKit::gui::coptBuildScript 0
    set ::ForceFieldToolKit::gui::coptStatus "IDLE"
    set ::ForceFieldToolKit::gui::coptFinalChargeTotal {}
    # Clear Edit Data Boxes
    ::ForceFieldToolKit::gui::coptClearEditData "cconstr"
    ::ForceFieldToolKit::gui::coptClearEditData "wie"
    ::ForceFieldToolKit::gui::coptClearEditData "results"

    # GenBonded Tab Setup
    # Initialize GenBonded Namespace
    ::ForceFieldToolKit::GenBonded::init
    # Initialize GenBonded GUI Settings

    # BondAngleOpt Tab Setup
    # Initialize BondAngleOpt Namespace
    ::ForceFieldToolKit::BondAngleOpt::init
    # Initialize BondAngleOpt GUI Settings
    set ::ForceFieldToolKit::gui::baoptStatus "IDLE"
    set ::ForceFieldToolKit::gui::baoptBuildScript 0
    # Clear Edit Data Boxes
    set ::ForceFieldToolKit::gui::baoptEditBA {}
    set ::ForceFieldToolKit::gui::baoptEditDef {}
    set ::ForceFieldToolKit::gui::baoptEditFC {}
    set ::ForceFieldToolKit::gui::baoptEditEq {}
    set ::ForceFieldToolKit::gui::baoptReturnObjCurrent "---"
    set ::ForceFieldToolKit::gui::baoptReturnObjPrevious "---"

    # GenDihScan Tab Setup
    # Initialize GenDihScan Namespace
    ::ForceFieldToolKit::GenDihScan::init
    # Initialize GUI Settings
    set ::ForceFieldToolKit::gui::gdsAtomLabels {}
    set ::ForceFieldToolKit::gui::gdsRepName {}
    # Clear Edit Data Boxes
    set ::ForceFieldToolKit::gui::gdsEditIndDef {}
    set ::ForceFieldToolKit::gui::gdsEditPlusMinus {}
    set ::ForceFieldToolKit::gui::gdsEditStepSize {}


    # DihOpt Tabl Setup
    # Initialize DihOpt Namespace
    ::ForceFieldToolKit::DihOpt::init
    # Initialize DihOpt GUI Settings
    set ::ForceFieldToolKit::gui::doptStatus "IDLE"
    set ::ForceFieldToolKit::gui::doptBuildScript 0
    set ::ForceFieldToolKit::gui::doptPlotQME 1
    set ::ForceFieldToolKit::gui::doptPlotMME 1
    # Clear Edit Data Boxes
    set ::ForceFieldToolKit::gui::doptEditDef {}
    set ::ForceFieldToolKit::gui::doptEditFC {}
    set ::ForceFieldToolKit::gui::doptEditMult {}
    set ::ForceFieldToolKit::gui::doptEditDelta {}
    set ::ForceFieldToolKit::gui::doptEditLock {}
    set ::ForceFieldToolKit::gui::doptRefineEditDef {}
    set ::ForceFieldToolKit::gui::doptRefineEditFC {}
    set ::ForceFieldToolKit::gui::doptRefineEditMult {}
    set ::ForceFieldToolKit::gui::doptRefineEditDelta {}
    set ::ForceFieldToolKit::gui::doptRefineEditLock {}
    set ::ForceFieldToolKit::gui::doptQMEStatus "EMPTY"
    set ::ForceFieldToolKit::gui::doptMMEStatus "EMPTY"
    set ::ForceFieldToolKit::gui::doptDihAllStatus "EMPTY"
    set ::ForceFieldToolKit::gui::doptEditColor {}
    set ::ForceFieldToolKit::gui::doptResultsPlotHandle {}
    set ::ForceFieldToolKit::gui::doptResultsPlotWin {}
    set ::ForceFieldToolKit::gui::doptResultsPlotCount {}
    set ::ForceFieldToolKit::gui::doptRefineStatus "IDLE"
    set ::ForceFieldToolKit::gui::doptRefineCount 0


    # INITIALIZE THE CONSOLE
    set ::ForceFieldToolKit::gui::consoleMessageCount 0
    set ::ForceFieldToolKit::gui::consoleState 1
    set ::ForceFieldToolKit::gui::consoleMaxHistory 100
}
#======================================================
proc ::ForceFieldToolKit::gui::resizeToActiveTab {} {
    # change the window size to match the active notebook tab

    # need to force gridder to update
    update idletasks

    # uncomment line below to resize width as well
    #set dimW [winfo reqwidth [.fftk_gui.hlf.nb select]]
    # line below does not resize width, as all tabs are designed with gracefull extension of width
    # note +/- for offset can be +- (multimonitor setup), so the expression needs to allow for BOTH symbols;
    # hend "[+-]+"
    regexp {([0-9]+)x[0-9]+[\+\-]+[0-9]+[\+\-]+[0-9]+} [wm geometry .fftk_gui] all dimW
    # manually set dimw to 750
    #set dimW 700
    set dimH [winfo reqheight [.fftk_gui.hlf.nb select]]
    #puts "${dimW}x${dimH}"
    #set dimW [expr {$dimW + 44}]
    if { $::ForceFieldToolKit::gui::consoleState } {
        set dimH [expr {$dimH + 190}]
    } else {
        set dimH [expr {$dimH + 135}]
    }
    wm geometry .fftk_gui [format "%ix%i" $dimW $dimH]
    # note: 44 and 47 take care of additional padding between nb tab and window edges

    update idletasks

}
#======================================================
proc ::ForceFieldToolKit::gui::consoleMessage { desc } {
    # send a message to the console

    # only send messages to console if it's turned on
    if { $::ForceFieldToolKit::gui::consoleState } {
        # lookup and format some data
        set count [format "%03d" $::ForceFieldToolKit::gui::consoleMessageCount]
        set timestamp [clock format [clock seconds] -format {%m/%d/%Y -- %I:%M:%S %p}]

        # write the message to the console
        .fftk_gui.hlf.console.log insert {} 0 -values [list $count $desc $timestamp]

        # increment the count
        incr ::ForceFieldToolKit::gui::consoleMessageCount

        # if number of messages exceeds max, remove last node
        # this is important to prevent taking too much memory
        set itemList [.fftk_gui.hlf.console.log children {}]
        if { [llength $itemList] > $::ForceFieldToolKit::gui::consoleMaxHistory } {
            .fftk_gui.hlf.console.log delete [lindex $itemList end]
        }
    }
}
#======================================================

#------------------------------------------------------
# BuildPar Specific
#------------------------------------------------------
proc ::ForceFieldToolKit::gui::bparAnalyzeMissingPars {} {
    # crosschecks molecule parameters against associated parameters
    # and fills the tv box with undefined parameters

    # localize some variables
    variable bparIdMissingAnalyzeMolID
    set RefParList $::ForceFieldToolKit::BuildPar::idMissingRefParList

    # run a sanity check
    if { ![::ForceFieldToolKit::BuildPar::sanityCheck idMissingAnalyze ] } { return }

    # Build the missing parameters list
    # read in the type definitions for reference parameter set
    set refPars [::ForceFieldToolKit::BuildPar::getRefPars $RefParList]

    # read in the type definitions for the molecule parameter set
    set molecPars [::ForceFieldToolKit::BuildPar::getMolecPars $::ForceFieldToolKit::BuildPar::idMissingPSF $::ForceFieldToolKit::BuildPar::idMissingPDB]

    # search the reference parameters for molecule parameters
    set missingPars [::ForceFieldToolKit::BuildPar::crossCheckPars $molecPars $refPars]
    # note: returns { bonds angles dihedrals vdws }

    # store some GUI-related information
    update idletasks
    set bparIdMissingAnalyzeMolID [molinfo top]
    array unset typeArray; array set typeArray {}

    # process bonds
    foreach bondEntry [topo getbondlist] {
        lassign $bondEntry bInd1 bInd2
        set sel [atomselect top "index $bInd1"]
        set type1 [$sel get type]
        $sel delete
        set sel [atomselect top "index $bInd2"]
        set type2 [$sel get type]
        $sel delete

        if { [info exists typeArray($type1,$type2)] } {
            lappend typeArray($type1,$type2) [list $bInd1 $bInd2]
        } elseif { [info exists typeArray($type2,$type1)] } {
            lappend typeArray($type2,$type1) [list $bInd1 $bInd2]
        } else {
            lappend typeArray($type1,$type2) [list $bInd1 $bInd2]
        }
    }

    # process angles
    foreach angleEntry [topo getanglelist] {
        lassign $angleEntry trash aInd1 aInd2 aInd3
        set sel [atomselect top "index $aInd1"]
        set type1 [$sel get type]
        $sel delete
        set sel [atomselect top "index $aInd2"]
        set type2 [$sel get type]
        $sel delete
        set sel [atomselect top "index $aInd3"]
        set type3 [$sel get type]
        $sel delete

        if { [info exists typeArray($type1,$type2,$type3)] } {
            lappend typeArray($type1,$type2,$type3) [list $aInd1 $aInd2 $aInd3]
        } elseif { [info exists typeArray($type3,$type2,$type1)] } {
            lappend typeArray($type3,$type2,$type1) [list $aInd1 $aInd2 $aInd3]
        } else {
            lappend typeArray($type1,$type2,$type3) [list $aInd1 $aInd2 $aInd3]
        }
    }

    # process dihedrals
    foreach dihEntry [topo getdihedrallist] {
        lassign $dihEntry trash dInd1 dInd2 dInd3 dInd4
        set sel [atomselect top "index $dInd1"]
        set type1 [$sel get type]
        $sel delete
        set sel [atomselect top "index $dInd2"]
        set type2 [$sel get type]
        $sel delete
        set sel [atomselect top "index $dInd3"]
        set type3 [$sel get type]
        $sel delete
        set sel [atomselect top "index $dInd4"]
        set type4 [$sel get type]
        $sel delete

        if { [info exists typeArray($type1,$type2,$type3,$type4)] } {
            lappend typeArray($type1,$type2,$type3,$type4) [list $dInd1 $dInd2 $dInd3 $dInd4]
        } elseif { [info exists typeArray($type4,$type3,$type2,$type1)] } {
            lappend typeArray($type4,$type3,$type2,$type1) [list $dInd1 $dInd2 $dInd3 $dInd4]
        } else {
            lappend typeArray($type1,$type2,$type3,$type4) [list $dInd1 $dInd2 $dInd3 $dInd4]
        }
    }

    # process nonbonded
    set sel [atomselect top all]
    set nonbList [$sel get type]
    $sel delete
    for {set i 0} {$i < [molinfo top get numatoms]} {incr i} {
        lappend typeArray([lindex $nonbList $i]) $i
    }
    unset nonbList

    # Fill the tv box
    # format the missing parameters and insert into the gui tv
    # bonds
    .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.bondsTv delete [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.bondsTv children {}]
    foreach bondDef [lindex $missingPars 0] {
        lassign $bondDef b1 b2
        if { [info exists typeArray($b1,$b2)] } {
            set indsList $typeArray($b1,$b2)
        } elseif { [info exists typeArray($b2,$b1)] } {
            set indsList $typeArray($b2,$b1)
        } else {
            set indsList {}
        }
        .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.bondsTv insert {} end -value [list $b1 $b2 1 $indsList]
    }

    # angles
    .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.anglesTv delete [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.anglesTv children {}]
    foreach angleDef [lindex $missingPars 1] {
        lassign $angleDef a1 a2 a3
        if { [info exists typeArray($a1,$a2,$a3)] } {
            set indsList $typeArray($a1,$a2,$a3)
        } elseif { [info exists typeArray($a3,$a2,$a1)] } {
            set indsList typeArray($a3,$a2,$a1)
        } else {
            set indsList {}
        }
        .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.anglesTv insert {} end -value [list $a1 $a2 $a3 1 $indsList]
    }

    # dihedrals
    .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.dihedralsTv delete [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.dihedralsTv children {}]
    foreach dihDef [lindex $missingPars 2] {
        lassign $dihDef d1 d2 d3 d4
        if { [info exists typeArray($d1,$d2,$d3,$d4)] } {
            set indsList $typeArray($d1,$d2,$d3,$d4)
        } elseif { [info exists typeArray($d4,$d3,$d2,$d1)] } {
            set indsList $typeArray($d4,$d3,$d2,$d1)
        } else {
            set indsList {}
        }
        .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.dihedralsTv insert {} end -value [list $d1 $d2 $d3 $d4 1 $indsList]
    }

    # nonbonded (not set here)
    .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.nonbondedTv delete [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.nonbondedTv children {}]
    foreach vdwDef [lindex $missingPars 3] {
        .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.nonbondedTv insert {} end -value [list $vdwDef 1 $typeArray($vdwDef)]
    }

    ::ForceFieldToolKit::gui::consoleMessage "Missing parameter analysis complete"
}
#======================================================
proc ::ForceFieldToolKit::gui::bparToggleStateMissingPars {tvtype} {
    # toggles the active/inactive state of selections
    # bound to double-click button-1

    # can handle state change for multi-item selections
    set itemIDlist [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.${tvtype} selection]

    foreach itemID $itemIDlist {
        set state [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.${tvtype} set $itemID active]
        if { $state == 1 } {
            # turn off/inactive
            .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.${tvtype} set $itemID active 0
            .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.${tvtype} item $itemID -tags inactive
        } elseif { $state == 0 } {
            # turn on/active
            .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.${tvtype} set $itemID active 1
            .fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.${tvtype} item $itemID -tags {}
        } else {
            # protects from overclicking and getting caught in a state change
            continue
        }
    }
}
#======================================================
proc ::ForceFieldToolKit::gui::bparShowMissingParsElements {} {
    # creates graphic objects to represent missing parameters

    # localize relevant variables
    variable bparIdMissingAnalyzeMolID

    # make sure that the molecule loaded during the analyze proc still exists
    if { [lsearch [molinfo list] $bparIdMissingAnalyzeMolID] == -1 } { return }

    # clear any existing objects
    ::ForceFieldToolKit::SharedFcns::ParView::clearParViewObjList -molid $bparIdMissingAnalyzeMolID

    # set material
    ::ForceFieldToolKit::SharedFcns::ParView::addMaterialObj -molid $bparIdMissingAnalyzeMolID -material Diffuse

    # build bonds
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparIdMissingAnalyzeMolID -color red
    foreach item [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.bondsTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.bondsTv set $item indsList] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparIdMissingAnalyzeMolID -type bond -indices $indSet
        }
    }

    # build angles
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparIdMissingAnalyzeMolID -color blue
    foreach item [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.anglesTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.anglesTv set $item indsList] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparIdMissingAnalyzeMolID -type angle -indices $indSet
        }
    }

    # build dihedrals
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparIdMissingAnalyzeMolID -color green
    foreach item [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.dihedralsTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.dihedralsTv set $item indsList] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparIdMissingAnalyzeMolID -type dihedral -indices $indSet
        }
    }

    # build nonbonded
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparIdMissingAnalyzeMolID -color magenta
    foreach item [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.nonbondedTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.missingPars.vizFrame.nonbondedTv set $item indsList] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparIdMissingAnalyzeMolID -type atom -indices $indSet
        }
    }
}
#======================================================
proc ::ForceFieldToolKit::gui::bparLoadRefVDWData {} {
    # general controller function for loading reference VDW
    # topology+parameter data into the treevew box

    #tk_messageBox -type ok -icon info \
    #    -message "Loading a Topology + Parameter File Pair" \
    #    -detail "The following dialogs will first request"


    # request the topology and parameter files
    set topFile [tk_getOpenFile -title "Select the TOPOLOGY File" -filetypes $::ForceFieldToolKit::gui::topType]
    set parFile [tk_getOpenFile -title "Select the PARAMETER File" -filetypes $::ForceFieldToolKit::gui::parType]

    # rudimentary file validation
    if { $topFile eq "" || $parFile eq "" || ![file exists $topFile] || ![file exists $parFile] } {
        tk_messageBox -type ok -icon warning -message "Load FAILED" -detail "Inappropriate files selected."
        return
    }

    # process the input files
    array set vdwData [::ForceFieldToolKit::gui::bparBuildVDWarray [list [list $topFile $parFile]]]

    # load the information into the treeview box
    foreach key [array names vdwData] {
        set ele [lindex $vdwData($key) 0]
        set type [lindex $vdwData($key) 1]
        set pars [lindex $vdwData($key) 2]
        set filename [lindex $vdwData($key) 3]
        set comments [lindex $vdwData($key) 4]
        lappend ::ForceFieldToolKit::gui::bparVDWtvNodeIDs [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv insert {} end -values [list $ele $type $pars $filename $comments]]
    }
    # clean up
    array unset vdwData

    # rebuild the elements and parSet drop down menus
    set eleList {}
    set parLis {}
    # find all elements in loaded par sets
    foreach entry [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv children {}] {
        lappend eleList [lindex [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv item $entry -values] 0]
        lappend parList [lindex [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv item $entry -values] 3]
    }
    # sort the elements
    set eleList [lsort -unique $eleList]
    set parList [lsort -unique $parList]
    # clear the old menu
    .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.ele.menu delete 0 end
    .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.parSet.menu delete 0 end
    # rebuild the new menus
    # ele menu
    .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.ele.menu add command -label "ALL" -command { set ::ForceFieldToolKit::gui::bparVDWele "ALL"; ::ForceFieldToolKit::gui::bparVDWshowEle }
    .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.ele.menu add separator
    foreach entry $eleList {
        .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.ele.menu add command -label $entry -command "set ::ForceFieldToolKit::gui::bparVDWele $entry; ::ForceFieldToolKit::gui::bparVDWshowEle"
    }
    # parSet menu
    .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.parSet.menu add command -label "ALL" -command { set ::ForceFieldToolKit::gui::bparVDWparSet "ALL"; ::ForceFieldToolKit::gui::bparVDWshowEle }
    .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.parSet.menu add separator
    foreach entry $parList {
        .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.parSet.menu add command -label $entry -command "set ::ForceFieldToolKit::gui::bparVDWparSet $entry; ::ForceFieldToolKit::gui::bparVDWshowEle"
    }


    # show only the selected ele
    if { $::ForceFieldToolKit::gui::bparVDWele == {} } { set ::ForceFieldToolKit::gui::bparVDWele "ALL" }
    if { $::ForceFieldToolKit::gui::bparVDWparSet == {} } { set ::ForceFieldToolKit::gui::bparVDWparSet "ALL" }

    ::ForceFieldToolKit::gui::bparVDWshowEle
}
#======================================================
proc ::ForceFieldToolKit::gui::bparVDWshowEle {} {
    # shows only the tv items for the selected element and parfile

    # detach all nodes currently in tv
    foreach item [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv children {}] {
        .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv detach $item
    }

    # cycle through all nodes in the node list and if ele/parSet match, reattach node using the move cmd
    # the node is added when...
    # ele = ALL    &&  parSet = ALL
    # ele = match  &&  parSet = ALL
    # ele = ALL    &&  parSet = match
    # ele = match  &&  parSet = match

    foreach item $::ForceFieldToolKit::gui::bparVDWtvNodeIDs {
        if { $::ForceFieldToolKit::gui::bparVDWele eq "ALL" && $::ForceFieldToolKit::gui::bparVDWparSet eq "ALL" } {
            .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv move $item {} 0
        } elseif { [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv set $item ele] eq $::ForceFieldToolKit::gui::bparVDWele && $::ForceFieldToolKit::gui::bparVDWparSet eq "ALL" } {
            .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv move $item {} 0
        } elseif { $::ForceFieldToolKit::gui::bparVDWele eq "ALL" && [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv set $item filename] eq $::ForceFieldToolKit::gui::bparVDWparSet } {
            .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv move $item {} 0
        } elseif { [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv set $item ele] eq $::ForceFieldToolKit::gui::bparVDWele && [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv set $item filename] eq $::ForceFieldToolKit::gui::bparVDWparSet} {
            .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv move $item {} 0
        }
    }


    # sort the treeview
    set eleTypeList {}
    foreach item [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv children {}] {
        set currEle [lindex [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv item $item -values] 0]
        set currType [lindex [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv item $item -values] 1]
        set currFile [lindex [.fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv item $item -values] 3]
        lappend eleTypeList [list $currEle $currType $currFile $item]
    }
    set orderedItemList {}
    foreach entry [lsort -dictionary $eleTypeList] {
        lappend orderedItemList [lindex $entry 3]
    }
    .fftk_gui.hlf.nb.buildpar.vdwPars.refvdw.tv children {} $orderedItemList
}
#======================================================
proc ::ForceFieldToolKit::gui::bparBuildVDWarray { fileList } {
    # builds an array containing VDW information

    #puts "filelist: $fileList"; flush stdout

    # build an array to match element to atomic weight
    # these are supported atoms
    array set eleByMass {
        0 H
        1 H
        4 HE
        12 C
        14 N
        16 O
        19 F
        20 NE
        23 NA
        24 MG
        27 AL
        31 P
        32 S
        35 CL
        39 K
        40 CA
        56 FE
        65 ZN
        80 BR
        127 I
        133 CS
    }

    # initialize
    array set vdwData {}

    # fileList should read:
    # {  {top1 par1} {top2 par2} ... {topN parN}  }

    foreach filePair $fileList {
        #puts "Processing filePair: $filePair"; flush stdout
        set topFile [lindex $filePair 0]
        set parFile [lindex $filePair 1]

        # TOPOLOGY FILE
        # parse MASS statments from topology
        set topIn [open $topFile r]
        while { ![eof $topIn] } {
            set inLine [gets $topIn]
            if { [regexp {^MASS} $inLine] } {
                set type [lindex $inLine 2]
                set massNum [expr {round([lindex $inLine 3])}]
                set comment [string trim [lindex [split $inLine !] end]]
                if { [info exists eleByMass($massNum)] } {
                    # if the mass number is supported, add the element
                    set vdwData($type) [list $eleByMass($massNum) {} {} {} {}]
                } else {
                    # otherwise declare the element as unknown
                    set vdwData($type) [list UNK {} {} {} {}]
                }
                lset vdwData($type) 1 $type
                lset vdwData($type) 4 $comment
            } else {
                continue
            }
        }; # end of reading topFile (while)

        close $topIn

        #puts "vdwData After topology parsing"
        #foreach key [array names vdwData] { puts "\t$vdwData($key)" }
        #flush stdout

        # PARAMETER FILE
        set rawParData {}
        set parIn [open $parFile r]
        # read in the NONBONDED section, with comments
        set readstate 0
        while { ![eof $parIn] } {
            set inLine [gets $parIn]
            switch -regexp $inLine {
                {^[ \t]*$} { continue }
                {^[ \t]*\*.*} { continue }
                {^[a-z]+} { continue }
                {^BONDS.*} { set readstate 0 }
                {^ANGLES.*} { set readstate 0 }
                {^DIHEDRALS.*} { set readstate 0 }
                {^IMPROPER.*} { set readstate 0 }
                {^CMAP.*} { set readstate 0 }
                {^NONBONDED.*} { set readstate 1 }
                {^HBOND.*} { set readstate 0 }
                {^END.*} { break }
                default {
                    if { $readstate } {
                        lappend rawParData $inLine
                    }
                }

            }; # end of lineread (switch)
        }; # end of reading parIn (while)

        close $parIn

        #puts "rawParData After reading parameter file"
        #foreach ele $rawParData { puts "\t$ele" }
        #flush stdout

        # process the vdw parameter data
        # burn the header
        while { [regexp {^[ \t]*!} [lindex $rawParData 0]] } {
            set rawParData [lreplace $rawParData 0 0]
        }

        #puts "rawParData After burning the header:"
        #foreach ele $rawParData { puts "\t$ele" }
        #flush stdout

        # build the processed vdw par data by folding two-line comments into one line
        set procParData {}
        foreach ele $rawParData {
            if { [regexp {^[ \t]+!} $ele] } {
                # line is a comment, append to previous line
                lset procParData end [concat [lindex $procParData end] $ele]
            } else {
                # line is not a comment, add to new processed list
                lappend procParData $ele
            }
        }

        #puts "procParData After folding comments:"
        #foreach ele $procParData { puts "\t$ele" }
        #flush stdout

        # reprocess the parameter data and insert into the vdwData array
        foreach ele $procParData {
            # split along comment denotation (!)
            set splitData [split $ele !]
            # parse out LJ data
            set ljData [string trim [lindex $splitData 0]]
            # build a comments list
            set comData {}
            for {set i 1} {$i < [llength $splitData]} {incr i} {
                lappend comData [string trim [lindex $splitData $i]]
            }
            # insert the data into the appropriate vdwData
            set type [lindex $ljData 0]
            set ljnorm [lrange $ljData 2 3]
            set lj14 [lrange $ljData 5 6]
            if { [info exists vdwData($type)] } {
                lset vdwData($type) 2 [list $ljnorm $lj14]
                lset vdwData($type) 3 [file tail $parFile]
                lset vdwData($type) 4 [linsert $comData 0 [lindex $vdwData($type) 4]]
            }
        }

        # message the console
        ::ForceFieldToolKit::gui::consoleMessage "VDW/LJ parameters loaded for [file rootname [file tail [lindex $filePair 1]]]"

    }; # end of cycling through file pairs (foreach)

    return [array get vdwData]
}
#======================================================
proc ::ForceFieldToolKit::gui::bparCGenFFAnalyze {} {
    # Construct molecule from input; populate parameter tvs

    # PASSED:  nothing
    # RETURNS: nothing

    # clear out any existing molecules
    if { $::ForceFieldToolKit::gui::bparCGenFFMolID != -1 && [lsearch [molinfo list] $::ForceFieldToolKit::gui::bparCGenFFMolID] > -1 }  {
        mol delete $::ForceFieldToolKit::gui::bparCGenFFMolID
        set ::ForceFieldToolKit::gui::bparCGenFFMolID -1
    }

    # clear parameter tv data
    foreach ele {bonds angles dihedrals impropers} {
        .fftk_gui.hlf.nb.buildpar.cgenff.pars.${ele}Tv delete [.fftk_gui.hlf.nb.buildpar.cgenff.pars.${ele}Tv children {}]
    }

    # analyze the input data (returns molid of the constructed molecule)
    # sanity checking is performed inside this function
    set ::ForceFieldToolKit::gui::bparCGenFFMolID [::ForceFieldToolKit::BuildPar::analyzeCGenFF]

    # constuct arrays that map parameter typedefs to indices (used to populate indList columns)
    set molid $::ForceFieldToolKit::gui::bparCGenFFMolID
    array unset bondArr; array set bondArr {}
    foreach bond [topo getbondlist -molid $molid] {
        lassign $bond ind1 ind2
        set sel [atomselect $molid "index $ind1"]; set type1 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind2"]; set type2 [$sel get type]; $sel delete
        lappend bondArr($type1,$type2) [list $ind1 $ind2]
        lappend bondArr($type2,$type1) [list $ind1 $ind2]
    }
    array unset angleArr; array set angleArr {}
    foreach angle [topo getanglelist -molid $molid] {
        lassign $angle def ind1 ind2 ind3
        set sel [atomselect $molid "index $ind1"]; set type1 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind2"]; set type2 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind3"]; set type3 [$sel get type]; $sel delete
        lappend angleArr($type1,$type2,$type3) [list $ind1 $ind2 $ind3]
        lappend angleArr($type3,$type2,$type1) [list $ind1 $ind2 $ind3]
    }
    array unset dihedralArr; array set dihedralArr {}
    foreach dihedral [topo getdihedrallist -molid $molid] {
        lassign $dihedral def ind1 ind2 ind3 ind4
        set sel [atomselect $molid "index $ind1"]; set type1 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind2"]; set type2 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind3"]; set type3 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind4"]; set type4 [$sel get type]; $sel delete
        lappend dihedralArr($type1,$type2,$type3,$type4) [list $ind1 $ind2 $ind3 $ind4]
        lappend dihedralArr($type4,$type3,$type2,$type1) [list $ind1 $ind2 $ind3 $ind4]
    }
    array unset improperArr; array set improperArr {}
    foreach improper [topo getimproperlist -molid $molid] {
        lassign $improper def ind1 ind2 ind3 ind4
        set sel [atomselect $molid "index $ind1"]; set type1 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind2"]; set type2 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind3"]; set type3 [$sel get type]; $sel delete
        set sel [atomselect $molid "index $ind4"]; set type4 [$sel get type]; $sel delete
        lappend improperArr($type1,$type2,$type3,$type4) [list $ind1 $ind2 $ind3 $ind4]
        lappend improperArr($type4,$type3,$type2,$type1) [list $ind4 $ind3 $ind2 $ind1]
    }


    # repopulate Tvs
    lassign $::ForceFieldToolKit::BuildPar::cgenffAnalogyPars bondData angleData dihedralData improperData
    foreach bond $bondData     {
        set indlist $bondArr([join [string trim [lindex $bond 0]] ","])
        lappend bond $indlist
        .fftk_gui.hlf.nb.buildpar.cgenff.pars.bondsTv insert {} end -values $bond
    }
    foreach angle $angleData   {
        set indlist $angleArr([join [string trim [lindex $angle 0]] ","])
        lappend angle $indlist
        .fftk_gui.hlf.nb.buildpar.cgenff.pars.anglesTv insert {} end -values $angle
    }
    foreach dih $dihedralData  {
        set indlist $dihedralArr([join [string trim [lindex $dih 0]] ","])
        lappend dih $indlist
        .fftk_gui.hlf.nb.buildpar.cgenff.pars.dihedralsTv insert {} end -values $dih
    }
    foreach impr $improperData {
        set indlist $improperArr([join [string trim [lindex $impr 0]] ","])
        lappend impr $indlist
        .fftk_gui.hlf.nb.buildpar.cgenff.pars.impropersTv insert {} end -values $impr
    }

    # expand tvs with data, collapse any empty tvs
    foreach ele { {bondData bonds 1} {angleData angles 3} {dihedralData dihedrals 5} {improperData impropers 6} } {
        upvar 0 [lindex $ele 0] data
        set tvtype [lindex $ele 1]
        set row [lindex $ele 2]

        if { [llength $data] == 0 } {
            # collapse
            .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvtype}Lbl configure -state disabled
            grid remove .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvtype}Tv
            grid remove .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvtype}Scroll
            grid rowconfigure .fftk_gui.hlf.nb.buildpar.cgenff.pars $row -weight 0
        } else {
            # expand
            .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvtype}Lbl configure -state normal
            grid .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvtype}Tv
            grid .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvtype}Scroll
            grid rowconfigure .fftk_gui.hlf.nb.buildpar.cgenff.pars $row -weight 1
        }
        unset data tvtype row
    }

    # update the TV title text based on whether existing parameters were also found
    # note that existing parameters are not shown, but are stored and will be written out to a separate parameter file
    lassign $::ForceFieldToolKit::BuildPar::cgenffExistingPars eBonds eAngles eDihedrals eImpropers
    if { [llength $eBonds] > 0 || [llength $eAngles] > 0 || [llength $eDihedrals] > 0 || [llength $eImpropers] > 0 } {
        .fftk_gui.hlf.nb.buildpar.cgenff.pars configure -text "CGenFF Parameter Data (existing parameters found, only showing missing parameters)"
    } else {
        .fftk_gui.hlf.nb.buildpar.cgenff.pars configure -text "CGenFF Parameter Data"
    }

    # done
    return
}
#======================================================
proc ::ForceFieldToolKit::gui::bparCGenFFWritePSFPDB {} {
    # writes molecule constructed from CGenFF to PSF/PDB file

    # PASSED: nothing
    # RETURNS: nothing

    # sanity checking
    set errorList {}
    set errorText ""
    # check that a molecule was loaded at some point
    if { $::ForceFieldToolKit::gui::bparCGenFFMolID == -1 } {
        lappend errorList "No molecule loaded from input"
    }
    # check that the loaded molecule still exists
    if { [lsearch [molinfo list] $::ForceFieldToolKit::gui::bparCGenFFMolID] == -1 } {
        lappend errorList "No molecule loaded from input no longer exists"
    }
    # check that the resname is defined
    if { [llength [string trim $::ForceFieldToolKit::BuildPar::cgenffResname]] == 0 } {
        lappend errorList "Resname is required to name the PSF/PDB files"
    }
    # check that there is a valid path which is writable
    if { [llength [string trim $::ForceFieldToolKit::BuildPar::cgenffOutPath]] == 0 } {
        lappend errorList "Empty path provided for output"
    }
    if { ![file writable $::ForceFieldToolKit::BuildPar::cgenffOutPath] } {
        lappend errorList "Output Folder is not writable"
    }
    # construct the error message, if necessary
    if { [llength $errorList] > 0 } {
        foreach ele $errorList {
            set errorText [concat $errorText\n$ele]
        }
        tk_messageBox \
            -type ok \
            -icon warning \
            -message "Application halting due to the following errors:" \
            -detail $errorText

        # there are errors, return the error response
        return
    }

    # Write the PSF/PDB file pair
    set sel [atomselect $::ForceFieldToolKit::gui::bparCGenFFMolID "all"]
    set fname [file join $::ForceFieldToolKit::BuildPar::cgenffOutPath $::ForceFieldToolKit::BuildPar::cgenffResname]
    $sel writepsf ${fname}.psf
    $sel writepdb ${fname}.pdb
    $sel delete
    unset fname

    # done
    return
}
#======================================================
proc ::ForceFieldToolKit::gui::bparCGenFFWritePAR {} {
    # writes parameter files from CGenFF input

    # PASSED: nothing
    # RETURNS: nothing

    # Note: We potentially write two separate files depending on the
    #   whether the user requested ALL parameters or only MISSING
    #   parameters from CGenFF Program when submitting

    # sanity checking
    set errorList {}
    set errorText ""
    # check that the resname is defined
    if { [llength [string trim $::ForceFieldToolKit::BuildPar::cgenffResname]] == 0} {
        lappend errorList "Resname is required to name the PAR files"
    }
    # check that there is a valid path which is writable
    if { [llength [string trim $::ForceFieldToolKit::BuildPar::cgenffOutPath]] == 0 } {
        lappend errorList "Empty path provided for output"
    }
    if { ![file writable $::ForceFieldToolKit::BuildPar::cgenffOutPath] } {
        lappend errorList "Output Folder is not writable"
    }
    # check that there is something to write
    set numbonds  [llength [.fftk_gui.hlf.nb.buildpar.cgenff.pars.bondsTv children {}]]
    set numangles [llength [.fftk_gui.hlf.nb.buildpar.cgenff.pars.anglesTv children {}]]
    set numdih    [llength [.fftk_gui.hlf.nb.buildpar.cgenff.pars.dihedralsTv children {}]]
    set numimpr   [llength [.fftk_gui.hlf.nb.buildpar.cgenff.pars.impropersTv children {}]]
    set numtotal [expr {$numbonds + $numangles + $numdih + $numimpr}]
    if { $numtotal == 0 } {
        lappend errorList "No parameter data to write"
    }
    unset numbonds numangles numdih numimpr numtotal

    # construct the error message, if necessary
    if { [llength $errorList] > 0 } {
        foreach ele $errorList {
            set errorText [concat $errorText\n$ele]
        }
        tk_messageBox \
            -type ok \
            -icon warning \
            -message "Application halting due to the following errors:" \
            -detail $errorText

        # there are errors, return the error response
        return
    }

    # process the data
    set bonds {}
    foreach ele [.fftk_gui.hlf.nb.buildpar.cgenff.pars.bondsTv children {}] {
        lassign [.fftk_gui.hlf.nb.buildpar.cgenff.pars.bondsTv item $ele -values] typedef k b0 penalty comment indlist
        lappend bonds [list $typedef [list $k $b0] $comment]
    }
    set angles {}
    foreach ele [.fftk_gui.hlf.nb.buildpar.cgenff.pars.anglesTv children {}] {
        lassign [.fftk_gui.hlf.nb.buildpar.cgenff.pars.anglesTv item $ele -values] typedef k theta kub s penalty comment indlist
        lappend angles [list $typedef [list $k $theta] [list $kub $s] $comment]
    }
    set dihedrals {}
    foreach ele [.fftk_gui.hlf.nb.buildpar.cgenff.pars.dihedralsTv children {}] {
        lassign [.fftk_gui.hlf.nb.buildpar.cgenff.pars.dihedralsTv item $ele -values] typedef k n delta penalty comment indlist
        lappend dihedrals [list $typedef [list $k $n $delta] $comment]
    }
    set impropers {}
    foreach ele [.fftk_gui.hlf.nb.buildpar.cgenff.pars.impropersTv children {}] {
        lassign [.fftk_gui.hlf.nb.buildpar.cgenff.pars.impropersTv item $ele -values] typedef k psi penalty comment indlist
        lappend impropers [list $typedef [list $k $psi] $comment]
    }

    # use the shared parameter writer
    set fname [file join $::ForceFieldToolKit::BuildPar::cgenffOutPath $::ForceFieldToolKit::BuildPar::cgenffResname]
    # write out missing pars
    ::ForceFieldToolKit::SharedFcns::writeParFile [list $bonds $angles $dihedrals $impropers {}] ${fname}.analogy.par
    # write out existing pars
    if { [llength $::ForceFieldToolKit::BuildPar::cgenffExistingPars] > 0 } {
        ::ForceFieldToolKit::SharedFcns::writeParFile [concat $::ForceFieldToolKit::BuildPar::cgenffExistingPars {}] ${fname}.existing.par
    }

    # done
    return
}
#======================================================
proc ::ForceFieldToolKit::gui::bparCGenFFSortPars {tvName colName colNum} {
    # Sorts parameters in specified tv
    # triggered by heading click

    # PASSED:
    #   tvName - name of tv to sort
    #   colName - name of column to sort by
    #   colNum - number of the column for the tv

    # RETURNS: nothing

    # setup a sort list
    set sortList {}
    # cycle through each child and get data from colName
    foreach child [.fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvName}Tv children {}] {
        lappend sortList [list $child [.fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvName}Tv set $child $colName]]
    }

    # tv ordering is performed by using a sort in the inverse order than desired
    # then cycle through the list from front to back, placing each ele at the front
    # of the tv

    # check the current sort column for the specified tvName
    # if the column is the current sort, invert existing order
    # otherwise perform a dictionary sort
    if { $::ForceFieldToolKit::gui::bparCGenFFTvSort($tvName) == $colNum } {
        # invert the existing order
        foreach ele $sortList {
            .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvName}Tv move [lindex $ele 0] {} 0
        }
    } else {
        # perform a dictionary sort
        foreach ele [lsort -index 1 -dictionary -decreasing $sortList] {
            .fftk_gui.hlf.nb.buildpar.cgenff.pars.${tvName}Tv move [lindex $ele 0] {} 0
        }
    }

    # update the sort array
    set ::ForceFieldToolKit::gui::bparCGenFFTvSort($tvName) $colNum

    # return
    return
}
#======================================================
proc ::ForceFieldToolKit::gui::bparCGenFFTvSelectionDidChange {} {
    # Constructs ParView Objects to visualize parameters in molecule structures

    # PASSED: nothing
    # RETURNS: nothing

    # localize relevant variables
    variable bparCGenFFMolID

    # bail if the molecule isn't set or no longer exists
    if { $bparCGenFFMolID == -1 || [lsearch [molinfo list] $bparCGenFFMolID] == -1 } { return }

    # clear out any old ParView data and (re)setup materials
    ::ForceFieldToolKit::SharedFcns::ParView::clearParViewObjList -molid $bparCGenFFMolID
    ::ForceFieldToolKit::SharedFcns::ParView::addMaterialObj -molid $bparCGenFFMolID -material AOShiny

    # bonds
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparCGenFFMolID -color red
    foreach item [.fftk_gui.hlf.nb.buildpar.cgenff.pars.bondsTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.cgenff.pars.bondsTv set $item indlist] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparCGenFFMolID -type bond -indices $indSet
        }
    }

    # angles
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparCGenFFMolID -color blue
    foreach item [.fftk_gui.hlf.nb.buildpar.cgenff.pars.anglesTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.cgenff.pars.anglesTv set $item indlist] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparCGenFFMolID -type angle -indices $indSet
        }
    }

    # dihedrals
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparCGenFFMolID -color green
    foreach item [.fftk_gui.hlf.nb.buildpar.cgenff.pars.dihedralsTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.cgenff.pars.dihedralsTv set $item indlist] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparCGenFFMolID -type dihedral -indices $indSet
        }
    }

    # impropers
    ::ForceFieldToolKit::SharedFcns::ParView::addColorObj -molid $bparCGenFFMolID -color magenta
    foreach item [.fftk_gui.hlf.nb.buildpar.cgenff.pars.impropersTv selection] {
        foreach indSet [.fftk_gui.hlf.nb.buildpar.cgenff.pars.impropersTv set $item indlist] {
            ::ForceFieldToolKit::SharedFcns::ParView::addParObject -molid $bparCGenFFMolID -type improper -indices $indSet
        }
    }

    # done
    return
}
#======================================================


#------------------------------------------------------
# GenZMat Specific
#------------------------------------------------------
proc ::ForceFieldToolKit::gui::gzmToggleLabels {} {
    # toggles atom labels for TOP molecule to help determine donor and acceptor indices

    variable gzmAtomLabels

    if { [llength $gzmAtomLabels] > 0 } {
        foreach label $gzmAtomLabels {graphics top delete $label}
        set gzmAtomLabels {}
    } else {
        draw color lime
        set selAll [atomselect top all]
        foreach ind [$selAll get index] {
            set sel [atomselect top "index $ind"]
            lappend gzmAtomLabels [draw text [join [$sel get {x y z}]] $ind size 3]
            $sel delete
        }
        $selAll delete
    }
}
#======================================================
proc ::ForceFieldToolKit::gui::gzmToggleSpheres {} {
    # toggles colored spheres for atoms in donor and acceptor lists to help check indices
    # blue spheres for donors
    # red spheres for acceptors
    # gree sphere for both donor AND acceptors

    variable gzmVizSpheresDon
    variable gzmVizSpheresAcc
    variable gzmVizSpheresBoth

    graphics top materials on
    graphics top material Diffuse

    # initialize local lists
    set donList $::ForceFieldToolKit::GenZMatrix::donList
    set accList $::ForceFieldToolKit::GenZMatrix::accList
    set bothList {}

    # find the overlap
    foreach ele $donList {
        if { [lsearch $accList $ele] != -1 } { lappend bothList $ele }
    }
    # remove overlaps from don and acc lists
    foreach ele $bothList {
        # remove from donList
        set donInd [lsearch $donList $ele]
        if { $donInd != -1 } { lreplace $donList $donInd $donInd }
        # remove from accList
        set accInd [lsearch $accList $ele]
        if { $accInd != -1 } { lreplace $accList $accInd $accInd }
    }

    # toggle the graphics elements
    if { [llength $gzmVizSpheresDon] > 0 } {
        foreach sphere $gzmVizSpheresDon {graphics top delete $sphere}
        set gzmVizSpheresDon {}
    } else {
        draw color blue
        foreach ind $donList {
            set sel [atomselect top "index $ind"]
            lappend gzmVizSpheresDon [graphics top sphere [join [$sel get {x y z}]] radius 0.2 resolution 30]
            $sel delete
        }
    }

    if { [llength $gzmVizSpheresAcc] > 0 } {
        foreach sphere $gzmVizSpheresAcc {graphics top delete $sphere}
        set gzmVizSpheresAcc {}
    } else {
        draw color red
        foreach ind $accList {
            set sel [atomselect top "index $ind"]
            lappend gzmVizSpheresAcc [graphics top sphere [join [$sel get {x y z}]] radius 0.2 resolution 30]
            $sel delete
        }
    }

    if { [llength $gzmVizSpheresBoth] > 0 } {
        foreach sphere $gzmVizSpheresBoth {graphics top delete $sphere}
        set gzmVizSpheresBoth {}
    } else {
        draw color green
        foreach ind $bothList {
            set sel [atomselect top "index $ind"]
            lappend gzmVizSpheresBoth [graphics top sphere [join [$sel get {x y z}]] radius 0.2 resolution 30]
            $sel delete
        }
    }
}
#======================================================
proc ::ForceFieldToolKit::gui::gzmAutoDetect {} {
    # very simple method to autodetecting donors and acceptors

    # add all hydrogens
    set selHydrogens [atomselect top "element H"]
    set ::ForceFieldToolKit::GenZMatrix::donList [$selHydrogens get index]
    $selHydrogens delete

    # add all heavy atoms with less than 4 bonded atoms (generally tetrahedral)
    set ::ForceFieldToolKit::GenZMatrix::accList {}
    set selHeavyAtoms [atomselect top "all and not element H"]
    foreach hvyatom [$selHeavyAtoms get index] {
        set sel [atomselect top "index $hvyatom"]
        if { [llength [lindex [$sel getbonds] 0]] < 4 } {
            lappend ::ForceFieldToolKit::GenZMatrix::accList $hvyatom
            if { [$sel get element] eq "C" } {
                lappend ::ForceFieldToolKit::GenZMatrix::donList $hvyatom
            }
        }
        $sel delete
    }
    $selHeavyAtoms delete
}
#======================================================


#------------------------------------------------------
# ChargeOpt Specific
#------------------------------------------------------
proc ::ForceFieldToolKit::gui::coptShowAtomLabels {} {
    # shows labels to aid in setting up charge optimizations
    # label can be none, index, name, type, charge

    variable coptAtomLabel
    variable coptAtomLabelInd

    # reset graphics
    foreach ind $coptAtomLabelInd {graphics top delete $ind}
    set $coptAtomLabelInd {}
    draw color lime

    # set new labels
    foreach atomInd [[atomselect top all] get index] {
        set sel [atomselect top "index $atomInd"]
        switch -exact $coptAtomLabel {
            "Index"  { lappend coptAtomLabelInd [draw text [join [$sel get {x y z}]] "[$sel get index]" size 3]  }
            "Name"   { lappend coptAtomLabelInd [draw text [join [$sel get {x y z}]] "[$sel get name]" size 3]   }
            "Type"   { lappend coptAtomLabelInd [draw text [join [$sel get {x y z}]] "[$sel get type]" size 3]   }
            "Charge" { lappend coptAtomLabelInd [draw text [join [$sel get {x y z}]] "[format "%0.3f" [$sel get charge]]" size 3] }
            default  {}
        }
        $sel delete
    }
}
#======================================================
proc ::ForceFieldToolKit::gui::coptSetEditData { box } {
    # grabs data from the currently selected Log File entry and copies into the Edit Box


    if { $box eq "cconstr" } {
        # for the Charge Constraints box (cconstr)
        set editData [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData item [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData selection] -values]
        set ::ForceFieldToolKit::gui::coptEditGroup [lindex $editData 0]
        set ::ForceFieldToolKit::gui::coptEditInit [lindex $editData 1]
        set ::ForceFieldToolKit::gui::coptEditLowBound [lindex $editData 2]
        set ::ForceFieldToolKit::gui::coptEditUpBound [lindex $editData 3]
        unset editData
    } elseif { $box eq "wie" } {
        # for the Water Interaction Energies box (wie)
        set editData [.fftk_gui.hlf.nb.chargeopt.qmt.wie.logData item [.fftk_gui.hlf.nb.chargeopt.qmt.wie.logData selection] -values]
        set ::ForceFieldToolKit::gui::coptEditLog [lindex $editData 0]
        set ::ForceFieldToolKit::gui::coptEditAtomName [lindex $editData 1]
        set ::ForceFieldToolKit::gui::coptEditWeight [lindex $editData 2]
        unset editData
    } elseif { $box eq "results" } {
        set editData [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups item [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups selection] -values]
        set ::ForceFieldToolKit::gui::coptEditFinalCharge [lindex $editData 2]
    }

}
#======================================================
proc ::ForceFieldToolKit::gui::coptClearEditData { box } {

    if {$box eq "cconstr"} {
        # clear charge constraints edit boxes
        set ::ForceFieldToolKit::gui::coptEditGroup {}
        set ::ForceFieldToolKit::gui::coptEditInit {}
        set ::ForceFieldToolKit::gui::coptEditLowBound {}
        set ::ForceFieldToolKit::gui::coptEditUpBound {}
    } elseif {$box eq "wie"} {
        # clear the water interaction energy edit boxes
        set ::ForceFieldToolKit::gui::coptEditLog {}
        set ::ForceFieldToolKit::gui::coptEditAtomName {}
        set ::ForceFieldToolKit::gui::coptEditWeight {}
    } elseif {$box eq "results" } {
        set ::ForceFieldToolKit::gui::coptEditFinalCharge {}
    }

}
#======================================================
proc ::ForceFieldToolKit::gui::coptGuessChargeGroups { molid } {

    # initialize some variables
    array set indexTree {}
    set typeFPList {}

    set cgNames {}
    set cgInit {}
    set cgLowBound {}
    set cgUpBound {}


    # set the list of all atoms
    set allList [lsort -dictionary [[atomselect $molid all] get index]]

    # cycle through each atom as the root
    foreach rootAtom $allList {

        # initialize the indexTree array for this particular atom
        # by setting node 0
        set indexTree($rootAtom) $rootAtom

        # initialize the traveledList
        set traveledList $rootAtom

        # initialize nodeCounter
        set nodeCount 1
        # traverse nodes until all atoms are covered
        while { [lsort -dictionary $traveledList] != $allList } {

            # initialize a temporary list to hold new atoms for this node
            set tmpNodeList {}

            # find bonded atoms for each atom in the preceeding node
            foreach precNodeAtom [lindex $indexTree($rootAtom) [expr {$nodeCount - 1}]] {

                # find the atoms
                set bondedAtoms [lindex [[atomselect $molid "index $precNodeAtom"] getbonds] 0]
                # check to see if we've already traveled to any of these atoms
                foreach bAtom $bondedAtoms {
                    if { [lsearch -exact -integer $traveledList $bAtom] == -1 } {
                        # new atom, append it to the list of atoms that we've been to so that we won't come back
                        lappend traveledList $bAtom
                        # add to the temp current node list
                        lappend tmpNodeList $bAtom
                    } else {
                        # we've already been to this atom, so skip it
                    }
                }; # end of travelList check foreach

            }; # end of node cycle foreach

            # now that we have only atoms that we haven't traveled to
            # we can write them to the current node
            lappend indexTree($rootAtom) $tmpNodeList

            # increment the node counter to move onto the next node
            incr nodeCount

        }; # end of while statement that traverses the atom tree

        # convert the indexTree into type fingerprint
        set typeFP {}
        foreach node $indexTree($rootAtom) {
            set nodeAtomTypes {}
            foreach atom $node {
                lappend nodeAtomTypes [[atomselect $molid "index $atom"] get type]
            }
            lappend typeFP [lsort -dictionary $nodeAtomTypes]
        }

        # append the fingerprint to the full list of fingerprints
        # if everything sorted properly, then the index should match the atom index
        lappend typeFPList $typeFP

    }; # end of foreach that cycles through each atom


    # define charge groups based on the type fingerprints
    set cgInd {}
    foreach atom $allList {
        # find the current atom's finger print
        set atomFP [lindex $typeFPList $atom]
        # search against the full list of finger prints, and sort the matches
        set fpMatches [lsort -dictionary [lsearch -exact -all $typeFPList $atomFP]]
        # append the matches to the master match file
        lappend cgInd $fpMatches
    }
    # remove duplicate matches
    set cgInd [lsort -dictionary -unique $cgInd]

    # if charge group is HA (non-polar hydrogens), then remove them
    # work from end to beginning so that items can be removed without shifting contents
    for {set i [expr {[llength $cgInd] - 1}] } {$i >= 0} {incr i -1} {
        set atomInd [lindex [lindex $cgInd $i] 0]
        set atomType [[atomselect $molid "index $atomInd"] get type]
        if { $atomType eq "HA" } {
            set cgInd [lreplace $cgInd $i $i]
        }
    }

    # convert to indices to atom name
    foreach cg $cgInd {
        set atomNames {}
        foreach atom $cg {
            lappend atomNames [[atomselect $molid "index $atom"] get name]
        }
        lappend cgNames $atomNames
    }

    # decide some guess parameters
    for {set i 0} {$i < [llength $cgInd]} {incr i} {
        set atomIndex [lindex [lindex $cgInd $i] 0]
        set temp [atomselect $molid "index $atomIndex"]

        # init value: grab PSF reCharge charge value
        lappend cgInit [format "%.4f" [$temp get charge]]

        # bounds
        switch -exact [$temp get element] {
            "H" {
                lappend cgLowBound "0.0"
                lappend cgUpBound "1.0"
            }

            "C" {
                lappend cgLowBound "-1.0"
                lappend cgUpBound "1.0"
            }

            "O" {
                lappend cgLowBound "-1.0"
                lappend cgUpBound "0.0"
            }

            "N" {
                lappend cgLowBound "-1.0"
                lappend cgUpBound "0.0"
            }

            default {
                lappend cgLowBound "-2.0"
                lappend cgHighBound "2.0"
            }
        }; # end switch

        $temp delete

    }

    #puts "Charge Groups: $cgNames"
    #puts "Init: $cgInit"
    #puts "Lower Bound: $cgLowBound"
    #puts "Upper Bound: $cgUpBound"

    # clear the treeview box
    .fftk_gui.hlf.nb.chargeopt.cconstr.chargeData delete [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData children {}]
    # insert data into treeview box
    for {set i 0} {$i < [llength $cgNames]} {incr i} {
        .fftk_gui.hlf.nb.chargeopt.cconstr.chargeData insert {} end -values [list "[lindex $cgNames $i]" "[lindex $cgInit $i]" "[lindex $cgLowBound $i]" "[lindex $cgUpBound $i]"]
    }

}
#======================================================
proc ::ForceFieldToolKit::gui::coptCalcChargeSum {} {
    # calculates the charge sum based on current charges
    # and the defined charge groups

    # build an exclude list for the atoms in the charge groups
    set excludeList {}
    foreach entry [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData children {}] {
        foreach atom [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData set $entry group] {
            lappend excludeList $atom
        }
    }

    # find atom names
    set temp [atomselect top all]
    set atomNames [$temp get name]
    unset temp

    # cycle through each atom name
    # if it's not defined in the charge groups,add the current charge
    set csum 0
    foreach atom $atomNames {
        if { [lsearch $excludeList $atom] == -1 } {
            set temp [atomselect top "name $atom"]
            set csum [expr {$csum + [$temp get charge]}]
            unset temp
        }
    }

    # return the charge sum
    return [format "%0.2f" [expr {-1*$csum}]]
}
#======================================================
proc ::ForceFieldToolKit::gui::coptCalcChargeSumNEW { molID } {
    # calculates the charge sum based on psf and override charges

    # initialize some variables
    set optAtomCharge 0
    set ovrAtomCharge 0
    set psfAtomCharge 0

    # build a list of atoms to optimize
    set optAtoms {}
    foreach entry [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData children {}] {
        foreach atom [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData set $entry group] {
            lappend optAtoms $atom
        }
    }

    # build a list of atoms to override and their charges, if active
    set ovrAtoms {}
    set ovrCharge {}
    if { $::ForceFieldToolKit::ChargeOpt::reChargeOverride } {
        foreach ele $::ForceFieldToolKit::ChargeOpt::reChargeOverrideCharges {
            lappend ovrAtoms [lindex $ele 0]
            lappend ovrCharge [lindex $ele 1]
        }
    }

    # find all atom names in TOP
    set temp [atomselect $molID all]
    set atomNames [$temp get name]
    $temp delete

    # cycle through each atom name in TOP
    foreach atom $atomNames {
        if { [lsearch $optAtoms $atom] != -1 } {
            # it's in the charge group, nothing to be done
            continue
        } elseif { [set ind [lsearch $ovrAtoms $atom]] != -1} {
            # take charge from override list
            set ovrAtomCharge [ expr { $ovrAtomCharge + [lindex $ovrCharge $ind] } ]
            continue
        } else {
            # take charge from TOP (psf)
            set temp [atomselect $molID "name $atom"]
            set psfAtomCharge [ expr { $psfAtomCharge + [$temp get charge] } ]
            $temp delete
            continue
        }
    }

    # calc the charge sum for the optimized atoms
    set optAtomCharge [ expr { (-1 * ($ovrAtomCharge + $psfAtomCharge)) + $::ForceFieldToolKit::gui::coptNetCharge } ]

    # return
    return [list [format "%0.2f" $optAtomCharge] [format "%0.2f" $ovrAtomCharge] [format "%0.2f" $psfAtomCharge]]

}
#======================================================
proc ::ForceFieldToolKit::gui::coptRunOpt {} {
    # procedure for button to run the charge optimization

    # reset some variables
    set ::ForceFieldToolKit::ChargeOpt::parList {}
    set ::ForceFieldToolKit::ChargeOpt::chargeGroups {}
    set ::ForceFieldToolKit::ChargeOpt::chargeInit {}
    set ::ForceFieldToolKit::ChargeOpt::chargeBounds {}
    set ::ForceFieldToolKit::ChargeOpt::logFileList {}
    set ::ForceFieldToolKit::ChargeOpt::atomList {}
    set ::ForceFieldToolKit::ChargeOpt::indWeights {}

    # build and set the parList from treeview data
    foreach tvItem [.fftk_gui.hlf.nb.chargeopt.input.parFilesBox children {}] {
        lappend ::ForceFieldToolKit::ChargeOpt::parList [lindex [.fftk_gui.hlf.nb.chargeopt.input.parFilesBox item $tvItem -values] 0]
    }

    # build and set the charge constraints from treeview data
    foreach tvItem [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData children {}] {
        set datavals [.fftk_gui.hlf.nb.chargeopt.cconstr.chargeData item $tvItem -values]
        lappend ::ForceFieldToolKit::ChargeOpt::chargeGroups [lindex $datavals 0]
        lappend ::ForceFieldToolKit::ChargeOpt::chargeInit [lindex $datavals 1]
        lappend ::ForceFieldToolKit::ChargeOpt::chargeBounds [list [lindex $datavals 2] [lindex $datavals 3]]
    }

    # build and set the logFileList, atomList, and indWeights from treeview data
    foreach tvItem [.fftk_gui.hlf.nb.chargeopt.qmt.wie.logData children {}] {
        set datavals [.fftk_gui.hlf.nb.chargeopt.qmt.wie.logData item $tvItem -values]
        lappend ::ForceFieldToolKit::ChargeOpt::logFileList [lindex $datavals 0]
        lappend ::ForceFieldToolKit::ChargeOpt::atomList [lindex $datavals 1]
        lappend ::ForceFieldToolKit::ChargeOpt::indWeights [lindex $datavals 2]
    }


    # print setup settings in debugging mode
    if { $::ForceFieldToolKit::ChargeOpt::debug } {
        ::ForceFieldToolKit::ChargeOpt::printSettings stdout
    }

    # run the optimization
    # first, check to see if build script setting is checked
    # if yes, then write a script that can be run independently
    # if no, then run optimization as normal
    if { $::ForceFieldToolKit::gui::coptBuildScript } {
        set ::ForceFieldToolKit::gui::coptStatus "Writing to script..."
        update idletasks
        ::ForceFieldToolKit::ChargeOpt::buildScript [file dirname $::ForceFieldToolKit::ChargeOpt::outFileName]/ChargeOptScript.tcl
        set ::ForceFieldToolKit::gui::coptStatus "IDLE"
        ::ForceFieldToolKit::gui::consoleMessage "Charge optimization script written"
    } else {
        set ::ForceFieldToolKit::gui::coptStatus "Running..."
        ::ForceFieldToolKit::gui::consoleMessage "Charge optimization started"
        update idletasks
        # run the optimization
        ::ForceFieldToolKit::ChargeOpt::optimize

        ## replaced by code just below which modifies tv box contents
        ## clear any old results and then load the new results
        ##.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups delete [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups children {}]
        ##foreach returnCharge $::ForceFieldToolKit::ChargeOpt::returnFinalCharges {
        ##    .fftk_gui.hlf.nb.chargeopt.results.container1.cgroups insert {} end -values [list [lindex $returnCharge 0] [lindex $returnCharge 1]]
        ##}

        ## NEW results tv box function that also holds previous run's values
        # move all final charges to previous charges
        foreach ele [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups children {}] {
            .fftk_gui.hlf.nb.chargeopt.results.container1.cgroups set $ele prevCharge [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups set $ele finalCharge]
            .fftk_gui.hlf.nb.chargeopt.results.container1.cgroups set $ele finalCharge {}
        }
        # build array of existing elements
        array unset cgroupArray
        array set cgroupArray {}
        foreach ele [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups children {}] {
            set cgroupArray([.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups set $ele group]) $ele
        }
        # read in new set of return charges
        foreach returnCharge $::ForceFieldToolKit::ChargeOpt::returnFinalCharges {
            if { [info exists cgroupArray([lindex $returnCharge 0])] } {
                .fftk_gui.hlf.nb.chargeopt.results.container1.cgroups set $cgroupArray([lindex $returnCharge 0]) finalCharge [lindex $returnCharge 1]
            } else {
                .fftk_gui.hlf.nb.chargeopt.results.container1.cgroups insert {} end -values [list [lindex $returnCharge 0] {} [lindex $returnCharge 1]]
            }
        }
        # clean up
        array unset cgroupArray
        ## <end new code>

        # update the charge total
        ::ForceFieldToolKit::gui::coptCalcFinalChargeTotal

        # set the staus label to idle
        set ::ForceFieldToolKit::gui::coptStatus "IDLE"
        ::ForceFieldToolKit::gui::consoleMessage "Charge optimization finished"
    }

    # DONE

}
#======================================================
proc ::ForceFieldToolKit::gui::coptParseLog { logFile } {
    # reads a log file from a previous charge optimization and imports
    # the final charge groups into the results treeview box

    # simple validation
    if { $logFile eq "" || ![file exists $logFile] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot find charge optimization LOG file."
        return
    }

    # open the file
    set inFile [open $logFile r]
    set readState 0

    # read through the file a line at a time
    while { [eof $inFile] != 1 } {
        set inLine [gets $inFile]

        # determine if we've reached the data that we're interested in, and read if we are
        switch -exact $inLine {
            "FINAL CHARGES" { set readState 1 }
            "END" { set readState 0 }
            default {
                if { $readState } {
                    .fftk_gui.hlf.nb.chargeopt.results.container1.cgroups insert {} end -values [list [lindex $inLine 0] {} [lindex $inLine 1]]
                } else {
                    continue
                }
            }
        }; # end switch
    }; # end of log file

    # update the charge total
    ::ForceFieldToolKit::gui::coptCalcFinalChargeTotal

    # clean up
    close $inFile
}
#======================================================
proc ::ForceFieldToolKit::gui::coptCalcFinalChargeTotal {} {
    variable coptFinalChargeTotal

    set cumsum 0

    # cycle through all items in the final charge groups box and sum the charge values
    foreach entryItem [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups children {}] {
        set data [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups item $entryItem -values]
        if { [lindex $data 2] != {} } {
            set cumsum [expr {$cumsum + ([llength [lindex $data 0]] * [lindex $data 2])}]
        } else {
            continue
        }
    }

    # set the final cumulative sum
    set coptFinalChargeTotal [format "%0.3f" $cumsum]
}
#======================================================
proc ::ForceFieldToolKit::gui::coptWriteNewPSF {} {
    # writes a PSF file with the updated charges

    # simple validation
    if { $::ForceFieldToolKit::ChargeOpt::psfPath eq "" || ![file exists $::ForceFieldToolKit::ChargeOpt::psfPath] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot find PSF file."
        return
    }
    if { $::ForceFieldToolKit::ChargeOpt::pdbPath eq "" || ![file exists $::ForceFieldToolKit::ChargeOpt::pdbPath] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot find PDB file."
        return
    }
    if { $::ForceFieldToolKit::gui::coptPSFNewPath eq "" } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Updated PSF filename was not specified."
        return
    }
    if { ![file writable [file dirname $::ForceFieldToolKit::gui::coptPSFNewPath]] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot write to output directory."
        return
    }

    # reload the PSF/PDB file pair
    mol new $::ForceFieldToolKit::ChargeOpt::psfPath
    mol addfile $::ForceFieldToolKit::ChargeOpt::pdbPath

    # reType/reCharge, taking into account reChargeOverride settings (if set)
    # reTypeFromPSF/reChargeFromPSF has been depreciated
    #::ForceFieldToolKit::SharedFcns::reTypeFromPSF $::ForceFieldToolKit::ChargeOpt::psfPath "top"
    #::ForceFieldToolKit::SharedFcns::reChargeFromPSF $ForceFieldToolKit::ChargeOpt::psfPath "top"
    if { $::ForceFieldToolKit::ChargeOpt::reChargeOverride } {
        foreach ovr $::ForceFieldToolKit::ChargeOpt::reChargeOverrideCharges {
            set temp [atomselect top "name [lindex $ovr 0]"]
            $temp set charge [lindex $ovr 1]
            $temp delete
        }
    }

    # cycle through loaded results data
    foreach CGentry [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups children {}] {

        # parse data values as a whole
        set data [.fftk_gui.hlf.nb.chargeopt.results.container1.cgroups item $CGentry -values]

        if { [lindex $data 2] ne "" } {
            # parse charge
            set charge [lindex $data 2]

            # reset the charge for each atom in the charge groups
            foreach atomName [lindex $data 0] {
                [atomselect top "name $atomName"] set charge $charge
            }
        } else {
            continue
        }
    }

    # write the psf file
    [atomselect top all] writepsf $::ForceFieldToolKit::gui::coptPSFNewPath

    # cleanup
    mol delete top

}
#======================================================


#------------------------------------------------------
# BondAngleOpt Specific
#------------------------------------------------------
#======================================================
proc ::ForceFieldToolKit::gui::baoptGuessPars {} {
    # guesses parameters and initial values for bonds/angles

    # gui message for user feedback (more for slow machines)
    set ::ForceFieldToolKit::gui::baoptStatus "Guessing Bond/Angle Initial Values"
    ::ForceFieldToolKit::gui::consoleMessage "Guessing Bond/Angle Initial Values--Start"
    update idletasks

    # set local variables for required input files
    set psf $::ForceFieldToolKit::BondAngleOpt::psf
    set hessLog $::ForceFieldToolKit::BondAngleOpt::hessLog
    set parInProg $::ForceFieldToolKit::BondAngleOpt::parInProg

    # sanity check to make sure that we have ~valid input
    set errorList {}; set errorText ""
    if { $psf eq "" } {
        lappend errorList "No PSF file was specified."
    } elseif { ![file exists $psf] } {
        lappend errorList "Cannot find PSF file."
    }
    if { $hessLog eq "" } {
        lappend errorList "No hessian LOG file was specified."
    } elseif { ![file exists $hessLog] } {
        lappend errorList "Cannot find hessian LOG file."
    }
    if { $parInProg eq "" } {
        lappend errorList "No in-progress PAR file was specified."
    } elseif { ![file exists $parInProg] } {
        lappend errorList "Cannot find in-progress PAR file."
    }

    if { [llength $errorList] > 0 } {
        foreach ele $errorList {
            set errorText [concat $errorText\n$ele]
        }
        tk_messageBox \
            -type ok \
            -icon warning \
            -message "Action halting due to the following error(s):" \
            -detail $errorText

        return
    }

    # load the hessian into a molecular area
    set hessLogID [mol new $psf]
    # reTypeFromPSF/reChargeFromPSF has been depreciated
    # ::ForceFieldToolKit::SharedFcns::reTypeFromPSF $psf $hessLogID
    ::QMtool::use_vmd_molecule $hessLogID
    ::QMtool::load_gaussian_log $hessLog $hessLogID

    # not sure if this step is necessary
    # get the internal coordinates
    #set zmatQm [::QMtool::get_internal_coordinates]

    # calculate the effective QM PES and eq geometry for all internal coords
    lassign [::ForceFieldToolKit::BondAngleOpt::computePESqm $hessLogID] zmatqmEff trashCollector

    # average replicates from guessZmatqmEff
    set replicateAvgZmat [::ForceFieldToolKit::SharedFcns::avgZmatReplicates $hessLogID $zmatqmEff]

    # process bonds/angles and crosscheck the averaged zmat against the in-progress file

    # clear out the tv box
    .fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt delete [.fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt children {}]

    # read in the in-prog parameters
    lassign [::ForceFieldToolKit::SharedFcns::readParFile $parInProg] bondPars anglePars trashCollector

    # bonds
    foreach bond [lsearch -index 0 -inline -all $replicateAvgZmat "bond"] {
        lassign $bond indDef typeDef fc eq trashCollector
        if { [lsearch -index 0 $bondPars $typeDef] != -1 } {
            # zmat bond found in in-progress pars
            .fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt insert {} end -values [list bond $typeDef $fc $eq]
        } elseif { [lsearch -index 0 $bondPars [lreverse $typeDef]] != -1 } {
            # zmat bond found in in-progress pars with reversed type definition
            .fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt insert {} end -values [list bond [lreverse $typeDef] $fc $eq]
        } else {
            # zmat bond not found in in-progress parameters
            continue
        }
    }

    # angles
    foreach angle [lsearch -regexp -index 0 -inline -all $replicateAvgZmat "angle|lbend"] {
        lassign $angle indDef typeDef fc eq trashCollector
        if { [lsearch -index 0 $anglePars $typeDef] != -1 } {
            # zmat angle found in in-progress pars
            .fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt insert {} end -values [list angle $typeDef $fc $eq]
        } elseif { [lsearch -index 0 $anglePars [lreverse $typeDef]] != -1 } {
            # zmat angle found in in-progress pars with reversed type definition
            .fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt insert {} end -values [list angle [lreverse $typeDef] $fc $eq]
        } else {
            # zmat angle not found in in-progress parameters
            continue
        }
    }

    # clean up
    mol delete $hessLogID
    unset psf hessLog parInProg errorList errorText hessLogID zmatqmEff bondPars anglePars trashCollector
    # unset zmatQm

    # gui message
    set ::ForceFieldToolKit::gui::baoptStatus "IDLE"
    ::ForceFieldToolKit::gui::consoleMessage "Guessing Bond/Angle Initial Values--Finish"
    update idletasks
}
#======================================================
proc ::ForceFieldToolKit::gui::baoptRunOpt {} {
    # procedure for button to run the bonds/angles optimization

    # reset some variables
    set ::ForceFieldToolKit::BondAngleOpt::inputBondPars {}
    set ::ForceFieldToolKit::BondAngleOpt::inputAnglePars {}
    set ::ForceFieldToolKit::BondAngleOpt::parlist {}

    # cycle through each item in the treeview box, sort bonds and angles into their respective lists
    foreach tvItem [.fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt children {}] {
        set itemData [.fftk_gui.hlf.nb.bondangleopt.pconstr.pars2opt item $tvItem -values]
        if { [lindex $itemData 0] eq "bond" } {
            lappend ::ForceFieldToolKit::BondAngleOpt::inputBondPars [lrange $itemData 1 3]
        } elseif { [lindex $itemData 0] eq "angle" } {
            lappend ::ForceFieldToolKit::BondAngleOpt::inputAnglePars [lrange $itemData 1 3]
        }
    }

    # build the list of parameter files
    foreach tvItem [.fftk_gui.hlf.nb.bondangleopt.input.parFiles children {}] {
        lappend ::ForceFieldToolKit::BondAngleOpt::parlist [lindex [.fftk_gui.hlf.nb.bondangleopt.input.parFiles item $tvItem -values] 0]
    }

    # add the in-progress par file to the (end of the) list
    lappend ::ForceFieldToolKit::BondAngleOpt::parlist $::ForceFieldToolKit::BondAngleOpt::parInProg

    # run the optimization
    # first, check to see if build script setting is checked
    if { $::ForceFieldToolKit::gui::baoptBuildScript } {
        # build a script instead of running directly
        set ::ForceFieldToolKit::gui::baoptStatus "Writing to script..."
        update idletasks
        ::ForceFieldToolKit::BondAngleOpt::buildScript BondedOpt-RunScript.tcl
        set ::ForceFieldToolKit::gui::baoptStatus "IDLE"
    } else {
        # run optimization directly
        ::ForceFieldToolKit::gui::consoleMessage "Bonded Optimization Started"
        set ::ForceFieldToolKit::gui::baoptStatus "Running..."
        update idletasks
        lassign [::ForceFieldToolKit::BondAngleOpt::optimize] results finalObjValue
        # update the results tv box
        .fftk_gui.hlf.nb.bondangleopt.results.pars2opt delete [.fftk_gui.hlf.nb.bondangleopt.results.pars2opt children {}]
        foreach ele $results { .fftk_gui.hlf.nb.bondangleopt.results.pars2opt insert {} end -values $ele }
        # update the results obj value lables
        set ::ForceFieldToolKit::gui::baoptReturnObjPrevious $::ForceFieldToolKit::gui::baoptReturnObjCurrent
        set ::ForceFieldToolKit::gui::baoptReturnObjCurrent $finalObjValue
        # update the status label and message the console log
        set ::ForceFieldToolKit::gui::baoptStatus "IDLE"
        ::ForceFieldToolKit::gui::consoleMessage "Bonded Optimization Finished"
    }

    # DONE
}
#======================================================


#------------------------------------------------------
# GenDihScan Specific
#------------------------------------------------------
#======================================================
proc ::ForceFieldToolKit::gui::gdsToggleLabels {} {
    # toggles atom labels for TOP molecule

    variable gdsAtomLabels

    if { [llength $gdsAtomLabels] > 0 } {
        foreach label $gdsAtomLabels {graphics top delete $label}
        set gdsAtomLabels {}
    } else {
        draw color lime
        foreach ind [[atomselect top all] get index] {
            set sel [atomselect top "index $ind"]
            lappend gdsAtomLabels [draw text [join [$sel get {x y z}]] $ind size 3]
            $sel delete
        }
    }
}
#======================================================
proc ::ForceFieldToolKit::gui::gdsImportDihedrals { psf pdb parfile } {
    # reads in a molecule and parameter file
    # if the molecule contains dihedrals that are defined in the parfile
    # returns the indices

    # validation
    set errorList {}
    set errorText ""

    if { $psf eq "" || ![file exists $psf] } { lappend errorList "Cannot find PSF file." }
    if { $pdb eq "" || ![file exists $pdb] } { lappend errorList "Cannot find PDB file." }
    if { $parfile eq "" || ![file exists $parfile] } { lappend errorList "Cannot find parameter file." }

    if { $errorList > 0 } {
        foreach ele $errorList {
            set errorText [concat $errorText\n$ele]
        }
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail $errorText
        return
    }


    # read the parameter file and parse out the dihedrals section
    set dihPars [lindex [::ForceFieldToolKit::SharedFcns::readParFile $parfile] 2]
    # build a 1D search index of unique type defs
    set dihTypeIndex {}
    foreach dih $dihPars {
        lappend dihTypeIndex [lindex $dih 0]
    }
    set dihTypeIndex [lsort -unique $dihTypeIndex]


    # load the molecule
    mol new $psf; mol addfile $pdb
    # retype from psf (can be removed once VMD psf reader is fixed to support CGenFF-styled types)
    # reTypeFromPSF has been depreciated
    #::ForceFieldToolKit::SharedFcns::reTypeFromPSF $psf top

    # grab the indices for all dihedrals (parse out only index information)
    set indDefList {}
    foreach entry [topo getdihedrallist] {
        lappend indDefList [lrange $entry 1 4]
    }

    # check if dihedral contains a linear bend, which shouldn't be scanned (or fit)
    set lbThresh 175.0
    # cycle through the dihedral list backwards (we'll be removing things based on position index)
    for {set i [expr {[llength $indDefList] - 1}]} {$i >= 0} {incr i -1} {
        if { [measure angle [lrange [lindex $indDefList $i] 0 2]] >= $lbThresh || [measure angle [lrange [lindex $indDefList $i] 1 3]] >= $lbThresh } {
            # remove the dihedral from the list
            set indDefList [lreplace $indDefList $i $i]
        }
    }

    # convert the index def list to type def list and element def list
    set typeDefList {}
    set eleDefList {}
    # cycle through each dihedral
    foreach dih $indDefList {
        set typeDef {}
        set eleDef {}
        # cycle through each index
        foreach ind $dih {
            set temp [atomselect top "index $ind"]
            lappend typeDef [$temp get type]
            lappend eleDef [$temp get element]
            $temp delete
        }
        # write the full typeDef to the list
        lappend typeDefList $typeDef
        lappend eleDefList $eleDef
    }


    # cycle through the typeDefLst
    # if the typeDef is in the dihTypeIndex (from par file),
    # then grab the index definition and measure the dihedral
    set unqCentralBondInds {}
    set returnData {}
    for {set i 0} {$i < [llength $typeDefList]} {incr i} {
        if { [lsearch -exact $dihTypeIndex [lindex $typeDefList $i]] != -1 || \
             [lsearch -exact $dihTypeIndex [lreverse [lindex $typeDefList $i]]] != -1 } {

                 # check to see if either end index is a hydrogen
                 if { [lindex $eleDefList $i 0] == "H" || [lindex $eleDefList $i 3] == "H" } { continue }

                 # check to see if the central bond is a duplicate (repeat scan)
                 set bondInds [lsort -increasing [lrange [lindex $indDefList $i] 1 2]]
                 if { [lsearch $unqCentralBondInds $bondInds] != -1 } { continue } else { lappend unqCentralBondInds $bondInds }

                 # append the data to return
                 lappend returnData [lindex $indDefList $i]

        } else {
            continue
        }
    }


    # clean up
    mol delete top

    # return the data
    return $returnData

}
#======================================================
proc ::ForceFieldToolKit::gui::gdsShowSelRep {} {
    # shows a representation of the selected tv entries in VMD
    # via a CPK representation

    # build a list of indices to include in the representation
    # based on the tv selection
    set indexList {}
    foreach ele [.fftk_gui.hlf.nb.genDihScan.dihs2scan.tv selection] {
        foreach ind [.fftk_gui.hlf.nb.genDihScan.dihs2scan.tv set $ele indDef] {
            lappend indexList $ind
        }
    }

    # build a list of rep names for the top molecule
    set currRepNames {}
    for {set i 0} {$i < [molinfo top get numreps]} {incr i} {
        lappend currRepNames [mol repname top $i]
    }

    # determine if there is already a representation in place
    # and if that rep still exists (i.e., the user hasn't deleted it)
    if { $::ForceFieldToolKit::gui::gdsRepName eq "" || [lsearch $currRepNames $::ForceFieldToolKit::gui::gdsRepName] == -1 } {
        # we need a new rep
        mol selection "index $indexList"
        mol representation CPK
        mol color Name
        mol addrep top

        set ::ForceFieldToolKit::gui::gdsRepName [mol repname top [expr {[molinfo top get numreps]-1}]]

    } else {
        # update the old rep
        set currRepId [mol repindex top $::ForceFieldToolKit::gui::gdsRepName]
        mol modselect $currRepId top "index $indexList"
    }
}
#======================================================


#------------------------------------------------------
# DihedralOpt Specific
#------------------------------------------------------
proc ::ForceFieldToolKit::gui::doptRunOpt {} {
    # procedure for button to run the dihedral optimization

    # initialize/reset some variables that will explicitely set by GUI
    set ::ForceFieldToolKit::DihOpt::parlist {}
    set ::ForceFieldToolKit::DihOpt::GlogFiles {}
    set ::ForceFieldToolKit::DihOpt::parDataInput {}
###    array unset ::ForceFieldToolKit::DihOpt::boundsInfo; array set ::ForceFieldToolKit::DihOpt::boundsInfo {}

    # build the parameter files list (parlist) from the TV box
    foreach tvItem [.fftk_gui.hlf.nb.dihopt.input.parFiles children {}] {
        lappend ::ForceFieldToolKit::DihOpt::parlist [lindex [.fftk_gui.hlf.nb.dihopt.input.parFiles item $tvItem -values] 0]
    }

    # build the gaussian log files list (qm target data) from the TV box
    foreach tvItem [.fftk_gui.hlf.nb.dihopt.qmt.tv children {}] {
        lappend ::ForceFieldToolKit::DihOpt::GlogFiles [.fftk_gui.hlf.nb.dihopt.qmt.tv item $tvItem -values]
    }

    # build the parameter data input list
    # requires the form:
    # {
    #   {typedef} {k mult delta}
    # }
    foreach tvItem [.fftk_gui.hlf.nb.dihopt.parSet.tv children {}] {
        set dihPars [.fftk_gui.hlf.nb.dihopt.parSet.tv item $tvItem -values]
        # set typeDef [lindex $dihPars 0]
        set typeDef {}
        foreach atype [lindex $dihPars 0] { lappend typeDef $atype }
        set fc [lindex $dihPars 1]
        set mult [lindex $dihPars 2]
        set delta [lindex $dihPars 3]
        set lock [lindex $dihPars 4]
        # if phase shift is 180, flip the sign of k and reset delta to 0
        # append lock and sign info to the boundsInfo array
#        if { $delta == 180 } {
#            set fc [expr {-1*$fc}]
#            set delta 0
##            set ::ForceFieldToolKit::DihOpt::boundsInfo($typeDef) [list $lock 180]
##        } else {
##            set ::ForceFieldToolKit::DihOpt::boundsInfo($typeDef) [list $lock 0]
#        }
##        lappend ::ForceFieldToolKit::DihOpt::boundsInfo $lock
        if { [string equal $lock "yes"] } {
           lappend ::ForceFieldToolKit::DihOpt::parDataInput [list $typeDef [list $fc $mult $delta 1]]
        } else {
           lappend ::ForceFieldToolKit::DihOpt::parDataInput [list $typeDef [list $fc $mult $delta 0]]
        }
    }

    if { $::ForceFieldToolKit::gui::doptBuildScript } {
        # build a script instead of running directly
        set ::ForceFieldToolKit::gui::doptStatus "Writing to script..."
        update idletasks
        ::ForceFieldToolKit::DihOpt::buildScript [file dirname $::ForceFieldToolKit::DihOpt::outFileName]/DihOptScript.tcl
        #puts "the build script function is not currently implemented"
        set ::ForceFieldToolKit::gui::doptStatus "IDLE"
        ::ForceFieldToolKit::gui::consoleMessage "Dihedral optimization run script written"
    } else {
        # run optimization directly
        set ::ForceFieldToolKit::gui::doptStatus "Running..."
        ::ForceFieldToolKit::gui::consoleMessage "Dihedral optimization started"
        update idletasks
        set finalOptData [::ForceFieldToolKit::DihOpt::optimize]
        if { $finalOptData == -1 } {
            set ::ForceFieldToolKit::gui::doptStatus "Halted on ERROR"
            ::ForceFieldToolKit::gui::consoleMessage "Dihedral optimization halted on error"
            update idletasks
            return
        }
        set ::ForceFieldToolKit::gui::doptStatus "Loading Results..."
        update idletasks

        # test QME, MMEi, and dihAll; update status labels in Vis. Results accordingly
        if { [llength $::ForceFieldToolKit::DihOpt::EnQM] != 0 } {
            set ::ForceFieldToolKit::gui::doptQMEStatus "Loaded"
        } else {
            set ::ForceFieldToolKit::gui::doptQMEStatus "ERROR"
        }
        if { [llength $::ForceFieldToolKit::DihOpt::EnMM] != 0 } {
            set ::ForceFieldToolKit::gui::doptMMEStatus "Loaded"
        } else {
            set ::ForceFieldToolKit::gui::doptMMEStatus "ERROR"
        }
        if { [llength $::ForceFieldToolKit::DihOpt::dihAllData] != 0 } {
            set ::ForceFieldToolKit::gui::doptDihAllStatus "Loaded"
        } else {
            set ::ForceFieldToolKit::gui::doptDihAllStatus "ERROR"
        }
        update idletasks

        # clear the Vis. Results treeview
        .fftk_gui.hlf.nb.dihopt.results.data.tv delete [.fftk_gui.hlf.nb.dihopt.results.data.tv children {}]
        set ::ForceFieldToolKit::gui::doptRefineCount 1
        # add the data to the Vis. Results treeview
        .fftk_gui.hlf.nb.dihopt.results.data.tv insert {} end -values [list "orig" [format "%.3f" [lindex $finalOptData 0]] "blue" [lindex $finalOptData 1] [lindex $finalOptData 2]]

        # clear and then build the refinement parameter definitions from the final values
#        .fftk_gui.hlf.nb.dihopt.refine.parSet.tv delete [.fftk_gui.hlf.nb.dihopt.refine.parSet.tv children {}]
#        set finalParList [lindex $finalOptData 2]
#        foreach ele $finalParList {
#            set typedef [lrange $ele 0 3]
#            set k [lindex $ele 4]
#            set mult [lindex $ele 5]
#            set delta [lindex $ele 6]
#            .fftk_gui.hlf.nb.dihopt.refine.parSet.tv insert {} end -values [list $typedef $k $mult $delta]
#        }

        # update the status label
        set ::ForceFieldToolKit::gui::doptStatus "IDLE"
        ::ForceFieldToolKit::gui::consoleMessage "Dihedral optimization finished"
        update idletasks

    }

    # DONE

}
#======================================================
proc ::ForceFieldToolKit::gui::doptRunManualRefine {} {
    # Preps, runs, and processes MM PES Refinement Parameters

    # Sanity Check
    # check that there are parameters
    set errorList {}
    if { [llength [.fftk_gui.hlf.nb.dihopt.refine.parSet.tv children {}]] == 0 } { lappend errorList "No parameters provided for refinment/refitting." }
    if { ![file exists $::ForceFieldToolKit::DihOpt::psf] } { lappend errorList "PSF not found." }
    if { ![file exists $::ForceFieldToolKit::DihOpt::pdb] } { lappend errorList "PDB not found." }
    if { ![info exists ::ForceFieldToolKit::DihOpt::dihAllData] } { lappend errorList "Dihedral Data is missing.  Rerun the initial optimization."}
    if { ![info exists ::ForceFieldToolKit::DihOpt::EnQM] } { lappend errorList "QM PES data is missing.  Rerun the initial optimization."}
    if { ![info exists ::ForceFieldToolKit::DihOpt::EnMM] } { lappend errorList "MMEi PES data is missing.  Rerun the initial optimization."}
    if { ![info exists ::ForceFieldToolKit::DihOpt::refineCutoff] || ![string is double $::ForceFieldToolKit::DihOpt::refineCutoff ]} { lappend errorList "Inappropriate refinement cutoff energy." }

    set errorText ""
    if { [llength $errorList] > 0 } {
        foreach ele $errorList {
            set errorText [concat $errorText\n$ele]
        }
        tk_messageBox \
            -type ok \
            -icon warning \
            -message "Application halting due to the following errors:" \
            -detail $errorText

        # there are errors, return the error response
        return 0
    }

    # build the input
    set parDataSave {}
    set parData {}
    foreach tvItem [.fftk_gui.hlf.nb.dihopt.refine.parSet.tv children {}] {
        set dihPars [.fftk_gui.hlf.nb.dihopt.refine.parSet.tv item $tvItem -values]
        lappend parDataSave [join $dihPars]
        #set typeDef [lindex $dihPars 0]
        set typeDef {}
        foreach atype [lindex $dihPars 0] { lappend typeDef $atype }
        set fc [lindex $dihPars 1]
        set mult [lindex $dihPars 2]
        set delta [lindex $dihPars 3]
        set lock [lindex $dihPars 4]
        # if phase shift is 180, flip the sign of k and reset delta to 0
        # append lock and sign info to the boundsInfo array
        if { [string equal $lock "yes"] } {
           lappend parData [list $typeDef [list $fc $mult $delta 1]]
        } else {
           lappend parData [list $typeDef [list $fc $mult $delta 0]]
        }
    }

    # call the function to compute {RMSE EnMMf}
    set result [::ForceFieldToolKit::DihOpt::manualRefinementCalculation $parData]

    # add the result to the results box
    #                                                                    label                                                         RMSE                              color   EnMMf data        parameters
    .fftk_gui.hlf.nb.dihopt.results.data.tv insert {} end -values [list "r[format "%02d" $::ForceFieldToolKit::gui::doptRefineCount]" [format "%.3f" [lindex $result 0]] "blue" [lindex $result 1] $parDataSave]
    incr ::ForceFieldToolKit::gui::doptRefineCount
}
#======================================================
proc ::ForceFieldToolKit::gui::doptRunRefine {} {
    # procedure for button to run the dihedral refinement/refitting

    # initialize some variables that will be explicitely set by GUI
    set ::ForceFieldToolKit::DihOpt::refineParDataInput {}
###    array unset ::ForceFieldToolKit::DihOpt::boundsInfo; array set ::ForceFieldToolKit::DihOpt::boundsInfo {}

    # build the parameter data input list
    # requires the form:
    # {
    #   {typedef} {k mult delta}
    # }
    foreach tvItem [.fftk_gui.hlf.nb.dihopt.refine.parSet.tv children {}] {
        set dihPars [.fftk_gui.hlf.nb.dihopt.refine.parSet.tv item $tvItem -values]
        #set typeDef [lindex $dihPars 0]
        set typeDef {}
        foreach atype [lindex $dihPars 0] { lappend typeDef $atype }
        set fc [lindex $dihPars 1]
        set mult [lindex $dihPars 2]
        set delta [lindex $dihPars 3]
        set lock [lindex $dihPars 4]
        # if phase shift is 180, flip the sign of k and reset delta to 0
        # append lock and sign info to the boundsInfo array
#        if { $delta == 180 } {
#            set fc [expr {-1*$fc}]
#            set delta 0
##            set ::ForceFieldToolKit::DihOpt::boundsInfo($typeDef) [list $lock 180]
##        } else {
##            set ::ForceFieldToolKit::DihOpt::boundsInfo($typeDef) [list $lock 0]
#        }
##        lappend ::ForceFieldToolKit::DihOpt::boundsInfo $lock
        if { [string equal $lock "yes"] } {
           lappend ::ForceFieldToolKit::DihOpt::refineParDataInput [list $typeDef [list $fc $mult $delta 1]]
        } else {
           lappend ::ForceFieldToolKit::DihOpt::refineParDataInput [list $typeDef [list $fc $mult $delta 0]]
        }
    }

    # launch the refinement
    set ::ForceFieldToolKit::gui::doptStatus "Running..."
    ::ForceFieldToolKit::gui::consoleMessage "Dihedral refinement started"
    update idletasks
    set finalRefineData [::ForceFieldToolKit::DihOpt::refine]
    if { $finalRefineData == -1 } {
        set ::ForceFieldToolKit::gui::doptStatus "Halted on ERROR"
        ::ForceFieldToolKit::gui::consoleMessage "Dihedral refinement halted on error"
        update idletasks
        return
    }

    set ::ForceFieldToolKit::gui::doptStatus "Loading Results..."
    update idletasks

    # add the data to the Viz. Results treeview
    .fftk_gui.hlf.nb.dihopt.results.data.tv insert {} end -values [list "r[format "%02d" $::ForceFieldToolKit::gui::doptRefineCount]" [format "%.3f" [lindex $finalRefineData 0]] "blue" [lindex $finalRefineData 1] [lindex $finalRefineData 2]]
    incr ::ForceFieldToolKit::gui::doptRefineCount

    # update the status label
    set ::ForceFieldToolKit::gui::doptStatus "IDLE"
    ::ForceFieldToolKit::gui::consoleMessage "Dihedral refinement finished"
    update idletasks

    # DONE

}
#======================================================
proc ::ForceFieldToolKit::gui::doptSetColor {} {
    # sets the color for all selected data sets
    # very simple program, but simplifies menu construction

    foreach dataId [.fftk_gui.hlf.nb.dihopt.results.data.tv selection] {
        .fftk_gui.hlf.nb.dihopt.results.data.tv set $dataId color $::ForceFieldToolKit::gui::doptEditColor
    }
}
#======================================================
proc ::ForceFieldToolKit::gui::doptBuildPlotWin {} {
    # builds the window for plotting the results data

    # localize variables
    variable doptP
    variable doptResultsPlotHandle

    # if the multiplot window already exists, then deiconify it
    if { [winfo exists .pte_plot] } {
        wm deiconify .pte_plot
        return
    }

    # build the window
    set doptP [toplevel ".pte_plot"]
    wm title $doptP "Plot DihOpt Results"

    # allow window to expand with .
    grid columnconfigure $doptP 0 -weight 1
    grid rowconfigure $doptP 0 -weight 1

    # set a default initial geometry
    wm geometry $doptP 700x580

    # build/grid a frame to hold the embedded multiplot
    ttk::frame $doptP.plotFrame
    grid $doptP.plotFrame -column 0 -row 0 -sticky nswe

    # build the multiplot
    set doptResultsPlotHandle [multiplot embed $doptP.plotFrame \
        -title "Selected DihOpt Fit Data" -xlabel "Conformation" -ylabel "Energy\n(kcal/mol)" \
        -xsize 680 -ysize 450 -ymin 0 -ymax 10 -xmin auto -xmax auto \
        -lines -linewidth 3]

    # build control panel
    ttk::frame $doptP.controls
    grid $doptP.controls -column 0 -row 1 -sticky nswe
    grid columnconfigure $doptP.controls 0 -weight 1

    # build/grid a separator
    ttk::separator $doptP.controls.sep0 -orient horizontal
    grid $doptP.controls.sep0 -column 0 -row 0 -sticky nswe -pady 2

    # build sliders
    ttk::frame $doptP.controls.sliders
        ttk::label $doptP.controls.sliders.xMinLbl -text "x-min" -anchor center
        ttk::scale $doptP.controls.sliders.xMin -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::gui::doptAdjustScale xmin }
        ttk::label $doptP.controls.sliders.xMaxLbl -text "x-max" -anchor center
        ttk::scale $doptP.controls.sliders.xMax -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::gui::doptAdjustScale xmax }
        ttk::label $doptP.controls.sliders.yMinLbl -text "y-min" -anchor center
        ttk::scale $doptP.controls.sliders.yMin -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::gui::doptAdjustScale ymin }
        ttk::label $doptP.controls.sliders.yMaxLbl -text "y-max" -anchor center
        ttk::scale $doptP.controls.sliders.yMax -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::gui::doptAdjustScale ymax }

    # grid sliders
    grid $doptP.controls.sliders -column 0 -row 1 -sticky nswe -padx "5 5"
        grid $doptP.controls.sliders.xMinLbl -column 0 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.sliders.xMin -column 1 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.sliders.xMaxLbl -column 0 -row 1 -sticky nswe -padx "4 0"
        grid $doptP.controls.sliders.xMax -column 1 -row 1 -sticky nswe -padx "4 0"

        grid $doptP.controls.sliders.yMinLbl -column 2 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.sliders.yMin -column 3 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.sliders.yMaxLbl -column 2 -row 1 -sticky nswe -padx "4 0"
        grid $doptP.controls.sliders.yMax -column 3 -row 1 -sticky nswe -padx "4 0"

    # configure slider controls
    grid columnconfigure $doptP.controls.sliders {1 3} -weight 1

    # build/grid a separator
    ttk::separator $doptP.controls.sep1 -orient horizontal
    grid $doptP.controls.sep1 -column 0 -row 2 -sticky nswe -pady 2

    # build xy controls
    ttk::frame $doptP.controls.xySet
        ttk::label $doptP.controls.xySet.xMinLbl -text "x-min" -anchor center
        ttk::entry $doptP.controls.xySet.xMin -textvariable ::ForceFieldToolKit::gui::doptScaleXmin -width 4 -justify center
        ttk::label $doptP.controls.xySet.xMaxLbl -text "x-max" -anchor center
        ttk::entry $doptP.controls.xySet.xMax -textvariable ::ForceFieldToolKit::gui::doptScaleXmax -width 4 -justify center
        ttk::label $doptP.controls.xySet.yMinLbl -text "y-min" -anchor center
        ttk::entry $doptP.controls.xySet.yMin -textvariable ::ForceFieldToolKit::gui::doptScaleYmin -width 4 -justify center
        ttk::label $doptP.controls.xySet.yMaxLbl -text "y-max" -anchor center
        ttk::entry $doptP.controls.xySet.yMax -textvariable ::ForceFieldToolKit::gui::doptScaleYmax -width 4 -justify center
        ttk::button $doptP.controls.xySet.set -text "Set Axis" -command {
            # update plot axes
            $::ForceFieldToolKit::gui::doptResultsPlotHandle configure \
                -xmin $::ForceFieldToolKit::gui::doptScaleXmin \
                -xmax $::ForceFieldToolKit::gui::doptScaleXmax \
                -ymin $::ForceFieldToolKit::gui::doptScaleYmin \
                -ymax $::ForceFieldToolKit::gui::doptScaleYmax
            $::ForceFieldToolKit::gui::doptResultsPlotHandle replot

            # update slider positions
            #if { $::ForceFieldToolKit::gui::doptScaleXmin ne "auto" } {.pte_plot.controls.sliders.xMin configure -value $::ForceFieldToolKit::gui::doptScaleXmin}
            #if { $::ForceFieldToolKit::gui::doptScaleXmax ne "auto" } {.pte_plot.controls.sliders.xMax configure -value $::ForceFieldToolKit::gui::doptScaleXmax}
            #if { $::ForceFieldToolKit::gui::doptScaleYmin ne "auto" } {.pte_plot.controls.sliders.yMin configure -value $::ForceFieldToolKit::gui::doptScaleYmin}
            #if { $::ForceFieldToolKit::gui::doptScaleYmax ne "auto" } {.pte_plot.controls.sliders.yMax configure -value $::ForceFieldToolKit::gui::doptScaleYmax}
        }

    # grid xy controls
    grid $doptP.controls.xySet -column 0 -row 3 -sticky ns -padx "5 5"
        grid $doptP.controls.xySet.xMinLbl -column 0 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.xMin -column 1 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.xMaxLbl -column 2 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.xMax -column 3 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.yMinLbl -column 4 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.yMin -column 5 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.yMaxLbl -column 6 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.yMax -column 7 -row 0 -sticky nswe -padx "4 0"
        grid $doptP.controls.xySet.set -column 8 -row 0 -sticky nswe -padx "4 0"

    # configure xy controls
    # nothing much to do here (placeholder)

    # build/grid a separator
    ttk::separator $doptP.controls.sep2 -orient horizontal
    grid $doptP.controls.sep2 -column 0 -row 4 -sticky nswe -pady 2

    # initialize the xy controls
    set ::ForceFieldToolKit::gui::doptScaleXmin "auto"
    set ::ForceFieldToolKit::gui::doptScaleXmax "auto"
    set ::ForceFieldToolKit::gui::doptScaleYmin "auto"
    set ::ForceFieldToolKit::gui::doptScaleYmax "auto"

    # build a points option
    ttk::separator $doptP.controls.xySet.vsep0 -orient vertical
    ttk::checkbutton $doptP.controls.xySet.points -variable ::ForceFieldToolKit::gui::doptPlotPoints -text "Show Points" -onvalue 1 -offvalue 0 -command {
        for {set i 0} {$i<[$::ForceFieldToolKit::gui::doptResultsPlotHandle nsets]} {incr i} {
            if { $::ForceFieldToolKit::gui::doptPlotPoints } {
                $::ForceFieldToolKit::gui::doptResultsPlotHandle configure -set $i -marker circle -radius 2.5 -fillcolor white -plot
            } else {
                $::ForceFieldToolKit::gui::doptResultsPlotHandle configure -set $i -marker none -plot
            }
        }
    }

    # grid the points option
    grid $doptP.controls.xySet.vsep0 -column 9 -row 0 -sticky nswe -padx "4 0"
    grid $doptP.controls.xySet.points -column 10 -row 0 -sticky nswe -padx "4 0"

    # initialize the points option
    set ::ForceFieldToolKit::gui::doptPlotPoints 0

    # when the window is closed, clean up
    bind .pte_plot <Destroy> {
        #$::ForceFieldToolKit::gui::doptResultsPlotHandle quit
        set ::ForceFieldToolKit::gui::doptResultsPlotHandle {}
        set ::ForceFieldToolKit::gui::doptP {}
    }

    # return the window
    return $doptP

}
#======================================================
proc ::ForceFieldToolKit::gui::doptPlotData { datasets colorsets legend } {
    # plots input y-coordinate datasets in an embedded multiplot window

    # localize variable
    variable doptP
    variable doptResultsPlotHandle

    # clear the dataset
    $doptResultsPlotHandle clear

    # initialize axis scales
    set xMax 0
    set yMax 0

    # cycle through each dataset
    for {set i 0} {$i < [llength $datasets] } {incr i} {
        # parse the y data
        set ydata [lindex $datasets $i]

        # find/check yMax
        foreach yPoint $ydata { if {$yPoint > $yMax} {set yMax $yPoint} }

        # build the x data
        set xdata {}
        for {set x 0} {$x < [llength $ydata]} {incr x} {
            lappend xdata $x
        }

        # find/check xMax
        if { [llength $xdata] > $xMax } { set xMax [llength $xdata] }

        # parse out the plot color
        set plotColor [lindex $colorsets $i]

        # parse out the legend text
        set legendTxt [lindex $legend $i]

        $doptResultsPlotHandle add $xdata $ydata -lines -linewidth 3 -linecolor $plotColor -legend $legendTxt

    }

    # update the plot
    if { $yMax > 10 } { set initYmax 10} else { set initYmax $yMax }
    $doptResultsPlotHandle configure -xmin 0 -xmax $xMax -ymin 0 -ymax $initYmax
    $doptResultsPlotHandle replot

    # update the slider controls
    .pte_plot.controls.sliders.xMin configure -from 0 -to $xMax -value 0
    .pte_plot.controls.sliders.xMax configure -from 0 -to $xMax -value $xMax
    .pte_plot.controls.sliders.yMin configure -from 0 -to $yMax -value 0
    .pte_plot.controls.sliders.yMax configure -from 0 -to $yMax -value $initYmax

}
#======================================================
proc ::ForceFieldToolKit::gui::doptAdjustScale {scaleType value} {
    # controller for the dihedral plot axes

    # localize important variables
    variable doptP
    variable doptResultsPlotHandle

    # adjust plot axes
    $doptResultsPlotHandle configure -$scaleType $value
    $doptResultsPlotHandle replot


}
#======================================================
proc ::ForceFieldToolKit::gui::doptLogParser { logfile } {
    # parses optimization and refinement logs and loads
    # the relevant information into the gui

    # initialize lists
    set qme {}
    set psf {}
    set pdb {}
    set mme {}
    set dihAll {}
    set rmse {}
    set mmef {}
    set parOut {}

    # open the log file for reading
    set inFile [open $logfile r]

    set readstate 0

    # read through the file a line at a time
    while { ![eof $inFile] } {
        set inLine [gets $inFile]

        # outter switch determines if we're entering or exiting a section of interest
        # inner switch write data to the appropriate list, or continues on
        switch -exact $inLine {
            "QMDATA" { set readstate qm }
            "PSF" { set readstate psf }
            "PDB" { set readstate pdb }
            "MME" { set readstate mme }
            "MMdihARRAY" { set readstate dih }
            "FINAL RMSE" { set readstate rmse }
            "FINAL STEP ENERGIES" {
                set readstate mmef
                # burn a line
                gets $inFile
            }
            "FINAL PARAMETERS" { set readstate par }
            "END" { set readstate 0 }
            default {
                switch -exact $readstate {
                    "qm" { lappend qme [lindex $inLine 2]}
                    "psf" { set psf $inLine }
                    "pdb" { set pdb $inLine }
                    "mme" { lappend mme $inLine }
                    "dih" { lappend dihAll $inLine}
                    "rmse" { set rmse [format %.4f $inLine] }
                    "mmef" { lappend mmef [lindex $inLine 2] }
                    "par" { lappend parOut [concat [lindex $inLine 1] [lrange $inLine 2 end] "off"] }
                    default { continue }
                }
            }

        }
    }

    # close the input log file
    close $inFile

    # setup the GUI and appropriate namespace variables

    # qme and mme from log are raw and need to be normalized
    set ::ForceFieldToolKit::DihOpt::EnQM [::ForceFieldToolKit::DihOpt::renorm $qme]
    set ::ForceFieldToolKit::DihOpt::EnMM [::ForceFieldToolKit::DihOpt::renorm $mme]

    # dihAll, psf, and pdb value are all fine as-is
    set ::ForceFieldToolKit::DihOpt::dihAllData $dihAll
    set ::ForceFieldToolKit::DihOpt::psf $psf
    set ::ForceFieldToolKit::DihOpt::pdb $pdb

    # update labels
    if { $::ForceFieldToolKit::DihOpt::EnQM ne "" } {
        set ::ForceFieldToolKit::gui::doptQMEStatus "Loaded"
    } else {
        set ::ForceFieldToolKit::gui::doptQMEStatus "ERROR"
    }

    if { $::ForceFieldToolKit::DihOpt::EnMM ne "" } {
        set ::ForceFieldToolKit::gui::doptMMEStatus "Loaded"
    } else {
        set ::ForceFieldToolKit::gui::doptMMEStatus "ERROR"
    }

    if { $::ForceFieldToolKit::DihOpt::dihAllData ne "" } {
        set ::ForceFieldToolKit::gui::doptDihAllStatus "Loaded"
    } else {
        set ::ForceFieldToolKit::gui::doptDihAllStatus "ERROR"
    }

    # add rmse, mmef, and parOut to the results data box
    .fftk_gui.hlf.nb.dihopt.results.data.tv insert {} end -values [list "import" $rmse "blue" $mmef $parOut]

}
#======================================================
proc ::ForceFieldToolKit::gui::doptLogWriter { filename rmse mmef parData } {
    # writes a dihedral optimization -styled log file

    # basename is used for a filename
    # rmse, mmef, and parData are important components of a log file

    # open the log file for writing
    set outFile [open $filename w]

    # write a header
    puts $outFile "============================================================="
    puts $outFile "Log file written directly from GUI for refit/refinement"
    puts $outFile "It will contain all necessary information for additional"
    puts $outFile "refitting/refining, and updating parameters in BuildPar"
    puts $outFile "but it will look a little different than the initial log file"
    puts $outFile "============================================================="

    # write "QMDATA"
    # since only QME is read in and stored from logs, this is all we can write
    # but it must be formatted in a similar manner

    puts $outFile "\nQMDATA"
    foreach ele $::ForceFieldToolKit::DihOpt::EnQM {
        puts $outFile "placeHolder placeHolder $ele"
    }
    puts $outFile "END"

    # write the psf path
    puts $outFile "\nPSF"
    puts $outFile "$::ForceFieldToolKit::DihOpt::psf"
    puts $outFile "END"

    # write the pdb path
    puts $outFile "\nPDB"
    puts $outFile "$::ForceFieldToolKit::DihOpt::pdb"
    puts $outFile "END"

    # write MME (subsection of MMDATA)
    puts $outFile "\nMME"
    foreach ele $::ForceFieldToolKit::DihOpt::EnMM {
        puts $outFile "$ele"
    }
    puts $outFile "END"

    # write the dihAllData
    puts $outFile "\nMMdihARRAY"
    foreach ele $::ForceFieldToolKit::DihOpt::dihAllData {
        puts $outFile "$ele"
    }
    puts $outFile "END"

    # write the final rmse
    puts $outFile "\nFINAL RMSE"
    puts $outFile "$rmse"
    puts $outFile "END"

    # write the step energies (mmef)
    puts $outFile "\nFINAL STEP ENERGIES"
    puts $outFile "QME\tMME(i)\tMME(f)\tQME-MME(f)"
    foreach ele $mmef {
        puts $outFile "placeholder placeholder $ele placeholder"
    }
    puts $outFile "END"

    # write the final paramater data
    puts $outFile "\nFINAL PARAMETERS"
    foreach ele $parData {
        puts $outFile "dihedral [list [lrange $ele 0 3]] [lindex $ele 4] [lindex $ele 5] [lindex $ele 6]"
    }
    puts $outFile "END"

    # clean up
    close $outFile

}
#======================================================

#------------------------------------------------------
# Developer-Related Procedures
#------------------------------------------------------

#======================================================
#======================================================
