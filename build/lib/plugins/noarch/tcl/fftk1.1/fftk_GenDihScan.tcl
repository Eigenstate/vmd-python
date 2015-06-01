#
# $Id: fftk_GenDihScan.tcl,v 1.9 2014/08/26 17:51:14 mayne Exp $
#

#======================================================
namespace eval ::ForceFieldToolKit::GenDihScan {
    variable psf
    variable pdb
    variable outPath
    variable basename
    variable qmProc
    variable qmCharge
    variable qmMem
    variable qmMult
    variable qmRoute
    variable dihData
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::init {} {
    
    # localize variables
    variable psf
    variable pdb
    variable outPath
    variable basename
    variable qmProc
    variable qmCharge
    variable qmMem
    variable qmMult
    variable qmRoute
    
    variable dihData

    # set variables
    set psf {}
    set pdb {}
    set outPath {}
    set basename {}
    ::ForceFieldToolKit::GenDihScan::resetGaussianDefaults
    #set qmProc 1
    #set qmCharge 0
    #set qmMem 1
    #set qmMult 1
    #set qmRoute "# opt=modredundant MP2/6-31g(d)"
    
    set dihData {}
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::sanityCheck {} {
    # checks to see that the appropriate information is set prior to running
    
    # returns 1 if all input is sane
    # returns 0 if there is a problem
    
    # localize relevant variables
    variable psf
    variable pdb
    variable outPath
    variable basename
    variable qmProc
    variable qmCharge
    variable qmMem
    variable qmMult
    variable qmRoute
    variable dihData
    
    # local variables
    set errorList {}
    set errorText ""
    
    # checks
    # psf
    if { $psf eq "" } {
        lappend errorList "No PSF file was specified."
    } else {
        if { ![file exists $psf] } { lappend errorList "Cannot find PSF file." }
    }
    
    # pdb
    if { $pdb eq "" } {
        lappend errorList "No PDB file was specified."
    } else {
        if { ![file exists $pdb] } { lappend errorList "Cannot find PDB file." }
    }

    # make sure that output folder is specified and writable
    if { $outPath eq "" } {
        lappend errorList "No output path was specified."
    } else {
        if { ![file writable $outPath] } { lappend errorList "Cannot write to output path." }
    }
    
    # make sure that basename is not empty
    if { $basename eq "" } { lappend errorList "No basename was specified." }

    # validate dihData
    if { [llength $dihData] == 0 } {
        lappend errorList "No dihedrals were entered for scanning."
    } else {
        foreach ele $dihData {
            # dih indices
            if { [llength [lindex $ele 0]] != 4 } { lappend errorList "Found inappropriate dihedral definition." }
            foreach ind [lindex $ele 0] {
                if { $ind < 0 || ![string is integer $ind] } { lappend errorList "Found inappropriate dihedral index." }
            }
            # plus/minus
            if { ![string is double [lindex $ele 1]] } { lappend errorList "Found inappropriate dihedral +/- value." }
            # step size
            if { [lindex $ele 2] <= 0 || ![string is double [lindex $ele 2]] } { lappend errorList "Found inappropriate dihedral step size." }
        }
    }
    
    # validate gaussian settings (not particularly vigorous validation)
    # qmProc (processors)
    if { $qmProc eq "" } { lappend errorList "No processors were specified." }
    if { $qmProc <= 0 || $qmProc != [expr int($qmProc)] } { lappend errorList "Number of processors must be a positive integer." }
    # qmMem (memory)
    if { $qmMem eq "" } { lappend errorList "No memory was specified." }
    if { $qmMem <= 0 || $qmMem != [expr int($qmMem)]} { lappend errorList "Memory must be a postive integer." }
    # qmCharge (charge)
    if { $qmCharge eq "" } { lappend errorList "No charge was specified." }
    if { $qmCharge != [expr int($qmCharge)] } { lappend errorList "Charge must be an integer." }
    # qmMult (multiplicity)
    if { $qmMult eq "" } { lappend errorList "No multiplicity was specified." }
    if { $qmMult < 0 || $qmMult != [expr int($qmMult)] } { lappend errorList "Multiplicity must be zero or a positive integer." }
    # qmRoute (route card for gaussian; just make sure it isn't empty)
    if { $qmRoute eq "" } { lappend errorList "Route card is empty." }


    # if there is an error, tell the user about it
    # return -1 to tell the calling proc that there is a problem
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

    # if you've made it this far, there are no errors
    return 1
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::buildGaussianFiles {} {
    # builds gaussian input files for scanning dihedral angles
    
    # localize variables
    variable psf
    variable pdb
    variable outPath
    variable basename
    variable qmProc
    variable qmCharge
    variable qmMem
    variable qmMult
    variable qmRoute
    
    variable dihData
    
    # run an input sanity check
    if { ![::ForceFieldToolKit::GenDihScan::sanityCheck] } { return }

    # assign Gaussian atom names and gather x,y,z for output com file
    mol new $psf; mol addfile $pdb
    set Gnames {}
    set atom_info {}
    for {set i 0} {$i < [molinfo top get numatoms]} {incr i} {
        set temp [atomselect top "index $i"]
        lappend atom_info [list [$temp get element][expr $i+1] [$temp get x] [$temp get y] [$temp get z]]
        lappend Gnames [$temp get element][expr $i+1]
        $temp delete
    }
    
    # cycle through each dihedral to scan
    set scanCount 1
    foreach dih $dihData {
        # change 0-based indices to 1-based
        set zeroInds [lindex $dih 0]
        set oneInds {}
        foreach ind $zeroInds {
            lappend oneInds [expr {$ind + 1}]
        }
        
        # negative scan
        # open the output file
        set outfile [open ${outPath}/${basename}.scan${scanCount}.neg.gau w]
        
        # write the header
        puts $outfile "%chk=${basename}.scan${scanCount}.neg.chk"
        puts $outfile "%nproc=$qmProc"
        puts $outfile "%mem=${qmMem}GB"
        puts $outfile "$qmRoute"
        puts $outfile ""
        puts $outfile "$basename Dihedral Scan at MP2/6-31G*"
        puts $outfile ""
        puts $outfile "$qmCharge $qmMult"
        # write coords
       foreach atom_entry $atom_info {
           puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
       }
       # write scan
       puts $outfile ""
       puts $outfile "D $oneInds S [expr int([expr [lindex $dih 1]/[lindex $dih 2]])] [format "%.6f" [expr {-1*[lindex $dih 2]}]]"
       
       close $outfile
       
       # positive scan
        # open the output file
        set outfile [open ${outPath}/${basename}.scan${scanCount}.pos.gau w]
        
        # write the header
        puts $outfile "%chk=${basename}.scan${scanCount}.pos.chk"
        puts $outfile "%nproc=$qmProc"
        puts $outfile "%mem=${qmMem}GB"
        puts $outfile "$qmRoute"
        puts $outfile ""
        puts $outfile "$basename Dihedral Scan at MP2/6-31G*"
        puts $outfile ""
        puts $outfile "$qmCharge $qmMult"
        # write coords
       foreach atom_entry $atom_info {
           puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
       }
       # write scan
       puts $outfile ""
       puts $outfile "D $oneInds S [expr int([expr [lindex $dih 1]/[lindex $dih 2]])] [format "%.6f" [lindex $dih 2]]"
       
       close $outfile    
       
       incr scanCount
        
    }
    
    # clean up
    mol delete top
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::resetGaussianDefaults {} {
    # resets gaussian settings to the default values

    # localize variables
    variable qmProc
    variable qmCharge
    variable qmMem
    variable qmMult
    variable qmRoute

    # set variables
    set qmProc 1
    set qmCharge 0
    set qmMem 1
    set qmMult 1
    set qmRoute "# opt=modredundant MP2/6-31g(d) Geom=PrintInputOrient"
}
#======================================================
# TORSION EXPLORER
#======================================================
namespace eval ::ForceFieldToolKit::GenDihScan::TorExplor {
    # namespace variables

    # GUI-related
    variable w
    variable psfType { {{PSF Files} {.psf}} {{All Files} *} }
    variable pdbType { {{PDB Files} {.pdb}} {{All Files} *} }
    variable logType { {{LOG Files} {.log}} {{All Files} *} }

    # Input-related
    foreach ele {psf pdb logs} { variable $ele "" }
    variable ptSize 4
    variable lnSize 2

    # Plot-related
    variable plothandle
    foreach ele {xmin xmax ymin ymax} { variable $ele "auto" }
    variable plotAutoscaling 1
    variable plotData
    variable plotMax
    
    # Color
    variable colorList { red orange green blue magenta purple \
                         pink orange3 lime cyan2 magenta2 mauve \
                         red2 orange2 green2 iceblue blue2 violet2 \
                         red3 green3 cyan cyan3 blue3 gray black }
    variable colorInd 0
    variable colorIdMap
    array unset colorIdMap
    array set colorIdMap { red 1 orange 3 green 7 blue 0 magenta 27 purple 11 \
                           pink 9 orange3 32 lime 12 cyan2 21 magenta2 28 mauve 13 \
                           red2 29 orange2 31 green2 19 iceblue 15 blue2 23 violet2 26 \
                           red3 30 green3 20 cyan 10 cyan3 22 blue3 24 gray 2 black 16 }

    # VMD-related
    variable molid -1
    variable scanIndsArr
    variable scanIndsPrevFrame ""
    variable repNameScanInds; # unique, non-changing
    variable repNameDihAnalysis ""; # unique, non-changing


    # Analysis-related
    variable dihAnalysisArr
    variable updateDihData 0
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::launchGUI {} {

    # style setup
    set vbuttonPadX 5; # vertically aligned std buttons
    set vbuttonPadY 0 
    set hbuttonPadX "5 0"; # horzontally aligned std buttons
    set hbuttonPadY 0
    set entryPadX 0; # single line entry
    set entryPadY 0
    set hsepPadX 10; # horizontal separators
    set hsepPadY 10
    set vsepPadX 0; # vertical separators
    set vsepPadY 0
    set labelFramePadX 0; # label frames
    set labelFramePadY "10 0"
    set labelFrameInternalPadding 5

    # special characters
    set downPoint \u25BC
    set rightPoint \u25B6
    # accept and cancel indicators
    set accept \u2713
    set cancel \u2715
    # motion indicators
    set upArrow \u2191
    set downArrow \u2193
    # other common symbols
    set ff \uFB00
    set plusMinus \u00B1
    set degree \u00B0
    set theta \u03B8
    set sub0 \u2080

    # setup the theme depending on what is available
    # SHOULD BE SET BY MAIN FFTK WINDOW
    #set themeList [ttk::style theme names]
    #if { [lsearch -exact $themeList "aqua"] != -1 } {
    #    ttk::style theme use aqua
    #    set placeHolderPadX 18
    #} elseif { [lsearch -exact $themeList "clam"] != -1 } {
    #    ttk::style theme use clam
    #} elseif { [lsearch -exact $themeList "classic"] != -1 } {
    #    ttk::style theme use classic
    #} else {
    #    ttk::style theme use default
    #}


    # BEGIN WINDOW SETUP
    variable w

    if { [winfo exists .torexplor] } {
        wm deiconify .torexplor
        return
    }
    set w [toplevel ".torexplor"]
    wm title $w "Torsion Explorer"

    # allow to expland with .
    grid columnconfigure $w 0 -weight 1
    grid rowconfigure    $w 0 -weight 1

    # default geometry
    wm geometry $w 1400x700

    # clean up after yourself when the gui is destroyed
    bind $w <Destroy> { 
        # delete the molecule, if we loaded anything and it still exists
        if { $::ForceFieldToolKit::GenDihScan::TorExplor::molid != -1 && [lsearch [molinfo list] $::ForceFieldToolKit::GenDihScan::TorExplor::molid] != -1 } {
                mol delete $::ForceFieldToolKit::GenDihScan::TorExplor::molid
            }
        
        # deleting the molecule should trigger the molidCleanupCallback, which deregisters both itself and the frameChangeCallback
        # but to be super cautious, let's search for them and deregister them if they are found
        # molidCleanUpCallback
        set traceCount [llength [lsearch -index 1 -all -regexp [trace info variable vmd_frame] "::ForceFieldToolKit::GenDihScan::TorExplor::molidCleanupCallback"]]
        for {set i 0} {$i < $traceCount} {incr i} { trace remove variable vmd_frame write ::ForceFieldToolKit::GenDihScan::TorExplor::molidCleanupCallback }
        # frameChangeCallback
        set traceCount [llength [lsearch -index 1 -all -regexp [trace info variable vmd_frame] "::ForceFieldToolKit::GenDihScan::TorExplor::frameChangeCallback"]]
        for {set i 0} {$i < $traceCount} {incr i} { trace remove variable vmd_frame write ::ForceFieldToolKit::GenDihScan::TorExplor::frameChangeCallback }
    }

    # build a high level frame (hlf)
    ttk::frame $w.hlf
    grid $w.hlf -column 0 -row 0 -sticky nswe

    grid columnconfigure $w.hlf 0 -weight 1
    grid rowconfigure    $w.hlf 0 -weight 1


    # input files frame
    # BUILD
    ttk::frame $w.hlf.inp

    ttk::label $w.hlf.inp.psflbl -text "PSF File:"
    ttk::entry $w.hlf.inp.psfentry -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::psf -width 40
    ttk::button $w.hlf.inp.psfbrowse -text "Browse" \
        -command {
            set tempfile [tk_getOpenFile -title "Select a PSF File" -filetypes $::ForceFieldToolKit::GenDihScan::TorExplor::psfType]
            if {![string eq $tempfile ""]} { set ::ForceFieldToolKit::GenDihScan::TorExplor::psf $tempfile }        
        }

    ttk::label $w.hlf.inp.pdblbl -text "PDB File:"
    ttk::entry $w.hlf.inp.pdbentry -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::pdb -width 40
    ttk::button $w.hlf.inp.pdbbrowse -text "Browse" -command {
            set tempfile [tk_getOpenFile -title "Select a PDB File" -filetypes $::ForceFieldToolKit::GenDihScan::TorExplor::pdbType]
            if {![string eq $tempfile ""]} { set ::ForceFieldToolKit::GenDihScan::TorExplor::pdb $tempfile }        
        }

    ttk::label $w.hlf.inp.loglbl -text "Log Files:"
    ttk::treeview $w.hlf.inp.logtv \
        -selectmode browse \
        -columns {short long} \
        -displaycolumns {short} \
        -show {} \
        -height 10 \
        -yscrollcommand "$w.hlf.inp.logtvScroll set"
    ttk::scrollbar $w.hlf.inp.logtvScroll -orient vertical -command "$w.hlf.inp.logtv yview"

    ttk::frame  $w.hlf.inp.logbuttons
    ttk::button $w.hlf.inp.logbuttons.add -text "Add" -command {
        set tempfiles [tk_getOpenFile -title "Select Gaussian Dihedral Scan LOG File(s)" -multiple 1 -filetypes $::ForceFieldToolKit::GenDihScan::TorExplor::logType]
        foreach tempfile $tempfiles {
            if {![string eq $tempfile ""]} { .torexplor.hlf.inp.logtv insert {} end -values [list [file tail $tempfile] $tempfile] }
        }
    }
    ttk::button $w.hlf.inp.logbuttons.del -text "Delete" -command { .torexplor.hlf.inp.logtv delete [.torexplor.hlf.inp.logtv selection] }
    ttk::button $w.hlf.inp.logbuttons.clear -text "Clear" -command { .torexplor.hlf.inp.logtv delete [.torexplor.hlf.inp.logtv children {}] }

    ttk::separator $w.hlf.inp.logbuttons.sep -orient horizontal
    ttk::button $w.hlf.inp.logbuttons.load -text "   Load   " \
        -command {
            .torexplor.hlf.inp.logbuttons.load configure -text "Loading..."
            .torexplor.hlf.inp.logbuttons.load configure -state disabled
            update idletasks
            set loglist {}
            foreach ele [.torexplor.hlf.inp.logtv children {}] { lappend loglist [lindex [.torexplor.hlf.inp.logtv item $ele -values] end] } 
            ::ForceFieldToolKit::GenDihScan::TorExplor::setup $loglist
            .torexplor.hlf.inp.logbuttons.load configure -text "Load"
            .torexplor.hlf.inp.logbuttons.load configure -state normal
            update idletasks
        }

    # GRID
    grid $w.hlf.inp -column 0 -row 0 -sticky nswe
            
    grid $w.hlf.inp.psflbl    -column 0 -row 0 -sticky nsw
    grid $w.hlf.inp.psfentry  -column 1 -row 0 -sticky nswe -padx $entryPadX   -pady $entryPadY  -columnspan 2
    grid $w.hlf.inp.psfbrowse -column 3 -row 0 -sticky nswe -padx $vbuttonPadX -pady $vbuttonPadY

    grid $w.hlf.inp.pdblbl    -column 0 -row 1 -sticky nsw
    grid $w.hlf.inp.pdbentry  -column 1 -row 1 -sticky nswe -padx $entryPadX   -pady $entryPadY -columnspan 2
    grid $w.hlf.inp.pdbbrowse -column 3 -row 1 -sticky nswe -padx $vbuttonPadX -pady $vbuttonPadY

    grid $w.hlf.inp.loglbl      -column 0 -row 2 -sticky nsw
    grid $w.hlf.inp.logtv       -column 0 -row 3 -sticky nswe -columnspan 2 -rowspan 4
    grid $w.hlf.inp.logtvScroll -column 2 -row 3 -sticky nswe               -rowspan 4

    grid $w.hlf.inp.logbuttons       -column 3 -row 3 -sticky nwe
    grid $w.hlf.inp.logbuttons.add   -column 0 -row 0 -sticky nswe -padx $vbuttonPadX -pady $vbuttonPadY
    grid $w.hlf.inp.logbuttons.del   -column 0 -row 1 -sticky nswe -padx $vbuttonPadX -pady $vbuttonPadY
    grid $w.hlf.inp.logbuttons.clear -column 0 -row 2 -sticky nswe -padx $vbuttonPadX -pady $vbuttonPadY
    grid $w.hlf.inp.logbuttons.sep   -column 0 -row 3 -sticky nswe -padx $hsepPadX    -pady $hsepPadY
    grid $w.hlf.inp.logbuttons.load  -column 0 -row 4 -sticky nswe -padx $vbuttonPadX -pady $vbuttonPadY

    # dihedral analysis
    # BUILD
    ttk::label $w.hlf.inp.dihLbl -text "Dihedral Analysis" -anchor w

    ttk::frame       $w.hlf.inp.dihFrame1
    ttk::button      $w.hlf.inp.dihFrame1.analyze -text "Analyze Trajectory" -command {::ForceFieldToolKit::GenDihScan::TorExplor::slopeAnalysis}
    ttk::label       $w.hlf.inp.dihFrame1.updateLbl -text "Update Dihedral Data" -anchor w
    ttk::radiobutton $w.hlf.inp.dihFrame1.updateOn -text "On" -variable ::ForceFieldToolKit::GenDihScan::TorExplor::updateDihData -value 1
    ttk::radiobutton $w.hlf.inp.dihFrame1.updateOff -text "Off" -variable ::ForceFieldToolKit::GenDihScan::TorExplor::updateDihData -value 0

    ttk::treeview $w.hlf.inp.dihDataTv \
        -selectmode browse \
        -columns {indDef typeDef d dd} \
        -displaycolumns {indDef typeDef d dd} \
        -show {headings} \
        -height 10 \
        -yscrollcommand "$w.hlf.inp.dihDataTvScroll set"
    ttk::scrollbar $w.hlf.inp.dihDataTvScroll -orient vertical -command "$w.hlf.inp.dihDataTv yview"
    bind $w.hlf.inp.dihDataTv <<TreeviewSelect>> { ::ForceFieldToolKit::GenDihScan::TorExplor::dihAnalysisSelDidChange }
    bind $w.hlf.inp.dihDataTv <KeyPress-Escape> { .torexplor.hlf.inp.dihDataTv selection set {} }

    $w.hlf.inp.dihDataTv tag configure activeScan -foreground red

    $w.hlf.inp.dihDataTv heading indDef  -text "Indices"   -anchor center
    $w.hlf.inp.dihDataTv heading typeDef -text "Type"      -anchor center
    $w.hlf.inp.dihDataTv heading d       -text "$theta"    -anchor center
    $w.hlf.inp.dihDataTv heading dd      -text "d${theta}" -anchor center

    $w.hlf.inp.dihDataTv column indDef  -width 100 -stretch 0
    $w.hlf.inp.dihDataTv column typeDef -width 250  -stretch 1
    $w.hlf.inp.dihDataTv column d       -width 100 -stretch 0
    $w.hlf.inp.dihDataTv column dd      -width 100 -stretch 0

    # GRID
    grid $w.hlf.inp.dihLbl -column 0 -row 8 -sticky nswe

    grid $w.hlf.inp.dihFrame1 -column 0 -row 9 -columnspan 3 -sticky nswe
        grid $w.hlf.inp.dihFrame1.analyze   -column 0 -row 0 -sticky nswe
        grid $w.hlf.inp.dihFrame1.updateLbl -column 1 -row 0 -sticky nswe
        grid $w.hlf.inp.dihFrame1.updateOn  -column 2 -row 0 -sticky nswe
        grid $w.hlf.inp.dihFrame1.updateOff -column 3 -row 0 -sticky nswe

    grid $w.hlf.inp.dihDataTv       -column 0 -row 10 -sticky nswe -columnspan 2 -rowspan 2
    grid $w.hlf.inp.dihDataTvScroll -column 2 -row 10 -sticky nswe               -rowspan 2


    # INP CONFIGURATIONS
    grid columnconfigure $w.hlf.inp {1} -weight 1 -minsize 250
    grid rowconfigure    $w.hlf.inp {10} -weight 1

    # plots frame
    # BUILD
    ttk::frame $w.hlf.plot

    # plot space
    ttk::frame $w.hlf.plot.plot
    set ::ForceFieldToolKit::GenDihScan::TorExplor::plothandle [multiplot embed $w.hlf.plot.plot \
        -title "QM Potential Energy Surface" \
        -xlabel "Scan Frame" \
        -ylabel "Rel. E." \
        -xsize 680 -ysize 450 -xmin 0 -xmax auto -ymin auto -ymaxe auto \
        -lines -linewidth 1]


    # controls
    ttk::frame $w.hlf.plot.controls

    # axis sliders
    ttk::frame $w.hlf.plot.controls.sliders
    ttk::label $w.hlf.plot.controls.sliders.xMinLbl -text "x-min" -anchor center
    ttk::scale $w.hlf.plot.controls.sliders.xMin -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPlotScale xmin }
    ttk::label $w.hlf.plot.controls.sliders.xMaxLbl -text "x-max" -anchor center
    ttk::scale $w.hlf.plot.controls.sliders.xMax -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPlotScale xmax }

    ttk::label $w.hlf.plot.controls.sliders.yMinLbl -text "y-min" -anchor center
    ttk::scale $w.hlf.plot.controls.sliders.yMin -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPlotScale ymin }
    ttk::label $w.hlf.plot.controls.sliders.yMaxLbl -text "y-max" -anchor center
    ttk::scale $w.hlf.plot.controls.sliders.yMax -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPlotScale ymax }

    # horizontal separators
    ttk::separator $w.hlf.plot.controls.hsep1 -orient horizontal
    ttk::separator $w.hlf.plot.controls.hsep2 -orient horizontal


    # axis manual entry
    ttk::frame  $w.hlf.plot.controls.xySet
    ttk::label  $w.hlf.plot.controls.xySet.xMinLbl -text "x-min" -anchor center
    ttk::entry  $w.hlf.plot.controls.xySet.xMin -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::xmin -width 4 -justify center
    ttk::label  $w.hlf.plot.controls.xySet.xMaxLbl -text "x-max" -anchor center
    ttk::entry  $w.hlf.plot.controls.xySet.xMax -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::xmax -width 4 -justify center
    ttk::label  $w.hlf.plot.controls.xySet.yMinLbl -text "y-min" -anchor center
    ttk::entry  $w.hlf.plot.controls.xySet.yMin -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::ymin -width 4 -justify center
    ttk::label  $w.hlf.plot.controls.xySet.yMaxLbl -text "y-max" -anchor center
    ttk::entry  $w.hlf.plot.controls.xySet.yMax -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::ymax -width 4 -justify center
    ttk::button $w.hlf.plot.controls.xySet.set -text "Set Axis" -command {
        $::ForceFieldToolKit::GenDihScan::TorExplor::plothandle configure \
            -xmin $::ForceFieldToolKit::GenDihScan::TorExplor::xmin -xmax $::ForceFieldToolKit::GenDihScan::TorExplor::xmax \
            -ymin $::ForceFieldToolKit::GenDihScan::TorExplor::ymin -ymax $::ForceFieldToolKit::GenDihScan::TorExplor::ymax
        $::ForceFieldToolKit::GenDihScan::TorExplor::plothandle replot
    }
    ttk::separator $w.hlf.plot.controls.xySet.vsep1 -orient vertical
    ttk::checkbutton $w.hlf.plot.controls.xySet.as -offvalue 0 -onvalue 1 -variable ::ForceFieldToolKit::GenDihScan::TorExplor::plotAutoscaling
    ttk::label $w.hlf.plot.controls.xySet.asLbl -text "Axis Autoscaling" -anchor w

    # point/line size
    ttk::frame $w.hlf.plot.controls.ptln
    ttk::label $w.hlf.plot.controls.ptln.ptlbl -text "Point Size:" -anchor w
    ttk::button $w.hlf.plot.controls.ptln.ptdecr -text "-" -width 0 \
        -command {
            if { $::ForceFieldToolKit::GenDihScan::TorExplor::ptSize != 1 } {
                incr ::ForceFieldToolKit::GenDihScan::TorExplor::ptSize -1;
                ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPointSize $::ForceFieldToolKit::GenDihScan::TorExplor::ptSize
            }
        }
    ttk::label $w.hlf.plot.controls.ptln.ptsize  -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::ptSize -width 2 -anchor center
    ttk::button $w.hlf.plot.controls.ptln.ptincr -text "+" -width 0 \
        -command {
            incr ::ForceFieldToolKit::GenDihScan::TorExplor::ptSize
            ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPointSize $::ForceFieldToolKit::GenDihScan::TorExplor::ptSize
        } 
    ttk::separator $w.hlf.plot.controls.ptln.vsep -orient vertical
    ttk::label $w.hlf.plot.controls.ptln.lnlbl -text "Line Width:" -anchor w
    ttk::button $w.hlf.plot.controls.ptln.lndecr -text "-" -width 0 \
        -command {
            if { $::ForceFieldToolKit::GenDihScan::TorExplor::lnSize != 1} {
                incr ::ForceFieldToolKit::GenDihScan::TorExplor::lnSize -1
                ::ForceFieldToolKit::GenDihScan::TorExplor::adjustLineSize $::ForceFieldToolKit::GenDihScan::TorExplor::lnSize
            }
        } 
    ttk::label $w.hlf.plot.controls.ptln.lnsize  -textvariable ::ForceFieldToolKit::GenDihScan::TorExplor::lnSize -width 2 -anchor center
    ttk::button $w.hlf.plot.controls.ptln.lnincr -text "+" -width 0 \
        -command {
            incr ::ForceFieldToolKit::GenDihScan::TorExplor::lnSize
            ::ForceFieldToolKit::GenDihScan::TorExplor::adjustLineSize $::ForceFieldToolKit::GenDihScan::TorExplor::lnSize
        } 

    # GRID
    grid $w.hlf.plot          -column 1 -row 0 -sticky nswe -rowspan 2
    grid $w.hlf.plot.plot     -column 0 -row 0 -sticky nswe
    grid $w.hlf.plot.controls -column 0 -row 1 -sticky nswe

    grid $w.hlf.plot.controls.sliders -column 0 -row 0 -sticky nswe -padx 10
    grid $w.hlf.plot.controls.sliders.xMinLbl -column 0 -row 0 -sticky nswe -padx "6 0"
    grid $w.hlf.plot.controls.sliders.xMin -column 1 -row 0 -sticky nswe -padx 6
    grid $w.hlf.plot.controls.sliders.xMaxLbl -column 0 -row 1 -sticky nswe -padx "6 0"
    grid $w.hlf.plot.controls.sliders.xMax -column 1 -row 1 -sticky nswe -padx 6
    grid $w.hlf.plot.controls.sliders.yMinLbl -column 2 -row 0 -sticky nswe
    grid $w.hlf.plot.controls.sliders.yMin -column 3 -row 0 -sticky nswe -padx "6 0"
    grid $w.hlf.plot.controls.sliders.yMaxLbl -column 2 -row 1 -sticky nswe
    grid $w.hlf.plot.controls.sliders.yMax -column 3 -row 1 -sticky nswe -padx "6 0"

    grid $w.hlf.plot.controls.hsep1 -column 0 -row 1 -sticky nswe -padx $hsepPadX -pady $hsepPadY

    grid $w.hlf.plot.controls.xySet -column 0 -row 2 -sticky nswe -padx 10
    grid $w.hlf.plot.controls.xySet.xMinLbl -column 0 -row 0 -sticky nswe -padx "2 0"
    grid $w.hlf.plot.controls.xySet.xMin -column 1 -row 0 -sticky nswe -padx 6
    grid $w.hlf.plot.controls.xySet.xMaxLbl -column 2 -row 0 -sticky nswe
    grid $w.hlf.plot.controls.xySet.xMax -column 3 -row 0 -sticky nswe -padx 6
    grid $w.hlf.plot.controls.xySet.yMinLbl -column 4 -row 0 -sticky nswe
    grid $w.hlf.plot.controls.xySet.yMin -column 5 -row 0 -sticky nswe -padx 6
    grid $w.hlf.plot.controls.xySet.yMaxLbl -column 6 -row 0 -sticky nswe
    grid $w.hlf.plot.controls.xySet.yMax -column 7 -row 0 -sticky nswe -padx 6
    grid $w.hlf.plot.controls.xySet.set -column 8 -row 0 -sticky nswe -padx 4
    grid $w.hlf.plot.controls.xySet.vsep1 -column 9 -row 0 -sticky ns -padx 6
    grid $w.hlf.plot.controls.xySet.as -column 10 -row 0 -sticky nswe
    grid $w.hlf.plot.controls.xySet.asLbl -column 11 -row 0 -sticky nswe -padx "0 2"

    grid $w.hlf.plot.controls.hsep2 -column 0 -row 3 -sticky nswe -padx $hsepPadX -pady $hsepPadY

    grid $w.hlf.plot.controls.ptln -column 0 -row 4 -sticky nw

    grid $w.hlf.plot.controls.ptln.ptlbl  -column 0 -row 0 -sticky nswe
    grid $w.hlf.plot.controls.ptln.ptdecr -column 1 -row 0 -sticky nw   -padx 4 -pady 4
    grid $w.hlf.plot.controls.ptln.ptsize -column 2 -row 0 -sticky nsw
    grid $w.hlf.plot.controls.ptln.ptincr -column 3 -row 0 -sticky nw   -padx 4 -pady 4
    grid $w.hlf.plot.controls.ptln.lnlbl  -column 4 -row 0 -sticky nswe
    grid $w.hlf.plot.controls.ptln.lndecr -column 5 -row 0 -sticky nw   -padx 4 -pady 4
    grid $w.hlf.plot.controls.ptln.lnsize -column 6 -row 0 -sticky nsw
    grid $w.hlf.plot.controls.ptln.lnincr -column 7 -row 0 -sticky nw   -padx 4 -pady 4

    grid columnconfigure $w.hlf.plot.controls.sliders {1 3} -weight 1
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::setup {logList} {
    # read log files, load scan frames, construct plot

    # localize some things
    foreach ele {psf pdb molid colorList colorIdMap colorInd plothandle plotMax plotData plotAutoscaling scanIndsArr repNameScanInds} {variable $ele}
    global vmd_frame vmd_molecule

    # some basic validation
    if { $psf eq "" || ![file exists $psf] } {tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot find PSF file"; return}
    if { $pdb eq "" || ![file exists $pdb] } {tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot find PDB file"; return}
    foreach l $logList {
        if { $l eq "" || ![file exists $l] } {tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot find LOG file"; return}
    }

    # reset gui and vmd where necessary
    set plotData {}
    set colorInd 0
    if { $molid != -1 } { mol delete $molid }

    # gather data
    set logData {}
    foreach log $logList {
        set parseResults [eval ::ForceFieldToolKit::GenDihScan::TorExplor::readLog $log]
        if { [llength $parseResults] == 3 } { lappend logData $parseResults }
    }

    # re-combine any bi-directional scans
    set combData {}
    while { [llength $logData] > 0 } {
        set searchInds [lindex $logData 0 0]
        set matchList [lsearch -index 0 -all [lrange $logData 1 end] $searchInds]
        if { [llength $matchList] == 0 } {
            # no match, just copy the data
            lappend combData [lindex $logData 0]
        } else {
            # concat all matched data
            # incr matched data indices to realign back to logData position
            for {set i 0} {$i < [llength $matchList]} {incr i} { lset matchList $i [expr {[lindex $matchList $i] + 1}] }
            # add the original data
            set concatEnList    [lindex $logData 0 1]
            set concatCoordList [lindex $logData 0 2]
            # concat the matched elements with the original data
            foreach ind $matchList {
                set concatEnList    [concat $concatEnList    [lindex $logData $ind 1]]
                set concatCoordList [concat $concatCoordList [lindex $logData $ind 2]]
            }
            # copy the combined data set
            lappend combData [list $searchInds $concatEnList $concatCoordList]
        }
        # remove the copied data from the original data list (go from end to back to preserve list index)
        foreach ind [lsort -integer -decreasing [concat 0 $matchList]] { set logData [lreplace $logData $ind $ind] }
    }

    # replace the original log data with the new combined data
    set logData $combData; unset combData
    
    # find global min
    set globalMin  Inf
    set globalMax -Inf
    foreach dset $logData {
        foreach en [lindex $dset 1] {
            #if { ![string is digit -strict $globalMin] || $en < $globalMin } { set globalMin $en }
            if { $en < $globalMin } { set globalMin $en }
            #if { ![string is digit -strict $globalMax] || $en > $globalMax } { set globalMax $en }
            if { $en > $globalMax } { set globalMax $en }
        }
    }

    set plotMax [expr {$globalMax - $globalMin}]

    # rescale energy data
    for {set i 0} {$i < [llength $logData]} {incr i} {
        for {set j 0} {$j < [llength [lindex $logData $i 1]]} {incr j} {
            lset logData $i 1 $j [expr { [lindex $logData $i 1 $j] - $globalMin }]
        }
    }

    # load coordinates into VMD
    set molid [mol new $psf waitfor all]
    set aCount [molinfo top get numatoms]

    # cycle through each log file
    array unset scanIndsArr; array set scanIndsArr {}
    set plotX 0; # x value for plot
    foreach log $logData {

        set currScanInds [lindex $log 0]

        # set color for log data
        set color [lindex $colorList $colorInd]
        incr colorInd
        # if we've reached end of the list, wrap around to beginning
        if { $colorInd == [llength $colorList] } { set colorInd 0 }

        set xdata {}
        set ydata {}
        # cycle through each step (en and coord)
        for {set i 0} {$i < [llength [lindex $log 1]]} {incr i} {
            # add frame
            mol addfile $pdb waitfor all $molid

            # add data to scanIndsArray
            set scanIndsColor $colorIdMap($color)
            set scanIndsArr($plotX) [list $currScanInds $scanIndsColor]

            # set user to energy
            set en [lindex $log 1 $i]
            set sel [atomselect $molid all]
            $sel set user $en
            $sel delete
            lappend xdata $plotX; incr plotX
            lappend ydata $en

            # move coordinates for each atom
            for {set j 0} {$j < $aCount} {incr j} {
                set sel [atomselect $molid "index $j"]
                $sel set {x y z} [list [lindex $log 2 $i $j]]
                $sel delete
            }
        }

        # add data to the plot
        lappend plotData [list $color $xdata $ydata]
    }

    # create the scanInds representation
    mol addrep $molid
    set tempRepID [expr {[molinfo $molid get numreps] - 1}]
    mol modstyle $tempRepID $molid {Licorice 0.1 25.0 25.0}
    set repNameScanInds [mol repname $molid $tempRepID]

    # rescale sliders to fit data
    foreach xs {xMin xMax} { .torexplor.hlf.plot.controls.sliders.${xs} configure -from 0 -to $plotX }
    foreach ys {yMin yMax} { .torexplor.hlf.plot.controls.sliders.${ys} configure -from 0 -to $plotMax }

    # if autoscaling is turned on, reset slider positions
    if { $plotAutoscaling } {
        $plothandle configure -xmin 0 -xmax $plotX -ymin 0 -ymax $plotMax
        .torexplor.hlf.plot.controls.sliders.xMin configure -value 0
        .torexplor.hlf.plot.controls.sliders.xMax configure -value $plotX
        .torexplor.hlf.plot.controls.sliders.yMin configure -value 0
        .torexplor.hlf.plot.controls.sliders.yMax configure -value $plotMax
    }
    
    # refresh plot
    eval ::ForceFieldToolKit::GenDihScan::TorExplor::plotData

    # put a trace on the frame number
    trace add variable vmd_frame write ::ForceFieldToolKit::GenDihScan::TorExplor::frameChangeCallback
    animate goto 0

    # put a trace on the molid so that we can detect if it has been deleted out from under us
    trace add variable vmd_molecule write ::ForceFieldToolKit::GenDihScan::TorExplor::molidCleanupCallback
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::readLog {log} {
    # reads data from Gaussian log
    # input: log file
    # returns:
    #   {
    #    ind_def (0-based)
    #    {step energies}
    #    {step coordinates} 
    #   }

    # initialize some variables
    set indDef {}; set stepEn {}; set stepCoords {}

    # open the log file for reading
    set infile [open $log r]
    while { ![eof $infile] } {
        # read a line at a time
        set inline [string trim [gets $infile]]

        switch -regexp $inline {
            {Initial Parameters} {
                # keep reading until finding the dihedral being scanned
                # and parse out 1-based indices, convert to 0-based
                while { ![regexp {^\!.*D\(([0-9]+),([0-9]+),([0-9]+),([0-9]+)\).*Scan[ \t]+\!$} [string trim [gets $infile]] full_match ind1 ind2 ind3 ind4] && ![eof $infile] } { continue }
                if { [eof $infile] } { return }
                foreach ele [list $ind1 $ind2 $ind3 $ind4] { lappend indDef [expr {$ele - 1}] }
            }

            {Input orientation:} {
                # clear any existing coordinates
                set currCoords {}
                # burn the header
                for {set i 0} {$i<=3} {incr i} { gets $infile }
                # parse coordinates
                while { [string range [string trimleft [set line [gets $infile]] ] 0 0] ne "-" } { lappend currCoords [lrange $line 3 5] }
            }

            {SCF[ \t]*Done:} {
                # parse E(RHF) energy; convert hartrees to kcal/mol
                set currEnergy [expr {[lindex $inline 4] * 627.5095}]
                # NOTE: this value will be overridden if E(MP2) is also found
            }

            {E2.*EUMP2} {
                # convert from Gaussian notation in hartrees to scientific notation
                set currEnergy [expr {[join [split [lindex [string trim $inline] end] D] E] * 627.5095}]
                # NOTE: this overrides the E(RHF) parse from above
            }

            {Optimization completed\.} {
                # we've reached the optimized conformation
                lappend stepEn $currEnergy
                lappend stepCoords $currCoords
            }

            default {continue}
        }
    }

    close $infile

    # reverse data if it's a negative scan
    if { [regexp {\.neg\.} $log] } {
        set stepEn [lreverse $stepEn]
        set stepCoords [lreverse $stepCoords]
    }

    # return the parsed data
    return [list $indDef $stepEn $stepCoords]
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::plotData {} {

    # localize variables
    foreach ele {plothandle plotMax molid plotData ptSize lnSize} { variable $ele }
    global vmd_frame

    # reset the plot
    $plothandle clear

    # add the indicator line
    set indX [list $vmd_frame($molid) $vmd_frame($molid)]
    set indY [list 0 $plotMax]
    $plothandle add $indX $indY -lines -linewidth 4 -linecolor "black"

    # add the data
    # { color xset yset }
    foreach dset $plotData {
        set hexRGB [::ForceFieldToolKit::GenDihScan::TorExplor::colorMapVMD2Tk [colorinfo rgb [lindex $dset 0]]]
        $plothandle add [lindex $dset 1] [lindex $dset 2] \
            -marker point \
            -radius $ptSize \
            -lines \
            -linewidth $lnSize \
            -fillcolor $hexRGB \
            -linecolor $hexRGB
    }

    # udpate the plot
    $plothandle replot
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPlotScale {scaleType value} {
    variable plothandle
    $plothandle configure -${scaleType} $value
    $plothandle replot
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::adjustPointSize {size} {
    variable plothandle
    for {set i 1} {$i < [$plothandle nsets]} {incr i} { $plothandle configure -set $i -radius $size }
    $plothandle replot
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::adjustLineSize {size} {
    variable plothandle
    for {set i 1} {$i < [$plothandle nsets]} {incr i} { $plothandle configure -set $i -linewidth $size }
    $plothandle replot
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::slopeAnalysis {} {
    # Computes dihedral and change in dihedral for every dihedral at every frame

    # localize relevant variables
    variable molid
    variable dihAnalysisArr

    # bail if trying to call without a molecule loaded for torexplor
    if { $molid == -1 } { return }

    # reset the dihedral analysis data array
    array unset dihAnalysisArr; array set dihAnalysisArr {}

    # build a list of index defined dihedrals
    set dihList {}
    foreach ele [topo getdihedrallist -molid $molid] {
        lappend dihList [lrange $ele 1 end]
    }

    set numframes [molinfo $molid get numframes]

    # cycle through each dihedral
    foreach ele $dihList {

        # define the type def
        set typedef {}
        foreach ind $ele {
            set sel [atomselect $molid "index $ind"]
            lappend typedef [$sel get type]
            $sel delete
        }

        # make an entry in the dihedral analysis tv box
        set currTvID [.torexplor.hlf.inp.dihDataTv insert {} end -values [list $ele $typedef {} {}] -tags {}]

        # cycle through trajectory
        set f_prev -1; set d_prev ""
        set f 0; set d [expr {abs([measure dihed $ele frame $f])}]
        
        for {set f_next 1} {$f_next < $numframes} {incr f_next} {
            # measure dihedral and compute change in dihedral
            set d_next [expr {abs([measure dihed $ele frame $f_next])}]
            if { $f_prev == -1 } {
                set dd [expr {abs([::ForceFieldToolKit::GenDihScan::TorExplor::computeSlope2 $f $d $f_next $d_next])}]
            } else {
                set dd [expr {abs([::ForceFieldToolKit::GenDihScan::TorExplor::computeSlope3 $f_prev $d_prev $f $d $f_next $d_next])}]
            }

            # add data to the array
            lappend dihAnalysisArr($f) [list $currTvID [format %.2f $d] [format %.2f $dd]]

            # roll the data backward
            set f_prev $f; set d_prev $d
            set f $f_next; set d $d_next
        }

        # add the last data point (2-point slope calc, no d_next)
        set dd [expr {abs([::ForceFieldToolKit::GenDihScan::TorExplor::computeSlope2 $f_prev $d_prev $f $d])}]
        lappend dihAnalysisArr($f) [list $currTvID [format %.2f $d] [format %.2f $dd]]
    }

    # sort the array within each frame by decreasing dd
    foreach arrName [array names dihAnalysisArr] {
        set dihAnalysisArr($arrName) [lsort -decreasing -index 2 $dihAnalysisArr($arrName)]
    }

    # update tv box using current frame
    global vmd_frame
    set f $vmd_frame($molid)
    set pos 0
    foreach ele $dihAnalysisArr($f) {
        lassign $ele tvid d dd
        .torexplor.hlf.inp.dihDataTv set $tvid d $d
        .torexplor.hlf.inp.dihDataTv set $tvid dd $dd
        .torexplor.hlf.inp.dihDataTv move $tvid {} $pos
        incr pos
    }
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::computeSlope2 { x1 y1 x2 y2 } {
    return [expr { ($y2 - $y1) / ($x2 - $x1) }]
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::computeSlope3 { x1 y1 x2 y2 x3 y3 } {
    set cnt 3
    set sumx  [expr { $x1 + $x2 + $x3 }]
    set sumy  [expr { $y1 + $y2 + $y3 }]
    set sumx2 [expr { pow($x1,2) + pow($x2,2) + pow($x3,2) }]
    set sumxy [expr { $x1*$y1 + $x2*$y2 + $x3*$y3 }]
    set xmean [expr { $sumx / $cnt }]
    set ymean [expr { $sumy / $cnt }]

    set slope [expr { ($sumxy - $sumx * $ymean) / ($sumx2 - $sumx * $xmean) }]

    return $slope
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::dihAnalysisSelDidChange {} {
    # Create/Update/Delete representation when the dihedral
    # analysis treeview selection changes

    # localize relevant variables
    variable molid
    variable repNameDihAnalysis

    set tempRepID [mol repindex $molid $repNameDihAnalysis]

    if { [llength [.torexplor.hlf.inp.dihDataTv selection]] == 0 } {
        # handle the deletion if the TV selection is empty
        if { $tempRepID == -1 } {
            # rep doesn't exist, nothing to do
            return
        } else {
            # delete the rep and reset the tracking variable
            mol delrep $tempRepID $molid
            set repNameDihAnalysis ""
        }
    } else {
        # handle the create/update if the TV selection is not empty
        # create the rep if it doesn't exist (new or has been deleted out from under us)
        if { $tempRepID == -1 } {
            # create the rep
            mol addrep $molid
            set tempRepID [expr {[molinfo $molid get numreps] -1}]
            set repNameDihAnalysis [mol repname $molid $tempRepID]
            mol modstyle $tempRepID $molid CPK 0.8 0.4 25.0 25.0
            mol modcolor $tempRepID $molid [list ColorID 5]
        }
        # update the rep
        set indList [.torexplor.hlf.inp.dihDataTv set [.torexplor.hlf.inp.dihDataTv selection] indDef]
        mol modselect $tempRepID $molid "index $indList"
    }
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::frameChangeCallback {args} {
    # As frame changes --
    # Updates the vertical frame indicator in plot
    # Updates scan rep in VMD main
    # Updates dihedral analysis (if updating)

    # Proc is registered to a write trace on vmd_frame array
    # note: key = molid, value = current frame

    # args passed by trace callback:
    # 0 - name1 - array name being traced -- expected value vmd_frame
    # 1 - name2 - index of array that has been changed
    # 2 - op    - operation that triggered the trace command (e.g., write)

    # localize relevant variables
    foreach ele {plothandle plotMax molid scanIndsArr scanIndsPrevFrame repNameScanInds dihAnalysisArr updateDihData} { variable $ele }
    global vmd_frame

    # if we're not changing the molid associated with torexplor, bail out
    if { [lindex $args 1] != $molid } { return }

    # move the vertical line (current frame indicator in plot)
    set f $vmd_frame($molid)
    $plothandle configure -set 0 -x [list $f $f] -y [list 0 $plotMax]
    $plothandle replot

    # update representation
    # lookup some data
    set scanIndsData $scanIndsArr($f)
    set currInds  [lindex $scanIndsData 0]
    set currColor [lindex $scanIndsData 1]
    
    set tempRepID [mol repindex $molid $repNameScanInds]

    # test for missing rep or if rep needs updating
    if { $tempRepID == -1 } {
        # our rep has been deleted out from under us, make a new one
        mol addrep $molid
        set tempRepID [expr {[molinfo $molid get numreps] - 1}]
        mol modstyle $tempRepID $molid {Licorice 0.1 25.0 25.0}
        set repNameScanInds [mol repname $molid $tempRepID]
        mol modselect $tempRepID $molid "index $currInds"
        mol modcolor $tempRepID $molid [list ColorID $currColor]
    } elseif {"$scanIndsPrevFrame" ne "$currInds"} {
        # we have switched scanInds, update the rep
        mol modselect $tempRepID $molid "index $currInds"
        mol modcolor $tempRepID $molid [list ColorID $currColor]
    }

    # update prev frame variable
    set scanIndsPrevFrame $currInds

    # update dihedral data tv box, if the update is active
    if { $updateDihData } {
        # update and sort the tv box data for the current frame
        set pos 0 ; # sort position counter

        foreach ele $dihAnalysisArr($f) {
            # update data
            lassign $ele tvid d dd
            .torexplor.hlf.inp.dihDataTv set $tvid d $d
            .torexplor.hlf.inp.dihDataTv set $tvid dd $dd

            # set formatting tags based on indDef
            set tvIndDef [.torexplor.hlf.inp.dihDataTv set $tvid indDef]
            if { "$currInds" ne "$tvIndDef" } {
                .torexplor.hlf.inp.dihDataTv item $tvid -tags {}
            } else {
                .torexplor.hlf.inp.dihDataTv item $tvid -tags {activeScan}
            }

            # array is pre-sorted so just move into place using position counter
            .torexplor.hlf.inp.dihDataTv move $tvid {} $pos
            incr pos
        } ; # end of frame element loop
    } ; # end of dihedral analysis data update
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::molidCleanupCallback {args} {
    # Cleans up if torexplor molecule is deleted
    # - Clears out plot data
    # - Clears out the dihedral analysis data
    # - Removes traces on vmd_frame and vmd_molecule
    # Called by a trace on the vmd_molecule

    # args
    # name1 - array (vmd_molecule)
    # name2 - element of array (molid)
    # op    - event that triggered callback (e.g., write)

    # localize necessary variables
    variable molid
    variable repNameScanInds; variable repNameDihAnalysis
    global vmd_molecule

    # 0 is the code for the deletion (1 is created, 2 is renamed)
    if { $vmd_molecule($molid) != 0 } { return }

    # dump plot data and clear plot data sets (NOTE: multiplot won't redraw empty canvas)
    variable plotData
    set plotData {}
    variable plothandle
    $plothandle clear

    # delete the dihedral analysis
    .torexplor.hlf.inp.dihDataTv delete [.torexplor.hlf.inp.dihDataTv children {}]
    variable updateDihData
    set updateDihData 0

    # remove the traces
    trace remove variable vmd_frame write ::ForceFieldToolKit::GenDihScan::TorExplor::frameChangeCallback
    trace remove variable vmd_molecule write ::ForceFieldToolKit::GenDihScan::TorExplor::molidCleanupCallback

    # reset molid and repNames back to initial defaults
    set molid -1
    set repNameScanInds ""
    set repNameDihAnalysis ""
}
#======================================================
proc ::ForceFieldToolKit::GenDihScan::TorExplor::colorMapVMD2Tk { rgbVMD } {
    # converts VMD's 0->1 RGB scale to 0->255 RGB scale and
    # returns hexadecimal RGB string used by Tk to color canvas items
    lassign $rgbVMD rIn gIn bIn

    set rgbScale 255

    set rOut [expr {int($rIn * $rgbScale)}]
    set gOut [expr {int($gIn * $rgbScale)}]
    set bOut [expr {int($bIn * $rgbScale)}]

    return [format #%02x%02x%02x $rOut $gOut $bOut]
}
#======================================================