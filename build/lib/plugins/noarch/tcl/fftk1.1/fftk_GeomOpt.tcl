#
# $Id: fftk_GeomOpt.tcl,v 1.9 2012/09/19 18:17:09 mayne Exp $
#

#======================================================
namespace eval ::ForceFieldToolKit::GeomOpt:: {

    variable pdb
    variable com
    
    variable qmProc
    variable qmMem
    variable qmCharge
    variable qmMult
    variable qmRoute

    variable logFile
    variable optPdb
    
}
#======================================================
proc ::ForceFieldToolKit::GeomOpt::init {} {
    
    # localize variables
    variable pdb
    variable com
    
    variable qmProc
    variable qmMem
    variable qmCharge
    variable qmMult
    variable qmRoute

    variable logFile
    variable optPdb
    
    # Set Variables to Initial value
    set pdb {}
    set com {}
    
    ::ForceFieldToolKit::GeomOpt::resetGaussianDefaults
    #set qmProc 1
    #set qmMem 1
    #set qmCharge 0
    #set qmMult 1
    #set qmRoute "\# MP2/6-31G* Opt=(Redundant) SCF=Tight"

    set logFile {}
    set optPdb {}
    
}
#======================================================
proc ::ForceFieldToolKit::GeomOpt::sanityCheck {} {
    # checks to see that appropriate information is set prior to running
    
    # returns 1 if all input is sane
    # returns 0 if there is a problem
    
    # localize relevant GeomOpt variables
    
    variable pdb
    variable com
    
    variable qmProc
    variable qmMem
    variable qmCharge
    variable qmMult
    variable qmRoute
    
    # local variables
    set errorList {}
    set errorText ""
    
    # checks
    # make sure that pdb is entered and exists
    if { $pdb eq "" } {
        lappend errorList "No PDB file was specified."
    } else {
        if { ![file exists $pdb] } { lappend errorList "Cannot find PDB file." }
    }
    
    # make sure that com is enetered and exists
    if { $com eq "" } {
        lappend errorList "No output path was specified."
    } else {
        if { ![file writable [file dirname $com]] } { lappend errorList "Cannot write to output path." }
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
    if { $qmMult < 0 || $qmMult != [expr int($qmMult)] } { lappend errorList "Multiplicity must be a positive integer." }
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
proc ::ForceFieldToolKit::GeomOpt::writeComFile {} {
    # writes the Gaussian input file for the geometry optimization
    
    # localize relevant variables
    variable pdb
    variable com
    variable qmProc
    variable qmMem
    variable qmCharge
    variable qmMult
    variable qmRoute

    # sanity check
    if { ![::ForceFieldToolKit::GeomOpt::sanityCheck] } { return }

    # procedure
    mol new $pdb
    
    # assign Gaussian atom names and gather x,y,z for output com file
    set Gnames {}
    set atom_info {}
    for {set i 0} {$i < [molinfo top get numatoms]} {incr i} {
        set temp [atomselect top "index $i"]
        lappend atom_info [list [$temp get element][expr $i+1] [$temp get x] [$temp get y] [$temp get z]]
        lappend Gnames [$temp get element][expr $i+1]
        $temp delete
    }

    # open the output com file
    set outfile [open $com w]
    
    # write the header
    puts $outfile "%chk=[file tail [file rootname $com]].chk"
    puts $outfile "%nproc=$qmProc"
    puts $outfile "%mem=${qmMem}GB"
    puts $outfile "$qmRoute"
    puts $outfile ""
    puts $outfile "<qmtool> simtype=\"Geometry optimization\" </qmtool>"
    puts $outfile ""
    puts $outfile "$qmCharge $qmMult"
    
    # write the coordinates
    foreach atom_entry $atom_info {
       puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
    }

    # empty line to terminate
    puts $outfile ""
    
    # clean up
    close $outfile
    mol delete top
}
#======================================================
proc ::ForceFieldToolKit::GeomOpt::loadLogFile {} {
    # loads the log file from the geometry optimization
    
    # localize relevant variables
    variable pdb
    variable logFile

    # check to makes sure that pdb is set
    if { $pdb eq "" || ![file exists $pdb] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "PDB file was not specified or could not be found."
        return
    }

    # make sure that logFile is set
    if { $logFile eq "" || ![file exists $logFile] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot opt Gaussian LOG file."
        return
    }


    # load the pdb file, followed by the coordinates from gaussian log
    set molId [mol new $pdb]
    set inFile [open $logFile r]

    while { ![eof $inFile] } {
        set inLine [string trim [gets $inFile]]
        if { $inLine eq "Input orientation:" } {
            # burn the coord header
            for {set i 0} {$i < 4} {incr i} { gets $inFile }
            # read coordinates
            set coords {}
            while { ![regexp {^-*$} [set inLine [string trim [gets $inFile]]]] } {
                lappend coords [lrange $inLine 3 5]
            }
            # add a new frame, set the coords 
            mol addfile $pdb
            for {set i 0} {$i < [llength $coords]} {incr i} {
                set temp [atomselect $molId "index $i"]
                $temp set x [lindex $coords $i 0]
                $temp set y [lindex $coords $i 1]
                $temp set z [lindex $coords $i 2]
                $temp delete
            }
            unset coords 
        } else {
            continue
        }
    }

    # clean up
    close $inFile


    ### old qmtool method (broken) ###
    #::QMtool::use_vmd_molecule $molId
    ##catch { ::QMtool::read_gaussian_log $logFile $molId }
    #::QMtool::read_gaussian_log $logFile $molId
    ##################################


    # message the console
    ::ForceFieldToolKit::gui::consoleMessage "Geometry optimization Gaussian log file loaded"
}
#======================================================
proc ::ForceFieldToolKit::GeomOpt::writeOptPDB {} {
    # writes a new pdb file with coordinates for optimized geometry
    
    # localize relevant variables
    variable pdb
    variable logFile
    variable optPdb

    # check to makes sure that pdb is set
    if { $pdb eq "" || ![file exists $pdb] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "PDB file was not specified or could not be found."
        return
    }

    # make sure that logFile is set
    if { $logFile eq "" || ![file exists $logFile] } {
        tk_messageBox -type ok -icon warning -message "Action halted on error!" -detail "Cannot opt Gaussian LOG file."
        return
    }

    # make sure that optPdb is set
    if { ![file writable [file dirname $optPdb]] } {
        tk_messageBox -type ok -icon warning -message "Action halded on error!" -detail "Cannot write to output directory."
        return
    }
    

    # load the pdb and load the coords from the log file
    set molId [mol new $pdb]

    # parse the geometry optimization LOG file
    set inFile [open $logFile r]

    while { ![eof $inFile] } {
        set inLine [string trim [gets $inFile]]
        if { $inLine eq "Input orientation:" } {
            # burn the coord header
            for {set i 0} {$i < 4} {incr i} { gets $inFile }
            # read coordinates
            set coords {}
            while { ![regexp {^-*$} [set inLine [string trim [gets $inFile]]]] } {
                lappend coords [lrange $inLine 3 5]
            }
            # (re)set the coords 
            for {set i 0} {$i < [llength $coords]} {incr i} {
                set temp [atomselect $molId "index $i"]
                $temp set x [lindex $coords $i 0]
                $temp set y [lindex $coords $i 1]
                $temp set z [lindex $coords $i 2]
                $temp delete
            }
            unset coords 
        } else {
            continue
        }
    }

    # write the new coords to file
    [atomselect $molId all] writepdb $optPdb
    
    # clean up 
    close $inFile
    mol delete $molId
    
    # message the console
    ::ForceFieldToolKit::gui::consoleMessage "Optimized geometry written to PDB file"
    
}
#======================================================
proc ::ForceFieldToolKit::GeomOpt::resetGaussianDefaults {} {
    # resets the gaussian settings to default

    # localize variables
    variable qmProc
    variable qmMem
    variable qmCharge
    variable qmMult
    variable qmRoute

    # reset to default
    set qmProc 1
    set qmMem 1
    set qmCharge 0
    set qmMult 1
    set qmRoute "\# MP2/6-31G* Opt=(Redundant) SCF=Tight Geom=PrintInputOrient"
}
#======================================================

