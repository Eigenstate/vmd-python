#
# $Id: fftk_BondAngleOpt.tcl,v 1.15 2015/12/31 05:17:51 gumbart Exp $
#

#======================================================
namespace eval ::ForceFieldToolKit::BondAngleOpt {

    variable psf
    variable pdb
    variable hessLog
    variable moleculeID
    
    variable minName
    variable namdbin
    variable outFile
    variable outFileName

    variable inputBondPars
    variable inputAnglePars
    variable parlist
    variable tempParName
    variable namdEnCommand
    
    variable uniqueTypes
    variable baInitial
    variable baIndList
    variable mapIndPar
    
    variable zmatqm
    variable zmatqmEff
    variable ringList
    variable atomTrans
    variable posTrans

    variable bondDev
    variable bondLB
    variable bondUB
    variable angDev
    variable angLB
    variable angUB

    variable bondscale
    variable angscale
    variable dxBond
    variable dxAng
    variable enscale
    variable geomWeight
    variable enWeight
    
    variable tol
    variable mode
    variable saT
    variable saTSteps
    variable saIter
    variable dhIter
    
    variable debug
    variable debugLog
    variable guiMode
    variable optCount
    
    variable parInProg
}
#======================================================
proc ::ForceFieldToolKit::BondAngleOpt::init {} {
    # localize + initialize variables
    variable psf {}
    variable pdb {}
    variable hessLog {}
    variable moleculeID {}
    
    variable minName "min-bondangles"
    variable namdbin "namd2"
    variable outFile {}
    variable outFileName "BondedOpt.log"

    variable inputBondPars {}
    variable inputAnglePars {}
    variable parlist {}
    variable tempParName "OPTTEMP.par"
    variable namdEnCommand ""
    
    variable uniqueTypes {}
    variable baInitial {}
    variable baIndList {}
    variable mapIndPar {}
    
    variable zmatqm {}
    variable zmatqmEff {}
    variable ringList {}
    variable atomTrans {}
    variable posTrans {}

    variable bondDev 0.25
    variable bondLB 0.0
    variable bondUB 1000.0
    variable angDev 10.0
    variable angLB 0.0
    variable angUB 300.0

    variable bondscale 0.03
    variable angscale 3.0
    variable enscale 1.0
    variable dxBond 0.1
    variable dxAng 5.0

    variable geomWeight 1.0
    variable enWeight 1.0
    
    variable tol 0.001
    variable mode "downhill"
    variable saT 25
    variable saTSteps 20
    variable saIter 15
    variable dhIter 500
    
    variable debug 0
    variable debugLog {}
    variable guiMode 1
    variable optCount 0
    
    variable parInProg {}
}
#======================================================
proc ::ForceFieldToolKit::BondAngleOpt::sanityCheck {} {
    # runs a sanity check on the input information prior to launching optimizatino
    
    # returns 1 if all input is sane
    # returns 0 if there are problems
    
    # localize relevant BondAngleOpt variables
    variable psf
    variable pdb
    variable hessLog
    variable parInProg
    variable parlist
    variable namdbin
    variable outFileName

    variable inputBondPars
    variable inputAnglePars

    variable mode
    variable tol
    variable geomWeight
    variable enWeight

    variable dhIter
    variable saT
    variable saTSteps
    variable saIter

    variable bondDev
    variable bondLB
    variable bondUB
    variable angDev
    variable angLB
    variable angUB
    
    # local variables
    set errorList {}
    set errorText ""
    
    # Input
    #------
    # make sure psf is entered and exists
    if { $psf eq "" } { lappend errorList "No PSF file was specified." } \
    else { if { ![file exists $psf] } { lappend errorList "Cannot find PSF file." } }
    
    # make sure pdb is entered and exists
    if { $pdb eq "" } { lappend errorList "No PDB file was specified." } \
    else { if { ![file exists $pdb] } { lappend errorList "Cannot find PDB file." } }
    
    # make sure hess log file is entered and exists
    if { $hessLog eq "" } { lappend errorList "No hessian LOG file was specified." } \
    else { if { ![file exists $hessLog] } { lappend errorList "Cannot find hessian LOG file." } }
    
    # make sure in-progress par file is entered and exists
    if { $parInProg eq "" } { lappend errorList "No in-progress PAR file was specified." } \
    else { if { ![file exists $parInProg] } { lappend errorList "Cannot find in-progress PAR file." } }
    
    # make sure there is at least one parameter file and that it/they exists
    if { [llength $parlist] == 0 } { lappend errorList "No parameter files were specified." } \
    else {
        foreach parFile $parlist {
            if { ![file exists $parFile] } { lappend errorList "Cannot open PAR file: $parFile." }
        }
    }
    
    # make sure namd2 command and/or file exists
    if { $namdbin eq "" } {
        lappend errorList "NAMD binary file (or command if in PATH) was not specified."
    } else { if { [::ExecTool::find $namdbin] eq "" } { lappend errorList "Cannot find NAMD binary file." } }
    
    # make sure that output log name is not empty and directory is writable
    if { $outFileName eq "" } { lappend errorList "Output LOF file was not specified." } \
    else { if { ![file writable [file dirname $outFileName]] } { lappend errorList "Cannot write to output LOG file directory." } }
    
    
    # Parameters to Optimize
    #-----------------------
    # make sure that there is at least one bond or angle to optimize
    if { [expr { [llength $inputBondPars] + [llength $inputAnglePars] } ] == 0 } { lappend errorList "No parameters to optimize" } \
    else {
        # check each set of bond parameters
        foreach ele $inputBondPars {
            lassign $ele typeDef fc eq
            if { [llength $typeDef] != 2 || [lindex $typeDef 0] eq "" || [lindex $typeDef 1] eq "" } { lappend errorList "Found inappropriate bond definition." }
            if { $fc eq "" || $fc < 0 || ![string is double $fc] } { lappend errorList "Found inappropriate force constant." }
            if { $eq eq "" || $eq < 0 || ![string is double $eq] } { lappend errorList "Found inappropriate b\u2080." }
        }

        # check each set of angle parameters
        foreach ele $inputAnglePars {
            lassign $ele typeDef fc eq
            if { [llength $typeDef] != 3 || [lindex $typeDef 0] eq "" || [lindex $typeDef 1] eq "" || [lindex $typeDef 2] eq "" } { lappend errorList "Found inappropriate angle definition." }
            if { $fc eq "" || $fc < 0 || ![string is double $fc] } { lappend errorList "Found inappropriate force constant." }
            if { $eq eq "" || $eq < 0 || ![string is double $eq] } { lappend errorList "Found inappropriate \u03F4." }
        }
    }
    
    # Advanced Settings
    #------------------
    # optimizer settings
    if { [lsearch -exact {downhill {simulated annealing}} $mode] == -1 } { lappend errorList "Unsupported optimization mode." } \
    else {
        # check tol
        if { $tol eq "" || $tol < 0 || ![string is double $tol] } { lappend errorList "Found inappropriate optimization tolerance setting." }
        # check geomweight
        if { $geomWeight eq "" || $geomWeight < 0 || ![string is double $geomWeight] } { lappend errorList "Found inappropriate optimization geometry weight." }
        # check enWeight
        if { $enWeight eq "" || $enWeight < 0 || ![string is double $enWeight] } { lappend errorList "Found inappropriate optimization energy weight." }
        # check downhill parameters
        if { $mode eq "downhill" } {
            if { $dhIter eq "" || ![string is integer $dhIter] } { lappend errorList "Found inappropriate Downhill Iter setting." }
        }
        # check simulated annealing parameters
        if { $mode eq "simulated annealing" } {
            if { $saT eq "" || ![string is double $saT] } { lappend errorList "Found inappropriate SA T setting." }
            if { $saTSteps eq "" || $saTSteps < 0 || ![string is integer $saTSteps] } { lappend errorList "Found inappropriate SA TSteps setting." }
            if { $saIter eq "" || $saTSteps < 0 || ![string is integer $saIter] } { lappend errorList "Found inappropriate SA Iter setting." }
        }
    }

    # check bounds
    if { $bondDev eq "" || ![string is double $bondDev] } { lappend errorList "Found inappropriate Bond Eq. Deviation value." }
    if { $bondLB eq "" || ![string is double $bondLB] } { lappend errorList "Found inappropriate Bond K lower bound." }
    if { $bondUB eq "" || ![string is double $bondUB] } { lappend errorList "Found inappropriate Bond K upper bound." }
    if { $bondLB >= $bondUB } { lappend errorList "Bond K lower bound is >= upper bound." }

    if { $angDev eq "" || ![string is double $angDev] } { lappend errorList "Found inappropriate Angle Eq. Deviation value." }
    if { $angLB eq "" || ![string is double $angLB] } { lappend errorList "Found inappropriate Angle K lower bound." }
    if { $angUB eq "" || ![string is double $angUB] } { lappend errorList "Found inappropriate Angle K upper bound." }
    if { $angLB >= $angUB } { lappend errorList "Angle K lower bound is >= upper bound." }



    # Other
    #------
    # make sure that the user can write to CWD (required for temporary files)
    if { ![file writable .] } { lappend errorList "Cannot write to CWD (required for temporary files)." }


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
proc ::ForceFieldToolKit::BondAngleOpt::optimize {} {
    # localize relevant variables
    variable psf
    variable pdb
    variable hessLog
    variable moleculeID
    
    variable minName
    variable namdbin
    variable outFile
    variable outFileName

    variable inputBondPars
    variable inputAnglePars
    variable parlist
    variable tempParName
    variable namdEnCommand
    variable parInProg
    
    variable uniqueTypes
    variable baInitial
    variable baIndList
    variable mapIndPar
    
    variable zmatqm
    variable zmatqmEff
    variable ringList
    variable atomTrans
    variable posTrans

    variable bondDev
    variable bondLB
    variable bondUB
    variable angDev
    variable angLB
    variable angUB

    variable bondscale
    variable dxBond
    variable angscale
    variable dxAng

    variable tol
    variable mode
    variable saT
    variable saTSteps
    variable saIter
    variable dhIter
    
    variable debug
    variable debugLog
    variable guiMode
    variable optCount
    
    #----------------------------------------

    if { $guiMode } {
        # run a sanity check
        if { ![::ForceFieldToolKit::BondAngleOpt::sanityCheck] } { return }
    }
    
    # open the log file
    set outFile [open $outFileName w]
    
    # if in debugging mode, open debugging output file
    # debug log filename is same as output filename with a .debug inserted
    if { $debug } {
        set debugLog [open "[file rootname $outFileName].debug.log" w]
        # run proc to print current settings, both to the console and the debugging log file
        ::ForceFieldToolKit::BondAngleOpt::printSettings stdout; flush stdout
        ::ForceFieldToolKit::BondAngleOpt::printSettings $debugLog; flush $debugLog
    }
    

    # load the Gaussian Log files from the hessian calculation
    if { $debug } { puts -nonewline $debugLog "loading hessian log file..."; flush $debugLog }
    set hessLogID [mol new $psf]
    # reTypeFromPSF has been depreciated
    #::ForceFieldToolKit::SharedFcns::reTypeFromPSF $psf $hessLogID 
    ::QMtool::use_vmd_molecule $hessLogID
    ::QMtool::load_gaussian_log $hessLog $hessLogID
    if { $debug } { puts $debugLog "DONE"; flush $debugLog }
    
    # store internal coordinates from the hessian calculation
    set zmatqm [::QMtool::get_internal_coordinates]
    if { $debug } {
        puts $debugLog "zmatqm:"
        foreach ele $zmatqm { puts $debugLog "  $ele" }
        flush $debugLog
    }
    
    # calculate effective QM PES and equilibrium geometry for all internal coordinates
    lassign [::ForceFieldToolKit::BondAngleOpt::computePESqm $hessLogID] zmatqmEff targetEnList targetGeomList
    
    if { $debug } {
        puts $debugLog "zmatqmEff:"
        foreach ele $zmatqmEff { puts $debugLog "  $ele" }
        flush $debugLog
        
        puts $debugLog "targetEnList:"
        foreach ele $targetEnList { puts $debugLog "  $ele" }
        flush $debugLog
        
        puts $debugLog "targetGeomList:"
        foreach ele $targetGeomList { puts $debugLog "  $ele" }
        flush $debugLog
    }

    
    # process the input bond and angle parameters (come from GUI)
    # note: inputBondPars format: { {typedef k B0} ... }
    # note: inputAnglePars format: { {typedef k theta0} ... }
    
    # initialize some lists
    set uniqueTypes {}; # format: { {bondTypeDef1} {bondTypeDef2} ... {angleTypeDef1} {angleTypeDef2} ... }
    set baInitial {}; # format: { bondFC1 bondEq1 bondFC2 bondEq2 ... angleFC1 angleEq1 angleFC2 angleEq2 ... }
    set baBounds {}; # format: { {{bond1Klb bond1Kub} {bond1EqMin bond1EqMax}} ... {{angle1Klb angle1Kub} {angle1EqMin angle1EqMax}} ... }

    # process bonds
    foreach ele $inputBondPars {
        lappend uniqueTypes [lindex $ele 0]
        set bondFC [lindex $ele 1]
        set bondEq [lindex $ele 2]
        lappend baInitial $bondFC $bondEq
        lappend baBounds [list $bondLB $bondUB] [list [expr { $bondEq - $bondDev} ] [expr { $bondEq + $bondDev }]]
    }
    
    # process angles
    foreach ele $inputAnglePars {
        lappend uniqueTypes [lindex $ele 0]
        #set angFC [expr { ($angUB+$angLB)/2.0 }]
        set angFC [lindex $ele 1]
        set angEq [lindex $ele 2]
        lappend baInitial $angFC $angEq
        set lbEq [expr { $angEq - $angDev }]
        set ubEq [expr { $angEq + $angDev }]
        if { $lbEq <= 0.0} {
           lappend baBounds [list $angLB $angUB] [list 0.0 $ubEq]
        } elseif { $ubEq > 180.0 } {
           lappend baBounds [list $angLB $angUB] [list $lbEq 180.0]
        } else {
           lappend baBounds [list $angLB $angUB] [list $lbEq $ubEq]
        }
    }
    

    
    if { $debug } {
        puts $debugLog "uniqueTypes:"
        foreach ele $uniqueTypes { puts $debugLog "  $ele" }
        flush $debugLog
        
        puts $debugLog "baInitial:"
        foreach {fc eq} $baInitial { puts $debugLog "  $fc\t$eq" }
        flush $debugLog
        
        puts $debugLog "baBounds:"
        foreach { kBounds eqBounds } $baBounds { puts $debugLog "  kBounds: $kBounds\teqBounds: $eqBounds" }
        flush $debugLog
    }
    
    # build a list for all bonds and angles that need checking
    set baIndList {}; # format: { {{bInd1 bInd2} {types} {type qmEqVal qmEnVal} } {{bInd1 bInd2} {types} ...} }

    # load the (re)typed molecule
    if { $debug } { puts -nonewline $debugLog "loading (re)typed molecule..."; flush $debugLog }
    set moleculeID [mol new $psf]
    mol addfile $pdb
    # reTypeFromPSF has been depreciated
    #::ForceFieldToolKit::SharedFcns::reTypeFromPSF $psf $moleculeID
    if { $debug } { puts $debugLog "DONE"; flush $debugLog }

    # get indicies for all bonds and angles in the molecule
    set indDefList {}
    foreach ele [topo getbondlist] { lappend indDefList $ele }   
    foreach ele [topo getanglelist] { lappend indDefList $ele }
    
    if { $debug } {
        puts $debugLog "indDefList:"
        foreach ele $indDefList { puts $debugLog "  $ele" }
        flush $debugLog
    }

    # create an ordered list of all bonds/angles in the zmat
    # to pick out specific target energies
    set zmatIndList {} 
    for {set i 1} {$i < [llength $zmatqmEff]} {incr i} {
       set tempInd [lindex $zmatqmEff $i 2]
       if { [llength $tempInd] <= 3} {
          lappend zmatIndList $tempInd
       }
    }
    
    if { $debug } {
        puts $debugLog "zmatIndList:"
        foreach ele $zmatIndList { puts $debugLog "  $ele" }
        flush $debugLog
    }    
  
    # mapIndPar gives the index in baInput of specific bond/angle
    # i.e. maps indicies to parameters
    set mapIndPar {}
    
    # search for these types in params-to-be-fitted
    foreach indDef $indDefList {
        # convert the index definition to a type definition
        set typeDef {}
        if { [llength $indDef] == 2 } {
           foreach ind [lrange $indDef 0 1] {
               set temp [atomselect top "index $ind"]
               lappend typeDef [$temp get type]
               $temp delete
           }

           set pos [lsearch $zmatIndList $indDef]
           if { $pos < 0 } { set pos [lsearch $zmatIndList [lreverse $indDef]] }

        } elseif { [llength $indDef] == 4 } {
           set indDef [lrange $indDef 1 3]
           foreach ind $indDef {
               set temp [atomselect top "index $ind"]
               lappend typeDef [$temp get type]
               $temp delete
           }
           set pos [lsearch $zmatIndList $indDef]
           if { $pos < 0 } { set pos [lsearch $zmatIndList [lreverse $indDef]] }
        }
        set typePos [lsearch $uniqueTypes $typeDef] 
        if { $typePos > -1 } {
           # baInitial is twice as long as uniqueTypes
                              #   indices   types      type (bond/angle/etc)                    qm eq. value                           qm energy 
           lappend baIndList [list $indDef $typeDef [list [lindex $zmatqmEff [expr $pos+1] 1] [lindex $targetGeomList $pos] [lindex $targetEnList $pos] ]]
#           lappend baIndList [list $indDef $typeDef [list [lindex $zmatqmEff [expr $pos+1] 1] [lindex $baInitial [expr 2*$typePos+1]] [lindex $targetEnList $pos] ]]
           lappend mapIndPar [expr 2*$typePos]
        } else {
           set typePos [lsearch $uniqueTypes [lreverse $typeDef]] 
           if { $typePos > -1 } {
              lappend baIndList [list $indDef [lreverse $typeDef] [list [lindex $zmatqmEff [expr $pos+1] 1] [lindex $targetGeomList $pos] [lindex $targetEnList $pos] ]]
#              lappend baIndList [list $indDef [lreverse $typeDef] [list [lindex $zmatqmEff [expr $pos+1] 1] [lindex $baInitial [expr 2*$typePos+1]] [lindex $targetEnList $pos] ]]
              lappend mapIndPar [expr 2*$typePos]
           }
        }
    }
    
    if { $debug } {
        puts $debugLog "mapIndPar: $mapIndPar"; flush $debugLog
        
        puts $debugLog "baIndList:"
        foreach ele $baIndList { puts $debugLog "  $ele" }
        flush $debugLog
    }
    
## NOT NEEDED (?)
    # ??? target qm energy value --cgm
##    set targetList {}
##    foreach ele $baIndList {
##        lappend targetlist [lindex $ele 2 2]
##    }
    
##    if { $debug } {
##        puts $debugLog "targetList: $targetlist"; flush $debugLog
##    }
    

    
    # Setup file(s) required by NAMD during the optimization
    
    # include a par filename that the optimize routine will modify on the fly
    # make sure that it is at the end of the parlist, so that it will overwrite
    # any preceding (unoptimized) parameters
    lappend parlist $tempParName
    
    if { $debug } {
        puts $debugLog "Final parList:"
        foreach ele $parlist { puts $debugLog "  $ele" }
        flush $debugLog
    }
    
    # write a NAMD configuration file used for minimization
    ::ForceFieldToolKit::SharedFcns::writeMinConf $minName $psf $pdb $parlist
    
    # build the namdEnergy cmd (will need access to this in other procs)
    set namdEnCommand "namdenergy -silent -psf [list $psf] -exe [list $namdbin] -all -sel \$sel -cutoff 1000"
    foreach par $parlist { 
        set namdEnCommand [concat $namdEnCommand "-par [list $par]"]
    }

    if { $debug } {
        puts $debugLog "namdEnCommand: $namdEnCommand"; flush $debugLog
    }
    
    
    #
    # SETUP THE OPTIMIZATION
    #
    
    if { $debug } { puts -nonewline $debugLog "\nSetting up Optimization..." }
    # reset the opt iteration counter
    set optCount 0

    # can do simple downhill or simulated annealing optimization
    if { $mode eq "downhill" } {
        set opt [optimization -downhill -tol $tol -iter $dhIter -function ::ForceFieldToolKit::BondAngleOpt::optBondsAngles]
    } elseif { $mode eq "simulated annealing"} {
        set opt [optimization -annealing -tol $tol -T $saT -iter $saIter -Tsteps $saTSteps -function ::ForceFieldToolKit::BondAngleOpt::optBondsAngles]
    } else {
        puts "ERROR - Unknown optimization mode.\nDownhill and Simulated Annealing are only currently supported modes"
        puts $outFile "ERROR - Unknown optimization mode.\nDownhill and Simulated Annealing are only currently supported modes"
        if { $debug } {
            puts $debugLog "ERROR - Unknown optimization mode."
            puts $debugLog "Only Downhill and Simulated Annealing are currently supported"
            puts $debugLog "Current mode set to: $mode"
        }
        return
    }
    
    # control simplex generation (helpful? not sure)
    # cgm -- consider converting "scale" variable to "simplexScale"
    # and grouping with bondscale and anglescale, so that we are keeping
    # optimizer-related variables together in an ~organized fashion
    # (noting that this would turn a local variable into a namespace variable)
    set scale 2.0
    $opt configure -bounds $baBounds
    $opt initsimplex $baInitial $scale

    if { $debug } {
        puts $debugLog "DONE"
        if { $mode eq "downhill" } {
            puts $debugLog "downhill: optimization -downhill -tol $tol -iter $dhIter -function ::ForceFieldToolKit::BondAngleOpt::optBondsAngles"
        } else {
            puts $debugLog "sa: optimization -annealing -tol $tol -T $saT -iter $saIter -Tsteps $saTSteps -function ::ForceFieldToolKit::BondAngleOpt::optBondsAngles"
        }
        flush $debugLog
    }


    #
    # Run the optimization
    #
    set result [$opt start]
    
    #
    # Process the optimization results
    #
    
    if { $debug } {
        puts $debugLog "\nRaw Result: $result"
        flush $debugLog
    }
    
    # result format: { bondFC1 bondEq1 bondFC2 bondEq2 ... angleFC1 angleEq1 ...}
    
    # write clearly formatted output to the log file for import into BuildPar
    # also build a result to pass back to the gui
    set guiReturnResults {}
    set guiReturnObj [format %.3f [lindex $result 1]]
    puts $outFile "\nFINAL PARAMETERS"
    set result [lindex $result 0]
    foreach { fc eq } $result type $uniqueTypes  {
    # write bonds
       if { [llength $type] == 2 } {
          puts $outFile "[list bond $type [format %.4f $fc] [format %.4f $eq]]"
          lappend guiReturnResults [list bond $type [format %.4f $fc] [format %.4f $eq]]
    # write angles
       } elseif { [llength $type] == 3 } {
          puts $outFile "[list angle $type [format %.4f $fc] [format %.4f $eq]]"
          lappend guiReturnResults [list angle $type [format %.4f $fc] [format %.4f $eq]]
       }
    }
    puts $outFile "END\n"
    
    # cleanup
    file delete $tempParName
    file delete $minName.conf
    file delete $minName.log
    foreach out {coor vel xsc} {
       file delete $minName.$out
       file delete $minName.$out.BAK
       file delete $minName.restart.$out
       file delete $minName.restart.$out.old
    }
    close $outFile
    mol delete $moleculeID
    mol delete $hessLogID
    
    if { $debug } {
        puts $debugLog ""
        puts $debugLog "DONE"
        puts $debugLog ""
        flush $debugLog
        close $debugLog
    }
    
    # return some results if we're running from the GUI
    if { $guiMode } { return [list $guiReturnResults $guiReturnObj] }
}
#======================================================
proc ::ForceFieldToolKit::BondAngleOpt::optBondsAngles { baInput } {
    # baInput format: { bondFC1 bondEq1 bondFC2 bondEq2 ... angleFC1 angleEq1 ...}
    
    # localize necessary variables
    variable psf
    variable moleculeID
    variable tempParName
    variable outFile
    variable minName
    variable namdbin
    
    variable uniqueTypes 
    variable baInitial 
    variable baIndList
    variable namdEnCommand
    variable mapIndPar

    variable bondscale
    variable angscale
    variable enscale
    variable geomWeight
    variable enWeight
    
    variable debug
    variable debugLog
    variable guiMode
    variable optCount

    #-------------------------------------------------------
    
    if { $debug } {
        puts $debugLog ""
        puts $debugLog "baInput:"
        foreach {fc eq} $baInput { puts $debugLog "  $fc\t$eq" }
        flush $debugLog
    }

    # have write the parameter file in optimize function
    # could also build a parameter list and use SharedFcns::writeParFile
    # this method is more direct, however
    
    # write a PAR File for the current set of bond/angle parameters
    set outParam [open "$tempParName" w]

    puts $outParam "BONDS"
    foreach { fc eq } $baInput type $uniqueTypes {
        if { [llength $type] == 2 } { puts $outParam "$type $fc $eq" }
    }
    puts $outParam "\nANGLES"
    foreach { fc eq } $baInput type $uniqueTypes  {
       if { [llength $type] == 3 } { puts $outParam "$type $fc $eq" }
    }
    puts $outParam "\nEND"
    close $outParam
    # done writing the parameter file
    
    # this will generate a lot of output, but we're likely in debugging mode for a reason
    if { $debug } {
        puts $debugLog ""
        puts $debugLog "Current PAR file:"
        set inFile [open $tempParName r]
        while { ![eof $inFile] } { puts $debugLog "[gets $inFile]" }
        close $inFile
        flush $debugLog
    }

    # run the NAMD minimization
    ::ExecTool::exec $namdbin $minName.conf
    mol addfile $minName.coor $moleculeID
    
    if { $debug } {
        puts $debugLog ""
        puts $debugLog "NAMD run complete"
        puts $debugLog ""
        flush $debugLog
    }

    # setup the objective function components
    set geomObj 0.0; set enObj 0.0; set totalObj 0.0
    
    # get the minimized bonds/angles, add to Eq objective value
    # baIndlist { {{bInd1 bInd2} {typedef} {bond|angle qmEqVal qmEnVal} } {{bInd1 bInd2} {typedef} ...} }
    # measure bonds, angles, compare to frame 0 (QM length) 
    set mmEnList [::ForceFieldToolKit::BondAngleOpt::computePESmm $moleculeID $baIndList $namdEnCommand]  

#    set targetList {}
#    foreach ele $baIndList {
#        lappend targetlist [lindex $ele 2 2]
#    }
###   puts "baIndlist $baIndList\n"
###   puts "mmEnList: $mmEnList\n"
    

    set i 0
    foreach entry $baIndList mmEn $mmEnList {
######        puts "mmEn: [lindex $mmEn 0]  qmEn?: [lindex $entry 2 2 0] "        
        set baCur [lindex $entry 0]
        if { [llength $baCur] == 2 } {
            set baCurVal [measure bond "[lindex $baCur 0] [lindex $baCur 1]" last]
            set gscale $bondscale
        } elseif { [llength $baCur] == 3 } {
            set baCurVal [measure angle "[lindex $baCur 0] [lindex $baCur 1] [lindex $baCur 2]" last]          
            set gscale $angscale
        }
        puts $outFile [format "  $baCur [lindex $entry 1] Eq: %2.3f Fc: %2.3f GeomDelta: %2.3f EnDelta: %2.3f" [lindex $baInput [expr { [lindex $mapIndPar $i] + 1 }]] [lindex $baInput [lindex $mapIndPar $i]] [expr { $baCurVal-[lindex $entry 2 1] }] [expr { [lindex $mmEn 0] - [lindex $entry 2 2 0] }]]
        
        set geomObj [expr $geomObj + $geomWeight*pow(($baCurVal - [lindex $entry 2 1])/$gscale,2)]

        set enObj [expr $enObj + 0.5*$enWeight*pow(([lindex $mmEn 0] - [lindex $entry 2 2 0])/$enscale,2)]
        set enObj [expr $enObj + 0.5*$enWeight*pow(([lindex $mmEn 1] - [lindex $entry 2 2 1])/$enscale,2)]

##        set enObj [expr $enObj + $enWeight*pow(($mmEn - [lindex $entry 2 2])/$enscale,2)]
        incr i
    }


    set totalObj [expr $enObj + $geomObj]
    puts $outFile [format "\nCurrent objective value: %.6f ( En: %.6f  Geom: %.6f )\n" $totalObj $enObj $geomObj]

   # clean up before next minimization
    animate delete beg [expr [molinfo $moleculeID get numframes] -1]

    if { $debug } {
        puts $debugLog ""
        puts $debugLog [format "Current objective value: %.6f ( En: %.6f  Geom: %.6f )" $totalObj $enObj $geomObj]
        puts $debugLog ""
        flush $debugLog
    }

    # update the status in the gui
    if { $guiMode } {
        incr optCount
        set ::ForceFieldToolKit::gui::baoptStatus "Running...Optimizing(iter:$optCount)"
        update idletasks
    }

    return $totalObj

}
#======================================================
proc ::ForceFieldToolKit::BondAngleOpt::buildScript { scriptFilename } {

    # localize variables
    #--------------------------
    # variables from optimize proc
    variable psf
    variable pdb
    variable hessLog
    variable moleculeID    
    variable minName
    variable namdbin
    variable outFile
    variable outFileName
    variable inputBondPars
    variable inputAnglePars
    variable parlist
    variable tempParName
    variable namdEnCommand
    variable parInProg    
    variable uniqueTypes
    variable baInitial
    variable baIndList
    variable mapIndPar    
    variable zmatqm
    variable zmatqmEff
    variable ringList
    variable atomTrans
    variable posTrans
    variable bondDev
    variable bondLB
    variable bondUB
    variable angDev
    variable angLB
    variable angUB
    variable bondscale
    variable dxBond
    variable angscale
    variable dxAng
    variable tol
    variable mode
    variable saT
    variable saTSteps
    variable saIter
    variable dhIter    
    variable debug
    variable debugLog
    variable guiMode
    variable optCount
    
    # (relevant) variables from optBondsAngles proc
    variable enWeight
    variable geomWeight
    variable enscale
    
    
    #--------------------------
    
    set scriptFile [open $scriptFilename w]
    # load required packages
    puts $scriptFile "\# Load required packages"
    puts $scriptFile "package require namdenergy"
    puts $scriptFile "package require optimization"
    puts $scriptFile "package require topotools"
    puts $scriptFile "package require forcefieldtoolkit"

    # check for tk, required by the qmtool dependency
    # NOTE: this prevents the build script from being run with VMD in text mode
    puts $scriptFile "if { !\[info exists tk_version\] } { puts \"ffTK BondAngleOpt build script cannot execute without tk due to dependency in QMtool.\"; exit }"
    
    # Variables to set
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::psf $psf"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::pdb $pdb"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::hessLog $hessLog"
    # moleculeID is set internally
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::minName $minName"

    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::namdbin $namdbin"
    # outfile is set internally
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::outFileName $outFileName"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::inputBondPars [list $inputBondPars]"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::inputAnglePars [list $inputAnglePars]"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::parlist [list $parlist]"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::tempParName $tempParName"
    # namdEnCommand is set internally
    #puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::namdEnCommand $namdEnCommand"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::parInProg $parInProg"
    # uniqueTypes is set internally
    # baInitial is set internally
    # baIndList is set internally
    # mapIndPar is set internally
    # zmatqm is set internally
    # zmatqmEff is set internally
    # ringList is set internally
    # atomTrans is set internally
    # posTrans is set internally
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::bondDev $bondDev"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::bondLB $bondLB"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::bondUB $bondUB"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::angDev $angDev"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::angLB $angLB"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::angUB $angUB"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::bondscale $bondscale"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::dxBond $dxBond"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::angscale $angscale"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::dxAng $dxAng"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::tol $tol"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::mode $mode"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::saT $saT"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::saTSteps $saTSteps"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::saIter $saIter"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::dhIter $dhIter"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::debug $debug"
    # debugLog is hardcoded
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::guiMode 0"
    # optCount is irrelevant when guiMode = 0
    
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::enWeight $enWeight"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::geomWeight $geomWeight"
    puts $scriptFile "set ::ForceFieldToolKit::BondAngleOpt::enscale $enscale"

    #--------------------------
    
    # launch the optimization
    puts $scriptFile "\n\# Run the Optimization"
    puts $scriptFile "::ForceFieldToolKit::BondAngleOpt::optimize"
    puts $scriptFile "\n\# Return gracefully"
    puts $scriptFile "return 1"
    
    # wrap up
    close $scriptFile
    return
}
#======================================================
proc ::ForceFieldToolKit::BondAngleOpt::printSettings { outfile } {
    # a tool to print all settings passsed to the bonds/angles
    # optimization routine, and any relevant settings in the 
    # BondAngleOpt namespace that will be accessed during the
    # optimization
    
    puts $outfile "=================================="
    puts $outfile " Bond/Angle Optimization Settings "
    puts $outfile "=================================="
    puts $outfile ""
    
    puts $outfile "INPUT"
    puts $outfile "psf: $::ForceFieldToolKit::BondAngleOpt::psf"
    puts $outfile "pdb: $::ForceFieldToolKit::BondAngleOpt::pdb"
    puts $outfile "Parameter Files:"
    puts $outfile "\tIn-Progress PAR File: $::ForceFieldToolKit::BondAngleOpt::parInProg"
    puts $outfile "\tAll PAR Files (in-progress + associated):"
    foreach par $::ForceFieldToolKit::BondAngleOpt::parlist { puts $outfile "\t\t$par" }
    puts $outfile "namdbin: $::ForceFieldToolKit::BondAngleOpt::namdbin"
    puts $outfile "outFileName: $::ForceFieldToolKit::BondAngleOpt::outFileName"
    puts $outfile "----------------------------------"
    puts $outfile ""
    
    puts $outfile "PARAMETERS TO OPTIMIZE"
    puts $outfile "Input Bond Pars:"
    foreach ele $::ForceFieldToolKit::BondAngleOpt::inputBondPars { puts $outfile "\t$ele" }
    puts $outfile "Input Angle Pars:"
    foreach ele $::ForceFieldToolKit::BondAngleOpt::inputAnglePars { puts $outfile "\t$ele" }
    puts $outfile "----------------------------------"
    puts $outfile ""
    
    puts $outfile "ADVANCED SETTINGS"
    puts $outfile "tol: $::ForceFieldToolKit::BondAngleOpt::tol"
    puts $outfile "Geom. Weight: $::ForceFieldToolKit::BondAngleOpt::geomWeight"
    puts $outfile "En. Weight: $::ForceFieldToolKit::BondAngleOpt::enWeight"
    puts $outfile "optimization mode: $::ForceFieldToolKit::BondAngleOpt::mode"
    puts $outfile "Downhill settings: Iterations-$::ForceFieldToolKit::BondAngleOpt::dhIter"
    puts $outfile "Simulated Annealing settings: Temp-$::ForceFieldToolKit::BondAngleOpt::saT, Steps-$::ForceFieldToolKit::BondAngleOpt::saTSteps, Iterations-$::ForceFieldToolKit::BondAngleOpt::saIter"
    puts $outfile "Adv. Bonds: Eq. Dev.-$::ForceFieldToolKit::BondAngleOpt::bondDev, K LB-$::ForceFieldToolKit::BondAngleOpt::bondLB UB-$::ForceFieldToolKit::BondAngleOpt::bondUB"
    puts $outfile "Adv. Angles: Eq. Dev.-$::ForceFieldToolKit::BondAngleOpt::angDev, K LB-$::ForceFieldToolKit::BondAngleOpt::angLB UB-$::ForceFieldToolKit::BondAngleOpt::angUB"
    puts $outfile "debugging: $::ForceFieldToolKit::BondAngleOpt::debug"
    puts $outfile "----------------------------------"
    puts $outfile ""
    
    puts $outfile "HARDCODED SETTINGS (set by init)"
    puts $outfile "minName: $::ForceFieldToolKit::BondAngleOpt::minName"
    puts $outfile "tempParName: $::ForceFieldToolKit::BondAngleOpt::tempParName"
    puts $outfile "bondscale: $::ForceFieldToolKit::BondAngleOpt::bondscale"
    puts $outfile "angscale: $::ForceFieldToolKit::BondAngleOpt::angscale"
    puts $outfile "enscale: $::ForceFieldToolKit::BondAngleOpt::enscale"
    puts $outfile "dxBond: $::ForceFieldToolKit::BondAngleOpt::dxBond"
    puts $outfile "dxAng: $::ForceFieldToolKit::BondAngleOpt::dxAng"
    puts $outfile "----------------------------------"    
    
    flush $outfile
    
}
#======================================================

#======================================================
#                   -- NEW PROCS -- 
# EVERY THING BELOW STILL NEEDS TESTING AND CLEANING
#======================================================

#======================================================
## original was compute_force_constants_from_inthessian in paratool
proc ::ForceFieldToolKit::BondAngleOpt::computePESqm { molid { BAonly 1} } {
   variable zmatqm

   variable EnList {}
   variable zmattargetk {}
   variable targetklist {}
   variable targetx0list {}
   variable depcoorlist {}
   set fclist {}
   set intcoorlist {}

   variable dxAng
   variable dxBond

   set inthessian_kcal [::QMtool::get_internal_hessian_kcal]
   set zmatqm [::QMtool::get_internal_coordinates]
   
   for {set i 1} {$i<[llength $zmatqm]} {incr i} {
      #set targetk  [lindex $zmat $i 4 0]
      set targetx0 [lindex $zmatqm $i 3]

      set entry [lindex $zmatqm $i]
      set type  [lindex $entry 1]

      # typically we only care about bonds/angles
      if { $BAonly && ![string match "*bond" $type] && ![regexp "angle|lbend" $type] } { continue }

      set dx $dxBond
      if {[regexp "angle|lbend|dihed" $type]} {
	 set dx $dxAng
      }

      # Generate two frames that represent a distortion of the current internal coordinate
      # in both directions. Since our set of internal coords is redundant, some coordinates
      # cannot be disturbed without changing other coordinates, too. These coordinates are
      # mutually dependent. We can then use measure $type to determine how much
      # other coordinates have changed with the perturbation. 

      # A general rule which coords are dependently distorted is as follows:
      # Bonds are all independent unless they are part of a ring. In that case everything
      # is kept fixed except the two atoms defining the bond.
      # Consequently the dependent conformations are all neighboring bonds and all angles these
      # atoms are involved in.
      # For angles the dependent coordinates are all other angles centered around the middle
      # atom. In case of a four-valent middle atom, the dihedrals ending in one (and only one)
      # leg of the angle are also dependent.
      # Impropers are always dependent on all angles made from the same atoms and all dihedrals
      # involving the same angles.

      set numframes [molinfo $molid get numframes]
      set icenter [expr {$numframes-1}]
      set ilower  [expr {$numframes}]
      set iupper  [expr {$numframes+1}]

      # make distortion assumes last frame of $molid is the minimized structure to distort
      # when done, the last two frames represent +/- distortions

## new!
##        foreach {x h1} [ ::ForceFieldToolKit::Distortion::make $molid $type [lindex $entry 2] $dx ] {}
      ::ForceFieldToolKit::Distortion::make $molid $type [lindex $entry 2] $dx

###puts "ran make on $molid $type [lindex $entry 2] $dx"

## old!
##      foreach {x h1} [make_distortion $molid $type [array get pos] [array get atom] -dx $dx] {}

      set rad2deg 57.2957795131;   # [expr 180.0/3.14159265358979]
      set deg2rad 0.0174532925199; # [expr 3.14159265358979/180.0]

      set depcoor {}
      set energy {0 0 0}
###      #set h1 $dx
###      if {[regexp "angle|lbend|dihed|imprp" $type]} {
###         puts "huh??? $h1"
###	 set h1 [expr {$deg2rad*$h1}]
###      }

      for {set j 1} {$j<[llength $zmatqm]} {incr j} {
	 # Make sure to fetch kc from lower diagonal
	 if {$i>=$j} {
	    set kc [lindex $inthessian_kcal [expr {$i-1}] [expr {$j-1}]]
	 } else {
	    set kc [lindex $inthessian_kcal [expr {$j-1}] [expr {$i-1}]]
	 }
	 lappend kclist $kc

	 set geom [lindex $zmatqm $j 1]
	 if {$geom=="lbend"} { set geom "angle" }
	 set xcenter [measure $geom [lindex $zmatqm $j 2] molid $molid frame $icenter]
	 set xupper  [measure $geom [lindex $zmatqm $j 2] molid $molid frame $iupper]
	 set xlower  [measure $geom [lindex $zmatqm $j 2] molid $molid frame $ilower]

### impropers too?
	 set h2 [format "%.3f" [expr {($xupper-$xcenter)}]]
         if { $h2 > 180.0 && $geom == "dihed" } {
            set h2 [expr $h2 - 360.0]
         }
            if { $h2 < -180.0 && $geom == "dihed" } {
            set h2 [expr $h2 + 360.0]
         }

	 if {abs($h2)<0.01} { continue }
	 if {[regexp "angle|lbend|dihed" $geom]} {
	    if {[format "%.2f" $h2]==180.0} { continue }
	    set h2 [expr {$h2*$deg2rad}]
	 }

	 # Since h2 is nonzero we are in presence of a dependent coordinate.
	 lappend depcoor $j

      }

      lappend depcoorlist $depcoor

      # Force constant f''(x)/2 = 0.5*(f(x-h) - 2f(x) + f(x+h))/h^2
      #foreach {Elower Ecenter Eupper} $energy {break}
      #puts " Elower=$Elower; Ecenter=$Ecenter; Eupper=$Eupper"
      #set rawk [expr {0.5*($Elower - 2.0*$Ecenter + $Eupper)/pow($h1,2)}]


      # Actually we want to generate a list of raw FCs and refine and scale them (and periodify the diheds).
      # For now this is ok.
      #lappend fclist $rawk
      lappend fclist [lindex $inthessian_kcal [expr {$i-1}] [expr {$i-1}]]

      # Get the coupling constants and compute energies left and right of equilibrium in order
      # to calculate the target potential surface for this motion.
      set energy {0 0 0}
      foreach rdep $depcoor {
	 set geom [lindex $zmatqm $rdep 1]
	 if {$geom=="lbend"} { set geom "angle" }
	 set xcenter [measure $geom [lindex $zmatqm $rdep 2] molid $molid frame $icenter]
	 set xupper  [measure $geom [lindex $zmatqm $rdep 2] molid $molid frame $iupper]
	 set xlower  [measure $geom [lindex $zmatqm $rdep 2] molid $molid frame $ilower]

### upper distortion
	 set h1 [expr {($xupper-$xcenter)}]
         if { $h1 > 180.0 && $geom == "dihed" } { set h1 [expr $h1 - 360.0] }
         if { $h1 < -180.0 && $geom == "dihed" } { set h1 [expr $h1 + 360.0] }

	 if {[regexp "angle|lbend|dihed" $geom]} {
	    if {[format "%.3f" $h1]==180.0} { continue }
	    set h1 [expr {$deg2rad*$h1}]
	 }
         set h1up $h1
###	 if {$h1up==0.0} { continue }
### lower distortion
	 set h1 [expr {($xcenter-$xlower)}]
         if { $h1 > 180.0 && $geom == "dihed" } { set h1 [expr $h1 - 360.0] }
         if { $h1 < -180.0 && $geom == "dihed" } { set h1 [expr $h1 + 360.0] }

	 if {[regexp "angle|lbend|dihed" $geom]} {
	    if {[format "%.3f" $h1]==180.0} { continue }
	    set h1 [expr {$deg2rad*$h1}]
	 }
         set h1low $h1
	 if {$h1low==0.0 && $h1up==0.0} { continue }

	 foreach cdep $depcoor {
	    # Make sure to fetch kc from lower diagonal
	    if {$rdep>=$cdep} {
	       set kc [lindex $inthessian_kcal [expr {$rdep-1}] [expr {$cdep-1}]]
	    } else {
	       set kc [lindex $inthessian_kcal [expr {$cdep-1}] [expr {$rdep-1}]]
	    }

	    if {$kc==0.0} { continue }

	    set geom [lindex $zmatqm $cdep 1]
	    if {$geom=="lbend"} { set geom "angle" }
	    set xcenter [measure $geom [lindex $zmatqm $cdep 2] molid $molid frame $icenter]
	    set xupper  [measure $geom [lindex $zmatqm $cdep 2] molid $molid frame $iupper]
	    set xlower  [measure $geom [lindex $zmatqm $cdep 2] molid $molid frame $ilower]
## upper
	    set h2 [expr {($xupper-$xcenter)}]
         if { $h2 > 180.0 && $geom == "dihed" } { set h2 [expr $h2 - 360.0] }
         if { $h2 < -180.0 && $geom == "dihed" } { set h2 [expr $h2 + 360.0] }

	    if {[regexp "angle|lbend|dihed" $geom]} {
	       if {[format "%.3f" $h2]==180.0} { continue }
	       set h2 [expr {$h2*$deg2rad}]
	    }
            set h2up $h2
## lower
	    set h2 [expr {($xupper-$xcenter)}]
         if { $h2 > 180.0 && $geom == "dihed" } { set h2 [expr $h2 - 360.0] }
         if { $h2 < -180.0 && $geom == "dihed" } { set h2 [expr $h2 + 360.0] }

	    if {[regexp "angle|lbend|dihed" $geom]} {
	       if {[format "%.3f" $h2]==180.0} { continue }
	       set h2 [expr {$h2*$deg2rad}]
	    }
            set h2low $h2

	    if {$h2up==0.0 && $h2low==0} { continue }
	    set eUP [expr {($kc)*$h1up*$h2up}]
	    set eLOW [expr {($kc)*$h1low*$h2low}]

	    set energy [vecadd $energy [list $eLOW 0.0 $eUP]]
	 }
      }

      # Now we can estimate the force constant for this potential by simply taking a numerical
      # derivative. 
      # Force constant f''(x)/2 = 0.5*(f(x-h) - 2f(x) + f(x+h))/h^2
      foreach {Elower Ecenter Eupper} $energy {break}
      set targetk [expr 0.5*($Elower - 2.0*$Ecenter + $Eupper)/pow($dx,2)]
      if {[regexp "angle|lbend|dihed|imprp" $type]} {
	 set targetk [expr pow($rad2deg,2)*$targetk]
      }

      lappend targetklist $targetk
      lappend targetx0list $targetx0
      lappend EnList [list $Elower $Eupper]

      # Delete the tmp frames

###   animate write dcd test-[lindex $zmatqm $i 0].dcd $molid

      animate delete beg $numframes $molid ; 
   }

   # energies for each int coord. distortion to be fitted

   set zmatqmEff [assign_fc_zmat $targetklist $molid $zmatqm $BAonly]
   return [list $zmatqmEff $EnList $targetx0list]

}
#======================================================
proc ::ForceFieldToolKit::BondAngleOpt::computePESmm { molid baIndList namdEn { BAonly 1} } {

   variable EnList {}
   set intcoorlist {}

   variable dxBond
   variable dxAng

   set minFrame [expr [molinfo $molid get numframes] -1]

## baIndList: { {{bInd1 bInd2} {types} {type qmEqVal qmEnVal} } {{bInd1 bInd2} {types} ...} }

   for {set i 0} {$i<[llength $baIndList]} {incr i} {

      set entry [lindex $baIndList $i]
      set indices [lindex $entry 0]
      set type [lindex $entry 2 0]

      # typically we only care about bonds/angles
      if { $BAonly && ([llength $indices] > 3 || [llength $indices] < 2)} { continue }

      set dx $dxBond
      if { [llength $indices] == 3 } {
	 set dx $dxAng
      }

## new!
::ForceFieldToolKit::Distortion::make $molid $type $indices $dx $minFrame 
## maybe? incr minFrame 2

##     foreach {x h1} [ ::ForceFieldToolKit::Distortion::make $molid $type $indices $dx $minFrame ] {}
## old!
##      foreach {x h1} [make_distortion $molid $type [array get pos] [array get atom] -dx $dx -startframe $minFrame] {}
   }

   # namd energy
   set sel [atomselect $molid all]
   set energyout [eval $namdEn]
   set minEn [lindex $energyout $minFrame end]

###   animate write dcd mm-all.dcd beg $minFrame $molid

   set mmEnList {}
   for {set i [expr $minFrame + 1]} {$i < [llength $energyout]} {incr i 2} {
      set enLOW [expr [lindex $energyout $i end] - $minEn]
      set enUP [expr [lindex $energyout [expr $i + 1] end] - $minEn]
      lappend mmEnList [list $enLOW $enUP]
   }

   animate delete beg [expr $minFrame + 1] end -1 $molid

   # return energy list
   return $mmEnList

}
#======================================================
### taken from paratool
#########################################################
# Assign the force constants to internal coordinates.   #
# Takes the hessian matrix in internal coordinates.     #
# Harmonic force constants for diheds are translated to #
# periodic potentials.                                  #
#########################################################
proc ::ForceFieldToolKit::BondAngleOpt::assign_fc_zmat { fclist molid zmatIn BAonly } {

   set num -1
   foreach entry $zmatIn {
      # Skip header
      if {$num==-1} { incr num; continue }
      set fc [lindex $fclist $num]
      set type [lindex $entry 1]

      if {[string match "dihed" $type] && !$BAonly} {
	 set delta 180.0
	 set n [::QMtool::get_dihed_periodicity $entry $molid]
	 set dihed [lindex $entry 3]
	 set pot [expr {1+cos($n*$dihed/180.0*3.14159265)}]

	 if {$pot<1.0} { set delta 0.0 }
	 set pot [expr {1+cos(($n*$dihed-$delta)/180.0*3.14159265)}]

	 if { $pot>0.1} { 
	    # If the equilib energy would be higher than 5% of the barrier height we choose the
	    # exact actual angle for delta.
	    # Since the minimum of cos(x) is at 180 deg we need to use an angle relative to 180.
	    set delta [expr {180.0+$dihed}]
	    puts "[lindex $entry 0] pot=$pot dihed=$dihed delta=$delta"
	 }
	 set fc [expr {$fc/double($n*$n)}]
	 lset zmatIn [expr {$num+1}] 4 [list $fc $n $delta]
      } else {
	 lset zmatIn [expr {$num+1}] 4 [list $fc [lindex $entry 3]]
      }
      lappend newfclist $fc
      lset zmatIn [expr {$num+1}] 5 "[regsub {[QCRMA]} [lindex $zmatIn [expr {$num+1}] 5] {}]Q"
      incr num
   }
   ## havefc, havepar
   lset zmatIn 0 6 1
   lset zmatIn 0 7 1

   return $zmatIn
}
#======================================================
### ALL PROCS BELOW THIS LINE ARE DEPRECATED
### KEEP FOR NOW UNTIL CONFIRMED 
### taken from paratool
### holding off on a full cleanup until we're sure we 
### trust it
proc ::ForceFieldToolKit::BondAngleOpt::make_distortion {molid type arrpos arratom args} {
## allowable types: bond angle lbend dihed imprp

   variable atomtrans
   variable postrans

   # default tiny values
   variable hbond  0.2;  #
   variable hangle 5.0; #0.2
   variable hdihed 5.0; #2.0 
   variable himprp 0.2

   variable ringlist [::util::find_rings $molid]

   array set pos  $arrpos
   array set atom $arratom
   array set base $arratom

   set dx {}
   set i [lsearch $args "-dx"]
   if {$i>=0 && $i<[llength $args]-1} {
      set dx [lindex $args [expr {$i+1}]]
   }

   set startframe [expr [molinfo $molid get numframes] - 1]
   set i [lsearch $args "-startframe"]
   if {$i>=0 && $i<[llength $args]-1} {
      set startframe [lindex $args [expr {$i+1}]]
   }


   set i [lsearch $args "-basemolindexes"]
   if {$i>=0 && $i<[llength $args]-1} {
      set i 0
      foreach ind [lindex $args [expr {$i+1}]] {
	 set base($i) $ind
	 incr i
      }
   }

### put it out here
   # Construct selections left and right of the conformation
   if {[string match "*bond" $type]} {
      # FIXME: I'm just using the first ring in case the bond belongs to 2 rings.
      set inring [lindex [bond_in_ring $base(0) $base(1)] 0]
##      set inring [lindex [::Paratool::bond_in_ring $base(0) $base(1)] 0]

      if {[llength $inring]} {
	 set vis {}; # will contain the two ring neighbors of atom(0)
	 set sel [atomselect $molid "index $base(0)"]
	 foreach nb [join [$sel getbonds]] {
	    if {[lsearch [lindex $ringlist $inring] $nb]>=0} {
	       # This neighbor of atom(0) is part of the ring
	       lappend vis $atomtrans($nb)
	       if {$nb!=$base(1)} { set base(-1) $nb }
	    }
	 }
	 $sel delete
	 set atom(-1) [::util::ldiff $vis $atom(1)]
	 set base(-2) [lindex [::util::reorder_ring [lindex $ringlist $inring] $base(0) $base(-1)] 2]
	 set atom(-2) $atomtrans($base(-2))
	 set pos(-1)  $postrans($base(-1))
	 set pos(-2)  $postrans($base(-2))
	 set indexes1 [::util::ldiff [::util::bondedsel $molid $atom(0) $vis -all] $vis]
	 set vis {}
	 set sel [atomselect $molid "index $base(1)"]
	 foreach nb [join [$sel getbonds]] {
	    if {[lsearch [lindex $ringlist $inring] $nb]>=0} {
	       lappend vis $atomtrans($nb)
	       if {$nb!=$base(0)} { set base(2) $nb }
	    }
	 }
	 $sel delete
	 set atom(2) [::util::ldiff $vis $atom(0)]
	 set base(3) [lindex [::util::reorder_ring [lindex $ringlist $inring] $base(1) $base(2)] 2]
	 set atom(3) $atomtrans($base(3))
	 set pos(2)  $postrans($base(2))
	 set pos(3)  $postrans($base(3))
	 set indexes2 [::util::ldiff [::util::bondedsel $molid $atom(1) $vis -all] $vis]
###	 set type bondring
############ added
         set indexes1 {}
         set indexes2 {}
	 set sel [atomselect $molid "index $base(0)"]
	 foreach nb [join [$sel getbonds]] {
	    if {[lsearch [lindex $ringlist $inring] $nb]<0} {
	       # This neighbor of atom(0) is NOT part of the ring
	       lappend nonring $atomtrans($nb)
	    }
	 }
	 $sel delete
         foreach ind $nonring {
            lappend indexes1 [::util::bondedsel $molid $atom(0) $ind -all]
         }
         set nonring {}
	 set sel [atomselect $molid "index $base(1)"]
	 foreach nb [join [$sel getbonds]] {
	    if {[lsearch [lindex $ringlist $inring] $nb]<0} {
	       # This neighbor of atom(1) is NOT part of the ring
	       lappend nonring $atomtrans($nb)
	    }
	 }
	 $sel delete
         foreach ind $nonring {
            lappend indexes2 [::util::bondedsel $molid $atom(1) $ind -all]
         }
        ## ring selections different, want to include starting atom
        set sel1 [atomselect $molid "index $indexes1 or index $atom(0)"]
        set sel2 [atomselect $molid "index $indexes2 or index $atom(1)"]      
   # now can distort just like normal bond
         set type bond
############
      } else {
	 set indexes1 [::util::bondedsel $molid $atom(0) $atom(1) -all]
	 set indexes2 [::util::bondedsel $molid $atom(1) $atom(0) -all]
      ## flipped the order of the excluded index, FIXED! (not ring)
      ## also flipped selection to make distortion go -/+!
      set sel2 [atomselect $molid "index $indexes1 and not index $atom(0)"]
      set sel1 [atomselect $molid "index $indexes2 and not index $atom(1)"]      
      }
      set del [expr {$hbond/2.0}]
   } elseif {[regexp "angle|lbend" $type]} {
##      set inring [::Paratool::angle_in_ring $base(0) $base(1) $base(2)]
      set inring [angle_in_ring $base(0) $base(1) $base(2)]
      if {[llength $inring]} {
	 set vis [list $atom(0) $atom(2)]
	 set indexes2 [::util::ldiff [::util::bondedsel $molid $atom(1) $vis -all] $vis]
	 set sel2 [atomselect $molid "index $indexes2 $atom(1)"]
	 set sel1 [atomselect $molid "(not index $indexes2)"]
	 set indexes1 [$sel1 list]
	 set type anglering
############ added
         set indexes1 {}
         set indexes2 {}
	 set sel [atomselect $molid "index $base(0)"]
	 foreach nb [join [$sel getbonds]] {
	    if {[lsearch [lindex $ringlist $inring] $nb]<0} {
	       # This neighbor of atom(0) is NOT part of the ring
	       lappend nonring $atomtrans($nb)
	    }
	 }
	 $sel delete
         foreach ind $nonring {
            lappend indexes1 [::util::bondedsel $molid $atom(0) $ind -all]
         }
         set nonring {}
	 set sel [atomselect $molid "index $base(2)"]
	 foreach nb [join [$sel getbonds]] {
	    if {[lsearch [lindex $ringlist $inring] $nb]<0} {
	       # This neighbor of atom(2) is NOT part of the ring
	       lappend nonring $atomtrans($nb)
	    }
	 }
	 $sel delete
         foreach ind $nonring {
            lappend indexes2 [::util::bondedsel $molid $atom(2) $ind -all]
         }
        ## ring selections different, want to include starting atom
        set sel1 [atomselect $molid "index $indexes1 or index $atom(0)"]
        set sel2 [atomselect $molid "index $indexes2 or index $atom(2)"]      
   # now can distort just like normal angle
         set type angle
############

      } else {
         # changed order of each bondedsel calculation!
	 set indexes1 [::util::bondedsel $molid $atom(1) $atom(0) -all]
	 set indexes2 [::util::bondedsel $molid $atom(1) $atom(2) -all]
	 set sel1 [atomselect $molid "index $indexes1 and not index $atom(1)"]
	 set sel2 [atomselect $molid "index $indexes2 and not index $atom(1)"]      
      }
      set del [expr {$hangle/2.0}]
   } elseif {$type=="dihed"} {
      set indexes1 [::util::bondedsel $molid $atom(1) $atom(2) -all]
      set indexes2 [::util::bondedsel $molid $atom(2) $atom(1) -all]
    ### temporary fix to make distortion go -/+
    ### still doesn't work for rings!
      set sel2 [atomselect $molid "index $indexes1 and not index $atom(2)"]
      set sel1 [atomselect $molid "index $indexes2 and not index $atom(1)"]      
      set del [expr {$hdihed/2.0}]
   } elseif {$type=="imprp"} {
      set indexes1 [::util::bondedsel $molid $atom(0) $atom(1) -all]
      set indexes2 [::util::bondedsel $molid $atom(1) $atom(0) -all]
      set sel1 [atomselect $molid "index $indexes1 and not index $atom(1)"]
      set sel2 [atomselect $molid "index $indexes2 and not index $atom(0)"]      
      set del [expr {$himprp/2.0}]
   } else { error " Unknown conformation type $type!" }
   
   # Optional user defined delta
   if {[llength $dx]} {
      set del [expr {$dx/2.0}]
   }

   set numframes [molinfo $molid get numframes]
   set last [expr {$numframes-1}]

   # Generate 2 new frames for f(x-h) and f(x+h)
   mol top $molid
##   animate goto end;
   animate dup frame $startframe $molid
   animate dup frame $startframe $molid
   $sel1 frame [expr {$last+1}];
   $sel2 frame [expr {$last+1}];

   if {[string match "*bond" $type]} {
      # FIXME: Should have a bondring type in which we correct for the change of 
      # neighboring bonds
      set bondvec [vecsub $pos(0) $pos(1)]
      set dir     [vecnorm $bondvec]

      $sel1 moveby [vecinvert [vecscale $del $dir]]
      $sel2 moveby [vecscale $del $dir]

      set del [expr {-$del}]
      $sel1 frame [expr {$last+2}]
      $sel2 frame [expr {$last+2}]
      $sel1 moveby [vecinvert [vecscale $del $dir]]
      $sel2 moveby [vecscale $del $dir]

      set x       [veclength $bondvec]

   } elseif {[regexp "bondring" $type]} {
      puts "DO WE EVER MATCH THIS?  BONDRING"
      puts "DO WE EVER MATCH THIS?  BONDRING"
      puts "DO WE EVER MATCH THIS?  BONDRING"
      puts "DO WE EVER MATCH THIS?  BONDRING"
      puts "DO WE EVER MATCH THIS?  BONDRING"
      puts "DO WE EVER MATCH THIS?  BONDRING"
      puts "DO WE EVER MATCH THIS?  BONDRING"
      puts "DO WE EVER MATCH THIS?  BONDRING"
      set bondvec [vecsub $pos(0) $pos(1)]
      set dir     [vecnorm $bondvec]

      #set bondvec1 [vecsub $pos(-1) $pos(0)]
      #set bondvec2 [vecsub $pos(2) $pos(1)]
      #set c1 [veclength $bondvec1]
      #set c2 [veclength $bondvec2]
      #set a1 [::Paratool::vecangle3 $pos(-2) $pos(-1) $pos(0)]
      #set a2 [::Paratool::vecangle3 $pos(3)  $pos(2)  $pos(1)]
      #set sel [atomselect $molid "index $atom(-1)"]
      #set pos(-1) [join [$sel get {x y z}]]
      #set h1 [expr {$a1-acos(($del-$c1*cos($a1))/$c1)}]
      #set h2 [expr {$a2-acos(($del-$c2*cos($a2))/$c2)}]

      set h $hangle
      set mat [trans angle $pos(-2) $pos(-1) $pos(0)  $h deg]
      $sel1 move $mat
      set mat [trans angle $pos(3)  $pos(2)  $pos(1)  $h deg]
      $sel2 move $mat

      $sel1 frame [expr {$last+2}]
      $sel2 frame [expr {$last+2}]
      set mat [trans angle $pos(-2) $pos(-1) $pos(0) -$h deg]
      $sel1 move $mat
      set mat [trans angle $pos(3)  $pos(2)  $pos(1) -$h deg]
      $sel2 move $mat

      set xupper [measure bond [list $atom(0) $atom(1)] molid $molid frame [expr {$last+2}]]
      set x  [veclength $bondvec]
      set del [expr {($xupper-$x)/2.0}]
   } elseif {[regexp "anglering" $type]} {
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      puts "DO WE EVER MATCH THIS?  ANGLERING"
      set x  [angle_from_coords $pos(0) $pos(1) $pos(2)]

      set deg2rad 0.0174532925199;
      set c1 [veclength [vecsub $pos(0) $pos(1)]]
      set c2 [veclength [vecsub $pos(2) $pos(1)]]
      set h [expr {0.5*$c1*$c2*(cos((0.5*$x+$del)*$deg2rad)-cos((0.5*$x)*$deg2rad))}]
      set dir [vecnorm [vecadd [vecnorm [vecsub $pos(0) $pos(1)]] [vecnorm [vecsub $pos(2) $pos(1)]]]]
      $sel1 moveby [vecinvert [vecscale $h $dir]]
      $sel2 moveby [vecscale $h $dir]

      set h [expr {-$h}]
      $sel1 frame [expr {$last+2}]
      $sel2 frame [expr {$last+2}]
      $sel1 moveby [vecinvert [vecscale $h $dir]]
      $sel2 moveby [vecscale $h $dir]
   } elseif {[regexp "angle|lbend" $type]} {
      set mat [trans angle $pos(2) $pos(1) $pos(0)  $del deg]
      $sel1 move $mat
      set mat [trans angle $pos(2) $pos(1) $pos(0) -$del deg]
      $sel2 move $mat

      set del [expr {-$del}]

      $sel1 frame [expr {$last+2}]
      $sel2 frame [expr {$last+2}]
      set mat [trans angle $pos(2) $pos(1) $pos(0)  $del deg]
      $sel1 move $mat
      set mat [trans angle $pos(2) $pos(1) $pos(0) -$del deg]
      $sel2 move $mat
      set x  [angle_from_coords $pos(0) $pos(1) $pos(2)]
   } elseif {[regexp  "dihed" $type]} {
      # FIXME: Should check how diheds behave in rings!
      set mat [trans bond $pos(2) $pos(1) -$del deg]
      $sel1 move $mat
      set mat [trans bond $pos(2) $pos(1)  $del deg]
      $sel2 move $mat

      #set del [expr {-$del}]
      $sel1 frame [expr {$last+2}]
      $sel2 frame [expr {$last+2}]
      set mat [trans bond $pos(2) $pos(1)  $del deg]
      $sel1 move $mat
      set mat [trans bond $pos(2) $pos(1) -$del deg]
      $sel2 move $mat
      set x [dihed_from_coords $pos(0) $pos(1) $pos(2) $pos(3)]
   } elseif {[regexp  "imprp" $type]} {
      set cb [vecsub $pos(1) $pos(2)]
      set cd [vecsub $pos(3) $pos(2)]
      set ab [vecsub $pos(1) $pos(0)]
      set s  [veccross $cb $cd]
      set r  [veccross $ab $s]
      set mat [trans bond [vecadd $pos(0) $r] $pos(0) -$del deg]
      $sel1 move $mat
      set mat [trans bond [vecadd $pos(0) $r] $pos(0)  $del deg]
      $sel2 move $mat

      set del [expr {-$del}]
      $sel1 frame [expr {$last+2}]
      $sel2 frame [expr {$last+2}]
      set mat [trans bond [vecadd $pos(0) $r] $pos(0) -$del deg]
      $sel1 move $mat
      set mat [trans bond [vecadd $pos(0) $r] $pos(0)  $del deg]
      $sel2 move $mat
      set x [dihed_from_coords $pos(0) $pos(1) $pos(2) $pos(3)]
   } else { return }

   return [list $x [expr {abs($del*2.0)}]]
}
#======================================================
### DEPRECATED
### taken from paratool
proc ::ForceFieldToolKit::BondAngleOpt::bond_in_ring {ind1 ind2} {
   variable ringlist
   set found {}
   set i 0
   foreach ring $ringlist {
      set pos1 [lsearch $ring $ind1]
      if {$pos1>=0} {
         set pos2 [lsearch $ring $ind2]
         if {$pos2>=0 && (abs($pos2-$pos1)==1 || abs($pos2-$pos1)==[llength $ring]-1)} {
            lappend found $i
         }
      }
      incr i
   }
   return $found
}
#======================================================
### DEPRECATED
### taken from paratool
proc ::ForceFieldToolKit::BondAngleOpt::angle_in_ring {ind1 ind2 ind3} {
   variable ringlist
   set found {}
   set i 0
   foreach ring $ringlist {
      set pos2 [lsearch $ring $ind2]   
      if {$pos2>=0} {
         set pos1 [lsearch $ring $ind1]
         if {$pos1>=0 && (abs($pos1-$pos2)==1 || abs($pos1-$pos2)==[llength $ring]-1)} {
            set pos3 [lsearch $ring $ind3]
            if {$pos3>=0 && (abs($pos3-$pos2)==1 || abs($pos3-$pos2)==[llength $ring]-1)} {
               lappend found $i
            }
         }
      }
      incr i
   }
   return $found
}
#======================================================
### DEPRECATED
### taken from paratool - Computes the angle between the specified coordinates
proc ::ForceFieldToolKit::BondAngleOpt::angle_from_coords { p0 p1 p2 } {
   set x [vecnorm [vecsub $p0 $p1]]
   set y [vecnorm [vecsub $p2 $p1]]
   return [expr {57.2957795786*acos([vecdot $x $y])}]
}
#======================================================
### DEPRECATED
### taken from paratool - Computes the angle between the specified coordinates
proc ::ForceFieldToolKit::BondAngleOpt::dihed_from_coords { coord1 coord2 coord3 coord4 } {
  set v1 [vecsub $coord1 $coord2]
  set v2 [vecsub $coord3 $coord2]
  set v3 [vecsub $coord4 $coord3]
  set cross1 [vecnorm [veccross $v2 $v1]]
  set cross2 [vecnorm [veccross $v2 $v3]]
  return [expr {57.2957795131 * acos([vecdot $cross1 $cross2])}]
}
