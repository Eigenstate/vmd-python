#
# $Id: fftk_ChargeOpt.tcl,v 1.25 2014/06/10 21:54:30 mayne Exp $
#

#======================================================
namespace eval ::ForceFieldToolKit::ChargeOpt {

    # Input Variables
    
    # Need to Manually Set
    variable psfPath
    variable pdbPath
    variable resName
    variable parList
    variable logFileList
    variable atomList
    variable indWeights
    
    variable chargeGroups
    variable chargeInit
    variable chargeBounds
    variable chargeSum
    
    variable baseHFLog
    variable baseMP2Log
    variable watLog
    
    variable outFile
    variable outFileName
    
    variable start
    variable end
    variable delta
    variable offset
    variable scale
    
    variable tol
    variable dWeight


    # Set in Procs
    variable QMEn
    variable QMDist
    variable refmolid

    variable dipoleQMcoords
    variable dipoleQMvec
    variable dipoleQMmag
    variable dipoleWeight
    
    variable simtype
    variable debug
    
    variable reChargeOverride
    variable reChargeOverrideCharges
    variable mode
    variable saT
    variable saTSteps
    variable saIter
    
    variable guiMode
    variable optCount
    variable mmCount
    
    variable shiftTrajs
    variable atomInfo
    variable atomDist
    variable ljPars
   
    variable returnFinalCharges

}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::init {} {

    # GUI Input
    variable psfPath
    variable pdbPath
    variable resName
    variable parList
    variable outFileName
    
    set psfPath {}
    set pdbPath {}
    set resName {}
    set parList {}
    set outFileName "ChargeOpt.log"

    # GUI Charge Constraints
    variable chargeGroups
    variable chargeInit
    variable chargeBounds
    variable chargeSum

    set chargeGroups {}
    set chargeInit {}
    set chargeBounds {}
    set chargeSum {}

    # GUI QM Target Data    
    variable baseHFLog
    variable baseMP2Log
    variable watLog
    variable logFileList
    variable atomList
    variable indWeights
    
    set baseHFLog {}
    set baseMP2Log {}
    set watLog {}
    set logFileList {}
    set atomList {}
    set indWeights {}   

    # ADV Settings  
    variable start
    variable end
    variable delta
    variable offset
    variable scale
    variable tol
    variable dWeight

    set start -0.4
    set end 0.4
    set delta 0.05
    set offset -0.2
    set scale 1.16
    set tol 0.005
    set dWeight 1.0
    
    # Other
    variable outFile
    variable QMEn
    variable QMDist
    variable refmolid
    variable simtype
    variable debug
    variable reChargeOverride
    variable reChargeOverrideCharges
    variable mode
    variable saT
    variable saTSteps
    variable dhIter 500
    variable saIter
    variable guiMode
    variable optCount
    variable mmCount
    variable returnFinalCharges
    variable shiftTrajs
    variable atomInfo
    variable atomDist
    variable ljPars
    variable dipoleQMcoords
    variable dipoleQMvec
    variable dipoleQMmag
    variable dipoleWeight
    
    set outFile {}
    set QMEn {}
    set QMDist {}
    set refmolid {}
    
    set simtype ""
    set debug 0
    set reChargeOverride 0
    set reChargeOverrideCharges {}
    set mode downhill
    set saT 25
    set saTSteps 20
    set saIter 15
    set dhIter 500

    set guiMode 1
    set optCount 0
    set mmCount 0
    
    set shiftTrajs {}
    array unset atomInfo
    array unset atomDist
    array unset ljPars
    
    set returnFinalCharges {}

    set dipoleQMcoords {}
    set dipoleQMvec {}
    set dipoleQMmag {}
    set dipoleWeight 1.0

}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::sanityCheck {} {
    # checks to see that appropriate information is set
    # prior to running the charge optimization
    
    # returns 1 if all input is sane
    # returns 0 if there is an error
    
    # localize relevant ChargeOpt variables
    variable psfPath
    variable pdbPath
    variable resName
    variable parList
    variable outFileName

    variable chargeGroups
    variable chargeInit
    variable chargeBounds
    variable chargeSum
    
    variable baseHFLog
    variable baseMP2Log
    variable watLog
    variable logFileList
    variable atomList
    variable indWeights
    
    variable start
    variable end
    variable delta
    variable offset
    variable scale

    variable mode
    variable tol
    variable dWeight
    variable dipoleWeight
    variable saT
    variable saTSteps
    variable dhIter
    variable saIter
    
    # local variables
    set errorList {}
    set errorText ""

    # local error flags
    set psfErrorFlag 0
    set pdbErrorFlag 0
    set aNameErrorFlag 0
    
    # build the error list based on what proc is checked (opt or psf rewrite)
    # INPUT
    # make sure psf is entered and exists
    if { $psfPath eq "" } {
        lappend errorList "No PSF file was specified."
        set psfErrorFlag 1
    } else {
        if { ![file exists $psfPath] } {
            lappend errorList "Cannot find PSF file."
            set psfErrorFlag 1
        }
    }
    
    # make sure the pdb is entered and exists
    if { $pdbPath eq "" } {
        lappend errorList "No PDB file was specified."
        set pdbErrorFlag 1
    } else {
        if { ![file exists $pdbPath] } {
            lappend errorList "Cannot find PDB file."
            set pdbErrorFlag 1
        }
    }

    # if valid psf/pdb, load molecule for checking
    if { !$psfErrorFlag && !$pdbErrorFlag } {
        set molid [mol new $psfPath waitfor all]
        mol addfile $pdbPath waitfor all $molid
    }

    # make sure residue name isn't empty and exists in the provide psf/pdb
    if { $resName eq "" } {
        lappend errorList "Residue name was not specified."
    } elseif { [info exists molid] } {
        set sel [atomselect $molid all]
        if { [lsearch [lsort -unique [$sel get resname]] $resName] == -1 } { lappend errorList "Residue name not found in molecule" }
        $sel delete
    } else {
        lappend errorList "Cannot test residue name due to an error with the PSF and/or PDB files."
    }
    
    # make sure there is a parameter file (init and one with at least TIP3 water)
    # and that they exist
    if { [llength $parList] == 0 } { lappend errorList "No parameter files were specified." } \
    else { foreach parFile $parList { if { ![file exists $parFile] } { lappend errorList "Cannot open prm file: $parFile." } } }
        
    # make sure output file name (outFileName) isn't blank, and user can write to output dir
    if { $outFileName eq "" } { lappend errorList "Output LOG file was not specified." } \
    else { if { ![file writable [file dirname $outFileName]] } { lappend errorList "Cannot write to output LOG directory." } }
    
    
    # CHARGE CONSTRAINTS
    # may need some work, although there may only be so much we can do here
    # charge groups
    # check that charge groups isn't empty
    if { [llength $chargeGroups] == 0 } {
        lappend errorList "Charge groups aren't set."
    } else {
        # check that each group contains at least one atom name
        foreach group $chargeGroups {
            if { $group eq "" } { lappend errorList "Found a charge group without an atom definition." }
        }
        
        # check initial charge
        foreach charge $chargeInit {
            if { $charge eq "" || ![string is double $charge] } { lappend errorList "Found inappropriate initial charge." }
            #<NO LONGER A REQUIREMENT> if { $charge == 0.0 } { lappend errorList "Initial charge should not be 0.0." }
        }
        
        # check bounds
        foreach bound $chargeBounds {
            if { [llength $bound] != 2 } { lappend errorList "Found Unbalanced bounds element." }
            if { [lindex $bound 0] eq "" || ![string is double [lindex $bound 0]] } { lappend errorList "Found inappropriate lower bound." }
            if { [lindex $bound 1] eq "" || ![string is double [lindex $bound 1]] } { lappend errorList "Found inappropriate upper bound." }
        }
        
        # check charge sum
        if { $chargeSum eq "" || ![string is double $chargeSum] } { lappend errorList "Found inappropriate charge sum." }
    }
    
    
    # QM TARGET DATA
    # may also need some work.
    # check cmpd QM single point energy log file is entered and exists
    if { $baseHFLog eq "" } { lappend errorLog "QM single point energy log (HF) file for the compound was not specified." } \
    else { if { ![file exists $baseHFLog] } { lappend errorLog "Cannot find QM single point energy (HF) log file for compound." } }
    
    # check cmpd QM single point energy (MP2) log file is entered and exists
    if { $baseMP2Log eq "" } { lappend errorLog "QM single point energy (MP2) log file for the compound was not specified." } \
    else { if { ![file exists $baseMP2Log] } { lappend errorLog "Cannot find QM single point energy (MP2) log file for compound." } }
    
    # check wat QM single point energy log file is entered and exists
    if { $watLog eq "" } { lappend errorLog "QM single point energy log file for water was not specified." } \
    else { if { ![file exists $watLog] } { lappend errorLog "Cannot find QM single point energy log file for water." } }
    
    # check that log file list isn't empty and each file exists

    if { [llength $logFileList] == 0 } {
        lappend errorList "No QM water-interaction energy log files loaded."
    } else {
        foreach logFile $logFileList { if { ![file exists $logFile] } { lappend errorLog "Cannot find water-interaction file: $logFile" } }
    }

    # check that atom names are entered and exist only once in the molecule
    foreach atom $atomList {
        if { $atom eq "" } {
            lappend errorList "Found unspecified atom name in QM water-interaction log list."
        } elseif { [info exists molid] && $resName ne "" } {
            set catchRet [catch {set sel [atomselect $molid "resname $resName and name $atom"]}]

            if { $catchRet == 1 } {
                # error and could not make selection
                lappend errorList "Atom name: $atom not found in molecule"
            } else {
                # selection made, test for more than one atom name
                if { [llength [$sel list]] > 1 } { lappend errorList "More than one atom with name: $atom found in molecule" }
                # either way, clean up the selection object
                $sel delete
            }
            unset catchRet
        } else {
            # no molecule loaded or no resname set, so we can't test
            # only report the error once (i.e., not for EVERY atom name)
            if { !$aNameErrorFlag } {
                lappend errorList "Cannot test atom names due to an error with the PSF, PDB, or resname."
                set aNameErrorFlag 1
            }
        }
    }
    
    # check weights
    foreach weight $indWeights {
        if { $weight < 0 || ![string is double $weight] } { lappend errorList "Found inappropriate weight in QM water-interaction log list." }
    }
    
    
    # ADVANCED SETTINGS
    # water shift settings - check that they are not empty and are numbers
    # start
    if { $start eq "" } { lappend errorList "Water shift start is not set." } \
    else { if { ![string is double $start] } { lappend errorList "Found inappropriate water shift start value." } }
    # end
    if { $end eq "" } { lappend errorList "Water shift end is not set." } \
    else { if { ![string is double $end] } { lappend errorList "Found inappropriate water shift end value." } }
    # delta
    if { $delta eq "" } { lappend errorList "Water shift delta is not set." } \
    else { if { ![string is double $delta] } { lappend errorList "Found inappropriate water shift delta value." } }
    # offset
    if { $offset eq "" } { lappend errorList "Water shift offset is not set." } \
    else { if { ![string is double $offset] } { lappend errorList "Found inappropriate water shift offset value." } }
    # scale
    if { $scale eq "" } { lappend errorList "Water shift scale is not set." } \
    else { if { ![string is double $scale] } { lappend errorList "Found inappropriate water shift scale value." } }

    # optimizer settings
    if { [lsearch -exact {downhill {simulated annealing}} $mode] == -1 } { lappend errorList "Unsupported optimization mode." } \
    else {
        # check tol
        if { $tol eq "" || $tol < 0 || ![string is double $tol] } { lappend errorList "Found inappropriate optimization tolerance setting." }
        # check dist weight 
        if { $dWeight eq "" || $dWeight < 0 || ![string is double $dWeight] } { lappend errorList "Found inappropriate distance weight setting." }
        # check dipole weight
        if { $dipoleWeight eq "" || $dipoleWeight < 0 || ![string is double $dipoleWeight] } { lappend errorList "Found inappropriate dipole weight setting." }
        # check simulated annealing parameters
        if { $mode eq "simulated annealing" } {
            if { $saT eq "" || ![string is double $saT] } { lappend errorList "Found inappropriate saT setting." }
            if { $saTSteps eq "" || $saTSteps < 0 || ![string is integer $saTSteps] } { lappend errorList "Found inappropriate saTSteps setting." }
            if { $saIter eq "" || $saIter < 0 || ![string is integer $saIter] } { lappend errorList "Found inappropriate saIter setting." }
        }
    }
    
    # clean up the test molecule, if one was loaded
    if { [info exists molid] } { mol delete $molid }

    # if there is an error, tell the user about it
    # return 0 to tell the calling proc that there is a problem
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
proc ::ForceFieldToolKit::ChargeOpt::optimize {} {
    # rebuild charge optimization
    
    # need to localize all variables
    variable psfPath
    variable pdbPath
    variable resName
    variable parList
    variable outFileName
    variable chargeGroups
    variable chargeInit
    variable chargeBounds
    variable chargeSum
    variable baseHFLog
    variable baseMP2Log
    variable watLog
    variable logFileList
    variable atomList
    variable indWeights
    variable start
    variable end
    variable delta
    variable offset
    variable scale
    variable tol
    variable dWeight
    variable outFile
    variable QMEn
    variable QMDist
    variable simtype
    variable debug
    variable reChargeOverride
    variable reChargeOverrideCharges
    variable mode
    variable saT
    variable saTSteps
    variable dhIter 500
    variable saIter
    variable guiMode
    variable optCount
    variable mmCout
    variable returnFinalCharges

    # new variables (rebuild)   
    variable refmolid; # modified from molid
    variable shiftTrajs
    variable atomInfo
    variable atomDist
    variable ljPars

    variable dipoleQMcoords
    variable dipoleQMvec
    variable dipoleQMmag
    
    # run a sanity check
    if { ![::ForceFieldToolKit::ChargeOpt::sanityCheck] } { return }
    
    #
    # GENERAL SETUP
    #
    
    # prepare the mol areas
    mol delete all

    # open output files
    set outFile [open $outFileName w]
    if { $debug } {
        set debugLog [open "[file rootname $outFileName].debug.log" w]
        ::ForceFieldToolKit::ChargeOpt::printSettings $debugLog
    }
    
    
    # misc
    set simtype "single point"

    #
    # GENERATING PSF FOR CMPD + WATER
    #

    # new version (without dependency on psfgen)
    # note: this version is more complex, but plays nice with long atom types
    set propList {name type charge element resid resname mass x y z chain segname}
    
    # compile cmpd info
    set cmpdMolID [mol new $psfPath]
    mol addfile $pdbPath waitfor all $cmpdMolID
    # reTypeFromPSF / reChargeFromPSF have been depreciated 
    # ::ForceFieldToolKit::SharedFcns::reTypeFromPSF $psfPath $cmpdMolID
    # ::ForceFieldToolKit::SharedFcns::reChargeFromPSF $psfPath $cmpdMolID
    
    set sel [atomselect $cmpdMolID all]
    set cmpdNumAtoms [molinfo $cmpdMolID get numatoms]
    set cmpdPropList [$sel get $propList]
    $sel delete
    set bondList [topo getbondlist -molid $cmpdMolID]
    set angList [topo getanglelist -molid $cmpdMolID]
    set dihList [topo getdihedrallist -molid $cmpdMolID]
    set imprpList [topo getimproperlist -molid $cmpdMolID]

    # build wat info
    set watPropList {
        {OH2 OT -0.834000 O 1 TIP3 15.9994  0.353  0.997  3.995 W WT}
        {H1  HT  0.417000 H 1 TIP3  1.0080  1.170  0.722  4.411 W WT}
        {H2  HT  0.417000 H 1 TIP3  1.0080 -0.261  1.107  4.721 W WT}
    }

    # combine cmpd + wat properties
    set cmpdwatPropList [concat $cmpdPropList $watPropList]
    set watOind $cmpdNumAtoms
    set watH1ind [expr {$cmpdNumAtoms + 1}]
    set watH2ind [expr {$cmpdNumAtoms + 2}]
    #set bondList [concat $bondList [list $watOind $watH1ind] [list $watOind $watH2ind]]
    lappend bondList [list $watOind $watH1ind] [list $watOind $watH2ind]
    lappend angList [list unknown $watH1ind $watOind $watH2ind]

    # generate the combined molecule
    set cmpdwatMolID [mol new atoms [expr {$cmpdNumAtoms + 3}]]
    animate dup $cmpdwatMolID
    # setup all of the properties
    set sel [atomselect $cmpdwatMolID all]
    $sel set $propList $cmpdwatPropList
    topo setbondlist $bondList
    topo setanglelist $angList
    topo setdihedrallist $dihList
    topo setimproperlist $imprpList

    # write the combined molecule to file
    $sel writepsf base-wat.psf
    $sel writepdb base-wat.pdb

    # clean up
    $sel delete
    mol delete $cmpdMolID
    mol delete $cmpdwatMolID
    unset propList cmpdMolID cmpdNumAtoms cmpdPropList bondList angList dihList imprpList
    unset watPropList watOind watH1ind watH2ind cmpdwatMolID

    # old version with dependency on psfgen
    # note: psfgen doesn't work with long atom types
    # Build the necessary psf/pdb for a water molecule from internal library (proc)
    # ::ForceFieldToolKit::ChargeOpt::writeWatPSF
    # ::ForceFieldToolKit::ChargeOpt::writeWatPDB
    
    # Construct a psf/pdb pair for Compound + Water
    # resetpsf
    # psfcontext reset
    # readpsf $psfPath
    # coordpdb $pdbPath
    # readpsf wat.psf
    # coordpdb wat.pdb
    # writepsf base-wat.psf
    # writepdb base-wat.pdb

    # load the psf into VMD
    set refmolid [mol new base-wat.psf]
    # reTypeFromPSF / reChargeFromPSF have been depreciated
    # ::ForceFieldToolKit::SharedFcns::reTypeFromPSF base-wat.psf $refmolid
    # ::ForceFieldToolKit::SharedFcns::reChargeFromPSF base-wat.psf $refmolid

    #
    # PARSE QM QUANTITIES
    #

    if { $guiMode } {
        set ::ForceFieldToolKit::gui::coptStatus "Running...Loading QM Data"
        update idletasks
    }
    
    # Parse Compound and Water Single Point Energy Calculations for QM Energy
    set Enwat [lindex [lindex [::ForceFieldToolKit::ChargeOpt::getscf $watLog] end] 1]
    set Enbase [lindex [lindex [::ForceFieldToolKit::ChargeOpt::getscf $baseHFLog] end] 1]
    
    # Parse energies and optimal water positions from QM Log files
    set QMEn {}
    foreach log $logFileList {
       # Parse energy, calculate interaction energy (QMEn)
       set Entot [lindex [lindex [::ForceFieldToolKit::ChargeOpt::getscf $log] end] 1]
       lappend QMEn [expr $scale*($Entot - $Enbase - $Enwat)]
       mol addfile base-wat.pdb waitfor all $refmolid 
    
       # be very conservative here, do not assume coordinates in
       # pdb match those in water-interaction log file

       # Parse Compound coordinates and move VMD atoms into position    
       set sel [atomselect $refmolid "resname $resName"]
       set molCoords [::ForceFieldToolKit::ChargeOpt::getMolCoords $log [$sel num]]
       for {set i 0} {$i < [$sel num]} {incr i} {
          set temp [atomselect $refmolid "index $i"]
          $temp set x [lindex [lindex $molCoords $i] 0]
          $temp set y [lindex [lindex $molCoords $i] 1]
          $temp set z [lindex [lindex $molCoords $i] 2]
          $temp delete
       }
       $sel delete

       # Parse Water coordinates and move VMD atoms into position
       # Don't want to assume water atoms are always in the same order
       set watCoords [::ForceFieldToolKit::ChargeOpt::getWatCoords $log]
       set sel [atomselect $refmolid "water and name OH2"]
       $sel moveto [lindex $watCoords 0]
       $sel delete
       set sel [atomselect $refmolid "water and name H1"]
       $sel moveto [lindex $watCoords 1]
       $sel delete
       set sel [atomselect $refmolid "water and name H2"]
       $sel moveto [lindex $watCoords 2]
       $sel delete
    }

    # measure the qm distances (unscaled)
    set QMDist {}
    for {set i 0} {$i < [llength $atomList]} {incr i} {
       set temp1 [atomselect $refmolid "name [lindex $atomList $i] and resname $resName" frame $i]
       set temp2 [atomselect $refmolid "water and name OH2" frame $i]   
       lappend QMDist [measure bond "[$temp1 get index] [$temp2 get index]" frame $i]
       $temp1 delete
       $temp2 delete
    }

    if { $debug } {
        puts $debugLog "QME(water): $Enwat"
        puts $debugLog "QME(cmpd): $Enbase"
        puts $debugLog "scaled QMEn: $QMEn"
        puts $debugLog "QMDist (unshifted): $QMDist"
        flush $debugLog
    }

    # parse dipole data
    set dipoleData [::ForceFieldToolKit::ChargeOpt::getDipoleData $baseMP2Log]
    set dipoleQMcoords [lindex $dipoleData 0]
    set dipoleQMvec [lindex $dipoleData 1]
    set dipoleQMmag [lindex $dipoleData 2]
    unset dipoleData

    if { $debug } {
        puts $debugLog "QM Standard Orientation Coordinates:"
        foreach ele $dipoleQMcoords {
            puts $debugLog $ele
        }
        puts $debugLog "QM Dipole Vector: $dipoleQMvec"
        puts $debugLog "QM Dipole Magnitude: $dipoleQMmag"
        flush $debugLog
    }

    #
    # BUILD THE SHIFT TRAJECTORIES
    #

    set shiftTrajs {}
    for {set i 0} {$i < [molinfo $refmolid get numframes]} {incr i} {

        # write qm coords to a temporary template pdb file
        set templateStruct [atomselect $refmolid "all" frame $i]
        $templateStruct writepdb base-wat-optpos.pdb
        $templateStruct delete

        # setup the psf into which the shift frames will be loaded
        set currTraj [mol new base-wat.psf waitfor all]

        # load each shifted water position as a frame
        for {set d $start} {$d <= $end} {set d [expr $d+$delta]} {
            mol addfile base-wat-optpos.pdb waitfor all $currTraj
            ::ForceFieldToolKit::ChargeOpt::shiftWat [lindex $atomList $i] $currTraj $d $offset
        }
        
        # retyping and re-charging
        # reTypeFromPSF / reChargeFromPSF have been depreciated
        # ::ForceFieldToolKit::SharedFcns::reTypeFromPSF base-wat.psf $currTraj
        # ::ForceFieldToolKit::SharedFcns::reChargeFromPSF base-wat.psf $currTraj
        # check to see if recharge is overridden in advanced settings
        if { $reChargeOverride } {
            foreach ovr $reChargeOverrideCharges {
                set temp [atomselect $currTraj "name [lindex $ovr 0]"]
                $temp set charge [lindex $ovr 1]
                $temp delete
            }
        }

        # add the molid to the list of shift trajectories
        lappend shiftTrajs $currTraj
    }
    
    # clean up
    # file delete wat.psf
    # file delete wat.pdb
    file delete base-wat.psf
    file delete base-wat.pdb
    file delete base-wat-optpos.pdb
    
    if { $debug } {
        puts $debugLog ""
        puts $debugLog "Reference trajectory loaded: $refmolid"
        puts $debugLog "Shift trajectories loaded: $shiftTrajs"
        flush $debugLog
    }
    
    
    #
    # BUILD VDW/LJ PARAMETER ARRAY/HASH
    #

    # parse the lj parameters from file(s)
    array unset ljPars
    array set ljPars {}
    foreach par $parList {
        set ljTemp [lindex [::ForceFieldToolKit::SharedFcns::readParFile $par] 4]
        foreach ele $ljTemp {
            set ljPars([lindex $ele 0]) [lindex $ele 1]
        }
        unset ljTemp
    }
    
    if { $debug } {
        puts $debugLog ""
        puts $debugLog "[array size ljPars] elements added to ljPar array"
        flush $debugLog
    }

    #
    # BUILD ATOM INFO AND DIST ARRAYS/HASHES
    #

    # find the molid for an example cmpd+water set
    # should be retyped/recharged (i.e. not the refmolid)
    set exMolid [lindex $shiftTrajs 0]    

    # find some info about cmpd and water
    set atomNames [[atomselect $exMolid "all and not water"] get name]
    set watH1ind [[atomselect $exMolid "water and name H1"] get index]
    set watH2ind [[atomselect $exMolid "water and name H2"] get index]
    set watOHind [[atomselect $exMolid "water and name OH2"] get index]

    # build the atomInfo array/hash
    array unset atomInfo
    array set atomInfo {}

    # cycle through each cmpd atom
    foreach aName $atomNames {
        set tempSel [atomselect $exMolid "name $aName and not water"]
        set ind [$tempSel get index]
        set type [$tempSel get type]
        set charge [$tempSel get charge]
        set atomInfo($aName) [list $ind $type $charge]
        #puts "adding atomInfo: $aName -- $ind, $type, $charge"
        $tempSel delete
    }
    
    if { $debug } {
        puts $debugLog "[array size atomInfo] elements added to atomInfo array"
        flush $debugLog
    }

    # build the atomdist array/hash
    array unset atomDist
    array set atomDist {}

    # cycle through each of the shift trajectories
    for {set i 0} {$i < [llength $shiftTrajs]} {incr i} {
        mol top [lindex $shiftTrajs $i]
        
        # cycle through each frame of the given trajectory
        for {set j 0} {$j < [molinfo top get numframes]} {incr j} {
            animate goto $j
            
            # measure the distance of every cmpd atom to each water atom
            foreach aName $atomNames {
                set atomInd [lindex $atomInfo($aName) 0]
                set dH1 [measure bond [list $atomInd $watH1ind]]
                set dH2 [measure bond [list $atomInd $watH2ind]]
                set dOH [measure bond [list $atomInd $watOHind]]
                set atomDist(${i},${j},${atomInd}) [list $dH1 $dH2 $dOH]
            }
        }
    }
    
    if { $debug } {
        puts $debugLog "[array size atomDist] elements added to atomDist"
        flush $debugLog
    }



    #
    # SETUP OPTIMIZATION
    #

    # reset counter to keep track of optimization iterations
    set optCount 0
    # if running from gui, update the status
    if { $guiMode } {
        set ::ForceFieldToolKit::gui::coptStatus "Running...Optimizing(iter:$optCount)"
        update idletasks
    }
    
    # setup the optimization mode (downhill or simulated annealing)
    if { $mode eq "downhill" } {
        set opt [optimization -downhill -tol $tol -iter $dhIter -function ::ForceFieldToolKit::ChargeOpt::optCharges]
    } elseif { $mode eq "simulated annealing" } {
        set opt [optimization -annealing -tol $tol -T $saT -iter $saIter -Tsteps $saTSteps -function ::ForceFieldToolKit::ChargeOpt::optCharges]  
    }
        
    # configure bounds and initialize

    $opt configure -bounds [lrange $chargeBounds 0 end-1]

    # set implicit boundary (sum of charges)    
    $opt configure -implicit ::ForceFieldToolKit::ChargeOpt::calcLeftover -ibounds [list [lindex $chargeBounds end]]

    $opt initsimplex [lrange $chargeInit 0 end-1]    

    if { $debug } {
        puts $debugLog "Beginning Optimization"
        puts $debugLog "\topt setup line: optimization -[list $mode] -tol $tol -function ::ForceFieldToolKit::ChargeOpt::optCharges"
        puts $debugLog "\topt configure line: configure -bound [lrange $chargeBounds 0 end-1]"
        puts $debugLog "\topt initsimplex line: initsimplex [lrange $chargeInit 0 end-1]"
        flush $debugLog
        update idletasks
    }
    
    #
    # RUN OPTIMIZATION
    #
    
    set result [$opt start]

    
    #
    # PROCESS OPTIMIZATION RESULTS
    #
    
    set finalCharges [lindex $result 0]
    # check for non-empty result.  a known cause is if the obj fcn blows up
    if { $finalCharges == {} } {
        tk_messageBox \
            -type ok \
            -icon warning \
            -message "Application halting on error" \
            -detail "The optimizer has returned an empty result.  This error is known to occur when the objective function exceeds a maximum value (1e15), causing the optimization to terminate abnormally.  Check the QM target data to ensure reasonable interaction energies, as well as, all optimization input."
        # close output and terminate the proc
        puts $outFile "\nffTK has halted on error.  The optimizer has returned an empty result, indicating an abnormal termination.\n"; flush $outFile; close $outFile
        if { $debug } { puts $debugLog "\nffTK has halted on error.  The optimizer has returned an empty result, indicating an abnormal termination.\n"; flush $debugLog; close $debugLog }
        return
    }
    # we should probably include a check to warn users when the optimizer is returning an extremely high objective value
    # include this information in the results section???

    # used to load optimization results into the gui
    set returnFinalCharges {}   

    set curChargeSum 0
    puts $outFile "FINAL CHARGES"
    for {set i 0} {$i < [expr [llength $chargeGroups] - 1]} {incr i} {
       set charge [format %1.3f [lindex $finalCharges $i]]
       set curChargeSum [expr $curChargeSum + [llength [lindex $chargeGroups $i]]*$charge]
       puts $outFile "[list [lindex $chargeGroups $i] $charge]"
       lappend returnFinalCharges [list [lindex $chargeGroups $i] $charge]
    }
  
    set leftover [expr ($chargeSum - $curChargeSum)*1.0/[llength [lindex $chargeGroups end]]]
    puts $outFile "[list [lindex $chargeGroups end] [format "%1.3f" $leftover]]"
    lappend returnFinalCharges [list [lindex $chargeGroups end] [format "%1.3f" $leftover]]
    
    puts $outFile "END"
    puts $outFile "\n Be sure to check sum of charges for rounding errors!"

    # some cleanup  
    #mol delete all
    close $outFile
    
    if { $debug } {
        puts $debugLog "Optimization result:"
        puts $debugLog "$result"
        flush $debugLog
        close $debugLog
    }    
    
    
    # DONE
    
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::calcSum { inpCharges } {
    # sums all charges
    
    # localize variables
    variable chargeSum
    variable chargeGroups
    variable debug
    variable guiMode
    
    set curChargeSum 0.0
    for {set i 0} {$i < [llength $chargeGroups]} {incr i} {
        set curChargeSum [expr {$curChargeSum + [llength [lindex $chargeGroups $i]]*[lindex $inpCharges $i]}]
    }
    return $curChargeSum
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::calcLeftover { inpCharges } {
    # dependent function for last charge
    
    # localize variables
    variable chargeSum
    variable chargeGroups
    variable debug
    variable guiMode
    
    # calculate the leftover charge (charge for last charge group)
    set curChargeSum 0.0
    for {set i 0} {$i < [expr {[llength $chargeGroups] - 1}]} {incr i} {
        set curChargeSum [expr {$curChargeSum + [llength [lindex $chargeGroups $i]]*[lindex $inpCharges $i]}]
    }
    set leftover [expr {($chargeSum - $curChargeSum)*1.0/[llength [lindex $chargeGroups end]]}]
    return $leftover
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::shiftWat {name1 molid dist {offset -0.2}} {

   set tempsel1 [atomselect $molid "not water and name $name1"]
   set tempsel2 [atomselect $molid "water and name OH2"]

   foreach let {x y z} {
     lappend v1 [$tempsel1 get $let]
   }

   foreach let {x y z} {
     lappend v2 [$tempsel2 get $let]
   }

   set unitV [vecnorm [vecsub $v2 $v1]]

   set tempsel3 [atomselect top "water"]

   foreach ind [$tempsel3 get index] {
     set temp [atomselect $molid "index $ind"]
     foreach let {x y z} {
       lappend p [$temp get $let]
     }
     set pnew [vecadd $p [vecscale [expr $offset + $dist] $unitV]]
     $temp set x [lindex $pnew 0]
     $temp set y [lindex $pnew 1]
     $temp set z [lindex $pnew 2]
     unset p
     $temp delete
   }

   $tempsel1 delete
   $tempsel2 delete
   $tempsel3 delete
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::computeIntE { currTraj currFrame } {
    # computs the MM water interaction energy for the specified trajectory frame
    
    # localize variables
    variable atomInfo
    variable atomDist
    variable ljPars

    # setup local variables
    set totalEele 0.0
    set totalElj 0.0
    set totalE 0.0
    set dielectric 1.0

    # hard coded TIP3P water and CHARMM nonbonded parameters
    set qH 0.417
    set qO -0.834

    set epsH -0.046
    set epsO -0.1521
    set rminH 0.2245
    set rminO 1.7682
    
    # cycle through each atom
    foreach aName [array names atomInfo] {
        # lookup atom info
        set aInfo $atomInfo($aName)
        set aInd [lindex $aInfo 0]
        set aType [lindex $aInfo 1]
        set qA [lindex $aInfo 2]

        # lookup lj parameters
        set epsA [lindex $ljPars($aType) 0]
        set rminA [lindex $ljPars($aType) 1]

        # lookup atom distances
        set aDistData $atomDist(${currTraj},${currFrame},${aInd})
        set dH1 [lindex $aDistData 0]
        set dH2 [lindex $aDistData 1]
        set dOH [lindex $aDistData 2]

        # calculate Eele
        set Eele [expr { 332.0636 * ( \
            ($qA * $qH) / ($dielectric * $dH1) +\
            ($qA * $qH) / ($dielectric * $dH2) +\
            ($qA * $qO) / ($dielectric * $dOH)  \
        )}]

        # calculate Elj
        set Elj [expr { \
            sqrt($epsA * $epsH) * (  pow(($rminA+$rminH)/$dH1,12) - 2.0*pow(($rminA+$rminH)/$dH1,6)  ) +\
            sqrt($epsA * $epsH) * (  pow(($rminA+$rminH)/$dH2,12) - 2.0*pow(($rminA+$rminH)/$dH2,6)  ) +\
            sqrt($epsA * $epsO) * (  pow(($rminA+$rminO)/$dOH,12) - 2.0*pow(($rminA+$rminO)/$dOH,6)  )  \
        }]
        

        # update the running totals
        set totalEele [expr { $totalEele + $Eele }]
        set totalElj [expr { $totalElj + $Elj }]
        set totalE [expr { $totalE + $Eele + $Elj }]

    }

    #puts "($currTraj,$currFrame) $totalEele \t $totalElj \t $totalE"

    return $totalE
    
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::optCharges { inpCharges } {
    # charge optimizing function
    
    # localize variables
    variable QMEn
    variable QMDist
    variable atomList
    variable dWeight
    variable chargeSum
    variable chargeBounds
    variable outFile
    variable chargeGroups
    variable refmolid
    variable shiftTrajs
    variable parList
    variable resName
    variable start
    variable end
    variable delta
    variable offset
    variable indWeights
    variable debug
    variable guiMode
    variable optCount 
    
    # new variables
    variable refmolid; # modified from molid
    variable shiftTrajs
    variable atomInfo
    variable atomDist
    variable ljPars

    variable dipoleQMcoords
    variable dipoleQMvec
    variable dipoleQMmag
    variable dipoleWeight
    
    # calculate the leftover charge (charge for last charge group)
    set curChargeSum 0.0
    for {set i 0} {$i < [expr {[llength $chargeGroups] - 1}]} {incr i} {
        set curChargeSum [expr {$curChargeSum + [llength [lindex $chargeGroups $i]]*[lindex $inpCharges $i]}]
    }
    set leftover [expr {($chargeSum - $curChargeSum)*1.0/[llength [lindex $chargeGroups end]]}]
    set charges $inpCharges
    lappend charges $leftover
    
    # update the charges in the atomInfo array
    for {set i 0} {$i < [llength $chargeGroups]} {incr i} {
        foreach aName [lindex $chargeGroups $i] {
            lset atomInfo($aName) 2 [lindex $charges $i]
        }
    }
    
    puts -nonewline $outFile "Current test charges: "
    for {set i 0} {$i < [llength $chargeGroups]} {incr i} {
        puts -nonewline $outFile [format "[lindex $chargeGroups $i] %2.3f   " [lindex $charges $i]]
    }
    puts $outFile ""

    # compute MM data
    set MMEn {}
    set MMDistdelta {}
    set dipoleMMvec {0 0 0}
    set dipoleMMmag {}
    
    # cycle through each trajectory in shiftTrajs
    for {set i 0} {$i < [llength $shiftTrajs]} {incr i} {
        
        # initialize values for determing Emin
        set dMin 0.0
        set enMin 100000000000.00
        
        # cycle through all frames to compute interaction energies
        #for {set j 0} {$j <= [expr {int(($end - $start)/$delta)}]} {incr j} {}
        for {set j 0} {$j < [molinfo [lindex $shiftTrajs 0] get numframes]} {incr j} {
            # compute the interaction energy
            set watIntE [::ForceFieldToolKit::ChargeOpt::computeIntE $i $j]
            
            # determine if this is the new low
            if { $watIntE < $enMin } {
                set dMin [expr { $start + $j * $delta }]
                set enMin $watIntE
            }
        }
        
        lappend MMEn $enMin
        lappend MMDistdelta $dMin
        
    }
    
    # compute the MM dipole contribution of each atom using the current charges
    foreach ele [array names atomInfo] {
        # get index
        set ind [lindex $atomInfo($ele) 0]
        # get current charge
        set currCharge [expr { 1.602176487E-19 * [lindex $atomInfo($ele) 2] }] ; # in coulombs
        # get qm std orientation xyz
        set xyz [vecscale 1E-10 [lindex $dipoleQMcoords $ind]] ; # in meters

        # debugging
        #puts "--atom--"
        #puts "ele: $ele"
        #puts "ind: $ind"
        #puts "currCharge: $currCharge"
        #puts "xyz: $xyz"

        # calc and add contribution to total vector from current atom (ele)
        set dipoleMMvec [vecadd $dipoleMMvec [vecscale [expr {1/3.33564E-30}] [vecscale $currCharge $xyz]]] ; # in Debeye
    }
    # calc the MM dipole vector magnitude
    set dipoleMMmag [veclength $dipoleMMvec]



    # Calculate the objective value
    set totalObj 0.0
    set enObj 0.0
    set distObj 0.0

    puts $outFile "Iteration: $optCount"

    ## scale energies by 0.2 kcal/mol and distances by 0.1 A (squared below - 0.04, 0.01)
    for {set i 0} {$i < [llength $atomList]} {incr i} {
        puts -nonewline $outFile [format "[lindex $atomList $i] QME: %1.3f MME: %1.3f (%1.3f)" [lindex $QMEn $i] [lindex $MMEn $i] [expr {[lindex $MMEn $i]-[lindex $QMEn $i]}]]
        puts $outFile [format "  QMD: %1.3f MMDistDelta: %1.3f" [lindex $QMDist $i] [lindex $MMDistdelta $i]]
        set enObj [expr $enObj + [lindex $indWeights $i]*pow([expr [lindex $QMEn $i] - [lindex $MMEn $i]],2)/0.04]
##      set Obj [expr $Obj + pow([expr [lindex $QMEn $i] - [lindex $MMEn $i]],2)/abs([lindex $QMEn $i])]
        set distObj [expr $distObj + [lindex $indWeights $i]*$dWeight*pow([lindex $MMDistdelta $i],2)/0.01]
##      set Obj [expr $Obj + $dWeight*pow([lindex $MMDistdelta $i],2)/[lindex $QMDist $i]]
    }

    

   # target 1.2-1.5, angle < 30; should be pretty bad if this can't be achieved
   # so set tolerances of 0.1 (^2=0.01) and 5 (^2=25); dipole weight should be 
   # scaled by number of charges to make it comparable
   # CGM -- added modifications to handle cases where QM dipole is exactly {0 0 0}
    if { [veclength $dipoleQMvec] == 0.0 } {
        set dipoleAng 0.0
    } else {
        set dipoleAng [::util::rad2deg [expr acos([vecdot [vecnorm $dipoleQMvec] [vecnorm $dipoleMMvec]])]]
    }
    if { $dipoleQMmag == 0.0 } { set dipoleQMmag 1e-2 }    
    set dipoleRatio [expr $dipoleMMmag*1.0/$dipoleQMmag]
    set scaledDipWeight [expr $dipoleWeight*[llength $atomList]] 

    puts $outFile [format "Dipole -- QM: {%1.3f %1.3f %1.3f} (%1.3f D) MM: {%1.3f %1.3f %1.3f} (%1.3f D) Ratio: %1.2f" [lindex $dipoleQMvec 0] [lindex $dipoleQMvec 1] [lindex $dipoleQMvec 2] $dipoleQMmag [lindex $dipoleMMvec 0] [lindex $dipoleMMvec 1] [lindex $dipoleMMvec 2] $dipoleMMmag $dipoleRatio]

    if { $dipoleRatio < 1.2 } {
       set dipObj [expr $scaledDipWeight * pow($dipoleRatio - 1.2,2)/0.01]
    } elseif { $dipoleRatio > 1.5 } {
       set dipObj [expr $scaledDipWeight * pow($dipoleRatio - 1.5,2)/0.01]
    } else {
       set dipObj 0.0
    }
    # scale by QM dipole magnitude, since average is ~ 1 Debye
    # this fixes issues when the dipoles are nearly zero
    if { $dipoleAng > 30.0 } {
       set dipObj [expr $dipObj + $dipoleQMmag * $scaledDipWeight * pow($dipoleAng - 30.0,2)/25.0]
    }

    set totalObj [expr {$enObj + $distObj + $dipObj}]

    puts $outFile [format "Current objective value: %.6f ( En: %.6f  Dist: %.6f Dipole: %0.6f )\n\n" $totalObj $enObj $distObj $dipObj] 

    #puts $outFile "Current objective value: $Obj\n\n\n"
    flush $outFile    
    
    # if running from gui, update the status menu
    incr optCount
    if { $guiMode } {
        set ::ForceFieldToolKit::gui::coptStatus "Running...Optimizing(iter:$optCount)"
        update idletasks
    }


    return $totalObj

}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::getscf { file } {
    variable simtype

   set scfenergies {}

   set fid [open $file r]

   set hart_kcal 1.041308e-21; # hartree in kcal
   set mol 6.02214e23;

   set num 0
   set ori 0
   set tmpscf {}
   set optstep 0
   set scanpoint 0
   
   while {![eof $fid]} {
      set line [string trim [gets $fid]]

      # Stop reading on errors
      if {[string match "Error termination*" $line]} { puts $line; return $scfenergies }

      # We only read Link0
      if {[string match "Normal termination of Gaussian*" $line]} { variable normalterm 1; break }
            
      if {$simtype=="Relaxed potential scan"} {
         if {[string match "Step number * out of a maximum of * on scan point * out of *" $line]} {
            set optstep   [lindex $line 2]
            set scanpoint [lindex $line 12]
            set scansteps [lindex $line 15]
#            puts "SCAN: optstep $optstep on scan point $scanpoint out of $scansteps"
         }
      }
            
     if {[string match "SCF Done:*" $line] || [string match "Energy=* NIter=*" $line]} {
         if {[string match "SCF Done:*" $line]} {
            set scf [lindex $line 4]
         } else {
            set scf [lindex $line 1]
         }
         set scfkcal [expr {$scf*$hart_kcal*$mol}]
         if {$num==0} { set ori $scf }
         set scfkcalori [expr {($scf-$ori)*$hart_kcal*$mol}]
         # In case of a relaxed potential scan we replace the previous energy of the same scanstep,
         # otherwise we just append all new scf energies
         if {$optstep==1 || !($simtype=="Relaxed potential scan")} {
            if {[llength $tmpscf]} { lappend scfenergies $tmpscf; set tmpscf {} }
#            puts [format "%i: SCF = %f hart = %f kcal/mol; rel = %10.4f kcal/mol" $num $scf $scfkcal $scfkcalori]
         }
         set tmpscf [list $num $scfkcal]

         incr num
      }

   }
   close $fid
   if {[llength $tmpscf]} { lappend scfenergies $tmpscf }

   return $scfenergies
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::getMolCoords { file numMolAtoms } {

   set fid [open $file r]
   ::QMtool::init_variables ::QMtool

   ::QMtool::read_gaussian_cartesians $fid qmtooltemppdb.pdb last
   file delete qmtooltemppdb.pdb
   set coordlist [lindex [::QMtool::get_cartesian_coordinates] 0]
    
   close $fid

   return [lrange $coordlist 0 [expr $numMolAtoms - 1]]
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::getWatCoords { file } {
   
   set fid [open $file r]
   ::QMtool::init_variables ::QMtool

   ::QMtool::read_gaussian_cartesians $fid qmtooltemppdb.pdb last
   file delete qmtooltemppdb.pdb
   set coordlist [lindex [::QMtool::get_cartesian_coordinates] 0]
   set atomlist [::QMtool::get_atomproplist]
   set numAtoms [llength $atomlist]

   set Hcount 0
   for {set i [expr $numAtoms - 4]} {$i < $numAtoms} {incr i} {
      set name [lindex [lindex $atomlist $i] 1]
      if { [string match "O*" $name] } {
         set Ocoord [lindex $coordlist $i]
      } elseif { [string match "H*" $name] && $Hcount == 1} {
         set H2coord [lindex $coordlist $i]
         set Hcount 2
      } elseif { [string match "H*" $name] && $Hcount == 0} {
         set H1coord [lindex $coordlist $i]
         set Hcount 1
      }
   }

   close $fid

   set coords [list $Ocoord $H1coord $H2coord]
   return $coords

}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::getDipoleData { filename } {
    # parse relevant dipole data from QM single point LOG file

    # initialize some variables
    set coords {}
    set qmVec {}
    set qmMag {}

    # open the file for reading
    set inFile [open $filename r]

    # parse the LOG file
    # burn lines until we find the std orientation
    while { [set inLine [string trim [gets $inFile]]] ne "Standard orientation:" } { continue }

    # once std orientation is found, burn header (4 lines)
    for {set i 0} {$i < 4} {incr i} { gets $inFile }

    # read in the coords
    while { ![regexp {^-*$} [set inLine [string trim [gets $inFile]]]] } {
        lappend coords [lrange $inLine 3 5]
    }

    # burn until find the dipole moment
    while { [set inLine [string trim [gets $inFile]]] ne "Dipole moment (field-independent basis, Debye):"} { continue }

    # parse the dipole moment
    set inLine [string trim [gets $inFile]]
    set qmVec [list [lindex $inLine 1] [lindex $inLine 3] [lindex $inLine 5]]
    set qmMag [lindex $inLine 7]

    # we're done with the LOG file
    unset inLine
    close $inFile

    return [list $coords $qmVec $qmMag]
    
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::writeMinConf { name psf pdb parlist {extrabFile ""} } {

   set conf [open "$name.conf" w]
   puts $conf "structure          $psf"    
   puts $conf "coordinates        $pdb"
   puts $conf "paraTypeCharmm     on"
   foreach par $parlist {
      puts $conf "parameters       $par"
   }
   puts $conf "temperature         310"
   puts $conf "exclude             scaled1-4"
   puts $conf "1-4scaling          1.0"
   puts $conf "cutoff              1000.0"
   puts $conf "switching           on"
   puts $conf "switchdist          1000.0"
   puts $conf "pairlistdist        1000.0"
   puts $conf "timestep            1.0 "
   puts $conf "nonbondedFreq       2"
   puts $conf "fullElectFrequency  4 "
   puts $conf "stepspercycle       20"
   puts $conf "outputName          $name"
   puts $conf "restartfreq         1000"
   if { $extrabFile != "" } {
      puts $conf "extraBonds          yes"
      puts $conf "extraBondsFile $extrabFile"
   }
   puts $conf "minimize            1000"
   puts $conf "reinitvels          310"
   puts $conf "run 0"
   close $conf

}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::writeWatPSF {} {

    set outfile [open wat.psf w]
set contents {PSF

       3 !NTITLE
 REMARKS original generated structure x-plor psf file
 REMARKS topology top_tip3p.inp
 REMARKS segment WT { first NONE; last NONE; auto none  }

       3 !NATOM
       1 WT   1    TIP3 OH2  OT    -0.834000       15.9994           0
       2 WT   1    TIP3 H1   HT     0.417000        1.0080           0
       3 WT   1    TIP3 H2   HT     0.417000        1.0080           0

       2 !NBOND: bonds
       1       2       1       3

       1 !NTHETA: angles
       2       1       3

       0 !NPHI: dihedrals


       0 !NIMPHI: impropers


       0 !NDON: donors


       0 !NACC: acceptors


       0 !NNB

       0       0       0

       1       0 !NGRP
       0       0       0
}

    puts $outfile "$contents"
    close $outfile
    
    #return wat.psf

}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::writeWatPDB {} {

    set outfile [open wat.pdb w]
    
    set contents {REMARK original generated coordinate pdb file
ATOM      1  OH2 TIP3X   1       0.353   0.997   3.995  1.00  0.00      WT   O
ATOM      2  H1  TIP3X   1       1.170   0.722   4.411  1.00  0.00      WT   H
ATOM      3  H2  TIP3X   1      -0.261   1.107   4.721  1.00  0.00      WT   H
END 
}
    puts $outfile "$contents"
    close $outfile
    
    #return wat.pdb
}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::printSettings { debugLog } {
    # a tool to print all settings passed to the charge optimization routine
    # and relevant settings in the ChargeOpt namespace that will be accessed
    # by the charge optimization routing
    

    puts $debugLog "=========================================="
    puts $debugLog " Charge Optimization GUI Debugging Output "
    puts $debugLog "=========================================="
    
    puts $debugLog "INPUT SECTION"
    puts $debugLog "psfPath: $::ForceFieldToolKit::ChargeOpt::psfPath"
    puts $debugLog "pdbPath: $::ForceFieldToolKit::ChargeOpt::pdbPath"
    puts $debugLog "resName: $::ForceFieldToolKit::ChargeOpt::resName"
    puts $debugLog "parList:"
    foreach item $::ForceFieldToolKit::ChargeOpt::parList {puts $debugLog "\t$item"}
    puts $debugLog "log file: $::ForceFieldToolKit::ChargeOpt::outFileName"
    puts $debugLog "-------------------------------------------"
    puts $debugLog "CHARGE CONSTRAINTS SECTION"
    #puts $debugLog "chargeGroups:"
    #foreach item $::ForceFieldToolKit::ChargeOpt::chargeGroups {puts $debugLog "\t$item"}
    puts $debugLog "chargeGroups: $::ForceFieldToolKit::ChargeOpt::chargeGroups"
    #puts $debugLog "chargeInit:"
    #foreach item $::ForceFieldToolKit::ChargeOpt::chargeInit {puts $debugLog "\t$item"}
    puts $debugLog "chargeInit: $::ForceFieldToolKit::ChargeOpt::chargeInit"
    #puts $debugLog "chargeBounds:"
    #foreach item $::ForceFieldToolKit::ChargeOpt::chargeBounds {puts $debugLog "\t$item"}
    puts $debugLog "chargeBounds: $::ForceFieldToolKit::ChargeOpt::chargeBounds"
    puts $debugLog "chargeSum: $::ForceFieldToolKit::ChargeOpt::chargeSum"
    puts $debugLog "-------------------------------------------"
    puts $debugLog "QM TARGET DATA SECTION"
    puts $debugLog "baseHFLog: $::ForceFieldToolKit::ChargeOpt::baseHFLog"
    puts $debugLog "baseMP2Log: $::ForceFieldToolKit::ChargeOpt::baseMP2Log"
    puts $debugLog "watLog: $::ForceFieldToolKit::ChargeOpt::watLog"
    puts $debugLog "logFileList:"
    foreach item $::ForceFieldToolKit::ChargeOpt::logFileList {puts $debugLog "\t$item"}
    #puts $debugLog "atomList:"
    #foreach item $::ForceFieldToolKit::ChargeOpt::atomList {puts $debugLog "\t$item"}
    puts $debugLog "atomList: $::ForceFieldToolKit::ChargeOpt::atomList"
    #puts $debugLog "indWeights:"
    #foreach item $::ForceFieldToolKit::ChargeOpt::indWeights {puts $debugLog "\t$item"}
    puts $debugLog "indWeights: $::ForceFieldToolKit::ChargeOpt::indWeights"
    puts $debugLog "-------------------------------------------"
    puts $debugLog "ADVANCED SETTINGS SECTION"
    puts $debugLog "start: $::ForceFieldToolKit::ChargeOpt::start"
    puts $debugLog "end: $::ForceFieldToolKit::ChargeOpt::end"
    puts $debugLog "delta: $::ForceFieldToolKit::ChargeOpt::delta"
    puts $debugLog "end: $::ForceFieldToolKit::ChargeOpt::end"
    puts $debugLog "offset: $::ForceFieldToolKit::ChargeOpt::offset"
    puts $debugLog "scale: $::ForceFieldToolKit::ChargeOpt::scale"
    puts $debugLog "tol: $::ForceFieldToolKit::ChargeOpt::tol"
    puts $debugLog "dWeight: $::ForceFieldToolKit::ChargeOpt::dWeight"
    puts $debugLog "dipoleWeight: $::ForceFieldToolKit::ChargeOpt::dipoleWeight"
    puts $debugLog "Optimization mode: $::ForceFieldToolKit::ChargeOpt::mode"
    puts $debugLog "Simulated Annealing Parameters: Temp. $::ForceFieldToolKit::ChargeOpt::saT, Steps $::ForceFieldToolKit::ChargeOpt::saTSteps, Iterations $::ForceFieldToolKit::ChargeOpt::saIter"
    puts $debugLog "Override ReChargeFromPSF: $::ForceFieldToolKit::ChargeOpt::reChargeOverride"
    puts $debugLog "Override Charges: $::ForceFieldToolKit::ChargeOpt::reChargeOverrideCharges"
    puts $debugLog "debug: $::ForceFieldToolKit::ChargeOpt::debug"
    puts $debugLog "=========================================="
    puts $debugLog ""
    flush $debugLog

}
#======================================================
proc ::ForceFieldToolKit::ChargeOpt::buildScript { scriptFileName } {
    # need to localize all variables
    variable psfPath
    variable pdbPath
    variable resName
    variable parList
    variable chargeGroups
    variable chargeInit
    variable chargeBounds
    variable chargeSum
    variable baseHFLog
    variable baseMP2Log
    variable watLog
    variable logFileList
    variable atomList
    variable indWeights
    variable start
    variable end
    variable delta
    variable offset
    variable scale
    variable tol
    variable dWeight
    variable dipoleWeight
    variable outFile
    variable outFileName
    variable QMEn
    variable QMDist
    variable refmolid
    variable simtype
    variable debug
    variable reChargeOverride
    variable reChargeOverrideCharges
    variable mode
    variable saT
    variable saTSteps
    variable dhIter
    variable saIter
    #variable guiMode

    
    set scriptFile [open $scriptFileName w]
    # load required packages
    puts $scriptFile "\# Load required packages"
    puts $scriptFile "package require forcefieldtoolkit"
    
    # set all chargeOpt variables
    puts $scriptFile "\n\# Set ChargeOpt Variables"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::psfPath $psfPath"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::pdbPath $pdbPath"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::resName $resName"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::parList {$parList}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::outFileName $outFileName"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::chargeGroups {$chargeGroups}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::chargeInit {$chargeInit}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::chargeBounds {$chargeBounds}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::chargeSum $chargeSum"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::baseHFLog $baseHFLog"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::baseMP2Log $baseMP2Log"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::watLog $watLog"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::logFileList {$logFileList}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::atomList {$atomList}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::indWeights {$indWeights}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::start $start"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::end $end"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::delta $delta"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::offset $offset"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::scale $scale"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::tol $tol"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::dWeight $dWeight"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::dipoleWeight $dipoleWeight"
    #puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::outFile $outFile"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::QMEn {$QMEn}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::QMDist {$QMDist}"
    #puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::refmolid $refmolid"
    #puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::simtype $simtype"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::debug $debug"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::reChargeOverride $reChargeOverride"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::reChargeOverrideCharges {$reChargeOverrideCharges}"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::mode \"$mode\""
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::dhIter $dhIter"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::saT $saT"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::saIter $saIter"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::saTSteps $saTSteps"
    puts $scriptFile "set ::ForceFieldToolKit::ChargeOpt::guiMode 0"
        
    # launch the optimization
    puts $scriptFile "\n\# Run the Optimization"
    puts $scriptFile "::ForceFieldToolKit::ChargeOpt::optimize"
    puts $scriptFile "\n\# Return gracefully"
    puts $scriptFile "return 1"
 
    # wrap up
    close $scriptFile
    return
}
#======================================================

##
## Charge Optimization Log Plotter (COLP)
##
#====================================
namespace eval ::ForceFieldToolKit::ChargeOpt::colp {

    variable w
    variable logPath
    variable plotColor
    variable plotHandle
    variable plotAutoscaling

    variable xmin
    variable xmax
    variable ymin
    variable ymax

}

#====================================
proc ::ForceFieldToolKit::ChargeOpt::colp::gui {} {

    # variables to initialize
    variable w

    # initialize
    ::ForceFieldToolKit::ChargeOpt::colp::init
    if { [winfo exists .colp] } {
        wm deiconify .colp
        return
    }

    set w [toplevel ".colp"]
    wm title $w "Charge Optimization Log Plotter (COLP)"

    # allow gui to resize with window
    grid columnconfigure $w 0 -weight 1
    grid rowconfigure $w 0 -weight 1

    # set a default geometry
    #wm geometry $w 500x500

    # build an hlf
    ttk::frame $w.hlf
    grid $w.hlf -column 0 -row 0 -sticky nswe

    # allow sections of the hlf to resize
    # set column expansions
    grid columnconfigure $w.hlf {0 1} -weight 0
    # set row expansions
    grid rowconfigure $w.hlf {0 2} -weight 0
    grid rowconfigure $w.hlf {1} -weight 1

    #
    #
    #

    # --- LOG LOADER ----------------------------------------------------------
    # build the section for loading the log file
    ttk::frame $w.hlf.loadLogFile
    ttk::separator $w.hlf.loadLogFile.sep1 -orient horizontal
    ttk::label $w.hlf.loadLogFile.lbl -text "LOG File:" -anchor w
    ttk::entry $w.hlf.loadLogFile.path -textvariable ::ForceFieldToolKit::ChargeOpt::colp::logPath -width 40
    ttk::button $w.hlf.loadLogFile.browse -text "Browse" \
        -command {
            set tempfile [tk_getOpenFile -title "Select a Charge Optimization LOG" -filetypes { {{LOG Files} {.log}} {{All Files} *} }]
            if {![string eq $tempfile ""]} { set ::ForceFieldToolKit::ChargeOpt::colp::logPath $tempfile }
        }
    ttk::button $w.hlf.loadLogFile.load -text "Load" \
        -command {
            # clear out any existing data
            .colp.hlf.data.energyTv delete [.colp.hlf.data.energyTv children {}]
            .colp.hlf.data.distTv delete [.colp.hlf.data.distTv children {}]
            .colp.hlf.data.objTv delete [.colp.hlf.data.objTv children {}]

            # parse the data from the log file and add to tv boxes
            ::ForceFieldToolKit::ChargeOpt::colp::loadLog $::ForceFieldToolKit::ChargeOpt::colp::logPath
        }
    ttk::separator $w.hlf.loadLogFile.sep2 -orient horizontal

    # grid the section for loading the log file
    grid $w.hlf.loadLogFile -column 0 -row 0 -columnspan 2 -sticky nswe
    grid columnconfigure $w.hlf.loadLogFile 0 -weight 0
    grid columnconfigure $w.hlf.loadLogFile 1 -weight 1
    grid columnconfigure $w.hlf.loadLogFile {2 3} -weight 0 -uniform ct1

    grid $w.hlf.loadLogFile.sep1 -column 0 -row 0 -columnspan 4 -sticky nswe -pady 4
    grid $w.hlf.loadLogFile.lbl -column 0 -row 1 -sticky nswe
    grid $w.hlf.loadLogFile.path -column 1 -row 1 -sticky nswe
    grid $w.hlf.loadLogFile.browse -column 2 -row 1 -sticky nswe
    grid $w.hlf.loadLogFile.load -column 3 -row 1 -sticky nswe
    grid $w.hlf.loadLogFile.sep2 -column 0 -row 2 -columnspan 4 -sticky nswe -pady 4
    # --- LOG LOADER ----------------------------------------------------------


    # --- DATA SECTION ----------------------------------------------------------
    # build the section for holding the data
    ttk::frame $w.hlf.data
    ttk::label $w.hlf.data.energyLbl -text "Energy" -anchor w
    ttk::treeview $w.hlf.data.energyTv -selectmode extended -yscrollcommand "$w.hlf.data.energyScroll set" -height 4
        $w.hlf.data.energyTv configure -column {atom color dE} -displaycolumns {atom color} -show {headings}
        $w.hlf.data.energyTv heading atom -text "atom" -anchor center
        $w.hlf.data.energyTv heading color -text "color" -anchor center
        $w.hlf.data.energyTv column atom -width 100 -stretch 0 -anchor center
        $w.hlf.data.energyTv column color -width 100 -stretch 0 -anchor center
    ttk::scrollbar $w.hlf.data.energyScroll -orient vertical -command "$w.hlf.data.energyTv yview"
    ttk::label $w.hlf.data.distLbl -text "Distance" -anchor w
    ttk::treeview $w.hlf.data.distTv -selectmode extended -yscrollcommand "$w.hlf.data.distScroll set" -height 4
        $w.hlf.data.distTv configure -column {atom color dD} -displaycolumns {atom color} -show {headings}
        $w.hlf.data.distTv heading atom -text "atom" -anchor center
        $w.hlf.data.distTv heading color -text "color" -anchor center
        $w.hlf.data.distTv column atom -width 100 -stretch 0 -anchor center
        $w.hlf.data.distTv column color -width 100 -stretch 0 -anchor center
    ttk::scrollbar $w.hlf.data.distScroll -orient vertical -command "$w.hlf.data.distTv yview"
    ttk::label $w.hlf.data.objLbl -text "Objective" -anchor w
    ttk::treeview $w.hlf.data.objTv -selectmode extended -yscrollcommand "$w.hlf.data.objScroll set" -height 4
        $w.hlf.data.objTv configure -column {type color obj} -displaycolumns {type color} -show {headings}
        $w.hlf.data.objTv heading type -text "type" -anchor center
        $w.hlf.data.objTv heading color -text "color" -anchor center
        $w.hlf.data.objTv column type -width 100 -stretch 0 -anchor center
        $w.hlf.data.objTv column color -width 100 -stretch 0 -anchor center
    ttk::scrollbar $w.hlf.data.objScroll -orient vertical -command "$w.hlf.data.objTv yview"

    # grid the section for holding the data
    grid $w.hlf.data -column 0 -row 1 -rowspan 2 -sticky nswe
    grid columnconfigure $w.hlf.data 0 -weight 0
    grid rowconfigure $w.hlf.data {1 3} -weight 1

    grid $w.hlf.data.energyLbl -column 0 -row 0 -sticky nswe
    grid $w.hlf.data.energyTv -column 0 -row 1 -sticky nswe
    grid $w.hlf.data.energyScroll -column 1 -row 1 -sticky nswe
    grid $w.hlf.data.distLbl -column 0 -row 2 -sticky nswe
    grid $w.hlf.data.distTv -column 0 -row 3 -sticky nswe
    grid $w.hlf.data.distScroll -column 1 -row 3 -sticky nswe
    grid $w.hlf.data.objLbl -column 0 -row 4 -sticky nswe
    grid $w.hlf.data.objTv -column 0 -row 5 -sticky nswe
    grid $w.hlf.data.objScroll -column 1 -row 5 -sticky nswe


    # TV Bindings
    # setup bindings to plot data on selection change
    bind $w.hlf.data.energyTv <<TreeviewSelect>> { ::ForceFieldToolKit::ChargeOpt::colp::plot }
    bind $w.hlf.data.distTv <<TreeviewSelect>> { ::ForceFieldToolKit::ChargeOpt::colp::plot }
    bind $w.hlf.data.objTv <<TreeviewSelect>> { ::ForceFieldToolKit::ChargeOpt::colp::plot }
    # setup binding to clear selection
    bind $w.hlf.data.energyTv <KeyPress-Escape> { .colp.hlf.data.energyTv selection remove [.colp.hlf.data.energyTv children {}] }
    bind $w.hlf.data.distTv <KeyPress-Escape> { .colp.hlf.data.distTv selection remove [.colp.hlf.data.distTv children {}] }
    bind $w.hlf.data.objTv <KeyPress-Escape> { .colp.hlf.data.objTv selection remove [.colp.hlf.data.objTv children {}] }
    # --- DATA SECTION ----------------------------------------------------------


    # --- PLOT ----------------------------------------------------------
    # build the section for holding the plot
    ttk::frame $w.hlf.plot
    set ::ForceFieldToolKit::ChargeOpt::colp::plotHandle [multiplot embed $w.hlf.plot \
        -title "Selected Charge Optimization Data" -xlabel "Iteration" -ylabel "delta\n  or\n obj" \
        -xsize 680 -ysize 450 -xmin 0 -xmax auto -ymin auto -ymax auto \
        -lines -linewidth 1]

    # grid the section for holding the plot
    grid $w.hlf.plot -column 1 -row 1 -sticky nswe
    # --- PLOT ----------------------------------------------------------


    # --- PLOT CONTROLS --------------------------------------------------------
    # build the section for holding the controls
    ttk::frame $w.hlf.controls
    grid $w.hlf.controls -column 1 -row 2 -sticky nswe
    grid columnconfigure $w.hlf.controls 0 -weight 1

    # make a frame for sliders
    ttk::frame $w.hlf.controls.sliders
    ttk::label $w.hlf.controls.sliders.xMinLbl -text "x-min" -anchor center
    ttk::scale $w.hlf.controls.sliders.xMin -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::ChargeOpt::colp::adjustScale xmin }
    ttk::label $w.hlf.controls.sliders.xMaxLbl -text "x-max" -anchor center
    ttk::scale $w.hlf.controls.sliders.xMax -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::ChargeOpt::colp::adjustScale xmax }

    ttk::label $w.hlf.controls.sliders.yMinLbl -text "y-min" -anchor center
    ttk::scale $w.hlf.controls.sliders.yMin -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::ChargeOpt::colp::adjustScale ymin }
    ttk::label $w.hlf.controls.sliders.yMaxLbl -text "y-max" -anchor center
    ttk::scale $w.hlf.controls.sliders.yMax -orient horizontal -from 0 -to 1.0 -command { ::ForceFieldToolKit::ChargeOpt::colp::adjustScale ymax }

    # grid the sliders
    grid $w.hlf.controls.sliders -column 0 -row 0 -sticky nswe
    grid $w.hlf.controls.sliders.xMinLbl -column 0 -row 0 -sticky nswe -padx "6 0"
    grid $w.hlf.controls.sliders.xMin -column 1 -row 0 -sticky nswe -padx 6
    grid $w.hlf.controls.sliders.xMaxLbl -column 0 -row 1 -sticky nswe -padx "6 0"
    grid $w.hlf.controls.sliders.xMax -column 1 -row 1 -sticky nswe -padx 6
    grid $w.hlf.controls.sliders.yMinLbl -column 2 -row 0 -sticky nswe
    grid $w.hlf.controls.sliders.yMin -column 3 -row 0 -sticky nswe -padx "6 0"
    grid $w.hlf.controls.sliders.yMaxLbl -column 2 -row 1 -sticky nswe
    grid $w.hlf.controls.sliders.yMax -column 3 -row 1 -sticky nswe -padx "6 0"

    # configure the sliders frame column/rows
    grid columnconfigure $w.hlf.controls.sliders {0 2} -weight 0
    grid columnconfigure $w.hlf.controls.sliders {1 3} -weight 1


    # separator    
    ttk::separator $w.hlf.controls.sep1 -orient horizontal
    grid $w.hlf.controls.sep1 -column 0 -row 1 -sticky nswe -pady 4

    # make a frame for axis dimensions manual entry
    ttk::frame $w.hlf.controls.xySet
    ttk::label $w.hlf.controls.xySet.xMinLbl -text "x-min" -anchor center
    ttk::entry $w.hlf.controls.xySet.xMin -textvariable ::ForceFieldToolKit::ChargeOpt::colp::xmin -width 4 -justify center
    ttk::label $w.hlf.controls.xySet.xMaxLbl -text "x-max" -anchor center
    ttk::entry $w.hlf.controls.xySet.xMax -textvariable ::ForceFieldToolKit::ChargeOpt::colp::xmax -width 4 -justify center
    ttk::label $w.hlf.controls.xySet.yMinLbl -text "y-min" -anchor center
    ttk::entry $w.hlf.controls.xySet.yMin -textvariable ::ForceFieldToolKit::ChargeOpt::colp::ymin -width 4 -justify center
    ttk::label $w.hlf.controls.xySet.yMaxLbl -text "y-max" -anchor center
    ttk::entry $w.hlf.controls.xySet.yMax -textvariable ::ForceFieldToolKit::ChargeOpt::colp::ymax -width 4 -justify center
    ttk::button $w.hlf.controls.xySet.set -text "Set Axis" -command {
        $::ForceFieldToolKit::ChargeOpt::colp::plotHandle configure -xmin $::ForceFieldToolKit::ChargeOpt::colp::xmin -xmax $::ForceFieldToolKit::ChargeOpt::colp::xmax -ymin $::ForceFieldToolKit::ChargeOpt::colp::ymin -ymax $::ForceFieldToolKit::ChargeOpt::colp::ymax
        $::ForceFieldToolKit::ChargeOpt::colp::plotHandle replot
}
    ttk::separator $w.hlf.controls.xySet.vsep1 -orient vertical
    ttk::checkbutton $w.hlf.controls.xySet.as -offvalue 0 -onvalue 1 -variable ::ForceFieldToolKit::ChargeOpt::colp::plotAutoscaling
    ttk::label $w.hlf.controls.xySet.asLbl -text "Axis Autoscaling" -anchor w

    # grid axis dimensions manual entry
    grid $w.hlf.controls.xySet -column 0 -row 2 -sticky nswe
    grid $w.hlf.controls.xySet.xMinLbl -column 0 -row 0 -sticky nswe -padx "2 0"
    grid $w.hlf.controls.xySet.xMin -column 1 -row 0 -sticky nswe -padx 6
    grid $w.hlf.controls.xySet.xMaxLbl -column 2 -row 0 -sticky nswe
    grid $w.hlf.controls.xySet.xMax -column 3 -row 0 -sticky nswe -padx 6
    grid $w.hlf.controls.xySet.yMinLbl -column 4 -row 0 -sticky nswe
    grid $w.hlf.controls.xySet.yMin -column 5 -row 0 -sticky nswe -padx 6
    grid $w.hlf.controls.xySet.yMaxLbl -column 6 -row 0 -sticky nswe
    grid $w.hlf.controls.xySet.yMax -column 7 -row 0 -sticky nswe -padx 6
    grid $w.hlf.controls.xySet.set -column 8 -row 0 -sticky nswe -padx 4
    grid $w.hlf.controls.xySet.vsep1 -column 9 -row 0 -sticky ns -padx 6
    grid $w.hlf.controls.xySet.as -column 10 -row 0 -sticky nswe
    grid $w.hlf.controls.xySet.asLbl -column 11 -row 0 -sticky nswe -padx "0 10"

    # configure the axis dimensions column/rows
    grid columnconfigure $w.hlf.controls.xySet 9 -weight 1

    # separator
    ttk::separator $w.hlf.controls.sep2 -orient horizontal
    grid $w.hlf.controls.sep2 -column 0 -row 3 -sticky nswe -pady 4

    # make a frame for plot color selection
    ttk::frame $w.hlf.controls.color
    ttk::label $w.hlf.controls.color.plotColorLbl -text "Set Plot Color:" -anchor center
    ttk::menubutton $w.hlf.controls.color.plotColor -direction below -menu $w.hlf.controls.color.plotColor.menu -textvariable ::ForceFieldToolKit::ChargeOpt::colp::plotColor -width 12
    menu $w.hlf.controls.color.plotColor.menu -tearoff no
        $w.hlf.controls.color.plotColor.menu add command -label "black" -command {::ForceFieldToolKit::ChargeOpt::colp::setColor "black"}
        $w.hlf.controls.color.plotColor.menu add command -label "red" -command {::ForceFieldToolKit::ChargeOpt::colp::setColor "red"}
        $w.hlf.controls.color.plotColor.menu add command -label "orange" -command {::ForceFieldToolKit::ChargeOpt::colp::setColor "orange"}
        $w.hlf.controls.color.plotColor.menu add command -label "green" -command {::ForceFieldToolKit::ChargeOpt::colp::setColor "green"}
        $w.hlf.controls.color.plotColor.menu add command -label "blue" -command {::ForceFieldToolKit::ChargeOpt::colp::setColor "blue"}
        $w.hlf.controls.color.plotColor.menu add command -label "purple" -command {::ForceFieldToolKit::ChargeOpt::colp::setColor "purple"}
        $w.hlf.controls.color.plotColor.menu add command -label "magenta" -command {::ForceFieldToolKit::ChargeOpt::colp::setColor "magenta"}

    # grid plot color selection
    grid $w.hlf.controls.color -column 0 -row 4 -sticky nswe
    grid $w.hlf.controls.color.plotColorLbl -column 0 -row 0 -sticky nswe
    grid $w.hlf.controls.color.plotColor -column 1 -row 0 -sticky nswe
    # --- PLOT CONTROLS --------------------------------------------------------

}
#====================================
proc ::ForceFieldToolKit::ChargeOpt::colp::init {} {

    # localize/initialize
    variable logPath ""
    variable plotColor ""
    variable plotHandle
    variable plotAutoscaling 1

    variable xmin "auto"
    variable xmax "auto"
    variable ymin "auto"
    variable ymax "auto"

}
#====================================
proc ::ForceFieldToolKit::ChargeOpt::colp::loadLog { logFile } {

    # initialize arrays
    array set en {}
    array set dist {}

    # initialize lists
    set objE {}
    set objD {}
    set objDip {}
    set objT {}
    set atomNames {}

    # initialize other things
    set inFile [open $logFile r]
    set readState 0
    set itCount 0
    set nameRead 1


    # parse data from the log file
    while { ![eof $inFile] } {

        # read in the next line of the file
        set inLine [gets $inFile]

        # based on the line contents, take some action
        switch -regexp $inLine {
            {^Current test charges} {}

            {^Iteration:} {
                # parse the iteraction value
                set itCount [lindex $inLine 1]
                # modify settings to turn line reading on
                set readState 1
                set aID 0
            }

            {^Dipole} {}

            {^Current objective} {
                # read obj data
                lappend objT [lindex $inLine 3]
                lappend objE [lindex $inLine 6]
                lappend objD [lindex $inLine 8]
                lappend objDip [lindex $inLine 10]
                # modify settings to turn line reading off
                set readState 0
                # turn off name reading (only required for first iteration)
                set nameRead 0
            }

            default {
                if { $readState } {
                    # parse the energy and distance values, add to the array
                    # based on atom ID values
                    lappend en($aID) [string range [lindex $inLine 5] 1 end-1]
                    lappend dist($aID) [lindex $inLine 9]

                    # parse atom names from the first iteration data
                    # turned off after the first iteration
                    if { $nameRead } {
                        lappend atomNames [lindex $inLine 0]
                    }

                    # advance the atom ID setting
                    incr aID

                } else {
                    continue
                }
            }
        }; # end switch
    }; # end while/file reading

    # clean up the file
    close $inFile

    # post-processing and add data to tv boxes
    # energy tv box
    foreach key [lsort -dictionary [array names en]] {
        .colp.hlf.data.energyTv insert {} end -values [list [lindex $atomNames $key] "black" $en($key)]
    }
    # distance tv box
    foreach key [lsort -dictionary [array names dist]] {
        .colp.hlf.data.distTv insert {} end -values [list [lindex $atomNames $key] "black" $dist($key)]
    }
    # obj tv box
    .colp.hlf.data.objTv insert {} end -values [list total "black" $objT]
    .colp.hlf.data.objTv insert {} end -values [list energy "red" $objE]
    .colp.hlf.data.objTv insert {} end -values [list distance "blue" $objD]
    .colp.hlf.data.objTv insert {} end -values [list dipole "green" $objDip]

    # clean up
    array unset en; array unset dist
    unset objE objD objDip objT atomNames 

}
#====================================
proc ::ForceFieldToolKit::ChargeOpt::colp::setColor { color } {
    # sets data colors

    # set the plot color menubutton to $color
    set ::ForceFieldToolKit::ChargeOpt::colp::plotColor $color

    # cycle through selected entries and set color
    # energy tv
    foreach ele [.colp.hlf.data.energyTv selection] {
        .colp.hlf.data.energyTv set $ele color $color
    }
    # distance tv
    foreach ele [.colp.hlf.data.distTv selection] {
        .colp.hlf.data.distTv set $ele color $color
    }
    # obj tv
    foreach ele [.colp.hlf.data.objTv selection] {
        .colp.hlf.data.objTv set $ele color $color
    }

}
#====================================
proc ::ForceFieldToolKit::ChargeOpt::colp::plot {} {
    # plots the selected datasets

    # localize variables
    variable plotHandle

    # aggregate the datasets
    set datasets {}; set colorsets {}; set legend {}
    # energy tv
    foreach ele [.colp.hlf.data.energyTv selection] {
        lappend datasets [.colp.hlf.data.energyTv set $ele dE]
        lappend colorsets [.colp.hlf.data.energyTv set $ele color]
        lappend legend [.colp.hlf.data.energyTv set $ele atom]-En
    }
    # distance tv
    foreach ele [.colp.hlf.data.distTv selection] {
        lappend datasets [.colp.hlf.data.distTv set $ele dD]
        lappend colorsets [.colp.hlf.data.distTv set $ele color]
        lappend legend [.colp.hlf.data.distTv set $ele atom]-Dist
    }
    # obj tv
    foreach ele [.colp.hlf.data.objTv selection] {
        lappend datasets [.colp.hlf.data.objTv set $ele obj]
        lappend colorsets [.colp.hlf.data.objTv set $ele color]
        lappend legend [.colp.hlf.data.objTv set $ele type]
    }

    # clear any existing data from the plot
    $plotHandle clear

    if { [llength $datasets] == 0 } { $plotHandle replot; return }

    # cycle through each dataset
    for {set i 0} {$i < [llength $datasets]} {incr i} {
        # parse the y data
        set ydata [lindex $datasets $i]

        # build the x data
        set xdata {}
        for {set x 0} {$x < [llength $ydata]} {incr x} {
            lappend xdata $x
        }

        # parse out the plot color
        set plotColor [lindex $colorsets $i]

        # parse out the legend text
        set legendTxt [lindex $legend $i]

        # add the data to the plot
        $plotHandle add $xdata $ydata -lines -linewidth 1 -linecolor $plotColor -legend $legendTxt
    }

    # update the scrolling controls
    # x-axis
    set totalDataPoints [llength [lindex [$plotHandle xdata] 0]]
    .colp.hlf.controls.sliders.xMin configure -from 0 -to $totalDataPoints
    .colp.hlf.controls.sliders.xMax configure -from 0 -to $totalDataPoints

    # y-axis
    set globalYmin 1000000000000000000000000.0
    set globalYmax -1000000000000000000000000.0
    foreach yDataSet [$plotHandle ydata] {
        foreach ele $yDataSet {
            if { $ele < $globalYmin } { set globalYmin $ele }
            if { $ele > $globalYmax } { set globalYmax $ele }
        }
    }
    if { $globalYmin > 0 } { set globalYmin 0 }
    if { $globalYmax < 0 } { set globalYmax 0 }
    .colp.hlf.controls.sliders.yMin configure -from $globalYmin -to $globalYmax
    .colp.hlf.controls.sliders.yMax configure -from $globalYmin -to $globalYmax

    # reset sliders and plot accordintly, if autoscaling is turned on
    if { $::ForceFieldToolKit::ChargeOpt::colp::plotAutoscaling } {
        $plotHandle configure -xmin 0 -xmax $totalDataPoints -ymin $globalYmin -ymax $globalYmax
        .colp.hlf.controls.sliders.xMin configure -value 0
        .colp.hlf.controls.sliders.xMax configure -value $totalDataPoints
        .colp.hlf.controls.sliders.yMin configure -value $globalYmin
        .colp.hlf.controls.sliders.yMax configure -value $globalYmax
    }

    # update the plot
    $plotHandle replot

}
#====================================
proc ::ForceFieldToolKit::ChargeOpt::colp::adjustScale {scaleType value} {
    # proc for adjusting the plot axis via scales

    # localize variables
    variable plotHandle
    
    # adjust plot axis
    $plotHandle configure -$scaleType $value
    $plotHandle replot

}
#====================================

