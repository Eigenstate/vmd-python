#
# $Id: fftk_BuildPar.tcl,v 1.18 2016/10/20 16:15:00 mayne Exp $
#

#======================================================
# BuildPar - A collection of tools used to analyze, prepare, and update parameter files
# Current Tools:
# -- Identify missing parameters for a molecule when compared against existing parameter set(s)
#    This tool can also write an initialized (zero-filled) parameter file
# -- Assign VDW/LJ parameters from analogy; includes a "parameter explorer"
# -- Update existing parameter files based on results from ffTK optimizations
# -- Analyze input from the CGenFF Program webserver (https://cgenff.paramchem.org/) to
#    construct a PSF/PDB file pair and provide a non-zero starting point for charges and
#    bonded parameters
#======================================================
namespace eval ::ForceFieldToolKit::BuildPar {

    # Identify Missing Parameters
    variable idMissingPSF
    variable idMissingPDB
    variable idMissingRefParList
    variable idMissingParOutPath

    # Assign VDW/LJ

    # Prepare from CGenFF Program
    variable cgenffMol
    variable cgenffStr
    variable cgenffOutPath
    variable cgenffResname
    variable cgenffChain
    variable cgenffSegment
    variable cgenffExistingPars
    variable cgenffAnalogyPars


    # Update PAR to Optimized Parameters
    variable updateInputParPath
    variable updateLogPath
    variable updateOutParPath
}
#======================================================
proc ::ForceFieldToolKit::BuildPar::init {} {
    # (Re)Initialize BuildPar namespace variables

    # localize variables (separated by tool)
    # Identify/Analyze missing parameters
    variable idMissingPSF
    variable idMissingPDB
    variable idMissingRefParList
    variable idMissingParOutPath

    # Assign VDW/LJ

    # Prepare from CGenFF Program
    variable cgenffMol {}
    variable cgenffStr {}
    variable cgenffOutPath {}
    variable cgenffResname LIG
    variable cgenffChain L
    variable cgenffSegment L
    variable cgenffExistingPars {}
    variable cgenffAnalogyPars {}

    # Update PAR to Optimized Parameters
    variable updateInputParPath
    variable updateLogPath
    variable updateOutParPath

    # initialize
    set idMissingPSF {}
    set idMissingPDB {}
    set idMissingRefParList {}
    set idMissingParOutPath {}

    set updateInputParPath {}
    set updateLogPath {}
    set updateOutParPath {}
}
#======================================================
proc ::ForceFieldToolKit::BuildPar::sanityCheck { procType } {
    # checks to see that appropriate information is set
    # prior to running any of the BuildPar procedures
    
    # returns 1 if all input is sane
    # returns 0 if there is a problem
    
    # local variables
    set errorList {}
    set errorText ""
    
    # Since BuildPar consists of several separate tools, the sanity check is passed an argument that
    # identifies the specific check to perform, for which it builds an error list
    switch -exact $procType {
        idMissingAnalyze {
            # check performed when analyzing the molecule (crosscheck against associated parameter sets)
            # localize relevant variables
            foreach var {idMissingPSF idMissingPDB idMissingRefParList} { variable $var }
            # make sure that a PSF file was entered and exists
            if { $idMissingPSF eq "" } { lappend errorList "No PSF file was specified." } \
            else { if { ![file exists $idMissingPSF] } { lappend errorList "Cannot find PSF file." } }
            # make sure that a PDB file was entered and exists
            if { $idMissingPDB eq "" } { lappend errorList "No PDB file was specified." } \
            else { if { ![file exists $idMissingPDB] } { lappend errorList "Cannot find PDB file." } }
            # if there are reference parameter files, make sure that they exist
            if { [llength $idMissingRefParList] > 0 } {
                foreach f $idMissingRefParList {
                    if { ![file exists $f] } { lappend errorList "Cannot find associated parameter file: $f" }
                }
            }
        }

        idMissingInitPars {
            # check performed when writing an initizialized parameter file
            # localize relevant variables
            foreach var {idMissingPSF idMissingParOutPath} { variable $var }
            # make sure that a PSF file was entered and exists
            if { $idMissingPSF eq "" } { lappend errorList "No PSF file was specified." } \
            else { if { ![file exists $idMissingPSF] } { lappend errorList "Cannot find PSF file." } }

            # make sure that the output path/filename were entered and user has write permissions to the directory
            if { $idMissingParOutPath eq "" } { lappend errorList "No output path was specificed." }
            if { $idMissingParOutPath ne "" && ![file writable [file dirname $idMissingParOutPath]] } { lappend errorList "Cannot write to the specified directory" }
        }

        updateOptPars {
            # check performed when updating parameters from ffTK optimization logs
            # localize variables
            foreach var {updateInputParPath updateLogPath updateOutParPath} { variable $var }
            # make sure that an input parameter file was entered and exists
            if { $updateInputParPath eq "" } { lappend errorList "No input parameter file was specified." } \
            else { if { ![file exists $updateInputParPath] } { lappend errorList "Cannot find input parameter file." } }

            # make sure that an optimization log file was entered and exists
            if { $updateLogPath eq "" } { lappend errorList "No optimization log file was specified." } \
            else { if { ![file exists $updateLogPath] } { lappend errorList "Cannot find optimization log file." } }

            # make sure a savename was entered and that the user can write to that directory
            if { $updateOutParPath eq "" } { lappend errorList "No output path was specified." }
            if { $updateOutParPath ne "" && ![file writable [file dirname $updateOutParPath]] } { lappend errorList "Cannot write to output path." }

        }
    }

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
proc ::ForceFieldToolKit::BuildPar::getRefPars { prmlist } {
    # reads in a list of parameter files and returns
    # lists of type definitions present in input prm files

    # initialize some lists
    set bonds {}
    set angles {}
    set dihedrals {}
    set impropers {}
    set vdws {}
    
    set partypes {bonds angles dihedrals impropers vdws}
    
    foreach prmfile $prmlist {
        # read in parameters from file
        set tempParList [::ForceFieldToolKit::SharedFcns::readParFile $prmfile]
        # cycle through parameter sections
        for {set i 0} {$i <= [llength $partypes]} {incr i} {
            # cycle through each parameter definition/entry
            foreach parDef [lindex $tempParList $i] {
                # append the typeDef to the appropriate list
                lappend [lindex $partypes $i] [lindex $parDef 0]
            }
        }
    }
    
    # remove duplicates (mostly to reduce list size that is stored/passed)
    set bonds [lsort -unique $bonds]
    set angles [lsort -unique $angles]
    set dihedrals [lsort -unique $dihedrals]
    set impropers [lsort -unique $impropers]
    set vdws [lsort -unique $vdws]

    # return the relevent lists for building a supplemental
    # zero-ed out prm file
    return [list $bonds $angles $dihedrals $vdws]
}
#======================================================
proc ::ForceFieldToolKit::BuildPar::getMolecPars { psf pdb {deleteMol 0} } {
    # finds bond, angle, dihedral, vdw parameters that are
    # required for a molecule
    
    # load the psf file
    mol new     $psf waitfor all
    mol addfile $pdb waitfor all
    
    # initilize some variables
    set uniq_bond_list {}
    set uniq_ang_list {}
    set uniq_dih_list {}
    set uniq_vdw_list {}

    # bonds
    set bond_list [topo getbondlist]
    foreach bond_entry $bond_list {
        set at1 [[atomselect top "index [lindex $bond_entry 0]"] get type]
        set at2 [[atomselect top "index [lindex $bond_entry 1]"] get type]
        set bond_type [list $at1 $at2]
        # test forward and reverse patterns for duplicate bond types
        if {[lsearch -exact $uniq_bond_list $bond_type] != -1} {
            #puts "bond: $bond_type is a forward duplicate"
        } elseif {[lsearch -exact $uniq_bond_list [lreverse $bond_type]] != -1} {
            #puts "bond: $bond_type is a reverse duplicate"
        } else {
            #puts "bond: $bond_type is unique"
            lappend uniq_bond_list $bond_type
        }
    }

    # angles
    set ang_list [topo getanglelist]
    foreach ang_entry $ang_list {
        set at1 [[atomselect top "index [lindex $ang_entry 1]"] get type]
        set at2 [[atomselect top "index [lindex $ang_entry 2]"] get type]
        set at3 [[atomselect top "index [lindex $ang_entry 3]"] get type]
        set ang_type [list $at1 $at2 $at3]
        # test forward and reverse patterns for duplicate angle types
        if {[lsearch -exact $uniq_ang_list $ang_type] != -1} {
            #puts "angle: $ang_type is a forward duplicate"
        } elseif {[lsearch -exact $uniq_ang_list [lreverse $ang_type]] != -1} {
            #puts "angle: $ang_type is a reverse duplicate"
        } else {
            #puts "angle: $ang_type is unique"
            lappend uniq_ang_list $ang_type
        }
    }

    # dihedrals
    set dih_list [topo getdihedrallist]
    foreach dih_entry $dih_list {
        set at1 [[atomselect top "index [lindex $dih_entry 1]"] get type]
        set at2 [[atomselect top "index [lindex $dih_entry 2]"] get type]
        set at3 [[atomselect top "index [lindex $dih_entry 3]"] get type]
        set at4 [[atomselect top "index [lindex $dih_entry 4]"] get type]
        set dih_type [list $at1 $at2 $at3 $at4]
        # test forward and reverse patterns for duplicate dihedral types
        if {[lsearch -exact $uniq_dih_list $dih_type] != -1} {
            #puts "dihedral: $dih_type is a forward duplicate"
        } elseif {[lsearch -exact $uniq_dih_list [lreverse $dih_type]] != -1} {
            #puts "dihdedral: $dih_type is a reverse duplicate"
        } else {
            #puts "dihedral: $dih_type is unique"
            lappend uniq_dih_list $dih_type
        }
    }

    # vdws
    set uniq_vdw_list [topo atomtypenames]
    
    # clean up
    if { $deleteMol } { mol delete top }
    
    # return uniq parameters present in molecule
    return [list $uniq_bond_list $uniq_ang_list $uniq_dih_list $uniq_vdw_list]
}
#======================================================
proc ::ForceFieldToolKit::BuildPar::crossCheckPars { molecPars refPars } {
    # cross checks molecule pars against a reference set of pars
    # returns molec pars not found in reference par set
    
    # initialize some lists
    set unpar_bonds {}
    set unpar_angles {}
    set unpar_diheds {}
    set unpar_vdws {}

    for {set i 0} {$i<4} {incr i} {
        foreach prmtype [lindex $molecPars $i] {
            if {[lsearch -exact [lindex $refPars $i] $prmtype] != -1} {
                #puts "\tparameter: $prmtype has forward parameter
                continue
            } elseif {[lsearch -exact [lindex $refPars $i] [lreverse $prmtype]] != -1} {
                #puts "\tparameter: $prmtype has reverse parameter
                continue
            } else {
                switch -exact $i {
                    0 {lappend unpar_bonds $prmtype}
                    1 {lappend unpar_angles $prmtype}
                    2 {lappend unpar_diheds $prmtype}
                    3 {lappend unpar_vdws $prmtype}
                }; # end switch
            }; # end prm search test (else)
        }; # end cycling through actual prm lists (foreach loop)
    }; # end cycling through each prm type (for loop)

    # return the missing parameters
    return [list $unpar_bonds $unpar_angles $unpar_diheds $unpar_vdws]
}
#======================================================
proc ::ForceFieldToolKit::BuildPar::buildInitParFile { missingPars } {
    # builds a zeroed-out parameter list

    # input: set of parameters to zero in the format:
    #           |  {
    # bonds     |       { {bt1 bt2} {bt1 bt2} ... }
    # angles    |       { {at1 at2 at3} {at1 at2 at3} ... }
    # dihedrals |       { {dt1 dt2 dt3 dt4} {dt1 dt2 dt3 dt4} ... }
    # nonbonded |       { t1 t1 t1...}
    #           |  }

    # output: a parameter file with zeroed-out terms

    # localize relevant variables
    variable idMissingParOutPath

    # run a sanity check; this will need to be updated
    if { ![::ForceFieldToolKit::BuildPar::sanityCheck idMissingInitPars] } { return }

    # format the missing parameters so that we can pass to a generic par writer proc
    set Bonds {}
    foreach bondDef [lindex $missingPars 0] {
        lappend Bonds [list $bondDef {0.0 0.0} {}]
    }
    set Angles {}
    foreach angleDef [lindex $missingPars 1] {
        lappend Angles [list $angleDef {0.0 0.0} {} {}]
    }
    set Dihedrals {}
    foreach dihDef [lindex $missingPars 2] {
        lappend Dihedrals [list $dihDef {0.0 1 0.0} {}]
    }
    # impropers (not set here)
    set vdws {}
    foreach vdwDef [lindex $missingPars 3] {
        lappend vdws [list $vdwDef {0.0 0.0} {} {! SET BY ANALOGY!!!}]
    }
    
    # build the parlist
    set parList [list $Bonds $Angles $Dihedrals {} $vdws]
    
    # write the zeroed out par file
    ::ForceFieldToolKit::SharedFcns::writeParFile $parList $idMissingParOutPath
}
#======================================================
proc ::ForceFieldToolKit::BuildPar::buildUpdatedParFile {} {
    # reads in a parameter file and optimization log (bonds/angles, dihedral)
    # writes a parameter file with updated parameters

    # localize variables
    variable updateInputParPath
    variable updateLogPath
    variable updateOutParPath
    
    # initialize some variables
    set optBonds {}
    set optAngles {}
    set optDihedrals {}

    # run a sanity check
    if { ![::ForceFieldToolKit::BuildPar::sanityCheck updateOptPars ] } { return }

    # read in the template par file
    set parData [::ForceFieldToolKit::SharedFcns::readParFile $updateInputParPath]
    
    # parse the optimization log file
    set inFile [open $updateLogPath r]
    set readState 0
    
    while { ![eof $inFile] } {
        # read a line at a time
        set inLine [gets $inFile]
        
        # determine if we've reached the final parameter data
        switch -exact $inLine {
            "FINAL PARAMETERS" { set readState 1 }
            "END" { set readState 0 }
            default {
                if { $readState } {
                    # parse and append parameters to appropriate list
                    switch -exact [lindex $inLine 0] {
                        "bond" { lappend optBonds [list [lindex $inLine 1] [list [lindex $inLine 2] [lindex $inLine 3]] {}] }
                        "angle" { lappend optAngles [list [lindex $inLine 1] [list [lindex $inLine 2] [lindex $inLine 3]] {} {}] }
                        "dihedral" { lappend optDihedrals [list [lindex $inLine 1] [list [lindex $inLine 2] [lindex $inLine 3] [lindex $inLine 4]] {}] }
                        default { continue }
                    }
                } else {
                    continue
                }
            }
        }; # end outer switch
    }; # end of while loop (parse log file)
    
    # update bonds
    # build a list of bond definitions from the input par data
    set oldBondDefs {}
    foreach bondEntry [lindex $parData 0] {
        lappend oldBondDefs [lindex $bondEntry 0]
    }
    # cycle through each bond with parameters to update
    foreach bond2update $optBonds {
        # search against the input parameter data
        set testfwd [lsearch $oldBondDefs [lindex $bond2update 0]]
        set testrev [lsearch $oldBondDefs [lreverse [lindex $bond2update 0]]]
        if { $testfwd == -1 && $testrev ==-1 } {
            puts "ERROR: Bond definition to update: [lindex $bond2update 0] was not found in input parameter set"
        } elseif { $testfwd > -1 } {
            lset parData 0 $testfwd 1 [lindex $bond2update 1]
        } elseif { $testrev > -1 } {
            lset parData 0 $testrev 1 [lindex $bond2update 1]
        }
    }
    
    # update angles
    # build a list of angle definitions from the input par data
    set oldAngleDefs {}
    foreach angleEntry [lindex $parData 1] {
        lappend oldAngleDefs [lindex $angleEntry 0]
    }
    # cycle through each angle with parameters to update
    foreach angle2update $optAngles {
        # search against the input parameter data
        set testfwd [lsearch $oldAngleDefs [lindex $angle2update 0]]
        set testrev [lsearch $oldAngleDefs [lreverse [lindex $angle2update 0]]]
        if { $testfwd == -1 && $testrev == -1 } {
            puts "ERROR: Angle definition to update: [lindex $angle2update 0] was not found in input parameter set"
        } elseif { $testfwd > -1 } {
            lset parData 1 $testfwd 1 [lindex $angle2update 1]
        } elseif { $testrev > -1 } {
            lset parData 1 $testrev 1 [lindex $angle2update 1]
        }
    }
    
    # update dihedrals
    # due to multiplicities, we have to handle this one differently
    
    # initialize and populate array for old dihedral information
    # oldDihPars(type def) = {  {k1 mult1 delta1} ...{kN multN deltaN}  }
    array set oldDihPars {}
    #array set oldDihCom {}
    foreach dihEntry [lindex $parData 2] {
        lappend oldDihPars([lindex $dihEntry 0]) [lindex $dihEntry 1]
        #lappend oldDihCom([lindex $dihEntry 0]) [lindex $dihEntry 2]
    }
    
    # cycle through each dihedral with parameters to update
    array set newDihPars {}
    foreach dih2update $optDihedrals {
        # search against the input parameter data and determine the type def order (fwd or rev)
        # if found in the old parameter set, blow up the old definition
        # if not found in old paremeter set, check the new one
        # if not found in either, then it's an error of some sort
        if { [info exists oldDihPars([lindex $dih2update 0])] } {
            set currTypeDef [lindex $dih2update 0]
            array unset oldDihPars [lindex $dih2update 0]
        } elseif { [info exists dihPars([lreverse [lindex $dih2update 0]])] } {
            set currTypeDef [lreverse [lindex $dih2update 0]]
            array unset oldDihPars [lreverse [lindex $dih2update 0]]
        } elseif { [info exists newDihPars([lindex $dih2update 0])] } {
            set currTypeDef [lindex $dih2update 0]
        } elseif { [info exists newDihPars([lreverse [lindex $dih2update 0]])] } {
            set currTypeDef [lreverse [lindex $dih2update 0]]
        } else {
            puts "ERROR: Dihedral definition to update: [lindex $dih2update 0] was not found in input parameter set"
            continue
        }
        
        # add to the new array
        lappend newDihPars($currTypeDef) [lindex $dih2update 1]
        #lappend newDihCom($currTypeDef) [lindex $dih2update 2]
    }
    
    # convert dihedral data array back into parData format
    set dihParUpdate {}
    # cycle through any remaining old parameters
    foreach key [array names oldDihPars] {
        # cycle through each mult/comment for a given typeDef
        for {set i 0} {$i < [llength $oldDihPars($key)]} {incr i} {
            #                          {def}      {k mult delta}             {comment}
            #lappend dihParUpdate [list $key [lindex $oldDihPars($key) $i] [lindex $oldDihCom($key) $i]]
            lappend dihParUpdate [list $key [lindex $oldDihPars($key) $i] {}]
        }
    }
    # cycle through new parameters
    foreach key [array names newDihPars] {
        # cycle through each mult/comment for a given typeDef
        for {set i 0} {$i < [llength $newDihPars($key)]} {incr i} {
            #                          {def}      {k mult delta}             {comment}
            #lappend dihParUpdate [list $key [lindex $newDihPars($key) $i] [lindex $newDihCom($key) $i]]
            lappend dihParUpdate [list $key [lindex $newDihPars($key) $i] {}]
        }
    }
    
    # replace the input dihedral parameters with the updated parameters
    lset parData 2 $dihParUpdate    


    
    # write the updated parameter file
    ::ForceFieldToolKit::SharedFcns::writeParFile $parData $updateOutParPath
}
#======================================================
proc ::ForceFieldToolKit::BuildPar::analyzeCGenFF {} {
    # Construct molecule using CGenFF input and parse out parameter information
    # Parameter information is split into existing pars and analogy pars

    # PASSED:  nothing
    # RETURNS: molid of the loaded molecule

    # NOTE: we will not concern ourselves with the GUI here, leaving that to a GUI proc

    # We have to to this manually because:
    # - CGenFF output tends to mangle resnames in topology definition (we should file a bug report / feature request)
    # - PSFGen doesn't read MOL2 files (that i'm aware of; conversion is potentially trivial but annoying)
    # - CGenFF Program output potentially changes atom names; atom match is performed on order (i.e., index)

    # load any required packages
    package require topotools

    # localize relevant variables
    variable cgenffMol
    variable cgenffStr
    variable cgenffResname
    variable cgenffChain
    variable cgenffSegment
    variable cgenffExistingPars
    variable cgenffAnalogyPars

    # sanity check
    # SANITY CHECK SHOULD GO HERE

    # load molecule
    set molid [mol new $cgenffMol waitfor all]

    # since we are changing the atom names below, the "name" coloring method gets mangled
    # go ahead and switch to color by element
    mol modcolor 0 $molid Element

    # set molecule-level data
    set sel [atomselect $molid "all"]
    $sel set resname $cgenffResname
    $sel set chain   $cgenffChain
    $sel set segname $cgenffSegment
    topo clearbonds

    # process the CGenFf Program data
    set infile [open $cgenffStr r]

    # handle the topology section of the stream file
    # only supports ATOM, BOND, IMPR sections
    # note: charge penalty is stored in beta
    set atomIndex_read 0 ; # key to map the entry to the loaded mol2 structure
    while { ![eof $infile] } {
        set inline [string trim [gets $infile]]

        switch [lindex $inline 0] {
            {END} { break }
            {ATOM} {
                set sel [atomselect $molid "index $atomIndex_read"]
                #set sel [atomselect $molid "name [lindex $inline 1]"]
                $sel set name   [lindex $inline 1]
                $sel set type   [lindex $inline 2]
                $sel set charge [lindex $inline 3]
                $sel set beta   [lindex $inline 5]
                $sel delete
                incr atomIndex_read
            }
            {BOND} {
                set atomNameList [lrange $inline 1 2]
                #puts "atomNameList: $atomNameList"; flush stdout
                #lassign $inline key a1 a2

                set atomIndexString ""
                foreach atomName $atomNameList {
                    set sel [atomselect $molid "name $atomName"]
                    set atomIndexString [concat $atomIndexString [$sel get index]]
                    $sel delete
                }

                #topo addbond $nameArr($a1) $nameArr($a2)
                #puts "attempting: topo addbond $atomIndexString"; flush stdout
                eval topo addbond $atomIndexString
                unset atomIndexString
            }
            {IMPR} {
                set atomNameList [lrange $inline 1 4]
                #puts "atomNameList: $atomNameList"; flush stdout
                #lassign $inline key a1 a2 a3 a4

                set atomIndexString ""
                foreach atomName $atomNameList {
                    set sel [atomselect $molid "name $atomName"]
                    set atomIndexString [concat $atomIndexString [$sel get index]]
                    $sel delete
                }

                #topo addimproper $nameArr($a1) $nameArr($a2) $nameArr($a3) $nameArr($a4)
                #puts "attemping: topo addimproper $atomIndexString"; flush stdout
                eval topo addimproper $atomIndexString
                unset atomIndexString
            }
        }
    }
    
    topo guessangles -molid $molid
    topo guessdihedrals -molid $molid
    topo guessatom element mass
    mol reanalyze $molid

    # Handle the parameter section of the stream file
    # Only currently supports bonds, angles, dihedrals, impropers
    # CGenFF Program supports two options -- 1) return ALL parameters, 2) return only missing parameters
    # We will split these into two different sets, so that only missing parameters can be loaded into GUI
    # It appears that even in case of option #1 CGenFF Program only hands back bonds, angles, dih, impr
    #   notably not returning vdw; this should be check periodically
    # Note that exising parameters are stored in the format expected by SharedFcns::writeParFile
    #   while missing parameters are stored in a format more amenable to populating tv boxes in GUI
    # Heavily modified from SharedFcns::readParFile
    set bonds_exist {}; set angles_exist {}; set dihedrals_exist {}; set impropers_exist {}
    set bonds_miss {};  set angles_miss {};  set dihedrals_miss {};  set impropers_miss {}
    set readstate 0
    while { ![eof $infile] } {
        set inLine [gets $infile]
        switch -regexp $inLine {
            {^[ \t]*$}     { continue }
            {^[ \t]*\*.*}  { continue }
            {^[ \t]*!.*}   { continue }
            {^[a-z]+}      { continue }
            {^BONDS.*}     { set readstate BOND }
            {^ANGLES.*}    { set readstate ANGLE }
            {^DIHEDRALS.*} { set readstate DIH }
            {^IMPROPER.*}  { set readstate IMPROP }
            {^CMAP.*}      { set readstate CMAP }
            {^NONBONDED.*} { set readstate VDW }
            {^HBOND.*}     { continue }
            {^END.*}       { break }
            default {
                # break out the parameter data vs comment data (if any)
                set inLineSplit [split $inLine \!]
                set prmData [lindex $inLineSplit 0]
                if { [llength $inLineSplit] > 1 } {
                    set prmComment [string trim [join [lrange $inLineSplit 1 end] ","]]
                } else {
                    set prmComment {}
                }
                # determine if parameter is existing or missing, and set flags accordingly
                set penaltySearch [lsearch -exact [string toupper $prmComment] "PENALTY="]
                if { $penaltySearch > -1 } { set penaltyScore [lindex $prmComment [expr $penaltySearch + 1]] }
                # parse the parameter data
                switch -exact $readstate {
                    0 { continue }
                    BOND {
                        set typedef [lrange $prmData 0 1]
                        set k       [lindex $prmData 2]
                        set b0      [lindex $prmData 3]
                        if { $penaltySearch == -1 } {
                            lappend bonds_exist [list $typedef [list $k $b0] $prmComment]
                        } else {
                            lappend bonds_miss [list $typedef $k $b0 $penaltyScore $prmComment]
                        }
                    }
                    ANGLE {
                        set typedef [lrange $prmData 0 2]
                        set k       [lindex $prmData 3]
                        set theta   [lindex $prmData 4]
                        set kub     [lindex $prmData 5]
                        set s       [lindex $prmData 6]
                        if { $penaltySearch == -1 } {
                            lappend angles_exist [list $typedef [list $k $theta] [list $kub $s] $prmComment]
                        } else {
                            lappend angles_miss [list $typedef $k $theta $kub $s $penaltyScore $prmComment]
                        }
                    }
                    DIH {
                        set typedef [lrange $prmData 0 3]
                        set k       [lindex $prmData 4]
                        set n       [lindex $prmData 5]
                        set delta   [lindex $prmData 6]
                        if { $penaltySearch == -1 } {
                            lappend dihedrals_exist [list $typedef [list $k $n $delta] $prmComment]
                        } else {
                            lappend dihedrals_miss [list $typedef $k $n $delta $penaltyScore $prmComment]
                        }
                    }
                    IMPROP {
                        set typedef [lrange $prmData 0 3]
                        set k       [lindex $prmData 4]
                        set psi     [lindex $prmData 6]
                        if { $penaltySearch == -1 } {
                            lappend impropers_exist [list $typedef [list $k $psi] $prmComment]
                        } else {
                            lappend impropers_miss [list $typedef $k $psi $penaltyScore $prmComment]
                        }
                    }
                    CMAP { continue }
                    VDW  { continue }
                }; # end of inner switch
                unset prmData; unset prmComment
            }
        }; # end of outer switch
    }; # end while (reading file)
    close $infile

    # store the data in the namespaced variable for later accession
    set cgenffExistingPars [list $bonds_exist $angles_exist $dihedrals_exist $impropers_exist]
    set cgenffAnalogyPars  [list $bonds_miss  $angles_miss  $dihedrals_miss  $impropers_miss]

    # return
    return $molid
}
#======================================================
