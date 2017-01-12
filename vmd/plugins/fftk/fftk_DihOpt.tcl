#
# $Id: fftk_DihOpt.tcl,v 1.17 2016/05/31 21:21:24 mayne Exp $
#

#======================================================
namespace eval ::ForceFieldToolKit::DihOpt {

    variable GlogFiles
    variable psf
    variable pdb
    variable parlist
    variable parDataInput
    variable boundInfo
    variable namdbin
    variable QMdata
    variable MMdata
    variable outFile
    variable outFileName

    variable EnQM
    variable EnMM
    variable dihAllData
    variable dihAll
    variable weights

    variable dihToFit
    variable currParData

    variable outFreq
    variable globalCount
    variable kmax
    variable cutoff

    variable tol
    variable mode
    variable saT
    variable saIter
    variable saTSteps
    variable saTExp

    variable debug
    variable debugLog
    variable guiMode
    variable keepMMTraj

    variable refineParDataInput
    variable refineCutoff
    variable refineMode
    variable refineKmax
    variable refineTol
    variable refinesaT
    variable refinesaTSteps
    variable refinesaIter
    variable refinesaTExp

    variable fixScanned

}
#======================================================
proc ::ForceFieldToolKit::DihOpt::init {} {
    # localize variables
    variable GlogFiles
    variable psf
    variable pdb
    variable parlist
    variable parDataInput
    variable namdbin
    variable QMdata
    variable MMdata
    #variable outFile
    variable outFileName

    variable EnQM
    variable EnMM
    variable dihAllData
    variable dihAll
    variable weights

    variable dihToFit
    variable currParData

    variable outFreq
    variable globalCount
    variable kmax
    variable cutoff

    variable tol
    variable mode
    variable saT
    variable saIter
    variable saTSteps
    variable saTExp

    variable debug
    variable guiMode
    variable keepMMTraj

    variable refineParDataInput
    variable refineCutoff
    variable refineMode
    variable refineKmax
    variable refineTol
    variable refinesaT
    variable refinesaTSteps
    variable refinesaIter
    variable refinesaTExp

    variable fixScanned

    # Initialize the variables
    set GlogFiles {}
    set psf {}
    set pdb {}
    set parlist {}
    set parDataInput {}
    set namdbin "namd2"
    set outFileName "DihOpt.log"
    set QMdata {}
    set MMdata {}
    set EnQM {}
    set EnMM {}
    set dihAllData {}
    array set dihAll {}
    set weights {}
    set parData {}
    set dihToFit {}
    set currParData {}
    set outFreq 100
    set globalCount 0
    set kmax 3.0
    set cutoff 10.0
    set tol 0.01
    set mode "simulated annealing"
    set saT 1000
    set saIter 10
    set saTSteps 100
    set saTExp 3
    set debug 0
    set guiMode 1
    set keepMMTraj 0

    set refineParDataInput {}
    set refineKmax 3.0
    set refineCutoff 10.0
    set refineTol 0.01
    set refineMode "simulated annealing"
    set refinesaT 1000
    set refinesaIter 10
    set refinesaTSteps 100
    set refinesaTExp 3

    set fixScanned 1

}
#======================================================
proc ::ForceFieldToolKit::DihOpt::sanityCheck { procType } {
    # runs a check on input data

    # returns 1 if all input is sane
    # returns 0 if there are errors

    # localize relevant variables
    variable psf
    variable pdb
    variable parlist
    variable namdbin
    variable outFileName
    variable GlogFiles
    variable parDataInput
    variable kmax
    variable cutoff
    variable mode
    variable tol
    variable saT
    variable saIter
    variable saTSteps
    variable saTExp
    variable outFreq

    variable EnQM
    variable EnMM
    variable dihAllData

    variable refineParDataInput
    variable refineCutoff
    variable refineMode
    variable refineKmax
    variable refineTol
    variable refinesaT
    variable refinesaTSteps
    variable refinesaIter
    variable refinesaTExp

    variable debug

    # local variables
    set errorList {}
    set errorText ""

    # checks will be different for optimization vs refinement/refitting
    switch -exact $procType {
        opt {
            # -------
            #  INPUT
            # -------
            # check psf
            if { $psf eq "" } { lappend errorList "No PSF file specified." } \
            else { if { ![file exists $psf] } { lappend errorList "Cannot find PSF file." } }

            # check pdb
            if { $pdb eq "" } { lappend errorList "No PDB file specified." } \
            else { if { ![file exists $pdb] } { lappend errorList "Cannot find PDB file." } }

            # make sure there is a parameter file
            # and that they exist
            if { [llength $parlist] == 0 } { lappend errorList "No parameter files were specified." } \
            else {
                foreach parFile $parlist {
                    if { ![file exists $parFile] } { lappend errorList "Cannot find prm file: $parFile." }
                }
            }

            # make sure namd2 command and/or file exists
            if { $namdbin eq "" } {
                lappend errorList "NAMD binary file (or command if in PATH) was not specified."
            } else { if { [::ExecTool::find $namdbin] eq "" } { lappend errorList "Cannot find NAMD binary file." } }

            # make sure that outFileName is set and user can write to output dir
            if { $outFileName eq "" } { lappend errorList "Output LOG file was not specified." } \
            else { if { ![file writable [file dirname $outFileName]] } { lappend errorList "Cannot write to output LOG directory." } }

            # ----------------
            #  QM TARGET DATA
            # ----------------
            # check log files
            if { [llength $GlogFiles] == 0 } { lappend errorList "No Gaussian log files specified." } \
            else {
                foreach log $GlogFiles {
                    if { ![file exists $log] } { lappend errorList "Cannot find log file: $log." }
                }
            }

            # ----------------
            #  INPUT PAR DATA
            # ----------------
            # check parDataInput { {typedef} {k mult delta lock?} ...}
            if { [llength $parDataInput] == 0 } { lappend errorList "No dihedral parameter settings entered." } \
            else {
                # cycle through each dihedral
                foreach dih $parDataInput {
                    puts $dih
                    # check type def
                    if { [llength [lindex $dih 0]] != 4 || \
                         [lindex $dih 0 0] eq "" || \
                         [lindex $dih 0 1] eq "" || \
                         [lindex $dih 0 2] eq "" || \
                         [lindex $dih 0 3] eq "" } { lappend errorList "Found inappropriate dihedral type definition." }
                    # check force constant
                    if { [lindex $dih 1 0] eq "" || \
                         ![string is double [lindex $dih 1 0]] || \
                         [lindex $dih 1 0] < 0 } { lappend errorList "Found inappropriate dihedral k." }
                    # check periodicity/mult
                    if { [lindex $dih 1 1] eq "" || \
                         [lindex $dih 1 1] < 1 || \
                         [lindex $dih 1 1] > 6 || \
                         ![string is integer [lindex $dih 1 1]] } { lappend errorList "Found inappropriate dihedral n (periodicity)." }
                    # check phase shift
                    if { [lindex $dih 1 2] eq "" || \
                         ![string is double [lindex $dih 1 2]] || \
                         ([lindex $dih 1 2] != 0 && [lindex $dih 1 2] != 180) } { lappend errorList "Found inappropriate dihedral \u03B4." }
                    # check phase lock
                    if { [lindex $dih 1 3] != 0 && [lindex $dih 1 3] != 1 } { lappend errorList "Found inappropriate phase lock." }
                }
            }

            # -------------------
            #  ADVANCED SETTINGS
            # -------------------
            # dihedral settings
            if { $kmax eq "" || $kmax < 0 || ![string is double $kmax] } { lappend errorList "Found inappropriate kmax." }
            if { $cutoff eq "" || $cutoff < 0 || ![string is double $cutoff] } { lappend errorList "Found inappropriate energy cutoff." }

            # optimizer settings
            if { [lsearch -exact {downhill {simulated annealing}} $mode] == -1 } { lappend errorList "Unsupported optimization mode." } \
            else {
                # check tol
                if { $tol eq "" || $tol < 0 || ![string is double $tol] } { lappend errorList "Found inappropriate optimization tolerance setting." }
                # check simulated annealing parameters
                if { $mode eq "simulated annealing" } {
                    if { $saT eq "" || ![string is double $saT] } { lappend errorList "Found inappropriate saT setting." }
                    if { $saTSteps eq "" || $saTSteps < 0 || ![string is integer $saTSteps] } { lappend errorList "Found inappropriate saTSteps setting." }
                    if { $saIter eq "" || $saTSteps < 0 || ![string is integer $saIter] } { lappend errorList "Found inappropriate saIter setting." }
                    if { $saTExp eq "" || $saTExp < 0 || ![string is integer $saTExp] } { lappend errorList "Found inappropriate saTExp setting." }
                }
            }

            # output freq
            if { $outFreq eq "" || $outFreq < 0 || ![string is integer $outFreq] } { lappend errorList "Found inappropriate output frequency." }

            # LOG FILE
            # make sure that the user can write to CWD
            if { ![file writable .] } { lappend errorList "Cannot write log file to CWD." }
        }

        refine {
            # -------------------------------------
            #  VARIABLES THAT SHOULD BE PRE-LOADED
            # -------------------------------------
            # check psf
            if { $psf eq "" } { lappend errorList "No PSF file specified." } \
            else { if { ![file exists $psf] } { lappend errorList "Cannot find PSF file." } }

            # check pdb
            if { $pdb eq "" } { lappend errorList "No PDB file specified." } \
            else { if { ![file exists $pdb] } { lappend errorList "Cannot find PDB file." } }

            # check EnQM and EnMM
            if { [llength $EnQM] == 0 } { lappend errorList "QME has not been loaded." }
            if { [llength $EnMM] == 0 } { lappend errorList "MMEi has not been loaded." }
            if { [llength $EnQM] != [llength $EnMM] } { lappend errorList "Mismatched QME and MMEi lists." }

            # check dihAllData
            if { [llength $dihAllData] == 0 } { lappend errorList "Dihedral measurements have not been loaded." }

            # ---------------------------
            #  REFINEMENT INPUT PAR DATA
            # ---------------------------
            # check parDataInput { {typedef} {k mult delta lock} ...}
            if { [llength $refineParDataInput] == 0 } { lappend errorList "No dihedral parameter settings entered." } \
            else {
                # cycle through each dihedral
                foreach dih $refineParDataInput {
                    # check type def
                    if { [llength [lindex $dih 0]] != 4 || \
                         [lindex $dih 0 0] eq "" || \
                         [lindex $dih 0 1] eq "" || \
                         [lindex $dih 0 2] eq "" || \
                         [lindex $dih 0 3] eq "" } { lappend errorList "Found inappropriate dihedral type definition." }
                    # check force constant
                    if { [lindex $dih 1 0] eq "" || \
                         ![string is double [lindex $dih 1 0]] } { lappend errorList "Found inappropriate dihedral k." }
                    # check periodicity/mult
                    if { [lindex $dih 1 1] eq "" || \
                         [lindex $dih 1 1] < 1 || \
                         [lindex $dih 1 1] > 6 || \
                         ![string is integer [lindex $dih 1 1]] } { lappend errorList "Found inappropriate dihedral n." }
                    # check phase shift
                    if { [lindex $dih 1 2] eq "" || \
                         ![string is double [lindex $dih 1 2]] } { lappend errorList "Found inappropriate dihedral \u03B4." }
                    # check lock
                }
            }

            # -------------------
            #  ADVANCED SETTINGS
            # -------------------
            # dihedral settings
            if { $refineKmax eq "" || $refineKmax < 0 || ![string is double $refineKmax] } { lappend errorList "Found inappropriate kmax." }
            if { $refineCutoff eq "" || $refineCutoff < 0 || ![string is double $refineCutoff] } { lappend errorList "Found inappropriate energy cutoff." }

            # optimizer settings
            if { [lsearch -exact {downhill {simulated annealing}} $refineMode] == -1 } { lappend errorList "Unsupported optimization mode." } \
            else {
                # check tol
                if { $refineTol eq "" || $refineTol < 0 || ![string is double $refineTol] } { lappend errorList "Found inappropriate optimization tolerance setting." }
                # check simulated annealing parameters
                if { $refineMode eq "simulated annealing" } {
                    if { $refinesaT eq "" || ![string is double $refinesaT] } { lappend errorList "Found inappropriate saT setting." }
                    if { $refinesaTSteps eq "" || $refinesaTSteps < 0 || ![string is integer $refinesaTSteps] } { lappend errorList "Found inappropriate saTSteps setting." }
                    if { $refinesaIter eq "" || $refinesaTSteps < 0 || ![string is integer $refinesaIter] } { lappend errorList "Found inappropriate saIter setting." }
                    if { $refinesaTExp eq "" || $refinesaTExp < 0 || ![string is integer $refinesaTExp] } { lappend errorList "Found inappropriate saTExp setting." }
                }
            }

            # output freq
            if { $outFreq eq "" || $outFreq < 0 || ![string is integer $outFreq] } { lappend errorList "Found inappropriate output frequency." }

            # since we're not writing a standard log file, but may be writing a debug log
            # we will need to check that we have write permission to CWD
            if { $debug } {
                if { ![file writable .] } { lappend errorList "Cannot write log file to CWD (debug log file)." }
            }
        }
    }; # end switch



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
proc ::ForceFieldToolKit::DihOpt::optimize {} {
    # launches and controls the dihedral optimization

    # variables
    variable outFile
    variable outFileName
    variable debug
    variable debugLog
    variable guiMode

    variable GlogFiles
    variable psf
    variable pdb
    variable parDataInput
##    variable boundsInfo

    variable namdbin
    variable parlist

    variable QMdata
    variable MMdata
    variable EnQM
    variable EnMM
    variable dihAllData
    variable dihAll
    variable weights
    variable cutoff
    variable kmax

    variable dihToFit
    variable currParData

    variable globalCount
    variable outFreq

    variable mode
    variable tol
    variable saT
    variable saIter
    variable saTSteps
    variable saTExp


    variable fixScanned



    # -----
    # SETUP
    # -----
    if { $guiMode } {
        # run a sanity check
        if { ![::ForceFieldToolKit::DihOpt::sanityCheck opt] } { return -1 }
    }

    set outFile [open $outFileName w]
    if { $debug } {
        set debugLog [open "[file rootname $outFileName].debug.log" w]
        ::ForceFieldToolKit::DihOpt::printSettings $debugLog
    }

    # -------
    # QM DATA
    # -------
    # parse Gaussian log files (QMdata)
    set QMdata [::ForceFieldToolKit::DihOpt::parseGlog $GlogFiles]
    # in the form: {  {dih indicies (0-based)} currDihVal QME {xyz coords} ...  }
    puts $outFile "QM data read from:"
    foreach log $GlogFiles {
        puts $outFile "\t$log"
    }
    flush $outFile

    # Write the QMdata to the log file
    # this is important, as it may be read in for refinement routine
    puts $outFile "\nQMDATA"
    foreach ele $QMdata {
        puts $outFile $ele
    }
    puts $outFile "END\n"
    flush $outFile

    # load QM conformations into VMD
    ::ForceFieldToolKit::DihOpt::vmdLoadQMData $psf $pdb $QMdata


    # Identify dih indices in molecule for dihedral type definitions
    # that will be optimized (and require constraints during MM relaxation)
    set dihToFit [::ForceFieldToolKit::DihOpt::getMolecDihPar $psf $pdb $parDataInput]
    # in the form: {  {index def} {type def} { {k1 mult1 delta1 lock?} {k2 mult2 delta2 lock?} ...}  }

    puts $outFile "\nDihedrals to be fit:"
    foreach dih $dihToFit {
        puts $outFile "$dih"
    }
    flush $outFile

    # -------
    # MM DATA
    # -------
    # compute the relaxed MM energy (with optimizing dih terms zeroed out)
    # note: dihedrals to be fit are constrained
    puts $outFile "\nPSF\n$psf\nEND"
    puts $outFile "\nPDB\n$pdb\nEND"
    flush $outFile

    set MMdata [calcMM $psf $pdb $dihToFit $namdbin $parlist]
    # in the form:
    # { {MME1 MME2 ... MMEN } {dihAll array} }

    # Write the MMdata to the log file
    # this is important, as it may be read in for refinement routine
    puts $outFile "\nMMDATA"
    puts $outFile "MME"
    foreach ele [lindex $MMdata 0] {
        puts $outFile $ele
    }
    puts $outFile "END"
    puts $outFile "\nMMdihARRAY"
    foreach ele [lindex $MMdata 1] {
        puts $outFile $ele
    }
    puts $outFile "END\n"
    flush $outFile

    # delete the loaded QM molecule set to reclaim memory
    mol delete top


    # ------------
    # OPTIMIZATION
    # ------------

    # some setup
    # parse out QM energy (EnQM), MM energy (EnMM), and measured dihedral angles (dihAll)
    set EnQM {}; set EnMM {}; #set dihAll {}
    for {set i 0} {$i < [llength $QMdata]} {incr i} {
        lappend EnQM [lindex $QMdata $i 2]
        #lappend EnMM [lindex $MMdata $i 0]
        lappend EnMM [lindex $MMdata 0 $i]
        #lappend dihAll [lindex $MMdata $i 1]
    }
    set dihAllData [lindex $MMdata 1]

    # normalize the energies to the global minimum in each set
    set EnQM [::ForceFieldToolKit::DihOpt::renorm $EnQM]
    set EnMM [::ForceFieldToolKit::DihOpt::renorm $EnMM]

    # set the weighting factors to ignore high energy (QME) conformations
    set weights {}
    for {set i 0} {$i < [llength $EnQM]} {incr i} {
        if {[lindex $EnQM $i] < $cutoff} {
            lappend weights 1
        } else {
            lappend weights 0
        }
    }

    if { $debug } {
        puts $debugLog "\n==========================================="
        puts $debugLog " Optimization Setup"
        puts $debugLog "==========================================="
        puts $debugLog "Setup Weights:"
        puts $debugLog "nQME\tnMME\tweight"
        for {set i 0} {$i < [llength $EnQM]} {incr i} {
            puts $debugLog "[lindex $EnQM $i]\t[lindex $EnMM $i]\t[lindex $weights $i]"
        }
        flush $debugLog
    }


    # setup bounds
    set bounds {}
    set init {}
    # convert the parameter data to a linear list of force constants (klist)
    set output [::ForceFieldToolKit::DihOpt::pardata2klist $dihToFit]
    set klist [lindex $output 0]
    set boundsInfo [lindex $output 1]

    # boundsInfo is similar to klist, form: { {delta1 lock1?} {delta2 lock2?} ...}
#    foreach ele1 $boundsInfo ele2 $klist {
#        if { [lindex $ele1 1] == 1 } {
#            # phase lock is turned on
#
#            if { [lindex $ele1 0] == 180 } {
#               lappend bounds [list [expr -1*$kmax] 0.0]
#               # initialize
#               if { $ele2 >= 0.0 || $ele2 < [expr -1*$kmax] } {
#                   lappend init [expr -1*$kmax/3.0]
#               } else {
#                   lappend init $ele2
#               }
#            } else {
#               lappend bounds [list 0.0 $kmax]
#               # initialize
#               if { $ele2 <= 0.0 || $ele2 > $kmax } {
#                   lappend init [expr $kmax/3.0]
#               } else {
#                   lappend init $ele2
#               }
#            }
#        } else {
#            # phase lock is turned off
#
#            lappend bounds [list [expr -1*$kmax] $kmax]
#            # initialize
#            if { $ele2 > $kmax || $ele2 < [expr -1*$kmax] || $ele2 == 0} {
#                lappend init [expr $kmax/3.0]
#            } else {
#                lappend init $ele2
#            }
#        }
#    }

    foreach ele1 $boundsInfo ele2 $klist {

        # setup the bounds
        if { [lindex $ele1 1] == 1 } {
            # lock phase is on, range is restricted based on delta value
            if { [lindex $ele1 0] == 0 } {
                lappend bounds [list 0.0 $kmax]
            } else {
                lappend bounds [list [expr -1*$kmax] 0.0]
            }
        } else {
            # lock phase is off, use the full range
            lappend bounds [list [expr -1*$kmax] $kmax]
        }

        # determine the initial value
        if { $ele2 > $kmax || $ele2 == 0 } {
            # k value exceeds kmax or is not set
            if { [lindex $ele1 0] == 180 } {
                lappend init [expr -1*$kmax/3.0]
            } else {
                lappend init [expr $kmax/3.0]
            }
        } else {
            # k value is in acceptable range
            if { [lindex $ele1 0] == 180 } {
                # delta=180 so we flip the sign
                lappend init [expr -1*$ele2]
            } else {
                # delta=0 so we keep leave k as positive
                lappend init $ele2
            }
        }

    }


#    # this is the same sort that is used to build the klist
#    foreach ele [lsort -unique -index 1 $dihToFit] {
#        # parse out the type definition
#        set typeDef [lindex $ele 1]
#        # for each multiplicity add a bound based on the boundInfo data
#        foreach ele [lindex $ele 2] {
#            if { [lindex $boundsInfo($typeDef) 0] eq "yes" } {
#                # phase lock set to yes
#                if { [lindex $boundsInfo($typeDef) 1] == 180 } {
#                    # phase locked at 180, k is restricted to negative
#                    lappend bounds [list [expr -1*$kmax] 0]
#                } else {
#                    # phase locked at 0, k is restricted to positive
#                    lappend bounds [list 0 $kmax]
#                }
#            } else {
#                # phase lock set to no; use generic bounds
#                lappend bounds [list [expr -1*$kmax] $kmax]
#            }
#        }
#    }

    if { $debug } {
        puts $debugLog "Initial State and Bounds:"
        puts $debugLog "\nInit\tLBound\tUBound:"
        for {set i 0} {$i < [llength $init]} {incr i} {
            puts $debugLog "[lindex $init $i]\t[lindex $bounds $i 0]\t[lindex $bounds $i 1]"
        }
        flush $debugLog
    }


    # opt setup
    # reset the optimization step counter
    set globalCount 0
    if { $mode eq "downhill" } {
        set opt [optimization -downhill -tol $tol -function ::ForceFieldToolKit::DihOpt::optDih]
    } elseif { $mode eq "simulated annealing" } {
        set opt [optimization -annealing -tol $tol -T $saT -iter $saIter -Tsteps $saTSteps -Texp $saTExp -function ::ForceFieldToolKit::DihOpt::optDih]
    } else {
        if { $debug } {puts $debugLog "ERROR: Unsupported optimziation mode.  Currently \"downhill\" and \"simulated annealing\" are supported"; flush $debugLog }
        error "ERROR: Unsupported optimziation mode.  Currently \"downhill\" and \"simulated annealing\" are supported"
    }

    $opt configure -bounds $bounds
    $opt initsimplex $init 1.0

    # Run the optimization
    set result [$opt start]

    if { $debug } {
        puts $debugLog "\n==========================="
        puts $debugLog " Optimization Results"
        puts $debugLog " Final Klist: [lindex $result 0]"
        puts $debugLog " Final Objective Value: [lindex $result 1]"
        puts $debugLog " Optimizer Iterations: $globalCount"
        puts $debugLog "==========================="
        flush $debugLog
    }

    # ---------
    # FINISH UP
    # ---------

    # Since not every optimization iteration is written to the log file
    # we need to recalc the energy from the final (optimized) klist

    # update the gui status
    if { $guiMode } {
        set ::ForceFieldToolKit::gui::doptStatus "Writing Final Energies"; update idletasks
    }

    puts $outFile "\n================================================="
    puts $outFile "FINAL RMSE"
    puts $outFile "[lindex $result 1]"
    puts $outFile "END"
    flush $outFile

    # write a header to the log file denoting that we're revisiting the optimized klist
    puts $outFile "\n================================================="
    puts $outFile "FINAL STEP ENERGIES"
    puts $outFile "QME   MME(i)   MME(f)   QME-MME(f)"; flush $outFile
    flush $outFile

    # parse out the final klist
    set finalKlist [lindex $result 0]
    # convert the optimized klist to pardata
    set finalParData [::ForceFieldToolKit::DihOpt::klist2pardata $dihToFit $finalKlist]
    # recalc the isolated dihedral energies
    set finalDihEnList [::ForceFieldToolKit::DihOpt::calcDihEnergy $dihAllData $finalParData]
    # calc final total MM energy, and normalize
    set EnMMf {}
    for {set i 0} {$i < [llength $EnQM]} {incr i} {
        lappend EnMMf [expr {[lindex $EnMM $i] + [lindex $finalDihEnList $i]}]
    }
    set EnMMf [::ForceFieldToolKit::DihOpt::renorm $EnMMf]


    for {set i 0} {$i < [llength $EnQM]} {incr i} {
        set QMt [lindex $EnQM $i]
        set MMt [lindex $EnMM $i]
        set MMft [lindex $EnMMf $i]
        puts $outFile [format "%1.3f  %1.3f  %1.3f  %1.3f" $QMt $MMt $MMft [expr $QMt - $MMft] ]; flush $outFile
    }

    puts $outFile "END"; flush $outFile

    if { $debug } { puts $debugLog "\nQME, MMEi, MMEf written to log file"; flush $debugLog }

    # write out final parameters in clearly formatted output to the log file
    # important for import into BuildPar

    # update the gui status
    if { $guiMode } {
        set ::ForceFieldToolKit::gui::doptStatus "Writing Final Parameters"; update idletasks
    }

    puts $outFile "\n\n================================================="
    puts $outFile "FINAL PARAMETERS"
    # prepare a formatted list of parameters
    set finalParList [::ForceFieldToolKit::DihOpt::prepDihPar $dihToFit $finalKlist]

    foreach ele $finalParList {
        puts $outFile "[list dihedral [lrange $ele 0 3] [lindex $ele 4] [lindex $ele 5] [lindex $ele 6] [lindex $ele 7]]"
    }
    puts $outFile "END\n"
    flush $outFile

    # clean up
    close $outFile

    if { $debug } {
        puts $debugLog "\n\nDONE"
        close $debugLog
    }

    if { $guiMode } {
        return [list [lindex $result 1] $EnMMf $finalParList]
    }

#    if { $guiMode } {
#        # rebuild the finalParList with the phase lock settings
#        set returnParList {}
#        foreach ele $finalParList {
#            set typeDef [lrange $ele 0 3]
#            set phaseLock [lindex $boundsInfo($typeDef) 0]
#            lappend returnParList [concat $ele $phaseLock]
#        }
#        return [list [lindex $result 1] $EnMMf $returnParList]
#    }

}
#======================================================
proc ::ForceFieldToolKit::DihOpt::manualRefinementCalculation {inputPars} {
    # Computes the MM PES and RMSE for the input parameter data
    # Use for manual refinement of dihedral parameters

    # localize relvant variables
    # the following variables should have been set from the initial run
    variable dihAllData
    variable EnQM
    variable EnMM
    variable psf
    variable pdb

    # Take the cutoff from the refine section,
    # which may or may not be what the user wants
    variable refineCutoff

    # cover the input parameters to parData format
    set parData [::ForceFieldToolKit::DihOpt::getMolecDihPar $psf $pdb $inputPars]

    # compute the dihedral energy based on input
    set dihEnList [::ForceFieldToolKit::DihOpt::calcDihEnergy $dihAllData $parData]
    # add the energy to the MMEi
    set EnMMf {}
    for {set i 0} {$i < [llength $dihEnList]} {incr i} {
        lappend EnMMf [expr [lindex $dihEnList $i] + [lindex $EnMM $i]]
    }
    # normalize
    set EnMMf [::ForceFieldToolKit::DihOpt::renorm $EnMMf]


    # Compute RMSE (taken directly from optDih routine)
    # build a weights list (used to ignore high energy QME conformations)
    set weights {}
    for {set i 0} {$i < [llength $EnQM]} {incr i} {
        if {[lindex $EnQM $i] < $refineCutoff} {
            lappend weights 1
        } else {
            lappend weights 0
        }
    }

    # calculate c (constant that normalizes QM values to MM values)
    # derived from dRMSE/dc = 0, it works out to weighted MM(AVG) - QM(AVG)
    set EmAvg 0
    set EqAvg 0
    set sumWeights 0
    for {set i 0} {$i < [llength $weights]} {incr i} {
        set EmAvg [expr $EmAvg + [lindex $weights $i]*( [lindex $EnMM $i] + [lindex $dihEnList $i] )]
        set EqAvg [expr $EqAvg + [lindex $weights $i]*[lindex $EnQM $i]]
        set sumWeights [expr $sumWeights + [lindex $weights $i]]
    }
    set c [expr ($EmAvg - $EqAvg)/$sumWeights]

    # RMSE = sqrt  ( wt sum (QME-MME+c)^2 / wt sum (1) )
    set Obj 0
    for {set i 0} {$i < [llength $weights]} {incr i} {
        set QMt [lindex $EnQM $i]
        set MMt [lindex $EnMM $i]
        set MMft [lindex $EnMMf $i]
        set Obj [expr $Obj + [lindex $weights $i]*pow($QMt - $MMt - [lindex $dihEnList $i] + $c,2)]
    }

    set Obj [expr sqrt($Obj/$sumWeights)]

    # return the RMSE value and MM PES data
    return [list $Obj $EnMMf]
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::refine {} {
    # refits dihedral parameters to QME and MMEi data that
    # was loaded during the initial fitting
    # hijacks DihOpt variables used in fitting

    # localize necessary variables

    # shared/preserved variables
    variable psf
    variable pdb
    variable EnQM
    variable EnMM
    variable dihAllData

    # shared/hijacked variables
    variable weights
    variable dihToFit
    variable outFile
    variable debug
    variable debugLog
    variable globalCount
    variable guiMode
##    variable boundsInfo

    # refinement variables
    variable refineParDataInput
    variable refineCutoff
    variable refineKmax
    variable refineMode
    variable refineTol
    variable refinesaT
    variable refinesaIter
    variable refinesaTSteps
    variable refinesaTExp
#    variable kmax

    # SETUP

    if { $guiMode } {
        # run a sanity check
        if { ![::ForceFieldToolKit::DihOpt::sanityCheck refine] } { return -1 }
    }

    # we won't write logs for refinement
    switch [vmdinfo arch] {
      WIN64 -
      WIN32 {
        set outFile [open NUL w]
      }
      default {
        set outFile [open /dev/null w]
      }
    }
    # we will write a debugging log if indicated
    if { $debug } {
        set debugLog [open refineDih.debug.log w]
        ::ForceFieldToolKit::DihOpt::printSettings $debugLog
    }


    # build the parData template
    set dihToFit [::ForceFieldToolKit::DihOpt::getMolecDihPar $psf $pdb $refineParDataInput]

    # setup the weights
    # set the weighting factors to ignore high energy (QME) conformations
    set weights {}
    for {set i 0} {$i < [llength $EnQM]} {incr i} {
        if {[lindex $EnQM $i] < $refineCutoff} {
            lappend weights 1
        } else {
            lappend weights 0
        }
    }

    if { $debug } {
        puts $debugLog "\n==========================================="
        puts $debugLog " Optimization Setup"
        puts $debugLog "==========================================="
        puts $debugLog "Setup Weights:"
        puts $debugLog "nQME\tnMME\tweight"
        for {set i 0} {$i < [llength $EnQM]} {incr i} {
            puts $debugLog "[lindex $EnQM $i]\t[lindex $EnMM $i]\t[lindex $weights $i]"
        }
        flush $debugLog
    }


    # setup bounds
    set bounds {}
    # setup init
    # note: these are only used locally to setup the opt
    set init {}

    # convert the parameter data to a linear list of force constants (klist)
    set output [::ForceFieldToolKit::DihOpt::pardata2klist $dihToFit]
    set klist [lindex $output 0]
    set boundsInfo [lindex $output 1]

    # boundsInfo is similar to klist, form: { {delta1 lock1?} {delta2 lock2?} ...}
    foreach ele1 $boundsInfo ele2 $klist {

        # setup the bounds
        if { [lindex $ele1 1] == 1 } {
            # lock phase is on, range is restricted based on delta value
            if { [lindex $ele1 0] == 0 } {
                lappend bounds [list 0.0 $refineKmax]
            } else {
                lappend bounds [list [expr -1*$refineKmax] 0.0]
            }
        } else {
            # lock phase is off, use the full range
            lappend bounds [list [expr -1*$refineKmax] $refineKmax]
        }

        # determine the initial value
        if { $ele2 > $refineKmax || $ele2 == 0 } {
            # k value exceeds refineKmax or is not set
            if { [lindex $ele1 0] == 180 } {
                lappend init [expr -1*$refineKmax/3.0]
            } else {
                lappend init [expr $refineKmax/3.0]
            }
        } else {
            # k value is in acceptable range
            if { [lindex $ele1 0] == 180 } {
                # delta=180 so we flip the sign
                lappend init [expr -1*$ele2]
            } else {
                # delta=0 so we keep leave k as positive
                lappend init $ele2
            }
        }

    }

    if { $debug } {
        puts $debugLog "Initial State and Bounds:"
        puts $debugLog "\nInit\tLBound\tUBound:"
        for {set i 0} {$i < [llength $init]} {incr i} {
            puts $debugLog "[lindex $init $i]\t[lindex $bounds $i 0]\t[lindex $bounds $i 1]"
        }
        flush $debugLog
    }


    # opt setup
    # reset the optimization step counter
    set globalCount 0
    if { $refineMode eq "downhill" } {
        set opt [optimization -downhill -tol $refineTol -function ::ForceFieldToolKit::DihOpt::optDih]
    } elseif { $refineMode eq "simulated annealing" } {
        set opt [optimization -annealing -tol $refineTol -T $refinesaT -iter $refinesaIter -Tsteps $refinesaTSteps -Texp $refinesaTExp -function ::ForceFieldToolKit::DihOpt::optDih]
    } else {
        if { $debug } {puts $debugLog "ERROR: Unsupported optimziation mode.  Currently \"downhill\" and \"simulated annealing\" are supported"; flush $debugLog }
        error "ERROR: Unsupported optimziation mode.  Currently \"downhill\" and \"simulated annealing\" are supported"
    }

    $opt configure -bounds $bounds
    $opt initsimplex $init 1.0


    # RUN
    set result [$opt start]

    if { $debug } {
        puts $debugLog "\n==========================="
        puts $debugLog " Optimization Results"
        puts $debugLog " Final Klist: [lindex $result 0]"
        puts $debugLog " Final Objective Value: [lindex $result 1]"
        puts $debugLog " Optimizer Iterations: $globalCount"
        puts $debugLog "==========================="
        flush $debugLog
    }


    # FINISH

    # parse out the final klist
    set finalKlist [lindex $result 0]
    # convert the optimized klist to pardata
    set finalParData [::ForceFieldToolKit::DihOpt::klist2pardata $dihToFit $finalKlist]
    # recalc the isolated dihedral energies
    set finalDihEnList [::ForceFieldToolKit::DihOpt::calcDihEnergy $dihAllData $finalParData]
    # calc final total MM energy, and normalize
    set EnMMf {}
    for {set i 0} {$i < [llength $EnQM]} {incr i} {
        lappend EnMMf [expr {[lindex $EnMM $i] + [lindex $finalDihEnList $i]}]
    }
    set EnMMf [::ForceFieldToolKit::DihOpt::renorm $EnMMf]

    # build the final parameter list
    set finalParList [::ForceFieldToolKit::DihOpt::prepDihPar $dihToFit $finalKlist]

    # clean up
    close $outFile
    if { $debug } {
        puts $debugLog "\n\nDONE"
        close $debugLog
    }

    return [list [lindex $result 1] $EnMMf $finalParList]

}
#======================================================
proc ::ForceFieldToolKit::DihOpt::optDih { kDihs } {
    # the target function of the optimization routine
    # pass in a list of dihedral force constants (klist)
    # returns an objective value

    # localize required variables
    variable EnQM; # normalized QM energies
    variable EnMM; # normalized relaxed MM energies
    variable dihAllData; # all dihedrals measured in the molecule (any typedef that is being fit?)
    variable weights; # 1 for conformations with EnMM < 10 kcal/mol
    variable dihToFit; # dihedral parameter data
    variable outFile; # log file
    variable debug; # switch for debug logging
    variable debugLog; # file handle for debug log
    variable guiMode; # determines if running from gui

    variable globalCount; # counts optimizer steps
    variable outFreq; # determines how often data is written to the log file

    # reset the parData with the new k values
    set currParData [::ForceFieldToolKit::DihOpt::klist2pardata $dihToFit $kDihs]

    if { $globalCount % $outFreq == 0 && $guiMode } {
         set ::ForceFieldToolKit::gui::doptStatus "Running Optimization (iter: $globalCount)"
         update idletasks
    }

    if { $debug && $globalCount % $outFreq == 0 } {
        puts $debugLog "\nCurrent Optimizer Iteration: $globalCount"
        puts $debugLog "Current parameter data:"
        foreach ele $currParData {
            puts $debugLog $ele
        }
        flush $debugLog
    }

    # write the current parameters to the logfile
    if { $globalCount % $outFreq == 0} {
        puts $outFile "step $globalCount  Current params (type kdih mult delta): "; flush $outFile
        set parOutList [::ForceFieldToolKit::DihOpt::prepDihPar $currParData]
        for {set i 0} {$i < [llength $parOutList]} {incr i 2} {
            if {[expr $i+1] < [llength $parOutList]} {
                puts $outFile "[lindex $parOutList $i]   [lindex $parOutList [expr $i+1]]"; flush $outFile
            } else {
                puts $outFile [lindex $parOutList $i]
            }
        }
    }

    # calculate the isolated dihedral energies
    set EnMMf {}
    set dihEnList [::ForceFieldToolKit::DihOpt::calcDihEnergy $dihAllData $currParData]

    # add to the relaxed total MM energy
    for {set i 0} {$i < [llength $dihEnList]} {incr i} {
        lappend EnMMf [expr [lindex $dihEnList $i] + [lindex $EnMM $i]]
    }
    # normalize
    set EnMMf [::ForceFieldToolKit::DihOpt::renorm $EnMMf]

    # calculate c (constant that normalizes QM values to MM values)
    # derived from dRMSE/dc = 0, it works out to weighted MM(AVG) - QM(AVG)
    set EmAvg 0
    set EqAvg 0
    set sumWeights 0
    for {set i 0} {$i < [llength $weights]} {incr i} {
        set EmAvg [expr $EmAvg + [lindex $weights $i]*( [lindex $EnMM $i] + [lindex $dihEnList $i] )]
        set EqAvg [expr $EqAvg + [lindex $weights $i]*[lindex $EnQM $i]]
        set sumWeights [expr $sumWeights + [lindex $weights $i]]
    }
    set c [expr ($EmAvg - $EqAvg)/$sumWeights]

    # write c value to log file
    if { $globalCount % $outFreq == 0} {
        puts $outFile "c: $c"
        puts $outFile "QME   MME(i)   MME(f)   QME-MME(f)"; flush $outFile
    }

    # calculate RMSE
    # RMSE = sqrt  ( wt sum (QME-MME+c)^2 / wt sum (1) )
    set Obj 0
    for {set i 0} {$i < [llength $weights]} {incr i} {
        set QMt [lindex $EnQM $i]
        set MMt [lindex $EnMM $i]
        set MMft [lindex $EnMMf $i]
        set Obj [expr $Obj + [lindex $weights $i]*pow($QMt - $MMt - [lindex $dihEnList $i] + $c,2)]

        # write individual contributions to log file
        if { $globalCount % $outFreq == 0} {
            puts $outFile [format "%1.3f  %1.3f  %1.3f  %1.3f" $QMt $MMt $MMft [expr $QMt - $MMft] ]; flush $outFile
        }
    }

    set Obj [expr sqrt($Obj/$sumWeights)]

    # write obj to log file
    if { $globalCount % $outFreq == 0} {
        puts $outFile "Current RMSE: $Obj\n\n"; flush $outFile
    }

    if { $debug && $globalCount % $outFreq == 0 } {
        puts $debugLog "Current Value for RMSE/Obj: $Obj"
        flush $debugLog
    }

    # advance the optimizer count
    incr globalCount

    # return the objective
    return $Obj
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::parseGlog { inputFiles } {
    # reads through all passed Gaussian log files
    # parses out:
    #   0. indices for scanned dih (converted to 0-based indices)
    #   1. energy at current conformation (converted to kcal/mol)
    #   2. xyz coordinates of current conformation

    # returns a list with the above items for each step of all passed log files
    # {
    #   { {indicies} {QM energy} {coordinates} }
    #   ...
    # }

    # inputFiles = Gaussian log files

    # localize forcefieldtoolkit debugging variables
    variable debug
    variable debugLog
    variable guiMode

    # init proc-wide variables
    set GlogData {}
    set currLogNum 1; set logCount [llength $inputFiles]

    if { $debug } {
        puts $debugLog "\n==========================================="
        puts $debugLog "       Parsing Gaussian Log Data"
        puts $debugLog "         $logCount log file(s) to read..."
        puts $debugLog "==========================================="
        flush $debugLog
    }

    foreach GlogFile $inputFiles {

        # if running from the GUI, update the status
        if { $guiMode } {
            set ::ForceFieldToolKit::gui::doptStatus "Parsing QM Log ${currLogNum} of ${logCount}"
            update idletasks
        }

        if { $debug } {
            puts $debugLog "\nParsing Gaussian LOG file ($currLogNum of ${logCount}): $GlogFile"
            flush $debugLog
        }

        # initialize log-wide variables
        set currDihDef {}; set currDihVal {}; set currCoords {}; set currEnergy {}
        set infile [open $GlogFile r]
        set tempGlogData {}

        # read through Gaussian Log File (Glog)
        while {[eof $infile] != 1} {
            # read a line in
            set inline [gets $infile]
            # parse line
            switch -regexp $inline {
                {Initial Parameters} {
                    # keep reading until finding the dihedral being scanned
                    while { [lindex [set inline [gets $infile]] 4] ne "Scan" } {
                        continue
                    }
                    # parse out the dihedral definition (1-based indices)
                    set scanDihDef [lindex $inline 2]
                    # strip out four indices from D(#1,#2,#3,#4)
                    set scanDihInds {}
                    foreach ind [split [string range $scanDihDef 2 [expr [string length $scanDihDef] - 2]] ","] {
                       lappend scanDihInds [expr $ind - 1]
                    }

                    if { $debug } {
                        puts $debugLog "Scan dihedral FOUND:"
                        puts $debugLog "\tGaussian Indicies: $scanDihDef"
                        puts $debugLog "\t0-based Indicies (VMD): $scanDihInds"
                        puts $debugLog "--------------------------------------"
                        puts $debugLog "Ind (VMD)\tCurrDih\tEnergy (kcal/mol)"
                        puts $debugLog "--------------------------------------"
                        flush $debugLog
                    }
                }

                {Input orientation:} {
                    # clear any existing coordinates
                    set currCoords {}
                    # burn the header
                    for {set i 0} {$i<=3} {incr i} {
                        gets $infile
                    }
                    # parse coordinates
                    while { [string range [string trimleft [set line [gets $infile]] ] 0 0] ne "-" } {
                        lappend currCoords [lrange $line 3 5]
                    }
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
                    # we've reached an optimized conformation
                    # keep reading until finding the scanned dihedral
                    while { [lindex [set inline [gets $infile]] 2] ne $scanDihDef } {
                        continue
                    }
                    # parse out the current dihedral value; round to integer
                    set currDihVal [expr { round([lindex $inline 3]) }]
                    # add the collected information to the master list
                    # lappend GlogData [list $scanDihInds $currDihVal $currEnergy $currCoords]
                    lappend tempGlogData [list $scanDihInds $currDihVal $currEnergy $currCoords]

                    if { $debug } {
                        puts $debugLog "$scanDihInds\t$currDihVal\t$currEnergy"
                        flush $debugLog
                    }
                }
            }; # end of line parse (switch)
        }; # end of cycling through Glog lines (while)

        # if the Gaussian log file runs the scan in negative direction, reverse the order of entries
        # if not explicitely in the negative direction, preserve the order of entries
        if { [lsearch -exact [split $GlogFile \.] "neg"] != -1 } {
            foreach ele [lreverse $tempGlogData] { lappend GlogData $ele }
        } else {
            foreach ele $tempGlogData { lappend GlogData $ele }
        }

        # clean up
        close $infile
        incr currLogNum

    }; # done Glog file (foreach)

    #puts "...DONE"
    return $GlogData
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::getMolecDihPar { psfFile pdbFile parinput } {
    # reads the psf/pdb pair
    # checks the input parameter data for dihedral type definitions to be optimized
    # returns a list with each element of the form:
    # {
    #   {indices} {type def} { {k1 mult1 delta1 lock?} {k2 mult2 delta2 lock?} ...}
    # }

    # note: parinput is of form:
    # {
    #   {type def} {k mult delta lock?}
    # }

    # localize debugging variables
    variable debug
    variable debugLog

    if { $debug } {
        puts $debugLog "\n==========================================="
        puts $debugLog "        Dihedral definitions to be"
        puts $debugLog "        constrained and optimized"
        puts $debugLog "==========================================="
        flush $debugLog
    }

    # initialize our return list
    set returnList {}

    # build an array that matches typedef to parameter set (e.g., {k mult delta lock?})
    foreach entry $parinput {
        lappend dihArray([lindex $entry 0]) [lindex $entry 1]
    }

    # load the molecule of interest
    mol new $psfFile
    mol addfile $pdbFile

    # retype (required until VMD/PSFGen is fixed to support CGenFF type names)
    # reTypeFromPSF/reChargeFromPSF has been depreciated
    #::ForceFieldToolKit::SharedFcns::reTypeFromPSF $psfFile "top"

    # find all dihedral index definitions and cycle through them
    set indDefList [topo getdihedrallist]

    foreach indDef $indDefList {
        # convert the index definition to a type definition
        set typeDef {}
        foreach ind [lrange $indDef 1 4] {
            set temp [atomselect top "index $ind"]
            lappend typeDef [$temp get type]
            $temp delete
        }

        # test forward and reverse type definitions for a match in the typedef array
        # this matches index def and type def to dih parameter data
        # if present, then we will add to the list of dihedrals required
        # in constraints and measurements during the optimization (returned)
        if { [info exists dihArray($typeDef)] } {
            lappend returnList [list [lrange $indDef 1 4] $typeDef $dihArray($typeDef)]
            if { $debug } { puts $debugLog "[list [lrange $indDef 1 4] $typeDef $dihArray($typeDef)]"; flush $debugLog }
            #puts "[list [lrange $indDef 1 4] $typeDef $dihArray($typeDef)]"
        } elseif { [info exists dihArray([lreverse $typeDef])] } {
            lappend returnList [list [lrange $indDef 1 4] [lreverse $typeDef] $dihArray([lreverse $typeDef])]
            if { $debug } { puts $debugLog "[list [lrange $indDef 1 4] [lreverse $typeDef] $dihArray([lreverse $typeDef])]"; flush $debugLog }
            #puts "[list [lrange $indDef 1 4] [lreverse $typeDef] $dihArray([lreverse $typeDef])]"
        }
    }

    # clean up
    mol delete top

    # return
    return $returnList
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::vmdLoadQMData { psfFile pdbFile GlogData } {
    # loads parsed QM data into VMD

    # localize debugging variables
    variable debug
    variable debugLog

    # load the PSF file to which frames will be added
    mol new $psfFile

    # cycle through each optimized structure, add a file
    foreach optStruct $GlogData {

        # load a new frame
        mol addfile $pdbFile

        # make sure that the number of atoms in the pdb file matches the number
        # of atoms in the GlogData
        set atomCount [molinfo top get numatoms]
        if { [llength [lindex $optStruct 3]] != $atomCount } {
            if { $debug } {puts $debugLog "ERROR: number of atoms in template PDB file does not match Gaussian log coordinate set"; flush $debugLog }
            error "ERROR: number of atoms in template PDB file does not match Gaussian log coordinate set"
        }

        # move each atom to GlodData coordinates
        for {set i 0} {$i < $atomCount} {incr i} {
            set currAtomSel [atomselect top "index $i"]
            $currAtomSel set x [lindex $optStruct 3 $i 0]
            $currAtomSel set y [lindex $optStruct 3 $i 1]
            $currAtomSel set z [lindex $optStruct 3 $i 2]

        }

    }; # end of optimized structures to load (foreach)

    # DONE
    if { $debug } { puts $debugLog "\nQM optimized conformations successfully loaded into VMD"; flush $debugLog }
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::calcMM { psfFile pdbFile toFit namdbin parlist } {
    # calculates the relaxed MM energy for each QM conformation
    # all dihedrals that are to be fit are fixed
    # the remainder of the molecule is minimized to prevent unrelated differences in
    # QM minimized and MM minimized conformations from poluting the dihedral terms
    # of interest

    # NOTE: assumes that all QM frames have been loaded into top

    # returns a list of MMenergy and measured dihedral values to be fit

    # localize variables
    variable debug
    variable debugLog
    variable guiMode
    variable keepMMTraj

    # special case
    variable fixScanned
    variable QMdata

    # parse out the dihedral index defs for dihs to be optimized
    set toFitInds {}
    foreach ele $toFit {
        lappend toFitInds [lindex $ele 0]
    }

    # write a temporary parameter file to zero out parameters for each dih (all common mults)
    # this is to protect against dih par definitions that we wish to optimize, but might
    # be present in other parameter files in the parlist
    set toFitTypes {}
    foreach ele $toFit {
        lappend toFitTypes [lindex $ele 1]
    }
    set parZeroFile [open parZeroFile.par w]
    puts $parZeroFile "BONDS\nANGLES\nDIHEDRAL"
    foreach ele [lsort -unique $toFitTypes] {
        puts $parZeroFile "$ele    0.0   1   0.0"
        puts $parZeroFile "$ele    0.0   2   0.0"
        puts $parZeroFile "$ele    0.0   3   0.0"
        puts $parZeroFile "$ele    0.0   4   0.0"
        puts $parZeroFile "$ele    0.0   6   0.0"
    }
    puts $parZeroFile "\nEND"
    close $parZeroFile
    lappend parlist "parZeroFile.par"

    # write a namd configuration file to run the minimization
    set name "min-geom"
    ::ForceFieldToolKit::SharedFcns::writeMinConf $name $psfFile temp.pdb $parlist extrabonds.txt

    # build the namdEnergy cmd
    set namdEn "namdenergy -silent -psf [list $psfFile] -exe [list $namdbin] -all -sel \$sel -cutoff 1000"
    foreach par $parlist {
        set namdEn [concat $namdEn "-par [list $par]"]
    }

    # make a selection for the QMdata conformations
    set all [atomselect top all]

    # add a new psf file, to which minimized coordinates will be
    # added for each namdEnergy calculation
    mol new $psfFile

    # cycle through each of the QMdata conformations
    if { $debug } {
        puts $debugLog "\n==========================================="
        puts $debugLog "       Relaxed MM Data"
        puts $debugLog "==========================================="
        puts $debugLog "MME (kcal/mmol)\tDihedral Angles to Fit (fixed)"
        flush $debugLog
    }
    set MMdata {}
    array set MMdihs {}
    for {set i 0} {$i < [molinfo [$all molid] get numframes]} {incr i} {

        # update the gui status label
        if { $guiMode } {
            set ::ForceFieldToolKit::gui::doptStatus "Calculating relaxed MME ([expr {$i+1}]/[molinfo [$all molid] get numframes])"
            update idletasks
        }

        $all frame $i
        $all writepdb temp.pdb

        # write and extrabonds file to fix the dihedrals that are to be fit
        set ebfile [open "extrabonds.txt" w]
        foreach dihed $toFitInds {
            set dih [measure dihed $dihed molid [$all molid] frame $i]
            puts $ebfile "dihedral $dihed 10000. $dih"
        }

        # add scanned dihedral to extrabonds file if flagged to do so (default)
        # this is only relevant when scanned dihedral is NOT being fitted
        if { $fixScanned } {
            set scanDihed [lindex $QMdata $i 0]
            set scanDih [measure dihed $scanDihed molid [$all molid] frame $i]
            puts $ebfile "dihedral $scanDihed 10000. $scanDih"
        }
        close $ebfile

        # run the namd minimization
        ::ExecTool::exec $namdbin $name.conf
        # load the minimized coordinates (psf was loaded above)
        mol addfile $name.coor

        # calculate the MM energy using namdenergy
        set sel [atomselect top all]
        set energyout [eval $namdEn]
        set MME [lindex $energyout end end]

        # measure dihedral angles of dihedrals being fit
        set toFitDihValues {}
        foreach dihed $toFitInds {
            set dih [measure dihed $dihed molid [$sel molid]]
            lappend toFitDihValues $dih
            lappend MMdihs($dihed) $dih
        }

        # add the data for this conformation to the MMdata list
        #lappend MMdata [list $MME $toFitDihValues ]
        lappend MMdata $MME

        if { $debug } { puts $debugLog "$MME\t$toFitDihValues"; flush $debugLog }

        # clean up before next minimization
        $sel delete
        if { !$keepMMTraj } {
            animate delete beg [expr [molinfo top get numframes] -1]
        }
    }

    if { $debug } {
        puts $debugLog "\n==========================================="
        puts $debugLog "           (new) dihAll array:"
        puts $debugLog "==========================================="
        foreach ele [array names MMdihs] {
            puts $debugLog "$ele:  $MMdihs($ele)"
        }
    }

    # cleanup
    if { $keepMMTraj } {
        variable outFileName
        set outDir [file dirname [file normalize $outFileName]]
        if { [file exists $outDir] && [file isdirectory $outDir] && [file writable $outDir] } {
            animate write dcd MM_scan_conformations.dcd top
        }
    }
    mol delete top
    file delete $name.conf
    file delete $name.log
    foreach out {coor vel xsc} {
        file delete $name.$out
        file delete $name.$out.BAK
        file delete $name.restart.$out
        file delete $name.restart.$out.old
    }
    file delete temp.pdb
    file delete extrabonds.txt
    file delete parZeroFile.par

    # return
    # first element is list of MME values, second is the (new) dihAll array
    return [list $MMdata [array get MMdihs]]
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::renorm { dataset } {
    # shifts data such that minimum value = 0

    # find the minimum value
    set min [lindex $dataset 0]
    foreach ele $dataset {
        if { $ele < $min } {
            set min $ele
        }
    }

    # shift all data
    foreach ele $dataset {
        lappend renormData [expr $ele - $min]
    }

    # return
    return $renormData
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::pardata2klist { parData } {
    # builds a linear list of force constants (k)
    # from nested dihedral parameter data, allowing for multiplicities

    # each element of parData input has the form:
    # {
    #   { {indices} {typedef1} { {k1 mult1 delta1 lock?} {k2 mult2 delta2 lock?} ...} }
    #   { {indices} {typedef2} { {k1 mult1 delta1 lock?}       ...            } }
    #   ...
    # }
    #         0         1             2,0                 2,1        2,X
    #
    # klist will have the form:
    # { k(typedef1,mult1) k(typedef1,mult2) k(typedef2,mult1) ... k(typedefN,multX) }

    set uniqueTypes [lsort -unique -index 1 $parData]
    foreach typeSet $uniqueTypes {
        foreach paramSet [lindex $typeSet 2] {
            lappend kList [lindex $paramSet 0]
            lappend boundsList [list [lindex $paramSet 2] [lindex $paramSet 3]]
        }
    }

    return [list $kList $boundsList]
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::klist2pardata { parData kList } {
    # replaces k values in parData with updated k values

    # klist has the form:
    # { k(typedef1,mult1) k(typedef1,mult2) k(typedef2,mult1) ... k(typedefN,multX) }
    #
    # each element of parData has the form:
    # {
    #   { {indices} {typedef1} { {k1 mult1 delta1 lock?} {k2 mult2 delta2 lock?} ...} }
    #   { {indices} {typedef2} { {k1 mult1 delta1 lock?}       ...            } }
    #   ...
    #         0         1             2,0                 2,1        2,X
    # }

    set uniqueTypes [lsort -unique -index 1 $parData]
    # count keeps track of position in klist to match the linear
    # klist to the nexted parameter sets in parData
    set count 0

    # match the k values to dihedrals with unique typedefs
    foreach ele $uniqueTypes {
        foreach paramSet [lindex $ele 2] {
            set delta 0.0
            set curK [lindex $kList $count]
            if {$curK < 0} {
                set curK [expr abs($curK)]
                set delta 180.0
            }
            lappend paramSetNew [list $curK [lindex $paramSet 1] $delta [lindex $paramSet 3]]
            incr count
        }
        lappend uniqueTypesNew [list [lindex $ele 0] [lindex $ele 1] $paramSetNew]
        lappend typesOnly [lindex $ele 1]
        unset paramSetNew
    }

    # match updated data from unique typedefs back to all typedefs (unique or duplicate)?
    # to reconstruct parData with new values
    foreach ele $parData {
        set ind [lsearch $typesOnly [lindex $ele 1]]
        lappend parDataNew [list [lindex $ele 0] [lindex $ele 1] [lindex $uniqueTypesNew $ind 2]]
    }

    # return
    return $parDataNew
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::prepDihPar { parData {kDihs {}} } {
    # build a list of dih parameter entries
    # convert the klist back to parData format

    if { [llength $kDihs] > 0} {
        set parData [::ForceFieldToolKit::DihOpt::klist2pardata $parData $kDihs]
    }

    # remove duplicate typesdefs and cycle through each dih
    set uniqueTypes [lsort -unique -index 1 $parData]
    for {set j 0} {$j < [llength $uniqueTypes]} {incr j} {
        # parse out the parameter sets
        set curParList [lindex $uniqueTypes $j 2]
        for {set n 0} {$n < [llength $curParList]} {incr n} {
            # set delta based on the sign of k
##            set delta 0.0
##            set curK [lindex $curParList $n 0]
##            if {$curK < 0} {
##                set curK [expr abs($curK)]
##                set delta 180.0
##            }

            set curK [lindex $curParList $n 0]
            set mult [lindex $curParList $n 1]
            set delta [lindex $curParList $n 2]
            set lock [lindex $curParList $n 3]
            # build the dih parameter entry
            lappend parList [format "[lindex $uniqueTypes $j 1] %2.3f %s %3.2f %s" $curK $mult $delta $lock]
        }
    }

    # return
    return $parList
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::calcDihEnergyORIG { dihAll parData } {
    # calculates MM energy for dihedral angles that are being optimized

    # dihAll has the form:
    # { {dih1 dih2 dih3 ... dihN} {dih1 dih2 dih3 ... dihN} ... }
    # each element contains measured dihedral angle for each dih being optimized
    # each element represents a conformation scanned by QM

    # parData has the form:
    # { {indices} {type def} { {params1} {params2} ...} }

    set PI 3.14159265359

    # loop over conformations
    for {set i 0} {$i < [llength $dihAll]} {incr i} {
        set curDihList [lindex $dihAll $i]
        set dihEn 0
        # loop over individual dihedrals in each conformation
        for {set j 0} {$j < [llength $curDihList]} {incr j} {
            set curParList [lindex [lindex $parData $j] 2]
            set curDih [lindex $curDihList $j]
            ## loop over multiplicities
            for {set n 0} {$n < [llength $curParList]} {incr n} {
                set kCur [lindex [lindex $curParList $n] 0]
                set mult [lindex [lindex $curParList $n] 1]
                set delta [lindex [lindex $curParList $n] 2]
                set dihEn [expr $dihEn + $kCur*(1 + cos($PI/180.0*($mult*$curDih - $delta)))]
            }
        }
        lappend dihEnList $dihEn
    }

    # return (summed dihedral energies for each conformation)
    return $dihEnList
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::calcDihEnergy { dihAll parData } {
    # calculates MM energy for dihedral angles that are being optimized

    # dihAll is the array of the form:
    # {
    #   { {index def} {frame1 frame2 ...} }
    # }
    # each key represents a dihedral index definition
    # each value represents dihedral value for the key for a conformation scanned by QM

    # parData has the form:
    # { {indices} {type def} { {params1} {params2} ...} }

    set PI 3.14159265359

    # rebuild the dihedrals array
    array set dihVals {}
    array set dihVals $dihAll

    # loop over conformations
    for {set i 0} {$i < [llength [lindex $dihAll 1]]} {incr i} {
        # initialize the energy for given conformation (frame)
        set dihEn 0
        # loop over individual dihedrals in each conformation
        for {set j 0} {$j < [llength $parData]} {incr j} {
            set currIndDef [lindex $parData $j 0]
            set currParList [lindex $parData $j 2]
            set currDihVal [lindex $dihVals($currIndDef) $i]
            # loop over multiplicities
            for {set n 0} {$n < [llength $currParList]} {incr n} {
                set kCur [lindex $currParList $n 0]
                set mult [lindex $currParList $n 1]
                set delta [lindex $currParList $n 2]
                set dihEn [expr $dihEn + $kCur*(1 + cos($PI/180.0*($mult*$currDihVal - $delta)))]
            }

        }
        # append the energy for given conformation to the list
        lappend dihEnList $dihEn
    }

    # return (summed dihedral energies for each conformation)
    return $dihEnList
}
#======================================================
proc ::ForceFieldToolKit::DihOpt::printSettings { printFile } {
    # prints the Dihedral Optimization Settings to file

    # Input
    puts $printFile "Settings for Dihedral Optimization"
    puts $printFile "PSF: $::ForceFieldToolKit::DihOpt::psf"
    puts $printFile "PDB: $::ForceFieldToolKit::DihOpt::pdb"
    puts $printFile "Parameter Files:"
    foreach pfile $::ForceFieldToolKit::DihOpt::parlist {
        puts $printFile "\t$pfile"
    }
    puts $printFile "namdbin: $::ForceFieldToolKit::DihOpt::namdbin"
    puts $printFile "output LOG: $::ForceFieldToolKit::DihOpt::outFileName"

    # QM Target Data
    puts $printFile "Gaussian Log Files:"
    foreach lfile $::ForceFieldToolKit::DihOpt::GlogFiles {
        puts $printFile "\t$lfile"
    }

    # Dihedral Parameter Settings
    puts $printFile "parDataInput:"
    foreach entry $::ForceFieldToolKit::DihOpt::parDataInput {
        puts $printFile "\t$entry"
    }

##    # Bounds info
##    puts $printFile "BoundsInfo:"
##    foreach ele [array names ::ForceFieldToolKit::DihOpt::boundsInfo] {
##        puts $printFile "\t$ele -- $::ForceFieldToolKit::DihOpt::boundsInfo($ele)"
##    }

    # Advanced Settings
    puts $printFile "Kmax: $::ForceFieldToolKit::DihOpt::kmax"
    puts $printFile "Energy Cutoff: $::ForceFieldToolKit::DihOpt::cutoff"
    puts $printFile "Opt. Tolerance: $::ForceFieldToolKit::DihOpt::tol"
    puts $printFile "Opt. Mode: $::ForceFieldToolKit::DihOpt::mode"
    puts $printFile "SA T: $::ForceFieldToolKit::DihOpt::saT"
    puts $printFile "SA Tsteps: $::ForceFieldToolKit::DihOpt::saTSteps"
    puts $printFile "SA Iterations: $::ForceFieldToolKit::DihOpt::saIter"
    puts $printFile "SA T Exp: $::ForceFieldToolKit::DihOpt::saTExp"
    puts $printFile "debug: $::ForceFieldToolKit::DihOpt::debug"
    puts $printFile "outFreq: $::ForceFieldToolKit::DihOpt::outFreq"

    puts $printFile ""

    flush $printFile

}
#======================================================
proc ::ForceFieldToolKit::DihOpt::buildScript { scriptFileName } {
    # builds a script that can be run from text mode

    # need to localize variables
    # input
    variable psf
    variable pdb
    variable parlist
    variable namdbin
    variable outFileName

    # qm target data
    variable GlogFiles

    # dihedral parameter settings
    variable parDataInput

    # advanced settings
    variable kmax
    variable cutoff
    variable mode
    variable tol
    variable saT
    variable saIter
    variable saTSteps
    variable saTExp
    variable debug
    variable outFreq
    variable fixScanned

    # build the script
    set scriptFile [open $scriptFileName w]

    # load required packages
    puts $scriptFile "\# Load the ffTK package"
    puts $scriptFile "package require forcefieldtoolkit"
    puts $scriptFile "\n\# Set DihOpt Variables"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::psf $psf"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::pdb $pdb"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::parlist [list $parlist]"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::namdbin $namdbin"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::outFileName $outFileName"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::GlogFiles [list $GlogFiles]"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::parDataInput [list $parDataInput]"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::kmax $kmax"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::cutoff $cutoff"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::mode [list $mode]"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::tol $tol"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::saT $saT"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::saIter $saIter"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::saTSteps $saTSteps"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::saTExp $saTExp"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::debug $debug"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::outFreq $outFreq"

    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::guiMode 0"
    puts $scriptFile "set ::ForceFieldToolKit::DihOpt::fixScanned $fixScanned"

    # launch the optimization
    puts $scriptFile "\n\# Run the optimization"
    puts $scriptFile "::ForceFieldToolKit::DihOpt::optimize"
    puts $scriptFile "\n\# Return gracefully"
    puts $scriptFile "return 1"

    # wrap up
    close $scriptFile
    return
}
#======================================================
