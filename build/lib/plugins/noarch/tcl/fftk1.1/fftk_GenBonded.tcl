#
# $Id: fftk_GenBonded.tcl,v 1.17 2014/06/10 21:34:17 mayne Exp $
#

#======================================================
namespace eval ::ForceFieldToolKit::GenBonded:: {

    # declare variables for generating gaussian input file
    variable geomCHK
    variable com
    variable qmProc
    variable qmMem
    variable qmRoute
    
    # declare variables for extracting PARs from Gaussian log
    variable psf
    variable pdb
    variable templateParFile
    variable glog
    variable blog

    # declare variables for differentiating angles from linear bends
    variable lbThresh
    
}
#======================================================
proc ::ForceFieldToolKit::GenBonded::init {} {
    # initializes GUI-related variables/settings for Calc. Bonded. Tab

    # localize + initialize variables for generating Gaussian input file
    variable geomCHK {}
    variable com "hess.gau"
    # run proc to set qmProc, qmMem, and qmRoute to defaults
    ::ForceFieldToolKit::GenBonded::resetGaussianDefaults

    # localize + initialize variables for extracting PARs from Gaussian log        
    variable psf {}
    variable pdb {}
    variable templateParFile {}
    variable glog {}
    variable blog "ExtractBondedPars.log"

    # localize + initialize variables for differentiating A from L
    variable lbThresh 175.0
    
}
#======================================================
proc ::ForceFieldToolKit::GenBonded::sanityCheck { procType } {
    # checks to make sure that input is sane
    
    # returns 1 if everything looks OK
    # returns 0 if there is a problem
    
    # localize GenBonded variables
    variable geomCHK
    variable com
    variable qmProc
    variable qmMem
    variable qmRoute
    
    variable psf
    variable pdb
    variable templateParFile
    variable glog
    variable blog

    variable lbThresh

    # local variables
    set errorList {}
    set errorText ""
    
    # build the error list based on what proc is checked
    switch -exact $procType {
        writeComFile {
            # validate gaussian settings (not particularly vigorous validation)
            # qmProc (processors)
            if { $qmProc eq "" } { lappend errorList "No processors were specified." }
            if { $qmProc <= 0 || $qmProc != [expr int($qmProc)] } { lappend errorList "Number of processors must be a positive integer." }
            # qmMem (memory)
            if { $qmMem eq "" } { lappend errorList "No memory was specified." }
            if { $qmMem <= 0 || $qmMem != [expr int($qmMem)]} { lappend errorList "Memory must be a postive integer." }
            # qmRoute (route card for gaussian; just make sure it isn't empty)
            if { $qmRoute eq "" } { lappend errorList "Route card is empty." }

            # make sure that geometry CHK is specified and exists
            if { $geomCHK eq "" } {
                lappend errorList "Checkpoint file from geometry optimization was not specified."
            } else {
                if { ![file exists $geomCHK] } { lappend errorList "Cannot find geometry optimization checkpoint file." }
            }
            
            # make sure that com file is specified and output dir is writable
            if { $com eq "" } {
                lappend errorList "Output COM file was not specified."
            } else {
                if { ![file writable [file dirname $com]] } { lappend errorList "Cannot write to output directory." }
            }
            # make sure that psf is entered and exists
            if { $psf eq "" } {
                lappend errorList "No PSF file was specified."
            } else {
                if { ![file exists $psf] } { lappend errorList "Cannot find PSF file." }
            }
            
            # make sure that pdb is entered and exists
            if { $pdb eq "" } {
                lappend errorList "No PDB file was specified."
            } else {
                if { ![file exists $pdb] } { lappend errorList "Cannot find PDB file." }
            }

            # check for a reasonable lbThreshold
            if { $lbThresh <= 0.0 || $lbThresh > 180.0 || $lbThresh eq "" } { lappend errorList "Unreasonable threshold of linear bend" }
        }
        
        extractBonded {
            # make sure that psf is entered and exists
            if { $psf eq "" } {
                lappend errorList "No PSF file was specified."
            } else {
                if { ![file exists $psf] } { lappend errorList "Cannot find PSF file." }
            }
            
            # make sure that pdb is entered and exists
            if { $pdb eq "" } {
                lappend errorList "No PDB file was specified."
            } else {
                if { ![file exists $pdb] } { lappend errorList "Cannot find PDB file." }
            }
            
            # make sure that template par file is entered and exists
            if { $templateParFile eq "" } {
                lappend errorList "No template parameter file was specified."
            } else {
                if { ![file exists $templateParFile] } { lappend errorList "Cannot find template parameter file." }
            }

            # make sure that gaussian log file is enetered and exists
            if { $glog eq "" } {
                lappend errorList "No Gaussian log file was specified."
            } else {
                if { ![file exists $glog] } { lappend errorList "Cannot find Gaussian log file." }
            }
            
            # make sure that output file is specified and output dir is writable
            if { $blog eq "" } {
                lappend errorList "Output file was not specified."
            } else {
                if { ![file writable [file dirname $blog]] } { lappend errorList "Cannot write to output directory." }
            }

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
proc ::ForceFieldToolKit::GenBonded::writeComFile {} {
    # writes the gaussian input file for the hessian calculation
    
    # localize necessary variables
    variable geomCHK
    variable com
    variable qmProc
    variable qmMem
    variable qmRoute
    variable psf
    variable pdb
    variable lbThresh

    # sanity check should go here
    if { ![::ForceFieldToolKit::GenBonded::sanityCheck writeComFile] } { return }

    # load the molecule psf/pdb to get the internal coordinates
    set logID [mol new $psf]
    mol addfile $pdb $logID
    ::QMtool::use_vmd_molecule $logID
    set zmat [::QMtool::modredundant_zmat]
  
    # make a copy of the CHK file to prevent Gaussian from overwriting the original
    set newCHKname "[file rootname $com].chk"
    file copy $geomCHK $newCHKname
    
    # write the com file
    set outfile [open $com w]
    puts $outfile "%chk=[file tail $newCHKname]"
    puts $outfile "%nproc=$qmProc"
    puts $outfile "%mem=${qmMem}GB"
    puts $outfile "$qmRoute"
    puts $outfile ""

    # shamelessly stolen from qmtool
    # First delete all existing internal coordinates
    puts $outfile "B * * K"
    puts $outfile "A * * * K"
    puts $outfile "L * * * K"
    puts $outfile "D * * * * K"
    #puts $outfile "O * * * * R"
  
    set num 0
    set lbList {}
    foreach entry $zmat {
        # skip the qmtool zmat header (first line)
        if {$num==0} { incr num; continue }

        set indexes {}
        foreach ind [lindex $entry 2] {
            lappend indexes [expr {$ind+1}]
        }
        set type [string toupper [string index [lindex $entry 1] 0]]

        # check for linear angle
        if { $type eq "A" && [lindex $entry 3] > $lbThresh } {
            # angle qualifies as a "linear bend"
            set type "L"
            lappend lbList $indexes
        }

        # check if standard dihedrals are part of linear bend (undefined)
        set skipflag 0
        if { $type eq "D" && [llength $lbList] > 0 } {
            # test each linear bend in lbList against current dih indices definition
            foreach ang $lbList {
                # test forward and rev angle definitions
                if { [string match "*$ang*" $indexes] || [string match "*[lreverse $ang]*" $indexes] } {
                    # positive test -> leave this dihedral out
                    set skipflag 1
                    incr num
                    break
                }
            }
        }
        if { $skipflag } { continue }
      
        # impropers modeled as dihedrals because Gaussian ignores out-of-plane bends
        if {$type=="I"} { set type "D" }
        if {$type=="O"} { set type "D" }

        # write the entry to the input file
        puts $outfile "$type $indexes A [regsub {[QCRM]} [lindex $entry 5] {}]"
        #puts $outfile "$type $indexes $val [regsub {[QCRM]} [lindex $entry 5] {}]"
        incr num
    }

    puts $outfile ""
    close $outfile
}
#======================================================
proc ::ForceFieldToolKit::GenBonded::resetGaussianDefaults {} {
    # resets the gaussian settings to the default values
    
    set ::ForceFieldToolKit::GenBonded::qmProc 1
    set ::ForceFieldToolKit::GenBonded::qmMem 1
    set ::ForceFieldToolKit::GenBonded::qmRoute "\# MP2/6-31G* Geom=(AllCheck,ModRedundant) Freq NoSymm IOp(7/33=1) SCF=Tight Guess=Read"
    #set ::ForceFieldToolKit::GenBonded::qmRoute "\# MP2/6-31G* Geom=(AllCheck,NewRedundant) Freq NoSymm Pop=(ESP,NPA) IOp(6/33=2,7/33=1) SCF=Tight"
}
#======================================================
proc ::ForceFieldToolKit::GenBonded::extractBonded {} {
    # averages the bond and parameters extracted from the hessian
    # using QMtool (provided as internal coordinates)
    # for entries found in the template parameter file
    
    # localize relevant variables
    variable psf
    variable pdb
    variable templateParFile
    variable glog
    variable blog
    
    # sanity check
    if { ![::ForceFieldToolKit::GenBonded::sanityCheck extractBonded] } { return }
    
    ::ForceFieldToolKit::gui::consoleMessage "Extracting bonded parameters"
    
    # open the log file for output
    set outfile [open $blog w]
    puts $outfile "Bonded Parameters Extracted from the Hessian\n"
    flush $outfile
    
    # setup the template parameters based on the initialized par file
    # read in the template parameter file
    set templatePars [::ForceFieldToolKit::SharedFcns::readParFile $templateParFile]
    
    # build an array for bonds, angles, and dihedrals
    array set templateBonds {}
    foreach bond [lindex $templatePars 0] {
        #                    {type def}                {k  b0}          {comment}
        set templateBonds([lindex $bond 0]) [list [lindex $bond 1] [lindex $bond 2]]
    }
    array set templateAngles {}
    foreach angle [lindex $templatePars 1] {
        #                       {type def}              {k  theta}       {ksub   s}         {comment}
        set templateAngles([lindex $angle 0]) [list [lindex $angle 1] [lindex $angle 2] [lindex $angle 3]]
    }

    # load the typed molecule
    set moleculeID [mol new $psf]
    mol addfile $pdb
    # reTypeFromPSF/reChargeFromPSF has been depreciated
    # ::ForceFieldToolKit::SharedFcns::reTypeFromPSF $psf "top"
    
    # load the Gaussian Log file from the hessian calculation
    set logID [mol new $psf]
    ::QMtool::use_vmd_molecule $logID
    ::QMtool::load_gaussian_log $glog $logID

    # grab the internal coords, which contains the parameters of interest
    set internal_coords [::QMtool::get_internal_coordinates]
    puts $outfile "\nInternal Coordinates:\n"
    puts $outfile "$internal_coords\n"
    flush $outfile

    # build a bonds par section from internal coordinates
    set bonds {}
    foreach entry [lsearch -inline -all -index 1 $internal_coords "bond"] {
        set inds [lindex $entry 2]
        set pars [lindex $entry 4]
        set typeDef {}
        foreach ind $inds {
            set temp [atomselect $moleculeID "index $ind"]
            lappend typeDef [$temp get type]
        }
        
        #              {  {type def} {k b} {comment} }
        lappend bonds [list $typeDef $pars {}]
        $temp delete
    }
    
    # build an angles par section from internal coordinates
    set angles {}
    foreach entry [lsearch -inline -all -index 1 $internal_coords "angle"] {
        set inds [lindex $entry 2]
        set pars [lindex $entry 4]
        set temp [atomselect $moleculeID "index $inds"]
        set typeDef {}
        foreach ind $inds {
            set temp [atomselect $moleculeID "index $ind"]
            lappend typeDef [$temp get type]
        }
        
        #               {  {typeDef} {k theta} {comment} }
        lappend angles [list $typeDef $pars {}]
        $temp delete
    }
    
    # Now average any duplicate parameters and update only
    # those found in the template parameter file
    
    # BONDS
    # initialize some variables
    set b_list {}
    set b_rts {}
    # cycle through each bond definition
    foreach bondEntry $bonds {
        # parse out parameter data
        # { {bond type def} {k b} {comment} }
        set typeDef [lindex $bondEntry 0]
        set k [lindex $bondEntry 1 0]
        set b [lindex $bondEntry 1 1]
        
        # test (forward and reverse)
        set testfwd [lsearch -exact $b_list $typeDef]
        set testrev [lsearch -exact $b_list [lreverse $typeDef]]
        
        if { $testfwd == -1 && $testrev == -1 } {
            # new bond type definition, append all values
            lappend b_list $typeDef
            lappend b_rts [list $k $b 1]
        } else {
            if { $testfwd > -1 } {
                set ind $testfwd
            } else {
                set ind $testrev
            }
            # repeat type definition found, add to running totals
            lset b_rts $ind 0 [expr {[lindex $b_rts $ind 0] + $k}]
            lset b_rts $ind 1 [expr {[lindex $b_rts $ind 1] + $b}]
            lset b_rts $ind 2 [expr {[lindex $b_rts $ind 2] + 1}]
        }
    }

    # update the bonds array
    for {set i 0} {$i < [llength $b_list]} {incr i} {
        # if the paratool bond is present in the template, update the k and b0 values

        # check for foward and reverse bond definitions
        if { [info exists templateBonds([lindex $b_list $i])] } {
            # bond definition matches in fwd orientation
            set currBondDef [lindex $b_list $i]
        } elseif { [info exists templateBonds([lreverse [lindex $b_list $i]])] } {
            # bond definition matches in rev orientation
            set currBondDef [lreverse [lindex $b_list $i]]
        } else {
            # bond definition did not match the template bonds
            continue
        }

        # calc the avg values from running totals data (b_rts)
        set avgK [expr {[lindex $b_rts $i 0]/[lindex $b_rts $i 2]}]
        set avgB0 [expr {[lindex $b_rts $i 1]/[lindex $b_rts $i 2]}]
        # update the value in the templateBonds array
        lset templateBonds($currBondDef) 0 [list $avgK $avgB0]     
    }    
    
    # write the header to denote the start of final parameters
    puts $outfile "FINAL PARAMETERS"; flush $outfile

    # write the updated bond parameters to the log file
    # updated to preserve bond parameter the same order as input file
    foreach ele [lindex $templatePars 0] {
        set type [lindex $ele 0]
        puts $outfile "[list bond $type [lindex $templateBonds($type) 0 0] [lindex $templateBonds($type) 0 1]]"
    }

    # DONE with BONDS


    # ANGLES
    # initialize some variables
    set a_list {}
    set a_rts {}
    set a_rtsub {}
    
    puts ""
    # cycle through each angle definition
    foreach angleEntry $angles {
        # parse out parameter data
        set typeDef [lindex $angleEntry 0]
        set k [lindex $angleEntry 1 0]
        set theta [lindex $angleEntry 1 1]
        set kub [lindex $angleEntry 2 0]
        set s [lindex $angleEntry 2 1]
        #puts "processing (${typeDef}) -- k: $k\ttheta: $theta\tkub: $kub\ts: $s"
        
        # test (forward and reverse)
        set testfwd [lsearch -exact $a_list $typeDef]
        set testrev [lsearch -exact $a_list [lreverse $typeDef]]
        if { $testfwd == -1 && $testrev == -1 } {
            # new angle definition, append all data
            #puts "new definition"
            lappend a_list $typeDef
            lappend a_rts [list $k $theta 1]
            # handle angle UB term, empty or not
            if { $kub ne "" } {
                lappend a_rtsub [list $kub $s 1]
            } else {
                lappend a_rtsub [list {} {} 0]
            }
        } else {
            # duplicate definition, update running totals and count
            #puts "duplicate definition"
            if { $testfwd > -1 } {
                set ind $testfwd
            } else {
                set ind $testrev
            }
            # update angle totals and count
            lset a_rts $ind 0 [expr {[lindex $a_rts $ind 0] + $k}]
            lset a_rts $ind 1 [expr {[lindex $a_rts $ind 1] + $theta}]
            lset a_rts $ind 2 [expr {[lindex $a_rts $ind 2] + 1}]
            # for UB term, update if not empty string (just ignore empty strings)
            if { $kub ne "" } {
                # how the term is updated depends on whether there are any UB terms stored already
                # i.e., if the count is above 0 we need to update, otherwise, just replace
                if { [lindex $a_rtsub $ind 2] > 0 } {
                    lset a_rtsub $ind 0 [expr {[lindex $a_rtsub $ind 0] + $kub}]
                    lset a_rtsub $ind 1 [expr {[lindex $a_rtsub $ind 1] + $s}]
                    lset a_rtsub $ind 2 [expr {[lindex $a_rtsub $ind 2] + 1}]
                } else {
                    lset a_rtsub $ind 0 $kub
                    lset a_rtsub $ind 1 $s
                    lset a_rtsub $ind 2 1
                }
            }; # end of UB term if
        }; # end of angles test
    }; # end of angles loop
    
    # update the angles array
    for {set i 0} {$i < [llength $a_list]} {incr i} {
        # if the paratool angle is present in the template, update the values

        # determine if fwd or rev angle definition is present
        if { [info exists templateAngles([lindex $a_list $i])] } {
            # angle definition matches in fwd orientation
            set currAngDef [lindex $a_list $i]
        } elseif { [info exists templateAngles([lreverse [lindex $a_list $i]])] } {
            # angle definition matches in rev orientation
            set currAngDef [lreverse [lindex $a_list $i]]
        } else {
            # angle definition did not match any template angles
            continue
        }

        # calc the avg values from angle running totals
        set avgK [expr {[lindex $a_rts $i 0]/[lindex $a_rts $i 2]}]
        set avgTheta [expr {[lindex $a_rts $i 1]/[lindex $a_rts $i 2]}]
        # update the angles data in the angles array
        lset templateAngles($currAngDef) 0 [list $avgK $avgTheta]
            
        # if kub and s are defined (count greater than zero), average them
        # otherwise set as undefined
        if { [lindex $a_rtsub $i 2] > 0 } {
            set avgKub [expr {[lindex $a_rtsub $i 0]/[lindex $a_rtsub $i 2]}]
            set avgS [expr {[lindex $a_rtsub $i 1]/[lindex $a_rtsub $i 2]}]
            # update with value
            lset templateAngles($currAngDef) 1 [list $avgKub $avgS]
        } else {
            # update as undefined
            lset templateAngles($currAngDef) 1 [list {} {}]
        }
    }

        
    # write the updated angle parameters to the log file
    # updated to preserve angle parameter order of the input file
    foreach ele [lindex $templatePars 1] {
        set type [lindex $ele 0]
        puts $outfile "[list angle $type [lindex $templateAngles($type) 0 0] [lindex $templateAngles($type) 0 1] {} {} {}]"
        # note: kub, s0, and comments will always be empty
    }

    # DONE with ANGLES

    puts $outfile "END\n"
    
    # clean up
    mol delete $moleculeID
    mol delete $logID
    close $outfile
    
    ::ForceFieldToolKit::gui::consoleMessage "Bonded parameter extraction finished"

}
#======================================================
