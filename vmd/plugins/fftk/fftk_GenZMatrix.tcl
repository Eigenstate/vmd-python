#
# $Id: fftk_GenZMatrix.tcl,v 1.13 2013/05/17 19:53:49 mayne Exp $
#
#======================================================
namespace eval ::ForceFieldToolKit::GenZMatrix:: {

    variable psfPath
    variable pdbPath
    variable outFolderPath
    variable basename
    
    variable donList
    variable accList
    variable atomLabels
    variable vizSpheresDon
    variable vizSpheresAcc
    
    variable qmProc
    variable qmMem
    variable qmRoute
    variable qmCharge
    variable qmMult
}
#======================================================
proc ::ForceFieldToolKit::GenZMatrix::init {} {

    # IO variables
    variable psfPath
    variable pdbPath
    variable outFolderPath
    variable basename

    # Hydrogen Bonding Variables    
    variable donList
    variable accList

    # QM Input File Variables   
    variable qmProc
    variable qmMem
    variable qmRoute
    variable qmCharge
    variable qmMult
    
    # Initialize
    set psfPath {}
    set pdbPath {}
    set outFolderPath {}
    set basename {}
    set donList {}
    set accList {}

    # initialize the gaussian defaults
    ::ForceFieldToolKit::GenZMatrix::resetGaussianDefaults
}
#======================================================
proc ::ForceFieldToolKit::GenZMatrix::sanityCheck {} {
    # checks to see that appropriate information is set prior to running
    
    # returns 1 if all input is sane
    # returns 0 if there is a problem
    
    # localize relevant GenZMatrix variables
    variable psfPath
    variable pdbPath
    variable outFolderPath
    variable basename
    
    variable donList
    variable accList
    
    variable qmProc
    variable qmMem
    variable qmRoute
    variable qmCharge
    variable qmMult
    
    # local variables
    set errorList {}
    set errorText ""
    
    # checks
    # make sure that psfPath is entered and exists
    if { $psfPath eq "" } {
        lappend errorList "No PSF file was specified."
    } else {
        if { ![file exists $psfPath] } { lappend errorList "Cannot find PSF file." }
    }
    
    # make sure that pdbPath is entered and exists
    if { $pdbPath eq "" } {
        lappend errorList "No PDB file was specified."
    } else {
        if { ![file exists $pdbPath] } { lappend errorList "Cannot find PDB file." }
    }
    
    # make sure that outFolderPath is specified and writable
    if { $outFolderPath eq "" } {
        lappend errorList "No output path was specified."
    } else {
        if { ![file writable $outFolderPath] } { lappend errorList "Cannot write to output path." }
    }
    
    # make sure that basename is not empty
    if { $basename eq "" } { lappend errorList "No basename was specified." }
    
    # it's OK if donor and/or acceptor lists are emtpy, nothing will be written
    
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
proc ::ForceFieldToolKit::GenZMatrix::genZmatrix {} {
    # computes the geometry-dependent position of water and
    # writes Gaussian input files for optimizing water interaction

    # initialize some variables
    variable outFolderPath
    variable basename
    variable donList
    variable accList
    variable qmProc
    variable qmMem
    variable qmRoute
    variable qmCharge
    variable qmMult
    
    # run sanity check
    if { ![::ForceFieldToolKit::GenZMatrix::sanityCheck] } { return }
    
    
    # assign Gaussian atom names and gather x,y,z for output gau file
    set Gnames {}
    set atom_info {}
    for {set i 0} {$i < [molinfo top get numatoms]} {incr i} {
        set temp [atomselect top "index $i"]
        lappend atom_info [list [$temp get element][expr $i+1] [$temp get x] [$temp get y] [$temp get z]]
        lappend Gnames [$temp get element][expr $i+1]
        $temp delete
    }

    #========#
    # DONORS #
    #========#
    foreach donorAtom $donList {

		set donorName [[atomselect top "index $donorAtom"] get name]

		# open output file
		set outname [file join $outFolderPath ${basename}-DON-${donorName}.gau]
		set outfile [open $outname w]

		# write the header
		puts $outfile "%chk=${basename}-DON-${donorName}.chk"
		puts $outfile "%nproc=$qmProc"
		puts $outfile "%mem=${qmMem}GB"
		puts $outfile "$qmRoute"
		puts $outfile ""
		puts $outfile "<qmtool> simtype=\"Geometry optimization\" </qmtool>"
		puts $outfile "${basename}-DON-${donorName}"
		puts $outfile ""
		puts $outfile "$qmCharge $qmMult"

		# write the cartesian coords for the molecule
		foreach atom_entry $atom_info {
		    puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
		}

		# build / write the zmatrix
		::ForceFieldToolKit::GenZMatrix::writeZmat $donorAtom donor $Gnames $outfile
		close $outfile
    }; # end DONORS

    #===========#
    # ACCEPTORS #
    #===========#
    foreach acceptorAtom $accList {
		set acceptorName [[atomselect top "index $acceptorAtom"] get name]

		# open output file
		set outname [file join $outFolderPath ${basename}-ACC-${acceptorName}.gau]
		set outfile [open $outname w]

		# write the header
		puts $outfile "%chk=${basename}-ACC-${acceptorName}.chk"
		puts $outfile "%nproc=$qmProc"
		puts $outfile "%mem=${qmMem}GB"
		puts $outfile "$qmRoute"
		puts $outfile ""
		puts $outfile "<qmtool> simtype=\"Geometry optimization\" </qmtool>"
		puts $outfile "${basename}-ACC-${acceptorName}"
		puts $outfile ""
		puts $outfile "$qmCharge $qmMult"

		# write the cartesian coords for the molecule
		foreach atom_entry $atom_info {
		    puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
		}

		# build / write the zmatrix
		::ForceFieldToolKit::GenZMatrix::writeZmat $acceptorAtom acceptor $Gnames $outfile
		close $outfile

		# CARBONYL EXCEPTIONS
		# there is a special exception for X=O cases (e.g. C=O, P=O, S=O)
		# where we need to write two additional files
		::ForceFieldToolKit::GenZMatrix::writeExceptionZMats $acceptorName $acceptorAtom $Gnames $atom_info

    }
}
#======================================================
proc ::ForceFieldToolKit::GenZMatrix::writeZmat { aInd class gnames outfile } {
	# builds and writes the z-matrix to file

	# passed:
	#		aInd = atom index for interacting atom
	#		class = acceptor / donor
	#		gnames = list of Gaussian-style names
	#		outfile = file handle where output is written
	
	# returns: nothing

	# make a selection for atom A (the interaction site), and find all attached atoms
	set aSel [atomselect top "index $aInd"]
	set bondlistA [lindex [$aSel getbonds] 0]

	# z-matrix will be different for n=1 and n>1 (each will have their own special cases as well)
	if { [llength $bondlistA] == 1} {
		#=======#
		# N = 1 #
		#=======#
		# when n=1 there are two cases in which only 1D scanning is possible:
		# a diatomic molecule, or when C--B--A angle is 180 degrees (i.e. linear)

		set diatomic 0; # defaults to false
		set linear 1; # defaults to true

		# check for a diatomic molecule
		if { [molinfo top get numatoms] == 2} { set diatomic 1 }
		# check if C--B--A is linear
		if { !$diatomic } {
			set bSel [atomselect top "index $bondlistA"]
			set bondlistB [lindex [$bSel getbonds] 0]
			foreach ele $bondlistB {
				# if C = A, skip
				if { $ele == $aInd } { continue }
				# check for a non-linear angle (+/- 2 degrees)
				if { [expr {abs([measure angle [list $aInd $bondlistA $ele]])}] <= 178.0 } {
					# found a non-linear C atom; unset the flag and stop looking
					set linear 0; break
				} else {
					# keep looking
					continue
				}
			}
			# clean up atom selections
			$bSel delete
		}

		if { $diatomic || $linear } {
			# if either diatomic or linear, build a zmatrix for a 1D scan
			
			# get the relevant gnames
			set aGname [lindex $gnames $aInd]
			set bGname [lindex $gnames $bondlistA]

			# for non-diatomic linear cases, we will define x in cartesian coords
			if { $linear } {
				set bSel [atomselect top "index $bondlistA"]
				set v1 [vecnorm [vecsub [measure center $aSel] [measure center $bSel]]]
				if { [lindex $v1 0] == 0 } {
					set v2 "1 0 0"
				} elseif { [lindex $v1 1] == 0 } {
					set v2 "0 1 0"
				} elseif { [lindex $v1 2] == 0 } {
					set v2 "0 0 1"
				} else {
					set v2 [list 1 [expr -1.0*[lindex $v1 0]/[lindex $v1 1]] 0]
				}
				set v2 [vecnorm $v2]
				set xPos [vecadd $v2 [measure center $aSel]]
			}

			# positioning of x is the same for donors and acceptors
			if { $diatomic } {
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s" x $aGname 1.0 $bGname 90.00]	
			} else {
				puts $outfile [format "%3s  %.4f  %.4f  %.4f" x [lindex $xPos 0] [lindex $xPos 1] [lindex $xPos 2]]
			}

			# write the rest of the z-matrix
			if { $class eq "donor" } {
				# donor
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s %7s  %6s" Ow $aGname rAH x 90.00 $bGname 180.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s %7s  %6s" H1w Ow 0.9572 $aGname 127.74 x 0.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s %7s  %6s\n" H2w Ow 0.9572 $aGname 127.74 x 180.00]
				puts $outfile "rAH 2.0"
			} else {
				# acceptor
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s %7s  %6s" Ow $aGname rAH x 90.00 $bGname 180.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s %7s  %6s" H2w Ow 0.9572 $aGname 104.52 x   0.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s %7s  %6s\n" H1w Ow 0.9572 H2w 104.52 x 0.0]
				puts $outfile "rAH 2.0"
			}

			# clean up and return
			$aSel delete
			if { $linear } { $bSel delete }
			return
			# done with special n=1 cases (diatomic or linear)

		} else {
			# handle the n=1 case 'normally'

			# find some information about B atom
			set bInd $bondlistA
			set bSel [atomselect top "index $bInd"]
			set bondlistB [lindex [$bSel getbonds] 0]

			# find a valid C atom
			set cInd {}
			foreach ele $bondlistB {
				# make sure that C != A, and is non-linear
				if { $ele == $aInd } {
					continue
				} elseif { [expr {abs([measure angle [list $aInd $bInd $ele]])}] >= 178.0 } {
					continue
				} else {
					set cInd $ele
				}
			}

			# make an atom selection of C atom
			set cSel [atomselect top "index $cInd"]

			# get gnames
			set aGname [lindex $gnames $aInd]
			set bGname [lindex $gnames $bInd]
			set cGname [lindex $gnames $cInd]

			if { $class eq "donor" } {
				# donor
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" x $aGname 1.0 $bGname 90.00 $cGname dih]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" Ow $aGname rAH x 90.00 $bGname 180.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" H1w Ow 0.9572 $aGname 127.74 x 0.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s\n" H2w Ow 0.9572 $aGname 127.74 x 180.00]
				puts $outfile "rAH  2.0"
				puts $outfile "dih  0.0"
			} else {
				# acceptor
				# call helper function to find probe position
				set probePos [::ForceFieldToolKit::GenZMatrix::placeProbe $aSel]
				# note that Gaussian doesn't like 180 degree angles, so we have to be a little clever
				# make some measurements of probe position
				set mAng [::QMtool::bond_angle $probePos [measure center $aSel] [measure center $cSel]]
				set mDih [::QMtool::dihed_angle $probePos [measure center $aSel] [measure center $cSel] [measure center $bSel]]

				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" H1w $aGname rAH $cGname [format %3.2f $mAng] $bGname [format %3.2f $mDih]]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" x H1w 1.0 $aGname 90.00 $cGname 0.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" Ow H1w 0.9527 x 90.00 $aGname 180.00]
				puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s\n" H2w Ow 0.9527 H1w 104.52 x dih]
				puts $outfile "rAH  2.0"
				puts $outfile "dih  0.0"
			}

			# clean up and return
			$aSel delete; $bSel delete; $cSel delete
			return			
		}; # end of 'normal' N=1

	} else {
		#=======#
		# N > 1 #
		#=======#

		# find some information about atom B
		set bInd [lindex $bondlistA 0]
		set bSel [atomselect top "index $bInd"]
		set cInd [lindex $bondlistA 1]
		set cSel [atomselect top "index $cInd"]

		# find gnames
		set aGname [lindex $gnames $aInd]
		set bGname [lindex $gnames $bInd]
		set cGname [lindex $gnames $cInd]

		# test if C is valid choice
		set abcAng [expr {abs([measure angle [list $aInd $bInd $cInd]])}]
		if { $abcAng < 2 || $abcAng > 178 } { set validC 0 } else {	set validC 1 }
		unset abcAng

		# find probe coords
		set probePos [::ForceFieldToolKit::GenZMatrix::placeProbe $aSel]
		set mAng [::QMtool::bond_angle $probePos [measure center $aSel] [measure center $bSel]]

		if { !$validC } {
			# if C is invalid, ABC are linear and we need a second dummy atom in lieu of the original C atom
			set cGname "x2"
			set x2Pos [coordtrans [trans center [measure center $aSel] axis [vecsub [measure center $cSel] [measure center $bSel]] 180.0 deg] $probePos]
			set mDih [::QMtool::dihed_angle $probePos [measure center $aSel] [measure center $bSel] $x2Pos]
			puts $outfile [format "%3s  %.4f  %.4f  %.4f" x2 [lindex $x2Pos 0] [lindex $x2Pos 1] [lindex $x2Pos 2]]
		} else {
			# C is valid, we can use it to define the dihedral
			set mDih [::QMtool::dihed_angle $probePos [measure center $aSel] [measure center $bSel] [measure center $cSel]]	
		}
		
		if { $class eq "donor" } {
			# donor
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" Ow $aGname rAH $bGname [format %3.2f $mAng] $cGname [format %3.2f $mDih]]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" x Ow 1.0 $aGname 90.00 $bGname dih]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" H1w Ow 0.9572 $aGname 127.74 x 0.00]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s\n" H2w Ow 0.9572 $aGname 127.74 x 180.00]
            puts $outfile "rAH  2.0"
            puts $outfile "dih  0.0"

		} else {
			# acceptor
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" H1w $aGname rAH $bGname [format %3.2f $mAng] $cGname [format %3.2f $mDih]]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" x H1w 1.0 $aGname 90.00 $bGname 0.00]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" Ow H1w 0.9572 x 90.00 $aGname 180.00]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s\n" H2w Ow 0.9572 H1w 104.52 x dih]
			puts $outfile "rAH  2.0"
			puts $outfile "dih  0.0"
		}

		# clean up atomselections and return
		$aSel delete; $bSel delete; $cSel delete
		return
	}
}
#======================================================
proc ::ForceFieldToolKit::GenZMatrix::placeProbe { aSel } {
	# helper function for genZmatrix
	# computes the position of the first probe atom
	# (i.e. Oxygen for HB donors or Hydrogen for HB acceptors)

	# takes in atomselection for the interacting atom
	# returns x,y,z of the first probe atom (i.e. O or H)

	# setup some variables
	set mol [$aSel molid]
	set aCoord [measure center $aSel]

	set bonded [lindex [$aSel getbonds] 0]
	set numBonded [llength $bonded]

	set all [atomselect $mol "all"]
	set allCenter [measure center $all]


	set count 0
	set normavg {0 0 0}


	# compute the probe position based on geometry
	# n = 1, i.e. b---a
	if { $numBonded == 1 } {
		set temp1 [atomselect $mol "index [lindex $bonded 0]"]
		set normavg [vecsub $aCoord [measure center $temp1]]
		incr count
		$temp1 delete
	}


	# n = 2, i.e. b1---a---b2
	if { $numBonded == 2 } {
		set temp1 [atomselect $mol "index [lindex $bonded 0]"]
		set bondvec1 [vecnorm [vecsub $aCoord [measure center $temp1]]]
		set temp2 [atomselect $mol "index [lindex $bonded 1]"]
		set bondvec2 [vecnorm [vecsub $aCoord [measure center $temp2]]]

		# check for linearity, project point away from rest of molecule
        if { abs([vecdot $bondvec1 $bondvec2]) > 0.95 } {
            # check that center of molecule doesn't already lie on our line, or worse, our atom
            # if it does, we don't care about where we place the water relative to the rest of
            # molecule
            set flag 0
            if { [veclength [vecsub $allCenter $aCoord]] < 0.01 } {
                set flag 1
            } elseif { abs([vecdot $bondvec1 [vecnorm [vecsub $allCenter $aCoord]]]) > 0.95 } {
                set flag 1
            }
            if { $flag } {
                if { [lindex $bondvec1 0] == 0 } {
                   set probeCoord] "1 0 0"
                } elseif { [lindex $bondvec1 1] == 0 } {
                   set probeCoord "0 1 0"
                } elseif { [lindex $bondvec1 2] == 0 } {
                   set probeCoord "0 0 1"
                } else {
                   set probeCoord [list 1 [expr -1.0*[lindex $bondvec1 0]/[lindex $bondvec1 1]] 0]
                }
                set probeCoord [vecadd $aCoord [vecscale 2.0 [vecnorm $probeCoord]]]
                return $probeCoord
            }
            set alpha [vecdot $bondvec1 [vecsub $allCenter $aCoord]]
            # same side as mol center
            set probeCoord [vecsub $allCenter [vecscale $alpha $bondvec1]]
            # reflect, extend
            set probeCoord [vecadd $aCoord [vecscale 2.0 [vecnorm [vecsub $aCoord $probeCoord]]]]
            return $probeCoord
        } else {
            set normavg [vecadd $bondvec1 $bondvec2]
        }

		incr count
		$temp1 delete; $temp2 delete; $all delete
	}

	# n > 2, there are many cases; majority are n=3 and n=4
	if { $numBonded > 2 } {
		# cycle through to find all normals
	    for {set i 0} {$i <= [expr $numBonded-3]} {incr i} {
	    
	        set temp1 [atomselect $mol "index [lindex $bonded $i]"]
	        # normalize bond vectors first
	        set normPos1 [vecadd $aCoord [vecnorm [vecsub [measure center $temp1] $aCoord]]]
	    
	        for {set j [expr $i+1]} {$j <= [expr $numBonded-2]} {incr j} {
				set temp2 [atomselect $mol "index [lindex $bonded $j]"]
				set normPos2 [vecadd $aCoord [vecnorm [vecsub [measure center $temp2] $aCoord]]]

				for {set k [expr $j+1]} {$k <= [expr $numBonded-1]} {incr k} {
					set temp3 [atomselect $mol "index [lindex $bonded $k]"]
					set normPos3 [vecadd $aCoord [vecnorm [vecsub [measure center $temp3] $aCoord]]]

					# get the normal vector to the plane formed by the three atoms
					set vec1 [vecnorm [vecsub $normPos1 $normPos2]]
					set vec2 [vecnorm [vecsub $normPos2 $normPos3]]
					set norm [veccross $vec1 $vec2]

					# check that the normal vector and atom of interest are on the same side of the plane
					set d [expr -1.0*[vecdot $norm $normPos1]]
					if { [expr $d + [vecdot $norm $aCoord]] < 0 } {
						set norm [veccross $vec2 $vec1]
					}

					# will average normal vectors at end
					set normavg [vecadd $normavg $norm]
					incr count
					$temp3 delete
				}
				$temp2 delete
	        }
	        $temp1 delete
	    }
	}

	# finish up and return
	set normavg [vecscale [expr 1.0/$count] $normavg]
	set probeCoord [vecadd $aCoord [vecscale 2.0 [vecnorm $normavg]]]
	return $probeCoord
}
#======================================================
proc ::ForceFieldToolKit::GenZMatrix::writeExceptionZMats { aName aInd gnames atom_info } {
	# checks if C=O, P=O, or S=O which require two additional interaction
	# files at alternate water positions (120 degrees off axis)

	# localize some required variables
	variable outFolderPath
	variable basename
	variable qmProc
	variable qmMem
	variable qmRoute
	variable qmCharge
	variable qmMult

	# lookup info from atoms A and B
	set aSel [atomselect top "index $aInd"]
	set bondlistA [lindex [$aSel getbonds] 0]
	set aGname [lindex $gnames $aInd]
	set aElem [string index $aGname 0]

	set bInd [lindex $bondlistA 0]
	set bSel [atomselect top "index $bInd"]
	set bondlistB [lindex [$bSel getbonds] 0]
	set bGname [lindex $gnames $bInd]
	set bElem [string index $bGname 0]

	# find a valid C atom (and associated information)
	set cInd {}
	foreach ele $bondlistB {
		# make sure that C != A, and is non-linear
		if { $ele == $aInd } {
			continue
		} elseif { [expr {abs([measure angle [list $aInd $bInd $ele]])}] >= 178.0 } {
			continue
		} else {
			set cInd $ele
		}
	}
	if { ![llength $cInd] } { return }
	set cSel [atomselect top "index $cInd"]
	set cGname [lindex $gnames $cInd]

	# check if exception case of X=O
	if { $aElem eq "O" && ($bElem eq "C" || $bElem eq "P" || $bElem eq "S" || $bElem eq "N") && [llength $bondlistA] == 1 } {

		# write two slightly different files
		foreach altPos {"a" "b"} dihed {180 0} {

			# open output file
			set outname [file join $outFolderPath ${basename}-ACC-${aName}-120${altPos}.gau]
			set outfile [open $outname w]

			# write the header
			puts $outfile "%chk=${basename}-ACC-${aName}-120${altPos}.chk"
			puts $outfile "%nproc=$qmProc"
			puts $outfile "%mem=${qmMem}GB"
			puts $outfile "$qmRoute"
			puts $outfile ""
			puts $outfile "<qmtool> simtype=\"Geometry optimization\" </qmtool>"
			puts $outfile "${basename}-ACC-${aName}-120${altPos}"
			puts $outfile ""
			puts $outfile "$qmCharge $qmMult"

			# write the cartesian coords for the molecule
			foreach atom_entry $atom_info {
			    puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
			}

			# write custom zmatrix
			set ang 120

			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" H1w $aGname rAH $bGname [format %3.2f $ang] $cGname [format %3.2f $dihed]]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" x H1w 1.0 $aGname 90.00 $bGname 0.00]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s" Ow H1w 0.9527 x 90.00 $aGname 180.00]
			puts $outfile [format "%3s  %7s  %6s  %7s  %6s  %7s  %6s\n" H2w Ow 0.9527 H1w 104.52 x dih]
			puts $outfile "rAH  2.0"
			puts $outfile "dih  0.0"

			# close up
			close $outfile
		}

		# clean up
		$cSel delete
	}

	# clean up and return
	$aSel delete; $bSel delete
	return
}
#======================================================
proc ::ForceFieldToolKit::GenZMatrix::writeSPfiles {} {
    # writes single point energy files required for charge optimization
    # hard coded for HF/6-31G* and MP2/6-31G*

    # localize some variables
    variable psfPath
    variable pdbPath
    variable outFolderPath
    variable basename
    variable qmProc
    variable qmMem
    variable qmCharge
    variable qmMult

    # write compound sp GAU file
    
    # assign Gaussian atom names and gather x,y,z for output com file
    mol new $psfPath
    mol addfile $pdbPath
    set Gnames {}
    set atom_info {}
    for {set i 0} {$i < [molinfo top get numatoms]} {incr i} {
        set temp [atomselect top "index $i"]
        lappend atom_info [list [$temp get element][expr $i+1] [$temp get x] [$temp get y] [$temp get z]]
        lappend Gnames [$temp get element][expr $i+1]
        $temp delete
    } 
    mol delete top

    # Write the HF Single Point File
    # open output file
    set outfile [open [file join $outFolderPath ${basename}-sp-HF.gau] w]

    # write the header
    puts $outfile "%chk=${basename}-sp-HF.chk"
    puts $outfile "%nproc=$qmProc"
    puts $outfile "%mem=${qmMem}GB"
    puts $outfile "# HF/6-31G* SCF=Tight"
    puts $outfile ""
    puts $outfile "<qmtool> simtype=\"Single point calculation\" </qmtool>"
    puts $outfile "${basename}-sp-HF"
    puts $outfile ""
    puts $outfile "$qmCharge $qmMult"

    # write the cartesian coords
    foreach atom_entry $atom_info {
        puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
    }

    puts $outfile ""
    close $outfile

    # Write the MP2 Single Point File
    # open output file
    set outfile [open [file join $outFolderPath ${basename}-sp-MP2.gau] w]

    # write the header
    puts $outfile "%chk=${basename}-sp-MP2.chk"
    puts $outfile "%nproc=$qmProc"
    puts $outfile "%mem=${qmMem}GB"
    puts $outfile "# MP2/6-31G* SCF=Tight Density=Current"
    puts $outfile ""
    puts $outfile "<qmtool> simtype=\"Single point calculation\" </qmtool>"
    puts $outfile "${basename}-sp-MP2"
    puts $outfile ""
    puts $outfile "$qmCharge $qmMult"

    # write the cartesian coords
    foreach atom_entry $atom_info {
        puts $outfile "[lindex $atom_entry 0] [lindex $atom_entry 1] [lindex $atom_entry 2] [lindex $atom_entry 3]"
    }
    puts $outfile ""
    close $outfile


    # Write TIP3P Single Point File
    set outfile [open [file join $outFolderPath wat-sp.gau] w]
    puts $outfile "%chk=wat-sp.chk"
    puts $outfile "%nproc=1"
    puts $outfile "%mem=1GB"
    puts $outfile "# RHF/6-31G* SCF=Tight"
    puts $outfile ""
    puts $outfile "wat-sp"
    puts $outfile ""
    puts $outfile "0 1"
    puts $outfile "O1"
    puts $outfile "H2 1 0.9572"
    puts $outfile "H3 1 0.9572 2 104.52"
    puts $outfile ""
    close $outfile
}
#======================================================
proc ::ForceFieldToolKit::GenZMatrix::resetGaussianDefaults {} {
    # resets gaussian settings to default

    # localize variables
    variable qmProc 
    variable qmMem
    variable qmCharge
    variable qmMult
    variable qmRoute

    # reset
    set qmProc 1
    set qmMem 1
    set qmCharge 0
    set qmMult 1
    set qmRoute "# HF/6-31G* Opt=(Z-matrix,MaxCycles=100) Geom=PrintInputOrient"
}
#======================================================
