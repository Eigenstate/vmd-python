#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds and other topology related properties.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topogromacs.tcl,v 1.11 2014/10/23 01:18:40 johns Exp $

# high level subroutines for supporting gromacs topology files.
#
# write a fake gromacs topology format file that can be used in combination
# with a .gro/.pdb coordinate file for generating .tpr files needed to use
# Some of the more advanced gromacs analysis tools for simulation data that
# was not generated with gromacs.
#
# IMPORTANT NOTE: this script differs from other topotools script in that
# it does not check whether fragments are fully contained in the selection.
# it will output a topology with exactly the same number of atoms as the
# selection has. in case of partially contained fragments, new molecule types
# will be created.
#
# Arguments:
# filename = name of topology file
# mol = molecule
# sel = selection
proc ::TopoTools::writegmxtop {filename mol sel {flags none}} {

    if {[catch {open $filename w} fp]} {
        vmdcon -err "writegmxtop: problem opening gromacs topology file: $fp\n"
        return -1
    }

    # get a list of fragments, i.e. individual molecules
    set fragmap [lsort -integer -unique [$sel get fragment]]
    set typemap [lsort -ascii -unique [$sel get type]]
    set selstr [$sel text]
    # defaults for bond/angle/dihedral/improper functional form
    set btype 1
    set atype 1
    set dtype 1
    set itype 1
    set writepairs 0
    if { $flags == "" } {
        vmdcon -info "Generating a 'faked' gromacs topology file: $filename"
        puts $fp "; 'fake' gromacs topology generated from topotools."
        puts $fp "; WARNING| the purpose of this topology is to allow using the  |WARNING"
        puts $fp "; WARNING| analysis tools from gromacs for non gromacs data.   |WARNING"
        puts $fp "; WARNING| it cannot be used for a simulation.                 |WARNING"
        puts $fp "\n\[ defaults \]\n; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ"
        puts $fp "1 3 yes 0.5 0.5"
        puts $fp "\n\[ atomtypes \]\n; name bond_type mass charge ptype sigma epsilon"
        foreach t $typemap {
            if {[string is integer $t]} {
                puts $fp "type$t C 1.0 0.0 A 0.0 0.0"
            } else {
                puts $fp "$t C 1.0 0.0 A 0.0 0.0"
            }
        }
        puts $fp "\n\[ bondtypes \]\n; i j func b0 kb\n  C C 1 0.13 1000.0 ; totally bogus"
        ; # puts $fp "\n\[ constrainttypes \]\n;"
        puts $fp "\n\[ angletypes \]\n; i j k func th0 cth\n  C C C 1 109.500 100.0 ; totally bogus"
        puts $fp "\n\[ dihedraltypes \]\n; i j k l func coefficients\n  C C C C 1 0.0 3 10.0 ; totally bogus"
    } else {
        vmdcon -info "Generating a real gromacs topology file: $filename"
        writecharmmparams $fp $mol $sel [lindex $flags 0]
        set btype 1
        set atype 5
        set dtype 9
        set itype 2
        set writepairs 1
    }

    set fraglist {}
    set fragcntr {}
    set nlold {}
    set tlold {}
    set count 0
    foreach frag $fragmap {
        set fsel [atomselect $mol "(fragment $frag) and ($selstr)"]
        set nlist [$fsel get name]
        set tlist [$fsel get type]
        if {[listcmp $nlist $nlold] || [listcmp $tlist $tlold]} {
            vmdcon -info "Found new moleculetype: fragment \#$frag natoms=[$fsel num]"
            display update ui
            if {[llength $fraglist] > [llength $fragcntr]} {
                lappend fragcntr $count
            }
            puts $fp ""
            set molname "molecule[llength $fragcntr]"
            lappend fraglist $molname
            set count 1
            set nlold $nlist
            set tlold $tlist
            puts $fp "\n\[ moleculetype \]"
            puts $fp "; Name      nrexcl\n$molname     3"
            puts $fp "\n\[ atoms \]"
            puts $fp "; nr  type  resnr residue atom cgnr charge  mass"
            set atmmap [$fsel get index]
            set resmap [lsort -integer -unique [$fsel get residue]]
            set nr 1
            # for charge group handling
            set cgnr 1
            set cgnum 0
            set cgsum 0.0
            foreach idx [$fsel get index] type [$fsel get type] \
                name [$fsel get name] residue [$fsel get residue] \
                resname [$fsel get resname] charge [$fsel get charge] \
                mass [$fsel get mass] {

                    # assume that charge group atoms are consecutively
                    # in the structure we are working on. gromacs also
                    # imposes a 32 atom limit on charge groups that we
                    # have to honor. to allow for rounding erros we assume
                    # that 0.01 is zero.
                    set cgcut 0.01
                    set cgmax 30
                    set cgsum [expr {$cgsum + $charge}]
                    incr cgnum
                    if { (($cgnum > 1) && (abs($cgsum - floor($cgsum + 0.5*$cgcut)) < $cgcut)) || ($cgnum > $cgmax) } {
                        set cgnum 0
                        incr cgnr
                        set cgsum 0.0
                    }

                    # fix up some data that gromacs cannok grok
                    if {[string is integer $type]} {set type "type$type"}
                    if {[string is integer $resname]} {set resname "RES$resname"}
                    set resid [lsearch -sorted -integer $resmap $residue]
                    incr resid
                    puts $fp [format "% 6d %11s % 6d %8s %6s % 6d %10.4f %10.4f"  \
                                  $nr $type $resid $resname $name $cgnr $charge $mass ]
                    incr nr
                }
            # end of loop over atoms

            if { $writepairs } {
                #Need to find the 1-4 pairs. For some dumb reason, grompp doesn't do this for you.
                set list [get14pairs $fsel]
                if {[llength $list]} {
                    puts $fp "\n\[ pairs \]\n; ai aj func"
                    foreach pair $list {
                        lassign $pair i j
                        set i [lsearch -sorted -integer $atmmap $i]
                        set j [lsearch -sorted -integer $atmmap $j]
                        incr i; incr j
                        puts $fp "$i $j 1"
                    }
                }
            }

            set list [bondinfo getbondlist $fsel none]
            if {[llength $list]} {
                puts $fp "\n\[ bonds \]\n; i  j  func"
                foreach b $list {
                    lassign $b i j
                    set i [lsearch -sorted -integer $atmmap $i]
                    set j [lsearch -sorted -integer $atmmap $j]
                    incr i; incr j
                    puts $fp "$i $j $btype"
                }
            }

            set list [angleinfo getanglelist $fsel]
            if {[llength $list] > 0} {
                puts $fp "\n\[ angles \]\n; i  j  k  func"
                foreach b $list {
                    lassign $b t i j k
                    set i [lsearch -sorted -integer $atmmap $i]
                    set j [lsearch -sorted -integer $atmmap $j]
                    set k [lsearch -sorted -integer $atmmap $k]
                    incr i; incr j; incr k
                    puts $fp "$i $j $k $atype"
                }
            }

            set list [dihedralinfo getdihedrallist $fsel]
            if {[llength $list] > 0} {
                puts $fp "\n\[ dihedrals \]\n; i  j  k  l  func"
                foreach b $list {
                    lassign $b t i j k l
                    set i [lsearch -sorted -integer $atmmap $i]
                    set j [lsearch -sorted -integer $atmmap $j]
                    set k [lsearch -sorted -integer $atmmap $k]
                    set l [lsearch -sorted -integer $atmmap $l]
                    incr i ; incr j; incr k ; incr l
                    puts $fp "$i $j $k $l $dtype"
                }
            }

            set list [improperinfo getimproperlist $fsel]
            if {[llength $list] > 0} {
                puts $fp "\n\[ dihedrals \]\n; i  j  k  l  func"
                foreach b $list {
                    lassign $b t i j k l
                    set i [lsearch -sorted -integer $atmmap $i]
                    set j [lsearch -sorted -integer $atmmap $j]
                    set k [lsearch -sorted -integer $atmmap $k]
                    set l [lsearch -sorted -integer $atmmap $l]
                    incr i ; incr j; incr k ; incr l
                    puts $fp "$i $j $k $l $itype"
                }
            }
            set list [crossterminfo getcrosstermlist $fsel]
            if {[llength $list] > 0} {
                puts $fp "\n\[ cmap \]\n; ai aj ak al am funct"
                foreach b $list {
                    lassign $b i j k l x y z m
                    set i [lsearch -sorted -integer $atmmap $i]
                    set j [lsearch -sorted -integer $atmmap $j]
                    set k [lsearch -sorted -integer $atmmap $k]
                    set l [lsearch -sorted -integer $atmmap $l]
                    set m [lsearch -sorted -integer $atmmap $m]
                    incr i ; incr j; incr k ; incr l ; incr m
                    puts $fp "$i $j $k $l $m 1"
                }
            }
        } else {
            incr count
        }
        $fsel delete
    }
    lappend fragcntr $count

    puts $fp "\n\[ system \]\n; Name\nvmdmolecule$mol\n"
    puts $fp "\n\[ molecules \]\n; Compound    \#mols"
    vmdcon -info "Found [llength $fraglist] moleculetypes."
    foreach name $fraglist num $fragcntr {
        vmdcon -info "$num x $name"
        puts $fp "$name    $num"
    }
    close $fp
    return
}

proc ::TopoTools::writegmxLJprm {fp lj mass types} {
    variable kjinkcal
    puts $fp "\n\[ atomtypes \]"
    puts $fp "; type atnum mass charge ptype sigma epsilon"
    set twoonesixth [expr { pow(2.0, 1.0/6)}]

    foreach dat $lj {

        set type [lindex $dat 0]
        if {[lsearch -exact $types $type] != -1} {
            # Sigma in gromacs is defined as the radius where the potential
            # crosses zero and not where it is minimal (rmin) as in CHARMM.
            # also it is given in nanometers and not angstrom.
            set sigma [expr {[lindex $dat 3] * .2 / $twoonesixth}]
            set epsilon [expr {abs([lindex $dat 2] * $kjinkcal) }]
            set m [dict get $mass $type]
            set idx [ptefrommass $m]
            puts $fp [format "%8s  %3d  %10.4f  0.000  A  %.12f  %.5f" $type $idx $m $sigma $epsilon]
        }
    }
    puts $fp "\n\[ pairtypes \]"
    puts $fp "; i j func sigma epsilon ; THESE ARE 1-4 INTERACTIONS, NOT NBFIX"
    # Sigma in gromacs is defined as the radius where the potential
    # crosses zero and not where it is minimal (rmin) as in CHARMM.
    # also it is given in nanometers and not angstrom.
    foreach dat $lj {
        if {[llength $dat] == 7} {
            set type1 [lindex $dat 0]
            if {[lsearch -exact $types $type1] != -1} {
                set sigma1 [expr {[lindex $dat 6] * .2 / $twoonesixth}]
                set epsilon1 [expr {abs([lindex $dat 5] * $kjinkcal) }]
                foreach dat2 $lj {
                    set type2 [lindex $dat2 0]
                    if {[lsearch -exact $types $type2] != -1} {
                        if {[llength $dat2] == 7} {
                            set sigma2 [expr {[lindex $dat2 6] * .2 / $twoonesixth}]
                            set epsilon2 [expr {abs([lindex $dat2 5] * $kjinkcal) }]
                        } else {
                            set sigma2 [expr {[lindex $dat2 3] * .2 / $twoonesixth}]
                            set epsilon2 [expr {abs([lindex $dat2 2] * $kjinkcal) }]
                        }
                        puts $fp [format "%8s %8s  1 %.12f  %.12f" $type1 $type2 \
                                      [expr {0.5 * ($sigma1 + $sigma2)}] \
                                      [expr {sqrt($epsilon1 * $epsilon2)}]]
                    }
                }
            }
        }
    }
}

proc ::TopoTools::writegmxbondprm {fp bonds types} {
    variable kjinkcal
    puts $fp "\n\[ bondtypes \]"
    puts $fp "; i j func b0 kb"
    foreach bond $bonds {
        lassign $bond type1 type2 k b0
        if {[findInTypes $types [list $type1 $type2]]} {
            puts $fp [format "%8s %8s  1  %.8f  %.2f" $type1 $type2 \
                          [expr {$b0 * 0.1}] \
                          [expr {$k * 2 * $kjinkcal / (0.1 * 0.1)}] ]
        }
    }
}

proc ::TopoTools::writegmxangleprm {fp angles types} {
    variable kjinkcal
    puts $fp "\n\[ angletypes \]"
    puts $fp "; i j k func theta ktheta ub0 kub"
    foreach angle $angles {
        lassign $angle type1 type2 type3 ktheta theta0 kub s0
        if {[findInTypes $types [list $type1 $type2 $type3]]} {
            puts $fp [format "%8s %8s %8s  5  %.6f %.6f %.8f %10.2f" \
                          $type1 $type2 $type3 $theta0 \
                          [expr {2 * $kjinkcal * $ktheta}] \
                          [expr {$s0 * 0.1}] \
                          [expr {$kub * 2 * $kjinkcal / (0.1 * 0.1)}]]
        }
    }
}

proc ::TopoTools::writegmxdihedralprm {fp dihedrals types} {
    variable kjinkcal
    puts $fp "\n\[ dihedraltypes \]"
    set delaywrite [list ]
    puts $fp "; i j k l func phi0 kphi n ; These are the proper dihedrals."
    foreach dihedral $dihedrals {
        lassign $dihedral t1 t2 t3 t4 k n delta
        if {[findInTypes $types [list $t1 $t2 $t3 $t4]]} {
            if { [string equal $t1 X] || [string equal $t4 X] || [string equal $t2 X] || [string equal $t3 X]} {
                lappend delaywrite [format "%8s %8s %8s %8s  9  %8.3f %12.6f %d" \
                          $t1 $t2 $t3 $t4 $delta [expr {$k * $kjinkcal}] $n]
            } else {
                puts $fp [format "%8s %8s %8s %8s  9  %8.3f %12.6f %d" \
                          $t1 $t2 $t3 $t4 $delta [expr {$k * $kjinkcal}] $n]
            }
        }
    }
    #Gromacs dihedral type parser isn't very clever. It looks for the first matching dihedral,
    #therefore wildcard dihedrals must come last.
    foreach element $delaywrite {
        puts $fp $element
    }
}

proc ::TopoTools::writegmximproperprm {fp impropers types} {
    variable kjinkcal
    puts $fp "\n\[ dihedraltypes \]"
    set delaywrite [list ]
    puts $fp "; i j k l func phi0 kphi ; These are the improper dihedrals."
    foreach dihedral $impropers {
        lassign $dihedral t1 t2 t3 t4 k n delta
        if {[findInTypes $types [list $t1 $t2 $t3 $t4]]} then {
            if { [string equal $t1 X] || [string equal $t4 X] || [string equal $t2 X] || [string equal $t3 X]} {
                lappend delaywrite [format "%8s %8s %8s %8s  2  %8.3f %12.6f" \
                          $t1 $t2 $t3 $t4 $delta [expr {2 * $k * $kjinkcal}]]
            } else {
                puts $fp [format "%8s %8s %8s %8s  2  %8.3f %12.6f" \
                          $t1 $t2 $t3 $t4 $delta [expr {2 * $k * $kjinkcal}]]
            }
        }
    }
    #Gromacs dihedral type parser isn't very clever. It looks for the first matching dihedral,
    #therefore wildcard dihedrals must come last.
    foreach element $delaywrite {
        puts $fp $element
    }
}

proc ::TopoTools::writegmxcmapprm {fp cmap types} {
    variable kjinkcal
    puts $fp "\n\[ cmaptypes \]"
    foreach term $cmap {
        set rest [lassign $term t1 t2 t3 t4 t5 n]
        if {[findInTypes $types [list $t1 $t2 $t3 $t4 $t5]]} then {
            puts $fp [format "%s %s %s %s %s 1 %d %d\\" $t1 $t2 $t3 $t4 $t5 $n $n]
            for {set i 0} {$i < [llength $rest]} { incr i } {
                if {[expr {$i % 10}] == 9} {
                    puts $fp [format "%.8f\\" [expr {$kjinkcal * [lindex $rest $i]}]]
                } else {
                    puts -nonewline $fp [format "%.8f " \
                                             [expr {$kjinkcal * [lindex $rest $i]}]]
                }
            }
            #Don't forget to put a newline after the last of the 576.
            puts $fp ""
        }
    }
}

proc ::TopoTools::writegmxnbfixprm {fp nbfix types} {
    variable kjinkcal
    puts $fp "\n\[ nonbond_params \]"
    puts $fp ";type1 type2 1 sigma epsilon"
    set twoonesixth [expr { pow(2.0, 1.0/6)}]
    foreach term $nbfix {
        lassign $term t1 t2 epsilon rmin
        if {[findInTypes $types [list $t1 $t2]]} {
            puts $fp [format "%8s %8s  1  %.12f %.12f" $t1 $t2 \
                          [expr {$rmin * 0.1 / $twoonesixth}] \
                          [expr {abs($epsilon * $kjinkcal)}]]
        }
    }
}

proc ::TopoTools::writecharmmparams {fp mol sel filelist} {
    puts $fp "\[ defaults \]\n; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ"
    # This is comb-rule 2, which sums sigmas and multiplies epsilons.
    #See section 5.3.2 of the gromacs manual.
    puts $fp "1 2 yes 1.0 1.0"
    set cmap [list ]
    set bonds [list ]
    set angles [list ]
    set dihedrals [list ]
    set impropers [list ]
    set mass [dict create]
    set lj [list ]
    set nbfix [list ]
    foreach paramfile $filelist {
        set fin [open $paramfile r]
        set fdat [read $fin]
        close $fin
        set data [split $fdat "\n"]
        foreach line $data {
            # Try to find a comment character.
            # If found, discard the remainder of the line.
            set idx [string first ! $line]
            # Subtract one here, since if found, we don't
            # want to include it in the substring.
            incr idx -1
            if {$idx < 0} {
                set idx end
            }
            set l [string range $line 0 $idx]
            # Split based on whitespace.
            set ss [regexp -inline -all -- {\S+} $l]
            # Fit the split to one of the (type aware) parameter parsings,
            # ignore it if it doesn't fit.
            switch [llength $ss] {
                4 {
                    #Length 4: Bonds, LJ, NBFIX, certain CMAP data lines, MASS
                    #CMAP
                    if {[string is double [lindex $ss 0]] &&
                        [string is double [lindex $ss 1]] &&
                        [string is double [lindex $ss 2]] &&
                        [string is double [lindex $ss 3]]} then {
                        set cmaptmp [concat $cmaptmp $ss]
                        if {[llength $cmaptmp] == [expr {6 + [lindex $cmaptmp 5] * [lindex $cmaptmp 5]}]} {
                            lappend cmap $cmaptmp
                        }
                        #Bonds
                    } elseif {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                              [string is alnum [lindex $ss 1]] && [string is ascii [lindex $ss 1]] &&
                              [string is double [lindex $ss 2]] && [lindex $ss 2] >= 0 &&
                              [string is double [lindex $ss 3]]} then {
                        lappend bonds $ss
                        #NBFIX
                    } elseif {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                              [string is alnum [lindex $ss 1]] && [string is ascii [lindex $ss 1]] &&
                              [string is double [lindex $ss 2]] && [lindex $ss 2] < 0 &&
                              [string is double [lindex $ss 3]]} then {
                        lappend nbfix $ss
                        #LJ
                    } elseif {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                              [string is double [lindex $ss 1]] &&
                              [string is double [lindex $ss 2]] && [lindex $ss 2] < 0 &&
                              [string is double [lindex $ss 3]]} then {
                        lappend lj $ss
                        #MASS
                    } elseif {[lindex $ss 0] == "MASS" &&
                              [string is integer [lindex $ss 1]] &&
                              [string is alnum [lindex $ss 2]] && [string is ascii [lindex $ss 2]] &&
                              [string is double [lindex $ss 3]]} then {
                        dict set mass [lindex $ss 2] [lindex $ss 3]
                    }
                }

                5 {
                    #Length 5: Angles, CMAP data lines. Also some mass lines that are formatted for top files.
                    #CMAP
                    if {[string is double [lindex $ss 0]] &&
                        [string is double [lindex $ss 1]] &&
                        [string is double [lindex $ss 2]] &&
                        [string is double [lindex $ss 3]] &&
                        [string is double [lindex $ss 4]]} then {
                        set cmaptmp [concat $cmaptmp $ss]
                        #Angles
                    } elseif {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                              [string is alnum [lindex $ss 1]] && [string is ascii [lindex $ss 1]] &&
                              [string is alnum [lindex $ss 2]] && [string is ascii [lindex $ss 2]] &&
                              [string is double [lindex $ss 3]] &&
                              [string is double [lindex $ss 4]]} then {
                        lappend ss 0.0 0.0
                        lappend angles $ss
                    } elseif {[lindex $ss 0] == "MASS" &&
                              [string is integer [lindex $ss 1]] &&
                              [string is alnum [lindex $ss 2]] && [string is ascii [lindex $ss 2]] &&
                              [string is double [lindex $ss 3]] &&
                              [string is alpha [lindex $ss 4]] && [string is ascii [lindex $ss 4]]} then {
                        dict set mass [lindex $ss 2] [lindex $ss 3]
                    }
                }
                7 {
                    #Length 7: Angles (w/UB), dihedrals, impropers, nonbonded with 1-4 seperate.
                    #Angles
                    if {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                        [string is alnum [lindex $ss 1]] && [string is ascii [lindex $ss 1]] &&
                        [string is alnum [lindex $ss 2]] && [string is ascii [lindex $ss 2]] &&
                        [string is double [lindex $ss 3]] &&
                        [string is double [lindex $ss 4]] &&
                        [string is double [lindex $ss 5]] &&
                        [string is double [lindex $ss 6]]} then {
                        lappend angles $ss
                        #Dihedrals
                    } elseif {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                              [string is alnum [lindex $ss 1]] && [string is ascii [lindex $ss 1]] &&
                              [string is alnum [lindex $ss 2]] && [string is ascii [lindex $ss 2]] &&
                              [string is alnum [lindex $ss 3]] && [string is ascii [lindex $ss 3]] &&
                              [string is double [lindex $ss 4]] &&
                              [string is integer [lindex $ss 5]] && [expr {[lindex $ss 5] > 0}] &&
                              [string is double [lindex $ss 6]]} then {
                        lappend dihedrals $ss
                        #Impropers
                    } elseif {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                              [string is alnum [lindex $ss 1]] && [string is ascii [lindex $ss 1]] &&
                              [string is alnum [lindex $ss 2]] && [string is ascii [lindex $ss 2]] &&
                              [string is alnum [lindex $ss 3]] && [string is ascii [lindex $ss 3]] &&
                              [string is double [lindex $ss 4]] &&
                              [string is integer [lindex $ss 5]] && [lindex $ss 5] == 0 &&
                              [string is double [lindex $ss 6]]} then {
                        lappend impropers $ss
                        #Nonbonded
                    } elseif {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                              [string is double [lindex $ss 1]] &&
                              [string is double [lindex $ss 2]] &&
                              [string is double [lindex $ss 3]] &&
                              [string is double [lindex $ss 4]] &&
                              [string is double [lindex $ss 5]] &&
                              [string is double [lindex $ss 6]]} then {
                        lappend lj $ss
                    }
                }
                9 {
                    #Length 9: CMAP declarations
                    if {[string is alnum [lindex $ss 0]] && [string is ascii [lindex $ss 0]] &&
                        [string is alnum [lindex $ss 1]] && [string is ascii [lindex $ss 1]] &&
                        [string is alnum [lindex $ss 2]] && [string is ascii [lindex $ss 2]] &&
                        [string is alnum [lindex $ss 3]] && [string is ascii [lindex $ss 3]] &&
                        [string is alnum [lindex $ss 4]] && [string is ascii [lindex $ss 4]] &&
                        [string is alnum [lindex $ss 5]] && [string is ascii [lindex $ss 5]] &&
                        [string is alnum [lindex $ss 6]] && [string is ascii [lindex $ss 6]] &&
                        [string is alnum [lindex $ss 7]] && [string is ascii [lindex $ss 7]] &&
                        [string is integer [lindex $ss 8]]} then {
                        set cmaptmp [list [lindex $ss 0] [lindex $ss 1] [lindex $ss 2] [lindex $ss 3] [lindex $ss 7] [lindex $ss 8]]
                    }
                }
            }
        }
    }
    set types [lsort -unique [$sel get type]]
    #In case only a parameter file without MASS lines is passed,
    #lookup what the masses should be based on what exists in the current molecule.
    foreach type $types {
        if { ! [dict exists $mass $type]} {
            set subset [atomselect $mol "type $type"]
            dict set mass $type [lindex [$subset get mass] 0]
            $subset delete
        }
    }
    #Write the parameter lists to the output file.
    writegmxLJprm $fp $lj $mass $types
    writegmxbondprm $fp $bonds $types
    writegmxangleprm $fp $angles $types
    writegmxdihedralprm $fp $dihedrals $types
    writegmximproperprm $fp $impropers $types
    writegmxcmapprm $fp $cmap $types
    writegmxnbfixprm $fp $nbfix $types
}


proc ::TopoTools::get14pairs { sel } {
    set bondtable [$sel getbonds]
    set excl12 [list ]
    set excl13 [list ]
    set excl14 [list ]
    set idxlist [$sel get index]
    foreach i [$sel get index] {
        set bonds [lsort [lindex $bondtable [lsearch -exact $idxlist $i]]]
        foreach j $bonds {
            set bondj [lsort [lindex $bondtable [lsearch -exact $idxlist $j]]]
            #To avoid making these lists blow up, we do simple comparisons here so we
            #only add them to the list once.
            if { $i < $j } {
                lappend excl12 [list $i $j]
                foreach k $bonds {
                    if {$k < $j} {
                        lappend excl13 [list $k $j]
                    }
                    if {$k != $j} {
                        foreach l $bondj {
                            if {$l != $i && $l != $k} {
                                if { $k < $l } {
                                    lappend excl14 [list $k $l]
                                } else {
                                    lappend excl14 [list $l $k]
                                }
                            }
                        }
                    }
                }
            } else {
                #i < j not needed for angle/1-3 interactions.
                foreach k $bonds {
                    if {$k < $j} {
                        lappend excl13 [list $k $j]
                    }
                }
            }
        }
    }
    set excl123 [concat $excl12 $excl13]
    set retlist [list ]
    #For cyclic systems (<6 membered rings), it is possible that elements determined by
    #bonding alone would be excluded from the 1-4 list since they are really 1-2 or 1-3 pairs.
    #Also, for 6-membered rings, the naive implementation will pick up pairs across the rings twice (once in each direction around the ring).
    #The second check makes sure that those pairs are only included once as they should be,
    #otherwise those terms are included twice in the pairlist, which is incorrect.
    foreach pair $excl14 {
        if {[lsearch -exact $excl123 $pair] == -1 && [lsearch -exact $retlist $pair] == -1} {
            lappend retlist $pair
        }
    }
    return $retlist
}
