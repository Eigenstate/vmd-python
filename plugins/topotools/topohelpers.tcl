#!/usr/bin/tclsh
# TopoTools, a VMD package to simplify manipulating bonds
# other topology related properties in VMD.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topohelpers.tcl,v 1.9 2014/08/19 16:45:04 johns Exp $

# some (small) helper functions

# compare two lists element by element.
# return 0 if they are identical, or 1 if not.
proc ::TopoTools::listcmp {a b} {
    if {[llength $a] != [llength $b]} {
        return 1
    }
    foreach aa $a bb $b {
        if {![string equal $aa $bb]} {
            return 1
        }
    }
    return 0
}

# angle definition list comparison function
proc ::TopoTools::compareangles {a b} {
    lassign $a at a1 a2 a3
    lassign $b bt b1 b2 b3

    # canonicalize
    if {$a1 > $a3} { set t $a1 ; set a1 $a3; set a3 $t }
    if {$b1 > $b3} { set t $b1 ; set b1 $b3; set b3 $t }

    # compare. first center, then left, then right atom, finally type.
    if {$a2 < $b2} {
        return -1
    } elseif {$a2 > $b2} {
        return 1
    } else {
        if {$a1 < $b1} {
            return -1
        } elseif {$a1 > $b1} {
            return 1
        } else {
            if {$a3 < $b3} {
                return -1
            } elseif {$a3 > $b3} {
                return 1
            } else {
                return [string compare $at $bt]
            }
        }
    }
}

# dihedral definition list comparison function
proc ::TopoTools::comparedihedrals {a b} {
    lassign $a at a1 a2 a3 a4
    lassign $b bt b1 b2 b3 b4

    # canonicalize
    if {($a2 > $a3) || (($a2 == $a3) && ($a1 > $a4))} {
        set t $a1; set a1 $a4; set a4 $t
        set t $a2; set a2 $a3; set a3 $t
    }
    if {($b2 > $b3) || (($b2 == $b3) && ($b1 > $b4))} {
        set t $b1; set b1 $b4; set b4 $t
        set t $b2; set b2 $b3; set b3 $t
    }
    # compare. first center bond, then outside atoms, then type. start from left.
    if {$a2 < $b2} {
        return -1
    } elseif {$a2 > $b2} {
        return 1
    } else {
        if {$a3 < $b3} {
            return -1
        } elseif {$a3 > $b3} {
            return 1
        } else {
            if {$a1 < $b1} {
                return -1
            } elseif {$a1 > $b1} {
                return 1
            } else {
                if {$a4 < $b4} {
                    return -1
                } elseif {$a4 > $b4} {
                    return 1
                } else {
                    return [string compare $at $bt]
                }
            }
        }
    }
}

# improper dihedral definition list comparison function
# this assumes that the improper definition follows the
# usual convention that the 3rd atom is connected to the
# other three via bonds.
proc ::TopoTools::compareimpropers {a b} {
    lassign $a at a1 a2 a3 a4
    lassign $b bt b1 b2 b3 b4

    # canonicalize. same as in guessdihedrals.
    if {($a1 > $a2)} { set t $a1; set a1 $a2; set a2 $t }
    if {($a2 > $a3)} { set t $a2; set a2 $a3; set a3 $t }
    if {($a1 > $a2)} { set t $a1; set a1 $a2; set a2 $t }
    if {($b1 > $b2)} { set t $b1; set b1 $b2; set b2 $t }
    if {($b2 > $b3)} { set t $b2; set b2 $b3; set b3 $t }
    if {($b1 > $b2)} { set t $b1; set b1 $b2; set b2 $t }

    # compare. first center atom, then outside atoms, then type. start from left.
    if {$a3 < $b3} {
        return -1
    } elseif {$a3 > $b3} {
        return 1
    } else {
        if {$a1 < $b1} {
            return -1
        } elseif {$a1 > $b1} {
            return 1
        } else {
            if {$a2 < $b2} {
                return -1
            } elseif {$a2 > $b2} {
                return 1
            } else {
                if {$a4 < $b4} {
                    return -1
                } elseif {$a4 > $b4} {
                    return 1
                } else {
                    return [string compare $at $bt]
                }
            }
        }
    }
}

# sort angle/dihedral/improper list and remove duplicates
proc ::TopoTools::sortsomething {what sel} {

    switch $what {
        angle     {
            setanglelist $sel [lsort -unique -command compareangles \
                                     [angleinfo getanglelist $sel]]
        }
        dihedral  {
            setdihedrallist $sel [lsort -unique -command comparedihedrals \
                                     [dihedralinfo getdihedrallist $sel]]
        }
        improper  {
            setimproperlist $sel [lsort -unique -command compareimpropers \
                                     [improperinfo getimproperlist $sel]]
        }
    }
}

# emulate the behavior of loading a molecule through
# the regular "mol new" command. the options $selmod
# argument allows to append an additional modified to
# the selection, e.g. 'user > 0.1' for variable number
# particle xyz trajectories.
proc ::TopoTools::adddefaultrep {mol {selmod none}} {
    mol color [mol default color]
    mol rep [mol default style]
    if {[string equal $selmod none]} {
        mol selection [mol default selection]
    } else {
        mol selection "([mol default selection]) and $selmod"
    }
    mol material [mol default material]
    mol addrep $mol
    display resetview
}

# guess the atomic number in the peridic table from the mass
proc ::TopoTools::ptefrommass {{amass 0.0}} {
    variable masses

    set idx 0
    foreach m $masses {
        # this catches most cases.
        # we check the few exceptions later.
        if {[expr abs($amass-$m)] < 0.65} {
            set idx [lsearch $masses $m]
        }
        # this is a hydrogen or deuterium and we flag it as hydrogen.
        if {($amass > 0.0 && $amass < 2.2)} {
            set idx 1
        }
        # Differentiate between Bismutium and Polonium.
        # The normal search will detect Polonium.
        if {($amass > 207.85 && $amass < 208.99)} {
            set idx 83
        }
        # Differentiate between Cobalt and Nickel
        # The normal search will detect Nickel.
        if {($amass > 56.50 && $amass < 58.8133)} {
            set idx 27
        }
    }
    return $idx
}

# This exists to eliminate unneeded parameters from CHARMM parameter files.
# There are some oddly formatted files (particularly older ones) that will
# give parameters for atoms that aren't given LJ parameters.
# Naturally, this is a problem, so we don't include parameters
# for atomtypes not present in the psf or that include a wildcard.
proc ::TopoTools::findInTypes {types l} {
    foreach element $l {
        if { [lsearch $types $element] == -1 && $element != "X"} {
            return 0
        }
    }
    return 1
}
