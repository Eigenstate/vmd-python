#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topodihedrals.tcl,v 1.11 2014/08/19 16:45:04 johns Exp $

# return info about dihedrals
# we list and count only dihedrals that are entirely within the selection.
proc ::TopoTools::dihedralinfo {infotype sel {flag none}} {

    set numdihedrals 0
    array set dihedraltypes {}
    set atomindex [$sel list]
    set dihedrallist {}

    foreach dihedral [join [molinfo [$sel molid] get dihedrals]] {
        lassign $dihedral t a b c d

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $d] >= 0) } {
            set dihedraltypes($t) 1
            incr numdihedrals
            lappend dihedrallist $dihedral
        }
    }
    switch $infotype {

        numdihedrals      { return $numdihedrals }
        numdihedraltypes  { return [array size dihedraltypes] }
        dihedraltypenames { return [lsort -ascii [array names dihedraltypes]] }
        getdihedrallist   { return $dihedrallist }
        default        { return "bug! shoot the programmer?"}
    }
}

# delete all contained dihedrals of the selection.
proc ::TopoTools::cleardihedrals {sel} {
    set mol [$sel molid]
    set atomindex [$sel list]
    set dihedrallist {}

    foreach dihedral [join [molinfo $mol get dihedrals]] {
        lassign $dihedral t a b c d

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0) } {
            lappend dihedrallist $dihedral
        }
    }
    molinfo $mol set dihedrals [list $dihedrallist]
}

# reset dihedrals to data in dihedrallist
proc ::TopoTools::setdihedrallist {sel dihedrallist} {

    set mol [$sel molid]
    set atomindex [$sel list]
    set newdihedrallist {}

    # set defaults
    set t unknown; set a -1; set b -1; set c -1; set d -1

    # preserve all dihedrals definitions that are not fully contained in $sel
    foreach dihedral [join [molinfo $mol get dihedrals]] {
        lassign $dihedral t a b c d

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0) } {
            lappend newdihedrallist $dihedral
        }
    }

    # append new ones, but only those contained in $sel
    foreach dihedral $dihedrallist {
        lassign $dihedral t a b c d

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $d] >= 0) } {
            lappend newdihedrallist $dihedral
        }
    }

    molinfo $mol set dihedrals [list $newdihedrallist]
}

# reset dihedrals to data in dihedrallist
proc ::TopoTools::retypedihedrals {sel} {

    set mol [$sel molid]
    set dihedrallist [dihedralinfo getdihedrallist $sel]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newdihedrallist {}

    foreach dihedral $dihedrallist {
        lassign $dihedral type i1 i2 i3 i4

        set idx [lsearch -sorted -integer $atomindex $i1]
        set a [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i2]
        set b [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i3]
        set c [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i4]
        set d [lindex $atomtypes $idx]

        if { ([string compare $b $c] > 0) \
                 || ( [string equal $b $c] && [string compare $a $d] > 0 ) } {
            set t $a; set a $d; set d $t
            set t $b; set b $c; set c $t
            set t $i1; set i1 $i4; set i4 $t
            set t $i2; set i2 $i3; set i3 $t
        }
        set type [join [list $a $b $c $d] "-"]

        lappend newdihedrallist [list $type $i1 $i2 $i3 $i4]
    }
    setdihedrallist $sel $newdihedrallist
}


# reset dihedrals to definitions derived from bonds.
# this includes retyping of the dihedrals.
proc ::TopoTools::guessdihedrals {sel} {

    set mol [$sel molid]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newdihedrallist {}

    set bondlist [bondinfo getbondlist $sel]
    set bonddata [$sel getbonds]

    # preserve all dihedrals definitions that are not fully contained in $sel
    foreach dihedral [join [molinfo $mol get dihedrals]] {
        lassign $dihedral t a b c d

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0) } {
            lappend newdihedrallist $dihedral
        }
    }

    # a topological dihedral is defined by a bond and atoms
    # bound to it that are not the bond itself
    foreach bond $bondlist {
        lassign $bond b1 b2
        set b1idx [lsearch -sorted -integer $atomindex $b1]
        set b1typ [lindex $atomtypes $b1idx]
        set b2idx [lsearch -sorted -integer $atomindex $b2]
        set b2typ [lindex $atomtypes $b2idx]
        foreach o1 [lindex $bonddata $b1idx] {
            foreach o2 [lindex $bonddata $b2idx] {
                if {($o1 == $b1) || ($o2 == $b1) || ($o1 == $b2) || ($o2 == $b2)} {
                    continue
                }
                set o1idx [lsearch -sorted -integer $atomindex $o1]
                set o1typ [lindex $atomtypes $o1idx]
                set o2idx [lsearch -sorted -integer $atomindex $o2]
                set o2typ [lindex $atomtypes $o2idx]
                if { ([string compare $b1typ $b2typ] > 0) \
                 || ( [string equal $b1typ $b2typ]
                      && [string compare $o1typ $o2typ] > 0 ) } {
                    set type [join [list $o2typ $b2typ $b1typ $o1typ] "-"]
                    lappend newdihedrallist [list $type $o2 $b2 $b1 $o1]
                } else {
                    set type [join [list $o1typ $b1typ $b2typ $o2typ] "-"]
                    lappend newdihedrallist [list $type $o1 $b1 $b2 $o2]
                }
            }
        }
    }
    setdihedrallist $sel $newdihedrallist
}


# define a new dihedral or change an existing one.
proc ::TopoTools::adddihedral {mol id1 id2 id3 id4 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4"} sel]} {
        vmdcon -err "topology adddihedral: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t
        set t $id1 ; set id1 $id4 ; set id4 $t
    }

    set dihedrals [join [molinfo $mol get dihedrals]]
    lappend dihedrals [list $type $id1 $id2 $id3 $id4]
    molinfo $mol set dihedrals [list $dihedrals]
    # this is not (yet) required
    $sel delete
    return
}

# delete a dihedral.
proc ::TopoTools::deldihedral {mol id1 id2 id3 id4 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4"} sel]} {
        vmdcon -err "topology deldihedral: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t
        set t $id1 ; set id1 $id4 ; set id4 $t
    }

    set newdihedrallist {}
    foreach dihedral [join [molinfo $mol get dihedrals]] {
        lassign $dihedral t a b c d
        if { ($a != $id1) || ($b != $id2) || ($c != $id3) || ($d != $id4) } {
            lappend newdihedrallist $dihedral
        }
    }
    molinfo $mol set dihedrals [list $newdihedrallist]
    # this is not (yet) required
    $sel delete
    return
}
