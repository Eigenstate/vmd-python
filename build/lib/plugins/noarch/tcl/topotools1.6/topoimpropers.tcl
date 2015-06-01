#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topoimpropers.tcl,v 1.11 2014/08/19 16:45:04 johns Exp $

# return info about impropers
# we list and count only impropers that are entirely within the selection.
proc ::TopoTools::improperinfo {infotype sel {flag none}} {

    set numimpropers 0
    array set impropertypes {}
    set atomindex [$sel list]
    set improperlist {}

    foreach improper [join [molinfo [$sel molid] get impropers]] {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $d] >= 0) } {
            set impropertypes($t) 1
            incr numimpropers
            lappend improperlist $improper
        }
    }
    switch $infotype {

        numimpropers      { return $numimpropers }
        numimpropertypes  { return [array size impropertypes] }
        impropertypenames { return [lsort -ascii [array names impropertypes]] }
        getimproperlist   { return $improperlist }
        default        { return "bug! shoot the programmer?"}
    }
}

# delete all contained impropers of the selection.
proc ::TopoTools::clearimpropers {sel} {
    set mol [$sel molid]
    set atomindex [$sel list]
    set improperlist {}

    foreach improper [join [molinfo $mol get impropers]] {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0) } {
            lappend improperlist $improper
        }
    }
    molinfo $mol set impropers [list $improperlist]
}

# reset impropers to data in improperlist
proc ::TopoTools::setimproperlist {sel improperlist} {

    set mol [$sel molid]
    set atomindex [$sel list]
    set newimproperlist {}

    # set defaults
    set t unknown; set a -1; set b -1; set c -1; set d -1

    # preserve all impropers definitions that are not contained in $sel
    foreach improper [join [molinfo $mol get impropers]] {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0) } {
            lappend newimproperlist $improper
        }
    }

    # append new ones, but only those contained in $sel
    foreach improper $improperlist {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $d] >= 0) } {
            lappend newimproperlist $improper
        }
    }

    molinfo $mol set impropers [list $newimproperlist]
}

# reset impropers to data in improperlist
proc ::TopoTools::retypeimpropers {sel} {

    set mol [$sel molid]
    set improperlist [improperinfo getimproperlist $sel]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newimproperlist {}

    foreach improper $improperlist {
        lassign $improper type i1 i2 i3 i4

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

        lappend newimproperlist [list $type $i1 $i2 $i3 $i4]
    }
    setimproperlist $sel $newimproperlist
}

# reset impropers to definitions derived from bonds.
# this includes retyping of the impropers.
# this step is different from guessing angles or dihedrals,
# as we are only looking for definitions that are unusual.

proc ::TopoTools::guessimpropers {sel {flags {}}} {
    # default tolerance is 5 degrees from planar
    set tolerance 5

    # parse optional flags
    foreach {key value} $flags {
        switch -- $key {
            tol -
            tolerance {set tolerance  $value}
            default {
                vmdcon -err "guessimpropers: unknown flag: $key"
                return -1
            }
        }
    }

    set mol [$sel molid]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newimproperlist {}

    set bonddata [$sel getbonds]
    set minangle [expr {180.0 - $tolerance}]

    # preserve all impropers definitions that are not fully contained in $sel
    foreach improper [join [molinfo $mol get impropers]] {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0) } {
            lappend newimproperlist $improper
        }
    }

    # a topological improper is defined by three bonds connected to
    # the same atom and their dihedral being almost in plane.
    foreach bonds $bonddata aidx $atomindex atyp $atomtypes {
        set nbnd [llength $bonds]
        if {$nbnd == 3} {
            lassign $bonds b1 b2 b3
            set ang [expr {abs([measure imprp [list $b1 $b2 $aidx $b3] molid $mol])}]
            if {$ang > $minangle} {
                set b1idx [lsearch -sorted -integer $atomindex $b1]
                set b1typ [lindex $atomtypes $b1idx]
                set b2idx [lsearch -sorted -integer $atomindex $b2]
                set b2typ [lindex $atomtypes $b2idx]
                set b3idx [lsearch -sorted -integer $atomindex $b3]
                set b3typ [lindex $atomtypes $b3idx]

                if {([string compare $b1typ $b2typ]) > 0} {
                    set t1 $b1typ; set b1typ $b2typ; set b2typ $t1
                    set t2 $b1; set b1 $b2; set b2 $t2
                }
                if {([string compare $b2typ $b3typ]) > 0} {
                    set t1 $b2typ; set b2typ $b3typ; set b3typ $t1
                    set t2 $b2; set b2 $b3; set b3 $t2
                }
                if {([string compare $b1typ $b2typ]) > 0} {
                    set t1 $b1typ; set b1typ $b2typ; set b2typ $t1
                    set t2 $b1; set b1 $b2; set b2 $t2
                }
                set type [join [list $b1typ $b2typ $atyp $b3typ] "-"]
                lappend newimproperlist [list $type $b1 $b2 $aidx $b3]
            }
        }
    }
    setimproperlist $sel $newimproperlist
}

# define a new improper or change an existing one.
proc ::TopoTools::addimproper {mol id1 id2 id3 id4 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4"} sel]} {
        vmdcon -err "topology addimproper: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t
        set t $id1 ; set id1 $id4 ; set id4 $t
    }

    set impropers [join [molinfo $mol get impropers]]
    lappend impropers [list $type $id1 $id2 $id3 $id4]
    $sel delete
    molinfo $mol set impropers [list $impropers]
}

# delete a improper.
proc ::TopoTools::delimproper {mol id1 id2 id3 id4 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4"} sel]} {
        vmdcon -err "topology delimproper: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t
        set t $id1 ; set id1 $id4 ; set id4 $t
    }

    set newimproperlist {}
    foreach improper [join [molinfo $mol get impropers]] {
        lassign $improper t a b c d
        if { ($a != $id1) || ($b != $id2) || ($c != $id3) || ($d != $id4) } {
            lappend newimproperlist $improper
        }
    }
    $sel delete
    molinfo $mol set impropers [list $newimproperlist]
}
