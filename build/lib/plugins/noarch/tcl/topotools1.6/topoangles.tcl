#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topoangles.tcl,v 1.12 2014/08/19 16:45:04 johns Exp $

# return info about angles
# we list and count only angles that are entirely within the selection.
proc ::TopoTools::angleinfo {infotype sel {flag none}} {

    set numangles 0
    array set angletypes {}
    set atomindex [$sel list]
    set anglelist {}

    foreach angle [join [molinfo [$sel molid] get angles]] {
        lassign $angle t a b c

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0) } {
            set angletypes($t) 1
            incr numangles
            lappend anglelist $angle
        }
    }
    switch $infotype {

        numangles      { return $numangles }
        numangletypes  { return [array size angletypes] }
        angletypenames { return [lsort -ascii [array names angletypes]] }
        getanglelist   { return $anglelist }
        default        { return "bug! shoot the programmer?"}
    }
}

# delete all fully contained angles of the selection.
proc ::TopoTools::clearangles {sel} {
    set mol [$sel molid]
    set atomindex [$sel list]
    set anglelist {}

    foreach angle [join [molinfo $mol get angles]] {
        lassign $angle t a b c

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0) } {
            lappend anglelist $angle
        }
    }
    molinfo $mol set angles [list $anglelist]
}

# reset angles to data in anglelist
proc ::TopoTools::setanglelist {sel anglelist} {

    set mol [$sel molid]
    set atomindex [$sel list]
    set newanglelist {}

    # set defaults
    set t unknown; set a -1; set b -1; set c -1

    # preserve all angles definitions that are not contained in $sel
    foreach angle [join [molinfo $mol get angles]] {
        lassign $angle t a b c

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0) } {
            lappend newanglelist $angle
        }
    }

    # append new ones, but only those fully contained in $sel
    foreach angle $anglelist {
        lassign $angle t a b c

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0) } {
            lappend newanglelist $angle
        }
    }

    molinfo $mol set angles [list $newanglelist]
}

# reset angles to data in anglelist
proc ::TopoTools::retypeangles {sel} {

    set mol [$sel molid]
    set anglelist [angleinfo getanglelist $sel]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newanglelist {}

    foreach angle $anglelist {
        lassign $angle type i1 i2 i3

        set idx [lsearch -sorted -integer $atomindex $i1]
        set a [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i2]
        set b [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i3]
        set c [lindex $atomtypes $idx]

        if { [string compare $a $c] > 0 } { set t $a; set a $c; set c $t }
        set type [join [list $a $b $c] "-"]

        lappend newanglelist [list $type $i1 $i2 $i3]
    }
    setanglelist $sel $newanglelist
}

# reset angles to definitions derived from bonds.
# this includes retyping of the angles.
proc ::TopoTools::guessangles {sel} {

    set mol [$sel molid]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newanglelist {}

    set bonddata [$sel getbonds]

    # preserve all angles definitions that are not fully contained in $sel
    foreach angle [angleinfo getanglelist $sel] {
        lassign $angle t a b c

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0) } {
            lappend newanglelist $angle
        }
    }

    # a topological angle is defined by two bonds that share an atom
    # bound to it that are not the bond itself
    foreach bonds $bonddata aidx $atomindex atyp $atomtypes {
        set nbnd [llength $bonds]
        for {set i 0} {$i < $nbnd-1} {incr i} {
            for {set j [expr {$i+1}]} {$j < $nbnd} {incr j} {
                set b1idx [lindex $bonds $i]
                set idx [lsearch -sorted -integer $atomindex $b1idx]
                set b1typ [lindex $atomtypes $idx]
                set b2idx [lindex $bonds $j]
                set idx [lsearch -sorted -integer $atomindex $b2idx]
                set b2typ [lindex $atomtypes $idx]
                if { ([string compare $b1typ $b2typ] > 0) } {
                    set t1 $b1typ; set b1typ $b2typ; set b2typ $t1
                    set t2 $b1idx; set b1idx $b2idx; set b2idx $t2
                }
                set type [join [list $b1typ $atyp $b2typ] "-"]

                # append only angles that are full contained in $sel
                if {([lsearch -sorted -integer $atomindex $b1idx] >= 0)          \
                        && ([lsearch -sorted -integer $atomindex $aidx] >= 0)   \
                        && ([lsearch -sorted -integer $atomindex $b2idx] >= 0) } {
                    lappend newanglelist [list $type $b1idx $aidx $b2idx]
                }
            }
        }
    }
    molinfo $mol set angles [list $newanglelist]
}

# define a new angle or change an existing one.
proc ::TopoTools::addangle {mol id1 id2 id3 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3"} sel]} {
        vmdcon -err "topology addangle: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id1 > $id3} {set t $id1 ; set id1 $id3 ; set id3 $t }

    set angles [join [molinfo $mol get angles]]
    lappend angles [list $type $id1 $id2 $id3]
    $sel delete
    molinfo $mol set angles [list $angles]
}

# delete an angle.
proc ::TopoTools::delangle {mol id1 id2 id3 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3"} sel]} {
        vmdcon -err "topology delangle: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id1 > $id3} {set t $id1 ; set id1 $id3 ; set id3 $t }

    set newanglelist {}
    foreach angle [join [molinfo $mol get angles]] {
        lassign $angle t a b c
        if { ($a != $id1) || ($b != $id2) || ($c != $id3) } {
            lappend newanglelist $angle
        }
    }
    $sel delete
    molinfo $mol set angles [list $newanglelist]
}
