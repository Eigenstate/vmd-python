#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# support for crossterms contributed by Josh Vermaas <vermaas2@illinois.edu>
#
# $Id: topocrossterms.tcl,v 1.4 2014/08/19 16:45:04 johns Exp $


proc ::TopoTools::crossterminfo {infotype sel {flag none}} {

    set numcrossterms 0
    set atomindex [$sel list]
    set crosstermlist {}

    # for backward compatibility with VMD versions before 1.9.2
    set ct {}
    if {[catch {molinfo [$sel molid] get crossterms} ct]} {
        vmdcon -warn "topotools: VMD [vmdinfo version] does not support crossterms"
        set ct {}
    }

    foreach crossterm [join $ct] {
        lassign $crossterm a b c d e f g h

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $d] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $e] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $f] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $g] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $h] >= 0)} {
            incr numcrossterms
            lappend crosstermlist $crossterm
        }
    }
    switch $infotype {

        numcrossterms      { return $numcrossterms }
        getcrosstermlist   { return $crosstermlist }
        default        { return "bug! shoot the programmer?"}
    }
}

# delete all contained crossterms of the selection.
proc ::TopoTools::clearcrossterms {sel} {
    set mol [$sel molid]
    set atomindex [$sel list]
    set crosstermlist {}

    # for backward compatibility with VMD versions before 1.9.2
    set ct {}
    if {[catch {molinfo [$sel molid] get crossterms} ct]} {
        vmdcon -warn "topotools: VMD [vmdinfo version] does not support crossterms"
        return -1
    }

    foreach crossterm [join $ct] {
        lassign $crossterm a b c d e f g h

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0)   \
                || ([lsearch -sorted -integer $atomindex $e] < 0)   \
                || ([lsearch -sorted -integer $atomindex $f] < 0)   \
                || ([lsearch -sorted -integer $atomindex $g] < 0)   \
                || ([lsearch -sorted -integer $atomindex $h] < 0)} {
            lappend crosstermlist $crossterm
        }
    }
    molinfo $mol set crossterms [list $crosstermlist]
}

# reset crossterms to data in crosstermlist
proc ::TopoTools::setcrosstermlist {sel crosstermlist} {

    set mol [$sel molid]
    set atomindex [$sel list]
    set newcrosstermlist {}

    # for backward compatibility with VMD versions before 1.9.2
    set ct {}
    if {[catch {molinfo $mol get crossterms} ct]} {
        vmdcon -warn "topotools: VMD [vmdinfo version] does not support crossterms"
        return -1
    }

    # set defaults
    set a -1; set b -1; set c -1; set d -1; set e -1; set f -1; set g -1; set h -1

    # preserve all crossterms definitions that are not contained in $sel
    foreach crossterm [join [molinfo $mol get crossterms]] {
        lassign $crossterm a b c d e f g h

        if {([lsearch -sorted -integer $atomindex $a] < 0)          \
                || ([lsearch -sorted -integer $atomindex $b] < 0)   \
                || ([lsearch -sorted -integer $atomindex $c] < 0)   \
                || ([lsearch -sorted -integer $atomindex $d] < 0)   \
                || ([lsearch -sorted -integer $atomindex $e] < 0)   \
                || ([lsearch -sorted -integer $atomindex $f] < 0)   \
                || ([lsearch -sorted -integer $atomindex $g] < 0)   \
                || ([lsearch -sorted -integer $atomindex $h] < 0)} {
            lappend crosstermlist $crossterm
        }
    }

    # append new ones, but only those contained in $sel
    foreach crossterm $crosstermlist {
        lassign $crossterm a b c d e f g h

        if {([lsearch -sorted -integer $atomindex $a] >= 0)          \
                && ([lsearch -sorted -integer $atomindex $b] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $c] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $d] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $e] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $f] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $g] >= 0)   \
                && ([lsearch -sorted -integer $atomindex $h] >= 0)} {
            lappend newcrosstermlist $crossterm
        }
    }

    molinfo $mol set crossterms [list $newcrosstermlist]
}

# define a new crossterm or change an existing one.
proc ::TopoTools::addcrossterm {mol id1 id2 id3 id4 id5 id6 id7 id8} {

    # for backward compatibility with VMD versions before 1.9.2
    set ct {}
    if {[catch {molinfo $mol get crossterms} ct]} {
        vmdcon -warn "topotools: VMD [vmdinfo version] does not support crossterms"
        return -1
    }

    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4 $id5 $id6 $id7 $id8"} sel]} {
        vmdcon -err "topology addcrossterm: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    #Cross terms are just two adjacent dihedrals, and so we apply the canonicalization operations seperately.
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t
        set t $id1 ; set id1 $id4 ; set id4 $t
    }
    if {$id6 > $id7} {
        set t $id6 ; set id2 $id7 ; set id7 $t
        set t $id5 ; set id5 $id8 ; set id8 $t
    }

    set crossterms [join [molinfo $mol get crossterms]]
    lappend crossterms [list $type $id1 $id2 $id3 $id4 $id5 $id6 $id7 $id8]
    $sel delete
    molinfo $mol set crossterms [list $crossterms]
}

# delete a crossterm.
proc ::TopoTools::delcrossterm {mol id1 id2 id3 id4 id5 id6 id7 id8} {

    # for backward compatibility with VMD versions before 1.9.2
    set ct {}
    if {[catch {molinfo $mol get crossterms} ct]} {
        vmdcon -warn "topotools: VMD [vmdinfo version] does not support crossterms"
        return -1
    }

    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4 $id5 $id6 $id7 $id8"} sel]} {
        vmdcon -err "topology delcrossterm: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    #Cross terms are just two adjacent dihedrals, and so we apply the canonicalization operations seperately.
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t
        set t $id1 ; set id1 $id4 ; set id4 $t
    }
    if {$id6 > $id7} {
        set t $id6 ; set id2 $id7 ; set id7 $t
        set t $id5 ; set id5 $id8 ; set id8 $t
    }

    set newcrosstermlist {}
    foreach crossterm [join [molinfo $mol get crossterms]] {
        lassign $crossterm a b c d e f g h
        if { ($a != $id1) || ($b != $id2) || ($c != $id3) || ($d != $id4) ||
            ($e != $id5) || ($f != $id6) || ($g != $id7) || ($h != $id8) } {
            lappend newcrosstermlist $crossterm
        }
    }
    $sel delete
    molinfo $mol set crossterms [list $newcrosstermlist]
}
