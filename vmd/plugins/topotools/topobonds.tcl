#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topobonds.tcl,v 1.15 2014/08/19 16:45:04 johns Exp $

# Return info about bonds.
# we list and count only bonds that are entirely within the selection.
proc ::TopoTools::bondinfo {infotype sel {flag none}} {

    set numbonds 0
    set bidxlist {}
    array set bondtypes {}

    set aidxlist [$sel list]
    set bondlist [$sel getbonds]
    set btyplist [$sel getbondtypes]
    set bordlist [$sel getbondorders]

    foreach a $aidxlist bl $bondlist tl $btyplist ol $bordlist {
        foreach b $bl t $tl o $ol {
            if {($a < $b) && ([lsearch -sorted -integer $aidxlist $b] != -1)} {
                incr numbonds
                switch $flag {
                    type   {lappend bidxlist [list $a $b $t]}
                    order  {lappend bidxlist [list $a $b $o]}
                    both   {lappend bidxlist [list $a $b $t $o]}
                    lammps {lappend bidxlist [list $numbonds $a $b $t]}
                    none   {lappend bidxlist [list $a $b]}
                }
            }
            set bondtypes($t) 1
        }
    }

    switch $infotype {
        numbonds      { return $numbonds }
        numbondtypes  { return [array size bondtypes] }
        bondtypenames { return [lsort -ascii [array names bondtypes]] }
        getbondlist   { return $bidxlist }
        default       { return "bug? shoot the programmer!"}
    }
}

# delete all contained bonds of the selection.
proc ::TopoTools::clearbonds {sel} {

    # special optimization for "all" selection.
    if {[string equal "all" [$sel text]]} {
        set nulllist {}
        for {set i 0} {$i < [$sel num]} {incr i} {
            lappend nullist {}
        }
        $sel setbonds $nullist
        return
    }

    set mol [$sel molid]
    foreach b [bondinfo getbondlist $sel none] {
        delbond $mol [lindex $b 0] [lindex $b 1]
    }
}

# guess bonds from atom radii. Interface to "mol bondsrecalc".
# XXX: currently only works for selection "all".
proc ::TopoTools::guessbonds {sel} {

    set mol [$sel molid]
    # special optimization for "all" selection.
    if {[string equal "all" [$sel text]]} {
        # Use VMD's built-in bond determination heuristic to guess the bonds
        mol bondsrecalc $mol

        # Mark the bonds as "validated" so VMD will write
        # them out when the structure gets written out,
        # e.g. to a PSF file, even if no other bond editing was done.
        mol dataflag $mol set bonds

        return
    } else {
        vmdcon -err "topo guessbonds: this feature currently only works with an 'all' selection"
        return
    }
}

# reset bonds to data in bondlist
proc ::TopoTools::setbondlist {sel flag bondlist} {

    clearbonds $sel
    set nbnd [llength $bondlist]
    if {$nbnd == 0} { return 0}
    # set defaults
    set n 0
    set t unknown
    set o 1
    set mol [$sel molid]
    set a -1
    set b -1
    set fract  [expr {100.0/$nbnd}]
    set deltat 2000
    set newt   $deltat

    # special optimization for "all" selection.
    if {[string equal "all" [$sel text]]} {
        set nulllist {}
        for {set i 0} {$i < [$sel num]} {incr i} {
            set blist($i) $nulllist
            set olist($i) $nulllist
            set tlist($i) $nulllist
        }
        foreach bond $bondlist {
            switch $flag {
                type   {lassign $bond a b t  }
                order  {lassign $bond a b o  }
                both   {lassign $bond a b t o}
                lammps {lassign $bond n a b t}
                none   {lassign $bond a b    }
            }
            lappend blist($a) $b
            lappend blist($b) $a
            lappend olist($a) $o
            lappend olist($b) $o
            lappend tlist($a) $t
            lappend tlist($b) $t
        }
        set dlist {}
        for {set i 0} {$i < [$sel num]} {incr i} {
            lappend dlist $blist($i)
        }
        $sel setbonds $dlist
        set dlist {}
        for {set i 0} {$i < [$sel num]} {incr i} {
            lappend dlist $olist($i)
        }
        $sel setbondorders $dlist
        set dlist {}
        for {set i 0} {$i < [$sel num]} {incr i} {
            lappend dlist $tlist($i)
        }
        $sel setbondtypes $dlist
        return 0
    }

    # XXX: fixme!
    # using addbond is very inefficient with a large number of bonds
    # that are being added. it is better to fill the corresponding
    # bondlists directly. the code above should be better, but uses
    # much more memory and needs to be generalized.

    # XXX: add sanity check on data format
    set i 0
    foreach bond $bondlist {
        incr i
        set time [clock clicks -milliseconds]
        if {$time > $newt} {
            set percent [format "%3.1f" [expr {$i*$fract}]]
            vmdcon -info "setbondlist: $percent% done."
            display update ui
            set newt [expr {$time + $deltat}]
        }
        switch $flag {
            type   {lassign $bond a b t  }
            order  {lassign $bond a b o  }
            both   {lassign $bond a b t o}
            lammps {lassign $bond n a b t}
            none   {lassign $bond a b    }
        }
        addbond $mol $a $b $t $o
    }
    return 0
}

# guess bonds type names from atom types.
proc ::TopoTools::retypebonds {sel} {

    set bondlist  [bondinfo getbondlist $sel none]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newbonds {}

    foreach bond $bondlist {
        set idx [lsearch -sorted -integer $atomindex [lindex $bond 0]]
        set a [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex [lindex $bond 1]]
        set b [lindex $atomtypes $idx]
        if { [string compare $a $b] > 0 } { set t $a; set a $b; set b $t }
        set type [join [list $a $b] "-"]
        lappend newbonds [list [lindex $bond 0] [lindex $bond 1] $type]
    }
    setbondlist $sel type $newbonds
}


# define a new bond or change an existing one.
proc ::TopoTools::addbond {mol id1 id2 type order} {
    if {$id1 == $id2} {
        vmdcon -err "topo addbond: invalid atom indices: $id1 $id2"
        return
    }

    if {[catch {atomselect $mol "index $id1 $id2"} sel]} {
        vmdcon -err "topo addbond: Invalid atom indices: $sel"
        return
    }

    # make sure we have consistent indexing
    lassign [$sel list] id1 id2

    set bonds [$sel getbonds]
    set bords [$sel getbondorders]
    set btype [$sel getbondtypes]

    set b1 [lindex $bonds 0]
    set b2 [lindex $bonds 1]
    set bo1 [lindex $bords 0]
    set bo2 [lindex $bords 1]
    set bt1 [lindex $btype 0]
    set bt2 [lindex $btype 1]

    # handle the first atom...
    set pos [lsearch -exact -integer $b1 $id2]
    if { $pos < 0} {
        lappend b1 $id2
        lappend bo1 $order
        lappend bt1 $type
    } else {
        set bo1 [lreplace $bo1 $pos $pos $order]
        set bt1 [lreplace $bt1 $pos $pos $type]
    }

    # ...and the second one.
    set pos [lsearch -exact -integer $b2 $id1]
    if { $pos < 0} {
        lappend b2 $id1
        lappend bo2 $order
        lappend bt2 $type
    } else {
        set bo2 [lreplace $bo2 $pos $pos $order]
        set bt2 [lreplace $bt2 $pos $pos $type]
    }

    # and write the modified data back.
    $sel setbonds [list $b1 $b2]
    if {![string equal $order 1.0]} {
        $sel setbondorders [list $bo1 $bo2]
    }
    if {![string equal $type unknown]} {
        $sel setbondtypes [list $bt1 $bt2]
    }
    $sel delete
}

# delete a bond.
proc ::TopoTools::delbond {mol id1 id2 {type unknown} {order 1.0}} {
    if {[catch {atomselect $mol "index $id1 $id2"} sel]} {
        vmdcon -err "topology delbond: Invalid atom indices: $sel"
        return
    }

    # make sure we have consistent indexing
    lassign [$sel list] id1 id2

    set bonds [$sel getbonds]

    set b1 [lindex $bonds 0]
    set b2 [lindex $bonds 1]

    # handle the first atom...
    set pos [lsearch -exact -integer $b1 $id2]
    if { $pos < 0} {
        ; # bond is not completely within selection. ignore
    } else {
        set b1 [lreplace $b1 $pos $pos]
    }

    # ...and the second one.
    set pos [lsearch -exact -integer $b2 $id1]
    if { $pos < 0} {
        ; # bond is not completely within selection. ignore...
    } else {
        set b2 [lreplace $b2 $pos $pos]
    }

    # and write the modified data back.
    $sel setbonds [list $b1 $b2]
    $sel delete
}
