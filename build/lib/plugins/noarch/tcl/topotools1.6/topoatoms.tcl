#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009,2010,2011,2012 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topoatoms.tcl,v 1.16 2015/02/13 21:32:42 johns Exp $

# Return info about atoms
# we list and count only bonds that are entirely within the selection.
proc ::TopoTools::atominfo {infotype sel {flag none}} {

    set atomtypes [lsort -ascii -unique [$sel get type]]

    switch $infotype {
        numatoms      { return [$sel num] }
        numatomtypes  { return [llength $atomtypes] }
        atomtypenames { return $atomtypes }
        default       { return "bug? shoot the programmer!"}
    }
}

# guess missing atomic property from periodic table data. numbers are
# taken from the corresponding lists in the molfile plugin header.
# TODO: additional guesses: element-name, mass-element, radius-element, ...
proc ::TopoTools::guessatomdata {sel what from} {
    variable elements
    variable masses
    variable radii

    set selstr [$sel text]

    switch -- "$what-$from" {
        lammps-data {
            # shortcut for lammps data files
            guessatomdata $sel element mass
            guessatomdata $sel name element
            guessatomdata $sel radius element
        }

        element-mass {
            foreach a [lsort -real -unique [$sel get mass]] {
                set s [atomselect [$sel molid] "mass $a and ( $selstr )"]
                $s set element [lindex $elements [ptefrommass $a]]
                $s delete
            }
        }

        element-name {
            foreach n [lsort -ascii -unique [$sel get name]] {
                set s [atomselect [$sel molid] "name '$n' and ( $selstr )"]
                set idx [lsearch -nocase $elements $n]
                if { $idx < 0} {
                    set n [string range $n 0 1]
                    set idx [lsearch -nocase $elements $n]
                    if {$idx < 0} {
                        set n [string range $n 0 0]
                        set idx [lsearch -nocase $elements $n]
                        if {$idx < 0} {
                            set n X
                        } else {
                            set n [lindex $elements $idx]
                        }
                    } else {
                        set n [lindex $elements $idx]
                    }
                } else {
                    set n [lindex $elements $idx]
                }
                $s set element $n
                $s delete
            }
        }

        element-type {
            foreach t [lsort -ascii -unique [$sel get type]] {
                set s [atomselect [$sel molid] "type '$t' and ( $selstr )"]
                set idx [lsearch -nocase $elements $t]
                if { $idx < 0} {
                    set t [string range $t 0 1]
                    set idx [lsearch -nocase $elements $t]
                    if {$idx < 0} {
                        set t [string range $t 0 0]
                        set idx [lsearch -nocase $elements $t]
                        if {$idx < 0} {
                            set t X
                        } else {
                            set t [lindex $elements $idx]
                        }
                    } else {
                        set t [lindex $elements $idx]
                    }
                } else {
                    set t [lindex $elements $idx]
                }
                $s set element $t
                $s delete
            }
        }

        mass-element {
            foreach e [lsort -ascii -unique [$sel get element]] {
                set s [atomselect [$sel molid] "element '$e' and ( $selstr )"]
                set idx [lsearch -nocase $elements $e]
                set m 0.0
                if {$idx >= 0} {
                    set m [lindex $masses $idx]
                }
                $s set mass $m
                $s delete
            }
        }

        name-element {
            # name is the same as element, only we go all uppercase.
            foreach e [lsort -ascii -unique [$sel get element]] {
                set s [atomselect [$sel molid] "element '$e' and ( $selstr )"]
                $s set name [string toupper $e]
                $s delete
            }
        }

        name-type {
            $sel set name [$sel get type]
        }

        radius-element {
            foreach e [lsort -ascii -unique [$sel get element]] {
                set s [atomselect [$sel molid] "element '$e' and ( $selstr )"]
                set idx [lsearch $elements $e]
                set r 2.0
                if {$idx >= 0} {
                    set r [lindex $radii $idx]
                }
                $s set radius $r
                $s delete
            }
        }

        type-element {
            # type is the same as element, only we go all uppercase.
            foreach e [lsort -ascii -unique [$sel get element]] {
                set s [atomselect [$sel molid] "element '$e' and ( $selstr )"]
                $s set type [string toupper $e]
                $s delete
            }
        }

        type-name {
            $sel set type [$sel get name]
        }

        default {
            vmdcon -err "guessatomdata: guessing '$what' from '$from' not implemented."
            vmdcon -err "Available are: element<-mass, element<-name, mass<element "
            vmdcon -err "name<element, radius<element name<type, type<element, type<name."
            return
        }
    }
}

