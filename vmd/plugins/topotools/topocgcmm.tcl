#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topocgcmm.tcl,v 1.8 2014/08/19 16:45:04 johns Exp $

# high level subroutines for CMM coarse grain forcefield support.

# proc to parse wataru's cg parameter file format
# returns the full parameter database as a list of lists.
proc ::TopoTools::parse_cgcmm_parms {{filename par_CG.prm}} {
    variable datadir

    citation_reminder
    set atomtypes {}
    array set nonbondtypes {}
    array set bondtypes {}
    array set angletypes {}
    array set dihedraltypes {}
    array set impropertypes {}
    set fn [file join $datadir $filename]

    if {[catch {open $fn r} fp]} {
        vmdcon -err "could not open parm file $fn: $fp"
        return {}
    }

    set lineno 0
    set section none
    while {[gets $fp line] >= 0} {
        incr lineno
        # skip over comment and empty lines
        if {[regexp {^\s*\#.*} $line ]} { continue }
        if {[regexp {^\s*$} $line ]} { continue }
        # end of file
        if {[regexp {^\s*<end>\s*$} $line ]} { break }
        # unit flag. currently assumed to be kcal_per_mol
        if {[regexp {^\s*UNIT} $line ]} { continue }
        if {[regexp {^\s*>>\s+(ATOM|BOND|ANGLE|NONBOND|DIHEDRAL|IMPROPER)} \
                 $line -> keyword] } {
            set section $keyword
            continue
        }
        if {[regexp {^\s*<<\s*$} $line -> ] } {
            set section none
            continue
        }
        switch $keyword {
            ATOM {
                lappend atomtypes [lindex $line 0]
            }
            BOND {
                lassign $line a b t k r
                set bondtypes($a-$b) [list $t $r $k]
                set bondtypes($b-$a) [list $t $r $k]
            }
            ANGLE {
                lassign $line a b c t k r
                set angletypes($a-$b-$c) [list $t $r $k]
                set angletypes($c-$b-$a) [list $t $r $k]
            }
            NONBOND {
                lassign $line a b t e s m c
                set nonbondtypes($a-$b) [list $t $e $s $c]
                set nonbondtypes($b-$a) [list $t $e $s $c]
            }
            DIHEDRAL -
            IMPROPER {
                vmdcon -warn "$section keyword not yet supported. skipping..."
            }
            default {
                vmdcon -err "unknown keyword $section. aborting..."
                close $fp
                return {}
            }
        }
    }
    close $fp

# return the accumulated data in a systematic way
    set ret {}
    lappend ret $atomtypes
    lappend ret [array get nonbondtypes]
    lappend ret [array get bondtypes]
    lappend ret [array get angletypes]
    lappend ret [array get dihedraltypes]
    lappend ret [array get impropertypes]
    return $ret
}

# proc to parse wataru's cg topology file format
# returns the topology information for a single molecule
proc ::TopoTools::parse_cgcmm_topo {molname {filename top_CG.prm}} {
    variable datadir

    citation_reminder
    set atomdata {} ; # name type mass charge
    set bonddata {} ; # list of {from to} pairs (no type)
    set improperdata {} ; # dunno yet.
    set fn [file join $datadir $filename]

    if {[catch {open $fn r} fp]} {
        vmdcon -err "could not open parm file $fn: $fp"
        return {}
    }

    set lineno 0
    set section none
    while {[gets $fp line] >= 0} {
        incr lineno
        # skip over comment and empty lines
        if {[regexp {^\s*\#.*} $line ]} { continue }
        if {[regexp {^\s*$} $line ]} { continue }
        # end of file
        if {[regexp {^\s*<end>\s*$} $line ]} { break }
        # we found a section header
        if {[regexp {^\s*>>\s+(\w+)\s*} $line -> keyword] } {
            set section $keyword
            continue
        }
        if {[regexp {^\s*<<\s*$} $line -> ] } {
            set section none
            continue
        }
        # from here on we only care about data when we are within the right section.
        if {[string equal $section none]} { continue }
        if {[string equal $section $molname]} {
            vmdcon -info "parsing $section molecule definition"
            set numatoms 0
            set numbonds 0
            set numimpropers 0
            set alist {}
            set blist {}
            while {[gets $fp line] >= 0} {
                incr lineno
                if {[regexp {^\s*NUM(ATOM|BOND|IMPR)=\s*(\d+)\s*} $line -> keyword num]} {
                    switch $keyword {
                        ATOM { set numatoms $num }
                        BOND { set numbonds $num }
                        IMPR { set numimpropers $num }
                    }
                    continue
                }
                if {[regexp {^\s*(ATOM|BOND) \s*(.*)$} $line -> keyword dat]} {
                    switch $keyword {
                        ATOM { lappend alist [lrange $dat 1 4] }
                        BOND { set blist [join [list $blist $dat] " " ] }
                    }
                    if {([llength $alist] >= $numatoms) && ([llength $blist] >= 2*$numbonds) } {
                        break
                    }
                    continue
                }
                vmdcon -warn "parse_cgcmm_topo: skipping unknown data: $line"
            }
            foreach a $alist {
                lappend atomdata $a
            }
            foreach {from to} $blist {
                lappend bonddata [list $from $to]
            }
            # return the accumulated data in lists
            return [list $atomdata $bonddata $improperdata ]
        } else {
            continue
        }
    }
    close $fp

    vmdcon -err "parse_cgcmm_topo: entry '$molname' not found in $filename."
    return {}
}

# gently remind people that the should cite the cg papers.
proc ::TopoTools::citation_reminder {args} {
    variable cgcmmciteme

    if {$cgcmmciteme} {
        vmdcon -info "In any publication of scientific results based in part or completely on"
        vmdcon -info "the use of CG-CMM force field or derived tools, please reference:"
        vmdcon -info "W. Shinoda, R. H. DeVane, M. L. Klein, Multi-property fitting and "
        vmdcon -info "parameterization of a coarse grained model for aqueous surfactants,"
        vmdcon -info "Molecular Simulation, 33, 27-36 (2007)"
        vmdcon -info "and:"
        vmdcon -info "W. Shinoda, R. H. DeVane, M. L. Klein, Coarse-grained molecular modeling"
        vmdcon -info "of non-ionic surfactant self-assembly, Soft Matter, 4, 2453-2462 (2008)"

        set cgcmmciteme 0
    }
    return
}

# little proc to convert the lj type flag to the LAMMPS version,
# which is supported by both, LAMMPS and HOOMD.
proc ::TopoTools::canonical_cgcmm_ljtype {ljtype} {
    switch -exact -- $ljtype {
        124    -
        lj12_4 -
        LJ12-4 { return lj12_4 }
        126    -
        lj12_6 -
        LJ12-6 { return lj12_6 }
        96    -
        lj9_6 -
        LJ9-6  { return lj9_6  }
        default { return $ljtype }
    }
    return $ljtype
}

