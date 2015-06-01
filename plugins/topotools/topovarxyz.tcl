#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009,2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: topovarxyz.tcl,v 1.5 2014/08/19 16:45:04 johns Exp $

# high level subroutines for supporting xyz
# trajectories with a varying number of particles.
#
# import an xmol-format xyz trajectory data file.
# this behaves almost like a molfile plugin and will create a
# new molecule and return its molecule id.
# the special kick is, that this proc will handle .xyz
# files with a varying number of atoms and insert the
# necessary padding atoms and then set the "user" field
# to either 0 or 1 depending on whether the corresponding
# atom is present in the current frame.
#
# Arguments:
# filename = name of data file
# flags = more flags. (currently not used)
proc ::TopoTools::readvarxyz {filename {flags none}} {
    if {[catch {open $filename r} fp]} {
        vmdcon -err "readvarxyz: problem opening xyz file: $fp\n"
        return -1
    }

    # initialize local variables
    set nframes    0 ; # total number of frames
    set typemap   {} ; # atom type map
    set typecount {} ; # atom type count for one frame
    set maxcount  {} ; # max. atom type count for all frames
    array set traj {} ; # temporary trajectory storage

    # to be able to determine the number of dummy atoms we first
    # have to parse and store away the whole trajectory and while
    # doing so count the number of atom types in each frame.
    while {[gets $fp line] >= 0} {
        set numlines -1
        if {[regexp {^\s*([0-9]+)} $line x numlines]} {
            # first line is number of atoms
        } else {
            set numlines -1
        }
        if {$numlines < 0} break

        # skip next line
        if {[catch {gets $fp line} msg]} {
            vmdcon -err "readvarxyz: error reading frame $nframes of xyz file: $msg. "
            break
        }

        # collect data for this frame.
        set frame {}
        for {set i 0} {$i < $numlines} {incr i} {
            if {[catch {gets $fp line} msg]} {
                vmdcon -err "readvarxyz: error reading frame $nframes of xyz file: $msg. "
                break
            }
            lassign $line a x y z

            # lookup atom type in typemap and add if not found.
            set idx [lsearch -exact $typemap $a]
            if {$idx < 0} {
                set idx [llength $typecount]
                lappend typemap $a
                lappend typecount 0
                lappend maxcount 0
            }
            lset typecount $idx [expr {[lindex $typecount $idx] + 1}]
            lappend frame [list $idx $x $y $z]
        }

        # update list of max atoms per type and reset per frame counter.
        set newmax {}
        set newcount {}
        foreach t $typecount m $maxcount {
            if {$t > $m} {
                lappend newmax $t
            } else {
                lappend newmax $m
            }
            lappend newcount 0
        }
        set maxcount $newmax
        set typecount $newcount

        # add frame to storage, sort coordinates by type index.
        set traj($nframes)  [lsort -integer -index 0 $frame]
        incr nframes
    }
    close $fp

    # determine required number of atoms.
    set natoms 0
    foreach n $maxcount {
        incr natoms $n
    }

    vmdcon -info "readvarxyz: read in $nframes frames requiring $natoms atoms storage.\nType map: $typemap\nMax type counts: $maxcount"

    # create an empty molecule and timestep
    set mol -1
    if {[catch {mol new atoms $natoms} mol]} {
        vmdcon -err "readvarxyz: problem creating empty molecule: $mol"
        return -1
    }
    mol rename $mol [file tail $filename]

    # initialize some atom properties
    set sel [atomselect $mol all]
    set aname {}
    foreach t $typemap n $maxcount {
        for {set i 0} {$i < $n} {incr i} {
            lappend aname $t
        }
    }
    $sel set name $aname
    $sel set type $aname
    guessatomdata $sel element name
    guessatomdata $sel radius element
    guessatomdata $sel mass element

    for {set i 0} {$i < $nframes} {incr i} {
        animate dup $mol
        set data {}
        set idx -1
        set count [lindex $maxcount $idx]
        set j 0
        foreach c $traj($i) {
            while {[lindex $c 0] > $idx} {
                incr idx
                for {set k $j} {$k < $count} {incr k} {
                    lappend data {0.0 0.0 0.0 -1.0}
                }
                set count [lindex $maxcount $idx]
                set j 0
            }
            incr j
            set line [lrange $c 1 end]
            lappend line 1.0
            lappend data $line
        }
        for {set k $j} {$k < $count} {incr k} {
            lappend data {0.0 0.0 0.0 -1.0}
        }
        incr idx
        while {$idx < [llength $maxcount]} {
            set count [lindex $maxcount $idx]
            for {set j 0} {$j < $count} {incr j} {
                lappend data {0.0 0.0 0.0 -1.0}
            }
            incr idx
        }
        $sel set {x y z user} $data
    }
    mol reanalyze $mol

    # add default representation
    # to make this work we have to add "user > 0"
    # to the selection string and have the selection
    # being re-evaluated in every step.
    variable newaddsrep
    if {$newaddsrep} {
        adddefaultrep $mol "user > 0"
        mol selupdate 0 $mol on
    }
    return $mol
}

# Arguments:
# filename = name of data file
# flags = more flags. (currently not used)
proc ::TopoTools::writevarxyz {filename mol sel {flags {}}} {
    if {[catch {open $filename w} fp]} {
        vmdcon -err "writevarxyz: problem opening xyz file: $fp\n"
        return -1
    }

    # largest possible frame number
    set maxframe [molinfo $mol get numframes]
    incr maxframe -1

    set first 0
    set last $maxframe
    set step 1
    set nframe 0
    set selmod {user > 0}

    # parse optional flags
    foreach {key value} $flags {
        switch -- $key {
            first  {set first  $value}
            last   {set last   $value}
            step   {set step   $value}
            selmod {set selmod $value}
            default {
                vmdcon -err "writevarxyz: unknown flag: $key"
                return -1
            }
        }
    }

    set writesel [atomselect $mol "([$sel text]) and $selmod"]
    for {set i $first} {$i <= $last} {incr i $step} {
        if {$i > $maxframe} continue

        $writesel frame $i
        $writesel update

        puts $fp [$writesel num]
        puts $fp " Frame: $nframe"
        foreach line [$writesel get {name x y z}] {
            puts $fp $line
        }
        incr nframe
    }
    close $fp

    $writesel delete
    return $nframe
}

