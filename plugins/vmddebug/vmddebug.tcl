#!/usr/bin/tclsh
#
# VMDDebug, an infrastructure for debugging and profiling tasks in VMD.
#
# Copyright (c) 2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: vmddebug.tcl,v 1.3 2013/04/15 14:40:56 johns Exp $

namespace eval ::VMDDebug:: {
    # for allowing compatibility checks in scripts 
    # depending on this package. we'll have to expect
    variable version 1.1
}

# help/usage/error message and online documentation.
proc ::VMDDebug::usage {} {
    vmdcon -info "usage: debug <command> \[args...\] <flags>"
    vmdcon -info ""
    vmdcon -info "commands:"
    vmdcon -info "  help            prints this message"
    vmdcon -info ""
    vmdcon -info "  atomselect      tracing atomselect usage"
    vmdcon -info "     (use 'debug atomselect help' for more info)"
    vmdcon -info ""
    return
}

# the main frontend command.
# this takes care of all sanity checks on arguments and
# then dispatches the subcommands to the corresponding
# subroutines. 
proc VMDDebug::debug { args } {

    # process extract subcommand from arguments
    set cmd {}
    set newargs {}
    set retval {}
    if {[llength $args] > 0} {
        set cmd [lindex $args 0]
        set newargs [lrange $args 1 end]
    } else {
        set newargs {}
        set cmd help
    }

    # branch out to subcommands
    switch -- $cmd {
        atomselect {
            if {[catch {package require debugatomsel 1.0} ver]} {
                vmdcon -err "Could not load debugatomsel package: $ver"
                return 1
            }
            set retval [::DebugAtomSelect::debug_atomselect $newargs]
        }

        help -
        default {
            usage
        }
    }
    return $retval
}


# insert the "debug" frontend command into the normal namespace
interp alias {} debug {} ::VMDDebug::debug

package provide vmddebug $::VMDDebug::version

