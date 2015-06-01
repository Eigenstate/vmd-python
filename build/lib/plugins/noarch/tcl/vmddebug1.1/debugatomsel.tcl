#!/usr/bin/tclsh
#
# DebugAtomSelect, a tool to track the usage of atom 
# selection functions in VMD Tcl scripts.
#
# Copyright (c) 2010,2011 by Axel Kohlmeyer <akohlmey@gmail.com>
# $Id: debugatomsel.tcl,v 1.3 2013/04/15 14:40:56 johns Exp $

##############################################################
# atomselect tracing and debugging
namespace eval ::DebugAtomSelect:: {
    variable version         1.1 ; # package version number
    variable verbose           0 ; # print a message whenever create/delete?
    variable status            0 ; # debugging on or off?
    variable started "not active"; # date when tracing started
    variable total_sel_count   0 ; # counter of total selections since active
    variable total_sel_added   0 ; # counter of selections added since active
    variable total_sel_deleted 0 ; # counter of selections deleted since active
    variable total_sel_autodel 0 ; # counter of selections automatically deleted
    variable wrapped_atomsels    ; # hash of selection procs that are wrapped
    variable alias_token         ; # token required to undo alias on atomselect.
    variable upproc_token        ; # token required to undo alias on upproc_del.
    array set wrapped_atomsels {}
    namespace export debug_atomselect
}

# this is the atomselect proxy that will trace the atomselect calls.
proc ::DebugAtomSelect::__atomselect_proxy {args} {
    variable verbose 
    variable total_sel_count
    variable total_sel_added
    variable wrapped_atomsels

    if {$verbose} {
        vmdcon -info "atomselect called with arguments: $args"
    }

    # the list command needs special treatment.
    if {([llength $args] > 0) && [string match [lindex $args 0] list]} {
        return [::__atomselect_real list]
    }
    
    set selname {}
    if {[catch {uplevel 1 ::__atomselect_real $args} selname]} {
        if {$verbose} {
            vmdcon -info "atomselect failed: $selname"
        }
        error $selname
    }
        
    incr total_sel_count
    incr total_sel_added
    rename ::$selname [join [list ::__ $selname _real] {}]
    set seltoken [interp alias {} ::$selname {} ::DebugAtomSelect::__atomselect_wrap $selname]
    if {$verbose} {
        vmdcon -info "wrapping selection function $selname / $seltoken"
    }
    set wrapped_atomsels($selname) $seltoken

    if {$verbose} {
        debug_atomselect stats
    }
    return $selname
}

# this is the wrapper for created atomselect functions to track deletes.
proc ::DebugAtomSelect::__atomselect_wrap {args} {
    variable verbose 
    variable total_sel_count
    variable total_sel_deleted
    variable wrapped_atomsels

    # remove first argument created by alias statement
    set selname [lindex $args 0]
    set args [lrange $args 1 end]

    if {$verbose} {
        vmdcon -info "$selname called with arguments: $args"
    }
    
    # the delete command needs special treatment.
    if {([llength $args] > 0) && [string match [lindex $args 0] delete]} {
        incr total_sel_count -1
        incr total_sel_deleted
        if {$verbose} {
            debug_atomselect stats1
        }
        # remove alias from interpreter, 
        interp alias {} $wrapped_atomsels($selname) {}
        rename [join [list ::__ $selname _real] {}] ::$selname
        # remove from list as well
        array unset wrapped_atomsels $selname
        return [uplevel 1 $selname delete]
    }
    return [uplevel 1 [join [list ::__ $selname _real] {}] $args]
}

# this is the wrapper for catching "localized" procedure deletes
proc ::DebugAtomSelect::__upproc_del_wrap {args} {
    variable verbose 
    variable total_sel_count
    variable total_sel_autodel
    variable wrapped_atomsels

    # get name of atom selection
    set selname [string range [lindex $args 0] 11 end]

    # find out if the selection has already been deleted
    if { [string length [array names wrapped_atomsels -exact $selname]] > 0} {
        incr total_sel_autodel
        incr total_sel_count -1

        vmdcon -warn "$selname is deleted automatically"

        if {$verbose} {
            debug_atomselect stats1
        }

        # remove alias from interpreter, 
        interp alias {} $wrapped_atomsels($selname) {}
        rename [join [list ::__ $selname _real] {}] ::$selname

        # remove from list as well
        array unset wrapped_atomsels $selname
        #rename ::$selname {}
    }
    return  [::__upproc_del_real [lindex $args 0] [lindex $args 1] [lindex $args 2]]
}


proc ::DebugAtomSelect::debug_atomselect {{flag help}} {
    variable verbose
    variable status
    variable started
    variable total_sel_count
    variable total_sel_added
    variable total_sel_deleted
    variable total_sel_autodel
    variable wrapped_atomsels
    variable alias_token
    variable upproc_token

    switch -- $flag {

        on  {
            if {$status} {
                if {$verbose} {
                    vmdcon -info \
                        "atomselect debugging already active $started"
                }
                return
            }
            if {![catch {rename ::atomselect ::__atomselect_real}]} {
                set alias_token [interp alias {} ::atomselect {} ::DebugAtomSelect::__atomselect_proxy]
                catch {rename ::upproc_del ::__upproc_del_real}
                set upproc_token [interp alias {} ::upproc_del {} ::DebugAtomSelect::__upproc_del_wrap]
                foreach selname [::__atomselect_real list] {
                    incr total_sel_count
                    rename ::$selname [join [list ::__ $selname _real] {}]
                    set seltoken [interp alias {} ::$selname {} ::DebugAtomSelect::__atomselect_wrap $selname]
                    if {$verbose} {
                        vmdcon -info "wrapping existing selection function $selname / $seltoken"
                    }
                    set wrapped_atomsels($selname) $seltoken
                }
            }
            set status 1
            set started "since [clock format [clock seconds]]"
        }
        
        off {
            if {!$status} {
                if {$verbose} {
                    vmdcon -info "atomselect debugging already inactive."
                }
                return
            }
            set status 0

            # remove alias and undo renaming
            interp alias {} $alias_token {}
            rename ::__atomselect_real ::atomselect
            interp alias {} $upproc_token {}
            rename ::__upproc_del_real ::upproc_del
            if {$verbose} {
                vmdcon -info "atomselect tracing now inactive"
            }
            debug_atomselect stats
            # unwrap the remaining atom selections.
            foreach {selname seltoken} [array get wrapped_atomsels] {
                vmdcon -info "unwrapping $selname  $seltoken"
                # remove alias from interpreter, 
                interp alias {} $seltoken {}
                rename [join [list ::__ $selname _real] {}] ::$selname
            }
            # clean up
            array unset wrapped_atomsels
            set total_sel_count   0
            set total_sel_added   0
            set total_sel_deleted 0
            set total_sel_autodel 0
            set started "not active"
            return
        }

        stats {
            vmdcon -info "atomselect statistics $started"
            if {$status} {
                vmdcon -info "total selections     : [llength [::__atomselect_real list]]"
            } else {
                vmdcon -info "total selections     : [llength [::atomselect list]]"
            }
            vmdcon -info "monitored selections : $total_sel_count"
            vmdcon -info "added selections     : $total_sel_added"
            vmdcon -info "deleted selections   : $total_sel_deleted"
            vmdcon -info "automatic deletes    : $total_sel_autodel"
            return
        }

        stats1 {
            # special hack when being called while selection is about to be deleted
            vmdcon -info "atomselect statistics $started"
            vmdcon -info "total selections     : [expr [llength [::__atomselect_real list]] - 1]"
            vmdcon -info "monitored selections : $total_sel_count"
            vmdcon -info "added selections     : $total_sel_added"
            vmdcon -info "deleted selections   : $total_sel_deleted"
            vmdcon -info "automatic deletes    : $total_sel_autodel"
            return
        }

        silent  {
            if {$verbose} {
                vmdcon -info "silencing verbose atomselect tracing"
            }
            set verbose 0
            return
        }

        verbose {
            set verbose 1
            vmdcon -info "verbose atomselect tracing enabled"
            return
        }

        help -  
        default {
            vmdcon -info "\nAtom selection tracing tool v$::DebugAtomSelect::version. Usage:\n"
            vmdcon -info "debug atomselect <flag>\n"
            vmdcon -info "Available flags:"
            vmdcon -info "  on      : enable tracing of atom selections"
            vmdcon -info "  off     : disable tracing of atom selections"
            vmdcon -info "  verbose : verbosely report atomselect operations"
            vmdcon -info "  silent  : don't report atomselect operations"
            vmdcon -info "  stats   : print statistics (active/added/deleted)"
            vmdcon -info "  help    : print this message\n"
        }
    }
}

package provide debugatomsel $::DebugAtomSelect::version
