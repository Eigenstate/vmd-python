############################################################################
#cr                                                                       
#cr            (C) Copyright 1995-2007 The Board of Trustees of the            
#cr                        University of Illinois                         
#cr                         All Rights Reserved                           
#cr                                                                       
############################################################################

############################################################################
# RCS INFORMATION:
#
# 	$RCSfile: atomselect.tcl,v $
# 	$Author: akohlmey $	$Locker:  $		$State: Exp $
#	$Revision: 1.9 $	$Date: 2011/02/26 17:55:31 $
#
############################################################################
# DESCRIPTION:
#   The ancillary "atomselect" functions.  These are a pseudo-lookup
# by atom select in that if the C++ code doesn't find the command,
# it calls "vmd_atomselect_<command>" so, for instance, "$sel move $tmat"
# is translated to "vmd_atomselect_moveby <selection> <tmat>"
############################################################################

proc vmd_atomselect_lmoveby {sel vectlist} {
    if {[llength $vectlist] != [$sel num]} {
	error [list atomselect: lmoveby: [llength $vectlist] \
	       vectors in list but [$sel num] in selection $sel]
    }
    set coords {}
    foreach v [$sel get {x y z}] {
	lappend coords [vecadd $v [lvarpop vectlist]]
    }
    $sel set {x y z} $coords
}

#  make 'n' duplicates of the args
proc ldup {count val} {
    set retval {}
    for {set i 0} {$i < $count} {incr i} {
        lappend retval $val
    }
    return $retval
}

# trivial, but here for completeness
proc vmd_atomselect_moveto {sel vect} {
    $sel set {x y z} [ldup [$sel num] $vect]
}
proc vmd_atomselect_lmoveto {sel vectlist} {
    $sel set {x y z} $vectlist
}

####### Note quite an "atomselect" term, but close enough to 
# merit being here

# make 'coord' be the center for all the active molecules.  This is
# called from the mouse pull-down menu "center" command item specific
# to a given atom
proc vmd_set_center {coord} {
    foreach mol [molinfo list] {
	if [molinfo $mol get active] {
	    molinfo $mol set center [list $coord]
	}
    }
}

#######
# print information about a given atom
proc vmd_print_atom_info {molid atomid} {
    set sel [atomselect $molid "index $atomid"]
    if {[$sel num] != 1} {
	vmdcon -error "Error in vmd_print_atom_info '$molid' '$atomid'"
	return
    }
    # get the attributes
    set attr {name type index resname resid chain segname x y z}
    set data [lindex [$sel get $attr] 0]
    foreach x $attr {
	set dat [lvarpop data]
	vmdcon -info "$x: $dat"
    }
}

##############################################################
# localize procedures in Tcl

# This makes a "local" procedure, in that the procedure is removed
# once the local scope is completed.
# Side effect -- if there is already a procedure with this name,
# it is overwritten
# This method works by setting the local variable "upproc_var_$name" and
# setting a trace on it upon unsetting (as when the context is finished)
# This calls "upproc_del" which calls 'rename' on the local proc


# given the $name as something like atomsel0
# I rename the function to nothing (calls the delete proc)
proc upproc_del {name element op} {
  set self [string range $name 11 end]
  rename $self {}
}

# Usage: upproc level procname
# Makes the procedure $procname "local" to the given scope (level takes
# the same parameters as "uplevel", ie, a number or #number).  This
# works by setting upproc_var_$procname in the given namespace and
# ties a trace from the variable to the deletion function.  If the proc
# is local to the current namespace and upproc is called, the current
# tie is removed.  There is no search of the call chain to remove other
# ties.

proc upproc {level procname} {
    # remove a current tie, if it exists
    uplevel 1 [list trace vdelete upproc_var_$procname u upproc_del]
    # ignore if not a number, else raise the level, since I'm in a proc
    if {! [catch {expr $level}]} {incr level}
    # make the local variable (I never use the actual definition)
    uplevel $level [list set upproc_var_$procname 1]
    # set the trace
    uplevel $level [list trace variable upproc_var_$procname u upproc_del]
}
