#########################################################################
#                                                                       #
#            (C) Copyright 1995-2007 The Board of Trustees of the            #
#                        University of Illinois                         #
#                         All Rights Reserved                           #
#                                                                       #
#########################################################################

############################################################################
# RCS INFORMATION:
#
# 	$RCSfile: vectors.tcl,v $
# 	$Author: saam $	$Locker:  $		$State: Exp $
#	$Revision: 1.14 $	$Date: 2008/04/22 21:36:53 $
#
############################################################################
# DESCRIPTION:
#  These routines handle the vector and matrix manipulations needed for
# doing 3D transformations.
#
############################################################################
# This is part of the VMD installation.
# For more information about VMD, see http://www.ks.uiuc.edu/Research/vmd

# a vector is a n element list of numbers (in column form)
# a matrix is a 4x4 matrix represeneted as a 4 element list of 4 
#   4 elements, in row major form
#


set M_PI 3.14159265358979323846

# Function:  veczero
# Returns :    the zero vector, {0 0 0}
proc veczero {} {
    return {0 0 0}
}

# Function:  vecdist {vector v1} {vector v2}
# Returns :  || v2 - v1 ||
# This is syntactic sugar.
proc vecdist {x y} {
    veclength [vecsub $x $y]
}

# Function: vecdot {vector x} {vector y}
# Returns :  the vector dot product v1 * v2
proc vecdot {x y} {
    if {[llength $x] != [llength $y]} {
	error "vecdot needs vectors of the same size: $x : $y"
    }
    set ret 0
    foreach t1 $x t2 $y {
	set ret [expr {$ret + $t1 * $t2}]
    }
    return $ret
}

# Function: veccross {v1} {v2}
# Returns :  cross product of v1 and v2
proc veccross {x y} {
    foreach {x1 x2 x3} $x { break }
    foreach {y1 y2 y3} $y { break }
    set ret {}
    lappend ret [expr {   $x2 * $y3 - $y2 * $x3}]
    lappend ret [expr { - $x1 * $y3 + $y1 * $x3}]
    lappend ret [expr {   $x1 * $y2 - $y1 * $x2}]
    return $ret
}

# Function:  veclength2 {v}
#  Returns:    the square of the vector length
proc veclength2 {v} {
    set retval 0
    foreach term $v {
	set retval [expr {$retval + $term * $term}]
    }
    return $retval
}

# Function:  vecnorm {v}
#  Returns:    the normal vector pointing along v
proc vecnorm {v} {
    set sum [veclength $v]
    set retval {}
    foreach term $v {
	lappend retval [expr {$term / $sum}]
    }
    return $retval
}

# Function:  vecinvert {v}
#  Returns:    a vector with all terms inverted
proc vecinvert {v} {
    set ret {}
    foreach i $v {
	lappend ret [expr {-$i}]
    }
    return $ret
}

# Function: vecmul {vector x} {vector y}
# Returns : compute  x_i * y_i and return the resulting vector.
proc vecmul {x y} {
    if {[llength $x] != [llength $y]} {
        error "vecmul needs vectors of the same size: $x : $y"
    }
    set ret {}
    foreach t1 $x t2 $y {
         lappend ret [expr {$t1 * $t2}]
    }
    return $ret
}

# Function: coordtrans {matrix} {vector}
# Returns :  the vector = {matrix} * {vector}
#   If the matrix is 4x4 and the vector is length 3, the 4th element is 1
proc coordtrans {m v} {
    if { [llength $v] == 3} {
	lappend v 1
	return [lrange [vectrans $m $v] 0 2]
    }
    return [vectrans $m $v]
}


# Function: transidentity
#  Returns:   the identity matrix
proc transidentity { } {
 return "{1.0 0.0 0.0 0.0} {0.0 1.0 0.0 0.0} {0.0 0.0 1.0 0.0} {0.0 0.0 0.0 1.0}"
}

# Function: transtranspose {matrix}
# Returns :  the transpose of the matrix, as a matrix -- must be 4x4
proc transtranspose {m} {
    lassign $m m1 m2 m3 m4
    lassign $m1  m11 m12 m13 m14
    lassign $m2  m21 m22 m23 m24
    lassign $m3  m31 m32 m33 m34
    lassign $m4  m41 m42 m43 m44
    set retval {}
    lappend retval [concat $m11 $m21 $m31 $m41]
    lappend retval [concat $m12 $m22 $m32 $m42]
    lappend retval [concat $m13 $m23 $m33 $m43]
    lappend retval [concat $m14 $m24 $m34 $m44]
    return $retval
}
    
# Function:  find_rotation_value <list reference>
#  Returns:    value of the rotation in radians with the appropriate
#              list elements removed
proc find_rotation_value {varname} {
    global M_PI
    upvar $varname a
    if {![info exists a]} {
	error "find_rotation_value: don't know upvar $varname"
    }

    set amount [expr [lvarpop a] + 0.0]
    set units [lvarpop a]
    if {$units == "rad" || $units == "radians" || $units == "radian"} {
	# set amount $amount
    } elseif {$units == "pi"} {
	set amount [expr $amount * $M_PI]
    } elseif {$units == "deg" || $units == "degrees" || $units == "degree"} {
	set amount [expr $amount / 180.0 * $M_PI]
    } else {
	if {$units != ""} {
	    lvarpush a $units
	}
	# default is degrees
	set amount [expr $amount / 180.0 * $M_PI]
    }
    return $amount
}



# Function:  transaxis {'x' | 'y' | 'z'} amount { | deg | rad | pi }
#  Returns:    the matrix to rotate "amount" radians about the given axis
# the default angle measurement is "degrees"
proc transaxis {axis args} {
    global M_PI
    if { $axis != "x" && $axis != "y" && $axis != "z" } {
	error "transaxis must get either x, y, or z, not $axis"
    }
    set amount [find_rotation_value args]
    if { $args != ""} {
	error "Unknown angle measurement '$args' in transaxis"
    }

    set cos [expr cos($amount)]
    set mcos [expr -$cos]
    set sin [expr sin($amount)]
    set msin [expr -$sin]
    if { $axis == "x" } {
	set retval           "{1.0 0.0 0.0 0.0}"
	lappend retval [concat 0.0 $cos $msin 0.0]
	lappend retval [concat 0.0 $sin $cos 0.0]
	lappend retval        {0.0 0.0 0.0 1.0}
	return $retval
    }
    if { $axis == "y" } {
	set retval {}
	lappend retval [concat  $cos 0.0  $sin 0.0]
	lappend retval          {0.0 1.0  0.0 0.0}
	lappend retval [concat  $msin 0.0  $cos 0.0]
	lappend retval          {0.0 0.0  0.0 1.0}
	return $retval
    }
    if { $axis == "z" } {
	set retval {}
	lappend retval [concat  $cos $msin 0.0 0.0]
	lappend retval [concat  $sin $cos 0.0 0.0]
	lappend retval          {0.0 0.0 1.0 0.0}
        lappend retval          {0.0 0.0 0.0 1.0}
	return $retval
    }
}

# Function: transoffset <vector>
#  Returns:  the matrix needed to translate by vector
proc transoffset {v} {
    lassign $v x y z
    return "{1.0 0.0 0.0 $x} {0.0 1.0 0.0 $y} {0.0 0.0 1.0 $z} {0.0 0.0 0.0 1.0}"
}


# Function: trans
#  this has lots of options
# 
proc trans {args} {
    set origin {0.0 0.0 0.0}
    set offset {0.0 0.0 0.0}
    set axis {1.0 0.0 0.0}
    set amount 0
    set rotmat [transidentity]


    while { [set keyword [lvarpop args]] != ""} {
	if { $keyword == "origin" } {
	    set origin [lvarpop args]
	    continue
	}
	if { $keyword == "offset" } {
	    set offset [lvarpop args]
	    continue
	}
	if { $keyword == "center" } {
	    set offset [lvarpop args]
	    set origin $offset
	    continue
	}
	# alias 'x' to 'axis x', 'y' to 'axis y', 'z' to 'axis z'
	if { $keyword == "x" || $keyword == "y" || $keyword == "z"} {
	    lvarpush args $keyword
	    set keyword "axis"
	}
	if { $keyword == "axis" } {
	    set axis [lvarpop args]
	    if {$axis == "x"} {
		set axis {1.0 0.0 0.0}
	    } elseif {$axis == "y"} {
		set axis {0.0 1.0 0.0}
	    } elseif {$axis == "z"} {
		set axis {0.0 0.0 1.0}
	    } elseif {[llength $axis] != 3} {
		error "transform: axis must be 'x', 'y', 'z' or a vector, not $axis"
	    }
	    # find out how much to rotate
	    set amount [find_rotation_value args]

	    # and apply to the current rotation matrix
	    set rotmat [transmult [transabout $axis $amount rad] $rotmat]
	    set axis {1.0 0.0 0.0}
	    set amount 0.0
	    continue
	}
	if { $keyword == "bond" } {
	    set v1 [lvarpop args]
	    set v2 [lvarpop args]
	    set origin $v1
	    set offset $v1
	    set axis [vecsub $v2 $v1]
	    # find out how much to rotate
	    set amount [find_rotation_value args]
#	    puts "Axis is: $axis"
	    set rotmat [transabout $axis $amount rad]
#	    puts "Rotmat is: $rotmat"
	    continue
	}
	if { $keyword == "angle" } {
	    set v1 [lvarpop args]
	    set v2 [lvarpop args]
	    set v3 [lvarpop args]
	    set origin $v2
	    set offset $v2
	    set axis [veccross [vecsub $v2 $v1] [vecsub $v3 $v2]]
	    if {[veclength $axis] <= 0.0} {
		if {[veclength [veccross [vecnorm [vecsub $v1 $v2]] {1.0 0.0 0.0}]] < 0.01} {
		    set axis {0.0 0.0 1.0}
		    puts "Warning: transform found degenerate 'angle'; using z axis"
		} else {
		    set axis {1.0 0.0 0.0}
		    puts "Warning: transform found degenerate 'angle'; using x axis"
		}
	    } else {
		set axis [vecnorm $axis]
	    }
	    # find out how much to rotate
	    set amount [find_rotation_value args]
	    set rotmat [transabout $axis $amount rad]
	    continue
	}
	error "Unknown command for 'transform': $keyword"

    }
    # end of while loop
    set origmat [transoffset [vecinvert $origin]]
    set offsetmat [transoffset $offset]
#    puts "Orig: $origmat"
#    puts "Offset: $offsetmat"
#    puts "Rotmat: $rotmat"
#    puts [list Result: [transmult $offsetmat $rotmat $origmat]]
    return [transmult $offsetmat $rotmat $origmat]
}
# end of "transform"

# Function: trans_from_rotate
# Returns a transformation matrix given a 3x3 rotation matrix
proc trans_from_rotate {rotate} {
  lassign $rotate a b c
  return "{$a 0} {$b 0} {$c 0} {0 0 0 1}"
}

# Function: trans_to_rotate
#  Returns: the upper left 3x3 rotation component
proc trans_to_rotate {trans_matrix} {
  lassign $trans_matrix a b c
  lassign $a a1 a2 a3
  lassign $b b1 b2 b3
  lassign $c c1 c2 c3
  return "{$a1 $a2 $a3} {$b1 $b2 $b3} {$c1 $c2 $c3}"
}

# Function: trans_from_offset
#  Returns: the transformation corresponding to an offset vector
proc trans_from_offset {offset} {
    return [transoffset $offset]
}

# Function: trans_to_offset
#  Returns: the transformation offset of the given matrix
proc trans_to_offset {trans_matrix} {
    return [coordtrans $trans_matrix {0 0 0}]
}
# set nothing "Okay!"
