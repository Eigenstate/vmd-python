# University of Illinois Open Source License
# Copyright 2004-2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: colormap_vmd.tcl,v 1.7 2018/11/06 23:02:49 johns Exp $
# 
# Developed by: Luthey-Schulten Group
# 			     University of Illinois at Urbana-Champaign
# 			     http://faculty.scs.illinois.edu/schulten/
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the Software), to deal with 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to 
# do so, subject to the following conditions:
# 
# - Redistributions of source code must retain the above copyright notice, 
# this list of conditions and the following disclaimers.
# 
# - Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimers in the documentation 
# and/or other materials provided with the distribution.
# 
# - Neither the names of the Luthey-Schulten Group, University of Illinois at
# Urbana-Champaign, nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL 
# THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS WITH THE SOFTWARE.
#
# Author(s): Elijah Roberts

# This package implements a color map for the sequence editor that colors
# sequence elements based upon the conservation of the element.

package provide multiseq 3.1

# Declare global variables for this package.
namespace eval ::MultiSeq::ColorMap::VMD {

    # Export the package namespace.
    namespace export getColorMap
    
    variable colors {}

    variable oldScaleMethod ""

# ---------------------------------------------------------------------
   proc loadColorMap {} {
      variable colors
      variable oldScaleMethod
      set ci "[colorinfo scale method] [colorinfo scale midpoint] [colorinfo scale min]"
      if { $ci == $oldScaleMethod } {
         return
      } else {
         set oldScaleMethod $ci
         set colors ""
         # Get the colors from the VMD palette.
         for {set i 0} {$i < [colorinfo max]} {incr i} {
            set components [colorinfo rgb $i]
            set r [expr int(double(0xFF)*[lindex $components 0])]
            set g [expr int(double(0xFF)*[lindex $components 1])]
            set b [expr int(double(0xFF)*[lindex $components 2])]
            lappend colors "#[format %02X $r][format %02X $g][format %02X $b]"
         }
#         puts "colors:  $colors"
      }
   }

# ---------------------------------------------------------------------
    # Gets the color that corresponds to the specified value.
    # args:     value - The value of which to retrieve the color, this should be between 0.0 and 1.0.
    # return:   A hex string representing the color associated with the passed in value.
    proc getColor {index} {

#       # Get the color from the VMD palette.
#       set components [colorinfo rgb $index]

#       # Get the color components
#       set r [expr int(double(0xFF)*[lindex $components 0])]
#       set g [expr int(double(0xFF)*[lindex $components 1])]
#       set b [expr int(double(0xFF)*[lindex $components 2])]

#       set color "#[format %02X $r][format %02X $g][format %02X $b]"
#       return $color

        variable colors
        return [lindex $colors $index]
    }
    
    proc getColorIndexForName {name} {
        return [colorinfo index $name]
    }
    
    # Maps a value from 0 to 1.0 to a color index.
    proc getColorIndexForValue {value} {
        return [expr int(($value)*([colorinfo max]-[colorinfo num]-1))+[colorinfo num]]
    }
    
    # Maps a color index to a value from 0 to 1.0.
    proc getColorValueForIndex {index} {
        return [expr double($index-[colorinfo num])/double([colorinfo max]-[colorinfo num]-1)]
    }



#    # Maps a value from -1.0 to 1.0 to a color index.
#    proc getColorIndexForValue {value} {
#        return [expr int((($value/2.0)+0.5)*([colorinfo max]-[colorinfo num]-1))+[colorinfo num]]
#    }
#    
#    # Maps a color index to a value from -1.0 to 1.0.
#    proc getColorValueForIndex {index} {
#        return [expr ((double($index-[colorinfo num])/double([colorinfo max]-[colorinfo num]-1))-0.5)*2.0]
#    }

}    
