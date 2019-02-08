# University of Illinois Open Source License
# Copyright 2004-2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: metric_percentid.tcl,v 1.6 2018/11/06 23:10:22 johns Exp $
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

package provide seqedit 1.1
package require seqdata 1.1

# Declare global variables for this package.
namespace eval ::SeqEdit::Metric::PercentIdentity {

    # Export the package namespace.
    namespace export calculate

# ------------------------------------------------------------------------
    # color by percid
    # args:     sequence IDs to compare
    # return:   nother
   proc color {sequenceIDs} {
      # Get the function to map a value to an index.
      set colorValueMap "[::SeqEditWidget::getColorMap]\::getColorIndexForValue"

      # Get the number of sequences.
      set numSequences [llength $sequenceIDs]

      # Initialize the color map.
      array unset counts 

      # Go through each sequence in the list.
      foreach sequenceID $sequenceIDs {

         # Get the sequence.
         set sequence [SeqData::getSeq $sequenceID]

         # Go through each element in the sequence.
         set elementIndex 0
         foreach element $sequence {

            # Set the count entry for this element.
            if {$element != "-"} {

               if {![info exists counts($elementIndex,$element)]} {
                  set counts($elementIndex,$element) 1
               } else {
                  incr counts($elementIndex,$element)
               }
            }

            incr elementIndex
         }
      }

      # Go through each sequence in the list.
      foreach sequenceID $sequenceIDs {

         # Get the sequence.
         set sequence [SeqData::getSeq $sequenceID]

         # Go through each element in the sequence.
         set elementIndex 0
         foreach element $sequence {

            # Set the color for this entry.
            if {$element != "-"} {
               if {$numSequences > 1} {
                  set pi [expr ($counts($elementIndex,$element)-1)/(double($numSequences)-1.0)]
               } else {
                  set pi 1.0
               }
            } else {
               set pi 0.0
            }
            seq set color $sequenceID $elementIndex [$colorValueMap $pi]

            # Increment the element counter.
            incr elementIndex
         }
      }
   }

# ------------------------------------------------------------------------
    # calculate percid and return as array
    # args:     sequence IDs to compare
    # return:   nother
   proc calculate {sequenceIDs} {

      # Get the function to map a value to an index.
      set colorValueMap "[::SeqEditWidget::getColorMap]\::getColorIndexForValue"

      # Get the number of sequences.
      set numSequences [llength $sequenceIDs]

      # Initialize the color map.
      array unset counts 

      # Go through each sequence in the list.
      foreach sequenceID $sequenceIDs {

         # Get the sequence.
         set sequence [SeqData::getSeq $sequenceID]

         # Go through each element in the sequence.
         set elementIndex 0
         foreach element $sequence {

            # Set the count entry for this element.
            if {$element != "-"} {

               if {![info exists counts($elementIndex,$element)]} {
                  set counts($elementIndex,$element) 1
               } else {
                  incr counts($elementIndex,$element)
               }
            }

            incr elementIndex
         }
      }

      array unset percMap 
      # Go through each sequence in the list.
      foreach sequenceID $sequenceIDs {

         # Get the sequence.
         set sequence [SeqData::getSeq $sequenceID]


         # Go through each element in the sequence.
         set elementIndex 0
         foreach element $sequence {

            # Set the color for this entry.
            if {$element != "-"} {
               if {$numSequences > 1} {
                  set pi [expr ($counts($elementIndex,$element)-1)/(double($numSequences)-1.0)]
               } else {
                  set pi 1.0
               }
            } else {
               set pi 0.0
            }

            set percMap($sequenceID,$elementIndex) $pi

            # Increment the element counter.
            incr elementIndex
         }
      }
      return [array get percMap]
   }
}
