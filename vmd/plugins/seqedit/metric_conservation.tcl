# University of Illinois Open Source License
# Copyright 2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: metric_conservation.tcl,v 1.5 2018/11/06 23:10:22 johns Exp $
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
namespace eval ::SeqEdit::Metric::Conservation {

    # Export the package namespace.
    namespace export setSeqColor

    # FIXME.  Need a calculate method that returns an array
    # with the values

# ------------------------------------------------------------------------
   # calculate values based on conservation
   proc calculate {sequenceIDs} {
      puts "metric_conservation.tcl.calculate.  start. seqIDs: $sequenceIDs"
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
                
               if {[info exists counts($elementIndex,$element)] == 0} {
                  set counts($elementIndex,$element) 1
               } else {
                  incr counts($elementIndex,$element)
               }
            }
                
            # Increment the element counter.
            incr elementIndex
         }
      }

      array unset metricMap 
            
      # Get the number of sequences.
      set numSequences [llength $sequenceIDs]
        
      # Go through each sequence in the list.
      foreach sequenceID $sequenceIDs {
        
         # Get the sequence.
         set sequence [SeqData::getSeq $sequenceID]
            
         # Go through each element in the sequence.
         set elementIndex 0
         foreach element $sequence {
                
            # Set the metric for this entry.
            if {$element != "-"} {
               set conservation [expr $counts($elementIndex,$element)/double($numSequences)]
               set metricMap($sequenceID,$elementIndex) $conservation
            } else {
               set metricMap($sequenceID,$elementIndex) 0.0
            }
            # Increment the element counter.
            incr elementIndex
         }
      }
      return [array get metricMap]
   } ; # end of calculate

# ------------------------------------------------------------------------
    # set colors based on conservation
    proc setSeqColor {sequenceIDs} {
        
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
                
                    if {[info exists counts($elementIndex,$element)] == 0} {
                        set counts($elementIndex,$element) 1
                    } else {
                        incr counts($elementIndex,$element)
                    }
                }
                
                # Increment the element counter.
                incr elementIndex
            }
        }
            
        # Get the function to map a value to an index.
        set colorNameMap "[::SeqEditWidget::getColorMap]\::getColorIndexForName"
        
        # Get the number of sequences.
        set numSequences [llength $sequenceIDs]
        
        # Go through each sequence in the list.
        foreach sequenceID $sequenceIDs {
        
            # Get the sequence.
            set sequence [SeqData::getSeq $sequenceID]
            
            # Go through each element in the sequence.
            set elementIndex 0
            foreach element $sequence {
                
                # Set the metric for this entry.
                if {$element != "-"} {
                    set conservation [expr $counts($elementIndex,$element)/double($numSequences)]
                    if {$conservation >= .95} {
                        seq set color $sequenceID $elementIndex [$colorNameMap iceblue]
                    } elseif {$conservation >= .70} {
                        seq set color $sequenceID $elementIndex [$colorNameMap pink]
                    } else {
                        seq set color $sequenceID $elementIndex [$colorNameMap white]
                    }
                } else {
                    seq set color $sequenceID $elementIndex [$colorNameMap white]
                }
                
                # Increment the element counter.
                incr elementIndex
            }
        }
    } ; # end of setSeqColor

# ------------------------------------------------------------------------
    proc calculatePerColumn {sequenceIDs} {
        
        # Initialize the counts.
        array unset counts 
        
        # Get the number of sequences.
        set numSequences [llength $sequenceIDs]
        
        # Go through each sequence in the list.
        foreach sequenceID $sequenceIDs {
        
            # Get the sequence.
            set sequence [SeqData::getSeq $sequenceID]
            
            # Go through each position in the sequence.
            set position 0
            foreach element $sequence {
                
                if {[info exists counts($position,elements)] == 0} {
                    set counts($position,elements) {}
                }
                    
                # Set the count entry for this element.
                if {$element != "-"} {
                
                    if {[info exists counts($position,$element)] == 0} {
                        lappend counts($position,elements) $element
                        set counts($position,$element) 1
                    } else {
                        incr counts($position,$element)
                    }
                }
                
                # Increment the element counter.
                incr position
            }
        }
        
        set conservations {}
        for {set position 0} {$position < [::SeqData::getSeqLength [lindex $sequenceIDs 0]]} {incr position} {
            set maxElement ""
            foreach element $counts($position,elements) {
                if {$maxElement == "" || $counts($position,$element) > $counts($position,$maxElement)} {
                    set maxElement $element
                }
            }
            lappend conservations [expr double($counts($position,$maxElement))/double($numSequences)]
        }
        
        return $conservations
    }
}
