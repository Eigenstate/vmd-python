# University of Illinois Open Source License
# Copyright 2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: metric_insertions.tcl,v 1.5 2018/11/06 23:10:22 johns Exp $
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
namespace eval ::SeqEdit::Metric::Insertions {

    # Export the package namespace.
    namespace export calculate

    # Gets the color that corresponds to the specified value.
    # args:     value - The value of which to retrieve the color, this should be between 0.0 and 1.0.
    # return:   A hex string representing the color associated with the passed in value.
    proc calculate {sequenceIDs} {
        
        # Get the function to map a value to an index.                  
        set colorValueMap "[::SeqEditWidget::getColorMap]\::getColorIndexForValue"
                
        # Make sure we have at least one sequence.
        if {[llength $sequenceIDs] == 0} {
            return {}
        }
        
        # Get the number of elements.
        set numberElements [::SeqData::getSeqLength [lindex $sequenceIDs 0]]
        
        # Find all of the sequences that have the same length.
        set matchingSequenceIDs {}
        foreach sequenceID $sequenceIDs {
        
            if {[::SeqData::getSeqLength $sequenceID] == $numberElements} {
                lappend matchingSequenceIDs $sequenceID
            }
        }
        set numberSequences [llength $matchingSequenceIDs]
        
        # Go through each element and calculate its insertion percentage.
        for {set i 0} {$i < $numberElements} {incr i} {
            
            # Get the total number of non-gaps elements at this position.
            set count 0
            foreach sequenceID $matchingSequenceIDs {
            
                # Get the element.
                set element [::SeqData::getElement $sequenceID $i]
                
                # If this is not a gap, increase the count.
                if {$element != "-"} {
                    incr count
                }
            }
            
            # Set the metric for each sequence.
            foreach sequenceID $matchingSequenceIDs {
#               puts "metric_insertions.calculate.setting color: seqID: sequenceID, i: $i"

                seq set color $sequenceID $i [$colorValueMap [expr (double($count))/(double($numberSequences))]]
            }
        }
    }
    
    proc calculatePerColumn {sequenceIDs} {
        
        # Initialize the counts.
        array unset counts 
        
        
        # Go through each position and calculate its insertion percentage.
        set insertions {}
        set numberPositions [::SeqData::getSeqLength [lindex $sequenceIDs 0]]
        set numberSequences [llength $sequenceIDs]
        for {set i 0} {$i < $numberPositions} {incr i} {
            
            # Get the total number of non-gaps at this position.
            set count 0
            foreach sequenceID $sequenceIDs {
                if {[::SeqData::getElement $sequenceID $i] != "-"} {
                    incr count
                }
            }
            lappend insertions [expr double($count)/double($numberSequences)]
        }
        
        return $insertions
    }
}
