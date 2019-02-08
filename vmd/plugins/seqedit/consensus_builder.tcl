# University of Illinois Open Source License
# Copyright 2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: consensus_builder.tcl,v 1.4 2018/11/06 23:10:22 johns Exp $
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
# Author(s): Jonathan Montoya, Elijah Roberts

# This package implements a color map for the sequence editor that colors
# sequence elements based upon the signatures.

package provide seqedit 1.1
package require seqdata 1.1

# Declare global variables for this package.
namespace eval ::SeqEdit::ConsensusBuilder {

    # Export the package namespace.
    namespace export calculate

    proc getConsensusSequence {sequenceIDs {upperCutoff 0.9} {lowerCutoff 0.5}} {
        
        set numberSequences [llength $sequenceIDs]
        set alignmentLength [::SeqData::getSeqLength [lindex $sequenceIDs 0]]
        
        # Make sure the sequences are aligned.
        for {set i 1} {$i < $numberSequences} {incr i} {
            if {[::SeqData::getSeqLength [lindex $sequenceIDs $i]] != $alignmentLength} {
                error "A consensus sequence can only be constructed for aligned sequences."
            }
        }
        
        # Make sure the cutoffs are valid.
        if {$upperCutoff < $lowerCutoff} {
            error "The upper bounds for a consensus sequence must be less than the lower bounds."
        }
        
        # Figure out the consensus sequence.
        set consensusSequence {}
        for {set i 0} {$i < $alignmentLength} {incr i} { 

            # Count the elements at this position.
            array unset counts 
            foreach sequenceID $sequenceIDs {
                set element [::SeqData::getElement $sequenceID $i]
            
                 # Increase the count entry for this element.
                 if {![info exists counts($element)]} {
                     set counts($element) 1
                 } else {
                     incr counts($element)
                 }
            }
            
            # Figure out which element has the most occurrences.
            set maxElement ""
            set nextMaxElement ""
            foreach element [array names counts] {
                if {$maxElement == "" || $counts($element) >= $counts($maxElement)} {
                    set nextMaxElement $maxElement
                    set maxElement $element
                }
            }
            
            # Determine the consensus element.
            if {$nextMaxElement != "" && $counts($maxElement) == $counts($nextMaxElement)} {
                lappend consensusSequence ""
            } elseif {double($counts($maxElement))/double($numberSequences) >= $upperCutoff} {
                lappend consensusSequence [string toupper $maxElement]
            } elseif {double($counts($maxElement))/double($numberSequences) >= $lowerCutoff} {
                lappend consensusSequence [string tolower $maxElement]
            } else {
                lappend consensusSequence ""
            }
            
            unset counts
        }
        
        return $consensusSequence
    }
}
		
