# University of Illinois Open Source License
# Copyright 2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: metric_signatures.tcl,v 1.5 2018/11/06 23:10:22 johns Exp $
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
namespace eval ::SeqEdit::Metric::Signatures {
    
    variable counts

    proc calculate {w} {
        
        set sequenceGroups {}
        array set options [showSignatureOptionsDialog $w [::SeqEditWidget::getGroups]]
        
        if {[info exists options(groups)] && [info exists options(groupConsensusCutoff)] && [info exists options(otherGroupMaxCutoff)] && [info exists options(otherGroupMaxGapFraction)] && [info exists options(maxConservedBlockDistance)] && [info exists options(minConservedBlockSize)]} {
            set sequenceGroups {}
            foreach groupName $options(groups) {
                lappend sequenceGroups [::SeqEditWidget::getSequencesInGroup $groupName]
            }

            performSignatureColoring $sequenceGroups $options(groupConsensusCutoff) $options(otherGroupMaxCutoff) $options(otherGroupMaxGapFraction) $options(maxConservedBlockDistance) $options(minConservedBlockSize)
        }
    }
    
    proc performSignatureColoring {sequenceGroups {groupConsensusCutoff 0.9} {otherGroupMaxCutoff 0.05} {otherGroupMaxGapFraction 0.5} {maxConservedBlockDistance 10} {minConservedBlockSize 2}} {
        
        # Get the function to map a value to an index.
        set colorNameMap "[::SeqEditWidget::getColorMap]\::getColorIndexForName"
        
        # Make sure the groups are valid.
        set ret [validateGroups $sequenceGroups]
        set sequenceGroups [lindex $ret 0]
        set numberSequences [lindex $ret 1]
        set alignmentLength [lindex $ret 2]
        
        # Get the signatures
        set signatures [calculateSignatures $sequenceGroups $groupConsensusCutoff $otherGroupMaxCutoff $otherGroupMaxGapFraction $maxConservedBlockDistance $minConservedBlockSize]
                        
        # Go through the groups.
        for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
                
            set groupSignature [lindex $signatures $groupIndex]
                
            # Go through the alignment.
            for {set elementIndex 0} {$elementIndex < $alignmentLength} {incr elementIndex} {
                
                # Get the signature for this group.
                set signatureType 1
                set signatureElement [lindex $groupSignature $elementIndex]
                if {[string length $signatureElement] == 0} {
                    set signatureType 0
                    set signatureElement ""
                } elseif {[string length $signatureElement] == 2 && [string index $signatureElement 1] == "!"} {
                    set signatureType 2
                    set signatureElement [string index $signatureElement 0]
                } elseif {[string length $signatureElement] == 2 && [string index $signatureElement 1] == "?"} {
                    set signatureType 3
                    set signatureElement [string index $signatureElement 0]
                }
                
                # Go through the seqeunces.
                foreach sequenceID [lindex $sequenceGroups $groupIndex] {
                    
                    # If this is not a signature, color it white.
                    if {$signatureType == 0} {
                        seq set color $sequenceID $elementIndex [$colorNameMap white]
                    
                    # If this is a signature, figure out how to color it.
                    } elseif {$signatureType == 1} {
                        set element [::SeqData::getElement $sequenceID $elementIndex]
                        if {$element == $signatureElement} {
                            seq set color $sequenceID $elementIndex [$colorNameMap blue]
                        } else {
                            seq set color $sequenceID $elementIndex [$colorNameMap red]
                        }
                        
                    # If this is a conserved signature, figure out how to color it.
                    } elseif {$signatureType == 2} {
                        set element [::SeqData::getElement $sequenceID $elementIndex]
                        if {$element == $signatureElement} {
                            seq set color $sequenceID $elementIndex [$colorNameMap iceblue]
                        } else {
                            seq set color $sequenceID $elementIndex [$colorNameMap white]
                        }
                        
                    # If this is a untrusted signature, figure out how to color it.
                    } elseif {$signatureType == 3} {
                        set element [::SeqData::getElement $sequenceID $elementIndex]
                        if {$element == $signatureElement} {
                            seq set color $sequenceID $elementIndex [$colorNameMap pink]
                        } else {
                            seq set color $sequenceID $elementIndex [$colorNameMap red]
                        }
                    }
                }
            }
        }
    }
    
    proc calculateSignatures {sequenceGroups {groupConsensusCutoff 0.9} {otherGroupMaxCutoff 0.05} {otherGroupMaxGapFraction 0.5} {maxConservedBlockDistance 10} {minConservedBlockSize 2}} {
        
        # Make sure the groups are valid.
        set ret [validateGroups $sequenceGroups]
        set sequenceGroups [lindex $ret 0]
        set numberSequences [lindex $ret 1]
        set alignmentLength [lindex $ret 2]
        
        array unset signatures 
        for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
            set signatures($groupIndex) {}
        }
        
        # Get the counts for the sequence groups.
        getGroupCounts $sequenceGroups $alignmentLength
        
        # Go through the alignment.
        for {set elementIndex 0} {$elementIndex < $alignmentLength} {incr elementIndex} {
            
            # Get the group signatures for this column.
            set elementSignatures [calculateElementSignatures $sequenceGroups $elementIndex $groupConsensusCutoff $otherGroupMaxCutoff $otherGroupMaxGapFraction]
        
            # Go through the groups.
            for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
                
                # Get the signature for this group.
                lappend signatures($groupIndex) [lindex $elementSignatures $groupIndex]
            }
        }
        
        # Delete the counts
        deleteGroupCounts
        
        # Perform any post processing on the signatures to make sure the are real signatures.
        for {set elementIndex 0} {$elementIndex < $alignmentLength} {incr elementIndex} {
            for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
                
                # Get the signature.
                set groupSignature $signatures($groupIndex)
                
                # If this is a group signature.
                set groupSignatureElement [lindex $groupSignature $elementIndex]
                if {[string length $groupSignatureElement] == 1} {
                    
                    # If necessary, make sure that we have a non-gap path to a conserved block within the window.
                    if {$minConservedBlockSize > 0} {
                        
                        # Find the nearest conserved block one either side.
                        set nearestLeftBlockIndex [findNearestConservedBlock $groupSignature $elementIndex $maxConservedBlockDistance $minConservedBlockSize "left"]
                        set nearestRightBlockIndex [findNearestConservedBlock $groupSignature $elementIndex $maxConservedBlockDistance $minConservedBlockSize "right"]
                        
                        # Make sure there is a consensus path to one of the blocks in each group.
                        for {set groupToCheckIndex 0} {$groupToCheckIndex < [llength $sequenceGroups]} {incr groupToCheckIndex} {
                            set foundLeftPathCount 0
                            set foundRightPathCount 0
                            foreach sequenceID [lindex $sequenceGroups $groupToCheckIndex] {
                                if {$nearestLeftBlockIndex != -1 && ![hasGapBetweenElements $sequenceID $nearestLeftBlockIndex $elementIndex]} {
                                    incr foundLeftPathCount
                                }
                                if {$nearestRightBlockIndex != -1 && ![hasGapBetweenElements $sequenceID $elementIndex $nearestRightBlockIndex]} {
                                    incr foundRightPathCount
                                }
                            }
                            
                            # If not enough sequences have a path, adjust the signature.
                            if {[expr double($foundLeftPathCount)/double([llength [lindex $sequenceGroups $groupToCheckIndex]])] < $otherGroupMaxGapFraction && [expr double($foundRightPathCount)/double([llength [lindex $sequenceGroups $groupToCheckIndex]])] < $otherGroupMaxGapFraction} {
                                set signatures($groupIndex) [lreplace $signatures($groupIndex) $elementIndex $elementIndex "$groupSignatureElement?"]
                                break
                            }
                        }
                    }                       
                }
            }
        }
        
        # Return the signatures.
        set fullSignatures {}
        for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
            lappend fullSignatures $signatures($groupIndex)
        }
        
        return $fullSignatures
    }
    
    proc findNearestConservedBlock {signature elementIndex maxConservedBlockDistance minConservedBlockSize side} {
        
        set conservedBlockSize 0
        set conservedBlockIndex -1
        if {$side == "left"} {
            for {set checkElementIndex [expr $elementIndex-1]} {$checkElementIndex >= 0 && $checkElementIndex >= [expr $elementIndex-$maxConservedBlockDistance]} {incr checkElementIndex -1} {
                
                
                # See if the element is conserved.
                set signatureElement [lindex $signature $checkElementIndex]
                if {[string length $signatureElement] == 2 && [string index $signatureElement 1] == "!"} {
                    incr conservedBlockSize
                    if {$conservedBlockSize == 1} {
                        set conservedBlockIndex $checkElementIndex
                    }
                    if {$conservedBlockSize == $minConservedBlockSize} {
                        return $conservedBlockIndex
                    }
                } else {
                    set conservedBlockSize 0
                    set conservedBlockIndex -1
                }
            }
        } elseif {$side == "right"} {
            for {set checkElementIndex [expr $elementIndex+1]} {$checkElementIndex < [llength $signature] && $checkElementIndex <= [expr $elementIndex+$maxConservedBlockDistance]} {incr checkElementIndex 1} {
                
                # See if the element is conserved.
                set signatureElement [lindex $signature $checkElementIndex]
                if {[string length $signatureElement] == 2 && [string index $signatureElement 1] == "!"} {
                    incr conservedBlockSize
                    if {$conservedBlockSize == 1} {
                        set conservedBlockIndex $checkElementIndex
                    }
                    if {$conservedBlockSize == $minConservedBlockSize} {
                        return $conservedBlockIndex
                    }
                } else {
                    set conservedBlockSize 0
                    set conservedBlockIndex -1
                }
            }
        }
        
        return -1
    }
    
    proc hasGapBetweenElements {sequenceID index1 index2} {
        for {set i [expr $index1+1]} {$i < $index2} {incr i} {
            if {[::SeqData::getElement $sequenceID $i] == "-"} {
                return 1
            }
        }
        return 0
    }
    
    proc validateGroups {sequenceGroups} {
	            
        # Make sure the sequences are all aligned.
        set numberSequences 0
        set alignmentLength 0
        for {set i 0} {$i < [llength $sequenceGroups]} {incr i} {
            
            # If this group is empty, remove it.
            if {[llength [lindex $sequenceGroups $i]] == 0} {
                set sequenceGroups [lreplace $sequenceGroups $i $i]
                incr i -1
                
            } else {
                
                # Go through each sequence in the group.
                foreach sequenceID [lindex $sequenceGroups $i] {
                
                    incr numberSequences
                    
                    # If we don't yet know the alignment length, get it.
                    if {$alignmentLength == 0} {
                        set alignmentLength [::SeqData::getSeqLength [lindex [lindex $sequenceGroups $i] 0]]
                        
                    # If the sequence has the wrong length, throw an error.
                    } elseif {[::SeqData::getSeqLength $sequenceID] != $alignmentLength} {
                        error "Signatures can only be constructed for aligned sequences."
                    }    
                }
            }
        }
        
        return [list $sequenceGroups $numberSequences $alignmentLength]
    }
    
    proc getGroupCounts {sequenceGroups alignmentLength} {
        
        variable counts
        
        # Initialize the count array.
        deleteGroupCounts
        array unset counts 
        
        # Get the counts for each element.
        for {set elementIndex 0} {$elementIndex < $alignmentLength} {incr elementIndex} {
            getGroupElementCounts $sequenceGroups $elementIndex
        }
    }
    
    proc deleteGroupCounts {} {
        
        variable counts
        
        # Delete the counts
        if {[info exists counts]} {
            unset counts
        }
    }
    
    proc getGroupElementCounts {sequenceGroups elementIndex} {
        
        variable counts
        
        # Go through each group to count the elements.
        for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {

            # Count the different elements at this position.
            set counts($elementIndex,$groupIndex,elementValues) {}
            foreach sequenceID [lindex $sequenceGroups $groupIndex] {
                set element [::SeqData::getElement $sequenceID $elementIndex]
            
                 # Increase the count entry for this element.
                 if {![info exists counts($elementIndex,$groupIndex,$element)]} {
                     if {$element != "-"} {
                         lappend counts($elementIndex,$groupIndex,elementValues) $element
                     }
                     set counts($elementIndex,$groupIndex,$element) 1
                 } else {
                     incr counts($elementIndex,$groupIndex,$element)
                 }
            }
        }
    }
    
    proc calculateElementSignatures {sequenceGroups elementIndex {groupConsensusCutoff 0.9} {otherGroupMaxCutoff 0.1} {otherGroupMaxGapFraction 0.5}} {
                
        variable counts
        
        # Get the consensus element for each group.
        set consensuses [getGroupConsensuses $sequenceGroups $elementIndex $groupConsensusCutoff]
        
        # Figure out some things about the consensuses.
        set allSame 1
        set allHaveConsensus 1
        for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
            if {$allSame && ([lindex $consensuses $groupIndex] == "" || [lindex $consensuses $groupIndex] != [lindex $consensuses 0])} {
                set allSame 0 
            }
            if {$allHaveConsensus && [lindex $consensuses $groupIndex] == ""} {
                set allHaveConsensus 0
            }
        }
        
        # Figure out the signatures.
        set signatures {}
        for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
            
            # If there is no consensus for this group, there is no signature.
            if {[lindex $consensuses $groupIndex] == ""} {
                lappend signatures ""
                
            # If all of the other groups have the same consensus, it is a conserved signature.
            } elseif {$allSame} {
                lappend signatures "[lindex $consensuses $groupIndex]!"
            
            # If all of the other groups have a signature and at least one is different, this is a signature.
            } elseif {$allHaveConsensus} {
                
                set isSignature 0
                
                # See if we need to check the composition even for signatures.
                if {$otherGroupMaxCutoff <= [expr 1.0-$groupConsensusCutoff]} {
                    set groupConsensusElement [lindex $consensuses $groupIndex]
                    for {set otherGroupIndex 0} {$otherGroupIndex < [llength $sequenceGroups]} {incr otherGroupIndex} {
                        
                        # Check the composition of every group that doesn't have the same signature.
                        if {[lindex $consensuses $otherGroupIndex] != $groupConsensusElement} {
                            
                            # If this groups has the signature element below the lower cutoff, this is a signature.
                            if {(![info exists counts($elementIndex,$otherGroupIndex,$groupConsensusElement)] || [expr double($counts($elementIndex,$otherGroupIndex,$groupConsensusElement))/double([llength [lindex $sequenceGroups $otherGroupIndex]])] < $otherGroupMaxCutoff)} {
                                set isSignature 1
                                break
                            }
                        }
                    }
                } else {
                    set isSignature 1
                }
                
                if {$isSignature} {
                    lappend signatures "[lindex $consensuses $groupIndex]"
                } else {
                    lappend signatures ""
                }
                
            # Otherwise, not all of the other groups have a signature so we have to check the composition.
            } else {
                
                set groupConsensusElement [lindex $consensuses $groupIndex]

                # If any of the other groups have too many gaps, this is not a signature.
                set otherGroupGaps 0
                for {set otherGroupIndex 0} {$otherGroupIndex < [llength $sequenceGroups]} {incr otherGroupIndex} {
                    if {$otherGroupIndex != $groupIndex && [info exists counts($elementIndex,$otherGroupIndex,-)] && [expr double($counts($elementIndex,$otherGroupIndex,-))/double([llength [lindex $sequenceGroups $otherGroupIndex]])] > $otherGroupMaxGapFraction} {
                        set otherGroupGaps 1
                        break
                    }
                }
                
                set isSignature 0
                for {set otherGroupIndex 0} {!$otherGroupGaps && $otherGroupIndex < [llength $sequenceGroups]} {incr otherGroupIndex} {
                    
                    # Check every other group except ourselves.
                    if {$otherGroupIndex != $groupIndex} {
                        
                        # If any of the other groups have this element below the lower cutoff, this is a signature (as long as it is not a gap column).
                        if {(![info exists counts($elementIndex,$otherGroupIndex,$groupConsensusElement)] || [expr double($counts($elementIndex,$otherGroupIndex,$groupConsensusElement))/double([llength [lindex $sequenceGroups $otherGroupIndex]])] < $otherGroupMaxCutoff)} {
                            set isSignature 1
                            break
                        }
                    }
                }                

                if {$isSignature} {                
                    lappend signatures $groupConsensusElement
                } else {
                    lappend signatures ""
                }
            }
        }
        
        return $signatures
    }
  
    proc getGroupConsensuses {sequenceGroups elementIndex groupConsensusCutoff} {
        
        variable counts
        
        # Go through each group to determine the signatures.
        set consensusElements {}
        for {set groupIndex 0} {$groupIndex < [llength $sequenceGroups]} {incr groupIndex} {
            
            # Figure out which element is the consensus of the group.
            set consensusElement ""
            set numberSequences [llength [lindex $sequenceGroups $groupIndex]]
            foreach element $counts($elementIndex,$groupIndex,elementValues) {
                if {double($counts($elementIndex,$groupIndex,$element))/double($numberSequences) >= $groupConsensusCutoff && ($consensusElement == "" || $counts($elementIndex,$groupIndex,$element) > $counts($elementIndex,$groupIndex,$consensusElement))} {
                    set consensusElement $element
                }
            }
            
            lappend consensusElements $consensusElement
        }
        
        return $consensusElements
    }
    
    # Dialog management variables.
    variable w
    variable oldFocus
    variable oldGrab
    variable grabStatus
    
    # Variable for indicating the user is finished choosing the options.
    variable finished
    
    # The options.
    variable options
    array set options {groups {} groupConsensusCutoff 0.9 otherGroupMaxCutoff 0.05 otherGroupMaxGapFraction 0.5 maxConservedBlockDistance 10 minConservedBlockSize 2}
    
    proc showSignatureOptionsDialog {parent groups} {
    
        variable w
        variable oldFocus
        variable oldGrab
        variable grabStatus
        variable finished
        variable options
        set finished 0
    
        # Create a new top level window.
        set w [createModalDialog ".signatureoptions" "Signature Calculation Options"]
        
        # Create the components.
        frame $w.center
            label $w.center.label1 -text "Select the groups to be used in the signature calculation:"
            frame $w.center.g1
                listbox $w.center.g1.groups -selectmode multiple -exportselection FALSE -height 10 -width 50 -yscrollcommand "$w.center.g1.scroll set"
                scrollbar $w.center.g1.scroll -command "$w.center.g1.groups yview"
                set selectedGroupIndices {}
                for {set i 0} {$i < [llength $groups]} {incr i} {
                    set group [lindex $groups $i]
                    $w.center.g1.groups insert end $group
                    if {$options(groups) == {} || [lsearch -exact $options(groups) $group] != -1} {
                        lappend selectedGroupIndices $i
                    }
                }
                foreach selectedGroupIndex $selectedGroupIndices {
                    $w.center.g1.groups selection set $selectedGroupIndex
                }
            label $w.center.groupConsensusCutoffL -text "Minimum fraction conserved to be a group signature:"
            entry $w.center.groupConsensusCutoff -textvariable "::SeqEdit::Metric::Signatures::options(groupConsensusCutoff)" -width 6
            label $w.center.otherGroupMaxCutoffL -text "Maximum fraction of signature allowed in other groups:"
            entry $w.center.otherGroupMaxCutoff -textvariable "::SeqEdit::Metric::Signatures::options(otherGroupMaxCutoff)" -width 6
            label $w.center.otherGroupMaxGapFractionL -text "Maximum fraction of gaps allowed in other groups:"
            entry $w.center.otherGroupMaxGapFraction -textvariable "::SeqEdit::Metric::Signatures::options(otherGroupMaxGapFraction)" -width 6            
            label $w.center.maxConservedBlockDistanceL -text "Maximum distance of signature from a conserved block:"
            entry $w.center.maxConservedBlockDistance -textvariable "::SeqEdit::Metric::Signatures::options(maxConservedBlockDistance)" -width 6            
            label $w.center.minConservedBlockSizeL -text "Minimum length of conserved block:"
            entry $w.center.minConservedBlockSize -textvariable "::SeqEdit::Metric::Signatures::options(minConservedBlockSize)" -width 6            
        frame $w.bottom
            frame $w.bottom.buttons
                button $w.bottom.buttons.accept -text "OK" -pady 2 -command "::SeqEdit::Metric::Signatures::but_ok"
                button $w.bottom.buttons.cancel -text "Cancel" -pady 2 -command "::SeqEdit::Metric::Signatures::but_cancel"
                bind $w <Return> {::SeqEdit::Metric::Signatures::but_ok}
                bind $w <Escape> {::SeqEdit::Metric::Signatures::but_cancel}
        
        # Layout the components.
        pack $w.center                  -fill both -expand true -side top -padx 5 -pady 5
        grid $w.center.label1           -column 1 -row 1 -sticky w -columnspan 2
        grid $w.center.g1               -column 1 -row 2 -sticky w -columnspan 2
        pack $w.center.g1.groups        -fill both -expand true -side left -padx 5 -pady 5
        pack $w.center.g1.scroll        -side right -fill y
        grid $w.center.groupConsensusCutoffL    -column 1 -row 3 -sticky w
        grid $w.center.groupConsensusCutoff     -column 2 -row 3 -sticky w -padx 5
        grid $w.center.otherGroupMaxCutoffL     -column 1 -row 4 -sticky w
        grid $w.center.otherGroupMaxCutoff      -column 2 -row 4 -sticky w -padx 5
        grid $w.center.otherGroupMaxGapFractionL    -column 1 -row 5 -sticky w
        grid $w.center.otherGroupMaxGapFraction     -column 2 -row 5 -sticky w -padx 5
        grid $w.center.maxConservedBlockDistanceL   -column 1 -row 6 -sticky w
        grid $w.center.maxConservedBlockDistance    -column 2 -row 6 -sticky w -padx 5
        grid $w.center.minConservedBlockSizeL       -column 1 -row 7 -sticky w
        grid $w.center.minConservedBlockSize        -column 2 -row 7 -sticky w -padx 5
        
        pack $w.bottom                  -fill x -side bottom
        pack $w.bottom.buttons          -side bottom
        pack $w.bottom.buttons.accept   -side left -padx 5 -pady 5
        pack $w.bottom.buttons.cancel   -side right -padx 5 -pady 5

        # Bind the window closing event.
        bind $w <Destroy> {::SeqEdit::Metric::Signatures::but_cancel}
        
        # Center the dialog.
        centerDialog $parent
        
        # Wait for the user to interact with the dialog.
        tkwait variable ::SeqEdit::Metric::Signatures::finished
        #puts "Size is [winfo reqwidth $w] [winfo reqheight $w]"

        # Destroy the dialog.
        destroyDialog        
        
        # Return the options.
        if {$finished == 1} {
            return [array get options]
        } else {
            return {}
        }
    }
    
    # Creates a new modal dialog window given a prefix for the window name and a title for the dialog.
    # args:     prefix - The prefix for the window name of this dialog. This should start with a ".".
    #           dialogTitle - The title for the dialog.
    # return:   The name of the newly created dialog.
    proc createModalDialog {prefix dialogTitle} {

        variable w
        variable oldFocus
        variable oldGrab
        variable grabStatus
        
        # Find a name for the dialog
        set unique 0
        set childList [winfo children .]
        while {[lsearch $childList $prefix$unique] != -1} {
            incr unique
        }

        # Create the dialog.        
        set w [toplevel $prefix$unique]
        
        # Set the dialog title.
        wm title $w $dialogTitle
        
        # Make the dialog modal.
        set oldFocus [focus]
        set oldGrab [grab current $w]
        if {$oldGrab != ""} {
            set grabStatus [grab status $oldGrab]
        }
        grab $w
        focus $w
        
        return $w
    }
    
    # Centers the dialog.
    proc centerDialog {{parent ""}} {
        
        variable w
        
        # Set the width and height, since calculating doesn't work properly.
        set width 388
        set height [expr 292+22]
        
        # Figure out the x and y position.
        if {$parent != ""} {
            set cx [expr {int ([winfo rootx $parent] + [winfo width $parent] / 2)}]
            set cy [expr {int ([winfo rooty $parent] + [winfo height $parent] / 2)}]
            set x [expr {$cx - int ($width / 2)}]
            set y [expr {$cy - int ($height / 2)}]
            
        } else {
            set x [expr {int (([winfo screenwidth $w] - [winfo reqwidth $w]) / 2)}]
            set y [expr {int (([winfo screenheight $w] - [winfo reqheight $w]) / 2)}]
        }
        
        # Make sure we are within the screen bounds.
        if {$x < 0} {
            set x 0
        } elseif {[expr $x+$width] > [winfo screenwidth $w]} {
            set x [expr [winfo screenwidth $w]-$width]
        }
        if {$y < 22} {
            set y 22
        } elseif {[expr $y+$height] > [winfo screenheight $w]} {
            set y [expr [winfo screenheight $w]-$height]
        }
            
        wm geometry $w +${x}+${y}
        wm positionfrom $w user
    }
    
    # Destroys the dialog. This method releases the dialog resources and restores the system handlers.
    proc destroyDialog {} {
        
        variable w
        variable oldFocus
        variable oldGrab
        variable grabStatus
        
        # Destroy the dialog.
        catch {focus $oldFocus}
        catch {
            bind $w <Destroy> {}
            destroy $w
        }
        if {$oldGrab != ""} {
            if {$grabStatus == "global"} {
                grab -global $oldGrab
            } else {
                grab $oldGrab
            }
        }
    }
    
    proc but_ok {} {
    
        variable w
        variable finished
        variable options

        # Save the options.        
        set selectedGroupIndices [$w.center.g1.groups curselection]
        if {[llength $selectedGroupIndices] < 2} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "You must select at least two groups for a signature calculation."
            return
        } else {
            set options(groups) {}
            foreach selectedGroupIndex $selectedGroupIndices {
                lappend options(groups) [$w.center.g1.groups get $selectedGroupIndex]
            }
        }

        # Close the window.
        set finished 1
    }
    
    proc but_cancel {} {
    
        variable finished
    
        # Close the window.    
        set finished 0
    }  
}
