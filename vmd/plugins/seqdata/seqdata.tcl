############################################################################
#cr
#cr            (C) Copyright 1995-2004 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: seqdata.tcl,v 1.13 2013/04/15 17:36:40 johns Exp $
#

# Sequence data structures and procedures for use in VMD sequence manipulation 
# plugins.

package provide seqdata 1.1

package require libbiokit 1.1

# Declare global variables for this package.
namespace eval ::SeqData:: {

    # Export the package namespace.
    namespace export SeqEditWidget
    
    # An array of the data for the sequences.
    variable seqs
#    array unset seqs 
    
    # A map of the notifications commands.
    variable sequenceCommands
#    array unset sequenceCommands 
  
  # The file were sequence annotations are stored.
  variable annotationsFile "$env(HOME)/.ms-annotations"

  # Type of backend we're using for annotations.  
  # Could be a file, database, ???
  variable backend "FILE"
  
variable i 0

}

# Resets the sequence data store.
proc ::SeqData::reset { } {

    variable seqs
    variable sequenceCommands
    
    # Reset the data structures.
    seq reset
    array unset seqs 
    array unset sequenceCommands 
}

# Adds a sequence to the current sequence store.
# args:     seq - A list of the elements in the sequence to be added.
#           name - The name of the sequence.
#           hasStruct - Y if this sequence has a structure associated with it 
#                       in VMD, otherwise N.
#           firstResidue - The id of the first element of the sequence in 
#                            the real world. For example, if a protein sequence
#                            is missing the first 6 residues, this value should
#                            be set to 6. This will aid in displaying 
#                            meaningful sequence identities.
# return:   The id of the newly added sequence.
proc ::SeqData::addSeq {seq name {hasStruct "N"} {firstResidue 1} {sources {}} {type unknown}} {
#   puts "seqdata.tcl.addSeq seq: $seq, name: $name, hasStruct: $hasStruct, fRes: $firstResidue, sources: $sources, type: $type"
    variable seqs
    variable i
    
    # Get the next sequence id and add it to the list of used ids.
    set seqNum [seq new $seq]
#    puts "seqdata.tcl.addSeq after saving: [seq get $seqNum]"

    if {$seqNum % 100 == 0 } {
       puts -nonewline stderr "$seqNum..."
       flush stderr
       if {$seqNum % 1000 == 0 } {
          puts stderr "sequences loaded"
       }

    }

    # Set the sequence data.
    set seqs($seqNum) 1
    set seqs($seqNum,ss) {}   ;# secondary structure
#    puts "setting name for seq $seqNum to '$name'"
    setName $seqNum $name
#    set seqs($seqNum,name) $name
    set seqs($seqNum,hasStruct) $hasStruct
    set seqs($seqNum,firstRes) $firstResidue
    set seqs($seqNum,annot) {}   ;# annotations
    set seqs($seqNum,annot,name) $name
    set seqs($seqNum,annot_curr) 0
#    set seqs($seqNum,type) $type
#    setType $seqNum [seq type $seqNum]

    
    set seqs($seqNum,sources) $sources
    # Try to guess any sources we are mising.
    fillInMissingSources $seqNum $name
    
    # Figure out this sequences taxonomy node, if we can.
    set seqs($seqNum,taxNode) [findTaxonomyNode $seqNum]
    set seqs($seqNum,ecNumber) [findEnzymeCommissionNumber $seqNum]
  
    # Return the sequnce id.
#    puts "size is still [seq length $seqNum]"
    return $seqNum
}

# ------------------------------------------------------------------
# This method allows the caller to specify handlers to receive notification events whenever the
# specified sequence is changed.
# args:     sequenceID - The id of the sequence to monitor.
#           notificationCommand - The command to execute whenever the sequence is changed. 
proc ::SeqData::setCommand {commandName sequenceID newCommand} {
    
    variable sequenceCommands
    
    set sequenceCommands($commandName,$sequenceID) $newCommand
}
# ------------------------------------------------------------------
proc ::SeqData::duplicateSequence {sequenceID {startingElement 0} {endingElement end}} {
    
    variable seqs
    variable sequenceCommands
    
    # Figure out the sequence data.
    if {$startingElement == "" || $startingElement == -1} {
        set sequence {}
        set startingElement 0
    } else {
        set sequence [seq get $sequenceID $startingElement $endingElement]
    }
    
    # Get the next sequence id and add it to the list of used ids.
    set newSequenceID [seq new $sequence]
    set seqs($newSequenceID) 1    
    set seqs($newSequenceID,ss) {}
    
    # Figure out the first residue index.
    set seqs($newSequenceID,firstRes) 1
    for {set i 0} {$i <= [llength $sequence]} {incr i} {
        set firstResidue [getResidueForElement $sequenceID [expr $i+$startingElement]]
        if {$firstResidue != -1} {
            set seqs($newSequenceID,firstRes) $firstResidue
            break
        }
    }
    
    # Copy the other attributes.
    copyAttributes $sequenceID $newSequenceID
    
    return $newSequenceID
}

# ------------------------------------------------------------------
proc ::SeqData::duplicateSequences {sequenceIDs} {
    set duplicates {}
    foreach sequenceID $sequenceIDs {
        lappend duplicates [::SeqData::duplicateSequence $sequenceID]
    }
    return $duplicates
}

# ------------------------------------------------------------------
proc ::SeqData::deleteSequences {sequenceIDs} {
    
    variable seqs
    variable sequenceCommands
    
    # Go through each sequence and delete it.
    foreach sequenceID $sequenceIDs {
        
        # Remove it from the sequence list.
        seq delete $sequenceID
        
        # Remove it from the sequence data store.
        foreach keyName [array names seqs "$sequenceID,*"] {
            unset seqs($keyName)
        }
        
        # Remove any commands associated with it.
        foreach keyName [array names sequenceCommands "*,$sequenceID"] {
            unset sequenceCommands($keyName)
        }
    }
}
# ------------------------------------------------------------------------
proc ::SeqData::copyAttributes {oldSequenceID newSequenceID} {
    
    variable seqs
    variable sequenceCommands
        
    # Copy the other attributes.
#    puts "copyAttributes: setting id $newSequenceID name to oldid $oldSequenceID '[getName $oldSequenceID]'"
    setName $newSequenceID [getName $oldSequenceID]
#    set seqs($newSequenceID,name) $seqs($oldSequenceID,name)
    set seqs($newSequenceID,hasStruct) "N"
    set seqs($newSequenceID,sources) $seqs($oldSequenceID,sources)
    set seqs($newSequenceID,taxNode) $seqs($oldSequenceID,taxNode)
    set seqs($newSequenceID,ecNumber) $seqs($oldSequenceID,ecNumber)
    set seqs($newSequenceID,annot_curr) $seqs($oldSequenceID,annot_curr)
#    set seqs($newSequenceID,type) $seqs($oldSequenceID,type)
    
    # If we have a notification command, call it.
    if {[info exists sequenceCommands(copyAttributes,$oldSequenceID)]} {
        $sequenceCommands(copyAttributes,$oldSequenceID) $oldSequenceID $newSequenceID
    }
        
    # Copy the annotations.
    copyAnnotations $oldSequenceID $newSequenceID
}

# --------------------------------------------------------------------------
proc ::SeqData::extractRegionsFromSequences {regions} {
    
    # Go through each region.
    set originalSequenceIDs {}
    set newSequenceIDs {}
    set prefixes {}
    set suffixes {}
    set prefixEndPositions {}
    set suffixStartPositions {}
    for {set i 0} {$i < [llength $regions]} {incr i 2} {

        # Get the sequence id and elements of the region.
        set regionSequenceID [lindex $regions $i]
        set regionPositions [lindex $regions [expr $i+1]]
        lappend originalSequenceIDs $regionSequenceID
        
        # Make sure that the region is continguous.
        set previousElement -1
        foreach element $regionPositions {
            
            # If one region wasn't contig, we can't process anything, so return.
            if {$previousElement != -1 && $element != [expr $previousElement+1]} {
                return {}
            }
            set previousElement $element
        }
        
        # Create a new sequence from the region.
        lappend newSequenceIDs [::SeqData::duplicateSequence $regionSequenceID [lindex $regionPositions 0] [lindex $regionPositions end]]
        
        # Get the sequence.
        set fullSequence [::SeqData::getSeq $regionSequenceID]
        
        # Save the portion of the sequence before the region.
        set prefix {}
        set prefixEndPosition [expr [lindex $regionPositions 0]-1]
        if {$prefixEndPosition >= 0} {
            set prefix [lrange $fullSequence 0 $prefixEndPosition]
        }
        lappend prefixes $prefix
        lappend prefixEndPositions $prefixEndPosition
        
        # Save the portion of the sequence after the region.
        set suffix {}
        set suffixStartPosition [expr [lindex $regionPositions end]+1]
        if {$suffixStartPosition < [::SeqData::getSeqLength $regionSequenceID]} {
            set suffix [lrange $fullSequence $suffixStartPosition end]
        }
        lappend suffixes $suffix
        lappend suffixStartPositions $suffixStartPosition
    }
    
    return [list $originalSequenceIDs $newSequenceIDs $prefixes \
                        $prefixEndPositions $suffixes $suffixStartPositions]
}
        
# --------------------------------------------------------------------------
proc ::SeqData::fillInMissingSources {sequenceID name} {
    
    # If we don't have a SwissProt source, see if we have a SwissProt name.
    if {[getSourceData $sequenceID "sp"] == {}} {
        if {[::SeqData::SwissProt::isValidSwissProtName $name]} {
            setSourceData $sequenceID "sp" [list "*" $name]
        }
    }
    
    # If we don't have a PDB source, see if we have a PDB name.
    if {[getSourceData $sequenceID "pdb"] == {}} {
        if {[set pdbCode [::SeqData::PDB::isValidPDBName $name]] != ""} {
            setSourceData $sequenceID "pdb" [list $pdbCode "*"]
        }
    }
    
    # If we don't have a SCOP source, see if we have a SCOP name.
    if {[getSourceData $sequenceID "scop"] == {}} {
        if {[::SeqData::SCOP::isValidSCOPName $name]} {
            setSourceData $sequenceID "scop" [list $name]
        }
    }
    
    # Fill in any data we can from known good sources.
    if {[llength [getSourceData $sequenceID "pdb"]] == 2 && [set swisssProtName [::SeqData::PDB::getSwissProtName [lindex [getSourceData $sequenceID "pdb"] 0]]] != ""} {
        setSourceData $sequenceID "sp" [list "*" $swisssProtName]
    } elseif {[llength [getSourceData $sequenceID "scop"]] == 1 && [set swisssProtName [::SeqData::SCOP::getSwissProtName [lindex [getSourceData $sequenceID "scop"] 0]]] != ""} {
        setSourceData $sequenceID "sp" [list "*" $swisssProtName]
    } elseif {[llength [getSourceData $sequenceID "sp"]] == 2 && [set pdbCode [::SeqData::PDB::getPdbCodeForSwissProtName [lindex [getSourceData $sequenceID "sp"] 1]]] != ""} {
        setSourceData $sequenceID "pdb" [list $pdbCode "*"]
    }
}

# ------------------------------------------------------------------------
# args:     seqNum - The id of the sequence to lookup.
# return:   A list containing the organism's lineage or an empty list if it is unknown.
proc ::SeqData::findTaxonomyNode {sequenceID} {

    set taxonomyNode ""

#    puts "seqdata.tcl.findTaxonomyNode. seqID: $sequenceID, sp: '[getSourceData $sequenceID sp]' gi: '[getSourceData $sequenceID gi]' tax: '[getSourceData $sequenceID tax]'"

    # See if it is in the Swiss Prot database.
    if {[llength [getSourceData $sequenceID "sp"]] == 2} {
        set taxonomyNode [::SeqData::SwissProt::getTaxonomyNode [lindex [getSourceData $sequenceID "sp"] 1]]
    
    # See if it is in the GenBank database.
    } elseif {[llength [getSourceData $sequenceID "gi"]] == 1} {    
        set taxonomyNode [::SeqData::GenBank::getTaxonomyNode [lindex [getSourceData $sequenceID "gi"] 0]]
    
    # See if we have a taxonomy identifier.
    } elseif {[llength [getSourceData $sequenceID "tax"]] == 1} {    
        set taxonomyNode [lindex [getSourceData $sequenceID "tax"] 0]
    
    # Otherwise, see if the name is a species name.
    } else {    
        set taxonomyNode [::SeqData::Taxonomy::findNodeBySpecies [getName $sequenceID]]
    }
    
    return $taxonomyNode
}

# ------------------------------------------------------------------------
proc ::SeqData::findEnzymeCommissionNumber {sequenceID} {

    set ecNumber ""
    
    # See if it is in the Swiss Prot database.
    if {[llength [getSourceData $sequenceID "sp"]] == 2} {
        set ecNumber [::SeqData::SwissProt::getEnzymeCommissionNumber [lindex [getSourceData $sequenceID "sp"] 1]]
    }
    
    return $ecNumber
}

# ------------------------------------------------------------------------
# Get the sequence from the sequence store.
# args:     seqNum - The id of the sequence to retrieve.
# return:   The list of the elements in the sequence.
proc ::SeqData::getSeq { sequenceID } {
    variable seqs
    if {[info exists seqs($sequenceID)]} {
        return [seq get $sequenceID]
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# ------------------------------------------------------------------
# Get the length of a sequence from the sequence store.
# args:     seqNum - The id of the sequence length to retrieve.
# return:   The length of the sequence.
proc ::SeqData::getSeqLength { sequenceID } {
    variable seqs
    if {[info exists seqs($sequenceID)]} {
        return [seq length $sequenceID]
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::setSeq {sequenceID sequence} {

    variable seqs
    variable sequenceCommands
    
    # Get the sequence.
    if {[info exists seqs($sequenceID)]} {
        
        # If we have an override behavior, call it.
        if {[info exists sequenceCommands(changeHandler,$sequenceID)]} {
             return [$sequenceCommands(changeHandler,$sequenceID) "setSeq" $sequenceID $sequence]
             
        # Otherwise just use the default behavior.
        } else {
            seq set $sequenceID $sequence
            if {[llength $seqs($sequenceID,ss)] != [getResidueCount $sequenceID]} {
                set seqs($sequenceID,ss) {}
            }
            
            return 1
        }
        
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getSecondaryStructure {sequenceID {includeGaps 0}} {
    
    variable seqs
    variable sequenceCommands
    
    if {[info exists seqs($sequenceID)]} {
        
        # If the secondary structure has not been calculated, calculate it.
        if {$seqs($sequenceID,ss) == {}} {
        
           # if sequence is a protein
           if {[::SeqData::getType $sequenceID] == "protein"} {

              # See if we have a calculation command for this sequence.
              if {[info exists sequenceCommands(secondaryStructureCalculation,$sequenceID)]} {
                
                  # Call the command and save the secondary structure.
                  setSecondaryStructure $sequenceID [$sequenceCommands(secondaryStructureCalculation,$sequenceID) $sequenceID]
                 
              # Otherwise assume the sequence is all coil.
              } else {
                  set ss {}
                  for {set i 0} {$i < [getResidueCount $sequenceID]} {incr i} {
                      lappend ss "C"
                  }
                  setSecondaryStructure $sequenceID $ss 
              }
           } elseif {[::SeqData::getType $sequenceID] == "rna" || [::SeqData::getType $sequenceID] == "dna"} {
              set ss {}
              for {set i 0} {$i < [getResidueCount $sequenceID]} {incr i} {
                 lappend ss "."
              }
              setSecondaryStructure $sequenceID $ss 
           }
        }
        
        if {$seqs($sequenceID,ss) != {}} {
            
            # If we need gaps, add them.
            if {$includeGaps} {
                set gappedSecondaryStructure {}
                for {set elementIndex 0; set residueIndex 0} {$elementIndex < [getSeqLength $sequenceID]} {incr elementIndex} {
                    
                    # See if the element is a gap.
                    if {[getElement $sequenceID $elementIndex] == "-"} {
                        lappend gappedSecondaryStructure "-"
                    } else {
                        lappend gappedSecondaryStructure [lindex $seqs($sequenceID,ss) $residueIndex]
                        incr residueIndex
                    }
                }
                return $gappedSecondaryStructure
            } else {
                return $seqs($sequenceID,ss)
            }
        }
        return {}
        
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::loadSecondaryStructure {sequenceID ssfilename} {
    
   variable seqs
   variable sequenceCommands
   set ss {}   
   if {[info exists seqs($sequenceID)]} {
        
      set ss {}
      # See if we have a file to load.
      if {$ssfilename != {}} {
         # read the file.
         set fp [open $ssfilename r]
         while {![eof $fp]} {
            set line [gets $fp]
            set ss [concat $ss [split $line {}]]
         }
         setSecondaryStructure $sequenceID $ss
         close $fp
            
      } else {
         # Otherwise assume the sequence is all coil.
         # seriously? (kv)
         for {set i 0} {$i < [getResidueCount $sequenceID]} {incr i} {
            lappend ss "."
         }
         setSecondaryStructure $sequenceID $ss 
      }
        
      return $ss
   } else {
      return -code error "seq $sequenceID doesn't exist"
   }
}

# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
proc ::SeqData::setSecondaryStructure {sequenceID secondaryStructure} {
    
    variable seqs
    variable sequenceCommands
    
    if {[info exists seqs($sequenceID)]} {
        
        if {[llength $secondaryStructure] == [getResidueCount $sequenceID]} {
            set seqs($sequenceID,ss) $secondaryStructure
        }
        
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
# Removes a list of elements from a sequence.
# args: seqNum - id of seq to modify. NOTE: These MUST be in increasing order.
#       elements - list of indexes of element to be removed from sequence.
proc ::SeqData::removeElements {seqNum elementIndexes} {
 
    variable seqs
    variable sequenceCommands
#    puts "seqdata.tcl.removeElements. seqNum: $seqNum, elemIdx: $elementIndexes, seqs: [array get seqs], seqComms: [array get sequenceCommands]"

    # Make sure the sequence exists.
    if {[info exists seqs($seqNum)]} {
      
        # If we have an override behavior, call it.
        if {[info exists sequenceCommands(changeHandler,$seqNum)]} {
             return [$sequenceCommands(changeHandler,$seqNum) "removeElements" $seqNum $elementIndexes]
             
        # Otherwise just use the default behavior.
        } else {
            
            # Get the sequence.
            set sequence [seq get $seqNum]
            
            # Remove all of the elements.
            set indexesRemoved 0
            foreach elementIndex $elementIndexes {
                
                # Adjust the index to account for the previously removed elements.
                incr elementIndex -$indexesRemoved
                
                # If this element exists, remove it.
                if {$elementIndex < [llength $sequence]} {
                    set sequence [lreplace $sequence $elementIndex $elementIndex]
                    incr indexesRemoved
                }
            }
            
            # Save the new sequence.
            seq set $seqNum $sequence
                
            # Reset the secondary structure, if necessary.
            if {[llength $seqs($seqNum,ss)] != [getResidueCount $seqNum]} {
                set seqs($seqNum,ss) {}
            }
            
            return 1
        }
        
    } else {
        
        return -code error "seq $seqNum doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getElement {sequenceID elementIndex} {
    variable seqs
    if {[info exists seqs($sequenceID)]} {
        return [seq get $sequenceID $elementIndex $elementIndex]
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getElements {sequenceID elementIndexes} {
    variable seqs
    if {[info exists seqs($sequenceID)]} {
        set elements {}
        foreach elementIndex $elementIndexes {
            lappend elements [seq get $sequenceID $elementIndex $elementIndex]
        }
        return $elements
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
# Set a list of elements from a sequence to be a specific element.
# args:     seqNum - The id of the sequence to modify. NOTE: These MUST be in increasing order.
#           elements - A list of the indexes of the element that should be removed from the sequence.
#           newElement - The new element.
proc ::SeqData::setElements {seqNum elementIndexes newElement} {

    variable seqs
    variable sequenceCommands

    # Make sure the sequence exists.
    if {[info exists seqs($seqNum)]} {
      
        # If we have an override behavior, call it.
        if {[info exists sequenceCommands(changeHandler,$seqNum)]} {
             return [$sequenceCommands(changeHandler,$seqNum) "setElements" $seqNum $elementIndexes $newElement]
             
        # Otherwise just use the default behavior.
        } else {
            
            # Get the sequence.
            set sequence [seq get $seqNum]
            
            # Go through all for the elements to be changed.
            foreach elementIndex $elementIndexes {
                
                # If this element exists, replace it.
                if {$elementIndex < [llength $sequence]} {
                    set sequence [lreplace $sequence $elementIndex $elementIndex $newElement]
                }
            }
            # Save the new sequence.
            seq set $seqNum $sequence
            
            # Reset the secondary structure, if necessary.
            if {[llength $seqs($seqNum,ss)] != [getResidueCount $seqNum]} {
                set seqs($seqNum,ss) {}
            }
            
            return 1
        }
        
    } else {
        
        return -code error "seq $seqNum doesn't exist"
    }
}
# -------------------------------------------------------------------------
# Inserts a new element into a sequence.
# args:     seqNum - The id of the sequence to modify.
#           position - The position at which to insert the new elements.
#           newElement - A list of the new elements.
proc ::SeqData::insertElements {seqNum position newElements} {
    variable seqs
    variable sequenceCommands
#   puts stderr "seqdata.tcl.insertElements begin: seqNum: $seqNum, pos: $position, ne: $newElements"

    # Make sure the sequence exists.
    if {[info exists seqs($seqNum)]} {

        # If we have an override behavior, call it.
        if {[info exists sequenceCommands(changeHandler,$seqNum)]} {
#           puts "a sequence command handler existed for seqNum $seqNum $sequenceCommands(changeHandler,$seqNum) insertElements $seqNum $position $newElements"
             return [$sequenceCommands(changeHandler,$seqNum) "insertElements" $seqNum $position $newElements]

        # Otherwise just use the default behavior.
        } else {
            
            # Insert the new elements.
            if {$position == "end"} {
                set position [expr [seq length $seqNum]-1]
            }
#            puts "seqdata.tcl:::SeqData::insertElements() preparing to seq set seqnum at $seqNum, position: $position."
            if {$position != 0} {
               seq set $seqNum [concat [seq get $seqNum 0 [expr $position-1]] $newElements [seq get $seqNum $position end]]
            } else {
               seq set $seqNum [concat $newElements [seq get $seqNum $position end]]
            }
            
            # Reset the secondary structure, if necessary.
            if {[llength $seqs($seqNum,ss)] != [getResidueCount $seqNum]} {
                set seqs($seqNum,ss) {}
            }
            
            return 1
        }
        
    } else {
        
        return -code error "seq $seqNum doesn't exist"
    }
}

# -------------------------------------------------------------------------
# proc ::SeqData::removeGaps {sequenceIDs {firstElement 0} {lastElement end} {removalType all}} {
#     
#     if {$lastElement == "end"} {
#         set lastElement [expr [::SeqData::getSeqLength [lindex $sequenceIDs 0]]-1]
#     }
#     
#     # Initialize the lists of elements to remove.
#     if {[info exists elementsToRemove]} {
#         unset elementsToRemove
#     }
#     array unset elementsToRemove 
#     foreach sequenceID $sequenceIDs {
#         set elementsToRemove($sequenceID) {}
#     }
#     
#     # Remove duplicate sequence ids.
#     set newSequenceIDs {}
#     foreach sequenceID $sequenceIDs {
#         if {[lsearch -exact $newSequenceIDs $sequenceID] == -1} {
#             lappend newSequenceIDs $sequenceID
#         }
#     }
#     set sequenceIDs $newSequenceIDs
#     
#     # Figure out which gaps to remove.
#     for {set i $firstElement} {$i <= $lastElement} {incr i} {
#         
#         # See if we should remove this gap.
#         set removeGap 0
#         if {$removalType == "redundant"} {
#             set removeGap 1
#             foreach sequenceID $sequenceIDs {
#                 if {$i < [::SeqData::getSeqLength $sequenceID] && [::SeqData::getElement $sequenceID $i] != "-"} {
#                     set removeGap 0
#                     break
#                 }
#             }
#         } elseif {$removalType == "all"} {
#             set removeGap 1
#         }
#         
#         # Get the elements to remove.
#         if {$removeGap} {
#             foreach sequenceID $sequenceIDs {
#                 if {$i < [::SeqData::getSeqLength $sequenceID] && [::SeqData::getElement $sequenceID $i] == "-"} {
#                     lappend elementsToRemove($sequenceID) $i
#                 }
#             }
#         }   
#     }
#     
#     # Remove the gaps.
#     foreach sequenceID $sequenceIDs {
#         ::SeqData::removeElements $sequenceID $elementsToRemove($sequenceID)
#     }
# }
 


# ----------------------------------------------------------------------
# Get the sequence name from the sequence store.
# args:     seqNum - The id of the sequence whose name is to be retrieved.
# return:   The name of the requested sequence.
proc ::SeqData::getName { seqNum } {

   return [seq name get $seqNum]

#  variable seqs
#
#  # Get the sequence name.
#  if {[info exists seqs($seqNum)]} {
#    return $seqs($seqNum,name)
#  } else {
#    return -code error "seq $seqNum doesn't exist"
#  }
}

# -------------------------------------------------------------------------
# Sets the sequence name.
# args:     seqNum - The id of the sequence whose name is to be set.
#           newName - The new name of the sequence.
proc ::SeqData::setName {sequenceID newName} {
#   puts "setting id $sequenceID to $newName"

  seq name set $sequenceID $newName
#
#  variable seqs
#
#  # Get the sequence name.
#  if {[info exists seqs($sequenceID)]} {
#    set seqs($sequenceID,name) $newName
#  } else {
#    return -code error "seq $sequenceID doesn't exist"
#  }
}

# -------------------------------------------------------------------------
proc ::SeqData::getSources {sequenceID} {

    variable seqs
    
    # Get the sequence name.
    if {[info exists seqs($sequenceID)]} {
        return $seqs($sequenceID,sources)
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::setSources {sequenceID sources} {

    variable seqs
    
    # Get the sequence name.
    if {[info exists seqs($sequenceID)]} {
        set seqs($sequenceID,sources) $sources
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getSourceData {sequenceID sourceName} {

    variable seqs
    
    # Get the sequence name.
    if {[info exists seqs($sequenceID)]} {
        foreach source $seqs($sequenceID,sources) {
            if {[lindex $source 0] == $sourceName} {
                return [lindex $source 1]
            }
        }
        return {}
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::setSourceData {sequenceID sourceName sourceData} {

    variable seqs
    
    # Get the sequence name.
    if {[info exists seqs($sequenceID)]} {
        
        # Try to replace the existing entry.
        for {set i 0} {$i < [llength $seqs($sequenceID,sources)]} {incr i} {
            if {[lindex [lindex $seqs($sequenceID,sources) $i] 0] == $sourceName} {
                set seqs($sequenceID,sources) [lreplace $seqs($sequenceID,sources) $i $i [list $sourceName $sourceData]]
                return
            }
        }
        
        # Otherwise just add it.
        lappend seqs($sequenceID,sources) [list $sourceName $sourceData]
        
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}


# -------------------------------------------------------------------------
proc ::SeqData::hasStruct { seqNum } {
  variable seqs

  if {[info exists seqs($seqNum)]} {
    return $seqs($seqNum,hasStruct)
  } else {
    return -code error "seq $seqNum doesn't exist"
  }
}

# -------------------------------------------------------------------------
proc ::SeqData::setHasStruct {seqNum hasStruct} {
  variable seqs
#  puts "seqdata.tcl.setHasStruct seqNum: $seqNum, hasStruct: $hasStruct"
#  parray seqs
  if {[info exists seqs($seqNum)]} {
    set seqs($seqNum,hasStruct) $hasStruct
  } else {
    return -code error "seq $seqNum doesn't exist"
  }
}

# -------------------------------------------------------------------------
proc ::SeqData::getFirstResidue {seqNum} {
  variable seqs

  if {[info exists seqs($seqNum)]} {
    return $seqs($seqNum,firstRes)
  } else {
    return -code error "seq $seqNum doesn't exist"
  }
}

# -------------------------------------------------------------------------
proc ::SeqData::setFirstResidue {sequenceID firstResidue} {
    
    variable sequenceCommands
    variable seqs
    
    # Make sure the sequence exists.
    if {[info exists seqs($sequenceID)]} {
        
        # If we have an override behavior, call it.
        if {[info exists sequenceCommands(changeHandler,$sequenceID)]} {
             return [$sequenceCommands(changeHandler,$sequenceID) "setFirstResidue" $sequenceID $firstResidue]
             
        # Otherwise just use the default behavior.
        } else {
            
            # Save the new first residue.
            set seqs($sequenceID,firstRes) $firstResidue
            
            return 1
        }
    
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getSpecificResidueCount {seqNum res} {
  variable seqs

  if {[info exists seqs($seqNum)]} {
    set residueCount 0
    foreach element [seq get $seqNum] {
        if {$element == $res} {
            incr residueCount
        }
    }
    return $residueCount
  } else {
    return -code error "seq $seqNum doesn't exist"
  }
}

# -------------------------------------------------------------------------
proc ::SeqData::getResidueCount {seqNum} {
  variable seqs

  if {[info exists seqs($seqNum)]} {
    set residueCount 0
    foreach element [seq get $seqNum] {
        if {$element != "-"} {
            incr residueCount
        }
    }
    return $residueCount
  } else {
    return -code error "seq $seqNum doesn't exist"
  }
}


# -------------------------------------------------------------------------
proc ::SeqData::getResidueForElement {sequenceID element} {
    
   variable seqs
   variable sequenceCommands
   if {[info exists seqs($sequenceID)]} {
      if {[info exists sequenceCommands(elementToResidueMapping,$sequenceID)]} {
         $sequenceCommands(elementToResidueMapping,$sequenceID) \
                                                  $sequenceID $element
      } else {
         set residue [seq resAt $sequenceID $element]
         if {$residue >= 0} {
            return [expr [::SeqData::getFirstResidue $sequenceID]+$residue]
         } else {
            return -1
         }
      }
   } else {
      return -code error "seq $sequenceID doesn't exist"
   }
}

# -------------------------------------------------------------------------
proc ::SeqData::getElementForResidue {sequenceID residue} {
    
    variable seqs
    variable sequenceCommands
    
    if {[info exists seqs($sequenceID)]} {
        if {[info exists sequenceCommands(residueToElementMapping,$sequenceID)]} {
             $sequenceCommands(residueToElementMapping,$sequenceID) $sequenceID $residue
        } else {
            return [seq posOf $sequenceID [expr $residue-[::SeqData::getFirstResidue $sequenceID]]]
        }
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
# This method gets the type of the specified sequence.
# args:     sequenceID - The sequence if for which the type should be retrieved.
# return:   The type of the sequence: protein, rna, dna
proc ::SeqData::getType {sequenceID} {
   return [seq type $sequenceID]

#    variable seqs
##    variable sequenceCommands
#    
#    if {[info exists seqs($sequenceID)]} {
#        return $seqs($sequenceID,type)
#    } else {
#        return -code error "seq $sequenceID doesn't exist"
#    }
}
    
# -------------------------------------------------------------------------
# This method sets the type of the specified sequence.
# args:     sequenceID - The sequence id for which the type should be set.
# args:     type - the type: protein, rna, dna
proc ::SeqData::setType {sequenceID type} {
   puts "can't set type" 
#    variable seqs
##    variable sequenceCommands
#    
#    if {[info exists seqs($sequenceID)]} {
#        set seqs($sequenceID,type) $type
#    } else {
#        return -code error "seq $sequenceID doesn't exist"
#    }
}
    
# -------------------------------------------------------------------------
# Gets the domain of life for the specified sequence.
# args:     seqNum - The id of the sequence to lookup.
# return:   The doamin of life (Eukaryota|Archaea|Bacteria) or and empty string ("") if the domain
#           of life is not known.
proc ::SeqData::getDomainOfLife {seqNum} {

    variable seqs
    
    # Make sure this is a valid sequence.
    if {[info exists seqs($seqNum)]} {
        
        # Try to get the domain of life from the annotations.
        set domain [getAnnotation $seqNum "domain-of-life"]
        
        # If we don't have a name in the annotations, see if we have a taxonomy node.
        if {$domain == "" && $seqs($seqNum,taxNode) != ""} {
            
            # Get the name from the taxonomy tree.
            set domain [::SeqData::Taxonomy::getDomainOfLife $seqs($seqNum,taxNode)]
        }
        
        return $domain
        
    } else {
        return -code error "seq $seqNum doesn't exist"
    }
}

# -------------------------------------------------------------------------
# args:     seqNum - The id of the sequence to lookup.
# return:   A list containing the organism's lineage or an empty list if it is unknown.
proc ::SeqData::getCommonName {seqNum} {

    variable seqs
    
    # Make sure this is a valid sequence.
    if {[info exists seqs($seqNum)]} {
        
        # Try to get the name from the annotations.
        set commonName [getAnnotation $seqNum "common-name"]
        
        # If we don't have a name in the annotations, see if we have a taxonomy node.
        if {$commonName == "" && $seqs($seqNum,taxNode) != ""} {
            
            # Get the name from the taxonomy tree.
            set commonName [::SeqData::Taxonomy::getCommonName $seqs($seqNum,taxNode)]
        }
        
        return $commonName
        
    } else {
        return -code error "seq $seqNum doesn't exist"
    }
}

# -------------------------------------------------------------------------
# args:     seqNum - The id of the sequence to lookup.
# return:   A list containing the organism's lineage or an empty list if it is unknown.
proc ::SeqData::getScientificName {seqNum} {

    variable seqs
    
    # Make sure this is a valid sequence.
    if {[info exists seqs($seqNum)]} {
        
        # Try to get the name from the annotations.
        set scientificName [getAnnotation $seqNum "scientific-name"]
        
        # If we don't have a name in the annotations, see if we have a taxonomy node.
        if {$scientificName == "" && $seqs($seqNum,taxNode) != ""} {
            
            # Get the name from the taxonomy tree.
            set scientificName [::SeqData::Taxonomy::getScientificName $seqs($seqNum,taxNode)]
        }
        
        return $scientificName
        
    } else {
        return -code error "seq $seqNum doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getShortScientificName {seqNum} {

    set name [getScientificName $seqNum]
    set nameParts [split $name]
    if {[llength $nameParts] <= 1} {
        return $name
    } elseif {[llength $nameParts] >= 2} {
        return "[string index [lindex $nameParts 0] 0]. [lindex $nameParts 1]"
    }
    return $name
}

# -------------------------------------------------------------------------
# Gets the lineage of the organism from which the sequence came.
# args:     seqNum - The id of the sequence to lookup.
# return:   A list containing the organism's lineage or an empty list if it is unknown.
proc ::SeqData::getLineage {seqNum {showHidden 0} {includeRanks 0} {includeSelf 0}} {

    variable seqs

#    if { $seqNum == 1} {
#       puts "seqdata.getLineage. seqs: [array get seqs]"
#    }

    # Make sure this is a valid sequence.
    if {[info exists seqs($seqNum)]} {
        
        # Try to get the name from the annotations.
        set lineage [getAnnotation $seqNum "lineage"]

        if {$lineage == ""} {
            set lineage {}
        }
        
        # If we didn't have a name in the annotations, 
        # see if we have a taxonomy node.
#        puts "seqdata.getLineage. seqNum: $seqNum, showHid: $showHidden, includeRanks: $includeRanks, incSelf: $includeSelf, lineage: '$lineage', taxNode: '$seqs($seqNum,taxNode)'" 
        if {$lineage == {} && $seqs($seqNum,taxNode) != ""} {
            
            # Get the name from the taxonomy tree.
            set lineage [::SeqData::Taxonomy::getLineage $seqs($seqNum,taxNode) $showHidden $includeRanks $includeSelf]
        }
        
        return $lineage
        
    } else {
        return -code error "seq $seqNum doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getLineageRank {sequenceID rank} {

    variable seqs
    
    # Make sure this is a valid sequence.
    if {[info exists seqs($sequenceID)]} {

        # Search the lineage for the specified rank.        
        set lineage [::SeqData::Taxonomy::getLineage $seqs($sequenceID,taxNode) 1 1 1]
        foreach level $lineage {
            if {[lindex $level 1] == $rank} {
                return [lindex $level 0]
            }
        }
        
        return ""

    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}


# -------------------------------------------------------------------------
proc ::SeqData::getEnzymeCommissionNumber {seqNum} {

    variable seqs
    
    # Make sure this is a valid sequence.
    if {[info exists seqs($seqNum)]} {
        
        # Try to get the name from the annotations.
        set ecNumber [getAnnotation $seqNum "ec-number"]
        
        # If we don't have a name in the annotations, see if we have a taxonomy node.
        if {$ecNumber == "" && $seqs($seqNum,ecNumber) != ""} {
            set ecNumber $seqs($seqNum,ecNumber)
        }
        
        return $ecNumber
        
    } else {
        return -code error "seq $seqNum doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::getEnzymeCommissionDescription {seqNum} {

    # If we have an annotation, use it.
    if {[getAnnotation $seqNum "ec-description"] != ""} {
        return [getAnnotation $seqNum "ec-description"]
    }

    # Otherwise, lookup the code.
    return [::SeqData::Enzyme::getDescription [getEnzymeCommissionNumber $seqNum]]
}

# -------------------------------------------------------------------------
proc ::SeqData::getTemperatureClass {sequenceID} {


    variable seqs
    
    # Make sure this is a valid sequence.
    if {[info exists seqs($sequenceID)]} {
        
        # Try to get the value from the annotations.
        set class [getAnnotation $sequenceID "temperature-class"]
        
        # If we don't have a value in the annotations, look it up.
        if {$class == "" && $seqs($sequenceID,taxNode) != ""} {
            set class [::SeqData::GrowthTemperature::getTemperatureClass $seqs($sequenceID,taxNode)]
        }
        
        return $class
        
    } else {
        return -code error "seq $sequenceID doesn't exist"
    }
}

# -------------------------------------------------------------------------
proc ::SeqData::addAnnotation { seqNum key value } {
  variable seqs

  if {[info exists seqs($seqNum)]} {
    set seqs($seqNum,annot,$key) $value
    #saveAnnotations $seqNum
  } else {
    return -code error "seq $seqNum doesn't exist"
  }
}

# -------------------------------------------------------------------------
proc ::SeqData::getAnnotation { seqNum key } {
  variable seqs

  if {[info exists seqs($seqNum)]} {
      
    if {[info exists seqs($seqNum,annot,$key)]} {
      return $seqs($seqNum,annot,$key)
    } else {
      if { $seqs($seqNum,annot_curr) == 0 } {
        #loadAnnotations $seqNum
        set seqs($seqNum,annot_curr) 1
      }
      if {[info exists seqs($seqNum,annot,$key)]} {
        return $seqs($seqNum,annot,$key)
      } else {
        return ""
      }
    }
  } else {
    return -code error "seq $seqNum doesn't exist"
  }
}

# -------------------------------------------------------------------------
# Return all this seq's annotations as an array
proc ::SeqData::getAllAnnotations { seqNum } {
  variable seqs

  if {[info exists seqs($seqNum)]} {
    return [array get seqs "$seqNum,annot,*"]
  } else {
    return -code error "Seq $seqNum doesn't exist"
  }
}

# -----------------------------------------------------------------------
# Copies all annotations from seqId1 to seqId2
# args: seqId1 - source sequence
#       seqId2 - destination sequence
proc ::SeqData::copyAnnotations { seqId1 seqId2 } {
  variable seqs

  foreach annotation [array names seqs "$seqId1,annot,*"] {
    set key [lindex [split $annotation ","] 2]
    set seqs($seqId2,annot,$key) $seqs($annotation)
  }
}

# -----------------------------------------------------------------------
# Writes seq annotations into the annotations back-end.
# args:  seqNum - the number of the sequence the annotation is for
# return:   1/0 success/fail  (from the backend function)
proc ::SeqData::saveAnnotations { seqNum } {
  variable backend
  variable seqs

  switch $backend {
    FILE { 
      foreach key [array names $seqs($seqNum,annot)] {
        return [writeAnnotationToFileBackend \
           [getName $seqNum] \
          $key $seqs($seqNum,annot,$key)]
      }
    }
    default {
      return -code error "saveAnnotations: No such backend $backend!"
    }
  }
}
 
# -----------------------------------------------------------------------
# Gets seq annotations from the annotations back-end.
# args:  seqNum - the number of the sequence to get annotations for
# return:   none
proc ::SeqData::loadAnnotations { seqNum } {
  variable backend
  variable seqs

  switch $backend {
    FILE { 
      set $seqs($seqNum,annot) \
        [getAnnotationsFromFileBackend [getName $seqNum]]
    }
    default {
       return -code error "loadAnnotations: No such backend $backend!"
    }
  }
}  


# -----------------------------------------------------------------------
# FILE backend functions
# The format for the annotations file is as follows:
#
# name1|key1=note 1|key2=note 2|key3=....
# name2|key5=note 1|key1=note 2|key7=....
# 
# any newlines that were embedded in the original user entry should be encoded 
# as "<NL>" for entry into the annotations file, and decoded back to a newline
# when the annotation is read out.
 
# Annotation writing function for FILE backend.
# args:   name - name of seq. to save the annotations under
#         key - name of the annotation
#         annotation - content of the annotation
# return: 1/0 success/fail
proc ::SeqData::writeAnnotationToFileBackend { name key annotation } {
  variable seqs
  variable annotationsFile

  if { ! [set fp [open $annotationsFile r]] } {
    return -code error "Can't open $annotationsFile for read!"
  }
  
  # Search out the annotations for this particular name



}

# -----------------------------------------------------------------------
# Annotation loading function for FILE backend.
# args:   name - name of seq. to load the annotations for
# return: an array of keys/annotations
proc ::SeqData::getAnnotationsFromFileBackend { name } {
  variable seqs
  variable annotationsFile

  if { ! [set fp [open $annotationsFile r]] } {
    return -code error "Can't open $annotationsFile for read!"
  }
}

# -----------------------------------------------------------------------
proc ::SeqData::getGaps {number} {
    set ret {}
    for {set i 0} {$i < $number} {incr i} {
        lappend ret "-"
    }
    return $ret
}


# -----------------------------------------------------------------------
proc ::SeqData::getPercentGC { seqID } {

	set sequence [getSeq $seqID]
	
	set gc 0
	
	foreach element $sequence {
		if { $element == "G" || $element == "C" } {
			incr gc
		}
	}

	set percent [expr ($gc * 100.0)/[getSeqLength $seqID]]
	set percent [string range $percent 0 [expr [string first "." $percent] + 4]]
	
	return $percent
}

