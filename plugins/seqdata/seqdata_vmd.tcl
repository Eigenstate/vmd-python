############################################################################
#cr
#cr            (C) Copyright 1995-2004 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: seqdata_vmd.tcl,v 1.10 2013/04/15 17:36:41 johns Exp $
#

# This file provides functions for reading and writing sequence data from FASTA formatted files.

package provide seqdata 1.1

namespace eval ::SeqData::VMD {

    # Export the package namespace.
    namespace export loadVMDSequences

    # The map of three letter to one letter amino acid codes.
    variable codes

    # The loaded parts.
    variable loadedMolecules

    # The loaded parts.
    variable loadedParts

    # The map of seqdata ids to vmd mol ids.
    variable sequenceIDToMolID

    # The map of vmd mol ids to seqdata ids.
    variable molIDToSequenceID                                 

    # The map of molecule and chain to residues.
    variable residueListsMap

    # The map of vmd residues to sequence elements.
    variable residueToElementMap

    # The map of sequence elements to vmd residues.
    variable elementToResidueMap

    # Whether the loader should ignor non registered structures.
    variable ignoreNonRegisteredStructures 0

    # Existing sequence ids that should be used for VMD structures.
    variable registeredSequences {}

    proc reset {} {
       array set ::SeqData::VMD::codes {
          ACE  *   ALA  A   ARG  R   ASN  N   ASP  D   
          ASX  B   CYS  C   CYX  C   GLN  Q   GLU  E   
          GLX  Z   GLY  G   HIS  H   HSE  H   HSD  H   
          HSP  H   ILE  I   LEU  L   LYS  K   MET  M   
          MSE  M   PHE  F   PRO  P   SER  S   THR  T   
          TRP  W   TYR  Y   VAL  V   A    A   C    C   
          G    G   T    T   U    U   ADE  A   CYT  C   
          GUA  G   THY  T   URA  U   TIP3 O   MG   M   
          MN   G   SOD  N   POT  K   HOH  O   ATP  *
          1MA  E   MAD  E   T6A  F   OMC  B   5MC  H   
          OMG  J   1MG  K   2MG  L   M2G  R   7MG  M
          G7M  M   5MU  T   RT   T   4SU  V   DHU  D   
          H2U  D   PSU  P   I    I   YG   Y}
       set ::SeqData::VMD::loadedMolecules {}
       array unset ::SeqData::VMD::loadedParts 
       array unset ::SeqData::VMD::sequenceIDToMolID 
       array unset ::SeqData::VMD::molIDToSequenceID 
       array unset ::SeqData::VMD::residueListsMap 
       array unset ::SeqData::VMD::residueToElementMap 
       array unset ::SeqData::VMD::elementToResidueMap 
       set ::SeqData::VMD::ignoreNonRegisteredStructures 0
       set ::SeqData::VMD::registeredSequences {}
    }
    reset

# ----------------------------------------------------------------------
    proc setIgnoreNonRegisteredStructures {newValue} {
        variable ignoreNonRegisteredStructures
        set ignoreNonRegisteredStructures $newValue
    }

# ----------------------------------------------------------------------
    proc registerSequenceForVMDStructure {name chain sequenceID} {
        variable registeredSequences
        lappend registeredSequences [list $name $chain $sequenceID]
    }

# ----------------------------------------------------------------------
# Updates list of VMD sequences into sequence store to correspond to what is 
# loaded into VMD.
# return:   list of two lists containing updates. first list contains 
#           any sequence ids that were newly loaded. The second list 
#           contains any sequence ids that were removed.
   proc updateVMDSequences {} {
      variable loadedMolecules
      variable loadedParts

      # Go through each molecule in VMD.
      set addedSequenceIDs {}
      set vmdMolIDs [molinfo list]
      foreach molID $vmdMolIDs {

         # See if we have not yet loaded this molecule.
         if {[lsearch $loadedMolecules $molID] == -1} {

            # Assume the molecule is initialized.
            set moleculeIntialized 1

            #  Get a list of the chains in the molecule.
            set parts [getPartsInMolecule $molID]

            # If molecule has no parts, it might later get some; for example, 
            # the script "mol new ; mol addfile 1atp" will initially generate a
            # molecule with no parts, but later addition of structure doesn't
            # get noticed because the molecule is already registered.
            # Justin Gullingsrud; 8/16/06
            if {[llength $parts] < 1} {
               set moleculeIntialized 0
            }

            # Go through each part.
            foreach part $parts {

               set chain [lindex $part 0]
               set segname [lindex $part 1]

               if {![info exists loadedParts($molID,$chain,$segname)]} {

                  # Get the atoms of the molecule.
                  if {$segname == ""} {
                     set atoms [atomselect $molID "chain $chain"]
                     set solvent [atomselect $molID \
                                         "chain $chain and (water or lipid)"]
                  } else {
                     set atoms [atomselect $molID \
                                      "chain $chain and segname \"$segname\""]
                     set solvent [atomselect $molID \
                           "chain $chain and segname \"$segname\" and (water or lipid)"]
                  }

                  # Make sure this parts is not a bunch of MD waters or lipids.
                  #$numatoms < 300 || [expr $numSolventAtoms/$numatoms] < 0.95
                  set numatoms [$atoms num]
                  set numSolventAtoms [$solvent num]
                  if {$numatoms != $numSolventAtoms} {

                     # Get the residue information.
                     set atomPropsList [$atoms get {resid insertion \
                                                   resname residue altloc}]
#puts "seqdata_vmd.updateVMDSequences.atomPropsList: $atomPropsList"
                     if {$numatoms == [llength $atomPropsList]} {

                        # Go through each residue.
                        set uniqueResIDs {}
                        set uniqueResidues {}
                        set residues {}
                        set sequence {}
                        foreach atomProps $atomPropsList {

                           # Get the residue id.
                           set resID [lindex $atomProps 0]
                           set insertion [lindex $atomProps 1]
                           set resName [lindex $atomProps 2]
                           set residue [lindex $atomProps 3]
                           set altloc [lindex $atomProps 4]

                           # Make sure we process each unique resid only once.
                           if {[lsearch $uniqueResIDs "$resID$insertion"]==-1} {
                              lappend uniqueResIDs "$resID$insertion"

                              # See if this is just a normal residue.
                              if {$insertion == " " && $altloc == ""} {
                                 lappend residues $resID
                                 lappend sequence [lookupCode $resName]

                              } else {
                                 # We must have either an insertion or 
                                 # an alternate location.

                                 # save the residue as a list.
                                 lappend residues [list $resID \
                                                      $insertion $altloc]
                                 lappend sequence [lookupCode $resName]

                                 # See if we should update any previous residues
                                 set updateIndex [lsearch $residues $resID]
                                 if {$updateIndex != -1} {
                                    set residues [lreplace $residues \
                                              $updateIndex $updateIndex \
                                              [list $resID " " ""]]
                                 }
                              }
                           }          

                           # Store the residue to check for VMD preload problem.
                           if {[lsearch $uniqueResidues $residue] == -1} {
                              lappend uniqueResidues $residue
                           }                        
                        }

#puts "seqdata_vmd.updateVMDSequences.molID: $molID,uniqRes:<$uniqueResidues>, resIDs: <$uniqueResIDs>"
                        # If we only have one residue, make sure we only 
                        # have one resid to avoid VMD preload problem.
                        #if [llength $uniqueResidues] > 1 || 
                        if {[llength $uniqueResidues] > 1 || \
                                             [llength $uniqueResIDs] == 1} {

                           # If there is more than one parts for this 
                           # molecule, use the parts as the name suffix.
                           set nameSuffix ""
                           if {[llength $parts] > 1} {
                              if {$segname != ""} {
                                 set nameSuffix "_$segname"
                              } else {
                                 set nameSuffix "_$chain"
                              }
                           }

                           # Add the sequence to the sequence store.
                           set sequenceID [addVMDSequence $molID $chain \
                                     $segname $sequence \
                                     "[getMoleculeName $molID]$nameSuffix" \
                                     [lindex $residues 0] $residues]
                           if {$sequenceID != -1} {

                              # Add sequence to list of seq ids we have added.
                              lappend addedSequenceIDs $sequenceID
                           }

                           # Mark that we have loaded this part.
                           set loadedParts($molID,$chain,$segname) 1

                        } else {

                           # Mark this molecule as not yet initialized.
                           set moleculeIntialized 0
                           break
                        }
                     }
                  }

                  # Remove the atomselect.
                  $atoms delete
                  $solvent delete
               }
            }

            # If we made it through all of the parts and the molecule 
            # was initialized, we have finished the entire molecule.
            if {$moleculeIntialized} {
               lappend loadedMolecules $molID
            }
         }
      }

      # Go through each molecule that is currently loaded.
      set removedSequenceIDs {}
      for {set i 0} {$i < [llength $loadedMolecules]} {incr i} {

         # Get the mol id.
         set loadedMolID [lindex $loadedMolecules $i]

         # See if the molecule is no longer in VMD.
         if {[lsearch $vmdMolIDs $loadedMolID] == -1} {

            # Remove it from the list of loaded molecules.
            set loadedMolecules [lreplace $loadedMolecules $i $i]
            incr i -1

            # Add seqids that correspond to this mol to list of remvoed ids.
            set removedSequenceIDs [concat $removedSequenceIDs \
                                  [getSequenceIDsForMolecule $loadedMolID]]
         }
      }

      # Return the lists.
      return [list $addedSequenceIDs $removedSequenceIDs]
   } ;# end of updateVMDSequences

# ------------------------------------------------------------------------
    # This method gets the name of a molecule.
    # args:     molID - The molecule of whicht o retrieve the name.
    # return:   The name of the molecule.
    proc getMoleculeName {molID} {

        # Get the molecule name.
        set moleculeName [molinfo $molID get name]
        regsub -nocase {\.pdb$} $moleculeName "" moleculeName
        return $moleculeName
    }

# ------------------------------------------------------------------------
    # This method gets a list of all of the chain and segname pairs in the specified molecule.
    # args:     molID - The VMD molecule id for the molecule.
    # return:   A list of the chain and segname pairs in the specified molecule.
    proc getPartsInMolecule {molID} {

        # Get the number of atoms and the number of frames.            
        set atoms [atomselect $molID "all"]

        #  Get a list of the chains in the molecule.
        set uniqueParts {}
        foreach part [$atoms get {chain segname}] {

            # Get the chain and segname.
            set chain [lindex $part 0]
            set segname [lindex $part 1]
            if {$chain != "X"} {
                set segname ""
            }

            # See if we have already found this part.
            set found 0
            foreach uniquePart $uniqueParts {
                if {$chain == [lindex $uniquePart 0] && $segname == [lindex $uniquePart 1]} {
                    set found 1
                    break
                }
            }
            if {!$found} {
                lappend uniqueParts [list $chain $segname]
            }
        }

        # Remove the atomselect.
        $atoms delete
        unset atoms

        return $uniqueParts        
    } ; #end of proc getPartsInMolecule 

# ---------------------------------------------------------------------------
    # This method adds a VMD sequence to the sequence store.
    # args:     molID - The VMD molecule id for which this is the sequence.
    #           sequence - The sequence itself.
    #           name - The name of the sequence.
    #           startElementId - The "real" index of the first element in the sequence.
    # return:   The sequence ID of the added sequence.
    proc addVMDSequence {molID chain segname sequence name \
                                                 firstResidue residues} {
#      puts "seqdata_vmd.AddVMDSequence, molID:$molID, chain:$chain,segname:$segname,sequence:$sequence,name:$name,firstResidue:$firstResidue,residues:$residues"
        variable ignoreNonRegisteredStructures
        variable registeredSequences
        variable sequenceIDToMolID
        variable molIDToSequenceID
        variable residueListsMap

        # See if we have a sequence registered for this structure.
        set sequenceID -1
        for {set i 0} {$i < [llength $registeredSequences]} {incr i} {

            # Get the sequence registration info.
            set registeredSequence [lindex $registeredSequences $i]

            # If we found a match, use it.
            if {[lindex $registeredSequence 0] == [getMoleculeName $molID] && ([lindex $registeredSequence 1] == $chain || [lindex $registeredSequence 1] == "*")} {

                # Get the sequence id.
                set sequenceID [lindex $registeredSequence 2]

                # Adjust the data.
                ::SeqData::setHasStruct $sequenceID "Y"
                ::SeqData::setFirstResidue $sequenceID $firstResidue
                ::SeqData::setType $sequenceID [determineType $molID $chain $segname]  

                # Remove the sequence from the registered list and stop the loop.
                set registeredSequences [lreplace $registeredSequences $i $i]
                break
            }
        }

        # Otherwise, create the new sequence in the sequence store.
        if {$sequenceID == -1} {

            # If we are ignoring non registered structures, quit.
            if {$ignoreNonRegisteredStructures} {return -1}

            # If this is a PDB file, set the source appropriately.            
            set sources {}
            if {[set pdbCode [::SeqData::PDB::isValidPDBName [getMoleculeName $molID]]] != ""} {
                set sources [list [list "pdb" [list $pdbCode $chain]]]
            }

            # Create the sequence.
            set sequenceID [::SeqData::addSeq $sequence $name "Y" $firstResidue $sources [determineType $molID $chain $segname]]
        }

        # Add a change handler to keeep the residue mappings up to date.
        ::SeqData::setCommand changeHandler $sequenceID "::SeqData::VMD::sequenceChanged"
        ::SeqData::setCommand elementToResidueMapping $sequenceID "::SeqData::VMD::getResidueForElement"
        ::SeqData::setCommand residueToElementMapping $sequenceID "::SeqData::VMD::getElementForResidue"
        ::SeqData::setCommand secondaryStructureCalculation $sequenceID "::SeqData::VMD::calculateSecondaryStructure"
        ::SeqData::setCommand copyAttributes $sequenceID "::SeqData::VMD::copyVMDAttributes"

#        puts "seqdata_vmd.addVMDSequence. seqID:$sequenceID, molID:$molID, chain:$chain, segname:$segname"
        # Save the sequence to mol id mapping.
        set sequenceIDToMolID($sequenceID) [list $molID $chain $segname]
        if {![info exists molIDToSequenceID($molID,$chain,$segname)]} {
            set molIDToSequenceID($molID,$chain,$segname) {}
        }
        lappend molIDToSequenceID($molID,$chain,$segname) $sequenceID

        # Save the residue list.
        set residueListsMap($molID,$chain,$segname) $residues

        # Compute the residue/element mappings.
        computeResidueElementMappings $sequenceID $molID $chain $segname        

        # Determine the first and last residues in the first segment.
        set residueListsMap($molID,$chain,$segname,firstRegionRange) [determineFirstRegionRange $sequenceID $molID]

        return $sequenceID 
    } ; #end of proc addVMDSequence 

# ---------------------------------------------------------------------
    # Figure out the first region starting and ending residues.
    proc determineFirstRegionRange {sequenceID molID} {

        variable elementToResidueMap

        # Get the sequence type.
        set type [::SeqData::getType $sequenceID]

        set inRegion 0
        set lastResidueInRegion {}
        set regionRange {}

        # Go through each element.
        for {set i 0} {$i < [::SeqData::getSeqLength $sequenceID]} {incr i} {

            # Get the selection string.
            set selectionString [::SeqData::VMD::getSelectionStringForElements $sequenceID $i]

            # See if we got an atom selection.
            if {$selectionString != "none"} {

                # Get the atom types.
                set atoms [atomselect $molID $selectionString]
                set atomTypes [$atoms get name]
                $atoms delete

                # See if we are in a region corresponding to our sequence type.
                if {($type == "protein" && [lsearch $atomTypes "CA"] != -1) || (($type == "nucleic" || $type == "rna" || $type == "dna") && [lsearch $atomTypes "P"] != -1)} {

                    # If we are not yet tracking elements, start, and add this element to the list.
                    if {!$inRegion} {
                        set inRegion 1
                        lappend regionRange $elementToResidueMap($sequenceID,$i)
                    }
                    set lastResidueInRegion $elementToResidueMap($sequenceID,$i)

                # Otherwise, we must have moved out of the region so we are done.
                } elseif {$inRegion} {
                    lappend regionRange $lastResidueInRegion
                    break                    
                }
            }
        }

        if {[llength $regionRange] == 1} {
            lappend regionRange $lastResidueInRegion
        }

        return $regionRange
    }

# -----------------------------------------------------------------------
    # This method copies the VMD attributes from one sequence to another.
    # args:     oldSequenceID - The old sequence id.
    #           newSequenceID - The new sequence id.
    proc copyVMDAttributes {oldSequenceID newSequenceID} {

        variable sequenceIDToMolID
        variable molIDToSequenceID

        # Copy the seqdata attributes
        ::SeqData::setHasStruct $newSequenceID "Y"

        # Add a change handler to keeep the residue mappings up to date.
        ::SeqData::setCommand changeHandler $newSequenceID "::SeqData::VMD::sequenceChanged"
        ::SeqData::setCommand elementToResidueMapping $newSequenceID "::SeqData::VMD::getResidueForElement"
        ::SeqData::setCommand residueToElementMapping $newSequenceID "::SeqData::VMD::getElementForResidue"
        ::SeqData::setCommand secondaryStructureCalculation $newSequenceID "::SeqData::VMD::calculateSecondaryStructure"
        ::SeqData::setCommand copyAttributes $newSequenceID "::SeqData::VMD::copyVMDAttributes"

        # Save the molid/sequence mappings.
        set molID [lindex $sequenceIDToMolID($oldSequenceID) 0]
        set chain [lindex $sequenceIDToMolID($oldSequenceID) 1]
        set segname [lindex $sequenceIDToMolID($oldSequenceID) 2]
        set sequenceIDToMolID($newSequenceID) [list $molID $chain $segname]
        if {![info exists molIDToSequenceID($molID,$chain,$segname)]} {
            set molIDToSequenceID($molID,$chain,$segname) {}
        }
        lappend molIDToSequenceID($molID,$chain,$segname) $newSequenceID

        # Compute the residue/element mappings.
        computeResidueElementMappings $newSequenceID $molID $chain $segname
    }

# -----------------------------------------------------------------------
   proc extractFirstRegionFromStructures {sequenceIDs {onlyStructs "N"}} {

      variable sequenceIDToMolID
      variable residueListsMap
      variable residueToElementMap
      variable elementToResidueMap

      # The regions we are extracting.
      set regions {}

      # Go through each sequence.
      foreach sequenceID $sequenceIDs {

         # If a structure, extract first segment that is of correct type.
         if {[::SeqData::hasStruct $sequenceID] == "Y"} {
            # Get the sequence info.
            set molID [lindex $sequenceIDToMolID($sequenceID) 0]
            set chain [lindex $sequenceIDToMolID($sequenceID) 1]
            set segname [lindex $sequenceIDToMolID($sequenceID) 2]

            # Make sure we have a first region.
            if {[llength $residueListsMap($molID,$chain,$segname,firstRegionRange)] == 2} {

               # Get the residue list.
               set residueList $residueListsMap($molID,$chain,$segname)

               # Get the range of the first region.
               set firstResidueInRegion [lindex $residueListsMap($molID,$chain,$segname,firstRegionRange) 0]
               set firstResidueInRegionIndex [lsearch $residueList $firstResidueInRegion]
               set lastResidueInRegion [lindex $residueListsMap($molID,$chain,$segname,firstRegionRange) 1]
               set lastResidueInRegionIndex [lsearch $residueList $lastResidueInRegion]

               # Figure out the first element to use.
               set firstElement 0
               set firstResidueIndex [lsearch $residueList $elementToResidueMap($sequenceID,$firstElement)]

               # Figure out the last element to use.
               set lastElement [expr [::SeqData::getSeqLength $sequenceID]-1]
               set lastResidueIndex [lsearch $residueList $elementToResidueMap($sequenceID,$lastElement)]

#               puts "seqdata_vmd.tcl.extractFirstRegionFromStructures. seqID: $sequenceID, molID: $molID, chain: $chain, segname: $segname, resList: $residueList, frir: $firstResidueInRegion, friri: $firstResidueInRegionIndex, lrir: $lastResidueInRegion, lriri: $lastResidueInRegionIndex, fri: $firstResidueIndex, lri: $lastResidueIndex, le: $lastElement"
               
               if {$firstResidueInRegionIndex != -1 && \
                   $firstResidueIndex != -1 && \
                             $firstResidueInRegionIndex > $firstResidueIndex} {
                  set firstElement $residueToElementMap($sequenceID,[join \
                                      [lrange $firstResidueInRegion 0 1] ","])
               }


               if {$lastResidueInRegionIndex != -1 && \
                   $lastResidueIndex != -1 && \
                       $lastResidueInRegionIndex < $lastResidueIndex } {
                  set lastElement $residueToElementMap($sequenceID,[join \
                                      [lrange $lastResidueInRegion 0 1] ","])
               }

               # Make sure to include any gaps on either side of the region.
               while {$firstElement > 0 && [::SeqData::getElements \
                                 $sequenceID [expr $firstElement-1]] == "-"} {
                  incr firstElement -1
               }
               while {$lastElement < [expr [::SeqData::getSeqLength \
                                                     $sequenceID]-1] && \
                      [::SeqData::getElements $sequenceID [expr \
                                                    $lastElement+1]] == "-"} {
                  incr lastElement 
               }

               # Add this region to the list.
               set elements {}
               for {set i $firstElement} {$i <= $lastElement} {incr i} {
                  lappend elements $i
               }
               lappend regions $sequenceID
               lappend regions $elements
                
               # Otherwise this structure has no first region, so use nothing.
            } else {
               lappend regions $sequenceID
               lappend regions {}
            }

         # Otherwise this is a sequence, so just use the whole thing.
         } elseif { $onlyStructs == "N"} {
            set elements {}
            for {set i 0} {$i < [::SeqData::getSeqLength $sequenceID]} {incr i} {
               lappend elements $i
            }
            lappend regions $sequenceID
            lappend regions $elements
         }
      }
#      puts "seqdata_vmd.tcl.extractFirstRegionFromStructures. seqIDs: $sequenceIDs, regs: $regions"
      return [::SeqData::extractRegionsFromSequences $regions]
   } ; # end of extractFirstRegionFromStructures 

# -----------------------------------------------------------------------
   # Lookup one-letter code from three-letter code.
   proc lookupCode {resname} {

      variable codes

      # Find the code.
      set result ""
      if {[catch { set result $codes($resname) } ]} {
         set result "?"
      } 
#      puts "seqdata_vmd.lookupCode. resname: $resname, res: $result"
    
      return $result
   }

# -----------------------------------------------------------------------
    # This method gets the secondary structure for the specified sequence.
    proc calculateSecondaryStructure {sequenceID} {

        variable sequenceIDToMolID
        variable elementToResidueMap

        # Get the mol id and chain.
        set molID [lindex $sequenceIDToMolID($sequenceID) 0]

        # Go through the elements.
        set secondaryStructure {}
        for {set elementIndex 0} {$elementIndex < [::SeqData::getSeqLength $sequenceID]} {incr elementIndex} {

            # See if the element is not a gap.
            if {$elementToResidueMap($sequenceID,$elementIndex) != ""} {

                # It is not a gap, so get the secondary structure type.
                set atoms [atomselect $molID [getSelectionStringForElements $sequenceID $elementIndex]]
                lappend secondaryStructure [lindex [$atoms get structure] 0]
                $atoms delete   
            }
        }

        return $secondaryStructure
    }
# ------------------------------------------------------------------------    
   # This method computes all of the residue to element mappings for a given sequence.
   # args:     sequenceID - The id of the sequence for which to recompute the mappings.
   proc computeResidueElementMappings {sequenceID molID chain segname} {
#      puts stderr "seqdata_vmd.computeResidueElementMappings.start seqID: $sequenceID, molID: $molID, chain: $chain, seg: $segname"
      variable residueListsMap
      variable residueToElementMap
      variable elementToResidueMap
#      for {set i [info level]} {$i > 0} {incr i -1} {
#         puts stderr "Level $i: [info level $i]"
#      }
#puts stderr "seqdata_vmd.tcl.computeResidueElementMappings seqID: $sequenceID, molID: $molID, chain: $chain, segname: $segname"

      # Get the first residue.
      set firstResidue [::SeqData::getFirstResidue $sequenceID]

      # Get the residue list.
      set residueList $residueListsMap($molID,$chain,$segname)

      # Search for the first residue, in case this is a fragment.
      set residueListIndex [lsearch $residueList $firstResidue]
      if {$residueListIndex == -1} {
         set residueListIndex 0
      }

#puts stderr "seqdata_vmd.tcl.crem 1st: $firstResidue, rl: $residueList, rli: $residueListIndex, seq: [::SeqData::getSeq $sequenceID]"
      # Go through the elements and map non gaps to vmd residues.
      set elementIndex 0
      foreach element [::SeqData::getSeq $sequenceID] {

         # See if the element is a gap.
         if {$element != "-"} {

            # It is not a gap, so save its residue and also save the reverse mapping.
            set residue [lindex $residueList $residueListIndex]
            set elementToResidueMap($sequenceID,$elementIndex) $residue
            set residueToElementMap($sequenceID,[join [lrange $residue 0 1] ","]) $elementIndex
            incr residueListIndex

         } else {

            # It is a gap, so save that it maps to no residue.
            set elementToResidueMap($sequenceID,$elementIndex) ""
         }

         incr elementIndex            
      }
#puts stderr "seqdata_vmd.tcl.computeResidueElementMappings end"
   }

# ------------------------------------------------------------------------    
    # Gets the VMD mol id associated with a sequence id.
    # args:     sequenceID - The sequence id.
    # return    A list containing the VMD mol id and chain that maps to the specified sequence id.
    proc getMolIDForSequence {sequenceID} {

        variable sequenceIDToMolID

        return $sequenceIDToMolID($sequenceID)
    }

    # Gets all of the sequence ids associated with a VMD molecule.
    # args:     molID - The VMD mol id of the molecule.
    # return    A list containing all of the sequence ids associated with the molecule.
    proc getSequenceIDsForMolecule {molID} {

        variable molIDToSequenceID

        # Go through all of the matches and add the sequences to the list.
        set sequenceIDs {}
        set names [array names molIDToSequenceID "$molID,*"]
        foreach name $names {
            set sequenceIDs [concat $sequenceIDs $molIDToSequenceID($name)]
        }

        return $sequenceIDs
    }

# -----------------------------------------------------------------------
    # Gets the list of sequence ids associated with a VMD molecule and chain.
    # args:     molID - The VMD mol id of the molecule.
    # return    A list containing all of the sequence ids associated with the molecule and chain.
    proc getSequenceIDForMolecule {molID chain segname} {

        variable molIDToSequenceID

        return $molIDToSequenceID($molID,$chain,$segname)
    }

# -----------------------------------------------------------------------
    proc getResidues {sequenceID {atomType "all"}} {

        variable sequenceIDToMolID

#        puts "seqdata_vmd.getResidues.  seqID: $sequenceID, at: $atomType, seqIDToMolID: [array get sequenceIDToMolID]"
        set molID [lindex $sequenceIDToMolID($sequenceID) 0]

        # Get the selection string.
        set selectionString [getSelectionStringForElements $sequenceID]

        # If we are just writing a specific atom type, append it to the selection.
        if {$atomType != "" && $atomType != "all"} {
            append selectionString " and $atomType"
        }

        # Get the residues.
        set uniqueResidues {}
        set residues {}
        set atoms [atomselect $molID $selectionString]
        set atomPropsList [$atoms get {resid insertion altloc}]
        $atoms delete
        unset atoms
        foreach atomProps $atomPropsList {

            # Get the residue id.
            set resID [lindex $atomProps 0]
            set insertion [lindex $atomProps 1]
            set altloc [lindex $atomProps 2]

            if {[lsearch $uniqueResidues "$resID$insertion"] == -1} {
                lappend uniqueResidues "$resID$insertion"
                if {$insertion == " " && $altloc == ""} {
                    lappend residues $resID
                } else {

                    lappend residues [list $resID $insertion $altloc]
                    set updateIndex [lsearch $residues $resID]
                    if {$updateIndex != -1} {
                        set residues [lreplace $residues $updateIndex $updateIndex [list $resID " " ""]]
                    }
                }
            }             
        }
        return $residues
    }

# -----------------------------------------------------------------------
    # Gets the residue associated with a given element of a sequence id.
    # args:     sequenceID - The sequence id.
    #           element - Element we are looking for
    # return    residue number, or "" if an invalid element is passed in
    proc getResidueForElement {sequenceID element} {

        variable elementToResidueMap

        if {[info exists elementToResidueMap($sequenceID,$element)]} {
           return $elementToResidueMap($sequenceID,$element)
        } else {
           return ""
        }
    }

# -----------------------------------------------------------------------
    # Gets the VMD mol id associated with a sequence id.
    # args:     sequenceID - The sequence id.
    # return    The VMD mol id that maps to the specified sequence id.
    proc getElementForResidue {sequenceID residue} {

        variable residueToElementMap

        if {[info exists residueToElementMap($sequenceID,[join [lrange $residue 0 1] ","])]} {
            return $residueToElementMap($sequenceID,[join [lrange $residue 0 1] ","])
        } elseif {[llength $residue] > 1 && [info exists residueToElementMap($sequenceID,[lindex $residue 0])]} {
            return $residueToElementMap($sequenceID,[lindex $residue 0])
        }

        return -1
    }

# --------------------------------------------------------------------------
   # This method determines the type of the molecule.
   # args:  sequenceID - sequence if for which the type should be retrieved.
   # return:   The type of the sequence: nucleic, protein, or unknown.
#   proc determineType {molID chain segname} 
   proc determineTypeFromSeq {sequenceID} {
      # have to figure out molID, chain, and segname

      variable sequenceIDToMolID

      if { [info exists sequenceIDToMolID($sequenceID)] } {
         # Select the atoms and write them out to the file.         
         set molID [lindex $sequenceIDToMolID($sequenceID) 0]
         set chain [lindex $sequenceIDToMolID($sequenceID) 1]
         set segname [lindex $sequenceIDToMolID($sequenceID) 2]

         return [determineType $molID $chain $segname]
      } else {
         return ""
      }
   }


# --------------------------------------------------------------------------
   # This method determines the type of the molecule.
   # args:  sequenceID - sequence if for which the type should be retrieved.
   # return:   The type of the sequence: nucleic, protein, or unknown.
   proc determineType {molID chain segname} {

      if {$segname == ""} {
         set proteinAtoms [atomselect $molID "chain $chain and protein"]
         set nucleicAtoms [atomselect $molID "chain $chain and nucleic"]
      } else {
         set proteinAtoms [atomselect $molID \
                          "chain $chain and segname \"$segname\" and protein"]
         set nucleicAtoms [atomselect $molID \
                          "chain $chain and segname \"$segname\" and nucleic"]
      }
      set proteinAtomCount [$proteinAtoms num]
      set nucleicAtomCount [$nucleicAtoms num]

      # See what type of molecule vmd thinks this is.
      set type "unknown"
      if {$proteinAtomCount > 0 && $nucleicAtomCount > 0} {
         if {$proteinAtomCount > $nucleicAtomCount} {
            set type "protein"
         } else {
            set type "nucleic"
         }
      } elseif {$proteinAtomCount > 0} {
         set type "protein"
      } elseif {$nucleicAtomCount > 0} {
         set type "nucleic"
      }

      # Delete the selection.
      $proteinAtoms delete
      unset proteinAtoms
      $nucleicAtoms delete
      unset nucleicAtoms

      # If vmd couldn't tell, try to figure it out ourselves.
      if {$type == "unknown"} {

         if {$segname == ""} {
            set proteinAtoms [atomselect $molID "chain $chain and name CA"]
            set nucleicAtoms [atomselect $molID "chain $chain and name P"]
         } else {
            set proteinAtoms [atomselect $molID \
                          "chain $chain and segname \"$segname\" and name CA"]
            set nucleicAtoms [atomselect $molID \
                           "chain $chain and segname \"$segname\" and name P"]
         }
         set proteinAtomCount [$proteinAtoms num]
         set nucleicAtomCount [$nucleicAtoms num]
         if {$proteinAtomCount > 0 && $nucleicAtomCount > 0} {
            if {$proteinAtomCount > $nucleicAtomCount} {
               set type "protein"
            } else {
               set type "nucleic"
            }
         } elseif {$proteinAtomCount > 0} {
            set type "protein"
         } elseif {$nucleicAtomCount > 0} {
            set type "nucleic"
         }
         $proteinAtoms delete
         unset proteinAtoms
         $nucleicAtoms delete
         unset nucleicAtoms
      }

      return $type
   } ; # end of determineType

# --------------------------------------------------------------------------
    proc writeStructure {sequenceID filename {atomType "all"} {elements "all"} {copyUser 0} {frame now}} {

        variable sequenceIDToMolID

        # Select the atoms and write them out to the file.        
        set molID [lindex $sequenceIDToMolID($sequenceID) 0]

        # Get the selection string.
        set selectionString [getSelectionStringForElements $sequenceID $elements]

        # If we are just writing a specific atom type, append it to the selection.
        if {$atomType != "" && $atomType != "all"} {
            append selectionString " and $atomType"
        }

        #Copy user data into beta feild
        if {$copyUser == 1} {
            set atoms [atomselect $molID $selectionString frame $frame]
            set beta [$atoms get beta]
            $atoms set beta [$atoms get user]
            $atoms writepdb $filename
            $atoms set beta $beta
            $atoms delete
            unset atoms

        } else {
            # Write the atoms.
            set atoms [atomselect $molID $selectionString frame $frame]
            $atoms writepdb $filename
            $atoms delete
            unset atoms
        }
    }

    proc getSelectionStringForSequence {sequenceID} {

        variable sequenceIDToMolID

        # Select the atoms and write them out to the file.        
        set molID [lindex $sequenceIDToMolID($sequenceID) 0]
        set chain [lindex $sequenceIDToMolID($sequenceID) 1]
        set segname [lindex $sequenceIDToMolID($sequenceID) 2]

        if {$segname == ""} {
            return "chain $chain"
        } else {
            return "chain $chain and segname \"$segname\""
        }
    }

# -----------------------------------------------------------------------
    proc getSelectionStringForElements {sequenceID {elements "all"}} {

        variable sequenceIDToMolID
        variable elementToResidueMap

#puts "seqdata_vmd.tcl.getSelectionStringForElements. start.  seqID: $sequenceID, elems: $elements"
        # If we are using all elements, get the list.
        if {$elements == "all"} {
            set elements {}
            for {set i 0} {$i < [::SeqData::getSeqLength $sequenceID]} \
                                                              {incr i} {
                lappend elements $i
            }
        }

        # Select the atoms and write them out to the file. 
        set molID [lindex $sequenceIDToMolID($sequenceID) 0]
        set chain [lindex $sequenceIDToMolID($sequenceID) 1]
        set segname [lindex $sequenceIDToMolID($sequenceID) 2]

        set resIDSelection ""
        set exceptionSelection ""
        foreach element $elements {

            # Get the residue that corresponds to the element and, 
            # if valid, add it to the string.
            # resid 618 619 or (resid 620 and insertion "A")
            set residue [getResidueForElement $sequenceID $element]
            if {$residue != ""} {
                if {[llength $residue] == 1} {
                    set resID $residue
                    if {$resIDSelection == ""} {
                        set resIDSelection "resid"
                    }
                    if {$resID < 0} {
                        append resIDSelection " \"" $resID "\""
                    } else {
                        append resIDSelection " " $resID
                    }
                } else {
                    set resID [lindex $residue 0]
                    set insertion [lindex $residue 1]
                    set altloc [lindex $residue 2]
                    if {$exceptionSelection != ""} {
                        append exceptionSelection " or "
                    }
                    if {$insertion != " " && $altloc != ""} {
                        append exceptionSelection "(resid \"$resID\" and insertion \"$insertion\" and altloc \"$altloc\" \"\")"
                    } elseif {$insertion != " "} {
                        append exceptionSelection "(resid \"$resID\" and insertion \"$insertion\")"
                    } elseif {$altloc != ""} {
                        append exceptionSelection "(resid \"$resID\" and altloc \"$altloc\" \"\")"
                    } else {
                        append exceptionSelection "(resid \"$resID\" and insertion \" \")"
                    }
                }
            }
        }

        set segnameSelection ""
        if {$segname != ""} {
            set segnameSelection "segname \"$segname\" and"
        }

        if {$resIDSelection == "" && $exceptionSelection != ""} {
            return "chain $chain and $segnameSelection ($exceptionSelection)"
        } elseif {$resIDSelection != "" && $exceptionSelection == ""} {
            return "chain $chain and $segnameSelection ($resIDSelection)"
        } elseif {$resIDSelection != "" && $exceptionSelection != ""} {
            return "chain $chain and $segnameSelection ($resIDSelection or $exceptionSelection)"
        }

        return "none"
    }

# ---------------------------------------------------------------------
    # This method is called by the SeqData package whenever one of the VMD sequences has been
    # changed.
    proc sequenceChanged {changeType sequenceID {arg1 ""} {arg2 ""}} {
#        puts stderr "seqdata_vmd.tcl, sequenceChanged, type: $changeType, seqId: $sequenceID, arg1: '$arg1', arg2: '$arg2'"
        # Call the appropriate function.
        if {$changeType == "setSeq"} {
            return [setSeq $sequenceID $arg1]
        } elseif {$changeType == "removeElements"} {
            return [removeElements $sequenceID $arg1]
        } elseif {$changeType == "setElements"} {
            return [setElements $sequenceID $arg1 $arg2]
        } elseif {$changeType == "insertElements"} {
            return [insertElements $sequenceID $arg1 $arg2]
        } elseif {$changeType == "setFirstResidue"} {
            return [setFirstResidue $sequenceID $arg1]
        } else {
            return 0
        }
    }

# --------------------------------------------------------------------------
    proc setSeq {sequenceID sequence} {

        variable ::SeqData::seqs
        variable sequenceIDToMolID

        # Set the new sequence.
        seq set $sequenceID $sequence

        # Recompute the mappings.
        computeResidueElementMappings $sequenceID [lindex $sequenceIDToMolID($sequenceID) 0] [lindex $sequenceIDToMolID($sequenceID) 1] [lindex $sequenceIDToMolID($sequenceID) 2]

        return 1
    }

# -------------------------------------------------------------------------
   proc setElements {sequenceID elementIndexes newElement} {
      puts "seqdata_vmd.setElements.  This proc does nothing"    
      return 0
   }

# --------------------------------------------------------------------------
    proc removeElements {sequenceID elementIndexes} {

        variable ::SeqData::seqs
        variable sequenceIDToMolID

        # Get the sequence.
        set sequence [seq get $sequenceID]

#        puts "seqdata_vmd.removeElements. seqId: $sequenceID, elemIdx: $elementIndexes, sequence: $sequence"
        # Make sure we are only removing gaps.
        foreach elementIndex $elementIndexes {
            if {[lindex $sequence $elementIndex] != "-"} {
                return 0
            }
        }

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

#        puts "seqdata_vmd.removeElements. new sequence: $sequence"
        # Save the new sequence.
        seq set $sequenceID $sequence


#      puts -nonewline "seqdata_vmd.tcl,removeElements() colors:"
#      for {set i 0} {$i < [seq length $sequenceID]} {incr i} {
#         puts -nonewline "[seq get color $sequenceID $i] "
#      }
#      puts "\nseqdata_vmd.tcl, removeElements() after remove. [seq get $sequenceID]"

        # Recompute the mappings.
        computeResidueElementMappings $sequenceID \
                   [lindex $sequenceIDToMolID($sequenceID) 0] \
                   [lindex $sequenceIDToMolID($sequenceID) 1] \
                   [lindex $sequenceIDToMolID($sequenceID) 2]

        return 1
    }

# -------------------------------------------------------------------------
   proc insertElements {sequenceID position newElements} {

#      puts stderr "seqdata_vmd.tcl, insertElements() begin seqID: $sequenceID, pos: $position, newElems: $newElements"
      variable ::SeqData::seqs
      variable sequenceIDToMolID

      # Make sure we are only inserting gaps.
      foreach newElement $newElements {
         if {$newElement != "-"} {
             return 0
         }
      }

      # Insert the new elements.
      if {$position == "end"} {
         set position [expr [seq length $sequenceID]-1]
      }
#      puts "seqdata_vmd.tcl, insertElements() before set [seq get $sequenceID]"
#      puts -nonewline "seqdata_vmd.tcl,insertElements() colors:"
#      for {set i 0} {$i < 10} {incr i} {
#         puts -nonewline "[seq get color $sequenceID $i] "
#      }

#      puts stderr "\nseqdata_vmd.tcl, insertElements() before set."

      if {$position == 0} {
#         puts stderr "pos was 0"
         seq set $sequenceID [concat $newElements \
                                     [seq get $sequenceID $position end]]
      } else {
#         puts stderr "pos was NOT 0"
#      puts stderr "\nseqdata_vmd.tcl, insertElements() before set. id:$sequenceID, passing in '[concat [seq get $sequenceID 0 [expr $position-1]] $newElements [seq get $sequenceID $position end]]'"
         seq set $sequenceID [concat [seq get $sequenceID 0 \
                            [expr $position-1]] \
                            $newElements [seq get $sequenceID $position end]]
      }
#      puts -nonewline "seqdata_vmd.tcl,insertElements() colors:"
#      for {set i 0} {$i < [seq length $sequenceID]} {incr i} {
#         puts -nonewline "[seq get color $sequenceID $i] "
#      }
#      puts stderr "\nseqdata_vmd.tcl, insertElements() after set."
#      puts stderr "\nseqdata_vmd.tcl, insertElements() [seq get $sequenceID]"
      # Recompute the mappings.
      computeResidueElementMappings $sequenceID \
                      [lindex $sequenceIDToMolID($sequenceID) 0] \
                      [lindex $sequenceIDToMolID($sequenceID) 1] \
                      [lindex $sequenceIDToMolID($sequenceID) 2]
#      puts "seqdata_vmd.tcl, insertElements() after compute [seq get color $sequenceID 1]"

      
#      puts stderr "seqdata_vmd.tcl, insertElements() END  seqID: $sequenceID, pos: $position, newElems: $newElements"
#      for {set i [info level]} {$i > 0} {incr i -1} {
#         puts stderr "Level $i: [info level $i]"
#      }
      return 1
   }

# -------------------------------------------------------------------------
    proc setFirstResidue {sequenceID firstResidue} {

        variable ::SeqData::seqs
        variable sequenceIDToMolID

        # Save the new first residue.
        set ::SeqData::seqs($sequenceID,firstRes) $firstResidue

        # Recompute the mappings.
        computeResidueElementMappings $sequenceID [lindex $sequenceIDToMolID($sequenceID) 0] [lindex $sequenceIDToMolID($sequenceID) 1] [lindex $sequenceIDToMolID($sequenceID) 2]

        return 1
    }

    variable bondStructure
    array unset bondStructure 

    proc loadBondStructure {sequenceID} {

        variable bondStructure

        # Get the molid for the sequence.
        set molID [lindex [getMolIDForSequence $sequenceID] 0]

        if {![info exists bondStructure($sequenceID)]} {
            set bondStructure($sequenceID) 1
            set atoms [atomselect $molID [getSelectionStringForSequence $sequenceID]]
            set numberAtoms [$atoms num]
            set indexList [$atoms get index]
            set bondsList [$atoms getbonds]
            $atoms delete

            # Load the info into the array.
            for {set i 0} {$i < $numberAtoms} {incr i} {
                set bondStructure($sequenceID,[lindex $indexList $i]) [lindex $bondsList $i]
            }
        }
    }

    proc areAtLeastBondsBetween {sequenceID resid1 insertion1 atomName1 resid2 insertion2 atomName2 minNumberBonds} {

        variable bondStructure

        # Get the molid for the sequence.
        set molID [lindex [getMolIDForSequence $sequenceID] 0]

        # Get the index of the two atoms.
        set atom1 [atomselect $molID "[getSelectionStringForSequence $sequenceID] and resid $resid1 and insertion \"$insertion1\" and name $atomName1"]
        set index1 [$atom1 get index]
        $atom1 delete
        set atom2 [atomselect $molID "[getSelectionStringForSequence $sequenceID] and resid $resid2 and insertion \"$insertion2\" and name $atomName2"]
        set index2 [$atom2 get index]
        $atom2 delete

        # If both atoms are the same, return appropriately.
        if {$index1 == $index2} {
            if {$minNumberBonds >= 1} {
                return 0
            } else {
                return 1
            }
        }

        # Perform a depth first search through the network and see if we find the second atom before the cutoff.
        set atomsProcessed [list $index1]
        set listToSearch $bondStructure($sequenceID,$index1)
        set nextListToSearch {}        
        for {set currentNumberBonds 1} {$currentNumberBonds < $minNumberBonds} {incr currentNumberBonds} {

            # Search the current list.
            foreach index $listToSearch {

                # If this is the atom, return false.
                if {$index == $index2} {
                    return 0
                }

                # Add this atom to the list of ones already processed.
                lappend atomsProcessed $index

                # Otherwise, if we have at least one more level to search, add the connected nodes to the list.
                if {$currentNumberBonds < [expr $minNumberBonds-1]} {
                    foreach connectedIndex $bondStructure($sequenceID,$index) {
                        if {[lsearch -exact $atomsProcessed $connectedIndex] == -1} {
                            lappend nextListToSearch $connectedIndex
                        }
                    }
                }
            }

            # Set the search up for the next level.
            set listToSearch $nextListToSearch
            set nextListToSearch {}
        }

        return 1
    }

    proc unloadBondStructure {sequenceID} {
    }
}
