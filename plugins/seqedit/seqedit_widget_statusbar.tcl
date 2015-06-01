############################################################################
#cr
#cr            (C) Copyright 1995-2004 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: seqedit_widget_statusbar.tcl,v 1.8 2013/04/15 17:40:21 johns Exp $
#

package provide seqedit_widget 1.1
package require libbiokit 1.1

# Define the package
namespace eval ::SeqEditWidget {
    
    # Export the package namespace.
    namespace export setStatusBarText

    # The current selection notification command, if any.
    variable enableAutomaticUpdates 1
    
    # The text currently being displayed in the status bar.
    variable statusBarText
    
    # The maximum number of seqeucnes to use in a sequence comparison.
    variable maxComparisonSequences 200

    
    ############################## PUBLIC METHODS ############################
    # Methods in this section can be called by external applications.        #
    ##########################################################################

    # Sets the current text of the status bar.
    # args:     text - The text to display on the status bar.
    proc setStatusBarText {a_statusBarText {a_enableAutomaticUpdates 1}} {
        
        variable statusBarText
        variable enableAutomaticUpdates
        
        set statusBarText $a_statusBarText
        set enableAutomaticUpdates $a_enableAutomaticUpdates
    }
    
    
    
   ############################# PRIVATE METHODS ############################
   # Methods in this section should not be called by external applications. #
   ##########################################################################
    
   proc initializeStatusBar {} {
      setSelectionNotificationCommand \
                                "::SeqEditWidget::statusBarSelectionChange"
   }

# --------------------------------------------------------------------------
   proc statusBarSelectionChange {} {
      variable enableAutomaticUpdates
      variable maxComparisonSequences
#      puts "seqedit_widget_statusbar.tcl.statusBarSelectionChange.start eau: $enableAutomaticUpdates, selType: [getSelectionType], cells:[getSelectedCells]"
      if {$enableAutomaticUpdates == 1} {
         if {[getSelectionType] == "cell"} {
            set cells [getSelectedCells]
            # If the selection is within one sequence.
            if {[llength $cells] == 2} {
               # Get the sequence id and element list.
               set sequenceID [lindex $cells 0]
               set elements [lindex $cells 1]
               # If only one element is selected.
#               puts "seqedit_widget_statusbar.tcl.statusBarSelectionChange.cell cells:[getSelectedCells]; seqID: $sequenceID, elem: $elements is: [seq get $sequenceID $elements $elements]"
               if {[llength $elements] == 1} {
                  set residueIndex [string trim [join [lrange \
                             [::SeqData::getResidueForElement $sequenceID \
                             [lindex $elements 0]] 0 1] ""]]
                  if {$residueIndex != "" && $residueIndex >= 0} {
                     set add ""
                     if {[seq get $sequenceID $elements $elements] == "?" } {
                        set molInf [::SeqData::VMD::getMolIDForSequence $sequenceID]
#                        set sel [atomselect [lindex $molInf 0] "chain [lindex $molInf 1] and resid $residueIndex and ( (all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ))" ]
                        set sel [atomselect [lindex $molInf 0] "chain [lindex $molInf 1] and resid $residueIndex" ]
                        if { [$sel get resname] != ""} {
                           set add ", Residue Code: [string range [$sel get resname] 0 3]"
                        }
                        $sel delete
                     }
                     setStatusBarText "Residue: $residueIndex$add"
                     return
                  }
                  # If multiple elements are selected. 
               } elseif {[llength $elements] > 1} {
                  # See if they are contiguous.
                  set contiguous 1
                  set previousElement -1
                  foreach element $elements {
                     if {$previousElement != -1 && $element != \
                                                  [expr $previousElement+1]} {
                        set contiguous 0
                        break
                     }
                     set previousElement $element
                  }
                  # If they are, show the range.
                  if {$contiguous} {
                     set firstResidueIndex ""
                     for {set i 0} {$i < [llength $elements] && \
                                           $firstResidueIndex == ""} {incr i} {
                        set firstResidueIndex [string trim [join [lrange \
                                 [::SeqData::getResidueForElement $sequenceID \
                                 [lindex $elements $i]] 0 1] ""]]
                     }
                     set lastResidueIndex ""
                     for {set i [expr [llength $elements]-1]} {$i >= 0 && \
                                         $lastResidueIndex == ""} {incr i -1} {
                        set lastResidueIndex [string trim [join [lrange \
                              [::SeqData::getResidueForElement $sequenceID \
                              [lindex $elements $i]] 0 1] ""]]
                     }
                     if {$firstResidueIndex != "" && $lastResidueIndex != ""} {
                        setStatusBarText \
                              "Residues: $firstResidueIndex\-$lastResidueIndex"
                        return
                     }
                     # If we have less than 10, list them.
                  } elseif {[llength $elements] <= 10} {
                     set residueIndices ""
                     foreach element $elements {
                        set residueIndex [string trim [join [lrange \
                               [::SeqData::getResidueForElement $sequenceID \
                               $element] 0 1] ""]]
                        if {$residueIndex != ""} {
                           if {$residueIndices == ""} {
                              append residueIndices $residueIndex
                           } else {
                              append residueIndices ",$residueIndex"
                           }
                        }
                     }
                     if {$residueIndices != ""} {
                        setStatusBarText "Residues: $residueIndices"
                        return
                     }
                  }
               }
            }
         } elseif {[getSelectionType] == "sequence"} {
            # Get the selected sequences.
            set sequenceIDs [getSelectedSequences]
            # If we have two sequences selected, show the stats.
            set numberSequences [llength $sequenceIDs]

            if {$numberSequences == 1 } {
               if { [::SeqData::getType $sequenceIDs] == "rna" } {
                  setStatusBarText \
                   "Percent GC Content: [::SeqData::getPercentGC $sequenceIDs]"
               } else {
                  setStatusBarText ""
               }
               return
            }
            if {$numberSequences == 2} {

               if {[catch {
                  # Get some stats of the sequences.
                  set pidMatrix [::Libbiokit::getPairwisePercentIdentity \
                                                                $sequenceIDs]
                  set pid [lindex [lindex $pidMatrix 0] 1]
#                  puts "seqedit_widget_statusbar.tcl.init_statusbar, pidMatrix: $pidMatrix, pid: $pid"
                  if {[::SeqData::hasStruct [lindex $sequenceIDs 0]] == "Y" &&\
                      [::SeqData::hasStruct [lindex $sequenceIDs 1]] == "Y"} {
                     set qhMatrix [::Libbiokit::getPairwiseQH $sequenceIDs]
                     set qh [expr 1.0-[lindex [lindex $qhMatrix 0] 1]]
                     set rmsdMatrix [::Libbiokit::getPairwiseRMSD $sequenceIDs]
                     set rmsd [lindex [lindex $rmsdMatrix 0] 1]
                     setStatusBarText \
                    "QH: $qh, RMSD: $rmsd, Percent Identity: [expr $pid*100.0]"
                  } else {
                     setStatusBarText "Percent Identity: [expr $pid*100.0]"
                  }
               } errorMessage] != 0} {
                  puts "Caught error generating stats on the two sequences: $errorMessage"
                  setStatusBarText ""
               }
               return
            } elseif {$numberSequences > 2 && \
                      $numberSequences <= $maxComparisonSequences} {
               if {[catch {
                  # Calculate the PID of the sequence.
                  set pidMatrix [::Libbiokit::getPairwisePercentIdentity \
                                                                $sequenceIDs]
                  set minimumPid 1.0
                  set maximumPid 0.0
                  set totalPid 0.0
                  set numberPids 0
                  set rowIndex 0
                  foreach row $pidMatrix {
                     for {set col 0} {$col < $rowIndex} {incr col} {
                        set element [lindex $row $col]
                        if {$element < $minimumPid} {
                           set minimumPid $element
                        }
                        if {$element > $maximumPid} {
                           set maximumPid $element
                        }
                        set totalPid [expr $totalPid+$element]
                        incr numberPids
                     }
                     incr rowIndex
                  }
#                  puts "seqedit_widget_statusbar.tcl.init_statusbar, minpid: $minimumPid, maxpid: $maximumPid"
                  setStatusBarText "Percent Identity (min,average,max): [format %1.2f [expr $minimumPid*100.0]], [format %1.2f [expr ($totalPid/$numberPids)*100.0]], [format %1.2f [expr $maximumPid*100.0]]"
               } errorMessage] != 0} {
                  puts "Caught error generating stats on the selected sequences: $errorMessage"
                  setStatusBarText ""
               }
               return
            }
         }
         # If we made it here, set the bar to be empty.
         setStatusBarText ""
      }
   } ; # end of proc statusBarSelectionChange 

}
