# University of Illinois Open Source License
# Copyright 2004-2007 Luthey-Schulten Group,
# All rights reserved.
#
# $Id: seqedit_widget_select.tcl,v 1.9 2018/11/06 23:10:22 johns Exp $
#
# Developed by: Luthey-Schulten Group
#      University of Illinois at Urbana-Champaign
#      http://faculty.scs.illinois.edu/schulten/
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
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
# Author(s): Elijah Roberts, Michael Bach

package provide seqedit_widget 1.1

# Define the package
namespace eval ::SeqEditWidget {

    # Export the package namespace.
    namespace export setSelectionNotificationCommand \
                     setSelectionNotificationVariable resetSelection \
                     getSelectionType

    # The current selection notification command, if any.
    variable selectionNotificationCommands {}

    # The current selection notification variable, if any.
    variable selectionNotificationVariableNames {}

    # The current mapping of selections.
    variable selectionMap
    array unset selectionMap 

    # The map to store any mouse drag state.
    variable dragStateMap
    array unset dragStateMap 

    # The type of the last click.
    variable lastClickType ""

    #list of selected sequences
    variable selectedSequences ""

    # list of selected columns
    variable selectedColumns ""

    #list of selected ranges
    variable selectedRanges ""

    #list of selected individual elements
    variable selectedElements ""

    #global variable telling if there is nothing selected
    variable noSelections 1

    variable rowMap


    ############################## PUBLIC METHODS ############################
    # Methods in this section can be called by external applications.        #
    ##########################################################################

# -------------------------------------------------------------------------
    # Set a command to be run when the current selection has changed.
    # args:     command - The command to be executed whenever the selection has changed.
    proc setSelectionNotificationCommand {command} {

        variable selectionNotificationCommands

        set index [lsearch $selectionNotificationCommands $command]
        if {$index == -1} {
            lappend selectionNotificationCommands $command
        }
    }

# -------------------------------------------------------------------------
    # Set a command to be run when the current selection has changed.
    # args:     command - The command to be executed whenever the selection has changed.
    proc setSelectionNotificationVariable {varName} {

        variable selectionNotificationVariableNames

        lappend selectionNotificationVariableNames $varName
    }

# -------------------------------------------------------------------------
    # Resets the editor selection so that nothing is selected.
    proc resetSelection {} {

        variable groupNames
        variable groupMap
        variable selectionMap
        variable numberElements

        variable selectedSequences
        variable selectedRanges
        variable selectedElements
        variable selectedColumns
        variable noSelections

        set noSelections 1

        set selectedSequences ""
        set selectedRanges ""
        set selectedElements ""
        set selectedColumns ""

        set selectionMap(type) "none"
    }

# --------------------------------------------------------------------------
   # Gets the current selection type. Valid values are "sequence" for a
   # selection of sequences,
   # "position" for a selection of positions across every sequence, "cell"
   # for a selection of
   # specific positions within a subset of sequences, and "none" if there
   # is currently no selection.
   # return:   The current selection type as "sequence", "position", "cell",
   # or "none".
   proc getSelectionType {} {
      variable selectionMap

      if { [info exists selectionMap(type)] } {
         return $selectionMap(type)
      } else {
         return "none"
      }
   }

# --------------------------------------------------------------------------
    # Gets the currently selected sequences. If the selection is not composed
    # completely
    # of fully selected sequences, this method returns an empty list.
    # return:   A list of the sequence ids of the currently selected sequences.
    proc getSelectedSequences {} {
        variable selectedSequences
        variable selectedRanges
        variable selectedElements
        variable noSelections
        variable groupNames
        variable groupMap
        variable selectionMap



        # Make sure the selection is currently composed of sequences.
        if {$selectionMap(type) == "sequence"} {

            return $selectedSequences

            # FIXME.  Why's this stuff even here?  Just not deletd

            # Go through the selection list and get all of the currently selected sequence ids.
#            set selectedSequenceIDs {}
#            foreach groupName $groupNames {
#                set sequenceIDs $groupMap($groupName,sequenceIDs)
#                foreach sequenceID $sequenceIDs {
#                    #if {$selectionMap($sequenceID,h) == 1} {
#                    #    lappend selectedSequenceIDs $sequenceID
#                    #}
#                }
#            }
#
#            return $selectedSequenceIDs
        }

        return {}
    }

# --------------------------------------------------------------------------

    proc setSelectedSequences {sequenceIDs {notifyListeners 1}} {
        #puts "Selecting sequences $sequenceIDs"
        deselectAllSequences
        foreach sequenceID $sequenceIDs {
            if {$sequenceID != ""} {
                setSelectedSequence $sequenceID 1 0
            }
        }

        # Notify any selection listeners.
        if {$notifyListeners} {
            notifySelectionChangeListeners
        }
    }

# --------------------------------------------------------------------------

   # Set the selection status of a sequence in the editor.
   # args:     sequenceID - The id of the sequence to select.
   #    add - (default 0) If the sequence should be added to the selection.
   #    flip - (default 0) 1 if selection for specified sequence should 
   #           be flipped.
   proc setSelectedSequence {sequenceID {add 0} {flip 0}} {
      variable selectedSequences
      variable selectedRanges
      variable selectedElements
      variable noSelections

      variable groupNames
      variable groupMap
      variable selectionMap
#      puts "selecting sequence $sequenceID, add: $add, flip: $flip, currentSelectedSeqs: $selectedSequences"


      # If we are not already selecting sequences, reset the map.
      if {$selectionMap(type) != "sequence" && $selectionMap(type) != "none"} {
         resetSelection
         redraw
      }

        # Set that we are selecting sequences.
      #puts "orig Selectedsequences = $selectedSequences"
      #puts "adding sequence $sequenceID"
      set selectionMap(type) "sequence"

      #set sequence [getSequenceInGroup [lindex $sequenceID 0] [lindex $sequenceID 1]]
      #set sequence [getSequenceInGroup [lindex $groupNames [lindex $sequenceID 0]] [lindex $sequenceID 1]]
      set sequence $sequenceID
      #puts "SEQUENCE = $sequence"
      set redrawSeqs $sequence

      set id [lsearch $selectedSequences $sequence]
      if { $id == -1 } {
         if { $add } {
            #puts "adding $sequence"
            set selectedSequences [lappend selectedSequences $sequence]
         } elseif { $flip } {
            #puts "Flipping"
            set selectedSequences [lappend selectedSequences $sequence]
            set redrawSeqs [concat $redrawSeqs $sequence]
         } else {
            #puts "setting only $sequence"
            set redrawSeqs [concat $redrawSeqs $selectedSequences]
            set selectedSequences $sequence

         }
      } else {
         if { $flip } {
            set selectedSequences [lreplace $selectedSequences $id $id]
         } else {
            #puts "setting only $sequence"
            set redrawSeqs [concat $redrawSeqs $selectedSequences]
            set selectedSequences $sequence
         }
      }

      set noSelections 0
      #puts "Selected sequences = $selectedSequences"
      #puts "redrawing sequences $redrawSeqs"
      redraw $redrawSeqs
#      puts "selecting sequence END $sequenceID, add: $add, flip: $flip, currentSelectedSeqs: $selectedSequences"
      return
   }

# --------------------------------------------------------------------------
   proc deselectAllSequences {} {

      variable groupNames
      variable groupMap
      variable selectionMap

      variable selectedSequences

      set selectedSequences ""
      redraw
      return
   }

# --------------------------------------------------------------------------
   # Gets the currently selected positions. If the selection is not composed completely
   # of fully selected positions, this method returns an empty list.
   # return:   A list of the indices of the currently selected positions.
   proc getSelectedPositions {} {
      variable selectedColumns

      return $selectedColumns
   }

# --------------------------------------------------------------------------
# Set selection status of a specific position of each sequence in the editor.
# args:     position - The position that should be selected in each sequence.
#           add - (default 0) If the position should be added to the selection.
#           flip - (default 0) 1 if the selection for the specified
#           position should be flipped.
   proc setSelectedPosition {position {add 0} {flip 0}} {

      variable groupNames
      variable groupMap
      variable selectionMap
      variable numberElements

      variable selectedSequences
      variable selectedColumns
      variable selectedRanges
      variable selectedElements
      variable noSelections

      set column $position

      # If we are not already selecting positions, reset the map.
      if {$selectionMap(type) != "position" && $selectionMap(type) != "none"} {
         resetSelection
         redraw
      }
      set selectionMap(type) "position"
      set selectionMap(firstPosition) $position

      #puts "COLUMN = $column"
      set redrawCols $column

      set id [lsearch $selectedColumns $column]
      if { $id == -1 } {
         if { $add } {
            #puts "adding $column"
            lappend selectedColumns [lappend selectedColumns $column]
         } elseif { $flip } {
            #puts "Flipping"
            set selectedColumns [lappend selectedColumns $column]
            set redrawCols [concat $redrawCols $column]
         } else {
            #puts "setting only $column"
            set redrawCols [concat $redrawCols $selectedColumns]
            set selectedColumns $column
         }
      } else {
         if { $flip } {
            set selectedColumns [lreplace $selectedColumns $id $id]
         } else {
            #puts "setting only $column"
            set redrawCols [concat $redrawCols $selectedColumns]
            set selectedColumns $column
         }
      }

      #puts "selectedColumns = $selectedColumns"
      #puts "redrawCols = $redrawCols"
      set noSelections 0
      redraw

      return
   }

# --------------------------------------------------------------------------
   # Set the selection status of a specific range of positions of each
   # sequence in the editor.
   # args: startingPosition - starting position within the sequences to select.
   #       endingPosition - The ending position within the sequences to select.
   proc setSelectedPositionRange {startingPosition endingPosition} {
#      puts "seqedit_widget_select.tcl:setSelectedPositionRange().start: startPos: $startingPosition, endPos: $endingPosition"

      variable groupNames
      variable groupMap
      variable selectionMap
      variable numberElements

      variable selectedSequences
      variable selectedColumns
      variable selectedRanges
      variable selectedElements
      variable noSelections
      #lappend selectedColumns {$startingPosition $endingPosition}


      # If we are not already selecting positions, reset the map.
      if {$selectionMap(type) != "position" && $selectionMap(type) != "none"} {
         resetSelection
      }

      for { set i $startingPosition} { $i <= $endingPosition } { incr i } {
         set selectedColumns [lappend selectedColumns $i]
      }

      # Set that we are selecting positions.
      set selectionMap(type) "position"
      set selectionMap(firstPosition) $startingPosition
      set selectionMap(lastPosition) $endingPosition

      set noSelections 0
      redraw

      return
   }

# -------------------------------------------------------------------------
    # Gets the currently selected cells. This method return all of
    # the cells that are currently
    # selected regardless of whether they were selected directly or
    # through a row or column selection.
    # return:   The currently selected cells in the format:
    #           {seq1 {pos1 pos2 ...} seq2 {pos1 pos2 ...} ...}
   proc getSelectedCells {} {
      variable groupNames
      variable groupMap
      variable selectionMap

      variable selectedSequences
      variable selectedColumns
      variable selectedRanges
      variable selectedElements
      variable noSelections
#      puts " seqedit_widget_select.tcl.getSelectedCells - start nosel: $noSelections -------"

      if { $noSelections } {
         return ""
      }

      set selectedCells {}

      # look for completely selected sequences
      foreach seq $selectedSequences {
         set selectedPositions {}
         for { set i 0 } { $i < [::SeqData::getSeqLength $seq] } { incr i } {
            lappend selectedPositions $i
         }
         lappend selectedCells $seq
         lappend selectedCells $selectedPositions
      }


      # look for completely selected columns
#      puts "seqedit_widget_select.getSelectedCells. after selSeq: $selectedCells, selColumns: $selectedColumns"
      if {[llength $selectedColumns] > 0} {
         # ok, we have some columns that are selected

         # sort them, so we have them in order
         set selectedColumns [lsort -integer -unique $selectedColumns]

         foreach seq [getSequences] {
            # find the max position for this specific sequence
            set maxPos [expr [SeqData::getSeqLength $seq]-1]

            # is the first column selected affecting this sequence?
            if {[lindex $selectedColumns 0] < $maxPos} {

               # we know that SOME cells in this sequence are selected
               lappend selectedCells $seq

               # is it just all of them?
               if {$maxPos >= [lindex $selectedColumns end] } {
                  lappend selectedCells $selectedColumns
               } else {
                  set tmpList [list]
                  foreach col $selectedColumns {
                     if { $col <= $maxPos } {
                        lappend tmpList $col
                     }
                  }
                  lappend selectedCells $tmpList

               }
               # now we have to make a list of columns that actually 
               # exist in this sequence that are selected





            } ;# end if on having residues selected for this sequence
         } ;# end loop over sequences
      } ;# end if on having columns selected

#      puts "seqedit_widget_select.getSelectedCells. after selSeq: $selectedCells, selColumns: $selectedColumns"









#      foreach col $selectedColumns {
#         puts "col is  $col"
#         foreach seq [getSequences] {
#            set maxPos [expr [SeqData::getSeqLength $seq]-1]
#            # we need to check and see if this sequence has this column listed
#            if { $col <= $maxPos } {
#               lappend selectedCells $seq
#               lappend selectedCells [list $col]
#            }
#         }
#      }




#      puts "seqedit_widget_select.getSelectedCells. after selCol: $selectedCells, selRanges: $selectedRanges"

      foreach  range $selectedRanges {
         ##puts "range = ($s1, $e1), ($s2, $e2)"
         set startingSeq [lindex $range 0]
         set endingSeq [lindex $range 2]
         if { $startingSeq > $endingSeq } {
            set tmp $endingSeq
            set endingSeq $startingSeq
            set startingSeq $tmp
         }
         set startCol [lindex $range 1]
         set endCol [lindex $range 3]
         if { $startCol > $endCol } {
            set tmp $endCol
            set endCol $startCol
            set startCol $tmp
         }
         #puts "range = ($startingSeq, $startCol), ($endingSeq, $endCol)"

         # Assuming the sequence is long enough to have the entire range
         # just set cols to all the numbers  (We'll adjust this later
         # on a sequence by sequence basis
         set cols {}
         for { set j $startCol }   { $j <= $endCol } { incr j } {
            lappend cols $j
         }

         set tmpList {}
         set seqList [getSequenceIDsInRows $startingSeq $endingSeq]
#         puts "seqedit_widget_select.tcl.getSelectedCells. seqList: $seqList"
         foreach currSeq $seqList {
            set maxPos [expr [SeqData::getSeqLength $currSeq]-1]
#            puts "seqedit_widget_select.tcl.getSelectedCells. $currSeq pos: $maxPos"
            if { $startCol <= $maxPos } {
               lappend tmpList $currSeq

               # for this particular sequence, we need to figure out
               # whether or not all the columns exist
               if { $endCol <= $maxPos } {
                  lappend tmpList $cols
               } else {
                  # we need to make up a different $cols
                  set tmpCols {}
                  for { set j $startCol }   { $j <= $maxPos } { incr j } {
                     lappend tmpCols $j
                  }
                  lappend tmpList $tmpCols
               }
            }

         }
#         puts "seqedit_widget_select.tcl.getSelectedCells. new tmpList: $tmpList"

         set selectedCells [concat $selectedCells $tmpList]


         # at this point, we know that startingSeq refers to the sequence
         # in the upper left
         # of the display, and endingSeq is a sequence below that.
         # now we need to translate this into the ordering
         # that is on the screen
#         for { set i $startingSeq } { $i <= $endingSeq } { incr i } {
#}
#         set sRow [determineRowFromSequence $startingSeq]
#         set eRow [determineRowFromSequence $endingSeq]
#         if { $eRow < $sRow } {
#            set tmp $eRow
#            set eRow $sRow
#            set sRow $tmp
#         }
#
##         puts "before loop. startingSeq:$startingSeq, endingSeq:$endingSeq, srow:[determineRowFromSequence $startingSeq], erow:[determineRowFromSequence $endingSeq]"
#
#         set tmpList {}
#         for { set i $sRow } { $i <= $eRow } { incr i } {
##            puts "i:$i, seq(i):[determineSequenceFromRow $i]"
#            set group [lindex $groupNames \
#                              [lindex [determineSequenceFromRow $i] 0]]
#            set seqRow [lindex [determineSequenceFromRow $i] 1]
#            if { $seqRow != -1} {
#
#               set currSeq [getSequenceInGroup $group $seqRow]
#               set maxPos [expr [SeqData::getSeqLength $currSeq]-1]
##               puts "seq: $currSeq, maxPos: $maxPos"
#
#               if { $startCol <= $maxPos } {
#                  lappend tmpList $currSeq
#
#                  # for this particular sequence, we need to figure out
#                  # whether or not all the columns exist
#                  if { $endCol <= $maxPos } {
#                     lappend tmpList $cols
#                  } else {
#                     # we need to make up a different $cols
#                     set tmpCols {}
#                     for { set j $startCol }   { $j <= $maxPos } { incr j } {
#                        lappend tmpCols $j
#                     }
#                     lappend tmpList $tmpCols
#                  }
#               }
#            }
#         }
#         puts "seqedit_widget_select.tcl.getSelectedCells. old tmpList: $tmpList"

#         set selectedCells [concat $selectedCells $tmpList]

      }

#      puts " getSelectedCells - before last foreach $selectedCells ---"
      foreach cell $selectedElements {
#         puts "checking to see isSelected [lindex $cell 0] [list [lindex $cell 1]]"
         if { 0 == [isElemInList [lindex $cell 0] [list [lindex $cell 1]] $selectedCells] } {
            lappend selectedCells [lindex $cell 0]
            lappend selectedCells [list [lindex $cell 1]]
         }
      }

#      puts " ------- seqedit_widget_select.tcl.getSelectedCells - end $selectedCells -------"
      return $selectedCells

    } ;# end of getSelectedCells

# -------------------------------------------------------------------------
   # eList is a list where elem 0, 2, 4, etc are the sequence
   # and elem 1, 3, 5 are elements in a sequence.  elements could
   # be a list themselves
   # example eList:   4 6 5 {0 2 3} 7 10
   proc isElemInList { currseq currelem eList } {
      foreach {cSeq cElem} $eList {
         if { $cSeq == $currseq } {
            if {[lsearch -exact $cElem $currelem] != -1 } {
               return 1
            }
         }
      }
      return 0
   }

# -------------------------------------------------------------------------
   # return the selected cells, but combine them by sequence.  For instance,
   # if we have:   0 2 0 3   in our selected cells, this will return 0 {2 3}
   #
   proc getSelectedCellsCombinedBySequence {} {
      set sel [::SeqEditWidget::getSelectedCells]

#      puts "seqedit_widget_select.tcl.getSelectedCellsCombinedBySequence.start sel: $sel"
      array unset comb 
      # set up our array where each sequence has the list of selected elements
      for {set i 0} {$i < [llength $sel]} {incr i 2} {
         # if we have a 'comb' array entry for this sequence...
         if { [info exists comb([lindex $sel $i])] } {
            # then we need to just add the existing elements onto that one
            # we have play the games with the eval concat to handle the
            # situation of 0 {1 2 3} 0 {7 8}.  Without the eval concat
            # we end up with 0 {1 2 3 {7 8}}
            set comb([lindex $sel $i]) [eval concat \
                 [lappend comb([lindex $sel $i]) [lindex $sel [expr $i+1]]] ]
#            set comb([lindex $sel $i]) [eval concat comb([lindex $sel $i])]
         } else {
            # otherwise, we have to create a new one
            set comb([lindex $sel $i]) [lindex $sel [expr $i+1]]
         }
      }

      # go through the array, adding things onto a list that we'll return
      set selCells ""
      foreach {key val} [array get comb] {
#         puts "adding $key : $val"
         lappend selCells $key
         lappend selCells [lsort -integer $val]
      }
      return $selCells
   }

# -------------------------------------------------------------------------
    # Set the selection status of a specific cell in the editor.
    # args:  sequenceID - The sequence that contains the cell to select.
    #        position - The position within the sequence of the cell to select.
    #        add - (default 0) If the cell should be added to the selection.
    #        flip - (default 0) 1 if the selection for the specified
    #               cell should be flipped.
   proc setSelectedCell {sequenceID position {add 0} {flip 0} \
                                         {doredraw 1} {notify 1}} {
      variable groupNames
      variable groupMap
      variable selectionMap
      variable numberElements

      variable selectedSequences
      variable selectedColumns
      variable selectedRanges
      variable selectedElements
      variable noSelections

#puts "seqedit_widget_select.tcl: setSelectedCell() seqID: $sequenceID, pos: $position add:$add, flip:$flip, doredraw:$doredraw, notify:$notify, type: $selectionMap(type) selSeq: $selectedSequences, selCol: $selectedColumns, selRng: $selectedRanges, selElem: $selectedElements"

      # If we are not already selecting elements, reset the map.
      if {$selectionMap(type) != "cell" && $selectionMap(type) != "none"} {
         resetSelection
         redraw
      }

      # declare that we are selecting elements.
      set selectionMap(type) "cell"

      set redrawSeqs $sequenceID
      foreach cell $selectedElements {
         lappend redrawSeqs [lindex $cell 0]
      }

      foreach { range } $selectedRanges {
         set startRngRow [lindex $range 0]
         set endRngRow [lindex $range 2]

         set redrawSeqs [concat $redrawSeqs [getSequenceIDsInRows $startRngRow $endRngRow]]

#         set start [determineRowFromSequence $startRngRow]
#         set end [determineRowFromSequence $endRngRow]
#         if { $start > $end } {
#            set start [determineRowFromSequence $endRngRow]
#            set end [determineRowFromSequence $startRngRow]
#         }
#         for { set i $start } { $i <= $end } { incr i } {
#            set group [lindex $groupNames [lindex [determineSequenceFromRow $i] 0]]
#            set seq [lindex [determineSequenceFromRow $i] 1]
#            set rowSeq [getSequenceInGroup $group $seq]
#            lappend redrawSeqs $rowSeq
#         }

      }

      set cell [list $sequenceID $position]

      set id [lsearch $selectedElements $cell]
#      puts "seqedit_widget_select.tcl.setSelectedCell. elems: $selectedElements cell: $cell, id: $id"
      #set redrawElements $cell
      if { $id == -1 } {
         if { $add } {
            #puts "adding $cell"
            set selectedElements [lappend selectedElements $cell]
         } elseif { $flip } {
            #puts "Flipping"
            set selectedElements [lappend selectedElements $cell]
            #set redrawElements [concat $redrawElements $cell]
         } else {
            #puts "setting only $cell"
            #set redrawElements [concat $redrawElements $selectedElements]
            set selectedRanges ""
            set selectedElements [list $cell]

         }
      } else {
         if { $flip } {
            set selectedElements [lreplace $selectedElements $id $id]
         } else {
            #puts "setting only $cell"
            #set redrawElements [concat $redrawElements $selectedElements]
            set selectedElements [list $cell]
         }
      }

      # set the namespace variable that says that we now have a selection
      set noSelections 0


#     puts "seqedit_widget_select.tcl.setSelectedCell. elems: $selectedElements"
      if { $doredraw } {
      #puts "redrawing sequences $redrawSeqs"
         redraw $redrawSeqs
      }

      return
   } ; # end of setSelectedCell

# -------------------------------------------------------------------------
    # Set the selection to be a specific range of cells.
    # args: startingSequenceID - first sequence that contains cells to select.
    #       endingSequenceID - last sequence that contains cells to select.
    #       startingPosition - starting pos within sequences of cells to select.
    #       endingPosition - ending pos within sequences of cells to select.
   proc setSelectedCellRange {startingSequenceID endingSequenceID \
                               startingPosition endingPosition {add 0}} {
      variable groupNames
      variable groupMap
      variable selectionMap
      variable numberElements

      variable selectedSequences
      variable selectedColumns
      variable selectedRanges
      variable selectedElements
      variable noSelections

#   puts "seqedit_widget_select.tcl: setSelectedCellRange ($startingSequenceID, $startingPosition) ($endingSequenceID, $endingPosition), groupNames: $groupNames"
      # If we are not already selecting elements, reset the map.
      if {$selectionMap(type) != "cell" && $selectionMap(type) != "none"} {
         resetSelection
         redraw
      }

      # Set that we are selecting elements.
      set selectionMap(type) "cell"

      set redrawSeqs ""
      foreach range $selectedRanges {
         #set range [list $s1 $e1 $s2 $e2]

         set redrawSeqs [concat $redrawSeqs [getSequenceIDsInRows [lindex $range 0] [lindex $range 2]]]

#         set start [determineRowFromSequence [lindex $range 0]]
#         set end [determineRowFromSequence [lindex $range 2]]
#
#         if { $start > $end } {  ;# reverse the two
#            set tmp $end
#            set end $start
#            set start $tmp
#         }
#
#         for { set i $start } { $i <= $end } { incr i } {
#            #lappend redrawSeqs $i
#            set group [lindex $groupNames \
#                                    [lindex [determineSequenceFromRow $i] 0]]
#            set seq [lindex [determineSequenceFromRow $i] 1]
#            lappend redrawSeqs [getSequenceInGroup $group $seq]
#         }
      }

      set id [lsearch $selectedRanges [list $startingSequenceID \
                        $startingPosition $endingSequenceID $endingPosition]]
      set coords [list $startingSequenceID $startingPosition \
                        $endingSequenceID $endingPosition]

      if { $id == -1 } {
         if { $add } {
            #puts "Adding range"
            lappend selectedRanges $coords
         } else {
            #puts "setting range"
            set selectedRanges [list $coords]
            set selectedCells ""
         }
      } else {
         if { !$add } {
            #puts "setting range"
            set selectedRanges [list $coords]
            set selectedCells ""
         }
      }

      set noSelections 0



#      set start [determineRowFromSequence $startingSequenceID]
#      set end [determineRowFromSequence $endingSequenceID]
#      if { $start > $end } {  ;# reverse the two
#         set tmp $end
#         set end $start
#         set start $tmp
#      }
#      for { set i $start } { $i <= $end } { incr i } {
#         set group [lindex $groupNames [lindex [determineSequenceFromRow $i] 0]]
#         set seq [lindex [determineSequenceFromRow $i] 1]
#         puts "seqedit_widget_select.tcl:setSelCellRange: i:$i, group:$group, seq:$seq"
#         if { $seq != -1 } {
#            set rowSeq [getSequenceInGroup $group $seq]
##            puts "seqedit_widget_select.tcl:setSelCellRange: i:$i, group:$group, seq:$seq, rowSeq:$rowSeq"
#            lappend redrawSeqs $rowSeq
#         }
#      }
##      puts "seqedit_widget_select.tcl:setSelCellRange: redrawSeqs: $redrawSeqs"
#
#      redraw $redrawSeqs


     redraw

      return
   }

# -------------------------------------------------------------------------
############################# PRIVATE METHODS #################################
# Methods in this section should only be called from this package.            #
###############################################################################

# ---------------------------------------------------------------------------
   proc notifySelectionChangeListeners {} {

      variable selectionNotificationCommands
      variable selectionNotificationVariableNames

      if {$selectionNotificationCommands != {}} {
         foreach selectionNotificationCommand $selectionNotificationCommands {
            $selectionNotificationCommand
         }
      }
      if {$selectionNotificationVariableNames != {}} {
         foreach selectionNotificationVariableName \
                                        $selectionNotificationVariableNames {
            set $selectionNotificationVariableName 1
         }
      }
   } ;# end of notifySelectionChangeListeners

# ---------------------------------------------------------------------------
   # Handle clicks on the row header.
   proc click_rowheader {x y type} {
#      puts "seqedit_widget_select.tcl. click_rowheader type = $type, x:$x,y:$y"
      variable groupNames
      variable selectionMap
      variable dragStateMap
      variable lastClickType

      # If this was a release and we are dragging, consider it a drop.
      if {$type == "release" && [info exists dragStateMap(startedDragging)] \
                                      && $dragStateMap(startedDragging) == 1} {
         drop_rowheader $x $y
         set lastClickType $type
      } else {

         # Get the row that was clicked on.
         set row [determineRowFromLocation $x $y]
         if {$row != -1} {

            # Get the sequence that is in the row.
            set sequence [determineSequenceFromRow $row]

            # Make sure there is a sequence in the row.
            if {$sequence != {} } {

               # Make sure it wasn't a group header.
               if {[lindex $sequence 1] != -1} {

                  # See if this sequence is already selected.
                  set sequenceIsSelected 0
                  set sequenceIsSelected [isSequenceSelected $sequence]

                  # If this wasn't a release, save dragging start information.
                  if {$type != "release"} {
                     array unset dragStateMap 
                     set dragStateMap(type) "rowheader"
                     set dragStateMap(startingRow) $row
                     set dragStateMap(startedDragging) 0
                     set dragStateMap(destinationRow) ""
                     set dragStateMap(insertionMarkerID) ""
                  }

                  # If the shift key was down for the click, select all of the rows in between the selections.
                  if {$type == "shift"} {
                     if {$selectionMap(type) == "sequence"} {
                        set sequenceIDs [getSequencesInGroups \
                                 [lindex $selectionMap(startSequence) 0] \
                                 [lindex $selectionMap(startSequence) 1] \
                                 [lindex $sequence 0] \
                                 [lindex $sequence 1]]
                        set add 0
                        foreach sequenceID $sequenceIDs {
                           setSelectedSequence $sequenceID $add 0
                           set add 1
                        }
                        notifySelectionChangeListeners
                     }

                  # Else if the control key was down, flip the selection.
                  } elseif {$type == "control"} {
#puts "seqedit_widget_select.tcl. click_rowheader.  control"
                     set selectionMap(startSequence) $sequence
                     setSelectedSequence [getSequenceInGroup \
                               [lindex $groupNames [lindex $sequence 0]] \
                               [lindex $sequence 1]] 0 1
                     notifySelectionChangeListeners

                     # Else if this was a release and last click was a 
                     # normal click and we are on a selected sequence, 
                     # set the selection to this sequence.
                  } elseif {$type == "release" && \
                           [info exists dragStateMap(startedDragging)] && \
                           $dragStateMap(startedDragging) == 0 && \
                           $lastClickType == "normal" && \
                           $sequenceIsSelected == 1} {
                     set selectionMap(startSequence) $sequence
                     setSelectedSequence [getSequenceInGroup \
                                  [lindex $groupNames [lindex $sequence 0]] \
                                  [lindex $sequence 1]] 0 0
                     notifySelectionChangeListeners
                     setSelectedSequence [getSequenceInGroup \
                                 [lindex $groupNames [lindex $sequence 0]] \
                                 [lindex $sequence 1]] 0 0
                  } elseif {$type == "release"} {
                     if { $lastClickType != "shift" && \
                          $lastClickType != "control" } {
                        set selectionMap(startSequence) $sequence
                        setSelectedSequence [getSequenceInGroup \
                               [lindex $groupNames [lindex $sequence 0]] \
                               [lindex $sequence 1]] 0 0
                     }
                     notifySelectionChangeListeners

                  # Otherwise it was just a normal click on an 
                  # unselected sequence, set the selection to this one sequence.
                  } elseif {$type == "normal" && $sequenceIsSelected == 0} {
                     if { [llength [getSelectedSequences]] > 1 } {
                        notifySelectionChangeListeners
                     } else {
                        set selectionMap(startSequence) $sequence

                        setSelectedSequence [getSequenceInGroup \
                              [lindex $groupNames [lindex $sequence 0]] \
                              [lindex $sequence 1]] 0 0
                        notifySelectionChangeListeners
                     }

                  }

                  # Set the last click type.
                  set lastClickType $type
               } else {

                  # If this was a control click on a header row, 
                  # pretend it was a right-click on a Mac.
                  if {$type == "control"} {
                     rightclick_rowheader $x $y
                  }
               }
            }
         }
      }
   } ;# end of click_rowheader

# ---------------------------------------------------------------------------
    # Handle drags on the row header.
    proc drag_rowheader {x y} {
        #puts "DRAGGING ROWS"
        variable editor
        variable groupNames
        variable groupMap
        variable numberRows
        variable firstGroup
        variable firstSequence
        variable cellColorForeground
        variable columnObjectMap
        variable dragStateMap

        # See if we are really dragging a row header.
        if {[info exists dragStateMap(type)] != 0 && $dragStateMap(type) == "rowheader"} {

            # Get the row that is being dropped onto.
            set destinationRow [determineRowFromLocation $x $y]

            # See if we are dropping onto a row.
            if {$destinationRow != -1 && ($dragStateMap(startingRow) != $destinationRow || $dragStateMap(startedDragging) == 1)} {

                set dragStateMap(startedDragging) 1
                set x1 $columnObjectMap(h,$destinationRow.x1)
                set x2 $columnObjectMap(h,$destinationRow.x2)
                set y1 $columnObjectMap(h,$destinationRow.y1)
                set y2 $columnObjectMap(h,$destinationRow.y2)

                # See if we are going above or below the new row.
                if {[expr $y-$y1] <= [expr $y2-$y]} {
                    set destinationPosition "before"
                } else {
                    set destinationPosition "after"
                }

                # If the destination has changed, update it.
                if {$dragStateMap(destinationRow) != $destinationRow || $dragStateMap(destinationPosition) != $destinationPosition} {

                    # Get the sequence that is in the destination row.
                    set destinationSequence [determineSequenceFromRow $destinationRow]

                    # Make sure there is a sequence in the destination row.
                    if {$destinationSequence != {}} {

                        # Set the new destination variables.
                        set dragStateMap(destinationRow) $destinationRow
                        set dragStateMap(destinationPosition) $destinationPosition

                        # If we already have a marker, delete it.
                        if {$dragStateMap(insertionMarkerID) != ""} {
                            $editor delete $dragStateMap(insertionMarkerID)
                        }

                        # Create a new marker.
                        if {$destinationPosition == "before"} {
                            set dragStateMap(insertionMarkerID) [$editor create line $x1 $y1 $x2 $y1 -width 2 -fill $cellColorForeground]
                        } else {
                            set dragStateMap(insertionMarkerID) [$editor create line $x1 [expr $y2+1] $x2 [expr $y2+1] -width 2 -fill $cellColorForeground]
                        }
                    }
                }
            }

            # See if we should try to scroll the screen up.
            if {$destinationRow == 0 || ($destinationRow == -1 && $y <= $columnObjectMap(h,0.y1))} {

                # If the screen can be scrolled up at all, scroll it up.
                if {($firstGroup == 0 && $firstSequence > -1) || ($firstGroup > 0)} {
                    scroll_vertical scroll -1 unit
                }

            # See if we should try to scroll the screen down.
            } elseif {$destinationRow == [expr $numberRows-1] || ($destinationRow == -1 && $y >= $columnObjectMap(h,[expr $numberRows-1].y1))} {

                # If the screen can be scrolled down at all, scroll it down.
                set lastSequence [determineSequenceFromRow [expr $numberRows-2]]
                if {$lastSequence != {}} {
                    set lastGroupIndex [lindex $lastSequence 0]
                    set lastSequenceIndex [lindex $lastSequence 1]
                    if {($lastGroupIndex < [expr [llength $groupNames]-1]) || ($lastGroupIndex == [expr [llength $groupNames]-1] && $lastSequenceIndex < [expr $groupMap([lindex $groupNames $lastGroupIndex],numberSequences)-1])} {
                        scroll_vertical scroll 1 unit
                    }
                }
            }
        }
    }

# ---------------------------------------------------------------------------
   # Handle drops on the row header.
   proc drop_rowheader {x y} {

      variable editor
      variable dragStateMap
      variable groupNames

      # See if we were really dragging something.
      if {[info exists dragStateMap(type)] != 0 && \
             $dragStateMap(type) == "rowheader" && \
             $dragStateMap(startedDragging) == 1} {

         # If we have a marker, delete it.
         if {$dragStateMap(insertionMarkerID) != ""} {
            $editor delete $dragStateMap(insertionMarkerID)
         }

         # Get sequence that is in dest row and make sure it is not a grouping.
         set sequence [determineSequenceFromRow $dragStateMap(destinationRow)]
         set groupIndex [lindex $sequence 0]
         set sequenceIndex [lindex $sequence 1]
         if {$dragStateMap(destinationPosition) == "after"} {
            incr sequenceIndex
         }
         if {$dragStateMap(destinationPosition) == "before" && \
                                                $sequenceIndex == -1} {
            incr groupIndex -1
            set sequenceIndex end
         }
         if {$groupIndex < 0} {
            set groupIndex 0
            set sequenceIndex 0
         }

         # Get the current selection.
         set selectedSequenceIDs [getSelectedSequences]
         moveSequences $selectedSequenceIDs [lindex $groupNames \
                                               $groupIndex] $sequenceIndex
         validateScrollRange 0 1
         setScrollbars
         redraw
      }

      # Remove everything from the drag state.
      array unset dragStateMap 
   }

# ---------------------------------------------------------------------------
   # Handle clicks on the column header.
   proc click_columnheader {x y type} {
      #puts "CLICKCOL type = $type"
      variable dragStateMap
      variable selectionMap

      # Get the row that was clicked on.
      set column [determineColumnFromLocation $x $y]
      if {$column != -1} {

         # Get the sequence that is in the row.
         set position [determinePositionFromColumn $column]

         # Make sure there is an element in the row.
         if {$position != -1} {

            # Save the drag state in case the user tries to drag the selection.
            set dragStateMap(type) "position"
            set dragStateMap(startingColumn) $column
            set dragStateMap(startingPosition) $position
            set dragStateMap(destinationColumn) $column
            set dragStateMap(destinationPosition) $position
            set dragStateMap(startedDragging) 0

            # If the shift key was down for the click, select all of the rows in between the selections.
            if {$type == "shift"} {
               #puts "COLSHIFT"
               #if {$selectionMap(type) == "position" && [info exists selectionMap(lastPosition)] != 0} {}
               if { $selectionMap(type) == "position" } {
                  set p2 $selectionMap(lastPosition)
                  set p1 $position
                  if {$selectionMap(lastPosition) < $position} {
                     set tmp $p2
                     set p2 $p1
                     set p1 $tmp
                  }
                  #puts "p1, p2 = $p1, $p2"
                  setSelectedPositionRange $p1 $p2
                  notifySelectionChangeListeners
               }
                # Else if the control key was down, flip the selection.
            } elseif {$type == "control"} {
               setSelectedPosition $position 0 1
               notifySelectionChangeListeners
               if {[info exists selectionMap(lastPosition)] != 0} {
                  unset selectionMap(lastPosition)
               }

            # Otherwise it was just a normal click, set the selection to this one sequence.
            } else {
               setSelectedPosition $position 0 0
               notifySelectionChangeListeners
               set selectionMap(lastPosition) $position
            }
         }
      }
   } ; #end of click_columnheader

# --------------------------------------------------------------------------
   # Handle drags on the row header.
   proc move_columnheader {x y} {

      variable dragStateMap

      # See if we are really dragging a row header.
      if {[info exists dragStateMap(type)] != 0 && \
                                     $dragStateMap(type) == "position"} {

         # Get the that is being dragged onto.
         set destinationColumn [determineColumnFromLocation $x $y]

         # See if we are dragging onto a different row or column.
         if {$destinationColumn != -1 && \
                      $dragStateMap(destinationColumn) != $destinationColumn} {

            # Save that we are dragging.
            set dragStateMap(startedDragging) 1

            # Set the new destination variable.
            set dragStateMap(destinationColumn) $destinationColumn

            # Get the position that is in the destination column.
            set destinationPosition \
                              [determinePositionFromColumn $destinationColumn]
            set dragStateMap(destinationPosition) $destinationPosition

            # Make sure there is a real position in the column.
            if {$destinationPosition != -1} {

               # Make sure we have the starting and ending ordered correctly.
               if {$dragStateMap(startingColumn) < $destinationColumn} {
                  set p1 $dragStateMap(startingPosition)
                  set p2 $destinationPosition
               } else {
                  set p2 $dragStateMap(startingPosition)
                  set p1 $destinationPosition
               }

               # Select all of the cells between the start and the end point.
               setSelectedPositionRange $p1 $p2
            }
         }
      }
   } ;# end of move_columnheader

# ------------------------------------------------------------------------
   proc release_columnheader {x y} {
      variable dragStateMap
      variable selectedColumns

#      puts "seqedit_widget_select.tcl.release_columnheader: begin.  selCols: $selectedColumns"
#      parray dragStateMap
#      set column [determineColumnFromLocation $x $y]

      # next 4 lines remove duplicate entries in selectedColumns
      # I'm not sure if this is the fastest way to do this, but
      # it works
      foreach l $selectedColumns {
         set a($l) 1
      }
#      parray a
      set selectedColumns [ array names a ]

      # If we were dragging, send a notification that selection has changed.
      if {[info exists dragStateMap(type)] != 0 && \
              $dragStateMap(type) == "position" && \
              $dragStateMap(startedDragging) == 1} {
         notifySelectionChangeListeners
      }
#      puts "seqedit_widget_select.tcl.release_columnheader: end.  selCols: $selectedColumns"
   }   ;# end of release_columnheader

# -----------------------------------------------------------------------
   # Handle clicks in a cell
   proc click_cell {x y type} {
      variable groupNames
      variable selectionMap
      variable dragStateMap

      # Rest the drag state.
      array unset dragStateMap 

      # Get the row that was clicked on.
      set row [determineRowFromLocation $x $y]
      set column [determineColumnFromLocation $x $y]
#      puts "seqedit_widget_select.tcl.click_cell ($row, $column)"
      if {$row != -1 && $column != -1} {

         # Get sequence that is in row and position that is in the column.
         set sequence [determineSequenceFromRow $row]
         set position [determinePositionFromColumn $column]

         # is there is a real sequence in row and a valid position in column.
         if {$sequence != {} && [lindex $sequence 1] != -1 && $position != -1} {

            # Save the drag state in case the user tries to drag the selection.
            set dragStateMap(type) "cell"
            set dragStateMap(startingRow) $row
            set dragStateMap(startingColumn) $column
            set dragStateMap(startingSequence) $sequence
            set dragStateMap(startingPosition) $position
            set dragStateMap(destinationRow) $row
            set dragStateMap(destinationColumn) $column
            set dragStateMap(destinationSequence) $sequence
            set dragStateMap(destinationPosition) $position
            set dragStateMap(startedDragging) 0

            # If shift key, select all of the rows in between the selections.
            if {[lindex $type 0] == "shift"} {
               #puts "CLICK_CELL_SHIFT"
               #if {$selectionMap(type) == "cell" && [info exists selectionMap(lastSequence)] != 0 && [info exists selectionMap(lastRow)] != 0 && [info exists selectionMap(lastPosition)] != 0} {}
               # Get selection indexes
               if { $selectionMap(type) == "cell" } {
                  if {$selectionMap(lastRow) < $row} {
                     set s1 [getSequenceInGroup [lindex $groupNames \
                            [lindex $selectionMap(lastSequence) 0]] \
                            [lindex $selectionMap(lastSequence) 1]]
                     set s2 [getSequenceInGroup [lindex $groupNames \
                            [lindex $sequence 0]] [lindex $sequence 1]]
                  } else {
                     set s2 [getSequenceInGroup [lindex $groupNames \
                            [lindex $selectionMap(lastSequence) 0]] \
                            [lindex $selectionMap(lastSequence) 1]]
                     set s1 [getSequenceInGroup [lindex $groupNames \
                            [lindex $sequence 0]] [lindex $sequence 1]]
                  }
                  if {$selectionMap(lastPosition) < $position} {
                     set p1 $selectionMap(lastPosition)
                     set p2 $position
                  } else {
                     set p2 $selectionMap(lastPosition)
                     set p1 $position
                  }
                  if {[llength $type] == 2 && [lindex $type 1] == "control"} {
                     setSelectedCellRange $s1 $s2 $p1 $p2 1
                  } else {
                     setSelectedCellRange $s1 $s2 $p1 $p2
                  }
                  notifySelectionChangeListeners
               }

               # Else if the control key was down, flip the selection.
            } elseif {[lindex $type 0] == "control"} {
               setSelectedCell [getSequenceInGroup [lindex $groupNames \
                    [lindex $sequence 0]] [lindex $sequence 1]] $position 0 1
               set selectionMap(lastRow) $row
               set selectionMap(lastSequence) $sequence
               set selectionMap(lastPosition) $position

               notifySelectionChangeListeners
               # Otherwise it was normal click, set selection to this sequence.
            } else {
               setSelectedCell [getSequenceInGroup [lindex $groupNames \
                    [lindex $sequence 0]] [lindex $sequence 1]] $position 0 0
               set selectionMap(lastRow) $row
               set selectionMap(lastSequence) $sequence
               set selectionMap(lastPosition) $position
               notifySelectionChangeListeners
            }
         }
      }
   } ;# end of click_cell

# ------------------------------------------------------------------------
   # Handle drags on the row header.  foobar.  Not right. being called
   # for a normal selection, too
   proc move_cell {x y} {

      variable editor
      variable groupNames
      variable groupMap
      variable numberRows
      variable firstGroup
      variable firstSequence
      variable cellColorForeground
      variable columnObjectMap
      variable dragStateMap

      # See if we are really dragging a row header.
      if {[info exists dragStateMap(type)] != 0 && \
                                       $dragStateMap(type) == "cell"} {

         # Get the row and column that are being dragged onto.
         set destinationRow [determineRowFromLocation $x $y]
         set destinationColumn [determineColumnFromLocation $x $y]

         # See if we are dragging onto a different row or column.
         if {$destinationRow != -1 && $destinationColumn != -1 && \
                ($dragStateMap(destinationRow) != $destinationRow || \
                $dragStateMap(destinationColumn) != $destinationColumn)} {

            # Save that we are dragging.
            set dragStateMap(startedDragging) 1

            # Set the new destination variables.
            set dragStateMap(destinationRow) $destinationRow
            set dragStateMap(destinationColumn) $destinationColumn

            # Get the sequence that is in the destination row.
            set destinationSequence [determineSequenceFromRow $destinationRow]
            set destinationPosition [determinePositionFromColumn \
                                                     $destinationColumn]
            set dragStateMap(destinationSequence) $destinationSequence
            set dragStateMap(destinationPosition) $destinationPosition

            # Make sure there is a real sequence in the row
            # and a valid position in the column.
            if {$destinationSequence != {} && \
                            [lindex $destinationSequence 1] != -1 && \
                            $destinationPosition != -1} {

               # Make sure we have the starting and ending ordered correctly.
               if {$dragStateMap(startingRow) < $destinationRow} {
                  set s1 [getSequenceInGroup [lindex $groupNames \
                              [lindex $dragStateMap(startingSequence) 0]]\
                              [lindex $dragStateMap(startingSequence) 1]]
                  set s2 [getSequenceInGroup [lindex $groupNames \
                              [lindex $destinationSequence 0]] \
                              [lindex $destinationSequence 1]]
               } else {
                  set s2 [getSequenceInGroup [lindex $groupNames \
                              [lindex $dragStateMap(startingSequence) 0]] \
                              [lindex $dragStateMap(startingSequence) 1]]
                  set s1 [getSequenceInGroup [lindex $groupNames \
                              [lindex $destinationSequence 0]] \
                              [lindex $destinationSequence 1]]
               }
               if {$dragStateMap(startingColumn) < $destinationColumn} {
                  set p1 $dragStateMap(startingPosition)
                  set p2 $destinationPosition
               } else {
                  set p2 $dragStateMap(startingPosition)
                  set p1 $destinationPosition
               }

               # Select all of cells between the start and the end point.
               setSelectedCellRange $s1 $s2 $p1 $p2

            }


            # See if we should try to scroll the screen up.
            #if {$destinationRow == 0 || ($destinationRow == -1 && $y <= $columnObjectMap(h,0.y1))} {

                # If the screen can be scrolled up at all, scroll it up.
            #    if {($firstGroup == 0 && $firstSequence > -1) || ($firstGroup > 0)} {
            #        scroll_vertical scroll -1 unit
            #    }

            # See if we should try to scroll the screen down.
            #} elseif {$destinationRow == [expr $numberRows-1] || ($destinationRow == -1 && $y >= $columnObjectMap(h,[expr $numberRows-1].y1))} {

                # If the screen can be scrolled down at all, scroll it down.
            #    set lastSequence [determineSequenceFromRow [expr $numberRows-2]]
            #    if {$lastSequence != {}} {
            #        set lastGroupIndex [lindex $lastSequence 0]
            #       set lastSequenceIndex [lindex $lastSequence 1]
            #        if {($lastGroupIndex < [expr [llength $groupNames]-1]) || ($lastGroupIndex == [expr [llength $groupNames]-1] && $lastSequenceIndex < [expr $groupMap([lindex $groupNames $lastGroupIndex],numberSequences)-1])} {
            #            scroll_vertical scroll 1 unit
            #        }
            #    }
            #}
         }
      }
   } ;# end of move_cell

# -------------------------------------------------------------------------
   proc release_cell {x y} {
      set row [determineRowFromLocation $x $y]
      set column [determineColumnFromLocation $x $y]
#      puts "seqedit_widget_select.tcl.release_cell ($row, $column)"

      variable dragStateMap
      variable groupNames

      # If dragging, send a notification that selection has changed.
      if {[info exists dragStateMap(type)] != 0 && \
                  $dragStateMap(type) == "cell" && \
                  $dragStateMap(startedDragging) == 1} {

         # if the current cell isn't where we started, we need to
         # remove the cell where we started dragging from the selection
         # list, since it is covered in the drag already
#         puts "in dragging..  startRow: $dragStateMap(startingRow), startCol: $dragStateMap(startingColumn)"
         if {! ($row == $dragStateMap(startingRow) &&
                $column == $dragStateMap(startingColumn)) } {

            set sqr [determineSequenceFromRow $dragStateMap(startingRow)]
			   set gp [lindex $groupNames [lindex $sqr 0]]
   			set sq [lindex $sqr 1]

            # this call will toggle off the starting cell from the
            # selection
            setSelectedCell \
                 [getSequenceInGroup $gp $sq] \
                 [determinePositionFromColumn $dragStateMap(startingColumn)] \
                 0 1
         }

         notifySelectionChangeListeners
      }
   }

# -------------------------------------------------------------------------
    ###########################################
    #
    # Determines if a specific column is selected
    #
    # Args:
    #  position: The column number
    #
   proc isColumnSelected { position } {
      variable selectedColumns
      foreach col $selectedColumns {
         if { $col == $position } {
            return 1
         }
      }
      return 0
   }

# -------------------------------------------------------------------------
    ###########################################
    #
    # Determines if a specific sequence is selected
    #
    # Args:
    #  sequence: The sequence ID
    #
    proc isSequenceSelected { sequence } {
	    variable selectedSequences
	    	foreach seq $selectedSequences {

		if { $seq == $sequence } {
			return 1
		}
	}

	return 0
    }

# -------------------------------------------------------------------------
   ###########################################
   #
   # Determines if a specific element is selected
   #
   # Args:
   #  seqID: The sequence ID the element is in
   #  element: The column of the specified element
   #
   proc isSelected { seqID element } {
      variable selectedSequences
      variable selectedColumns
      variable selectedRanges
      variable selectedElements
      variable noSelections

#      puts "seqedit_widget_select.tcl.isSelected.start. seqID:$seqID elem:$element, selSeqs: $selectedSequences selCols: $selectedColumns, selRanges: $selectedRanges, selElem: $selectedElements"

      if { $noSelections } {
         return 0
      }

      foreach seq $selectedSequences {
         if { $seq == $seqID } {
#            puts " foreach seq selectedSequences \n"
            return 1
         }
      }

      foreach col $selectedColumns {

         if { $col == $element } {
#            puts "foreach col selectedColumns\n"
            return 1
         }

      }

      variable rowMap
      variable numberRows
#      puts "seqedit_widget_select.isSelected numRows: $numberRows, rowMap: [array get rowMap]"

#      set seqRow [determineRowFromSequence $seqID]   

      foreach range $selectedRanges {

         set seqList [getSequenceIDsInRows  [lindex $range 0] [lindex $range 2]]
         set e1 [lindex $range 1]
         set e2 [lindex $range 3]
         if { $e1 > $e2} {
            set t $e1
            set e1 $e2
            set e2 $t
         }

         if { [lsearch -exact $seqList $seqID] != -1 } {
            if { $element >= $e1 && $element <= $e2 } {
#            puts "seqRow: $seqRow, s1: $s1, s2: $s2 if  element >= e1 && element <= e2 \n"
               return 1
            }
         }
      }

      foreach selElement $selectedElements {
         if { $seqID == [lindex $selElement 0] && \
              $element == [lindex $selElement 1] } {
#            puts "foreach selElement selectedElements\n"
            return 1
         }
      }
#      puts "returning 0"
      return 0
   }

# -----------------------------------------------------------------------
    ###########################################
    #
    # Determines the row a sequence is in
    #
    # Args:
    #  seqID: The sequence ID the element is in
    #
    # Returns:
    #  The row number of the sequence, or -1 if if doesn't exist
    #
#    proc determineRowFromSequence { seqID } {
#
#	    variable rowMap
#
#       if {![info exists rowMap($seqID)]} {
#          return -1
#       }
#
#	    set row $rowMap($seqID)
##       puts "determineRowFromSequence seqId: $seqID, row: $row"
#
#	    if { [string is digit $row] } {
#		    return $row
#	    } else {
#		    return -1
#	    }
#
#    }

# -----------------------------------------------------------------------
#   ###########################################
#   #
#   # Creates the rowMap array, a mapping of sequences to row numbers
#   #
#   proc createRowMap { } {
#      variable rowMap
#      variable numberRows
#      variable groupNames
##      puts "seqedit_widget_select.createRowMap.start"
#      array unset rowMap
#
#      for { set i 0 } { $i < $numberRows } { incr i } {
#         set group [lindex $groupNames [lindex \
#                                        [determineSequenceFromRow $i] 0]]
#         set seq [lindex [determineSequenceFromRow $i] 1]
##         puts -nonewline "{i:$i, sfr:[determineSequenceFromRow $i], s:$seq, "
#         if { $seq == -1 } {
#            continue
#         }
#         if { $seq == "" } {
#            continue
#         }
#         set rowSeq [getSequenceInGroup $group $seq]	   	
##         puts -nonewline "rs:$rowSeq} rm:<[array get rowMap]>"
#         set rowMap($rowSeq) $i
#      }
##      puts "\n seqedit_widget_select.createRowMap: [array get  rowMap]"
#   }
# -----------------------------------------------------------------------

}

