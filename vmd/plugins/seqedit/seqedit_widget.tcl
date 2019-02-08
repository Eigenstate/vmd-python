# University of Illinois Open Source License
# Copyright 2004-2007 Luthey-Schulten Group,
# All rights reserved.
#
# $Id: seqedit_widget.tcl,v 1.18 2018/11/06 23:10:22 johns Exp $
#
# Developed by: Luthey-Schulten Group
#               University of Illinois at Urbana-Champaign
#               http://faculty.scs.illinois.edu/schulten/
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
# Author(s): Elijah Roberts, Michael Bach

package provide seqedit_widget 1.1
package require seqdata 1.1

# Declare global variables for this package.
namespace eval ::SeqEditWidget:: {

    # Export the package namespace.
    namespace export SeqEditWidget

    # Setup some default colors.
    variable headerColorActive "#D2D2D2"
    variable headerColorInactive "#D3D3D3"
    variable headerColorForeground "#000000"
    variable headerNumberingColor "#808080"
    variable cellColorActive "#FFFFFF"
    variable cellColorInactive "#C0C0C0"
    variable cellColorForeground "#000000"
    variable cellTextReplacementColor "#D3D3D3"
    variable selectionColor "#FFFF3E"
    variable checkColorActive "#000080"
    variable infobuttonColorActive "#005479"
    variable infobuttonFontColorActive "#FFFFFF"
    variable repbuttonColorActive "#005479"
    variable repbuttonFontColorActive "#FFFFFF"
    variable vmdbuttonColorActive "#D60811"
    variable vmdbuttonFontColorActive "#FFFFFF"

    variable currentTitle 
#    variable maxSequencesInGroup 0

    # The name of the editor widget.
    variable widget

    # The name of the editor portion of the widget.
    variable editor

    # The status bar.
    variable statusBar

    # The group popup menu.
    variable groupPopupMenu

    # The representation popup menu.
    variable repPopupMenu

    # The vmd popup menu.
    variable vmdPopupMenu

    # Parameters for the popup menu.
    variable popupMenuParameters

    # The current width of the sequence
    variable width

    # Handle to the sequence editor window.
    variable height

    # The current width of a cell in the editor.
    variable cellWidth

    # The current height of a cell in the editor.
    variable cellHeight

    # The current width of a cell in the editor.
    variable headerCellWidth

    # The current height of a cell in the editor.
    variable headerCellHeight

    # The objects that make up the grid columns.
    variable columnObjectMap

    # The number of columns currently present in the editor.
    variable numberCols

    # The number of rows currently present in the editor.
    variable numberRows

    # The current groups in the editor.
    variable groupNames

    # The data for the groups.
    variable groupMap

    # The first sequence element being displayed.
    variable firstElement

    # The first group being displayed.
    variable firstGroup

    # The first sequence being displayed.
    variable firstSequence

    # The length of the longest sequence.
    variable numberElements

    # The current color map handler
    variable colorMap

    # The current mapping of sequence ids to representation.
    variable representationMap

    # The font being used for the header display.
    variable headerFont

    # The font being used for the cell display.
    variable cellFont

    # The width of a number in the current cell font.
    variable cellFontNumberWidth

    # The font being used for the group headings.
    variable groupHeaderFont

    # The font being used for the header buttons.
    variable buttonFont

    # The current mapping of marked sequences.
    variable markMap

    # The last clicked group.
    variable clickedGroup

    # The current sequence removal notification commands.
    variable sequenceRemovalNotificationCommands

    # The scale of the zoomed out image
    variable imageScale 1.0

    variable drawZoom 0

    # Scaled first column
    variable firstx 0

    variable lastx 0

    # Scaled first row
    variable firsty 0

    # Scaled last row
    variable lasty 0

    # Scale for the zoomed out window
    variable scale 1

    # The canvas for the zoomed out window
    variable mainCanvas

    # Photo image of the zoomed out view
    variable photo

# --------------------------------------------------------------------------
    # Reset the package to get initial variable values.
    proc reset {} {

        # Reset the package variables.
        set ::SeqEditWidget::widget ""
        set ::SeqEditWidget::editor ""
        set ::SeqEditWidget::statusBar ""
        set ::SeqEditWidget::currentTitlePopupMenu ""
        set ::SeqEditWidget::groupPopupMenu ""
        set ::SeqEditWidget::repPopupMenu ""
        set ::SeqEditWidget::vmdPopupMenu ""
        set ::SeqEditWidget::popupMenuParameters ""
        set ::SeqEditWidget::width 0
        set ::SeqEditWidget::height 0
        set ::SeqEditWidget::cellWidth 0
        set ::SeqEditWidget::cellHeight 0
        set ::SeqEditWidget::headerCellWidth 0
        set ::SeqEditWidget::headerCellHeight 0
        array unset ::SeqEditWidget::columnObjectMap 
        set ::SeqEditWidget::numberCols 0
        set ::SeqEditWidget::numberRows 0
        set ::SeqEditWidget::groupNames {}
        array unset ::SeqEditWidget::groupMap 
        set ::SeqEditWidget::firstElement 0
        set ::SeqEditWidget::firstGroup 0
        set ::SeqEditWidget::firstSequence -1
        set ::SeqEditWidget::numberElements 0
        set ::SeqEditWidget::colorMap "::SeqEdit::ColorMap::Default"
        array unset ::SeqEditWidget::representationMap 
        set ::SeqEditWidget::headerFont ""
        set ::SeqEditWidget::cellFont ""
        set ::SeqEditWidget::cellFontNumberWidth 0
        set ::SeqEditWidget::groupHeaderFont ""
        set ::SeqEditWidget::buttonFont ""
        array unset ::SeqEditWidget::markMap 
        set ::SeqEditWidget::clickedGroup ""
        set ::SeqEditWidget::sequenceRemovalNotificationCommands {}
    }


    reset
}

# --------------------------------------------------------------------------
proc ::SeqEditWidget::getCurrentTitleText {num} {
   variable titleTypes
   return [lindex $titleTypes $num]
}

# --------------------------------------------------------------------------
proc ::SeqEditWidget::getRowTitle {sequenceID} {
   variable currentTitle
   set ret ""
   switch [getCurrentTitleText $currentTitle] {
      "Sequence Name" { set ret "[SeqData::getName $sequenceID]" }
      "Sequence Length" { set ret "[SeqData::getSeqLength $sequenceID]" }
      "Sequence Sources" { set ret "[SeqData::getSources $sequenceID]" }
      "Sequence Type" { set ret "[SeqData::getType $sequenceID]" }
      "Domain Of Life" { set ret "[SeqData::getDomainOfLife $sequenceID]" }
      "Common Name" { set ret "[SeqData::getCommonName $sequenceID]" }
      "Scientific Name - Long" {set ret "[SeqData::getScientificName $sequenceID]" }
      "Scientific Name - Short" {set ret "[SeqData::getShortScientificName $sequenceID]" }
      "Enzyme Commission Number" {set ret "[SeqData::getEnzymeCommissionNumber $sequenceID]" }
      "Temperature Class" {set ret "[SeqData::getTemperatureClass $sequenceID]" }
      "Percent GC" {set ret "[SeqData::getPercentGC $sequenceID]" }
      - { set ret "Unknown" }
   }

   if { $ret == ""} {
      return "<[SeqData::getName $sequenceID]>"
   } else {
      return $ret
   }
}



############################## PUBLIC METHODS #################################
# Methods in this section can be called by external applications.             #
###############################################################################
# --------------------------------------------------------------------------
# Creates a new sequence editor.
# args:     a_control - The frame that the widget should be shown in.
#           a_cellWidth - The width of a cell in the editor.
#           a_cellWidth - The height of a cell in the editor.
proc ::SeqEditWidget::createWidget {a_widget a_cellWidth a_cellHeight} {

    variable widget
    variable editor
    variable groupPopupMenu
    variable repPopupMenu
    variable width
    variable height
    variable cellColorInactive
    variable titleTypes
    variable currentTitle
    variable currentTitlePopupMenu

    set currentTitle 0

    set titleTypes [ list "Sequence Name" \
                          "Sequence Length" \
                          "Percent GC" \
                          "Domain Of Life" \
                          "Common Name" \
                          "Scientific Name - Short" \
                          "Scientific Name - Long" \
                          "Sequence Sources" \
                          "Enzyme Commission Number" \
                          "Sequence Type" \
                          "Temperature Class" \
                   ]

    set widget $a_widget

    #Create the components of the widget.
    frame $widget.center
    set editor [canvas $widget.center.editor -background $cellColorInactive]
    scrollbar $widget.center.yscroll -orient vertical -command {::SeqEditWidget::scroll_vertical}

    frame $widget.bottom
    scrollbar $widget.bottom.xscroll -orient horizontal -command {::SeqEditWidget::scroll_horzizontal}
    frame $widget.bottom.spacer -width [$widget.center.yscroll cget -width]
    set statusBar [label $widget.bottom.statusbar -textvariable "::SeqEditWidget::statusBarText" -anchor w -relief sunken -borderwidth 1]

    pack $widget.center -side top -fill both -expand true
    pack $widget.center.editor -side left -fill both -expand true
    pack $widget.center.yscroll -side right -fill y
    pack $widget.bottom -side bottom -fill x
    pack $widget.bottom.spacer -side right
    pack $widget.bottom.xscroll -side top -fill x -expand true
    pack $widget.bottom.statusbar -side bottom -fill x -expand true

    # Listen for resize events.
    bind $editor <Configure> {::SeqEditWidget::component_configured %W %w %h}

    # Calculate some basic information about the editor.
    set width [$editor cget -width]
    set height [$editor cget -height]

    # Set the cell size.
    setCellSize $a_cellWidth $a_cellHeight false

    # Set the scrollbars.
    setScrollbars

    # Create the grid.
    createCells

    # Create the group popup menu.
    set groupPopupMenu [menu $widget.groupPopupMenu -title "Grouping" -tearoff no]
    $groupPopupMenu add command -label "Insert Group..." -command "::SeqEditWidget::menu_insertgroup"
    $groupPopupMenu add command -label "Rename Group..." -command "::SeqEditWidget::menu_renamegroup"
    $groupPopupMenu add command -label "Delete Group" -command "::SeqEditWidget::menu_deletegroup"
    $groupPopupMenu add command -label "Add to Group..." -command "::SeqEditWidget::menu_addtogroup"
    $groupPopupMenu add separator
    $groupPopupMenu add command -label "Mark Group" -command "::SeqEditWidget::menu_markgroup 1"
    $groupPopupMenu add command -label "Unmark Group" -command "::SeqEditWidget::menu_markgroup 0"
    $groupPopupMenu add command -label "Mark All" -command "::SeqEditWidget::menu_markall 1"
    $groupPopupMenu add command -label "Unmark All" -command "::SeqEditWidget::menu_markall 0"

    # Create the representation popup menu.
    set repPopupMenu [menu $widget.repPopupMenu -title "Representation" -tearoff no]
    $repPopupMenu add command -label "Duplicate" -command "::SeqEditWidget::menu_duplicate"
    $repPopupMenu add separator
    $repPopupMenu add command -label "Sequence" -command "::SeqEditWidget::menu_setrepresentation sequence"
    $repPopupMenu add command -label "Bar" -command "::SeqEditWidget::menu_setrepresentation bar"
    $repPopupMenu add command -label "Secondary Structure" -command "::SeqEditWidget::menu_setrepresentation secondary"

    # Create the group popup menu.
    set currentTitlePopupMenu [menu $widget.currentTitlePopupMenu -title "Current Title" -tearoff no]
    for {set x 0} {$x < [llength $titleTypes]} {incr x} {
       $currentTitlePopupMenu add radio -command "::SeqEditWidget::setCurrentTitle" -label [getCurrentTitleText $x] -variable "::SeqEditWidget::currentTitle" -value $x
    }

    # set the key listener.
    bind $editor <KeyPress> {::SeqEditWidget::editor_keypress %K}
    bind $editor <<Cut>>  {::SeqEditWidget::editor_cut}
    bind $editor <<Copy>>  {::SeqEditWidget::editor_copy}
    bind $editor <<Paste>>  {::SeqEditWidget::editor_paste}
    focus $editor

    # Initialize the status bar.
    initializeStatusBar
}

# --------------------------------------------------------------------------
# Adds a command to be run when sequences have been removed from the editor.
# args:     command - The command to be executed whenever sequences have been removed.
proc ::SeqEditWidget::addRemovalNotificationCommand {command} {

    variable sequenceRemovalNotificationCommands

    lappend sequenceRemovalNotificationCommands $command
}

# --------------------------------------------------------------------------
# Creates a new sequence editor.
# args:     a_control - The frame that the widget should be shown in.
#           a_cellWidth - The width of a cell in the editor.
#           a_cellWidth - The height of a cell in the editor.
proc ::SeqEditWidget::setCellSize {a_cellWidth a_cellHeight {redraw true}} {

    variable editor
    variable width
    variable height
    variable cellWidth
    variable cellHeight
    variable headerCellWidth
    variable headerCellHeight
    variable numberCols
    variable numberRows
    variable headerFont
    variable cellFont
    variable cellFontNumberWidth
    variable groupHeaderFont
    variable buttonFont
    set cellWidth $a_cellWidth
    set cellHeight $a_cellHeight

    # Set up any settings that are based on the cell size.
    set headerCellWidth 200
    set headerCellHeight 18
    set numberCols [expr (($width-$headerCellWidth)/$cellWidth)+1]
    set numberRows [expr (($height-$headerCellHeight)/$cellHeight)+1]
    set fontChecker [$editor create text 0 0 -anchor nw -text ""]
    if {$headerFont != ""} {font delete $headerFont}
    if {$cellFont != ""} {font delete $cellFont}
    if {$groupHeaderFont != ""} {font delete $groupHeaderFont}
    if {$buttonFont != ""} {font delete $buttonFont}
    set defaultFont [$editor itemcget $fontChecker -font]
    set headerFont [font create headerFont -family [font actual $defaultFont -family] -size [font actual $defaultFont -size]]
    set buttonFont [font create buttonFont -family "courier" -size 10]
    if {$cellHeight >= 12 && $cellWidth >= 12} {
        set cellFont [font create cellFont -family [font actual $defaultFont -family] -size [expr ($cellHeight+1)/2]]
        set cellFontNumberWidth [font measure $cellFont "9"]
        set groupHeaderFont [font create groupHeaderFont -family [font actual $defaultFont -family] -size [expr ($cellHeight+3)/2] -weight bold]
    } else {
        set cellFont ""
        set cellFontNumberWidth 0
        set groupHeaderFont ""
    }
    $editor delete $fontChecker

    validateScrollRange

    # Redraw the component, if requested to.
    if {$redraw == 1 || $redraw == "true" || $redraw == "TRUE"} {
        deleteCells
        setScrollbars
        createCells
        redraw
    }
}

# --------------------------------------------------------------------------
# Gets the length of the current alignment.
proc ::SeqEditWidget::getNumberPositions {} {

    variable numberElements

    return $numberElements
}

# --------------------------------------------------------------------------
# Gets the sequences that are currently being displayed by the editor.
proc ::SeqEditWidget::getSequences {} {

    variable groupNames
    variable groupMap

    set sequenceIDs {}
    foreach groupName $groupNames {
        set groupSequenceIDs $groupMap($groupName,sequenceIDs)
        set sequenceIDs [concat $sequenceIDs $groupSequenceIDs]
    }
    return $sequenceIDs
}

# --------------------------------------------------------------------------
proc ::SeqEditWidget::containsSequence {sequenceID} {

    variable groupNames
    variable groupMap

    foreach groupName $groupNames {
        if {[lsearch $groupMap($groupName,sequenceIDs) $sequenceID] != -1} {
            return 1
        }
    }

    return 0
}

# --------------------------------------------------------------------------
# Sets the sequences that are currently being displayed by the editor.
# args:     sequenceIDs - A list of the sequence ids.
#           groupName - The
proc ::SeqEditWidget::setSequences {sequenceIDs {groupName ""}} {

    variable groupNames
    variable groupMap
    variable firstElement
    variable firstGroup
    variable firstSequence
    variable numberElements

    # Reset all of the sequence storage variables.
    foreach groupName $groupNames {
        removeAllSequencesFromGroup $groupName
    }
    set numberElements 0
    set firstElement 0
    set firstGroup 0
    set firstSequence -1
    resetColoring all 0
    setRepresentations $sequenceIDs "sequence" 0

    # Add the sequences to the specified group.
    addSequences $sequenceIDs $groupName
}

# --------------------------------------------------------------------------
# Add specified sequences to those that are currently being displayed by editor.
# args:     sequenceIDs - A list of the sequence ids to add.
proc ::SeqEditWidget::addSequences {sequenceIDs {groupName ""} {position end} {redraw 1}} {

    variable numberElements
    variable groupNames
    variable groupMap

    #Figure out if the max sequence length has increased.
    foreach sequenceID $sequenceIDs {

        #Get the sequence.
        set sequenceLength [SeqData::getSeqLength $sequenceID]

#        puts "seqeditwidget::addseq: seq: $sequenceID, length: $sequenceLength"

        #Compare the length to the max.
        if {$sequenceLength > $numberElements} {
            set numberElements $sequenceLength
        }
    }

    # If a group wasn't specified, use the first group.
    if {$groupName == ""} {
        set groupName [lindex $groupNames 0]
    }

    # If the group doesn't exists, create it.
    createGroup $groupName 0

    # If we are inserting at the end, translate to an integer.
    if {$position == "end"} {set position $groupMap($groupName,numberSequences)}

    # Add the sequences to the specified group.
    set groupMap($groupName,sequenceIDs) [concat [lrange $groupMap($groupName,sequenceIDs) 0 [expr $position-1]] $sequenceIDs [lrange $groupMap($groupName,sequenceIDs) $position end]]
    set groupMap($groupName,numberSequences) [llength $groupMap($groupName,sequenceIDs)]

    # Reset the coloring map for the new sequences.
    resetColoring $sequenceIDs 0
    setRepresentations $sequenceIDs "sequence" 0

    # Reset the mark and selection maps.
    resetMarks 1
    resetSelection

    # Set the scrollbars.
    setScrollbars

    #Redraw the widget.
    if {$redraw == 1} {redraw}
}

# --------------------------------------------------------------------------
# Remove all of the sequences from a given group.
# args:     groupName - The index of the group from which to remove all of the sequences.
proc ::SeqEditWidget::removeAllSequences {{redraw 1}} {

    variable groupNames
    variable firstElement
    variable firstGroup
    variable firstSequence
    variable numberElements

    # Reset all of the sequence storage variables.
    set groupNames {}
    array unset groupMap 
    set numberElements 0
    set firstElement 0
    set firstGroup 0
    set firstSequence -1
    resetColoring all 0

    #Redraw the widget.
    if {$redraw == 1} {redraw}
}

# --------------------------------------------------------------------------
# Removes the specified sequences from the editor
# args: sequenceIDs - A list of sequence ids to be removed from the editor.
#       redraw - Whether to redraw the editor after removing the sequence. (0/1)
proc ::SeqEditWidget::removeSequences {sequenceIDs {redraw 1}} {

    variable numberElements
    variable groupNames
    variable groupMap
    variable sequenceRemovalNotificationCommands

    # Go through each group and remove any sequences in the list.
    set recalculateNumberElements 0
    foreach groupName $groupNames {

        # If the group has any sequences.
        if {$groupMap($groupName,numberSequences) > 0 } {

            # Go through each sequence.
            foreach sequenceID $sequenceIDs {
                if {[set lidx [lsearch $groupMap($groupName,sequenceIDs) $sequenceID]] != -1 } {

                    # Remove each one from the list and decrement the sequence counter for the group.
                    set groupMap($groupName,sequenceIDs) [lreplace $groupMap($groupName,sequenceIDs) $lidx $lidx]
                    incr groupMap($groupName,numberSequences) -1

                    # If this sequence was the longest one in the editor, mark that we need to recalculate the max.
                    if {[SeqData::getSeqLength $sequenceID] == $numberElements} {
                        set recalculateNumberElements 1
                    }
                }
            }
        }
    }

    #Figure out the new maximum sequence length, if necessary.
    if {$recalculateNumberElements == 1} {
        set numberElements 0
        foreach groupName $groupNames {
            set groupSequenceIDs $groupMap($groupName,sequenceIDs)
            foreach groupSequenceID $groupSequenceIDs {
                set sequenceLength [SeqData::getSeqLength $groupSequenceID]
                if {$sequenceLength > $numberElements} {
                    set numberElements $sequenceLength
                }
            }
        }
    }

    # Reset the mark and selection maps.
    resetMarks 1
    resetSelection

    # Set the scrollbars.
    setScrollbars

    if {$redraw == 1} {redraw}

    # Notify any listeners.
    foreach sequenceRemovalNotificationCommand $sequenceRemovalNotificationCommands {
        $sequenceRemovalNotificationCommand $sequenceIDs
    }
}

# --------------------------------------------------------------------------
# Updates the specified sequences in the editor.
proc ::SeqEditWidget::updateSequences {sequenceIDs} {

    variable groupNames
    variable groupMap
    variable numberElements
    variable markMap

    # Go through each of the new sequences.
    foreach sequenceID $sequenceIDs {

        # Figure out if the max sequence length has increased.
        if {[SeqData::getSeqLength $sequenceID] > $numberElements} {
            set numberElements [SeqData::getSeqLength $sequenceID]
        }
    }

    # Reset the coloring map for the new sequences.
    resetColoring $sequenceIDs 0

    # Reset the mark and selection maps.
    resetMarks 1
    resetSelection

    # Set the scrollbars.
    setScrollbars

    #Redraw the widget.
    redraw
}

# --------------------------------------------------------------------------
proc ::SeqEditWidget::duplicateSequences {sequenceIDs {redraw 1}} {

    variable groupNames
    variable groupMap

    # Go through each sequence.
    foreach sequenceID $sequenceIDs {

        # Create a copy of the sequence.
        set newSequenceID [::SeqData::duplicateSequence $sequenceID]

        # Find the group and position of the original sequence.
        set addGroupName ""
        set addGroupPosition -1
        foreach groupName $groupNames {
            set position [lsearch $groupMap($groupName,sequenceIDs) $sequenceID]
            if {$position != -1} {
                set addGroupName $groupName
                set addGroupPosition [expr $position+1]
                break
            }
        }

        # If we found the source sequence, add the new one right after it.
        if {$addGroupName != "" && $addGroupPosition != -1}  {

            # Add the sequences to the specified group.
            set groupMap($addGroupName,sequenceIDs) [concat [lrange $groupMap($addGroupName,sequenceIDs) 0 [expr $addGroupPosition-1]] $newSequenceID [lrange $groupMap($addGroupName,sequenceIDs) $addGroupPosition end]]
            set groupMap($addGroupName,numberSequences) [llength $groupMap($addGroupName,sequenceIDs)]
        }

        # Reset the coloring map for the new sequence.
        resetColoring $newSequenceID 0
        setRepresentations $newSequenceID [getRepresentations $sequenceID] 0
    }

    # Reset the mark and selection maps.
    resetMarks 1
    resetSelection

    # Set the scrollbars.
    setScrollbars

    #Redraw the widget.
    if {$redraw == 1} {redraw}
}

# --------------------------------------------------------------------------
# Replaces each sequence in the first list with the corresponding sequence in the second list.
# args:     originalSequenceIDs - The sequences to be replaced.
#           replacementSequenceIDs - The new sequences.
proc ::SeqEditWidget::replaceSequences {originalSequenceIDs replacementSequenceIDs} {

    variable groupNames
    variable groupMap
    variable numberElements
    variable markMap

    # Go through each of the new sequences.
    for {set i 0} {$i < [llength $replacementSequenceIDs]} {incr i} {

        # Get the sequence.
        set sequenceID [lindex $replacementSequenceIDs $i]
        set sequence [SeqData::getSeq $sequenceID]

        # Figure out if the max sequence length has increased.
        if {[llength $sequence] > $numberElements} {
            set numberElements [llength $sequence]
        }

        # See if we can find this sequence's old identity.
        if {$i < [llength $originalSequenceIDs]} {

            # Get the id of the old sequence.
            set oldSequenceID [lindex $originalSequenceIDs $i]

            # Go through the groups and replace any occurrences of the old id with the new one.
            foreach groupName $groupNames {

                # Get the sequence ids.
                set groupSequenceIDs $groupMap($groupName,sequenceIDs)

                # See if the old sequence id is in the list.
                set position [lsearch $groupSequenceIDs $oldSequenceID]
                if {$position != -1} {
                    set groupMap($groupName,sequenceIDs) [lreplace $groupSequenceIDs $position $position $sequenceID]
                }
            }

            # Preserve the mark state of the old sequence.
            set markMap($sequenceID) $markMap($oldSequenceID)

        # Otherwise, just add the it to the first group.
        } else {
            addSequences $sequenceID [lindex $groupNames 0]
        }

    }

    # Reset the coloring map for the new sequences.
    resetColoring $replacementSequenceIDs 0
    setRepresentations $replacementSequenceIDs [getRepresentations $originalSequenceIDs] 0

    # Reset the mark and selection maps.
    resetMarks 1
    resetSelection

    # Set the scrollbars.
    setScrollbars

    #Redraw the widget.
    redraw
}

# --------------------------------------------------------------------------
# Moves a list of sequences ids from their existing locations into the specified group,
# optionally at a specified position.
# args:     groupName - The group to which to add the sequences.
#           movingSequenceIDs - The list of sequences ids to move to the group.
#           position - (default end) If the list should be added at a specific position in
#               the group, specify it here. An empty string signifies the end of the list.
proc ::SeqEditWidget::moveSequences {movingSequenceIDs moveToGroupName {position end} {redraw 1}} {

    variable groupMap
    variable groupNames

    # Create the group, if it does not exist.
    createGroup $moveToGroupName 0

    # Figure out the default position, if necessary.
    if {$position == "end"} {set position $groupMap($moveToGroupName,numberSequences)}

    # Go through all of the groups.
    foreach groupName $groupNames {

        # Go through all of the sequence ids in the group and create a new list without the moving elements.
        set sequenceIDs $groupMap($groupName,sequenceIDs)
        set newSequenceIDs {}
        for {set j 0} {$j < [llength $sequenceIDs]} {incr j} {

            # Get the sequence id.
            set sequenceID [lindex $sequenceIDs $j]

            # Add the sequence id to the new list if it is not in the moving list.
            if {[lsearch $movingSequenceIDs $sequenceID] == -1} {
                lappend newSequenceIDs $sequenceID

            # Otherwise, see if we need to adjust the position to account for removing this sequence id.
            } elseif {$groupName == $moveToGroupName && $j < $position} {
                incr position -1
            }
        }

        # Set the new list for the group.
        set groupMap($groupName,sequenceIDs) $newSequenceIDs
        set groupMap($groupName,numberSequences) [llength $newSequenceIDs]
    }

    # Add the sequences to the specified group.
    set groupMap($moveToGroupName,sequenceIDs) [concat [lrange $groupMap($moveToGroupName,sequenceIDs) 0 [expr $position-1]] $movingSequenceIDs [lrange $groupMap($moveToGroupName,sequenceIDs) $position end]]
    set groupMap($moveToGroupName,numberSequences) [llength $groupMap($moveToGroupName,sequenceIDs)]

    if {$redraw == 1} {redraw}
}

# --------------------------------------------------------------------------
# Creates a new group and returns its index. If a group with the name already exists,
# the index of the existing group is returned.
# args:     groupName - The name of the group to create.
# return:   The index of the newly created group.
proc ::SeqEditWidget::createGroup {groupName {redraw 1}} {

    variable groupNames

    return [insertGroup $groupName end $redraw]
}

# --------------------------------------------------------------------------
# Renames a group. If a group with the name already exists, nothing is done.
# args:     groupName - The old name of the group.
#           newName - The new name of the group.
proc ::SeqEditWidget::renameGroup {groupName newName {redraw 1}} {

    variable groupNames
    variable groupMap

    # Rename the group if it exists and the new name des not.
    if {[lsearch $groupNames $groupName] != -1 && [lsearch $groupNames $newName] == -1} {
        set groupIndex [lsearch $groupNames $groupName]
        set groupNames [lreplace $groupNames $groupIndex $groupIndex $newName]
        set groupMap($newName,sequenceIDs) $groupMap($groupName,sequenceIDs)
        set groupMap($newName,numberSequences) $groupMap($groupName,numberSequences)
        unset groupMap($groupName,sequenceIDs)
        unset groupMap($groupName,numberSequences)
        if {$redraw == 1} {redraw}
    }
}

# --------------------------------------------------------------------------
# Inserts a new group at the specified position. If a group with the name already exists,
# the index of the existing group is returned.
# args:     groupName - The name of the group to create.
#           position - The position at which the group should be created.
proc ::SeqEditWidget::insertGroup {groupName {position end} {redraw 1}} {

    variable groupNames
    variable groupMap

    # Add the group if it does not yet exist.
    set index [lsearch $groupNames $groupName]
    if {$index == -1} {
        set groupNames [linsert $groupNames $position $groupName]
        set groupMap($groupName,sequenceIDs) {}
        set groupMap($groupName,numberSequences) 0
        set index $position
        if {$redraw == 1} {redraw}
    }
}

# --------------------------------------------------------------------------
# Delete the specified group. All of the groups sequences will be placed in the previous group.
# args:     groupName - The name of the group to delete.
proc ::SeqEditWidget::deleteGroup {groupName {redraw 1}} {

    variable groupNames
    variable groupMap

    # Find the group in the list.
    set index [lsearch $groupNames $groupName]
    if {$index != -1 && [llength $groupNames] > 1} {

        if {$index > 0} {

            # Figure out the previous group.
            set previousGroup [lindex $groupNames [expr $index-1]]

            # Remove this group from the list.
            set groupNames [lreplace $groupNames $index $index]

            # Append this group's sequences to the end of the previous group.
            set groupMap($previousGroup,sequenceIDs) [concat $groupMap($previousGroup,sequenceIDs) $groupMap($groupName,sequenceIDs)]
            set groupMap($previousGroup,numberSequences) [llength $groupMap($previousGroup,sequenceIDs)]

            # Remove the sequences from this group.
            unset groupMap($groupName,sequenceIDs)
            unset groupMap($groupName,numberSequences)

        } else {

            # Figure out the next group.
            set nextGroup [lindex $groupNames [expr $index+1]]

            # Remove this group from the list.
            set groupNames [lreplace $groupNames $index $index]

            # Append this group's sequences to the beginning of the next group.
            set groupMap($nextGroup,sequenceIDs) [concat $groupMap($groupName,sequenceIDs) $groupMap($nextGroup,sequenceIDs)]
            set groupMap($nextGroup,numberSequences) [llength $groupMap($nextGroup,sequenceIDs)]

            # Remove the sequences from this group.
            unset groupMap($groupName,sequenceIDs)
            unset groupMap($groupName,numberSequences)
        }

        if {$redraw == 1} {redraw}
    }
}

# --------------------------------------------------------------------------
# Sets the groups that are currently in the editor. Sequences that are in a group whose name matches
# the name of a new group, will be put into that group. All other sequences will be put into the first
# group.
# args:     newGroupNames - A list of the new group names.
proc ::SeqEditWidget::setGroups {newGroupNames {redraw 1} {defaultGroupName ""}} {

    variable groupNames
    variable groupMap

    # Get the initial sequences in the new first group.
    if {$defaultGroupName == ""} {
        set defaultGroupName [lindex $newGroupNames 0]
    }
    set defaultGroupSequenceIDs {}

    # Go through the existing groups and if they do not have a match, move the sequences to the new first group.
    foreach groupName $groupNames {

        # See if this group has no match or it is the new first group.
        if {[lsearch $newGroupNames $groupName] == -1 || $groupName == $defaultGroupName} {

            # Add the sequences to the new first group.
            set sequenceIDs $groupMap($groupName,sequenceIDs)
            set numberSequences $groupMap($groupName,numberSequences)
            foreach sequenceID $sequenceIDs {
                lappend defaultGroupSequenceIDs $sequenceID
            }

            # Zero out the old group.
            set groupMap($groupName,sequenceIDs) {}
            set groupMap($groupName,numberSequences) 0
        }
    }

    # Set the new first group.
    set groupMap($defaultGroupName,sequenceIDs) $defaultGroupSequenceIDs
    set groupMap($defaultGroupName,numberSequences) [llength $defaultGroupSequenceIDs]

    # Set the new group name list.
    set groupNames $newGroupNames

    # Create any new groups that do not yet exist.
    foreach groupName $groupNames {
        if {[info exists groupMap($groupName,sequenceIDs)] == 0} {
            set groupMap($groupName,sequenceIDs) {}
            set groupMap($groupName,numberSequences) 0
        }
    }

    # Redraw the editor, if we were supposed to.
    if {$redraw == 1} {redraw}
}

# --------------------------------------------------------------------------
# Gets a list of the current group names.
# return:   A list of the groups currently in the editor.
proc ::SeqEditWidget::getGroups {} {

    variable groupNames

    return $groupNames
}

# --------------------------------------------------------------------------
# Gets the group name that the specified sequence is currently a member of.
# arguments: seqId - sequence to find
# return: group name of the group containing the sequence, or "" if it was not found.
proc ::SeqEditWidget::getGroup {sequenceId} {

    variable groupMap
    variable firstGroup
    variable groupNames

    foreach groupName $groupNames {
        if {[lsearch $groupMap($groupName,sequenceIDs) $sequenceId] != -1} {
            return $groupName
        }
    }

    return "Unknown"
}

# --------------------------------------------------------------------------
# Gets all of the sequence ids of the sequences that are in the specified group.
# args:     groupName - The name of the group.
# return:   The sequence ids that is at the position.
proc ::SeqEditWidget::getSequencesInGroup {groupName} {

    variable groupMap

    return $groupMap($groupName,sequenceIDs)
}

# --------------------------------------------------------------------------
# Gets the sequence id of the sequence that is at the specified index of the specified group.
# args:     groupName - The name of the group.
#           sequenceIndex - The index of the sequence.
# return:   The sequence id that is at the position.
proc ::SeqEditWidget::getSequenceInGroup {groupName sequenceIndex} {

    variable groupMap
#    puts "seqedit_widget.tcl.getSequenceInGroup. groupName: $groupName, seqIdx: $sequenceIndex, groupMap [array get groupMap]"
    return [lindex $groupMap($groupName,sequenceIDs) $sequenceIndex]
}

# --------------------------------------------------------------------------
# Gets the sequence ids that are contained in the specified range of grouped sequences.
# args:     startGroupIndex - The index of the starting group.
#           startSequenceIndex - The index of the starting sequence in the starting group.
#           endGroupIndex - The index of the ending sequence in the starting group.
#           endSequenceIndex - The index of the ending sequence in the starting group.
# return:   A list of the sequence ids that are in the specified range.
proc ::SeqEditWidget::getSequencesInGroups {startGroupIndex startSequenceIndex endGroupIndex endSequenceIndex} {

    variable groupNames
    variable groupMap

    # If the indexes are out of order, reverse them.
    if {$startGroupIndex > $endGroupIndex || ($startGroupIndex == $endGroupIndex && $startSequenceIndex > $endSequenceIndex)} {
        set temp $startGroupIndex
        set startGroupIndex $endGroupIndex
        set endGroupIndex $temp
        set temp $startSequenceIndex
        set startSequenceIndex $endSequenceIndex
        set endSequenceIndex $temp
    }

    # Go through the groups and get the sequence ids.
    set returningSequenceIDs {}
    for {set i $startGroupIndex} {$i <= $endGroupIndex} {incr i} {

        set sequenceIDs $groupMap([lindex $groupNames $i],sequenceIDs)

        set first 0
        if {$i == $startGroupIndex} {set first $startSequenceIndex}
        set last [expr [llength $sequenceIDs]-1]
        if {$i == $endGroupIndex} {set last $endSequenceIndex}

        for {set j $first} {$j <= $last} {incr j} {
            lappend returningSequenceIDs [lindex $sequenceIDs $j]
        }
    }

    return $returningSequenceIDs
}

# --------------------------------------------------------------------------
# Get the index of a group given its name.
# args:     groupName - The name of the group for which to retrieve the index.
# return:   The index of the group or -1 if it was not found.
proc ::SeqEditWidget::getGroupIndex {groupName} {

    variable groupNames

    return [lsearch $groupNames $groupName]
}

# --------------------------------------------------------------------------
# Remove all of the sequences from a given group.
# args:     groupName - The index of the group from which to remove all of the sequences.
proc ::SeqEditWidget::removeAllSequencesFromGroup {groupName} {

    variable groupMap

    set groupMap($groupName,sequenceIDs) {}
    set groupMap($groupName,numberSequences) 0
}

# --------------------------------------------------------------------------
# Resets the editor selection so that nothing is selected.
proc ::SeqEditWidget::resetMarks {{preserveMarks 0}} {

    variable groupNames
    variable groupMap
    variable markMap

    # Go through each group and create or update entries for its sequences.
    foreach groupName $groupNames {
        set sequenceIDs $groupMap($groupName,sequenceIDs)
        foreach sequenceID $sequenceIDs {
            if {[info exists markMap($sequenceID)] == 0 || $preserveMarks == 0} {
                set markMap($sequenceID) 0
            } else {
            }
        }
    }
}

# --------------------------------------------------------------------------
# Gets the currently marked sequences.
# return:   A list of the sequence ids that are currently marked.
proc ::SeqEditWidget::getMarkedSequences {} {

    variable groupNames
    variable groupMap
    variable markMap

    # Go through the selection list and get all of the currently selected sequence ids.
    set markedSequenceIDs {}
    foreach groupName $groupNames {
        set sequenceIDs $groupMap($groupName,sequenceIDs)
        foreach sequenceID $sequenceIDs {
            if {$markMap($sequenceID) == 1} {
                lappend markedSequenceIDs $sequenceID
            }
        }
    }

        return $markedSequenceIDs
}

# --------------------------------------------------------------------------
# Set the passed in sequences ids to the specified mark state.
# args:     sequenceIDs - The list of sequence ids that should be marked.
proc ::SeqEditWidget::setMarksOnSequences {sequenceIDs {value 1}} {

    variable markMap

    # Initialize the list of sequences to redraw.
    set sequenceIDsToRedraw {}

    # Mark the sequence as selected in the mark map.
    foreach sequenceID $sequenceIDs {
        set markMap($sequenceID) $value
        lappend sequenceIDsToRedraw $sequenceID
    }

    # Show the selection changes.
    redraw $sequenceIDsToRedraw
}

# --------------------------------------------------------------------------
# Gets the color mapper used by the editor to map metrics to a color.
# return:   The current color map handler.
proc ::SeqEditWidget::getColorMap {} {

    variable colorMap
    return $colorMap
}

# --------------------------------------------------------------------------
# Sets the color mapper used by the editor to map metrics to a color.
# args:     newcolorMap - The new color mapper.
proc ::SeqEditWidget::setColorMap {newColorMap} {

    variable colorMap

    # Save the new color mapp.
    set colorMap $newColorMap

    # Redraw the widget.
    redraw
}

# --------------------------------------------------------------------------
# This method resets the coloring of the specified sequences. It can also be used to initialize the
# coloring map for new sequences.
# args:     sequenceIDs - The sequences to reset or all to reset every sequence in the editor.
proc ::SeqEditWidget::resetColoring {{sequenceIDs "all"} {redraw 1}} {

   variable colorMap

   # If we are resetting everything, get all of the sequence ids.
   if {$sequenceIDs == "all"} {
      set sequenceIDs [getSequences]
   }

   # Get the index of the background color.
   set colorIndex [$colorMap\::getColorIndexForName "white"]

   # Reset the coloring of the specified sequences.
   foreach sequenceID $sequenceIDs {
#      puts "setting seq color..  seqID(2): $sequenceID, length(4): [expr [seq length $sequenceID]-1], color(5): $colorIndex"
      seq set color $sequenceID 0 [expr [seq length $sequenceID]-1] $colorIndex
   }

   if {$redraw == 1} {redraw $sequenceIDs}
}

# --------------------------------------------------------------------------
# Gets the current coloring of the sequences.
# return:   The current coloring map.
proc ::SeqEditWidget::getColoring {} {

    variable coloringMap

    return [array get coloringMap]
}

# --------------------------------------------------------------------------
# Sets coloring of the specified sequences using the specified coloringMetric.
# args:     coloringMetric - The metric to use for the calculation
#           sequenceIDs - The sequence ids of which to set the coloring
#           colorByGroup - 1 if coloring metric should be run per group, else 0
proc ::SeqEditWidget::setColoring {coloringMetric {sequenceIDs "all"} \
                               {colorByGroup 0} {redraw 1} {fullControl 0}} {
   variable groupNames

#   puts "seqedit_widget.setColoring. metric: $coloringMetric, seqIDs: $sequenceIDs, cbg: $colorByGroup, redraw: $redraw, fullCont: $fullControl"

   if {$coloringMetric != ""} {

      # If this metric needs full control, just call it.
      if {$fullControl} {

         # Run the metric.
         eval [join $coloringMetric " "]

      # Otherwise, perform some processing.
      } else {

         # If we are coloring everything, get the ids.
         if {$sequenceIDs == "all"} {
            set sequenceIDs [getSequences]
         }

         # If we are coloring by group, process one group at a time.
         if {$colorByGroup} {

            # Go through each group.
            foreach groupName $groupNames {

               # Get subset of the passed in sequences that are in this group.
               set metricSequenceIDs {}
               set groupSequenceIDs [::SeqEditWidget::getSequencesInGroup \
                                                                $groupName]
               foreach sequenceID $sequenceIDs {
                  if {[lsearch $groupSequenceIDs $sequenceID] != -1} {
                     lappend metricSequenceIDs $sequenceID
                  }
               }

               # Process this set.
               $coloringMetric $metricSequenceIDs
            }

         # Otherwise, process all of the sequences at once.
         } else {
            # Get the coloring metrics for the sequences.
            $coloringMetric $sequenceIDs
         }
      }

      if {$redraw == 1} {redraw $sequenceIDs}
   }
} ; # end of setColoring

# --------------------------------------------------------------------------
proc ::SeqEditWidget::getRepresentations {sequenceIDs} {

    variable representationMap

    # Go through each sequence id and add its representation to the list.
    set ret {}
    foreach sequenceID $sequenceIDs {
        lappend ret $representationMap($sequenceID)
    }

    return $ret
}

# --------------------------------------------------------------------------
proc ::SeqEditWidget::setRepresentations {sequenceIDs representations {redraw 1}} {

    variable representationMap

    # Make sure we have a valid representation list.
    if {[llength $representations] == 1 || [llength $representations] == [llength $sequenceIDs]} {

        # Go through each sequence id and set its representation.
        for {set i 0} {$i < [llength $sequenceIDs]} {incr i} {
            set sequenceID [lindex $sequenceIDs $i]
            if {[llength $representations] == 1} {
                set representationMap($sequenceID) [lindex $representations 0]
            } else {
                set representationMap($sequenceID) [lindex $representations $i]
            }
        }
    }

    if {$redraw == 1} {redraw $sequenceIDs}
}
# --------------------------------------------------------------------------




############################# PRIVATE METHODS #################################
# Methods in this section should only be called from this file.               #
###############################################################################


proc ::SeqEditWidget::validateScrollRange {{checkHorizontal 1} {checkVertical 1}} {

    variable numberCols
    variable numberRows
    variable groupNames
    variable groupMap
    variable firstElement
    variable numberElements
    variable firstGroup
    variable firstSequence

    # Check the horizontal scroll range.
    if {$checkHorizontal == 1} {
        if {$numberElements > $numberCols} {
            if {$firstElement > ($numberElements-$numberCols+1)} {
                set firstElement [expr $numberElements-$numberCols+1]
            } elseif {$firstElement < 0} {
                set firstElement 0
            }
        } else {
            set firstElement 0
        }
    }

    # Check the vertical scroll range.
    if {$checkVertical == 1} {
        # Figure out what the maximum values are for the first group and first sequence.
        set maxGroup 0
        set maxSequence -1
        set range [expr $numberRows-1]
        for {set i [expr [llength $groupNames]-1]} {$i >= 0} {incr i -1} {

            set numberSequences $groupMap([lindex $groupNames $i],numberSequences)

            # See if the sequences from this group fit.
            if {$range <= $numberSequences} {
                set maxGroup $i
                set maxSequence [expr $numberSequences-$range]
                break;
            }
            incr range -$numberSequences

            # See if the title row for this group fits.
            if {$range == 1} {
                set maxGroup $i
                set maxSequence -1
                break;
            }
            incr range -1
        }

        # If the group and sequence are past the limit, set them to the limit.
        if {$firstGroup == $maxGroup} {
            if {$firstSequence > $maxSequence} {
                set firstSequence $maxSequence
            }
        } elseif {$firstGroup > $maxGroup} {
            set firstGroup $maxGroup
            set firstSequence $maxSequence
        }
    }
}

# ----------------------------------------------------------------------------
# Sets the scroll bars.
proc ::SeqEditWidget::setScrollbars {} {

    variable widget
    variable groupNames
    variable groupMap
    variable firstElement
    variable firstGroup
    variable firstSequence
    variable numberCols
    variable numberRows
    variable numberElements

    # Set the scroll bars.
    if {$numberElements > $numberCols} {
        $widget.bottom.xscroll set [expr $firstElement/($numberElements.0+1.0)] [expr ($firstElement+$numberCols)/($numberElements.0+1.0)]
    } else {
        $widget.bottom.xscroll set 0 1
    }



    #
    set currentLine 0
    set maxLines 0
    set numberGroups [expr [llength $groupNames]]
    for {set i 0} {$i < $numberGroups} {incr i} {

        # Get the number of sequences in this group.
        set numberSequences $groupMap([lindex $groupNames $i],numberSequences)

        # Increment the max totals.
        incr maxLines 1
        incr maxLines $numberSequences

        # If we have not yet reached the current group, increment the current totals.
        if {$i < $firstGroup} {
            incr currentLine 1
            incr currentLine $numberSequences
        } elseif {$i == $firstGroup} {
            incr currentLine [expr $firstSequence+1]
        }
    }

    if {$maxLines >= $numberRows} {
        $widget.center.yscroll set [expr $currentLine/double($maxLines+1)] [expr ($currentLine+$numberRows)/double($maxLines+1)]
    } else {
        $widget.center.yscroll set 0 1
    }
    catch {
       drawBox
    } drawImageErr
    #puts "Multiseq Zoom) Error in drawbox in setScrollbars: $drawImageErr"
}; # end setScrollbars

# ----------------------------------------------------------------------------
# Creates a new grid of cells in the editor
proc ::SeqEditWidget::createCells {} {

    variable editor
    variable numberCols

    # Create all of the columns for the editor
    createHeaderColumn
    for {set i 0} {$i < $numberCols} {incr i} {
        createColumn $i
    }
}

# ----------------------------------------------------------------------------
proc ::SeqEditWidget::setCurrentTitle {} {
   variable currentTitle
   variable columnObjectMap
   variable editor

   $editor itemconfigure $columnObjectMap(h,h.textid) \
                  -text [::SeqEditWidget::getCurrentTitleText $currentTitle]
   redraw
}

# ----------------------------------------------------------------------------
# Creates a new header column in the editor.
proc ::SeqEditWidget::createHeaderColumn {} {

    variable editor
    variable cellHeight
    variable headerCellWidth
    variable headerCellHeight
    variable columnObjectMap
    variable numberRows
    variable headerColorActive
    variable headerColorForeground
    variable cellColorInactive
    variable cellColorActive
    variable cellColorForeground
    variable headerNumberingColor
    variable checkColorActive
    variable infobuttonColorActive
    variable infobuttonFontColorActive
    variable repbuttonColorActive
    variable repbuttonFontColorActive
    variable vmdbuttonColorActive
    variable vmdbuttonFontColorActive
    variable headerFont
    variable cellFont
    variable cellFontNumberWidth
    variable buttonFont
    variable currentTitle

    # Set the starting location.
    set x 4
    set y 2

    # Create the header cell.
    set cellx1 $x
    set cellx2 [expr $headerCellWidth]
    set cellxc [expr $cellx1+(($cellx2-$cellx1)/2)]
    set celly1 $y
    set celly2 [expr $headerCellHeight-2]
    set cellyc [expr $celly1+(($celly2-$celly1)/2)]
    set boxid [$editor create rectangle $cellx1 $celly1 $cellx2 $celly2 -fill $headerColorActive -outline $headerColorActive]
    set textid [$editor create text $cellx1 $cellyc -font $headerFont -anchor w -text [::SeqEditWidget::getCurrentTitleText $currentTitle]]

    bindMouseCommands $editor $textid "::SeqEditWidget::click_titleHeader %x %y"

    # Create the separator and tick lines.
    set separatorid [$editor create line $cellx1 [expr $celly2+1] [expr $cellx2+1] [expr $celly2+1] -fill $headerColorForeground]
    set tickid [$editor create line [expr $cellx2-1] $celly1 [expr $cellx2-1] [expr $celly2+1] -fill $headerColorForeground]

    # Store the header cell objects.
    set columnObjectMap(h,h.boxid) $boxid
    set columnObjectMap(h,h.textid) $textid
    set columnObjectMap(h,h.separatorid) $separatorid
    set columnObjectMap(h,h.tickid) $tickid

    # Go through each row and create its row header.
    set y $headerCellHeight
    for {set row 0} {$row < $numberRows} {incr row} {

        # Create the cell for this row.
        set celly1 $y
        set celly2 [expr $celly1+$cellHeight-1]
        set cellyc [expr $celly1+(($celly2-$celly1)/2)]
        set columnObjectMap(h,$row.x1) $cellx1
        set columnObjectMap(h,$row.x2) $cellx2
        set columnObjectMap(h,$row.y1) $celly1
        set columnObjectMap(h,$row.y2) $celly2

        # Create the checkbox.
        if {[expr $celly2-$celly1-5] < 10} {
            set checkboxSize [expr $celly2-$celly1-5]
            set checkboxx1 [expr $cellx1+2]
            set checkboxy1 [expr $celly1+2]
            set checkboxx2 [expr $checkboxx1+$checkboxSize]
            set checkboxy2 [expr $checkboxy1+$checkboxSize]
            set checkboxid [$editor create rectangle $checkboxx1 $checkboxy1 $checkboxx2 $checkboxy2 -fill $cellColorActive -outline $cellColorForeground]
            set checkid [$editor create rectangle [expr $checkboxx1+1] [expr $checkboxy1+1] [expr $checkboxx2-1] [expr $checkboxy2-1] -fill $checkColorActive -outline $checkColorActive]
        } else {
            set checkboxSize 10
            set checkboxx1 [expr $cellx1+2]
            set checkboxy1 [expr $cellyc-5]
            set checkboxx2 [expr $checkboxx1+$checkboxSize]
            set checkboxy2 [expr $checkboxy1+$checkboxSize]
            set checkboxid [$editor create rectangle $checkboxx1 $checkboxy1 $checkboxx2 $checkboxy2 -fill $cellColorActive -outline $cellColorForeground]
            set checkid [$editor create polygon [expr $checkboxx1+2] [expr $checkboxy1+4] [expr $checkboxx1+4] [expr $checkboxy1+6] [expr $checkboxx1+8] [expr $checkboxy1+2]  [expr $checkboxx1+8] [expr $checkboxy1+4] [expr $checkboxx1+4] [expr $checkboxy1+8] [expr $checkboxx1+2] [expr $checkboxy1+6] -fill $checkColorActive -outline $checkColorActive]
        }
        set columnObjectMap(h,$row.checkboxid) $checkboxid
        set columnObjectMap(h,$row.checkid) $checkid
        bindMouseCommands $editor $checkboxid "::SeqEditWidget::click_rowcheckbox %x %y"
        bindMouseCommands $editor $checkid "::SeqEditWidget::click_rowcheckbox %x %y"

        # Create the cell text for this row, if we have a font.
        if {$cellFont != ""} {

            set textid [$editor create text [expr $cellx1+$checkboxSize+4] $cellyc -font $cellFont -anchor w]
            set columnObjectMap(h,$row.textid) $textid
            set columnObjectMap(h,$row.textstring) ""
            set columnObjectMap(h,$row.font) $cellFont
            bindMouseCommands $editor $textid "::SeqEditWidget::click_rowheader %x %y normal" "::SeqEditWidget::click_rowheader %x %y shift" "::SeqEditWidget::click_rowheader %x %y control" "" "::SeqEditWidget::drag_rowheader %x %y" "::SeqEditWidget::click_rowheader %x %y release" "::SeqEditWidget::rightclick_rowheader %x %y"
        }

        # Create the cell numbering for this row, if we have a font.
        set numberWidth 0
        if {$cellFont != ""} {

            set numberWidth [expr ($cellFontNumberWidth*4)+4]
            set numberid [$editor create text [expr $cellx2-4] $cellyc -font $cellFont -anchor e -fill $headerNumberingColor]
            set columnObjectMap(h,$row.numberid) $numberid
            set columnObjectMap(h,$row.numberstring) ""
            bindMouseCommands $editor $numberid "::SeqEditWidget::click_rowheader %x %y normal" "::SeqEditWidget::click_rowheader %x %y shift" "::SeqEditWidget::click_rowheader %x %y control" "" "::SeqEditWidget::drag_rowheader %x %y" "::SeqEditWidget::click_rowheader %x %y release" "::SeqEditWidget::rightclick_rowheader %x %y"
        }

        # Create the info button.
        if {[expr $celly2-$celly1-5] < 10} {
            set infobuttonSize [expr $celly2-$celly1-5]
            set infobuttonx1 [expr $cellx2-$numberWidth-4-$infobuttonSize]
            set infobuttony1 [expr $celly1+2]
            set infobuttonx2 [expr $infobuttonx1+$infobuttonSize]
            set infobuttony2 [expr $infobuttony1+$infobuttonSize]
            set infobuttonid [$editor create rectangle $infobuttonx1 $infobuttony1 $infobuttonx2 $infobuttony2 -fill $infobuttonColorActive -outline $cellColorForeground]
            set infobuttontextid -1
        } else {
            set infobuttonSize 10
            set infobuttonx1 [expr $cellx2-$numberWidth-4-$infobuttonSize]
            set infobuttony1 [expr $cellyc-5]
            set infobuttonx2 [expr $infobuttonx1+$infobuttonSize+1]
            set infobuttony2 [expr $infobuttony1+$infobuttonSize]
            set infobuttonid [$editor create rectangle $infobuttonx1 $infobuttony1 $infobuttonx2 $infobuttony2 -fill $infobuttonColorActive -outline $cellColorForeground]
            set infobuttontextid [$editor create text [expr $infobuttonx1+3] [expr $infobuttony1-1] -font $buttonFont -anchor nw -text "i" -fill $infobuttonFontColorActive]
        }
        set columnObjectMap(h,$row.infobuttonid) $infobuttonid
        set columnObjectMap(h,$row.infobuttontextid) $infobuttontextid
        bindMouseCommands $editor $infobuttonid "::SeqEditWidget::click_rownotes %x %y"
        if {$infobuttontextid != -1} {
            bindMouseCommands $editor $infobuttontextid "::SeqEditWidget::click_rownotes %x %y"
        }

        # Create the representation button.
        if {[expr $celly2-$celly1-5] < 10} {
            set repbuttonSize [expr $celly2-$celly1-5]
            set repbuttonx1 [expr $cellx2-$numberWidth-4-($repbuttonSize*2)-2]
            set repbuttony1 [expr $celly1+2]
            set repbuttonx2 [expr $repbuttonx1+$repbuttonSize]
            set repbuttony2 [expr $repbuttony1+$repbuttonSize]
            set repbuttonid [$editor create rectangle $repbuttonx1 $repbuttony1 $repbuttonx2 $repbuttony2 -fill $repbuttonColorActive -outline $cellColorForeground]
            set repbuttontextid -1
        } else {
            set repbuttonSize 10
            set repbuttonx1 [expr $cellx2-$numberWidth-4-($repbuttonSize*2)-2]
            set repbuttony1 [expr $cellyc-5]
            set repbuttonx2 [expr $repbuttonx1+$repbuttonSize+1]
            set repbuttony2 [expr $repbuttony1+$repbuttonSize]
            set repbuttonid [$editor create rectangle $repbuttonx1 $repbuttony1 $repbuttonx2 $repbuttony2 -fill $repbuttonColorActive -outline $cellColorForeground]
            set repbuttontextid [$editor create text [expr $repbuttonx1+3] [expr $repbuttony1-1] -font $buttonFont -anchor nw -text "r" -fill $repbuttonFontColorActive]
        }
        set columnObjectMap(h,$row.repbuttonid) $repbuttonid
        set columnObjectMap(h,$row.repbuttontextid) $repbuttontextid
        bindMouseCommands $editor $repbuttonid "::SeqEditWidget::click_rowbutton rep %x %y"
        if {$repbuttontextid != -1} {
            bindMouseCommands $editor $repbuttontextid "::SeqEditWidget::click_rowbutton rep %x %y"
        }

        # Create the vmd button.
        if {[expr $celly2-$celly1-5] < 10} {
            set vmdbuttonSize [expr $celly2-$celly1-5]
            set vmdbuttonx1 [expr $cellx2-$numberWidth-4-($vmdbuttonSize*3)-4]
            set vmdbuttony1 [expr $celly1+2]
            set vmdbuttonx2 [expr $vmdbuttonx1+$vmdbuttonSize]
            set vmdbuttony2 [expr $vmdbuttony1+$vmdbuttonSize]
            set vmdbuttonid [$editor create rectangle $vmdbuttonx1 $vmdbuttony1 $vmdbuttonx2 $vmdbuttony2 -fill $vmdbuttonColorActive -outline $cellColorForeground]
            set vmdbuttontextid -1
        } else {
            set vmdbuttonSize 10
            set vmdbuttonx1 [expr $cellx2-$numberWidth-4-($vmdbuttonSize*3)-4]
            set vmdbuttony1 [expr $cellyc-5]
            set vmdbuttonx2 [expr $vmdbuttonx1+$vmdbuttonSize+1]
            set vmdbuttony2 [expr $vmdbuttony1+$vmdbuttonSize]
            set vmdbuttonid [$editor create rectangle $vmdbuttonx1 $vmdbuttony1 $vmdbuttonx2 $vmdbuttony2 -fill $vmdbuttonColorActive -outline $cellColorForeground]
            set vmdbuttontextid [$editor create text [expr $vmdbuttonx1+3] [expr $vmdbuttony1-1] -font $buttonFont -anchor nw -text "v" -fill $vmdbuttonFontColorActive]
        }
        set columnObjectMap(h,$row.vmdbuttonid) $vmdbuttonid
        set columnObjectMap(h,$row.vmdbuttontextid) $vmdbuttontextid
        bindMouseCommands $editor $vmdbuttonid "::SeqEditWidget::click_rowbutton vmd %x %y"
        if {$vmdbuttontextid != -1} {
            bindMouseCommands $editor $vmdbuttontextid "::SeqEditWidget::click_rowbutton vmd %x %y"
        }

        # Create the background.
        set boxid [$editor create rectangle $cellx1 $celly1 $cellx2 $celly2 -fill $cellColorInactive -outline $cellColorInactive]
        set separatorid [$editor create line $cellx1 $celly2 $cellx2 $celly2 -fill $cellColorInactive]
        set tickid [$editor create line [expr $cellx2-1] $celly1 [expr $cellx2-1] $celly2 -fill $cellColorInactive]
        set columnObjectMap(h,$row.boxid) $boxid
        set columnObjectMap(h,$row.boxcolor) $cellColorInactive
        set columnObjectMap(h,$row.separatorid) $separatorid
        set columnObjectMap(h,$row.tickid) $tickid
        bindMouseCommands $editor $boxid "::SeqEditWidget::click_rowheader %x %y normal" "::SeqEditWidget::click_rowheader %x %y shift" "::SeqEditWidget::click_rowheader %x %y control" "" "::SeqEditWidget::drag_rowheader %x %y" "::SeqEditWidget::click_rowheader %x %y release" "::SeqEditWidget::rightclick_rowheader %x %y"

        # Mark that the header is in an inactive state.
        set columnObjectMap(h,$row.active) 0

        # Move the y down.
        set y [expr $celly2+1]
    }
}

# ----------------------------------------------------------------------------
# Creates a new column in the editor.
# args:     col - The index of the column to create.
proc ::SeqEditWidget::createColumn {col} {

    variable editor
    variable cellWidth
    variable cellHeight
    variable headerCellWidth
    variable headerCellHeight
    variable columnObjectMap
    variable numberRows
    variable headerColorInactive
    variable cellColorInactive
    variable cellColorForeground
    variable headerFont
    variable cellFont

    # Set the starting location.
    set x $headerCellWidth
    set y 2

    # Create the header cell.
    set cellx1 [expr $x+$col*$cellWidth]
    set cellx2 [expr $cellx1+$cellWidth-1]
    set cellxc [expr $cellx1+($cellWidth/2)]
    set cellxq1 [expr $cellx1+($cellWidth/4)-1]
    set cellxq3 [expr $cellx2-($cellWidth/4)+1]
    set celly1 $y
    set celly2 [expr $headerCellHeight-2]
    set cellyc [expr $celly1+(($celly2-$celly1)/2)]
    set boxid [$editor create rectangle $cellx1 $celly1 [expr $cellx2+1] [expr $celly2+1] -fill $headerColorInactive -outline $headerColorInactive]
    set textid [$editor create text $cellxc $cellyc -font $headerFont -anchor center]

    # Set up the selection bindings.
    bindMouseCommands $editor $boxid  "::SeqEditWidget::click_columnheader %x %y none" "::SeqEditWidget::click_columnheader %x %y shift" "::SeqEditWidget::click_columnheader %x %y control" "" "::SeqEditWidget::move_columnheader %x %y" "::SeqEditWidget::release_columnheader %x %y"
    bindMouseCommands $editor $textid  "::SeqEditWidget::click_columnheader %x %y none" "::SeqEditWidget::click_columnheader %x %y shift" "::SeqEditWidget::click_columnheader %x %y control" "" "::SeqEditWidget::move_columnheader %x %y" "::SeqEditWidget::release_columnheader %x %y"

    # Create the separator and tick lines.
    set separatorid [$editor create line $cellx1 [expr $celly2+1] [expr $cellx2+1] [expr $celly2+1] -fill $headerColorInactive]
    set tickid [$editor create line $cellx2 [expr $celly2-1] $cellx2 [expr $celly2+1] -fill $headerColorInactive]

    # Store the header cell objects.
    set columnObjectMap($col,h.active) 0
    set columnObjectMap($col,h.boxid) $boxid
    set columnObjectMap($col,h.textid) $textid
    set columnObjectMap($col,h.textstring) ""
    set columnObjectMap($col,h.separatorid) $separatorid
    set columnObjectMap($col,h.tickid) $tickid
    set columnObjectMap($col,h.boxcolor) $headerColorInactive
    set columnObjectMap($col,h.x1) $cellx1
    set columnObjectMap($col,h.x2) $cellx2
    set columnObjectMap($col,h.y1) $celly1
    set columnObjectMap($col,h.y2) $celly2

    # If we are overlapping a text object from the previous column header, bring it to the front.
    if {$col > 0} {
        $editor raise $columnObjectMap([expr $col-1],h.textid) $boxid
    }

    # Go through each row and create its components for this column.
    set y $headerCellHeight
    for {set row 0} {$row < $numberRows} {incr row} {

        # Calculate some coordinates.
        set celly1 $y
        set celly2 [expr $celly1+$cellHeight-1]
        set cellyc [expr $celly1+($cellHeight/2)]
        set cellyq1 [expr $celly1+($cellHeight/4)-1]
        set cellyq3 [expr $celly2-($cellHeight/4)+1]

        # Create the box.
        set boxid [$editor create rectangle $cellx1 $celly1 [expr $cellx2+1] [expr $celly2+1] -fill $cellColorInactive -outline $cellColorInactive]
        set columnObjectMap($col,$row.boxid) $boxid
        set columnObjectMap($col,$row.boxcolor) $cellColorInactive
        bindCellObject $boxid

        # Move the box to below the previous column's box so all of the boxes are beneath everything else.
        if {$col > 0} {
            $editor lower $boxid $columnObjectMap([expr $col-1],$row.boxid)
        }

        # Create the text for this cell, if we have a font.
        if {$cellFont != ""} {
            set textid [$editor create text $cellxc $cellyc -state hidden -font $cellFont -anchor center]
            set columnObjectMap($col,$row.textid) $textid
            set columnObjectMap($col,$row.textstring) ""
            bindCellObject $textid
        }

        # Figure out the left and right sides for icons.
        set cellxf $cellx1
        set cellxr [expr $cellx2+1]

        # Create the bar and line for this cell
        set barid [$editor create polygon $cellxf $cellyq1 $cellxr $cellyq1 $cellxr $cellyq3 $cellxf $cellyq3 -state hidden -fill $cellColorInactive -outline $cellColorInactive]
        set lineid [$editor create line $cellxf $cellyc $cellxr $cellyc -state hidden -fill $cellColorForeground]
        set columnObjectMap($col,$row.barid) $barid
        set columnObjectMap($col,$row.barcolor) $cellColorInactive
        set columnObjectMap($col,$row.lineid) $lineid
        set columnObjectMap($col,$row.linecolor) $cellColorInactive
        bindCellObject $barid
        bindCellObject $lineid

        # Create the alpha helix icons for this cell.
        set cellyt [expr $celly1+1]
        set cellyb [expr $celly2-1]
        set alpha0 [$editor create polygon $cellxf $cellyt $cellxr $cellyt $cellxr $cellyc $cellxf $cellyb -state hidden -fill $cellColorInactive -outline $cellColorInactive]
        set columnObjectMap($col,$row.alpha0id) $alpha0
        set columnObjectMap($col,$row.alpha0color) $cellColorInactive
        set alpha1 [$editor create polygon $cellxf $cellyt $cellxr $cellyt $cellxr $cellyb $cellxf $cellyc -state hidden -fill $cellColorInactive -outline $cellColorInactive]
        set columnObjectMap($col,$row.alpha1id) $alpha1
        set columnObjectMap($col,$row.alpha1color) $cellColorInactive
        set alpha2 [$editor create polygon $cellxf $cellyt $cellxr $cellyc $cellxr $cellyb $cellxf $cellyb -state hidden -fill $cellColorInactive -outline $cellColorInactive]
        set columnObjectMap($col,$row.alpha2id) $alpha2
        set columnObjectMap($col,$row.alpha2color) $cellColorInactive
        set alpha3 [$editor create polygon $cellxf $cellyc $cellxr $cellyt $cellxr $cellyb $cellxf $cellyb -state hidden -fill $cellColorInactive -outline $cellColorInactive]
        set columnObjectMap($col,$row.alpha3id) $alpha3
        set columnObjectMap($col,$row.alpha3color) $cellColorInactive
        bindCellObject $alpha0
        bindCellObject $alpha1
        bindCellObject $alpha2
        bindCellObject $alpha3

        # Create the beta sheet icons for this cell.
        set arrow [$editor create polygon $cellxf $cellyq1 $cellxc $cellyq1 $cellxc $cellyt $cellxr $cellyc $cellxc $cellyb $cellxc $cellyq3 $cellxf $cellyq3 -state hidden -fill $cellColorInactive -outline $cellColorInactive]
        set columnObjectMap($col,$row.arrowid) $arrow
        set columnObjectMap($col,$row.arrowcolor) $cellColorInactive
        bindCellObject $arrow

        # Mark that the row is inactive.
        set columnObjectMap($col,$row.active) 0

        # Move down to the next row.
        set y [expr $celly2+1]
    }
}

# ----------------------------------------------------------------------------
proc ::SeqEditWidget::bindMouseCommands {canvas object {click ""} \
                           {shiftClick ""} {controlClick ""} \
                           {shiftControlClick ""} {motion ""} \
                           {release ""} {rightClick ""}} {

    if {$click != ""} {
        $canvas bind $object <ButtonPress-1> $click
    }
    if {$shiftClick != ""} {
        $canvas bind $object <Shift-ButtonPress-1> $shiftClick
    }
    if {$motion != ""} {
        $canvas bind $object <B1-Motion> $motion
    }
    if {$release != ""} {
        $canvas bind $object <ButtonRelease-1> $release
    }
    if {$::tcl_platform(os) == "Darwin"} {
        if {$controlClick != ""} {
            $canvas bind $object <Command-ButtonPress-1> $controlClick
        }
        if {$shiftControlClick != ""} {
            $canvas bind $object <Shift-Command-ButtonPress-1> $shiftControlClick
        }
        if {$rightClick != ""} {
            $canvas bind $object <Control-ButtonPress-1> $rightClick
        }
        if {$rightClick != ""} {
            $canvas bind $object <ButtonPress-2> $rightClick
        }
    } else {
        if {$controlClick != ""} {
            $canvas bind $object <Control-ButtonPress-1> $controlClick
        }
        if {$shiftControlClick != ""} {
            $canvas bind $object <Shift-Control-ButtonPress-1> $shiftControlClick
        }
        if {$rightClick != ""} {
            $canvas bind $object <ButtonPress-3> $rightClick
        }
    }
}

# ----------------------------------------------------------------------------
proc ::SeqEditWidget::bindCellObject {objectID} {

    variable editor

    bindMouseCommands $editor $objectID \
                 "::SeqEditWidget::click_cell %x %y none" \
                 "::SeqEditWidget::click_cell %x %y shift" \
                 "::SeqEditWidget::click_cell %x %y control" \
                 "::SeqEditWidget::click_cell %x %y {shift control}" \
                 "::SeqEditWidget::move_cell %x %y" \
                 "::SeqEditWidget::release_cell %x %y"
}

# ----------------------------------------------------------------------------
# Delete the cells in the current editor.
proc ::SeqEditWidget::deleteCells {} {

    variable editor
    variable columnObjectMap

    # Get a list of all of the objects on the canvas.
    set objectNames [array names columnObjectMap]

    # Delete each object.
    foreach objectName $objectNames {
        if {[string first "id" $objectName] != -1} {
            $editor delete $columnObjectMap($objectName)
        }
    }

    # Reinitialize the object map.
    unset columnObjectMap
    array unset columnObjectMap 
}


# ----------------------------------------------------------------------------
proc ::SeqEditWidget::redrawColumnHeader {} {
   variable firstElement
   variable numberCols
   variable numberElements
   variable selectionColor
   variable headerColorActive
   variable columnObjectMap
   variable headertext
   variable editor
   variable headerColor
   variable headerColorForeground
   variable headerColorInactive

   set elementIndex $firstElement
   for {set col 0} {$col < $numberCols} {incr col} {

      # See if this column has data in it.
      if {$elementIndex < $numberElements} {

         # Get the header text.
         set headertext ""
         if {$elementIndex == 0 || [expr $elementIndex%10] == 9} {
            set headertext "[expr $elementIndex+1]"
         }

         # See if the header is selected.
         if { [isColumnSelected $elementIndex] } {
            set headerColor $selectionColor
         } else {
            set headerColor $headerColorActive
         }

         # Update the parts of the cell that have changed.
         set columnObjectMap($col,h.active) 1
         if {$columnObjectMap($col,h.textstring) != $headertext} {
            set columnObjectMap($col,h.textstring) $headertext
            $editor itemconfigure $columnObjectMap($col,h.textid) \
                                                   -text $headertext
         }
         if {$columnObjectMap($col,h.boxcolor) != $headerColor} {
            set columnObjectMap($col,h.boxcolor) $headerColor
            $editor itemconfigure $columnObjectMap($col,h.boxid) \
                                                   -fill $headerColor
            $editor itemconfigure $columnObjectMap($col,h.boxid) \
                                               -outline $headerColor
            $editor itemconfigure $columnObjectMap($col,h.separatorid) \
                                           -fill $headerColorForeground
            $editor itemconfigure $columnObjectMap($col,h.tickid) \
                                             -fill $headerColorForeground
         }

      } elseif {$columnObjectMap($col,h.active) == 1} {

         # Draw the cell as inactive.
         set columnObjectMap($col,h.active) 0
         $editor itemconfigure $columnObjectMap($col,h.textid) -text ""
         $editor itemconfigure $columnObjectMap($col,h.boxid) \
                                               -fill $headerColorInactive
         $editor itemconfigure $columnObjectMap($col,h.boxid) \
                                            -outline $headerColorInactive
         $editor itemconfigure $columnObjectMap($col,h.separatorid) \
                                               -fill $headerColorInactive
         $editor itemconfigure $columnObjectMap($col,h.tickid) \
                                               -fill $headerColorInactive

      }
      incr elementIndex
   }

} ; #end of ::SeqEditWidget::redrawColumnHeader helper proc

# ----------------------------------------------------------------------------
# Redraws the widget.
# args: redrawSequenceID - (default {}) list of specific sequence ids to redraw
#               or an empty list to redraw all of the sequences.
proc ::SeqEditWidget::redraw {{redrawSequenceID {}}} {
#   puts "\n\n -------------- seqedit_widget.tcl.redraw BEGIN ------------------\n\n"
   variable editor
   variable columnObjectMap
   variable firstElement
   variable firstGroup
   variable firstSequence
   variable numberRows
   variable groupNames
   variable groupMap
   variable numberCols
   variable headerColorActive
   variable cellColorActive
   variable cellColorInactive
   variable cellColorForeground
   variable cellTextReplacementColor
   variable selectionColor
   variable checkColorActive
   variable infobuttonColorActive
   variable infobuttonFontColorActive
   variable repbuttonColorActive
   variable repbuttonFontColorActive
   variable vmdbuttonColorActive
   variable vmdbuttonFontColorActive
   variable colorMap
   variable representationMap
   variable cellFont
   variable groupHeaderFont
   variable markMap
   variable selectionMap

#   createRowMap

   # basically, check to see if the color map has changed.  If so, recache
   # the new
   ::MultiSeq::ColorMap::VMD::loadColorMap

   # Get the function to call to lookup a color from an index.
   set colorMapLookup "$colorMap\::getColor"
#   set colorMapLookup "$colorMap\::getColorValueForIndex"

   # If we are updating the whole editor, redraw the column header row.
   if {$redrawSequenceID == {}} {
      redrawColumnHeader
   }

   # Figure out which group and sequence we are working with.
   set groupIndex $firstGroup
   set maxGroup [expr [llength $groupNames]-1]
   if {$groupIndex <= $maxGroup} {
      set groupName [lindex $groupNames $groupIndex]
      set sequenceIndex $firstSequence
      set numberSequences $groupMap($groupName,numberSequences)
   } else {
      set groupName ""
      set sequenceIndex 0
      set numberSequences 0
   }

   # Go through each row and draw it.
   for {set row 0} {$row < $numberRows} {incr row; incr sequenceIndex} {
#      puts "redrawing row $row.  sequenceIndex is $sequenceIndex."

      # If finished with the current group and there is another one, move to it.
      if {$sequenceIndex >= $numberSequences && $groupIndex < $maxGroup} {

         incr groupIndex
         set groupName [lindex $groupNames $groupIndex]
         set sequenceIndex -1
         set numberSequences $groupMap($groupName,numberSequences)
      }
#      puts "check.  groupIndex: $groupIndex, name: $groupName, seqInd: $sequenceIndex numSeq: $numberSequences"

      # See if this row has a sequence in it.
      if {$sequenceIndex < $numberSequences} {

         # See if this row is a grouping row.
         if {$sequenceIndex == -1} {

            # If just redrawing a set of sequences, we don't need 
            # to worry about grouping rows so continue on.
            if {$redrawSequenceID != {}} {
               continue
            }

            # If not in the correct active state, adjust the order of the items.
            if {$columnObjectMap(h,$row.active) != 1} {

               # Rearrange the necessary items.
               set columnObjectMap(h,$row.active) 1
               if {$cellFont != ""} {
                  $editor raise $columnObjectMap(h,$row.textid) \
                                          $columnObjectMap(h,$row.boxid)
                  $editor lower $columnObjectMap(h,$row.numberid) \
                                             $columnObjectMap(h,$row.boxid)
               }
               $editor lower $columnObjectMap(h,$row.checkboxid) \
                                              $columnObjectMap(h,$row.boxid)
               $editor lower $columnObjectMap(h,$row.checkid) \
                                              $columnObjectMap(h,$row.boxid)
               $editor lower $columnObjectMap(h,$row.infobuttonid) \
                                              $columnObjectMap(h,$row.boxid)
               if {$columnObjectMap(h,$row.infobuttontextid) != -1} {
                  $editor lower $columnObjectMap(h,$row.infobuttontextid) \
                                             $columnObjectMap(h,$row.boxid)
               }
               $editor lower $columnObjectMap(h,$row.repbuttonid) \
                                             $columnObjectMap(h,$row.boxid)
               if {$columnObjectMap(h,$row.repbuttontextid) != -1} {
                  $editor lower $columnObjectMap(h,$row.repbuttontextid) \
                                             $columnObjectMap(h,$row.boxid)
               }
               $editor lower $columnObjectMap(h,$row.vmdbuttonid) \
                                            $columnObjectMap(h,$row.boxid)
               if {$columnObjectMap(h,$row.vmdbuttontextid) != -1} {
                  $editor lower $columnObjectMap(h,$row.vmdbuttontextid) \
                                              $columnObjectMap(h,$row.boxid)
               }
            }

            # Redraw any visible items.
            if {$cellFont != ""} {

               # If the font has changed, update the necessary fields.
               if {$columnObjectMap(h,$row.font) != $groupHeaderFont} {
                  set columnObjectMap(h,$row.font) $groupHeaderFont
                  $editor itemconfigure $columnObjectMap(h,$row.textid) \
                                                    -font $groupHeaderFont
               }

               # If the sequence name has changed, update the name field.
               if {$columnObjectMap(h,$row.textstring) != $groupName} {
                  set columnObjectMap(h,$row.textstring) $groupName
                  $editor itemconfigure $columnObjectMap(h,$row.textid) \
                                                          -text $groupName
               }
            }
            if {$columnObjectMap(h,$row.boxcolor) != $headerColorActive} {
               set columnObjectMap(h,$row.boxcolor) $headerColorActive
               $editor itemconfigure $columnObjectMap(h,$row.boxid) \
                                                    -fill $headerColorActive
               $editor itemconfigure $columnObjectMap(h,$row.boxid) \
                                                -outline $headerColorActive
               $editor itemconfigure $columnObjectMap(h,$row.separatorid) \
                                                  -fill $cellColorForeground
               $editor itemconfigure $columnObjectMap(h,$row.tickid) \
                                                 -fill $cellColorForeground
            }

            # No sequence data is associated with a group header.
            set sequence {}

         } else {               ;# Otherwise this is a regular row. (not
                                 #         grouping)

            # Get the sequence id.
            set sequenceID [lindex $groupMap($groupName,sequenceIDs) \
                                                          $sequenceIndex]
#            puts -nonewline "seqedit_widget.tcl redraw seqId: $sequenceID "
            # If just redrawing a set of sequences and this is 
            #not one of them, continue on.
            if {$redrawSequenceID != {}} {
               if {[lsearch $redrawSequenceID $sequenceID] == -1} {
                  continue
               }
            }

            # If not in the correct active state, adjust the order of the items.
            if {$columnObjectMap(h,$row.active) != 2} {

               # Rearrange the necessary items.
               set columnObjectMap(h,$row.active) 2
               if {$cellFont != ""} {
                  $editor raise $columnObjectMap(h,$row.textid) \
                                              $columnObjectMap(h,$row.boxid)
                  $editor raise $columnObjectMap(h,$row.numberid) \
                                              $columnObjectMap(h,$row.boxid)
               }
               $editor raise $columnObjectMap(h,$row.checkboxid) \
                                           $columnObjectMap(h,$row.boxid)
               $editor raise $columnObjectMap(h,$row.infobuttonid) \
                                           $columnObjectMap(h,$row.boxid)
               if {$columnObjectMap(h,$row.infobuttontextid) != -1} {
                  $editor raise $columnObjectMap(h,$row.infobuttontextid) \
                                           $columnObjectMap(h,$row.infobuttonid)
               }
               $editor raise $columnObjectMap(h,$row.repbuttonid) \
                                           $columnObjectMap(h,$row.boxid)
               if {$columnObjectMap(h,$row.repbuttontextid) != -1} {
                  $editor raise $columnObjectMap(h,$row.repbuttontextid) \
                                           $columnObjectMap(h,$row.repbuttonid)
               }
            }

            # If this sequence is marked, show the check.
            if {$sequenceID != "" && $markMap($sequenceID) == 1} {
               $editor raise $columnObjectMap(h,$row.checkid) \
                                      $columnObjectMap(h,$row.checkboxid)
            } else {
               $editor lower $columnObjectMap(h,$row.checkid) \
                                      $columnObjectMap(h,$row.boxid)
            }

            # If we have a structure, show the vmd button.

            if { [::SeqData::hasStruct $sequenceID] == "Y"} {
               $editor raise $columnObjectMap(h,$row.vmdbuttonid) \
                                      $columnObjectMap(h,$row.boxid)
               if {$columnObjectMap(h,$row.vmdbuttontextid) != -1} {
                  $editor raise $columnObjectMap(h,$row.vmdbuttontextid) \
                                         $columnObjectMap(h,$row.vmdbuttonid)
               }
            } else {
               $editor lower $columnObjectMap(h,$row.vmdbuttonid) \
                                      $columnObjectMap(h,$row.boxid)
               if {$columnObjectMap(h,$row.vmdbuttontextid) != -1} {
                  $editor lower $columnObjectMap(h,$row.vmdbuttontextid) \
                                         $columnObjectMap(h,$row.boxid)
               }
            }

            # Redraw any visible items.
            if {$cellFont != ""} {

               # If the font has changed, update the necessary fields.
               if {$columnObjectMap(h,$row.font) != $cellFont} {
                  set columnObjectMap(h,$row.font) $cellFont
                  $editor itemconfigure $columnObjectMap(h,$row.textid) \
                                         -font $cellFont
                  $editor itemconfigure $columnObjectMap(h,$row.numberid) \
                                         -font $cellFont
               }

               # If the row name has changed, update the name field.
               set newName [getRowTitle $sequenceID]
               if {$columnObjectMap(h,$row.textstring) != $newName} {
                  set columnObjectMap(h,$row.textstring) $newName
                  $editor itemconfigure $columnObjectMap(h,$row.textid) \
                                         -text $newName
               }

               # If first residue number has changed, update the number field.
               set firstResidueIndex ""
               set maxElement [expr $firstElement+$numberCols-1]
#puts -nonewline "maxElem: $maxElement"
               if {$maxElement > [::SeqData::getSeqLength $sequenceID]} {
                  set maxElement [::SeqData::getSeqLength $sequenceID]
               }
#puts -nonewline ",maxElem2: $maxElement, res:"
               for {set i $firstElement} {$i < $maxElement} {incr i} {
                  set firstResidueIndex [::SeqData::getResidueForElement \
                                         $sequenceID $i]
#puts -nonewline "$firstResidueIndex,"
                  if {$firstResidueIndex >= 0} {
                     set firstResidueIndex [string trim [join [lrange \
                                            $firstResidueIndex 0 1] ""]]
#puts "$firstResidueIndex"
                     break
                  }
               }
               if {$columnObjectMap(h,$row.numberstring)!= $firstResidueIndex\
                   &&\
                  $firstResidueIndex >= 0} {
                  set columnObjectMap(h,$row.numberstring) $firstResidueIndex
                  $editor itemconfigure $columnObjectMap(h,$row.numberid) \
                                         -text $firstResidueIndex
               }
            }

            # Figure out the color for the box.
            if { [isSequenceSelected $sequenceID] } {
               set headerColor $selectionColor
            } else {
               set headerColor $headerColorActive
            }
            if {$columnObjectMap(h,$row.boxcolor) != $headerColor} {
               set columnObjectMap(h,$row.boxcolor) $headerColor
               $editor itemconfigure $columnObjectMap(h,$row.boxid) \
                                      -fill $headerColor
               $editor itemconfigure $columnObjectMap(h,$row.boxid) \
                                      -outline $headerColor
               $editor itemconfigure $columnObjectMap(h,$row.separatorid) \
                                      -fill $cellColorForeground
               $editor itemconfigure $columnObjectMap(h,$row.tickid) \
                                      -fill $cellColorForeground
            }

            # Get the sequence data for the representation.
            if {$representationMap($sequenceID) == "secondary"} {

               # Get the secondary structure.
               set sequence [lrange [SeqData::getSecondaryStructure \
                                      $sequenceID 1] $firstElement \
                                      [expr $firstElement+$numberCols-1]]
#puts "seqedit_widget.tcl.redraw just set seq (2nd) to $sequence"
               # Figure out the starting helix count.
               set helixCount 0

               # Figure out if the last element is the end of a beta strand.
               set endsWithBetaArrow 0

            } else {
               set sequence [lrange [SeqData::getSeq $sequenceID] \
                           $firstElement [expr $firstElement+$numberCols-1]]
#puts "seqedit_widget.tcl.redraw just set seq (!2nd) to $sequence"
            }
         }

         # Go through each column that has an element in it.
         set col 0
         set elementIndex $firstElement
#         puts "seqedit_widget.tcl.redraw() starting to loop through elements of $sequence"
         foreach element $sequence {

#            puts "element: $element, rep is $representationMap($sequenceID)"
            # See which representations we are showing.
            if {$representationMap($sequenceID) == "sequence"} {

               # If we are not in correct active state, adjust order of items.
               if {$columnObjectMap($col,$row.active) != 1} {

                  # Rearrange the necessary items.
                  set columnObjectMap($col,$row.active) 1
                  if {$cellFont != ""} {
                     $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                       -state normal
                  }
                  $editor itemconfigure $columnObjectMap($col,$row.barid) \
                                                    -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.lineid) \
                                                    -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha0id) \
                                                    -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha1id) \
                                                    -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha2id) \
                                                    -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha3id) \
                                                    -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.arrowid) \
                                                    -state hidden
               }

               # See if we need to adjust the text of the element.
               if {$element == "-"} {

                  set elementColor "#FFFFFF"
                  if {$cellFont == ""} {
                     set element ""
                  } else {
                     set element "."
                  }
               } else {
#                  puts -nonewline stderr "seqedit_widget.redraw() "
#                  flush stderr
#                  puts -nonewline "element is '$element', seqID: '$sequenceID', elemIndx: '$elementIndex', seqColor: '[seq get color $sequenceID $elementIndex]' color: '[$colorMapLookup [seq get color $sequenceID $elementIndex]]"
#                  puts "'"
                  # Get some info about the element.
                  set elementColor [$colorMapLookup [seq get color \
                                                 $sequenceID $elementIndex]]

                  if {$cellFont == "" && $elementColor == "#FFFFFF"} {
                     set elementColor $cellTextReplacementColor
                  }
               }
#               puts "after init set. col:$col, row:$row, seqId:$sequenceID,elemIdx:$elementIndex, elemClr:$elementColor, old: $columnObjectMap($col,$row.boxcolor), isSelected: [isSelected $sequenceID $elementIndex]"
#               puts -nonewline "."
               # See if we need to highlight this element
               if { [isSelected $sequenceID $elementIndex] } {
#                  puts "getting ready to set intensity.  elemClr: $elementColor, seqId:  $sequenceID, elemIndx: $elementIndex"
                  set intensityDecrease [expr -((1.0-[getIntensity \
                                                  $elementColor])/2.0)]
                  set elementColor [getBrightenedColor $selectionColor \
                                    $intensityDecrease $intensityDecrease \
                                    $intensityDecrease]
               }

               # Update the parts of the cell that have changed.
               if {$cellFont != ""} {
                  if {$columnObjectMap($col,$row.textstring) != $element} {
                     set columnObjectMap($col,$row.textstring) $element
                     $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                      -text $element
                  }
               }

               if {$columnObjectMap($col,$row.boxcolor) != $elementColor} {
#                  puts stderr "showing change. col:'$col', row:'$row', seqId:'$sequenceID',elemIdx:'$elementIndex', elemClr:'$elementColor', old: '$columnObjectMap($col,$row.boxcolor)'"
                  set columnObjectMap($col,$row.boxcolor) $elementColor
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                   -fill $elementColor
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                   -outline $elementColor

                  if {[info exists columnObjectMap($col,$row.textid)]} {
                     foreach {r g b} [winfo rgb . $elementColor] { break }

                     set tColor [format "#%02X%02X%02X" \
                                   [expr ($r >> 8) ^ 128] \
                                   [expr ($g >> 8) ^ 128] \
                                   [expr ($b >> 8) ^ 128]] 
#                  puts -nonewline "elm: $elementColor, xor: $tColor, "
#                  set tColor [format "#%02X%02X%02X" \
#                                   [expr (($r >> 8) + 128) & 255] \
#                                   [expr (($g >> 8) + 128) & 255] \
#                                   [expr (($b >> 8) + 128) & 255]] 
#                  puts "add: $tColor"
                     $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                   -fill $tColor
                  }
               }

            } elseif {$representationMap($sequenceID) == "bar"} {

               # If not in correct active state, adjust order of the items.
               if {$columnObjectMap($col,$row.active) != 2} {

                  # Rearrange the necessary items.
                  set columnObjectMap($col,$row.active) 2
                  if {$cellFont != ""} {
                     $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                      -state hidden
                  }
                  $editor itemconfigure $columnObjectMap($col,$row.alpha0id) \
                                                   -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha1id) \
                                                   -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha2id) \
                                                   -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha3id) \
                                                   -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.arrowid) \
                                                   -state hidden
               }

               # Hide the bar and the line.
               $editor itemconfigure $columnObjectMap($col,$row.barid) \
                                                -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.lineid) \
                                                -state hidden

               # Make sure the box is the correct color.
               set boxColor $cellColorActive
               #if {[seq get sel $sequenceID $elementIndex]} {
               #    set boxColor $selectionColor
               #}
               if {$columnObjectMap($col,$row.boxcolor) != $boxColor} {
                  set columnObjectMap($col,$row.boxcolor) $boxColor
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                   -fill $boxColor
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                   -outline $boxColor
               }

               # Setup item colors; if a gap, set bar and line to box color.
               if {$element == "-"} {
                  set iconName ""
                  set iconColorName ""
                  set iconColor ""


               # Otherwise, show just the bar.
               } else {
                  set iconName "barid"
                  set iconColorName "barcolor"
                  set iconColor [$colorMapLookup [seq get color \
                                                   $sequenceID $elementIndex]]
               }

               if {$iconName != ""} {

                  # Adjust the color for any selection.
                  #if {[seq get sel $sequenceID $elementIndex]} {
                  #    set intensityDecrease [expr -((1.0-[getIntensity $iconColor])/2.0)]
                  #    if {$intensityDecrease > -0.2} {
                  #        set intensityDecrease -0.2
                  #    }
                  #    set iconColor [getBrightenedColor $selectionColor $intensityDecrease $intensityDecrease $intensityDecrease]
                  #}

                  # Make the canvas changes.
                  $editor itemconfigure $columnObjectMap($col,$row.$iconName) \
                                                   -state normal
                  if {$columnObjectMap($col,$row.$iconColorName)!= $iconColor} {
                     set columnObjectMap($col,$row.$iconColorName) $iconColor
                     $editor itemconfigure $columnObjectMap($col,$row.$iconName) -fill $iconColor
                     if {$iconName != "lineid"} {
                        $editor itemconfigure $columnObjectMap($col,$row.$iconName) -outline $iconColor
                     }
                  }
               }

            } elseif {$representationMap($sequenceID) == "secondary"} {

               # If not in correct active state, adjust the order of the items.
               if {$columnObjectMap($col,$row.active) != 3} {

                  # Rearrange the necessary items.
                  set columnObjectMap($col,$row.active) 3
                  if {$cellFont != ""} {
                     $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                              -state hidden
                  }
               }

               # Move all of the possible icons back.
               $editor itemconfigure $columnObjectMap($col,$row.barid) \
                                                        -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.lineid) \
                                                        -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha0id) \
                                                        -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha1id) \
                                                        -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha2id) \
                                                        -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha3id) \
                                                        -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.arrowid) \
                                                        -state hidden

               # Make sure the box is the correct color.
               set boxColor $cellColorActive
               #if {[seq get sel $sequenceID $elementIndex]} {
               #    set boxColor $selectionColor
               #}
               if {$columnObjectMap($col,$row.boxcolor) != $boxColor} {
                  set columnObjectMap($col,$row.boxcolor) $boxColor
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                           -fill $boxColor
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                           -outline $boxColor
               }

               # Setup the item colors; if this is a gap, just show the box.
               if {$element == "-"} {

                  set iconName ""
                  set iconColorName ""
                  set iconColor ""

                    # If this is a helix, show the proper icon.
               } elseif {$element== "H" || $element == "G" || $element == "I"} {

                  # Set the stuff.
                  set iconName "alpha$helixCount"
                  append iconName "id"
                  set iconColorName "alpha$helixCount"
                  append iconColorName "color"
                  set iconColor [$colorMapLookup [seq get color \
                                                 $sequenceID $elementIndex]]

                  if {$iconColor == "#FFFFFF"} {
                     set iconColor $cellColorForeground
                  }

                  # Increment the helix section.
                  incr helixCount
                  if {$helixCount > 3} {
                     set helixCount 0
                  }

                  # If this is a strand, show the proper icon.
               } elseif {$element == "E"} {

                  set iconName "barid"
                  set iconColorName "barcolor"
                  set iconColor [$colorMapLookup [seq get color \
                                                 $sequenceID $elementIndex]]
                  if {$iconColor == "#FFFFFF"} {
                     set iconColor $cellColorForeground
                  }
                  set helixCount 0

                  # See if this should really be an arrow.
                  if {[expr $col+1] < [llength $sequence]} {
                     for {set i [expr $col+1]} {$i < [llength $sequence]} \
                                                                 {incr i} {
                        set nextElement [lindex $sequence $i]
                        if {$nextElement == "-"} {
                           continue
                        } elseif {$nextElement == "E"} {
                           break
                        } else {
                           set iconName "arrowid"
                           set iconColorName "arrowcolor"
                           break
                        }
                     }
                  } elseif {$endsWithBetaArrow} {
                     set iconName "arrowid"
                     set iconColorName "arrowcolor"
                  } 

               }  elseif {$element == "(" || $element == "<" || $element == ")" || $element == ">" || $element == "."} {
                  # if this is an RNA stem
                  set iconName "textid"
                  set elementColor [$colorMapLookup [seq get color \
                                                 $sequenceID $elementIndex]]
                  if {$cellFont == "" && $elementColor == "#FFFFFF"} {
                     set elementColor $cellTextReplacementColor
                  }
                  set helixCount 0

                  # Otherwise, just show the line.
               } else {

                  set iconName "lineid"
                  set iconColorName "linecolor"
                  set iconColor $cellColorForeground
                  set helixCount 0
               }

               # Raise and color the correct icon.
               if {$iconName != ""} {

                  # Adjust the color for any selection.
                  #if {[seq get sel $sequenceID $elementIndex]} {
                  #    set intensityDecrease [expr -((1.0-[getIntensity $iconColor])/2.0)]
                  #    if {$intensityDecrease > -0.2} {
                  #        set intensityDecrease -0.2
                  #    }
                  #    set iconColor [getBrightenedColor $selectionColor $intensityDecrease $intensityDecrease $intensityDecrease]
                  #}

                  # Make the canvas changes.
                  $editor itemconfigure $columnObjectMap($col,$row.$iconName) \
                                                               -state normal

                  if {$iconName == "textid"} {
                     if {$cellFont != ""} {
                        if {$columnObjectMap($col,$row.textstring) != $element} {
                           set columnObjectMap($col,$row.textstring) $element
                           $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                      -text $element
                        }
                     }
                     if {$columnObjectMap($col,$row.boxcolor) != $elementColor} {
                        set columnObjectMap($col,$row.boxcolor) $elementColor
                        $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                             -fill $elementColor
                        $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                             -outline $elementColor

                        if {[info exists columnObjectMap($col,$row.textid)]} {
                           foreach {r g b} [winfo rgb . $elementColor] { 
                              break 
                           }
      
                           set tColor [format "#%02X%02X%02X" \
                                 [expr ($r >> 8) ^ 128] \
                                 [expr ($g >> 8) ^ 128] \
                                 [expr ($b >> 8) ^ 128]] 
                           $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                             -fill $tColor
                        }
                     }
                  } elseif {$columnObjectMap($col,$row.$iconColorName) != $iconColor} {
                     set columnObjectMap($col,$row.$iconColorName) $iconColor
                     $editor itemconfigure $columnObjectMap($col,$row.$iconName) -fill $iconColor
                     if {$iconName != "lineid"} {
                        $editor itemconfigure $columnObjectMap($col,$row.$iconName) -outline $iconColor
                     }
                  }
               }

            } else {

               # If not in correct active state, adjust the order of the items.
               if {$columnObjectMap($col,$row.active) != 0} {

                  # Rearrange the necessary items.
                  set columnObjectMap($col,$row.active) 0
                  if {$cellFont != ""} {
                     $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                              -state hidden
                  }
                  $editor itemconfigure $columnObjectMap($col,$row.barid) \
                                                           -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.lineid) \
                                                           -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha0id) \
                                                           -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha1id) \
                                                           -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha2id) \
                                                           -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.alpha3id) \
                                                           -state hidden
                  $editor itemconfigure $columnObjectMap($col,$row.arrowid) \
                                                           -state hidden

                  # Draw the cell as inactive.
                  set columnObjectMap($col,$row.boxcolor) $cellColorInactive
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                               -fill $cellColorInactive
                  $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                               -outline $cellColorInactive

               }
            }

            incr col
            incr elementIndex
         }

         # Go through the rest of the columns and make them inactive.
         for {} {$col < $numberCols} {incr col} {

            # If not in correct active state, adjust the order of the items.
            if {$columnObjectMap($col,$row.active) != 0} {

               # Rearrange the necessary items.
               set columnObjectMap($col,$row.active) 0
               if {$cellFont != ""} {
                  $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                        -state hidden
               }
               $editor itemconfigure $columnObjectMap($col,$row.barid) \
                                                     -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.lineid) \
                                                     -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha0id) \
                                                     -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha1id) \
                                                     -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha2id) \
                                                     -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha3id) \
                                                     -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.arrowid) \
                                                     -state hidden

               # Draw the cell as inactive.
               set columnObjectMap($col,$row.boxcolor) $cellColorInactive
               $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                     -fill $cellColorInactive
               $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                                     -outline $cellColorInactive

            } else {
               break
            }
         }

      } else { ;# else the $sequenceIndex >= $numberSequences

         # Draw the header cell as inactive.
         if {$columnObjectMap(h,$row.active) != 0} {

            set columnObjectMap(h,$row.active) 0
            if {$cellFont != ""} {
               $editor lower $columnObjectMap(h,$row.textid) \
                                             $columnObjectMap(h,$row.boxid)
               $editor lower $columnObjectMap(h,$row.numberid) \
                                              $columnObjectMap(h,$row.boxid)
            }
            $editor lower $columnObjectMap(h,$row.checkboxid) \
                                           $columnObjectMap(h,$row.boxid)
            $editor lower $columnObjectMap(h,$row.checkid) \
                                           $columnObjectMap(h,$row.boxid)
            $editor lower $columnObjectMap(h,$row.infobuttonid) \
                                           $columnObjectMap(h,$row.boxid)
            if {$columnObjectMap(h,$row.infobuttontextid) != -1} {
               $editor lower $columnObjectMap(h,$row.infobuttontextid) \
                                              $columnObjectMap(h,$row.boxid)
            }
            $editor lower $columnObjectMap(h,$row.repbuttonid) \
                                           $columnObjectMap(h,$row.boxid)
            if {$columnObjectMap(h,$row.repbuttontextid) != -1} {
               $editor lower $columnObjectMap(h,$row.repbuttontextid) \
                                              $columnObjectMap(h,$row.boxid)
            }
            $editor lower $columnObjectMap(h,$row.vmdbuttonid) \
                                           $columnObjectMap(h,$row.boxid)
            if {$columnObjectMap(h,$row.vmdbuttontextid) != -1} {
               $editor lower $columnObjectMap(h,$row.vmdbuttontextid) \
                                              $columnObjectMap(h,$row.boxid)
            }
            if {$columnObjectMap(h,$row.boxcolor) != $cellColorInactive} {
               set columnObjectMap(h,$row.boxcolor) $cellColorInactive
               $editor itemconfigure $columnObjectMap(h,$row.boxid) \
                                              -fill $cellColorInactive
               $editor itemconfigure $columnObjectMap(h,$row.boxid) \
                                              -outline $cellColorInactive
               $editor itemconfigure $columnObjectMap(h,$row.separatorid) \
                                              -fill $cellColorInactive
               $editor itemconfigure $columnObjectMap(h,$row.tickid) \
                                              -fill $cellColorInactive
            }
         }

         # Go through each column.
         for {set col 0} {$col < $numberCols} {incr col} {

            # If not in correct active state, adjust the order of the items.
            if {$columnObjectMap($col,$row.active) != 0} {

               # Rearrange the necessary items.
               set columnObjectMap($col,$row.active) 0
               if {$cellFont != ""} {
                  $editor itemconfigure $columnObjectMap($col,$row.textid) \
                                                 -state hidden
               }
               $editor itemconfigure $columnObjectMap($col,$row.barid) \
                                              -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.lineid) \
                                              -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha0id) \
                                              -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha1id) \
                                              -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha2id) \
                                              -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.alpha3id) \
                                              -state hidden
               $editor itemconfigure $columnObjectMap($col,$row.arrowid) \
                                              -state hidden

               # Draw the cell as inactive.
               set columnObjectMap($col,$row.boxcolor) $cellColorInactive
               $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                              -fill $cellColorInactive
               $editor itemconfigure $columnObjectMap($col,$row.boxid) \
                                              -outline $cellColorInactive

            } else {
               break
            }
         } ;# end for loop over columns
      }
   } ;# end of loop over each row

   redrawZoomMainCanvas

#   puts "\n\n -------------- seqedit_widget.tcl.redraw END ------------------\n\n"
} ;# end of redraw()

# ----------------------------------------------------------------------------
# This method is called be the window manager when a component of the widget has been reconfigured.
# args:     a_name - The name of the component that was reconfigured.
#           a_width - The new width of the component.
#           a_height - The new height of the component.
proc ::SeqEditWidget::component_configured {a_name a_width a_height} {

    variable editor
    variable width
    variable height
    variable cellWidth
    variable cellHeight
    variable headerCellWidth
    variable headerCellHeight
    variable numberCols
    variable numberRows
    variable firstElement
    variable firstGroup
    variable firstSequence


    # Check to see if the window is being resized.
    if {$a_name == $editor && ($a_width != $width || $a_height != $height)} {

        # Save the new width and height.
        set width $a_width

        set height $a_height

        # See if the number of rows or columns has changed.
        if {$numberCols != [expr (($width-$headerCellWidth)/$cellWidth)+1] || $numberRows != [expr (($height-$headerCellHeight)/$cellHeight)+1]} {

            # Save the new number of rows and columns.
            set numberCols [expr (($width-$headerCellWidth)/$cellWidth)+1]
            set numberRows [expr (($height-$headerCellHeight)/$cellHeight)+1]

            # Make sure we are not out of scroll range.
            validateScrollRange

            # Create the new editor and redraw it.
            deleteCells
            setScrollbars
            createCells
            redraw
        }
    }
}

# ----------------------------------------------------------------------------
# This method is called be the horizontal scroll bar when its state has changed.
proc ::SeqEditWidget::scroll_horzizontal {{action 0} {amount 0} {type 0}} {

    variable firstElement
    variable numberCols
    variable numberElements

    # Perform the scroll.
    if {$action == "scroll" && ($type == "units" || $type == "unit")} {
        set firstElement [expr $firstElement+$amount]
    } elseif {$action == "scroll" && ($type == "pages" || $type == "page")} {
        set firstElement [expr $firstElement+($numberCols-2)*$amount]
    } elseif {$action == "moveto"} {
        set firstElement [expr int(($numberElements+1)*$amount)]
    }
    # Make sure we didn't scroll out of range.
    validateScrollRange 1 0

    # Set the scroll bars.
    setScrollbars
    #puts "Redraw took [time redraw]"
    redraw
}

# ----------------------------------------------------------------------------
# This method is called to ensure that a cell is visible.
proc ::SeqEditWidget::ensureCellIsVisible {sequenceID position} {

    variable firstElement
    variable numberCols

    # Track if we need to scroll.
    set needToScroll 0

    # See if we are out of the horizontal range.
    if {$position < $firstElement} {
        set firstElement $position
        set needToScroll 1
    } elseif { $position >= [expr $firstElement+$numberCols-1]} {
        set firstElement [expr $position-$numberCols+2]
        set needToScroll 1
    }

    # Make sure we didn't scroll out of range and then redraw.
    if {$needToScroll == 1} {
        validateScrollRange
        setScrollbars
        redraw

        return 1
    }

    return 0
}



# ----------------------------------------------------------------------------
# This method is called by the vertical scroll bar when its state has changed.
proc ::SeqEditWidget::scroll_vertical {{action 0} {amount 0} {type 0}} {

    variable groupNames
    variable groupMap
    variable firstGroup
    variable firstSequence
    variable numberRows
#    puts "::SeqEditWidget::scroll_vertical.start.firstSequence:$firstSequence"

    # See what kind of scroll this was.
    if {$action == "scroll"} {

        # Figure out how far we moved.
        set lines 0
        if {$type == "units" || $type == "unit"} {
            set lines $amount
        } elseif {$type == "pages" || $type == "page"} {
            set lines [expr ($numberRows-2)*$amount]
        }
#        puts "scrolling $lines lines"
        # If we moved some amount, figure out where we are now and redraw.
        if {$lines != 0} {
            while {$lines != 0} {
                set numberSequences $groupMap([lindex $groupNames \
                                                 $firstGroup],numberSequences)
                if {$lines > 0} {
                    if {$lines < [expr $numberSequences-($firstSequence)]} {
                        incr firstSequence $lines
                        break
                    } elseif {$firstGroup == [expr [llength $groupNames]-1]} {
                        set firstSequence $groupMap([lindex $groupNames $firstGroup],numberSequences)
                        break
                    } else {
                        incr lines [expr -($numberSequences-($firstSequence))]
                        incr firstGroup
                        set firstSequence -1
                    }
                } elseif {$lines < 0} {
                    if {$lines > [expr -($firstSequence+2)]} {
                        incr firstSequence $lines
                        break
                    } elseif {$firstGroup == 0} {
                        set firstSequence -1
                        break
                    } else {
                        incr lines [expr $firstSequence+2]
                        incr firstGroup -1
                        set firstSequence [expr $groupMap([lindex $groupNames $firstGroup],numberSequences)-1]
                    }
                }
            }
        }

        # Make sure we didn't scroll out of range.
        validateScrollRange 0 1

        # Set the scroll bars.
        setScrollbars
        redraw
#        puts "::SeqEditWidget::scroll_vertical.after redraw.firstSequence:$firstSequence"

    } elseif {$action == "moveto"} {

        set lines {}
        set numberGroups [expr [llength $groupNames]]
        for {set i 0} {$i < $numberGroups} {incr i} {

            # Add a line for the group header.
            lappend lines [list $i -1]

            # Get the number of sequences in this group.
            set numberSequences $groupMap([lindex $groupNames $i],numberSequences)
            for {set j 0} {$j < $numberSequences} {incr j} {
                lappend lines [list $i $j]
            }
        }

        set lineIndex [expr int(([llength $lines]-1)*$amount)]
        if {$lineIndex < 0} {
            set lineIndex 0
        } elseif {$lineIndex >= [llength $lines]} {
            set lineIndex [expr [llength $lines]-1]
        }
        set line [lindex $lines $lineIndex]
        set newFirstGroup [lindex $line 0]
        set newFirstSequence [lindex $line 1]

        # If we change position, perform the scroll.
        if {$newFirstGroup != $firstGroup || $newFirstSequence != $firstSequence} {

            set firstGroup $newFirstGroup
            set firstSequence $newFirstSequence
            validateScrollRange 0 1
            setScrollbars
            redraw
        }
    }
} ; # end of scroll_vertical

# -------------------------------------------------------------------------

# Handle clicks on the row header.
proc ::SeqEditWidget::click_rowcheckbox {x y} {

    variable groupNames
    variable markMap

    # Get the row that was clicked on.
    set row [determineRowFromLocation $x $y]
    if {$row != -1} {

        # Get the sequence that is in the row.
        set sequence [determineSequenceFromRow $row]

        # Make sure there is a sequence in the row.
        if {$sequence != {}} {

            # Get the new mark state.
            set sequenceID [getSequenceInGroup [lindex $groupNames [lindex $sequence 0]] [lindex $sequence 1]]
            set state $markMap($sequenceID)
            if {$state == 0} {
                set state 1
            } else {
                set state 0
            }

            # Get the lsit of currently selecetd sequences.
            set selectedSequenceIDs [getSelectedSequences]

            # If this sequence in in the selected list, set all of them as marked.
            if {[lsearch $selectedSequenceIDs $sequenceID] != -1} {
                setMarksOnSequences $selectedSequenceIDs $state

            # Otherwise just set this sequence as marked.
            } else {
                setMarksOnSequences [list $sequenceID] $state
            }
        }
    }
}

# -----------------------------------------------------------------------
# Handle clicks on the row header.
proc ::SeqEditWidget::click_rownotes {x y} {

    variable widget
    variable groupNames

    # Get the row that was clicked on.
    set row [determineRowFromLocation $x $y]
    if {$row != -1} {

        # Get the sequence that is in the row.
        set sequence [determineSequenceFromRow $row]

        # Make sure there is a sequence in the row.
        if {$sequence != {} && [lindex $sequence 1] != -1} {
            set sequenceID [getSequenceInGroup [lindex $groupNames [lindex $sequence 0]] [lindex $sequence 1]]
            if {[::SeqData::Notes::showEditNotesDialog $widget $sequenceID]} {
                redraw $sequenceID
            }
        }
    }
}

# -----------------------------------------------------------------------
# Handle clicks on the title bar.
proc ::SeqEditWidget::click_titleHeader {x y} {

    variable widget
    variable currentTitlePopupMenu

    # Figure out the popup location.
    set px [expr $x+[winfo rootx $widget]]
    set py [expr $y+[winfo rooty $widget]]

    tk_popup $currentTitlePopupMenu $px $py
}

# -----------------------------------------------------------------------
# Handle clicks on the row header.
proc ::SeqEditWidget::click_rowbutton {buttonType x y} {

    variable widget
    variable groupNames
    variable repPopupMenu
    variable vmdPopupMenu
    variable popupMenuParameters

    # Get the row that was clicked on.
    set row [determineRowFromLocation $x $y]
    if {$row != -1} {

        # Get the sequence that is in the row.
        set sequence [determineSequenceFromRow $row]

        # Make sure there is a sequence in the row.
        if {$sequence != {} && [lindex $sequence 1] != -1} {

            # Get the sequence id.
            set sequenceID [getSequenceInGroup [lindex $groupNames [lindex $sequence 0]] [lindex $sequence 1]]

            # Get the currently selected sequences.
            set selectedSequenceIDs [getSelectedSequences]

            # If this sequence is in the selected list, set that all of them should be affected.
            if {[lsearch $selectedSequenceIDs $sequenceID] != -1} {
                set popupMenuParameters $selectedSequenceIDs

            # Otherwise set that just this sequence should be affected.
            } else {
                set popupMenuParameters $sequenceID
            }

            # Figure out the popup location.
            set px [expr $x+[winfo rootx $widget]]
            set py [expr $y+[winfo rooty $widget]]

            # Bring up the group popup menu.
            if {$buttonType == "rep"} {
                tk_popup $repPopupMenu $px $py
            } elseif {$buttonType == "vmd" && $vmdPopupMenu != ""} {
                tk_popup $vmdPopupMenu $px $py
            }
        }
    }
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::rightclick_rowheader {x y} {

    variable widget
    variable clickedGroup
    variable groupPopupMenu

    # Get the row that was clicked on.
    set row [determineRowFromLocation $x $y]
    if {$row != -1} {

        # Get the sequence that is in the row.
        set sequence [determineSequenceFromRow $row]

        # Make sure there is a sequence in the row.
        if {$sequence != {}} {

            # If this is a group row, save the group index and popup the group menu.
            if {[lindex $sequence 1] == -1} {

                set clickedGroup [lindex $sequence 0]

                # Figure out the popup location.
                set px [expr $x+[winfo rootx $widget]]
                set py [expr $y+[winfo rooty $widget]]

                # Bring up the group popup menu.
                tk_popup $groupPopupMenu $px $py
            }
        }
    }

}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_insertgroup {} {

    variable widget
    variable groupNames
    variable clickedGroup

    array set options [::SeqEdit::GetGroupName::showGetGroupNameDialog $widget "Insert Group" "Enter group name"]
    if {[array size options] > 0 && $options(name) != ""} {
        insertGroup $options(name) $clickedGroup
    }
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_renamegroup {} {

    variable widget
    variable groupNames
    variable clickedGroup

    array set options [::SeqEdit::GetGroupName::showGetGroupNameDialog $widget "Rename Group" "Enter new group name" [lindex $groupNames $clickedGroup]]
    if {[array size options] > 0 && $options(name) != ""} {
        renameGroup [lindex $groupNames $clickedGroup] $options(name)
    }
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_deletegroup {} {

    variable groupNames
    variable clickedGroup

    deleteGroup [lindex $groupNames $clickedGroup]
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_addtogroup {} {

    variable groupNames
    variable clickedGroup

    set filename [tk_getOpenFile -filetypes {{{FASTA Files} {.fasta}} {{All Files} * }} -title "Add Sequences to [lindex $groupNames $clickedGroup] Group"]
    if {$filename != ""} {
        set sequences [::SeqData::Fasta::loadSequences $filename]
        addSequences $sequences [lindex $groupNames $clickedGroup]
    }
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_markgroup {{value 1}} {

    variable groupNames
    variable clickedGroup

    setMarksOnSequences [getSequencesInGroup [lindex $groupNames $clickedGroup]] $value
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_markall {{value 1}} {

    setMarksOnSequences [getSequences] $value
}


# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_duplicate {} {

    variable popupMenuParameters

    duplicateSequences $popupMenuParameters
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::menu_setrepresentation {type} {

    variable popupMenuParameters

    setRepresentations $popupMenuParameters $type
}

#--------------------------------------------------------------------------
# Gets the row that contains the specified x and y position.
# return:   The row the is at the specified position or -1 if it there is no valid row there.
proc ::SeqEditWidget::determineRowFromLocation {x y} {

   variable columnObjectMap
   variable numberRows

   for {set i 0} {$i < $numberRows} {incr i} {
      if {$y >= $columnObjectMap(h,$i.y1) && $y <= $columnObjectMap(h,$i.y2)} {
         return $i
      }
   }

   return -1
}

#--------------------------------------------------------------------------
# Get list of sequences ids starting with beginRow and ending with endRow
proc ::SeqEditWidget::getSequenceIDsInRows {beginRow endRow} {

   variable firstGroup
   variable firstSequence
   variable groupNames

   # let's handle this special case first
   if {$beginRow == $endRow} {
      return [list $beginRow]
   }

   # create a list of all the sequences in all of the groups
   set seqList [getSequenceList]

   # find the first, the last, decide which has the first/last index
   set beginIndex [lsearch -exact $seqList $beginRow]
   set endIndex [lsearch -exact $seqList $endRow]

   # do we need to flip 'em?
   if { $beginIndex > $endIndex } {
      set t $endIndex
      set endIndex $beginIndex
      set beginIndex $t
   }

   # return a list of the sequences in between the two
   #puts "I think it is [lrange $seqList $beginIndex $endIndex]"
   return [lrange $seqList $beginIndex $endIndex]

   set seqList {}
   puts "seqedit_widget.getSequenceIDsInRows fg: $firstGroup, gn: $groupNames, fs: $firstSequence, beg: $beginRow, end: $endRow, drflbeg: [determineRowFromLocation $beginRow 0] drflend: [determineRowFromLocation $endRow 0]"

   incr beginRow [expr -$firstSequence]
   incr endRow [expr -$firstSequence]

   puts "seqedit_widget.getSequenceIDsInRows.after delta/flip beg: $beginRow, end: $endRow"

   for {set c $beginRow} {$c <= $endRow} {incr c} {
      set sfr [determineSequenceFromRow $c]
      puts "seqedit_widget.getSequenceIDsInRows sfr: $sfr"
      lappend seqList [getSequenceInGroup [lindex $groupNames [lindex $sfr 0]] \
                                          [lindex $sfr 1]]
   }
   return $seqList
}

#--------------------------------------------------------------------------
# Gets sequences in a list in the order that they are currently in on the
# screen
# return:   A list containing sequence IDs in order
proc ::SeqEditWidget::getSequenceList {} {

    variable groupNames
    variable groupMap
    variable firstGroup
    # Go through each group.
    set theList {}
    set numberGroups [llength $groupNames]
    for {set groupIndex $firstGroup} {$groupIndex < $numberGroups } {incr groupIndex} {
       set theList [concat $theList $groupMap([lindex $groupNames $groupIndex],sequenceIDs)]
    }

    return $theList
}

#--------------------------------------------------------------------------
# Gets the sequence that is currently being displayed in the specified row.
# args:     row - The row to check.
# return:   A list containing two elements: 1. the index of the group; 2. the index of the sequence
#           in the group. If no sequence were in the row an empty list is returned.
proc ::SeqEditWidget::determineSequenceFromRow {row} {

    variable groupNames
    variable groupMap
    variable firstGroup
    variable firstSequence
    variable numberRows

    # Go through each group.
    set offset $firstSequence
    set numberGroups [llength $groupNames]
    for {set groupIndex $firstGroup} {$groupIndex < $numberGroups && $row >= 0} {incr groupIndex} {

        # Get the number of sequences in this group.
        set numberSequences $groupMap([lindex $groupNames $groupIndex],numberSequences)

        # See if the row is in this group.
        if {[expr $row+$offset] < $numberSequences} {

            # It is, so figure out and return the sequence info.
            return [list $groupIndex [expr $row+$offset]]

        } else {

            # It is not, so go to the next group.
            incr row [expr -($numberSequences-$offset)]
            set offset -1
        }
    }

    return {}
}


# -----------------------------------------------------------------------
# Gets the column that contains the specified x and y location.
# return:   The column the is at the specified location or -1 if it there is no valid column there.
proc ::SeqEditWidget::determineColumnFromLocation {x y} {

    variable columnObjectMap
    variable numberCols

    for {set i 0} {$i < $numberCols} {incr i} {
        if {$x >= $columnObjectMap($i,h.x1) && $x <= $columnObjectMap($i,h.x2)} {
            return $i
        }
    }

    return -1
}

# -----------------------------------------------------------------------
# Gets the element that is currently being displayed in the specified column.
# args:     column - The column to check.
# return:   The index of the element in the specified column. or -1 no element is in the column.
proc ::SeqEditWidget::determinePositionFromColumn {column} {

    variable firstElement
    variable numberElements

    set elementIndex [expr $firstElement+$column]
    if {$elementIndex < $numberElements} {return $elementIndex}
    return -1
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::getColorComponents {color} {

#    puts "color $color"
    # Get the color components
    set r "0x[string range $color 1 2]"
    if {$r == "0x 0"} {
        set r 0.0
    } else {
        set r [expr double($r)]
    }
    set g "0x[string range $color 3 4]"
    if {$g == "0x 0"} {
        set g 0.0
    } else {
        set g [expr double($g)]
    }
    set b "0x[string range $color 5 6]"
    if {$b == "0x 0"} {
        set b 0.0
    } else {
        set b [expr double($b)]
    }

    return [list $r $g $b]
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::getBrightenedColor {color rPercentage gPercentage bPercentage} {

    set components [getColorComponents $color]
    set r [lindex $components 0]
    set g [lindex $components 1]
    set b [lindex $components 2]

    set r [expr int($r+($r*$rPercentage))]
    set g [expr int($g+($g*$gPercentage))]
    set b [expr int($b+($b*$bPercentage))]

    set color "#[format %2X $r][format %2X $g][format %2X $b]"
    return $color
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::getIntensity {color} {

    set components [getColorComponents $color]
    set max 0.0
    if {[lindex $components 0] > $max} {
        set max [lindex $components 0]
    }

    if {[lindex $components 1] > $max} {
        set max [lindex $components 1]
    }

    if {[lindex $components 2] > $max} {
        set max [lindex $components 2]
    }

    return [expr $max/255.0]
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::saveAsPS { } {
  variable editor

  set outFile [tk_getSaveFile -initialfile "plot.ps" \
                 -title "Enter filename for PS output"]

  if { $outFile == "" } {
    return
  }

  $editor postscript -file $outFile

  return
}


# -----------------------------------------------------------------------
####################################
#
# turnOffZoom
#
# Disables the zoom window
#
proc ::SeqEditWidget::turnOffZoom { } {
   variable drawZoom
   set drawZoom 0
   #pack forget .multiseq.photoWindow
   catch {
      wm state .win withdrawn
   }
}

####################################
#
# turnOnZoom
#
# Enables the zoom window
#
proc ::SeqEditWidget::turnOnZoom { } {

   variable drawZoom
   set drawZoom 1
   #drawImage
   drawCanvas
   wm state .win normal
}

#####################################
#
# toggleZoom
#
# Switches the current state of the
# Zoom window
#
proc ::SeqEditWidget::toggleZoom { } {

   variable drawZoom

   if { !$drawZoom } {

      turnOffZoom
   } else {

      turnOnZoom
   }
}

# -----------------------------------------------------------------------
proc ::SeqEditWidget::redrawZoomMainCanvas { } {
   variable drawZoom
   if { $drawZoom } {
      fillMainCanvas
   }
}


# -----------------------------------------------------------------------
proc ::SeqEditWidget::fillMainCanvas { } {

#   variable maxSequencesInGroup
   variable colorMap
   variable mainCanvas
   variable numberElements
   variable scale

   set groupList [getGroups]

   # Get the function to call to lookup a color from an index.
   set colorMapLookup "$colorMap\::getColor"
#   puts "fillNainCanvas: start.  Lookup: $colorMapLookup"

   set sequencesInGroup ""
   set numRows 0
   foreach groupName $groupList {
      incr numRows

      # header
      incr col
      set sequencesInGroup [getSequencesInGroup $groupName]
#      if { [llength $sequencesInGroup] > $maxSequencesInGroup} {
#         set maxSequencesInGroup [llength $sequencesInGroup]
#      }
      foreach seq $sequencesInGroup {
         incr numRows
         set sequence [SeqData::getSeq $seq]
         set lengthSeq [llength $sequence]
         set colScale [expr $col * $scale]
         for {set i 0} { $i < $numberElements} { incr i } {
            set fillColor "\#e0e0e0"
            if { $i < $lengthSeq && [lindex $sequence $i] != "-"} {
               set fillColor [$colorMapLookup [seq get color $seq $i]] 
            }

            set scaled [expr { $i * $scale } ]
            $mainCanvas create rectangle \
                         $scaled $colScale \
                         [expr $scaled + $scale] \
                         [expr $colScale + $scale] \
                        -fill $fillColor \
                        -tags dataScalable -outline ""
         }
         incr col
#         puts -nonewline "$col,"
      }
   }
   drawBox
#   puts "fillNainCanvas: end"

}


#####################################
#
# drawCanvas
#
# Draws the image for the zoomed out window
#
proc ::SeqEditWidget::drawCanvas { } {

   variable mainCanvas
#   variable maxSequencesInGroup

   catch {
      destroy .win
   }
   if { [winfo exists .win] } {
      set win .win
   } else {
      set win [toplevel .win -width 400 -height 300]
   }
   wm title $win "Zoom Window"

   set row 0
   set col 0

   variable scale
   set scale 1

#   set canvasWidth 500
#   set canvasHeight 500
#   set xcanwindowmax 500
#   set ycanwindowmax 500

   set topFrame [frame .win.topFrame -borderwidth 2 -background black]
   set bottomFrame [frame .win.bottomFrame]

   set canvasFrame [frame $topFrame.canvasFrame]
   set yscrollFrame [frame $topFrame.yscrollFrame]
   set xscrollFrame [frame $bottomFrame.xscrollFrame]
   set scaleFrame [frame .win.scaleFrame]

   set mainCanvas [canvas $canvasFrame.mainCanvas -width 1 -height 1 -bg \#c0c0c0 -borderwidth 2]
#    set mainCanvas [canvas $canvasFrame.mainCanvas -width $canvasWidth -height $canvasHeight -bg \#c0c0c0 -borderwidth 2]

   fillMainCanvas

   #scaling bar
   set currScale $scale
   set scaleBar [scale $scaleFrame.scaleBar -orient horizontal -from -10.01 -to 10.01 -length 200 -sliderlength 30  -resolution 1.0 -tickinterval 2 -repeatinterval 1 -showvalue true  -command "::SeqEditWidget::scaleBoxes \$::SeqEditWidget::scale"]

   set yscroll [scrollbar $yscrollFrame.yscroll -command "$mainCanvas yview"]
   set xscroll [scrollbar $xscrollFrame.xscroll -orient horizontal -command "$mainCanvas xview"]

   #Pack
#   set canvasWidth [expr { $maxSequencesInGroup * $scale }]
#   set canvasHeight [expr { $numRows * $scale}]

   $mainCanvas configure -yscrollcommand "$yscroll set"
   $mainCanvas configure -xscrollcommand "$xscroll set"
   $mainCanvas configure -scrollregion [$mainCanvas bbox all]
   $mainCanvas configure -borderwidth 1

   pack $topFrame -side top -anchor nw -fill both -expand true
   pack $bottomFrame -side top -fill x -expand false
   pack $canvasFrame -side left -anchor nw -fill both -expand true
   pack $yscrollFrame -side left -fill y -expand false
   pack $xscrollFrame -side top -fill x -expand false
   pack $scaleFrame -side top

   pack $mainCanvas -side left -fill both -expand true -anchor nw
   pack $scaleBar -side top
   pack $yscroll -side left -fill y -expand true
   pack $xscroll -side top -fill x -expand true

   bind $win <Destroy> {::SeqEditWidget::turnOffZoom}
   bind $mainCanvas <ButtonPress-1> "::SeqEditWidget::imageCanvasScroll %x %y"

}; # end drawCanvas

####################################
#
# scaleBoxes
#
# Scales the zoomed out image
#
# origScale: The current scale of the zoomed out image
# newScale: The new scale for the zoomed out image
#
proc ::SeqEditWidget::scaleBoxes { origScale newScale} {

    variable scale
    variable mainCanvas
    variable imageScale

    if { $newScale < 0 } {
       incr newScale -1
    } else {
       incr newScale
    }

    set scaler [expr (1.0 * $newScale) / $origScale]

    if { $origScale < 0 } {
       if { $newScale < 0 } {
          # -1 to -2 (1/2 scale)
          set scaler [expr 1.0 * $origScale / $newScale]
       } else {
          # -2 to 2 (4x scale)
          set scaler [expr -1.0 * $newScale * $origScale]
       }
    } else {
       if { $newScale < 0 } {
          # 2 to -2 (1/4 scale)
          set scaler [expr -1.0 / ($newScale * $origScale)]
       } else {
          # 2 to 3 (2x to 3x scale)
          set scaler [expr 1.0 * $newScale / $origScale]
       }
    }

    if { $scaler < 0 } {
       set scaler [expr -1.0 * $scaler]
    }

    $mainCanvas scale dataScalable 0 0 $scaler $scaler
    set scale $newScale
    $mainCanvas configure -scrollregion [$mainCanvas bbox all]
    drawBox
}

####################################
#
# drawBox
#
# draws the box on the zoomed out window
# showing the current area shown in the main window
#
proc ::SeqEditWidget::drawBox { } {

   variable firstSequence
   variable firstElement
   variable firstGroup
   variable scale
   variable numberRows
   variable numberCols
   variable groupMap

   variable mainCanvas

   catch {
      $mainCanvas delete windowFrame
   }

   set firstx $firstElement
   set firsty $firstSequence

   set foo ""
   for { set i 0 } { $i < $firstGroup } { incr i } {
      if { [catch {
            incr firsty
            set groupName [lindex [getGroups] $i]
# puts "Group $groupName firsty: $firsty"            
            incr firsty $groupMap($groupName,numberSequences)

         } foo] } {
       puts "Multiseq Zoom) Error in setting firstGroup: $foo"
      }
   }
   set boxScale $scale
   if { $scale < 0 } {
      set boxScale [expr -1.0 / $scale]
   }

   incr firsty
   if { $firsty < 0 } {
      set firsty 0
   }
   if { $firstx < 0 } {
      set firstx 0
   }

   # foobar.  mainCanvas isn't always defined here (on the next line):
# can't read "mainCanvas": no such variable
#     while executing
#         ... next line
#         (procedure "drawBox" line 50)
#             invoked from within
#            "drawBox"
#    puts "groups: [getGroups] firstSeq: $firstSequence, firstElem: $firstElement, firstGroup: $firstGroup, firstx: $firstx, boxScale: $boxScale, firsty: $firsty, numCols: $numberCols, numRows: $numberRows. Getting ready to [expr $firstx * $boxScale] [expr $firsty * $boxScale] [expr ($firstx + $numberCols - 1) * $boxScale] [expr ($firsty + $numberRows - 1) * $boxScale]" 
   $mainCanvas create rectangle [expr $firstx * $boxScale] \
                                [expr $firsty * $boxScale] \
                                [expr ($firstx + $numberCols - 1) * $boxScale] \
                                [expr ($firsty + $numberRows - 1) * $boxScale] \
                                -width 2 \
                                -fill "" \
                                -tags windowFrame

}

# -------------------------------------------------------------------
proc ::SeqEditWidget::canvasScrollY {args} {

    variable mainCanvas

  eval $mainCanvas yview $args
}

proc ::SeqEditWidget::canvasScrollX {args} {

    variable mainCanvas
  eval $mainCanvas xview $args

  return
}

####################################
#
# imageCanvasScroll
#
# Scrolls the zoomed out image
#
# parameters:
#  x: the x-coordinate of the zoomed out image
#  y: the y-coordinate of the zoomed out image
#
proc ::SeqEditWidget::imageCanvasScroll { x y } {

   variable imageScale
   variable groupMap
   variable firstGroup
   variable firstSequence
   variable numberRows
   variable numberCols
   variable numberElements
   variable scale
   variable mainCanvas

   set imageScale $scale
   if { $scale < 0 } {
      set imageScale [expr -1.0 / $scale]
   }

   set colWidth $imageScale

   set canvasHeight [lindex [$mainCanvas bbox all] 3]
   set canvasWidth [lindex [$mainCanvas bbox all] 2]


   set yScroll .win.topFrame.yscrollFrame.yscroll
   set xScroll .win.bottomFrame.xscrollFrame.xscroll

   set xcoords [$xScroll get]
   set ycoords [$yScroll get]
   set firstHeight [lindex $ycoords 0]
   set lastHeight [lindex $ycoords 1]
   set left [lindex $xcoords 0]
   set right [lindex $xcoords 1]

   set scaledx [expr round(($x + ($left * $canvasWidth))/ $imageScale)]
   set scaledy [expr round(($y + ($firstHeight * $canvasHeight))/ $imageScale )]

   set scaledx [expr $scaledx - ($numberCols/2)]
   set scaledy [expr $scaledy - ($numberRows/2)]

   set numseqs 0
   set y $scaledy
   foreach groupName [getGroups] {

         incr numseqs $groupMap($groupName,numberSequences)
         incr numseqs

   }

   if { $scaledy > $numseqs } {
      set scaledy [expr $numseqs - $numberRows]
   }

   set SeqEditWidget::firstElement $scaledx
   set ::SeqEditWidget::firstSequence $scaledy

   set groupId 0
   set foo ""
   set done 0
   set firstGroup 0
   set firstSequence 0


   set relativeRow 0
   if { $scaledy < 0 } {
      set relativeRow -1
   }

   set groupId 0
   set firstGroup 0
   set totalRows 0
   set currGroup [lindex [getGroups] $groupId]
   for { set i 0 } { $i < $scaledy } { incr i } {
      if { [expr $totalRows + $numberRows] >= $numseqs } {
         break
      }
      if { $relativeRow == $groupMap($currGroup,numberSequences) } {
         set relativeRow 0
         incr groupId
         set firstGroup $groupId
         set currGroup [lindex [getGroups] $groupId]
      } else {
         incr relativeRow
      }
      incr totalRows
   }

   if { [expr $relativeRow + $numberRows ] > $numseqs } {
      set relativeRow [expr $numseqs - $numberRows]
   }

   if { $relativeRow < 0 } {
      set relativeRow -1
   }

   set firstSequence $relativeRow
   scroll_vertical
   scroll_horzizontal
   drawBox

}; #end imageCanvasScroll
