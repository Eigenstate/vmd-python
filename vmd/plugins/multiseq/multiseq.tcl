# University of Illinois Open Source License
# Copyright 2004-2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: multiseq.tcl,v 1.38 2018/11/06 23:02:49 johns Exp $
# 
# Developed by: Luthey-Schulten Group
#      University of Illinois at Urbana-Champaign
#      http://faculty.scs.illinois.edu/schulten/
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
# Author(s): Elijah Roberts, John Eargle, Dan Wright, Michael Bach

# Versions
#   2.1 , revised to 3.0 - Kirby tweaks
#

package provide multiseq 3.1
package require mafft 1.0
package require blast 1.1
package require upgma_cluster 1.2
package require libbiokit 1.1
package require phylotree 1.1
package require psipred 1.1
package require seqdata 1.1
package require seqedit 1.1
package require seqedit_widget 1.1
package require stamp 1.2
package require multiseqdialog 1.1
package require colorscalebar 
package require multiplot 1.2
package require multitext 1.0

namespace eval ::MultiSeq:: {
    namespace export MultiSeq

    # Version info.
    variable version "3"
    variable release ".1"


    # For using plotter
    #namespace import ::Plotter::*
    namespace import ::SeqEditWidget::*

    variable printedReference 0

    # current status of sequences
    variable alignfailed 0
    variable txtWinIdx 0
    variable selections "" ;# array of atomselections
    variable molids      ;# array of molids we're working with
    variable w ""        ;# handle to window
    variable shown 0
    variable rmsdWin ".rmsdDialog"           ;# handle to rmsd dialog
    variable stampWin ".stampDialog"         ;# handle to stamp dialog
    variable highlightWin ".highlightDialog" ;# handle to highlight dialog
    variable selectWinQ   ".selectWindowQ"   ;# handle to Q select dialog
    variable selectWinS   ".selectWindowS"   ;# handle to Seq select dialog
    global env
    variable pdbDir                          ;# directory to write pdb files
    variable pdbDirSet 0                     ;# whether pdbDir is set
    variable lastsavedfile alignment.ps
    variable label_canvas "" ;# Variable to hold the label display canvas handle
    variable mol_canvas ""  ;# Variable to hold the sequence display canvas handle
    variable seqs           ;# Variable to hold all the sequences.
    variable num_mols ""    ;# number of loaded molecules
    variable alignment_exists 0 ;# store whether or not a displayable alignment exists
    variable labelwidth 38      ;# default label display area width
    variable curLabelWidth 0    ;# current label width
    variable canwidth 660       ;# default sequence display area width
    variable canheight 400  ;# default display area height
    variable leftmargin 5   ;# margin before letter display on canvas
    variable sqsize 8       ;# Size of the square placed before the mol name
    variable spacing 18     ;# distance allocated to each line
    variable charwidth 7    ;# horizontal space allocated to each character
    variable textstart 14   ;# vertical starting point for text
    variable linestart 23   ;# vertical starting point for lines
    variable xcanmax 0
    variable ycanmax 0      ;# variables to hold width/height of current canvas
    variable monofont
    variable align_sels
    variable label_text_id  ;# holds id of the label canvas text object
    variable mol_text_id    ;# holds id of the mol canvas text object
    variable molTextArray   ;# holds text widgets corresponding to the sequences
    variable seqColorSet    ;# whether or not the sequence text should be colored
    variable molColorGrid   ;# holds gui colored rectangles for display 
                   ;# behind sequence characters; similar to highlightGrid
    variable selColor "none"    ;# color style for VMD molecule representations

    # variables for selection highlighting in VMD OpenGL Display
    variable rep                ;# array of VMD molecule representations
    variable bondRad 0.5        ;# radius of highlights
    variable bondRes 10         ;# resolution of highlights

    variable bgcolor "#00CCCC" ;# the canvas background color

    # variables for residue selection on the mols canvas
    variable x1 0   ;# initial x position of mouse-click
    variable y1 0   ;# initial y position of mouse-click
    variable startShiftPressed   ;# whether the mouse-click was a shift-click
    variable sb "obj"
    variable so ""
    variable eo 0   ;# id of mouse-drawn rectangle
    variable highlightGrid    ;# array to hold the highlighted rectangles

    variable save_viewpoints   

    # Variables for species name lookup
    variable astral2Pdb
    variable astral2PdbLoaded 0
    variable pdb2Sp
    variable pdb2SpLoaded 0
    variable sp2SpeciesName
    variable sp2SpeciesNameLoaded 0
    variable sp2Domain
    variable sp2DomainLoaded 0

    # Variables for sequence identity coloring
    variable seqIdPerRes
    variable seqIdPerResUpToDate

    # Variables for selectDialog
    variable selectWinParam "Q"  ;# Q or S for Q or seq ID per res
    variable selectWinCompQ "GT"  ;# GT or LT for greater then/less then
    variable selectWinCompS "GT"  ;# GT or LT for greater then/less then
    variable selectWinLimitQ "0.0"
    variable selectWinLimitS "0.0"
    #variable checkQPerRes   ;# 0 or 1 (true or false)
    variable aboveOrBelowQ  ;# "above" or "below"
    variable qScoreLimit    ;# 0.0-1.0
    variable qPerResUpToDate
    #variable checkSeqIdPerRes  ;# 0 or 1 (true or false)
    variable aboveOrBelowSeqId ;# "above" or "below"
    variable seqIdScoreLimit   ;# 0.0-1.0

    variable curTree
    variable showRmsd 0

    # Location of sequence editor annotations file
    # Moved to SeqData plugin
    #variable annotationsFile "$env(HOME)/.ms-annotations"

    # The current atom selection mode 0 for off 1 for on.
    variable atomSelectionMode 0

    # Size of cells for seqedit widget
    variable cellSize 16

    # Seq IDs we own in SeqData module
    variable seqlist_unaligned {}
    variable seqlist_aligned {}

    # mapping of VMD molids to seqids
    variable seqMap
    array unset seqMap 

    # mapping of seqids to VMD molids
    variable structMap
    array unset structMap 

    # Random string for starting tmp file names
    variable filePrefix

    # The temp directory location.
    variable tempDir

    # List of created files to use for their eventual deletion
    variable tempFiles {}

    # The current highlight style.
    variable highlightStyle "Licorice"

    # The current highlight style.
    variable highlightColor {ColorID 4}

    # Map of the current representation for each sequence.
    variable representations
    array unset representations 

    # Variable to hold mappings between VMD structures and their sequences
    variable VMDStructSeqMap
    array unset VMDStructSeqMap 

    # The name that the last alignment was saved as.
    variable sessionFilename "untitled.multiseq"

    # The current coloring options.
    variable coloringOptions "all"

    # "Find" dialog selection variables.
    variable findSelectionsActive 0
    variable findSelectionsPosition 0
    variable findSelections 
    variable numFindSelections 0
    array unset findSelections 
    variable inSetNewFind 0

    # Show color scale bar
    variable showColorScaleBar 0

    # Tree-sequence mappings.
    variable treeNodeToSequenceIDMap
    array unset treeNodeToSequenceIDMap 
    variable sequenceIDToTreeNodeMap
    array unset sequenceIDToTreeNodeMap 

    variable filename ""; # filename for text editing

    variable coloringMetric ""; # the current coloring metric

}

proc multiseq {} {

    if {[catch {
        ::MultiSeq::printReference
        ::MultiSeq::createMultiSeq
        ::MultiSeq::showMultiSeq        
    } errorMessage] != 0} {
        global errorInfo
        set callStack [list "MultiSeq Error) " $errorInfo]
        puts [join $callStack "\n"]
        tk_messageBox -type ok -icon error -parent $w -title "Error" -message "MultiSeq failed to start with the following message:\n\n$errorMessage"
        return
    }

    return $::MultiSeq::w
}

namespace eval ::MultiSeq:: {

    proc printReference {} {

        variable printedReference
        variable version
        variable release

        # Print out the reference message.
        if {!$printedReference} {
            set printedReference 1
            puts "MultiSeq Info) MultiSeq r$version$release"
            puts "MultiSeq Reference) In any publication of scientific results based completely or"
            puts "MultiSeq Reference) in part on the use of MultiSeq, please reference:"
            puts "MultiSeq Reference) Elijah Roberts, John Eargle, Dan Wright, and Zaida Luthey-"
            puts "MultiSeq Reference) Schulten. MultiSeq: Unifying sequence and structure data for"
            puts "MultiSeq Reference) evolutionary analysis. BMC Bioinformatics. 2006,7:382."
        }
    }

    proc createMultiSeq {} {

        # Just create the window and initialize data structures
        # No molecule has been selected yet
        # Also set up traces on VMD variables

        global env
        variable selections
        variable w
        variable sessionFilename
        variable i
        variable mol_canvas
        variable label_canvas
        variable canwidth
        variable labelwidth
        variable canheight
        variable monofont
        variable spacing
        variable cellSize
        variable ::SeqEditWidget::widget
        variable ::SeqEditWidget::vmdPopupMenu
        variable filePrefix

        # If already initialized, just turn on 
        if {$w != "" && [winfo exists .multiseq] } {
          wm deiconify $w
          return
        }

        # Set the default temp directory options.
        set filePrefix "multiseq-[lindex [split [expr rand()] .] 1]"
 
        if {[testTempFileSelection $env(TMPDIR) 0]} { 
           setTempFileOptions $env(TMPDIR) "multiseq-[lindex [split [expr rand()] .] 1]"
        } else {
           tk_messageBox -title "MultiSeq Preferences Setup" -icon info \
                -type ok -message \
                "MultiSeq needs a work directory where it can store temporary files.  On the next screen, you will need to specify a folder where you have permission to write temporary files."
           menu_chooseTempDir 
        }

        setPreferences

        # If we don't have a preferences file, set up the preferences.
        if {![::MultiSeqDialog::loadRCFile]} {
            ::MultiSeqDialog::showPreferencesDialog "MultiSeq Preferences" "::MultiSeq::setPreferences"

        # Otherwise, just check for updates.
        } else {
            ::MultiSeqDialog::checkForMetadataUpdates
        }
        setPreferences

        set monofont tkFixed

        set w [toplevel ".multiseq"]
        wm title $w $sessionFilename
        wm minsize $w 500 300

        frame $w.top

        createMenu $w

        # Create the editor canvas.
        frame $w.top.seqdisplay
        ::SeqEditWidget::createWidget $w.top.seqdisplay $cellSize $cellSize

        # Create the VMD popup menu.
        set ::SeqEditWidget::vmdPopupMenu [menu $::SeqEditWidget::widget.vmdPopupMenu -title "VMD" -tearoff no]
        $::SeqEditWidget::vmdPopupMenu add command -label "Show Molecule" -command "::MultiSeq::menu_vmdpopup showmol"
        $::SeqEditWidget::vmdPopupMenu add command -label "Hide Molecule" -command "::MultiSeq::menu_vmdpopup hidemol"
        $::SeqEditWidget::vmdPopupMenu add separator
        $::SeqEditWidget::vmdPopupMenu add command -label "Show Chain" -command "::MultiSeq::menu_vmdpopup showrep"
        $::SeqEditWidget::vmdPopupMenu add command -label "Hide Chain" -command "::MultiSeq::menu_vmdpopup hiderep"
        $::SeqEditWidget::vmdPopupMenu add separator
        $::SeqEditWidget::vmdPopupMenu add cascade -label "Change Representation" -menu $::SeqEditWidget::vmdPopupMenu.rep
        menu $::SeqEditWidget::vmdPopupMenu.rep -tearoff no
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "Bonds" -command "::MultiSeq::menu_vmdpopup changerep Bonds"
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "VDW" -command "::MultiSeq::menu_vmdpopup changerep  VDW"
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "CPK" -command "::MultiSeq::menu_vmdpopup changerep  CPK"
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "Lines" -command "::MultiSeq::menu_vmdpopup changerep  Lines"
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "Licorice" -command "::MultiSeq::menu_vmdpopup changerep  Licorice"
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "Trace" -command "::MultiSeq::menu_vmdpopup changerep  Trace"
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "New Ribbons" -command "::MultiSeq::menu_vmdpopup changerep  NewRibbons"
        $::SeqEditWidget::vmdPopupMenu.rep add command -label "New Cartoon" -command "::MultiSeq::menu_vmdpopup changerep  NewCartoon"

        # Create the group.
        ::SeqEditWidget::createGroup "VMD Protein Structures"

        # Setup the editor listeners.
        ::SeqEditWidget::setSelectionNotificationCommand "::MultiSeq::highlightSelectedCells"
        ::SeqEditWidget::addRemovalNotificationCommand "::MultiSeq::sequencesRemoved"

        # Set the color map.
        ::MultiSeq::ColorMap::VMD::loadColorMap
        ::SeqEditWidget::setColorMap "::MultiSeq::ColorMap::VMD"

        pack $w.top.seqdisplay -fill both -expand 1 ;# -padx {0 1} -pady {0 1}
        pack $w.top -fill both -expand 1

        # Bind the window show and hide events.
        bind $w <Map> {::MultiSeq::showMultiSeq}
        bind $w <Unmap> {::MultiSeq::hideMultiSeq}
    }

# --------------------------------------------------------------------------
    proc showMultiSeq {} {

        variable shown
        variable representations

        if {!$shown} {
            set shown 1

            # Create the color bar.
            #::ColorBar::init

            # Setup all of the VMD event lsiteners.
            setupVMDEventListeners

            # Set up the molecule list
            VMDMoleculesUpdated

            # Show any representations we created.
            foreach key [array names representations "*,sequence"] {
                if {[regexp {^(\d+),sequence$} $key unused sequenceID] == 1} {
                    set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                    foreach repName $representations($key) {
                        set repIndex [mol repindex $molID $repName]
                        if {$repIndex != -1} {
                            mol showrep $molID $repIndex on
                        }
                    }
                }
            }
            foreach key [array names representations "*,selection"] {
                if {[regexp {^(\d+),selection$} $key unused sequenceID] == 1} {
                    set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                    set repName $representations($key)
                    set repIndex [mol repindex $molID $repName]
                    if {$repIndex != -1} {
                        mol showrep $molID $repIndex on
                    }
                }
            }

            # Turn off any representations we should hide.
            foreach key [array names representations "*,hidden"] {
                if {[regexp {^(\d+),hidden$} $key unused sequenceID] == 1} {
                    set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                    set repName $representations($key)
                    set repIndex [mol repindex $molID $repName]
                    if {$repIndex != -1} {
                        mol showrep $molID $repIndex off
                    }
                }
            }

             # Redraw the editor.
            ::SeqEditWidget::redraw
        }
    }

# ----------------------------------------------------------------------
   proc quitMultiSeq {} {
      hideMultiSeq
      seq cleanup
   }
# ----------------------------------------------------------------------
    proc hideMultiSeq {} {

        variable w
        variable shown
        variable filePrefix
        variable tempDir
        variable tempFiles
        variable representations
        global vmd_frame
        global vmd_initialize_structure
        global vmd_trajectory_read
        global vmd_quit
        global vmd_pick_atom

        if {$shown} {

            set shown 0

            # Destroy the color bar.
            #::ColorBar::destroy

            # Delete any files with the temp prefix.
            if {[llength [glob -nocomplain $tempDir/$filePrefix.*]] > 0 } {
               puts "Deleting $tempDir/$filePrefix.*"
               foreach file [glob -nocomplain $tempDir/$filePrefix.*] {
#                puts "Found $file"
                   if {[file exists $file]} {
                       file delete -force $file
                   }
               }    
            }

            # Delete any other temp files that were created.
            foreach file $tempFiles {
                if {[file exists $file]} {
                    file delete -force $file
                }
            }

            # Hide any representations we created.
            foreach key [array names representations "*,sequence"] {
                if {[regexp {^(\d+),sequence$} $key unused sequenceID] == 1} {
                    set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                    foreach repName $representations($key) {
                        set repIndex [mol repindex $molID $repName]
                        if {$repIndex != -1} {
                            mol showrep $molID $repIndex off
                        }
                    }
                }
            }
            foreach key [array names representations "*,selection"] {
                if {[regexp {^(\d+),selection$} $key unused sequenceID] == 1} {
                    set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                    set repName $representations($key)
                    set repIndex [mol repindex $molID $repName]
                    if {$repIndex != -1} {
                        mol showrep $molID $repIndex off
                    }
                }
            }

            # Turn on any representations we turned off.
            foreach key [array names representations "*,hidden"] {
                if {[regexp {^(\d+),hidden$} $key unused sequenceID] == 1} {
                    set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                    set repName $representations($key)
                    set repIndex [mol repindex $molID $repName]
                    if {$repIndex != -1} {
                        mol showrep $molID $repIndex on
                    }
                }
            }

#            seq cleanup

            # Clean up VMD callbacks.
            trace remove variable vmd_frame write "::MultiSeq::VMDMoleculesUpdated"
            trace remove variable vmd_initialize_structure write "::MultiSeq::VMDMoleculesUpdated"
            trace remove variable vmd_trajectory_read write "::MultiSeq::VMDMoleculesUpdated"
            trace remove variable vmd_pick_atom write "::MultiSeq::VMDAtomSelected"
            trace remove variable vmd_quit write "::MultiSeq::hideMultiSeq"
        }
    }

# ----------------------------------------------------------------------
    proc printStackTrace {} {

        global errorInfo
        set callStack [list "MultiSeq Error) " $errorInfo]
        puts [join $callStack "\n"]
    }

# ----------------------------------------------------------------------
    # This method creates the menu being sure to set the menu width to avoid truncating in OS X
    proc createMenu { w } {

        variable highlightStyle
        variable highlightColor

        # Create menubar
        frame $w.top.menubar -relief raised -bd 2
        pack $w.top.menubar -padx 1 -fill x -side top

        # File menu
        menubutton $w.top.menubar.file -text "File" -underline 0 -menu $w.top.menubar.file.menu
        $w.top.menubar.file config -width 5
        pack $w.top.menubar.file -side left
        menu $w.top.menubar.file.menu -tearoff no
        $w.top.menubar.file.menu add command -label "New Session" -command "::MultiSeq::menu_newSession"
        $w.top.menubar.file.menu add command -label "Load Session..." -command "::MultiSeq::menu_loadSession"
        $w.top.menubar.file.menu add command -label "Save Session..." -command "::MultiSeq::menu_saveSession"
        $w.top.menubar.file.menu add separator
        $w.top.menubar.file.menu add command -label "Import Data..." -command "::MultiSeq::menu_importData"
        $w.top.menubar.file.menu add command -label "Export Data..." -command "::MultiSeq::menu_exportData"
        $w.top.menubar.file.menu add separator
        $w.top.menubar.file.menu add command -label "Save Screenshot..." -command "::SeqEditWidget::saveAsPS"
        $w.top.menubar.file.menu add separator
        $w.top.menubar.file.menu add command -label "Preferences..." -command "::MultiSeqDialog::showPreferencesDialog {MultiSeq Preferences} ::MultiSeq::setPreferences"
        $w.top.menubar.file.menu add command -label "Choose work directory..." -command "::MultiSeq::menu_chooseTempDir"
        $w.top.menubar.file.menu add separator
        $w.top.menubar.file.menu add command -label "Cleanup Representations" -command "::MultiSeq::menu_cleanupReps"
        $w.top.menubar.file.menu add command -label "Quit MultiSeq" -command "menu multiseq off"

        # Search menu
        menubutton $w.top.menubar.edit -text "Edit" -menu $w.top.menubar.edit.menu
        $w.top.menubar.edit config -width 5
        pack $w.top.menubar.edit -side left
        menu $w.top.menubar.edit.menu -tearoff no
        $w.top.menubar.edit.menu add cascade -label "Enable Editing" -menu $w.top.menubar.edit.menu.editing
        menu $w.top.menubar.edit.menu.editing -tearoff no
        $w.top.menubar.edit.menu.editing add radio -label "Off" -variable "::SeqEditWidget::editingMode" -value 0
        $w.top.menubar.edit.menu.editing add radio -label "Gaps Only" -variable "::SeqEditWidget::editingMode" -value 1
        $w.top.menubar.edit.menu.editing add radio -label "Full" -variable "::SeqEditWidget::editingMode" -value 2
        $w.top.menubar.edit.menu add command -label "Remove Gaps..." -command "::MultiSeq::menu_removeGaps"
        $w.top.menubar.edit.menu add separator
        $w.top.menubar.edit.menu add command -label "Cut" -command "event generate $w.top.seqdisplay.center.editor <<Cut>>"
        $w.top.menubar.edit.menu add command -label "Copy" -command "event generate $w.top.seqdisplay.center.editor <<Copy>>"
        $w.top.menubar.edit.menu add command -label "Paste" -command "event generate $w.top.seqdisplay.center.editor <<Paste>>"
        $w.top.menubar.edit.menu add separator
        #$w.top.menubar.edit.menu add command -label "Edit in external editor" -command "::SimpleEdit::doSimpleEdit"
        $w.top.menubar.edit.menu add command -label "Edit in text editor" -command "::MultiSeq::textview_doSimpleEdit"

        # Search menu
        menubutton $w.top.menubar.search -text "Search" -menu $w.top.menubar.search.menu
        $w.top.menubar.search config -width 7
        pack $w.top.menubar.search -side left
        menu $w.top.menubar.search.menu -tearoff no
        bind $w "<Control-f>" {::MultiSeq::menu_find new}
        bind $w "<F3>" {::MultiSeq::menu_find next}
        bind $w "<Shift-F3>" {::MultiSeq::menu_find prev}
        $w.top.menubar.search.menu add command -label "Find..." -command "::MultiSeq::menu_find new"
        $w.top.menubar.search.menu add command -label "Find Next" -command "::MultiSeq::menu_find next"
        $w.top.menubar.search.menu add command -label "Find Previous" -command "::MultiSeq::menu_find prev"
        $w.top.menubar.search.menu add separator
        $w.top.menubar.search.menu add command -label "Select Contact Shells..." -command "::MultiSeq::menu_selectContactShells"
        $w.top.menubar.search.menu add command -label "Select Non-Redundant Set..." -command "::MultiSeq::menu_selectNRSet"
        $w.top.menubar.search.menu add command -label "Select Residues..." -command "::MultiSeq::SelectResidues::showSelectResiduesDialog $w"

        # Tools menu
        menubutton $w.top.menubar.tools -text "Tools" -menu $w.top.menubar.tools.menu
        $w.top.menubar.tools config -width 6
        pack $w.top.menubar.tools -side left
        menu $w.top.menubar.tools.menu -tearoff no
        $w.top.menubar.tools.menu add command -label "Stamp Structural Alignment" -command "::MultiSeq::menu_stamp"
        $w.top.menubar.tools.menu add command -label "Sequence Alignment" -command "::MultiSeq::menu_seqAlign"
        $w.top.menubar.tools.menu add command -label "Phylogenetic Tree" -command "::MultiSeq::menu_phylotree"
        $w.top.menubar.tools.menu add command -label "Plot Data" -command "::MultiSeq::menu_plot"

        # Options menu.
        menubutton $w.top.menubar.options -text "Options" -menu $w.top.menubar.options.menu
        $w.top.menubar.options config -width 8
        pack $w.top.menubar.options -side left
        menu $w.top.menubar.options.menu -tearoff no
        $w.top.menubar.options.menu add cascade -label "Atom Picking" -menu $w.top.menubar.options.menu.atomselect
        menu $w.top.menubar.options.menu.atomselect -tearoff no
        $w.top.menubar.options.menu.atomselect add radio -label "Off" -variable "::MultiSeq::atomSelectionMode" -value 0 -command "::MultiSeq::menu_atomSelectionMode"
        $w.top.menubar.options.menu.atomselect add radio -label "On" -variable "::MultiSeq::atomSelectionMode" -value 1 -command "::MultiSeq::menu_atomSelectionMode"        
        $w.top.menubar.options.menu add separator
        $w.top.menubar.options.menu add cascade -label "Grouping" -menu $w.top.menubar.options.menu.groupby
        menu $w.top.menubar.options.menu.groupby -tearoff no
        $w.top.menubar.options.menu.groupby add command -label "From Selection..." -command "::MultiSeq::menu_group selection"
        $w.top.menubar.options.menu.groupby add command -label "Molecule Type" -command "::MultiSeq::menu_group type"
        $w.top.menubar.options.menu.groupby add command -label "Taxonomy..." -command "::MultiSeq::menu_group taxonomy"
        $w.top.menubar.options.menu.groupby add command -label "Custom..." -command "::MultiSeq::menu_group custom"

        # View menu
        menubutton $w.top.menubar.view -text "View" -menu $w.top.menubar.view.menu
        $w.top.menubar.view config -width 5
        pack $w.top.menubar.view -side left
        menu $w.top.menubar.view.menu -tearoff no
        $w.top.menubar.view.menu add cascade -label "Zoom" -menu $w.top.menubar.view.menu.zoom
        menu $w.top.menubar.view.menu.zoom -tearoff no
        $w.top.menubar.view.menu.zoom add command -label "Zoom In" -accelerator "Ctrl +" -command "[namespace current]::menu_zoomIn"
        bind $w "<Control-plus>" {"::MultiSeq::menu_zoomIn"}
        bind $w "<Control-equal>" {"::MultiSeq::menu_zoomIn"}
        bind $w "<Command-plus>" {"::MultiSeq::menu_zoomIn"}
        bind $w "<Command-equal>" {"::MultiSeq::menu_zoomIn"}
        $w.top.menubar.view.menu.zoom add command -label "Zoom Out" -accelerator "Ctrl -" -command "[namespace current]::menu_zoomOut"
        bind $w "<Control-minus>" {"::MultiSeq::menu_zoomOut"}
        bind $w "<Command-minus>" {"::MultiSeq::menu_zoomOut"}
        $w.top.menubar.view.menu.zoom add separator        
        $w.top.menubar.view.menu.zoom add command -label "25%" -command "[namespace current]::menu_zoom 4"
        $w.top.menubar.view.menu.zoom add command -label "50%" -command "[namespace current]::menu_zoom 8"
        $w.top.menubar.view.menu.zoom add command -label "75%" -command "[namespace current]::menu_zoom 12"
        $w.top.menubar.view.menu.zoom add command -label "100%" -command "[namespace current]::menu_zoom 16"
        $w.top.menubar.view.menu.zoom add command -label "150%" -command "[namespace current]::menu_zoom 24"
        $w.top.menubar.view.menu.zoom add command -label "200%" -command "[namespace current]::menu_zoom 30"
        $w.top.menubar.view.menu add separator        
        $w.top.menubar.view.menu add cascade -label "Coloring" -menu $w.top.menubar.view.menu.coloring
        menu $w.top.menubar.view.menu.coloring -tearoff no
        $w.top.menubar.view.menu.coloring add radiobutton -label "Apply to All" -variable "::MultiSeq::coloringOptions" -value "all"
        $w.top.menubar.view.menu.coloring add radiobutton -label "Apply by Group" -variable "::MultiSeq::coloringOptions" -value "group"
        $w.top.menubar.view.menu.coloring add radiobutton -label "Apply to Marked" -variable "::MultiSeq::coloringOptions" -value "marked"
        $w.top.menubar.view.menu.coloring add separator        
        $w.top.menubar.view.menu.coloring add command -label "None" -command "::MultiSeq::menu_changeColoring {}"
        $w.top.menubar.view.menu.coloring add command -label "Add Current Selection" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Selection::calculate"
        $w.top.menubar.view.menu.coloring add command -label "Alignment Position" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Element::calculate"
        $w.top.menubar.view.menu.coloring add cascade -label "Contact" -menu $w.top.menubar.view.menu.coloring.contacts
        menu $w.top.menubar.view.menu.coloring.contacts -tearoff no
        $w.top.menubar.view.menu.coloring.contacts add command -label "Number Contacts" -command "::MultiSeq::menu_changeColoring ::Libbiokit::Metric::calculateContactsPerResidue"
        $w.top.menubar.view.menu.coloring.contacts add command -label "Contact Order" -command "::MultiSeq::menu_changeColoring ::Libbiokit::Metric::calculateContactOrderPerResidue"
        $w.top.menubar.view.menu.coloring add command -label "Insertions" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Insertions::calculate"
        $w.top.menubar.view.menu.coloring add command -label "Mutual Information..." -command "::MultiSeq::menu_changeColoring {::SeqEdit::Metric::MutualInformation::calculate $w} 1"
        $w.top.menubar.view.menu.coloring add command -label "Qres" -command "::MultiSeq::menu_changeColoring ::Libbiokit::Metric::colorQres"
        $w.top.menubar.view.menu.coloring add command -label "Residue Type" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Type::calculate"
        $w.top.menubar.view.menu.coloring add command -label "RMSD" -command "::MultiSeq::menu_changeColoring ::Libbiokit::Metric::calculateRMSDNormalized"
        $w.top.menubar.view.menu.coloring add command -label "Sequence Conservation" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Conservation::setSeqColor"
        $w.top.menubar.view.menu.coloring add command -label "Sequence Entropy (strict)" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Entropy::calculateStrict"
        $w.top.menubar.view.menu.coloring add command -label "Sequence Entropy (similar)" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Entropy::calculateSimilar"
        $w.top.menubar.view.menu.coloring add command -label "Sequence Identity" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::PercentIdentity::color"
        $w.top.menubar.view.menu.coloring add cascade -label "Sequence Similarity" -menu $w.top.menubar.view.menu.coloring.similarity
        menu $w.top.menubar.view.menu.coloring.similarity -tearoff no
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 30" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate30"
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 40" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate40"
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 50" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate50"
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 60" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate60"
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 70" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate70"
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 80" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate80"
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 90" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate90"
        $w.top.menubar.view.menu.coloring.similarity add command -label "BLOSUM 100" -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculate100"
        $w.top.menubar.view.menu.coloring.similarity add command -label "Custom..." -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Similarity::calculateCustom"
        $w.top.menubar.view.menu.coloring add command -label "Signatures..." -command "::MultiSeq::menu_changeColoring {::SeqEdit::Metric::Signatures::calculate $w} 1"
        $w.top.menubar.view.menu.coloring add command -label "Custom..." -command "::MultiSeq::menu_changeColoring custom"
        $w.top.menubar.view.menu.coloring add separator        
        $w.top.menubar.view.menu.coloring add command -label "Import..." -command "::MultiSeq::menu_changeColoring ::SeqEdit::Metric::Import::readFile"
        $w.top.menubar.view.menu.coloring add command -label "Refresh Colors" -command "::MultiSeq::ColorMap::VMD::loadColorMap; ::SeqEditWidget::setColorMap ::MultiSeq::ColorMap::VMD"        
        $w.top.menubar.view.menu add cascade -label "Highlight Style" -menu $w.top.menubar.view.menu.highlightmenu
        menu $w.top.menubar.view.menu.highlightmenu -tearoff no
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "Bonds" -command "[namespace current]::setHighlightStyle Bonds"
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "VDW" -command "[namespace current]::setHighlightStyle VDW"
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "Lines" -command "[namespace current]::setHighlightStyle Lines"
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "CPK" -command "[namespace current]::setHighlightStyle CPK"
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "Licorice" -command "[namespace current]::setHighlightStyle Licorice"
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "Trace" -command "[namespace current]::setHighlightStyle Trace"
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "New Ribbons" -command "[namespace current]::setHighlightStyle NewRibbons"
        $w.top.menubar.view.menu.highlightmenu add radiobutton -label "New Cartoon" -command "[namespace current]::setHighlightStyle NewCartoon"
        $w.top.menubar.view.menu add cascade -label "Highlight Color" -menu $w.top.menubar.view.menu.highlightcolor
        menu $w.top.menubar.view.menu.highlightcolor -tearoff no
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Name" -command "[namespace current]::setHighlightColor Name"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Type" -command "[namespace current]::setHighlightColor Type"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "ResName" -command "[namespace current]::setHighlightColor ResName"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "ResType" -command "[namespace current]::setHighlightColor ResType"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Molecule" -command "[namespace current]::setHighlightColor Molecule"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Beta" -command "[namespace current]::setHighlightColor Beta"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "User" -command "[namespace current]::setHighlightColor User"
        $w.top.menubar.view.menu.highlightcolor add separator        
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Blue" -command "[namespace current]::setHighlightColor {ColorID 0}"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Red" -command "[namespace current]::setHighlightColor {ColorID 1}"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Orange" -command "[namespace current]::setHighlightColor {ColorID 3}"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Yellow" -command "[namespace current]::setHighlightColor {ColorID 4}"
        $w.top.menubar.view.menu.highlightcolor add radiobutton -label "Ice Blue" -command "[namespace current]::setHighlightColor {ColorID 15}"
        $w.top.menubar.view.menu add separator        
        #$w.top.menubar.view.menu add checkbutton -label "Color scale" -variable "::ColorBar::colorBarState"
	$w.top.menubar.view.menu add checkbutton -label "Color scale" -command "save_vp 1;if { \$::MultiSeq::showColorScaleBar == 1 } {::ColorScaleBar::delete_color_scale_bar;set ::MultiSeq::showColorScaleBar 0} else {::ColorScaleBar::color_scale_bar .5 .05 0 1 0 100 2 white 0 -1.325 -1 1; set ::MultiSeq::showColorScaleBar 1};retrieve_vp 1"
	#$w.top.menubar.view.menu add checkbutton -label "Color scale" -command "::ColorScaleBar::gui"
	$w.top.menubar.view.menu add checkbutton -variable ::SeqEditWidget::drawZoom -label "Zoom Window" -command "::SeqEditWidget::toggleZoom"

		
        # Help menu
        menubutton $w.top.menubar.help -text "Help" -menu $w.top.menubar.help.menu
        $w.top.menubar.help config -width 5
        pack $w.top.menubar.help -side right
        menu $w.top.menubar.help.menu -tearoff no
        $w.top.menubar.help.menu add command -label "MultiSeq Home Page" -command "vmd_open_url http://faculty.scs.illinois.edu/schulten/multiseq"
#        $w.top.menubar.help.menu add command -label "MultiSeq Help" -command "vmd_open_url http://faculty.scs.illinois.edu/schulten/multiseq/manual"
         $w.top.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www]]plugins/multiseq"
    }

# ----------------------------------------------------------------------
   # This methods sets up the VMD event listeners.
   proc setupVMDEventListeners {} {

      global vmd_frame
      global vmd_initialize_structure
      global vmd_trajectory_read
      global vmd_quit
      global vmd_pick_atom

      # Update the molecules when molecules are deleted or added
      trace add variable vmd_frame write "::MultiSeq::VMDMoleculesUpdated"
      trace add variable vmd_initialize_structure write \
                                         "::MultiSeq::VMDMoleculesUpdated"
      trace add variable vmd_trajectory_read write \
                                         "::MultiSeq::VMDMoleculesUpdated"

      # Trace for an atom selected in VMD.
      trace add variable vmd_pick_atom write "::MultiSeq::VMDAtomSelected"

      # Trace for program exit, to cleanup temp files etc
      trace add variable vmd_quit write "::MultiSeq::quitMultiSeq"
   }

#   ----------------------------------------------------------------
   # This method is called by the interpreter whenever VMD's molecule list may have changed.
   proc VMDMoleculesUpdated {args} {
#      puts "multiseq.tcl.VMDMoleculesUpdated.start $args"
      variable representations
      variable w

      if {[catch {
         # Load any new sequences from VMD.
         set updatedSequences [::SeqData::VMD::updateVMDSequences]
         set addedSequences [lindex $updatedSequences 0]
         set removedSequences [lindex $updatedSequences 1]

         # Add existing non-linked sequences to editor
         set newlyAddedSequences {}
         foreach addedSequence $addedSequences {
            if {![::SeqEditWidget::containsSequence $addedSequence]} {
               lappend newlyAddedSequences $addedSequence
            }
         }

         # See if we have any newly added sequences.
         if {$newlyAddedSequences != {} } {

            # Add them to the editor.
            foreach newlyAddedSequence $newlyAddedSequences {
               if {[::SeqData::getType $newlyAddedSequence] == "protein"} {
                  ::SeqEditWidget::addSequences $newlyAddedSequence \
                                                   "VMD Protein Structures"
               } elseif {[::SeqData::getType $newlyAddedSequence] == "rna" || \
                         [::SeqData::getType $newlyAddedSequence] == "dna"} {
                  ::SeqEditWidget::addSequences $newlyAddedSequence \
                                                      "VMD Nucleic Structures"
               } else {
                  ::SeqEditWidget::addSequences $newlyAddedSequence \
                                                      "VMD Structures"
               }
            }
         }

         # Go through each added structure and set its initial representations.
         foreach sequenceID $addedSequences {

            # Set the first representation of the molecule to off.
            set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
            if {[molinfo $molID get numreps] == 1} {
               mol showrep $molID 0 off
               set representations($sequenceID,hidden) [mol repname $molID 0]
            }

            # Create representations.
            createSequenceRepresentations $sequenceID
         }

         # See if any sequences were removed.
         if {$removedSequences != {} } {

            # Remove them from the editor.
            ::SeqEditWidget::removeSequences $removedSequences

            # Remove them from the representation list.
            foreach sequenceID $removedSequences {
               if {[info exists representations($sequenceID,sequence)]} {
                  unset representations($sequenceID,sequence)
               }
               if {[info exists representations($sequenceID,selection)]} {
                  unset representations($sequenceID,selection)
               }
               if {[info exists representations($sequenceID,hidden)]} {
                  unset representations($sequenceID,hidden)
               }
            }
         }

         # Redraw the editor.
         ::SeqEditWidget::redraw
         # Redraw color bar
         # ::ColorBar::reinit

         # changes in the structure can cause the status bar to need to
         # change
         ::SeqEditWidget::statusBarSelectionChange


      } errorMessage] != 0} {
         global errorInfo
         set callStack [list "MultiSeq Error) " $errorInfo]
         puts [join $callStack "\n"]
         tk_messageBox -type ok -icon error -parent $w -title "Error" -message \
                  "VMDMoleculesUpdated failed with the following message:\n\n$errorMessage"
      }
   } ;# end of VMDMoleculesUpdated

   # ----------------------------------------------------------------
    proc createSequenceRepresentations {sequenceID} {

        variable representations

        # Get the mol id and chain of the sequence.        
        set molID [lindex [::SeqData::VMD::getMolIDForSequence $sequenceID] 0]
        set type [::SeqData::getType $sequenceID]

        # If we only have one atom per residue, use a tube representation for protein and nucleic.
        if {$type == "protein" || $type == "rna" || $type == "dna"} {
            set atoms [atomselect $molID [::SeqData::VMD::getSelectionStringForElements $sequenceID 0]]
            if {[$atoms num] == 1} {
                set type "tube"
            }
            $atoms delete
        }

        # Create the representation for this sequence.
        if {$type == "protein"} {

            # Add the representation, if we don't already have it.
            if {![info exists representations($sequenceID,sequence)] || [mol repindex $molID $representations($sequenceID,sequence)] == -1} {                
                set representations($sequenceID,sequence) [createRepresentation $molID "NewCartoon" "Name" [::SeqData::VMD::getSelectionStringForSequence $sequenceID]]
            }

        } elseif {$type == "rna" || $type == "dna"} {

            # Add the representations, if we don't already have them.
            if {![info exists representations($sequenceID,sequence)] || [mol repindex $molID $representations($sequenceID,sequence)] == -1} {                
                set representations($sequenceID,sequence) [createRepresentation $molID "NewCartoon" {ColorID 5} [::SeqData::VMD::getSelectionStringForSequence $sequenceID]]
            }

        } elseif {$type == "tube"} {

            # Add the representations, if we don't already have them.
            if {![info exists representations($sequenceID,sequence)] || [mol repindex $molID $representations($sequenceID,sequence)] == -1} {                
                set representations($sequenceID,sequence) [createRepresentation $molID "Tube" "Name" [::SeqData::VMD::getSelectionStringForSequence $sequenceID]]
            }

        } else {

            # Add the representation, if we don't already have it.
            if {![info exists representations($sequenceID,sequence)] || [mol repindex $molID $representations($sequenceID,sequence)] == -1} {                
                set representations($sequenceID,sequence) [createRepresentation $molID {Licorice 0.300000 8.000000 6.000000} "Name" [::SeqData::VMD::getSelectionStringForSequence $sequenceID]]
            }
        }
    } ; # end of createSequenceRepresentations

# -------------------------------------------------------------------------
    proc createRepresentation {molID style color selection} {
        mol rep $style
        mol color $color
        mol addrep $molID
        set name [mol repname $molID [expr [molinfo $molID get numreps]-1]]
        mol modselect [mol repindex $molID $name] $molID $selection
        return $name
    }

# -------------------------------------------------------------------------
   proc VMDAtomSelected {args} {
      global vmd_pick_mol
      global vmd_pick_atom
      global vmd_pick_shift_state
      variable atomSelectionMode

      # Set the mouse mode appropriately.
      if {$atomSelectionMode == 1} {

         # Delete any label that was created.
         if {[llength [label list Atoms]] > 0} {
            label delete Atoms [expr [llength [label list Atoms]]-1]
         }

         # Get the mol id.
         set molID $vmd_pick_mol

         # Get the chain and residue.
         set atom [atomselect $molID "index $vmd_pick_atom"]
         set chain [$atom get chain]
         set segname ""
         if {$chain == "X"} {
            set segname [$atom get segname]
         }
         set residue [lindex [$atom get {resid insertion altloc}] 0]

         # Get the sequence and the element.
         set addMode $vmd_pick_shift_state
         set sequenceIDs [::SeqData::VMD::getSequenceIDForMolecule \
                                                     $molID $chain $segname]
         foreach sequenceID $sequenceIDs {
            if {[::SeqEditWidget::containsSequence $sequenceID]} {
               set position [::SeqData::VMD::getElementForResidue \
                                                        $sequenceID $residue]
               ::SeqEditWidget::setSelectedCell $sequenceID $position $addMode 0
               set addMode 1
            }
         }

         # Delete the atom selection.
         $atom delete
         unset atom
      }
   } ; # end of VMDAtomSelected

# ------------------------------------------------------------------------
    # This method sets the current selection highlighting style.
    proc setHighlightStyle {newHighlightStyle} {

        variable highlightStyle
        variable representations

        # Set the new default style.
        set highlightStyle $newHighlightStyle

        # Adjust any representations that we have already created.
        foreach key [array names representations "*,selection"] {
            if {[regexp {(\d+),selection} $key unused sequenceID] == 1} {
                set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                set repIndex [mol repindex $molID $representations($sequenceID,selection)]
                if {$repIndex != -1} {
                    mol modstyle $repIndex $molID $highlightStyle
                }
            }
        }
    }

# ------------------------------------------------------------------------
   # This method sets the current selection highlighting color.
   proc setHighlightColor {newHighlightColor} {

      variable highlightColor
      variable representations

      # Set the new default color.
      set highlightColor $newHighlightColor

      # Adjust any representations that we have already created.
      foreach key [array names representations "*,selection"] {
         if {[regexp {(\d+),selection} $key unused sequenceID] == 1} {
            set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]
            set repIndex [mol repindex $molID \
                                       $representations($sequenceID,selection)]
            if {$repIndex != -1} {
               mol modcolor $repIndex $molID $highlightColor
            }
         }
      }
   }


# -------------------------------------------------------------------------
   # highlights the currently selected residues.
   proc highlightSelectedCells {} {

      variable highlightStyle
      variable highlightColor
      variable representations


#      set selected [::SeqEditWidget::getSelectedCells]
#      puts "multiseq.highlightSelectedCells.selected: $selected"

      set selected [::SeqEditWidget::getSelectedCellsCombinedBySequence]
#      puts "multiseq.highlightSelectedCells.selected: $selected"

#      parray representations

      # Go through each sequences that has something selected.
      set selectedSequenceIDs {}
      for {set i 0} {$i < [llength $selected]} {incr i 2} {

         # Get the sequence id.
         set sequenceID [lindex $selected $i]
         lappend selectedSequenceIDs $sequenceID

         # Make sure this is a structure.            
         if {[::SeqData::hasStruct $sequenceID] == "Y"} {
#            puts "we have a structure: seqID: $sequenceID"
#            parray representations

            # Get the mol id and chain of the sequence.        
            set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]

            # Make sure we have a representation for this sequence id.
            if {![info exists representations($sequenceID,selection)] || \
               [mol repindex $molID $representations($sequenceID,selection)] \
                                                                == -1} {
               set representations($sequenceID,selection) \
                                  [createRepresentation $molID $highlightStyle \
                                                       $highlightColor "none"]
            }

            # Get the selection string for the elements.                
            set selectionString [::SeqData::VMD::getSelectionStringForElements \
                                     $sequenceID [lindex $selected [expr $i+1]]]
#            puts "selectionString: $selectionString"
            # Set the selection for the molecule.
            mol modselect [mol repindex $molID \
                                 $representations($sequenceID,selection)] \
                                 $molID $selectionString
         }
      }

      # Reset the selection for any molecules not currently selected.
      foreach key [array names representations "*,selection"] {
         if {[regexp {(\d+),selection} $key unused sequenceID] == 1} {
            if {[lsearch $selectedSequenceIDs $sequenceID] == -1} {
               set molID [lindex [SeqData::VMD::getMolIDForSequence \
                                                            $sequenceID] 0]
               set repIndex [mol repindex $molID \
                                    $representations($sequenceID,selection)]
               if {$repIndex != -1} {
                  mol modselect $repIndex $molID "none"
               }
            }
         }
      }
#      puts "at end, we have reps:"
#      parray representations
   }  ;# end of highlightSelectedCells

# ---------------------------------------------------------------------------
    # called whenever sequences are removed from the editor.
    proc sequencesRemoved {sequenceIDs} {

        variable representations

        # Go through each sequence.
        foreach sequenceID $sequenceIDs {

            # See if this is a structure.
            if {[::SeqData::hasStruct $sequenceID] == "Y"} {

                # Get the molecule id.
                set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]

                # Remove any sequence representations.
                if {[info exists representations($sequenceID,sequence)]} {
                    foreach repName $representations($sequenceID,sequence) {
                        set repIndex [mol repindex $molID $repName]
                        if {$repIndex != -1} {
                            mol delrep $repIndex $molID
                        }
                    }
                    unset representations($sequenceID,sequence)
                }

                # Remove any selection representations.
                if {[info exists representations($sequenceID,selection)]} {
                    set repIndex [mol repindex $molID $representations($sequenceID,selection)]
                    if {$repIndex != -1} {
                        mol delrep $repIndex $molID
                    }
                    unset representations($sequenceID,selection)
                }
            }
        }
    }

# -----------------------------------------------------------------------
    proc menu_vmdpopup {menuAction {actionParam ""}} {

        variable ::SeqEditWidget::popupMenuParameters
        variable representations

        # Go through the sequences.
        foreach sequenceID $::SeqEditWidget::popupMenuParameters {

            # Get the molid and chain.
            set molID [lindex [SeqData::VMD::getMolIDForSequence $sequenceID] 0]

            # See what we need to do.
            if {$menuAction == "showmol"} {

                # Show the molecule
                mol on $molID

            } elseif {$menuAction == "hidemol"} {

                # Hide the molecule
                mol off $molID

            } elseif {$menuAction == "showrep"} {

                # If we have a sequence representation, show it.
                if {[info exists representations($sequenceID,sequence)]} {
                    foreach repName $representations($sequenceID,sequence) {
                        set repIndex [mol repindex $molID $repName]
                        if {$repIndex != -1} {
                            mol showrep $molID $repIndex on
                        }
                    }
                }

            } elseif {$menuAction == "hiderep"} {

                # If we have a sequence representation, hide it.
                if {[info exists representations($sequenceID,sequence)]} {
                    foreach repName $representations($sequenceID,sequence) {
                        set repIndex [mol repindex $molID $repName]
                        if {$repIndex != -1} {
                            mol showrep $molID $repIndex off
                        }
                    }
                }
            } elseif {$menuAction == "changerep"} {

                # If we have a sequence representation, hide it.
                if {[info exists representations($sequenceID,sequence)]} {
                    foreach repName $representations($sequenceID,sequence) {
                        set repIndex [mol repindex $molID $repName]
                        if {$repIndex != -1} {
                            mol modstyle $repIndex $molID $actionParam
                        }
                    }
                }
            }
        }
    }

# -----------------------------------------------------------------------
   proc menu_newSession {} {
      # Confirmation dialog
      set notSure [tk_dialog .areYouSure "Confirm new session" \
          "Are you sure you wish to start a new session?\nAll data from your current session will be lost." "warning" 0 "Yes" "No"]
      if { $notSure } {
         return
      }

      variable w
      variable sessionFilename

      # Reset the session name.
      set sessionFilename "untitled.multiseq"
      wm title $w $sessionFilename

      # Reset VMD and the editor.
      foreach molID [molinfo list] {
         mol delete $molID
      }
      ::SeqEditWidget::setSequences {}
      ::SeqEditWidget::setGroups {}
   }

# --------------------------------------------------------------------------
   proc menu_loadSession {} {
      variable w
      variable sessionFilename

      set filename [tk_getOpenFile -filetypes {{{MultiSeq Sessions} \
                         {.multiseq}} {{All Files} *}} -title "Load Session"]
      if {$filename != ""} {

         # print the filename.
         set sessionFilename [lindex [split $filename "/\\"] end]
         wm title $w $sessionFilename

         # Reset VMD and the editor.
         foreach molID [molinfo list] {
            mol delete $molID
         }
         ::SeqEditWidget::setSequences {}
         ::SeqEditWidget::setGroups {}

         # Load the alignment.
         set sequenceIDs [::SeqData::Fasta::loadSequences $filename]

         # Add the sequences into their groups.
         set inUnknownGroup {}
         set structuresToLoad {}
         foreach sequenceID $sequenceIDs {

            # Get the multiseq data.
            set msData [::SeqData::getSourceData $sequenceID "ms2"]
            if {$msData == {}} {
               set msData [::SeqData::getSourceData $sequenceID "ms"]
            }
            #puts "multiseq.loadSession seqID: $sequenceID,msData: <$msData>"

            # If we have a starting residue, set it.
            if {[llength $msData] >= 6} {
               ::SeqData::setFirstResidue $sequenceID [lindex $msData 5]
            }

            # If we have a group, use it.
            if {[llength $msData] >= 2} {
               ::SeqEditWidget::addSequences $sequenceID \
                                                     [lindex $msData 1] end 0
            } else {
               lappend inUnknownGroup $sequenceID
            }

            # If we have a structure associated with this sequence, load it.
            if {[llength $msData] >= 4 && [lindex $msData 2] != "X"} {
               # Register sequence that structure should be correlated to.
               ::SeqData::VMD::registerSequenceForVMDStructure \
                             [lindex $msData 2] [lindex $msData 3] $sequenceID

               # Add the file to the list, if it is not already there.
               if {[lsearch $structuresToLoad \
                              "$filename.data/[lindex $msData 2].pdb"] == -1} {
                  lappend structuresToLoad \
                                        "$filename.data/[lindex $msData 2].pdb"
               }
            }

            # Set the representation for the sequence.
            if {[llength $msData] >= 5} {
               ::SeqEditWidget::setRepresentations $sequenceID \
                                                         [lindex $msData 4] 0
            }
         }

#         puts "multiseq.tcl.loadSession after sequence loop"

         # Add the sequences in unknown groups.
         if {[llength $inUnknownGroup] > 0} {
            ::SeqEditWidget::addSequences $inUnknownGroup "Sequences" end 0
         }

         # Import the coloring.  At this point, structures should already
         # be there
         if {[file exists "$filename.data/metric.dat"]} {
            ::SeqEdit::Metric::Import::setFilename "$filename.data/metric.dat"
            ::SeqEditWidget::setColoring ::SeqEdit::Metric::Import::readFile \
                                                               $sequenceIDs 0
         }

         # puts "multiseq.tcl.loadSession before metadata"
         # Import the metadata
         if {[file exists "$filename.data/meta.dat"]} {
            set seqs [::SeqEditWidget::getSequences]

            set fp [open "$filename.data/meta.dat"]

            # Read data into some structures
            set linenum 0
            while {[gets $fp line] >= 0} {
               set splitline [split $line "^"]
               set name [lindex $splitline 0]
               set fields [lrange $splitline 1 end]

               if {[::SeqData::getName [lindex $seqs $linenum]] == $name} {
                  foreach field $fields {
                     set annotation [split $field "|"]
                     ::SeqData::addAnnotation [lindex $seqs $linenum] \
                                  [lindex $annotation 0] [lindex $annotation 1]
                  }
               } else {
                  puts "MultiSeq Error) on line $linenum of metadata: names don't match.  Loaded seq: <[::SeqData::getName [lindex $seqs $linenum]]>, Seq in file: <$name>"
               }
               incr linenum
            }
         }

         #puts "multiseq.tcl.loadSession before setIgnoreNonRegisteredStructs"
         # Add the structures.
         ::SeqData::VMD::setIgnoreNonRegisteredStructures 1
#        puts "multiseq.loadSession add struct. structs2load: $structuresToLoad"
         foreach structureToLoad $structuresToLoad {
            set molID [mol new $structureToLoad waitfor all]
         }
#         puts "multiseq.tcl.loadSession before after idle"
         after idle {
            ::SeqData::VMD::setIgnoreNonRegisteredStructures 0
            ::MultiSeq::applyColoringToMolecules
         }
#         puts "multiseq.tcl.loadSession before redraw"

         # Redraw the editor.            
         ::SeqEditWidget::redraw
      }
   } ;# end of proc menu_loadSession 

# ----------------------------------------------------------------------------
   # This method saves the current alignment.
   proc menu_saveSession {} {

      variable w
      variable sessionFilename

      # Get the filename to use.
      set filename [tk_getSaveFile -filetypes {{{MultiSeq Sessions} \
           {.multiseq}}} -title "Save Session" -initialfile $sessionFilename]

      # If we got a filename, save the sequences.
      if { $filename != "" } {

         # Save the new session names.
         set sessionFilename [lindex [split $filename "/\\"] end]
         wm title $w $sessionFilename

         # Update all of the sequences with the multiseq data.
         set moleculesToSave {}
         foreach sequenceID [::SeqEditWidget::getSequences] {

            # Set the multiseq data.
            set sequenceName [::SeqData::getName $sequenceID]
            regsub -all {\s} $sequenceName _ sequenceName
            set groupName [::SeqEditWidget::getGroup $sequenceID]
            regsub -all {\s} $groupName _ groupName
            set msData [list $sequenceName $groupName]

            # Set out molecule data based on if it is a sequence or a structure.
            if {[::SeqData::hasStruct $sequenceID] == "Y"} {

               # Figure out the molecule name and chain.
               set molID [lindex [::SeqData::VMD::getMolIDForSequence \
                                                             $sequenceID] 0]
               set molName [::SeqData::VMD::getMoleculeName $molID]
               set molChain [lindex [::SeqData::VMD::getMolIDForSequence \
                                                              $sequenceID] 1]
               lappend msData $molName $molChain

               # Add molecule to list of those to save, if not already there.
               if {[lsearch $moleculesToSave $molID] == -1} {
                  lappend moleculesToSave $molID
               }

            } else {
               lappend msData "X" "X"  
            }

            # Set the multiseq representation.
            lappend msData [lindex [::SeqEditWidget::getRepresentations \
                                                              $sequenceID] 0]

            # Add the starting residue.
            lappend msData [::SeqData::getFirstResidue $sequenceID]

            # Save the multiseq data to the sequence.
            ::SeqData::setSourceData $sequenceID "ms2" $msData
         }

         # Save the sequences.
         ::SeqData::Fasta::saveSequences [::SeqEditWidget::getSequences] \
                                                                 $filename

         # Make sure the data directory exists.
         file mkdir "$filename.data"

         # Save the metric data...  AKA the coloring
         ::SeqEdit::Metric::Import::setFilename "$filename.data/metric.dat"
         ::SeqEdit::Metric::Import::writeFile

         # Save the structures.
         foreach molID $moleculesToSave {            
            set atoms [atomselect $molID all]
            $atoms writepdb \
                   "$filename.data/[SeqData::VMD::getMoleculeName $molID].pdb"
            $atoms delete
         }

         # Save metadata
         set fp [open "$filename.data/meta.dat" "w"]
         foreach sequenceID [::SeqEditWidget::getSequences] {
           # get data 
           set sequenceName [::SeqData::getName $sequenceID]
           if {[array exists annotations]} {
            unset annotations
           }
           array set annotations [::SeqData::getAllAnnotations $sequenceID]
           puts -nonewline $fp "$sequenceName"
           # write data 
           # I'm using weird field seperators so that there's no confusion
           # with valid data in the written fields: '|' to seperate 
           # list elements, and '^' to seperate different entries
           foreach key [array names annotations] {
            set name [regsub {.*,.*,} $key ""] 
            set annotation [regsub {\|} $annotations($key) ":"]
            if {$annotation != ""} {
              puts -nonewline $fp "^$name|$annotation"
            }
           }
           puts $fp ""
         }
         close $fp
      }
   } ; # end of menu_saveSession

# -------------------------------------------------------------------------
   proc importFromFile opt {
      variable w
      variable tempDir
      variable tempFiles

      upvar $opt options
      # Go through each filename.
      set filenames [regexp -inline -all -- {[^\"\,]+} $options(filenames)]
      foreach filename $filenames {

         set filename [string trim $filename]

         # If it is a fasta file, bring in its sequences.
         if {[string first ".fasta" $filename] != -1} {

            # Load the sequences.
            puts "MultiSeq [clock format [clock seconds]]) Loading sequences."
            set sequences [::SeqData::Fasta::loadSequences $filename]
            puts "MultiSeq [clock format [clock seconds]]) Done Loading [llength $sequences] sequences."

            # Get the sequences already in the editor.
            puts "MultiSeq [clock format [clock seconds]]) Getting previously loaded sequences."
            set editorSequences [::SeqEditWidget::getSequences]
            puts "MultiSeq [clock format [clock seconds]]) Done Getting previously loaded sequences."

            # Go through each sequence.
            puts "MultiSeq [clock format [clock seconds]]) Checking to see if we just need to update."                  
            set newSequences {}
            set updatedSequences {}
            foreach sequenceID $sequences {

               # See if this sequence name is already in the editor.
               set matchingSequenceID -1
               foreach editorSequenceID $editorSequences {
                  if {[::SeqData::getName $sequenceID] == [::SeqData::getName $editorSequenceID]} {
                     set matchingSequenceID $editorSequenceID
                     break
                  }
               }

               # If there is already a sequence with this name, just update the alignment.
               if {$matchingSequenceID != -1} {
                  ::SeqData::setSeq $matchingSequenceID [::SeqData::getSeq $sequenceID]
                  lappend updatedSequences $matchingSequenceID

               # Otherwise add the sequence.
               } else {
                  lappend newSequences $sequenceID
               }
            }
            puts "MultiSeq [clock format [clock seconds]]) Done just checking to see if we need to update sequences."

            # If we have updated sequences, update them.
            puts "MultiSeq [clock format [clock seconds]]) Actually update sequences."
            if {[llength $updatedSequences] > 0} {
               ::SeqEditWidget::updateSequences $updatedSequences
            }
            puts "MultiSeq [clock format [clock seconds]]) Done Updating sequences."

            # If we have new sequences, add them.
            puts "MultiSeq [clock format [clock seconds]]) Adding new sequences."
            if {[llength $newSequences] > 0} {
               ::SeqEditWidget::addSequences $newSequences "Sequences"
               if {$options(loadStructures)} {
                  loadStructuresForSequences $newSequences
               }
            }
            puts "MultiSeq [clock format [clock seconds]]) Done Adding new sequences."

         # ALN file, handled like FASTA
         } elseif {[string first ".aln" $filename] != -1} {

            # Load the sequences.
            set sequences [::SeqData::Aln::loadSequences $filename]

            # Get the sequences already in the editor.
            set editorSequences [::SeqEditWidget::getSequences]

            # Go through each sequence.
            set newSequences {}
            set updatedSequences {}
            foreach sequenceID $sequences {

               # See if this sequences is already in the editor.
               set matchingSequenceID -1
               foreach editorSequenceID $editorSequences {
                  if {[::SeqData::getName $sequenceID] == [::SeqData::getName $editorSequenceID]} {
                     set matchingSequenceID $editorSequenceID
                     break
                  }
               }

               # If there is already a sequence with this name, just update the alignment.
               if {$matchingSequenceID != -1} {
                  ::SeqData::setSeq $matchingSequenceID [::SeqData::getSeq $sequenceID]
                  lappend updatedSequences $matchingSequenceID

               # Otherwise add the sequence.
               } else {
                  lappend newSequences $sequenceID
               }
            }

            # If we have updated sequences, update them.
            if {[llength $updatedSequences] > 0} {
               ::SeqEditWidget::updateSequences $updatedSequences
            }

            # If we have new sequences, add them.
            if {[llength $newSequences] > 0} {
               ::SeqEditWidget::addSequences $newSequences "Sequences"
               if {$options(loadStructures)} {
                  loadStructuresForSequences $newSequences
               }
            }

         # ALN file, handled like FASTA
         } elseif {[string first ".phy" $filename] != -1 || [string first ".ph" $filename] != -1} {

            # Load the sequences.
            set sequences [::SeqData::Phy::loadSequences $filename]

            # Get the sequences already in the editor.
            set editorSequences [::SeqEditWidget::getSequences]

            # Go through each sequence.
            set newSequences {}
            set updatedSequences {}
            foreach sequenceID $sequences {

               # See if this sequences is already in the editor.
               set matchingSequenceID -1
               foreach editorSequenceID $editorSequences {
                  if {[::SeqData::getName $sequenceID] == [::SeqData::getName $editorSequenceID]} {
                     set matchingSequenceID $editorSequenceID
                     break
                  }
               }

               # If there is already a sequence with this name, just update the alignment.
               if {$matchingSequenceID != -1} {
                  ::SeqData::setSeq $matchingSequenceID [::SeqData::getSeq $sequenceID]
                  lappend updatedSequences $matchingSequenceID

               # Otherwise add the sequence.
               } else {
                  lappend newSequences $sequenceID
               }
            }

            # If we have updated sequences, update them.
            if {[llength $updatedSequences] > 0} {
               ::SeqEditWidget::updateSequences $updatedSequences
            }

            # If we have new sequences, add them.
            if {[llength $newSequences] > 0} {
               ::SeqEditWidget::addSequences $newSequences "Sequences"
               if {$options(loadStructures)} {
                  loadStructuresForSequences $newSequences
               }
            }

         # NEX file, handled like ALN
         } elseif {[string first ".nex" $filename] != -1} {

            # Load the sequences.
            set sequences [::SeqData::Nex::loadSequences $filename]

            # Get the sequences already in the editor.
            set editorSequences [::SeqEditWidget::getSequences]

            # Go through each sequence.
            set newSequences {}
            set updatedSequences {}
            foreach sequenceID $sequences {

               # See if this sequences is already in the editor.
               set matchingSequenceID -1
               foreach editorSequenceID $editorSequences {
                  if {[::SeqData::getName $sequenceID] == [::SeqData::getName $editorSequenceID]} {
                     set matchingSequenceID $editorSequenceID
                     break
                  }
               }

               # If already a seq with this name, just update alignment.
               if {$matchingSequenceID != -1} {
                  ::SeqData::setSeq $matchingSequenceID [::SeqData::getSeq $sequenceID]
                  lappend updatedSequences $matchingSequenceID

               # Otherwise add the sequence.
               } else {
                  lappend newSequences $sequenceID
               }
            }

            # If we have updated sequences, update them.
            if {[llength $updatedSequences] > 0} {
               ::SeqEditWidget::updateSequences $updatedSequences
            }

            # If we have new sequences, add them.
            if {[llength $newSequences] > 0} {
               ::SeqEditWidget::addSequences $newSequences "Sequences"
               if {$options(loadStructures)} {
                  loadStructuresForSequences $newSequences
               }
            }

         # If this is a SCOP domain, load it.
         } elseif {[::SeqData::SCOP::isValidSCOPName $filename]} {

            # Download the file.
            set structureFilename "$tempDir/$filename"
            ::SeqData::Astral::saveStructure $filename $structureFilename
            lappend tempFiles $structureFilename

            # Add the structure.
            set molID [mol new $structureFilename type pdb waitfor all]

         # If this is a SCOP identifier, load the class.
         } elseif {[::SeqData::SCOP::isValidSCOPIdentifier $filename]} {

            # Get the SCOP structures for this class.
            foreach structure [::SeqData::SCOP::getStructureNamesForIdentifier $filename 0] {

               # Download the file.
               set structureFilename "$tempDir/$structure"
               ::SeqData::Astral::saveStructure $structure $structureFilename
               lappend tempFiles $structureFilename

               # Add the structure.
               mol new $structureFilename type pdb waitfor all
            }

         # If this is an ent file, assume it is really a PDB.
         } elseif {[string first ".ent" [string tolower $filename]] != -1} {

            # Add the structure.
            set molID [mol new $filename type pdb waitfor all]

         # Otherwise let VMD load the file.
         } else {

            # Add the structure.
            set molID [mol new $filename waitfor all]
         }
      }
   }

# -------------------------------------------------------------------------
   proc importFromBlast opt {

      variable w

      upvar $opt options

#      puts "multiseq.tcl.importFromBlast.start. options [array get options]"

      # Calculate the e value.
      if {$options(eScore) < 0} {
         set eScore [format "%0.[expr -$options(eScore)]f" [expr pow(10,$options(eScore))]]
      } else {
         set eScore [format "%0.f" [expr pow(10,$options(eScore))]]
      }

      # See if we are performing a search across whole sequences.
      set blastResults {}
      set selectedResults {}
      if {$options(profileType) == "all" || \
                                   $options(profileType) == "marked"} {

         # Get the sequences.
         if {$options(profileType) == "all"} {
            set sequenceIDs [::SeqEditWidget::getSequences]
         } elseif {$options(profileType) == "marked"} {
            set sequenceIDs [::SeqEditWidget::getMarkedSequences]
         }

         # If doing profile search, make sure the sequences are aligned.
         if {[llength $sequenceIDs] > 1} {
            foreach sequenceID $sequenceIDs {
               if {[::SeqData::getSeqLength $sequenceID] != [::SeqData::getSeqLength [lindex $sequenceIDs 0]]} {

                  puts -nonewline "MultiSeq error) Seq lengths: "
                  foreach sID $sequenceIDs {
                     puts -nonewline "[::SeqData::getSeqLength $sID], "
                  }
                  puts "."

                  tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The sequences must be aligned before they can be used in a BLAST profile search."
                  return
               }
            }
         }

         # Perform the search.
         if {[catch {
            set blastResults [::Blast::searchDatabase $sequenceIDs $options(database) $eScore $options(iterations) $options(maxResults)]
#            puts "multiseq.tcl.importFromBlast, results: $blastResults"
         } errorMessage] != 0} {
            printStackTrace
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The BLAST search failed with the following message:\n\n$errorMessage\n\nPlease ensure the local BLAST installation is configured correctly and that its location has been entered into the MultiSeq preferences dialog."
            return
         }

         if {$blastResults != {}} {         
            set selectedResults [::Blast::ResultViewer::showBlastResultViewerDialog $w [lindex $sequenceIDs 0] $blastResults $options(eScore)]
         } else {
            tk_messageBox -type ok -icon info -parent $w -title "No Results Found" -message "The BLAST search did not find any matches."
            return
         }


      # See if we are performing a search across a residue selection.
      } elseif {$options(profileType) == "selected"} {

         # Get the selected cells.
         set selected [::SeqEditWidget::getSelectedCells]

         # Go through each sequence that has something selected.
         set selectionLength -1
         set selectionSequenceIDs {}
         for {set i 0} {$i < [llength $selected]} {incr i 2} {

            # Get the sequence id and elements of the selection.
            set selectedSequenceID [lindex $selected $i]
            set selectedElements [lindex $selected [expr $i+1]]

            # Make sure that the selection is of the correct length.
            if {$selectionLength != -1 && [llength $selectedElements] != $selectionLength} {
               tk_messageBox -type ok -icon error -parent $w -title "Error" -message "A BLAST profile search can only be performed with a selection that is of the same length in each sequence."
               return
            }
            set selectionLength [llength $selectedElements]

            # Make sure that the selection is continguous.
            set previousElement -1
            foreach element $selectedElements {
               if {$previousElement != -1 && $element != [expr $previousElement+1]} {
                  tk_messageBox -type ok -icon error -parent $w -title "Error" -message "A BLAST profile search can only be performed with a selection that is contiguous."
                  return
               }
               set previousElement $element
            }

            # Create a new sequence from the selection for use in the BLAST search.
            lappend selectionSequenceIDs [::SeqData::duplicateSequence $selectedSequenceID [lindex $selectedElements 0] [lindex $selectedElements end]]
         }

         # Perform the search.
         if {[catch {
            set blastResults [::Blast::searchDatabase $selectionSequenceIDs $options(database) $eScore $options(iterations) $options(maxResults)]
         } errorMessage] != 0} {
            printStackTrace
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The BLAST search failed with the following message:\n\n$errorMessage"
            return
         }

         if {$blastResults != {}} {         
            set selectedResults [::Blast::ResultViewer::showBlastResultViewerDialog $w [lindex $selectionSequenceIDs 0] $blastResults $options(eScore)]
         } else {
            tk_messageBox -type ok -icon info -parent $w -title "No Results Found" -message "The BLAST search did not find any matches."
            return
         }
      }

      # Add the result sequences to the editor.
      if {$selectedResults != {}} {         
         ::SeqEditWidget::addSequences $selectedResults "BLAST Results"
         if {$options(loadStructures)} {
            loadStructuresForSequences $selectedResults
         }
      }
   }

# -------------------------------------------------------------------------
   # This method allows the user to add new sequences to the editor.
   proc menu_importData {} {

      variable w

      array set options [::MultiSeq::Import::showImportOptionsDialog $w]
#      puts "multiseq.tcl.menu_importData. [array get options]"
      if {[array size options] > 0} {

         # See if we are importing from files.
         if {$options(dataSource) == "file" && $options(filenames) != {}} {
            importFromFile options

         # See if we are importing from blast.
         } elseif {$options(dataSource) == "blast"} {
            importFromBlast options

         } 
      }

#      puts "Done importing."
   }

# ------------------------------------------------------------------------
    # This method allows the user to add new sequences to the editor.
    proc menu_exportData {} {

        variable w
        variable tempDir
        variable filePrefix
        variable tempFiles

        array set options [::MultiSeq::Export::showExportOptionsDialog $w]
        if {[array size options] > 0 && $options(filename) != ""} {

            # Get the filename.
            set filename $options(filename)

            # See if we are exporting whole sequences.
            if {$options(dataSource) == "all" || $options(dataSource) == "marked"} {

                # Get the sequences.
                if {$options(dataSource) == "all"} {
                    set sequenceIDs [::SeqEditWidget::getSequences]
                } elseif {$options(dataSource) == "marked"} {
                    set sequenceIDs [::SeqEditWidget::getMarkedSequences]
                }

                # See what we are exporting.
                if {$options(dataType) == "fasta"} {
                    ::SeqData::Fasta::saveSequences $sequenceIDs $filename {} {} $options(fastaIncludeSources)
                } elseif {$options(dataType) == "aln"} {
                    ::SeqData::Aln::saveSequences $sequenceIDs $filename 
                } elseif {$options(dataType) == "nex"} {
                    ::SeqData::Nex::saveSequences $sequenceIDs $filename 
                } elseif {$options(dataType) == "pir"} {
                    ::SeqData::Pir::saveSequences $sequenceIDs $filename 
                } elseif {$options(dataType) == "phy"} {
                    ::SeqData::Phy::saveSequences $sequenceIDs $filename 
                } elseif {$options(dataType) == "secondary"} {
                    ::SeqData::Fasta::saveSecondaryStructure $sequenceIDs $filename                    
                } elseif {$options(dataType) == "pdb"} {

                    # Get the structures.
                    set structureIDs {}
                    foreach sequenceID $sequenceIDs {
                        if {[::SeqData::hasStruct $sequenceID] == "Y"} {
                            lappend structureIDs $sequenceID
                        }
                    }

                    # Write out the structure files.
                    if {[llength $structureIDs] == 1} {
                        ::SeqData::VMD::writeStructure [lindex $structureIDs 0] $filename "all" "all" $options(pdbCopyUser)
                    } elseif {[llength $structureIDs] > 1} {
                        foreach structureID $structureIDs {
                            ::SeqData::VMD::writeStructure $structureID "$filename[::SeqData::getName $structureID].pdb" "all" "all" $options(pdbCopyUser)
                        }
                    }

                } elseif {$options(dataType) == "coloring"} {

                    # Get the function to map a value to an index.
                    set colorIndexMap "[::SeqEditWidget::getColorMap]\::getColorValueForIndex"

                    # Save the metric data.
                    set fp [open $filename "w"]
                    foreach sequenceID $sequenceIDs {

                        # Write the sequence name.
                        set sequenceName [::SeqData::getName $sequenceID]
                        regsub -all {\s} $sequenceName _ sequenceName
                        puts -nonewline $fp $sequenceName

                        # Write the sequence metric values.
                        set numberElements [SeqData::getSeqLength $sequenceID] 
                        for {set i 0} {$i < $numberElements} {incr i} {
                            set colorValue [$colorIndexMap [seq get color $sequenceID $i]]
                            puts -nonewline $fp " $colorValue"
                        }
                        puts $fp ""
                    }
                    close $fp

                }

            # See if we are exporting across a residue selection.
            } elseif {$options(dataSource) == "selected"} {

                # Get the selected cells.
                set selected [::SeqEditWidget::getSelectedCells]
#                puts "selected cells: $selected"

                # See what we are exporting.
                if {$options(dataType) == "fasta" || $options(dataType) == "aln" || $options(dataType) == "nex" || $options(dataType) == "pir" || $options(dataType) == "phy" || $options(dataType) == "secondary"} {

                    # Make the new partial sequences.
                    set selectionLength -1
                    set selectionSequenceIDs {}
                    for {set i 0} {$i < [llength $selected]} {incr i 2} {

                        # Get the sequence id and elements of the selection.
                        set selectedSequenceID [lindex $selected $i]
                        set selectedElements [lindex $selected [expr $i+1]]

                        # Create a new sequence from the selection.
                        lappend selectionSequenceIDs [::SeqData::duplicateSequence $selectedSequenceID [lindex $selectedElements 0] [lindex $selectedElements end]]
                    }

                    # Export the data.
                    if {$options(dataType) == "fasta"} {
                        ::SeqData::Fasta::saveSequences $selectionSequenceIDs $filename {} {} $options(fastaIncludeSources)
                    } elseif {$options(dataType) == "aln"} {
                        ::SeqData::Aln::saveSequences $selectionSequenceIDs $filename    
                    } elseif {$options(dataType) == "nex"} {
                        ::SeqData::Nex::saveSequences $selectionSequenceIDs $filename
                    } elseif {$options(dataType) == "pir"} {
                        ::SeqData::Pir::saveSequences $selectionSequenceIDs $filename
                    } elseif {$options(dataType) == "phy"} {
                        ::SeqData::Phy::saveSequences $selectionSequenceIDs $filename
                    } elseif {$options(dataType) == "secondary"} {
                        ::SeqData::Fasta::saveSecondaryStructure $selectionSequenceIDs $filename
                    }

                } elseif {$options(dataType) == "pdb"} {

                    # Get the structures.
                    set structureSelection {}
                    for {set i 0} {$i < [llength $selected]} {incr i 2} {

                        # Get the sequence id and elements of the selection.
                        set selectedSequenceID [lindex $selected $i]
                        set selectedElements [lindex $selected [expr $i+1]]
                        if {[::SeqData::hasStruct $selectedSequenceID] == "Y"} {
                            lappend structureSelection $selectedSequenceID
                            lappend structureSelection $selectedElements
                        }
                    }

                    # Write out the structure selections.
                    if {[llength $structureSelection] == 2} {
                        set selectedStructureID [lindex $structureSelection 0]
                        set selectedElements [lindex $structureSelection 1]
                        ::SeqData::VMD::writeStructure $selectedStructureID $filename "all" $selectedElements $options(pdbCopyUser)
                    } elseif {[llength $structureSelection] > 2} {

                        for {set i 0} {$i < [llength $structureSelection]} {incr i 2} {
                            set selectedStructureID [lindex $structureSelection $i]
                            set selectedElements [lindex $structureSelection [expr $i+1]]
                            ::SeqData::VMD::writeStructure $selectedStructureID "$filename[::SeqData::getName $selectedStructureID].pdb" "all" $selectedElements $options(pdbCopyUser)
                        }
                    }


                } elseif {$options(dataType) == "coloring"} {

                    # Get the function to map a value to an index.
                    set colorIndexMap "[::SeqEditWidget::getColorMap]\::getColorValueForIndex"

                    # Save the metric data.
                    set fp [open $filename "w"]

                    # Go through each sequence that has something selected.
                    for {set i 0} {$i < [llength $selected]} {incr i 2} {

                        # Get the sequence id and elements of the selection.
                        set selectedSequenceID [lindex $selected $i]
                        set selectedElements [lindex $selected [expr $i+1]]

                        # Write the sequence name.
                        set sequenceName [::SeqData::getName $selectedSequenceID]
                        regsub -all {\s} $sequenceName _ sequenceName
                        puts -nonewline $fp $sequenceName

                        # Write the sequence metric values.
                        foreach selectedElement $selectedElements {
                            set colorValue [$colorIndexMap [seq get color $selectedSequenceID $selectedElement]]
                            puts -nonewline $fp " $colorValue"
                        }
                        puts $fp ""
                    }
                    close $fp
                }                
            }                
        }
    }

# ----------------------------------------------------------------------
    proc loadStructuresForSequences {sequenceIDs} {

        variable tempDir
        variable tempFiles

        # Go through the sequences and if we can identify the source, load the corresponding structure.
        foreach sequenceID $sequenceIDs {

            # If it has a PDB source, load it. 
            set msData [::SeqData::getSourceData $sequenceID "pdb"]
            if {[llength $msData] == 2} {

                # Register the sequence that the structure should be correlated to.
                ::SeqData::VMD::registerSequenceForVMDStructure [lindex $msData 0] [lindex $msData 1] $sequenceID

                # Let VMD load the structure from the PDB website.
                set molID [mol pdbload [lindex $msData 0]]

            }

            # If it has a SCOP source, load it. 
            set msData [::SeqData::getSourceData $sequenceID "scop"]
            if {[llength $msData] == 1} {

                # Download the file.
                set structureFilename "$tempDir/[lindex $msData 0]"
                ::SeqData::Astral::saveStructure [lindex $msData 0] $structureFilename
                lappend tempFiles $structureFilename

                # Register the sequence that the structure should be correlated to.
                ::SeqData::VMD::registerSequenceForVMDStructure [lindex $msData 0] "*" $sequenceID

                # Add the structure.
                set molID [mol new $structureFilename waitfor all]
            }
        }
    }

# -----------------------------------------------------------------------
    # Displays a dialog to choose the directory for temporary MultiSeq files.
    proc menu_chooseTempDir {} {

        variable tempDir
        variable filePrefix

        if {[info exists tempDir] == 0} {
           set tempDir "/"
        }

        set gooddir 0
        while { $gooddir == 0 } {
          set dir [tk_chooseDirectory -mustexist true -title "Choose Temp Directory" -initialdir "$tempDir"]
          if { $dir == "" } {
            return
          }
          if { [file writable "$dir"] } {
            set gooddir 1
          } else {
            tk_messageBox -type ok -message "Error: $dir is not writable"
          }
        }

        # Set the temp directory options for those packages that need it.
        setTempFileOptions $dir $filePrefix       
    }

# -------------------------------------------------------------------------
    proc menu_removeGaps {} {

        variable w

        array set options [::SeqEdit::RemoveGaps::showOptionsDialog $w]
        if {[array size options] > 0} {

            # Get the options.
            set removalType $options(removalType)

            # Get the sequence ids.
            set sequenceIDs {}
            if {$options(selectionType) == "all"} {

                set sequenceIDs [::SeqEditWidget::getSequences]
                set firstElement 0
                set lastElement [expr [::SeqEditWidget::getNumberPositions]-1]

            } elseif {$options(selectionType) == "marked"} {

                set sequenceIDs [::SeqEditWidget::getMarkedSequences]
                set firstElement 0
                set lastElement [expr [::SeqEditWidget::getNumberPositions]-1]

            } elseif {$options(selectionType) == "selected"} {

                # Get the selected cells.
                set selected [::SeqEditWidget::getSelectedCells]

                # Go through each sequence that has something selected.
                set selectionLength -1
                set selectionSequenceIDs {}
                set firstElement -1
                set lastElement -1
                for {set i 0} {$i < [llength $selected]} {incr i 2} {

                    # Get the sequence id and elements of the selection.
                    set selectedSequenceID [lindex $selected $i]
                    set selectedElements [lindex $selected [expr $i+1]]

                    # Make sure that the selection is of the correct length.
                    if {($selectionLength != -1 && [llength $selectedElements] != $selectionLength) ||
                        ($firstElement != -1 && [lindex $selectedElements 0] != $firstElement) ||
                        ($lastElement != -1 && [lindex $selectedElements end] != $lastElement)} {
                        tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Gaps can only be removed a contiguous selection area."
                        return
                    }
                    set selectionLength [llength $selectedElements]

                    # Make sure that the selection is contiguous.
                    set previousElement -1
                    foreach element $selectedElements {
                        if {$previousElement != -1 && $element != [expr $previousElement+1]} {
                            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Gaps can only be removed a contiguous selection area."
                            return
                        }
                        set previousElement $element
                    }

                    # Create new sequences from the selection.
                    lappend sequenceIDs $selectedSequenceID
                    if {$firstElement == -1 && $lastElement == -1} {
                        set firstElement [lindex $selectedElements 0]
                        set lastElement [lindex $selectedElements end]
                    }
                }
            }

            # Initialize the lists of elements to remove.
            array unset elementsToRemove 
#            foreach sequenceID $sequenceIDs {
#                set elementsToRemove($sequenceID) {}
#            }

            # Figure out which gaps to remove.
            for {set i $firstElement} {$i <= $lastElement} {incr i} {

                # See if we should remove this gap.
                set removeGap 0
                if {$removalType == "redundant"} {
                    set removeGap 1
                    foreach sequenceID $sequenceIDs {
                        if {$i < [::SeqData::getSeqLength $sequenceID] && [::SeqData::getElements $sequenceID $i] != "-"} {
                            set removeGap 0
                        }
                    }
                } elseif {$removalType == "all"} {
                    set removeGap 1
                }

                # Get the elements to remove.
                if {$removeGap} {
                    foreach sequenceID $sequenceIDs {
                        if {$i < [::SeqData::getSeqLength $sequenceID] && [::SeqData::getElements $sequenceID $i] == "-"} {
                            lappend elementsToRemove($sequenceID) $i
                        }
                    }
                }   
            }

#            puts "multiseq.tcl.menu_removeGaps.seqIDs: $sequenceIDs, toRemove: [array get elementsToRemove]"
            # Remove the gaps.
            foreach sequenceID $sequenceIDs {
               if { [info exists elementsToRemove($sequenceID)] } {
                  set oldSequenceLength [::SeqData::getSeqLength $sequenceID]
                  ::SeqData::removeElements $sequenceID $elementsToRemove($sequenceID)
# foobar.  next call doesn't seem to actually do anything anymore
#                  ::SeqEditWidget::updateColorMap $sequenceID $oldSequenceLength [::SeqData::getSeqLength $sequenceID] "delete" $elementsToRemove($sequenceID)
               }
            }

            # Redraw the editor.
            ::SeqEditWidget::redraw 
            ::SeqEditWidget::notifySelectionChangeListeners
        }
    }

# ------------------------------------------------------------------------
    proc testTempFileSelection {a_tempDir {verbose 1}} {
       variable isOK 1
        if { $a_tempDir == "" } {
           set msg "No work directory has been specified.  You will need to set one via File | Choose Work Directory.  (set a TMPDIR environment variable for future invocations)."
           puts "MultiSeq Error) $msg"
           if {$verbose} {
              tk_messageBox -type ok -title "Error" -message "$msg"
           }
           set isOK 0
        }

        if { ! [file writable "$a_tempDir"] } {
           set msg "Work directory given \"$a_tempDir\" is not writeable.  You will need to choose a writeable directory via File | Choose Work Directory."
           puts "MultiSeq Error) $msg"
           if {$verbose} {
              tk_messageBox -type ok -title "Error" -message "$msg"
           }
           set isOK 0
        }
        return $isOK
    }


# ------------------------------------------------------------------------
    proc setTempFileOptions {a_tempDir a_filePrefix} {
        variable tempDir

        variable filePrefix
        set filePrefix $a_filePrefix

        if {[testTempFileSelection $a_tempDir]} {

           # Save the new options.
           set tempDir $a_tempDir

           puts "MultiSeq Info) Setting new temp options to $tempDir $filePrefix"

           # Set the options for those packages that need them.        
           ::Blast::setTempFileOptions $tempDir $filePrefix
           ::Blast::setArchitecture [vmdinfo arch]
           ::ClustalW::setTempFileOptions $tempDir $filePrefix       
           ::ClustalW::setArchitecture [vmdinfo arch]
           ::UPGMA_Cluster::setTempFileOptions $tempDir $filePrefix       
           ::Libbiokit::setTempFileOptions $tempDir $filePrefix       
           ::MultiSeqDialog::setTempFileOptions $tempDir $filePrefix
           ::MultiSeqDialog::setArchitecture [vmdinfo arch]        
           ::Psipred::setTempFileOptions $tempDir $filePrefix
           ::Psipred::setArchitecture [vmdinfo arch]        
           ::STAMP::setTempFileOptions $tempDir $filePrefix
           ::STAMP::setArchitecture [vmdinfo arch]
           #::SimpleEdit::setFilename $tempDir $filePrefix
           ::MultiSeq::textview_setFilename $tempDir $filePrefix
           ::Mafft::setTempFileOptions $tempDir $filePrefix
           ::Mafft::setArchitecture [vmdinfo arch]
       }
    }

# -------------------------------------------------------------------------
    proc setPreferences {} {
        global env

        ::Blast::setBlastProgramDirs [::MultiSeqDialog::getDirectory "blast"] [::MultiSeqDialog::getVariable "blast" "BLASTMAT"] [::MultiSeqDialog::getVariable "blast" "BLASTDB"]
        ::Psipred::setPackageOptions [::MultiSeqDialog::getDirectory "psipred"] [::MultiSeqDialog::getVariable "psipred" "PSIPREDDATA"] [::MultiSeqDialog::getVariable "psipred" "PSIPREDDB"]
        ::Mafft::setMafftProgramDirs [::MultiSeqDialog::getDirectory "mafft"] 

        if { [::MultiSeqDialog::getDirectory "editor"] != "" } {
          set env(EDITOR) [::MultiSeqDialog::getDirectory "editor"]
        }
    }

# ------------------------------------------------------------------------
    proc menu_cleanupReps {} {

        variable representations

        # Delete any representations we created.
        foreach key [array names representations "*,sequence"] {
            if {[regexp {^(\d+),sequence$} $key unused sequenceID] == 1} {
                set molID [lindex [::SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                foreach repName $representations($key) {
                    set repIndex [mol repindex $molID $repName]
                    if {$repIndex != -1} {
                        mol delrep $repIndex $molID
                    }
                }
            }
        }
        foreach key [array names representations "*,selection"] {
            if {[regexp {^(\d+),selection$} $key unused sequenceID] == 1} {
                set molID [lindex [::SeqData::VMD::getMolIDForSequence $sequenceID] 0]
                set repName $representations($key)
                set repIndex [mol repindex $molID $repName]
                if {$repIndex != -1} {
                    mol delrep $repIndex $molID
                }
            }
        }
    }

# --------------------------------------------------------------------------
# helper procedure that won't be called by anything by selectResidues
# uses upvar for the array as a convenience
   proc setSelCellsForSelectResidues { val calcProc identOp selCellArray \
                                                              sequenceIDs} {
      upvar 1 $selCellArray selectedCells

#      puts "setSelCellsForSelectResidues. val:$val, calc<$calcProc>, seqIDs:$sequenceIDs"

      set sequenceID ""
      set elementIndex ""

      if {[regexp {[0-9.]+} $val] && $val >= 0 && $val <= 1} {
#      puts "setSelCellsForSelectResidues. inside test"

         # Get the map.
         array set map [$calcProc $sequenceIDs]
#         parray map

         # Go through each element in the map.
         set elements [array names map]
         foreach element $elements {

            # Get the value, sequence id, and element.
            set sequenceID [lindex [split $element ","] 0]
            set elementIndex [lindex [split $element ","] 1]
            set elementValue $map($element)

            # If the value is within our search range, select it.
            if {[::SeqData::getElements $sequenceID $elementIndex] != "-" \
                                && (($identOp == ">=" && \
                                     $elementValue >= $val) || \
                                    ($identOp == "<=" && \
                                     $elementValue <= $val))} {

               # If the cell doesn't exist yet, add it.
               if {![info exists selectedCells($sequenceID,$elementIndex)]} {
                  set selectedCells($sequenceID,$elementIndex) 1
               }

               # Otherwise it is out of range so set it as not selected.
            } else {
               set selectedCells($sequenceID,$elementIndex) 0
            }
         }   
      }
   } ; # end of setSelCellsForSelectResidues

# --------------------------------------------------------------------------
   # This method allows the user to select a set of residues.
   proc selectResidues {arg_options} {
      variable w
#      puts "selectRes: <[::SeqEditWidget::getSelectedCells]>"

      # Get the selection options.
      array set options $arg_options

#      puts "told to select residues:"
#      parray options

      if {[array size options] > 0} {

         # Reset the current selection.
         ::SeqEditWidget::resetSelection

         # See which sequences we are performing the selection on.
         if {$options(selectionType) == "marked"} {        
            set sequenceIDs [::SeqEditWidget::getMarkedSequences]
         } else {
            set sequenceIDs [::SeqEditWidget::getSequences]
         }

         array unset selectedCells 

         set val $options(identity)
         if {[regexp {[0-9.]+} $val] && $val >= 0 && $val <= 100} {
            set val [expr $val/100.0]
         }
         set calcProc "::SeqEdit::Metric::PercentIdentity::calculate"
         setSelCellsForSelectResidues $val $calcProc \
                          $options(identityOperator)  selectedCells $sequenceIDs

         set val $options(qscore)
         set calcProc "::Libbiokit::Metric::calculateQres"
         setSelCellsForSelectResidues $val $calcProc \
                          $options(identityOperator)  selectedCells $sequenceIDs

#         puts "selectedCells is:"
#         parray selectedCells

         # Add the selected cells to the selection.
         set sequenceID ""
         set elementIndex ""
         foreach element [array names selectedCells] {
            if {$selectedCells($element) == 1} {
               set sequenceID [lindex [split $element ","] 0]
               set elementIndex [lindex [split $element ","] 1]                
#               puts "setting $sequenceID $elementIndex"
               ::SeqEditWidget::setSelectedCell $sequenceID \
                                                        $elementIndex 1 0 0 0
#               puts -nonewline [::SeqEditWidget::getSelectedCells]
            }
         }
#         puts "selectRes after setSelCells: <[::SeqEditWidget::getSelectedCells]>"

         # Make sure that last cell selected is visible, 
         # redraw, and notify any listeners.
#         if {$sequenceID != "" && $elementIndex != ""} {
#            ::SeqEditWidget::ensureCellIsVisible $sequenceID $elementIndex
#         }
         ::SeqEditWidget::redraw
         ::SeqEditWidget::notifySelectionChangeListeners
      } ; # end if on actually having options sent in
#      puts "selectRes at end: <[::SeqEditWidget::getSelectedCells]>"
   } ; # end of selectResidues 

# ------------------------------------------------------------------------
    proc menu_find { type } {
        variable w
        variable findSelectionsActive 
        variable findSelectionsPosition
        variable findSelections
        variable numFindSelections
        variable inSetNewFind

        if {$type == "new"} {
            # Get the find options.
            array set options [::MultiSeq::Find::showFindDialog $w]

            if {[array size options] > 0} {

              # See which sequences we are finding within.
              if {$options(selectionType) == "marked"} {        
                  set markedSequenceIDs [::SeqEditWidget::getMarkedSequences]
              } else {
                  set markedSequenceIDs [::SeqEditWidget::getSequences]
              }

              # Clear selections
              ::SeqEditWidget::resetSelection

              # Clear find variables
              set findSelectionsActive 0
              set findSelectionsPosition 0
              set numFindSelections 0
              array unset findSelections 

              # Lock out the selection change listener for finding
              set inSetNewFind 1

              # Iterate through all the requested sequences, find
              # the search string, and highlight all occurences.

              # First split the search string into its own list
              set searchList [split [string toupper $options(searchString)] ""]

              set selIdx 0
              foreach seq $markedSequenceIDs {
                set sequence [::SeqData::getSeq $seq]

                # Iterate through elements
                for {set i 0} {$i < [llength $sequence]} {incr i} {
                  # Check and see if we start to match here
                  if {[lindex $sequence $i] == [lindex $searchList 0]} {
                    # We do, so check the consecutive elements of sequence
                    # for matches to searchList
                    set searchMatch 1
                    for {set j 1} {$j < [llength $searchList]} {incr j} {
                      if {[lindex $sequence [expr $i+$j]] !=
                          [lindex $searchList $j]} {
                        set searchMatch 0
                        break
                      } else {
                        set searchMatch 1
                      }
                    }
                    # See if we matched the whole thing, if we did, add a 
                    # selection to our internal array.
                    if {$searchMatch == 1} {
                      set findSelectionsActive 1
                      set findSelections($selIdx,seq) $seq
                      set findSelections($selIdx,startPos) $i
                      set findSelections($selIdx,endPos) [expr $i+[llength $searchList]]
                      incr selIdx
                    }
                  }
                }
              }
              # Check and see if we matched anything.
              if {$findSelectionsActive == 0} {
                # No, we didn't!
                tk_messageBox -type ok -icon info -message "No matches to your search" -parent $w
              } else {
                # Select all matching cells.
                set numFindSelections $selIdx
                for {set i 0} {$i < $numFindSelections} {incr i} {
                  for {set resIdx $findSelections($i,startPos)} \
                    {$resIdx < $findSelections($i,endPos)} \
                    {incr resIdx} {
                    ::SeqEditWidget::setSelectedCell $findSelections($i,seq) $resIdx 1 0 0 0
                  }
                }
                # Selections have all been made, redraw and notify 
                # selection change listners
                ::SeqEditWidget::redraw
                ::SeqEditWidget::notifySelectionChangeListeners
                ::SeqEditWidget::ensureCellIsVisible $findSelections(0,seq) $findSelections(0,startPos)
                # Clear lock on local change listener
                set inSetNewFind 0
              }
            }
         } elseif {$type == "next"} {
            # Do we have an active find session?
            if {$findSelectionsActive == 0} {
              # No, we don't.
              tk_messageBox -type ok -icon info -message "No previous find to continue! Before using \"Find next\", you must have an active find session started with the \"Find...\" command" -parent $w
            } else {
              # Just re-center the display onto the next selection, and incr
              # the position counter
              if {$findSelectionsPosition < [expr $numFindSelections-1]} {
                incr findSelectionsPosition
                ::SeqEditWidget::ensureCellIsVisible $findSelections($findSelectionsPosition,seq) $findSelections($findSelectionsPosition,endPos)
              } else {
                # We're at the end already, go back to the beginning.
                set findSelectionsPosition 0
                ::SeqEditWidget::ensureCellIsVisible $findSelections($findSelectionsPosition,seq) $findSelections($findSelectionsPosition,startPos)
              }
            }
         } elseif {$type == "prev"} {
            # Do we have an active find session?
            if {$findSelectionsActive == 0} {
              # No, we don't.
              tk_messageBox -type ok -icon info -message "No previous find to continue! Before using \"Find previous\", you must have an active find session started with the \"Find...\" command" -parent $w
            } else {
              if {$findSelectionsPosition == 0} {
                # We're at the beginning already, go back to the end.
                set findSelectionsPosition [expr $numFindSelections-1]
                ::SeqEditWidget::ensureCellIsVisible $findSelections($findSelectionsPosition,seq) $findSelections($findSelectionsPosition,endPos)
              } else {
                # Just re-center the display onto the next selection, and decr
                # the position counter
                incr findSelectionsPosition -1
                ::SeqEditWidget::ensureCellIsVisible $findSelections($findSelectionsPosition,seq) $findSelections($findSelectionsPosition,startPos)
              }
            }
        } else {
          puts "MultiSeq Error) menu_find: unknown type"
        }
    }

# --------------------------------------------------------------------------
   proc menu_selectNRSet {} {
      variable w

      # Get the selection options.
      array set options [::MultiSeq::SelectNRSet::showSelectNRSetDialog $w]
      if {[array size options] > 0} {
         # See which sequences we are performing the selection on.
         if {$options(selectionType) == "marked"} {
            set sequenceIDs [::SeqEditWidget::getMarkedSequences]
         } else {
            set sequenceIDs [::SeqEditWidget::getSequences]
         }

         # See if we are seeding with the selected sequences.
         set numberSequencesToPreserve 0
         if {$options(seedWithSelected)} {

            # Get the selected sequences.
            set selectedSequenceIDs [::SeqEditWidget::getSelectedSequences]
            if {$selectedSequenceIDs != {}} {

               # Remove any of the selected sequences from the list.
               foreach selectedSequenceID $selectedSequenceIDs {
                  set index [lsearch $sequenceIDs $selectedSequenceID]
                  if {$index != -1} {
                     set sequenceIDs [lreplace $sequenceIDs $index $index]
                  }
               }

               # Make a new list with the selected sequences at the top.
               set sequenceIDs [concat $selectedSequenceIDs $sequenceIDs]
               set numberSequencesToPreserve [llength $selectedSequenceIDs]
            }
         }

         set type 0
         # See which method we are using.
         if {$options(method) == "sequence"} {
            # What if multiple types?
            set sequenceTypes ""
            set i 0
#            puts "multiseq.menu_selectNRSet.seqs are [::SeqEditWidget::getSequences]"
            foreach sequence $sequenceIDs {
               if { [::SeqData::VMD::determineTypeFromSeq $sequence] != "" } {
                  ::SeqData::setType $sequence \
                               [::SeqData::VMD::determineTypeFromSeq $sequence]
                  lappend sequenceTypes $sequence
                  lappend sequenceTypes [::SeqData::getType $sequence]
               }
            }
            if { [llength $sequenceTypes ] > 0 } {
               set firstType [::SeqData::getType [lindex $sequenceTypes 0]]
               set i 0
               foreach {sequenceID sequenceType} $sequenceTypes {
                  if { $sequenceType != $firstType } {
                     set message "The sequences could not be aligned\n"
                     append message "because there is more than one type in the alignment\n"
                     append message "Only sequences of the same type (RNA, DNA or Protein) may be aligned.\n\n"
                     append message "Sequence [::SeqData::getName 0] is of type $firstType\n"
                     append message "Sequence [::SeqData::getName $sequenceID] is of type $sequenceType\n"
                     tk_messageBox -type ok -icon error -parent $w \
                                                -title "Error" -message $message
                     return
                  }
                  incr i
               }
            }

            set firstSeqType [::SeqData::VMD::determineTypeFromSeq \
                                   [lindex [::SeqEditWidget::getSequences] 0]]
#            puts "firstSeqType = $firstSeqType"
            if { $firstSeqType == "protein" } {
               set type 0
            } elseif { $firstSeqType == "dna" } {
               set type 1
            } elseif { $firstSeqType == "rna" } {
               set type 1
            }
            # Get the nr set.
            if {[catch {
                  set cutoffType 0
                  set cutoff $options(identityCutoff)
                  if {$options(seqPercentCutoff) < 100} {
                     set cutoffType 1
                     set cutoff $options(seqPercentCutoff)
                  }
                  set nrSequenceIDs [::Libbiokit::getNonRedundantSequences \
                           $sequenceIDs $cutoffType $cutoff $options(gapScale) \
                           $numberSequencesToPreserve $type]
            } errorMessage] != 0} {
               printStackTrace
               tk_messageBox -type ok -icon error -parent $w -title "Error" \
                            -message "The redundancy search failed with the following message:\n\n$errorMessage"
               return
            }

            # Select the nr set.
            ::SeqEditWidget::setSelectedSequences $nrSequenceIDs

            # Show the ordering in the phylogenetic tree.
            showQROrdering $nrSequenceIDs

         # See which method we are using.
         } elseif {$options(method) == "structure"} {

            # Remove any sequences that do not have a structure.
            set structureIDs {}
            foreach sequenceID $sequenceIDs {
               if {[::SeqData::hasStruct $sequenceID] == "Y"} {
                  lappend structureIDs $sequenceID
               }
            }

            # Get the nr set.
            if {[catch {
               set nrStructureIDs [::Libbiokit::getNonRedundantStructures \
                                  $structureIDs $options(qhCutoff) \
                                  $numberSequencesToPreserve ]
            } errorMessage] != 0} {
               printStackTrace
               tk_messageBox -type ok -icon error -parent $w -title "Error" \
                         -message "The redundancy search failed with the following message:\n\n$errorMessage"
               return
            }

            # Select the nr set.
            ::SeqEditWidget::setSelectedSequences $nrStructureIDs

            # Show the ordering in the phylogenetic tree.
            showQROrdering $nrStructureIDs
         }
      }
   } ; # end of menu_selectNRSet 

# --------------------------------------------------------------------------
    proc showQROrdering {sequenceIDs} {

        variable treeNodeToSequenceIDMap

        # Get the active tree, if there is one.
        set treeID [::PhyloTree::getActiveTree]
        if {[::PhyloTree::windowExists] && $treeID != -1} {

            # Set the qr ordering of each node in the tree.
            set qrAttributeName "QR Ordering"
            set leafNodes [::PhyloTree::Data::getLeafNodes $treeID [::PhyloTree::Data::getTreeRootNode $treeID]]
            foreach node $leafNodes {

                if {![info exists treeNodeToSequenceIDMap($treeID,$node)] || [lsearch $sequenceIDs $treeNodeToSequenceIDMap($treeID,$node)] == -1} {
                    ::PhyloTree::Data::setNodeAttribute $treeID $node $qrAttributeName ""
                } else {
                    ::PhyloTree::Data::setNodeAttribute $treeID $node $qrAttributeName [expr [lsearch $sequenceIDs $treeNodeToSequenceIDMap($treeID,$node)]+1] {} 1
                }
            }

            # Redraw the tree.
            set ::PhyloTree::Widget::shownAttributes($qrAttributeName) 1
            ::PhyloTree::redraw
        }        
    }

# --------------------------------------------------------------------------
    proc menu_selectContactShells {} {

        variable w

        # Get the selection options.
        array set options [::MultiSeq::SelectContactShell::showSelectContactShellsDialog $w]
        if {[array size options] > 0} {

            # See which sequences we are performing the selection on.
            if {$options(selectionType) == "marked"} {        
                set markedSequenceIDs [::SeqEditWidget::getMarkedSequences]
            } else {
                set markedSequenceIDs [::SeqEditWidget::getSequences]
            }

            # Get the selected cells.
            set selected [::SeqEditWidget::getSelectedCells]

            # Reset the current selection.
            ::SeqEditWidget::resetSelection

            # Go through each sequences that has something selected.
            set lastSequenceID {}
            set lastElement {}
            for {set i 0} {$i < [llength $selected]} {incr i 2} {

                # Get the sequence id.
                set selectedSequenceID [lindex $selected $i]

                # Make sure we have a structure for it.
                if {[::SeqData::hasStruct $selectedSequenceID] == "Y"} {

                    # Get the mol id.
                    set selectedMolID [lindex [SeqData::VMD::getMolIDForSequence $selectedSequenceID] 0]

                    # Go through each element that is selected.
                    foreach element [lindex $selected [expr $i+1]] {

                        # Get the VMD residue id.
                        set selectedResidue [::SeqData::getResidueForElement $selectedSequenceID $element]
                        if {$selectedResidue != ""} {

                            # Go through each of the marked sequences.
                            set firstShell {}
                            foreach markedSequenceID $markedSequenceIDs {
                                if {[::SeqData::hasStruct $markedSequenceID] == "Y"} {

                                    # Get the marked sequence's mol id and chain.
                                    set markedMolID [lindex [::SeqData::VMD::getMolIDForSequence $markedSequenceID] 0]
                                    set markedChain [lindex [::SeqData::VMD::getMolIDForSequence $markedSequenceID] 1]
                                    set markedSegname [lindex [::SeqData::VMD::getMolIDForSequence $markedSequenceID] 2]

                                    # Find the residues in this sequence in contact with the selected residue.
                                    set selectionString [::SeqData::VMD::getSelectionStringForElements $selectedSequenceID $element]
                                    set selectedAtoms [atomselect $selectedMolID $selectionString]
                                    foreach atomPosition [$selectedAtoms get {x y z}] {
                                        set x1 [expr [lindex $atomPosition 0]-$options(contactDistance)]
                                        set x2 [expr [lindex $atomPosition 0]+$options(contactDistance)]
                                        set y1 [expr [lindex $atomPosition 1]-$options(contactDistance)]
                                        set y2 [expr [lindex $atomPosition 1]+$options(contactDistance)]
                                        set z1 [expr [lindex $atomPosition 2]-$options(contactDistance)]
                                        set z2 [expr [lindex $atomPosition 2]+$options(contactDistance)]
                                        set closeAtoms [atomselect $markedMolID "[::SeqData::VMD::getSelectionStringForSequence $markedSequenceID] and x >= $x1 and x <= $x2 and y >= $y1 and y <= $y2 and z >= $z1 and z <= $z2"]
                                        foreach closeAtomProperties [$closeAtoms get {x y z resid insertion altloc}] {
                                            set closePosition [lreplace $closeAtomProperties 3 end]
                                            set closeResidue [lreplace $closeAtomProperties 0 2]
                                            if {[vecdist $atomPosition $closePosition] <= $options(contactDistance) && [lsearch $firstShell "$markedMolID,$markedChain,$markedSegname,$closeResidue"] == -1 && ($markedMolID != $selectedMolID || [string trim [join $closeResidue ""]] != [string trim [join $selectedResidue ""]])} {
                                                lappend firstShell "$markedMolID,$markedChain,$markedSegname,$closeResidue"
                                            }
                                        }
                                        $closeAtoms delete
                                    }
                                    $selectedAtoms delete
                                }
                            }

                            # If we are selecting the first shell, select the residues.
                            if {$options(shell) == "first" || $options(shell) == "firstandsecond"} {
                                foreach item $firstShell {
                                    set molID [lindex [split $item ","] 0]
                                    set chain [lindex [split $item ","] 1]
                                    set segname [lindex [split $item ","] 2]
                                    set residue [lindex [split $item ","] 3]
                                    set lastSequenceIDs [::SeqData::VMD::getSequenceIDForMolecule $molID $chain $segname]
                                    foreach lastSequenceID $lastSequenceIDs {
                                        if {[::SeqEditWidget::containsSequence $lastSequenceID]} {
                                            set lastElement [::SeqData::VMD::getElementForResidue $lastSequenceID $residue]
                                            ::SeqEditWidget::setSelectedCell $lastSequenceID $lastElement 1 0 0 0
                                        }
                                    }
                                }
                            }

                            # See if we are looking for the second shell.
                            if {$options(shell) == "second" || $options(shell) == "firstandsecond"} {

                                # Go through each residue in the first shell and find their contacts.
                                set secondShell {}
                                foreach item $firstShell {

                                    set firstShellMolID [lindex [split $item ","] 0]
                                    set firstShellChain [lindex [split $item ","] 1]
                                    set firstShellSegname [lindex [split $item ","] 2]
                                    set firstShellResidue [lindex [split $item ","] 3]

                                    # Find the residues in contact with the first shell residue.
                                    foreach markedSequenceID $markedSequenceIDs {
                                        if {[::SeqData::hasStruct $markedSequenceID] == "Y"} {

                                            # Get the marked sequence's mol id and chain.
                                            set markedMolID [lindex [SeqData::VMD::getMolIDForSequence $markedSequenceID] 0]
                                            set markedChain [lindex [SeqData::VMD::getMolIDForSequence $markedSequenceID] 1]
                                            set markedSegname [lindex [SeqData::VMD::getMolIDForSequence $markedSequenceID] 2]

                                            # Find the residues in this sequence in contact with the first shell residue.
                                            set firstShellAtoms [atomselect $firstShellMolID "resid \"$firstShellResidue\""]
                                            foreach atomPosition [$firstShellAtoms get {x y z}] {
                                                set x1 [expr [lindex $atomPosition 0]-$options(contactDistance)]
                                                set x2 [expr [lindex $atomPosition 0]+$options(contactDistance)]
                                                set y1 [expr [lindex $atomPosition 1]-$options(contactDistance)]
                                                set y2 [expr [lindex $atomPosition 1]+$options(contactDistance)]
                                                set z1 [expr [lindex $atomPosition 2]-$options(contactDistance)]
                                                set z2 [expr [lindex $atomPosition 2]+$options(contactDistance)]
                                                set closeAtoms [atomselect $markedMolID "[::SeqData::VMD::getSelectionStringForSequence $markedSequenceID] and x >= $x1 and x <= $x2 and y >= $y1 and y <= $y2 and z >= $z1 and z <= $z2"]                                            
                                                foreach closeAtomProperties [$closeAtoms get {x y z resid insertion altloc}] {
                                                    set closePosition [lreplace $closeAtomProperties 3 end]
                                                    set closeResidue [lreplace $closeAtomProperties 0 2]
                                                    if {[vecdist $atomPosition $closePosition] <= $options(contactDistance) && [lsearch $secondShell "$markedMolID,$markedChain,$markedSegname,$closeResidue"] == -1 && [lsearch $firstShell "$markedMolID,$markedChain,$markedSegname,$closeResidue"] == -1 && ($markedMolID != $selectedMolID || [string trim [join $closeResidue ""]] != [string trim [join $selectedResidue ""]])} {
                                                        lappend secondShell "$markedMolID,$markedChain,$markedSegname,$closeResidue"
                                                    }
                                                }
                                                $closeAtoms delete
                                            }
                                            $firstShellAtoms delete
                                        }
                                    }
                                }

                                # Select the residues.
                                foreach item $secondShell {
                                    set molID [lindex [split $item ","] 0]
                                    set chain [lindex [split $item ","] 1]
                                    set segname [lindex [split $item ","] 2]
                                    set residue [lindex [split $item ","] 3]
                                    set lastSequenceIDs [::SeqData::VMD::getSequenceIDForMolecule $molID $chain $segname]
                                    foreach lastSequenceID $lastSequenceIDs {
                                        if {[::SeqEditWidget::containsSequence $lastSequenceID]} {
                                            set lastElement [::SeqData::VMD::getElementForResidue $lastSequenceID $residue]
                                            ::SeqEditWidget::setSelectedCell $lastSequenceID $lastElement 1 0 0 0
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            # Make sure that the last cell selected is visible, redraw, and notify any listeners.
            if {$lastSequenceID != "" && $lastElement != ""} {
                ::SeqEditWidget::ensureCellIsVisible $lastSequenceID $lastElement
            }
            ::SeqEditWidget::redraw
            ::SeqEditWidget::notifySelectionChangeListeners
        }
    }

# -----------------------------------------------------------------------
   # This method performs a structural alignment with stamp.
   proc menu_stamp {} {

      variable w

      # Get the Stamp options.
      array set options [Stamp::showStampOptionsDialog $w]
      if {[array size options] > 0} {

         # Get the stamp parameters.
         set npass $options(npass)
         set scanscore $options(scanscore)
         set scanslide $options(scanslide)
         set slowscan $options(slowscan)
         set scan $options(scan)

         # Extract the appropriate regions.
         set preservePrefixSuffixAlignment 0
         set extractedRegions {}
         if {$options(alignmentType) == "all" || $options(alignmentType) == "marked"} {

            # Get the sequences.
            if {$options(alignmentType) == "all"} {        
               set sequenceIDs [::SeqEditWidget::getSequences]
            } elseif {$options(alignmentType) == "marked"} {
               set sequenceIDs [::SeqEditWidget::getMarkedSequences]
            }
#            puts "multiseq.tcl.menu_stamp.seqs: $sequenceIDs"

            # Extract the region of the structure that are of the correct type.
            set extractedRegions [::SeqData::VMD::extractFirstRegionFromStructures $sequenceIDs "Y"]
            if {$extractedRegions == {}} {
               tk_messageBox -type ok -icon error -parent $w -title "Error" -message "One or more structures did not contain valid data."
               return
            }

            # Mark that we should ignore the rest of the alignment.
            set preservePrefixSuffixAlignment 0

         } elseif {$options(alignmentType) == "selected"} {

            # Extract the selected regions.
            set extractedRegions [::SeqData::extractRegionsFromSequences [::SeqEditWidget::getSelectedCells]]
            if {$extractedRegions == {}} {
               tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The selection within each sequence must be contiguous."
               return
            }

            # Mark that we should preserve the rest of the alignment.
            set preservePrefixSuffixAlignment 1
         }
# puts "multiseq.tcl.menu_stamp.  extractedReg: $extractedRegions"
         # Exract the region info.
         set originalSequenceIDs [lindex $extractedRegions 0]
         set regionSequenceIDs [lindex $extractedRegions 1]
         set prefixes [lindex $extractedRegions 2]
         set prefixEndPositions [lindex $extractedRegions 3]
         set suffixes [lindex $extractedRegions 4]
         set suffixStartPositions [lindex $extractedRegions 5]

         # Remove any sequences that do not have a structure.
#         set regionStructIDs {}
#         foreach sequenceID $regionSequenceIDs {
#             if {[::SeqData::hasStruct $sequenceID] == "Y"} {
#                 lappend regionStructIDs $sequenceID
#             }
#         }
#puts "regseqs: $regionSequenceIDs, regstructs: $regionStructIDs"
#         set originalStructIDs {}
#         foreach sequenceID $originalSequenceIDs {
#             if {[::SeqData::hasStruct $sequenceID] == "Y"} {
#                 lappend originalStructIDs $sequenceID
#             }
#         }
#puts "origseqs: $originalSequenceIDs, origstructs: $originalStructIDs"

         # Make sure all of the structures are of the correct type.

         # Make sure all of the residues have an atom for stamp.

         # Align the structures.
         if {[catch {
#            set alignedSequenceIDs [::STAMP::alignStructures $regionSequenceIDs $scan $scanslide $scanscore $slowscan $npass]
            set alignedSequenceIDs [::STAMP::alignStructures $regionSequenceIDs $scan $scanslide $scanscore $slowscan $npass]
         } errorMessage] != 0} {
            printStackTrace
            set ll [llength [split $errorMessage "\n"]]
            if { $ll < 25 } {
               tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The STAMP alignment failed with the following error message:\n$errorMessage."
            } else {
               set ans [tk_messageBox -type yesno -icon error -parent $w -title "Error" -message "The STAMP alignment failed.  To see the complete error message ($ll lines), select \"Yes\".  Otherwise, select \"No\""]
               if { $ans == "yes" } {
                  set errorTextWindow [multitext]
                  $errorTextWindow text $errorMessage
               }
            }
            return
         }

         # Put the sequences back together again.
         set originalSequenceIDs [concatenateRegionsIntoSequences $originalSequenceIDs $alignedSequenceIDs $prefixes $prefixEndPositions $suffixes $suffixStartPositions $preservePrefixSuffixAlignment]
         if {$originalSequenceIDs == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Concatenation of the alignment data failed. The data may be in an inconsistent state."
            return
         }

         # Update the sequences in the editor.
         ::SeqEditWidget::updateSequences $originalSequenceIDs

         # Apply the transformation to the VMD molecules.
         applyStampTransformations $originalSequenceIDs
      }
   } ; # end of menu_stamp

# -----------------------------------------------------------------------
   proc concatenateRegionsIntoSequences {originalSequenceIDs \
                             regionSequenceIDs prefixes prefixEndPositions \
                             suffixes suffixStartPositions \
                             preservePrefixSuffixAlignment} {
#      puts "multiseq.tcl.concatenateRegionsIntoSequences: osi: $originalSequenceIDs, rsi: $regionSequenceIDs, p: $prefixes, pep: $prefixEndPositions, s: $suffixes, ssp: $suffixStartPositions, ppsa: $preservePrefixSuffixAlignment"
  
      set osiLength [llength $originalSequenceIDs]
      # Make sure all of the lists are the same length.
      if {[llength $regionSequenceIDs] != $osiLength ||\
            [llength $prefixes] != $osiLength || \
            [llength $prefixEndPositions] != $osiLength || \
            [llength $suffixes] != $osiLength || \
            [llength $suffixStartPositions] != $osiLength } {
         return {}
      }

      # Figure out the positioning.
      set maxPrefixEndPosition -1
      set minSuffixStartPosition -1
      for {set i 0} {$i < [llength $originalSequenceIDs]} {incr i} {
         set prefixEndPosition [lindex $prefixEndPositions $i]
         set suffix [lindex $suffixes $i]
         set suffixStartPosition [lindex $suffixStartPositions $i]
         if {$maxPrefixEndPosition == -1 || $prefixEndPosition > $maxPrefixEndPosition} {
            set maxPrefixEndPosition $prefixEndPosition
         }
         if {$suffix != {} && ($minSuffixStartPosition == -1 || $suffixStartPosition < $minSuffixStartPosition)} {
            set minSuffixStartPosition $suffixStartPosition
         }
      }

      # Copy the region sequence data into the original sequence ids.
      for {set i 0} {$i < [llength $originalSequenceIDs]} {incr i} {

         # Get the data for this sequence.
         set originalSequenceID [lindex $originalSequenceIDs $i]
         set regionSequenceID [lindex $regionSequenceIDs $i]
         set prefix [lindex $prefixes $i]
         set suffix [lindex $suffixes $i]
         set prefixEndPosition [lindex $prefixEndPositions $i]
         set suffixStartPosition [lindex $suffixStartPositions $i]

         # See if we are preserving the alignment of the prefixes and the suffixes.
         if {$preservePrefixSuffixAlignment} {

            # Get any gaps needed to keep the prefix in alignment.
            set prefixGaps [::SeqData::getGaps [expr $maxPrefixEndPosition-$prefixEndPosition]]

            # Get any gaps needed keep the suffix in alignment.
            set suffixGaps {}
            if {$suffix != {}} {
               set suffixGaps [::SeqData::getGaps [expr $suffixStartPosition-$minSuffixStartPosition]]
            }

            # Concatenate the sequence.
            ::SeqData::setSeq $originalSequenceID [concat $prefix $prefixGaps [::SeqData::getSeq $regionSequenceID] $suffixGaps $suffix]

         # Otherwise, ignore the alignment of the prefixes and suffixes.
         } else {

            # Get any gaps needed to gap out the prefix.
            set prefixGaps [::SeqData::getGaps [expr $maxPrefixEndPosition-$prefixEndPosition]]

            # Concatenate the sequence.
            ::SeqData::setSeq $originalSequenceID [concat $prefixGaps $prefix [::SeqData::getSeq $regionSequenceID] $suffix]
         }
      }

      return $originalSequenceIDs
   } ; # end of concatenateRegionsIntoSequences 

# -----------------------------------------------------------------------
    # Apply view transform by moving molecules around.
    proc applyStampTransformations {structureIDs} {

        array set viewrot [::STAMP::getRotations]
        array set viewtrans [::STAMP::getTransformations]

        # Save the viewpoint first -- just in case we need to get it back.
        saveViewpoint

        for {set i 0} {$i < [llength $structureIDs]} {incr i} {

            set transformationMatrix {}

            for {set j 0} {$j < 3} {incr j} {
                lappend transformationMatrix [concat [lindex $viewtrans($i) $j] [lindex $viewrot($i) $j]]
            }
            lappend transformationMatrix {0.000000 0.000000 0.000000 1.0000000}

            # Move the atoms.
            set molID [lindex [::SeqData::VMD::getMolIDForSequence [lindex $structureIDs $i]] 0]
            set atoms [atomselect $molID "all"]  
            $atoms move $transformationMatrix
            $atoms delete
        }

        # Reset the view.
        #mol top [lindex [molinfo list] 0]
        #::ColorBar::reinit
        # display resetview
    }

# --------------------------------------------------------------------------
   # This method performs a sequence alignment.
   proc menu_seqAlign { } {

      variable w
      array set options [::SeqEdit::SeqAlign::showSeqAlignOptionsDialog $w \
                                             [::SeqEditWidget::getGroups]]
      if {[info exists options(alignmentType)] && \
                                 $options(alignmentType) == "multiple"} {
         performMultipleAlignment $options(multipleAlignmentType) \
                                         $options(seqAlignProg)
      } elseif {[info exists options(alignmentType)] && \
                              $options(alignmentType) == "sequence-profile"} {
         performSequenceProfileAlignment $options(profile) \
                                         $options(seqAlignProg)
      } elseif {[info exists options(alignmentType)] && \
                               $options(alignmentType) == "profile-profile"} {
         performProfileProfileAlignment $options(profile1) \
                                      $options(profile2) $options(seqAlignProg)
      }
   }

# --------------------------------------------------------------------------
    # This method align multiple sequences 
    proc performMultipleAlignment {alignmentType seqAlignProg} {

        variable w

        # Extract the appropriate regions.
        set preservePrefixSuffixAlignment 0
        set extractedRegions {}
        if {$alignmentType == "all" || $alignmentType == "marked"} {

            # Get the sequences.
            if {$alignmentType == "all"} {        
                set sequenceIDs [::SeqEditWidget::getSequences]
            } elseif {$alignmentType == "marked"} {
                set sequenceIDs [::SeqEditWidget::getMarkedSequences]
            }

            # Extract the region of the structure that are of the correct type.
            set extractedRegions [::SeqData::VMD::extractFirstRegionFromStructures $sequenceIDs]
            if {$extractedRegions == {}} {
                tk_messageBox -type ok -icon error -parent $w -title "Error" -message "One or more structures did not contain valid data."
                return
            }

            # Mark that we should ignore the rest of the alignment.
            set preservePrefixSuffixAlignment 0

        } elseif {$alignmentType == "selected"} {

            # Extract the selected regions.
            set extractedRegions [::SeqData::extractRegionsFromSequences [::SeqEditWidget::getSelectedCells]]
            if {$extractedRegions == {}} {
                tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The selection within each sequence must be contiguous."
                return
            }

            # Mark that we should preserve the rest of the alignment.
            set preservePrefixSuffixAlignment 1
        }

        # Exract the region info.
        set originalSequenceIDs [lindex $extractedRegions 0]
        set regionSequenceIDs [lindex $extractedRegions 1]
        set prefixes [lindex $extractedRegions 2]
        set prefixEndPositions [lindex $extractedRegions 3]
        set suffixes [lindex $extractedRegions 4]
        set suffixStartPositions [lindex $extractedRegions 5]

        # Make sure all of the structures are of the correct type.

        # Align the structures.
        if {[catch {
           if {$seqAlignProg == "mafft"} {
              set alignedSequenceIDs [::Mafft::alignSequences $regionSequenceIDs]
           } else {
              set alignedSequenceIDs [::ClustalW::alignSequences $regionSequenceIDs]
           }
        } errorMessage] != 0} {
            printStackTrace
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The sequence alignment failed with the following message:\n\n$errorMessage"
            return
        }

        # Verify the aligned sequences.
        if {[llength $alignedSequenceIDs] == [llength $regionSequenceIDs]} {

            # Make sure all of the aligned sequences have the same number of residues.
            for {set i 0} {$i < [llength $regionSequenceIDs]} {incr i} {
                set regionResidueCount [::SeqData::getResidueCount [lindex $regionSequenceIDs $i]]
                set alignedResidueCount [::SeqData::getResidueCount [lindex $alignedSequenceIDs $i]]
                if {$regionResidueCount != $alignedResidueCount} {
                    puts "MultiSeq Error) Sequence alignment dropped residues while aligning [::SeqData::getName [lindex $regionSequenceIDs $i]]. The original number of residues was $regionResidueCount, but aligned number of residues was $alignedResidueCount."
                    tk_messageBox -type ok -icon error -parent $w -title "Error" -message "The sequence alignment failed because all of the residues in the sequence could not be recognized. sequence: [::SeqData::getName [lindex $regionSequenceIDs $i]]."
                    return
                }
            }

        } else {
            tk_messageBox -type ok -icon error -parent $w -title "Error" \
            -message "The Multiple Alignment sequence alignment failed (num of aligned sequences does not match number sent to alignment program)."
            return
        }

        # Puts the sequences back together again.
        set originalSequenceIDs [concatenateRegionsIntoSequences $originalSequenceIDs $alignedSequenceIDs $prefixes $prefixEndPositions $suffixes $suffixStartPositions $preservePrefixSuffixAlignment]
        if {$originalSequenceIDs == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Concatenation of the alignment data failed. The data may be in an inconsistent state."
            return
        }

        # Update the sequences in the editor.
        ::SeqEditWidget::updateSequences $originalSequenceIDs
    }

# --------------------------------------------------------------------------
    # This method performs a sequence profile alignment.
    proc performSequenceProfileAlignment {profile seqAlignProg} {

        variable colorMap
        variable cellSize
        variable w

        set profileSequenceIDs [::SeqEditWidget::getSequencesInGroup $profile]
        set markedSequenceIDs [::SeqEditWidget::getMarkedSequences]
        set originalSequenceIDs [concat $profileSequenceIDs $markedSequenceIDs]

        # Extract the region of the structures that are of the correct type.
        set profileExtractedRegions [::SeqData::VMD::extractFirstRegionFromStructures $profileSequenceIDs]
        if {$profileExtractedRegions == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "One or more structures did not contain valid data."
            return
        }
        set markedExtractedRegions [::SeqData::VMD::extractFirstRegionFromStructures $markedSequenceIDs]
        if {$markedExtractedRegions == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "One or more structures did not contain valid data."
            return
        }

        # Exract the region info.
        set profileRegionSequenceIDs [lindex $profileExtractedRegions 1]
        set markedRegionSequenceIDs [lindex $markedExtractedRegions 1]
        set regionSequenceIDs [concat $profileRegionSequenceIDs $markedRegionSequenceIDs]
        set prefixes [concat [lindex $profileExtractedRegions 2] [lindex $markedExtractedRegions 2]]
        set prefixEndPositions [concat [lindex $profileExtractedRegions 3] [lindex $markedExtractedRegions 3]]
        set suffixes [concat [lindex $profileExtractedRegions 4] [lindex $markedExtractedRegions 4]]
        set suffixStartPositions [concat [lindex $profileExtractedRegions 5] [lindex $markedExtractedRegions 5]]

        # Perform the alignment.
        if {[catch {
           if {$seqAlignProg == "mafft"} {
              set alignedSequenceIDs [::Mafft::alignSequencesToProfile $profileRegionSequenceIDs $markedRegionSequenceIDs]
           } else {
              set alignedSequenceIDs [::ClustalW::alignSequencesToProfile $profileRegionSequenceIDs $markedRegionSequenceIDs]
           }
        } errorMessage] != 0} {
            printStackTrace
            tk_messageBox -type ok -icon error -parent $w -title "Error" \
                 -message "The sequence alignment failed with the following message:\n\n$errorMessage"
            return
        }

        # Verify the aligned sequences.
        if {[llength $alignedSequenceIDs] == [llength $regionSequenceIDs]} {

            # Make sure all of the aligned sequences have the same number of residues.
            for {set i 0} {$i < [llength $regionSequenceIDs]} {incr i} {
#                set unknownResidueCount [::SeqData::getSpecificResidueCount [lindex $regionSequenceIDs $i] "?"]
                set unknownResidueCount 0
                set regionResidueCount [::SeqData::getResidueCount [lindex $regionSequenceIDs $i]]
                set alignedResidueCount [::SeqData::getResidueCount [lindex $alignedSequenceIDs $i]]
                if {$regionResidueCount != ($unknownResidueCount + $alignedResidueCount)} {
                    puts "MultiSeq Error) Residues dropped while aligning [::SeqData::getName [lindex $regionSequenceIDs $i]]. The original number of residues was $regionResidueCount, but aligned number of residues was $alignedResidueCount."
                    tk_messageBox -type ok -icon error -parent $w \
                         -title "Error" \
                         -message "The sequence alignment failed because all residues not recognized in sequence [::SeqData::getName [lindex $regionSequenceIDs $i]]."
                    return
                }
            }

        } else {
            tk_messageBox -type ok -icon error -parent $w -title "Error" \
            -message "The Sequence Profile sequence alignment failed (num of aligned sequences does not match number sent to alignment program)."
            return
        }

        # Puts the sequences back together again.
        set originalSequenceIDs [concatenateRegionsIntoSequences $originalSequenceIDs $alignedSequenceIDs $prefixes $prefixEndPositions $suffixes $suffixStartPositions 0]
        if {$originalSequenceIDs == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Concatenation of the alignment data failed. The data may be in an inconsistent state."
            return
        }

        # Update the sequences in the editor.
        ::SeqEditWidget::updateSequences $originalSequenceIDs
    }

# --------------------------------------------------------------------------
    # This method performs a sequence profile alignment.
    proc performProfileProfileAlignment {profile1 profile2 seqAlignProg} {

        variable colorMap
        variable cellSize

        set profile1SequenceIDs [::SeqEditWidget::getSequencesInGroup $profile1]
        set profile2SequenceIDs [::SeqEditWidget::getSequencesInGroup $profile2]
        set originalSequenceIDs [concat $profile1SequenceIDs $profile2SequenceIDs]

        # Extract the region of the structures that are of the correct type.
        set profile1ExtractedRegions [::SeqData::VMD::extractFirstRegionFromStructures $profile1SequenceIDs]
        if {$profile1ExtractedRegions == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "One or more structures did not contain valid data."
            return
        }
        set profile2ExtractedRegions [::SeqData::VMD::extractFirstRegionFromStructures $profile2SequenceIDs]
        if {$profile2ExtractedRegions == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "One or more structures did not contain valid data."
            return
        }

        # Exract the region info.
        set profile1RegionSequenceIDs [lindex $profile1ExtractedRegions 1]
        set profile2RegionSequenceIDs [lindex $profile2ExtractedRegions 1]
        set regionSequenceIDs [concat $profile1RegionSequenceIDs $profile2RegionSequenceIDs]
        set prefixes [concat [lindex $profile1ExtractedRegions 2] [lindex $profile2ExtractedRegions 2]]
        set prefixEndPositions [concat [lindex $profile1ExtractedRegions 3] [lindex $profile2ExtractedRegions 3]]
        set suffixes [concat [lindex $profile1ExtractedRegions 4] [lindex $profile2ExtractedRegions 4]]
        set suffixStartPositions [concat [lindex $profile1ExtractedRegions 5] [lindex $profile2ExtractedRegions 5]]

        # Perform the alignment.
        if {[catch {
           if {$seqAlignProg == "mafft"} {
              set alignedSequenceIDs [::Mafft::alignProfiles $profile1RegionSequenceIDs $profile2RegionSequenceIDs]
           } else {
              set alignedSequenceIDs [::ClustalW::alignProfiles $profile1RegionSequenceIDs $profile2RegionSequenceIDs]
           }
        } errorMessage] != 0} {
            printStackTrace
            tk_messageBox -type ok -icon error -parent $w -title "Error" \
                   -message "The sequence alignment failed with the following message:\n\n$errorMessage"
            return
        }

        # Verify the aligned sequences.
        if {[llength $alignedSequenceIDs] == [llength $regionSequenceIDs]} {

            # Make sure all of the aligned sequences have the same number of residues.
            for {set i 0} {$i < [llength $regionSequenceIDs]} {incr i} {
                set regionResidueCount [::SeqData::getResidueCount [lindex $regionSequenceIDs $i]]
                set alignedResidueCount [::SeqData::getResidueCount [lindex $alignedSequenceIDs $i]]
                if {$regionResidueCount != $alignedResidueCount} {
                    puts "MultiSeq Error) Residues dropped while aligning [::SeqData::getName [lindex $regionSequenceIDs $i]]. The original number of residues was $regionResidueCount, but aligned number of residues was $alignedResidueCount."
                    tk_messageBox -type ok -icon error -parent $w \
                        -title "Error" \
                        -message "The sequence alignment failed because all residues not recognized in sequence [::SeqData::getName [lindex $regionSequenceIDs $i]]."
                    return
                }
            }

        } else {
            tk_messageBox -type ok -icon error -parent $w -title "Error" \
                   -message "The Profile to Profile sequence alignment failed (num of aligned sequences does not match number sent to alignment program)."
            return
        }

        # Puts the sequences back together again.
        set originalSequenceIDs [concatenateRegionsIntoSequences $originalSequenceIDs $alignedSequenceIDs $prefixes $prefixEndPositions $suffixes $suffixStartPositions 0]
        if {$originalSequenceIDs == {}} {
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Concatenation of the alignment data failed. The data may be in an inconsistent state."
            return
        }

        # Update the sequences in the editor.
        ::SeqEditWidget::updateSequences $originalSequenceIDs
    }

# -------------------------------------------------------------------------
   proc menu_phylotree {} {

      variable w

      array set options [::MultiSeq::Phylotree::showOptionsDialog $w]
      if {[array size options] > 0} {

         # Get the sequence ids.
         set sequenceIDs {}
         if {$options(selectionType) == "all"} {
            set sequenceIDs [::SeqEditWidget::getSequences]
         } elseif {$options(selectionType) == "marked"} {
            set sequenceIDs [::SeqEditWidget::getMarkedSequences]
         } elseif {$options(selectionType) == "selected"} {

            # Get the selected cells.
#            puts "multiseq.tcl. menu_phylotree. getSelCells [::SeqEditWidget::getSelectedCells]"
            set selected [::SeqEditWidget::getSelectedCellsCombinedBySequence]
#            puts "multiseq.tcl. menu_phylotree.getSelCellCombBySeq $selected"
            # Go through each sequence that has something selected.
            set selectionLength -1
            set selectionSequenceIDs {}
            for {set i 0} {$i < [llength $selected]} {incr i 2} {

               # Get the sequence id and elements of the selection.
               set selectedSequenceID [lindex $selected $i]
               set selectedElements [lindex $selected [expr $i+1]]

               # Make sure that the selection is of the correct length.
               if {$selectionLength != -1 && [llength $selectedElements] \
                                                      != $selectionLength} {
                  tk_messageBox -type ok -icon error -parent $w -title "Error" \
                        -message "A phylogenetic tree can only be made of a selection that is of the same length in each sequence."
                  return
               }
               set selectionLength [llength $selectedElements]

               # Make sure that the selection is contiguous.
               set previousElement -1
               foreach element $selectedElements {
                  if {$previousElement != -1 && $element != \
                                                [expr $previousElement+1]} {
                     tk_messageBox -type ok -icon error -parent $w -title \
                              "Error" -message \
                              "A phylogenetic tree can only be made of a selection that is contiguous."
                     return
                  }
                  set previousElement $element
               }

               # Create new sequences from the selection.
               set newSequenceID [::SeqData::duplicateSequence \
                      $selectedSequenceID [lindex $selectedElements 0] \
                      [lindex $selectedElements end]]
               ::SeqData::addAnnotation $newSequenceID "parent-sequence-id" \
                                                         $selectedSequenceID
               lappend sequenceIDs $newSequenceID
            }
         }

         # If we are only using aligned columnns, create a new set of seqeunces.
         if {$options(alignedOnly)} {

            # Figure out which positions to keep.
            set maxGapFraction 0.05
            puts "Ignoring columns with more than [expr 100.0*$maxGapFraction]\% gaps."
            set positionsToKeep {}
            for {set position 0} {$position < [::SeqData::getSeqLength \
                                     [lindex $sequenceIDs 0]]} {incr position} {
               set numberGaps 0
               foreach sequenceID $sequenceIDs {
                  if {[::SeqData::getElement $sequenceID $position] == "-"} {
                     incr numberGaps
                  }
               }
               set gapFraction [expr double($numberGaps)/double([llength \
                                                                 $sequenceIDs])]
               if {$gapFraction <= $maxGapFraction} {
                  lappend positionsToKeep $position
               }
            }

            # Create a new alignment with only those positions we are keeping.
            puts "Number of columns below cutoff: [llength $positionsToKeep]."
            set newSequenceIDs {}
            foreach sequenceID $sequenceIDs {
               set sequence [::SeqData::getSeq $sequenceID]
               set newSequence {}
               foreach position $positionsToKeep {
                  lappend newSequence [lindex $sequence $position]
               }
               set newSequenceID [::SeqData::duplicateSequence $sequenceID]
               ::SeqData::setSeq $newSequenceID $newSequence
               ::SeqData::addAnnotation $newSequenceID "parent-sequence-id" \
                                                                $sequenceID
               lappend newSequenceIDs $newSequenceID
            }
            set sequenceIDs $newSequenceIDs
         }

         # Get a list of just the structures.
         set structureIDs {}
         foreach sequenceID $sequenceIDs {
            if {[::SeqData::hasStruct $sequenceID] == "Y"} {
               lappend structureIDs $sequenceID
            }
         }

         # Show the tree window.
         ::PhyloTree::createWindow 0
         ::PhyloTree::Widget::addSelectionNotificationCommand \
                             "::MultiSeq::phyloTreeToSeqEditSelectionConnector"
         ::PhyloTree::Widget::addTreeChangeNotificationCommand \
                                        "::MultiSeq::phyloTreeChangedConnector"
         ::SeqEditWidget::setSelectionNotificationCommand \
                             "::MultiSeq::seqEditToPhyloTreeSelectionConnector"

         # See if we need to calculate the QH tree.
         if {$options(qh) == 1} {

            set matrix [::Libbiokit::getPairwiseQH $structureIDs]
            set treeData [::UPGMA_Cluster::createUPGMATree $matrix]
            set treeID [::PhyloTree::JE::loadTreeData "QH Structure Tree" \
                                                                  $treeData]
            if {$treeID != ""} {
               crossReferenceTreeByIndex $treeID $structureIDs
               ::PhyloTree::Data::setTreeUnits $treeID "delta QH"
               ::PhyloTree::addTrees $treeID
            }
         }

         # See if we need to calculate the RMSD tree.
         if {$options(rmsd) == 1} {       

            set matrix [::Libbiokit::getPairwiseRMSD $structureIDs]
            set treeData [::UPGMA_Cluster::createUPGMATree $matrix]
            set treeID [::PhyloTree::JE::loadTreeData "RMSD Structure Tree" \
                                                                $treeData]
            if {$treeID != ""} {
               crossReferenceTreeByIndex $treeID $structureIDs
               ::PhyloTree::Data::setTreeUnits $treeID "A RMSD"
               ::PhyloTree::addTrees $treeID
            }
         }

         # See if we need to calculate the PID tree.
         if {$options(pid) == 1} {       

            set matrix [::Libbiokit::getPairwisePercentIdentity $sequenceIDs]

            # Transform the matrix into a distance matrix.
            set distanceMatrix {}
            foreach row $matrix {
               set distanceRow {}
               foreach element $row {
                  lappend distanceRow [expr 1.0-$element]
               }
               lappend distanceMatrix $distanceRow
            }

            set treeData [::UPGMA_Cluster::createUPGMATree $distanceMatrix]
            set treeID [::PhyloTree::JE::loadTreeData \
                                     "Percent Identity Sequence Tree" $treeData]
            if {$treeID != ""} {
               crossReferenceTreeByIndex $treeID $sequenceIDs
               ::PhyloTree::Data::setTreeUnits $treeID "delta PID"
               ::PhyloTree::addTrees $treeID
            }
         }

         # See if we need to calculate the CLUSTALW tree.
         if {$options(clustalw) == 1} {                       
            set sequenceNames {}
            for {set i 0} {$i < [llength $sequenceIDs]} {incr i} {
               lappend sequenceNames $i
            }
            set treeID [::ClustalW::calculatePhylogeneticTree $sequenceIDs \
                                                              $sequenceNames]
            if {$treeID != ""} {
               crossReferenceTreeByIndex $treeID $sequenceIDs
               ::PhyloTree::Data::setTreeUnits $treeID "changes per site"
               ::PhyloTree::addTrees $treeID
            }
         }

         # See if we need to calculate the MAFFT tree.
         if {$options(mafft) == 1} {                       
            set sequenceNames {}
            for {set i 0} {$i < [llength $sequenceIDs]} {incr i} {
               lappend sequenceNames $i
            }
            set treeID [::Mafft::calculatePhylogeneticTree $sequenceIDs \
                                                              $sequenceNames]
            if {$treeID != ""} {
               fixNodeNames $treeID 
               crossReferenceTreeByIndex $treeID $sequenceIDs
               ::PhyloTree::Data::setTreeUnits $treeID "changes per site"
               ::PhyloTree::addTrees $treeID
            }
         }

         # See if we need to load trees from a file.
         if {$options(file) == 1 && $options(filename) != "" && \
                                             [file exists $options(filename)]} {

            set treeIDs {}
            set extension [string tolower [file extension $options(filename)]]
            if {$extension == ".jet"} {
               set treeIDs [::PhyloTree::JE::loadTreeFile $options(filename)]                
            } elseif {$extension == ".nex" || $extension == ".nxs"} {
               set treeIDs [::PhyloTree::Nexus::loadTreeFile $options(filename)]
            } elseif {$extension == ".dnd" || $extension == ".ph" || 
                                                       $extension == ".tre"} {
               set treeIDs [::PhyloTree::Newick::loadTreeFile $options(filename)]
            } else {
               if {[catch {set treeIDs [::PhyloTree::Nexus::loadTreeFile \
                                            $options(filename)]} msg] == 0} {
               } elseif {[catch {set treeIDs [::PhyloTree::Newick::loadTreeFile\
                                              $options(filename)]} msg] == 0} {
               } elseif {[catch {set treeIDs [::PhyloTree::JE::loadTreeFile \
                                              $options(filename)]} msg] == 0} {
               } else {
                  tk_messageBox -type ok -icon error -parent $w -title "Error"\
                              -message "The format of the specified file was not recognized."
                  return
               }
            }
            if {$treeIDs != {}} {
               crossReferenceTreesByName $treeIDs $sequenceIDs
               ::PhyloTree::addTrees $treeIDs
            }
         }
      }
   }

# -------------------------------------------------------------------------
    proc fixNodeNames {treeID } {
       # Get the leaf nodes.
       set leafNodes [::PhyloTree::Data::getLeafNodes $treeID [::PhyloTree::Data::getTreeRootNode $treeID]]

       foreach node $leafNodes {
          if { [info exists first] } {
             unset first
          }

          set nodeName [::PhyloTree::Data::getNodeName $treeID $node]

#          puts "starting with $nodeName"
# for a given fasta file line of   > 3 scientific Name
# MAFFT will set a node name of   X_3_scientific_Name
# we need to extract the '3'
          regexp "\_(.*?)\_" $nodeName everything first 

          if { ! [info exists first] } {
             regexp "(.*)\_(.*)" $nodeName everything zero first 
          }

#          puts "setting $nodeName to $first"

          ::PhyloTree::Data::setNodeName $treeID $node $first
       }
    }


# -------------------------------------------------------------------------
    proc crossReferenceTreeByIndex {treeID sequenceIDs} {

        variable treeNodeToSequenceIDMap
        variable sequenceIDToTreeNodeMap

        # Get the leaf nodes.
        set leafNodes [::PhyloTree::Data::getLeafNodes $treeID [::PhyloTree::Data::getTreeRootNode $treeID]]

        # Assign the properties to the nodes using the node name as the index into the list.
        foreach node $leafNodes {

            set nodeName [::PhyloTree::Data::getNodeName $treeID $node]
#            puts "node is $node, name is $nodeName"
            if {$nodeName < [llength $sequenceIDs]} {

                # Get the sequence id and name.
                set sequenceID [lindex $sequenceIDs $nodeName]
                set sequenceName [::SeqData::getName $sequenceID]

                # Replace the node name and set any node attributes.
                ::PhyloTree::Data::setNodeName $treeID $node $sequenceName
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Enzyme::Name" [::SeqData::getEnzymeCommissionDescription $sequenceID]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Enzyme::Code" [::SeqData::getEnzymeCommissionNumber $sequenceID]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Domain of Life" [::SeqData::getDomainOfLife $sequenceID] {"Archaea" "Bacteria" "Eukaryota" "Virus"}
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Kingdom" [::SeqData::getLineageRank $sequenceID "kingdom"]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Phylum" [::SeqData::getLineageRank $sequenceID "phylum"]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Class" [::SeqData::getLineageRank $sequenceID "class"]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Order" [::SeqData::getLineageRank $sequenceID "order"]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Family" [::SeqData::getLineageRank $sequenceID "family"]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Genus" [::SeqData::getLineageRank $sequenceID "genus"]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Species" [::SeqData::getScientificName $sequenceID]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Abbr. Species" [::SeqData::getShortScientificName $sequenceID]
                ::PhyloTree::Data::setNodeAttribute $treeID $node "Temperature Class" [::SeqData::getTemperatureClass $sequenceID] {"psychrophilic" "hyperthermophilic" "thermophilic" "mesophilic"}
                set parentID [::SeqData::getAnnotation $sequenceID "parent-sequence-id"]
                if {$parentID != ""} {
                    set treeNodeToSequenceIDMap($treeID,$node) $parentID
                    set sequenceIDToTreeNodeMap($parentID,$treeID) $node
                    ::PhyloTree::Data::setNodeAttribute $treeID $node "MultiSeq Group" [::SeqEditWidget::getGroup $parentID]
                } else {    
                    set treeNodeToSequenceIDMap($treeID,$node) $sequenceID
                    set sequenceIDToTreeNodeMap($sequenceID,$treeID) $node
                    ::PhyloTree::Data::setNodeAttribute $treeID $node "MultiSeq Group" [::SeqEditWidget::getGroup $sequenceID]
                }

                # Change the node name in the distance matrix.
                set distanceMatrix [::PhyloTree::Data::getDistanceMatrix $treeID]
                if {$distanceMatrix != {}} {
                    set newDistanceMatrix {}
                    foreach row $distanceMatrix {
                        if {[lindex $row 0] == $nodeName} {
                            lappend newDistanceMatrix [lreplace $row 0 0 $sequenceName]
                        } else {
                            lappend newDistanceMatrix $row
                        }
                    }
                    ::PhyloTree::Data::setDistanceMatrix $treeID $newDistanceMatrix
                }
            }            
        }
    }

# -------------------------------------------------------------------------
    proc crossReferenceTreesByName {treeIDs sequenceIDs} {

        variable treeNodeToSequenceIDMap
        variable sequenceIDToTreeNodeMap

        foreach treeID $treeIDs {

            # Get the leaf nodes.
            set leafNodes [::PhyloTree::Data::getLeafNodes $treeID [::PhyloTree::Data::getTreeRootNode $treeID]]

            # Assign the properties to the nodes using the node name as the index into the list.
            foreach node $leafNodes {

                # Get the node name.
                set nodeName [::PhyloTree::Data::getNodeName $treeID $node]
                regsub -all {[^a-z0-9]} [string tolower $nodeName] "" genericNodeName

                # See if we can find a sequence with the same name.
                foreach sequenceID $sequenceIDs {

                    # If we found a matching sequence name, use it.
                    set sequenceName [::SeqData::getName $sequenceID]
                    regsub -all {[^a-z0-9]} [string tolower $sequenceName] "" genericSequenceName
                    if {$genericSequenceName == $genericNodeName} {
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Enzyme::Name" [::SeqData::getEnzymeCommissionDescription $sequenceID]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Enzyme::Code" [::SeqData::getEnzymeCommissionNumber $sequenceID]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Domain of Life" [::SeqData::getDomainOfLife $sequenceID] {"Archaea" "Bacteria" "Eukaryota" "Virus"}
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Kingdom" [::SeqData::getLineageRank $sequenceID "kingdom"]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Phylum" [::SeqData::getLineageRank $sequenceID "phylum"]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Class" [::SeqData::getLineageRank $sequenceID "class"]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Order" [::SeqData::getLineageRank $sequenceID "order"]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Family" [::SeqData::getLineageRank $sequenceID "family"]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Genus" [::SeqData::getLineageRank $sequenceID "genus"]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Species" [::SeqData::getScientificName $sequenceID]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Taxonomy::Abbr. Species" [::SeqData::getShortScientificName $sequenceID]
                        ::PhyloTree::Data::setNodeAttribute $treeID $node "Temperature Class" [::SeqData::getTemperatureClass $sequenceID] {"psychrophilic" "hyperthermophilic" "thermophilic" "mesophilic"}
                        set parentID [::SeqData::getAnnotation $sequenceID "parent-sequence-id"]
                        if {$parentID != ""} {
                            set treeNodeToSequenceIDMap($treeID,$node) $parentID
                            set sequenceIDToTreeNodeMap($parentID,$treeID) $node
                            ::PhyloTree::Data::setNodeAttribute $treeID $node "MultiSeq Group" [::SeqEditWidget::getGroup $parentID]
                        } else {    
                            set treeNodeToSequenceIDMap($treeID,$node) $sequenceID
                            set sequenceIDToTreeNodeMap($sequenceID,$treeID) $node
                            ::PhyloTree::Data::setNodeAttribute $treeID $node "MultiSeq Group" [::SeqEditWidget::getGroup $sequenceID]
                        }
                        break
                    }
                }
            }
        }
    }

# -------------------------------------------------------------------------
    proc phyloTreeToSeqEditSelectionConnector {} {

        variable treeNodeToSequenceIDMap

        # Get the sequence ids of the selected nodes.
        set treeID [::PhyloTree::getActiveTree]
        if {$treeID != -1} {
            set selectedSequenceIDs {}
            foreach node [::PhyloTree::Widget::getSelectedNodes] {
                if {[info exists treeNodeToSequenceIDMap($treeID,$node)]} {
                    lappend selectedSequenceIDs $treeNodeToSequenceIDMap($treeID,$node)
                }
            }

            if {![compareLists $selectedSequenceIDs [::SeqEditWidget::getSelectedSequences]]} {
                ::SeqEditWidget::setSelectedSequences $selectedSequenceIDs 1
            }
        }
    }

# -------------------------------------------------------------------------
    proc phyloTreeChangedConnector {} {

        variable sequenceIDToTreeNodeMap

        # Get the sequence ids of the selected nodes.
        set treeID [::PhyloTree::getActiveTree]
        if {[::PhyloTree::windowExists] && $treeID != -1} {
            set selectedNodes {}
            foreach sequenceID [::SeqEditWidget::getSelectedSequences] {
                if {[info exists sequenceIDToTreeNodeMap($sequenceID,$treeID)]} {
                    lappend selectedNodes $sequenceIDToTreeNodeMap($sequenceID,$treeID)
                }
            }

            if {![compareLists $selectedNodes [::PhyloTree::Widget::getSelectedNodes]]} {
                ::PhyloTree::Widget::setSelectedNodes $selectedNodes 0 0 1
            }
        }
    }

# -------------------------------------------------------------------------
    proc seqEditToPhyloTreeSelectionConnector {} {

        variable sequenceIDToTreeNodeMap

        # Get the sequence ids of the selected nodes.
        set treeID [::PhyloTree::getActiveTree]
        if {[::PhyloTree::windowExists] && $treeID != -1} {
            set selectedNodes {}
            foreach sequenceID [::SeqEditWidget::getSelectedSequences] {
                if {[info exists sequenceIDToTreeNodeMap($sequenceID,$treeID)]} {
                    lappend selectedNodes $sequenceIDToTreeNodeMap($sequenceID,$treeID)
                }
            }

            if {![compareLists $selectedNodes [::PhyloTree::Widget::getSelectedNodes]]} {
                ::PhyloTree::Widget::setSelectedNodes $selectedNodes 0 0 1
            }
        }
    }

# -------------------------------------------------------------------------
    proc compareLists {list1 list2} {

        foreach item $list1 {
            if {$item != ""} {
                if {[lsearch $list2 $item] == -1} {
                    return 0
                }
            }
        }
        foreach item $list2 {
            if {$item != ""} {
                if {[lsearch $list1 $item] == -1} {
                    return 0
                }
            }
        }
        return 1
    }

# ------------------------------------------------------------------------
   # This method allows the user to plot data
   proc menu_plot {} {

      variable w
      variable tempDir
      variable filePrefix
      variable tempFiles

      array set options [::MultiSeq::Plot::showPlotDialog $w]
      if {[array size options] > 0 && [llength $options(plotType)] == 4} {

         # First, should check and see what the user wants to plot data for
         # and extract the sequences/range of data/whatever from the rest
         # of multiseq.  Some metrics won't apply to non-structure
         # data so we'll need to check.
         if {$options(selectionType) == "all" || $options(selectionType) == "marked"} {
            # Get the sequences.
            if {$options(selectionType) == "all"} {
               set sequenceIDs [::SeqEditWidget::getSequences]
            } elseif {$options(selectionType) == "marked"} {
               set sequenceIDs [::SeqEditWidget::getMarkedSequences]
            }

         # See if we are only plotting for the selected regions.
         } elseif {$options(selectionType) == "selected"} {
            # Get the selected cells.
            set selected [::SeqEditWidget::getSelectedCells]

            # Go through each sequence that has something selected.
            set sequenceIDs {}
            set selectionLength -1
            for {set i 0} {$i < [llength $selected]} {incr i 2} {
               # Get the sequence id and elements of the selection.
               set selectedSequenceID [lindex $selected $i]
               set selectedElements [lindex $selected [expr $i+1]]

               # Make sure that the selection is of the correct length.
               if {$selectionLength != -1 && [llength $selectedElements] != $selectionLength} {
                  tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Only selections of the same length may be plotted."
                  return
               }
               set selectionLength [llength $selectedElements]

               # Make sure that the selection is continguous.
               set previousElement -1
               foreach element $selectedElements {
                  if {$previousElement != -1 && $element != [expr $previousElement+1]} {
                     tk_messageBox -type ok -icon error -parent $w -title "Error" -message "Only contiguous selections can be plotted."
                     return
                  }
                  set previousElement $element
               }

               # Create a new sequence from the selection for use in the BLAST search.
               lappend sequenceIDs [::SeqData::duplicateSequence $selectedSequenceID [lindex $selectedElements 0] [lindex $selectedElements end]]
            }
         }

         # User wants to plot RMSD per residue.
         set metricName [lindex $options(plotType) 0]
         if {$metricName == "customMetric"} {
            set metricName $options(customMetric)
         }
         array set metricMap [$metricName $sequenceIDs]

         # Copy the coloring metrics into the coloring map.
         set coloringMetricKeys [array names coloringMetricMap]
         foreach coloringMetricKey $coloringMetricKeys {
            set coloringMap($coloringMetricKey,raw) $coloringMetricMap($coloringMetricKey)
            set coloringMap($coloringMetricKey,rgb) [$colorMapper $coloringMetricMap($coloringMetricKey)]
         }


         set plotArgs {}
         foreach sequenceID $sequenceIDs {
            set coords {}
            for {set i 0} {$i < [::SeqData::getSeqLength $sequenceID]} {incr i} {
               if {$metricMap($sequenceID,$i) >= 0.0} {
                  lappend coords [list [expr $i+1] $metricMap($sequenceID,$i)]
               } else {
                  lappend coords [list [expr $i+1] ""]
               }
            }

            lappend plotArgs [list $coords [::SeqData::getName $sequenceID]]
         }

         #plotData [lindex $options(plotType) 1] [lindex $options(plotType) 2] [lindex $options(plotType) 3] $plotArgs
         set xlabel [lindex $options(plotType) 1]
         set ylabel [lindex $options(plotType) 2]
         set wtitle [lindex $options(plotType) 3]
         set data_list $plotArgs

         set mplot [multiplot -title $wtitle -lines -xlabel $xlabel -ylabel $ylabel]

         foreach data $data_list {
            set coordList [lindex $data 0]
            set label [lindex $data 1]
            set xlist ""
            set ylist ""
            foreach pair $coordList {
               set y [lindex $pair 1]
               if { $y == "" } { 
                  set y 0 
               }
               lappend xlist [lindex $pair 0]
               lappend ylist $y
            }
            $mplot add $xlist $ylist -legend $label
         }
         variable mplotWindow
         set mplotWindow ""

         append mplotWindow [string range $mplot 0 [expr 1 + [string last "::" $mplot]]] "w"
         eval set mplotWindow $$mplotWindow

         set bottomFrame [frame $mplotWindow.bottom -background white -height 50 -border 0]
         set psButton [button $mplotWindow.bottom.psButton -command [namespace current]::PSout -text "Save plot to PostScript file"]

         pack $psButton -side left 
         pack $bottomFrame -fill x -side bottom   

         $mplot replot	    
      }
   } ;# end of menu_plot

# -------------------------------------------------------------------------
    proc menu_atomSelectionMode {} {

        variable atomSelectionMode

        # Set the mouse mode appropriately.
        if {$atomSelectionMode == 0} {
            mouse mode 0
        } elseif {$atomSelectionMode == 1} {
            mouse mode 4 2
        }
    }

# ----------------------------------------------------------------------
   proc menu_changeColoring {coloringMetric {fullControl 0}} {
      variable w
      variable coloringOptions

      set ::MultiSeq::coloringMetric $coloringMetric

#      puts "multiseq.menu_changeColoring.metric: $coloringMetric, fullCont: $fullControl, colOpt: $coloringOptions"

      # Figure out the options.
      if {$coloringOptions == "marked"} {
         set sequenceIDs [::SeqEditWidget::getMarkedSequences]
         set colorByGroup 0
      } elseif {$coloringOptions == "group"} {
         set sequenceIDs [::SeqEditWidget::getSequences]
         set colorByGroup 1
      } else {
         set sequenceIDs [::SeqEditWidget::getSequences]
         set colorByGroup 0
      }
#      puts "multiseq.menu_changeColoring.seqIDs: $sequenceIDs, cbg: $colorByGroup"

      # See if we are resetting the coloring.
      if {$coloringMetric == {}} {
         ::SeqEditWidget::resetColoring $sequenceIDs

      # See if we are importing the coloring.
      } elseif {$coloringMetric == "::SeqEdit::Metric::Import::readFile"} {

         # Get the filename and pass it to the importer.            
         set filename [tk_getOpenFile -filetypes {{{Dat Files} {.dat}} \
                           {{All Files} * }} -title "Load Custom Metric Data"]
         if {$filename != ""} {
            ::SeqEdit::Metric::Import::setFilename $filename
            ::SeqEditWidget::setColoring $coloringMetric $sequenceIDs \
                                                              $colorByGroup
         }

      # See if we are importing the coloring.
      } elseif {$coloringMetric == "custom"} {

         # Get the name of the custom metric.
         array set options [::MultiSeq::GetInput::showGetInputDialog $w \
                             "Color By Custom Metric" "Enter procedure name"]
         if {[array size options] > 0 && $options(value) != ""} {

            # Color by the new metric.
            ::SeqEditWidget::setColoring $options(value) $sequenceIDs \
                                                              $colorByGroup
         }

      # Otherwise apply the new coloring.
      } else {
         ::SeqEditWidget::setColoring $coloringMetric $sequenceIDs \
                                                 $colorByGroup 1 $fullControl
      }

      # Apply the coloring to the VMD molecules.
      applyColoringToMolecules $sequenceIDs
   } ; # end of menu_changeColoring

# ---------------------------------------------------------------------------
   proc applyColoringToMolecules {{sequenceIDs all}} {
      variable representations

      # If we are applying to all sequences, get the ids.
      if {$sequenceIDs == "all"} {
         set sequenceIDs [::SeqEditWidget::getSequences]
      }

      # Get the function to map a value to an index.
      set colorIndexMap "[::SeqEditWidget::getColorMap]\::getColorValueForIndex"

      # Go through each sequence that has a VMD equivalent.
      foreach sequenceID $sequenceIDs {
         if {[::SeqData::hasStruct $sequenceID] == "Y"} {
            # Get the mol id.
            set molID [lindex [::SeqData::VMD::getMolIDForSequence \
                                                          $sequenceID] 0]

            # Get the number of elements in the sequence.
            set numberElements [SeqData::getSeqLength $sequenceID] 
            # Go through each elem that has residue and set user color value.
            for {set i 0} {$i < $numberElements} {incr i} {
               set selStr \
                  [::SeqData::VMD::getSelectionStringForElements $sequenceID $i]
               if {$selStr != "none"} {
                  set atoms [atomselect $molID $selStr]
                  set colorValue [$colorIndexMap [seq get color $sequenceID $i]]
#                  puts "OpenGL:seq: $sequenceID, elem: $i, seqCol:[seq get color $sequenceID $i], hexcolor: [::MultiSeq::ColorMap::VMD::getColor [seq get color $sequenceID $i]], color: $colorValue" 
                  $atoms set "user" $colorValue
                  $atoms delete
               }
            }
#            puts "multiseq.tcl.applyColoringToMolecules applied to seqID: $sequenceID"

            # Make sure we have sequence representations.
            createSequenceRepresentations $sequenceID

            # Modify the representations to show the coloring.

            #foreach repName $representations($sequenceID,sequence) {}
            foreach repName [array names representations "$sequenceID,*"] {
               set repID $representations($repName)
               set repIndex [mol repindex $molID $repID]
               mol modcolor $repIndex $molID "User"
               mol scaleminmax $molID $repIndex 0 1.0
            }
         }
      }
   } ; # end of applyColoringToMolecules

# -------------------------------------------------------------------------
    proc menu_group {groupType} {

        variable w

        if {$groupType == "none"} {

            ::SeqEditWidget::setGroups {"Sequences"}

        } elseif {$groupType == "taxonomy"} {

            array set options [::SeqEdit::TaxonomicGrouping::showOptionsDialog $w]
            if {[array size options] > 0} {

                # Get the sequences to group.
                set sequenceIDs {}
                if {$options(selectionType) == "all"} {
                    set sequenceIDs [::SeqEditWidget::getSequences]
                } elseif {$options(selectionType) == "marked"} {
                    set sequenceIDs [::SeqEditWidget::getMarkedSequences]
                }

                # Get the level we are grouping by.
                set groupingLevel [lindex {superkingdom kingdom phylum class order family genus species} $options(level)]

                # Get a list of the groups and the sequences in each.
                set groups {}
                array unset groupSequences 
                set ungroupedSequences {}
                foreach sequenceID $sequenceIDs {

                    set group [::SeqData::getLineageRank $sequenceID $groupingLevel]

                    if {$group != ""} {
                        if {[lsearch $groups $group] == -1} {
                            lappend groups $group
                            set groupSequences($group) {}
                        }
                        lappend groupSequences($group) $sequenceID
                    } else {
                        lappend ungroupedSequences $sequenceID
                    }
                }
                set groups [lsort $groups]

                # Create the groups and move the sequence into them.
                if {[llength $ungroupedSequences] > 0} {
                    ::SeqEditWidget::createGroup "Unknown" 0
                    ::SeqEditWidget::moveSequences $ungroupedSequences "Unknown" end 0
                }
                foreach group $groups {
                    ::SeqEditWidget::createGroup $group 0
                    ::SeqEditWidget::moveSequences $groupSequences($group) $group end 0
                }

                # Remove any empty groups.
                foreach group [::SeqEditWidget::getGroups] {
                    if {[llength [::SeqEditWidget::getSequencesInGroup $group]] == 0} {
                        ::SeqEditWidget::deleteGroup $group
                    }
                }

                # Redraw the editor.
                ::SeqEditWidget::redraw
            }

        } elseif {$groupType == "type"} {

            # Get the currently selected sequences.
            set sequenceIDs [::SeqEditWidget::getSequences]

            # Organize the sequences by group.
            set protein {}
            set nucleic {}
            set unknown {}
            foreach sequenceID $sequenceIDs {
                if {[::SeqData::getType $sequenceID] == "protein"} {
                    lappend protein $sequenceID
                } elseif {[::SeqData::getType $sequenceID] == "rna" || [::SeqData::getType $sequenceID] == "dna" } {
                    lappend nucleic $sequenceID
                } else {
                    lappend unknown $sequenceID
                }
            }

            # Move the sequences into their groups.
            ::SeqEditWidget::createGroup "Protein" 0
            ::SeqEditWidget::moveSequences $protein "Protein" end 0
            ::SeqEditWidget::createGroup "Nucleic" 0
            ::SeqEditWidget::moveSequences $nucleic "Nucleic" end 0
            ::SeqEditWidget::createGroup "Unknown" 0
            ::SeqEditWidget::moveSequences $unknown "Unknown" end 0

            # Remove any empty groups.
            foreach group [::SeqEditWidget::getGroups] {
                if {[llength [::SeqEditWidget::getSequencesInGroup $group]] == 0} {
                    ::SeqEditWidget::deleteGroup $group
                }
            }

            # Redraw the editor.
            ::SeqEditWidget::redraw

        } elseif {$groupType == "selection"} {

            # Get the currently selected sequences.
            set sequenceIDs [::SeqEditWidget::getSelectedSequences]

            # Make sure we have some selected sequences.
            if {$sequenceIDs != {}} {

                # Get a name for the new group.
                array set options [::SeqEdit::GetGroupName::showGetGroupNameDialog $w "Create Group From Selection" "Enter group name"]
                if {[array size options] > 0 && $options(name) != ""} {

                    # Create the group.
                    set groupIndex [::SeqEditWidget::createGroup $options(name)]

                    # Move the sequences into the group.
                    ::SeqEditWidget::moveSequences $sequenceIDs $options(name)
                }
            }


        } elseif {$groupType == "custom"} {
            set newGroups [::SeqEdit::CustomizeGroups::showCustomizeGroupsDialog $w [::SeqEditWidget::getGroups]]
            if {$newGroups != {}} {
                ::SeqEditWidget::setGroups $newGroups
            }
        }   
    }

    proc menu_zoomIn {} {

        variable w
        variable cellSize

        if {$cellSize < 10} {
            set cellSize [expr $cellSize+1]
            ::SeqEditWidget::setCellSize $cellSize $cellSize
        } elseif {$cellSize >= 10 && $cellSize <= 28} {
            set cellSize [expr $cellSize+2]
            ::SeqEditWidget::setCellSize $cellSize $cellSize
        }
    }

    proc menu_zoom {value} {

        variable w
        variable cellSize

        set cellSize $value
        ::SeqEditWidget::setCellSize $cellSize $cellSize
    }

    proc menu_zoomOut {} {

        variable w
        variable cellSize

        if {$cellSize > 4 && $cellSize <= 10} {
            set cellSize [expr $cellSize-1]
            ::SeqEditWidget::setCellSize $cellSize $cellSize
        } elseif {$cellSize >= 10} {
            set cellSize [expr $cellSize-2]
            ::SeqEditWidget::setCellSize $cellSize $cellSize
        }
    }
}

# Viewpoint save/restore code from vmdmovie1.1
proc ::MultiSeq::saveViewpoint {} {

    variable save_viewpoints

    if [info exists save_viewpoints] {unset save_viewpoints}

    # get the current matricies
    foreach mol [list_molids] {
        set save_viewpoints($mol) [molinfo $mol get { center_matrix rotate_matrix scale_matrix global_matrix }]
    }

    return
}


# 
proc ::MultiSeq::restoreViewpoint {} {

    variable save_viewpoints
    foreach mol [list_molids] {
        if [info exists save_viewpoints($mol)] {
          molinfo $mol set { center_matrix rotate_matrix scale_matrix global_matrix } $save_viewpoints($mol)
        }
    }

    return
}


# Returns a simple list of molids we're using from the molids array
proc ::MultiSeq::list_molids { } {
  variable molids
  set tmplist {}

  for {set i 0} {$i < [array size molids]} {incr i} {
    lappend tmplist $molids($i)
  }

    return $tmplist
}

# view_change_render.tcl
#
# A script to save current viewpoints and animate
# a smooth 'camera move' between them,
# rendering each frame to a numbered .rgb file
#  (Can also do the 'camera moves' without
#   writing files.)
#
# Barry Isralewitz, 2003-Oct-13
# barryi@ks.uiuc.edu
#
# Warning: this script does the math cheaply, i.e. only interpolates 
# transformation matrices, many sorts of extreme viewpoint changes will look
# odd to bizzare
#
# Usage:
# In the vmd console type 
#       save_vp 1
# to save your current viewpoint
#
# type
#       retrieve_vp 1  
# to retrieve that viewpoint  (You can replace '1' with an integer < 10000)
#
# After you've saved more than 1 viewpoint
#  move_vp_render 1 8 200 /tmp myMove 25 smooth
# will move from viewpoint 1 to viewpoint 8 smoothly, recording 25 frames 
# to /tmp/myMove####.rgb, with #### ranging from 0200 -- 0224. 
#
#       move_vp 1 43
# will retrieve viewpoint 1 then smoothly move to viewpoint 43.
# (move_vp does not render .rgb files, move_vp_render does) 
#  Note warning above.  Extreme moves that cause obvious protein distortion
#can be done in two or three steps.
#
# To specify animation frames used, use
#    move_vp 1 43 200
# will move from viewpoint 1 to 43 in 200 steps.  If this is not specified, a 
# default 50 frames is used
#
# To specify smooth/jerky accelaration, use
#   move_vp 1 43 200 smooth
#   move_vp_render 1 8 200 /tmp myMove 25 smooth 
#   or
#   move_vp 1 43 200 sharp
#   move_vp_render 1 8 200 /tmp myMove 25 sharp 
#
# the 'smooth' option accelerates and deccelrates the transformation
# the 'sharp' option gives constant velocity
# 
# To write viewpoints to a file, 
# write_vps my_vp_file.tcl
# 
# viewpoints with integer numbers 0-10000 are saved   
#
# To retrieve viewpoints from a file,
# source my_vp_file.tcl  
#
#
proc scale_mat {mat scaling} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr $scaling * [lindex [lindex [lindex $mat 0] $i] $j] ]
    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}

proc div_mat {mat1 mat2} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr  (0.0 + [lindex [lindex [lindex $mat1 0] $i] $j]) / ( [lindex [lindex [lindex $mat2 0] $i] $j] )]

    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}

proc sub_mat {mat1 mat2} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr  (0.0 + [lindex [lindex [lindex $mat1 0] $i] $j]) - ( [lindex [lindex [lindex $mat2 0] $i] $j] )]

    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}

proc power_mat {mat thePower} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr pow( [lindex [lindex [lindex $mat 0] $i] $j], $thePower)]
    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}

proc mult_mat {mat1 mat2} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr  (0.0 + [lindex [lindex [lindex $mat1 0] $i] $j]) * [lindex [lindex [lindex $mat2 0] $i] $j] ]
    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}

proc add_mat {mat1 mat2} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr  (0.0 + [lindex [lindex [lindex $mat1 0] $i] $j]) + [lindex [lindex [lindex $mat2 0] $i] $j] ]
    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}

proc save_vp {view_num} {
  global viewpoints
  if [info exists viewpoints($view_num)] {unset viewpoints($view_num)}
  # get the current matricies
  foreach mol [molinfo list] {
    set viewpoints($view_num,$mol,0) [molinfo $mol get rotate_matrix]
    set viewpoints($view_num,$mol,1) [molinfo $mol get center_matrix]
    set viewpoints($view_num,$mol,2) [molinfo $mol get scale_matrix]
    set viewpoints($view_num,$mol,3) [molinfo $mol get global_matrix]
  }
} 

proc justify {justify_frame} {
  if { $justify_frame < 1 } { 
    set frametext "0000"
  } elseif { $justify_frame < 10 } {
    set frametext "000$justify_frame"
  } elseif {$justify_frame < 100} {
    set frametext "00$justify_frame"
  } elseif {$justify_frame <1000} {
    set frametext "0$justify_frame"
  }  else {
    set frametext $justify_frame
  }
  return $frametext

}

proc retrieve_vp {view_num} {
  global viewpoints
  foreach mol [molinfo list] {
    if [info exists viewpoints($view_num,$mol,0)] {
      molinfo $mol set rotate_matrix   $viewpoints($view_num,$mol,0)
      molinfo $mol set center_matrix   $viewpoints($view_num,$mol,1)
      molinfo $mol set scale_matrix   $viewpoints($view_num,$mol,2)
      molinfo $mol set global_matrix   $viewpoints($view_num,$mol,3)
    } else {
      #puts "View $view_num was not saved"}
  }
}


### SimpleEdit functions

proc ::MultiSeq::textview_doSimpleEdit { } {
  variable filename

  #if {[exportSimpleEditorFile $filename] == 1} {
  #  if {[invokeEditor $filename] == 1} {
  #    if {[importSimpleEditorFile $filename] == 1} {
  #      return 1
  #    }
  #    return 0
  #  }
  #  return 0
  #}
  textview_exportSimpleEditorFile $filename
  set editWindow [multitext]
  $editWindow openfile $filename
  set windowHandle [$editWindow getWindowHandle]
#  puts "windowHandle is $windowHandle"
#  set editWindow [textview_tk]
#  ::TextView::textview $filename

  bind $windowHandle <Destroy> "::MultiSeq::textview_importSimpleEditorFile $filename"

}

# Exports the simple editor file.
# args: path to the file to write
# return: 1 if the file is successfully written, or 0 on error
proc ::MultiSeq::textview_exportSimpleEditorFile { path } {
  # first try to open the output file
  if { [catch {set fd [open $path "w"]}] > 0 } {
    return 0
  }

  set groups [::SeqEditWidget::getGroups]

  # write out the data...
  foreach group $groups {
    puts $fd "# $group"
    foreach seq [::SeqEditWidget::getSequencesInGroup $group] {
      puts $fd "[::SeqData::getName $seq]\t[join [::SeqData::getSeq $seq] {}]"
    }
  }

  close $fd

  return 1
}

# Imports the simple editor file after editing.
# args: path to file to read
# return: 1 if file is successfully read, or 0 on error
proc ::MultiSeq::textview_importSimpleEditorFile { path } {
#   puts "multiseq.tcl.textview_importSimpleEditorFile.start path: $path"
  # first try to open the input file
  if { [catch {set fd [open $path "r"]}] > 0 } {
    return 0
  }

  # read the file, overwriting sequences where names match --
  # which should be all sequences, but this makes it safe against
  # some kinds of weirdness
  set editorSequences [::SeqEditWidget::getSequences]
  set updatedSequences {}
  while {![eof $fd]} {
    set line [gets $fd]

    # Make sure line isn't a comment
    if { [regexp {^#.*$} $line] == 0 } {
      # simple 2-part line -- name and sequence
      set lineparts [split $line]
      set name [lindex $lineparts 0]
      set seq [split [lindex $lineparts 1] {}]
      set matchingSeqID -1

      foreach editorSequenceID $editorSequences {
        if {[::SeqData::getName $editorSequenceID] == $name} {
          set matchingSeqID $editorSequenceID
          lappend updatedSequences $matchingSeqID

          break
        }
      }
      if {$matchingSeqID != -1} {
        ::SeqData::setSeq $matchingSeqID $seq
      }
    }
  }

  close $fd
  ::SeqEditWidget::updateSequences $updatedSequences

  return 1
}


# Runs the editor
proc ::MultiSeq::textview_invokeEditor { path } {
  puts "MultiSeq) spawning internal text editor"
  return [::TextView::textview "$path"]

}

# Sets up the simple edit temp file
proc ::MultiSeq::textview_setFilename { dir prefix } {
  variable filename

  set filename ""

  append filename $dir "/" $prefix ".simpleEdit"
}


