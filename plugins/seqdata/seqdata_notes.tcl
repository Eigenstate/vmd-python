############################################################################
#cr
#cr         (C) Copyright 1995-2004 The Board of Trustees of the
#cr                  University of Illinois
#cr                   All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#      $RCSfile: seqdata_notes.tcl,v $
#      $Author: kvandivo $      $Locker:  $          $State: Exp $
#      $Revision: 1.7 $      $Date: 2011/02/08 19:01:39 $
#
############################################################################

package provide seqdata 1.1

# Declare global variables for this package.
namespace eval ::SeqData::Notes {

   # Export the package functions.
   namespace export showEditNotesDialog

   # Dialog management variables.
   variable w
   variable oldFocus
   variable oldGrab
   variable grabStatus
   
   # Variable for indicating the user is finished choosing the options.
   variable finished
   
   # Whether the info was changed.
   variable changed
   
   # The current sequence id.
   variable sequenceID
   
   # Secondary structure file
   variable ssfilename
   variable fileExist

# ---------------------------------------------------------------------------
   # Creates dialog that allows user to modify listing of groups that 
   # should appear in the editor.
   # args:    groups - A list of the groups currently in the editor.
   # return:   New name of sequence if changed, otherwise an empty string.
   proc showEditNotesDialog {parent a_sequenceID} {
   
      variable w
      variable oldFocus
      variable oldGrab
      variable grabStatus
      variable finished
      variable changed
      variable sequenceID
      variable type
      variable currentType
	  variable ssfilename
	  variable fileExist
      set finished 0
      set changed 0
      set fileExist 0
	  set sequenceID $a_sequenceID
      set type [::SeqData::getType $sequenceID]
      if { $type == "protein" } { set currentType "Protein" }
      if { $type == "dna" } { set currentType "DNA" }
      if { $type == "rna" } { set currentType "RNA" }
     
      # check to see if we already have a 'w'.  If so, get rid of it
      # other code in package doesn't work for multiple windows
      if {[info exists w] && $w != "" && [winfo exists ".editseqinfo" ] } {
         destroy $w
      } 

      # Create a new top level window.
      set w [createModalDialog ".editseqinfo" "Edit Sequence Information"]
      
      # Create the components.
      frame $w.center
         label $w.center.lname -text "Sequence Name:"
         entry $w.center.name -width 30
         label $w.center.lscientific -text "Source Organism:"
         entry $w.center.scientific -width 30
         label $w.center.lcommon -text "Common Name:"
         label $w.center.ltype -text "Sequence Type"
         tk_optionMenu $w.center.type ::SeqData::Notes::currentType \
                                                    "Protein" "RNA" "DNA"

         
         entry $w.center.common -width 30
         label $w.center.lec -text "EC Number:"
         entry $w.center.ec -width 30
         label $w.center.lecd -text "EC Description:"
         entry $w.center.ecd -width 30
         label $w.center.ldesc -text "Description:"
         frame $w.center.g3 -relief sunken -borderwidth 1
            text $w.center.g3.desc -width 30 -height 4 \
                                   -yscrollcommand "$w.center.g3.yscroll set"
            scrollbar $w.center.g3.yscroll -command "$w.center.g3.desc yview"
         label $w.center.lsources -text "Data Sources:"
         frame $w.center.g4 -relief sunken -borderwidth 1
            text $w.center.g4.sources -width 30 -height 5 \
                                    -yscrollcommand "$w.center.g4.yscroll set"
            scrollbar $w.center.g4.yscroll -command "$w.center.g4.sources yview"
         label $w.center.llineage -text "Lineage:"
         frame $w.center.g1 -relief sunken -borderwidth 1
            text $w.center.g1.lineage -width 30 -height 6 \
                                    -yscrollcommand "$w.center.g1.yscroll set"
            scrollbar $w.center.g1.yscroll -command "$w.center.g1.lineage yview"
         label $w.center.lnotes -text "Notes"
         frame $w.center.g2 -relief sunken -borderwidth 1
            text $w.center.g2.notes -width 46 -height 16 \
                                    -yscrollcommand "$w.center.g2.yscroll set"
            scrollbar $w.center.g2.yscroll -command "$w.center.g2.notes yview"
         label $w.center.lss -text "Secondary Structure"
         button $w.center.calcss -text "Predict" -pady 2 \
                      -command "::SeqData::Notes::but_predictSecondaryStructure"
         frame $w.center.g5 -relief sunken -borderwidth 1
            text $w.center.g5.ss -width 30 -height 4 \
                                    -yscrollcommand "$w.center.g5.yscroll set"
            scrollbar $w.center.g5.yscroll -command "$w.center.g5.ss yview"
		 button $w.center.loadss -text "Load" -pady 2 \
                      -command "::SeqData::Notes::but_loadSecondaryStructure"
		 entry $w.center.ssfilename -textvariable \
                             "::SeqData::Notes::ssfilename" -width 20
		 button $w.center.ssfilenamebrs -text "Browse..." -pady 2 \
                       -command "::SeqData::Notes::but_browsefile"
      frame $w.bottom
         frame $w.bottom.buttons
            button $w.bottom.buttons.accept -text "OK" -pady 2 \
                                    -command "::SeqData::Notes::but_ok"
            button $w.bottom.buttons.cancel -text "Cancel" -pady 2 \
                                    -command "::SeqData::Notes::but_cancel"
            bind $w <Return> {::SeqData::Notes::but_ok}
            bind $w <Escape> {::SeqData::Notes::but_cancel}
      
      # Layout the components.
      pack $w.center           -fill both -expand true -side top -padx 5 -pady 5
      grid $w.center.lname         -column 1 -row 1 -sticky w
      grid $w.center.name          -column 2 -row 1 -sticky w
      grid $w.center.lscientific     -column 1 -row 2 -sticky w
      grid $w.center.scientific      -column 2 -row 2 -sticky w
      grid $w.center.lcommon        -column 1 -row 3 -sticky w
      grid $w.center.common         -column 2 -row 3 -sticky w
      grid $w.center.ltype         -column 1 -row 4 -sticky w
      grid $w.center.type          -column 2 -row 4 -sticky w
      grid $w.center.lec           -column 1 -row 5 -sticky w
      grid $w.center.ec            -column 2 -row 5 -sticky w
      grid $w.center.lecd          -column 1 -row 6 -sticky w
      grid $w.center.ecd           -column 2 -row 6 -sticky w
      grid $w.center.ldesc         -column 1 -row 7 -sticky nw
      grid $w.center.g3            -column 2 -row 7 -sticky w -pady 4
      pack $w.center.g3.desc        -fill both -expand true -side left
      pack $w.center.g3.yscroll      -side right -fill y
      grid $w.center.lsources       -column 1 -row 8 -sticky nw
      grid $w.center.g4            -column 2 -row 8 -sticky w -pady 4
      pack $w.center.g4.sources      -fill both -expand true -side left -fill x
      pack $w.center.g4.yscroll      -side right -fill y
      grid $w.center.llineage       -column 1 -row 9 -sticky nw
      grid $w.center.g1            -column 2 -row 9 -sticky w -pady 4
      pack $w.center.g1.lineage      -fill both -expand true -side left
      pack $w.center.g1.yscroll      -side right -fill y
      grid $w.center.lnotes         -column 1 -row 10 -sticky w -columnspan 2
      grid $w.center.g2            -column 1 -row 11 -sticky w -columnspan 2
      pack $w.center.g2.notes       -fill both -expand true -side left
      pack $w.center.g2.yscroll      -side right -fill y
      grid $w.center.lss           -column 1 -row 12 -sticky w -columnspan 2
      grid $w.center.calcss         -column 1 -row 13 -sticky w
      grid $w.center.g5            -column 2 -row 13 -sticky w
      pack $w.center.g5.ss         -fill both -expand true -side left
      pack $w.center.g5.yscroll      -side right -fill y
	  grid $w.center.loadss         -column 1 -row 14 -sticky w
	  grid $w.center.ssfilename		 -column 2 -row 14 -sticky w
	  grid $w.center.ssfilenamebrs	 -column 2 -row 14 -sticky e
      pack $w.bottom              -fill x -side bottom
      pack $w.bottom.buttons        -side bottom
      pack $w.bottom.buttons.accept   -side left -padx 5 -pady 5
      pack $w.bottom.buttons.cancel   -side right -padx 5 -pady 5
     

      # Fill in the data.
      $w.center.name insert 0 [::SeqData::getName $sequenceID]
      $w.center.scientific insert 0 [::SeqData::getScientificName $sequenceID]
      $w.center.common insert 0 [::SeqData::getCommonName $sequenceID]
      $w.center.g3.desc insert end [::SeqData::getAnnotation $sequenceID \
                                                              description]
      foreach sequenceSource [::SeqData::getSources $sequenceID] {
         $w.center.g4.sources insert end \
             "[lindex $sequenceSource 0]=[join [lindex $sequenceSource 1] ","]"
         $w.center.g4.sources insert end "\n"
      }

      set lineage [::SeqData::getLineage $sequenceID]
      foreach field $lineage {
         $w.center.g1.lineage insert end $field\n
      }
      $w.center.ec insert 0 [::SeqData::getEnzymeCommissionNumber $sequenceID]
      $w.center.ecd insert 0 [::SeqData::getEnzymeCommissionDescription \
                                                             $sequenceID]
      $w.center.g2.notes insert end [::SeqData::getAnnotation $sequenceID notes]
      $w.center.g5.ss insert end [join [::SeqData::getSecondaryStructure \
                                                             $sequenceID] ""]

      # Bind the window closing event.
      bind $w <Destroy> {::SeqData::Notes::but_cancel}
      
      # Center the dialog.
      centerDialog $parent
      
      # Wait for the user to interact with the dialog.
      tkwait variable ::SeqData::Notes::finished
      #puts "Size is [winfo reqwidth $w] [winfo reqheight $w]"

      # Destroy the dialog.
      destroyDialog      
      
      # Return the groups.
      return $changed
   }
   
# ---------------------------------------------------------------------------
   # Creates new modal dialog window given a prefix for window name and 
   # a title for the dialog.
   # args:    prefix - prefix for window name of dialog. should start with ".".
   #         dialogTitle - The title for the dialog.
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

      # next line added to make it work so that we only have one
      # seqedit notes window (was broken for multiple windows)
      set unique "" 

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
      focus $w
      return $w
   }
   
# ---------------------------------------------------------------------------
   # Centers the dialog.
   proc centerDialog {{parent ""}} {
      
      variable w
      
      # Set the width and height, since calculating doesn't work properly.
      set width 358
      set height [expr 662+22]
      
      # Figure out the x and y position.
      if {$parent != ""} {
         set cx [expr {int ([winfo rootx $parent] + [winfo width $parent] / 2)}]
         set cy [expr {int ([winfo rooty $parent] + \
                                                [winfo height $parent] / 2)}]
         set x [expr {$cx - int ($width / 2)}]
         set y [expr {$cy - int ($height / 2)}]
         
      } else {
         set x [expr {int (([winfo screenwidth $w] - [winfo reqwidth $w]) / 2)}]
         set y [expr {int (([winfo screenheight $w] - \
                                                [winfo reqheight $w]) / 2)}]
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
   
# ---------------------------------------------------------------------------
   # Destroys the dialog. This method releases the dialog resources 
   # and restores the system handlers.
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
   
# ---------------------------------------------------------------------------
   proc but_ok {} {
   
      variable w
      variable finished
      variable changed
      variable sequenceID
      variable type
      variable currentType
      
      # If we changed the name, save it.
      if {[$w.center.name get] != [::SeqData::getName $sequenceID]} {
         set newName [string trim [$w.center.name get]]
         regsub -all {\s} $newName "_" newName
         ::SeqData::setName $sequenceID $newName
         ::SeqData::addAnnotation $sequenceID "name" $newName
         set changed 1
      }
      
      #If we changed the type, save it"
      if {[string tolower $currentType] != [::SeqData::getType $sequenceID]} {
         ::SeqData::setType $sequenceID [string tolower $currentType]
      }

      # If we changed the EC number, save it.
      if {[$w.center.ec get] != \
                           [::SeqData::getEnzymeCommissionNumber $sequenceID]} {
         ::SeqData::addAnnotation $sequenceID "ec-number" \
                                              [string trim [$w.center.ec get]]
         set changed 1
      }
      
      # If we entered an EC description without an EC number.
      if {[$w.center.ec get] == "" && \
          [$w.center.ecd get] != [::SeqData::getEnzymeCommissionDescription \
                                                               $sequenceID]} {
         ::SeqData::addAnnotation $sequenceID "ec-description" \
                                            [string trim [$w.center.ecd get]]
         set changed 1
      }
      
      # If we changed the description, notes,...
      if {[$w.center.g3.desc get 1.0 end] != \
                       [::SeqData::getAnnotation $sequenceID "description"]} {
         ::SeqData::addAnnotation $sequenceID "description" \
                                  [string trim [$w.center.g3.desc get 1.0 end]]
         set changed 1
      }
      if {[$w.center.g2.notes get 1.0 end] != \
                              [::SeqData::getAnnotation $sequenceID "notes"]} {
         ::SeqData::addAnnotation $sequenceID "notes" \
                                 [string trim [$w.center.g2.notes get 1.0 end]]
         set changed 1
      }
      if {[$w.center.scientific get] != \
                    [::SeqData::getAnnotation $sequenceID "scientific-name"]} {
         ::SeqData::addAnnotation $sequenceID "scientific-name" \
                                       [string trim [$w.center.scientific get]]
         set changed 1
      }
      if {[$w.center.common get] != \
                         [::SeqData::getAnnotation $sequenceID "common-name"]} {
         ::SeqData::addAnnotation $sequenceID "common-name" \
                                           [string trim [$w.center.common get]]
         set changed 1
      }
      if {[split [string trim [$w.center.g5.ss get 1.0 end]] ""] != \
                              [::SeqData::getSecondaryStructure $sequenceID]} {
         ::SeqData::setSecondaryStructure $sequenceID \
                                    [split [string toupper [string trim \
                                    [$w.center.g5.ss get 1.0 end]]] ""]
         set changed 1
      }

      # Close the window.
      set finished 1
   }                              
   
# ---------------------------------------------------------------------------
   proc but_cancel {} {
   
      variable finished
   
      # Close the window.   
      set finished 0
   }

# ---------------------------------------------------------------------------
   proc but_predictSecondaryStructure {} {
   
      variable w
      variable sequenceID
      
      # If psipred is available and configured, use it.
      set checkPack [::Psipred::checkPackageConfiguration]
      if {$checkPack == {}} {
         $w.center.g5.ss delete 1.0 end
         $w.center.g5.ss insert end [join \
                        [::Psipred::calculateSecondaryStructure $sequenceID] ""]
      } else {
         tk_messageBox -type ok -icon error -parent $w -title "Error" \
              -message "PSIPred hasn't been properly configured:\n$checkPack"
      }

   }
# ---------------------------------------------------------------------------
	proc but_loadSecondaryStructure {} {
	  
	  variable w
      variable sequenceID
	  variable ssfilename
	  variable fileExist
	  
	  if {$fileExist} {
         # display the secondary structure
         $w.center.g5.ss delete 1.0 end
		 $w.center.g5.ss insert end [join \
                       [::SeqData::loadSecondaryStructure $sequenceID $ssfilename] ""]
         #$w.center.g5.ss insert end [join {"((((...))))"} ""]
		} else {
			tk_messageBox -type ok -icon error -parent $w -title "Error" \
                  -message "You have chosen to import from a file, but didn't give a file name"
            return
	  }
	}
# ---------------------------------------------------------------------------
	proc but_browsefile {} {
	  
	  variable w
      variable sequenceID
	  variable ssfilename
	  variable fileExist
	  
	  # load secondary structure from file
	  set ssfilename [tk_getOpenFile -multiple 0 -filetypes {\
		  								   {{BRACKET Files} {.bracket}} \
                                          {{All Files} * }} \
                                          -title "Load Secondary Structure"]
      if {$ssfilename != {}} {
         #set choices(filenames) "\"[join $filenames \",\"]\""
		 #set choices(filenames) $filenames
		 set fileExist 1
      }
	}
# ---------------------------------------------------------------------------
}

