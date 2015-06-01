############################################################################
#cr
#cr            (C) Copyright 1995-2004 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: clustalw_preferences.tcl,v 1.6 2015/05/20 21:53:42 kvandivo Exp $
#

package provide seqedit 1.1

# Declare global variables for this package.
namespace eval SeqEdit::SeqAlign {

    # Export the package functions.
    namespace export showSeqAlignOptionsDialog

    # Dialog management variables.
    variable w
    variable oldFocus
    variable oldGrab
    variable grabStatus
    
    # Variable for indicating the user is finished choosing the options.
    variable finished
    
    # The options.
    variable seqAlignOptions
    array unset seqAlignOptions 
    
    # Variables used by the widgets.
    variable alignmentType "multiple"
    variable multipleAlignmentType "all"
    variable seqAlignProg "clustalw"
    
    # Creates a dialog to get the user's options for running SeqAlign.
    # args:     profiles - A list of the profiles that are currently available for profile alignment.
    # return:   An array containing the user's choices.
    proc showSeqAlignOptionsDialog {parent profiles} {
    
        variable w
        variable oldFocus
        variable oldGrab
        variable grabStatus
        variable finished
        variable seqAlignOptions
        set finished 0
        array unset seqAlignOptions 
    
        # Create a new top level window.
        set w [createModalDialog ".seqAlignoptions" "Sequence Alignment Options"]
        
        # Create the components.
        frame $w.center

           frame $w.center.g0 -relief sunken -borderwidth 1
              labelframe $w.center.g0.a -relief raised \
                          -borderwidth 1 -text "Alignment Program"
                radiobutton $w.center.g0.a.clustalw -text "ClustalW" \
                        -variable "::SeqEdit::SeqAlign::seqAlignProg" \
                        -value "clustalw"
#                if { [::MultiSeqDialog::getDirectory "mafft"] != "" } {
                   radiobutton $w.center.g0.a.mafft -text "MAFFT" \
                        -variable "::SeqEdit::SeqAlign::seqAlignProg" \
                        -value "mafft"
#                } else {
#                   radiobutton $w.center.g0.a.mafft \
#                        -text "MAFFT (configure in preferences)" \
#                        -variable "::SeqEdit::SeqAlign::seqAlignProg" \
#                        -value "mafft" -state disabled
#
#                }

            radiobutton $w.center.multiple -text "Multiple Alignment" -variable "::SeqEdit::SeqAlign::alignmentType" -value "multiple"
            frame $w.center.g1
                frame $w.center.g1.s -width 25
                frame $w.center.g1.a -relief sunken -borderwidth 1
                    frame $w.center.g1.a.b -relief raised -borderwidth 1
                        radiobutton $w.center.g1.a.b.all -text "Align All Sequences" -variable "::SeqEdit::SeqAlign::multipleAlignmentType" -value "all"
                        radiobutton $w.center.g1.a.b.marked -text "Align Marked Sequences" -variable "::SeqEdit::SeqAlign::multipleAlignmentType" -value "marked"
                        radiobutton $w.center.g1.a.b.selected -text "Align Selected Regions" -variable "::SeqEdit::SeqAlign::multipleAlignmentType" -value "selected"
            radiobutton $w.center.seqpro -text "Profile/Sequence Alignment" -variable "::SeqEdit::SeqAlign::alignmentType" -value "sequence-profile"
            frame $w.center.g2
                frame $w.center.g2.s -width 25
                frame $w.center.g2.a -relief sunken -borderwidth 1
                    frame $w.center.g2.a.b -relief raised -borderwidth 1 -width 50
                        label $w.center.g2.a.b.label -text "Align marked sequences to group:"
                        frame $w.center.g2.a.b.s
                            listbox $w.center.g2.a.b.s.groups -selectmode single -exportselection FALSE -height 6 -width 30 -yscrollcommand "$w.center.g2.a.b.s.scroll set"
                            scrollbar $w.center.g2.a.b.s.scroll -command "$w.center.g2.a.b.s.groups yview"
                            foreach profile $profiles {
                                $w.center.g2.a.b.s.groups insert end $profile
                            }
            radiobutton $w.center.propro -text "Profile/Profile Alignment" -variable "::SeqEdit::SeqAlign::alignmentType" -value "profile-profile"
            frame $w.center.g3
                frame $w.center.g3.s -width 25
                frame $w.center.g3.a -relief sunken -borderwidth 1
                    frame $w.center.g3.a.b -relief raised -borderwidth 1 -width 50
                        label $w.center.g3.a.b.label -text "Select two groups to align:"
                        frame $w.center.g3.a.b.s
                            listbox $w.center.g3.a.b.s.groups -selectmode multiple -exportselection FALSE -height 6 -width 30 -yscrollcommand "$w.center.g3.a.b.s.scroll set"
                            scrollbar $w.center.g3.a.b.s.scroll -command "$w.center.g3.a.b.s.groups yview"
                            foreach profile $profiles {
                                $w.center.g3.a.b.s.groups insert end $profile
                            }
        frame $w.bottom
            frame $w.bottom.buttons
                button $w.bottom.buttons.accept -text "OK" -pady 2 -command "::SeqEdit::SeqAlign::but_ok"
                button $w.bottom.buttons.cancel -text "Cancel" -pady 2 -command "::SeqEdit::SeqAlign::but_cancel"
                bind $w <Return> {::SeqEdit::SeqAlign::but_ok}
                bind $w <Escape> {::SeqEdit::SeqAlign::but_cancel}
        
        # Layout the components.
        pack $w.center                  -fill both -expand true -side top -padx 5 -pady 5


        grid $w.center.g0               -column 1 -row 1 -sticky w
        pack $w.center.g0.a           -fill both -expand true -side left
        grid $w.center.g0.a.clustalw       -column 1 -row 1 -sticky w
        grid $w.center.g0.a.mafft    -column 1 -row 2 -sticky w


        grid $w.center.multiple         -column 1 -row 2 -sticky w
        grid $w.center.g1               -column 1 -row 3 -sticky w
        pack $w.center.g1.s             -fill both -expand true -side left 
        pack $w.center.g1.a             -fill both -expand true -side left -padx 5 -pady 5
        pack $w.center.g1.a.b           -fill both -expand true -side left
        grid $w.center.g1.a.b.all       -column 1 -row 1 -sticky w
        grid $w.center.g1.a.b.marked    -column 1 -row 2 -sticky w
        grid $w.center.g1.a.b.selected  -column 1 -row 3 -sticky w
        grid $w.center.seqpro           -column 1 -row 4 -sticky w
        grid $w.center.g2               -column 1 -row 5 -sticky w
        pack $w.center.g2.s             -fill both -expand true -side left 
        pack $w.center.g2.a             -fill both -expand true -side left -padx 5 -pady 5
        pack $w.center.g2.a.b           -fill both -expand true -side left
        grid $w.center.g2.a.b.label     -column 1 -row 1 -sticky w
        grid $w.center.g2.a.b.s         -column 1 -row 2 -sticky w
        pack $w.center.g2.a.b.s.groups  -fill both -expand true -side left -padx 5 -pady 5
        pack $w.center.g2.a.b.s.scroll  -side right -fill y
        grid $w.center.propro           -column 1 -row 6 -sticky w
        grid $w.center.g3               -column 1 -row 7 -sticky w
        pack $w.center.g3.s             -fill both -expand true -side left 
        pack $w.center.g3.a             -fill both -expand true -side left -padx 5 -pady 5
        pack $w.center.g3.a.b           -fill both -expand true -side left
        grid $w.center.g3.a.b.label     -column 1 -row 1 -sticky w
        grid $w.center.g3.a.b.s         -column 1 -row 2 -sticky w
        pack $w.center.g3.a.b.s.groups  -fill both -expand true -side left -padx 5 -pady 5
        pack $w.center.g3.a.b.s.scroll  -side right -fill y
        
        pack $w.bottom                  -fill x -side bottom
        pack $w.bottom.buttons          -side bottom
        pack $w.bottom.buttons.accept   -side left -padx 5 -pady 5
        pack $w.bottom.buttons.cancel   -side right -padx 5 -pady 5

        # Bind the window closing event.
        bind $w <Destroy> {::SeqEdit::SeqAlign::but_cancel}
        
        # Center the dialog.
        centerDialog $parent
        
        # Wait for the user to interact with the dialog.
        tkwait variable ::SeqEdit::SeqAlign::finished
        #puts "Size is [winfo reqwidth $w] [winfo reqheight $w]"

        # Destroy the dialog.
        destroyDialog        
        
        # Return the options.
        if {$finished == 1} {
            return [array get seqAlignOptions]
        } else {
            return {}
        }
    }
   
# ------------------------------------------------------
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
    
# ------------------------------------------------------
    # Centers the dialog.
    proc centerDialog {{parent ""}} {
        
        variable w
        
        # Set the width and height, since calculating doesn't work properly.
        set width 287
        set height [expr 420+22]
        
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
    
# ------------------------------------------------------
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
    
# ------------------------------------------------------
    proc but_ok {} {
    
        variable w
        variable finished
        variable seqAlignOptions
        variable alignmentType
        variable seqAlignProg
        variable multipleAlignmentType

        # Save the options.        
        set seqAlignOptions(alignmentType) $alignmentType;
        set seqAlignOptions(seqAlignProg) $seqAlignProg
        set seqAlignOptions(multipleAlignmentType) $multipleAlignmentType;
        
        # If we are doing a sequence-profile alignment, get the data.
        if {$alignmentType == "sequence-profile"} {
            set selected [$w.center.g2.a.b.s.groups curselection]
            if {[llength $selected] == 1} {
                set seqAlignOptions(profile) [$w.center.g2.a.b.s.groups get [lindex $selected 0]]
            } else {
                tk_messageBox -type ok -icon error -parent $w -title "Error" -message "You must select the group containing the profile to which you would like to align the marked sequences."
                return
            }
        }

        # If we are doing a profile-profile alignment, get the data.
        if {$alignmentType == "profile-profile"} {
            set selected [$w.center.g3.a.b.s.groups curselection]
            #puts "Selected group:$selected:"
            if {[llength $selected] == 2} {
                set seqAlignOptions(profile1) [$w.center.g3.a.b.s.groups get [lindex $selected 0]]
                set seqAlignOptions(profile2) [$w.center.g3.a.b.s.groups get [lindex $selected 1]]
            } else {
                tk_messageBox -type ok -icon error -parent $w -title "Error" -message "You must select the two groups containing the two profiles that you would like to combine."
                return
            }
        }

        # Close the window.
        set finished 1
    }
    
# ------------------------------------------------------
    proc but_cancel {} {
    
        variable finished
    
        # Close the window.    
        set finished 0
    }
# ------------------------------------------------------
}

