############################################################################
#cr
#cr            (C) Copyright 1995-2004 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: removegaps_options.tcl,v $
#       $Author: kvandivo $        $Locker:  $             $State: Exp $
#       $Revision: 1.3 $       $Date: 2009/12/15 19:57:43 $
#
############################################################################

package provide seqedit 1.1

# Declare global variables for this package.
namespace eval ::SeqEdit::RemoveGaps {

    # Export the package functions.
    namespace export showOptionsDialog
    
    # Dialog management variables.
    variable w
    variable oldFocus
    variable oldGrab
    variable grabStatus
    
    # Variable for indicating the user is finished choosing the options.
    variable finished 1
    
    # The user's choices.
    variable choices
    array set choices {"selectionType" "all"
                       "removalType" "redundant"}
    
    # Creates a dialog to get the user's choices.
    # args:     parent - The parent wondow for this dialog.
    # return:   An array containing the user's choices.
    proc showOptionsDialog {parent} {
    
        variable w
        variable oldFocus
        variable oldGrab
        variable grabStatus
        variable finished
        variable choices
        set finished 0
    
        # Create a new top level window.
        set w [createModalDialog ".phylotreeoptions" "Remove Gaps"]
        
        # Create the components.
        frame $w.center
            label $w.center.choice -text "Remove gaps from:"
            frame $w.center.g1 -relief sunken -borderwidth 1
                frame $w.center.g1.b -relief raised -borderwidth 1
                    radiobutton $w.center.g1.b.all -text "All Sequences" -variable "::SeqEdit::RemoveGaps::choices(selectionType)" -value "all"
                    radiobutton $w.center.g1.b.marked -text "Marked Sequences" -variable "::SeqEdit::RemoveGaps::choices(selectionType)" -value "marked"
                    radiobutton $w.center.g1.b.selected -text "Selected Regions" -variable "::SeqEdit::RemoveGaps::choices(selectionType)" -value "selected"                    
                        
            label $w.center.l1 -text "Remove these types of gaps:"
            radiobutton $w.center.redundant -text "Redundant gaps" -variable "::SeqEdit::RemoveGaps::choices(removalType)" -value "redundant"
            radiobutton $w.center.all -text "All gaps" -variable "::SeqEdit::RemoveGaps::choices(removalType)" -value "all"
            
        frame $w.bottom
            frame $w.bottom.buttons
                button $w.bottom.buttons.accept -text "OK" -pady 2 -command "::SeqEdit::RemoveGaps::but_ok"
                button $w.bottom.buttons.cancel -text "Cancel" -pady 2 -command "::SeqEdit::RemoveGaps::but_cancel"
        
        # Layout the components.
        pack $w.center                  -fill both -expand true -side top -padx 5 -pady 5
        grid $w.center.choice           -column 1 -row 1 -sticky nw -pady 8
        grid $w.center.g1               -column 2 -row 1 -sticky nw -padx 5 -pady 8
        pack $w.center.g1.b             -fill both -expand true -side left
        grid $w.center.g1.b.all             -column 1 -row 1 -sticky w
        grid $w.center.g1.b.marked          -column 1 -row 2 -sticky w
        grid $w.center.g1.b.selected        -column 1 -row 3 -sticky w
        grid $w.center.l1               -column 1 -row 3 -sticky nw -columnspan 2
        grid $w.center.redundant        -column 2 -row 4 -sticky nw
        grid $w.center.all              -column 2 -row 5 -sticky nw

        pack $w.bottom                  -fill x -side bottom
        pack $w.bottom.buttons          -side bottom
        pack $w.bottom.buttons.accept   -side left -padx 5 -pady 5
        pack $w.bottom.buttons.cancel   -side left -padx 5 -pady 5

        # Bind the window closing event.
        bind $w <Destroy> {"SeqEdit::RemoveGaps::but_cancel"}
        
        # Center the dialog.
        centerDialog $parent
        
        # Wait for the user to interact with the dialog.
        tkwait variable "::SeqEdit::RemoveGaps::finished"
        #puts "Size is [winfo reqwidth $w] [winfo reqheight $w]"
        
        # Destroy the dialog.
        destroyDialog        
        
        # Return the options.
        if {$finished == 1} {
            return [array get choices]
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
        set width 267
        set height [expr 180+22]
        
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
        variable choices

        # Close the window.
        set finished 1
    }
    
    proc but_cancel {} {
    
        variable finished
    
        # Close the window.    
        set finished 0
    }
}
