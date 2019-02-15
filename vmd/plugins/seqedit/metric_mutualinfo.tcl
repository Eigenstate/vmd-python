# University of Illinois Open Source License
# Copyright 2007-2008 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: metric_mutualinfo.tcl,v 1.5 2018/11/06 23:10:22 johns Exp $
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
# Author(s): Elijah Roberts

# This package implements a color map for the sequence editor that colors
# sequence elements based upon mutual information.

package provide seqedit 1.1
package require seqdata 1.1

# Declare global variables for this package.
namespace eval ::SeqEdit::Metric::MutualInformation {
    
    variable counts

    proc calculate {w} {
        
        set sequenceGroups {}
        array set options [showOptionsDialog $w [::SeqEditWidget::getGroups]]
        
        if {[info exists options(groups)] && [info exists options(normalize)] && [info exists options(minValue)] && [info exists options(maxGapFraction)]} {
            set sequenceGroups {}
            foreach groupName $options(groups) {
                lappend sequenceGroups [::SeqEditWidget::getSequencesInGroup $groupName]
            }

            set miValues [::Libbiokit::Entropy::calculateGroupMutualInformation $sequenceGroups 0.69314718056 $options(normalize) $options(minValue) $options(maxGapFraction) "N?"]
            performColoring $sequenceGroups $miValues
        }
    }
    
    proc performColoring {sequenceGroups miValues} {
        
        # Get the maximum value.
        set maxMiValue ""
        foreach miValue $miValues {
            if {$maxMiValue == "" || $miValue > $maxMiValue} {
                set maxMiValue $miValue
            }
        }
        
        # Get the function to map a value to an index.
        set colors {}
        set colorValueMap "[::SeqEditWidget::getColorMap]\::getColorIndexForValue"
        foreach miValue $miValues {
            if {$maxMiValue != ""} {
                lappend colors [$colorValueMap $miValue/$maxMiValue]
            } else {
                lappend colors [$colorValueMap $miValue]
            }
        }
        
        # Go through the groups.
        foreach sequenceIDs $sequenceGroups {
            foreach sequenceID $sequenceIDs {
                for {set position 0} {$position < [llength $colors]} {incr position} {
                    seq set color $sequenceID $position [lindex $colors $position]
                }
            }
        }
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
    array set options {groups {} normalize 0 minValue 0.7 maxGapFraction 0.3}
    
    proc showOptionsDialog {parent groups} {
    
        variable w
        variable oldFocus
        variable oldGrab
        variable grabStatus
        variable finished
        variable options
        set finished 0
    
        # Create a new top level window.
        set w [createModalDialog ".mutualinfooptions" "Mutual Information Options"]
        
        # Create the components.
        frame $w.center
            label $w.center.label1 -text "Select the groups to be used in the calculation:"
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
            checkbutton $w.center.normalize -text "Normalize by sequence entropy" -variable "::SeqEdit::Metric::MutualInformation::options(normalize)" -offvalue 0 -onvalue 3
            label $w.center.minValueL -text "Minimum information value to display:"
            entry $w.center.minValue -textvariable "::SeqEdit::Metric::MutualInformation::options(minValue)" -width 6
            label $w.center.maxGapFractionL -text "Maximum fraction of gaps allowed in a group:"
            entry $w.center.maxGapFraction -textvariable "::SeqEdit::Metric::MutualInformation::options(maxGapFraction)" -width 6            
        frame $w.bottom
            frame $w.bottom.buttons
                button $w.bottom.buttons.accept -text "OK" -pady 2 -command "::SeqEdit::Metric::MutualInformation::but_ok"
                button $w.bottom.buttons.cancel -text "Cancel" -pady 2 -command "::SeqEdit::Metric::MutualInformation::but_cancel"
                bind $w <Return> {::SeqEdit::Metric::MutualInformation::but_ok}
                bind $w <Escape> {::SeqEdit::Metric::MutualInformation::but_cancel}
        
        # Layout the components.
        pack $w.center                  -fill both -expand true -side top -padx 5 -pady 5
        grid $w.center.label1           -column 1 -row 1 -sticky w -columnspan 2
        grid $w.center.g1               -column 1 -row 2 -sticky w -columnspan 2
        pack $w.center.g1.groups        -fill both -expand true -side left -padx 5 -pady 5
        pack $w.center.g1.scroll        -side right -fill y
        grid $w.center.normalize        -column 1 -row 3 -sticky w -columnspan 2
        grid $w.center.minValueL        -column 1 -row 4 -sticky w
        grid $w.center.minValue         -column 2 -row 4 -sticky w -padx 5
        grid $w.center.maxGapFractionL  -column 1 -row 5 -sticky w
        grid $w.center.maxGapFraction   -column 2 -row 5 -sticky w -padx 5
        
        pack $w.bottom                  -fill x -side bottom
        pack $w.bottom.buttons          -side bottom
        pack $w.bottom.buttons.accept   -side left -padx 5 -pady 5
        pack $w.bottom.buttons.cancel   -side right -padx 5 -pady 5

        # Bind the window closing event.
        bind $w <Destroy> {::SeqEdit::Metric::MutualInformation::but_cancel}
        
        # Center the dialog.
        centerDialog $parent
        
        # Wait for the user to interact with the dialog.
        tkwait variable ::SeqEdit::Metric::MutualInformation::finished
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
        set height [expr 258+22]
        
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
            tk_messageBox -type ok -icon error -parent $w -title "Error" -message "You must select at least two groups for a mutual information calculation."
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
