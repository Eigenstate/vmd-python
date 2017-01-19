#
# $Id: modal.tcl,v 1.2 2013/04/15 17:30:43 johns Exp $
#

proc ::Rmsdvt::Inputbox {master w title text {default {}} {ok_btn "Apply"} {cancel_btn "Cancel"} } {
    
    global frmReturn
    #set frmReturn ""
    set retval ""
    
    catch {destroy $w}
    toplevel $w -class inputdlg
    wm title $w $title
    wm iconname $w $title
    wm protocol $w WM_DELETE_WINDOW { }
    wm resizable $w 0 0
    wm transient $w $master
    
    option add *$w.*borderWidth 1
    option add *$w.*Button.padY 2
    option add *$w.*Menubutton.padY 0  
        
    
    # Create dialog
    pack [frame $w.bot -relief raised -bd 2] -side bottom -fill both
    pack [frame $w.top -relief raised -bd 2] -side top -fill both -expand 1
    option add *Inputbox.msg.wrapLength 3i widgetDefault
    
    # Set dialog text
    label $w.msg -justify left -text $text;#-font {Times 18}
    
    # Create input box
    entry $w.input -textvariable [namespace current]::retval -relief sunken -bd 1
    $w.input delete 0 end
    $w.input insert end $default
    bind $w.input <Return> "$w.b0 invoke"
    bind $w <Destroy> {set frmReturn {}}
    
    
    pack $w.msg -in $w.top -side top -expand 1 -fill both -padx 3m -pady 3m
    pack $w.input -in $w.top -side top -expand 1 -fill x -padx 3m -pady 3m
    
    # Buttons
    button $w.b0 -text $ok_btn -command "set frmReturn \[$w.input get\]"
    button $w.b1 -text $cancel_btn -command {set frmReturn "can_zel"}
    
    grid $w.b0 -in $w.bot -column 0 -row 0 -stick nswe -padx 10
    grid $w.b1 -in $w.bot -column 1 -row 0 -stick nswe -padx 10
    
    wm withdraw $w
    update idletasks
    
    
    set x [expr [winfo screenwidth $w]/2 - [winfo reqwidth $w]/2 - [winfo vrootx [winfo parent $w]]]
    set y [expr [winfo screenheight $w]/2 - [winfo reqheight $w]/2 - [winfo vrooty [winfo parent $w]]]
    wm geometry $w +$x+$y
    wm deiconify $w
    
    set oldfocus [focus]
    set oldgrab [grab current $w]
    if {$oldgrab != ""} {
        set grabstatus [grab status $oldgrab]
    }
    
    grab $w
    focus $w.input
    tkwait variable frmReturn
    
    set retval $frmReturn
    
    catch {focus $oldfocus}
    catch {
        bind $w Destroy {}
        destroy $w
    }
    
    if {$oldgrab != ""} {
        if {$grabstatus == "global"} {
            grab -global $oldgrab
        } else {
            grab $oldgrab
        }
    }
    

    #puts "Returning $retval"
    return $retval
}
