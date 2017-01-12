# hello emacs this is -*- tcl -*-
#
# GUI around pbctools.
#
# (c) 2009 by Olaf Lenz <lenzo@mpip-mainz.mpg.de>
########################################################################
#
# $Id: pbcgui.tcl,v 1.5 2014/09/09 20:00:17 johns Exp $
#
# create package and namespace and default all namespace global variables.
package provide pbcgui 2.8
package require pbctools 2.8

namespace eval ::pbcgui:: {
    namespace export pbcgui

    variable w;                # handle to the base widget.

    variable molid        "0"; # molid of the molecule
    variable moltxt        "(none)"; # title the molecule

}

proc ::pbcgui::pbcgui {args} {
    variable w

    ## stolen from GofRGUI by Axel Kohlmeyer
    # main window frame
    set w .pbcgui
    catch {destroy $w}
    toplevel    $w
    wm title    $w "PBCTools GUI"
    wm iconname $w "PBCToolsGUI"
    wm minsize  $w 520 200

    # frame for settings
    set in $w.in
    labelframe $in -bd 2 -relief ridge -text "Settings:" -padx 1m -pady 1m
    pack $in -side top -fill both

    # Molecule selector
    frame $in.molid
    label $in.molid.l -text "Use Molecule:" -anchor w
    menubutton $in.molid.m -relief raised -bd 2 -direction flush \
        -text "test text" -textvariable ::pbcgui::moltxt \
        -menu $in.molid.m.menu
    menu $in.molid.m.menu -tearoff no
    pack $in.molid.l -side left
    pack $in.molid.m -side left
    pack $in.molid -side top
    grid config $in.molid.l  -column 0 -row 0 -columnspan 1 -rowspan 1 -sticky "snew"
    grid config $in.molid.m  -column 1 -row 0 -columnspan 1 -rowspan 1 -sticky "snew"
    grid columnconfigure $in.molid 0 -weight 1 -minsize 10
    grid columnconfigure $in.molid 1 -weight 3 -minsize 10

    # listen to updates in the molecule list
    UpdateMolecule
    global vmd_molecule
    trace variable vmd_molecule w ::pbcgui::UpdateMolecule
}


# callback for VMD menu entry
proc pbcgui_tk_cb {} {
  ::pbcgui::pbcgui
  return $::pbcgui::w
}

# update molecule list
proc ::pbcgui::UpdateMolecule {args} {
    variable w
    variable moltxt
    variable molid
    global vmd_molecule

    puts "UpdateMolecule was called!"

    # Update the molecule browser
    set mollist [molinfo list]
    $w.foot configure -state disabled
    $w.in.molid.m configure -state disabled
    $w.in.molid.m.menu delete 0 end
    set moltxt "(none)"

    if { [llength $mollist] > 0 } {
        $w.foot configure -state normal
        $w.in.molid.m configure -state normal
        foreach id $mollist {
            $w.in.molid.m.menu add radiobutton -value $id \
                -command {global vmd_molecule ; if {[info exists vmd_molecule($::pbcgui::molid)]} {set ::pbcgui::moltxt "$::pbcgui::molid:[molinfo $::pbcgui::molid get name]"} {set ::pbcgui::moltxt "(none)" ; set molid -1} } \
                -label "$id [molinfo $id get name]" \
                -variable ::pbcgui::molid
            if {$id == $molid} {
                if {[info exists vmd_molecule($molid)]} then {
                    set moltxt "$molid:[molinfo $molid get name]"
                } else {
                    set moltxt "(none)"
                    set molid -1
                }
            }
        }
    }
}

############################################################
# Local Variables:
# mode: tcl
# time-stamp-format: "%u %02d.%02m.%y %02H:%02M:%02S %s"
# End:
############################################################
