# hello emacs this is -*- tcl -*-
#
# Small package with GUI to add a (dynamic) representation 
# of a dipole to selections
#
# (c) 2006-2009 by Axel Kohlmeyer <akohlmey@cmm.chem.upenn.edu>
# (c) 2018 by Axel Kohlmeyer <akohlmey@gmail.com>
#     
########################################################################
#
# $Id: dipwatch.tcl,v 1.6 2018/10/22 14:50:35 johns Exp $
#
# create package and namespace and default all namespace global variables.

namespace eval ::DipWatch:: {
    # exported functions
    namespace export dipwatchgui   ; # pop up GUI
    namespace export dipwatchrun   ; # (re)activate dipole tracing
    namespace export dipwatchstop  ; # stop dipole tracing 
    namespace export dipwatchset   ; # set parameters from command line
    namespace export dipwatchexport; # export dipole values to file

    variable w;               # handle to the base widget.
    variable version "1.3";   # plugin version
    variable numdips 6;       # number of dipoles (could be made dynamical)
    variable diptoggle;       # on/off flags for dipoles
    variable dipmolid;        # molecule id for dipoles
    variable dipoldmolid;     # old molecule id for dipoles
    variable dipselstr;       # selection string for dipoles
    variable dipselfun;       # selection function (result of atomselect)
    variable dipselupd;       # on/off flags for updating selections
    variable dipcolor;        # color of dipoles
    variable dipscale;        # scaling factor for dipole arrow
    variable dipradius;       # radius for dipole arrow
    variable dipvalue;        # value of the dipole moment
    variable dipgidlist;      # dipole gid lists

    for {set i 0} {$i < $numdips} {incr i} {
        set diptoggle($i) 0
        set dipmolid($i)  0
        set dipoldmolid($i)  0
        set dipselstr($i) all
        set dipselfun($i) "none"
        set dipselupd($i) 0
        set dipcolor($i) red
        set dipscale($i) 1.0
        set dipradius($i) 0.2
        set dipcenter($i) "-masscenter"
        set dipvalue($i) {  0.00 D}
        set dipgidlist($i) {}
    }
}

package provide dipwatch $DipWatch::version


#####################
# text mode interface to change settings
proc ::DipWatch::dipwatchset {dipid args} {
    variable diptoggle
    variable dipmolid
    variable dipselstr
    variable dipselupd
    variable dipcolor
    variable dipscale
    variable dipradius

    # check for correct dipole ID.
    if { ![info exists diptoggle($dipid)] } { 
        puts stderr "dipwatchset: illegal dipole id '$dipid'."
        return
    }

    # test if proper number arguments was given
    set n_args [llength $args]
    if { [expr fmod($n_args,2)] } { 
        puts stderr "dipwatchset: incorrect number of arguments."
        return
    }
    
    # check the complete remainder of the command line for validity
    for {set i 0} {$i < $n_args} {incr i 2} {
        set key [lindex $args $i]
        set suppkw {toggle mol sel color scale radius}
        if {[lsearch -exact $suppkw $key] < 0} {
            puts stderr "dipwatchset: unknown flag '$key', supported flags are: $suppkw"
            return
        }
    }
    # and now process those flags
    for {set i 0} {$i < $n_args} {incr i 2} {
        set key [lindex $args $i]
        set val [lindex $args [expr {$i + 1}]]

        switch $key {

            toggle {
                if {$val != $diptoggle($dipid)} {
                    set diptoggle($dipid) $val
                    ButtonToggle $dipid
                }
            }

            mol    {
                if {[string equal $val top]} {
                    set dipmolid($dipid)  [molinfo top]
                } else {
                    set dipmolid($dipid)  $val
                }
            }

            update {
                if {[string equal -nocase $val yes] || [string equal -nocase $val y]
                    || [string equal -nocase $val 1] || [string equal -nocase $val true]
                    || [string equal -nocase $val on]} {
                    set dipselupd($dipid) 1
                } else {
                    set dipselupd($dipid) 0
                }
            }

            sel    {set dipselstr($dipid) $val}
            color  {set dipcolor($dipid)  $val}
            scale  {set dipscale($dipid)  $val}
            radius {set dipradius($dipid) $val}
        }
    }
    CheckMolID $dipid
}
#####################
# write dipole moment trajectory with current settings
proc ::DipWatch::dipwatchexport {dipid {fname "none"} {step 1}} {
    variable w
    variable dipmolid
    variable dipselstr
    variable dipselupd
    variable dipcenter

    set molid $dipmolid($dipid)
    if {[string equal $fname "none"]} {
        set fname [tk_getSaveFile -defaultextension .dat -initialfile "dipole.dat" \
                       -filetypes { { {Generic Data File} {.dat .data} } \
                                        { {Generic Text File} {.txt} } \
                                        { {All Files} {.*} } } \
                       -title {Save dipole data to file} -parent $w]
    }
    if {! [string length $fname] } return ; # user has canceled file selection.
    if { [catch {set fp [open $fname w]} errmsg] } {
        tk_dialog .errmsg {Dipole Output Error} "Could not open file $fname for writing:\n$errmsg" error 0 Dismiss
        return
    } else {
        set sel [atomselect $molid $dipselstr($dipid)]
        puts $fp "\# frame    dip_x     dip_y      dip_z    |dip|"
        set nf [molinfo $molid get numframes]
        for {set i 0} {$i < $nf} {incr i $step} {
            $sel frame $i
            if { $dipselupd($dipid) } { $sel update }
            if {! [catch {measure dipole $sel -debye $dipcenter($dipid)} vector]} {
                puts $fp "$i  [lindex $vector 0]  [lindex $vector 1]  [lindex $vector 2]  [veclength $vector]"
            }
        }
        $sel delete
        close $fp
    }
}


proc ::DipWatch::CheckMolID {dip} {
    variable diptoggle
    variable dipmolid
    variable dipoldmolid
    variable dipgidlist
    variable dipselfun

    if {$dipmolid($dip) != $dipoldmolid($dip) } {
        if {![string equal $dipselfun($dip) "none"]} {
            if { [lsearch -exact [info commands] $dipselfun($dip)] >= 0} {
                uplevel #0 "$dipselfun($dip) delete"
            }
            set dipselfun($dip) "none"
        }
        if {$diptoggle($dip)} {
            foreach g $dipgidlist($dip) { 
                graphics $dipoldmolid($dip) delete $g
            }
        }
        set dipoldmolid($dip) $dipmolid($dip)
    }
    DrawDips
}

proc ::DipWatch::EntryUpdate {dip} {
    DrawDips
    return 1
}


# update molecule list
proc ::DipWatch::UpdateMolecule { args } {
    variable w
    variable diptoggle
    variable dipmolid
    variable dipselfun
    variable dipgidlist
    global vmd_molecule

    set mollist [molinfo list]
    set f $w.frame

    # Update the molecule browsers (and enable/disable selectors)
    foreach i [lsort [array names diptoggle]] {

        # handle the case that a molecule got deleted first.
        # we delete the related selection and disable the dipole display
        if { [lsearch -exact $mollist $dipmolid($i)] < 0 } {
            set dipgidlist($i) {} ; # graphics objects are automatically gone.
            if { [lsearch -exact [info commands] $dipselfun($i)] >= 0} {
                uplevel #0 "$dipselfun($i) delete" ; # selections not.
            }
            set dipselfun($i) "none"
            set dipmolid($i) [lindex $mollist 0]
            set dipmolid($i) [lindex $mollist 0]
            set diptoggle($i) 0
        }

        $f.m$i.menu delete 0 end
        $f.m$i configure -state disabled
        $f.b$i configure -state disabled
        if { [llength $mollist] != 0 } {
            foreach id $mollist {
                $f.m$i.menu add radiobutton -value $id \
                    -label "$id [molinfo $id get name]" \
                    -variable ::DipWatch::dipmolid($i) \
                    -command "::DipWatch::CheckMolID $i"
            }
            $f.b$i configure -state normal
            $f.m$i configure -state normal 
            if { [lsearch -exact $mollist $dipmolid($i)] < 0} {
                set dipmolid($i) [lindex $mollist 0]
                set dipoldmolid($i) [lindex $mollist 0]
                set dipgidlist($i) {}
            }
        } else {
            set diptoggle($i) 0
        }
    }
}

#################
# the heart of the matter. draw a dipole.
proc ::DipWatch::draw_dipole {mol sel {update 0} {color red} {scale 1.0} {radius 0.2} {dipidx 0}} {
    variable dipvalue
    variable dipcenter

    set res 6
    set gidlist {}
    set filled yes
    if { $update } { $sel update }

    # perhaps this should use the center information
    if {[catch {measure center $sel weight mass} center]} {
        if {[catch {measure center $sel} center]} {
            puts stderr "problem computing dipole center: $center"
            return {}
        }
    }
    if {[catch {measure dipole $sel -debye $dipcenter($dipidx)} vector]} {
        puts stderr "problem computing dipole vector: $vector"
        return {}
    }

    set dipvalue($dipidx) [format "%6.2f D" [veclength $vector]]
    set vechalf [vecscale [expr $scale * 0.5] $vector]

    lappend gidlist [graphics $mol color $color]
    lappend gidlist [graphics $mol cylinder [vecsub $center $vechalf] \
                         [vecadd $center [vecscale 0.7 $vechalf]] \
                         radius $radius resolution $res filled $filled]
    lappend gidlist [graphics $mol cone [vecadd $center [vecscale 0.7 $vechalf]] \
                         [vecadd $center $vechalf] radius [expr $radius * 1.7] \
                             resolution $res]
    return $gidlist
}

# check whether a selection is still valid and the same
proc ::DipWatch::diptestsel {molid sel selstr} {
   
    # this case should have been processed by UpdateMolecule already...
    if { [lsearch -exact [molinfo list] $molid] < 0 } {
        return 1
    }

    # check whether a selection has been defined
    if { [string equal $sel "none"] } {
        return 1
    }
    if { [lsearch -exact [info commands] $sel] < 0} {
        return 1
    }

    # check whether the selection string has changed
    if { 1 != [string equal [$sel text] $selstr] } {
        uplevel #0 $sel delete
        return 1
    }
    return 0
}


###########################3
# update all dipoles
proc ::DipWatch::DrawDips {args} {
    variable w;
    variable dipgidlist;
    variable diptoggle;
    variable dipmolid;
    variable dipselstr;
    variable dipselfun;
    variable dipselupd;
    variable dipcolor;
    variable dipscale;
    variable dipradius;

    display update off
    foreach i [array names dipgidlist] {
        if {$diptoggle($i)} {
            if { [diptestsel $dipmolid($i) $dipselfun($i) $dipselstr($i)] } {
                if {[catch {atomselect $dipmolid($i) $dipselstr($i)} sel]} {
                    puts stderr "dipwatch.tcl: problem creating atom selection for dipole $i: $sel"
                    continue ; # skip this one and try next entry in list.
                }
                $sel global
                set dipselfun($i) $sel
            }
            set sel $dipselfun($i)
            foreach g $dipgidlist($i) { 
                graphics $dipmolid($i) delete $g
            }
            set dipgidlist($i) [draw_dipole $dipmolid($i) $sel $dipselupd($i) \
                                    $dipcolor($i) $dipscale($i) $dipradius($i) $i]
        }
    }
    display update on
}

#################
# fix up dipole drawing after an en-/disable event
proc ::DipWatch::ButtonToggle {args} {
    variable w
    variable diptoggle
    variable dipmolid
    variable dipgidlist


    set dip [lindex $args 0]
    if {$diptoggle($dip) == 0} {
        foreach g $dipgidlist($dip) { 
            graphics $dipmolid($dip) delete $g
        }
    }

    DrawDips
}
#################
# initialization.
# create main window layout
proc ::DipWatch::dipwatchgui {} {
    variable w
    variable dipgidlist
    variable diptoggle

    # main window frame
    set w .dipwatchgui
    catch {destroy $w}
    toplevel    $w
    wm title    $w "Dipole Monitoring Tool" 
    wm iconname $w "DipWatch" 
    wm minsize  $w 300 200

    # menubar
    frame $w.menubar -relief raised -bd 2
    pack $w.menubar -side top -padx 1 -fill x
    menubutton $w.menubar.help -text Help -underline 0 -menu $w.menubar.help.menu
    # XXX - set menubutton width to avoid truncation in OS X
    $w.menubar.help config -width 5

    # Help menu
    menu $w.menubar.help.menu -tearoff no
    $w.menubar.help.menu add command -label "About" \
        -command {tk_messageBox -type ok -title "About DipWatch" \
                      -message "The dipwatch plugin provides a script and a GUI to draw arrows inside a molecule to represent the dipole moment of a given selection.\n\nVersion $DipWatch::version\n(c) 2006-2009 by Axel Kohlmeyer\n<akohlmey@cmm.chem.upenn.edu>\n(c) 2018 by Axel Kohlmeyer\n<akohlmey@gmail.com>"}
    $w.menubar.help.menu add command -label "Help..." \
        -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/dipwatch"
    pack $w.menubar.help -side right

    # main frame
    set f $w.frame
    frame $f -bd 2 -relief raised
    pack $f -side top -fill x

    grid configure $w.menubar -row 0 -column 0 -sticky "snew"
    grid configure $w.frame -row 1 -column 0 -sticky "snew"
    grid rowconfigure $w 0 -weight 0
    grid rowconfigure $w 1 -weight 1 -minsize 200
    grid columnconfigure $w 0 -weight 1 -minsize 200

    # create entries for the list of dipoles
    label $f.b -text "Dipole \#:"
    label $f.m -text "Molecule \#:"
    label $f.s -text "Selection:"
    label $f.u -text "Update:"
    label $f.c -text "Color:"
    label $f.f -text "Scaling:"
    label $f.r -text "Radius:"
    label $f.v -text "Value:"
    label $f.o -text "Write File:"
    grid configure $f.b -row 0 -column 0 -sticky "snew"
    grid configure $f.m -row 0 -column 1 -sticky "snew"
    grid configure $f.s -row 0 -column 2 -sticky "snew"
    grid configure $f.u -row 0 -column 3 -sticky "snew"
    grid configure $f.c -row 0 -column 4 -sticky "snew"
    grid configure $f.f -row 0 -column 5 -sticky "snew"
    grid configure $f.r -row 0 -column 6 -sticky "snew"
    grid configure $f.v -row 0 -column 7 -sticky "snew"
    grid configure $f.o -row 0 -column 8 -sticky "snew"
    grid rowconfigure $f 0 -weight 0 -minsize 10

    set colors [colorinfo colors]

    foreach i [lsort [array names dipgidlist]] {
        checkbutton $f.b$i -variable ::DipWatch::diptoggle($i) \
            -text "Dipole $i:" -command "::DipWatch::ButtonToggle $i"
        entry $f.s$i -textvariable ::DipWatch::dipselstr($i) \
            -validatecommand "::DipWatch::EntryUpdate $i" \
            -validate focusout
        checkbutton $f.u$i -variable ::DipWatch::dipselupd($i)
        menubutton $f.m$i -relief raised -bd 2 -direction flush \
            -textvariable  ::DipWatch::dipmolid($i) \
            -menu $f.m$i.menu -width 10
        menu $f.m$i.menu -tearoff no
        menubutton $f.c$i -relief raised -bd 2 -direction flush \
            -textvariable  ::DipWatch::dipcolor($i) \
            -menu $f.c$i.menu -width 10
        menu $f.c$i.menu -tearoff no
        foreach c $colors {
            $f.c$i.menu add radiobutton -value $c \
                -variable ::DipWatch::dipcolor($i) -label $c \
                -command "::DipWatch::DrawDips $i"

        }
        entry $f.f$i -textvariable ::DipWatch::dipscale($i) -width 5 \
            -validatecommand "::DipWatch::EntryUpdate $i" \
            -validate focusout
        entry $f.r$i -textvariable ::DipWatch::dipradius($i) -width 5 \
            -validatecommand "::DipWatch::EntryUpdate $i" \
            -validate focusout
        label $f.v$i -textvariable ::DipWatch::dipvalue($i) -width 10 -relief raised 
        button $f.o$i \
            -text "Write" -command "::DipWatch::dipwatchexport $i"

        ButtonToggle $i
        pack $f.b$i -side left 
        pack $f.m$i -side left 
        pack $f.s$i -side left 
        pack $f.u$i -side left 
        pack $f.c$i -side left 
        pack $f.f$i -side left
        pack $f.r$i -side left
        pack $f.v$i -side left
        pack $f.o$i -side left
        grid configure $f.b$i -row [expr $i + 1] -column 0 -sticky "snew"
        grid configure $f.m$i -row [expr $i + 1] -column 1 -sticky "snew"
        grid configure $f.s$i -row [expr $i + 1] -column 2 -sticky "snew"
        grid configure $f.u$i -row [expr $i + 1] -column 3 -sticky "snew"
        grid configure $f.c$i -row [expr $i + 1] -column 4 -sticky "snew"
        grid configure $f.f$i -row [expr $i + 1] -column 5 -sticky "snew"
        grid configure $f.r$i -row [expr $i + 1] -column 6 -sticky "snew"
        grid configure $f.v$i -row [expr $i + 1] -column 7 -sticky "snew"
        grid configure $f.o$i -row [expr $i + 1] -column 8 -sticky "snew"
        grid rowconfigure $f [expr $i + 1] -weight 0 
    }
    grid columnconfigure $f 0 -weight 0
    grid columnconfigure $f 1 -weight 1 -minsize 15
    grid columnconfigure $f 2 -weight 5 -minsize 15
    grid columnconfigure $f 3 -weight 1 -minsize 10
    grid columnconfigure $f 4 -weight 1 -minsize 15
    grid columnconfigure $f 5 -weight 1 -minsize 10
    grid columnconfigure $f 6 -weight 1 -minsize 10 
    grid columnconfigure $f 7 -weight 1 -minsize 15 
    grid columnconfigure $f 8 -weight 1 -minsize 15 

    UpdateMolecule
    dipwatchrun

    global vmd_molecule
    trace variable vmd_molecule w ::DipWatch::UpdateMolecule
}

proc ::DipWatch::dipwatchrun {args} {
    global vmd_frame
    trace variable vmd_frame w    ::DipWatch::DrawDips
}

proc ::DipWatch::dipwatchstop {args} {
    variable numdips
    variable dipgidlist
    variable diptoggle
    global vmd_frame

    for {set i 0} {$i < $numdips} {incr i} {
        if {$diptoggle($dip)} {
            foreach g $dipgidlist($dip) { 
                graphics $dipmolid($dip) delete $g
            }
            set dipgidlist($dip) {}
        }
    }
    trace vdelete vmd_frame w    ::DipWatch::DrawDips
}

# callback for VMD menu entry
proc dipwatch_tk_cb {} {
    ::DipWatch::dipwatchgui
    return $::DipWatch::w
}
