# hello emacs this is -*- tcl -*-
#
# GUI around 'measure gofr' and 'measure rdf'.
#
# (c) 2006-2013 by Axel Kohlmeyer <akohlmey@gmail.com>
########################################################################
#
# $Id: gofrgui.tcl,v 1.13 2013/04/15 15:49:37 johns Exp $
#
# create package and namespace and default all namespace global variables.

namespace eval ::GofrGUI:: {
    namespace export gofrgui

    variable w;                # handle to the base widget.
    variable version    "1.3"; # plugin version      

    variable molid       "-1"; # molid of the molecule to grab
    variable moltxt        ""; # title the molecule to grab
    
    variable selstring1    ""; # selection string for first selection
    variable selstring2    ""; # selection string for second selection

    variable delta      "0.1"; # delta for histogram
    variable rmax      "10.0"; # max r in histogram

    variable usepbc       "1"; # apply PBC
    variable doupdate     "0"; # update selections during calculation

    variable first        "0"; # first frame
    variable last        "-1"; # last frame
    variable step         "1"; # frame step delta 

    variable cell_a     "0.0"; #
    variable cell_b     "0.0"; #
    variable cell_c     "0.0"; #
    variable cell_alpha "90.0"; #
    variable cell_beta  "90.0"; #
    variable cell_gamma "90.0"; #

    variable rlist         {}; # list of r values
    variable glist         {}; # list of normalized g(r) values   
    variable ilist         {}; # list of integrated g(r) values
    variable hlist         {}; # list of unnormalized histogram data
    variable frlist   "0 0 0"; # list of all, skipped, processed with alg1 frames, ...

    variable plotgofr     "1"; # plot the g(r) using multiplot
    variable plotint      "0"; # plot number integral

    variable dowrite      "0"; # write output to file.
    variable callrdf      "0"; # call "measure rdf" instead of "measure gofr"
    variable outfile "gofr.dat"; # name of output file

    variable numcuda  [vmdinfo numcudadevices]; # is CUDA available?
    variable cannotplot   "0"; # is multiplot available?
}

package provide gofrgui $GofrGUI::version


#################
# the heart of the matter. run 'measure gofr'.
proc ::GofrGUI::runmeasure {} {
    variable w

    variable molid
    variable selstring1
    variable selstring2
    variable delta
    variable rmax
    variable usepbc
    variable doupdate
    variable first
    variable last
    variable step

    variable rlist
    variable glist
    variable ilist
    variable hlist
    variable frlist

    variable cannotplot
    variable plotgofr
    variable plotint

    variable dowrite 
    variable outfile
    variable callrdf

    set errmsg {}
    set cannotplot [catch {package require multiplot 1.1}]

    set sel1 {}
    set sel2 {} 
    if {[catch {atomselect $molid "$selstring1"} sel1] \
            || [catch {atomselect $molid "$selstring2"} sel2] } then {
        tk_dialog .errmsg {g(r) Error} "There was an error creating the selections:\n$sel1\n$sel2" error 0 Dismiss
        return
    }

    if {$callrdf} {
        if {[catch {measure rdf $sel1 $sel2 delta $delta rmax $rmax usepbc $usepbc \
                        selupdate $doupdate first $first last $last step $step} \
                 errmsg ] } then {
            tk_dialog .errmsg {g(r) Error} "There was an error running 'measure rdf':\n\n$errmsg" error 0 Dismiss
            $sel1 delete
            $sel2 delete
            return
        } else {
            lassign $errmsg rlist glist ilist hlist frlist
            after 3000 {catch {destroy .succmsg}}
            tk_dialog .succmsg {g(r) Success} "g(r) calculation successful! [lindex $frlist 0] frames total, [lindex $frlist 2] frames processed, [lindex $frlist 1] frames skipped." info 0 OK
        }
    } else {
        if {[catch {measure gofr $sel1 $sel2 delta $delta rmax $rmax usepbc $usepbc \
                        selupdate $doupdate first $first last $last step $step} \
                 errmsg ] } then {
            tk_dialog .errmsg {g(r) Error} "There was an error running 'measure gofr':\n\n$errmsg" error 0 Dismiss
            $sel1 delete
            $sel2 delete
            return
        } else {
            lassign $errmsg rlist glist ilist hlist frlist
            after 3000 {catch {destroy .succmsg}}
            tk_dialog .succmsg {g(r) Success} "g(r) calculation successful! [lindex $frlist 0] frames total, [lindex $frlist 2] frames processed, [lindex $frlist 1] frames skipped." info 0 OK
        }
    }

    # display g(r)
    if {$plotgofr} {
        if {$cannotplot} then {
            tk_dialog .errmsg {g(r) Error} "Multiplot is not available. Enabling 'Save to File'." error 0 Dismiss
            set dowrite 1
        } else {
            set ph [multiplot -x $rlist -y $glist -title "g(r)" -lines -linewidth 3 -marker point -plot -hline {1 -width 1}]
        }
    }

    # display number integral
    if {$plotint} {
        if {$cannotplot} then {
            tk_dialog .errmsg {g(r) Error} "Multiplot is not available. Enabling 'Save to File'." error 0 Dismiss
            set dowrite 1
        } else {
            set ph [multiplot -x $rlist -y $ilist -title "Number integral over g(r)" -lines -linewidth 3 -marker point -plot -hline {5 -width 1} -hline {10 -width 1} -hline {20 -width 1}]
        }
    }

    # Save to File
    if {$dowrite} {
        set outfile [tk_getSaveFile -defaultextension .dat -initialfile $outfile \
                         -filetypes { { {Generic Data File} {.dat .data} } \
                          { {XmGrace Multi-Column Data File} {.nxy} } \
                          { {Generic Text File} {.txt} } \
                          { {All Files} {.*} } } \
                         -title {Save g(r) Data to File} -parent $w]
        set fp {}
        if {[string length $outfile]} {
            if {[catch {open $outfile w} fp]} then {
                tk_dialog .errmsg {g(r) Error} "There was an error opening the output file '$outfile':\n\n$fp" error 0 Dismiss
            } else {
                foreach r $rlist g $glist i $ilist {
                    puts $fp "$r $g $i"
                }
                close $fp
            }
        }
    }

    # clean up
    $sel1 delete
    $sel2 delete
}

#################
# set unit cell
proc ::GofrGUI::set_unitcell {} {
    variable molid
    variable unitcell_a
    variable unitcell_b
    variable unitcell_c
    variable unitcell_alpha
    variable unitcell_beta
    variable unitcell_gamma

    set n [molinfo $molid get numframes]

    for {set i 0} {$i < $n} {incr i} {
        molinfo $molid set frame $i
        molinfo $molid set a $unitcell_a
        molinfo $molid set b $unitcell_b
        molinfo $molid set c $unitcell_c
        molinfo $molid set alpha $unitcell_alpha
        molinfo $molid set beta $unitcell_beta
        molinfo $molid set gamma $unitcell_gamma
    }
}

#################
# set unit cell dialog
proc ::GofrGUI::unitcellgui {args} {
    variable clicked
    variable molid
    variable unitcell_a
    variable unitcell_b
    variable unitcell_c
    variable unitcell_alpha
    variable unitcell_beta
    variable unitcell_gamma

    set d .unitcellgui

    catch {destroy $d}
    toplevel $d -class Dialog
    wm title $d {Set Unit Cell}
    wm protocol $d WM_DELETE_WINDOW {set clicked -1}
    wm minsize  $d 240 200  

    # only make the dialog transient if the parent is viewable.
    if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
        wm transient $d [winfo toplevel [winfo parent $d]]
    }

    # outer frame
    frame $d.top
    frame $d.bot
    $d.top configure -relief raised -bd 1
    $d.bot configure -relief flat -bd 1
    grid $d.top -row 0 -column 0 -sticky snew
    grid $d.bot -row 1 -column 0 -sticky ew
    grid columnconfigure $d 0 -weight 1
    grid rowconfigure $d 0 -weight 100
    grid rowconfigure $d 1 -weight 1

    # retrieve current values for cell dimensions
    lassign [molinfo $molid get {a b c alpha beta gamma} ] \
         unitcell_a unitcell_b unitcell_c unitcell_alpha unitcell_beta unitcell_gamma

    # dialog contents:
    label $d.top.head -justify center -relief raised -text {Set unit cell parameters for all frames:}
    label $d.top.la  -justify left -text {Length a:}
    label $d.top.lb  -justify left -text {Length b:}
    label $d.top.lc  -justify left -text {Length c:}
    label $d.top.lal -justify left -text {Angle alpha:}
    label $d.top.lbe -justify left -text {Angle beta:}
    label $d.top.lga -justify left -text {Angle gamma:}
    entry $d.top.ea  -justify left -textvariable ::GofrGUI::unitcell_a
    entry $d.top.eb  -justify left -textvariable ::GofrGUI::unitcell_b
    entry $d.top.ec  -justify left -textvariable ::GofrGUI::unitcell_c
    entry $d.top.eal -justify left -textvariable ::GofrGUI::unitcell_alpha
    entry $d.top.ebe -justify left -textvariable ::GofrGUI::unitcell_beta
    entry $d.top.ega -justify left -textvariable ::GofrGUI::unitcell_gamma

    # dialog layout
    grid $d.top.head -column 0 -row 0 -columnspan 2 -sticky snew 
    grid rowconfigure $d.top 0 -weight 2
    set i 1
    foreach l "$d.top.la $d.top.lb $d.top.lc $d.top.lal $d.top.lbe $d.top.lga" \
        e "$d.top.ea $d.top.eb $d.top.ec $d.top.eal $d.top.ebe $d.top.ega" {
            grid $l -column 0 -row $i -sticky snew 
            grid $e -column 1 -row $i -sticky snew
            grid rowconfigure $d.top $i -weight 1
            incr i
        }
    grid columnconfigure $d.top 0 -weight 1
    grid columnconfigure $d.top 1 -weight 2
    
    # buttons
    button $d.bot.ok  -text {Set unit cell} -command {set clicked 1 ; ::GofrGUI::set_unitcell}
    button $d.bot.can -text {Cancel}        -command {set clicked 1}
    grid $d.bot.ok  -column 0 -row 0 -padx 10 -pady 4
    grid $d.bot.can -column 1 -row 0 -padx 10 -pady 4
    grid columnconfigure $d.bot 0 -weight 1
    grid columnconfigure $d.bot 1 -weight 1

    bind $d <Destroy> {set clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

#################
# build GUI.
proc ::GofrGUI::gofrgui {args} {
    variable w
    variable numcuda
    variable molid

    # main window frame
    set w .gofrgui
    catch {destroy $w}
    toplevel    $w
    wm title    $w "Radial Pair Distribution Function g(r)" 
    wm iconname $w "GofrGUI" 
    wm minsize  $w 610 250  

    # top level dialog components
    # menubar
    frame $w.menubar -relief raised -bd 2 
    # frame for settings
    labelframe $w.in -bd 2 -relief ridge -text "Settings:" -padx 1m -pady 1m
    # computation action button
    button $w.foot -text {Compute g(r)} -command [namespace code runmeasure]

    # layout main canvas
    grid $w.menubar -row 0 -column 0 -sticky new
    grid $w.in      -row 1 -column 0 -sticky snew
    grid $w.foot    -row 2 -column 0 -sticky sew
    grid columnconfigure $w 0 -minsize 250 -weight 1
    grid rowconfigure    $w 0 -weight 1    -minsize 25
    grid rowconfigure    $w 1 -weight 10   -minsize 150
    grid rowconfigure    $w 2 -weight 1    -minsize 25

    ###################
    # menubar contents.
    menubutton $w.menubar.util -text Utilities -underline 0 -menu $w.menubar.util.menu -anchor w
    menubutton $w.menubar.help -text Help -underline 0 -menu $w.menubar.help.menu -anchor e

    # Utilities menu.
    menu $w.menubar.util.menu -tearoff no
    $w.menubar.util.menu add command -label "Set unit cell dimensions" \
                           -command ::GofrGUI::unitcellgui
    # Help menu.
    menu $w.menubar.help.menu -tearoff no
    $w.menubar.help.menu add command -label "About" \
               -command {tk_messageBox -type ok -title "About g(r) GUI" \
                              -message "The g(r) GUI provides a graphical interface to the 'measure gofr' and 'measure rdf' commands in VMD. g(r) refers to spherical atomic radial distribution functions, a special case of pair correlation functions.\n\nVersion $::GofrGUI::version\n(c) 2006-2013 by Axel Kohlmeyer\n<akohlmey@gmail.com>"}
    $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/gofrgui"

    # set layout of menubar
    grid $w.menubar.util x $w.menubar.help -row 0 -sticky ew
    grid configure $w.menubar.util -sticky w
    grid rowconfigure $w.menubar 0 -minsize 20
    grid columnconfigure $w.menubar 0 -weight 1   -minsize 10
    grid columnconfigure $w.menubar 1 -weight 100 -minsize 0
    grid columnconfigure $w.menubar 2 -weight 1   -minsize 10

    #################
    # subdivide and layout the settings frame
    frame $w.in.molid
    frame $w.in.sel
    frame $w.in.frame
    frame $w.in.parm
    frame $w.in.opt
    grid $w.in.molid -row 0 -column 0 -sticky snew
    grid $w.in.sel   -row 1 -column 0 -sticky snew
    grid $w.in.frame -row 2 -column 0 -sticky snew
    grid $w.in.parm  -row 3 -column 0 -sticky snew
    grid $w.in.opt   -row 4 -column 0 -sticky snew
    grid columnconfigure $w.in 0 -weight 1
    grid rowconfigure    $w.in 0 -weight 1 
    grid rowconfigure    $w.in 1 -weight 1 
    grid rowconfigure    $w.in 2 -weight 1 
    grid rowconfigure    $w.in 3 -weight 1 
    grid rowconfigure    $w.in 4 -weight 1 


    # Molecule selector
    set i $w.in.molid
    label $i.l -text "Use Molecule:" -anchor w
    menubutton $i.m -relief raised -bd 2 -direction flush \
	-text "test text" -textvariable ::GofrGUI::moltxt \
	-menu $i.m.menu
    menu $i.m.menu -tearoff no
    grid $i.l $i.m -sticky snew -row 0
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 3

    # Selections
    set i $w.in.sel
    label $i.al -text "Selection 1:" -anchor e
    entry $i.at -width 20 -textvariable ::GofrGUI::selstring1
    label $i.bl -text "Selection 2:" -anchor e
    entry $i.bt -width 20 -textvariable ::GofrGUI::selstring2
    grid $i.al $i.at $i.bl $i.bt -row 0 -sticky snew
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 1
    grid columnconfigure $i 2 -weight 1
    grid columnconfigure $i 3 -weight 1

    # Frame range
    set i $w.in.frame
    label $i.t -text "Frames:" -anchor w
    label $i.fl -text "First:" -anchor e
    entry $i.ft -width 4 -textvariable ::GofrGUI::first
    label $i.ll -text "Last:" -anchor e
    entry $i.lt -width 4 -textvariable ::GofrGUI::last
    label $i.sl -text "Step:" -anchor e
    entry $i.st -width 4 -textvariable ::GofrGUI::step
    grid $i.t $i.fl $i.ft $i.ll $i.lt $i.sl $i.st -row 0 -sticky snew
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 3
    grid columnconfigure $i 2 -weight 2
    grid columnconfigure $i 3 -weight 3
    grid columnconfigure $i 4 -weight 2
    grid columnconfigure $i 5 -weight 3
    grid columnconfigure $i 6 -weight 2

    # parameters
    set i $w.in.parm
    label $i.l -text {Histogram Parameters: } -anchor w
    label $i.deltal -text {delta r:} -anchor e
    entry $i.deltat -width 5 -textvariable ::GofrGUI::delta
    label $i.rmaxl -text {max. r:} -anchor e
    entry $i.rmaxt -width  5 -textvariable ::GofrGUI::rmax
    grid $i.l $i.deltal $i.deltat $i.rmaxl $i.rmaxt -row 0 -sticky snew
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 3
    grid columnconfigure $i 2 -weight 2
    grid columnconfigure $i 3 -weight 3
    grid columnconfigure $i 4 -weight 2

    # options
    set i $w.in.opt
    checkbutton $i.pbc -relief groove -text {Use PBC} -variable ::GofrGUI::usepbc
    checkbutton $i.upd -relief groove -text {Update Selections} -variable ::GofrGUI::doupdate
    checkbutton $i.plg -relief groove -text {Display g(r)} -variable ::GofrGUI::plotgofr
    checkbutton $i.pli -relief groove -text {Display Int(g(r))} -variable ::GofrGUI::plotint
    checkbutton $i.sav -relief groove -text {Save to File} -variable ::GofrGUI::dowrite
    checkbutton $i.rdf -relief groove -text {Use GPU code} -variable ::GofrGUI::callrdf
    grid $i.pbc $i.upd $i.plg $i.pli $i.sav $i.rdf -row 0 -sticky snew
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 1
    grid columnconfigure $i 2 -weight 1
    grid columnconfigure $i 3 -weight 1
    grid columnconfigure $i 4 -weight 1
    grid columnconfigure $i 5 -weight 1

    if {$numcuda < 1} {
        $w.in.opt.rdf configure -state disabled
    }

    UpdateMolecule
    EnDisable
    global vmd_molecule
    trace variable vmd_molecule w ::GofrGUI::UpdateMolecule
    trace variable ::GofrGUI::molid w ::GofrGUI::EnDisable
}

# en-/disable buttons that depend on a proper molecule being selected
proc ::GofrGUI::EnDisable {args} {
    variable molid
    variable w

    if {$molid < 0 } {
        $w.foot configure -state disabled
        $w.menubar.util configure -state disabled
    } else {
        $w.foot configure -state normal
        $w.menubar.util configure -state normal
    }
}

# callback for VMD menu entry
proc gofrgui_tk_cb {} {
  ::GofrGUI::gofrgui 
  return $::GofrGUI::w
}

# update molecule list
proc ::GofrGUI::UpdateMolecule {args} {
    variable w
    variable moltxt
    variable molid
    global vmd_molecule

    # Update the molecule browser
    set mollist [molinfo list]
    $w.in.molid.m configure -state disabled
    $w.in.molid.m.menu delete 0 end
    set moltxt "(none)"

    if { [llength $mollist] > 0 } {
        $w.in.molid.m configure -state normal 
        foreach id $mollist {
            $w.in.molid.m.menu add radiobutton -value $id \
                -command {global vmd_molecule ; if {[info exists vmd_molecule($::GofrGUI::molid)]} {set ::GofrGUI::moltxt "$::GofrGUI::molid:[molinfo $::GofrGUI::molid get name]"} {set ::GofrGUI::moltxt "(none)" ; set molid -1} } \
                -label "$id [molinfo $id get name]" \
                -variable ::GofrGUI::molid
            if {$id == $molid} {
                if {[info exists vmd_molecule($molid)]} then {
                    set moltxt "$molid:[molinfo $molid get name]"  
                } else {
                    set moltxt "(none)"
                    set molid -1
                }
            }
        }
    } else {
        set moltxt "(none)"
        set molid -1
    }
}

############################################################
# Local Variables:
# mode: tcl
# time-stamp-format: "%u %02d.%02m.%y %02H:%02M:%02S %s"
# End:
############################################################
