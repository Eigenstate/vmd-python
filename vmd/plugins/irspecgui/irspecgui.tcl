# hello emacs this is -*- tcl -*-
#
# GUI around 'specden'.
#
# (c) 2006-2013 by Axel Kohlmeyer <akohlmey@gmail.com>
########################################################################
#
# $Id: irspecgui.tcl,v 1.17 2013/04/15 15:59:39 johns Exp $
#
# create package and namespace and default all namespace global variables.
package require specden 1.1

namespace eval ::IRspecGUI:: {
    namespace export irspecgui

    variable w;                # handle to the base widget.
    variable version    "1.3"; # plugin version      

    variable molid       "-1"; # molid of the molecule to grab
    variable moltxt        ""; # title the molecule to grab
    
    variable selstring  "all"; # selection string for selection
    variable doupdate     "0"; # update selection during calculation

    variable deltat   "0.001"; # time delta between frames
    variable tunit        "2"; # unit for time delta (0=a.u.,1=fs,2=ps) 
    variable tunittxt    "ps"; # unit for time delta (0=a.u.,1=fs,2=ps) 
    variable maxfreq "2000.0"; # max frequency

    variable temp     "300.0"; # temperature in Kelvin
    variable correct "harmonic" ; # correction method

    variable oversampl    "1"; # oversampling

    variable first        "0"; # first frame
    variable last        "-1"; # last frame
    variable step         "1"; # frame step delta 

    variable flist         {}; # list of frequencies
    variable slist         {}; # list of spectral densities

    variable doplot       "1"; # plot the spectrum using multiplot
    variable dowrite      "0"; # write output to file.
    variable outfile "spec.dat"; # name of output file

    variable cannotplot   "0"; # is multiplot available?
}

package provide irspecgui $::IRspecGUI::version

#####################
# copy charges from beta field
proc ::IRspecGUI::copybtocharge {} {
    variable molid
    variable selstring

    set sel {}
    if {($molid < 0) || ([molinfo num] < 1)} {
        tk_dialog .errmsg {IRspec Error} "No valid molecule available or selected.\nCannot assign charge." error 0 Dismiss
        return
    }
    if {[catch {atomselect $molid "$selstring"} sel] } then {
        tk_dialog .errmsg {IRspec Error} "There was an error creating the selection:\n$sel" error 0 Dismiss
        return
    }
    $sel set charge [$sel get beta]
    $sel delete
}

#####################
# guess charges via APBSrun (could it be using autopsf???)
proc ::IRspecGUI::guesscharge {} {
    variable molid
    variable selstring

    set errmsg {}
    set sel {}
    if {[catch {package require apbsrun 1.2} errmsg] } {
        tk_dialog .errmsg {IRspec Error} "Could not load the APBSrun package needed to guess charges:\n$errmsg" error 0 Dismiss
        return
    }
    if {($molid < 0) || ([molinfo num] < 1)} {
        tk_dialog .errmsg {IRspec Error} "No valid molecule available or selected.\nCannot assign charge." error 0 Dismiss
        return
    }
    if {[catch {atomselect $molid "$selstring"} sel] } then {
        tk_dialog .errmsg {IRspec Error} "There was an error creating the selection:\n$sel" error 0 Dismiss
        return
    }
    ::APBSRun::set_parameter_charges $sel 
    $sel delete
}

#####################
# read charges from file.
proc ::IRspecGUI::readcharge {} {
    variable w
    variable molid

    set fname [tk_getOpenFile -initialfile "charges.dat" \
                         -filetypes { { {Generic Data File} {.dat .data} } \
                          { {Generic Text File} {.txt} } \
                          { {All Files} {.*} } } \
                         -title {Load atom name to charge mapping file} -parent $w]
    if {! [string length $fname] } return ; # user has canceled file selection.
    if {($molid < 0) || ([molinfo num] < 1)} {
        tk_dialog .errmsg {IRspec Error} "No valid molecule available or selected.\nCannot assign charge." error 0 Dismiss
        return
    }
    if { ![file exists $fname] || [catch {set fp [open $fname r]} errmsg] } {
        tk_dialog .errmsg {IRspec Error} "Could not open file $fname for reading:\n$errmsg" error 0 Dismiss
        return
    } else {
        # Load the charges from file
        set lnr 0
        while {-1 != [gets $fp line]} {
            incr lnr
            if {![regexp {^\s*#} $line]} {
                set line [regexp -all -inline {\S+} $line]
                if {[llength $line] >= 2} {
                    if {[catch {atomselect $molid "name [lindex $line 0]"} sel]} {
                       tk_dialog .errmsg {IRspec Error} "Error applying charge info from file $fname line $lnr:\n$sel\n" error 0 Dismiss
                    } else {
                       $sel set charge [lindex $line 1]
                       $sel delete
                    }
                }
            }
        }
        close $fp
    }
}

#################
# the heart of the matter. run 'measure dipole' and compute spectral densities.
proc ::IRspecGUI::runmeasure {} {
    variable w

    variable molid
    variable selstring
    variable doupdate

    variable deltat
    variable tunit
    variable maxfreq
    variable oversampl
    variable temp
    variable correct

    variable first
    variable last
    variable step

    variable flist
    variable slist

    variable cannotplot
    variable doplot
    variable dowrite 
    variable outfile

    set errmsg {}
    set cannotplot [catch {package require multiplot}]

    # we need a selection for 'measure dipole'...
    set sel {}
    if {[catch {atomselect $molid "$selstring"} sel] } then {
        tk_dialog .errmsg {IRspec Error} "There was an error creating the selection:\n$sel" error 0 Dismiss
        return
    }

    # we need some frames
    set nframes [molinfo $molid get numframes]
    set from $first
    set to $last
    if {$last == -1} {set to $nframes}
    if {($to < $first) || ($first < 0) || ($step < 0) || ($step > $nframes)} {
        tk_dialog .errmsg {IRspec Error} "Invalid frame range given: $first:$last:$step" error 0 Dismiss
        return
    }
    set diplist {}
    set flist {}
    set slist {}

    # Make sure that we have a dipole moment.
    set p_sum 0.0
    set n_sum 0.0
    foreach charge [$sel get charge] {
        if {$charge < 0} {
            set n_sum [expr {$n_sum - $charge}]
        } else {
            set p_sum [expr {$p_sum + $charge}]
        }
    }
    if { ($p_sum < 0.1) || ($n_sum < 0.1) } {
        tk_dialog .errmsg {IRspec Error} "Insufficent charges to form a dipole. Please check your selection, load a proper topology file or assign charges manually." error 0 Dismiss
        $sel delete
        return
    }
    
    # detect if we have enough data available.
    set tval [expr $step * $deltat * [lindex "1.0 41.3413741313826 41341.3741313825" $tunit]]
    set ndat [expr {($to - $from) / $step}]
    set nn [expr {$ndat*$maxfreq/219474.0*$tval/(2.0*3.14159265358979)}]
    if { [expr {$nn + 2}] > $ndat } {
        tk_dialog .errmsg {IRspec Error} "Not enough data for frequency range. Please lower the maximum frequency or load more frames to the trajectory." error 0 Dismiss
        $sel delete
        return
    }

    # collect time series data for trajectory.
    for {set i $from} {$i<$to} {set i [expr {$i + $step}]} {
        $sel frame $i
        if {$doupdate} {
             catch {$sel update}
        }
        lappend diplist [measure dipole $sel -geocenter]
    }
    if {[catch {specden $diplist $tval $maxfreq $correct $temp $oversampl} errmsg ] } then {
        tk_dialog .errmsg {IRspec Error} "There was an error running 'specden':\n\n$errmsg" error 0 Dismiss
        $sel delete
        return
    } else {
        lassign $errmsg flist slist
    }

    # detect when specden produces crap. memory allocation bug in specden?
    set nans [lsearch -all $slist "nan"]
    if {[llength $nans] > 0 } {
        tk_dialog .errmsg {IRspec Error} "There was an internal error in 'specden':\n Please try changing some parameters slightly and recalculate" error 0 Dismiss
        $sel delete
        return
    }

    # display spectrum
    if {$doplot} {
        if {$cannotplot} then {
            tk_dialog .errmsg {IRspec Error} "Multiplot is not available. Enabling 'Save to File'." error 0 Dismiss
            set dowrite 1
        } else {
            set ph [multiplot -x $flist -y $slist -title "Spectral Densities" -lines -linewidth 3 -marker point -plot ]
        }
    }

    # Save to File
    if {$dowrite} {
        set outfile [tk_getSaveFile -defaultextension dat -initialfile $outfile \
                         -filetypes { { {Generic Data File} {.dat .data} } \
                          { {XmGrace Multi-Column Data File} {.nxy} } \
                          { {Generic Text File} {.txt} } \
                          { {All Files} {.*} } } \
                         -title {Save Spectral Density to File} -parent $w]
        set fp {}
        if {[string length $outfile]} {
            if {[catch {open $outfile w} fp]} then {
                tk_dialog .errmsg {IRspec Error} "There was an error opening the output file '$outfile':\n\n$fp" error 0 Dismiss
            } else {
                foreach f $flist s $slist {
                    puts $fp "$f $s"
                }
                close $fp
            }
        }
    }

    # clean up
    $sel delete
}

#################
# build GUI.
proc ::IRspecGUI::irspecgui {args} {
    variable w

    # main window frame
    set w .irspecgui
    catch {destroy $w}
    toplevel    $w
    wm title    $w "IR Spectra Calculation" 
    wm iconname $w "IRspecGUI" 
    wm minsize  $w 480 240  

    # top level dialog components
    # menubar
    frame $w.menubar -relief raised -bd 2 
    # frame for settings
    labelframe $w.in -bd 2 -relief ridge -text "Settings:" -padx 1m -pady 1m
    # computation action buttons
    button $w.foot -text {Compute Spectrum} -command [namespace code runmeasure]

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
    $w.menubar.util.menu add command -label "Guess atomic charges from CHARMM parameters." \
                           -command ::IRspecGUI::guesscharge
    $w.menubar.util.menu add command -label "Load name<->charge map from file." \
                           -command ::IRspecGUI::readcharge
    $w.menubar.util.menu add command -label "Copy charges from beta field." \
                           -command ::IRspecGUI::copybtocharge

    # Help menu.
    menu $w.menubar.help.menu -tearoff no
    $w.menubar.help.menu add command -label "About" \
               -command {tk_messageBox -type ok -title "About IRspec GUI" \
                              -message "The IRspec GUI provides a graphical interface to compute spectral densities from dipole time series data using the 'measure dipole' command in VMD. Several corrections for intensities can be applied.\n\nVersion $::IRspecGUI::version\n(c) 2006-2013 by Axel Kohlmeyer\n<akohlmey@gmail.com>"}
    $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/irspecgui"

    # set layout of menubar
    grid $w.menubar.util x $w.menubar.help -row 0 -sticky ew
    grid configure $w.menubar.util -sticky w
    grid rowconfigure $w.menubar 0 -minsize 20
    grid columnconfigure $w.menubar 0 -weight 1   -minsize 10
    grid columnconfigure $w.menubar 1 -weight 100 -minsize 0
    grid columnconfigure $w.menubar 2 -weight 1   -minsize 10

    ########################
    # subdivide and layout the settings frame
    frame $w.in.molid
    frame $w.in.sel
    frame $w.in.frame
    frame $w.in.parm
    frame $w.in.corr
    frame $w.in.opt
    grid $w.in.molid -row 0 -column 0 -sticky snew
    grid $w.in.sel   -row 1 -column 0 -sticky snew
    grid $w.in.frame -row 2 -column 0 -sticky snew
    grid $w.in.parm  -row 3 -column 0 -sticky snew
    grid $w.in.corr  -row 4 -column 0 -sticky snew
    grid $w.in.opt   -row 5 -column 0 -sticky snew
    grid columnconfigure $w.in 0 -weight 1
    grid rowconfigure    $w.in 0 -weight 1 
    grid rowconfigure    $w.in 1 -weight 1 
    grid rowconfigure    $w.in 2 -weight 1 
    grid rowconfigure    $w.in 3 -weight 1 
    grid rowconfigure    $w.in 4 -weight 1 
    grid rowconfigure    $w.in 5 -weight 1 

    # Molecule selector
    set i $w.in.molid
    label $i.l -text "Use Molecule:" -anchor w
    menubutton $i.m -relief raised -bd 2 -direction flush \
	-textvariable ::IRspecGUI::moltxt \
	-menu $i.m.menu
    menu $i.m.menu -tearoff no
    grid $i.l -sticky snew -row 0 -column 0
    grid $i.m -sticky snew -row 0 -column 1
    grid rowconfigure $i 0 -weight 1
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 3

    # Selection
    set i $w.in.sel
    label $i.al -text "Selection:" -anchor w
    entry $i.at -width 20 -textvariable ::IRspecGUI::selstring
    checkbutton $i.upd -relief groove -text {Update Selection} -variable ::IRspecGUI::doupdate
    grid $i.al  -column 0 -row 0 -sticky snew
    grid $i.at  -column 1 -row 0 -sticky snew
    grid $i.upd -column 2 -row 0 -sticky snew
    grid rowconfigure $i 0 -weight 1
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 3
    grid columnconfigure $i 2 -weight 4

    # Frame range
    set i $w.in.frame
    label $i.t -text "Frames:" -anchor w
    label $i.fl -text "First:" -anchor e
    entry $i.ft -width 4 -textvariable ::IRspecGUI::first
    label $i.ll -text "Last:" -anchor e
    entry $i.lt -width 4 -textvariable ::IRspecGUI::last
    label $i.sl -text "Step:" -anchor e
    entry $i.st -width 4 -textvariable ::IRspecGUI::step
    grid $i.t  -column 0 -row 0 -sticky snew
    grid $i.fl -column 1 -row 0 -sticky snew
    grid $i.ft -column 2 -row 0 -sticky snew
    grid $i.ll -column 3 -row 0 -sticky snew
    grid $i.lt -column 4 -row 0 -sticky snew
    grid $i.sl -column 5 -row 0 -sticky snew
    grid $i.st -column 6 -row 0 -sticky snew
    grid rowconfigure $i 0 -weight 1
    grid columnconfigure $i 0 -weight 1 -minsize 10
    grid columnconfigure $i 1 -weight 3 -minsize 10
    grid columnconfigure $i 2 -weight 2 -minsize 10
    grid columnconfigure $i 3 -weight 3 -minsize 10
    grid columnconfigure $i 4 -weight 2 -minsize 10
    grid columnconfigure $i 5 -weight 3 -minsize 10
    grid columnconfigure $i 6 -weight 2 -minsize 10

    # parameters
    set i $w.in.parm
    label $i.l -text {Time between frames: } -anchor w
    entry $i.deltat -width 5 -textvariable ::IRspecGUI::deltat
    menubutton $i.unit -relief raised -bd 2 -direction flush \
	-textvariable ::IRspecGUI::tunittxt -menu $i.unit.menu
    menu $i.unit.menu -tearoff no
    $i.unit.menu add radiobutton -value 0 \
                -command {set ::IRspecGUI::tunittxt "a.u."} \
                -label "a.u. (0.0242 fs)" \
                -variable ::IRspecGUI::tunit
    $i.unit.menu add radiobutton -value 1 \
                -command {set ::IRspecGUI::tunittxt "fs"} \
                -label "fs" \
                -variable ::IRspecGUI::tunit
    $i.unit.menu add radiobutton -value 2 \
                -command {set ::IRspecGUI::tunittxt "ps"} \
                -label "ps" \
                -variable ::IRspecGUI::tunit
    label $i.mfl -text {   max Freq (cm^-1):} 
    entry $i.maxf -width 8 -textvariable ::IRspecGUI::maxfreq
    grid $i.l      -column 0 -row 0 -sticky snew
    grid $i.deltat -column 1 -row 0 -sticky snew
    grid $i.unit   -column 2 -row 0 -sticky snew
    grid $i.mfl    -column 3 -row 0 -sticky snew
    grid $i.maxf   -column 4 -row 0 -sticky snew
    grid rowconfigure $i 0 -weight 1
    grid columnconfigure $i 0 -weight 1 -minsize 10
    grid columnconfigure $i 1 -weight 3 -minsize 10
    grid columnconfigure $i 2 -weight 1 -minsize 10
    grid columnconfigure $i 3 -weight 5 -minsize 10
    grid columnconfigure $i 4 -weight 2 -minsize 10

    # Correction
    set i $w.in.corr
    label $i.al -text "Temperature in K:" -anchor w
    entry $i.at -width 20 -textvariable ::IRspecGUI::temp
    label $i.ml -text "   Correction:" -anchor w
    menubutton $i.meth -relief raised -bd 2 -direction flush \
	-textvariable ::IRspecGUI::correct -menu $i.meth.menu
    menu $i.meth.menu -tearoff no
    foreach m "harmonic fourier classic kubo schofield" {
        $i.meth.menu add radiobutton -value $m \
                -variable ::IRspecGUI::correct -label $m
    }
    grid $i.al   -column 0 -row 0 -sticky snew
    grid $i.at   -column 1 -row 0 -sticky snew
    grid $i.ml   -column 2 -row 0 -sticky snew
    grid $i.meth -column 3 -row 0 -sticky snew
    grid rowconfigure $i 0 -weight 1
    grid columnconfigure $i 0 -weight 1
    grid columnconfigure $i 1 -weight 2
    grid columnconfigure $i 2 -weight 1
    grid columnconfigure $i 3 -weight 2 -minsize 150

    # options
    set i $w.in.opt
    label $i.ovl -text {Oversampling:} -anchor w
    entry $i.ovr -width 8 -textvariable ::IRspecGUI::oversampl
    checkbutton $i.plt -relief groove -text {Display Spectrum} -variable ::IRspecGUI::doplot
    checkbutton $i.sav -relief groove -text {Save to File} -variable ::IRspecGUI::dowrite
    grid $i.ovl -column 0 -row 0 -sticky snew
    grid $i.ovr -column 1 -row 0 -sticky snew
    grid $i.plt -column 2 -row 0 -sticky snew
    grid $i.sav -column 3 -row 0 -sticky snew
    grid rowconfigure $i 0 -weight 1
    grid columnconfigure $i 0 -weight 1 -minsize 10
    grid columnconfigure $i 1 -weight 2 -minsize 10
    grid columnconfigure $i 2 -weight 2 -minsize 10
    grid columnconfigure $i 3 -weight 2 -minsize 10

    UpdateMolecule
    EnDisable
    global vmd_molecule
    trace variable vmd_molecule w ::IRspecGUI::UpdateMolecule
    trace variable ::IRspecGUI::molid w ::IRspecGUI::EnDisable
}

# en-/disable buttons that depend on a proper molecule being selected
proc ::IRspecGUI::EnDisable {args} {
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
proc irspecgui_tk_cb {} {
  ::IRspecGUI::irspecgui 
  return $::IRspecGUI::w
}

# update molecule list
proc ::IRspecGUI::UpdateMolecule {args} {
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
                -command {global vmd_molecule ; if {[info exists vmd_molecule($::IRspecGUI::molid)]} {set ::IRspecGUI::moltxt "$::IRspecGUI::molid:[molinfo $::IRspecGUI::molid get name]"} {set ::IRspecGUI::moltxt "(none)" ; set molid -1} } \
                -label "$id [molinfo $id get name]" \
                -variable ::IRspecGUI::molid
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

