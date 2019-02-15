# VMD Plumed tool  - a GUI to compute collective variables
# over a trajectory
#
#     Author              Toni Giorgino  (toni.giorgino@cnr.it)
#
#     (c) 2012-           National Research Council of Italy
#     (c) 2009-2012       Universitat Pompeu Fabra 
#
#     See Toni Giorgino, "Plumed-GUI: an environment for the
#     interactive development of molecular dynamics analysis and
#     biasing scripts" (2014) Computer Physics Communications, Volume
#     185, Issue 3, March 2014, Pages 1109-1114,
#     doi:10.1016/j.cpc.2013.11.019, or arXiv:1312.3190
#
#     This program is available under either the 3-clause BSD license,
#     (e.g. see http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html)
#


# To reload:
#  destroy .plumed; source vmdplumed.tcl; plumed_tk

package provide plumed 2.7

package require Tk 8.5
package require http

# vmd_install_extension plumed plumed_tk "Analysis/Collective variable analysis (PLUMED)"

namespace eval ::Plumed:: {
    namespace export plumed
    variable plugin_name "Plumed-GUI collective variable analysis tool"

    variable debug 0;		       	# extra log info
    variable highlight_error_ms 12000;  # error message held this long
    variable plumed_default_version 2;  # default PLUMED to use if none found
    
    variable plumed2_online_docbase "http://plumed.github.io/doc-vVERSION/user-doc/html"
    variable github_repository "https://github.com/tonigi/vmd_plumed/"

    variable plot_points 0;	       	# show data markers
    variable w;				# handle to main window

    variable textfile unnamed.plumed; 	# new file name

    # Header of short help...
    variable text_instructions_header \
"Enter collective variable definitions below, in your engine's syntax.  
Click 'Plot' to evaluate them on the 'top' trajectory.  
VMD atom selections in square brackets expand automatically."

    # ...short help V1
    variable text_instructions_example_v1 \
"For example:\n
    protein-> \[chain A and name CA\] protein<-
    ligand->  \[chain B and noh\]          ligand<-\n
    DISTANCE LIST <protein> <ligand>
    COORD LIST <protein> <ligand>  NN 6 MM 12 D_0 5.0 R_0 0.5 "

    # ...short help V2
    variable text_instructions_example_v2 \
"For example:\n
    protein: COM ATOMS=\[chain A and name CA\]
    ligand:  COM ATOMS=\[chain B and noh\]\n
    DISTANCE ATOMS=protein,ligand\n
Default UNITS are nm, ps and kJ/mol unless changed.
Right mouse button provides help on keywords."

    # ...short help VMDCV
    variable text_instructions_example_vmdcv \
{For example:

    colvar {
        name this_proteins_gyration_radius
        gyration {
            atoms {
                [ protein and name CA ]       # Keyword 'atomNumbers' is implied
            }
        }
    }
}

    # Example scripts (file new)
    variable empty_meta_inp_v1 "\nDISTANCE LIST 1 200      ! Just an example\n"
    variable empty_meta_inp_v2 "
UNITS  LENGTH=A  ENERGY=kcal/mol  TIME=ps\n
d1:    DISTANCE ATOMS=1,200                     # Just an example\n"
    variable empty_meta_inp_vmdcv "
# Just an example
colvar {
    name d
    distance { 
	group1 { atomNumbers 1 }
	group2 { \[ serial 200 \] }
    }
}\n"

    # Not found text
    variable plumed_not_found_message \
"Neither PLUMED binaries 'plumed' (2.x) nor 'driver' (1.3) were found in the system's executable path.\n
You will be able to edit analysis scripts, but not to evaluate them.\n
Please see the Help menu for installation instructions or to attempt to download and install prebuilt Windows executables."

    # Attempt download message
    variable help_win32_install_query \
"Attempt download and installation of PLUMED binaries from the Plumed-GUI website?"

    variable help_win32_install_error \
"Sorry - automatic download and installation of one or both PLUMED
binaries failed. Check user permissions, network, antivirus."

    # Used in file requestors
    variable file_types {
	{"Plumed-GUI Files" { .plumed .metainp .meta_inp } }
	{"Text Files" { .txt .TXT} }
	{"All Files" * }
    }
}

proc plumed_tk {} {
    Plumed::plumed
    return $Plumed::w
}


proc ::Plumed::plumed {} {
    variable w
    variable textfile
    variable plugin_name
    variable plumed_version
    variable pbc_type 1
    variable pbc_boxx
    variable pbc_boxy
    variable pbc_boxz

    # If already initialized, just turn on
    if { [winfo exists .plumed] } {
	wm deiconify .plumed
	raise .plumed
	return
    }

    set w [toplevel ".plumed" -bg [ttk::style lookup . -background]]
    wm title $w "$plugin_name"
#    wm resizable $w 0 0


    # OS-specific UI tweaks. $mod is the menu accelerator
    set win32_install_state disabled
    global tcl_platform
    switch $tcl_platform(platform) {
	windows {
	    set win32_install_state normal
	    append ::env(PATH) ";" \
		[file nativename [file join $::env(APPDATA) "Plumed-GUI"]]
	}
    }

    lassign [getModifiers] mod modifier
    if {[tk windowingsystem] eq "aqua"} {
	event add <<B3>> \
	    <Control-ButtonPress-1> \
	    <ButtonPress-2> \
	    <ButtonPress-3>
    } else {
	event add <<B3>> <ButtonPress-3>
    }

    # Attempt to workaround https://github.com/tonigi/vmd_plumed/issues/1
    # https://github.com/yyamasak/TkSQLite-AES128/blob/master/tksqlite.tcl#L1174
    switch -exact -- $::ttk::currentTheme {
    	aqua {
    	    rename ttk::_scrollbar ttk::__scrollbar
#    	    interp alias {} ::ttk::scrollbar {} ::scrollbar
    	}
    }



    # If PBC exist, use them
    catch { molinfo top get {a b c} } vmdcell
    if { [llength $vmdcell] == 3 } {
	lassign $vmdcell a b c 
	if { [expr $a * $b * $c ] > 1.0 } {
	    lassign $vmdcell pbc_boxx pbc_boxy pbc_boxz
	}
    }
	


    ## MENU ============================================================
    menu $w.menubar

    ## file menu
    $w.menubar add cascade -label File  -underline 0 -menu $w.menubar.file
    menu $w.menubar.file -tearoff no
    $w.menubar.file add command -label "New" -command  Plumed::file_new
    $w.menubar.file add command -label "Open..." -command Plumed::file_open -acce $mod+O
    $w.menubar.file add command -label "Save" -command  Plumed::file_save -acce $mod+S
    $w.menubar.file add command -label "Save as..." -command  Plumed::file_saveas -acce $mod+Shift+S
    $w.menubar.file add command -label "Export..." -command  Plumed::file_export -acce $mod+E
    $w.menubar.file add separator
    # batch was here
    $w.menubar.file add command -label "Quit" -command  Plumed::file_quit
    bind $w <$modifier-o> Plumed::file_open
    bind $w <$modifier-s> Plumed::file_save
    bind $w <$modifier-S> Plumed::file_saveas
    bind $w <$modifier-e> Plumed::file_export
#    bind $w <$modifier-v> "::Plumed::tk_textPaste_modern $::Plumed::w.txt.text;break"

    ## edit
    $w.menubar add cascade  -label Edit -underline 0 -menu $w.menubar.edit
    menu $w.menubar.edit -tearoff no
    $w.menubar.edit add command -label "Undo" -command  "$::Plumed::w.txt.text edit undo" -acce $mod+Z
    $w.menubar.edit add command -label "Redo" -command  "$::Plumed::w.txt.text edit redo"
    $w.menubar.edit add separator
    $w.menubar.edit add command -label "Cut" -command  "tk_textCut $::Plumed::w.txt.text" -acce $mod+X
    $w.menubar.edit add command -label "Copy" -command  "tk_textCopy $::Plumed::w.txt.text" -acce $mod+C
    $w.menubar.edit add command -label "Paste" -command  "::Plumed::tk_textPaste_modern $::Plumed::w.txt.text" -acce $mod+V
    $w.menubar.edit add separator
    $w.menubar.edit add command -label "Select all" -command "$::Plumed::w.txt.text tag add sel 1.0 end" -acce $mod+A
    bind $w <$modifier-a> "$::Plumed::w.txt.text tag add sel 1.0 end"

    ## Templates
    $w.menubar add cascade -label "Templates" -underline 0 -menu $w.menubar.insert
    menu $w.menubar.insert -tearoff yes

    ## Structural
    $w.menubar add cascade -label Structure -underline 0 -menu $w.menubar.structure
    menu $w.menubar.structure -tearoff no
    $w.menubar.structure add command -label "Build reference structure..." -command Plumed::reference_gui
    $w.menubar.structure add command -label "Insert native contacts CV..." -command Plumed::nc_gui
    $w.menubar.structure add command -label "Insert backbone torsion \u03c6/\u03c8/\u03c9 CVs..." \
	-command Plumed::rama_gui
    $w.menubar.structure add command -label "Insert group for secondary structure RMSD..." \
	-command Plumed::ncacocb_gui
    $w.menubar.structure add command -label "Display gradients and forces..." \
	-command Plumed::show_forces_gui

    ## help menu
    $w.menubar add cascade -label Help -underline 0 -menu $w.menubar.help
    menu $w.menubar.help -tearoff no
    $w.menubar.help add command -label "Getting started with Plumed-GUI" \
	-command "vmd_open_url $Plumed::github_repository/blob/master/doc/README.md"
    $w.menubar.help add command -label "What to cite" \
        -command "vmd_open_url http://dx.doi.org/10.1016/j.cpc.2013.11.019"
    $w.menubar.help add separator
    $w.menubar.help add command -label "How to install PLUMED's binaries" \
	-command "vmd_open_url $Plumed::github_repository/blob/master/doc/INSTALL-PLUMED-FIRST.md"
    $w.menubar.help add command -label "Attempt download of prebuilt Windows driver binaries" \
	-command ::Plumed::help_win32_install -state $win32_install_state
    $w.menubar.help add separator
    $w.menubar.help add command -label "What is PLUMED?" \
        -command "vmd_open_url http://www.plumed.org"
    $w.menubar.help add command -label "PLUMED 2.2 user's guide and CV syntax" \
	-command "vmd_open_url $Plumed::plumed2_online_docbase/index.html"
    $w.menubar.help add command -label "PLUMED 1.3 user's guide and CV syntax" \
	-command "vmd_open_url http://www.plumed.org/documentation"
    $w.menubar.help add separator
    $w.menubar.help add command -label "VMD Colvars homepage" \
	    -command "vmd_open_url http://colvars.github.io/" 
    $w.menubar.help add command -label "VMD Colvars manual" \
	    -command "vmd_open_url http://colvars.github.io/colvars-refman-vmd/colvars-refman-vmd.html"
    $w.menubar.help add separator
    $w.menubar.help add command -label "About the $plugin_name" \
	-command [namespace current]::help_about

    $w configure -menu $w.menubar


    # Built bottom-up for... historical reasons. Also, looks like the
    # last packed widget becomes "elastic"

    ## PLOT ============================================================
    pack [  ttk::frame $w.plot ] -side bottom -fill x 
    pack [  ttk::button $w.plot.plot -text "Plot"   \
	   -command [namespace current]::do_compute ]  \
	-side left -fill x -expand 1 


    ## OPTIONS ============================================================
    pack [  ttk::labelframe $w.options -relief ridge  -text "Options"  ] \
	-side bottom -fill x

    pack [  ttk::frame $w.options.line1   ]  -side top -fill x
    pack [  ttk::frame $w.options.line1.pbc   ]  -side left -fill x
    pack [  ttk::label $w.options.line1.pbc.text -text "PBC: " ] -side left -expand 0
    pack [  ttk::radiobutton $w.options.line1.pbc.pbcno -value 1 -text "None  " \
	       -variable [namespace current]::pbc_type ] -side left
    pack [  ttk::radiobutton $w.options.line1.pbc.pbcdcd -value 2 -text "From trajectory  " \
	       -variable [namespace current]::pbc_type ] -side left
    pack [  ttk::radiobutton $w.options.line1.pbc.pbcbox -value 3 -text "Box:" \
	       -variable [namespace current]::pbc_type ] -side left
    pack [  ttk::entry $w.options.line1.pbc.boxx -width 6 -textvariable [namespace current]::pbc_boxx ] -side left
    pack [  ttk::entry $w.options.line1.pbc.boxy -width 6 -textvariable [namespace current]::pbc_boxy ] -side left
    pack [  ttk::entry $w.options.line1.pbc.boxz -width 6 -textvariable [namespace current]::pbc_boxz ] -side left

    pack [  ttk::checkbutton $w.options.line1.inspector -text "Mark data points" \
	       -variable  [namespace current]::plot_points ] -side right

    # ----------------------------------------
    pack [  ttk::frame $w.options.line2 ]  -fill x
    pack [  ttk::label $w.options.line2.version -text "Engine: " ] -side left -expand 0
    pack [  ttk::radiobutton $w.options.line2.v1 -value 1 -text "Plumed 1.3  "        \
		-variable [namespace current]::plumed_version              \
		-command [namespace current]::plumed_version_changed    	  ] -side left 
    pack [  ttk::radiobutton $w.options.line2.v2 -value 2 -text "Plumed 2.x  "         \
		-variable [namespace current]::plumed_version              \
		-command [namespace current]::plumed_version_changed       ] -side left
    pack [  ttk::radiobutton $w.options.line2.vmdcv -value vmdcv -text "VMD Colvars "         \
		-variable [namespace current]::plumed_version              \
		-command [namespace current]::plumed_version_changed       ] -side left

    pack [  ttk::label $w.options.line2.text -text "       Path to executable: " ] -side left -expand 0
    pack [  ttk::entry $w.options.line2.path -width 5 -textvariable \
	       [namespace current]::driver_path ] -side left -expand 1 -fill x
    pack [  ttk::button $w.options.line2.browse -text "Browse..." \
	   -command [namespace current]::location_browse   ] -side left -expand 0



    ## TEXT ============================================================
    ttk::frame $w.txt
    ttk::label $w.txt.label  -textvariable Plumed::textfile -anchor center
    text $w.txt.text -wrap none -undo 1 -autoseparators 1 -bg #ffffff -bd 2 \
	-yscrollcommand [list $::Plumed::w.txt.vscr set] -font {Courier 12}
	
    ttk::scrollbar $w.txt.vscr -command [list $::Plumed::w.txt.text yview]
    pack $w.txt.label -side top   -fill x 
    pack $w.txt.vscr  -side right -fill y    
    pack $w.txt.text  -side left  -fill both -expand yes
    pack $w.txt                   -fill both -expand 1 -side top


    ## POPUP ============================================================
    menu $w.txt.text.popup -tearoff 0
    bind $w.txt.text <<B3>> { ::Plumed::popup_menu %x %y %X %Y }

    ## FINALIZE ============================================================
    plumed_path_lookup;		# sets plumed_version
    file_new;			# inserts skeleton, depends on plumed_version
    instructions_update;	# because file_new inserts an empty text
}


# ==================================================

# Returns a text representation of an empty input file, depending on
# the current plumed_version
proc ::Plumed::empty_meta_inp {} {
    variable plumed_version
    variable empty_meta_inp_v1
    variable empty_meta_inp_v2
    variable empty_meta_inp_vmdcv
    switch $plumed_version {
	1  {return $empty_meta_inp_v1}
	2  {return $empty_meta_inp_v2}
	vmdcv {return $empty_meta_inp_vmdcv}
    } 
}


proc ::Plumed::file_new { } {
    variable w
    variable textfile
    set textfile "unnamed.plumed"

    $w.txt.text delete 1.0 {end - 1c}
    label $w.txt.text.instructions -text "(...)" -justify left \
	-relief solid -padx 2m -pady 2m -background #eee
    # bind $w.txt.text.instructions <1> { destroy %W }  ;# Click to close
    $w.txt.text window create 1.0 -window $w.txt.text.instructions \
	-padx 100 -pady 10
    $w.txt.text insert end [empty_meta_inp]
    instructions_update
}


proc ::Plumed::file_open { } {
    variable w
    variable textfile
    variable file_types

    set fn [tk_getOpenFile -filetypes $file_types ]
    if { $fn == "" } { return }

    if [ catch { set txt [read_file $fn] } ] {
	tk_messageBox -title "Error" -parent .plumed -message "Failed to open file $fn" -icon error
	return
    }

    $w.txt.text delete 1.0 {end - 1c}
    $w.txt.text insert end $txt
    set textfile $fn
}

proc ::Plumed::file_save { } {
    variable w
    variable textfile

    if { $textfile == "unnamed.plumed" } {
	Plumed::file_saveas
	return
    }

    set rc [ catch { set fd [open $textfile "w"] } ]
    if { $rc == 1} {
	tk_messageBox -title "Error" -parent .plumed -message "Failed to open file $textfile" -icon error
	return
    } else {
	puts $fd  [getText] 
	close $fd
    }
}

proc ::Plumed::file_saveas { } {
    variable w
    variable file_types
    variable textfile

    set ind [file dirname $textfile]
    set inf [file tail $textfile]
    set newfile [tk_getSaveFile -filetypes $file_types \
		      -initialdir $ind -initialfile $inf ]
    if { $newfile == ""} {
	return
    } else {
	set textfile $newfile
	file_save
    }
}

proc ::Plumed::file_export { } {
    variable w
    variable plumed_version
    set file_types {
	{"All Files" * }
    }
    set textfile [tk_getSaveFile -filetypes $file_types \
		       -initialfile "META_INP"       ]
    if { $textfile == "" || [ catch { set fd [open $textfile "w"] } ] == 1} {
	puts "failed to open file $textfile"
	return
    }
    puts $fd  [ Plumed::replace_serials [getText] ]
    if {$plumed_version==1} { puts $fd  "ENDMETA" }
    close $fd
}


# Well, not really quit
proc ::Plumed::file_quit { } {
    variable w
    wm withdraw $w
}

# Browse for executable
proc ::Plumed::location_browse { } {
    variable driver_path
    set tmp [ tk_getOpenFile  ]
    if { $tmp != "" } {
	set driver_path $tmp
    }
}


# Attempt auto install
proc ::Plumed::help_win32_install {} {
    variable help_win32_install_query
    variable help_win32_install_error

    set r [tk_messageBox -title "Attempt binary installation" -parent .plumed \
	       -message $help_win32_install_query -type okcancel -icon question ]
    if { $r != "ok" } { return }

    set destdir [file join $::env(APPDATA) "Plumed-GUI"]
    puts "Attempting automated installation into $destdir"
    puts "Installation may fail for permissions, network, antivirus."
    file mkdir $destdir

    set url_driver {http://tonigi.github.io/vmd_plumed/binaries/driver.exe}
    vmd_mol_urlload $url_driver [file join $destdir driver.exe]

    set url_plumed {http://tonigi.github.io/vmd_plumed/binaries/plumed.exe}
    vmd_mol_urlload $url_plumed [file join $destdir plumed.exe]

    plumed_path_lookup
}


# About dialog
proc ::Plumed::help_about { {parent .plumed} } {
    variable plugin_name
    set at @

    tk_messageBox -title "About" -parent $parent -message \
"
$plugin_name
Version loaded: [package present plumed] (available: [package versions plumed])

Toni Giorgino <toni.giorgino${at}cnr.it>
National Research Council of Italy (CNR)

Before 2012: 
Computational Biophysics Group (GRIB-IMIM-UPF),
Universitat Pompeu Fabra

Citation:
Giorgino T.  PLUMED-GUI: An environment for the interactive development of molecular dynamics analysis and biasing scripts. Comp. Phys. Comm. 2014 Mar;185(3):1109-14. arXiv:1312.3190


"
}



# ==================================================
# TCL utility functions (non plumed specific)


# http://wiki.tcl.tk/772
proc ::Plumed::tmpdir { } {
    global tcl_platform
    switch $tcl_platform(platform) {
	unix {
	    set tmpdir /var/tmp   ;# or even $::env(TMPDIR), at times.
	} macintosh {
	    set tmpdir $::env(TRASH_FOLDER)  ;# a better place?
	} default {
	    set tmpdir [pwd]
	    catch {set tmpdir $::env(TMP)}
	    catch {set tmpdir $::env(TEMP)}
	}
    }
    return $tmpdir
}

proc ::Plumed::transpose matrix {
    set cmd list
    set i -1
    foreach col [lindex $matrix 0] {append cmd " \$[incr i]"}
    foreach row $matrix {
        set i -1
        foreach col $row {lappend [incr i] $col}
    }
    eval $cmd
}


# Return the contents of a file
proc ::Plumed::read_file { fname } {
    set fd [open $fname r]
    set dtext [read $fd]
    close $fd
    return $dtext
}

# from rmsdtt
proc ::Plumed::index2rgb {i} {
  set len 2
  lassign [colorinfo rgb $i] r g b
  set r [expr int($r*255)]
  set g [expr int($g*255)]
  set b [expr int($b*255)]
  #puts "$i      $r $g $b"
  return [format "#%.${len}X%.${len}X%.${len}X" $r $g $b]
}

proc ::Plumed::dputs { text } {
    variable debug
    if {$debug} {
	puts "DEBUG: $text"
    }
}



# Attempt to access URL, return numeric code.
proc ::Plumed::get_url_ncode {url} {
    set ch [::http::geturl $url]
    set status  [::http::ncode $ch]
    ::http::cleanup $ch
    return $status
}


# Return the content of the given URL, or throw error if not
# found. Only code 200 considered OK. Unused.
proc ::Plumed::get_url {url} {
    set ch [::http::geturl $url]
    set status  [::http::ncode $ch]
    if {$status==200} {
	set data [::http::data $ch]
	::http::cleanup $ch
	return $data
    } else {
	set ecode [::http::code $ch]
	::http::cleanup $ch
	error $ecode
    }
}




# ==================================================
# TCL utility functions (Plumed specific)

# Write a PDB file using charges and masses in the topology
# (required by PLUMED's "driver" utility). 
proc ::Plumed::writePlumed { sel filename } {
    set old [ $sel get { x y z resid occupancy beta } ]

    $sel set x 0;		# workaround for xyz>1000 in PDB
    $sel set y 0
    $sel set z 0
    $sel set resid 1;	# workaround for PLUMED not reading ascii RESID
    $sel set occupancy [ $sel get mass ]	
    $sel set beta [ $sel get charge ]
    # FIXME: plumed does not read serial > 100k
    $sel writepdb $filename

    $sel set { x y z resid occupancy beta } $old
}


# Plumed v1, vmdcv: space-separated
# Plumed v2: comma-separated
proc ::Plumed::replace_serials { intxt }  {
    variable plumed_version
    set re {\[(.+?)\]}
    set lorig {};		# list of all substituted blocks
    set lnew {};		# list of replacements
    set lcount {};		# list of replacement lengths
    # Iterate over all square-bracket blocks
    while { [ regexp $re $intxt junk orig ] } {
	lappend lorig $orig
	set as [ atomselect top $orig ]
	set new [ $as get serial ]
	$as delete
	lappend lcount [llength $new]

	# adjust syntax depending on engine
	switch $plumed_version {
	    1 {	      # no change
	      }
	    2 {
		      set new [string map { " " , } $new ]
	      }
	    vmdcv {
		      set new "atomNumbers $new"
	          }
	}

	# modify intxt in-place
	lappend lnew $new
	regsub $re $intxt $new intxt
    }

    # Build output
    set out $intxt
    append out "
 
# The above script includes the following replacements, based on 
# a structure named [molinfo top get name] with [molinfo top get numatoms] atoms.\n#\n"
    foreach orig $lorig new $lnew cnt $lcount {
	append out "# \[$orig\] -> (list of $cnt atoms)\n"
    }
    return $out
}







# ==================================================

# TONI context menus and other UI stuff

# Enable/disable recursively, http://stackoverflow.com/questions/2219947/how-to-make-all-widgets-state-to-disabled-in-tk
proc ::Plumed::set_state_recursive {state path} {
    catch {$path configure -state $state}
    foreach child [winfo children $path] {
        set_state_recursive $state $child
    }
}


# Get the content of the editor
proc ::Plumed::getText {} {
	variable w
	return [$w.txt.text get 1.0 {end -1c}]
}

# Preferred modifier key: short and long name
proc ::Plumed::getModifiers {} {
    if {[tk windowingsystem] eq "aqua"} {
	return {Command Command}
    } else {
	return {Ctrl Control}
    }
}

# Paste with replacement. Copied from 8.5. I did not want to replace tk_textPaste for everyone.
proc ::Plumed::tk_textPaste_modern w {
    global tcl_platform
    if {![catch {::tk::GetSelection $w CLIPBOARD} sel]} {
	set oldSeparator [$w cget -autoseparators]
	if {$oldSeparator} {
	    $w configure -autoseparators 0
	    $w edit separator
	}
	if {[tk windowingsystem] ne "x11-OLDWAY"} {
	    catch { $w delete sel.first sel.last }
	}
	$w insert insert $sel
	if {$oldSeparator} {
	    $w edit separator
	    $w configure -autoseparators 1
	}
    }
}


# http://www.megasolutions.net/tcl/right-click-menu-49868.aspx
# http://wiki.tcl.tk/16317
# possibly replace by tklib version
proc ::Plumed::setBalloonHelp {w msg args} {
  array set opt [concat {
      -tag ""
    } $args]
  if {$msg ne ""} then {
    set toolTipScript\
	[list [namespace current]::showBalloonHelp %W [string map {% %%} $msg]]
    set enterScript [list after 1000 $toolTipScript]
    set leaveScript [list after cancel $toolTipScript]
    append leaveScript \n [list after 200 [list destroy .balloonHelp]]
  } else {
    set enterScript {}
    set leaveScript {}
  }
  if {$opt(-tag) ne ""} then {
    switch -- [winfo class $w] {
      Text {
        $w tag bind $opt(-tag) <Enter> $enterScript
        $w tag bind $opt(-tag) <Leave> $leaveScript
      }
      Canvas {
        $w bind $opt(-tag) <Enter> $enterScript
        $w bind $opt(-tag) <Leave> $leaveScript
      }
      default {
        bind $w <Enter> $enterScript
        bind $w <Leave> $leaveScript
      }
    }
  } else {
    bind $w <Enter> $enterScript
    bind $w <Leave> $leaveScript
  }
}

proc ::Plumed::showBalloonHelp {w msg} {
  set t .balloonHelp
  catch {destroy $t}
  toplevel $t -bg black
  wm overrideredirect $t yes
  if {$::tcl_platform(platform) == "macintosh"} {
    unsupported1 style $t floating sideTitlebar
  }
  pack [label $t.l -text [subst $msg] -bg yellow -font {Helvetica 12}]\
    -padx 1\
    -pady 1
  set width [expr {[winfo reqwidth $t.l] + 2}]
  set height [expr {[winfo reqheight $t.l] + 2}]
  set xMax [expr {[winfo screenwidth $w] - $width}]
  set yMax [expr {[winfo screenheight $w] - $height}]
  set x [winfo pointerx $w]
  set y [expr {[winfo pointery $w] + 20}]
  if {$x > $xMax} then {
    set x $xMax
  }
  if {$y > $yMax} then {
    set y $yMax
  }
  wm geometry $t +$x+$y
  set destroyScript [list destroy .balloonHelp]
  bind $t <Enter> [list after cancel $destroyScript]
  bind $t <Leave> $destroyScript
}


# RAMACHANDRAN ==================================================                                                 

proc ::Plumed::rama_gui { } {
    if { [winfo exists .plumedrama] } {
	wm deiconify .plumedrama
	raise .plumedrama
	return
    }

    variable rama_sel "protein"
    variable rama_phi 1
    variable rama_psi 1
    variable rama_omega 0

    toplevel .plumedrama -bd 4
    wm title .plumedrama "Insert backbone torsion CVs"
    pack [ ttk::label .plumedrama.head1 -text "Insert torsion CVs for the backbone of the matched residues.
N-CA-C atom naming is assumed for backbone atoms.
Dihedrals involving atoms outside the selection are skipped.
"  -justify center -anchor center -pad 3 ] -side top -fill x 

    pack [ ttk::frame .plumedrama.sel ] -side top -fill x
    pack [ ttk::label .plumedrama.sel.txt -text "Selection: backbone and " ] -side left -fill x
    pack [ ttk::entry .plumedrama.sel.in -width 20 -textvariable [namespace current]::rama_sel ] -side left -expand 1 -fill x

    pack [ ttk::frame .plumedrama.cv ] -side top -fill x
    pack [ ttk::label .plumedrama.cv.txt -text "Dihedral angles: " ] -side left -fill x
    pack [ ttk::checkbutton .plumedrama.cv.phi -text Phi -variable  [namespace current]::rama_phi ] -side left
    pack [ ttk::checkbutton .plumedrama.cv.psi -text Psi -variable  [namespace current]::rama_psi ] -side left
    pack [ ttk::checkbutton .plumedrama.cv.omega -text Omega -variable  [namespace current]::rama_omega ] -side left

    pack [ ttk::frame .plumedrama.act ] -side top -fill x
    pack [ ttk::button .plumedrama.act.ok -text "Insert"  -command \
	       { Plumed::rama_insert } ] -side left -fill x -expand 1
    pack [ ttk::button .plumedrama.act.close -text "Close"  \
	       -command {  destroy .plumedrama }   ] -side left -fill x -expand 1

}

# Insert Ramachandran angles. Uses the following atoms
#       Cm   N CA C   Np CAp
# phi    +   +  + +
# psi        +  + +   +
# omega         + +   +  +
# where "m/p" means previous/next residue Computation is done on the
# base of "residue" (unique), but they will be printed as "resid"
# (human-readable)

proc ::Plumed::rama_insert {} {
    variable rama_sel
    variable rama_phi
    variable rama_psi
    variable rama_omega
    variable w

    set nnew 0

    set sel [atomselect top "($rama_sel) and name CA and backbone"]
    set rlist [lsort -integer -uniq [$sel get residue]]
    $sel delete

    if {[llength $rlist] == 0} {
	tk_messageBox -icon error -title "Error" -parent .plumedrama -message "Selection is empty."
	return	
    }

    foreach r $rlist {
	if {$r == 0} {
	    set rm1 1000000;	# non-existent residue, kludge not to atomselect a negative one
	} else {
	    set rm1 [expr $r-1]
	}
	set rp1 [expr $r+1]
	set Cm  [atomselect top "($rama_sel) and backbone and residue $rm1 and name C"]
	set N   [atomselect top "($rama_sel) and backbone and residue $r and name N"]
	set CA  [atomselect top "($rama_sel) and backbone and residue $r and name CA"]
	set C   [atomselect top "($rama_sel) and backbone and residue $r and name C"]
	set Np  [atomselect top "($rama_sel) and backbone and residue $rp1 and name N"]
	set CAp [atomselect top "($rama_sel) and backbone and residue $rp1 and name CA"]
	set rid [format "%s%d" [$CA get resname] [$CA get resid]]; # human readable
	if {$rama_phi && [rama_insert_cv_maybe $Cm $N $CA $C             PHI $rid] } { 	incr nnew }
	if {$rama_psi && [rama_insert_cv_maybe     $N $CA $C  $Np        PSI $rid] } {	incr nnew }
	if {$rama_omega && [rama_insert_cv_maybe      $CA $C  $Np $CAp  OMEGA $rid] } { incr nnew }
	$Cm delete; 
	$N delete; $CA delete; $C delete
	$Np delete; $CAp delete
    }
    $w.txt.text insert insert "# The above list contains $nnew backbone torsion CVs\n" 
}


# Return the line computing a torsion CV defined by the arguments iff all of them are valid
proc ::Plumed::rama_insert_cv_maybe {A B C D angle rid} {
    variable w
    variable plumed_version
    set cv_lines_v1_v2 { - "TORSION LIST %d %d %d %d  ! %s_%s\n"
	                   "TORSION ATOMS=%d,%d,%d,%d  LABEL=%s_%s\n" }
    set oos_msg "# No dihedral %s for residue %s: out of selection\n"
    set topo_msg "# No dihedral %s for residue %s: chain break\n"
    if { [$A num]==0 || [$B num]==0 || [$C num]==0 || [$D num]==0 } {
	set r [format $oos_msg $angle $rid]
	set ok 0
    } elseif { [llength [lsort -uniq -integer [list \
  	         [$A get fragment] [$B get fragment] [$C get fragment] [$D get fragment] ]]] != 1 } {
	# above; check that all fragment IDs are equal (check enforced on topology)
	# could also be segid or chain (check enforced on "logical" structure)
	set r [format $topo_msg $angle $rid]
	set ok 0
    } else {
	set cv_line [lindex $cv_lines_v1_v2 $plumed_version ]
	set r [format $cv_line  \
		   [$A get serial] [$B get serial] [$C get serial] [$D get serial]  \
		   $rid $angle ]
	set ok 1
    } 
    $w.txt.text insert insert $r
    return $ok
}


# SECONDARY RMSD aka N-CA-C-O-CB REFERENCE ==================================================                                                 

# Alpha, parabeta, antibeta ordered lists ==================================================  
proc ::Plumed::secondary_rmsd {N CA C O CB} {
    set Nsel [atomselect top $N]
    set CAsel [atomselect top $CA]
    set Csel [atomselect top $C]
    set Osel [atomselect top $O]
    set CBsel [atomselect top $CB]

    set ret {}
    set lens [list [$Nsel num] [$CAsel num] [$Csel num] [$Osel num] [$CBsel num]]
    set lens [lsort -integer -uniq $lens]
    if {[llength $lens] != 1} {
	error "Atom selection lengths are different: [$Nsel num] C, [$CAsel num] CA, [$Csel num] C, [$Osel num] O, [$CBsel num] CB"
    } else {
	set ret [list [$Nsel get serial] [$CAsel get serial] [$Csel get serial] [$Osel get serial] [$CBsel get serial]]
	set ret [transpose $ret]; # transpose
	set ret [concat {*}$ret]; # flatten
    }

    $Nsel delete
    $CAsel delete
    $Csel delete
    $Osel delete
    $CBsel delete
    return $ret
}


proc ::Plumed::ncacocb_gui {} {
    set n .plumed_ncacocb
    if { [winfo exists $n] } {
	wm deiconify $n
	raise $n
	return
    }

    variable ncacocb_com  "protein"
    variable ncacocb_N    "name N"
    variable ncacocb_CA   "name CA"
    variable ncacocb_C    "name C"
    variable ncacocb_O    "name O"
    variable ncacocb_CB   "name CB or (resname GLY and name HA2)"
    variable ncacocb_grp  "bb"

    toplevel $n -bd 4
    wm title $n "Insert group for secondary structure RMSD"

    pack [ ttk::label $n.head1 -text "Build a group containing atoms N, CA, C, O, CB in a selection for\nuse with the ALPHARMSD, ANTIBETARMSD, PARABETARMSD CVs.\n(Plumed 1 only.)"  -justify center -anchor center -pad 3 ] -side top -fill x 

    # http://wiki.tcl.tk/1433
    pack [ ttk::frame $n.tab ] -side top -fill x -expand 1

    # Group name
    set a grp
    set l [ttk::label $n.tab.l$a -text "Group name "]
    set e [ttk::entry $n.tab.e$a -justify center -textvariable "Plumed::ncacocb_grp" ]
    grid $l $e; grid $l -sticky e; grid $e -sticky ew

    # Selection
    set a com
    set l [ttk::label $n.tab.l$a -text "Selection "]
    set e [ttk::entry $n.tab.e$a -justify center -textvariable "Plumed::ncacocb_com" ]
    grid $l $e; grid $l -sticky e; grid $e -sticky ew

    # and
    grid x [ttk::label $n.tab.and -text "and"] 

    foreach a {N CA C O CB} {
	set l [ttk::label $n.tab.l$a -text "selection for $a "]
	set e [ttk::entry $n.tab.e$a -justify center -textvariable "Plumed::ncacocb_$a" ]
	grid $l $e 
	grid $l -sticky e
	grid $e -sticky ew
    }
    grid columnconfigure .plumed_ncacocb.tab 1 -weight 1

    # Spacer
    pack [ ttk::label $n.spacer ] -side top -fill x

    # Insert / Close
    pack [ ttk::frame $n.act ] -side top -fill x
    pack [ ttk::button $n.act.insert -text "Insert"  -command \
	       { Plumed::ncacocb_insert } ] -side left -fill x -expand 1
    pack [ ttk::button $n.act.close -text "Close"  \
	       -command "destroy $n"   ] -side left -fill x -expand 1

}

proc ::Plumed::ncacocb_insert {} {
    variable w
    variable ncacocb_com
    variable ncacocb_N    
    variable ncacocb_CA   
    variable ncacocb_C    
    variable ncacocb_O    
    variable ncacocb_CB   
    variable ncacocb_grp

    if [catch {
	secondary_rmsd "($ncacocb_com) and ($ncacocb_N)" \
	    "($ncacocb_com) and ($ncacocb_CA)" \
	    "($ncacocb_com) and ($ncacocb_C)" \
	    "($ncacocb_com) and ($ncacocb_O)" \
	    "($ncacocb_com) and ($ncacocb_CB)"
    } e ] {
	puts "Error: $e"
	tk_messageBox -title "Error" -parent .plumed_ncacocb -message "$e" -icon error
	return
    } 

    set nl [expr {[llength $e]/5.}]
    set s "$ncacocb_grp-> $e $ncacocb_grp<-"
    $w.txt.text insert insert "\n$s\n" 
    $w.txt.text insert insert "# The above list contains backbone definition for $nl residues: $ncacocb_com and, resp., [list  $ncacocb_N $ncacocb_CA $ncacocb_C $ncacocb_O $ncacocb_CB]\n" 
}



# BUILD REFERENCE ==================================================                                                 

proc ::Plumed::reference_gui { } {
    if { [winfo exists .plumedref] } {
	wm deiconify .plumedref
	raise .plumedref
	return
    }

    variable plumed_version
    variable refalign "backbone"
    variable refmeas "name CA"
    variable reffile "reference.pdb"
    variable refmol top
    variable ref_allframes 1
    variable ref_mindmsd 0
    variable ref_status_text "(No reference written yet)"

    set ref_allframes_state normal
    if {$plumed_version==1} { 
	set ref_allframes_state disabled 
	set ref_allframes 0
    }

    toplevel .plumedref -bd 4 -bg [ttk::style lookup . -background]
    wm title .plumedref "Build reference structure"
    pack [ ttk::label .plumedref.title -text "Convert top molecule's frames into\na reference file for MSD-type analysis:" -justify center -anchor center -pad 3] -side top -fill x -expand 1
    pack [ ttk::frame .plumedref.align ] -side top -fill x
    pack [ ttk::label .plumedref.align.aligntext -text "Alignment set: " ] -side left
    pack [ ttk::entry .plumedref.align.align -width 20 -textvariable [namespace current]::refalign ] -side left -expand 1 -fill x
    pack [ ttk::frame .plumedref.meas ] -side top -fill x
    pack [ ttk::label .plumedref.meas.meastext -text "Displacement set: " ] -side left
    pack [ ttk::entry .plumedref.meas.meas -width 20 -textvariable [namespace current]::refmeas ] -side left -expand 1 -fill x
    pack [ ttk::frame .plumedref.mol ] -side top -fill x
    pack [ ttk::label .plumedref.mol.moltext -text "Renumber for molecule id: " ] -side left
    pack [ ttk::entry .plumedref.mol.mol -width 20 -textvariable [namespace current]::refmol ]\
	-side left -expand 1 -fill x
    pack [ ttk::frame .plumedref.file ] -side top -fill x
    pack [ ttk::label .plumedref.file.filetxt -text "File to write: " ] -side left
    pack [ ttk::entry .plumedref.file.file -width 20 -textvariable [namespace current]::reffile ]\
	-side left -expand 1 -fill x
    pack [ ttk::button .plumedref.file.filebrowse -text "Browse..." -command { 
	Plumed::reference_set_reffile [ tk_getSaveFile  -initialfile "$::Plumed::reffile" ] 
    } ] -side left -expand 0
    pack [ ttk::checkbutton .plumedref.multiframe -text "Multi-frame reference (all loaded frames)" \
	       -variable [namespace current]::ref_allframes -state $ref_allframes_state ] -side top -fill x
    pack [ ttk::frame .plumedref.subset ] -side top -fill x
    pack [ ttk::label .plumedref.subset.text -text "Minimum MSD between consecutive frames (Å²): " ] -side left
    pack [ ttk::entry .plumedref.subset.val -width 20 -textvariable [namespace current]::ref_mindmsd ]\
	-side left -expand 1 -fill x
    
    pack [ ttk::labelframe .plumedref.status  -text "Output" -padding 3]  -side top -fill x
    pack [ ttk::label .plumedref.status.text  -textvariable [namespace current]::ref_status_text ] -side top 

    pack [ ttk::frame .plumedref.act ] -side top -fill x
    pack [ ttk::button .plumedref.act.ok -text "Write" \
	       -command { Plumed::reference_write } ] -side left -fill x -expand 1
    pack [ ttk::button .plumedref.act.cancel -text "Close" \
	       -command {  destroy .plumedref }   ] -side left -fill x -expand 1
}


proc ::Plumed::reference_set_reffile { x } { 
    variable reffile; 
    if { $x != "" } {set reffile $x} 
}; # why??



proc ::Plumed::reference_write {} {
    variable ref_allframes
    variable reffile
 
    if [ catch {
	if { $ref_allframes == 0 } {
	    set fn [molinfo top get frame]
	    reference_write_subset $reffile $fn 
	    puts "File $reffile written."
	} else {
	    set subset [reference_compute_subset]
	    reference_write_subset $reffile $subset
	    puts "Multi-frame $reffile written with full trajectory. See $reffile.log."
	}
    } exc ] {
	tk_messageBox -title "Error" -parent .plumedref -message $exc -icon error
    }
}


# Subset the trajectory according to the "skip until rmsd at least"
# criterion. Assemble result
proc ::Plumed::reference_compute_subset {} {
    variable refalign
    variable refmeas
    variable ref_mindmsd
    variable reffile

    set N [molinfo top get numframes]
    if {$N<2} {	error "At least two frames needed" }

    set lf [open "$reffile.log" w]

    set sel_fr 0;		# selected frames (first is always selected)
    set sel_dr {};		# selected delta MSD
    
    set selalign_i [atomselect top $refalign]
    set selalign_j [atomselect top $refalign]
    set selmeas_i [atomselect top $refmeas]
    set selmeas_j [atomselect top $refmeas]

    set i 0;
    while {$i<[expr $N-1]} {
	$selalign_i frame $i
	$selmeas_i frame $i
	for {set j [expr $i+1]} {$j<$N} {incr j} {
	    $selalign_j frame $j
	    $selmeas_j frame $j
	    set msd_ij [expr [rmsd_1 $selmeas_i $selmeas_j $selalign_i $selalign_j]**2]
	    puts $lf "# MSD($i->$j) = $msd_ij"
	    if {$msd_ij>=$ref_mindmsd} {
		lappend sel_dr $msd_ij
		lappend sel_fr $j
		puts $lf "# Frame $j selected"
		break
	    }
	}
	set i $j
    }

    # Iterate over all selected pairs
    for {set ii 0} {$ii<[llength $sel_fr]} {incr ii} {
	set i [lindex $sel_fr $ii]
	$selalign_i frame $i
	$selmeas_i frame $i
	for {set jj 0} {$jj<[llength $sel_fr]} {incr jj} {
	    set j [lindex $sel_fr $jj]
	    $selalign_j frame $j
	    $selmeas_j frame $j
	    set msd_ij [expr [rmsd_1 $selmeas_i $selmeas_j $selalign_i $selalign_j]**2]
	    puts $lf "$ii $jj $msd_ij"
	}
	puts $lf "";		# keep gnuplot pm3d happy
    }

    $selalign_i delete;     $selalign_j delete;
    $selmeas_i delete;      $selmeas_j delete; 

    set nsel [llength $sel_fr]
    if [catch {
	set avg [format "%.2f" [vecmean $sel_dr]]
	set lambda [format "%.2f" [expr 2.3/$avg]]
    }] {
	set avg "---"
	set lambda "---"
    }

    puts $lf "# Frames selected: [llength $sel_fr]"
    puts $lf "# Frames: $sel_fr"
    puts $lf "# Average MSD: $avg"
    close $lf

    reference_update_status $nsel $avg $lambda
    return $sel_fr
}


# Set the status message. FIXME encoding fail in win32
proc ::Plumed::reference_update_status {nf admsd lambda} {
    variable ref_status_text;
    set ref_status_text [format "Frames written: %d
Average ΔMSD: %s Å²
Suggested λ: %s/Å²" $nf $admsd $lambda]
}


# Compute rmsd of all frames of sel wrt currently selected frame in
# ref.  Align ref1 to ref2, and measure RMSD of sel1
# wrt sel2. sel1 and ref1 should belong to the same molecule (the
# trajectory under study, multiple frames).  Sel2 and ref2 should
# belong to the same molecule (the reference frame).  
proc ::Plumed::rmsd_1 { sel1 sel2 ref1 ref2 } {
    set oco [ $sel1 get { x y z } ]
    set xform [measure fit $ref1 $ref2]
    $sel1 move $xform
    if {$sel2 != "ROTATE"} {
	set rmsd [measure rmsd $sel1 $sel2]
	$sel1 set {x y z} $oco
    }
    return $rmsd
}






# Uses class variables to get the selection strings
proc ::Plumed::reference_write_subset { fileout subset } {
    variable refalign
    variable refmeas
    variable refmol

    # From where new serials are taken
    set asnew [ atomselect $refmol "($refalign) or ($refmeas)" ]
    set newserial [ $asnew  get serial ]
    $asnew delete

    set asref [ atomselect top "($refalign) or ($refmeas)" ]
    set oldserial [ $asref  get serial ]

    
    if { [llength $oldserial] != [llength $newserial] } {
	$asref delete
	error "Selection ($refalign) or ($refmeas) matches a different number of atoms in molecule $refmol ([llength $newserial] matched atoms) with respect to the top molecule ([llength $oldserial] atoms)."
    }

    set asall [ atomselect top all]
    set asalign [ atomselect top $refalign ] 
    set asmeas  [ atomselect top $refmeas ] 

    set old [ $asall get {occupancy beta segname} ]; # backup
    
    $asall set occupancy 0
    $asall set beta 0
    $asall set segname XXXX
    
    $asalign set occupancy 1
    $asmeas  set beta 1
    $asref   set segname YYYY

    set tmpf [ file join [ Plumed::tmpdir ] "reftmp.[pid].pdb" ]
    set pdb_text {}
    
    foreach fn $subset {
	animate write pdb $tmpf beg $fn end $fn waitfor all
	set fd [open $tmpf r]
	append pdb_text [read $fd]
	close $fd
    }
    file delete $tmpf

    $asall set {occupancy beta segname} $old; # restore
    $asall delete
    $asref delete
    $asalign delete
    $asmeas delete

    # i.e. grep YYYY $tmpd/reftmp.pdb > $fileout
    # plumed <1.3 had a bug in PDB reader, which required
    # non-standard empty chain: ## set line [string replace $line 21 21 " "]
    set fdw [ open $fileout w ]
    set i 0
    foreach line [split $pdb_text "\n"] {
	# Only passthrough lines marked with YYYY
	if [ regexp {YYYY} $line ] {
	    # replace serial
	    set line [string replace $line 6 10 \
			  [ format "%5s" [ lindex $newserial $i ] ] ]
	    puts $fdw $line
	    incr i
	} elseif [ regexp END $line ] { # and ENDs
	    puts $fdw $line
	    set i 0
	}
    }
    close $fdw
}





# NATIVE CONTACTS ==================================================

proc ::Plumed::nc_gui { } { 
    if { [winfo exists .plumednc] } {
	wm deiconify .plumednc
	raise .plumednc
	return
    }

    variable nc_selA "protein and name CA"
    variable nc_selB ""
    variable nc_cutoff 7
    variable nc_dresid 0
    variable nc_destmol top
    variable nc_groupname nc
    variable nc_implementation coordination
    variable plumed_version

    set nc_impl_state normal
    if {$plumed_version==1} { 
	set nc_impl_state disabled 
    }

    toplevel .plumednc -bd 4 -bg [ttk::style lookup . -background]
    wm title .plumednc "Insert native contacts CV"
    pack [ ttk::label .plumednc.head1 -text "Insert a CV and group definitions required to define a native contacts CV.
The current frame of the top molecule is taken as the native state."  -justify center -anchor center -pad 3 ] -side top -fill x 

    pack [ ttk::frame .plumednc.sel1 ] -side top -fill x
    pack [ ttk::label .plumednc.sel1.txt -text "Selection 1: " ] -side left -fill x
    pack [ ttk::entry .plumednc.sel1.sel -width 50 -textvariable [namespace current]::nc_selA ] -side left -expand 1 -fill x

    pack [ ttk::frame .plumednc.sel2 ] -side top -fill x
    pack [ ttk::label .plumednc.sel2.txt -text "Selection 2 (optional): " ] -side left -fill x
    pack [ ttk::entry .plumednc.sel2.sel -width 40 -textvariable [namespace current]::nc_selB ] -side left -expand 1 -fill x

    pack [ ttk::frame .plumednc.cutoff ] -side top -fill x
    pack [ ttk::label .plumednc.cutoff.txt -text "Distance cutoff (\u00C5): " ] -side left -fill x
    pack [ ttk::entry .plumednc.cutoff.sel -width 10 -textvariable [namespace current]::nc_cutoff ] -side left -expand 1 -fill x
    pack [ ttk::label .plumednc.cutoff.txt2 -text "      Single selection: |\u0394 resid| \u2265 " ] -side left -fill x
    pack [ ttk::entry .plumednc.cutoff.dresid -width 10 -textvariable [namespace current]::nc_dresid ] -side left -expand 1 -fill x
    Plumed::setBalloonHelp .plumednc.cutoff.dresid "Consider contact pairs only if they span at least N monomers in the sequence (by resid attribute). 
   0 - consider all contact pairs;
   1 - ignore contacts within the same residue;
   2 - also ignore contacts between neighboring monomers; and so on."

    pack [ ttk::frame .plumednc.destmol ] -side top -fill x
    pack [ ttk::label .plumednc.destmol.txt -text "Renumber for molecule id: " ] -side left -fill x
    pack [ ttk::entry .plumednc.destmol.sel -width 10 -textvariable [namespace current]::nc_destmol ] -side left -expand 1 -fill x

    pack [ ttk::frame .plumednc.groupname ] -side top -fill x
    pack [ ttk::label .plumednc.groupname.txt -text "Label for PLUMED groups/CV: " ] -side left -fill x
    pack [ ttk::entry .plumednc.groupname.sel -width 20 -textvariable [namespace current]::nc_groupname ] -side left -expand 1 -fill x

    pack [ ttk::frame .plumednc.impl ] -side top -fill x
    pack [ ttk::label .plumednc.impl.txt -text "Implementation: " ] -side left -fill x
    pack [ ttk::radiobutton .plumednc.impl.coordination -value coordination -text "COORDINATION  " \
	       -variable [namespace current]::nc_implementation -state $nc_impl_state   	  ] -side left 
    pack [ ttk::radiobutton .plumednc.impl.distances -value distances -text "DISTANCES  "        \
	       -variable [namespace current]::nc_implementation -state $nc_impl_state    	  ] -side left 

    pack [ ttk::label .plumednc.preview -text "Click `Count' to compute the number of contacts." ] -side top -fill x 

    pack [ ttk::frame .plumednc.act ] -side top -fill x
    pack [ ttk::button .plumednc.act.preview -text "Count"  -command \
	       { Plumed::nc_preview } ] -side left -fill x -expand 1 
    pack [ ttk::button .plumednc.act.insert -text "Insert"  -command \
	       { Plumed::nc_insert } ] -side left -fill x -expand 1 
    pack [ ttk::button .plumednc.act.close -text "Close"  \
	       -command {  destroy .plumednc }   ] -side left -fill x -expand 1
}


proc ::Plumed::nc_compute { } {
    variable nc_selA
    variable nc_selB
    variable nc_cutoff
    variable nc_destmol
    variable nc_dresid
    
    # See RMSD trajectory tool enhanced with native contacts
    # http://www.multiscalelab.org/utilities/RMSDTTNC
    set sel1 [ atomselect top $nc_selA ] 
    set sel2 [ atomselect $nc_destmol $nc_selA ]
    if { [$sel1 num] != [$sel2 num] } {
	tk_messageBox -title "Error" -parent .plumednc -message "Selection ($nc_selA) has different number of atoms in molecule top ([$sel1 num]) versus $nc_destmol ([$sel2 num])." -icon error
	return
    }

    if { $nc_selB != "" } {
	set sel1B [ atomselect top $nc_selB ] 
	set sel2B [ atomselect $nc_destmol $nc_selB ]
	if { [$sel1B num] != [$sel2B num] } {
	    tk_messageBox -title "Error" -parent .plumednc -message "Selection ($nc_selB) has different number of atoms in molecule top ([$sel1B num]) versus $nc_destmol ([$sel2B num])." -icon error
	    return
	}
    } else {
	set sel1B 0
	set sel2B 0 
    }

    # mapping index of top -> serials of $nc_destmol
    # sel1 and sel1B are in top
    # sel2 and sel2B are in $nc_destmol
    array set i1_s2 {}
    foreach idx1 [$sel1 get index]   ser2 [$sel2 get serial] {
	set i1_s2($idx1) $ser2
    }

    # Add atoms in selB if intermolecular
    if { $sel1B != 0 } {
	foreach idx1 [$sel1B get index]  ser2 [$sel2B get serial] {
	    set i1_s2($idx1) $ser2
	}
    }

    # Prepare list of resids in reference
    array set i1resid {}
    if { $sel1B == 0 } {
	foreach i [$sel1 get index] resid [$sel1 get resid] {
	    set i1resid($i) $resid
	}
    }

    # Get native contacts (sel1)
    if { $sel1B != 0 } {
	set nclist [transpose [ measure contacts $nc_cutoff $sel1 $sel1B ] ]
	puts "\nDEBUG: reference is [$sel1 text], sel1B is [$sel1B text], nclist has [llength $nclist]"
    } else {
	set nclist [transpose [ measure contacts $nc_cutoff $sel1 ] ]
    }
    set ncref_full [ llength $nclist ]

    # Convert pair list as atomnos, removing close pairs if needed
    set ncl {}
    foreach pairs $nclist {
	set i1 [lindex $pairs 0]
	set i2 [lindex $pairs 1]
	if { $sel1B == 0 } {
	    if { [ expr abs( $i1resid($i1) - $i1resid($i2) ) ] < $nc_dresid } {
		puts "DEBUG: Removing contact pair $i1-$i2 (resid $i1resid($i1) - $i1resid($i2) )"
		continue
	    }
	}
	set p1 $i1_s2($i1)
	set p2 $i1_s2($i2)
	lappend ncl [ list $p1 $p2 ]
    }

    puts "CONTACTS: [ llength $nclist ] in reference, [ llength $ncl ] after removing close resids"
    return $ncl
}

proc ::Plumed::nc_preview { } {
    .plumednc.preview configure -text "Counting, please wait..."
    update
    set ncl [ Plumed::nc_compute ]
    set ncn [ llength $ncl ]
    puts "NC: $ncl "
    .plumednc.preview configure -text "There are $ncn native contacts."
}


proc ::Plumed::nc_insert { } {
    variable nc_groupname 
    variable nc_cutoff
    variable nc_implementation
    variable plumed_version
    variable w

    set nc [ Plumed::nc_compute ]
    .plumednc.preview configure -text "There are [llength $nc] native contacts."
    if { [llength $nc ] == 0 } {
	tk_messageBox -title "Error" -parent .plumednc -message "There are no contacts in the currently selected frame." -icon error
	return
    }
    set ncl [ transpose $nc  ]

    switch $plumed_version {
	1 {
	    append txt1 "${nc_groupname}_1-> [lindex $ncl 0] ${nc_groupname}_1<-\n"
	    append txt1 "${nc_groupname}_2-> [lindex $ncl 1] ${nc_groupname}_2<-\n\n"
	    append txtX "COORD LIST <${nc_groupname}_1> <${nc_groupname}_2> PAIR NN 6 MM 12 D_0 $nc_cutoff R_0 0.5\n"
	} 
	2 {
	    switch $nc_implementation {
		coordination {
		    append txt1 ""
		    append txtX "${nc_groupname}_a: GROUP ATOMS={[lindex $ncl 0]}\n"
		    append txtX "${nc_groupname}_b: GROUP ATOMS={[lindex $ncl 1]}\n"
		    append txtX "$nc_groupname:   COORDINATION GROUPA=${nc_groupname}_a GROUPB=${nc_groupname}_b  PAIR  D_0=$nc_cutoff R_0=0.5\n"
		}
		distances {
		    set i 0
		    set tmp ""
		    foreach a1 [lindex $ncl 0] a2 [lindex $ncl 1] {
			incr i
			append tmp " ATOMS$i=$a1,$a2"
		    }
		    append txt1 ""
		    append txtX "$nc_groupname:   DISTANCES   LESS_THAN={RATIONAL R_0=0.5 D_0=$nc_cutoff}  $tmp\n"
		}
	    }
	}
    }
    $w.txt.text insert 1.0 "$txt1"
    $w.txt.text insert insert "$txtX"
}

# ERROR HANDLING ==================================================


proc ::Plumed::get_label_from_line {line} {
    

}


proc ::Plumed::get_action_from_line {line} {
    

}



# Lookup label in v2 syntax
proc ::Plumed::highlight_error_label {label etext} {
    variable w
    variable highlight_error_ms
    set t $w.txt.text

    # match label prefixed by word boundary and followed by colon, or
    # prefixed by LABEL= and followed by word boundary. 
    set pos [$t search -regexp "(\\y$label:|LABEL=$label\\y)" 1.0]

    # The first half of the regexp should match only at the beginning
    # of the line, but some bug is gobbling all whitespace in
    # preceding lines.
    ## set pos [$t search -regexp "(^\\s*?$label:|LABEL=$label\\y)" 1.0]

    if {$pos != ""} {
	dputs "Label found at $pos"
	$t see $pos
	# lassign [split $pos .] line char
	# if {$char == 0} { incr line; set pos "$line.1" }
	## NOW highlight the line, show error, wait, remove hl
	$t tag add errorTag "$pos linestart" "$pos lineend"
	$t tag configure errorTag -background yellow -foreground red
	setBalloonHelp $t $etext -tag errorTag
	after $highlight_error_ms "$w.txt.text tag delete errorTag"
    } else {
	puts "Label not found in text area"
    }
}


# Handle version changes ==================================================


proc ::Plumed::structuremenu_update {} {
    variable plumed_version
    switch $plumed_version {
	1 {
	    .plumed.menubar entryconfigure 4 -state normal
	    .plumed.menubar.structure entryconfigure 0 -state normal
	    .plumed.menubar.structure entryconfigure 1 -state normal
	    .plumed.menubar.structure entryconfigure 2 -state normal
	    .plumed.menubar.structure entryconfigure 3 -state normal
	    .plumed.menubar.structure entryconfigure 4 -state disabled
	}
	2 {
	    .plumed.menubar entryconfigure 4 -state normal
	    .plumed.menubar.structure entryconfigure 0 -state normal
	    .plumed.menubar.structure entryconfigure 1 -state normal
	    .plumed.menubar.structure entryconfigure 2 -state normal
	    .plumed.menubar.structure entryconfigure 3 -state disabled
	    .plumed.menubar.structure entryconfigure 4 -state normal
	    destroy .plumed_ncacocb
	}
	vmdcv {
	    .plumed.menubar entryconfigure 4 -state disabled
	    .plumed.menubar.structure entryconfigure 0 -state disabled
	    .plumed.menubar.structure entryconfigure 1 -state disabled
	    .plumed.menubar.structure entryconfigure 2 -state disabled
	    .plumed.menubar.structure entryconfigure 3 -state disabled
	    .plumed.menubar.structure entryconfigure 4 -state disabled
	    destroy .plumedref
	    destroy .plumednc
	    destroy .plumedrama
	    destroy .plumed_ncacocb
	}
    }
}



proc ::Plumed::instructions_update {} {
    variable w
    variable plumed_version
    variable text_instructions_header
    variable text_instructions_example_v1
    variable text_instructions_example_v2
    variable text_instructions_example_vmdcv
    switch $plumed_version {
	1 { set txt "$text_instructions_header $text_instructions_example_v1" }
	2 { set txt "$text_instructions_header $text_instructions_example_v2" }
	vmdcv { set txt "$text_instructions_header $text_instructions_example_vmdcv" }
    }
    catch { $w.txt.text.instructions configure -text $txt } err
}


# Look for plumed (v2), then driver (v1)
proc ::Plumed::plumed_path_lookup {} {
    variable plumed_version
    variable plumed_default_version
    variable driver_path_v1 "(Not found)"
    variable driver_path_v2 "(Not found)"
    variable driver_path
    variable plumed_not_found_message

    set plumed_version 0
    auto_reset
    set dr [auto_execok driver]
    if {$dr!=""} {
	set driver_path_v1 [lindex $dr 0]
	set plumed_version 1
    }

    set dr [auto_execok plumed]
    if {$dr!=""} {
	set driver_path_v2 [lindex $dr 0]
	set plumed_version 2
    }

    if {$plumed_version==0} {
	# Oddly, give time to extensions menu to close
	after 100 { 
	    tk_messageBox -icon warning -title "PLUMED not found" -parent .plumed -message $Plumed::plumed_not_found_message
	}
	set plumed_version $plumed_default_version
    } 
    plumed_version_changed

}



# May be invoked by GUI or upon searching paths. We assume that plumed
# version is either 1 or 2 (not 0)
proc ::Plumed::plumed_version_changed {} {
    variable plumed_version
    variable driver_path_v1
    variable driver_path_v2
    variable driver_path
    variable w

    switch $plumed_version {
	1  {
	    set driver_path $driver_path_v1
	    set_state_recursive normal $w.options.line1.pbc
	}
	2  {
	    set driver_path $driver_path_v2
    	    set_state_recursive normal $w.options.line1.pbc
	}
	vmdcv {
	    set driver_path "(Not needed)"
	    set_state_recursive disabled $w.options.line1.pbc
	}
    } 

    instructions_update
    structuremenu_update
    templates_populate_menu
}

# ==================================================
# Templates

proc ::Plumed::templates_populate_menu {} {
    variable w
    variable plumed_version
    variable templates_list_v1
    variable templates_list_v2
    variable templates_list_vmdcv

    lassign [getModifiers] mod modifier

    switch $plumed_version {
	1      {set templates $templates_list_v1}
	2      {set templates $templates_list_v2}
	vmdcv  {set templates $templates_list_vmdcv}
    } 

    $w.menubar.insert delete 0 last
    foreach { disp insr } $templates {
	if {$disp == "-" } {
	    $w.menubar.insert add separator
	} else {
	    $w.menubar.insert add command -label $disp \
		-command [list Plumed::templates_insert_line $insr]
	}
    }
    $w.menubar.insert add separator
    $w.menubar.insert add command -label "...see manual for the full list" -state disabled

    switch $plumed_version {
	1 {
	    bind $w <$modifier-g> "$::Plumed::w.menubar.insert invoke 1" 
	    $w.menubar.insert entryconfigure 1 -accelerator $mod+G
	}
	2 {
	    bind $w <$modifier-g> "$::Plumed::w.menubar.insert invoke 1" 
	    $w.menubar.insert entryconfigure 1 -accelerator $mod+G
	    bind $w <$modifier-m> "$::Plumed::w.menubar.insert invoke 2" 
	    $w.menubar.insert entryconfigure 2 -accelerator $mod+M
	}
	vmdcv {
	    bind $w <$modifier-g> "$::Plumed::w.menubar.insert invoke 1" 
	    $w.menubar.insert entryconfigure 1 -accelerator $mod+G
	}
    }
}


# Insert line at cursor
proc ::Plumed::templates_insert_line {line} {
    variable w
    $w.txt.text edit separator
    $w.txt.text insert {insert linestart} "$line\n"
}

				      

# Plumed::templates_list_v1 is in a separate file in the same
# package

# Plumed::templates_list_v2 is in a separate, autogenerated
# file in the same package


# ==================================================
# Context-sensitive pop up

# Invoked upon right-click
proc ::Plumed::popup_menu {x y X Y} {
    variable plumed_version
    variable w
    variable template_keyword_hash
    variable template_full_hash

    # No menu for plumed 1
    if {$plumed_version==1} { return }

    # Get word at mouse
    set t $w.txt.text
    set word [$w.txt.text get "@$x,$y wordstart" "@$x,$y wordend"]
    set word [string trim $word]

    # Build popup
    $t.popup delete 0 last
    if {$word != ""} {
	set uword [string toupper $word]
	$t.popup add command -label "Lookup $uword in documentation..." \
	    -command "[namespace current]::popup_local_or_remote_help $uword"
	$t.popup add separator

	# Short template
	if { [info exists template_keyword_hash($uword)] } {
	    $t.popup add command -label {Insert template line below cursor} \
		-command "[namespace current]::popup_insert_line \{$template_keyword_hash($uword)\}"
	} else {
	    $t.popup add command -label "No template for keyword $uword" -state disabled
	}

	# Long template
	if { [info exists template_full_hash($uword)] } {
	    $t.popup add command -label {Insert full template line below cursor} \
		-command "[namespace current]::popup_insert_line \{$template_full_hash($uword)\}"

	    # Build lists of mandatory and optional keywords
	    set okw_l {}
	    set kw_l  {}
	    foreach kw $template_full_hash($uword) {
		if { $kw == $uword } { continue }
		if [ regexp {\[(.+)\]} $kw junk kw ] {
		    lappend okw_l $kw; # in brackets? push in optional
		} else { 
		    lappend kw_l $kw; # push in regular
		}
	    }
	    
	    if {[llength $kw_l] > 0} {
		$t.popup add separator
		$t.popup add command -label "Parameters:" -state disabled
		foreach kw $kw_l {
		    $t.popup add command -label "   $kw" \
			-command "[namespace current]::popup_insert_keyword $kw"
		}
	    }

	    if {[llength $okw_l] > 0} {
		$t.popup add separator
		$t.popup add command -label "Optional modifiers:" -state disabled
		foreach kw $okw_l {
		    $t.popup add command -label "   $kw" \
			-command "[namespace current]::popup_insert_keyword $kw"
		}
	    }
	}
    } else {
	$t.popup add command -label "No keyword here" -state disabled
    }
    tk_popup $w.txt.text.popup $X $Y
}

# Insert line below cursor
proc ::Plumed::popup_insert_line {line} {
    variable w
    $w.txt.text edit separator
    $w.txt.text insert {insert lineend} "\n# $line"
}

# Insert word at cursor
proc ::Plumed::popup_insert_keyword {kw} {
    variable w
    $w.txt.text edit separator
    $w.txt.text insert insert " $kw"
}


# Return a sensible base url for documentation (cached)
proc ::Plumed::popup_help_url {} {
    variable popup_help_url_cached; # static
    variable driver_path
    variable plumed2_online_docbase

    if {![info exists popup_help_url_cached]} {
	# 1. try local
	set htmlfile [exec $driver_path --standalone-executable info --user-doc]
	if [file readable $htmlfile] {
	    set popup_help_url_cached [file dirname $htmlfile]
	    puts "Info: using local help pages."
	} else {
	    # 2. try version-specific remote
	    set url $plumed2_online_docbase
	    set url [string map [list "VERSION" [exec $driver_path --standalone-executable info --version]] $url]
	    puts "URL IS $url"
	    if {[lsearch -exact {200 301} [get_url_ncode "$url/index.html"]] >= 0} {
		puts "Info: local help pages not available, using remote pages at $url"
		set popup_help_url_cached $url
	    } else {
		puts "Warning: local and remote help pages not available, using fallback"
		set popup_help_url_cached "http://www.plumed.org"
	    }
	}
    }
    return $popup_help_url_cached
}

# Convert word to doxygen-generated filename
proc ::Plumed::popup_keyword_underscorify {p} {
    set pu [string tolower $p];	# lower
    set pu [join [split $pu ""] _]; # intermix underscore
    set pu [regsub {___} $pu __];   # ___ -> __
    return "_$pu";		    # prepend underscore
}

# Do what it takes to open Doxygen-generated help on keyword
proc ::Plumed::popup_local_or_remote_help {kw} {
    if {$kw == ""} { return }
    set docroot [popup_help_url]
    set kwlu [popup_keyword_underscorify $kw]
    set htmlpage "$docroot/$kwlu.html"
    vmd_open_url $htmlpage
}



# ==================================================
# Version-independent stuff

# If outfile is not given, plot
proc ::Plumed::do_compute {{outfile ""}} {
    variable plumed_version 
    variable driver_path

    if {[molinfo top]==-1 || [molinfo top get numframes] < 2} {
	tk_messageBox -title "Error" -icon error -parent .plumed -message \
	    "A top molecule and at least two frames are required to plot."
	return 
    }

    # Internal VMD is handled differently enough.
    if {$plumed_version == "vmdcv"} {
	do_compute_vmdcv
	return
    }

    if {![file executable $driver_path]} { 
	tk_messageBox -title "Error" -icon error -parent .plumed -message \
	    "The plumed executable is required. See manual for installation instructions."
	return }

    # Prepare temp. dir and files
    set tmpd [file join [tmpdir] vmdplumed.[pid]]
    file mkdir $tmpd

    set meta [file join $tmpd META_INP]
    set pdb [file join $tmpd temp.pdb] 
    set dcd [file join $tmpd temp.dcd]
    set colvar [file join $tmpd COLVAR]

    writePlumed [atomselect top all] $pdb
    animate write dcd $dcd waitfor all
    file delete $colvar

    # Prepare command
    switch $plumed_version {
	1 {
	    Plumed::write_meta_inp_v1 $meta
	    set pbc [ get_pbc_v1 ]
	    set cmd [list $driver_path -dcd $dcd -pdb $pdb -plumed $meta {*}$pbc]
	}
	2 {
	    write_meta_inp_v2 $meta $colvar
	    set pbc [get_pbc_v2]
	    set cmd [list $driver_path --standalone-executable driver {*}$pbc --mf_dcd $dcd --pdb $pdb --plumed $meta  ]
	}
    }

    # Run. V1 requires pushd
    puts "Executing: $cmd"

    if {$plumed_version==1} { cd_push $tmpd }
    if { [ catch { exec {*}$cmd } driver_stdout ] ||
	 ! [file readable $colvar]  } {
	set failure 1
    } else {
	set failure 0
    }
    if {$plumed_version==1} { cd_push - }

    # Results
    puts $driver_stdout
    puts "-----------"
    puts "Temporary files are in directory $tmpd"

    # Parse if v2
    if { $failure } {
	puts "Something went wrong. Check above messages."
	tk_messageBox -icon error -title "Error" -parent .plumed -message \
	    "PLUMED returned an error while executing the script. Please find error messages in the console. "
	if {$plumed_version==2 && \
		[regexp -line {^PLUMED: ERROR .+ with label (.+?) : (.+)} \
				       $driver_stdout junk label etext] } {
	    dputs "Trying to highlight label $label -- $etext "
	    highlight_error_label $label $etext
	}
    } else {
	if {$outfile != ""} {
	    file copy -force $colvar $outfile
	} else {
	    Plumed::do_plot $colvar $driver_stdout
	}
    }

}


# Assume a well-formed COLVAR file: one header line
# FIXME - assuming time is the first column
proc ::Plumed::do_plot { { out COLVAR } { txt ""  } } {
    variable w
    variable plot_points

    # slurp $out
    set fd [open $out r]
    set data {}
    set header {}
    set nlines 0
    while {[gets $fd line]>=0} {
	if [regexp {^#!} $line] {
	    set op [lindex $line 1]
	    if { $op == "FIELDS" } {
		# remove hash-FIELDS-time . Now header contains CV names
		set header [lreplace $line 0 2]
	    } else {
		continue;		# skip other headers (eg periodicity)
	    }
	} else {
	    lappend data $line
	    incr nlines
	}
    }
    close $fd

    if { [llength $header] == 0 } {
	puts "No FIELDS columns found. It usually means that you have no COLVAR defined (or PLUMED < 1.3)."
	return
    } elseif { $nlines == 0 } {
	puts "No output in COLVAR. Please check above messages."
	return
    } elseif { $nlines == 1 } {
	puts "Single frame output. Omitting plot"
	return
    }

    set data [transpose $data]

    # pop the time column
    set ltime [lindex $data 0]
    set data [lreplace $data 0 0]
    set cv_n [llength $data]

    if { $plot_points } {
	set pt circle
    } else {
	set pt none
    }

    set cvplot [multiplot -title "Collective variables" -xlabel "Time (frames)" \
		    -nostats ]

    for { set i 0 } { $i < $cv_n } {incr i } {
	set coln [ expr ($i-1)%16 ]
	set color [index2rgb $coln]
	$cvplot add $ltime [lindex $data $i] -legend [lindex $header $i] \
	    -lines -marker $pt -radius 2 -fillcolor $color \
	    -color $color -nostats
    }

    $cvplot replot

}

# V1 output file can't be changed so we need to CD
# http://wiki.tcl.tk/1034
proc ::Plumed::cd_push {{dir {}}} {
    variable ::Plumed::cd_lastdir
    set pwd [pwd]
    if {$dir eq "-"} {
        if {![info exists cd_lastdir]} { return }
        set dir $cd_lastdir
    } elseif {[llength [info level 0]] == 1} {
        # no $dir specified - go home
        set code [catch {cd } res]
    }
    if {![info exists code]} {
        set code [catch {cd $dir} res]
    }
    if {!$code} { set cd_lastdir $pwd }
    return -code $code $res
}


# ==================================================                                                 
# V1-specific stuff

proc ::Plumed::write_meta_inp_v1 { meta } { 
    set text [getText]
    set fd [open $meta w]
    puts $fd [ Plumed::replace_serials $text ]
    puts $fd "PRINT W_STRIDE 1  ! line added by vmdplumed for visualization"
    puts $fd "ENDMETA" 
    close $fd
}

proc ::Plumed::get_pbc_v1 { } {
    variable pbc_type
    variable pbc_boxx
    variable pbc_boxy
    variable pbc_boxz
    set pbc [ switch $pbc_type {
	1 {format "-nopbc"}
	2 {format "" }
	3 {format "-cell $pbc_boxx $pbc_boxy $pbc_boxz" } } ]
    return $pbc
}



# ==================================================                                                 
# V2-specific stuff

proc ::Plumed::write_meta_inp_v2 { meta colvar } { 
    set text [getText]
    set fd [open $meta w]
    puts $fd [ Plumed::replace_serials $text ]
    puts $fd "# line added by vmdplumed for visualization"
    puts $fd "PRINT ARG=* FILE=$colvar"
    close $fd
}

# Box is in nm
proc ::Plumed::get_pbc_v2 { } {
    variable pbc_type
    variable pbc_boxx
    variable pbc_boxy
    variable pbc_boxz
    set largebox 100000
    set pbc [ switch $pbc_type {
	1 {format "--box $largebox,$largebox,$largebox"}
	2 {format "" }
	3 {format "--box %f,%f,%f" \
	       [expr $pbc_boxx/10.0] \
	       [expr $pbc_boxy/10.0] \
	       [expr $pbc_boxz/10.0] } } ]
    return $pbc
}



# ========================================
# Force-display stuff

# Run driver with --dump-forces. Extensive refactoring needed.
proc ::Plumed::show_forces_compute { } {
    variable driver_path

    if {[molinfo top]==-1 || [molinfo top get numframes] < 1} {
	tk_messageBox -title "Error" -icon error -parent .plumed -message \
	    "A top molecule and at least one frame is required to plot."
	return 
    }

    if {![file executable $driver_path]} { 
	tk_messageBox -title "Error" -icon error -parent .plumed -message \
	    "The plumed executable is required. See manual for installation instructions."
	return }

    # Prepare temp. dir and files
    set tmpd [file join [tmpdir] vmdplumed.[pid]]
    file mkdir $tmpd

    set meta [file join $tmpd META_INP]
    set pdb [file join $tmpd temp.pdb] 
    set dcd [file join $tmpd temp.dcd]
    set colvar [file join $tmpd COLVAR]
    set forces [file join $tmpd FORCES]

    writePlumed [atomselect top all] $pdb
    animate write dcd $dcd waitfor all
    file delete $colvar

    write_meta_inp_v2 $meta $colvar
    set pbc [get_pbc_v2]
    set cmd [list $driver_path --standalone-executable driver {*}$pbc --mf_dcd $dcd --pdb $pdb --plumed $meta --dump-forces $forces  ]

    puts "Executing: $cmd"

    if { [ catch { exec {*}$cmd } driver_stdout ] ||
	 ! [file readable $colvar]  } {
	set failure 1
    } else {
	set failure 0
    }


    # Results
    puts $driver_stdout
    puts "-----------"
    puts "Temporary files are in directory $tmpd"

    # Parse if v2
    if { $failure } {
	puts "Something went wrong. Check above messages."
	return {}
    }

    set force_list [parse_forces $forces]
    return $force_list
}


# Parse a forces file like
# NATOMS
# FBX FBY FBZ
# X F1X F1Y F1Z
# X ...
# repeated for a number of frames. Return a list of lists 
proc ::Plumed::parse_forces {fname} {
    set ff [open $fname r]
    set force_list {}
    while {[gets $ff nat]>=0} {
	gets $ff boxforces
	set this_frame_forces {}
	for {set a 0} {$a < $nat} {incr a} {
	    # Delete atom name
	    gets $ff line
	    set fxyz [lreplace $line 0 0]
	    lappend this_frame_forces $fxyz
	}
	lappend force_list $this_frame_forces
    }
    close $ff
    return $force_list
}


proc ::Plumed::show_forces_gui {} {
    set tl .plumed_show_forces
    if { [winfo exists $tl] } {
	wm deiconify $tl
	raise $tl
	return
    }

    variable show_forces_scale 1.00
    variable show_forces_data

    toplevel $tl
    wm title $tl "Display gradients and forces"

    pack [ ttk::frame $tl.pad -padding 8 ] -side top -fill x
    
    set n $tl.pad
    pack [ ttk::label $n.head1 -text "Display the force vector that would be applied to each atom." \
	       -justify center -anchor center -pad 3 ] -side top -fill x 

    pack [ ttk::label $n.explain -text "To visualize the effect of a bias on a CV\nyou may want to apply a constant unitary force to it, e.g.:\n\nRESTRAINT ARG=mycv AT=0 SLOPE=-1" \
	       -justify center -anchor center -pad 3 ] -side top -fill x 

    # http://wiki.tcl.tk/1433
    pack [ ttk::frame $n.scale ] -side top -fill x -expand 1

    pack [ ttk::label $n.scale.lab -text "Arrow scale: "] -side left
    pack [ ttk::scale  $n.scale.scale -from -20 -to 20 -value 0 -length 200 \
	       -orient h -command ::Plumed::show_forces_scale_changed]  -side left -fill x -expand 1
    pack [ ttk::label $n.scale.value -text 1.0 -width 8 -anchor e] -side left
    pack [ ttk::label $n.scale.unit -text "Å per kJ/mol/nm" ] -side left

    wm protocol $tl WM_DELETE_WINDOW {
	::Plumed::show_forces_stop
	destroy .plumed_show_forces
    }

    set show_forces_data [show_forces_compute]

    if {$show_forces_data eq ""} {
	tk_messageBox -icon error \
	    -title "Error" \
	    -message "PLUMED returned an error while executing the script. Please find error messages in the console. " \
	    -parent .plumed_show_forces 
	show_forces_stop
	destroy .plumed_show_forces
    } elseif [show_forces_is_null $show_forces_data] {
	tk_messageBox -icon info \
	    -message "All acting forces are null. Consider adding a RESTRAINT or another biasing statement." \
	    -title "Null forces" \
	    -parent .plumed_show_forces
	show_forces_stop
	destroy .plumed_show_forces
    } else {
	show_forces_start
	show_forces_draw_frame
    }
}

proc ::Plumed::show_forces_is_null {data} {
    foreach fd $data {
	foreach fa $fd {
	    if [catch {veclength $fa} fal] {
		set fal 1
	    }
	    if {$fal>0} {
		return 0
	    }
	}
    }
    return 1
}

proc ::Plumed::show_forces_scale_changed {vraw} {
    variable show_forces_scale
    set v [expr 10**($vraw/10)]
    set show_forces_scale $v
    set vr [format "%.2f" $v]
    .plumed_show_forces.pad.scale.value configure -text $vr
    show_forces_draw_frame
}

proc ::Plumed::show_forces_start {} {
    # http://www.ks.uiuc.edu/Training/Tutorials/vmd-imgmv/imgmv/tutorial-html/node3.html#SECTION00032000000000000000
    global vmd_frame
    trace variable vmd_frame([molinfo top]) w ::Plumed::show_forces_draw_frame
}

proc ::Plumed::show_forces_stop {} {
    global vmd_frame
    graphics top delete all
    trace vdelete vmd_frame([molinfo top]) w ::Plumed::show_forces_draw_frame
}

proc ::Plumed::show_forces_draw_frame {args} {
    variable show_forces_data
    variable show_forces_scale 
    
    # global vmd_frame
    set fno [molinfo top get frame]

    set fd [lindex $show_forces_data $fno]

    set as [atomselect top all]
    $as frame $fno
    set xyz_all [$as get {x y z}]
    $as delete

    #  Iterate over atoms
    set err 0
    set sum 0
    graphics top delete all
    foreach d $fd x $xyz_all {
	if {[catch {vecscale $show_forces_scale $d} ds]} {
	    set err 1
	} else {
	    draw_arrow $x $ds
	    set sum [expr {$sum+[veclength $d]}]
	}
    }

    if {$err==1} {
	tk_messageBox -icon warning \
	    -message "Some gradient components are NAN or infinite.\nThey will not be shown." \
	    -title "Numerical problem" \
	    -parent .plumed_show_forces
    }
    
}

# Draw an arrow at x in direction d
proc ::Plumed::draw_arrow {x d {r .1} {tip .2}} {
    set min_len 0.1
    set xf [vecadd $x $d]
    if {[veclength $d] > $min_len} {
	set xtip [vecadd $xf [vecscale $tip [vecnorm $d]]]
	graphics top cylinder $x $xf radius $r filled yes
	graphics top cone $xf $xtip radius [expr 2*$r]
    }
}


				  




# ==================================================                                                 
# VMDCV-specific stuff

proc ::Plumed::do_compute_vmdcv {} {
	catch {cv delete} e
	cv molid top
	set script [replace_serials [getText] ]
        if [ catch { cv config $script } e ] {
	    tk_messageBox -icon error -title "Error" -parent .plumed \
		-message "Colvar module returned the following problem...\n\n$e." \
		-detail "Further messages are found in VMD's text console."
	    cv reset
	    return
	} 

	set fname [file join [tmpdir] "vmd_plumed.[pid].dat"]
	set fd [open $fname w]
	puts "Generating temporary data file $fname"

	# Make a pseudo-plumed-v2 header
	set cvnames [vmdcv_expand_cvlist ]
	puts $fd "#! FIELDS time $cvnames"
	
	# Print all the CV values
	for {set f 0} {$f<[molinfo top get numframes]} { incr f } {
		cv frame $f
		cv update
		puts -nonewline $fd [vmdcv_vec2list [cv printframe]]
	}
	close $fd

	do_plot $fname
}

# Remove parenthesis and commas
proc ::Plumed::vmdcv_vec2list v {
	return [string map { ( {} , {} ) {} } $v ]
}

# Return [cv list] but with vectors expanded 
proc ::Plumed::vmdcv_expand_cvlist {} {
	set out {}
	cv frame 0
	cv update
 	foreach c [cv list] {
		set v [cv colvar $c value]
		set vv [vmdcv_vec2list $v]; 
		set nc [llength $vv]
		if {$nc==1} {
			lappend out $c
		} else {
			for {set i 1} {$i<=$nc} {incr i} {
				lappend out "${c}_$i"
			}
		}
	}				
	return $out
}
		

