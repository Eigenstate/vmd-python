#
# GUI for chirality plugin
#

package provide chirality_gui 1.2

namespace eval ::chirality::GUI:: {

  package require chirality

  proc resetGUI { } {

    # pkg_mkIndex doesn't like the following command
    # variable currentSelText $::chirality::defaultSelText
    variable currentSelText {all}

    variable chiralityList {}
    set chiralityList [list]
    variable currentMol {none}

  }
  resetGUI

}

proc chirality_gui { } { return [eval ::chirality::GUI::chirality_gui] }

proc ::chirality::GUI::chirality_gui { } {

  variable w
  # Background color selected listbox elements (from AutoPSF)
  variable selectcolor lightsteelblue 

  variable nullMolString "none"
  variable currentMol
  variable molMenuButtonText
  trace add variable [namespace current]::currentMol write [namespace code {
    variable currentMol
    variable molMenuButtonText
    if { ! [catch { molinfo $currentMol get name } name ] } {
      set molMenuButtonText "$currentMol: $name"
    } else {
      set molMenuButtonText $currentMol
    }
  # } ]
  set currentMol $nullMolString
  variable usableMolLoaded 0

  # If already initialized, just turn on
  if { [winfo exists .chirality] } {
    wm deiconify $w
    return
  }
  set w [toplevel ".chirality"]
  wm title $w "chirality"
  wm resizable $w 0 0

  # Add a menu bar
  frame $w.menubar -relief raised -bd 2
  pack $w.menubar -padx 1 -fill x

  menubutton $w.menubar.help -text "Help" -underline 0 -menu $w.menubar.help.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.help config -width 5

  # Help menu
  menu $w.menubar.help.menu -tearoff no
  $w.menubar.help.menu add command -label "About" \
    -command {tk_messageBox -type ok -title "About chirality" \
    -message "The chirality plugin automates detection and correction of chirality errors in protein and nucleic acid structures."}
  $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/chirality"

  pack $w.menubar.help -side right

  set row 0

  ############## frame for checking structure #################
  grid [labelframe $w.check -bd 2 -relief ridge -text "Check protein or nucleic acid structure for chirality errors" -padx 1m -pady 1m] -row $row -columnspan 2 -sticky nsew
  incr row
  
  set f [frame $w.check.frame]
  set irow 0
  grid [label $f.mollabel -text "Molecule: "] \
    -row $irow -column 0 -sticky e
  grid [menubutton $f.mol -textvar [namespace current]::molMenuButtonText \
    -menu $f.mol.menu -relief raised] \
    -row $irow -column 1 -columnspan 1 -sticky ew
  menu $f.mol.menu -tearoff no
  incr irow
  
  fill_mol_menu $f.mol.menu
  trace add variable ::vmd_initialize_structure write [namespace code "
    fill_mol_menu $f.mol.menu
  # " ]

  grid [label $f.sellabel -text "Selection: "] \
    -row $irow -column 0 -sticky e
  grid [entry $f.sel -width 50 \
    -textvariable [namespace current]::currentSelText] \
    -row $irow -column 1 -columnspan 1 -sticky ew
  incr irow

  grid [button $f.checkbutton -text "Check structure" \
    -command ::chirality::GUI::chiralityCheck ] \
    -row $irow -column 0 -columnspan 2 -sticky ew
  incr irow

  pack $f -side top -padx 0 -pady 0 -expand 1 -fill x
  pack $w.check -side top -padx 5 -pady 5 -expand 1 -fill x

  ############## frame for listing chirality errors #################
  grid [labelframe $w.list -bd 2 -relief ridge -text "Identified chirality errors" -padx 1m -pady 1m] -row $row -column 0 -columnspan 3 -sticky nsew
  set f [frame $w.list.frame]
  variable listformattext "  residue  |    chiral center    | chain segname | atom | moved"
  label $f.label -font {tkFixed 9} -textvar [namespace current]::listformattext -relief flat -justify left
  scrollbar $f.scroll -command "$f.list yview"
  listbox $f.list -activestyle dotbox -yscroll "$f.scroll set" \
    -font {tkFixed 9} -width 63 -height 8 -setgrid 1 -selectmode extended \
    -selectbackground $selectcolor \
    -listvariable [namespace current]::chiralityList
  pack $f.label -side top -anchor w
  pack $f.list $f.scroll -side left -fill y -expand 1
  pack $f -side top -padx 0 -pady 0 -expand 1 -fill x

  set f [frame $w.list.frame2]
  button $f.showbutton -text "Show selected chiral center" -command [namespace current]::chiralityShow
  pack $f.showbutton -fill x -side bottom
  pack $f -side top -padx 2 -pady 0 -expand 1 -fill x
  pack $w.list -side top -padx 5 -pady 5 -expand 1 -fill x

  ##### frame for selecting/moving atoms in cis peptide bonds #####
  grid [labelframe $w.atoms -bd 2 -relief ridge -text "Move hydrogen atoms in chiral centers" -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  set f [frame $w.atoms.frame]
  set irow 0
  grid [label $f.label -text "Tag atom to be moved for selected chiral centers:"] -row $irow -column 0 -columnspan 2 -sticky w
  #incr irow
  grid [button $f.setbutton1 -text "hydrogen" -command "[namespace current]::chiralitySet H"] -row $irow -column 0 -sticky ew
  grid [button $f.setbutton2 -text "none" -command "[namespace current]::chiralitySet X"] -row $irow -column 1 -sticky ew
  incr irow
  grid [button $f.movebutton -text "Move tagged atoms for selected chiral centers" -command [namespace current]::chiralityMove] -row $irow -column 0 -columnspan 2 -sticky ew
  incr irow
  pack $f.label 
  pack $f.movebutton -expand 1 -fill x -side bottom
  pack $f.setbutton1 $f.setbutton2 -side left -expand 1 -fill x
  pack $f -side top -padx 0 -pady 0 -expand 1 -fill x
  pack $w.atoms -side top -padx 5 -pady 5 -expand 1 -fill x

  set f [frame $w.fix]
  button $f.minbutton -text "Minimize/equilibrate selected chiral centers" -command [namespace current]::chiralityMinimize
  button $f.minbuttonMDFF -text "Minimize/equilibrate selected cis peptide bonds using MDFF" -command [namespace current]::chiralityMinimizeMDFF
  button $f.resetbutton -text "Reset chirality plugin" -command [namespace current]::chiralityReset
  button $f.luckybutton -state disabled -text "I'm feeling lucky" -command [namespace current]::chiralityLucky
  pack $f.minbutton -side top -expand 1 -fill x
  pack $f.minbuttonMDFF -side top -expand 1 -fill x
  pack $f.resetbutton -side left -expand 1 -fill x
  pack $f.luckybutton -side right -expand 1 -fill x
  pack $f -side top -padx 5 -pady 2 -expand 1 -fill x

}

# Adapted from autopsf, which is adapted from pmepot...
proc ::chirality::GUI::fill_mol_menu {name} {
  variable currentMol
  variable nullMolString
  $name delete 0 end

  set molList {}
  foreach mm [array names ::vmd_initialize_structure] {
    if { $::vmd_initialize_structure($mm) != 0} {
      lappend molList $mm
      $name add radiobutton -variable [namespace current]::currentMol \
      -value $mm -label "$mm [molinfo $mm get name]" \
    }
  }

  #set if any non-Graphics molecule is loaded
  if {[lsearch -exact $molList $currentMol] == -1} {
    if {[lsearch -exact $molList [molinfo top]] != -1} {
      set currentMol [molinfo top]
    } else { set currentMol $nullMolString }
  }
}


proc ::chirality::GUI::chiralityCheck { } {

  variable currentMol
  variable currentSelText

  if { $currentMol == "none" } {
    tk_messageBox -type ok -message "Please load the molecule to be processed."
    return
  }
  if {[chirality check -mol $currentMol -seltext $currentSelText -gui 1] != -1} {
    updateList
  }

}

proc ::chirality::GUI::updateList { } {

  variable chiralityList
  variable currentMol
  variable w

  set selectedList [$w.list.frame.list curselection]
  set rawReturnList [chirality list all -mol $currentMol -gui 1]

  set chiralityList {}
  foreach {chiral names action moved} $rawReturnList {
    set residue [lindex $chiral 0]
    set resid   [lindex $chiral 1]
    set resname [lindex $chiral 2]
    set chain   [lindex $chiral 3]
    set segname [lindex $chiral 4]
    if { $action == {X} } {
      set action {none}
    }
    if { $moved == -1 } {
      set moved {no}
    } else {
      set moved {yes}
    }
    lappend chiralityList [format "%-3s %-8s %-23s %-1s    %-4s     %-4s    %-3s" $resname $resid $names $chain $segname $action $moved]
  }
  
  # Keep the same chiral centers selected as before
  foreach chiral $selectedList {
    $w.list.frame.list selection set $chiral
  }

  return

}

proc ::chirality::GUI::chiralityShow { } {
  variable w
  variable currentMol

  set thisChiral [$w.list.frame.list curselection]
  if {[llength $thisChiral] > 1} {
    tk_messageBox -type ok -message "Please select a chiral center."
  } elseif { $thisChiral != {} } {
    chirality show $thisChiral -mol $currentMol
  }
  return
}

proc ::chirality::GUI::chiralitySet { action } {
  variable w
  variable currentMol

  set selectedList [$w.list.frame.list curselection]
  if { $selectedList != {} } {
    foreach chiral $selectedList {
      chirality set $action -cen $chiral -mol $currentMol -gui 1
    }
  }
  updateList

  return
}

proc ::chirality::GUI::chiralityMove { } {
  variable w
  variable currentMol

  set selectedList [$w.list.frame.list curselection]
  if { $selectedList != {} } {
    foreach chiral $selectedList {
      chirality move $chiral -mol $currentMol -gui 1
    }
  }
  updateList

  return
}

proc ::chirality::GUI::chiralityMinimize { } {
  variable w
  variable currentMol

  set selectedList [$w.list.frame.list curselection]
  chirality minimize $selectedList -mol $currentMol -gui 1

  return
}

proc ::chirality::GUI::chiralityMinimizeMDFF { } {
  variable w
  variable currentMol

  set selectedList [$w.list.frame.list curselection]
  chirality minimize $selectedList -mol $currentMol -gui 1 -mdff 1

  return
}

proc ::chirality::GUI::chiralityReset { } {
  resetGUI
  chirality reset
  return
}

proc ::chirality::GUI::chiralityLucky { } {
  tk_messageBox -type ok -message "Are you really?"
  return
}

# This gets called by VMD the first time the menu is opened.
proc chirality_tk_cb {} {
  chirality_gui   ;# start the GUI
  return $::chirality::GUI::w
}

