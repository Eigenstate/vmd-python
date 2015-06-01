#
# GUI for cispeptide plugin
#
# $Id: cispeptide_gui.tcl,v 1.6 2015/04/01 21:11:54 ryanmcgreevy Exp $
#

package provide cispeptide_gui 1.2

namespace eval ::cispeptide::GUI:: {

  package require cispeptide 

  proc resetGUI { } {

    # pkg_mkIndex doesn't like the following command
    # variable currentSelText $::cispeptide::defaultSelText
    variable currentSelText {same fragment as protein}

    # variable splitMethod {segname}
    variable cispeptideList {}
    set cispeptideList [list]
    variable currentMol {none}

  }
  resetGUI

}

proc cispeptide_gui { } { return [eval ::cispeptide::GUI::cispeptide_gui] }

proc ::cispeptide::GUI::cispeptide_gui { } {

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
  if { [winfo exists .cispeptide] } {
    wm deiconify $w
    return
  }
  set w [toplevel ".cispeptide"]
  wm title $w "cispeptide"
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
    -command {tk_messageBox -type ok -title "About cispeptide" \
    -message "The cispeptide plugin automates detection and correction of cis peptide bonds in protein structures."}
  $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/cispeptide"

  pack $w.menubar.help -side right

  set row 0
  
  ############## frame for checking structure #################
  grid [labelframe $w.check -bd 2 -relief ridge -text "Check protein structure for cis peptide bonds" -padx 1m -pady 1m] -row $row -columnspan 3 -sticky nsew
  incr row
  
  set f [frame $w.check.frame]
  set irow 0
  grid [label $f.mollabel -text "Molecule: "] \
    -row $irow -column 0 -sticky e
  grid [menubutton $f.mol -textvar [namespace current]::molMenuButtonText \
    -menu $f.mol.menu -relief raised] \
    -row $irow -column 1 -columnspan 2 -sticky ew
  menu $f.mol.menu -tearoff no
  incr irow
  
  fill_mol_menu $f.mol.menu
  trace add variable ::vmd_initialize_structure write [namespace code "
    fill_mol_menu $f.mol.menu
  # " ]

  grid [label $f.sellabel -text "Selection: "] \
    -row $irow -column 0 -sticky e
  grid [entry $f.sel -width 45 \
    -textvariable [namespace current]::currentSelText] \
    -row $irow -column 1 -columnspan 2 -sticky ew
  incr irow

#  grid [label $f.naterm -text "Split chains by: "] \
#    -row $irow -column 0 -sticky e
#  grid [radiobutton $f.splitchain -value "chain" -text "chain" \
#    -variable [namespace current]::splitMethod] \
#    -row $irow -column 1 -sticky w
#  grid [radiobutton $f.splitsegname -value "segname" -text "segname" \
#    -variable [namespace current]::splitMethod] \
#    -row $irow -column 2 -sticky w
#  incr irow 

  grid [button $f.checkbutton -text "Check structure" \
    -command ::cispeptide::GUI::cispeptideCheck ] \
    -row $irow -column 0 -columnspan 3 -sticky ew
  incr irow

  pack $f -side top -padx 0 -pady 0 -expand 1 -fill x
  pack $w.check -side top -padx 5 -pady 5 -expand 1 -fill x

  ############## frame for listing cis peptide bonds #################
  grid [labelframe $w.list -bd 2 -relief ridge -text "Identified cis peptide bonds" -padx 1m -pady 1m] -row $row -column 0 -columnspan 3 -sticky nsew
  set f [frame $w.list.frame]
  variable listformattext " 1st residue | 2nd residue | chain segname | atom | moved"
  label $f.label -font {tkFixed 9} -textvar [namespace current]::listformattext -relief flat -justify left
  scrollbar $f.scroll -command "$f.list yview"
  listbox $f.list -activestyle dotbox -yscroll "$f.scroll set" \
    -font {tkFixed 9} -width 58 -height 8 -setgrid 1 -selectmode extended \
    -selectbackground $selectcolor \
    -listvariable [namespace current]::cispeptideList
  pack $f.label -side top -anchor w
  pack $f.list $f.scroll -side left -fill y -expand 1
  pack $f -side top -padx 0 -pady 0 -expand 1 -fill x

  set f [frame $w.list.frame2]
  button $f.showbutton -text "Show selected cis peptide bond" -command [namespace current]::cispeptideShow
  pack $f.showbutton -fill x -side bottom
  pack $f -side top -padx 2 -pady 0 -expand 1 -fill x
  pack $w.list -side top -padx 5 -pady 5 -expand 1 -fill x

  ##### frame for selecting/moving atoms in cis peptide bonds #####
  grid [labelframe $w.atoms -bd 2 -relief ridge -text "Select and move atoms in cis peptide bonds" -padx 1m -pady 1m] -row $row -column 0 -columnspan 3 -sticky nsew
  set f [frame $w.atoms.frame]
  set irow 0
  grid [label $f.label -text "Tag atom to be moved for selected cis peptide bonds:"] -row $irow -column 0 -columnspan 3 -sticky w
  #incr irow
  grid [button $f.setbutton1 -text "hydrogen" -command "[namespace current]::cispeptideSet H"] -row $irow -column 0 -sticky ew
  grid [button $f.setbutton2 -text "oxygen" -command "[namespace current]::cispeptideSet O"] -row $irow -column 1 -sticky ew
  grid [button $f.setbutton3 -text "none" -command "[namespace current]::cispeptideSet X"] -row $irow -column 2 -sticky ew
  incr irow
  grid [button $f.movebutton -text "Move tagged atoms for selected cis peptide bonds" -command [namespace current]::cispeptideMove] -row $irow -column 0 -columnspan 3 -sticky ew
  incr irow
  pack $f.label 
  pack $f.movebutton -expand 1 -fill x -side bottom
  pack $f.setbutton1 $f.setbutton2 $f.setbutton3 -side left -expand 1 -fill x
  pack $f -side top -padx 0 -pady 0 -expand 1 -fill x
  pack $w.atoms -side top -padx 5 -pady 5 -expand 1 -fill x

  set f [frame $w.fix]
  button $f.minbutton -text "Minimize/equilibrate selected cis peptide bonds" -command [namespace current]::cispeptideMinimize
  button $f.minbuttonMDFF -text "Minimize/equilibrate selected cis peptide bonds using MDFF" -command [namespace current]::cispeptideMinimizeMDFF
  button $f.resetbutton -text "Reset cispeptide plugin" -command [namespace current]::cispeptideReset
  button $f.luckybutton -state disabled -text "I'm feeling lucky" -command [namespace current]::cispeptideLucky
  pack $f.minbutton -side top -expand 1 -fill x
  pack $f.minbuttonMDFF -side top -expand 1 -fill x
  pack $f.resetbutton -side left -expand 1 -fill x
  pack $f.luckybutton -side right -expand 1 -fill x
  pack $f -side top -padx 5 -pady 2 -expand 1 -fill x

}

# Adapted from autopsf, which is adapted from pmepot...
proc ::cispeptide::GUI::fill_mol_menu {name} {
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


proc ::cispeptide::GUI::cispeptideCheck { } {

  variable currentMol
  variable currentSelText
  # variable splitMethod

  if { $currentMol == "none" } {
    tk_messageBox -type ok -message "Please load the molecule to be processed."
    return
  }
  # if {[cispeptide check -split $splitMethod -mol $currentMol -seltext $currentSelText -gui 1] != 1} {
  #   updateList
  # }
  if {[cispeptide check -mol $currentMol -seltext $currentSelText -gui 1] != -1} {
    updateList
  }

}

proc ::cispeptide::GUI::updateList { } {

  variable cispeptideList
  variable currentMol
  variable w

  set bondList [$w.list.frame.list curselection]
  set rawReturnList [cispeptide list all -mol $currentMol -gui 1]

  set cispeptideList {}
  foreach {item1 item2 action moved} $rawReturnList {
    set residue1 [lindex $item1 0]
    set resid1   [lindex $item1 1]
    set resname1 [lindex $item1 2]
    set chain1   [lindex $item1 3]
    set segname1 [lindex $item1 4]
    set residue2 [lindex $item2 0]
    set resid2   [lindex $item2 1]
    set resname2 [lindex $item2 2]
    set chain2   [lindex $item2 3]
    set segname2 [lindex $item2 4]
    if { $chain1 != $chain2 } {
      puts "cispeptide) Warning: Residues involved in a cis peptide bond don't have the same chain."
    }
    if { $segname1 != $segname2 } {
      puts "cispeptide) Warning: Residues involved in a cis peptide bond don't have the same segname."
    }
    if { $action == {X} } {
      set action {none}
    }
    if { $moved == -1 } {
      set moved {no}
    } else {
      set moved {yes}
    }
    lappend cispeptideList [format "%-3s %-9s %-3s %-9s   %1s     %-4s     %-4s    %-3s" $resname1 $resid1 $resname2 $resid2 $chain1 $segname1 $action $moved]
  }

  # Keep the same bonds selected as before
  foreach bond $bondList {
    $w.list.frame.list selection set $bond
  }

}

proc ::cispeptide::GUI::cispeptideShow { } {
  variable w
  variable currentMol

  set thisBond [$w.list.frame.list curselection]
  if {[llength $thisBond] > 1} {
    tk_messageBox -type ok -message "Please select a single cis peptide bond."
  } elseif { $thisBond != {} } {
    cispeptide show $thisBond -mol $currentMol
  }
  return
}

proc ::cispeptide::GUI::cispeptideSet { action } {
  variable w
  variable currentMol

  set bondList [$w.list.frame.list curselection]
  if { $bondList != {} } {
    foreach bond $bondList {
      cispeptide set $action -cpn $bond -mol $currentMol -gui 1
    }
  }
  updateList

  return
}

proc ::cispeptide::GUI::cispeptideMove { } {
  variable w
  variable currentMol

  set bondList [$w.list.frame.list curselection]
  if { $bondList != {} } {
    foreach bond $bondList {
      cispeptide move $bond -mol $currentMol -gui 1
    }
  }
  updateList

  return
}

proc ::cispeptide::GUI::cispeptideMinimize { } {
  variable w
  variable currentMol

  set bondList [$w.list.frame.list curselection]
  cispeptide minimize $bondList -mol $currentMol -gui 1

  return
}

proc ::cispeptide::GUI::cispeptideMinimizeMDFF { } {
  variable w
  variable currentMol

  set bondList [$w.list.frame.list curselection]
  cispeptide minimize $bondList -mol $currentMol -gui 1 -mdff 1

  return
}

proc ::cispeptide::GUI::cispeptideReset { } {
  resetGUI
  cispeptide reset
  return
}

proc ::cispeptide::GUI::cispeptideLucky { } {
  tk_messageBox -type ok -message "Are you really?"
  return
}

# This gets called by VMD the first time the menu is opened.
proc cispeptide_tk_cb {} {
  cispeptide_gui   ;# start the GUI
  return $::cispeptide::GUI::w
}

