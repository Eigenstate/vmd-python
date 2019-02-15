#
# Add ions to neutralize or achieve a desired concentration.
#
# Marcos Sotomayor, based on autoionize from Ilys Balabin.
# 
# Partially rewritten by Leonardo Trabuco, July 2010.
#
# $Id: autoionizegui.tcl,v 1.14 2018/04/20 18:49:42 jribeiro Exp $

package require autoionize
package provide autoionizegui 1.5

proc autoigui {} {
  return [::Autoi::autoi_gui]
}
 
namespace eval ::Autoi:: {
  namespace export autoi_gui

  variable w

  variable psffile
  variable pdbfile
  variable outprefix
  variable saltName
  variable ionicStrength
  variable saltConcentration
  variable neutralize
  variable userdef
  variable na
  variable cl
  variable mdistfmol
  variable mdistbion
  variable segid
  variable ksegid
  variable kclc
  variable placeMode

}

proc ::Autoi::autoi_gui {} {
  variable w

  ::Autoi::init_gui

  if { [winfo exists .autoigui] } {
    wm deiconify .autoigui
    return
  }
  set w [toplevel ".autoigui"]
  wm title $w "Autoionize"

  grid rowconfigure $w 0 -weight 1
  grid columnconfigure $w 0 -weight 1

  labelframe $w.doc
  label $w.doc.label -text "Randomly place ions in a previously solvated system" -wraplength 300
  button $w.doc.help -text "Help" -justify center -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/autoionize"
  
  grid $w.doc.label -row 1 -column 1 -columnspan 1 -sticky we -pady 2 -padx 2
  grid $w.doc.help -row 1 -column 2 -columnspan 1 -sticky ns -pady 2 -padx 2
  grid columnconfigure $w.doc 1 -weight 1
  grid columnconfigure $w.doc 2 -weight 0

  # Input
  frame $w.input
  label $w.input.label -text "Input:" -anchor w
  label $w.input.psflabel -text "PSF: " -anchor w
  entry $w.input.psfpath -width 44 -textvariable ::Autoi::psffile
  button $w.input.psfbutton -text "Browse" \
    -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} { set ::Autoi::psffile $tempfile }
    }
  label $w.input.pdblabel -text "PDB: " -anchor w
  entry $w.input.pdbpath -width 44 -textvariable ::Autoi::pdbfile
  button $w.input.pdbbutton -text "Browse" \
    -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} { set ::Autoi::pdbfile $tempfile }
    }

  grid $w.input.label     -row 1 -column 1 -columnspan 3 -sticky w 
  grid $w.input.psflabel  -row 2 -column 1 -columnspan 1 -sticky w 
  grid $w.input.psfpath   -row 2 -column 2 -columnspan 1 -sticky w 
  grid $w.input.psfbutton -row 2 -column 3 -columnspan 1 -sticky w 
  grid $w.input.pdblabel  -row 3 -column 1 -columnspan 1 -sticky w 
  grid $w.input.pdbpath   -row 3 -column 2 -columnspan 1 -sticky w 
  grid $w.input.pdbbutton -row 3 -column 3 -columnspan 1 -sticky w 

  # Output prefix / choose salt
  frame $w.outputsalt
  label $w.outputsalt.outlabel -text "Output prefix: " -anchor w
  entry $w.outputsalt.outprefix -width 25 -textvariable ::Autoi::outprefix
  label $w.outputsalt.saltlabel -text " Choose salt: " -anchor w
  tk_optionMenu $w.outputsalt.saltname ::Autoi::saltName NaCl KCl CsCl MgCl2 CaCl2 ZnCl2

  grid $w.outputsalt.outlabel  -row 1 -column 1 -columnspan 1 -sticky w 
  grid $w.outputsalt.outprefix -row 1 -column 2 -columnspan 1 -sticky w
  grid $w.outputsalt.saltlabel -row 1 -column 3 -columnspan 1 -sticky ew
  grid $w.outputsalt.saltname  -row 1 -column 4 -columnspan 1 -sticky ew

  # Ion placement mode
  labelframe $w.mode -text "Ion placement mode" -padx 2 -pady 4
  radiobutton $w.mode.neut -value "neutralize" -variable [namespace current]::placeMode
  label $w.mode.neutlabel -textvariable [namespace current]::neutLabel
  radiobutton $w.mode.sc -value "sc" -variable [namespace current]::placeMode
  label $w.mode.sclabel -textvariable [namespace current]::scLabel
  entry $w.mode.scvalue -width 8 -textvariable ::Autoi::saltConcentration
  label $w.mode.sclabel2 -text "mol/L"
  # radiobutton $w.mode.is -value "is" -variable [namespace current]::placeMode
  # label $w.mode.islabel -textvariable [namespace current]::isLabel
  # entry $w.mode.isvalue -width 8 -textvariable ::Autoi::ionicStrength
  # label $w.mode.islabel2 -text "mol/L"
  radiobutton $w.mode.userdef -value "userdef" -variable [namespace current]::placeMode
  label $w.mode.userdeflabel -textvariable [namespace current]::userdefLabel
  entry $w.mode.nsod -width 3 -textvariable ::Autoi::nSOD
  label $w.mode.sod -text "Na+"
  entry $w.mode.ncla -width 3 -textvariable ::Autoi::nCLA
  label $w.mode.cla -text "Cl-"
  entry $w.mode.npot -width 3 -textvariable ::Autoi::nPOT
  label $w.mode.pot -text "K+"
  entry $w.mode.nces -width 3 -textvariable ::Autoi::nCES
  label $w.mode.mg -text "Mg2+"
  entry $w.mode.ncal -width 3 -textvariable ::Autoi::nCAL
  label $w.mode.ces -text "Cs+"
  entry $w.mode.nmg -width 3 -textvariable ::Autoi::nMG
  label $w.mode.cal -text "Ca2+"
  entry $w.mode.nzn2 -width 3 -textvariable ::Autoi::nZN2
  label $w.mode.zn2 -text "Zn2+"

  grid $w.mode.neut  -row 1 -column 1 -columnspan 1 -sticky w 
  grid $w.mode.neutlabel  -row 1 -column 2 -columnspan 14 -sticky w 
  grid $w.mode.sc  -row 2 -column 1 -columnspan 1 -sticky w 
  grid $w.mode.sclabel  -row 2 -column 2 -columnspan 10 -sticky w 
  grid $w.mode.scvalue  -row 2 -column 12 -columnspan 2 -sticky w
  grid $w.mode.sclabel2  -row 2 -column 14 -columnspan 2 -sticky w
  # grid $w.mode.is  -row 3 -column 1 -columnspan 1 -sticky w 
  # grid $w.mode.islabel  -row 3 -column 2 -columnspan 10 -sticky w 
  # grid $w.mode.isvalue  -row 3 -column 12 -columnspan 2 -sticky w
  # grid $w.mode.islabel2  -row 3 -column 14 -columnspan 2 -sticky w
  grid $w.mode.userdef -row 4 -column 1 -columnspan 1 -sticky w
  grid $w.mode.userdeflabel -row 4 -column 2 -columnspan 9 -sticky w
  grid $w.mode.nsod -row 5 -column 2 -columnspan 1 -sticky w
  grid $w.mode.sod -row 5 -column 3 -columnspan 1 -sticky w
  grid $w.mode.ncla -row 5 -column 4 -columnspan 1 -sticky w
  grid $w.mode.cla -row 5 -column 5 -columnspan 1 -sticky w
  grid $w.mode.npot -row 5 -column 6 -columnspan 1 -sticky w
  grid $w.mode.pot -row 5 -column 7 -columnspan 1 -sticky w
  grid $w.mode.nmg -row 5 -column 8 -columnspan 1 -sticky w
  grid $w.mode.mg -row 5 -column 9 -columnspan 1 -sticky w
  grid $w.mode.nces -row 5 -column 10 -columnspan 1 -sticky w
  grid $w.mode.ces -row 5 -column 11 -columnspan 1 -sticky w
  grid $w.mode.ncal -row 5 -column 12 -columnspan 1 -sticky w
  grid $w.mode.cal -row 5 -column 13 -columnspan 1 -sticky w
  grid $w.mode.nzn2 -row 5 -column 14 -columnspan 1 -sticky w
  grid $w.mode.zn2 -row 5 -column 15 -columnspan 1 -sticky w

  # Other options
  frame $w.options
  label $w.options.minfromlabel -text "Mininum distance from solute: "
  entry $w.options.minfrom -width 4 -textvariable ::Autoi::mdistfmol
  label $w.options.minfromlabel2 -text "Angstroms"
  label $w.options.minbetweenlabel -text "Minimum distance between ions: "
  entry $w.options.minbetween -width 4 -textvariable ::Autoi::mdistbion
  label $w.options.minbetweenlabel2 -text "Angstroms"
  label $w.options.segnamelabel -text "Segment name of placed ions:"
  entry $w.options.segname -width 4 -textvariable ::Autoi::segid

  grid $w.options.minfromlabel -row 1 -column 1 -columnspan 1 -sticky w
  grid $w.options.minfrom -row 1 -column 2 -columnspan 1 -sticky w
  grid $w.options.minfromlabel2 -row 1 -column 3 -columnspan 1 -sticky w
  grid $w.options.minbetweenlabel -row 2 -column 1 -columnspan 1 -sticky w
  grid $w.options.minbetween -row 2 -column 2 -columnspan 1 -sticky w
  grid $w.options.minbetweenlabel2 -row 2 -column 3 -columnspan 1 -sticky w
  grid $w.options.segnamelabel -row 3 -column 1 -columnspan 1 -sticky w
  grid $w.options.segname -row 3 -column 2 -columnspan 1 -sticky w

  # Run autoionize button
  frame $w.run
  button $w.run.button -width 56 -text "Autoionize" -command ::Autoi::run_autoi

  grid $w.run.button -row 1 -column 1 -columnspan 1 -sticky we

  grid $w.doc -row 1 -column 1 -columnspan 3 -sticky we -padx 2 -pady 3
  grid $w.input -row 2 -column 1 -columnspan 3 -sticky we -padx 4 -pady 3
  grid $w.outputsalt -row 3 -column 1 -columnspan 4 -sticky we -padx 4 -pady 3
  grid $w.mode -row 4 -column 1 -columnspan 15 -sticky we -padx 4 -pady 3
  grid $w.options -row 5 -column 1 -columnspan 3 -sticky we -padx 6 -pady 3
  grid $w.run -row 6 -column 1 -columnspan 1 -sticky we -padx 6 -pady 3

  trace add variable ::Autoi::saltName write ::Autoi::updateLabels
 
  return $w
}

# Update some labels on the GUI
proc ::Autoi::updateLabels {args} {
  variable neutLabel
  variable scLabel
  variable isLabel
  variable userdefLabel
  variable saltName

  set neutLabel "Only neutralize system with $saltName"
  set scLabel "Neutralize and set $saltName concentration to "
  set isLabel "Neutralize and set $saltName ionic strength to "
  set userdefLabel "User-defined number of ions:"
  update 

  return
}

# Set up variables before opening the GUI
proc ::Autoi::init_gui {} {

  variable psffile
  variable pdbfile
  variable outprefix
  variable saltName
  variable ionicStrength
  variable saltConcentration
  variable neutralize
  variable userdef
  variable na
  variable cl
  variable mdistfmol
  variable mdistbion
  variable segid
  variable ksegid
  variable kclc
  variable placeMode

  # 
  # Check if the top molecule has both pdb and psf files loaded: if it does,
  # use those as a default; otherwise, leave these fields blank.
  # 
  #
  set psffile {}
  set pdbfile {}
  set kclc 0
  if {[molinfo num] != 0} {
    foreach filename [lindex [molinfo top get filename] 0] \
            filetype [lindex [molinfo top get filetype] 0] {
      if { [string equal $filetype "psf"] } {
        set psffile $filename
      } elseif { [string equal $filetype "pdb"] } {
        set pdbfile $filename
      }
    }
    # Make sure both a pdb and psf are loaded
    if { $psffile == {} || $pdbfile == {} } {
      set psffile {}
      set pdbfile {}
    } 
  }

  set outprefix "ionized"
  set segid "ION"
  set saltName {NaCl}
  set ionicStrength 0.3
  set saltConcentration 0.15
  set neutralize 1
  set userdef 0
  set mdistfmol 5
  set mdistbion 5

  # Supported values for placeMode match the placement modes in the command line:
  # - neutralize: only neutralize system (minimal ion conditions)
  # - is: neutralize and set the ionic strength
  # - sc: neutralize and set the salt concentration
  # - nions: add specified number of ions
  set placeMode {neutralize}

  updateLabels

  return
}

# Run autoionize from the GUI. Assembles a command line and passes it to
# autoionize
proc ::Autoi::run_autoi {} {
  variable psffile
  variable pdbfile
  variable outprefix
  variable ionicStrength
  variable saltConcentration
  variable saltName
  #variable na
  #variable cl
  variable mdistfmol
  variable mdistbion
  variable segid
  #variable kclc
  variable placeMode
  variable nSOD
  variable nCLA
  variable nPOT
  variable nCES
  variable nCAL
  variable nMG
  variable nZN2

  set command_line {}

  if { ($psffile == {}) || ($pdbfile == {} ) } {
    puts "autoionize: need file names"
    return
  }
  append command_line [concat "-psf" [format "{%s}" $psffile] "-pdb" [format "{%s}" $pdbfile]]

  if { $outprefix == {} } {
    puts "autoionize: need output filename"
    return
  }
  set command_line [concat $command_line "-o" $outprefix]

  if { $segid == {} } {
    puts "autoionize: need segid"
    return
  }
  set command_line [concat $command_line "-seg" $segid]

  switch $placeMode {
    neutralize {
      set command_line [concat $command_line "-neutralize"]
    }
    is {
      set command_line [concat $command_line "-is" $ionicStrength]
    }
    sc {
      set command_line [concat $command_line "-sc" $saltConcentration]
    }
    userdef {
      set nionsList {}
      if {$nSOD > 0} {
        lappend nionsList [list SOD $nSOD]
      }
      if {$nCAL > 0} {
        lappend nionsList [list CAL $nCAL]
      }
      if {$nCLA > 0} {
        lappend nionsList [list CLA $nCLA]
      }
      if {$nPOT > 0} {
        lappend nionsList [list POT $nPOT]
      }
      if {$nCES > 0} {
        lappend nionsList [list CES $nCES]
      }
      if {$nMG > 0} {
        lappend nionsList [list MG $nMG]
      }
      if {$nZN2 > 0} {
        lappend nionsList [list ZN2 $nZN2]
      }
      if {[llength $nionsList] == 0} {
        puts "autoionize: number of ions needed."
        return
      }
      set command_line [concat $command_line "-nions" [list $nionsList]]
    }
    default {error "autionize: internal error."}
  }

  if {$placeMode != {userdef}} {
    switch $saltName {
      NaCl {
        set cation "SOD"
        set anion "CLA"
      }
      KCl {
        set cation "POT"
        set anion "CLA"
      }
      CsCl {
        set cation "CES"
        set anion "CLA"
      }
      MgCl2 {
        set cation "MG"
        set anion "CLA"
      }
      CaCl2 {
        set cation "CAL"
        set anion "CLA"
      }
      ZnCl2 {
        set cation "ZN2"
        set anion "CLA"
      }
      default {error "autoionize: internal error."}
    }
    set command_line [concat $command_line "-cation" $cation "-anion" $anion]
  }

  if { $mdistfmol == {} } {
    puts "autoionize: need min distance from molecule "
    return
  }
  set command_line [concat $command_line "-from" $mdistfmol]

  if { $mdistfmol == {} } {
    puts "autoionize: need min distance between ions "
    return
  }
  set command_line [concat $command_line "-between" $mdistbion]
  
  puts "DEBUG: command_line = $command_line"
  eval autoionize $command_line

  return

}

proc ::Autoi::is_number {args} {
  if {[llength $args] != 1} {
    return 0
  }

  set x [lindex $args 0]
  if { ($x == {}) || [catch {expr $x + 0}]} {
    return 0
  } else {
    return 1
  }
}

