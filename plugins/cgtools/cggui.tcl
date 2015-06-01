##
## Gui for cg builder and reverse cg "builder"
##
## Author: Kirby Vandivort
##         biocore@ks.uiuc.edu
##         vmd@ks.uiuc.edu
##
## $Id: cggui.tcl,v 1.36 2013/04/03 20:18:33 kvandivo Exp $
##

## Tell Tcl that we're a package and any dependencies we may have

# version
#  0.22 - added SBCG proc to extract bond/angle params from AA simulation
#  0.23 - changed wording on bond/angle extraction output fname. Added
#         scaling of bond/angle constants in a parameter file


package provide cggui 1.2

package require cgtools 1.2
package require cgnetworking 1.2
# package require autopsf
package require solvate

namespace eval ::cggui:: {
#  namespace export cggui

  # window handle
  variable w                                          
  variable toCGResidueFrame                                          
  variable toCGShapeFrame                                          
  variable fromCGFrame                                          
  variable mapSBFrame                                          
  variable ljParamFrame                                          
  variable scaleBAFrame
  variable baFromSimFrame
  variable chooseCGFrame                                          
  variable chooseFrame                                          
  variable pdbfile
  variable outprefix "cg_"
  variable outfromprefix "aa_ref_"
  variable nullMolString "none loaded.  Load a molecule first."
  variable isToCGMolLoaded 0
  variable isFromCGMolLoaded 0

  variable isMapAAMolLoaded 0
  variable isMapCGMolLoaded 0
  variable isMapRefMolLoaded 0

  variable isljparamMolLoaded 0
  variable currentToCGMol $nullMolString
  variable currentFromCGMol $nullMolString
  variable currentFromCGAAMol $nullMolString
  variable ljparamCurrentMol $nullMolString
  variable toResCGMenu
  variable toShapeCGMenu
  variable fromMolMenu
  variable fromAAMolMenu
  variable mapAAMenu
  variable mapCGMenu
  variable mapRefMenu
  variable ljparamMenu
  variable tomolMenuText
  variable frommolMenuText
  variable fromaamolMenuText
  variable mapcgMenuText
  variable mapaaMenuText
  variable maprefMenuText
  variable cgpath
  variable toCGoutpdbfile
  variable toCGoutallpdbfile
  variable toCGouttopfile
  variable toCGoutparmfile
  variable fromCGoutpdbfile
  variable revcgfile
  variable do_rotations 1

  variable numBeadsLabel

  variable shapeChoice "0"
  variable edmPath
  variable toCGshapeResName "FUP"
  variable toCGshapeNamePrefix "B"
  variable toCGshapeNumCGBeads
  variable toCGshapeNumSteps
  variable toCGshapeEpsInit "0.3"
  variable toCGshapeEpsFinal "0.05"
  variable toCGshapeLambdaInit
  variable toCGshapeLambdaFinal "0.01"
  # 0=get bond info all-atom backbone connections; 1=use user-specified distance
  variable toCGshapeBondMethod "0" 
  variable toCGshapeBondCutoff "25"
  variable toCGshapeFracCutoff "0.01"

  variable toCGShapeUseMass 
  variable toCGshapeMass 
  variable drawn 0

  #  kAtomsPerDXPoint chosen to make hook.dx generate 15 beads.
  #  taken from the number of points in the file (listed as the last
  # integer in the object 3 line).  24955 / 550 / 3 => 15, which is what
  # we want
  variable kAtomsPerDXPoint 550
  variable kAtomsPerSitusPoint 11300

  variable kAtomsPerBead 500.0
  variable kStepsPerBead 200
  variable kLambdaMult 0.2

  variable menuChoice 0
  variable ionize 0
  variable autopsf 0
  variable solvate 0

  variable annealInFile ""
  variable annealChoice
  variable annealConfig 1
  variable annealParList         {}
  lappend annealParList [file join \
                           $env(CHARMMPARDIR) par_all27_prot_lipid_na.inp]
  variable annealPSFFile

  variable ljparamInparfile
  variable ljparamInEnergy
  variable ljparamInRadius
  variable ljparamOutparfile

  variable baFromSimCGpsfFile
  variable baFromSimCGpdbFile
  variable baFromSimAApdbFile
  variable baFromSimAAdcdFile
  variable baFromSimInTemp
  variable baFromSimOutparfile
  variable baFromSimOutbondparfile
  variable baFromSimOutangleparfile

  variable scaleBAinFile
  variable scaleBAoutFile
  variable scaleBAbondScale 1
  variable scaleBAbondCutoff 0
  variable scaleBAangleScale 1
  variable scaleBAangleCutoff 0


}

# -------------------------------------------------------------------------
# should the user be able to pick 'next'
proc ::cggui::toggleNextButton { a b c } {
   variable menuChoice
   variable w
   if { $menuChoice > 0 } {   
     $w.chooseFrame.next configure -state normal
   } else {
     $w.chooseFrame.next configure -state disabled
   }
}

# -------------------------------------------------------------------------
proc ::cggui::toggleMassBox { a b c } {
   variable toCGShapeUseMass
   variable toCGShapeFrame
   variable drawn

   if { $drawn == 0 } {
      return
   }
   if { $toCGShapeUseMass == 1 } {
      $toCGShapeFrame.mass.numText configure -state normal
      $toCGShapeFrame.mass.num configure -state normal
   } else {
      $toCGShapeFrame.mass.numText configure -state disabled
      $toCGShapeFrame.mass.num configure -state disabled
   }

}

# -------------------------------------------------------------------------
# see if we should calculate the suggested number of steps for the user
proc ::cggui::calcShapeValues { } {
   variable toCGshapeNumSteps
   if {[string trim $::cggui::toCGshapeNumCGBeads] == "" || \
       ! [string is integer $::cggui::toCGshapeNumCGBeads] || \
       $::cggui::toCGshapeNumCGBeads < 1 } {
          return 0
   }
#   if {[string trim $::cggui::toCGshapeNumSteps] == "" || \
#       ! [string is integer $::cggui::toCGshapeNumSteps] || \
#       $::cggui::toCGshapeNumSteps < 1 } {
      set toCGshapeNumSteps [ expr { $::cggui::kStepsPerBead * $::cggui::toCGshapeNumCGBeads } ]
#   }

   variable toCGshapeLambdaInit
#   if {[string trim $::cggui::toCGshapeLambdaInit] == "" || \
#       ! [string is double $::cggui::toCGshapeLambdaInit] || \
#       $::cggui::toCGshapeLambdaInit < 0 } {
      set toCGshapeLambdaInit [ expr { $::cggui::kLambdaMult * $::cggui::toCGshapeNumCGBeads } ]
#   }
   return 1
}

# -------------------------------------------------------------------------
proc ::cggui::shapeBondMeth { } {
   variable toCGShapeFrame
   if { $::cggui::toCGshapeBondMethod == "0" } {
      # cutoff text box
      $toCGShapeFrame.parms.toCGshapeBondCutoff configure -state disabled

   } elseif { $::cggui::toCGshapeBondMethod == "1" } {
      $toCGShapeFrame.parms.toCGshapeBondCutoff configure -state normal
   }

}
# -------------------------------------------------------------------------
proc ::cggui::shapeSourceChoice { } {
   variable toCGShapeFrame
   variable shapeChoice
   variable toCGShapeUseMass

   if { $shapeChoice == "0" } {
      # molecule
      $toCGShapeFrame.parms.fracCutoffLabel configure -state disabled
      $toCGShapeFrame.parms.toCGshapeFracCutoff configure -state disabled

      $toCGShapeFrame.mollable configure -state normal
      $toCGShapeFrame.mol configure -state normal
#      $toCGShapeFrame.mol.menu configure -state normal

      $toCGShapeFrame.edmFilePathText configure -state disabled
      $toCGShapeFrame.edmPath configure -state disabled
      $toCGShapeFrame.edmButton configure -state disabled

      $toCGShapeFrame.outFiles.outallpdblabel configure -state normal
      $toCGShapeFrame.outFiles.toCGoutallpdbfile configure -state normal

      set toCGShapeUseMass 0
   } elseif { $shapeChoice == "1" } {
      # EDM
      $toCGShapeFrame.parms.fracCutoffLabel configure -state normal
      $toCGShapeFrame.parms.toCGshapeFracCutoff configure -state normal

      $toCGShapeFrame.mollable configure -state disabled
      $toCGShapeFrame.mol configure -state disabled
#      $toCGShapeFrame.mol.menu configure -state disabled

      $toCGShapeFrame.edmFilePathText configure -state normal
      $toCGShapeFrame.edmPath configure -state normal
      $toCGShapeFrame.edmButton configure -state normal

      $toCGShapeFrame.outFiles.outallpdblabel configure -state disabled
      $toCGShapeFrame.outFiles.toCGoutallpdbfile configure -state disabled

      # probably want this to normally be 1
      #set ::cggui::toCGShapeUseMass 1
      set toCGShapeUseMass 0
   }

}


# -------------------------------------------------------------------------
#
# Create the window and initialize data structures
#
proc ::cggui::cggui {} {
#   puts "very beginning of ::cggui::cggui";
  variable toResCGMenu
  variable toShapeCGMenu
  variable fromMolMenu
  variable fromAAMolMenu
  variable mapAAMenu
  variable mapCGMenu
  variable mapRefMenu
  variable w
  variable chooseFrame
  variable chooseCGFrame
  variable toCGResidueFrame
  variable toCGShapeFrame
  variable fromCGFrame
  variable mapSBFrame                                          
  variable ljParamFrame                                          
  variable baFromSimFrame                                          
  variable scaleBAFrame                                          
  variable numBeads

#  ::cggui::init_vars

#  puts "before everything starts\n";

  trace add variable ::cggui::currentToCGMol write ::cggui::tomolmenuaux
  trace add variable ::cggui::currentFromCGMol write ::cggui::frommolmenuaux
  trace add variable ::cggui::currentFromCGAAMol write ::cggui::fromaamolmenuaux

  trace add variable ::cggui::currentMapCGMol write ::cggui::mapcgmenuaux
  trace add variable ::cggui::currentMapAAMol write ::cggui::mapaamenuaux
  trace add variable ::cggui::currentMapRefMol write ::cggui::maprefmenuaux

  trace add variable ::cggui::ljparamCurrentMol write ::cggui::ljparammenuaux

  trace remove variable ::cggui::menuChoice write ::cggui::toggleNextButton
  trace add variable ::cggui::menuChoice write ::cggui::toggleNextButton

  trace add variable ::cggui::toCGShapeUseMass write ::cggui::toggleMassBox

  # If already initialized, just turn on
  if { [winfo exists .cggui] } {
    wm deiconify $w
    return
  }

  set w [toplevel ".cggui"]
  wm title $w "CG Builder - Main Menu"

  #Add a menubar
  frame $w.menubar -relief raised -bd 2
  #grid  $w.menubar -padx 1 -column 0 -columnspan 5 -row 0 -sticky ew
  pack $w.menubar -padx 1 -fill x

  menubutton $w.menubar.help -text "Help" -underline 0 \
    -menu $w.menubar.help.menu
  $w.menubar.help config -width 5
  pack $w.menubar.help -side right

  ## help menu
  menu $w.menubar.help.menu -tearoff no
  $w.menubar.help.menu add command -label "About" \
    -command {tk_messageBox -type ok -title "About CG Tools" \
              -message "Coarse Graining in VMD."}
  $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www]]plugins/cgtools"

# now, we define a few frames


## ---------------------------------------------------------------------------
## ---------------------- START choose order FRAME ---------------------------
  set chooseFrame [frame $w.chooseFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $chooseFrame.introText -text "Coarse Graining"] \
     -row $row -column 0 -columnspan 2 
  incr row

  # -----------------------------------------------
  grid [labelframe $chooseFrame.resCg -bd 2 -relief ridge \
            -text "Residue-Based Coarse Graining (RBCG) Tools" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  incr row

  grid [radiobutton $chooseFrame.resCg.r1 -text \
                         "Create RBCG Model" \
                   -variable ::cggui::menuChoice -value 1 ] \
        -row 0 -column 0 -sticky w

  grid [radiobutton $chooseFrame.resCg.r2 -text \
                  "Reverse Previously RBCG Model Back To All-Atom" \
                   -variable ::cggui::menuChoice -value 3 ] \
        -row 1 -column 0 -sticky w
  # ---

  grid [labelframe $chooseFrame.sCg -bd 2 -relief ridge \
            -text "Shape-Based Coarse Graining (SBCG) Tools" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  incr row

  grid [radiobutton $chooseFrame.sCg.r1 -text \
                         "Create SBCG Model" \
                   -variable ::cggui::menuChoice -value 2 ] \
        -row 0 -column 0 -sticky w

  grid [radiobutton $chooseFrame.sCg.r2 -text \
                  "Map A Previously Generated SBCG Model To An All-Atom Model" \
                   -variable ::cggui::menuChoice -value 4 ] \
        -row 1 -column 0 -sticky w

  grid [radiobutton $chooseFrame.sCg.r3 -text \
                  "Assign Lennard-Jones Params For CG Model From All-Atom" \
                   -variable ::cggui::menuChoice -value 5 ] \
        -row 2 -column 0 -sticky w

  grid [radiobutton $chooseFrame.sCg.r4 -text \
             "Extract Bond/Angle Params of CG Model from All-Atom Simulation" \
                   -variable ::cggui::menuChoice -value 6 ] \
        -row 3 -column 0 -sticky w

  grid [radiobutton $chooseFrame.sCg.r5 -text \
             "Scale Bond/Angle Spring Constants In A Parameter File" \
                   -variable ::cggui::menuChoice -value 7 ] \
        -row 4 -column 0 -sticky w


  grid [button $chooseFrame.next -text "Next ->" -state disabled \
        -command {
           if { $::cggui::menuChoice > 0 } {   
              pack forget $::cggui::chooseFrame   
              if { $::cggui::menuChoice == 1 } {   
                 pack $::cggui::toCGResidueFrame   
                 wm title $::cggui::w "CG Builder - Residue-Based CG"
              } elseif { $::cggui::menuChoice == 2 } {
                 #pack $::cggui::toCGResidueFrame   
                 pack $::cggui::toCGShapeFrame   
                 wm title $::cggui::w "CG Builder - Shape-Based CG"
              } elseif { $::cggui::menuChoice == 3 } {
                 pack $::cggui::fromCGFrame   
                 wm title $::cggui::w "Reverse Residue-Based CG"
              } elseif { $::cggui::menuChoice == 4 } {
                 pack $::cggui::mapSBFrame   
                 wm title $::cggui::w "Map Shaped-Based CG To All Atom"
              } elseif { $::cggui::menuChoice == 5 } {
                 pack $::cggui::ljParamFrame   
                 wm title $::cggui::w "Get Lennard-Jones Params for CG model"
              } elseif { $::cggui::menuChoice == 6 } {
                 pack $::cggui::baFromSimFrame 
                 wm title $::cggui::w "Extract Bond/Angle Params from AA Sim"
              } elseif { $::cggui::menuChoice == 7 } {
                 pack $::cggui::scaleBAFrame 
                 wm title $::cggui::w "Scale Bond/Angle Spring Constants"
              } else {
                 tk_messageBox -type ok -message \
                    "Unknown Menu Choice.  Contact Developers" -title "Error!"
              }
           }   
        }] \
        -row $row -column 0 -columnspan 2 -sticky ew
  incr row

## ---------------------- END   choose order FRAME ---------------------------
## ---------------------------------------------------------------------------

#   ::cggui::createToCGResidueFrame {}
#puts "before doing residue TO frame"
## ---------------------------------------------------------------------------
## ---------------------- START to CG via RESIDUE FRAME ---------------------
  set toCGResidueFrame [frame $w.toCGResidueFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $toCGResidueFrame.introText -text "Coarse Grain Builder"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row
  grid [label $toCGResidueFrame.introTextDesc1 -text \
          "Convert an all-atom representation to coarse-grained"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row
  grid [label $toCGResidueFrame.introTextDesc2 -text \
          "using residue-based coarse graining."] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  # -----------------------------------------------
  # molecule chooser
  #puts "before grid"
  grid [label $toCGResidueFrame.mollable -text "Molecule: "] \
    -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $toCGResidueFrame.mol -textvar [namespace current]::tomolMenuText \
    -menu $toCGResidueFrame.mol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"
  set toResCGMenu [menu $toCGResidueFrame.mol.menu -tearoff no]
  incr row
  ::cggui::fill_mol_menu toResCGMenu ::cggui::isToCGMolLoaded \
                                             currentToCGMol


  #puts "before cg database"
  # -----------------------------------------------
  # deal with the CG database
  grid [labelframe $toCGResidueFrame.database -bd 2 -relief ridge \
            -text "CG Database" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  incr row

#  frame $toCGResidueFrame.database -relief groove -bd 3
#                       -text "[file join $env(CGTOOLSDIR) protein.cgc]"] \

#  puts stderr "$::env(CGTOOLSDIR)\n\n\n\n"

  set rowDB 0

  grid [label $toCGResidueFrame.database.pLabel -text "Proteins"] \
      -row $rowDB -column 0 -sticky w
  grid [label $toCGResidueFrame.database.pPath \
                   -text "([file join $::env(CGTOOLSDIR) protein.cgc])"] \
      -row $rowDB -column 1 -columnspan 2 -sticky w
  grid [button $toCGResidueFrame.database.paddbutton -text "Add" \
        -command {
           ::cgtools::read_db [file join $::env(CGTOOLSDIR) protein.cgc]
           $::cggui::numBeadsLabel configure \
                                  -text "[llength $::cgtools::convbeads]"
           $::cggui::toCGResidueFrame.database.paddbutton \
                                                    configure -state disabled
           $::cggui::toCGResidueFrame.database.paddbutton \
                                                    configure -text "Added!"
        }] -row $rowDB -column 3 -sticky e
  incr rowDB

  grid [label $toCGResidueFrame.database.wLabel -text "Water"] \
      -row $rowDB -column 0 -sticky w
  grid [label $toCGResidueFrame.database.wPath \
                       -text "([file join $::env(CGTOOLSDIR) water.cgc])"] \
      -row $rowDB -column 1 -columnspan 2 -sticky w
  grid [button $toCGResidueFrame.database.waddbutton -text "Add" \
        -command {
           ::cgtools::read_db [file join $::env(CGTOOLSDIR) water.cgc]
           $::cggui::numBeadsLabel configure \
                                  -text "[llength $::cgtools::convbeads]"
           $::cggui::toCGResidueFrame.database.waddbutton \
                                                    configure -state disabled
           $::cggui::toCGResidueFrame.database.waddbutton \
                                                    configure -text "Added!"
        }] -row $rowDB -column 3 -sticky e
  incr rowDB

  grid [label $toCGResidueFrame.database.udLabel -text "User Defined"] \
      -row $rowDB -column 0 -sticky w

  grid [entry $toCGResidueFrame.database.udpath -width 46 \
                                      -textvariable ::cggui::cgpath ] \
      -row $rowDB -column 1 -sticky ew
  grid [button $toCGResidueFrame.database.udbutton -text "Browse" \
        -command {
           set tempfile [tk_getOpenFile]
           if {![string equal $tempfile ""]} { set ::cggui::cgpath $tempfile }
        }] -row $rowDB -column 2 -sticky w
  grid [button $toCGResidueFrame.database.udaddbutton -text "Add" \
        -command {
           ::cgtools::read_db $::cggui::cgpath
           $::cggui::numBeadsLabel configure \
                                  -text "[llength $::cgtools::convbeads]"
        }] -row $rowDB -column 3 -sticky e

  incr rowDB


  grid [label $toCGResidueFrame.database.beadsText \
                              -text "Bead Definitions Currently Loaded:"] \
     -row $rowDB -column 1 -sticky w
  set ::cggui::numBeadsLabel [label $toCGResidueFrame.database.beadsNum \
                                  -text "[llength $::cgtools::convbeads]"]
  grid $::cggui::numBeadsLabel -row $rowDB -column 2 -sticky e
  incr rowDB

#  # -----------------------------------------------
#  # deal with the Extras
#  grid [labelframe $toCGResidueFrame.extras -bd 2 -relief ridge -text "Extras" \
#            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
#  incr row
#
#  checkbutton $toCGResidueFrame.extras.autopsf -text "Auto Create PSF(Inactive)" \
#                                     -width 8 \
#                                     -variable [namespace current]::autopsf
#
#  checkbutton $toCGResidueFrame.extras.solvate -text "Solvate(Inactive)" -width 8 \
#                                     -variable [namespace current]::solvate
#
##  checkbutton $toCGResidueFrame.extras.ionize -text "Ionize" -width 7 \
##                                     -variable [namespace current]::ionize
#
#  pack $toCGResidueFrame.extras.autopsf -side left -fill x -expand 1
#  pack $toCGResidueFrame.extras.solvate -side left -fill x -expand 1
#  pack $toCGResidueFrame.extras.ionize  -side left -fill x -expand 1

  #grid $toCGResidueFrame.database -row $row -column 0 -sticky ew -columnspan 2
  #incr row
  # -----------------------------------------------
  # Output filenames
  grid [label $toCGResidueFrame.outpdblabel -text "Output PDB: "] \
    -row $row -column 0 -sticky w
  grid [entry $toCGResidueFrame.toCGoutpdbfile -width 30 -textvariable ::cggui::toCGoutpdbfile] \
    -row $row -column 1 -sticky ew
  incr row
  grid [label $toCGResidueFrame.revcglabel -text "Rev CG File: "] -row $row \
               -column 0 -sticky w 
  grid [entry $toCGResidueFrame.revcgfile -width 30 \
           -textvariable ::cggui::revcgfile] -row $row -column 1 -sticky ew 
  incr row


  # -----------------------------------------------
  grid [button $toCGResidueFrame.back -text "Back To Previous Screen" \
        -command { \
            pack forget $::cggui::toCGResidueFrame   
            set ::cggui::menuChoice 0
            wm title $::cggui::w "CG Builder - Method Selection"
            pack $::cggui::chooseFrame }] \
        -row $row -column 0 -sticky w
  grid [button $toCGResidueFrame.applyDB -text "Build Coarse Grain Model" \
        -command ::cggui::buildResidueCGExecute] \
        -row $row -column 1 -sticky e
  incr row

## ---------------------- END    to residue CG rep FRAME -------------------
## ---------------------------------------------------------------------------


#puts "before doing shape TO frame"
## ---------------------------------------------------------------------------
## ---------------------- START to shape CG rep FRAME ----------------------
  set toCGShapeFrame [frame $w.toCGShapeFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $toCGShapeFrame.introTextDesc2 -text \
          "Shape-based Coarse Grained Builder."] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  # -----------------------------------------------
  grid [label $toCGShapeFrame.choiceText -text \
          "First, do you want to CG a Molecule or an Electron Density Map?"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  set ::cggui::shapeChoice "0"

  grid [radiobutton  $toCGShapeFrame.molText -text "Molecule" \
                            -value "0" \
                            -variable [namespace current]::shapeChoice \
                            -command ::cggui::shapeSourceChoice ] \
     -row $row -column 0 -sticky w

  grid [radiobutton  $toCGShapeFrame.edmText -text \
                            "Electron Density Map" \
                            -value "1" \
                            -variable [namespace current]::shapeChoice \
                            -command ::cggui::shapeSourceChoice ] \
     -row $row -column 1 -sticky e
  incr row

  # -----------------------------------------------
  # molecule chooser
  #puts "before grid"
  grid [label $toCGShapeFrame.mollable -text "Molecule: "] \
                -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $toCGShapeFrame.mol -textvar \
                        [namespace current]::tomolMenuText \
    -menu $toCGShapeFrame.mol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"

  set toShapeCGMenu [menu $toCGShapeFrame.mol.menu -tearoff no]
  incr row
  ::cggui::fill_mol_menu toShapeCGMenu ::cggui::isToCGMolLoaded \
                                             currentToCGMol

  # -----------------------------------------------
  grid [label $toCGShapeFrame.edmFilePathText -text \
          "Electron Density Map file (SITUS or .dx file type)"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  grid [entry $toCGShapeFrame.edmPath -width 31 -textvariable ::cggui::edmPath ] \
      -row $row -column 0 -sticky ew
  grid [button $toCGShapeFrame.edmButton -text "Browse" \
        -command {
           set tempfile [tk_getOpenFile]
           if {![string equal $tempfile ""]} { \
              set ::cggui::edmPath $tempfile \
           }
           ::cggui::setEDMValues
        }] -row $row -column 1 -sticky w
  incr row

  # Define Mass
  grid [labelframe $toCGShapeFrame.mass -bd 2 -relief ridge \
            -text "Mass Of CG Model" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  incr row

  grid [checkbutton $toCGShapeFrame.mass.choiceButton -variable \
             [namespace current]::toCGShapeUseMass] -row 0 -column 0 -sticky w

  grid [label $toCGShapeFrame.mass.choiceText -text \
                 "Define Total Mass of CG Model?       "] \
        -row 0 -column 1 -sticky w

  grid [label $toCGShapeFrame.mass.numText -text \
                 "     Total Mass:"] \
        -row 0 -column 2 -sticky e

  grid [entry $toCGShapeFrame.mass.num -width 7 \
               -textvariable ::cggui::toCGshapeMass] \
    -row 0 -column 3 -sticky e

  # -----------------------------------------------
  # Learning Parameters
  grid [labelframe $toCGShapeFrame.parms -bd 2 -relief ridge \
            -text "Learning Parameters" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  incr row

  set row2 0
  grid [label $toCGShapeFrame.parms.numCGBeadsLabel -text \
               "Number of CG Beads"] -row $row2 -column 0 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeNumCGBeads -width 7 \
               -validate focusout \
               -validatecommand [namespace current]::calcShapeValues \
               -textvariable ::cggui::toCGshapeNumCGBeads] \
    -row $row2 -column 1 -sticky w

  grid [label $toCGShapeFrame.parms.numStepsLabel -text \
               "Number of Learning Steps"] -row $row2 -column 2 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeNumSteps -width 7 \
               -textvariable ::cggui::toCGshapeNumSteps] \
    -row $row2 -column 3 -sticky e
  incr row2

  grid [label $toCGShapeFrame.parms.epsInitLabel -text \
               "Initial eps"] -row $row2 -column 0 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeEpsInit -width 7 \
               -textvariable ::cggui::toCGshapeEpsInit] \
    -row $row2 -column 1 -sticky w

  grid [label $toCGShapeFrame.parms.epsFinalLabel -text \
               "Final eps"] -row $row2 -column 2 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeEpsFinal -width 7 \
               -textvariable ::cggui::toCGshapeEpsFinal] \
    -row $row2 -column 3 -sticky e
  incr row2

  grid [label $toCGShapeFrame.parms.lambdaInitLabel -text \
               "Initial Lambda"] -row $row2 -column 0 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeLambdaInit -width 7 \
               -textvariable ::cggui::toCGshapeLambdaInit] \
    -row $row2 -column 1 -sticky w

  grid [label $toCGShapeFrame.parms.lambdaFinalLabel -text \
               "Final Lambda"] -row $row2 -column 2 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeLambdaFinal -width 7 \
               -textvariable ::cggui::toCGshapeLambdaFinal] \
    -row $row2 -column 3 -sticky e
  incr row2

  grid [label $toCGShapeFrame.parms.fracCutoffLabel -text \
                               "Frac Cutoff (0 <= x < 1.0)" -state disabled] \
            -row $row2 -column 0 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeFracCutoff \
                  -width 7 \
                  -textvariable ::cggui::toCGshapeFracCutoff -state disabled] \
            -row $row2 -column 1 -sticky w
  incr row2

  grid [radiobutton  $toCGShapeFrame.parms.bondCutAA -text \
                            "Determine Bonds From All Atom" \
                            -value "0" \
                            -variable [namespace current]::toCGshapeBondMethod \
                            -command [namespace current]::shapeBondMeth ] \
     -row $row2 -column 0 -columnspan 2 -sticky w

  grid [radiobutton  $toCGShapeFrame.parms.bondCutDist -text \
                            "Provide Bond Cutoff" \
                            -value "1" \
                            -variable [namespace current]::toCGshapeBondMethod \
                            -command [namespace current]::shapeBondMeth ] \
     -row $row2 -column 2 -sticky w


#  grid [label $toCGShapeFrame.parms.bondCutoffLabel -text \
#               "Bond Cutoff"] -row $row2 -column 0 -sticky e
  grid [entry $toCGShapeFrame.parms.toCGshapeBondCutoff -width 7 \
               -textvariable ::cggui::toCGshapeBondCutoff -state disabled] \
    -row $row2 -column 3 -sticky w
  incr row2

  # -----------------------------------------------
  # Output Parameters
  grid [labelframe $toCGShapeFrame.outFiles -bd 2 -relief ridge \
            -text "Output Parameters" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  incr row

  set row2 0 
  grid [label $toCGShapeFrame.outFiles.cgReslabel -text  "CG ResidueName" ] \
        -row $row2 -column 0 -sticky w 
  grid [entry $toCGShapeFrame.outFiles.toCGshapeRes -width 4 \
            -textvariable ::cggui::toCGshapeResName] \
       -row $row2 -column 1 -sticky w

  grid [label $toCGShapeFrame.outFiles.cgNamelabel -text \
               "CG Name Prefix"] -row $row2 -column 2 -sticky e
  grid [entry $toCGShapeFrame.outFiles.toCGshapeNamePrefix -width 2 \
               -textvariable ::cggui::toCGshapeNamePrefix] \
    -row $row2 -column 3 -sticky e
  incr row2


  grid [label $toCGShapeFrame.outFiles.outpdblabel -text \
               "Coarse-Grained PDB: "] -row $row2 -column 0 -columnspan 2 -sticky w
  grid [entry $toCGShapeFrame.outFiles.toCGoutpdbfile -width 30 \
               -textvariable ::cggui::toCGoutpdbfile] \
    -row $row2 -column 2 -columnspan 2 -sticky ew
  incr row2

  grid [label $toCGShapeFrame.outFiles.outallpdblabel -text \
         "All-Atom Reference PDB: "] -row $row2 -column 0 -columnspan 2 -sticky w
  grid [entry $toCGShapeFrame.outFiles.toCGoutallpdbfile -width 30 \
        -textvariable ::cggui::toCGoutallpdbfile] \
    -row $row2 -column 2 -columnspan 2 -sticky ew
  incr row2



  grid [label $toCGShapeFrame.outFiles.outtoplabel -text "CG Topology File: "] \
    -row $row2 -column 0 -columnspan 2 -sticky w
  grid [entry $toCGShapeFrame.outFiles.toCGouttopfile -width 30 \
                -textvariable ::cggui::toCGouttopfile] \
    -row $row2 -column 2 -columnspan 2 -sticky ew
  incr row2

  grid [label $toCGShapeFrame.outFiles.outparmlabel -text "CG Parameter File: "] \
    -row $row2 -column 0 -columnspan 2 -sticky w
  grid [entry $toCGShapeFrame.outFiles.toCGoutparmfile -width 30 \
                -textvariable ::cggui::toCGoutparmfile] \
    -row $row2 -column 2 -columnspan 2 -sticky ew
  incr row2

  # -----------------------------------------------
  grid [button $toCGShapeFrame.back -text "Back To Previous Screen" \
        -command { \
            pack forget $::cggui::toCGShapeFrame   
            set ::cggui::menuChoice 0
            wm title $::cggui::w "CG Builder - Method Selection"
            pack $::cggui::chooseFrame }] \
        -row $row -column 0 -sticky w
  grid [button $toCGShapeFrame.applyDB -text "Build Coarse Grain Model" \
        -command ::cggui::buildShapeCGExecute] \
        -row $row -column 1 -sticky e
  incr row

  set statLblFrame [labelframe $toCGShapeFrame.statFrame -bd 2 -relief ridge \
         -text "Status" -padx 1m -pady 1m]
  set statBoxText [text $toCGShapeFrame.statFrame.statBox -state disabled \
         -yscrollcommand "$toCGShapeFrame.statFrame.scroll set" -setgrid true \
         -width 55 -height 9 -wrap word]
  set statBoxScroll [scrollbar $toCGShapeFrame.statFrame.scroll -command \
         "$toCGShapeFrame.statFrame.statBox yview"]

  grid $statLblFrame -row $row -column 0 -columnspan 2 -sticky nsew
  grid $statBoxText -row 0 -column 0 -sticky ew
  grid $statBoxScroll -row 0 -column 1 -sticky ens
  incr row

  # ok.. we are now drawn
  set ::cggui::drawn 1

  # this needs to be done after all of the widgets exist
  ::cggui::shapeSourceChoice

#  ::cggui::addStatusLine "\n  Ready..."

## ---------------------- END    to shape CG rep FRAME -------------------
## ---------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## ---------------------- START from CG rep FRAME ---------------------------
  set fromCGFrame [frame $w.fromCGFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $fromCGFrame.introText -text "Reverse Coarse Graining"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row
  grid [label $fromCGFrame.introTextDesc -text \
          "Convert residue-based coarse-grained back to an all-atom representation."] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  # -----------------------------------------------
  # CG molecule chooser
  #puts "before grid"
  grid [label $fromCGFrame.cgmollable -text "Coarse-Grained Molecule: "] \
    -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $fromCGFrame.cgmol -textvar [namespace current]::frommolMenuText\
    -menu $fromCGFrame.cgmol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"
  set fromMolMenu [menu $fromCGFrame.cgmol.menu -tearoff no]
  incr row
  fill_mol_menu fromMolMenu ::cggui::isFromCGMolLoaded \
                                    currentFromCGMol

  # -----------------------------------------------
  # All atom molecule chooser
  #puts "before grid"
  grid [label $fromCGFrame.aamollable -text "All-Atom Molecule: "] \
    -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $fromCGFrame.aamol -textvar [namespace current]::fromaamolMenuText\
    -menu $fromCGFrame.aamol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"
  set fromAAMolMenu [menu $fromCGFrame.aamol.menu -tearoff no]
  incr row
  fill_mol_menu fromAAMolMenu ::cggui::isFromCGMolLoaded \
                                    currentFromCGAAMol

  grid [radiobutton $fromCGFrame.rotlabel -variable ::cggui::do_rotations -text "Do rotations to optimize reversal: " -value 1] \
    -row $row -column 0 -sticky w
  incr row


  grid [label $fromCGFrame.revcglabel -text "Rev CG File: "] \
    -row $row -column 0 -sticky w
  grid [entry $fromCGFrame.revcgfile -width 30 -textvariable ::cggui::revcgfile ] \
      -row $row -column 1 -sticky ew
  grid [button $fromCGFrame.revcgfilebutton -text "Browse" \
        -command {
           set tempfile [tk_getOpenFile]
           if {![string equal $tempfile ""]} { set ::cggui::revcgfile $tempfile }
        }] -row $row -column 2 -sticky w
  incr row

  # Output filenames
  grid [label $fromCGFrame.outpdblabel -text "Output PDB: "] \
    -row $row -column 0 -sticky w
  grid [entry $fromCGFrame.fromCGoutpdbfile -width 30 -textvariable ::cggui::fromCGoutpdbfile] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

# -----------------------------------------------
# deal with the Annealing
  grid [labelframe $fromCGFrame.anneal -bd 2 -relief raised -text \
               "Anneal All-Atom PDB With NAMD" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 3 -sticky nsew
  incr row

  set anRow 0
  grid [label $fromCGFrame.anneal.choiceText -text \
                 "Prepare NAMD Configuration File For Simulated Annealing?"] \
        -row $anRow -column 0 -columnspan 2 -sticky w

  grid [checkbutton $fromCGFrame.anneal.choiceButton -variable \
             [namespace current]::annealConfig] -row $anRow -column 2 -sticky w
  incr anRow

  # Parameter files ----------------- BEGIN
  grid [labelframe $fromCGFrame.anneal.par -bd 2 -text \
                                         "Parameter files" -padx 1m -pady 1m] \
                         -row $anRow -column 0 -columnspan 3 -sticky nsew

  frame $fromCGFrame.anneal.par.multi
  scrollbar $fromCGFrame.anneal.par.multi.scroll -command \
                              "$fromCGFrame.anneal.par.multi.list yview"
  listbox $fromCGFrame.anneal.par.multi.list -yscroll \
                           "$fromCGFrame.anneal.par.multi.scroll set" \
                       -width 50 -height 3 -setgrid 1 -selectmode extended \
                       -listvariable ::cggui::annealParList
  pack $fromCGFrame.anneal.par.multi.list \
            $fromCGFrame.anneal.par.multi.scroll -side left -fill y -expand 1

  frame  $fromCGFrame.anneal.par.multi.buttons
  button $fromCGFrame.anneal.par.multi.buttons.add -text "Add" -command {
     set tempfile [tk_getOpenFile -filetypes { \
                       {{Parameter Files}       {.par}        } \
                       {{Parameter Files}      {.inp}        } 
                       {{All Files}        *             }} ]
     if {![string equal $tempfile ""]} { 
        lappend ::cggui::annealParList $tempfile 
     }
  }
  button $fromCGFrame.anneal.par.multi.buttons.delete -text "Delete" -command {
     foreach i [$::cggui::fromCGFrame.anneal.par.multi.list curselection] {
        $::cggui::fromCGFrame.anneal.par.multi.list delete $i
     }
  }
  pack $fromCGFrame.anneal.par.multi.buttons.add \
              $fromCGFrame.anneal.par.multi.buttons.delete -expand 1 -fill x
  pack $fromCGFrame.anneal.par.multi.list -side left  -fill x -expand 1
  pack $fromCGFrame.anneal.par.multi.scroll \
                      $fromCGFrame.anneal.par.multi.buttons -side left \
                      -fill y -expand 1
  pack $fromCGFrame.anneal.par.multi -expand 1 -fill x
  incr anRow
  # Parameter files ----------------- END

  grid [label $fromCGFrame.anneal.psfLabel -text "PSF Filename: "] \
    -row $anRow -column 0 -sticky w
  grid [entry $fromCGFrame.anneal.psfFile -width 30 \
                             -textvariable ::cggui::annealPSFFile ] \
      -row $anRow -column 1 -sticky ew
  grid [button $fromCGFrame.anneal.psffilebutton -text "Browse" \
        -command {
           set tempfile [tk_getOpenFile]
           if {![string equal $tempfile ""]} { 
              set ::cggui::annealPSFFile $tempfile 
           }
        }] -row $anRow -column 2 -sticky w
  incr anRow




#  radiobutton  $fromCGFrame.anneal.remote -text "Anneal remotely (BioCoRE)" \
#                                    -value "0" \
#                                    -variable [namespace current]::annealChoice
#
#  radiobutton  $fromCGFrame.anneal.local -text "Anneal locally" \
#                                    -value "1" \
#                                    -variable [namespace current]::annealChoice
#
#  radiobutton  $fromCGFrame.anneal.dont -text "Don't Anneal" \
#                                    -value "2" \
#                                    -variable [namespace current]::annealChoice
#
#  checkbutton $fromCGFrame.anneal.prepInput -text "(But Prepare NAMD Input File)" \
#                                    -variable [namespace current]::annealInFile
#
#  grid $fromCGFrame.anneal.remote -row 0 -column 0 -columnspan 2 -sticky w
#  grid $fromCGFrame.anneal.local -row 1 -column 0 -columnspan 2 -sticky w
#  grid $fromCGFrame.anneal.dont -row 2 -column 0 -sticky w
#  grid $fromCGFrame.anneal.prepInput -row 2 -column 1 -sticky e

#  pack $fromCGFrame.anneal.remote -fill x -expand 1
#  pack $fromCGFrame.anneal.local -fill x -expand 1
#  pack $fromCGFrame.anneal.dont -side left -fill x -expand 1
#  pack $fromCGFrame.anneal.prepInput -side right -fill x -expand 1

  # -----------------------------------------------
  # -----------------------------------------------
  grid [button $fromCGFrame.back -text "Back To Previous Screen" \
        -command { \
            pack forget $::cggui::fromCGFrame   
            set ::cggui::menuChoice 0
            wm title $::cggui::w "CG Builder - Method Selection"
            pack $::cggui::chooseFrame }] \
        -row $row -column 0 -sticky w

  grid [button $fromCGFrame.applyDB -text "Reverse CG" \
        -command ::cggui::buildAAModelExecute] \
        -row $row -column 1 -columnspan 2 -sticky e
  incr row

## ---------------------- END    from CG rep FRAME ---------------------------
## ---------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## ---------------------- START map shape-based CG FRAME -----------------
  set mapSBFrame [frame $w.mapSBFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $mapSBFrame.introText -text "Map Shaped-Based CG To All Atom"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row
  grid [label $mapSBFrame.introTextDesc -text \
          "Map a SBCG Model to Domains of an All Atom Model"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  # -----------------------------------------------
  # CG molecule chooser
  #puts "before grid"
  grid [label $mapSBFrame.cgmollable -text "Coarse-Grained Molecule: "] \
    -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $mapSBFrame.cgmol -textvar [namespace current]::mapcgMenuText\
    -menu $mapSBFrame.cgmol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"
  set mapCGMenu [menu $mapSBFrame.cgmol.menu -tearoff no]
  incr row
  fill_mol_menu mapCGMenu ::cggui::isMapCGMolLoaded \
                                    currentMapCGMol 

  # -----------------------------------------------
  # Reference molecule chooser
  #puts "before grid"
  grid [label $mapSBFrame.refmollable -text "Reference Molecule: "] \
    -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $mapSBFrame.refmol -textvar [namespace current]::maprefMenuText\
    -menu $mapSBFrame.refmol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"
  set mapRefMenu [menu $mapSBFrame.refmol.menu -tearoff no]
  incr row
  fill_mol_menu mapRefMenu ::cggui::isMapRefMolLoaded \
                                    currentMapRefMol

  # -----------------------------------------------
  # All atom molecule chooser
  #puts "before grid"
  grid [label $mapSBFrame.aamollable -text "All-Atom Molecule To Map Onto: "] \
    -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $mapSBFrame.aamol -textvar [namespace current]::mapaaMenuText\
    -menu $mapSBFrame.aamol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"
  set mapAAMenu [menu $mapSBFrame.aamol.menu -tearoff no]
  incr row
  fill_mol_menu mapAAMenu ::cggui::isMapAAMolLoaded \
                                    currentMapAAMol

  # Output filenames
  grid [label $mapSBFrame.outpdblabel -text "Output PDB Filename: "] \
    -row $row -column 0 -sticky w
  grid [entry $mapSBFrame.mapOutpdbfile -width 30 -textvariable ::cggui::mapOutpdbfile] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # -----------------------------------------------
  # -----------------------------------------------
  grid [button $mapSBFrame.back -text "Back To Previous Screen" \
        -command { \
            pack forget $::cggui::mapSBFrame   
            set ::cggui::menuChoice 0
            wm title $::cggui::w "CG Builder - Method Selection"
            pack $::cggui::chooseFrame }] \
        -row $row -column 0 -sticky w

  grid [button $mapSBFrame.applyDB -text "Map Model" \
        -command ::cggui::mapSBExecute] \
        -row $row -column 1 -columnspan 2 -sticky e
  incr row

## ---------------------- END map shape-based CG FRAME -----------------
## ---------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## ---------------------- START baFromSim FRAME -----------------
  set baFromSimFrame [frame $w.baFromSimFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $baFromSimFrame.introText -text "Extract Bond/Angle Params of a CG Molecule"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row
  grid [label $baFromSimFrame.introTextDesc -text \
          "from the All Atom Model Simulation"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  # Input CG PSF filename
  grid [label $baFromSimFrame.infile1label -text "CG PSF File: "] \
    -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimInfile1 -width 30 -textvariable ::cggui::baFromSimCGpsfFile ] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # Input CG PDB filename
  grid [label $baFromSimFrame.infile2label -text "CG PDB File: "] \
    -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimInfile2 -width 30 -textvariable ::cggui::baFromSimCGpdbFile ] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # Input AA PDB filename
  grid [label $baFromSimFrame.infile3label -text "AA PDB File (beta values CG bead IDs): "] \
    -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimInfile3 -width 30 -textvariable ::cggui::baFromSimAApdbFile ] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # Input AA DCD filename
  grid [label $baFromSimFrame.infile4label -text "AA DCD File: "] \
    -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimInfile4 -width 30 -textvariable ::cggui::baFromSimAAdcdFile ] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # simulation temperature
  grid [label $baFromSimFrame.inTemplabel -text "Simulation Temperature: "] \
    -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimInTemp -width 30 -textvariable ::cggui::baFromSimInTemp] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # Output parameter filename
  grid [label $baFromSimFrame.outparlabel -text "Parameter output filename: "] \
       -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimOutparfile -width 30 -textvariable ::cggui::baFromSimOutparfile] \
       -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # Output bond parameter filename
  grid [label $baFromSimFrame.outbondparlabel -text "Bond Data output filename: "] \
       -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimOutbondparfile -width 30 -textvariable ::cggui::baFromSimOutbondparfile] \
       -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # Output angle parameter filename
  grid [label $baFromSimFrame.outangleparlabel -text "Angle Data output filename: "] \
       -row $row -column 0 -sticky w
  grid [entry $baFromSimFrame.baFromSimOutangleparfile -width 30 -textvariable ::cggui::baFromSimOutangleparfile] \
       -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # -----------------------------------------------
  # -----------------------------------------------
  grid [button $baFromSimFrame.back -text "Back To Previous Screen" \
        -command { \
            pack forget $::cggui::baFromSimFrame   
            set ::cggui::menuChoice 0
            wm title $::cggui::w "CG Builder - Method Selection"
            pack $::cggui::chooseFrame }] \
        -row $row -column 0 -sticky w

  grid [button $baFromSimFrame.applyDB -text "Extract Parameters" \
        -command ::cggui::baFromSimExecute] \
        -row $row -column 1 -columnspan 2 -sticky e
  incr row

## ---------------------- END baFromSim FRAME -----------------
## ---------------------------------------------------------------------------

## ---------------------- START scaleBAFrame -----------------
  set scaleBAFrame [frame $w.scaleBAFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $scaleBAFrame.introText -text "Scale Bond/Angle Spring Constants in a Parameter"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  grid [label $scaleBAFrame.introTextDesc -text \
          "File That Are Above A Cutoff"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  # Input Parameter filename
  grid [label $scaleBAFrame.infile1label -text "Input Parameter File: "] \
    -row $row -column 0 -sticky w
  grid [entry $scaleBAFrame.scaleBAInfile1 -width 30 -textvariable ::cggui::scaleBAinFile ] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # bond scale value 
  grid [label $scaleBAFrame.inBondScalelabel -text "Bond Spring Constant Scaling: "] \
    -row $row -column 0 -sticky w
  grid [entry $scaleBAFrame.inBondScale -width 30 -textvariable ::cggui::scaleBAbondScale] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # bond scale cutoff value 
  grid [label $scaleBAFrame.inBondCutofflabel -text "Bond Spring Constant Cutoff, kcal/(mol A**2): "] \
    -row $row -column 0 -sticky w
  grid [entry $scaleBAFrame.inBondCutoff -width 30 -textvariable ::cggui::scaleBAbondCutoff] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # angle scale value 
  grid [label $scaleBAFrame.inAngleScalelabel -text "Angle Spring Constant Scaling: "] \
    -row $row -column 0 -sticky w
  grid [entry $scaleBAFrame.inAngleScale -width 30 -textvariable ::cggui::scaleBAangleScale] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # angle scale cutoff value 
  grid [label $scaleBAFrame.inAngleCutofflabel -text "Angle Spring Constant Cutoff, kcal/(mol rad**2) : "] \
    -row $row -column 0 -sticky w
  grid [entry $scaleBAFrame.inAngleCutoff -width 30 -textvariable ::cggui::scaleBAangleCutoff] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # Output Parameter filename
  grid [label $scaleBAFrame.outfile1label -text "Output Parameter File: "] \
    -row $row -column 0 -sticky w
  grid [entry $scaleBAFrame.scaleBAOutfile1 -width 30 -textvariable ::cggui::scaleBAoutFile ] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row





  # -----------------------------------------------
  # -----------------------------------------------
  grid [button $scaleBAFrame.back -text "Back To Previous Screen" \
        -command { \
            pack forget $::cggui::scaleBAFrame   
            set ::cggui::menuChoice 0
            wm title $::cggui::w "CG Builder - Method Selection"
            pack $::cggui::chooseFrame }] \
        -row $row -column 0 -sticky w

  grid [button $scaleBAFrame.applyDB -text "Scale Values" \
        -command ::cggui::scaleBAExecute] \
        -row $row -column 1 -columnspan 2 -sticky e
  incr row

## ---------------------- END scaleBAFrame -----------------
## ---------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## ---------------------- START assign LJ Params FRAME -----------------
  set ljParamFrame [frame $w.ljParamFrame]
  set row 0

  #puts "before intro text"
  # intro text
  grid [label $ljParamFrame.introText -text "Assign Lennard-Jones Params For"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row
  grid [label $ljParamFrame.introTextDesc -text \
          "A CG Model based on the All Atom Model"] \
     -row $row -column 0 -columnspan 2 -sticky w
  incr row

  # Input parameter filename
  grid [label $ljParamFrame.inparlabel -text "Original CG Param File: "] \
    -row $row -column 0 -sticky w
  grid [entry $ljParamFrame.ljparamInparfile -width 30 -textvariable ::cggui::ljparamInparfile] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # -----------------------------------------------
  # AA molecule chooser
  #puts "before grid"
  grid [label $ljParamFrame.aamollable -text "All Atom Structure: "] \
    -row $row -column 0 -sticky w
  #puts "before grid2"
  grid [menubutton $ljParamFrame.aamol -textvar [namespace current]::ljparamMenuText\
    -menu $ljParamFrame.aamol.menu -relief raised] \
    -row $row -column 1 -columnspan 4 -sticky w
  #puts "before menu"
  set ljparamMenu [menu $ljParamFrame.aamol.menu -tearoff no]
  incr row
  fill_mol_menu ljparamMenu ::cggui::isljparamMolLoaded \
                                    ljparamCurrentMol 

  # max energy for LJ well depth 
  grid [label $ljParamFrame.inEnergylabel -text "Max energy for LJ well depth (in kcal/mol): "] \
    -row $row -column 0 -sticky w
  grid [entry $ljParamFrame.ljparamInEnergy -width 30 -textvariable ::cggui::ljparamInEnergy] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # addition to LJ radius
  grid [label $ljParamFrame.inRadiuslabel -text "Addition to LJ radius (in A): "] \
    -row $row -column 0 -sticky w
  grid [entry $ljParamFrame.ljparamInRadius  -width 30 -textvariable ::cggui::ljparamInRadius] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row


  # Output filename
  grid [label $ljParamFrame.outparlabel -text "Output Parameter filename: "] \
    -row $row -column 0 -sticky w
  grid [entry $ljParamFrame.ljparamOutparfile -width 30 -textvariable ::cggui::ljparamOutparfile] \
    -row $row -column 1 -columnspan 2 -sticky ew
  incr row

  # -----------------------------------------------
  # -----------------------------------------------
  grid [button $ljParamFrame.back -text "Back To Previous Screen" \
        -command { \
            pack forget $::cggui::ljParamFrame   
            set ::cggui::menuChoice 0
            wm title $::cggui::w "CG Builder - Method Selection"
            pack $::cggui::chooseFrame }] \
        -row $row -column 0 -sticky w

  grid [button $ljParamFrame.applyDB -text "Assign Parameters" \
        -command ::cggui::ljParamExecute] \
        -row $row -column 1 -columnspan 2 -sticky e
  incr row

## ---------------------- END assign LJ Params FRAME -----------------
## ---------------------------------------------------------------------------

#puts "after doing FROM frame"
  #pack [label $w.revcglabel -text "Test Text "] 

  pack $chooseFrame 

  # this trace lets the plugin determine when you have loaded a molecule
  # in VMD
  trace add variable ::vmd_initialize_structure write \
    ::cggui::vmd_init_struct_trace

}

# -------------------------------------------------------------------------
proc ::cggui::buildResidueCGExecute {} {
   variable autopsf
   variable solvate
   variable ionize

   if {[llength $::cgtools::convbeads] == 0} {
            tk_messageBox -type ok -message "No bead definitions loaded.
Need to load database file." -title "Error!"
      return
   }

   if {$::cggui::isToCGMolLoaded == 0} {
      tk_messageBox -type ok -title "Error!" \
                   -message "Need to load and select a valid molecule." 
      return
   }
   if {[string trim $::cggui::toCGoutpdbfile] == ""} {
      tk_messageBox -message "Need to specify an output PDB filename." \
                   -type ok -title "Error!"
      return
   }

   if {[string trim $::cggui::revcgfile] == ""} {
      tk_messageBox -message "Need to specify an output reverse 
coarse grain filename." \
                   -type ok -title "Error!"
      return
   }

   ::cgtools::apply_database $::cggui::currentToCGMol \
                             $::cggui::toCGoutpdbfile \
                             $::cggui::revcgfile

   set molid [mol new "$::cggui::toCGoutpdbfile"]

   # Add a rep, and find the rep number of our newly added rep
   set vdwrepindex 0
   while  { ! [catch {mol repname $currentMol} ] } {
      incr vdwrepindex
   }
   incr vdwrepindex
   mol addrep $molid

   mol modstyle $vdwrepindex $molid {VDW 2.0}
   mol modmaterial $vdwrepindex $molid {Transparent}

   # run autopsf?
   if { $autopsf } { }

   # handle solvate/ionize

   if { $solvate } {
      solvate ${basename}.psf $::cggui::toCGoutpdbfile -o ${basename} -t 12
   }

}

# -------------------------------------------------------------------------
proc ::cggui::buildShapeCGExecute {} {

   # do a bunch of checking on the input values provided


# some things only need to be checked if we are doing a molecule...
   if { $cggui::shapeChoice == "0" } {
      # molecule

      if {$::cggui::isToCGMolLoaded == 0} {
         tk_messageBox -type ok \
                      -message "Need to load and select a valid molecule." \
                      -title "Error!"
         return
      }

      if {[string trim $::cggui::toCGoutallpdbfile] == ""} {
         tk_messageBox -message \
                 "Need to specify an output All-Atom Reference PDB filename." \
                 -type ok -title "Error!"
         return
      }

   } elseif { $cggui::shapeChoice == "1" } {
      # EDM
      if {[string trim $::cggui::toCGshapeFracCutoff] == "" || \
          ! [string is double $::cggui::toCGshapeFracCutoff] || \
          $::cggui::toCGshapeFracCutoff < 0.0 || \
          $::cggui::toCGshapeFracCutoff >= 1.0 } {
         tk_messageBox -message "Frac Cutoff needs to be 0 <= Cutoff < 1." \
                      -type ok -title "Error!"
         return
      }

      if {[string trim $::cggui::edmPath] == ""} {
         tk_messageBox -message \
                 "Need to specify a path to a SITUS or .dx file." \
                 -type ok -title "Error!"
         return
      }
   }

   set massValue -1
   if { $::cggui::toCGShapeUseMass == 1 } {
      # need to have a positive value for the mass
      if {[string trim $::cggui::toCGshapeMass] == "" || \
          ! [string is double $::cggui::toCGshapeMass] || \
          $::cggui::toCGshapeMass < 0 } {
         tk_messageBox -message "Total Mass needs to be a positive number." \
                   -type ok -title "Error!"
         return
      }
      set massValue $::cggui::toCGshapeMass
   } 

   if {[string trim $::cggui::toCGshapeNumCGBeads] == "" || \
       ! [string is integer $::cggui::toCGshapeNumCGBeads] || \
       $::cggui::toCGshapeNumCGBeads < 1 } {
      tk_messageBox -type ok -title "Error!" \
              -message "Number of CG beads needs to be a positive integer." 
      return
   }

   if {[string trim $::cggui::toCGshapeNumSteps] == "" || \
       ! [string is integer $::cggui::toCGshapeNumSteps] || \
       $::cggui::toCGshapeNumSteps < 1 } {
      tk_messageBox -message "Number of steps needs to be a positive integer." \
                   -type ok -title "Error!"
      return
   }

   if {[string trim $::cggui::toCGshapeEpsInit] == "" || \
       ! [string is double $::cggui::toCGshapeEpsInit] || \
       $::cggui::toCGshapeEpsInit < 0 } {
      tk_messageBox -message "Initial eps needs to be a positive number." \
                   -type ok -title "Error!"
      return
   }

   if {[string trim $::cggui::toCGshapeEpsFinal] == "" || \
       ! [string is double $::cggui::toCGshapeEpsFinal] || \
       $::cggui::toCGshapeEpsFinal < 0 } {
      tk_messageBox -message "Final eps needs to be a positive number." \
                   -type ok -title "Error!"
      return
   }

   if {[string trim $::cggui::toCGshapeLambdaInit] == "" || \
       ! [string is double $::cggui::toCGshapeLambdaInit] || \
       $::cggui::toCGshapeLambdaInit < 0 } {
      tk_messageBox -message "Initial Lambda needs to be a positive number." \
                   -type ok -title "Error!"
      return
   }

   if {[string trim $::cggui::toCGshapeLambdaFinal] == "" || \
       ! [string is double $::cggui::toCGshapeLambdaFinal] || \
       $::cggui::toCGshapeLambdaFinal < 0 } {
      tk_messageBox -message "Final Lambda needs to be a positive number." \
                   -type ok -title "Error!"
      return
   }

   if {[string trim $::cggui::toCGshapeBondCutoff] == "" || \
       ! [string is double $::cggui::toCGshapeBondCutoff] || \
       $::cggui::toCGshapeBondCutoff < 0 } {
      tk_messageBox -message "Bond Cutoff needs to be a positive number." \
                   -type ok -title "Error!"
      return
   }

   if {[string trim $::cggui::toCGshapeResName] == "" || \
   [string length [string trim $::cggui::toCGshapeResName]] > 3 } {
      tk_messageBox -type ok -title "Error!" \
                -message "CG Residue Name needs to be 3 or fewer characters." 
      return
   }

   if {[string trim $::cggui::toCGshapeNamePrefix] == "" || \
   [string length [string trim $::cggui::toCGshapeNamePrefix]] > 1 } {
      tk_messageBox -type ok -title "Error!" \
                -message "CG Name Prefix needs to be a single character." 
      return
   }

   if {[string trim $::cggui::toCGoutpdbfile] == ""} {
      tk_messageBox -message "Need to specify an output PDB filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::toCGouttopfile] == ""} {
      tk_messageBox -message "Need to specify an output topology filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::toCGoutparmfile] == ""} {
      tk_messageBox -message "Need to specify an output parameter filename." \
                   -type ok -title "Error!"
      return
   }

   if { $cggui::shapeChoice == "0" } {
      # molecule

      # let's warn the user if they haven't provided a PSF file.
      set ok 0
      foreach i [join [molinfo $::cggui::currentToCGMol get filetype]] {
         if {($i=="psf") || ($i=="parm") || ($i=="parm7")} {
            set ok 1
         }
      }

      if { $ok == 0 } {
         tk_messageBox -message \
"No PSF file loaded.  I'll go 
ahead and do the coarse graining, 
but you might want to run again
after loading a PSF file." \
                   -type ok -title "Warning!"
      }

      ::cgnetworking::networkCGMolecule ::cggui::addStatusLine \
          $::cggui::currentToCGMol \
          $::cggui::toCGshapeResName \
          $::cggui::toCGshapeNamePrefix \
          $::cggui::toCGoutpdbfile $::cggui::toCGoutallpdbfile \
          $::cggui::toCGouttopfile $::cggui::toCGoutparmfile \
          $::cggui::toCGshapeNumCGBeads \
          $::cggui::toCGshapeNumSteps \
          $::cggui::toCGshapeEpsInit $::cggui::toCGshapeEpsFinal \
          $::cggui::toCGshapeLambdaInit $::cggui::toCGshapeLambdaFinal \
          $::cggui::toCGshapeBondMethod $::cggui::toCGshapeBondCutoff  \
          $massValue

   } elseif { $cggui::shapeChoice == "1" } {
      # EDM
      ::cgnetworking::networkCGEDM ::cggui::addStatusLine \
          $::cggui::edmPath \
          $::cggui::toCGshapeResName \
          $::cggui::toCGshapeNamePrefix \
          $::cggui::toCGoutpdbfile \
          $::cggui::toCGouttopfile $::cggui::toCGoutparmfile \
          $::cggui::toCGshapeNumCGBeads \
          $::cggui::toCGshapeNumSteps \
          $::cggui::toCGshapeEpsInit $::cggui::toCGshapeEpsFinal \
          $::cggui::toCGshapeLambdaInit $::cggui::toCGshapeLambdaFinal \
          $::cggui::toCGshapeBondMethod $::cggui::toCGshapeBondCutoff \
          $::cggui::toCGshapeFracCutoff $massValue
   }

#   puts "getting ready to mol new $::cggui::toCGoutpdbfile"
   set molid [mol new "$::cggui::toCGoutpdbfile"]

   # Add a rep, and find the rep number of our newly added rep
   set vdwrepindex 0
   while  { ! [catch {mol repname $currentMol} ] } {
      incr vdwrepindex
   }
   incr vdwrepindex
   mol addrep $molid

   mol modstyle $vdwrepindex $molid {VDW 3.0}

   # IF  we started with a molecule, make the CG transparent
   # XXX: should eventually make these options settable so that the
   # output can be determined by the user.  Would be good to have
   # an 'output options' button that can be clicked to set these 
   # settings
#   if { $cggui::shapeChoice == "0" } {
#      mol modmaterial $vdwrepindex $molid {Transparent}
#   }
#   mol representation { VDW 1.0 } 
#   mol modrep 0 $molid

}


# -------------------------------------------------------------------------
proc ::cggui::addStatusLine {strText} {
   variable toCGShapeFrame
   ::cggui::addGenericStatusLine $toCGShapeFrame $strText
}

# -------------------------------------------------------------------------
proc ::cggui::addGenericStatusLine {frameName strText} {
   $frameName.statFrame.statBox configure -state normal
   $frameName.statFrame.statBox insert end "$strText\n"
   $frameName.statFrame.statBox see end
   update idletask
   $frameName.statFrame.statBox configure -state disabled
}


# -------------------------------------------------------------------------
# do reverse CG
proc ::cggui::buildAAModelExecute {} {
   if {$::cggui::isToCGMolLoaded == 0} {
      tk_messageBox -type ok \
                   -message "Need to load and select a valid CG molecule." \
                   -title "Error!"
      return
   }
   if {$::cggui::fromCGoutpdbfile == ""} {
      tk_messageBox -type ok \
                   -message "Need to specify an output PDB filename." \
                   -title "Error!"
      return
   }
   if {$::cggui::revcgfile == ""} {
      tk_messageBox -type ok \
                   -message "Need to specify an output reverse
coarse grain file." \
                   -title "Error!"
      return
   }

   # do any checking of values
   if { $::cggui::annealConfig } {
      if {[string trim $::cggui::annealParList] == ""} {
         tk_messageBox -type ok \
                   -message "Need to specify at least one parameter file." \
                   -title "Error!"
         return
      }
      if {$::cggui::annealPSFFile == ""} {
         tk_messageBox -type ok \
                   -message "Need to specify a PSF file for the annealing." \
                   -title "Error!"
         return
      }

   }

   ::cgtools::apply_reversal $::cggui::currentFromCGMol \
                             $::cggui::revcgfile \
                             $::cggui::currentFromCGAAMol \
                             $::cggui::fromCGoutpdbfile  \
                             $::cggui::do_rotations

   set molid [mol new "$::cggui::fromCGoutpdbfile"]

#   # Add a rep, and find the rep number of our newly added rep
#   set vdwrepindex 0
#   while  { ! [catch {mol repname $currentMol} ] } {
#      incr vdwrepindex
#   }
#   incr vdwrepindex
#   mol addrep $molid
#
#   mol modstyle $vdwrepindex $molid {VDW 2.0}



   # write the annealing config file for NAMD?

#   if { $::cggui::annealInFile || $::cggui::annealChoice != "2" } {}
   if { $::cggui::annealConfig} {

      ::cgtools::make_anneal_config $::cggui::fromCGoutpdbfile \
                    $::cggui::annealPSFFile \
                    $::cggui::annealParList \
                    "out.namd" "" 0 
   }

}

# -------------------------------------------------------------------------
proc ::cggui::mapSBExecute {} {
   if {$::cggui::isMapCGMolLoaded == 0} {
      tk_messageBox -type ok -title "Error!" \
                   -message "Need to load and select a valid CG Molecule." 
      return
   }
   if {$::cggui::isMapAAMolLoaded == 0} {
      tk_messageBox -type ok -title "Error!" \
                  -message "Need to load and select a valid All Atom Molecule." 
      return
   }
   if {$::cggui::isMapRefMolLoaded == 0} {
      tk_messageBox -type ok -title "Error!" \
                 -message "Need to load and select a valid Reference Molecule." 
      return
   }
   if {[string trim $::cggui::mapOutpdbfile] == ""} {
      tk_messageBox -message "Need to specify an output PDB filename." \
                   -type ok -title "Error!"
      return
   }

   ::cgtools::mapCGMolecule foobar \
                            $::cggui::currentMapAAMol \
                            $::cggui::currentMapCGMol \
                            $::cggui::currentMapRefMol \
                            $::cggui::mapOutpdbfile

   set molid [mol new "$::cggui::mapOutpdbfile"]

   # Add a rep, and find the rep number of our newly added rep
   set vdwrepindex 0
   while  { ! [catch {mol repname $currentMol} ] } {
      incr vdwrepindex
   }
   incr vdwrepindex
   mol addrep $molid

   mol modstyle $vdwrepindex $molid {VDW 2.0}
   mol modmaterial $vdwrepindex $molid {Transparent}
}

# -------------------------------------------------------------------------
proc ::cggui::baFromSimExecute {} {
   if {[string trim $::cggui::baFromSimCGpsfFile] == ""} {
      tk_messageBox -message "Need to specify a CG PSF filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::baFromSimCGpdbFile] == ""} {
      tk_messageBox -message "Need to specify a CG PDB filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::baFromSimAApdbFile] == ""} {
      tk_messageBox -message "Need to specify an AA PDB filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::baFromSimAAdcdFile] == ""} {
      tk_messageBox -message "Need to specify an AA DCD filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::baFromSimInTemp] == ""} {
      tk_messageBox -message "Need to specify a simulation temperature." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::baFromSimOutparfile] == ""} {
      tk_messageBox -message "Need to specify an output parameter filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::baFromSimOutbondparfile] == ""} {
      tk_messageBox -message "Need to specify an output bond parameter filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::baFromSimOutangleparfile] == ""} {
      tk_messageBox -message "Need to specify an output angle parameter filename." \
                   -type ok -title "Error!"
      return
   }

   ::cgnetworking::all_ba ::cggui::addStatusLine \
                            [string trim $::cggui::baFromSimCGpsfFile]  \
                            [string trim $::cggui::baFromSimCGpdbFile]  \
                            [string trim $::cggui::baFromSimAApdbFile]  \
                            [string trim $::cggui::baFromSimAAdcdFile]  \
                            [string trim $::cggui::baFromSimInTemp] \
                            [string trim $::cggui::baFromSimOutparfile] \
                            [string trim $::cggui::baFromSimOutbondparfile] \
                            [string trim $::cggui::baFromSimOutangleparfile] 

}

# -------------------------------------------------------------------------
proc ::cggui::scaleBAExecute {} {
   if {[string trim $::cggui::scaleBAinFile] == ""} {
      tk_messageBox -message "Need to specify an input parameter filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::scaleBAoutFile] == ""} {
      tk_messageBox -message "Need to specify an output parameter filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::scaleBAbondScale] == ""} {
      tk_messageBox -message "Need to specify a bond scaling factor." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::scaleBAbondCutoff] == ""} {
      tk_messageBox -message "Need to specify a bond scaling cutoff." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::scaleBAangleScale] == ""} {
      tk_messageBox -message "Need to specify an angle scaling factor." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::scaleBAangleCutoff] == ""} {
      tk_messageBox -message "Need to specify an angle scaling cutoff." \
                   -type ok -title "Error!"
      return
   }

   ::cgnetworking::scaleParameterConstants ::cggui::addStatusLine \
                            [string trim $::cggui::scaleBAinFile]  \
                            [string trim $::cggui::scaleBAbondScale]  \
                            [string trim $::cggui::scaleBAbondCutoff]  \
                            [string trim $::cggui::scaleBAangleScale]  \
                            [string trim $::cggui::scaleBAangleCutoff] \
                            [string trim $::cggui::scaleBAoutFile] 

}

# -------------------------------------------------------------------------
proc ::cggui::ljParamExecute {} {
   if {$::cggui::isljparamMolLoaded == 0} {
      tk_messageBox -type ok -title "Error!" \
                  -message "Need to load and select a valid All-Atom Molecule." 
      return
   }
   if {[string trim $::cggui::ljparamOutparfile] == ""} {
      tk_messageBox -message "Need to specify an output parameter filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::ljparamInparfile] == ""} {
      tk_messageBox -message "Need to specify an input parameter filename." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::ljparamInEnergy] == ""} {
      tk_messageBox -message "Need to specify a maximum energy." \
                   -type ok -title "Error!"
      return
   }
   if {[string trim $::cggui::ljparamInRadius] == ""} {
      tk_messageBox -message "Need to specify a radius." \
                   -type ok -title "Error!"
      return
   }

   ::cgtools::sasa_LJ_networking ::cggui::addStatusLine \
                            $::cggui::ljparamInparfile  \
                            $::cggui::ljparamCurrentMol \
                            $::cggui::ljparamOutparfile \
                            $::cggui::ljparamInEnergy \
                            $::cggui::ljparamInRadius


}

# -------------------------------------------------------------------------
proc ::cggui::vmd_init_struct_trace {structure index op} {
  #puts "in vmd_init_struct:  struct: '$structure', index: '$index', op: '$op'"
  variable toResCGMenu
  variable toShapeCGMenu

  variable fromMolMenu
  variable fromAAMolMenu

  variable mapAAMenu
  variable mapCGMenu
  variable mapRefMenu

  variable ljparamMenu

  #Accessory proc for traces on the mol menu
  fill_mol_menu toResCGMenu ::cggui::isToCGMolLoaded currentToCGMol
  fill_mol_menu toShapeCGMenu ::cggui::isToCGMolLoaded currentToCGMol
  fill_mol_menu fromMolMenu ::cggui::isFromCGMolLoaded currentFromCGMol
  fill_mol_menu fromAAMolMenu ::cggui::isFromCGMolLoaded \
                                    currentFromCGAAMol

  fill_mol_menu mapCGMenu ::cggui::isMapCGMolLoaded currentMapCGMol
  fill_mol_menu mapRefMenu ::cggui::isMapRefMolLoaded currentMapRefMol
  fill_mol_menu mapAAMenu ::cggui::isMapAAMolLoaded currentMapAAMol

  fill_mol_menu ljparamMenu ::cggui::isljparamMolLoaded ljparamCurrentMol
}

# -------------------------------------------------------------------------
proc cggui_tk {} {
#   puts "before invoking the constructor";
  ::cggui::cggui
#   puts "after invoking the constructor";
  return $::cggui::w
}

# -------------------------------------------------------------------------
#Proc to get all the current molecules for a menu
#For now, shamefully ripped off from the NAMDEnergy plugin, which
#shamelessly ripped off the PME plugin
proc ::cggui::fill_mol_menu {menuName2 isloaded currentMolecule} {
  #puts "fill_mol_menu: start. $menuName2 $isloaded $[namespace current]::$currentMolecule"
  upvar 1 $menuName2 menuName
  upvar 1 $isloaded isMolLoaded
  variable nullMolString

#  puts "$menuName"

  # do we need to redraw the currently selected molecule text?  We only want to
  # do this if the user is just loading in the first molecule.  We can test this
  # by seeing what elements are already in the menu.
  set redo [expr {[$menuName index end] == "none"}]
#  puts "end index is [$menuName index end].  redo is $redo"

  $menuName delete 0 end

  set molList ""
  #puts "fill_mol_menu: before foreach"
  foreach mm [array names ::vmd_initialize_structure] {
  #puts "fill_mol_menu: before if"
    if { $::vmd_initialize_structure($mm) != 0} {
      lappend molList $mm
      #puts "adding $mm [molinfo $mm get name]"
      $menuName add radiobutton -variable \
                                       [namespace current]::$currentMolecule \
                           -value $mm -label "$mm [molinfo $mm get name]"
    }
  }


#set if any non-Graphics molecule is loaded
  if {[lsearch -exact $molList $currentMolecule] == -1} {
    if {[lsearch -exact $molList [molinfo top]] != -1} {
#  puts "before if.  end is '[$menuName index end]'"
       if {$::cggui::shapeChoice == "0" && $redo == 1} {
#  puts "in if"
          set [namespace current]::$currentMolecule [molinfo top]
          set isMolLoaded 1
       }
    } else {
      set [namespace current]::$currentMolecule $nullMolString
      set isMolLoaded  0
    }
  }
#  puts "near end of fill_mol_menu"
}

# -------------------------------------------------------------------------
proc ::cggui::setEDMValues {} {

   # let's open the DX file and grab a few values out of it...
   set line ""
   if [catch {set channel [open $::cggui::edmPath r]}] {
      addStatusLine "Error: opening input file ($::cggui::edmPath) failed"
      return
   }
#   addStatusLine "Reading DX file..."
   set line_1 "#"
   while {$line_1 == "#"} {
      gets $channel line
      set line_1 [lindex $line 0]
   }

   set fileType [ ::cgnetworking::getFileType $::cggui::edmPath ]

   if { $fileType == "DX" } {
      set Nx [lindex $line 5]
      set Ny [lindex $line 6]
      set Nz [lindex $line 7]
   } elseif { $fileType == "SITUS" } {
      set Nx [lindex $line 4]
      set Ny [lindex $line 5]
      set Nz [lindex $line 6]
   } else {
      addStatusLine "Unknown File Type!!!"
      return
   }

   close $channel

   if { [string trim $Nx] == "" || \
      ! [string is integer $Nx] || \
        [string trim $Ny] == "" || \
      ! [string is integer $Ny] || \
        [string trim $Nz] == "" || \
      ! [string is integer $Nz] } {
      addStatusLine "WARNING!!! DX File doesn't seem to be formatted properly!"
   } else {
      set numPoints [expr {int ($Nx * $Ny * $Nz / 3)}]

      # now we have some values we can play with
      if { $fileType == "DX" } {
         set ::cggui::toCGshapeNumCGBeads [expr \
                             {int( $numPoints / $::cggui::kAtomsPerDXPoint) } ]
      } elseif { $fileType == "SITUS" } {
         set ::cggui::toCGshapeNumCGBeads [expr \
                          {int( $numPoints / $::cggui::kAtomsPerSitusPoint) } ]
      }
      if { $::cggui::toCGshapeNumCGBeads < 1 } {
          set ::cggui::toCGshapeNumCGBeads 1
      }
      set ::cggui::toCGshapeNumSteps [expr \
                  {$::cggui::kStepsPerBead * $::cggui::toCGshapeNumCGBeads } ]
      set ::cggui::toCGshapeLambdaInit [expr \
                    {$::cggui::kLambdaMult * $::cggui::toCGshapeNumCGBeads } ]
   }

   # let's get the base name.  relative and no extension, with prefixes
   # stripped off
   set rootName [string trimleft [file tail [file rootname $::cggui::edmPath]] \
                                                     $::cggui::outprefix] ;  
#   set rootName [file rootname $::cggui::edmPath]
   set [namespace current]::toCGoutpdbfile $rootName.pdb
   set [namespace current]::toCGouttopfile $rootName.top
   set [namespace current]::toCGoutparmfile $rootName.par

}

# -------------------------------------------------------------------------
proc ::cggui::tomolmenuaux {mol index op} {
   #puts "in tomolmenuaux:  mol: '$mol', index: '$index', op: '$op'"
  #Accessory proc for the trace on currentToCGMol
  variable currentToCGMol
  variable tomolMenuText
  if { ! [catch { molinfo $currentToCGMol get name } name ] } {
#puts "tomolmenuaux: setting molMenuText to $currentToCGMol: $name"
     set tomolMenuText "$currentToCGMol: $name"

     # let's get the base name.  relative and no extension, with prefixes
     # stripped off
     set shortFile [string trimleft [file tail [file rootname $name]]  \
                                                     $::cggui::outprefix] ;  
#puts "tomolmenuaux: shortfile $shortFile"

    variable toCGoutpdbfile "$::cggui::outprefix$shortFile.pdb"
    variable toCGoutallpdbfile "$::cggui::outfromprefix$shortFile.pdb"
    variable toCGouttopfile "$::cggui::outprefix$shortFile.top"
    variable toCGoutparmfile "$::cggui::outprefix$shortFile.par"
    variable revcgfile "$::cggui::outprefix$shortFile.rcg"

    # let's set some initial values
    set sel [atomselect $currentToCGMol "all" ]
    set ::cggui::toCGshapeNumCGBeads [expr {int(ceil([$sel num] / $::cggui::kAtomsPerBead)) } ]
#    if { $::cggui::toCGshapeNumCGBeads < 1 } {
#       set ::cggui::toCGshapeNumCGBeads 1
#    }
    set ::cggui::toCGshapeNumSteps [expr {$::cggui::kStepsPerBead * $::cggui::toCGshapeNumCGBeads } ]
    set ::cggui::toCGshapeLambdaInit [expr {$::cggui::kLambdaMult * $::cggui::toCGshapeNumCGBeads } ]
  } else { 
#puts "tomolmenuaux: setting molMenuText to $currentToCGMol"
set tomolMenuText "$currentToCGMol" }

}

# -------------------------------------------------------------------------
proc ::cggui::fromaamolmenuaux {mol index op} {
   #puts "in fromAAmolmenuaux:  mol: '$mol', index: '$index', op: '$op'"
#   upvar 1 $mol molVarName
  #Accessory proc for the trace on currentFromCGMol
  variable currentFromCGAAMol
  variable currentFromCGMol
  variable fromaamolMenuText
  if { ! [catch { molinfo $currentFromCGAAMol get name } name ] } {
#puts "fromAAmolmenuaux: setting molMenuText to $currentFromCGAAMol: $name"
    set fromaamolMenuText "$currentFromCGAAMol: $name"

    # let's get the base name.  relative and no extension, with prefixes
    # stripped off
    set shortFile [string trimleft [file tail [file rootname $name]]  \
                                                     $::cggui::outprefix] ;  

    variable fromCGoutpdbfile "$::cggui::outfromprefix$shortFile.pdb"

    variable revcgfile "$::cggui::outprefix$shortFile.rcg"

#    # now, let's set the revcgfile name based on the currently chosen
## cg molecule name
#    if { ! [catch { molinfo $currentFromCGMol get name } name ] } {
#       set shortFile [string trimleft [file tail [file rootname $name]]  \
#                                                     $::cggui::outprefix] ;  
#    }

    # let's see if this molecule has a PSF file loaded into it.
    set psfIndex [lsearch [lindex [molinfo $::cggui::currentFromCGAAMol \
                                                    get {filetype}] 0] "psf"]
    if { $psfIndex != -1 } {
       set ::cggui::annealPSFFile [lindex [lindex \
               [molinfo $::cggui::currentFromCGAAMol get {filename}] 0] \
               $psfIndex ]
    } else {
       set ::cggui::annealPSFFile ""
    }
  } else { 
  #puts "fromAAmolmenuaux: setting molMenuText to $currentFromCGAAMol"
  set fromaamolMenuText "$currentFromCGAAMol" }
}

# -------------------------------------------------------------------------
proc ::cggui::frommolmenuaux {mol index op} {
   #puts "in frommolmenuaux:  mol: '$mol', index: '$index', op: '$op'"
#   upvar 1 $mol molVarName
  #Accessory proc for the trace on currentFromCGMol
  variable currentFromCGMol
  variable frommolMenuText
  if { ! [catch { molinfo $currentFromCGMol get name } name ] } {
#puts "frommolmenuaux: setting molMenuText to $currentFromCGMol: $name"
    set frommolMenuText "$currentFromCGMol: $name"
    variable fromCGoutpdbfile "$::cggui::outfromprefix$name"
    
  } else { 
    #puts "frommolmenuaux: setting molMenuText to $currentFromCGMol"
    set frommolMenuText "$currentFromCGMol" 
  }
}

# -------------------------------------------------------------------------
proc ::cggui::mapcgmenuaux {mol index op} {
  variable currentMapCGMol
  variable mapcgMenuText
  if { ! [catch { molinfo $currentMapCGMol get name } name ] } {
    set mapcgMenuText "$currentMapCGMol: $name"
    variable fromCGoutpdbfile "$::cggui::outfromprefix$name"
    
  } else { 
    set mapcgMenuText "$currentMapCGMol" 
  }
}

# -------------------------------------------------------------------------
proc ::cggui::maprefmenuaux {mol index op} {
  variable currentMapRefMol
  variable maprefMenuText
  if { ! [catch { molinfo $currentMapRefMol get name } name ] } {
    set maprefMenuText "$currentMapRefMol: $name"
  } else { 
    set maprefMenuText "$currentMapRefMol" 
  }
}

# -------------------------------------------------------------------------
proc ::cggui::mapaamenuaux {mol index op} {
  variable currentMapAAMol
  variable mapaaMenuText
  if { ! [catch { molinfo $currentMapAAMol get name } name ] } {
    set mapaaMenuText "$currentMapAAMol: $name"
  } else { 
    set mapaaMenuText "$currentMapAAMol" 
  }
}

# -------------------------------------------------------------------------
proc ::cggui::ljparammenuaux {mol index op} {
  variable ljparamCurrentMol
  variable ljparamMenuText
  if { ! [catch { molinfo $ljparamCurrentMol get name } name ] } {
    set ljparamMenuText "$ljparamCurrentMol: $name"
  } else { 
    set ljparamMenuText "$ljparamCurrentMol" 
  }
}

# -------------------------------------------------------------------------

