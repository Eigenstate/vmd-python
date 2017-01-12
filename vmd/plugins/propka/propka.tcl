# Copyright (c) 2009-2012  Michal Rostkowski and Jan H. Jensen  
#
# $Id: propka.tcl,v 1.7 2016/10/21 15:02:18 johns Exp $
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#         * Redistributions of source code must retain the above
#         * copyright notice, this list of conditions and the
#         * following disclaimer.
#         * Redistributions in binary form must reproduce the
#         * above copyright notice, this list of conditions and the
#         * following disclaimer in the documentation and/or other
#         * materials provided with the distribution.
#         * Neither the name of the University of Copenhagen nor
#         * the names of its contributors may be used to endorse
#         * or promote products derived from this software
#         * without specific prior written permission.
#
#         THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#         CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#         INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#         MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#         DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
#         BE LIABLE FOR ANY DIRECT, INDIRECT, CINCIDENTAL, SPECIAL, EXEMPLARY,
#         OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#         OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#         OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#         LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#         (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
#         USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#         OF SUCH DAMAGE.
#
#      For questions or comments contact jhjensen@chem.ku.dk 
#
#
#
#         PROPKA GUI: THE VISUAL MOLECULAR DYNAMICS (VMD) PLUGIN
#                       VERSION 1.1, Release 6
#
#                   Michal Rostkowski, Jan H. Jensen. 
#                       Department of Chemistry
#                        Copenhagen University
#                           27 January 2012
#
#
#   Reference:
#      "Graphical Analysis of pH-dependent Properties of Proteins predicted using PROPKA"
#       Michal Rostkowski, Mats HM Olsson, Chresten R Sondergaard and Jan H Jensen 
#       BMC Structural Biology 2011 11:6. 
#
#***********************************************************
#
#              THIS PROGRAM ALLOWS TO DISPLAY PROTEIN PKA VALUES 
#               ACCORDING TO THE PROPKA 2 and 3 CALCULATIONS 
#              MORE INFORMATION ABOUT PROPKA CAN BE FOUND AT:
#           http://propka.ki.ku.dk/~luca/wiki/index.php/PROPKA_2.0 
#           http://propka.ki.ku.dk/~luca/wiki/index.php/PROPKA_3.1_Tutorial
#
#                 CALCULATIONS CAN BE SUBMITTED ONLINE AT:
#                       http://propka.ki.ku.dk/
#
#***********************************************************
#

package require multiplot
package provide propka 1.1

namespace eval ::PropKa:: {
   namespace export propka
   variable w
   variable pkafile
   variable pdbfile 
   variable jobdir ".";     # set directory to the CWD
   variable outdir ".";     # set directory to the CWD

## System Settings:
## Windows Users
   variable propkapy_win {{C:\Program Files\PropKa\Propka3.0\propka.py}};
   variable propkapy31_win {{C:\Program Files\PropKa\Propka3.1\propka.py}};
   variable pythonpath_win {{python.exe}};
   
## Linux and Mac Users
   variable propkapy propka.py
   variable propkapy31 propka31.py
   variable pythonpath ""
   
## PropKa Running Parameters
   variable propka_py_ver "";            # PROPKA version (By default PROPKA.py is used when empty)
   variable propkaver 31;                # Run PROPKA 2 or 3 or 3.1

## GUI's To Do (Default options)
   variable showallpkares 0;             # show all ionizable residues (on/off) / (1/0)
   variable showallpkalabels 0;          # swow labels for all ionizable residues (on/off)
   variable showshiftedpka 0;            # show residues of maximim shifted pka on/off (1/0)
   variable showshiftedpkalabels 1;      # swow labels for ionizable residues with max. shifted pka (on/off)
   variable showabsshiftedpka 0;         # show residues of maximum shifted absolute pka on/off
   variable showabsshiftedpkalabels 1;   # swow labels for ionizable residues with abs. max. shifted pka (on/off)
   variable nummaxshifted 4;             # how many shifted residues to show
   variable maxshiftedtype 1;            # which kind of shifts to show (0/1/2)
   variable maxstabtype 1;               # which kind of delGs to show (0/1/2)
   variable plotener 0;                  # plot unfolding energy on/off
   variable plotcharge 0;                # plot charges for folded and unfolded states
   variable showshiftedby 0;             # show residues with pka shifted more than $thresholdpka
   variable thresholdpka;                # threshold pKa value to show residues (set def value or not ??)
   variable loadpdb 1;                   # read in appropriate pdb file (on/off)
   variable runpkamode 1;                # run PropKa calculations when loading PDB (off/on/options for loading in the Run Window)
   variable showhbonds 1;                # show hydrogen bonds between picked residue and its determinants (on/off)
   variable pickedlist {}
   variable pickedpkalist {}
   variable pickmode 1;                  #Show pKa determinants in Pick Mode by default (2 is for showinkg pKa value only)
   variable drawnids {}
   variable tracestatus 0
   variable topdrawn ""
   variable hbondsid ""
   variable allpkalabelsdrawn ""
   variable ph 7                         # Default pH for DelG calculations
   variable showdelg
   variable showmaxdelg
   variable repsrem 1
   variable toprepslistmols              # Keeps the information for which molecules top shifted or top contributions to delG reps were shown 
   variable labelsshifted 1;             # To show shifted labes ofr different molecules (on/off)
   variable coloredlabels 1;             # To show labels for different loaded molecules in different colours (on/off) Color shemes are given below
   variable colorlabels {white yellow blue2 gray cyan mauve green violet magenta red2 orange2}
   variable colordeterm {silver orange blue tan lime pink green3 violet3 magenta3 red3 orange3}
   variable labelsmolid 1
   variable infos 1;                     # Puts pKa informations in the console mode. (on/off) 
   variable readtyr 0;                   # Work with Tyrosines (on/off)
   variable readcystyr 1;                # Work with Tyrosines and Cysteines (on/off)
   variable runpropkamode 1;
   variable confsincl 0;                 # To load all configurations if found for the PropKa Results
   variable runpdbfile;                  # Variable to store absolute pathname for pdb to run propka with
   variable stabregions 0;               # Highlit of the stab and destab regions by different colours
   variable seltoshow "";
   variable pkaftype;                    # Used to control propka version to work with
   variable reference_shown 0; 
   variable loadedpkafiles ""
   variable namelist "GLY ALA VAL LEU ILE MET PHE TRP PRO SER THR CYS TYR ASN GLN ASP GLU LYS ARG HIS C- N+";
   variable abbrevlist "G A V L I M F W P S T C Y N Q D E K R H C- N+";
   variable consider_coupled 0; 
   variable runpkaopts "";
   variable recalc_coupled 0;
   variable include_residue_in_profile 1;
   variable del_label "DelG";
   variable dGQ_label "dG";
   variable molid_to_plot_for "";
   variable delpH 1;
   variable minpH 0;
   variable maxpH 14;
   variable determ_todisplay_threshold 0.0; # threshold for restricting which determinants should be displayed interactively (pka shift contribution)
   
}



set pkaftypes { 
  { {pKa Files}   {.pka} }
  { {All Files}        * }
 }

 set pdbftypes {
  { {Pdb Files}   {.pdb} }
  { {All Files}        * }
 }

 
 proc ::PropKa::propka_info {}  {
     
    puts "\nPropKa GUI Reference) In your publications of scientific results based"
    puts "PropKa GUI Reference) on the use of the PropKa GUI, please reference;"
    puts "PropKa GUI Reference) Michal Rostkowski, Mats H.M. Olsson,"
    puts "PropKa GUI Reference) Chresten R. Sondergaard and Jan H. Jensen:"
    puts "PropKa GUI Reference) Graphical analysis of pH-dependent properties of proteins"
    puts "PropKa GUI Reference) predicted using PROPKA. BMC Bioinformatics. 2011,11:6\n"
 
} 


proc ::PropKa::propka_gui {}  {
   variable w
   variable pkaftypes
   variable pdbftypes
   variable reference_shown
   

   if { [winfo exist .propka] } {
      wm deiconify $w
      raise $w
      return
   }

    set w [toplevel ".propka"]
    wm title $w "PropKa"

    set ::PropKa::max_contrib_labeltext "With maximum contributions to $::PropKa::del_label"
    set ::PropKa::show_contrib_labeltext "Show their contributions to $::PropKa::del_label"
    set ::PropKa::show_picked_labeltext "$::PropKa::del_label contributions"
    set ::PropKa::stab_label Stabilizing
    set ::PropKa::destab_label Destabilizing
    
    
    ##
    ## The Menu Bar
    ##
    frame $w.menubar -relief raised -bd 2
    pack $w.menubar -padx 1 -fill x

    ## File Menu
    menubutton $w.menubar.file -text "File" -menu $w.menubar.file.menu
    $w.menubar.file config -width 4
    menu $w.menubar.file.menu -tearoff no
    $w.menubar.file.menu add command -label "Load PROPKA Results File..." -command {
        set filetest [tk_getOpenFile -title "PROPKA Results File" -multiple 1 -filetypes $pkaftypes -initialdir $::PropKa::jobdir]
        if { $filetest != "" } {
            set ::PropKa::coupled_found_for ""
            set ::PropKa::ligands_found_for ""
            foreach tmpfile $filetest {
                set ::PropKa::pkafile $tmpfile
                set ::PropKa::jobdir [file dirname $::PropKa::pkafile]
                cd $::PropKa::jobdir
                wm title $::PropKa::w "PropKa   $::PropKa::pkafile"
                ::PropKa::propka_results
            }
            
            if { $::PropKa::coupled_found_for != "" } {
               ::PropKa::propka_recalculate_coupled 
            }
            
            if { $::PropKa::ligands_found_for != "" } {
                tk_dialog .errmsg {PropKa Warning} "Ligands found in the following structures:\n\n[join $::PropKa::ligands_found_for \n]" error 0 Dismiss
            }
        }
    }
    
    $w.menubar.file.menu add command -label "Run PROPKA" -command ::PropKa::propka_load_pdb_window
    
 
    ##
    ##Options  Menu
    ##
    menubutton $w.menubar.options -text "Options" -menu $w.menubar.options.menu
    $w.menubar.options config -width 7
    menu $w.menubar.options.menu -tearoff no
    $w.menubar.options.menu add cascade -label "Labels" -menu $w.menubar.options.menu.labels
    menu $w.menubar.options.menu.labels
    $w.menubar.options.menu.labels add checkbutton -label "Include MoleculeID in Labels" -variable ::PropKa::labelsmolid
    $w.menubar.options.menu.labels add checkbutton -label "Shifted Labels for Loaded Molecules" -variable ::PropKa::labelsshifted
    $w.menubar.options.menu.labels add checkbutton -label "Colored Labes for Loaded Molecules" -variable ::PropKa::coloredlabels
    

    $w.menubar.options.menu add separator 
    
    
    $w.menubar.options.menu add checkbutton -label "Plot \(Un\)Folding Energy" -variable ::PropKa::plotener -command ::PropKa::propka_plotener
    $w.menubar.options.menu add checkbutton -label "Plot Charge" -variable ::PropKa::plotcharge -command ::PropKa::propka_plotchrg
    $w.menubar.options.menu add checkbutton -label "Show H-bonds with Determinants" -variable ::PropKa::showhbonds
    $w.menubar.options.menu add checkbutton -label "Load PDB with PROPKA Results File" -variable ::PropKa::loadpdb
    $w.menubar.options.menu add checkbutton -label "Include Tyrosines" -variable ::PropKa::readtyr -command { set ::PropKa::readcystyr 0; ::PropKa::propka_reread }
    $w.menubar.options.menu add checkbutton -label "Include Tyrosines and Cysteines" -variable ::PropKa::readcystyr -command { set ::PropKa::readtyr 0; ::PropKa::propka_reread }
    $w.menubar.options.menu add checkbutton -label "Remove Labels with Representations" -variable ::PropKa::repsrem 
    $w.menubar.options.menu add checkbutton -label "Print Info" -variable ::PropKa::infos

    $w.menubar.options.menu add checkbutton -label "Colored Regions" -variable ::PropKa::stabregions -command ::PropKa::propka_color_regions
    $w.menubar.options.menu add checkbutton -variable ::PropKa::consider_coupled -label "Consider Coupled Residues"
    $w.menubar.options.menu add checkbutton -label "Calculate Charge" -variable ::PropKa::calculate_charge -command ::PropKa::propka_update_main_window 
    $w.menubar.options.menu add command -label "Plot Window" -command ::PropKa::profiles_window
    $w.menubar.options.menu add cascade -label "Displayed Determinant Threshold:" -menu $w.menubar.options.menu.displ_determ_threshold
    menu $w.menubar.options.menu.displ_determ_threshold
    $w.menubar.options.menu.displ_determ_threshold add radiobutton -variable ::PropKa::determ_todisplay_threshold -value 0.0 -label 0.0
    $w.menubar.options.menu.displ_determ_threshold add radiobutton -variable ::PropKa::determ_todisplay_threshold -value 0.1 -label 0.1
    $w.menubar.options.menu.displ_determ_threshold add radiobutton -variable ::PropKa::determ_todisplay_threshold -value 0.2 -label 0.2
    $w.menubar.options.menu.displ_determ_threshold add radiobutton -variable ::PropKa::determ_todisplay_threshold -value 0.3 -label 0.3
    $w.menubar.options.menu.displ_determ_threshold add radiobutton -variable ::PropKa::determ_todisplay_threshold -value 0.5 -label 0.5
    $w.menubar.options.menu.displ_determ_threshold add radiobutton -variable ::PropKa::determ_todisplay_threshold -value 0.7 -label 0.7
    $w.menubar.options.menu.displ_determ_threshold add radiobutton -variable ::PropKa::determ_todisplay_threshold -value 1.0 -label 1.0
    

    

    ##
    ##Help Menu
    ##
    menubutton $w.menubar.help -text "Help" -menu $w.menubar.help.menu
    $w.menubar.help config -width 5
    menu $w.menubar.help.menu -tearoff no
#    $w.menubar.help.menu add command -label "PROPKA Web Interface..." -command "vmd_open_url http://propka.ki.ku.dk"
#    $w.menubar.help.menu add command -label "GUI Homepage..." -command "vmd_open_url http://propka.ki.ku.dk/~luca/wiki/index.php/GUI_Web"
    $w.menubar.help.menu add command -label "PROPKA Home page..." -command "vmd_open_url http://propka.org"
    $w.menubar.help.menu add command -label "PROPKA GUI video tutorial..." -command "vmd_open_url https://www.youtube.com/watch?v=bonpPmJdsNM"
    $w.menubar.help.menu add command -label "PROPKA GUI Documentation..." -command "vmd_open_url http://www.ks.uiuc.edu/Research/vmd/plugins/propka/"


    pack $w.menubar.file -side left
    pack $w.menubar.options -side left
    pack $w.menubar.help -side right
   
    
    
    
    ##
    ## Main Window
    ##
    frame $w.displopts -borderwidth 1
    label $w.displopts.title -text "Show Residues"
    grid $w.displopts.title -row 0 -sticky w -columnspan 7
   

    checkbutton $w.displopts.maxshifteddispl -variable ::PropKa::showshiftedpka -anchor w -command { 
                set ::PropKa::showshiftedby 0
                ::PropKa::propka_shifted_pkas 
    }
    grid $w.displopts.maxshifteddispl -row 1 -column 0 -sticky w
    grid [entry $w.displopts.showshifted -textvar ::PropKa::nummaxshifted -width 2 -justify center ] -row 1 -column 1 -sticky w
    grid [label $w.displopts.top -text "With maximum shifted pKas  OR"] -row 1 -column 2 -sticky w -columnspan 7
    
    checkbutton $w.displopts.shifted -text "Shifted by more than:" -variable ::PropKa::showshiftedby -command { 
                set ::PropKa::showshiftedpka 0 
                set ::PropKa::showmaxdelg 0
                ::PropKa::propka_shifted_pkas
    }
    grid $w.displopts.shifted -row 1 -column 9 -sticky w -columnspan 6
    grid [entry $w.displopts.threshold -textvar ::PropKa::thresholdpka -justify center -width 3] -row 1 -column 15 -sticky w

    radiobutton $w.displopts.maxshiftedabs -variable ::PropKa::maxshiftedtype -value 1 -text "Absolute" -command ::PropKa::propka_shifted_pkas 
    radiobutton $w.displopts.maxshifteddown -variable ::PropKa::maxshiftedtype -value 2 -text "Down" -command ::PropKa::propka_shifted_pkas 
    radiobutton $w.displopts.maxshiftedup -variable ::PropKa::maxshiftedtype -value 3 -text "Up" -command ::PropKa::propka_shifted_pkas 
    
    grid $w.displopts.maxshiftedabs -row 2 -column 3 -sticky w -columnspan 2
    grid $w.displopts.maxshifteddown -row 2 -column 5 -sticky w -columnspan 5
    grid $w.displopts.maxshiftedup -row 2 -column 10 -sticky w -columnspan 3
    
    checkbutton $w.displopts.delglabels -textvariable ::PropKa::show_contrib_labeltext -variable ::PropKa::showdelg -anchor w -command {
                ::PropKa::propka_shifted_pkas
    }
    
    
    grid $w.displopts.delglabels -row 3 -column 1 -sticky w -columnspan 9
    grid [label $w.displopts.phtext -text "pH = "] -row 3 -column 14 -sticky w 
    grid [entry $w.displopts.delgph -textvar ::PropKa::ph -justify center -width 3] -row 3 -column 15 -sticky w
    grid [checkbutton $w.displopts.alldelg -textvariable ::PropKa::max_contrib_labeltext -variable ::PropKa::showmaxdelg -anchor w -command ::PropKa::propka_delg] -row 4 -column 0 -columnspan 12 -sticky w 
    
    radiobutton $w.displopts.maxabsstab -variable ::PropKa::maxstabtype -value 1 -text "Absolute" -command {set ::PropKa::showmaxdelg 1; ::PropKa::propka_delg}
    radiobutton $w.displopts.maxstab -variable ::PropKa::maxstabtype -value 2 -textvariable ::PropKa::stab_label -command {set ::PropKa::showmaxdelg 1; ::PropKa::propka_delg}
    radiobutton $w.displopts.maxdestab -variable ::PropKa::maxstabtype -value 3 -textvariable ::PropKa::destab_label -command {set ::PropKa::showmaxdelg 1; ::PropKa::propka_delg}
    
  
    
    grid $w.displopts.maxabsstab -row 5 -column 3 -sticky w -columnspan 3
    grid $w.displopts.maxstab -row 5 -column 5 -sticky w -columnspan 5
    grid $w.displopts.maxdestab -row 5 -column 10 -sticky w -columnspan 5
    
    
    grid [checkbutton $w.displopts.allpkas -text "All ionizable" -variable ::PropKa::showallpkares -anchor w -command ::PropKa::propka_show_all] -row 6 -column 0 -columnspan 3 -sticky w 

    button $w.displopts.showresi -text "Show Selection:" -command ::PropKa::show_selection -width 18
    entry $w.displopts.showresi_sel -textvariable "::PropKa::seltoshow" -justify center -width 30
    
    grid $w.displopts.showresi -row 7 -column 0 -sticky w -columnspan 5
    grid $w.displopts.showresi_sel -row 7 -column 5 -sticky w -columnspan 12
    
    pack $w.displopts -padx 5 -pady 3 -anchor w
    
    frame $w.pickopts -borderwidth 1
    label $w.pickopts.title -text "Pick Mode - Show:"
    grid $w.pickopts.title -row 0 -sticky w -columnspan 4
    
    
    radiobutton $w.pickopts.pickdeterm -variable ::PropKa::pickmode -value 1 -text "pKa determinants" -command ::PropKa::picking
    grid $w.pickopts.pickdeterm -row 1 -column 0 -sticky w
    radiobutton $w.pickopts.pickpka -variable ::PropKa::pickmode -value 2 -text "pKa values" -command ::PropKa::picking
    grid $w.pickopts.pickpka -row 1 -column 1 -sticky w
    radiobutton $w.pickopts.picdelg -variable ::PropKa::pickmode -value 3 -textvariable ::PropKa::show_picked_labeltext -command ::PropKa::picking
    grid $w.pickopts.picdelg -row 1 -column 2 -sticky w
    radiobutton $w.pickopts.pickmode -variable ::PropKa::pickmode -value 0 -text "Off" -command ::PropKa::picking
    grid $w.pickopts.pickmode -row 1 -column 3 -sticky w

    
    pack $w.pickopts -padx 5 -pady 3 -anchor w
    
    frame $w.clear -borderwidth 1
    
    button $w.clear.clearlabels -text "  Delete all Labels  " -command ::PropKa::propka_remove_all_labels
    grid $w.clear.clearlabels -row 0 -column 1 
    
    button $w.clear.clearrep -text "   Delete added Representations  " -command ::PropKa::propka_remove_allreps
    grid $w.clear.clearrep -row 0 -column 3

    pack $w.clear -padx 5 -pady 3 -anchor center
    
    if {$reference_shown == 0 } {
        ::PropKa::propka_info
        set reference_shown 1
    }
    
}

proc propka_tk {} {
   ::PropKa::propka_gui
   return $::PropKa::w
}


proc ::PropKa::propka_update_main_window {} {

        if {$::PropKa::calculate_charge == 0 } {
           set ::PropKa::del_label DelG
           set ::PropKa::show_contrib_labeltext "Show their contributions to $::PropKa::del_label"
           set ::PropKa::stab_label Stabilizing
           set ::PropKa::destab_label Destabilizing
           set ::PropKa::dGQ_label dG
        } else {
           set ::PropKa::del_label Q
           set ::PropKa::show_contrib_labeltext "Show their $::PropKa::del_label"
           set ::PropKa::stab_label Positive
           set ::PropKa::destab_label Negative
           set ::PropKa::dGQ_label Q
        }

        set ::PropKa::max_contrib_labeltext "With maximum contributions to $::PropKa::del_label"
        set ::PropKa::show_picked_labeltext "$::PropKa::del_label contributions"

}



proc ::PropKa::show_selection {} {
      variable seltoshow
      
      if { $seltoshow != "" } {
         mol representation licorice
         mol selection $seltoshow
         mol addrep top
      }
}




proc ::PropKa::propka_load_pdb_window {} { 



    variable wrun
    
    if { [winfo exist .loadpdb] } {
      wm deiconify $wrun
      raise $wrun
      return
   }

    variable selpka "" 
    
    set wrun [toplevel ".loadpdb"]
    wm title $wrun "Run PROPKA For The:"

    frame $wrun.pdbload -borderwidth 1
    radiobutton $wrun.pdbload.runpkamode1 -variable ::PropKa::runpkamode -value 1 -text "TOP Molecule"
    radiobutton $wrun.pdbload.runpkamode2 -variable ::PropKa::runpkamode -value 2 -text "Specified Selection for the TOP Molecule:"
    radiobutton $wrun.pdbload.runpkamode3 -variable ::PropKa::runpkamode -value 3 -text "PDB File:"
    
    
    
    entry $wrun.pdbload.select -textvariable "::PropKa::selpka" -justify center -justify center -width 20

    if { $::PropKa::propkaver == 2 } { 
       set state_for_propka2 normal
       set state_for_propka3 disabled
    } else {
       set state_for_propka2 disabled
       set state_for_propka3 normal
    }
   

    checkbutton $wrun.pdbload.configs -state $state_for_propka2 -variable ::PropKa::confsincl -text "Include Configurations"
    checkbutton $wrun.pdbload.protonly -variable ::PropKa::runprotonly -text "Prot. Only"
    checkbutton $wrun.pdbload.sepchains -state $state_for_propka3 -variable ::PropKa::separatechains -text "Split Chains"
    
    label $wrun.pdbload.runpkaoptlab -state $state_for_propka2 -text "Include Chain\(s\) ID\(s\)  (separated by \",\") "
    entry $wrun.pdbload.runpkaopts -state $state_for_propka2 -textvariable "::PropKa::runpkaopts" -justify center -width 3
    
    label $wrun.pdbload.propkaverlab -text "PROPKA:"

    checkbutton $wrun.pdbload.coupled -state $state_for_propka3 -variable ::PropKa::consider_coupled -text "Consider Coupled Residues"
   
    set pdbfiletest ""
    
    entry $wrun.pdbload.openinfo -textvariable "::PropKa::runpdbfile" -justify center -width 35
    button $wrun.pdbload.openbrowser -text "Browse..." -pady 2 -command {
        set pdbfiletest [tk_getOpenFile -title "Choose PDB File" -multiple 1 -filetypes $pdbftypes -initialdir $::PropKa::jobdir]
        if { $pdbfiletest != "" } {
            puts $pdbfiletest
            set ::PropKa::runpdbfile $pdbfiletest
            set ::PropKa::runpkamode 3
        }
    }
    
    button $wrun.pdbload.run -text "               Run PROPKA               " -pady 2 -command { 
       
        set ::PropKa::coupled_found_for ""
        set ::PropKa::ligands_found_for ""
        
        if { $::PropKa::runpkamode == 3 } { 
            
            if { $pdbfiletest == "" &&  $::PropKa::runpdbfile != "" } {
               set pdbfiletest $::PropKa::runpdbfile
            }
            foreach pdbfiletmp $pdbfiletest {
                set splitpdbs ""
                set splitedchainlist ""
                set splitedfileslist ""
                if { $::PropKa::separatechains == 1 } {
                    set ::PropKa::runprotonly 1
                    set splitpdbs 1
                }

                if { $::PropKa::runprotonly == 1 } {

                    set pdbfile $pdbfiletmp
                    set pdbsource [open $pdbfile r]
                    if { $splitpdbs == "" } {
                        set pdbfiletarg [file rootname $pdbfile]_prot.pdb
                        set pdbtarget [open $pdbfiletarg w]

                        puts pdbfiletarg
                        puts $pdbfiletarg
                        puts pdbtarget
                        puts $pdbtarget
                        
                        set splitedfileslist $pdbfiletarg
                        
                    } 
                    
                    puts pdbfile
                    puts $pdbfile

                    puts pdbsource
                    puts $pdbsource

                    
                    while {![eof $pdbsource]} {
                        set line [gets $pdbsource]
                        if {[lindex $line 0] == "ATOM" } {
                           if { $splitpdbs == 1 } {
                              set chaintmp [lindex [string range $line 20 22] 0]
                              if { [lsearch $splitedchainlist $chaintmp] == "-1" } {
                                 if { $splitedfileslist != "" } {
                                    close $pdbtarget
                                 }
                                 lappend splitedchainlist $chaintmp
                                 set pdbfiletarg [file rootname $pdbfile]_prot$chaintmp.pdb
                                 set pdbtarget [open $pdbfiletarg w]
                                 lappend splitedfileslist $pdbfiletarg
                                 puts $pdbfiletarg
                                 puts $pdbtarget
                              }
                           }
                           puts $pdbtarget $line
                        }
                    }
                    close $pdbsource
                    close $pdbtarget
                    if { $splitpdbs == "" } {
                         set ::PropKa::runpdbfile $splitedfileslist
                        ::PropKa::propka_pdb
                    } else {
                        foreach newpdb $splitedfileslist {
                          puts "New pdb set; $newpdb"
                          set ::PropKa::runpdbfile $newpdb
                          ::PropKa::propka_pdb
                        }
                    }
                    
                    
                } else {
                     set ::PropKa::runpdbfile $pdbfiletmp
                    ::PropKa::propka_pdb
                }
                
            }
        } else { 
            ::PropKa::propka_pdb
        }
        if {$::PropKa::coupled_found_for != "" } {
            ::PropKa::propka_recalculate_coupled
        }
        
        if { $::PropKa::ligands_found_for != "" } {
            tk_dialog .errmsg {PropKa Warning} "Ligands found in the following structures:\n\n[join $::PropKa::ligands_found_for \n]" error 0 Dismiss
        }
    }
   
    
    radiobutton $wrun.pdbload.runpkver2 -variable ::PropKa::propkaver -value 2 -text "2.0" -command {
        $::PropKa::wrun.pdbload.configs config -state normal
        $::PropKa::wrun.pdbload.runpkaoptlab config -state normal
        $::PropKa::wrun.pdbload.runpkaopts config -state normal
        $::PropKa::wrun.pdbload.sepchains config -state disabled
        
    }
    
    radiobutton $wrun.pdbload.runpkver3 -variable ::PropKa::propkaver -value 3 -text "3.0" -command {
        $::PropKa::wrun.pdbload.configs config -state disabled
        $::PropKa::wrun.pdbload.runpkaoptlab config -state disabled
        $::PropKa::wrun.pdbload.runpkaopts config -state disabled
        $::PropKa::wrun.pdbload.sepchains config -state normal
    }
    
    radiobutton $wrun.pdbload.runpkver31 -variable ::PropKa::propkaver -value 31 -text "3.1" -command {
        $::PropKa::wrun.pdbload.configs config -state disabled
        $::PropKa::wrun.pdbload.runpkaoptlab config -state disabled
        $::PropKa::wrun.pdbload.runpkaopts config -state disabled
        $::PropKa::wrun.pdbload.sepchains config -state normal
    }
    
    
    grid $wrun.pdbload.runpkamode1 -column 0 -row 0 -sticky w -columnspan 4
    grid $wrun.pdbload.runpkamode2 -column 0 -row 1 -sticky w -columnspan 6
    grid $wrun.pdbload.select -column 6 -row 1 -sticky w -columnspan 4
    grid $wrun.pdbload.runpkamode3 -column 0 -row 2 -sticky w -columnspan 3
    grid $wrun.pdbload.openinfo -column 3 -row 2 -sticky w -columnspan 5
    grid $wrun.pdbload.openbrowser -column 8 -row 2 -sticky w
       
    

    grid $wrun.pdbload.propkaverlab -column 1 -row 3 -sticky w -columnspan 2
    
    grid $wrun.pdbload.runpkaoptlab -column 3 -row 3 -sticky w -columnspan 6
    grid $wrun.pdbload.runpkaopts -column 8 -row 3 -sticky w -columnspan 4

    grid $wrun.pdbload.run -column 3 -row 8 -sticky w -columnspan 8
   
    grid $wrun.pdbload.runpkver2 -column 2 -row 4 -sticky w
    grid $wrun.pdbload.runpkver3 -column 2 -row 5 -sticky w
    grid $wrun.pdbload.runpkver31 -column 2 -row 6 -sticky w

    
    grid $wrun.pdbload.configs -column 3 -row 4 -sticky w -columnspan 4

    grid $wrun.pdbload.protonly -column 3 -row 5 -sticky w -columnspan 2
    grid $wrun.pdbload.sepchains -column  5 -row 5 -sticky w -columnspan 2
  
    grid $wrun.pdbload.coupled -column  3 -row 6 -sticky w -columnspan 3
    pack $wrun.pdbload -fill both -expand true -side top -padx 5 -pady 5
    
    return $wrun

}

proc ::PropKa::propka_info_window { args } {

    variable coupled_found_for_run
    variable coupledlog
    variable wcoupled

    
    if { [winfo exist .coupled_info] } {
    
       foreach log $coupled_found_for_run {
            $wcoupled.main.info insert end "\n============$log\============\n\n"
            $wcoupled.main.info insert end [join $coupledlog($log) "\n"] 
            $wcoupled.main.info insert end "\n\n\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n"
       }
       
       wm deiconify $wcoupled
       raise $wcoupled
       return
    
    }
     
    set wcoupled [toplevel ".coupled_info"]
    wm title $wcoupled "Copled residues info:"
    
    frame $wcoupled.main -borderwidth 1
    text $wcoupled.main.info -background white -yscrollcommand "$wcoupled.main.vscr set" -font {courier 8} -width 115
    scrollbar $wcoupled.main.vscr -command "$wcoupled.main.info yview"
    
    pack $wcoupled.main.vscr -side right -fill y
    pack $wcoupled.main 
    
    foreach log $coupled_found_for_run {
        $wcoupled.main.info insert end "\n============$log\============\n\n"
        $wcoupled.main.info insert end [join $coupledlog($log) "\n"] 
        $wcoupled.main.info insert end "\n\n\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n"
    }

    pack $wcoupled.main.info -fill both -expand true

}

proc ::PropKa::propka_recalculate_coupled {} {

   set ::PropKa::recalc_coupled [tk_dialog .errmsg {PropKa Warning} " Coupled residues found in the following structures:\n\n[join $::PropKa::coupled_found_for \n]\n\n          For details check the PROPKA output" error 0 Dismiss Recalculate]
                                
   if { $::PropKa::recalc_coupled == 1 } {

        set ::PropKa::coupled_found_for_run ""
        foreach file_to_recalculate $::PropKa::coupled_found_for {
            set ::PropKa::runpdbfile $file_to_recalculate.pdb
            set ::PropKa::propkaver 31
            set ::PropKa::consider_coupled 1
            set ::PropKa::runpkamode 3
            file copy -force $::PropKa::runpdbfile $file_to_recalculate\_alt_state.pdb
            ::PropKa::propka_pdb
        }                   
        if { $::PropKa::coupled_found_for_run != "" } {
            ::PropKa::propka_info_window
        }
        set ::PropKa::runpdbfile ""
        set ::PropKa::recalc_coupled 0
    }
}

proc ::PropKa::propka_reread {} {

    variable pkaftype
    variable loadedpkafiles
    
    set currmol m[molinfo top]

    if { [lsearch $loadedpkafiles $currmol ] == "-1" } {
       return
    }

    if { $pkaftype($currmol) == 3 } {
        ::PropKa::propka3_read
    } else {
        ::PropKa::propka_read
    }
    
}

    
proc ::PropKa::propka_pdb {} {

    variable pdbfile
    variable pkafile
    variable runpkamode
    variable loadpdb
    variable jobdir
    
    variable confsincl
    variable selpka
    variable pdbftypes
    variable runpdbfile
    variable runpkaopts
    variable propkaver
    
    variable propkapy_win
    variable propkapy31_win
    variable pythonpath_win
    variable propka_py_ver
    
    variable propkapy
    variable propkapy31
    variable pythonpath ""
    
    variable recalc_coupled
    variable consider_coupled
    
    variable coupled_found_for_run
    variable coupledlog
     
    
    if { $runpkaopts != "" } {
        set propkaopts "-c $runpkaopts"
        set opts [split $runpkaopts ","]
        puts $propkaopts
    } else {
        set propkaopts ""
    }
    

    set loadpdb 0

    if { $runpkamode == 3 } {
            set pdbfile $runpdbfile
            set jobdir [file dirname $pdbfile]
            set tailname [file tail $pdbfile]
            set loadpdb 1
    } else {
       if { [molinfo num] == 0 } {
          tk_dialog .errmsg {PropKa Warning} "No TOP Molecule Found!" error 0 Dismiss
          return
       }
       set tailname [molinfo top get name]
       set pathtmp [molinfo top get filename]
       set loc [expr [string last "\{" $pathtmp] + 1] 
       set pathtofile [string range $pathtmp $loc end-$loc]
       set jobdir [file dirname $pathtofile]
       set pdbfile $tailname
   }
     
     

    if { $runpkamode == 2 } {
        
        if { $selpka != ""} { 
            if { [molinfo num] == 0 } {
                tk_dialog .errmsg {PropKa Warning} "No TOP Molecule Found!" error 0 Dismiss
                return
            }
            puts "Selection :atomselect top \"$selpka\""

            set selection [atomselect top "$selpka"]

            set pdbfiletest [tk_getSaveFile -title "Save PDB File for the Given Selection" -initialdir $::PropKa::jobdir]
            if { $pdbfiletest != "" } {
                set pdbfile $pdbfiletest
                set jobdir [file dirname $pdbfile]
                $selection writepdb $pdbfile
                $selection delete
                set tailname [file tail $pdbfile]
                set loadpdb 1
            }
        } else {
            tk_dialog .errmsg {PropKa Warning} "No Selection For the Current TOP Molecule Found!" error 0 Dismiss
            return
        }
    }
    
    cd $jobdir
    
    
    # Initializing PROPKA calculations
    set propka_opts ""
    
    if { [file exist $pdbfile] == 1 } {

        if { $propkaver == 3 || $propkaver == 31 } {
            if { $consider_coupled == 1 && $recalc_coupled == 1 } {
                set propka_opts "-d"
            }         

            set errorlog ""

            if { [regexp WIN [vmdinfo arch]] } {
                if { $propkaver == 31 } {
                    catch {eval exec $pythonpath_win $propkapy31_win $propka_opts $tailname} unwanted
                } else {
                    catch {eval exec $pythonpath_win $propkapy_win $propka_opts $tailname} unwanted
                }
                
                 puts "PROPKA Called"
            }  else { 
                if { $propkaver == 31 } {
                    catch { eval exec $pythonpath $propkapy31 $propka_opts $propka_py_ver $tailname} unwanted
                } else {
                    catch { eval exec $pythonpath $propkapy $propka_opts $propka_py_ver $tailname} unwanted
                }
                    puts "PROPKA Called"
            }
            

            set unwanted [split $unwanted \n] 
            set loadfiles 1
            
            
            set read_coupled_info 0
            

            
            foreach line $unwanted {
                
                if { [regexp Traceback $line] } {
                    set loadfiles 0
                    lappend errorlog "Problem with pKa calculations\n$prevline"
                }
                
                if { $loadfiles == 0 } {
                   lappend errorlog "$line\n"
                }
                
                if { $consider_coupled == 1 } { 
                    if {[regexp -- "Detecting non-covalently" $line] } { 
                        set read_coupled_info 1
                        lappend coupled_found_for_run $tailname  
                        set coupledlog($tailname) ""
                        set coupled_info_file [open $tailname-CoupledInfo.txt w]
                    }
                    
                    if {[regexp -- "DESOLVATION" $line] && $read_coupled_info == 1} {
                        set read_coupled_info 0
                        close $coupled_info_file
                    }
                    
                    if { $read_coupled_info == 1 } {
                        regsub -all {\033\[31m|\033\[30m} $line "" line
                        lappend coupledlog($tailname) $line
                        puts $coupled_info_file $line
                    }
                    
                }
                
                set prevline $line
            }
            
            if { $loadfiles == 0 } {
                tk_dialog .errmsg "PropKa Warning; Structure $tailname Not Loaded" [join $errorlog] error 0 Dismiss
            }
        } else {
            catch {exec propka2.0.pl -i $tailname $propkaopts} unwanted
        }
    } else {
        tk_dialog .errmsg {PropKa Warning} "PDB File Not Found" error 0 Dismiss
        return
    }
    

    if { $runpkaopts != "" } {
        set outslist [glob [file rootname $tailname]_[join $opts ""]*.pdb]
    } else {
        set outslist [glob [file rootname $tailname]*.pdb ]
    }
  
  
    # When PROPKA3.0 or 3.1 will be used
    if { $propkaver != 2 } {

        if { [file exist [file rootname $tailname].pka] == 1 && $loadfiles == 1} {
            if { $propka_opts == "-d" } {
               set pkafile [file rootname $tailname]_alt_state.pka
            } else {
               set pkafile [file rootname $tailname].pka
            }
        } else {
          return
        }
         ::PropKa::propka_results
          return
    }
    
    # When PROPKA2 is to be used
    if { [lsearch $outslist "*conf*"] != "-1" } { 
       set loadpdb 1
        if { $confsincl == 1 } {
            foreach out $outslist {
              set pdbfile $out
              if { [file exist $pdbfile.pka] == 1 } {
                set pkafile $pdbfile.pka
                ::PropKa::propka_results
              } else {
                tk_dialog .errmsg {PropKa Warning} "Unable to find correct PROPKA results file!" error 0 Dismiss
                return
              }
            }
          } else {
              set pdbfile [lindex $outslist [lsearch $outslist "*conf1*"] ]
              if { [file exist $pdbfile.pka] == 1 } {
                  set pkafile $pdbfile.pka
              } else {
                  tk_dialog .errmsg {PropKa Warning} "Unable to find correct PROPKA results file!" error 0 Dismiss
                  return
              }
              ::PropKa::propka_results
          }
      } else {
          if { [file exist $pdbfile.pka] == 1 } {
             set pkafile $pdbfile.pka
          } else {
             tk_dialog .errmsg {PropKa Warning} "Unable to find correct PROPKA results file!" error 0 Dismiss
             return
          }
          ::PropKa::propka_results
    }
}




proc ::PropKa::profiles_window {} {

    variable wprofiles
    variable loadedpkafiles
    
    if { [winfo exist .profiles_handle] } {
      wm deiconify $wprofiles
      raise $wprofiles
      return
   }
   
   
    set wprofiles [toplevel ".profiles_handle"]
    wm title $wprofiles "pH profile plotting tool"
    
    frame $wprofiles.main -borderwidth 1
    label $wprofiles.main.header -text "Plot pH profile of: "
    label $wprofiles.main.header_res -text "Residues List"
    label $wprofiles.main.header_molid -text "Mol ID"
    radiobutton $wprofiles.main.calc_delG -variable ::PropKa::calculate_charge -value 0 -text DelG -command ::PropKa::propka_update_main_window
    radiobutton $wprofiles.main.calcQ -variable ::PropKa::calculate_charge -value 1 -text Q -command {
        ::PropKa::propka_update_main_window
    }
    
   
    radiobutton $wprofiles.main.plot_for -variable ::PropKa::include_residue_in_profile -value 1 -text "For       -OR- "
    radiobutton $wprofiles.main.exlude_in_plot -variable ::PropKa::include_residue_in_profile -value 0 -text "Excluding  selected residues"

    entry $wprofiles.main.selection -textvariable "::PropKa::seltoshow"  -width 33
    button $wprofiles.main.plot_selection -text "Show Selection" -width 18 -background gray -command { 
        if { [llength $::PropKa::molid_to_plot_for] == 1 } {
            mol on [string range $::PropKa::molid_to_plot_for 1 end]
        }
        ::PropKa::show_selection 
    }

    
    label $wprofiles.main.delPHlabel -text "Del pH"
    label $wprofiles.main.minPHlabel -text "Min pH"
    label $wprofiles.main.maxPHlabel -text "Max pH"
    
    entry $wprofiles.main.delPHentry -textvariable "::PropKa::delpH" -width 3 -justify center 
    entry $wprofiles.main.minPHentry -textvariable "::PropKa::minpH" -width 3 -justify center 
    entry $wprofiles.main.maxPHentry -textvariable "::PropKa::maxpH" -width 3 -justify center 
    
    grid $wprofiles.main.header -row 0 -column 0 -sticky w -columnspan 3
    grid $wprofiles.main.calc_delG -row 0 -column 3 -sticky w
    grid $wprofiles.main.calcQ -row 0 -column 4 -sticky w
    
    grid $wprofiles.main.header_res -row 0 -column 7 -sticky we -columnspan 4
    grid $wprofiles.main.header_molid -row 0 -column 13 -sticky w
    
    grid $wprofiles.main.plot_for -row 1 -column 0 -sticky w -columnspan 3
    grid $wprofiles.main.exlude_in_plot -row 1 -column 3 -sticky w -columnspan 4

    grid $wprofiles.main.selection -row 2 -column 1 -sticky e -columnspan 5
 
    
    if {[lsearch $::PropKa::loadedpkafiles m[molinfo top]] != "-1" } {
       set residlist_to_plot $::PropKa::residlist(m[molinfo top])
    } else {
       set residlist_to_plot "" 
    }
    
    scrollbar $wprofiles.main.yscroll -command "$wprofiles.main.reslistbox yview"
    listbox $wprofiles.main.reslistbox -selectmode extended -listvariable residlist_to_plot -yscroll "$wprofiles.main.yscroll set" -height 10

    scrollbar $wprofiles.main.mol_yscroll -command "$wprofiles.main.molid_list_choice yview"
    listbox $wprofiles.main.molid_list_choice -listvariable ::PropKa::loadedpkafiles -selectmode extended -yscroll "$wprofiles.main.mol_yscroll set" -width 4 -height 10
    grid $wprofiles.main.molid_list_choice -row 1 -column 8 -sticky e -columnspan 1
    
    bind $wprofiles.main.molid_list_choice <<ListboxSelect>> { 
         if {$::PropKa::loadedpkafiles != ""} {
            set residlist_to_plot_tmp ""
            set mol_select [$::PropKa::wprofiles.main.molid_list_choice curselection]
            set ::PropKa::molid_to_plot_for ""
            foreach m $mol_select {
                lappend ::PropKa::molid_to_plot_for [$::PropKa::wprofiles.main.molid_list_choice get $m]
            }
           
            if {[llength $::PropKa::molid_to_plot_for] == 1 } { 
                mol top [string range $::PropKa::molid_to_plot_for 1 end]
                puts "molid to make a plot for has been changed to [string range $::PropKa::molid_to_plot_for 1 end]"
                foreach res $::PropKa::residlist($::PropKa::molid_to_plot_for) {
                    lappend residlist_to_plot_tmp "$::PropKa::name($res) $res $::PropKa::pka($res) \($::PropKa::shiftpka($res)\)"
                }
                set residlist_to_plot $residlist_to_plot_tmp
            } else {
              set ::PropKa::include_residue_in_profile 1
              set ::PropKa::seltoshow ""
              puts "Multiple molid selection occured"             
            }
            
         }
    }
    
    

    bind $wprofiles.main.reslistbox <<ListboxSelect>> { 
       set selection_made [$::PropKa::wprofiles.main.reslistbox curselection]
       set res_selection ""
       foreach index $selection_made {
         puts [$::PropKa::wprofiles.main.reslistbox get $index]
         
        set resid_picked [lindex [$::PropKa::wprofiles.main.reslistbox get $index] 1]
         
        if {[regexp -- "-" $resid_picked]} {
           lappend res_selection "[lindex [::PropKa::propka_get_sel $resid_picked] 0] and name [lindex [split $resid_picked "-_"] 1]"
        } else {
           lappend res_selection [lindex [::PropKa::propka_get_sel $resid_picked] 0]
        }
        set ::PropKa::molid_to_plot_for m[lindex [::PropKa::propka_get_sel $resid_picked] 2]
        set ::PropKa::seltoshow [join $res_selection " or "] 
       }
    }

    grid $wprofiles.main.reslistbox -row 1 -column 7 -sticky w -columnspan 4 -rowspan 5
    grid $wprofiles.main.yscroll -row 1 -column 11 -sticky ns -rowspan 5
    
    
    grid $wprofiles.main.molid_list_choice -row 1 -column 13 -sticky e -columnspan 1 -rowspan 5
    grid $wprofiles.main.mol_yscroll -row 1 -column 14 -sticky ns -rowspan 5

    button $wprofiles.main.generate_plot -text "Plot profile" -command ::PropKa::propka_makepHprofile -width 4 -background green

    grid $wprofiles.main.plot_selection -row 3 -column 2 -sticky we -columnspan 3
    grid $wprofiles.main.generate_plot -row 5 -column 1 -sticky we -columnspan 5
       
    grid $wprofiles.main.delPHlabel -row 4 -column 1 -columnspan 1 -sticky e
    grid $wprofiles.main.minPHlabel -row 4 -column 3 -columnspan 1 -sticky e 
    grid $wprofiles.main.maxPHlabel -row 4 -column 5 -columnspan 1 -sticky w
    
    grid $wprofiles.main.delPHentry -row 4 -column 2 -columnspan 1 -sticky w
    grid $wprofiles.main.minPHentry -row 4 -column 4 -columnspan 1 -sticky w
    grid $wprofiles.main.maxPHentry -row 4 -column 6 -columnspan 1 -sticky w 
  
    pack $wprofiles.main -side left
 
}



proc ::PropKa::propka_makepHprofile {}  { 

    if { [molinfo list] == "" } {
        return
    }
    
    set pathtmp [molinfo top get filename]
    set loc [expr [string last "\{" $pathtmp] + 1] 
    set pathtofile [string range $pathtmp $loc end-$loc]
    set jobdir [file dirname $pathtofile]
    
    cd $jobdir
    
    variable residlist
    variable abbrevname
    
    variable ph 
    
    variable seltoshow
    variable chainlist
    variable calculate_charge 
    
    variable include_residue_in_profile 
    
    variable namelist
    
    variable delpH
    variable minpH
    variable maxpH
    
 
    set reslist_to_plot ""
    variable molid_to_plot_for
    
    set currmol_tmp m[molinfo top]
    set pH_orig $ph
    
    set chargename ""
    set delgq dG
    set title "DelG Contributions"
    set ylabel "\[e\]" 
    
    if {$calculate_charge == 1 } {
       set chargename "_charge"
       set delgq dQ
       set title "Q"
       set ylabel "\[q\]" 
    }
    
    set make_a_plot 1
    set plotting_started 0
    if {[llength $molid_to_plot_for] > 1} {
       set seltoshow ""
       set currmol_tmp $molid_to_plot_for
    }

  foreach currmol $currmol_tmp {
    set list_check ""
    set list_to_plot ""
    set list_not_to_plot ""
    set total($currmol) ""
    set pH_file [open profile_[file rootname [molinfo [string range $currmol 1 end] get name]]$chargename.txt w]

    if {$seltoshow != "" } {
        set seltmp [atomselect top $seltoshow]
        set list_to_plot_tmp [$seltmp get "resid chain resname name"]

        foreach sel $list_to_plot_tmp {
           if {[lsearch $namelist [lindex $sel 2]] == "-1" } {
              set sel "[lindex $sel 2]-[lindex $sel 3]_ [lindex $sel 0] [lindex $sel 1]"
           } else {
              set sel "[lindex $sel 0] [lindex $sel 1]"
           }
        
           if {[lsearch $list_check $sel] == "-1"} {
              lappend list_check $sel
           }
        }
         
        set list_to_plot $list_check
        
        $seltmp delete
        foreach resitmp $list_to_plot {
            if { [lsearch $chainlist($currmol) [lindex $resitmp end]] != "-1" } {
                set res_to_plot [join $resitmp ""]$currmol
            } else {
                set res_to_plot [join [lrange $resitmp 0 end-1] ""][lindex $chainlist($currmol) 0]$currmol
            }
            
            puts "residue to make a plot for is $res_to_plot"
            
            if { [lsearch $residlist($currmol) $res_to_plot] != "-1" } {
                lappend reslist_to_plot $res_to_plot
            }
            
            if { [lsearch $residlist($currmol) N$res_to_plot] != "-1" } {
               lappend reslist_to_plot N$res_to_plot
            }

            if {[lsearch $residlist($currmol) C$res_to_plot] != "-1" } {
                lappend reslist_to_plot C$res_to_plot
            }
            
        }
        
        if {$include_residue_in_profile == 0} {
            set list_not_to_plot $reslist_to_plot
            set reslist_to_plot $residlist($currmol)
            set make_a_plot 0 
        }  
        
    } else {
        set reslist_to_plot $residlist($currmol)
        set make_a_plot 0 
    }

    set pHdata ""
    set pHdata_formatted ""
    for  {set ph $minpH} { $ph <= $maxpH} {set ph [expr $ph + $delpH]} {
           lappend pHdata $ph
           lappend pHdata_formatted [format %7.2f $ph]
           set dgq_total($ph) 0
           set ddgtotal_without_exluded($ph) 0
    }
    
    puts $pH_file "              pH/$delgq  [join $pHdata_formatted "   "]\n"
       
    foreach res $reslist_to_plot {
      
      for  {set ph $minpH} { $ph <= $maxpH} {set ph [expr $ph + $delpH]} {
           set dgtmp1 [::PropKa::propka_calc_delg $res]
           if { $make_a_plot != 0 } {
             if { [lsearch $residlist($currmol) N$res] != "-1"} {
               set dgtmp1 [expr $dgtmp1 + [::PropKa::propka_calc_delg N$res]]
             }
             if { [lsearch $residlist($currmol) C$res] != "-1"} {
                set dgtmp1 [expr $dgtmp1 + [::PropKa::propka_calc_delg  C$res]]
             }
           }
           set dgtmp1 [format {%5.2f} $dgtmp1]

           if { $dgtmp1 == "-0.00" } {
              set dgtmp " [string range $dgtmp1 1 end]"
           } else {
              set dgtmp $dgtmp1
           }
           
           lappend dgq($res) [format %7.2f $dgtmp]
                      
           if { $include_residue_in_profile == 0 && [lsearch $list_not_to_plot $res] == "-1" } {
               set ddgtotal_without_exluded($ph) [expr $ddgtotal_without_exluded($ph) + $dgtmp]
           }
           set dgq_total($ph) [expr $dgq_total($ph) + $dgtmp]
          
      }
      
            
      if {$make_a_plot == 1} {
         if { $plotting_started == 0 } {
             set plothandle [multiplot -x $pHdata -y $dgq($res) -title $title -lines -linewidth 3 -marker point -plot \
                 -xlabel pH -ylabel $ylabel -legend "$res"]
             set plotting_started 1
         } else {
            $plothandle add $pHdata $dgq($res) -marker point -legend "$res" -linewidth 3 -plot
         }
         
      }
      puts $pH_file "[format %4s $abbrevname($res)] [format %15s $res] [join $dgq($res) "   "]"
       
   }
   

   if {$make_a_plot == 0 } { 
      for {set ph $minpH} { $ph <= $maxpH} {set ph [expr $ph + $delpH]} {
          lappend total($currmol) "[format %7.2f $dgq_total($ph)]"
          if {$include_residue_in_profile == 0} {
             lappend total_exl [format %6.2f $ddgtotal_without_exluded($ph)]
          }


      } 
      
      puts $pH_file ""
      puts $pH_file "            Total    [join $total($currmol) "   "]"

      if {$include_residue_in_profile == 0} {
         puts $pH_file "\n          Tot-NoSel  [join $total_exl "   "]"
         puts $pH_file "\n     Exluded residues are: $list_not_to_plot"
         
         set plothandle [multiplot -x $pHdata -y $total($currmol) -title "$title" -lines -linewidth 3 -marker point -plot \
                 -xlabel pH -ylabel $ylabel -legend Total]
         $plothandle add $pHdata $total_exl -marker point -legend "Total without selected residues" -linewidth 3 -plot
      }
      puts "\npH profile has been written out"
      

   }
   
   set ph $pH_orig
   close $pH_file
  }
  
  
   if {[llength $molid_to_plot_for] > 1} {
     set compare_profile_data ""
     lappend compare_profile_data "                    pH/$delgq  [join $pHdata_formatted "   "]\n\n"
     set molplotted 0     
     foreach currmol $molid_to_plot_for {
       incr molplotted
       set name [molinfo [string range $currmol 1 end] get name]
       lappend compare_profile_data  "[format %25s $name]  [join $total($currmol) "   "]\n"
       if {$molplotted == 1} {
           set plothandle [multiplot -x $pHdata -y $total($currmol) -title "$title" -lines -linewidth 3 -marker point -plot \
                 -xlabel pH -ylabel $ylabel -legend $name ]
       } else {
            $plothandle add $pHdata $total($currmol) -marker point -legend $name -linewidth 3 -plot
       }
       
   }
   
   set profile_compare_file [open profilecombined$chargename.txt w]  
   foreach line $compare_profile_data {
        puts $profile_compare_file $line
    }
     
    
   close $profile_compare_file
   
  }
  

}



proc ::PropKa::propka_check_ver {} {

    variable pkafile 
    
    set outfile [open $pkafile r]
    
    set line [gets $outfile]
    
    set pkaftype 2
    
    if {[ regexp -- "propka3" $line ]} {
        set pkaftype 3
    }
    
    puts "Output File from PROPKA $pkaftype found" 
    

    close $outfile
    
    return $pkaftype
    
}

proc ::PropKa::propka_results {} {


    variable plotener
    variable plotcharge
    variable showpdb
    variable pkafile
    variable pdbfile
    variable pkaftype
    variable loadpdb
    variable showallpkares
    variable showallpkalabels
    variable pdbloaded 0
    
    variable coloredlabels
    variable labelsshifted
    
## load the appropiate pdb files
        set molidtoload [expr [lindex [molinfo list] end] + 1]
        

        if { $molidtoload > 7 } {
             set labelsshifted 0   
         }

            
        set pkaftype_tmp [::PropKa::propka_check_ver]
        
        
        if { $loadpdb == 1 } {
            if { $pkaftype_tmp == 3 } {
                mol new [file rootname $pkafile].pdb
                set pdbloaded 1
            } else {
                mol new [file rootname $pkafile]
                set pdbloaded 1
            }

        }
        
        
        set pkaftype(m[molinfo top]) $pkaftype_tmp
        
        mol off all
        mol on top
        mol showrep [molinfo top] 0 off
        if {$showallpkares != 1} {
            mol representation NewRibbons
            mol addrep top
        }
        
## read the PROPKA result file

    if { $pkaftype_tmp == 3 } {
        ::PropKa::propka3_read
    } else {
        ::PropKa::propka_read
    }
    
    
## plot energy 

    if {$plotener == 1} {
        ::PropKa::propka_plotener
    }

## plot charge for folded and unfolded states vs. pH
    if {$plotcharge == 1} {
        ::PropKa::propka_plotchrg
    }
    
    if { $showallpkalabels == 1 } { 
        set showallpkares 1
    }
    
    ::PropKa::propka_show_all

    ::PropKa::propka_shifted_pkas
    
    ::PropKa::picking

}



proc ::PropKa::picking {} { 
    
    global vmd_pick_atom
    
    variable tracestatus
    variable pickmode
 
    if {$pickmode != 0 && $tracestatus == 0} {
        trace add variable vmd_pick_atom write ::PropKa::propka_contrib_fctn
        set tracestatus 1
    }
    
    if {$pickmode == 0 && $tracestatus == 1} {
        trace remove variable vmd_pick_atom write ::PropKa::propka_contrib_fctn
        set tracestatus 0
    }
}



proc ::PropKa::propka3_read {} {
   variable phdata;          # pH data storage for plotting
   variable foldener;        # Fold Energy storage for plotting
   variable phchrg;          # pH data storage for plotting charge dependence
   variable unfoldchrg;      # Unfolded state charge storage for plotting
   variable foldchrg;        # Folded state charge storage for plotting
   variable chainlist;       # Chain list, if avaialable
   variable resinfo;         # Residue description
   variable reslist;         # List of residues.
   variable chainsexist
   variable residlist
   variable pka
   variable pkaprec
   variable name 
   variable modelpka
   variable shiftpka
   variable abbrevname
   variable loadpdb
   variable loadedpkafiles
   variable pkafile
   
## Variables for the second reading. 
    variable sidehb
    variable backhb
    variable ccinteract
    variable description
    variable totaldesolv
    variable totaldesolv_prec
    variable infos
    variable readtyr
    variable readcystyr
    
    variable consider_coupled
  
    set currmol m[molinfo top]
    set chainsexist($currmol) ""
    set chainlist($currmol) ""
    
    set resinfo($currmol) ""

    set phdata($currmol) ""
    set phchrg($currmol) ""
    
    set chainid ""
    set coupled_found 0
    set RE_contrib_found 0
    

    variable namelist 
    variable abbrevlist
    
    
    # Ligands-related variables:
    variable ligandsnames_found
    variable ligands_exist
    
    if { [molinfo num] == 0 } {
        return
    }


    set outfile [open $pkafile r]
    
    set ligandsnames_found($currmol) ""
    set non_standard_format 0
    while {![eof $outfile]} {
        set line [gets $outfile]
        if {[ regexp -- "ligand atom-type" $line ]} {
            set line_orig ""
            while {![regexp {\-\-} $line_orig]} {
                set line_orig [gets $outfile]
             if {[string index $line_orig 3] != " " && [string index $line_orig 3] != "-"} { 
                set line $line_orig
                if { [llength [string range $line_orig 0 12]] != 3 } {
                    set non_standard_format 1
                    set line "[lindex [string range $line 3 5] 0] [lindex [string range $line 6 10] 0] [string index $line 11] [lindex [string range $line 13 22] 0] [lindex [string range $line 25 32] 0] [lindex [string range $line 44 52] 0]"
                }
                    if {[regexp {CYS|TYR|ASP|GLU|LYS|ARG|HIS|ASX|GLX|N\+|C\-} [lindex $line 0]]} {
           
                        set id [lindex $line 1 ]
                        # Separate N+ and C- contributions                   
                  
                        set idpure $id
                    
                        if { [lindex $line 0] == "N+" } {
                            set idpure $id
                            set id N$id
                        }
                        
                        if { [lindex $line 0] == "C-" } {
                           set idpure $id
                           set id C$id
                        }
                    
                    
                        set chainidtmp [lindex $line 2]
                    
                        if { $chainidtmp == "X" } {
                            set chainid ""
                        } else {
                            if { $chainidtmp != $chainid && [lsearch $chainlist($currmol) $chainidtmp] == "-1"} {
                                lappend chainlist($currmol) $chainidtmp
                            }
                            set chainid $chainidtmp
                        }
                    

                        set resid $id$chainid$currmol
                    
                        set name($resid) [lindex $line 0]
                        
                        set abbrevname($resid) [lindex $abbrevlist [lsearch $namelist $name($resid)]]
                        set tmp_description $abbrevname($resid)$idpure
                    } else {
                   
                        set ligandname [lindex $line 0]
                        
                        set chaintmpsel ""
                        
                        set chainid [lindex $line 2]
                        
                        if { $chainsexist($currmol) == 1 } {
                            set chaintmpsel " and chain $chainid"
                        }
                        set ligandatomnametmp [lindex $line 1]
                        set tmp_description $ligandname\-$ligandatomnametmp
                        
                        if { [lsearch $ligandsnames_found($currmol) $ligandname] == "-1" } {
                            lappend ligandsnames_found($currmol) $ligandname
                            set ligands_exist 1
                        }

                        set ligandtmpsel [atomselect top "resname $ligandname and name $ligandatomnametmp$chaintmpsel"]
                        set ligandinfo [$ligandtmpsel get "resid"]

                        set idpure [lindex $ligandinfo 0]
                        
                        set id $tmp_description\_$idpure
                    
                        set resid $id$chainid$currmol
                    
                        set name($resid) [lindex $line 0]
                        
                        set abbrevname($resid) ""

                    }

                    set pka($resid) [format {%3.1f} [lindex $line 3]]
                    set pkaprec($resid) [lindex $line 3]
                    set modelpka($resid) [lindex $line 4]
                    set shiftpka($resid) [format {%3.1f} [expr $pkaprec($resid) - $modelpka($resid)]]
                    set shiftpka_abs($resid) [expr abs($shiftpka($resid))]
                    
                    if { [llength $chainlist($currmol)] > 1 && $chainid != [lindex $chainlist($currmol) 0] } {
                        set description($resid) "$tmp_description\($chainid\)"
                    } else {
                        set description($resid) "$tmp_description"
                    }
                    
                    
                    if { [llength $chainlist($currmol)] > 1 } {            
                        set chainsexist($currmol) 1
                    } elseif { [llength $chainlist($currmol)] == 1 } {
                        set chainsexist($currmol) 0
                    }


                    ##Exclude Tyrosines and Cysteines, and wrong pKa values (only those higher than 30 units)

                    if { $readtyr == 1 } {
                        set residchoice CYS
                    } elseif { $readcystyr == 1 } {
                        set residchoice Z
                    } else {
                        set residchoice CYS|TYR
                    }
                    
                    if { ![regexp -- "$residchoice" $name($resid)] } {
                        if { $pka($resid) < 30 } {
                            lappend reslist($chainid$currmol) $id
                            lappend residlist($currmol) $resid
                            lappend resinfo($currmol) "$description($resid) $pka($resid) $shiftpka($resid) $shiftpka_abs($resid) $name($resid) $id $chainid"
                        }
                    }
                }
            }
        }
        
        
        if {[ regexp -- "Free energy of   folding" $line ]} {
            for {set i 0} {$i < 14} {incr i} {
                set ph ""
                regexp { +([0-9]+.[0-9]+) +(\-?[0-9]+.[0-9]+)} [gets $outfile] a ph fen 
                if { $ph != "" } {
                   lappend phdata($currmol) $ph
                   lappend foldener($currmol) $fen
                }
            }
        }
        

       
        if {[ regexp -- "pH  unfolded  folded" $line ]} {
            for {set i 0} {$i < 14} {incr i} {
                set ph ""
                regexp { +([0-9]+.[0-9]+) +(\-?[0-9]+.[0-9]+) +(\-?[0-9]+.[0-9]+)} [gets $outfile] a ph unfoldcharge foldcharge
                if { $ph != "" } {
                    lappend phchrg($currmol) $ph
                    lappend unfoldchrg($currmol) $unfoldcharge
                    lappend foldchrg($currmol) $foldcharge
                }
            }
        }
        
    }
        
    
    if { [llength $chainlist($currmol)] > 1 } {
        puts "More than one chain recognized. Chain description will be used for residues"
        if { $infos != 0 } {
            puts ""
            puts "Chains found: $chainlist($currmol)"
            puts ""
            foreach chain $chainlist($currmol) { 
                puts "Chain $chain residues:"
                puts $reslist($chain$currmol)
                puts ""
            }
            puts $resinfo($currmol)
        }
    } else { 
        puts "No multiple chains found"
        if { $infos != 0 } {
            puts $reslist($chainid$currmol)
            puts $resinfo($currmol)
            puts ""
        }
    }

    close $outfile

    
    set line_orig ""  
## Reopening propKa results file and second (full description) reading
    set outfile [open $pkafile r]
    while {![regexp -- "SUMMARY OF " $line_orig]} {
        set line_orig [gets $outfile]
        if {[string index $line_orig 16] == "*" } {
           set coupled_found 1
        }
        
        if {[regexp -- "BURIED     REGULAR      RE " $line_orig]} {
            set RE_contrib_found 1
        }
        if { [string index $line_orig 23] == "%" } {

            set line $line_orig
            if { $non_standard_format == 1 } {
            set line "[lindex [string range $line 0 2] 0] [lindex [string range $line 3 7] 0] [string index $line 8] [lindex [string range $line 10 16] 0] [lindex [string range $line 19 22] 0]  [string index $line 23] [lindex [string range $line 26 32] 0] \
                        [lindex [string range $line 33 37] 0] [lindex [string range $line 39 44] 0] [lindex [string range $line 46 49] 0]\
                        [lindex [string range $line 51 56] 0] [lindex [string range $line 58 60] 0] [lindex [string range $line 61 64] 0] [string index $line 66]\
                        [lindex [string range $line 69 74] 0] [lindex [string range $line 76 78] 0] [lindex [string range $line 79 83] 0] [string index $line 84]\
                        [lindex [string range $line 87 92] 0] [lindex [string range $line 93 96] 0] [lindex [string range $line 97 101] 0] [string index $line 102]"
            }
            if {[regexp {CYS|TYR|ASP|GLU|LYS|ARG|HIS|ASX|GLX|N\+|C\-} [lindex $line 0]] } {
                set chainid [lindex $line 2]
                set tmpname [lindex $line 0]
                set tmpid [lindex $line 1]
                
                if { $tmpname == "N+" } {
                    set tmpid N$tmpid
                }
                
                if { $tmpname == "C-" } {
                    set tmpid C$tmpid
                }
                    
             } elseif { [string index $line_orig 0] !=  "-" } {
             
                set chainid [lindex $line 2]

                set tmpname [lindex $line 0]
             
                set tmp_ligandatomnane [lindex $line 1]
             
                set ligand $tmpname\-$tmp_ligandatomnane
                if { $chainsexist($currmol) == 1 } {
                    set ligandtmpsel [atomselect top "resname $tmpname and name $tmp_ligandatomnane and chain $chainid"]
                } else {
                    set ligandtmpsel [atomselect top "resname $tmpname and name $tmp_ligandatomnane"]
                }
                
                set ligands_exist 1
                set tmppureid [$ligandtmpsel get "resid"]
                
                if { [llength $tmppureid] > 1 } {
                
                }       
         
                set tmpid $ligand\_$tmppureid
            
             }
             
             set resid $tmpid$chainid$currmol
            
             set locate($resid) [lindex $line 4]
             set desolvefs($resid) [lrange $line 6 9]

             if { $RE_contrib_found == 1 } {
                set totaldesolv_prec($resid) [lindex $desolvefs($resid) 0]
             } else {
                set totaldesolv_prec($resid) [expr [lindex $desolvefs($resid) 0] + [lindex $desolvefs($resid) 2]]
             }
             set totaldesolv($resid) [format {%3.1f} $totaldesolv_prec($resid)]

             set sidehb($resid) [list [::PropKa::propka_getLigandID $resid [lrange $line 10 13]]]
             set backhb($resid) [list [::PropKa::propka_getLigandID $resid [lrange $line 14 17]]]
             set ccinteract($resid) [list [::PropKa::propka_getLigandID $resid [lrange $line 18 21]]]
           
             while { $line_orig != "" } {
                set line_orig [gets $outfile]
                
                set line $line_orig
                if { $non_standard_format == 1 } {
                    set line "[lindex [string range $line 0 2] 0] [lindex [string range $line 3 7] 0] [string index $line 8]\
                        [lindex [string range $line 51 56] 0] [lindex [string range $line 58 60] 0] [lindex [string range $line 61 64] 0] [string index $line 66]\
                        [lindex [string range $line 69 74] 0] [lindex [string range $line 76 78] 0] [lindex [string range $line 79 83] 0] [string index $line 84]\
                        [lindex [string range $line 87 92] 0] [lindex [string range $line 93 96] 0] [lindex [string range $line 97 101] 0] [string index $line 102]"
                }
                
                if {[llength $line_orig] > 0 } {
                    if { [lindex $line 4] != "XXX" } {
                        lappend sidehb($resid) [::PropKa::propka_getLigandID $resid [lrange $line 3 6]]                 
                    }
            
                    if { [lindex $line 8] != "XXX" } {
                        lappend backhb($resid) [::PropKa::propka_getLigandID $resid [lrange $line 7 10]]
                    }
                
                    if { [lindex $line 12] != "XXX" } {
                        lappend ccinteract($resid) [::PropKa::propka_getLigandID $resid [lrange $line 11 14]]
                    }
                }
             }
        }
    }
    
    
  puts "PROPKA Results File Was Read In"
  if { $ligandsnames_found($currmol) != "" } {
     variable ligands_found_for
     lappend ligands_found_for [file rootname [file tail $pkafile]]   
  }
  
  if { $coupled_found == "1" &&  $consider_coupled == 1 } {
     variable coupled_found_for
     lappend coupled_found_for [file rootname [file tail $pkafile]]
  }

  lappend loadedpkafiles $currmol

}



proc ::PropKa::propka_getLigandID { resid determinfo } {

    
    variable namelist
    variable ligands_exist
    variable chainlist

    set tmpname [lindex $determinfo 1]
    set resinfo [::PropKa::propka_get_sel $resid]
    set mol [lindex $resinfo 2]
    set tmpid [lindex $determinfo 2]

    if {[lsearch $namelist $tmpname] != "-1" || $tmpname == "XXX" } {
       return $determinfo
    } elseif {[string length $tmpname] == 2 && $tmpid < 999999} {
       set ligands_exist 1
       return "[lrange $determinfo 0 1] $tmpname [lindex $determinfo 3]\_$tmpid"
    } else {
       set tmpchain [lindex $determinfo 3]
       set tmpchain_seltext ""
       if { [llength $chainlist(m$mol)] > 1 } {
            set tmpchain_seltext " and chain $tmpchain"
       } 
       set determsel [atomselect $mol "resname $tmpname and name $tmpid$tmpchain_seltext"]
       set determresid_s [$determsel get resid]
       set ligands_exist 1
       if {[llength $determresid_s] == 1 } { 
          return "[lrange $determinfo 0 2] [lindex $determinfo 3]\_[lindex $determresid_s 0]" 
        } else {
          set dist 20
          set sel [atomselect $mol [lindex $resinfo 0]]
          set coor [measure center $sel]
          
          foreach id $determresid_s { 
            set determsel_2 [atomselect $mol "resname $tmpname and name $tmpid and chain $tmpchain and resid $id"]
            set determcoor [measure center $determsel_2]
            set tmpdist [expr sqrt(pow(([lindex $coor 0] - [lindex $determcoor 0]),2) + pow(([lindex $coor 1] - [lindex $determcoor 1]),2) + pow(([lindex $coor 2] - [lindex $determcoor 2]),2))] 
            if { $tmpdist < $dist } {
               set dist $tmpdist
               set closest_resid $id
            }
          }
          return "[lrange $determinfo 0 2] [lindex $determinfo 3]\_$closest_resid" 
       }
    }
}

proc ::PropKa::propka_read {} {
   variable phdata;          # pH data storage for plotting
   variable unfoldener;      # Unfold Energy storage for plotting

   variable phchrg;          # pH data storage for plotting charge dependence
   variable unfoldchrg;      # Unfolded state charge storage for plotting
   variable foldchrg;        # Folded state charge storage for plotting
   variable chainlist;       # Chain list, if avaialable
   variable resinfo;         # Residue description
   variable reslist;         # List of residues.
   variable chainsexist
   
   
   variable pka
   variable pkaprec
   variable name 
   variable modelpka
   variable shiftpka
   variable abbrevname

   variable loadpdb
   variable loadedpkafiles

   variable pkafile
   
## Variables for the second reading.
    variable sidehb
    variable backhb
    variable ccinteract
    variable description
    variable totaldesolv
    variable totaldesolv_prec
    
    variable infos
    variable readtyr
    variable readcystyr
    
    variable residlist

    set currmol m[molinfo top]

## Setting some initial values
    set chainsexist($currmol) ""
    set chainlist($currmol) ""
    
    set resinfo($currmol) ""

    set phdata($currmol) ""
    set phchrg($currmol) ""
    
    set chainid ""
    
    variable namelist
    variable abbrevlist
    
    
    variable ligandsnames_found; # Just Ligands Resnames
    variable ligands_exist;
    

    if { [molinfo num] == 0 } {
        return
    }

 
## Propka File Test
    set test1 0
    set test_pkaresults 0
    set outfile [open $pkafile r] 
    for {set i 1} {$i < 10} {incr i} {
            set line [gets $outfile]
            if { [ regexp -- "PROPKA: A PROTEIN PKA PREDICTOR" $line ] } {
                set test1 1 
            }
            if { [ regexp -- "Missing side-chain" $line ] } {
               set test_pkaresults 1
            }
    }
    
    if {$test1 != 1} {
      tk_dialog .errmsg {PropKa Warning} "pKa results may inlcude ligand(s)" error 0 Dismiss
    }
    
    set ligandsnames_found($currmol) "" 
    
    set warning_shown 0
    
    while {![eof $outfile]} {
        set line [gets $outfile]
        if {[ regexp -- "RESIDUE    pKa   pKmodel" $line ]} {
            while {![regexp {\-\-} $line]} {
                set line [gets $outfile]
                if {[regexp {CYS|TYR|ASP|GLU|LYS|ARG|HIS|ASX|GLX|N\+|C\-} [lindex $line 0]]} {
                   if {[string index $line 10] != " " && [string index $line 10] != "X"} {
                       set chaintmp [string index $line 10]
                       if {$chaintmp != $chainid && [lsearch $chainlist($currmol) $chaintmp] == "-1"} {
                          lappend chainlist($currmol) $chaintmp

                       }
                       set chainid $chaintmp
                   }

                  set id [lindex [string range $line 7 9] 0 ]

                  set idpure $id
#          Separate N+ and C-                     
                   if { [lindex [string range $line 3 5] 0] == "N+" } {
                       set idpure $id
                       set id N$id
                   }              
                   if { [lindex [string range $line 3 5] 0] == "C-" } {
                       set idpure $id
                       set id C$id
                   }
                    
                   set resid $id$chainid$currmol
                    

                   if { $infos != 0 } {
                       puts $resid
                   }
                    

                   set name($resid) [lindex [string range $line 3 5] 0]
                
                   set pka($resid) [format {%3.1f} [lindex [string range $line 13 17] 0]]
                   set pkaprec($resid) [lindex [string range $line 13 17] 0]
                   set modelpka($resid) [lindex [string range $line 23 27] 0 ]
                   set shiftpka($resid) [format {%3.1f} [expr $pkaprec($resid) - $modelpka($resid)]]
                   set shiftpka_abs($resid) [expr abs($shiftpka($resid))]
                   set abbrevname($resid) [lindex $abbrevlist [lsearch $namelist $name($resid)]]
                   if { [llength $chainlist($currmol)] > 1 && $chainid != [lindex $chainlist($currmol) 0] } {
                      set description($resid) "$abbrevname($resid)$idpure\($chainid\)"
                   } else { 
                      set description($resid) $abbrevname($resid)$idpure
                   }
                   
                   if { [llength $chainlist($currmol)] > 1 } { 
                      set chainsexist($currmol) 1
                   } elseif { [llength $chainlist($currmol)] == 1 } {
                      set chainsexist($currmol) 0
                   }

                   lappend reslist($chainid$currmol) $id
                   lappend residlist($currmol) $resid
                    
                    ##Exclude Tyrosines and Cysteines and wrong pKa values (only those higher than 30 units)
                    if { $readtyr == 1 } {
                        set residchoice CYS
                    } elseif { $readcystyr == 1 } {
                        set residchoice Z
                    } else {
                        set residchoice CYS|TYR
                    }
                    
                    if { ![regexp -- "$residchoice" $name($resid)] } {
                        if { $pka($resid) < 30 } {
                            lappend resinfo($currmol) "$description($resid) $pka($resid) $shiftpka($resid) $shiftpka_abs($resid) $name($resid) $id $chainid"
                        }
                    }
                # Creating pKa data for ligands    
                } elseif  { [string index $line 3] != " " && [string index $line 0] == " " } {
                    set ligandnametmp [lindex $line 0]
                    set ligandname [string range $ligandnametmp 0 2]
                    set chaintmpsel ""
                    
                    if { $chainsexist($currmol) == 1 } {
                       set chainid [string index $ligandnametmp end]
                       if { [string length $ligandnametmp] == 4 } {
                          set chaintmpsel " and chain $chainid"
                       } elseif { $warning_shown == 0 }  {
                          tk_dialog .errmsg {PropKa Warning} "Cannot Assign Chain For Ligands" error 0 Dismiss
                          set warning_shown 1    
                       } 
                    }
                    
                    set ligandatomnametmp [lindex $line 1]
                    if { [lsearch $ligandsnames_found($currmol) $ligandname] == "-1" } {
                        lappend ligandsnames_found($currmol) $ligandname
                        set ligands_exist 1
                    }
                    set ligandtmpsel [atomselect top "resname $ligandname and name $ligandatomnametmp$chaintmpsel"]
                    set ligandinfo [$ligandtmpsel get "resid"]
                    set idpure [lindex $ligandinfo 0]
                    set id $ligandname\-$ligandatomnametmp\_$idpure
                    set resid $id$chainid$currmol
                    
                    if { $infos != 0 } {
                        puts "Ligand's RESID is: $resid"
                    }
                   
                    set name($resid) $ligandatomnametmp
                 
                    set pka($resid) [format {%3.1f} [lindex [string range $line 13 17] 0]]
                    set pkaprec($resid) [lindex [string range $line 13 17] 0]
                    set modelpka($resid) [lindex [string range $line 23 27] 0 ]
                    set shiftpka($resid) [format {%3.1f} [expr $pkaprec($resid) - $modelpka($resid)]]
                    set shiftpka_abs($resid) [expr abs($shiftpka($resid))]
                    set abbrevname($resid) ""
                    set description($resid) $ligandname\-$ligandatomnametmp
                    
                    lappend reslist($chainid$currmol) $id 
                    lappend residlist($currmol) $resid

                    if { $pka($resid) < 30 } {
                       lappend resinfo($currmol) "$description($resid) $pka($resid) $shiftpka($resid) $shiftpka_abs($resid) $name($resid) $id $chainid"
                    }
                }
            }
        }


        ##Reading unfolding energy vs pH
        if {[ regexp -- "Free energy of unfolding" $line ]} {
            for {set i 0} {$i < 14} {incr i} {
                regexp { ?([0-9]+.[0-9]+) +(\-?[0-9]+.[0-9]+)} [gets $outfile] a ph unfen 
                lappend phdata($currmol) $ph
                lappend unfoldener($currmol) $unfen
            }
        }



        ##Reading charges vs pH
        if {[ regexp -- "Protein charge of folded and unfolded" $line ]} {
            for {set i 0} {$i < 14} {incr i} {
                regexp { ?([0-9]+.[0-9]+) +(\-?[0-9]+.[0-9]+) +(\-?[0-9]+.[0-9]+)} [gets $outfile] a ph foldcharge unfoldcharge
                lappend phchrg($currmol) $ph
                lappend unfoldchrg($currmol) $unfoldcharge
                lappend foldchrg($currmol) $foldcharge
            }
        }
    }
        

    if { $chainsexist($currmol) == 1 } {
        puts "More than one chain recognized. Chain description will be used for residues"
        if { $infos != 0 } {
            puts ""
            puts "Chains found: $chainlist($currmol)"
            puts ""
            foreach chain $chainlist($currmol) { 
                puts "Chain $chain residues:"
                puts $reslist($chain$currmol)
                puts $resinfo($currmol)
                puts ""
            }
        }
    } else { 
        puts "No multiple chains found"
        if { $infos != 0 } {
            puts $reslist($chainid$currmol)
            puts $resinfo($currmol)
            puts ""
        }
    }



    close $outfile
    
    set acidschaincount 0
    set acidsid 0
    set hischaincount 0
    set cyschaincount 0
    set tyrchaincount 0
    set lyschaincount 0
    set argchaincount 0
    set ntermchaincount 0
    set hisid 0
    set cysid 0
    set tyrid 0
    set lysid 0
    set argid 0
    set ntermid 0
    

## Reopening propKa results file and second (full description) reading
    set outfile [open $pkafile r]
    while {![regexp -- "SUMMARY OF " $line]} {
        set line [gets $outfile]
        if {[regexp {BURIED|SUFACE|BONDED} $line] } { 
         if { [regexp {ALA|ARG|ASN|ASP|CYS|GLN|GLU|GLY|HIS|ILE|LEU|LYS|MET|PHE|PRO|SER|THR|TRP|TYR|VAL|ASX|GLX|N\+|C\-} [lindex $line 0]]} {
            set testchain 0
            set tmpchain ""
            set tmpname [lindex $line 0]
            set tmpid [lindex $line 1]
            set tmppka [lindex $line 2]
            
## Chain recognition for the acidic residues
            if { $chainsexist($currmol) == 1 } {
                if {[regexp {ASP|GLU|C\-} $tmpname] } {
                    if { $acidsid >= $tmpid } {
                        incr acidschaincount
                    }
                    set acidsid $tmpid
                    set tmpchain [lindex $chainlist($currmol) $acidschaincount]
                }
                
## Chain recognition for the basis residues
                if {[regexp {HIS} $tmpname]} {
                    if { $hisid >= $tmpid } {
                        incr hischaincount
                    }
                    set hisid $tmpid
                    set tmpchain [lindex $chainlist($currmol) $hischaincount]
                }
                
                if {[regexp {CYS} $tmpname]} {
                    if { $cysid >= $tmpid } {
                        incr cyschaincount
                    }
                    set cysid $tmpid
                    set tmpchain [lindex $chainlist($currmol) $cyschaincount]
                }
                
                if {[regexp {TYR} $tmpname]} {
                    if { $tyrid >= $tmpid } {
                        incr tyrchaincount
                    }
                    set tyrid $tmpid
                    set tmpchain [lindex $chainlist($currmol) $tyrchaincount]
                }
                
                if {[regexp {LYS} $tmpname]} {
                    if { $lysid >= $tmpid } {
                        incr lyschaincount
                    }
                    set lysid $tmpid
                    set tmpchain [lindex $chainlist($currmol) $lyschaincount]
                }
                
                if {[regexp {ARG} $tmpname]} {
                    if { $argid >= $tmpid } {
                        incr argchaincount
                    }
                    set argid $tmpid
                    set tmpchain [lindex $chainlist($currmol) $argchaincount]
                }
                
                if {[regexp {N\+} $tmpname]} {
                    if { $ntermid >= $tmpid } {
                        incr ntermchaincount
                    }
                    set ntermid $tmpid
                    set tmpchain [lindex $chainlist($currmol) $ntermchaincount]
                }
                
            }

            if { $chainsexist($currmol) == 0 } {
                set tmpchain $chainlist($currmol)
            }
            
            # Different treatment for the N+ and C-
             if { $tmpname == "N+" } {
                set tmpid N$tmpid
            }
                
            if { $tmpname == "C-" } {
               set tmpid C$tmpid
            }
          # Reading data for the ligands         
          } elseif { [string index $line 0] !=  "-" && [string index $line 0] !=  "" } {
             
             set tmpchain ""
             if { $chainsexist($currmol) == 0 } {
                set tmpchain $chainlist($currmol)
             }
             set ligands_exist 1
             set tmpname [lindex $line 0]
             
             set tmp_ligandatomnane [lindex $line 1]
             
             set ligand $tmpname\-$tmp_ligandatomnane
             if { $chainsexist($currmol) == 1 } {
                lappend ligands_loaded $ligand
                set chainindex [expr [llength [lsearch -all $ligands_loaded $ligand]] - 1]
                set tmpchain [lindex $chainlist($currmol) $chainindex]
                set ligandtmpsel [atomselect top "resname $tmpname and name $tmp_ligandatomnane and chain $tmpchain"]
             } else {
                set ligandtmpsel [atomselect top "resname $tmpname and name $tmp_ligandatomnane"]
             }
             
             set tmppureid [$ligandtmpsel get "resid"]
             set tmpid $ligand\_$tmppureid
             
          }
            set resid $tmpid$tmpchain$currmol
            
            
            set locate($resid) [lindex $line 3]
            set desolvefs($resid) [string range $line 26 46]
            set totaldesolv_prec($resid) [expr [lindex $desolvefs($resid) 0] + [lindex $desolvefs($resid) 2]]
            set totaldesolv($resid) [format {%3.1f} $totaldesolv_prec($resid)]

            set sidehb($resid) [list [::PropKa::propka_add_chain_for_determs $resid [string range $line 50 62]]]
            set backhb($resid) [list [::PropKa::propka_add_chain_for_determs $resid [string range $line 66 78]]]
            set ccinteract($resid) [list [::PropKa::propka_add_chain_for_determs $resid [string range $line 82 95]]]

            while { $line != "  " } {
                set line [gets $outfile]
                
                if { ![ regexp -- "0.00|    " [string range $line 51 54]] && [string range $line 51 54] != "" } {
                    lappend sidehb($resid) [::PropKa::propka_add_chain_for_determs $resid [string range $line 50 62]]
                }
            
                if { ![ regexp -- "0.00|    " [string range $line 67 70]] && [string range $line 67 70] != "" } {
                    lappend backhb($resid) [::PropKa::propka_add_chain_for_determs $resid [string range $line 66 78]]
                }
                
                if { ![ regexp -- "0.00|    " [string range $line 83 86]] && [string range $line 83 86] != "" } {
                    lappend ccinteract($resid) [::PropKa::propka_add_chain_for_determs $resid [string range $line 82 95]]
                }
            }
        }
    }
    
    if { $test_pkaresults  == 1} { 
        tk_dialog .errmsg {PropKa Warning} "Missing side-chain atoms have been reported\nPlease check the PROPKA output file" error 0 Dismiss
    }
    
    puts "PROPKA Results File Was Read In"
    lappend loadedpkafiles $currmol
     
}

proc ::PropKa::propka_add_chain_for_determs { resid determinfo } {

    variable chainlist
    variable infos
    variable namelist
    variable ligands_exist
   
    set resinfo [::PropKa::propka_get_sel $resid]
    set mol [lindex $resinfo 2]
    set tmpid [lindex $determinfo 2]
    set tmpname [lindex $determinfo 1]
    
    set chain 0
    if {$tmpname == "000"} { 
       return "$determinfo 0"
    }

    if { $tmpname == "N+" || $tmpname == "C-" } {
       set tmp_sel [atomselect $mol "resid $tmpid and name CA"]
       set tmpname [lindex [$tmp_sel get resname] 0]
    }
    
    set sel [atomselect $mol [lindex $resinfo 0]]

    set dist 20

    lassign [$sel get {x y z}] coor
    if {[lsearch $namelist $tmpname] == "-1"} {
        set ligand 1
        set sel_text "resname $tmpname and name $tmpid"
        set ligands_exist 1
    } else {
        set ligand 0
        set sel_text "resid $tmpid and name CA"
    }
    
    foreach tmpchain $chainlist(m$mol) {
        if {[llength $chainlist(m$mol)] > 1 } {
            set sel_text1 "$sel_text and chain $tmpchain"
        } else {
            set sel_text1 $sel_text
        }
        
        set determsel [atomselect $mol "$sel_text1"]
        set determname [lindex [$determsel get resname] 0]
        set determresid_s [$determsel get resid]

        foreach id $determresid_s {
            if { $determname == $tmpname } {
                set determsel_2 [atomselect $mol "$sel_text1 and resid $id"]
                set determcoor [measure center $determsel_2]
                set tmpdist [expr sqrt(pow(([lindex $coor 0] - [lindex $determcoor 0]),2) + pow(([lindex $coor 1] - [lindex $determcoor 1]),2) + pow(([lindex $coor 2] - [lindex $determcoor 2]),2))] 
                if { $tmpdist < $dist } {
                    set chain $tmpchain
                    set dist $tmpdist
                    set closest_resid $id
                }
            }
        }
    }

    if { $ligand == 1 } {
        set chain $chain\_$closest_resid
    }
    return "$determinfo $chain"
    
}



proc ::PropKa::propka_get_sel { resid } {

    variable chainlist 

    set id [split $resid m]
          
    set tmpmol [lindex $id 1]
    
    set res [lindex $id 0]
    set ligand 0
    set ligand_selection ""
    if {[string first "-" $res] != "-1" } {
        set ligand 1    
        set ligandinfo [split $res "_"]
        set ligandnames [split [lindex $ligandinfo 0] "-"]
        
        set ligand_selection " and resname [lindex $ligandnames 0]"
        set res [lindex $ligandinfo 1]
        
    } elseif {[string index $res 0] == "N" || [string index $res 0] == "C"} {
        set res [string range $res 1 end]
    }
    
    if { [llength $chainlist(m$tmpmol)] >  1 } {
       set chain [string index $res end]
       set id [string range $res 0 end-1]
               
       set selection "resid $id and chain $chain$ligand_selection"
    } elseif {[llength $chainlist(m$tmpmol)] ==  1} {
       set id [string range $res 0 end-1]
       set selection "resid $id$ligand_selection"
       set chain ""
    } else {
       set id $res
       set selection "resid $id$ligand_selection"
       set chain ""
    }  

    return "{$selection} $id $tmpmol $chain"
}



proc ::PropKa::propka_plotener {} {
## Plotting energy vs. pH using multiplot VMD plugin
    variable phdata
    variable unfoldener

    variable foldener
    variable plotener
    variable infos
    variable loadedpkafiles
    variable pkaftype
    
    if { [molinfo num] == 0 } {
        return
    }
    
    set currmol m[molinfo top]
    

    set plotener 0
    

    if { [lsearch $loadedpkafiles $currmol ] == "-1" } {
       return
    }
    
    
    if {$phdata($currmol) == "" } {
        tk_dialog .errmsg {PropKa Warning} "Missing energy profile. \nCheck PROPKA calculations" error 0 Dismiss
        return 
    }

    if { $pkaftype($currmol) == 3 } {
        set energies $foldener($currmol)
        set plottitle "Free Energy of Folding"
    } else {
        set plottitle "Free Energy of Unfolding"
        set energies $unfoldener($currmol)
    }
        
    
    if { $infos != 0 } {
        puts "Plot Data"
        puts $phdata($currmol)

        puts $energies
    }
    
    set plothandle [multiplot -x $phdata($currmol) -y $energies -title $plottitle -lines -linewidth 3 -marker point -plot \
                -xlabel pH -ylabel "\[kcal/mol\]"]
}



proc ::PropKa::propka_plotchrg {} {
## Plotting Unfolding energy vs. pH using multiplot VMD plugin

    variable phchrg
    variable unfoldchrg
    variable foldchrg
    variable plotcharge
    variable infos
    variable loadedpkafiles
    
    if { [molinfo num] == 0 } {
        return
    }
    
    set currmol m[molinfo top]
    

    if { [lsearch $loadedpkafiles $currmol ] == "-1" } {
       return
    }
    
    if {$phchrg($currmol) == "" } {
        tk_dialog .errmsg {PropKa Warning} "Missing charge profile. \nCheck PROPKA calculations" error 0 Dismiss
        set plotcharge 0
        return 
    }

    if { $infos != 0 } {
        puts "Plot Data"
        puts $phchrg($currmol)
        puts $unfoldchrg($currmol)
        puts $foldchrg($currmol)
    }

    set plothandle [multiplot -x $phchrg($currmol) -y $unfoldchrg($currmol) -title "Charge" -lines -linewidth 3 -marker point -plot \
                -xlabel pH -ylabel "\[e\]" -legend "Unfolded state"]
    
    $plothandle add $phchrg($currmol) $foldchrg($currmol) -marker point -legend "Folded state" -linewidth 3 -plot 

    set plotcharge 0
    
}


proc ::PropKa::propka_show_all {} {

   variable showallpkalabels
   variable allpkalabelsdrawn
   variable showallpkares
   variable chainsexist
   variable chainlist
   variable reslist
   variable infos
   variable pka
   variable loadedpkafiles
   variable ligandsnames_found
   
   set todel ""
   set allpkalabelsdrawn_tmp ""
   
   set currmol m[molinfo top]
   
   
    if { [molinfo num] == 0 } {
     return
    }
   
    if { [lsearch $loadedpkafiles $currmol ] == "-1" } {
       return
    }
   
    if { $showallpkares != 1 } {
        if { $showallpkalabels == 1 } {
            if { $allpkalabelsdrawn != "" } {
            ## Deleting labels only for The Top molecule 
                foreach id $allpkalabelsdrawn {
                    if { [regexp $currmol $id] } { 
                        lappend todel $id
                    } else { 
                        lappend allpkalabelsdrawn_tmp $id
                    }
                }
                if { $todel != "" } {
                    ::PropKa::propka_remove_label $todel
                    set allpkalabelsdrawn $allpkalabelsdrawn_tmp
                }
            }
        }
        set showallpkalabels 0
        return
    }
    
    mol showrep [molinfo top] 0 off
    mol representation lines

    
    # Exluding letters from the selection
    
    if {$chainsexist($currmol) == 1} {
        foreach chain $chainlist($currmol) {
            if { $infos != 0 } {
                puts "Ionizable residues in chain $chain:"
                puts $reslist($chain$currmol)
            }
            set reslist_NoLigandnames ""
            foreach res_tmp $reslist($chain$currmol) {
                if {[string first "-" $res_tmp] == "-1" } {
                    if {[string index $res_tmp 0 ] == "C" || [string index $res_tmp 0 ] == "N" } {
                       lappend reslist_NoLigandnames [string range $res_tmp 1 end]
                    } else {
                        lappend reslist_NoLigandnames $res_tmp
                    }
                } else {
                    set resid_tmp [lindex [split $res_tmp "_"] end]
                    if { [lsearch $reslist_NoLigandnames $resid_tmp] == "-1" } {
                        lappend reslist_NoLigandnames $resid_tmp
                    }
                }
            }
            
            atomselect macro pKaRes$chain "resid $reslist_NoLigandnames"
            set selection_text "chain $chain and pKaRes$chain"
            mol selection $selection_text
            mol addrep top
        }
    } else {
        if { $chainsexist($currmol) == 0} {
            set tmpchain $chainlist($currmol)
        } else {
            set tmpchain ""
        }
        
        
        if { $infos != 0 } {
                puts "Ionizable residues shown:"
                puts $reslist($tmpchain$currmol)
        }
        
        set reslist_NoLigandnames ""
        foreach res_tmp $reslist($tmpchain$currmol) {
           if {[string first "-" $res_tmp] == "-1"} {
                    if {[string index $res_tmp 0 ] == "C" || [string index $res_tmp 0 ] == "N" } {
                       lappend reslist_NoLigandnames [string range $res_tmp 1 end]
                    } else {
                        lappend reslist_NoLigandnames $res_tmp
                    }
           } else {
               set resid_tmp [lindex [split $res_tmp "_"] end]
               if { [lsearch $reslist_NoLigandnames $resid_tmp] == "-1"} {
                  lappend reslist_NoLigandnames $resid_tmp
              }
           }
        }
        

        atomselect macro pKaRes "resid $reslist_NoLigandnames"
        mol selection pKaRes
        mol addrep top
    }
    

    if {$showallpkalabels == 1 && $ligandsnames_found($currmol) == ""} { 
        if {$chainsexist($currmol) == 1} {
                foreach chain $chainlist($currmol) {
                    foreach residue $reslist($chain$currmol) {
                    set atom CB
#       Try to make labels for N+ and C-
                        if {![regexp 99.99 $pka($residue$chain$currmol)]} {
                            set residtmp $residue
                            if {[string index $residue 0 ] == "N"} {
                               set atom N
                               set residtmp [string range $residue 1 end]
                            }
                            if {[string index $residue 0 ] == "C" } {
                                set atom C
                                set residtmp [string range $residue 1 end]
                            }
                            lassign [[atomselect top "resid $residtmp and name $atom and chain $chain"] get {x y z}] resicoor
                            ::PropKa::propka_label $residue$chain$currmol $resicoor " $pka($residue$chain$currmol)"
                            lappend allpkalabelsdrawn $residue$chain$currmol
                        }
                    }
                }
        } else {
            foreach residue $reslist($tmpchain$currmol) { 
                set atom CB
                if {![regexp 99.99 $pka($residue$tmpchain$currmol)]} {
                    set residtmp $residue
#       Try to make a labels for N+ and C-
                    if {[string index $residue 0 ] == "N"} {
                        set atom N
                        set residtmp [string range $residue 1 end]
                    }
                    if {[string index $residue 0 ] == "C" } {
                        set atom C
                        set residtmp [string range $residue 1 end]
                    }
                    lassign [[atomselect top "resid $residtmp and name $atom"] get {x y z}] resicoor
                    ::PropKa::propka_label $residue$tmpchain$currmol $resicoor " $pka($residue$tmpchain$currmol)"
                    lappend allpkalabelsdrawn $residue$tmpchain$currmol
                }
            }
        }
        
    }
}



proc ::PropKa::propka_color_regions {} { 

variable toprepslist
variable topdrawn
variable stabregions

set molid [molinfo top]
set currmol m$molid

    if { [regexp $currmol $topdrawn] } {
        if { $stabregions == 1 } {
            foreach toprep $toprepslist($currmol) {
               mol modcolor $toprep $molid beta
            }
        } else { 
            foreach toprep $toprepslist($currmol) {
                mol modcolor $toprep $molid name
            }
        }
    }

}

proc ::PropKa::propka_shifted_pkas {} {

   variable showallpkalabels
   variable resinfo
   variable thresholdpka
   variable nummaxshifted
   variable maxshiftedtype
   variable showshiftedpkalabels
   variable showshiftedby
   variable showabsshiftedpkalabels
   variable showshiftedpka
   variable topdrawn
   variable showdelg
   variable toprepslistmols
   variable repsrem
   variable showmaxdelg 
   variable coloredlabels
   variable colorlabels
   variable chainsexist
   variable chainlist
   variable infos
   variable loadedpkafiles
   
   if { [molinfo num] == 0 } {
    return
   }
    
   set currmol m[molinfo top]
   set molid [molinfo top]
    
   if { [lsearch $loadedpkafiles $currmol ] == "-1" } {
      return
   }

   if { $showmaxdelg == 1 } {
      set showshiftedpka 1
      set showshiftedby 0
      set showmaxdelg 0
   }
    
    mol representation licorice


    if { $coloredlabels == 1 && $molid <= [expr [llength $colorlabels] - 1 ] } {
        graphics top color [lindex $colorlabels $molid]
    } elseif { $showallpkalabels == 0 } { 
        graphics top color white
    } else { 
        graphics top color yellow
    }
      
## delete previous labels every time this procedure is initialized

    ::PropKa::propka_remove_max_for_top
    
    
    if { $chainsexist($currmol) == 1 } { 
        set maxshifted [expr [llength $chainlist($currmol)] * $nummaxshifted]
    } else {
        set maxshifted $nummaxshifted
    }
    
    if { $showshiftedpka == 1 } {
    

## Absolute shifted
        if { $maxshiftedtype == 1 } {
            set sorted_pka [lsort -real -decreasing -index 3 $resinfo($currmol)]
            if { $infos != 0 } {
                puts "$maxshifted Residues with the maximum shifted ABSOLUTE pKa values in the molecule [molinfo top] are:"
            }
        }
## Shifted down
        if { $maxshiftedtype == 2 } {
            if { $infos != 0 } {
                puts "$maxshifted Residues with the maximum shifted DOWN pKa values in the molecule [molinfo top] are:"
            }
            set sorted_pka [lsort -real -index 2 $resinfo($currmol)]
        }    
## Shifted up    
        if { $maxshiftedtype == 3 } {
            if { $infos != 0 } {
                puts "$maxshifted Residues with the maximum shifted UP pKa values in the molecule [molinfo top] are:"
            }
            set sorted_pka [lsort -real -decreasing -index 2 $resinfo($currmol)]
        }
        
        ::PropKa::propka_show_shifted $sorted_pka $currmol $showshiftedpkalabels 0 0 [expr $maxshifted - 1]
            
        ::PropKa::propka_color_regions
 
    }

    if { $showshiftedby == 1} {
        if { $maxshiftedtype == 1 } {
            set sorted_pka [lsort -real -decreasing -index 3 $resinfo($currmol)]
            if { $infos != 0 } {
                puts "Residues with ABSOLUTE pKa values shifted by more than $thresholdpka in the molecule [molinfo top] are:"
            }
        }
        
        if { $maxshiftedtype == 2 } {
            set sorted_pka_tmp [lsort -real -index 2 $resinfo($currmol)]
            foreach element $sorted_pka_tmp { 
                if { [lindex $element 2 ] <= 0 } {
                    lappend sorted_pka $element
                }
            }
            if { $infos != 0 } {
                puts "Residues with pKa values shifted DOWN by more than $thresholdpka in the molecule [molinfo top] are:"
            }
        }
        
        if { $maxshiftedtype == 3 } {
            set sorted_pka_tmp [lsort -real -decreasing -index 2 $resinfo($currmol)]
            foreach element $sorted_pka_tmp { 
                if { [lindex $element 2 ] >= 0 } {
                    lappend sorted_pka $element
                }
            }
            if { $infos != 0 } {
                puts "Residues with pKa values shifted UP by more than $thresholdpka in the molecule [molinfo top] are:"
            }
        }

    ::PropKa::propka_show_shifted $sorted_pka $currmol $showshiftedpkalabels $thresholdpka 0 end
    ::PropKa::propka_color_regions
    }
    
}


proc ::PropKa::propka_show_shifted { ionizables currmol labels threshold start end } { 

    variable chainsexist
    variable chainlist
    variable topdrawn
    variable toprepslist
    variable toprepslistmols
    variable showdelg
    variable labelsshifted
    variable labelsmolid
    variable infos
    variable pka
    variable namelist
    variable abbrevlist
    variable infotext
    variable showmutdeterm
    variable wrmutfile
    variable dGQ_label
   
    
    mol representation licorice
    
    set molindex ""
    set label ""  
    set betastab 35
    set betadestab 0
    set incrf 5
    set newbeta 0
           
    foreach top_pka [lrange $ionizables $start $end] {
        if { [lindex $top_pka 3] >= $threshold } {
             set topid [lindex $top_pka 5]

              set labelatom CB
    
    # For different treatment of the N and C term.        
              set topidNC $topid
              if {[string index $topid 0 ] == "N"} {
                set labelatom N
                set topid [string range $topid 1 end]
              }
              if {[string index $topid 0 ] == "C" } {
                set labelatom C
                set topid [string range $topid 1 end]
              }
             
            set chainsel "" 
            if { $chainsexist($currmol) == 1 } {
                set topchain [lindex $top_pka 6]
                set selection_text "resid $topid and chain $topchain"
                set chainsel " and chain $topchain"
                set residueid $topidNC$topchain$currmol
            } else {
                if { $chainsexist($currmol) == 0} {
                    set tmpchain $chainlist($currmol)
                } else {
                    set tmpchain ""
                }
                set selection_text "resid $topid"
                set residueid $topidNC$tmpchain$currmol
            }
            

                     
            # Ligands Added
            set sel_atom_tmp " and alpha"
            if {[string first "-" $topidNC] != "-1" } { 
                set liganntmp [split [lindex [split $topidNC "_"] 0] "-"]
                set topid [lindex [split $topidNC "_"] end]
                set selection_text "resname [lindex $liganntmp 0] and resid $topid$chainsel"
                set sel_atom_tmp " and name [lindex $liganntmp 1]"
                set labelatom [lindex $liganntmp 1]
            }
            
            
#To Keep the information about added representations
            set molid [string range $currmol 1 end]
            set rep [lindex [mol list $molid] 12]
           
            mol selection $selection_text
            mol addrep top


            if {[lindex $top_pka 2] > 0 } {
                set betastab [expr $betastab - $incrf]
                set newbeta $betastab
                if { $newbeta < 25 } {
                    set newbeta 25
                }
            } else {
                set betadestab [expr $betadestab + $incrf]
                set newbeta $betadestab
                if { $newbeta > 25 } {
                    set newbeta 25
                }
            }
            
            set selection1 [atomselect $molid "$selection_text"]
            set sel1 [atomselect $molid "$selection_text$sel_atom_tmp"]
            lassign [$sel1 get {x y z}] csel1
            
           
            $selection1 set beta $newbeta
            
            lappend toprepslist($currmol) $rep
            lappend toprepslistmols $currmol
            
            if { $labelsmolid == 1 } {
                set molindex $currmol
            }
             
            lassign [[atomselect top "$selection_text and name $labelatom"] get {x y z}] resicoor

            if { $labelsshifted == 1 && $molid != 0 && $molid <= 7} {
               set resicoor [vecscale [expr 1 + (0.01 * $molid)] $resicoor]
            }
            lappend topdrawn $residueid
            
            if { $showdelg == 1 } {
                set label " [lindex $top_pka 0]$molindex: [lindex $top_pka 1]($dGQ_label [format {%3.1f} [::PropKa::propka_calc_delg $residueid]])"
                set labelf [format {%13s %4s %3s %6s} [lindex $top_pka 0]$molindex: [lindex $top_pka 1] ($dGQ_label [format {%3.1f} [::PropKa::propka_calc_delg $residueid]])]
            } else {
                set label " [lindex $top_pka 0]$molindex: [lindex $top_pka 1](shift [lindex $top_pka 2])"
                set labelf [format {%13s %4s %6s %6s} [lindex $top_pka 0]$molindex: [lindex $top_pka 1] (shift [lindex $top_pka 2])]
            }
                
            if { $labels == 1 } { 
               ::PropKa::propka_label $residueid $resicoor $label 
            }
            
            if { $infos != 0 } {
                puts $labelf
            } 
        } else {
            break
        }
    }
}


 

proc ::PropKa::propka_label { resid coords label }  { 

    variable drawnids
    variable labelid

    if { [lsearch $drawnids $resid] >= 0 } {
        ::PropKa::propka_remove_label $resid
    }
    set mol [lindex [split $resid m] 1]
    set labelid($resid) [graphics $mol text $coords $label]
    lappend drawnids $resid
    
}


proc ::PropKa::propka_remove_label { labellist } { 
    
    variable drawnids
    variable labelid

    foreach id $labellist {
        set mol [lindex [split $id m] 1]
        graphics $mol delete $labelid($id)
        set loc [lsearch $drawnids $id]
        lreplace $drawnids $loc $loc
        set $labelid($id) ""
    }
}


proc ::PropKa::propka_hbond { determ selection pickedmol res draw type } {

    variable drawnhbonds
    variable hbondsid
    
    variable chainsexist

    set dist 20
    
    if { $draw == 0 } {
    
        if { [lsearch $hbondsid $res] >= 0 } {
            foreach id $drawnhbonds($res) {
                graphics $pickedmol delete $id
            }
            set loc [lsearch $hbondsid $res]
            lreplace $hbondsid $loc $loc
            set drawnhbonds($res) ""
            
        }
        return
    }
    
    
   
    set sel1 [atomselect $pickedmol "$selection and name \"N.*|O.*|S.*\" and not name N O"]

    if { $type == 1 } {
        if { $chainsexist(m$pickedmol) == 1 } {
            set sel2 [atomselect $pickedmol "resid [string range $determ 0 end-1] and chain [string range $determ end end] and name \"N.*|O.*|S.*\" and not name N O"]
        } else {
            set sel2 [atomselect $pickedmol "resid $determ and name \"N.*|O.*|S.*\" and not name N O"]
        }
    }
    
    if { $type == 2 } {
        if { $chainsexist(m$pickedmol) == 1 } {
            set sel2 [atomselect $pickedmol "resid [string range $determ 0 end-1] and chain [string range $determ end end] and name N O"]
        } else {
            set sel2 [atomselect $pickedmol "resid $determ and name N O"]
        }
    }

        set atoms1 [$sel1 get index]
        set atoms2 [$sel2 get index]


    foreach atom1 $atoms1 {
        foreach atom2 $atoms2 {
            set dist_tmp [measure bond "$atom1 $atom2" molid $pickedmol]
            if { $dist_tmp < $dist } {
                set coord1 [lindex [[atomselect $pickedmol "index $atom1"] get {x y z}] 0]
                set coord2 [lindex [[atomselect $pickedmol "index $atom2"] get {x y z}] 0]
                set dist $dist_tmp
            }
        }
    }
    
    lappend drawnhbonds($res) [graphics $pickedmol line $coord1 $coord2 style dashed]
    lappend hbondsid $res
    
}





proc ::PropKa::propka_remove_all_labels { } {

    variable drawnids
    variable labelid
    
    variable drawnhbonds
    variable hbondsid
    
    variable determ_labels_ids
    
    variable pickedlist
    variable picked
    
    variable pickedpka
    variable pickedpkalist

    
    set currmol m[molinfo top]
    
    set todel "" 
    
    foreach id $pickedlist {
        if { [regexp $currmol $id] } { 
            set picked($id) 0
            set determ_labels_ids($id) ""
        }
    }

    foreach id $pickedpkalist {
        if { [regexp $currmol $id] } { 
            set pickedpka($id) 0
        }
    }

    foreach id $hbondsid {
        if { [regexp $currmol $id] } { 
            ::PropKa::propka_hbond 0 0 top $id 0 0
        }
    }
    
    if { $drawnids != "" } {
        foreach in $drawnids { 
            if { [regexp $currmol $in] } {     
                   lappend todel $in
            }
        }
        if { [llength $todel] > 0 } {
           ::PropKa::propka_remove_label $todel
        }
    }
    
      foreach usedlabels { Atoms Bonds Dihredrals Angles } { 
            label delete $usedlabels
      }

}


proc ::PropKa::propka_remove_allreps { } {
    
    set reps [lindex [mol list top] 12]
    for {set i 1} {$i <= $reps} {incr i} {
        mol delrep 1 top
    }
    variable showshiftedpka 0
    variable showshiftedby 0
    variable showmaxdelg 0
    variable showdelg 0
    variable showallpkares 0
    
}


proc ::PropKa::propka_remove_max_for_top {} {

variable topdrawn
variable repsrem
variable toprepslistmols
variable toprepslist

set currmol m[molinfo top]


if { [regexp $currmol $topdrawn] } {
        foreach topid $topdrawn {
            if { [regexp $currmol $topid] } { 
                lappend todel $topid
                set loc [lsearch $topdrawn $topid]
                lreplace $topdrawn $loc $loc
            }
        }
        ::PropKa::propka_remove_label $todel
        if { $repsrem == 1 } {
            if { [regexp $currmol $toprepslistmols] } {
                foreach i [lsort -decreasing -real $toprepslist($currmol)] {
                    mol delrep $i top
                    set loc [lsearch $toprepslist($currmol) $i]
                    lreplace $toprepslist($currmol) $loc $loc
                }
            set toprepslist($currmol) ""
            }
        }
    }
}


proc ::PropKa::propka_delg {} {

    variable resinfo
    variable reslist
    variable description
    variable nummaxshifted
    variable pka
    variable showdelg
    variable ph
    variable coloredlabels 
    variable colorlabels
    variable topdrawn
    variable showmaxdelg
    variable chainsexist
    variable chainlist
    variable maxstabtype
    variable infos
    variable loadedpkafiles
    variable showshiftedpka 0
    variable showshiftedby 0
    variable stab_label
    variable destab_label
    variable dGQ_label
    
    set todel ""
    set totaldelg 0
    set totalchrg 0
 

    if { [molinfo num] == 0 } {
        return
    }
    
    set currmol m[molinfo top]
    
    if { [lsearch $loadedpkafiles $currmol ] == "-1" } {
       return
    }
    
    if { $coloredlabels == 1 && [molinfo top] <= [expr [llength $colorlabels] - 1 ] } {
        graphics top color [lindex $colorlabels [molinfo top]]
    }
    
    
    ::PropKa::propka_remove_max_for_top 
    
    if { $showmaxdelg != 1 } {
        set showdelg 0
        return
    }


    variable delglist
    set delglist($currmol) ""
    
    variable readtyr 
    variable readcystyr
   
    if { $readtyr == 1 } {
        set condition "CYS"
    } elseif { $readcystyr == 1 } {
        set condition ""
    } else {
        set condition "TYR|CYS"
    }
    
    
    if { $chainsexist($currmol) == 1 || $chainsexist($currmol) == 0 } { 
        set maxshifted [expr [llength $chainlist($currmol)] * $nummaxshifted]
        foreach tmpres $resinfo($currmol) {
            set res [lindex $tmpres 5][lindex $tmpres 6]$currmol
            if { ![regexp \"$condition\" [lindex $tmpres 4]] } {
                set dg [::PropKa::propka_calc_delg $res]
                set absdg [expr abs($dg)]
                lappend delglist($currmol) "[lrange $tmpres 0 1] $dg $absdg [lrange $tmpres 4 end]"
                set totaldelg [expr $totaldelg + $dg]
                set totalchrg [expr $totalchrg + 0]
            }
        }
    } else {
        set maxshifted $nummaxshifted
        foreach tmpres $resinfo($currmol) {
            set res [lindex $tmpres 5]$currmol
            if { ![regexp \"$condition\" [lindex $tmpres 4]] } {
                set dg [::PropKa::propka_calc_delg $res]
                set absdg [expr abs($dg)]
                set totaldelg [expr $totaldelg + $dg]
                lappend delglist($currmol) "[lrange $tmpres 0 1] $dg $absdg [lrange $tmpres 4 end]"
            }
        }
    }
    
    
    if { $maxstabtype == 1 } {
            set sorted_delg [lsort -real -decreasing -index 3 $delglist($currmol)]
            if { $infos != 0 } {
                puts "$maxshifted Residues with the highest contrib. to the $dGQ_label in the molecule [molinfo top] are:"
            }
    }
      
    if { $maxstabtype == 2 } {
        set sorted_delg [lsort -real -decreasing -index 2 $delglist($currmol)]
        if { $infos != 0 } {
            puts "$maxshifted Residues with the highest $stab_label contrib. to the $dGQ_label in the mol [molinfo top] are:"
        }
    }

    if { $maxstabtype == 3 } {
        set sorted_delg [lsort -real -index 2 $delglist($currmol)]
        if { $infos != 0 } {
            puts "$maxshifted Residues with the highest $destab_label contrib. to the $dGQ_label in the mol [molinfo top] are:"
        }
    }

        set showdelg 1
        
        ::PropKa::propka_show_shifted $sorted_delg $currmol 1 0 0 [expr $maxshifted - 1]
    
        ::PropKa::propka_color_regions
        
    if { $infos != 0 } {
        puts "Total DelG for molecule [molinfo top] is: [format {%3.2f} $totaldelg]"
    }

}


proc ::PropKa::propka_calc_delg { res } {

    variable pkaprec
    variable modelpka
    variable ph
    variable name
    variable calculate_charge
    variable totaldesolv_prec


    if { $totaldesolv_prec($res) < 0.0 || [regexp {HIS|N\+|LYS|ARG} $name($res)] } {
       set charge_state 1
    } else {
        set charge_state -1
    }
    
    
    set pkaprime $modelpka($res)
    
    set dq [expr (log10(1+pow(10, ($ph - $pkaprec($res))))) - (log10(1+pow(10, ($ph - $modelpka($res)))))]
    set dg [expr 1.36*$dq + 1.36*($pkaprime - $modelpka($res))]
    
    if {$calculate_charge == "1"} {
       set x_folded [expr pow(10, ($charge_state * ($pkaprec($res) - $ph)))]
       set q_folded [expr $charge_state * $x_folded/(1+$x_folded)]
       return $q_folded
    } else {    
       return $dg
    }

}


proc ::PropKa::propka_match_contrib { sel mol tmpid tmpname } {

    variable chainlist
    variable infos

    set dist 20

    lassign [$sel get {x y z}] coor

    foreach tmpchain $chainlist(m$mol) {
        set determsel [atomselect $mol "resid $tmpid and chain $tmpchain"]
        set determname [lindex [$determsel get resname] 0]
        if { $determname == $tmpname } {
            set determcoor [measure center $determsel]
            set tmpdist [expr sqrt(pow(([lindex $coor 0] - [lindex $determcoor 0]),2) + pow(([lindex $coor 1] - [lindex $determcoor 1]),2) + pow(([lindex $coor 2] - [lindex $determcoor 2]),2))] 
            if { $tmpdist < $dist } {
                set chain $tmpchain
                set dist $tmpdist
            }
        }
    }

    return $chain
    
}


proc ::PropKa::propka_contrib_fctn { args } {

    global vmd_pick_atom
    global vmd_pick_mol
    
    variable showallpkalabels
    variable chainsexist
    variable namelist
    variable abbrevlist


    variable drawnids
    variable determ_labels_ids
    
    variable pickedlist
    variable picked
    variable pickmode
    
    variable pickedpka
    variable pickedpkalist
    
    variable pka
    variable abbrevname
    variable repsrem
    variable determ_rep
    
    variable coloredlabels
    variable colorlabels
    variable colordeterm
    
    variable labelsshifted
    variable labelsmolid
    variable infos
    variable chainlist
    
    variable residlist
    variable toprepslist
    variable pkaftype
    
    variable dGQ_label


    set backhb_list ""
    set sidehb_list ""
    set molindex ""
    set determ_list ""

    set sel [atomselect $vmd_pick_mol "index $vmd_pick_atom"]

    set pickedresid [$sel get resid] 
    set pickedchaintmp [$sel get chain]
    set pickedmol m$vmd_pick_mol

    set nonizoniz "GLY ALA VAL LEU ILE MET PHE TRP PRO SER THR ASN GLN"
    
    set pickedname [$sel get resname]
    if {$pickedchaintmp == "X" } {
        if {$pkaftype($pickedmol) == 3} { 
            set pickedchain A
        } elseif { [lsearch $namelist $pickedname] == "-1"} {
            set pickedchain [lindex $chainlist($pickedmol) 0]
        } else {
            set pickedchain ""
        }
    } else {
        set pickedchain $pickedchaintmp
    }
    
     
    set restmp $pickedresid$pickedchain$pickedmol
    
    
    # To work with ligands Interactively
    set chain_sel_text ""       
    if { $chainsexist($pickedmol) == 1 } {
        set chain_sel_text " and chain $pickedchaintmp"
    } 
    
   set restmpNC ""
   set terminaltext ""
           
   #For the N terminal group
   if { [lsearch $residlist($pickedmol) N$restmp] != "-1"} {
       set restmpNC N$restmp
       if { $pickmode == 3 } { 
           set terminaltext "and $dGQ_label[format {%3.1f} [::PropKa::propka_calc_delg $restmpNC]]\(N\+\)"
       } else {
           set terminaltext "and $pka($restmpNC)\(N\+\)"
       }
   }
            
   #For the C terminal group
   if { [lsearch $residlist($pickedmol) C$restmp] != "-1"} {
       set restmpNC C$restmp
       if { $pickmode == 3 } { 
          set terminaltext "and $dGQ_label[format {%3.1f} [::PropKa::propka_calc_delg $restmpNC]](\C\-\)"
       } else {
          set terminaltext "and $pka($restmpNC)\(\C\-\)"
       }
   }
   
    
    if { [lsearch $namelist $pickedname] == "-1" } {
       puts "  Picked Ligand is $pickedname" 
       puts "   If Non-ionizable ligand atom was picked,\n   ionizable ligand atoms should be highlighted with VDW spheres\n   and listed below:"
       
       ## add vdW spheres for ionizable ligand atoms
       mol representation VDW 0.3 8.0
       set pickedatomname [$sel get name]
       set restmp_lig $pickedname\-$pickedatomname\_$restmp
       if { [lsearch $residlist($pickedmol) $restmp_lig] == "-1" } {
           foreach resi $residlist($pickedmol) {
              set resispl [split $resi "-|_"]
              if { [lindex $resispl 0] == $pickedname && [lindex $resispl end] == $restmp } {
                 set label [lindex $resispl 1]
                 puts "     $label"
                 set selection "resname [lindex $resispl 0] and resid $pickedresid and name $label$chain_sel_text"
                 set macroname "$label\-[lindex $resispl 0]"
                 
                 set rep [lindex [mol list $vmd_pick_mol] 12]
                 atomselect macro Lig_$label $selection
                 mol selection Lig_$label
                 mol addrep $vmd_pick_mol
                 lappend toprepslist($pickedmol) $rep
              } 
           }
           
       } else {
         set selection_text "resname $pickedname and resid $pickedresid and name $pickedatomname$chain_sel_text"
       }
      set restmp $restmp_lig
    } else {
       # To avoid nonionizable residues ! 
       if { [lsearch $nonizoniz $pickedname] != "-1" && $terminaltext == "" } {
          puts "Non-ionizable residue picked"
          return      
       
       }
       if {$pickedname == "GLY" } {
         set selection_text "resid $pickedresid and name CA$chain_sel_text"
       } else {
         set selection_text "resid $pickedresid and name CB$chain_sel_text"
       
       }
    }
    if { $coloredlabels == 1 && $vmd_pick_mol <= [expr [llength $colorlabels] -1 ] } {
        graphics $vmd_pick_mol color [lindex $colorlabels $vmd_pick_mol]
    } elseif { $showallpkalabels == 0 } { 
       graphics $vmd_pick_mol color white
    } else { 
        graphics $vmd_pick_mol color yellow
    }
     
                    
    if { $labelsmolid == 1 } {
        set molindex $pickedmol
    }

    lassign [[atomselect $vmd_pick_mol $selection_text] get {x y z}] resicoor
            
    if { $vmd_pick_mol > 7 } {
        set labelsshifted_curr 0
    } else {
        set labelsshifted_curr $labelsshifted
    }
            
    if { $labelsshifted_curr == 1 && $vmd_pick_mol != 0 } {
        set resicoor [vecscale [expr 1 + (0.01 * $vmd_pick_mol)] $resicoor]
    } 

## Show pKa or DeltaG value only when molecule is picked.

    if {$pickmode == 2 || $pickmode == 3} {
    
        if {[lsearch $pickedpkalist $restmp] >= 0} {
            incr pickedpka($restmp)
        } else { 
            set pickedpka($restmp) 1
            lappend pickedpkalist $restmp
        }

        
        if { $pickedpka($restmp) == 4 } {
            if { [lsearch $drawnids $restmp] >= 0 } {
                ::PropKa::propka_remove_label $restmp
            }
            set pickedpka($restmp) 0
        }
        
        if { $pickedpka($restmp) == 1 } {
            if { [lsearch $nonizoniz $pickedname] != "-1" } {
                 set labelpka " [lrange $terminaltext 1 end]"
                 ::PropKa::propka_label $restmp $resicoor $labelpka
            } else {
                if { $pickmode == 3 } { 
                    set labelpka " $abbrevname($restmp)$pickedresid$pickedchain$molindex $dGQ_label[format {%3.1f} [::PropKa::propka_calc_delg $restmp]] $terminaltext"
                    ::PropKa::propka_label $restmp $resicoor $labelpka
                } else { 
                    set labelpka " $abbrevname($restmp)$pickedresid$pickedchain$molindex $pka($restmp) $terminaltext"
                    ::PropKa::propka_label $restmp $resicoor $labelpka
                }  
            }
            if { $infos != 0 } {
                puts "$labelpka"
            }
        }
        return
    }
    
    if {[lsearch $pickedlist $restmp] >= 0} {
        incr picked($restmp)
    } else { 
        set picked($restmp) 1
        lappend pickedlist $restmp
    }
    
    
## Delete determinants labels when molecule picked fourth time
    if { $picked($restmp) == 4 } {
    
        if { [lsearch $drawnids $restmp] >= 0 } {
            ::PropKa::propka_remove_label $restmp
        }
        set picked($restmp) 0
        
        if { [lsearch $nonizoniz $pickedname] == "-1" } {
           ::PropKa::propka_remove_label $determ_labels_ids($restmp)
           set determ_labels_ids($restmp) ""
        
## Delete Hbonds if they were shown befeore ... 
           ::PropKa::propka_hbond 0 0 $vmd_pick_mol $restmp 0 0
        
           #For Ligands !
           if { $repsrem == 1 } {
               foreach rep_to_rem [lsort -real -decreasing $determ_rep($restmp)] {
                   mol delrep $rep_to_rem $vmd_pick_mol
               }
               set determ_rep($restmp) ""
           }
        }
    }
    
    
    if {$picked($restmp) == 1} {    
        mol representation CPK  
        puts "pKa determinants mode"
        if { $terminaltext != "" } { 
            set terminal_determs [::PropKa::propka_make_determs $vmd_pick_mol $restmpNC $pickedresid 1]
            puts "Picked residue has an ionizable terminal group. \n Its pKa determinants are:" 
            foreach det $terminal_determs {
               puts "[format {%12s} [lindex $det 0]]  [lrange $det 1 end]"
            }
            if { [lsearch $nonizoniz $pickedname] != "-1" } {
                set labelpka " [lrange $terminaltext 1 end]"
                ::PropKa::propka_label $restmp $resicoor $labelpka
            }
        }
        ::PropKa::propka_make_determs $vmd_pick_mol $restmp $pickedresid 0
    }; # to work every fourth time
    
}



proc ::PropKa::propka_make_determs { vmd_pick_mol restmp pickedresid infomode } {

    variable showallpkalabels
    variable chainsexist
    variable sidehb
    variable backhb
    variable ccinteract
    variable namelist
    variable abbrevlist
    variable determ_labels_ids
    variable totaldesolv
    variable pka
    variable abbrevname
    variable showhbonds
    variable determ_rep
    variable coloredlabels
    variable colorlabels
    variable colordeterm
    variable labelsshifted
    variable labelsmolid
    variable infos
    variable residlist
    variable determ_todisplay_threshold
    
    set onechain ""
    set backhb_list ""
    set sidehb_list ""
    set molindex ""
    set determ_list ""
    set pickedmol m$vmd_pick_mol

    global vmd_pick_atom
    
    set pickedinfo [::PropKa::propka_get_sel $restmp]
    
    set selection_text [lindex $pickedinfo 0]
    
    if {$chainsexist($pickedmol) == 1} {
        set pickedchain [lindex $pickedinfo end]

    } else {
        set pickedchain ""

    }
    
    if { $chainsexist($pickedmol) == 0 } {
       set onechain [string index [lindex [split $restmp m] 0] end]
    }
    
    if { $labelsmolid == 1 } {
        set molindex $pickedmol
    }

    set commonlabel($restmp) "" 

    set ligand_selection_list ""
    set ligand_determ_list ""
    set ligand_atom_selection_list "" 
    
   foreach contrib $sidehb($restmp) {
        set ligand 0
        set chain_sel_text ""
        if { !([lindex $contrib 1] == "000" || [lindex $contrib 1] == "XXX") && [expr abs([lindex $contrib 0])] >= $determ_todisplay_threshold } {
            set tmpid [lindex $contrib 2]
            set tmpname [lindex $contrib 1]
            set chaininfo [split [lindex $contrib 3] "_"]           
            if {[lsearch $namelist $tmpname] == "-1" } {
               set ligatomname $tmpid
               set ligd [lindex $chaininfo 1]
               set tmpid $tmpname\-$tmpid\_$ligd
               set ligand 1
               set abbrev ""
            }        
            if { $chainsexist($pickedmol) == 1 } {

                set determchain [lindex $chaininfo 0]
                set chain_sel_text " and chain $determchain"
                set tmpid $tmpid$determchain
            }

            if { $ligand == 0 } {
                lappend determ_list $tmpid
## In case N+ is found, it is treated as a backbone h-bond for h-bond drawing procedure
                if { [lindex $contrib 1] == "N+" } { 
                    set abbrev [lindex $contrib 1]
                    lappend backhb_list $tmpid
## In case C- is found, it is treated as a backbone h-bond for h-bond drawing procedure                    
                } elseif { [lindex $contrib 1] == "C-" } {
                    set abbrev [lindex $contrib 1]
                } else {
                    set abbrev [lindex $abbrevlist [lsearch $namelist [lindex $contrib 1]]]
                    lappend sidehb_list $tmpid
                }
            } else {
                lappend ligand_determ_list $tmpid
                lappend ligand_atom_selection_list "resname $tmpname and name $ligatomname$chain_sel_text and resid $ligd"
                set ligand_selection "resname $tmpname$chain_sel_text and resid $ligd"
                if {[lsearch $ligand_selection_list $ligand_selection] == "-1" } {
                        lappend ligand_selection_list $ligand_selection
                }
            }
            set label($tmpid) "$abbrev$tmpid$molindex [format {%3.1f} [lindex $contrib 0]](SHB)"
        }
    }
    

   
    foreach contrib $backhb($restmp) {
        set ligand 0
        set chain_sel_text ""        
        if { !([lindex $contrib 1] == "000" || [lindex $contrib 1] == "XXX") && [expr abs([lindex $contrib 0])] >= $determ_todisplay_threshold } {
            set tmpid [lindex $contrib 2] 
            set tmpname [lindex $contrib 1]
            set chaininfo [split [lindex $contrib 3] "_"]
            if {[lsearch $namelist $tmpname] == "-1" } {
               set ligatomname $tmpid
               set ligd [lindex $chaininfo 1]
               set tmpid $tmpname\-$tmpid\_$ligd
               set ligand 1
               set abbrev ""
            }                        
            if { $chainsexist($pickedmol) == 1 } {
                set determchain [lindex $chaininfo 0]
                set chain_sel_text " and chain $determchain"
                set tmpid $tmpid$determchain
            }
            set determtmp $tmpid$onechain$pickedmol
            lappend backhb_list $tmpid
            
            ## add label for the same residue which is picked when it is its own determinant
            if { $determtmp == $restmp } {
                set commonlabel($restmp) "[format {%3.1f} [lindex $contrib 0]](BHB)"
            }
            if {[lsearch $determ_list $tmpid] >= 0 || [lsearch $ligand_determ_list $tmpid] >= 0 } {
                lappend label($tmpid) "[format {%3.1f} [lindex $contrib 0]](BHB)"
            } else {
                if { $ligand == 0 } {
                lappend determ_list $tmpid
                    if { [lindex $contrib 1] == "N+" || [lindex $contrib 1] == "C-" } {
                        set abbrev [lindex $contrib 1]
                    } else {
                        set abbrev [lindex $abbrevlist [lsearch $namelist [lindex $contrib 1]]]
                    }
                } else {
                    lappend ligand_determ_list $tmpid
                    lappend ligand_atom_selection_list "resname $tmpname and name $ligatomname$chain_sel_text and resid $ligd"
                    set ligand_selection "resname $tmpname$chain_sel_text and resid $ligd"
                    if {[lsearch $ligand_selection_list $ligand_selection] == "-1" } {
                        lappend ligand_selection_list $ligand_selection
                    }
                }
                set label($tmpid) "$abbrev$tmpid$molindex [format {%3.1f} [lindex $contrib 0]](BHB)"

            }   
        }
    } 


    foreach contrib $ccinteract($restmp) {
        set ligand 0 
        set chain_sel_text ""
        if { !([lindex $contrib 1] == "000" || [lindex $contrib 1] == "XXX") && [expr abs([lindex $contrib 0])] >= $determ_todisplay_threshold } {
            set tmpid [lindex $contrib 2] 
            set tmpname [lindex $contrib 1]
            set chaininfo [split [lindex $contrib 3] "_"]
            if {[lsearch $namelist $tmpname] == "-1" } {
               set ligatomname $tmpid
               set ligd [lindex $chaininfo 1]
               set tmpid $tmpname\-$tmpid\_$ligd
               set ligand 1
               set abbrev ""
            }
            
            if { $chainsexist($pickedmol) == 1 } {
                set determchain [lindex $chaininfo 0]
                set chain_sel_text " and chain $determchain"
                set tmpid $tmpid$determchain
            }

            if {[lsearch $determ_list $tmpid] >= 0 || [lsearch $ligand_determ_list $tmpid] >= 0 } {
                lappend label($tmpid) "[format {%3.1f} [lindex $contrib 0]](CC)"
            } else {
                if { $ligand == 0 } {
                    lappend determ_list $tmpid
                    if { [lindex $contrib 1] == "N+" || [lindex $contrib 1] == "C-" } {
                        set abbrev [lindex $contrib 1]
                    } else {
                        set abbrev [lindex $abbrevlist [lsearch $namelist [lindex $contrib 1]]]
                    }
                } else {
                    lappend ligand_determ_list $tmpid
                    lappend ligand_atom_selection_list "resname $tmpname and name $ligatomname$chain_sel_text and resid $ligd"
                    set ligand_selection "resname $tmpname$chain_sel_text and resid $ligd"
                    if {[lsearch $ligand_selection_list $ligand_selection] == "-1" } {
                        lappend ligand_selection_list $ligand_selection
                    }
                }
                set label($tmpid) "$abbrev$tmpid$molindex [format {%3.1f} [lindex $contrib 0]](CC)"
            }
        }
    } 


    
    if { $vmd_pick_mol > 7 } {
        set labelsshifted_curr 0
    } else {
        set labelsshifted_curr $labelsshifted
    }
            
    
    set determs [list "$abbrevname($restmp)$restmp $totaldesolv($restmp) DS $commonlabel($restmp)"]

    if { $infomode == 1 } {
       if { $determ_list != "" || $ligand_determ_list != ""} {
          foreach det $determ_list {
            lappend determs $label($det)
          }     
          foreach ligand_det $ligand_determ_list {
             lappend determs $label($ligand_det)
          }
        } else {
            lappend determs NoDeterms
        }
        return $determs
    }

    if { $determ_list != "" || $ligand_determ_list != "" } {
        if { $infos != 0 } {
            puts "Picked residue $abbrevname($restmp)$restmp determinants found:"
            foreach det $determ_list { 
                puts "[format {%16s} [lindex $label($det) 0]]  [lrange $label($det) 1 end]"
            }
            #Ligands
            foreach ligand_det $ligand_determ_list {
                puts "[format {%16s} [lindex $label($ligand_det) 0]]  [lrange $label($ligand_det) 1 end]"
            }
        } 
    
        
        if { $chainsexist($pickedmol) == 1 } {
            set chainsdetermlist ""
            foreach determinant $determ_list {
                set assigndeterm [string range $determinant end end]
                if {[lsearch $chainsdetermlist $assigndeterm] == "-1"} {
                    lappend chainsdetermlist $assigndeterm
                }
                lappend determsinchain($assigndeterm) [string range $determinant 0 end-1]
            }
            
            foreach determchaintmp $chainsdetermlist {
                set determtext "resid $determsinchain($determchaintmp) and chain $determchaintmp"
                atomselect macro determ$pickedresid\inChain$determchaintmp $determtext
                mol selection determ$pickedresid\inChain$determchaintmp
                
                lappend determ_rep($restmp) [lindex [mol list $vmd_pick_mol] 12]
                mol addrep $vmd_pick_mol 
            }
        } elseif { $determ_list != "" } {
            set determtext "resid $determ_list"
            atomselect macro determ$pickedresid $determtext
            mol selection determ$pickedresid
            lappend determ_rep($restmp) [lindex [mol list $vmd_pick_mol] 12]
            mol addrep $vmd_pick_mol 
        }
        

        foreach ligand_select $ligand_selection_list {
            if {[llength $ligand_select] == 8 } {
               set lig_chain_sel_tmp "_chain[lindex $ligand_select 4]"
            } else {
               set lig_chain_sel_tmp ""
            }
                        
            set lig_tmp [lindex $ligand_select 1]
            atomselect macro determ$pickedresid\Lig$lig_tmp$lig_chain_sel_tmp $ligand_select
            mol selection determ$pickedresid\Lig$lig_tmp$lig_chain_sel_tmp

            lappend determ_rep($restmp) [lindex [mol list $vmd_pick_mol] 12]
            mol addrep $vmd_pick_mol 
        }
        
        if { $coloredlabels == 1 && $vmd_pick_mol <= [expr [llength $colordeterm] - 1 ] } {
            graphics $vmd_pick_mol color [lindex $colordeterm $vmd_pick_mol]
        } else {
            graphics $vmd_pick_mol color silver
        }

        foreach id $determ_list {
            if  { $chainsexist($pickedmol) == 1 } {
                if { [[atomselect $vmd_pick_mol "resid [string range $id 0 end-1] and chain [string range $id end end] and name CA"] get resname] == "GLY" } {
                    lassign [[atomselect $vmd_pick_mol "resid [string range $id 0 end-1] and chain [string range $id end end] and name CA"] get {x y z}] resicoor
                } else {
                    lassign [[atomselect $vmd_pick_mol "resid [string range $id 0 end-1] and chain [string range $id end end] and name CB"] get {x y z}] resicoor
                }
            } else {
                if { [[atomselect $vmd_pick_mol "resid $id and name CA"] get resname] == "GLY" } { 
                    lassign [[atomselect $vmd_pick_mol "resid $id and name CA"] get {x y z}] resicoor
                } else {
                    lassign [[atomselect $vmd_pick_mol "resid $id and name CB"] get {x y z}] resicoor
                }
            }
            
            if { $labelsshifted_curr == 1 && $vmd_pick_mol != 0 } {
                    set resicoor [vecscale [expr 1 + (0.01 * $vmd_pick_mol)] $resicoor]
            }
            
            lappend determ_labels_ids($restmp) $id$onechain$pickedmol
            ::PropKa::propka_label $id$onechain$pickedmol $resicoor " $label($id)"
        }
        
        
        # Ligand Atoms
        mol representation VDW 0.4 8.0
        
        set lig_count 0
        foreach ligandatom $ligand_atom_selection_list { 
            set ligatom_tmp [lindex $ligandatom 4]
            set seltmp [atomselect $vmd_pick_mol $ligandatom]
            lassign [$seltmp get {x y z}] lig_atom_coor
            atomselect macro determ$pickedresid\_$ligatom_tmp\Lig $ligandatom
            mol selection determ$pickedresid\_$ligatom_tmp\Lig
            lappend determ_rep($restmp) [lindex [mol list $vmd_pick_mol] 12]
            
            mol addrep $vmd_pick_mol 
           
            if { $labelsshifted_curr == 1 && $vmd_pick_mol != 0 } {
                    set lig_atom_coor [vecscale [expr 1 + (0.01 * $vmd_pick_mol)] $lig_atom_coor]
            }
            set id [lindex $ligand_determ_list $lig_count]
            lappend determ_labels_ids($restmp) $id$onechain$pickedmol
            ::PropKa::propka_label $id$onechain$pickedmol $lig_atom_coor " $label($id)"
            
            incr lig_count
        }
    
        if { $showhbonds == 1 } {

            if { $sidehb_list != ""} {
                foreach res $sidehb_list {
                    ::PropKa::propka_hbond $res $selection_text $vmd_pick_mol $restmp 1 1
                }
            }
    
            if { $backhb_list != ""} {
                foreach res $backhb_list {
                    ::PropKa::propka_hbond $res $selection_text $vmd_pick_mol $restmp 1 2
                }
            }
        }
    
    }
    
    
    if { $coloredlabels == 1 && $vmd_pick_mol <= [expr [llength $colordeterm] - 1 ]} {
       graphics $vmd_pick_mol color [lindex $colorlabels $vmd_pick_mol]
    } elseif { $showallpkalabels == 0 } { 
       graphics $vmd_pick_mol color white
    } else { 
        graphics $vmd_pick_mol color yellow
    }
    
     
    ## Modify picked residue Label 
    set pickedlabel ""
    
    set sel_for_ligand [atomselect $vmd_pick_mol "index $vmd_pick_atom"]
    set pickedname [$sel_for_ligand get resname]
    set pickedatomname [$sel_for_ligand get name]
    if { [lsearch $namelist $pickedname] == "-1" } { 
       lassign [[atomselect $vmd_pick_mol "$selection_text and name $pickedatomname"] get {x y z}] resicoor
       lappend pickedlabel " $pickedname\-$pickedatomname\_$pickedresid$pickedchain$molindex $pka($restmp)"
    } else {
      lassign [[atomselect $vmd_pick_mol "$selection_text and name CB"] get {x y z}] resicoor
      lappend pickedlabel " $abbrevname($restmp)$pickedresid$pickedchain$molindex $pka($restmp)"
    }
   
    if { $labelsshifted_curr == 1 && $vmd_pick_mol != 0 } {
        set resicoor [vecscale [expr 1 + (0.01 * $vmd_pick_mol)] $resicoor]
    }
 

   if { $totaldesolv($restmp) != 0 } {
        lappend pickedlabel "\[$totaldesolv($restmp)\(DS\)\]"
   }
   if { $commonlabel($restmp) != "" } {
        lappend pickedlabel "\[$commonlabel($restmp)\]"
   }
   
   
   # Additional Check and info for the N and C terminal groups
   #For the N terminal group
   if { [lsearch $residlist($pickedmol) N$restmp] != "-1"} {
       set restmpNC N$restmp
       lappend pickedlabel "and $pka($restmpNC)\(N\+\)"
   }       
   #For the C terminal group
   if { [lsearch $residlist($pickedmol) C$restmp] != "-1"} {
       set restmpNC C$restmp
       lappend pickedlabel "and $pka($restmpNC)\(\C\-\)"
   }
        
   set pickedlabel "[join $pickedlabel]"
   
   ::PropKa::propka_label $restmp $resicoor $pickedlabel
   
   if { $infos != 0 } {
       puts "(Picked $pickedlabel)"

       if { $determ_list == "" && $ligand_determ_list == "" } {
           puts "No pKa determinant residues found\n"
       }
   }
  

}
