#
# $Id: bendix.tcl,v 1.4 2013/04/15 15:29:48 johns Exp $
#
#===============================================================================
#
#  bendix.tcl 
#
#    A protein characterization software for atomistic and coarse-grained proteins.
#    Bendix abstracts secondary structure without sacrificing conformation, 
#    and allows quantification and visualisation of helix geometry.
#    Bendix is a plugin for Visual Molecular Dynamics (VMD).
#
#    For more information, please see the website:
#    http://sbcb.bioch.ox.ac.uk/Bendix
#    
#
#    Authors:  A Caroline E Dahl, Matthieu Chavent and Mark S P Sansom
#              University of Oxford, February 2013
#
#    Citation: Bendix: Intuitive helix geometry analysis and abstraction
#              Anna Caroline E. Dahl; Matthieu Chavent; Mark S.P. Sansom
#              Bioinformatics (2012) 28 (16): 2193-2194. 
#
#    Please send feature requests and bug reports to caroline.dahl@dtc.ox.ac.uk.
#    Thank you.
#
#
# .........::::::: University of Illinois Open Source License :::::::.........
# Copyright 2012-2013 Structural Bioinformatics and Computational Biochemistry unit
#                     University of Oxford 
#                     All rights reserved.
# 
# Developed by:       A Caroline E Dahl
#                     Structural Bioinformatics and Computational Biochemistry unit
#                     University of Oxford
#                     http://sbcb.bioch.ox.ac.uk
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the Software), to deal with 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to 
# do so, subject to the following conditions:
# 
# - Redistributions of source code must retain the above copyright notice, 
# this list of conditions and the following disclaimers.
# 
# - Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimers in the documentation 
# and/or other materials provided with the distribution.
# 
# - Neither the names of Structural Bioinformatics and Computational Biochemistry,
# University of Oxford, nor the names of its contributors may be used to endorse
# or promote products derived from this Software without specific prior written
# permission.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL 
# THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS WITH THE SOFTWARE.
#
#
#===============================================================================


#### Installation notes (see also http://sbcb.bioch.ox.ac.uk/Bendix/faq.html):
#
#   1. Identify the path to your installed VMD package,
#      e.g. /Computer/Programs/VMD/
#   2. Decompress and save bendix1.1 and its content in a special VMD plugin subdirectory:
#      /Computer/Programs/VMD/plugins/noarch/tcl/
#      N.B. On Macs: right-click the VMD icon in Applications, choose "Show Package Contents"
#      and navigate to Contents > vmd > plugins > noarch > tcl
#   3. Identify (or create) and open the the vmd user settings file: vmdrc
#      You find it in (e.g.) /Computer/Programs/VMD/ or in your home directory
#      (N.B. this is a hidden file in Linux and MAC distributions)
#   4. Append these lines to vmdrc:
#       vmd_install_extension bendix bendix "Visualization/Bendix"
#       menu main on
#   5. Restart VMD.
#      Bendix appears under the VMD Main window tab
#      Extensions > Visualization > Bendix
#
#	.. alternatively load bendix from VMD's terminal or Tk Console using these commands:
#		source /dir/where/you/saved/bendix1.1/bendix.tcl
#		bendix
#
#### Code maintenance notes:
#
#      English spelling is used throughout.
#      Internal procs are initially indented by 2 spaces to save space.
#      The code is divided up into the following sections:
#            Create global variables
#            Main GUI setup
#            Procs:
#                GUI-related
#                Load and save files
#                User help and acknowledgements
#                Draw non-bendices
#                Helix analysis
#                Trajectory and molecule ID detection
#                Particle and subset validation
#                Error messages
#                Hide, erase and reset
#                Main bendix proc (long):
#                  Assign helicity, retrieve coordinates, calculate helix axis,
#                  apply spline, calculate angles, draw to screen.
#
#_______________________________________________________________________________

package provide bendix 1.1
package require tile
package require multiplot

################################################################################
#                                 GLOBAL VARs                                  #
################################################################################

global env

namespace eval ::bendix:: {
	variable proteinID "0";
	variable particle_name "CA";
	variable previous_particle_name "CA";
	variable subset "";
	variable helix_radius "2.2";
	variable angle_max "20.0";
	variable angle_colour_scale "RGB";
	variable uniform_colour 1 ;
	variable uniform_colour_type 2;
	variable input_colour "red";
	variable curvature_graph_X "";
	variable curvature_graph_Y "";
	variable points_to_graph "";
	variable spline_resolution "4";
	variable TurnResolution 4;
	variable previous_TurnResolution 4;
	variable AngleSide 3.6;
	variable frame 0;
	variable previous_frame 0;
	variable rep_zero 1;
	variable autoColour 1;
	variable autoCartoon 0;
	variable autoNewCartoon 0;
	variable normalised_startsandstops "";
	variable list_of_chains "";
	variable xyz "";
	variable helix_assignment "";
	variable helix_assignment_by_backbone "";
	variable startIndex "";
	variable endIndex "";
	variable spline_coords "";
	variable spline_startsandstops "";
	variable residueIndex "";
	variable backbone_radius "0.2";
	variable helix_type 1;
	variable CG 0;
	variable AThelix_string "";
	variable String_for_cartoonify "";
	variable CGbackbone 0;
	variable maximumAngle_per_frame "";
	variable vmdFrame "";
	variable helixNumber 0;
	variable MartiniBackbone "";
	variable MartiniHelix "";
	variable MartiniSheet "";
	variable MartiniBackboneNew "";
	variable MartiniHelixNew "";
	variable MartiniSheetNew "";
	variable list_of_indices_where_Chains_start "";
	variable previous_helix_type "";
	variable CG_Backbone_indexNs "";
	variable CG_Backbone_normalised_startsandstops "";
	variable quick_and_dirty 0;
	variable slow_and_pretty 0;
	variable helix_coord_at_startNstop "";
	variable helix_indices_at_startNstop "";
	variable heres_a_helix "";
	variable angle_per_turn_per_frame {}
	variable angle_per_AA_per_frame_BARRED ""
	variable helix_axis_per_helix_per_frame_BARRED ""
	variable AngleAutoScale 1;
	variable 3D_surf 0;
	variable z_squeeze 3.0;
	variable frame_squeeze 3.0;
	variable subset_by_backbone "";
	variable proteinID_by_backbone "";
	variable die 1;
	variable die_backbone 1;
	variable frame_done_before 0;
	variable first_surf_drawn "";
	variable first_axis_drawn "";
	variable last_axis_drawn "";
	variable axesON	0;
	variable surfed 0;
	variable Resolution_wrt_ends
	variable CG_old_sheet_rep_list ""
	variable First_backbone_drawn_list ""
	variable Last_backbone_drawn_list ""
	variable CG_old_rep_list ""
	variable old_rep_list ""
	variable CG_cartoonified_list 0
	variable cartoonified_list 0
	variable packed 0
	variable Sugeta_point 0
	variable surf_repID "";
	variable index_of_helix_assignment_used_for_angle "";
	variable StoreData 0;
	variable AAN_associated_w_splineIndex
	variable realAAs_for_23D_graph_Xaxis
	variable fixed_particleNames ""
	variable custom_particle 0
			 
	#### Selection warning variables
	variable tested_alt_particle_for_empty_select_AAs 0
	variable tested_alt_particle_for_empty_start_and_end_AA_N_of_chains 0
	variable tested_alt_particle_for_inexistent_start_and_end_AA_N_of_chains 0
	variable tested_alt_particle_for_empty_residueNs 0
	variable tested_alt_particle_for_empty_startIndex 0
	variable autolooper 0

	########## Speed-up variables
	variable previous_proteinID "0";
	variable previous_subset "";
	variable previous_helix_assignment "";
	variable previous_spline_resolution "4";
}


# ::bendix::bendix -------------------------------------------------------------
#    Create the GUI and run relevant procs if called via GUI.
#
#    Results: 
#    A robust GUI that allows the user to choose from available settings
#    and execute them by drawing to the VMD Display.
#    Examples of settings are helix assignment, treatment of non-bendices,
#    colouring, 2D and 3D graphing and resolution.
#    Analysis tools are reachable from the menubar.
#
#    Details of procs that are internal to ::bendix::bendix are given
#    when they appear, below.
#
# ------------------------------------------------------------------------------

proc ::bendix::bendix {} {
		
	puts stdout "	\nInfo) Bendix v1.1 by Caroline Dahl - software that lets your protein abstractions be naturally curvy.\
					\nInfo) http://sbcb.bioch.ox.ac.uk/Bendix/ \
					\nInfo) If Bendix contributes towards your publication, please cite:\
					\nInfo)    Dahl ACE, Chavent M and Sansom MSP (2012) \
					\nInfo)    Bendix: intuitive helix geometry analysis and abstraction. \
					\nInfo)    Bioinformatics 28 (16): 2193-2194\n"

	variable selection
	variable bix
	global vmd_frame;
	global vmd_initialize_structure
	global env
	global auto_path
	global vmd_molecule
	global vmd_quit

	#### Detect available molecules and set proteinID to an existent molecule.
	set N_loaded_molecules [molinfo num]
	set ID_mols [molinfo list]
	set ::bendix::proteinID [lindex $ID_mols 0]

	#### Needs a molecule?
	if {! [string compare $::bendix::proteinID top]} {
		set ::bendix::proteinID [molinfo top]
	}

	#### vmd_frame and vmd_molecule are VMD-managed arrays. When this array changes, call relevant Bendix procs.
	trace add variable vmd_molecule write {::bendix::set_molIDs;};
	trace add variable vmd_frame write ::bendix::sscache
	trace add variable vmd_quit write "::bendix::quit"

	#### Note the initial protein position in the VMD Display, so that Erase (and Reset) can get back to it:
	set N_loaded_Mols [molinfo num]
	if {$N_loaded_Mols != 0} {
		global original_protein_viewpoint
		set original_protein_viewpoint [molinfo $::bendix::proteinID get {center_matrix rotate_matrix scale_matrix global_matrix}]
	}

	#### Make a list out of a ::bendix::variable. Load lists with fifty unset or zero-variables
	#### bendix will update these lists as it runs, to keep track of what is drawn.
	foreach list_counter {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49} {
		foreach list_name {::bendix::First_backbone_drawn_list ::bendix::Last_backbone_drawn_list ::bendix::CG_old_sheet_rep_list ::bendix::CG_old_rep_list ::bendix::old_rep_list} {
			lappend $list_name ""
		}
		foreach cartoonified_list {::bendix::CG_cartoonified_list ::bendix::cartoonified_list} {
			lappend $cartoonified_list "0"
		}
	}


################################################################################
#                                  GUI SETUP                                   #
################################################################################
	
	if [winfo exists .bex] {
		wm deiconify $bix
		return
	}
	
	set bix [toplevel ".bex"]
	wm title $bix "bendix"
	wm attributes $bix -alpha 1.0;
	wm geometry $bix +100+100
	wm resizable $bix 0 0; 
	# Latter means non-resizable. Ensures that the 'forgotten' menu works 
	# and doesn't play up when the user (accidentally) resizes the window.

	frame $bix.top

	#### Menu
	frame $bix.menubar -relief raised -bd 2
	pack $bix.menubar -padx 1 -fill x -side top

	menubutton $bix.menubar.openfiles -text "File" -underline 0 -menu $bix.menubar.openfiles.menu
	$bix.menubar.openfiles config -width 5	
	pack $bix.menubar.openfiles -side left
	menu $bix.menubar.openfiles.menu -tearoff no    	
	$bix.menubar.openfiles.menu add command -label "Open helix assignment file..." -command ::bendix::openHelixFile
	$bix.menubar.openfiles.menu add command -label "Open colouring file..." -command ::bendix::openColourFile
	$bix.menubar.openfiles.menu add command -label "Save helix assignment..." -command ::bendix::saveHelixFile
	$bix.menubar.openfiles.menu add command -label "Save colouring..." -command ::bendix::saveColourFile
	$bix.menubar.openfiles.menu add command -label "Save surf data..." -command ::bendix::saveSurfData
	$bix.menubar.openfiles.menu add command -label "Save axes data..." -command ::bendix::saveAxisData
	$bix.menubar.openfiles.menu add command -label "Reset and Quit" -command ::bendix::quit
	
	menubutton $bix.menubar.analysis -text "Analysis" -underline 0 -menu $bix.menubar.analysis.menu	
	$bix.menubar.analysis config -width 8
	pack $bix.menubar.analysis -side left
	menu $bix.menubar.analysis.menu -tearoff no	
	$bix.menubar.analysis.menu add command -label "Plot: Angle along helix" -command ::bendix::create_graph -state disabled
	$bix.menubar.analysis.menu add command -label "Plot: Maximum angle per helix over time" -command ::bendix::create_MaxAngle_graph -state disabled
	$bix.menubar.analysis.menu add command -label "Surf: Angle along helix over time..." -command ::bendix::popup_for_Surf -state disabled
	$bix.menubar.analysis.menu add checkbutton -label "Store dynamic data" -variable ::bendix::StoreData -state disabled 
	$bix.menubar.analysis.menu add checkbutton -label "Store helix axes" -variable ::bendix::StoreAxes
	
	menubutton $bix.menubar.howto -text "Help" -underline 0 -menu $bix.menubar.howto.menu	
	$bix.menubar.howto config -width 5
	pack $bix.menubar.howto -side left
	menu $bix.menubar.howto.menu -tearoff no	
	$bix.menubar.howto.menu add command  -label "What is bendix?" -command ::bendix::shortbendix
	$bix.menubar.howto.menu add command  -label "Quick help!" -command ::bendix::quickBendix
	$bix.menubar.howto.menu add command  -label "Online Tutorial" -command "vmd_open_url {http://sbcb.bioch.ox.ac.uk/Bendix/tutorial.html}"
	$bix.menubar.howto.menu add command  -label "Online Troubleshooter" -command "vmd_open_url {http://sbcb.bioch.ox.ac.uk/Bendix/faqLess.html#QA}"
	
	menubutton $bix.menubar.bendixInfo -text "About" -underline 0 -menu $bix.menubar.bendixInfo.menu	
	$bix.menubar.bendixInfo config -width 7
	pack $bix.menubar.bendixInfo -side left
	menu $bix.menubar.bendixInfo.menu -tearoff no	
	$bix.menubar.bendixInfo.menu add command -label "Who made bendix?" -command ::bendix::WhoMadebendix

	#### Main GUI Body

	labelframe $bix.lf1 -text "Protein selection" -pady 3 -padx 6 -font {-weight bold -family helvetica}
	labelframe $bix.lf2 -text "Helix features" -pady 3 -padx 6 -font {-weight bold -family helvetica}
	labelframe $bix.lf3 -text "How to draw non-Bendices" -pady 3 -padx 6 -font {-weight bold -family helvetica}

	frame $bix.lf1.field1 -borderwidth 1
	frame $bix.lf1.field2 -borderwidth 1
	frame $bix.lf1.field3 -borderwidth 1
	frame $bix.lf1.field4 -borderwidth 1
	frame $bix.lf1.field5 -borderwidth 1

	frame $bix.lf2.field1 -borderwidth 1
	frame $bix.lf2.field4 -borderwidth 1
	frame $bix.lf2.field5 -borderwidth 1
	frame $bix.lf2.field6 -borderwidth 1
	frame $bix.lf2.field7 -borderwidth 1
	frame $bix.lf2.field8 -borderwidth 1

	frame $bix.lf3.field1 -borderwidth 1

	frame $bix.field1
	frame $bix.field2
	frame $bix.field3
	frame $bix.field4

	#### labelfield 1: 1-4
	message $bix.lf1.field1.msg -width 100 -text "Mol ID:"
	set N_loaded_molecules [molinfo num]
	set ID_mols [molinfo list]
	set N_mols ""
	for {set molN 0} {$molN < $N_loaded_molecules} {incr molN } {
		lappend N_mols [lindex $ID_mols $molN]
	}
	spinbox $bix.lf1.field1.molname -values $N_mols -state readonly -width 3
	checkbutton $bix.lf1.field1.cg -justify right -text "CG" -variable ::bendix::CG \
		-command ::bendix::MakeCG

	message $bix.lf1.field1.spacer -text "" -width 10
	message $bix.lf1.field1.name -text "Backbone particle(s):" -width 190
	entry $bix.lf1.field1.particleName -textvariable ::bendix::particle_name -background white \
		-selectbackground yellow2 -width 18
		
	message $bix.lf1.field2.msg -justify left -width 100 -text "Subset:"
	entry $bix.lf1.field2.molname -textvariable ::bendix::subset -background white \
		-selectbackground yellow2 -width 50

	message $bix.lf1.field3.msg -justify left -width 430 \
		-text "  For example 'resid 1 to 20 and (chain B or chain C) '. Default is the full protein." \
		-font {helvetica 8}
	message $bix.lf1.field4.msg -justify left -text "Helices:" -width 100
	entry $bix.lf1.field4.helices -textvariable ::bendix::helix_assignment -background white \
		-selectbackground yellow2 -width 42
	button $bix.lf1.field4.clear -command {set ::bendix::helix_assignment ""; set ::bendix::String_for_cartoonify "" } \
		-text "Clear" -font {helvetica 8} -background white
	message $bix.lf1.field5.msg -justify left -text "  Helix start and end resid numbers.\
If left blank, helicity is auto-assigned." -font {helvetica 8} -foreground black -width 430

	#### labelfield 2: 1-7
	radiobutton $bix.lf2.field1.straight -variable ::bendix::helix_type -value 0 -text "Straight helices" \
		-command ::bendix::cylinders
	radiobutton $bix.lf2.field1.bendix -variable ::bendix::helix_type -value 1 -text "Bendices: use every " \
		-command ::bendix::bendices
	set turns [list 1 2 3 4 5 6 7 8 9 10];
	spinbox $bix.lf2.field1.cp -values $turns -state readonly -width 3
	$bix.lf2.field1.cp set 4
	message $bix.lf2.field1.spacer -text "residues" -width 50
	
	radiobutton $bix.lf2.field5.colourbutton0 -variable ::bendix::uniform_colour -value 0 \
		-text "Heatmap color:" -command ::bendix::Hello_curvature
	
	message $bix.lf2.field5.msg2 -justify right -width 50 -text " Scale:" -foreground "gray60" -padx 1
	ttk::combobox $bix.lf2.field5.listbox -width 6 -values [list RWB BWR RGryB BGryR RGB BGR RWG GWR GWB BWG BlkW WBlk] -state readonly
	# NB. Requires the tile package.
	
	message $bix.lf2.field5.msg -justify right -width 100 -text " Color threshold:" -foreground "gray60" -padx 1
	entry $bix.lf2.field5.molname -textvariable ::bendix::angle_max -state disabled \
		-background white -selectbackground yellow2 -width 4 
	message $bix.lf2.field5.msg3 -justify right -width 30 -text "Side:" -foreground "gray60"
	entry $bix.lf2.field5.angleL -textvariable ::bendix::AngleSide -background white -selectbackground yellow2 \
		-width 3 -state disabled 
	
	radiobutton $bix.lf2.field6.colourbutton1 -variable ::bendix::uniform_colour -value 1 \
		-text "Uniform color:" \
		-command ::bendix::Hello_uniform
	message $bix.lf2.field6.spacer -text "  "
	radiobutton $bix.lf2.field6.uniformcolour0 -variable ::bendix::uniform_colour_type -value 0 -text "full selection" 
	radiobutton $bix.lf2.field6.uniformcolour1 -variable ::bendix::uniform_colour_type -value 1 -text "by chain" 
	radiobutton $bix.lf2.field6.uniformcolour2 -variable ::bendix::uniform_colour_type -value 2 -text "by helix" 

	message $bix.lf2.field7.msg -justify left -width 140 -text "Color(s):"
	entry $bix.lf2.field7.molname -textvariable ::bendix::input_colour -state disabled \
		-background white -selectbackground yellow2 -width 40
	checkbutton $bix.lf2.field7.checkON -text "Auto" -variable ::bendix::autoColour -command ::bendix::autoON

	message $bix.lf2.field8.msg -width 200 -text "Separate colors by space" -font {helvetica 8} \
		-foreground "gray60"
	
	message $bix.lf2.msg -text "Material:" -width 100 
	ttk::combobox $bix.lf2.listbox -width 15 -values [list Opaque Transparent BrushedMetal Diffuse Ghost Glass1 Glass2 Glass3 Glossy HardPlastic MetallicPastel Steel Translucent Edgy EdgyShiny EdgyGlass Goodsell AOShiny AOChalky AOEdgy BlownGlass GlassBubble] -state readonly
	# NB. Requires the tile package.
	
	message $bix.lf2.spacer -text ""
	message $bix.lf2.msg2 -justify left -width 60 -text "Radius:"
	entry $bix.lf2.molname -textvariable ::bendix::helix_radius -background white -selectbackground yellow2 \
		-width 4 
	message $bix.lf2.msg3 -justify left -width 70 -text "Resolution:"
	entry $bix.lf2.molname2 -textvariable ::bendix::spline_resolution -background white -selectbackground yellow2 \
		-width 3 

	#### labelframe 3: Reps for else than Bendified
	checkbutton $bix.lf3.field1.rep0 -text "Rep 0" -variable ::bendix::rep_zero -command ::bendix::hiderep_zero
	message $bix.lf3.field1.spacer -text "   "
	checkbutton $bix.lf3.field1.cartoon -text "Cartoon" -variable ::bendix::autoCartoon \
		-command ::bendix::cartoonify
	checkbutton $bix.lf3.field1.newcartoon -text "NewCartoon" -variable ::bendix::autoNewCartoon \
		-command ::bendix::cartoonify
	checkbutton $bix.lf3.field1.tube -text "Tube" -variable ::bendix::autoTube -command ::bendix::cartoonify
	message $bix.lf3.field1.spacer2 -text "  "
	checkbutton $bix.lf3.field1.quick -text "Draft" -variable ::bendix::quick_and_dirty \
		-command ::bendix::drawBackbone 
	checkbutton $bix.lf3.field1.slow -text "Join" -variable ::bendix::slow_and_pretty \
		-command ::bendix::drawBackbone 

	button $bix.field2.boutonaffiche0 -command {set ::bendix::autolooper 0; ::bendix::mainDrawingLoop} \
		-text "Draw" -background green
	button $bix.field2.boutonaffiche1 -command ::bendix::erase -text "Erase" -background tomato -width 5
	button $bix.field2.settings -command {
						bind .bex <Unmap> { }
						if {$::bendix::packed == 0} {
							pack forget .bex.lf2
							pack forget .bex.lf2.field5
							pack forget .bex.lf2.field6
							pack forget .bex.lf3

							pack .bex.lf1 -side top -padx 4 -anchor w -expand true -fill x -pady 4
							pack .bex.lf2.field1 -side top
							pack .bex.lf2.field5 -side top -anchor w
							pack .bex.lf2.field6 -side top -anchor w
							pack .bex.lf2.field7 -side top -anchor w
							pack .bex.lf2.field8 -side top
							pack .bex.lf2 -side top -ipady 5 -padx 4 -anchor w -expand true -fill x
							pack .bex.lf2.msg -side left
							pack .bex.lf2.listbox -side left
							pack .bex.lf2.spacer -side left
							pack .bex.lf2.molname2 -side right
							pack .bex.lf2.msg3 -side right
							pack .bex.lf2.molname -side right
							pack .bex.lf2.msg2 -side right
							pack .bex.lf3 -side top -pady 4 -padx 4 -anchor w -expand true -fill x
							.bex.field2.settings configure -text "Hide settings"
						} else {
							pack forget .bex.lf3
							pack forget .bex.lf1
							pack forget .bex.lf2.field1
							pack forget .bex.lf2.field7
							pack forget .bex.lf2.field8
							pack forget .bex.lf2.msg
							pack forget .bex.lf2.listbox
							pack forget .bex.lf2.spacer -side left
							pack forget .bex.lf2.molname2 -side right
							pack forget .bex.lf2.msg3 -side right
							pack forget .bex.lf2.molname -side right
							pack forget .bex.lf2.msg2 -side right
							pack .bex.lf3
							.bex.field2.settings configure -text "Settings"
						}
						bind .bex <Unmap> {::bendix::quit}
						set ::bendix::packed [expr 1 - $::bendix::packed]
					} -text "Settings" -width 10 -font {helvetica 9}

	#### Visualise:
	pack $bix.lf1.field1 -side top -anchor w
	pack $bix.lf1.field1.msg -side left
	pack $bix.lf1.field1.molname -side left
	pack $bix.lf1.field1.spacer -side left
	pack $bix.lf1.field1.name -side left
	pack $bix.lf1.field1.particleName -side left
	pack $bix.lf1.field1.cg -side right -padx 3

	pack $bix.lf1.field2 -side top -anchor w
	pack $bix.lf1.field2.msg -side left
	pack $bix.lf1.field2.molname -side right -anchor e -expand true -fill x

	pack $bix.lf1.field3 -side top -anchor w
	pack $bix.lf1.field3.msg -side left 

	pack $bix.lf1.field4 -side top -anchor w
	pack $bix.lf1.field4.msg -side left
	pack $bix.lf1.field4.clear -side right -padx 2
	pack $bix.lf1.field4.helices -side right -expand true -fill x

	pack $bix.lf1.field5 -anchor w
	pack $bix.lf1.field5.msg -side left

	pack $bix.lf1 -side top -padx 4 -anchor w -expand true -fill x -pady 4
	pack forget $bix.lf1

	#### First packed gets its selected location
	pack $bix.lf2.field1 -side top -anchor w
	pack $bix.lf2.field1.straight -side left
	pack $bix.lf2.field1.bendix -side left
	pack $bix.lf2.field1.spacer -side right
	pack $bix.lf2.field1.cp -side right
	
	pack $bix.lf2.field5 -side top -anchor w
	pack $bix.lf2.field5.colourbutton0 -side left -anchor w
	pack $bix.lf2.field5.angleL -side right
	pack $bix.lf2.field5.msg3 -side right
	pack $bix.lf2.field5.molname -side right
	pack $bix.lf2.field5.msg -side right
	pack $bix.lf2.field5.listbox -side right
	pack $bix.lf2.field5.msg2 -side right
	
	pack $bix.lf2.field6 -side top -anchor w
	pack $bix.lf2.field6.colourbutton1 -side left
	pack $bix.lf2.field6.spacer -side left
	pack $bix.lf2.field6.uniformcolour2 -side right
	pack $bix.lf2.field6.uniformcolour1 -side right
	pack $bix.lf2.field6.uniformcolour0 -side right

	pack $bix.lf2.field7 -side top -anchor w
	pack $bix.lf2.field7.msg -side left
	pack $bix.lf2.field7.checkON -side right
	pack $bix.lf2.field7.molname -side right

	pack $bix.lf2.field8 -side top
	pack $bix.lf2.field8.msg -side bottom

	pack $bix.lf2 -side top -ipady 5 -padx 4 -anchor w -expand true -fill x
	pack $bix.lf2.msg -side left
	pack $bix.lf2.listbox -side left
	pack $bix.lf2.spacer -side left
	pack $bix.lf2.molname2 -side right
	pack $bix.lf2.msg3 -side right
	pack $bix.lf2.molname -side right
	pack $bix.lf2.msg2 -side right

	pack forget $bix.lf2.field1
	pack forget $bix.lf2.field7
	pack forget $bix.lf2.field8
	pack forget $bix.lf2.msg
	pack forget $bix.lf2.listbox
	pack forget $bix.lf2.spacer -side left
	pack forget $bix.lf2.molname2 -side right
	pack forget $bix.lf2.msg3 -side right
	pack forget $bix.lf2.molname -side right
	pack forget $bix.lf2.msg2 -side right

	####
	pack $bix.lf3.field1 -side top -padx 2
	pack $bix.lf3.field1.rep0 -side left
	pack $bix.lf3.field1.spacer -side left
	pack $bix.lf3.field1.slow -side right
	pack $bix.lf3.field1.quick -side right
	pack $bix.lf3.field1.spacer2 -side right
	pack $bix.lf3.field1.newcartoon -side right
	pack $bix.lf3.field1.cartoon -side right
	pack $bix.lf3.field1.tube -side right

	pack $bix.lf3 -side top -pady 4 -padx 4 -anchor w -expand true -fill x

	####
	pack $bix.field2 -side bottom
	pack $bix.field2.boutonaffiche0 -side left -padx 10 -pady 5
	pack $bix.field2.settings -side right -padx 10 -pady 5
	pack $bix.field2.boutonaffiche1 -side right -padx 10 -pady 5
  
    #### Upon window close (..or minimisation), exit Bendix properly.
    bind $bix <Unmap> {
		::bendix::quit
	}

	## Why does the wm for Delete window not work? 
	## In order to quit and reset upon Bendix window closure, 
	## I was forced to bind the quit and reset function to Unmap instead,
	## which unfortunately works on minimisation, too. 
	#	wm protocol .bex WM_DELETE_WINDOW {
	#		::bendix::quit
	#	}
	
	#### Hit Return to run the Bendix main drawing loop
    bind $bix <Return> {set ::bendix::autolooper 0; ::bendix::mainDrawingLoop}

################################################################################
#                                GUI PROCS                                     #
################################################################################


# ::bendix::MakeCG -------------------------------------------------------------
#    Updates the GUI to dis/allow CG-related settings,
#    update fields with default CG or AT settings
#    and display/hide ir/relevant text.
#    If CG, it stores what particle names to seek in the protein,
#    depending on the sought secondary structure.
# ------------------------------------------------------------------------------

  proc ::bendix::MakeCG {} {
	if {$::bendix::CG == 1} {
		.bex.lf3.field1.cartoon configure -state disable
		.bex.lf3.field1.newcartoon configure -state disable
		.bex.lf3.field1.tube configure -state disable

		set ::bendix::MartiniBackbone "B\[CNESTH\].*"
		set ::bendix::MartiniHelix "BH.*"
		set ::bendix::MartiniSheet "BE.*"
		set ::bendix::MartiniBackboneNew "BB\[cnesth\].*"
		set ::bendix::MartiniHelixNew "BBh.*"
		set ::bendix::MartiniSheetNew "BBe.*"
		set ::bendix::particle_name "CA B.*"

	} else {
		if {$::bendix::autoCartoon == 1} {
			.bex.lf3.field1.cartoon configure -state normal
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.tube configure -state disable
			.bex.lf3.field1.slow configure -state disable 
			.bex.lf3.field1.quick configure -state disable
		} elseif {$::bendix::autoNewCartoon == 1} {
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.newcartoon configure -state normal
			.bex.lf3.field1.tube configure -state disable
			.bex.lf3.field1.slow configure -state disable 
			.bex.lf3.field1.quick configure -state disable
		} elseif {$::bendix::autoTube == 1} {
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.tube configure -state normal
			.bex.lf3.field1.slow configure -state disable 
			.bex.lf3.field1.quick configure -state disable
		} elseif {$::bendix::slow_and_pretty == 1} {
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.tube configure -state disable
			.bex.lf3.field1.slow configure -state normal 
			.bex.lf3.field1.quick configure -state disable
		} elseif {$::bendix::quick_and_dirty == 1} {
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.tube configure -state disable
			.bex.lf3.field1.slow configure -state disable 
			.bex.lf3.field1.quick configure -state normal
		} else {
			.bex.lf3.field1.cartoon configure -state normal
			.bex.lf3.field1.newcartoon configure -state normal
			.bex.lf3.field1.tube configure -state normal
			.bex.lf3.field1.slow configure -state normal
			.bex.lf3.field1.quick configure -state normal
		}
		set ::bendix::particle_name "CA"
	}
  }

# ::bendix::Hello_curvature ----------------------------------------------------
#    Updates the GUI to allow Bendix-related settings,
#    display/hide or dis/enable ir/relevant text or choices
# ------------------------------------------------------------------------------

  proc ::bendix::Hello_curvature { } {
	.bex.lf2.field5.molname configure -state normal
	.bex.lf2.field7.molname configure -state disable
	.bex.lf2.field7.checkON configure -state disable
	.bex.lf2.field6.uniformcolour0 configure -state disable
	.bex.lf2.field6.uniformcolour1 configure -state disable
	.bex.lf2.field6.uniformcolour2 configure -state disable
	.bex.lf2.field5.msg3 configure -foreground black
	.bex.lf2.field5.msg configure -foreground black
	.bex.lf2.field5.msg2 configure -foreground black
	.bex.lf2.field5.angleL configure -state normal
	.bex.lf2.field7.msg configure -foreground "gray60"
	.bex.lf2.field8.msg configure -foreground "gray60"
	.bex.menubar.analysis.menu entryconfigure 0 -state normal
	.bex.menubar.analysis.menu entryconfigure 1 -state normal
	.bex.menubar.analysis.menu entryconfigure 2 -state normal
	.bex.menubar.analysis.menu entryconfigure 3 -state normal
  }

# ::bendix::Hello_uniform ------------------------------------------------------
#    Updates the GUI to allow uniform-colouring-related settings
#    and disable/hide irrelevant text or choices.
# ------------------------------------------------------------------------------

  proc ::bendix::Hello_uniform { } {
	.bex.lf2.field5.molname configure -state disable
	if {$::bendix::autoColour != 1} {
		.bex.lf2.field7.molname configure -state normal
		.bex.lf2.field8.msg configure -foreground black
	}
	.bex.lf2.field7.checkON configure -state normal
	.bex.lf2.field6.uniformcolour0 configure -state normal
	.bex.lf2.field6.uniformcolour1 configure -state normal
	.bex.lf2.field6.uniformcolour2 configure -state normal
	.bex.lf2.field5.msg configure -foreground "gray60"
	.bex.lf2.field5.msg2 configure -foreground "gray60"
	.bex.lf2.field5.msg3 configure -foreground "gray60"
	.bex.lf2.field5.angleL configure -state disabled
	.bex.lf2.field7.msg configure -foreground black
	.bex.menubar.analysis.menu entryconfigure 0 -state disabled
	.bex.menubar.analysis.menu entryconfigure 1 -state disabled
	.bex.menubar.analysis.menu entryconfigure 2 -state disabled
	.bex.menubar.analysis.menu entryconfigure 3 -state disabled
  }

# ::bendix::cylinders ----------------------------------------------------------
#    Updates the GUI to allow [cylinder rendition of helices]-related
#    settings and disable/hide irrelevant text or choices.
# ------------------------------------------------------------------------------

  proc ::bendix::cylinders {} {
	set ::bendix::uniform_colour 1
	.bex.lf2.field5.colourbutton0 configure -state disable
	.bex.lf2.field1.cp configure -state disabled
	::bendix::Hello_uniform
  }

# ::bendix::bendices -----------------------------------------------------------
#    Updates the GUI to allow bendix-related settings
#    and disable/hide irrelevant text or choices.
# ------------------------------------------------------------------------------

  proc ::bendix::bendices {} {
	if {$::bendix::uniform_colour == 0 } {
		.bex.lf2.field5.molname configure -state normal
		.bex.lf2.field5.msg configure -foreground black
		.bex.lf2.field5.msg2 configure -foreground black
		.bex.lf2.field5.msg3 configure -foreground black
	}
	.bex.lf2.field5.colourbutton0 configure -state normal
	.bex.lf2.field1.cp configure -state readonly
  }

# ::bendix::autoON -------------------------------------------------------------
#    Updates the GUI to dis/allow anto-colouring of helices
#    and dis/enable the colour input field.
# ------------------------------------------------------------------------------ 

  proc ::bendix::autoON {} {
	if {$::bendix::autoColour == 1} {
		.bex.lf2.field7.molname configure -state disable
		.bex.lf2.field8.msg configure -foreground "gray60"
	} else {
		.bex.lf2.field7.molname configure -state normal
		.bex.lf2.field8.msg configure -foreground black
	}
  }


################################################################################
#                            LOAD AND SAVE FILES                               #
################################################################################

# ::bendix::openHelixFile ------------------------------------------------------
#    Pop-ups the GUI for loading helix assignment from a previously saved file. 
#    Inserts found text in the ::bendix::helix_assignment field and variable,
#    and closes the file.
# ------------------------------------------------------------------------------

  proc ::bendix::openHelixFile { } {
	set myHelixAssignmentFile [tk_getOpenFile -title "Load helix assignment"]
	if { $myHelixAssignmentFile == "" } {
		return;
	}
	set helixfileID [open $myHelixAssignmentFile r];
	set ::bendix::helix_assignment ""
	while { [gets $helixfileID line] >= 0 } {
		.bex.lf1.field4.helices insert end [format "%s" $line]
	}
	close $helixfileID;
#	close $myHelixAssignmentFile;
  }

# ::bendix::openColourFile -----------------------------------------------------
#    Pop-ups the GUI for loading helix colouring from a previously saved file.
#    Edits the GUI display to uncheck the Auto-colouring tickbox
#    and tick the uniform colour box.
#    Calls the ::bendix::Hello_uniform proc to edit the GUI further.
#    Erases any previous ::bendix::input_colour field/variable data
#    and inserts found text, and closes the file.
# ------------------------------------------------------------------------------

  proc ::bendix::openColourFile { } {
	.bex.lf2.field7.molname configure -state normal
	.bex.lf2.field8.msg configure -foreground black
	set ::bendix::autoColour 0
	set ::bendix::uniform_colour 1
	::bendix::Hello_uniform
	set ::bendix::input_colour ""
	set myColourFile [tk_getOpenFile -title "Load helix colour scheme"]
	if { $myColourFile == "" } {
		return;
	}
	set colourFileID [open $myColourFile r];
	while { [gets $colourFileID line] >= 0 } {
		.bex.lf2.field7.molname insert end [format "%s" $line]
	}
	close $colourFileID;
  }

# ::bendix::saveHelixFile ------------------------------------------------------
#    Pop-ups the GUI for saving helix starts and stops
#    in the ::bendix::helix_assignment field and variable to file.
# ------------------------------------------------------------------------------

  proc ::bendix::saveHelixFile {} {
	set myHelixSaveFile [tk_getSaveFile -title "Save helix assignment"]
	if { $myHelixSaveFile == "" } {
		return;
	}
	set helixSavefileID [open $myHelixSaveFile w]
	puts $helixSavefileID $::bendix::helix_assignment
	close $helixSavefileID;
  }

# ::bendix::saveColourFile -----------------------------------------------------
#    Pop-ups the GUI for saving colour per helix
#    in the ::bendix::input_colour field and variable to file.
# ------------------------------------------------------------------------------

  proc ::bendix::saveColourFile {} {
	set mySaveHelixColourFile [tk_getSaveFile -title "Save helix colour scheme"]
	if { $mySaveHelixColourFile == "" } {
		return;
	}
	set fileSaveHelixColourID [open $mySaveHelixColourFile w]
	puts $fileSaveHelixColourID $::bendix::input_colour
	close $fileSaveHelixColourID;
  }

# ::bendix::saveSurfData -------------------------------------------------------
#    Edits collected frame number, helix residue and angle data
#    to be tab- and newline-spaced. This is done to conform
#    to major graphing software standards for input data,
#    so it can be copy-pasted.
#    Pop-ups the GUI for saving this data, writes to the user-specified file
#    and closes the file.
# ------------------------------------------------------------------------------

  proc ::bendix::saveSurfData {} {
	set one_row ""
	set all_rows ""
	set one_row_tabbed ""
	set all_rows_tabbed ""
	set all_turns "0\t"
	foreach turnN $::bendix::curvature_graph_X {
		append all_turns $turnN "\t"
	}
	foreach item $::bendix::angle_per_AA_per_frame_BARRED {
		if {([expr int($item)] == $item) && $item != "0.0"} {
			if {$one_row != ""} {
				append all_rows $one_row "\n"
				set one_row ""
			}
			if {$one_row_tabbed != ""} {
				append all_rows_tabbed $one_row_tabbed "\n"
				set one_row_tabbed ""
			}
		}
		append one_row $item " "
		append one_row_tabbed $item "\t"
	}
	append all_rows $one_row
	append all_rows_tabbed $one_row_tabbed
	set myDataFile [tk_getSaveFile -title "Save surf data"]
	if { $myDataFile == "" } {
		return;
	}
	set fileDataID [open $myDataFile w]
	puts $fileDataID "Hi! Below is your helix curvature data.\
		\n\nFor clear results, open this document in an editor that recognises newline characters.\
		\nColumns are delimited by tabs, and rows by newline characters.\
		\n\nRow 1 = Residues. If you chose to save multiple helices' data, these are listed sequentially\
		(as are angles).\n\nColumn 1 = Frame numbers, in the order that you played the trajectory.\
		To collect data starting from a different frame, load your frame of choice, hit \[Erase\]\
		in the bendix main window (this deletes any collected data), \[Draw\] and play your trajectory.\
		\n\n\n==================================================================================\n\n\
		$all_turns\n$all_rows_tabbed"
	close $fileDataID;
  }
  
# ::bendix::saveAxisData -------------------------------------------------------
#    Checks that data exists; if it does not it prompts the user to turn on coordinate storage.
#    Pop-ups the GUI for saving coordinate data, writes to the user-specified file
#    and closes the file.
# ------------------------------------------------------------------------------

  proc ::bendix::saveAxisData {} {
	
	if {$::bendix::helix_axis_per_helix_per_frame_BARRED == ""} {
		#### Where there's nothing to graph, return error.
		catch {destroy .messpopNoAxesCoords}
		toplevel .messpopNoAxesCoords
		wm geometry .messpopNoAxesCoords +100+150
		#grab .messpopNoAxesCoords
		wm title .messpopNoAxesCoords "No axis data to save yet."
		message .messpopNoAxesCoords.msg1 -width 350 \
-text "Bendix detects no saved coordinate data to write to file.\
\nYou need to enable axis coordinate storage by selecting 'Store helix axes' under Bendix' Analysis menu.\nThen redraw your protein (by clicking Erase followed by Draw) to generate axis data." -pady 15 -padx 10
		button .messpopNoAxesCoords.ok -text Return -background green \
			-command {
				destroy .messpopNoAxesCoords
	 			return 0
			}
		pack .messpopNoAxesCoords.msg1 -side top 
		pack .messpopNoAxesCoords.ok -side bottom -pady 5

	    #### Upon window close or minimisation, exit properly.
	    #bind .messpopNoAxesCoords <Unmap> {
		#	destroy .messpopNoAxesCoords
		#	return 0
		#}

	} else {
	#### Save stored coordinates:
	
		set myAxisFile [tk_getSaveFile -title "Save axes data"]
		if { $myAxisFile == "" } {
			return;
		}
		set fileDataID [open $myAxisFile w]
		puts $fileDataID "Hi! Below is your Bendix axis data.\
			\nN.B. The helix axis algorithm requires 5 or more residues to generate an axis, so only helices that contain 5 or more residues are listed here.\
			\nFor more information, see http://sbcb.bioch.ox.ac.uk/Bendix/faqLess_Tech.html or the original paper by Sugeta and Miyazawa in Biopolymers 1967 5(7):673-679.
			\n\nCoordinates are tabbed and different helices' data are separated by newline characters. For clear results, open this document in an editor that recognises newline and tab characters.\
			\nFrame numbers are in the order that you played the trajectory.\
			To collect data starting from a different frame, load your frame of choice, hit \[Erase\]\
			in the bendix main window (this deletes any collected data), \[Draw\] and play your trajectory.\
			\n\n\n==================================================================================
			$::bendix::helix_axis_per_helix_per_frame_BARRED\n"
			close $fileDataID;
	}
  }


################################################################################
#                        USER HELP AND ACKNOWLEDGEMENTS                        #
################################################################################

# ::bendix::quickBendix --------------------------------------------------------
#    Sources a picture which shows how to use the Bendix GUI, as a pop-up
# ------------------------------------------------------------------------------

  proc ::bendix::quickBendix {} {
	global get_file
	global auto_path
	global env
	catch {destroy .pop_howto};
	toplevel .pop_howto -background white
	wm geometry .pop_howto +100+100
	wm title .pop_howto "Welcome!"
	wm protocol .pop_howto WM_DELETE_WINDOW {
		destroy .pop_howto
		return
	}
	set directFile [file join $env(VMDDIR) plugins noarch tcl bendix1.1 GUI-tips.gif]
	set gui_howto [image create photo -file $directFile]

	label .pop_howto.gui -image $gui_howto -background white
	button .pop_howto.okb -text Return -background green -command {
	destroy .pop_howto ; return }

	pack .pop_howto.gui -side top -padx 5 -pady 5
	pack .pop_howto.okb -side bottom -pady 10
  }

# ::bendix::shortbendix --------------------------------------------------------
#    Creates a pop-up which explains what Bendix is for,
#    and its compatibility with VMD.
# ------------------------------------------------------------------------------

  proc ::bendix::shortbendix {} {
	catch {destroy .pop_whatis}
	toplevel .pop_whatis -background white
	wm geometry .pop_whatis +100+150
	#grab .pop_whatis
	wm title .pop_whatis "This is bendix."
	message .pop_whatis.msg0 -width 300 -text "" -pady 5 -background white
	message .pop_whatis.msg1 -width 300 \
		-text "bendix is a plugin that makes it easier to characterize proteins.\
It allows you to calculate and visualize both dynamic and static helix geometry,\
and abstracts helices without sacrificing conformation.\
Moreover it accepts both coarse-grained and atomistic proteins." -padx 15 -pady 10 -background white
	message .pop_whatis.msg2 -width 300 -text "bendix allows you to display protein helices as cylinders\
that follow the helix axis. This captures conformational information that is lost if you use classical,\
straight cylinders. Coarse-grained proteins also benefit from backbone and beta-sheet display.\
\nHelix geometry can be analysed both qualitatively and quantitatively, and data is 2D or 3D graphed,\
with the option to export for viewing with common graphing packages.\
\nWith its multiple settings, bendix is easily tailored."\
		-padx 15 -pady 10 -background white
	message .pop_whatis.msg3 -width 280 -text "bendix integrates with the rest of VMD,\
so there is nothing stopping you from using bendix in addition to, say, Graphical Representations."\
		-padx 15 -pady 10 -background white
		
	message .pop_whatis.msg4 -width 280 -text "For more information, please refer to" -padx 15 -background white
	button .pop_whatis.bendixwebsite -text "the bendix website" -command "vmd_open_url {http://sbcb.bioch.ox.ac.uk/Bendix/index.html}"
	message .pop_whatis.msg5 -width 280 -text "or" -padx 15 -background white
	button .pop_whatis.abstractwebsite -text "the original paper" -command "vmd_open_url {http://bioinformatics.oxfordjournals.org/cgi/content/abstract/bts357?ijkey=HOmiKCIgVkhB9bg&keytype=ref}"
	message .pop_whatis.msg6 -width 280 -text "\nEnjoy!" -padx 15 -background white
	global env
	set directFile2 [file join $env(VMDDIR) plugins noarch tcl bendix1.1 Monsieur_Bendix_smaller_still.gif]
	set smallBendixGIF [image create photo -file $directFile2]
	label .pop_whatis.bendixGIF -image $smallBendixGIF -background white
	button .pop_whatis.okb -text Return -command {destroy .pop_whatis ; return }

	pack .pop_whatis.msg0 .pop_whatis.msg1 .pop_whatis.msg2 .pop_whatis.msg3 -side top -anchor w
	pack .pop_whatis.msg4 .pop_whatis.bendixwebsite .pop_whatis.msg5 .pop_whatis.abstractwebsite .pop_whatis.msg6 -side top 
	pack .pop_whatis.bendixGIF -side top -pady 5
	pack .pop_whatis.okb -side bottom -pady 5
	

  }

# ::bendix::WhoMadebendix ------------------------------------------------------
#    Creates a pop-up which states authorship and contact details.
# ------------------------------------------------------------------------------

  proc ::bendix::WhoMadebendix {} {
	catch {destroy .pop_whodid}
	toplevel .pop_whodid
	wm geometry .pop_whodid +100+150
#	grab .pop_whodid
	wm title .pop_whodid "Acknowledgements"
	frame .pop_whodid.field1
	frame .pop_whodid.field2
	frame .pop_whodid.field3

	message .pop_whodid.msg1 -width 330 \
		-text "\n\nBendix is developed by Caroline Dahl, part of the Structural Bioinformatics and Computational Biochemistry unit at the University of Oxford.\
\n\nDr Matthieu Chavent and Professor Mark Sansom supervised the project.\n\nIf Bendix contributes towards your publication, please cite:" -padx 10
	message .pop_whodid.cite -width 330 -text "Dahl ACE, Chavent M and Sansom MSP (2012)\nBendix: intuitive helix geometry analysis and abstraction.\nBioinformatics 28 (16): 2193-2194" -padx 10 -foreground  SlateGray4
	message .pop_whodid.field1.msg2 -width 350 -text "\nTo get in touch, please email"
	message .pop_whodid.field1.msg3 -width 350 -text "caroline.dahl AT dtc.ox.ac.uk\n"

	button .pop_whodid.field2.www -text "or check the website" -command "vmd_open_url {http://sbcb.bioch.ox.ac.uk/currentmembers/dahl.php}"
	message .pop_whodid.field3.spacer -text " "
	button .pop_whodid.okb -text Return -background green -command {destroy .pop_whodid ; return }

	pack .pop_whodid.msg1 .pop_whodid.cite -side top

	pack .pop_whodid.field1 -side top
	pack .pop_whodid.field1.msg2 .pop_whodid.field1.msg3 -side top

	pack .pop_whodid.field2 -side top
	pack .pop_whodid.field2.www -side bottom -pady 5

	pack .pop_whodid.field3
	pack .pop_whodid.field3.spacer -side top

	pack .pop_whodid.okb -side bottom -pady 10

  }


################################################################################
#                             DRAW NON-BENDICES                                #
################################################################################

# ::bendix::drawBackbone -------------------------------------------------------
#    Iff CG backbone is not already drawn to screen, depicts subset CG protein
#    that has not been assigned helicity in the associated
#    $::bendix::helix_assignment field.
#
#    The user chooses from Slow or Fast backbone implementation.
#
#    Slow: Graphics primitives that join helices
#    Slow notes indicies of helix starts and stops, and compares these
#    to backbone indices to ID where backbone is adjacent to helix. 
#    If adjacent, a line is drawn betwen the saved geometrical centre
#    of the helix end and the backbone start, and the coordinates
#    of the geometrical centre of the helix end are added
#    to the backbone's coordinates for future reference.
#    Backbone is drawn by joining consequitive backbone indices by lines
#    unless the interdistance is > 1nm. Slow is modulated so that, if run again
#    with no change to helix assignment or subset, stored backbone indices
#    and their starts and stops allow rapid re-graphing (good for trajectories).
#    Sheets are depicted by lines drawn between consequitive sheet particles
#    using set radius 0.7 nm and a 1.2 nm cone between the last two indices.
#    Backbone graphics primitives' identities are stored for easy removal.
#
#    Fast: Dynamic Bonds Representation
#    Fast adds a 0.2 radius, 4.6 Angtrom Dynamic Bonds Representation
#    of non-helix-assigned indices. A second, 0.7 radius, 4.6 Angstrom
#    Dynamic Bonds Representation is created for sheets.
#    Backbone Representation identities are stored for easy removal.
# ------------------------------------------------------------------------------

  proc ::bendix::drawBackbone {} {
	  
	#### Check that a molecule is loaded. If not untick boxes.
	set N_loaded_Mols [molinfo num]
	if {$N_loaded_Mols == 0} {
		if {$::bendix::quick_and_dirty == 1} {
			set ::bendix::quick_and_dirty 0
		} elseif {$::bendix::slow_and_pretty == 1} {
			set ::bendix::slow_and_pretty 0
		}
		return
	}

	#### Update the proteinID, since ::cartoonify is not connected to the MainFunction
	set ::bendix::proteinID [.bex.lf1.field1.molname get]
	
	#### Fix material
	set material_choice [.bex.lf2.listbox get]
	if {$material_choice == ""} {
		graphics $::bendix::proteinID material EdgyShiny
	} else {
		graphics $::bendix::proteinID material $material_choice
	}

	if {$::bendix::slow_and_pretty == 1 || $::bendix::quick_and_dirty == 1} {
		set ::bendix::die_backbone 0
	} else {
		set ::bendix::die_backbone 1
		.bex.lf3.field1.quick configure -state normal
		.bex.lf3.field1.slow configure -state normal
	}


	#### If the frame updates or Erase is on, delete existent graphics representations.
	if {($::bendix::previous_frame != $::bendix::frame) || $::bendix::die_backbone == 1 } {
		if { $::bendix::die_backbone == 1 } {
		#### Only erase the Rep if no backbone is wanted at all. It updates automatically.
			if { [lindex $::bendix::CG_old_sheet_rep_list $::bendix::proteinID] != ""} {
				mol delrep [lindex $::bendix::CG_old_sheet_rep_list $::bendix::proteinID] $::bendix::proteinID
				lset ::bendix::CG_old_sheet_rep_list $::bendix::proteinID ""
			}
			if { [lindex $::bendix::CG_old_rep_list $::bendix::proteinID] != ""} {
				mol delrep [lindex $::bendix::CG_old_rep_list $::bendix::proteinID] $::bendix::proteinID
				lset ::bendix::CG_old_rep_list $::bendix::proteinID ""
				lset ::bendix::CG_cartoonified_list $::bendix::proteinID 0
			}
		}
		if { [lindex $::bendix::First_backbone_drawn_list $::bendix::proteinID] != "" && [lindex $::bendix::Last_backbone_drawn_list $::bendix::proteinID] != ""} {
			for {set kill [lindex $::bendix::First_backbone_drawn_list $::bendix::proteinID]} {$kill <= [lindex $::bendix::Last_backbone_drawn_list $::bendix::proteinID]} {incr kill} {
				graphics $::bendix::proteinID delete $kill 
			}
			lset ::bendix::CG_cartoonified_list $::bendix::proteinID 0
		}	
	
	}
	#### Implemented auto-set variables to cater to auto-erase at Draw:
	lset ::bendix::First_backbone_drawn_list $::bendix::proteinID ""
	lset ::bendix::Last_backbone_drawn_list $::bendix::proteinID ""

	if { $::bendix::die_backbone == 0} {
		
		### Check the particle type validity if either of these have changed:
		### proteinID, subset or particle_name
		### or if fixed_particleNames is empty.
		if {$::bendix::previous_proteinID != $::bendix::proteinID || $::bendix::previous_subset != $::bendix::subset || $::bendix::previous_particle_name != $::bendix::particle_name || $::bendix::fixed_particleNames == ""} {
			::bendix::ParticleSubsetOK
		}
		
		######################### Slow and Accurate #############################
		if {$::bendix::slow_and_pretty == 1 && $::bendix::quick_and_dirty == 0} {
			.bex.lf3.field1.quick configure -state disabled
			lset ::bendix::CG_cartoonified_list $::bendix::proteinID 1
			#### Note the N of reps already made so that bendix may erase the CG-backbone when necessary.
			set AllThingsDrawn [graphics $::bendix::proteinID list]
			if {$AllThingsDrawn != ""} {
				lset ::bendix::First_backbone_drawn_list $::bendix::proteinID [expr {[lindex $AllThingsDrawn [expr {[llength $AllThingsDrawn] -1}]] +1}]
			} else {
				lset ::bendix::First_backbone_drawn_list $::bendix::proteinID 0
			}
			if {$::bendix::CG_Backbone_indexNs == "" || $::bendix::CG_Backbone_normalised_startsandstops == "" || $::bendix::helix_assignment != $::bendix::previous_helix_assignment || $::bendix::helix_assignment != $::bendix::helix_assignment_by_backbone || $::bendix::subset != $::bendix::previous_subset || $::bendix::subset != $::bendix::subset_by_backbone || $::bendix::proteinID != $::bendix::previous_proteinID || $::bendix::proteinID != $::bendix::proteinID_by_backbone} {
				### Clear variables so they can be re-written to reflect a new subset or helicity, where necessary. 
				set ::bendix::CG_Backbone_normalised_startsandstops ""

				if {$::bendix::String_for_cartoonify != ""} {
					set ::bendix::CG_Backbone_indexNs ""
					if {$::bendix::CG == 1} {
						if {$::bendix::subset != ""} {
							set CG_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames ) and not $::bendix::String_for_cartoonify"]
							#set CG_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames CA \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" ) and not $::bendix::String_for_cartoonify"]
						} else {
							set CG_backbone [atomselect $::bendix::proteinID "not $::bendix::String_for_cartoonify and (name $::bendix::fixed_particleNames )"]
							#set CG_backbone [atomselect $::bendix::proteinID "not $::bendix::String_for_cartoonify and (name $::bendix::fixed_particleNames CA \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" )"]
						}
					} else {
						if {$::bendix::subset != ""} {
							#set CG_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and backbone and not $::bendix::String_for_cartoonify and not sheet"]
							set CG_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames) and not $::bendix::String_for_cartoonify and not sheet"]
						} else {
							#set CG_backbone [atomselect $::bendix::proteinID "backbone and not $::bendix::String_for_cartoonify and not sheet"]
							set CG_backbone [atomselect $::bendix::proteinID "(name $::bendix::fixed_particleNames) and not $::bendix::String_for_cartoonify and not sheet"]
						}
					}
					set ::bendix::CG_Backbone_indexNs [$CG_backbone get index];
					set CG_Backbone_residNs [$CG_backbone get resid];
					$CG_backbone delete;
					set indexNs_lastIndex [expr {[llength $::bendix::CG_Backbone_indexNs] -1}]
					set lastIndexN [lindex $::bendix::CG_Backbone_indexNs $indexNs_lastIndex]
					set non_helix_Indices [lindex $::bendix::CG_Backbone_indexNs 0]
					
					set index_point_atomselect [atomselect $::bendix::proteinID "index $non_helix_Indices"]
					set index_point_xyz [lindex [$index_point_atomselect get {x y z}] 0]
					$index_point_atomselect delete;
					set backboneXYZs "{$index_point_xyz}"; # very first coordinate covered.
					set backbone_start_index_recorded ""
					#set backbone_start_AA_recorded ""
					set ::bendix::heres_a_helix ""
					graphics $::bendix::proteinID color gray
					
					### Loop through all backbone indices:
					for {set indexindex 0} {$indexindex < $indexNs_lastIndex} {incr indexindex} {
						set thisIndex [lindex $::bendix::CG_Backbone_indexNs $indexindex]
						set nextIndex [lindex $::bendix::CG_Backbone_indexNs [expr {$indexindex +1}]]
						
						## For future use to join beta-sheets better:
						if {$backbone_start_index_recorded == ""} {
							set backbone_start_index_recorded $thisIndex
							set backbone_start_AA_recorded [lindex $CG_Backbone_residNs $indexindex]
						}
										
						## Retrieve coordinate(s)
						if {$backboneXYZs == ""} {
							set index_point_atomselect [atomselect $::bendix::proteinID "index $thisIndex"]
							set index_point_xyz_thisIndex [lindex [$index_point_atomselect get {x y z}] 0]
							$index_point_atomselect delete;
							set backboneXYZs "{$index_point_xyz_thisIndex}"
						}
						
						if {$non_helix_Indices == ""} {
							set non_helix_Indices $thisIndex
						} 	
						
						set index_point_atomselect [atomselect $::bendix::proteinID "index $nextIndex"]
						set index_point_xyz_nextIndex [lindex [$index_point_atomselect get {x y z}] 0]
						$index_point_atomselect delete;
							
						set	distance_between_points 1
						#### Get distance between points IFF their residue numbers are consequitive (if they're not, they'll fail anyways):
						if {[expr [lindex $CG_Backbone_residNs $indexindex] +1] == [lindex $CG_Backbone_residNs [expr $indexindex +1]]} {
							set distance_between_points [vecdist [lindex $backboneXYZs [expr [llength $backboneXYZs]-1]] $index_point_xyz_nextIndex]
						}
							
						
						### If the next resid follows the former one, they're considered sequential and saved in the same list
						### Alternatively, or the residue-counter has reached the end of the residues, bendix draws.
					
						### If backbone residues are:
						### sequential, within 1nm AND it's not the very last index
						### --> save their index and coordinate:
						if {[expr [lindex $CG_Backbone_residNs $indexindex] +1] == [lindex $CG_Backbone_residNs [expr $indexindex +1]] && $distance_between_points < 10 && $indexindex != [expr {$indexNs_lastIndex -1}]} {
							lappend backboneXYZs $index_point_xyz_nextIndex
							lappend non_helix_Indices $nextIndex
							
						} else {
							
							### If it's the very last index, and it follows from the previous index, and the interdistance is <10, lappend its coord and non_helix_Indices:
							if {[expr [lindex $CG_Backbone_residNs $indexindex] +1] == [lindex $CG_Backbone_residNs [expr $indexindex +1]] && $distance_between_points < 10 && [expr $indexindex + 1] == $indexNs_lastIndex} {
								lappend backboneXYZs $index_point_xyz_nextIndex
								lappend non_helix_Indices $nextIndex
							}
							
							#### The next backbone residue does not follow the former (or it's the last resid)
							#### Compare with saved helix start and end indices to see whether saved helix end coordinates should be appended to the start or end of this backbone coordinate stretch:
							## Search through all stored helix end indices:
							
							set gotStart 0
							set gotEnd 0
							for {set helixIndex 0} {$helixIndex < [llength $::bendix::helix_indices_at_startNstop]} {incr helixIndex} {	
								if {$::bendix::CG == 1} {
									set Nindices_bn_AAs 5
								} else {
									set Nindices_bn_AAs 22; # was 12, 19, 20, ...
								}
								if {$gotEnd == 0 || $gotStart == 0} {
								### Stop the loop prematurely if end and start indices are found.
									### I'm looking for a helix index around the first backbone index:
									if {[lindex $::bendix::helix_indices_at_startNstop $helixIndex] <= [lindex $non_helix_Indices 0] && [lindex $::bendix::helix_indices_at_startNstop $helixIndex] >= [expr [lindex $non_helix_Indices 0] - $Nindices_bn_AAs]} {
										## helixindex must be smaller/eq to backboneIndex and larger/eq to backboneIndex-16.
										## Detected a backbone index that follows a helix: first backbone index
										## Insert coord/index at start of backbone list
										
										## helix coord:
										set helixEnd_coordinate [lindex $::bendix::helix_coord_at_startNstop $helixIndex]
														
										## Backbone coord:						
										set backboneEnd_atomSelection [set index_point_atomselect [atomselect $::bendix::proteinID "index [lindex $non_helix_Indices 0]"]]
										set backboneEnd_coordinate [lindex [$index_point_atomselect get {x y z}] 0]
										$backboneEnd_atomSelection delete;
										
										#### Get distance between points:
										set distance_between_points [vecdist $helixEnd_coordinate $backboneEnd_coordinate]

										if {$distance_between_points < 10} {
										##### Same stretch: Only the 1st segment-end is plugged with a sphere.
											lappend ::bendix::heres_a_helix [lindex $non_helix_Indices 0]
											set backboneXYZs [linsert $backboneXYZs 0 $helixEnd_coordinate];
											set gotStart 1
										}																	
									}
									
									### I'm looking for a helix index around the last backbone index:	
									if { [lindex $::bendix::helix_indices_at_startNstop $helixIndex] >= [lindex $non_helix_Indices [expr {[llength $non_helix_Indices]-1}]] && [lindex $::bendix::helix_indices_at_startNstop $helixIndex] <= [expr [lindex $non_helix_Indices [expr {[llength $non_helix_Indices]-1}]]+ $Nindices_bn_AAs]} {
										
										## [lindex $CG_Backbone_residNs [expr $indexindex +1]] > [lindex $CG_Backbone_residNs $indexindex]; # next AA is bigger than the previous AA (or it's a new chain)
										## helixindex must be larger/eq to the last backboneIndex and smaller/eq to the last backboneIndex+16.
										## Detected a backbone index that predates a helix: last backbone index
										## Insert coord/index at end of backbone list
										
										### helix coordinate:
										set helixStart_coordinate [lindex $::bendix::helix_coord_at_startNstop $helixIndex]
										
										### backbone coordinate:
										set backbone_coordinate [lindex $backboneXYZs [expr [llength $backboneXYZs] -1]]
										
										## Calculate the distance:
										set distance_between_points [vecdist $helixStart_coordinate $backbone_coordinate]
										
										if {$distance_between_points < 10} {
											lappend ::bendix::heres_a_helix [lindex $non_helix_Indices [expr {[llength $non_helix_Indices] -1}]]
											lappend backboneXYZs $helixStart_coordinate;
											set gotEnd 1
										}
									}								
								}
							}
							
							#### If no starting-coordinate was ended for a helix-backbone junction, add a coordinate to join a possible AT sheet:
							# Add distance- and sheet check:
							if {$gotStart == 0 && $::bendix::CG == 0} {
								
								set firstBackboneAA $backbone_start_AA_recorded
								set done 0
								set index_point_atomselect [atomselect $::bendix::proteinID "resid [expr $backbone_start_AA_recorded -1] and name CA and sheet"]
								set possCA_indices [$index_point_atomselect get index]
								if {$possCA_indices != ""} {
									for {set seekIndex 0} {$seekIndex <= [expr [llength $possCA_indices] -1]} {incr seekIndex} {
										### Checking if [lindex $possCA_indices $seekIndex] is nearby $backbone_start_index_recorded:
										if {[lindex $possCA_indices $seekIndex] <= $backbone_start_index_recorded &&  [lindex $possCA_indices $seekIndex] >= [expr $backbone_start_index_recorded -30] && $done == 0} {
										### the index sought needs to be no higher than the first backbone index
										### and no smaller than the first backbone index -30
										### Found that [lindex $possCA_indices $seekIndex] is a good candidate joint for $backbone_start_index_recorded"
											set beta_point_atomselect [atomselect $::bendix::proteinID "index [lindex $possCA_indices $seekIndex]"]
											set beta_xyz [lindex [$beta_point_atomselect get {x y z}] 0]
											$beta_point_atomselect delete;
											$index_point_atomselect delete
											
											### Check separation: only <1nm separations are joined.
											set distance_between_sheetHelix_points [vecdist $beta_xyz [lindex $backboneXYZs 0]]
											if {$distance_between_sheetHelix_points < 10} {
												set backboneXYZs [linsert $backboneXYZs 0 $beta_xyz];	
											}
											set done 1					
										}
									}
								}
							}
							
							### Use the collected coordinates and indices. They're probably connected unless they are separated by >1nm (covered below)
							set lastXYZ_index [expr {[llength $backboneXYZs] -1}]; # I'm drawing between n and n+1
							if {$lastXYZ_index > 0} {
									
								lappend ::bendix::CG_Backbone_normalised_startsandstops [expr {$indexindex - $lastXYZ_index}]; # To enable the speed-up loop.
								### Always make a sphere at the start. It's not covered by the spline graphics.
								graphics $::bendix::proteinID sphere [lindex $backboneXYZs 0] radius $::bendix::backbone_radius resolution 20
								#graphics $::bendix::proteinID cylinder [lindex $backboneXYZs 0] [lindex $backboneXYZs 1] radius $::bendix::backbone_radius resolution 20;

								#set index_pointB [lindex $non_helix_Indices $draw]
								#set index_pointB_atomselect [atomselect $::bendix::proteinID "index $index_pointB"]
								#set index_pointB_xyz [lindex [$index_pointB_atomselect get {x y z}] 0]
								#$index_pointB_atomselect delete

								#### Get distance between points:
								#set distance_between_points [vecdist $index_pointA_xyz $index_pointB_xyz]

								##### Same stretch: Only the 1st segment-end is plugged with a sphere.
								
								#### HERMITE SPLINE
								# Create __Phantom knots___ to allow the last extensions above to feature in the spline.
								# This section is only applicable to points undergoing splines. 
								# Resolution 1 means no between-knots spline-points generated, 
								# only copies of the previous knot, so spline should be disallowed.
								# First create a synthetic point at the very end so that the spline works,
								# using synth end-point just made.
								# The $lastXYZ_index is ok to use since we're only using the most peripheral 2 points, 
								# the extension with exactly 1 residue length between points.
							
								### Phantom knots downwards and upwards:
								set backboneXYZs_lastIndex [expr {[llength $backboneXYZs]-1}]
								set last_coord [lindex $backboneXYZs $backboneXYZs_lastIndex]
								set nexttolast_coord [lindex $backboneXYZs [expr {$backboneXYZs_lastIndex -1}]]
								set last_vector [vecsub $last_coord $nexttolast_coord]
								set last_final_point [vecadd $last_vector $last_coord]
								set list_copy $backboneXYZs
								set backboneXYZs [lappend list_copy $last_final_point]
		
								set first_coord [lindex $backboneXYZs 0]
								set second_coord [lindex $backboneXYZs 1]
								set first_vector [vecsub $first_coord $second_coord]
								set first_final_point [vecadd $first_vector $first_coord]
								set list_copy $backboneXYZs
								set backboneXYZs [linsert $list_copy 0 $first_final_point]
		
								#### Hermite Spline treatment of Select_list with Catmull-Rom gradient using the constant 0.5
								set backboneXYZs_lastIndex [expr {[llength $backboneXYZs] -1}]; # Whereof 2 'fake' points on either side: 1 ghost and one extension. So minimum 4 coords to use.
								set CR_constant 0.5
								set splined_backbone_coords {}
								set spline_index 0
								
								# The last backboneXYZs index is $backboneXYZs_lastIndex. Going through them from 1 (necessarily) to, and incl, [expr $backboneXYZs_lastIndex -2]."
								for {set k 1} {$k <= [expr $backboneXYZs_lastIndex -2]} {incr k 1} {
									# point +1 - (-1)
									set m1_x [expr {[expr {[lindex [lindex $backboneXYZs [expr {$k+1}]] 0] - [lindex [lindex $backboneXYZs [expr {$k-1}]] 0]}]*$CR_constant}]
									set m1_y [expr {[expr {[lindex [lindex $backboneXYZs [expr {$k+1}]] 1] - [lindex [lindex $backboneXYZs [expr {$k-1}]] 1]}]*$CR_constant}]
									set m1_z [expr {[expr {[lindex [lindex $backboneXYZs [expr {$k+1}]] 2] - [lindex [lindex $backboneXYZs [expr {$k-1}]] 2]}]*$CR_constant}]
									
									# +2 - (0)
									set m2_x [expr {[expr {[lindex [lindex $backboneXYZs [expr {$k+2}]] 0] - [lindex [lindex $backboneXYZs $k] 0]}]*$CR_constant}]
									set m2_y [expr {[expr {[lindex [lindex $backboneXYZs [expr {$k+2}]] 1] - [lindex [lindex $backboneXYZs $k] 1]}]*$CR_constant}]
									set m2_z [expr {[expr {[lindex [lindex $backboneXYZs [expr {$k+2}]] 2] - [lindex [lindex $backboneXYZs $k] 2]}]*$CR_constant}]
		
								
									for {set l 0} {$l <= $::bendix::spline_resolution} {incr l 1} {
										# s from 0 to 1, in constant increase increments, the size of which depends on resolution. 
										# NB The 0th point is catered for above, at k==0.
										# E.g. Resolution 10: 1/10, 1/10, 2/10, 3/10, 4/10.. 10/10 = 0, 0.1, 0.2, 0.3 ... 1 = s (varying t from 0 to 1)
										
										set Resolution_1dp [expr $::bendix::spline_resolution*1.0]
										set s [expr {$l/$Resolution_1dp}]
										
										# The Hermite Spline Blending functions: 
										set h1 [expr {2.0*$s*$s*$s - 3.0*$s*$s +1.0}]
										set h2 [expr {-2.0*$s*$s*$s + 3.0*$s*$s}]
										set h3 [expr {$s*$s*$s - 2.0*$s*$s +$s}]
										set h4 [expr {$s*$s*$s - $s*$s}]
		
									 	set px [expr {$h1*[lindex [lindex $backboneXYZs $k] 0] + $h2*[lindex [lindex $backboneXYZs [expr {$k+1}]] 0] + $h3*$m1_x + $h4*$m2_x}] 
										set py [expr {$h1*[lindex [lindex $backboneXYZs $k] 1] + $h2*[lindex [lindex $backboneXYZs [expr {$k+1}]] 1] + $h3*$m1_y + $h4*$m2_y}] 
										set pz [expr {$h1*[lindex [lindex $backboneXYZs $k] 2] + $h2*[lindex [lindex $backboneXYZs [expr {$k+1}]] 2] + $h3*$m1_z + $h4*$m2_z}]
							
										set spline_point "$px $py $pz"
										lappend splined_backbone_coords $spline_point
									}
								}
								for {set splineDraw 0} {$splineDraw < [expr [llength $splined_backbone_coords]-1]} {incr splineDraw} {
									graphics $::bendix::proteinID cylinder [lindex $splined_backbone_coords $splineDraw] [lindex $splined_backbone_coords [expr $splineDraw + 1]] radius $::bendix::backbone_radius resolution 30
									graphics $::bendix::proteinID sphere [lindex $splined_backbone_coords [expr $splineDraw + 1]] radius $::bendix::backbone_radius resolution 20
								}
								
								if {$nextIndex != $lastIndexN} {
								#### If first of next backbone stretch isn't the last index in the protein:  (for the speed-up module, I think)
									lappend ::bendix::CG_Backbone_normalised_startsandstops $indexindex
								} else {
									lappend ::bendix::CG_Backbone_normalised_startsandstops [expr {$indexindex + 1}]
								}

							} else {
								lappend ::bendix::CG_Backbone_normalised_startsandstops $indexindex
								lappend ::bendix::CG_Backbone_normalised_startsandstops $indexindex
							} 
							#set non_helix_Indices $nextIndex
							set non_helix_Indices ""
							set backbone_helix 0
							set helix_backbone 0
							set backboneXYZs {}; # erase between each new draw.
							set backbone_start_index_recorded ""
						}
					}
				} else {
					### $::bendix::String_for_cartoonify == ""
					### No helices have been written out, so drawing backbone for Everything.
					
					### If a custom particle type is added, it's helicity, sheeticity or backbonicity won't be pre-known. Use it as a backbone particle.
					### In case Bendix has never been Drawn, set necessary ::bendix::variables
					
					if {$::bendix::CG == 1} {
						if {$::bendix::subset != ""} {
							set CG_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames)"]
						} else {
							set CG_backbone [atomselect $::bendix::proteinID "name $::bendix::fixed_particleNames"]
						}
					} else {
						if {$::bendix::subset != ""} {
							set CG_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames)"]; # removed 'backbone' --problematic if non-orthodox protein backbone.
						} else {
							set CG_backbone [atomselect $::bendix::proteinID "(name $::bendix::fixed_particleNames )"]; # removed 'backbone' --problematic if non-orthodox protein backbone.
						}
					}
					set ::bendix::CG_Backbone_indexNs [$CG_backbone get index]
					set CG_Backbone_residNs [$CG_backbone get resid]
					$CG_backbone delete
					set backbone_xyz {}
					set last_residueIndexN [expr [llength $::bendix::CG_Backbone_indexNs] -1]
					
					for {set residueCounter 0} {$residueCounter < $last_residueIndexN} {incr residueCounter} {
					
						set currentAA [lindex $::bendix::CG_Backbone_indexNs $residueCounter]
						set backboneParticle_atomselect [atomselect $::bendix::proteinID "index $currentAA"]
						append backbone_xyz " " [$backboneParticle_atomselect get {x y z}]
						$backboneParticle_atomselect delete
						
						### Only do this CPU-intensive step if necessary:
						set distance_between_points 1
						if {[lindex $CG_Backbone_residNs $residueCounter] < [lindex $CG_Backbone_residNs [expr $residueCounter + 1]]} {
							### Next coord:
							set nextAA [lindex $::bendix::CG_Backbone_indexNs [expr $residueCounter + 1]]
							set backboneParticle2_atomselect [atomselect $::bendix::proteinID "index $nextAA"]
							set temp_xyz [lindex [$backboneParticle2_atomselect get {x y z}] 0]
							$backboneParticle2_atomselect delete
							
							#### Get distance between points:
							set distance_between_points [vecdist [lindex $backbone_xyz [expr [llength $backbone_xyz]-1]] $temp_xyz]
						}
					
						### If the next residue is smaller than the current one OR it's the very last residue in the protein OR there's over 1nm between them, stop and draw:
						if {[lindex $CG_Backbone_residNs $residueCounter] > [lindex $CG_Backbone_residNs [expr $residueCounter + 1]] || $residueCounter == [expr $last_residueIndexN -1] || $distance_between_points > 10 } {
							
							if {$residueCounter == [expr $last_residueIndexN -1]} {
								### Save the very last coord.
								set lastAA [lindex $::bendix::CG_Backbone_indexNs $last_residueIndexN]
								set backboneParticle_atomselect [atomselect $::bendix::proteinID "index $lastAA"]
								append backbone_xyz " " [$backboneParticle_atomselect get {x y z}]
								$backboneParticle_atomselect delete
							}
							graphics $::bendix::proteinID color gray
							set last_indexN_of_storedCoords [expr [llength $backbone_xyz] -1]
							if {$last_indexN_of_storedCoords > 0} {
							### Need at least two points to spline using phantom knots.
																								
								#lappend ::bendix::CG_Backbone_normalised_startsandstops [expr {$indexindex - $lastXYZ_index}]; # Made to enable speed-up.
								### Always make a sphere at the start. It's not covered by the spline graphics.
								graphics $::bendix::proteinID sphere [lindex $backbone_xyz 0] radius $::bendix::backbone_radius resolution 20
								graphics $::bendix::proteinID cylinder [lindex $backbone_xyz 0] [lindex $backbone_xyz 1] radius $::bendix::backbone_radius resolution 30
								#### HERMITE SPLINE
								# Create __Phantom knots___ to allow the last extensions above to feature in the spline.
								# This section is only applicable to points undergoing splines. 
								# Resolution 1 means no between-knots spline-points generated, 
								# only copies of the previous knot, so spline should be disallowed.
								# First create a synthetic point at the very end so that the spline works,
								# using synth end-point just made.
								# The $lastXYZ_index is ok to use since we're only using the most peripheral 2 points, 
								# the extension with exactly 1 residue length between points.
								
								### Phantom knots downwards and upwards:
								set backbone_xyz_lastIndex [expr {[llength $backbone_xyz]-1}]
								set last_coord [lindex $backbone_xyz $backbone_xyz_lastIndex]
								set nexttolast_coord [lindex $backbone_xyz [expr {$backbone_xyz_lastIndex -1}]]
								set last_vector [vecsub $last_coord $nexttolast_coord]
								set last_final_point [vecadd $last_vector $last_coord]
								set list_copy $backbone_xyz
								set backbone_xyz [lappend list_copy $last_final_point]
		
								set first_coord [lindex $backbone_xyz 0]
								set second_coord [lindex $backbone_xyz 1]
								set first_vector [vecsub $first_coord $second_coord]
								set first_final_point [vecadd $first_vector $first_coord]
								set list_copy $backbone_xyz
								set backbone_xyz [linsert $list_copy 0 $first_final_point]
		
								#### Hermite Spline treatment of Select_list with Catmull-Rom gradient using the constant 0.5
								set backbone_xyz_lastIndex [expr {[llength $backbone_xyz] -1}]; # Whereof 2 'fake' points on either side: 1 ghost and one extension. So minimum 4 coords to use.
								set CR_constant 0.5
								set splined_backbone_coords {}
								set spline_index 0
								
								# The last backbone_xyz index is $backbone_xyz_lastIndex. Going through them from 1 (necessarily) to, and incl, [expr $backbone_xyz_lastIndex -2]."
								for {set k 1} {$k <= [expr $backbone_xyz_lastIndex -2]} {incr k 1} {
									# point +1 - (-1)
									set m1_x [expr {[expr {[lindex [lindex $backbone_xyz [expr {$k+1}]] 0] - [lindex [lindex $backbone_xyz [expr {$k-1}]] 0]}]*$CR_constant}]
									set m1_y [expr {[expr {[lindex [lindex $backbone_xyz [expr {$k+1}]] 1] - [lindex [lindex $backbone_xyz [expr {$k-1}]] 1]}]*$CR_constant}]
									set m1_z [expr {[expr {[lindex [lindex $backbone_xyz [expr {$k+1}]] 2] - [lindex [lindex $backbone_xyz [expr {$k-1}]] 2]}]*$CR_constant}]
									
									# +2 - (0)
									set m2_x [expr {[expr {[lindex [lindex $backbone_xyz [expr {$k+2}]] 0] - [lindex [lindex $backbone_xyz $k] 0]}]*$CR_constant}]
									set m2_y [expr {[expr {[lindex [lindex $backbone_xyz [expr {$k+2}]] 1] - [lindex [lindex $backbone_xyz $k] 1]}]*$CR_constant}]
									set m2_z [expr {[expr {[lindex [lindex $backbone_xyz [expr {$k+2}]] 2] - [lindex [lindex $backbone_xyz $k] 2]}]*$CR_constant}]
		
								
									for {set l 1} {$l <= 4} {incr l 1} {
										# s from 0 to 1, in constant increase increments, the size of which depends on resolution. 
										# NB The 0th point is catered for above, at k==0.
										# E.g. Resolution 10: 1/10, 1/10, 2/10, 3/10, 4/10.. 10/10 = 0, 0.1, 0.2, 0.3 ... 1 = s (varying t from 0 to 1)
										
										set s [expr {$l/4.0}]
										
										# The Hermite Spline Blending functions: 
										set h1 [expr {2.0*$s*$s*$s - 3.0*$s*$s +1.0}]
										set h2 [expr {-2.0*$s*$s*$s + 3.0*$s*$s}]
										set h3 [expr {$s*$s*$s - 2.0*$s*$s +$s}]
										set h4 [expr {$s*$s*$s - $s*$s}]
		
									 	set px [expr {$h1*[lindex [lindex $backbone_xyz $k] 0] + $h2*[lindex [lindex $backbone_xyz [expr {$k+1}]] 0] + $h3*$m1_x + $h4*$m2_x}] 
										set py [expr {$h1*[lindex [lindex $backbone_xyz $k] 1] + $h2*[lindex [lindex $backbone_xyz [expr {$k+1}]] 1] + $h3*$m1_y + $h4*$m2_y}] 
										set pz [expr {$h1*[lindex [lindex $backbone_xyz $k] 2] + $h2*[lindex [lindex $backbone_xyz [expr {$k+1}]] 2] + $h3*$m1_z + $h4*$m2_z}]
							
										set spline_point "$px $py $pz"
										lappend splined_backbone_coords $spline_point

									}
								}
								for {set splineDraw 0} {$splineDraw < [expr [llength $splined_backbone_coords]-1]} {incr splineDraw} {
									graphics $::bendix::proteinID cylinder [lindex $splined_backbone_coords $splineDraw] [lindex $splined_backbone_coords [expr $splineDraw + 1]] radius $::bendix::backbone_radius resolution 30
									graphics $::bendix::proteinID sphere [lindex $splined_backbone_coords [expr $splineDraw + 1]] radius $::bendix::backbone_radius resolution 20
								}
	
#								if {$nextIndex != $lastIndexN} {
								#### If first of next backbone stretch isn't the last index in the protein:  (for the speed-up module)
#									lappend ::bendix::CG_Backbone_normalised_startsandstops $indexindex
#								} else {
#									lappend ::bendix::CG_Backbone_normalised_startsandstops [expr {$indexindex + 1}]
#								}
							}
							set backbone_xyz {}
						}
					}

					#lappend ::bendix::CG_Backbone_normalised_startsandstops $M
					unset backbone_xyz
				}
				set ::bendix::helix_assignment_by_backbone $::bendix::helix_assignment
				set upcount 0
			} else {
				#### Module to speed up re-drawing, where variables are present. Spline not implemented.
				graphics $::bendix::proteinID color gray
				set upcount 0
				for {set loop 0} {$loop < [llength $::bendix::CG_Backbone_normalised_startsandstops]} {incr loop 2} {
					for {set saved_index_counter [lindex $::bendix::CG_Backbone_normalised_startsandstops $loop]} {$saved_index_counter <= [lindex $::bendix::CG_Backbone_normalised_startsandstops [expr {$loop + 1}]]} {incr saved_index_counter} {
						if {[lindex $::bendix::CG_Backbone_indexNs $saved_index_counter] == [lindex $::bendix::heres_a_helix $upcount]} {
						#### Junction between helix and backbone detected.
						#### Find out if the junction is helix-backbone or backbone-helix:
							if {[lindex $::bendix::CG_Backbone_indexNs $saved_index_counter] < [lindex $::bendix::helix_indices_at_startNstop $upcount]} {
							#### smaller backbone than helix = backbone-helix junction.
								set backboneParticleA_atomselect [atomselect $::bendix::proteinID "index [lindex $::bendix::CG_Backbone_indexNs $saved_index_counter]"]
								set backboneParticleA_xyz_flaw [$backboneParticleA_atomselect get {x y z}]
								set h_backboneParticleA_xyz [lindex $backboneParticleA_xyz_flaw 0]
								$backboneParticleA_atomselect delete
								graphics $::bendix::proteinID sphere $h_backboneParticleA_xyz radius $::bendix::backbone_radius resolution 20
								set h_backboneParticleB_xyz [lindex $::bendix::helix_coord_at_startNstop $upcount]
								graphics $::bendix::proteinID cylinder $h_backboneParticleA_xyz $h_backboneParticleB_xyz radius $::bendix::backbone_radius resolution 30
								graphics $::bendix::proteinID sphere $h_backboneParticleB_xyz radius $::bendix::backbone_radius resolution 20
							} else {
							#### smaller helix index than backbone = helix-backbone junction.
								set h_backboneParticleA_xyz [lindex $::bendix::helix_coord_at_startNstop $upcount]
								graphics $::bendix::proteinID sphere $h_backboneParticleA_xyz radius $::bendix::backbone_radius resolution 20
								set backboneParticleB_atomselect [atomselect $::bendix::proteinID "index [lindex $::bendix::CG_Backbone_indexNs $saved_index_counter]"]
								set backboneParticleB_xyz_flaw [$backboneParticleB_atomselect get {x y z}]
								set h_backboneParticleB_xyz [lindex $backboneParticleB_xyz_flaw 0]
								$backboneParticleB_atomselect delete
								graphics $::bendix::proteinID cylinder $h_backboneParticleA_xyz $h_backboneParticleB_xyz radius $::bendix::backbone_radius resolution 30
								graphics $::bendix::proteinID sphere $h_backboneParticleB_xyz radius $::bendix::backbone_radius resolution 20
							}
							incr upcount
						}
						#### Draw cylinders between backbone particles:
						if {[expr {[lindex $::bendix::CG_Backbone_normalised_startsandstops [expr {$loop + 1} ]] - [lindex $::bendix::CG_Backbone_normalised_startsandstops $loop ]}] > 0 } {
						#### If we're not dealing with just a single backbone-index, surrounded by something else:
							if {$saved_index_counter == [lindex $::bendix::CG_Backbone_normalised_startsandstops $loop]} {
							#### If it's right at the start:
								set backboneParticleA_atomselect [atomselect $::bendix::proteinID "index [lindex $::bendix::CG_Backbone_indexNs $saved_index_counter]"]
								set backboneParticleA_xyz_flaw [$backboneParticleA_atomselect get {x y z}]
								set backboneParticleA_xyz [lindex $backboneParticleA_xyz_flaw 0]
								$backboneParticleA_atomselect delete
								graphics $::bendix::proteinID sphere $backboneParticleA_xyz radius $::bendix::backbone_radius resolution 20				
							} else {
								set backboneParticleB_atomselect [atomselect $::bendix::proteinID "index [lindex $::bendix::CG_Backbone_indexNs $saved_index_counter]"]
								set backboneParticleB_xyz_flaw [$backboneParticleB_atomselect get {x y z}]
								set backboneParticleB_xyz [lindex $backboneParticleB_xyz_flaw 0]
								$backboneParticleB_atomselect delete
								graphics $::bendix::proteinID cylinder $backboneParticleA_xyz $backboneParticleB_xyz radius $::bendix::backbone_radius resolution 30
								graphics $::bendix::proteinID sphere $backboneParticleB_xyz radius $::bendix::backbone_radius resolution 20
								set backboneParticleA_xyz $backboneParticleB_xyz
							}
						}
					}
					set backboneParticleA_xyz ""
					set backboneParticleB_xyz ""
					set h_backboneParticleA_xyz ""
					set h_backboneParticleB_xyz ""
				}
				set ::bendix::helix_assignment_by_backbone $::bendix::helix_assignment
				set ::bendix::subset_by_backbone $::bendix::subset
				set ::bendix::proteinID_by_backbone $::bendix::proteinID; #_____fixed!
			}

			#### Draw Beta-sheets for Slow and Accurate
			if {$::bendix::CG == 1} {
				if {$::bendix::String_for_cartoonify != ""} {
					if {$::bendix::subset != ""} {
						set CG_sheet [atomselect $::bendix::proteinID "($::bendix::subset) and (name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" ) and not $::bendix::String_for_cartoonify"]
					} else {
						set CG_sheet [atomselect $::bendix::proteinID "(name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" ) and not $::bendix::String_for_cartoonify"]
					}
				} else {
					if {$::bendix::subset != ""} {
						set CG_sheet [atomselect $::bendix::proteinID "($::bendix::subset) and (name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" )"]
					} else {
						set CG_sheet [atomselect $::bendix::proteinID "(name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" )"]
					}
				}

				set individualAAs [$CG_sheet get resid]
				$CG_sheet delete;
				
				### Only draw sheets if atomselect generated resids, i.e. if the system is MARTINI CG.
				if {$individualAAs != ""} {
					graphics $::bendix::proteinID color cyan2
					set backbone_xyz {}
					set oldAA [lindex $individualAAs 0]
					set oldoldAA [lindex $individualAAs 0]
					set backboneParticle_atomselect [atomselect $::bendix::proteinID "resid $oldAA and (name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" )"]
					set oldAA_xyz [lindex [$backboneParticle_atomselect get {x y z}] 0]
					set oldoldAA_xyz [lindex [$backboneParticle_atomselect get {x y z}] 0]	
					$backboneParticle_atomselect delete;
					for {set residueCounter 1} {$residueCounter < [llength $individualAAs]} {incr residueCounter} {
						set newAA [lindex $individualAAs $residueCounter]
						set backboneParticle_atomselect [atomselect $::bendix::proteinID "resid $newAA and (name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" )"]
						set newAA_xyz [lindex [$backboneParticle_atomselect get {x y z}] 0]
						$backboneParticle_atomselect delete
						# If subsequent, note coordinate and draw.
						# If non-subsequent, draw cone on last and start new sheet.
						if {$oldoldAA != $oldAA} {
							if {($newAA == [expr {$oldAA + 1}]) && ($oldAA == [expr {$oldoldAA + 1}])} {
								graphics $::bendix::proteinID sphere $oldoldAA_xyz radius [expr {$::bendix::backbone_radius + 0.5}] resolution 20
								graphics $::bendix::proteinID cylinder $oldoldAA_xyz $oldAA_xyz radius [expr $::bendix::backbone_radius + 0.5] resolution 30
							} elseif {($newAA != [expr {$oldAA + 1}]) && ($oldAA == [expr {$oldoldAA + 1}])} {
								graphics $::bendix::proteinID sphere $oldoldAA_xyz radius [expr {$::bendix::backbone_radius + 0.5}] resolution 20
								graphics $::bendix::proteinID cone $oldoldAA_xyz $oldAA_xyz radius [expr {$::bendix::backbone_radius + 1}] resolution 20
							}
						}
						set oldoldAA $oldAA
						set oldAA $newAA
						set oldoldAA_xyz $oldAA_xyz
						set oldAA_xyz $newAA_xyz
					}
					graphics $::bendix::proteinID sphere $oldoldAA_xyz radius [expr {$::bendix::backbone_radius + 0.5}] resolution 20
					graphics $::bendix::proteinID cone $oldoldAA_xyz $oldAA_xyz radius [expr {$::bendix::backbone_radius + 1}] resolution 20
				}
				unset CG_sheet individualAAs
				
			} else {
				### Seek out sheet-helix joints, that are provided with an extra linker-line to improve graphics
				### Ideal would be to use the same spline as VMD; now the linkage still leaves a small gap.
				set sheetAtomselect [atomselect $::bendix::proteinID "sheet and name CA"]
				set sheetIndices [$sheetAtomselect get index]
				$sheetAtomselect delete
				if {$sheetIndices != ""} {
					set last_sheetIndex [expr [llength $sheetIndices] -1]
					set firstSheetIndex [lindex $sheetIndices 0]
					set sheet_lastIndex_list ""
					for {set sheetIndex 1} {$sheetIndex <= $last_sheetIndex} {incr sheetIndex} {
						if {[expr $firstSheetIndex +25] < [lindex $sheetIndices $sheetIndex]} {
							lappend sheet_lastIndex_list $firstSheetIndex
						}
						set firstSheetIndex [lindex $sheetIndices $sheetIndex]
					}
					lappend sheet_lastIndex_list [lindex $sheetIndices $last_sheetIndex]
					
					foreach {helixStart helixStop} $::bendix::helix_indices_at_startNstop {helixStartCoord helixEndCoord} $::bendix::helix_coord_at_startNstop {
						## Nevermind the helixStop; we're interested in helixStarts only - that's where Join does a poor job.
						foreach sheet_lastIndex $sheet_lastIndex_list {
							if {$sheet_lastIndex < $helixStart && $sheet_lastIndex > [expr $helixStart - 15]} {
								
								## Retrieve sheet coord:
								set sheet_point_atomselect [atomselect $::bendix::proteinID "sheet and index $sheet_lastIndex"]
								set sheet_xyz [lindex [$sheet_point_atomselect get {x y z}] 0]
								$sheet_point_atomselect delete;
								
								## Check distance between potential match
								set distance_between_sheetHelix_points [vecdist $sheet_xyz $helixStartCoord]
								
								if {$distance_between_sheetHelix_points < 10} {
									### Draw a joint.
									graphics $::bendix::proteinID cylinder $sheet_xyz $helixStartCoord radius $::bendix::backbone_radius resolution 30
								}
							}
						}
					}
				}								
				
				### Atomistic proteins benefit from VMD sheet rendering
				::bendix::cartoonify
			}; #### Drawing Beta-sheets end.
			
			set AllThingsDrawn [graphics $::bendix::proteinID list]
			lset ::bendix::Last_backbone_drawn_list $::bendix::proteinID [lindex $AllThingsDrawn [expr {[llength $AllThingsDrawn] -1}]]
			
		} elseif {$::bendix::quick_and_dirty == 1 && $::bendix::slow_and_pretty == 0 && [lindex $::bendix::CG_cartoonified_list $::bendix::proteinID] == 0 && $::bendix::CG == 1} {

		############################## DYNAMIC BONDS AS BACKBONE: QUICK BUT IMPERFECT LINKAGE ('Draft') ##############################
			.bex.lf3.field1.slow configure -state disabled
			lset ::bendix::CG_cartoonified_list $::bendix::proteinID 1
			
			#### Backbone
			if {$::bendix::String_for_cartoonify != ""} {
				if {$::bendix::subset != ""} {
					mol selection "(($::bendix::subset) and (name $::bendix::fixed_particleNames) and not $::bendix::String_for_cartoonify)"
					#mol selection "(($::bendix::subset) and (name $::bendix::fixed_particleNames \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" ) and not $::bendix::String_for_cartoonify)"
					
				} else {
					mol selection "(name $::bendix::fixed_particleNames) and not $::bendix::String_for_cartoonify"
					#mol selection "(name $::bendix::fixed_particleNames \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" ) and not $::bendix::String_for_cartoonify"
				}
			} else {
				if {$::bendix::subset != ""} {
					mol selection "($::bendix::subset) and (name $::bendix::fixed_particleNames)"
					#mol selection "($::bendix::subset) and (name $::bendix::fixed_particleNames \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" )"
				} else {
					mol selection "(name $::bendix::fixed_particleNames)"
					#mol selection "(name $::bendix::fixed_particleNames \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" )"
				}
			}
			
			
			mol addrep $::bendix::proteinID
			#### I add exactly one representation, so I know that the last repID is created by bendix.
			set N_reps_so_far [molinfo $::bendix::proteinID get numreps]
			set N_reps_so_far [expr {$N_reps_so_far -1}]; # the index.
			
			#### Roundabout way to actually get the ID of the just-added rep, so we can delete it later:
			set repNumber [mol repname $::bendix::proteinID $N_reps_so_far];
			lset ::bendix::CG_old_rep_list $::bendix::proteinID [mol repindex $::bendix::proteinID $repNumber]
			mol modstyle $N_reps_so_far $::bendix::proteinID DynamicBonds 4.6 0.2
			if {$::bendix::autoColour == 1 && $::bendix::uniform_colour_type==1} {
				mol modcolor $N_reps_so_far $::bendix::proteinID Chain
			} else {
				mol modcolor $N_reps_so_far $::bendix::proteinID ColorID 2
			}

			#### Beta-sheets
			if {$::bendix::String_for_cartoonify != ""} {
				if {$::bendix::subset != ""} {
					mol selection "(($::bendix::subset) and ((name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" ) and not $::bendix::String_for_cartoonify))"
				} else {
					mol selection "((name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" ) and not $::bendix::String_for_cartoonify)"
				}
			} else {
				if {$::bendix::subset != ""} {
					mol selection "(($::bendix::subset) and (name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" ))"
				} else {
					mol selection "(name \"$::bendix::MartiniSheet\" or name \"$::bendix::MartiniSheetNew\" )"
				}
			}

			mol addrep $::bendix::proteinID
			#### I add exactly one representation, so I know that the last repid is created by Bendix.
			set N_reps_so_far [molinfo $::bendix::proteinID get numreps]
			set N_reps_so_far [expr {$N_reps_so_far -1}]; # the index.
			
			#### Roundabout way to actually get the ID of the just-added rep, so we can delete it later:
			set repNumber [mol repname $::bendix::proteinID $N_reps_so_far];
			lset ::bendix::CG_old_sheet_rep_list $::bendix::proteinID [mol repindex $::bendix::proteinID $repNumber]
			mol modstyle $N_reps_so_far $::bendix::proteinID DynamicBonds 4.6 0.7 
			if {$::bendix::autoColour == 1 && $::bendix::uniform_colour_type==1} {
				mol modcolor $N_reps_so_far $::bendix::proteinID Chain
			} else {
				mol modcolor $N_reps_so_far $::bendix::proteinID ColorID 2;
			}
			
		} elseif {$::bendix::quick_and_dirty == 1 && $::bendix::slow_and_pretty == 0 && $::bendix::CG == 0} {
			::bendix::cartoonify
		}; # End of DYNAMIC BONDS (GUI 'Draft') loop
		
	} elseif {$::bendix::quick_and_dirty == 0 && $::bendix::slow_and_pretty == 0 && [lindex $::bendix::CG_cartoonified_list $::bendix::proteinID] == 1} {
		
	#### BACKBONE ERASER LOOP:
		if { [lindex $::bendix::First_backbone_drawn_list $::bendix::proteinID] != "" && [lindex $::bendix::Last_backbone_drawn_list $::bendix::proteinID] != ""} {
			for {set kill [lindex $::bendix::First_backbone_drawn_list $::bendix::proteinID]} {$kill <= [lindex $::bendix::Last_backbone_drawn_list $::bendix::proteinID]} {incr kill} {
				graphics $::bendix::proteinID delete $kill 
			}
			lset ::bendix::First_backbone_drawn_list $::bendix::proteinID ""
			lset ::bendix::Last_backbone_drawn_list $::bendix::proteinID ""
		}
		if { [lindex $::bendix::CG_old_sheet_rep_list $::bendix::proteinID] != "" } {
			# Erasing CG draft sheets
			mol delrep [lindex $::bendix::CG_old_sheet_rep_list $::bendix::proteinID] $::bendix::proteinID
			lset ::bendix::CG_old_sheet_rep_list $::bendix::proteinID ""
		}
		if { [lindex $::bendix::CG_old_rep_list $::bendix::proteinID] != "" } {
			# Erasing draft CG backbone
			mol delrep [lindex $::bendix::CG_old_rep_list $::bendix::proteinID] $::bendix::proteinID
			lset ::bendix::CG_old_rep_list $::bendix::proteinID ""
		}
		# Erasing CG DrawBackbone.
		.bex.lf3.field1.quick configure -state normal
		.bex.lf3.field1.slow configure -state normal
		lset ::bendix::CG_cartoonified_list $::bendix::proteinID 0
			
		### Erase Rep-sheets, where available:	
		if {$::bendix::CG == 0} {
			::bendix::cartoonify
		}
		
	} elseif {$::bendix::die_backbone == 1 && $::bendix::CG == 0} {
		::bendix::cartoonify
	}; # Die == 0 loop ends.
  }

# ::bendix::cartoonify ---------------------------------------------------------
#    If AT backbone is not already drawn to screen,
#    or helicity or subset has changed since last Draw,
#    depicts subset AT protein that has not been assigned helicity
#    in the associated $::bendix::helix_assignment field, using Representations.
#    Where available, the Representation is coloured according to helices'
#    colouring. The Representation added is stored for easy removal.
# ------------------------------------------------------------------------------

  proc ::bendix::cartoonify {} {

	#### Check that a molecule is loaded. If not untick boxes.
	set N_loaded_Mols [molinfo num]
	if {$N_loaded_Mols == 0} {
		if {$::bendix::autoCartoon == 1} {
			set ::bendix::autoCartoon 0
		} elseif {$::bendix::autoNewCartoon == 1} {
			set ::bendix::autoNewCartoon 0
		} elseif {$::bendix::autoTube == 1} {
			set ::bendix::autoTube 0
		} elseif {$::bendix::quick_and_dirty == 1} {
			set ::bendix::quick_and_dirty 0
		} elseif {$::bendix::slow_and_pretty == 1} {
			set ::bendix::slow_and_pretty 0
		}
		return
	}
	#### Update the proteinID, since ::cartoonify is not connected to the MainFunction
	set ::bendix::proteinID [.bex.lf1.field1.molname get]
	
	### If subset of helicity fields have changed, non-Bendices will need redrawing to reflect the new bendices.
	### Remove current AT rep, mark this proteinID as not cartoonified and allow below loop to redraw reps
	if {$::bendix::previous_subset != $::bendix::subset || $::bendix::previous_helix_assignment != $::bendix::helix_assignment} {
		if { [lindex $::bendix::old_rep_list $::bendix::proteinID] != "" } {
			mol delrep [lindex $::bendix::old_rep_list $::bendix::proteinID] $::bendix::proteinID
			lset ::bendix::old_rep_list $::bendix::proteinID ""
		}
		lset ::bendix::cartoonified_list $::bendix::proteinID 0
	}
	
	if { [lindex $::bendix::cartoonified_list $::bendix::proteinID] == 0} {	

		if {$::bendix::autoCartoon == 1 || $::bendix::autoNewCartoon == 1 || $::bendix::autoTube == 1 || $::bendix::quick_and_dirty == 1 || $::bendix::slow_and_pretty == 1} {
			if {$::bendix::subset != ""} {
				if {$::bendix::slow_and_pretty == 1 && $::bendix::String_for_cartoonify != ""} {
					mol selection "($::bendix::subset) and sheet and not $::bendix::String_for_cartoonify"
				} elseif {$::bendix::slow_and_pretty == 1} {
					mol selection "($::bendix::subset) and sheet"
				} elseif {$::bendix::quick_and_dirty == 1 && $::bendix::String_for_cartoonify != ""} {
					mol selection "($::bendix::subset) and (name $::bendix::fixed_particleNames) and not $::bendix::String_for_cartoonify"
				} elseif {$::bendix::quick_and_dirty == 1} {
					mol selection "($::bendix::subset) and (name $::bendix::fixed_particleNames)" 		
				} elseif {$::bendix::helix_assignment != "" && $::bendix::String_for_cartoonify != ""} {
					mol selection "($::bendix::subset) and not $::bendix::String_for_cartoonify"
				} elseif {$::bendix::CG == 0 && $::bendix::String_for_cartoonify != ""} {
					mol selection "($::bendix::subset) and not helix"
				} else {
					mol selection "($::bendix::subset)"
				}
			} else {
			#### No subset detected
				if {$::bendix::slow_and_pretty == 1 && $::bendix::String_for_cartoonify != ""} {
					mol selection "sheet and not $::bendix::String_for_cartoonify"
				} elseif {$::bendix::slow_and_pretty == 1} {
					mol selection "sheet"
				} elseif {$::bendix::quick_and_dirty == 1 && $::bendix::String_for_cartoonify != ""} {
					mol selection "(name $::bendix::fixed_particleNames) and not $::bendix::String_for_cartoonify"
				} elseif {$::bendix::quick_and_dirty == 1} {
					mol selection "(name $::bendix::fixed_particleNames)" 
				} elseif {$::bendix::helix_assignment != "" && $::bendix::String_for_cartoonify != ""} {
					mol selection "not $::bendix::String_for_cartoonify"
				} elseif {$::bendix::CG == 0 && $::bendix::helix_assignment == ""} {
					mol selection "protein"; # so that when no helices are specified as helicity, it's reflected as an all-backbone protein.
					#mol selection "not helix"
				} else {
					mol selection "protein"
				}
			}
			lset ::bendix::cartoonified_list $::bendix::proteinID 1
			mol addrep $::bendix::proteinID
			#### I add exactly one representation, so I know that the last repID is bendix-created.
			set N_reps_so_far [molinfo $::bendix::proteinID get numreps]
			set N_reps_so_far [expr {$N_reps_so_far -1}]; # the index.
			
			#### Roundabout way to actually get the ID of the just-added rep, so we can delete it later:
			set repNumber [mol repname $::bendix::proteinID $N_reps_so_far];
			lset ::bendix::old_rep_list $::bendix::proteinID [mol repindex $::bendix::proteinID $repNumber]
			
			#### Colour by chain or entire protein, if auto-colour is on.
			if {$::bendix::uniform_colour == 1 && $::bendix::autoColour == 1} {
				if {$::bendix::uniform_colour_type == 1} {
					mol modcolor $N_reps_so_far $::bendix::proteinID Chain
				} else {
					mol modcolor $N_reps_so_far $::bendix::proteinID ColorID 2
				}
			} else {
				mol modcolor $N_reps_so_far $::bendix::proteinID ColorID 2
			}
		}

		#### Fix Material
		set material_choice [.bex.lf2.listbox get];
		if {$material_choice == ""} {
			set material_choice EdgyShiny
		}
		if {$::bendix::slow_and_pretty == 1} {
			mol modstyle $N_reps_so_far $::bendix::proteinID NewCartoon
			mol modmaterial $N_reps_so_far $::bendix::proteinID $material_choice
			.bex.lf3.field1.quick configure -state disable
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.tube configure -state disable
			
		} elseif {$::bendix::quick_and_dirty == 1} {
			mol modstyle $N_reps_so_far $::bendix::proteinID DynamicBonds 4.6 0.2 
			mol modmaterial $N_reps_so_far $::bendix::proteinID $material_choice
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.tube configure -state disable
			.bex.lf3.field1.slow configure -state disable
		} elseif {$::bendix::autoCartoon == 1} {
			mol modstyle $N_reps_so_far $::bendix::proteinID Cartoon
			mol modmaterial $N_reps_so_far $::bendix::proteinID $material_choice
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.tube configure -state disable
			.bex.lf3.field1.slow configure -state disable
			.bex.lf3.field1.quick configure -state disable
		} elseif {$::bendix::autoNewCartoon == 1} {
			mol modstyle $N_reps_so_far $::bendix::proteinID NewCartoon
			mol modmaterial $N_reps_so_far $::bendix::proteinID $material_choice
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.tube configure -state disable
			.bex.lf3.field1.slow configure -state disable
			.bex.lf3.field1.quick configure -state disable
		} elseif {$::bendix::autoTube == 1} {
			mol modstyle $N_reps_so_far $::bendix::proteinID Tube 0.2 $::bendix::spline_resolution
			mol modmaterial $N_reps_so_far $::bendix::proteinID $material_choice
			.bex.lf3.field1.cartoon configure -state disable
			.bex.lf3.field1.newcartoon configure -state disable
			.bex.lf3.field1.slow configure -state disable
			.bex.lf3.field1.quick configure -state disable
		} else {
			#### This may occur if helicity or subset is edited pre-Draw
			if { [lindex $::bendix::old_rep_list $::bendix::proteinID] != "" } {
				mol delrep [lindex $::bendix::old_rep_list $::bendix::proteinID] $::bendix::proteinID
				lset ::bendix::old_rep_list $::bendix::proteinID ""
			}
			.bex.lf3.field1.cartoon configure -state normal
			.bex.lf3.field1.newcartoon configure -state normal
			.bex.lf3.field1.tube configure -state normal
			.bex.lf3.field1.slow configure -state normal
			.bex.lf3.field1.quick configure -state normal
			lset ::bendix::cartoonified_list $::bendix::proteinID 0
		}
		#### The above (drawing cartoons) is allowed for molecules that are not previously cartooned.
		
	#### Eraser-loop
	} elseif {$::bendix::autoCartoon == 0 && $::bendix::autoNewCartoon == 0 && $::bendix::autoTube == 0 && $::bendix::quick_and_dirty == 0 && ($::bendix::slow_and_pretty == 0 && $::bendix::die_backbone == 1)} {

		if { [lindex $::bendix::old_rep_list $::bendix::proteinID] != "" } {
			mol delrep [lindex $::bendix::old_rep_list $::bendix::proteinID] $::bendix::proteinID
			lset ::bendix::old_rep_list $::bendix::proteinID ""
		}
		.bex.lf3.field1.cartoon configure -state normal
		.bex.lf3.field1.newcartoon configure -state normal
		.bex.lf3.field1.tube configure -state normal
		.bex.lf3.field1.slow configure -state normal
		.bex.lf3.field1.quick configure -state normal
		lset ::bendix::cartoonified_list $::bendix::proteinID 0
	}
  }


################################################################################
#                               HELIX ANALYSIS                                 #
################################################################################

# ::bendix::popup_for_Surf -----------------------------------------------------
#    Checks for only 1 assigned helix. If yes, stores the protein viewpoint
#    before hiding all MolIDs and creating a new MolID for easy graph and erase
#    without interfering with other displayed protein(s).
#    Creates an angle-graphing GUI wherefrom the user can plot 2D or 3D
#    helix angle dynamics. Optional display settings exist
#    (such as axes labels and graph axis compression) and one may save data.
#    Exit by X is prohibited, which forces quit by Return
#    which deletes the purpose-created MolID and resets the Display viewpoint
#    so that any change to viewpoint done unto graphs do not interfere
#    with the original protein view.
# ------------------------------------------------------------------------------

  proc ::bendix::popup_for_Surf {} {

	#### To catch zero frames
	set frameSum 0
	foreach frameN $::bendix::vmdFrame {
		set frameSum [expr {$frameSum + $frameN}]
	}
	if {$::bendix::points_to_graph == ""} {
		::bendix::MissingAngles
	} elseif {$::bendix::vmdFrame == "" || $frameSum == 0 || [llength $::bendix::vmdFrame] == 1} {
		catch {destroy .noframes2}
		toplevel .noframes2
		wm geometry .noframes2 +100+150
		#grab .noframes2
		wm title .noframes2 "No trajectory loaded?"
		message .noframes2.msg1 -width 350 \
			-text "bendix does not detect frames!\nHave you loaded and (importantly!) run your trajectory using bendices? Also, is 'Store dynamic data' ticked under bendix' Analysis menu?\n\nIf bendix drawing is on, as you update a frame, bendices are drawn to the screen and their coordinates stored for analysis, along with the current frame number.\n\nPlease play your trajectory using bendices and ensure that 'Store dynamic data' is ticked, then retry Analysis." -pady 15 -padx 15
		button .noframes2.okb -text Oops -background green -command {destroy .noframes2 ; return 0}
		pack .noframes2.msg1 -side top 
		pack .noframes2.okb -side bottom -pady 7
		return 0
	} elseif {[llength $::bendix::index_of_helix_assignment_used_for_angle] > 1 } {
	#### Check if the user is trying to graph >1 helices - not recommended.
		catch {destroy .bendix_graphs_max_1_helix}
		toplevel .bendix_graphs_max_1_helix
		wm geometry .bendix_graphs_max_1_helix 370x170+100+150
		#grab .bendix_graphs_max_1_helix
		wm title .bendix_graphs_max_1_helix "3D graph of helices"
		message .bendix_graphs_max_1_helix.msg1 -width 350 \
			-text "Please change your helix assignment to display ONE helix.\n\nbendix can't visualise multiple\
				bendices simultaneoulsy.\nIf you want to graph multiple bendices, either save the data\
				(File > Save surf data...) and use alternative graphing software, or graph one bendix at a time."\
			-pady 15 -padx 5
		button .bendix_graphs_max_1_helix.ok \
			-text OK \
			-pady 5 \
			-background green \
			-command {destroy .bendix_graphs_max_1_helix; return }
		pack .bendix_graphs_max_1_helix.msg1 \
			-side top 
		pack .bendix_graphs_max_1_helix.ok \
			-side bottom \
			-pady 5

	} else {
		
		### Save the protein viewpoint
		global protein_viewpoints
		set protein_viewpoints($::bendix::proteinID) [molinfo $::bendix::proteinID get {center_matrix rotate_matrix scale_matrix global_matrix}]
		
		#Hide other proteins
		set all_existing_molIDs [molinfo list]
		foreach molID $all_existing_molIDs {
			mol off $molID;
		}

		set ::bendix::surf_repID [mol new]
		catch {destroy .pop_surf}
		toplevel .pop_surf
		wm geometry .pop_surf +100+150
		grab .pop_surf
		wm title .pop_surf "Angle analysis"

		wm protocol .pop_surf WM_DELETE_WINDOW {
			
			### Show hidden protein(s) and check that the surf's molID, and the protein molID, whose data was used for the surf, exists.
			set all_existing_molIDs [molinfo list]
			set target_proteinID_exists 0
			set target_surfID_exists 0
			
			if {$all_existing_molIDs != ""} {
				foreach molID $all_existing_molIDs {
					mol on $molID;
					if {$molID == $::bendix::proteinID} {
							set target_proteinID_exists 1
					}
					if {$::bendix::surf_repID != ""} {
						if {$molID == $::bendix::surf_repID} {
							set target_surfID_exists 1
						}
					}
				}
			}
				
			### If surf molID exists, save the current surf perspective to re-enable it for the next loaded surf
			global edited_surf_viewpoint
			if {$::bendix::surf_repID != "" && $target_surfID_exists == 1} {
				set edited_surf_viewpoint([expr $::bendix::surf_repID + 1]) [molinfo $::bendix::surf_repID get {center_matrix rotate_matrix scale_matrix global_matrix}]
				#set edited_surf_viewpoint [molinfo $::bendix::surf_repID get {center_matrix rotate_matrix scale_matrix global_matrix}]
				mol delete $::bendix::surf_repID
				set ::bendix::surf_repID ""
			}
	
			set ::bendix::first_surf_drawn ""
			set ::bendix::first_axis_drawn ""
			set ::bendix::last_axis_drawn ""
			set ::bendix::surfed 0
			set ::bendix::axesON 0
			
			#### Unset any memory of past surf reorientations
			global edited_surf_viewpoint
			if {$::bendix::surf_repID != "" && $target_surfID_exists == 1} {
				if {[info exists edited_surf_viewpoint([expr $::bendix::surf_repID +1])]} {
					unset edited_surf_viewpoint([expr $::bendix::surf_repID +1])
				}
			}
				
			#### Check that target protein exists, and if it does, return to its original protein viewpoint. Else centre the new top molecule.
			global protein_viewpoints
			if {$target_proteinID_exists == 1} {
				molinfo $::bendix::proteinID set {center_matrix rotate_matrix scale_matrix global_matrix} $protein_viewpoints($::bendix::proteinID)
			} else {
				### VMD's viewpoint centers on the top molID automatically.
				display resetview
			}
			
			if {$all_existing_molIDs == ""} {
				display resetview
			}
			destroy .pop_surf
			return
		}

		#### Menu
		frame .pop_surf.menubar -relief raised -bd 2
		pack .pop_surf.menubar -padx 1 -fill x -side top

		menubutton .pop_surf.menubar.openfiles -text "File" -underline 0 -menu .pop_surf.menubar.openfiles.menu
		.pop_surf.menubar.openfiles config -width 5	
   		pack .pop_surf.menubar.openfiles -side left
		menu .pop_surf.menubar.openfiles.menu -tearoff no		
		.pop_surf.menubar.openfiles.menu add command -label "Save surf data..." -command ::bendix::saveSurfData
		.pop_surf.menubar.openfiles.menu add command -label "Erase and return" \
			-command {
				::bendix::erase_surf
				#### Unset any memory of past surf reorientations
				global edited_surf_viewpoint
				if {$::bendix::surf_repID != ""} {
					if {[info exists edited_surf_viewpoint([expr $::bendix::surf_repID +1])]} {
						unset edited_surf_viewpoint([expr $::bendix::surf_repID +1])
					}
				}
				
				### Show hidden protein(s)
				set all_existing_molIDs [molinfo list]
				if {$all_existing_molIDs != ""} {
					set target_proteinID_exists 0
					foreach molID $all_existing_molIDs {
						mol on $molID;
						if {$molID == $::bendix::proteinID} {
							set target_proteinID_exists 1
						}
					}
					#### Check that target protein exists, and if it does, return to its original protein viewpoint. Else centre the new top molecule.
					global protein_viewpoints
					if {$target_proteinID_exists == 1} {
						molinfo $::bendix::proteinID set {center_matrix rotate_matrix scale_matrix global_matrix} $protein_viewpoints($::bendix::proteinID)
					} else {
						### VMD's viewpoint centers on the top molID automatically.
						display resetview
					}
				} else {
					display resetview
				}
				destroy .pop_surf
				return
			}

		labelframe .pop_surf.lf -text "Heatmap settings" -pady 5 -padx 6 -font {-weight bold -family helvetica}

		frame .pop_surf.lf.field1
		frame .pop_surf.lf.field2
		frame .pop_surf.lf.field3
		frame .pop_surf.lf.field4
		frame .pop_surf.lf.field5
		frame .pop_surf.lf.field6
		frame .pop_surf.field7

		message .pop_surf.lf.field1.msg1 -width 300 \
			-text "Please choose from the settings below, then hit Surf." -pady 3
		radiobutton .pop_surf.lf.field2.2d -variable ::bendix::3D_surf \
			-value 0 \
			-text "2D" \
			-command {
				.pop_surf.lf.field3.msg2 configure -foreground "gray60"
				.pop_surf.lf.field3.anglex configure -state disable
			}
		radiobutton .pop_surf.lf.field2.3d -variable ::bendix::3D_surf -value 1 -text "3D" \
			-command {
				.pop_surf.lf.field3.msg2 configure -foreground black
				.pop_surf.lf.field3.anglex configure -state normal
			}

		message .pop_surf.lf.field3.msg -justify left -width 150 -text "Compression:"
		if {$::bendix::3D_surf == 0} {
			message .pop_surf.lf.field3.msg2 -justify left -width 150 -text "3D peaks" -foreground "gray60"
			entry .pop_surf.lf.field3.anglex \
				-textvariable ::bendix::z_squeeze \
				-state disabled \
				-background white \
				-selectbackground yellow2 \
				-width 4
		} else {
			message .pop_surf.lf.field3.msg2 -justify left -width 150 -text "3D peaks" -foreground black
			entry .pop_surf.lf.field3.anglex -textvariable ::bendix::z_squeeze \
				-background white \
				-selectbackground yellow2 \
				-width 4
		}
		message .pop_surf.lf.field3.spacer -justify left -width 60 -text " "
		message .pop_surf.lf.field3.msg3 -justify left -width 150 -text "Frame axis"
		entry .pop_surf.lf.field3.framex \
			-textvariable ::bendix::frame_squeeze \
			-background white \
			-selectbackground yellow2 \
			-width 4
		checkbutton .pop_surf.lf.field4.axesON -text "Show axes" \
			-variable ::bendix::axesON \
			-command ::bendix::surf_axes
		checkbutton .pop_surf.lf.field5.checkON -text "Scale heatmap by the maximum angle in the trajectory" \
			-variable ::bendix::AngleAutoScale \
			-command {
				if {$::bendix::axesON == 1} { 
					::bendix::erase_surf
					set ::bendix::axesON 1
					::bendix::create_AngleHelixTime_graph
					::bendix::surf_axes
					.pop_surf.field7.graph configure -state disable
				} else {
					::bendix::erase_surf
					::bendix::create_AngleHelixTime_graph
					.pop_surf.field7.graph configure -state disable
				}
			}
		message .pop_surf.lf.field6.msg -justify left -width 300 \
			-text "Alternatively the colour threshold is used." \
			-font {helvetica 8}
		button .pop_surf.field7.graph -text Surf -background green \
			-command { ::bendix::create_AngleHelixTime_graph; .pop_surf.field7.graph configure -state disable}
		button .pop_surf.field7.erase -text Erase -background tomato -command {::bendix::erase_surf; .pop_surf.field7.erase configure -state disable}
		
		if {$::bendix::surfed == 1} {
			.pop_surf.field7.graph configure -state disable
		} else {
			.pop_surf.field7.erase configure -state disable
		}
		

		button .pop_surf.field7.okb -text Return \
			-command {
				::bendix::erase_surf
				#### Unset any memory of past surf reorientations
				global edited_surf_viewpoint
				if {$::bendix::surf_repID != ""} {
					if {[info exists edited_surf_viewpoint([expr $::bendix::surf_repID +1])]} {
						unset edited_surf_viewpoint([expr $::bendix::surf_repID +1])
					}
				}
				
				### Show hidden protein(s) and c.f. existing MolIDs to what molID's data Surf uses. 
				### If the user accidentally deletes this molecule, allow exit and re-focus viewpoint on the new top molecule, if any.
				set all_existing_molIDs [molinfo list]
				if {$all_existing_molIDs != ""} {
					set target_proteinID_exists 0
					foreach molID $all_existing_molIDs {
						mol on $molID;
						if {$molID == $::bendix::proteinID} {
							set target_proteinID_exists 1
						}
					}
					#### Check that target protein exists, and if it does, return to its original protein viewpoint. Else centre the new top molecule.
					global protein_viewpoints
					if {$target_proteinID_exists == 1} {
						molinfo $::bendix::proteinID set {center_matrix rotate_matrix scale_matrix global_matrix} $protein_viewpoints($::bendix::proteinID)
					} else {
						### VMD's viewpoint centers on the top molID automatically.
						display resetview
					}
				} else {
					display resetview
				}
				destroy .pop_surf
				return
			}
	
		pack .pop_surf.lf.field1 -side top -anchor w
		pack .pop_surf.lf.field1.msg1 -side left
	
		pack .pop_surf.lf.field2 -side top -anchor w
		pack .pop_surf.lf.field2.2d -side left
		pack .pop_surf.lf.field2.3d -side right

		pack .pop_surf.lf.field3 -side top -anchor w
		pack .pop_surf.lf.field3.msg -side left
		pack .pop_surf.lf.field3.msg2 -side left
		pack .pop_surf.lf.field3.anglex -side left
		pack .pop_surf.lf.field3.spacer -side left
		pack .pop_surf.lf.field3.msg3 -side left
		pack .pop_surf.lf.field3.framex -side left

		pack .pop_surf.lf.field4 -side top -anchor w
		pack .pop_surf.lf.field4.axesON -side left

		pack .pop_surf.lf.field5 -side top -anchor w
		pack .pop_surf.lf.field5.checkON -side left

		pack .pop_surf.lf.field6 -side top
		pack .pop_surf.lf.field6.msg -side left

		pack .pop_surf.field7 -side bottom
		pack .pop_surf.field7.graph -side left -padx 10 -pady 5
		pack .pop_surf.field7.erase -side left -padx 10 -pady 5
		pack .pop_surf.field7.okb -side right -padx 10 -pady 5

		pack .pop_surf.lf -side top -padx 4 -pady 2 -anchor w -expand true -fill x
	}
  }

# ::bendix::erase_surf ---------------------------------------------------------
#    Saves the current graph viewpoint for easy redraw using the same prespective.
#    Auto-erases graphics primitives used to depict 2D/3D graphs by MolID deletion.
#    Corrects the Angle graphing GUI to enable Surfs again
#    and unchecks the Draw axes box.
# ------------------------------------------------------------------------------

  proc ::bendix::erase_surf {} {
	
	### Check that the surfID exists
	set all_existing_molIDs [molinfo list]
	set target_surfID_exists 0
	set ::bendix::first_surf_drawn ""
	set ::bendix::first_axis_drawn ""
	set ::bendix::last_axis_drawn ""
	.pop_surf.field7.graph configure -state normal
	set ::bendix::surfed 0
	set ::bendix::axesON 0
	
	if {$all_existing_molIDs != "" && $::bendix::surf_repID != ""} {
		foreach molID $all_existing_molIDs {
			if {$molID == $::bendix::surf_repID} {
				set target_surfID_exists 1
			}
		}
		
		### Save the current perspective, prepare for the next
		global edited_surf_viewpoint
		if {$target_surfID_exists == 1} {
			set edited_surf_viewpoint([expr $::bendix::surf_repID + 1]) [molinfo $::bendix::surf_repID get {center_matrix rotate_matrix scale_matrix global_matrix}]
			mol delete $::bendix::surf_repID
		}
	}
	set ::bendix::surf_repID ""
  }

# ::bendix::create_AngleHelixTime_graph ----------------------------------------
#    A 2D or 3D surf of angle vs helix length vs frame number is created
#    by triangulation using graphics primitives, and coloured by local angle size.
#    Calls the MissingAngles proc if angles are not stored
#    and prevents further proc if > 1 helix is assigned.
#    Identifies the maximum angle throughout a trajectory
#    to optionally scale all other angles for colouring purposes.
#    Optionally scales the frame or angle axis to allow user control
#    of the look of the output graph.
# ------------------------------------------------------------------------------

  proc ::bendix::create_AngleHelixTime_graph {} {

	#### To catch zero frames
	set frameSum 0
	foreach frameN $::bendix::vmdFrame {
		set frameSum [expr {$frameSum + $frameN}]
	}
	if {$::bendix::points_to_graph == ""} {
		::bendix::MissingAngles
	} elseif {$::bendix::vmdFrame == "" || $frameSum == 0 || [llength $::bendix::vmdFrame] == 1 } {
		catch {destroy .noframes}
		toplevel .noframes
		wm geometry .noframes +100+150
		#grab .noframes
		wm title .noframes "No trajectory loaded?"
		message .noframes.msg1 -width 350 \
			-text "bendix does not detect frames!\nHave you loaded and (importantly!) run your trajectory using bendices? Also, is 'Store dynamic data' ticked under bendix' Analysis menu?\n\nIf bendix drawing is on, as you update a frame, bendices are drawn to the screen and their coordinates stored for analysis, along with the current frame number.\n\nPlease play your trajectory using bendices and ensure that 'Store dynamic data' is ticked, then retry Analysis." -pady 15 -padx 15
		button .noframes.okb -text Oops -background green -command {destroy .noframes ; return 0}
		pack .noframes.msg1 -side top 
		pack .noframes.okb -side bottom -pady 7
	    #### Upon window close or minimisation, exit properly.
	    #bind .noframes <Unmap> {
		#	destroy .noframes ; return 0
		#}
		return 0
	} elseif {[llength $::bendix::index_of_helix_assignment_used_for_angle] > 1 } {
		catch {destroy .messpopHelixN}
		toplevel .messpopHelixN
		wm geometry .messpopHelixN +100+150
		#grab .messpopHelixN
		wm title .messpopHelixN "3D graph of helices"
		message .messpopHelixN.msg1 -width 350 \
			-text "Visualisation superimposes multiple bendices' graphs. This makes it hard to read.\
\n\nIf you want to graph multiple bendices, either save the data and use alternative graphing software,\
or graph one bendix at a time." -pady 15 -padx 5
		button .messpopHelixN.ok -text Return -background green \
			-command {
				::bendix::erase_surf
				
				### Show hidden protein(s) and c.f. existing MolIDs to what molID's data Surf uses. 
				### If the user accidentally deletes this molecule, allow exit and re-focus viewpoint on the new top molecule, if any.
				set all_existing_molIDs [molinfo list]
				if {$all_existing_molIDs != ""} {
					set target_proteinID_exists 0
					foreach molID $all_existing_molIDs {
						mol on $molID;
						if {$molID == $::bendix::proteinID} {
							set target_proteinID_exists 1
						}
					}
					#### Check that target protein exists, and if it does, return to its original protein viewpoint. Else centre the new top molecule.
					global protein_viewpoints
					if {$target_proteinID_exists == 1} {
						molinfo $::bendix::proteinID set {center_matrix rotate_matrix scale_matrix global_matrix} $protein_viewpoints($::bendix::proteinID)
					} else {
						### VMD's viewpoint centers on the top molID automatically.
						display resetview
					}
				} else {
					display resetview
				}
				destroy .messpopHelixN
				destroy .pop_surf
				return 0
			}
			
		pack .messpopHelixN.msg1 -side top 
		pack .messpopHelixN.ok -side bottom -pady 5

	    #### Upon window close or minimisation, exit properly.
	    #bind .messpopHelixN <Unmap>
		wm protocol .messpopHelixN WM_DELETE_WINDOW {	
			::bendix::erase_surf
			
			### Show hidden protein(s)
			set all_existing_molIDs [molinfo list]
			if {$all_existing_molIDs != ""} {
				set target_proteinID_exists 0
				foreach molID $all_existing_molIDs {
					mol on $molID;
					if {$molID == $::bendix::proteinID} {
						set target_proteinID_exists 1
					}
				}
				#### Check that target protein exists, and if it does, return to its original protein viewpoint. Else centre the new top molecule.
				global protein_viewpoints
				if {$target_proteinID_exists == 1} {
					molinfo $::bendix::proteinID set {center_matrix rotate_matrix scale_matrix global_matrix} $protein_viewpoints($::bendix::proteinID)
				} else {
					### VMD's viewpoint centers on the top molID automatically.
					display resetview
				}
			} else {
				display resetview
			}
			destroy .messpopHelixN
			destroy .pop_surf
			return 0
		}

	} else {
		if {$::bendix::surf_repID == ""} {
			# New molID, for easy graph removal
			set ::bendix::surf_repID [mol new]
			mol top $::bendix::surf_repID
		} else {
			# Axes drawing or the surf window has already created the ID 
		}
		
		#### Fix Angle colour scale
		set scale_choice [.bex.lf2.field5.listbox get]
		if {$scale_choice == ""} {
			color scale method RGB
		} else {
			color scale method $scale_choice
		}
		
		if {$::bendix::AngleAutoScale == 1} {	
		#### Uses XTC-determined maximum and scales all else accordingly.	
			set sorted_MaxAngles_max_min [lsort -real -decreasing $::bendix::maximumAngle_per_frame]
 			set max_angle [lindex $sorted_MaxAngles_max_min 0]
		} else {
		#### Uses user-defined maxthreshold as reference.
			set max_angle $::bendix::angle_max
		}
		set ::bendix::frame_squeeze [expr {$::bendix::frame_squeeze/1.0}];
		set ::bendix::z_squeeze [expr {$::bendix::z_squeeze/1.0}]
		set avg_Xinterdistance [expr [lindex $::bendix::curvature_graph_X 1] - [lindex $::bendix::curvature_graph_X 0]]; # NB if larger than 1, it'll ruin the graph..
		set avg_Xinterdistance [expr [lindex $::bendix::curvature_graph_X 0] - $avg_Xinterdistance]

		### Plus one necessary as triangles are drawn between curr and next value.
		for {set frameN 0} {$frameN < [expr {[llength $::bendix::vmdFrame] -1}]} {incr frameN} {
			for {set turn -1} {$turn < [expr {[llength $::bendix::curvature_graph_X] -1}]} {incr turn} {
				
				### Special first triangles based on x 0.something to 1 -- error if really big splinepoint distances  (== low resolution graph)
				if {$turn == -1} {
					#### Colour it in:
					set scaled_angle [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] 0] / $max_angle}]	
					if {$scaled_angle < 0.9375 } {
						set colour_value [expr {round(1000-$scaled_angle*1000)}]
						graphics $::bendix::surf_repID color $colour_value
					} else {
						graphics $::bendix::surf_repID color 33
					}
					
					### NB Below can give error ("can't do division with nothing") if frames have been manually deleted from the trajectory.
					if {$::bendix::3D_surf == 1} {
						#### Draw 3D surf: The first triangulated unit is not implemented in 3D.
						#draw triangle "$avg_Xinterdistance [expr {[lindex $::bendix::vmdFrame $frameN] /$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] 0]/$::bendix::z_squeeze}]" "$avg_Xinterdistance [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame [expr {$frameN +1}]] 0]/$::bendix::z_squeeze}]" "[lindex $::bendix::curvature_graph_X 0] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame [expr {$frameN +1}]] 1]/$::bendix::z_squeeze}]"
						#draw triangle "$avg_Xinterdistance [expr {[lindex $::bendix::vmdFrame $frameN] /$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] 0]/$::bendix::z_squeeze}]" "[lindex $::bendix::curvature_graph_X 0] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] 1]/$::bendix::z_squeeze}]" "[lindex $::bendix::curvature_graph_X 0] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame [expr {$frameN +1}]] [expr 1] ]/$::bendix::z_squeeze}]"
				
					} else {
						#### Draw 2D heatmap:
						draw triangle "$avg_Xinterdistance [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] 0" "$avg_Xinterdistance [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] 0" "[lindex $::bendix::curvature_graph_X 0] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] 0"
						draw triangle "$avg_Xinterdistance [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] 0" "[lindex $::bendix::curvature_graph_X 0] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] 0" "[lindex $::bendix::curvature_graph_X 0] [expr {[lindex $::bendix::vmdFrame [expr $frameN +1]]/$::bendix::frame_squeeze}] 0"
					}
				} else {
					set scaled_angle [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] [expr {$turn+1}]] / $max_angle}]	
					if {$scaled_angle < 0.9375 } {
						set colour_value [expr {round(1000-$scaled_angle*1000)}]
						graphics $::bendix::surf_repID color $colour_value
					} else {
						graphics $::bendix::surf_repID color 33
					}
					### NB Below can give error ("can't do division with nothing") if frames have been manually deleted from the trajectory.
					if {$::bendix::3D_surf == 1} {
						draw triangle "[lindex $::bendix::curvature_graph_X $turn] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] $turn]/$::bendix::z_squeeze}]" "[lindex $::bendix::curvature_graph_X $turn] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame [expr {$frameN +1}]] $turn]/$::bendix::z_squeeze}]" "[lindex $::bendix::curvature_graph_X [expr {$turn +1}]] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame [expr {$frameN +1}]] [expr {$turn +1}]]/$::bendix::z_squeeze}]"
						draw triangle "[lindex $::bendix::curvature_graph_X $turn] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] $turn]/$::bendix::z_squeeze}]" "[lindex $::bendix::curvature_graph_X [expr {$turn +1}]] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame $frameN] [expr {$turn +1}]]/$::bendix::z_squeeze}]" "[lindex $::bendix::curvature_graph_X [expr {$turn +1}]] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] [expr {[lindex [lindex $::bendix::angle_per_turn_per_frame [expr {$frameN +1}]] [expr {$turn +1}] ]/$::bendix::z_squeeze}]"
				
					} else {
						draw triangle "[lindex $::bendix::curvature_graph_X $turn] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] 0" "[lindex $::bendix::curvature_graph_X $turn] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] 0" "[lindex $::bendix::curvature_graph_X [expr {$turn +1}]] [expr {[lindex $::bendix::vmdFrame [expr {$frameN +1}]]/$::bendix::frame_squeeze}] 0"
						draw triangle "[lindex $::bendix::curvature_graph_X $turn] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] 0" "[lindex $::bendix::curvature_graph_X [expr {$turn +1}]] [expr {[lindex $::bendix::vmdFrame $frameN]/$::bendix::frame_squeeze}] 0" "[lindex $::bendix::curvature_graph_X [expr {$turn +1}]] [expr {[lindex $::bendix::vmdFrame [expr $frameN +1]]/$::bendix::frame_squeeze}] 0"
				
					}
				}
			}
		}
		set ::bendix::surfed 1
		.pop_surf.field7.erase configure -state normal
		
		### Fix viewpoint
		global edited_surf_viewpoint
		
		if {[info exists edited_surf_viewpoint($::bendix::surf_repID) ]} {
			molinfo $::bendix::surf_repID set {center_matrix rotate_matrix scale_matrix global_matrix} $edited_surf_viewpoint($::bendix::surf_repID)
		} else {
			display resetview
		}
	}
  }

# ::bendix::surf_axes ----------------------------------------------------------
#    Called by the ::bendix::popup_for_Surf proc to visualise labelled axes and,
#    in the case of the 2D surf heatmap, create a colour scale. 
#    All is drawn using graphics primitives whose identities are stored
#    for easy removal separate to the graph.
# ------------------------------------------------------------------------------

  proc ::bendix::surf_axes {} {

	if {$::bendix::points_to_graph == ""} {
		#### Where there's nothing to graph, return error.
		::bendix::MissingAngles

	} elseif {[llength $::bendix::index_of_helix_assignment_used_for_angle] > 1 } {
		catch {destroy .messpopHelixNaxes}
		toplevel .messpopHelixNaxes
		wm geometry .messpopHelixNaxes +100+150
		#grab .messpopHelixNaxes
		wm title .messpopHelixNaxes "3D graph of helices"
		message .messpopHelixNaxes.msg1 -width 350 \
-text "Visualisation superimposes multiple bendices' graphs. This makes it hard to read.\
\n\nIf you want to graph multiple bendices, either save the data and use alternative graphing software,\
or graph one bendix at a time." -pady 15 -padx 5
		button .messpopHelixNaxes.ok -text Return -background green \
			-command {
				destroy .messpopHelixNaxes
	 			return 0
			}
		pack .messpopHelixNaxes.msg1 -side top 
		pack .messpopHelixNaxes.ok -side bottom -pady 5

	    #### Upon window close or minimisation, exit properly.
	    #bind .messpopHelixNaxes <Unmap> {
		#	destroy .messpopHelixNaxes
		#	return 0
		#}

	} else {
		
		if {$::bendix::axesON == 1} {
			#### If the box is checked:
			#### Keep track of what has been drawn already:
			if {$::bendix::surf_repID != ""} {
				set AllQualitativeThingsDrawn [graphics $::bendix::surf_repID list]
			} else {
				# New molID, for easy graph removal
				set ::bendix::surf_repID [mol new]
				mol top $::bendix::surf_repID
				set AllQualitativeThingsDrawn ""
				
				### Fix viewpoint (depreciated, since prespective reset confuses, rather than helps, the user at the end of Surf module reset)
				#set surf_viewpoints($::bendix::surf_repID) {{{1 0 0 -75.8589} {0 1 0 -75.0031} {0 0 1 -158.161} {0 0 0 1}} {{1 0.000365972 0.00386288 0} {-0.00219332 0.874808 0.484487 0} {-0.00320129 -0.484521 0.874803 0} {0 0 0 1}} {{0.0414482 0 0 0} {0 0.0414482 0 0} {0 0 0.0414482 0} {0 0 0 1}} {{1 0 0 2.34001} {0 1 0 4.63997} {0 0 1 0} {0 0 0 1}}}
				#set surf_viewpoints($::bendix::proteinID) 
				#molinfo $::bendix::surf_repID set {center_matrix rotate_matrix scale_matrix global_matrix} $surf_viewpoints($::bendix::surf_repID)
				#mol top $::bendix::surf_repID
				#display resetview
			}
			if {$AllQualitativeThingsDrawn != ""} {
				set ::bendix::first_axis_drawn [expr {[lindex $AllQualitativeThingsDrawn [expr {[llength $AllQualitativeThingsDrawn] -1}]] +1}]
			} else {
				set ::bendix::first_axis_drawn 0
			}

			if {$::bendix::AngleAutoScale == 1} {	
			#### Uses XTC-determined maximum and scales all else accordingly.		
				set sorted_MaxAngles_max_min [lsort -real -decreasing $::bendix::maximumAngle_per_frame]
				set max_angle [lindex $sorted_MaxAngles_max_min 0]
			} else {
			#### Uses user-defined maxthreshold as reference.
				set max_angle $::bendix::angle_max
			}
			set ::bendix::frame_squeeze [expr {$::bendix::frame_squeeze/1.0}]
			set ::bendix::z_squeeze [expr {$::bendix::z_squeeze/1.0}]
			set Y_shift [expr [lindex $::bendix::vmdFrame 0]/$::bendix::frame_squeeze]

			#### Drawing an axis for Helix residues
			graphics $::bendix::surf_repID color gray
			graphics $::bendix::surf_repID line "0 $Y_shift 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +1}] $Y_shift 0" width 2 style solid
			graphics $::bendix::surf_repID cone "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +1}] $Y_shift 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +2}] $Y_shift 0" radius 0.7 resolution 30
			
			###Minor edit for better positioning in 3D
			if {$::bendix::3D_surf == 0} {
				graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] $Y_shift 0" "Residues"
			} else {
				graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {$Y_shift} -2] 0" "Residues"
			}


			#### Stored residue numbers that correspond to a precise location on the helix surf graph:
			if {$::bendix::TurnResolution == 1} {
				set fontSize "0.8"
			} else {
				set fontSize "1"
			}
			
			foreach XaxisN $::bendix::realAAs_for_23D_graph_Xaxis {
				set x_index 0
				set currXvalue_at_1dp [expr 1.0*$XaxisN]
				foreach allX_axis_values $::bendix::curvature_graph_X {
					if {$currXvalue_at_1dp == $allX_axis_values} {
						if {$currXvalue_at_1dp == 1.0} {
							if {$::bendix::3D_surf == 0} {
								set midPoint_bn_points [expr [lindex $::bendix::curvature_graph_X 0] - [expr [expr [lindex $::bendix::curvature_graph_X 1] - [lindex $::bendix::curvature_graph_X 0]]/2.0]]; # Average loci based on adjacent spots.
							} else {
								set midPoint_bn_points [expr [lindex $::bendix::curvature_graph_X 0] + [expr [expr [lindex $::bendix::curvature_graph_X 1] - [lindex $::bendix::curvature_graph_X 0]]/2.0]]; # Average loci based on adjacent spots.
							}
						} else {
							set midPoint_bn_points [expr [lindex $::bendix::curvature_graph_X [expr $x_index -1]] + [expr [expr [lindex $::bendix::curvature_graph_X $x_index] - [lindex $::bendix::curvature_graph_X [expr $x_index -1]]]/2.0]]
						}
						graphics $::bendix::surf_repID line "$midPoint_bn_points [expr -0.6 + $Y_shift] 0" "$midPoint_bn_points $Y_shift 0" width 2 style solid
						graphics $::bendix::surf_repID text "[expr $midPoint_bn_points -0.2] [expr -3.5 + $Y_shift] 0" "$XaxisN" size $fontSize
					}
					incr x_index
				}				
			}
			
			#### Drawing an axis for 2D graph Frame Number (different side if 3D graph)
			if {$::bendix::3D_surf == 0} {
				set the_FrameN_X 0
				set the_FrameSide 1
			} else {
				set the_FrameN_X [lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]]
				set the_FrameSide -1
			}
			
			graphics $::bendix::surf_repID color gray
			graphics $::bendix::surf_repID line "$the_FrameN_X $Y_shift 0" "$the_FrameN_X [expr {[lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]]/$::bendix::frame_squeeze +1}] 0" width 2 style solid
			graphics $::bendix::surf_repID cone "$the_FrameN_X [expr {[lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]]/$::bendix::frame_squeeze +1}] 0" "$the_FrameN_X [expr {[lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]]/$::bendix::frame_squeeze +2}] 0" radius 0.7 resolution 30
			
			#### Minor edit for correcting axis label position
			if {$::bendix::3D_surf == 0} {
				graphics $::bendix::surf_repID text "-8 [expr {[lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]]/$::bendix::frame_squeeze +3.5}] 0" "Frame number"
			} else {
				graphics $::bendix::surf_repID text "[expr {$the_FrameN_X -0}] [expr {[lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]]/$::bendix::frame_squeeze +7}] 0" "Frame number"
			}
			
			foreach line {0 1 2 3 4 5} {
			#### Make 6 equispaced notches along the Frame axis (5 in 3D)
				if {$::bendix::3D_surf == 1 && $line == 0} {
					# No first frame written out in 3D (overlaps with 'Residues' text)
				} else {
					graphics $::bendix::surf_repID line "$the_FrameN_X [expr [expr {[expr {[expr {[expr [lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]] - [lindex $::bendix::vmdFrame 0]]/	$::bendix::frame_squeeze} ] / 5}] *$line}] +$Y_shift] 0" "[expr {$the_FrameN_X - [expr {$the_FrameSide * 0.6}] }] [expr [expr {[expr {[expr {[expr [lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]] - [lindex $::bendix::vmdFrame 0]]/ $::bendix::frame_squeeze} ] / 5}] *$line}] + $Y_shift] 0" width 2 style solid
					set frameText [expr {int([expr [expr {[expr { [expr [lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]] - [lindex $::bendix::vmdFrame 0]]/5.0 }] *$line}] + [lindex $::bendix::vmdFrame 0]])}]
					#### Minor edit for correcting axis number position
					if {$::bendix::3D_surf == 0} {
						graphics $::bendix::surf_repID text "-5.3 [expr [expr {[expr {[expr {[expr [lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]] - [lindex $::bendix::vmdFrame 0]]/ $::bendix::frame_squeeze} ] / 5}] *$line}] +$Y_shift] 0" "$frameText"
					} else {
						graphics $::bendix::surf_repID text "[expr {$the_FrameN_X +2 }] [expr [expr {[expr {[expr {[expr [lindex $::bendix::vmdFrame [expr {[llength $::bendix::vmdFrame] -1}]] - [lindex $::bendix::vmdFrame 0]]/	$::bendix::frame_squeeze} ] / 5}] *$line}] +$Y_shift] 0" "$frameText"
					}
				}
			}
			
			if {$::bendix::3D_surf == 1} {
			#### Drawing an axis for Angle (peaks)
				graphics $::bendix::surf_repID line "0 0 0" "0 0 [expr {$max_angle / $::bendix::z_squeeze + 1}]" width 2 style solid
				graphics $::bendix::surf_repID cone "0 0 [expr {$max_angle / $::bendix::z_squeeze + 1}]" "0 0 [expr {$max_angle / $::bendix::z_squeeze + 2}]" radius 0.7 resolution 30
				graphics $::bendix::surf_repID text "-5 -2 [expr {$max_angle / $::bendix::z_squeeze + 3}]" "Angle"
				
				foreach angleNotch {1 2 3 4} {
					graphics $::bendix::surf_repID line "0 0 [expr {[expr {[expr {$max_angle / $::bendix::z_squeeze}] /4}] *$angleNotch}]" "-0.6 -0.6 [expr {[expr {[expr {$max_angle / $::bendix::z_squeeze}] /4}] *$angleNotch}]" width 2 style solid
		 			set AngleFloat [expr {[expr {$max_angle/4 }] *$angleNotch}]
		 			
		 			set AngleText_1dp [expr {double(round(10*$AngleFloat))/10}]; # Novel 1dp by rounding rather than cutting off at 1dp
		 			
#					#### Adjust this 6 dp floating point to a 1 dp floating point, which fits the screen:
#					set splitFloat [split $AngleFloat {}]
#					set AngleText [lindex $splitFloat 0]
#					set decimalpoint 0
#					for {set count 1} {$count < [expr {[llength $splitFloat]-1}]} {incr count} {
#						if {[lindex $splitFloat $count] == "."} {
#						#### Detected the decimal point
#							set decimalpoint 1
#						}
#						if {$decimalpoint == 1} {
#							#### Add the point and one dp to the string (last thing done to it)
#							lappend AngleText [lindex $splitFloat $count]
#							lappend AngleText [lindex $splitFloat [expr {$count +1}]]
#							set decimalpoint 2
#						}
#						if {$decimalpoint != 2} {
#						#### If no point detected, add to the string.
#							lappend AngleText [lindex $splitFloat $count]
#						}
#					}
#					set AngleText_1dp [join $AngleText ""]
					
					graphics $::bendix::surf_repID text "-5 -3 [expr {[expr {[expr {$max_angle / $::bendix::z_squeeze}] /4}] *$angleNotch}]" "$AngleText_1dp"
				}
				
			} else {
			#### Colour bar for 2D heatmap, using scale-compatible colour
				set scaled_angle [expr {0/50}]
				set colour_value [expr {round(1000-$scaled_angle*1000)}]
				graphics $::bendix::surf_repID color $colour_value
				#### 0th colourbar
				graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {3 + $Y_shift}] 0"	"[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {5 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {5 + $Y_shift}] 0"
				graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {3 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {3 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {5 + $Y_shift}] 0"
				graphics $::bendix::surf_repID color gray
				graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5.5}] [expr {4 + $Y_shift}] 0" "0"
				#### Adjust the colour scale output to one decimal point
				set scaleN2 [expr {$max_angle / 5}]
				set scaleN3 [expr {[expr {$max_angle / 5}] *2}]
				set scaleN3b [expr {[expr {$max_angle / 5}] *3}]
				set scaleN3c [expr {[expr {$max_angle / 5}] *4}]
				set scaleN4 $max_angle
				set scaleCount 2
				
				foreach colourscaleN [list $scaleN2 $scaleN3 $scaleN3b $scaleN3c $scaleN4] {

					set AngleText_1dp [expr {double(round(10*$colourscaleN))/10}]
					
					if {$scaleCount == 2} {
						set scaled_angle 0.2
						set colour_value [expr {round(1000-$scaled_angle*1000)}]

						graphics $::bendix::surf_repID color $colour_value
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {5 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {7 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {7 + $Y_shift}] 0"
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {5 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {5 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {7 + $Y_shift}] 0"
						graphics $::bendix::surf_repID color gray
						graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5.5}] [expr {6 + $Y_shift}] 0" "$AngleText_1dp"
					} elseif {$scaleCount == 3} {
						set scaled_angle 0.4
						set colour_value [expr {round(1000-$scaled_angle*1000)}]
						graphics $::bendix::surf_repID color $colour_value
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {7 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {9 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {9 + $Y_shift}] 0"
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {7 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {7 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {9 + $Y_shift}] 0"
						graphics $::bendix::surf_repID color gray
						graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5.5}] [expr {8 + $Y_shift}] 0" "$AngleText_1dp"
					} elseif {$scaleCount == 4} {
						set scaled_angle 0.6
						set colour_value [expr {round(1000-$scaled_angle*1000)}]
						graphics $::bendix::surf_repID color $colour_value
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {9 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {11 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {11 + $Y_shift}] 0"
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {9 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {9 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {11 + $Y_shift}] 0"
						graphics $::bendix::surf_repID color gray
						graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5.5}] [expr {10 + $Y_shift}] 0" "$AngleText_1dp"
					} elseif {$scaleCount == 5} {
						set scaled_angle 0.8
						set colour_value [expr {round(1000-$scaled_angle*1000)}]
						graphics $::bendix::surf_repID color $colour_value
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {11 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {13 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {13 + $Y_shift}] 0"
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {11 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {11 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {13 + $Y_shift}] 0"
						graphics $::bendix::surf_repID color gray
						graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5.5}] [expr {12 + $Y_shift}] 0" "$AngleText_1dp"
					} else {
						### Max
						graphics $::bendix::surf_repID color 33
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {13 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {15 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {15 + $Y_shift}] 0"
						graphics $::bendix::surf_repID triangle "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {13 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {13 + $Y_shift}] 0" "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5}] [expr {15 + $Y_shift}] 0"
						graphics $::bendix::surf_repID color gray
						graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +5.5}] [expr {14 + $Y_shift}] 0" "$AngleText_1dp"
					}
					set scaleCount [expr {$scaleCount +1}]
				}
				#### Heatmap colour scale title
				graphics $::bendix::surf_repID text "[expr {[lindex $::bendix::curvature_graph_X [expr {[llength $::bendix::curvature_graph_X] -1}]] +3}] [expr {16.5 + $Y_shift}] 0" "Heatmap colour scale"
			}
			set AllQualitativeThingsDrawn [graphics $::bendix::surf_repID list]
			set ::bendix::last_axis_drawn [lindex $AllQualitativeThingsDrawn [expr {[llength $AllQualitativeThingsDrawn] -1}]]
		} else {
			if {$::bendix::first_axis_drawn != ""&& $::bendix::last_axis_drawn != ""} {
				for {set kill $::bendix::first_axis_drawn} {$kill <= $::bendix::last_axis_drawn} {incr kill} {
					graphics $::bendix::surf_repID delete $kill 
				}
				set ::bendix::first_axis_drawn ""
				set ::bendix::last_axis_drawn ""
			}
			global edited_surf_viewpoint
			set edited_surf_viewpoint($::bendix::surf_repID) [molinfo $::bendix::surf_repID get {center_matrix rotate_matrix scale_matrix global_matrix}]
		}
		### Fix viewpoint
		global edited_surf_viewpoint
		
		if {$::bendix::surfed == 1} {
			set edited_surf_viewpoint($::bendix::surf_repID) [molinfo $::bendix::surf_repID get {center_matrix rotate_matrix scale_matrix global_matrix}]
		} else {
			if {[info exists edited_surf_viewpoint($::bendix::surf_repID)]} {
				molinfo $::bendix::surf_repID set {center_matrix rotate_matrix scale_matrix global_matrix} $edited_surf_viewpoint($::bendix::surf_repID)
			} else {
				display resetview
			}
		}
	}
  }

# ::bendix::create_graph -------------------------------------------------------
#    Stored angles per helix length (in residue numbers) are plotted
#    using Multiplot. Traces are drawn using VMD's in-built colour sequence
#    (see Graphical Representation's ColorIDs), to allow the user to deduce
#    the identity of a plotted helix by subsequently drawing the helix/ces
#    in auto-coloured, uniform colour.
#    Proc enabled by angle colouring. Returns if no angles are stored
#    or the Multiplot package can not be required. Multiplot plots are drawn
#    in a pop-up window that allows data export by default.
# ------------------------------------------------------------------------------

  proc ::bendix::create_graph {} {
	if {$::bendix::points_to_graph == ""} {
		::bendix::MissingAngles
	} else {
		if [catch {package require multiplot} msg] {
			catch {destroy .nonoMultiplot}
			toplevel .nonoMultiplot
			wm geometry .nonoMultiplot 370x250+100+150
			#grab .nonoMultiplot
			wm title .nonoMultiplot "No Multiplot!"
			message .nonoMultiplot.msg1 -width 350 \
				-text "Multiplot is not installed on your computer!\
						\n\nbendix uses the Multiplot plugin make 2D graphs.\n\
						Do you have the latest VMD version?" -pady 15 -padx 5
			button .nonoMultiplot.okb -text Return -background green -command {destroy .nonoMultiplot ; return 0}
			pack .nonoMultiplot.msg1 -side top 
			pack .nonoMultiplot.okb -side bottom -pady 5

		    #### Upon window close or minimisation, exit properly.
	    	#bind .nonoMultiplot <Unmap> {
			#	destroy .nonoMultiplot
			#	return 0
			#}
		}
		set title "Helix axis curvature along helix length"
		set xlab "Distance along the helix (approximate number of residues)"
		set ylab "Curvature (Degree)"
		set graph_colours "blue red gray orange yellow tan green magenta brown pink cyan purple black turquoise green4 red3 blue4 violet maroon red2 green3 blue2 red3 green2 blue3 white"; 
		set Xs {}
		set Ys {}
		set N_helices_over12 [llength $::bendix::points_to_graph]
		set spline_listAdded {}

		#### ID where separate helices start by normalising select_list_lastIndex's into sequential Ns, e.g. 6 10 5 --> 6 16 21
		for {set splineadd 0} {$splineadd < $N_helices_over12} {incr splineadd} {
			if {$splineadd == 0} {
		 		lappend spline_listAdded [lindex $::bendix::points_to_graph $splineadd]
			} else {
				set newTerm [expr {[lindex $::bendix::points_to_graph $splineadd] + [lindex $spline_listAdded [expr {$splineadd -1}]] + 1}]
				lappend spline_listAdded $newTerm
			}
		}
		set graph_colours "blue red gray orange yellow tan green magenta brown pink cyan purple black turquoise green4 red3 blue4 violet maroon red2 green3 blue2 red3 green2 blue3 white"; 
		set Xs {}
		set Ys {}
		set b 0
		set colorN 0
		set using_old_startstop_fixed 0
		set chain 1
		
		for {set c 0} {$c < [llength $::bendix::curvature_graph_X]} {incr c} {
			if {$N_helices_over12 == 1 || [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle 1]] == "" } {
			#### If only one helix, (or repeats across chains) supply correct AA Ns:
				if {[info exists startAA_if_no_other_exists(0)]} {
					### Repeat across chains:
					lappend Xs [expr [expr {[lindex $::bendix::curvature_graph_X $c] + $startAA_if_no_other_exists(0)}] -1]; # Fixed by -1 since X starts from 1 now, not 0.
				} else {
					lappend Xs [expr [expr {[lindex $::bendix::curvature_graph_X $c] + [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]]}] -1]; # Fixed by -1 since X starts from 1 now, not 0.
				}
				set xlab "Distance along the helix (approximate resid number)"
			} else {
				lappend Xs [lindex $::bendix::curvature_graph_X $c]
			}
			lappend Ys [lindex $::bendix::curvature_graph_Y $c]
			
			if {$c == [lindex $spline_listAdded $b] } {
			#### End of a helix: plot.
				if {$b == 0} {
					if {$::bendix::list_of_chains != "" && [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle 1]] == ""} {
						### Only print out the first chain if we're not at risk of only printing it for the first helix start and stop in several in a chain.
						set plothandle [multiplot -x $Xs -y $Ys -linecolor [lindex $graph_colours $colorN] -fillcolor [lindex $graph_colours $colorN] -marker point -title $title -xlabel $xlab -ylabel $ylab -plot -legend "resid [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]] to [lindex $::bendix::helix_assignment [expr [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]+1]] (chain [lindex $::bendix::list_of_chains 0])"]
					} else {
						set plothandle [multiplot -x $Xs -y $Ys -linecolor [lindex $graph_colours $colorN] -fillcolor [lindex $graph_colours $colorN] -marker point -title $title -xlabel $xlab -ylabel $ylab -plot -legend "resid [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]] to [lindex $::bendix::helix_assignment [expr [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]+1]]"]
					}
					set startAA_if_no_other_exists($b) [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]]
					set endAA_if_no_other_exists($b) [lindex $::bendix::helix_assignment [expr [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]+1]]
					set startstop_fixed 0
				} else {
					if {[lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]] == "" && $startAA_if_no_other_exists(0) != ""} {
						if {$using_old_startstop_fixed > $startstop_fixed} {
							set using_old_startstop_fixed 0
							incr chain
						}
						if {[lindex $::bendix::list_of_chains $chain] != ""} {
							$plothandle add $Xs $Ys -linecolor [lindex $graph_colours $colorN] -fillcolor [lindex $graph_colours $colorN] -marker point -plot -legend "resid $startAA_if_no_other_exists($using_old_startstop_fixed) to $endAA_if_no_other_exists($using_old_startstop_fixed) (chain [lindex $::bendix::list_of_chains $chain])"
						} else {
							$plothandle add $Xs $Ys -linecolor [lindex $graph_colours $colorN] -fillcolor [lindex $graph_colours $colorN] -marker point -plot -legend "resid $startAA_if_no_other_exists($using_old_startstop_fixed) to $endAA_if_no_other_exists($using_old_startstop_fixed)"
						}
						incr using_old_startstop_fixed
					} else {
						### If helix start- and stop residues exist, use and store these.
						$plothandle add $Xs $Ys -linecolor [lindex $graph_colours $colorN] -fillcolor [lindex $graph_colours $colorN] -marker point -plot -legend "resid [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]] to [lindex $::bendix::helix_assignment [expr [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]+1]]"
						set startAA_if_no_other_exists($b) [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]]
						set endAA_if_no_other_exists($b) [lindex $::bendix::helix_assignment [expr [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]+1]]
						incr startstop_fixed
					}
				}
				incr b
				incr colorN
				if {$colorN == 26} {
					set colorN 0
				}
				set Xs {}
				set Ys {}
			}
		}
		if {[info exists startAA_if_no_other_exists(0)]} {
			array unset array startAA_if_no_other_exists
			array unset endAA_if_no_other_exists
		}
	}
  }

# ::bendix::create_MaxAngle_graph ----------------------------------------------
#    The maximum angle per helix, per frame, is plotted using Multiplot.
#    Traces are drawn using VMD's in-built colour sequence
#    (see Graphical Representation's ColorIDs), to allow the user to deduce
#    the identity of a plotted helix by subsequently drawing the helix/ces
#    in auto-coloured, uniform colour. Proc enabled by angle colouring. 
#    Returns if no angles are stored, no frames exist or the Multiplot package
#    can not be required. Multiplot plots are drawn in a pop-up window
#    that allows data export by default.
# ------------------------------------------------------------------------------

  proc ::bendix::create_MaxAngle_graph {} {

	set frameSum 0
	foreach frameN $::bendix::vmdFrame {
		set frameSum [expr {$frameSum + $frameN}]
	}

	if [catch {package require multiplot} msg] {
		catch {destroy .noMultiplot}
		toplevel .noMultiplot
		wm geometry .noMultiplot 370x250+100+150
		#grab .noMultiplot
		wm title .noMultiplot "No Multiplot!"
		message .noMultiplot.msg1 -width 350 \
			-text "Multiplot is not installed on your computer!\n\nbendix uses the Multiplot plugin make 2D graphs.\
					\nDo you have the latest VMD version?" -pady 15 -padx 5
		button .noMultiplot.okb -text Return -background green -command {destroy .noMultiplot ; return 0}
		pack .noMultiplot.msg1 -side top 
		pack .noMultiplot.okb -side bottom -pady 5

	    #### Upon window close or minimisation, exit properly.
    	#bind .noMultiplot <Unmap> {
		#	destroy .noMultiplot
		#	return 0
		#}

	} elseif {$::bendix::points_to_graph == ""} {
		::bendix::MissingAngles

	} elseif {$::bendix::vmdFrame == "" || $frameSum == 0 || [llength $::bendix::vmdFrame] == 1 } {
		catch {destroy .noframes}
		toplevel .noframes
		wm geometry .noframes +100+150
		#grab .noframes
		wm title .noframes "No trajectory loaded?"
		message .noframes.msg1 -width 350 \
			-text "bendix does not detect frames!\nHave you loaded and (importantly!) run your trajectory using bendices? Also, is 'Store dynamic data' ticked under bendix' Analysis menu?\n\nIf bendix drawing is on, as you update a frame, bendices are drawn to the screen and their coordinates stored for analysis, along with the current frame number.\n\nPlease play your trajectory using bendices and ensure that 'Store dynamic data' is ticked, then retry Analysis." -pady 15 -padx 15
		button .noframes.okb -text Oops -background green -command {destroy .noframes ; return 0}
		pack .noframes.msg1 -side top 
		pack .noframes.okb -side bottom -pady 5

	    #### Upon window close or minimisation, exit properly.
    	#bind .noframes <Unmap> {
		#	destroy .noframes
		#	return 0
		#}

	} else {
		set title "Maximum angle per helix vs time"
		set xlab "Frame number"
		set ylab "Helix angle (Degree)"
		set graph_colours "blue red gray orange yellow tan green magenta brown pink cyan purple black turquoise green4 red3 blue4 violet maroon red2 green3 blue2 red3 green2 blue3 white"; 
		#### Setup to use the standard VMD colour sequence as far as Multiplot allows.
		set Xs {}
		set Ys {}
		set colorN 0
		set b 0
		set N_qualifying_helices [llength $::bendix::index_of_helix_assignment_used_for_angle]

		#### Per helix, jump through the MaxAngles-list (which is populated by Ns for all helices) in steps 
		#### equal to N helices, so I get only angles - and FrameNs - for the helix of interest. Then plot. 
		#### Do this for each helix in turn.
		
		for {set helices 1} {$helices <= $N_qualifying_helices} {incr helices} {
			for {set framesN [expr {$helices -1}]} {$framesN < [llength $::bendix::vmdFrame]} {incr framesN $N_qualifying_helices} {
				lappend Xs [lindex $::bendix::vmdFrame $framesN]
				lappend Ys [lindex $::bendix::maximumAngle_per_frame $framesN]
			}
			
			if {$helices == 1} {
				set plothandle [multiplot -x $Xs -y $Ys \
					-linecolor [lindex $graph_colours $colorN] \
					-fillcolor [lindex $graph_colours $colorN] \
					-marker point \
					-title $title \
					-xlabel $xlab \
					-ylabel $ylab \
					-plot \
					-legend "resid [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]] to [lindex $::bendix::helix_assignment [expr [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]+1]]"	]
					
					
			} else {
				$plothandle add $Xs $Ys \
					-linecolor [lindex $graph_colours $colorN] \
					-fillcolor [lindex $graph_colours $colorN] \
					-marker point \
					-plot \
					-legend "resid [lindex $::bendix::helix_assignment [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]] to [lindex $::bendix::helix_assignment [expr [lindex $::bendix::index_of_helix_assignment_used_for_angle $b]+1]]"
			}
			incr colorN
			incr b
			if {$colorN == 26} {
				set colorN 0
			}
			set Xs {}
			set Ys {}
		}
	}
  }
  
# ::bendix::angle --------------------------------------------------------------
#    Calculates the angle between two vectors.
#    Called by angle-calculating functions.
# ------------------------------------------------------------------------------

  proc ::bendix::angle {A B C} {
	global M_PI
	set BA [vecsub $A $B]
	set BC [vecsub $C $B]
	set cos_angle_between_in_radians [expr {[vecdot $BA $BC]/ ([veclength $BC]*[veclength $BA])}]
	if {$cos_angle_between_in_radians < -1} {
		set cos_angle_between_in_radians -1
	}
	set angle [expr {180 - acos($cos_angle_between_in_radians)*(180.0/$M_PI)}]
	return $angle
  }


################################################################################
#                     TRAJECTORY AND MOLECULE ID DETECTION                     #
################################################################################

# ::bendix::sscache ------------------------------------------------------------
#    Called whenever the frame changes, to update frame-relevant variables 
#    and call the ::bendix::mainDrawingLoop procedure. 
#    If the trajectory is re-run, graph variables are erased
#    to prevent overlapping graphics.
# ------------------------------------------------------------------------------

  proc ::bendix::sscache {name index op} {
	if {$::bendix::die == 0} {
		global vmd_frame
		set temp_curr_frame $vmd_frame($::bendix::proteinID);
		if {$::bendix::frame > $temp_curr_frame} {
		#### Re-running a trajectory erases any previous data.
			set ::bendix::points_to_graph ""
			set ::bendix::curvature_graph_X ""
			set ::bendix::curvature_graph_Y ""
			set ::bendix::maximumAngle_per_frame ""
			set ::bendix::vmdFrame ""
			set ::bendix::angle_per_turn_per_frame {}
			set ::bendix::angle_per_AA_per_frame_BARRED ""
			set ::bendix::helix_axis_per_helix_per_frame_BARRED ""
		}
		set ::bendix::frame $vmd_frame($::bendix::proteinID);
		::bendix::mainDrawingLoop
		set ::bendix::frame_done_before 0
	} elseif {$::bendix::die_backbone == 0} {
		#### Loop to enable Backbone drawing by trajectory, if only backbone is wanted.
		global vmd_frame
		set ::bendix::frame $vmd_frame($::bendix::proteinID);
		::bendix::drawBackbone
		set ::bendix::frame_done_before 0
	} elseif {$::bendix::frame_done_before == 0} {
		global vmd_frame
		set ::bendix::previous_frame $vmd_frame($::bendix::proteinID)
		set ::bendix::frame_done_before 1
	}
  }

# ::bendix::set_molIDs ---------------------------------------------------------
#    Called whenever a molecule is loaded or deleted,
#    to update the molecule ID spinbox and erase graphics for deleted molecules.
# ------------------------------------------------------------------------------

  proc ::bendix::set_molIDs {} {
	set N_loaded_molecules [molinfo num]
	if {$N_loaded_molecules == 0} {
		.bex.lf1.field1.molname configure -values "";
		# Ensure that the molID field is cleared.
		.bex.lf1.field1.molname set ""; 
		# Reset non-bendices' tickboxes
		set ::bendix::autoCartoon 0
		set ::bendix::autoNewCartoon 0
		set ::bendix::autoTube 0
		set ::bendix::quick_and_dirty 0
		set ::bendix::slow_and_pretty 0
		set ::bendix::rep_zero 1
		# Disable Rep0 button
		.bex.lf3.field1.rep0 configure -state disable; 
	} else {
		# Enable Rep0 button
		.bex.lf3.field1.rep0 configure -state normal; 
		set pre_exists 0
		set temp_proteinID [.bex.lf1.field1.molname get]
		set ID_mols [molinfo list];
		set N_mols ""
		for {set molN 0} {$molN < $N_loaded_molecules} {incr molN } {
			if {[lindex $ID_mols $molN] == $temp_proteinID} {
				set pre_exists 1
			}
			lappend N_mols [lindex $ID_mols $molN]
		}
		# Add new molecule IDs to the spinbox
		.bex.lf1.field1.molname configure -values $N_mols
		
		#### If the proteinID that the user last worked on still exists, set the molID field to this.
		if {$pre_exists == 1} {
			.bex.lf1.field1.molname set $temp_proteinID
		} else {
			set ::bendix::helix_assignment ""
		}
	}
	
	#### Erase graphics for deleted molcules, where present.
	set whole_array_element_names [array names vmd_molecule];
	# list of array elements (unordered). Deleted molecules' elements are kept with value 0.
	for {set count 0} {$count <= [expr {[llength $whole_array_element_names] -1}]} {incr count} {
		set molID [lindex $whole_array_element_names $count]
		if {$vmd_molecule($molID) == 0} {
		#### Callback detected a deleted molecule. Its bendix Reps are therefore deleted, too, where available.
			if { [lindex $::bendix::CG_old_sheet_rep_list $molID] != "" } {
				mol delrep [lindex $::bendix::CG_old_sheet_rep_list $molID] $molID
				lset ::bendix::CG_old_sheet_rep_list $molID ""
			}
			if { [lindex $::bendix::CG_old_rep_list $molID] != "" } {
				mol delrep [lindex $::bendix::CG_old_rep_list $molID] $molID
				lset ::bendix::CG_old_rep_list $molID ""
			}
			if { [lindex $::bendix::First_backbone_drawn_list $molID] != "" && [lindex $::bendix::Last_backbone_drawn_list $molID] != ""} {
				graphics $molID delete all
				lset ::bendix::First_backbone_drawn_list $molID ""
				lset ::bendix::Last_backbone_drawn_list $molID ""
			}
			lset ::bendix::CG_cartoonified_list $molID 0
		}
	}
  }

################################################################################
#                       PARTICLE AND SUBSET VALIDATION                         #
################################################################################

# ::bendix::ParticleSubsetOK ---------------------------------------------------
#    Check that a particle type is chosen and if not, uses default.
#    Edits the particle type(s) to be compatible with atomselection.
#    Where a subset is chosen, tests that the particle type(s) 
#    generate resids, and if not, alters particle types or,
#    if out of available particle types, asks the user for another subset.
# ------------------------------------------------------------------------------

  proc ::bendix::ParticleSubsetOK {} {
	#### Test for validity of particle choice 
	## remove spaces, tabs et.c.
	set trimmed_particleName [string trim $::bendix::particle_name]; 
	
	## If no particle names, set them to default.
	if {$trimmed_particleName == ""} {
		if {$::bendix::CG == 1} {
			set ::bendix::particle_name "CA B.*"
			set trimmed_particleName "CA B.*"
		} else {
			set ::bendix::particle_name "CA"
			set trimmed_particleName "CA"
		}
		set ::bendix::custom_particle 0						
	} 
						
	## Re-list the names without unnecessary spaces
	set split_particle_names [split $trimmed_particleName ""]; # split at each char, whether space, non-space or ".
	foreach particleCharacter $split_particle_names {
		set matchedBadChar [string match "\"" $particleCharacter]
		if {$matchedBadChar != 1} {
			if {$particleCharacter != " "} {
				append fixed_particleName $particleCharacter
			} elseif {$fixed_particleName != ""} {
				append fixed_particleNames "\"$fixed_particleName\" "
				set fixed_particleName ""
			}
		}
	}
	set lost_last_name [string trim $fixed_particleName]
	if {$lost_last_name != ""} {
		append fixed_particleNames "\"$fixed_particleName\""
	}
	if {$fixed_particleNames == ""} {
		set fixed_particleNames "\"$trimmed_particleName\""
	} else {
		## Trim away trailing space
		set fixed_particleNames [string trim $fixed_particleNames]
	}
	
	## Check if not-default particlenames select anything
	if {$fixed_particleNames != "\"CA\"" && $fixed_particleNames != "\"CA\" \"B.*\"" && $fixed_particleNames != "\"B.*\" \"CA\""} {
		set ::bendix::custom_particle 1
	} else {
		set ::bendix::custom_particle 0
	}
		
	### Test the atom type to see if it selects anything:	
	set testing_particle [atomselect $::bendix::proteinID "name $fixed_particleNames"]
	set select_AAs [$testing_particle get resid];
	$testing_particle delete
	
	if {$select_AAs == ""} {
		## If nothing selected in the entire system, force default particle names
		if {$::bendix::custom_particle == 1} {
			if {$::bendix::CG == 1} {
				set ::bendix::particle_name "CA B.*"
				set fixed_particleNames "\"CA\" \"B.*\""
				set testing_particle [atomselect $::bendix::proteinID "name $fixed_particleNames"]
				set select_AAs [$testing_particle get resid];
				$testing_particle delete
				
				## CG checks for both AT and CG names (CA and B.*), so if nil found, AT won't find anything either.
				if {$select_AAs == ""} {
					set ::bendix::fixed_particleNames ""
					::bendix::unknown_custom_particle_type
					return
					error "Use another backbone particle."
				} else {
					set ::bendix::custom_particle 0
				}
			} else {
				### AT
				set ::bendix::particle_name "CA"
				set fixed_particleNames "\"CA\""
				set testing_particle [atomselect $::bendix::proteinID "name $fixed_particleNames"]
				set select_AAs [$testing_particle get resid];
				$testing_particle delete
				if {$select_AAs == ""} {
					
					### Change AT to CG, test that default particle type.
					set ::bendix::CG 1
					::bendix::MakeCG
					set ::bendix::particle_name "CA B.*"
					set fixed_particleNames "\"CA\" \"B.*\""
					set testing_particle [atomselect $::bendix::proteinID "name $fixed_particleNames"]
					set select_AAs [$testing_particle get resid];
					$testing_particle delete
					
					## An initial AT input requires a check with CG particles, since they're not covered by CA. If still nil, throw error:
					if {$select_AAs == ""} {
						set ::bendix::fixed_particleNames ""
						::bendix::unknown_custom_particle_type
						return
						error "Use another backbone particle."
					} else {
						set ::bendix::custom_particle 0
					}
				} else {
					set ::bendix::custom_particle 0
				}
			}
		### Default particle were used. If AT, test the alternate particle type, else throw error.
		} elseif {$::bendix::CG == 1} {
			## CG checks for both AT and CG names (CA and B.*), so if nil found, AT won't find stuff either.
			set ::bendix::fixed_particleNames ""
			::bendix::unknown_particle_type
			return
			error "Neither AT or CG particles generate a selection. Consider another backbone particle name."

		} else {
			###  An initial AT input requires a check with CG particles, since they're not covered by CA. 
			set ::bendix::CG 1
			::bendix::MakeCG
			set ::bendix::particle_name "CA B.*"
			set fixed_particleNames "\"CA\" \"B.*\""
			set testing_particle [atomselect $::bendix::proteinID "name $fixed_particleNames"]
			set select_AAs [$testing_particle get resid];
			$testing_particle delete
			
			## An initial AT input requires a check with CG particles, since they're not covered by CA. I still nil, throw error:
			if {$select_AAs == ""} {
				set ::bendix::fixed_particleNames ""
				::bendix::unknown_particle_type
				return
				error "Use another backbone particle."
			} else {
				set ::bendix::custom_particle 0
			}
		}
	}		
		
	#### Test for validity of subset
	# Check that the chosen subset and particle type results in a atom selection. 
	# If not, change particle type (AT <--> CG) and retry. 
	# If still no atoms selected, Return and notify the user.

	if {$::bendix::subset != ""} {
		set testing_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $fixed_particleNames )"]
		set select_AAs [$testing_backbone get resid];
		#set select_AAs [$testing_backbone get residue]; # Resid:s repeat across chains whereas residue doesn't. However Residue becomes problematic when treating both CG and AT.
		$testing_backbone delete

		## If nothing is atomselected by the currect particletype and subset:
		## change the particle type, rerun mainDrawingLoop and, when no selection is Still made,
		## print errormessage to screen, asking the user to change the subset and/or particle type.
		if {$select_AAs == ""} {
			
			## If custom particles were chosen, try default CG or AT particles and rerun.
			if {$::bendix::custom_particle == 1} {
				if {$::bendix::CG == 1} {
					set ::bendix::particle_name "CA B.*"
				} else {
					set ::bendix::particle_name "CA"
				}
				::bendix::mainDrawingLoop
			
			## If default particles, try the alternate particle type (CG or AT)	
			} elseif {$::bendix::tested_alt_particle_for_empty_select_AAs == 0 || $::bendix::autolooper == 0 } {
				#### The subset tested does not contain any AT particles. Test with CG:
				#### Un/tick the box to try the alternate particle type and rerun the ::mainDrawingLoop
				set ::bendix::CG [expr {1 - $::bendix::CG}]
				::bendix::MakeCG
				set ::bendix::tested_alt_particle_for_empty_select_AAs 1
				set ::bendix::autolooper 1
				::bendix::mainDrawingLoop
			} else {
				#### Ensure that the window does not already exist (which causes problems)
				catch {destroy .incorrect_subset}
				toplevel .incorrect_subset
				wm geometry .incorrect_subset +100+250
				#grab .incorrect_subset
				wm title .incorrect_subset "Nothing in subset"
				message .incorrect_subset.msg -width 300 \
					-text "bendix can not identify either custom, atomistic or coarse-grained particles in the subset that you have selected.\
\nPlease change your subset before continuing." -pady 15 -padx 25
				button .incorrect_subset.ok -text OK -background green -command {destroy .incorrect_subset ; return }
				pack .incorrect_subset.msg -side top
				pack .incorrect_subset.ok -side bottom -pady 5
				set ::bendix::die 1
				set ::bendix::die_backbone 1

			    #### Upon window close or minimisation, exit properly.
			   	#bind .incorrect_subset <Unmap> {
				#	destroy .incorrect_subset
				#	return
				#}
				#return					
			}
		} else {
			set ::bendix::fixed_particleNames $fixed_particleNames
		}
	} else {
		### No subset --> immediate variable setting.
		set ::bendix::fixed_particleNames $fixed_particleNames
	}; # End of subset test-loop
  }

################################################################################
#                                ERROR MESSAGES                                #
################################################################################

# ::bendix::MissingAngles ------------------------------------------------------
#    Pop-up that twarts attempts to graph angles along helices
#    when no angles are stored, and informs the user how to store angles.
# ------------------------------------------------------------------------------

  proc ::bendix::MissingAngles {} {
	catch {destroy .messpopMissingAngles}
	toplevel .messpopMissingAngles
	wm geometry .messpopMissingAngles +150+200
	#grab .messpopMissingAngles
	wm title .messpopMissingAngles "No angles to graph"
	message .messpopMissingAngles.msg1 -width 300 -text "bendix does not detect anything to graph.\
\n\nDid you create curvature-coloured helices\nbefore trying to graph them?\nAlso, if you wish to analyse time-varying data, ensure that 'Store dynamic data' is ticked (you find it under bendix' Analysis menu)." -pady 15 -padx 15
	button .messpopMissingAngles.ok -text OK -background green -command {destroy .messpopMissingAngles ; return 0}
	pack .messpopMissingAngles.msg1 -side top 
	pack .messpopMissingAngles.ok -side bottom -pady 5

    #### Upon window close or minimisation, exit properly.
   	#bind .messpopMissingAngles <Unmap> {
	#	destroy .messpopMissingAngles
	#	return 0
	#}
  }

# ::bendix::incorrect_residue_Ns -----------------------------------------------
#    Called when Bendix detects invalid user-assigned resid numbers.
# ------------------------------------------------------------------------------
  proc ::bendix::incorrect_residue_Ns {} {
	catch {destroy .incorrectAA_N}
	toplevel .incorrectAA_N
	wm geometry .incorrectAA_N +100+250
	wm protocol .incorrectAA_N WM_DELETE_WINDOW {
		destroy .incorrectAA_N
		return
	}
	
#	grab .incorrectAA_N
	global start_and_end_AA_N_of_chains
	wm title .incorrectAA_N "Incorrect resid numbers"
	message .incorrectAA_N.msg -width 420 \
		-text "Your protein (or your selected subset) has chain(s)\n\
$::bendix::list_of_chains \nthat start and end on the following resid numbers:\n$start_and_end_AA_N_of_chains \
\n\nPlease ensure that you assign helices within these boundaries,\nand that you use only numeric input.\nFor example: 1 10 refers to a helix between resid 1 and resid 10." -pady 15 -padx 15 -justify center
	button .incorrectAA_N.ok -text OK -background green \
		-command {
			destroy .incorrectAA_N
			return
		}
	set ::bendix::die 1; 	# If a movie is playing, cease drawing. Drawing would cause as many Error messages as there are updated Frame numbers (lots), 
							# which would stop the movie playing and generally annoy the user.
	set ::bendix::die_backbone 1;
	
	pack .incorrectAA_N.msg -side top
	pack .incorrectAA_N.ok -side bottom -pady 5
	#vwait buttonreturn;  # Unless set, this prod will cause VMD to halt and be un-closable.

    #### Upon window close or minimisation, exit properly.
   	#bind .incorrectAA_N <Unmap> {
	#	destroy .incorrectAA_N
	#	return
	#}
	return
  }

# ::bendix::no_helicity --------------------------------------------------------
#    Called when Bendix detects no helicity,
#    to explain why no drawing occurred and suggest solutions.
# ------------------------------------------------------------------------------

  proc ::bendix::no_helicity {} {
	#### Ensure that the window does not already exist (which causes problems)
	catch {destroy .messpop}
	toplevel .messpop
	wm geometry .messpop +100+150
	#grab .messpop
	wm title .messpop "No helices?"
	message .messpop.msg1 -width 300 \
		-text "bendix could not detect helicity in your protein.\nIt tried to treat your protein as both atomistic\
and coarse-grained, but to no avail.\n\nHelix-detection may fail if:\n\n\
A) STRIDE (if atomistic) or DSSP (if MARTINI coarse grained) does not assign helicity in your protein\n\n\
B) Your protein consists of particles that can not be assigned a secondary structure automatically.\n\n\
To assign helicity yourself, please enter helix start and stop resid numbers in the Helices field.\n\
To view just the protein backbone, please select an option under 'How to draw non-Bendices'." \
		-pady 15 -padx 5
	button .messpop.okb -text OK -background green -command {destroy .messpop ; return 0} 
	pack .messpop.msg1 -side top 
	pack .messpop.okb -side bottom -pady 5
	set ::bendix::die 1
  }

# ::bendix::unknown_particle_type ----------------------------------------------
#    Called when Bendix detects neither AT nor CG protein in proteinID,
#    to explain why no drawing occurred.
# ------------------------------------------------------------------------------

  proc ::bendix::unknown_particle_type {} {
	catch {destroy .weird_particle}
	toplevel .weird_particle
	wm geometry .weird_particle +100+250
	#grab .weird_particle
	wm title .weird_particle "Unknown particle type!"
	message .weird_particle.msg -width 300 \
		-text "Bendix tried both atomistic PDB\
and coarse-grained particles, but neither of these selected anything in your protein. Consider using an alternative particle."\
		-pady 15 -padx 15 -justify "center"
	button .weird_particle.ok -text OK -background green -command {destroy .weird_particle ; return 0}
	pack .weird_particle.msg -side top
	pack .weird_particle.ok -side bottom -pady 5
	set ::bendix::die 1
	set ::bendix::die_backbone 1

  }
  
# ::bendix::unknown_custom_particle_type ----------------------------------------------
#    Called when Bendix detects neither custom, AT nor CG protein in proteinID,
#    to explain why no drawing occurred.
# ------------------------------------------------------------------------------

  proc ::bendix::unknown_custom_particle_type {} {
	catch {destroy .weird_custom_particle}
	toplevel .weird_custom_particle
	wm geometry .weird_custom_particle +100+250
	#grab .weird_custom_particle
	wm title .weird_custom_particle "Unknown particle type!"
	message .weird_custom_particle.msg -width 300 \
		-text "Bendix tried your chosen custom particle type, atomistic PDB\
and coarse-grained particles, but neither of these selected anything in your protein. Consider using an alternative particle."\
		-pady 15 -padx 15 -justify "center"
	button .weird_custom_particle.ok -text OK -background green -command {destroy .weird_custom_particle ; return 0}
	pack .weird_custom_particle.msg -side top
	pack .weird_custom_particle.ok -side bottom -pady 5
	set ::bendix::die 1
	set ::bendix::die_backbone 1

  }

# ::bendix::same_chain_names ---------------------------------------------------
#    Called when Bendix detects two different chains with the same name.
#    This means that there is no distinct resid identifier,
#    so bendix asks you to alter the input file.
# ------------------------------------------------------------------------------

  proc ::bendix::same_chain_names {} {
	catch {destroy .same_chain}
	toplevel .same_chain
	wm geometry .same_chain +100+250
	#grab .same_chain
	wm title .same_chain "Same chain, different resid."
	message .same_chain.msg -width 350 \
		-text "Bendix just detected two different chains,\nbut they have the same chain name!\n\
Please change the name of one chain\nand reload the file to continue."\
		-pady 15 -padx 15 -justify "center"
	button .same_chain.ok -text OK -background green -command {destroy .same_chain ; return 0}
	pack .same_chain.msg -side top -padx 37
	pack .same_chain.ok -side bottom -pady 5
	set ::bendix::die 1
	set ::bendix::die_backbone 1

  }


################################################################################
#                             HIDE, ERASE AND RESET                            #
################################################################################

# ::bendix::hiderep_zero -------------------------------------------------------
#    Dis/enable the zeroth representation,
#    depending on the user's GUI settings
# ------------------------------------------------------------------------------

  proc ::bendix::hiderep_zero {} {
	#### Update the proteinID to contain the current molecule
	set ::bendix::proteinID [.bex.lf1.field1.molname get]
	if {$::bendix::proteinID != ""} {
		if {$::bendix::rep_zero == 0} {
			mol showrep $::bendix::proteinID 0 off
		} else {
			mol showrep $::bendix::proteinID 0 on
		}
	}
  }

# ::bendix::erase --------------------------------------------------------------
#    Erases all current MolID's graphics primitives 
#    and resets relevant variables, including GUI variables.
# ------------------------------------------------------------------------------

  proc ::bendix::erase {} {
	#### Check that a molecule is present. If not, notify the user.
	set N_loaded_Mols [molinfo num]
	if {$N_loaded_Mols == 0} {
		# Ensure that the molID field is cleared.
		.bex.lf1.field1.molname set ""; 
		catch {destroy .pop_noerase}
		toplevel .pop_noerase
		wm geometry .pop_noerase +200+300
		#grab .pop_noerase
		wm title .pop_noerase "No molecule?"
		
		message .pop_noerase.msg1 -width 400 -text "Nothing to erase!" -pady 15 -padx 35
		message .pop_noerase.msg2 -width 400 -text "Please load your molecule,\nthen use bendix." -padx 35
		button .pop_noerase.okb -text OK -background green -command {destroy .pop_noerase ; return 0}
		
		pack .pop_noerase.msg1 .pop_noerase.msg2 -side top
		pack .pop_noerase.okb -side bottom -pady 5

	    #### Upon window close or minimisation), exit properly.
	   	#bind .pop_noerase <Unmap> {
		#	destroy .pop_noerase
		#	return 0
		#}
		#return

	} else {
		#### Update the proteinID to contain the current molecule
		set ::bendix::proteinID [.bex.lf1.field1.molname get]

		#### CG graphics
		graphics $::bendix::proteinID delete all
	}
		#### At any point during a slow movie, I may Erase all and stop bendix from slowing it down by drawing every frame.
	set ::bendix::die 1; 
	set ::bendix::die_backbone 1;
	
	if {$::bendix::CG == 0} {
		set ::bendix::autoCartoon 0
		set ::bendix::autoNewCartoon 0
		set ::bendix::autoTube 0
		::bendix::cartoonify
		#### Ensure that any AT non-bendix reps are deleted, also.
	}
	
	if {$::bendix::quick_and_dirty == 1 || $::bendix::slow_and_pretty == 1 } {
		set ::bendix::quick_and_dirty 0
		set ::bendix::slow_and_pretty 0
		# Clean up Draft and Join Representations.
		::bendix::drawBackbone;
	}
	set ::bendix::AThelix_string ""
	set ::bendix::points_to_graph ""
	set ::bendix::curvature_graph_X ""
	set ::bendix::curvature_graph_Y ""
	set ::bendix::maximumAngle_per_frame ""
	set ::bendix::vmdFrame ""
	set ::bendix::angle_per_turn_per_frame {}
	set ::bendix::angle_per_AA_per_frame_BARRED ""
	set ::bendix::helix_axis_per_helix_per_frame_BARRED ""
	set ::bendix::first_surf_drawn "";
	set ::bendix::first_axis_drawn "";
	set ::bendix::last_axis_drawn "";
	set ::bendix::axesON 0;
	set ::bendix::surfed 0
	set ::bendix::index_of_helix_assignment_used_for_angle "";
	if {$::bendix::helix_assignment == ""} {
		set ::bendix::String_for_cartoonify ""; #### Testing..
	}
  }

# ::bendix::quit ---------------------------------------------------------------
#    Erase any drawn graphics primitives for all MolIDs
#    and reset all Bendix settings to original values.
#    Reset the protein viewpoint to the original when Bendix was first loaded.
# ------------------------------------------------------------------------------

  proc ::bendix::quit {} {

	global vmd_quit
	global vmd_frame

	#### Loop to ensure that Bendix-popups are closed and, in the case of the surf module, reset, prior to leaving the main bendix. 
	#### Catch ensures no error messages, which otherwise disrupts the script.
	catch {
		destroy .pop_whatis 
		destroy .pop_howto
		destroy .pop_whodid
		destroy .incorrectAA_N
		::bendix::erase_surf
		#### Unset any memory of past surf reorientations
		global edited_surf_viewpoint
		if {$::bendix::surf_repID != ""} {
			if {[info exists edited_surf_viewpoint([expr $::bendix::surf_repID +1])]} {
				unset edited_surf_viewpoint([expr $::bendix::surf_repID +1])
			}
		}
		### Show hidden protein(s)
		set all_existing_molIDs [molinfo list]
		set target_proteinID_exists 0
		foreach molID $all_existing_molIDs {
			mol on $molID;
			if {$molID == $::bendix::proteinID} {
				set target_proteinID_exists 1
			}
		}
		#### Return to original protein viewpoint (necessary if you're coming from Surf)
		global protein_viewpoints
		if {$target_proteinID_exists == 1} {
			molinfo $::bendix::proteinID set {center_matrix rotate_matrix scale_matrix global_matrix} $protein_viewpoints($::bendix::proteinID)
		} else {
			display resetview
		}
		
		destroy .pop_surf
	}
			
	#### Delete all surplus graphics, per molecule ID
	set all_existing_molIDs [molinfo list]
	foreach molID $all_existing_molIDs {
		#### Erases all graphics primitives
		graphics $molID delete all
		#### Erases all CG Draft Reps (sheets and backbone)
		if { [lindex $::bendix::CG_old_sheet_rep_list $molID] != "" } {
			mol delrep [lindex $::bendix::CG_old_sheet_rep_list $molID] $molID
		}
		if { [lindex $::bendix::CG_old_rep_list $molID] != "" } {
			mol delrep [lindex $::bendix::CG_old_rep_list $molID] $molID
		}
		#### Erases all AT Reps
		if { [lindex $::bendix::old_rep_list $molID] != "" } {
			mol delrep [lindex $::bendix::old_rep_list $molID] $molID
		}

		#### Return the molecule (but not rep0), in case bendix' Surf module removed this: 
		#mol showrep $molID 0 on;
		mol on $molID;
	}

	#### Delete CG backbone graphics arrays
	foreach list_counter {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49} {
		foreach list_name {::bendix::First_backbone_drawn_list ::bendix::Last_backbone_drawn_list ::bendix::CG_old_sheet_rep_list ::bendix::CG_old_rep_list ::bendix::old_rep_list} {
			lset $list_name $list_counter ""
		}
		foreach cartoonified_list {::bendix::CG_cartoonified_list ::bendix::cartoonified_list} {
			lset $cartoonified_list $list_counter "0"
		}
	}

	#### Reset GUI Display to non-Advanced
	pack forget .bex.lf3
	pack forget .bex.lf1
	pack forget .bex.lf2.field1
	pack forget .bex.lf2.field7
	pack forget .bex.lf2.field8
	pack forget .bex.lf2.msg
	pack forget .bex.lf2.listbox
	pack forget .bex.lf2.spacer -side left
	pack forget .bex.lf2.molname2 -side right
	pack forget .bex.lf2.msg3 -side right
	pack forget .bex.lf2.molname -side right
	pack forget .bex.lf2.msg2 -side right
	pack .bex.lf3
	.bex.field2.settings configure -text "Settings"
	set ::bendix::packed 0

	#### Reset GUI values
	.bex.lf2.listbox set "";
	.bex.lf2.field5.listbox set "";
	.bex.lf2.field1.cp set 4; # This is TurnResolution.

	#### Reset Display viewpoint and variables if a molecule is loaded
	set N_loaded_Mols [molinfo num]
	if {$N_loaded_Mols >= 1} {	
		#### Resets bendix arrays and variables
		::bendix::erase
		#### Reset Display viewpoint -- Depreciated since it disorients, rather than helps, the user.
		#global original_protein_viewpoint
		#if {[info exists original_protein_viewpoint]} {
		#	molinfo $::bendix::proteinID set {center_matrix rotate_matrix scale_matrix global_matrix} $original_protein_viewpoint
		#}
	}

	#### Reset variables to Original:
	set ::bendix::proteinID "0";
	set ::bendix::subset "";
	set ::bendix::helix_radius "2.2";
	set ::bendix::angle_max "20.0";
	set ::bendix::uniform_colour 1 ;
	set ::bendix::uniform_colour_type 2;
	set ::bendix::input_colour "red";
	set ::bendix::curvature_graph_X "";
	set ::bendix::curvature_graph_Y "";
	set ::bendix::points_to_graph "";
	set ::bendix::spline_resolution "4";
	set ::bendix::previous_TurnResolution 4;
	set ::bendix::TurnResolution 4;
	set ::bendix::AngleSide 3.6;
	set ::bendix::frame 0;
	set ::bendix::previous_frame 0;
	set ::bendix::rep_zero 1;
	set ::bendix::autoColour 1;
	set ::bendix::autoCartoon 0;
	set ::bendix::autoNewCartoon 0;
	set ::bendix::normalised_startsandstops "";
	set ::bendix::list_of_chains "";
	set ::bendix::xyz "";
	set ::bendix::helix_assignment "";
	set ::bendix::helix_assignment_by_backbone "";
	set ::bendix::startIndex "";
	set ::bendix::endIndex "";
	set ::bendix::spline_coords "";
	set ::bendix::spline_startsandstops "";
	set ::bendix::residueIndex "";
	set ::bendix::backbone_radius "0.2";
	set ::bendix::helix_type 1;
	set ::bendix::CG 0;
	set ::bendix::AThelix_string "";
	set ::bendix::String_for_cartoonify "";
	set ::bendix::CGbackbone 0;
	set ::bendix::maximumAngle_per_frame "";
	set ::bendix::vmdFrame "";
	set ::bendix::helixNumber 0;
	set ::bendix::MartiniBackbone "";
	set ::bendix::MartiniHelix "";
	set ::bendix::MartiniSheet "";
	set ::bendix::MartiniBackboneNew "";
	set ::bendix::MartiniHelixNew "";
	set ::bendix::MartiniSheetNew "";
	set ::bendix::previous_helix_type "";
	set ::bendix::CG_Backbone_indexNs "";
	set ::bendix::CG_Backbone_normalised_startsandstops "";
	set ::bendix::quick_and_dirty 0;
	set ::bendix::slow_and_pretty 0;
	set ::bendix::helix_coord_at_startNstop "";
	set ::bendix::helix_indices_at_startNstop "";
	set ::bendix::heres_a_helix "";
	set ::bendix::angle_per_turn_per_frame {}
	set ::bendix::angle_per_AA_per_frame_BARRED ""
	set ::bendix::helix_axis_per_helix_per_frame_BARRED ""
	set ::bendix::AngleAutoScale 1;
	set ::bendix::3D_surf 0;
	set ::bendix::z_squeeze 3.0;
	set ::bendix::frame_squeeze 3.0;
	set ::bendix::subset_by_backbone "";
	set ::bendix::proteinID_by_backbone "";
	set ::bendix::die 1;
	set ::bendix::die_backbone 1;
	set ::bendix::frame_done_before 0;
	set ::bendix::first_surf_drawn "";
	set ::bendix::first_axis_drawn "";
	set ::bendix::last_axis_drawn "";
	set ::bendix::axesON	0;
	set ::bendix::surfed 0;
	set ::bendix::previous_proteinID "0";
	set ::bendix::previous_subset "";
	set ::bendix::previous_helix_assignment "";
	set ::bendix::previous_spline_resolution "4";
	set ::bendix::tested_alt_particle_for_empty_select_AAs 0
	set ::bendix::tested_alt_particle_for_empty_start_and_end_AA_N_of_chains 0
	set ::bendix::tested_alt_particle_for_inexistent_start_and_end_AA_N_of_chains 0
	set ::bendix::tested_alt_particle_for_empty_residueNs 0
	set ::bendix::tested_alt_particle_for_empty_startIndex 0
	set ::bendix::autolooper 0
	set ::bendix::index_of_helix_assignment_used_for_angle "";
	set ::bendix::StoreData 0;
	set ::bendix::StoreAxes 0;
	array unset ::bendix::AAN_associated_w_splineIndex
	array unset ::Resolution_wrt_ends
	set ::bendix::realAAs_for_23D_graph_Xaxis "";
	set ::bendix::custom_particle 0
	set ::bendix::previous_particle_name ""
	set ::bendix::particle_name "CA"
	set ::bendix::fixed_particleNames ""

	#### Alter GUI setting to display the original setup
	::bendix::Hello_uniform
	::bendix::MakeCG
	::bendix::bendices
	::bendix::autoON

	#trace remove variable vmd_quit write "::bendix::quit"

	wm withdraw .bex
  }
  
  
################################################################################
#                             MAIN BENDIX FUNCTION                             #
################################################################################

# ::bendix::mainDrawingLoop ----------------------------------------------------
#
#                             Main Bendix Function
#
#    This large proc can be broken down into parts.
#    These parts are described below. 
#    Additional comments are provided in the code for guidance.
#
#
#                ~~~~~~ Initialisation and input control ~~~~~~
#
#    Reset variables. Ensure that GUI field entries are in integer
#    or floating point form, as necessary. Ensure that the chosen particle type
#    selects indices in the protein subset, otherwise notify the user, 
#    suggest solutions and Return. 
#
#
#    ~~~~ Assign helices, validate this choice and retrieve coordinates: ~~~~
#
#    Where helix assignment exists, check that these are an even number
#    (that each helix start is followed by an end), otherwise notify the user,
#    suggest solutions and Return. Store any user-provided chain starts
#    and stops (a larger residue number followed by a smaller residue number).
#    Control that the starts and ends of the subset protein,
#    divided into starts and stops of individual chains where available,
#    are equal to or extend beyond the	user's helix assignment
#    and that this helix assignment gives rise to index selection. 
#    If not, notify the user of available chains and their start and end
#    residue numbers, or about alternative particle type choices (AT or CG),
#    as necessary.
#
#    Once it has been established that the helix assignment is valid,
#    a REGEX string is made that reflects the user's helix assignment,
#    by chain where available. Normalised numbers are assigned that describe
#    where helices start and end. Helix assignment validation
#    and REGEX interpretation is followed by helix coordinate retrieval.
#    The REGEX string is used to retrieve helices' residues. Particle type
#    and successive residue numbers are matched to a single index,
#    wherefrom coordinates are derived.
#
#    Where helix assignment does not exist, create a REGEX string
#    that denotes the subset's helicity. This is done by using VMD's in-built
#    STRIDE helix assignment for atomistic proteins, and by selecting MARTINI's
#    helix-participating coarse-grained particle types. The REGEX string is used
#    to retrieve helices' residues. Particle type and successive residue numbers
#    are matched to a single index, wherefrom coordinates are derived. 
#
#    This loop is modulated: if the frame has changed and relevant variables 
#    (i.e. resid indices, REGEX expressions) are pre-stored,
#    said indices are used to derive new helix coordinates without re-assigning
#    or checking validity.
#
#
#       ~~~~~~ Draw straight helices, avoiding the spline: ~~~~~~
#
#    If straight helices are requested, identify helix length and,
#    if enough data points exist for vector algebraic treatment
#    to get axis points, retrieve first and last axis point coordinates.
#    Extend these points peripherally to reflect the helix length 'lost'
#    to axis averaging and store these coordinates to meet with CG Join backbone.
#    Helix colouring is done according to user GUI choices,
#    where colouring by helix angle is disabled.
#
#    Once helix position and colour is calculated, graphics primitives
#    - cylinders ended by spheres, of radius $::bendix::helix_radius -
#    are drawn to screen and a variable is updated to denote the previous choice
#    to represent helices as straight cylinders.
#    Helix length <= 3 are depicted by a sphere on average or choice coordinates.
#
#
#	   ~~~~~~ Generate spline representation of bendices: ~~~~~~
#
#    Reset variables. 
#    Retrieve the frequency that the user wishes to save axis points
#    (which gives number of control points), from 'use every N residues'. 
#    Per helix, if a helix is longer than 4 residues, reset helix-specific variables,
#    generate points on the helix axis according to Sugeta and Miyazawa '67 and store them. 
#    $select_list is populated by these coordinates acccording to 
#    the user-assigned control point frequency and distance away from the helix end. 
#    Residue numbers corresponding to the $select_list are stored.
#    To ensure full helix length, the first and last geometrical centre
#    are always stored, irrespective of chosen control point interval. 
#    The distance between the last and next-to-last stored axis point is stored
#    for use during spline creation to optimise spline point spacing.
#    Axis generation results in helix length loss. This is remedied by extending
#    the helix' geometrical centre ends by sourcing and vector scaling up to
#    four of the most peripheral geometrical centres, depending on availability,
#    to give vectors that give the direction in which to extend. 
#    Helices are extended in each direction by one residue length, 
#    according to the distance between local end-residues.
#    
#    If the chosen Resolution is larger than 1, the $select_list
#    is appended at start and end by ghost points that are necessary
#    for Catmull-Rom gradient calculation for the spline
#    (if resolution equals one, the spline equals the input coordinates, so is skipped). 
#    N.B. that ghost point coordinates are not part of the final spline point
#    list. The $select_list is looped through in windows of 4 coordinates
#    to give $::bendix::spline_resolution number of spline points per 
#    input-coordinate, according to the Hermite spline algorithm. 
#    The only exception to this set spline point resolution are spline points for extensions, 
#    that are given one residue length's worth of spline points, and the next-to-last axis point,
#    where the number of spline points is corrected using the previously 
#    mentioned, stored factor. Spline coordinates generated per helix are stored
#    as a list ($::bendix::spline_coords) and a variable
#    ($::bendix::spline_startsandstops) stores list indices that correspond
#    to helix starts and ends, to distinguish spline coordinates belonging 
#    to a particular helix. 
#    Where Resolution is 1, $::bendix::spline_coords is set to $select_list.
#
#
# ~ Draw graphics primitives using spline points, and collect data for graphs: ~
#
#    Determine the user's choice of Material and reset graph data collecting
#    variables. Per helix that is longer than 4 residues, populate a local list
#    with spline coordinates that are relevant to this helix.
#    Determine the user's choice of helix colouring. 
#
#
#                      ~~~~~~ Colour by helix angle ~~~~~~ 
#    Seeing as user input for angle side and frequency of axis point usage 
#    is done in units of residues, but Bendix deals in spline points at this point:
#    Convert Residue distances to spline point numbers and vice versa.
#    If helices are to be coloured by helix angle, determine whether
#    there are enough spline points to calculate at least one angle between
#    adjacent angle side lengths. If not, a zero-angle coloured cylinder
#    (or sphere for helices of length < 4 residues) is made between
#    the first and last spline point.
#
#    Where sufficient spline points exist for angle calculation,
#    bendix colouration and angle calculation is broken down into two parts: peripheral spline 
#    points that are a distance equal to less than one angle side away from the end, and helix middle,
#    where the angle can be computed. 
#    Peripheral spline points are given linearly varying angle values
#    between zero and their nearest middle angle value; increasing for
#    helix starts and decreasing for helix ends. When observed in the plot, 
#    this linear trace also serves to show users what angle values to ignore.
#    The calculated angle divided by the MaxAngle
#    (set by the user in the GUI by colour threshold) determines
#    a bendix segment colour that reflects angle acuity. Angle measurements
#    are stored and compared with same-helix angles to give the maximum angle
#    per helix and frame if the user has chosen to store angles.
#    Spline point indices are checked if they correspond to the location 
#    of a control point, in which case the amino acid residue number corresponding 
#    to this spline point is stored for precision-graphing purposes.
#    Non-control point spline points are given averaged residue numbers
#    and are not written out in the Surf graph.
#    Once the angle, and thus the colour, has been determined
#    for a bendix segment, it is drawn to screen by cylinder and sphere
#    graphics primitives. All helix ends are capped by 0-angle coloured spheres.
#
#
#                   ~~~~~~ Colour helices uniformly ~~~~~~
#    The user has a choice of three uniform helix colour schemes; full protein,
#    per chain or per helix. In either case, the user may choose to provide
#    his or her own colour(s), or use a default colour sequence. 
#    Where the user supplies helix colour(s); if less colours are provided
#    than required to colour the protein subset, any unaccounted-for protein
#    is coloured white. Once colouring is determined, the bendix segment
#    is drawn as previously described.
#
#
#                 ~~~~~~ Special case for small helices ~~~~~~
#    Small helices (equal to, or less than one helical turn) are coloured
#    according to same colour scheme choice as the rest of the protein,
#    but since too few axis points exist to measure an angle,
#    small helices are always coloured as zero angle if angle-indicative
#    colouring is chosen. 
#    A 1 to 3 residue helix (by error or otherwise) is displayed as single sphere.
#
#
#   ~~~~~~ Store peripheral helix coordinates for CG backbone drawing ~~~~~~
#    At the start and end of each helix, the two most peripheral helix
#    coordinates are appended to a list of helix start and stop coordinates
#    so that CG backbone can be successfully matched up with adjacent helices.
#
#
#                    ~~~~~~ Final considerations ~~~~~~
#    After a helix is processed, variables related to single helices are reset
#    and a helix counter (used to determine individual helix colouring)
#    is increased by one.
#
#    After all helices are processed, relevant variables are updated
#    to reflect that the protein subset was processed, and what settings
#    were used, so that future changes to esthetic settings only result 
#    in rerunning of the Drawing module of the ::bendix::mainDrawingLoop,
#    rather than helix reassignment, coordinate storage and spline creation.
#    Backbone representation is also dealt with after helix processing; 
#    if a certain representation is indicated by the user, relevant procs
#    are called (::bendix::cartoonify for AT and ::bendix::drawBackbone for CG).
#
#    Lastly variables for last frame, molecule ID, subset and helix assignment
#    are updated.
#
# ------------------------------------------------------------------------------

  proc ::bendix::mainDrawingLoop {} {

	##### Catch when no proteins are loaded
	set N_loaded_Mols [molinfo num]
	if {$N_loaded_Mols == 0} {
		.bex.lf3.field1.rep0 configure -state disable; # Disable Rep0 button
		.bex.lf1.field1.molname set ""
		catch {destroy .pop_load}
		toplevel .pop_load
		wm geometry .pop_load +200+300
		#grab .pop_load
		wm title .pop_load "No molecule?"
		
		message .pop_load.msg1 -width 400 -text "Bendix detects no input file!" -pady 15 -padx 35
		message .pop_load.msg2 -width 400 -text "Please load your molecule,\nthen use bendix." -padx 35 -pady 5
		button .pop_load.okb -text OK -background green -command {destroy .pop_load ; return 0}
		
		pack .pop_load.msg1 .pop_load.msg2 -side top
		pack .pop_load.okb -side bottom -pady 5
		
	    #### Upon window close or minimisation), exit properly.
	   	#bind .pop_load <Unmap> {
		#	destroy .pop_load
		#	return 0
		#}
		#return
		
	### Only attempt to draw if the given molecule is present.
	} else {
		# Enable Rep0 button
		.bex.lf3.field1.rep0 configure -state normal; 

	set ::bendix::proteinID [.bex.lf1.field1.molname get]
	
	#### Delete all graphics upon Draw, so user need not hit Erase to update the display
	graphics $::bendix::proteinID delete all

	#### Reset variables
	# Set the var '::bendix::die' to zero when the user Draws. Allows auto-updated display upon frame-change.
	set ::bendix::die 0; 
	if {$::bendix::previous_frame != $::bendix::frame} {
		graphics $::bendix::proteinID delete all
		lset ::bendix::First_backbone_drawn_list $::bendix::proteinID ""
		lset ::bendix::Last_backbone_drawn_list $::bendix::proteinID ""
		set ::bendix::first_surf_drawn ""
		set ::bendix::first_axis_drawn "";
		set ::bendix::last_axis_drawn "";
		set ::bendix::axesON 0;
		set ::bendix::surfed 0
	}
	
	if {$::bendix::previous_proteinID != $::bendix::proteinID} {
	#### In case a new molecule is loaded - reset Graph Data variables
		set ::bendix::AThelix_string ""
		set ::bendix::points_to_graph ""
		set ::bendix::curvature_graph_X ""
		set ::bendix::curvature_graph_Y ""
		set ::bendix::maximumAngle_per_frame ""
		set ::bendix::vmdFrame ""
		set ::bendix::angle_per_turn_per_frame {}
		set ::bendix::angle_per_AA_per_frame_BARRED ""
		set ::bendix::helix_axis_per_helix_per_frame_BARRED ""
		set ::bendix::first_surf_drawn "";
		set ::bendix::first_axis_drawn "";
		set ::bendix::last_axis_drawn "";
		set ::bendix::axesON 0;
		set ::bendix::surfed 0
		set ::bendix::index_of_helix_assignment_used_for_angle "";
	}
	
	#### Additionally, if Helicity is changed, erase Analysis data points to avoid scrambling graphs
	if {$::bendix::previous_helix_assignment != $::bendix::helix_assignment} {
		set ::bendix::points_to_graph ""
		set ::bendix::curvature_graph_X ""
		set ::bendix::curvature_graph_Y ""
		set ::bendix::maximumAngle_per_frame ""
		set ::bendix::vmdFrame ""
		set ::bendix::angle_per_turn_per_frame {}
		set ::bendix::angle_per_AA_per_frame_BARRED ""
		set ::bendix::helix_axis_per_helix_per_frame_BARRED ""
		set ::bendix::index_of_helix_assignment_used_for_angle "";
	}

	#### Ensure that field input are within reasonable range, and (where applicable) in decimal form:
	if {$::bendix::spline_resolution < 1} {
		set ::bendix::spline_resolution 1
	}
	
	##set ::bendix::spline_resolution [expr {$::bendix::spline_resolution/1.0}] ; #Make 1d.p. at calculation-stage instead.______________________
	if {$::bendix::angle_max < 0.1} {
		set ::bendix::angle_max 0.1
	}
	set ::bendix::TurnResolution [.bex.lf2.field1.cp get]
	if {$::bendix::TurnResolution == ""} {
		set ::bendix::TurnResolution 4
	}
	
	if {$::bendix::AngleSide == ""} {
		set ::bendix::AngleSide 3.6
	} elseif {$::bendix::AngleSide < 1} {
		set ::bendix::AngleSide 1
	}
	
	### Only perform this check-loop if either of these changed:
	### proteinID, subset, particle_name
	### of fixed_particleNames is unset.
	if {$::bendix::previous_proteinID != $::bendix::proteinID || $::bendix::previous_subset != $::bendix::subset || $::bendix::previous_particle_name != $::bendix::particle_name || $::bendix::fixed_particleNames == ""} {
		::bendix::ParticleSubsetOK
	}

	if {$::bendix::previous_proteinID != $::bendix::proteinID || $::bendix::previous_subset != $::bendix::subset || $::bendix::previous_helix_assignment != $::bendix::helix_assignment || $::bendix::helix_assignment == "" || $::bendix::previous_helix_type != $::bendix::helix_type || $::bendix::previous_particle_name != $::bendix::particle_name} {
		
################################################################################
#                           USER-ASSIGNED HELICITY                             #
################################################################################
		if {$::bendix::helix_assignment != ""} {
		#### The user has assigned helices.
		#### Only do this time-consuming loop if necessary, i.e. if the user has defined novel helicity.
			set N_assigned_helices [llength $::bendix::helix_assignment]
			set N_chains 1
			
			#### TESTING USER ASSIGNMENT AGAINST REAL PROTEIN DATA
			#### Check for an odd number of helix starts, in which case Return and notify.
			set maybe_integer [expr {$N_assigned_helices%2 }];
			if {$maybe_integer == 1} {
				catch {destroy .incorrect_N_startstop}
				toplevel .incorrect_N_startstop
				wm geometry .incorrect_N_startstop +100+250
				#grab .incorrect_N_startstop
				wm title .incorrect_N_startstop "Missing helix resid number"
				message .incorrect_N_startstop.msg -width 300 \
					-text "You have entered an odd number of helix starts and stops.\
Please ensure that all helix start resid are paired with an end resid.\nUse only numeric input; for example 1 10 refers to a helix between resid 1 and resid 10." -pady 15 -padx 25
				button .incorrect_N_startstop.ok -text OK -background green -command {destroy .incorrect_N_startstop ; return }
				pack .incorrect_N_startstop.msg -side top
				pack .incorrect_N_startstop.ok -side bottom -pady 5

			    #### Upon window close or minimisation, exit properly.
			   	#bind .incorrect_N_startstop <Unmap> {
				#	destroy .incorrect_N_startstop
				#	return
				#}
				#return
				
			### Catch odd Ns of residues
			} else {

			#### USER-ASSIGNED PROTEIN CHAINS
			#### Derive each chain's start and stop residue number from $::bendix::helix_assignment. Store it.
			set assigned_chain_start_and_stop_AA [lindex $::bendix::helix_assignment 0]
			set assigned_chain_stops {}
			for {set assigned_N 1} {$assigned_N < $N_assigned_helices} {incr assigned_N} {
				if {[lindex $::bendix::helix_assignment $assigned_N] < [lindex $::bendix::helix_assignment [expr {$assigned_N -1}]]} {
				#### A larger AA number followed by a smaller AA number is taken as the start of a new chain. 
				#### Otherwise do helix assignment per chain, separately. 
					incr N_chains
					lappend assigned_chain_stops [lindex $::bendix::helix_assignment [expr {$assigned_N -1}]]
					lappend assigned_chain_start_and_stop_AA [lindex $::bendix::helix_assignment [expr {$assigned_N -1}]]
					lappend assigned_chain_start_and_stop_AA [lindex $::bendix::helix_assignment $assigned_N]
				}
			}
			lappend assigned_chain_start_and_stop_AA [lindex $::bendix::helix_assignment [expr {$assigned_N -1}]];
			lappend assigned_chain_stops [lindex $::bendix::helix_assignment [expr {$assigned_N -1}]]

			#### REAL PROTEIN CHAIN DETAIL: STORE CHAIN NAMES AND START/END RESIDUE NUMBERS
			#### Store the subset's residue numbers, chains and indices, for checking against the user-assignment.
			
			if {$::bendix::CG == 1} {
				# Using coarse-grained particles
				if {$::bendix::subset != ""} {
					set all_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames)"]
					#set all_backbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name CA or (name $::bendix::fixed_particleNames \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" ))"]
				} else {
					set all_backbone [atomselect $::bendix::proteinID "name $::bendix::fixed_particleNames"]
					#set all_backbone [atomselect $::bendix::proteinID "name CA or (name $::bendix::fixed_particleNames \"$::bendix::MartiniBackbone\" \"$::bendix::MartiniBackboneNew\" )"]
				}
				set all_AAs [$all_backbone get resid]; 
				#set all_AAs [$all_backbone get residue]; # depreciated; not compatible with both CG and AT; whereas residues are sequential in one, they're not in the other, so calculations that rely on sequence fail.
				set all_chains [$all_backbone get chain]
				set all_index [$all_backbone get index]
				$all_backbone delete
			} else {
				# Using atomistic particles
				if {$::bendix::subset != ""} {
					set all_protein [atomselect $::bendix::proteinID "($::bendix::subset) and name $::bendix::fixed_particleNames"]
				} else {
					set all_protein [atomselect $::bendix::proteinID "name $::bendix::fixed_particleNames"]
				}
				set all_AAs [$all_protein get resid];
				set all_chains [$all_protein get chain]
				set all_index [$all_protein get index];
				$all_protein delete;
			}

			#### If chains and resid numbers were identified,
			#### store subsequent chains and the residue number and index that they start and stop at.
			if {$all_chains != "" && $all_AAs != ""} {
				set ::bendix::list_of_chains [lindex $all_chains 0]
				global start_and_end_AA_N_of_chains
				set start_and_end_AA_N_of_chains [lindex $all_AAs 0]

				for {set chain_index 1} {$chain_index < [llength $all_chains]} {incr chain_index} {
				#### Register what AA N starts and ends each chain:
					if {[lindex $all_chains $chain_index] != [lindex $all_chains [expr {$chain_index -1}]] || [lindex $all_AAs $chain_index] < [lindex $all_AAs [expr {$chain_index -1}]]} {
					#### If the chain letter [A] != the following chain letter [B], 
					#### OR am amino acid number is followed by a smaller one,
					#### register both chains' amino acid numbers.
						if {[lindex $all_chains $chain_index] == [lindex $all_chains [expr {$chain_index -1}]]} {
							::bendix::same_chain_names
							return
							error "Change chain names."
						}
						lappend ::bendix::list_of_chains [lindex $all_chains $chain_index];
						lappend start_and_end_AA_N_of_chains [lindex $all_AAs [expr {$chain_index -1}]];
						lappend start_and_end_AA_N_of_chains [lindex $all_AAs $chain_index];
					}
				}
				lappend start_and_end_AA_N_of_chains [lindex $all_AAs [expr {$chain_index -1}]];
				if {[lindex $all_chains [expr {$chain_index -1}]] != [lindex $all_chains [expr {$chain_index -2}]]} {
					lappend ::bendix::list_of_chains [lindex $all_chains [expr {$chain_index -1}]];
				}

			#### If only resid numbers (not chains) were obtained from the selection,
			#### register chain start and stop resid by detecting when a large resid number
			#### is followed by a smaller one.
			} elseif {$all_AAs != ""} {
				global start_and_end_AA_N_of_chains
				set start_and_end_AA_N_of_chains [lindex $all_AAs 0]

				for {set AA_index 1} {$AA_index < [llength $all_AAs]} {incr AA_index} {
					if {[lindex $all_AAs $AA_index] < [lindex $all_AAs [expr {$AA_index -1}]]} {
					#### If the next resid number is smaller than the previous, it's a new chain.
						lappend start_and_end_AA_N_of_chains [lindex $all_AAs [expr {$AA_index -1}]]
						lappend start_and_end_AA_N_of_chains [lindex $all_AAs $AA_index]
					}
				}
				## AA N of the end of the last chain.
				lappend start_and_end_AA_N_of_chains [lindex $all_AAs [expr {$AA_index -1}]]; 
			}
			unset all_AAs all_chains all_index

			#### If, from the above, resid numbers where chains start and stop were written,
			#### compare it to user-assigned fragment start and stop resid numbers.
			#### If user-assigned chain start resids are smaller/larger than the real chain start/end,
			#### warn, inform the user of valid resid numbers and return.
			if {[info exists start_and_end_AA_N_of_chains]} {
				if {$start_and_end_AA_N_of_chains != ""} {

					if {[llength $assigned_chain_start_and_stop_AA] == [llength $start_and_end_AA_N_of_chains]} {
					#### There should be equal amounts of assigned and REAL chain stop/start AA Ns.
						for {set helixN_checker 0} {$helixN_checker < [llength $assigned_chain_start_and_stop_AA]} {incr helixN_checker 2} {
							if {[lindex $assigned_chain_start_and_stop_AA $helixN_checker] < [lindex $start_and_end_AA_N_of_chains $helixN_checker] || [lindex $assigned_chain_start_and_stop_AA [expr {$helixN_checker +1}]] > [lindex $start_and_end_AA_N_of_chains [expr {$helixN_checker +1}]]} {
								::bendix::incorrect_residue_Ns
								### Errors are the only way I know of stopping script execution within a nested loop 
								### without creating new variables. They are unfortunately ugly, but harmless.
								error "Please correct your subset and/or helix assignment."
							}
						}
					} else {
					#### Extra loop to catch proteins with different chains but sequential resid numbers:
						set mended_REAL_starts_and_ends [lindex $start_and_end_AA_N_of_chains 0]
						for {set chaincounter 1} {$chaincounter < [expr {[llength $start_and_end_AA_N_of_chains] -1}]} {incr chaincounter 2} {
							if {[lindex $start_and_end_AA_N_of_chains $chaincounter] != [expr {[lindex $start_and_end_AA_N_of_chains [expr {$chaincounter + 1}]] -1}]} {
							#### Only lappend Real, mended chain AAs if not subsequent AAs in different chains:
								lappend mended_REAL_starts_and_ends [lindex $start_and_end_AA_N_of_chains $chaincounter]
								lappend mended_REAL_starts_and_ends [lindex $start_and_end_AA_N_of_chains [expr {$chaincounter +1}]]
							}
						}
						lappend mended_REAL_starts_and_ends [lindex $start_and_end_AA_N_of_chains [expr {[llength $start_and_end_AA_N_of_chains] -1}]]
						for {set chainNs 0} {$chainNs < [llength $assigned_chain_start_and_stop_AA]} {incr chainNs 2} {
							if {[lindex $assigned_chain_start_and_stop_AA $chainNs] < [lindex $mended_REAL_starts_and_ends $chainNs] || [lindex $assigned_chain_start_and_stop_AA [expr {$chainNs +1}]] > [lindex $mended_REAL_starts_and_ends [expr {$chainNs +1}]]} {
								::bendix::incorrect_residue_Ns
								### Errors are the only way I know of stopping script execution within a nested loop 
								### without creating new variables. They are unfortunately ugly, but harmless.
								error "Please correct your subset and/or helix assignment."
							}
						}
					}
				} else {
				#### The variable start_and_end_AA_N_of_chains is empty. Try changing particle type and rerun.
					if {$::bendix::tested_alt_particle_for_empty_start_and_end_AA_N_of_chains == 0 || $::bendix::autolooper == 0} {
						#### Un/tick the box to try the alternate particle type and rerun the ::mainDrawingLoop
						set ::bendix::CG [expr {1 - $::bendix::CG}]
						#### Update the GUI
						::bendix::MakeCG; 
						set ::bendix::tested_alt_particle_for_empty_start_and_end_AA_N_of_chains 1
						set ::bendix::autolooper 1
						::bendix::mainDrawingLoop
					} else {
						::bendix::unknown_particle_type
						return
					}
				}
			} else {
				#### The variable start_and_end_AA_N_of_chains does not exist. Try changing particle type and rerun.
				if {$::bendix::tested_alt_particle_for_inexistent_start_and_end_AA_N_of_chains == 0 || $::bendix::autolooper == 0} {
					#### Un/tick the box to try the alternate particle type and rerun the ::mainDrawingLoop
					set ::bendix::CG [expr {1 - $::bendix::CG}]
					::bendix::MakeCG
					set ::bendix::tested_alt_particle_for_inexistent_start_and_end_AA_N_of_chains 1
					set ::bendix::autolooper 1
					::bendix::mainDrawingLoop
				} else {
					::bendix::unknown_particle_type
					return
				}
			}

			#### CREATE USER-ASSIGNMENT-BASED REGEX:
			#### If the user has assigned helicity in multiple chains, 
			#### use a chain-specific Regex to denote helix starts and stops, as defined by the user.
			if {$N_chains > 1} {
				set chainN 1
				set ::bendix::AThelix_string "(chain [lindex $::bendix::list_of_chains [expr {$chainN -1}]] and ("
				for {set helixN 0} {$helixN < [expr {[llength $::bendix::helix_assignment] -2}]} {incr helixN 2} {
					append ::bendix::AThelix_string "resid [lindex $::bendix::helix_assignment $helixN] to [lindex $::bendix::helix_assignment [expr {$helixN +1}]]"
					
					### A resid followed by a lower number resid is defined as a new chain
					if {[lindex $::bendix::helix_assignment [expr {$helixN +1}]] > [lindex $::bendix::helix_assignment [expr {$helixN +2}]]} {
						incr chainN
						append ::bendix::AThelix_string ") or chain [lindex $::bendix::list_of_chains [expr {$chainN -1}]] and ("
					} else {
						append ::bendix::AThelix_string " or "
					}
				}
				append ::bendix::AThelix_string "resid [lindex $::bendix::helix_assignment $helixN] to [lindex $::bendix::helix_assignment [expr {$helixN +1}]]))"
				### Several chains detected, and the following helicity and chains derived: $::bendix::AThelix_string
				
				### Extra loop to correct if several chains have sequential resids, leading to erroneous chain-assignment:
				if {$chainN == 1} {
					### .. so the increasing chain loop was never entered.
					set ::bendix::AThelix_string "("
					for {set helixNo 0} {$helixNo <= [expr {[llength $::bendix::helix_assignment] -2}]} {incr helixNo 2} {
						if {$helixNo == [expr {[llength $::bendix::helix_assignment] -2}]} {
							append ::bendix::AThelix_string "resid [lindex $::bendix::helix_assignment $helixNo] to [lindex $::bendix::helix_assignment [expr {$helixNo + 1}]])"
						} else {
							append ::bendix::AThelix_string "resid [lindex $::bendix::helix_assignment $helixNo] to [lindex $::bendix::helix_assignment [expr {$helixNo + 1}]] or ";
						}
					}
				}
				
			} else {
			#### The user assigned helicity to a single chain. Create Regex.
				set ::bendix::AThelix_string "("
				for {set helixNo 0} {$helixNo <= [expr {[llength $::bendix::helix_assignment] -2}]} {incr helixNo 2} {
					if {$helixNo == [expr {[llength $::bendix::helix_assignment] -2}]} {
						append ::bendix::AThelix_string "resid [lindex $::bendix::helix_assignment $helixNo] to [lindex $::bendix::helix_assignment [expr {$helixNo + 1}]])"
					} else {
						append ::bendix::AThelix_string "resid [lindex $::bendix::helix_assignment $helixNo] to [lindex $::bendix::helix_assignment [expr {$helixNo + 1}]] or ";
					}
				}
			}
			set ::bendix::String_for_cartoonify $::bendix::AThelix_string

################################################################################
#                               GET COORDINATES                                #
################################################################################
			#### Get coordinates for user-assigned helices, using regex's
			#### Get atomselection's indices in order to get coordinates
			
			if {$::bendix::CG == 0 } {
			#### AT : use 'backbone' as part of particle-search
				if {$::bendix::subset == ""} {					
					set helixbackbone [atomselect $::bendix::proteinID "(name $::bendix::fixed_particleNames ) and $::bendix::AThelix_string" frame $::bendix::frame]
					#set helixbackbone [atomselect $::bendix::proteinID "backbone and name $::bendix::fixed_particleNames and $::bendix::AThelix_string" frame $::bendix::frame]
					### 'backbone' is depreciated since VMD has rules for what backbones are, leading to empty selections if the particle name is altered beyond CA
				} else {
					set helixbackbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames ) and $::bendix::AThelix_string" frame $::bendix::frame];
				}
			} else {
				
			#### CG : Bond or Martini-specific particle names are replaced by the potential, user-specified, custom particle
				if {$::bendix::subset == ""} {
					set helixbackbone [atomselect $::bendix::proteinID "(name $::bendix::fixed_particleNames ) and $::bendix::AThelix_string"]; # Custom particle names
				} else {
					set helixbackbone [atomselect $::bendix::proteinID "($::bendix::subset) and (name $::bendix::fixed_particleNames ) and $::bendix::AThelix_string"]; # Custom particle names
				}
			}
			set residueNs [$helixbackbone get resid]
			### Discontinue if no residues were selected
			if {$residueNs != ""} {
				### Resid numbers were selected by the above REGEX: $residueNs
			
				set ::bendix::residueIndex [$helixbackbone get index]
				set Natoms [$helixbackbone num]
				$helixbackbone delete
				#### Helix start and end indices are used in Straight helices and to keep track of helix colouring,
				set ::bendix::startIndex [lindex $::bendix::residueIndex 0]
				set ::bendix::endIndex ""
				set ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex 0]
				set endN 0
				set N_chains 0
				set ::bendix::normalised_startsandstops 0
				set ::bendix::xyz {}

				#### Using the atomselection, go through each atom index and retrieve its coordinate.
				#### It uses select, existing atoms, and does not assume successive index numbers, 
				#### so is robust to missing atoms.
				#### normalised starts_and_stops are used for xyz-list coordinate recovery.
			
				for {set N 1} {$N < $Natoms} {incr N} {
					set oldAAN [lindex $residueNs [expr {$N-1}]]
					set newAAN [lindex $residueNs $N]
					set thisIndex [lindex $::bendix::residueIndex [expr {$N - 1}]]
					set xyzIndex [atomselect $::bendix::proteinID "index $thisIndex"]
		
					if {$N == 1} {
						set ::bendix::xyz [$xyzIndex get {x y z}]
					} else {
						append ::bendix::xyz " " [$xyzIndex get {x y z}]
					}
					$xyzIndex delete

					#### If two successive residue numbers are not equal and the first +1 is not equal to the second,
					#### store index and residue numbers.
					if {$oldAAN != $newAAN && $newAAN != [expr {$oldAAN + 1}]} {
						append ::bendix::startIndex " " [lindex $::bendix::residueIndex $N]
						lappend ::bendix::endIndex [lindex $::bendix::residueIndex [expr {$N -1}]]
						lappend ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex [expr {$N -1}]]
						lappend ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex $N]
						append ::bendix::normalised_startsandstops " " [expr {$N - 1}]
						append ::bendix::normalised_startsandstops " " $N
					}
				}

				lappend ::bendix::endIndex [lindex $::bendix::residueIndex [expr {$N - 1}]]
				lappend ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex [expr {$N - 1}]]
				append ::bendix::normalised_startsandstops " " [expr {$N - 1}]
				set thisIndex [lindex $::bendix::residueIndex [expr {$N -1}]]
				set xyzIndex [atomselect $::bendix::proteinID "index $thisIndex"]
				append ::bendix::xyz " " [$xyzIndex get {x y z}]
				$xyzIndex delete
				catch {unset residueNs Natoms endN N_chains oldAAN newAAN thisIndex}
				
			} else {
			#### Based on the particle types & subset that the user assigned, bendix could not assign helices.
			#### It could be that there are no helices in this area.
				if {$::bendix::tested_alt_particle_for_empty_residueNs == 0 || $::bendix::autolooper == 0} {
					#### Start_and_end_AA_N_of_chains does not exist.
					#### Un/tick the box to try the alternate particle type and rerun the ::mainDrawingLoop
					set ::bendix::CG [expr {1 - $::bendix::CG}]
					::bendix::MakeCG
					set ::bendix::tested_alt_particle_for_empty_residueNs 1
					set ::bendix::autolooper 1
					::bendix::mainDrawingLoop
				} else {
					::bendix::no_helicity
					return
				}
			}
		}

################################################################################
#                            AUTOMATIC HELIX ASSIGNMENT                        #
################################################################################	
		} elseif {$::bendix::helix_assignment == "" } {
		#### Where helix-assignment does not exist, bendix auto-assigns it for bendices.
		#### Get auto-assigned helices - by STRIDE for AT, by DSSP pre-assigned MARTINI particles for CG
		
			if {$::bendix::CG == 0} {
				### Get all AT helix coordinates, using STRIDE:
				set ::bendix::String_for_cartoonify "helix"
				if {$::bendix::subset != ""} {
					set helixbackbone [atomselect $::bendix::proteinID "helix and name $::bendix::fixed_particleNames and ($::bendix::subset)" frame $::bendix::frame]
					#set helixbackbone [atomselect $::bendix::proteinID "helix and backbone and ($::bendix::subset)" frame $::bendix::frame]
				} else { 
					set helixbackbone [atomselect $::bendix::proteinID "helix and name $::bendix::fixed_particleNames" frame $::bendix::frame]
					#set helixbackbone [atomselect $::bendix::proteinID "helix and backbone" frame $::bendix::frame]
				}
			} else {
			#### Get coordinates for CG, by selecting MARTINI particles that correspond to helices.
				
				if {$::bendix::subset != ""} {
					set helixbackbone [atomselect $::bendix::proteinID "(name \"$::bendix::MartiniHelix\" or name \"$::bendix::MartiniHelixNew\" ) and ($::bendix::subset)" frame $::bendix::frame]
				} else { 
					set helixbackbone [atomselect $::bendix::proteinID "(name \"$::bendix::MartiniHelix\" or name \"$::bendix::MartiniHelixNew\" )" frame $::bendix::frame]
				}
			}
			set residueNs [$helixbackbone get resid];
			set chainNames [$helixbackbone get chain]; 
			### NB These lists have the same length. A chain char is provided per resid, e.g. A A A A B B B B B 
			
			if {$residueNs != ""} {
				if {$chainNames != ""} {
					set ::bendix::String_for_cartoonify "(chain [lindex $chainNames 0] and (resid [lindex $residueNs 0] "
					set ::bendix::list_of_chains [lindex $chainNames 0]
				} else {
					set ::bendix::String_for_cartoonify "(resid [lindex $residueNs 0] "
				}
				set ::bendix::helix_assignment "[lindex $residueNs 0] "; 
				set ::bendix::residueIndex [$helixbackbone get index]
				set Natoms [$helixbackbone num]
				$helixbackbone delete
				set ::bendix::startIndex [lindex $::bendix::residueIndex 0]
				set ::bendix::endIndex ""
				set ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex 0]
				set endN 0
				set N_chains 0
				set ::bendix::normalised_startsandstops 0

################################################################################
#                               GET COORDINATES                                #
################################################################################
				#### Get coordinates for auto-assigned helices.
				set ::bendix::xyz {}
				set indices_where_chainNames_change [list 0]
				if {$chainNames != ""} {
					set indices_where_chainNames_change ""
					set chainNameChange 0
					set lastChainIndex [expr {[llength $chainNames] -1}]
					for {set currChainIndex 1} {$currChainIndex < $lastChainIndex} {incr currChainIndex} {
						if {[lindex $chainNames [expr {$currChainIndex -1}]] ne [lindex $chainNames $currChainIndex]} {
							lappend indices_where_chainNames_change $currChainIndex
							### Found a chain index change at $indices_where_chainNames_change for these chain names: $chainNames
							set chainNameChange 1
						}
					}
					if {$chainNameChange == 0} {
						### So that below doesn't give error (looking for a list)
						set indices_where_chainNames_change [list 0]
					}					
				}
				
				set currChainIndex 0
				for {set N 1} {$N < $Natoms} {incr N} {
					set oldAAN [lindex $residueNs [expr {$N-1}]]
					set newAAN [lindex $residueNs $N]
					set thisIndex [lindex $::bendix::residueIndex [expr {$N - 1}]]
					set xyzIndex [atomselect $::bendix::proteinID "index $thisIndex"]
					set N_helices_for_graph 0
	
					if {$N == 1} {
						set ::bendix::xyz [$xyzIndex get {x y z}]
					} else {
						append ::bendix::xyz " " [$xyzIndex get {x y z}]
					}
					
					$xyzIndex delete
					if {$oldAAN != $newAAN && $newAAN != [expr {$oldAAN + 1}]} {
						append ::bendix::helix_assignment "$oldAAN $newAAN "
						
						### New residue is smaller than previous residue: New chain:
						if {$oldAAN > $newAAN} {
							if {$chainNames != ""} {
								### Retrieving new chain names for the REGEX
								append ::bendix::String_for_cartoonify "to $oldAAN) or chain [lindex $chainNames [lindex $indices_where_chainNames_change $currChainIndex]] and (resid $newAAN "
								lappend ::bendix::list_of_chains [lindex $chainNames [lindex $indices_where_chainNames_change $currChainIndex]]
								incr currChainIndex
							} else {
								::bendix::same_chain_names
								return
								error "Change chain names."
							}							
						} elseif {$N == [lindex $indices_where_chainNames_change $currChainIndex]} {
							### N is never 0, so if there are no different chains noted, this loop will not be called.
							### Out of these chainNames: $chainNames , I use the first changed chain index, at index [lindex $indices_where_chainNames_change $currChainIndex]
							append ::bendix::String_for_cartoonify "to $oldAAN) or chain [lindex $chainNames [lindex $indices_where_chainNames_change $currChainIndex]] and (resid $newAAN "
							lappend ::bendix::list_of_chains [lindex $chainNames [lindex $indices_where_chainNames_change $currChainIndex]]
							incr currChainIndex						
						} else {
							append ::bendix::String_for_cartoonify "to $oldAAN or resid $newAAN "
						}
						
						append ::bendix::startIndex " " [lindex $::bendix::residueIndex $N]
						lappend ::bendix::endIndex [lindex $::bendix::residueIndex [expr {$N -1}]]
						lappend ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex [expr {$N -1}]]
						lappend ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex $N]
						incr N_helices_for_graph
						if {$endN == 0} {
							append ::bendix::normalised_startsandstops " " [expr {$N - 1}]
							incr endN
						} else {
							append ::bendix::normalised_startsandstops " " [expr {$N - 1}]
						}
						append ::bendix::normalised_startsandstops " " $N
					}
				}
				append ::bendix::helix_assignment "$newAAN"
				lappend ::bendix::endIndex [lindex $::bendix::residueIndex [expr {$N -1}]]
				lappend ::bendix::helix_indices_at_startNstop [lindex $::bendix::residueIndex [expr {$N -1}]]
				append ::bendix::normalised_startsandstops " " [expr {$N - 1}]
				set thisIndex [lindex $::bendix::residueIndex [expr {$N -1}]]
				set xyzIndex [atomselect $::bendix::proteinID "index $thisIndex"]
				append ::bendix::xyz " " [$xyzIndex get {x y z}]
				if {$chainNames != ""} {
					append ::bendix::String_for_cartoonify "to $newAAN))"
				} else {
					append ::bendix::String_for_cartoonify "to $newAAN)"
				}
				
				$xyzIndex delete
				catch {unset residueNs Natoms endN N_chains}

			} else {
			#### Based on the particle types & subset that the user assigned, bendix could not auto-assign helices. 
			#### It could be that there are no helices in this area.
				if {$::bendix::custom_particle == 1} {
					set ::bendix::custom_particle 0
					::bendix::MakeCG
					::bendix::mainDrawingLoop					
				} elseif {$::bendix::tested_alt_particle_for_empty_residueNs == 0 || $::bendix::autolooper == 0} {
					#### Start_and_end_AA_N_of_chains does not exist.
					#### Un/tick the box to try the alternate particle type and rerun the ::mainDrawingLoop
					set ::bendix::CG [expr {1 - $::bendix::CG}]
					::bendix::MakeCG
					set ::bendix::tested_alt_particle_for_empty_residueNs 1
					set ::bendix::autolooper 1
					::bendix::mainDrawingLoop
				} else {
					::bendix::no_helicity
					return
				}
			}
		}
		
	} elseif {$::bendix::previous_frame != $::bendix::frame && $::bendix::residueIndex != ""} {
	# && $::bendix::helix_type == 1 : Depreciated since coordinates needs updating for straight helices as well, or trajectory graphics are not updated as the frame changes.
	#### Loop used by an updated frame (rather than helix or subset assignment changes). Store updated coordinates without re-assigning them.

		set N_indices [expr {[llength $::bendix::residueIndex] -1}]
		for {set n 0} {$n <= $N_indices} {incr n} {
			set thisIndex [lindex $::bendix::residueIndex $n]
				set xyzIndex [atomselect $::bendix::proteinID "index $thisIndex"]
			if {$n==0} {
				set ::bendix::xyz [$xyzIndex get {x y z}]
			} else {
				append ::bendix::xyz " " [$xyzIndex get {x y z}]
			}
				$xyzIndex delete
		}
	} ;#### Helix assignment and coordinate retrieval module ends.

################################################################################
#                      STRAIGHT HELICES - no spline needed                     #
################################################################################
	if {$::bendix::helix_type == 0 && $::bendix::startIndex != ""} {		
		
		#### Fix Material
		set material_choice [.bex.lf2.listbox get]
		if {$material_choice == ""} {
			graphics $::bendix::proteinID material EdgyShiny
		} else {
			graphics $::bendix::proteinID material $material_choice
		}
		set ::bendix::helix_coord_at_startNstop ""; 
		#### Variables stored in order to attach CG backbone correctly, later in ::drawbackbone (if called)
		set helix_list_length [llength $::bendix::normalised_startsandstops]
		set helixcount 0
		
		#### Loop per helix:
		for {set j 0} {$j <= [expr {$helix_list_length - 2}]} {incr j 2} {
			
			set N_particles_in_this_helix [expr {[lindex $::bendix::normalised_startsandstops [expr {$j + 1}]] - [lindex $::bendix::normalised_startsandstops $j] +1 }]
			if {$N_particles_in_this_helix > 3} {
				if {$N_particles_in_this_helix > 4} {
					
					#### Sugeta and Miyasawa '67 algorithm to calculate the helix axis at start and end of straight helices.
					### Do start...
					set i [lindex $::bendix::normalised_startsandstops $j]
					
					# Vectors that join C alphas:
					set vec12 [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] [lindex $::bendix::xyz $i]]
					set vec23 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] [lindex $::bendix::xyz [expr {$i + 1}]]]
					set vec34 [vecsub [lindex $::bendix::xyz [expr {$i + 3}]] [lindex $::bendix::xyz [expr {$i + 2}]]]
					
					set dv13 [vecsub $vec12 $vec23] 
					set dv24 [vecsub $vec23 $vec34]
					
					set cross_product [veccross $dv13 $dv24]
					set cross_product [vecnorm $cross_product]
					
					set dmag [veclength $dv13]
					set emag [veclength $dv24]
					set dot_product [vecdot $dv13 $dv24]
					set dmag_times_emag [expr $dmag*$emag]
					set costheta [expr $dot_product / $dmag_times_emag]
					
					set costheta1 [expr 2.0*[expr 1.0 - $costheta]]
					set radmag [expr sqrt($dmag_times_emag)/ $costheta1]
					
					set dv13 [vecnorm $dv13]
					set dv24 [vecnorm $dv24]
					
					set rad [vecscale $dv13 $radmag]
					
					set result_point [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] $rad]
					
					### ... do end.
					set i [expr {[lindex $::bendix::normalised_startsandstops [expr {$j + 1}]] - 3}]
					
					# Vectors that join C alphas:
					set vec12 [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] [lindex $::bendix::xyz $i]]
					set vec23 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] [lindex $::bendix::xyz [expr {$i + 1}]]]
					set vec34 [vecsub [lindex $::bendix::xyz [expr {$i + 3}]] [lindex $::bendix::xyz [expr {$i + 2}]]]
					
					set dv13 [vecsub $vec12 $vec23] 
					set dv24 [vecsub $vec23 $vec34]
					
					set cross_product [veccross $dv13 $dv24]
					set cross_product [vecnorm $cross_product]
					
					set dmag [veclength $dv13]
					set emag [veclength $dv24]
					set dot_product [vecdot $dv13 $dv24]
					set dmag_times_emag [expr $dmag*$emag]
					set costheta [expr $dot_product / $dmag_times_emag]
					
					set costheta1 [expr 2.0*[expr 1.0 - $costheta]]
					set radmag [expr sqrt($dmag_times_emag)/ $costheta1]
					
					set dv13 [vecnorm $dv13]
					set dv24 [vecnorm $dv24]
					
					set rad2 [vecscale $dv24 $radmag]
					
					set result_point2 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] $rad2]
				} elseif {$N_particles_in_this_helix == 4} {
					
					#### Sugeta and Miyasawa '67 algorithm to calculate the helix axis at start and end of straight helices, using the fact that it generates 2 axis points per 4 input C-alphas.
					set i [lindex $::bendix::normalised_startsandstops $j]
					
					# Vectors that join C alphas:
					set vec12 [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] [lindex $::bendix::xyz $i]]
					set vec23 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] [lindex $::bendix::xyz [expr {$i + 1}]]]
					set vec34 [vecsub [lindex $::bendix::xyz [expr {$i + 3}]] [lindex $::bendix::xyz [expr {$i + 2}]]]
					
					set dv13 [vecsub $vec12 $vec23] 
					set dv24 [vecsub $vec23 $vec34]
					
					set cross_product [veccross $dv13 $dv24]
					set cross_product [vecnorm $cross_product]
					
					set dmag [veclength $dv13]
					set emag [veclength $dv24]
					set dot_product [vecdot $dv13 $dv24]
					set dmag_times_emag [expr $dmag*$emag]
					set costheta [expr $dot_product / $dmag_times_emag]
					
					set costheta1 [expr 2.0*[expr 1.0 - $costheta]]
					set radmag [expr sqrt($dmag_times_emag)/ $costheta1]
					
					set dv13 [vecnorm $dv13]
					set dv24 [vecnorm $dv24]
					
					set rad [vecscale $dv13 $radmag]
					set rad2 [vecscale $dv24 $radmag]
					
					#### In the original algorithm, two points are generated per 4-residue sliding window,
					#### so this is used to populate even relatively small, 4-residue helices, with a start and end axis point.
					set result_point [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] $rad]
					set result_point2 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] $rad2]
				}
				
				#### The vector algebraic alorithm shortened the helix. Extend the helix at both ends to recover the real helix length:
				# Upwards:
				set extension_vector_upwards_direction [vecsub $result_point $result_point2]
				set extension_vector_upwardsL [veclength $extension_vector_upwards_direction]
				set extension_vector_factor [expr {1.8/$extension_vector_upwardsL}]
				set extension_vector_correct_L [vecscale $extension_vector_upwards_direction $extension_vector_factor]
				set extension_vector_applied_to_correct_ori [vecadd $extension_vector_correct_L $result_point]
				set result_point $extension_vector_applied_to_correct_ori

				# Downwards:
				set extension_vector_upwards_direction [vecsub $result_point2 $result_point]
				set extension_vector_upwardsL [veclength $extension_vector_upwards_direction]
				set extension_vector_factor [expr {1.8/$extension_vector_upwardsL}]
				set extension_vector_correct_L [vecscale $extension_vector_upwards_direction $extension_vector_factor]
				set extension_vector_applied_to_correct_ori [vecadd $extension_vector_correct_L $result_point2]
				set result_point2 $extension_vector_applied_to_correct_ori
				
				#### Collect data for meeting helices by backbone, later (Straight cylinders)
				lappend ::bendix::helix_coord_at_startNstop $result_point
				lappend ::bendix::helix_coord_at_startNstop $result_point2

			}
			
			#### COLOUR 
			if {$::bendix::uniform_colour_type == 0 || $::bendix::uniform_colour == 0} {
			#### Colour entire protein
				if {$::bendix::autoColour== 1 || $::bendix::uniform_colour == 0} {
					graphics $::bendix::proteinID color 0
				} else {
					set user_chosen_colours [split $::bendix::input_colour]
					if {[lindex $user_chosen_colours 0] != ""} {
						graphics $::bendix::proteinID color [lindex $user_chosen_colours 0]
					} else {
						graphics $::bendix::proteinID color white
					}
				}
			} elseif {$::bendix::uniform_colour_type == 1} {
			#### Colour by chain
				set firstIndex_atomselect [atomselect $::bendix::proteinID "index [lindex $::bendix::startIndex $helixcount]"]
				set current_chain [$firstIndex_atomselect get chain];
				$firstIndex_atomselect delete
				if {$::bendix::autoColour == 1} {
					if {$helixcount == 0} {
						set chain_colour 0
						set chain_letter_list $current_chain
					} else {
						if {[lindex $chain_letter_list $chain_colour] != $current_chain} {
							incr chain_colour
							set chain_letter_list [lappend chain_letter_list $current_chain]
						}
					}
				} else {
				#### User-defined chain-colouring.
					set user_chosen_colours [split $::bendix::input_colour]
					if {$helixcount == 0} {
						if {[lindex $user_chosen_colours 0] != ""} {
							set chain_colour [lindex $user_chosen_colours 0]
						} else {
							set chain_colour white
						}
						set chain_letter_list $current_chain
						set N_chains 0
					} else {
						if {[lindex $chain_letter_list $N_chains] != $current_chain} {
							set chain_letter_list [lappend chain_letter_list $current_chain]
							incr N_chains
							if {[lindex $user_chosen_colours $N_chains] != ""} {
								set chain_colour [lindex $user_chosen_colours $N_chains]
							} else {
								set chain_colour white
							}
						}
					}
				}
				graphics $::bendix::proteinID color $chain_colour
			} elseif {$::bendix::uniform_colour_type == 2} {
			#### Colour by helix
				if {$::bendix::autoColour== 1} {
					if {$helixcount >= 32} {
						set helixcount 0
					}
					graphics $::bendix::proteinID color $helixcount
				} else {
					set user_chosen_colours [split $::bendix::input_colour]
					if {[lindex $user_chosen_colours $helixcount] != ""} {
						set helixcolour [lindex $user_chosen_colours $helixcount]
					} else {
						set helixcolour white
					}
					graphics $::bendix::proteinID color $helixcolour
				} 
			}
			
			#### Draw primitives depending on number of available axis points
			if {$N_particles_in_this_helix > 3} { 
				graphics $::bendix::proteinID sphere $result_point radius $::bendix::helix_radius resolution 20
				graphics $::bendix::proteinID cylinder $result_point $result_point2 radius $::bendix::helix_radius resolution 30
				graphics $::bendix::proteinID sphere $result_point2 radius $::bendix::helix_radius resolution 20
			} elseif {$N_particles_in_this_helix == 3 || $N_particles_in_this_helix == 2} {
				### Draw a sphere in the middle of the 1st and last coordinate
				graphics $::bendix::proteinID sphere [vecscale [vecadd [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] ] 0.5] radius $::bendix::helix_radius resolution 20
				
				#### Collect data for meeting helices by backbone, later (Straight cylinders)
				lappend ::bendix::helix_coord_at_startNstop [vecscale [vecadd [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] ] 0.5]
				lappend ::bendix::helix_coord_at_startNstop [vecscale [vecadd [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] ] 0.5]
			} else {
				### But surely there are no 1-residue helices..? Just in case, a sphere here:
				graphics $::bendix::proteinID sphere [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] radius $::bendix::helix_radius resolution 20
				
				#### Collect data for meeting helices by backbone, later (Straight cylinders)
				lappend ::bendix::helix_coord_at_startNstop [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] 
				lappend ::bendix::helix_coord_at_startNstop [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]]
			}
			incr helixcount
		}
		#### Update variables for the next draw
		set ::bendix::previous_helix_type 0

	} elseif {$::bendix::startIndex == ""} {
		if {$::bendix::tested_alt_particle_for_empty_startIndex == 0 || $::bendix::autolooper == 0} {
			#### Un/tick the box to try the alternate particle type and rerun the ::mainDrawingLoop
			set ::bendix::CG [expr {1 - $::bendix::CG}]
			::bendix::MakeCG
			set ::bendix::tested_alt_particle_for_empty_startIndex 1
			set ::bendix::autolooper 1
			::bendix::mainDrawingLoop
		} else {
			::bendix::no_helicity
			return
		}
	} else {
		### Bendices.

################################################################################
#                              GENERATE HELIX AXIS                             #
################################################################################


		### This module is only run if one of these is true:
		#
		# These variables differ:
		# 	proteinID vs previous_proteinID
		# 	subset vs previous_subset
		# 	helix_assignment vs previous_helix_assignment
		# 	spline_resolution vs previous_spline_resolution
		# 	TurnResolution vs previous_TurnResolution
		# 	frame vs previous_frame
		# This variable is nil:
		# 	spline_startsandstops
		# or
		# 	previous_helix_type was straight
		#	StoreAxes is 1 (axis coordinates are being stored)
		
		# NB that helix assignment was just done by the loop above.
		if {$::bendix::previous_proteinID != $::bendix::proteinID || $::bendix::previous_subset != $::bendix::subset || $::bendix::previous_helix_assignment != $::bendix::helix_assignment || $::bendix::previous_spline_resolution != $::bendix::spline_resolution || $::bendix::previous_TurnResolution != $::bendix::TurnResolution || $::bendix::previous_frame != $::bendix::frame || $::bendix::spline_startsandstops == "" || $::bendix::previous_helix_type == 0 || $::bendix::StoreAxes == 1} {
		
			#### Clean slate for novel/re-written spline coordinates and array lengths.
			set ::bendix::spline_coords "";
			set ::bendix::spline_startsandstops "";
			set ::bendix::helix_coord_at_startNstop ""; 
			#### Variables stored in order to attach CG backbone correctly, later in ::drawbackbone (if called)
			set helix_list_length [llength $::bendix::normalised_startsandstops]
			set helixcount 0
			set helix_counter 0
			set splineNs {}
			set spline_list {}
			set list_copy {}
			set counter 0
			set ID_helix 1
			set ::bendix::curvature_graph_Y {}
			set ::bendix::curvature_graph_X {}
			
			if {$::bendix::StoreAxes == 1} {
				### Add frame number to the axis coordinate file output:
				global vmd_frame
				set ::bendix::frame $vmd_frame($::bendix::proteinID)
				set axes_list "\nFrame: $::bendix::frame\n"
			}			
			
			set ::bendix::points_to_graph {};
			array unset ::bendix::Resolution_wrt_ends
			set exact_AANs {}
			set spline_fraction_at_helixEnd {}
			array unset ::bendix::AAN_associated_w_splineIndex
			
			#### Loop per helix:
			for {set j 0} {$j <= [expr {$helix_list_length - 2}]} {incr j 2} {
				set point_list {}
				set select_list {}
				set exact_AANs {}
				set ::bendix::Sugeta_point 0
				set N_particles_in_this_helix [expr {[lindex $::bendix::normalised_startsandstops [expr {$j + 1}]] - [lindex $::bendix::normalised_startsandstops $j] +1 }]
				
				#### Long helices:
				if {$N_particles_in_this_helix > 4} {
					
					if {$::bendix::StoreAxes == 1} {
						### Add helix residue numbers to the axis coordinate file output:
						append axes_list "Resid [lindex $::bendix::helix_assignment $j] to [lindex $::bendix::helix_assignment [expr $j + 1]]:\t"
					}
					set counter 0
					set residueN_vs_Select_list 2; # Since Sugeta-points match up with the middle-point of residues, starting at residue 2.
					set adjusted_start 0
					
					for {set i [lindex $::bendix::normalised_startsandstops $j]} {$i <= [expr {[lindex $::bendix::normalised_startsandstops [expr {$j + 1}]] - 3}]} {incr i} {
						
						#### Generate the helix axis according to Sugeta and Miyazawa '67
						# Vectors that join C alphas:
						set vec12 [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] [lindex $::bendix::xyz $i]]
						set vec23 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] [lindex $::bendix::xyz [expr {$i + 1}]]]
						set vec34 [vecsub [lindex $::bendix::xyz [expr {$i + 3}]] [lindex $::bendix::xyz [expr {$i + 2}]]]
						
						set dv13 [vecsub $vec12 $vec23] 
						set dv24 [vecsub $vec23 $vec34]
						
						set cross_product [veccross $dv13 $dv24]
						set cross_product [vecnorm $cross_product]
						
						set dmag [veclength $dv13]
						set emag [veclength $dv24]
						set dot_product [vecdot $dv13 $dv24]
						set dmag_times_emag [expr $dmag*$emag]
						set costheta [expr $dot_product / $dmag_times_emag]
						
						set costheta1 [expr 2.0*[expr 1.0 - $costheta]]
						set radmag [expr sqrt($dmag_times_emag)/ $costheta1]
						
						set dv13 [vecnorm $dv13]
						set dv24 [vecnorm $dv24]
						
						set rad [vecscale $dv13 $radmag]
						set rad2 [vecscale $dv24 $radmag]
						
						set result_point [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] $rad]
						
						### Possible to averaging this and previous axis-points, since data exists. 
						### To stay close to the original axis algorithm, Bendix does not take advantage of this.
						#if {$::bendix::Sugeta_point != 0} {
						#	### If previous calculated axis exists, use it to average this one.
						#	set Sugeta_add [vecadd $::bendix::Sugeta_point $result_point]
						#	set result_point [vecscale $Sugeta_add 0.5]
						#}
						
						lappend point_list $result_point

						#### Rules for how many points to save per helix turn:
						# Always save the start and end points (the end point is the Sugeta point).
						
						# FIRST point
						if {$i == [lindex $::bendix::normalised_startsandstops $j]} {
							## First add the 1st AA. The corresponding point will be added by extension.
							lappend exact_AANs 1; # Makes sense since you always extend backwards by 1 aa.
							
							## Then add the current AA:
							lappend select_list $result_point
							lappend exact_AANs $residueN_vs_Select_list
							
							#graphics top color yellow; #sugeta
							#graphics top sphere $result_point radius 0.4 resolution 20; #sugeta
							set counter 0
							
						### LAST point: remember we're dealing with two points here: last and Sugeta very last, calculated now. 
						} elseif {$i == [expr {[lindex $::bendix::normalised_startsandstops [expr {$j + 1}]] - 3}] } {
							
							## if you happen upon an Every-Nth-point-to-save, grab it.
							if {$::bendix::TurnResolution == $counter} {
								lappend select_list $result_point
								lappend exact_AANs $residueN_vs_Select_list
								set counter 0
								#graphics top color black; #sugeta
								#graphics top sphere $result_point radius 0.4 resolution 20; #sugeta
							}
							
							### ..and the last point:
							### Avoiding the last Sugeta point, which wasn't part of their original algorithm in any case. 
							### Do extension instead, unless absolutely necessary; where there are only 2 residues in the helix, which won't be precise in any case.
							set N_points [llength $point_list]
							if {$N_points == 1} {
								## If just one other coordinate saved, you need the Sugeta; bad or not.
								## This was previously implemented by default, for all helix lengths.
								set ::bendix::Sugeta_point [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] $rad2]
								lappend point_list $::bendix::Sugeta_point
								lappend select_list $::bendix::Sugeta_point
							} else {
								set last_point [expr {[llength $point_list]-1}] 
								if {$N_points == 2} {
									## Linear extend from the last two points.
									set extension_point8 [lindex $point_list [expr {$last_point -1}]]
									set extension_point9 [lindex $point_list $last_point]
								} else {
									### 1AA interdistance, but using averaged direction vectors of the last 3 points (still relatively local)
									set extension_point8 [vecscale [vecadd [lindex $point_list [expr {$last_point -1}]] [lindex $point_list [expr {$last_point -2}]]] 0.5]
									set extension_point9 [vecscale [vecadd [lindex $point_list $last_point] [lindex $point_list [expr {$last_point -1}]]] 0.5]
								}
								set extension_vector_downwards_direction [vecsub $extension_point9 $extension_point8]
								set extension_vector_applied_to_correct_ori [vecadd $extension_vector_downwards_direction [lindex $point_list $last_point]]
					
								### lappend the extension.
								lappend point_list $extension_vector_applied_to_correct_ori
								set select_list [lappend select_list $extension_vector_applied_to_correct_ori]	
							}						
							lappend exact_AANs [expr $residueN_vs_Select_list +1]
							
							### Adjust the N of spline-points to add for the distance to the extension (always 1!):
							if {$::bendix::TurnResolution != 1} {
							## If cp is every 1 AA, all points are treated equally, with equal N of spline points. Otherwise:
							
								set lastSavedAAN [lindex $exact_AANs [expr [llength $exact_AANs] -2]]; # Last select point, pre-Sugeta.
								set currAA [lindex $exact_AANs [expr [llength $exact_AANs] -1]]; # SugetaPoint: very last, just saved above.
								set TurnResolution_at1dp [expr $::bendix::TurnResolution*1.0];
								set diff [expr [expr {$currAA - $lastSavedAAN}] / $TurnResolution_at1dp]; # This is the fraction of spline points for the last helix section before the last real end-point.
								## ::Resolution_wrt_ends is not the actual resolution, but a factor to multiply the selected spline Resolution to give the correct N of splie points to the Sugeta point.
								set ::bendix::Resolution_wrt_ends($j) $diff
							} else {
								# Turn resolution 1 means all points are saved, so equal spline point distribution everywhere.
								set ::bendix::Resolution_wrt_ends($j) 1.0
							}
							
							lappend exact_AANs [expr $residueN_vs_Select_list +2] ; #For the final extension point, to come after this loop.
							
						} elseif {$::bendix::TurnResolution == $counter} {
							lappend select_list $result_point
							lappend exact_AANs $residueN_vs_Select_list ;
							#graphics top color black; #sugeta
							#graphics top sphere $result_point radius 0.4 resolution 20; #sugeta
							set counter 0
						}						
	
						incr counter
						incr residueN_vs_Select_list;
					}
					
					#### Extend helices in either direction to counter wrt point half-way down a helix-turn
					# 1.	Get coords of available peripheral points in point-list
					# 2.	Get vector in the correct orientation
					# 3.	Make point in opposite direction of vector 

					set N_points [llength $point_list]
					set last_point [expr {[llength $point_list]-1}] 
					
#					### Better to keep close control of what goes into the extension algorithm. Below 'non-local' averaged extensions are depreciated in the final Bendix version:
#					if {$N_points >= 5} {
#						set extension_point1 [vecscale [vecadd [lindex $point_list 0] [lindex $point_list 1] [lindex $point_list 2] [lindex $point_list 3]] 0.25]
#						set extension_point2 [vecscale [vecadd [lindex $point_list 1] [lindex $point_list 2] [lindex $point_list 3] [lindex $point_list 4]] 0.25]
#						set extension_point8 [vecscale [vecadd [lindex $point_list [expr {$last_point -1}]] [lindex $point_list [expr {$last_point -2}]] [lindex $point_list [expr {$last_point -3}]] [lindex $point_list [expr {$last_point -4}]]] 0.25]
#						set extension_point9 [vecscale [vecadd [lindex $point_list $last_point] [lindex $point_list [expr {$last_point -1}]] [lindex $point_list [expr {$last_point -2}]] [lindex $point_list [expr {$last_point -3}]]] 0.25]
#					} elseif {$N_points >= 4} {
#						## used to be >=4
#						set extension_point1 [vecscale [vecadd [lindex $point_list 0] [lindex $point_list 1] [lindex $point_list 2] ] 0.333333333]
#						set extension_point2 [vecscale [vecadd [lindex $point_list 1] [lindex $point_list 2] [lindex $point_list 3] ] 0.333333333]
#					
#						set extension_point8 [vecscale [vecadd [lindex $point_list [expr {$last_point -1}]] [lindex $point_list [expr {$last_point -2}]] [lindex $point_list [expr {$last_point -3}]]] 0.333333333]
#						set extension_point9 [vecscale [vecadd [lindex $point_list $last_point] [lindex $point_list [expr {$last_point -1}]] [lindex $point_list [expr {$last_point -2}]]] 0.333333333]
#					} elseif {$N_points == 3} {
#						set extension_point1 [vecscale [vecadd [lindex $point_list 0] [lindex $point_list 1]] 0.5]
#						set extension_point2 [vecscale [vecadd [lindex $point_list 1] [lindex $point_list 2]] 0.5]
#						### 1AA interdistance
#						set extension_point8 [vecscale [vecadd [lindex $point_list [expr {$last_point -1}]] [lindex $point_list [expr {$last_point -2}]]] 0.5]
#						set extension_point9 [vecscale [vecadd [lindex $point_list $last_point] [lindex $point_list [expr {$last_point -1}]]] 0.5]
#					} else 

					### Instead: very local-control of extensions. 
					### Uses 1-point-per-residue coordinates, irrespective of Save-every-Nth-AA, so allows precise extension length.
					if {$N_points >= 2} {
						## More averaging is only implemented for the start, 
						## since the end Sugeta-point is now already replaced with up to 3 averaged coordinates, where available.
						## So the end-extension is already half-averaged, since the end value is averaged, and is used here.
						## NB this only affects aesthetics - extensions are not treated as 'real' axis points, 
						## and their axis curvature is not calculated (they make up the linear part of the angle vs resid graph)
						if {$N_points > 2} {
							set extension_point1 [vecscale [vecadd [lindex $point_list 0] [lindex $point_list 1]] 0.5]
							set extension_point2 [vecscale [vecadd [lindex $point_list 1] [lindex $point_list 2]] 0.5]
						} else {
							set extension_point1 [lindex $point_list 0]
							set extension_point2 [lindex $point_list 1]
						}
						set extension_point8 [lindex $point_list [expr {$last_point -1}]]
						set extension_point9 [lindex $point_list $last_point]
					}

					#### Extend upwards..
					set extension_vector_upwards_direction [vecsub $extension_point1 $extension_point2]
					# 1-2 means change from 2 to 1, i.e. towards 1, towards the helix top.
					### Extension points are Always 1 residueL apart ..which is exactly what we need for extending the helix.
					set extension_vector_applied_to_correct_ori [vecadd $extension_vector_upwards_direction [lindex $point_list 0]]
					set list_copy $select_list
					
					set select_list [linsert $list_copy 0 $extension_vector_applied_to_correct_ori];
					#graphics top color blue; #sugeta
					#graphics top sphere $extension_vector_applied_to_correct_ori radius 0.3 resolution 20; #sugeta					

					#### ..and downwards.
					set extension_vector_downwards_direction [vecsub $extension_point9 $extension_point8]
					set extension_vector_applied_to_correct_ori [vecadd $extension_vector_downwards_direction [lindex $point_list $last_point]]
					
					### lappend the extension.
					set select_list [lappend select_list $extension_vector_applied_to_correct_ori]
					#graphics top color blue; #sugeta
					#graphics top sphere $extension_vector_applied_to_correct_ori radius 0.3 resolution 20; #sugeta
					
	
					if {$::bendix::StoreAxes == 1} {
						### Add tabbed coordinates per complete helix:
						foreach axis_coord_to_store $select_list {
							append axes_list "$axis_coord_to_store \t"
						}
						append axes_list "\n"
					}

					if {$::bendix::spline_resolution > 1} {
						
					#### HERMITE SPLINE
					# Create 'Phantom knots' to allow the last extensions above to feature in the spline.
					# This section is only applicable to points undergoing splines. 
					# Resolution 1 means no between-knots spline-points generated, 
					# only copies of the previous knot, so spline should be disallowed.
					# First create a synthetic point at the very end so that the spline works,
					# using synth end-point just made.
					# The $select_list is ok to use since we're only using the most peripheral 2 points, 
					# the extension with exactly 1 residue length between points.
					
						### Phantom knots downwards and upwards:
						set select_list_lastIndex [expr {[llength $select_list]-1}]
						set last_coord [lindex $select_list $select_list_lastIndex]
						set nexttolast_coord [lindex $select_list [expr {$select_list_lastIndex -1}]]
						set last_vector [vecsub $last_coord $nexttolast_coord]
						set last_final_point [vecadd $last_vector $last_coord]
						set list_copy $select_list
						set select_list [lappend list_copy $last_final_point]
						#graphics top color pink; #spline
						#graphics top sphere $last_final_point radius 0.4 ; #spline

						set first_coord [lindex $select_list 0]
						set second_coord [lindex $select_list 1]
						set first_vector [vecsub $first_coord $second_coord]
						set first_final_point [vecadd $first_vector $first_coord]
						set list_copy $select_list
						set select_list [linsert $list_copy 0 $first_final_point]
						#graphics top color pink; #axis
						#graphics top sphere $first_final_point radius 0.4 ; #spline
						
						#### Hermite Spline treatment of Select_list with Catmull-Rom gradient using the constant 0.5
						set select_list_lastIndex [expr {[llength $select_list] -1}]; # Whereof 2 'fake' points on either side: 1 ghost and one extension.  3?
						set CR_constant 0.5
						set spline_indices_of_Reals {}
						set spline_index 0
						
						# The last select_list index is $select_list_lastIndex. Going through them from 1 (necessarily) to, and incl, [expr $select_list_lastIndex -2]."
						for {set k 1} {$k <= [expr $select_list_lastIndex -2]} {incr k 1} {
							# point +1 - (-1)
							set m1_x [expr {[expr {[lindex [lindex $select_list [expr {$k+1}]] 0] - [lindex [lindex $select_list [expr {$k-1}]] 0]}]*$CR_constant}]
							set m1_y [expr {[expr {[lindex [lindex $select_list [expr {$k+1}]] 1] - [lindex [lindex $select_list [expr {$k-1}]] 1]}]*$CR_constant}]
							set m1_z [expr {[expr {[lindex [lindex $select_list [expr {$k+1}]] 2] - [lindex [lindex $select_list [expr {$k-1}]] 2]}]*$CR_constant}]
							
							# +2 - (0)
							set m2_x [expr {[expr {[lindex [lindex $select_list [expr {$k+2}]] 0] - [lindex [lindex $select_list $k] 0]}]*$CR_constant}]
							set m2_y [expr {[expr {[lindex [lindex $select_list [expr {$k+2}]] 1] - [lindex [lindex $select_list $k] 1]}]*$CR_constant}]
							set m2_z [expr {[expr {[lindex [lindex $select_list [expr {$k+2}]] 2] - [lindex [lindex $select_list $k] 2]}]*$CR_constant}]

							#### Adjustments to the N of spline points at the start and end of the helix:
							# Fixing half-a-turn extended point by only allowing half the number of spline points there.
							if {$::bendix::TurnResolution != 1} {
								
								### First and last points: Extensions:
								if {$k == 1 || $k == [expr $select_list_lastIndex -2]} {
									
									## Always extend by 1 residue, so however many spline points one residue should have is what the Extensions should have.
									set Resolution_wrt_ends [expr {floor([expr {$::bendix::spline_resolution * [expr {1.0/$TurnResolution_at1dp}]}])}]
									
								### The next-to-last exact point 
								} elseif {$k == [expr {$select_list_lastIndex - 3}]} {
									set Resolution_wrt_ends [expr {floor([expr {$::bendix::spline_resolution * $::bendix::Resolution_wrt_ends($j) }])}]; # Multiply by the calculated end-factor.
								} else {
								### All normal, middle points.
									set Resolution_wrt_ends [expr {floor($::bendix::spline_resolution)}]
								}
							} else {
								### All points, whether exact or extensions, have 1AA interdistance, so the same spline resolution applies everywhere.
								set Resolution_wrt_ends [expr {floor($::bendix::spline_resolution)}]
							}
							
							### The spline start is special; need save this point separately:
							if {$k == 1} {

								# Because of the way the spline function works:
																
								lappend spline_indices_of_Reals $spline_index; # Cover the first, extension-based spline point.
								set ::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$spline_index) [lindex $exact_AANs [expr $k -1]]
								lappend ::bendix::spline_coords [lindex $select_list $k]
								incr spline_index; # every time a spline point coordinate is saved.
								
								#graphics top color green; #spline
								#graphics top sphere [lindex $select_list $k] radius 0.35 ; #spline
							}
							
							if {$Resolution_wrt_ends < 1} {
								if {$Resolution_wrt_ends >= 0.5} {
									set Resolution_wrt_ends 1.0 ; #Since division by 0.0 fails, below.
								} else {
									set Resolution_wrt_ends 0
								}
							}
							
							#### Special consideration needs be made to save this next point, since the spline function won't save it (the loop will run from 1 to 0):
							if {$Resolution_wrt_ends < 1} {
								
								### Do the end-point:								 
								set ::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$spline_index) [lindex $exact_AANs $k]
								lappend ::bendix::spline_coords [lindex $select_list [expr $k + 1]]
								lappend spline_indices_of_Reals $spline_index;
								incr spline_index; # every time a spline coord is saved.
								
								### If in debugging Developer mode, draw spot:
								#graphics top color black; #spline
								#graphics top sphere [lindex $select_list [expr $k + 1]] radius 0.35; #spline
													
							} else {
							
								for {set l 1} {$l <= $Resolution_wrt_ends} {incr l 1} {
									# s from 0 to 1, in constant increase increments, the size of which depends on resolution. 
									# NB The 0th point is catered for above, at k==0.
									# E.g. Resolution 10: 1/10, 1/10, 2/10, 3/10, 4/10.. 10/10 = 0, 0.1, 0.2, 0.3 ... 1 = s (varying t from 0 to 1)
									
									set s [expr {$l/$Resolution_wrt_ends}]
									
									# The Hermite Spline Blending functions: 
									set h1 [expr {2.0*$s*$s*$s - 3.0*$s*$s +1.0}]
									set h2 [expr {-2.0*$s*$s*$s + 3.0*$s*$s}]
									set h3 [expr {$s*$s*$s - 2.0*$s*$s +$s}]
									set h4 [expr {$s*$s*$s - $s*$s}]
	
								 	set px [expr {$h1*[lindex [lindex $select_list $k] 0] + $h2*[lindex [lindex $select_list [expr {$k+1}]] 0] + $h3*$m1_x + $h4*$m2_x}] 
									set py [expr {$h1*[lindex [lindex $select_list $k] 1] + $h2*[lindex [lindex $select_list [expr {$k+1}]] 1] + $h3*$m1_y + $h4*$m2_y}] 
									set pz [expr {$h1*[lindex [lindex $select_list $k] 2] + $h2*[lindex [lindex $select_list [expr {$k+1}]] 2] + $h3*$m1_z + $h4*$m2_z}]
						
									set spline_point "$px $py $pz"
									#graphics top color red; #spline
									lappend ::bendix::spline_coords $spline_point
									
									if {$l == $Resolution_wrt_ends} {
									## Each last spline point is the 'next' real residue. If $spline_index == 0, it's already accounted for.
										# Save the index of Real splinepoints, so they can be matched to Residues in graphs:
										lappend spline_indices_of_Reals $spline_index; # What spline points to trust with real residues.
										set ::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$spline_index) [lindex $exact_AANs $k]
										#graphics top color black; #spline
									}
									#graphics top sphere "$px $py $pz" radius 0.35 ;#spline 
									incr spline_index; # Keep a counter ticking that increases each time a new spline coordinate is added.
								}
							}
						}
						set this_helix_last_splinePoint_index [expr $spline_index -1]
					} else {
					#### If spline_resolution == 1, (one point per Sugeta-calculated axis point) then the axis IS the Sugeta points. 
					#### Don't spline, just use the select_list as coordinates.
					
						if {$::bendix::spline_coords == ""} {
							set ::bendix::spline_coords $select_list
						} else {
							append ::bendix::spline_coords " " $select_list
						}
						## Create a list with indices of the Real indices (all indices but the first and last, which are Extension points):
						set this_helix_last_splinePoint_index [expr {[llength $select_list] -1}];
						
						## Remove Extension-points
						set spline_indices_of_Reals {}
						for {set in 0} {$in <= $this_helix_last_splinePoint_index} {incr in 1} {
							set ::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$in) [lindex $exact_AANs $in]
							lappend spline_indices_of_Reals $in; # not necessary in the end
						}
					}

					set this_helix_last_splinePoint_index [expr {[llength $::bendix::spline_coords] -1}]

					#### Collect spline-coordinates so that bendix can run without the spline-coordinate loop, where possible
					#### Normalise select_list_lastIndex's: [6 10 2] --> [0 6 7 16 17 18]
					if {$::bendix::spline_startsandstops == ""} {
						lappend ::bendix::spline_startsandstops 0;
					}
					lappend ::bendix::spline_startsandstops $this_helix_last_splinePoint_index;
					lappend ::bendix::spline_startsandstops [expr {$this_helix_last_splinePoint_index +1}];
					
					unset counter i N_points last_point extension_point1 extension_point2 extension_point8 extension_point9
					unset extension_vector_upwards_direction extension_vector_downwards_direction list_copy

				# End of large-helix loop.
				
				### Small, 4-residue helices
				} elseif {$N_particles_in_this_helix == 4} {
					set i [lindex $::bendix::normalised_startsandstops $j]
						
					#### Generate the helix axis according to Sugeta and Miyazawa '67
					#### Two points are generated per 4-residue sliding window,
					#### so a 4-residue helix is populated by a start and end axis point.
					
					# Vectors that join C alphas:
					set vec12 [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] [lindex $::bendix::xyz $i]]
					set vec23 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] [lindex $::bendix::xyz [expr {$i + 1}]]]
					set vec34 [vecsub [lindex $::bendix::xyz [expr {$i + 3}]] [lindex $::bendix::xyz [expr {$i + 2}]]]
						
					set dv13 [vecsub $vec12 $vec23] 
					set dv24 [vecsub $vec23 $vec34]
						
					set cross_product [veccross $dv13 $dv24]
					set cross_product [vecnorm $cross_product]
						
					set dmag [veclength $dv13]
					set emag [veclength $dv24]
					set dot_product [vecdot $dv13 $dv24]
					set dmag_times_emag [expr $dmag*$emag]
					set costheta [expr $dot_product / $dmag_times_emag]
						
					set costheta1 [expr 2.0*[expr 1.0 - $costheta]]
					set radmag [expr sqrt($dmag_times_emag)/ $costheta1]
						
					set dv13 [vecnorm $dv13]
					set dv24 [vecnorm $dv24]
						
					set rad [vecscale $dv13 $radmag]
					set rad2 [vecscale $dv24 $radmag]
						
					set result_point [vecsub [lindex $::bendix::xyz [expr {$i + 1}]] $rad]
					set result_point2 [vecsub [lindex $::bendix::xyz [expr {$i + 2}]] $rad2]
					
					#graphics top sphere $result_point radius 0.35; #spline
					#graphics top sphere $result_point2 radius 0.35; #spline
					
					set ::bendix::xyz [lreplace $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j] [lindex $::bendix::normalised_startsandstops $j] $result_point]
					set ::bendix::xyz [lreplace $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr $j+1]] [lindex $::bendix::normalised_startsandstops [expr $j+1]] $result_point2]
					
					
				}
				unset point_list select_list N_particles_in_this_helix
				incr helix_counter
				
			}; # End of loop per helix, irrespective of length.
			set ::bendix::spline_startsandstops [lreplace $::bendix::spline_startsandstops end end]
			#### Clean-up
			unset helix_list_length helixcount splineNs spline_list ID_helix

		}; # SPLINE loop module ends. Next I draw using these coordinates. 

################################################################################
#                                    DRAW                                      #
################################################################################

		#### Fix Material
		set material_choice [.bex.lf2.listbox get]
		if {$material_choice == ""} {
			graphics $::bendix::proteinID material EdgyShiny
		} else {
			graphics $::bendix::proteinID material $material_choice
		}
		
		#### Fix Heatmap colour scale
		set scale_choice [.bex.lf2.field5.listbox get]
		if {$scale_choice == ""} {
			color scale method RGB
		} else {
			color scale method $scale_choice
		}
		
		global vmd_frame
		set ::bendix::frame $vmd_frame($::bendix::proteinID)
		lappend ::bendix::angle_per_AA_per_frame_BARRED $::bendix::frame
		set ::bendix::curvature_graph_Y {}
		set ::bendix::curvature_graph_X {}
		set ::bendix::points_to_graph {};
		set ::bendix::realAAs_for_23D_graph_Xaxis {}
		set ::bendix::index_of_helix_assignment_used_for_angle "";
		set helix_counter 0
		set helixcount 0				
		set counter 0
		set helix_list_length [llength $::bendix::normalised_startsandstops]
		set h 0
		set startDone 1.0
		set N_of_angleValues_in_this_helix 0
		set effectiveAngleSide [expr $::bendix::AngleSide + 1]; # Because the actual N of C-alphas in a 3.6 AA helix turn is 4.6.
		
		#### Conversion of things measured in residues - the angle Side and every N residues - to the Bendix currency: spline points.
		# s.p. per select AA * every N real AAs to select
		set N_splinePoints_per_RealAA [expr $::bendix::spline_resolution * [expr 1.0/$::bendix::TurnResolution]]
		
		### Side with length 4AA is Actually between 5AA. So editing this:
		set angleSide_in_splinePoints [expr {floor([expr $effectiveAngleSide * $N_splinePoints_per_RealAA])}]; 
		set second_angleSide_in_splinePoints [expr {floor([expr [expr $effectiveAngleSide -1] * $N_splinePoints_per_RealAA])}]; 
		set total_angleSide_in_splinePoints [expr [expr [expr [expr $effectiveAngleSide*2]-1] * $N_splinePoints_per_RealAA] - [expr $N_splinePoints_per_RealAA]]
		set total_angleSide_in_splinePoints [expr {floor($total_angleSide_in_splinePoints)}]
			
		## Use floor since the user should be able to say whether her helix is measurable using a certain Side. Side already needs more points than advertised; use floor.
		set angleSide_in_splinePoints [expr int($angleSide_in_splinePoints)]
		set second_angleSide_in_splinePoints [expr int($second_angleSide_in_splinePoints)]
		set total_angleSide_in_splinePoints [expr int($total_angleSide_in_splinePoints)]
		
		### but to hit the rigth INDEX, I need it to be one less: points 12345 have indices 01234.
		set angleSide_as_splinePointIndex [expr $angleSide_in_splinePoints -1];
		set second_angleSide_as_splinePointIndex [expr $second_angleSide_in_splinePoints -1];
		set total_angleSide_in_splinePointIndex [expr $total_angleSide_in_splinePoints]
		
		### Test that the last angle index is even, e.g. 01234 gives an angle centered about 2, and needs an even last indexN
		set maybe_odd_last_indexN [expr {$total_angleSide_in_splinePointIndex%2}]
		if {$maybe_odd_last_indexN == 1} {
			### Odd last angle index = no certain angle apex, so lessen index by one to give a clear apex:
			set total_angleSide_in_splinePointIndex [expr $total_angleSide_in_splinePointIndex -1]
		}
		
		set total_splinePointIndex_reqPerAngle $total_angleSide_in_splinePointIndex
		set angleSide_as_splinePointIndex [expr $total_angleSide_in_splinePointIndex/2]
		
		## Catch resolution1 with high Side and TurnResolution
		if {$angleSide_as_splinePointIndex < 1} {
			set angleSide_as_splinePointIndex 1
			set second_angleSide_as_splinePointIndex 1
		}
		

		#### Loop per helix:
		for {set j 0} {$j <= [expr {$helix_list_length - 2}]} {incr j 2} {

			set N_of_angleValues_in_this_helix 0
			set N_particles_in_this_helix [expr {[lindex $::bendix::normalised_startsandstops [expr {$j + 1}]] - [lindex $::bendix::normalised_startsandstops $j] +1 }]
			
			### Reality check: is this enough AAs to fit an angle with Side?
			if {$N_particles_in_this_helix >= [expr $effectiveAngleSide *2] && $::bendix::TurnResolution < $N_particles_in_this_helix} {
				set long_enough_helix_for_Sidex2 1
			} else {
				set long_enough_helix_for_Sidex2 0
			}
					
			### The Sugeta algorithm requires 4+ points, else no spline points; just straight coords.
			if {$N_particles_in_this_helix > 4} {
				set spline_list {}
				for {set counting [lindex $::bendix::spline_startsandstops $h]} {$counting <= [lindex $::bendix::spline_startsandstops [expr {$h + 1}]]} {incr counting} {
					lappend spline_list [lindex $::bendix::spline_coords $counting]
				}
				set this_helix_last_splinePoint_index [expr {[llength $spline_list] -1}]; # to be matched with index of splinePoints required for angle.
				
				incr h 2
				
				if {$::bendix::uniform_colour == 0 && $this_helix_last_splinePoint_index >= $total_splinePointIndex_reqPerAngle && $long_enough_helix_for_Sidex2 == 1} {
					lappend ::bendix::index_of_helix_assignment_used_for_angle $j; #Collect indices of amino acid numbers for Multiplot-legend access.
					#### Colour by ANGLE, independently of what scale is used.
					## If the N of splinePoints are larger than the N required to calculate the angle:
				
						set last_angleUsable_splinePoint [expr $this_helix_last_splinePoint_index - $angleSide_as_splinePointIndex]
						####Middle by k:s:
						
						for {set k $angleSide_as_splinePointIndex} {$k <= $last_angleUsable_splinePoint} {incr k 1} {
							# Calculate the angle
							set helix_angle [::bendix::angle [lindex $spline_list [expr {$k - $angleSide_as_splinePointIndex}]] [lindex $spline_list $k] [lindex $spline_list [expr {$k + $angleSide_as_splinePointIndex}]] ]
							
							#### Adjust spurious angles:
							if {$helix_angle > 90} {
								set helix_angle [expr {180-$helix_angle}]
							}							
							
							#### START!
							# If it's the first calculated point, add y-values from 0 to this value for spline points with index 0 to [expr $angleSide_as_splinePointIndex -1]
							if {$k == $angleSide_as_splinePointIndex} {
								
								## Draw the first sphere in blue, per def.
								graphics $::bendix::proteinID color 1000
																
								graphics $::bendix::proteinID sphere [lindex $spline_list [expr {$k - $angleSide_as_splinePointIndex}]] radius $::bendix::helix_radius resolution 20; #draw
								lappend ::bendix::curvature_graph_X 1.0
								if {$::bendix::StoreData == 1} {
									### Store realiable (real) AA Ns for reproduction in the graph
									lappend ::bendix::realAAs_for_23D_graph_Xaxis 1
								}								
								
								set Yvalues_that_need_Xvalues -1
								set partitionedAngle [expr $helix_angle/$angleSide_as_splinePointIndex];
								for {set helixStartBit 0} {$helixStartBit < $angleSide_as_splinePointIndex} {incr helixStartBit} {
									set partitionedAngle_bit [expr $partitionedAngle * $helixStartBit]
									lappend ::bendix::curvature_graph_Y $partitionedAngle_bit
									if {$::bendix::StoreData == 1} {
										lappend ::bendix::angle_per_AA_per_frame_BARRED $partitionedAngle_bit
									}
									incr N_of_angleValues_in_this_helix
									incr Yvalues_that_need_Xvalues
									
									
									#### Per angle evaluated, draw:
									set scaled_angle [expr {$partitionedAngle_bit/$::bendix::angle_max}]
									if {$scaled_angle < 0.9375 } {
										set colour_value [expr {round(1000-$scaled_angle*1000)}];
									} else {
										set colour_value 33
									}
									graphics $::bendix::proteinID color $colour_value; #draw

									#### Draw primitives:
									# A cylinder ontop the last sphere:
									graphics $::bendix::proteinID cylinder [lindex $spline_list $helixStartBit] [lindex $spline_list [expr {$helixStartBit+1}]] radius $::bendix::helix_radius resolution 30; #draw
									# ..And a ball in the same colour, unless last (= 0-angle coloured, per definition.)
									graphics $::bendix::proteinID sphere [lindex $spline_list [expr {$helixStartBit+1}]] radius $::bendix::helix_radius resolution 20; #draw
									
								}
							}
							
							
							##### Done for each new k: store the angle
							# Store these angles in a growing list.
							lappend ::bendix::curvature_graph_Y $helix_angle
							incr N_of_angleValues_in_this_helix
							### If the user has chosen to save dynamic data, store it.
							if {$::bendix::StoreData == 1} {
								lappend ::bendix::angle_per_AA_per_frame_BARRED $helix_angle
							}
							incr Yvalues_that_need_Xvalues
							
							
							##### If it isn't end, draw a cylinder & sphere:
							if {$k != $last_angleUsable_splinePoint} {
								
								#### Per angle evaluated, draw:
								set scaled_angle [expr {$helix_angle/$::bendix::angle_max}]
								if {$scaled_angle < 0.9375 } {
									set colour_value [expr {round(1000-$scaled_angle*1000)}];
								} else {
									set colour_value 33
								}
								graphics $::bendix::proteinID color $colour_value; #draw
								#graphics $::bendix::proteinID color red; #drawControl
	
								#### Draw primitives:
								# A cylinder ontop the last sphere:
								graphics $::bendix::proteinID cylinder [lindex $spline_list $k] [lindex $spline_list [expr {$k+1}]] radius $::bendix::helix_radius resolution 30; #draw
								# ..And a ball in the same colour, unless last (= 0-angle coloured, per definition.)

								### End sphere:
								graphics $::bendix::proteinID sphere [lindex $spline_list [expr {$k+1}]] radius $::bendix::helix_radius resolution 20; #draw
							}


							#### A REAL RESIDUE NUMBER DETECTED!
							# If the spline point is a Real Residue (i.e. it corresponds to a residue loci, as defined by 'Use every N residues' in the GUI), 
							# store the real Residue, including the ones leading up to it, for the x-axis, to indicate precision at this point. Include the last, 'this', point.
							# Surrounding angle-values for residues are approximate due to their being spline-approximations, rather than due to an axis control-point.
							
							if {[info exists ::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$k)]} {
										
								if {$::bendix::StoreData == 1} {
									### Store realiable (real) AA Ns for reproduction in the graph
									lappend ::bendix::realAAs_for_23D_graph_Xaxis $::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$k)
								}
													
								## Adjust for the sixe of the x-axis value that needs populating. 
								## E.g. if the current AA for x is 6 and the last AA covered was 4, the difference is used, linearly, for unpopulated Y-values.
								set targetX [expr $::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$k) - $startDone]
													
								# ID the N of Y-values without X-values
								set Xpartition [expr $targetX/[expr 1.0 * $Yvalues_that_need_Xvalues]]
																
								## Partition the Y-values linearly unto the available X-axis.
								for {set xes 1} {$xes <= $Yvalues_that_need_Xvalues} {incr xes} {
									lappend ::bendix::curvature_graph_X [expr $startDone + [expr $xes * $Xpartition]]
								}
																								
								## Update startDone with the new, latest X-value
								set startDone $::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$k)								
								set Yvalues_that_need_Xvalues 0
							}
					
							###### For THE HELIX END:
							if {$k == $last_angleUsable_splinePoint} {
								# You have the AA N for the very last spline point. It's quoted as a Real.
								
								if {$::bendix::StoreData == 1} {
									### Store realiable (real) AA Ns for reproduction in the graph
									lappend ::bendix::realAAs_for_23D_graph_Xaxis $::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$this_helix_last_splinePoint_index)
								}
								
								# Check whether any X-values are needed (if the last one wasn't a Real)
								# I just did extrapolation of the x-axis to the last point.
								# Just use the last (x,y) value as your starting point.
								
								if {$Yvalues_that_need_Xvalues == 0} {
									# Last noted X number (for which a Y-value already exists) was $startDone.
									# The y-value is $helix_angle
									# Last X in this helix is $::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$this_helix_last_splinePoint_index), with y-value 0.  
									
									# Compute decreasing values from last angle calculated:
									set partitionedY [expr $helix_angle/$angleSide_as_splinePointIndex]
									set partitionedX [expr [expr {$::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$this_helix_last_splinePoint_index) - $startDone}]/[expr $angleSide_as_splinePointIndex *1.0]]
									
									set Xcounter 1
									for {set helixEndBit [expr $angleSide_as_splinePointIndex - 1]} {$helixEndBit >= 0} {incr helixEndBit -1} {
										
										### Y-values decreasing from $helix_angle to 0
										set current_partitionedY [expr $partitionedY * $helixEndBit];
										lappend ::bendix::curvature_graph_Y $current_partitionedY
										if {$::bendix::StoreData == 1} {
											lappend ::bendix::angle_per_AA_per_frame_BARRED $current_partitionedY
										}
										incr N_of_angleValues_in_this_helix
										
										### X-values increasing from $startDone to $::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$this_helix_last_splinePoint_index)
										lappend ::bendix::curvature_graph_X [expr $startDone + $partitionedX*$Xcounter]
										incr Xcounter
										
										#### Per angle evaluated, draw:
										set scaled_angle [expr {$current_partitionedY/$::bendix::angle_max}]
										if {$scaled_angle < 0.9375 } {
											set colour_value [expr {round(1000-$scaled_angle*1000)}];
										} else {
											set colour_value 33
										}
										
										graphics $::bendix::proteinID color $colour_value
										
										#### Draw primitives:
										# A cylinder ontop the last sphere: Counter-2 means between k+0 and k+1 for the first cylinder.
										graphics $::bendix::proteinID cylinder [lindex $spline_list [expr $k + [expr $Xcounter -2]]] [lindex $spline_list [expr {$k + [expr $Xcounter -1]}]] radius $::bendix::helix_radius resolution 30; #draw
										# ..And a ball in the same colour:
										graphics $::bendix::proteinID sphere [lindex $spline_list [expr {$k + [expr $Xcounter -1]}]] radius $::bendix::helix_radius resolution 20; #draw
									}
								}
								
								if {$Yvalues_that_need_Xvalues > 0} {
									# Last noted X number (for which a Y-value already exists) was $startDone.
									# The y-value is $helix_angle
									# Last X in this helix is $::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$this_helix_last_splinePoint_index), with y-value 0.
									
									# Y: generate decreasing values from last value calculated.
									set partitionedY [expr $helix_angle/$angleSide_as_splinePointIndex]
									set counter 0
									for {set helixEndBit [expr $angleSide_as_splinePointIndex - 1]} {$helixEndBit >= 0} {incr helixEndBit -1} {
										
										### Y-values decreasing from $helix_angle to 0
										set current_partitionedY [expr $partitionedY * $helixEndBit ]
										lappend ::bendix::curvature_graph_Y $current_partitionedY
										if {$::bendix::StoreData == 1} {
											lappend ::bendix::angle_per_AA_per_frame_BARRED $current_partitionedY
										}
										incr N_of_angleValues_in_this_helix
										
										#### Per angle evaluated, draw:
										set scaled_angle [expr {$current_partitionedY/$::bendix::angle_max}]
										if {$scaled_angle < 0.9375 } {
											set colour_value [expr {round(1000-$scaled_angle*1000)}];
										} else {
											set colour_value 33
										}
										
										graphics $::bendix::proteinID color $colour_value; #draw
										#graphics $::bendix::proteinID color blue; #drawControl
										
										#### Draw primitives:
										# A cylinder ontop the last sphere:
										graphics $::bendix::proteinID cylinder [lindex $spline_list [expr $k + $counter]] [lindex $spline_list [expr {$k + [expr {$counter + 1}]}]] radius $::bendix::helix_radius resolution 30; #draw
										# ..And a ball in the same colour:
										graphics $::bendix::proteinID sphere [lindex $spline_list [expr {$k + [expr $counter + 1]}]] radius $::bendix::helix_radius resolution 20; #draw
										incr counter
									}
									
									# X: do increasing values from last calculated.
									set N_XpointsNeeded [expr $angleSide_as_splinePointIndex + $Yvalues_that_need_Xvalues]
									set partitionedX [expr [expr {$::bendix::AAN_associated_w_splineIndex($::bendix::proteinID,$j,$this_helix_last_splinePoint_index) - $startDone}]/ [expr 1.0*$N_XpointsNeeded]]
									
									for {set lastXSide 1} {$lastXSide <= $N_XpointsNeeded} {incr lastXSide 1} {
										### Y-values decreasing from $helix_angle to 0
										set current_partitionedX [expr $partitionedX*$lastXSide]
										lappend ::bendix::curvature_graph_X [expr $startDone + $current_partitionedX]
									}						
								}
							}
						}
						#### Iff qualifying helix AND dynamic data is being stored, ID maximum angle in this helix:
						# For each helix, angles between adjacent turns are measured. These angles are lappended to $::bendix::curvature_graph_Y
						# To get the highest angle per helix (not just the entire $::bendix::curvature_graph_Y, which may contain multiple helices), 
						# I calculate what part of $::bendix::curvature_graph_Y is relevant and take the highest of these numbers.
						if {$::bendix::StoreData == 1} {
							set winner_angle 0
							set N_gathered_angles_so_far [llength $::bendix::curvature_graph_Y]
		
							# The current $N_of_angleValues_in_this_helix is the number of angle values that have been evaluated for the current helix. 
							# It shows me how many numbers to use from the end of $::bendix::curvature_graph_Y.
							# NB that it does not contain the last angles at the end of the helix, whose values decend to give zero at the end. 
							# These angles are of no interest to the MaxAngle calculation, so are left out to speed up the algorithm.
							for {set maxAng [expr {$N_gathered_angles_so_far - $N_of_angleValues_in_this_helix}]} {$maxAng <$N_gathered_angles_so_far} {incr maxAng} {
								if {[lindex $::bendix::curvature_graph_Y $maxAng] > $winner_angle} {
									set winner_angle [lindex $::bendix::curvature_graph_Y $maxAng]
								}
							}
							lappend ::bendix::maximumAngle_per_frame $winner_angle
							global vmd_frame
							set ::bendix::frame $vmd_frame($::bendix::proteinID)
							lappend ::bendix::vmdFrame $::bendix::frame
						}
					
					# Make it an integer.
					lappend ::bendix::points_to_graph [expr $N_of_angleValues_in_this_helix -1];
					
				} elseif {$::bendix::uniform_colour == 1 || $this_helix_last_splinePoint_index < $total_splinePointIndex_reqPerAngle || $long_enough_helix_for_Sidex2 == 0} {
				#### Uniform colour:
				
					if {$::bendix::uniform_colour == 0 } {
					### Left-over angle-coloured helices that are too small for angle-colouring	
						graphics $::bendix::proteinID color 1000
					
					} elseif {$::bendix::uniform_colour_type == 0} {
					#### Colour full protein
						if {$::bendix::autoColour == 1} {
							graphics $::bendix::proteinID color 0
						} else {
							set user_chosen_colours [split $::bendix::input_colour]
							if {[lindex $user_chosen_colours 0] != ""} {		
								graphics $::bendix::proteinID color [lindex $user_chosen_colours 0]
							} else {
								graphics $::bendix::proteinID color white
							}
						}
					} elseif {$::bendix::uniform_colour_type == 1} {
					#### Colour by chain
						if {$::bendix::startIndex != ""} {
							set current_index [lindex $::bendix::startIndex $helix_counter]
							set atomselect_index [atomselect $::bendix::proteinID "index $current_index"]
							set current_chain [$atomselect_index get chain];
							$atomselect_index delete
							if {$::bendix::autoColour == 1} {
								if {$helix_counter == 0} {
									set chain_colour 0
									set chain_letter_list $current_chain
								} else {
									if {[lindex $chain_letter_list $chain_colour] != $current_chain} {
										incr chain_colour
										set chain_letter_list [lappend chain_letter_list $current_chain]
									}
								}
							} else {
								#### User-defined chain-colouring.
								set user_chosen_colours [split $::bendix::input_colour]
								if {$helix_counter == 0} {
									if {[lindex $user_chosen_colours 0] != ""} {
										set chain_colour [lindex $user_chosen_colours 0]
									} else {
										set chain_colour white
									}
									set chain_letter_list $current_chain
									incr N_chains
								} else {
									if {[lindex $chain_letter_list [expr {$N_chains -1}]] != $current_chain} {
										set chain_letter_list [lappend chain_letter_list $current_chain]
										if {[lindex $user_chosen_colours $N_chains] != ""} {
											set chain_colour [lindex $user_chosen_colours $N_chains]
										} else {
											set chain_colour white
										}
										incr N_chains
									}
								}
							}
						} else {
						#### Colouring problem results in uniform, entire protein colouring:
							if {$::bendix::autoColour == 1} {
								set chain_colour 0
							} else {
								set user_chosen_colours [split $::bendix::input_colour]
								set chain_colour [lindex $user_chosen_colours 0]
							}
							graphics $::bendix::proteinID color $chain_colour
						}
						graphics $::bendix::proteinID color $chain_colour
					} elseif {$::bendix::uniform_colour_type == 2} {
					#### Colour by Helix
						if {$::bendix::autoColour== 1} {
							if {$helixcount >= 32} {
								set helixcount 0
							}
							graphics $::bendix::proteinID color $helixcount
						} else {
							set user_chosen_colours [split $::bendix::input_colour]
							if {[lindex $user_chosen_colours $helixcount] != ""} {
								set helixcolour [lindex $user_chosen_colours $helixcount]
							} else {
								set helixcolour white
							}
							graphics $::bendix::proteinID color $helixcolour
						}
					}
					for {set k 0} {$k < $this_helix_last_splinePoint_index} {incr k 1} {
					# For all spline lists with at least 2 spline points in them, as required by the > 4 particles loop which contains this loop.
						if {$k == 0} {
							# Only cap start and end by a sphere once, to save on drawn graphics primitives to screen:
							graphics $::bendix::proteinID sphere [lindex $spline_list $k] radius $::bendix::helix_radius resolution 20; #draw
							graphics $::bendix::proteinID cylinder [lindex $spline_list $k] [lindex $spline_list [expr {$k+1}]] radius $::bendix::helix_radius resolution 30; #draw
							graphics $::bendix::proteinID sphere [lindex $spline_list [expr {$k+1}]] radius $::bendix::helix_radius resolution 20; #draw
						} else {
							graphics $::bendix::proteinID cylinder [lindex $spline_list $k] [lindex $spline_list [expr {$k+1}]] radius $::bendix::helix_radius resolution 30; #draw
 							graphics $::bendix::proteinID sphere [lindex $spline_list [expr {$k+1}]] radius $::bendix::helix_radius resolution 20; #draw
 						}
					}
				}
				# For drawing 'Join' backbone in the future: store coordinates of helices' starts and stops
				lappend ::bendix::helix_coord_at_startNstop [lindex $spline_list 0]
				lappend ::bendix::helix_coord_at_startNstop [lindex $spline_list $this_helix_last_splinePoint_index]
				
								
			} else {
			#### Small helices: cylinder between start and end coordinate.
				if {$::bendix::uniform_colour == 0} {
				#### Colour by zero-angle
					set scaled_angle [expr {0/50}]
					set colour_value [expr {round(1000-$scaled_angle*1000)}]
					graphics $::bendix::proteinID color $colour_value

				} elseif {$::bendix::uniform_colour == 1} {
					if {$::bendix::uniform_colour_type == 0} {
					#### Colour entire protein
						if {$::bendix::autoColour== 1} {
							graphics $::bendix::proteinID color 0
						} else {
							set user_chosen_colours [split $::bendix::input_colour]
							if {[lindex $user_chosen_colours 0] != ""} {
								graphics $::bendix::proteinID color [lindex $user_chosen_colours 0]
							} else {
								graphics $::bendix::proteinID color white
							}
						}
					} elseif {$::bendix::uniform_colour_type == 1} {
						if {$::bendix::startIndex != ""} {
							set current_index [lindex $::bendix::startIndex $helix_counter]
							set atomselect_index [atomselect $::bendix::proteinID "index $current_index"]
							set current_chain [$atomselect_index get chain]
							$atomselect_index delete
							if {$::bendix::autoColour == 1} {
								if {$helix_counter == 0} {
									set chain_colour 0
									set chain_letter_list $current_chain
								} else {
									if {[lindex $chain_letter_list $chain_colour] != $current_chain} {
										incr chain_colour
										set chain_letter_list [lappend chain_letter_list $current_chain]
									}
								}
							} else {
								set user_chosen_colours [split $::bendix::input_colour]
								if {$helix_counter == 0} {
									if {[lindex $user_chosen_colours 0] != ""} {
										set chain_colour [lindex $user_chosen_colours 0]
									} else {
										set chain_colour white
									}
									set chain_letter_list $current_chain
									incr N_chains
								} else {
									if {[lindex $chain_letter_list [expr {$N_chains -1}]] != $current_chain} {
										set chain_letter_list [lappend chain_letter_list $current_chain]
										if {[lindex $user_chosen_colours $N_chains] != ""} {
											set chain_colour [lindex $user_chosen_colours $N_chains]
										} else {
											set chain_colour white
										}
										incr N_chains
									}
								}
							}
						} else {
						#### Problematic colouring is solved by uniform, entire molecule colouring
							if {$::bendix::autoColour == 1} {
								set chain_colour 0
							} else {
								set user_chosen_colours [split $::bendix::input_colour]
								set chain_colour [lindex $user_chosen_colours 0]
							}
						}
						graphics $::bendix::proteinID color $chain_colour
					} elseif {$::bendix::uniform_colour_type == 2} {
						#### Colour by helix
						if {$::bendix::autoColour== 1} {
							if {$helixcount >= 32} {
								set helixcount 0
							}
							graphics $::bendix::proteinID color $helixcount
						} else {
							set user_chosen_colours [split $::bendix::input_colour]
							if {[lindex $user_chosen_colours $helixcount] != ""} {
								set helixcolour [lindex $user_chosen_colours $helixcount]
							} else {
								set helixcolour white
							}
							graphics $::bendix::proteinID color $helixcolour
						}
					}
				}
				
				if {$N_particles_in_this_helix == 4} {
					### Draw a cylinder between 1st and last coordinates, as determined by the Sugeta algorithm
					graphics $::bendix::proteinID sphere [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] radius $::bendix::helix_radius resolution 20; #draw
					graphics $::bendix::proteinID cylinder [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] radius $::bendix::helix_radius resolution 30; #draw
					graphics $::bendix::proteinID sphere [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] radius $::bendix::helix_radius resolution 20; #draw
					
					# For drawing 'Join' backbone in the future: store coordinates of helices' starts and stops
					lappend ::bendix::helix_coord_at_startNstop [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]]
					lappend ::bendix::helix_coord_at_startNstop [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]]
					# 4-AA helix detected, and 1 AA apart Ns were added to the helix_coord_at_startNstop.
					
				} elseif {$N_particles_in_this_helix == 3 || $N_particles_in_this_helix == 2} {
					### Draw a sphere in the middle of the 1st and last coordinate
					graphics $::bendix::proteinID sphere [vecscale [vecadd [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] ] 0.5] radius $::bendix::helix_radius resolution 20; #draw
					lappend ::bendix::helix_coord_at_startNstop [vecscale [vecadd [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] ] 0.5]
					lappend ::bendix::helix_coord_at_startNstop [vecscale [vecadd [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops [expr {$j + 1}]]] ] 0.5]
					# 2 or 3-AA helix detected, and the same AA N was added to the helix_coord_at_startNstop, twice.
				} else {
					### But surely there are no 1-residue helices..? Just in case, a sphere here:
					graphics $::bendix::proteinID sphere [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] radius $::bendix::helix_radius resolution 20; #draw
					
					# For drawing 'Join' backbone in the future: store coordinates of helices' starts and stops
					lappend ::bendix::helix_coord_at_startNstop [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]]
					lappend ::bendix::helix_coord_at_startNstop [lindex $::bendix::xyz [lindex $::bendix::normalised_startsandstops $j]] 
					# 1-AA helix detected, and the same AA N was added to the helix_coord_at_startNstop, twice.
				}
			}
			#### All above is done wrt 1 helix, per frame. Prepare variables wrt the next helix:
			incr helix_counter
			incr helixcount
			set spline_list {}
			set startDone 1.0

			### If the user has chosen to save dynamic data, add a newline to the file for usability.
			if {$::bendix::StoreData == 1} {
				append ::bendix::angle_per_AA_per_frame_BARRED "\n"
				if {$::bendix::StoreAxes == 1} {
				}
			}
			set N_of_angleValues_in_this_helix 0
		}
		set ::bendix::previous_spline_resolution $::bendix::spline_resolution
		set ::bendix::previous_TurnResolution $::bendix::TurnResolution
		set ::bendix::previous_helix_type 1
		set ::bendix::helixNumber [expr {[llength $::bendix::helix_assignment] /2}]
	}

################################################################################
#                               DRAW NON-HELIX                                 #
################################################################################

	if {$::bendix::CG == 0 } {
		if {$::bendix::slow_and_pretty == 1} {
			::bendix::drawBackbone
		} elseif {$::bendix::autoCartoon == 1 || $::bendix::autoNewCartoon == 1 || $::bendix::autoTube == 1} {
			::bendix::cartoonify
		}
	} else {
		if { $::bendix::quick_and_dirty == 1 || $::bendix::slow_and_pretty == 1} {
			::bendix::drawBackbone
		}
	}
	#### Un/hide rep0 according to user's GUI settings:
	::bendix::hiderep_zero
	
################################################################################
#                            SET VARs FOR DRAWING                              #
################################################################################

	set ::bendix::tested_alt_particle_for_empty_select_AAs 0
	set ::bendix::tested_alt_particle_for_empty_start_and_end_AA_N_of_chains 0
	set ::bendix::tested_alt_particle_for_inexistent_start_and_end_AA_N_of_chains 0
	set ::bendix::tested_alt_particle_for_empty_residueNs 0
	set ::bendix::tested_alt_particle_for_empty_startIndex 0
	set ::bendix::previous_proteinID $::bendix::proteinID
	set ::bendix::previous_subset $::bendix::subset
	set ::bendix::previous_helix_assignment $::bendix::helix_assignment
	set ::bendix::previous_frame $::bendix::frame
	set ::bendix::previous_particle_name $::bendix::particle_name
	if {$::bendix::StoreData == 1} {
		lappend ::bendix::angle_per_turn_per_frame $::bendix::curvature_graph_Y
	}
	if {$::bendix::StoreAxes == 1} {
		if {[info exists axes_list]} {
			append ::bendix::helix_axis_per_helix_per_frame_BARRED "$axes_list"
		}
	}
  }
  }
}

# bendix_tk --------------------------------------------------------------------
#    Necessary to load bendix via VMD startup
# ------------------------------------------------------------------------------
proc bendix_tk {} {
	set N_loaded_Mols [molinfo num]
	if {$N_loaded_Mols == 0} {
		catch {destroy .pop_load}
		toplevel .pop_load
		wm geometry .pop_load +200+300
		#grab .pop_load
		wm title .pop_load "Welcome!"
		
		wm protocol .pop_load WM_DELETE_WINDOW {
			destroy .pop_load
			return
		}
		message .pop_load.msg1 -width 400 -text "To use bendix,\nplease load a molecule." -padx 35 -pady 15
		button .pop_load.okb -text OK -background green -command {destroy .pop_load ; return }

		pack .pop_load.msg1 -side top
		pack .pop_load.okb -side bottom -pady 5
	
		::bendix::bendix
		# Disable Rep0 button
		.bex.lf3.field1.rep0 configure -state disable
		return $::bendix::bix
	} else {
		::bendix::bendix
		return $::bendix::bix
	}
}

#### Launch the bendix GUI from the Tk Console by sourcing it (but makes Bendix start the moment VMD startups, so is disabled here)
#bendix_tk
