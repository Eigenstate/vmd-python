############################################################################
#cr                                                                       
#cr            (C) Copyright 1995-2007 The Board of Trustees of the            
#cr                        University of Illinois                         
#cr                         All Rights Reserved                           
#cr                                                                       
############################################################################

############################################################################
# RCS INFORMATION:
#
# 	$RCSfile: loadplugins.tcl,v $
# 	$Author: johns $	$Locker:  $		$State: Exp $
#	$Revision: 1.52 $	$Date: 2016/11/21 23:07:07 $
#
############################################################################
# DESCRIPTION:
#   install the "core" vmd/tcl procedures and variables
#
############################################################################

# This is part of the VMD installation.
# For more information about VMD, see http://www.ks.uiuc.edu/Research/vmd

#######################################
# Add the plugins directories (plugins/$ARCH/tcl and plugins/noarch/tcl) to
# the Tcl package search path.
lappend auto_path [file join $env(VMDDIR) plugins [vmdinfo arch] tcl]
lappend auto_path [file join $env(VMDDIR) plugins noarch tcl]


#######################################
# Add plugin directories (plugins/$ARCH/python/<pkg> and plugins/noarch/python/<pkg>) 
# to the Python package search path.  The path list is semicolon-delimited on
# Windows, and colon-delimited everywhere else.
# Put everything a function that we'll delete later
proc tmpfunc {} {
  global env
  global tcl_platform
  set delim :
  if { [string match $tcl_platform(platform) windows] } {
    set delim {;}
  }
  set archpath [file join $env(VMDDIR) plugins [vmdinfo arch] python]
  set noarchpath [file join $env(VMDDIR) plugins noarch python]

  if { ! [info exists env(PYTHONPATH)] } {
    set env(PYTHONPATH) $noarchpath
  } else {
    append env(PYTHONPATH) $delim $noarchpath
  }
  foreach p [glob -nocomplain -types d "$archpath/*" "$noarchpath/*"]  {
    append env(PYTHONPATH) $delim $p
  }
}
tmpfunc
rename tmpfunc ""


#######################################
# define a convenience function for scanning plugins
proc vmd_plugin_scandirectory { dir pattern } {
  # check that the directory exists
  if { ! [file isdirectory $dir] } {
    puts "Error) Plugin directory '$dir'\ndoes not exist!"
    return
  }
  set num 0
  foreach lib [glob -directory $dir -nocomplain $pattern] {
    if { [catch {plugin dlopen $lib} result] } {
      puts "Warning) Unable to dlopen '$lib':"
      puts "  $result"
    } else {
      incr num $result
    }
  }
  if { $num } {
    plugin update
    puts "Info) Dynamically loaded $num plugins in directory:"
    puts "Info) $dir"
  }
  return
}


#######################################
# Add user-defined path for Tcl packages as well as file plugins.
# Load the file plugins before the VMDDIR path to make it easier
# on developers of new plugins.
if { [info exists env(VMDPLUGINPATH)] } {
  foreach path [split $env(VMDPLUGINPATH) ":"] {
    lappend auto_path [file join $path tcl]

    set pluginpath [file join $path molfile]
    if [catch {vmd_plugin_scandirectory $pluginpath *.so} msg] {
      puts "Loading shared library plugins from $pluginpath failed:"
      puts $msg
    }
  }
}


#######################################
# Load the molecule file reader plugins
if [catch {vmd_plugin_scandirectory [file join $env(VMDDIR) plugins [vmdinfo arch] molfile] *.so} msg] {
    puts "Loading shared library plugins failed: $msg"
}


#######################################
# This function loads a package and installs it in the Extensions menu. The
# package's "menu window name" will be the same as package. This behavior can
# be overriden by specifying winname (for backwards compatibility).
proc vmd_install_extension {package tk_callback menupath {winname ""}} {
  if ![string length $winname] {set winname $package}

  if [catch {package require $package} msg] {
    puts "The $package package could not be loaded:\n$msg"
  } elseif [catch {menu tk register $winname $tk_callback $menupath} msg] {
    puts "The $package window could not be created:\n$msg"
  }
}


#######################################
# These packages create new menu items in the VMD menubar, and therefore
# should be loaded automatically by VMD.  They should, however, not be
# loaded until after the .vmdrc file is read, so that users (and VMD
# maintainers) can customize the Tcl search paths and insert their own 
# packages.
proc vmd_load_extension_packages {} {
  global tk_version
  global env
  if ![info exists tk_version] return

  ### Analysis menu  
  vmd_install_extension parsefep   ParseFEP_tk_cb "Analysis/Analyze FEP Simulation"
  vmd_install_extension apbsrun    apbsrun_tk_cb  "Analysis/APBS Electrostatics"
  vmd_install_extension plumed     plumed_tk      "Analysis/Collective variable analysis (PLUMED)"
  vmd_install_extension contactmap contactmap     "Analysis/Contact Map"
  vmd_install_extension heatmapper heatmapper     "Analysis/Heat Mapper"
  vmd_install_extension hbonds     hbonds_tk_cb   "Analysis/Hydrogen Bonds"
  vmd_install_extension ilstools   ilstools_tk    "Analysis/Implicit Ligand Sampling"
  vmd_install_extension irspecgui  irspecgui_tk_cb  "Analysis/IR Spectral Density Calculator"
  vmd_install_extension multiseq   multiseq         "Analysis/MultiSeq"
  vmd_install_extension namdenergy namdenergy_tk_cb "Analysis/NAMD Energy"
  vmd_install_extension namdplot   namdplot_tk    "Analysis/NAMD Plot"
  vmd_install_extension networkview networkviewgui "Analysis/NetworkView"
  vmd_install_extension nmwiz      nmwiz_tk       "Analysis/Normal Mode Wizard"
  vmd_install_extension pmepot_gui pmepot_gui     "Analysis/PME Electrostatics" pmepot
  vmd_install_extension propka     propka_tk      "Analysis/PropKa"
  vmd_install_extension gofrgui    gofrgui_tk_cb  "Analysis/Radial Pair Distribution Function g(r)"
  vmd_install_extension ramaplot ramaplot_tk     "Analysis/Ramachandran Plot"
  vmd_install_extension rmsdtool rmsdtool_tk_cb  "Analysis/RMSD Calculator" rmsd
  vmd_install_extension rmsdtt   rmsdtt_tk_cb    "Analysis/RMSD Trajectory Tool"
  vmd_install_extension rmsdvt   rmsdvt_tk       "Analysis/RMSD Visualizer Tool"
  vmd_install_extension saltbr   saltbr_tk_cb    "Analysis/Salt Bridges"
  vmd_install_extension symmetrytool symmetrytool_tk "Analysis/Symmetry Tool"
  vmd_install_extension zoomseq  zoomseq_tk      "Analysis/Sequence Viewer" sequence
  vmd_install_extension timeline timeline        "Analysis/Timeline"  
# wait for bug fixes
#  vmd_install_extension truncate_trajectory tt_GUI_tk "Analysis/Truncate Trajectory"
  vmd_install_extension volmapgui volmapgui_tk    "Analysis/VolMap Tool" volmap
 

  ### BioCoRE menu 
#  vmd_install_extension biocorelogin   biocorelogin_tk_cb   "BioCoRE/Login"
#  vmd_install_extension biocorechat    biocorechat_tk_cb    "BioCoRE/Chat"
#  vmd_install_extension biocorepubsync biocorepubsync_tk_cb "BioCoRE/Share VMD Views"
#  vmd_install_extension biocoreutil    biocoreutil_tk       "BioCoRE/Utilities"

  ### Data menu
  vmd_install_extension dataimport dataimport_tk   "Data/Data Import"
  vmd_install_extension pdbtool  pdbtool_tk_cb     "Data/PDB Database Query"
# XXX The STING DB has changed and is no longer mirrored/hosted at Columbia,
# so this plugin is disabled until it can be updated.
#  vmd_install_extension stingtool stingtool_tk_cb  "Data/STING Database Query"
  vmd_install_extension multitext multitext_tk     "Data/Text Editor"

  ### Modeling menu 
  vmd_install_extension autoionizegui autoigui  "Modeling/Add Ions" autoionize
  vmd_install_extension solvate  solvategui     "Modeling/Add Solvation Box"
  vmd_install_extension autopsf  autopsf_tk_cb  "Modeling/Automatic PSF Builder"
  vmd_install_extension cggui    cggui_tk       "Modeling/CG Builder"
  vmd_install_extension dowser_gui dowser_tk_cb "Modeling/Dowser"
  vmd_install_extension chirality_gui chirality_tk_cb "Modeling/Fix Chirality Errors"
  vmd_install_extension cispeptide_gui cispeptide_tk_cb "Modeling/Fix Cis Peptide Bonds"
  vmd_install_extension forcefieldtoolkit fftk "Modeling/Force Field Toolkit"
  vmd_install_extension inorganicbuilder inorganicBuilder_tk "Modeling/Inorganic Builder"
  vmd_install_extension mdff_gui mdffgui_tk     "Modeling/MDFF"
  vmd_install_extension membrane membrane_tk    "Modeling/Membrane Builder"
  vmd_install_extension mergestructs mergestructs_tk "Modeling/Merge Structures"
  vmd_install_extension molefacture molefacture_tk "Modeling/Molefacture"
  vmd_install_extension mutator  mutator_tk     "Modeling/Mutate Residue"
  vmd_install_extension nanotube nanotube_tk    "Modeling/Nanotube Builder"
  vmd_install_extension torsionplot torsionplot_tk "Modeling/TorsionPlot"
#  vmd_install_extension paratool paratool_tk_cb "Modeling/Parameterization Tool"

  ### Simulation menu
  vmd_install_extension alascan  alascan_tk   "Simulation/Alanine Scan Calculation"
  vmd_install_extension autoimd  autoimd_tk   "Simulation/AutoIMD (NAMD)"
  vmd_install_extension imdmenu  imdmenu_tk   "Simulation/IMD Connect (NAMD)" imd
  vmd_install_extension namdgui  namdgui_tk   "Simulation/NAMD Graphical Interface"
  vmd_install_extension qwikmd   qwikmd       "Simulation/QwikMD"
  vmd_install_extension qmtool   qmtool_tk_cb "Simulation/QMTool" 


  ### Visualization menu  
  vmd_install_extension bendix   bendix         "Visualization/Bendix"
  vmd_install_extension navigate navigate_tk_cb "Visualization/Camera Navigator (Keys)" 
  vmd_install_extension navfly   navfly_tk_cb   "Visualization/Camera Navigator (Mouse)"
  vmd_install_extension cliptool cliptool_tk_cb "Visualization/Clipping Plane Tool"
  vmd_install_extension clonerep clonerep_tk_cb "Visualization/Clone Representations"
  vmd_install_extension colorscalebar colorscalebar_tk_cb "Visualization/Color Scale Bar"
  vmd_install_extension dipwatch dipwatch_tk_cb "Visualization/Dipole Moment Watcher"

# Intersurf is turned off until we generate fresh binaries
#  switch [vmdinfo arch] {
#    WIN32 -
#    LINUX {
#      vmd_install_extension intersurf intersurf_cb  "Visualization/Intersurf"
#    }
#  }

  vmd_install_extension vmdmovie vmdmovie_tk_cb "Visualization/Movie Maker"
  vmd_install_extension multimolanim molanim_tk_cb "Visualization/Multiple Molecule Animation"
  vmd_install_extension palettetool palettetool_tk_cb "Visualization/PaletteTool"
  vmd_install_extension ruler ruler_tk "Visualization/Ruler"
  vmd_install_extension remote remotegui_tk "Visualization/Remote Control"
  vmd_install_extension viewchangerender_gui vcr_tk_cb "Visualization/ViewChangeRender"
  vmd_install_extension ViewMaster viewmaster_tk_cb "Visualization/ViewMaster" viewmaster
  vmd_install_extension vdna vdna_tk_cb "Visualization/Virtual DNA Viewer"

  ### Others 
  vmd_install_extension vmdtkcon vmdtkcon     "Tk Console" tkcon 
  vmd_install_extension vmdprefs vmdPrefs     "VMD Preferences"

  ### Python
# XXX don't turn on Python plugins yet until I solve the crashing problem 
# that occurs when TkInter windows are closed
#  # Install Python-based plugins if VMD was compiled with Python support.
#  if [llength [info commands gopython]] {
#    if [catch {gopython -command "import ied"} msg] {
#      puts "Unable to load IED package:"
##      puts $msg
#    }
#  }

##
#  XXX while I agree with the idea, I think this implementation will
#      break if installed in directories containing spaces in the 
#      names?  The purpose here is to allow extra extension
#      rpms to be produced with populate another directory.
#      we should probably come up with a nicer way of integrating
#      extra extension registration with this built-in stuff, so that
#      things are properly sorted in the gui regardless of the order
#      of the registration process.
#
  ### Load Extensions from additional files (e.g. from add-on RPMs)
  if {![catch {glob "$env(VMDDIR)/scripts/init.d/*.tcl"} addons ]} {
    foreach ext $addons {
      source $ext
    }
  }
  
  
  ### Here, load packages that do not have their own GUI window
  package require ruler

}




