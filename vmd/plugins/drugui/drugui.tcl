# Copyright (c) 2010, Ahmet Bakan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package provide drugui 1.0

package require solvate
package require autoionize
package require psfgen
package require pbctools
package require exectool


set DRUGGABILITY_PATH $env(DRUGGABILITY_PATH)
set PACKAGE_PATH "$DRUGGABILITY_PATH"

namespace eval ::druggability:: {
  namespace export druggability

  variable version 1.0

  variable w

  # Variables for system setup
  variable molid -1
  # input files
  variable protein_psf
  variable protein_pdb
  # probe percentages
  variable percent_ipro 70
  variable percent_ibut 0
  variable percent_acam 10
  variable percent_acetipam 20
  variable percent_total 100
  # solvation and ionization
  variable solvent_padding 6
  variable neutralize 1
  # output options
  variable output_prefix
  variable write_conf 1
  variable n_sims 1
  variable sim_length 40
  variable par_files


  # Variables for trajectory analysis
  # input files
  variable sysid -1
  variable system_psf
  variable system_pdb
  variable system_dcds
  # selection
  variable system_selstr "(helix or sheet) and name CA"
  # grid options
  variable grid_spacing 0.5
  variable probe_contact_dist 2.5
  # grid options, not used
  variable grid_water 0
  variable grid_hydrophobic 0
  variable grid_polar 0
  variable grid_negative 0
  variable grid_positive 0
  # output options
  variable use_volmap 1
  variable run_python 1
  variable pythonexe [::ExecTool::find "python"]
  variable outputdir ""
  variable pyprefix ""
  # trajectory options
  variable wrap_solvent 1
  variable save_aligned_system 0
  variable save_aligned_protein 0
  variable snapshot_from 0
  variable snapshot_to "end"

  # DIA options
  variable dia_visible 0
  variable dia_temp 300
  variable dia_delta_g -1
  variable dia_n_probes 7
  variable dia_min_n_probes 6
  variable dia_merge_radius 5.5
  variable dia_low_affinity 10
  variable dia_n_solutions 3
  variable dia_max_charge 2
  variable dia_n_charged 3
  variable dia_n_frames 1

  variable grid_list

  # Visualize results
  variable pdb_heavyatoms
  variable pdb_hotspots
  variable system_dcd_aligned
  variable dcd_step 10
  variable delete_others 0
  variable high_resolution 0
  variable protein_surface 0
  variable probe_radius_scale 1.6

  variable ligandpdbs
  variable pypickle
  variable dial_radius 1.5
  variable dial_deltag -0.5

  # Logvew window counter
  variable logcount 0
  variable lognames [list]
  variable titles [list "1) Prepare System" \
"2) Calculate Grids" \
"3) Assess Druggability" \
"4) Evaluate a Site" \
"5) Visualize Results"]
  variable interfaces [list "prepare" "process" "analyze" "evaluate" "visualize"]
  variable which_mode [lindex $titles 0]


}

proc druggability::Load_results {} {
  # Load protein heavy atoms, probe hotspots, and drug-like solutions
  variable pdb_heavyatoms
  variable pdb_hotspots
  variable sysid
  variable system_psf
  variable system_dcd_aligned
  variable dcd_step
  variable delete_others
  variable high_resolution
  variable dia_merge_radius
  variable protein_surface
  variable probe_radius_scale
  set simid -1

  if {$delete_others} {
    foreach i [molinfo list] { mol delete $i }
  } else {
    foreach i [molinfo list] { mol off $i }
  }

  set resol 6
  if {$high_resolution} {
    set resol 50
  }



  if {$sysid > -1} {
    set simid $sysid
  } elseif {$::druggability::system_psf != "" &&
            $::druggability::system_dcd_aligned != ""} {
    set simid [mol new $::druggability::system_psf]
    mol addfile $system_dcd_aligned type dcd step $dcd_step waitfor all
  }

  if {$simid > -1} {
    for {set i 0} {$i < [molinfo $simid get numreps]} {incr i} {
      mol delrep $i $simid
    }
    mol addrep $simid
    mol modstyle 0 $simid Lines 3
    mol modcolor 0 $simid Name
    mol smoothrep $simid 0 2
    mol modselect 0 $simid "noh protein"

    mol addrep $simid
    mol modstyle 1 $simid NewCartoon 0.3 $resol 4.1
    mol modcolor 1 $simid Structure
    mol smoothrep $simid 1 2


    mol addrep $simid
    mol modstyle 2 $simid Licorice 0.3 $resol $resol
    mol modcolor 2 $simid Name
    mol modselect 2 $simid "noh resname PRO2 IPRO IBUT IBTN IPAM ACET ACTT ACAM and same residue as within 3 of protein"
    mol selupdate 2 $simid on
    mol smoothrep $simid 2 2
    mol top $simid
  }

  # Check if both filenames are specified
  #if {[string length [string trim $pdb_hotspots]] == 0 ||
  #    [string length [string trim $pdb_heavyatoms]] == 0} {
  #  tk_messageBox -type ok -title "ERROR" \
  #    -message "Both protein and hotspot pdb files must be specified."
  #  return
  #}
  set heavyatoms -1
  if {[string length [string trim $pdb_heavyatoms]] != 0} {
    foreach i [molinfo list] { mol off $i }

    set heavyatoms [mol new $pdb_heavyatoms]
    mol modstyle 0 $heavyatoms Lines 3

    mol addrep $heavyatoms
    mol modstyle 1 $heavyatoms NewCartoon 0.3 $resol 4.1
    mol modcolor 1 $heavyatoms Structure

    mol addrep $heavyatoms
    mol modstyle 2 $heavyatoms VDW 1.0 $resol
    mol modcolor 2 $heavyatoms Name
    mol modselect 2 $heavyatoms "not protein"

    if {$protein_surface} {
      mol addrep $heavyatoms
      mol modstyle 3 $heavyatoms MSMS
      mol modcolor 3 $heavyatoms ColorID 8
      mol modmaterial 3 $heavyatoms Transparent
    }
  }

  set hotid -1
  if {[string length [string trim $pdb_hotspots]] != 0} {
    color scale method RWB
    set hotid [mol new $pdb_hotspots]
    mol modstyle 0 $hotid VDW 0.4 $resol
    mol modcolor 0 $hotid Beta

    set betamin 1000
    set betamax -1000
    foreach beta [[atomselect $hotid "all"] get beta] {
      if {$beta < $betamin} {set betamin $beta}
      if {$beta > $betamax} {set betamax $beta}
    }

    set n_sites [expr [llength [glob -nocomplain [string range $pdb_hotspots 0 [expr [string length $pdb_hotspots] -17]]site*.pdb]] - \
                      [llength [glob -nocomplain [string range $pdb_hotspots 0 [expr [string length $pdb_hotspots] -17]]site*_soln_*.pdb]] ]


    if ($n_sites) {
      for {set i 1} {$i <= $n_sites} {incr i} {

        if {[llength [glob -nocomplain [string range $pdb_hotspots 0 [expr [string length $pdb_hotspots] -17]]site_$i\*.pdb]] > 1} {
          set filelist [lsort [glob -nocomplain [string range $pdb_hotspots 0 \
                                  [expr [string length $pdb_hotspots] -17]]site_$i\_*.pdb]]
          foreach site $filelist {
            set hotid [mol new $site]
            mol modstyle 0 $hotid DynamicBonds $dia_merge_radius 0.3 $resol
            mol modcolor 0 $hotid Molecule
            mol addrep $hotid
            mol modstyle 1 $hotid VDW 0.4 $resol
            mol modcolor 1 $hotid Beta
            mol modmaterial 1 $hotid Opaque
            mol scaleminmax $hotid 1 $betamin $betamax
            mol addrep $hotid
            mol modstyle 2 $hotid VDW $probe_radius_scale $resol
            mol modcolor 2 $hotid Beta
            mol modmaterial 2 $hotid Transparent
            mol scaleminmax $hotid 2 $betamin $betamax
            mol off $hotid
          }
        }
      }
    } else {
      tk_messageBox -type ok -title "WARNING" \
        -message "No druggable sites were identified."
      return 0
    }
  }

  if {$simid > -1} {
    mol top $simid
    display resetview
  } elseif {$heavyatoms > -1} {
    mol top $heavyatoms
    display resetview
  } elseif {$heavyatoms > -1} {
    mol top $hotid
    display resetview
  }



  # also load view logfile if it exists
  set logfilename "[string range $pdb_hotspots 0 [expr [string length $pdb_hotspots] -18]].log"
  if {[file exists $logfilename]} {
    ::druggability::Logview $logfilename
  }


}

proc druggability::Logview {logfilename} {
  variable logcount
  variable lognames
  set logindex [lsearch $lognames $logfilename]
  set log .somenonsense
  if {$logindex > -1} {
    set windowname "log$logindex"
    set log .$windowname
  }
  if {[winfo exists $log] == 0} {
    if {$logindex > -1} {
      lset lognames $logindex "somenonsense"
    }
    set logindex $logcount
    lappend lognames $logfilename
    set windowname "log$logindex"
    set log [toplevel ".$windowname"]
    wm title $log "Logfile [lindex [file split $logfilename] end] ($logfilename)"
    wm resizable $log 1 1
    incr logcount

    text $log.text -bg White -bd 2 \
      -yscrollcommand ".$windowname.vscr set"
    scrollbar $log.vscr -command ".$windowname.text yview"
    pack $log.text -side left -fill both -expand 1
    pack $log.vscr -side right -fill y
  }

  $log.text configure -state normal
  #set count 0
  #set tabwidth 0
  #foreach family [lsort -dictionary [font families]] {
  #    $log.text tag configure f[incr count] -font [list $family 10]
  #    $log.text insert end ${family}:\t {} \
  #            "This is a simple sampler\n" f$count
  #    set w [font measure [$log.text cget -font] ${family}:]
  #    if {$w+5 > $tabwidth} {
  #        set tabwidth [expr {$w+5}]
  #        $log.text configure -tabs $tabwidth
  #    }
  #}
  $log.text delete 1.0 end
  set logfile [open $logfilename "r"]
  set line ""
  while {[gets $logfile line] != -1} {
    $log.text insert end "$line\n"
  }
  close $logfile
  $log.text yview moveto 1
  $log.text configure -state disabled
}

proc ::druggability::druggui {} {
  variable w

  global env

  # Determine whether PSF and PDB are loaded (based on Solvate plugin code)
  set ::druggability::protein_psf ""
  set ::druggability::protein_pdb ""
  if {[molinfo num] != 0} {
    foreach filename [lindex [molinfo top get filename] 0] \
            filetype [lindex [molinfo top get filetype] 0] {
      if { [string equal $filetype "psf"] } {
        set ::druggability::protein_psf $filename
      } elseif { [string equal $filetype "pdb"] } {
        set ::druggability::protein_pdb $filename
      }
      set ::druggability::molid [molinfo top get id]
    }
    if {$::druggability::protein_psf == "" ||
        $::druggability::protein_pdb == "" } {
      set ::druggability::protein_psf ""
      set ::druggability::protein_pdb ""
      set ::druggability::molid -1
    }
  }


  # If already initialized, just turn on
  if [winfo exists .druggui] {
    wm deiconify .druggui
    raise .druggui
    return
  }

  # Initialize window
  set w [toplevel .druggui]
  wm title $w "Druggability GUI v$::druggability::version"
  wm resizable $w 0 0

  set wif [frame $w.interface_frame]
  button $wif.help -text "?" -padx 0 -pady 3 -command {
      tk_messageBox -type ok -title "HELP" \
      -message "Use option menu to change the active interface. There are\
five interfaces to facilitate consequtive steps of Druggability assessment.\n\n\
[lindex $::druggability::titles 0]\n\
Prepare protein in a water-probe mixture box or in a water-only box using\
this interface. Also, by default generic NAMD input files are outputed.\
Protein PSF and PDB files that also \
contains cofactors/ions etc. are required from the user.\n\n\
[lindex $::druggability::titles 1]\n\
Align trajectories and calculate probe occupancy grids using\
this interface. System PSF/PDB files (from \
previous step) and DCD files (generated by simulating the system)\
are required from the user. Optionally, this interface also performs\
druggability assessment (see 3).\n\n\
[lindex $::druggability::titles 2]\n\
Assess druggability of the target protein using this interface.\
Probe occupancy grids from step 2 are required.\
This interface can be used to try different parameters.\
The analysis performed in this step will be save in a file with dso.gz extension\
to facilitate quick evaluation of specific sites using interface 4.\n\n\
[lindex $::druggability::titles 3]\n\
Evaluate druggability of a specific site using this interface. A DSO file\
from step 3 is required. A specific site can be defined using a bound ligand.\
Future versions will offer other ways to define a specific site.\n\n\
[lindex $::druggability::titles 4]\n\
Visualize hotspots and druggability solutions using this interface."}
  variable titles
  tk_optionMenu $wif.list ::druggability::which_mode "System Setup"
  $wif.list.menu delete 0
  $wif.list.menu add radiobutton -label [lindex $titles 0] \
    -variable ::druggability::which_mode \
    -command {::druggability::Switch_mode "prepare"}
  $wif.list.menu add radiobutton -label [lindex $titles 1] \
    -variable ::druggability::which_mode \
    -command {::druggability::Switch_mode "process"}
  $wif.list.menu add radiobutton -label [lindex $titles 2] \
    -variable ::druggability::which_mode \
    -command {::druggability::Switch_mode "analyze"}
  $wif.list.menu add radiobutton -label [lindex $titles 3] \
    -variable ::druggability::which_mode \
    -command {::druggability::Switch_mode "evaluate"}
  $wif.list.menu add radiobutton -label [lindex $titles 4] \
    -variable ::druggability::which_mode \
    -command {::druggability::Switch_mode "visualize"}
  pack $wif.help -side left
  pack $wif.list -side left -expand 1 -fill x
  pack $wif -pady 2 -expand 1 -fill x

  # Set main frame
  set mf [frame $w.main_frame]

  # VISUALIZE results
  set mfv [frame $mf.visualize]
  # Select input files
  set mfvif [labelframe $mfv.input_files -text "Input files and options:" -bd 2]
  # Heavy atoms file
  grid [button $mfvif.heavy_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "File that contains heavy atoms of the system (_heavyatoms.pdb),\
excluding water, probes, and counter ions."}] \
    -row 1 -column 0 -sticky w
  grid [label $mfvif.heavy_label -text "Protein:"] \
    -row 1 -column 1 -sticky w
  grid [entry $mfvif.heavy_path -width 36 \
      -textvariable ::druggability::pdb_heavyatoms] \
    -row 1 -column 2 -columnspan 6 -sticky ew
  grid [button $mfvif.heavy_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile \
                    -filetypes {{"PDB files" { .pdb .PDB }} {"All files" *}}]
      if {![string equal $tempfile ""]} {
        set ::druggability::pdb_heavyatoms $tempfile } }] \
    -row 1 -column 8 -sticky w -padx 0 -pady 0


  grid [button $mfvif.hotspots_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "This file should contain all hotspots (_all_hotspots.pdb).\
Other PDB files containing results will be seeked in the same folder."}] \
    -row 2 -column 0 -sticky w
  grid [label $mfvif.hotspots_label -text "Hotspots:"] \
    -row 2 -column 1 -sticky w
  grid [entry $mfvif.hotspots_path -width 36 \
      -textvariable ::druggability::pdb_hotspots] \
    -row 2 -column 2 -columnspan 6 -sticky ew
  grid [button $mfvif.hotspots_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile \
                    -filetypes {{"PDB files" { .pdb .PDB }} {"All files" *}}]
      if {![string equal $tempfile ""]} {
        set ::druggability::pdb_hotspots $tempfile } }] \
    -row 2 -column 8 -sticky w -padx 0 -pady 0

  grid [button $mfvif.psf_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "System PSF file. System DCD file should also be provided with this file."}] \
    -row 3 -column 0 -sticky w
  grid [label $mfvif.psf_label -text "System PSF:"] \
    -row 3 -column 1 -sticky w
  grid [entry $mfvif.psf_path -width 36 \
      -textvariable ::druggability::system_psf] \
    -row 3 -column 2 -columnspan 6 -sticky ew
  grid [button $mfvif.psf_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile \
                    -filetypes {{"PSF files" { .psf .PSF }} {"All files" *}}]
      if {![string equal $tempfile ""]} {
        set ::druggability::system_psf $tempfile } }] \
    -row 3 -column 8 -sticky w -padx 0 -pady 0

  grid [button $mfvif.dcd_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Aligned system DCD file should be provided. \
System PDF file must also be provied with this file."}] \
    -row 4 -column 0 -sticky w
  grid [label $mfvif.dcd_label -text "Aligned DCD:"] \
    -row 4 -column 1 -sticky w
  grid [entry $mfvif.dcd_path -width 36 \
      -textvariable ::druggability::system_dcd_aligned] \
    -row 4 -column 2 -columnspan 6 -sticky ew
  grid [button $mfvif.dcd_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile\
                    -filetypes {{"DCD files" { .dcd .DCD }} {"All files" *}}]
      if {![string equal $tempfile ""]} {
        set ::druggability::system_dcd_aligned $tempfile } }] \
    -row 4 -column 8 -sticky w -padx 0 -pady 0

  grid [button $mfvif.stride_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Allows skipping some of the snaphosts for faster loading."}] \
    -row 5 -column 0 -sticky w
  grid [label $mfvif.stride_label -text "DCD load step:"] \
    -row 5 -column 1 -sticky w
  grid [entry $mfvif.stride_entry -width 4 \
      -textvariable ::druggability::dcd_step] \
    -row 5 -column 2 -sticky w -padx 0 -pady 0

  grid [button $mfvif.del_oth_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If checked, all existing molecules will be deleted."}] \
    -row 7 -column 0 -sticky w
  grid [label $mfvif.del_oth_label -text "Delete existing:"] \
    -row 7 -column 1 -sticky w
  grid [checkbutton $mfvif.del_oth_check -text "" \
      -variable ::druggability::delete_others] \
    -row 7 -column 2 -sticky w -padx 0 -pady 0

  grid [label $mfvif.separator_label -text " "] \
    -row 7 -column 3 -sticky w

  grid [button $mfvif.high_res_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If checked, higher resolution representations will be generated."}] \
    -row 7 -column 4 -sticky e
  grid [label $mfvif.high_res_label -text "High resolution:"] \
    -row 7 -column 5 -sticky w
  grid [checkbutton $mfvif.high_res_check -text "" \
      -variable ::druggability::high_resolution] \
    -row 7 -column 6 -sticky w -padx 0 -pady 0
  grid [label $mfvif.space_label -text "                        "] \
    -row 7 -column 7 -sticky w

  grid [button $mfvif.pro_sur_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If checked, MSMS surface representation will be generated for the protein."}] \
    -row 8 -column 0 -sticky w
  grid [label $mfvif.pro_sur_label -text "Protein surface:"] \
    -row 8 -column 1 -sticky w
  grid [checkbutton $mfvif.pro_sur_check -text "" \
      -variable ::druggability::protein_surface] \
    -row 8 -column 2 -sticky w -padx 0 -pady 0


  pack $mfvif -side top -ipadx 0 -ipady 5 -fill x -expand 1
  # Visualize
  button $mfv.button -text "Visualize Results" -command ::druggability::Load_results -bd 3
  pack $mfv.button

  # Prepare System and Simulation Files
  set mfa [frame $mf.prepare]
  # Select input files
  set mfaif [labelframe $mfa.input_files -text "Protein structure and coordinate files:" -bd 2]
  # Protein PSF
  grid [button $mfaif.psf_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Protein PSF file should contain all components of the system\
before the solvent and probe molecules are added. This may include\
structural/functional ions, cofactors, etc. As a kindly reminder, please also\
make sure that the protonation states of histidines, cysteines, or other relevant\
residues are set properly and, if any, sulfide bridging cysteines are patched correctly."}] \
    -row 1 -column 0 -sticky w
  grid [label $mfaif.psf_label -text "PSF:"] -row 1 -column 1 -sticky w
  grid [entry $mfaif.psf_path -width 38 \
      -textvariable ::druggability::protein_psf] \
    -row 1 -column 2 -sticky ew
  grid [button $mfaif.psf_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile \
                    -filetypes {{"PSF files" { .psf .PSF }} {"All files" *}}]
      if {![string equal $tempfile ""]} {
        set ::druggability::protein_psf $tempfile
      } }] \
    -row 1 -column 3 -sticky w
  # Protein PDB
  grid [button $mfaif.pdb_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "This file must contain coordinate data corresponding to the\
entities described in the PSF file."}] \
    -row 2 -column 0 -sticky w
  grid [label $mfaif.pdb_label -text "PDB:"] \
    -row 2 -column 1 -sticky w
  grid [entry $mfaif.pdb_path -width 38 \
      -textvariable ::druggability::protein_pdb] \
    -row 2 -column 2 -sticky ew
  grid [button $mfaif.pdb_browse -text "Browse" -width 6 -pady 1 -command {
        set tempfile [tk_getOpenFile \
          -filetypes {{"PDB files" { .pdb .PDB }} {"All files" *}}]
        if {![string equal $tempfile ""]} {
          set ::druggability::protein_pdb $tempfile
        } }] \
    -row 2 -column 3 -sticky w
  # Load PSF and PDB files
  grid [button $mfaif.button -text "Load PSF and PDB files (optional)" \
      -command {::druggability::Load_protein} -justify center -pady 2] \
    -row 3 -column 0 -columnspan 4
  pack $mfaif -side top -ipadx 0 -ipady 5 -fill x -expand 1

  # Enter probe molecule percentages
  set mfapo [labelframe $mfa.probe_options -text "Probe composition:" -bd 2 -pady 2]
  grid [label $mfapo.ipro_label -text "% Isopropanol:"] \
    -row 0 -column 1 -sticky w
  grid [entry $mfapo.ipro_percent -width 3 -validate focusout \
      -vcmd {::druggability::Validate_percent_total} \
      -textvariable ::druggability::percent_ipro] \
    -row 0 -column 2 -sticky w
  grid [button $mfapo.ipro_add10 -text "+10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ipro 10}] \
    -row 0 -column 3 -sticky w
  grid [button $mfapo.ipro_add5 -text "+5" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ipro 5}] \
    -row 0 -column 4 -sticky w
  grid [button $mfapo.ipro_set0 -text "0" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ipro 0}] \
    -row 0 -column 5 -sticky w
  grid [button $mfapo.ipro_min5 -text "-5" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ipro -5}] \
    -row 0 -column 6 -sticky w
  grid [button $mfapo.ipro_min10 -text "-10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ipro -10}] \
    -row 0 -column 7 -sticky w

  grid [label $mfapo.ibut_label -text "% Isobutane:"] \
    -row 1 -column 1 -sticky w
  grid [entry $mfapo.ibut_percent -width 3 -validate focusout \
      -vcmd {::druggability::Validate_percent_total} \
      -textvariable ::druggability::percent_ibut] \
    -row 1 -column 2 -sticky w
  grid [button $mfapo.ibut_add10 -text "+10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ibut 10}] \
    -row 1 -column 3 -sticky w
  grid [button $mfapo.ibut_add5 -text "+5" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ibut 5}] \
    -row 1 -column 4 -sticky w
  grid [button $mfapo.ibut_set0 -text "0" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ibut 0}] \
    -row 1 -column 5 -sticky w
  grid [button $mfapo.ibut_min5 -text "-5" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ibut -5}] \
    -row 1 -column 6 -sticky w
  grid [button $mfapo.ibut_min10 -text "-10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage ibut -10}] \
    -row 1 -column 7 -sticky w

  grid [label $mfapo.acam_label -text "% Acetamide:"] \
    -row 2 -column 1 -sticky w
  grid [entry $mfapo.acam_percent -width 3 -validate focusout \
      -vcmd {::druggability::Validate_percent_total} \
      -textvariable ::druggability::percent_acam] \
    -row 2 -column 2 -sticky w
  grid [button $mfapo.acam_add10 -text "+10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acam 10}] \
    -row 2 -column 3 -sticky w
  grid [button $mfapo.acam_add5 -text "+5" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acam 5}] \
    -row 2 -column 4 -sticky w
  grid [button $mfapo.acam_set0 -text "0" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acam 0}] \
      -row 2 -column 5 -sticky w
  grid [button $mfapo.acam_min5 -text "-5" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acam -5}] \
    -row 2 -column 6 -sticky w
  grid [button $mfapo.acam_min10 -text "-10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acam -10}] \
    -row 2 -column 7 -sticky w

  grid [button $mfapo.acetipam_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Simulation system needs to be neutral, hence equivalent\
amounts of charged probes are allowed."}] \
    -row 3 -column 0 -sticky w
  grid [label $mfapo.acetipam_label -text "% Acetate(-) + Isopropylamine(+):"] \
    -row 3 -column 1 -sticky w
  grid [entry $mfapo.acetipam_percent -width 3 -validate focusout \
      -vcmd {::druggability::Validate_percent_total} \
      -textvariable ::druggability::percent_acetipam] \
    -row 3 -column 2 -sticky w
  grid [button $mfapo.acetipam_add10 -text "+10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acetipam 10}] \
    -row 3 -column 3 -sticky w
  grid [button $mfapo.acetipam_set0 -text "0" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acetipam 0}] \
    -row 3 -column 5 -sticky w
  grid [button $mfapo.acetipam_min10 -text "-10" -padx 0 -pady 0 -command {
      ::druggability::Change_percentage acetipam -10}] \
    -row 3 -column 7 -sticky w

  grid [button $mfapo.total_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Percentages must sum up to 100 or 0. If percentages sum up to 0,\
system will only be solvated. If percentages are multiples of 5 or 10, system will\
contain lesser number of solvent/probe molecules."}] \
    -row 4 -column 0 -sticky w
  grid [label $mfapo.total_label -text "Total of probe percentages: "] \
    -row 4 -column 1 -sticky w
  grid [entry $mfapo.total_percent -width 3 -disabledbackground lavender \
      -state disabled -textvariable ::druggability::percent_total] \
    -row 4 -column 2 -sticky w
  grid [button $mfapo.total_set0 -text "0" -padx 0 -pady 0 -command {
    ::druggability::Change_percentage all 0}] -row 4 -column 5 -sticky w

  pack $mfapo -side top -ipadx 0 -ipady 5 -fill x -expand 1

  set mfasi [labelframe $mfa.solion_options \
    -text "Solvation and ionization options:" -bd 2]
  grid [button $mfasi.padding_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "This is the half of the initial distance between the protein\
and its imaginary copies under periodic boundary conditions. For systems with\
probes, the resulting padding distance will be slightly larger, due to\
constraint of preserving the ratio of 20 water molecules per probe molecule."}] \
    -row 0 -column 0 -sticky w
  grid [label $mfasi.padding_label -text "Simulation box padding (A): "] \
    -row 0 -column 1 -sticky w
  grid [entry $mfasi.padding_entry -width 3 \
    -textvariable ::druggability::solvent_padding] \
    -row 0 -column 2 -sticky ew

  grid [label $mfasi.separatpr_label -text "   "] \
    -row 0 -column 3 -sticky w

  grid [button $mfasi.neutralize_help -text "?" -padx 0 -pady 0 -command {
    tk_messageBox -type ok -title "HELP" \
      -message "By default, counter ions will be added to neutralize a charged\
system. A charged system (if the protein is charged) may be obtained by unchecking this option."}] \
    -row 0 -column 4 -sticky w
  grid [label $mfasi.neutralize_label \
      -text "Add counter ions: "] \
    -row 0 -column 5 -sticky w
  grid [checkbutton $mfasi.neutralize_check -text "" \
      -variable ::druggability::neutralize] \
    -row 0 -column 6 -sticky w
  pack $mfasi -side top -ipadx 0 -ipady 5 -fill x -expand 1

  set mfaoo [labelframe $mfa.output_options -text "Output options:" -bd 2]

  grid [button $mfaoo.outdir_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Output folder, default is current working directory."}] \
    -row 0 -column 0 -sticky w
  grid [label $mfaoo.outdir_label -text "Output folder:"] \
    -row 0 -column 1 -sticky w
  grid [entry $mfaoo.outdir_path -width 18 \
      -textvariable ::druggability::outputdir] \
    -row 0 -column 2 -columnspan 4 -sticky ew
  grid [button $mfaoo.dcd_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_chooseDirectory]
      if {![string equal $tempfile ""]} {
        set ::druggability::outputdir $tempfile
      }}] \
    -row 0 -column 6 -sticky w

  grid [button $mfaoo.prefix_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "All output files and folders will start with this prefix.\
A unique and descriptive prefix choice may allow running multiple simulations in the same folder."}] \
    -row 1 -column 0 -sticky w
  grid [label $mfaoo.prefix_label -text "Output prefix:"] \
    -row 1 -column 1 -sticky w
  grid [entry $mfaoo.prefix_path -width 10 \
      -textvariable ::druggability::output_prefix] \
    -row 1 -column 2 -sticky w

  grid [label $mfaoo.separator_label -text "   "] \
    -row 1 -column 3 -sticky w

  grid [button $mfaoo.write_conf_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Minimization, equilibration, and simulation configuration\
files, and necessary parameter files will be written. Simulation parameters\
cannot be edited by the user within this GUI."}] \
    -row 1 -column 4 -sticky w
  grid [label $mfaoo.write_conf_label -text "Write NAMD input:"] \
    -row 1 -column 5 -sticky w
  grid [checkbutton $mfaoo.write_conf_check -text "" \
      -variable ::druggability::write_conf] \
    -row 1 -column 6 -sticky w

  grid [button $mfaoo.nsim_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If more than 1 is specified, multiple simulation input files\
will be generated. These simulations will differ by their random number\
generator seeds. This will result in different trajectories. Multiple simulations\
will share output of the same minmization run."}] \
    -row 3 -column 0 -sticky w
  grid [label $mfaoo.nsim_label -text "Number of sims:"] \
    -row 3 -column 1 -sticky w
  grid [entry $mfaoo.nsim_field -width 3 -textvariable ::druggability::n_sims] \
    -row 3 -column 2 -sticky w

  grid [button $mfaoo.sim_length_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "This is the length of the productive run in nanoseconds.\
Defaults for minimization (2000 steps) and equilibration (60 ps for water only\
systems, 900 ps for probe containing systems) is not included."}] \
    -row 3 -column 4 -sticky w
  grid [label $mfaoo.sim_length_label -text "Sim length (ns):"] \
    -row 3 -column 5 -sticky w
  grid [entry $mfaoo.sim_length_field -width 4 \
      -textvariable ::druggability::sim_length] \
    -row 3 -column 6 -sticky w

  grid [button $mfaoo.par_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If a system requires parameters in addition those defined in\
par_all27_prot_lipid_na.inp, additional filenames may be specified here.\
Specified files will be copied to parameters folder. If a specified file\
does not contain CHARMM format parameters, NAMD runtime error will occur."}] \
    -row 5 -column 0 -sticky w
  grid [label $mfaoo.par_label -text "Additional\nparameters:"] \
    -row 5 -column 1 -rowspan 2 -sticky wn
  grid [frame $mfaoo.par_frame] \
    -row 5 -column 2 -rowspan 2 -columnspan 4 -sticky w
  scrollbar $mfaoo.par_frame.scroll -command "$mfaoo.par_frame.list yview"
  listbox $mfaoo.par_frame.list \
    -activestyle dotbox -yscroll "$mfaoo.par_frame.scroll set" \
    -width 29 -height 3 -setgrid 1 -selectmode browse \
    -listvariable ::druggability::par_files
  frame $mfaoo.par_frame.buttons
  pack $mfaoo.par_frame.list $mfaoo.par_frame.scroll -side left -fill y -expand 1

  grid [button $mfaoo.par_add -text "Add" -width 6 -command [namespace code {
        set tempfile [tk_getOpenFile -multiple 1\
          -filetypes { {{CHARMM parameter files} {.prm .inp}} {{All files} {*}} }]
        if {$tempfiles!=""} {
          foreach tempfile $tempfiles {
            if {[lsearch $::druggability::par_files $tempfile] > -1} {
              tk_messageBox -type ok -title "WARNING" \
                -message "$tempfile has already been added to the list."
            } else {
              lappend ::druggability::par_files $tempfile
            }
          }
        }
        }] -pady 1] \
  -row 5 -column 6 -sticky w
  grid [button $mfaoo.par_delete -text "Remove" -width 6 \
        -command [namespace code {
      foreach i [.druggui.main_frame.prepare.output_options.par_frame.list curselection] {
        .druggui.main_frame.prepare.output_options.par_frame.list delete $i
      } }] -pady 1] \
  -row 6 -column 6 -sticky w

  pack $mfaoo -side top -ipadx 0 -ipady 5 -fill x -expand 1

  # Prepare System
  button $mfa.button -text "Prepare System" -command ::druggability::Prepare_system -bd 3
  pack $mfa.button

  # Process Trajectory and Calculate Grids
  set mfb [frame $mf.process]

  # Select input files
  set mfbif [labelframe $mfb.input_files -text "System structure, coordinate, and trajectory files:" -bd 2]
  grid [button $mfbif.psf_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "The PSF file of the simulated system."}] \
    -row 1 -column 0 -sticky w
  grid [label $mfbif.psf_label -text "PSF:"] \
    -row 1 -column 1 -sticky w
  grid [entry $mfbif.psf_path -width 38 -textvariable ::druggability::system_psf] \
    -row 1 -column 2 -sticky ew
  grid [button $mfbif.psf_browse -text "Browse" -width 6 -pady 1 -command {
        set tempfile [tk_getOpenFile -filetypes \
                      {{"PSF files" { .psf .PSF }} {"All files" *}}]
        if {![string equal $tempfile ""]} {
          set ::druggability::system_psf $tempfile
          set ::druggability::prefix [string range [lindex [file split $tempfile] end] 0 end-4]
          set ::druggability::outputdir [file join {*}[lrange [file split $tempfile] 0 end-1]]
          }
        }] \
    -row 1 -column 3 -sticky w

  grid [button $mfbif.pdb_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "The PDB file that contains the initial coordinates of the simulation."}] \
    -row 2 -column 0 -sticky w
  grid [label $mfbif.pdb_label -text "PDB:"] \
    -row 2 -column 1 -sticky w
  grid [entry $mfbif.pdb_path -width 38 \
      -textvariable ::druggability::system_pdb] \
    -row 2 -column 2 -sticky ew
  grid [button $mfbif.pdb_browse -text "Browse" -width 6 -pady 1 -command {
          set tempfile [tk_getOpenFile -filetypes \
            {{"PDB files" { .pdb .PDB }} {"All files" *}}]
          if {![string equal $tempfile ""]} {
            set ::druggability::system_pdb $tempfile
            } }] \
    -row 2 -column 3 -sticky w

  grid [button $mfbif.button -text "Load PSF and PDB files" \
    -command {::druggability::Load_system} -justify center -pady 2] \
    -row 3 -column 0 -columnspan 4

  grid [button $mfbif.selstr_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Selection will be used to align the protein. If protein has\
flexible loops or termini, they may be excluded from superimposition using this\
selection box. When Show button is clicked, protein as ribbon and selected\
atoms as spheres will be shown."}] \
    -row 4 -column 0 -sticky w
  grid [label $mfbif.selstr_label -text "Selection:"] \
    -row 4 -column 1 -sticky w
  grid [entry $mfbif.selstr_path -width 38 \
      -textvariable ::druggability::system_selstr] \
    -row 4 -column 2 -sticky ew
  grid [button $mfbif.selstr_button -text "Show" -width 6 -pady 1 \
      -command {::druggability::Show_system_selection} -justify center] \
    -row 4 -column 3

  grid [button $mfbif.dcd_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Multiple DCD files from the same simulation or from different\
simulations of the same system can be specified. DCD files will be read and\
processed in the order they are specified here."}] \
    -row 5 -column 0 -sticky w
  grid [label $mfbif.dcd_label -text "DCDs:"] \
    -row 5 -column 1 -sticky w
  grid [frame $mfbif.dcd_frame] \
    -row 5 -column 2 -rowspan 2 -columnspan 1 -sticky w
  scrollbar $mfbif.dcd_frame.scroll -command "$mfbif.dcd_frame.list yview"
  listbox $mfbif.dcd_frame.list -activestyle dotbox \
    -yscroll "$mfbif.dcd_frame.scroll set" \
    -width 37 -height 3 -setgrid 1 -selectmode browse \
    -listvariable ::druggability::system_dcds
  frame $mfbif.dcd_frame.buttons
  pack $mfbif.dcd_frame.list $mfbif.dcd_frame.scroll \
    -side left -fill y -expand 1

  grid [button $mfbif.dcd_add -text "Add" -width 6 -pady 1 \
        -command [namespace code {
        set tempfiles [tk_getOpenFile -multiple 1\
          -filetypes { {{DCD files} {.dcd .DCD}} {{All files} {*}} }]
        if {$tempfiles!=""} {
          foreach tempfile $tempfiles {
            if {[lsearch $::druggability::system_dcds $tempfile] > -1} {
              tk_messageBox -type ok -title "WARNING" \
                -message "$tempfile has already been added to the list."
            } else {
              lappend ::druggability::system_dcds $tempfile
            }
          }
        }
      }]] \
    -row 5 -column 3 -sticky w
  grid [button $mfbif.dcd_delete -text "Remove" \
        -width 6 -pady 1 -command [namespace code {
      foreach i [.druggui.main_frame.process.input_files.dcd_frame.list curselection] {
        .druggui.main_frame.process.input_files.dcd_frame.list delete $i
      } }]] \
    -row 6 -column 3 -sticky w


  pack $mfbif -side top -ipadx 0 -ipady 5 -fill x -expand 1

  # Trajectory options
  set mfbto [labelframe $mfb.trajectory_options -text "Trajectory options:" -bd 2]
  grid [button $mfbto.wrap_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "During alignment of snapshots, coordinates of molecules\
falling out of the original simulation box will be wrapped using PBCtools\
plug-in. This step is required for druggability calculations. If intended\
process involves only alignmen of the protein and saving resulting\
trajectory, this step may be omitted."}] \
    -row 0 -column 0 -sticky w
  grid [label $mfbto.dcd_wrap_label -text "Wrap solvent/probe molecules:"] \
    -row 0 -column 1 -sticky w
  grid [checkbutton $mfbto.dcd_wrap_check -text "" \
      -variable ::druggability::wrap_solvent] \
    -row 0 -column 2 -sticky w
  grid [label $mfbto.separator_label -text "   "] \
    -row 0 -column 3 -sticky w


  grid [button $mfbto.dcd_system_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If checked, processed trajectory will be saved. If multiple\
DCD files are processed, they will be concatenated in the order they are\
specified."}] \
    -row 4 -column 0 -sticky w
  grid [label $mfbto.dcd_system_label -text "Save processed trajectory:"] \
    -row 4 -column 1 -sticky w
  grid [checkbutton $mfbto.dcd_system_check -text "" \
      -variable ::druggability::save_aligned_system] \
    -row 4 -column 2 -sticky w

  grid [button $mfbto.dcd_protein_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If checked, processed trajectory containing only protein\
atoms (including cofactors, structural/functional ions) will be saved. All\
water and probe molecules, and counter ions will be excluded."}] \
    -row 4 -column 4 -sticky w
  grid [label $mfbto.dcd_protein_label -text "Save protein trajectory:"] \
    -row 4 -column 5 -sticky w
  grid [checkbutton $mfbto.dcd_protein_check -text "" \
      -variable ::druggability::save_aligned_protein] \
    -row 4 -column 6 -sticky w
  pack $mfbto -side top -ipadx 0 -ipady 5 -fill x -expand 1

  # Grid options
  set mfbdo [labelframe $mfb.grid_options -text "Grid calculation options:" -bd 2]

  grid [button $mfbdo.outdir_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Output folder, default is current working directory."}] \
    -row 0 -column 0 -sticky w
  grid [label $mfbdo.outdir_label -text "Output folder:"] \
    -row 0 -column 1 -sticky w
  grid [entry $mfbdo.outdir_path -width 18 \
      -textvariable ::druggability::outputdir] \
    -row 0 -column 2 -columnspan 5 -sticky ew
  grid [button $mfbdo.dcd_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_chooseDirectory]
      if {![string equal $tempfile ""]} {
        set ::druggability::outputdir $tempfile
      }}] \
    -row 0 -column 7 -sticky w


  grid [button $mfbdo.volmap_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Grids will be calculated in the current VMD session using\
Volmap plugin. Grid data is written in OpenDX format."}] \
    -row 1 -column 0 -sticky w
  grid [label $mfbdo.volmap_label -text "Calculate grids:"] \
    -row 1 -column 1 -sticky w
  grid [checkbutton $mfbdo.volmap_check -text "" \
      -variable ::druggability::use_volmap] \
    -row 1 -column 2 -sticky w


  grid [button $mfbdo.prefix_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Grid and python file names will start with this prefix."}] \
    -row 1 -column 4 -sticky e
  grid [label $mfbdo.prefix_label -text "Output prefix:"] \
    -row 1 -column 5 -columnspan 2 -sticky w
  grid [entry $mfbdo.prefix_path -width 14 \
      -textvariable ::druggability::prefix] \
    -row 1 -column 7 -columnspan 2 -sticky w

  grid [button $mfbdo.spacing_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "The size of a grid element, along X, Y, and Z dimensions.\
0.5 works best for druggability index calculations."}] \
    -row 3 -column 0 -sticky w
  grid [label $mfbdo.spacing_label -text "Grid resolution (A): "] \
    -row 3 -column 1 -sticky w
  grid [entry $mfbdo.spacing_entry -width 3 \
      -textvariable ::druggability::grid_spacing] \
    -row 3 -column 2 -columnspan 4 -sticky w

  grid [button $mfbdo.condist_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Only probe molecules having at least one atom within the\
contact distance of the protein atoms will be counted in grid calculations."}] \
    -row 3 -column 4 -sticky e
  grid [label $mfbdo.condist_label -text "Contact distance (A): "] \
    -row 3 -column 5 -columnspan 3 -sticky w
  grid [entry $mfbdo.condist_entry -width 3 \
      -textvariable ::druggability::probe_contact_dist] \
    -row 3 -column 8 -sticky w

  grid [button $mfbdo.extra_grid_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "In addition to central carbon atom grids, which are used\
for druggability analysis, atom type based grids can be outputed by selecting these options.\
These grids contain only heavy atom counts. Only heavy atoms that are\
within 4 A of a protein heavy atom are considered. Hydrophobic grid will\
contain all carbon atoms of isobutane and isopropanol. Polar grid will\
contain oxygen and nitrogen atoms of isopropanol and acetamide. Negative\
grid will contain both oxygens of acetate. Positive grid will contain\
nitrogen atom of isopropylamine. Water grid will contain only oxygen\
atoms of water molecules."}] \
    -row 4 -column 0 -sticky w
  grid [label $mfbdo.extra_grid_label -text "Additional grids:"] \
    -row 4 -column 1 -sticky w
  grid [checkbutton $mfbdo.grid_hydrophobic -text "Hydrophobic" \
      -variable ::druggability::grid_hydrophobic] \
    -row 4 -column 2 -columnspan 3 -sticky w
  grid [checkbutton $mfbdo.grid_polar -text "Polar" \
      -variable ::druggability::grid_polar] \
    -row 4 -column 5 -sticky w
  grid [checkbutton $mfbdo.grid_positive -text "+" \
      -variable ::druggability::grid_positive] \
    -row 4 -column 6 -sticky w
  grid [checkbutton $mfbdo.grid_negative -text "-" \
      -variable ::druggability::grid_negative] \
    -row 4 -column 7 -sticky w
  grid [checkbutton $mfbdo.grid_water -text "Water" \
      -variable ::druggability::grid_water] \
    -row 4 -column 8 -sticky w

  grid [button $mfbdo.python_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "A Python file to assess druggability will be written by\
default. If this option is checked, Python file will be executed and\
results will be loaded into this session."}] \
    -row 7 -column 0 -sticky w
  grid [label $mfbdo.python_label -text "Evaluate grids:"] \
    -row 7 -column 1 -sticky w
  grid [checkbutton $mfbdo.python_check -text "" \
      -variable ::druggability::run_python] \
    -row 7 -column 2 -sticky w

  grid [button $mfbdo.diaoptions_view -text "Show options and parameters" -padx 5 -pady 1 -bd 1 -command {
      if {$::druggability::dia_visible} {
        pack forget $::druggability::w.main_frame.process.dia_options;
        set ::druggability::dia_visible 0;
        $::druggability::w.main_frame.process.grid_options.diaoptions_view configure -relief raised;
        $::druggability::w.main_frame.process.grid_options.diaoptions_view configure -text "Show options and parameters";
      } else {
        pack forget $::druggability::w.main_frame.process.button
        pack $::druggability::w.main_frame.process.dia_options -side top -ipadx 0 -ipady 5 -fill x -expand 1;
        pack $::druggability::w.main_frame.process.button
        set ::druggability::dia_visible 1;
        $::druggability::w.main_frame.process.grid_options.diaoptions_view configure -relief sunken
        $::druggability::w.main_frame.process.grid_options.diaoptions_view configure -text "Hide options and parameters";
      }}] \
    -row 7 -column 3 -columnspan 5 -sticky w


  pack $mfbdo -side top -ipadx 0 -ipady 5 -fill x -expand 1


  set mfbdia [labelframe $mfb.dia_options -text "Druggability options and parameters:" -bd 2]
  grid [button $mfbdia.temp_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Temperature of the system in the productive simulation\
(NAMD configuration files generated by this GUI sets temperature to 300 K)."}] \
    -row 0 -column 0 -sticky w
  grid [label $mfbdia.temp_label -text "Temperature (K):"] \
    -row 0 -column 1 -sticky w
  grid [entry $mfbdia.temp_entry -width 3 \
      -textvariable ::druggability::dia_temp] \
    -row 0 -column 2 -sticky w

  grid [label $mfbdia.separator_label -text " "] \
    -row 0 -column 3 -sticky w

  grid [button $mfbdia.radius_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Probe merge radius in angstroms. Twice the size of\
largest effective probe radius gives reasonable solutions."}] \
    -row 0 -column 4 -sticky w
  grid [label $mfbdia.radius_label -text "Probe merge radius (A):"] \
    -row 0 -column 5 -sticky w
  grid [entry $mfbdia.radius_entry -width 3 \
      -textvariable ::druggability::dia_merge_radius] \
    -row 0 -column 6 -sticky w

  grid [button $mfbdia.frames_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If grid files were prepared using VMD Volmap, this needs\
to be set to 1, since Volmap prepares frame averaged grid data. If grid\
files were prepared using another program (e.g. ptraj) this may need to\
be set to the number of frames used in grid preparation."}] \
    -row 2 -column 0 -sticky w
  grid [label $mfbdia.frames_label -text "Number of frames:"] \
    -row 2 -column 1 -sticky w
  grid [entry $mfbdia.frames_entry -width 3 \
      -textvariable ::druggability::dia_n_frames] \
    -row 2 -column 2 -sticky w

  grid [button $mfbdia.probes_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Number of probe binding hotspots to merge to make a\
drug-size solution."}] \
    -row 2 -column 4 -sticky w
  grid [label $mfbdia.probes_label -text "Number of hotspots to merge:"] \
    -row 2 -column 5 -sticky w
  grid [entry $mfbdia.probes_entry -width 3 \
      -textvariable ::druggability::dia_n_probes] \
    -row 2 -column 6 -sticky w

  grid [button $mfbdia.delta_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Probe binding free energy to determine binding hotspots."}] \
    -row 4 -column 0 -sticky w
  grid [label $mfbdia.delta_label -text "Hotspot dG (kcal/mol):"] \
    -row 4 -column 1 -sticky w
  grid [entry $mfbdia.delta_entry -width 3 \
      -textvariable ::druggability::dia_delta_g] \
    -row 4 -column 2 -sticky w

  grid [button $mfbdia.minnpro_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Minimum number of hotspots in an acceptable drug-size solution."}] \
    -row 4 -column 4 -sticky w
  grid [label $mfbdia.minnpro_label -text "Minimum number of hotspots:"] \
    -row 4 -column 5 -sticky w
  grid [entry $mfbdia.minnpro_entry -width 3 \
      -textvariable ::druggability::dia_min_n_probes] \
    -row 4 -column 6 -sticky w

  grid [button $mfbdia.affinity_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Lowest affinity to report a solution in micromolar units."}] \
    -row 6 -column 0 -sticky w
  grid [label $mfbdia.affinity_label -text "Lowest affinity (uM):"] \
    -row 6 -column 1 -sticky w
  grid [entry $mfbdia.affinity_entry -width 3 \
      -textvariable ::druggability::dia_low_affinity] \
    -row 6 -column 2 -sticky w

  grid [button $mfbdia.charge_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Maximum absolute charge to accept solutions."}] \
    -row 6 -column 4 -sticky w
  grid [label $mfbdia.charge_label -text "Maximum absolute charge (e): "] \
    -row 6 -column 5 -sticky w
  grid [entry $mfbdia.charge_entry -width 3 \
      -textvariable ::druggability::dia_max_charge] \
    -row 6 -column 6 -sticky w

  grid [button $mfbdia.solutions_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Number of solutions to report in each distinct potential\
binding site."}] \
    -row 8 -column 0 -sticky w
  grid [label $mfbdia.solutions_label -text "Number of solutions:"] \
    -row 8 -column 1 -sticky w
  grid [entry $mfbdia.solutions_entry -width 3 \
      -textvariable ::druggability::dia_n_solutions] \
    -row 8 -column 2 -sticky w

  grid [button $mfbdia.ncharged_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Maximum number of charged hotspots in a solution."}] \
    -row 8 -column 4 -sticky w
  grid [label $mfbdia.ncharged_label -text "Number of charged hotspots: "] \
    -row 8 -column 5 -sticky w
  grid [entry $mfbdia.ncharged_entry -width 3 \
      -textvariable ::druggability::dia_n_charged] \
    -row 8 -column 6 -sticky w



  grid [button $mfbdia.pyexe_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "You can specify the path to the correct version of Python here."}] \
    -row 11 -column 0 -sticky w
  grid [label $mfbdia.pyexe_label -text "Python executable:"] \
    -row 11 -column 1 -sticky w
  grid [entry $mfbdia.pyexe_path -width 18 \
      -textvariable ::druggability::pythonexe] \
    -row 11 -column 2 -columnspan 4 -sticky ew
  grid [button $mfbdia.dcd_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} {
        set ::druggability::pythonexe $tempfile
      }}] \
    -row 11 -column 6 -sticky w

  # Calculate Grids
  button $mfb.button -text "Calculate Grids" -command ::druggability::Process_system -bd 3
  pack $mfb.button

  # Prepare System and Simulation Files
  set mfd [frame $mf.analyze]

  set mfddia [labelframe $mfd.dia_options -text "Options and parameters:" -bd 2]
  grid [button $mfddia.name_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "  "}] \
    -row 0 -column 0 -sticky w
  grid [label $mfddia.name_label -text "Output prefix:"] \
    -row 0 -column 1 -sticky w
  grid [entry $mfddia.name_entry -width 20 \
      -textvariable ::druggability::pyprefix] \
    -row 0 -column 2 -columnspan 5 -sticky w

  grid [button $mfddia.temp_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Temperature of the system in the productive simulation\
(NAMD configuration files generated by this GUI sets temperature to 300 K)."}] \
    -row 1 -column 0 -sticky w
  grid [label $mfddia.temp_label -text "Temperature (K):"] \
    -row 1 -column 1 -sticky w
  grid [entry $mfddia.temp_entry -width 3 \
      -textvariable ::druggability::dia_temp] \
    -row 1 -column 2 -sticky w

  grid [label $mfddia.separator_label -text " "] \
    -row 1 -column 3 -sticky w

  grid [button $mfddia.radius_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Probe merge radius in angstroms. Twice the size of largest\
effective probe radius gives reasonable solutions."}] \
    -row 1 -column 4 -sticky w
  grid [label $mfddia.radius_label -text "Probe merge radius (A):"] \
    -row 1 -column 5 -sticky w
  grid [entry $mfddia.radius_entry -width 3 \
      -textvariable ::druggability::dia_merge_radius] \
    -row 1 -column 6 -sticky w

  grid [button $mfddia.frames_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If grid files were prepared using VMD Volmap, this needs\
to be set to 1, since Volmap prepares frame averaged grid data. If grid\
files were prepared using another program (e.g. ptraj) this may need to\
be set to the number of frames used in grid preparation."}] \
    -row 2 -column 0 -sticky w
  grid [label $mfddia.frames_label -text "Number of frames:"] \
    -row 2 -column 1 -sticky w
  grid [entry $mfddia.frames_entry -width 3 \
      -textvariable ::druggability::dia_n_frames] \
    -row 2 -column 2 -sticky w

  grid [button $mfddia.probes_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Number of probe binding hotspots to merge to make a\
drug-size solution."}] \
    -row 2 -column 4 -sticky w
  grid [label $mfddia.probes_label -text "Number of hotspots to merge:"] \
    -row 2 -column 5 -sticky w
  grid [entry $mfddia.probes_entry -width 3 \
      -textvariable ::druggability::dia_n_probes] \
    -row 2 -column 6 -sticky w

  grid [button $mfddia.delta_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Probe binding free energy to determine binding hotspots."}] \
    -row 4 -column 0 -sticky w
  grid [label $mfddia.delta_label -text "Hotspot dG (kcal/mol):"] \
    -row 4 -column 1 -sticky w
  grid [entry $mfddia.delta_entry -width 3 \
      -textvariable ::druggability::dia_delta_g] \
    -row 4 -column 2 -sticky w

  grid [button $mfddia.minnpro_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Minimum number of hotspots in an acceptable drug-size solution."}] \
    -row 4 -column 4 -sticky w
  grid [label $mfddia.minnpro_label -text "Minimum number of hotspots:"] \
    -row 4 -column 5 -sticky w
  grid [entry $mfddia.minnpro_entry -width 3 \
      -textvariable ::druggability::dia_min_n_probes] \
    -row 4 -column 6 -sticky w

  grid [button $mfddia.affinity_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Lowest affinity to report a solution in micromolar units."}] \
    -row 6 -column 0 -sticky w
  grid [label $mfddia.affinity_label -text "Lowest affinity (uM):"] \
    -row 6 -column 1 -sticky w
  grid [entry $mfddia.affinity_entry -width 3 \
      -textvariable ::druggability::dia_low_affinity] \
    -row 6 -column 2 -sticky w

  grid [button $mfddia.charge_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Maximum absolute charge to accept solutions."}] \
    -row 6 -column 4 -sticky w
  grid [label $mfddia.charge_label -text "Maximum absolute charge (e): "] \
    -row 6 -column 5 -sticky w
  grid [entry $mfddia.charge_entry -width 3 \
      -textvariable ::druggability::dia_max_charge] \
    -row 6 -column 6 -sticky w

  grid [button $mfddia.solutions_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Number of solutions to report in each distinct potential binding site."}] \
    -row 8 -column 0 -sticky w
  grid [label $mfddia.solutions_label -text "Number of solutions:"] \
    -row 8 -column 1 -sticky w
  grid [entry $mfddia.solutions_entry -width 3 \
      -textvariable ::druggability::dia_n_solutions] \
    -row 8 -column 2 -sticky w

  grid [button $mfddia.ncharged_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Maximum number of charged hotspots in a solution."}] \
    -row 8 -column 4 -sticky w
  grid [label $mfddia.ncharged_label -text "Number of charged hotspots: "] \
    -row 8 -column 5 -sticky w
  grid [entry $mfddia.ncharged_entry -width 3 \
      -textvariable ::druggability::dia_n_charged] \
    -row 8 -column 6 -sticky w

  grid [button $mfddia.pyexe_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "You can specify the path to Python executable here."}] \
    -row 11 -column 0 -sticky w
  grid [label $mfddia.pyexe_label -text "Python executable:"] \
    -row 11 -column 1 -sticky w
  grid [entry $mfddia.pyexe_path -width 18 \
      -textvariable ::druggability::pythonexe] \
    -row 11 -column 2 -columnspan 4 -sticky ew
  grid [button $mfddia.dcd_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} {
        set ::druggability::pythonexe $tempfile }}] \
    -row 11 -column 6 -sticky w

  set mfdif [labelframe $mfd.input_files -text "Probe grid files:" -bd 2]
  grid [button $mfdif.dcd_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Specify grid files here. Probe types are recognized from\
filenames. Naming should be similar to systemname_probetype.dx, e.g. default_IPRO.dx."}] \
    -row 5 -column 0 -sticky w
  grid [frame $mfdif.dcd_frame] \
    -row 5 -column 2 -rowspan 2 -columnspan 1 -sticky w
  scrollbar $mfdif.dcd_frame.scroll -command "$mfdif.dcd_frame.list yview"
  listbox $mfdif.dcd_frame.list -activestyle dotbox \
    -yscroll "$mfdif.dcd_frame.scroll set" \
    -width 47 -height 3 -setgrid 1 -selectmode browse \
    -listvariable ::druggability::grid_list
  frame $mfdif.dcd_frame.buttons
  pack $mfdif.dcd_frame.list $mfdif.dcd_frame.scroll \
    -side left -fill y -expand 1

  grid [button $mfdif.dcd_add -text "Add" -width 6 -pady 1 \
        -command [namespace code {
        set tempfiles [tk_getOpenFile -multiple 1\
          -filetypes { {{DCD files} {.dx .DX}} {{All files} {*}} }]
        if {$tempfiles!=""} {
          foreach tempfile $tempfiles {
            set ::druggability::outputdir [file join {*}[lrange [file split $tempfile] 0 end-1]]
            set filename [lindex [file split $tempfile] end]
            set recognized 0

            #foreach probe_type "IPRO ipro PRO2 pro2 IPAM ipam IBUT ibut ACET acet ACAM acam"

            foreach probe_type "IPRO PRO2 IPAM IBUT IBTN ACET ACTT ACAM" {
              if { [string first $probe_type $filename] > -1} {
                set recognized 1
                set probe_type_fn $probe_type
                set probe_type [string toupper $probe_type]
                if {$probe_type == "PRO2"} {
                  set probe_type "IPRO"
                } elseif {$probe_type == "IBTN"} {
                  set probe_type "IBUT"
                } elseif {$probe_type == "ACTT"} {
                  set probe_type "ACET"
                }

                set notadded 1
                #if {[lsearch $::druggability::grid_list "$probe_type $tempfile"] > -1}
                foreach temp $::druggability::grid_list {
                  if {[lsearch $temp "$probe_type"] > -1} {
                    tk_messageBox -type ok -title "ERROR" \
                      -message "$probe_type grid has already been added to the list."
                    set notadded 0
                  }
                }
                if {$notadded} {
                  lappend ::druggability::grid_list "$probe_type $tempfile"
                  set ::druggability::pyprefix [join [lrange [split $filename .] 0 end-1]]
                  set which [string first $probe_type_fn $::druggability::pyprefix]
                  set ::druggability::pyprefix [string replace $::druggability::pyprefix [expr $which -1] [expr $which + 4] ""]
                }
              }
            }
            if {$recognized == 0} {
              tk_messageBox -type ok -title "ERROR" \
                -message "Probe type could not be recognized for $filename."
            }
          }
        }
      }]] \
    -row 5 -column 3 -sticky w
  grid [button $mfdif.dcd_delete -text "Remove" \
        -width 6 -pady 1 -command [namespace code {
      foreach i [.druggui.main_frame.analyze.input_files.dcd_frame.list curselection] {
        .druggui.main_frame.analyze.input_files.dcd_frame.list delete $i
      } }]] \
    -row 6 -column 3 -sticky w

  grid [button $mfdif.outdir_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Output folder, default is current working directory."}] \
    -row 7 -column 0 -sticky w
  grid [label $mfdif.outdir_label -text "Output folder:"] \
    -row 7 -column 1 -sticky w
  grid [entry $mfdif.outdir_path -width 18 \
      -textvariable ::druggability::outputdir] \
    -row 7 -column 2 -sticky ew
  grid [button $mfdif.dcd_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_chooseDirectory]
      if {![string equal $tempfile ""]} {
        set ::druggability::outputdir $tempfile
      }}] \
    -row 7 -column 3 -sticky w

  pack $mfdif -side top -ipadx 0 -ipady 5 -fill x -expand 1;

  pack $mfddia -side top -ipadx 0 -ipady 5 -fill x -expand 1;

  button $mfd.button -text "Assess Druggability" -command ::druggability::Assess_druggability -bd 3
  pack $mfd.button


  # Prepare System and Simulation Files
  set mfe [frame $mf.evaluate]

  set mfeif [labelframe $mfe.input_files -text "Input files:" -bd 2]
  grid [button $mfeif.par_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "Ligand coordinates will be used to determine the site. PDB files should contain only ligand atoms."}] \
    -row 0 -column 0 -sticky w
  grid [label $mfeif.par_label -text "Ligand PDBs:"] \
    -row 0 -column 1 -sticky w
  grid [frame $mfeif.par_frame] \
    -row 0 -column 2 -rowspan 2 -sticky w
  scrollbar $mfeif.par_frame.scroll -command "$mfeif.par_frame.list yview"
  listbox $mfeif.par_frame.list \
    -activestyle dotbox -yscroll "$mfeif.par_frame.scroll set" \
    -width 36 -height 3 -setgrid 1 -selectmode browse \
    -listvariable ::druggability::ligandpdbs
  frame $mfeif.par_frame.buttons
  pack $mfeif.par_frame.list $mfeif.par_frame.scroll -side left -fill y -expand 1

  grid [button $mfeif.par_add -text "Add" -width 6 -command [namespace code {
        set tempfiles [tk_getOpenFile -multiple 1 \
          -filetypes { {{PDB files} {.pdb .PDB}} {{All files} {*}} }]
        if {$tempfiles!=""} {
          foreach tempfile $tempfiles {
            if {[lsearch $::druggability::ligandpdbs $tempfile] > -1} {
              tk_messageBox -type ok -title "WARNING" \
                -message "$tempfile has already been added to the list."
            } else {
              lappend ::druggability::ligandpdbs $tempfile
            }
          }
        }
        }] -pady 1] \
  -row 0 -column 3 -sticky w
  grid [button $mfeif.par_delete -text "Remove" -width 6 \
        -command [namespace code {
      foreach i [.druggui.main_frame.evaluate.input_files.par_frame.list curselection] {
        .druggui.main_frame.evaluate.input_files.par_frame.list delete $i
      } }] -pady 1] \
  -row 1 -column 3 -sticky w

  grid [button $mfeif.dso_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "A DSO (Druggability Suite Object) file that contains a\
previously saved analysis is required for evaluation of a specific site."}] \
    -row 3 -column 0 -sticky w
  grid [label $mfeif.dso_label -text "DSO file:"] \
    -row 3 -column 1 -sticky w
  grid [entry $mfeif.dso_path -width 37 \
      -textvariable ::druggability::pypickle] \
    -row 3 -column 2 -sticky ew
  grid [button $mfeif.dso_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile \
                    -filetypes {{"DSO files" { .dso .dso.gz }} {"All files" *}}]
      if {![string equal $tempfile ""]} {
        set ::druggability::pypickle $tempfile } }] \
    -row 3 -column 3 -sticky w -padx 0 -pady 0

  pack $mfeif -side top -ipadx 0 -ipady 5 -fill x -expand 1;

  set mfedia [labelframe $mfe.dia_options -text "Options and parameters:" -bd 2]

  grid [button $mfedia.solutions_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "When evaluating a ligand bound site, grid elements within\
a specific distance from the ligand atoms will be considered. User can\
adjust this distance using this parameter."}] \
    -row 1 -column 0 -sticky w
  grid [label $mfedia.solutions_label -text "Within ligand atoms (A):"] \
    -row 1 -column 1 -sticky w
  grid [entry $mfedia.solutions_entry -width 3 \
      -textvariable ::druggability::dial_radius] \
    -row 1 -column 2 -sticky w

  grid [label $mfedia.separator_label -text "       "] \
    -row 1 -column 3 -sticky w

  grid [button $mfedia.temp_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "The maximum dG will be used to determine probe binding\
hotspots in the site of interest. This parameter must have a negative value."}] \
    -row 1 -column 4 -sticky w
  grid [label $mfedia.temp_label -text "Maximum dG (kcal/mol):"] \
    -row 1 -column 5 -sticky w
  grid [entry $mfedia.temp_entry -width 3 \
      -textvariable ::druggability::dial_deltag] \
    -row 1 -column 6 -sticky w

  grid [button $mfedia.pyexe_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "You can specify the path to the correct version of Python here."}] \
    -row 11 -column 0 -sticky w
  grid [label $mfedia.pyexe_label -text "Python executable:"] \
    -row 11 -column 1 -sticky w
  grid [entry $mfedia.pyexe_path -width 18 \
      -textvariable ::druggability::pythonexe] \
    -row 11 -column 2 -columnspan 4 -sticky ew
  grid [button $mfedia.dcd_browse -text "Browse" -width 6 -pady 1 -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} {
        set ::druggability::pythonexe $tempfile } }] \
    -row 11 -column 6 -sticky w

  grid [button $mfedia.high_res_help -text "?" -padx 0 -pady 0 -command {
      tk_messageBox -type ok -title "HELP" \
        -message "If checked, higher resolution representations\
will be generated for the ligand and the hotspots."}] \
    -row 17 -column 0 -sticky e
  grid [label $mfedia.high_res_label -text "High resolution:"] \
    -row 17 -column 1 -sticky w
  grid [checkbutton $mfedia.high_res_check -text "" \
      -variable ::druggability::high_resolution] \
    -row 17 -column 2 -sticky w -padx 0 -pady 0

  pack $mfedia -side top -ipadx 0 -ipady 5 -fill x -expand 1;

  button $mfe.button -text "Evaluate Site" -command ::druggability::Evaluate_ligand -bd 3
  pack $mfe.button

  pack $mfa -side top -padx 0 -pady 0 -fill x -expand 1
  pack $mf -side top -padx 0 -pady 0 -fill x -expand 1

  return $w
}

proc ::druggability::Change_percentage {which what} {
  # Adjust probe percentages
  # which: probe type
  # what: percent increment
  variable percent_ipro
  variable percent_ibut
  variable percent_acam
  variable percent_acetipam
  if {$which == "ipro"} {
    if {$what == 0} {
      set percent_ipro 0
    } else {
      incr percent_ipro $what
    }
  } elseif {$which == "ibut"} {
    if {$what == 0} {
      set percent_ibut 0
    } else {
      incr percent_ibut $what
    }
  } elseif {$which == "acam"} {
    if {$what == 0} {
      set percent_acam 0
    } else {
      incr percent_acam $what
    }
  } elseif {$which == "acetipam"} {
    if {$what == 0} {
      set percent_acetipam 0
    } else {
      incr percent_acetipam $what
    }
  } elseif {$which == "all"} {
    set percent_ipro 0
    set percent_ibut 0
    set percent_acam 0
    set percent_acetipam 0
  }
  ::druggability::Validate_percent_total
}

proc ::druggability::Validate_percent_total {} {
  # change the bacground color of the entry box showing total
  # mistyrose is invalid, lavender if valid
  variable percent_ipro
  variable percent_ibut
  variable percent_acam
  variable percent_acetipam
  variable percent_total
  variable w
  set total [expr $percent_acam + $percent_ibut + $percent_ipro + $percent_acetipam]
  if {$total != $percent_total} {
    set percent_total $total
    if {$percent_total == 0 || $percent_total == 100} {
      $w.main_frame.prepare.probe_options.total_percent configure \
        -disabledbackground lavender
    } else {
      $w.main_frame.prepare.probe_options.total_percent configure \
        -disabledbackground mistyrose
    }
  }
  return 1
}

proc ::druggability::Switch_mode {which_mode} {
  # change GUI layout
  variable w
  pack forget $w.main_frame.prepare $w.main_frame.process $w.main_frame.analyze $w.main_frame.evaluate $w.main_frame.visualize
  pack $w.main_frame.$which_mode -side top -padx 0 -pady 0 -fill x -expand 1
  variable interface $which_mode
}


proc ::druggability::Assess_druggability {} {
  variable pyprefix
  variable pythonexe
  variable grid_list
  variable outputdir
  if {[llength $grid_list] == 0} {
    tk_messageBox -type ok -title "ERROR" \
      -message "At least one grid file must be specified."
    return 0
  }

  if {$pyprefix == ""} {
    tk_messageBox -type ok -title "ERROR" \
      -message "File prefix must be specified."
    return 0
  }
  if {$pythonexe == ""} {
    set pythonexe [::ExecTool::find -interactive python]
  }

  ::druggability::Write_python
  exec $pythonexe [file join "$outputdir" "$pyprefix.py"] critical
  variable pdb_heavyatoms [file join "$outputdir" "$pyprefix\_heavyatoms.pdb"]
  variable pdb_hotspots [file join "$outputdir" "$pyprefix" "$pyprefix\_all_hotspots.pdb"]
  if {![file exists $pdb_heavyatoms]} {
    set pdb_heavyatoms "[pwd]/$pyprefix\_heavyatoms.pdb"
  }

  ::druggability::Load_results
}

proc ::druggability::Load_protein {} {
  # load protein and set molid
  if {$::druggability::protein_pdb != "" && $::druggability::protein_psf != ""} {
    set ::druggability::molid [mol new $::druggability::protein_psf]
    mol addfile $::druggability::protein_pdb type pdb molid $::druggability::molid
    return 1
  } else {
    tk_messageBox -type ok -title "ERROR" \
      -message "Both PSF and PDB files must be specified."
    return 0
  }
}

proc ::druggability::Load_system {} {
  # load protein and set sysid
  if {$::druggability::system_pdb != "" && $::druggability::system_psf != ""} {
    set ::druggability::sysid [mol new $::druggability::system_psf]
    mol addfile $::druggability::system_pdb type pdb sysid $::druggability::sysid
    return 1
  } else {
    tk_messageBox -type ok -title "ERROR" \
      -message "Both PSF and PDB files must be specified."
    return 0
  }
}

proc ::druggability::Show_system_selection {} {
  # display system selection
  # selection as spheres and protein as tube
  if {$::druggability::sysid < 0} {
    if {![::druggability::Load_system]} {return}
  }
  if {[molinfo $::druggability::sysid get numreps] > 0} {
    mol showrep $::druggability::sysid 0 off
  }
  for {set i [molinfo $::druggability::sysid get numreps]} {$i > 0} {incr i -1} {
    #mol showrep $::druggability::sysid $i off
    mol delrep $i $::druggability::sysid
  }
  mol addrep $::druggability::sysid
  set repid [expr [molinfo $::druggability::sysid get numreps] -1]
  mol modstyle $repid $::druggability::sysid Tube
  mol modcolor $repid $::druggability::sysid Structure

  mol addrep $::druggability::sysid
  incr repid
  mol modstyle $repid $::druggability::sysid Vdw 0.4
  mol modcolor $repid $::druggability::sysid Name
  mol modselect $repid $::druggability::sysid $::druggability::system_selstr
}


proc ::druggability::Process_system {} {

  # HOW THE CODE WORKS
  # The code will
  # (1)   Reads PSF, PDB and DCD files
  # (2)   TRANSLATES the geometric center of the system of interest to the origin
  # (3)   writes heavy atoms of the system (excluding solvent) for visualization
  # (4)   WRAPs solvent molecules into the simulation box
  # (5)   ROTATEs the system to minimize rms
  # (6)   writes aligned DCD files
  #
  # NOTE: The particular order of TRANSLATION, WRAPPING, and ROTATION is important
  #       To be correctly done, WRAPPING needs to come before ROTATION.
  # NOTE 2: After this step is complete, DCD file contains all coordinate
  #       information. The original DCD files may be deleted, but it should be
  #       noted that it is not possible to restore the rectangular periodic box
  #       from the aligned DCD file. Hence, a snapshot of the system from the
  #       middle of the aligned DCD cannot be used to continue the simulations.
  #       If you have such plans, keep the original files.
  #       If you want to start a simulation from the last snapshot,
  #       you can instead use sim1/PROTEINandPROBES.coor file.
  #       So, you may delete original DCDs, but be cautioned.
  variable w
  variable sysid
  variable system_psf
  variable system_pdb
  variable system_dcds
  variable system_selstr
  variable grid_spacing
  variable grid_water
  variable grid_hydrophobic
  variable grid_polar
  variable grid_negative
  variable grid_positive
  variable use_volmap
  variable run_python
  variable prefix
  variable pyprefix
  variable wrap_solvent
  variable save_aligned_system
  variable save_aligned_protein
  variable pdb_hotspots
  variable pdb_heavyatoms
  variable probe_contact_dist
  variable grid_list
  variable outputdir
  global env

  if {$::druggability::system_pdb == "" || $::druggability::system_psf == ""} {
    tk_messageBox -type ok -title "ERROR" \
      -message "Both PSF and PDB files must be specified."
    return
  }

  if {[string length [string trim $system_dcds]] == 0} {
    tk_messageBox -type ok -title "ERROR" \
      -message "At least one DCD file must be specified."
    return
  }

  if {$::druggability::sysid < 0} {
    if {[::druggability::Load_system] == 0} { return }
  }

  set pyprefix $prefix

  set log_file [open [file join "$outputdir" "$prefix.log"] a]
  puts $log_file "---==## [clock format [clock seconds]] #==---"
  puts $log_file "Version: $::druggability::version"
  puts $log_file "Info: Log file is opened for logging analysis of $prefix."


  mol off top
  foreach dcd_file $system_dcds {
    mol addfile $dcd_file type dcd waitfor all
    puts $log_file "Input: $dcd_file is loaded."
  }
  mol on top

  puts $log_file "Input: [expr [molinfo top get numframes] -1] frames have been loaded."

  puts $log_file "Selection: $system_selstr"
  set protein [atomselect top "$system_selstr"]
  puts $log_file "Selection: [$protein num] atoms are selected"
  set selected_atoms ""
  foreach ich [$protein get chain] ires [$protein get resid] nres [$protein get resname] natom [$protein get name] {
    lappend selected_atoms "$ich\_$nres$ires\_$natom"
  }
  puts $log_file "Selection: $selected_atoms"

  set num_steps [molinfo top get numframes]
  if {$wrap_solvent == 1 } {
    # 2) Eliminate protein translation translation
    #   use the selection to center the protein at the origin
    set protein [atomselect top "$system_selstr"]
    set all [atomselect top "all"]

    for {set frame 0} {$frame < $num_steps} {incr frame} {
      $protein frame $frame
      $all frame $frame
      $all moveby [vecmul {-1 -1 -1} [measure center $protein]]
    }
    puts $log_file "Trajectory: Geometric center of the selection has been moved to the origin."


  # 6) wrap solvent molecules back into the box
  #   This selection tells PBCtools to wrap SOLVENT and ION secments
    # pbc wrap -molid top -first 1 -last last -sel "segid XXX PROB ION \"WT.*\" or water or ion"
    pbc wrap -molid top -first 1 -last last -sel "all" -compound fragment -center origin
    puts $log_file "Trajectory: Solvent and probe molecules out of the original simulation box have been wrapped."
  }


  # 7) Eliminate rigid-body rotation the system to minimize RMSD
  #   use frame 0 for the reference
  set reference [atomselect top "$system_selstr" frame 0]
  #   the frame being compared
  set compare [atomselect top "$system_selstr"]
  #   note that after calculation of rotation matrix ALL atoms will be rotated
  set all [atomselect top "all"]
  for {set frame 1} {$frame < $num_steps} {incr frame} {
          # get the correct frame
          $compare frame $frame
          $all frame $frame
          # compute the transformation
          set trans_mat [measure fit $compare $reference]
          # do the alignment
          $all move $trans_mat
  }
  puts $log_file "Trajectory: Alignment is complete."

  # 5) write heavy atoms excluding WT* ION and XXX segments
  set protein [atomselect top "noh and not water and not segid XXX PROB ION \"WT.*\"" frame 0]
  $protein writepdb [file join $outputdir "$prefix\_heavyatoms.pdb"]
  puts $log_file "Output: System heavy atoms are written into file $prefix\_heavyatoms.pdb."


  if {$save_aligned_system} {
    # 8) Save the trajectory of entire system and only the protein
    #   Begin saving from the first frame, remember we loaded PDB file too.
    animate write dcd [file join $outputdir $prefix\_aligned.dcd] beg 1
    puts $log_file "Trajectory: System trajectory is saved as $prefix\_aligned.dcd."
  }

  if {$save_aligned_protein} {
    #   save the protein trajectory
    set protein [atomselect top "not water and not segid ION \"WT.*\""]
    animate write dcd [file join $outputdir $prefix\_protein_aligned.dcd] beg 1 sel $protein
    puts $log_file "Trajectory: Protein trajectory is saved as $prefix\_protein_aligned.dcd."
  }

  if {$use_volmap && [[atomselect top "resname IPRO PRO2 IBUT IBTN IPAM ACAM ACET ACTT"] num]} {
    #   Determine the size of the system
    set selWater [atomselect top "water" frame 0]
    set minmaxW [measure minmax $selWater]
    puts $log_file "Grid: Minimum and maximum coordinates: $minmaxW"
    set minW [lindex $minmaxW 0]
    set maxW [lindex $minmaxW 1]
    set minWx [lindex $minW 0]
    set minWy [lindex $minW 1]
    set minWz [lindex $minW 2]
    set maxWx [lindex $maxW 0]
    set maxWy [lindex $maxW 1]
    set maxWz [lindex $maxW 2]

    set xLength [expr ($maxWx - $minWx)]
    set yLength [expr ($maxWy - $minWy)]
    set zLength [expr ($maxWz - $minWz)]

    set nX [expr int([expr [expr ceil($xLength)]/$grid_spacing])]
    set nY [expr int([expr [expr ceil($yLength)]/$grid_spacing])]
    set nZ [expr int([expr [expr ceil($zLength)]/$grid_spacing])]


    animate delete beg 0 end 0 $sysid
    animate goto start
    set grid_list [list]
    foreach resname {"IPRO PRO2" "IBUT IBTN" "IPAM" "ACAM" "ACET ACTT"} {
      if {[[atomselect top "resname $resname"] num] > 0} {
        #volmap occupancy [atomselect top "resname $resname and name C2 and same residue as within 4 of protein"] -points -allframes -res 0.5 -combine avg -minmax [measure minmax $selWater] -checkpoint 0 -o $prefix\_$resname\_within.dx
        set moveback 0
        if {[[atomselect top "resname $resname and name C2 and same residue as (exwithin $probe_contact_dist of protein)"] num] == 0} {
          set moveback 1
          set movetocenter [atomselect top "resname $resname and name C2"]
          set movetocenter [atomselect top "index [lindex [$movetocenter get index] 0]"]
          set movelocation [measure center $movetocenter]
          set movetocenter [atomselect top "same residue as index [lindex [$movetocenter get index] 0]"]
          $movetocenter moveby [vecmul {-1 -1 -1} $movelocation]
        }
        volmap occupancy [atomselect top "resname $resname and name C2 and same residue as (exwithin $probe_contact_dist of protein)"] -points -allframes -res $grid_spacing -combine avg -minmax $minmaxW -checkpoint 0 -o [file join "$outputdir" "$prefix\_[lindex $resname 0].dx"]
        if {$moveback} {
          $movetocenter moveby $movelocation
        }
        puts $log_file "Grid: Grid file for [lindex $resname 0] is written as $prefix\_[lindex $resname 0].dx."
        lappend grid_list "[lindex $resname 0] $prefix\_[lindex $resname 0].dx"

      }
    }


    if {$grid_water} {
      volmap occupancy [atomselect top "noh water and (exwithin 4 of noh protein)"] -points -allframes -res $grid_spacing -combine avg -minmax $minmaxW -checkpoint 0 -o [file join "$outputdir" "$prefix\_water.dx"]
    }
    if {$grid_hydrophobic} {
      volmap occupancy [atomselect top "resname IPRO PRO2 IBUT IBTN and name C1 C2 C3 C4 and (exwithin 4 of noh protein)"] -points -allframes -res $grid_spacing -combine avg -minmax $minmaxW -checkpoint 0 -o [file join "$outputdir" "$prefix\_hydrophobic.dx"]
    }

    if {$grid_polar && [[atomselect top "resname ACAM IPRO PRO2"] num] > 0} {
      volmap occupancy [atomselect top "resname ACAM IPRO PRO2 and name OH2 O3 N4 and (exwithin 4 of noh protein)"] -points -allframes -res $grid_spacing -combine avg -minmax $minmaxW -checkpoint 0 -o [file join "$outputdir" "$prefix\_polar.dx"]
    }
    if {$grid_negative && [[atomselect top "resname ACET ACTT"] num] > 0} {
      volmap occupancy [atomselect top "resname ACET ACTT and name O3 O4 and (exwithin 4 of noh protein)"] -points -allframes -res $grid_spacing -combine avg -minmax $minmaxW -checkpoint 0 -o [file join "$outputdir" "$prefix\_negative.dx"]
    }
    if {$grid_positive && [[atomselect top "resname IPAM"] num] > 0} {
      volmap occupancy [atomselect top "resname IPAM and name N4 and (exwithin 4 of noh protein)"] -points -allframes -res $grid_spacing -combine avg -minmax $minmaxW -checkpoint 0 -o [file join "$outputdir" "$prefix\_positive.dx"]
    }

    if {$run_python} {
      puts $log_file "Python: Running Python file."
      ::druggability::Assess_druggability
    } else {
      puts $log_file "Python: $prefix.py is written to perform the rest of the analysis."
      ::druggability::Write_python
    }

  }

  close $log_file

  ::druggability::Logview [file join "$outputdir" "$prefix.log"]

  set ::druggability::sysid -1

  tk_messageBox -type ok -title "Analysis Complete" \
    -message "Trajectories are processed. See $prefix.log file."

}

proc ::druggability::Get_python_header {} {
  global PACKAGE_PATH
  global env
  variable pyheader "#!/usr/bin/python\n"
  variable pyheader "$pyheader\nimport os.path"
  variable pyheader "$pyheader\nimport sys\n"
  variable pyheader "$pyheader\nPYTHONPATH = '$env(PYTHONPATH)'\n"
  variable pyheader "$pyheader\nfor path in PYTHONPATH.split(':'):"
  variable pyheader "$pyheader\n    if path and not path in sys.path:"
  variable pyheader "$pyheader\n        sys.path.append(path)"
  variable pyheader "$pyheader\n# If Python druggability package was not installed, enter the path to contents of the tarball"
  variable pyheader "$pyheader\n# e.g.  PACKAGE_PATH = '/home/username/druggability/lib'"
  variable pyheader "$pyheader\nPACKAGE_PATH = '$PACKAGE_PATH'\n"
  variable pyheader "$pyheader\nif PACKAGE_PATH:"
  variable pyheader "$pyheader\n    sys.path.append(PACKAGE_PATH)"
  variable pyheader "$pyheader\ntry:"
  variable pyheader "$pyheader\n    from druggability import *"
  variable pyheader "$pyheader\nexcept ImportError:"
  variable pyheader "$pyheader\n    raise ImportError('druggability package was not found. Edit {0:s} '"
  variable pyheader "$pyheader\n                      'to proceed with calculation'.format(__file__))\n"
  variable pyheader "$pyheader\nif len(sys.argv) > 1:"
  variable pyheader "$pyheader\n  # for VMD use. if console logging is enabled, VMD raises an error"
  variable pyheader "$pyheader\n  verbose = sys.argv\[1\]"
  variable pyheader "$pyheader\nelse:"
  variable pyheader "$pyheader\n  verbose = 'info'"

  return $pyheader
}

proc ::druggability::Write_python {} {
  variable dia_merge_radius
  variable dia_temp
  variable dia_max_charge
  variable dia_delta_g
  variable dia_n_solutions
  variable dia_n_probes
  variable dia_min_n_probes
  variable dia_low_affinity
  variable dia_n_frames
  variable dia_n_charged
  variable grid_list
  variable pyprefix
  variable pypickle
  variable pyheader
  variable outputdir
  global env

  set py_file [open [file join "$outputdir" "$pyprefix.py"] w]
  puts $py_file "[::druggability::Get_python_header]"
  puts $py_file "probe_grids = \["
  foreach agrid $grid_list {
    set resname [string range $agrid 0 3]
    set gridfile [string range $agrid 5 end]
    puts $py_file "('$resname', '[file join $outputdir $gridfile]'),"
  }
  puts $py_file "\]"
  #puts $py_file "reinit = False"
  #puts $py_file "# initialize analysis with name and workdir"
  #puts $py_file "if os.path.join('$pyprefix', '$pyprefix' + '.dso.gz'):"
  #puts $py_file "    dia = pickler(os.path.join('$pyprefix', '$pyprefix' + '.dso.gz'), verbose=verbose)"
  #puts $py_file "    for probe_type, grid_file in probe_grids:"
  #puts $py_file "        probe = dia.get_probe(probe_type)
  #puts $py_file "else:"
  #puts $py_file "   reinit = True"
  #puts $py_file "if "
  puts $py_file "dia = DIA('$pyprefix', workdir='[file join $outputdir $pyprefix]', verbose=verbose)"
  puts $py_file "# check parameters for their correctness"
  puts $py_file "dia.set_parameters(temperature=$dia_temp) # K (productive simulation temperature)"
  puts $py_file "dia.set_parameters(delta_g=$dia_delta_g) # kcal/mol (probe binding hotspots with lower values will be evaluated)"
  puts $py_file "dia.set_parameters(n_probes=$dia_n_probes) # (number of probes to be merged to determine achievable affinity of a potential site)"
  puts $py_file "dia.set_parameters(min_n_probes=$dia_min_n_probes) # (minimum number of probes to be merged for an acceptable soltuion)"
  puts $py_file "dia.set_parameters(merge_radius=$dia_merge_radius) # A (distance within which two probes will be merged)"
  puts $py_file "dia.set_parameters(low_affinity=$dia_low_affinity) # microMolar (potential sites with affinity better than this value will be reported)"
  puts $py_file "dia.set_parameters(n_solutions=$dia_n_solutions) # (number of drug-size solutions to report for each potential binding site)"
  puts $py_file "dia.set_parameters(max_charge=$dia_max_charge) # (maximum absolute total charge, where total charge is occupancy weighted sum of charges on probes)"
  puts $py_file "dia.set_parameters(n_charged=$dia_n_charged) # (maximum number of charged hotspots in a solution)"
  puts $py_file "dia.set_parameters(n_frames=$dia_n_frames) # number of frames (if volmap was used (.dx), 1 is correct)"
  puts $py_file "for probe_type, grid_file in probe_grids: dia.add_probe(probe_type, grid_file)"
  puts $py_file "# Do the calculations"
  puts $py_file "dia.perform_analysis()"
  puts $py_file "dia.pickle()"
  puts $py_file "# Evaluate a ligand. Be sure that the ligand bound structure is superimposed"
  puts $py_file "# onto PROTEIN_heavyatoms.pdb"
  puts $py_file "#dia.evaluate_ligand('only_ligand_coordinates.pdb') # remove # sign from the beginning of line"
  close $py_file
  set pypickle "$pyprefix/$pyprefix.dso.gz"
}

proc ::druggability::Write_pyligand {} {
  variable pypickle
  variable pyheader
  variable pyprefix
  variable ligandpdbs
  variable dial_radius
  variable dial_deltag
  global env
  set pyprefix [lindex [split [lindex [file split $pypickle] end] "."] 0]
  variable outputdir [file join {*}[lrange [file split $pypickle] 0 end-2]]


  set py_file [open "$pyprefix\_ligand.py" w]
  puts $py_file "[::druggability::Get_python_header]"
  puts $py_file "# initialize analysis with name and workdir"
  puts $py_file "dia = pickler('$pypickle', verbose=verbose)"

  puts $py_file "# Evaluate a ligand. Be sure that the ligand bound structure is superimposed"
  puts $py_file "# onto PROTEIN_heavyatoms.pdb"
  foreach lpdb $ligandpdbs {
    puts $py_file "dia.evaluate_ligand('[file join $outputdir $lpdb]', radius=$dial_radius, delta_g=$dial_deltag)"
  }
  close $py_file
}

proc ::druggability::Evaluate_ligand {} {
  variable pypickle
  variable ligandpdbs
  variable probe_radius_scale

  if {[string length [string trim $pypickle]] == 0} {
    tk_messageBox -type ok -title "ERROR" \
      -message "A DSO file must be specified."
    return
  }
  if {[llength $ligandpdbs] == 0} {
    tk_messageBox -type ok -title "ERROR" \
      -message "At least one ligand PDB file must be specified."
    return
  }

  ::druggability::Write_pyligand
  variable pyprefix
  variable pythonexe
  variable outputdir

  if {$pythonexe == ""} {
    set pythonexe [::ExecTool::find python -interactive]
  }
  exec $pythonexe [file join $outputdir $pyprefix\_ligand.py] critical


  variable high_resolution
  variable dia_merge_radius
  set resol 6
  if {$high_resolution} {
    set resol 50
  }

  color scale method RWB
  foreach lpdb $ligandpdbs {
    set hotid [mol new "$lpdb"]
    mol modstyle 0 $hotid Licorice 0.3 $resol $resol
    mol modcolor 0 $hotid Name

    set pdbfn [lindex [file split $lpdb] end]
    set hotid [mol new "$pyprefix/$pyprefix\_$pdbfn"]
    mol modstyle 0 $hotid DynamicBonds $dia_merge_radius 0.3 $resol
    mol modcolor 0 $hotid Molecule
    mol addrep $hotid
    mol modstyle 1 $hotid VDW 0.4 $resol
    mol modcolor 1 $hotid Beta
    mol addrep $hotid
    mol modstyle 2 $hotid VDW $probe_radius_scale $resol
    mol modcolor 2 $hotid Beta
    mol modmaterial 2 $hotid Transparent
  }
  set logfilename "$pyprefix/$pyprefix.log"
  if {[file exists $logfilename]} {
    ::druggability::Logview "[pwd]/$logfilename"
  }


}


proc ::druggability::Prepare_system {} {

  # WHAT IS NEW?
  # 2.1 - Bug fixes, and file checks
  # 2.0 - Allows setup of systems containing multiple probe tybes
  # 2.0 - Improved system setup provides lesser number of solvent atoms
  # 2.0 - Cleans up intermediate files
  # 2.0 - Outputs a log file for trouble shooting, and further intstructions
  # 2.0 - NAMD configuration files are prepared for a single or multiple
  #       simulations
  # HOW THE CODE WORKS
  # The code will
  # (1)   solvate the protein, or everything in the PDB/PSF files that you provide
  # (2)   add counter ions to neutralize the system
  # (3)   adjust the water/probe ratio to a predefined level (1 probe to 20 water)
  # (4)   write output files for each simulation
  #       Setups of multiple simulations differ only at random number seeds.
  #       This will be sufficient to result in a different trajectory.
  variable w
  variable protein_psf
  variable protein_pdb
  variable percent_ipro
  variable percent_ibut
  variable percent_acam
  variable percent_acetipam
  variable solvent_padding
  variable neutralize
  variable output_prefix
  variable write_conf
  variable n_sims
  variable sim_length
  variable par_files
  variable outputdir

  if {$outputdir != ""} {
      if {![file isdirectory $outputdir]} {
        if {[catch {file mkdir $outputdir}]} {
          tk_messageBox -type ok -title "ERROR" \
            -message "Could not make output folder: $outputdir"
          return

        }
      }
  }

  if {$::druggability::protein_pdb == "" || $::druggability::protein_psf == ""} {
    tk_messageBox -type ok -title "ERROR" \
      -message "Both PSF and PDB files must be specified."
    return
  }

  set percent_total [expr $percent_acam + $percent_ibut + $percent_ipro + $percent_acetipam]

  if {$percent_total != 100 && $percent_total != 0} {
    tk_messageBox -type ok -title "ERROR" \
      -message "Probe percentages must sum up to 100 or 0.\nCurrent total is $percent_total"
    return
  }

  #if {$solvent_padding < 4} {
  #  tk_messageBox -type ok -title "ERROR" \
  #    -message "Solvent box padding parameter must be larger than 4 A."
  #  return
  #}

  if {[string length [string trim $output_prefix]] == 0} {
    tk_messageBox -type ok -title "ERROR" \
      -message "Please enter a descriptive name (prefix) for the system."
    return
  }

  global env
  global DRUGGABILITY_PATH
  if {!([file exists [file join $DRUGGABILITY_PATH probe.psf]] &&
        [file exists [file join $DRUGGABILITY_PATH probe.pdb]] &&
        [file exists [file join $DRUGGABILITY_PATH probe.top]] &&
        [file exists [file join $DRUGGABILITY_PATH probe.prm]])} {
    tk_messageBox -type ok -title "ERROR" \
      -message "One of probe psf, pdb, top, or prm files is not found in $DRUGGABILITY_PATH."
    return
  }

  set log_file [open [file join "$outputdir" "$output_prefix.log"] a]
  puts $log_file "---==## [clock format [clock seconds]] #==---"
  puts $log_file "Version: $::druggability::version"
  puts $log_file "Info: Logging started for setup of $output_prefix."
  puts $log_file "Solvation: Box padding $solvent_padding A."
  set intermediate [file join "$outputdir" "intermediate"]
  if {$percent_total == 100} {
    # Note that the solvation padding is larger than what you want.
    # Extra waters and probe molecules will be removed after ions are added
    solvate $protein_psf $protein_pdb -t [expr $solvent_padding + 5] \
      -o "$intermediate" -rotate -rotinc 10  \
      -spdb [file join $DRUGGABILITY_PATH probe.pdb] \
      -spsf [file join $DRUGGABILITY_PATH probe.psf] \
      -stop [file join $DRUGGABILITY_PATH probe.top] \
      -ks "name OH2" -ws 62.3572
  } else {
      solvate $protein_psf $protein_pdb -t $solvent_padding \
      -o $intermediate -rotate -rotinc 10
  }

  set protein [atomselect top "not segid \"WT.*\""]
  $protein writepdb $protein_pdb

  # DELETE solvated molecule

  if {$neutralize} {
    set totalcharge 0
    foreach charge [[atomselect top "all"] get charge] {
      set totalcharge [expr $totalcharge + $charge]
    }
    # number of CL and NA atoms are determined
    puts $log_file "Ionization: System has a total charge of $totalcharge electrons."
    if {$totalcharge > 0} {
        set nna 0
        set ncl [expr round($totalcharge)]
        puts $log_file "Ionization: $ncl chloride ions will be added."
    } else {
        set ncl 0
        set nna [expr -1 * round($totalcharge)]
        puts $log_file "Ionization: $nna sodium ions will be added."
    }
    if {$ncl > 0 | $nna > 0} {
        autoionize -psf "$intermediate.psf" -pdb "$intermediate.pdb" \
        -o "$intermediate" -from 5 -between 5 -ncl $ncl -nna $nna -seg ION
        puts $log_file "Ionization: System is ionized to become neutral."
    }
  }

  #===============================================================================
  # START - ADJUST RELATIVE NUMBER of WATER and PROBE molecules
  # minimum and maximum coordinates for PROTEIN is calculated.
  # if a molecule other than a PROTEIN is of interest, change selection in the next line.
  set protein [atomselect top "not water and not segid ION \"WT.*\""]
  set minmaxP [measure minmax $protein]
  set minP [lindex $minmaxP 0]
  set maxP [lindex $minmaxP 1]
  set minPx [lindex $minP 0]
  set minPy [lindex $minP 1]
  set minPz [lindex $minP 2]
  set maxPx [lindex $maxP 0]
  set maxPy [lindex $maxP 1]
  set maxPz [lindex $maxP 2]

  set pad $solvent_padding
  set selWstring "water and name OH2 and x > [expr $minPx-$pad] and y > [expr $minPy-$pad] and z > [expr $minPz-$pad] and x < [expr $maxPx+$pad] and y < [expr $maxPy+$pad] and z < [expr $maxPz+$pad]"
  # select waters in the box of requested size
  set selWater [atomselect top "$selWstring"]
  set nWater [$selWater num]
  set indicesWater [$selWater get index]

  if {$percent_total == 100} then {
    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # THE RATIO OF WATER TO PROBE MOLECULE (INTEGERS)
    # - 20 is an ideal ratio. It has worked fine in the test cases.
    # - Less than 20 leads to underestimates.
    # - This parameter may be set by the user, but seems unnecessarily complicating
    set selPstring "resname IPRO and name OH2 and x > [expr $minPx-$pad] and y > [expr $minPy-$pad] and z > [expr $minPz-$pad] and x < [expr $maxPx+$pad] and y < [expr $maxPy+$pad] and z < [expr $maxPz+$pad]"
    set water2probeRatio 20

    set percent_acet [expr $percent_acetipam / 2]
    set percent_ipam [expr $percent_acetipam - $percent_acet]
    if {[expr $percent_ipro % 100] == 0 && [expr $percent_ibut % 100] == 0 &&
        [expr $percent_acam % 100] == 0 && [expr $percent_acet % 100] == 0 &&
        [expr $percent_ipam % 100] && 0} {
        set modWater $water2probeRatio
    } elseif {[expr $percent_ipro % 50] == 0 && [expr $percent_ibut % 50] == 0 &&
              [expr $percent_acam % 50] == 0 && [expr $percent_acet % 50] == 0 &&
              [expr $percent_ipam % 50] == 0} {
        set modWater [expr $water2probeRatio * 2]
    } elseif {[expr $percent_ipro % 25] == 0 && [expr $percent_ibut % 25] == 0 &&
              [expr $percent_acam % 25] == 0 && [expr $percent_acet % 25] == 0 &&
              [expr $percent_ipam % 25] == 0} {
        set modWater [expr $water2probeRatio * 4]
    } elseif {[expr $percent_ipro % 20] == 0 && [expr $percent_ibut % 20] == 0 &&
              [expr $percent_acam % 20] == 0 && [expr $percent_acet % 20] == 0 &&
              [expr $percent_ipam % 20] == 0} {
        set modWater [expr $water2probeRatio * 5]
    } elseif {[expr $percent_ipro % 10] == 0 && [expr $percent_ibut % 10] == 0 &&
              [expr $percent_acam % 10] == 0 && [expr $percent_acet % 10] == 0 &&
              [expr $percent_ipam % 10] == 0} {
        set modWater [expr $water2probeRatio * 10]
    } else {
        set modWater [expr $water2probeRatio * $water2probeRatio]
    }

    set howManyMoreWater [expr $modWater - $nWater % $modWater]
    set pad 2.5
    set addWater [atomselect top "water and name OH2 and exwithin $pad of index $indicesWater"]
    while {[$addWater num] < $howManyMoreWater && $pad < [expr $solvent_padding+5] } {
        set pad [expr $pad + 0.1]
        set addWater [atomselect top "water and name OH2 and exwithin $pad of index $indicesWater"]
    }
    set indicesWater "$indicesWater [lrange [$addWater get index] 0 [expr $howManyMoreWater - 1]]"

    # select probes
    set selProbe [atomselect top "$selPstring"]
    set nProbe [$selProbe num]
    set indicesProbe [$selProbe get index]

    # select more probes
    set howManyMoreProbe [expr [llength $indicesWater] / $water2probeRatio - $nProbe]
    puts $howManyMoreProbe
    set pad 7.0
    set addProbe [atomselect top "resname IPRO and name OH2 and exwithin $pad of index $indicesProbe"]
    while {[$addProbe num] < $howManyMoreProbe && $pad < [expr $solvent_padding+5]} {
        set pad [expr $pad + 0.25]
        set addProbe [atomselect top "resname IPRO and name OH2 and exwithin $pad of index $indicesProbe"]
    }
    set indicesProbe "$indicesProbe [lrange [$addProbe get index] 0 [expr $howManyMoreProbe - 1]]"
    set indicesProbe [lsort $indicesProbe]
    set nProbe [llength $indicesProbe]
    puts $log_file "Statistics: System contains [llength $indicesWater] water and $nProbe probe molecules"
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # END - ADJUST RELATIVE NUMBER of WATER and PROBE molecules
  }

  # WRITE PDB files for SOLVENT and IONS
  # PSFGEN
  package require psfgen
  psfcontext reset
  if {$percent_total == 100} {topology [file join $DRUGGABILITY_PATH probe.top] }
  topology [file join $env(CHARMMTOPDIR) top_all27_prot_lipid_na.inp]
  readpsf $protein_psf
  coordpdb $protein_pdb

  if {$percent_total == 100} then {
  #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  # START - PROBE RENUMBER and MUTATE
  # Renumber probe molecule resids starting from 1
  # This is useful when mutating probe molecules
    set residProbe 1
    foreach indexProbe $indicesProbe {
      set sel [atomselect top "same residue as index $indexProbe"]
      $sel set resid $residProbe
      $sel set chain "X"
      incr residProbe
    }
    # Write probe molecules into an intermediate file
    set sel [atomselect top "same residue as index $indicesProbe"]
    $sel writepdb "$intermediate.pdb"

    # Calculate number of copies of each probe
    set howmanyIsopropanol [::tcl::mathfunc::int [expr $nProbe * $percent_ipro / 100.0]]
    puts $log_file "Statistics: System contains $howmanyIsopropanol isopropanol molecules."
    set howmanyIsobutane [::tcl::mathfunc::int [expr $nProbe * $percent_ibut / 100.0]]
    puts $log_file "Statistics: System contains $howmanyIsobutane isobutane molecules."
    set howmanyAcetamide [::tcl::mathfunc::int [expr $nProbe * $percent_acam / 100.0]]
    puts $log_file "Statistics: System contains $howmanyAcetamide acetamide molecules."
    set howmanyAcetate [::tcl::mathfunc::int [expr $nProbe * $percent_acet / 100.0]]
    puts $log_file "Statistics: System contains $howmanyAcetate acetate molecules."
    set howmanyIsopropylamine [::tcl::mathfunc::int [expr $nProbe * $percent_ipam / 100.0]]
    puts $log_file "Statistics: System contains $howmanyIsopropylamine isopropylamine molecules."

    # Perform mutations of IPRO to other probes if requested
    set residProbe 1
    segment XXX {
      pdb "$intermediate.pdb"
      if {$percent_ipro < 100} {
        while {$residProbe <= $nProbe} {
          set whichProbe [::tcl::mathfunc::int [expr rand() * 5]]
          if {$whichProbe == 0 && $howmanyIsopropanol > 0} {
            incr residProbe
            incr howmanyIsopropanol -1
          } elseif {$whichProbe == 1 && $howmanyIsobutane > 0} {
            mutate $residProbe IBUT
            incr residProbe
            incr howmanyIsobutane -1
          } elseif {$whichProbe == 2 && $howmanyAcetamide > 0} {
            mutate $residProbe ACAM
            incr residProbe
            incr howmanyAcetamide -1
          } elseif {$whichProbe == 3 && $howmanyAcetate > 0} {
            mutate $residProbe ACET
            incr residProbe
            incr howmanyAcetate -1
          } elseif {$whichProbe == 4 && $howmanyIsopropylamine > 0} {
            mutate $residProbe IPAM
            incr residProbe
            incr howmanyIsopropylamine -1
          }
        }
      }
    }

  coordpdb "$intermediate.pdb" XXX
  # END - PROBE RENUMBER and MUTATE
  #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  }

  # Write and build ION segment
  set sel [atomselect top "segid ION"]
  $sel writepdb "$intermediate.pdb"
  segment ION {pdb "$intermediate.pdb"}
  coordpdb "$intermediate.pdb" ION

  # Write and build WT* segments
  set sel [atomselect top "segid \"WT.*\""]
  set segidWTs [lsort -unique [$sel get segid]]
  foreach segidWT $segidWTs {
    set sel [atomselect top "segid $segidWT and index $indicesWater"]
    # While at it, renumber water molecule resids starting from 1 for each segment
    set residWater 1
    foreach indexWater [$sel get index] {
      set sel [atomselect top "same residue as index $indexWater"]
      $sel set resid $residWater
      incr residWater
    }
    set sel [atomselect top "segid $segidWT and (same residue as index $indicesWater)"]
    $sel writepdb "$intermediate.pdb"
    segment $segidWT {pdb "$intermediate.pdb"}
    coordpdb "$intermediate.pdb" $segidWT
  }

  # Guess coordinates of mutated probe molecules
  guesscoord

  # Write structure and coordinate data
  writepsf [file join "$outputdir" "$output_prefix.psf"]
  puts $log_file "Output: Structural data is written into $output_prefix.psf file."
  writepdb [file join "$outputdir" "$output_prefix.pdb"]
  puts $log_file "Output: Coordinate data is written into $output_prefix.pdb file."


  #============================================================================
  # SET OCCUPANCY and BETA columns for constraints
  foreach i [molinfo list] { mol off $i }

  mol new [file join "$outputdir" "$output_prefix.psf"]
  mol addfile [file join "$outputdir" "$output_prefix.pdb"]


  # Write PDB file with constraints
  set all [atomselect top "all"]
  $all set beta 0
  $all set occupancy 0
  # protein heavy atoms BETA 1
  set protein [atomselect top "noh and not water and not segid XXX ION \"WT.*\""]
  $protein set beta 1
  # alpha carbons OCCUPANCY 1
  set protein [atomselect top "protein and name CA and not segid XXX ION \"WT.*\""]
  $protein set occupancy 1
  set geomcent [measure center $protein]
  set all [atomselect top "all"]
  $all moveby [vecmul {-1 -1 -1} $geomcent]
  $all writepdb [file join "$outputdir" "$output_prefix.pdb"]
  set protein [atomselect top "noh and not water and not segid XXX ION \"WT.*\""]
  $protein writepdb [file join "$outputdir" "$output_prefix\_heavyatoms.pdb"]

  # REPRESENTATIONS


  mol addrep top
  mol modstyle 1 top NewCartoon
  mol modcolor 1 top Structure
  mol addrep top
  mol modstyle 2 top Licorice
  mol modselect 2 top "noh and not protein and not water and not ion and not segid XXX ION \"WT.*\""
  mol addrep top
  mol modstyle 3 top Vdw
  mol modselect 3 top "(water or ion) and not segid XXX ION \"WT.*\""
  mol addrep top
  mol modstyle 4 top Vdw
  mol modselect 4 top "segid ION"
  if {$percent_total == 100} {
    mol addrep top
    mol modstyle 5 top Licorice
    mol modselect 5 top "segid XXX"
  }

  #============================================================================
  # WRITE XSC FILE
  # Get box dimensions and print them
  set selWater [atomselect top "noh water"]
  set minmaxW [measure minmax $selWater]
  set minW [lindex $minmaxW 0]
  set maxW [lindex $minmaxW 1]
  set minWx [lindex $minW 0]
  set minWy [lindex $minW 1]
  set minWz [lindex $minW 2]
  set maxWx [lindex $maxW 0]
  set maxWy [lindex $maxW 1]
  set maxWz [lindex $maxW 2]

  if {$percent_total == 0} then {
    set desired_density 0.65
  } else {
    set desired_density 0.62
  }


  set total_mass [vecsum [[atomselect top "all"] get mass]]
  set dimScale [::tcl::mathfunc::pow [expr $total_mass / $desired_density / ($maxWx - $minWx) / ($maxWy - $minWy) / ($maxWz - $minWz)] [expr 1.0 / 3.0]]
  set xLength [expr ($maxWx - $minWx)*$dimScale]
  set yLength [expr ($maxWy - $minWy)*$dimScale]
  set zLength [expr ($maxWz - $minWz)*$dimScale]

  set xsc_file [open [file join "$outputdir" "$output_prefix.xsc"] w]
  puts $xsc_file "# NAMD extended system configuration output file"
  puts $xsc_file "#\$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z"
  puts $xsc_file "0 $xLength  0  0 0 $yLength  0 0  0 $zLength  0  0  0"
  close $xsc_file
  puts $log_file "Output: Extended system coordinates are written into $output_prefix.xsc file."

  puts $log_file "Statistics: System contains [[atomselect top all] num] atoms"
  puts $log_file "Statistics: Mass of the system is $total_mass amu"
  puts $log_file "Statistics: Density of the system is [expr $total_mass / $xLength / $yLength / $zLength] amu/A^3"

  #===============================================================================

  # DELETE intermediate files
  file delete "$intermediate.pdb"
  file delete "$intermediate.psf"
  file delete "$intermediate.log"
  puts $log_file "Cleanup: Intermediate coordinate files have been deleted."
  #===============================================================================
  if {$write_conf} {
    set sh_file [open [file join "$outputdir" "$output_prefix.sh"] w]
    set namd2path [::ExecTool::find "namd2"]
    if {$namd2path == ""} {
        puts $sh_file "NAMD=\"/path/to/namd2 +p4\""
      } else {
        puts $sh_file "NAMD=\"$namd2path +p4\""
    }
    set pardir [file join "$outputdir" "parameters"]
    if {![file exists "$pardir"]} {file mkdir "$pardir"}
    if {![file exists [file join "$pardir" "par_all27_prot_lipid_na.inp"]]} {
      file copy [file join $env(CHARMMPARDIR) par_all27_prot_lipid_na.inp] [file join "$pardir" "par_all27_prot_lipid_na.inp"]
    }
    if {$percent_acetipam > 0} {
      if {![file exists [file join "$pardir" "probe.prm"]]} {
        file copy [file join $DRUGGABILITY_PATH probe.prm] [file join "$pardir" "probe.prm"]
      }
    }
    set par_filenames [list]
    foreach par_file $par_files {
      set par_filename [lindex [file split $par_file] end]
      if {![file exists [file join "$pardir" "$par_filename"]]} {
        file copy $par_file [file join "$pardir" "$par_filename"]
      }
      lappend par_filenames $par_filename
    }

    puts $log_file "Simulation: Parameter files are copied into ./parameter folder."
    set minfix "_min"
    file mkdir [file join "$outputdir" "$output_prefix$minfix"]
    set namd_file [open [file join "$outputdir" "$output_prefix$minfix" "min.conf"] w]
    puts $namd_file "coordinates     ../$output_prefix.pdb"
    puts $namd_file "structure       ../$output_prefix.psf"
    puts $namd_file "paraTypeCharmm  on"
    puts $namd_file "parameters      ../parameters/par_all27_prot_lipid_na.inp"
    if {$percent_acetipam > 0} {puts $namd_file "parameters      ../parameters/probe.prm"}
    foreach par_filename $par_filenames {puts $namd_file "parameters      ../parameters/$par_filename"}
    puts $namd_file "outputname      $output_prefix"
    puts $namd_file "binaryoutput    no"
    puts $namd_file "restartname     $output_prefix"
    puts $namd_file "restartfreq     10000"
    puts $namd_file "binaryrestart   no"
    puts $namd_file "timestep        1.0"
    puts $namd_file "cutoff          10.0"
    puts $namd_file "switching       on"
    puts $namd_file "switchdist      8.0"
    puts $namd_file "pairlistdist    12.0"
    puts $namd_file "margin          1.0"
    puts $namd_file "exclude         scaled1-4"
    puts $namd_file "temperature     0"
    puts $namd_file "seed            12345"
    puts $namd_file "constraints     on"
    puts $namd_file "consref         ../$output_prefix.pdb"
    puts $namd_file "conskfile       ../$output_prefix.pdb"
    puts $namd_file "conskcol        B"
    puts $namd_file "constraintScaling  1.0"
    puts $namd_file "PME             yes"
    puts $namd_file "PMEGridSpacing  1.0"
    puts $namd_file "extendedSystem  ../$output_prefix.xsc"
    puts $namd_file "wrapWater       on"
    #puts $namd_file "wrapAll         on"
    puts $namd_file "minimize        2000"
    close $namd_file
    puts $sh_file "cd $output_prefix$minfix"
    puts $sh_file "\$NAMD min.conf > min.log"

    puts $log_file "Simulation: NAMD configuration files for minimization are written into folder $output_prefix$minfix."

    for {set i 1} {$i <= $n_sims} {incr i} {
      # MKDIR for each simulation
      if {$i == 1} {
        set suffix "_sim"
      } else {
        set suffix "_sim$i"
      }
      file mkdir [file join "$outputdir" "$output_prefix$suffix"]
      puts $sh_file "cd ../$output_prefix$suffix"
      set randomSeed [::tcl::mathfunc::int [expr rand() * 1000000]]

      set namd_file [open [file join "$outputdir" "$output_prefix$suffix" "eq1.conf"] w]
      puts $namd_file "coordinates     ../$output_prefix$minfix/$output_prefix.coor"
      puts $namd_file "structure       ../$output_prefix.psf"
      puts $namd_file "paraTypeCharmm  on"
      puts $namd_file "parameters      ../parameters/par_all27_prot_lipid_na.inp"
      if {$percent_acetipam > 0} {puts $namd_file "parameters      ../parameters/probe.prm"}
      foreach par_filename $par_filenames {puts $namd_file "parameters      ../parameters/$par_filename"}
      puts $namd_file "outputname      $output_prefix"
      puts $namd_file "binaryoutput    no"
      puts $namd_file "restartname     $output_prefix"
      puts $namd_file "restartfreq     2000"
      puts $namd_file "binaryrestart   no"
      puts $namd_file "DCDfreq         2000"
      puts $namd_file "DCDfile         eq1.dcd"
      puts $namd_file "outputEnergies  2000"
      puts $namd_file "timestep        2.0"
      puts $namd_file "fullElectFrequency 2"
      puts $namd_file "nonbondedFreq      1"
      puts $namd_file "rigidBonds      all"
      puts $namd_file "cutoff          10.0"
      puts $namd_file "switching       on"
      puts $namd_file "switchdist      8.0"
      puts $namd_file "pairlistdist    12.0"
      puts $namd_file "margin          1.0"
      puts $namd_file "exclude         scaled1-4"
      puts $namd_file "temperature     100"
      puts $namd_file "seed            $randomSeed"
      puts $namd_file "constraints     on"
      puts $namd_file "consref         ../$output_prefix.pdb"
      puts $namd_file "conskfile       ../$output_prefix.pdb"
      puts $namd_file "conskcol        B"
      puts $namd_file "constraintScaling  0.5"
      puts $namd_file "PME             yes"
      puts $namd_file "PMEGridSpacing  1.0"
      puts $namd_file "langevin         on"
      puts $namd_file "langevinTemp     100"
      puts $namd_file "langevinDamping  5"
      puts $namd_file "langevinHydrogen off"
      puts $namd_file "useGroupPressure      yes"
      puts $namd_file "useFlexibleCell       no"
      puts $namd_file "useConstantArea       no"
      puts $namd_file "useConstantRatio      no"
      puts $namd_file "langevinPiston        on"
      puts $namd_file "langevinPistonTarget  1.01325"
      puts $namd_file "langevinPistonPeriod  100.0"
      puts $namd_file "langevinPistonDecay   50.0"
      puts $namd_file "langevinPistonTemp    300"
      #puts $namd_file "rescaleFreq      100"
      puts $namd_file "extendedSystem  ../$output_prefix.xsc"
      puts $namd_file "wrapWater       on"
      #puts $namd_file "wrapAll         on"
      puts $namd_file "reinitvels      100"
      puts $namd_file "for \{set T 100\} \{\$T < 300\} \{incr T 10\} \{"
      #puts $namd_file "    rescaleTemp      \$T;"
      puts $namd_file "    langevinTemp      \$T;"
      puts $namd_file "    run              1000;"
      puts $namd_file "\}"
      #puts $namd_file "rescaleTemp      300;"
      puts $namd_file "langevinTemp     300"
      puts $namd_file "run           40000;"
      close $namd_file
      puts $sh_file "\$NAMD eq1.conf > eq1.log"

      if {$percent_total == 100} then {
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        set namd_file [open [file join "$outputdir" "$output_prefix$suffix" "eq2.conf"] w]
        puts $namd_file "coordinates     $output_prefix.coor"
        puts $namd_file "structure       ../$output_prefix.psf"
        puts $namd_file "paraTypeCharmm  on"
        puts $namd_file "parameters      ../parameters/par_all27_prot_lipid_na.inp"
        if {$percent_acetipam > 0} {puts $namd_file "parameters      ../parameters/probe.prm"}
        foreach par_filename $par_filenames {puts $namd_file "parameters      ../parameters/$par_filename"}
        puts $namd_file "outputname      $output_prefix"
        puts $namd_file "binaryoutput    no"
        puts $namd_file "restartname     $output_prefix"
        puts $namd_file "restartfreq     2000"
        puts $namd_file "binaryrestart   no"
        puts $namd_file "DCDfreq         2000"
        puts $namd_file "DCDfile         eq2.dcd"
        puts $namd_file "outputEnergies  2000"
        puts $namd_file "timestep        2.0"
        puts $namd_file "fullElectFrequency 2"
        puts $namd_file "nonbondedFreq      1"
        puts $namd_file "rigidBonds      all"
        puts $namd_file "cutoff          10.0"
        puts $namd_file "switching       on"
        puts $namd_file "switchdist      8.0"
        puts $namd_file "pairlistdist    12.0"
        puts $namd_file "margin          1.0"
        puts $namd_file "exclude         scaled1-4"
        puts $namd_file "velocities      $output_prefix.vel"
        puts $namd_file "seed            $randomSeed"
        puts $namd_file "constraints     on"
        puts $namd_file "consref         ../$output_prefix.pdb"
        puts $namd_file "conskfile       ../$output_prefix.pdb"
        puts $namd_file "conskcol        B"
        puts $namd_file "constraintScaling  1.0"
        puts $namd_file "PME             yes"
        puts $namd_file "PMEGridSpacing  1.0"
        #puts $namd_file "rescaleFreq      100"
        puts $namd_file "langevin         on"
        #puts $namd_file "langevinTemp     300"
        puts $namd_file "langevinDamping  5"
        puts $namd_file "langevinHydrogen off"
        puts $namd_file "extendedSystem  $output_prefix.xsc"
        puts $namd_file "wrapWater       on"
        #puts $namd_file "wrapAll         on"
        puts $namd_file "for \{set T 300\} \{\$T < 600\} \{incr T  10\} \{"
        #puts $namd_file "    rescaleTemp      \$T;"
        puts $namd_file "    langevinTemp     \$T;"
	      puts $namd_file "    run             1000;"
        puts $namd_file "\}"
        #puts $namd_file "rescaleTemp      600;"
        puts $namd_file "langevinTemp     600"
        puts $namd_file "run             300000;"
        puts $namd_file "for \{set T 570\} \{\$T >= 300\} \{incr T -30\} \{"
        puts $namd_file "    langevinTemp     \$T;"
        puts $namd_file "	   run             1000;"
        puts $namd_file "\}"
        close $namd_file
        puts $sh_file "\$NAMD eq2.conf > eq2.log"

        set namd_file [open [file join "$outputdir" "$output_prefix$suffix" "eq3.conf"] w]
        puts $namd_file "coordinates     $output_prefix.coor"
        puts $namd_file "structure       ../$output_prefix.psf"
        puts $namd_file "paraTypeCharmm  on"
        puts $namd_file "parameters      ../parameters/par_all27_prot_lipid_na.inp"
        if {$percent_acetipam > 0} {puts $namd_file "parameters      ../parameters/probe.prm"}
        foreach par_filename $par_filenames {puts $namd_file "parameters      ../parameters/$par_filename"}
        puts $namd_file "outputname      $output_prefix"
        puts $namd_file "binaryoutput    no"
        puts $namd_file "restartname     $output_prefix"
        puts $namd_file "restartfreq     2000"
        puts $namd_file "binaryrestart   no"
        puts $namd_file "DCDfreq         2000"
        puts $namd_file "DCDfile         eq3.dcd"
        puts $namd_file "outputEnergies  2000"
        puts $namd_file "timestep        2.0"
        puts $namd_file "fullElectFrequency 2"
        puts $namd_file "nonbondedFreq      1"
        puts $namd_file "rigidBonds      all"
        puts $namd_file "cutoff          10.0"
        puts $namd_file "switching       on"
        puts $namd_file "switchdist      8.0"
        puts $namd_file "pairlistdist    12.0"
        puts $namd_file "margin          1.0"
        puts $namd_file "exclude         scaled1-4"
        puts $namd_file "velocities      $output_prefix.vel"
        puts $namd_file "seed            $randomSeed"
        #puts $namd_file "constraints     on"
        #puts $namd_file "consref         ../$output_prefix.pdb"
        #puts $namd_file "conskfile       ../$output_prefix.pdb"
        #puts $namd_file "conskcol        O"
        #puts $namd_file "constraintScaling  0.01"
        puts $namd_file "PME             yes"
        puts $namd_file "PMEGridSpacing  1.0"
        puts $namd_file "langevin         on"
        puts $namd_file "langevinTemp     300"
        puts $namd_file "langevinDamping  5"
        puts $namd_file "langevinHydrogen off"
        puts $namd_file "useGroupPressure      yes"
        puts $namd_file "useFlexibleCell       no"
        puts $namd_file "useConstantArea       no"
        puts $namd_file "useConstantRatio      no"
        puts $namd_file "langevinPiston        on"
        puts $namd_file "langevinPistonTarget  1.01325"
        puts $namd_file "langevinPistonPeriod  100.0"
        puts $namd_file "langevinPistonDecay   50.0"
        puts $namd_file "langevinPistonTemp    300"
        puts $namd_file "extendedSystem  $output_prefix.xsc"
        puts $namd_file "wrapWater       on"
        #puts $namd_file "wrapAll         on"
        puts $namd_file "run                  300000"
        close $namd_file
        puts $sh_file "\$NAMD eq3.conf > eq3.log"
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      }

      set namd_file [open [file join "$outputdir" "$output_prefix$suffix" "sim.conf"] w]
      puts $namd_file "coordinates     $output_prefix.coor"
      puts $namd_file "structure       ../$output_prefix.psf"
      puts $namd_file "paraTypeCharmm  on"
      puts $namd_file "parameters      ../parameters/par_all27_prot_lipid_na.inp"
      if {$percent_acetipam > 0} {puts $namd_file "parameters      ../parameters/probe.prm"}
      foreach par_filename $par_filenames {puts $namd_file "parameters      ../parameters/$par_filename"}
      puts $namd_file "outputname      $output_prefix"
      puts $namd_file "binaryoutput    no"
      puts $namd_file "restartname     $output_prefix"
      puts $namd_file "restartfreq     2000"
      puts $namd_file "binaryrestart   no"
      puts $namd_file "DCDfreq         2000"
      puts $namd_file "DCDfile         sim.dcd"
      puts $namd_file "outputEnergies  2000"
      puts $namd_file "timestep        2.0"
      puts $namd_file "fullElectFrequency 2"
      puts $namd_file "nonbondedFreq      1"
      puts $namd_file "rigidBonds      all"
      puts $namd_file "cutoff          10.0"
      puts $namd_file "switching       on"
      puts $namd_file "switchdist      8.0"
      puts $namd_file "pairlistdist    12.0"
      puts $namd_file "margin          1.0"
      puts $namd_file "exclude         scaled1-4"
      puts $namd_file "velocities      $output_prefix.vel"
      puts $namd_file "seed            $randomSeed"
      puts $namd_file "PME             yes"
      puts $namd_file "PMEGridSpacing  1.0"
      puts $namd_file "langevin         on"
      puts $namd_file "langevinTemp     300"
      puts $namd_file "langevinDamping  5"
      puts $namd_file "langevinHydrogen off"
      puts $namd_file "useGroupPressure      yes"
      puts $namd_file "useFlexibleCell       no"
      puts $namd_file "useConstantArea       no"
      puts $namd_file "useConstantRatio      no"
      puts $namd_file "langevinPiston        on"
      puts $namd_file "langevinPistonTarget  1.01325"
      puts $namd_file "langevinPistonPeriod  100.0"
      puts $namd_file "langevinPistonDecay   50.0"
      puts $namd_file "langevinPistonTemp    300"
      puts $namd_file "extendedSystem  $output_prefix.xsc"
      puts $namd_file "wrapWater       on"
      #puts $namd_file "wrapAll         on"
      puts $namd_file "run                  [expr $sim_length * 500000]"
      close $namd_file
      puts $sh_file "\$NAMD sim.conf > sim.log"

      set namd_file [open [file join "$outputdir" "$output_prefix$suffix" "simrestart.conf"] w]
      puts $namd_file "coordinates     $output_prefix.coor"
      puts $namd_file "structure       ../$output_prefix.psf"
      puts $namd_file "paraTypeCharmm  on"
      puts $namd_file "parameters      ../parameters/par_all27_prot_lipid_na.inp"
      if {$percent_acetipam > 0} {puts $namd_file "parameters      ../parameters/probe.prm"}
      foreach par_filename $par_filenames {puts $namd_file "parameters      ../parameters/$par_filename"}
      puts $namd_file "outputname      $output_prefix"
      puts $namd_file "binaryoutput    no"
      puts $namd_file "restartname     $output_prefix"
      puts $namd_file "restartfreq     2000"
      puts $namd_file "binaryrestart   no"
      puts $namd_file "DCDfreq         2000"
      puts $namd_file "# DON'T forget to rename DCD files incrementally for each restart"
      puts $namd_file "DCDfile         sim1.dcd"
      puts $namd_file "outputEnergies  2000"
      puts $namd_file "timestep        2.0"
      puts $namd_file "firsttimestep   XXXXX this value is found in $output_prefix.coor"
      puts $namd_file "fullElectFrequency 2"
      puts $namd_file "nonbondedFreq      1"
      puts $namd_file "rigidBonds      all"
      puts $namd_file "cutoff          10.0"
      puts $namd_file "switching       on"
      puts $namd_file "switchdist      8.0"
      puts $namd_file "pairlistdist    12.0"
      puts $namd_file "margin          1.0"
      puts $namd_file "exclude         scaled1-4"
      puts $namd_file "velocities      $output_prefix.vel"
      puts $namd_file "seed            $randomSeed"
      puts $namd_file "PME             yes"
      puts $namd_file "PMEGridSpacing  1.0"
      puts $namd_file "langevin         on"
      puts $namd_file "langevinTemp     300"
      puts $namd_file "langevinDamping  5"
      puts $namd_file "langevinHydrogen off"
      puts $namd_file "useGroupPressure      yes"
      puts $namd_file "useFlexibleCell       no"
      puts $namd_file "useConstantRatio      no"
      puts $namd_file "langevinPiston        on"
      puts $namd_file "langevinPistonTarget  1.01325"
      puts $namd_file "langevinPistonPeriod  100.0"
      puts $namd_file "langevinPistonDecay   50.0"
      puts $namd_file "langevinPistonTemp    300"
      puts $namd_file "extendedSystem  $output_prefix.xsc"
      puts $namd_file "wrapWater       on"
      #puts $namd_file "wrapAll         on"
      puts $namd_file "run             XXXXX"
      close $namd_file
      puts $sh_file "#\$NAMD simrestart.conf > simrestart.log"

      puts $log_file "Simulation: NAMD configuration files for simulation $i are written into folder $output_prefix$suffix."
    }

    close $sh_file
  }
  puts $log_file "Simulation: Simulation folders also contains a restart file template."
  puts $log_file "Simulation: NAMD commands are written in $output_prefix.sh."


  close $log_file
  ::druggability::Logview [file join "$outputdir" "$output_prefix.log"]

  tk_messageBox -type ok -title "Setup Complete" \
    -message "Setup of $output_prefix is complete. See $output_prefix.log file."


}

proc drugui_tk {} {
  ::druggability::druggui
  #set ::druggability::which_mode [lindex $::druggability::titles [lsearch $::druggability::interfaces $::druggability::interface]]
  #::druggability::Switch_mode $::druggability::interface
  #return $::druggability::w
}
#druggability_tk
