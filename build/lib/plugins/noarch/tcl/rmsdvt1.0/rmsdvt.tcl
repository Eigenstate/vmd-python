##
## RMSD Visualizer 1.0
##
## $Id: rmsdvt.tcl,v 1.4 2013/04/15 17:30:43 johns Exp $
##
## VMD plugin (Tcl extension) for calculating and visualizing RMSD and RMSF calculations
##
## Authors: Anssi Nurminen, Sampo Kukkurainen, Laurie S. Kaguni, Vesa P. Hytönen
## Institute of Biomedical Technology
## University of Tampere
## Tampere, Finland
## and
## BioMediTech, Tampere, Finland
## 30-01-2012 
##
## email: anssi.nurminen_a_uta.fi
##
##

##
## Example of loading the plugin in the VMD console: 
## vmd_install_extension rmsdvt rmsdvt_tk "Analysis/RMSD Visualizer Tool"


## Provide Tcl package
package provide rmsdvt 1.0
 

namespace eval ::Rmsdvt:: {
  
    #namespace export rmsdvt 
      
    #GUI Window  
    variable iW
    
    variable iDataDir
    
    variable iSelMolId
    variable iSelMolFrames
    
    # GUI Trajectory box variables, iTraj
    # keys are: ref_frame, 
    #           frames_checkbox, frames_from, frames_to
    #           step_checkbox, step_size    
    #           window_checkbox, window_size
    variable iTrajOptions
    
    
    # Array containing plugin general settings
    # keys are: heatmap, backbone, 2dplot
    variable iSettings
    
    # GUI checkbox variable for Atom Selection modifiers
    # keys: backbone, trace, noh
    variable iAtomSelectionModifiers
    
    
    # Variables for storing all results: iRes*
    # iRes* arrays have unique run numbers as their keys
    # Arrays
    variable iResListboxArray 
    
    # Results data. format "key,run_id"
    # keys are: id, name, selection_str, numofframes, frames_from, frames_to, window_size, 
    #           ref_frame, ref_mol, totalframes, step_size, heatmap_type, hm_handle, hm_x_label, hm_y_label
    variable iRes
    
    # Results RMSD calculation data. format "run_id,resid,fr"
    # Note: resid is not the same as residue. 
    #       resid is VMD numbering, residue is molecules own numbering.
    variable iResRMSDAvg  
    variable iResRMSDResidues    

    # array of saved atom selections (key:description,  value:selectionstr)       
    variable iSavedAtomSelections
    
    
    # Integer
    variable iResRunningIndex    
    
    variable iExitting
    
    variable iCancelled
    
    
    proc initialize {} {
    
        #puts "INITIALIZE: rmsdvt_tk"
        
        variable iDataDir ""
        
        variable iSelMolId -1
        variable iSelMolFrames -2
        
        variable iAtomSelectionModifiers
        
        # GUI Trajectory box variables, iTraj        
        variable iTrajOptions
        
        variable iSettings
        
        # Variables for storing results, iRes
        variable iResListboxArray 
        
        variable iRes
              
        variable iResRMSDAvg  
        variable iResRMSDResidues                
        
        variable iResRunningIndex 0    
        
        variable iExitting 0
        variable iCancelled 0
        
        
        variable iSavedAtomSelections
        #::Rmsdvt::ClearResults
        
        array set iSavedAtomSelections {}
        
        # Set Trajectory options initial values
        array set iTrajOptions {}
        # radio-options "frame" and "window"
        set iTrajOptions(ref_selection) 0
        set iTrajOptions(ref_mol) [list "self"]
        set iTrajOptions(ref_frame) 0
        set iTrajOptions(ref_window) 0
        set iTrajOptions(frames_checkbox) 0
        set iTrajOptions(frames_from) 0
        set iTrajOptions(frames_to) 0
        set iTrajOptions(step_checkbox) 0
        set iTrajOptions(step_size) 1
        #set iTrajOptions(skip_checkbox) 0
        #set iTrajOptions(skip_from) 0
        #set iTrajOptions(skip_to) 0
        #set iTrajOptions(window_checkbox) 0
        
        
        # Set general settings
        array set iSettings {}
        set iSettings(heatmap) "resid"
        set iSettings(backbone) "C CA N O"
        set iSettings(2dplot) "multiplot"
        
        # Atom selection modifiers
        array set iAtomSelectionModifiers {}
        set iAtomSelectionModifiers(backbone) 0
        set iAtomSelectionModifiers(trace) 0
        set iAtomSelectionModifiers(noh) 0
        
        # Results data structures                            
        array set iRes {}
        
        array set iResRMSDAvg {}
        array set iResRMSDResidues {}
        
        # GUI listboxes
        array set iResListboxArray {}
        
        
    }
    initialize
}


#
# Plugin registration procedure
#
proc rmsdvt_tk {} {
    #puts "REGISTER: rmsdvt_tk"
    ::Rmsdvt::rmsdvt
    return $::Rmsdvt::iW
}



# Load other files
if { [info exists ::env(RMSDVTDIR)] } {
    
    #puts "RMSDVT folder: $env(RMSDVTDIR)"
    set plugin_folder $::env(RMSDVTDIR)
    
    # Remove curly brackets if they exist
    if { [string equal [string range $plugin_folder 0 0] "\{"] } {
        set plugin_folder [string range $plugin_folder 1 end-1]
    }
    
    
    if { [file exists [join [list $plugin_folder "/" rmsdvt-gui.tcl] ""]] } {    
        # Load source file
        source [file join $plugin_folder rmsdvt-gui.tcl]  
        source [file join $plugin_folder modal.tcl]
    } else {
        # Source file missing
        puts "ERROR: Source file [join [list $plugin_folder "/" rmsdvt-gui.tcl] ""]] not found"
    }       
} else {
    # Environment variable not set. Check pkgIndex.tcl
    puts "ERROR: Environment variable env(RMSDVTDIR) not set. Please restart VMD." 
}



# Stores current settings for result calculation into iRes* variables
proc ::Rmsdvt::CreateNewResult { } {
 
    variable iW
    variable iSelMolId
    variable iTrajOptions
    variable iResRunningIndex
    
    variable iRes
    
    # listboxes
    variable iResListboxArray
       
    
   
    set new_run $iResRunningIndex
    incr iResRunningIndex
    
    set iRes(run,$new_run) $new_run
    set iRes(name,$new_run) [molinfo $iSelMolId get name]
    set iRes(id,$new_run) [molinfo $iSelMolId get id]
    set iRes(selection_str,$new_run) [[namespace current]::AtomselectionStr]
    set iRes(ref_frame,$new_run) $iTrajOptions(ref_frame)
    set iRes(heatmap_type,$new_run) ""
    
    # Reference molecule
    set iRes(ref_mol,$new_run) $iTrajOptions(ref_mol)    
    # Change possible special value "self" into mol Id
    if { [string equal $iTrajOptions(ref_mol) "self"] } {        
        set iRes(ref_mol,$new_run) $iRes(id,$new_run)       
    }
    
    
    # Step variable
    if { $iTrajOptions(step_checkbox) } { 
        set iRes(step_size,$new_run) $iTrajOptions(step_size)
        if { $iRes(step_size,$new_run) < 1 } { set iRes(step_size,$new_run) 1 }
    } else {
        set iRes(step_size,$new_run) 1
    }
    
    
     
    # RMSF variables
    if { $iTrajOptions(ref_selection) == 1} { 
        # Window selected
        set iRes(window_size,$new_run) $iTrajOptions(ref_window)
    } else {
        # Frame selected
        set iRes(window_size,$new_run) -1
    }
    
    set rmsf_tag ""
    if { $iRes(window_size,$new_run) >= 0 } {
        set rmsf_tag " RMSF:$iRes(window_size,$new_run) "
    }
        
    # Frames Variables
    if { $iTrajOptions(frames_checkbox) } {
        set iRes(frames_from,$new_run) $iTrajOptions(frames_from) 
        set iRes(frames_to,$new_run) $iTrajOptions(frames_to)    
    } else {
        set iRes(frames_from,$new_run) 0
        set iRes(frames_to,$new_run) [expr [molinfo $iSelMolId get numframes] - 1]
    }

    
    
    set iRes(numofframes,$new_run) [expr (($iRes(frames_to,$new_run) - $iRes(frames_from,$new_run)) / $iRes(step_size,$new_run)) + 1]
    set iRes(totalframes,$new_run) [molinfo $iSelMolId get numframes]
    
    
    # DEBUG
    #puts "New run: $new_run name: [molinfo $iSelMolId get name]"
    
    # Add to listboxes
    $iResListboxArray(run) insert end $new_run
    $iResListboxArray(mol) insert end "$iRes(id,$new_run) : [molinfo $iSelMolId get name]"
    $iResListboxArray(res) insert end ""
    
    [namespace current]::SetListBoxResult $new_run
    
    # return created run number
    return $new_run
}


proc ::Rmsdvt::CalculateRmsdAvgThroughTrajectory { } {
  
    variable iW
    variable iSelMolId
    
    variable iTrajOptions
   
    variable iRes
    variable iResRunningIndex 
    variable iResRMSDAvg
    variable iResListboxArray
    
    [namespace current]::UpdateStatusText "Calculating RMSD..."
    [namespace current]::RefreshTrajectoryOptions
    
    #TODO checks
    #has frames?
    #ref frame in range?
    if { $iSelMolId < 0 } {
        puts "RMSDVT: No molecule selected"
        [namespace current]::UpdateStatusText "No molecule selected."
        return -1
    }
    
    set ref_mol $iSelMolId
    # get molid if reference selection is "self"
    if { [string is integer -strict $iTrajOptions(ref_mol)] } { set ref_mol $iTrajOptions(ref_mol) }
    

    
    # Do checks
    if { [lsearch [molinfo list] $iSelMolId] < 0 || [molinfo $iSelMolId get numframes] == 0 } {        
        puts "ERROR: Selected Molecule has no frames."
        [namespace current]::UpdateStatusText "Selected Molecule has no frames."
        return -1
    } elseif { [lsearch [molinfo list] $ref_mol] < 0 || [molinfo $ref_mol get numframes] == 0 } {
        puts "ERROR: Selected reference molecule has no frames."
        [namespace current]::UpdateStatusText "Selected reference molecule has no frames."
        return -1        
    }
    
    # Get selected atoms string
    set selection_str [[namespace current]::AtomselectionStr]    
    
    # Check atom selection validity
    if { [string equal $selection_str ""] } {
        #Selection string empty
        [namespace current]::showMessage "No selection made. Use atom selection \"protein\" to select every atom in molecule."
        return -1
    }     
    

    #puts "RMSDVT: Calculating RMSD for [molinfo $iSelMolId get numframes] frames..."
    
    # Get initial frames for a selection check
    set ref_frame $iTrajOptions(ref_frame)
    set frames_from 0
    #set frames_from [expr [molinfo $iSelMolId get numframes] - 1]
    

    if { $iTrajOptions(frames_checkbox) } {        
        set frames_from $iTrajOptions(frames_from)
    } 

    # Set reference to measurements
    # Catch atom selection syntax errors
    set sel_err ""
    if { [catch {set ref_sel [atomselect $ref_mol $selection_str frame $ref_frame]} sel_err] } {
        
        if { [string length $sel_err] > 60 } {
            set sel_err [string range $sel_err 0 60]
            set sel_err "$sel_err..."
        }
        [namespace current]::UpdateStatusText "ERROR: $sel_err"   
	    return -1
    }    
    
    #set ref_sel [atomselect $ref_mol $selection_str frame $ref_frame]
    set cur_sel [atomselect $iSelMolId $selection_str frame $frames_from]
       
    # Check if selection is sensible
    if { [$ref_sel num] < 1 || [$cur_sel num] < 1 } {
        puts "ERROR: No atoms in selection. (Mol:$iSelMolId -f$iTrajOptions(frames_from) Ref:$ref_mol -f$iTrajOptions(ref_frame) Str:$selection_str"
        [namespace current]::UpdateStatusText "Unable to create plot: No atoms in selection. Ref: [$ref_sel num], Sel: [$cur_sel num]"
        $cur_sel delete
        $ref_sel delete
        return
    } elseif { [$ref_sel num] != [$cur_sel num] } {
        puts "ERROR: Selection size differs in reference molecule ([$ref_sel num]/[$cur_sel num]). (Mol:$iSelMolId -f$iTrajOptions(frames_from) Ref:$ref_mol -f$iTrajOptions(ref_frame) Str:$selection_str"
        [namespace current]::UpdateStatusText "Error: Selection sizes differ.  Ref: [$ref_sel num]  Sel: [$cur_sel num]"
        $cur_sel delete
        $ref_sel delete
        return
            
    }
    
    #puts "cursel: id:$iSelMolId str:\"$selection_str\" frame: $frames_from num: [$cur_sel num]"
    #puts "refsel: id:$ref_mol str:\"$selection_str\" frame: $ref_frame num: [$ref_sel num]"
    
    #
    # Create new result
    # Note: From now on we should only use things stored in iRes array (not iTrajOptions)
    #
    set cur_run [[namespace current]::CreateNewResult]    
    

    # Running counter for calculated number of frames
    set frames 0 
    
    set reference_frames_count [$ref_sel num]
    set last_reference_frame_exceeded -1    
    
    
    #Do calculation and fill iResRMSDAvg with results
    for {set fr $iRes(frames_from,$cur_run)} {$fr <= $iRes(frames_to,$cur_run)} {incr fr $iRes(step_size,$cur_run)} {	
    
        if { $iRes(window_size,$cur_run) >= 0 } {
            # RMSF/Windowed MODE, move reference frame
            # Note: If frames are limited from entire range,  reference frame
            #       can still be taken from outside specified limited range
            set rmsf_fr [expr $fr - $iRes(window_size,$cur_run)]
            if { $rmsf_fr >= $reference_frames_count } { 
                set rmsf_fr [expr $reference_frames_count - 1]
                if { $last_reference_frame_exceeded == -1 } { set last_reference_frame_exceeded $fr }
            }        
            if { $rmsf_fr < 0 } { set rmsf_fr 0 }                            
            
            $ref_sel frame $rmsf_fr
            $ref_sel update                 
        }        
            
        $cur_sel frame $fr
        $cur_sel update
        
        # Using selections that have for example clauses like "within" in them can cause selection size to change
        # during trajectory
        if { [$ref_sel num] != [$cur_sel num] } {
            set errortxt "Selection size changes during trajectory. Frame: $fr Atoms: Ref: [$ref_sel num], Sel: [$cur_sel num]"
            [namespace current]::UpdateStatusText $errortxt
            puts "ERROR: Selection size mismatch: $errortxt"
            puts "Use only atom selections that do not change through the trajectory."
            $cur_sel delete
            $ref_sel delete
            # Select and Remove created results
            $iResListboxArray(res) selection set [expr [$iResListboxArray(run) index end] - 1]
            [namespace current]::ListSelection $iResListboxArray(res)
            [namespace current]::RemoveSelectedResults
            return
        }        
        
        # MEASURE RMSD
        set iResRMSDAvg($cur_run,$fr) [measure rmsd $ref_sel $cur_sel] 
        
        #DEBUG
        #puts "iResRMSDAvg($cur_run,$fr) = $iResRMSDAvg($cur_run,$fr)"  
                             
        incr frames
        
    }              
    
    $cur_sel delete
    $ref_sel delete
    
    set iRes(numofframes,$cur_run) $frames
    #puts "RMSD calculation complete."    
    [namespace current]::UpdateStatusText "RMSD calculation complete."
    
    if { $last_reference_frame_exceeded > -1 } {
        puts "WARNING: Last reference frame was exceeced by RMSF Window. \
              Last reference frame used as reference for frames $last_reference_frame_exceeded- . \
              This Warning is only shown once."
    }      
    
        
    # if only one result exists, select it
    if { [$iResListboxArray(run) index end] == 1 } {
        $iResListboxArray(res) selection set 0
        [namespace current]::ListSelection $iResListboxArray(res)
    }
    
}


proc ::Rmsdvt::CalculateHeatmapThroughTrajectory { } {
 
    variable iRes
    variable iResListboxArray
    variable iSettings
    variable iResRMSDResidues
    
    [namespace current]::UpdateStatusText "Setting up heatmap calculation..."
    
    
    # File for saving heatmap plot result and opening with the Heatmapper plugin
    # Requires setting RMSDVTDIR environment variable which is set in the pkgIndex.tcl file
    set heatmapfile [join [list [[namespace current]::DataDir] "/" "temp.hm"] ""]    
    
    # Do Check for errors
    if { [llength [$iResListboxArray(run) curselection]] < 1 } {
        puts "ERROR: No results selected"
        [namespace current]::UpdateStatusText "No results selected."
        return -1
    } 
    
    
    # Get a list of selected indices
    set selections [$iResListboxArray(run) curselection]
    
    # Selected indices into run numbers
    set runs [list]
    foreach run $selections {
        lappend runs [$iResListboxArray(run) get $run]
    }    

    if { [llength runs] < 1 } {
        puts "ERROR: nothing selected"
        [namespace current]::UpdateStatusText "Nothing selected."
        return -1
    }      
    
    
    #TODO comparison heatmaps
    if { [llength runs] > 1 } {
        puts "Info: No comparison heatmap support yet"
    }    
    

    #DEBUG TODO loop
    #set runs [lrange runs 0 0]
    #puts "Runs: $runs"
    
    foreach run $runs {
    
        
        #puts "RMSDV: Calculating RMSD heatmap for run $run"    
        
        
        # Check if results have already been calculated
        if { [llength [array get iResRMSDResidues $run,*]] } {
            # Results already exist, see type...
            if { [string equal $iSettings(heatmap) $iRes(heatmap_type,$run)] } {
                puts "RMSDVT: Using previous results."
                [namespace current]::UpdateStatusText "Using previous calculation's results."
                [namespace current]::SaveHeatmapResultsToFile $run $heatmapfile
                
                if { [[namespace current]::PlotHeatmap $run $heatmapfile] >= 0 } {
                    # No errors opening heatmap
                    [namespace current]::UpdateStatusText "Showing heatmap."    
                }
                # Done!
                continue
            } else {
                #delete previous results and carry on
                array unset iResRMSDResidues $run,*
                set iRes(heatmap_type,$run) ""
                [namespace current]::SetListBoxResult $run
            }
        }
        
        # Check that needed molecule still exists
        if { [lsearch [molinfo list] $iRes(id,$run)] < 0 } {
            puts "ERROR: Molecule $iRes(id,$run) not found. Has molecule been deleted since result was calculated?"        
            [namespace current]::UpdateStatusText "Molecule $iRes(id,$run) not found. Please redo RMSD."
            continue
        } elseif { [molinfo $iRes(id,$run) get numframes] != $iRes(totalframes,$run) } {
            puts "ERROR: Frame count mismatch. Has molecule trajectory changed since result was calculated?"
            [namespace current]::UpdateStatusText "Frame count mismatch. Please redo RMSD."
            continue
        } elseif { [lsearch [molinfo list] $iRes(ref_mol,$run)] < 0 || [molinfo $iRes(ref_mol,$run) get numframes] == 0 } {
            puts "ERROR: Selected reference molecule ($iRes(ref_mol,$run)) has no frames."
            [namespace current]::UpdateStatusText "Selected reference molecule has no frames."
            continue        
        } elseif { [molinfo $iRes(ref_mol,$run) get numframes] <= $iRes(frames_to,$run) } {
            puts "ERROR: Selected reference molecule ($iRes(ref_mol,$run)) has only [molinfo $iRes(ref_mol,$run) get numframes] frames."
            [namespace current]::UpdateStatusText "Selected reference molecule has only [molinfo $iRes(ref_mol,$run) get numframes] frames."
            continue        
        }
        
    
        [namespace current]::DoCalculateHeatmapThroughTrajectory $run $heatmapfile
        
    }
       
}

proc ::Rmsdvt::PrintSelectedUnits { {aResult 0} } {
    
    variable iSelMolId
    variable iTrajOptions
    variable iSettings
    variable iRes
    
    #puts "Atom: $iSelMolId, Sel: \"[[namespace current]::AtomselectionStr]\" frame: $iTrajOptions(frames_from)"
    
    set selStr ""
    set selId -1
    set selFrame -1
    
    if { $aResult == 0 } {
        set selStr [[namespace current]::AtomselectionStr] 
        set selId $iSelMolId
        
        if { $iTrajOptions(frames_checkbox) } {
            set selFrame $iTrajOptions(frames_from)    
        } else {
            set selFrame 0
        }
        
    } elseif { [info exists iRes(id,[[namespace current]::SelectedRun])] && 
               [info exists iRes(selection_str,[[namespace current]::SelectedRun])] &&
               [info exists iRes(frames_from,[[namespace current]::SelectedRun])] } {
                   
       set selStr $iRes(selection_str,[[namespace current]::SelectedRun])
       set selId $iRes(id,[[namespace current]::SelectedRun])
       set selFrame $iRes(frames_from,[[namespace current]::SelectedRun])
    } else {
        [namespace current]::UpdateStatusText "ERROR: Cannot find result's data"
        return -1
    }
    
    
    set selAtoms ""
    
    set sel_err ""
    if { [catch {set selAtoms [atomselect $selId $selStr frame $selFrame]} sel_err] } {
        if { [string length $sel_err] > 60 } {
            set sel_err [string range $sel_err 0 60]
            set sel_err "$sel_err..."
        }        
        [namespace current]::UpdateStatusText "ERROR: $sel_err"   
	    return -1
    }     

    #puts "NUM: [$selAtoms num]"
    
    set listingStr "resid"
    
    switch $iSettings(heatmap) {
        resid {      set listingStr "resid"  } 
        residue {    set listingStr "residue" } 
        backbone {   set listingStr "resid"}
        backbone2 {  set listingStr "residue"  }
        sidechain {  set listingStr "resid" }
        sidechain2 { set listingStr "residue" }
        atom {       set listingStr "index"  }   
    }     
        
    
    array set uniques {}
    foreach unit [$selAtoms get $listingStr] {
                
        if { ![info exists uniques($unit)] } {           
            set uniques($unit) 1
        }
    }    
    
    #puts "uniques size: [llength [array names uniques]]"
    
    set sorted_uniques [lsort -integer [array names uniques]]
    array unset uniques
    
        
    #puts "sorted_uniques size: [llength $sorted_uniques]"
    
    foreach uni $sorted_uniques { 
        puts -nonewline "$uni "
    }
    
    puts "\n\nSelection Type: $listingStr, Size: [llength $sorted_uniques]"
    puts "Selection mol:$selId, frame:$selFrame, atomsel: \"$selStr\"\n"
    
    array unset sorted_uniques
    $selAtoms delete
}


# Calculate data for a heatmap presentation
# Calculate for selected result(s)
proc ::Rmsdvt::DoCalculateHeatmapThroughTrajectory { aRun aFilename } {
  
    variable iW
    variable iTrajOptions
    
    variable iSelMolId
    variable iRes
    variable iResRMSDResidues

    variable iResListboxArray
    
    variable iSettings
    
    variable iCancelled
    
    #puts "DoCalculateHeatmapThroughTrajectory"

    set selAtoms [atomselect $iRes(id,$aRun) $iRes(selection_str,$aRun) frame $iRes(frames_from,$aRun)]
    
    if { [$selAtoms num] == 0 } {
        puts "ERROR: No atoms in selection. (Mol:$iRes(id,$aRun)Str:$iRes(selection_str,$aRun) F:$iRes(ref_frame,$aRun))"
        [namespace current]::UpdateStatusText "Unable to create plot: No atoms in selection."
        return
    }
        
    # Disable UI-elements for the duration of the calculation
    $iW.menubar.result configure -state disabled
    $iW.menubar.options configure -state disabled
    $iW.top.pushfr.rmsd configure -state disabled
    $iW.top.pushfr.align configure -state disabled
    $iW.menubar.file configure -state disabled
    [namespace current]::SetBottomButtonsStatus 1    
    update
    
    
    # Possible choices are: resid, residue, backbone, sidechain, atom
    #                       also backbone2 and sidechain2 which use residues instead of resids
    set iRes(heatmap_type,$aRun) $iSettings(heatmap)
    [namespace current]::CreateHeatMapOptionsString $aRun
    
    
    #Construct selection string
    set selStr ""
    set listingStr "residue"
    set calc_units "resids"
    set flip_order 0;
    
    switch $iRes(heatmap_type,$aRun) {
        resid {      set selStr "residue %s";                 set calc_units "resids"; set flip_order 1; } 
        residue {    set selStr "residue %s";               set calc_units "residues"; } 
        backbone {   set selStr "residue %s and name $iSettings(backbone)";    set calc_units "resids"; set flip_order 1; }
        backbone2 {  set selStr "residue %s and name $iSettings(backbone)";    set calc_units "residues";  }
        sidechain {  set selStr "residue %s and sidechain";   set calc_units "resids"; set flip_order 1; }
        sidechain2 { set selStr "residue %s and sidechain"; set calc_units "residues"; }
        atom {       set selStr "index %s";                 set calc_units "indices"; set listingStr "index";  }   
    }     
    
    # Get all atoms that are in the selection grouped by either resid or residue
    # (or by index in case of atoms, which means no grouping)
    # Note: RESIDUEs are VMD-given numbering and RESIDs are molecule's own numberings (from molecule data file)
    array set uniques {}
    foreach atom [$selAtoms get $listingStr] {
                
        if { ![info exists uniques($atom)] } {
            # Get and save resid number for each selection component
            set sel_temp [atomselect $iRes(id,$aRun) "residue $atom" frame $iRes(frames_from,$aRun)]
            set uniques($atom) [lindex [$sel_temp get resid] 0]
            $sel_temp delete
        }
    }
    
    # Sort unique residues into a list
    # TODO is sorting really required here?
    set goThruList [lsort -integer [array names uniques]]

    set progress_update 20
    
    set frame_count [expr ($iRes(frames_to,$aRun) - $iRes(frames_from,$aRun)) / $iRes(step_size,$aRun)]
    if { $frame_count > 120 } { set progress_update 10 }
    if { $frame_count > 260 } { set progress_update 5 }
    if { $frame_count > 380 } { set progress_update 2 }
    if { $frame_count > 499 } { set progress_update 1 }
    
    set progress $progress_update
    set calc_done 0
    set calc_total [llength $goThruList]
    set frame_count 0
    
    
    set last_reference_frame_exceeded -1 
    #puts "LOOP: $iRes(frames_from,$aRun) -> $iRes(frames_to,$aRun)"
    
    [namespace current]::Cancellable 1
    
    set getclicks 1
    set start_calc [clock clicks -milliseconds]
    set one_calc 0
    
    # Calculate and store results into iResRMSDResidues array           
    foreach residue $goThruList {    
             
        # Set reference selection for residue
        set ref_sel [atomselect $iRes(ref_mol,$aRun) [format $selStr $residue] frame $iRes(ref_frame,$aRun)] 
        set cur_sel [atomselect $iRes(id,$aRun) [format $selStr $residue] frame $iRes(frames_from,$aRun)]
                                  
        #puts "ref_sel: $iRes(ref_mol,$aRun) [format $selStr $residue] frame $iRes(ref_frame,$aRun) has: [$ref_sel num]"
        #puts "cur_sel: $iRes(id,$aRun) [format $selStr $residue] frame $iRes(frames_from,$aRun) has: [$cur_sel num]"
        
        set frame_count 0        
        set cur_run $aRun
        set reference_frames_count [$ref_sel num]
        
        
        for {set fr $iRes(frames_from,$cur_run)} {$fr <= $iRes(frames_to,$cur_run)} {incr fr $iRes(step_size,$cur_run)} {
                  
            if { $iRes(window_size,$cur_run) >= 0 } {
                # RMSF/Windowed MODE, move reference frame
                # Note: If frames are limited from entire range,  reference frame
                #       can still be taken from outside specified limited range
                set rmsf_fr [expr $fr - $iRes(window_size,$cur_run)]
                if { $rmsf_fr >= $reference_frames_count } { 
                    set rmsf_fr [expr $reference_frames_count - 1]
                    if { $last_reference_frame_exceeded == -1 } { set last_reference_frame_exceeded $fr }
                }                 
                if { $rmsf_fr < 0 } { set rmsf_fr 0 }     
                $ref_sel frame $rmsf_fr
                $ref_sel update                 
            }     
            
            $cur_sel frame $fr
            $cur_sel update 
            set rmsd [measure rmsd $ref_sel $cur_sel]            
            
            # Save into results array
            if { !$flip_order } {
                # Do not flip'em
                set iResRMSDResidues($cur_run,$residue:$uniques($residue),$fr) $rmsd
                
            } else {
                # flip'em
                # resid before residue numbering
                set iResRMSDResidues($cur_run,$uniques($residue):$residue,$fr) $rmsd
            }
            
            #puts "Resid $residue, frame: $fr rmsd: $rmsd"
            
            incr frame_count
            
            # Make Ui more responsive during calculation
            if { [expr $fr % 50] == 0 } { update }
        }
        
        $ref_sel delete
        $cur_sel delete    
        
        incr progress
        incr calc_done
        
        
        if { $iCancelled == 1 } {
            # RE-enable UI buttons
            [namespace current]::SetBottomButtonsStatus
            $iW.menubar.result configure -state normal
            $iW.menubar.options configure -state normal
            $iW.top.pushfr.rmsd configure -state normal
            $iW.top.pushfr.align configure -state normal
            $iW.menubar.file configure -state normal    
            
            [namespace current]::UpdateStatusText "Heatmap calculation cancelled." 
            
            array unset iResRMSDResidues $cur_run,*
            [namespace current]::Cancellable 0
            set iCancelled 0
            
            return;
        }
        
        if { $getclicks == 1 } {
            set getclicks 0            
            set one_calc [expr [clock clicks -milliseconds] - $start_calc]
                
        }
        
        # Show progress update
        if { $progress >= $progress_update } {   
            
            set remaining ""
            if { $one_calc != 0 } {                
                set mins [expr (($one_calc * ($calc_total - $calc_done)) / 1000) / 60]
                #set mins 0
                #while { $secs >= 60 } { incr mins; set secs [expr $secs - 60]; }
                if { $mins > 1 } { 
                    set remaining "   (~$mins mins)" 
                } elseif { $mins == 1 } {
                    set remaining "   (under 2 mins)" 
                } else {
                    set remaining "   (under 1 min)" 
                }           
            }
            set done [format "% 4d" $calc_done]                     
            [namespace current]::UpdateStatusText "Calculating RMSD for all $calc_units... $done/$calc_total $calc_units$remaining"
            set progress 0
        }        
        
    }
    
    
    [namespace current]::Cancellable 0
    set iCancelled 0
    
    array unset uniques
    
    if { $last_reference_frame_exceeded > -1 } {
        puts "WARNING: Last reference frame was exceeced by RMSF Window. \
              Last reference frame used as reference for frames $last_reference_frame_exceeded- . \
              This Warning is only shown once."
    }    
    
    puts "RMSDVT: RMSD calculation complete. [string toupper $calc_units 0 0]: $calc_done Frames: $frame_count "

    set iRes(hm_x_label,$aRun) "Frames"
    set iRes(hm_y_label,$aRun) [string totitle $calc_units]
    
    # Set what numbering is used for calculated atoms/residues
    if { !$flip_order } {
        # Do not flip'em
        if { [string equal $iRes(heatmap_type,$aRun) "atom"] } {
            set iRes(numbering,$aRun) "index:resid"
        } else {
            set iRes(numbering,$aRun) "residue:resid"
        }    
    } else {
        # Flip'em
        set iRes(numbering,$aRun) "resid:residue"
    }
    
    # Result must be selected and now has heatmap data
    $iW.menubar.result.menu entryconfigure 4 -state normal
    
    # Mark result as having existing calculated heatmap data
    
    # Get listbox index of run
    [namespace current]::SetListBoxResult $aRun

    #Save into a File        
    [namespace current]::UpdateStatusText "Saving results to file..."
    
    #DEBUG
    #puts "Saving to file: $aFilename"
    [namespace current]::SaveHeatmapResultsToFile $aRun $aFilename
    
    [namespace current]::UpdateStatusText "Results saved. Creating heatmap..."
    
    #[namespace current]::UpdateStatusText "File written."

    set retval [catch {[[namespace current]::PlotHeatmap $aRun $aFilename]}]
        
    if { $retval >= 0 } {
        # No errors opening heatmap
        [namespace current]::UpdateStatusText "Showing heatmap."    
    } else {
        [namespace current]::UpdateStatusText "Error opening heatmap."    
    }
    
    # RE-enable UI buttons
    [namespace current]::SetBottomButtonsStatus
    $iW.menubar.result configure -state normal
    $iW.menubar.options configure -state normal
    $iW.top.pushfr.rmsd configure -state normal
    $iW.top.pushfr.align configure -state normal
    $iW.menubar.file configure -state normal
    #puts -nonewline $fileId $data

    
}

proc ::Rmsdvt::SetListBoxResult { aRun } {

    variable iResListboxArray
    variable iRes
    
    set result_str ""
    
    set rmsf_tag ""
    if { $iRes(window_size,$aRun) >= 0 } { set rmsf_tag " RMSF:$iRes(window_size,$aRun)" }
    
    set step_tag ""
    if { $iRes(step_size,$aRun) > 1 } { set step_tag " Step:$iRes(step_size,$aRun)" }    
    
    set heatmap_tag ""
    if { ![string equal $iRes(heatmap_type,$aRun) ""] } { 
        switch $iRes(heatmap_type,$aRun) {
            resid { set heatmap_tag "\[HM - resid\]" } 
            residue { set heatmap_tag "\[HM - residue\]" } 
            backbone { set heatmap_tag "\[HM - resid & bb\]" }
            backbone2 { set heatmap_tag "\[HM - residue & backbone\]" }
            sidechain { set heatmap_tag "\[HM - resid & sc\]" }
            sidechain2 { set heatmap_tag "\[HM - residue & sc\]" }
            atom { set heatmap_tag "\[HM - atoms\]" }   
        }               
    }
    
    set selectionstr_tag ""
    if { [string length $iRes(selection_str,$aRun)] > 17 } { set selectionstr_tag "..." }    
    
    set result_str [concat $result_str [string range $iRes(selection_str,$aRun) 0 16] $selectionstr_tag \
                           [format "(%s frames)" $iRes(numofframes,$aRun)] \
                           $rmsf_tag $step_tag $heatmap_tag]
    
    # Get listbox index of run
    set list_index [lsearch -integer [$iResListboxArray(run) get 0 end] $aRun]
    if { $list_index < 0 } { puts "ERROR: cannot find run $aRun from listbox."; return }
    
    set list_selection [$iResListboxArray(res) curselection]
    $iResListboxArray(res) insert $list_index $result_str
    $iResListboxArray(res) delete [expr $list_index + 1]
    # Revert selection status to the state it was
    foreach item $list_selection {
      $iResListboxArray(res) selection set $item
    }       
}

proc ::Rmsdvt::AddToListBoxResult { aRun aAddition } {

    variable iResListboxArray
    
    # Get listbox index of run
    set list_index [lsearch -integer [$iResListboxArray(run) get 0 end] $aRun]
    if { $list_index < 0 } { puts "ERROR: cannot find run $aRun from listbox." }
    
    set list_selection [$iResListboxArray(res) curselection]
    $iResListboxArray(res) insert $list_index [concat [$iResListboxArray(res) get $list_index] $aAddition]
    $iResListboxArray(res) delete [expr $list_index + 1]
    # Revert selection status to the state it was
    foreach item $list_selection {
      $iResListboxArray(res) selection set $item
    }
    
}

proc ::Rmsdvt::CreateHeatMapOptionsString { aRun } {

    variable iRes
    
    set ylabel ""    
    switch $iRes(heatmap_type,$aRun) {
        resid { set ylabel "Resid" } 
        residue { set ylabel "Residue" } 
        backbone { set ylabel "\"Resid\nBackbone\"" }
        backbone2 { set ylabel "\"Residue\nBackbone\"" }
        sidechain { set ylabel "\"Resid\nSidechain\"" }
        sidechain2 { set ylabel "\"Residue\nSidechain\"" }
        atom { set ylabel "Index" }   
    }            
    
    set rmsf_tag ""
    if { $iRes(window_size,$aRun) >= 0 } {
        set rmsf_tag " RMSF:$iRes(window_size,$aRun)"
    }    
       
        
    array set arr  [list "-title" "\"$iRes(name,$aRun): $iRes(selection_str,$aRun)$rmsf_tag\"" \
                    "-xlabel" "Frames" \
                    "-ylabel" $ylabel \
                    "-xorigin" "$iRes(frames_from,$aRun)" ]
     
    set opt_str ""
                         
    foreach key [array names arr] {
         set opt_str [join [concat $opt_str $key $arr($key)] " "]
    }
                    
                    
    #puts "Opt_str: $opt_str"
    set iRes(hm_option_str,$aRun) $opt_str
}

proc ::Rmsdvt::SortNumbering { {aArg1 "0:0"} {aArg2 "0:0"} } {
    
    set a1 [split $aArg1 ":"]
    set a2 [split $aArg2 ":"]
    
    for {set i 0} { $i < [llength $a1] } {incr i} {     
        if { [lindex $a1 $i] < [lindex $a2 $i] } {
            return -1    
        } elseif { [lindex $a1 $i] > [lindex $a2 $i] } {
            return 1
        } 
    }
    
    return 0
}




proc ::Rmsdvt::SaveHeatmapResultsToFile { {aRun -1} {aFilename ""} } {
    
    variable iRes
    variable iResRMSDResidues
    
    variable iTrajOptions
    variable iW
    
    if { $aRun < 0 } { set aRun [[namespace current]::SelectedRun] }
    if { $aRun < 0 } { return }
    
    if { [string equal $aFilename ""] } {
        # Open save file dialog     
        set types {
            {{HeatMapper files} {.hm}}
            {{Text files} {.txt}}
            {{All files}        *   }
        }        
        #strip out some illegal chars and add suffix .hm
        set name_suggestion "$iRes(name,$aRun)_[string range $iRes(selection_str,$aRun) 0 6]"
        set name_suggestion [string map [list " " "_" ">" "-" "<" "-" "." "_" "\{" "(" "\}" ")"] $name_suggestion]
        set name_suggestion [join [concat $name_suggestion ".hm"] ""]
        
        
        set newfile [tk_getSaveFile \
                    -title "Choose file name" -parent $iW \
                    -initialdir [pwd] -filetypes $types -initialfile $name_suggestion]    
                 
        if { ![llength $newfile] } { return }
        set aFilename $newfile
    }
    
    #puts "Saving results to file..."
        
    # Create a list of residues.    
    set rmsd_entries [array names iResRMSDResidues $aRun,*,$iRes(frames_from,$aRun)]
    set residList [list]
    
    # residList has one of the following formats
    # 1. residue:resid
    # 2. resid:residue
    # 3. index:resid
    foreach residue $rmsd_entries {                        
        lappend residList [lindex [split $residue ","] 1]
    }        
        
    
    # Make sure it's sorted from smallest to largest
    #set residList [lsort -integer $residList]
    set residList [lsort -command ::Rmsdvt::SortNumbering $residList]
    
    #puts "Residlist: $residList"
    #puts "aRun: $aRun"
    #puts "aFilename: $aFilename"
    #puts "RMSDVT: Number of resids for run $aRun: [llength $residList]"  
    
    # TODO file name and error handling
    set fileId [open $aFilename "w"]

    set rmsf_tag ""
    if { $iRes(window_size,$aRun) >= 0 } {
        set rmsf_tag " RMSF:$iRes(window_size,$aRun)"
    }
    
    #puts $iRes(hm_option_str,$aRun)
    puts $fileId "-title \"$iRes(name,$aRun): $iRes(selection_str,$aRun)$rmsf_tag\""
    puts $fileId "-xlabel $iRes(hm_x_label,$aRun)"
    puts $fileId "-ylabel $iRes(hm_y_label,$aRun)"
    puts $fileId "-xorigin $iRes(frames_from,$aRun)"
    puts $fileId "-xstep $iRes(step_size,$aRun)"
    puts $fileId "-numbering $iRes(numbering,$aRun)"
    ##puts $fileId "-yorigin firstresidue"
    
    
    # Note: iResRMSDResidues($run,$resid,$fr)
    
    # For each resid (not residue)
    foreach resid $residList {
        
        set datarow [list $resid:]
                   
        # For each frame     
        for {set fr $iRes(frames_from,$aRun)} { $fr <= $iRes(frames_to,$aRun) } {incr fr $iRes(step_size,$aRun)} {
            lappend datarow [join [concat $iResRMSDResidues($aRun,$resid,$fr) ";"] ""]
        }
        
        # One residue, one line in file
        # DEBUG
        puts $fileId [join $datarow ""]
        #puts $datarow
    }    
    
    #file written
    close $fileId
    puts "RMSDVT: File $aFilename written."    
    
    
}


proc ::Rmsdvt::IndexToRgb { aIndex } {

  lassign [colorinfo rgb $aIndex] r g b
  set r [expr {int($r*255)}]
  set g [expr {int($g*255)}]
  set b [expr {int($b*255)}]

  return [format "#%.2X%.2X%.2X" $r $g $b]
}



proc ::Rmsdvt::PlotUsingMultiplot { } {
 
    variable iW
          
    variable iResRMSDAvg
    variable iRes
    
    variable iResListboxArray
    

    # Get a list of selected indices
    set selections [$iResListboxArray(run) curselection]
    
    # Selected indices into run numbers
    set runs [list]
    foreach run $selections {
        lappend runs [$iResListboxArray(run) get $run]
    }
    
    set multiple ""
    if { [llength $runs] > 1 } { set multiple "s" }
     
    puts "RMSDVT: Plotting run$multiple: $runs"
    
    # Check for errors
    if { [llength $runs] < 1 } {
        puts "ERROR: No results selected"
        return -1;    
    } elseif { [llength [array names iRes run,*]] < 1 } {        
        puts "ERROR: results empty"
        return -1;
    } elseif { [catch {package require multiplot} msg] } {
        [namespace current]::showMessage "Package multiplot not installed."
        puts "ERROR: Plotting in Multiplot not available."
        return -1
    }
    
    # Check that all runs exist in iResRuns array
    foreach temp $runs {
        if { [string equal [array get iRes run,$temp] ""] } {
            puts "ERROR: Result $temp not found!"
        }
    }
    
    # Set up multiplot graph info
    set rms_sel [[namespace current]::AtomselectionStr]
    set title "RMSD vs Frame:"
    set xlab "Frame"
    set ylab "RMSD (Å)"
    
    # Title    
    array set uniqueMoleculeNames {}
    foreach run $runs {
        set uniqueMoleculeNames($iRes(name,$run)) 0
    }
    
    # Sort unique molecule names into a list
    set molNamesList [lsort -dictionary [array names uniqueMoleculeNames]]
    array unset uniqueMoleculeNames    
    
    set title [concat $title [join $molNamesList ", "]]
    
    
    #set up plot handle
    #set plothandle [multiplot -title $title -xlabel $xlab -ylabel $ylab -nostats]
    
    #Use dummy X and Y lists to suppress multiplot warning message about empty data plots
    set plothandle [multiplot -x [list 0] -y [list 0] -title $title -xlabel $xlab -ylabel $ylab -nostats]
    
        
    set plotsDrawn 0
    
    #For each result run
    foreach r $runs {
        
        #does selected run exist?
        if { ![info exists iRes(run,$r)] } {
            puts "RMSDVT: Skipping run $r"
            continue
        }
        
        set cur_run $r
        
        #Use colors 0-15        
        set color_num [expr {$plotsDrawn % 16}]        
        set color [IndexToRgb $color_num]
        
        #puts $iRes(name,0)
        set iname "\"$iRes(name,$cur_run):$iRes(selection_str,$cur_run)\""
        if { $iRes(window_size,$cur_run) >= 0 } {
            set iname [join [list $iname "RMSF:$iRes(window_size,$cur_run)"] " "]
        }
        
        #puts "iname: $iname"
        
        #puts "Adding plots for: $iRes(numofframes,$cur_run) frames"
        set x_values [list]
        set y_values [list]
        
        for {set i $iRes(frames_from,$cur_run)} { $i <= $iRes(frames_to,$cur_run) } {incr i $iRes(step_size,$cur_run)} {
        
            lappend x_values $i
            lappend y_values $iResRMSDAvg($cur_run,$i)
            
            #if { $iRes(numofframes,$cur_run) == 1} {
            #    $plothandle add $i $iResRMSDAvg($cur_run,$i) -marker circle -radius 4 -nolines -fillcolor $color -linecolor $color -nostats -legend $iname
            #} else {
            #    $plothandle add $i $iResRMSDAvg($cur_run,$i) -marker point -radius 2 -fillcolor $color -linecolor $color -nostats -legend $iname
            #}
        }
        
        $plothandle add $x_values $y_values -marker point -radius 2 -fillcolor $color -linecolor $color -nostats -legend $iname
        
        #puts "Added plots for: $iRes(numofframes,$cur_run) frames"
        
        incr plotsDrawn
    }
        
    $plothandle configure -autoscale
    $plothandle replot
    
    #array unset {this stuff}
    #set {this stuff(one)} 1
    #parray {this stuff}                   
}

proc ::Rmsdvt::PlotHeatmap { aRun aFilename } {

    variable iRes
    
    if { ![llength $aFilename] } {     
        puts "ERROR: Bad heatmap filename."
        return -1
    } elseif { [catch {package require heatmapper} msg] } {
        [namespace current]::showMessage "Package heatmapper not installed."
        puts "ERROR: Plotting with Heatmapper plugin not available. Install Heatmapper plugin."
        return -1
    }
    
    #catch { destroy .heatmapper }
    
    set iRes(hm_handle,$aRun) [heatmapper -loadfile "\"$aFilename\""]
       
}

proc ::Rmsdvt::SortedListOfResultsFrames { aRun } {
 
    # Check for errors
    if { [string equal iRes(run,$aRun)] } {
        puts "ERROR: SortedListOfResultsFrames: no results found for Run: $aRun"
        return [list]
    }
    
    
    
    
       
}



# Aligns molecule's every trajectory frame with it's reference frame atom positions
# Alignment done according to "SelectedAtoms" text selection
# Alignment is also called superimposition
proc ::Rmsdvt::AlignMoleculeThroughTrajectory {} {
  
    variable iW
    variable iSelMolId
    variable iTrajOptions
    
    [namespace current]::RefreshTrajectoryOptions
    
    # Check for selection errors
    if { $iSelMolId < 0 } {
        [namespace current]::UpdateStatusText "No molecule selected."  
        return -1
    } elseif { [molinfo $iSelMolId get numframes] < 1 } { 
        [namespace current]::UpdateStatusText "No trajectory available."
        return -1
    }
           
    
    # Set frames for looping
    if { !$iTrajOptions(frames_checkbox) } {
        # If "Frames from" checkbox is unselected loop from first to last frame
        set iTrajOptions(frames_from) 0
        set iTrajOptions(frames_to) [expr [molinfo $iSelMolId get numframes] - 1]
    }    
    
    set selection_str [[namespace current]::AtomselectionStr]
    
    set ref_mol $iSelMolId
    # if reference molecule ID selection is something else than "self"
    if { [string is integer -strict $iTrajOptions(ref_mol)] } { set ref_mol $iTrajOptions(ref_mol) }
    
    # Catch atom selection syntax errors
    set sel_err ""
    if { [catch {set ref_sel [atomselect $ref_mol $selection_str frame $iTrajOptions(ref_frame)]} sel_err] } {
        if { [string length $sel_err] > 60 } {
            set sel_err [string range $sel_err 0 60]
            set sel_err "$sel_err..."
        }        
        [namespace current]::UpdateStatusText "ERROR: $sel_err"   
	    return -1
    }      
    
    #set ref_sel [atomselect $ref_mol $selection_str frame $iTrajOptions(ref_frame)]
    set cur_sel [atomselect $iSelMolId $selection_str frame $iTrajOptions(frames_from)]
    # $sel1 is the selection for the selected atoms only.
    # When moving, you really want to move the whole molecule
    set move_sel [atomselect $iSelMolId "all" frame $iTrajOptions(frames_from)]
    
    
    # Check: Selections need to have the same number of atoms for align to work
    if { [$ref_sel num] != [$cur_sel num] } {        
        [namespace current]::UpdateStatusText "Unable to align, atom numbers need to match. Ref: [$ref_sel num] Selected mol: [$cur_sel num]"
        $cur_sel delete
        $ref_sel delete
        return
    }
    
    
    # Set step
    # Note: Does it really make sense to use step size with alignment?
    #       Even if trajectory is really long, it should not be a performance issue
    set step_size 1
    set step_tag ""
    if { $iTrajOptions(step_checkbox) } { 
        set step_size $iTrajOptions(step_size) 
        set step_tag " (Step: $step_size)"
    }
      
    
    # Set up to show alignment progress with a crude progressbar
    set progress_loops [expr ($iTrajOptions(frames_to) - $iTrajOptions(frames_from)) / $step_size]
    set progress_count [expr $progress_loops / 20]
    set progress_status 0
    set progress_counter 0
    set progress_show 0
    set progress_tick 20
    
    if { $progress_count > 40 } { 
        set progress_tick [expr $progress_loops / 40]
        set progress_count 40 
    }
    if { $progress_count > 4 } {     
        set progress_show 1
        set progress_bar [list ]
        for {set bb 0} {$bb < $progress_count} {incr bb} {
            lappend progress_bar "/"
        }
        set status_text [join [concat "Aligning molecule through trajectory  " [join $progress_bar ""] ""]]
        [namespace current]::UpdateStatusText $status_text
    } else {
        # For short loops progression is not shown at all
        set status_text "Aligning molecule through trajectory..."
        [namespace current]::UpdateStatusText $status_text
    }
    
    # Warnings about possibly unwated behavior with alignment
    set show_warning_dg 0
    set warning_dg_msg ""
    
    # Show warning if in windowed reference mode
    if { $iTrajOptions(ref_selection) == 1 } {
        puts "WARNING: Doing alignment in windowed (RMSF) mode! Windowed mode updates alignment reference through the trajectory."
        set warning_dg_msg "Doing alignment in windowed mode! (RMSF)\nWindowed mode updates alignment reference through the trajectory.\n"
        set show_warning_dg 1
    }
    # Show warning if step size not 1
    if { $step_size != 1 } {
        puts "WARNING: Doing alignment with step size: $step_size! Not all frames in trajectory will be aligned."
        set warning_dg_msg "$warning_dg_msg\nDoing alignment with step size: $step_size!\nNot all frames in trajectory will be aligned."
        set show_warning_dg 1
    }    
    
    if { $show_warning_dg == 1 } {
        [namespace current]::UpdateStatusText "Showing warning dialog... click OK to proceed with alignment"
        set answer [tk_messageBox -title "Warning" -message "Proceed with alignment?\n\n$warning_dg_msg" -type okcancel -icon question -parent $iW]
        switch -- $answer {
            ok { [namespace current]::UpdateStatusText "Proceeding with alignment."; }
            cancel { [namespace current]::UpdateStatusText "Alignment cancelled."; return; }
        }          
    }
    
    
    set reference_frames_count [$ref_sel num]
    set last_reference_frame_exceeded -1
    
    # Loop through frames    
    for {set fr $iTrajOptions(frames_from)} {$fr <= $iTrajOptions(frames_to)} {incr fr $step_size} {	
        
        if { $iTrajOptions(ref_selection) == 1 } {
            # RMSF/Windowed MODE, move reference frame
            # Note: If frames are limited from entire range,  reference frame
            #       can still be taken from outside specified limited range
            set rmsf_fr [expr $fr - $iTrajOptions(ref_window)]
            if { $rmsf_fr >= $reference_frames_count } { 
                set rmsf_fr [expr $reference_frames_count - 1]
                if { $last_reference_frame_exceeded == -1 } { set last_reference_frame_exceeded $fr }
            }                  
            if { $rmsf_fr < 0 } { set rmsf_fr 0 } 
            $ref_sel frame $rmsf_fr
            $ref_sel update                 
        }             
        
        $cur_sel frame $fr
        $cur_sel update        
                
        # Using selections that have for example clauses like "within" in them can cause selection size to change
        # during trajectory
        if { [$ref_sel num] != [$cur_sel num] } {
            set errortxt "Selection size changes during trajectory. Frame: $fr Atoms: Ref: [$ref_sel num], Sel: [$cur_sel num]"
            [namespace current]::UpdateStatusText $errortxt
            puts "ERROR: Selection size mismatch: $errortxt."
            puts "Alignment stopped at frame [expr $fr - 1]. Use only atom selections that do not change through the trajectory."
            $cur_sel delete
            $ref_sel delete
            $move_sel delete
            return
        }            
        
        
        # Use transformation matrix and move command
        # Note: reference selection is the 2nd argument for "measure fit"
        set transformation_matrix [measure fit $cur_sel $ref_sel]
        # Move the whole molecule
        $move_sel frame $fr
        $move_sel update                 
        # Align!
        $move_sel move $transformation_matrix
        
        
        # Progress bar handling
        if { $progress_show == 1 } {
            incr progress_counter
            if { $progress_counter == $progress_tick && $progress_status < $progress_count } {
                set progress_counter 0
                
                lset progress_bar $progress_status "|"      
                incr progress_status
                #lreplace           
                set status_text [join [concat "Aligning molecule through trajectory  " [join $progress_bar ""] ""]]
                [namespace current]::UpdateStatusText $status_text
                #[namespace current]::UpdateStatusText $progress_bar
            }
        }
        
    }        
    
    $cur_sel delete
    $ref_sel delete
    $move_sel delete
    
    if { $last_reference_frame_exceeded > -1 } {
        puts "WARNING: Last reference frame was exceeded by RMSF Window. Last reference frame used as reference for frames $last_reference_frame_exceeded-"
    }
    
    [namespace current]::UpdateStatusText "Mol $iSelMolId aligned based on atom selection through frames: $iTrajOptions(frames_from)-$iTrajOptions(frames_to)$step_tag."
}

proc ::Rmsdvt::RemoveSelectedResults { } {
    
    variable iW
    variable iRes    
    variable iResRMSDAvg  
    variable iResRMSDResidues
    
    variable iResListboxArray
    
    #check for errors
    if { ![array exists iResListboxArray] } {
        puts "ERROR: No results array"
        return -1
    } elseif { [llength [$iResListboxArray(run) curselection]] < 1 } {
        puts "ERROR: No results selected"
        return -1
    }
    
    set cur_sels [$iResListboxArray(run) curselection]
    set cur_runs [list]
    
    #get run numbers from indices
    foreach sel [array names cur_sels] {
        lappend cur_runs [$iResListboxArray(run) get $sel $sel]
    }
    
    
    #remove from listboxes
    foreach key [array names iResListboxArray] {        
        $iResListboxArray($key) selection clear $cur_sels $cur_sels
        $iResListboxArray($key) delete $cur_sels $cur_sels           
    }
    
    #remove data
    foreach run $cur_runs {
        
        # Arrays
        array unset iRes *,$run        
        
        # Multidimesional arrays
        array unset iResRMSDAvg $run,*
        array unset iResRMSDResidues $run,*              
          
       
    }
       

    $iW.menubar.result configure -state disabled
    [namespace current]::SetBottomButtonsStatus
    
    
    
}



proc ::Rmsdvt::ClearResults { } {

    variable iW
    
    variable iResRunningIndex
    variable iRes
    variable iResRMSDAvg  
    variable iResRMSDResidues    
    
    variable iResListboxArray
    
    
    # Arrays
    array unset iRes *    
    
    # Multidimensional arrays
    array unset iResRMSDAvg *
    array unset iResRMSDResidues *    
    
    #puts "HERE!"
    
    # ListBox arrays, clear listbox contents
    if { [array exists iResListboxArray] } {
                
        foreach key [array names iResListboxArray] {
            $iResListboxArray($key) selection clear 0 end
            $iResListboxArray($key) delete 0 end               
        }
        
        if { [info exists $iW.menubar.result] } {
            
        }        
        
        #puts "SET!!"    
        $iW.menubar.result configure -state disabled            
        [namespace current]::SetBottomButtonsStatus        
        
        
        
    }
    

    
    #Reset result numbering
    set iResRunningIndex 0
    
    #array unset iResListboxArray *
    
    
    # TODO
    #$iW.menubar.result.menu -state disabled
}
