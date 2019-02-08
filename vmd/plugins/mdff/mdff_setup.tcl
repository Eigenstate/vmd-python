############################################################################
#cr
#cr            (C) Copyright 1995-2009 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: mdff_setup.tcl,v $
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.23 $       $Date: 2017/05/25 15:37:02 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>
#
# mdff gridpdb -- creates a pdb file docking
# mdff setup   -- writes a NAMD config file for docking
#

package require readcharmmpar
package require exectool
package provide mdff_setup 0.4

namespace eval ::MDFF::Setup:: {

  variable defaultDiel 80
  variable defaultScaling1_4 1.0
  variable defaultGScale 0.3
  variable defaultTemp 300
  variable defaultNumSteps 500000
  variable defaultMinimize 200
  variable defaultConsCol B
  variable defaultFixCol O
  variable defaultMargin 0
  variable defaultDir [pwd]
  variable defaultLite 0
  variable defaultGridOff 0

  variable defaultK 10.0
  variable defaultGridpdbSel {noh and (protein or nucleic)}
  #xMDFF related variables
  variable defaultxMDFF    0
  variable defaultBFS      0
  variable defaultBSharp   0
  variable defaultMask     0
  variable defaultAverageMap     0
  variable defaultRefSteps 20000
  variable defaultCrystPDB 0
  variable defaultMaskRes 5
  variable defaultMaskCutoff 5
  variable defaultxMDFFSel "noh and not water"
  #IMD related variables
  variable defaultIMD 0
  variable defaultIMDPort 2000
  variable defaultIMDFreq 1
  variable defaultIMDWait "no"
  variable defaultIMDIgnore "no"
  #REMDFF related variables
  variable defaultREMDFF 0
  variable defaultReplicas 6
  #variable defaultAutoSmooth 0
}

proc ::MDFF::Setup::mdff_setup_usage { } {
    
  variable defaultDiel
  variable defaultScaling1_4
  variable defaultGScale
  variable defaultTemp
  variable defaultParFile
  variable defaultNumSteps
  variable defaultMinimize
  variable defaultConsCol 
  variable defaultFixCol
  variable defaultMargin 
  variable defaultDir
  variable defaultLite
  variable defaultGridOff

  #xMDFF related variables
  variable defaultxMDFF   
  variable defaultBFS      
  variable defaultBSharp
  variable defaultMask     
  variable defaultAverageMap     
  variable defaultRefSteps
  variable defaultCrystPDB 
  variable defaultMaskRes
  variable defaultMaskCutoff
  variable defaultxMDFFSel
  
  #IMD related variables 
  variable defaultIMD
  variable defaultIMDPort
  variable defaultIMDFreq
  variable defaultIMDWait
  variable defaultIMDIgnore

  #REMDFF related variables
  variable defaultREMDFF
  variable defaultReplicas
#  variable defaultAutoSmooth

  ::MDFF::Setup::init_files

  puts "Usage: mdff setup -o <output prefix> -psf <psf file> -pdb <pdb file> -griddx <griddx file(s)> ?options?"
  puts "Options:" 
  puts "  -gridpdb  -- pdb file(s) for docking (default: -pdb, use a list if multiple maps)"
  puts "  -diel     -- dielectric constant (default: $defaultDiel; 1 with -pbc or -gbis)" 
  puts "  -temp     -- temperature in Kelvin (default: $defaultTemp)" 
  puts "  -ftemp    -- final temperature (default: $defaultTemp)" 
  puts "  -gscale   -- scaling factor for the grid (default: $defaultGScale)" 
  puts "  -extrab   -- extrabonds file (default: none)" 
  puts "  -conspdb  -- pdb file with constrained atoms (default: none)"
  puts "  -conscol  -- force constant column in conspdb (default: beta)"
  puts "  -fixpdb   -- pdb file with fixed atoms (default: none)"
  puts "  -fixcol   -- column in fixpdb (default: occupancy)"
  puts "  -scal14   -- 1-4 scaling (default: $defaultScaling1_4)"
  puts "  -step     -- docking protocol step (default: 1)" 
  #puts "  -parfiles -- parameter file list (default $defaultParFile)"
  puts "  -parfiles -- parameter file list"
  puts "  -minsteps -- number of minimization steps (default $defaultMinimize)"
  puts "  -numsteps -- number of time steps to run (default: $defaultNumSteps)" 
  puts "  -margin   -- extra length in patch dimension during simulation (default: $defaultMargin)"
  puts "  -pbc      -- use periodic boundary conditions (for explicit solvent)"
  puts "  -gbis     -- use generalized Born implicit solvent (not compatible with -pbc)"
  puts "  -dir      -- Working Directory (default: $defaultDir)"
  puts "  --lite    -- use gridforcelite, a faster but less accurate calculation of mdff forces"
  puts "  --gridoff -- turn off gridforces, so no mdff forces are applied"
#IMD options
  puts "IMD Options:"
  puts "  --imd     -- turn on Interactive Molecular Dynamics (IMD)"
  puts "  -imdport  -- port for IMD connection"
  puts "  -imdfreq  -- timesteps between sending IMD coordinates"
  puts "  --imdwait -- wait for IMD connection"
  puts "  --imdignore -- ignore steering forces from VMD" 
#REMDFF options
  puts " --remdff   -- turn on resolution exchange mdff (default mode: off)" 
  puts " -replicas  -- number of replicas for resolution exchange mdff (default: $defaultReplicas)" 
  #autosmooth not currently supported here because it is done by the GUI.
  #puts " --autosmooth  -- automatically generate smoothed potentials for resolution exchange mdff" 
#xMDFF options only!
  puts "xMDFF Options:"
  puts "  --xmdff   -- set up xMDFF simulation.  The following options apply to xMDFF only."  
  puts "  -refs     -- reflection data file(s) (mtz or cif). Required for xMDFF (use a list for multiple maps)"  
  puts "  -xmdffsel -- atom selection for xMDFF map generation (default: $defaultxMDFFSel)"  
  puts "  -refsteps -- number of refinement steps between map generation (default: $defaultRefSteps)"  
  puts "  -crystpdb -- text file (can be PDB) with PDB formatted CRYST line to supply symmetry information (default: none, but recommended)"  
  puts "  -averagemap    -- generate density maps which are averaged from multiple previous steps (default: $defaultAverageMap (off). 1 for on. One entry per map defined in -refs )" 
  puts "  -mask    -- clean generated maps by applying a binary mask around structure (default: $defaultMask (off) 1 for on. One entry per map defined in -refs)" 
  puts "  -mask_res -- resolution of mask density in Angstroms (default: $defaultMaskRes) One entry per map defined in -refs" 
  puts "  -mask_cutoff -- cutoff distance of mask density in Angstroms (default: $defaultMaskCutoff) One entry per map defined in -refs" 
  puts "  -bsharp   -- apply beta factor sharpening (or smoothing) to the map(s) (default: $defaultBSharp (off). Negative values for smoothing, positive for sharpening, 'a' for automatic determination (sharpening only)."  
  puts "  --bfs     -- calculate beta factors during every map generation step (default: off)"
}

proc ::MDFF::Setup::mdff_setup { args } {

  variable defaultDiel 
  variable defaultScaling1_4
  variable defaultGScale 
  variable defaultTemp 
  variable defaultParFile
  variable defaultNumSteps 
  variable defaultMinimize
  variable defaultConsCol 
  variable defaultFixCol
  variable namdTemplateFile
  variable xMDFFTemplateFile
  variable REMDFFTemplateFile
  variable REMDFFLoadFile
  variable REMDFFShowFile
  variable REMDFFFile
  variable REMDFFReset
  variable REMDFFREADME
  variable xMDFFScriptFile
  variable defaultMargin 
  variable defaultDir
  variable defaultLite
  variable defaultGridOff

  #xMDFF related variables
  variable defaultxMDFF   
  variable defaultBFS      
  variable defaultBSharp      
  variable defaultMask     
  variable defaultAverageMap    
  variable defaultRefSteps
  variable defaultCrystPDB
  variable defaultMaskRes
  variable defaultMaskCutoff 
  variable defaultxMDFFSel
  
  #IMD related variables 
  variable defaultIMD
  variable defaultIMDPort
  variable defaultIMDFreq
  variable defaultIMDWait
  variable defaultIMDIgnore
  
  #REMDFF related variables
  variable defaultREMDFF
  variable defaultReplicas
#  variable defaultAutoSmooth
  
  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_setup_usage
    error ""
  }

  # Get NAMD template and parameter files
  ::MDFF::Setup::init_files

  # Periodic simulation?
  set pos [lsearch -exact $args {-pbc}]
  if { $pos != -1 } {
    set pbc 1
    set args [lreplace $args $pos $pos]
  } else {
    set pbc 0
  }

  # Using GB implicit solvent?
  set pos [lsearch -exact $args {-gbis}]
  if { $pos != -1 } {
    set gbison 1
    set args [lreplace $args $pos $pos]
  } else {
    set gbison 0
  }

  if {$pbc && $gbison} {
	mdff_setup_usage
	error "Use of -gbis is not compatible with -pbc."
  }
  
  # parse switches
#  foreach {name val} $args {
#    switch -- $name {
#      -o          { set arg(o)        $val }
#      -psf        { set arg(psf)      $val } 
#      -pdb        { set arg(pdb)      $val }
#      -gridpdb    { set arg(gridpdb)  $val }
#      -diel       { set arg(diel)     $val }
#      -scal14     { set arg(scal14)   $val }
#      -temp       { set arg(temp)     $val }
#      -ftemp      { set arg(ftemp)    $val }
#      -griddx     { set arg(griddx)   $val }
#      -gscale     { set arg(gscale)   $val }
#      -extrab     { set arg(extrab)   $val }
#      -conspdb    { set arg(conspdb)  $val }
#      -conscol    { set arg(conscol)  $val }
#      -fixpdb     { set arg(fixpdb)   $val }
#      -fixcol     { set arg(fixcol)   $val }
#      -step       { set arg(step)     $val }
#      -parfiles   { set arg(parfiles) $val }
#      -numsteps   { set arg(numsteps) $val }
#      -minsteps   { set arg(minsteps) $val }
#      -margin     { set arg(margin)   $val }
#      #begind xMDFF related options
#      -xmdff      { set arg(xmdff)    $val }
#      -refsteps   { set arg(refsteps) $val }
#      -mask       { set arg(mask)     $val }
#      -bfs        { set arg(bfs)      $val }
#      -refs       { set arg(refs)     $val }
#      -crystpdb   { set arg(crystpdb) $val }
#    }
#  }
   
  for {set i 0} {$i < [llength $args]} {incr i} {
    switch -- [lindex $args $i] {
      -o          { set arg(o)        [lindex $args [expr $i + 1]] }
      -psf        { set arg(psf)      [lindex $args [expr $i + 1]] } 
      -pdb        { set arg(pdb)      [lindex $args [expr $i + 1]] }
      -gridpdb    { set arg(gridpdb)  [lindex $args [expr $i + 1]] }
      -diel       { set arg(diel)     [lindex $args [expr $i + 1]] }
      -scal14     { set arg(scal14)   [lindex $args [expr $i + 1]] }
      -temp       { set arg(temp)     [lindex $args [expr $i + 1]] }
      -ftemp      { set arg(ftemp)    [lindex $args [expr $i + 1]] }
      -griddx     { set arg(griddx)   [lindex $args [expr $i + 1]] }
      -gscale     { set arg(gscale)   [lindex $args [expr $i + 1]] }
      -extrab     { set arg(extrab)   [lindex $args [expr $i + 1]] }
      -conspdb    { set arg(conspdb)  [lindex $args [expr $i + 1]] }
      -conscol    { set arg(conscol)  [lindex $args [expr $i + 1]] }
      -fixpdb     { set arg(fixpdb)   [lindex $args [expr $i + 1]] }
      -fixcol     { set arg(fixcol)   [lindex $args [expr $i + 1]] }
      -step       { set arg(step)     [lindex $args [expr $i + 1]] }
      -parfiles   { set arg(parfiles) [lindex $args [expr $i + 1]] }
      -numsteps   { set arg(numsteps) [lindex $args [expr $i + 1]] }
      -minsteps   { set arg(minsteps) [lindex $args [expr $i + 1]] }
      -margin     { set arg(margin)   [lindex $args [expr $i + 1]] }
      -dir        { set arg(dir)      [lindex $args [expr $i + 1]] } 
      --lite      { set arg(lite)       1 }
      --gridoff   { set arg(gridoff)    1 }
      #begin REMDFF related options
      --remdff    { set arg(remdff)   1 }
      -replicas   { set arg(replicas) [lindex $args [expr $i + 1]] }
      #--autosmooth { set arg(autosmooth)   1 }
      #begind xMDFF related options
      --xmdff     { set arg(xmdff)    1 }
      -refsteps   { set arg(refsteps) [lindex $args [expr $i + 1]] }
      -mask       { set arg(mask)     [lindex $args [expr $i + 1]] }
      -bsharp     { set arg(bsharp)     [lindex $args [expr $i + 1]] }
      -averagemap { set arg(averagemap) [lindex $args [expr $i + 1]] }
      --bfs       { set arg(bfs)      1 }
      -refs       { set arg(refs)     [lindex $args [expr $i + 1]] }
      -xmdffsel   { set arg(xmdffsel) [lindex $args [expr $i + 1]] }
      -crystpdb   { set arg(crystpdb) [lindex $args [expr $i + 1]] }
      -mask_res   { set arg(maskres) [lindex $args [expr $i + 1]] }
      -mask_cutoff   { set arg(maskcutoff) [lindex $args [expr $i + 1]] }
      #begin IMD related options
      --imd       { set arg(imd)     1 }
      -imdport    { set arg(imdport)  [lindex $args [expr $i + 1]] }
      -imdfreq    { set arg(imdfreq)  [lindex $args [expr $i + 1]] }
      --imdwait   { set arg(imdwait) "yes" }
      --imdignore { set arg(imdignore) "yes" }
    }
  }
    
 
  if { [info exists arg(o)] } {
    set outprefix $arg(o)
  } else {
    mdff_setup_usage
    error "Missing output files prefix."
  }
  
  if { [info exists arg(psf)] } {
    set psf $arg(psf)
  } else {
    mdff_setup_usage
    error "Missing psf file."
  }
  
  if { [info exists arg(pdb)] } {
    set pdb $arg(pdb)
  } else {
    mdff_setup_usage
    error "Missing pdb file."
  }

  if { [info exists arg(diel)] } {
    set diel $arg(diel)
  } elseif {$pbc || $gbison} {
    set diel 1
  } else {
    set diel $defaultDiel
  }

  if { [info exists arg(scal14)] } {
    set scal14 $arg(scal14)
  } else {
    set scal14 $defaultScaling1_4
  }

  if { [info exists arg(temp)] } {
    set itemp $arg(temp)
  } else {
    set itemp $defaultTemp
  }

  if { [info exists arg(ftemp)] } {
    set ftemp $arg(ftemp)
  } else {
    set ftemp $itemp
  }

  if { [info exists arg(parfiles)] } {
    set parfiles $arg(parfiles)
  } else {
    file copy -force $defaultParFile .
    set parfiles [file tail $defaultParFile]
  }

  if { [info exists arg(numsteps)] } {
    set numsteps $arg(numsteps)
  } else {
    set numsteps $defaultNumSteps
  }


  if { [info exists arg(minsteps)] } {
    set minsteps $arg(minsteps)
  } else {
    set minsteps $defaultMinimize
  }

  if { [info exists arg(griddx)] } {
    set grid $arg(griddx) 
  } else {
    mdff_setup_usage 
    error "Missing grid dx file name."
  }
  
  if { [info exists arg(gscale)] } {
    set gscale $arg(gscale)
  } else {
    set gscale $defaultGScale
  }
  
  if { [info exists arg(extrab)] } {
    set extrab $arg(extrab)
  } else {
    set extrab 0
  }
  
  if { [info exists arg(gridpdb)] } {
    set gridpdb $arg(gridpdb)
  } else {
    set gridpdb $pdb
  }

  if { [info exists arg(conspdb)] } {
    set conspdb $arg(conspdb)
  } else {
    set conspdb 0
  }

  if { [info exists arg(conscol)] } {
    set conscol $arg(conscol)
  } else {
    set conscol $defaultConsCol
  }

  if { [info exists arg(fixpdb)] } {
    set fixpdb $arg(fixpdb)
  } else {
    set fixpdb 0
  }

  if { [info exists arg(fixcol)] } {
    set fixcol $arg(fixcol)
  } else {
    set fixcol $defaultFixCol
  }

  if { [info exists arg(minsteps)] } {
    set minsteps $arg(minsteps)
  } else {
    set minsteps $defaultMinimize
  }

  if { [info exists arg(margin)] } {
    set margin $arg(margin)
  } else {
    set margin $defaultMargin
  }
  
  if { [info exists arg(dir)] } {
    set dir $arg(dir)
  } else {
    set dir $defaultDir
  }
  
  if { [info exists arg(lite)] } {
    set lite $arg(lite)
  } else {
    set lite $defaultLite
  }
  
  if { [info exists arg(gridoff)] } {
    set gridoff $arg(gridoff)
  } else {
    set gridoff $defaultGridOff
  }

  if { [info exists arg(step)] } {
    set step $arg(step)
  } else {
    # puts "No step number was specified. Assuming step 1.."
    set step 1
  }
  
  #puts "starting REMDFF section"

  if { [info exists arg(remdff)] } {
    set remdff $arg(remdff)
  } else {
    set remdff $defaultREMDFF
  }
  if {$remdff} {
    if { [info exists arg(replicas)] } {
      set replicas $arg(replicas)
    } else {
      set replicas $defaultReplicas
    }
    #if { [info exists arg(autosmooth)] } {
    #  set autosmooth $arg(autosmooth)
    #} else {
    #  set autosmooth $defaultAutoSmooth
    #}
  }
  
  puts "starting xmdff section"
  if { [info exists arg(xmdff)] } {
    set xmdff $arg(xmdff)
  } else {
    set xmdff $defaultxMDFF
  }
  if { $xmdff } {
    if {$remdff} {
      error "xMDFF not currently compatible with replica exchange."
    }
    if { [info exists arg(refs)] } {
      set refs $arg(refs)
    } else {
      mdff_setup_usage
      error "Missing structure factors file."
    }
    
    if { [info exists arg(xmdffsel)] } {
      set xmdffsel $arg(xmdffsel)
    } else {
      set xmdffsel $defaultxMDFFSel
    }

    if { [info exists arg(mask)] } {
      set mask $arg(mask)
    } else {
      set mask $defaultMask
    }
    
    if { [info exists arg(bsharp)] } {
      set bsharp $arg(bsharp)
    } else {
      set bsharp $defaultBSharp
    }
    
    if { [info exists arg(averagemap)] } {
      set averagemap $arg(averagemap)
    } else {
      set averagemap $defaultAverageMap
    }

    if { [info exists arg(bfs)] } {
      set bfs $arg(bfs)
    } else {
      set bfs $defaultBFS
    }
  
    if { [info exists arg(crystpdb)] } {
      set crystpdb $arg(crystpdb)
    } else {
      set crystpdb $defaultCrystPDB
    }

    if { [info exists arg(refsteps)] } {
      set refsteps $arg(refsteps)
    } else {
      set refsteps $defaultRefSteps
    }
    
    if { [info exists arg(maskres)] } {
      set maskres $arg(maskres)
    } else {
      set maskres $defaultMaskRes
    }
    
    if { [info exists arg(maskcutoff)] } {
      set maskcutoff $arg(maskcutoff)
    } else {
      set maskcutoff $defaultMaskCutoff
    }
  }
  puts "starting IMD section"
  if { [info exists arg(imd)] } {
    set imd $arg(imd)
  } else {
    set imd $defaultIMD
  }
  
  if { $imd } {
    if { [info exists arg(imdport)] } {
      set imdport $arg(imdport)
    } else {
      set imdport $defaultIMDPort
    }

    if { [info exists arg(imdfreq)] } {
      set imdfreq $arg(imdfreq)
    } else {
      set imdfreq $defaultIMDFreq
    }

    if { [info exists arg(imdwait)] } {
      set imdwait $arg(imdwait)
    } else {
      set imdwait $defaultIMDWait
    }
  
    if { [info exists arg(imdignore)] } {
      set imdignore $arg(imdignore)
    } else {
      set imdignore $defaultIMDIgnore
    }
  }

  if { $remdff } {
    file copy -force $REMDFFTemplateFile $dir
    file copy -force $REMDFFFile $dir
    file copy -force $REMDFFLoadFile $dir
    file copy -force $REMDFFShowFile $dir
    file copy -force $REMDFFREADME $dir
    file copy -force $REMDFFReset $dir
    
   
    set out [open [file join $dir "remdff.namd"] w] 
    set outname [file join $dir ${outprefix}-step${step}]
    puts $out "set num_replicas $replicas"
    #set min_temp 300
    #set max_temp 300
    puts $out "set steps_per_run 1000"
    puts $out "set num_runs 10000"
    puts $out "\# num_runs should be divisible by runs_per_frame * frames_per_restart"
    puts $out "set runs_per_frame 1"
    puts $out "set frames_per_restart 10"
    puts $out "set namd_config_file $outname.namd"
    puts $out "\# directories must exist"
    puts $out "set output_root output/\%s/${outprefix}-step${step}"

    puts $out "\# the following used only by show_replicas.vmd"
    puts $out "set psf_file $psf"
    puts $out "set initial_pdb_file $pdb"
#    set fit_pdb_file "test-0003-target.pdb"

    puts $out "\# prevent VMD from reading replica-mdff.namd by trying command only NAMD has"
    puts $out "if { ! \[catch numPes\] } { source replica-mdff.namd }"

    close $out 
    
#For now, mdff gui does this because mdff gui makes potentials, while this does not. Might want to change that.
    #set initialMapDir [file join $dir "initialmaps/"]
    #file mkdir $initialMapDir
    #if {$autosmooth} {
    #  for {set i 0} {$i < $replicas} {incr i} {
    #    file mkdir [file join $dir "output/$i"]
    #    file rename -force [file join $dir [lindex $grid $i]] [file join $initialMapDir "$i.dx"] 
    #    file delete -force [file join $dir [lindex $grid $i]]
    #    file link "[file join $dir [lindex $grid $i]]" [file join $initialMapDir "$i.dx"] 
    #  }
    #}
    
  } elseif {$xmdff} {
    file copy -force $xMDFFTemplateFile $dir
    file copy -force $xMDFFScriptFile $dir
    file delete "maps.params"
    file delete [list [glob -nocomplain "maps*.params"]]

    for {set i 0} {$i < [llength $refs]} {incr i} {
      if [catch {exec phenix.maps} result] {
        puts $result
      } else {
        set frpdb [open "maps.params" "r"]
        set spdb [read $frpdb]
        close $frpdb
        set fwpdb [open "maps.params" "w"]
        set refname [file tail [lindex $refs $i]]

        regsub "pdb_file_name = None" $spdb "pdb_file_name = mapinput.pdb" spdb
        regsub "file_name = None" $spdb "file_name = $refname" spdb
        regsub -all "exclude_free_r_reflections = False" $spdb "exclude_free_r_reflections = True" spdb
        if {[lindex $bsharp $i] != 0} {
          regsub -all "sharpening = False" $spdb "sharpening = True" spdb
          if {[lindex $bsharp $i] != "a"} {
            regsub -all "sharpening_b_factor = None" $spdb "sharpening_b_factor = [lindex $bsharp $i]" spdb
          }
        }
        puts $fwpdb $spdb
        close $fwpdb

        file rename -force "maps.params" [file join $dir "maps$i.params"]
      }
    }

  } else {
    # Copy NAMD template file to working directory
    file copy -force $namdTemplateFile $dir
  }
  set outname [file join $dir ${outprefix}-step${step}]
  puts "mdff) Writing NAMD configuration file ${outname}.namd ..."
  
  set out [open ${outname}.namd w]     

  puts $out "###  Docking -- Step $step" 
  puts $out " "   
  puts $out "set PSFFILE $psf"        
  puts $out "set PDBFILE $pdb"
  if {$gridpdb == $pdb && [llength $grid] > 1} {
    puts $out "set GRIDPDB [list [lrepeat [llength $grid] $gridpdb]]"
  } else {
    puts $out "set GRIDPDB [list $gridpdb]"
  }
  puts $out "set GBISON $gbison"
  puts $out "set DIEL $diel"        
  puts $out "set SCALING_1_4 $scal14"
  puts $out "set ITEMP $itemp"   
  puts $out "set FTEMP $ftemp"  
  if {$xmdff} {
    puts $out "\#Note: all xMDFF generated density file names should only come after any other \n \
              \#density names in the GRIDFILE list, if they exist."
                 
  }
  if {!$remdff} { puts $out "set GRIDFILE [list $grid]" }   
  if {$gscale == $defaultGScale && [llength $grid] > 1} {
    puts $out "set GSCALE [list [lrepeat [llength $grid] $gscale]]"
  } else {
    puts $out "set GSCALE [list $gscale]"   
  }
  puts $out "set EXTRAB [list $extrab]"   
  puts $out "set CONSPDB $conspdb"
  if {$conspdb != "0" } {
    puts $out "set CONSCOL $conscol"
  }
  puts $out "set FIXPDB  $fixpdb"
  if {$fixpdb != "0" } {
    puts $out "set FIXCOL $fixcol"
  }
  if {$gridoff} {
    puts $out "set GRIDON 0"
  } else {
    puts $out "set GRIDON 1"
  }
  
  if {$xmdff} {
    puts $out "set REFINESTEP $refsteps"
    puts $out "set REFS [list $refs]"
    puts $out "set XMDFFSEL \"$xmdffsel\""
    puts $out "set BFS $bfs"
    if {$mask == $defaultMask && [llength $refs] > 1} {
      puts $out "set MASK [list [lrepeat [llength $refs] $mask]]"
    } else {
      puts $out "set MASK [list $mask]"
    }
    if {$maskres == $defaultMaskRes && [llength $refs] > 1} {
      puts $out "set MASKRES [list [lrepeat [llength $refs] $maskres]]"
    } else {
      puts $out "set MASKRES [list $maskres]"
    }
    if {$maskcutoff == $defaultMaskCutoff && [llength $refs] > 1} {
      puts $out "set MASKCUTOFF [list [lrepeat [llength $refs] $maskcutoff]]"
    } else {
      puts $out "set MASKCUTOFF [list $maskcutoff]"
    }
    puts $out "set CRYSTPDB $crystpdb"
    if {$averagemap == $defaultAverageMap && [llength $refs] > 1} {
      puts $out "set AVERAGE [list [lrepeat [llength $refs] $averagemap]]"
    } else {
      puts $out "set AVERAGE [list $averagemap]"
    }
    if {$bsharp == $defaultBSharp && [llength $refs] > 1} {
      puts $out "set BSHARP [list [lrepeat [llength $refs] $bsharp]]"
    } else {
      puts $out "set BSHARP [list $bsharp]"
    }
  }
  puts $out " " 
  
  
  if {$step >  1 } {
    set prevstep [expr $step - 1]
    set inputname "${outprefix}-step${prevstep}"
    set prevnamd "${inputname}.namd"
    if { ![file exists $prevnamd] } {
      puts "Warning: Previous NAMD configuration file $prevnamd not found." 
      puts "You may need to manually edit the variable INPUTNAME in the file ${outname}.namd."
    }
    if {!$remdff} {   puts $out "set INPUTNAME $inputname" } 
  }

  if {!$remdff} { puts $out "set OUTPUTNAME ${outprefix}-step${step}" }
  puts $out " "
  if {!$remdff} { puts $out "set TS $numsteps" }
  if {!$remdff} { puts $out "set MS $minsteps" }
  puts $out " "
  puts $out "set MARGIN $margin"
  puts $out " "
  puts $out "####################################"
  puts $out " "
  puts $out "structure \$PSFFILE"
  puts $out "coordinates \$PDBFILE"
  puts $out " "
  puts $out "paraTypeCharmm on"
  foreach par $parfiles {
    puts $out "parameters $par"
  }
  if $pbc {
    puts $out ""
    puts $out "if {\[info exists INPUTNAME\]} {"
    puts $out "  BinVelocities \$INPUTNAME.restart.vel"
    puts $out "  BinCoordinates \$INPUTNAME.restart.coor"
    puts $out "  ExtendedSystem \$INPUTNAME.restart.xsc"
    puts $out "} else {"
    puts $out "  temperature \$ITEMP"
    ::MDFF::Setup::get_cell $psf $pdb $out
    puts $out "}"

    puts $out "PME yes"
    puts $out "PMEGridSpacing 1.0"
    puts $out "PMEPencils 1"

    puts $out "wrapAll on"

  } else {
    puts $out ""
    puts $out "if {\[info exists INPUTNAME\]} {"
    puts $out "  BinVelocities \$INPUTNAME.restart.vel"
    puts $out "  BinCoordinates \$INPUTNAME.restart.coor"
    puts $out "} else {"
    puts $out "  temperature \$ITEMP"
    puts $out "}"

  }
  puts $out " "
  if {$imd && !$remdff} {
    puts $out "IMDon on"
    puts $out "IMDport $imdport"
    if {$imdfreq > 0} {
      puts $out "IMDfreq $imdfreq"
    }
    puts $out "IMDwait $imdwait"
    puts $out "IMDignore $imdignore"
  }
  puts $out " "
  if {$lite} {
    puts $out "gridforcelite on"
    puts $out " "
  }
  if {$xmdff} {
    puts $out "source [file tail $xMDFFTemplateFile]"
  } elseif {$remdff} {
    puts $out "source [file tail $REMDFFTemplateFile]"
  } else {
    puts $out "source [file tail $namdTemplateFile]"
  }
  close $out

}

proc ::MDFF::Setup::get_cell {psf pdb out} {
  set molid [mol new $psf type psf waitfor all]
  mol addfile $pdb type pdb waitfor all

  set sel [atomselect $molid {noh water}]

  if { [$sel num] == 0 } {
    $sel delete
    mol delete $molid
    error "Could not determine the periodic cell information. No water molecules were found in the input structure."
  }
  set minmax [measure minmax $sel]
  set vec [vecsub [lindex $minmax 1] [lindex $minmax 0]]
  puts $out "  cellBasisVector1 [lindex $vec 0] 0 0"
  puts $out "  cellBasisVector2 0 [lindex $vec 1] 0"
  puts $out "  cellBasisVector3 0 0 [lindex $vec 2]"
  set center [measure center $sel]
  puts $out "  cellOrigin $center"
  $sel delete
  
  mol delete $molid

}


proc ::MDFF::Setup::mdff_gridpdb_usage { } {
 
  variable defaultGridpdbSel

  puts "Usage: mdff gridpdb -psf <input psf> -pdb <input pdb> -o <output pdb> ?options?"
  puts "Options:" 
  puts "  -seltext   -- atom selection text  (default: $defaultGridpdbSel)"
}

proc ::MDFF::Setup::mdff_gridpdb { args } {

  variable defaultGridpdbSel

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_gridpdb_usage
    error ""
  }

  # parse switches
  foreach {name val} $args {
    switch -- $name {
      -psf        { set arg(psf)      $val }
      -pdb        { set arg(pdb)      $val }
      -o          { set arg(o)        $val }
      -seltext    { set arg(seltext)  $val }
    }
  }
    
  if { [info exists arg(o)] } {
    set gridpdb $arg(o)
  } else {
    mdff_gridpdb_usage
    error "Missing output gridpdb file name."
  }
  
  if { [info exists arg(pdb)] } {
    set pdb $arg(pdb)
  } else {
    mdff_gridpdb_usage
    error "Missing pdb file."
  }

  if { [info exists arg(psf)] } {
    set psf $arg(psf)
  } else {
    mdff_gridpdb_usage
    error "Missing psf file."
  }

  if { [info exists arg(seltext)]} {
    set seltext $arg(seltext)
  } else {
    set seltext $defaultGridpdbSel
  }
  
  set molid [mol new $psf type psf waitfor all]
  mol addfile $pdb type pdb waitfor all
  set all [atomselect $molid all]
  $all set occupancy 0

  if { $seltext == "all" } {
    $all set beta [$all get mass]
    $all set occupancy 1
  } else {
    $all set beta 0
    set sel [atomselect $molid $seltext]
    if {[$sel num] == 0} {
      error "empty atomselection"
    } else {
      $sel set occupancy 1
      $sel set beta [$sel get mass]
    }  
    $sel delete
  }

  $all writepdb $gridpdb
  $all delete
  
  return 

}

proc ::MDFF::Setup::init_files {} {
  global env
  variable defaultParFile
  variable namdTemplateFile
  variable xMDFFTemplateFile
  variable xMDFFScriptFile
  variable REMDFFTemplateFile
  variable REMDFFFile
  variable REMDFFREADME
  variable REMDFFLoadFile
  variable REMDFFShowFile
  variable REMDFFReset
  set defaultParFile [file join $env(CHARMMPARDIR) par_all36_prot.prm]
  set namdTemplateFile [file join $env(MDFFDIR) mdff_template.namd]
  set xMDFFTemplateFile [file join $env(MDFFDIR) xmdff_template.namd]
  set xMDFFScriptFile [file join $env(MDFFDIR) xmdff_phenix.tcl]
  set REMDFFTemplateFile [file join $env(MDFFDIR) remdff_template.namd]
  set REMDFFFile [file join $env(MDFFDIR) replica-mdff.namd]
  set REMDFFLoadFile  [file join $env(MDFFDIR) load-mdff-results.tcl]
  set REMDFFShowFile [file join $env(MDFFDIR) show_replicas_mdff.vmd]
  set REMDFFREADME [file join $env(MDFFDIR) README_REMDFF.txt]
  set REMDFFReset [file join $env(MDFFDIR) resetmaps.sh]
}

proc ::MDFF::Setup::mdff_constrain_usage { } {

  variable defaultK
  variable defaultConsCol
 
  puts "Usage: mdff constrain <atomselection> -o <pdb file> ?options?"
  puts "Options:"
  puts "  -col <column> (default: $defaultConsCol)"
  puts "  -k <force constant in kcal/mol/A^2> (default: $defaultK)"
  
}

proc ::MDFF::Setup::mdff_fix_usage { } {

  variable defaultFixCol
 
  puts "Usage: mdff fix <atomselection> -o <pdb file> ?options?"
  puts "Options:"
  puts "  -col <column> (default: $defaultFixCol)"
  
}

proc ::MDFF::Setup::mdff_constrain { args } {

  variable defaultK
  variable defaultConsCol

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_constrain_usage
    error ""
  }

  set sel [lindex $args 0]
  if { [$sel num] == 0 } {
    error "empty atomselection"
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -o     { set arg(o)     $val }
      -col   { set arg(col)   $val }
      -k     { set arg(k)     $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(o)] } {
    set outputFile $arg(o)
  } else {
    mdff_constrain_usage
    error "Missing output pdb file."
  }

  if { [info exists arg(col)] } {
    set col $arg(col)
  } else {
    set col $defaultConsCol
  }

  if { $col == "beta" || $col == "B" } {
    set col "beta"
  } elseif { $col == "occupancy" || $col == "O" } {
    set col "occupancy"
  } elseif { $col == "x" || $col == "X" } {
    set col "x"
  } elseif { $col == "y" || $col == "Y" } {
    set col "y"
  } elseif { $col == "z" || $col == "Z" } {
    set col "z"
  } else {
    error "Unrecognized column."
  }

  if { [info exists arg(k)] } {
    set k $arg(k)
  } else {
    set k $defaultK
  }

  set molid [$sel molid]
  set all [atomselect $molid all]
  set bakCol [$all get $col]
  $all set $col 0
  $sel set $col $k
  $all writepdb $outputFile
  $all set $col $bakCol
  $all delete

  return

}

proc ::MDFF::Setup::mdff_fix { args } {

  variable defaultFixCol

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_fix_usage
    error ""
  }

  set sel [lindex [lindex $args 0] 0]
  if { [$sel num] == 0 } {
    error "mdff_constrain: empty atomselection."
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -o     { set arg(o)     $val }
      -col   { set arg(col)   $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(o)] } {
    set outputFile $arg(o)
  } else {
    mdff_fix_usage
    error "Missing output pdb file."
  }

  if { [info exists arg(col)] } {
    set col $arg(col)
  } else {
    set col $defaultFixCol
  }

  if { $col == "beta" || $col == "B" } {
    set col "beta"
  } elseif { $col == "occupancy" || $col == "O" } {
    set col "occupancy"
  } elseif { $col == "x" || $col == "X" } {
    set col "x"
  } elseif { $col == "y" || $col == "Y" } {
    set col "y"
  } elseif { $col == "z" || $col == "Z" } {
    set col "z"
  } else {
    error "Unrecognized column."
  }

  set molid [$sel molid]
  set all [atomselect $molid all]
  set bakCol [$all get $col]
  $all set $col 0
  $sel set $col 1
  $all writepdb $outputFile
  $all set $col $bakCol
  $all delete

  return

}
