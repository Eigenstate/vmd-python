package require mdff
set argv $env(VMDARG)
#psf file
set psf [lindex $argv 0]   
#OUTPUTNAME,INPUTNAME or PDB
set coord [lindex $argv 1]
#GRIDFILE, WARNING WILL OVERWRITE FILE
set maps [lindex $argv 2]
#diffraction data
set refs [lindex $argv 3]
#Do we want to use individual adp refinement?
set bfs [lindex $argv 4]
#Do we want to mask the map?
set mask [lindex $argv 5]
#pdb which never changes that contains a valid CRYST1 line for symmetry information
set crystpdb [lindex $argv 6]
#resolution of mask density
set mask_res [lindex $argv 7]
#cutoff for mask density
set mask_cutoff [lindex $argv 8]
#atom selection for map creation
set map_sel [lindex $argv 9]
#do we want average maps?
set avg [lindex $argv 10]

proc calc_betafactors {crystpdb refs} {
  puts "calculating beta factors..."
  #Calcs bfactors against first refs only!
  #may not like symmetry format in cif/mtz.  may have to add a --symmetry="pdbwithCRYST1line.pdb" that never gets changed
  if {$crystpdb != 0} {
    exec phenix.refine "betainitial.pdb" "[lindex $refs 0]" refinement.refine.strategy=individual_adp --symmetry=$crystpdb --overwrite
  } else {
    exec phenix.refine "betainitial.pdb" "[lindex $refs 0]" refinement.refine.strategy=individual_adp --overwrite
  }
  file copy -force "betainitial_refine_001.pdb" "mapinput.pdb"
}

proc remove_cryst {} {
  set frpdb [open "mapinput.pdb" "r"]
  set spdb [read $frpdb]
  close $frpdb
  set fwpdb [open "mapinput.pdb" "w"]
  regsub -all -line {^.*CRYST.*$} $spdb "" spdb
  puts $fwpdb $spdb
  close $fwpdb
}

proc generate_maps {sel maps refs mask mask_res mask_cutoff avg} {
  for {set i 0} {$i < [llength $refs]} {incr i} {
    if {![lindex $avg $i]} {
      set map [lindex $maps [expr [llength $maps]-[llength $refs]+$i]]
      puts "computing density map..."
      exec phenix.maps "maps$i.params"

#in case of crashes due to R-frees
#  exec phenix.remove_free_from_map mapinput_map_coeffs.mtz 3p5Af_f.mtz
#  exec phenix.mtz2map mtz_file=map_coeffs_without_freer_set.mtz pdb_file=mapinput.pdb
#  mdff griddx -i map_coeffs_without_freer_set_2mFo-DFc.ccp4 -o "$map"

      file delete -force "xmdff_density$i.dx"
      mdff griddx -i mapinput_2mFo-DFc_map.ccp4 -o "$map"
      mdff griddx -i "$map" -o "xmdff_density$i.dx"

#masking begin
      if {[lindex $mask $i]} {
        puts "masking map..."
        volmap mask $sel -res [lindex $mask_res $i] -cutoff [lindex $mask_cutoff $i] -o "mask.dx" 
        voltool mult -i1 "xmdff_density$i.dx" -i2 "mask.dx" -o "xmdff_density$i.dx"
        mdff griddx -i "xmdff_density$i.dx" -o "$map"
        #mdff griddx -i "$map" -o "xmdff_density.dx"
      }
#masking end
    }
  }
  avg_maps $sel $maps $refs $mask $mask_res $mask_cutoff $avg 
}

proc avg_maps {sel maps refs mask mask_res mask_cutoff avg} {
  set numframes [molinfo [$sel molid] get numframes]

  for {set i 0} {$i < [llength $refs]} {incr i} {
    if {[lindex $avg $i] } {
      if {$numframes >=5} {
        set map [lindex $maps [expr [llength $maps]-[llength $refs]+$i]]
        puts "computing average map..."
        
        file delete -force "xmdff_density$i.dx"
        file delete -force "average_maps$i.dat"
        for {set j 0} {$j < 5} {incr j} {
          $sel frame [expr $numframes - 1 - $j]
          write_phenixpdb $sel "mapinput.pdb"
          exec phenix.maps "maps$i.params"
          file copy -force "mapinput_map_coeffs.mtz"  "mapinput_map_coeffs$j.mtz"
          set avgmapsdat [open "average_maps$i.dat" "a+"]
          puts $avgmapsdat "mapinput_map_coeffs$j.mtz"
          close $avgmapsdat
        }
        $sel frame last
        write_phenixpdb $sel "mapinput.pdb"
        set labin "FP=2FOFCWT PHIB=PH2FOFCWT"
        exec phenix.average_map_coeffs file_list=average_maps$i.dat labin_mtz=$labin
        exec phenix.mtz2map mtz_file=mapfile.mtz pdb_file=mapinput.pdb

        mdff griddx -i mapfile_1.ccp4 -o "$map"
        mdff griddx -i "$map" -o "xmdff_density$i.dx"

      } else {
       set map [lindex $maps [expr [llength $maps]-[llength $refs]+$i]]
       puts "Not enough frames for average map of $map. Computing normal map instead"
       exec phenix.maps "maps$i.params"
#in case of crashes due to R-frees
#  exec phenix.remove_free_from_map mapinput_map_coeffs.mtz 3p5Af_f.mtz
#  exec phenix.mtz2map mtz_file=map_coeffs_without_freer_set.mtz pdb_file=mapinput.pdb
#  mdff griddx -i map_coeffs_without_freer_set_2mFo-DFc.ccp4 -o "$map"

      file delete -force "xmdff_density$i.dx"
      mdff griddx -i mapinput_2mFo-DFc_map.ccp4 -o "$map"
      mdff griddx -i "$map" -o "xmdff_density$i.dx"
      }
#masking begin
      if {[lindex $mask $i]} {
        puts "masking map..."
        volmap mask $sel -res [lindex $mask_res $i] -cutoff [lindex $mask_cutoff $i] -o "mask.dx" 
        voltool mult -i1 "xmdff_density$i.dx" -i2 "mask.dx" -o "xmdff_density$i.dx"
        mdff griddx -i "xmdff_density$i.dx" -o "$map"
        #mdff griddx -i "$map" -o "xmdff_density.dx"
      }
#masking end
    }
  }
}

proc write_phenixpdb {sel filename} {
  $sel set occupancy 1
  $sel writepdb $filename

  set frpdb [open $filename "r"]
  set spdb [read $frpdb]
  close $frpdb
  set fwpdb [open $filename "w"]
  regsub -all "HSD" $spdb "HIS" spdb
  regsub -all "HSE" $spdb "HIS" spdb
  regsub -all "URA" $spdb "  U" spdb
  regsub -all "ADE" $spdb "  A" spdb
  regsub -all "CYT" $spdb "  C" spdb
  regsub -all "GUA" $spdb "  G" spdb
  regsub -all "THY" $spdb "  T" spdb
  regsub -all "CYN" $spdb "CYS" spdb
  regsub -all -line {^.*CRYST.*$} $spdb " " spdb
  puts $fwpdb $spdb
  close $fwpdb
}


file delete -force "mapinput.pdb"
file delete -force "[list $maps]"
file delete -force "mask.dx"
file delete -force "mapinput_2mFo-DFc_map.ccp4"
file delete -force "betainitial.pdb"
file delete -force "betainitial_refine_001.pdb"

mol new $psf waitfor all

if {[lsearch $avg 1] != -1 && [file exists "$coord.dcd"]} {
  mol addfile "$coord.dcd" waitfor all
} elseif {[file exists "$coord.restart.coor"]} {
  mol addfile "$coord.restart.coor" waitfor all
} else {
  mol addfile "$coord" waitfor all
}

set sel [atomselect top "$map_sel"]

#$sel frame last
#write_phenixpdb $sel

if ($bfs) {
  write_phenixpdb $sel "betainitial.pdb"
  calc_betafactors $crystpdb $refs
} else {
  write_phenixpdb $sel "mapinput.pdb"
} 
remove_cryst
generate_maps $sel $maps $refs $mask $mask_res $mask_cutoff $avg 

exit
