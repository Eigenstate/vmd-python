## NAME: truncate_trajectory_1
##
## $Id: trunctraj.tcl,v 1.2 2013/04/15 17:51:37 johns Exp $
##
## SYNOPSIS:
##   truncate_trajectory_1 creates smaller .psf and .dcd files from bigger ones 
##   by keeping only a specified number of waters. The final trajectory may optionaly 
##   contain non-water hetero compounds too. The user can specify their kind and number 
##   otherwise the program will guess which non-water hetero compounds must be included.
##
## AUTHOR:
##   Thomas Evangelidis (te8624@mbg.duth.gr)
##
## VERSION:
##   trial version (last update 24/12/2008)
##
##  TODO:
##    Do something with the topology file, make it work for other types of macromoleculs like lipids and carbonhydrates.
##
##
##
##
##
##  INSTALLATION:
##  cd ${TT_GUI_HOME}
##  pkg_mkIndex ./
##  vmd_install_extension truncate_trajectory tt_GUI_tk "Analysis/Truncate Trajectory"
##
##

package provide truncate_trajectory 1.5

namespace eval TruncateTrajectory:: {
    namespace export truncate_trajectory
    variable pi 3.14159265358979323846264338327950288419716639937510
    variable keepAtoms_res 
    variable keepAtoms_seg 
    variable keepAtoms_type 
    variable frm
    variable HeteroCompounds
    variable w
    variable frameNum
    # default values:
    variable trajectory_psf "No file selected"
    variable trajectory_dcd "No file selected"
    variable macromolecule "protein" 
    variable selectedAtoms "all"
    variable N_waters 1000
    variable answer "true_false"
    variable answer1 "true" 
    variable answer2 "false"
    variable catdcdPath "/home/thomas/Documents/Molecular_Dynamics/catdcd"
    variable workDir "."
    variable toppar "./toppar/top_all22_prot.inp"
}





#
# Setter methods
#

proc TruncateTrajectory::Setter { my_trajectory_psf my_trajectory_dcd my_macromolecule my_selectedAtoms my_N_waters my_answer1 my_answer2 my_workDir } {

variable trajectory_psf $my_trajectory_psf trajectory_dcd $my_trajectory_dcd macromolecule $my_macromolecule selectedAtoms $my_selectedAtoms
variable N_waters $my_N_waters answer1 $my_answer1 answer2 $my_answer2 workDir $my_workDir

}

proc TruncateTrajectory::AddHeteroCompound { seg number types } {
    lappend compound $seg $number [split $types \ ]
    variable HeteroCompounds
    lappend HeteroCompounds $compound
}

proc TruncateTrajectory::catdcdPathSetter { my_catdcdPath } {
    variable catdcdPath $my_catdcdPath
}

#proc TruncateTrajectory::Setter { my_trajectory_psf my_trajectory_dcd } {

#  variable trajectory_psf
#  variable trajectory_dcd
#  set trajectory_psf $my_trajectory_psf 
#  set trajectory_dcd $my_trajectory_dcd 

#}

proc TruncateTrajectory::DCDSetter { my_trajectory_dcd } {

variable trajectory_dcd
set trajectory_dcd $my_trajectory_dcd

}


# An adapted version of Justin Gullingsrud's BigDCD script
#
# Purpose: Use this script to analyze one or more DCD files that don't fit into 
# memory.  The script will arrage for your analysis function to be called
# each time a frame is loaded, then delete the frame from memory.
# The analysis script must accept one argument; BigDCD will keep track of how
# many timesteps have been loaded and call your script with that number.
# 
proc adaptedBigdcd { script args } {
  global bigdcd_frame bigdcd_proc bigdcd_firstframe vmd_frame


  set ::bigdcd_is_done 0
  
  set bigdcd_frame 0
  set bigdcd_firstframe [molinfo top get numframes]
  set bigdcd_proc $script
  
  uplevel #0 trace variable vmd_frame w bigdcd_callback
  foreach dcd $args {
    animate read dcd $dcd waitfor 0
  }
}

proc bigdcd_callback { name1 name2 op } {
  global bigdcd_frame bigdcd_proc bigdcd_firstframe vmd_frame
 
  # If we're out of frames, we're also done
  set thisframe $vmd_frame($name2)
  if { $thisframe < $bigdcd_firstframe } {
    bigdcd_done
    return
  }
 
  incr bigdcd_frame
  if { [catch {uplevel #0 $bigdcd_proc $bigdcd_frame} msg] } { 
    puts stderr "bigdcd aborting at frame $bigdcd_frame\n$msg"
    bigdcd_done
    return
  }
  animate delete beg $thisframe end $thisframe 
  return $msg
}

proc bigdcd_done { } {
  set ::bigdcd_is_done 1
  puts "bigdcd_done"
  uplevel #0 trace vdelete vmd_frame w bigdcd_callback
}

proc adaptedBigdcd_wait_till_done {} {
  while {! $::bigdcd_is_done} {
    display update
  }
}


#
#The following procedure concatenates pdb files.
#
proc  TruncateTrajectory::ConcatPDB { file_1 file_2 } {

  set ch1 [open $file_1 RDWR]
  #To concatenate 2 pdb files you have to remove the word "END" from the first pdb file.
  seek $ch1 -200 end

  while {[eof $ch1] !=1} {
    gets $ch1 pdbline

    #First remove the word "END".
    if [string match END $pdbline]==1 {
      seek $ch1 -4 current
      puts $ch1 \ \ \ \ 
      #Then append a copy of the second at the end of the first.
      set ch2 [open $file_2 RDWR]
      fcopy $ch2 $ch1
    }

    unset pdbline
  }

  close $ch1
  close $ch2
}




#
#The following procedure puts the proper water molecules and ions around parts of the macromolecule that penetrate the unit cell boundaries.
#
proc  TruncateTrajectory::PeriodicContacts { type Xvector Yvector Zvector } {

  upvar frm frm
  upvar cutoff cutoff
  upvar n n
  variable macromolecule 
  variable selectedAtoms
  variable workDir
  
  set sys [mol new "${workDir}/wholeframe_$frm.pdb"]
  set macromol [atomselect $sys $macromolecule]

  #Create the periodic image of the macromol indicated by $type
  $macromol moveby [list $Xvector $Yvector $Zvector]
  animate write pdb "${workDir}/$type.pdb" beg 0 end 0 sel $macromol $sys

  #VMD cannot select atoms from an image created by the "moveby" or "move" command , so in order to select the closest to the 
  #periodic image of the macromolecule water molecules you have to concatenate 2 pdb files -one with the original macromolecule and 
  #another one with its image-.
  TruncateTrajectory::ConcatPDB "${workDir}/$type.pdb" "${workDir}/watersonly.pdb"

  #Now find which water molecules and ions are closer than $cutoff Angstroms from the image.
  set newsys [mol new "${workDir}/$type.pdb"]
  set watersphere [atomselect $newsys "same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule) and not $macromolecule" ]

  #Use the exact opposite translation (-x) to move these molecules and ions (if they exist), in a manner that they will be within 
  #$cutoff distance #from the original macromolecule.
  if { [$watersphere list] != {} } {
    $watersphere moveby [list [expr { -1*$Xvector }] [expr { -1*$Yvector }] [expr { -1*$Zvector }]]
    incr n
    #Save them in a pdb file 
    animate write pdb "${workDir}/extrawaters$n.pdb" beg 0 end 0 sel $watersphere $newsys
    $watersphere delete
  }

  mol delete $newsys
  mol delete $sys
  file delete "${workDir}/$type.pdb"
  $macromol delete
  unset Xvector Yvector Zvector
}



#
# The classic grep command where the output is writen in a file instead of printed on the screen.
#
proc  TruncateTrajectory::grep {re args outputfile} {
    set outfileID [open $outputfile "w"]
    set files [eval glob -types f $args]
    foreach file $files {
       set fp [open $file]
       while {[gets $fp line] >= 0} {
          if [regexp -- $re $line] {
             if {[llength $files] > 1} {puts -nonewline $file:}
             puts $outfileID $line
           }
       }
       close $fp
       close $outfileID
    }
 }


proc  TruncateTrajectory::IncludeWaters { frm } {

  variable pi 
  variable N_waters 
  variable keepAtoms_res 
  variable keepAtoms_seg 
  variable keepAtoms_type 
  variable answer1 
  variable answer2 
  variable HeteroCompounds
  variable macromolecule 
  variable selectedAtoms
  variable workDir
  variable frameNum
  
  set a [molinfo top get a]
  set b [molinfo top get b]
  set c [molinfo top get c]
  set alpha [expr { [molinfo top get alpha]*$pi/180 }]
  set beta [expr { [molinfo top get beta]*$pi/180 }]
  set gamma [expr { [molinfo top get gamma]*$pi/180 }]
  
  #For each frame of the trajectory creates a .pdb named "wholeframe_$frm.pdb' which will contain all atoms. This trick will 
  #make our job easier
  animate write pdb "${workDir}/wholeframe_$frm.pdb" beg 0 end 0 waitfor all top
  set sys [mol new "${workDir}/wholeframe_$frm.pdb"]
  set macromol [atomselect $sys $macromolecule]

  #It starts with a quite high cutoff distance (15) and decreases it gradualy scales it down until it reaches a distance within which there are 
  #slightly more waters than those required ($N_waters). 
  set cutoff 30
  while 1 {
    set waters [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule) and water)"]
    if { [llength [$waters list]] > [expr {3*$N_waters}] } {
      set cutoff [expr $cutoff -5]
      set listLength [llength [$waters list]]
      $waters delete
    } else {
      #When that distance is found increase $cutoff again by +1.
      set cutoff [expr $cutoff +5]
      $waters delete
      break
    }
  }
  
  while 1 {
    set waters [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule) and water)"]
    if { [llength [$waters list]] > [expr 3*$N_waters] } {
      set cutoff [expr $cutoff -1]
      $waters delete
    } else {
      #When that distance is found increase $cutoff again by +1.
      set cutoff [expr $cutoff +1]
      $waters delete
      break
    }
  }
  
  #Now we are sure that all the waters we want are within $cutoff of the macromolecule. Note that we are studying the original frame 
  #at the moment.
  
  #Create a pdb file containing all the molecules and ions, exept from the macromolecule and those hetero compounds that are within 
  #$cutoff from it.
  set notmacromol [atomselect $sys "not (same residue as within $cutoff of ($selectedAtoms and $macromolecule) or $macromolecule)"]
  animate write pdb "${workDir}/watersonly.pdb" beg 0 end 0 sel $notmacromol $sys
  
  #
  #Create the 26 closest periodic images 
  #
  set n 0
  
  #Start with +x translation
  ::TruncateTrajectory::PeriodicContacts +x $a 0 0
  
  #Follow the same procedure for translation -x.
  ::TruncateTrajectory::PeriodicContacts -x [expr { -1*$a }] 0 0
  
  #For translation +y.
  ::TruncateTrajectory::PeriodicContacts +y [expr { sin($pi/2-$gamma)*$b }] [expr { cos($pi/2-$gamma)*$b }] 0
  
  #For translation -y.
  ::TruncateTrajectory::PeriodicContacts -y [expr { -1*sin($pi/2-$gamma)*$b }] [expr { -1*cos($pi/2-$gamma)*$b }] 0
  
  #For translation +z.
  ::TruncateTrajectory::PeriodicContacts +z 0 [expr { sin($pi/2-$alpha)*$c }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation -z.
  ::TruncateTrajectory::PeriodicContacts -z 0 [expr { -1*sin($pi/2-$alpha)*$c }] [expr { -1*cos($pi/2-$alpha)*$c }]

  #For translation +x+y.
  ::TruncateTrajectory::PeriodicContacts +x+y [expr { $a+sin($pi/2-$gamma)*$b }] [expr { cos($pi/2-$gamma)*$b }] 0
  
  #For translation +x-y.
  ::TruncateTrajectory::PeriodicContacts +x-y [expr { $a-1*sin($pi/2-$gamma)*$b }] [expr { -1*cos($pi/2-$gamma)*$b }] 0
  
  #For translation +x+z.
  ::TruncateTrajectory::PeriodicContacts +x+z $a [expr { sin($pi/2-$alpha)*$c }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation +x-z.
  ::TruncateTrajectory::PeriodicContacts +x-z $a [expr { -1*sin($pi/2-$alpha)*$c }] [expr { -1*cos($pi/2-$alpha)*$c }]
  
  #For translation -x+y.
  ::TruncateTrajectory::PeriodicContacts -x+y [expr { -1*$a+sin($pi/2-$gamma)*$b}] [expr { cos($pi/2-$gamma)*$b }] 0
  
  #For translation -x-y.
  ::TruncateTrajectory::PeriodicContacts -x-y [expr { -1*($a+sin($pi/2-$gamma)*$b) }] [expr { -1*cos($pi/2-$gamma)*$b }] 0
  
  #For translation -x+z.
  ::TruncateTrajectory::PeriodicContacts -x+z [expr { -1*$a }] [expr { sin($pi/2-$alpha)*$c }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation -x-z.
  ::TruncateTrajectory::PeriodicContacts -x-z [expr { -1*$a }] [expr { -1*sin($pi/2-$alpha)*$c }] [expr { -1*cos($pi/2-$alpha)*$c }]
  
  #For translation -y+z.
  ::TruncateTrajectory::PeriodicContacts -y+z [expr { -1*sin($pi/2-$gamma)*$b }] [expr { -1*cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation -y-z.
  ::TruncateTrajectory::PeriodicContacts -y-z [expr { -1*sin($pi/2-$gamma)*$b }] [expr { -1*(cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c) }] [expr { -1*cos($pi/2-$alpha)*$c }]
  
  #For translation +y+z.
  ::TruncateTrajectory::PeriodicContacts +y+z [expr { sin($pi/2-$gamma)*$b }] [expr { cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation +y-z.
  ::TruncateTrajectory::PeriodicContacts +y-z [expr { sin($pi/2-$gamma)*$b }] [expr { cos($pi/2-$gamma)*$b-1*sin($pi/2-$alpha)*$c }] [expr { -1*cos($pi/2-$alpha)*$c }]
  
  #For translation +x+y+z.
  ::TruncateTrajectory::PeriodicContacts +x+y+z [expr { $a+sin($pi/2-$gamma)*$b }] [expr { cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation +x+y-z.
  ::TruncateTrajectory::PeriodicContacts +x+y-z [expr { $a+sin($pi/2-$gamma)*$b }] [expr { cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c }] [expr { -1*cos($pi/2-$alpha)*$c }]
  
  #For translation +x-y+z.
  ::TruncateTrajectory::PeriodicContacts +x-y+z [expr { $a-sin($pi/2-$gamma)*$b }] [expr { sin($pi/2-$alpha)*$c-cos($pi/2-$gamma)*$b }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation +x-y-z.
  ::TruncateTrajectory::PeriodicContacts +x-y-z [expr { $a-sin($pi/2-$gamma)*$b }] [expr { -1*(cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c) }] [expr { -1*(cos($pi/2-$alpha)*$c) }]
  
  #For translation -x+y+z.
  ::TruncateTrajectory::PeriodicContacts -x+y+z [expr { sin($pi/2-$gamma)*$b-$a }] [expr { cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation -x+y-z.
  ::TruncateTrajectory::PeriodicContacts -x+y-z [expr { sin($pi/2-$gamma)*$b-$a }] [expr { cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c }] [expr { -1*cos($pi/2-$alpha)*$c }]
  
  #For translation -x-y+z.
  ::TruncateTrajectory::PeriodicContacts -x-y+z [expr { -1*(sin($pi/2-$gamma)*$b+$a) }] [expr { sin($pi/2-$alpha)*$c-cos($pi/2-$gamma)*$b }] [expr { cos($pi/2-$alpha)*$c }]
  
  #For translation -x-y-z.
  ::TruncateTrajectory::PeriodicContacts -x-y-z [expr { -1*(sin($pi/2-$gamma)*$b+$a) }] [expr { -1*(cos($pi/2-$gamma)*$b+sin($pi/2-$alpha)*$c) }] [expr { -1*(cos($pi/2-$alpha)*$c) }]
  
  #For the identity operation.
  set watersphere [ atomselect $sys "same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule)" ]
  incr n
  animate write pdb "${workDir}/extrawaters$n.pdb" beg 0 end 0 sel $watersphere $sys
  
  #
  #Now put all 27 files together and create a .pdb file (named trunc_frame_$frm.pdb) containing the macromolecule and the 
  #heterocompounds which are closer than $cutoff Angstroms.
  #
  animate write pdb "${workDir}/truncframe_$frm.pdb" beg 0 end 0 sel $macromol $sys
  
  for {set m 1 } { $m<=$n } { incr m } {
    TruncateTrajectory::ConcatPDB "${workDir}/truncframe_$frm.pdb" "${workDir}/extrawaters$m.pdb"
    file delete "${workDir}/extrawaters$m.pdb"
  }
  
  file delete "${workDir}/watersonly.pdb"
  $macromol delete
  $watersphere delete
  $notmacromol delete
  mol delete $sys
  
  #The program searches for the a cutoff distance within which the number of waters is slightly lower that $N_waters.
  set sys [mol new "${workDir}/truncframe_$frm.pdb"]
  while 1 {
    set waters [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule)) and water"]
    set waternum [llength [$waters list]]
    if { $waternum > [expr 3*$N_waters] } {
      set cutoff [expr $cutoff -0.25]
      $waters delete
      unset waternum
    } else {
      break
    }
  }
  
  #It then saves the indices of the waters found and increases slightly the cutoff distance. So this time the number of waters 
  #within $cutoff of the macromolecule is slightly higher than $N_waters.
  set add_waternum [expr $N_waters-$waternum/3]
  set more_waters_ [atomselect $sys "(same residue as exwithin [expr $cutoff + 0.25] of ($selectedAtoms and $macromolecule)) and water"]
  set more_waters__ [$more_waters_ list]
  $more_waters_ delete
  #It removes the waters found twice (which happens to be within $cutoff and within $cutoff + 0.25).
  foreach atomID [$waters list] {
    set pos [lsearch -integer $more_waters__ $atomID]
    if { $pos != -1 } {
      set more_waters__ [lreplace $more_waters__ $pos $pos]
    }
    unset pos atomID
  }
  $waters delete
  
  #The remaining indices include all 3 atoms (OH2,H1,H2) of each water molecule. So here it makes a list of all different 
  #resid-segname pairs (each pair is included 3 times, one for each atom of a water molecule).
  set more_waters [atomselect $sys "index [string trim [list $more_waters__] \}\{ ]"]
  set reslist [$more_waters get resid]
  set seglist [$more_waters get segname]
  set res_seg_list {}
  foreach res $reslist seg $seglist {
    append ResSeg $res _ $seg
    if { [lsearch -exact -start end $res_seg_list $ResSeg]==-1 } {
      lappend res_seg_list $ResSeg
    }
    unset ResSeg res seg
  }
  $more_waters delete
  #Finally is measures the minimum distance of each water from the macromolecule, and makes a list. It sorts that list by 
  #increasing order and keeps only the number of waters that we need to reach $N_waters.
  set Rmin_list_ [list 0 0 30]
  set Rmin_list [list $Rmin_list_]
  unset Rmin_list_
  
  foreach res_seg $res_seg_list {
  
    set pair [split $res_seg _]
    unset res_seg
    set extra_water [atomselect $sys "resid [lindex $pair 0] and segname [lindex $pair 1]"]
    
    #We don't have to measure distances from all the atoms of the macromolecule. We already know that each water 
    #is more than ($cutoff-1) and less that $cutoff far from the macromolecule. Obviously a "within $cutoff" macro will 
    #catch some atoms of the macromolecule but not all, so our computation will be much faster!
    set check_atoms [atomselect $sys "(within [expr $cutoff + 0.25] of index [$extra_water list]) and ($selectedAtoms and $macromolecule)"]
    set Rmin 30
    foreach water_atom [$extra_water list] {
    
      foreach protein_atom [$check_atoms list] {
      
        set R [measure bond [list $water_atom $protein_atom]]
        
        if { $R<$Rmin } {
          unset Rmin
          set Rmin $R
        }
      }
    }
    
    set Rmin_list [linsert $Rmin_list end [list [lindex $pair 0] [lindex $pair 1] $Rmin]]
    
  }
  
  #It sorts the list of minimum distances, by increasing order, and removes the redundant residues.
  set Rmin_list [lsort -real -index end $Rmin_list]
  set Rmin_list [lreplace $Rmin_list $add_waternum end]
  
  #Finally it selects all the $N_waters closest waters and writes them in a new .pdb together with the macromolecule.
  set macromol [atomselect $sys $macromolecule]
  animate write pdb "${workDir}/final_frame_$frm.pdb" beg 0 end 0 sel $macromol $sys
  $macromol delete
  set _waters__ [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule)) and water"]
  set _waters_ [$_waters__ list]
  $_waters__ delete
  foreach sublist $Rmin_list {
    set res [lindex $sublist 0]
    set seg [lindex $sublist 1]
    set extra_water [atomselect $sys "segname $seg and resid $res"]
    append _waters_ \  [$extra_water list]
    $extra_water delete
    unset seg res
  }
  set only_water [atomselect $sys "index [string trim [list $_waters_] \}\{ ]"]
  animate write pdb "${workDir}/only_water.pdb" beg 0 end 0 sel $only_water $sys
  TruncateTrajectory::ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/only_water.pdb"
  file delete "${workDir}/only_water.pdb"
  
#
#This part refers to the case where the user wants to include no-water hetero compounds in the final trajectory but did not specify 
#them.
#
  if { $TruncateTrajectory::answer1=="true" && $TruncateTrajectory::answer2=="false" } {
    #
    #In a dcd file all frames must contain the same atoms.Moreover the molecules to which they belong must remain intact.The
    #following part selects which atoms must be included in each frame.
    #
    
    if { $frm==1 } {
  
      set chID1 [open "${workDir}/data_res.txt" a+]
      set chID2 [open "${workDir}/data_seg.txt" a+]
      set chID3 [open "${workDir}/data_type.txt" a+]
      #At first it works with the first frame.
      set atomlist [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule)) and not (water or $macromolecule)"]
      if { [llength [$atomlist list]]!=0 } {
        animate write pdb "${workDir}/hetero_not_water.pdb" beg 0 end 0 sel $atomlist $sys
        TruncateTrajectory::ConcatPDB "${workDir}/final_frame_1.pdb" "${workDir}/hetero_not_water.pdb"
      }
      #In order to locate the same atom in 2 different pdbs you have to use its residue ID, its segment ID, and its 
      #type name. Below the program creates 3 "keepAtoms" lists which will contain the IDs of all the non-water hetero 
      #atoms, which will be included in all the frames of the final trajectory.
      set keepAtoms_res [[list $atomlist] get resid]
      set keepAtoms_seg [[list $atomlist] get segid]
      set keepAtoms_type [[list $atomlist] get type]
  
      #It also saves these IDs into 3 different files to save some memory.These files will be deleted at the end of 
      #the computation.
      puts $chID1 "[list $keepAtoms_res]"
      puts $chID2 "[list $keepAtoms_seg]"
      puts $chID3 "[list $keepAtoms_type]"
  
      close $chID1
      close $chID2
      close $chID3
  
    } else {
      set chID1 [open "${workDir}/data_res.txt" WRONLY]
      set chID2 [open "${workDir}/data_seg.txt" WRONLY]
      set chID3 [open "${workDir}/data_type.txt" WRONLY]
      seek $chID1 0 end
      seek $chID2 0 end
      seek $chID3 0 end

      #The following part finds which "non-water hetero" atoms of this final_frame_$frm.pdb were not included
      #in the previous final_frames_$frm.pdbs, and puts the IDs of those atoms in the 3 "keepAtoms" lists.
      set atomlist [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule)) and not (water or $macromolecule)"]
      if { [llength [$atomlist list]]!=0 } {
        animate write pdb "${workDir}/hetero_not_water.pdb" beg 0 end 0 sel $atomlist $sys
        TruncateTrajectory::ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/hetero_not_water.pdb"
      }
      set checkAtoms_res [[list $atomlist] get resid]
      set checkAtoms_seg [[list $atomlist] get segid]
      set checkAtoms_type [[list $atomlist] get type]
      $atomlist delete
  
      puts $chID1 "[list $checkAtoms_res]"
      puts $chID2 "[list $checkAtoms_seg]"
      puts $chID3 "[list $checkAtoms_type]"
  
      foreach resID $checkAtoms_res segID $checkAtoms_seg typeID $checkAtoms_type {
        set pos_res [lsearch -exact -all $keepAtoms_res $resID]
        if [list $pos_res]=={} {
          set keepAtoms_res [linsert $keepAtoms_res end $resID]
          set keepAtoms_seg [linsert $keepAtoms_seg end $segID]
          set keepAtoms_type [linsert $keepAtoms_type end $typeID]
          continue
        } else {
          set check -1
          foreach pos_seg $pos_res {
            set segment [lindex $keepAtoms_seg $pos_seg]
            set type [lindex $keepAtoms_type $pos_seg]
            if { $segment==$segID && $type==$typeID } {
              set check 1
              break
            }
          }
          if { $check==-1 } {
            set keepAtoms_res [linsert $keepAtoms_res end $resID]
            set keepAtoms_seg [linsert $keepAtoms_seg end $segID]
            set keepAtoms_type [linsert $keepAtoms_type end $typeID]
          }
        }
  
        unset resID segID typeID
      }
      unset checkAtoms_res checkAtoms_seg checkAtoms_type
  
      close $chID1
      close $chID2
      close $chID3
    }
    file delete "${workDir}/hetero_not_water.pdb"
#
#The following part referes to the case in which the user has defined the exact number and kind of non-water hetero
#compounds to be included in the final trajectory.
#
  } elseif { $TruncateTrajectory::answer1=="true" && $TruncateTrajectory::answer2=="true" } {

    #The program takes below each given compound and works on it separately.
    foreach compound $TruncateTrajectory::HeteroCompounds {

      mol delete $sys
      set sys [mol new "${workDir}/truncframe_$frm.pdb"]
      #The concept is to increase the cutoff distance gradualy and each time to use the "exwithin macro" to see if the number 
      #of residues of this type is equal to the one defined by the user. The initial cutoff is 4 A by default.
      set cutoff 1
      set num 0
      #First it retrieves the segname, the atom types and the number of occurrence of this compound (as specified by the user).
      set seg [lindex $compound 0]
      set types [lsort [lrange $compound 2 end]]
      set N [lindex $compound 1]
      #Then it measures the number of these residues in truncframe_$frm.pdb (remember that this file contains all atoms of the 
      #27 nearest unit cells -26 periodic plus the identity- which are within a slightly longer cutoff than the one we need, see 
      #lines 100,212).
      set hetero_atoms [atomselect $sys "segname $seg and type [list $types]"]
      set residue_num [expr [llength [$hetero_atoms list]]/[llength $types]]
      #If this number is lower than the one we need, it adds to final_frame_$frm.pdb all those residues contained in 
      #truncframe_$frm.pdb, and continues with wholeframe_$frm.pdb.
      if { $residue_num<$N } {
        set notresid [lsort -unique [$hetero_atoms get resid]]
        set N [expr $N-$residue_num]
        animate write pdb "${workDir}/hetero_compounds.pdb" beg 0 end 0 sel $hetero_atoms $sys
        TruncateTrajectory::ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/hetero_compounds.pdb"
        mol delete $sys
        unset sys
  
        set sys [mol new "${workDir}/wholeframe_$frm.pdb"]
        if { $residue_num>0 } {
          set hetero_atoms [atomselect $sys "not water and segname $seg and type [list $types] and not resid [string trim $notresid \}\{]"]
        } elseif { $residue_num==0 } {
          set hetero_atoms [atomselect $sys "not water and segname $seg and type [list $types]"]
        }

        #It also measures the number of these residues in wholeframe_$frm.pdb.
        set whole_residue_num [expr [llength [$hetero_atoms list]]/[llength $types]]
        if { $whole_residue_num<=$N } {
            #If it contains exactly N, or if it doesn't contain sufficient number of residues to reach N, then ...
            if { $whole_residue_num<$N } {
              puts "\n\n\nNumber of occurrence incorrect! \nMaximum number of occurrence for compound: $seg $types  is [expr $whole_residue_num + $residue_num] . The program will continue and include just [expr $whole_residue_num + $residue_num] residues of this compound.\n\n\n"
            }
          animate write pdb "${workDir}/hetero_compounds.pdb" beg 0 end 0 sel $hetero_atoms $sys
          ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/hetero_compounds.pdb"
          continue
        }

      #If truncframe_$frm.pdb contains exactly N such residues, then it put them all in final_frame_$frm.pdb and continues with the 
      #next compound.
      } elseif { $residue_num==$N } {
        animate write pdb "${workDir}/hetero_compounds.pdb" beg 0 end 0 sel $hetero_atoms $sys
        ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/hetero_compounds.pdb"
        continue
      }

      #Finaly if runcframe_$frm.pdb contains more than N residues, it enteres the while loop and measures their minimum distance 
      #from the macromolecule.
      $hetero_atoms delete
      set addAtoms {}
  
      while 1 {

        if { $residue_num==0 || $residue_num>[lindex $compound 1] } {
          #It selects all non-water atoms with segname "$seg" and type "$types", which are within cutoff
          #distance from the macromolecule.
          set Hetero [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule)) and (not water and segname $seg and type [list $types])"]
        } else {
          #It selects all non-water atoms with segname "$seg" and type "$types", which are within cutoff distance from the 
          #macromolecule, and are not included in final_frame_$frm.pdb.
          set Hetero [atomselect $sys "(same residue as exwithin $cutoff of ($selectedAtoms and $macromolecule)) and (not water and segname $seg and type [list $types])and not resid [string trim $notresid \}\{]"]
        }
        set resList [$Hetero get resid]
        $Hetero delete
  
        #Then it retrieves their resids. If the compound includes more than one type of atom, each resid will exist 
        #multiple times in the list. In that case the programs keeps only one copy of each of them.
        set differentRes [lsort -unique $resList]
        unset resList
        if { [llength $differentRes]!=0 } {
  
          foreach res $addAtoms {
              set pos [lsearch $differentRes $res]
              set differentRes [lreplace $differentRes $pos $pos]
              unset pos res
          }
  
        }
        set num [expr [llength $differentRes]+[llength $addAtoms]]

        #Now it checks the number of compounds of that kind ($num) found during all the iterations of the while loop. If it is lower 
        #than the one defined by the user ($N), the cutoff is increased by 1. If there were valid residue found during the last 
        #iteration of the while loop, their resids are saved into a list ($addAtoms).
        if { $num<$N } {
  
          foreach res $differentRes {
            lappend addAtoms $res
          }
          set cutoff [expr $cutoff+2]

          #If $num is equal with $N, then all the residues found are added to final_frame_$frm.pdb and the program continues with 
          #the next compound.
        } elseif { $num==$N } {
  
          foreach res $differentRes {
            lappend addAtoms $res
          }
          set putAtoms [atomselect $sys "resid [ string trim $addAtoms \}\{ ] and segname $seg and type [list $types]"]
          animate write pdb "${workDir}/hetero_compounds.pdb" beg 0 end 0 sel $putAtoms $sys
          $putAtoms delete
           TruncateTrajectory::ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/hetero_compounds.pdb"
          break

          #If $num is higher than $N, then it must measure distances from the macromolecule.
        } elseif { $num>$N } {
          #Our objective is to find the N closest to the macromolecule residues. The valid residues found during the previous
          #iterations of the while loop were less than $N, so they will all be included in the final_frame_$frm.pdb. Thus if and we
          #require 2 more residues to reach $N we only need to find which 2 from the ones found during the last iteration are the 
          #closest.
          set moreAtoms [atomselect $sys "resid [ string trim $differentRes \}\{ ] and segname $seg and type [list $types]"]
          set reslist [$moreAtoms get resid]
          $moreAtoms delete
          #Below the program measures the minimum distance for each valid residue and makes a list.
          set Rmin_list [list -1 1000]
          set Rmin_list_ [list $Rmin_list]
          unset Rmin_list
          foreach res $reslist {
            set extra_atoms [atomselect $sys "resid $res and segname $seg and type [list $types]"]
            #We don't have to measure distances from all the atoms of the macromolecule. We already know that each valid residue 
            #is more than ($cutoff-1) and less that $cutoff far from the macromolecule. Obviously a "within $cutoff" macro will 
            #catch some atoms of the macromolecule but not all!
            set check_atoms [atomselect $sys "(within $cutoff of index [$extra_atoms list]) and ($selectedAtoms and $macromolecule)"]
            set Rmin 30
            foreach hetero_atom [$extra_atoms list] {
              foreach protein_atom [$check_atoms list] {
                set R [measure bond [list $hetero_atom $protein_atom]]
                if { $R<$Rmin } {
                  unset Rmin
                  set Rmin $R
                  unset R
                }
                unset protein_atom
              }
              unset hetero_atom
            }
            set Rmin_list_ [linsert $Rmin_list_ end [list $res $Rmin]]
            $extra_atoms delete
            $check_atoms delete
            unset Rmin res
          }
          #It sorts the list of minimum distances, by increasing order, and removes the redundant residues.
          set Rmin_list [lsort -real -index end $Rmin_list_]
          unset Rmin_list_
          set keep [expr $N-[llength $addAtoms]]
          set RminList [lreplace $Rmin_list $keep end]
          unset keep
          #Finaly it puts their resids in the $addAtoms list and adds all the $N closest residues found in the final_frame_$frm.pdb.
          foreach sublist $RminList {
              set res [lindex $sublist 0]
              lappend addAtoms $res
              unset res sublist
          }
          set hetero_atoms [atomselect $sys "segname $seg and resid [string trim $addAtoms \}\{] and type [list $types]"]
  
          animate write pdb "${workDir}/hetero_compounds.pdb" beg 0 end 0 sel $hetero_atoms $sys
           TruncateTrajectory::ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/hetero_compounds.pdb"
          $hetero_atoms delete
  
          break
        }
  
      }
      unset addAtoms
  
    }
  
}
  mol delete $sys
  file delete "${workDir}/hetero_compounds.pdb"
  file delete "${workDir}/wholeframe_$frm.pdb"
  file delete "${workDir}/truncframe_$frm.pdb"
  unset a b c alpha beta gamma

}



#
#This part refers to the case where the user wants to include no-water hetero compounds in the final trajectory but didn't specify 
#them.
#

proc  TruncateTrajectory::IncludeHetatms { frm } {

  variable keepAtoms_res 
  variable keepAtoms_seg 
  variable keepAtoms_type 
  variable hetero_atoms 
  variable macromolecule 
  variable selectedAtoms
  variable workDir
  variable frameNum
  
  #Below the program takes each truncframe_$frm.pdb and retrieves the IDs of its non-water hetero atoms.
  set chID1 [open "${workDir}/data_res.txt" r]
  set chID2 [open "${workDir}/data_seg.txt" r]
  set chID3 [open "${workDir}/data_type.txt" r]
  
  animate write pdb "${workDir}/wholeframe_$frm.pdb" beg 0 end 0 waitfor all top
  set sys [mol new "${workDir}/wholeframe_$frm.pdb"]
  set putAtoms_res {}
  set putAtoms_seg {}
  set putAtoms_type {}
  set checkAtoms_res [lindex [read $chID1] [expr $frm-1]]
  set checkAtoms_seg [lindex [read $chID2] [expr $frm-1]]
  set checkAtoms_type [lindex [read $chID3] [expr $frm-1]]
  
  
  # Then it checks the 3 "keepAtoms" lists and finds which non-water hetero atoms are not included in that specific 
  # truncframe_$frm.pdb.The IDs of those atoms are saved into 3 "putAtoms" lists.
  foreach resID $keepAtoms_res segID $keepAtoms_seg typeID $keepAtoms_type {

    set pos_res [lsearch -exact -all $checkAtoms_res $resID]
    if [list $pos_res]=={} {
      set putAtoms_res [linsert $putAtoms_res end $resID]
      set putAtoms_seg [linsert $putAtoms_seg end $segID]
      set putAtoms_type [linsert $putAtoms_type end $typeID]
      continue
    } else {
      set check -1
      foreach pos_seg $pos_res {
        set segment [lindex $checkAtoms_seg $pos_seg]
        set type [lindex $checkAtoms_type $pos_seg]
        if { $segment==$segID && $type==$typeID } {
          set check 1
          break
        }
      }
      if { $check==-1 } {
        set putAtoms_res [linsert $putAtoms_res end $resID]
        set putAtoms_seg [linsert $putAtoms_seg end $segID]
        set putAtoms_type [linsert $putAtoms_type end $typeID]
      }
    }
  
  }
  # If the program has found any extra atoms it uses their IDs to locate them in a copy of the original "full frame" (wholeframe_
  # $frm.pdb) and adds them to final_frame_$frm.pdb.
  if [llength $putAtoms_res]!=0  {
    set extra_Atoms {}
  
    foreach seg $putAtoms_seg res $putAtoms_res typ $putAtoms_type {
      set extra_atom [atomselect $sys "segname $seg and resid $res and type $typ"]
      set extra_Atoms [linsert $extra_Atoms end [$extra_atom list]]
      $extra_atom delete
      unset seg res typ
    }
    set extra_Atoms [atomselect $sys "index [string trim [list $extra_Atoms] \}\{ ]"]
    animate write pdb "${workDir}/extra_Atoms.pdb beg" 0 end 0 waitfor all sel $extra_Atoms $sys
    TruncateTrajectory::ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/extra_Atoms.pdb"
  
    $extra_Atoms delete
    unset putAtoms_seg putAtoms_res putAtoms_type checkAtoms_res checkAtoms_seg checkAtoms_type
    file delete "${workDir}/extra_Atoms.pdb"
  }
  
  mol delete $sys
  file delete "${workDir}/wholeframe_$frm.pdb"
  
  close $chID1
  close $chID2
  close $chID3

}


proc TruncateTrajectory::HetatmsNotSpecified {} {

  variable frm
  variable trajectory_psf
  variable trajectory_dcd
  # default values:
  variable macromolecule 
  variable selectedAtoms
  variable N_waters 
  variable answer1 
  variable answer2
  variable catdcdPath
  variable workDir
  variable frameNum
  
  mol delete all
  
  mol load psf "$trajectory_psf"
  
  adaptedBigdcd TruncateTrajectory::IncludeHetatms "$trajectory_dcd"
  
  adaptedBigdcd_wait_till_done

  file delete "${workDir}/data_res.txt"
  file delete "${workDir}/data_seg.txt"
  file delete "${workDir}/data_type.txt"
  
  #
  #Some segments like ION may not consist of the same atom types (i.e. SOD, CLA, etc). These segments must have the same indices in 
  #all the frames of the final trajectory(the one that the user will create with psfgen and catdcd plugins). Otherwise we may see a SOD 
  #in the position #of a CLA, and vice versa. The program first recreates the final_frame_1.pdb, this time without the redundant lines 
  #which were created by the "ConcatPDB" proc.
  mol new "${workDir}/final_frame_1.pdb"
  animate write pdb "${workDir}/final_frame_1.pdb" beg 0 end 0 waitfor all sel [atomselect top all] top
  mol delete all
  set mol1 [mol new "${workDir}/final_frame_1.pdb"]
  set hetero_atoms [ atomselect top "hetero and not water" ]
  
  if { [llength [$hetero_atoms list]]!=0 } {
    #It then takes final_frame_1.pdb as a pattern, and redistributes the atoms of the other final_frame_$frm.pdbs in order to have 
    #the same succession ==> same index.
    regexp "Total\ frames:\ (\[0-9]+)" [exec $catdcdPath -num "${workDir}/devision.dcd"] matchStr devisionFrameNum
    for {set frm 2 } {$frm <= $devisionFrameNum} {incr frm} {
  
      puts "Redistributing atoms in frame_$frm"
      set mol2 [mol new "${workDir}/final_frame_$frm.pdb"]
      mol top $mol1
      set ch1 [open "${workDir}/hetero_atomlist.pdb" a+]
      foreach atom_ID [$hetero_atoms list] {
        set addAtom [atomselect $mol1 "index $atom_ID"]
        set res [$addAtom get resid]
        set seg [$addAtom get segname]
        set typ [$addAtom get type]
        set extra_atom [atomselect $mol2 "segname $seg and resid $res and type $typ"]
        animate write pdb "${workDir}/redistribution_file.pdb" beg 0 end 0 sel $extra_atom $mol2
        set ch2 [open "${workDir}/redistribution_file.pdb" r]
        seek $ch2 -83 end
        gets $ch2 line
        puts $ch1 $line
        close $ch2
        $extra_atom delete
        $addAtom delete
        unset res seg typ atom_ID line
      }
      puts $ch1 "END"
      close $ch1
      mol top $mol2
      set protein_and_water [atomselect $mol2 "$macromolecule or water"]
      animate write pdb "${workDir}/final_frame_$frm.pdb" beg 0 end 0 waitfor all sel $protein_and_water $mol2
      TruncateTrajectory::ConcatPDB "${workDir}/final_frame_$frm.pdb" "${workDir}/hetero_atomlist.pdb"
      mol delete $mol2
      $protein_and_water delete
      unset mol2 ch1 ch2
      file delete "${workDir}/hetero_atomlist.pdb"
    }
    $hetero_atoms delete
    file delete "${workDir}/redistribution_file.pdb"
    unset frm
  }

}



proc TruncateTrajectory::AssembleTrajectory {} {

    variable trajectory_dcd
    variable catdcdPath
    variable macromolecule
    variable workDir


regexp "Total\ frames:\ (\[0-9]+)" [exec $catdcdPath -num "${workDir}/devision.dcd"] matchStr devisionFrameNum
for {set frm 1 } {$frm <= $devisionFrameNum} {incr frm} {

  mol new "${workDir}/final_frame_$frm.pdb"
  
  resetpsf
  # Read topology file
  topology $TruncateTrajectory::toppar
  
  set residueNames [lsort -unique [[atomselect top $macromolecule] get chain]]
  foreach macromol_resname $residueNames {
    TruncateTrajectory::grep "${macromol_resname}\ *$" "${workDir}/final_frame_$frm.pdb" "${workDir}/frag${macromol_resname}_${frm}.pdb"
    
    # Build macromolecule segment
    segment $macromol_resname {
      pdb "${workDir}/frag${macromol_resname}_${frm}.pdb"
    }
    
    # Read protein coordinates from PDB file
    pdbalias atom ILE CD1 CD     ; # formerly "alias atom ..."
    coordpdb "${workDir}/frag${macromol_resname}_$frm.pdb" ${macromol_resname}
  }
  
  set segmentNames [lsort -unique [[atomselect top water] get segname]]
  foreach water_segname $segmentNames {
    TruncateTrajectory::grep "${water_segname}\ *$" "${workDir}/final_frame_$frm.pdb" "${workDir}/seg_${water_segname}_${frm}.pdb"
    
    # Build water segment WT2
    segment $water_segname {
      pdb "${workDir}/seg_${water_segname}_$frm.pdb"
    }
    
    # Read water coordinaes from PDB file
    coordpdb "${workDir}/seg_${water_segname}_$frm.pdb" ${water_segname}
    
  }
  
  set segmentNames [lsort -unique [[atomselect top "hetero and not water"] get segname]]
  foreach hetero_segname $segmentNames {
    TruncateTrajectory::grep "${hetero_segname}\ *$" "${workDir}/final_frame_$frm.pdb" "${workDir}/seg_${hetero_segname}_${frm}.pdb"
    
    segment $hetero_segname {
      auto none
      pdb "${workDir}/seg_${hetero_segname}_$frm.pdb"
    }
  
    # Read hetero compound coordinaes from PDB file
    coordpdb "${workDir}/seg_${hetero_segname}_$frm.pdb" ION
  }
  
  # Guess missing coordinates
  guesscoord
  
  # Write structure and coordinate files
  writepsf "${workDir}/frame_$frm.psf"
  writepdb "${workDir}/frame_$frm.pdb"
  
  # End of psfgen commands
  mol new "${workDir}/frame_$frm.pdb"
  animate write dcd "${workDir}/frame_$frm.dcd" beg 0 end 0 top
  mol delete all
  resetpsf
  
  # Use catdcd to create the final truncated trajectory.
  if { $frm==1 } {
    exec $catdcdPath -o "${workDir}/final_1.dcd" -dcd "${workDir}/frame_1.dcd"
    file delete "${workDir}/frame_1.dcd"
    file delete "${workDir}/frame_1.pdb"
  } else {
    exec $catdcdPath -o "${workDir}/final_$frm.dcd" -dcd "${workDir}/final_[expr $frm-1].dcd" -dcd "${workDir}/frame_$frm.dcd"
    file delete "${workDir}/final_[expr $frm-1].dcd"
    file delete "${workDir}/frame_$frm.dcd"
    file delete "${workDir}/frame_$frm.psf"
    file delete "${workDir}/frame_$frm.pdb"
  }
  set fragfiles [glob -nocomplain "${workDir}/frag*_${frm}.pdb"]
  foreach fragfile $fragfiles {
    file delete $fragfile
  }
  set segfiles [glob -nocomplain "${workDir}/seg_*_${frm}.pdb"]
  foreach segfile $segfiles {
    file delete $segfile
  }
  #file delete "${workDir}/wt1_$frm.pdb"
  #file delete "${workDir}/wt2_$frm.pdb"
  #file delete "${workDir}/wt3_$frm.pdb"
  #file delete "${workDir}/wt4_$frm.pdb"
  file delete "${workDir}/final_frame_$frm.pdb"
}

file rename -force "${workDir}/final_[expr $frm-1].dcd" "${workDir}/truncated_devision.dcd"
#puts "file rename -force ${workDir}/final_[expr $frm-1].dcd ${workDir}/truncated_devision.dcd"

unset  residueNames segmentNames fragfiles 
}


proc TruncateTrajectory::RunPlugin {} {

  variable trajectory_psf
  variable trajectory_dcd
  variable answer1
  variable answer2
  variable workDir
  variable catdcdPath
  variable frameNum
  
  set original_dcd $trajectory_dcd
  ### count number of frames ###
  if [regexp "Total\ frames:\ (\[0-9]+)" [exec $catdcdPath -num "$trajectory_dcd"] matchStr frameNum] {
    puts "number of frames = $frameNum"
  
  
  for { set start 0 } { $start <= $frameNum } { set start [expr $start+10] } {
    
    if { [expr {$frameNum-$start}] >= 10 } {
        set end [expr {$start + 9} ]
    } else {
        set end $frameNum
    }
    exec $catdcdPath -o "${workDir}/devision.dcd" -first $start -last $end "$original_dcd"
    
    TruncateTrajectory::DCDSetter "${workDir}/devision.dcd"
    mol load psf "$trajectory_psf"
    adaptedBigdcd TruncateTrajectory::IncludeWaters "${workDir}/devision.dcd"
    adaptedBigdcd_wait_till_done
    
    if { $answer1 == "true" && $answer2 == "false" } {
        TruncateTrajectory::HetatmsNotSpecified
    }
    TruncateTrajectory::AssembleTrajectory
    
    if { $start >= 10 && $end < $frameNum } { # for intermediate frames
      exec $catdcdPath -o "${workDir}/truncated_trajectory_0-${end}.dcd" "${workDir}/truncated_trajectory_0-[expr {${end}-10}].dcd" "${workDir}/truncated_devision.dcd"
      file delete "${workDir}/truncated_trajectory_0-[expr {${end}-10}].dcd"
      #file rename -force "${workDir}/truncated_devision.dcd" "${workDir}/truncated_trajectory_.dcd"
    } elseif { $start >= 0 && $end == $frameNum } { # for the last less that 10 frames
        exec $catdcdPath -o "${workDir}/truncated_trajectory.dcd" "${workDir}/truncated_trajectory_0-[expr {${start}-1}].dcd" "${workDir}/truncated_devision.dcd"
        file delete "${workDir}/truncated_trajectory_0-[expr {${start}-1}].dcd"
        #exec $catdcdPath -o "${workDir}/truncated_trajectory.dcd" "${workDir}/truncated_trajectory.dcd" "${workDir}/truncate_devision.dcd"
    } else { # for the first 10 frames
        file rename -force "${workDir}/truncated_devision.dcd" "${workDir}/truncated_trajectory_${start}-${end}.dcd"
    }
  }
  file rename -force "${workDir}/frame_1.psf" "${workDir}/truncated_trajectory.psf"
  file delete "${workDir}/truncated_devision.dcd"
  file delete "${workDir}/devision.dcd"

    puts "Program finished successfully !!!"

    mol load psf "${workDir}/truncated_trajectory.psf"
    mol addfile "${workDir}/truncated_trajectory.dcd"
  }
}




#source /home/thomas/Documents/Molecular_Dynamics/adapted_bigdcd.tcl


proc TruncateTrajectory::tt_GUI {} {

  variable w
  
  
  # make the initial window
  set w [toplevel ".tt_GUI"]
  wm title $w "Truncate Trajcetory"
  wm resizable $w 0 1 
  puts "w = $w"
  label $w.psfLab -text {  Select trajcetory .psf file :  }
  entry $w.psfPathEntry -width 40 -textvariable TruncateTrajectory::trajectory_psf
  
  button $w.psfBut -command {set types {
          { ".psf Files"     {.psf}    }
          }
          set tempfile [tk_getOpenFile -filetypes $types -parent .]
          set TruncateTrajectory::trajectory_psf $tempfile} \
          -text Open 
          
  label $w.dcdLab -text {  Select trajcetory .dcd file :  }
  entry $w.dcdPathEntry -width 40 -textvariable TruncateTrajectory::trajectory_dcd
  
  button $w.dcdBut \
          \
          -command {set types {
          { ".dcd Files"     {.dcd}    }
          }
          set tempfile [tk_getOpenFile -filetypes $types -parent .]
          set TruncateTrajectory::trajectory_dcd $tempfile} \
          -text Open 
          
  label $w.catdcdLab -text {  Select catdcd executable :  }
  entry $w.catdcdPathEntry -width 40 -textvariable TruncateTrajectory::catdcdPath
  
  button $w.catdcdBut -text "Browse" \
    -command {
      set tempfile [tk_getOpenFile -parent .]
      if {![string equal $tempfile ""]} { set TruncateTrajectory::catdcdPath $tempfile }
    }
          
  label $w.workDirLab -text {  Select Work Directory :  }
  entry $w.workDirEntry -width 40 -textvariable TruncateTrajectory::workDir
  
  button $w.workDirBut -text "Browse" \
    -command {
      TruncateTrajectory::chooseTempDir
      #set tempfile [tk_getOpenFile]
      #if {![string equal $tempDir ""]} { set ::TruncateTrajectory::workDir $tempDir }
    }
    proc chooseTempDir {} {
        set gooddir 0
        while { $gooddir == 0 } {
          set dir [ tk_chooseDirectory -mustexist true -title "Choose Work Directory" -initialdir TruncateTrajectory::workDir -parent . ]
          if { $dir == "" } {
            return
          }
          if { [file writable "$dir"] } {
            set gooddir 1
            set TruncateTrajectory::workDir $dir
          } else {
            tk_messageBox -type ok -message "Error: $dir is not writable"
          }
        }
    }
    
  label $w.topparLab -text {Select the correct topology file :}
  entry $w.topparEntry -width 40 -textvariable TruncateTrajectory::toppar
  
  button $w.topparBut -text "Browse" \
    -command {
      set tempfile [tk_getOpenFile -parent .]
      if {![string equal $tempfile ""]} { set TruncateTrajectory::toppar $tempfile }
    }

          
  label $w.macromolLab -text "Enter macromolecule selection:"
  entry $w.macromolEnt -width 40 -relief sunken -bd 2 -textvariable TruncateTrajectory::macromolecule
  focus $w.macromolEnt
          
  label $w.atomLab -text "Enter atom selection:"
  entry $w.atomEnt -width 40 -relief sunken -bd 2 -textvariable TruncateTrajectory::selectedAtoms
  focus $w.atomEnt
          
  radiobutton $w.noHetatmCBut -text "Include no hetero compounds apart from waters" -variable TruncateTrajectory::answer -value "false_false" -anchor w
  radiobutton $w.inclHetatmCBut -text "Let the program include the closest\nhetero compounds automatically" -variable TruncateTrajectory::answer -value "true_false" -anchor w
  radiobutton $w.defHetatmCBut -text "Define hetero compounds to be included" -variable TruncateTrajectory::answer -value "true_true" -anchor w
  
  set waterState "disabled" 
  set cutoffState "disabled"
  
  label $w.constNumLab -text "Constant number of waters"
  ### Replace radiobutton with the following ###
  #radiobutton .waterRBut -text "Constant number of waters" -variable constWaters -command {.waterEnt configure -state normal 
  #.cutoffEnt configure -state disabled}
  #radiobutton .cutoffRBut -text "Constant cutoff distance" -variable constWaters -command {.waterEnt configure -state disabled 
  #.cutoffEnt configure -state normal}
  
  entry $w.waterEnt -width 40 -relief sunken -bd 2 -textvariable TruncateTrajectory::N_waters -state normal
  #entry .waterEnt -width 40 -relief sunken -bd 2 -textvariable N_waters -state $waterState
  #entry .cutoffEnt -width 40 -relief sunken -bd 2 -textvariable cutoff  -state $cutoffState
  
  
  scrollbar $w.s -command "$w.l yview"
  listbox $w.l -yscroll "$w.s set" -width 60
  bind $w.l <Double-B1-ButtonRelease> {setLabel [$w.l get active]}
  label $w.segLab -text "segname : "
  entry $w.segEnt -width 10 -relief sunken -bd 2 -textvariable seg
  focus $w.segEnt
  label $w.numLab -text "number : "
  entry $w.numEnt -width 10 -relief sunken -bd 2 -textvariable num
  focus $w.numEnt
  label $w.typeLab -text "type : "
  entry $w.typeEnt -width 10 -relief sunken -bd 2 -textvariable type
  focus $w.typeEnt
  button $w.addBut -command {$TruncateTrajectory::w.l insert end "$seg $num \{$type\}"} -text "add"
  button $w.removeBut -command {$TruncateTrajectory::w.l delete [$TruncateTrajectory::w.l curselection] [$TruncateTrajectory::w.l curselection]} -text "remove"
  
  button $w.submitBut -text "   submit   " -command { 
    regexp "(.*)\_(.*)" $TruncateTrajectory::answer matchStr answer1 answer2
    #TruncateTrajectory::Setter DCD_with_cell.psf DCD_with_cell.dcd "protein" "resid 1 to 50" "500" "true" "false" "/home/thomas/Documents/Downloads" 
    TruncateTrajectory::catdcdPathSetter $TruncateTrajectory::catdcdPath
    set TruncateTrajectory::HeteroCompounds [$TruncateTrajectory::w.l get 0 end]
    #TruncateTrajectory::AddHeteroCompound "ION" "2" "SOD"
    #TruncateTrajectory::AddHeteroCompound "ION" "1" "CLA"
    TruncateTrajectory::Setter $TruncateTrajectory::trajectory_psf $TruncateTrajectory::trajectory_dcd $TruncateTrajectory::macromolecule $TruncateTrajectory::selectedAtoms $TruncateTrajectory::N_waters $answer1 $answer2 $TruncateTrajectory::workDir
    TruncateTrajectory::RunPlugin
  }
  
  grid $w.l -row 10 -column 0 -sticky w -columnspan 3
  grid $w.s -row 10 -column 2 -sticky wn
  grid $w.addBut -row 10 -column 2 -sticky s
  grid $w.removeBut -row 11 -column 2 -sticky s
  
  #The "sticky" option aligns items to the left (west) side
  grid $w.noHetatmCBut -row 6 -column 1 -sticky w
  grid $w.inclHetatmCBut -row 7 -column 1 -sticky w
  grid $w.defHetatmCBut -row 8 -column 1 -sticky w
  grid $w.constNumLab -row 4 -column 0 -sticky w
  #grid $w.cutoffRBut -row 5 -column 0 -sticky w
  grid $w.waterEnt -row 4 -column 1 -sticky w
  #grid $w.cutoffEnt -row 5 -column 1 -sticky w
          
  grid $w.psfLab -row 0 -column 0
  grid $w.psfPathEntry -row 0 -column 1
  grid $w.psfBut -row 0 -column 2
  grid $w.dcdLab -row 1 -column 0
  grid $w.dcdPathEntry -row 1 -column 1
  grid $w.dcdBut -row 1 -column 2
  grid $w.catdcdLab -row 14 -column 0
  grid $w.catdcdPathEntry -row 14 -column 1
  grid $w.catdcdBut -row 14 -column 2
  grid $w.workDirLab -row 15 -column 0
  grid $w.workDirEntry -row 15 -column 1
  grid $w.workDirBut -row 15 -column 2
  grid $w.topparLab -row 16 -column 0
  grid $w.topparEntry -row 16 -column 1
  grid $w.topparBut -row 16 -column 2
  grid $w.macromolLab -row 2 -column 0 -sticky e
  grid $w.macromolEnt -row 2 -column 1 -sticky w -columnspan 2
  grid $w.atomLab -row 3 -column 0 -sticky e
  grid $w.atomEnt -row 3 -column 1 -sticky w -columnspan 2
  
  grid $w.segLab -row 9 -column 0 -sticky w
  grid $w.segEnt -row 9 -column 0 -sticky e
  grid $w.numLab -row 9 -column 1 -sticky w
  grid $w.numEnt -row 9 -column 1 -sticky s
  grid $w.typeLab -row 9 -column 1 -sticky e
  grid $w.typeEnt -row 9 -column 2 -sticky e
  grid $w.submitBut -row 17 -column 1 -sticky e

}


proc tt_GUI_tk {} {
    # If already initialized, just deiconify and return
  if { [winfo exists ".tt_GUI"] } {
    wm deiconify ".tt_GUI"
    raise ".tt_GUI"
    return
  }
  TruncateTrajectory::tt_GUI
  return $TruncateTrajectory::w
}


#tt_GUI_tk
