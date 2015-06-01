#
# $Id: fftk_distort.tcl,v 1.3 2014/02/11 17:15:05 mayne Exp $
#

namespace eval ::ForceFieldToolKit::Distortion:: {
   variable maxdepth 20;   # max depth to follow bond graphs
   variable maxringsize 20; # max ring size to be considered in ring finder
}

proc ::ForceFieldToolKit::Distortion::init {} {
   variable maxdepth 20
   variable maxringsize 20
}

proc ::ForceFieldToolKit::Distortion::make {mol type indexlist dx {frame -1} } {
   variable molid $mol
   variable maxdepth
   variable maxringsize

   # For each ring a list of atom indices describing the ring is returned. 
   # In case of nested rings (e.g. naphtalene) only the inner rings are kept.
##   variable ringlist [::util::find_rings $molid -maxringsize $maxringsize]
   variable ringlist [::util::find_rings $molid -all -maxringsize $maxringsize]

##   puts "Found [llength $ringlist] rings."

   if { $frame < 0 } {
      set frame [expr [molinfo $molid get numframes] - 1]
   }
##   puts "frame used: $frame"
##   set sel [atomselect $molid "all" frame $frame]
##   $sel writepdb testframe.pdb
##   $sel delete

   #if {[string match "bond" $type]} {
   #   distort_bond [lindex $indexlist 0] [lindex $indexlist 1] $dx $frame
   #} elseif {[string match "angle" $type]} {
   #   distort_angle [lindex $indexlist 0] [lindex $indexlist 1] [lindex $indexlist 2] $dx $frame
   #}
   if {[regexp {bond} $type]} {
      distort_bond [lindex $indexlist 0] [lindex $indexlist 1] $dx $frame
   } elseif {[regexp {angle|lbend} $type]} {
      distort_angle [lindex $indexlist 0] [lindex $indexlist 1] [lindex $indexlist 2] $dx $frame
   }
}

# Distort the bond given by $ind0 and $ind1 by $dx Angstrom in each direction.
# Two additional frames will be created containing the distorted structures.
proc ::ForceFieldToolKit::Distortion::distort_bond {ind0 ind1 dx frame} {
   if {$ind0==$ind1} {
      puts "distort_bond: The two atoms are identical!"
      return
   }

   variable maxdepth
   variable molid

   # Get bond vector direction
   set atom0 [atomselect $molid "index $ind0" frame $frame] 
   set atom1 [atomselect $molid "index $ind1" frame $frame]
   set pos0 [lindex [$atom0 get {x y z}] 0]
   set pos1 [lindex [$atom1 get {x y z}] 0]
   set bondvec [vecsub $pos0 $pos1]
   if {![veclength $bondvec]} {
      puts "distort_bond: The two atom positions are identical!"
      return
   }
   # gets distorted by $dx/2 in EACH direction
   set dx [expr $dx/2.0]
   set displace [vecscale $dx [vecnorm $bondvec]]

   # For each ring a list of atom indices describing the ring is returned. 
   # In case of nested rings (e.g. naphtalene) only the inner rings are kept.
   variable ringlist; # [::util::find_rings $molid]
   
 
   # FIXME: I'm just using the first ring in case the bond belongs to 2 rings.
   set inring [lindex [bond_in_ring $ringlist $ind0 $ind1] 0]

   set selL {}
   set selR {}

   # First handle the case where the bond is NOT part of a ring
   if {![llength $inring]} {
      # In order to move the two wings of the molecule independently
      # we are creating two selections:
      # First get all atoms in the bond graph left of the bond, 
      # i.e. following the graph in direction $ind0 $ind1.
      # The bond itself is included in the resulting list.
      # Then also get the atoms right of the bond.
      # 
      #  2       4--5
      #   \     /      LEFT  of 0--1: 0,1,2,3
      #	   0---1       RIGHT of 0--1: 0,1,4,5,6
      #   /     \ 
      #  3       6
      #     
      set indexesL [::util::bondedsel $molid $ind1 $ind0 -all]
      set indexesR [::util::bondedsel $molid $ind0 $ind1 -all]

      # Create selections of atoms left and right of the bond respectively
      set selL [atomselect $molid "index $indexesL and not index $ind1"]
      set selR [atomselect $molid "index $indexesR and not index $ind0"]      

   } else {
      # Bond is part of a ring.
      # We distort the bond by moving the bond atoms together with the
      # molecular graphs originating in their other bond partners.
      # Depending on the topology this will distort other
      # internal coordinates, too.
      set atomL [atomselect $molid "index $ind0"]
      set atomR [atomselect $molid "index $ind1"]
      set indexesL [follow_selected_bonds $molid $ind0 [lindex $ringlist $inring] -maxdepth $maxdepth]
      set indexesR [follow_selected_bonds $molid $ind1 [lindex $ringlist $inring] -maxdepth $maxdepth]
      set selL [atomselect $molid "index $ind0 $indexesL"]
      set selR [atomselect $molid "index $ind1 $indexesR"]

##      puts "Bond is part of ring $inring"
   }
   create_distorted_frames_bond $molid $frame $selL $selR $displace
}


# Distort the angle given by indices $ind0, $ind1, $ind2 by $dx degrees.
# Two additional frames will be created containing the distorted structures.
proc ::ForceFieldToolKit::Distortion::distort_angle {ind0 ind1 ind2 dx frame} {
   if {$ind0==$ind1 || $ind0==$ind2 || $ind1==$ind2} {
      puts "distort_angle: Two atoms are identical!"
      return
   }
   variable maxdepth
   variable molid
##   set frame [molinfo $molid get frame]

   # For each ring a list of atom indices describing the ring is returned. 
   # In case of nested rings (e.g. naphtalene) only the inner rings are kept.
   variable ringlist; #[::util::find_rings $molid]
   
 
   set inring [angle_in_ring $ringlist $ind0 $ind1 $ind2]

   set selL {}
   set selR {}

   if {![llength $inring]} {
      # In order to move the two wings of the molecule independently
      # we are creating two selections:
      # First get all atoms in the bond graph LEFT of the angle, 
      # i.e. following the graph in direction $ind2 $ind1 $ind0.
      # Of the atoms involved in the angle only $ind0 will be in the 
      # selection. Other atoms bound to the center atom $ind0 are
      # not selected and thus won't be moved.
      #    
      #         0--4
      #        /      LEFT  of angle 0-1-2: 0,4
      #	  3---1       RIGHT of angle 0-1-2: 2,5,6
      #        \ 
      #         2--5
      #        /
      #       6
      #
      set indexesL [::util::bondedsel $molid $ind1 $ind0 -all]
      set indexesR [::util::bondedsel $molid $ind1 $ind2 -all]
      set selL [atomselect $molid "index $indexesL and not index $ind1 $ind2"]
      set selR [atomselect $molid "index $indexesR and not index $ind1 $ind0"]
      
      create_distorted_frames_angle $molid $frame $selL $selR $ind0 $ind1 $ind2 $dx 

   } elseif {[llength $inring]} {


      if {[llength [bond_in_ring $ringlist $ind0 $ind1]]==2 ||
          [llength [bond_in_ring $ringlist $ind1 $ind2]]==2} {
         # One of the two bonds is shared by two rings.
         # We just move the middle atom in the angle plane to change the
         # angle. Of course the bond lengths will also change but we will live
         # with that.
##         puts "angle part of 2 rings"
   
         set selL [atomselect $molid "index $ind1"]
         set selR [atomselect $molid "none"]

      } else {

         # All three atoms of the angle are part of a single ring.
         # We follow the simplest approach by just moving the middle atom and
         # everything that's connected to it in the angle plane to change the
         # angle. Of course the bond lengths will also change but we will live
         # with that.
###         puts "Angle in ring"

         if {[llength $inring]>1} {
            # Angle is shared by multiple rings.
            # (Take a look at the twistane molecule for example.)
            # Since the section of all three atoms is shared by the rings it
            # is enough for the following to take only one of the into account.
       
            puts "Angle in multiple rings"
            set inring [lindex $inring 0]
         }
      
         # Find atoms bonded to the center that are not part of the ring.
         # Then get the atom indices from the molecular graphs originating in
         # these non-ring bonds.
         set indexes [follow_selected_bonds $molid $ind1 [lindex $ringlist $inring] -maxdepth $maxdepth]
         # We need only one selection to move the atoms
         set selL [atomselect $molid "index $indexes"]
         set selR [atomselect $molid "none"]
      
      }




#    elseif {[llength $inring]==1} 
#
#      # Find atoms bonded to the center that are not part of the ring.
#      # Then get the atom indices from the molecular graphs originating in 
#      # these non-ring bonds.
#      set indexes [follow_selected_bonds $molid $ind1 [lindex $ringlist $inring] -maxdepth $maxdepth]
#
#      # We need only one selection to move the atoms
#      set selL [atomselect $molid "index $indexes"]
#      set selR [atomselect $molid "none"]
    
      # Get coordinates of the three angle atoms
      set sel0 [atomselect top "index $ind0"]
      set pos0 [join [$sel0 get {x y z}]]
      set sel1 [atomselect top "index $ind1"]
      set pos1 [join [$sel1 get {x y z}]]
      set sel2 [atomselect top "index $ind2"]
      set pos2 [join [$sel2 get {x y z}]]

      # Displacement will be along the sum of the bond vectors
      set bonddir10 [vecnorm [vecsub $pos1 $pos0]] 
      set bonddir12 [vecnorm [vecsub $pos1 $pos2]]
      set displacevec  [vecnorm [vecadd $bonddir10 $bonddir12]]

      # Trying to set the amount of the linear displacement so that the
      # resulting change in angle is approximately $dx.
      # FIXME: Something is wrong here. I get displacements that are acceptable
      #        for practical purposes, but they don't seem to correspond to the desired
      #        change in angles..
      # I think it's fixed - also you can't assume that the two bond lengths are equal, 
      # I wasted a lot of time to derive the general case...
##      set angle [measure angle [list $ind0 $ind1 $ind2]]
##      set deg2rad [expr {3.14159265/180.0}]
##      set a [veclength [vecsub $pos1 $pos0]]
##      set b [veclength [vecsub $pos1 $pos2]]
##      set d [expr sqrt(pow($a,2)+pow($b,2) - 2*$a*$b*cos($deg2rad*$angle))]
##      set d2 [expr $b*$d/($a+$b)]
##      set theta [expr asin(($b/$d2)*sin($deg2rad*0.5*$angle))]
##      set h [expr ($d2/sin($deg2rad*0.5*$angle))*sin($deg2rad*0.5*$angle + $theta)]
##      set newh [expr ($d2/sin($deg2rad*0.5*($angle-$dx)))*sin($deg2rad*0.5*($angle-$dx) + $theta)]
##      set displacevec [vecscale $displacevec [expr {$h-$newh}]]
## ####      puts "new: $h $newh a: $a b: $b d: $d theta: $theta d2: $d2 angle: $angle dx: $dx"
##      # Since we are linearly displacing a selection we can use 
##      # the bond distortion code here:
## ###      puts "frame used: $frame for index $ind0 $ind1 $ind2"
##      create_distorted_frames_bond $molid $frame $selL $selR $displacevec


#### JAN'S NEW PROC, NEED TO TEST
      # Trying to set the amount of the linear displacement so that the
      # resulting change in angle is approximately $dx.
      # FIXME: The displacements lead to the correct change of the angles
      #        only if the two bond lengths are identical
      #        But for practical purposes this shouldn't matter.
      set b [veclength [vecsub $pos1 $pos0]]
      if {[veclength [vecsub $pos1 $pos2]]<$b} {
         set b [veclength [vecsub $pos1 $pos2]]
      }
      set angle [measure angle [list $ind0 $ind1 $ind2]]

      set deg2rad [expr {3.14159265/180.0}]
      set alpha [expr {$deg2rad*0.5*$angle}]
      set alphalarge [expr {$deg2rad*0.5*($angle+$dx)}]
      set alphasmall [expr {$deg2rad*0.5*($angle-$dx)}]

      set delta1 [expr {$b*cos($alpha)-$b*sin($alpha)/tan($alphalarge)}]
      set delta2 [expr {$b*cos($alpha)-$b*sin($alpha)/tan($alphasmall)}]

      set displace1 [vecscale $displacevec [expr { $delta1}]]
      set displace2 [vecscale $displacevec [expr {-$delta2}]]

      # Since we are linearly displacing a selection we can use
      # the bond distortion code here:
      create_distorted_frames_bond $molid $frame $selL $selR $displace1 $displace2



   } 

  # FIXME: We are missing the case where the angle is part of multiple rings!
}


# Append two frames to the molecule where selL and selR are shifted
# with respect to the current frame. In the first added frame the two
# selections are displaced symmetrically along the vecctor $displace
# so that their distance is decreased. In the second added frame the 
# distance is increased accordingly. 
proc ::ForceFieldToolKit::Distortion::create_distorted_frames_bond {molid frame selL selR displace {displace2 {}}} {

   set numframes [molinfo $molid get numframes]
   set last [expr {$numframes-1}]
##   set last $frame
##   set frame [molinfo $molid get frame]

   # In case only one displacement parameter is given use the same
   # value for both directions 
   if {![llength $displace2]} {
      set displace2 $displace
   }

   # Generate 2 new frames for f(x-h) and f(x+h)
   animate dup frame $frame $molid
   animate dup frame $frame $molid
   molinfo $molid set frame $frame
   
###  puts "bond frame used: $frame/ [molinfo $molid get numframes] for selL: [$selL text] selR: [$selR text]"

   #puts "L: [$selL list]"
   #puts "R: [$selR list]"
   #puts "Displace by [veclength $displace]"

   # First distortion makes bond shorter
   $selL frame [expr {$last+1}];
   $selR frame [expr {$last+1}];
   $selL moveby [vecinvert $displace]
   $selR moveby $displace
   
   # Second distortion stretches bond 
   # (the directions of the displacement are inverted now)
   $selL frame [expr {$last+2}]
   $selR frame [expr {$last+2}]
   $selL moveby $displace2
   $selR moveby [vecinvert $displace2]
}

# Append two frames to the molecule where $selL and $selR are displaced in
# such a way that the angle defined by atom indices $ind0, $ind1, $ind2 
# is changed wrt to the current frame. In the first additional frame the
# angle is decreased by delta in the second one it is increased by delta.
proc ::ForceFieldToolKit::Distortion::create_distorted_frames_angle {molid frame selL selR ind0 ind1 ind2 delta} {
   set numframes [molinfo $molid get numframes]
##   set last $frame
   set last [expr {$numframes-1}]
##   set frame [molinfo $molid get frame]

   # Generate 2 new frames for f(x-h) and f(x+h)
   animate dup frame $frame $molid
   animate dup frame $frame $molid
   molinfo $molid set frame $frame

###  puts "angle frame used: $frame/ [molinfo $molid get numframes] on selL: [$selL text] selR: [$selR text]"

   # Get coordinates of the three angle atoms
   set sel0 [atomselect top "index $ind0"]
   set pos0 [join [$sel0 get {x y z}]]
   set sel1 [atomselect top "index $ind1"]
   set pos1 [join [$sel1 get {x y z}]]
   set sel2 [atomselect top "index $ind2"]
   set pos2 [join [$sel2 get {x y z}]]

   # need to cut angle in half
   set delta [expr $delta/2.0]

   # First distortion makes angle smaller
   $selL frame [expr {$last+1}];
   $selR frame [expr {$last+1}];
   set mat [trans angle $pos2 $pos1 $pos0  $delta deg]
   $selL move $mat
   set mat [trans angle $pos2 $pos1 $pos0 -$delta deg]
   $selR move $mat
   
   # Second distortion makes angle bigger   
   $selL frame [expr {$last+2}]
   $selR frame [expr {$last+2}]
   set mat [trans angle $pos2 $pos1 $pos0 -$delta deg]
   $selL move $mat
   set mat [trans angle $pos2 $pos1 $pos0  $delta deg]
   $selR move $mat
}



# Check if the given atom is part of a ring where
# $ringlist is the result of ::util::find_rings.
proc atom_in_ring {ringlist index} {
   set found {}
   set i 0
   foreach ring $ringlist {
      set pos [lsearch $ring $index]
      if {$pos>=0} {
         lappend found $i
      }
      incr i
   }
   return $found
}

# Check if the given bond is part of a ring where
# $ringlist is the result of ::util::find_rings.
proc bond_in_ring {ringlist ind1 ind2} {
   set found {}
   set i 0
   foreach ring $ringlist {
      set pos1 [lsearch $ring $ind1]
      if {$pos1>=0} {
         set pos2 [lsearch $ring $ind2]
         if {$pos2>=0 && (abs($pos2-$pos1)==1 || abs($pos2-$pos1)==[llength $ring]-1)} {
            lappend found $i
         }
      }
      incr i
   }
   return $found
}


# Check if the given angle is part of a ring where
# $ringlist is the result of ::util::find_rings.
proc angle_in_ring {ringlist ind1 ind2 ind3} {
   set found {}
   set i 0
   foreach ring $ringlist {
      set pos2 [lsearch $ring $ind2]
      if {$pos2>=0} {
         set pos1 [lsearch $ring $ind1]
         if {$pos1>=0 && (abs($pos1-$pos2)==1 || abs($pos1-$pos2)==[llength $ring]-1)} {
            set pos3 [lsearch $ring $ind3]
            if {$pos3>=0 && (abs($pos3-$pos2)==1 || abs($pos3-$pos2)==[llength $ring]-1)} {
               lappend found $i
            }
         }
      }
      incr i
   }
   return $found
}

# Follow all bonds on atom $ind except the ones listed in $excludedbondatoms
# and return atom indices of the molecular graphs originating in the selected
# bonds. If one of these graphs leads back to the first atom $ind, then the
# atoms of this circular graph won't be appended to the resulting atom list.
# Useful e.g. to select the bond trees connected to a ring atom while ignoring
# the other ring atoms.
proc follow_selected_bonds {molid ind excludedbondatoms args} {
   set maxdepth [::util::getargs $args "-maxdepth" 20]

   set sel [atomselect $molid "index $ind"]
   set selectedbonds [::util::ldiff [join [$sel getbonds]] $excludedbondatoms]

   set indexes {}
   foreach bondatom $selectedbonds {
      lappend indexes [::util::bondedsel $molid $ind $bondatom -all -maxdepth $maxdepth]
   }

   # Remove paths that lead back to the starting atom
   set result $ind
   foreach path $indexes {
      if {[lsearch $path $ind]<0} {
         lappend result $path
      }
   }

   return [join $result]
}
   
      
## OLD PROC
## Follow all bonds on atom $ind except the ones listed in $excludedbondatoms 
## and return atom indices of the molecular graphs originating in the selected
## bonds.
## Useful e.g. to select the bond trees connected to a ring atom while ignoring 
## the other ring atoms.
#proc follow_selected_bonds {molid ind excludedbondatoms args} {
#   set maxdepth [::util::getargs $args "-maxdepth" 20]

#   set sel [atomselect $molid "index $ind"]
###   puts [$sel getbonds]
#   set selectedbonds [::util::ldiff [join [$sel getbonds]] $excludedbondatoms]
#   
#   set indexes $ind
#   foreach bondatom $selectedbonds {
#      lappend indexes [::util::bondedsel $molid $ind $bondatom -all -maxdepth $maxdepth]
#   }
#   
#   return [join $indexes]
#}

