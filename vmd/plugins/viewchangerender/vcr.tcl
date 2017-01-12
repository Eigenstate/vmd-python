# view_change_render.tcl
#
# $Id: vcr.tcl,v 1.24 2016/02/18 22:38:36 johns Exp $
#
# Johan Strumpfer, Barry Isralewitz, Jordi Cohen
# Oct 2003; updated Feb 2007; updated Mar 2012
# johanstr@ks.uiuc.edu
#
# A script to save current viewpoints and animate
# a smooth 'camera move' between them. Can also
# rendering each frame to a numbered .rgb file
#
# Usage:
# The following commands are provided in the ::VCR:: namespace:
#
# VCR viewpoint list manipulation:
#   write_vps filename              -  write all saved viewpoints to file
#   renum_vp n1 n2                  -  renumber viewpoint n1 -> n2
#   save_vp n                       -  save current viewpoint in position n   
#   remove_vp n                     -  remove viewpoint in position n
#   insert_vp                       -  insert the current viewpoint into 
#                                      position n, shifting positions of 
#                                      viewpoints >= n by 1 if needed.
#   clear_vps                       -  remove all stored viewpoints
#   load_vps filename               -  source the tcl script with saved
#                                      viewpoints                        
#   list_vps                        -  list all currently stored viewpoints
#   retrieve_vp n                   -  load viewpoint at position n
#   play_vp n1 n2                   -  play through the list of current
#                                      viewpoints starting at n1 and ending at
#                                      n2, loading each in turn
#   initialise_movevp n1 n2         -  initialisation routine called by move_vp
#                                      that sets up the interpolation from
#                                      viewpoint n1 to n2. 
#                                      required for move_vp_increment
#   move_vp_increment               -  move the camera and increment length
#                                      ::VCR::stepsize along path set up in
#                                      initialse_movevp
#   movetime_vp n1 n2 t             -  move the camera from viewpoint n1 to
#                                      viewpoint n2 in t seconds
#   move_vp n1 n2 N                 -  move the camera from viewpoint n1 to
#                                      viewpoint n2 in N steps
#
#   move_vp and movetime_vp notes:
#   - move_vp can take ad additional argument smooth or sharp, e.g.,
#     move_vp n1 n2 N smooth 
#          or 
#     move_vp n1 n2 N sharp
#
#   - Adding the smooth keyword uses a constant acceleration and then 
#     deceleration to move the camera from start to finish. 
#     Adding the keyword sharp uses a constant velocity to move 
#     the camera from start to finish.
#
#   - The position n1 or n2 in movetime_vp or move_vp can be given 
#     as "here" to move to/from the current camera position
#
# VCR movie sequence manipulation:
#   movie_clear                     - clear any existing movie viewpoint 
#                                     sequence and associated transition times
#
#   movie_add_vp_time n time        - add a viewpoint to the movie, with an
#                                     associated transition time in seconds.
#
#   movie_print_stats               - print information about the current
#                                     viewpoints, movie transitions,
#                                     and total movie duration

package require vmdmovie
# need util package for quaternions
package require utilities
package provide viewchangerender 1.8

namespace eval ::VCR:: {
    proc checkRepChangesTrace { } {
        set traces [trace info variable ::VCR::current_vp] 
        if { [llength $traces ] == 0 } {
            return 0
        } else {
            foreach t $traces {
                if { [lindex $t 1] == "::VCR::updaterepcallback" } {
                    return 1
                }
            }
            return 0
        }
    }
    
    proc enableRepChanges { } {
        variable repchanges
        if { [::VCR::checkRepChangesTrace] == 0 } {
            trace add variable ::VCR::current_vp write ::VCR::updaterepcallback
        }
        set repchanges 1
    }

    proc disableRepChanges { } {
        variable repchanges
        if { [::VCR::checkRepChangesTrace] == 1 } {
            set traces [trace info variable ::VCR::current_vp] 
            foreach t $traces {
                if { [lindex $t 1] == "::VCR::updaterepcallback" } {
                    trace remove variable ::VCR::current_vp write ::VCR::updaterepcallback
                }
            }
        }
        set repchanges 0
    }

    proc createMovieVars { {force 0} } {
        if { $force == 1 } {
          set ::VCR::movieTimeList {}
          set ::VCR::movieList {}
          set ::VCR::movieTime 0.0
          set ::VCR::MovieMakerStatus 0
          set ::VCR::movietimescale 1.0
          set ::VCR::movieDuration 0.0
        } else {
          if { ![info exists ::VCR::movieTimeList] } {
            set ::VCR::movieTimeList {}
          }
          if { ![info exists ::VCR::movieList] } {
            set ::VCR::movieList {}
          }
          if { ![info exists ::VCR::movieTime] } {
            set ::VCR::movieTime 0.0
          }
          if { ![info exists ::VCR::MovieMakerStatus] } {
            set ::VCR::MovieMakerStatus 0
          }
          if { ![info exists ::VCR::movietimescale] } {
            set ::VCR::movietimescale 1.0
          }
          if { ![info exists ::VCR::movieDuration] } {
            set ::VCR::movieDuration 0.0
          }
        }
        return
    }

    variable forcetrjgoto 0
    variable curtrjframe -1
    variable repchanges 0
    createMovieVars
    enableRepChanges
}

proc ::VCR::scale_mat {mat scaling} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr $scaling * [lindex [lindex [lindex $mat 0] $i] $j] ]
    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}


proc ::VCR::sub_mat {mat1 mat2} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr  (0.0 + [lindex [lindex [lindex $mat1 0] $i] $j]) - ( [lindex [lindex [lindex $mat2 0] $i] $j] )]

    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}


proc ::VCR::add_mat {mat1 mat2} {
  set bigger ""
  set outmat ""
  for {set i 0} {$i<=3} {incr i} {
    set r ""
    for {set j 0} {$j<=3} {incr j} {            
      lappend r  [expr  (0.0 + [lindex [lindex [lindex $mat1 0] $i] $j]) + [lindex [lindex [lindex $mat2 0] $i] $j] ]
    }
    lappend outmat  $r
  }
  lappend bigger $outmat
  return $bigger
}


proc ::VCR::matrix_to_euler {mat} {
  set pi 3.1415926535
  set R31 [lindex $mat 0 2 0]
  
  if {$R31 == 1} {
    set phi1 0.
    set psi1 [expr atan2([lindex $mat 0 0 1],[lindex $mat 0 0 2]) ]
    set theta1 [expr -$pi/2]
  } elseif {$R31 == -1} {
    set phi1 0.
    set psi1 [expr atan2([lindex $mat 0 0 1],[lindex $mat 0 0 2]) ]
    set theta1 [expr $pi/2]
  } else {
    set theta1 [expr -asin($R31)]
    # Alternate correct solution with a different trajectory:
    # set theta1 [expr $pi + asin($R31)]
    set cosT [expr cos($theta1)]
    set psi1 [expr  atan2([lindex $mat 0 2 1]/$cosT,[lindex $mat 0 2 2]/$cosT) ]
    set phi1 [expr  atan2([lindex $mat 0 1 0]/$cosT,[lindex $mat 0 0 0]/$cosT) ]
  }

  return "$theta1 $phi1 $psi1"
}


proc ::VCR::euler_to_matrix {euler} {
  set theta [lindex $euler 0]
  set phi [lindex $euler 1]
  set psi [lindex $euler 2]
    
  set mat {}
  lappend mat [list [expr cos($theta)*cos($phi)] [expr sin($psi)*sin($theta)*cos($phi) - cos($psi)*sin($phi)] [expr cos($psi)*sin($theta)*cos($phi) + sin($psi)*sin($phi)] 0. ]

  lappend mat [list [expr cos($theta)*sin($phi)] [expr sin($psi)*sin($theta)*sin($phi) + cos($psi)*cos($phi)] [expr cos($psi)*sin($theta)*sin($phi) - sin($psi)*cos($phi)] 0. ]
  
  lappend mat [list [expr -sin($theta)] [expr sin($psi)*cos($theta)] [expr cos($psi)*cos($theta)] 0. ]
    
  lappend mat [list 0. 0. 0. 1. ]
       
  return [list $mat]
}


proc ::VCR::write_vps {filename} {
  variable viewpoints
  variable representations
  set myfile [open $filename w]
  puts $myfile "\#This file contains viewpoints for the VMD viewchangerender plugin."
  puts $myfile "\#Type 'source $filename' from the VMD command window to load these viewpoints.\n"
  puts $myfile "proc viewchangerender_restore_my_state {} {"
  puts $myfile "  variable ::VCR::viewpoints\n"
  
  foreach v [array names viewpoints] {
    if [string equal -length 5 $v "here,"] {continue}
    puts $myfile "  set ::VCR::viewpoints($v) { $viewpoints($v) }"
  }

  foreach r [array names representations] {
    if [string equal -length 5 $r "here,"] {continue}
    puts $myfile "  set ::VCR::representations($r) \[list $representations($r) \]"
  }

  if { [info exists ::VCR::movieList] } {
     puts $myfile "  set ::VCR::movieList \"$::VCR::movieList\""
     puts $myfile "  set ::VCR::movieTimeList \"$::VCR::movieTimeList\""
     puts $myfile "  set ::VCR::movieTime $::VCR::movieTime"
  }
  if { [info exists ::VCR::movieDuration] } {
     puts $myfile "  set ::VCR::movieDuration $::VCR::movieDuration"
     puts $myfile "  ::VCR::calctimescale 0"
  }
  puts $myfile "  global PrevScreenSize"
  puts $myfile "  set PrevScreenSize \[display get size\]"
  puts $myfile "  proc RestoreScreenSize \{\} \{ global PrevScreenSize; display resize \[lindex \$PrevScreenSize 0\] \[lindex \$PrevScreenSize 1\] \}"
  puts $myfile "  display resize [display get size]"
  puts $myfile "  if { \[parallel noderank\] == 0 } {"
  puts $myfile "    puts \"Loaded viewchangerender viewpoints file $filename \""
  puts $myfile "    puts \"Note: The screen size has been changed to that stored in the viewpoints file.\""
  puts $myfile "    puts \"To restore it to its previous size type this into the Tcl console:\\n  RestoreScreenSize\""
  puts $myfile "  }"
  puts $myfile "  return"
  puts $myfile "}\n\n"
  puts $myfile "viewchangerender_restore_my_state\n\n" 
  close $myfile
}


proc ::VCR::renum_vp {view_num viewnumNew} {
  variable viewpoints
  if { ([info exists viewpoints($view_num,0)]) && !([info exists viewpoints($viewnumNew,0)]) } { 
    set viewpoints($viewnumNew,0) $viewpoints($view_num,0)
    set viewpoints($viewnumNew,1) $viewpoints($view_num,1)
    set viewpoints($viewnumNew,2) $viewpoints($view_num,2)
    set viewpoints($viewnumNew,3) $viewpoints($view_num,3)
    set viewpoints($viewnumNew,4) $viewpoints($view_num,4)
    ::VCR::renum_molrepstate $view_num $viewnumNew
    ::VCR::remove_vp $view_num
   }
} 


proc ::VCR::save_vp {view_num {mol top}} {
  variable viewpoints
  #if { !([molinfo $mol get drawn]) } {
  #    error "Molecule $mol is not drawn. Please specify currently drawn molecule with\n   save_vp view_num molid"
  #}
  if [info exists viewpoints($view_num,0)] { unset viewpoints($view_num,0) }
  if [info exists viewpoints($view_num,1)] { unset viewpoints($view_num,1) }
  if [info exists viewpoints($view_num,2)] { unset viewpoints($view_num,2) }
  if [info exists viewpoints($view_num,3)] { unset viewpoints($view_num,3) }
  if [info exists viewpoints($view_num,4)] { unset viewpoints($view_num,4) }
  set viewpoints($view_num,0) [molinfo $mol get rotate_matrix]
  set viewpoints($view_num,1) [molinfo $mol get center_matrix]
  set viewpoints($view_num,2) [molinfo $mol get scale_matrix]
  set viewpoints($view_num,3) [molinfo $mol get global_matrix]

  # range test frame, since we can get a -1 from molecules that
  # have no trajectory data (e.g. volumetric only)
  set curframenum [molinfo $mol get frame]
  if { $curframenum < 0 } {
    set curframenum 0
  }
  set viewpoints($view_num,4) $curframenum

  ::VCR::save_molrepstate $view_num

} 


proc ::VCR::remove_vp {view_num} {
  variable viewpoints
  if [info exists viewpoints($view_num,0)] { unset viewpoints($view_num,0) }
  if [info exists viewpoints($view_num,1)] { unset viewpoints($view_num,1) }
  if [info exists viewpoints($view_num,2)] { unset viewpoints($view_num,2) }
  if [info exists viewpoints($view_num,3)] { unset viewpoints($view_num,3) }
  if [info exists viewpoints($view_num,4)] { unset viewpoints($view_num,4) }
  ::VCR::remove_molrepstate $view_num
}


proc ::VCR::insert_vp {view_num {mol top}} {
    variable viewpoints
    if [info exists viewpoints($view_num,0)] {
        set vp [expr $view_num + 1]
        while { [info exists viewpoints($vp,0)] } {
            incr vp
        }
        while { $vp > $view_num} {
            ::VCR::renum_vp [expr $vp-1] $vp
            #set viewpoints($vp,0) $viewpoints([expr $vp-1],0)
            #set viewpoints($vp,1) $viewpoints([expr $vp-1],1)
            #set viewpoints($vp,2) $viewpoints([expr $vp-1],2)
            #set viewpoints($vp,3) $viewpoints([expr $vp-1],3)
            #set viewpoints($vp,4) $viewpoints([expr $vp-1],4)
            incr vp -1
        }
    }
    ::VCR::remove_vp $view_num
    ::VCR::save_vp $view_num $mol
}


proc ::VCR::clear_vps {} {
    variable viewpoints
    set listed {}
    foreach v [array names viewpoints] {
        unset viewpoints($v)
    }
    variable representations
    foreach r [array names representations] {
        unset representations($r)
    }
}


proc ::VCR::load_vps { fname } {
    variable viewpoints
    source $fname
}


proc ::VCR::list_vps {} {
    variable viewpoints
    set listed {}
    foreach v [array names viewpoints] {
        set v0 [lindex [split $v ","] 0]
        if {[lsearch -exact $listed $v0] == -1 && $v0 != "here"} {
            lappend listed $v0
        }
    }
    return [lsort -integer $listed]
}


proc ::VCR::play_vp { first last {morph_frames 50} args} { 
    variable viewpoints
    puts "VCR) first: $first    last: $last"
    if { !([info exists viewpoints($first,0)]) } {
        puts "VCR) Starting view $first was not saved" 
    }
    if { !([info exists viewpoints($last,0)]) } {
        puts "VCR) Ending view $last was not saved" 
    }
    if {!([info exists viewpoints($first,0)] && [info exists viewpoints($last,0)])} {
        error "VCR) play_vp failed, don't have both start and end viewpoints"
    }

    set inc 1
    if { $first > $last } {set inc -1}
    ::VCR::retrieve_vp $first
    set cur $first
    while { $cur != $last } {
        set next [expr $cur+$inc]
        set allthere 1
        if { !([info exists viewpoints($next,0)]) } {
                set allthere 0
        }
        if { $allthere == 1 } {
            ::VCR::move_vp $cur $next $morph_frames sharp
            set cur $next
        } else {
            puts "VCR) Viewpoint $next does not seem to exist, moving in to next one in list."
        }
    }
}

#### Move the one step further


proc ::VCR::initialise_movevp { start end args } {
  variable viewpoints 
  variable smooth
  variable tumble
  variable ninja
  variable render
  variable move
  variable movenorm
  variable stepsize
  variable beginvp
  variable finalvp 
  variable framestep
  variable tracking 1

  set pi 3.1415926535
  
  set smooth 0
  set tumble 0
  set ninja  0
  set render 0
  if {[lsearch $args "smooth"] > -1}  {set smooth 1} 
  if {[lsearch $args "sharp"] > -1 }  {set smooth 0} ;#default
#  if {[lsearch $args "tumble"] > -1}  {set tumble 1}
  if {[lsearch $args "ninja"] > -1 }  {set ninja 1}
  if {[lsearch $args "-render"] > -1} {set render 1}  ;# only for use by move_vp_render
  

#  if {$render} {set framenum $::VCR::first_frame_num}
  
  if {$start == "here" || $end == "here"} {save_vp "here"}

  # Make sure that we aren't trying to access something that doesn't exist            
    if { !([info exists viewpoints($start,0)]) } {
    error "Starting view $start was not saved" 
  }

  if { !([info exists viewpoints($end,0)]) } {
    error "Ending view $end was not saved" 
  }
  set beginvp {}
  set finalvp {}
  for {set i 0} {$i < 5} {incr i} {
    lappend beginvp $viewpoints($start,$i)
    lappend finalvp $viewpoints($end,$i)
  }
  set begin_euler [::VCR::matrix_to_euler [lindex $beginvp 0] ]
  set final_euler [::VCR::matrix_to_euler [lindex $finalvp 0] ]
  # Make sure to take the quickest path!
  set diff [vecsub $final_euler $begin_euler]
  for {set i 0} {$i < 3} {incr i} {
    if  {[lindex $diff $i] > $pi} {
      set final_euler [lreplace $final_euler $i $i [expr [lindex $final_euler $i] -2.*$pi]]
    } elseif {[lindex $diff $i] < [expr -$pi]} {
      set final_euler [lreplace $final_euler $i $i [expr 2.*$pi + [lindex $final_euler $i]]]
    }
  }
  # Check done
  set framestep 1
  # ninja rotates the camera the long way around
  if {$ninja} {
    set final_euler [lreplace $final_euler 2 2 [expr 2.*$pi + [lindex $final_euler 2]]]
  }
  set rotate_diff [vecsub $final_euler $begin_euler]
  set center_diff  [::VCR::sub_mat [lindex $finalvp 1]   [lindex $beginvp 1]]
  set scale_diff   [::VCR::sub_mat [lindex $finalvp 2]   [lindex $beginvp 2]]
  set global_diff  [::VCR::sub_mat [lindex $finalvp 3]   [lindex $beginvp 3]]
  set finalframe   [lindex $finalvp 4]
  set beginframe   [lindex $beginvp 4]
  set frame_diff   [expr $finalframe - $beginframe]
  set ::VCR::finalframe $finalframe
  set move {}
  set movenorm 0
  lappend move $rotate_diff
  foreach e $rotate_diff {
    set movenorm [expr $movenorm + abs($e)]
  }
  lappend move $center_diff
  for {set i 0} { $i < 4} {incr i} {
    foreach e [lindex [lindex $center_diff 0] $i] {
        set movenorm [expr $movenorm + abs($e)]
    }
  }
  lappend move $scale_diff
  for {set i 0} { $i < 4} {incr i} {
     foreach e [lindex [lindex $scale_diff 0] $i] {
        set movenorm [expr $movenorm + abs($e)]
    }
  }
  lappend move $global_diff
  for {set i 0} { $i < 4} {incr i} {
    foreach e [lindex [lindex $global_diff 0] $i] {
        set movenorm [expr $movenorm + abs($e)]
    }
  }
  lappend move $frame_diff
  set movenorm [expr $movenorm + abs($frame_diff)]
  set stepsize 0

  if { $::VCR::repchanges == 1 } {
    set repmove [::VCR::init_repmove $start $end]
    if {[llength $repmove] > 0} {
        lappend move $repmove
    }
  }

  #puts "llength move = [llength $move]"
}


proc ::VCR::retrieve_vp {view_num} {
  variable forcetrjgoto
  variable curtrjframe
  variable viewpoints
  variable current_vp 
  foreach mol [molinfo list] {
    if [info exists viewpoints($view_num,0)] {
      molinfo $mol set rotate_matrix   $viewpoints($view_num,0)
      molinfo $mol set center_matrix   $viewpoints($view_num,1)
      molinfo $mol set scale_matrix    $viewpoints($view_num,2)
      molinfo $mol set global_matrix   $viewpoints($view_num,3)

      set newtrjframe [expr int($viewpoints($view_num,4))]

      # range test frame, since we can get a -1 from molecules that
      # have no trajectory data (e.g. volumetric only)
      if { $newtrjframe < 0 } {
        set newtrjframe 0 
      }

      # only do a 'goto' if we absolutely have to
      if { $forcetrjgoto || ($curtrjframe != $newtrjframe) } {
        animate goto $newtrjframe
        set curtrjframe $newtrjframe
      }  
    } else {
      puts "VCR) View $view_num was not saved"
    }
  }
  restore_molrepstate $view_num
}

proc ::VCR::movetime_vp_wait {start runtime } {
     ::VCR::movetime_vp $start $start $runtime 1
}

# MOVE FROM VIEWPOINT start TO end IN runtime SECONDS
# NOTE: SMOOTH ACCELERATE AND DECELERATE NOT IMPLEMENTED
# looks to be accurate to about 0.01 seconds
proc ::VCR::movetime_vp {start end runtime {forcewait 0} } {
   variable current_vp
   set ::VCR::abort 0
   if {($start != $end)} {
       variable tracking
      ::VCR::initialise_movevp $start $end
      if { ($::VCR::movenorm < 0.05) && ($forcewait == 0) && [llength $::VCR::move] == 5 } {
            ::VCR::delrepchanges
            if { $::VCR::repchanges == 1} {
                ::VCR::restore_molrepstate $end
            }
            return
      }
      set t [time { ::VCR::retrieve_vp $start }]
      set t2 [time {set runtime [expr $runtime*1000000]}]
      set t2 [expr [lindex [split $t2] 0]+0.0]
      set Ttot $runtime
      set tracking 1.0
      set j 1
      set spf 0.0
      while { $tracking > 0 && $::VCR::abort == 0 } {
            set t [expr [lindex [split $t] 0] + $t2*10]
            set runtime [expr 0.0 + $runtime - $t]
            set spf [expr ($spf*($j-1)+$t)/$j]  
            set ::VCR::stepsize [expr $tracking/$runtime*$spf]
            set diff [expr $tracking - $::VCR::stepsize]
            if { $diff < 0 || $::VCR::stepsize < 0} {
                set ::VCR::stepsize $tracking
                set tracking 0
            } else {
                set tracking $diff
            }
            incr j
            set t [time { ::VCR::move_vp_increment }]
            display update ui
       }
      if {$start == "here" || $end == "here"} {remove_vp "here"}
 
    } elseif { ($forcewait != 0 ) } {
         set t [time { ::VCR::restore_molrepstate $end } ]
         set t [expr [lindex [split $t] 0]] 
         set t [expr $runtime - ($t/1000000)]
         set ::VCR::stepsize [expr 100./($t*1000)]
         set tracking 1.0
         while { $tracking > 0 && $::VCR::abort == 0 } {
            #puts "VCR) Sleeping $t"
            after [expr int(100)]
            set tracking [expr $tracking-$::VCR::stepsize]
            display update ui
         }
    }

    ::VCR::delrepchanges
    set ::VCR::abort 0
    if { $::VCR::repchanges == 1} {
        ::VCR::restore_molrepstate $end
    }

}

proc ::VCR::move_vp {start end {morph_frames -1} {framestep 1} args} {
      ::VCR::initialise_movevp $start $end $args
      variable current_vp
      set ::VCR::framestep $framestep
      if {$morph_frames == -1 && $framestep == 1} {
        set morph_frames [lindex $::VCR::move 4]
      } elseif {$morph_frames == -1} {
        set morph_frames 50
      }
      if { $morph_frames == 0 } {
        set morph_frames 50
      }

      set ::VCR::stepsize [expr 1.0/$morph_frames]
     
      if {$start != "here"} {
            puts "VCR) Going to first viewpoint"
            ::VCR::retrieve_vp $start
      }
      set tracking 1
      set j 0
      
      while { $tracking > 0 } {
          #set scaling to apply for this individual frame
          if {$::VCR::smooth} {
            #accelerate smoothly to start and stop 
            set theta [expr 3.1415927*(1.0 +$j)/($morph_frames+1)] 
            set ::VCR::stepsize [expr (1. - cos($theta))*0.5-(1-$tracking)]
            if { $::VCR::stepsize == 0 } { set ::VCR::stepsize 0.0001 }
            #puts "   $theta $VCR::stepsize $tracking"
          }

          if { $tracking < $::VCR::stepsize || $::VCR::stepsize < 0 } { 
               set ::VCR::stepsize [expr $tracking] 
               set tracking 0
          } else {
            set tracking [expr $tracking - $::VCR::stepsize]
          }
          ::VCR::move_vp_increment
          incr j
              
    #      RENDER FUNCTIONALITY DISABLED - THIS WILL BE INCORPORATED VIA MOVIEMAKER
    #      if {$::VCR::render} {
    #        set frametext [format "%06d" $framenum]
    #        render snapshot [file join $::VCR::dirName $::VCR::filePrefixName.$frametext.rgb]  
    #        puts "Rendering frame [file join $::VCR::dirName $::VCR::filePrefixName.$frametext.rgb]"
    #        incr framenum
    #      }
      }


    ::VCR::delrepchanges
    if {$start == "here" || $end == "here"} {remove_vp "here"}
    set current_vp $end
}


proc ::VCR::init_move_vp_Movie { vplist frameskip {framestep 1} args} {
  set start [lindex $vplist 0]
  set end [lindex $vplist 1]
  set ::VCR::movieList $vplist
  ::VCR::initialise_movevp $start $end $args
  set ::VCR::movieListPos 1
  set nVP [llength $vplist]
  set totFrames 0
  for { set n 1 } { $n < $nVP } {incr n} {
    set m1 [lindex $vplist [expr $n-1]]
    set m2 [lindex $vplist $n]
    set totFrames [expr $totFrames + abs($::VCR::viewpoints($m2,4)-$::VCR::viewpoints($m1,4))]
 }

  set ::VCR::framestep $framestep
  set morph_frames 1
  if { [lindex $::VCR::move 4] != 0 } {
    set morph_frames [expr double(abs([lindex $::VCR::move 4]))/$frameskip]
  }

  set ::VCR::stepsize [expr 1.0/$morph_frames]
 
  if {$start != "here"} {
        puts "VCR) Going to first viewpoint"
        ::VCR::retrieve_vp $start
  }
  set ::VCR::tracking 1
  set ::VCR::CurMovieMakerFrame 0
  return $totFrames
}

  
# PROC WILL MOVE THE VIEWPOINT as a fraction $::VCR::stepsize along the
# direction $::VCR::move
proc QSdensity { r } {
    if { $r > 3.0 } {
        set r [expr $r-3.0]
        set d [expr 30*$r+6*pow($r,2)+10.*pow($r,0.5)+23.0]
    } elseif { $r > 1.0 } {
        set d [expr 254.83*exp(-1.018*$r)-5071*pow($r,-0.0189)+4980.]
    } else {
        set d $r
    }
    return $d
}

proc ::VCR::move_vp_increment {} {
    variable forcetrjgoto
    variable curtrjframe
    set topmol [molinfo top]
    variable beginvp
    variable finalvp
    foreach mol [molinfo list] {
      set random {}
      if {$::VCR::tumble} {
         set randscale 0.1
      } else {
         set randscale 0.0
      }
      lappend random [expr  $randscale*rand()]
      lappend random [expr  $randscale*rand()]
      lappend random [expr  $randscale*rand()]
      set ::VCR::current {}
      lappend ::VCR::current [molinfo $mol get rotate_matrix]
      lappend ::VCR::current [molinfo $mol get center_matrix] 
      lappend ::VCR::current [molinfo $mol get scale_matrix ]
      lappend ::VCR::current [molinfo $mol get global_matrix]
      lappend ::VCR::current [molinfo $mol get frame]
      # range test frame, since we can get a -1 from molecules that
      # have no trajectory data (e.g. volumetric only)
      set curframenum [molinfo $mol get frame]
      if { $curframenum < 0 } {
        set curframenum 0
      }
      lappend ::VCR::current $curframenum

      #changed to use quaternions
      #set euler [matrix_to_euler [lindex $::VCR::current 0]]
      for {set i 0} {$i < 4} {incr i} {
          if {$i == 0} {      
             #changed to use quaternions
             #set euler [vecadd [vecadd $euler [vecscale $::VCR::stepsize [lindex $::VCR::move 0]]] $random]
             #lset ::VCR::current 0 [euler_to_matrix $euler ]
             lset ::VCR::current 0 [list [::util::quatinterpmatrices [lindex $beginvp 0 0] [lindex $finalvp 0 0] [expr {1.0 - $::VCR::tracking}]]]
          } else {
             lset ::VCR::current $i [add_mat [lindex $::VCR::current $i] [scale_mat [lindex $::VCR::move $i] $::VCR::stepsize]]
          }    
      }
      
      molinfo $mol set rotate_matrix [lindex $::VCR::current 0]
      molinfo $mol set center_matrix [lindex $::VCR::current 1]
      molinfo $mol set scale_matrix  [lindex $::VCR::current 2]
      molinfo $mol set global_matrix [lindex $::VCR::current 3]
      if {$::VCR::framestep == 1 && $mol == $topmol} {
        set f [expr $::VCR::finalframe - ([lindex $::VCR::move 4] * $::VCR::tracking) ]

        set newtrjframe [expr $f]

        # range test frame, since we can get a -1 from molecules that
        # have no trajectory data (e.g. volumetric only)
        if { $newtrjframe < 0 } {
          set newtrjframe 0
        }

        # only do a 'goto' if we absolutely have to
        if { $forcetrjgoto || ($curtrjframe != $newtrjframe) } {
          animate goto $newtrjframe
          set curtrjframe $newtrjframe
        }  

        if { $::VCR::repchanges == 1 && [llength $::VCR::move] > 5 } {
            set repmove [lindex $::VCR::move 5]
            foreach r $repmove {
                set molid    [lindex $r 0]
                set repid    [lindex $r 1]
                set r0 [lreplace [lindex [molinfo $molid get "{rep $repid}"] 0] 0 0]
                set repscale [lindex $r 3]
                set dr [vecscale $::VCR::stepsize [lindex $repscale]]
                set r1 [vecadd $r0 $dr]
                set rad  [lindex $r1 0]
                set dens [QSdensity $rad]
                set grid [lindex $r1 2]
                set qual [lindex $r1 3]
                    
                mol modstyle $repid $molid QuickSurf $rad $dens $grid $qual
            }
        }
      }
    }
    display update 
}

proc ::VCR::rephash { molid repid } {
    set h ""
    set i $repid
    foreach {r s c m} [molinfo $molid get "{rep $i} {selection $i} {color $i} {material $i}"] { break }
    append h "${r}"
    append h "-${s}"
    append h "-${c}"
    append h "-${m}"
    return [string map { " " "_" } $h]
}

proc ::VCR::shortrephash { molid repid } {
    set h ""
    set i $repid
    foreach {s c m} [molinfo $molid get "{selection $i} {color $i} {material $i}"] { break }
    append h "${s}"
    append h "-${c}"
    append h "-${m}"
    return [string map { " " "_" } $h]
}

proc ::VCR::convRepHashToShortRepHash { rh } {

    set a [split $rh "-"]
    set h [lindex $a 1]
    append h "-[lindex $a 2]"
    append h "-[lindex $a 3]"
    return $h
}

proc ::VCR::init_repmove { start end } {
    #puts "init_repmove: $start $end"
    variable representations
    set reptransitions [list]
    foreach molid [molinfo list] {
        set name [molinfo $molid get name]
        if { [info exists representations($start,$name)] && 
                                [info exists representations($end,$name)] } {
            set StartRepStates $representations($start,$name)            
            set EndRepStates $representations($end,$name)            
            set ShortEndRepStates [list]
            foreach e $EndRepStates {
                lappend ShortEndRepStates [convRepHashToShortRepHash $e]
            }
            #puts "$ShortEndRepStates"
            for {set r 0} { $r < [molinfo $molid get numreps] } {incr r} {
                set repname [lindex [lindex [molinfo $molid get "{rep $r}"] 0] 0]
                if { $repname == "QuickSurf" } {
                    set rh [::VCR::rephash $molid $r]
                    if { [lsearch -ascii -exact $StartRepStates $rh] != -1 } {
                       set srh [::VCR::shortrephash $molid $r] 
                        if {[lsearch -ascii -exact $ShortEndRepStates $srh] != -1 } {
                            set n [lsearch -ascii -exact $ShortEndRepStates $srh]
                            #puts "Match found!:"
                            #puts "$rh\n[lindex $EndRepStates $n]"
                            set startrep [lindex [molinfo $molid get "{rep $r}"] 0]
                            set startQSparams [lreplace $startrep 0 0]
                            set endrep [lindex $EndRepStates $n]
                            set endrep [string map { "_" " " } [lindex [split $endrep "-"] 0]]
                            set endQSparams [lreplace $endrep 0 0]
                            #puts "$startQSparams ->\n$endQSparams"
                            set diff [vecsub $endQSparams $startQSparams]
                            if { [ veclength $diff ] > 0.01 } {
                                lappend reptransitions [list $molid $r $startQSparams $diff]
                            }
                        }
                    }
                }
            }
        }
    }
    #puts $reptransitions
    set reptrans [list]
    set topmol [molinfo top]
    foreach rep $reptransitions {
        set m [lindex $rep 0]
        set nr [molinfo $m get numreps]
        set r [lindex $rep 1]
        set params [lindex $rep 2]
        set diff [lindex $rep 3]

        foreach {s c mat} [molinfo $m get "{selection $r} {color $r} {material $r}"] { break }
        mol showrep $m $r off
        mol representation QuickSurf $params
        mol selection $s
        mol color $c
        mol material $mat
        mol addrep $m
        mol showrep $m $nr on
        lappend reptrans [list $m $nr $params $diff]
    }
    #return $reptransitions
    return $reptrans
}

proc ::VCR::delrepchanges { } {
    if { $::VCR::repchanges == 1 && [info exists ::VCR::move] && [llength $::VCR::move] > 5 } {
        set repmove [lindex $::VCR::move 5]
        set newr [llength $repmove]
        foreach r $repmove {
            set molid    [lindex $r 0]
            set nr [expr [molinfo $molid get numreps] -1 ]
            mol delrep $nr $molid
            #puts "Delete rep $nr from mol $molid"
        }
    }
}

proc ::VCR::save_molrepstate { id } {
    variable representations
    foreach molid [molinfo list]  {
        set name [molinfo $molid get name]
        set repstates [list]
        set rephashes [list]
        if { [molinfo $molid get displayed] == 1 } {
            for {set r 0} { $r < [molinfo $molid get numreps] } {incr r} {
                if { [mol showrep $molid $r] == 1 } {
                    lappend rephashes [::VCR::rephash $molid $r]
                }
            }
            set representations($id,$name) $rephashes
        }
    }

}

proc ::VCR::remove_molrepstate { id } {
    variable representations
    foreach molid [molinfo list]  {
    set name [molinfo $molid get name] 
        if {[info exists representations($id,$name)]} {
            unset representations($id,$name) 
        }
    }
}

proc ::VCR::renum_molrepstate { old_id new_id } {
  variable representations
  foreach molid [molinfo list]  {
    set name [molinfo $molid get name]
    if { ([info exists representations($old_id,$name)]) && !([info exists representations($new_id,$name)]) } { 
        set representations($new_id,$name) $representations($old_id,$name) 
        ::VCR::remove_molrepstate $old_id
    }
  }
}


proc ::VCR::restore_molrepstate { id } {
    variable representations
    display update off
    if {$id != "here"} {
        foreach molid [molinfo list]  {
            set name [molinfo $molid get name]
            if [info exists representations($id,$name)] {
                mol on $molid
                #make sure top molecule is displayed
                #for VCR
                #mol top $molid
                set repstates $representations($id,$name)
                for {set r 0} { $r < [molinfo $molid get numreps] } {incr r} {
                    set rh [::VCR::rephash $molid $r]
                    if { [lsearch -ascii -exact $repstates $rh] != -1 } {
                        mol showrep $molid $r 1
                    } else {
                        mol showrep $molid $r 0
                    }
                }
            } else {
                mol off $molid
            }
        }
   }
    display update
    display update on
}


proc ::VCR::toggleRepChanges { } {
    variable repchanges
    if { [::VCR::checkRepChangesTrace] == 0 } {
        trace add variable ::VCR::current_vp write ::VCR::updaterepcallback
        set repchanges 1
    } else {
        set traces [trace info variable ::VCR::current_vp] 
        foreach t $traces {
            if { [lindex $t 1] == "::VCR::updaterepcallback" } {
                trace remove variable ::VCR::current_vp write ::VCR::updaterepcallback
            }
        }
        set repchanges 0
    }
}

proc ::VCR::checkRepChangesTrace { } {
    set traces [trace info variable ::VCR::current_vp] 
    if { [llength $traces ] == 0 } {
        return 0
    } else {
        foreach t $traces {
            if { [lindex $t 1] == "::VCR::updaterepcallback" } {
                return 1
            }
        }
        return 0
    }
}
    
proc ::VCR::updaterepcallback { args } {
    ::VCR::restore_molrepstate $::VCR::current_vp
}

proc ::VCR::updateAllRepStates { args } {
    variable representations
   
    foreach id [::VCR::list_vps] {
        foreach molid [molinfo list] {
            set name [molinfo $molid get name]
            ###ASSUME REPS HAVE ONLY BEEN ADDED
            set curshowrep [lindex $representations($id,$name) 0]
            set curhashes [lindex $representations($id,$name) 2]
            set rold 0
            set newshowrep [list]
            set newhashes [list]
            for {set r 0} { $r < [molinfo $molid get numreps] } {incr r} {
                set rh1 ::VCR::rephash $molid $r
                #stored hash
                set rh2 [lindex $curhashes $rold]
                if { [string compare $rh1 $rh2] != 0 } {
                    lappend newshowrep 0
                } else {
                    lappend newshowrep [lindex $curshowrep $rold] 
                    lappend newhashes $rh2
                    incr rold
                }
            }
        }
    }
}

proc ::VCR::calctimescale { args } {
    if { $::VCR::MovieMakerStatus == 1 && $::MovieMaker::movietype == "userdefined" } {
        if {$::VCR::movieTime > 0 && $::MovieMaker::framerate > 0 } {
            set ::VCR::movietimescale [expr  ($::MovieMaker::numframes/$::MovieMaker::framerate)/$::VCR::movieTime]
        }
        ::VCR::updateMovieTime
    } elseif { $::VCR::MovieMakerStatus == 1 } {
        ::VCR::updateMovieMakerDuration
    } elseif { $::VCR::MovieMakerStatus == 0 } {
        if { $::VCR::movieDuration > 0 && $::VCR::movieTime > 0 } {
            set ::VCR::movietimescale [expr  ($::VCR::movieDuration)/$::VCR::movieTime]        
        } else {
            set ::VCR::movietimescale 1.0
        }
    }
}

proc ::VCR::updateMovieMakerDuration { args } {
    ::MovieMaker::durationChanged [expr $::VCR::movieDuration]
    set ::MovieMaker::movieduration [expr $::VCR::movieDuration]
}

proc ::VCR::restoretimescale { } {
    set ::VCR::movietimescale 1.0 
    ::VCR::updateMovieTime
    trace remove variable ::MovieMaker::numframes write ::VCR::calctimescale
}

proc ::VCR::updateMovieTime {} {
    ::VCR::createMovieVars
    set movieTime 0.0
    set ::VCR::movieDuration 0.0
    set mind 0
    foreach m [::VCR::getmovieList] t [::VCR::getmovieTimeList] {
     if { $mind < [expr [llength $::VCR::movieList]-1] } {
        set ::VCR::movieDuration [expr $::VCR::movieDuration + $t*$::VCR::movietimescale]
        set movieTime [expr $movieTime +  $t]
      }
      incr mind
    }
    set ::VCR::movieDuration [format "%-6.2f" $::VCR::movieDuration]
    ::VCR::setmovieTime $movieTime
}

proc ::VCR::enableMovieMaker { {parallel 0}  } {
    ::VCR::disableMovieMaker
    if { [llength $::VCR::movieList] > 1 } {
      set ::VCR::MovieMakerStatus 1
      if { $parallel == 0 } {
        trace add variable ::MovieMaker::userframe write ::VCR::moviecallback
        #puts "VCR =---= MovieMaker connected"
      } else {
        trace add variable ::MovieMaker::userframe write ::VCR::parallelmoviecallback
        #puts "VCR =---= parallel MovieMaker connected"
      }
      set ::VCR::originalMovieMakerTime $::MovieMaker::movieduration   
      ::VCR::updateMovieMakerDuration 
      trace add variable ::MovieMaker::numframes write ::VCR::calctimescale
    } else {
       puts "VCR) Error: You need 2 or more viewpoints in movie list to enable movie maker."
    } 
}

proc ::VCR::disableMovieMaker {} {
    set ::VCR::MovieMakerStatus 0
    set tracelist [trace info variable ::MovieMaker::userframe]
    foreach t $tracelist {
        if { ![string compare $t "write ::VCR::moviecallback"] } {
          trace remove variable ::MovieMaker::userframe write ::VCR::moviecallback
          #puts "VCR =---= MovieMaker disconnected"
        } elseif { ![string compare $t "write ::VCR::parallelmoviecallback"] } {
          trace remove variable ::MovieMaker::userframe write ::VCR::parallelmoviecallback
          #puts "VCR =---= MovieMaker disconnected"
        }

    }
    trace remove variable ::MovieMaker::numframes write ::VCR::calctimescale
    if { [info exists ::VCR::originalMovieMakerTime] } {
        ::MovieMaker::durationChanged [expr $::VCR::originalMovieMakerTime]
        set ::MovieMaker::movieduration [expr $::VCR::originalMovieMakerTime]
    }
}

proc ::VCR::toggleMovieMaker { {parallel 0}  } {
    if { $::VCR::MovieMakerStatus == 0 } {
        ::VCR::enableMovieMaker $parallel
    } elseif { $::VCR::MovieMakerStatus == 1 } {
        ::VCR::disableMovieMaker
    }
}

proc ::VCR::writeRenderFile {} {
    set out test.tcl
    puts $out "source LH2Demo.vmd"
    puts $out "load_vps LH2DemoStates.tcl"
    puts $out "if { \$::VCR::MovieMakerStatus != 1 } {"
    puts $out "    ::VCR::enableMovieMaker"
    puts $out "}"
    puts $out "::VCR::updateMovieTime"
    puts $out "::MovieMakes::makemovie \"./frames/\" test targaframes Snapshot \$::VCR::movieDuration userdefined"
    close $out
}

proc ::VCR::writeRenderComponentsFile { outputname workingdirectory basename renderer } {
    save_state $outputname.vmd
    ::VCR::write_vps "$outputname.vcr"
    set out [open "$outputname.tcl" "a"]
    puts $out "source vcr.tcl"
    puts $out "source $outputname.vmd"
    puts $out "source $outputname.vcr"
 
    set wd $workingdirectory
    set bname $basename
    set nl [llength $::VCR::movieList]
    set VP0 [lindex $::VCR::movieList 0]
    for {set n 1} {$n < $nl } {incr n} {
        set VP1 [lindex $::VCR::movieList $n]
        set t   [lindex $::VCR::movieTimeList [expr $n-1]]
        set ProcNum [format "Render%05d" $n]
        set DirName [file join $wd [format "%05d" $n]]
        set duration [format "%6.3f" [expr $t*$::VCR::movietimescale]]
        puts $out "\nproc $ProcNum {} {"
        puts $out "  set ::VCR::movieList \[list $VP0 $VP1\]"
        puts $out "  set ::VCR::movieTimeList \[list $t\]"
        puts $out "  ::VCR::updateMovieTime"
        puts $out "  puts \"Movie time: \$::VCR::movieTime. Duration \$::VCR::movieDuration\""
        puts $out "  file mkdir $DirName"
        puts $out "  set ::MovieMaker::movieformat targaframes"
        puts $out "  ::MovieMaker::makemovie $DirName $bname targaframes $renderer $duration userdefined"
        puts $out "}\n"
        set VP0 $VP1
    }
    #puts $out "source LH2Demo.vmd"
    puts $out "set ::VCR::movieDuration $::VCR::movieDuration"
    puts $out ""
    puts $out "if { \$::VCR::MovieMakerStatus != 1 } {"
    puts $out "    ::VCR::enableMovieMaker"
    puts $out "}"
    puts $out ""
    puts $out "display ambientocclusion on"
    puts $out "display aombient 0.7"
    puts $out "display aodirect 0.3"
    puts $out "for {set n 1} { \$n < $nl } { incr n } {"
    puts $out "     set ProcNum \[format \"Render%05d\" \$n\]"
    puts $out "     eval \$ProcNum"
    puts $out "}"
    close $out
}

proc ::VCR::writeParallelRenderFile { outputname workingdirectory basename renderer } {
    save_state $outputname.vmd
    ::VCR::write_vps "$outputname.vcr"
    set out [open "$outputname.tcl" "w"]
    puts $out "play parallelRenderCmd.tcl"
    puts $out "play vcr.tcl"
    puts $out "play $outputname.vmd"
    puts $out "play $outputname.vcr"
    puts $out "axes location off"

    puts $out "set ::MovieMaker::framerate 24"
    puts $out "set ::MovieMaker::trjstep 1"
    puts $out "set ::MovieMaker::userframe 0"    
    puts $out "::VCR::enableMovieMaker 1"
    puts $out "set ::VCR::movieDuration $::VCR::movieDuration"
    puts $out "::VCR::updateMovieMakerDuration"
    puts $out "set ::MovieMaker::numframes \[expr \$::VCR::movieDuration*\$::MovieMaker::framerate\]"
    puts $out "parallel barrier"
    puts $out "render_vcr_movie $workingdirectory \"$basename.%05d.tga\" $::MovieMaker::numframes $renderer"
    close $out
}


proc ::VCR::movieabortcallback { args } {
    ::VCR::delrepchanges
    ::VCR::retrieve_vp "movieinit"
    ::VCR::remove_vp "movieinit"
    trace remove variable ::MovieMaker::abort write ::VCR::movieabortcallback
}

proc ::VCR::moviemakerDone_callback { args } {
    if { $::MovieMaker::statusstep > 2 } {
        ::VCR::movieabortcallback
    }
    trace remove variable ::MovieMaker::statusstep write ::VCR::moviemakerDone_callback
}

proc ::VCR::moviecallback { args } {
  if {$::MovieMaker::userframe == 0 } { 
        set ::VCR::tracking 0
        set ::VCR::movieListPos 0
        set totFrames 0
        set nVP [llength $::VCR::movieList]

        for { set n 1 } { $n < $nVP } {incr n} {
            set m1 [lindex $::VCR::movieList [expr $n-1]]
            set m2 [lindex $::VCR::movieList $n]
            set totFrames [expr $totFrames + abs($::VCR::viewpoints($m2,4)-$::VCR::viewpoints($m1,4))]

        }
        set ::VCR::totFrames $totFrames
        set ::VCR::framestep $::MovieMaker::trjstep
        set ::VCR::CurMovieMakerFrame -1
  }
  if {$::MovieMaker::userframe < $::MovieMaker::numframes && $::MovieMaker::userframe > $::VCR::CurMovieMakerFrame} {
      if {$::VCR::tracking == 0} {
            if { $::VCR::movieListPos == 0 } { 
                if { [parallel noderank] == 0 } {
                  puts "VCR) Generating frames with View-Change-Render."
                  puts "VCR) Movie Maker settings:"
                  puts "VCR) \tframerate=$::MovieMaker::framerate fps"
                  puts "VCR) \tduration=$::MovieMaker::movieduration sec" 
                  puts "VCR) \tframeskip=$::MovieMaker::trjstep"
                  puts "VCR) \ttotal animation frames=$::MovieMaker::numframes"
                  puts "VCR) View-Change-Render settings:"
                  puts "VCR) \ttotal trajectory frames=$::VCR::totFrames"
                  puts "VCR) \ttotal time=[expr $::VCR::movieTime*$::VCR::movietimescale]"
                  puts "VCR) \tRetrieving first viewpoint"
                }

                trace add variable ::MovieMaker::abort write ::VCR::movieabortcallback
                trace add variable ::MovieMaker::statusstep write ::VCR::moviemakerDone_callback
                ::VCR::save_vp "movieinit"
                ::VCR::retrieve_vp [lindex $::VCR::movieList 0]
                ::VCR::restore_molrepstate [lindex $::VCR::movieList 0]
            }
            set start [lindex $::VCR::movieList $::VCR::movieListPos]
            set T [lindex $::VCR::movieTimeList $::VCR::movieListPos]
            incr ::VCR::movieListPos
            if { $::VCR::movieListPos < [llength $::VCR::movieList] } {
                set end [lindex $::VCR::movieList $::VCR::movieListPos]
                puts -nonewline "VCR) Reached viewpoint $start proceeding to viewpoint $end"
                if {[expr $::VCR::movieListPos -1] > 0} {
                    if { $::VCR::repchanges == 1 && [llength $::VCR::move] > 5 } {
                        set repmove [lindex $::VCR::move 5]
                        foreach r [lsort -index 1  -integer $repmove] {
                            set molid    [lindex $r 0]
                            set repid    [lindex $r 1]
                            mol delrep  $repid $molid
                        }
                    }
                }
                ::VCR::restore_molrepstate $start
                ::VCR::initialise_movevp $start $end 
                set morph_frames [expr $T/$::VCR::movieTime]
                set morph_frames [expr floor($morph_frames*$::MovieMaker::numframes)]
                set ::VCR::stepsize [expr 1.0/$morph_frames]
                set ::VCR::tracking 1
                puts " in ${morph_frames} frames."
            } else {
                set ::MovieMaker::userframe $::MovieMaker::numframes
            }
      }
      if {$::VCR::tracking > 0} {
          incr ::VCR::CurMovieMakerFrame 
          if { $::VCR::tracking < $::VCR::stepsize || $::VCR::stepsize < 0 } { 
               set ::VCR::stepsize [expr $::VCR::tracking] 
               set ::VCR::tracking 0
          } else {
            set ::VCR::tracking [expr $::VCR::tracking - $::VCR::stepsize]
          }
          ::VCR::move_vp_increment
       }
   } 
}

 

proc ::VCR::init_parallelmovie { } {
        if { [parallel noderank] == 0 } {
          puts "VCR) Initialising pVCR movie"
        }
        set ::VCR::tracking 0
        set ::VCR::movieListPos 0
        set totFrames 0
        set nVP [llength $::VCR::movieList]
        set ::VCR::movieStart_vp  [list]
        set ::VCR::movieEnd_vp    [list]
        set ::VCR::movieFrameStep [list]
        set frames 0
        set totaltime 0
        for { set n 1 } { $n < $nVP } {incr n} {
            set m1 [lindex $::VCR::movieList [expr $n-1]]
            set m2 [lindex $::VCR::movieList $n]
            set totFrames [expr $totFrames + abs($::VCR::viewpoints($m2,4)-$::VCR::viewpoints($m1,4))]
            set T [lindex $::VCR::movieTimeList [expr $n-1]]
            set morph_frames [expr $T/$::VCR::movieTime]
            set morph_time   [expr $morph_frames*$::MovieMaker::movieduration]
            set morph_frames [expr floor($morph_frames*$::MovieMaker::numframes)]
            set frames [expr $frames+$morph_frames]
            set totaltime [expr $totaltime+$morph_time]
            for { set i 1 } { $i <= $morph_frames } { incr i } {
                lappend ::VCR::movieStart_vp $m1
                lappend ::VCR::movieEnd_vp $m2
                lappend ::VCR::movieFrameStep [expr $i/$morph_frames]
            }
        }
        if { [parallel noderank] == 0 } {
          puts "VCR) Total Frames = $frames   Total Time = $totaltime"
        }
        for { set i 0 } { $i < $::MovieMaker::numframes } { incr i } {
            set fno [format "%04d" $i]
            set svp [lindex $::VCR::movieStart_vp $i]
            set evp [lindex $::VCR::movieEnd_vp $i]
            set vpstep [lindex $::VCR::movieFrameStep $i]
            #puts "VCR) \[$fno\] startvp=$svp endvp=$evp step=$vpstep"
        }
        set ::VCR::totFrames $totFrames
        set ::VCR::framestep $::MovieMaker::trjstep

        if { [parallel noderank] == 0 } {
          puts "VCR) Generating frames with View-Change-Render in parallel."
          puts "VCR) Movie Maker settings:"
          puts "VCR) \tframerate=$::MovieMaker::framerate fps"
          puts "VCR) \tduration=$::MovieMaker::movieduration sec" 
          puts "VCR) \tframeskip=$::MovieMaker::trjstep"
          puts "VCR) \ttotal animation frames=$::MovieMaker::numframes"
          puts "VCR) View-Change-Render settings:"
          puts "VCR) \ttotal trajectory frames=$::VCR::totFrames"
          puts "VCR) \ttotal time=[expr $::VCR::movieTime*$::VCR::movietimescale]"
          puts "VCR) \tllength of start_vps = [llength $::VCR::movieStart_vp]"
          puts "VCR) \tRetrieving first viewpoint"
        }

        trace add variable ::MovieMaker::abort write ::VCR::movieabortcallback
        trace add variable ::MovieMaker::statusstep write ::VCR::moviemakerDone_callback
        #::VCR::save_vp "movieinit"
        set ::VCR::parallelmovieinitialised 1
}


proc ::VCR::parallelmoviecallback { args } {
  if { ![info exists ::VCR::parallelmovieinitialised]  } { 
        ::VCR::init_parallelmovie
  }
  if {$::MovieMaker::userframe < [llength $::VCR::movieStart_vp] } {
      set startvp [lindex $::VCR::movieStart_vp $::MovieMaker::userframe]
      set finalvp [lindex $::VCR::movieEnd_vp   $::MovieMaker::userframe]
      ::VCR::retrieve_vp $startvp
      if { $::VCR::repchanges == 1 } {
         ::VCR::delrepchanges
         ::VCR::restore_molrepstate $startvp
      }
      ::VCR::initialise_movevp $startvp $finalvp
      if {$::MovieMaker::userframe > 0 } {
        set ::VCR::stepsize [lindex $::VCR::movieFrameStep $::MovieMaker::userframe]
        set ::VCR::tracking [expr 1.0-$::VCR::stepsize]
        set tracking [format "%5.4f" $::VCR::tracking]
        #puts "VCR) \[$::MovieMaker::userframe\] In transition $startvp->$finalvp $tracking"

        ## XXX this is where special user-defined callbacks should go...
        # drawRCCofactorlights $startvp $tracking

        ::VCR::move_vp_increment
     }
  }
  if { $::MovieMaker::userframe >= [expr $::MovieMaker::numframes - 1] } {
    if { [info exists ::VCR::parallelmovieinitialized] } {
        unset ::VCR::parallelmovieinitialized
    }
  }
}


 
proc ::VCR::setmovieTime { val } {
    set ::VCR::movieTime $val
}

proc ::VCR::getmovieTimeList {} {
    return $::VCR::movieTimeList
}

proc ::VCR::getmovieList {} {
    return $::VCR::movieList
}


##### VCR hacks for script-based movie generation ####
# XXX

proc ::VCR::movie_clear { } {
  set ::VCR::movieList {}
  set ::VCR::movieTimeList {}
  set ::VCR::movieTime 0.0
}


proc ::VCR::movie_add_vp_time { m time } {
  # if there's an existing movie list, update the transition time
  # from the previous frame, otherwise just append
  set mf $::VCR::viewpoints($m,4)
  if { [llength $::VCR::movieList] > 0 } {
    set a [expr [llength $::VCR::movieList]-1]

    # this code tries to guess the transition time from trajectory frame diff
    # but not needed since it is provided by the user
#    set n [lindex $::VCR::movieList $a]
#    set nf $::VCR::viewpoints($n,4)
#    set diff [expr abs($mf-$nf)]
#    if { $diff == 0 } { set diff 12 }
#    ##UPDATE MovieTimeList $n (next to last) with Appropriate Transition to $m
#    set t [expr double($diff)/$::MovieMaker::framerate]
    set t [expr double($time)]
    lset ::VCR::movieTimeList $a $t
  }
  lappend ::VCR::movieTimeList 0.0
  lappend ::VCR::movieList $m

  # update total movie time
  ::VCR::updateMovieTime

  # assume movie maker will be used...
#  if {  $::vcr_gui::MovieMakerStatus == "*** enabled ***" } {
    ::VCR::updateMovieMakerDuration
#  }
}


proc ::VCR::movie_print_stats {} {
  if { [parallel noderank] == 0 } {
    set nvps [llength [::VCR::list_vps]]
    set ntrans [llength $::VCR::movieList]

    puts "VCR) movie stats:"
    puts "VCR)   viewpoints: $nvps"
    puts "VCR)   movie transitions: $ntrans"
    puts "VCR)   total movie duration: $::VCR::movieTime sec"
  }
}


##### End VCR hacks for script-based movie generation ####


proc ::VCR::goto_prev { {movetime 0} } {
    #puts "VCR) Goto_prev!"
    if {![info exists ::VCR::current_vplist_pos]} {
        set next_vp [expr [llength [list_vps]] -1 ]
    } else {
        set next_vp [expr $::VCR::current_vplist_pos - 1]
        if { $next_vp < 0 } {
            set next_vp [expr [llength [list_vps]] -1 ]
        }
    }
    set ::VCR::current_vplist_pos $next_vp
    set vp [lindex [list_vps] $next_vp]
    #puts "VCR) $next_vp: $vp"
    if {$movetime == 0} {
        ::VCR::retrieve_vp $vp   
    } else {
        ::VCR::movetime_vp here $vp $movetime
    }
}



proc ::VCR::goto_next { {movetime 0} } {
    #puts "VCR) Goto_next!"
    if {![info exists ::VCR::current_vplist_pos]} {
        set next_vp 0
    } else {
        set next_vp [expr $::VCR::current_vplist_pos + 1]
        if { $next_vp >= [llength [list_vps]] } {
            set next_vp 0
        }
    }
    set ::VCR::current_vplist_pos $next_vp
    set vp [lindex [list_vps] $next_vp]
    #puts "VCR) $next_vp: $vp"
    if {$movetime == 0} {
        ::VCR::retrieve_vp $vp   
    } else {
        ::VCR::movetime_vp here $vp $movetime
    }
}

proc ::VCR::set_keyboard_keys {} {
    user add key h { ::VCR::goto_prev }
    user add key j { ::VCR::goto_prev 0.5 }
    user add key k { ::VCR::goto_next 0.5 }
    user add key l { ::VCR::goto_next }
}   

proc ::VCR::set_Aux_keys {} {
    user add key Aux-0 { ::VCR::goto_prev }
    user add key Aux-1 { ::VCR::goto_prev 0.5 }
    user add key Aux-2 { ::VCR::goto_next 0.5 }
    user add key Aux-3 { ::VCR::goto_next }
}   
