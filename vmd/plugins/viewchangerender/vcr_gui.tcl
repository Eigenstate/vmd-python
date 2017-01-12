# view_change_render.tcl
#
# $Id: vcr_gui.tcl,v 1.16 2016/02/18 22:38:36 johns Exp $
#
# Johan Strumpfer, Barry Isralewitz, Jordi Cohen
# Oct 2003; updated Feb 2007; updated Mar 2012
# johanstr@ks.uiuc.edu
#

##
## TODO:
##   The "renumber" viewpoint feature is broken in some cases, and 
##   it can result in a viewpoint that contains some garbage.
##

package require vmdmovie
package require viewchangerender
package provide viewchangerender_gui 1.8

namespace eval ::vcr_gui:: {
    package require viewchangerender
    package require vmdmovie

    proc fillmlist {} {
        set mind 0
        set movieTime 0.0
        set ::vcr_gui::duration 0.0
        ::VCR::updateMovieTime
        ::VCR::createMovieVars 
        foreach m [::VCR::getmovieList] t [::VCR::getmovieTimeList] {
            if {[string is integer $m]} {
                if { $mind < [expr [llength $::VCR::movieList]-1] } {
                    if { $mind < [llength $::vcr_gui::mlist] } {
                        lset ::vcr_gui::mlist $mind [format "% 5d   % 5.2f" $m [expr $t*$::VCR::movietimescale]]
                    } else {
                        lappend ::vcr_gui::mlist [format "% 5d   % 5.2f" $m [expr $t*$::VCR::movietimescale]]
                    }
                } else {
                    if { $mind < [llength $::vcr_gui::mlist] } {
                        lset ::vcr_gui::mlist $mind [format "% 5d         " $m ]
                    } else {
                        lappend ::vcr_gui::mlist [format "% 5d         " $m ]
                    }                   
                }
                incr mind
            }
        }
    }


    proc resetGUI { } {
       if { ![info exists ::vcr_gui::selectedtransitiontime] } {
            set ::vcr_gui::selectedtransitiontime " "
        }

        if { ![info exists ::vcr_gui::mlist] } {
            set ::vcr_gui::mlist {}
        }
        if { ![info exists ::vcr_gui::timescale] } {
           set ::vcr_gui::timescale 1.0
        }
        if { ![info exists ::vcr_gui::MovieMakerStatus] } {      
           set ::vcr_gui::MovieMakerStatus "disabled"
        }
        if { ![info exists ::vcr_gui::totaltimeedit] } {      
           set ::vcr_gui::totaltimeedit 0
        }
        ::VCR::createMovieVars 
        variable selectcolor lightsteelblue
        if { ![info exists ::vcr_gui::timescale] } {
            variable timescale 1.0
        }
        variable duration 0.0
        variable movetime 0.5
        set ::vcr_gui::vplist [::VCR::list_vps]
        ::vcr_gui::fillmlist
        trace remove variable ::MovieMaker::userframe write ::VCR::moviecallback
        for { set i 0 } { $i < [llength [trace info variable ::MovieMaker::numframes]] } { incr i } {
            trace remove variable ::MovieMaker::numframes write ::vcr_gui::calctimescale
        }
    }
    resetGUI

}

proc vcr_gui { } { return [eval ::vcr_gui::vcr_gui] }



proc ::vcr_gui::vcr_gui { } {
   variable vplist
   variable mlist
   variable MovieMakerStatus
   variable selectcolor 
   variable timescale 
   variable duration 
   variable movetime
   variable w
   ::vcr_gui::fillmlist
   #if { $::vcr_gui::forceredraw } {
   #   destroy .vcr_gui
   #}
   if { [winfo exists .vcr_gui ] } {
            wm deiconify .vcr_gui
            focus .vcr_gui
            return
    } 
    set w [toplevel .vcr_gui]
    wm title $w "View-Change-Render"
    wm resizable $w 0 0
   
    ### menubar 
    frame $w.menubar -relief raised -bd 2
    menubutton $w.menubar.file -text "File" -underline 0 -menu $w.menubar.file.menu
    $w.menubar.file config -width 3 
    menu $w.menubar.file.menu -tearoff no 
    $w.menubar.file.menu add command -label "New"  -command ::vcr_gui::new_viewpointlist
    $w.menubar.file.menu add command -label "Save" -command ::vcr_gui::save_viewpointlist 
    $w.menubar.file.menu add command -label "Load" -command ::vcr_gui::load_viewpointlist 
    $w.menubar.file.menu add command -label "Exit" -command { destroy .vcr_gui }

    menubutton $w.menubar.settings -text "Settings" -underline 0 -menu $w.menubar.settings.menu
    $w.menubar.settings config -width 7
    menu $w.menubar.settings.menu -tearoff no 
    $w.menubar.settings.menu add command -label "Disable representation changes"  -command ::vcr_gui::togglerepchanges
    
    if { $::VCR::repchanges == 0 } {
        $w.menubar.settings entryconfigure 0 -label "Enable representation changes"
    }
#    $w.menubar.file.menu add command -label "Save" -command ::vcr_gui::save_viewpointlist 

   pack $w.menubar.file -side left 
   pack $w.menubar.settings -side left
   grid $w.menubar -sticky ew -columnspan 3 -row 0 -column 0 -padx 0

   ### frame that containing the list of viewpoints
   labelframe $w.vp -bd 2 -relief ridge -text "Viewpoint #" -padx 1m -pady 1m   
   frame $w.vp.list
   scrollbar $w.vp.list.scroll -command "$w.vp.list.list yview" -takefocus 0
   listbox $w.vp.list.list -activestyle dotbox -yscroll "$w.vp.list.scroll set" -font {tkFixed 10} -width 6 -height 10 -setgrid 1 -selectmode browse -selectbackground $selectcolor -listvariable ::vcr_gui::vplist

   pack $w.vp.list.list -side left -fill both -expand 1
   pack $w.vp.list.scroll -side left -fill y
   pack $w.vp.list -padx 1m -pady 1m -fill y -expand 1


   ### frame that contains the commands for moving between viewpoints
   frame $w.cmd
   button $w.cmd.retrieve -text "Retrieve Viewpoint" -command ::vcr_gui::retr_vp
   button $w.cmd.moveto   -text "Move To Viewpoint" -command ::vcr_gui::moveto_vp
   label  $w.cmd.movetimelabel -text "     Move in: "
   entry  $w.cmd.movetime -textvariable ::vcr_gui::movetime -width 5
   label  $w.cmd.movetimelabelpst -text "secs" 


   pack $w.cmd.retrieve -fill x
   pack $w.cmd.moveto -fill x 
   pack $w.cmd.movetimelabel $w.cmd.movetime  -side left 
   pack $w.cmd.movetimelabelpst -side top  -fill y  
 
   ### frame to add viewpoint to movie list
   frame $w.moviecmd
   button $w.moviecmd.addtomovie -text "Add Viewpoint to Movie" -command ::vcr_gui::addtomovie
   button $w.moviecmd.addalltomovie -text "Add All Viewpoints to Movie" -command ::vcr_gui::addalltomovie
   pack $w.moviecmd.addtomovie 
   pack $w.moviecmd.addalltomovie


   ### frame to edit viewpoints
   frame $w.edit
   button $w.edit.save     -text "Save Viewpoint" -command ::vcr_gui::save_view
   button $w.edit.replace     -text "Replace Viewpoint" -command ::vcr_gui::replace_vp
   button $w.edit.insert   -text "Insert Viewpoint" -command ::vcr_gui::insrt_vp
   button $w.edit.delete   -text "Delete Viewpoint" -command ::vcr_gui::delete_vp
   button $w.edit.renumber   -text "Renumber Viewpoint" -command ::vcr_gui::renumber
   pack $w.edit.save -fill x
   pack $w.edit.replace -fill x
   pack $w.edit.insert -fill x
   pack $w.edit.delete -fill x
   pack $w.edit.renumber -fill x


   ### meta-frame to hold all the movie list controls
   labelframe $w.moviemaker -bd 2 -relief ridge -text "Movie" -padx 1m -pady 1m   
   ## reordering & editing of the movie list
   frame $w.moviemaker.editing
   label $w.moviemaker.editing.labeltop -text "\n"
   button $w.moviemaker.editing.removefrommovie -text "Remove From Movie" -command ::vcr_gui::removefrommovie
   button $w.moviemaker.editing.moveup -text "Move Up" -command ::vcr_gui::moveup
   button $w.moviemaker.editing.movedown -text "Move Down" -command ::vcr_gui::movedown
   button $w.moviemaker.editing.clear -text "Clear list" -command ::vcr_gui::clearmlist
   #button $w.moviemaker.editing.writerenderscript -text "Write render script" 
   pack $w.moviemaker.editing.labeltop -side top
   pack $w.moviemaker.editing.removefrommovie -fill x -side top
   pack $w.moviemaker.editing.moveup -fill x -side top
   pack $w.moviemaker.editing.movedown -fill x -side top
   pack $w.moviemaker.editing.clear -fill x -side top
   #pack $w.moviemaker.editing.writerenderscript -side bottom -fill x
   
   ## the actual movie list showing the viewpoint number and lenght of time between transitions
   frame $w.moviemaker.list
   frame $w.moviemaker.list.listframe
   label $w.moviemaker.list.listtext -text "Movie List  Transition\n    (vp #)       Time (s)" -justify center
   listbox $w.moviemaker.list.listframe.list -activestyle dotbox -yscroll "$w.moviemaker.list.listframe.scroll set" -font {TkFixedFont 10} -width 15 -height 10 -setgrid 1 -selectmode browse -selectbackground $selectcolor -listvariable ::vcr_gui::mlist
   bind $w.moviemaker.list.listframe.list <Return> { ::vcr_gui::edit_time }
   bind $w.moviemaker.list.listframe.list <KP_Enter> { ::vcr_gui::edit_time }
    

   scrollbar $w.moviemaker.list.listframe.scroll -command "$w.moviemaker.list.listframe.list yview" -takefocus 0 -orient vertical
   label $w.moviemaker.list.timelabel -text "Total time(s): "
   if { !$::vcr_gui::totaltimeedit} {
      entry $w.moviemaker.list.time -textvariable ::VCR::movieDuration -width 5 -state disabled -disabledforeground black

   } else {
      entry $w.moviemaker.list.time -textvariable ::VCR::movieDuration -width 5 -state normal -bg white
   }
   bind $w.moviemaker.list.time <Return> { ::vcr_gui::settotaltime }
   bind $w.moviemaker.list.time <KP_Enter> { ::vcr_gui::settotaltime }

   pack $w.moviemaker.list.listtext  -expand 1 -anchor w 
   pack $w.moviemaker.list.listframe.list  -fill y -side left -padx 1
   pack $w.moviemaker.list.listframe.scroll -fill y -side left 
   pack $w.moviemaker.list.listframe
   pack $w.moviemaker.list.timelabel -side left
   pack $w.moviemaker.list.time
  
   ## frame containing the editing of transition timing and activation of the MovieMaker trace
   frame $w.moviemaker.timing
   label $w.moviemaker.timing.labeltop -text "MovieMaker"
   if { $vcr_gui::MovieMakerStatus == "disabled" } { 
    label $w.moviemaker.timing.labelstatus -textvariable ::vcr_gui::MovieMakerStatus -foreground red
    button $w.moviemaker.timing.switch -text "Enable" -command ::vcr_gui::toggleMovieMaker
   } else {
    label $w.moviemaker.timing.labelstatus -textvariable ::vcr_gui::MovieMakerStatus -foreground darkgreen
    button $w.moviemaker.timing.switch -text "Disable" -command ::vcr_gui::toggleMovieMaker
   } 
   button $w.moviemaker.timing.preview -text "Preview Movie" -command ::vcr_gui::preview
   button $w.moviemaker.timing.edittime -text "Change Transition Time" -command ::vcr_gui::edit_time -width 19
   entry $w.moviemaker.timing.transtime -textvariable ::vcr_gui::selectedtransitiontime -state disabled -relief flat -width 5
   bind $w.moviemaker.timing.transtime <Return> { ::vcr_gui::edit_time }
   bind $w.moviemaker.timing.transtime <KP_Enter> { ::vcr_gui::edit_time }

   if { !$::vcr_gui::totaltimeedit} {
       button $w.moviemaker.timing.settotaltime -text "Set Total Time" -command ::vcr_gui::settotaltime
   } else {
       button $w.moviemaker.timing.settotaltime -text "Save" -command ::vcr_gui::settotaltime
   }
   pack $w.moviemaker.timing.labeltop -side top
   pack $w.moviemaker.timing.labelstatus -pady 0 -side top
   pack $w.moviemaker.timing.switch -fill x -pady 5 -side top
   pack $w.moviemaker.timing.preview -fill x  -side top
   pack $w.moviemaker.timing.edittime -side top -fill x
   pack $w.moviemaker.timing.transtime -side top
   pack $w.moviemaker.timing.settotaltime -fill x -side bottom

   pack $w.moviemaker.editing -side left -fill y
   pack $w.moviemaker.list  -side left   -fill y
   pack $w.moviemaker.timing -side left  -fill y 


   grid $w.cmd -padx 0 -columnspan 1 -column 0 -row 1 -rowspan 1 -sticky ew
   grid $w.moviecmd -padx 0 -columnspan 1 -column 0 -row 2 -rowspan 1 -sticky ew
   grid $w.vp -padx 1 -columnspan 1 -column 1 -row 1 -rowspan 2 -sticky ew
   grid $w.edit -padx 0 -columnspan 1 -column 2 -row 1 -rowspan 2 -sticky w
   grid $w.moviemaker -pady 1 -columnspan 3 -column 0 -row 3 -rowspan 2 -sticky ns

   ::vcr_gui::toggle_settotaltimegui

   # check to see if someone has already loaded a viewpoints file
   # or populated VCR via scripts before the GUI got opened.
   set ::vcr_gui::vplist [::VCR::list_vps]
}

proc ::vcr_gui::scrollMovieList { a b } {
    .vcr_gui.moviemaker.list.listframe.list yview $a $b
}


proc ::vcr_gui::vp_distance { n m } {
    set l 0.0
    for {set i 0 } { $i < 1 } { incr i } {
        for { set j 0 } {$j < 4} {incr j} {
            set v1 [lindex [lindex $::VCR::viewpoints($n,$i) 0] $j]
            set v2 [lindex [lindex $::VCR::viewpoints($m,$i) 0] $j]
            puts "\( $v1 \)  - \( $v2 \)"
           set l [expr $l + [vecdist $v1 $v2] ]
        }
    }
    return $l
}

proc ::vcr_gui::retr_vp {} {
    set num [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    if { [llength $num] == 1 } {
        ::VCR::retrieve_vp $num
    } else {
       ::vcr_gui::selwarning
    }
}


proc ::vcr_gui::moveto_vp {} {
    set num [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    if { [llength $num ] == 1 } {
        if {$::vcr_gui::movetime == -1} {
            ::VCR::move_vp here $num 
        } else {
            ::VCR::movetime_vp here $num $::vcr_gui::movetime
        }
    } else {
        ::vcr_gui::selwarning
    }
}


proc ::vcr_gui::movetointime_vp {} {
    set num [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]] 
    if { [llength $num ] == 1 } {
        ::VCR::movetime_vp here $num $::vcr_gui::movetime
        set ::vcr_gui::vplist [::VCR::list_vps]
    } else {
        ::vcr_gui::selwarning
    }
}


proc ::vcr_gui::save_view {} {
    set num [expr [lindex $::vcr_gui::vplist [expr [llength $::vcr_gui::vplist]-1]]+1]
    ::VCR::save_vp $num
    ::VCR::save_molrepstate $num
    set ::vcr_gui::vplist [::VCR::list_vps]
}


proc ::vcr_gui::replace_vp {} {
    set num [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    if { [llength $num ] == 1 } {
        ::VCR::save_vp $num
#//       ::VCR::save_molrepstate $num
        set ::vcr_gui::vplist [::VCR::list_vps]
    } else {
        ::vcr_gui::selwarning
    }

}


proc ::vcr_gui::insrt_vp {} {
    set num [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    if { [llength $num ]== 1 } {
        ::VCR::insert_vp $num
        set ::vcr_gui::vplist [::VCR::list_vps]
    } else {
        ::vcr_gui::selwarning
    }
}

proc ::vcr_gui::selwarning {} {
    tk_messageBox -icon error -type ok -title "VCR Error" -message "You need to select a viewpoint."
}

proc ::vcr_gui::movieselwarning {} {
    tk_messageBox -icon error -type ok -title "VCR Error" -message "You need to select a viewpoint in the movie list."
}

proc ::vcr_gui::delete_vp {} {
    set num [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    if { [llength $num ] != 1 } {
       ::vcr_gui::selwarning
        return
    }
    ::VCR::remove_vp [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    set ::vcr_gui::vplist [::VCR::list_vps]
}


proc ::vcr_gui::renumber {} {
    set num [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    if { [llength $num ] != 1 } {
       ::vcr_gui::selwarning
        return
    }
    set w [toplevel .renumberer]
    set ::vcr_gui::n1 [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    wm title $w "Renumber Viewpoint"
    set ::vcr_gui::n2 ""
    frame $w.edit
    label  $w.edit.text -text "Enter new number of viewpoint $::vcr_gui::n1:"
    entry  $w.edit.replacenum -textvariable ::vcr_gui::n2 -width 5
    button $w.edit.ok     -text "Ok" -command { ::VCR::renum_vp $::vcr_gui::n1 $::vcr_gui::n2 ;   set ::vcr_gui::vplist [::VCR::list_vps]; unset ::vcr_gui::n2; destroy .renumberer }
    button $w.edit.cancel     -text "Cancel" -command { destroy .renumberer } 

    pack $w.edit.text    
    pack $w.edit.replacenum
    pack $w.edit.ok
    pack $w.edit.cancel    
    grid $w.edit -padx 1 -columnspan 1 -column 0 -row 0 -rowspan 2 -sticky ew
}


proc ::vcr_gui::new_viewpointlist {} {
    ::VCR::clear_vps
    set ::vcr_gui::vplist [::VCR::list_vps]
}


proc ::vcr_gui::save_viewpointlist {} {
   set filename [tk_getSaveFile -parent .vcr_gui -defaultextension ".tcl"]
   ::VCR::write_vps $filename
}


proc ::vcr_gui::loadviewpoints { file } {
   ::VCR::load_vps $file
   set ::vcr_gui::vplist [::VCR::list_vps]
   set ::vcr_gui::mlist {}
   ::vcr_gui::fillmlist
}

proc ::vcr_gui::load_viewpointlist {} {
   if { [llength $::vcr_gui::vplist] > 0 } {
        set answer [tk_messageBox -type yesno -title "VCR Warning" -message "You have viewpoints already loaded.\n By proceeding you will overwrite all\nviewpoints with the same id numbers as the file\nyou wish to load. Do you want to continue?"]
        if {$answer == "no" } { return}
    }
   set filename [tk_getOpenFile -parent .vcr_gui -defaultextension ".tcl"]
   if { $filename != "" } {
        ::VCR::load_vps $filename
   }
   set ::vcr_gui::vplist [::VCR::list_vps]
   set ::vcr_gui::mlist {}
   ::vcr_gui::fillmlist
}

proc ::vcr_gui::addalltomovie {} {
    for { set i 0 } { $i < [llength $::vcr_gui::vplist] } { incr i } {
        set m [lindex $::vcr_gui::vplist $i]
        set mf $::VCR::viewpoints($m,4)
        if { [llength $::VCR::movieList] > 0 } {
            set a [expr [llength $::VCR::movieList]-1]
            set n [lindex $::VCR::movieList $a]
            set nf $::VCR::viewpoints($n,4)
            set diff [expr abs($mf-$nf)]
            if { $diff == 0 } { set diff 12 }
        ##UPDATE MovieTimeList $n (next to last) with Appropriate Transition to $m
            set t [expr double($diff)/$::MovieMaker::framerate]
            lset ::VCR::movieTimeList $a $t
        }
        lappend ::VCR::movieTimeList 0.0
        lappend ::VCR::movieList $m
        lappend ::vcr_gui::mlist ""
        ::vcr_gui::fillmlist
        if {  $::vcr_gui::MovieMakerStatus == "*** enabled ***" } {
          ::VCR::updateMovieMakerDuration
#        ::vcr_gui::calctimescale
        }
    }
} 

proc ::vcr_gui::addtomovie {} {
    set m [lindex $::vcr_gui::vplist [.vcr_gui.vp.list.list curselection]]
    if { [llength $m] != 1 } {
        ::vcr_gui::selwarning
        return
    }
    set mf $::VCR::viewpoints($m,4)
    if { [llength $::VCR::movieList] > 0 } {
        set a [expr [llength $::VCR::movieList]-1]
        set n [lindex $::VCR::movieList $a]
        set nf $::VCR::viewpoints($n,4)
        set diff [expr abs($mf-$nf)]
        if { $diff == 0 } { set diff 12 }
        ##UPDATE MovieTimeList $n (next to last) with Appropriate Transition to $m
        set t [expr double($diff)/$::MovieMaker::framerate]
        lset ::VCR::movieTimeList $a $t
    }
    lappend ::VCR::movieTimeList 0.0
    lappend ::VCR::movieList $m
    lappend ::vcr_gui::mlist ""
    ::vcr_gui::fillmlist
    if {  $::vcr_gui::MovieMakerStatus == "*** enabled ***" } {
       ::VCR::updateMovieMakerDuration
#        ::vcr_gui::calctimescale
    }
}


proc ::vcr_gui::removefrommovie {} {
    set m [.vcr_gui.moviemaker.list.listframe.list curselection]
    if { [llength $m] != 1 } {
        ::vcr_gui::movieselwarning
        return
    } 
    set ::VCR::movieList [lreplace $::VCR::movieList $m $m]
    set ::VCR::movieTimeList [lreplace $::VCR::movieTimeList $m $m]
    set ::vcr_gui::mlist [lreplace $::vcr_gui::mlist $m $m]
    if { [llength $::VCR::movieList] > 0 &&  $m > 0 && $m < [llength $::VCR::movieList]} {
        set a [ expr $m - 1 ]
        set m [lindex $::VCR::movieList $m]
        set n [lindex $::VCR::movieList $a]
        set nf $::VCR::viewpoints($n,4)
        set mf $::VCR::viewpoints($m,4)
        #puts "$nf ($n)->$mf ($m)"
        set diff [expr abs($mf-$nf)]
        if { $diff == 0 } { set diff 12 }
        ##UPDATE MovieTimeList $n (next to last) with Appropriate Transition to $m
        set t [expr double($diff)/$::MovieMaker::framerate]
        lset ::VCR::movieTimeList $a $t
    }
    #.vcr_gui.moviemaker.list.listframe.list selection clear $m
    ::vcr_gui::fillmlist
    if {  $::vcr_gui::MovieMakerStatus == "*** enabled ***" } {
       ::VCR::updateMovieMakerDuration
#        ::vcr_gui::calctimescale
    }
}


proc ::vcr_gui::clearmlist {} {
    set ::VCR::movieList {}
    set ::VCR::movieTimeList {}
    set ::vcr_gui::mlist {}
    set ::VCR::movieTime 0.0
    set ::vcr_gui::duration 0.0
}


proc ::vcr_gui::toggleMovieMaker {} {
    ::VCR::toggleMovieMaker
    if { $::VCR::MovieMakerStatus == 1 } {
        if { [llength $::VCR::movieList] > 1 } {
          set ::vcr_gui::MovieMakerStatus "*** enabled ***"
          .vcr_gui.moviemaker.timing.labelstatus configure -foreground darkgreen
          .vcr_gui.moviemaker.timing.switch configure -text "Disable" 

        }
    } else {
        set ::vcr_gui::MovieMakerStatus "disabled"
        .vcr_gui.moviemaker.timing.labelstatus configure -foreground red
        .vcr_gui.moviemaker.timing.switch configure -text "Enable" 
    }
}


proc ::vcr_gui::moveup {} {
    set m [.vcr_gui.moviemaker.list.listframe.list curselection]
    if { [llength $m] != 1 } {
        ::vcr_gui::movieselwarning
        return
    }
    if {$m > 0} {
        set n [expr $m - 1]
        set mVal [lindex $::VCR::movieList $m]
        set nVal [lindex $::VCR::movieList $n]
        set mT   [lindex $::VCR::movieTimeList $m]
        set nT   [lindex $::VCR::movieTimeList $n]
        lset ::VCR::movieList $m $nVal
        lset ::VCR::movieList $n $mVal
        .vcr_gui.moviemaker.list.listframe.list selection clear $m
        .vcr_gui.moviemaker.list.listframe.list selection set $n
        set changes {}
        if { $n > 0 } {
            lappend changes [expr $n-1]
        }
        lappend changes $n
        if { $m < [expr [llength $::VCR::movieList]-1] } {
            lappend changes $m
        }
        foreach a $changes {
            set n [lindex $::VCR::movieList $a]
            set m [lindex $::VCR::movieList [expr $a+1]]
            set nf $::VCR::viewpoints($n,4)
            set mf $::VCR::viewpoints($m,4)
            #puts "$nf ($n)->$mf ($m)"
            set diff [expr abs($mf-$nf)]
            if { $diff == 0 } { set diff 12 }
            ##UPDATE MovieTimeList $n (next to last) with Appropriate Transition to $m
            set t [expr double($diff)/$::MovieMaker::framerate]
            lset ::VCR::movieTimeList $a $t
        }       
        ::vcr_gui::fillmlist
        ::VCR::updateMovieMakerDuration

    }
}


proc ::vcr_gui::movedown {} {
    set m [.vcr_gui.moviemaker.list.listframe.list curselection]
    if { [llength $m] != 1 } {
        ::vcr_gui::movieselwarning
        return
    }
    if {[expr $m+1] < [llength $::VCR::movieList]} {
        set n [expr $m + 1]
        set mVal [lindex $::VCR::movieList $m]
        set nVal [lindex $::VCR::movieList $n]
        set mT   [lindex $::VCR::movieTimeList $m]
        set nT   [lindex $::VCR::movieTimeList $n]
        lset ::VCR::movieList $m $nVal
        lset ::VCR::movieList $n $mVal
        .vcr_gui.moviemaker.list.listframe.list selection clear $m
        .vcr_gui.moviemaker.list.listframe.list selection set $n
        set changes {}
        if { $m > 0 } {
            lappend changes [expr $m-1]
        }
        lappend changes $m
        if { $n < [expr [llength $::VCR::movieList]-1] } {
            lappend changes $n
        }
        foreach a $changes {
            set n [lindex $::VCR::movieList $a]
            set m [lindex $::VCR::movieList [expr $a+1]]
            set nf $::VCR::viewpoints($n,4)
            set mf $::VCR::viewpoints($m,4)
            #puts "$nf ($n)->$mf ($m)"
            set diff [expr abs($mf-$nf)]
            if { $diff == 0 } { set diff 12 }
            ##UPDATE MovieTimeList $n (next to last) with Appropriate Transition to $m
            set t [expr double($diff)/$::MovieMaker::framerate]
            lset ::VCR::movieTimeList $a $t
        }       
       ::vcr_gui::fillmlist
       ::VCR::updateMovieMakerDuration
    }
}

proc ::vcr_gui::abortpreview {} {
    set ::VCR::abort 1
    set ::vcr_gui::previewaborted 1
}

proc ::vcr_gui::disableMovieFrameButtonsExcept { args } {
    set b_affected [list .vcr_gui.moviemaker.editing.removefrommovie\
                         .vcr_gui.moviemaker.editing.moveup\
                         .vcr_gui.moviemaker.editing.movedown\
                         .vcr_gui.moviemaker.editing.clear\
                         .vcr_gui.moviemaker.timing.switch\
                         .vcr_gui.moviemaker.timing.settotaltime\
                         .vcr_gui.moviemaker.timing.edittime\
                         .vcr_gui.moviecmd.addtomovie\
                         .vcr_gui.moviecmd.addalltomovie\
                         .vcr_gui.moviemaker.editing.writerenderscript\
                         .vcr_gui.moviemaker.timing.preview\
                    ]
    #remove writerenderscript button for now
    lappend args .vcr_gui.moviemaker.editing.writerenderscript
    foreach b $b_affected {
          if { [lsearch -exact $args $b] == -1 } {
               $b configure -state disabled
           }
    }    
}

proc ::vcr_gui::enableMovieFrameButtonsExcept { args } {
    set b_affected [list .vcr_gui.moviemaker.editing.removefrommovie\
                         .vcr_gui.moviemaker.editing.moveup\
                         .vcr_gui.moviemaker.editing.movedown\
                         .vcr_gui.moviemaker.editing.clear\
                         .vcr_gui.moviemaker.timing.switch\
                         .vcr_gui.moviemaker.timing.settotaltime\
                         .vcr_gui.moviemaker.timing.edittime\
                         .vcr_gui.moviecmd.addtomovie\
                         .vcr_gui.moviecmd.addalltomovie\
                         .vcr_gui.moviemaker.editing.writerenderscript\
                         .vcr_gui.moviemaker.timing.preview\
                    ]
    #remove writerenderscript button for now
    lappend args .vcr_gui.moviemaker.editing.writerenderscript
    foreach b $b_affected {
          if { [lsearch -exact $args $b] == -1 } {
               $b configure -state normal
           }
    }    
}



proc ::vcr_gui::preview {} {
    ::VCR::save_vp "here"
    set ::vcr_gui::previewaborted 0
    set n [expr [llength $::VCR::movieList]-1]
    set OrigSel -1
    set OrigYView [lindex [.vcr_gui.moviemaker.list.listframe.list yview] 0]
    if { [llength [.vcr_gui.moviemaker.list.listframe.list curselection]] == 1 } {
        set OrigSel [.vcr_gui.moviemaker.list.listframe.list curselection]
        .vcr_gui.moviemaker.list.listframe.list selection clear $OrigSel
    }
    .vcr_gui.moviemaker.list.listframe.list selection set 0
    .vcr_gui.moviemaker.timing.preview configure -text "Abort Preview" -command ::vcr_gui::abortpreview
    .vcr_gui.moviemaker.list.listframe.list yview 0
    set i 0
    ::vcr_gui::disableMovieFrameButtonsExcept .vcr_gui.moviemaker.timing.preview
    while { $i < $n && $::vcr_gui::previewaborted == 0 } {
        display update ui
        set j [expr $i+1]
        set t [expr [lindex $::VCR::movieTimeList $i]*$::vcr_gui::timescale]
        ::VCR::movetime_vp [lindex $::VCR::movieList $i] [lindex $::VCR::movieList $j] $t 1
        .vcr_gui.moviemaker.list.listframe.list selection clear $i
        .vcr_gui.moviemaker.list.listframe.list selection set $j
        .vcr_gui.moviemaker.list.listframe.list yview $j
        incr i
   }
   .vcr_gui.moviemaker.timing.preview configure -text "Preview Movie" -command ::vcr_gui::preview
   ::vcr_gui::enableMovieFrameButtonsExcept .vcr_gui.moviemaker.timing.preview
   unset ::vcr_gui::previewaborted
   for {set i 0 } {$i < [expr $n +1] } {incr i } {
     .vcr_gui.moviemaker.list.listframe.list selection clear $i
   }
   if { $OrigSel > -1 } {
     .vcr_gui.moviemaker.list.listframe.list selection set $OrigSel
   }
  .vcr_gui.moviemaker.list.listframe.list yview moveto $OrigYView

   ::VCR::retrieve_vp "here"
   ::VCR::remove_vp "here"
}


proc ::vcr_gui::setnewtime {} {
  
        lset ::VCR::movieTimeList $::vcr_gui::m [expr $::vcr_gui::selectedtransitiontime/$::VCR::movietimescale] 
        ::vcr_gui::fillmlist
        unset ::vcr_gui::m
     
    if { $::vcr_gui::MovieMakerStatus == "*** enabled ***" && [llength [trace info variable ::MovieMaker::numframes]] > 0 } {
       ::VCR::updateMovieMakerDuration
#        ::MovieMaker::durationChanged [expr int(round($::vcr_gui::duration))]
#        set ::MovieMaker::movieduration [expr int(round($::vcr_gui::duration))]
    }
}


proc ::vcr_gui::setnewtotaltime {} {
    if {$::VCR::movieDuration <= 0 } {
        tk_messageBox -type ok -icon error -title "VCR Error" -message "Movie time must be positive."
        set ::VCR::movieDuration $::VCR::oldmovieDuration
        return -1
    }
    set ::VCR::movieDuration [format "%6.2f" $::VCR::movieDuration]
    if { $::VCR::movieTime > 0 } {
            set ::VCR::movietimescale [expr $::VCR::movieDuration/$::VCR::movieTime]
            ::vcr_gui::fillmlist
        if { $::vcr_gui::MovieMakerStatus == "*** enabled ***" &&\
            [llength [trace info variable ::MovieMaker::numframes]] > 0 } {
           ::VCR::updateMovieMakerDuration
        }
    }
    return 0
}

proc ::vcr_gui::toggle_settotaltimegui {} {
    if { $::vcr_gui::totaltimeedit } {
       set ::vcr_gui::reEnableMovieMaker 0
      .vcr_gui.moviemaker.list.time configure -state normal -bg white -relief sunken
      #.vcr_gui.moviemaker.list.time configure -bg white
      .vcr_gui.moviemaker.timing.settotaltime configure -text "Save"
       if { $::VCR::MovieMakerStatus == 1 } {
            ::vcr_gui::toggleMovieMaker
            set ::vcr_gui::reEnableMovieMaker 1
       }
       ::vcr_gui::disableMovieFrameButtonsExcept .vcr_gui.moviemaker.timing.settotaltime
       set ::vcr_gui::oldfocus [focus]
       focus .vcr_gui.moviemaker.list.time
    } elseif { $::vcr_gui::totaltimeedit == 0 } { 
      .vcr_gui.moviemaker.list.time configure -state disabled -disabledforeground black -relief flat
      .vcr_gui.moviemaker.timing.settotaltime configure -text "Set Total Time"
      if { [info exists ::vcr_gui::reEnableMovieMaker] && $::vcr_gui::reEnableMovieMaker == 1 } {
         ::vcr_gui::toggleMovieMaker 
      }
      if { [info exists ::vcr_gui::reEnableMovieMaker] } {
        unset ::vcr_gui::reEnableMovieMaker
      }
      ::vcr_gui::enableMovieFrameButtonsExcept .vcr_gui.moviemaker.timing.settotaltime
      if { [info exists ::vcr_gui::oldfocus] } {
            focus $::vcr_gui::oldfocus 
            unset ::vcr_gui::oldfocus
       }
   } 
}

proc ::vcr_gui::settotaltime {} {
    if { !$::vcr_gui::totaltimeedit } {
       set ::VCR::oldmovieDuration $::VCR::movieDuration
       set ::vcr_gui::totaltimeedit 1
       ::vcr_gui::toggle_settotaltimegui 
    } elseif { [::vcr_gui::setnewtotaltime] == 0 } {
          unset ::VCR::oldmovieDuration
          set ::vcr_gui::totaltimeedit 0
          ::vcr_gui::toggle_settotaltimegui 
    }
}

proc ::vcr_gui::toggle_edit_time_gui {} {
    if { $::vcr_gui::selectedtransitiontime == " " } {
       ::vcr_gui::enableMovieFrameButtonsExcept .vcr_gui.moviemaker.timing.edittime
      .vcr_gui.moviemaker.timing.transtime configure -state disabled -bg white -relief flat
      .vcr_gui.moviemaker.timing.edittime configure -text "Change Transition Time"
      focus .vcr_gui.moviemaker.list.listframe.list
    } else {
       ::vcr_gui::disableMovieFrameButtonsExcept .vcr_gui.moviemaker.timing.edittime
      .vcr_gui.moviemaker.timing.transtime configure -state normal -bg white -relief sunken
      .vcr_gui.moviemaker.timing.edittime configure -text "Save Transition Time"
       focus .vcr_gui.moviemaker.timing.transtime
    }

}

proc ::vcr_gui::edit_time {} {

    if { $::vcr_gui::selectedtransitiontime == " " } {
        set ::vcr_gui::m [.vcr_gui.moviemaker.list.listframe.list curselection]
        if { [llength $::vcr_gui::m] != 1 } {
            ::vcr_gui::movieselwarning
            return
        }
        if { $::vcr_gui::m < [expr [llength $::VCR::movieTimeList] -1] } {
           set ::vcr_gui::selectedtransitiontime [format "%5.2f" [expr [lindex $::VCR::movieTimeList $::vcr_gui::m]*$::VCR::movietimescale]]
           set ::vcr_gui::reEnableMovieMaker $::VCR::MovieMakerStatus
           if { $::VCR::MovieMakerStatus == 1 } {
                ::vcr_gui::toggleMovieMaker
           }
        }
    } else {
        if { $::vcr_gui::selectedtransitiontime > 0 } {
            ::vcr_gui::setnewtime
            set ::vcr_gui::selectedtransitiontime " "
            if { [info exists ::vcr_gui::reEnableMovieMaker] && $::vcr_gui::reEnableMovieMaker == 1 } {
               ::vcr_gui::toggleMovieMaker 
            }
            if { [info exists ::vcr_gui::reEnableMovieMaker] } {
              unset ::vcr_gui::reEnableMovieMaker
            } 
       }
    }
    ::vcr_gui::toggle_edit_time_gui 
}


proc ::vcr_gui::togglerepchanges {} {
    ::VCR::toggleRepChanges
    if { $::VCR::repchanges == 0 } {
        .vcr_gui.menubar.settings.menu entryconfigure 0 -label "Enable representation changes"
    } else {
        .vcr_gui.menubar.settings.menu entryconfigure 0 -label "Disable representation changes"
    }
    
}

proc vcr_tk_cb {} {
  vcr_gui   ;# start the GUI
  return $::vcr_gui::w
}
