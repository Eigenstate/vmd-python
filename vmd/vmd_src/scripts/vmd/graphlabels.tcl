############################################################################
#cr                                                                       
#cr            (C) Copyright 1995-2007 The Board of Trustees of the            
#cr                        University of Illinois                         
#cr                         All Rights Reserved                           
#cr                                                                       
############################################################################

# Callback for plotting values of labels.  Triggered by the "graph" button
# on the Labels menu.



# This callback sends data to xmgr, one dataset at a time.  If xmgrace is
# not found, it reverts to the save dialog.
proc vmd_labelcb_xmgr { args } {
  global vmd_graph_label
  foreach item $vmd_graph_label {
    foreach { type id } $item { break }
    set data [label graph $type $id]
    set input "@type xy\n@  title \"$item\"\n"
    set i 0
    foreach elem $data {
      append input "  $i $elem\n"
      incr i
    }
    set rc [catch {exec xmgrace -pipe << $input &} msg]
    if { $rc } {
      vmd_labelcb_save
    }
  }
}

# This callback sends data to the MultiPlot plugin one dataset at a time.  
# If the Plotter plugin is not found, it reverts to the save dialog.
proc vmd_labelcb_multiplot { args } {
  global vmd_graph_label

  set rc [catch {
    set j 0
    set colorlist {black red green blue white Skyblue3 darkgreen orange 
                   pink cyan tan maroon purple grey yellow silver lime}
 
    foreach item $vmd_graph_label {
      foreach { type id } $item { break }
      set data [label graph $type $id]

      # build list of label values
      set xlist {}
      set ylist {}
      set i 0
      foreach elem $data {
        lappend xlist $i
        lappend ylist $elem
        incr i
      }

      set color [lindex $colorlist [expr $j % [llength $colorlist]]]
      if { $j > 0 } {
        $plothandle add $xlist $ylist -linecolor $color -legend "$id";
        $plothandle replot;
      } else {
        set plothandle [multiplot -x $xlist -y $ylist \
                      -xlabel "Frame" -ylabel $type -title "Label Graph" \
                      -lines -linewidth 1 -linecolor $color \
                      -marker point -legend "$id" -plot];
      }
      incr j;
    }
  } ]
  if { $rc } {
    vmd_labelcb_save
  }
}


# This callback sends data to the Plotter plugin one dataset at a time.  
# If the Plotter plugin is not found, it reverts to the save dialog.
proc vmd_labelcb_plotter { args } {
  global vmd_graph_label
  foreach item $vmd_graph_label {
    foreach { type id } $item { break }
    set data [label graph $type $id]

    # build list of label values
    set plotlist {}
    set i 0
    foreach elem $data {
      lappend plotlist [list $i [format "%.2f" $elem]]
      incr i
    }
    set datalist [list [list $plotlist {"label"} ]]
    set rc [catch {::Plotter::plotData timestep $item "Label Graph" $datalist} msg]
    if { $rc } {
      vmd_labelcb_save
    }
  }
}


# This callback simply saves the data to a file of the user's choice using
# the Tk dialog box if available, otherwise through the text interface.
proc vmd_labelcb_save { args } {
  global vmd_graph_label tk_version
  foreach item $vmd_graph_label {
    foreach { type id } $item { break }
    set data [label graph $type $id]
    set title "Enter filename to save label data for $item"
    if [info exists tk_version] {
      set fname [tk_getSaveFile -title $title]
    } else {
      puts $title
      gets stdin fname
    }
    if { [string length $fname] } {
      set fd [open $fname w]
      foreach elem $data { puts $fd $elem }
      close $fd
    }
  }
}


# Choose a callback based on the platform: xmgr for unix, save for everyone
# else (for now).  Exception: if a command named vmd_labelcb_user is defined,
# use that one instead of the default.
proc vmd_labelcb { args } {
  global tcl_platform
  if { [llength [info commands vmd_labelcb_user]] } {
    vmd_labelcb_user $args
  } else {
    switch $tcl_platform(platform) {
      unix { 
        # Set the display variable to :0.0, unless it's already been set
        global env
        if { ![info exists env(DISPLAY)] } {
          puts "Setting DISPLAY environment variable to :0.0."
          set env(DISPLAY) :0.0
        }
        vmd_labelcb_multiplot $args
      }
      default {
        vmd_labelcb_multiplot $args
      }
    }
  }
}

trace variable vmd_graph_label w vmd_labelcb
