## NAME: color_scale_bar
## 
## $Id: colorscalebar.tcl,v 1.26 2018/03/08 06:25:38 johns Exp $
##
## SYNOPSIS:
##   color_scale_bar draws a color bar on the screen to show all of the
##   current colors. It also allows to add a title and shows labels beside 
##   the color bar to show the range of the mapped values. The label and
##   color range can be taken from input or automaticall read from a 
##   selected molecule and representation.
##
## VERSION: 3.0
##    Requires VMD version:  1.8.8 or later
## 
## PROCEDURES:
##    color_scale bar
## 
## DESCRIPTION:
##    To draw a color scale bar with length=1.5, width=0.25, the range of
##    mapped values is 0~128, and you want 8 labels.
##    color_scale_bar 1.5  0.25  0  128 8
## 
## COMMENTS: The size of the bar also depends on the zoom scale.
## 
## AUTHOR:
##    Wuwei Liang (gtg088c@prism.gatech.edu)
##
##    New version 2 built on Wuwei Liang's code, by Dan Wright 
##                  <dtwright@uiuc.edu>
##
##    Plugin-ized version by John Stone
##    Various subsequent bug fixes by John Stone
##
##    GUI overhaul and option to select molecule and representation
##    where the color scale is taken from by Axel Kohlmeyer.
## 
##    New version 3 rewritten to avoid using line primitives that 
##    don't work well in ray tracing engines, replaced with cylinders
##    and color-per-vertex triangle primitives by John Stone
##

# This function draws a color bar to show the color scale
# length = the length of the color bar
# width = the width of the color bar
# min = the minimum value to be mapped
# max = the maximum mapped value
# label_num = the number of labels to be displayed

namespace eval ::ColorScaleBar:: {
  namespace export color_scale_bar delete_color_scale_bar
  variable version      3.0; # version number of this package.
  variable w
  variable molid        "0"; # id of the molecule
  variable moltxt        ""; # title the molecule
  variable repid        "0"; # id of the representation
  variable reptxt        ""; # title/name of the rep
  variable bar_mol       -1
  variable bar_text "Color Scale:"; # legend text for the colorbar
  variable text_show    "0"; # whether to show the legend.
#  variable bar_orient   "0"; # orientation of the colorbar (vertical=0, horizontal=1)
#  variable bar_text_pos "s"; # position of the colorbar (n,s,w,e)
#  variable text_orient  "0"; # orientation of the text label to the bar.
  variable lengthsetting    0.8
  variable widthsetting     0.05
  variable autoscale        0
  variable fixedsetting     1
  variable minvalue         0
  variable maxvalue         100
  variable axislabels       5
  variable textcolor        white
  variable fpformat         1      # 0=decimal, 1=scientific
}

package provide colorscalebar $ColorScaleBar::version

proc ::ColorScaleBar::color_scale_bar {{length 0.5} {width 0.05} {auto_scale 1} {fixed 1} {min 0} {max 100} {label_num 5} {text 16} {fp_format 0} {x_pos -1.0} {y_pos -1.0} {replacebar 1} {molid top} {repid 0} {showlegend 0} {legend "Color Scale:"}} {
  variable bar_mol

  set m $molid
  if {[string equal "$molid" "top"]} {set m [molinfo top]}

  # if there's already a color scale bar molecule, delete and replace it
  if {$replacebar == 1} {
    delete_color_scale_bar
  }

  # So that the draw cmds will work right, must save top mol and set top
  # to our new created mol, then set it back later.
  set old_top [molinfo top]
  if { $old_top == -1 } {
    vmdcon -err "Color Scale Bar Plugin: No molecules loaded"
    return -1;
  }

  array set viewpoints {}
  foreach mol [molinfo list] {
  # save orientation and zoom parameters for each molecule.
    set viewpoints($mol) [molinfo $mol get { 
    center_matrix rotate_matrix scale_matrix global_matrix}]
  } 
 
  # don't update the display while we do this since otherwise there
  # will be thousands of individual draw commands going on 
  display update off
  display resetview

  # XXX: the previous heuristics for setting the min/max values only worked, if there
  # was exactly one molecule using a color scale and the colorized rep was the first
  # representation. so quite often it could not work. color scales are per rep (and mol),
  # so we have to pick _one_representation.
  if {$auto_scale == 1} {
    set min  999999
    set max -999999
    set minmax {0.0 0.0}
    if {[catch {mol scaleminmax $m $repid} minmax]} {
      #XXX: print error message, if in text mode.
      set min 0
      set max 0
    } else {
      set min [lindex $minmax 0]
      set max [lindex $minmax 1]
    }
  }

  # check for situation where all mols were skipped by the catch statement
  if { $min > $max } {
    set min 0
    set max 0
  }

  # Create a seperate molid to draw in, so it's possible for the user to 
  # delete the bar.
  set bar_mol [mol new]
  mol top $bar_mol
  mol rename top "Color Scale Bar"

  # If a fixed bar was requested...
  if {$fixed == 1} {
    mol fix $bar_mol
  }

  # set position relative to top molecule 
  # We want to draw relative to the location of the top mol so that the bar 
  # will always show up nicely.
  #set center [molinfo $old_top get center]
  #set center [regsub -all {[{}]} $center ""]
  #set center [split $center]
  #set start_y [expr [lindex $center 1] - (0.5 * $length)]
  #set use_x [expr 1+[lindex $center 0]]
  #set use_z [lindex $center 2]

  # set in absolute screen position
  set start_y [expr (-0.5 * $length) + $y_pos]
  set use_x $x_pos
  set use_z 0

  # draw background border behind bar, same color as text
  draw color $text

  # disable material properties for the color scale bar background
  # so that it looks truly black (no specular) when it's set black
  set bw [expr $width * 0.05]
  set lx [expr $use_x             - $bw]
  set rx [expr $use_x   + $width  + $bw] 
  set ly [expr $start_y           - $bw]
  set uy [expr $start_y + $length + $bw]
  set bz [expr $use_z - 0.00001]
  
#  draw line "$lx $ly $bz" "$lx $uy $bz" width 2
#  draw line "$lx $uy $bz" "$rx $uy $bz" width 2
#  draw line "$rx $uy $bz" "$rx $ly $bz" width 2
#  draw line "$rx $ly $bz" "$lx $ly $bz" width 2

  draw cylinder "$lx $ly $bz" "$lx $uy $bz" radius $bw
  draw cylinder "$lx $uy $bz" "$rx $uy $bz" radius $bw
  draw cylinder "$rx $uy $bz" "$rx $ly $bz" radius $bw
  draw cylinder "$rx $ly $bz" "$lx $ly $bz" radius $bw

  # draw the color bar
  set mincolorid [colorinfo num] 
  set maxcolorid [expr [colorinfo max] - 1]
  set numscaleids [expr $maxcolorid - $mincolorid]
  set colincstep 8 
  set step [expr $length / double($numscaleids) ]
  draw material Diffuse
#  draw materials off
  for {set colorid $mincolorid } { $colorid <= $maxcolorid } {incr colorid $colincstep } {
    set cur_y [ expr $start_y + ($colorid - $mincolorid) * $step ]
    set next_y [ expr $start_y + ($colincstep + $colorid - $mincolorid) * $step ]
    set next_x [expr $use_x+$width]
    set next_col [expr $colorid + $colincstep]
#    draw color $colorid
#    draw line "$use_x $cur_y $use_z"  "$next_x $cur_y $use_z"
    draw tricolor "$use_x $cur_y $use_z"  "$next_x $cur_y $use_z"  "$next_x $next_y $use_z"  {0 0 1} {0 0 1} {0 0 1}  $colorid $colorid $next_col
    draw tricolor "$next_x $next_y $use_z"  "$use_x $next_y $use_z"  "$use_x $cur_y $use_z"  {0 0 1} {0 0 1} {0 0 1}  $next_col $next_col $colorid
  }

  # draw the labels
  set coord_x [expr (1.2*$width)+$use_x];
  set step_size [expr $length / $label_num]
  set color_step [expr double($numscaleids)/$label_num]
  set value_step [expr ($max - $min ) / double ($label_num)]
  draw color $text
  for {set i 0} {$i <= $label_num } { incr i 1} {
    set coord_y [expr $start_y+$i * $step_size ]
    set cur_text [expr $min + $i * $value_step ]

    set labeltxt ""
    if { $fp_format == 0 } {
      # format the string in decimal notation
      # we save a spot for a leading '-' sign
      set labeltxt [format "% 7.2f"  $cur_text]
    } else {
      # format the string in scientific notation
      # we save a spot for a leading '-' sign
      # since there are only 1024 distinct colors, there's no point in 
      # displaying more than 3 decimal places after the decimal point
      set labeltxt [format "% #.3e"  $cur_text]
    }
    draw text  "$coord_x $coord_y $use_z" "$labeltxt"
#    draw line "[expr $use_x+$width] $coord_y $use_z" "[expr $use_x+(1.45*$width)] $coord_y $use_z" width 2
    draw cylinder "[expr $use_x+$width] $coord_y $use_z" "[expr $use_x+(1.45*$width)] $coord_y $use_z" radius $bw
  }

  if {$showlegend == 1} {
      # set in absolute screen position
      draw color $text
      set use_y [expr {$start_y + $length + 0.15}]
      draw text  "$x_pos $use_y $use_z" "$legend"
  }
  # re-set top
  if { $old_top != -1 } {
    mol top $old_top
  }

  foreach mol [molinfo list] {
    if {$mol == $bar_mol} continue
    # restore orientation and zoom
    molinfo $mol set {center_matrix rotate_matrix scale_matrix
    global_matrix} $viewpoints($mol)
  }

  display update on

  return 0
}

# if there's a color scale bar molecule, delete and replace it
proc ::ColorScaleBar::delete_color_scale_bar { } {
  variable bar_mol

  foreach m [molinfo list] {
    if {$m == $bar_mol || [string compare [molinfo $m get name] "{Color Scale Bar}"] == 0} {
      mol delete $m
    }
  }

  # invalidate bar_mol
  set bar_mol -1
}


proc ::ColorScaleBar::gui { } {
  variable w
  variable lengthsetting
  variable widthsetting
  variable minvalue
  variable maxvalue
  variable axislabels
  variable textcolor
  variable fpformat
  variable autoscale
  variable text_show

  # If already initialized, just turn on
  if { [winfo exists .colorscalebargui] } {
    wm deiconify $w
    return
  }

  set w [toplevel ".colorscalebargui"]
  wm title $w "Color Scale Bar"
  wm resizable $w 0 0

  ##
  ## make the menu bar
  ##
  frame $w.menubar -relief raised -bd 2 ;# frame for menubar
  pack $w.menubar -padx 1 -fill x
  menubutton $w.menubar.help -text "Help   " -underline 0 -menu $w.menubar.help.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.help config -width 5

  ##
  ## help menu
  ##
  menu $w.menubar.help.menu -tearoff no
  $w.menubar.help.menu add command -label "About" \
               -command {tk_messageBox -type ok -title "About Color Scale Bar GUI" \
                              -message "The Color Scale Bar GUI is a wrapper around the color_scale_bar procedure that adds a color scale legend to the VMD display.\n\nBased on color_scale_bar.tcl written by Wuwei Liang and then rewritten by Dan Wright.\nPlugin version and bug fixes by John Stone.\nFurther improvements by Axel Kohlmeyer and John Stone\n\nVersion $::ColorScaleBar::version"}
  $w.menubar.help.menu add command -label "Help..." -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/colorscalebar"
  pack $w.menubar.help -side right
  pack $w.menubar -fill x -anchor w

  frame $w.bar

  # Molecule selector
  label $w.bar.mlabel -text "Use Molecule:" -anchor w
  menubutton $w.bar.m -relief raised -bd 2 -direction flush \
      -text "test text" -textvariable ::ColorScaleBar::moltxt \
      -menu $w.bar.m.menu
  menu $w.bar.m.menu -tearoff no

  # Representation selector
  label $w.bar.rlabel -text "Use Representation:" -anchor w
  menubutton $w.bar.r -relief raised -bd 2 -direction flush \
      -text "test text" -textvariable ::ColorScaleBar::reptxt \
      -menu $w.bar.r.menu
  menu $w.bar.r.menu -tearoff no

  label $w.bar.llabel -text "Color bar length:"
  entry $w.bar.lentry -width 4 -relief sunken -bd 2 \
    -textvariable ::ColorScaleBar::lengthsetting

  label $w.bar.wlabel -text "Color bar width:"
  entry $w.bar.wentry -width 4 -relief sunken -bd 2 \
    -textvariable ::ColorScaleBar::widthsetting

  label $w.bar.alabel -text "Autoscale:"
  radiobutton $w.bar.aoff -text "Off" -value "0" \
    -variable "::ColorScaleBar::autoscale" \
    -command [namespace code {
      trace vdelete vmd_molecule w ::ColorScaleBar::UpdateMolecule
      trace vdelete vmd_replist w ::ColorScaleBar::UpdateReps
      trace vdelete ::ColorScaleBar::molid w ::ColorScaleBar::UpdateReps
      $w.bar.m configure -state disabled
      $w.bar.r configure -state disabled
    }]
  radiobutton $w.bar.aon  -text "On"  -value "1" \
    -variable "::ColorScaleBar::autoscale" \
    -command [namespace code {
      UpdateMolecule
      UpdateReps
      global vmd_molecule
      global vmd_replist
      trace variable vmd_molecule w ::ColorScaleBar::UpdateMolecule
      trace variable vmd_replist w ::ColorScaleBar::UpdateReps
      trace variable ::ColorScaleBar::molid w ::ColorScaleBar::UpdateReps
    }]

  label $w.bar.tlabel -text "Color bar title:"
  entry $w.bar.tentry -width 12 -relief sunken -bd 2 \
    -textvariable ::ColorScaleBar::bar_text

  label $w.bar.dlabel -text "Display title:"
  radiobutton $w.bar.doff -text "Off" -value "0" \
    -variable "::ColorScaleBar::text_show" \
    -command [namespace code { $w.bar.tentry configure -state disabled }]
  radiobutton $w.bar.don  -text "On"  -value "1" \
    -variable "::ColorScaleBar::text_show" \
    -command [namespace code { $w.bar.tentry configure -state normal }]

  label $w.bar.minlabel -text "Minimum scale value:"
  entry $w.bar.minentry -width 4 -relief sunken -bd 2 \
    -textvariable ::ColorScaleBar::minvalue

  label $w.bar.maxlabel -text "Maximum scale value:"
  entry $w.bar.maxentry -width 4 -relief sunken -bd 2 \
    -textvariable ::ColorScaleBar::maxvalue

  label $w.bar.xlabel -text "Number of axis labels:"
  entry $w.bar.xentry -width 4 -relief sunken -bd 2 \
    -textvariable ::ColorScaleBar::axislabels

  label $w.bar.clabel -text "Color of labels:"
  eval [concat "tk_optionMenu $w.bar.cchooser ::ColorScaleBar::textcolor" \
            [colorinfo colors] ]

  grid config $w.bar.llabel   -column 0 -row  0 -sticky w
  grid config $w.bar.lentry   -column 1 -row  0 -columnspan 2 -sticky ew
  grid config $w.bar.wlabel   -column 0 -row  1 -sticky w
  grid config $w.bar.wentry   -column 1 -row  1 -columnspan 2 -sticky ew
  grid config $w.bar.dlabel   -column 0 -row  2 -sticky w
  grid config $w.bar.doff     -column 1 -row  2 -sticky ew
  grid config $w.bar.don      -column 2 -row  2 -sticky ew
  grid config $w.bar.tlabel   -column 0 -row  3 -sticky w
  grid config $w.bar.tentry   -column 1 -row  3 -columnspan 2 -sticky ew
  grid config $w.bar.alabel   -column 0 -row  4 -sticky w
  grid config $w.bar.aoff     -column 1 -row  4 -sticky ew
  grid config $w.bar.aon      -column 2 -row  4 -sticky ew
  grid config $w.bar.mlabel   -column 0 -row  5 -columnspan 1 -sticky w
  grid config $w.bar.m        -column 1 -row  5 -columnspan 2 -sticky ew
  grid config $w.bar.rlabel   -column 0 -row  6  -columnspan 1 -sticky w
  grid config $w.bar.r        -column 1 -row  6 -columnspan 2 -sticky ew
  grid config $w.bar.minlabel -column 0 -row  7 -sticky w
  grid config $w.bar.minentry -column 1 -row  7 -columnspan 2 -sticky ew
  grid config $w.bar.maxlabel -column 0 -row  8 -sticky w
  grid config $w.bar.maxentry -column 1 -row  8 -columnspan 2 -sticky ew
  grid config $w.bar.xlabel   -column 0 -row  9 -sticky w
  grid config $w.bar.xentry   -column 1 -row  9 -columnspan 2 -sticky ew
  grid config $w.bar.clabel   -column 0 -row 10 -sticky w
  grid config $w.bar.cchooser -column 1 -row 10 -columnspan 2 -sticky ew
  grid columnconfigure $w.bar 0 -weight 20
  grid columnconfigure $w.bar 1 -weight 10
  grid columnconfigure $w.bar 2 -weight 10

  frame $w.labelformat
  label $w.labelformat.label -text "Label format: "
  radiobutton $w.labelformat.decimal -text "Decimal" -value "0" \
    -variable ::ColorScaleBar::fpformat
  radiobutton $w.labelformat.scientific -text "Scientific" -value "1" \
    -variable ::ColorScaleBar::fpformat
  pack $w.labelformat.label $w.labelformat.decimal $w.labelformat.scientific  -side left -anchor w -fill x

  button $w.drawcolorscale -text "Draw Color Scale Bar" \
      -command [namespace code {
      if { [color_scale_bar $lengthsetting $widthsetting $autoscale $fixedsetting $minvalue $maxvalue $axislabels $textcolor $fpformat -1.0 -1.0 1 $molid $repid  $text_show $bar_text] == -1 } { 
        tk_dialog .errmsg "Color Scale Bar Error" "Color Scale Bar Plugin: No molecules loaded" error 0 Dismiss
      }
      }]
  button $w.delcolorscale -text "Delete Color Scale Bar" \
    -command ::ColorScaleBar::delete_color_scale_bar 

  pack $w.menubar $w.bar $w.labelformat $w.drawcolorscale $w.delcolorscale -anchor w -fill x 

  # now set initial state of GUI according to namespace variable values

  if {$text_show == 1} {
      $w.bar.tentry configure -state normal
  } else {
      $w.bar.tentry configure -state disabled
  }

  global vmd_molecule
  global vmd_replist
  UpdateMolecule
  UpdateReps
  if {$autoscale == 1} {
    trace variable vmd_molecule w ::ColorScaleBar::UpdateMolecule
    trace variable vmd_replist w ::ColorScaleBar::UpdateReps
    trace variable ::ColorScaleBar::molid w ::ColorScaleBar::UpdateReps
  } else {
    trace vdelete vmd_molecule w ::ColorScaleBar::UpdateMolecule
    trace vdelete vmd_replist w ::ColorScaleBar::UpdateReps
    trace vdelete ::ColorScaleBar::molid w ::ColorScaleBar::UpdateReps
    $w.bar.m configure -state disabled
    $w.bar.r configure -state disabled
  }

  return $w
}

proc colorscalebar_tk_cb { } {
  ::ColorScaleBar::gui
  return $::ColorScaleBar::w
}

# update molecule list
proc ::ColorScaleBar::UpdateMolecule {args} {
    variable w
    variable moltxt
    variable molid
    global vmd_molecule

    # Update the molecule browser
    set mollist [molinfo list]
    $w.bar.m configure -state disabled
    $w.bar.m.menu delete 0 end
    set moltxt "(none)"

    if { [llength $mollist] > 0 } {
        $w.bar.m configure -state normal 
        foreach id $mollist {
            set numatoms 0
            set numvols 0
            lassign [molinfo $id get numatoms] numatoms
            # XXX: the following needs support from VMD. re-enable and
            # adapt when management of volume data sets is rewritten.
            # lassign [molinfo $id get numatoms numvolumedata] numatoms numvols
            ## skip frames w/o atoms and volumetric data sets
            #if {($numatoms < 1) && ($numvols < 1) } continue
            # XXX: this is a workaround to not add the color scale 
            #      bar molecule to the list.
            if {[string equal [molinfo $id get name] "{Color Scale Bar}"]} continue
            $w.bar.m.menu add radiobutton -value $id \
                -command {global vmd_molecule ; \
                           if {[info exists vmd_molecule($::ColorScaleBar::molid)]} { \
                               set ::ColorScaleBar::moltxt "$::ColorScaleBar::molid:[molinfo $::ColorScaleBar::molid get name]"} {set ::ColorScaleBar::moltxt "(none)" ; set molid -1} } \
                -label "$id [molinfo $id get name]" \
                -variable ::ColorScaleBar::molid
            if {$id == $molid} {
                if {[info exists vmd_molecule($molid)]} then {
                    set moltxt "$molid:[molinfo $molid get name]"  
                } else {
                    set moltxt "(none)"
                    set molid -1
                }
            }
        }
    }
}

# update representation list
proc ::ColorScaleBar::UpdateReps {args} {
    variable w
    variable molid
    variable repid
    variable reptxt
    global vmd_molecule

    # Update the rep browser
    $w.bar.r configure -state disabled
    $w.bar.r.menu delete 0 end
    set reptxt "(none)"
    # disable if invalid input
    if {[catch "molinfo $molid get numreps" numreps]} {return}

    if { $numreps > 0 } {
        $w.bar.r configure -state normal 
        for {set id 0} {$id < $numreps} {incr id} {
            set repname [lindex [molinfo $molid get [list [list rep $id]]] 0 0]
            $w.bar.r.menu add radiobutton -value $id \
                -command [list set ::ColorScaleBar::reptxt "$id:$repname"] \
                -label "$id $repname" \
                -variable ::ColorScaleBar::repid
            if {$id == $repid} {
                set reptxt "$id:$repname"  
            }
        }
    }
}

