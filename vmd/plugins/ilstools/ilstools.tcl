##
## Plugin for setting up ILS runs
##
## $Id: ilstools.tcl,v 1.16 2013/04/15 14:43:44 johns Exp $
##
package require readcharmmpar

# XXX hotfix for release. multiseq contains a scrolledframe
# convenience widget, that needs to be moved to a generic package.
package require multiseqdialog 1.1

namespace eval ::ILStools:: {
  # window
  variable w                         ;# handle to main window
   namespace import ::MultiSeqDialog::scrolledframe::scrolledframe
  variable selectcolor   lightsteelblue

  # set version number of this plugin
  variable version        1.5

  # volmap ils parameters
  variable mapfilename    ils_out.dx
  variable runscript      run_ils.tcl
  variable prmfiles       {}
  variable usepbc         0
  variable minmaxsel      "protein"
  variable alignsel       "protein and name CA"
  variable alignmol       -1
  variable alignframe     0
  variable rewrap         1
  variable sysmol         -1
  variable probemol       "none"
  variable probename      "---"
  variable probeopt       ""
  variable mapres         1.0
  variable subres         3
  variable orient         5
  variable maxen          85
  variable temperature    300
  variable nonb_cutoff    12.0
  variable prmfiles
  variable xsize  10
  variable ysize  10
  variable zsize  10
  variable xori   0
  variable yori   0
  variable zori   0
  variable gid    {}
  variable prev_mol    {}

  array set volreps          {}
  
  catch {trace remove variable ::vmd_trajectory_read write ::ILStools::update_mol_menus} err
  catch {trace remove variable ::vmd_molecule        write ::ILStools::update_mol_menus} err
}

package provide ilstools $::ILStools::version

##
## Main routine
## Create the window and initialize data structures
##
proc ilstools_tk {} {
  ILStools::create_gui
  return .ilstools
}

proc ILStools::create_gui {} {
   variable w
   variable selectcolor   
   variable version

   # If already initialized, just turn on
   if {[winfo exists .ilstools]} {
      wm deiconify .ilstools
      raise .ilstools
      #::ILStools::redraw_box
      return
   }


   # Initialize window
   set w [toplevel .ilstools]
   wm title $w "ILS Tools - Calculation Setup" 
   wm resizable $w 0 1
   wm protocol $w WM_DELETE_WINDOW {
#      puts "sysmol $ILStools::sysmol"
      if {[ILStools::have_valid_molecule $ILStools::sysmol]} {
#         puts "gid: $::ILStools::gid"
         foreach g $::ILStools::gid {
            graphics $::ILStools::sysmol delete $g
         }
         set ::ILStools::gid {}
      }
      wm withdraw .ilstools
   }

   # AK changed 2009/05/29: 
   # build a frame container that can be scrolled
   # this will replace the toplevel widget for all
   # subsequent operations except for menu definitions.
   # in this case we allow scrolling only in y but
   scrolledframe $w.f -width 500 -height 580 -fill both -yscroll "$w.s set"
   scrollbar $w.s -command "$w.f yview"

   grid $w.f -row 0 -column 0 -sticky nsew
   grid $w.s -row 0 -column 1 -sticky ns
   grid rowconfigure $w 0 -weight 1
   grid columnconfigure $w 0 -weight 1
   
   set f $w.f.scrolled

   labelframe $f.doc 
   label $f.doc.head -text "Implicit Ligand Sampling (ILS) v$version" -font {Helvetica 10 bold} 
   label $f.doc.label -text "Calculates the free energy potential of a small probe molecule on a 3D grid." -wraplength 270   
   button $f.doc.help -text "Help" -justify center \
      -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/ilstools"
   grid $f.doc.head  -row 1 -column 1 -columnspan 2 -sticky we -pady 2 -padx 2
   grid $f.doc.label -row 2 -column 1 -columnspan 1 -sticky we -pady 2 -padx 2
   grid $f.doc.help  -row 2 -column 2 -columnspan 1 -sticky ns -pady 2 -padx 2
   grid columnconfigure $f.doc 1 -weight 1
   grid columnconfigure $f.doc 2 -weight 0

   # frame for the top 3 lines of the 
   frame $f.head
   # Probe molecule
   label $f.head.probelabel -text "Probe Molecule:" -anchor w
   menubutton $f.head.probemol -relief raised -bd 2 -direction flush \
       -textvariable ::ILStools::probemol -width 6 \
       -menu $f.head.probemol.menu
   menu $f.head.probemol.menu -tearoff no

   label $f.head.probeor -text "   or choose predefined probe:" -anchor w
   tk_optionMenu $f.head.probename ILStools::probename {---} Xenon Hydrogen Nitrogen \
      Oxygen {Nitric Oxide} {Nitrous Oxide} {Nitrogen Dioxide} {Carbon Monoxide} \
      {Carbon Dioxide} Acetylene Ethene Methane
   grid  $f.head -row 2 -column 1 -columnspan 3 -sticky we -pady 2 -padx 2
   grid  $f.head.probelabel -in $f.head -row 0 -column 0 -sticky w -pady 2 -padx 2
   grid  $f.head.probemol   -in $f.head -row 0 -column 1 -sticky w -pady 2
   grid  $f.head.probeor    -in $f.head -row 0 -column 2 -sticky w -pady 2 -padx 2
   grid  $f.head.probename  -in $f.head -row 0 -column 3 -sticky we -pady 2 -padx 2
   grid  columnconfigure $f.head  0 -weight 0 
   grid  columnconfigure $f.head  1 -weight 0 
   grid  columnconfigure $f.head  2 -weight 1
   grid  columnconfigure $f.head  3 -weight 0 -minsize 100

   # System molecule
   label $f.mollabel -text "System molecule:" -anchor w
   menubutton $f.inputmol -relief raised -bd 2 -direction flush \
       -textvariable ::ILStools::sysmol \
       -menu $f.inputmol.menu
   menu $f.inputmol.menu -tearoff no
   grid  $f.mollabel  -in $f.head -row 1 -column 0 -columnspan 1 -sticky w -pady 2 -padx 2
   grid  $f.inputmol  -in $f.head -row 1 -column 1 -columnspan 1 -sticky w -pady 2  
   
   # DX output file
   label $f.filelabel -text "Write map to file:" -anchor w
   entry $f.outfile -textvariable ::ILStools::mapfilename
   button $f.browse -text "Browse..." -width 10 \
      -command ILStools::dialog_getdestfile
   grid $f.filelabel  -in $f.head -row 2 -column 0 -columnspan 1 -sticky w  -pady 2 -padx 2
   grid $f.outfile    -in $f.head -row 2 -column 1 -columnspan 2 -sticky we -pady 2 -padx 2
   grid $f.browse     -in $f.head -row 2 -column 3 -columnspan 1 -sticky e  -pady 2
   grid columnconfigure $f 2   -weight 1
   grid columnconfigure $f {1 3} -weight 0

   ### parameter file list ###
   labelframe $f.para -bd 2 -relief ridge -text "Parameter files" -padx 1m -pady 1m
   frame $f.para.list
   scrollbar $f.para.list.scroll -command "$f.para.list.list yview"
   listbox $f.para.list.list -activestyle dotbox -yscroll "$f.para.list.scroll set" \
      -width 40 -height 3 -setgrid 1 -selectmode browse -selectbackground $selectcolor \
      -listvariable ::ILStools::prmfiles
   frame  $f.para.list.buttons
   button $f.para.list.buttons.add -width 10 -text "Add" -command ::ILStools::dialog_getprmfile
   button $f.para.list.buttons.delete -width 10 -text "Delete" -command {
      foreach i [.ilstools.f.scrolled.para.list.list curselection] {
         .ilstools.f.scrolled.para.list.list delete $i
         #set ::Paratool::paramsetlist  [lreplace $::Paratool::paramsetlist $i $i]
      }
   }
   pack $f.para.list.list  -side left  -fill both -expand 1
   pack $f.para.list.scroll $f.para.list.buttons -side left -fill y
   pack $f.para.list.buttons -side right -fill y
   pack $f.para.list.buttons.add $f.para.list.buttons.delete -fill y -side top -expand 1
   pack $f.para.list -side top  -expand 1 -fill x


   ### Alignment ###
   labelframe $f.align -text "Alignment" 

   # Selection
   label $f.align.sellabel -text "Align all frames based on atom selection:" \
      -justify left -anchor w
   entry  $f.align.selentry -textvariable ::ILStools::alignsel
   button $f.align.selbutt -text "Update" -width 10 -command ::ILStools::update_sel_align
   grid $f.align.sellabel -row 1 -column 1 -columnspan 4 -sticky we -padx 3 -pady 2
   grid $f.align.selentry -row 2 -column 1 -columnspan 4 -sticky we -padx 2 -pady 2
   grid $f.align.selbutt  -row 2 -column 5 -columnspan 1 -sticky e  -padx 2 -pady 2 
   label $f.align.mollabel -text "Align to molecule" 
   menubutton $f.align.mol -relief raised -bd 2 -direction flush \
       -textvariable ::ILStools::alignmol \
       -menu $f.align.mol.menu
   menu $f.align.mol.menu -tearoff no
   label $f.align.framelabel -text "frame:" 
   entry $f.align.frameentry -textvariable ::ILStools::alignframe -width 8
   checkbutton $f.align.rewrap -text "Recenter selection and re-wrap solvent" \
      -variable ::ILStools::rewrap
   button $f.align.wrapnow -text "Wrap now" -width 10 -command ::ILStools::wrapnow
   grid $f.align.mollabel   -row 3 -column 1 -sticky w -padx 2 -pady 2
   grid $f.align.mol        -row 3 -column 2 -sticky w -padx 2 -pady 2
   grid $f.align.framelabel -row 3 -column 3 -sticky w -padx 2 -pady 2
   grid $f.align.frameentry -row 3 -column 4 -sticky w -padx 2 -pady 2
   grid $f.align.rewrap     -row 4 -column 1 -sticky w -padx 2 -pady 2 -columnspan 4 
   grid $f.align.wrapnow    -row 4 -column 5 -sticky w -padx 2 -pady 2

   grid columnconfigure $f.align {4}       -weight 1
   grid columnconfigure $f.align {1 2 3 5} -weight 0

   ### Grid minmax box ###
   labelframe $f.minmax -text "Map dimensions" 

   # Selection
   label $f.minmax.sellabel -text "Create grid based on minmax of atom selection:" \
      -justify left -anchor w
   entry  $f.minmax.selentry -textvariable ::ILStools::minmaxsel
   button $f.minmax.selbutt -text "Update" -width 10 -command ::ILStools::update_sel_minmax
   grid $f.minmax.sellabel -row 1 -column 1 -columnspan 2 -sticky we -padx 3 -pady 2
   grid $f.minmax.selentry -row 2 -column 1 -columnspan 1 -sticky we -padx 2 -pady 2
   grid $f.minmax.selbutt  -row 2 -column 2 -columnspan 1 -sticky e  -padx 2 -pady 2 
   grid columnconfigure $f.minmax {1} -weight 1
   grid columnconfigure $f.minmax {2} -weight 0

   frame $f.minmax.grid
   grid $f.minmax.grid  -row 3 -column 1 -columnspan 2 -sticky we -padx 2 -pady 2

   # Resize grid
   label $f.minmax.grid.dimlabel -text "Grid size:" \
      -justify left -anchor w
   labelspinbox $f.minmax.grid.dimx -text "X-size" -textvariable ILStools::xsize \
      -from 1 -to 1000 -increment 1 -width 8
   labelspinbox $f.minmax.grid.dimy -text "Y-size" -textvariable ILStools::ysize \
      -from 1 -to 1000 -increment 1 -width 8
   labelspinbox $f.minmax.grid.dimz -text "Z-size" -textvariable ILStools::zsize \
      -from 1 -to 1000 -increment 1 -width 8

   grid $f.minmax.grid.dimlabel -row 3 -column 1 -sticky we -pady 2 -padx 2
   grid $f.minmax.grid.dimx     -row 4 -column 1 -sticky we -pady 2 -padx 2
   grid $f.minmax.grid.dimy     -row 5 -column 1 -sticky we -pady 2 -padx 2
   grid $f.minmax.grid.dimz     -row 6 -column 1 -sticky we -pady 2 -padx 2
 
   # Grid origin
   label $f.minmax.grid.orilabel -text "Grid origin:" \
      -justify left -anchor w
   labelspinbox $f.minmax.grid.orix -text "origin X" -textvariable ILStools::xori \
      -from -9999 -to 9999 -increment 1 -width 8
   labelspinbox $f.minmax.grid.oriy -text "origin Y" -textvariable ILStools::yori \
      -from -9999 -to 9999 -increment 1 -width 8
   labelspinbox $f.minmax.grid.oriz -text "origin Z" -textvariable ILStools::zori \
      -from -9999 -to 9999 -increment 1 -width 8

   grid $f.minmax.grid.orilabel -row 3 -column 2 -sticky we -pady 2 -padx 2
   grid $f.minmax.grid.orix     -row 4 -column 2 -sticky we -pady 2 -padx 2
   grid $f.minmax.grid.oriy     -row 5 -column 2 -sticky we -pady 2 -padx 2
   grid $f.minmax.grid.oriz     -row 6 -column 2 -sticky we -pady 2 -padx 2

   # Input options
   labelframe $f.options -text "Options" 

   set frame $f.options
   grid [label $frame.reslabel -text "Resolution:" -anchor w] \
      -row 2 -column 1 -columnspan 1 -sticky ew -padx 2 -pady 2
   grid [entry $frame.res -textvariable ::ILStools::mapres -width 5] \
      -row 2 -column 2 -columnspan 1 -sticky ew -padx 2 -pady 2 
   grid [label $frame.resunits -text "Angstrom" -anchor w] \
      -row 2 -column 3 -columnspan 1 -sticky ew -padx 2 -pady 2 

   grid [label $frame.subrlabel -text "Subsampling:" -anchor w] \
      -row 3 -column 1 -columnspan 1 -sticky ew -padx 2 -pady 2
   grid [spinbox $frame.subr -textvariable ::ILStools::subres -width 5 -from 1 -to 10 -increment 1] \
      -row 3 -column 2 -columnspan 1 -sticky ew -padx 2 -pady 2 
   grid [label $frame.subrunits -text "points/dim" -anchor w] \
      -row 3 -column 3 -columnspan 1 -sticky ew -padx 2 -pady 2 

   grid [label $frame.orientlabel -text "Orientations:" -anchor w] \
      -row 4 -column 1 -columnspan 1 -sticky ew -padx 2 -pady 2
   grid [spinbox $frame.orient -textvariable ::ILStools::orient -width 5 -from 1 -to 100 -increment 1 ] \
      -row 4 -column 2 -columnspan 1 -sticky ew -padx 2 -pady 2 
   grid [label $frame.orientunits -text "" -anchor w] \
      -row 4 -column 3 -columnspan 1 -sticky ew -padx 2 -pady 2 

   grid [label $frame.maxelabel -text "Max. Energy:" -anchor w] \
      -row 5 -column 1 -columnspan 1 -sticky ew -padx 2 -pady 2
   grid [entry $frame.maxe -textvariable ::ILStools::maxen -width 5] \
      -row 5 -column 2 -columnspan 1 -sticky ew -padx 2 -pady 2 
   grid [label $frame.maxeunits -text "kT" -anchor w] \
      -row 5 -column 3 -columnspan 1 -sticky ew -padx 2 -pady 2 

   grid [label $frame.templabel -text "Temperature:" -anchor w] \
      -row 6 -column 1 -columnspan 1 -sticky ew -padx 2 -pady 2
   grid [entry $frame.temp -textvariable ::ILStools::temperature -width 5] \
      -row 6 -column 2 -columnspan 1 -sticky ew -padx 2 -pady 2 
   grid [label $frame.tempunits -text "Kelvin" -anchor w] \
      -row 6 -column 3 -columnspan 1 -sticky ew -padx 2 -pady 2 

   grid [label $frame.culabel -text "Nonbonded cutoff:" -anchor w] \
      -row 7 -column 1 -columnspan 1 -sticky ew -padx 2 -pady 2
   grid [entry $frame.cut -textvariable ::ILStools::nonb_cutoff -width 5] \
      -row 7 -column 2 -columnspan 1 -sticky ew -padx 2 -pady 2  
   grid [label $frame.cutunits -text "Angstrom" -anchor w] \
      -row 7 -column 3 -columnspan 1 -sticky ew -padx 2 -pady 2 

   
   checkbutton $frame.pbc -text "Use periodic boundary conditions" \
      -variable ::ILStools::usepbc
   grid $frame.pbc -row 8 -column 1 -columnspan 3 -sticky w -pady 2

   # FRAME: do it button
   frame $f.doit
   button $f.doit.write   -text "Write input file" -justify center \
      -command {
         if {[ILStools::check_input]} {
            ILStools::dialog_getrunscript
         }
      }
   #button $f.doit.compute -text "Compute" -justify center -command ILStools::run

   grid $f.doit.write   -row 1 -column 1 -sticky we
   #grid $f.doit.compute -row 1 -column 2 -sticky w


   grid $f.doc     -row 1  -column 1 -columnspan 3 -sticky we -padx 2 -pady 3
   grid $f.para    -row 5  -column 1 -columnspan 3 -sticky we -padx 2 -pady 3
   grid $f.align   -row 6  -column 1 -columnspan 3 -sticky we -padx 2 -pady 3
   grid $f.minmax  -row 7  -column 1 -columnspan 3 -sticky we -padx 2 -pady 3
   grid $f.options -row 10 -column 1 -columnspan 3 -sticky we -padx 2 -pady 3 
   grid $f.doit    -row 11 -column 1 -columnspan 3 -sticky we -padx 2 -pady 3

   # Try to guess which one is the probe (tiny) and which is the
   # system molecule (big, should have a psf).
   variable sysmol
   set smallest 30
   set biggest 0
   set mollist [molinfo list] 
   foreach m [molinfo list] {
      set natoms [molinfo $m get numatoms]
      if {$natoms<$smallest} {
         set ILStools::probemol $m
      }
      if {$natoms>$biggest &&
          [lsearch [join [molinfo $m get filetype]] psf] >= 0} {
         set ILStools::sysmol $m
      }
   }
   if {[llength $mollist] && [lsearch $mollist $ILStools::sysmol]>=0} {
     variable alignmol $ILStools::sysmol
     set abc [molinfo $ILStools::sysmol get {a b c}]
     if {[expr {[lindex $abc 0]*[lindex $abc 1]*[lindex $abc 2]!=0}]} {
       set ILStools::usepbc 1
     }
   }
   # Update panes
   update_mol_menus
   update_sel_minmax

   # Draw the outlines of the grid and the region considered
   # for nonbonded interaction
   redraw_box

   # Draw periodic cell boundaries
   #package require pbctools
   #pbc box_draw -shiftcenterrel {-0.5 -0.5 -0.5} -width 1

   variable prev_mol $sysmol

   bind $f.align.selentry <Return> {
      ILStools::update_sel_align
   }
   bind $f.minmax.selentry <Return> {
      ILStools::update_sel_minmax
   }

   trace add variable ::vmd_trajectory_read write ::ILStools::update_mol_menus
   trace add variable ::vmd_molecule        write ::ILStools::update_mol_menus
   trace add variable ::ILStools::sysmol    write ::ILStools::update_mol
   trace add variable ::ILStools::xsize     write ::ILStools::redraw_box
   trace add variable ::ILStools::ysize     write ::ILStools::redraw_box
   trace add variable ::ILStools::zsize     write ::ILStools::redraw_box
   trace add variable ::ILStools::xori      write ::ILStools::redraw_box
   trace add variable ::ILStools::yori      write ::ILStools::redraw_box
   trace add variable ::ILStools::zori      write ::ILStools::redraw_box
}

proc ILStools::labelspinbox {w args} {
   set text [::util::getargs $args "-text" {}]
   set pos [lsearch $args {-text}]
   if {$pos>=0} {
      set args [lreplace $args $pos [expr {$pos+1}]]
   }
   frame $w
   label $w.xlabel -text "$text "
   eval spinbox $w.dimx $args
   grid $w.xlabel -row 1 -column 1 -sticky we
   grid $w.dimx   -row 1 -column 2 -sticky we
}


#
proc ILStools::update_sel_minmax {args} {
   variable sysmol
   variable minmaxsel
   variable alignmol
   variable alignsel
   variable alignframe

   if {![have_valid_molecule $sysmol] } { return 0 }
   if {![catch {atomselect $sysmol $minmaxsel frame 0} sel]} {
      if {![$sel num]} {
	  tk_messageBox -type ok -icon warning -title "No atoms selected" \
	    -message "There are no atoms in your selection defining the map dimensions!"
      }
      if {!($alignmol==$sysmol && $alignframe==0)} {
         if {![update_sel_align]} {
            return 0
         }
#          if {[catch {atomselect $sysmol   $alignsel frame 0} asel] ||
#              [catch {atomselect $alignmol $alignsel frame $alignframe} ref]} {
#             tk_messageBox -type ok -icon error -title "Selection Error" \
#                -message "Error creating atom selection: [$sel text]"
#             return 0
#          }
         # Align frame 0 with the reference
#          set mat [measure fit $asel $ref]
#          set all [atomselect $sysmol all frame 0]
#          $all move $mat
      }

      # Update minmax coordinates
      set minmax [measure minmax $sel]
      variable xori [expr {round([lindex $minmax 0 0]-0.5)}]
      variable yori [expr {round([lindex $minmax 0 1]-0.5)}]
      variable zori [expr {round([lindex $minmax 0 2]-0.5)}]
      set size [vecsub [lindex $minmax 1] [list $xori $yori $zori]]
      variable xsize [expr {round([lindex $size 0]+0.5)}]
      variable ysize [expr {round([lindex $size 1]+0.5)}]
      variable zsize [expr {round([lindex $size 2]+0.5)}]
      $sel delete
      return 1
   } else {
      tk_messageBox -type ok -icon error -title "Selection Error" \
         -message "Error creating atom selection: $sel"
      return 0
   }
}


# Check if the specified alignment selection text is valid
proc ILStools::update_sel_align {args} {
   variable sysmol
   variable alignsel
   variable alignmol
   variable alignframe

   if {! [have_valid_molecule $sysmol]   } { return 0 }
   if {! [have_valid_molecule $alignmol] } { return 0 }

   if {[catch {atomselect $sysmol $alignsel} sel]} {
      tk_messageBox -type ok -icon error -title "Selection Error" \
         -message "Error creating atom selection for molecule $sysmol: $sel"
      return 0
   }

   if {$alignframe<0 || $alignframe>=[molinfo $alignmol get numframes]} {
      tk_messageBox -type ok -icon error -title "Bad Frame" \
         -message "There is no frame $alignframe in molecule $alignmol"
      return 0
   }

   if {[catch {atomselect $alignmol $alignsel frame $alignframe} asel]} {
      tk_messageBox -type ok -icon error -title "Selection Error" \
         -message "Error creating atom selection for molecule $alignmol: $asel"
      return 0
   }

   if {![$asel num]} {
      tk_messageBox -type ok -icon error -title "No atoms selected" \
	 -message "There are no atoms in your selection for the alignment!"
      return 0   
   }

   if {$alignmol!=$sysmol || $alignframe!=0} {
      if {[$sel num]!=[$asel num]} {
         tk_messageBox -type ok -icon error -title "Bad Selection" \
            -message "Number of atom selected for alignment differs in system and reference molecules!"
         return 0
      }

      # Align frame 0 with the reference
      set mat [measure fit $sel $asel]
      set all [atomselect $sysmol all frame 0]
      $all move $mat
   }

   return 1
}


# Check if the given molecule $mol is valid and $mol has at least one frame
proc ILStools::have_valid_molecule {{mol top}} {
   set mollist [molinfo list]

   if {![llength $mollist]} { return 0 }
   lappend mollist top
   if {[lsearch -ascii -exact $mollist $mol] < 0} { return 0 }
   
   if {![molinfo $mol get numframes]} { return 0 }
   return 1
}


# Redraw the boxes indicating the grid dimensions and
# the size of the nonbonded interaction region
proc ILStools::redraw_box {args} {
   variable sysmol 
   variable nonb_cutoff
   variable xsize
   variable ysize
   variable zsize
   variable xori
   variable yori
   variable zori

   if {! [have_valid_molecule $sysmol]} return
   # delete previous box
   variable gid
   foreach g $gid {
      graphics $sysmol delete $g
   }
   set gid {}

   set ori [list $xori $yori $zori]
   set corner [vecadd $ori [list $xsize $ysize $zsize]]
   draw_box $ori $corner
   set ori [vecsub $ori [vecscale {1 1 1} $nonb_cutoff]]
   set corner [vecadd $corner [vecscale {1 1 1} $nonb_cutoff]]
   draw_box $ori $corner
}


# Draw a box with minmax coordinates $coord1 and $coord2 
proc ILStools::draw_box {coord1 coord2 args} {
   variable sysmol 
   set mol $sysmol

   set width [::util::getargs $args "width" 1]
   set minx [lindex $coord1 0]
   set miny [lindex $coord1 1]
   set minz [lindex $coord1 2]
   set maxx [lindex $coord2 0]
   set maxy [lindex $coord2 1]
   set maxz [lindex $coord2 2]

   variable gid
   # draw the new lines
   lappend gid [graphics $mol color green]
   lappend gid [graphics $mol line [list $minx $miny $minz] [list $maxx $miny $minz] width $width]
   lappend gid [graphics $mol line [list $minx $miny $minz] [list $minx $maxy $minz] width $width]
   lappend gid [graphics $mol line [list $minx $miny $minz] [list $minx $miny $maxz] width $width]

   lappend gid [graphics $mol line [list $maxx $miny $minz] [list $maxx $maxy $minz] width $width]
   lappend gid [graphics $mol line [list $maxx $miny $minz] [list $maxx $miny $maxz] width $width]

   lappend gid [graphics $mol line [list $minx $maxy $minz] [list $maxx $maxy $minz] width $width]
   lappend gid [graphics $mol line [list $minx $maxy $minz] [list $minx $maxy $maxz] width $width]

   lappend gid [graphics $mol line [list $minx $miny $maxz] [list $maxx $miny $maxz] width $width]
   lappend gid [graphics $mol line [list $minx $miny $maxz] [list $minx $maxy $maxz] width $width]

   lappend gid [graphics $mol line [list $maxx $maxy $maxz] [list $maxx $maxy $minz] width $width]
   lappend gid [graphics $mol line [list $maxx $maxy $maxz] [list $minx $maxy $maxz] width $width]
   lappend gid [graphics $mol line [list $maxx $maxy $maxz] [list $maxx $miny $maxz] width $width]
}


proc ILStools::update_mol {args} {
   variable prev_mol 

   # delete previous box
   variable gid
   foreach g $gid {
      graphics $prev_mol delete $g
   }
   set gid {}

   variable prev_mol $ILStools::sysmol

   update_sel_minmax
}


# traced command to autoupdate menus when number of mols is changed
proc ILStools::update_mol_menus {args} {
   variable w
   variable sysmol

   set mollist [molinfo list]
   set some_mols_have_coords 0
   
   # System molecule
   set f $w.f.scrolled
   set menu $f.inputmol.menu
   $menu delete 0 last
   $f.inputmol configure -state disabled
   #$menu add radiobutton -label top -value top -variable ILStools::sysmol
   #if [llength $mollist] {
   #   $menu add separator
   #   if {[molinfo top get frame] < 0} {
   #      $menu entryconfigure 0 -state disabled
   #   }
   #} 
   foreach mol $mollist {
      $menu add radiobutton -label "$mol: [molinfo $mol get name]" -value $mol \
         -variable ILStools::sysmol
      if {[molinfo $mol get numframes] > 0} {
         set some_mols_have_coords 1
      } else {
         $menu entryconfigure [expr $mol + 2] -state disabled
      }
   }
   if {$some_mols_have_coords} {
      $f.inputmol configure -state normal
   }

   # Probe molecule
   set menu $f.head.probemol.menu
   $menu delete 0 end
   $f.head.probemol configure -state disabled
   foreach mol $mollist {
      $menu add radiobutton -label "$mol: [molinfo $mol get name]" -value $mol \
         -variable ILStools::probemol
      if {[molinfo $mol get frame] < 0 || $mol==$sysmol} {
         $menu entryconfigure [expr $mol + 2] -state disabled
      }
   }
   if {$some_mols_have_coords} {
      $f.head.probemol configure -state normal
   }

   # Alignment reference molecule
   set menu $f.align.mol.menu
   $menu delete 0 end
   $f.align.mol configure -state disabled
   foreach mol $mollist {
      $menu add radiobutton -label "$mol: [molinfo $mol get name]" -value $mol \
         -variable ILStools::alignmol
      if {[molinfo $mol get frame] < 0} {
         $menu entryconfigure [expr $mol + 2] -state disabled
      }
   }
   if {$some_mols_have_coords} {
      $f.align.mol configure -state normal
   }
}


# Align the selection $sel for all frames, but carry out
# only the transformation part of the alignment.
# This is useful as a preparation for re-wrapping the solvent.
# The rotation has to be omitted since a rotated system cannot
# be wrapped properly.
proc ILStools::shift_to_center {sel {refid none}} {
   set molid [$sel molid]
   if {$refid=="none" || $refid==""} { set refid $molid }
   set ref [atomselect $refid [$sel text] frame 0]
   set all [atomselect $molid all]

   vmdcon -info "Shifting selection to center for all frames"
   for {set i 0} {$i<[molinfo $molid get numframes]} {incr i} {
      $sel frame $i
      $all frame $i
      set mat [measure fit $sel $ref]
      set shift [lindex $mat 0 3]
      lappend shift [lindex $mat 1 3]
      lappend shift [lindex $mat 2 3]
      $all moveby $shift
   }
}

proc ILStools::wrapnow {} {
   variable sysmol
   variable alignmol
   variable alignsel

   # Check for errors in the selection
   if {![update_sel_align]} { return }

   set asel [atomselect $sysmol $alignsel]

   # Align all frames of the protein, but only the transformation
   # part since a rotated system cannot be wrapped properly.
   shift_to_center $asel $alignmol

   # Now we have a system were the protein does not shift anymore.
   # (It still rotates but we have to live with that.)
   # We rewrap the solvent so that our PBC box is always centered
   # at the COM of the protein
   vmdcon -info "Wrapping solvent into periodic boundaries:"
   package require pbctools
   pbc wrap -molid $sysmol -orthorhombic -compound residue -all \
            -center com -centersel $alignsel

   update_sel_minmax
}


# Return a list of strings to be inserted into the ILS run script.
# These commands set up the probe molecule for the calculation.
proc ILStools::setup_probe {} {
   global env
   variable probemol 
   variable probename  
   variable probeopt
   vmdcon -info "Setting up the probe..."
   set selcmds [list "# Set up the probe"]
   lappend selcmds ""
   lappend selcmds "# WARNING: The probe parameters have only been verified to"
   lappend selcmds "#          reproduce experimental solvation free energies for"
   lappend selcmds "#          xenon, oxygen, nitric oxide and carbon monoxide probes."
   lappend selcmds "#          Use parameters for other probes as a starting point"
   lappend selcmds "#          for optimization."


   switch $probename {
      "Xenon" {
         set probefile xenon.xyz
         set selcmds {}
         set probeopt "-probevdw {{-0.494 2.24}}"
      }
      "Hydrogen" {
         set probefile hydrogen.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "# Parameters from CHARMM type HA. Verify solvation energy!"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "\$psel set radius     1.32;  # VDW radius"
         lappend selcmds "\$psel set occupancy -0.022; # VDW epsilon"
      }
      "Oxygen" {
         set probefile oxygen.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "\$psel set radius     1.7;  # VDW radius"
         lappend selcmds "\$psel set occupancy -0.12; # VDW epsilon"
      }
      "Nitrogen" {
         set probefile nitrogen.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "\$psel set radius     1.85;  # VDW radius"
         lappend selcmds "\$psel set occupancy -0.2;   # VDW epsilon"
      }
      "Nitric Oxide" {
         set probefile nitricoxide.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set n \[atomselect \$pmol \"name N\"\]"
         lappend selcmds "\$n set radius     1.85;  # VDW radius"
         lappend selcmds "\$n set occupancy -0.2;   # VDW epsilon"
         lappend selcmds "set o \[atomselect \$pmol \"name O\"\]"
         lappend selcmds "\$o set radius     1.7;   # VDW radius"
         lappend selcmds "\$o set occupancy -0.12;  # VDW epsilon"
      }
      "Nitrous Oxide" {
         set probefile nitrousoxide.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set n \[atomselect \$pmol \"name N\"\]"
         lappend selcmds "\$n set radius     1.85;  # VDW radius"
         lappend selcmds "\$n set occupancy -0.2;   # VDW epsilon"
         lappend selcmds "set o \[atomselect \$pmol \"name O\"\]"
         lappend selcmds "\$o set radius     1.7;   # VDW radius"
         lappend selcmds "\$o set occupancy -0.12;  # VDW epsilon"
      }
      "Nitrogen Dioxide" {
         set probefile nitrogendioxide.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set n \[atomselect \$pmol \"name N\"\]"
         lappend selcmds "\$n set radius     1.85;  # VDW radius"
         lappend selcmds "\$n set occupancy -0.2;   # VDW epsilon"
         lappend selcmds "set o \[atomselect \$pmol \"name O\"\]"
         lappend selcmds "\$o set radius     1.7;   # VDW radius"
         lappend selcmds "\$o set occupancy -0.12;  # VDW epsilon"
      }
      "Carbon Monoxide" {
         set probefile carbonmonoxide.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set c \[atomselect \$pmol \"name C\"\]"
         lappend selcmds "\$c set radius     2.1;  # VDW radius"
         lappend selcmds "\$c set occupancy -0.11; # VDW epsilon"
         lappend selcmds "set o \[atomselect \$pmol \"name O\"\]"
         lappend selcmds "\$o set radius     1.7;  # VDW radius"
         lappend selcmds "\$o set occupancy -0.12; # VDW epsilon"
      }
      "Carbon Dioxide" {
         set probefile carbondioxide.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set c \[atomselect \$pmol \"name C\"\]"
         lappend selcmds "\$c set radius     2.1;  # VDW radius"
         lappend selcmds "\$c set occupancy -0.11; # VDW epsilon"
         lappend selcmds "set o \[atomselect \$pmol \"name O\"\]"
         lappend selcmds "\$o set radius     1.7;  # VDW radius"
         lappend selcmds "\$o set occupancy -0.12; # VDW epsilon"
      }
      "Acetylene" {
         set probefile acetylene.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set c \[atomselect \$pmol \"name C\"\]"
         lappend selcmds "\$c set radius     1.7"
         lappend selcmds "\$c set occupancy -0.11"
         lappend selcmds "set h \[atomselect \$pmol \"name H\"\]"
         lappend selcmds "\$h set radius     1.26"
         lappend selcmds "\$h set occupancy -0.026; # HE2, ethene"
      }
      "Methane" {
         set probefile methane.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set c \[atomselect \$pmol \"name C\"\]"
         lappend selcmds "\$c set radius     1.7"
         lappend selcmds "\$c set occupancy -0.11"
         lappend selcmds "set h \[atomselect \$pmol \"name H\"\]"
         lappend selcmds "\$h set radius     1.26"
         lappend selcmds "\$h set occupancy -0.026; # HE2, ethene"
      }
      "Ethene" {
         set probefile ethene.xyz
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new ilsprobe_$probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds "set c \[atomselect \$pmol \"name C\"\]"
         lappend selcmds "\$c set radius     1.7"
         lappend selcmds "\$c set occupancy -0.11"
         lappend selcmds "set h \[atomselect \$pmol \"name H\"\]"
         lappend selcmds "\$h set radius     1.26"
         lappend selcmds "\$h set occupancy -0.026; # HE2, ethene"
      }
      "---" {
         set psel [atomselect $probemol all]
         # Since no file format I know stores the atomic radii we use the beta
         # field in the PDB file to store them
         $psel set beta [$psel get radius]
         set probefile "ilsprobe_"
         append probefile [file rootname [file tail [molinfo $probemol get name]]]
         append probefile .pdb
         $psel writepdb $probefile
         set probeopt "-probesel \$psel"
         lappend selcmds "set pmol \[mol new $probefile\]"
         lappend selcmds "set psel \[atomselect \$pmol all\]"
         lappend selcmds ""
         lappend selcmds "# Since there is no field for radii in the PDB file we stored"
         lappend selcmds "# them in the beta field and have to copy them back now."
         lappend selcmds "\$psel set radius \[\$psel get beta\]"
      }
   }

   if {$probename!="---" && [llength $selcmds]} {
      set pwdprobefile [file join [pwd] ilsprobe_$probefile]
      if {[file exists $pwdprobefile]} {
         file delete $pwdprobefile
      }
      file copy [file join $env(ILSTOOLSDIR) $probefile] $pwdprobefile
      #set ILStools::probemol [mol new $pwdprobefile]
   }

   return $selcmds
}


proc ILStools::check_input {} {
   variable sysmol
   variable prmfiles
   variable mapfilename
   variable probemol 
   variable probename

   if {![llength $mapfilename]} {
      tk_messageBox -type ok -icon error -title "Missing DX File Name" \
         -message "Please specify name of dx file for the resulting map!"
      return 0
   }
   if {![llength $prmfiles]} {
      tk_messageBox -type ok -icon error -title "Missing Parameter File" \
         -message "Please specify at least one parameter file!"
      return 0
   }
   
   ILStools::readcharmmparams $prmfiles
   if {![ILStools::assigncharmmparams $sysmol]} {
      tk_messageBox -type ok -icon error -title "Missing VDW Parameters" \
         -message "Could not assign VDW parameters to all atoms. \nMake sure you load the same parameter files used for the MD simulation!"
      return 0
   }

   if {$probemol=="none" && $probename=="---"} {
      tk_messageBox -type ok -icon error -title "Missing probe molecule" \
         -message "No probe molecule specified!"
      return 0   
   }

   if {$probemol==$sysmol} {
      tk_messageBox -type ok -icon error -title "Bad probe molecule" \
         -message "Probe molecule cannot be the same as system molecule!"
      return 0   
   }

   if {![molinfo $sysmol get numframes]} {
      tk_messageBox -type ok -icon error -title "No coordinates in molecule" \
         -message "There are no coordiantes in your system molecule (# frames = 0)!"
      return 0   
   }

   return [update_sel_align]
}


# Write a TCL input file taht can be used to run iLS
proc ILStools::write_input {} {
   variable version
   variable sysmol
   variable mapres
   variable subres
   variable orient
   variable temperature
   variable nonb_cutoff
   variable maxen
   variable runscript
   variable prmfiles
   variable probeopt
   variable alignmol
   variable alignsel
   variable alignframe
   variable rewrap
   variable mapfilename
   variable usepbc
   set pbc {}
   if {$usepbc} { set pbc "-pbc" }

   variable psf {}
   variable coorfiles {}
   # Get coordinate and structure files from VMD
   foreach i [join [molinfo $sysmol get filetype]] j [join [molinfo $sysmol get filename]] {
      if {$i=="psf"} {
         set psf $j
      }
      if {$i=="pdb" || $i=="dcd"} {
         lappend coorfiles $j
      }
   }

   set file [open $runscript w]
   puts $file "# Implicit Ligand Sampling run script"
   puts $file "# ==================================="
   puts $file ""
   puts $file "# Running ILS calculation:"
   puts $file ""
   puts $file "# vmd -dispdev text -e <name_of_this_file>"
   puts $file ""
   puts $file "# You will need VMD 1.8.7. or higher."
   puts $file ""
   puts $file "# ILStools plugin of the same or higher version than the one"
   puts $file "# that generated it needs to be available for execution."
# XXX: AK changed 2010/03/15
# The generated script calls some procedures in the ::ILStools:: namespace,
# better to require the package right at the beginning and also add a test 
# to make sure that we don't use a (potentially incompatible) older version
# than the one used to generate the script.
   puts $file "if {\[catch {package require ilstools $version} pkgver\]} {"
   puts $file "    vmdcon -err \"This script requires the ilstools plugin v$version.\""
   puts $file "    exit 1"
   puts $file "}"
   puts $file ""
   puts $file "# Change the input parameters below to your liking."
   puts $file "# The filenames used in this script are relative to the directory"
   puts $file "# for which it was generated but you can of course change then."
   puts $file ""
   puts $file "# If you have a CUDA enabled Nvidia GPU VMD will use the GPU for"
   puts $file "# the computation. Since the all the GPU resources will then be"
   puts $file "# used for ILS your graphical display will freeze up, so don't be"
   puts $file "# surprised. After finishing each frame the display will briefly"
   puts $file "# be updated and freeze again."
   puts $file ""
   puts $file "# Comment this line out to prevent the use of the CUDA implementation:"
   puts $file "set env(VMDCUDAILS) 1"
# XXX: AK changed 2010/03/15
# The device id should be picked from the VMD internal device pool. At the moment
# the ILS code always picks device 0 in case the environment variable is not set.
# So better put it here explicitly for people to see and to show the default.
   puts $file "# Set the device id of the GPU to be used with the CUDA ILS implementation:"
   puts $file "set env(VMDILSCUDADEVICE) 0"
   puts $file ""
   puts $file "# You might want to do a quick test with 1 frame first to see if"
   puts $file "# the syntax is correct and to determine the approximate runtime"
   puts $file "# per frame."
   puts $file ""
   puts $file "# Adjustable parameters:"
   puts $file "# ----------------------"
   puts $file ""
   puts $file "# First and last frames to process"
   puts $file "set first 0"
   puts $file "set last  [expr {[molinfo $sysmol get numframes]-1}]"
   puts $file ""
   puts $file "# Resolution of final downsampled map in Angstrom"
   puts $file "set res    $mapres"
   puts $file ""
   puts $file "# Subsampling of each dimension during computation"
   puts $file "# i.e. each gridpoint of the final map will actually"
   puts $file "# be downsampled from subres^3 points."
   puts $file "set subres $subres"
   puts $file ""
   puts $file "# Control of the angular spacing of probe orientation vectors,"
   puts $file "# i.e. the number of probe conformers generated."
   puts $file "#"
   puts $file "#   1: use 1 orientation only"
   puts $file "#   2: use 6 orientations (vertices of octahedron)"
   puts $file "#   3: use 8 orientations (vertices of hexahedron)"
   puts $file "#   4: use 12 orientations (faces of dodecahedron)"
   puts $file "#   5: use 20 orientations (vertices of dodecahedron)"
   puts $file "#   6: use 32 orientations (faces+vert. of dodecahedron)"
   puts $file "#  >6: geodesic subdivisions of icosahedral faces"
   puts $file "#      with frequency 1, 2, ..."
   puts $file "#"
   puts $file "#   For each orientation a number of rotamers will be"
   puts $file "#   generated. The angular spacing of the rotations"
   puts $file "#   around the orientation vectors is chosen to be about"
   puts $file "#   the same as the angular spacing of the orientation"
   puts $file "#   vector itself."
   puts $file "#   If the probe ha at least one symmetry axis then the"
   puts $file "#   rotations around the orientation vectors are reduced"
   puts $file "#   accordingly. If there is an infinite oder axis (linear"
   puts $file "#   molecule) the rotation will be omitted."
   puts $file "#   In case there is an additional perpendicular C2 axis"
   puts $file "#   the half of the orientations will be ignored so that"
   puts $file "#   there are no antiparallel pairs."
   puts $file "#"
   puts $file "#   Probes with tetrahedral symmetry:"
   puts $file "#   Here conf denotes the number of rotamers for each of"
   puts $file "#   the 8 orientations defined by the vertices of the"
   puts $file "#   tetrahedron and its dual tetrahedron."
   puts $file "set orient   $orient"
   puts $file ""
   puts $file "# Cutoff energy above which the occupancy is regarded zero"
   puts $file "# For GPUs energies of more than 87 always correspond to"
   puts $file "# floating point values of zero for the occupancy. Hence"
   puts $file "# there is no point going higher than that."
   puts $file "set maxen  $maxen"
   puts $file ""
   puts $file "# Temperature of the MD simualtion"
   puts $file "set T   $temperature"
   puts $file ""
   puts $file "# Nonbonded interaction cutoff"
   puts $file "set cutoff $nonb_cutoff"
   puts $file ""
   variable xsize 
   variable ysize 
   variable zsize 
   variable xori  
   variable yori  
   variable zori  
   set p1 [list $xori $yori $zori]
   set p2 [vecadd $p1 [list $xsize $ysize $zsize]]
   set minmax [list [list $p1 $p2]]
   puts $file "# The minmax box defining the free energy map"
   puts $file "# (two opposite corners of the grid)"
   puts $file "set minmax $minmax"
   puts $file ""
   puts $file "# The DX file containing the free energy map"
   puts $file "set dxfile $mapfilename"
   puts $file ""
   puts $file "# -------------------------------------------------------"
   puts $file ""
   foreach line [setup_probe] {
      puts $file $line
   }
   puts $file ""
   puts $file "# -------------------------------------------------------"
   puts $file ""
   puts $file "# This script depends on the ilstools package"
   puts $file "package require ilstools"
   puts $file ""
   puts $file "# Load the molecule"
   puts $file "set molid \[mol new $psf\]"
   foreach cf $coorfiles {
      if {[llength $coorfiles]==1} {
         puts $file "mol addfile \"$cf\" waitfor all first \$first last \$last"
      } else {
         puts $file "mol addfile \"$cf\" waitfor all"
      }
   }
   puts $file ""
   set transform {}
   if {$alignmol==$sysmol} {
      puts $file "# Selection used for alignment"
      puts $file "set asel \[atomselect \$molid \"$alignsel\" frame 0\]"
   } else {
      set apsf {}
      set acoor {}
      # Get coordinate and structure files from VMD
      foreach i [join [molinfo $alignmol get filetype]] \
              j [join [molinfo $alignmol get filename]] {
         if {$i=="psf"} {
            set apsf $j
         }
         if {$i=="pdb" || $i=="dcd"} {
            lappend acoor $j
         }
      }

      puts $file "# Load the alignment reference molecule"
      if {[string length $apsf]} {
         puts $file "set refid \[mol new \"$apsf\"\]"
         foreach cf $acoor {
            puts $file "mol addfile \"$cf\" waitfor all"
         }
      } else {
         puts $file "set refid \[mol new \"[lindex $acoor 0]\" waitfor all\]"
      }
      puts $file ""
      puts $file "# Selections used for alignment"
      puts $file "set ref  \[atomselect \$refid \"$alignsel\" frame $alignframe\]"
      puts $file "set asel \[atomselect \$molid \"$alignsel\" frame 0\]"
      puts $file ""
      puts $file "# Compute the transformation matrix for fitting first data"
      puts $file "# frame to the reference frame."
      puts $file "set transmat \[measure fit \$asel \$ref\]"
      puts $file "set all \[atomselect \$molid all frame 0\]"
      puts $file "\$all move \$transmat"
      set transform "-transform \$transmat"
   }
   if {$rewrap} {
      puts $file ""
      puts $file "# Align all frames of the protein, but only the transformation"
      puts $file "# part since a rotated system cannot be wrapped properly."
      if {$alignmol==$sysmol} {
         puts $file "ILStools::shift_to_center \$asel"
      } else {
         puts $file "ILStools::shift_to_center \$asel \$refid"
      }
      puts $file ""
      puts $file "# Now we have a system were the protein (or whatever you used"
      puts $file "# as basis of the alignment) does not shift anymore."
      puts $file "# It still rotates but we have to live with that."
      puts $file "# We rewrap the solvent so that our PBC box is always centered"
      puts $file "# at the COM of the protein."
      puts $file "package require pbctools"
      puts $file "pbc wrap -molid \$molid -orthorhombic -compound residue -all \\"
      puts $file "         -center com -centersel \[\$asel text\]"
   }
   puts $file ""
   puts $file "# Set the radius and occupancy field for each atom to the"
   puts $file "# VDW rmin and epsilon values from the force field"
   if {[llength $prmfiles]==1} {
      puts $file "ILStools::readcharmmparams $prmfiles"
   } else {
      puts $file "ILStools::readcharmmparams [list $prmfiles]"
   }
   puts $file "ILStools::assigncharmmparams \$molid"

   puts $file ""
   puts $file "# Run ILS"
   puts $file "volmap ils \$molid \$minmax -cutoff \$cutoff $pbc \\"
   puts $file "    -res \$res -subres \$subres $probeopt -orient \$orient \\"
   puts $file "    -alignsel \$asel $transform -maxenergy \$maxen \\"
   puts $file "    -T \$T -first 0 -last \[expr \$last-\$first-1\] \\"
   puts $file "    -o \$dxfile"
   puts $file ""
   puts $file "# Quit VMD when done with ILS calculation"
   puts $file "quit"
   close $file
}

proc ILStools::command {} {
   variable sysmol

   set cmd [list volmap ils $sysmol]

   lappend cmd -res
   lappend cmd $ILStools::mapres

   lappend cmd -subres
   lappend cmd $ILStools::subres

   lappend cmd -T
   lappend cmd $ILStools::temperature

   lappend cmd -cutoff
   lappend cmd $ILStools::nonb_cutoff
  
   lappend cmd -o
   lappend cmd $ILStools::mapfilename

   return $cmd
}


# Currently not used...
proc ILStools::run {} {
   variable sysmol

   set cmd [command]
   vmdcon -info "running: $cmd"
   #eval $volcmd
  
   if { $ILStools::output_destination == "mol" }  {
      ### The goal here is to show the computed rep. However, since its impossible
      ### to get the num volids for a molecules, this can't be done!
      ### So we chest and show the last volumetric map by assuming that all volmaps
      ### are generated by this plugin, and simply counting...
    set mol $ILStools::dest_molid
    if {"$mol" == "top"} {set mol [molinfo top]}
    
    if {![info exists ILStools::volreps($mol,repname)] || [mol repindex $mol $ILStools::volreps($mol,repname)] < 0} {
      # create new rep:
      mol color ColorID 8
      mol rep Isosurface 0.5 0 0 0 1  ;# find way to show last volume!!!
      mol selection all
      mol material Opaque
      mol addrep $mol
     
      set ILStools::volreps($mol,repname) [mol repname $mol [expr [molinfo top get numreps] - 1]]
      set ILStools::volreps($mol,volid) 0
    } else {
      #update old rep:
      set rep [mol repindex $mol $ILStools::volreps($mol,repname)]
      incr ILStools::volreps($mol,volid)
      mol modstyle $rep $mol Isosurface 0.5 $ILStools::volreps($mol,volid) 0 0 1
    }
  }
}



proc ILStools::dialog_getdestfile {} {
   variable w
   set newfile [tk_getSaveFile -title "Save volmap as..." -parent $w \
                   -defaultextension .dx -initialdir [pwd] \
                   -filetypes {{"DX File" .dx} {"All files" *}}]
   if {[string length $newfile] > 0} {
      set newfile [relativepath $newfile]
      set ILStools::mapfilename $newfile
      set ILStools::output_destination "file"
   }
}

proc ILStools::dialog_getprmfile {} {
   variable w
   set newfile [tk_getOpenFile -title "Load parameter file..." -parent $w \
                   -defaultextension .par -initialdir [pwd] \
                   -filetypes {{"CHARMM parameter file" {par*.inp .par .prm .rtf}} {"All files" *}}]
   if {[string length $newfile] > 0} {
      set newfile [relativepath $newfile]
      lappend ILStools::prmfiles $newfile
   }
}

proc ILStools::dialog_getrunscript {} {
   variable w
   set newfile [tk_getSaveFile -title "Save runscript as..." -parent $w \
                   -defaultextension .dx -initialfile $ILStools::runscript \
                   -initialdir [pwd] \
                   -filetypes {{"TCL File" .tcl} {"All files" *}}]
   if {[string length $newfile] > 0} {
      set newfile [relativepath $newfile]
      set ILStools::runscript $newfile
      ILStools::write_input
   }
}


# If the directory path of $file equals pwd then remove the
# path from the file name. If the path is the directory above
# pwd then replace the path by "../"
proc ILStools::relativepath {file} {
   if {[file dirname $file] == [pwd]} {
      return [file tail $file]
   } elseif {[file dirname $file] == [file dirname [pwd]]} {
      return [file join .. [file tail $file]]
   }
   return $file
}


##################################################################################
#
#  The following procs were written by Jordi Cohen:
#  Functions for getting VDW parameters from a list of charmm parameter 
#  files, for use with the ligand and slow ligand map types. They should be 
#  supplanted by the use of the readcharmmpar package, and also needs to be 
#  made safer (i.e. reentrant). But for now, this works well.
#
##################################################################################


proc ILStools::readcharmmparams {parfiles} {
   variable nonbonded_table 
   variable nonbonded_wildcard_list {}
   array unset nonbonded_table 

   if ![llength $parfiles] {
      set parfiles [list [file join $env(CHARMMPARDIR) par_all27_prot_lipid_na.inp]]
   }
   
   foreach filename $parfiles {
      set file [open $filename "r"]

      #Find start of NONBONDED section
      while {[gets $file line] >= 0} {
         if {[lindex [split $line] 0] == "NONBONDED"} {break}
      }
  
      #Read NONBONDED params
      while {[gets $file line] >= 0} {
         if {[lindex [split $line] 0] == "HBOND"} break
         if {[lindex [split $line] 0] == "END"} break
         if {[lindex [split $line] 0] == "BONDS"} break
         if {[lindex [split $line] 0] == "IMPROPER"} break
         if {[lindex [split $line] 0] == "ANGLES"} break
         if {[lindex [split $line] 0] == "DIHEDRALS"} break
         
         if {[scan $line "%s %*f %f %f" type epsilon rmin] == 3} {
            if {[string index $line 0] == "!"} {
               set type [string range $type 1 end]
               if [string match "*\[%/*]*" $type] {
                  set replaceindex [string first "%" $type]
                  if {$replaceindex >= 0} {set type [string replace $type $replaceindex $replaceindex "?"]}
                  #puts "WILDCARD $type $epsilon $rmin"
                  set nonbonded_wildcard_list [linsert $nonbonded_wildcard_list 0 "$epsilon $rmin"]
                  set nonbonded_wildcard_list [linsert $nonbonded_wildcard_list 0 $type]
               }
            } else {
               #puts "$type $epsilon $rmin"
               lappend nonbonded_list $type
               lappend nonbonded_list "$epsilon $rmin"
            }
         }
      }
      
      close $file
   }
   
   array unset nonbonded_table
   array unset nonbonded_wc_table
   array set nonbonded_table $nonbonded_list
  
  #puts  $nonbonded_wildcard_list

}




proc ILStools::assigncharmmparams {{molid top}} {
   variable nonbonded_table
   variable nonbonded_wildcard_list
   set atomtypes [[atomselect $molid all] get type]
   
   set atomradii {}
   set atomepsilon {}
   set atomnotfound {}
   
   foreach type $atomtypes {
      if [catch {
         lappend atomradii [lindex $nonbonded_table($type) 1]
         lappend atomepsilon [lindex $nonbonded_table($type) 0]
      }] {
         set foundmatch false 
         foreach {pattern data} $nonbonded_wildcard_list {
            if [string match $pattern $type] {
               lappend atomradii [lindex $data 1]
               lappend atomepsilon [lindex $data 0]
               set foundmatch true
               break
            }
         }
         
         if !$foundmatch {
            lappend atomradii 0.
            lappend atomepsilon 0.
            lappend atomnotfound $type
         }
      } 
   }
   
   [atomselect $molid all] set radius $atomradii
   [atomselect $molid all] set occupancy $atomepsilon

   if {[llength $atomnotfound] > 0} {
      set atomnotfound [lsort -unique $atomnotfound]
      foreach type $atomnotfound {
         vmdcon -warn "Could not find parameters for atom type $type"
      }
      return 0
   }
   
   return 1
}


