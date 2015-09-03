#
# Solvate plugin - generate a water box and add solute
#
# $Id: solvate.tcl,v 1.62 2015/04/12 07:06:47 johns Exp $
#
# generate water block coordinates (VMD)
# replicate water block (psfgen)
# combine water block and solute in psfgen (psfgen)
# Create pdb with just the water you want (VMD)
# merge the solute and the cutout water (psfen)
#
# Changes since version 1.0:
#   Fixed a bug in the water overlap code which left waters close to the
#     solute.  Never figured out what was wrong; probably just using the wrong
#     coordinates.
#
#   Added a sanity check to make sure that solvate didn't leave any waters 
#     near the solute.
#
#   Added experimental support for plugin-based I/O in psfgen, in support of
#     very large structures
#
# TODO?
# Seperate command-line parsing code from solvation code.  solvate{} should
# still handle the command line, and possibly load the PSF and PDB into VMD.
# A seperate proc should perform the actual solvation, accepting a VMD molid
# (rather than filenames) and other solvation parameters.
# 
# The command line could include a new -molid option (other options should
# remain unchanged to preserve backward-compatability with old scripts) and
# file paths can be removed entirely from GUI in favor of a drop-down menu
# of VMD molecules. In addition to the GUI improvement, these changes will
# allow users to use solvate with other file format combinations that
# provide the same information as the PSF/PDB combination.

package require psfgen 1.5
package provide solvate 1.7

proc solvate_usage { } {
  vmdcon -info "Usage: solvate <psffile> <pdbfile> <option1?> <option2?>..."
  vmdcon -info "Usage: solvate <option1?> <option2?>...  to just create a water box" 
  vmdcon -info "Options:"
  vmdcon -info "    -o <output prefix> (data will be written to output.psf/output.pdb)"
  vmdcon -info "    -s <segid prefix> (should be either one or two letters; default WT)"
  vmdcon -info "    -b <boundary> (minimum distance between water and solute, default 2.4)"
  vmdcon -info "    -minmax {{xmin ymin zmin} {xmax ymax zmax}}"
  vmdcon -info "    -rotate (rotate molecule to minimize water volume)"
  vmdcon -info "    -rotsel <selection> (selection of atoms to check for rotation)"
  vmdcon -info "    -rotinc <increment> (degree increment for rotation)"
  vmdcon -info "    -t <pad in all directions> (override with any of the following)"
  vmdcon -info "    -x <pad negative x>"
  vmdcon -info "    -y <pad negative y>"
  vmdcon -info "    -z <pad negative z>"
  vmdcon -info "    +x <pad positive x>"
  vmdcon -info "    +y <pad positive y>"
  vmdcon -info "    +z <pad positive z>"
  vmdcon -info "    The following options allow the use of solvent other than water:"
  vmdcon -info "      -spsf <solventpsf> (PSF file for nonstandard solvent)"
  vmdcon -info "      -spdb <solventpdb> (PDB file for nonstandard solvent)"
  vmdcon -info "      -stop <solventtop> (Topology file for nonstandard solvent)"
  vmdcon -info "      -ws <size> (Box length for nonstandard solvent)"
  vmdcon -info "      -ks <keyatom> (Atom occuring once per residue for nonstandard solvent)"
  error ""
}

proc solvate {args} {
  global errorInfo errorCode
  set oldcontext [psfcontext new]  ;# new context
  set errflag [catch { eval solvate_core $args } errMsg]
  set savedInfo $errorInfo
  set savedCode $errorCode
  psfcontext $oldcontext delete  ;# revert to old context
  if $errflag { error $errMsg $savedInfo $savedCode }
}

proc solvate_core {args} {
  global env 
  global bounds
  set jsinput  0
  set jsoutput 0

  set fullargs $args

# Set some defaults

  # PSF and PDB files, and other info, of the solvent box
  set solventpsf "$env(SOLVATEDIR)/wat.psf"
  set solventpdb "$env(SOLVATEDIR)/wat.pdb"
  set solventtop "$env(SOLVATEDIR)/wat.top"
  set watsize 65.4195 ;# side length of the solvent box
  set keysel "name OH2" ;# name of a key atom that occurs once per residue

 
  # Print usage information if no arguments are given
  if { ![llength $args] } {
    solvate_usage
  }

  # The first argument that starts with a "-" marks the start of the options.
  # Arguments preceding it, if any, must be the psf and pdb files.
  set arg0 [lindex $args 0]


  if { [llength $args] >= 1 && [string range $arg0 0 0] != "-" } {
    if { [string match "*\.js" [lindex $args 0]] } {
      set jsinput 1
      set psffile [lindex $args 0]
      set pdbfile [lindex $args 0]
      set args [lrange $args 1 end]
    } elseif { [llength $args] >= 2 && [string range $arg0 0 0] != "-" } {
      set psffile [lindex $args 0]
      set pdbfile [lindex $args 1]
      set args [lrange $args 2 end]
    }
  }

  # Toggle the rotate flag if present
  set rot [lsearch $args "-rotate"]
  if {$rot != -1} {
    set rotate 1
    set args [lreplace $args $rot $rot]
  } else {
    set rotate 0
  }

  set rotselind [lsearch $args "-rotsel"]
  if {$rotselind != -1} {
    set rotsel [lindex $args [expr $rotselind + 1]]
    set args [lreplace $args $rotselind [expr $rotselind + 1]]
  } else {
    set rotsel "all"
  }
  
  set rotincind [lsearch $args "-rotinc"]
  if {$rotincind != -1} {
    set rotinc [lindex $args [expr $rotincind + 1]]
    set args [lreplace $args $rotincind [expr $rotincind + 1]]
    set rotinc [expr 360 / $rotinc]
  } else {
    set rotinc 36
  }
 
  foreach elem { -b +x +y +z -x -y -z -minmax -t -o -spsf -spdb -stop -ws -ks} {
    set bounds($elem) 0
  }
  set bounds(-s) WT
  set bounds(-b) 2.4

  set n [llength $args]
  # check for even number of args
  if { [expr fmod($n,2)] } { solvate_usage }
    
  #
  # Get all command line options
  #
  for { set i 0 } { $i < $n } { incr i 2 } {
    set key [lindex $args $i]
    set val [lindex $args [expr $i + 1]]
    if { ! [info exists bounds($key)] } {
      solvate_usage 
    }
    set cmdline($key) $val 
  }

  # Get a nonstandard solvent box, if specified
  if { [info exists cmdline(-spsf)] } {
    set solventpsf $cmdline(-spsf)
  }

  if { [info exists cmdline(-spdb)] } {
    set solventpdb $cmdline(-spdb)
  }

  if { [info exists cmdline(-stop)] } {
    set solventtop $cmdline(-stop)
  }
  if { [info exists cmdline(-ws)] } {
    set watsize $cmdline(-ws)
  }
  if { [info exists cmdline(-ks)] } {
    set keysel $cmdline(-ks)
  }

  # 
  # Get minmax if specified, or use minmax of solute
  #

  # 
  # If -t was specified, use it for all pads
  #
  if { [info exists cmdline(-t)] } {
    foreach elem { -x -y -z +x +y +z } {
      set bounds($elem) $cmdline(-t)
    }
  }

  # 
  # Fill in all other specified options
  #  
  set outputname solvate
  if { [info exists cmdline(-o)] } {
    set outputname $cmdline(-o)

    # check for ".js" format output and override normal
    # psf/pdb pair behavior if detected
    if { [string match "*\.js" $outputname] } {
      vmdcon -info "Using js file format for output"
      set jsoutput 1
      # remove the .js suffix
      set outputname [string range $outputname 0 end-3]
    }
  }

  #Open and use a logfile
  set logfile [open "$outputname.log" w]
  puts $logfile "Running solvate with arguments: $fullargs"

  # If the rotate flag is present, rotate the molecule
  # Note that rotate is meaningless if we're doing water box only
  if {$rotate == 1 && [info exists pdbfile]} {
    ::Solvate::rotate_save_water $pdbfile $rotsel $rotinc $logfile
  }

  if { [info exists cmdline(-minmax) ] } {
    set bounds(-minmax) $cmdline(-minmax)
  } else {
    if { [info exists psffile] } {  
      if { $jsinput } {
        if {$rotate == 0} {
          mol new $psffile waitfor all
        } else {
          mol new $pdbfile-rotated-tmp.pdb waitfor all
        }
      } else {
        if {$rotate == 0} {
          mol new $psffile waitfor all
          mol addfile $pdbfile waitfor all
        } else {
          mol new $psffile waitfor all
          mol addfile $pdbfile-rotated-tmp.pdb waitfor all
        }
      }

      if {[molinfo top get numframes] == 0} {
        error "Couldn't load psf/pdb files!"
        return
      }
      set sel [atomselect top all]
      set bounds(-minmax) [measure minmax $sel]
      mol delete top
    } else {
      error "No psf/pdb, so minmax must be specified."
    }
  }

  foreach elem [array names cmdline] {
    set bounds($elem) $cmdline($elem)
  }

  # convert segment prefix to upper case to plase psfgen - Josh Vermaas
  set $bounds(-s) [string toupper $bounds(-s)]

  set env(SOLVATEPREFIX) $bounds(-s)
  set prefix $bounds(-s)

  foreach {min max} $bounds(-minmax) {} 
  set min [vecsub $min [list $bounds(-x) $bounds(-y) $bounds(-z)]]
  set max [vecadd $max [list $bounds(+x) $bounds(+y) $bounds(+z)]]


  #
  # generate combined psf/pdb containing solute and one replica of water
  # VMD can't do multi-molecule atom selections...
  #
  vmdcon -info  "generating solute plus one replica of water..."
  puts $logfile "generating solute plus one replica of water..."
  if { [info exists psffile] } {
    if { $jsinput } {
      # need to eliminate redundancies here
      # readmol js $psffile
      if {$rotate == 0} {
        readmol js $pdbfile
      } else {
        readmol js $pdbfile-rotated-tmp.js
        file delete $pdbfile-rotated-tmp.js
      }
    } else {
      readpsf $psffile
      if {$rotate == 0} {
        coordpdb $pdbfile
      } else {
        coordpdb $pdbfile-rotated-tmp.pdb
        file delete $pdbfile-rotated-tmp.pdb
      }
    }
  }

  readpsf $solventpsf
  coordpdb $solventpdb

  if { $jsoutput } {
    writemol js combine.js
  } else {
    writepsf combine.psf
    writepdb combine.pdb
  }
 
  delatom QQQ


  #
  # Read combined structure back in and generate a new psf/pdb file with just
  # the waters we want.
  #
  if { $jsoutput } {
    mol new combine.js waitfor all
    file delete combine.js
  } else {
    mol new combine.psf waitfor all
    mol addfile combine.pdb waitfor all
    file delete combine.psf
    file delete combine.pdb
  }

  vmdcon -info  "generating water residue lists..."
  puts $logfile "generating water residue lists..."
  set wat [atomselect top "segid QQQ"]
  set wat_unique_res [lsort -unique -integer [$wat get resid]]
  set wat_unique_resname [list]

  # check to see if we have a single or multiple solvent residue types
  set wat_resnames [lsort -unique [$wat get resname]]
  if { [llength $wat_resnames] == 1 } {
    vmdcon -info  "single water residue type, creating residue list..."
    puts $logfile "single water residue type, creating residue list..."

    set cnt [llength $wat_unique_res]
    for { set i 0 } { $i < $cnt } { incr i } {
      lappend wat_unique_resname $wat_resnames
    }
  } else {
    #
    # XXX This step is VERY SLOW on huge structures and needs redesigning
    #     The loop over residues yields quadratic computational complexity,
    #     so it runs excruciatingly slow on 100 million atom structures.
    #
    vmdcon -info  "multiple water residue types, computing residue lists..."
    puts $logfile "multiple water residue types, computing residue lists..."
    if { [llength $wat_unique_res] > 1000000 } {
      vmdcon -info  "Warning: this step may take a long time..."
      puts $logfile "Warning: this step may take a long time..."
    }

    foreach resid $wat_unique_res {
      set tmpsel [atomselect top "segid QQQ and resid $resid"]
      lappend wat_unique_resname [lindex [$tmpsel get resname] 0]
      $tmpsel delete
    }
  }


  #
  # Extract info about where to put the water replicas
  #
  set rwat $bounds(-b)
  foreach {xmin ymin zmin} $min {}
  foreach {xmax ymax zmax} $max {}

  set dx [expr $xmax - $xmin]
  set dy [expr $ymax - $ymin]
  set dz [expr $zmax - $zmin]

  set nx [expr int(($dx+2*$rwat)/$watsize) + 1]
  set ny [expr int(($dy+2*$rwat)/$watsize) + 1]
  set nz [expr int(($dz+2*$rwat)/$watsize) + 1]
  set numsegs [expr $nx * $ny * $nz]
  vmdcon -info  "replicating $numsegs water segments, $nx by $ny by $nz"
  puts $logfile "replicating $numsegs water segments, $nx by $ny by $nz"


  #
  # Check that we won't run out of segment name characters, and switch to 
  # using hexadecimal or alphanumeric naming schemes in cases where decimal
  # numbered segnames won't fit into the field width.
  #
  set segstrcheck "$prefix$numsegs"
  set usehex 0
  set usealphanum 0
  if { [string length $segstrcheck] > 4 } {
    set usehex 1
  } 

  set nstrcheck [string toupper [format "%x" $numsegs]]
  set segstrcheck "$prefix$nstrcheck"
  if { [string length $segstrcheck] > 4 } {
    set usehex 0
    set usealphanum 1
  }
 
  # 
  # Warn the user about the choice of segname generation scheme if we 
  # weren't able to use decimal numbers appended to the prefix. 
  # 
  if { $usehex } {
    vmdcon -warn "Warning: decimal naming would overrun segname field"
    vmdcon -warn "Warning: using hexadecimal segnames instead..."
    puts $logfile "Warning: decimal naming would overrun segname field"
    puts $logfile "Warning: using hexadecimal segnames instead..."
  }
  if { $usealphanum } {
    vmdcon -warn "Warning: decimal or hex naming would overrun segname field"
    vmdcon -warn "Warning: using alphanumeric segnames instead..."
    puts $logfile "Warning: decimal or hex naming would overrun segname field"
    puts $logfile "Warning: using alphanumeric segnames instead..."
  }


  #
  # generate replicas
  #
  topology $solventtop
  set n 0
  set seglist {}

  set watres [$wat get resid]
  set watname [$wat get name] 

  set solute [atomselect top "not segid QQQ"]
  set solutebox [measure minmax $solute]
  $solute delete

  set minx [expr [lindex $solutebox 0 0] - $rwat]
  set miny [expr [lindex $solutebox 0 1] - $rwat]
  set minz [expr [lindex $solutebox 0 2] - $rwat]
  set maxx [expr [lindex $solutebox 1 0] + $rwat]
  set maxy [expr [lindex $solutebox 1 1] + $rwat]
  set maxz [expr [lindex $solutebox 1 2] + $rwat]

  set startreplication [clock seconds]
  set lastreplicamsg  [clock seconds]
  for { set i 0 } { $i < $nx } { incr i } {
    set movex    [expr $xmin + $i * $watsize]
    set movexmax [expr $movex + $watsize]
    set xoverlap 1
    if {$movex > $maxx || $movexmax < $minx} {
      set xoverlap 0
    }
    for { set j 0 } { $j < $ny } { incr j } {
      set movey    [expr $ymin + $j * $watsize]
      set moveymax [expr $movey + $watsize]
      set yoverlap 1
      if {$movey > $maxy || $moveymax < $miny} {
        set yoverlap 0
      }
      for { set k 0 } { $k < $nz } { incr k } {
        set movez [expr $zmin + $k * $watsize]
        set movezmax [expr $movez + $watsize]
        set zoverlap 1
        if {$movez > $maxz || $movezmax < $minz} {
          set zoverlap 0
        }

        # vmdcon -info "coords: $movex $movey $movez  $movexmax $moveymax $movezmax,  $xoverlap $yoverlap $zoverlap"

        set vec [list $movex $movey $movez]
        $wat moveby $vec 

        # Create new water replica... 
        incr n
        if { $usehex } {
          set nstr [string toupper [format "%x" $n]]
        } elseif { $usealphanum } {
          set nstr [format "%c%c%c" [expr $n/26/26 + 65] [expr $n/26%26 + 65] [expr $n%26 + 65]] 
        } else {
          set nstr $n
        }
        segment ${prefix}$nstr {
          first NONE
          last NONE
          foreach res $wat_unique_res resname $wat_unique_resname {
            residue $res $resname
          }
        }

        # XXX this step takes about 35% of the runtime for one replica
        #     when generating large systems
        lappend seglist ${prefix}$nstr
        foreach resid $watres name $watname pos [$wat get {x y z}] {
          coord ${prefix}$nstr $resid $name $pos
        }

        # delete waters that either overlap, or are outside the box
        if { $xoverlap && $yoverlap && $zoverlap } {
          # find and delete overlapping waters and those outside the box.
          # XXX this step can take 65% of the runtime for one replica
          #     when generating large systems.  Most of this time is due to
          #     the 'within' selection, but about 10% of the overall runtime
          #     is due to the six tests for the min/max box dimensions.
          #     We only do the costly overlap test when we know that the 
          #     bounding box containing the solvent and 
          #     the bounding box containing the solute have some overlap.
          set sel [atomselect top "segid QQQ and $keysel and same residue as \
                   (x < $xmin or x > $xmax or \
                    y < $ymin or y > $ymax or \
                    z < $zmin or z > $zmax or \
                    within $rwat of (not segid QQQ))"]
        } else {
          # find and delete waters outside the box
          set sel [atomselect top "segid QQQ and $keysel and same residue as \
                   (x < $xmin or x > $xmax or \
                    y < $ymin or y > $ymax or \
                    z < $zmin or z > $zmax)"]
        }
        foreach resid [$sel get resid] {
          # Use catch because the atom might have already been deleted 
          catch { delatom ${prefix}$nstr $resid }
        }
        unset upproc_var_$sel 
    
        $wat moveby [vecinvert $vec] 

        # generate progress status messages for long running jobs
        set replicationelapsed [expr [clock seconds] - $startreplication]
        set lastreplicaelapsed [expr [clock seconds] - $lastreplicamsg]
        if { $lastreplicaelapsed > 30 } {
          set lastreplicamsg [clock seconds]
          set runmin [expr $replicationelapsed / 60.0]
          set totalmin [expr $replicationelapsed * $numsegs / ($n * 60.0)]
          set finishmin [expr $totalmin - $runmin]
          if { $finishmin > 1 } {
            set str [format "solvate) elapsed time %.2f min, finished in %.0f min" $runmin $finishmin]
            vmdcon -info $str
          }
        }  
      } 
    }
  }
  
  if { $jsoutput } {
    writemol js $outputname.js
  } else {
    writepsf $outputname.psf
    writepdb $outputname.pdb
  }

  # delete the current psfgen context before we load the newly 
  # generated files, otherwise we'll end up temporarily using over twice as
  # much memory until we return from this routine.
  resetpsf
  mol delete top 

  # Load the final solvated structure
  if { [catch {
  if { $jsoutput } {
    mol new $outputname.js waitfor all
  } else {  
    mol new     $outputname.psf waitfor all
    mol addfile $outputname.pdb waitfor all
  } } errorcode] } {
    error "Couldn't load output files. Please make sure that you have enough disk space, and that you didn't set solvent box boundaries that would produce zero placed waters."
  }

  # write the bounding box information to the pdb
  set sel [atomselect top all]
  set bbox [vecsub $max $min]
  molinfo top set a [lindex $bbox 0]
  molinfo top set b [lindex $bbox 1]
  molinfo top set c [lindex $bbox 2]
  if {$jsoutput} {
    $sel writejs $outputname.js
  } else {
    $sel writepdb $outputname.pdb
  }
  $sel delete


  # Test to make sure we didn't miss any waters.  Add a fudge factor 
  # of sqrt(3 * .001^2) to the distance check because of limited precision
  # in the PDB file.
  if { $jsoutput && [molinfo top get numatoms] > 4000000 } { 
    vmdcon -info "Skipping extra structure overlap safety check..."
  } else {
    vmdcon -info "Extra structure overlap safety check..."
    # If we use too small of a radius reduction, we are subject to 
    # floating point rounding in text formats such as PDB that are
    # only accurate to 0.001 A.
    # Another problem with decreasing rwat by a fixed value is that
    # it will not behave correctly if the user uses a different 
    # solvent that requires a different probe/padding radius.
    set rwat [expr $rwat * 0.999]
    # set rwat [expr $rwat - 0.001732]
    set sel [atomselect top "segid $seglist and within $rwat of (not segid $seglist)"]
    set num [$sel num]
    $sel delete
    if { $num != 0 } {
      vmdcon -err "Found $num water atoms near the solute!  Please report this bug to"
      vmdcon -err "vmd@ks.uiuc.edu, including, if possible, your psf and pdb file."
      error "Solvate failed."  
    }
  }

  vmdcon -info "Solvate completed successfully."
  puts $logfile "Solvate completed successfully."

  close $logfile
  return [list $min $max]
}
  
proc solvategui {} {
  return [::Solvate::solvate_gui]
}
# XXX hotfix for release. multiseq contains a scrolledframe
# convenience widget, that needs to be moved to a generic package.
package require multiseqdialog 1.1
 
namespace eval ::Solvate:: {
  namespace export solvate_gui
  namespace import ::MultiSeqDialog::scrolledframe::scrolledframe

  variable w
  variable psffile
  variable pdbfile
  variable waterbox
  variable outprefix
  variable segid
  variable boundary
  variable min
  variable max
  variable use_mol_box
  variable minpad
  variable maxpad
  variable rotate
  variable rotsel
  variable rotinc
  set rotsel "all"
  set rotinc 10
  variable usealtsolv
  variable altsolvpdb
  variable altsolvpsf
  variable altsolvtop
  variable altsolvws
  variable altsolvks
  set usealtsolv 0
}

proc ::Solvate::solvate_gui {} {
  variable w
  ::Solvate::init_gui

  if { [winfo exists .solvategui] } {
    wm deiconify .solvategui
    return
  }
  set w [toplevel ".solvategui"]
  wm title $w "Solvate"
  wm resizable $w 0 1

  #scrolledframe $w.f -width 450 -height 500  -yscroll "$w.s set"
  #scrollbar $w.s -command "$w.f yview"

  #grid $w.f -row 0 -column 0 -sticky nsew
  #grid $w.s -row 0 -column 1 -sticky ns
  #grid rowconfigure $w 0 -weight 1
  #grid columnconfigure $w 0 -weight 1
  set f $w

  frame $f.input
  grid [label $f.input.label -text "Input:"] \
    -row 0 -column 0 -columnspan 1 -sticky w
  grid [checkbutton $f.input.water_button -text "Waterbox Only" \
    -variable ::Solvate::waterbox] -row 0 -column 1 -columnspan 1 -sticky w
  grid [label $f.input.psflabel -text "PSF: "] \
    -row 1 -column 0 -sticky w
  grid [entry $f.input.psfpath -width 30 -textvariable ::Solvate::psffile] \
    -row 1 -column 1 -sticky ew
  grid [button $f.input.psfbutton -text "Browse" \
    -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} { set ::Solvate::psffile $tempfile }
    }] -row 1 -column 2 -sticky w
  grid [label $f.input.pdblabel -text "PDB: "] \
    -row 2 -column 0 -sticky w
  grid [entry $f.input.pdbpath -width 30 -textvariable ::Solvate::pdbfile] \
    -row 2 -column 1 -sticky ew
  grid [button $f.input.pdbbutton -text "Browse" \
    -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} { set ::Solvate::pdbfile $tempfile }
    }] -row 2 -column 2 -sticky w
  grid [checkbutton $f.input.rotate_button -text "Rotate to minimize volume" \
    -variable ::Solvate::rotate] -row 3 -column 0 -columnspan 2 -sticky w
  grid [label $f.input.inclabel -text "Rotation Increment (deg): "] \
    -row 3 -column 1 -sticky e
  grid [entry $f.input.rotinc -width 8 -textvariable ::Solvate::rotinc] \
    -row 3 -column 2 -sticky w
  grid [label $f.input.sellabel -text "Selection for Rotation: "] \
   -row 4 -column 0 -sticky w
  grid [entry $f.input.rotsel -width 20 -textvariable ::Solvate::rotsel] \
    -row 4 -column 1 -sticky w
  grid columnconfigure $f.input 1 -weight 1
  pack $f.input -side top -padx 10 -pady 3 -expand 1 -fill x

  frame $f.output
  grid [label $f.output.label -text "Output:"] \
    -row 0 -column 0 -columnspan 1 -sticky w
  grid [entry $f.output.outpath -width 30 -textvariable ::Solvate::outprefix] \
    -row 0 -column 1 -columnspan 2 -sticky ew
  grid [button $f.output.outbutton -text "Browse" \
    -command {
      set tempfile [tk_getOpenFile]
      if {![string equal $tempfile ""]} { set ::Solvate::outprefix $tempfile }
    }] -row 0 -column 3 -sticky w
  grid columnconfigure $f.output 0 -weight 1
  pack $f.output -side top -padx 10 -pady 3 -expand 1 -fill x

  frame $f.seg
  grid [label $f.seg.seglabel -text "Segment ID Prefix:"] \
    -row 0 -column 0 -sticky w
  grid [entry $f.seg.segentry -width 8 -textvariable ::Solvate::segid] \
    -row 0 -column 1 -sticky ew
  grid [label $f.seg.boundlabel -text "Boundary: "] \
    -row 1 -column 0 -sticky w
  grid [entry $f.seg.boundentry -width 8 -textvariable ::Solvate::boundary] \
    -row 1 -column 1 -sticky ew
  grid columnconfigure $f.seg 1 -weight 1
  pack $f.seg -side top -padx 10 -pady 1 -expand 1 -fill x

  frame $f.minmax
  grid [label $f.minmax.label -text "Box Size:"] \
    -row 0 -column 0 -columnspan 4 -sticky w
  grid [checkbutton $f.minmax.boxbutton -text "Use Molecule Dimensions" \
    -variable ::Solvate::use_mol_box] -row 0 -column 4 -columnspan 3 -sticky w
  grid [label $f.minmax.minlabel -text "Min: "] -row 1 -column 0 -sticky w
  grid [label $f.minmax.xminlabel -text "x: "] -row 1 -column 1 -sticky w
  grid [entry $f.minmax.xminentry -width 6 -textvar ::Solvate::min(x)] \
    -row 1 -column 2 -sticky ew
  grid [label $f.minmax.yminlabel -text "y: "] -row 1 -column 3 -sticky w
  grid [entry $f.minmax.yminentry -width 6 -textvar ::Solvate::min(y)] \
    -row 1 -column 4 -sticky ew
  grid [label $f.minmax.zminlabel -text "z: "] -row 1 -column 5 -sticky w
  grid [entry $f.minmax.zminentry -width 6 -textvar ::Solvate::min(z)] \
    -row 1 -column 6 -sticky ew
  grid [label $f.minmax.maxlabel -text "Max: "] -row 2 -column 0 -sticky w
  grid [label $f.minmax.xmaxlabel -text "x: "] -row 2 -column 1 -sticky w
  grid [entry $f.minmax.xmaxentry -width 6 -textvar ::Solvate::max(x)] \
    -row 2 -column 2 -sticky ew
  grid [label $f.minmax.ymaxlabel -text "y: "] -row 2 -column 3 -sticky w
  grid [entry $f.minmax.ymaxentry -width 6 -textvar ::Solvate::max(y)] \
    -row 2 -column 4 -sticky ew
  grid [label $f.minmax.zmaxlabel -text "z: "] -row 2 -column 5 -sticky w
  grid [entry $f.minmax.zmaxentry -width 6 -textvar ::Solvate::max(z)] \
    -row 2 -column 6 -sticky ew
  ::Solvate::waterbox_state
  grid columnconfigure $f.minmax {2 4 6} -weight 1
  pack $f.minmax -side top -padx 10 -pady 1 -expand 1 -fill x

  frame $f.padding
  grid [label $f.padding.label -text "Box Padding:"] \
    -row 0 -column 0 -columnspan 7 -sticky w
  grid [label $f.padding.minlabel -text "Min: "] -row 1 -column 0 -sticky w
  grid [label $f.padding.xminlabel -text "x: "] -row 1 -column 1 -sticky w
  grid [entry $f.padding.xminentry -width 6 -textvar ::Solvate::minpad(x)] \
    -row 1 -column 2 -sticky ew
  grid [label $f.padding.yminlabel -text "y: "] -row 1 -column 3 -sticky w
  grid [entry $f.padding.yminentry -width 6 -textvar ::Solvate::minpad(y)] \
    -row 1 -column 4 -sticky ew
  grid [label $f.padding.zminlabel -text "z: "] -row 1 -column 5 -sticky w
  grid [entry $f.padding.zminentry -width 6 -textvar ::Solvate::minpad(z)] \
    -row 1 -column 6 -sticky ew
  grid [label $f.padding.maxlabel -text "Max: "] -row 2 -column 0 -sticky w
  grid [label $f.padding.xmaxlabel -text "x: "] -row 2 -column 1 -sticky w
  grid [entry $f.padding.xmaxentry -width 6 -textvar ::Solvate::maxpad(x)] \
    -row 2 -column 2 -sticky ew
  grid [label $f.padding.ymaxlabel -text "y: "] -row 2 -column 3 -sticky w
  grid [entry $f.padding.ymaxentry -width 6 -textvar ::Solvate::maxpad(y)] \
    -row 2 -column 4 -sticky ew
  grid [label $f.padding.zmaxlabel -text "z: "] -row 2 -column 5 -sticky w
  grid [entry $f.padding.zmaxentry -width 6 -textvar ::Solvate::maxpad(z)] \
    -row 2 -column 6 -sticky ew
  grid columnconfigure $f.padding {2 4 6} -weight 1
  pack $f.padding -side top -padx 10 -pady 1 -expand 1 -fill x

  frame $f.altsolv
  grid [checkbutton $f.altsolv.usealtsolv -text "Use nonstandard solvent" -variable ::Solvate::usealtsolv] \
    -row 0 -column 0 -columnspan 7 -sticky w
  grid [label $f.altsolv.pdblabel -text "Solvent box PDB: "] -row 1 -column 0 -columnspan 3 -sticky w
  grid [entry $f.altsolv.pdbentry -width 20 -textvar ::Solvate::altsolvpdb] \
    -row 1 -column 3 -columnspan 4 -sticky ew
  grid [label $f.altsolv.psflabel -text "Solvent box PSF: "] -row 2 -column 0 -columnspan 3 -sticky w
  grid [entry $f.altsolv.psfentry -width 20 -textvar ::Solvate::altsolvpsf] \
    -row 2 -column 3 -columnspan 4 -sticky ew
  grid [label $f.altsolv.toplabel -text "Solvent box topology: "] -row 3 -column 0 -columnspan 3 -sticky w
  grid [entry $f.altsolv.topentry -width 20 -textvar ::Solvate::altsolvtop] \
    -row 3 -column 3 -columnspan 4 -sticky ew
  grid [label $f.altsolv.sizelabel -text "Solvent box side length: "] -row 4 -column 0 -columnspan 3 -sticky w
  grid [entry $f.altsolv.sizeentry -width 20 -textvar ::Solvate::altsolvws] \
    -row 4 -column 3 -columnspan 4 -sticky ew
  grid [label $f.altsolv.kslabel -text "Solvent box key selection: "] -row 5 -column 0 -columnspan 3 -sticky w
  grid [entry $f.altsolv.ksentry -width 20 -textvar ::Solvate::altsolvks] \
    -row 5 -column 3 -columnspan 4 -sticky ew
  grid columnconfigure $f.altsolv {2 4 6} -weight 1
  pack $f.altsolv -side top -padx 10 -pady 1 -expand 1 -fill x
  ::Solvate::altsolv_state

  pack [button $f.solvate -text "Solvate" -command ::Solvate::run_solvate] \
    -side top -padx 10 -pady 1 -expand 1 -fill x

 
  return $w
}

# Set up variables before opening the GUI
proc ::Solvate::init_gui {} {
  variable psffile
  variable pdbfile
  variable waterbox
  variable outprefix
  variable segid
  variable boundary
  variable min
  variable max
  variable use_mol_box
  variable minpad
  variable maxpad
  variable rotate

  # 
  # Check if the top molecule has both pdb and psf files loaded: if it does,
  # use those as a default; otherwise, leave these fields blank and create a
  # waterbox.
  #
  set psffile {}
  set pdbfile {}
  set waterbox 1
  set use_mol_box 0
  if {[molinfo num] != 0} {
    foreach filename [lindex [molinfo top get filename] 0] \
            filetype [lindex [molinfo top get filetype] 0] {
      if { [string equal $filetype "psf"] } {
        set psffile $filename
      } elseif { [string equal $filetype "pdb"] } {
        set pdbfile $filename
      }
    }
    # Make sure both a pdb and psf are loaded
    if { $psffile == {} || $pdbfile == {} } {
      set psffile {}
      set pdbfile {}
    } else {
      set waterbox 0
      set use_mol_box 1
    }
  }

  set rotate 0 
  set outprefix "solvate"
  set segid "WT"
  set boundary 2.4
  array set minpad {x 0 y 0 z 0}
  array set maxpad {x 0 y 0 z 0}

  # Add traces to the checkboxes, so various widgets can be disabled
  # appropriately
  if {[llength [trace info variable ::Solvate::waterbox]] == 0} {
    trace add variable ::Solvate::waterbox write ::Solvate::waterbox_state
  }
  if {[llength [trace info variable ::Solvate::use_mol_box]] == 0} {
    trace add variable ::Solvate::use_mol_box write ::Solvate::molbox_state
  }
  if {[llength [trace info variable ::Solvate::rotate]] == 0} {
    trace add variable ::Solvate::rotate write ::Solvate::rotate_state
#    set rotate 0
  }
  if {[llength [trace info variable ::Solvate::usealtsolv]] == 0} {
    trace add variable ::Solvate::usealtsolv write ::Solvate::altsolv_state
  }
}

# Run solvate from the GUI. Assembles a command line and passes it to
# solvate
proc ::Solvate::run_solvate {} {
  variable psffile
  variable pdbfile
  variable waterbox
  variable outprefix
  variable segid
  variable boundary
  variable min
  variable max
  variable use_mol_box
  variable minpad
  variable maxpad
  variable rotate
  variable rotsel
  variable rotinc
  variable usealtsolv
  variable altsolvpdb
  variable altsolvpsf
  variable altsolvtop
  variable altsolvws
  variable altsolvks

  set command_line {}

  if { !$waterbox } {
    if { ($psffile == {}) || ($pdbfile == {} ) } {
      vmdcon -err "solvate: need file names"
      return
    }
    append command_line [format "{%s} {%s}" $psffile $pdbfile]
  }

  if { $outprefix == {} } {
    vmdcon -err "solvate: need output filename"
    return
  }
  set command_line [concat $command_line "-o" $outprefix]

  if { $segid == {} } {
    vmdcon -err "solvate: need segid"
    return
  }
  set command_line [concat $command_line "-s" $segid]

  if { !$use_mol_box } {
    if { ![eval ::Solvate::is_number $min(x)] ||
         ![eval ::Solvate::is_number $min(y)] ||
         ![eval ::Solvate::is_number $min(z)] ||
         ![eval ::Solvate::is_number $max(x)] ||
         ![eval ::Solvate::is_number $max(y)] ||
         ![eval ::Solvate::is_number $max(z)] } {
      vmdcon -err "solvate: need numeric minmax"
      return
    }
    set command_line [concat $command_line "-minmax" [list [list \
      [list $min(x) $min(y) $min(z)] [list $max(x) $max(y) $max(z)]]]]
  }

  if { $rotate != 0 } {
    set command_line [concat $command_line "-rotate"]
    set command_line [concat $command_line "-rotsel \"" $rotsel "\""]
    set command_line [concat $command_line "-rotinc " $rotinc]
  }

  if { ![eval ::Solvate::is_number $minpad(x)] ||
       ![eval ::Solvate::is_number $minpad(y)] ||
       ![eval ::Solvate::is_number $minpad(z)] ||
       ![eval ::Solvate::is_number $maxpad(x)] ||
       ![eval ::Solvate::is_number $maxpad(y)] ||
       ![eval ::Solvate::is_number $maxpad(z)] } {
    vmdcon -err "solvate: need numeric padding"
    return
  }
  set command_line [concat $command_line "-x" $minpad(x)]
  set command_line [concat $command_line "-y" $minpad(y)]
  set command_line [concat $command_line "-z" $minpad(z)]
  set command_line [concat $command_line "+x" $maxpad(x)]
  set command_line [concat $command_line "+y" $maxpad(y)]
  set command_line [concat $command_line "+z" $maxpad(z)]

  set command_line [concat $command_line "-b" $boundary]

# Check if we want to use an alternate solvent, and apply it if needed
  if {$usealtsolv} {
    if {$altsolvpdb=="" || $altsolvpsf=="" || $altsolvtop=="" || $altsolvws=="" || $altsolvks==""} {
      error "Missing required information for alternative solvent! Please fill out all fields or uncheck 'Use nonstandard solvent'"
      return
    }
    set command_line [concat $command_line "-spdb" $altsolvpdb]
    set command_line [concat $command_line "-spsf" $altsolvpsf]
    set command_line [concat $command_line "-stop" $altsolvtop]
    set command_line [concat $command_line "-ws" $altsolvws]
    set command_line [concat $command_line "-ks \"$altsolvks\""]
  }


  eval solvate $command_line
}

# Disable or enable widgets according to the current status of the
# "Waterbox Only" checkbutton
proc ::Solvate::waterbox_state {args} {
  variable w
  variable waterbox
  variable use_mol_box

  set f $w.f.scrolled
  # Disable the "Use Molecule Dimensions" button and input file fields
  if {$waterbox} {
    set use_mol_box 0
    if {[winfo exists $f.minmax.boxbutton]} {
      $f.minmax.boxbutton configure -state disabled
    }
    if {[winfo exists $f.input]} {
      $f.input.psfpath configure -state disabled
      $f.input.psfbutton configure -state disabled
      $f.input.pdbpath configure -state disabled
      $f.input.pdbbutton configure -state disabled
    }
  } else {
    if {[winfo exists $f.minmax.boxbutton]} {
      $f.minmax.boxbutton configure -state normal
    }
    if {[winfo exists $f.input]} {
      $f.input.psfpath configure -state normal
      $f.input.psfbutton configure -state normal
      $f.input.pdbpath configure -state normal
      $f.input.pdbbutton configure -state normal
    }
  }

}

# Disable the nonstandard solvent section unless we're using an alternate solvent
proc ::Solvate::altsolv_state {args} {
  variable w
  variable usealtsolv

  set f $w.f.scrolled
  if {!$usealtsolv} {
    if {[winfo exists $f.altsolv]} {
      $f.altsolv.pdbentry configure -state disabled
      $f.altsolv.psfentry configure -state disabled
      $f.altsolv.topentry configure -state disabled
      $f.altsolv.sizeentry configure -state disabled
      $f.altsolv.ksentry configure -state disabled
    }
  } else {
    if {[winfo exists $f.altsolv]} {
      $f.altsolv.pdbentry configure -state normal
      $f.altsolv.psfentry configure -state normal
      $f.altsolv.topentry configure -state normal
      $f.altsolv.sizeentry configure -state normal
      $f.altsolv.ksentry configure -state normal
    }
  }

}

# Disable or enable widgets according to the current status of the
# "Use Molecule Dimensions" checkbutton
proc ::Solvate::molbox_state {args} {
  variable w
  variable use_mol_box

  # XXX - TODO: Display the molecule box size 
  # disable the boxsize fields if using molecule dimensions.
  set f $w.f.scrolled
  if {[winfo exists $f.minmax]} {
    if {$use_mol_box} {
      $f.minmax.xminentry configure -state disabled
      $f.minmax.yminentry configure -state disabled
      $f.minmax.zminentry configure -state disabled
      $f.minmax.xmaxentry configure -state disabled
      $f.minmax.ymaxentry configure -state disabled
      $f.minmax.zmaxentry configure -state disabled
    } else {
      $f.minmax.xminentry configure -state normal
      $f.minmax.yminentry configure -state normal
      $f.minmax.zminentry configure -state normal
      $f.minmax.xmaxentry configure -state normal
      $f.minmax.ymaxentry configure -state normal
      $f.minmax.zmaxentry configure -state normal
    }
  }
}

# Disable or enable widgets according to the current status of the
# "Use Molecule Dimensions" checkbutton
proc ::Solvate::rotate_state {args} {
  variable w
  variable rotate

  set f $w.f.scrolled
  if {[winfo exists $f.input]} {
    if {$rotate} {
      $f.input.rotinc configure -state normal
      $f.input.rotsel configure -state normal
    } else {
      $f.input.rotinc configure -state disabled
      $f.input.rotsel configure -state disabled
    }
  }
}

proc ::Solvate::is_number {args} {
  if {[llength $args] != 1} {
    return 0
  }

  set x [lindex $args 0]
  if { ($x == {}) || [catch {expr $x + 0}]} {
    return 0
  } else {
    return 1
  }
}

proc ::Solvate::rotate_save_water {pdbload selection N_rot logfile} {
  global bounds

  vmdcon -info "Loading the structure for rotation..."
  mol new $pdbload waitfor all
  vmdcon -info "done"

  ##################################################
  # Set the center to (0 0 0).
  ##################################################
  set A [atomselect top all]
  set minus_com [vecsub {0.0 0.0 0.0} [measure center $A]]
  $A moveby $minus_com
  ##################################################

  # Set the number of atoms.
  set N [$A num]
  # some error checking
  if {$N <= 0} {
    error "No atoms in the molecule"
  }

  set B [atomselect top $selection]
  set N_B [$B num]
  if {$N_B <= 0} {
    error "need a selection with atoms"
  }

  set tmp [measure minmax $B]
  set L_x [expr [lindex [lindex $tmp 1] 0] - [lindex [lindex $tmp 0] 0] + $bounds(-x) + $bounds(+x)]
  set L_y [expr [lindex [lindex $tmp 1] 1] - [lindex [lindex $tmp 0] 1] + $bounds(-y) + $bounds(+y)] 
  set L_z [expr [lindex [lindex $tmp 1] 2] - [lindex [lindex $tmp 0] 2] + $bounds(-z) + $bounds(+z)]

  set V_0 [expr $L_x*$L_y*$L_z]
  set kV1 0
  set kV2 0
  vmdcon -info "Initial volume is $V_0"

  ##
  ## Find the position (using rotations) corresponding to the smallest volume.
  ##
  vmdcon -info "Rotating the system..."
  set d_phi [expr 360.0/$N_rot]
  set d_theta [expr 360.0/$N_rot]

  for {set k1 1} {$k1 < [expr $N_rot + 1]} {incr k1} {
    $A move [trans axis z $d_phi deg]
    for {set k2 1} {$k2 < [expr $N_rot + 1]} {incr k2} {
      $A move [trans axis x $d_theta deg]

      set tmp [measure minmax $B]
      set L_x [expr [lindex [lindex $tmp 1] 0] - [lindex [lindex $tmp 0] 0] + $bounds(-x) + $bounds(+x)]
      set L_y [expr [lindex [lindex $tmp 1] 1] - [lindex [lindex $tmp 0] 1] + $bounds(-y) + $bounds(+y)]
      set L_z [expr [lindex [lindex $tmp 1] 2] - [lindex [lindex $tmp 0] 2] + $bounds(-z) + $bounds(+z)]
      set V [expr $L_x*$L_y*$L_z]

      if {$V < $V_0} {
        set V_0 $V
        set kV1 $k1
        set kV2 $k2
      }
    }
  }
  vmdcon -info "done"
  vmdcon -info ""

  ##
  ## Make rotations.
  ##
  vmdcon -info "New volume is $V_0"
  puts $logfile "New volume is $V_0"
  $A move [trans axis z [expr $kV1*$d_phi] deg]
  $A move [trans axis x [expr $kV2*$d_theta] deg]
  vmdcon -info "The system was rotated by [expr $kV1*$d_phi] degrees around Z axis and [expr $kV2*$d_theta] degrees around X axis."
  puts $logfile "The system was rotated by [expr $kV1*$d_phi] degrees around Z axis and [expr $kV2*$d_theta] degrees around X axis."
  ##################################################

  $A writepdb $pdbload-rotated-tmp.pdb
}
