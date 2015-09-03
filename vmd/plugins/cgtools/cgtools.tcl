# cgtools is a vmd package that allows conversion to and from coarse grain 
#  representations. See the docs for details on file formats.
#
# $Id: cgtools.tcl,v 1.18 2013/04/15 15:35:30 johns Exp $
#
package provide cgtools 1.2

namespace eval ::cgtools:: {
  # List of currently recognized conversions
  variable convbeads [list] 
  variable idxarray
  variable startposarray
  variable massarray
  namespace export read_db apply_reversal apply_database sasa_LJ_networking 
}

proc ::cgtools::resetconvs {} {
  #Proc to reset the list of conversions currently being held
  variable convbeads
  set convbeads [list]
}

proc ::cgtools::read_db {file} {
  #Read all cg conversions in file and add them to the convbeads list
  variable convbeads

  set infile [open $file "r"]
  
  while {[gets $infile line] >= 0} {
    if {[regexp {^CGBEGIN} $line]} {
      set newbead [read_bead $infile]
      lappend convbeads $newbead
      puts $newbead
    }
  }

  close $infile
}

proc ::cgtools::make_anneal_config {pdb psf par conffile logfile {isRemote 0} } {
  variable idxarray
  variable massarray
  variable startposarray

#  set idxarray [list]
#  set massarray [list]
#  set startposarray [list]

  mol load psf $psf pdb $pdb
  set sel [atomselect top all]

  set min [lindex [measure minmax $sel] 0]
  set max [lindex [measure minmax $sel] 1]
  set center [measure center $sel]
  
  set diffvec [vecsub $max $min]
  set xvec [lindex $diffvec 0]
  set yvec [lindex $diffvec 1]
  set zvec [lindex $diffvec 2]
  set xpme [expr int(pow(2,(int(log($xvec)/log(2)) + 1)))]
  set ypme [expr int(pow(2,(int(log($yvec)/log(2)) + 1)))]
  set zpme [expr int(pow(2,(int(log($zvec)/log(2)) + 1)))]

  set conf [open $conffile "w"]

  #Now, write the file itself
  puts $conf "\# Automatically generated configuration for annealing a revcg structure"
  if { $isRemote } {
     puts $conf "structure          [file tail $psf]"
     puts $conf "coordinates        [file tail $pdb]"
     for {set i 0} {$i < [llength $par] } {incr i} {
        puts $conf "parameters         [file tail [lindex $par $i]]"
     }
  } else {
     puts $conf "structure          $psf"
     puts $conf "coordinates        $pdb"
     for {set i 0} {$i < [llength $par] } {incr i} {
        puts $conf "parameters         [lindex $par $i]"
     }
  }
  puts $conf "set temperature    298"
  puts $conf "set outputname     $conffile"
  puts $conf "firsttimestep      0"
  puts $conf "paraTypeCharmm	    on"
  puts $conf "temperature         298"
  puts $conf "exclude             scaled1-4"
  puts $conf "1-4scaling          1.0"
  puts $conf "cutoff              12."
  puts $conf "switching           on"
  puts $conf "switchdist          10."
  puts $conf "pairlistdist        13.5"
  puts $conf "rigidbonds          water"
  puts $conf "timestep            1.0  "
  puts $conf "nonbondedFreq       2"
  puts $conf "fullElectFrequency  4  "
  puts $conf "stepspercycle       10"
#  puts $conf "langevin            on   "
#  puts $conf "langevinDamping     5    "
#  puts $conf "langevinTemp        298"
#  puts $conf "langevinHydrogen    off  "
  puts $conf "cellBasisVector1    $xvec    0.   0."
  puts $conf "cellBasisVector2     0.   $yvec   0. "
  puts $conf "cellBasisVector3     0.    0   $zvec"
  puts $conf "cellOrigin         $center"
  puts $conf "wrapAll             on"
  puts $conf "PME                 yes"
  puts $conf "PMEGridSizeX        $xpme"
  puts $conf "PMEGridSizeY        $ypme"
  puts $conf "PMEGridSizeZ        $zpme"
  puts $conf "useFlexibleCell       no"
  puts $conf "useConstantArea       no"
  puts $conf "outputName          $conffile"
  puts $conf "restartfreq         500"
  puts $conf "dcdfreq             150"
  puts $conf "xstFreq             200"
  puts $conf "outputEnergies      100"
  puts $conf "outputPressure      100"
  puts $conf "tclForces on"
  puts $conf "tclForcesScript     $conffile-constr.tcl"
  puts $conf "reassignFreq 500"
  puts $conf "reassignTemp 610"
  puts $conf "reassignIncr -10"
  puts $conf "reassignHold 300"
  puts $conf "minimize            5000"
  puts $conf "reinitvels          610"
  puts $conf "run 20000"
  puts $conf "minimize 1000"


  close $conf
  $sel delete

  #Write the tclforces file
  set tclf [open "$conffile-constr.tcl" "w"]
  puts $tclf "# Automatically generated tcl script for constraining centers of mass"
  puts $tclf "# Note that this is a hideously ugly way of doing things -- don't use this"
  puts $tclf "# for anything except annealing a cg structure"
  puts $tclf ""
  puts $tclf "set beadcenters {$startposarray}"
  puts $tclf "set atomlists {$idxarray}"
  puts $tclf "set atommasses {$massarray}"
  puts $tclf "set k 5.0"
  puts $tclf "set t 0"
  puts $tclf "set TclFreq 20"
  puts $tclf ""
  puts $tclf ""
  puts $tclf "# Add all the atoms"
  puts $tclf "foreach atomlist \$atomlists {"
  puts $tclf "  foreach atom \$atomlist {"
  puts $tclf "    addatom \$atom"
  puts $tclf "  }"
  puts $tclf "}"
  puts $tclf ""
  puts $tclf "proc calcforces {} {"
  puts $tclf "  global beadcenters atomlists atommasses k t TclFreq"
  puts $tclf ""
  puts $tclf ""
  puts $tclf "  # Get the coordinates for our timestep"
  puts $tclf "  loadcoords coordinate"
  puts $tclf ""
  puts $tclf "  # Loop through the ex-beads and constrain each center of mass"
  puts $tclf "  foreach bead \$beadcenters atomlist \$atomlists masslist \$atommasses {"
  puts $tclf "    # Loop over the atoms in the bead and find its center of mass"
  puts $tclf "    set com \[list 0 0 0\]"
  puts $tclf "    set mtot 0"
  puts $tclf "    foreach atom \$atomlist mass \$masslist {"
  puts $tclf "      set myx \[lindex \$coordinate(\$atom) 0\]"
  puts $tclf "      set myy \[lindex \$coordinate(\$atom) 1\]"
  puts $tclf "      set myz \[lindex \$coordinate(\$atom) 2\]"
  puts $tclf ""
  puts $tclf "      set mycoors \[list \$myx \$myy \$myz\]"
  puts $tclf ""
  puts $tclf "      set com \[vecadd \$com \[vecscale \$mycoors \$mass\]\]"
  puts $tclf "      set mtot \[expr \$mtot + \$mass\]"
  puts $tclf "    }"
  puts $tclf ""
  puts $tclf "    set com \[vecscale \$com \[expr 1.0 / \$mtot\]\]"
  puts $tclf ""
  puts $tclf "    # Find the vector between the center of mass and the anchor, and apply a force"
  puts $tclf "    set delvec \[vecsub \$bead \$com\]"
  puts $tclf "    set dist2 \[vecdot \$delvec \$delvec\]"
  puts $tclf "    set dist1 \[expr sqrt(\$dist2)\]"
  puts $tclf "    set fmag \[expr \$dist2 \* \$k\]"
  puts $tclf "    set delvec \[vecscale \$delvec \[expr 1.0 / \$dist1\]\]"
  puts $tclf "    set fvec \[vecscale \$delvec \$fmag\]"
  puts $tclf "    set fvec \[vecscale \$fvec \[expr 1.0 / \$mtot\]\]"
  puts $tclf ""
  puts $tclf "    # Loop through the atoms and apply a mass-weighted force to each"
  puts $tclf "    foreach atom \$atomlist mass \$masslist {"
  puts $tclf "      set myf \[vecscale \$fvec \$mass\]"
  puts $tclf "      addforce \$atom \$myf"
  puts $tclf "    }"
  puts $tclf ""
  puts $tclf "  }"
  puts $tclf ""
  puts $tclf "  return"
  puts $tclf "}"

  puts $tclf "# Auxilliary procs to use with vectors"
  puts $tclf "proc vecmult {vec scalar} {"
  puts $tclf "  set newarr \[list\]"
  puts $tclf "  foreach elem \$vec {"
  puts $tclf "    set newelem \[expr \$elem * \$scalar\]"
  puts $tclf "    lappend newarr \$newelem"
  puts $tclf "  }"
  puts $tclf "  return \$newarr"
  puts $tclf "}"
  puts $tclf ""
  puts $tclf "proc vecdot {vec1 vec2} {"
  puts $tclf "  set newval 0"
  puts $tclf "  foreach elem1 \$vec1 elem2 \$vec2 {"
  puts $tclf "    set newval \[expr \$newval + (\$elem1 * \$elem2)\]"
  puts $tclf "  }"
  puts $tclf "  return \$newval"
  puts $tclf "}"
  puts $tclf ""
  puts $tclf "proc vecsub {vec1 vec2} {"
  puts $tclf "  set newarr \[list\]"
  puts $tclf "  foreach elem1 \$vec1 elem2 \$vec2 {"
  puts $tclf "    set newelem \[expr \$elem1 - \$elem2\]"
  puts $tclf "    lappend newarr \$newelem"
  puts $tclf "  }"
  puts $tclf "  return \$newarr"
  puts $tclf "}"

  close $tclf


  mol delete top
}


  

proc ::cgtools::read_bead {fstream} {
  #Given a file stream currently starting a new bead, read the bead's components
  # and return the new atom list
  # The bead "object" is simply a list of beads and atoms, as defined in make_atom
  # The first entry in this list is the "bead"; the others are the target atoms

  set mybead [list]
  
  while {[gets $fstream line] && ![regexp {^CGEND} $line]} {
    if {[regexp {^\#} $line]} { 
     continue 
    }

    #split the line up into fields and make a new atom
    set linearray [split $line]
    set linearray [noblanks $linearray]
    set newatom [make_atom [lindex $linearray 0] [lindex $linearray 1] [lindex $linearray 2]]
    lappend mybead $newatom
  }

  return $mybead
}

proc ::cgtools::make_atom {resname atomname resoffset} {
  #Create a new cg bead/atom with atomname in element 0,resname in element 1, and a resid
  # offset in element 2

  set newatom [list]

#  set newatom(Resname) $resname
#  set newatom(Atomname) $atomname
#  set newatom(Offset) $resoffset
  
  lappend newatom $atomname
  lappend newatom $resname
  lappend newatom $resoffset

  return $newatom
}

proc ::cgtools::apply_reversal {molid revcgfile origmolid outputfile {revbonds 0}} {

  #Apply the reverse transformations in revcgfile to the cg molecule in molid, using the initial all atom structure from origmolid

  # If revbonds is nonzero, attempts will be made to keep atoms with inter-bead bonds
  # as close as possible to their original positions by applying rotations

  variable idxarray ;# 1-indexed list of atoms in each bead
  variable startposarray
  variable massarray

  #Read the reverse cg file line by line, and apply each in turn
  set infile [open $revcgfile "r"]
  set idxarray [list]
  set startposarray [list]
  set massarray [list]
  set indexarray [list] ;# 0-indexed version of idxarray


#  set idxfile [open "revcg.idx" "w"]
#  set startfile [open "revcg_starts.dat" "w"]

  while {[gets $infile line] >= 0} {
    set linelist [split $line]
    set linelist [noblanks $linelist]
    set resname [lindex $linelist 0]
    set beadname [lindex $linelist 1]
    set resid [lindex $linelist 2]
    set segid [lindex $linelist 3]
    set indices [lreplace $linelist 0 3]
    
    set aasel [atomselect $origmolid "index $indices"]
    set cgsel [atomselect $molid "resname $resname and name $beadname and resid $resid and segid $segid"]
    
    set aacen [measure center $aasel weight mass]
    set cgcen [lindex [$cgsel get {x y z}] 0]
    
    set movevec [vecsub $cgcen $aacen]
    $aasel moveby $movevec

    #Write NAMD-tclforces indices to file
    set indlist [list]
    set realindlist [list]
    set masslist [list]
    foreach index [$aasel get index] mass [$aasel get mass] {
      lappend realindlist $index
      set index [expr $index + 1]
      lappend indlist $index
      lappend masslist $mass
    }

    lappend indexarray $realindlist
    lappend idxarray $indlist
    lappend massarray $masslist

    #Write center of mass position to file
    lappend startposarray $cgcen

    $aasel delete
    $cgsel delete
  }

  close $infile

  # If requested, rotate the beads to maintain bonds
  if {$revbonds > 0} {
    foreach atomlist $indexarray {
      set thisbeadsel [atomselect $origmolid "index $atomlist"]
      set bondlist [$thisbeadsel getbonds]
      #puts "prelimbonds: $bondlist"
      #puts "beadcontents: $atomlist"
      set beadbondlist [list]
      foreach myind $atomlist bondset $bondlist {
        foreach mybond $bondset {
      #    puts "looking for partner $mybond for atom $myind"
          if { [lsearch -exact $atomlist $mybond] == -1 } {
      #      puts "This partner is not in the current bead"
            # Find which bead this *does* belong to
      #      puts "looking for $mybond anywhere in the indexarray"
            for {set i 0} {$i < [llength $indexarray]} {incr i} {
              if { [lsearch -exact [lindex $indexarray $i] $mybond] > -1 } {
                lappend beadbondlist [list $myind $i]
      #          puts "This partner is in bead $i"
                break
              }
            }
          }
        }
      }

      # Now find and apply an optimal rotation to this bead component
      # Get a list of COM vectors to the atoms in this bead bonded to others
      set compos [measure center $thisbeadsel weight mass]
      set invecs [list]
      set outvecs [list]
      #puts "bonds: $beadbondlist"
      foreach entrylist $beadbondlist {
        set myind [lindex $entrylist 0]
        set otherbead [lindex $entrylist 1]
        set myatomsel [atomselect $origmolid "index $myind"]
        set oatomsel [atomselect $origmolid "index [lindex $indexarray $otherbead]"]
        set mycen [join [$myatomsel get {x y z}] ]
        set ocen [measure center $oatomsel weight mass]
        set myvec [vecnorm [vecsub $mycen $compos]]
        set ovec [vecnorm [vecsub $ocen $compos]]
        lappend invecs $myvec
        lappend outvecs $ovec
      }

      # Convert the vectors to lie on the unit sphere
      set myvecsph [list]
      set ovecsph [list]

      foreach vec $invecs {
        lappend myvecsph [vecnorm $vec]
      }

      foreach vec $outvecs {
        lappend ovecsph [vecnorm $vec]
      }

      puts "performing sphere search"

      # Search over the sphere to find a rotation minimizing the
      # distances between corresponding vectors
      puts "vectors: $myvecsph"
      puts "outvecs: $ovecsph"

      set bestvals [list 0 0]
      set minmag [expr 1.0 * [llength $myvecsph]]
      for {set xrot 0} {$xrot < 360} {incr xrot 5} {
        for {set zrot 0} {$zrot <= 180} {incr zrot 5} {
          set delmag 0
          foreach invec $myvecsph outvec $ovecsph {
              set invec [transform_unit_vec $invec $xrot $zrot]
              set delmag [expr $delmag + abs(1.0 - [vecdot $invec $outvec])]
          }

          #puts "$xrot $zrot: $delmag"
          if {$delmag < $minmag} {
            set minmag $delmag
            set bestvals [list $xrot $zrot]
          }
        }
      }

      # Now apply the transformations recommended by this optimization
      set bestx [lindex $bestvals 0]
      set bestz [lindex $bestvals 1]
      puts "best transformation is $bestx $bestz"
      set beadcen [measure center $thisbeadsel weight mass]
      $thisbeadsel move [trans center $beadcen offset $beadcen axis x $bestx deg]
      $thisbeadsel move [trans center $beadcen offset $beadcen axis z $bestz deg]

      $thisbeadsel delete

    }
    # (end of actions for each bead)

  }  ;# end of changes for revbonds
  set sel [atomselect $origmolid all]
  $sel writepdb $outputfile
  $sel delete
}

proc ::cgtools::transform_unit_vec {invec xrot zrot} {
# Transform a unit vector by rotation about the x and z axes
  set tmat [trans axis x $xrot deg axis z $zrot deg]
  set newcoords [list]
  for {set i 0} {$i < 3} {incr i} {
    set tvec [lrange [lindex $tmat $i] 0 2]
    lappend newcoords [vecdot $tvec $invec]
  }
  return $newcoords
}

proc ::cgtools::cart_to_sphere {vec} {
# take a vector in cartesian 3-space and return the theta/phi components of its
# equivalent in spherical coordinates
  set mag [veclength $vec]
  set rad_to_deg [expr 180.0 / 3.1415927]
  set x [lindex $vec 0]
  set y [lindex $vec 1]
  set z [lindex $vec 2]
  set theta [expr $rad_to_deg * atan( $y / $x ) ]
  set phi [expr $rad_to_deg * acos($z / $mag)]
  return [list $theta $phi]
}

proc cgtools::const_rad_dom {coords} {
  # constrains spherical coordinates to the appropriate domain [0,360), [0,180)
  set theta [lindex $coords 0]
  set phi [lindex $coords 1]

# First bring phi into [-180, 180)
  while {$phi < -180.0} {
    set phi [expr $phi + 360.0]
  }

  while {$phi >= 180.0} {
    set phi [expr $phi - 360.0]
  }

# then get the real value
  if {$phi < 0} {
    set phi [expr 360.0 + $phi]
    set theta [expr $theta + 180]
  }

# now do the same for theta
  while {$theta < 0} {
    set theta [expr $theta + 360.0]
  }

  while {$theta >= 360.0} {
    set theta [expr $theta - 360.0]
  }

  return [list $theta $phi]
}


  

proc ::cgtools::apply_database {molid outputfile revcgfile} {
  #Applies the contents of the current convbeads database to the
  # selected molecule, and writes the result to OUTPUTFILE

  variable convbeads

  if {[llength $convbeads] == 0} {
     puts "CG Tool Error) No bead definitions loaded.  Can't apply."
     return
  }
  #Open file for reverse coarse graining information
  # format of file is resname beadname segid index1 index2 index3...
  # where the first 3 fields come from the CG bead, and the indices are corresponding
  # all atom indices
  set rcgout [open $revcgfile "w"]

  #Beads which should be kept and written are tagged with occupancy 1
  # All other atoms should have occupancy 0
  set allsel [atomselect $molid all]
  set oldocc [$allsel get occupancy]
  set oldxyz [$allsel get {x y z}]
  $allsel set occupancy 0

  #Loop through the conversion database and do each bead type
  foreach cgbead $convbeads {
    apply_bead $cgbead $molid $rcgout
  }

  set writesel [atomselect $molid "occupancy 1"]
  $writesel writepdb $outputfile

  $writesel delete
  $allsel set occupancy $oldocc
  $allsel set {x y z} $oldxyz
  $allsel delete

  close $rcgout
}

proc ::cgtools::prepare_water {molid {simple 0}} {
# Make sure waters are consecutively numbered and spatially localized for CG procedure
  set watsel [atomselect $molid water] 
  $watsel set segid [lindex [$watsel get segid] 0]
  $watsel set resid 0
  $watsel delete

# Start looping over waters; pick one, and take the three unassigned waters closest to it to make a bead
  set watkeys [atomselect $molid "water and oxygen and resid 0"]
  set resnum 1
  set totalwat [$watkeys num]

  if {$simple != 0} {
    foreach index [$watkeys get index] {
    puts "Done with $resnum of $totalwat waters"
      set sel [atomselect $molid "same residue as index $index"]
      $sel set resid $resnum
      $sel delete
      incr resnum
    }
    $watkeys delete
    return
  }

  while {[$watkeys num] > 0} {
    puts "Done with $resnum of $totalwat waters"

    set mykey [lindex [$watkeys get index] 0]
    set keyres [atomselect $molid "same residue as index $mykey"]
    $keyres set resid $resnum
    incr resnum

# Grow a selection around the key until we find 3 other waters
    set r 4.0
    set othersel [atomselect $molid "water and oxygen and resid 0 and within $r of index [$keyres get index]"]

    while {[$othersel num] < 3} {
      set r [expr $r + 2.0]
      $othersel delete
      set othersel [atomselect $molid "water and oxygen and resid 0 and within $r of index [$keyres get index]"]
    }
  
    $othersel delete
    $keyres delete
  
    set otherkeys [$othersel get index]
    $othersel delete

    for {set i 0} {$i < [llength $otherkeys] && $i < 3} {incr i} {
      #puts "DEBUG: otherkeys $otherkeys i $i"
      set mywat [atomselect $molid "same residue as index [lindex $otherkeys $i]"]
      $mywat set resid $resnum
      incr resnum
      $mywat delete
    }

    set watkeys [atomselect $molid "water and oxygen and resid 0"]
  }
  $watkeys delete
}

proc ::cgtools::apply_bead {cgbead molid revcgfile} {
  #Applies the conversion specified in CGBEAD to the molecule MOLID
  # This means going though the molecule, matching everything that
  # corresponds to the first element of cgbead, and then building
  # each of those beads in turn

#puts "apply_bead: cgbead: '$cgbead', molid: $molid, file: $revcgfile"

  set beadname [lindex [lindex $cgbead 0] 0]
  set beadres [lindex [lindex $cgbead 0] 1]
  set beadoff [lindex [lindex $cgbead 0] 2]
  set cgbead [lreplace $cgbead 0 0]
  set headname [lindex [lindex $cgbead 0] 0]
  set headres [lindex $[lindex $cgbead 0] 1]
  set headoff [lindex $[lindex $cgbead 0] 2]
  set cgbead [lreplace $cgbead 0 0]

#puts "apply_bead: beadname: '$beadname', beadres: $beadres, beadoff: $beadoff"
#puts "headname: '$headname', headres: $headres, headoff: $headoff"
#puts "apply_bead: cgbead: '$cgbead'"

#puts "DEBUG: Looking to make cgbead $beadname $beadres $beadoff with head atom $headname $headres $headoff"

  #Find all of the atoms matching the head definition
  if {$headres == "*"} {
    set headbeads [atomselect $molid "name $headname"]
  } else {
    set headbeads [atomselect $molid "name $headname and resname $headres"]
  }
  $headbeads set occupancy 1
#  $headbeads set name $beadname
  if {$beadres != "*"} {
     $headbeads set resname $beadres
  }

  #Make three arrays of the qualifying characteristics of subordinate beads
  set names [list]
  set resnames [list]
  set offsets [list]
  foreach subatom $cgbead {
    lappend names [lindex $subatom 0]
    lappend resnames [lindex $subatom 1]
    lappend offsets [lindex $subatom 2]
  }

  #Now, for each head atom we've found, adjust its position according to 
  # where its children are
  foreach index [$headbeads get index] resid [$headbeads get resid] segid [$headbeads get segid] {
#puts "apply_bead: index: $index, resid: $resid, segid: $segid"
    set headatom [atomselect $molid "index $index"]
    if {[join [$headatom get occupancy] ] < 0} {
      $headatom delete
      continue
    }
#    puts "Applying bead with head $index"
    $headatom set resid [expr $resid - $headoff]
    $headatom set name $beadname
    set resid [expr $resid - $headoff]

    if {$segid == ""} {
       set fullbeadsel "occupancy >= 0 and ( index $index"
    } else {
       set fullbeadsel "occupancy >= 0 and segid $segid and ( index $index"
    }

    foreach name $names resname $resnames offset $offsets {
      set fullbeadsel "$fullbeadsel or \{ name $name and resid [expr $resid + $offset] and resname "
      if {$resname == "*"} {
         set fullbeadsel "$fullbeadsel [$headatom get resname]\} "
      } else {
         set fullbeadsel "$fullbeadsel $resname \}"
      }
    }

    set fullbeadsel " $fullbeadsel )"

    set mybeadsel [atomselect $molid "$fullbeadsel"]
    $headatom moveto [measure center $mybeadsel weight mass]

    puts $revcgfile "[$headatom get resname] [$headatom get name] [$headatom get resid] [$headatom get segid] [$mybeadsel get index]"
    
    $mybeadsel set occupancy -1
    $headatom set occupancy 1

    $headatom delete
    $mybeadsel delete
  }

  $headbeads delete
    
}

proc noblanks {mylist} {
  set newlist [list]
  foreach elem $mylist {
    if {$elem != ""} {
      lappend newlist $elem
    }
  }

  return $newlist
}


# Each CG bead in CGId molecule has a respective domain
# in the all-atom reference file (refId). Atoms from each
# domain of refId can be corresponded to atoms of AAId
# (for two atoms to be declared identical, segname, resid, 
# and name should match);
# a bead that represents the center of mass of a given refId domain
# should be moved to the center of mass of the corresponding AAId domain.
proc ::cgtools::mapCGMolecule { statusProc AAId CGId refId outPDB} {

   set AAsel [atomselect $AAId all]
   set CGsel [atomselect $CGId all]
   set refsel [atomselect $refId all]

   set Naa [$AAsel num]
   set Ncg [$CGsel num]
   set Nref [$refsel num]

# Fill out the array of coordinates for CG beads.
   set cg_pos_0 [$CGsel get {x y z}]
   for {set i 1} {$i <= $Ncg} {incr i} {
      set cg_pos($i) {0.0 0.0 0.0}
      set cg_norm($i) 0.0
   }

# Lists of segnames, resnames, and names for AAId molecule.
   set AAsegname [$AAsel get segname]
   set AAresid [$AAsel get resid]
   set AAname [$AAsel get name]
# Lists of coordinates and masses for AAId molecule.
   set AApos [$AAsel get {x y z}]
   set AAmass [$AAsel get mass]
   
# For AAId, set all betas to 0 initially.
   $AAsel set beta 0
   for {set i 0} {$i < $Naa} {incr i} {
#      puts "i = $i"

      # Beta field for each atom in refId contains the serial number
      # of the CG bead corresponding to the domain containing the atom.
      # Find the atom in refId which is the same as the one currently chosen from AAId.
      set A [atomselect $refId "name [lindex $AAname $i] and resid [lindex $AAresid $i] and segname [lindex $AAsegname $i]"]
      # If refId does not contain this atom, ignore it.
      # If refId contains one or more atoms satisfying the name, resid, and segname
      # of the chosen atom from AAId, use coordinates and mass of the first
      # of such atoms from refId (in principle, there should be only one such atom).
      if {[$A num] > 0} {
         # Get the serial of the bead to whose domain this refId atom belongs.
         set i_CG [expr int([lindex [$A get beta] 0])]
         # The bead will be moved to the center of mass of the corresponding 
         # AAId domain; add the coordinates of the current AAId atom to the
         # variable that will be used to obtain the coordinates of the bead.
         set cg_pos($i_CG) [vecadd $cg_pos($i_CG) [vecscale [lindex $AAmass $i] [lindex $AApos $i]]]
         set cg_norm($i_CG) [expr $cg_norm($i_CG) + [expr [lindex $AAmass $i]]]
      }
      $A delete
   }

# Update coordinates for CG beads.
   for {set i 1} {$i <= $Ncg} {incr i} {
      set cg_pos($i) [vecscale [expr 1.0/$cg_norm($i)] $cg_pos($i)]
      set A [atomselect $CGId "serial $i"]
      $A set x [lindex $cg_pos($i) 0]
      $A set y [lindex $cg_pos($i) 1]
      $A set z [lindex $cg_pos($i) 2]
      $A delete
   }
   set A [atomselect $CGId all]
#Write the output PDB.
   $A writepdb $outPDB
   $A delete
# Move beads back to old coordinates.
   for {set i 0} {$i < $Ncg} {incr i} {
      set A [atomselect $CGId "index $i"]
      $A set x [lindex [lindex $cg_pos_0 $i] 0]
      $A set y [lindex [lindex $cg_pos_0 $i] 1]
      $A set z [lindex [lindex $cg_pos_0 $i] 2]
      $A delete
   }

# clean up the memory that we grabbed at the first
   $AAsel delete
   $CGsel delete
   $refsel delete
#   mol delete $CGId

# Load the created CG structure into VMD.
   mol load pdb $outPDB
   mol modstyle 0 top VDW 5.000000 10.000000
}

# -----------------------------------------------------
# This proc assigns Lennard-Jones (LJ) parameters for a
# coarse-grained (CG) structure based on the all-atom one.
# It extracts solvent accessible surface area (SASA)
# for each atomic domain representing a CG bead from an
# all-atom structure corresponding to the CG structure.
# The values extracted are SASA for the whole domain
# (in the context of the rest of the structure) and SASA for
# hydrophobic residues only. These values are used to assign the
# LJ well depths E_i to individual CG beads, as
# E_i = ELJ * (SASA_i[hydrophobic]/SASA_i[total])^2, where
# ELJ is an adjustable constant.
# Radius of gyration is used to compute LJ radii; LJ radius R_i is
# obtained as R_i = r_gyr_i + RLJ, where r_gyr_i is
# the radius of gyration for all atoms represented by the i-th CG bead
# and RLJ is an adjustable constant.
##################################################
# INPUT DATA:
#
# CG parameter file, "par_CG";
#
# ID ("pdbrefID") of the all-atom reference structure that should be
# loaded in VMD already; this structure should have beta values
# filled with numbers corresponding to the CG beads' numbers;
# since radius of gyration is computed, it is better to load
# the PSF file into pdbrefID too, so that correct masses are used;
#
# the file to where the output is written, "f_out";
#
# maximum energy value for the LJ well depth, "ELJ" (kcal/mol);
#
# an addition to the LJ radius RLJ (A).
##################################################
proc ::cgtools::sasa_LJ_networking {statusProc par_CG pdbrefID f_out ELJ RLJ} {

# Check if we can write the output file.
set outfile [open $f_out w]
close $outfile

# Find out number of CG beads.
# puts "Checking the number of CG beads..."
set NAA [[atomselect $pdbrefID all] num]
set NCG 0
for {set i 0} {$i < $NAA} {incr i} {
  set Ntmp [expr int([[atomselect $pdbrefID "index $i"] get beta])]
  if {$Ntmp > $NCG} {
    set NCG $Ntmp
  }
}
# puts "done."
# puts "According to the reference molecule (ID $pdbrefID) NCG = $NCG."
# puts ""


# Read the CG parameter file and extract LJ parameters.
##################################################
set fdata ""
set par_CG_file [open $par_CG r];

set outfile [open $f_out w]
# Find where non-bonded entries start.
gets $par_CG_file fdata
set tmp [lindex $fdata 0]
while {$tmp != "NONBONDED"} {
  puts $outfile "$fdata"
  gets $par_CG_file fdata
  set tmp [lindex $fdata 0]
}

# Read current LJ parameters
# (basically, this is done to initialize arrays.).
set k 0
set i 0
while {($k < $NCG) && ($i < 100000000)} {
  incr i
  gets $par_CG_file fdata
  if {[string range [lindex $fdata 0] 0 0] != "!"} {
    incr k
    set CG($k.name) [lindex $fdata 0]
    set CG($k.E) [lindex $fdata 2]
    set CG($k.r) [lindex $fdata 3]
  }
}

##################################################


##################################################
# Loop over all CG beads; find SASA and r_gyr.   #
##################################################
set A_all [atomselect $pdbrefID all]
for {set k 1} {$k <= $NCG} {incr k} {
  # puts "Bead $k of $NCG"
  set A [atomselect $pdbrefID "beta $k"]
  set A_hphob [atomselect $pdbrefID "beta $k and hydrophobic"]
  set tmp [measure sasa 0.0 $A_all -restrict $A]
  set CG($k.sasa) [measure sasa 0.0 $A_all -restrict $A]
  set CG($k.sasa_hphob) [measure sasa 0.0 $A_all -restrict $A_hphob]
  
  # r_gyr
  if {[$A num] > 0} {
    set CG($k.r) [expr [measure rgyr $A] + $RLJ]
  }
  
  $A delete
  $A_hphob delete
  if {$CG($k.sasa) == 0} {
    set CG($k.sasa_ratio) 0.0
  } else {
    set CG($k.sasa_ratio) [expr $CG($k.sasa_hphob)/$CG($k.sasa)]
  }
}
$A_all delete
##################################################
##################################################


##################################################
# Append the output to a file.                  #
##################################################

puts $outfile "NONBONDED"
puts $outfile "!"
puts $outfile "!V(Lennard-Jones) = Eps,i,j\[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6\] "
puts $outfile "!"
puts $outfile "!epsilon: kcal/mole, Eps,i,j = sqrt(eps,i * eps,j) "
puts $outfile "!Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j "
puts $outfile "!"
puts $outfile "!atom  ignored    epsilon      Rmin/2 "
puts $outfile "!"

##################################################
for {set k 1} {$k <= $NCG} {incr k} {
  puts $outfile [format "%-5s%10f%11.6f%12.6f" \
                            $CG($k.name) 0.0 \
			    [expr (-1.0)*$ELJ*$CG($k.sasa_ratio)*$CG($k.sasa_ratio)] \
			    $CG($k.r)]
}
##################################################
puts $outfile ""
puts $outfile "END"
puts $outfile ""

close $outfile
##################################################
##################################################

}


