############################################################################
#cr
#cr            (C) Copyright 1995-2007 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: save_state.tcl,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.47 $        $Date: 2014/12/05 21:56:17 $
#
############################################################################
#
# VMD save state $Revision: 1.47 $
#
#
# Script to save current viewpoint, representations etc to a file.
#
# Usage:
# In the vmd console type 
#       save_state foo.vmd
# to save your work in file foo.  Starting vmd with
#       vmd -e foo.vmd
# will restore most of your previous session.
#  
# In particular the script will attempt to restore
#     Molecules read in from files (structure, coords, trajectories, EDM, etc)
#     Graphics objects (spheres, cylenders...)
#     all representations
#     the transformation matrices for each molecule
#     which color to use for which representation, definitions of colors 0..33
#     material properties
#     atom selection macros
#
# XXX It doesn't currently restore:
#     Trajectories loaded via the edit menu, with deleted frames, etc
#     User provided data fields such as "user", "beta", "mol volume", etc.
#     Any data produced or modified by scripts, even atom positions etc 
#     Interactive MD sessions
#     Global display properties such as window resolution, window position,
#       lighting, stereo, axes, stage, ...
#


#
# Save transformation matrices, centering, scaling, translation, etc.
#
proc save_viewpoint {} {
   global viewpoints
   if [info exists viewpoints] {unset viewpoints}
   # get the current matricies
   foreach mol [molinfo list] {
      set viewpoints($mol) [molinfo $mol get {
        center_matrix rotate_matrix scale_matrix global_matrix}]
   }
}


#
# Restore transformation matrices, centering, scaling, translation, etc.
#
proc restore_viewpoint {} {
   global viewpoints
   foreach mol [molinfo list] {
      if [info exists viewpoints($mol)] {
         molinfo $mol set {center_matrix rotate_matrix scale_matrix
           global_matrix} $viewpoints($mol)
      }
   }
}


#
# Save all representations and their complete settings
#
proc save_reps {} {
  global representations
  foreach mol [molinfo list] {
    set representations($mol) ""
    for {set i 0} {$i < [molinfo $mol get numreps]} {incr i} {
      set rep [molinfo $mol get "{rep $i} {selection $i} {color $i} {material $i}"]
      lappend rep [mol showperiodic $mol $i]
      lappend rep [mol numperiodic $mol $i]
      lappend rep [mol showrep $mol $i]
      lappend rep [mol selupdate $i $mol]
      lappend rep [mol colupdate $i $mol]
      lappend rep [mol scaleminmax $mol $i]
      lappend rep [mol smoothrep $mol $i]
      lappend rep [mol drawframes $mol $i]

      # save per-representation clipping planes...
      set cplist {}
      for {set cp 0} {$cp < [mol clipplane num]} {incr cp} {
        set newcp {}
        foreach cpstat { center color normal status } {
          lappend newcp [mol clipplane $cpstat $cp $i $mol]
        }
        lappend cplist $newcp
      }
      lappend rep $cplist
      lappend representations($mol) $rep
    }
  }
}


#
# Restore all representations and their complete settings
#
# XXX this code is an incomplete testing routine and doesn't restore everything
# that is saved.
proc restore_reps {} {
  global representations
  foreach mol [molinfo list] {
    if [info exists representations($mol)] {
      #delete current representations
      for {set i 0} {$i < [molinfo $mol get numreps]} {incr i} {
        mol delrep $i $mol
      }

      #restore saved representations
      foreach rep $representations($mol) {
        puts $rep
        lassign $rep r s c m
        mol representation $r
        mol color $c
        mol selection $s
        mol material $m
        mol addrep $mol
      }
    }
  }
}


#
# Top level state saving routine, presents a file dialog when run from Tk,
# or a text prompt for the filename if Tk is unavailable.
#
proc save_state {{file EMPTYFILE}} {
  global representations
  global viewpoints
  save_viewpoint
  save_reps

  # If no file was given, get a filename.  Use the Tk file dialog if 
  # available, otherwise get it from stdin. 
  if {![string compare $file EMPTYFILE]} {
    set title "Enter filename to save current VMD state:"
    set filetypes [list {{VMD files} {.vmd}} {{All files} {*}}]
    if { [info commands tk_getSaveFile] != "" } {
      set file [tk_getSaveFile -defaultextension ".vmd" \
        -title $title -filetypes $filetypes]
    } else {
      puts "Enter filename to save current VMD state:"
      set file [gets stdin]
    }
  }
  if { ![string compare $file ""] } {
    return
  }

  set fildes [open $file w]
  puts $fildes "\#!/usr/local/bin/vmd"
  puts $fildes "\# VMD script written by save_state \$Revision: 1.47 $"

  set vmdversion [vmdinfo version]
  puts $fildes "\# VMD version: $vmdversion"

  puts $fildes "set viewplist {}"
  puts $fildes "set fixedlist {}"
  save_materials     $fildes
  save_atomselmacros $fildes
  save_display       $fildes

  foreach mol [molinfo list] {
    set files [lindex [molinfo $mol get filename] 0]
    set specs [lindex [molinfo $mol get filespec] 0]
    set types [lindex [molinfo $mol get filetype] 0]
    set nfiles [llength $files]
    if { $nfiles >= 1 } {
      set filecmd [list mol new [lindex $files 0] type [lindex $types 0]]
      set specstr [join [list [lindex $specs 0] waitfor all]]
      puts $fildes "$filecmd $specstr"
    } else {
      puts $fildes "mol new"
    }
    for { set i 1 } { $i < $nfiles } { incr i } {
      set filecmd [list mol addfile [lindex $files $i] type [lindex $types $i]]
      set specstr [join [list [lindex $specs $i] waitfor all]]
      puts $fildes "$filecmd $specstr"
    }
    foreach g [graphics $mol list] {
      puts $fildes "graphics top [graphics $mol info $g]"
    }
    puts $fildes "mol delrep 0 top"
    if [info exists representations($mol)] {
      set i 0
      foreach rep $representations($mol) {
        foreach {r s c m pbc numpbc on selupd colupd colminmax smooth framespec cplist} $rep { break }
        puts $fildes "mol representation $r"
        puts $fildes "mol color $c"
        puts $fildes "mol selection {$s}"
        puts $fildes "mol material $m"
        puts $fildes "mol addrep top"
        if {[string length $pbc]} {
          puts $fildes "mol showperiodic top $i $pbc"
          puts $fildes "mol numperiodic top $i $numpbc"
        }
        puts $fildes "mol selupdate $i top $selupd"
        puts $fildes "mol colupdate $i top $colupd"
        puts $fildes "mol scaleminmax top $i $colminmax"
        puts $fildes "mol smoothrep top $i $smooth"
        puts $fildes "mol drawframes top $i {$framespec}"
        
        # restore per-representation clipping planes...
        set cpnum 0
        foreach cp $cplist {
          foreach { center color normal status } $cp { break }
          puts $fildes "mol clipplane center $cpnum $i top {$center}"
          puts $fildes "mol clipplane color  $cpnum $i top {$color }"
          puts $fildes "mol clipplane normal $cpnum $i top {$normal}"
          puts $fildes "mol clipplane status $cpnum $i top {$status}"
          incr cpnum
        }

        if { !$on } {
          puts $fildes "mol showrep top $i 0"
        }
        incr i
      } 
    }
    puts $fildes [list mol rename top [lindex [molinfo $mol get name] 0]]
    if {[molinfo $mol get drawn] == 0} {
      puts $fildes "molinfo top set drawn 0"
    }
    if {[molinfo $mol get active] == 0} {
      puts $fildes "molinfo top set active 0"
    }
    if {[molinfo $mol get fixed] == 1} {
      puts $fildes "lappend fixedlist \[molinfo top\]"
    }

    puts $fildes "set viewpoints(\[molinfo top\]) [list $viewpoints($mol)]"
    puts $fildes "lappend viewplist \[molinfo top\]"
    if {$mol == [molinfo top]} {
      puts $fildes "set topmol \[molinfo top\]"
    }
    puts $fildes "\# done with molecule $mol"
  } 
  puts $fildes "foreach v \$viewplist \{"
  puts $fildes "  molinfo \$v set {center_matrix rotate_matrix scale_matrix global_matrix} \$viewpoints(\$v)"
  puts $fildes "\}"
  puts $fildes "foreach v \$fixedlist \{"
  puts $fildes "  molinfo \$v set fixed 1"
  puts $fildes "\}"
  puts $fildes "unset viewplist"
  puts $fildes "unset fixedlist"
  if {[llength [molinfo list]] > 0} {
    puts $fildes "mol top \$topmol"
    puts $fildes "unset topmol"
  }
  save_colors $fildes
  save_labels $fildes
  close $fildes
}


#
# Save all colors and coloring schemes
#
# An up-to-date list of color categories can be produced using
#   foreach c [colorinfo categories] {
#     foreach cc [colorinfo category $c] {
#       lappend colcatlist "$c,$cc" [colorinfo category $c $cc]
#     }
#   }
#
# The list of default rgb values is generated by
#   for {set c 0} {$c < 2*[colorinfo num]} {incr c} {
#     lappend def_rgb [colorinfo rgb $c]
#   }
proc save_colors {fildes} {
  puts $fildes "proc vmdrestoremycolors \{\} \{"

  # save definitions of all color scales
  foreach method [colorinfo scale methods] {
    puts $fildes [concat "  color scale colors" $method [color scale colors $method]]
  }

  set scale [colorinfo scale method]
  puts $fildes "  color scale method $scale"

  set colcatlist { 
    Display,Background black
    Display,Foreground white
    Axes,X red Axes,Y green Axes,Z blue Axes,Origin cyan Axes,Labels white
    Stage,Even blue Stage,Odd green
    Labels,Dihedrals cyan Labels,Angles yellow Labels,Bonds white
    Labels,Atoms green
    Name,H white Name,O red Name,N blue Name,C cyan Name,S yellow Name,P
    tan Name,Z silver
    Type,H white Type,O red Type,N blue Type,C cyan Type,S yellow Type,P
    tan Type,Z silver
    Element,H white Element,O red Element,N blue Element,C cyan 
    Element,S yellow Element,P tan Element,Zn silver
    Resname,ALA blue Resname,ARG white Resname,ASN tan Resname,ASP red 
    Resname,CYS yellow Resname,CYX yellow
    Resname,GLY white Resname,GLU pink Resname,GLN
    orange Resname,HIS cyan Resname,HSE cyan Resname,HSD cyan Resname,HSP cyan
    Resname,ILE green Resname,LEU pink
    Resname,LYS cyan Resname,MET yellow Resname,PHE purple Resname,PRO
    ochre Resname,SER yellow Resname,THR mauve Resname,TRP silver
    Resname,TYR green Resname,VAL tan Resname,ADE blue Resname,CYT orange
    Resname,GUA yellow Resname,THY purple Resname,URA green Resname,TIP
    cyan Resname,TIP3 cyan Resname,WAT cyan Resname,SOL cyan Resname,H2O
    cyan Resname,LYR purple Resname,ZN silver Resname,NA yellow
    Resname,CL green 
    Restype,Unassigned cyan Restype,Solvent yellow Restype,Nucleic_Acid
    purple Restype,Basic blue Restype,Acidic red Restype,Polar green
    Restype,Nonpolar white Restype,Ion tan
    Highlight,Proback green Highlight,Nucback yellow Highlight,Nonback 
    blue
    {Structure,Alpha Helix} purple Structure,3_10_Helix mauve
    Structure,Pi_Helix red Structure,Extended_Beta yellow
    Structure,Bridge_Beta tan Structure,Turn cyan Structure,Coil white
  }

  set def_rgb {
    {0.25 0.25 1.0} {1.0 0.0 0.0} {0.35 0.35 0.35} {0.8 0.5 0.2}
    {0.8 0.8 0.0} {0.5 0.5 0.2} {0.6 0.6 0.6} {0.2 0.7 0.2} {1.0 1.0 1.0}
    {1.0 0.6 0.6} {0.25 0.75 0.75} {0.65 0.3 0.65} {0.5 0.9 0.4}
    {0.9 0.4 0.7} {0.5 0.3 0.0} {0.5 0.75 0.75} {0.0 0.0 0.0}
  } 
    
  array set def_colcat $colcatlist

  puts $fildes "  set colorcmds \{"
  foreach c [colorinfo categories] {
    foreach cc [colorinfo category $c] {
      set col [colorinfo category $c $cc]
      if {![info exists def_colcat($c,$cc)] ||
        [string compare $col $def_colcat($c,$cc)]} {
        puts $fildes "    \{color $c \{$cc\} $col\}"
      }
    }
  }
  puts $fildes "  \}"

  puts $fildes "  foreach colcmd \$colorcmds \{"
  puts $fildes "    set val \[catch \{eval \$colcmd\}\]" 
  puts $fildes "  \}"

  set cnum [colorinfo num]
  for {set c 0} {$c < $cnum} {incr c} {
    set rgb [colorinfo rgb $c]
    if  {[string compare $rgb [lindex $def_rgb $c]]}  {
      puts $fildes "  color change rgb $c $rgb"
    }
  }

  puts $fildes "\}"
  puts $fildes "vmdrestoremycolors"
}


proc save_materials { filedes } {
  puts $filedes "proc vmdrestoremymaterials \{\} \{"

  set mlist [material list]
  # materials 0 and 1 cannot be deleted, but they can be modified.
  # The user may already have identically named materials loaded,
  # only create new materials if they don't already exist
  puts $filedes "  set mlist \{ $mlist \}"
  puts $filedes "  set mymlist \[material list\]"
  puts $filedes "  foreach mat \$mlist \{"
  puts $filedes "    if \{ \[lsearch \$mymlist \$mat\] == -1 \} \{ "
  puts $filedes "      material add \$mat"
  puts $filedes "    \}"
  puts $filedes "  \}"

  # once we know all material names exist, update their settings
  for { set i 0 } { $i < [llength $mlist] } { incr i } {
    set mat [lindex $mlist $i]
    lassign [material settings $mat] amb spec dif shin mirr opac outl outlw transmode
    puts $filedes "  material change ambient $mat $amb"
    puts $filedes "  material change diffuse $mat $dif"
    puts $filedes "  material change specular $mat $spec"
    puts $filedes "  material change shininess $mat $shin"
    puts $filedes "  material change mirror $mat $mirr"
    puts $filedes "  material change opacity $mat $opac"
    puts $filedes "  material change outline $mat $outl"
    puts $filedes "  material change outlinewidth $mat $outlw"
    puts $filedes "  material change transmode $mat $transmode"
  }

  puts $filedes "\}"
  puts $filedes "vmdrestoremymaterials"
}


#
# Save all machine-portable display settings that are not user-preferences
# 
proc save_display { file } {
  ##
  ## Save camera / projection parameters
  ##
  puts $file "# Display settings"

  set eyesep      [display get eyesep] 
  set focallength [display get focallength]
  set height      [display get height]
  set distance    [display get distance]
  set projection  [display get projection]
  set nearclip    [display get nearclip]
  set farclip     [display get farclip]
  puts $file "display eyesep       $eyesep"
  puts $file "display focallength  $focallength"
  puts $file "display height       $height"
  puts $file "display distance     $distance"
  puts $file "display projection   $projection"
  puts $file "display nearclip set $nearclip"
  puts $file "display farclip  set $farclip"

  ##
  ## Save depth cueing parameters
  ## 
  set cueonoff    [display get depthcue]
  set cuestart    [display get cuestart]
  set cueend      [display get cueend  ]
  set cuedensity  [display get cuedensity]
  set cuemode     [display get cuemode ]
  puts $file "display depthcue   $cueonoff"

  # set cuestart and cuend TWICE to overcome cases where the new and
  # old values conflict (e.g. new start greater than old end, etc)
  puts $file "display cuestart   $cuestart"
  puts $file "display cueend     $cueend"
  puts $file "display cuestart   $cuestart"
  puts $file "display cueend     $cueend"

  puts $file "display cuedensity $cuedensity"
  puts $file "display cuemode    $cuemode"

  ##
  ## Save ray tracing parameters:
  ##   shadows, AO, DoF
  ##
  set shadowsonoff [display get shadows]
  set aoonoff      [display get ambientocclusion] 
  set aoambient    [display get aoambient]
  set aodirect     [display get aodirect]
  set dofonoff     [display get dof]
  set dofaperture  [display get dof_fnumber]
  set doffocaldist [display get dof_focaldist] 

  puts $file "display shadows $shadowsonoff" 
  puts $file "display ambientocclusion $aoonoff"
  puts $file "display aoambient $aoambient"
  puts $file "display aodirect $aodirect"
  puts $file "display dof $dofonoff"
  puts $file "display dof_fnumber $dofaperture"
  puts $file "display dof_focaldist $doffocaldist"
}


#
# Save all atom labels
#
proc save_labels { file } {
  set atomlist [label list Atoms]
  set bondlist [label list Bonds]
  set anglelist [label list Angles]
  set dihedrallist [label list Dihedrals]

  # label text size currently applies to all labels in all categories,
  # so we only print it once.
  puts $file "label textsize [label textsize]"

  set ind 0
  foreach atom $atomlist {
    lassign $atom atom1 value show
    lassign $atom1 mol index

    puts $file "label add Atoms ${mol}/${index}"
    if { [string compare $show show] } { 
      # don't show this label.  Do this by simply repeating the add command
      puts $file "label add Atoms ${mol}/${index}"
    } 
    puts $file [list label textoffset Atoms $ind [label textoffset Atoms $ind]]
    puts $file [list label textformat Atoms $ind [label textformat Atoms $ind]]
    incr ind
  } 

  set ind 0
  foreach bond $bondlist {
    lassign $bond atom1 atom2 value show
    lassign $atom1 mol1 index1
    lassign $atom2 mol2 index2
    puts $file "label add Bonds ${mol1}/${index1} ${mol2}/${index2}"
    if { [string compare $show show] } { 
      # don't show this label.  Do this by simply repeating the add command
      puts $file "label add Bonds ${mol1}/${index1} ${mol2}/${index2}"
    } 
    puts $file [list label textoffset Bonds $ind [label textoffset Bonds $ind]]
    incr ind
  }

  set ind 0
  foreach angle $anglelist {
    lassign $angle atom1 atom2 atom3 value show
    lassign $atom1 mol1 index1
    lassign $atom2 mol2 index2
    lassign $atom3 mol3 index3
    puts $file "label add Angles ${mol1}/${index1} ${mol2}/${index2} ${mol3}/${index3}"
    if { [string compare $show show] } { 
      # don't show this label.  Do this by simply repeating the add command
      puts $file "label add Angles ${mol1}/${index1} ${mol2}/${index2} ${mol3}/${index3}"
    } 
    puts $file [list label textoffset Angles $ind [label textoffset Angles $ind]]
    incr ind
  }

  set ind 0
  foreach dihedral $dihedrallist {
    lassign $dihedral atom1 atom2 atom3 atom4 value show
    lassign $atom1 mol1 index1
    lassign $atom2 mol2 index2
    lassign $atom3 mol3 index3
    lassign $atom4 mol4 index4
    puts $file "label add Dihedrals ${mol1}/${index1} ${mol2}/${index2} ${mol3}/${index3} ${mol4}/${index4}"
    if { [string compare $show show] } { 
      # don't show this label.  Do this by simply repeating the add command
      puts $file "label add Dihedrals ${mol1}/${index1} ${mol2}/${index2} ${mol3}/${index3} ${mol4}/${index4}"
    } 
    puts $file [list label textoffset Dihedrals $ind [label textoffset Dihedrals $ind]]
    incr ind
  }
}


#
# Save atom selection macros
#
proc save_atomselmacros { file } {
  set mlist [atomselect macro]
  if { [llength $mlist] } {
    puts $file "# Atom selection macros"
    foreach macro [atomselect macro] {
      puts $file [list atomselect macro $macro [atomselect macro $macro]]
    }
  }
}
