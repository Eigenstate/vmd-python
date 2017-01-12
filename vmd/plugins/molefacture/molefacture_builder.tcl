##
## $Id: molefacture_builder.tcl,v 1.39 2016/02/04 19:43:05 johns Exp $
##

# This file contains all procs relating to molecule building, fragments, and the like

proc ::Molefacture::add_all_hydrogen {} {
   variable tmpmolid
   variable openvalencelist

   mol reanalyze $tmpmolid
   update_openvalence

   set sel [atomselect $tmpmolid "occupancy >= 0.5"]
   set mindexlist [$sel get index]

   set i 0
   foreach mindex $mindexlist {
 #     .molefac.val.list.list selection clear 0 end
  #    .molefac.val.list.list selection set $mindex
      variable picklist $mindex
      set numhyd [lindex $openvalencelist $i]
      set j 0

      while {$j < $numhyd} {
        add_hydrogen_noupdate
        incr j
      }

      incr i
    }

    update_openvalence
    variable bondlist [bondlist]
    variable anglelist [anglelist]
      
}

proc ::Molefacture::add_hydrogen_noupdate { } {
  #Same as add_hydrogen, but doesn't call update_openvalence, which
  #is a major timesuck for big structures
   variable tmpmolid
   variable openvalencelist
   variable atomaddlist
   variable picklist

   variable availablehyd

   #Check to make sure we don't need more dummy hydrogens
   if {$availablehyd <= 0} {
     set sel [atomselect $tmpmolid all]
     hrealloc $sel
     $sel delete
   }

   foreach mindex $picklist {
      set dummies [atomselect $tmpmolid "occupancy 0"]
#puts "[$dummies num]"
      set hindex [lindex [$dummies list] 0]
#puts "$hindex"
      set hydro  [atomselect $tmpmolid "index $hindex"]
#puts "[$hydro num]"
      set mother [atomselect $tmpmolid "index $mindex"]
      set mcoor [join [$mother get {x y z}]]
      #draw text $mcoor MO
      #puts "[lindex $openvalencelist $mindex]==2 && [lindex $atomaddlist $mindex]"

      set hdist 1.1
      set hdir {0 0 0}
      set bonds [$mother getbonds]
      if  {[llength [lindex $bonds 0]] > 0} {
        set ante  [atomselect $tmpmolid "index [join $bonds]"]
        set oldhydrogens [atomselect $tmpmolid "index [join $bonds] and occupancy 0.5"]
      } else {
            set ante [atomselect $tmpmolid none]
            set oldhydrogens [atomselect $tmpmolid none]
      }
      set numadd [expr [$oldhydrogens num] + 1]
##puts "Mindices: $mindex $numadd"
      set newcoor [calc_geo $mindex $numadd]
#puts "newcoor: $newcoor"
      if {[llength newcoor] == 0} {
        #Then we have an error
        puts "Warning: This atom can't accept any more bonds"
      }

      set movecoor [lindex $newcoor 0]
      #puts "movecoor: $movecoor"
      $hydro moveto $movecoor
      $hydro set occupancy 0.5
      #puts "[$mother get segid]"
#      $hydro set segid   [join [$mother get segid]]
      $hydro set segid   [$mother get segid]
      $hydro set resid   [$mother get resid]
      $hydro set resname [$mother get resname]
      $hydro set atomicnumber 1
      $hydro set name H
      $hydro set type H
      #mol modrep 0 top
      vmd_addbond $mindex $hindex

      set newcoor [lreplace $newcoor 0 0]
      
      for {set i 0} {$i < [$oldhydrogens num]} {incr i} {
        if {[llength $newcoor] == 0} {break}
        set moveatom [atomselect $tmpmolid "index [lindex [$oldhydrogens get index] $i]"]
        set movecoor [lindex $newcoor 0]
        $moveatom moveto $movecoor
        set newcoor [lreplace $newcoor 0 0]
        vmd_addbond $mindex [$moveatom get index] 1
      }


#      set motheratomadd [lindex atomaddlist $mindex]
#      lappend motheratomadd $hindex
#      lset atomaddlist $mindex $motheratomadd
#      lappend atomaddlist 0
   }

#   update_openvalence


   # Restore the selection
}

proc ::Molefacture::add_hydrogen {mindex} {
   add_hydrogen_noupdate

   update_openvalence

   # Update the displayed bondlist
   variable bondlist [bondlist]
   variable anglelist [anglelist]
}

proc ::Molefacture::read_fragmentlist {file} {
  #This proc will read a set of fragments in the molefacture fragment DB format
  #from a file, and then add them to the fragmentlist array

  variable fragmentlist
  variable periodic

  set infile [open $file r]

  while {[gets $infile line]>=0} {
#          puts $line
    if {[regexp {(^FRAGMENT\s*)(\w*)} $line match head fragname] > 0} {
      set myfrag [list]
      while {[gets $infile line]>=0 && [regexp {^END} $line]==0} {
        set i [regexp {\w+\s+\w+\s+(\w+)\s+\w+\s+\w+\s+\w+\s+(\-?\w+\.\w+)\s+(\-?\w+\.\w+)\s+(\-?\w+\.\w+)\s+\w+\.\w+\s+\w+\.\w+\s+\w+\s+(\w+)} $line match name x y z element]
        #set elementid [lsearch $periodic $element]
        set myatom [list $name $element $x $y $z]
        lappend myfrag $myatom
      }
      set fragmentlist($fragname) $myfrag 
    }
  }
  close $infile
}

proc ::Molefacture::replace_hydrogen_with_fragment {fragpath hydindex {retval 0}} {
  #This procedure will replace a hydrogen (which must have exactly 1 bond)
  #with a fragment
  # If retval == 1, will return a list of the newly added atom indices

  variable tmpmolid
  variable availablehyd

  set basesel [atomselect $tmpmolid "index $hydindex"]
  if {[llength [$basesel getbonds]]>1 || [$basesel get element] != "H"} {
      puts "You can only replace singly bonded hydrogen atoms!"
      return 1
  }



  # Open the fragment file and read all the atoms to be added
  set fragfile [open $fragpath "r"]
  gets $fragfile line
  set names [list]
  set addx [list]
  set addy [list]
  set addz [list]
  set addelem [list]
  #First get the h-replacing atom
  gets $fragfile line
  set linearray [split $line]
  set linearray [noblanks $linearray]
  set repname [lindex $linearray 0]
  set repx [lindex $linearray 1]
  set repy [lindex $linearray 2]
  set repz [lindex $linearray 3]
  set repelem [lindex $linearray 4]

  #Now all of the others
  gets $fragfile line
  while {[string range $line 0 5] != "BONDS"} {
    set linearray [split $line]
    set linearray [noblanks $linearray]
    lappend names [lindex $linearray 0]
    lappend addx [lindex $linearray 1]
    lappend addy [lindex $linearray 2]
    lappend addz [lindex $linearray 3]
    lappend addelem [lindex $linearray 4]
#  puts "$linearray"
    gets $fragfile line
  }

  #Finally, read the specified bonds
  set bondpart1 [list]
  set bondpart2 [list]
  set bondorders [list]
  while {[gets $fragfile line] != -1} {
    set linearray [split $line]
    set linearray [noblanks $linearray]
    lappend bondpart1 [lindex $linearray 0]
    lappend bondpart2 [lindex $linearray 1]
    lappend bondorders [lindex $linearray 2]
  }
  close $fragfile

   #Check to make sure we don't need more dummy hydrogens
   if {$availablehyd <= [llength $names]} {
     puts "before hrealloc tmpmolid is $tmpmolid"
     set sel [atomselect $tmpmolid all]
     hrealloc $sel
     $sel delete
     puts "tmpmolid is now $tmpmolid"
     set basesel [atomselect $tmpmolid "index $hydindex"]
   }

  $basesel set occupancy 0.8

  # Get residue/segment/etc information
  set myres [$basesel get resname]
  set myresid [$basesel get resid]
  set mysegname [$basesel get segname]
  set mysegid [$basesel get segid]

  #Orient our coordinate system
  set mothers [$basesel getbonds]
  set mindex [lindex [lindex $mothers 0] 0]
  set msel [atomselect $tmpmolid "index $mindex"]
  set basecoor [lindex [$basesel get {x y z}] 0]
  set mcoor [lindex [$msel get {x y z}] 0]
  set myvec [vecsub $basecoor $mcoor]

  #Move the hydrogen to the proper place and replace it with a new atom
  $basesel moveto [list $repx 0 0]
  $basesel set name $repname
  $basesel set element $repelem
  $basesel set type $repname

  #List for processing bonds later
  set addedatoms [list]
  lappend addedatoms $hydindex


  #Now, the new piece is built near the origin, and then translated and rotated
  #to the proper position
  foreach name $names x $addx y $addy z $addz elem $addelem {
    set newid [add_atom $name $name $myres $myresid $mysegname $mysegid $elem $x $y $z 0.0]
    lappend addedatoms $newid
  }

  # Adjust the order of the first bond, if desired
  if {[lindex $bondpart2 0] == -1} {
    set order [lindex $bondorders 0]
    set bondpart1 [lreplace $bondpart1 0 0]
    set bondpart2 [lreplace $bondpart2 0 0]
    set bondorders [lreplace $bondorders 0 0]
    set i 1
    while {$i < $order} {
      raise_bondorder [list [lindex [$msel get index] 0] [lindex [$basesel get index] 0]]
      incr i
    }
  }
    
    

  #Add the specified bonds
  foreach atom1 $bondpart1 atom2 $bondpart2 order $bondorders {
    vmd_addbond [lindex $addedatoms $atom1] [lindex $addedatoms $atom2] $order
  }

  #Properly orient and translate the new fragment
  set newsel [atomselect $tmpmolid "index $addedatoms"]
  $newsel move [transvec $myvec]
  $newsel moveby [list [$msel get x] [$msel get y] [$msel get z]]
  $newsel set occupancy 0.8
#  $newsel moveby [vecscale $myvec $repx]

  # Update molecule characteristics
  update_openvalence

  set newindices [$newsel get index]

  $newsel delete
  $msel delete
  $basesel delete

  if {$retval == 1} {
    return $newindices
  }
}

proc ::Molefacture::new_mol_from_fragment {fragpath} {
  #This procedure creates a new molecule from a given base fragment
  #changes to previously edited molecules will be lost

  set response [new_mol_gui]
  if {$response == 0} {return}

  variable tmpmolid

  # Open the fragment file and read all the atoms to be added
  set fragfile [open $fragpath "r"]
  gets $fragfile line
  set linearray [split $line]
  set linearray [noblanks $linearray]
  set myresname [string range [lindex $linearray 1] 0 2]
  gets $fragfile line
  while {[string range $line 0 5] != "BONDS"} {
    set linearray [split $line]
    set linearray [noblanks $linearray]
    lappend names [lindex $linearray 0]
    lappend addx [lindex $linearray 1]
    lappend addy [lindex $linearray 2]
    lappend addz [lindex $linearray 3]
    lappend addelem [lindex $linearray 4]
#  puts "$linearray"
#  puts $addelem
    gets $fragfile line
  }

  #Finally, read the specified bonds
  set bondpart1 [list]
  set bondpart2 [list]
  set bondorders [list]
  while {[gets $fragfile line] != -1} {
    set linearray [split $line]
    set linearray [noblanks $linearray]
    lappend bondpart1 [lindex $linearray 0]
    lappend bondpart2 [lindex $linearray 1]
    lappend bondorders [lindex $linearray 2]
  }
  close $fragfile

  set addedatoms [list]

  #Build the fragment itself
  foreach name $names x $addx y $addy z $addz elem $addelem {
    #puts "DEBUG: $name | $name | $myresname | 1 | A | 1 | $elem | $x | $y | $z"
    set newid [add_atom $name $name $myresname 1 "A" 1 $elem $x $y $z 0.0]
    lappend addedatoms $newid
  }

  #Add the specified bonds
  foreach atom1 $bondpart1 atom2 $bondpart2 order $bondorders {
    #puts "DEBUG: vmd_addbond [lindex $addedatoms $atom1] [lindex $addedatoms $atom2] $order"
    vmd_addbond [lindex $addedatoms $atom1] [lindex $addedatoms $atom2] $order
  }

  # Update molecule characteristics
  update_openvalence

  #Recenter view
  display resetview
}

proc ::Molefacture::read_fragment_file {dbfile} {
  #Proc to read a fragment DB file
  #This currently clobbers the old fragment DB in memory
  #DB file format will be documented elsewhere

  variable addfrags
  array unset addfrags

  set fragfile [open $dbfile "r"]
  while {[gets $fragfile line] >= 0} {
    set myarr [split $line]
    set name [lindex $myarr 0]
    set fname [lindex $myarr 1]
    set myarr [lreplace $myarr 0 0 [regsub -all _ $name " "]]
    set myarr [lreplace $myarr 1 1 [file join $::env(MOLEFACTUREDIR) lib fragments $fname]]
    array set addfrags $myarr
  }
  close $fragfile
    
}
    
proc ::Molefacture::read_basefrag_file {dbfile} {
  #Proc to read a base fragment DB file
  #This currently clobbers the old fragment DB in memory
  #DB file format will be documented elsewhere

  variable basefrags
  array unset basefrags

  set fragfile [open $dbfile "r"]
  while {[gets $fragfile line] >= 0} {
    set myarr [split $line]
    set name [lindex $myarr 0]
    set fname [lindex $myarr 1]
    set myarr [lreplace $myarr 0 0 [regsub -all _ $name " "]]
    set myarr [lreplace $myarr 1 1 [file join $::env(MOLEFACTUREDIR) lib basemol $fname]]
    array set basefrags $myarr
  }
  close $fragfile
    
}

proc ::Molefacture::add_basefrag_file {dbfile} {
  #Proc to read a base fragment DB file
  #This currently clobbers the old fragment DB in memory
  #DB file format will be documented elsewhere

  variable basefrags
#  array unset basefrags

  set fragfile [open $dbfile "r"]
  while {[gets $fragfile line] >= 0} {
    set myarr [split $line]
    set name [lindex $myarr 0]
    set fname [lindex $myarr 1]
    set myarr [lreplace $myarr 0 0 [regsub -all _ $name " "]]
    set myarr [lreplace $myarr 1 1 [file join [file dirname $dbfile] $fname]]
    if {[array get $name] != ""} {
      array unset $name
    }
    array set basefrags $myarr
  }
  close $fragfile

}

proc ::Molefacture::add_frag_file {dbfile} {
  #Proc to read a base fragment DB file
  #DB file format will be documented elsewhere

  variable addfrags
#  array unset basefrags

  set fragfile [open $dbfile "r"]
  while {[gets $fragfile line] >= 0} {
    set myarr [split $line]
    set name [lindex $myarr 0]
    set fname [lindex $myarr 1]
    set myarr [lreplace $myarr 0 0 [regsub -all _ $name " "]]
    set myarr [lreplace $myarr 1 1 [file join [file dirname $dbfile] $fname]]
    if {[array get $name] != ""} {
      array unset $name
    }
    array set addfrags $myarr
  }
  close $fragfile
}

proc ::Molefacture::set_prot_parent {} {
# Set the selected atom to be the target of future protein growth

  variable tmpmolid
  variable picklist
  if {[llength $picklist] != 1} {
    tk_messageBox -icon error -type ok -title "Error" -message "You need to select exactly one atom as the C-terminal end of the protein"
    return
  }

  set newparent [atomselect $tmpmolid "index $picklist"]
  if {[$newparent get element] != "H"} {
    tk_messageBox -icon warning -type ok -title "Select a hydrogen atom" -message "Please select a hydrogen atom as the template for future growth. This should be attached to the carbonyl carbon of your current C-terminal residue"
    return
  }

  set sel [atomselect $tmpmolid "(same residue as index $picklist) and name N CA C HC1 HN"]
  if {[$sel num] != 5} {
    tk_messageBox -icon warning -type ok -title "Warning" -message "Warning: Not all atoms necessary for protein growth are present in the selected residue. Further chain growth may not work correctly."
  }
  $sel delete

  $newparent set name "HC1"
  $newparent delete

  variable protlasthyd
  set protlasthyd $picklist
}



proc ::Molefacture::add_aa {aaname} {
  variable aapath
  variable protlasthyd
  variable tmpmolid

  if {$protlasthyd ==  -1} {
    ::Molefacture::new_mol_from_fragment [file join $aapath GLY-end.mfrag]
    set newsel [atomselect $tmpmolid all]
    $newsel set resname $aaname
    ::Molefacture::set_newmol_phipsi
  } else {
    set newselind [::Molefacture::add_amino_acid_base [file join $aapath GLY.mfrag] $protlasthyd $aaname]
    set newsel [atomselect $tmpmolid "index $newselind"]
  }

  # Set the new value of protlasthyd
  set protlasthyd [lindex [$newsel get index] [lsearch [$newsel get name] "HC1"]]
  set hareplace [lindex [$newsel get index] [lsearch [$newsel get name] "HA1"]]

  # Replace the appropriate hydrogen with the side chain
  if {$aaname != "GLY"} {
    set haother [atomselect $tmpmolid "index [lindex [$newsel get index] [lsearch [$newsel get name] HA2]]"]
    $haother set name "HA"
    $haother delete
    $newsel set resname $aaname
    replace_hydrogen_with_fragment [file join $aapath "$aaname.mfrag"] $hareplace
  }

  $newsel delete


}

proc ::Molefacture::add_nuc {nucname} {
# Add a nucleotide to a growing DNA or RNA chain
  variable nucpath
  variable nuclasthyd
  variable tmpmolid
  variable nuctype
  variable prevo3ind

  set nucbasefrag "dna_backbone"
  if {$nuctype == "RNA"} {
    set nucbasefrag "rna_backbone"
  }

  if {$nuclasthyd <  0 || $nuclasthyd == ""} {
    ::Molefacture::new_mol_from_fragment [file join $nucpath "$nucbasefrag.mfrag"]
    set newsel [atomselect $tmpmolid all]
    $newsel set resname $nucname
    set prevo3ind -1
  } else {
    # Get the index of the O3' atom, needed for dihedrals
    set lasthydsel [atomselect $tmpmolid "index $nuclasthyd"]
    set prevo3ind [join [join [$lasthydsel getbonds] ] ]
    $lasthydsel delete
    set newselind [::Molefacture::replace_hydrogen_with_fragment [file join $nucpath "${nucbasefrag}_added.mfrag"] $nuclasthyd 1]
    set newsel [atomselect $tmpmolid "index $newselind"]
    $newsel set resname $nucname
  }

  # Set the new value of protlasthyd
  set nuclasthyd [lindex [$newsel get index] [lsearch [$newsel get name] "HM1"]]
  set hareplace [lindex [$newsel get index] [lsearch [$newsel get name] "HX"]]

  $newsel set resname $nucname

  # Replace the appropriate dummy atom with the base
  set baseindices [replace_hydrogen_with_fragment [file join $nucpath "$nucname.mfrag"] $hareplace 1]
  $newsel delete

  mol reanalyze $tmpmolid

  # set dihedral angles
  ::Molefacture::set_nuc_diheds $baseindices
}

proc ::Molefacture::add_amino_acid_base {fragpath oldhydindex fragname {diheds 1}} {
  #This procedure will replace a hydrogen (which must have exactly 1 bond)
  #with a new amino acid. This should ONLY be used by the protein builder
  #This only builds the backbone; you need to add the side chain separately
  #Returns a selection containing the newly added atoms
  # If diheds is set to 0, the phi and psi angles are not adjusted

  variable tmpmolid
  variable maxaasize
  variable availablehyd

  set hydindex $oldhydindex

  set basesel [atomselect $tmpmolid "index $hydindex"]
  if {[llength [$basesel getbonds]]>1 || [$basesel get element] != "H"} {
      puts "You can only replace singly bonded hydrogen atoms!"
      return 1
  }

  # because we're going to make selections used during positioning, add more
  # dummy atoms now if they're needed
  if {$availablehyd <= $maxaasize} {
    set sel [atomselect $tmpmolid all]
    hrealloc $sel
    $sel delete
    $basesel delete
    set basesel [atomselect $tmpmolid "index $hydindex"]
  }

  # Open the fragment file and read all the atoms to be added
  set fragfile [open $fragpath "r"]
  gets $fragfile line
  set names [list]
  set addx [list]
  set addy [list]
  set addz [list]
  set addelem [list]
  #First get the h-replacing atom
  gets $fragfile line
  set linearray [split $line]
  set linearray [noblanks $linearray]
  set repname [lindex $linearray 0]
  set repx [lindex $linearray 1]
  set repy [lindex $linearray 2]
  set repz [lindex $linearray 3]
  set repelem [lindex $linearray 4]

  #Now all of the others
  gets $fragfile line
  while {[string range $line 0 5] != "BONDS"} {
    set linearray [split $line]
    set linearray [noblanks $linearray]
    lappend names [lindex $linearray 0]
    lappend addx [lindex $linearray 1]
    lappend addy [lindex $linearray 2]
    lappend addz [lindex $linearray 3]
    lappend addelem [lindex $linearray 4]
#  puts "$linearray"
    gets $fragfile line
  }

  #Finally, read the specified bonds
  set bondpart1 [list]
  set bondpart2 [list]
  set bondorders [list]
  while {[gets $fragfile line] != -1} {
    set linearray [split $line]
    set linearray [noblanks $linearray]
    lappend bondpart1 [lindex $linearray 0]
    lappend bondpart2 [lindex $linearray 1]
    lappend bondorders [lindex $linearray 2]
  }
  close $fragfile

  # Get residue/segment/etc information
  set myres $fragname
  set myresid [expr [$basesel get resid] + 1]
  set mysegname [$basesel get segname]
  set mysegid [$basesel get segid]

  #Orient our coordinate system
  set mothers [$basesel getbonds]
  set mindex [lindex [lindex $mothers 0] 0]
  set msel [atomselect $tmpmolid "index $mindex"]
  set mothersbonds [$msel getbonds]
  set mbsel [atomselect $tmpmolid "index [lindex $mothersbonds 0]"]
  set mcoind [lindex [$mbsel get index] [lsearch [$mbsel get name] "O"]]
  $mbsel delete
  set basecoor [lindex [$basesel get {x y z}] 0]
  set mcoor [lindex [$msel get {x y z}] 0]
  set myvec [vecsub $basecoor $mcoor]

  #Move the hydrogen to the proper place and set its characteristics
  #puts "basesel: index $hydindex [$basesel get index]"
  $basesel moveto [list $repx 0 0]

  set bondtarget [lindex [$basesel getbonds] 0]
  vmd_delbond $bondtarget $hydindex

  $basesel set occupancy 0.1
  set hydindex [add_atom $repname $repname $myres $myresid $mysegname $mysegid $repelem [join [$basesel get x]] [join [$basesel get y]] [join [$basesel get z]] 0.0]
  $basesel delete
  vmd_addbond $hydindex $bondtarget 1.0
  set basesel [atomselect $tmpmolid "index $hydindex"]
  #puts "Newbase: index $hydindex [$basesel get index]"
  #$basesel moveto [list $repx 0 0]
  #$basesel set name $repname
  #$basesel set element $repelem
  #$basesel set resid $myresid

  #List for processing bonds later
  set addedatoms [list]
  lappend addedatoms $hydindex

  #Now, the new piece is built near the origin, and then translated and rotated
  #to the proper position
  foreach name $names x $addx y $addy z $addz elem $addelem {
    set newid [add_atom $name $name $myres $myresid $mysegname $mysegid $elem $x $y $z 0.0]
    lappend addedatoms $newid
  }

  # Adjust the order of the first bond, if desired
  if {[lindex $bondpart2 0] == -1} {
    set order [lindex $bondorders 0]
    set bondpart1 [lreplace $bondpart1 0 0]
    set bondpart2 [lreplace $bondpart2 0 0]
    set bondorders [lreplace $bondorders 0 0]
    set i 1
    while {$i < $order} {
      raise_bondorder [list [lindex [$msel get index] 0] [lindex [$basesel get index] 0]]
      incr i
    }
  }

  #Add the specified bonds
  foreach atom1 $bondpart1 atom2 $bondpart2 order $bondorders {
    #puts "$atom1 $atom2 $order"
    vmd_addbond [lindex $addedatoms $atom1] [lindex $addedatoms $atom2] $order
  }

  #Properly orient and translate the new fragment
  set newsel [atomselect $tmpmolid "index $addedatoms"]
  $newsel move [transvec $myvec]
  $newsel moveby [list [$msel get x] [$msel get y] [$msel get z]]
#  $newsel moveby [vecscale $myvec $repx]

  # Set the proper phi/psi values
  # phi is between N and CA, psi between CA and C
  variable phi
  variable psi

  set nind [lindex [$newsel get index] [lsearch -exact [$newsel get name] N]]
  set caind [lindex [$newsel get index] [lsearch -exact [$newsel get name] CA]]
  set cind [lindex [$newsel get index] [lsearch -exact [$newsel get name] C]]
  set oind [lindex [$newsel get index] [lsearch -exact [$newsel get name] HC1]]
  set nhind [lindex [$newsel get index] [lsearch -exact [$newsel get name] HN]]

#puts "inds: $nind $caind $cind $oind $mcoind"
#puts "Source: [$newsel get name]"

  set prevind [lindex [$msel get index] 0]
  set casel [atomselect $tmpmolid "index $caind"]
  set csel [atomselect $tmpmolid "index $cind"]
  set nsel [atomselect $tmpmolid "index $nind"]
  set prevsel [atomselect $tmpmolid "index $prevind"]


  #Properly orient the backbone
  set ormovesel [atomselect $tmpmolid "(index [$newsel get index]) and not name N"]
  set dihedral [measure dihed [list [list $mcoind $tmpmolid] [list $prevind $tmpmolid] [list $nind $tmpmolid] [list $caind $tmpmolid] ]]

  set delta [expr $dihedral + 180]
  set mat [trans bond [lindex [$nsel get {x y z}] 0] [lindex [$prevsel get {x y z}] 0] $delta deg]
  $ormovesel move $mat
  set mat [trans bond [lindex [$nsel get {x y z}] 0] [lindex [$prevsel get {x y z}] 0] 180 deg]
  $ormovesel move $mat
  $ormovesel delete

  #Measure and set phi
  set phimovesel [atomselect $tmpmolid "(index [$newsel get index]) and not (name N HN CA)"]
  set dihedral [measure dihed [list [list $prevind $tmpmolid] [list $nind $tmpmolid] [list $caind $tmpmolid] [list $cind $tmpmolid] ]]

  set delta [expr -1 * ($phi - $dihedral)]
#puts "DEBUG: $dihedral $phi $delta"
#puts "DEBUG: [lindex [$casel get {x y z}] 0] [lindex [$basesel get {x y z}] 0]"
  set mat [trans bond [lindex [$casel get {x y z}] 0] [lindex [$basesel get {x y z}] 0] $delta deg]
  if {$diheds != 0} {$phimovesel move $mat}
#  puts "[lindex [lindex [label list Dihedrals] end] 4]"

  #Make sure we moved it right
#  set newdihed [lindex [lindex [label list Dihedrals] end] 4]
#  if {$newdihed > 0 && $phi < 0} {
#    set mat [trans bond [lindex [$casel get {x y z}] 0] [lindex [$basesel get {x y z}] 0] 180 deg]
#    $phimovesel move $mat
#  }

  $phimovesel delete

  #Measure and set psi
  set psimovesel [atomselect $tmpmolid "(index [$newsel get index]) and not (name N HN CA HA1 HA2)"]
  set dihedral [measure dihed [list [list $nind $tmpmolid] [list $caind $tmpmolid] [list $cind $tmpmolid] [list $oind $tmpmolid] ]]
  
  set delta [expr -1 * ($psi - $dihedral)]
#puts "DEBUG: $dihedral $psi $delta"
  set mat [trans bond [lindex [$csel get {x y z}] 0] [lindex [$casel get {x y z}] 0] $delta deg]
  if {$diheds != 0} {$psimovesel move $mat}
#  puts "[lindex [lindex [label list Dihedrals] end] 4]"

  #Make sure we moved it right
#  set newdihed [lindex [lindex [label list Dihedrals] end] 4]
#  if {$newdihed > 0 && $psi < 0} {
#    set mat [trans bond [lindex [$csel get {x y z}] 0] [lindex [$casel get {x y z}] 0] 180 deg]
#    $psimovesel move $mat
#  }

  $psimovesel delete

  # Update molecule characteristics
  update_openvalence

  set newselind [$newsel get index]

  #Delete atom selections
  $nsel delete
  $prevsel delete
  $newsel delete
  $msel delete
  $basesel delete
  $casel delete
  $csel delete

  return $newselind
}

proc ::Molefacture::set_nuc_diheds {baseindices} {
  #Set dihedral angles of a newly added nucleotide
  variable tmpmolid
  variable ralpha
  variable rbeta
  variable rgamma
  variable rdelta
  variable repsilon
  variable rchi
  variable rzeta
  variable nuclasthyd
  variable prevo3ind

  set ressel [atomselect $tmpmolid "same residue as index $nuclasthyd"]

  set atomids [$ressel get index]
  set atomnames [$ressel get name]

  $ressel delete

  set phosind [lindex $atomids [lsearch -exact $atomnames P]]
  set o5ind [lindex $atomids [lsearch -exact $atomnames O5']]
  set c5ind [lindex $atomids [lsearch -exact $atomnames C5']]
  set c4ind [lindex $atomids [lsearch -exact $atomnames C4']]
  set c3ind [lindex $atomids [lsearch -exact $atomnames C3']]
  set o3ind [lindex $atomids [lsearch -exact $atomnames O3']]
  set h3ind [lindex $atomids [lsearch -exact $atomnames H3']]
  set c1ind [lindex $atomids [lsearch -exact $atomnames C1']]
  set c2ind [lindex $atomids [lsearch -exact $atomnames C2']]

  # Measure and move the alpha and zeta angles (which involve the previous residue)
  if {$prevo3ind >= 0} {
    set zetamovesel [atomselect $tmpmolid "(same residue as index $phosind)"]
    set prevPsel [atomselect $tmpmolid "name P and same residue as index $prevo3ind"]
    set prevPind [lindex [$prevPsel get index] 0]
    $prevPsel delete
    #set_dihedral $zetamovesel $rzeta $prevPind $prevo3ind $phosind $o5ind

    set alphamovesel [atomselect $tmpmolid "(same residue as index $phosind) and (not index $phosind)"]
    #set_dihedral $alphamovesel $ralpha $prevo3ind $phosind $o5ind $c5ind
    $alphamovesel delete
  }

  # Measure and move the beta angle
  set betamovesel [atomselect $tmpmolid "(same residue as index $phosind) and (not index $phosind $o5ind and not name O1P O2P O5T)"]
  #set_dihedral $betamovesel $rbeta $phosind $o5ind $c5ind $c4ind
  $betamovesel delete

  # Measure and move the gamma angle
  set gammamovesel [atomselect $tmpmolid "(same residue as index $phosind) and (not index $phosind $o5ind c5ind $c4ind and not name O1P O2P O5T H5' H5'')"]
  #set_dihedral $gammamovesel $rgamma $o5ind $c5ind $c4ind $c3ind
  $gammamovesel delete

  # Measure and move the delta angle
  # this one is a bit tricky due to the ring closure; we pretend the c4-o1 bond
  #  doesn't exist
  set deltamovesel [atomselect $tmpmolid "(same residue as index $phosind) and (not index $phosind $o5ind c5ind $c4ind $c3ind and not name O1P O2P O5T H5' H5'')"]
  #set_dihedral $deltamovesel $rdelta $c5ind $c4ind $c3ind $o3ind
  $deltamovesel delete

  # measure and move the epsilon angle
  set epsilonmovesel [atomselect $tmpmolid "index $o3ind or (name HM1 and same residue as index $o3ind)"]
  #set_dihedral $epsilonmovesel $repsilon $c4ind $c3ind $o3ind $h3ind
  $epsilonmovesel delete

  # Measure and move the chi angle, which includes the base
  set chimovesel [atomselect $tmpmolid "index $baseindices and not index [lindex $baseindices 0]"]
  #set_dihedral $chimovesel $rchi $c2ind $c1ind [lindex $baseindices 0] [lindex $baseindices 1]
  $chimovesel delete

}

proc ::Molefacture::set_newmol_phipsi {} {
  #Set phi and psi angles of the first amino acid in a chain
  variable tmpmolid
  variable phi
  variable psi

  set allsel [atomselect $tmpmolid "occupancy >= 0.4"]

  set prevind [lindex [$allsel get index] [lsearch -exact [$allsel get name] HT2]]
  set nind [lindex [$allsel get index] [lsearch -exact [$allsel get name] N]]
  set caind [lindex [$allsel get index] [lsearch -exact [$allsel get name] CA]]
  set cind [lindex [$allsel get index] [lsearch -exact [$allsel get name] C]]
  set oind [lindex [$allsel get index] [lsearch -exact [$allsel get name] HC1]]

  set basesel [atomselect $tmpmolid "index $nind"]
  set casel [atomselect $tmpmolid "index $caind"]
  set csel [atomselect $tmpmolid "index $cind"]
  
  #Measure and set phi
  set phimovesel [atomselect $tmpmolid "not (name N HT1 HT2 CA)"]
  set dihedral [measure dihed [list [list $prevind $tmpmolid] [list $nind $tmpmolid] [list $caind $tmpmolid] [list $cind $tmpmolid] ]]

  set delta [expr -1 * ($dihedral - $phi)]
  set mat [trans bond [lindex [$casel get {x y z}] 0] [lindex [$basesel get {x y z}] 0] $delta deg]
  $phimovesel move $mat
  $phimovesel delete

  #Measure and set psi
  set psimovesel [atomselect $tmpmolid "not (name N HT1 HT2 CA HA1 HA2)"]
  set dihedral [measure dihed [list [list $nind $tmpmolid] [list $caind $tmpmolid] [list $cind $tmpmolid] [list $oind $tmpmolid] ]]
  
  set delta [expr -1 * ($dihedral - $psi)]
  set mat [trans bond [lindex [$csel get {x y z}] 0] [lindex [$casel get {x y z}] 0] $delta deg]
  $psimovesel move $mat
  $psimovesel delete

  $csel delete
  $allsel delete
  $casel delete
  $basesel delete

}

proc ::Molefacture::add_proline {} {
  variable aapath
  variable protlasthyd
  variable tmpmolid

  set aaname "PRO"

  if {$protlasthyd ==  -1} {
    ::Molefacture::new_mol_from_fragment [file join $aapath PRO.mfrag]
    set newsel [atomselect $tmpmolid all]
    $newsel set resname $aaname
#    ::Molefacture::set_newmol_phipsi
  } else {
    set newselind [::Molefacture::add_amino_acid_base [file join $aapath PRO.mfrag] $protlasthyd $aaname 0]
    set newsel [atomselect $tmpmolid "index $newselind"]
  }

  # Set the new value of protlasthyd
  set protlasthyd [lindex [$newsel get index] [lsearch [$newsel get name] "HC1"]]
#  set hareplace [lindex [$newsel get index] [lsearch [$newsel get name] "HA1"]]

  $newsel delete


}

proc ::Molefacture::write_topfile {filename selection} {
# Write a topology file containing all the entries necessary for the current molecule
# For now, internal coordinates, DONOR cards, and the like are ignored

  variable improper_centeratoms

  variable tmpmolid

  variable mass_by_element

  unique_atomnames

  set sel [atomselect $tmpmolid "($selection) and (occupancy>=0.8)"]

  set topfile [open $filename w]
  set topcontents [list] ;# List for storing things that should be written at the end

  puts $topfile "*>>>>>> CHARMM topology file generated by Molefacture <<<<<<"
  puts $topfile "27 1"
  puts $topfile ""

  set resnames [list] ;# Names of all residues encountered so far
  set atomnames [list] ;# All atom types needed
  set atomelems [list] ;# Elements of all atoms in atomnames
  set atommasses [list] ;# Masses of all atoms in atommasses

#Loop through residues and write the necessary entries
    foreach resid [lsort -unique [$sel get resid]] {
      set ressel [atomselect $tmpmolid "($selection) and (occupancy >= 0.5) and (resid $resid)"]
      array set namearray {} 
      set nbonds 1 ;# Number of bonds written so far

      # Get residue info
      set myresname [lindex [$ressel get resname] 0]
      set mycharge [vecsum [$ressel get charge]]

      # Make sure it isn't a duplicate
      if {[lsearch -exact $resnames $myresname] != -1} {
        puts "Warning: Found multiple instances of residue $myresname. Only the first will be written."
        continue
      }

      lappend resnames $myresname

      #Write the residue heading
      lappend topcontents [format "RESI %4s    % 5.2f\n" $myresname $mycharge]
      lappend topcontents "GROUP\n"

      # Loop through the atoms and write them all
      # In the process, record impropers that need to be written


      foreach index [$ressel get index] name [$ressel get name] type [$ressel get type] charge [$ressel get charge] element [$ressel get element] mass [$ressel get mass] {

      # If this has a hydrogen-like mass, it could be left over from
      # molecule building
      if {$mass < 2.0} {
#        puts "Looking for mass of element $element"
        set tmpmass [array get mass_by_element $element]
#        puts "got $tmpmass"
        if {$tmpmass != {}} {
          set mass [lindex $tmpmass 1]
        }
      }


      # Add this type to the types database if it isn't already there
        set currloc [lsearch -exact $atomnames $type]
        if {$currloc == -1} {
          # Then list the new atom
          lappend atomnames $type
          lappend atomelems $element
          lappend atommasses $mass
        } else {
          if {[string trim [lindex $atomelems $currloc]] != [string trim $element] || [expr [lindex $atommasses $currloc] - $mass] >= 0.001} {
             puts "Warning: Multiple incompatible definitions of type $type. Using the first"
          }
        }

        lappend topcontents [format "ATOM %4s %4s % .5f\n" $name $type $charge]
        array set namearray [list $index $name]
      }

      # Loop through again and this time write all the bonds we need
       lappend topcontents "BOND "
      foreach name [$ressel get name] index [$ressel get index] bonds [$ressel getbonds] bondorders [$ressel getbondorders] {
#        puts "DEBUG: bond $bonds | bo $bondorders"
        foreach bond $bonds bo $bondorders {
          if {$bo != 0 && $index < $bond} {
           #Make sure neither atom is missing
            if {$name == "" || [lindex [array get namearray $bond] 1] == ""} {
              puts "Warning: Bond found between two residues; it will be ignored"
              continue
            }
            if {[expr $nbonds % 4] == 0} {
       #       puts "DEBUG: nbonds $nbonds"
              lappend topcontents "\n"
              lappend topcontents "BOND "
            }

#            puts "DEBUG: [array get namearray $bond]"
#            puts "DEBUG: name $name | othername: [array get namearray $bond]"
           lappend topcontents "$name [lindex [array get namearray $bond] 1]  "
       #    puts "appended bond $name [lindex [array get namearray $bond] 1]"
           incr nbonds
          }
        }
      }

      # Go through one more time, and add impropers for all atoms that have appropriate types and exactly three bonds
      foreach name [$ressel get name] type [$ressel get type] index [$ressel get index] bonds [$ressel getbonds] {
        if { [lsearch $improper_centeratoms $type] >= 0 && [llength $bonds] == 3} {
          set bondednames [list]
          foreach bind $bonds {
            set tmpind [lsearch [$ressel get index] $bind]
            set othername [lindex [$ressel get name] $tmpind]
            lappend bondednames $othername
          }
          lappend topcontents "\nIMPR $name [lindex $bondednames 0] [lindex $bondednames 1] [lindex $bondednames 2]"
        }
      }


      lappend topcontents "\n\n"

     #Now we're done with the resudie
     $ressel delete
    }

  # Now, write all the MASS tags
  set i 1
  foreach type $atomnames elem $atomelems mass $atommasses {
    set ostring [format "MASS %5i %-4s  %8.5f %2s" $i $type $mass $elem]
    puts $topfile $ostring
  }

  puts $topfile ""
  puts $topfile "AUTO ANGLES DIHE"
  puts $topfile ""

# Finally, output all the residues
  foreach line $topcontents {
    puts -nonewline $topfile $line
  }

  puts $topfile "\nEND\n\n"

# Clean up
  close $topfile
  $sel delete
}

proc ::Molefacture::write_fep_hybrid_topfile {filename constantselout fullselin outgoingsel incomingsel totalcharge} {
# write a hybrid topology file for morphing between outgoingsel and incomingsel

  variable tmpmolid

  variable mass_by_element

  set topfile [open $filename w]
  set topcontents [list] ;# List for storing things that should be written at the end

  puts $topfile "*>>>>>> CHARMM topology file generated by Molefacture <<<<<<"
  puts $topfile "27 1"
  puts $topfile ""

  set resnames [list] ;# Names of all residues encountered so far
  set atomnames [list] ;# All atom types needed
  set atomelems [list] ;# Elements of all atoms in atomnames
  set atommasses [list] ;# Masses of all atoms in atommasses

#Loop through residues and write the necessary entries
  set myresname [lindex [$constantselout get resname] 0]
  array set namearray {} 
  set nbonds 1 ;# Number of bonds written so far

  #Write the residue heading
  lappend topcontents [format "RESI %4s    % .5f\n" $myresname $totalcharge]
  lappend topcontents "GROUP\n"

  array set oldcomnamearray {}
  array set newcomnamearray {}

  foreach index [$constantselout get index] name [$constantselout get name] {
    array set oldcomnamearray [list $index $name]
  }

  foreach index [$fullselin get index] name [$fullselin get name] {
    array set newcomnamearray [list $index $name]
  }

  # Loop through the atoms and write them all
  set addedbonds [list]

  foreach ressel [list $constantselout $outgoingsel $incomingsel] selid [list 0 1 2] {
    array unset namearray 
    foreach index [$ressel get index] name [$ressel get name] type [$ressel get type] charge [$ressel get charge] element [$ressel get element] mass [$ressel get mass] {

      # If this has a hydrogen-like mass, it could be left over from
      # molecule building
      if {$mass < 2.0} {
        #        puts "Looking for mass of element $element"
        set tmpmass [array get mass_by_element $element]
        #        puts "got $tmpmass"
        if {$tmpmass != {}} {
          set mass [lindex $tmpmass 1]
      }
    }

      # Add this type to the types database if it isn't already there
        set currloc [lsearch -exact $atomnames $type]
        if {$currloc == -1} {
          # Then list the new atom
          lappend atomnames $type
          lappend atomelems $element
          lappend atommasses $mass
        } else {
          #puts "Loc: $currloc"
          #puts [lindex atomelems $currloc]
#          if {[lindex atomelems $currloc] != $element || [lindex $atommasses $currloc] != $mass}
        }

        lappend topcontents [format "ATOM %4s %4s % .5f\n" $name $type $charge]
        array set namearray [list $index $name]
      }

      # Loop through again and this time write all the bonds we need
       #lappend topcontents "BOND "
       set nbonds 1
      foreach name [$ressel get name] index [$ressel get index] bonds [$ressel getbonds] bondorders [$ressel getbondorders] {
#        puts "DEBUG: bond $bonds | bo $bondorders"
        foreach bond $bonds bo $bondorders {
          if {$bo != 0} {
           #Make sure neither atom is missing
            set othername [lindex [array get namearray $bond] 1]
            if {$othername == ""} {
              if {$selid == 1} {
                set othername [lindex [array get oldcomnamearray $bond] 1]
              } 
              if {$selid == 2} {
                set othername [lindex [array get newcomnamearray $bond] 1]
              }
            }
            if {$othername == ""} {
              continue
            }

           if {[string compare $name $othername] > 0} {
             set bondstring "$name $othername  "
           } else {
             set bondstring "$othername $name  "
           }
           if {[lsearch $addedbonds $bondstring] < 0} {
              if {[expr $nbonds % 4] == 1} {
                lappend topcontents "BOND "
              }
             lappend topcontents $bondstring
             lappend addedbonds $bondstring
              if {[expr $nbonds % 4] == 0} {
         #       puts "DEBUG: nbonds $nbonds"
                lappend topcontents "\n"
                #lappend topcontents "BOND "
              }
             incr nbonds
           }
          }
        }
      }
      lappend topcontents "\n"
      }
      lappend topcontents "\n"

  # Now, write all the MASS tags
  set i 1
  foreach type $atomnames elem $atomelems mass $atommasses {
    set ostring [format "MASS %5i %-4s  %8.5f %2s" $i $type $mass $elem]
    puts $topfile $ostring
  }

  puts $topfile ""
  puts $topfile "AUTO ANGLES DIHE"
  puts $topfile ""

# Finally, output all the residues
  foreach line $topcontents {
    puts -nonewline $topfile $line
  }

  puts $topfile "\nEND\n\n"

# Clean up
  close $topfile
}

proc ::Molefacture::set_dihedral {movesel newval ind1 ind2 ind3 ind4} {
# Acting on atoms in tmpmolid, move the selection movesel such that
#  the dihedral formed by the input atom indices is $newval
  variable tmpmolid

# Validate input
  if {$ind1 == "" || $ind2 == "" || $ind3 == "" || $ind4 == ""} {
    puts "WARNING: Couldn't find one or more atoms for set_dihedral"
    return
  }

  set dihedral [measure dihed [list [list $ind1 $tmpmolid] [list $ind2 $tmpmolid] [list $ind3 $tmpmolid] [list $ind4 $tmpmolid] ]]

  set bsel1 [atomselect $tmpmolid "index $ind2"]
  set bsel2 [atomselect $tmpmolid "index $ind3"]

  set delta [expr -1 * ($dihedral - $newval)]
  set mat [trans bond [lindex [$bsel1 get {x y z}] 0] [lindex [$bsel2 get {x y z}] 0] $delta deg]
  $movesel move $mat

  $bsel1 delete
  $bsel2 delete
}






