#
# ssrestraints - generates NAMD config file parameters to restraint secondary
# structure elements.
#
# Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
# Elizabeth Villa <villa@ks.uiuc.edu>
#
# $Id: ssrestraints.tcl,v 1.15 2013/04/15 17:43:07 johns Exp $
#

package require rnaview
package require ssrestraints_stride
package provide ssrestraints 1.1

namespace eval ::SSRestraints:: {

  variable defaultKprot   200
  variable defaultKnadih  200
  variable defaultKnabond 200
  variable defaultSelText "helix or extended_beta or nucleic"
#  variable defaultBpType "*"
#  variable defaultBpCt   "*"
  variable defaultNA 3
  variable cutoffBP 10
  variable goodBP {{+/+} {-/-} {W/W} {stacked} {syn} {H/S} {S/H} {H/H} {H/W} {W/H} {W/S} {S/W}}
  #{H/.} {./H} {H/?} {?/H}
  variable ideal 0

  # H-bond restraints
  variable defaultHB 0 ;# for now, turned off by default
  variable defaultHBDonorSel "name N and backbone and (helix or extended_beta)"
  variable defaultHBAcceptorSel "name O and backbone and (helix or extended_beta)"
  variable defaultHBBondK 20.0
  variable defaultHBAngleK 20.0
  variable defaultHBDcut 3.5
  variable defaultHBAcut 35.0

}

proc ssrestraints { args } { return [eval ::SSRestraints::ssrestraints $args] }

proc ::SSRestraints::ssrestraints_usage { } {

  variable defaultSelText
  variable defaultKprot   
  variable defaultKnadih  
  variable defaultKnabond 
#  variable defaultBpType
#  variable defaultBpCt 
  variable defaultNA
  variable defaultHBDonorSel
  variable defaultHBAcceptorSel
  variable defaultHBBondK
  variable defaultHBAngleK
  variable defaultHBDcut
  variable defaultHBAcut

  puts "Usage: ssrestraints -psf <input psf> -pdb <input pdb> -o <output file> ?options?"
  puts "Options:"
  puts "  -sel <selection text> (default: $defaultSelText)"
  puts "  -k_prot <force constant for protein dihedrals> (default: $defaultKprot)"
  puts "  -k_na_bond <force constant for the nucleic acid bonds> (default: $defaultKnabond)"
  puts "  -k_na_dih <foce constant for the nucleic acid dihedrals> (default: $defaultKnadih)"
  puts "  -ideal -- use ideal phi and psi angles for alpha helices, 3-10 helices, and beta strands"
  puts "  -hbonds -- restrain H-bonds"
  puts "  -hbdonorsel <H-bond donor selection text> (default: $defaultHBDonorSel)"
  puts "  -hbaccsel <H-bond acceptor selection text> (default: $defaultHBAcceptorSel)"
  puts "  -hbbondk <force constant for H-bonds> (default: $defaultHBBondK)"
  puts "  -hbanglek <force constant for H-bonds> (default: $defaultHBAngleK)"
  puts "  -hbdcut <H-bond distance cut-off in Angstroms> (default: $defaultHBDcut)"
  puts "  -hbacut <H-bond angle cut-off in degrees> (default: $defaultHBAcut)"
  puts "  -labels -- add labels for visualization of restraints"
#  puts "  -bptype <base pair type> See RNAVIEW documentation (default: $defaultBpType)"
#  puts "  -bpct < cis or trans base pair> See RNAVIEW documentation  (default: $defaultBpCt)"
  puts "  -na <restraints for nucleic acid> (default: $defaultNA)"
  puts "      0: dihedrals of all residues"
  puts "      1: dihedrals of base-paired residues"
  puts "      2: two bonds between base paired residues"
  puts "      3: both options 1 and 2"
  return

}

proc ::SSRestraints::ssrestraints { args } {

  variable defaultSelText
  variable defaultKprot   
  variable defaultKnadih  
  variable defaultKnabond 
  variable defaultNA
#  variable defaultBpType
#  variable defaultBpCt 
  variable defaultHB
  variable defaultHBDonorSel
  variable defaultHBAcceptorSel
  variable defaultHBBondK
  variable defaultHBAngleK
  variable defaultHBDcut
  variable defaultHBAcut

  variable molid
  variable seltext
  variable k_prot
  variable k_na_dih
  variable k_na_bond
  variable out
  variable labels
#  variable bptype
#  variable bpct
  variable convertResid
  variable ideal

  set nargs [llength $args]
  if { $nargs == 0 } {
    ssrestraints_usage
    error ""
  }

  # should we add labels?
  set pos [lsearch -exact $args {-labels}]
  if { $pos != -1 } {
    set labels 1
    set args [lreplace $args $pos $pos]
  } else {
    set labels 0
  }

  # should we consider ideal equilibrium values?
  set pos [lsearch -exact $args {-ideal}]
  if { $pos != -1 } {
    set ideal 1
    set args [lreplace $args $pos $pos]
  } 

  # should we restrain H-bonds?
  set pos [lsearch -exact $args {-hbonds}]
  if { $pos != -1 } {
    set hbonds 1
    set args [lreplace $args $pos $pos]
  } else {
    set hbonds $defaultHB
  }
  
  foreach {name val} $args {
    switch -- $name {
      -psf       { set arg(psf)       $val }
      -pdb       { set arg(pdb)       $val }
      -sel       { set arg(sel)       $val }
      -o         { set arg(o)         $val }
      -k_prot    { set arg(k_prot)    $val }
      -k_na_bond { set arg(k_na_bond) $val }
      -k_na_dih  { set arg(k_na_dih)  $val }
#      -bptype { set arg(bptype) $val}
#      -bpct   { set arg(bpct)   $val}
      -na { set arg(na) $val }
      -hbdonorsel { set arg(hbdonorsel) $val }
      -hbaccsel { set arg(hbaccsel) $val }
      -hbbondk { set arg(hbbondk) $val }
      -hbanglek { set arg(hbanglek) $val }
      -hbdcut { set arg(hbdcut) $val }
      -hbacut { set arg(hbacut) $val }
    }
  }

  if { [info exists arg(sel)] } {
    set seltext $arg(sel)
  } else {
    set seltext $defaultSelText
  }

  if { [info exists arg(o)] } {
    set outputFileName $arg(o)
  } else {
    error "Missing output file."
  }

  if { [info exists arg(psf)] } {
    set psffile $arg(psf)
  } else {
    error "Missing psf file."
  }

  if { [info exists arg(pdb)] } {
    set pdbfile $arg(pdb)
  } else {
    error "Missing pdb file."
  }

  if { [info exists arg(k_prot)] } {
    set k_prot $arg(k_prot)
  } else {
    set k_prot $defaultKprot
  }

  if { [info exists arg(k_na_dih)] } {
    set k_na_dih $arg(k_na_dih)
  } else {
    set k_na_dih $defaultKnadih
  }

  if { [info exists arg(k_na_bond)] } {
    set k_na_bond $arg(k_na_bond)
  } else {
    set k_na_bond $defaultKnabond
  }

  if { [info exists arg(na)] } {
    if {$arg(na) == 0 || $arg(na) == 1 || $arg(na) == 2 || $arg(na) == 3} { 
      set na $arg(na)
    } else {
      error "Option -na can only be 0, 1, 2, or 3."
    }
  } else {
    set na $defaultNA
  }

#  if { [info exists arg(bptype)] } {
#    set bptype $arg(bptype)
#  } else {
#    set bptype $defaultBpType
#  }


#  if { [info exists arg(bpct)] } {
#    set bpct $arg(bpct)
#  } else {
#    set bpct $defaultBpCt
#  }

  # H-bonds options

  if { [info exists arg(hbdonorsel)] } {
    set hbdonorsel $arg(hbdonorsel)
  } else {
    set hbdonorsel $defaultHBDonorSel
  }

  if { [info exists arg(hbaccsel)] } {
    set hbaccsel $arg(hbaccsel)
  } else {
    set hbaccsel $defaultHBAcceptorSel
  }

  if { [info exists arg(hbbondk)] } {
    set hbbondk $arg(hbbondk)
  } else {
    set hbbondk $defaultHBBondK
  }

  if { [info exists arg(hbanglek)] } {
    set hbanglek $arg(hbanglek)
  } else {
    set hbanglek $defaultHBAngleK
  }

  if { [info exists arg(hbdcut)] } {
    set hbdcut $arg(hbdcut)
  } else {
    set hbdcut $defaultHBDcut
  }

  if { [info exists arg(hbacut)] } {
    set hbacut $arg(hbacut)
  } else {
    set hbacut $defaultHBAcut
  }

  set molid [mol new $psffile type psf waitfor all]
  mol addfile $pdbfile type pdb waitfor all

  set sel [atomselect $molid "protein"]
  if { [$sel num] != 0 } {
    ::SSRestraints::STRIDE::create_structure_macros -mol $molid -split segname -o tmp_structure_macros.tcl
    ::SSRestraints::STRIDE::set_structure tmp_structure_macros.tcl -mol $molid
    file delete tmp_structure_macros.tcl
  }
  $sel delete

  # extraBonds config file for NAMD
  set out [open $outputFileName w]

  if $hbonds {
    ::SSRestraints::restrain_hbonds $hbdonorsel $hbaccsel $hbbondk $hbanglek $hbdcut $hbacut
  }

  ::SSRestraints::restrain_protein
  ::SSRestraints::restrain_na $na

  close $out
  set ideal 0

  return

}

### PROTEIN ###

proc ::SSRestraints::restrain_protein { } {

  variable molid
  variable seltext

  set sel [atomselect $molid "protein and backbone and ($seltext)"]
  if { [$sel num] == 0 } {
    puts "ssrestraints) The given selection does not contain protein."
    $sel delete
    return
  }
  puts "ssrestraints) Restraining protein residues..."
  set selRestrainResidues [lsort -unique [$sel get residue]]
  $sel delete

  foreach residue $selRestrainResidues {
    ::SSRestraints::restrain_protein_residue $residue
  }
  unset selRestrainResidues

  return

}



# Restrain phi and psi angles of a given residue
proc ::SSRestraints::restrain_protein_residue { residue } {

  variable molid
  variable out
  variable k_prot
  variable labels
  variable ideal

  # For both phi and psi angles, we need N, CA, and C. If any of these
  # atoms cannot be found, that's a deal break
  set selN [atomselect $molid "residue $residue and name N"]
  set selCA [atomselect $molid "residue $residue and name CA"]
  set selC [atomselect $molid "residue $residue and name C"]
  if { [$selN num] == 0 || [$selCA num] == 0 || [$selC num] == 0 } {
    puts "ssrestraints) Warning: Protein residue $residue will be ignored due to missing atoms."
    $selN delete
    $selCA delete
    $selC delete
    return
  }
  set indexN [$selN get index]
  set indexCA [$selCA get index]
  set indexC [$selC get index]

  #
  # Restrain the phi dihedral angle
  #

  # Find the index of the C atom from the previous residue by searching
  # the atoms bonded to N for an atom named C, as in AtomSel.C
  set indexCprev -1
  foreach index [lindex [$selN getbonds] 0] {
    set selIndex [atomselect $molid "index $index"]
    if { [$selIndex get name] == "C" } {
      set indexCprev $index
      $selIndex delete
      break
    }
    $selIndex delete
  }
  if { $indexCprev == -1 } {
    puts "ssrestraints) Warning: Phi dihedral angle of protein residue $residue will not be restrained due to missing previous C atom."
  } else {
    if $ideal {
      set struct [$selN get structure] 
      switch $struct  {
        #B {set phi XX}
        #C {set phi XX}
        E {set phi -120} 
        G {set phi -74}
        H {set phi -57} 
        T {set phi -60} 
        default {set phi 0}
      }
    } else {
      set phi [$selN get phi]
    }
    puts $out "dihedral $indexCprev $indexN $indexCA $indexC $k_prot $phi"
    if $labels {
      label add Dihedrals $molid/$indexCprev $molid/$indexN $molid/$indexCA $molid/$indexC
    }
  }

  #
  # Restrain the psi dihedral angle
  #
  
  # Find the index of the N atom from the next residue by searching the
  # atoms bonded to C for an atom named N, as in AtomSel.C
  set indexNnext -1
  foreach index [lindex [$selC getbonds] 0] {
    set selIndex [atomselect $molid "index $index"]
    if { [$selIndex get name] == "N" } {
      set indexNnext $index
      $selIndex delete
      break
    }
    $selIndex delete
  }
  if { $indexNnext == -1 } {
    puts "ssrestraints: Warning: Psi dihedral angle of protein residue $residue will not be restrained due to missing next N atom."
  } else {
    if $ideal {
      set struct [$selN get structure]
      switch $struct  {
        #B {set phi XX}
        #C {set phi XX}
        E {set psi 113}
        G {set psi -4}
        H {set psi -47}
        T {set psi 30}
        default {set psi 0}
      }
    } else {
      set psi [$selC get psi]
    }
    puts $out "dihedral $indexN $indexCA $indexC $indexNnext $k_prot $psi"
    if $labels {
      label add Dihedrals $molid/$indexN $molid/$indexCA $molid/$indexC $molid/$indexNnext
    }
  }

  $selN delete
  $selCA delete
  $selC delete

  return

}


### NUCLEIC ACIDS ###

proc ::SSRestraints::getIndex { residue name } {

  variable molid

  set sel [atomselect $molid "residue $residue and name $name"]
  if { [$sel num] == 0 } {
    $sel delete
    return -1
  }
  set index [$sel get index]
  $sel delete

  return $index

}

proc ::SSRestraints::restrain_na { na } {
  
  variable molid
  variable seltext
  variable k_na_bond
  variable out
  variable labels
  variable convertResid
  variable cutoffBP 
  variable goodBP
#  variable bptype
#  variable bpct 

  set sel [atomselect $molid "nucleic and ($seltext)"]
  if { [$sel num] == 0 } {
    puts "ssrestraints) The given selection does not contain nucleic acid."
    $sel delete
    return
  }
  #puts "ssrestraints) Restraining nucleic acid base pairs..."

#  -na <restraints for nucleic acid> (default: $defaultNA)"
#      0: dihedrals of all residues"
#      1: dihedrals of base paired residues"
#      2: two bonds between base-paired residues"
#      3: both options 1 and 2"

  if { $na == 0 } {
    puts "ssrestraints) Restraining dihedrals of all nucleic acid residues in the selection..."
    foreach residue [lsort -unique [$sel get residue]] {
      ::SSRestraints::restrain_na_residue $residue
    }
    $sel delete
    return
  }

  if { $na == 2 || $na == 3 } {
    puts "ssrestraints) Defining two bonds between base-paired residues..."
    set bonds 1
  } else {
    set bonds 0
  }

  if { $na == 1 || $na == 3 } {
    puts "ssrestraints) Restraining dihedrals of base-paired residues..."
    set dihedrals 1
  } else {
    set dihedrals 0
  }

  # Create a psf/pdb with unique resids to calculate base pairs
  set all [atomselect $molid all]
  $all writepsf "tmp_unique_resid.psf"
  $all writepdb "tmp_unique_resid.pdb"
  set molidUniqueResid [mol new "tmp_unique_resid.psf" type psf waitfor all]
  mol addfile "tmp_unique_resid.pdb" type pdb waitfor all
  
  # Get the indices of the atoms in the selection
  set selInd [$sel get index]
  
  # Assign unique resids and create a conversion table
  set uniqueResid 1
  foreach residue [lsort -integer -unique [$sel get residue]] {
    set selRes [atomselect $molidUniqueResid "residue $residue"]
    $selRes set resid $uniqueResid
    $selRes delete
    # convertion table to go back to residue after base-pair analysis
    set convertResid($uniqueResid) $residue
    incr uniqueResid
  }
  $sel delete
  $all delete
  
  # Get base pair information
  rnaview -mol $molidUniqueResid -sel "index $selInd" -bpseq "rnaview.bpseq" -ext

  # Delete temporary molecule
  mol delete $molidUniqueResid

  # Delete intermediate files
  file delete base_pair_statistics.out rnaview.pdb.out \
      tmp_unique_resid.psf tmp_unique_resid.pdb

  # Parse bpseq to get a list of base pairs that will be restrained
  set file [open "rnaview.bpseq" r]

  set bondList {}
  set dihList {}

  while { ![eof $file] } {
    gets $file line 
    # ignore blank lines
    if { ![regexp -expanded -- {^[\t ]*$} $line] } {
      set uniqueResid1  [lindex $line 0]
      set resname1      [lindex $line 1]
      set uniqueResid2  [lindex $line 2]
      set bptype        [lindex $line 3]

      # if this residue is not base paired, continue
      if { $uniqueResid2 == 0 || [lsearch -exact $goodBP $bptype] == -1} {
        continue
      }
          
      set residue1 $convertResid($uniqueResid1)
      set residue2 $convertResid($uniqueResid2)

      set sel1 [atomselect $molid "residue $residue1"]
      set sel2 [atomselect $molid "residue $residue2"]
      set resname1 [lsort -unique [$sel1 get resname]]
      set resname2 [lsort -unique [$sel2 get resname]]
      $sel1 delete
      $sel2 delete

      switch -- $resname1 {
        A - ADE - RA - G - GUA - RG {
          set index1a [getIndex $residue1 "N9"]
          set index1b [getIndex $residue1 "N1"]
        }
        T - THY - U - URA - RU - C - CYT - RC {
          set index1a [getIndex $residue1 "N1"]
          set index1b [getIndex $residue1 "N3"]
        }
        default {
          puts "ssrestraints) Warning: unrecognized resname $resname1 for residue $residue1. This base pair will not be restrained."
          continue
        }
      }

      switch -- $resname2 {
        A - ADE - RA - G - GUA - RG {
          set index2a [getIndex $residue2 "N9"]
          set index2b [getIndex $residue2 "N1"]
        }
        T - THY - U - URA - RU - C - CYT - RC {
          set index2a [getIndex $residue2 "N1"]
          set index2b [getIndex $residue2 "N3"]
        }
        default {
          puts "ssrestraints) Warning: unrecognized resname $resname2 for residue $residue2. This base pair will not be restrained."
          continue
        }
      }

      if { $index1a == -1 || $index1b == -1 ||
           $index2a == -1 || $index2b == -1 } {
        puts "ssrestraints) Warning: base pair between residues $residue1 and $residue2 will not be restrained due to missing atoms."
        continue
      }

      # Is this base pair too far appart? 
      set refb [measure bond [list $index1b $index2b] molid $molid]
      if { $refb > $cutoffBP } {
        puts "ssrestraints) Warning: residues $residue1 ($resname1) and $residue2 ($resname2) are too far apart and will not be restrained."
        continue
      } else {

        if $bonds {

          # Was this bond already created?
          if { [lsearch -regexp $bondList "$index1b:$index2b"] == -1 } {
            lappend bondList "$index1b:$index2b"
            lappend bondList "$index2b:$index1b"
            
            puts $out "bond $index1b $index2b $k_na_bond $refb"
            
            # add bonds for visualization
            if $labels {
              label add Bonds $molid/$index1b $molid/$index2b
            }
          }
          
          # Was this bond already created?
          if { [lsearch -regexp $bondList "$index1a:$index2a"] == -1 } {
            lappend bondList "$index1a:$index2a"
            lappend bondList "$index2a:$index1a"
            
            set refa [measure bond [list $index1a $index2a] molid $molid]
            puts $out "bond $index1a $index2a $k_na_bond $refa"
            
            # add bonds for visualization
            if $labels {
              label add Bonds $molid/$index1a $molid/$index2a
            }
          }

        }

        # add residues to dihList (will remove redundancy later)
        if $dihedrals {
          lappend dihList $residue1
          lappend dihList $residue2
        }

      }

    } ;# if { ![regexp -expanded -- {^[\t ]*$} $line] }

  } ;# while { ![eof $file] } 

  close $file

  # Restrain dihedrals 
  if $dihedrals {
    foreach res [lsort -unique $dihList] {
      ::SSRestraints::restrain_na_residue $res
    }
  }

  unset bondList
  unset dihList
  unset convertResid

  return

}
 
# Restrain dihedral angles of a given residue
proc ::SSRestraints::restrain_na_residue { residue } {

  variable molid
  variable out
  variable k_na_dih
  variable labels

  #  The atoms needed are:  
  #  -O3' P    O5'  C5'   C4'  C3'  O3'  +P  +O5' O4' C1' (N1 C2 or N9 C4)


  # Find the atoms in this residue
  set selExist [atomselect $molid "residue $residue and name P O5' C5' C4' C3' O3' O4' C1' N1 C2"]
  if { [$selExist num] != 10} {
    puts "ssrestraints) Warning: Nucleic acid residue $residue will be ignored due to missing atoms."   
    $selExist delete
   return
  }

  set restype [lsort -unique [$selExist get resname]]
  switch -- $restype {
    T - THY - U - URA - RU - C - CYT - RC {
      set atomlist {P O5' C5' C4' C3' O3' O4' C1' N1 C2} 
  }     
    A - ADE - RA - G - GUA - RG {
    set atomlist {P O5' C5' C4' C3' O3' O4' C1' N9 C4} 
  } 
    default {
      puts "ssrestraints) Warning: Nucleic acid residue $residue will \ 
be ignored due to base $restype not recognized."
      return
    }
  }
      
  foreach name $atomlist {
    set seldih($name) [atomselect $molid "residue $residue and name $name"]
    set indexna($name) [$seldih($name) get index]
  }

  #
  # Restrain the alpha dihedral angle
  #

  # Find the index of the O3' atom from the previous residue by searching
  # the atoms bonded to P for an atom named O3'
  set indexO3prev -1
  foreach index [lindex [$seldih(P) getbonds] 0] {
    set selIndex [atomselect $molid "index $index"]
    if { [$selIndex get name] == "O3'" } {
      set indexO3prev $index
      $selIndex delete
      break
    }
    $selIndex delete
  }

  if { $indexO3prev == -1 } {
    puts "ssrestraints) Warning: Alpha dihedral angle of nucleic acid residue $residue will not be restrained due to missing previous O3' atom."
  } else {
    set alpha [measure dihed [list \
                                  [list $indexO3prev  $molid] \
                                  [list $indexna(P)   $molid] \
                                  [list $indexna(O5') $molid] \
                                  [list $indexna(C5') $molid]]]
    puts $out "dihedral $indexO3prev $indexna(P) $indexna(O5') $indexna(C5') $k_na_dih $alpha"
    if $labels {
      label add Dihedrals $molid/$indexO3prev\
        $molid/$indexna(P)\
        $molid/$indexna(O5')\
        $molid/$indexna(C5')
    }
  }

  #
  # Restrain the  epsilon and zeta dihedral angles
  #

  # Find the index of the P and O5' atoms from the next residue by searching 
  # the atoms bonded to O3' and +P for atoms named P and O5', respectively
  set indexPnext -1
  set indexO5next -1

  foreach index [lindex [$seldih(O3') getbonds] 0] {
    set selIndex [atomselect $molid "index $index"]
    if { [$selIndex get name] == "P" } {
      set indexPnext $index
      $selIndex delete
      break
    }
    $selIndex delete
  }
  if { $indexPnext == -1 } {
    puts "ssrestraints: Warning: Epsilon and Zeta dihedral angles of nucleic acid residue $residue will not be restrained due to missing next P atom."
  } else {
    set epsilon [measure dihed [list \
                                    [list $indexna(C4') $molid] \
                                    [list $indexna(C3') $molid] \
                                    [list $indexna(O3') $molid] \
                                    [list $indexPnext $molid]]]
    puts $out "dihedral $indexna(C4') $indexna(C3') $indexna(O3') $indexPnext $k_na_dih $epsilon"
    if $labels {
      label add Dihedrals $molid/$indexna(C4')\
        $molid/$indexna(C3')\
        $molid/$indexna(O3')\
        $molid/$indexPnext
    }
  
    set selPnext [atomselect $molid "index $indexPnext"]  
    foreach index [lindex [$selPnext getbonds] 0] {
      set selIndex [atomselect $molid "index $index"]
      if { [$selIndex get name] == "O5'" } {
        set indexO5next $index
        $selIndex delete
        break
      }
      $selIndex delete
    } 
    $selPnext delete
    if { $indexO5next == -1 } {
      puts "ssrestraints: Warning: Zeta dihedral angle of nucleic acid residue $residue will not be restrained due to missing next O5' atom."
    } else {
      set zeta [measure dihed [list \
                                   [list $indexna(C3') $molid] \
                                   [list $indexna(O3') $molid] \
                                   [list $indexPnext   $molid] \
                                   [list $indexO5next  $molid]]]
      puts $out "dihedral $indexna(C3') $indexna(O3') $indexPnext $indexO5next $k_na_dih $zeta"
      if $labels {
        label add Dihedrals $molid/$indexna(C3')\
          $molid/$indexna(O3')\
          $molid/$indexPnext\
          $molid/$indexO5next
      }
    }
  }


  #
  # Restrain the beta dihedral angle
  #
  
  set beta [measure dihed [list \
                               [list $indexna(P) $molid] \
                               [list $indexna(O5') $molid] \
                               [list $indexna(C5') $molid] \
                               [list $indexna(C4') $molid]]]
  puts $out "dihedral $indexna(P) $indexna(O5') $indexna(C5') $indexna(C4') $k_na_dih $beta"
  if $labels {
    label add Dihedrals $molid/$indexna(P)\
      $molid/$indexna(O5')\
      $molid/$indexna(C5')\
      $molid/$indexna(C4')
  }

  #
  # Restrain the gamma dihedral angle
  #

  set gamma [measure dihed [list \
                                [list $indexna(O5') $molid] \
                                [list $indexna(C5') $molid] \
                                [list $indexna(C4') $molid] \
                                [list $indexna(C3') $molid]]]
  puts $out "dihedral $indexna(O5') $indexna(C5') $indexna(C4') $indexna(C3') $k_na_dih $gamma"
  if $labels {
    label add Dihedrals $molid/$indexna(O5')\
      $molid/$indexna(C5') \
      $molid/$indexna(C4') \
      $molid/$indexna(C3')
  }

  #
  # Restrain the delta dihedral angle
  #

  set delta [measure dihed [list \
                                [list $indexna(C5') $molid] \
                                [list $indexna(C4') $molid] \
                                [list $indexna(C3') $molid] \
                                [list $indexna(O3') $molid]]]
  puts $out "dihedral $indexna(C5') $indexna(C4') $indexna(C3') $indexna(O3') $k_na_dih $delta"
  if $labels {
    label add Dihedrals $molid/$indexna(C5')\
      $molid/$indexna(C4')\
      $molid/$indexna(C3')\
      $molid/$indexna(O3')
  }

  #
  # Restrain the chi dihedral angle
  #

  if {$restype == "CYT" || $restype == "URA" || $restype == "THY" ||
      $restype == "C"   || $restype == "U"   || $restype == "T" ||
      $restype == "RC"   || $restype == "RU" } {
    set chi [measure dihed [list \
                                [list $indexna(O4') $molid] \
                                [list $indexna(C1') $molid] \
                                [list $indexna(N1)  $molid] \
                                [list $indexna(C2)  $molid]]]
    puts $out "dihedral $indexna(O4') $indexna(C1') $indexna(N1) $indexna(C2) $k_na_dih $chi"
    if $labels {
      label add Dihedrals $molid/$indexna(O4')\
        $molid/$indexna(C1')\
        $molid/$indexna(N1) \
        $molid/$indexna(C2)
    }
  } else {
    set chi [measure dihed [list \
                            [list $indexna(O4') $molid] \
                            [list $indexna(C1') $molid] \
                            [list $indexna(N9)  $molid] \
                            [list $indexna(C4) $molid]]]
    puts $out "dihedral $indexna(O4') $indexna(C1') $indexna(N9) $indexna(C4) $k_na_dih $chi"
    if $labels {
      label add Dihedrals $molid/$indexna(O4')\
        $molid/$indexna(C1')\
        $molid/$indexna(N9) \
        $molid/$indexna(C4)
    }
  }
    
  foreach name $atomlist {
    $seldih($name) delete
  }
  unset indexna

  return

}

# Restrain H-bonds (based on a script by Peter Freddolino)
proc ::SSRestraints::restrain_hbonds {donorsel accsel bondk anglek dcut acut} {

  variable molid
  variable out
  variable labels

  set selDonor [atomselect $molid "$donorsel"]
  set selAcceptor [atomselect $molid "$accsel"]
  
  set hbdat [measure hbonds $dcut $acut $selDonor $selAcceptor]

  $selDonor delete
  $selAcceptor delete

  set donoratoms [lindex $hbdat 0]
  set accatoms [lindex $hbdat 1]
  set hatoms [lindex $hbdat 2]

  foreach donor $donoratoms acceptor $accatoms hyd $hatoms {
    set basedist [measure bond [list $hyd $acceptor]]
    set baseangle [measure angle [list $donor $hyd $acceptor]]

    puts $out "bond $hyd $acceptor $bondk $basedist"
    puts $out "angle $donor $hyd $acceptor $anglek $baseangle"

    if $labels {
      label add Bonds $molid/$hyd $molid/$acceptor
      label add Angles $molid/$donor $molid/$hyd $molid/$acceptor
    }

  }

  return

}
