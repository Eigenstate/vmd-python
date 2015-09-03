#!/usr/local/bin/vmd -dispdev text

# embed (parts of) protein into a membrane
# Ilya Balabin (ilya@ks.uiuc.edu), 2002-2003
#
# You need: a) membrane structure (membrane.psf/pdb);
# b) properly oriented and aligned to the membrane
# protein structure (protein.psf/pdb)


# set echo on for debugging
echo on

# need psfgen module and topology
package require psfgen
topology top_all27_prot_lipid.inp

# load structures
resetpsf
readpsf membrane.psf
coordpdb membrane.pdb
readpsf protein.psf
coordpdb protein_aligned.pdb

# can delete some protein segments; list them in brackets on next line
set pseg2del   { }
foreach seg $pseg2del {
  delatom $seg
}

# write temporary structure
set temp "temp"
writepsf $temp.psf
writepdb $temp.pdb

# reload full structure (do NOT resetpsf!)
mol load psf $temp.psf pdb $temp.pdb

# select and delete lipids that overlap protein:
# any atom to any atom distance under 0.8A
# (alternative: heavy atom to heavy atom distance under 1.3A)
set sellip [atomselect top "resname POPE"]
set lseglist [lsort -unique [$sellip get segid]]
foreach lseg $lseglist {
  # find lipid backbone atoms
  set selover [atomselect top "segid $lseg and within 0.8 of protein"]
  # delete these residues
  set resover [lsort -unique [$selover get resid]]
  foreach res $resover {
    delatom $lseg $res
  }
}

# delete lipids that stick into gaps in protein
foreach res { } {delatom $LIP1 $res}
foreach res { } {delatom $LIP2 $res}

# delete lipids that fall out of the PBC box
# the following numbers are for example only; yours are different!
set xmin -55
set xmax  41
set ymin -51
set ymax  34
foreach lseg {"LIP1" "LIP2"} {
  # find lipid backbone atoms
  set selover [atomselect top "segid $lseg and (x<$xmin or x>$xmax or y<$ymin or y>$ymax)"]
  # delete these residues
  set resover [lsort -unique [$selover get resid]]
  foreach res $resover {
    delatom $lseg $res
  }
}

# write full structure
writepsf protein_mem.psf
writepdb protein_mem.pdb

# clean up
file delete $temp.psf
file delete $temp.pdb

# non-interactive script
quit
