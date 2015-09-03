#!/usr/local/bin/vmd -dispdev text
# replacing Na+ with K+ (or anything else with anything else)
# Ilya Balabin (ilya@ks.uiuc.edu), 2002-2003

# define input files here
set psffile "ionized.psf"
set pdbfile "ionized.pdb"
set prefix  "sod2pot"

# define what ions to replace with what ions
set ionfrom "SOD"
set ionto "POT"

# do not change anything below this line
package require psfgen
topology top_all27_prot_lipid_pot.inp

puts "\nSod2pot) Reading ${psffile}/${pdbfile}..."
resetpsf
readpsf $psffile
coordpdb $pdbfile
mol load psf $psffile pdb $pdbfile

set sel [atomselect top "name $ionfrom"]
set poslist [$sel get {x y z}]
set seglist [$sel get segid]
set reslist [$sel get resid]
set num [llength $reslist]
puts "Sod2pot) Found ${num} ${ionfrom} ions to replace..."

set num 0
foreach segid $seglist resid $reslist {
    delatom $segid $resid
    incr num
}
puts "Sod2pot) Deleted ${num} ${ionfrom} ions"

segment $ionto {
    first NONE
    last NONE
    foreach res $reslist {
	residue $res $ionto
    }
}
set num [llength $reslist]
puts "Sod2pot) Created ${num} topology entries for ${ionto} ions"

set num 0
foreach xyz $poslist res $reslist {
    coord $ionto $res $ionto $xyz
    incr num
}
puts "Sod2pot) Set coordinates for ${num} ${ionto} ions"

writepsf "${prefix}.psf"
writepdb "${prefix}.pdb"
puts "Sod2pot) Wrote ${prefix}.psf/${prefix}.pdb"
puts "Sod2pot) All done."

quit
