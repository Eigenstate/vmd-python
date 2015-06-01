mol new cg_POPE.psf
mol addfile POPE-noSolv.dcd waitfor all

set nf [molinfo top get numframes] 

for {set i 0 } {$i < $nf} {incr i} { 

animate goto ${i}
set all [atomselect top all]
${all} moveby [vecinvert [measure center ${all}]]

$all writepdb ./dcdNew/frame${i}.pdb

} 
