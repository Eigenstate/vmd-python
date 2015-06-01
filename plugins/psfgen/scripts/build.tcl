
# psfgen script
# Justin Gullingsrud
# justin@ks.uiuc.edu
#
# This script uses psfgen to create a psf file from the 
# given molecule.  It assumes that a topology file 
# has already been loaded, but then tries hard to make
# everything else automatic.

# We still need a command to erase the current structure, so
# this will probably work only once for now.

# Usage: start psfgen, source this script, then run
#	psfgen <pdbfile> [<outputname>]
# <outputname> is optional; if provided, the name of the generated pdb
# and psf files will be outputname.pdb and outputname.psf

# Set the variable TOPOFILE to the location of your Charmm topology file. 
 
set TOPOFILE /Projects/justin/toppar/charmm27/top_all27_prot_na.inp

proc psfalias { } {

	# Define common aliases
	# Here's for nucleics
	alias residue G GUA
	alias residue C CYT
	alias residue A ADE
	alias residue T THY
	alias residue U URA

	foreach bp { GUA CYT ADE THY URA } {
		alias atom $bp "O5\*" O5'
		alias atom $bp "C5\*" C5'
		alias atom $bp "O4\*" O4'
		alias atom $bp "C4\*" C4'
		alias atom $bp "C3\*" C3'
		alias atom $bp "O3\*" O3'
		alias atom $bp "C2\*" C2'
		alias atom $bp "O2\*" O2'
		alias atom $bp "C1\*" C1'
	}

	alias atom ILE CD1 CD
	alias atom SER HG HG1
	alias residue HIS HSD

	# Heme aliases
	alias residue HEM HEME
	alias atom HEME "N A" NA
	alias atom HEME "N B" NB
	alias atom HEME "N C" NC
	alias atom HEME "N D" ND

	# Water aliases
	alias residue HOH TIP3
	alias atom TIP3 O OH2

	# Ion aliases
	alias residue K POT
	alias atom K K POT 
}

proc del { foo } {
	exec rm $foo
}


# Split a PDB file into separate files; split at the point that residues fail
# to increase 1 by 1.  Return a list of the files created 

proc splitpdb { fname } {
	set in [open $fname r]
 	set nseg 0

	set oldres -1
	set curres -1

	foreach line [split [read -nonewline $in] \n] {
		set head [string range $line 0 5]
		if { ![string compare $head "ATOM  "] || 
		     ![string compare $head "HETATM"] } {
			set curres [string range $line 22 25]
			set resdif [expr $curres - $oldres]
			if { $oldres != -1 && $resdif != 0 && $resdif != 1 } {
				# Close the old file
				puts $out END
				close $out
				set oldres -1
			}
			if { $oldres == -1 } {
				# Start a new file
				incr nseg
				set newname "${fname}_${nseg}.pdb"
				set out [open $newname w]
				lappend fnamelist $newname
			}
			puts $out $line
			set oldres $curres
		
		}
	}
	close $out
	close $in
	return $fnamelist
}

proc build { pdb {outname psfgen}} {
	global TOPOFILE 
	topology $TOPOFILE 
	psfalias
	set nseg 0
	foreach segfile [splitpdb $pdb] {
		incr nseg
		set segid "P${nseg}"
		segment $segid {
			pdb $segfile
		}
		coordpdb $segfile $segid
		del $segfile
	}
	guesscoord
	writepdb ${outname}.pdb 
	writepsf ${outname}.psf
}

# This allows the script to be run as a command line argument to tclsh,
# e.g.  psfgen build.tcl input.pdb outputname
# If the script is sourced, it does nothing (ok, it prints a 1).
catch {eval build $argv} 
