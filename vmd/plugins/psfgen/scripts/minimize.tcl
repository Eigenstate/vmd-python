
proc startnamd { input output } {
	exec namd2 $input > $output &
}

# This script changes nonzero to 0 and 0 to 1 in the occupancy column.
# Useful for making a fixed atoms file since psfgen sets the occupancy of
# built atoms to 1 and old atoms to 0.
proc flip_occupancy { inpdb outpdb } {
	set in [open $inpdb r]
	set out [open $outpdb w]
	foreach line [split [read -nonewline $in] \n] {
		set head [string range $line 0 5]
		if { ![string compare $head "ATOM  "] ||
		     ![string compare $head "HETATM"] } {
			     set oldocc [string range $line 54 59]
			     set newocc [format %6.2f [expr int($oldocc) ^ 1]]
			     set line [string replace $line 54 59 $newocc]
		}
		puts $out $line
	}
	close $in
	close $out
	return
}

set SCRIPTDIR /home/justin/projects/psfgen/scripts
proc addparams { file } {
	global SCRIPTDIR
  	puts $file "paraTypeCharmm	on"
	puts $file "parameters 		${SCRIPTDIR}/par_all27_prot_na.inp"
	return
}

proc addfixed { file fixpdb } {
	puts $file "fixedAtoms		on"
	puts $file "fixedAtomsFile	$fixpdb"
	puts $file "fixedAtomsCol	O"
	return
}

proc minimize { psf pdb outputname nsteps } {
	global SCRIPTDIR	
	set tfile [open ${SCRIPTDIR}/template.conf r]
	set tdata [read $tfile]
	close $tfile

	flip_occupancy $pdb fix.pdb

	set minfile [open minimize.conf w]
	puts $minfile "structure	$psf"
	puts $minfile "coordinates	$pdb"
	puts $minfile "temperature	0"
	addparams $minfile
	addfixed $minfile fix.pdb
	puts $minfile "minimization 	on"
	puts $minfile "outputname	$outputname"
	puts $minfile "numsteps		$nsteps"	
	puts $minfile $tdata
	close $minfile
	
        startnamd minimize.conf minimize.namd
        puts "Started NAMD minimization"
}

