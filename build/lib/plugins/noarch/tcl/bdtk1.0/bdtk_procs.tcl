proc ::BDTK::initializeDimensions {args} {
	set nargs [llength $args]
	if {$nargs != 1 && $nargs != 2} { set s 0; set e end }
	if {$nargs == 1} { set s 0; set e $args }
	if {$nargs == 2} { set s [lindex $args 0]; set e [lindex $args 1] }

	set elements {ax ay az bx by bz cx cy cz ox oy oz}
	foreach el [lrange $elements $s $e] { variable $el 0}
}


proc ::BDTK::readxsc {xsfile} {
	set content ""
	if {![catch {open $xsfile r} ch]} {
		while { [gets $ch line] != -1 } {
			if { [regexp "^#" $line] } { continue } ; # ignore comments
			set content [split $line]
		}
		close $ch
	}

	set n [llength $content]
	if {$n > 11} { return [lrange $content 1 12] } 
	if {$n > 8}  { return [list [lrange $content 1 9] 0 0 0] } 
	return [lrepeat 12 0]
}

proc ::BDTK::writexsc {xscfile} {
	foreach el {ax ay az bx by bz cx cy cz ox oy oz} { variable $el }
	if {![catch {open $xscfile w} ch]} {
		puts  $ch "# NAMD extended system configuration restart file"
		puts  $ch "#\$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z"
		puts  $ch "0 $ax $ay $az $bx $by $bz $cx $cy $cz $ox $oy $oz"
		close $ch
	}
	if {![file readable $xscfile]} {
		tk_messageBox -icon error -message "Error!" -detail "Failed to write XSC file".
		return 1
	}
    return 0
}

# Get min/max coordinates of the system
# Find the bounding box
proc ::BDTK::boundingbox {} {

	foreach el {ax ay az bx by bz cx cy cz ox oy oz} { variable $el }
	# set [set el] [set ::BDTK::$el]; # OMG, this also works!
	
	set a [list $ax $ay $az];
	set b [list $bx $by $bz];
	set c [list $cx $cy $cz];

	set d [vecadd $a $b];       foreach {dx dy dz} $d { break }
	set e [vecadd $a $c];       foreach {ex ey ez} $e { break }
	set f [vecadd $b $c];       foreach {fx fy fz} $f { break }
	set g [vecadd $a $b $c];    foreach {gx gy gz} $g { break }

	set xlist [lsort -real [list 0.0 $ax $bx $cx $dx $ex $fx $gx]]
	set ylist [lsort -real [list 0.0 $ay $by $cy $dy $ey $fy $gy]]
	set zlist [lsort -real [list 0.0 $az $bz $cz $dz $ez $fz $gz]]

	set xmin [lindex $xlist 0]; set xmax [lindex $xlist end]
	set ymin [lindex $ylist 0]; set ymax [lindex $ylist end]
	set zmin [lindex $zlist 0]; set zmax [lindex $zlist end]

	set sizex [expr {$xmax - $xmin}]
	set sizey [expr {$ymax - $ymin}]
	set sizez [expr {$zmax - $zmin}]

	set minx [expr {$ox - 0.5*$sizex}]; set maxx [expr {$ox + 0.5*$sizex}];
	set miny [expr {$oy - 0.5*$sizey}]; set maxy [expr {$oy + 0.5*$sizey}];
	set minz [expr {$oz - 0.5*$sizez}]; set maxz [expr {$oz + 0.5*$sizez}];		

	return [list [list $minx $miny $minz] [list $maxx $maxy $maxz]]
	# returning list of lists just for simplicity of assigning the result
	# in the calling proc
}

# Currently unused
proc ::BDTK::gettempdir {} {
	if [info exists ::env(TMPDIR)] {
		return $::env(TMPDIR)
	} else {
		switch [vmdinfo arch] { 
			WIN64 -
			WIN32 {
				return "c:/"
			}    
			MACOSXX86_64 -
			MACOSXX86 -
			MACOSX {
				return "/"
			}    
			default {
				return "/tmp"
			}    
		}    
	}
}

proc ::BDTK::readmaterials {} {

	set ::BDTK::allmaterials {}; # initialize list of materials
	set file [file join $::env(BDTKDIR) database membrane.dat]

	set result {}
	if {![catch {open $file "r"} ch]} {

		while {[gets $ch line] >= 0} {
			if {[regexp "^#" $line]} { continue } ;# skip comments
			if {[string length [string trim $line]] < 1} { continue } ; # skip empty lines
			lappend result $line
		}

		if {[llength $result]>0} { set ::BDTK::allmaterials $result; }
		close $ch
	}
}

proc ::BDTK::readbdparticles {} {
	set ::BDTK::listofbdparticles ""; # initialize list of bd particles

	set file [file join $::env(BDTKDIR) database ions.dat]
	set results ""
	if {![catch {open $file "r"} ch]} {
		while {[gets $ch line] >= 0} {
			if {[regexp "^#" $line]} { continue } ;# skip comments
			if {[string length [string trim $line]] < 1} { continue } ; # skip empty lines
			lappend result $line; # Name, code, charge, diffusion, radius, epsilon
		}
		set len [llength $result]
		if {$len>0} {
			set ::BDTK::listofbdparticles $result;
			set ::BDTK::numberofbdparticles [lrepeat $len 0]
		}
		close $ch
	}
}


proc ::BDTK::readshapes {} {

	set ::BDTK::allshapes {}   ; # initialize list of shapes
	set ::BDTK::shapeImages {} ;

	set directory [file join $::env(BDTKDIR) database shapes]
	set files [glob -type f -directory $directory -nocomplain *.shp]

	foreach file $files {
		set result {}
		if {![catch {open $file "r"} ch]} {
			while {[gets $ch line] >= 0} {
				if {[regexp "^#" $line]} { continue } ;# skip comments
				if {[string length [string trim $line]] < 1} { continue } ; # skip empty lines
				lappend result $line
			}
			if {[llength $result]>0} {
				lappend ::BDTK::allshapes $result
				lappend ::BDTK::shapeImages "[file rootname $file].gif"
			}
			close $ch
		}
	}
}

# Find interaction potentials of two species 
# defined by their codes (code1 & code2)
# in the database of potentials
proc ::BDTK::findPotential {args} {
	if {[llength $args] != 2} { return "" }
	set code1 [lindex $args 0]
	set code2 [lindex $args 1]

    set file [file join $::env(BDTKDIR) database interactions.dat]
    if {![catch {open $file r} in]} {
        set output ""
        while {[gets $in line] >= 0} {
            if {[regexp "^#" $line]} { continue } ; # skip comment lines
            if {[string length [string trim $line]] < 1} { continue } ; # skip empty lines
            set tok [concat $line]

            set in1 [lsearch [lrange $tok 0 1] $code1]; # find code1 in the list
            if { $in1 == -1 } { continue; };            # continue if haven't found it
            set in2 [expr {1 - $in1}];                  # if code1 was found, check the other potential
            if { [lindex $tok $in2] != $code2 } { continue; }; # if it's not what we need, continue

            set output [file normalize [file join $::env(BDTKDIR) database [lindex $tok 2]]]
            # puts "Interaction potential for $code1 - $code2 :  $output"
        }
        close $in
        return $output
    }
    return ""
}

proc ::BDTK::putd {args} {
	puts "BDTK: $args"
}
