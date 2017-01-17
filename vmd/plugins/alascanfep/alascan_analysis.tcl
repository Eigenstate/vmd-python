

#######################
#Analysis of FEP output
#######################

proc ::alascan::analyzeFEP {} {
	
	variable w
	variable w1
	variable w2
	variable w2exists
	variable w1exists
	variable tempParse

	#unpack the setup frames
	pack forget $w.input $w.intro1 $w.selected $w.selected.fr $w.lstparm $w.lstparm.fr $w.tlist $w.feppath $w.parameters $w.fep 
	if {$w2exists ==1} {destroy $w2}
	if {$w1exists ==1} {destroy $w1} 
	destroy $w.intro2 $w.input_ana $w.status $w.l $w.frame $w.tit	

	frame $w.intro2  
	label $w.intro2.label -text "Analyze FEP output" -relief ridge -padx 50m
	pack $w.intro2 -fill x -padx 2 -pady 1 -expand 1
	pack $w.intro2.label -fill x

	#take the psf and pdb file from vmd if there is one
	if {[molinfo num] != 0} {
    	foreach fname [lindex [molinfo top get filename] 0] ftype [lindex [molinfo top get filetype] 0] {
      	if { [string equal $ftype "pdb"] } {
        	set ::alascan::pdbfile_ana $fname
      		} elseif { [string equal $ftype "psf"] } {
			set ::alascan::psffile_ana $fname
			}
    		}
 	}

	#input PSF and PDB files
	labelframe $w.input_ana -bd 2 -relief ridge -text "Input" 
	grid [label $w.input_ana.psffile -text "PSF file (native):"] -row 1 -column 0 -sticky w
	grid [entry $w.input_ana.psfpath -width 35 -textvariable ::alascan::psffile_nat] -row 1 -column 1 -sticky ew
	grid [button $w.input_ana.psfbutton -text "Browse" -command {
		set tempfile [tk_getOpenFile]
		if { ![string equal $tempfile ""] } {set ::alascan::psffile_nat $tempfile} }] -row 1 -column 2 -sticky w

	grid [label $w.input_ana.pdbfile -text "PDB file (native):"] -row 2 -column 0 -sticky w
	grid [entry $w.input_ana.pdbpath -width 35 -textvariable ::alascan::pdbfile_nat] -row 2 -column 1 -sticky ew
	grid [button $w.input_ana.pdbbutton -text "Browse" -command {
		set tempfile [tk_getOpenFile]
		if {![string equal $tempfile ""]} { set ::alascan::pdbfile_nat $tempfile} }] -row 2 -column 2 -sticky w

	grid [label $w.input_ana.fepdir -text "Path to read FEP outfiles:"] -row 3 -column 0 -sticky w
	grid [entry $w.input_ana.xscpath -width 35 -textvariable ::alascan::fepdir] -row 3 -column 1 -sticky ew
	grid [button $w.input_ana.xscbutton -text "Browse" -command {
		set tempfile [tk_chooseDirectory]
		if {![string equal $tempfile ""]} { set ::alascan::fepdir $tempfile} }] -row 3 -column 2 -sticky w

	grid [label $w.input_ana.temp -text "Temperature:"] -row 4 -column 0 -sticky w
	grid [entry $w.input_ana.tempVal -width 35 -textvariable ::alascan::tempParse] -row 4 -column 1 -sticky ew
	
	grid [button $w.input_ana.next -text "Analyze fepout files" -command ::alascan::parseStatus] -row 5 -column 0 -sticky w 
	grid columnconfigure $w.input_ana 1 -weight 1
	pack $w.input_ana -side top -padx 4 -pady 2 -expand 1 -fill x


} ;#closing of ::alascan::analyzeFEP


proc ::alascan::parseStatus {} {
	variable w
	variable running 0
	variable done 0
	variable fepdir
	variable psffile_nat
	variable pdbfile_nat
	variable tempParse

	destroy $w.status $w.l $w.frame $w.tit
	if { $psffile_nat == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "No PSF file loaded"
		return 0
		}

 	if { $pdbfile_nat == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "No PDB file loaded!"
		return 0
		}

 	if { $fepdir == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "Choose a path to read FEP outfiles!"
		return 0
		}

	if { $tempParse == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "Missing parameter: Temperature!!!"
		return 0
		}

	#host guest status
	if {$::alascan::hg_sel == 1} {set hg 1}
	if {$::alascan::hg_sel == 0} {set hg 0}	

	#nr of mutants
	set listDir [glob -type d -dir $fepdir *]
	set nrMutants 0
	if {$hg==0} {
		foreach var $listDir {
			set ckPsf [glob -nocomplain -dir $var *.psf]
			set ckFep [glob -nocomplain -dir $var *.fep]
			set ckFepout [glob -nocomplain -dir $var forward.fepout]
			set ckFepout1 [glob -nocomplain -dir $var backward.fepout]
			if {$ckPsf != "" && $ckFep != "" && $ckFepout != "" && $ckFepout1 != ""} {
				incr nrMutants
				}
			}
		}
	if {$hg==1} {
		foreach var $listDir {
			set ck_host_dir [file isdirectory $var/Host]
			set ck_hostGuest_dir [file isdirectory $var/Host-Guest ]
			set ck_host_psf [glob -nocomplain -dir $var/Host *.psf]
			set ck_host_fep [glob -nocomplain -dir $var/Host *.fep]
			set ck_host_fepout [glob -nocomplain -dir $var/Host forward.fepout]
			set ck_host_fepout1 [glob -nocomplain -dir $var/Host backward.fepout]
			set ck_hostGuest_psf [glob -nocomplain -dir $var/Host-Guest *.psf]
			set ck_hostGuest_fep [glob -nocomplain -dir $var/Host-Guest *.fep]
			set ck_hostGuest_fepout [glob -nocomplain -dir $var/Host-Guest forward.fepout]
			set ck_hostGuest_fepout1 [glob -nocomplain -dir $var/Host-Guest backward.fepout]
			if {$ck_host_dir == 1 && $ck_hostGuest_dir ==1 && $ck_host_psf != "" && $ck_host_fep != "" && $ck_host_fepout != "" && $ck_host_fepout1 != "" \
				&& $ck_hostGuest_psf != "" && $ck_hostGuest_fep != "" && $ck_hostGuest_fepout != ""  && $ck_hostGuest_fepout1 != ""} {
				incr nrMutants	
			    }
			}		
		}
	puts "\n\nTotal number of mutants: $nrMutants\n\n"
	if {$hg==0} {labelframe $w.status -bd 2 -relief ridge -text "FEP Analysis - Status"}
	if {$hg==1} {labelframe $w.status -bd 2 -relief ridge -text "FEP Analysis * Host-Guest System * Status"}
	grid [label $w.status.l1 -text "Total number of mutations: $nrMutants" -font {-family times -size 16} -fg blue] \
		-row 0 -column 0 -sticky {} -padx 10 -pady 8
	grid [label $w.status.l2 -text "$running In progress..." -font {-family times -size 16} -fg red] \
		-row 0 -column 1 -sticky {} -pady 8
	grid [label $w.status.l3 -text "$done Done" -font {-family times -size 16} -fg green] \
		-row 0 -column 2 -sticky {} -padx 20 -pady 8
	grid columnconfigure $w.status 1 -weight 1
	pack $w.status -side top -padx 4 -pady 5 -expand 1 -fill x
	update
	if {$nrMutants >0} {
		::alascan::parseFEP
		} else {
			tk_messageBox -icon error -type ok -title Message -parent $w \
			-message "FEP outfiles are not found or not in the required format, Please refer the manual!"
			return 0
			}
} ; #closing of ::alascan::parseStatus

proc ::alascan::parseFEP {} {

	variable w
	variable psffile_nat
	variable pdbfile_nat
	variable fepdir
	variable resname_nat {}
	variable resid_nat {}
	variable segname_nat {}
	variable parseResname {}
	variable parseResid {}
	variable parseSegname {}
	variable sstruct_nat {}
	variable totalFenergy {}
	variable totalError {}
	variable done
	variable hyst {}
	variable dispHyst {}
	variable dispHyst_hg {}
	variable dispHyst_host {}
	variable hyst_hg {}
	variable hyst_host {}
	variable totalFenergy_hg {}
	variable totalError_hg {}
	variable totalFenergy_host {}
	variable totalError_host {}
	variable parseResname_hg {}
	variable parseResid_hg {}
	variable parseSegname_hg {}
	variable ddg_s1 {}  
	destroy $w.l $w.frame $w.tit

	#host guest status
	if {$::alascan::hg_sel == 1} {set hg 1}
	if {$::alascan::hg_sel == 0} {set hg 0}	

	#read native pdb and collect residue informations
	mol delete top
	mol load psf $psffile_nat pdb $pdbfile_nat	
	set caAtom_nat [atomselect top "name CA"]
	set resname_nat [$caAtom_nat get {resname}]
	set resid_nat [$caAtom_nat get {resid}]
	set segname_nat [$caAtom_nat get {segname}]
	set sstruct_nat [$caAtom_nat get {structure}]

	#list of directories from the "FEP outfiles" path
	set listDir [glob -type d -dir $fepdir *]

	foreach var $listDir {
		set fileOk 1; 

		if {$hg==0} {
			set tempPsf [glob -nocomplain -dir $var *.psf]
			set tempFep [glob -nocomplain -dir $var *.fep]
			set tempFepout [glob -nocomplain -dir $var *.fepout]
			file delete $var/ParseFEP.log
			if {$tempPsf == "" || $tempFep == "" || $tempFepout == ""} { set fileOk 0 }
			}

		if {$hg==1} {
			set tempPsf_host [glob -nocomplain -dir $var/Host *.psf]
			set tempPsf_hg [glob -nocomplain -dir $var/Host-Guest *.psf]
			set tempFep_host [glob -nocomplain -dir $var/Host *.fep]
			set tempFep_hg [glob -nocomplain -dir $var/Host-Guest *.fep]
			set tempFepout_host [glob -nocomplain -dir $var/Host *.fepout]
			set tempFepout_hg [glob -nocomplain -dir $var/Host-Guest *.fepout]
			file delete $var/ParseFEP.log
			if {$tempPsf_host == "" || $tempPsf_hg == "" || $tempFep_host == "" || $tempFep_hg == "" || $tempFepout_host == "" \
			    || $tempFepout_hg == ""} { set fileOk 0 }
			}		

		if {$hg==0} {
		if {$fileOk == 1} {
			
			#fepoutfiles	
			if {[file exists $var/forward.fepout]} {
				set tempfwd [file join $var forward.fepout]
				}
			if {[file exists $var/backward.fepout]} {
				set tempbwd [file join $var backward.fepout]
				} else { set unidir 1} 

				mol delete top
				mol load psf  [lindex $tempPsf 0] pdb [lindex $tempFep 0]
				set caAtom [atomselect top "name CBB"]
				set temp_parseResname [$caAtom get {resname}]
				set temp_parseResid [$caAtom get {resid}]
				set temp_parseSegname [$caAtom get {segname}]
				lappend ::alascan::parseResname $temp_parseResname
				lappend ::alascan::parseResid $temp_parseResid
				lappend ::alascan::parseSegname $temp_parseSegname
				set stat "$temp_parseResid $temp_parseResname"
				$w.status.l2 configure -text "$stat In progress..."
				update	
				eval package require parsefep
				eval parsefep -forward $tempfwd -backward $tempbwd -bar	
				file copy -force ParseFEP.log $var
				file delete ParseFEP.log

				#read ParseFEP log files and append data
				set templog [file join $var ParseFEP.log]
				set tempfile [open $templog "r"]
				while {[gets $tempfile line] >= 0} {
					if {[regexp {^BAR-estimator: total free energy change} $line]} {
						set tempsplit [split $line " "]
						set ener [format "%.1f" [lindex $tempsplit 6]]
						set erro [format "%.1f" [lindex $tempsplit 11]]
						lappend ::alascan::totalFenergy $ener
						lappend ::alascan::totalError $erro
						}
					if {[regexp {^backward:} $line]} {
						set tempbwd [regexp -all -inline {\S+} $line]
						set tempHyst [lindex $tempbwd 3]
						}
					}
				lappend ::alascan::hyst $tempHyst
				close $tempfile
			
			incr done
			$w.status.l3 configure -text "$done Done"
			update
			} 
			}
	
		if {$hg==1} {
		   if {$fileOk == 1} {

			#host-guest
			set hg_Fok 1; 
			if {[file exists $var/Host-Guest/forward.fepout]} {
				set tempfwd_gh [file join $var/Host-Guest forward.fepout]
				} else {set hg_Fok 0}
			if {[file exists $var/Host-Guest/backward.fepout]} {
				set tempbwd_gh [file join $var/Host-Guest backward.fepout]
				} else {set hg_Fok 0}

			if {[file exists $var/Host/forward.fepout]} {
				set tempfwd_h [file join $var/Host forward.fepout]
				} else {set hg_Fok 0}
			if {[file exists $var/Host/backward.fepout]} {
				set tempbwd_h [file join $var/Host backward.fepout]
				} else {set hg_Fok 0}

			if {$hg_Fok==1} {
			#tempfwd_gh and tempbwd_gh
				mol delete top
				mol load psf  [lindex $tempPsf_hg 0] pdb [lindex $tempFep_hg 0]
				set caAtom [atomselect top "name CBB"]
				set temp_parseResname_hg [$caAtom get {resname}]
				set temp_parseResid_hg [$caAtom get {resid}]
				set temp_parseSegname_hg [$caAtom get {segname}]
				set stat "$temp_parseResid_hg $temp_parseResname_hg"
				$w.status.l2 configure -text "$stat In progress..."
				update	
				eval package require parsefep
				eval parsefep -forward $tempfwd_gh -backward $tempbwd_gh -bar	
				file copy -force ParseFEP.log $var/Host-Guest
				file delete ParseFEP.log

				#read ParseFEP log files and append data **host-guest**
				set templog [file join $var/Host-Guest ParseFEP.log]
				set tempfile [open $templog "r"]
				while {[gets $tempfile line] >= 0} {
					if {[regexp {^BAR-estimator: total free energy change} $line]} {
						set tempsplit [split $line " "]
						set ener [format "%.1f" [lindex $tempsplit 6]]
						set erro [format "%.1f" [lindex $tempsplit 11]]
						lappend ::alascan::totalFenergy_hg $ener
						lappend ::alascan::totalError_hg $erro
						}
					if {[regexp {^backward:} $line]} {
						set tempbwd [regexp -all -inline {\S+} $line]
						set tempHyst [lindex $tempbwd 3]
						}
					}
				lappend ::alascan::hyst_hg $tempHyst
				close $tempfile



			#tempfwd_h and tempbwd_h
				mol delete top
				mol load psf  [lindex $tempPsf_host 0] pdb [lindex $tempFep_host 0]
				set caAtom [atomselect top "name CBB"]
				set temp_parseResname_host [$caAtom get {resname}]
				set temp_parseResid_host [$caAtom get {resid}]
				set temp_parseSegname_host [$caAtom get {segname}]
				set stat "$temp_parseResid_host $temp_parseResname_host"
				eval package require parsefep
				eval parsefep -forward $tempfwd_h -backward $tempbwd_h -bar
				file copy -force ParseFEP.log $var/Host
				file delete ParseFEP.log

				#read ParseFEP log files and append data **host**
				set templog [file join $var/Host ParseFEP.log]
				set tempfile [open $templog "r"]
				while {[gets $tempfile line] >= 0} {
					if {[regexp {^BAR-estimator: total free energy change} $line]} {
						set tempsplit [split $line " "]
						set ener [format "%.1f" [lindex $tempsplit 6]]
						set erro [format "%.1f" [lindex $tempsplit 11]]
						lappend ::alascan::totalFenergy_host $ener
						lappend ::alascan::totalError_host $erro
						}
					if {[regexp {^backward:} $line]} {
						set tempbwd [regexp -all -inline {\S+} $line]
						set tempHyst [lindex $tempbwd 3]
						}
					}
				lappend ::alascan::hyst_host $tempHyst
				close $tempfile
				

				##append parsed data
				if {$temp_parseResname_hg == $temp_parseResname_host && \
				    $temp_parseResid_hg   == $temp_parseResid_host && \
				    $temp_parseSegname_hg == $temp_parseSegname_host} {
				lappend ::alascan::parseResname_hg $temp_parseResname_hg
				lappend ::alascan::parseResid_hg $temp_parseResid_hg
				lappend ::alascan::parseSegname_hg $temp_parseSegname_hg
				} else {lappend ::alascan::parseResname_hg "host guest system does not match - $var"}


			} 
			incr done
			$w.status.l3 configure -text "$done Done"
			update

		   };#fileOk==1
		};#closing if hg==1


		}; # "foreach var $listDir
 
		::alascan::summaryFEP	

} ;#::alascan::parseFEP


proc ::alascan::gradient {min max} {
	array set grad {}
	set range 0; set j 0
	for {set i [expr int($min)]} {$i<[expr int($max)]} {incr i} {incr range}
	
	foreach {r1 g1 b1} {0.0 0.0 255.0} break
	foreach {r2 g2 b2} {255.0 0.0 0.0} break

	if {$range <=1} {
		set rStep 0.0 ; set gStep 0.0 ; set bStep 255.0
      		} else {
        		set rStep [expr {($r2-$r1)/($range)}]
        		set gStep [expr {($g2-$g1)/($range)}]
        		set bStep [expr {($b2-$b1)/($range)}]
      			}
	for {set i [expr int($min)]} {$i <=[expr int($max)]} {incr i 1} {
        	set r [expr {int($rStep * $j + $r1)}]
      		set g [expr {int($gStep * $j + $g1)}]
		set b [expr {int($bStep * $j + $b1)}]
		incr j
		set grad($i) [format "#%.2X%.2X%.2X" $r $g $b]
      		}
	return [array get grad]
} 

proc ::alascan::gradient2 {min max} {
	array set grad {}

	foreach {r1 g1 b1} {0.0 0.0 255.0} break
	foreach {r2 g2 b2} {255.0 0.0 0.0} break

	set min1 $min
	set a 1.0; set count 1
	while {$a == 1} {
	set count [expr {$count+1}]
	set min1 [format %.1f [expr {$min1+0.1}]]
	if {$min1 > $max} {set a 0}
	}

	set nrange [expr {$count-2}]
	if {$nrange <1} {
		set rStep 0.0 ; set gStep 0.0 ; set bStep 255.0
		} else {
        		set rStep [expr {($r2-$r1)/($nrange)}]
        		set gStep [expr {($g2-$g1)/($nrange)}]
        		set bStep [expr {($b2-$b1)/($nrange)}]
			}

	set j $min; set k 0
	for {set i 1} {$i < $count} {incr i } {
        	set r [expr {int($rStep * $k + $r1)}]
      		set g [expr {int($gStep * $k + $g1)}]
		set b [expr {int($bStep * $k + $b1)}]
		set grad($j) [format "#%.2X%.2X%.2X" $r $g $b]
		incr k
		set j [format %.1f [expr {$j+0.1}]]
      		}

	return [array get grad]

}

proc ::alascan::gradLabel {} {
	set range 140
	foreach {r1 g1 b1} {0.0 0.0 255.0} break
	foreach {r2 g2 b2} {255.0 0.0 0.0} break
        set r_step [expr {($r2-$r1) / ($range)}]
        set g_step [expr {($g2-$g1) / ($range)}]
        set b_step [expr {($b2-$b1) / ($range)}]
      	set steps {}
      	for {set i 0} {$i <= $range} {incr i 1} {
        	set r [expr {int($r_step * $i + $r1)}]
        	set g [expr {int($g_step * $i + $g1)}]
		set b [expr {int($b_step * $i + $b1)}]
   	        lappend steps [format "#%.2X%.2X%.2X" $r $g $b]
      		}
      return $steps
}

proc ::alascan::sscol {l} {
	if {$l eq "C"} {set col "snow"
	} elseif {$l eq "E"} {set col "yellow"
	} elseif {$l eq "T"} {set col "cyan"
	} elseif {$l eq "B"} {set col "sienna"
	} elseif {$l eq "H"} {set col "violet"
	} elseif {$l eq "G"} {set col "blue"
	} elseif {$l eq "I"} {set col "red"
	}
}

proc ::alascan::summaryFEP {} {
	variable w
	variable resname_nat 
	variable resid_nat
	variable segname_nat
	variable parseResname
	variable parseResid
	variable parseResid
	variable parseSegname
	variable parseSegname
	variable parseResname_uni
	variable parseResid_uni
	variable parseSegname_uni
	variable sstruct_nat
	variable totalFenergy
	variable totalError
	variable tempParse
	variable hyst
	variable kt
	variable array grad
	set twokt [expr 2 * $kt]
	destroy $w.status $w.l $w.tit; update

	#host guest status
	if {$::alascan::hg_sel == 1} {set hg 1}
	if {$::alascan::hg_sel == 0} {set hg 0}

	#lookup table for the single protein system
	array set lkup_bi_delg [ list I2A -7.1 L2A 11.5 M2A 1.1 F2A -8.4 W2A -22.3 Y2A 8.8 V2A -1.3 S2A -3.9 T2A 17.1 \
			              N2A 77.5 Q2A 56.1 C2A -1.0 G2A 6.9 P2A -20.6 R2A 263.5 H2A 36.7 K2A 43.9 D2A 131.2 E2A 109.9 ]
	array set lkup_bi_hys  [ list I2A 0.2 L2A 0.1 M2A -0.2 F2A 0.1 W2A 0.1 Y2A 0.1 V2A 0.1 S2A 0.1 T2A 0.0 \
			              N2A -0.2 Q2A 0.1 C2A -0.2 G2A -0.0 P2A 0.0 R2A -0.1 H2A 0.1 K2A -0.2 D2A -0.2 E2A 0.0 ]	

	set totalres_nat [llength $resid_nat]
	set totalheight_nat [expr $totalres_nat * 22]

     if {$hg==0} {
	if {$totalFenergy != ""} {
		set path 0
		foreach id_parse $parseResid name_parse $parseResname segname_parse $parseSegname dg $totalFenergy {
			set temp_index [lsearch -exact $resid_nat $id_parse]
			if {[lindex $segname_nat $temp_index] == $segname_parse} {
				set resname_nat [lreplace $resname_nat $temp_index $temp_index $name_parse]
				set tmp_ddg_s1 [format %.1f [expr $dg-$lkup_bi_delg($name_parse)]]
				lappend ::alascan::ddg_s1 $tmp_ddg_s1
				}
			
			}
		set dGsorted [lsort -real -increasing $::alascan::ddg_s1]
		set dGmin [lindex $dGsorted 0]; set dGmax [lindex $dGsorted end]
		}

       } ; #hg==0

       if {$hg==1} {
		set path 3
		#host_guest
		foreach id_parse $::alascan::parseResid_hg name_parse $::alascan::parseResname_hg \
			segname_parse $::alascan::parseSegname_hg dg_hg $::alascan::totalFenergy_hg \
				dg_host $::alascan::totalFenergy_host {
			set temp_index [lsearch -exact $resid_nat $id_parse]
			if {[lindex $segname_nat $temp_index] == $segname_parse} {
				set resname_nat [lreplace $resname_nat $temp_index $temp_index $name_parse]
				set tmp_ddg_s1 [format %.1f [expr $dg_hg-$dg_host]]
				lappend ::alascan::ddg_s1 $tmp_ddg_s1
				}
			
			}
		set dGsorted [lsort -real -increasing $::alascan::ddg_s1]
		set dGmin [lindex $dGsorted 0]; set dGmax [lindex $dGsorted end]

        }; #hg==1
	#gradient
	set temp_range [expr {$dGmin-$dGmax}]
	if {$temp_range > 0} {set temp_range [expr -($temp_range)]}
	if {$temp_range > 10} {
		array set gradientStep [::alascan::gradient $dGmin $dGmax ]
		} else {
			array set gradientStep2 [::alascan::gradient2 $dGmin $dGmax]
			}
	
	#create canvas to display the residues along with free energy values
	#color spec for the secondary structure
	frame $w.l -bd 2 -relief ridge
	canvas $w.l.col -width 480 -height 32
	$w.l.col create text 38 19 -text "Secondary\nstructure" -font tkFixed
	$w.l.col create text 152 7 -text "C  E  T  B  H  G  I" -font tkFixed 
	$w.l.col create rect 79 12 99 30 -outline snow -fill snow
	$w.l.col create rect 100 12 119 30 -outline yellow -fill yellow
	$w.l.col create rect 120 12 139 30 -outline cyan -fill cyan
	$w.l.col create rect 140 12 159 30 -outline sienna -fill sienna
	$w.l.col create rect 160 12 179 30 -outline violet -fill violet
	$w.l.col create rect 180 12 202 30 -outline blue -fill blue
	$w.l.col create rect 202 12 222 30 -outline red -fill red
	$w.l.col create rect 79 12 79 30 -outline black -fill black
	$w.l.col create rect 100 12 100 30 -outline black -fill black
	$w.l.col create rect 120 12 120 30 -outline black -fill black
	$w.l.col create rect 140 12 140 30 -outline black -fill black
	$w.l.col create rect 160 12 160 30 -outline black -fill black
	$w.l.col create rect 180 12 180 30 -outline black -fill black
	$w.l.col create rect 200 12 200 30 -outline black -fill black
	$w.l.col create rect 222 12 222 30 -outline black -fill black

	#gradient scale
	$w.l.col create rect 320 12 460 24 -outline black -fill green	
	set gradStep [::alascan::gradLabel]
	set j 320; set temp_col blue
	if {$dGmin == $dGmax} {
	      	$w.l.col create rect 320 12 460 24 -fill blue
	   } else {
	      	for {set i 0} {$i <= 140} {incr i} {
        	$w.l.col create line $j 12 $j 24  -tags gradient -fill [lindex $gradStep $i]
		incr j
      		}
	     }

 	$w.l.col create text 390 7 -text "\u0394\u0394G" -font tkFixed
	$w.l.col create text 320 29.5 -text "$dGmin" -font tkFixed
	$w.l.col create text 460 29.5 -text "$dGmax" -font tkFixed
	pack $w.l -side top -padx 4 -pady 6 -expand 1 -fill x
	grid $w.l.col 	

	#title
	frame $w.tit
	canvas $w.tit.l1 -width 480 -height 30
	$w.tit.l1 create text 55 15 -text "Residue" -font "tkFixed 12 bold"
	$w.tit.l1 create text 240 7.5 -text "\u0394G(kcal/mol)" -font "tkFixed 12 bold"
	$w.tit.l1 create text 420 15 -text "\u0394\u0394G(kcal/mol)" -font "tkFixed 12 bold"
	if {$hg==0} {$w.tit.l1 create text 180 27 -text "In Protein" -font "tkFixed 12 bold"} 
	if {$hg==0} {$w.tit.l1 create text 293 27 -text "In Solvent" -font "tkFixed 12 bold"}
	if {$hg==1} {$w.tit.l1 create text 180 27 -text "Host-Guest" -font "tkFixed 12 bold"} 
	if {$hg==1} {$w.tit.l1 create text 293 27 -text "Host" -font "tkFixed 12 bold"}
	pack $w.tit -side top -pady 2 -expand 1 -fill x
	grid $w.tit.l1

	#list of residues 
	frame $w.frame
	canvas $w.frame.canvas -width 480 -height 400 -yscrollcommand "$w.frame.right set" \
		-scrollregion [list 0 0 480 $totalheight_nat]
	scrollbar $w.frame.right -orient vertical -command "$w.frame.canvas yview"
	pack $w.frame -fill x -side top -expand 1
	grid $w.frame.canvas $w.frame.right -sticky news
	

	if {$path == 0} {
	set var1 1
	foreach i $resname_nat j $segname_nat k $resid_nat l $sstruct_nat {
		set col [::alascan::sscol $l]		
		set temp_ind [lsearch -exact $parseResid $k]
		if {$temp_ind != "-1"} {

			if {[lindex $parseSegname $temp_ind] == $j} {
			set deltaG [lindex $totalFenergy $temp_ind]
			set deltaGerror [lindex $totalError $temp_ind]
			set dispHyst [lindex $hyst $temp_ind]
			if {$dispHyst < 0} {set dispHyst [expr -($dispHyst)]}
			set checkp [lindex $::alascan::ddg_s1 $temp_ind]
			if {$temp_range > 10} {
				set checkp1 [expr int($checkp)]
				set dgcol "$gradientStep($checkp1)"
			      } else {
					set checkp1 [format %.1f $checkp]
					set dgcol "$gradientStep2($checkp1)"
				}
			} else {
				set deltaG {}; set deltaGerror {}; set dispHyst {};
				}
		} else {
			set deltaG {}; set deltaGerror {}; set dispHyst {}
			}

		if {$dispHyst<$kt} {
			set colHyst green
			} elseif {$dispHyst<$twokt} {
			set colHyst orange
			} else { set colHyst red}
		set var [format %.2f [expr $var1 * 21.6]]; incr var1
		set data [format "%5s %3s %1s" $k $i $j]
		set energies [format "%5s" $deltaG]
		$w.frame.canvas create text 38 $var -text $data -font tkFixed	
		$w.frame.canvas create rectangle 83 [expr $var-10.5] 120 [expr $var+10.5] -outline $col -fill $col
		if {$deltaG != "" && $deltaGerror != "" && $dispHyst != {}} {
			$w.frame.canvas create text 144 $var -text "$energies" -font tkFixed
			$w.frame.canvas create text 182 $var -text "([format %.1f $dispHyst])" -font tkFixed
			$w.frame.canvas create oval 204 [expr $var-10] 224 [expr $var+10] -outline $colHyst -fill $colHyst
			$w.frame.canvas create text 265 $var -text "$lkup_bi_delg($i)" -font tkFixed
			$w.frame.canvas create text 303 $var -text "($lkup_bi_hys($i))" -font tkFixed
			$w.frame.canvas create oval 325 [expr $var-10] 345 [expr $var+10] -outline green -fill green
			set ddG [format %.1f [expr $deltaG-$lkup_bi_delg($i)]]
			$w.frame.canvas create text 385 $var -text "$ddG" -font tkFixed	
			$w.frame.canvas create rectangle 420 [expr $var-10.5] 470 [expr $var+10.5] -outline $dgcol -fill $dgcol
	

			}
		}
	} ;#if pathr==0

	if {$path == 3} {
	set var1 1
	foreach i $resname_nat j $segname_nat k $resid_nat l $sstruct_nat {
		set col [::alascan::sscol $l]		
		set temp_ind [lsearch -exact $::alascan::parseResid_hg $k]
		if {$temp_ind != "-1"} {

			if {[lindex $::alascan::parseSegname_hg $temp_ind] == $j} {
			set deltaG_hg [lindex $::alascan::totalFenergy_hg $temp_ind]
			set deltaG_host [lindex $::alascan::totalFenergy_host $temp_ind]
			set deltaGerror_hg [lindex $::alascan::totalError_hg $temp_ind]
			set deltaGerror_host [lindex $::alascan::totalError_host $temp_ind]
			set dispHyst_hg [lindex $::alascan::hyst_hg $temp_ind]
			if {$dispHyst_hg < 0} {set dispHyst_hg [expr -($dispHyst_hg)]}
			set dispHyst_host [lindex $::alascan::hyst_host $temp_ind]
			set tempDeltaG_hg [expr int($deltaG_hg)]
			set tempDeltaG_host [expr int($deltaG_host)]
			set checkp [lindex $::alascan::ddg_s1 $temp_ind]
			
			if {$temp_range > 10} {
				set checkp1 [expr int($checkp)]
				set dgcol "$gradientStep($checkp1)"
				} else {
					set checkp1 [format %.1f $checkp]
					set dgcol "$gradientStep2($checkp1)"
				}

			if {$dispHyst_hg < $kt} {
				set colHyst_hg green
				} elseif {$dispHyst_hg<$twokt} {
				set colHyst_hg orange
				} else { set colHyst_hg red}

			if {$dispHyst_host<$kt} {
				set colHyst_host green
				} elseif {$dispHyst_host<$twokt} {
				set colHyst_host orange
				} else { set colHyst_host red}
			} else {
				set deltaG_hg {}; set deltaGerror_hg {}; set dispHyst_hg {};
				}
		} else {
			set deltaG_hg {}; set deltaGerror_hg {}; set dispHyst_hg {}
			}

		set var [format %.2f [expr $var1 * 21.6]]; incr var1
		set data [format "%5s %3s %1s" $k $i $j]
		if {$deltaG_hg != {} && $deltaG_host != {}} {
			set energies_hg [format "%5s" $deltaG_hg]
			set energies_host [format "%5s" $deltaG_host]
			}
		$w.frame.canvas create text 38 $var -text $data -font tkFixed	
		$w.frame.canvas create rectangle 83 [expr $var-10.5] 120 [expr $var+10.5] -outline $col -fill $col
		if {$deltaG_hg != "" && $deltaGerror_hg != "" && $dispHyst_hg != {}} {
			$w.frame.canvas create text 144 $var -text "$energies_hg" -font tkFixed
			$w.frame.canvas create text 182 $var -text "([format %.1f $dispHyst_hg])" -font tkFixed
			$w.frame.canvas create oval 204 [expr $var-10] 224 [expr $var+10] -outline $colHyst_hg -fill $colHyst_hg
			$w.frame.canvas create text 265 $var -text "$energies_host" -font tkFixed
			$w.frame.canvas create text 303 $var -text "([format %.1f $dispHyst_host])" -font tkFixed
			$w.frame.canvas create oval 325 [expr $var-10] 345 [expr $var+10] -outline $colHyst_host -fill $colHyst_host
			set ddG [format %.1f [expr $deltaG_hg-$deltaG_host]]
			$w.frame.canvas create text 385 $var -text "$ddG" -font tkFixed	
			$w.frame.canvas create rectangle 420 [expr $var-10.5] 470 [expr $var+10.5] -outline $dgcol -fill $dgcol
			}
		}
	} ;# path 3
} ;#::alascan::summaryFEP



