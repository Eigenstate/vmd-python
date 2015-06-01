#######################################################################
# ParseFEP
#######################################################################
# ParseFEP is a tool for analyzing the results of a NAMD FEP run.
# ParseFEP computes: - free-energy differences
#                    - first-order error estimates
#                    - Gram-Charlier interpolations
#                    - simple overlap sampling free-energy differences
#                    - enthalpy and entropy differences
# ParseFEP displays: - free-energy time series
#                    - probability distributions
# ParseFEP assumes that Grace and ImageMagick are installed.
# All computations in kcal/mol.
#######################################################################
# Chris Chipot, Liu Peng, 2007-2012
# 1.8           2013-12-20
#######################################################################
# $Id: parseFEP.tcl,v 1.14 2014/12/10 16:08:57 johns Exp $

#######################################################################
# START-UP
#######################################################################


#----------------------------------------------------------------------
# INCLUSION OF PARSER. MAY 28, 2009, C.C.
#----------------------------------------------------------------------


package require exectool 1.2
package require multiplot

package provide  parsefep 1.9

namespace eval ::ParseFEP:: {
	namespace export parsefepgui
	namespace export parsefep

variable w

variable version 1.9.2

variable temperature 300.0
variable gcindex 	  ;#0 false; 1 true
variable max_order      
variable gaussian_guess  ;#0 false; 1 true
variable combine_method  ;#1 SOS  ; 0 BAR
variable dispindex       ;#0 false; 1 true
variable entropyindex    ;#0 false; 1 true
variable interfilesindex ;

variable fepofile
variable fepbofile
variable nb_file
variable nb_sample ;#
variable nb_equil ;# number of step before the equilibrium.
variable FEPfreq
variable t1
variable t2

variable k 0.001987200
variable kT  [expr $k * $temperature ]
variable delta_a 0.0

#----------------------------------------------------------------------
#  CORRECTION ADDED FEBRUARY 04, 2010, C.C.
#----------------------------------------------------------------------

variable variance_gauss
variable error_gauss
variable variance
variable error 
variable square_error 
variable square_error_gauss 

#----------------------------------------------------------------------
# END CORRECTION 
#----------------------------------------------------------------------

variable fep_delta_a_forward 
variable fep_delta_a_backward 
variable fep_delta_u_forward 
variable fep_delta_u_backward 
variable fep_delta_s_forward 
variable fep_delta_s_backward 

#---------------------------------------------------------------------
# drawing the free energy profile
#---------------------------------------------------------------------

}

proc ::ParseFEP::init_data { } {
	set ::ParseFEP::combine_method  0 
	set ::ParseFEP::gcindex         0
	set ::ParseFEP::entropyindex    0
	set ::ParseFEP::gaussian_guess  0
	set ::ParseFEP::max_order       ""
	set ::ParseFEP::dispindex       0
	set ::ParseFEP::fepofile        ""
	set ::ParseFEP::fepbofile       ""
	set ::ParseFEP::interfilesindex 0
}

proc ::ParseFEP::parsefepgui {} {	

	init_data

	puts "\n--------------------------"
	puts "ParseFEP: Version $::ParseFEP::version"
	puts "--------------------------"

# add traces to the checkboxes, so various widgets can be disabled appropriately
	
	if {[llength [trace info variable [namespace current]::fepbofile ]] == 0 } {
		trace add variable [namespace current]::fepbofile write ::ParseFEP::reading_para_sosbar
	}

	if {[llength [trace info variable [namespace current]::dispindex ]] == 0 } {
		trace add variable [namespace current]::dispindex write ::ParseFEP::reading_para_interfiles
	}

##########################
## read parameter section
##########################

	variable w

#De-minimize if the window is already running
	if { [winfo exists .parseFEP] } {
		wm deiconify $w
		raise .parseFEP
		return
	}

	set w [toplevel ".parseFEP"]
	wm title $w "ParseFEP"
	wm resizable $w yes yes 
	set row 0

#Add a menubar
	frame $w.menubar -relief raised -bd 2
	grid  $w.menubar -padx 1 -column 0 -columnspan 5 -row $row -sticky ew
	menubutton $w.menubar.help -text "Help" -underline 0 \
	-menu $w.menubar.help.menu
	$w.menubar.help config -width 5
	pack $w.menubar.help -side right
 
## help menu
	menu $w.menubar.help.menu -tearoff no
	$w.menubar.help.menu add command -label "About" \
	 -command {tk_messageBox -type ok -title "About ParseFEP" \
		-message "A tool for parsing the result of FEP output"}
	$w.menubar.help.menu add command -label "Usage" \
		-command {tk_messageBox -type ok -title "Useage:" \
		-message "Useage: \n \
			 ParseFEP is a tool for analyzing the results of a NAMD FEP run. \n\
			   ParseFEP computes:  \n \
				- free-energy differences \n\
				- first-order error estimates \n\
				- Gram-Charlier interpolations \n \
				- simple overlap sampling free-energy differences \n \
				- enthalpy and entropy differences \n \
			   ParseFEP displays:  \n\
				- free-energy time series \n \
				- probability distributions "
			}
	$w.menubar.help.menu add command -label "help..." \
		    -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/parsefep"
	 incr row

#Selection of parameter
	grid [label $w.enerlabel -text "Parameters "] \
		-row $row -column 0  -sticky w 
	incr row

#Input for temperature
	grid [label $w.templable -text "Temperature: "  -anchor w ] \
		-row $row -column 0 -sticky w
	grid [entry $w.temp -width 20 \
		-textvariable [namespace current]::temperature] \
		-row $row -column 1 -sticky w

	incr row

	grid [label $w.gcindex -text "Gram-Charlier order : "  -anchor w ] \
		-row $row -column 0 -sticky w
	grid [entry $w.temp2 -width 5 \
		-textvariable [namespace current]::max_order ] \
		-row $row -column 1 -sticky w

	incr row

	grid [checkbutton $w.disp -text "disp (this option is restricted to Unix-like systems)" \
       		-variable [namespace current]::dispindex ]  \
		-row $row -column 0 -sticky e
	grid [radiobutton $w.dispwithinterfiles    -text "Do or" -state disabled   \
		-variable [namespace current]::interfilesindex  -value "1" ]    \
		-row $row -column 1 -sticky ew
	grid [radiobutton $w.dispwithoutinterfiles -text "Don't keep intermediate files (for plotting purposes)" -state disabled \
		-variable [namespace current]::interfilesindex  -value "0" ] \
		-row $row -column 2 -sticky ew

	incr row
	grid [checkbutton $w.entropy -text   "entropy"                \
        	-variable [namespace current]::entropyindex ]         \
		-row $row -column 0 -sticky w
	grid [checkbutton $w.gg      -text   "Gaussian approximation" \
		-variable [namespace current]::gaussian_guess ]       \
		-row $row -column 1 -sticky w

	incr row

#selection for fep output file
	grid [label $w.fepolabel -text "FEP output file "] -row $row -column 0 -sticky w
	grid [entry $w.fepofile -width 20 -textvar [namespace current]::fepofile] -row $row -column 1 -columnspan 2 -sticky ew
	frame $w.fepobuttons 
	button $w.fepobuttons.fepobrowse -text "Browse" -command [ namespace code {
		set filetypes { 
			{{NAMD output file} {.fepout}}
			{{All Files} {*}}
		}
		set ::ParseFEP::fepofile [tk_getOpenFile -filetypes $filetypes]
		}
	]
	
	pack $w.fepobuttons.fepobrowse  -side left -fill x 
	grid $w.fepobuttons -row $row -column 2 -columnspan 1 -sticky nsew 
	incr row

#selection for file
	grid [label $w.fepbolabel -text "FEP(backward) output file"] -row $row -column 0 -sticky w
	grid [entry $w.fepbofile -width 20 -textvar [namespace current]::fepbofile] -row $row -column 1 -columnspan 2 -sticky ew
	frame $w.fepbobuttons 
	button $w.fepbobuttons.fepbobrowse -text "Browse" -command [ namespace code {
		set filetypes { 
        		{{NAMD output file} {.fepout}}
		{{All Files} {*}}
	        }
	        set ::ParseFEP::fepbofile [tk_getOpenFile -filetypes $filetypes]
	} ]
	pack $w.fepbobuttons.fepbobrowse  -side left -fill x 
	grid $w.fepbobuttons -row $row -column 2 -columnspan 1 -sticky nsew 
	incr row

#  SOS or BAR 

	grid [label $w.textfor_overlap -text "Combine forward and backward sampling: "  ] -row $row -column 0 -sticky w
	incr row
	grid [radiobutton $w.sosindex  -text "SOS-estimator" -state disabled -variable ::ParseFEP::combine_method -value "1" ] -row $row -column 0 -sticky w
	grid [radiobutton $w.barindex  -text "BAR-estimator" -state disabled -variable ::ParseFEP::combine_method -value "2" ] -row $row -column 1 -sticky w
	incr row

	grid [button $w.runbutton -text "Run FEP parsing" \
		-command [ namespace code {

		if { [string length $::ParseFEP::fepofile] < 1 } {
			tk_dialog .errmsg {NamdPlot Error} "No FEP (inward) logfile selected." error 0 Dismiss
			return
		}

		if { $::ParseFEP::max_order !=  "" } {
			set ::ParseFEP::gcindex   1  ;# run the gram-chalier expansion
			puts "ParseFEP: Gram-Charlier expansion up to order $::ParseFEP::max_order "
		} else {	
			set ::ParseFEP::gcindex   0 
			puts "ParseFEP: No valid expansion order provided. Therefore, Gram-Charlier analysis of convergence will not be performed. "
		}

		namdparse
	}
	]] -row $row -column 0 -columnspan 5 -sticky nsew 

	incr row
	 
}

# This gets called by VMD the first time the menu is opened.
proc  ParseFEP_tk_cb {} {	
	
	::ParseFEP::parsefepgui 

	return $::ParseFEP::w
	
}

proc parsefepgui {}       {return [eval ::ParseFEP::parsefepgui      ]}
proc parsefep    { args } {return [eval ::ParseFEP::parsefepcmd $args]}

proc ::ParseFEP::reading_para_sosbar {args} {
	variable w

	if { $::ParseFEP::fepbofile == "" } {
		$w.barindex configure -state disabled
		$w.sosindex configure -state disabled
	} else {
		$w.barindex configure -state normal
		$w.sosindex configure -state normal
	}
}

proc ::ParseFEP::reading_para_interfiles {args} {
	variable w
	if { $::ParseFEP::dispindex == 0 } {
		$w.dispwithinterfiles    configure -state disabled 
		$w.dispwithoutinterfiles configure -state disabled
	} else {
		$w.dispwithinterfiles    configure -state normal
		$w.dispwithoutinterfiles configure -state normal
	}
}

proc ::ParseFEP::namdparse {  } {

#######################################################################
# initialize the variable
#######################################################################

set  ::ParseFEP::delta_a 0.0

#----------------------------------------------------------------------
#  CORRECTION ADDED FEBRUARY 04, 2010, C.C.
#----------------------------------------------------------------------

set  ::ParseFEP::variance_gauss 0.0
set  ::ParseFEP::error_gauss 0.0
set  ::ParseFEP::variance 0.0
set  ::ParseFEP::error 0.
set  ::ParseFEP::square_error 0.0
set  ::ParseFEP::square_error_gauss 0.0

#----------------------------------------------------------------------
# END CORRECTION 
#----------------------------------------------------------------------

set ::ParseFEP::fep_delta_a_forward 0.0
set ::ParseFEP::fep_delta_u_forward 0.0
set ::ParseFEP::fep_delta_s_forward 0.0


#---------------------------------------------------------------------
# drawing the free energy profile
#---------------------------------------------------------------------

set ::ParseFEP::fwdlog        ""
set ::ParseFEP::bwdlog        ""
set ::ParseFEP::overlap_array ""
set ::ParseFEP::file_lambda   ""


#######################################################################
# OUTPUT FILE
#######################################################################
	set namdlogfile $::ParseFEP::fepofile

	file delete ParseFEP.log 
	set temp [open ParseFEP.log "a+"]
		puts $temp "#========================================================================================" 
		puts $temp "#                     Free energy perturbation    " 
		puts $temp "#----------------------------------------------------------------------------------------" 
		puts $temp "#forward/backward          \u03BB              \u0394\u0394A                 \u0394A                  \u03b4\u03b5   "
		puts $temp "#                                                 " 
		puts $temp "#----------------------------------------------------------------------------------------" 
		
		set file_temp [open $namdlogfile "r"]
			while { [gets $file_temp line] >= 0 } {
				if {[regexp change $line]}  { 
					set elem_temp [lindex $line  7] 
					break
				}
			}
		close $file_temp

		puts $temp [ format "forward:            %9.4f           %9.4f           %9.4f           %9.4f  "  $elem_temp  0.0 0.0 0.0]

	close $temp

	if { $::ParseFEP::gcindex == 1 } { 
		file delete gc.log
		set temp [open gc.log "w+"]
			puts $temp "#=================================================="
			puts $temp "#        G-C expansion    "
			puts $temp "#--------------------------------------------------"
			puts $temp "#forward/backward                \u0394\u0394A             \u0394A"
			puts $temp "#                          " 
			puts $temp "#--------------------------------------------------"
			puts $temp [format "for/back:  gc:            %8.4f         %8.4f      "       0.000       0.000  ]
		close $temp
	}

	if { $::ParseFEP::entropyindex == 1  } {
		file delete entropy.log
		set temp [open entropy.log "a+"]
			puts $temp " "
			puts $temp "#========================================================================================="
			puts $temp "#                  Perturbation theory              " 
			puts $temp "#-----------------------------------------------------------------------------------------" 
			puts $temp "#forward/backward             \u03bb             \u0394\u0394U          \u0394U            \u0394\u0394S             \u0394S" 
			puts $temp "#                                                    "
			puts $temp "#-----------------------------------------------------------------------------------------"	
			puts $temp [format "forward:  entropy         %8.4f      %8.4f      %8.4f      %8.4f      %8.4f"  $elem_temp  0. 0. 0. 0. ]
		close $temp
  
	}

#----------------------------------------------------------------------
# CORRECTION ADDED DECEMBER 27, 2005, C.C.
#----------------------------------------------------------------------
# CORRECTION ADDED APRIL 21, 2009, C.C.
#----------------------------------------------------------------------
# MODIFICATION MAY 28, 2009, C.C.
#----------------------------------------------------------------------

	puts "ParseFEP: Get the number of samples per window"

	set free_energy_tempo ""
	set nb_equil_temp 0
	set nb_file_temp 0

 	set infile [open $namdlogfile "r"]
	while { [gets $infile line] >= 0 } {
		if { [regexp  FepEnergy $line ] } {
			lappend free_energy_tempo   " [lindex $line 1]   [lindex $line 6]   [lindex $line 9]"
	 	   }
		if { [regexp change $line]} {
			incr nb_file_temp 
		}		
	}
 	close $infile

 	set infile2 [open $namdlogfile "r"]
	while { [gets $infile2 line] >= 0 } {
		if { [regexp  EQUIL $line ] } {
			set ::ParseFEP::nb_equil [ string trim [lindex $line 0] "#" ]
			break
	 	   }
	}
 	close $infile2

	set ::ParseFEP::nb_file $nb_file_temp

	set t2 [lindex  [lindex $free_energy_tempo 1] 0]
	set t1 [lindex  [lindex $free_energy_tempo 0] 0]

	set ::ParseFEP::FEPfreq [expr  $t2 - $t1]
	set ::ParseFEP::nb_equil [expr $::ParseFEP::nb_equil / $::ParseFEP::FEPfreq ]

#----------------------------------------------------------------------
# END CORRECTION 
#----------------------------------------------------------------------

	set  tempo_size  [ llength $free_energy_tempo]
	unset free_energy_tempo

	set  ::ParseFEP::nb_sample [expr $tempo_size / $::ParseFEP::nb_file ]

	puts "ParseFEP: Split time-series in $::ParseFEP::nb_file files"

	puts "ParseFEP: $::ParseFEP::FEPfreq steps between stored samples"
	puts "ParseFEP: $::ParseFEP::nb_sample effective samples per windows"
	puts "ParseFEP: $::ParseFEP::nb_equil effective equilibration steps per window"
	puts "ParseFEP: Parse time-series"

#puts "$::ParseFEP::nb_equil is just ok "
	
#normal analysis
	::ParseFEP::normal_parse_log $::ParseFEP::fepofile forward 

	if  { $::ParseFEP::fepbofile != "" }  {
		::ParseFEP::normal_parse_log  $::ParseFEP::fepbofile backward	
# inaccuracy
		::ParseFEP::inaccuracy_estimation
	}
	
	::ParseFEP::combine_file 
	
	if { $::ParseFEP::combine_method  == 1 }  { 
		::ParseFEP::sos_estimate  
	}

	if { $::ParseFEP::combine_method  == 2 }  {
		::ParseFEP::bar_estimate
	}

	if { $::ParseFEP::dispindex == 1 }  {

		global tcl_platform 
		set OS $tcl_platform(platform)
		
		if { [regexp Windows $OS ]} {
			::ParseFEP::plotting_free_energy_profile
		}
		
		if { [regexp unix $OS ] } {
			::ParseFEP::fepdisp_unix
		}

	}
	puts "ParseFEP: All calculations complete."
}

proc ::ParseFEP::combine_file { } {
#######################################################################
# combine the file of entropy.log, gc.log and ParseFEP.log ( if fileutil is accessible, this part can be replaced by it. )
#######################################################################

	set target_file   [open ParseFEP.log "a+"]

	if  { $::ParseFEP::entropyindex == 1}  {
		set source_file [open entropy.log "r"]
		while { [gets $source_file line] >= 0 } {
			puts $target_file "$line"
		}	
		puts $target_file  "  "  

		close $source_file
		file delete entropy.log 
	}

	if  {$::ParseFEP::gcindex == 1 } {		
		set source_file [open gc.log "r"]

		while { [gets $source_file line] >= 0 } {
			puts $target_file "$line"
		}	
		puts $target_file  "  "

		close $source_file
		file delete gc.log 
	}

	close $target_file	

##############################################
#### end of this part.
##############################################
}

proc ::ParseFEP::normal_parse_log { namdlogfile  fororback} {

	puts "ParseFEP: The file been analysed is $namdlogfile"
	set num_indicator 0
 	set window  0

	set ::ParseFEP::file_lambda ""

 	set infile [open $namdlogfile "r"]
	while { [gets $infile line] >= 0 } {
		if { [regexp change $line]} {
			lappend ::ParseFEP::file_lambda  $line 
		}		
	}
 	close $infile

	set energy_difference ""
	set file_entropy ""
	set file ""	

        set infile [open $namdlogfile "r"]
	set size [file size "$namdlogfile"]

#set testfile [open textfile "w"] 
	set data_file [split [read $infile $size ] "\n" ] 

	foreach line $data_file  {

		if { [ regexp  FepEnergy $line ] } {

			incr  num_indicator  

			if { [expr   $num_indicator > $::ParseFEP::nb_equil  ]} {
 
## accumulating for the Error analysis on first-order expansion of free energy
##
##    the lable of string in one line is begin with  0, for tcl.
##

				lappend file  "[lindex $line 1]  [lindex $line 6 ]  [lindex $line 9 ]"
				lappend file_entropy "[lindex $line 1]  [lindex $line 2 ]  [lindex $line 3]  [lindex $line 4]  [lindex $line 5 ]"
			}

			if {  $num_indicator == $::ParseFEP::nb_sample  } {

				incr window
	
				set mean_xi [::ParseFEP::analysis_normal_result $file  $window  $fororback ]
		##		set kappa_now [::ParseFEP::cal_tau  [lindex $mean_xi 0 ] $file]

				if { $::ParseFEP::gcindex == 1 } {
					 ::ParseFEP::Gram_Charlier_analysis $file $mean_xi  $fororback
				}

				::ParseFEP::FEP_formula  $file	$file_entropy  $window $mean_xi $fororback
			
				set file ""
				set file_entropy ""
				set num_indicator 0

			}
		}
	}

	close $infile	

}


proc ::ParseFEP::analysis_normal_result { array  window   fororback} {
   
	set length_array [llength $array]
## calculate the mean
	set  mean_accum 0.
	foreach elem_now  $array {	
		foreach { i j k } $elem_now {break }
		set mean_accum [expr { $mean_accum +  $j } ]
	}	
	set mean [expr { $mean_accum /  $length_array}   ]

## calculate the sigma
	set sigma_accum 0.
	foreach elem_now $array  {	
		foreach {i j k} $elem_now {break } 
		set sigma_accum [expr { $sigma_accum + (  $j - $mean ) ** 2 } ]
	}	
	set sigma [expr { sqrt ($sigma_accum / ($length_array +1  ))  }  ]

#######################################################################
#  COMPUTE xi
#######################################################################
        set   xi     [ expr $sigma / sqrt(2) ]

	puts "$fororback:   "
	puts "$fororback:  ParseFEP:==========================="
	puts "$fororback:  ParseFEP:Window  $window"
	puts "$fororback:  ParseFEP:==========================="
	puts "$fororback:                 <\u0394 U>  = $mean"
	puts "$fororback:                   \u03c3    = $sigma"
	puts "$fororback:                   \u03be    = $xi "

	
	return $xi

}


proc ::ParseFEP::block_sigma { mean size_block  num_block  F_array } {

##
##  N : size of block 
##  n : number of block 
## 

	set Fav_accum 0
	set index_block  0

	set sigma_factor_accum 0.

	foreach F_i $F_array {
		incr index_block 
		set Fav_accum  [ expr { $F_i + $Fav_accum } ]
		if { [expr { $index_block == $size_block } ] } {
			set sigma_factor_accum [ expr   { ( $Fav_accum / $size_block    -  $mean  ) ** 2.0+ $sigma_factor_accum }   ] 
			set Fav_accum 0
			set index_block  0
		}		
	}

	return  $sigma_factor_accum
}


proc ::ParseFEP::mean_sigma {array } {
	
	set nSample [llength  $array ]
## mean of the boltzmann factor  
	set mean_accum 0.
	foreach   factor_now   $array  {  
		set mean_accum [expr  $factor_now + $mean_accum  ]
	}
	set mean  [expr $mean_accum  / $nSample]
	

##  sigma of  the boltzmann factor  
	set sigma_accum 0.
	foreach factor_now  $array    {  
		set sigma_accum [ expr   { ( $factor_now - $mean ) ** 2 + $sigma_accum } ]
	}

	set sigma [expr { sqrt( $sigma_accum  /  $nSample )  } ]

	set  result "$mean  $sigma "

	return $result 

}



proc  ::ParseFEP::Gram_Charlier_analysis  { array_of_file   xi  fororback } {
	
	puts  "$fororback:  ParseFEP:  Gram-Charlier analysis of convergence"
	
#######################################################################
#  CHANGE OF VARIABLES: x = U / 2 xi
#######################################################################

	set array_deltaG_now ""
	
	foreach deltaG $array_of_file {
		foreach  {i j k } $deltaG  {break } 
		lappend array_deltaG_now  [ expr {   $j * 1.0 / 2 / $xi  / $::ParseFEP::k /$::ParseFEP::temperature }  ]
	}	
	
#######################################################################
#  COMPUTE HERMITE POLYNOMIALS
#######################################################################

	puts "$fororback:            Average Hermite polynomials:"
	puts "$fororback:            -------------------------------------------"

#######################################################################
#  INITIALIZE WITH THE TWO FIRST TERMS
#######################################################################
	set hermite "1.0"
	
#######################################################################
#  0-th ORDER
#######################################################################
	set hermite_previous ""

	foreach deltaG  $array_deltaG_now  {
		lappend hermite_previous  1.0
	}		

#######################################################################
#  1-th ORDER
#######################################################################
	set hermite_now ""
	
	foreach deltaG  $array_deltaG_now  {
		lappend hermite_now  [expr { $deltaG * 2 }  ]
	}		

	set h1_accum 0.0

	foreach deltaG $hermite_now {
		set h1_accum [expr $h1_accum + $deltaG  ]
	}
	set h1 [expr $h1_accum / [llength  $array_deltaG_now] ]

	lappend hermite $h1

#######################################################################
#  RECURSIVE FORMULA
#######################################################################	

	for {set n 1} { $n < [expr 1 + $::ParseFEP::max_order ] }  { incr n  } {
		
		set hermite_back ""

		foreach elem_array $array_deltaG_now  elem_hermite_now  $hermite_now  elem_hermite_previous $hermite_previous  {
			
			lappend hermite_back  [expr  { 2 *  $elem_array   * $elem_hermite_now  - 2 * $n * $elem_hermite_previous  }]
		}

		set hp_accum 0.
		foreach elem_hermite $hermite_back {
			set hp_accum [expr {  $hp_accum + $elem_hermite }  ]
		}
		set length_back [llength $hermite_back ]
		lappend hermite [expr { $hp_accum /  $length_back } ]
	
		set hermite_previous $hermite_now
		set hermite_now 	$hermite_back
	}

#######################################################################
#  COMPUTE FACTORIALS
#######################################################################

	set factorial "1.0"

#######################################################################
#  RECURSIVE FORMULA
#######################################################################
	
	set u 1
	
	for {set n 1} { $n < [expr 2 + $::ParseFEP::max_order ] }  { incr n  } {
		set v [expr $n *  $u ]
		lappend factorial $v
		set u $v		
	}

#######################################################################
#  GATHER HERMITE POLYNOMIALS CONTRIBUTIONS
#######################################################################

	set s 0.
	set n_hermite 0 
	foreach elem_hermite $hermite elem_factorial  $factorial  {
	
		set s [expr { $s +  (-1) ** $n_hermite *  $elem_hermite  * $xi **$n_hermite /  $elem_factorial }  ]

		if { $s <= 0. } {
			puts [format "$fororback:            n = %2d: -kT ln <exp(-\u0394U/kT)> =  nan "  $n_hermite  ]
		} else {
			set gc_temp [expr 	{ $::ParseFEP::kT * log(exp ($xi ** 2) *$s) }  ]
			puts [format "$fororback:            n = %2d: -kT ln <exp(-\u0394U/kT)> = %12.6f"  $n_hermite   $gc_temp ]
		}
		incr n_hermite 
	}
	
	if  { $s <= 0.} {
		set instant_delta_a "nan"
	} else {
		set instant_delta_a  [expr  { -1.0 * $::ParseFEP::k * $::ParseFEP::temperature * log(exp (1.0* $xi ** 2) * 1.0 *$s)  }  ]
	}
	
	#puts [format "%7.4f" $instant_delta_a ]
	
	if { $s > 0. } {
		set ::ParseFEP::delta_a [expr $::ParseFEP::delta_a + $instant_delta_a ]
	}	
	
	puts "$fororback:            -------------------------------------------"
	puts "$fororback:            \u0394\u0394A (G-C) =  $instant_delta_a   \u0394A (G-C)   = $::ParseFEP::delta_a"
	
	set temp_file [open gc.log "a+" ]
		if { $s <=0. } {
			if { $fororback == "forward" } { 
				puts $temp_file [ format "forward:   gc:                 nan         %8.4f"   $::ParseFEP::delta_a ] 
			} else {
				puts $temp_file [ format "backward:  gc:                 nan         %8.4f"   $::ParseFEP::delta_a ] 
			}		
		} else {
			if { $fororback == "forward" } { 
				puts $temp_file [ format "forward:   gc:            %8.4f         %8.4f"  $instant_delta_a $::ParseFEP::delta_a ] 
			} else {
				puts $temp_file [ format "backward:  gc:            %8.4f         %8.4f"  $instant_delta_a $::ParseFEP::delta_a ] 
			}
		}
	close $temp_file

#######################################################################
#  END OF GRAM-CHARLIER INTERPOLATION
#######################################################################
}


proc ::ParseFEP::FEP_formula { file  file_entropy   window mean_deltaG fororback } {

#####################
	set temp  [open ParseFEP.log "a+"]

	if { $fororback == "backward"  && $window == 1 } {
		set ::ParseFEP::fep_delta_s_backward  $::ParseFEP::fep_delta_s_forward
		set ::ParseFEP::fep_delta_u_backward  $::ParseFEP::fep_delta_u_forward
		set ::ParseFEP::fep_delta_a_backward  $::ParseFEP::fep_delta_a_forward	
		set ::ParseFEP::error  0.
		set ::ParseFEP::square_error 0.

		set file_temp [open  $::ParseFEP::fepbofile "r"]
			while { [gets $file_temp line] >= 0} {
				if { [regexp change $line] } {
					set  elem_temp [lindex $line 7]
					break
				}
			}
		close $file_temp

		puts $temp  [format "backward:           %9.4f           %9.4f           %9.4f           %9.4f  "  $elem_temp    0.00      $::ParseFEP::fep_delta_a_backward 0.00    ]
	}

	close $temp
	
	if { $fororback == "backward"  && $window == 1 && $::ParseFEP::entropyindex == 1 } {
		set tempn [open entropy.log "a+"]
                puts $tempn [format "backward: entropy         %8.4f      %8.4f      %8.4f      %8.4f      %8.4f"   $elem_temp  0.0     $::ParseFEP::fep_delta_u_backward   0.0  $::ParseFEP::fep_delta_s_backward ]
		close $tempn
	}

#####################


#######################################################################
#  COMPUTE CORRELATION LENGTH FOR <exp(-Delta U/kT)>
#######################################################################


	set nSample [llength $file ]

	if  { [expr $mean_deltaG !=0 ] } {
		set enlarge_temp  0.
		set j_temp 0.
		foreach deltaG_now $file  {  
			foreach {i  j  k}  $deltaG_now  { break }
			if { [expr $j_temp < $j ]} { set j_temp  $j }
		}

##  convert delta_G to boltzmann factor 
		set bolt_array ""
		foreach deltaG_now $file  {  
			foreach {i  j  k}  $deltaG_now  { break }
			lappend bolt_array [expr  {   (   exp( -1 *  $j  / $::ParseFEP::k /$::ParseFEP::temperature )) } ]
		}
	
		set mean_sigma [::ParseFEP::mean_sigma $bolt_array ]
		foreach { mean_factor sigma_factor }  $mean_sigma  { break }

## Sampling ratio for exp(-dV/RT)
## using block averages
	
		set array_var_tau ""

		set kmax [expr int ( log ( $nSample ) / log ( 2 ) )  ]
		if  { [ expr 2 ** $kmax  > $nSample]  } { incr kmax -1 ] }

		
		puts  "$fororback:  COMPUTE CORRELATION LENGTH FOR <exp(-\u0394U/kT)> "	
	        puts  "$fororback:  -------------------------------------------------------------------------------------------------------------"
	        puts  "$fororback:       blocks         samples            \u03c3**2            error               1+2\u03ba"	
     		puts  "$fororback:  -------------------------------------------------------------------------------------------------------------"
		
		set k_index ""
		set k $kmax
		while {  [expr $k > 0 ]}  { 	lappend k_index $k; incr k  -1  }

		foreach k $k_index {
			
			 # N : size of block averages
			set N_block [expr int (  $nSample / ( 2 ** $k ) ) ] 

			#set factor_av_array	[ ::ParseFEP::block $N_block  $bolt_array ]		
		 
			#number of block			
			set num_block [expr   $nSample/ $N_block ]  ;# caution:   int( ) has been performed. nSample, N_block are integer. 

			set sigma_accum [ ::ParseFEP::block_sigma $mean_factor $N_block $num_block $bolt_array ]

			set var [expr  $sigma_accum    /   ( $num_block - 1. )  / $num_block ]
			
			set error  [expr $var * sqrt( 2.0 ) / sqrt($num_block - 1.) ] 

			set  tau   [expr  $nSample * $var / ($sigma_factor ** 2) ] 
			
			puts [format "$fororback:         %5d      %5d           %2.5f            %2.5f             %2.5f"   $N_block     $num_block      $var        $error     $tau  ]

			lappend array_var_tau $var $tau
		 }


		 puts  "$fororback:  -----------------------------------------------------------------------------------------------------"
		 puts  "$fororback:  ParseFEP: Summary of thermodynamic quantities in kcal/mol"
	
		 foreach {t c } {0. 0.} {break }	
		 while { $c == 0 } {
			set t [expr $t+ 0.001]
			set f 0.000001
			set tmax 0.

			foreach {var_now tau_now}  $array_var_tau {
				if {  [ expr $var_now/$f  < $t +1  &&  $var_now / $f > 1 - $t ]  } {
					set fmin $var_now
					set tmax $tau_now
				}				
				set f $var_now
			}
			set c $tmax 
	 	}

		set kappa  $c
		
	} else {
		set kappa 1.0
	}

#######################################################################
#  COMPUTE CORRELATION LENGTH FOR <exp(-Delta U/2kT)>
#######################################################################

	if  { [expr  $mean_deltaG != 0 ] } {
	
##  convert delta_G to boltzmann factor 
		set bolt_array ""
		foreach deltaG_now $file  {  
			foreach {, j_line ,} $deltaG_now { break }
			lappend bolt_array [expr  {  exp( -1 *  $j_line  / 2 / $::ParseFEP::kT ) }  ]
		}

		set mean_sigma [::ParseFEP::mean_sigma $bolt_array  ]
		foreach {mean_factor sigma_factor} $mean_sigma  { break }

## Sampling ratio for exp(-dV/RT)
## using block averages
	
		set tau 0.0
		set array_var_tau ""
		set error 0.

		set kmax [expr int ( log ( $nSample ) / log ( 2 ) )  ]
		if  { [expr 2 ** $kmax  > $nSample  ]} { incr kmax -1 }
		
		set k_index ""
		set k $kmax
		while {  [expr $k > 0 ]}  { lappend k_index $k; incr k  -1 }

		puts  "$fororback:    COMPUTE CORRELATION LENGTH FOR <exp(-\u0394U/2kT)>"	
	        puts  "$fororback:  -------------------------------------------------------------------------------------------------------------"
	        puts  "$fororback:       blocks         samples            \u03c3**2             error               1+2\u03ba"	
     		puts  "$fororback:  -------------------------------------------------------------------------------------------------------------"

		foreach k $k_index {	
			
			# N : size of block averages
			set N_block [expr int (  $nSample / ( 2 ** $k ) ) ] 	
		 
			#number of block			
			set num_block [expr   $nSample/ $N_block ]  ;# caution:   int( ) has been performed. nSample, N_block are integer. 

			set sigma_accum [ ::ParseFEP::block_sigma $mean_factor $N_block $num_block $bolt_array ]

			set var [expr  $sigma_accum    /   ( $num_block - 1. )  / $num_block ]
			
			set error  [expr $var * sqrt( 2.0 ) / sqrt($num_block - 1.) ] 
			set  tau   [expr  $nSample * $var / ($sigma_factor ** 2) ] 
			
			puts [format "$fororback:         %5d      %5d           %2.5f            %2.5f             %2.5f"   $N_block     $num_block      $var        $error     $tau  ]

			lappend array_var_tau $var $tau
		 }


		 puts  "$fororback:  -----------------------------------------------------------------------------------------------------"
		 puts  "$fororback:  ParseFEP: Summary of thermodynamic quantities in kcal/mol"
	
		 foreach {t c } {0. 0.} {break }	
		 while { $c == 0 } {
			set t [expr $t+ 0.001]
			set f 0.000001
			set tmax 0.

			foreach {var_now tau_now}  $array_var_tau {
				if {  [ expr $var_now/$f  < $t +1  &&  $var_now / $f > 1 - $t ]  } {
					set fmin $var_now
					set tmax $tau_now
				}				
				set f $var_now
			}
			set c $tmax 
	 	}

		set kappa2  $c
		
	} else {
		set kappa2 1.0
	}
##################################################################################
# the end of calculation of correlation length 
##################################################################################

	set instant_accum 0. 

	foreach elem_file $file {
		foreach { , j_line ,} $elem_file  {break}
		set instant_accum [ expr { $instant_accum +  exp ( -1. *  $j_line / $::ParseFEP::kT ) } ]
	}	

	set instant_fep_delta_a  [expr   -1. * $::ParseFEP::kT  * log ( 1.0* $instant_accum / [llength $file ] ) ]

	if { $fororback == "forward" } {
		set ::ParseFEP::fep_delta_a_forward [expr $::ParseFEP::fep_delta_a_forward + $instant_fep_delta_a ]

		puts "$fororback:            -------------------------------------------"
		puts "$fororback:            \u0394\u0394A (FEP) =  $instant_fep_delta_a     \u0394A (FEP)   = $::ParseFEP::fep_delta_a_forward"
		puts "$fororback:            -------------------------------------------"

	} else {
		set ::ParseFEP::fep_delta_a_backward [expr $::ParseFEP::fep_delta_a_backward + $instant_fep_delta_a ]

		puts "$fororback:            -------------------------------------------"
		puts "$fororback:            \u0394\u0394A (FEP) =  $instant_fep_delta_a     \u0394A (FEP)   = $::ParseFEP::fep_delta_a_backward"
		puts "$fororback:            -------------------------------------------"
	}


	if { $::ParseFEP::entropyindex == 1 }	{

		foreach {s1 s2 s3  n} { 0. 0. 0. 0 } { break }

		foreach elem_diff $file_entropy {
			foreach  { , j k l m }  $elem_diff { break } 
			set doller1  [expr  {$j   + $l  }  ] 
			set doller2  [expr  {$k + $m } ]
			incr n
			set s1 [expr { $s1 * ($n - 1 ) * 1.0 / $n + exp ( - ($doller2 - $doller1 ) / $::ParseFEP::k  / $::ParseFEP::temperature ) * $doller2  * 1.0 / $n }  ]
			set s2 [expr { $s2 * ($n - 1 ) *1.0 / $n + exp ( - ($doller2 - $doller1 ) / $::ParseFEP::k  / $::ParseFEP::temperature)  * 1.0 / $n} ]
			set s3 [expr { $s3 * ($n - 1 ) *1.0 / $n + $doller1  * 1.0 / $n }  ]
			set temp [expr ( $doller2 - $doller1 )]

		 	# puts "$s1 $s2 $s3 $doller1 $doller2 $j $k  $l  $m"

		}
		
		set instant_fep_delta_u    [expr  {  $s1  /  $s2  -  $s3  } ]

		set   instant_fep_delta_s  [expr { $::ParseFEP::temperature * ($::ParseFEP::k *log ( $s2 *1.0  ) + $instant_fep_delta_u / $::ParseFEP::temperature  ) } ]

		set ideltai	[lindex [lindex $::ParseFEP::file_lambda [expr  $window  -1 ] ] 8 ]
		set i           [lindex [lindex $::ParseFEP::file_lambda [expr  $window  -1 ] ] 7 ]
		if { $fororback == "forward" } {
	
			set ::ParseFEP::fep_delta_u_forward [ expr { $::ParseFEP::fep_delta_u_forward + $instant_fep_delta_u}   ]
			set ::ParseFEP::fep_delta_s_forward [ expr { $::ParseFEP::fep_delta_s_forward + $instant_fep_delta_s}   ]

			puts "$fororback:            \u0394\u0394U (FEP) =  $instant_fep_delta_u   \u0394U (FEP)   = $::ParseFEP::fep_delta_u_forward"
	    		puts "$fororback:            -------------------------------------------"
			puts "$fororback:            \u0394\u0394S (FEP) =  $instant_fep_delta_s   T \u0394S (FEP) = $::ParseFEP::fep_delta_s_forward"
			puts "$fororback:            -------------------------------------------"

		}  else {

			set ::ParseFEP::fep_delta_u_backward [ expr { $::ParseFEP::fep_delta_u_backward + $instant_fep_delta_u}   ]
			set ::ParseFEP::fep_delta_s_backward [ expr { $::ParseFEP::fep_delta_s_backward + $instant_fep_delta_s}   ]

			puts "$fororback:            \u0394\u0394U (FEP) =  $instant_fep_delta_u   \u0394U (FEP)   = $::ParseFEP::fep_delta_u_backward"
	    		puts "$fororback:            -------------------------------------------"
			puts "$fororback:            \u0394\u0394S (FEP) =  $instant_fep_delta_s   T \u0394S (FEP) = $::ParseFEP::fep_delta_s_backward"
			puts "$fororback:            -------------------------------------------"

		}
	
		set temp_file [open entropy.log "a+" ]


		if { $fororback == "forward" } {

			puts  $temp_file [format   "forward:  entropy         %8.4f      %8.4f      %8.4f      %8.4f      %8.4f" $ideltai  $instant_fep_delta_u     $::ParseFEP::fep_delta_u_forward   $instant_fep_delta_s  $::ParseFEP::fep_delta_s_forward ] 
		} else {
			puts  $temp_file [format   "backward: entropy         %8.4f      %8.4f      %8.4f      %8.4f      %8.4f" $ideltai  $instant_fep_delta_u     $::ParseFEP::fep_delta_u_backward   $instant_fep_delta_s  $::ParseFEP::fep_delta_s_backward ] 
		}

		close $temp_file 

	}
	
	puts  "$fororback:   errorFEP: Error estimate from 1-st order perturbation theory:"
	# puts  "$fororback:   D E B U G: kappa=$kappa kappa2=$kappa2"

	foreach {average_a_accum  average_b_accum   average_c_accum   average_e_accum   average_f_accum} { 0. 0. 0. 0. 0. 0. } { break }
		
	foreach elem_file $file 	{
		
		foreach { , j , }  $elem_file { break  } 

		set average_a_accum [expr {  $average_a_accum + exp ( -2 * $j / $::ParseFEP::kT ) } ]
		set average_b_accum [expr { $average_b_accum + exp ( -1 * $j / $::ParseFEP::kT ) } ]
		set average_c_accum [expr { $average_c_accum + exp ( -1 * $j /2 /$::ParseFEP::kT ) } ]
		set average_e_accum [expr { $average_e_accum + $j ** 2 } ]
		set average_f_accum [expr { $average_f_accum + $j } ]
	}
		
	set n [llength $file ]
	
	set average_a [expr { $average_a_accum / $n} ]
	set average_b [expr { $average_b_accum / $n} ]
	set average_c [expr { $average_c_accum / $n} ]
	set average_e [expr { $average_e_accum / $n} ]
	set average_f  [expr { $average_f_accum / $n} ]

	set fluctuat_gauss [expr { $average_e - $average_f ** 2} ]
	set fluctuat   [expr { $average_a - $average_b ** 2}  ]
	set fluctuat2 [expr { $average_b - $average_c ** 2} ]

	set sampling [expr $n / $kappa ]
	set sampling2 [expr $n / $kappa2 ]

	set kt [expr  { $::ParseFEP::k * $::ParseFEP::temperature } ]	
	
	set instant_variance [ expr  { $kt ** 2 * $fluctuat / $average_b ** 2}  ]
	
	puts [format "$fororback:                     kT                                       = %20.6f"     	 $kt ]
	puts [format "$fororback:                     1+2\u03ba                                     = %20.6f"    	 $kappa ]
	puts [format "$fororback:                     N/(1+2\u03ba)                                 = %20.6f"   	 $sampling]
	puts [format "$fororback:                     <exp(-2\u0394U/kT)>                           = %20.6f" $average_a]
	puts [format "$fororback:                     <exp(-\u0394U/kT)>                            = %20.6f"  $average_b]
	puts [format "$fororback:                     <exp(-\u0394U/2kT) >                          = %20.6f" $average_c]
	puts [format "$fororback:                     \u03c3**2 = <exp(-2\u0394U/kT)> - <exp(-\u0394U/kT)>**2 = %20.6f" $fluctuat ]
	puts [format "$fororback:                     \u03c3**2 = <exp(-\u0394U/kT)> - <exp(-\u0394U/2kT)>**2 = %20.6f" $fluctuat2 ]
	puts "$fororback:                     -----------------------------------------------------------"	

	
	set ::ParseFEP::variance [expr $::ParseFEP::variance + $instant_variance ]
	set instant_error [expr $instant_variance / $sampling ]
	set ::ParseFEP::square_error [expr $::ParseFEP::square_error + $instant_error ]
	set instant_error [expr sqrt ( $instant_error ) ]
	set ::ParseFEP::error  [expr sqrt ( $::ParseFEP::square_error ) ]

	puts [format "$fororback:                     \u0394\u03c3**2                                 = %7.4f"   $instant_variance ]
	puts [format "$fororback:                     \u03c3**2                                  = %7.4f"    $::ParseFEP::variance ]
	puts [format "$fororback:                     \u0394\u03b4\u03b5                                   = %7.4f"   $instant_error ]
	puts [format "$fororback:                     \u03b4\u03b5                                    = %7.4f"    $::ParseFEP::error  ]

	puts "$fororback:                     -----------------------------------------------------------"

#  GAUSSIAN CASE
	if { $::ParseFEP::gaussian_guess } {

		set instant_variance_gauss   [expr  exp(2 * ($instant_fep_delta_a  - $average_f) / $kt ) * (1.0 + 2.0 * $fluctuat_gauss / $kt ** 2.0) ]
		set instant_variance_gauss   [expr ($instant_variance_gauss - 1.0) * $kt ** 2.0 ]
		set ::ParseFEP::variance_gauss 		 [expr $::ParseFEP::variance_gauss + $instant_variance_gauss ]
		set instant_error_gauss  	   [expr $instant_variance_gauss / $sampling ]

		set ::ParseFEP::square_error_gauss 	 [expr $::ParseFEP::square_error_gauss + $instant_error_gauss ]
       
		set instant_error_gauss [expr sqrt(abs($instant_error_gauss))]

	        set ::ParseFEP::error_gauss  [expr sqrt(abs($::ParseFEP::square_error_gauss)) ]

		puts [format  "$fororback:                     \u0394\u03c3**2(2nd-order)                      = %7.4f"  $instant_variance_gauss ]
		puts [format  "$fororback:                     \u03c3**2(2nd-order)                       = %7.4f"  $::ParseFEP::variance_gauss  ]
		puts [format  "$fororback:                     \u0394\u03b4\u03b5(2nd-order)                        = %7.4f" $instant_error_gauss  ]
		puts [format  "$fororback:                     \u03b4\u03b5(2nd-order)                         = %7.4f"  $::ParseFEP::error_gauss   ]
		puts "$fororback:                     -----------------------------------------------------------"

	}	

#######################################################################
#  PRINT FREE-ENERGY DIFFERENCES
#######################################################################
	
	set lambda 	   [lindex [lindex $::ParseFEP::file_lambda [expr $window -1 ] ]  7  ]
	set lambda_dlambda [lindex [lindex $::ParseFEP::file_lambda [expr  $window  -1 ] ] 8 ]

	set temp [open ParseFEP.log "a+"]

	if { $fororback == "forward" } {

		puts $temp  [format "forward:            %9.4f           %9.4f           %9.4f           %9.4f  " $lambda_dlambda    $instant_fep_delta_a      $::ParseFEP::fep_delta_a_forward  $::ParseFEP::error     ]
		lappend ::ParseFEP::fwdlog " $instant_fep_delta_a $::ParseFEP::fep_delta_a_forward  $sampling2 $average_c $fluctuat2 $instant_error $::ParseFEP::error  $lambda $lambda_dlambda"
	} else {
		puts $temp  [format "backward:           %9.4f           %9.4f           %9.4f           %9.4f  " $lambda_dlambda    $instant_fep_delta_a      $::ParseFEP::fep_delta_a_backward $::ParseFEP::error     ]
		lappend ::ParseFEP::bwdlog "$instant_fep_delta_a $::ParseFEP::fep_delta_a_backward  $sampling2 $average_c $fluctuat2 $instant_error $::ParseFEP::error  $lambda $lambda_dlambda"
	}

	close $temp

}

proc ::ParseFEP::inaccuracy_estimation {} {

		puts "Inaccuracy estimation"
		puts "======================================================================================================"                     
		puts "             i            i+\u03b4i         P0_overlap_P1   \u03b4\u03b5(forward)/\u03b5   \u03b4\u03b5(backward)/\u03b5 "            
		puts "======================================================================================================"   

	        set temp_log [open ParseFEP.log "a+"]
                puts $temp_log "Inaccuracy estimation"
                puts $temp_log "======================================================================================================"
                puts $temp_log "                   i        i+\u03b4i          P0_overlap_P1   \u03b4\u03b5(forward)/\u03b5   \u03b4\u03b5(backward)/\u03b5"          
                puts $temp_log "======================================================================================================"



		foreach  elem_fwd  $::ParseFEP::fwdlog elem_bwd $::ParseFEP::bwdlog {

############################################################
		set infile   [open $::ParseFEP::fepofile "r"]
		set infile2 [open $::ParseFEP::fepbofile "r"]
		foreach { , , , , , , , i ideltai} $elem_fwd { break } 

## just reading the data_forward 	
		set recording_index 0		
		set data_forward ""

		while { [gets $infile line] >= 0 } {
			if { [regexp "#NEW FEP WINDOW" $line ] && [regexp $i $line ] && [regexp $ideltai $line ] } { set recording_index 1}
			if { [regexp "#STARTING COLLECTION" $line ]  &&  $recording_index == 1} { set recording_index 2}
			if {  $recording_index == 2 } {
				if { [regexp "#Free energy change " $line ] } { break}
				if { $recording_index  && [ regexp  "FepEnergy: " $line ] } {
					foreach {, , , , , ,  temp } $line {break}
					lappend data_forward  $temp
				}
			}
		}

## just reading the data_backward 	
		set recording_index 0	
		set data_backward ""

		while { [gets $infile2 line] >= 0 } {
			if { [regexp "#NEW FEP WINDOW" $line ] && [regexp $i $line ] && [regexp $ideltai $line ] } { set recording_index 1 }
			if { [regexp "#STARTING COLLECTION" $line ]  &&  $recording_index == 1 } { set recording_index 2 }
			if {  $recording_index == 2 } {
				if { [regexp "#Free energy change " $line ] } { break}
				if { $recording_index  && [ regexp  "FepEnergy: " $line ] } {
					foreach {, , , , , ,  temp } $line {break}
					lappend data_backward  [ expr -1.0 * $temp ]
				}
			}
		}

		close $infile2
		close $infile

## displaying the histrogram

			set  nb_data  [llength $data_forward]
			set  nb_bin   [expr  { int ( sqrt( $nb_data ) ) } ]

                                set min1 999999999999; set max1 -999999999999;
				set min2 999999999999; set max2 -999999999999;
                                foreach elem1 $data_forward elem2 $data_backward {
                                                if {  $elem1 < $min1   } {  set min1 $elem1 }
                                                if {  $elem1 > $max1  } {  set max1 $elem1 }
                                                if {  $elem2 < $min2   } {  set min2 $elem2 }
                                                if {  $elem2 > $max2  } {  set max2 $elem2 }
                                }
		
				if { $min1 < $min2    } { set min $min1 } else { set min $min2 }
				if { $max1  > $max2 } { set max $max1 } else { set max $max2 }

                                set min   [ expr { $min - 0.1 * sqrt(sqrt( $min ** 2 )) }  ]
                                set max   [ expr { $max + 0.1 * sqrt(sqrt( $max ** 2  )) } ]
                                set delta [ expr { ( $max- $min ) / $nb_bin }        ]

                                set data_list_forward   ""
				set data_list_backward  ""

                                for { set j 0  } { $j <= $nb_bin } { incr j } {
                                        lappend data_list_forward   0
					lappend data_list_backward  0
                                }

                                set sum1 0
                                foreach elem2 $data_forward {
                                        set  index     [expr { int (( $elem2  - $min ) / $delta    )  } ]
                                        set  temp      [lindex   $data_list_forward  $index ]
					incr temp 
                                        set  data_list_forward [lreplace $data_list_forward  $index $index  $temp ]
                                        incr sum1
                                }

                                set sum2 0

                                foreach elem2 $data_backward {
                                        set  index     [expr { int (( $elem2  - $min ) / $delta    )  } ]
                                        set  temp      [lindex   $data_list_backward  $index ]
					incr temp
                                        set  data_list_backward [lreplace $data_list_backward  $index $index  $temp ]
                                        incr sum2
                                }

				set data_combine ""
                                set temp_forward   ""
				set temp_backward  ""

                                foreach elem1 $data_list_forward   elem2 $data_list_backward {
	                                lappend  temp_forward      [expr { ( $elem1 * 1.0 ) / ( $sum1 * 1.0 )  }]
                                        lappend  temp_backward  [expr { ( $elem2 * 1.0 ) / ( $sum2 * 1.0 )  }]
				}

				set data_list_forward     $temp_forward 
				set data_list_backward $temp_backward 

                                foreach elem1 $data_list_forward   elem2 $data_list_backward {
					set p_now 0.
					if {            $elem1  > $elem2 } {
						set p_now $elem1
					} elseif { $elem1  < $elem2 } {
						set p_now $elem2
					} else  { 
						set p_now $elem1
					}
					lappend data_combine $p_now
				}

## calculate the overlap degree.
				set accum_sum 0.0
				foreach elem $data_combine {
					set accum_sum [expr $accum_sum + $elem ]
				}

				set overlap [ expr 100.0 * (2.0 - $accum_sum) ]
###############################################################
# calculation the relative inaccuracy.

				foreach  {accum_f       accum_f_old        accum_g       accum_g_old} { 0.0  0.0  0.0  0.0} { }
				foreach  {min_f             max_g                  index                                       } { 0 0 0 }                    { }

				foreach  elem1  $data_list_forward  elem2 $data_list_backward {
					set accum_f_old  $accum_f
					set accum_g_old $accum_g
					set accum_f    [expr  $accum_f + $elem1 ]
					set accum_g  [expr   $accum_g + $elem2 ]
				        if {  $accum_f   > 0.01  && $accum_f_old < 0.01 }  {          set min_f    $index  } 			
					if {  $accum_g  > 0.99  && $accum_g_old < 0.99 }  {          set max_g  $index  }
					incr index
				}

				foreach  { index                 f_rela_accu                g_rela_accu }                 {0    0.0  0.0}          { }
				foreach  elem1  $data_list_forward  elem2 $data_list_backward {
					if {$index < $min_f } { set f_rela_accu [expr $f_rela_accu + $elem2 *  ( $min  + $index * $delta )]   }
					incr index		
					if {$index > $max_g } { set g_rela_accu [expr $g_rela_accu + $elem1 * ( $min + $index * $delta ) ]  }			
				}

###############################################################

				puts  [format "inaccuracy:  %7.5f        %7.5f         %7.2f         %7.2f         %7.2f"  $i   $ideltai    $overlap   $f_rela_accu   $g_rela_accu  ]
				puts $temp_log [format "inaccuracy:  %7.5f        %7.5f         %7.2f         %7.2f         %7.2f"  $i   $ideltai    $overlap   $f_rela_accu   $g_rela_accu  ]
		}
		puts "======================================================================================================"   
		close $temp_log
}

proc ::ParseFEP::sos_estimate {} {
		
#######################################################################
#  SIMPLE OVERLAP SAMPLING 
#######################################################################

	puts "ParseFEP: Simple overlap sampling on $::ParseFEP::fepofile and $::ParseFEP::fepbofile runs"
	puts "the free energy estimated by SOS(simple overlap sampling )"
	puts "========================================================================================================================================================="                     
	puts  "                        forward                                                 backward                                             SOS"               
	puts  "--------------------------------------------------------------    ------------------------------------------------------   ------------------------------------"                     
	puts  "         \u0394\u0394A     \u0394A  n/1+2\u03ba <exp(-\u0394U/2kT)> \u03c3**2    \u0394\u03b4e     \u03b4e      \u0394\u0394A    \u0394A    n/1+2\u03ba <exp(-\u0394U/2kT)> \u03c3**2    \u0394\u03b4e     \u03b4e      \u0394\u0394A     \u0394A     \u0394\u03b4e     \u03b4e "  
	puts  "========================================================================================================================================================="      
          

	set temp [open ParseFEP.log "a+"]

	puts $temp "ParseFEP: Simple overlap sampling on $::ParseFEP::fepofile and $::ParseFEP::fepbofile runs"
	puts $temp "the free energy estimated by SOS(simple overlap sampling )"
	puts $temp "======================================================================================================================================================================================"                     
	puts $temp "                              forward                                                                   backward                                                           SOS               "               
	puts $temp "-------------------------------------------    -------------------------------------------    ---------------------------------------"                     
	puts $temp "            \u0394\u0394A        \u0394A    n/1+2\u03ba   <exp(-\u0394/2kT)> \u03c3**2     \u0394\u03b4e        \u03b4e       \u0394\u0394A        \u0394A      n/1+2\u03ba   <exp(-\u0394U/2kT)>  \u03c3**2     \u0394\u03b4e        \u03b4e       \u0394\u0394A        \u0394A        \u0394\u03b4e        \u03b4e "    
	puts $temp "====================================================================================================================================================================================="      
          
	set A 0. 
	set E 0.

######################################
# reverse the order of bwdlog 
######################################
	set ::ParseFEP::bwdlog  [lreverse $::ParseFEP::bwdlog ] 

	set i_bwd_accum 0.
	set n_bwd_accum 0. 	

	foreach  elem_fwd  $::ParseFEP::fwdlog elem_bwd $::ParseFEP::bwdlog   {

		foreach { i_fwd j_fwd k_fwd l_fwd m_fwd n_fwd o_fwd  i ideltai} $elem_fwd {i_bwd j_bwd k_bwd l_bwd m_bwd n_bwd o_bwd} $elem_bwd { break }
		
		set dA [expr { -1. * $::ParseFEP::kT * log( $l_fwd *1.0 / $l_bwd ) } ]		
		set A [expr {$A + $dA}]
		
		set dV1 [expr {$::ParseFEP::kT ** 2.0 / $l_fwd ** 2.0 * $m_fwd}]
		set dV2 [expr { $::ParseFEP::kT ** 2.0 / $l_bwd ** 2.0 * $m_bwd }]

		set dE1 [expr {$dV1 / $k_fwd}]
		set dE2 [expr {$dV2 / $k_bwd }]
		set dE [expr { $dE1 + $dE2}]
		set E   [expr {$E + $dE} ]
		set sigma_dE [expr sqrt($dE) ]
		set sigma_E [expr sqrt($E)]

		set i_bwd [expr -1. * $i_bwd]
	        set i_bwd_accum [expr $i_bwd_accum + $i_bwd + 0.0       ] ; set j_bwd $i_bwd_accum	
		set n_bwd_accum [expr $n_bwd_accum + 1.0 * $n_bwd ** 2  ] ; set o_bwd [expr sqrt($n_bwd_accum)]

		puts  [format "SOS: %7.2f %7.2f %9.3f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %9.3f %7.2f %7.2f %7.2f %7.2f %7.2f %7.3f %7.4f %7.4f"  $i_fwd $j_fwd $k_fwd $l_fwd $m_fwd $n_fwd $o_fwd $i_bwd $j_bwd $k_bwd $l_bwd $m_bwd $n_bwd $o_bwd $dA $A $sigma_dE $sigma_E]

		puts $temp [format   "SOS: %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"   $i_fwd $j_fwd $k_fwd $l_fwd $m_fwd $n_fwd $o_fwd $i_bwd $j_bwd $k_bwd $l_bwd $m_bwd $n_bwd $o_bwd $dA $A $sigma_dE $sigma_E ]
			
	}

	puts  $temp  "====================================================================================================================="
	puts  "========================================================================================================================================================="
	puts "SOS-estimator: total free energy change is $A , total error is $sigma_E"
	puts  $temp "SOS-estimator: total free energy change is $A , total error is $sigma_E"

	close $temp
	
#######################################################################
#  END OF SIMPLE OVERLAP SAMPLING 
#######################################################################

}

proc ::ParseFEP::bar_estimate {} {

	set A ""
	set A_now 0.
	set sigma_2_now 0.

	set temp_log [open ParseFEP.log "a+"]
	puts $temp_log "==================================================================================================================================="
	puts $temp_log "            	   BAR        "
	puts $temp_log "-----------------------------------------------------------------------------------------------------------------------------------"
	puts $temp_log "BAR-estimator:         \u03bb             \u03bb+\u03b4\u03bb          \u0394\u0394A             \u0394A            C               \u0394\u03b4e            \u03b4e       "
	puts $temp_log "                          "
	puts $temp_log "-----------------------------------------------------------------------------------------------------------------------------------"

	puts "  "
	puts "  "
	puts "the free energy estimated by BAR (Bennett acceptance ratio )"
	puts "=========================================================================================================================================="                     

	puts "BAR-estimator:        \u03bb             \u03bb+\u03b4\u03bb           \u0394\u0394A            \u0394A              C             \u0394\u03b4e            \u03b4e      "
	puts  "=========================================================================================================================================="      
          

	foreach elem_fwd  $::ParseFEP::fwdlog   elem_bwd $::ParseFEP::bwdlog {

		set infile [open $::ParseFEP::fepofile "r"]
		set infile2 [open $::ParseFEP::fepbofile "r"]
		foreach { , , , , , , , i ideltai} $elem_fwd { break } 

## just reading the data_forward 	
		set recording_index 0	
	
		set data_forward ""

		while { [gets $infile line] >= 0 } {
			if { [regexp "#NEW FEP WINDOW" $line ] && [regexp $i $line ] && [regexp $ideltai $line ] } { set recording_index 1}
			if { [regexp "#STARTING COLLECTION" $line ]  &&  $recording_index == 1} { set recording_index 2}
			if {  $recording_index == 2 } {
				if { [regexp "#Free energy change " $line ] } { break}
				if { $recording_index  && [ regexp  "FepEnergy: " $line ] } {
					foreach {, , , , , ,  temp } $line {break}
					lappend data_forward  $temp
				}
			}
		}

## just reading the data_backward 	
		set recording_index 0	

		set data_backward ""

		while { [gets $infile2 line] >= 0 } {
			if { [regexp "#NEW FEP WINDOW" $line ] && [regexp $i $line ] && [regexp $ideltai $line ] } { set recording_index 1 }
			if { [regexp "#STARTING COLLECTION" $line ]  &&  $recording_index == 1 } { set recording_index 2 }
			if {  $recording_index == 2 } {
				if { [regexp "#Free energy change " $line ] } { break}
				if { $recording_index  && [ regexp  "FepEnergy: " $line ] } {
					foreach {, , , , , ,  temp } $line {break}
					lappend data_backward  [ expr -1.0 * $temp ]
				}
			}
		}

		close $infile2
		close $infile

## data has already been extracted from the fepout file. the next is to performe the BAR estimate.

		set len_forward  [llength $data_forward] 
		set len_backward [llength $data_backward] 

		set instant_accum 0. 

		foreach { i_fwd j_fwd k_fwd l_fwd } $elem_fwd {i_bwd j_bwd k_bwd l_bwd } $elem_bwd { break }

		set deltaA_0  [expr   -1. * $::ParseFEP::kT  * log ( ($l_fwd * 1.0) / (1.0 * $l_bwd)  )  ]		

		set C_0  [ ::ParseFEP::deltaAtoC $deltaA_0 $len_forward $len_backward ]	
	
		set deltaA_1 [ ::ParseFEP::CtodeltaA $C_0 $data_forward $data_backward  $len_forward $len_backward ]
		set C_1 [ ::ParseFEP::deltaAtoC $deltaA_1 $len_forward $len_backward ]

		while {  abs($deltaA_1 - $deltaA_0 ) > 0.00045   } {
		
			set deltaA_0 $deltaA_1
			set C_0 $C_1
				
			set deltaA_1 [ ::ParseFEP::CtodeltaA $C_0 $data_forward $data_backward  $len_forward $len_backward ]
			set C_1 [ ::ParseFEP::deltaAtoC $deltaA_1 $len_forward $len_backward ]
		}
	
		set sigma_2_window_now  [ ::ParseFEP::sigma_2_BAR $data_forward $data_backward  $len_forward $len_backward $C_1  $k_fwd $k_bwd  ]
		set sigma_2_now [expr $sigma_2_now + $sigma_2_window_now  ]

		lappend A $deltaA_1
		set A_now [expr $A_now + $deltaA_1]

		set Dde [expr sqrt($sigma_2_window_now) ]
		set de  [expr sqrt($sigma_2_now) ]  

		puts  [format   "BAR-estimator:    %9.5f      %9.5f     %9.4f      %9.4f      %9.4f      %9.4f      %9.4f "  $i    $ideltai   $deltaA_1 $A_now  $C_1  $Dde $de  ]
		puts $temp_log  [format   "BAR-estimator:    %9.5f      %9.5f     %9.4f      %9.4f      %9.4f      %9.4f      %9.4f "  $i    $ideltai   $deltaA_1 $A_now  $C_1  $Dde $de ]

	}

		puts  "=========================================================================================================================================="      
       		puts "BAR-estimator: total free energy change is $A_now , total error is $de"
		puts  $temp_log "BAR-estimator: total free energy change is $A_now , total error is $de"
		close $temp_log
}


proc  ::ParseFEP::deltaAtoC {deltaA length_forward length_backward } {

	set C [expr  { $deltaA + $::ParseFEP::kT * log (  $length_backward *1.0 /  $length_forward    ) } ]	
	return $C
}

proc  ::ParseFEP::CtodeltaA {C data_list_forward data_list_backward length_forward length_backward } {
	
	set accum_for 0.
	set accum_back 0.

	foreach elem_for $data_list_forward {
		set accum_for [expr {  $accum_for + 1/(1+exp( 1/$::ParseFEP::kT*(  $elem_for - $C ) ) )  }  ]
	}

	foreach elem_back $data_list_backward {
		set accum_back [expr  { $accum_back + 1/(1+exp(-1/$::ParseFEP::kT*(   $elem_back - $C ) ) ) }  ]
	}	

	set deltaA [expr {  $::ParseFEP::kT * log( 1.0* $accum_back /   $accum_for ) + $C  - $::ParseFEP::kT * log(  $length_backward *1.0 / $length_forward )  } ]

	return $deltaA
}

proc  ::ParseFEP::sigma_2_BAR {data_list_forward data_list_backward length_forward length_backward C  N0_kappa N1_kappa } {
	set accum_first_0 		0.
	set accum_second_0 	0.
	set accum_first_1 		0.
	set accum_second_1 	0.

	foreach elem_back $data_list_backward {
		
		set temp [expr {1/(1+exp( -1. /$::ParseFEP::kT * ( $elem_back - $C) ) ) } ]
		set accum_first_1 [ expr  { $accum_first_1 +  $temp }  ]
		set accum_second_1  [ expr  { $accum_second_1 +  $temp  ** 2 } ]

	}

	foreach elem_for $data_list_forward {

		set temp  [ expr {1/(1+exp( 1/$::ParseFEP::kT * ($elem_for - $C) ) )}  ]
		set accum_first_0 [ expr  { $accum_first_0 +  $temp  }  ]
		set accum_second_0  [ expr {  $accum_second_0 + $temp ** 2  } ]

	}
	
	set mean_second_0  [expr { $accum_second_0 /  $length_forward} ]
	set mean_first_0  [expr { $accum_first_0 /  $length_forward} ]
	set mean_second_1  [expr { $accum_second_1 /  $length_backward} ]
	set mean_first_1  [expr { $accum_first_1 /  $length_backward} ]

	set sigma_2 [expr { ( $::ParseFEP::kT * $::ParseFEP::kT / $N0_kappa ) * (  $mean_second_0 /  $mean_first_0 ** 2  - 1 ) \
				+ ( $::ParseFEP::kT * $::ParseFEP::kT  / $N1_kappa ) * (  $mean_second_1 /  $mean_first_1 ** 2  - 1 ) } ]
	return $sigma_2
}

########################################################
#### \\  //  ||\     //|   
####  \\//   ||\\   //||   
####  //\\   || \\ // ||   
#### //  \\  ||  \\/  || grace
########################################################


proc ::ParseFEP::fepdisp_unix { } {

## clear the old files

	::ParseFEP::Clear

###########################################
# handle the forward output file 
###########################################
	set num_indicator 0;  	set window  0

	set file_entropy "" ; 	set file ""	

	set infile  [open $::ParseFEP::fepofile  "r"]
	set data_file [split [read   $infile [file size "$::ParseFEP::fepofile"] ] "\n" ] 
	close $infile	

	set fep_delta_a 0.0; set fep_delta_u 0.0; set fep_delta_s 0.0

	set nb_data [expr { int ( $::ParseFEP::nb_sample - $::ParseFEP::nb_equil ) } ] 

	set  nb_bin  [expr  { int ( sqrt( $nb_data ) ) } ]

	set first_step 0.

##############################################
#  start reading the file 
##############################################
	foreach line $data_file  {

		if { [ regexp  FepEnergy $line ] } {

			incr  num_indicator  
			if {  [expr   $num_indicator ==  1 ] } {
				foreach { , temp_step } $line { break } 
				set first_step $temp_step 
			}
			 
#########################################################
# handle the data in every window
#########################################################
			if {   $num_indicator > $::ParseFEP::nb_equil   } {

				foreach { , temp1 , , , , temp6 , , temp9 } $line {break}
				set temp1 [ expr { ( $temp1 - $first_step - $::ParseFEP::nb_equil * $::ParseFEP::FEPfreq ) * 1.0  } ]
					lappend file  " $temp1  $temp6  $temp9"	

				foreach {  , temp1  temp2  temp3 temp4 temp5 } $line {break} 
				set temp1 [ expr { $temp1 - $first_step - $::ParseFEP::nb_equil * $::ParseFEP::FEPfreq } ]
					lappend file_entropy  " $temp1  $temp2  $temp3  $temp4  $temp5 "
			} 
		}
		
			if {   [regexp  "#Free energy change for lambda"  $line ] } {

				incr window	
			################################################################
			#	store the data for the disp
			################################################################

				set temp_file [ open [format "file%d.dat" $window ] "w"]							
		
				foreach data $file  {  puts $temp_file "$data" }

				close $temp_file
			################################################################
			# DISPLAY DENSITY OF STATES
			################################################################
				
				set min 999999999999; set max -999999999999;
				foreach elem $file {
					foreach {, elem2 }  $elem { break } 
						if {  $elem2 < $min  } {  set min $elem2 } 
						if {  $elem2 > $max  } {  set max $elem2 }
				}
			
				set min   [ expr { $min - 0.1 * sqrt( $min ** 2 ) }  ]
				set max   [ expr { $max + 0.1 * sqrt( $max ** 2 ) }  ] 
				set delta [ expr { ( $max- $min ) / $nb_bin }        ] 		
			
				set data_list ""
				for { set i 0  } { $i <= $nb_bin } { incr i } {
					lappend data_list 0
				}

				set sum 0
				foreach elem $file {
					foreach {, elem2 }  $elem { break } 
						set  index     [expr { int (( $elem2  - $min ) / $delta    )  } ]
						set  temp      [lindex   $data_list  $index ] 
						incr temp 
						set  data_list [lreplace $data_list  $index $index  $temp ] 
					incr sum 
				}			
	
				set sum [expr { $sum * $delta * 1.0 } ] 

				set temp_file [ open [format "file%d.dat.hist" $window ] "w"]							

				set j 0
				foreach elem $data_list  { 
					incr j 
					set i [expr {  ( $j + 0.5 )* $delta + $min } ]
						puts  $temp_file [format "%15.4f %15.4f %25.4f %25.4f"  $i  [expr { $elem / $sum * 1.0  }] [expr { exp ( - 1.987* $::ParseFEP::temperature /1000 * $i)  } ] [expr { $elem / $sum * exp ( -  1.987* $::ParseFEP::temperature /1000* $i)} ] ]
				}			

				close $temp_file

			#############################################################    
			# compute the entropy for every window in forward output file
			#############################################################
				if { $::ParseFEP::entropyindex == 1 } { 
					set energy_difference ""
					foreach elem $file_entropy {
						foreach { , elem2 elem3 elem4 elem5 } $elem {break }
						lappend energy_difference "[expr { $elem2 + $elem4 } ] [expr  { $elem3 + $elem5 } ]"					
					}			
				
			############################################################
			# record the file.entropy 
			############################################################

					set temp_entropy_file [ open [format "file%d.dat.entropy" $window ] "w"]	
					foreach {s1 s2 s3 n } { 0. 0. 0. 0} {break }
					foreach elem $energy_difference {
						foreach { elem1 elem2 }  $elem { break } 
							incr n  
							set s1 [expr  {$s1 * ($n - 1 ) * 1.0 / $n  + exp ( -1 * (  $elem2 - $elem1 ) / $::ParseFEP::kT ) * $elem2 * 1.0 / $n } ] 
							set s2 [expr  {$s2 * ($n - 1 ) * 1.0 / $n  + exp ( -1 * ($elem2 - $elem1 ) / $::ParseFEP::kT ) }  *1.0 /  $n] 
							set s3 [expr  {$s3 * ($n - 1 ) * 1.0 / $n  + $elem1 * 1.0 / $n } ]

						set instant_fep_delta_u [expr { $s1 /  $s2 - $s3  } ]	
						set instant_fep_delta_s [expr {  $::ParseFEP::kT* log ($s2 * 1.0 ) + $s1 / $s2 - $s3 } ]

						puts $temp_entropy_file [format "%10d  %12.6f  %15.6f  %12.6f  %15.6f  %12.6f  %12.6f  %12.6f  " [expr { $::ParseFEP::FEPfreq * ( $n -1) } ]  [expr { $elem2 - $elem1} ]  $s1  $s2   $s3  [ expr { -1. * $::ParseFEP::kT*log($s2 * 1.0 / $n)} ]  $instant_fep_delta_u  $instant_fep_delta_s ]

					}

					close $temp_entropy_file	

					set fep_delta_u [ expr  { $fep_delta_u + $instant_fep_delta_u} ]

					set fep_delta_s [expr { $fep_delta_s + $instant_fep_delta_s } ] 							

				}

			#############################################################
			# initile the parameter for next window
			#############################################################
				set file ""
				set file_entropy ""
				set num_indicator 0
######
			}
	}

	
###########################################
# handle the backward output file 
###########################################
	
	if {  $::ParseFEP::fepbofile != "" } {
		set num_indicator 0 ; 	set window  0

		set file_entropy "" ;	set file ""	

		set infile  [open $::ParseFEP::fepbofile  "r"]
		set data_file [split [read  $infile  [file size "$::ParseFEP::fepbofile"] ] "\n" ] 
		close $infile 

		#set rev_lambda ""
		#set rev_G ""

		foreach line $data_file  {
	
			if { [ regexp  "FepEnergy" $line ] } {
	
				incr  num_indicator  

				if {  [expr   $num_indicator == 1 ] } {
					foreach { , temp_step } $line { break } 
					set first_step $temp_step 
				}
	
				if { [expr   $num_indicator > $::ParseFEP::nb_equil  ]} {
 	
					foreach { , temp1 , , , , temp6 , , temp9 } $line {break}
					set temp6 [expr { -1.0 * $temp6 } ]; 	set temp9 [expr { -1.0 * $temp9 } ]
					set temp1 [ expr ( $temp1 - $first_step - $::ParseFEP::nb_equil * $::ParseFEP::FEPfreq ) * 1.0  ]
					lappend file   " $temp1  $temp6  $temp9 "
	
					foreach {  , temp1  temp2  temp3 temp4 temp5 } $line {break} 
   					set temp1 [ expr $temp1 - $first_step - $::ParseFEP::nb_equil * $::ParseFEP::FEPfreq ]
					lappend file_entropy " $temp1  $temp2  $temp3  $temp4 $temp5"
				} 
		}
		
			if {   [regexp  "#Free energy change for lambda"  $line ] } {
					incr window	
################################################################
#	store the data for the disp
################################################################

					set index [expr { $::ParseFEP::nb_file +1 - $window } ]
					set temp_file [open [format "file%d.dat.rev" $index ]  "w"]
	
					foreach data $file  { puts $temp_file "$data" }
					close $temp_file

			################################################################
			# DISPLAY DENSITY OF STATES
			################################################################
				
					set min 999999999999; set max -999999999999;
					foreach elem $file {
						foreach {, elem2 }  $elem { break } 
							if {  $elem2 < $min  } {  set min $elem2 } 
							if {  $elem2 > $max  } {  set max $elem2 }
					}
				
					set min [expr { $min - 0.1 * sqrt( $min ** 2 ) } ]
					set max [expr { $max + 0.1 * sqrt( $max ** 2  ) } ] 
					set delta [ expr { ( $max- $min ) / $nb_bin } ] 		
				
					set data_list ""
					for { set i 0  } { $i <= $nb_bin } { incr i } {
						lappend data_list 0
					}
	
					set sum 0
					foreach elem $file {
						foreach {, elem2 }  $elem { break } 
							set index [expr { int (( $elem2  - $min ) / $delta   )  } ] 
							set temp [lindex $data_list $index ] ;	incr temp 
							set data_list [lreplace $data_list  $index $index  $temp ] 
						incr sum 
					}					

					set sum [expr { $sum * $delta * 1.0 } ] 
					set j 0

					set index [expr { $::ParseFEP::nb_file +1 - $window } ]
					set temp_file [ open [format "file%d.dat.rev.hist" $index ] "w"]							

					foreach elem $data_list  { 
						incr j 
						set i [expr {  ( $j + 0.5 ) * $delta + $min } ]
						if { [expr   $elem / $sum * 1.0  > 0.00049   ] } {
							puts  $temp_file [format "%15.4f %15.4f %25.4f %25.4f" $i  [expr { $elem / $sum * 1.0  }] [expr { exp (  - 1.987* $::ParseFEP::temperature /1000 * $i)  } ] [expr { $elem / $sum * exp (  - 1.987* $::ParseFEP::temperature /1000 * $i)} ] ]
						}
					}			

					close $temp_file

					#if {$::ParseFEP::entropyindex == 1 } { 
					#	set temp_entropy_file [ open [format "forwardentropytemp%d.dat" $index ] "w"]	
					#	foreach  data_entropy $file_entropy  { puts $temp_entropy_file "$data_entropy" }
					#	close $temp_entropy_file
					#}
			#############################################################
			# initile the parameter for next window
			#############################################################
					set file ""
					set file_entropy ""
					set num_indicator 0
			
				}
		}
	}

################################################################
# XmGrace VISUALIZATION
################################################################

	set nb_data [expr  $::ParseFEP::nb_sample - $::ParseFEP::nb_equil ]
	set minor [expr  int ($nb_data * $::ParseFEP::FEPfreq / 8  ) ]
	set major [expr $minor *4 ]

#######################################################################
# NUMBER OF PAGES
#######################################################################
	set nb_page [ expr 1 + ($::ParseFEP::nb_file - 1) /16 ]  

	puts [format  "ParseFEP: Parsing data into %4d XmGrace sheets"  $nb_page ] 

#######################################################################
# FREE-ENERGY TIME SERIES
#######################################################################

	puts "ParseFEP: Free-energy time series" 
	
	set page 1 ; set line 1 ; set marker 0 

	set index 0 

	while { $page <= $nb_page } { 

		file delete [format "grace.%d.exec" $page ] 

		set temp [open [format "grace.%d.exec" $page ] "a+"]  
                puts $temp {     xmgrace -pexec "arrange (4,4,0.10,0.40,0.20,OFF,OFF,OFF)"  \
                                                         -pexec "with string ;
                                                           string on ;
                                                           string g0 ;
                                                           string loctype view ;
                                                           string 0.030, 0.950 ;
                                                           string char size 1.6 ; }
		puts $temp [format  "string def \\\"\\\\f{Helvetica-Bold}ParseFEP\\\\f{Helvetica}: Free energy sheet %d \\\" \" \\" $page  ]


#######################################################################
# FOR SOS ANALYSIS, DISPLAY FORWARD AND BACKWARD TIME SERIES
#######################################################################
	
		if { $::ParseFEP::fepbofile != "" } {

			while { $index < 16 * $page     } {
				incr marker
				set file_index [expr $index + 1 ]
				set index_now [expr $index  %16 ]

				if { [file exists [format "file%d.dat" $file_index] ] }  {
                                        set lambda1 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 7 ]]
                                        set lambda2 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 8 ]]
					set posx    [expr {0.100 +  int ( ($index%16) % 4 ) * 0.295 }]
					set posy    [expr {0.910 -  int ( ($index%16) / 4 ) * 0.210 }]
		
					if { $index_now == 7 } {
						puts $temp [format "      -graph  %d  -block %s -bxy 1:3 -block %s -bxy 1:3 -pexec \"xaxis ticklabel off ; " $index_now [format "file%d.dat" $file_index ]  [format "file%d.dat.rev" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {                                                                        xaxis ticklabel format decimal ; 
                                                                        yaxis label char size 1.8 ; 
                                                                        yaxis label place spec ; 
                                                                        yaxis label place -0.10,-0.25 ; 
                                                                        yaxis label  \"\\r{180}\\f{Symbol}D\\1G\\0\\s\\1fwd\\0\\N,\\R{red}\\f{Symbol}D\\1G\\0\\s\\1rev\\0\\N\\R{black}\\f{Helvetica} (kcal/mol)\" ;
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ;   }
                    				puts $temp [format "                                                                           string g%d ; " $index_now  ]
                                                puts $temp [format "                                                                             string loctype view ;" ]
                                                puts $temp [format "                                                                             string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "                                                                             string char size 0.8 ;" ]
                                                puts $temp [format "                                                                             string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}= %6.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
					} elseif { $index_now == 12 || $index_now == 14 || $index_now == 15   }  {
						puts $temp [format "      -graph %d -block %s -bxy 1:3 -block %s -bxy 1:3 -pexec \"xaxis ticklabel on ; " $index_now [format "file%d.dat" $file_index ]  [format "file%d.dat.rev" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {                                                                        xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "                                                                           string g%d ; " $index_now  ]
                                                puts $temp [format "                                                                             string loctype view ;" ]
                                                puts $temp [format "                                                                             string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "                                                                             string char size 0.8 ;" ]
                                                puts $temp [format "                                                                             string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}= %6.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
					} elseif {  $index_now == 13 } {
		       				puts $temp [format "      -graph %d -block %s -bxy 1:3 -block %s -bxy 1:3 -pexec \"xaxis ticklabel on ; " $index_now [format "file%d.dat" $file_index ]  [format "file%d.dat.rev" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {                                                                        xaxis ticklabel format decimal ; 
                                                                        xaxis label char size 1.8 ; 
                                                                        xaxis label place spec ; 
                                                                        xaxis label place  0.15,0.07 ; 
                                                                        xaxis label \"\\f{Helvetica} MD step\" ;
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "                                                                           string g%d ; " $index_now  ]
                                                puts $temp [format "                                                                             string loctype view ;" ]
                                                puts $temp [format "                                                                             string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "                                                                             string char size 0.8 ;" ]
                                                puts $temp [format "                                                                             string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}= %6.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
       					} else {
						puts $temp [format "      -graph %d -block %s -bxy 1:3 -block %s -bxy 1:3 -pexec \"xaxis ticklabel off ; " $index_now [format "file%d.dat" $file_index ]  [format "file%d.dat.rev" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {                                                                        xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "                                                                          string g%d ; " $index_now  ]
                                                puts $temp [format "                                                                            string loctype view ;" ]
                                                puts $temp [format "                                                                            string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "                                                                            string char size 0.8 ;" ]
                                                puts $temp [format "                                                                            string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}= %6.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
    					}
				} else {
					puts $temp [format "        -pexec \"kill g%d\" \\" $index_now ]
				}

				incr index 
				incr line
			}
#################################################################
# STANDARD DISPLAY OF ONE TIME SERIES
# if backward output file is not exist. 
#################################################################
		} else {
			while { $index < 16 * $page     } {
				incr marker 
				set file_index [expr $index + 1 ]
				set index_now [expr $index  %16 ]

				if { [file exists [format "file%d.dat" $file_index] ] }  {
	                                set lambda1 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 7 ]]
                                        set lambda2 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 8 ]]
					set posx [expr {0.100 + int ( ($index%16) % 4 ) * 0.295 }]
					set posy [expr {0.910 -  int ( ($index%16) / 4 ) * 0.210 }]
		
					if { $index_now == 7 } {
						puts $temp [format "      -graph  %d  -block %s -bxy 1:3 -pexec \"xaxis ticklabel off ; " $index_now [format "file%d.dat" $file_index ]  ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {    xaxis ticklabel format decimal ; 
                                                                        yaxis label char size 1.8 ; 
                                                                        yaxis label place spec ; 
                                                                        yaxis label place -0.10,-0.25 ; 
                                                                        yaxis label  \"\\r{180}\\f{Symbol}D\\1G\\0\\f{Helvetica} (kcal/mol)\" ;
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ;   }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
					} elseif { $index_now == 12 || $index_now == 14 || $index_now == 15   } {
						puts $temp [format "      -graph %d -block %s -bxy 1:3 -pexec \"xaxis ticklabel on ; " $index_now [format "file%d.dat" $file_index ]  ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {    xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
					} elseif {  $index_now == 13 } {
		       				puts $temp [format "      -graph %d -block %s -bxy 1:3 -pexec \"xaxis ticklabel on ; " $index_now [format "file%d.dat" $file_index ]  ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {    xaxis ticklabel format decimal ; 
                                                                        xaxis label char size 1.8 ; 
                                                                        xaxis label place spec ; 
                                                                        xaxis label place  0.15,0.07 ; 
                                                                        xaxis label \"\\f{Helvetica} MD step\" ;
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
       					} else {
						puts $temp [format "      -graph %d -block %s -bxy 1:3 -pexec \"xaxis ticklabel off ; " $index_now [format "file%d.dat" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {   xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
    					}
				} else {
					puts $temp [format "        -pexec \"kill g%d\" \\" $index_now ]
				}

				incr index 
				incr line
			}			
		}
	
		puts $temp [format "        -hardcopy -printfile free-energy.%d.png -hdevice PNG"  $page ]

		close $temp 
		#exec display [format "free-energy.%d.png"  $page ]
		incr page 


	}


#######################################################################
# DISPLAY ENTHALPY AND ENTROPY
#######################################################################
	if { $::ParseFEP::entropyindex == 1 } {
		set n15_data [expr  $::ParseFEP::nb_sample - $::ParseFEP::nb_equil ]
		set minor [expr  int ($nb_data * $::ParseFEP::FEPfreq / 8 )  ]
		set major [expr $minor * 4  ]

#######################################################################
# ENTHALPY AND ENTROPY TIME SERIES
#######################################################################

		puts "ParseFEP: Enthalpy and entropy time series"
	
		set page 1 ; set line 1 ; set marker 0
		set index 0

		while { $page <= $nb_page } { 

			file delete [format "grace.entropy.%d.exec" $page ]

			set temp [open [format "grace.entropy.%d.exec" $page ] "a+"]
			puts $temp {	 xmgrace -pexec "arrange (4,4,0.10,0.40,0.20,OFF,OFF,OFF)"  \
							 -pexec "with string ;
							   string on ;
		                            		   string g0 ;
        		                    	  	   string loctype view ;
        		                    		   string 0.030, 0.950 ;
        		                    		   string char size 1.6 ; }
			puts $temp [format  "string def \\\"\\\\f{Helvetica-Bold}ParseFEP\\\\f{Helvetica}: Enthalpy and entropy sheet %d \\\" \" \\" $page  ]


	
			while { $index < 16 * $page  } {
				incr marker
				set file_index [expr $index + 1 ]
				set index_now [expr $index  %16 ]

				if { [file exists [format "file%d.dat.entropy" $file_index] ] }  {
                                        set lambda1 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 7 ]]
                                        set lambda2 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 8 ]]
					set posx [expr {0.100 + int (($index%16)% 4 ) * 0.295 }]
					set posy [expr {0.910 - int ( ($index%16) / 4 ) * 0.210 }]
		
					if { $index_now == 7 } {
						puts $temp [format "      -graph  %d  -block %s -bxy 1:7 -block %s -bxy 1:8 -pexec \" xaxis tick on;; " $index_now [format "file%d.dat.entropy" $file_index ]  [format "file%d.dat.entropy" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {    xaxis ticklabel off ;
									xaxis ticklabel format decimal ; 
                                                                        yaxis label char size 1.8 ; 
                                                                        yaxis label place spec ; 
                                                                        yaxis label place -0.10,-0.25 ; 
                                                                        yaxis label  \"\\r{180}\\f{Symbol}D\\1U\\0,\\R{red}\\1T\\0\\f{Symbol}D\\1S\\0\\R{black}\\f{Helvetica} (kcal/mol)\" ;
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ;   }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
					} elseif { $index_now == 12 || $index_now == 14 || $index_now == 15   } {
						puts $temp [format "      -graph %d -block %s -bxy 1:7 -block %s -bxy 1:8 -pexec \"xaxis tick on ; " $index_now [format "file%d.dat.entropy" $file_index ]  [format "file%d.dat.entropy" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {    xaxis ticklabel on ;
									xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
					} elseif {  $index_now == 13 } {
		       				puts $temp [format "      -graph %d -block %s -bxy 1:7 -block %s -bxy 1:8 -pexec \"xaxis tick on ; " $index_now [format "file%d.dat.entropy" $file_index ]  [format "file%d.dat.entropy" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;"  [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {    xaxis ticklabel on ;
									xaxis ticklabel format decimal ; 
                                                                        xaxis label char size 1.8 ; 
                                                                        xaxis label place spec ; 
                                                                        xaxis label place  0.15,0.07 ; 
                                                                        xaxis label \"\\f{Helvetica} MD step\" ;
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
       					} else {
						puts $temp [format "      -graph %d -block %s -bxy 1:7 -block %s -bxy 1:8 -pexec \"xaxis tick on ; " $index_now [format "file%d.dat.entropy" $file_index ]  [format "file%d.dat.entropy" $file_index ] ]
#						puts $temp [format "                                                                        world xmax %d ;" [expr $::ParseFEP::FEPfreq* $nb_data]  ]
                                                puts $temp {    xaxis ticklabel off ;
									xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 0 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
                                                                        yaxis ticklabel char size 0.8 ;
                                                                        with string ;
                                                                             string on ; }
                    				puts $temp [format "          string g%d ; " $index_now  ]
                                                puts $temp [format "          string loctype view ;" ]
                                                puts $temp [format "          string %8.4f, %8.4f ;" $posx $posy ]
                                                puts $temp [format "          string char size 0.8 ;" ]
                                                puts $temp [format "          string def \\\"\\\\f{Symbol}l\\\\f{Helvetica}=%8.4f to %8.4f \\\" \" \\"  $lambda1 $lambda2 ]
    					}
				} else {
					puts $temp [format "        -pexec \"kill g%d\" \\" $index_now ]
				}
				incr index 
				incr line
			}
	
			puts $temp [format "        -hardcopy -printfile entropy.%d.png -hdevice PNG"  $page ]
			close $temp 
			#exec display [format "entropy.%d.png"  $page ]

			incr page 
		}
	}

#######################################################################
# DISPLAY DENSITY OF STATES
#######################################################################

	puts  "ParseFEP: Compute probability distributions"


#######################################################################
# HISTOGRAMMING      
#######################################################################


#######################################################################
# FOR SOS ANALYSIS, COMPUTE TWO PDFs
#######################################################################


	set page 1 ; set line 1 ; set marker 0
	set index 0

	while { $page <= $nb_page } { 

		file delete [format "grace.hist.%d.exec" $page ]

		set temp [open [format "grace.hist.%d.exec" $page ] "a+"]
		puts $temp {	 xmgrace -pexec "arrange (4,4,0.10,0.40,0.20,OFF,OFF,OFF)"  \
							 -pexec "with string ;
							   string on ;
		                            		   string g0 ;
        		                    	  	   string loctype view ;
        		                    		   string 0.030, 0.950 ;
        		                    		   string char size 1.6 ; }
		puts $temp [format  "string def \\\"\\\\f{Helvetica-Bold}ParseFEP\\\\f{Helvetica}: Probability distribution sheet %d \\\" \" \\" $page  ]


	
		if { $::ParseFEP::fepbofile != "" } {

			while { $index < 16 * $page   } {
				
				incr marker
				set file_index [expr $index + 1 ]
				set index_now  [expr $index  %16 ]

				if { [file exists [format "file%d.dat.hist" $file_index] ] }  {
                                        set lambda1 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 7 ]]
                                        set lambda2 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 8 ]]
					set posx [expr {0.100 + int (($index%16)% 4 ) * 0.295 }]
					set posy [expr {0.910 - int ( ($index%16) / 4 ) * 0.210 }]
		
####################################################
# decide the value of the tick spacing
####################################################

                        set infile [open [format "file%d.dat.hist"     $file_index ]  "r"]
                        set size [file size  [format "file%d.dat.rev.hist" $file_index ]  ]
                        set data_file [split [read $infile $size ] "\n" ]

                        set forwardxmin [lindex [lindex $data_file 0] 0]

                        foreach elem $data_file {
                                if   {  [lindex $elem 1] > 0.03          }  {set forwardxmin [lindex $elem 0] ; break}
                        }

			set forwardymax 0
			set forwardxmax -100.0
			foreach elem $data_file {
				if   {  [lindex $elem 1] > $forwardymax  }  {set forwardymax  [lindex $elem 1] }
				if   {  [lindex $elem 1] > 0.03          }  {set forwardxmax  [lindex $elem 0] }
			}

############################ seperating line  ######################
                        set infile [open [format "file%d.dat.rev.hist" $file_index ] "r"]
                        set size [file size  [format "file%d.dat.rev.hist" $file_index ]  ]
                        set data_file [split [read $infile $size ] "\n" ]
			
                        set backwardxmin [lindex [lindex $data_file 0] 0]
			
			foreach elem $data_file {
                                if   {  [lindex $elem 1] > 0.03          }  {set backwardxmin [lindex $elem 0] ; break}
                        }	

                        set backwardymax 0
			set backwardxmax -100.0
                        foreach elem $data_file {
                                if   {  [lindex $elem 1] > $backwardymax }  {set backwardymax [lindex $elem 1]  }
				if   {  [lindex $elem 1] > 0.03          }  {set backwardxmax [lindex $elem 0] }
                        }

			if       { $forwardxmin < $backwardxmin  && $forwardxmax < $backwardxmax  } {	
				set xlen [expr $backwardxmax - $forwardxmin ]
			} elseif { $forwardxmin > $backwardxmin  && $forwardxmax < $backwardxmax  } {
				set xlen [expr $backwardxmax - $backwardxmin]
			} elseif { $forwardxmin < $backwardxmin  && $forwardxmax > $backwardxmax  } {	
				set xlen [expr $forwardxmax  - $forwardxmin ]
			} else  {
				set xlen [expr $forwardxmax  - $backwardxmin]
			}

				set xtickmajor [expr round($xlen*1.2/0.5/2 ) *0.5]

			if {$forwardymax > $backwardymax } {	
				set ytickmajor [ expr round($forwardymax/0.1/2) * 0.1 ]
			} else {
				set ytickmajor [ expr round($backwardymax/0.1/2) * 0.1 ]
			}
################################################################

					if { $index_now == 7 } {
						puts $temp [format "      -graph  %d  -block %s -bxy 1:2 -block %s -bxy 1:2 -pexec \"xaxis tick on ; " $index_now [format "file%d.dat.hist" $file_index ]  [format "file%d.dat.rev.hist" $file_index ] ]
						puts $temp [format "    yaxis tick on;"]
                                                puts $temp [format "    yaxis tick major %8.3f ; "  $ytickmajor  ]
                                                puts $temp [format "    yaxis tick minor ticks %8d ; " 1 ]
                                                puts $temp {            xaxis ticklabel on;
                                                                        xaxis ticklabel format decimal ;
                                                                        yaxis label char size 1.8 ; 
                                                                        yaxis label place spec ; 
                                                                        yaxis label place -0.10,-0.25 ; 
                                                                        yaxis label \"\\r{180}\\1P\\0\\s\\1fwd\\0\\N\\f{Helvetica}(\\f{Symbol}D\\1U\\0\\f{Helvetica} ), \\R{red}\\1P\\0\\s\\1rev\\0\\N\\f{Helvetica}(\\f{Symbol}D\\1U\\0\\f{Helvetica} )\" ;
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
								}
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
					} elseif { $index_now == 12 || $index_now == 14 || $index_now == 15   }  {
						puts $temp [format "      -graph %d -block %s -bxy 1:2 -block %s -bxy 1:2 -pexec \"xaxis tick on ; " $index_now [format "file%d.dat.hist" $file_index ]  [format "file%d.dat.rev.hist" $file_index ] ]
                                                puts $temp [format "    yaxis tick on;"]
                                                puts $temp [format "    yaxis tick major %8.3f ; "  $ytickmajor  ]
                                                puts $temp [format "    yaxis tick minor ticks %8d ; "  1 ]
                                                puts $temp {            xaxis ticklabel on; 
                                                                        xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
									yaxis ticklabel on; 
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
 							   }
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
					} elseif {  $index_now == 13 } {
		       				puts $temp [format "      -graph %d -block %s -bxy 1:2 -block %s -bxy 1:2 -pexec \"xaxis tick on ; " $index_now [format "file%d.dat.hist" $file_index ]  [format "file%d.dat.rev.hist" $file_index ] ]
                                                puts $temp [format "    yaxis tick on;"]
                                                puts $temp [format "    yaxis tick major %8.3f ; "  $ytickmajor  ]
                                                puts $temp [format "    yaxis tick minor ticks %8d ; "  1 ]
                                                puts $temp {            xaxis ticklabel on;
                                                                        xaxis ticklabel format decimal ;
                                                                        xaxis label char size 1.8 ; 
                                                                        xaxis label place spec ; 
                                                                        xaxis label place  0.15,0.07 ; 
                                                                        xaxis label \"\\f{Symbol}D\\1U\\0\\f{Helvetica} (kcal/mol)\" ;
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
 								}
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
       					} else {
						puts $temp [format "      -graph %d -block %s -bxy 1:2 -block %s -bxy 1:2 -pexec \"xaxis tick on ; " $index_now [format "file%d.dat.hist" $file_index ]  [format "file%d.dat.rev.hist" $file_index ] ]
                                                puts $temp [format "    yaxis tick on;"]
                                                puts $temp [format "    yaxis tick major %8.3f ; "  $ytickmajor  ]
                                                puts $temp [format "    yaxis tick minor ticks %8d ; " 1 ]
                                                puts $temp {            xaxis ticklabel on;
                                                                        xaxis ticklabel format decimal ;
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
  								}
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
					}
				} else {
					puts $temp [format "        -pexec \"kill g%d\" \\" $index_now ]
				}

				incr index 
				incr line
			}
#################################################################
# STANDARD DISPLAY OF ONE TIME SERIES
# if backward output file is not exist. 
#################################################################
		} else {
			while { $index < 16 * $page   } {
				incr marker 
				set file_index [expr $index + 1 ]
				set index_now [expr $index  %16 ]

				if { [file exists [format "file%d.dat.hist" $file_index] ] }  {
                                        set lambda1 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 7 ]]
                                        set lambda2 [format "%8.4f"    [lindex [lindex $::ParseFEP::file_lambda  $index ] 8 ]]
					set posx [expr {0.100 + int (($index%16)% 4 ) * 0.295 }]
					set posy [expr {0.910 - int (($index%16) / 4 ) * 0.210 }]
		
					if { $index_now == 7 } {
						puts $temp [format "      -graph  %d -block %s -bxy 1:2 -block %s -bxy 1:3 -block %s -bxy 1:4 -pexec \"xaxis ticklabel on ; " $index_now  [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ]  ]
                                                puts $temp {    xaxis ticklabel format decimal ; 
                                                                        yaxis label char size 1.8 ; 
                                                                        yaxis label place spec ; 
                                                                        yaxis label place -0.10,-0.25 ; 
                                                                        yaxis label  \"\\r{180}\\1P\\0\\f{Helvetica}(\\f{Symbol}D\\1U\\0\\f{Helvetica} )\" ;
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
 								}
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
					} elseif { $index_now == 12 || $index_now == 14 || $index_now == 15   } {
						puts $temp [format "      -graph  %d -block %s -bxy 1:2 -block %s -bxy 1:3 -block %s -bxy 1:4 -pexec \"xaxis ticklabel on ; " $index_now  [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ]  ]
                                                puts $temp {    xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
              								}
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
					} elseif {  $index_now == 13 } {
		       				puts $temp [format "        -graph %d  -block %s -bxy 1:2 -block %s -bxy 1:3 -block %s -bxy 1:4 -pexec \"xaxis ticklabel on ;" $index_now [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ] ]

                                                puts $temp {    xaxis ticklabel format decimal ; 
                                                                        xaxis label char size 1.8 ; 
                                                                        xaxis label place spec ; 
                                                                        xaxis label place  0.15,0.07 ; 
                                                                        xaxis label \"\\f{Symbol}D\\1U\\0\\f{Helvetica} (kcal/mol)\" ;
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
      								}
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
       					} else {
						puts $temp [format "      -graph %d -block %s -bxy 1:2 -block %s -bxy 1:3 -block %s  -bxy 1:4 -pexec \"xaxis ticklabel on ; " $index_now [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ] [format "file%d.dat.hist" $file_index ] ]
                                                puts $temp {    xaxis ticklabel format decimal ; 
                                                                        xaxis ticklabel prec 1 ; 
                                                                        xaxis ticklabel font \"Helvetica\" ;
                                                                        xaxis ticklabel char size 0.8 ;
                                                                        yaxis ticklabel format decimal ; 
                                                                        yaxis ticklabel prec 3 ; 
                                                                        yaxis ticklabel font \"Helvetica\" ;
 								}
						puts $temp   [format "                                                                        yaxis ticklabel char size 0.8 \" \\" ]
					}
				} else {
					puts $temp [format "        -pexec \"kill g%d\" \\" $index_now ]
				}

				incr index 
				incr line
			}			
		}
	
		puts $temp [format "        -hardcopy -printfile probability.%d.png -hdevice PNG"  $page ]
		close $temp 
		#exec display [format "probability.%d.png"  $page ]
		incr page 
	}

#######################################################################
# SUMMARY OF ParseFEP COMPUTATIONS 
#######################################################################

	puts  "ParseFEP: Display summary"
	
	file delete temp.ParseFEP.log   
	file delete temp.reverse.log
	file delete temp.entropy.log 
	file delete temp.reverse.entropy.log 

	set temp1 [open  temp.ParseFEP.log "a+" ]
	if {  $::ParseFEP::fepbofile != ""   } { set temp2 [open  temp.reverse.log "a+"]}
	if {  $::ParseFEP::entropyindex == 1 } { set temp3 [open  temp.entropy.log "a+"]}
	if {  $::ParseFEP::entropyindex == 1  &&   $::ParseFEP::fepbofile != ""  } { 
		set temp4 [open  temp.reverse.entropy.log "a+"]
	}

	
	set source [open ParseFEP.log  "r" ]
	set data_file [split [read  $source  [file size "ParseFEP.log"] ] "\n" ] 
	close $source 

	foreach line $data_file {
		if { [ regexp  "forward:        " $line ] } { 
			foreach { , elem2 elem3 elem4 elem5 } $line {break }
			puts $temp1  [format  "%8.4f %8.4f %8.4f %8.4f" $elem2 $elem3 $elem4 $elem5  ]
		}

		if { $::ParseFEP::fepbofile != ""  && [ regexp  "backward:        " $line ] } { 
			foreach { , elem2 elem3 elem4 elem5 } $line {break }
			puts $temp2  [format  "%8.4f %8.4f %8.4f %8.4f" $elem2 $elem3 $elem4 $elem5  ]
		}

		if { $::ParseFEP::entropyindex == 1  && [ regexp  "forward:" $line ] && [ regexp "entropy"  $line ]  } { 
			foreach { , , elem3 elem4 elem5 elem6 elem7 }  $line { break } 
			puts $temp3  [format  "%8.4f %8.4f %8.4f %8.4f %8.4f" $elem3 $elem4 $elem5 $elem6 $elem7 ] 
		}

		if { $::ParseFEP::entropyindex == 1  &&  $::ParseFEP::fepbofile != ""  && [ regexp  "backward:" $line ] && [ regexp "entropy"  $line ]  } { 
			foreach { , , elem3 elem4 elem5 elem6 elem7 }  $line { break } 
			puts $temp4  [format  "%8.4f %8.4f %8.4f %8.4f %8.4f" $elem3 $elem4 $elem5 $elem6 $elem7 ] 
		}
	}

	close $temp1 
	if { $::ParseFEP::fepbofile    != "" } {close $temp2} 
	if { $::ParseFEP::entropyindex == 1  } {close $temp3}
	if { $::ParseFEP::entropyindex == 1 && $::ParseFEP::fepbofile != "" } {close $temp4}

	file delete grace.summary.exec 

	set temp [open grace.summary.exec "a+"]

	if {  $::ParseFEP::entropyindex  == 1  && $::ParseFEP::fepbofile == ""  } { 

	        puts  $temp { xmgrace -pexec "arrange (3,1,0.10,0.40,0.20,OFF,OFF,OFF)"   \
				   -pexec "with string ;
        	                        string on ;
	                                string g0 ;
        	                        string loctype view ;
        	                        string 0.030, 0.950 ;
        	                        string char size 1.6 ;
        	                        string def \"\\f{Helvetica-Bold}ParseFEP\\f{Helvetica}: Summary \" " \
				-graph 0 -block temp.ParseFEP.log -bxy 1:3 -pexec "xaxis ticklabel on ;
                                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\f{Symbol}D\\1G\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
				-graph 1 -block temp.entropy.log -bxy 1:3 -pexec "xaxis ticklabel on ;
                                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\f{Symbol}D\\1U\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
				-graph 2 -block temp.entropy.log -bxy 1:5 -pexec "xaxis ticklabel on ;
		                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         xaxis label char size 1.2 ;
                                                         xaxis label place spec ;
                                                         xaxis label place  0.00,0.07 ;
                                                         xaxis label \"\\f{Symbol}l\" ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\1T\\0\\f{Symbol}D\\1S\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
							-hardcopy -printfile summary.png -hdevice PNG 
				}
	}
	if  {  $::ParseFEP::fepbofile !=  "" &&  $::ParseFEP::entropyindex  == 1  }  {

	        puts  $temp { xmgrace -pexec "arrange (3,1,0.10,0.40,0.20,OFF,OFF,OFF)"   \
				   -pexec "with string ;
        	                        string on ;
	                                string g0 ;
        	                        string loctype view ;
        	                        string 0.030, 0.950 ;
        	                        string char size 1.6 ;
        	                        string def \"\\f{Helvetica-Bold}ParseFEP\\f{Helvetica}: Summary \" " \
				-graph 0 -block temp.ParseFEP.log -bxy 1:3 -block temp.reverse.log -bxy 1:3 -pexec  "xaxis ticklabel on ;
                                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\f{Symbol}D\\1G\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
				-graph 1 -block temp.entropy.log -bxy 1:3 -block temp.reverse.entropy.log -bxy 1:3 -pexec "xaxis ticklabel on ;
                                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\f{Symbol}D\\1U\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
				-graph 2 -block temp.entropy.log -bxy 1:5 -block temp.reverse.entropy.log -bxy 1:5 -pexec "xaxis ticklabel on ;
		                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         xaxis label char size 1.2 ;
                                                         xaxis label place spec ;
                                                         xaxis label place  0.00,0.07 ;
                                                         xaxis label \"\\f{Symbol}l\" ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\1T\\0\\f{Symbol}D\\1S\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
							-hardcopy -printfile summary.png -hdevice PNG 		
		}
	}
	if  {  $::ParseFEP::fepbofile != "" &&  $::ParseFEP::entropyindex  == 0  } {

#######################################################################
# RETRIEVE FREE-ENERGY DIFFERENCES FROM REVERSE RUN 
#######################################################################

		puts $temp { xmgrace -pexec "arrange (3,1,0.10,0.40,0.20,OFF,OFF,OFF)" \
					  -pexec "with string ;
		                                 string on ;
                		                 string g0 ;
                		                 string loctype view ;
                		                 string 0.030, 0.950 ;
                		                 string char size 1.6 ;
                		                 string def \"\\f{Helvetica-Bold}ParseFEP\\f{Helvetica}: Summary \" " \
				-graph 0 -block temp.ParseFEP.log -bxy 1:3 -block temp.reverse.log -bxy 1:3 -pexec "xaxis ticklabel on ;
                                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         xaxis label char size 1.2 ;
                                                         xaxis label place spec ;
                                                         xaxis label place  0.00,0.07 ;
                                                         xaxis label \"\\f{Symbol}l\" ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\f{Symbol}D\\1G\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
         						-pexec "kill g1" \
        						-pexec "kill g2" \
							-hardcopy -printfile summary.png -hdevice PNG 
			}
	}
	if { $::ParseFEP::entropyindex  == 0  && $::ParseFEP::fepbofile == "" } {
		puts $temp {       echo xmgrace -pexec "arrange (3,1,0.10,0.40,0.20,OFF,OFF,OFF)" \
				 -pexec "with string ;
        	                         string on ;
        	                         string g0 ;
        	                         string loctype view ;
        	                         string 0.030, 0.950 ;
        	                         string char size 1.6 ;
        	                         string def \"\\f{Helvetica-Bold}ParseFEP\\f{Helvetica}: Summary \" " \
				-graph 0 -block temp.ParseFEP.log -bxy 1:3 -pexec "xaxis ticklabel on ;
                                                         xaxis tick major 0.1 ; 
                                                         xaxis tick minor 0.02 ;
                                                         xaxis ticklabel format decimal ; 
                                                         xaxis ticklabel prec 1 ; 
                                                         xaxis ticklabel font \"Helvetica\" ;
                                                         xaxis ticklabel char size 0.8 ;
                                                         xaxis label char size 1.2 ;
                                                         xaxis label place spec ;
                                                         xaxis label place  0.00,0.07 ;
                                                         xaxis label \"\\f{Symbol}l\" ;
                                                         yaxis label char size 1.2 ;
                                                         yaxis label place spec ;
                                                         yaxis label place  0.00,-1.15 ;
                                                         yaxis label  \"\\r{180}\\f{Symbol}D\\1G\\0\\f{Helvetica} (kcal/mol)\" ;
                                                         yaxis ticklabel format decimal ; 
                                                         yaxis ticklabel prec 3 ; 
                                                         yaxis ticklabel font \"Helvetica\" ;
                                                         yaxis ticklabel char size 0.8" \
							-pexec "kill g1" \
							-pexec "kill g2" \
							-hardcopy -printfile summary.png -hdevice PNG  
				    }
	}

	close $temp 

################################################################
# draw the png files
################################################################

	#set platform-specific executable suffix
	set archexe ""
	switch [vmdinfo arch ] {
		WIN64 -
		WIN32 {
			set archexe ".exe"
		}
	}

	set xmgraceexe [format "xmgrace%s" $archexe]
	set xmgracecmd [::ExecTool::find -interactive -description "xmgrace"  $xmgraceexe]
	if { $xmgracecmd == {}  } {
		puts "Cannot find $xmgraceexe, aborting. Please install xmgrace, and retry this plugin again."
		::ParseFEP::Clear
		return 
	}

	for {set  i  1 } {  $i < $page } { incr i } {

	  set foo [catch { ::ExecTool::exec chmod u+x     [format   "grace.%d.exec"        $i ] >&@ stdout }]
	  set foo [catch { ::ExecTool::exec               [format "./grace.%d.exec"        $i ] >&@ stdout }]
	
	  set foo [catch { ::ExecTool::exec chmod u+x     [format   "grace.hist.%d.exec"   $i ] >&@ stdout }]
	  set foo [catch { ::ExecTool::exec               [format "./grace.hist.%d.exec"   $i ] >&@ stdout }]

	  if {  $::ParseFEP::entropyindex  == 1 } {
	     set foo [catch {::ExecTool::exec chmod u+x   [format "grace.entropy.%d.exec"  $i ] >&@ stdout }]
	     set foo [catch {::ExecTool::exec           [format "./grace.entropy.%d.exec"  $i ] >&@ stdout }]
		}
	}
	
	set displayexe [format "display%s" $archexe]
	set displaycmd [::ExecTool::find -interactive -description "display"  $displayexe  ]
	if { $displaycmd == {}  } {
	       puts "Cannot find  $displayexe, aborting. Please install ImageMagick, and retry this plugin again."
#	       ::ParseFEP::Clear
	       return
	}

 	for { set i 1 } { $i < $page } { incr i } {
		set foo [catch { ::ExecTool::exec display   [format  "free-energy.%d.png"    $i ]  &  }] 
		set foo [catch { ::ExecTool::exec display   [format  "probability.%d.png"    $i ]  &  }] 
		if {  $::ParseFEP::entropyindex == 1} {
			set foo [catch { ::ExecTool::exec display [format "entropy.%d.png"   $i ]  &  }] 
		}
	}

	::ExecTool::exec chmod u+x grace.summary.exec 
	::ExecTool::exec ./grace.summary.exec
	::ExecTool::exec display summary.png &

	if { $::ParseFEP::interfilesindex == 0} {
		::ParseFEP::Clear
	}
}
################################################################
# delete the temp files
################################################################

proc ::ParseFEP::Clear {} {

	file delete grace.summary.exec 
	file delete temp.ParseFEP.log 
	file delete temp.entropy.log 
	file delete temp.reverse.log 
	file delete temp.reverse.entropy.log 

	for { set window 1 } { $window  <=  $::ParseFEP::nb_file  } { incr window }  {
			file delete [format "file%d.dat" 	  $window] 
			file delete [format "file%d.dat.entropy"  $window] 
			file delete [format "file%d.dat.hist"     $window] 
			file delete [format "file%d.dat.rev"      $window] 
			file delete [format "file%d.dat.rev.hist" $window] 
	}

       set page  [ expr $::ParseFEP::nb_file / 16 +1 ] 
	for {set  i  1 } {  $i <= $page } { incr i } {
		file delete [format "grace.%d.exec" 	    $i ]
		file delete [format "grace.hist.%d.exec"    $i ]
		file delete [format "grace.entropy.%d.exec" $i ]
	}

}

proc ::ParseFEP::parsefepcmd { args } {

	init_data

	puts "\n--------------------------"
	puts "ParseFEP: Version $::ParseFEP::version"
	puts "--------------------------"

	set narges [llength $args]
	if { $narges == 0 } {
		parsefep_usage 
		error ""
	}

	for {set argnum 0 } { $argnum < [llength $args] } {incr argnum} {
		set arg [lindex $args $argnum]
		set val [lindex $args [expr $argnum + 1 ] ]
		switch -- $arg {
			"-forward"  { set ::ParseFEP::fepofile    $val ; incr argnum }
			"-backward" { set ::ParseFEP::fepbofile   $val ; incr argnum }
			"-entropy"  { set ::ParseFEP::entropyindex   1 }
			"-gauss"    { set ::ParseFEP::gaussian_guess 1 }
			"-gc"       { set ::ParseFEP::gcindex        1 ; set ::ParseFEP::max_order $val; incr argnum }
			"-sos"      { set ::ParseFEP::combine_method 1 }
			"-bar"      { set ::ParseFEP::combine_method 2 }  
			default	    { error "unknown arguments: $arg $val" ; parsefep_usage }
		}
	}
	namdparse
}

proc ::ParseFEP::parsefep_usage {} {
	puts "Usage: parsefep <option1> <option2> ..."	
	puts "NOTE: parsefep is a tootl for analyzing the results of FEP simulations"
	puts " -forward   <file name> the name of the fep output file for the forward simulation"
	puts " -entropy   calculate the enthaply and entropy differences"
	puts " -gc        <max order> Gram-Charlier interpolations, the max_order should be defined"
	puts " -gauss     applying Gaussian distribution to model the underlying probability distribution"
	puts " -backward  <file name> the name of the fep output files for the backward simulation" 
	puts " -<sos|bar> applying simple over sampling method to combine the forward and backward simulations"
	puts "            or applying bennett acceptance ratio method to combine the forward and backward simulations"
}
