##
## $Id: runsqm.tcl,v 1.9 2012/01/12 15:52:15 johanstr Exp $
##
package provide runsqm 0.2

namespace eval ::runsqm:: {
  variable sqmpath  "sqm";# path to sqm executable
  variable sqmsettingsok 1
  variable qmtheory "PM3"
  variable scfconv   1.0e-8
  variable scfconvcoeff   1
  variable scfconvexp     -8
  variable errconvcoeff   1
  variable errconvexp   -1
  variable itrmax    1000
  variable delINOUT  0
  variable verbosity 0
  variable pseudodiag 1
  variable pseudodiagcrit 0.05
  variable printcharges 0
  variable peptidecorr 0
  variable maxcyc 9999
  variable ntpr 10
  variable grmstol 0.02
  variable ndiisattempts 0
  variable ndiismatrices 6

  variable sqmhere 0
  variable sqmbin ""
  variable newsqm 0
  if { [vmdinfo arch] == "WIN32" || [vmdinfo arch] == "WIN64" } {
	set sqmbin "sqm.exe"
  } else {
        set sqmbin "sqm"
  } 
}

  proc ::runsqm::checkfornewsqm {} {
    ::runsqm::writequickSQMin
    catch { exec $::runsqm::sqmpath -O -i tmpin -o tmpout } stderrout
    file delete tmpout
    file delete tmpin
    if { [string first "printbondorders" $stderrout] != -1 } {
      set ::runsqm::newsqm 0
    } else {
      set ::runsqm::newsqm 1
    }
    return    
  }

  proc ::runsqm::writequickSQMin {} {

    set header "Run semi-empirical calculation for VMD\n"
    append header " &qmmm\n   qm_theory = '$::runsqm::qmtheory',\n   qmcharge = 0,\n"
    append header "   scfconv = ${::runsqm::scfconvcoeff}d${::runsqm::scfconvexp},\n"
    append header "   errconv = ${::runsqm::errconvcoeff}d${::runsqm::errconvexp},\n"
    append header "   itrmax = $::runsqm::itrmax,\n   verbosity = $::runsqm::verbosity,\n"
    append header "   pseudo_diag = $::runsqm::pseudodiag,\n"
    append header "   pseudo_diag_criteria = $::runsqm::pseudodiagcrit,\n   printcharges = $::runsqm::printcharges,\n"
    append header "   peptide_corr = $::runsqm::peptidecorr,\n   maxcyc = 0,\n"
    append header "   ntpr = $::runsqm::ntpr,\n   grms_tol = $::runsqm::grmstol,\n"
    append header "   ndiis_attempts = $::runsqm::ndiisattempts,\n   ndiis_matrices = $::runsqm::ndiismatrices,\n"
    append header "   printbondorders = 1\n /\n"
    set fo [open tmpin "w"]
    puts $fo $header
    flush $fo
    close $fo
    return 
  
  }

  proc ::runsqm::sqminit {{interactive 0}} {
    if { ![info exists ::runsqm::sqmhere] } {
        set ::runsqm::sqmhere 0
    }
    if { $::runsqm::sqmhere == 0} {
        set curpath $::runsqm::sqmpath
        if { ![file executable $::runsqm::sqmpath] } {
            set amberhomehere 0
            foreach key [array names ::env] {
                if { $key == "AMBERHOME" && $::env($key) != "" && $::env($key) != " " } {
                  set amberhomehere 1
                }
            }  
            if { $amberhomehere == 1 } {
                if {$interactive == 0} {
                    set  ::runsqm::sqmpath [::ExecTool::find  -description "sqm"  -path [file join $::env(AMBERHOME) bin $::runsqm::sqmbin ] $::runsqm::sqmbin]
                } else {
                    set  ::runsqm::sqmpath [::ExecTool::find -interactive -description "sqm"  -path [file join $::env(AMBERHOME) bin $::runsqm::sqmbin ] $::runsqm::sqmbin]
                }
            } else {
                if {$interactive == 0} {
                    set  ::runsqm::sqmpath [::ExecTool::find  -description "sqm" $::runsqm::sqmbin]
                } else {
                    set  ::runsqm::sqmpath [::ExecTool::find -interactive -description "sqm" $::runsqm::sqmbin]
                }            
            }
        }
        ::runsqm::writequickSQMin
         
        if {![file executable $::runsqm::sqmpath]} {
          set ::runsqm::sqmpath "\"$::runsqm::sqmpath not executable\" #"
          set ::runsqm::sqmhere 0
        } else {
          set ::runsqm::sqmhere 1
          ::runsqm::checkfornewsqm  
        }
        file delete tmpout
        file delete tmpin
    }
    return $::runsqm::sqmhere
  }


proc ::runsqm::setdefaults {} {
  variable ::runsqm::qmtheory "PM3"
  variable ::runsqm::scfconvcoeff   1
  variable ::runsqm::scfconvexp     -8
  variable ::runsqm::errconvcoeff   1
  variable ::runsqm::errconvexp   -1
  variable ::runsqm::itrmax    1000
  variable ::runsqm::delINOUT  0
  variable ::runsqm::verbosity 0
  variable ::runsqm::pseudodiag 1
  variable ::runsqm::pseudodiagcrit 0.05
  variable ::runsqm::printcharges 0
  variable ::runsqm::peptidecorr 0
  variable ::runsqm::maxcyc 9999
  variable ::runsqm::ntpr 10
  variable ::runsqm::grmstol 0.02
  variable ::runsqm::ndiisattempts 0
  variable ::runsqm::ndiismatrices 6
}


proc ::runsqm::checkpars {} {
  set display_error 0 
  set msg ""
  if {$::runsqm::maxcyc == 0} {
      set display_error 1 
      append msg "Warning: With maxcyc = 0 only a single point calculation will be run.\n"
   }
   if {$::runsqm::itrmax > 1000} {
      set display_error 1
      append msg "Warning: If SCF convergence hasn't been reached by 1000 steps it is unlikely to converge. It is recommended that the maximum number of SCF steps is set to 1000 or less.\n"
   }
   if {$::runsqm::scfconvcoeff < 1 || $::runsqm::scfconvcoeff > 9} {
     set display_error 2
     append msg "Error: SCF convergence criteria must have a coefficient between 1 and 9.\n"
   }
   if {$::runsqm::errconvcoeff < 1 || $::runsqm::errconvcoeff > 9} {
     set display_error 2
     append msg "Error: Maximum SCF error criteria must have a coefficient between 1 and 9.\n"
   }
   if { $display_error == 2 } {
    tk_messageBox -message $msg -type ok -title "Check SQM parameters" -icon error
   } elseif { $display_error == 1} {
     tk_messageBox -message $msg -type ok -title "Check SQM parameters" -icon warning
     destroy .sqmsettingsgui
   } else {
     destroy .sqmsettingsgui
   }
}

proc ::runsqm::writeSQMinput { atom_selection total_charge sporgo fname }  {
    set AM1atomtypes [list "H" "C" "N" "O" "F" "Al" "Si" "P" "S" "Cl" "Zn" "Ge" "Br" "I" "Hg"]
    set qmcharge("H") 1
    set qmcharge("C") 6
    set qmcharge("N") 7
    set qmcharge("O") 8
    set qmcharge("F") 9
    set qmcharge("Al") 13
    set qmcharge("Si") 14
    set qmcharge("P") 15
    set qmcharge("S") 16
    set qmcharge("Cl") 17
    set qmcharge("Zn") 30
    set qmcharge("Ge") 32
    set qmcharge("Br") 135
    set qmcharge("I") 53
    set qmcharge("Hg") 80
    set num_a [$atom_selection num]
    set mI [$atom_selection molid]
    set allfound 1
    set maxcyctmp $::runsqm::maxcyc
    if { $sporgo == "SP" } {
        variable ::runsqm::maxcyc 0
    }    
    #puts "Calculting Mulliken charges and Bond Orders using SQM"
    #puts "Writing SQM input file : $fname"
    set header "Run semi-empirical calculation for VMD\n"
    append header " &qmmm\n   qm_theory = '$::runsqm::qmtheory',\n   qmcharge = $total_charge,\n"
    append header "   scfconv = ${::runsqm::scfconvcoeff}d${::runsqm::scfconvexp},\n"
    append header "   errconv = ${::runsqm::errconvcoeff}d${::runsqm::errconvexp},\n"
    append header "   itrmax = $::runsqm::itrmax,\n   verbosity = $::runsqm::verbosity,\n"
    append header "   pseudo_diag = $::runsqm::pseudodiag,\n"
    append header "   pseudo_diag_criteria = $::runsqm::pseudodiagcrit,\n   printcharges = $::runsqm::printcharges,\n"
    append header "   peptide_corr = $::runsqm::peptidecorr,\n   maxcyc = $::runsqm::maxcyc,\n"
    append header "   ntpr = $::runsqm::ntpr,\n   grms_tol = $::runsqm::grmstol,\n"
    append header "   ndiis_attempts = $::runsqm::ndiisattempts,\n   ndiis_matrices = $::runsqm::ndiismatrices,\n"
    if { $::runsqm::newsqm == 1 } {
      append header "   printbondorders = 1\n"
    }
    append header " /"
    set fo [open $fname "w"]
    puts $fo $header
    variable ::runsqm::maxcyc $maxcyctmp
    set a_type [$atom_selection get element]
    set a_x [$atom_selection get x]
    set a_y [$atom_selection get y]
    set a_z [$atom_selection get z]
    for {set i 0} {$i < $num_a} {incr i} {
        set c_str " "
        append c_str [format "%-3d" $qmcharge("[lindex $a_type $i]")]
        append c_str [format "%-7s" [lindex $a_type $i]]
        append c_str [format "% 12.6f" [lindex $a_x $i]]
        append c_str [format "% 12.6f" [lindex $a_y $i]]
        append c_str [format "% 12.6f" [lindex $a_z $i]]  
        puts $fo $c_str 
    }
    flush $fo
    close $fo
}

proc ::runsqm::readSQMout { fname cm } {
    #puts "Reading SQM outputfile: $fname"
    set fi [open $fname "r"]
    set converged 0
    set n 0
    set a_x {}
    set a_y {}
    set a_z {}
    set bondordersfound 0
    while { [gets $fi line] >= 0 } {
        if { $converged > 0 } {
#        	puts $line
        }
        set line [string trim $line]
        if { [string length $line] > 1 } {
            if { [string compare $line "RESULTS"] == 0 } {
                set converged 1
            } elseif { $converged == 1 
                && [string compare $line "Atom    Element       Mulliken Charge"] == 0 } {
                set converged 2
            } elseif { $converged == 2 } {
                if { [string first "Total Mulliken Charge =" $line] == -1 } {
                    set splt [split [regsub -all {[ \t\n]+} "$line" { }]]
                    set a_type($n) [lindex $splt 1]
                    set a_mullik($n) [lindex $splt 2]
                    set BO($n,$n) 0.0
                    incr n
                } else {
                    set converged 3
                    set splt [split [regsub -all {[ \t\n]+} "$line" { }]]
                    #puts "readSQMout: Number of atoms = $n, total charge = [lindex $splt 4]" 
                }
            } elseif { $converged == 3 &&
                [string first "QMMM: QM_NO.   MM_NO.  ATOM" $line] != -1 } {
                    set converged 4
            } elseif { $converged == 4 } {
                if { [string first "QMMM:" $line] != -1 } {
                    set splt [split [regsub -all {[ \t\n]+} "$line" { }]]
                    #set i [expr [lindex $splt 1]-1]
                    #set a_pos($i) {[lindex $splt 4] [lindex $splt 5] [lindex $splt 6]}
                    lappend a_x [lindex $splt 4] 
                    lappend a_y [lindex $splt 5] 
                    lappend a_z [lindex $splt 6] 
                } else { 
                    set converged 5
                }
            } elseif { $converged == 5 && [string first "BOND_ORDER" $line] != -1 } {
                set converged 6
            } elseif { $converged == 6 } {
                if { [string first "QMMM:" $line] != -1 } {
                   set bondordersfound 1
                   set splt [split [regsub -all {[ \t\n]+} "$line" { }]]
                   set i1 [expr [lindex $splt 1] - 1]
                   set i2 [expr [lindex $splt 3] - 1]
                   set BO($i1,$i2) [expr [lindex $splt 5]]
                   set BO($i2,$i1) [expr [lindex $splt 5]]
                   set BO($i1,$i1) [expr ($BO($i1,$i1) + $BO($i1,$i2))]
                   set BO($i2,$i2) [expr ($BO($i2,$i2) + $BO($i1,$i2))]
                } else {
                    set converged 7
                }
            }
        }
    }
    if { $cm == "cm1a" && $bondordersfound == 1 } {
       set charges [::runsqm::calc_cm1a [array get a_type] [array get a_mullik] [array get BO]]
    } elseif {$cm == "cm1a"} {
        puts "WARNING: The modified SQM is not available, assinging $::runsqm::qmtheory Mulliken charges instead of CM1A*1.14"
        set charges {}
    	for { set i 0 } { $i < $n } { incr i } {
	    lappend charges $a_mullik($i)
        }
    } else {
        set charges {}
    	for { set i 0 } { $i < $n } { incr i } {
	    lappend charges $a_mullik($i)
        }
    }
    return [list $a_x $a_y $a_z $charges]
}

proc ::runsqm::calc_cm1a { pl_type pl_mullik pl_bo } {
    array set a_type $pl_type
    array set a_mullik $pl_mullik
    array set a_bo $pl_bo
    set c("N")   0.3846
    set c("NC") -0.3846
    set c("CN") $c("NC")
    set c("F")   0.1468
    set c("Cl")  0.0405
    set c("Br")  0.1761
    set c("S")  -0.1311
    set c("I")   0.2380
    set d("HN")  0.0850
    set d("NH") $d("HN")
    set d("HO")  0.1447
    set d("OH") $d("HO")
    set d("NC") -0.0880
    set d("CN") $d("NC")
    set d("O")  -0.0283
    set d("F")   0.0399
    set d("HSi") 0.0640
    set d("SiH") $d("HSi")
    set d("S")  -0.0956
    set d("Cl") -0.0276
    set d("Br") -0.0802
    set d("I")  -0.1819
    set d("OS") -0.0600
    set d("SO") $d("OS")
    set d("NO") -0.0630
    set d("ON") -0.0630
    set num_a [array size a_type]
    #puts "Calculating CM1A charges for $num_a atoms."
    for { set i 0 } { $i < $num_a } { incr i } {
        set pref1 0.0
        set pref2 0.0
        if { [info exists c("$a_type($i)")] } {
            set pref1 [expr $pref1 + ($c("$a_type($i)"))]
        }
        if { [string compare $a_type($i) "N"] == 0 } {
            for { set j 0 } { $j < $num_a } { incr j } {
                if { $j != $i && [string compare $a_type($j) "C"] == 0 } {
                    
                    set pref1 [expr $pref1 + 0.5 * ($c("NC")) * (tanh(($a_bo($i,$j)-2.3)/0.1)+1)]
                    set pref2 [expr $pref2 + 0.5 * ($d("NC")) * (tanh(($a_bo($i,$j)-2.3)/0.1)+1)]
                } elseif { $j != $i && [string compare $a_type($j) "O"] == 0 } {
                    set pref2 [expr $pref2 + $a_bo($i,$j)*$d("NO")]
                }
            }
        }
        set dq($i) [expr $a_mullik($i)*$pref1+$pref2]
        if { [info exists d("$a_type($i)")] } {
            set dq($i) [expr $dq($i) + $d("$a_type($i)")]
        }
        if { [string compare $a_type($i) "H"] == 0 } {
            for { set j 0 } { $j < $num_a } { incr j } {
                if { $j != $i && [info exists d("H$a_type($j)")] } {
                    set dq($i) [expr $dq($i)+$a_bo($i,$j)*$d("H$a_type($j)")]
                }
            }
        } elseif { [string compare $a_type($i) "O"] == 0 } {
            for { set j 0 } { $j < $num_a } { incr j } {
                if { $j != $i && [string compare $a_type($j) "S"] == 0 } {
                    set dq($i) [expr $dq($i)+$a_bo($i,$j)*$d("OS")]
                }
            }
        }
    }
    set cm1charges [list]
    for {set i 0} { $i < $num_a} {incr i} {
        set a_cm1($i) [expr $a_mullik($i)+$a_bo($i,$i)*$dq($i)]
        for { set j 0 } { $j < $num_a } { incr j } {
            if { $i != $j } {
                set a_cm1($i) [expr $a_cm1($i) - $a_bo($i,$j)*$dq($j)]
            }
        }
        set a_cm1($i) [expr $a_cm1($i)*1.14]
	lappend cm1charges $a_cm1($i)
        #puts "$i $a_type($i) $a_mullik($i) $a_cm1($i)"
    }

   return $cm1charges    
}

proc ::runsqm::get_charges { selection totalcharge cm  } {
     if { [::runsqm::sqminit] == 1 } {
	    file delete "mdout"
	    file delete "mdin"
	    ::runsqm::writeSQMinput $selection $totalcharge "SP" "mdin"
            if { $runsqm::newsqm == 1 && $cm == "cm1a" } {
               puts "Calculating CM1A*1.14 charges using SQM."
            } else {
               puts "Calculating $::runsqm::qmtheory $cm charges using SQM."
            }
	    puts "$::runsqm::sqmpath -O -i mdin -o mdout"
            exec $::runsqm::sqmpath -O -i mdin -o mdout
	    set cm1charges [::runsqm::readSQMout "mdout" $cm]
	    $selection set charge [lindex $cm1charges 3]
            #if { $runsqm::newsqm == 1 && $cm == "cm1a" } {
	    #   puts "CM1A*1.14 charges calculated and assigned."
            #} else {
            #   puts "$::runsqm::qmtheory $cm charges calculated and assigned"
            #}
            puts "SQM calculation complete and charges assigned."
	}
}

proc ::runsqm::SQMopt { selection totalcharge } {
     if { [::runsqm::sqminit] == 1 } {
	     ::runsqm::writeSQMinput $selection $totalcharge "GO" "mdin"
	     puts "Calculating $::runsqm::qmtheory minimized structure using SQM."
	     puts "$::runsqm::sqmpath -O -i mdin -o mdout"
	     exec $::runsqm::sqmpath -O -i mdin -o mdout
	     set xyzc [::runsqm::readSQMout "mdout" "none"]
	     #puts $xyzc
	     $selection set x [lindex $xyzc 0]
	     $selection set y [lindex $xyzc 1]
	     $selection set z [lindex $xyzc 2]
	     #$selection set charge [lindex $xyzc 3]
             puts "SQM calculation complete and atom positions updated"

	}
} 
