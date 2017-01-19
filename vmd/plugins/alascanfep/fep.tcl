##############################################################
# FEP SCRIPT
# Jerome Henin <jhenin@ifr88.cnrs-mrs.fr>
#
# Changes:
# 2010-04-24: added runFEPmin
# 2009-11-17: changed for NAMD 2.7 keywords
# 2008-06-25: added TI routines
# 2007-11-01: fixed runFEP to handle backwards transformations
#             (i.e. dLambda < 0)
##############################################################

##############################################################
# Example NAMD input:
#
# source fep.tcl
#
# alch                  on
# alchFile              system.fep
# alchCol               B
# alchOutFreq           10
# alchOutFile           system.fepout
# alchEquilSteps        500
#
# set nSteps      5000
# set init {0 0.05 0.1}
# set end {0.9 0.95 1.0}
#
# runFEPlist $init $nSteps
# runFEP 0.1 0.9 0.1 $nSteps
# runFEPlist $end $nSteps
##############################################################

##############################################################
# proc runFEPlist { lambdaList nSteps }
#
# Run n FEP windows joining (n + 1) lambda-points
##############################################################

proc runFEPlist { lambdaList nSteps } {
    # Keep track of window number
    global win
    if {![info exists win]} {
      set win 1
    }

    set l1 [lindex $lambdaList 0]
    foreach l2 [lrange $lambdaList 1 end] {
      print [format "Running FEP window %3s: Lambda1 %-6s Lambda2 %-6s \[dLambda %-6s\]"\
        $win $l1 $l2 [expr $l2 - $l1]]
      firsttimestep 0
      alchLambda       $l1
      alchLambda2      $l2
      run $nSteps

      set l1 $l2
      incr win
    }
}


##############################################################
# proc runFEP { start stop dLambda nSteps }
#
# FEP windows of width dLambda between values start and stop
##############################################################

proc runFEP { start stop dLambda nSteps } {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    runFEPlist $ll $nSteps
}


##############################################################
##############################################################

proc runFEPmin { start stop dLambda nSteps nMinSteps temp} {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    if { $nMinSteps > 0 } { 
      alchLambda       $start
      alchLambda2      $start
      minimize $nMinSteps
      reinitvels $temp
    }

    runFEPlist $ll $nSteps
}

##############################################################
##############################################################

proc runTIlist { lambdaList nSteps } {
    # Keep track of window number
    global win
    if {![info exists win]} {
	    set win 1
    }

    foreach l $lambdaList {
	    print [format "Running TI window %3s: Lambda %-6s "	$win $l ]
	    firsttimestep 0
	    alchLambda       $l
	    run $nSteps
	    incr win
    }
}


##############################################################
##############################################################

proc runTI { start stop dLambda nSteps } {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    runTIlist $ll $nSteps
}

##############################################################
# Increment lambda and try to correct truncation errors around
# 0 and 1
##############################################################

proc increment { lambda dLambda } {
    set epsilon 1e-15
    set new [expr { $lambda + $dLambda }]

    if { [expr $new > - $epsilon && $new < $epsilon] } {
      return 0.0
    }
    if { [expr ($new - 1) > - $epsilon && ($new - 1) < $epsilon] } {
      return 1.0
    }
    return $new
}
