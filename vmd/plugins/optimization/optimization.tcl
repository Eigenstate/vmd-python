# The OPTIMIZATION plugin can be used to perform multidimensional 
# minimization of functions. 
#
# $Id: optimization.tcl,v 1.14 2013/04/15 16:41:38 johns Exp $
#

# The plugin can use two different algorithms described in Numerical Recipes:
# 1) The first is the downhill-simplex method due to Nelder and Mead, which 
#    is a slow, but failsafe method that is just walking downwards. It has
#    the advantage of not requiring any derivatives.
#    It's worth reading the according chapter in the recipes:-)
# 2) The second method is a simulated annealing variant of that algorithm.

# The plugin provids a convenient interface to these algorithms. You simply have
# to define a function to be optimized depending on a deliberate number of parameters.
# E.g. if you have some least squares error function depending on two parameters that
# returns a scalar then optimization will find the set of parameters that minimized
# that scalar. Similar to the atomselect command it returns and object through which
# you can configure and control the optimization. For each call a different unique
# namespace (+accessory procedures) is generated so the you can maintain several
# optimization projects at the same time.

# I have extended the algorithm so that you can define lower and upper bounds
# for the parameter search. Thus you can prevent the optimizer to try using
# values that won't make sense.

# Further you have the possibility to analyze the optimization.
# After the optimization completed (either by convergence or by max. iterations)
# you can request lists for the fitting parameters and the error function during
# the opt run or even plot these using multiplot.

# It is written completely in TCL but if the function to be optimized is some
# complicated thing written in C and wrapped in TCL you can to even expensive
# computations very fast. Thats because the optimization itself is actually mere
# book keeping and feeding the function with new parameters.

# Examples:
# ---------
### First we have to define a function to optimize, in this case a 2 dimensional parabola: 
#   proc parabel2 {xy} {return [expr {pow([lindex $xy 0]-1,2)+pow([lindex $xy 1]-3,2)}]}

### Now we can set up an optimization problem. The return value is a unique handler used to 
### control the optimization. Thus you can manage multiple problems at the same time..
### We are using the downhill simplex algorithm, the relative convergence tolerance is 0.01.
#   set opt [optimization -downhill -tol 0.01 -function parabel2]

### The simplex must consist of ndim+1 vertices, while ndim is the number of independent 
### variables of your function. The initsimplex command can be used to automatically 
### construct a simplex around a start value.
#   $opt initsimplex {2 1}
#   > {{2 1} {2.02 1} {2 1.01}} {5.0 5.0404 4.9601}

### Note that you could also construct the simplex yourself and provide it
### using the -simplex option.

### Perform the optimization:
### The resulting optimal parameters and their corresponding function value
### are returned
# set result [$opt start]

### The results can also be returned any time later using the following syntax:
# set result [$opt result]

### We can plot how the variables and function values developed during the optimization:
# $opt analyze

### Now we set the tolerance to a lower value and start from the current vertex again:
# $opt configure -tol 0.000001
# $opt start

# When we are done we can delete all the data
# $opt quit

### If -miny is specified the optimization finishes, if the function value gets below
### this boundary. Useful for optimizing error functions.
# set opt [optimization -simplex {{2 1} {4 1} {2 5}} -miny 0.1 -function parabel2]
# $opt start
# $opt quit

### Let's try it again but with parameter boundaries defined, i.e.
### the optimizer can vary the values only within the given range.
### Here the first parameter may vary between 0 and 10 the second
### one between -1 and 5.
# set opt [optimization -downhill -tol 0.01 -function parabel2]
# $opt configure -bounds {{0 10} {-1 5}}
# $opt initsimplex {2 1}
# $opt start
# $opt analyze
# $opt quit

### The next example is a 1D double well potential. The function f(x) = 2*x - 8*(x)**2 + (x)**4 
### has two minima and the starting point is on the slope to the higher minimum.
### Thus the downhill simplex will only find the local minimum
### whereas simulated annealing can find the global optimum.

# proc doublewell {x} { return [expr 2.0*$x - 8*pow($x,2) + pow($x,4)] }
# set opt [optimization -tol 0.001 -function doublewell]
# $opt configure -annealing
# $opt initsimplex 3
# $opt start

### Now we reinitialize the simplex and try simulated annealing:
# $opt initsimplex 3
# set [opt optimization -annealing -tol 0.0001 -T 25 -iter 20 -Tsteps 15 -function doublewell]
# $opt start
# $opt analyze
# $opt quit

# Depending on the values for the initial temperature T, the number of iterations per
# cycle iter and the number of temperature cycles Tsteps the optimizer sould find the global
# minimum.

package provide optimization 1.3

namespace eval ::Optimize:: {
   proc initialize {} {
      variable projectlist {}
      variable projectcount -1
   }
   initialize
}

proc ::Optimize::init { } {
   incr ::Optimize::projectcount
   set ns "::Optimize::Opt${::Optimize::projectcount}"

   if {[namespace exists $ns]} {
      puts "Reinitializing namespace $ns."
   } else {
      puts "Creating namespace $ns"
   }

   namespace eval $ns {
      variable p {};         # the current simplex
      variable y {};         # function evaluated at the simplex vertices
      variable psum {};      # This list is used to communicate between subroutines
      variable pbounds {};   # Upper and lower boundaries for each parameter
      variable delta 0.005;    # if bounds violated, move in by delta
      variable ndim 0;       # Number of independent variables
      variable k 0;          # Number of points in complex, defaults to 2*ndim
      variable logp {};      # Log the values of the best vertex
      variable logy {};      # Log the result of the best vertex
      variable logT {};      # Log the temperature during simulated annealing
      variable logysize {}
      variable logpavg {}
      variable T0   1.0;     # Starting temperature for simulated annealing
      variable T    1.0;     # Current temperature for simulated annealing
      variable Texp 1;       # exponent determining the temperature decay (1=linear, 2=quadratic, ...)
      variable Tsteps 2;     # Number of temperature cycles 
      variable rfreq 3;      # Every $rfreq steps replacec the first vertex with the best ever values
      variable pbest {};  
      variable ybest 1e15;   # Initialize best result to a huge number
      variable yhi;          # The worst vertex in the simplex
      variable function {};  # The full name of the procedure to evaluate
      variable implicit {};  # additional constraints
      variable ibounds {};   # bounds on additional constraints
      variable alpha 1.3;    # factor for reflecting complex worst point
      variable ftol 0.0001;  # Fractional tolerance
      variable miny {};      # If functional value gets below $miny, consider it converged.
      variable iter 0;       # Current number of iterations
      variable totaliter 0;  # Totaal number of iterations (for annealing)
      variable maxiter 1000; # Maximum number of iterations
      variable maxiterCheck 10; # Maximum number of iterations for implicit bound checking
      variable algo "downhill"; # Algorithm to use: downhill or annealing
      variable debug 0;      # Print debugging output?
      variable success 0;    # Successfully converged?

      # HERE COMES A NUMBER OF PROCEDURES FOR EACH PROJECT:

      # Create a projecthandle procedure that provides some commands to control the optimization
      # It's full name will be returned when you invoke optimization.
      proc handle { command args } {
	 variable w
	 switch $command {
	    namespace { return [namespace current] }
	    configure { 
 	       set pos [lsearch $args "-downhill"]
	       if {$pos>=0} { variable algo "downhill" }
 	       set pos [lsearch $args "-annealing"]
	       if {$pos>=0} { variable algo "annealing" }

 	       set pos [lsearch $args "-simplex"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable p [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-result"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable y [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-tol"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable ftol [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-bounds"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable pbounds [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-miny"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable miny [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-iter"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable maxiter [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-Tsteps"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable Tsteps [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-T"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable T0 [expr {double([lindex $args [expr {$pos+1}]])}]
	       }

 	       set pos [lsearch $args "-Texp"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable Texp [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-restartfreq"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable rfreq [lindex $args [expr $pos+1]]
	       }

	       set pos [lsearch $args "-function"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable function [lindex $args [expr $pos+1]]
	       }

	       set pos [lsearch $args "-implicit"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable implicit [lindex $args [expr $pos+1]]
	       }

	       set pos [lsearch $args "-ibounds"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable ibounds [lindex $args [expr $pos+1]]
	       }

 	       set pos [lsearch $args "-debug"]
	       if {$pos>=0 && $pos+1<[llength $args]} {
		  variable debug [lindex $args [expr $pos+1]]
	       }
	    }
	    initsimplex {
	       set scale 0.1
	       if {[llength $args]>=2} {
		  set scale [lindex $args 1]
	       } 
               variable ibounds
               variable implicit
               if { [llength $implicit] > 0 && [llength $ibounds] == [llength $implicit] } {
  	          return [construct_initial_complex [lindex $args 0]]
               } else {
  	          return [construct_initial_simplex [lindex $args 0] $scale]
               }
	    }
	    analyze {
	       analyze
	    }
	    log {
	       variable logp
	       variable logy
	       variable logT
	       return [list $logp $logy $logT]
	    }
	    numiter {
	       variable totaliter
	       return $totaliter
	    }
	    success {
	       variable success
	       return $success
	    }
	    result {
	       variable pbest
	       variable ybest
	       return [list $pbest $ybest]
	    }
	    start {
	       variable p
               variable ibounds
               variable implicit

	       if {![llength $p]} {
		  error "No simplex specified!"
	       }
	       puts "Starting [expr {[llength $p]-1}]-dimensional optimization."
	       variable y
	       if {![llength $y]} {
		  variable function
		  set i 0
		  foreach vertex $p {
		     puts "Initializing vertex $i: $vertex"
		     lappend y [$function $vertex]
		     incr i
		  }
	       }
	       if {[llength $p]!=[llength $y]} {
		  error "::Optimize::init: Length of result vector and simplex differ ([llength $p]!=[llength $y])."
	       }

	       set i 0
	       puts "Starting optimization with following simplex:"
	       foreach pp $p yy $y {
		  puts "$i: $pp --> $yy"
		  incr i
	       }

	       variable pbest {}
	       variable ybest 1e15

	       variable algo
	       variable iter
	       variable success
	       if {$algo=="downhill"} {
                  if { [llength $implicit] > 0 && [llength $ibounds] == [llength $implicit] } {
   		     foreach {p y} [downhill_complex_optimization] {break}
		  } else {
                     foreach {p y} [downhill_simplex_optimization] {break}
                  }
		  variable totaliter $iter
		  if {$success} { 
		     puts "Optimization converged after $totaliter steps."
		  }		  
		  return [list [lindex $p 0] [lindex $y 0]]
	       } elseif {$algo=="annealing"} {
		  variable logp {}
		  variable logy {}
		  variable logT {}
		  variable Tsteps
		  variable T0
		  variable Texp
		  variable maxiter
		  variable totaliter 0
		  puts "Simulated annealing:"
		  puts "Initial temperature T0=$T0"
		  puts "Number of T cycles Tsteps=$Tsteps"
		  for {set i 0} {$i<$Tsteps} {incr i} {
		     # Every $rfreq steps replacec the first vertex with the best ever values
		     variable rfreq
		     if {$i>0 && !($i%$rfreq)} {
			variable p
			variable y
			variable ybest
			variable pbest
			set havebest 0
			foreach pp $p yy $y {
			   puts "ybest=$ybest; yy=$yy"
			   if {$yy==$ybest} { set havebest 1 }
			}
			# We should only replace the vertex if the simplex does not currently
			# contain the best ever point.
			if {!$havebest} {
			   puts "Restarting with best-ever point:"
			   puts "p=$pbest"
			   puts "y=$ybest"
			   lset p 0 $pbest
			   lset y 0 $ybest
			}
		     }
		  
		     # Set the new temperature and run optimization cycle
		     variable T [expr {$T0*pow(1-$i/double($Tsteps),$Texp)}]
		     puts "Simulated annealing step $i: T=$T"
		     simulated_annealing $T 
		     incr totaliter $iter
		     if {$success} { 
			puts "Already converged after T=$T"
			break
		     }
		  }
		  puts "Simulated annealing step [expr {$i+1}]: T=0"
		  simulated_annealing 0.0
		  if {$success} { 
		     puts "Simulated annealing converged."
		  }		  
		  return [list $pbest $ybest]

	       } else {
		  error "Algorithm must be specified (-downhill | -annealing)."
	       }
	    }
	    quit   { 
	       namespace delete [namespace current]
	       return
	    }
	 }
      }

      # Multidimensional mimimization of the function $function(x) where x[1..ndim]
      # is a vector in ndim dimensions by the downhill simlex method of Nelder and Mead.
      # The matrix p[1..ndim+1][1..ndim] consists of ndim+1 rows of ndim-dimensional 
      # vectors which are the vertices of the starting simplex.
      # The vector y[1..ndim+1] is initialized with the values evaluated at the
      # ndim+1 vertices (rows) of p. ftol is the fractional convergence tolerance to 
      # be achieved in the function value.
      proc downhill_simplex_optimization {} {
	 variable debug
	 variable y
	 variable p 
	 variable psum [get_psum $p]
	 variable ftol
	 variable function
	 variable miny
	 variable maxiter
	 variable success 0
	 variable logp {}
	 variable logy {}
	 variable logysize {}
	 variable logpavg {}
	 variable iter 0; # Number of function evaluations
	 set TINY 1e-10
	 variable ndim [expr {[llength $y]-1}]
	 while {1} {
	    # First we must determine which point is the highest (worst),
	    # next-highest and lowest.
	    set pysort {}
	    foreach row $p val $y {
	       lappend pysort [list $val $row]
	    }
	    set pysort [lsort -real -index 0 $pysort]

	    set y {}
	    set p {}
	    foreach row $pysort {
	       lappend y [lindex $row 0]
	       lappend p [lindex $row 1]
	    }
	    set pbest  [lindex $p 0] 
	    set ybest  [lindex $y 0] 
	    set yworst [lindex $y end] 
	    lappend logp $pbest 
	    lappend logy $ybest
	    lappend logysize [expr {$yworst-$ybest}]
	    lappend logpavg [vecscale $psum [expr {1.0/double($ndim+1)}]]

	    # Compute fractional range from highest to lowest and return if satisfactory
	    set rtol [expr {2.0*abs($yworst-$ybest)/(abs($yworst)+abs($ybest)+$TINY)}]
	    if {$rtol<$ftol} { 
	       if {$debug>0} { puts "Tolerance converged: rtol<ftol ($rtol<$ftol)" }
	       break
	    }
	    # If minimum is given check if lowest y is smaller than that
	    if {[llength $miny]} {
	       if {$ybest<$miny} {
		  if {$debug>0} { puts "Lower boundary reached ($ybest<$miny)." }
		  break
	       }
	    }

	    if {$debug>1} {
	       puts "Evaluation $iter: y=$y; rtol=$rtol"
	       if {$debug>2} {
		  puts "p=$p"
	       }
	    }
	    if {$iter>=$maxiter} {
	       if {$debug>0} {
		  puts "Maximum number of iterations ($iter/$maxiter) exceeded (rtol=$rtol, ftol=$ftol)."
		  if {$debug>1} {
		     puts "result p=$p; y=$y"
		  }
	       }
	       variable success 0
	       return [list $p $y]
	    }

	    incr iter 2
	    # Begin a new iteration
	    # First extrapolate by a factor -1 through the face of the simplex across from the 
	    # high point, i.e. reflect the simplex from the high point.
	    set ytry [amotry -1.0]
	    if {$ytry <= $ybest} {
	       # Gives a better result than the best point, so try an additional 
	       # extrapolation by a factor 2
	       set ytry [amotry 2.0]
	    } elseif {$ytry >= [lindex $y 1]} {
	       # The reflected point is worse than the second highest, so look for an intermediate 
	       # lower point, i.e., so a one-dimensional contraction.
	       set ytry [amotry 0.5]
	       if {$ytry >= $yworst} {
		  # Can't seem to get rid of that high point.
		  # Better contract around the lowest (best) point.
		  set i 1
		  foreach row [lrange $p 1 end] {
		     #set j 0
		     #foreach val $row {
			#set new [expr {0.5*($val+[lindex $p 0 $j])}]
			#lset p    $i $j $new
			#lset psum $j $new
			#incr j
		     #}
		     set pnew [vecscale 0.5 [vecadd $row $pbest]]
		     lset p $i $pnew

		     if {$debug>2} { puts "contraction $i pnew=$pnew" }
		     lset y $i [$function $pnew]
		     incr i
		  }
		  incr iter $ndim
		  set psum [get_psum $p]
	       }
	    } else { incr iter -1 }
	 }
	 
	 variable success 1
	 return [list $p $y]
      }

      # Extrapolates by a factor fac through the face of the simplex across from the
      # high point, tries it, and replaces the hight point if the new point is better.
      proc amotry {fac} {
	 variable p
	 variable psum
	 variable pbounds
	 variable y
	 variable ndim
	 variable function
	 variable implicit
         variable ibounds
	 variable debug
	 set fac1 [expr {(1.0-$fac)/$ndim}]
	 set fac2 [expr {$fac1-$fac}]
	 set ptry {}

	 # Construct the trial point
	 set ptry [vecsub [vecscale $psum $fac1] [vecscale $fac2 [lindex $p end]]]

         # Check parameter boundaries
##       if {abs($fac)>=1 && [llength $pbounds]}
         if {[llength $pbounds]} {
            foreach bound $pbounds pt $ptry {
               foreach {lower upper} $bound {}   
               if {([llength $lower] && $pt<$lower) || ([llength $upper] && $pt>$upper)} {
                  # Return something worse than the worst point
## puts "boundary violation: pt=$pt"
                  return [expr {[lindex $y end]+10000}]
               }
            }
         } 

	 # Evaluate the function at the trial point
	 set ytry [$function $ptry]
	 if {$debug>2} { puts "amotry $ndim $fac: ptry=$ptry ytry=$ytry" }

	 # If it's better than the highest, then replace the highest.
	 if {$ytry<[lindex $y end]} {
	    lset y end $ytry

	    set psum [vecadd $psum [vecsub $ptry [lindex $p end]]]
	    lset p end $ptry
	 }
	 return $ytry
      }

      proc downhill_complex_optimization {} {
	 variable debug
         variable delta 
         variable p
         variable y 
         variable ndim
         variable k
         variable bounds
         variable ibounds
         variable maxiterCheck
	 variable ftol
	 variable function
	 variable miny
         variable iter 0
	 variable maxiter
	 variable success 0
         variable alpha
	 set TINY 1e-10

        while {1} {
            # find minimum
            set indMin 0
            for {set icm 1} {$icm < $k} {incr icm} {
               if { [lindex $y $indMin] > [lindex $y $icm] } {
                  set indMin $icm
               }
            }
          
            #find max
            set indMax 0
            for {set icm 1} {$icm < $k} {incr icm} {
               if { [lindex $y $indMax] <= [lindex $y $icm] } {
                  set indMax $icm
               }
            }
   
            set yworst [lindex $y $indMax]
            set ybest [lindex $y $indMin]
	    # Compute fractional range from highest to lowest and return if satisfactory
	    set rtol [expr {2.0*abs($yworst-$ybest)/(abs($yworst)+abs($ybest)+$TINY)}]
	    if {$rtol<$ftol} { 
	       if {$debug>0} { puts "Tolerance converged: rtol<ftol ($rtol<$ftol)" }
	       break
	    }
	    # If minimum is given check if lowest y is smaller than that
	    if {[llength $miny]} {
	       if {$ybest<$miny} {
		  if {$debug>0} { puts "Lower boundary reached ($ybest<$miny)." }
		  break
	       }
	    }

	    if {$debug>1} {
	       puts "Evaluation $iter: y=$y; rtol=$rtol"
	       if {$debug>2} {
		  puts "p=$p"
	       }
	    }
	    if {$iter>=$maxiter} {
	       if {$debug>0} {
		  puts "Maximum number of iterations ($iter/$maxiter) exceeded (rtol=$rtol, ftol=$ftol)."
		  if {$debug>1} {
		     puts "result p=$p; y=$y"
		  }
	       }
	       variable success 0
             # sort array before returning it
	       set pysort {}
               foreach row $p val $y {
	          lappend pysort [list $val $row]
	       }
               set pysort [lsort -real -index 0 $pysort]
	       set y {}
               set p {}
	       foreach row $pysort {
	          lappend y [lindex $row 0]
	          lappend p [lindex $row 1]
       	       }
	       return [list $p $y]
	    }

            set centroid [calcCentroid $indMax]
            lset p $indMax [vecsub [vecscale [expr 1+$alpha] $centroid] [vecscale $alpha [lindex $p $indMax]]]
            lset y $indMax [$function [lindex $p $indMax]]
            incr iter
         
            set suc [checkBounds $indMax]
            lset y $indMax [$function [lindex $p $indMax]]
       
            set iterCheck 0
            while { $iterCheck < $maxiterCheck  } {
               set indMaxNew 0
               for {set icm 1} {$icm < $k} {incr icm} {
                  if {[lindex $y $indMaxNew] < [lindex $y $icm]} {
                     set indMaxNew $icm
                  }
               }
   
               if {$indMaxNew == $indMax} {
                  lset p $indMax [vecscale 0.5 [vecadd [lindex $p $indMax] $centroid]]
                  incr iterCheck
               } else {
                  break
               }
               set suc [checkBounds $indMax]
               lset y $indMax [$function [lindex $p $indMax]] 
            }
            incr iter $iterCheck
         }
	 variable success 1
         # sort array before returning it
	 set pysort {}
         foreach row $p val $y {
	    lappend pysort [list $val $row]
	 }
         set pysort [lsort -real -index 0 $pysort]
	 set y {}
         set p {}
	 foreach row $pysort {
	    lappend y [lindex $row 0]
	    lappend p [lindex $row 1]
	 }
         return [list $p $y]
      }

      proc checkBounds { ind } {

         variable p
         variable psum
         variable pbounds
         variable y
         variable ndim
         variable function
         variable implicit
         variable ibounds
         variable debug
         variable delta
         variable maxiterCheck
      
         set kt 1
         set countCheck 0
         while { $kt && ($countCheck <= $maxiterCheck) } {
            set kt 0
            set temp {}
            for {set i 0} {$i < $ndim} {incr i} {
               if { [lindex [lindex $p $ind] $i] <= [lindex $pbounds $i 0]} {
                  lappend temp [expr [lindex $pbounds $i 0] + $delta]
                  if { $debug } {
                     puts "boundary violation: pt=[lindex $p $ind $i], reset to [expr [lindex $pbounds $i 0] + $delta]"
                  }
               } elseif { [lindex $pbounds $i 1] <= [lindex $p $ind $i]} {
                  lappend temp [expr [lindex $pbounds $i 1] - $delta]
                  if { $debug } {
                     puts "boundary violation: pt=[lindex $p $ind $i], reset to [expr [lindex $pbounds $i 1] - $delta]"
                  }
               } else {
                  lappend temp [lindex [lindex $p $ind] $i]
               }
            }
            lset p $ind $temp

            for {set j 0} {$j < [llength $implicit]} {incr j} {
               if { [llength $implicit] > 0 && [llength $ibounds] != [llength $implicit] } {
                  puts "Number of implicit constraints ([llength $implicit]) and bounds ([llength $ibounds]) do not match!"
                  break
               }
               set consfunc [lindex $implicit $j]
               set violate 0
               if { [$consfunc [lindex $p $ind]] <= [lindex $ibounds $j 0] } {
                  set violate 1
               }
               if { [lindex $ibounds $j 1] <= [$consfunc [lindex $p $ind]] } {
                  set violate 1
               }
 
               if { $violate } {
                  set kt 1
                  set centroid [calcCentroid $ind]
                  if { $debug } {
                     puts "implicit boundary violation (try: [lindex $p $ind], cent: $centroid), move and try again ($countCheck/$maxiterCheck)"
                  }
                  lset p $ind [vecscale 0.5 [vecadd [lindex $p $ind] $centroid]]
               }
            }
            incr countCheck
         }
         if { $countCheck >= $maxiterCheck } {
            return 0
         } else {  
            return 1
         }
      }


      proc calcCentroid { ind } {
         variable n
         variable p
         variable k

         set xc [lindex $p 0]
         for {set j 1} {$j < $k} {incr j} {
            set xc [vecadd $xc [lindex $p $j]]
         }
         set xc [vecscale [expr 1/($k-1.0)] [vecsub $xc [lindex $p $ind]]]
         return $xc
      }
   


      proc get_psum {p} {
	 set psum [lindex $p 0]
	 if {[llength $p]==1} { return $psum }
	 
	 foreach row [lrange $p 1 end] {
	    set psum [vecadd $psum $row]
	 }
	 return $psum
      }

      proc simulated_annealing {{temperature {}}} {
	 if {[llength $temperature]} { variable T $temperature }
	 variable debug
	 variable y 
	 variable yhi
	 variable p 
	 variable psum [get_psum $p]

	 variable ftol
	 variable function
	 variable miny
	 variable maxiter
	 variable iter 0; # Number of function evaluations

	 variable logp 
	 variable logy 
	 variable logT

	 set TINY 1e-10
	 variable ndim [expr {[llength $y]-1}]
	 set mpts [expr {$ndim+1}]
	 while {1} {
	    # First we must determine which point is the highest (worst),
	    # next-highest and lowest.
	    set ilo 0
	    set ihi 1
	    # Every vertex gets a thermal fluctuation added
	    set ylo [expr {[lindex $y 0]-$T*log(rand())}]; # we subtract since log(x)<0 if x<1.
	    set ynhi $ylo
	    set yhi [expr {[lindex $y 1]-$T*log(rand())}]

	    if {$ylo > $yhi} {
	       set ihi 0;
	       set ilo 1;
	       set ynhi $yhi;
	       set yhi $ylo;
	       set ylo $ynhi;
	    }
	    for {set i 2} {$i<$mpts} {incr i} {
	       set yt [expr {[lindex $y $i]-$T*log(rand())}]; # More thermal fluctuation
	       
	       if {$yt <= $ylo} {
		  set ilo $i;
		  set ylo $yt;
	       }
	       if {$yt > $yhi} {
		  set ynhi $yhi;
		  set ihi $i;
		  set yhi $yt;
	       } elseif {$yt > $ynhi} {
		  set ynhi $yt;
	       }
	    }

	    lappend logp [lindex $p $ilo]
	    lappend logy $ylo
	    lappend logT $T


	    # Compute fractional range from highest to lowest and return if satisfactory
	    # Also, if minimum is given check if lowest y is smaller than that.
	    set rtol [expr {2.0*abs($yhi-$ylo)/(abs($yhi)+abs($ylo)+$TINY)}]
	    if {$rtol<$ftol || $iter>=$maxiter || ([llength $miny] && [lindex $y $ilo]<$miny)} { 
	       if {$debug>0} { 
		  if {$rtol<$ftol} {
		     puts "Tolerance converged: rtol<ftol ($rtol<$ftol)"
		  } elseif {$iter>=$maxiter} {
		     puts "Maximum number of iterations ($iter/$maxiter) exceeded (rtol=$rtol, ftol=$ftol)."
		  } else {
		     puts "Lower boundary reached: miny=$miny"
		  }
	       }
	       if {$iter<$maxiter} { variable success 1 }
	       set swap [lindex $y 0]
	       lset y 0 [lindex $y $ilo]
	       lset y $ilo $swap
	       set swap [lindex $p 0];
	       lset p 0 [lindex $p $ilo];
	       lset p $ilo $swap;
	       break
	    }

	    if {$debug>1} {puts "Evaluation $iter: p=$p y=$y; rtol=$rtol"}
	    
	    incr iter 2
	    # Begin a new iteration
	    # First extrapolate by a factor -1 through the face of the simplex across from the 
	    # high point, i.e. reflect the simplex from the high point.
	    set ytry [amotsa -1.0 $ihi]
	    if {$ytry <= $ylo} {
	       # Gives a better result than the best point, so try an additional 
	       # extrapolation by a factor 2
	       set ytry [amotsa 2.0 $ihi]
	    } elseif {$ytry >= $ynhi} {
	       # The reflected point is worse than the second highest, so look for an intermediate 
	       # lower point, i.e., so a one-dimensional contraction.
	       set ysave $yhi
	       set ytry [amotsa 0.5 $ihi]
	       if {$ytry >= $ysave} {
		  # Can't seem to get rid of that high point.
		  # Better contract around the lowest (best) point.
		  set i 0
		  foreach row $p {
		     set j 0
		     if {$i!=$ilo} {
			#foreach val $row {
			 #  set new [expr {0.5*($val+[lindex $p $ilo $j])}]
			  # lset p    $i $j $new
			   #lset psum $j $new
			   #incr j
			#}
			set pnew [vecscale 0.5 [vecadd $row [lindex $p $ilo]]]
			lset p $i $pnew

			if {$debug>2} { puts "contraction $i pnew=$pnew" }
			lset y $i [$function $pnew]
		     }
		     incr i
		  }
		  incr iter $ndim
		  set psum [get_psum $p]
	       }
	    } else { incr iter -1 }
	 }

	 return [list $p $y]
      }

      proc amotsa {fac ihi} {
	 variable p
	 variable psum
	 variable pbounds
	 variable y
	 variable yhi
	 variable ndim
	 variable function
	 variable debug
	 set fac1 [expr {(1.0-$fac)/$ndim}]
	 set fac2 [expr {$fac1-$fac}]
	 set ptry {}

	 set ptry [vecsub [vecscale $psum $fac1] [vecscale $fac2 [lindex $p $ihi]]]

	 # Check parameter boundaries
##	 if {abs($fac)>=1 && [llength $pbounds]} 
	 if {[llength $pbounds]} {
	    foreach bound $pbounds pt $ptry {
	       foreach {lower upper} $bound {}
	       if {([llength $lower] && $pt<$lower) || ([llength $upper] && $pt>$upper)} {
		  # Return something worse than the worst point
##puts "boundary violation: pt=$pt"
###                  return [expr {[lindex $y end]+10000}]
		  return [expr {$yhi+10000}]
	       }
	    }
	 }

	 # Evaluate the function at the trial point
	 set ytry [$function $ptry]

	 variable ybest
	 if {$ytry <= $ybest} { 
	    # Save the best-ever.
	    variable pbest
	    set pbest $ptry
	    set ybest $ytry;
	 }

	 # We added a thermal fluctuation to all the current vertices, but we subtract it here,
	 # so as to give the simplex a thermal Brownian motion: It likes to accept any suggested change.
	 variable T
	 set yflu [expr {$ytry+$T*log(rand())}]; # we add since log(x)<0 if x<1.

	 if {$debug>2} { puts "amotsa $fac: ptry=$ptry ytry=$ytry" }
	 # If it's better than the highest, then replace the highest.
	 if {$yflu<$yhi} {
	    lset y $ihi $ytry
	    set yhi $yflu

	    set psum [vecadd $psum [vecsub $ptry [lindex $p $ihi]]]
	    lset p $ihi $ptry
	 }
	 return $yflu
      }

      proc construct_initial_complex { p0list } {

         variable ndim [llength $p0list]
         variable k [expr $ndim*2]  
         variable p
         variable y
         variable function
         variable implicit
         variable pbounds
         variable ibounds

         lappend p $p0list
         lappend y [$function $p0list]

         for {set i 1} {$i < $k} {incr i} {
            set temp {}
            for {set j 0} {$j < $ndim} {incr j} {
               lappend temp 0
            } 
            lappend p $temp
            lappend y 0
         }
         for {set i 1} {$i < $k} {incr i} {
            set temp {}
            for {set j 0} {$j < $ndim} {incr j} {
               lappend temp [expr [lindex $pbounds $j 0]+rand()*([lindex $pbounds $j 1] - [lindex $pbounds $j 0])]
            }
            lset p $i $temp
            set suc [checkBounds $i]
            lset y $i [$function [lindex $p $i]]
            if { !($suc) } {
               set i [expr $i - 1]
            }
         }
	 return [list $p $y]
      }

      # Constructs an initial simplex from a single vertex
      proc construct_initial_simplex {p0list {scale 0.1}} {
	 variable ndim [llength $p0list] 
	 variable function
	 variable p {}
	 variable y {}
         variable pbounds
         variable delta

         set i 0
         if {[llength $pbounds]} {
            foreach bound $pbounds pt $p0list {
               foreach {lower upper} $bound {}
               if {[llength $lower] && $pt<$lower} {
                  set p0list [lreplace $p0list $i $i [expr $lower + $delta]]
               } elseif {[llength $upper] && $pt>$upper} {
                  set p0list [lreplace $p0list $i $i [expr $upper - $delta]]
               }
            incr i
            }
         }
	 lappend p $p0list
	 lappend y [$function $p0list]

	 set i 0
	 foreach p0 $p0list {
	    set new [expr {$p0+$scale*$p0}]
            # use the bounds to construct the trial point if present
            if {[llength $pbounds] == [llength $p0list]} {
               set bound [lindex $pbounds $i]
               foreach {lower upper} $bound {}
               set new [expr {$lower + rand()*($upper - $lower)}]
            }
            set newp [lreplace $p0list $i $i $new]
	    lappend p $newp
	    lappend y [$function $newp]
	    incr i
	 }
	 return [list $p $y]
      }

      proc analyze {} {
	 variable logp
	 variable logy
	 variable logT
	 variable logysize
	 variable logpavg
	 variable debug

 	 set x {}
 	 set i 0
 	 foreach step $logp {
 	    lappend x $i
 	    incr i
 	 }

	 foreach step $logp {
	    set n 0
	    foreach p $step {
	       lappend param($n) $p
	       incr n
	    }
	 }

	 for {set i 0} {$i<$n} {incr i} {
	    multiplot -y $param($i) -title "Variable $i" -xlabel "Steps" -linecolor red -linewidth 2 -plot
	 }

	 set plot [multiplot -y $logy -title "Function value" -xlabel "Steps" -linecolor red -linewidth 2 -plot]
	 $plot configure -legend "function value y" -plot
	 if {[llength $logT]} {
	    $plot add $x $logT -linecolor green -linewidth 2 -legend "temperatue T" -plot
	 }
	 if {[llength $logysize]} {
	    $plot add $x [vecscale 0.2 $logysize] -linecolor orange -linewidth 2 -legend "ameba size / 5" -plot
	 }
	 if {$debug>2 && [llength $logpavg]} {
	    variable function
	    set yavg {}
	    foreach psum $logpavg {
	       lappend yavg [$function $psum]
	    }
	    $plot add $x $yavg -linecolor magenta -linewidth 2 -legend "ameba center (avg y)" -plot
	 }
      }

   } ; # END namespace $ns

   return "::Optimize::Opt${::Optimize::projectcount}::handle"
}

proc optimization { args } {
   set keyword [lindex $args 0]
   if {![llength $keyword]} { return }
   if {$keyword=="list"} {
      set plist {}
      foreach project [namespace children ::Optimize "Opt*"] { 
	 lappend plist [subst $project]::handle
      }
      return $plist
   } elseif {$keyword=="reset"} {
      foreach projh [namespace children ::Optimize "Opt*"] {
	 namespace delete $projh
      }
      return
   }

   set projecthandle [::Optimize::init]
   #puts "$projecthandle configure $args"
   eval $projecthandle configure $args

   return $projecthandle
}



