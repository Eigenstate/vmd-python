# Generate a single wall carbon nanotube
#
# $Id: nanotube.tcl,v 1.18 2014/09/18 20:00:43 johns Exp $
#

proc ::Nanotube::nanotube_usage { } {
    variable ccbond
    vmdcon -info "Usage: nanotube -l <length> -n <n> -m <m> \[-b <0|1>\] \[-a <0|1>\] \[-d <0|1>\] \[-i <0|1>\] \[-cc <blength>\] \[-ma <C-C|B-N>\]"
    vmdcon -info "  <length> is length of nanotube in nanometers"
    vmdcon -info "  <n> and <m> are the chiral indices of the nanotube"
    vmdcon -info "  <blength> is the length of a bond in nanometers (default: 0.1418)"    
    vmdcon -info "  -ma C-C/B-N selects between carbon and boron-nitride (default: C-C)"
    vmdcon -info "  -b 0/1 turns generation of bonds off/on (default: on)"
    vmdcon -info "  -a 0/1 turns generation of angles off/on (default: on)"
    vmdcon -info "  -d 0/1 turns generation of dihedrals off/on (default: on)"
    vmdcon -info "  -i 0/1 turns generation of impropers off/on (default: on)"
    vmdcon -info "  The -a/-d/-i flags only have an effect if -b 1 is used"
}

proc ::Nanotube::nanotube_core { args } {
    # Check if proper #arguments was given
    set n_args [llength $args]
    if { [expr {fmod($n_args,2)}] } { 
        vmdcon -err "nanotube: wrong number of arguments"
        vmdcon -err ""
        nanotube_usage 
        return -1
    }
    if { ($n_args < 6) || ($n_args > 18) } { 
        vmdcon -err "nanotube: wrong number of arguments"
        vmdcon -err ""
        nanotube_usage 
        return -1
    }

    # build a full topology by default
    # N.B.: CHARMM does not use impropers with benzene rings.
    set cmdline(-b) 1
    set cmdline(-a) 1
    set cmdline(-d) 1
    set cmdline(-i) 0
    set cmdline(-ma) C-C
    set cmdline(-cc) 0.1418
   
    for { set i 0} {$i < $n_args} {incr i 2} {
        set key [lindex $args $i]
        set val [lindex $args [expr {$i + 1}]]
        set cmdline($key) $val
    }

    # Check if mandatory options are defined
    if { ![info exists cmdline(-l)] \
             || ![info exists cmdline(-n)] \
             || ![info exists cmdline(-m)] } {
        nanotube_usage
        return -1
    }
  
    # Set nanotube parameters
    set length $cmdline(-l)
    set n $cmdline(-n)
    set m $cmdline(-m)
    set a [expr {10.0*$cmdline(-cc)}]
    set pi 3.14159265358979323846

    #Check that input is reasonable
    if { $n < 0 || $m < 0 || int($n) != $n || int($m) != $m} {
        vmdcon -err "nanotube: n and m must be positive integers"
        return -1
    }
    if {$m==0 && $n==0} {
        vmdcon -err "nanotube: n and m can not both be zero"
        return -1
    }
    if {$length <= 0} {
        vmdcon -err "nanotube: Nanotube length must be a positive value"
        return -1
    }

    if {($cmdline(-ma) != {C-C}) && ($cmdline(-ma) != {B-N})} {
        vmdcon -err "nanotube: Material (-ma) must be either 'C-C' or 'B-N'"
        return -1
    }


    #Calculate greatest common divisor d_R
    set num1 [expr {2*$m + $n}]
    set num2 [expr {2*$n + $m}]
    while { $num1 != $num2 } {
        if { $num1 > $num2 } {
            set num1 [expr {$num1 - $num2}]
        } else { 
            set num2 [expr {$num2 - $num1}] 
        }
    }
    set d_R $num1

    #Compute geometric properties
    set C [expr {$a*sqrt(3*($n*$n + $m*$n + $m*$m))}]
    set R [expr {$C/(2*$pi)}]
    set L_cell [expr {sqrt(3)*$C/$d_R}]

    # compensate for the fact that inscribed polygons have
    # a shorter circumference than a circle, thus we have
    # to fudge the radius of the NT a bit or bonds will be short
    set nchord [expr {2.0*$m*sqrt(3*($n*$n + $m*$n + $m*$m))/sqrt(3*$m*$m)}]
    set rfudge [expr {$pi/$nchord/sin($pi/$nchord)}]

    #Number of unit cells
    set N_cell [expr {ceil($length*10/$L_cell)}]

    #Index min/max
    set pmin 0
    set pmax [expr {int(ceil($n + ($n + 2*$m)/$d_R))}]
    #  set pmax [expr {int(ceil($C*$n + $L_cell*($n + 2*$m)/sqrt(3)))}]
    #  set qmin [expr {int(floor(-$L_cell*(2*$n + $m)/($C*sqrt(3))))}]
    set qmin [expr {int(floor(-(2*$n + $m)/$d_R))}]
    set qmax $m
    set i 0

    #Generate unit cell coordinates
    for {set q $qmin} {$q <= $qmax} {incr q} {
        for {set p $pmin} {$p <= $pmax} {incr p} {

            #First basis atom
            set xprime1 [expr {3.0*$a*$a*($p*(2.0*$n + $m) + $q*($n + 2.0*$m))/(2.0*$C)}]
            set yprime1 [expr {3.0*sqrt(3.0)*$a*$a*($p*$m - $q*$n)/(2.0*$C)}]

            #Second basis atom
            set xprime2 [expr {$xprime1 + 3.0*$a*$a*($n + $m)/(2.0*$C)}]
            set yprime2 [expr {$yprime1 - $a*$a*sqrt(3.0)*($n - $m)/(2.0*$C)}]

            set phi1 [expr {$xprime1/$R}]
            set phi2 [expr {$xprime2/$R}]

            set x1 [expr {$rfudge*$R*cos($phi1)}]
            set x2 [expr {$rfudge*$R*cos($phi2)}]
            set y1 [expr {$rfudge*$R*sin($phi1)}]
            set y2 [expr {$rfudge*$R*sin($phi2)}]
            set z1 $yprime1
            set z2 $yprime2

      #Store coordinates of unit cell in an array
      #   0 <= xprime1 < C and 0 <= yprime1 < L_cell
            if {0 <= $xprime1 \
                    && $p*(2.0*$n + $m) + $q*($n + 2.0*$m) < 2.0*($n*$n + $n*$m + $m*$m) \
                    && 0 <= $yprime1 \
                    && $d_R*($p*$m - $q*$n) < 2.0*($n*$n + $n*$m + $m*$m) } {
                set coord1($i,0) $x1
                set coord1($i,1) $y1
                set coord1($i,2) $z1
        
                set coord2($i,0) $x2
                set coord2($i,1) $y2
                set coord2($i,2) $z2

                incr i
            }
        }
    }

    set num_atom $i
    set k 0

    #Create new molecule with one frame
    set mol [mol new atoms [expr {int(2 * $num_atom * $N_cell + 0.4999)}]]
    animate dup $mol
    set sel [atomselect $mol all]
    set asel [atomselect $mol {index % 2 == 0}]
    set bsel [atomselect $mol {index % 2 == 1}]
    #Set default values for all atoms
    if {$cmdline(-ma) == {C-C}} {
        #Add representation for molecule
        mol rename $mol "($n,$m) Carbon Nanotube"
        foreach key {name resname segid element type mass radius chain charge} value {C CNT TUB C CA 12.0107 1.7 X 0.0} {
            $sel set $key $value
        }
    } elseif {$cmdline(-ma) == {B-N}} {
        #Add representation for molecule
        mol rename $mol "($n,$m) Boron Nitride Nanotube"
        foreach key {name resname segid element type mass radius chain charge} value {B BNT TUB B B 10.811 1.7265 X 1.05} {
            $asel set $key $value
        }
        foreach key {name resname segid element type mass radius chain charge} value {N BNT TUB N N 14.0067 1.6825 X -1.05} {
            $bsel set $key $value
        }
    }
    $asel delete
    $bsel delete

    molinfo $mol set {a b c} [list 100.0 100.0 [expr {$N_cell*$L_cell}]]

    #Generate nanotube coordinates
    set xyzlist {}
    for {set j 0} { $j < $N_cell } {incr j} {
        for {set i 0} {$i < $num_atom} {incr i} {
            lappend xyzlist [list $coord1($i,0) $coord1($i,1) [expr {$coord1($i,2) + $j*$L_cell}]]
            lappend xyzlist [list $coord2($i,0) $coord2($i,1) [expr {$coord2($i,2) + $j*$L_cell}]]
        }
    }
    $sel set {x y z} $xyzlist


    # only build topology information if requested
    if {($cmdline(-b) == "on") || ($cmdline(-b) == 1)} {
        # stash away current radius information for reliable bond searching
        set oldrad [$sel get radius]
        $sel set radius [expr {1.2*$a}]
        mol bondsrecalc $mol
        # restore original radius
        $sel set radius $oldrad

        # set bond types. this will also trigger the flag that
        # the bonds will be written out through molfile.
        ::TopoTools::retypebonds $sel

        if {($cmdline(-a) == "on") || ($cmdline(-a) == 1)} {
            ::TopoTools::guessangles $sel
        }
        if {($cmdline(-d) == "on") || ($cmdline(-a) == 1)} {
            ::TopoTools::guessdihedrals $sel
        }
        if {($cmdline(-i) == "on") || ($cmdline(-a) == 1)} {
            ::TopoTools::guessimpropers $sel {tolerance 45}
        }
    }
    mol reanalyze $mol

    # generate a visualization and have better look at the CNT.
    ::TopoTools::adddefaultrep $mol
    rotate x by -90

    $sel set resid [$sel get serial]
    $sel delete
    return $mol
}

# insert the textmode command variant into the default namespace
interp alias {} nanotube {} ::Nanotube::nanotube_core
