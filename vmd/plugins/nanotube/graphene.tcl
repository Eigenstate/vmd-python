# Generate graphene sheets
#
# $Id: graphene.tcl,v 1.10 2018/01/08 22:37:14 johns Exp $
#

proc ::Nanotube::graphene_usage { } {
    vmdcon -info "Usage: graphene -lx <length> -ly <length> -type <edge type> \[-nlayers <number of layers>\]  \[-b <0|1>\] \[-a <0|1>\] \[-d <0|1>\] \[-i <0|1>\] \[-cc <blength>\] \[-ma <C-C|B-N>\]"
    vmdcon -info "  <length> is the edge length in nanometers"
    vmdcon -info "  <edge type> is the type of edge (armchair or zigzag)"
    vmdcon -info "  <blength> is the length of a bond in nanometers (default: 0.1418)"    
    vmdcon -info "  <number of layers> is the number of layers of graphene (default: 1)"
    vmdcon -info "  -ma C-C/B-N selects between carbon and boron-nitride (default: C-C)"
    vmdcon -info "  -b 0/1 turns generation of bonds off/on (default: on)"
    vmdcon -info "  -a 0/1 turns generation of angles off/on (default: on)"
    vmdcon -info "  -d 0/1 turns generation of dihedrals off/on (default: on)"
    vmdcon -info "  -i 0/1 turns generation of impropers off/on (default: on)"
    vmdcon -info "  The -a/-d/-i flags only have an effect if -b 1 is used"
}

proc ::Nanotube::graphene_core { args } {
    # Check if proper #arguments was given
    set n_args [llength $args]
    if { [expr fmod($n_args,2)] } {
        vmdcon -err "graphene: wrong number of arguments"
        vmdcon -err ""
        graphene_usage 
        return -1
    }
    if { ($n_args < 6) || ($n_args > 20) } {
        vmdcon -err "graphene: wrong number of arguments"
        vmdcon -err ""
        graphene_usage
        return -1
    }

    # build a full topology by default
    set cmdline(-b) 1
    set cmdline(-a) 1
    set cmdline(-d) 1
    set cmdline(-i) 1 
    set cmdline(-ma) C-C
    set cmdline(-cc) 0.1418

    for { set i 0} {$i < $n_args} {incr i 2} {
        set key [lindex $args $i]
        set val [lindex $args [expr $i + 1]]
        set cmdline($key) $val
    }

    # Check if mandatory options are defined
    foreach a {-lx -ly -type} {
        if { ![info exists cmdline($a)] } {
            vmdcon -err "graphene: required flag '$a' is missing\n"
            graphene_usage
            return -1
        }
    }
  
    if { [info exists cmdline(-nlayers)] } {
        set nlayers $cmdline(-nlayers)
    } else {
        set nlayers 1
    }
    # Set graphene parameters
    set lx $cmdline(-lx)
    set ly $cmdline(-ly)
    set type $cmdline(-type)
    set a [expr {10.0*$cmdline(-cc)}]
    set pi 3.14159265358979323846

    #Check that input is reasonable
    if { $lx <=0 || $ly <= 0 } {
        vmdcon -err "graphene: Edge length must be positive"
        return -1
    }
    if { ($type != "armchair") && ($type != "zigzag")} {
        vmdcon -err "graphene: Type must be either 'armchair' or 'zigzag'"
        return -1
    }

    if {($cmdline(-ma) != {C-C}) && ($cmdline(-ma) != {B-N})} {
        vmdcon -err "graphene: Material (-ma) must be either 'C-C' or 'B-N'"
        return -1
    }

    #Number of unit cells
    if {$type=="armchair"} {
        set Lx_cell [expr 2*$a*sin(60*$pi/180)]
        set Ly_cell [expr 3*$a]
    } else {
        set Lx_cell [expr 3*$a]
        set Ly_cell [expr 2*$a*sin(60*$pi/180)]
    }

    set Nx_cell [expr ceil($lx*10/$Lx_cell)]
    set Ny_cell [expr ceil($ly*10/$Ly_cell)]

    #Index min/max
    set i 0

    #Generate unit cell coordinates
    if {$type=="armchair"} {
        set r1 "0 0 0"
        set r2 "[expr -$a*sin(60*$pi/180)] [expr $a*cos(60*$pi/180)] 0"
        set r3 [vecadd $r2 "0 $a 0"]
        set r4 "0 [expr 2*$a] 0"
        set l_shift "0 $a 0"
    } else {
        set r1 "0 0 0"
        set r2 "[expr -$a*cos(60*$pi/180)] [expr $a*sin(60*$pi/180)] 0"
        set r3 "$a 0 0"
        set r4 [vecadd $r2 "[expr 2*$a] 0 0"]
        set l_shift "[expr $a/2] [expr $a*sin(60*$pi/180)] 0" 
    }

    #Generate graphene coordinates
    set xyzlist {}
    set num_atoms 0
    for {set k 0} { $k < $nlayers} {incr k} {
        for {set j 0} { $j < $Ny_cell } {incr j} {
            for {set i 0} { $i < $Nx_cell } {incr i} {
                set r_shift "[expr $i*$Lx_cell] [expr $j*$Ly_cell] [expr $k*3.35]"
                if {[expr $k%2]!=0} {set r_shift [vecadd $r_shift $l_shift]}
                lappend xyzlist [vecadd $r1 $r_shift] 
                lappend xyzlist [vecadd $r2 $r_shift]
                lappend xyzlist [vecadd $r3 $r_shift]
                lappend xyzlist [vecadd $r4 $r_shift]
                incr num_atoms 4
            }
        }
    }

    #Create new molecule with one frame
    set mol [mol new atoms $num_atoms]
    animate dup $mol
    set sel [atomselect $mol all]
    set asel [atomselect $mol {index % 2 == 0}]
    set bsel [atomselect $mol {index % 2 == 1}]
    set mat Graphene
    #Set default values for all atoms
    if {$cmdline(-ma) == {C-C}} {
        foreach key {name resname segid element type mass radius chain charge} value {C GRA SHT C CA 12.0107 1.7 X 0.0} {
            $sel set $key $value
        }
    } elseif {$cmdline(-ma) == {B-N}} {
        set mat {Boron Nitride}
        foreach key {name resname segid element type mass radius chain charge} value {B BNS SHT B B 10.811 1.7265 X 1.05} {
            $asel set $key $value
        }
        foreach key {name resname segid element type mass radius chain charge} value {N BNS SHT N N 14.0067 1.6825 X -1.05} {
            $bsel set $key $value
        }
    }
    $asel delete
    $bsel delete

    $sel set {x y z} $xyzlist


    #Add representation for molecule
    if {$type=="armchair"} {mol rename $mol "Armchair $mat Sheet"}
    if {$type=="zigzag"} {mol rename $mol "Zigzag $mat Sheet"}

    # only build topology information that is enabled
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
            ::TopoTools::guessimpropers $sel {tolerance 5}
        }
    }
    mol reanalyze $mol
    ::TopoTools::adddefaultrep $mol

    $sel set resid [$sel get serial]
    $sel delete
    return $mol
}

# insert the textmode command variant into the default namespace
interp alias {} graphene {} ::Nanotube::graphene_core
