
# Set SCRIPTDIR to wherever you install this package
set SCRIPTDIR /home/justin/projects/psfgen/scripts
source $SCRIPTDIR/build.tcl
source $SCRIPTDIR/minimize.tcl

proc buildmin { pdb { buildname NONE } { outputname NONE } } {
  if { ![string compare $buildname NONE]} {
    set buildname "[lindex [split $pdb '.'] 0]_build"
  }

  build $pdb $buildname
  
  if { ![string compare $outputname NONE]} {
    set outputname "[lindex [split $pdb '.'] 0]_min"
  }

  minimize "${buildname}.psf" "${buildname}.pdb" $outputname 100
}

 
