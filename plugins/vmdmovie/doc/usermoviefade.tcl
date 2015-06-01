##
## A very simple example of a user-defined movie script
##
## This example rotates a structure, and fades it in as the movie
## progresses.
##
## To try it out, run the "setupmovie" proc (see the end of this script)
## which loads a structure, displays it with VDW rep, creates and assigns
## a new material, and enables the user-defined movie callback.
## Once that's ready, you can open the movie maker plugin, set the 
## movie type to user-defined, and click go and you should see the script
## put through its' paces.


# fade in a material as the movie progresses
proc dofadein { start end curframe numframes materialname } {
  if { $curframe < $start } {
    material change opacity $materialname 0.0
  } elseif { $curframe >= $start && $curframe <= $end } {
    set ops [expr $curframe - $start]
    set ope [expr $end - $start]
    set opacity [expr $ops / double($ope) ]
    material change opacity $materialname $opacity
  } elseif { $curframe > $end } {
    material change opacity $materialname 1.0
  }
}

## update the display as the movie runs
proc moviecallback { args } {
  puts "User-defined movie frame update callback frame: $::MovieMaker::userframe
 / $::MovieMaker::numframes"

  dofadein 48 96 $::MovieMaker::userframe $::MovieMaker::numframes Fade
  rotate y by 5
}

## Easy-to-use proc to enable the user-defined movie frame callback
proc enablemoviecallback { }  {
  trace add variable ::MovieMaker::userframe write moviecallback
}

## Easy-to-use proc to disable the user-defined movie frame callback
proc disablemoviecallback { } {
  trace remove variable ::MovieMaker::userframe write moviecallback
}

## setup for making the movie
proc setupmovie { } {
  mol delete all
  mol new 1ap9

  material add Fade copy Transparent
  mol modstyle 0 0 VDW 1.000000 8.000000
  mol modmaterial 0 0 Fade

  enablemoviecallback

  after idle { puts "Don't forget to set movie type to user-defined in the Movie Settings" }
}  

## disable movie callback, any other cleanup....
proc finishmovie {} {
  material delete Fade
  disablemoviecallback
}







