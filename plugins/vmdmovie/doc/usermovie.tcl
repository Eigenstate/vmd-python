## User-defined movie frame update callback procedure, invoked every time
## a movie frame is rendered.
proc moviecallback { args } {
  puts "User-defined movie frame update callback frame: $::MovieMaker::userframe
 / $::MovieMaker::numframes"

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

