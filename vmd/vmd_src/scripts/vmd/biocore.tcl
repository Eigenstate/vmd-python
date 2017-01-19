############################################################################
#cr
#cr            (C) Copyright 1995-2007 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: biocore.tcl,v $
#       $Author: kvandivo $        $Locker:  $             $State: Exp $
#       $Revision: 1.8 $        $Date: 2009/06/15 21:57:47 $
#
############################################################################
#
# BioCoRE web-based script bootstrap code 
#
# $Id: biocore.tcl,v 1.8 2009/06/15 21:57:47 kvandivo Exp $
#

# http 2.4 provides a -binary option to geturl which is need for downloading
# binary trajectory files.
package require http 2.4

#
# Check for BioCoRE special startup, downloading and executing a 
# bootstrap script from a URL passed in through a BIOCORE_URL 
# environment variable.
#
proc check_biocore {} {
  global env
  global tcl_platform

  if { [catch {set url $env(BIOCORE_URL)} foo] } {
#    puts "Failed to detect BioCoRE URL."
#    puts "Reason: $foo"
    set url ""
  } else {
    set url $env(BIOCORE_URL)
  }

  switch $tcl_platform(platform) {
    windows {
      set tmpfile [file join / vmd[pid]biocore.tcl]
    }
    default {
      set tmpfile [file join $env(TMPDIR) vmd[pid]biocore.tcl]
    }
  }     

  if {[string length $url] > 0} {
    puts "Initiating automatic download of BioCoRE scripts..."
    puts "BioCoRE URL: $url"
    vmdhttpcopy $url $tmpfile
    if {[file exists $tmpfile] > 0} {
      source $tmpfile
      file delete -force $tmpfile
    } else {
      puts "Failed to create temporary BioCoRE script file."
    }
  }
}

# Copy a URL to a file 
proc vmdhttpcopy { url file {chunk 8192} } {
  # let's see if the remote file is available
  set stateArray [::http::geturl $url -validate 1 -timeout 2000]
  set httpCode [::http::code $stateArray]
  set httpStatus [::http::status $stateArray]
  ::http::cleanup $stateArray

  if {[regexp -nocase {ok} $httpCode]} {
     # OK.  It is there
     set out [open $file w]
     puts -nonewline stderr "Starting.."
     set token [::http::geturl $url -channel $out -progress vmdhttpProgress \
                                              -blocksize $chunk -binary 1]
     close $out
     # This ends the line started by http::Progress
     puts stderr " Done!"
     upvar #0 $token state
     set max 0
     foreach {name value} $state(meta) {
       if {[string length $name] > $max} {
         set max [string length $name]
       }
       if {[regexp -nocase ^location$ $name]} {
         # Handle URL redirects
         puts stderr "Location:$value"
         return [copy [string trim $value] $file $chunk]
       }
     }
     incr max

   #  foreach {name value} $state(meta) {
   #    puts [format "%-*s %s" $max $name: $value]
   #  }

     ::http::cleanup $token

  } elseif {[regexp {timeout} $httpStatus]} {
     # Didn't get the HEAD response in the timeout time.  
     puts stderr "$url is not available: Website not responding"
  } else {
     # Got a response, but it wasn't 'ok'
     puts stderr "$url is not available: $httpCode"
  }

}

proc vmdhttpProgress {args} {
  puts -nonewline stderr . ; flush stderr
}

