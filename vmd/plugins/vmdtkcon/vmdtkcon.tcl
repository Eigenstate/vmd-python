# This file contains a wrapper function to properly load TkCon into VMD.
#
# $Id: vmdtkcon.tcl,v 1.11 2013/04/15 17:56:20 johns Exp $
#

package provide vmdtkcon 1.1

proc vmdtkcon { } {
  # the TKCONDIR environment variable gets set from the pkgIndex.tcl file.
  uplevel #0 {source [file join $env(TKCONDIR) tkcon-modified.tcl]}
  tkcon attach main
  tkcon title "VMD TkConsole"
  
  set ::tkcon::OPT(calcmode)  1
  set ::tkcon::OPT(hoterrors) 0
  
  #TkCon can define aliases (shorthand) to functions
  #alias sel atomselect
    
  return $::tkcon::PRIV(root)
}
