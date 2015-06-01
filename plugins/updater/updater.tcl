##
## VMD Plugin Updater Version 1.0
##
## A script to find and download new plugins automatically, so that
## VMD users don't have to do it by hand.
##
## Author: John E. Stone
##         johns@ks.uiuc.edu
##         vmd@ks.uiuc.edu
##
## $Id: updater.tcl,v 1.5 2006/03/09 18:55:22 johns Exp $
##
##
## Home Page
## ---------
##   http://www.ks.uiuc.edu/Research/vmd/
##

## Tell Tcl that we're a package and any dependencies we may have
package require http 2.4
package provide updater 1.0

# hack to make starting easier
proc pluginupdater {} {
  variable foobar
  # Don't destroy the main window, because we want to register the window
  # with VMD and keep reusing it.  The window gets iconified instead of
  # destroyed when closed for any reason.
  #set foobar [catch {destroy $::Updater::w  }]  ;# destroy any old windows

  ::Updater::updater   ;# start the plugin updater
}

namespace eval ::Updater:: {
  namespace export pluginupdater
  # window handles
  variable w                         ;# handle to main window
  variable plugindir                 ;# target VMD plugin directory
  variable workdir                   ;# directory to download temp files to
  variable indexfile                 ;# plugin index file
  variable serverurl                 ;# web server URL containing plugin tree
  variable indexurl                  ;# plugin index file at server
}
  
##
## Main routine
## Create the window and initialize data structures
##
proc ::Updater::updater {} {
  variable w
  variable workdir
  variable plugindir
  variable indexurl
  global tcl_platform
  global env

  switch [vmdinfo arch] { 
    WIN64 -
    WIN32 {
      set workdir "c:/temp"
    }
    default {
      set workdir "/tmp"
    } 
  }
  set plugindir    $env(VMDDIR)/plugins
  set serverurl    http://www.ks.uiuc.edu/Research/vmd/plugins
  set indexurl     $serverurl/plugins-[vmdinfo version].idx

  # If already initialized, just turn on
  if { [winfo exists .pluginupdater] } {
    wm deiconify $w
    return
  }

  set w [toplevel ".pluginupdater"]
  # Register this window with VMD so that it appears in the main GUI.
  menu tk add pluginupdater $w
  wm title $w "VMD Plugin Updater" 
  wm resizable $w 0 0

  frame $w.workdir    ;# frame for data entry areas
  button $w.workdir.button -text "Working directory:" \
    -command "::Updater::getworkdir"
  label $w.workdir.label -textvariable ::Updater::workdir
  pack $w.workdir.button $w.workdir.label -side left -anchor w

  ##
  ## Ok/Cancel/Help buttons etc 
  ##
  frame $w.bottom     ;# frame for ok/cancel buttons
  label $w.bottom.label -text "Update Plugins"
  button $w.bottom.ok     -text "Go"      -command "::Updater::updateplugins"
  button $w.bottom.cancel -text "Dismiss" -command "wm withdraw $w"
  button $w.bottom.help   -text "Plugin Updater Help" \
    -command "::Updater::help" 
  pack $w.bottom.label -side top -anchor w
  pack $w.bottom.ok $w.bottom.cancel $w.bottom.help -side left -anchor w

  # pack up all of the frames
  pack $w.workdir $w.bottom -side top -pady 10 -fill x
}

##
## Help for the updater
##
proc ::Updater::help {} {
  set hw [toplevel ".moviehelp"]
  wm title $hw "Plugin Updater Help"
  wm resizable $hw 0 0

  message $hw.msg -width 25c -justify left -relief raised -bd 2 \
          -text "This is a pre-release version of the plugin updater."

  button $hw.ok -text "Ok" -command "destroy $hw"
  pack $hw.msg $hw.ok -side top -pady 10 -fill x
}


##
## Test for file creation capability for work areas
##
proc ::Updater::testfilesystem {} {
  variable workdir;
 
  # test access permissions on working directory
  if {[file isdirectory $workdir] != 1} {
    return 0; # failure 
  }    
  if {[file readable  $workdir] != 1} {
    return 0; # failure 
  }    
  if {[file writable  $workdir] != 1} {
    return 0; # failure 
  }    

  return 1; # success  
}


##
## Get directory name
##
proc ::Updater::getworkdir {} {
  variable workdir
  variable newdir

  set newdir [tk_chooseDirectory \
    -title "Choose working directory for temp files" \
    -initialdir $workdir -mustexist true]

  if {[string length $newdir] > 0} {
    set workdir $newdir 
  } 
}

##
## download files over HTTP
##
# Copy a URL to a file and print meta-data
proc ::Updater::httpcopy { url file {chunk 4096} } {
  set out [open $file w]
  set token [::http::geturl $url -channel $out \
        -progress ::Updater::httpProgress -blocksize $chunk -binary 1]
  close $out
  # This ends the line started by http::Progress
  puts stderr ""
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

  return $token
}

proc ::Updater::httpProgress {args} {
  puts -nonewline stderr . ; flush stderr
}



##
## Update the plugins
##
proc ::Updater::updateplugins {} {
  variable workdir;
  variable indexfile;
  variable indexurl;
  set errcode ""

  # get the plugin index file
  puts "Contacting plugin web server for index file..."
  puts "Downloading index file: $indexurl ..."
  set indexfile [file join $workdir vmd[pid]plugin.idx]
  if { [catch {::Updater::httpcopy $indexurl $indexfile} errcode] } {
    puts "Failed to download plugin index, aborting update process."
    return;
  }

  if {[file exists $indexfile] > 0} {
    puts "Succesfully downloaded index file."
  } else {
    return
  }

  puts "Validating plugin index file..."
  if {[validateindexfile $indexfile]} {
    puts "Processing plugin index file..."
  }

  puts "Deleting working files"
  file delete -force $indexfile
}


##
## Create a list of existing plugins
##
proc ::Updater::listplugins {} {
  set plugintypes {}
  set pluginlist [plugin list]
  foreach {pluginentry} $pluginlist {
    lappend plugintypes [lindex $pluginentry 0]
  }
  set plugintypes [lsort -unique $plugintypes] 
  if {[llength $pluginlist] < 1} {
    return 0
  }
  puts "Plugin types: $plugintypes"

  foreach {plugintype} $plugintypes {
    puts ""
    puts "Plugin type: $plugintype"
    puts "      File Type   Version   Authors"    
    puts "---------------------------------------------------------------"

    set pluginlist [plugin list $plugintype]
    if {[llength $pluginlist] < 1} {
      continue
    }
  
    set pluginnames       {}
    set pluginmajversions {}
    set pluginminversions {}
    set pluginauthors     {}
    foreach {pluginentry} $pluginlist {
      set pluginname [lindex $pluginentry 1]
      lappend pluginnames $pluginname
      plugin info $plugintype $pluginname tmp
      lappend pluginmajversions $tmp(majorversion)
      lappend pluginminversions $tmp(minorversion)
      lappend pluginauthors     $tmp(author)
    }

    for {set i 0} {$i < [llength $pluginnames]} {incr i} {
      set versionstring [format "%15s   %3d.%-3d   %-40s" [lindex $pluginnames $i] [lindex $pluginmajversions $i] [lindex $pluginminversions $i] [lindex $pluginauthors $i]]
      puts $versionstring
    }
  }
}


##
## Fill lists with plugin data
##
proc ::Updater::enumerateplugins { names majversions minversions authors types } {
  upvar $names       pluginnames
  upvar $majversions pluginmajversions
  upvar $minversions pluginminversions
  upvar $authors     pluginauthors
  upvar $types       plugintypelist
 
  set plugintypes {}
  set pluginlist [plugin list]
  foreach {pluginentry} $pluginlist {
    lappend plugintypes [lindex $pluginentry 0]
  }
  set plugintypes [lsort -unique $plugintypes]
  if {[llength $pluginlist] < 1} {
    return 0; # failure
  }

  set pluginnames       {}
  set pluginmajversions {}
  set pluginminversions {}
  set pluginauthors     {}
  set plugintypelist    {}
  foreach {plugintype} $plugintypes {
    set pluginlist [plugin list $plugintype]
    if {[llength $pluginlist] < 1} {
      continue
    }

    foreach {pluginentry} $pluginlist {
      set pluginname [lindex $pluginentry 1]
      lappend pluginnames $pluginname
      plugin info $plugintype $pluginname tmp
      lappend pluginmajversions $tmp(majorversion)
      lappend pluginminversions $tmp(minorversion)
      lappend pluginauthors     $tmp(author)
      lappend plugintypelist    $plugintype
    }
  }

  return 1; # success
}


##
## Create a stub plugin index file listing plugins and versions
## This version doesn't work fully, as VMD doesn't actually store
## enough information to make this workable yet.  However, a few
## simple heuristics help bring it closer to reality.
##
proc ::Updater::makestubindex { filename } {
  if {[::Updater::enumerateplugins names majors minors authors types]} {
    switch [vmdinfo version] {
      1.8 { 
        puts "Building plugin index using VMD 1.8 heuristics" 
      }
      default { 
        puts "Warning: plugin index generator written for VMD 1.8."
        puts "Building plugin index using VMD 1.8 heuristics" 
      }
    }    

    set fd [open $filename w]
    puts $fd "PLUGIN INDEX FILE"
    puts $fd [vmdinfo version]
    for {set i 0} {$i < [llength $names]} {incr i} {
      set pluginname "plugin.so"
      if {[string equal [lindex $types $i] {mol file converter}]} { 
          set pluginname "babelplugin.so"
      } else {
        switch [lindex $names $i] {
          psf      { set pluginname "psfplugin.so" }
          pdb      { set pluginname "pdbplugin.so" }
          dcd      { set pluginname "dcdplugin.so" }
          gro      { set pluginname "gromacsplugin.so" }
          g96      { set pluginname "gromacsplugin.so" }
          trr      { set pluginname "gromacsplugin.so" }
          xtc      { set pluginname "gromacsplugin.so" }
          parm     { set pluginname "parmplugin.so" }
          crd      { set pluginname "crdplugin.so" }
          crdbox   { set pluginname "crdplugin.so" }
          namdbin  { set pluginname "namdbinplugin.so" }
          webpdb   { set pluginname "webpdbplugin.so" }
          grasp    { set pluginname "graspplugin.so" }
          edm      { set pluginname "edmplugin.so" }
          raster3d { set pluginname "r3dplugin.so" }
          rst7     { set pluginname "rst7plugin.so" }
          parm7    { set pluginname "parm7plugin.so" }
          xyz      { set pluginname "xyzplugin.so" }
          cube     { set pluginname "cubeplugin.so" }
          default {
            set pluginname "plugin.so"
          }
        }
      }
      set versionstring [format "%20s  %3d  %3d  %s" [lindex $names $i] \
                        [lindex $majors $i] [lindex $minors $i] \
                        $pluginname]
      puts $fd $versionstring
    } 

    # end the plugin list with sentinel text
    puts $fd "ENDPLUGININDEX"

    close $fd
  }
}


##
## Validate plugin index file
##
proc ::Updater::validateindexfile { indexfilename } {
  if [ catch {open $indexfilename r} idx] {
    puts stderr "Cannot open the file $fileName"
    return 0;  # fail, exit early, no need to close files etc.
  }

  # read first line of index file and check for authenticity
  set firstline [gets $idx]
  if {![string equal $firstline "PLUGIN INDEX FILE"]} {
    puts "Downloaded index file is invalid." 
    close $idx
    return 0;
  }  

  # read VMD version line from index file
  set secondline [gets $idx]
  if {![string equal $secondline [vmdinfo version]]} {
    puts "Index file version doesn't match VMD version." 
    close $idx
    return 0;
  }  
 
  close $idx
  return 1;
}


##
## Fill lists with plugin data
##
proc ::Updater::readindexfile { indexfilename names majversions minversions files } {
  upvar $names       pluginnames
  upvar $majversions pluginmajversions
  upvar $minversions pluginminversions
  upvar $files       pluginfiles

  if [ catch {open $indexfilename r} idx] {
    puts stderr "Cannot open the file $fileName"
    return 0;  # fail, exit early, no need to close files etc.
  }

  # read first line of index file and check for authenticity
  set firstline [gets $idx]
  if {![string equal $firstline "PLUGIN INDEX FILE"]} {
    puts "Downloaded index file is invalid."
    close $idx
    return 0;
  }

  # read VMD version line from index file
  set secondline [gets $idx]
  if {![string equal $secondline [vmdinfo version]]} {
    puts "Index file version doesn't match VMD version."
    close $idx
    return 0;
  }

  # read in plugin version information
  set pluginnames        {}
  set pluginmajversions  {}
  set pluginminversions  {}
  set pluginfiles        {}
  while {![eof $idx]} {
    set entry [gets $idx]
    set checkend ""
    scan $entry "%s" checkend
    if {[string equal $checkend "ENDPLUGININDEX"]} {
      break;
    }
    
    scan $entry "%s %d %d %s" pname pmaj pmin pfile
    lappend pluginnames       $pname 
    lappend pluginmajversions $pmaj
    lappend pluginminversions $pmin
    lappend pluginfiles       $pfile
  }

  close $idx
  return 1;
}


##
## Compare existing plugins with plugins from an index file, giving a 
## lists of plugins which should be updated.
##
proc ::Updater::makeupdatelists { indexfilename names majors minors files } {
  upvar $names    pluginnames
  upvar $majors   pluginmajversions
  upvar $minors   pluginminversions
  upvar $files    pluginfiles

  set pluginnames       {}
  set pluginmajversions {} 
  set pluginminversions {} 
  set pluginfiles       {} 

  if {![enumerateplugins oldnames oldmajors oldminors oldauthors oldtypes]} {
    puts "Failed enumerating built-in plugins."
    return 0; # failure
  }

  if {![readindexfile $indexfilename newnames newmajors newminors newfiles]} { 
    puts "Failed parsing index file."
    return 0; # failure
  } 

  for {set i 0} {$i < [llength $newnames]} {incr i} {
    set name [lindex $newnames $i]
    set oldindex [lsearch $oldnames $name]
    set doupdate 0
    if {[expr $oldindex < 0]} {
      # found an entirely new plugin
      set doupdate 1
    } else {
      # found a plugin similar to one we've already got
      # have to compare version numbers to determine if we want it or not
      if {[lindex $newmajors $i] > [lindex $oldmajors $oldindex]} {
        set doupdate 1
      } elseif {[expr [lindex $newmajors $i] == [lindex $oldmajors $oldindex]]} {
        if {[lindex $newminors $i] > [lindex $oldminors $oldindex]} {
          set doupdate 1
        }
      }
    } 

    if {$doupdate > 0} {
      puts [format "Adding plugin %s" [lindex $newnames  $i]]
      lappend pluginnames       [lindex $newnames  $i]
      lappend pluginmajversions [lindex $newmajors $i]
      lappend pluginminverisons [lindex $newminors $i]
      lappend pluginfiles       [lindex $newfiles  $i]
    }
  }

  return 1; # success
}
