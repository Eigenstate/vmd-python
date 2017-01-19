# remote is a vmd plugin that provides a GUI for remote access
#
# Copyright (c) 2013-2011 The Board of Trustees of the University of Illinois
#
# $Id: remote.tcl,v 1.18 2013/10/17 15:04:57 kvandivo Exp $
#

package provide remote 0.6

namespace eval ::remote:: {
# define all namespace vars here
  # window handle
  variable w                                          

  variable currentUserList
  variable userListMenu

  variable portValue
  variable modeChoice 1
  variable userMenuText

  variable workdir 
  variable globalCounter [pid]

  variable serverSocket

  variable webserverHash
  variable commandHash
  variable buttonGroupHash

  namespace export updateData
# list all exported proc names
#  namespace export name1 name2
}

# -------------------------------------------------------------------------
#
# Create the window and initialize data structures
#
proc ::remote::remote {} {
#  puts "starting in ::remote::remote"
  variable w
  variable userListMenu
  variable userMenuText
  variable portValue
  variable modeChoice
  variable workdir

# let's get a possible URL
  set ip [::http::data [::http::geturl "http://www.ks.uiuc.edu/~kvandivo/getip.html"]]

  setMode    ; # let the C side know that we have set the mode to something

  setInitialWorkdir

# If already initialized, just turn on
  if { [winfo exists .remote] } {
    wm deiconify $w
    return
  }

  set w [toplevel ".remote"]
  wm title $w "Mobile Device Remote Control"

  #Add a menubar
  frame $w.menubar -relief raised -bd 2
  #grid  $w.menubar -padx 1 -column 0 -columnspan 5 -row 0 -sticky ew
  pack $w.menubar -padx 1 -fill x

  menubutton $w.menubar.help -text "Help" -underline 0 \
    -menu $w.menubar.help.menu
  $w.menubar.help config -width 5
  pack $w.menubar.help -side right

  ## help menu
  menu $w.menubar.help.menu -tearoff no
  $w.menubar.help.menu add command -label "About" \
    -command {tk_messageBox -type ok -title "About VMD Remote" \
              -message "Remote Control Gui."}
  $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www]]plugins/remote"

# now, let's define the GUI

## ---------------------------------------------------------------------------
## ------------------------------ START main FRAME ---------------------------
  set win [frame $w.win]
  set row 0

  # -----------------------------------------------
  # intro text
  grid [label $win.introText -text "Mobile Device Remote Control"] \
     -row $row -column 0 -columnspan 2 
  incr row

#  grid [label $win.apiText -text \
#     "Remote API Version Supported: [::mobile get APIsupported]"] \
#     -row $row -column 0 -columnspan 2 
#  incr row

  grid [label $win.ipText -text \
     "This machine's IP (maybe): $ip"] \
     -row $row -column 0 -columnspan 2 
  incr row

#  # -----------------------------------------------
#  # host name, to help people out
#  grid [label $win.hostlabel -text "Host Name: "] \
#    -row $row -column 0 -sticky w
#  grid [label $win.hostvalue -text [info hostname]] \
#    -row $row -column 1 -sticky ew
#  incr row

## copied from vmdmovie
  grid [button $win.workdirbutton -text "Set working directory:" \
    -command "::remote::getworkdir"] -row $row -column 0
  grid [label $win.workdirlabel -textvariable ::remote::workdir ] \
        -row $row -column 1
  incr row

  # -----------------------------------------------
  # Incoming port number
  grid [label $win.portlabel -text "Port Number: "] \
    -row $row -column 0 -sticky w
  grid [entry $win.portValue -width 5 -textvariable \
       [namespace current]::portValue  ] \
    -row $row -column 1 -sticky ew
  incr row

  bind $win.portValue <Return> {::remote::setPort}

  # -----------------------------------------------
  grid [labelframe $win.mode -bd 2 -relief ridge \
            -text "Mode" \
            -padx 1m -pady 1m] -row $row -column 0 -columnspan 2 -sticky nsew
  incr row

  # ---
     grid [radiobutton $win.mode.off -text "Off" \
                   -variable [namespace current]::modeChoice -value 0 -command \
                   "[namespace current]::setMode" ] \
        -row 0 -column 0 -sticky w

     grid [radiobutton $win.mode.move -text "Move" \
                   -variable [namespace current]::modeChoice -value 1 -command \
                   "[namespace current]::setMode" ] \
        -row 0 -column 1 -sticky w

     grid [radiobutton $win.mode.anim -text "Animate" \
                   -variable [namespace current]::modeChoice -value 2 -command \
                   "[namespace current]::setMode" ] \
        -row 0 -column 2 -sticky w

     grid [radiobutton $win.mode.track -text "Tracker" \
                   -variable [namespace current]::modeChoice -value 3 -command \
                   "[namespace current]::setMode" ] \
        -row 0 -column 3 -sticky w

     grid [radiobutton $win.mode.user -text "User" \
                   -variable [namespace current]::modeChoice -value 4 -command \
                   "[namespace current]::setMode" ] \
        -row 0 -column 4 -sticky w

  # -----------------------------------------------
  # Current User List
  grid [labelframe $win.userFrame -bd 2 -text "Connected Clients"       \
                                              -padx 1m -pady 1m]      \
                   -row $row -column 0 -columnspan 2 -sticky nsew
    # -----
    variable listBox
    set listBox [listbox $win.userFrame.userList -activestyle none -yscroll "$win.userFrame.s set" -width 35] 

    bind $listBox <ButtonRelease-1> { ::remote::processClick %W}

    grid $listBox -row 0 -column 0 -sticky news 
    grid [scrollbar $win.userFrame.s -command "$win.userFrame.userList yview"] \
                                -row 0 -column 1 -sticky news
    # -----

  incr row
#
  # -----------------------------------------------
  global vmd_mobile_state_changed
  global vmd_mobile_device_command
  trace add variable vmd_mobile_state_changed write "::remote::updateData"
  trace add variable vmd_mobile_device_command write "::remote::commandReceived"

  #
  configureButtonSets

  updateData

  pack $win 

}

# -------------------------------------------------------------------------
proc ::remote::processClick {W} {
   variable currentUserList

# we only care about this if someone is connected....
   if { [llength $currentUserList] > 0} {

      set userClicked [lindex $currentUserList [$W curselection]]

      # if they were already active, we don't need to do anything
      if { [lindex $userClicked 2] != 1} {
#         puts "sending ::mobile set activeClient $userClicked"
         ::mobile set activeClient [lindex $userClicked 0] [lindex $userClicked 1]
  
         updateData
      }
   }
}

# -------------------------------------------------------------------------
proc ::remote::removeAllButtonsForGroup { group } {
   variable buttonGroupHash
   if { [info exists buttonGroupHash($group)] } {
      unset buttonGroupHash($group)
   } 
}

# -------------------------------------------------------------------------
# -group will typically be the plugin name.  [a-zA-Z]
# -name is the text for the button itself   [a-zA-Z].  Shorter the better
# -tclcode is what you want to execute when button is pressed
# -requireInControl: 1 if requesting mobile device must be "in
# control" to execute code.  0 if requesting mobile device doesn't
# need to be in control.  Typically, if the code is going to modify
# the display on the desktop you will want to require the mobile
# device to be in control.  If the code is just generating output
# for the mobile device, or taking a snapshot or something, you
# might not need to require the user to be in control.
proc ::remote::addButton { group name tclcode requireInControl } {
   variable commandHash
   variable buttonGroupHash

   # save this by the arbitrary hash number so we can get back to
   # it when we get the request from the mobile client
   set msg [list $group $name $tclcode $requireInControl]
   set hashNum [getSingleHashKey commandHash]
   set commandHash($hashNum) $msg

   # now we want to save the pertinent parts in a hash that will
   # get sent to the client
   if { [info exists buttonGroupHash($group)] } {
      set buttons $buttonGroupHash($group)
   } else {
      set buttons [list]
   }
   set buttons [lappend buttons [list $name $hashNum]]
   set buttonGroupHash($group) $buttons

   # testing
#   parray commandHash
#   parray buttonGroupHash

}

# -------------------------------------------------------------------------
proc ::remote::getButtonConfigurationString {} {
   variable buttonGroupHash
   # now we need to send the buttonGroupHash out to the client
  
   # msg format:  # of button groups followed by, for each button group:
   #                buttonGroupName followed by # of buttons, and for
   #                                          each button:
   #                         buttonName ReturnMsg
# so, for:
#   buttonGroupHash(Timeline) = {Get 824242165}
#   buttonGroupHash(vcr)      = {Next 769396887} {Prev 777179245}
   # msg is:
#       2 Timeline 1 Get 824242165 vcr 2 Next 769396887 Prev 777179245

   set strBuild [array size buttonGroupHash]
    
   foreach key [array names buttonGroupHash] {
      set lst $buttonGroupHash($key)
      set strBuild " $strBuild $key [llength $lst]"
      foreach item $lst {
         set strBuild " $strBuild $item"
      }
   }

   return [string trim $strBuild]
} 

# -------------------------------------------------------------------------
proc ::remote::sendButtonConfiguration {} {
   set strBuild [ getButtonConfigurationString ]

   set currentUserList [::mobile get clientList]
#   puts "[clock format [clock seconds]] Sending button conf ($strBuild) to ($currentUserList)"

   # msg type 2 is 'buttons'
   foreach client $currentUserList {
      sendButtonConfigurationToClient [lindex $client 0] [lindex $client 1] [getButtonConfigurationString]
   }
}

# -------------------------------------------------------------------------
proc ::remote::sendButtonConfigurationToClient {clientNick clientIp strConf} {
#  puts "[clock format [clock seconds]] Sending button conf ($strConf) to ($clientNick $clientIp)"
   ::mobile sendMsg $clientNick $clientIp 2 $strConf
}

# -------------------------------------------------------------------------
proc ::remote::setPort {args} {
   variable portValue
   ::mobile port $portValue
}


# -------------------------------------------------------------------------
proc remotegui_tk {} {
  ::remote::remote
  return $::remote::w
}

# -------------------------------------------------------------------------
proc ::remote::commandReceived {args} {
  global vmd_mobile_device_command
  variable commandHash

  set cmd $vmd_mobile_device_command
  # need to decode command

#  puts "recv'd $cmd"

  # first two pieces of info are the nick and IP
  set nick [lindex $cmd 0]
  set ip [lindex $cmd 1]

# third is the basic command.  There might be more after that.. we 
# aren't using them yet, though
  set command [lindex $cmd 2]


  # case's given below correspond to calls to sendSpecificCommand
  # on the java side, with numbers being set by COMMAND_ at top of
  # VMDMobile.java
  switch  $command  {
     "0" {
          takeSnapshot $nick $ip
     }
     "1" { 
          # we need to look into the hash table to determine what we
          # should be doing
          set hashKey [lindex $cmd 3]
#        puts "inside. hashKey $hashKey"
          set cmdSet $commandHash($hashKey)

#puts "hashKey is $hashKey, cmdSet is $cmdSet"
#puts "0: [lindex $cmdSet 0] 1: [lindex $cmdSet 1] 2: [lindex $cmdSet 2] 3: [lindex $cmdSet 3]"
          if { [lindex $cmdSet 3] == 0 } {
          # we now need to run list element 2, if it is allowable
#             puts "1:eval'ing [lindex $cmdSet 2]"
             eval [lindex $cmdSet 2]
          } elseif { [::remote::isInControl $nick $ip] == 1 } {
#             puts "2:eval'ing [lindex $cmdSet 2]"
             eval [lindex $cmdSet 2]
          }
     }
     "2" { 
        sendButtonConfigurationToClient $nick $ip [getButtonConfigurationString]
     }
     "3" {
        display resetview
     }
  }
}
# -------------------------------------------------------------------------
proc ::remote::isInControl { nick ip } {
#   puts "starting isInControl with $nick and $ip"
  set currentUserList [::mobile get clientList]
  foreach client $currentUserList {
#     puts "testing $client against $nick and $ip"
     if { [lindex $client 2] == 1 && $nick == [lindex $client 0] && $ip == [lindex $client 1] } {
             return 1
     } 
  }
  return 0
}
# -------------------------------------------------------------------------
proc ::remote::updateData {args} {
  variable currentUserList
  variable listBox
  variable portValue
  variable modeChoice

  # get the current list of clients....
  set currentUserList [::mobile get clientList]
  set newMode [::mobile get mode]

   # list is a list of lists.  
# user 0
   # user name
   # user IP
   # Is user in control? 1 for yes, 0 for no
# user 1
   # user name
   # user IP
   # Is user in control? 1 for yes, 0 for no
# etc

  $listBox delete 0 end

  if { $newMode == "off" } {
    $listBox insert 0 "  Mode is currently set to Off"
  } else {

    if {[llength $currentUserList] == 0} {
      $listBox insert 0 "No client(s) connected"
    } else {
      foreach client $currentUserList {
         $listBox insert end "[lindex $client 0] ([lindex $client 1])"
      }

# xxx: do we need to set them active?  index 2 will be 1 if yes, 0 if no
      set lbIndex 0
      set anyActive 0
      foreach client $currentUserList {
#         puts "client is $client and lbIndex is $lbIndex"
         if { [lindex $client 2] == 1} {
            set anyActive 1
#            puts "activating"
#            $listBox activate $lbIndex
            $listBox selection set $lbIndex $lbIndex
         }
         incr lbIndex
      }
      if { $anyActive == 0} {
         puts "none active"
      }
    }
  }

  set portValue [::mobile get port]

  switch $newMode {
     off { set modeChoice 0 }
     move { set modeChoice 1 }
     animate { set modeChoice 2 }
     tracker { set modeChoice 3 }
     user { set modeChoice 4 }
  }

# xxx: this isn't the ideal place to trigger this.  It needs to be
# triggered by a request for the button state from the mobile device.
  sendButtonConfiguration
}   ;# end of remote::updateData 

# -------------------------------------------------------------------------
proc ::remote::setMode {} {
   variable modeChoice
   switch $modeChoice {
     0 { ::mobile mode off }
     1 { ::mobile mode move }
     2 { ::mobile mode animate }
     3 { ::mobile mode tracker }
     4 { ::mobile mode user }
  }
}

# -------------------------------------------------------------------------
#
proc ::remote::acceptConnection {channel address port} {

   variable workdir
   variable webserverHash

#    # check for an allowed address
#    if { $address != "127.0.0.1" } {
#puts -nonewline $channel "HTTP/1.0 403 Forbidden\n"
#puts -nonewline $channel "Content-type: text/html\n\n"
#puts $channel "Sorry, you must connect from the local machine."
#close $channel
#puts "Connection attempt from $address denied."
#return
#    }

# xxx: check to make sure address is from someone we trust

    # get the http request
    set request [gets $channel]

    # parse it
    set range 0
    set encoded_req ""
    set req ""
    regexp {GET\s+/?(\S+)} $request range encoded_req
    regsub -all {\%20} $encoded_req " " req

# xxx: check to make sure that request has passed authentication
# by eventually requiring something to be in the request

    if { [info exists webserverHash($req)] } {
      set fInfo $webserverHash($req)
      set deleteAfterDownload [lindex $fInfo 0]
      set filename [lindex $fInfo 1]

      if { [file exist $filename ] } {
        set fp [open $filename rb]
        set data [read $fp]
        close $fp

        set mimeType [getMimetype $filename]

        # return a nice response
        puts -nonewline $channel "HTTP/1.0 200 OK\n"
        puts -nonewline  $channel "Content-type: $mimeType\n\n"
        fconfigure $channel -translation binary -encoding binary
        puts -nonewline $channel $data
        flush $channel
        close $channel

        # xxx: wouldn't want to delete file if it is being sent to multiple
        # clients
        if { $deleteAfterDownload == 1 } {
          file delete $filename
        }
      } else {
        puts -nonewline $channel "HTTP/1.1 404 Object Not Found\n\n"
        flush $channel
        close $channel
      }
    } else {
      puts -nonewline $channel "HTTP/1.1 404 Object Not Found\n\n"
      flush $channel
      close $channel
    }

}

#socket -server accept 2000

# -------------------------------------------------------------------------
proc ::remote::getMimetype { fname } {
   if { [regexp {(?i).BMP$} $fname] } { return "image/bmp" }
   if { [regexp {(?i).PNG$} $fname] } { return "image/png" }
   if { [regexp {(?i).JPG$} $fname] } { return "image/jpeg" }
   if { [regexp {(?i).HTML$} $fname] } { return "text/html" }

   return "text/plain"
}

# -------------------------------------------------------------------------
proc ::remote::getSingleHashKey { var } {

   # clock clicks has better never collide with anything already in
   # the hash
   set num [clock clicks]
   while { [info exists var($num)] } {
      set num [clock clicks]
   }
   return $num
}

# -------------------------------------------------------------------------
proc ::remote::getWwwFileHash { path deleteAfterDownload } {
   variable webserverHash

   set num [getSingleHashKey webserverHash]

   set webserverHash($num) [list $deleteAfterDownload $path]
   return $num
}

# -------------------------------------------------------------------------
proc ::remote::distributeImageToAll { fname title } {
  set currentUserList [::mobile get clientList]
  foreach client $currentUserList {
     distributeFile [lindex $client 0] [lindex $client 1] $fname $title
  }
}

# -------------------------------------------------------------------------
proc ::remote::distributeFile { clientNick clientIp fname title } {
   variable serverSocket
# set up webserver, if not already set up
   if { [info exists serverSocket] == 0} {
      set serverSocket [socket -server ::remote::acceptConnection 5141]
   }

   set msg [list $title [getWwwFileHash $fname 0] ]

   # notify client(s)
   # '1' means that the message is a file that needs to be picked up
   ::mobile sendMsg $clientNick $clientIp 1 $msg
}

# -------------------------------------------------------------------------
proc ::remote::takeSnapshot { clientNick clientIp} {
   variable workdir
   variable globalCounter
   variable serverSocket

# xxx: check to make sure that clientNick and clientIp are
# valid and able to take snapshots

   if { [info exists workdir] == 0} {
      setInitialWorkdir
   }

   set numberToSend $globalCounter
   incr globalCounter

# get the name for the file that we are going to save
# note:  render overwrites files that are there, so 
   set fname "$workdir/snapshot[pid].$numberToSend.bmp"

# render
#   render TachyonInternal $fname
   render snapshot $fname

# let's see if we can make this into a smaller file
   if { [::ExecTool::find convert] != "" } {
      set newfname "$fname.jpg"
      set xmax 1280
      set ymax 740
      set dispsize [display get size]
      set x [lindex $dispsize 0]
      set y [lindex $dispsize 1]
      if {$x > $xmax} {
        set f [expr $xmax*1.0/$x]
        set y [expr $f*$y]
        set x "$xmax"
      }
      set newsize "${x}x${y}"
      ::ExecTool::exec convert $fname -resize "$newsize" $newfname
      set fname $newfname
   }

# set up webserver, if not already set up
   if { [info exists serverSocket] == 0} {
      set serverSocket [socket -server ::remote::acceptConnection 5141]
   }

   set nameToSend [getWwwFileHash $fname 1]
   # notify client(s)
   # '0' means that the message is a snapshot file that needs picked up
   ::mobile sendMsg $clientNick $clientIp 0 $nameToSend
}

# -------------------------------------------------------------------------
##
## Test for file creation capability for work areas
## copied from vmdmovie
##
proc ::remote::testfilesystem {} {
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


# -------------------------------------------------------------------------
##
## Get directory name
## copied from vmdmovie
##
proc ::remote::getworkdir {} {
  variable workdir
  variable newdir

  set newdir [tk_chooseDirectory \
    -title "Choose Directory For Temp Files" \
    -initialdir $workdir -mustexist true]

  if {[string length $newdir] > 0} {
    set workdir $newdir
  }
}

# -------------------------------------------------------------------------
proc ::remote::setInitialWorkdir {} {
  variable workdir

  global env
  if [info exists env(TMPDIR)] {
    set workdir $env(TMPDIR)
  } else {
    switch [vmdinfo arch] {
      WIN64 -
      WIN32 {
        set workdir "c:/"
      }
      MACOSXX86_64 -
      MACOSXX86 -
      MACOSX {
        set workdir "/"
      }
      default {
        set workdir "/tmp"
      }
    }
  }


}
proc ::remote::configureButtonSets {} {
  configureDemoButtons
}
# -------------------------------------------------------------------------
proc ::remote::configureDemoButtons {} {
   addButton Demo LoadDNA { mol new 3bse} 1

   addButton Demo NewRib { mol modstyle 0 top NewRibbons } 1
   addButton Demo BckBone { mol modcolor 0 top Backbone } 1
   addButton Demo DeleteMol { mol delete top } 1
}

# -------------------------------------------------------------------------
proc ::remote::configureRepresentations {} {
   addButton Representations CPK { mol modstyle 0 top CPK } 1
   addButton Representations VDW { mol modstyle 0 top VDW } 1
   addButton Representations NewCar { mol modstyle 0 top NewCartoon } 1
   addButton Representations NewRib { mol modstyle 0 top NewRibbons } 1
   addButton Representations QikSurf { mol modstyle 0 top QuickSurf } 1

}
# -------------------------------------------------------------------------
proc ::remote::configurePlugins {} {

   # No spaces in names!  for now

   # view change renderer
   addButton ViewChangeRenderer Prev { ::VCR::goto_prev } 1
   addButton ViewChangeRenderer Prev0.5 { ::VCR::goto_prev 0.5 } 1
   addButton ViewChangeRenderer Next { ::VCR::goto_next } 1
   addButton ViewChangeRenderer Next0.5 { ::VCR::goto_next 0.5 } 1

   # view master
   addButton ViewMaster Prev { ::ViewMaster::do_restore_prev } 1
   addButton ViewMaster Next { ::ViewMaster::do_restore_next } 1
   addButton ViewMaster SaveView { ::ViewMaster::do_save } 1

   sendButtonConfiguration

}
# -------------------------------------------------------------------------

