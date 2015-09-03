##
## MultiText 1.0
##
## A script to view/edit textual (molecule) info via a simple Tk interface
##
## Author: Kirby Vandivort/John Stone
##         kvandivo@ks.uiuc.edu
##         johns@ks.uiuc.edu
##         vmd@ks.uiuc.edu
##
## $Id: multitext.tcl,v 1.2 2009/02/04 19:03:03 kvandivo Exp $
##

#
# Skeleton of plugin that can have multiple, independent instances
#
#  Multi code stolen shamelessly from Jan Saam's Multiplot plugin
# You can have several independent instances of "multiText" runnning at the 
# same time because the data are kept in different namespaces.
# It returns a instancehandle which you can use to control an existing instance.

# Usage:
# set instancehandle [multitext ?reset|list? ?options?]
#    reset --- Ends instances and deletes all namespaces and instancehandles
#    list  --- Returns list of all existing instancehandles
#  no 'options' currently exist

# You can use the returned instancehandle to control the instance:
# $instancehandle namespace|configure|quit ?options?

# $instancehandle namespace --Returns the current namespace
# $instancehandle newfile   --Clean out text window and start over
# $instancehandle openfile 'arg' --Inserts text from arg filename
# $instancehandle text 'string' --Inserts textstring in arg variable/literal
# $instancehandle getWindowHandle --Get wm-ready handle, useful for Binding
# $instancehandle quit      --Destroy the instance, delete all data etc


# Examples:
# ---------
#   package require multitext

#   set instancehandle [multitext ]

### Now we configure the instance
#   $instancehandle text "This is the text"

#     or

#   $instancehandle openfile "/tmp/filename.txt"

### Close the instance/destroy window
#   $instancehandle quit

package provide multitext 1.0

namespace eval ::MultiText:: {
   proc initialize {} {
      variable instancecount -1
   }
   initialize
}

proc ::MultiText::init_instance { } {
   incr ::MultiText::instancecount
   set ns "::MultiText::Instance${::MultiText::instancecount}"

#   if {[namespace exists $ns]} {
#      puts "Reinitializing namespace $ns."
#   } else {
#      puts "Creating namespace $ns"
#   }

   namespace eval $ns {
      variable namespace ::instancehandle${::MultiText::instancecount}
      variable w .multitextinstancewindow${::MultiText::instancecount}
      variable txtFileName "untitled.txt"

      # ------------------------------------------------------------
      # create main window frame 
      catch {destroy $w} 
      toplevel $w
      wm title $w "Text Editor: $txtFileName"
      wm resizable $w 0 0
      wm protocol $w WM_DELETE_WINDOW "[namespace current]::instancehandle quit"

      #menu $w.menu -tearoff 0
      #menu $w.menu.file -tearoff 0
      #$w.menu add cascade -label "File" -menu $w.menu.file -underline 0
      
      #menu $w.menu.file.open -tearoff 0
      
      #$w.menu.file add command -label "Quit" -command "[namespace current]::instancehandle quit"
      #$w configure -menu $w.menu

      ##
      ## make the menu bar
      ##
      frame $w.menubar -relief raised -bd 2 ;# frame for menubar
      pack $w.menubar -padx 1 -fill x

      menubutton $w.menubar.help -text Help -underline 0 -menu $w.menubar.help.menu
      menubutton $w.menubar.file -text File -underline 0 -menu $w.menubar.file.menu

      ##
      ## help menu
      ##
      menu $w.menubar.help.menu -tearoff no
      $w.menubar.help.menu add command -label "Help..." -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/multitext"
      # XXX - set menubutton width to avoid truncation in OS X
      $w.menubar.help config -width 5

      menu $w.menubar.file.menu -tearoff no
      $w.menubar.file.menu add command -label "New" -command  [namespace current]::newfile
      $w.menubar.file.menu add command -label "Open" -command [namespace current]::loadfile
      $w.menubar.file.menu add command -label "Save" -command  [namespace current]::savefile
      $w.menubar.file.menu add command -label "Save As" -command  [namespace current]::saveasfile
      $w.menubar.file.menu add command -label "Print instancehandle in console" -command "puts [namespace current]::instancehandle"
      $w.menubar.file config -width 5
      pack $w.menubar.file -side left
      pack $w.menubar.help -side right


      ##
      ## main window area
      ## 
      frame $w.txt
      label $w.txt.label -width 80 -relief sunken -bg White -textvariable \
                                            [namespace current]::txtFileName
      text $w.txt.text -bg White -bd 2 -yscrollcommand "$[namespace current]::w.txt.vscr set"
      scrollbar $w.txt.vscr -command "$[namespace current]::w.txt.text yview"
      pack $w.txt.label 
      pack $w.txt.text $w.txt.vscr -side left -fill y
 
      pack $w.menubar $w.txt


      # ------------------------------------------------------------
      # Create a instancehandle procedure that provides some commands to control
      # the instance.  This procedure is how you interact with the instance.
      # Its full name will be returned when you invoke MultiText.
      proc instancehandle { command args } {
#         puts "in instancehandle { command = $command args = $args }"
         variable w
         switch $command {
# add commands here that you want to be able to do, using 'args' to get values
#           randomCommand { 
               # handle args
               # return values as needed
#           }

            # a couple of standard commands
            namespace { return [namespace current] }
            quit   { 
# has everything been saved?
               destroy $w
               namespace delete [namespace current]
               return
            }
            newfile {
               [namespace current]::newfile
            }
            openfile {
               [namespace current]::openfile $args
            }
            getWindowHandle {
               return $w
            }
            text {
               [namespace current]::importtext [join $args]
            }
         }
      } ; #end of instancehandle

      # ------------------------------------------------------------
      proc newfile { } {
        variable w
        variable txtFileName

       $w.txt.text delete 1.0 {end - 1c}

        set txtFileName "untitled.txt"
        wm title $w "Text Editor: $txtFileName"
      }

      # ------------------------------------------------------------
      proc openfile { fname } {
         variable txtFileName
         variable w

         if {[catch {open $fname "r"} fd] } {
           set [namespace current]::txtFileName "untitled.txt"
           return
         }

         newfile

         set txtFileName $fname
        wm title $w "Text Editor: $txtFileName"

         set line ""
         while {[gets $fd line] != -1} {
           set dtext "$line\n"
           $w.txt.text insert end $dtext
         } 

         close $fd
      }

      # ------------------------------------------------------------
      proc loadfile { } {
        variable w
        variable txtFileName

        set file_types {
          {"Tcl Files" { .tcl .TCL .tk .TK} }
          {"Text Files" { .txt .TXT} }
          {"All Files" * }
        }

        set txtFileName [tk_getOpenFile -filetypes $file_types \
                -initialdir pwd -initialfile $txtFileName ]

        openfile $txtFileName
      }

      # ------------------------------------------------------------
      proc importtext { txt } {
        variable w
        variable txtFileName

        set txtFileName "untitled.txt"
        wm title $w "Text Editor: $txtFileName"

        newfile
        $w.txt.text insert end $txt
      }

      # ------------------------------------------------------------
      proc savefile { } {
        variable w
        variable txtFileName

#        puts "Saving to file $txtFileName"
        if {[catch {open $txtFileName "w"} fd] } {
          tk_dialog .errmsg {Text Editor Error} "Failed to write file $txtFileName\n$fd" error 0 Dismiss
          return
        }

        puts $fd [$w.txt.text get 1.0 {end -1c}]

        close $fd
      }

      # ------------------------------------------------------------
      proc saveasfile { } {
        variable w
        variable txtFileName

        set file_types {
          {"Tcl Files" { .tcl .TCL .tk .TK} }
          {"Text Files" { .txt .TXT} }
          {"All Files" * }
        }

        set txtFileName [tk_getSaveFile \
                  -initialdir pwd -initialfile $txtFileName ]

        wm title $w "Text Editor: $txtFileName"
        if {[catch {open $txtFileName "w"} fd] } {
          tk_dialog .errmsg {Text Editor Error} "Failed to open file $txtFileName\n$fd" error 0 Dismiss
          return
        }

        puts $fd [$w.txt.text get 1.0 {end -1c}]

        close $fd
      }
      # ------------------------------------------------------------

   } ; # END namespace $ns

   return "::MultiText::Instance${::MultiText::instancecount}::instancehandle"
}

proc multitext { args } {
#   puts "begin: multitext { args }"
   set keyword [lindex $args 0]
#   if {![llength $keyword]} { return }
   if {$keyword=="list"} {
      set plist {}
      foreach instance [namespace children ::MultiText "Instance*"] { 
         lappend plist [subst $instance]::instancehandle
      }
      return $plist
   } elseif {$keyword=="reset"} {
      foreach instanceh [namespace children ::MultiText "Instance*"] {
         destroy $[subst $instanceh]::w;
         namespace delete $instanceh
      }
      return
   }

#   puts "before init_instance in multitext { args }"

   set instancehandle [::MultiText::init_instance]
   eval $instancehandle configure $args

   return $instancehandle
}

proc multitext_tk {} {
  set ih [ multitext ]
  return [$ih getWindowHandle]
}



