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
# 	$RCSfile: hotkeys.tcl,v $
# 	$Author: johns $	$Locker:  $		$State: Exp $
#	$Revision: 1.12 $	$Date: 2011/01/28 17:10:44 $
#
############################################################################
# DESCRIPTION:
#  These are the default hotkey settings for VMD.  This file
#  is sourced by vmdinit.tcl during startup.
#
############################################################################

# This is sourced by vmdinit.tcl .  In order to not update during
# every call to "user" I turn of the display updating, but I want
# to restore it to its original state, in case this is sourced or somethin
set vmd_hotkeys_tmp [display update status]
display update off

############ Add help messages as the defaults for unassigned keys
foreach char { 1 2 3 4 5 6 7 8 9 0 \
               a b c d e f g h i j k l m n o p q r s t u v w x y z \
               A B C D E F G H I J K L M N O P Q R S T U V W X Y Z } {
  set keycmd [format "user add key $char {puts \"Type 'user add key %s {my VMD commands...}' to use this key\"}" $char]
  eval $keycmd

  foreach modifier { Alt Aux Control } {
    set keycmd [format "user add key $modifier-$char {puts \"Type 'user add key $modifier-$char {my VMD commands...}' to use this key\"}"]
    eval $keycmd
  }
}

# handle special keys
foreach key { Escape Up Down Left Right Page_Up Page_Down \
              Home End Insert Delete } {
  set keycmd [format "user add key $key {puts \"Type 'user add key $key {my VMD commands...}' to use this key\"}"]
  eval $keycmd
}

# handle function keys
foreach key { 1 2 3 4 5 6 7 8 9 10 11 12 } {
  set keycmd [format "user add key F$key {puts \"Type 'user add key F$key {my VMD commands...}' to use this key\"}"]
  eval $keycmd
}



########### VI-LIKE ROTATION KEYS
# rotate down with the 'j' key
user add key j {rotate x by 2}
# rotate up with the 'k' key
user add key k {rotate x by -2}
# rotate left with the 'l' key
user add key l {rotate y by 2}
# rotate right with the 'h' key
user add key h {rotate y by -2}
# rotate couterclockwise with the 'g' key
user add key g {rotate z by 2}
# rotate clockwise with the 'G' key
user add key G {rotate z by -2}

########### EMACS-LIKE ROTATION KEYS
# rotate down with the Cntl-n
user add key Control-n {rotate x by 2}
# rotate up with the Cntl-p
user add key Control-p {rotate x by -2}
# rotate right with the Cntl-f
user add key Control-f {rotate y by 2}
# rotate left with the Cntl-b
user add key Control-b {rotate y by -2}
# No clockwise, counter-clockwise equivalents to VI ; just use G/g

# SCALING KEYS
# make larger with the 'a' key
user add key Control-a {scale by 1.1}
# make smaller with the 'z' key (this prevents conflict with 'z' rotations)
user add key Control-z {scale by 0.9}

# QUIT COMMANDS
user add key Alt-q {quit confirm}
user add key Alt-Q quit

# MENU SHORTCUTS -- they close the open to guarantee the menu is on the top
user add key Alt-M {menu main off ; menu main on}
user add key Alt-f {menu files off ; menu files on}
user add key Alt-g {menu graphics off ; menu graphics on}
user add key Alt-f {menu files off ; menu files on}
user add key Alt-l {menu labels off ; menu labels on}
user add key Alt-r {menu render off ; menu render on}
user add key Alt-d {menu display off ; menu display on}
user add key Alt-c {menu color off ; menu color on}
user add key Alt-t {menu tool off; menu tool on}

# RESET DISPLAY
user add key Control-r {display resetview}
# Emulate SwissPDB Viewer
user add key =         {display resetview}

# FAMILIAR CONTROLS
user add key r {mouse mode rotate}
user add key t {mouse mode translate}
user add key s {mouse mode scale}
user add key p {mouse mode pick}
# query
user add key 0 {mouse mode pick 0}
# center
user add key c {mouse mode pick 1}
# atom
user add key 1 {mouse mode pick 2}
# bond
user add key 2 {mouse mode pick 3}
# angle
user add key 3 {mouse mode pick 4}
# dihedral
user add key 4 {mouse mode pick 5}
# move atom
user add key 5 {mouse mode pick 6}
# move residue
user add key 6 {mouse mode pick 7}
# move fragment
user add key 7 {mouse mode pick 8}
# move molecule
user add key 8 {mouse mode pick 9}
# move highlighted rep
user add key 9 {mouse mode pick 13}
# force atom
user add key % {mouse mode pick 10}
# force residue
user add key ^ {mouse mode pick 11}
# force fragment
user add key & {mouse mode pick 12}

user add key x {rock x by 1 -1}
user add key X {rock x by 1 70}
user add key y {rock y by 1 -1}
user add key Y {rock y by 1 70}
user add key z {rock z by 1 -1}
user add key Z {rock z by 1 70}

# animation contols
user add key + {animate next}
user add key - {animate prev}
user add key . {animate forward}
user add key > {animate forward}
user add key , {animate reverse}
user add key < {animate reverse}
user add key / {animate pause}
user add key ? {animate pause}
user add key \[ {animate goto start}
user add key \] {animate goto end}


# revert the display to its original status
display update $vmd_hotkeys_tmp
unset vmd_hotkeys_tmp

# invert the current hyper text mode
user add key Alt-h {hyperref invert}

