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
# 	$RCSfile: www.tcl,v $
# 	$Author: johns $	$Locker:  $		$State: Exp $
#	$Revision: 1.6 $	$Date: 2007/01/12 20:11:32 $
#
############################################################################
# DESCRIPTION:
#   Click on an atom, update the browser.  This should be best be done
# by hand or as a menu item/hot key (hint, use the invert option).
# Actually, it is now toggled by 'Alt-h' in hotkeys.tcl
#
############################################################################


# By default this is turned OFF! To turn it on by default,
# uncomment the "hyperref on" line below

# turn on/off the hypertext clicking
set vmd_hyperref_info_flag 0

proc hyperref {on_or_off} {
    global vmd_hyperref_info_flag
    if {$on_or_off == "invert"} {
	set on_or_off [expr ! $vmd_hyperref_info_flag]
    }
    global vmd_pick_atom
    # make sure it is only turned on or off once
    if {$on_or_off == "on" || $on_or_off == "1"} {
	if {! $vmd_hyperref_info_flag} {
	    trace variable vmd_pick_atom w vmd_hyperref_update
	    set vmd_hyperref_info_flag 1
	    puts "Hypertext atom picking is turned on"
	}
    } elseif {$on_or_off == "off" || $on_or_off == "0"} {
	trace vdelete vmd_pick_atom w vmd_hyperref_update
	set vmd_hyperref_info_flag 0
	puts "Hypertext atom picking has been turned off"
    } else {
	puts "command is: hyperref <on | off | invert>"
    }
    return
}
# Turn hyper-referencing on by default
# hyperref on

# given a selection text , associate the url
# eg.:  hyperselection top "segname PRO1" http://www/display/pro1.html
proc hyperselection {molid selection_text url} {
    global vmd_hyperref_urls
    set molid [molinfo $molid get id]
    # waste space, easier to do
    set sel [atomselect $molid $selection_text]
    foreach atom [$sel list] {
	set vmd_hyperref_urls($molid,$atom) $url
    }
}

# called when an atom is picked
# tell browser to update
proc vmd_hyperref_update {args} {
    global vmd_mouse_mode vmd_mouse_submode
    # must be in mouse mode 
    global vmd_pick_atom vmd_pick_mol vmd_hyperref_urls
    # find the url
    if {! [info exists vmd_hyperref_urls($vmd_pick_mol,$vmd_pick_atom)] } {
	return
    }
    set url $vmd_hyperref_urls($vmd_pick_mol,$vmd_pick_atom)
    if {$url == {}} {
	return
    }
    vmd_open_url $url
}
