#==============================================================================
# $Id: vmdprefs.tcl,v 1.11 2014/12/17 21:42:07 mayne Exp $
#==============================================================================
#
# VMD Prefs - A preferences plugin for customizing the VMD environment
#
# Authors:
#   Christopher G. Mayne
#   Tajkhorshid Laboratory (http://csbmb.beckman.illinois.edu/)
#   Beckman Inistitute for Advanced Science and Techonology
#   University of Illinois, Urbana-Champaign
#   cmayne2@illinois.edu
#   http://www.ks.uiuc.edu/~mayne
#
# Usage:
#   VMD Prefs is designed to be used through the provided GUI,
#   launched from the Extensions menu.
#
# Documentation:
#   http://www.ks.uiuc.edu/Research/vmd/plugins/vmdprefs
#
#==============================================================================

#=============================
# package setup
package provide vmdprefs 1.0
#=============================
namespace eval ::vmdPrefs {
    
    variable w

    # Load/Save    
    #-----------
    variable loadPath
    variable savePath
    variable useTheme
    
    # Tabs
    #------
    variable settings
    
    # Menus
    # <menu variables will be declared here>
    variable editMenusPluginsLbl ""
    variable editMenusPluginsOnOff ""
    variable editMenusPluginsX ""
    variable editMenusPluginsY ""
    
    
    # Colors
    # <color variables will be declared here>

    # Custom
    # <custom variables will be declared here>
    variable customDesc
}
#=============================
proc vmdPrefs {} {
    # global cmd for launching the gui
    return [eval ::vmdPrefs::gui]
}
#=============================
proc ::vmdPrefs::gui {} {

    # localize/initialize necessary variables
    # (will be filled in as we progress, as necessary)

    variable w
    
    # setup style elements
    # (we'll deal with this later)
    
    # commonly used symbols
    set upArrow [format "%c" 8593]
    set downArrow [format "%c" 8595]
    set accept [format "%c" 10003]
    set cancel [format "%c" 10005]
    
    # initialization procedure
    ::vmdPrefs::init
    
    # setup the GUI window
    if { [winfo exists .vmdPrefs] } {
        wm deiconify .vmdPrefs
        return
    }
    
    set w [toplevel ".vmdPrefs"]
    wm title $w "VMD Preferences Panel"
    
    # allow .vmdPrefs to expand/contract with the window
    grid columnconfigure $w 0 -weight 1
    grid rowconfigure $w 0 -weight 1
    
    # set a default initial geometry
    wm geometry $w 710x620
    
    # build a high level frame (hlf)
    # everything will be stored inside this frame
    ttk::frame $w.hlf
    grid $w.hlf -column 0 -row 0 -sticky nswe
    # allow the hlf to expand/contract appropriately
    grid columnconfigure $w.hlf 0 -weight 1
    grid rowconfigure $w.hlf {4} -weight 1
    
    #----------------------
    # header panel
    #----------------------
    
    # build the load/save panel
    ttk::frame $w.hlf.loadSave -padding 4

    ttk::label $w.hlf.loadSave.loadLbl -text "Load Theme:" -anchor e
    ttk::entry $w.hlf.loadSave.loadPath -textvariable ::vmdPrefs::loadPath -width 44 -justify left
    ttk::button $w.hlf.loadSave.loadBrowse -text "Browse" \
        -command {
            set tempfile [tk_getOpenFile -title "Select VMDPrefs Theme File"]
            if {![string eq $tempfile ""]} { set ::vmdPrefs::loadPath $tempfile }
        }
    ttk::button $w.hlf.loadSave.load -text "Load" -command { ::vmdPrefs::readThemeFile $::vmdPrefs::loadPath }
    
    ttk::label $w.hlf.loadSave.saveLbl -text "Save Theme:" -anchor e
    ttk::entry $w.hlf.loadSave.savePath -textvariable ::vmdPrefs::savePath -width 44 -justify left
    ttk::button $w.hlf.loadSave.saveAs -text "Save As" \
        -command {
            set tempfile [tk_getSaveFile -title "Select VMDPrefs Theme File"]
            if {![string eq $tempfile ""]} { set ::vmdPrefs::savePath $tempfile }
        }
    ttk::button $w.hlf.loadSave.write -text "Write" -command { ::vmdPrefs::writeThemeFile $::vmdPrefs::savePath }    
        
    # grid the load/save panel
    grid $w.hlf.loadSave -column 0 -row 0 -sticky nswe
    
    grid $w.hlf.loadSave.loadLbl -column 0 -row 0 -sticky nswe
    grid $w.hlf.loadSave.loadPath -column 1 -row 0 -sticky nswe
    grid $w.hlf.loadSave.loadBrowse -column 2 -row 0 -sticky nswe
    grid $w.hlf.loadSave.load -column 3 -row 0 -sticky nswe
    
    grid $w.hlf.loadSave.saveLbl -column 0 -row 1 -sticky nswe
    grid $w.hlf.loadSave.savePath -column 1 -row 1 -sticky nswe
    grid $w.hlf.loadSave.saveAs -column 2 -row 1 -sticky nswe
    grid $w.hlf.loadSave.write -column 3 -row 1 -sticky nswe
    
    # configure the loadSave panel
    # columns
    # set the entry boxes to expand, all others are fixed
    grid columnconfigure $w.hlf.loadSave 1 -weight 1
    grid columnconfigure $w.hlf.loadSave {0 2 3} -weight 0
    # make buttons same width
    grid columnconfigure $w.hlf.loadSave {2 3} -uniform ct1
    # rows
    # give row heights the same fixed value
    grid rowconfigure $w.hlf.loadSave {0 1} -weight 0 -uniform rt1    
    
    # build / grid a separator
    ttk::separator $w.hlf.sep1 -orient horizontal
    grid $w.hlf.sep1 -column 0 -row 1 -sticky nswe -pady 2 -padx 2


    # build an control button panel
    ttk::frame $w.hlf.controlButtons
    ttk::button $w.hlf.controlButtons.writeVMDRC -text "Write Settings to VMDRC" -command { ::vmdPrefs::writeVMDRC }
    ttk::button $w.hlf.controlButtons.queryAll -text "Query All VMD Settings" -command { ::vmdPrefs::queryAllSettings }
    ttk::button $w.hlf.controlButtons.pushAll -text "Push All Settings to VMD" -command { ::vmdPrefs::pushAllSettings }

    # grid the control button panel
    grid $w.hlf.controlButtons -column 0 -row 2 -sticky ns
    grid $w.hlf.controlButtons.writeVMDRC -column 0 -row 0 -sticky nswe
    grid $w.hlf.controlButtons.queryAll -column 1 -row 0 -sticky nswe
    grid $w.hlf.controlButtons.pushAll -column 2 -row 0 -sticky nswe

    # configure the control button panel
    grid columnconfigure $w.hlf.controlButtons {0 1 2} -uniform ct1

    # build / grid a separator
    ttk::separator $w.hlf.sep2 -orient horizontal
    grid $w.hlf.sep2 -column 0 -row 3 -sticky nswe -pady "4 0" -padx 2    

    #----------------------
    # tabbed panel (notebook)
    #----------------------
    
    # build/grid the notebook
    ttk::notebook $w.hlf.nb
    grid $w.hlf.nb -column 0 -row 4 -sticky nswe
    
    #----------------------
    # menus
    #----------------------
    
    # add a tab for startup menus
    ttk::frame $w.hlf.nb.menus
    $w.hlf.nb add $w.hlf.nb.menus -text "Menus"
    
    # build
    ttk::treeview $w.hlf.nb.menus.tv -selectmode browse -yscroll "$w.hlf.nb.menus.scroll set"
        $w.hlf.nb.menus.tv configure -column {name on_off locX locY} -displaycolumns {name on_off locX locY} -show {tree headings} -height 10
        $w.hlf.nb.menus.tv column #0 -width 75 -stretch 0 -anchor center
        $w.hlf.nb.menus.tv heading name -text "Menu" -anchor center
        $w.hlf.nb.menus.tv column name -width 150 -stretch 1 -anchor center
        $w.hlf.nb.menus.tv heading on_off -text "On/Off" -anchor center \
            -command {
                if { [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] != -1 } {
                    set firstState [.vmdPrefs.hlf.nb.menus.tv set [lindex [.vmdPrefs.hlf.nb.menus.tv children [.vmdPrefs.hlf.nb.menus.tv selection]] 0] on_off]
                    set newState [expr ( $firstState eq "on" ) ? "off" : "on"]
                    foreach ele [.vmdPrefs.hlf.nb.menus.tv children [.vmdPrefs.hlf.nb.menus.tv selection]] {
                        .vmdPrefs.hlf.nb.menus.tv set $ele on_off $newState
                    }
                    unset firstState newState
                } else { return }
            }
        $w.hlf.nb.menus.tv column on_off -width 100 -stretch 0 -anchor center
        $w.hlf.nb.menus.tv heading locX -text "X" -anchor center \
            -command {
                if { [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] != -1 } {
                    foreach ele [.vmdPrefs.hlf.nb.menus.tv children [.vmdPrefs.hlf.nb.menus.tv selection]] {
                        .vmdPrefs.hlf.nb.menus.tv set $ele locX 0
                    }
                } else { return }
            }
        $w.hlf.nb.menus.tv column locX -width 75 -stretch 0 -anchor center
        $w.hlf.nb.menus.tv heading locY -text "Y" -anchor center \
            -command {
                if { [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] != -1 } {
                    foreach ele [.vmdPrefs.hlf.nb.menus.tv children [.vmdPrefs.hlf.nb.menus.tv selection]] {
                        .vmdPrefs.hlf.nb.menus.tv set $ele locY 0
                    }
                } else { return }
            }
        $w.hlf.nb.menus.tv column locY -width 75 -stretch 0 -anchor center
    ttk::scrollbar $w.hlf.nb.menus.scroll -orient vertical -command "$w.hlf.nb.menus.tv yview"
    ttk::button $w.hlf.nb.menus.showHide -text "Show/Hide" \
        -command {
            if { [.vmdPrefs.hlf.nb.menus.tv selection] != {} && [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] == -1 } {
                set name [ .vmdPrefs.hlf.nb.menus.tv set [.vmdPrefs.hlf.nb.menus.tv selection] name ]
                expr { ( [menu $name status] eq "on" ) ? [menu $name off] : [menu $name on] }
                unset name
            } else { return }
        }
    ttk::button $w.hlf.nb.menus.toggleOnOff -text "Toggle On/Off" \
        -command {
            if { [.vmdPrefs.hlf.nb.menus.tv selection] != {} && [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] == -1 } {
                set id [.vmdPrefs.hlf.nb.menus.tv selection]
                set currState [.vmdPrefs.hlf.nb.menus.tv set $id on_off]
                set newState [expr ( $currState eq "on" ) ? "off" : "on"]
                .vmdPrefs.hlf.nb.menus.tv set $id on_off $newState
                unset id currState newState
            } else { return }
        }
    ttk::button $w.hlf.nb.menus.getXY -text "Get (X,Y)" \
        -command {
            if { [.vmdPrefs.hlf.nb.menus.tv selection] != {} && [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] == -1 } {
                set id [.vmdPrefs.hlf.nb.menus.tv selection]
                set name [.vmdPrefs.hlf.nb.menus.tv set $id name]
                .vmdPrefs.hlf.nb.menus.tv set $id locX [lindex [menu $name loc] 0]
                .vmdPrefs.hlf.nb.menus.tv set $id locY [lindex [menu $name loc] 1]
                unset id name
            } else { return }
        }
    ttk::button $w.hlf.nb.menus.resetXY -text "Reset (X,Y)" \
        -command {
            if { [.vmdPrefs.hlf.nb.menus.tv selection] != {} && [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] == -1 } {
            .vmdPrefs.hlf.nb.menus.tv set [.vmdPrefs.hlf.nb.menus.tv selection] locX 0
            .vmdPrefs.hlf.nb.menus.tv set [.vmdPrefs.hlf.nb.menus.tv selection] locY 0
            } else { return }
        }
    ttk::separator $w.hlf.nb.menus.sep1 -orient horizontal
    ttk::button $w.hlf.nb.menus.queryAll -text "Query Menus" -command { ::vmdPrefs::menusQuery }
    ttk::button $w.hlf.nb.menus.pushAll -text "Push Menus" -command { ::vmdPrefs::menusPush }

    
    # grid
    grid $w.hlf.nb.menus.tv -row 0 -column 0 -rowspan 8 -sticky nswe
    grid $w.hlf.nb.menus.scroll -row 0 -column 1 -rowspan 8 -sticky nswe

    grid $w.hlf.nb.menus.queryAll -row 0 -column 2 -sticky nswe
    grid $w.hlf.nb.menus.pushAll -row 1 -column 2 -sticky nswe
    grid $w.hlf.nb.menus.sep1 -row 2 -column 2 -sticky nswe -pady 4
    grid $w.hlf.nb.menus.showHide -row 3 -column 2 -sticky nswe
    grid $w.hlf.nb.menus.toggleOnOff -row 4 -column 2 -sticky nswe
    grid $w.hlf.nb.menus.getXY -row 5 -column 2 -sticky nswe
    grid $w.hlf.nb.menus.resetXY -row 6 -column 2 -sticky nswe


    # configure rows/columns
    grid rowconfigure $w.hlf.nb.menus {0 1 3 4 5 6} -weight 0 -uniform rt1
    grid rowconfigure $w.hlf.nb.menus 7 -weight 1
    grid columnconfigure $w.hlf.nb.menus 0 -weight 1 -minsize 150
        
    # bindings
    bind $w.hlf.nb.menus.tv <KeyPress-Escape> { .vmdPrefs.hlf.nb.menus.tv selection set {} }
    bind $w.hlf.nb.menus.tv <Double-Button-1> {
            if { [.vmdPrefs.hlf.nb.menus.tv selection] != {} && [lsearch [.vmdPrefs.hlf.nb.menus.tv children {}] [.vmdPrefs.hlf.nb.menus.tv selection]] == -1 } {
                set id [.vmdPrefs.hlf.nb.menus.tv selection]
                set currState [.vmdPrefs.hlf.nb.menus.tv set $id on_off]
                set newState [expr ( $currState eq "on" ) ? "off" : "on"]
                .vmdPrefs.hlf.nb.menus.tv set $id on_off $newState
                unset id currState newState
            } else { return }
    }
    
    # load data (initializes to current values)
    # priority menus (commonly used menus that we think should be highlighted)
    .vmdPrefs.hlf.nb.menus.tv insert {} end -id common -text "Common"
    .vmdPrefs.hlf.nb.menus.tv item common -open 1
    set priorityMenus {main graphics tkcon files}
    foreach ele $priorityMenus {
        set status [menu $ele status]
        lassign [menu $ele loc] x y
        # menu x,y of 0,0 is undefined
        if { $x == 0 && $y == 0 } {
            .vmdPrefs.hlf.nb.menus.tv insert common end -id $ele -values [list $ele $status {} {}]
        } else {
            .vmdPrefs.hlf.nb.menus.tv insert common end -id $ele -values [list $ele $status $x $y]
        }
        unset status x y
    }

    # all other (non-priority) menus
    .vmdPrefs.hlf.nb.menus.tv insert {} end -id other -text "Other"
    .vmdPrefs.hlf.nb.menus.tv item other -open 0
    foreach ele [lsort -dictionary [menu list]] {
        if { [lsearch -exact $priorityMenus $ele] == -1 } {
            set status [menu $ele status]
            lassign [menu $ele loc] x y
            # menu x,y of 0,0 is undefined
            if { $x == 0 && $y == 0 } {
                .vmdPrefs.hlf.nb.menus.tv insert other end -id $ele -values [list $ele $status {} {}]
            } else {
                .vmdPrefs.hlf.nb.menus.tv insert other end -id $ele -values [list $ele $status $x $y]
            }
            unset status x y
        }
    }
    

    #----------------------
    # display
    #----------------------
    
    # add a tab for display
    ttk::frame $w.hlf.nb.display
    $w.hlf.nb add $w.hlf.nb.display -text "Display"
    
    grid columnconfigure $w.hlf.nb.display {0 2} -weight 1 -uniform ct1
    grid columnconfigure $w.hlf.nb.display 1 -weight 0
    
    set disp $w.hlf.nb.display
    
    # build display header
    ttk::frame $disp.header
    
    ttk::checkbutton $disp.header.controlMode -text "Control Mode" -variable ::vmdPrefs::displayControlMode -onvalue 1 -offvalue 0
    ttk::button $disp.header.getSettings -text "Query VMD Display Settings" -command { ::vmdPrefs::displayGetCurrentVMD }
    ttk::button $disp.header.pushSettings -text "Push Display Settings to VMD" -command { ::vmdPrefs::displayPushSettingsToVMD }

    # grid display header
    grid $disp.header -column 0 -columnspan 3 -row 0 -sticky ns -pady 8
    grid $disp.header.controlMode -column 0 -row 0 -sticky nswe -padx "0 8"
    grid $disp.header.getSettings -column 1 -row 0 -sticky nswe -padx "0 4"
    grid $disp.header.pushSettings -column 2 -row 0 -sticky nswe
    
    grid columnconfigure $disp.header {1 2} -uniform ct1
    
    # build/grid a separator
    ttk::separator $disp.sep1 -orient horizontal
    grid $disp.sep1 -column 0 -columnspan 3 -row 1 -sticky nswe -pady 4
    
    
    # build frame1 (left panel)
    ttk::frame $disp.frame1

    ttk::label $disp.frame1.reposLbl -text "Reposition (X,Y):" -anchor w
    ttk::entry $disp.frame1.reposX -textvariable ::vmdPrefs::displaySettings(reposX) -width 5 -justify center -validate key -validatecommand { string is integer %P }
            bind .vmdPrefs.hlf.nb.display.frame1.reposX <FocusOut> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(reposX)] && [string is integer $::vmdPrefs::displaySettings(reposY)] } {display reposition $::vmdPrefs::displaySettings(reposX) $::vmdPrefs::displaySettings(reposY)}}
            bind .vmdPrefs.hlf.nb.display.frame1.reposX <KeyPress-Return> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(reposX)] && [string is integer $::vmdPrefs::displaySettings(reposY)] } {display reposition $::vmdPrefs::displaySettings(reposX) $::vmdPrefs::displaySettings(reposY)}}
    ttk::entry $disp.frame1.reposY -textvariable ::vmdPrefs::displaySettings(reposY) -width 5 -justify center -validate key -validatecommand { string is integer %P }
            bind .vmdPrefs.hlf.nb.display.frame1.reposY <FocusOut> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(reposX)] && [string is integer $::vmdPrefs::displaySettings(reposY)] } {display reposition $::vmdPrefs::displaySettings(reposX) $::vmdPrefs::displaySettings(reposY)}}
            bind .vmdPrefs.hlf.nb.display.frame1.reposY <KeyPress-Return> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(reposX)] && [string is integer $::vmdPrefs::displaySettings(reposY)] } {display reposition $::vmdPrefs::displaySettings(reposX) $::vmdPrefs::displaySettings(reposY)}}
            #ttk::button $disp.frame1.reposGet -text "Get (X,Y)" -command { puts "I haven't figured this one out yet" }
    
    ttk::label $disp.frame1.resizeLbl -text "Resize (W,H):" -anchor w
    ttk::entry $disp.frame1.resizeW -textvariable ::vmdPrefs::displaySettings(resizeW) -width 5 -justify center -validate key -validatecommand { string is integer %P }
            bind .vmdPrefs.hlf.nb.display.frame1.resizeW <FocusOut> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(resizeW)] && [string is integer $::vmdPrefs::displaySettings(resizeH)] } {display resize $::vmdPrefs::displaySettings(resizeW) $::vmdPrefs::displaySettings(resizeH)}}
            bind .vmdPrefs.hlf.nb.display.frame1.resizeW <KeyPress-Return> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(resizeW)] && [string is integer $::vmdPrefs::displaySettings(resizeH)] } {display resize $::vmdPrefs::displaySettings(resizeW) $::vmdPrefs::displaySettings(resizeH)}}
    ttk::entry $disp.frame1.resizeH -textvariable ::vmdPrefs::displaySettings(resizeH) -width 5 -justify center -validate key -validatecommand { string is integer %P }
            bind .vmdPrefs.hlf.nb.display.frame1.resizeH <FocusOut> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(resizeW)] && [string is integer $::vmdPrefs::displaySettings(resizeH)] } {display resize $::vmdPrefs::displaySettings(resizeW) $::vmdPrefs::displaySettings(resizeH)}}
            bind .vmdPrefs.hlf.nb.display.frame1.resizeH <KeyPress-Return> { if { $::vmdPrefs::displayControlMode && [string is integer $::vmdPrefs::displaySettings(resizeW)] && [string is integer $::vmdPrefs::displaySettings(resizeH)] } {display resize $::vmdPrefs::displaySettings(resizeW) $::vmdPrefs::displaySettings(resizeH)}}
    # ttk::button $disp.frame1.resizeGet -text "Get (W,H)" -command { lassign [display get size] ::vmdPrefs::displaySettings(resizeW) ::vmdPrefs::displaySettings(resizeH) }
    
    ttk::label $disp.frame1.projLbl -text "Projection Type:" -anchor w
    ttk::menubutton $disp.frame1.proj -direction below -menu $disp.frame1.proj.menu -textvariable ::vmdPrefs::displaySettings(projection) -width 10
    menu $disp.frame1.proj.menu -tearoff no
        foreach ele [display get projections] {
            .vmdPrefs.hlf.nb.display.frame1.proj.menu add command -label $ele -command "set ::vmdPrefs::displaySettings(projection) $ele; if { \$::vmdPrefs::displayControlMode } { display projection $ele }"
        }

    ttk::label $disp.frame1.axesLbl -text "Axes Location:" -anchor w
    ttk::menubutton $disp.frame1.axes -direction below -menu $disp.frame1.axes.menu -textvariable ::vmdPrefs::displaySettings(axes) -width 10
    menu $disp.frame1.axes.menu -tearoff no
        foreach ele [axes locations] {
            .vmdPrefs.hlf.nb.display.frame1.axes.menu add command -label "$ele" -command "set ::vmdPrefs::displaySettings(axes) $ele; if { \$::vmdPrefs::displayControlMode } { axes location $ele }"
        }

    ttk::label $disp.frame1.depthcueLbl -text "Depth Cueing:" -anchor w
    ttk::radiobutton $disp.frame1.depthcueOn -text "On" -variable ::vmdPrefs::displaySettings(depthcue) -value "on" -command { if { $::vmdPrefs::displayControlMode} {display depthcue on} }
    ttk::radiobutton $disp.frame1.depthcueOff -text "Off" -variable ::vmdPrefs::displaySettings(depthcue) -value "off" -command { if { $::vmdPrefs::displayControlMode} {display depthcue off} }
    
    ttk::label $disp.frame1.cuemodeLbl -text "Cue Mode:" -anchor w
    ttk::menubutton $disp.frame1.cuemode -direction below -menu $disp.frame1.cuemode.menu -textvariable ::vmdPrefs::displaySettings(cuemode) -width 10
    menu $disp.frame1.cuemode.menu -tearoff no
        $disp.frame1.cuemode.menu add command -label "Linear" \
            -command {
                set ::vmdPrefs::displaySettings(cuemode) "Linear"
                if { $::vmdPrefs::displayControlMode } { display cuemode "Linear" }
                .vmdPrefs.hlf.nb.display.frame1.cuestartLbl configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.minus configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.cuestart configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.plus configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cueendLbl configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.minus configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.cueend configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.plus configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuedensityLbl configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.minus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.cuedensity configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.plus configure -state disabled
            }
        $disp.frame1.cuemode.menu add command -label "Exp" \
            -command {
                set ::vmdPrefs::displaySettings(cuemode) "Exp" 
                if { $::vmdPrefs::displayControlMode } { display cuemode "Exp" }
                .vmdPrefs.hlf.nb.display.frame1.cuestartLbl configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.minus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.cuestart configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.plus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendLbl configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.minus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.cueend configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.plus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuedensityLbl configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.minus configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.cuedensity configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.plus configure -state normal
            }
        $disp.frame1.cuemode.menu add command -label "Exp2" \
            -command {
                set ::vmdPrefs::displaySettings(cuemode) "Exp2" 
                if { $::vmdPrefs::displayControlMode } { display cuemode "Exp2" }
                .vmdPrefs.hlf.nb.display.frame1.cuestartLbl configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.minus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.cuestart configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.plus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendLbl configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.minus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.cueend configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cueendFrame.plus configure -state disabled
                .vmdPrefs.hlf.nb.display.frame1.cuedensityLbl configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.minus configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.cuedensity configure -state normal
                .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.plus configure -state normal
            }

    # programmatically build -value+ repatitive elements (entry format: arrayKey GUIlabel incrementValue VMDcmd)
    foreach ele {
        {cuestart "Cue Start:" 0.1}
        {cueend "Cue End:" 0.1}
        {cuedensity "Cue Density:" 0.1}
        {nearclip "Near Clip:" 0.1}
        {farclip "Far Clip:" 0.1}
        {eyesep "Eye Sep.:" 0.1}
        {focalLength "Focal Length:" 0.1}
        {height "Screen Height:" 0.1}
        {distance "Screen Dist.:" 0.1}
    } {
        lassign $ele lbl desc scale
        
        ttk::label $disp.frame1.${lbl}Lbl -text $desc -anchor w
        ttk::frame $disp.frame1.${lbl}Frame
        ttk::button $disp.frame1.${lbl}Frame.minus -text "-" -width 1 -command "::vmdPrefs::displayPlusMinus [list $lbl [expr {-1*$scale}]]"
        ttk::entry $disp.frame1.${lbl}Frame.$lbl -textvariable ::vmdPrefs::displaySettings($lbl) -width 5 -justify center
            bind .vmdPrefs.hlf.nb.display.frame1.${lbl}Frame.$lbl <FocusOut> "::vmdPrefs::displayFormatManualEntry $lbl"
            bind .vmdPrefs.hlf.nb.display.frame1.${lbl}Frame.$lbl <KeyPress-Return> "::vmdPrefs::displayFormatManualEntry $lbl"
        ttk::button $disp.frame1.${lbl}Frame.plus -text "+" -width 1 -command "::vmdPrefs::displayPlusMinus [list $lbl $scale]"
    
    }
    
    # programmatically set entry box validations for -value+ elements
    # double
    foreach ele {cuestart cueend cuedensity nearclip farclip eyesep focalLength height} {
        .vmdPrefs.hlf.nb.display.frame1.${ele}Frame.${ele} configure -validate key -validatecommand { string is double %P }
    }
    # manually set entry box validations
    # double or negative
    .vmdPrefs.hlf.nb.display.frame1.distanceFrame.distance configure -validate key -validatecommand { regexp {^-?[0-9]*.?[0-9]*$} %P }
    

    # grid the diplay frame 1
    grid $disp.frame1 -column 0 -row 2 -sticky ns

    grid $disp.frame1.reposLbl -column 0 -row 0 -sticky nswe
    grid $disp.frame1.reposX -column 1 -row 0 -sticky nswe
    grid $disp.frame1.reposY -column 2 -row 0 -sticky nswe
    #grid $disp.frame1.reposGet -column 3 -row 0 -sticky nswe
    
    grid $disp.frame1.resizeLbl -column 0 -row 1 -sticky nswe
    grid $disp.frame1.resizeW -column 1 -row 1 -sticky nswe
    grid $disp.frame1.resizeH -column 2 -row 1 -sticky nswe
    # grid $disp.frame1.resizeGet -column 3 -row 1 -sticky nswe
    
    grid $disp.frame1.projLbl -column 0 -row 2 -sticky nswe
    grid $disp.frame1.proj -column 1 -columnspan 2 -row 2 -sticky nswe
    
    grid $disp.frame1.axesLbl -column 0 -row 3 -sticky nswe
    grid $disp.frame1.axes -column 1 -columnspan 2 -row 3 -sticky nswe
    
    grid $disp.frame1.depthcueLbl -column 0 -row 4 -sticky nswe
    grid $disp.frame1.depthcueOn -column 1 -row 4 -sticky nswe
    grid $disp.frame1.depthcueOff -column 2 -row 4 -sticky nswe
    
    grid $disp.frame1.cuemodeLbl -column 0 -row 5 -sticky nswe
    grid $disp.frame1.cuemode -column 1 -columnspan 2 -row 5 -sticky nswe
    
    set i 6
    foreach ele { cuestart cueend cuedensity nearclip farclip eyesep focalLength height distance } {
        grid $disp.frame1.${ele}Lbl -column 0 -row $i -sticky nswe
        grid $disp.frame1.${ele}Frame -column 1 -columnspan 3 -row $i -sticky nswe
        grid $disp.frame1.${ele}Frame.minus -column 0 -row 0 -sticky nswe
        grid $disp.frame1.${ele}Frame.${ele} -column 1 -row 0 -sticky nswe
        grid $disp.frame1.${ele}Frame.plus -column 2 -row 0 -sticky nswe
        grid columnconfigure $disp.frame1.${ele}Frame 1 -weight 4
        grid columnconfigure $disp.frame1.${ele}Frame {0 2} -weight 1
        
        incr i
    }
    unset i
    

    # build/grid separator between frames 1 & 2
    ttk::separator $disp.sep2 -orient vertical
    grid $disp.sep2 -column 1 -row 2 -sticky nswe -padx 4
    

    # build display frame 2 (right panel)
    ttk::frame $disp.frame2
    
    ttk::label $disp.frame2.cullingLbl -text "Culling:" -anchor w
    ttk::radiobutton $disp.frame2.cullingOn -text "On" -variable ::vmdPrefs::displaySettings(culling) -value "on" -command { if { $::vmdPrefs::displayControlMode} {display culling on} }
    ttk::radiobutton $disp.frame2.cullingOff -text "Off" -variable ::vmdPrefs::displaySettings(culling) -value "off" -command { if { $::vmdPrefs::displayControlMode} {display culling off} }
    
    ttk::label $disp.frame2.fpsLbl -text "FPS Indicator:" -anchor w
    ttk::radiobutton $disp.frame2.fpsOn -text "On" -variable ::vmdPrefs::displaySettings(fps) -value "on" -command { if { $::vmdPrefs::displayControlMode} {display fps on} }
    ttk::radiobutton $disp.frame2.fpsOff -text "Off" -variable ::vmdPrefs::displaySettings(fps) -value "off" -command { if { $::vmdPrefs::displayControlMode} {display fps off} }

    ttk::label $disp.frame2.light0Lbl -text "Light 0:" -anchor w
    ttk::radiobutton $disp.frame2.light0On -text "On" -variable ::vmdPrefs::displaySettings(light0) -value "on" -command { if { $::vmdPrefs::displayControlMode} {light 0 on} }
    ttk::radiobutton $disp.frame2.light0Off -text "Off" -variable ::vmdPrefs::displaySettings(light0) -value "off" -command { if { $::vmdPrefs::displayControlMode} {light 0 off} }

    ttk::label $disp.frame2.light1Lbl -text "Light 1:" -anchor w
    ttk::radiobutton $disp.frame2.light1On -text "On" -variable ::vmdPrefs::displaySettings(light1) -value "on" -command { if { $::vmdPrefs::displayControlMode} {light 1 on} }
    ttk::radiobutton $disp.frame2.light1Off -text "Off" -variable ::vmdPrefs::displaySettings(light1) -value "off" -command { if { $::vmdPrefs::displayControlMode} {light 1 off} }

    ttk::label $disp.frame2.light2Lbl -text "Light 2:" -anchor w
    ttk::radiobutton $disp.frame2.light2On -text "On" -variable ::vmdPrefs::displaySettings(light2) -value "on" -command { if { $::vmdPrefs::displayControlMode} {light 2 on} }
    ttk::radiobutton $disp.frame2.light2Off -text "Off" -variable ::vmdPrefs::displaySettings(light2) -value "off" -command { if { $::vmdPrefs::displayControlMode} {light 2 off} }
    
    ttk::label $disp.frame2.light3Lbl -text "Light 3:" -anchor w
    ttk::radiobutton $disp.frame2.light3On -text "On" -variable ::vmdPrefs::displaySettings(light3) -value "on" -command { if { $::vmdPrefs::displayControlMode} {light 3 on} }
    ttk::radiobutton $disp.frame2.light3Off -text "Off" -variable ::vmdPrefs::displaySettings(light3) -value "off" -command { if { $::vmdPrefs::displayControlMode} {light 3 off} }

    ttk::label $disp.frame2.backgroundLbl -text "Background:" -anchor w
    ttk::menubutton $disp.frame2.background -direction below -menu $disp.frame2.background.menu -textvariable ::vmdPrefs::displaySettings(background) -width 10
    menu $disp.frame2.background.menu -tearoff no
        $disp.frame2.background.menu add command -label "color" -command { set ::vmdPrefs::displaySettings(background) "color"; if { $::vmdPrefs::displayControlMode } {display backgroundgradient off} }
        $disp.frame2.background.menu add command -label "gradient" -command { set ::vmdPrefs::displaySettings(background) "gradient"; if { $::vmdPrefs::displayControlMode } {display backgroundgradient on} }
        
    ttk::label $disp.frame2.stageLbl -text "Stage:" -anchor w
    ttk::menubutton $disp.frame2.stage -direction below -menu $disp.frame2.stage.menu -textvariable ::vmdPrefs::displaySettings(stage) -width 10
    menu $disp.frame2.stage.menu -tearoff no
        foreach ele [stage locations] {
            $disp.frame2.stage.menu add command -label $ele -command "set ::vmdPrefs::displaySettings(stage) $ele; if { \$::vmdPrefs::displayControlMode } { stage location $ele }"
        }

    ttk::label $disp.frame2.stereoLbl -text "Stereo:" -anchor w
    ttk::menubutton $disp.frame2.stereo -direction below -menu $disp.frame2.stereo.menu -textvariable ::vmdPrefs::displaySettings(stereo) -width 10
    menu $disp.frame2.stereo.menu -tearoff no
        foreach ele [display get stereomodes] {
            $disp.frame2.stereo.menu add command -label $ele -command "set ::vmdPrefs::displaySettings(stereo) [list $ele]; ; if { \$::vmdPrefs::displayControlMode } { display stereo [list $ele] }"
        }
    
    ttk::label $disp.frame2.stereoswapLbl -text "Stereo Eye Swap:" -anchor w
    ttk::radiobutton $disp.frame2.stereoswapOn -text "On" -variable ::vmdPrefs::displaySettings(stereoswap) -value "on" -command { if { $::vmdPrefs::displayControlMode } {display stereoswap on} }
    ttk::radiobutton $disp.frame2.stereoswapOff -text "Off" -variable ::vmdPrefs::displaySettings(stereoswap) -value "off" -command { if { $::vmdPrefs::displayControlMode } {display stereoswap off} }

    ttk::label $disp.frame2.cachemodeLbl -text "Cache Mode:" -anchor w
    ttk::radiobutton $disp.frame2.cachemodeOn -text "On" -variable ::vmdPrefs::displaySettings(cachemode) -value "On" -command { if { $::vmdPrefs::displayControlMode } {display cachemode On} }
    ttk::radiobutton $disp.frame2.cachemodeOff -text "Off" -variable ::vmdPrefs::displaySettings(cachemode) -value "Off" -command { if { $::vmdPrefs::displayControlMode } {display cachemode Off} }

    ttk::label $disp.frame2.rendermodeLbl -text "Render Mode:" -anchor w
    ttk::menubutton $disp.frame2.rendermode -direction below -menu $disp.frame2.rendermode.menu -textvariable ::vmdPrefs::displaySettings(rendermode) -width 10
    menu $disp.frame2.rendermode.menu -tearoff no
        foreach ele [display get rendermodes] {
            $disp.frame2.rendermode.menu add command -label $ele -command "set ::vmdPrefs::displaySettings(rendermode) $ele; if { \$::vmdPrefs::displayControlMode } { display rendermode $ele }"
        }
    
    ttk::separator $disp.frame2.sep1 -orient horizontal
    ttk::label $disp.frame2.extRenderOpt -text "External Renderer Options" -anchor w

    ttk::label $disp.frame2.shadowsLbl -text "Shadows:" -anchor w
    ttk::radiobutton $disp.frame2.shadowsOn -text "On" -variable ::vmdPrefs::displaySettings(shadows) -value "on" -command { if { $::vmdPrefs::displayControlMode } {display shadows on; display ambientocclusion $::vmdPrefs::displaySettings(ambientocclusion)} }
    ttk::radiobutton $disp.frame2.shadowsOff -text "Off" -variable ::vmdPrefs::displaySettings(shadows) -value "off" -command { if { $::vmdPrefs::displayControlMode } {display shadows off; display ambientocclusion $::vmdPrefs::displaySettings(ambientocclusion)} }
    # for whatever reason, the shadows setting doesn't update in Display Settings until ambientocclusion is called
    
    ttk::label $disp.frame2.ambientocclusionLbl -text "Ambient Occlusion:" -anchor w
    ttk::radiobutton $disp.frame2.ambientocclusionOn -text "On" -variable ::vmdPrefs::displaySettings(ambientocclusion) -value "on" \
        -command {
            .vmdPrefs.hlf.nb.display.frame2.aoambientLbl configure -state normal
            .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.minus configure -state normal
            .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.aoambient configure -state normal
            .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.plus configure -state normal
            .vmdPrefs.hlf.nb.display.frame2.aodirectLbl configure -state normal
            .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.minus configure -state normal
            .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.aodirect configure -state normal
            .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.plus configure -state normal
            if { $::vmdPrefs::displayControlMode } {display ambientocclusion on}
        }
    ttk::radiobutton $disp.frame2.ambientocclusionOff -text "Off" -variable ::vmdPrefs::displaySettings(ambientocclusion) -value "off" \
        -command {
            .vmdPrefs.hlf.nb.display.frame2.aoambientLbl configure -state disabled
            .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.minus configure -state disabled
            .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.aoambient configure -state disabled
            .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.plus configure -state disabled
            .vmdPrefs.hlf.nb.display.frame2.aodirectLbl configure -state disabled
            .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.minus configure -state disabled
            .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.aodirect configure -state disabled
            .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.plus configure -state disabled
            if { $::vmdPrefs::displayControlMode } {display ambientocclusion off}
        }
    
    ttk::label $disp.frame2.aoambientLbl -text "AO Ambient:" -anchor w
    ttk::frame $disp.frame2.aoambientFrame
    ttk::button $disp.frame2.aoambientFrame.minus -text "-" -width 1 -command { ::vmdPrefs::displayPlusMinus aoambient -0.1 "display aoambient"}
    ttk::entry $disp.frame2.aoambientFrame.aoambient -textvariable ::vmdPrefs::displaySettings(aoambient) -width 5 -justify center -validate key -validatecommand { string is double %P }
    ttk::button $disp.frame2.aoambientFrame.plus -text "+" -width 1 -command { ::vmdPrefs::displayPlusMinus aoambient 0.1 "display aoambient"}

    ttk::label $disp.frame2.aodirectLbl -text "AO Direct:" -anchor w
    ttk::frame $disp.frame2.aodirectFrame
    ttk::button $disp.frame2.aodirectFrame.minus -text "-" -width 1 -command { ::vmdPrefs::displayPlusMinus aodirect -0.1 "display aodirect"}
    ttk::entry $disp.frame2.aodirectFrame.aodirect -textvariable ::vmdPrefs::displaySettings(aodirect) -width 5 -justify center -validate key -validatecommand { string is double %P }
    ttk::button $disp.frame2.aodirectFrame.plus -text "+" -width 1 -command { ::vmdPrefs::displayPlusMinus aodirect 0.1 "display aodirect"}

    
    # grid display frame 2
    grid $disp.frame2 -column 2 -row 2 -sticky ns

    grid $disp.frame2.cullingLbl -column 0 -row 0 -sticky nswe
    grid $disp.frame2.cullingOn -column 1 -row 0 -sticky nswe
    grid $disp.frame2.cullingOff -column 2 -row 0 -sticky nswe

    grid $disp.frame2.fpsLbl -column 0 -row 1 -sticky nswe
    grid $disp.frame2.fpsOn -column 1 -row 1 -sticky nswe
    grid $disp.frame2.fpsOff -column 2 -row 1 -sticky nswe

    grid $disp.frame2.light0Lbl -column 0 -row 2 -sticky nswe
    grid $disp.frame2.light0On -column 1 -row 2 -sticky nswe
    grid $disp.frame2.light0Off -column 2 -row 2 -sticky nswe

    grid $disp.frame2.light1Lbl -column 0 -row 3 -sticky nswe
    grid $disp.frame2.light1On -column 1 -row 3 -sticky nswe
    grid $disp.frame2.light1Off -column 2 -row 3 -sticky nswe
    
    grid $disp.frame2.light2Lbl -column 0 -row 4 -sticky nswe
    grid $disp.frame2.light2On -column 1 -row 4 -sticky nswe
    grid $disp.frame2.light2Off -column 2 -row 4 -sticky nswe

    grid $disp.frame2.light3Lbl -column 0 -row 5 -sticky nswe
    grid $disp.frame2.light3On -column 1 -row 5 -sticky nswe
    grid $disp.frame2.light3Off -column 2 -row 5 -sticky nswe
    
    grid $disp.frame2.backgroundLbl -column 0 -row 6 -sticky nswe
    grid $disp.frame2.background -column 1 -columnspan 3 -row 6 -sticky nswe

    grid $disp.frame2.stageLbl -column 0 -row 7 -sticky nswe
    grid $disp.frame2.stage -column 1 -columnspan 3 -row 7 -sticky nswe
    
    grid $disp.frame2.stereoLbl -column 0 -row 8 -sticky nswe
    grid $disp.frame2.stereo -column 1 -columnspan 3 -row 8 -sticky nswe
    
    grid $disp.frame2.stereoswapLbl -column 0 -row 9 -sticky nswe
    grid $disp.frame2.stereoswapOn -column 1 -row 9 -sticky nswe
    grid $disp.frame2.stereoswapOff -column 2 -row 9 -sticky nswe
    
    grid $disp.frame2.cachemodeLbl -column 0 -row 10 -sticky nswe
    grid $disp.frame2.cachemodeOn -column 1 -row 10 -sticky nswe
    grid $disp.frame2.cachemodeOff -column 2 -row 10 -sticky nswe
    
    grid $disp.frame2.rendermodeLbl -column 0 -row 11 -sticky nswe
    grid $disp.frame2.rendermode -column 1 -columnspan 3 -row 11 -sticky nswe
    
    grid $disp.frame2.sep1 -column 0 -columnspan 4 -row 12 -sticky nswe -pady 4
    grid $disp.frame2.extRenderOpt -column 0 -columnspan 2 -row 13 -sticky nswe
    
    grid $disp.frame2.shadowsLbl -column 0 -row 14 -sticky nswe
    grid $disp.frame2.shadowsOn -column 1 -row 14 -sticky nswe
    grid $disp.frame2.shadowsOff -column 2 -row 14 -sticky nswe

    grid $disp.frame2.ambientocclusionLbl -column 0 -row 15 -sticky nswe
    grid $disp.frame2.ambientocclusionOn -column 1 -row 15 -sticky nswe
    grid $disp.frame2.ambientocclusionOff -column 2 -row 15 -sticky nswe
    
    grid $disp.frame2.aoambientLbl -column 0 -row 16 -sticky nswe
    grid $disp.frame2.aoambientFrame -column 1 -columnspan 3 -row 16 -sticky nswe
    grid $disp.frame2.aoambientFrame.minus -column 1 -row 0 -sticky nswe
    grid $disp.frame2.aoambientFrame.aoambient -column 2 -row 0 -sticky nswe
    grid $disp.frame2.aoambientFrame.plus -column 3 -row 0 -sticky nswe
    grid columnconfigure $disp.frame2.aoambientFrame 2 -weight 4
    grid columnconfigure $disp.frame2.aoambientFrame {1 3} -weight 1

    grid $disp.frame2.aodirectLbl -column 0 -row 17 -sticky nswe
    grid $disp.frame2.aodirectFrame -column 1 -columnspan 3 -row 17 -sticky nswe
    grid $disp.frame2.aodirectFrame.minus -column 1 -row 0 -sticky nswe
    grid $disp.frame2.aodirectFrame.aodirect -column 2 -row 0 -sticky nswe
    grid $disp.frame2.aodirectFrame.plus -column 3 -row 0 -sticky nswe
    grid columnconfigure $disp.frame2.aodirectFrame 2 -weight 4
    grid columnconfigure $disp.frame2.aodirectFrame {1 3} -weight 1

    
    # take care of conditionally disable/active widgets based on initialized values
    if { $::vmdPrefs::displaySettings(cuemode) eq "Linear" } {
        .vmdPrefs.hlf.nb.display.frame1.cuestartLbl configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.minus configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.cuestart configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.plus configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cueendLbl configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cueendFrame.minus configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cueendFrame.cueend configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cueendFrame.plus configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cuedensityLbl configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.minus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.cuedensity configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.plus configure -state disabled
    } elseif { $::vmdPrefs::displaySettings(cuemode) eq "Exp" || $::vmdPrefs::displaySettings(cuemode) eq "Exp2" } {
        .vmdPrefs.hlf.nb.display.frame1.cuestartLbl configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.minus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.cuestart configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cuestartFrame.plus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cueendLbl configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cueendFrame.minus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cueendFrame.cueend configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cueendFrame.plus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame1.cuedensityLbl configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.minus configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.cuedensity configure -state normal
        .vmdPrefs.hlf.nb.display.frame1.cuedensityFrame.plus configure -state normal
    }
    
    if { $::vmdPrefs::displaySettings(ambientocclusion) eq "on" } {
        .vmdPrefs.hlf.nb.display.frame2.aoambientLbl configure -state normal
        .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.minus configure -state normal
        .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.aoambient configure -state normal
        .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.plus configure -state normal
        .vmdPrefs.hlf.nb.display.frame2.aodirectLbl configure -state normal
        .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.minus configure -state normal
        .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.aodirect configure -state normal
        .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.plus configure -state normal
    } else {
        .vmdPrefs.hlf.nb.display.frame2.aoambientLbl configure -state disabled
        .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.minus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.aoambient configure -state disabled
        .vmdPrefs.hlf.nb.display.frame2.aoambientFrame.plus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame2.aodirectLbl configure -state disabled
        .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.minus configure -state disabled
        .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.aodirect configure -state disabled
        .vmdPrefs.hlf.nb.display.frame2.aodirectFrame.plus configure -state disabled
    }

    #----------------------
    # colors
    #----------------------
    
    # add a tab for colors
    ttk::frame $w.hlf.nb.colors
    $w.hlf.nb add $w.hlf.nb.colors -text "Colors"
    
    grid columnconfigure $w.hlf.nb.colors 0 -weight 1; # allows both header and body to stretch in width
    grid rowconfigure $w.hlf.nb.colors 2 -weight 1; # allows body to stretch in height
    
    # build header section
    ttk::frame $w.hlf.nb.colors.header
    ttk::checkbutton $w.hlf.nb.colors.header.controlMode -text "Control Mode" -variable ::vmdPrefs::colorsControlMode -onvalue 1 -offvalue 0
    ttk::button $w.hlf.nb.colors.header.query -text "Query VMD Colors" -command { ::vmdPrefs::colorsGetCurrentVMD }
    ttk::button $w.hlf.nb.colors.header.push -text "Push Colors to VMD" -command { ::vmdPrefs::colorsPushSettingsToVMD }
    ttk::button $w.hlf.nb.colors.header.defaults -text "Load VMD Defaults" -command { ::vmdPrefs::colorsResetToDefaults }
    
    # grid header section
    grid $w.hlf.nb.colors.header -column 0 -row 0 -sticky ns -pady 8
    grid $w.hlf.nb.colors.header.controlMode -column 0 -row 0 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.colors.header.query -column 1 -row 0 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.colors.header.push -column 2 -row 0 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.colors.header.defaults -column 3 -row 0 -sticky nswe
    
    grid columnconfigure $w.hlf.nb.colors.header {1 2 3} -uniform ct1


    # build/grid separator
    ttk::separator $w.hlf.nb.colors.sep1 -orient horizontal
    grid $w.hlf.nb.colors.sep1 -column 0 -row 1 -sticky nswe -pady 4

    # build/grid the body section
    ttk::frame $w.hlf.nb.colors.body    
    grid $w.hlf.nb.colors.body -column 0 -row 2 -sticky nswe
    
    # configure appropriate stretching
    grid columnconfigure $w.hlf.nb.colors.body 0 -weight 1; # allows tv boxes to stretch by width
    grid rowconfigure $w.hlf.nb.colors.body {3 13} -weight 1; # allows tv boxes to stretch by height
    
    
    # build elements for the color categories DB settings
    # label for category tv
    ttk::label $w.hlf.nb.colors.body.categoryLbl -text "Assign Colors to VMD Elements" -anchor w
    
    # make a tv box for color categories and elements
    ttk::treeview $w.hlf.nb.colors.body.categoryTv -selectmode extended -yscroll "$w.hlf.nb.colors.body.categoryTvScroll set"
        $w.hlf.nb.colors.body.categoryTv configure -column {ele color} -displaycolumns {ele color} -show {tree headings} -height 5
        $w.hlf.nb.colors.body.categoryTv column #0 -width 150 -stretch 1 -anchor center
        $w.hlf.nb.colors.body.categoryTv column ele -width 150 -stretch 1 -anchor center
        $w.hlf.nb.colors.body.categoryTv column color -width 150 -stretch 1 -anchor center
        $w.hlf.nb.colors.body.categoryTv heading #0 -text "Category" -anchor center
        $w.hlf.nb.colors.body.categoryTv heading ele -text "Element" -anchor center
        $w.hlf.nb.colors.body.categoryTv heading color -text "Color" -anchor center
        
        bind .vmdPrefs.hlf.nb.colors.body.categoryTv <KeyPress-Escape> { .vmdPrefs.hlf.nb.colors.body.categoryTv selection set {} }
        #bind .vmdPrefs.hlf.nb.colors.body.categoryTv <FocusOut> { .vmdPrefs.hlf.nb.colors.body.categoryTv selection set {} }
        
    ttk::scrollbar $w.hlf.nb.colors.body.categoryTvScroll -orient vertical -command "$w.hlf.nb.colors.body.categoryTv yview"
    
    # build controls for setting category colors
    ttk::label $w.hlf.nb.colors.body.categoryColorSetLbl -text "Set Color:" -anchor w
    ttk::menubutton $w.hlf.nb.colors.body.categoryColorSet -direction below -menu $w.hlf.nb.colors.body.categoryColorSet.menu -textvariable ::vmdPrefs::colorsCategoryColor -width 10
        menu $w.hlf.nb.colors.body.categoryColorSet.menu -tearoff no
        foreach color [colorinfo colors] {
            .vmdPrefs.hlf.nb.colors.body.categoryColorSet.menu add command -label "$color" -command "set ::vmdPrefs::colorsCategoryColor $color; ::vmdPrefs::colorsCategoryColorSet $color"
        }
        
        
    # grid the category database section
    grid $w.hlf.nb.colors.body.categoryLbl -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.colors.body.categoryTv -column 0 -row 1 -rowspan 3 -sticky nswe
    grid $w.hlf.nb.colors.body.categoryTvScroll -column 1 -row 1 -rowspan 3 -sticky nswe
    grid $w.hlf.nb.colors.body.categoryColorSetLbl -column 2 -row 1 -sticky nswe
    grid $w.hlf.nb.colors.body.categoryColorSet -column 2 -row 2 -sticky nswe    
    
    
    # build the section for color definitions
    # build a label for the color def box
    ttk::label $w.hlf.nb.colors.body.colorDefLbl -text "Define Colors Available in VMD" -anchor w
    
    # build the color definitions tv box
    ttk::treeview $w.hlf.nb.colors.body.colorDefTv -selectmode browse -yscroll "$w.hlf.nb.colors.body.colorDefTvScroll set"
        $w.hlf.nb.colors.body.colorDefTv configure -column {ind color r g b} -show {headings} -height 5
        $w.hlf.nb.colors.body.colorDefTv column ind -width 50 -stretch 1 -anchor center
        $w.hlf.nb.colors.body.colorDefTv column color -width 150 -stretch 0 -anchor center
        $w.hlf.nb.colors.body.colorDefTv column r -width 75 -stretch 0 -anchor center
        $w.hlf.nb.colors.body.colorDefTv column g -width 75 -stretch 0 -anchor center
        $w.hlf.nb.colors.body.colorDefTv column b -width 75 -stretch 0 -anchor center
        $w.hlf.nb.colors.body.colorDefTv heading ind -text "Index" -anchor center
        $w.hlf.nb.colors.body.colorDefTv heading color -text "VMD Color" -anchor center
        $w.hlf.nb.colors.body.colorDefTv heading r -text "R" -anchor center
        $w.hlf.nb.colors.body.colorDefTv heading g -text "G" -anchor center
        $w.hlf.nb.colors.body.colorDefTv heading b -text "B" -anchor center
        
        bind .vmdPrefs.hlf.nb.colors.body.colorDefTv <KeyPress-Escape> { .vmdPrefs.hlf.nb.colors.body.colorDefTv selection set {} }
        #bind .vmdPrefs.hlf.nb.colors.body.colorDefTv <FocusOut> { .vmdPrefs.hlf.nb.colors.body.colorDefTv selection set {} }
        bind .vmdPrefs.hlf.nb.colors.body.colorDefTv <<TreeviewSelect>> {
            set id [.vmdPrefs.hlf.nb.colors.body.colorDefTv selection]
            if { $id eq "" } {
                unset id
                return
            } else {
                set r [expr { int(round([.vmdPrefs.hlf.nb.colors.body.colorDefTv set $id r] * 255.0)) }]
                set g [expr { int(round([.vmdPrefs.hlf.nb.colors.body.colorDefTv set $id g] * 255.0)) }]
                set b [expr { int(round([.vmdPrefs.hlf.nb.colors.body.colorDefTv set $id b] * 255.0)) }]
    
                set ::vmdPrefs::colorsR $r
                set ::vmdPrefs::colorsG $g
                set ::vmdPrefs::colorsB $b
                ::vmdPrefs::colorsUpdateScales; ::vmdPrefs::colorsUpdateColorBox
                set ::vmdPrefs::colorsTclNamedColorsLbl [::vmdPrefs::colorsLookupByRGB]            

                unset id r g b
            }
        }
        
    ttk::scrollbar $w.hlf.nb.colors.body.colorDefTvScroll -orient vertical -command "$w.hlf.nb.colors.body.colorDefTv yview"
    
    # build controls to setting/assigning/picking colors
    ttk::separator $w.hlf.nb.colors.body.sep1 -orient horizontal
    ttk::button $w.hlf.nb.colors.body.setColorDef -text "Set RGB of Selected" \
        -command {
            set r [format "%0.3f" [expr { $::vmdPrefs::colorsR / 255.0 }]]
            set g [format "%0.3f" [expr { $::vmdPrefs::colorsG / 255.0 }]]
            set b [format "%0.3f" [expr { $::vmdPrefs::colorsB / 255.0 }]]
            
            foreach ele [.vmdPrefs.hlf.nb.colors.body.colorDefTv selection] {
                .vmdPrefs.hlf.nb.colors.body.colorDefTv set $ele r $r
                .vmdPrefs.hlf.nb.colors.body.colorDefTv set $ele g $g
                .vmdPrefs.hlf.nb.colors.body.colorDefTv set $ele b $b
                if { $::vmdPrefs::colorsControlMode } { color change rgb $ele [format %0.3f $r] [format %0.3f $g] [format %0.3f $b] }
            }
            unset r g b
        }

    ttk::frame $w.hlf.nb.colors.body.rgbFrame
    ttk::label $w.hlf.nb.colors.body.rgbFrame.rLbl -text "R:" -width 3 -anchor center
    ttk::entry $w.hlf.nb.colors.body.rgbFrame.rEntry -textvariable ::vmdPrefs::colorsR -width 3 -justify center -validate key -validatecommand { string is integer %P }
    ttk::label $w.hlf.nb.colors.body.rgbFrame.gLbl -text "G:" -width 3 -anchor center
    ttk::entry $w.hlf.nb.colors.body.rgbFrame.gEntry -textvariable ::vmdPrefs::colorsG -width 3 -justify center -validate key -validatecommand { string is integer %P }
    ttk::label $w.hlf.nb.colors.body.rgbFrame.bLbl -text "B:" -width 3 -anchor center
    ttk::entry $w.hlf.nb.colors.body.rgbFrame.bEntry -textvariable ::vmdPrefs::colorsB -width 3 -justify center -validate key -validatecommand { string is integer %P }
    
    # bindings to sync rgb entry boxes with scales
    bind $w.hlf.nb.colors.body.rgbFrame.rEntry <KeyPress-Return> {::vmdPrefs::colorsUpdateScales}
    bind $w.hlf.nb.colors.body.rgbFrame.rEntry <FocusOut> {::vmdPrefs::colorsUpdateScales}
    bind $w.hlf.nb.colors.body.rgbFrame.gEntry <KeyPress-Return> {::vmdPrefs::colorsUpdateScales}
    bind $w.hlf.nb.colors.body.rgbFrame.gEntry <FocusOut> {::vmdPrefs::colorsUpdateScales}
    bind $w.hlf.nb.colors.body.rgbFrame.bEntry <KeyPress-Return> {::vmdPrefs::colorsUpdateScales}
    bind $w.hlf.nb.colors.body.rgbFrame.bEntry <FocusOut> {::vmdPrefs::colorsUpdateScales}

    scale $w.hlf.nb.colors.body.rScale -orient horizontal -relief sunken -from 0 -to 255 -variable ::vmdPrefs::colorsRScale -showvalue false -bg red -activebackground red -command {set ::vmdPrefs::colorsR $::vmdPrefs::colorsRScale; ::vmdPrefs::colorsUpdateColorBox; set ::vmdPrefs::colorsTclNamedColorsLbl [::vmdPrefs::colorsLookupByRGB]; ::vmdPrefs::null}
    scale $w.hlf.nb.colors.body.gScale -orient horizontal -relief sunken -from 0 -to 255 -variable ::vmdPrefs::colorsGScale -showvalue false -bg green -activebackground green -command {set ::vmdPrefs::colorsG $::vmdPrefs::colorsGScale; ::vmdPrefs::colorsUpdateColorBox; set ::vmdPrefs::colorsTclNamedColorsLbl [::vmdPrefs::colorsLookupByRGB]; ::vmdPrefs::null}
    scale $w.hlf.nb.colors.body.bScale -orient horizontal -relief sunken -from 0 -to 255 -variable ::vmdPrefs::colorsBScale -showvalue false -bg blue -activebackground blue -command {set ::vmdPrefs::colorsB $::vmdPrefs::colorsBScale; ::vmdPrefs::colorsUpdateColorBox; set ::vmdPrefs::colorsTclNamedColorsLbl [::vmdPrefs::colorsLookupByRGB]; ::vmdPrefs::null}
    
    ttk::menubutton $w.hlf.nb.colors.body.namedColors -direction below -menu $w.hlf.nb.colors.body.namedColors.menu -textvariable ::vmdPrefs::colorsTclNamedColorsLbl
    menu $w.hlf.nb.colors.body.namedColors.menu -tearoff no
        foreach ele [lsort -dictionary [array names ::vmdPrefs::tclNamedColors]] {
            $w.hlf.nb.colors.body.namedColors.menu add command -label $ele -command "lassign [list $::vmdPrefs::tclNamedColors($ele)] ::vmdPrefs::colorsR ::vmdPrefs::colorsG ::vmdPrefs::colorsB; ::vmdPrefs::colorsUpdateScales; set ::vmdPrefs::colorsTclNamedColorsLbl $ele; ::vmdPrefs::colorsUpdateColorBox"
        }
    
    ttk::button $w.hlf.nb.colors.body.pickColor -text "Use Color Picker" \
        -command {
            if { [set hexColor [tk_chooseColor]] ne "" } {
                scan $hexColor "\#%2x%2x%2x" red green blue
                lassign [list $red $green $blue] ::vmdPrefs::colorsR ::vmdPrefs::colorsG ::vmdPrefs::colorsB
                ::vmdPrefs::colorsUpdateScales; ::vmdPrefs::colorsUpdateColorBox
                set ::vmdPrefs::colorsTclNamedColorsLbl [::vmdPrefs::colorsLookupByRGB]
            }
        }   

    label $w.hlf.nb.colors.body.colorBox -text ""
    
    # grid the color definitions
    grid $w.hlf.nb.colors.body.colorDefLbl -column 0 -row 4 -sticky nswe -pady "5 0"
    grid $w.hlf.nb.colors.body.colorDefTv -column 0 -row 5 -rowspan 9 -sticky nswe
    grid $w.hlf.nb.colors.body.colorDefTvScroll -column 1 -row 5 -rowspan 9 -sticky nswe
        
    grid $w.hlf.nb.colors.body.setColorDef -column 2 -row 5 -sticky nswe -pady "2 0"
    grid $w.hlf.nb.colors.body.sep1 -column 2 -row 6 -sticky nswe -pady "4 2"
    
    grid $w.hlf.nb.colors.body.rgbFrame -column 2 -row 7 -sticky nswe -pady "2 0"
    grid $w.hlf.nb.colors.body.rgbFrame.rLbl -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.colors.body.rgbFrame.rEntry  -column 1 -row 0 -sticky nswe
    grid $w.hlf.nb.colors.body.rgbFrame.gLbl -column 2 -row 0 -sticky nswe
    grid $w.hlf.nb.colors.body.rgbFrame.gEntry  -column 3 -row 0 -sticky nswe
    grid $w.hlf.nb.colors.body.rgbFrame.bLbl -column 4 -row 0 -sticky nswe
    grid $w.hlf.nb.colors.body.rgbFrame.bEntry  -column 5 -row 0 -sticky nswe
    
    grid $w.hlf.nb.colors.body.rScale -column 2 -row 8 -sticky nswe -pady "2 0"
    grid $w.hlf.nb.colors.body.gScale -column 2 -row 9 -sticky nswe
    grid $w.hlf.nb.colors.body.bScale -column 2 -row 10 -sticky nswe
    
    grid $w.hlf.nb.colors.body.namedColors -column 2 -row 11 -sticky nswe -pady "2 0"
    grid $w.hlf.nb.colors.body.pickColor -column 2 -row 12 -sticky nswe -pady "2 0"
    
    grid $w.hlf.nb.colors.body.colorBox -column 2 -row 13 -sticky nswe -pady "2 0"


    # build controls to set color scale
    ttk::label $w.hlf.nb.colors.body.colorScaleLbl -text "Define Color Scale" -anchor w
    ttk::frame $w.hlf.nb.colors.body.colorScale
    ttk::label $w.hlf.nb.colors.body.colorScale.methodLbl -text "Method:" -anchor center
    ttk::menubutton $w.hlf.nb.colors.body.colorScale.method -direction below -menu $w.hlf.nb.colors.body.colorScale.method.menu -textvariable ::vmdPrefs::colorsScaleMethod -width 5
    menu $w.hlf.nb.colors.body.colorScale.method.menu -tearoff no
        foreach ele [colorinfo scale methods] {
            $w.hlf.nb.colors.body.colorScale.method.menu add command -label $ele -command "set ::vmdPrefs::colorsScaleMethod $ele; if { \$::vmdPrefs::colorsControlMode } {color scale method $ele}; ::vmdPrefs::colorsUpdateColorScaleBox"
        }
    ttk::label $w.hlf.nb.colors.body.colorScale.offsetLbl -text "Offset:" -anchor center
    ttk::label $w.hlf.nb.colors.body.colorScale.offsetVal -anchor center -width 4 -textvariable ::vmdPrefs::colorsScaleOffset
    scale $w.hlf.nb.colors.body.colorScale.offset -orient horizontal -from -1 -to 1 -variable ::vmdPrefs::colorsScaleOffset -showvalue false -digits 3 -resolution 0.01 -bg grey50 -activebackground grey50 -command { if { $::vmdPrefs::colorsControlMode } {color scale min $::vmdPrefs::colorsScaleOffset}; ::vmdPrefs::colorsUpdateColorScaleBox; ::vmdPrefs::null }
    ttk::label $w.hlf.nb.colors.body.colorScale.midpointLbl -text "Midpoint:" -anchor center
    ttk::label $w.hlf.nb.colors.body.colorScale.midpointVal -anchor center -width 4 -textvariable ::vmdPrefs::colorsScaleMidpoint
    scale $w.hlf.nb.colors.body.colorScale.midpoint -orient horizontal -from 0 -to 1 -variable ::vmdPrefs::colorsScaleMidpoint -showvalue false -digits 3 -resolution 0.01 -bg grey50 -activebackground grey50 -command { if { $::vmdPrefs::colorsControlMode } {color scale midpoint $::vmdPrefs::colorsScaleMidpoint}; ::vmdPrefs::colorsUpdateColorScaleBox; ::vmdPrefs::null }
    canvas $w.hlf.nb.colors.body.colorScale.scaleBox -height 10

    # grid controls to set color scale
    grid $w.hlf.nb.colors.body.colorScaleLbl -column 0 -row 14 -sticky nswe -pady "5 0"
    grid $w.hlf.nb.colors.body.colorScale -column 0 -columnspan 3 -row 15 -sticky nswe
    grid $w.hlf.nb.colors.body.colorScale.methodLbl -column 0 -row 0 -sticky nswe -padx "2 0"
    grid $w.hlf.nb.colors.body.colorScale.method -column 1 -row 0 -sticky nswe -padx "2 0"
    grid $w.hlf.nb.colors.body.colorScale.offsetLbl -column 2 -row 0 -sticky nswe -padx "5 0"
    grid $w.hlf.nb.colors.body.colorScale.offsetVal -column 3 -row 0 -sticky nswe -padx "2 0"
    grid $w.hlf.nb.colors.body.colorScale.offset -column 4 -row 0 -sticky we -padx "2 0"
    grid $w.hlf.nb.colors.body.colorScale.midpointLbl -column 5 -row 0 -sticky nswe -padx "5 0"
    grid $w.hlf.nb.colors.body.colorScale.midpointVal -column 6 -row 0 -sticky nswe -padx "2 0"
    grid $w.hlf.nb.colors.body.colorScale.midpoint -column 7 -row 0 -sticky we -padx "2 0"
    grid $w.hlf.nb.colors.body.colorScale.scaleBox -column 0 -columnspan 8 -row 1 -sticky nswe -padx 2 -pady 2
    
    grid rowconfigure $w.hlf.nb.colors.body.colorScale 1 -minsize 2 -weight 0

    # force geometry manager to actually build everything
    update
    # finish building the canvas-based color box for scale
    set ::vmdPrefs::colorsLineIDs {}
    set boxWidth [winfo width .vmdPrefs.hlf.nb.colors.body.colorScale.scaleBox]
    set boxHeight [winfo height .vmdPrefs.hlf.nb.colors.body.colorScale.scaleBox]
    set lineNum 200.0
    set lineWidth [expr {$boxWidth / $lineNum}]
    for {set i 0} {$i < $lineNum} {incr i} {
        set xPt [expr { ($i / $lineNum) * $boxWidth }]
        lappend ::vmdPrefs::colorsLineIDs [$w.hlf.nb.colors.body.colorScale.scaleBox create line $xPt 0 $xPt $boxHeight -width $lineWidth -fill grey]
        unset xPt
    }
    unset boxWidth boxHeight lineNum lineWidth
    

    # query VMD for current settings and load relevant data
    ::vmdPrefs::colorsGetCurrentVMD
    

    #----------------------
    # representations
    #----------------------
    
    # add a tab for reps
    ttk::frame $w.hlf.nb.reps
    $w.hlf.nb add $w.hlf.nb.reps -text "Representations"

    # configure the tab rows/columns    
    grid columnconfigure $w.hlf.nb.reps 0 -weight 1; # allows reps frame to stretch in width
    #grid rowconfigure $w.hlf.nb.reps 0 -weight 1; # allows reps frame to stretch in height

    # build control buttons
    ttk::frame $w.hlf.nb.reps.controlButtons
    ttk::checkbutton $w.hlf.nb.reps.controlButtons.controlMode -text "Control Mode" -variable ::vmdPrefs::repsControlMode -onvalue 1 -offvalue 0
    ttk::button $w.hlf.nb.reps.controlButtons.query -text "Query VMD Rep. Defaults" -command { ::vmdPrefs::repsQueryRepDefaults }
    ttk::button $w.hlf.nb.reps.controlButtons.push -text "Push Rep. Defaults to VMD" -command { ::vmdPrefs::repsPushRepDefaults }

    # grid control buttons
    grid $w.hlf.nb.reps.controlButtons -column 0 -row 0 -sticky ns -pady 8
    grid $w.hlf.nb.reps.controlButtons.controlMode -column 0 -row 0 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.reps.controlButtons.query -column 1 -row 0 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.reps.controlButtons.push -column 2 -row 0 -sticky nswe -padx "0 4"

    # configure control buttons
    grid columnconfigure $w.hlf.nb.reps.controlButtons {1 2} -uniform ct1


    # build/grid a separator
    ttk::separator $w.hlf.nb.reps.controlSep -orient horizontal
    grid $w.hlf.nb.reps.controlSep -column 0 -row 1 -sticky nswe -pady 4 -padx 2

    # build/grid the default representations settings label
    ttk::label $w.hlf.nb.reps.repDefaultsLbl -text "Default Attributes for New Representations" -anchor w
    grid $w.hlf.nb.reps.repDefaultsLbl -column 0 -row 2 -sticky nswe
    
    
    # build new rep defaults
    ttk::frame $w.hlf.nb.reps.defaults
    
    ttk::label $w.hlf.nb.reps.defaults.colorMethodLbl -text "Coloring Method:" -anchor center
    ttk::menubutton $w.hlf.nb.reps.defaults.colorMethod -direction below -menu $w.hlf.nb.reps.defaults.colorMethod.menu -textvariable ::vmdPrefs::reps(colorMethod) -width 15
    menu $w.hlf.nb.reps.defaults.colorMethod.menu -tearoff no
        foreach ele {Name Type Element ResName ResType ResID Chain SegName Conformation
                     Molecule {Secondary Structure} ColorID Beta Occupancy Mass Charge
                     Radial Position-X Position-Y Position-Z User1 User2 User3 User4
                     {Physical Time} Timestep Velocity Fragment Index Backbone Throb
                     Volume} {
            $w.hlf.nb.reps.defaults.colorMethod.menu add command -label $ele -command "set ::vmdPrefs::reps(colorMethod) [list $ele]; if { \"$ele\" eq \"ColorID\" } { .vmdPrefs.hlf.nb.reps.defaults.colorID configure -state normal } else { .vmdPrefs.hlf.nb.reps.defaults.colorID configure -state disabled }; if { \$::vmdPrefs::repsControlMode } { ::vmdPrefs::repsPushRepDefaults }"
        }
    ttk::menubutton $w.hlf.nb.reps.defaults.colorID -direction below -menu $w.hlf.nb.reps.defaults.colorID.menu -textvariable ::vmdPrefs::reps(colorID) -width 8
    menu $w.hlf.nb.reps.defaults.colorID.menu -tearoff no
    foreach color [colorinfo colors] {
        $w.hlf.nb.reps.defaults.colorID.menu add command -label $color -command "set ::vmdPrefs::reps(colorID) $color; if { \$::vmdPrefs::repsControlMode } { ::vmdPrefs::repsPushRepDefaults }"
    }

    ttk::label $w.hlf.nb.reps.defaults.materialLbl -text "Material:" -anchor center
    ttk::menubutton $w.hlf.nb.reps.defaults.material -direction below -menu $w.hlf.nb.reps.defaults.material.menu -textvariable ::vmdPrefs::reps(material) -width 15
    menu $w.hlf.nb.reps.defaults.material.menu -tearoff no
        foreach ele [material list] {
            $w.hlf.nb.reps.defaults.material.menu add command -label $ele -command "set ::vmdPrefs::reps(material) $ele; if { \$::vmdPrefs::repsControlMode } { ::vmdPrefs::repsPushRepDefaults }"
        }    
    
    ttk::label $w.hlf.nb.reps.defaults.drawMethodLbl -text "Drawing Method:" -anchor center
    ttk::menubutton $w.hlf.nb.reps.defaults.drawMethod -direction below -menu $w.hlf.nb.reps.defaults.drawMethod.menu -textvariable ::vmdPrefs::reps(drawMethod) -width 15
    menu $w.hlf.nb.reps.defaults.drawMethod.menu -tearoff no
        foreach ele {Lines Bonds DynamicBonds HBonds Points VDW CPK Licorice Polyhedra Trace Tube
                     Ribbons NewRibbons Cartoon NewCartoon PaperChain Twister QuickSurf MSMS Surf
                     VolumeSlice Isosurface FieldLines Orbital Beads Dotted Solvent} {
            $w.hlf.nb.reps.defaults.drawMethod.menu add command -label $ele -command "set ::vmdPrefs::reps(drawMethod) $ele; if { \$::vmdPrefs::repsControlMode } { ::vmdPrefs::repsPushRepDefaults }"
        }
    
    ttk::label $w.hlf.nb.reps.defaults.selectionLbl -text "Selection:" -anchor center
    ttk::entry $w.hlf.nb.reps.defaults.selection -textvariable ::vmdPrefs::reps(selection) -justify left
    bind .vmdPrefs.hlf.nb.reps.defaults.selection <KeyPress-Return> "if { \$::vmdPrefs::repsControlMode } { ::vmdPrefs::repsPushRepDefaults }"
    bind .vmdPrefs.hlf.nb.reps.defaults.selection <FocusOut> "if { \$::vmdPrefs::repsControlMode } { ::vmdPrefs::repsPushRepDefaults }"
    
        
    # grid method / material
    grid $w.hlf.nb.reps.defaults -column 0 -row 3 -sticky ns
    grid $w.hlf.nb.reps.defaults.colorMethodLbl -column 0 -row 1 -sticky nswe
    grid $w.hlf.nb.reps.defaults.colorMethod -column 1 -row 1 -sticky nswe
    grid $w.hlf.nb.reps.defaults.colorID -column 2 -row 1 -sticky nswe
    if { $::vmdPrefs::reps(colorMethod) ne "ColorID"} { $w.hlf.nb.reps.defaults.colorID configure -state disabled }
    
    grid $w.hlf.nb.reps.defaults.materialLbl -column 0 -row 2 -sticky nswe
    grid $w.hlf.nb.reps.defaults.material -column 1 -row 2 -sticky nswe

    grid $w.hlf.nb.reps.defaults.drawMethodLbl -column 0 -row 3 -sticky nswe
    grid $w.hlf.nb.reps.defaults.drawMethod -column 1 -row 3 -sticky nswe
    
    grid $w.hlf.nb.reps.defaults.selectionLbl -column 0 -row 4 -sticky nswe
    grid $w.hlf.nb.reps.defaults.selection -column 1 -columnspan 2 -row 4 -sticky nswe

    # reset values based on current default representation settings
    if { [llength [mol default color]] == 2} {
        set ::vmdPrefs::reps(colorMethod) [lindex [mol default color] 0]
        set ::vmdPrefs::reps(colorID) [lindex [colorinfo colors] [lindex [mol default color] 1]]
        $w.hlf.nb.reps.defaults.colorID configure -state normal
    } else {
        set ::vmdPrefs::reps(colorMethod) [mol default color]
    }
    set ::vmdPrefs::reps(material) [mol default material]
    set ::vmdPrefs::reps(drawMethod) [mol default style]
    set ::vmdPrefs::reps(selection) [mol default selection]


    # NOTE: the representation definitions section is hidden until the required functionality
    #       is added to VMD, at which point the next 3 lines commented out by ## should be reinstated    
    # build/grid a separator
    ttk::separator $w.hlf.nb.reps.sep1 -orient horizontal
    # grid $w.hlf.nb.reps.sep1 -column 0 -row 2 -sticky nswe -pady 4

    
    # build/grid rep definition lbl
    ttk::label $w.hlf.nb.reps.repDefinitionsLbl -text "Representation Definitions" -anchor w
    # grid $w.hlf.nb.reps.repDefinitionsLbl -column 0 -row 3 -sticky nswe
    
    
    # build representation definitions
    ttk::frame $w.hlf.nb.reps.definitions
    
    ttk::treeview $w.hlf.nb.reps.definitions.tv -selectmode browse -yscroll "$w.hlf.nb.reps.definitions.tvScroll set"
        $w.hlf.nb.reps.definitions.tv configure -column {prop value adjust} -displaycolumns {prop value} -show {tree headings} -height 10
        $w.hlf.nb.reps.definitions.tv column #0 -width 150 -stretch 1 -anchor center
        $w.hlf.nb.reps.definitions.tv column prop -width 150 -stretch 1 -anchor center
        $w.hlf.nb.reps.definitions.tv column value -width 100 -stretch 1 -anchor center
        $w.hlf.nb.reps.definitions.tv heading #0 -text "Drawing Method" -anchor center
        $w.hlf.nb.reps.definitions.tv heading prop -text "Property" -anchor center
        $w.hlf.nb.reps.definitions.tv heading value -text "Value" -anchor center
    ttk::scrollbar $w.hlf.nb.reps.definitions.tvScroll -orient vertical -command "$w.hlf.nb.reps.definitions.tv yview"
    
    # setup tv bindings
    bind $w.hlf.nb.reps.definitions.tv <<TreeviewSelect>> { ::vmdPrefs::repsModifyDefinition }
    
    ttk::frame $w.hlf.nb.reps.definitions.modFrame
    ttk::frame  $w.hlf.nb.reps.definitions.modFrame.plusMinus
    ttk::button $w.hlf.nb.reps.definitions.modFrame.plusMinus.minus -text "-" -width 1 -state disabled \
        -command {
            if { [set tempVal [format "%0.${::vmdPrefs::repDefPlusMinusSigFigs}f" [expr {$::vmdPrefs::repDefPlusMinus - $::vmdPrefs::repDefPlusMinusIncr}]]] < 0 } {
                set ::vmdPrefs::repDefPlusMinus [format "%0.${::vmdPrefs::repDefPlusMinusSigFigs}f" 0.00]; unset tempVal
            } else {
                set ::vmdPrefs::repDefPlusMinus $tempVal; unset tempVal
            }
        }
    ttk::entry  $w.hlf.nb.reps.definitions.modFrame.plusMinus.entry -textvariable ::vmdPrefs::repDefPlusMinus -width 5 -justify center -state disabled -validate key -validatecommand { string is double %P }
    ttk::button $w.hlf.nb.reps.definitions.modFrame.plusMinus.plus -text "+" -width 1 -state disabled \
        -command {
            set ::vmdPrefs::repDefPlusMinus [format "%0.${::vmdPrefs::repDefPlusMinusSigFigs}f" [expr {$::vmdPrefs::repDefPlusMinus + $::vmdPrefs::repDefPlusMinusIncr}]]
        }

    
    ttk::frame $w.hlf.nb.reps.definitions.modFrame.scale
    ttk::label $w.hlf.nb.reps.definitions.modFrame.scale.lbl -textvariable ::vmdPrefs::repDefScale -anchor center -width 5 -state disabled
    scale $w.hlf.nb.reps.definitions.modFrame.scale.scale -orient horizontal -relief sunken -variable ::vmdPrefs::repDefScale -showvalue false -command {} -state disabled
    
    ttk::frame $w.hlf.nb.reps.definitions.modFrame.dropDown
    ttk::menubutton $w.hlf.nb.reps.definitions.modFrame.dropDown.menu -direction below -menu $w.hlf.nb.reps.definitions.modFrame.dropDown.menu.menu -textvariable ::vmdPrefs::repDefMenu -width 10 -state disabled
    menu $w.hlf.nb.reps.definitions.modFrame.dropDown.menu.menu -tearoff no
    
    ttk::separator $w.hlf.nb.reps.definitions.modFrame.sep -orient horizontal
    ttk::button $w.hlf.nb.reps.definitions.modFrame.accept -text "Accept Value" -state disabled \
        -command {
            if { [.vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.entry cget -state] ne "disabled" } {
                .vmdPrefs.hlf.nb.reps.definitions.tv set [.vmdPrefs.hlf.nb.reps.definitions.tv selection] value $::vmdPrefs::repDefPlusMinus
            } elseif { [.vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.scale cget -state] ne "disabled" } {
                .vmdPrefs.hlf.nb.reps.definitions.tv set [.vmdPrefs.hlf.nb.reps.definitions.tv selection] value $::vmdPrefs::repDefScale
            } elseif { [.vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu cget -state] ne "disabled" } {
                .vmdPrefs.hlf.nb.reps.definitions.tv set [.vmdPrefs.hlf.nb.reps.definitions.tv selection] value $::vmdPrefs::repDefMenu
            } else {
                .vmdPrefs.hlf.nb.reps.definitions.tv set [.vmdPrefs.hlf.nb.reps.definitions.tv selection] value "error"
            }
        }
    

    # grid/configure representation defaults
    # grid $w.hlf.nb.reps.definitions -column 0 -row 4 -sticky nswe
    grid columnconfigure $w.hlf.nb.reps.definitions 0 -weight 1
    grid columnconfigure $w.hlf.nb.reps.definitions {1 2} -weight 0

    grid $w.hlf.nb.reps.definitions.tv -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.reps.definitions.tvScroll -column 1 -row 0 -sticky nswe
    
    grid $w.hlf.nb.reps.definitions.modFrame -column 2 -row 0

    grid $w.hlf.nb.reps.definitions.modFrame.plusMinus -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.reps.definitions.modFrame.plusMinus.minus -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.reps.definitions.modFrame.plusMinus.entry -column 1 -row 0 -sticky nswe
    grid $w.hlf.nb.reps.definitions.modFrame.plusMinus.plus -column 2 -row 0 -sticky nswe
    grid columnconfigure $w.hlf.nb.reps.definitions.modFrame.plusMinus 1 -weight 1
    
    grid $w.hlf.nb.reps.definitions.modFrame.scale -column 0 -row 1 -sticky nswe
    grid $w.hlf.nb.reps.definitions.modFrame.scale.lbl -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.reps.definitions.modFrame.scale.scale -column 1 -row 0 -sticky nswe
    
    grid $w.hlf.nb.reps.definitions.modFrame.dropDown -column 0 -row 2 -sticky nswe
    grid $w.hlf.nb.reps.definitions.modFrame.dropDown.menu -column 0 -row -0 -sticky nswe
    grid columnconfigure $w.hlf.nb.reps.definitions.modFrame.dropDown 0 -weight 1
    
    #grid $w.hlf.nb.reps.definitions.modFrame.sep -column 0 -row 3 -sticky nswe -pady 4
    grid $w.hlf.nb.reps.definitions.modFrame.accept -column 0 -row 4 -sticky nswe -pady "4 0"

    
    # fill the rep definitions TV box
    # notation: {parent {child1 child2 childN} }
    # children settings: { PropertyLabel DefaultValue {adjType adjVal1 adjVal2 adjValN} }
    # adjType can be plusMinus, scale, menu
    # plusMinus adjValues: increment #_SigFigs, note {} = int
    # scale adjValues: min max step/resolution
    # menu adjValues: list of values
    foreach ele {        
        { Lines
            { Thickness 1 {plusMinus 1 {}} }
        }
        { Bonds
            {{Bond Radius}     0.3 {plusMinus 0.1 1} }
            {{Bond Resolution}  10 {plusMinus 5  {}} }
        }
        { DynamicBonds
            {{Distance Cutoff} 1.6 {plusMinus 0.1 1} }
            {{Bond Radius}     0.3 {plusMinus 0.1 1} }
            {{Bond Resolution}  6  {plusMinus 5  {}} }
        }
        { HBonds
            {{Distance Cutoff} 3.0 {plusMinus 0.1 1} }
            {{Angle Cutoff}    20  {plusMinus 2  {}} }
            {{Line Thickness}   1  {plusMinus 1  {}} }
        }
        { Points
            {Size 1 {plusMinus 1 {}} }
        }
        { VDW
            {{Sphere Scale}      1.0 {plusMinus 0.1 1} }
            {{Sphere Resolution} 12  {plusMinus 5  {}} }
        }
        { CPK
            {{Sphere Scale}      1.0 {plusMinus 0.1 1} }
            {{Sphere Resolution} 10  {plusMinus 1  {}} }
            {{Bond Radius}       0.3 {plusMinus 0.1 1} }
            {{Bond Resolution}   10  {plusMinus 1  {}} }
        }
        { Licorice
            {{Sphere Resolution} 10  {plusMinus  1 {}} }
            {{Bond Radius}       0.3 {plusMinus 0.1 1} }
            {{Bond Resolution}   10  {plusMinus  1 {}} }
        }
        { Polyhedra
            {{Distance Cutoff} 1.6 {plusMinus 0.1 1} }
        }
        { Trace
            {{Bond Radius}    0.3 {plusMinus 0.1 1} }
            {{Bond Resolution} 10 {plusMinus  1 {}} }
        }
        { Tube
            {Radius    0.3 {plusMinus 0.1 1} }
            {Resolution 10 {plusMinus  1 {}} }
        }
        { Ribbons
            {Width      2  {plusMinus 1   {}} }
            {Radius    0.3 {plusMinus 0.1  1} }
            {Resolution 10 {plusMinus 1   {}} }
        }
        { NewRibbons
            {{Spline Style} {Catmull-Rom} {menu {Catmull-Rom} {B-spline}} }
            {{Aspect Ratio}     3.00      {scale 1.00 10.00 0.50}         }
            {{Thickness}        0.30      {scale 0.10 10.00 0.10}         }
            {{Resolution}        10       {plusMinus 1 {}}                }
        }
        { Cartoon
            {{Beta Sheet Thickness}    5  {plusMinus 1  {}} }
            {{Helix/Coil Radius}      2.1 {plusMinus 0.1 1} }
            {{Helix/Coil Resolution}  12  {plusMinus 1  {}} }
        }
        { NewCartoon
            {{Spline Style} {Catmull-Rom} {menu {Catmull-Rom} {B-spline}} }
            {{Aspect Ratio}     4.10      {scale 1.00 10.00 0.50}         }
            {{Thickness}        0.30      {scale 0.10 10.00 0.10}         }
            {{Resolution}        10       {plusMinus 1 {}}                }
        }
        { PaperChain
            {{Bipyramid Height}  1.00  {scale 0.10 10.00 0.01} }
            {{Max. Ring Size}     10   {plusMinus 1 {}} }
        }
        { Twister
            {{State Ribbons At}  {Ring Centroid}  {menu {Ring Centroid} {Ring Edge}} }
            {{Hide Shared Links} {No}             {menu {No} {Yes}}                  }
            {{Steps in Ribbon}   10               {plusMinus 1 {}}                   }
            {{Max. Ring Size}    10               {plusMinus 1 {}}                   }
            {{Max. Linking Distance} 5            {plusMinus 1 {}}                   }
            {{Ribbon Width}      0.30             {scale 0.10 10.00 0.10}            }
            {{Ribbon Height}     0.05             {scale 0.01 10.00 0.01}            }
        }
        { QuickSurf
            {{Resolution}       1.00    {scale     0.50 8.00 0.10} }
            {{Radius Scale}     1.0     {plusMinus 0.1  1}         }
            {{Density Isovalue} 0.5     {plusMinus 0.1  1}         }
            {{Grid Spacing}     1.0     {plusMinus 0.1  1}         }
            {{Surface Quality} {Medium} {menu Low Medium High Max} }
        }
        { MSMS
            {{Which Atoms}           {Selected}      {menu Selected All} }
            {{Representation Method} {Solid Surface} {menu {Solid Surface} Wireframe} }
            {{Sample Density}        1.5             {plusMinus 0.1 1} }
            {{Probe Radius}          1.5             {plusMinus 0.1 1} }
        }
        { Surf
            {{Representation Method}  {Solid Surface}  {menu {Solid Surface} Wireframe} }
            {{Probe Radius}           1.4              {plusMinus 0.1 1}                }
        }
        { Beads
            {{Sphere Scale}  1.0  {plusMinus 0.1 1 } }
            {{Sphere Resolution} 12 {plusMinus 1 {}} }
        }
        { Dotted
            {{Sphere Scale}  1.0  {plusMinus 0.1 1 } }
            {{Sphere Resolution} 12 {plusMinus 1 {}} }
        }
        { Solvent
            {{Representation Method}  {Points}  {menu Points Crosses Mesh} }
            {{Detail Level}           7         {plusMinus 1 {}          } }
            {{Probe Radius}           0.0       {plusMinus 0.1 1         } }
        }
    } {
        # build the parent
        set parentID [lindex $ele 0]
        .vmdPrefs.hlf.nb.reps.definitions.tv insert {} end -id $parentID -text "$parentID"
        # build the children
        for {set i 1} {$i < [llength $ele]} {incr i} {
            .vmdPrefs.hlf.nb.reps.definitions.tv insert $parentID end -values [lindex $ele $i]
        }
    }     
    
    #----------------------
    # materials
    #----------------------

    # add a tab for custom code
    ttk::frame $w.hlf.nb.materials
    $w.hlf.nb add $w.hlf.nb.materials -text "Materials"

    grid columnconfigure $w.hlf.nb.materials 0 -weight 1; # allows both header and body to stretch in width
    #grid rowconfigure $w.hlf.nb.colors 2 -weight 1; # allows body to stretch in height
    
    # build header section
    ttk::frame $w.hlf.nb.materials.header
    ttk::checkbutton $w.hlf.nb.materials.header.controlMode -text "Control Mode" -variable ::vmdPrefs::materialsControlMode -onvalue 1 -offvalue 0 \
        -command {
            # if turning command mode on, push the settings to VMD; this is to sync and helps keep things like material names straight
            if { $::vmdPrefs::materialsControlMode == 1 } { ::vmdPrefs::materialsPushSettingsToVMD }
        }
    ttk::button $w.hlf.nb.materials.header.query -text "Query VMD Materials" -command { ::vmdPrefs::materialsGetCurrentVMD }
    ttk::button $w.hlf.nb.materials.header.push -text "Push Materials to VMD" -command { ::vmdPrefs::materialsPushSettingsToVMD }
    ttk::button $w.hlf.nb.materials.header.defaults -text "Load VMD Defaults" -command { ::vmdPrefs::materialsGetDefaults; if { $::vmdPrefs::materialsControlMode } { ::vmdPrefs::materialsPushSettingsToVMD } }
    
    ttk::button $w.hlf.nb.materials.header.addMaterial -text "Add Material" \
        -command {
            set newName [::vmdPrefs::materialsGenerateNewName]
            if { [.vmdPrefs.hlf.nb.materials.body.tv selection] ne "" } {
                # copy current selected
                set dataList [.vmdPrefs.hlf.nb.materials.body.tv item [.vmdPrefs.hlf.nb.materials.body.tv selection] -values]
                lset dataList 0 $newName
                set insertIndex [expr {[.vmdPrefs.hlf.nb.materials.body.tv index [.vmdPrefs.hlf.nb.materials.body.tv selection]] + 1}]
                .vmdPrefs.hlf.nb.materials.body.tv insert {} $insertIndex -values $dataList
                
                if { $::vmdPrefs::materialsControlMode } {
                    # add the new material to the DB
                    material add $newName
                    # set the properties accordingly
                    set i 1
                    foreach prop {ambient diffuse specular shininess mirror opacity outline outlinewidth transmode} {
                        material change $prop [lindex $dataList 0] [lindex $dataList $i]
                        incr i
                    }
                    unset i
                }
                
            } else {
                # no current selection; zero out values and add at the end
                .vmdPrefs.hlf.nb.materials.body.tv insert {} end -values [list $newName 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0]
                
                if { $::vmdPrefs::materialsControlMode } {
                    # add the new material to the DB
                    material add $newName
                    # set the properties accordingly
                    foreach prop {ambient diffuse specular shininess mirror opacity outline outlinewidth transmode} { material change $prop $newName 0.000 }
                }
            }
         }
    ttk::button $w.hlf.nb.materials.header.delMaterial -text "Delete Material" \
        -command {
            set currName [.vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] Name]
            if { $currName ne "Opaque" && $currName ne "Transparent" } {
                .vmdPrefs.hlf.nb.materials.body.tv delete [.vmdPrefs.hlf.nb.materials.body.tv selection]
                set ::vmdPrefs::matName ""
                if { $::vmdPrefs::materialsControlMode && [lsearch [material list] $currName] != -1 } { material delete $currName }
            }
            unset currName
        }
    ttk::frame $w.hlf.nb.materials.header.moveMaterial
    ttk::button $w.hlf.nb.materials.header.moveMaterial.up -text [format "%c" 8593] \
        -command {
            # ID of current
            set currentID [.vmdPrefs.hlf.nb.materials.body.tv selection]
            # ID of previous
            if {[set previousID [.vmdPrefs.hlf.nb.materials.body.tv prev $currentID ]] ne ""} {
                # Index of previous
                set previousIndex [.vmdPrefs.hlf.nb.materials.body.tv index $previousID]
                # Move ahead of previous
                .vmdPrefs.hlf.nb.materials.body.tv move $currentID {} $previousIndex
                unset previousIndex
            }
            unset currentID previousID
        }
    ttk::button $w.hlf.nb.materials.header.moveMaterial.down -text [format "%c" 8595] \
        -command {
            # ID of current
            set currentID [.vmdPrefs.hlf.nb.materials.body.tv selection]
            # ID of Next
            if {[set previousID [.vmdPrefs.hlf.nb.materials.body.tv next $currentID ]] ne ""} {
                # Index of Next
                set previousIndex [.vmdPrefs.hlf.nb.materials.body.tv index $previousID]
                # Move below next
                .vmdPrefs.hlf.nb.materials.body.tv move $currentID {} $previousIndex
                unset previousIndex
            }
            unset currentID previousID
        }
    
    # grid header section
    grid $w.hlf.nb.materials.header -column 0 -row 0 -sticky ns -pady 8
    grid $w.hlf.nb.materials.header.controlMode -column 0 -row 0 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.materials.header.query -column 1 -row 0 -sticky nswe -padx "0 4" -pady "0 4"
    grid $w.hlf.nb.materials.header.push -column 2 -row 0 -sticky nswe -padx "0 4" -pady "0 4"
    grid $w.hlf.nb.materials.header.defaults -column 3 -row 0 -sticky nswe -pady "0 4"
    
    grid $w.hlf.nb.materials.header.addMaterial -column 1 -row 1 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.materials.header.delMaterial -column 2 -row 1 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.materials.header.moveMaterial -column 3 -row 1 -sticky nswe
    grid $w.hlf.nb.materials.header.moveMaterial.up -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.materials.header.moveMaterial.down -column 1 -row 0 -sticky nswe
    grid columnconfigure $w.hlf.nb.materials.header.moveMaterial {0 1} -uniform ct1 -weight 1
    
    grid columnconfigure $w.hlf.nb.materials.header {1 2 3} -uniform ct1


    # build/grid separator
    ttk::separator $w.hlf.nb.materials.sep1 -orient horizontal
    grid $w.hlf.nb.materials.sep1 -column 0 -row 1 -sticky nswe -pady 4
    
    
    # build 
    
    # build/grid the body
    ttk::frame $w.hlf.nb.materials.body
    grid $w.hlf.nb.materials.body -column 0 -row 2 -sticky nswe
    
    # configure appropriate stretching
    grid columnconfigure $w.hlf.nb.materials.body 0 -minsize 125 -weight 1
    grid columnconfigure $w.hlf.nb.materials.body {1 2 3 4 5 6 7 8} -minsize 75 -weight 0
    
    #
    ttk::treeview $w.hlf.nb.materials.body.tv -selectmode extended -yscroll "$w.hlf.nb.materials.body.tvScroll set"
        $w.hlf.nb.materials.body.tv configure \
            -column {Name Ambient Diffuse Specular Shininess Mirror Opacity Outline OutlineWidth AngleModTrans} \
            -displaycolumns {Name Ambient Diffuse Specular Shininess Mirror Opacity Outline OutlineWidth} \
            -show {headings} \
            -height 10
        $w.hlf.nb.materials.body.tv column Name -width 125 -stretch 1 -anchor center
        $w.hlf.nb.materials.body.tv heading Name -text "Name" -anchor center
        foreach ele {Ambient Diffuse Specular Shininess Mirror Opacity Outline OutlineWidth} {
            $w.hlf.nb.materials.body.tv column $ele -width 75 -stretch 0 -anchor center
            $w.hlf.nb.materials.body.tv heading $ele -text $ele -anchor center
        }
    ttk::scrollbar $w.hlf.nb.materials.body.tvScroll -orient vertical -command "$w.hlf.nb.materials.body.tv yview"
    
    # setup the tv binding
    bind $w.hlf.nb.materials.body.tv <KeyPress-Escape> { .vmdPrefs.hlf.nb.materials.body.tv selection set {}; set ::vmdPrefs::matName "" }
    bind $w.hlf.nb.materials.body.tv <KeyPress-Delete> {
        set matName [.vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] Name]
        if { $matName eq "Opaque" || $matName eq "Transparent"} {
            return
        } else {
            .vmdPrefs.hlf.nb.materials.body.tv delete [.vmdPrefs.hlf.nb.materials.body.tv selection]; set ::vmdPrefs::matName ""
        }
    }
    bind $w.hlf.nb.materials.body.tv <<TreeviewSelect>> {
        if { [.vmdPrefs.hlf.nb.materials.body.tv selection] eq "" } { return }
        
        # set the scales
        set data [.vmdPrefs.hlf.nb.materials.body.tv item [.vmdPrefs.hlf.nb.materials.body.tv selection] -values]
        set ::vmdPrefs::matName [lindex $data 0]
        set i 1
        foreach ele {Ambient Diffuse Specular Shininess Mirror Opacity Outline OutlineWidth} {
            set ::vmdPrefs::matScale-${ele} [lindex $data $i]
            incr i
        }
        unset i
        set ::vmdPrefs::matAMT [expr {int([lindex $data end])}]
        
        # modify the state of the delete material button (opaque and transparent cannot be deleted)
        set currName [.vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] Name]
        if { $currName eq "Opaque" || $currName eq "Transparent" } {
            .vmdPrefs.hlf.nb.materials.header.delMaterial configure -state disabled
        } else {
            .vmdPrefs.hlf.nb.materials.header.delMaterial configure -state normal
        }
        unset currName
    }
    
    # grid
    grid $w.hlf.nb.materials.body.tv -column 0 -columnspan 9 -row 0 -sticky nswe
    grid $w.hlf.nb.materials.body.tvScroll -column 9 -row 0 -sticky nswe
    
    # build/grid part I of the modifications section (name + angle-modified transparency)
    ttk::entry $w.hlf.nb.materials.body.nameLbl -textvariable ::vmdPrefs::matName -justify center -width 1
    ttk::button $w.hlf.nb.materials.body.changeName -text "Change Name" -width 12 \
        -command {
            set oldName [.vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] Name]
            if { [::vmdPrefs::materialsUniqNameCheck $::vmdPrefs::matName] == 1 } {
                # name change is uniq
                .vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] Name $::vmdPrefs::matName
                if { $::vmdPrefs::materialsControlMode } { material rename $oldName $::vmdPrefs::matName }
            } else {
                # name is not uniq, do nothing (for the time being)
                return
            }
        }
    ttk::checkbutton $w.hlf.nb.materials.body.amt -text "      Angle\n  Modulated\nTransparency" -onvalue 1 -offvalue 0 -variable ::vmdPrefs::matAMT \
        -command {
            .vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] AngleModTrans $::vmdPrefs::matAMT
            if { $::vmdPrefs::materialsControlMode } { material change transmode [.vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] Name] $::vmdPrefs::matAMT }
        }
    
    grid $w.hlf.nb.materials.body.nameLbl -column 0 -row 1 -sticky nswe -pady "2 0"
    grid $w.hlf.nb.materials.body.changeName -column 0 -row 2 -sticky ns
    grid $w.hlf.nb.materials.body.amt -column 0 -row 3 -rowspan 1 -sticky ns -pady "4 0"
    
    # build/grid part II of the modifications section programmatically (scales)
    set column 1
    foreach ele {Ambient Diffuse Specular Shininess Mirror Opacity Outline OutlineWidth} {
        scale $w.hlf.nb.materials.body.scale-${ele} \
            -orient vertical \
            -showvalue false \
            -from 1 -to 0 \
            -resolution 0.001 -digit 4 \
            -bg #78a4ff -activebackground #78a4ff \
            -variable ::vmdPrefs::matScale-${ele} \
            -command "::vmdPrefs::matUpdateScaleValues $ele"
        ttk::label $w.hlf.nb.materials.body.lbl-${ele} -width 5 -textvariable ::vmdPrefs::matScale-${ele} -anchor center
        
        grid $w.hlf.nb.materials.body.scale-${ele} -column $column -row 1 -rowspan 3 -pady "2 0"
        grid $w.hlf.nb.materials.body.lbl-${ele} -column $column -row 4 -sticky nswe
        incr column
    }
    unset column

    # outline is 0 to 4
    $w.hlf.nb.materials.body.scale-Outline configure -from 4    

    # fill the tv with data
    ::vmdPrefs::materialsGetCurrentVMD
    
    #----------------------
    # custom
    #----------------------

    # add a tab for custom code
    ttk::frame $w.hlf.nb.custom
    $w.hlf.nb add $w.hlf.nb.custom -text "Custom"
    
    # build
    ttk::treeview $w.hlf.nb.custom.tv -selectmode browse -yscroll "$w.hlf.nb.custom.tvScroll set"
        $w.hlf.nb.custom.tv configure -column {desc code} -displaycolumns {desc} -show {headings} -height 3
        $w.hlf.nb.custom.tv heading desc -text "Description" -anchor w
    
    # set treeview bindings
    bind $w.hlf.nb.custom.tv <<TreeviewSelect>> {
        set data [.vmdPrefs.hlf.nb.custom.tv item [.vmdPrefs.hlf.nb.custom.tv selection] -values]
        # copy description
        set ::vmdPrefs::customDesc [lindex $data 0]
        # copy code
        .vmdPrefs.hlf.nb.custom.code delete 0.0 end
        .vmdPrefs.hlf.nb.custom.code insert 0.0 [lindex $data 1]
    }
    
    ttk::scrollbar $w.hlf.nb.custom.tvScroll -orient vertical -command "$w.hlf.nb.custom.tv yview"    
    ttk::button $w.hlf.nb.custom.new -text "New" -command { .vmdPrefs.hlf.nb.custom.tv insert {} end -values [list "new" ""] }
    ttk::frame $w.hlf.nb.custom.move
    ttk::button $w.hlf.nb.custom.move.up -text $upArrow -width 1 \
        -command {
            # ID of current
            set currentID [.vmdPrefs.hlf.nb.custom.tv selection]
            # ID of previous
            if { [set previousID [.vmdPrefs.hlf.nb.custom.tv prev $currentID]] ne "" } {
                # index of previous
                set previousIndex [.vmdPrefs.hlf.nb.custom.tv index $previousID]
                # move ahead of previous
                .vmdPrefs.hlf.nb.custom.tv move $currentID {} $previousIndex
                unset previousIndex
            }
            unset currentID previousID
        }
    ttk::button $w.hlf.nb.custom.move.down -text $downArrow -width 1 \
        -command {
            # ID of current
            set currentID [.vmdPrefs.hlf.nb.custom.tv selection]
            # ID of next
            if { [set nextID [.vmdPrefs.hlf.nb.custom.tv next $currentID]] ne "" } {
                # index of next
                set nextIndex [.vmdPrefs.hlf.nb.custom.tv index $nextID]
                # move below next
                .vmdPrefs.hlf.nb.custom.tv move $currentID {} $nextIndex
                unset nextIndex
            }
            unset currentID nextID

        }
    ttk::separator $w.hlf.nb.custom.sep1 -orient horizontal
    ttk::button $w.hlf.nb.custom.delete -text "Delete" \
        -command {
            .vmdPrefs.hlf.nb.custom.tv delete [.vmdPrefs.hlf.nb.custom.tv selection]
            set ::vmdPrefs::customDesc ""
            .vmdPrefs.hlf.nb.custom.code delete 0.0 end
        }
    ttk::button $w.hlf.nb.custom.clear -text "Clear" \
        -command {
            .vmdPrefs.hlf.nb.custom.tv delete [.vmdPrefs.hlf.nb.custom.tv children {}]
            set ::vmdPrefs::customDesc ""
            .vmdPrefs.hlf.nb.custom.code delete 0.0 end            
        }

    ttk::separator $w.hlf.nb.custom.sep2 -orient horizontal
      
    ttk::label $w.hlf.nb.custom.descLbl -text "Description:" -anchor w
    ttk::entry $w.hlf.nb.custom.desc -textvariable ::vmdPrefs::customDesc
    ttk::button $w.hlf.nb.custom.update -text "Update" \
        -command {
            .vmdPrefs.hlf.nb.custom.tv set [.vmdPrefs.hlf.nb.custom.tv selection] desc $::vmdPrefs::customDesc
            .vmdPrefs.hlf.nb.custom.tv set [.vmdPrefs.hlf.nb.custom.tv selection] code [.vmdPrefs.hlf.nb.custom.code get 1.0 {end -1c}]
        }
    ttk::label $w.hlf.nb.custom.codeLbl -text "Code:" -anchor w
    text $w.hlf.nb.custom.code -wrap word -height 5 -yscrollcommand "$w.hlf.nb.custom.codeScroll set"
    ttk::scrollbar $w.hlf.nb.custom.codeScroll -orient vertical -command "$w.hlf.nb.custom.code yview"
        
    # grid
    grid $w.hlf.nb.custom.tv -column 0 -row 0 -rowspan 6 -sticky nswe
    grid $w.hlf.nb.custom.tvScroll -column 1 -row 0 -rowspan 6 -sticky nswe    
    grid $w.hlf.nb.custom.new -column 2 -row 0 -sticky nswe
    grid $w.hlf.nb.custom.move -column 2 -row 1 -sticky nswe
    grid $w.hlf.nb.custom.move.up -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.custom.move.down -column 1 -row 0 -sticky nswe
    grid $w.hlf.nb.custom.sep1 -column 2 -row 2 -sticky nswe
    grid $w.hlf.nb.custom.delete -column 2 -row 3 -sticky nswe
    grid $w.hlf.nb.custom.clear -column 2 -row 4 -sticky nswe
    
    grid $w.hlf.nb.custom.sep2 -column 0 -row 6 -columnspan 3 -sticky nswe -pady 4
    
    grid $w.hlf.nb.custom.descLbl -column 0 -row 7 -sticky nswe
    grid $w.hlf.nb.custom.update -column 2 -row 7 -sticky nswe
    grid $w.hlf.nb.custom.desc -column 0 -row 8 -sticky nswe
    grid $w.hlf.nb.custom.codeLbl -column 0 -row 9 -sticky nswe
    grid $w.hlf.nb.custom.code -column 0 -row 10 -sticky nswe
    grid $w.hlf.nb.custom.codeScroll -column 1 -row 10 -sticky nswe
    
    
    # configure how the frame adjusts
    grid columnconfigure $w.hlf.nb.custom 0 -weight 1
    grid rowconfigure $w.hlf.nb.custom 10 -weight 1
    grid columnconfigure $w.hlf.nb.custom.move {0 1} -weight 1
    
  
    #----------------------
    # VMDRC
    #----------------------
    # controls some internal settings that are recorded in vmdrcfiles

    # add a tab for vmdrc code
    ttk::frame $w.hlf.nb.vmdrc
    $w.hlf.nb add $w.hlf.nb.vmdrc -text "VMDRC"

    grid columnconfigure $w.hlf.nb.vmdrc 0 -weight 1

    # build vmdrc location
    ttk::frame $w.hlf.nb.vmdrc.rcpath
    ttk::label $w.hlf.nb.vmdrc.rcpath.lbl -text "VMDRC Path:" -anchor center
    ttk::entry $w.hlf.nb.vmdrc.rcpath.path -textvariable ::vmdPrefs::rcPath -justify left
    ttk::button $w.hlf.nb.vmdrc.rcpath.browse -text "Browse" \
        -command {
            set tempPath [tk_chooseDirectory -title "Select Directory for Writing VMDRC File"]
            if {![string eq $tempPath ""]} { set ::vmdPrefs::$tempPath}
        }

    # grid vmdrc location
    grid $w.hlf.nb.vmdrc.rcpath -column 0 -row 0 -sticky nswe -pady "10 0"
    grid $w.hlf.nb.vmdrc.rcpath.lbl    -column 0 -row 0 -sticky nswe
    grid $w.hlf.nb.vmdrc.rcpath.path   -column 1 -row 0 -sticky nswe
    grid $w.hlf.nb.vmdrc.rcpath.browse -column 2 -row 0 -sticky nswe

    grid columnconfigure $w.hlf.nb.vmdrc.rcpath 1 -weight 1

    # separator
    ttk::separator $w.hlf.nb.vmdrc.sep1 -orient horizontal
    grid $w.hlf.nb.vmdrc.sep1 -column 0 -row 1 -sticky nswe -padx 2 -pady 10
    
    # build on/off switches for activating/deactivating sections in a theme files
    ttk::frame $w.hlf.nb.vmdrc.switches
    ttk::label $w.hlf.nb.vmdrc.switches.lbl -text "Activate/Deactivate Sections of Theme File" -anchor w

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.menusOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(menus) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.menusOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(menus) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.menusLbl -text "Menus"

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.displayOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(display) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.displayOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(display) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.displayLbl -text "Display"

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.colorsAssignOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(colorsAssign) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.colorsAssignOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(colorsAssign) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.colorsAssignLbl -text "Colors -- Assign Colors to VMD Elements"

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.colorsDefineOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(colorsDefine) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.colorsDefineOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(colorsDefine) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.colorsDefineLbl -text "Colors -- Define Colors"

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.colorsScaleOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(colorsScale) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.colorsScaleOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(colorsScale) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.colorsScaleLbl -text "Colors -- Color Scale"

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.repsOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(reps) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.repsOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(reps) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.repsLbl -text "Representations"

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.materialsOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(materials) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.materialsOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(materials) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.materialsLbl -text "Materials"

    ttk::radiobutton $w.hlf.nb.vmdrc.switches.customOn -text "on" -variable ::vmdPrefs::vmdrcSwitches(custom) -value "1" -command {}
    ttk::radiobutton $w.hlf.nb.vmdrc.switches.customOff -text "off" -variable ::vmdPrefs::vmdrcSwitches(custom) -value "0" -command {}
    ttk::label $w.hlf.nb.vmdrc.switches.customLbl -text "Custom"

    # grid on/off switches for activating/deactivating sections in a theme files
    grid $w.hlf.nb.vmdrc.switches -column 0 -row 2 -sticky ns

    grid $w.hlf.nb.vmdrc.switches.lbl             -column 0 -row 0 -sticky nswe -columnspan 4

    grid $w.hlf.nb.vmdrc.switches.menusOn         -column 0 -row 1 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.menusOff        -column 1 -row 1 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.menusLbl        -column 2 -row 1 -sticky nswe
    grid $w.hlf.nb.vmdrc.switches.displayOn       -column 0 -row 2 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.displayOff      -column 1 -row 2 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.displayLbl      -column 2 -row 2 -sticky nswe
    grid $w.hlf.nb.vmdrc.switches.colorsAssignOn  -column 0 -row 3 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.colorsAssignOff -column 1 -row 3 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.colorsAssignLbl -column 2 -row 3 -sticky nswe
    grid $w.hlf.nb.vmdrc.switches.colorsDefineOn  -column 0 -row 4 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.colorsDefineOff -column 1 -row 4 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.colorsDefineLbl -column 2 -row 4 -sticky nswe
    grid $w.hlf.nb.vmdrc.switches.colorsScaleOn   -column 0 -row 5 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.colorsScaleOff  -column 1 -row 5 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.colorsScaleLbl  -column 2 -row 5 -sticky nswe
    grid $w.hlf.nb.vmdrc.switches.repsOn          -column 0 -row 6 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.repsOff         -column 1 -row 6 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.repsLbl         -column 2 -row 6 -sticky nswe
    grid $w.hlf.nb.vmdrc.switches.materialsOn     -column 0 -row 7 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.materialsOff    -column 1 -row 7 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.materialsLbl    -column 2 -row 7 -sticky nswe
    grid $w.hlf.nb.vmdrc.switches.customOn        -column 0 -row 8 -sticky nswe -padx "0 4"
    grid $w.hlf.nb.vmdrc.switches.customOff       -column 1 -row 8 -sticky nswe -padx "0 8"
    grid $w.hlf.nb.vmdrc.switches.customLbl       -column 2 -row 8 -sticky nswe


    #-------------------
    # try to read in the current VMDRC file
    global env
    switch [vmdinfo arch] {
        WIN64 -
        WIN32 { set vmdrcfilename [file join $env(HOME) vmd.rc] }
        default { set vmdrcfilename [file join $env(HOME) .vmdrc] }
    }
    # does it exist?
    if { [file exists $vmdrcfilename] } {
        # does it conform to vmdprefs formatting?
        set inFile [open $vmdrcfilename r]
        gets $inFile
        set inLine [string trim [gets $inFile]]
        close $inFile
        if { $inLine eq "\# VMDPrefs Theme File"} { ::vmdPrefs::readThemeFile $vmdrcfilename }
    }

    #-------------------
    # return
    return $w    
}
#=============================
#
# GENERAL PROCS
#
#=============================
proc ::vmdPrefs::init {} {
    # initialize the gui

    global env
        
    # load/write
    set ::vmdPrefs::loadPath ""
    set ::vmdPrefs::savePath ""
    
    # menus

    # display
    set ::vmdPrefs::displayControlMode 0 ; # control vmd properties based on widget settings
    array unset ::vmdPrefs::displaySettings; array set ::vmdPrefs::displaySettings {}
    ::vmdPrefs::displayGetCurrentVMD
    # OpenGL window position cannot be queried, initialize to reasonable values
    set ::vmdPrefs::displaySettings(reposX) 100
    set ::vmdPrefs::displaySettings(reposY) 600
    
    # colors
    set ::vmdPrefs::colorsControlMode 0
    set ::vmdPrefs::colorsCategoryColor "blue"
    set ::vmdPrefs::colorsR 0; set ::vmdPrefs::colorsRScale 0
    set ::vmdPrefs::colorsG 0; set ::vmdPrefs::colorsGScale 0 
    set ::vmdPrefs::colorsB 0; set ::vmdPrefs::colorsBScale 0
    set ::vmdPrefs::colorsScaleMethod [colorinfo scale method]
    set ::vmdPrefs::colorsScaleOffset [format %0.2f [colorinfo scale min]]
    set ::vmdPrefs::colorsScaleMidpoint [format %0.2f [colorinfo scale midpoint]]
    
    set ::vmdPrefs::colorsTclNamedColorsLbl ""
    array unset ::vmdPrefs::tclNamedColors; array set ::vmdPrefs::tclNamedColors {}
    array unset ::vmdPrefs::tclNamedColorsByRGB; array set ::vmdPrefs::tclNamedColorsByRGB {}
    ::vmdPrefs::initTclNamedColors
    set ::vmdPrefs::colorsLineIDs {}
    
    # representations
    set ::vmdPrefs::repsControlMode 0
    array unset ::vmdPrefs::reps
    array set ::vmdPrefs::reps {
        colorMethod Name
        colorID blue
        material Opaque
        drawMethod Lines
        selection all
    }
    
    set ::vmdPrefs::repDefPlusMinus ""
    set ::vmdPrefs::repDefPlusMinusIncr ""
    set ::vmdPrefs::repDefPlusMinusSigFigs ""
    set ::vmdPrefs::repDefScale "0.00"
    set ::vmdPrefs::repDefMenu ""
    
    # materials
    set ::vmdPrefs::materialsControlMode 0
    set ::vmdPrefs::matName ""
    set ::vmdPrefs::matAMT 0
    
    # custom
    set ::vmdPrefs::customDesc ""

    # vmdrc
    set ::vmdPrefs::rcPath $env(HOME)
    array set ::vmdPrefs::vmdrcSwitches { menus 1 display 1 colorsAssign 1 colorsDefine 1 colorsScale 1 reps 1 materials 1 custom 1}    
}
#=============================
proc ::vmdPrefs::writeVMDRC {} {
    # writes all settings to a valid vmdrc file in env(HOME)

    # localize relevant variables
    variable rcPath

    # determine what system-specific filename to look for
    switch [vmdinfo arch] {
        WIN64 -
        WIN32 { set vmdrcFileName vmd.rc }
        default { set vmdrcFileName .vmdrc }
    }

    # make sure that we have write permission to rcPath (defaults to env(HOME))

    if { ![file exists $rcPath] || ![file writable $rcPath] } {
        tk_messageBox -icon warning -type ok \
            -message "Cannot write VMDRC to the target directory:\n\n$rcPath\n\nMake sure that this folder exists and user has write permission."
        return
    }

    # if a vmdrc file already exists in env(HOME), make a backup copy
    if { [file exists [file join $rcPath $vmdrcFileName]] } {
        # check for an existing backup copy
        if { [file exists [file join $rcPath ${vmdrcFileName}.bak]] } {
            set reply [ tk_messageBox -icon warning -type yesnocancel \
                -message "A previous VMDRC backup file was found.\nOverwrite the old backup?" ]
            switch $reply {
                yes { file copy -force [file join $rcPath $vmdrcFileName] [file join $rcPath ${vmdrcFileName}.bak] }
                no { # pass }
                cancel { return }
            }
        } else {
            file copy -force [file join $rcPath $vmdrcFileName] [file join $rcPath ${vmdrcFileName}.bak]
        }
    }

    # write a theme file as the vmdrc
    ::vmdPrefs::writeThemeFile [file join $rcPath $vmdrcFileName]
}
#=============================
proc ::vmdPrefs::queryAllSettings {} {
    # queries all VMD settings that are present in vmdPrefs
    # with the exception of custom code

    ::vmdPrefs::menusQuery
    ::vmdPrefs::displayGetCurrentVMD
    ::vmdPrefs::colorsGetCurrentVMD
    ::vmdPrefs::repsQueryRepDefaults
    ::vmdPrefs::materialsGetCurrentVMD
}
#=============================
proc ::vmdPrefs::pushAllSettings {} {
    # pushes all vmdPrefs-controllable settings to VMD
    # currently excludes custom code

    ::vmdPrefs::menusPush
    ::vmdPrefs::displayPushSettingsToVMD
    ::vmdPrefs::colorsPushSettingsToVMD
    ::vmdPrefs::repsPushRepDefaults
    ::vmdPrefs::materialsPushSettingsToVMD

    # custom code
    #foreach ele [.vmdPrefs.hlf.nb.custom.tv children {}] {
    #    eval [.vmdPrefs.hlf.nb.custom.tv set $ele -code]
    #}
}
#=============================
proc ::vmdPrefs::writeThemeFile { filename } {
    # writes a theme file containing relevant settings
    
    # open the file for writing
    set outFile [open $filename w]
    
    # Header
    # ------
    puts $outFile "#"
    puts $outFile "# VMDPrefs Theme File"
    puts $outFile "#"
    puts $outFile "# VMDPrefs was written by:"
    puts $outFile "#    Christopher Mayne"
    puts $outFile "#    Tajkhorshid Laboratory (http://csbmb.beckman.illinois.edu)"
    puts $outFile "#    Beckman Institute for Advanced Science and Technology"
    puts $outFile "#    University of Illinois at Urbana-Champaign"
    puts $outFile "#"
    puts $outFile "# VMDPrefs is currently maintained by:"
    puts $outFile "#    Christopher Mayne | cmayne2@illinois.edu | http://www.ks.uiuc.edu/~mayne"
    puts $outFile "#"
    puts $outFile "# ------------------------------------------------------------------------------"
    puts $outFile "# WARNING: Do not edit this file by hand!"
    puts $outFile "#    VMDPrefs uses a very specific parsing scheme when reading theme files."
    puts $outFile "#    To avoid potentially corrupting the format, please use the GUI to make"
    puts $outFile "#    any setting changes and write a new theme file."
    puts $outFile "# ------------------------------------------------------------------------------"
    puts $outFile "#"
    puts $outFile ""


    puts $outFile ""
    # wait until VMD is done initializing before trying to setup the graphics environment
    puts $outFile "after idle \{"
    # only load the graphics releated settings if running VMD GUI (i.e. not in text-mode)
    puts $outFile "  if \{ \[info exists tk_version\] \} \{"
    # when running a .vmdrc files, VMD does not assume that the main menu is on
    # which should almost always be the case, and when not, it will be modified under menus
    puts $outFile ""
    puts $outFile "    \# High-Level Settings"
    puts $outFile "    menu main on"
    puts $outFile ""
    flush $outFile

    
    # Menus
    # -----
    # writes all menu on/off and x,y to file
    # note that non-existant menu names will not throw an error when setting x,y or on/off
    puts $outFile "    # <menus>"
    if { $::vmdPrefs::vmdrcSwitches(menus) == 1 } {
        puts $outFile "    # state : active"
    } else {
        puts $outFile "    # state : inactive"
    }
    puts $outFile "    if \{ $::vmdPrefs::vmdrcSwitches(menus) \} \{"

    foreach menuClass [.vmdPrefs.hlf.nb.menus.tv children {}] {
        foreach ele [.vmdPrefs.hlf.nb.menus.tv children $menuClass] {
            set data [.vmdPrefs.hlf.nb.menus.tv item $ele -values]
            # leave out undefined x,y positions
            if { [lindex $data 2] eq "" || [lindex $data 3] eq "" } {
                puts $outFile "      menu [lindex $data 0] [lindex $data 1]"    
            } else {
                puts $outFile "      menu [lindex $data 0] [lindex $data 1] ; menu [lindex $data 0] move [lindex $data 2] [lindex $data 3]"
            }
            unset data
        }
    }

    puts $outFile "    \}"
    puts $outFile "    # </menus>\n"
    flush $outFile
    
    # Display
    # -------
    puts $outFile "    # <display>"
    if { $::vmdPrefs::vmdrcSwitches(display) == 1 } {
        puts $outFile "    # state : active"
    } else {
        puts $outFile "    # state : inactive"
    }
    puts $outFile "    if \{ $::vmdPrefs::vmdrcSwitches(display) \} \{"

    puts $outFile "      display reposition $::vmdPrefs::displaySettings(reposX) $::vmdPrefs::displaySettings(reposY)"
    puts $outFile "      display resize $::vmdPrefs::displaySettings(resizeW) $::vmdPrefs::displaySettings(resizeH)"
    
    # programmatically write out all "display" settings
    foreach ele {
        projection
        depthcue cuemode cuestart cueend cuedensity
        eyesep focalLength
        height distance
        culling
        fps
        stereoswap
        cachemode
        rendermode
        shadows
        ambientocclusion aoambient aodirect
    } {
        puts $outFile "      display $ele $::vmdPrefs::displaySettings($ele)"
    }

    # other
    puts $outFile "      display stereo \"$::vmdPrefs::displaySettings(stereo)\""
    puts $outFile "      display nearclip set $::vmdPrefs::displaySettings(nearclip)"
    puts $outFile "      display farclip set $::vmdPrefs::displaySettings(farclip)"

    # write out backgrond (requires an if-then)
    if { $::vmdPrefs::displaySettings(background) eq "gradient" } {
        puts $outFile "      display backgroundgradient on"
    } else {
        puts $outFile "      display backgroundgradient off"
    }
    
    # axes
    puts $outFile "      axes location $::vmdPrefs::displaySettings(axes)"
    
    # write out lights
    foreach lightNum {0 1 2 3} {
        puts $outFile "      light $lightNum $::vmdPrefs::displaySettings(light${lightNum})"
    }    
    
    # wrap up the display section
    puts $outFile "    \}"
    puts $outFile "    # </display>\n"
    flush $outFile

    
    # Colors
    # ------
    puts $outFile "    # <colors>"

    # color assignments
    if { $::vmdPrefs::vmdrcSwitches(colorsAssign) == 1 } {
        puts $outFile "    # state (assign) : active"
    } else {
        puts $outFile "    # state (assign) : inactive"
    }
    puts $outFile "    if \{ $::vmdPrefs::vmdrcSwitches(colorsAssign) \} \{"
    foreach category [.vmdPrefs.hlf.nb.colors.body.categoryTv children {}] {
        foreach ele [.vmdPrefs.hlf.nb.colors.body.categoryTv children [list $category]] {
            lassign [.vmdPrefs.hlf.nb.colors.body.categoryTv item $ele -values] element color
            puts $outFile "      color add item $category [list $element] $color"
            puts $outFile "      color $category [list $element] $color"
            unset element color
        }
    }
    puts $outFile "    \}"

    # color definitions
    if { $::vmdPrefs::vmdrcSwitches(colorsDefine) == 1 } {
        puts $outFile "    # state (define) : active"
    } else {
        puts $outFile "    # state (define) : inactive"
    }
    puts $outFile "    if \{ $::vmdPrefs::vmdrcSwitches(colorsDefine) \} \{"
    foreach ele [.vmdPrefs.hlf.nb.colors.body.colorDefTv children {}] {
        lassign [.vmdPrefs.hlf.nb.colors.body.colorDefTv item $ele -values] ind color r g b
        puts $outFile "      color change rgb $color [format %0.3f $r] [format %0.3f $g] [format %0.3f $b]"
        unset ind color r g b    
    }
    puts $outFile "    \}"
    
    # color scales
    if { $::vmdPrefs::vmdrcSwitches(colorsScale) == 1 } {
        puts $outFile "    # state (scale) : active"
    } else {
        puts $outFile "    # state (scale) : inactive"
    }
    puts $outFile "    if \{ $::vmdPrefs::vmdrcSwitches(colorsScale) \} \{"
    puts $outFile "      color scale method $::vmdPrefs::colorsScaleMethod"
    puts $outFile "      color scale midpoint $::vmdPrefs::colorsScaleMidpoint"
    puts $outFile "      color scale min $::vmdPrefs::colorsScaleOffset"
    puts $outFile "    \}"

    puts $outFile "    # </colors>\n"
    flush $outFile

    # Representations
    # ---------------
    puts $outFile "    # <representations>"
    if { $::vmdPrefs::vmdrcSwitches(reps) == 1 } {
        puts $outFile "    # state : active"
    } else {
        puts $outFile "    # state : inactive"
    }
    puts $outFile "    if \{ $::vmdPrefs::vmdrcSwitches(reps) \} \{"

    if { $::vmdPrefs::reps(colorMethod) eq "ColorID" } {
        set colorIndMap [colorinfo colors]
        set colorInd [lsearch -exact $colorIndMap $::vmdPrefs::reps(colorID)]
        puts $outFile "      mol default color \{$::vmdPrefs::reps(colorMethod) $colorInd\}"
    } else {
        puts $outFile "      mol default color $::vmdPrefs::reps(colorMethod)"
    }
    puts $outFile "      mol default material $::vmdPrefs::reps(material)"
    puts $outFile "      mol default style $::vmdPrefs::reps(drawMethod)"
    puts $outFile "      mol default selection \{$::vmdPrefs::reps(selection)\}"

    puts $outFile "    \}"
    puts $outFile "    # </representations>\n"
    
    # Materials
    # ---------
    puts $outFile "    # <materials>"
    if { $::vmdPrefs::vmdrcSwitches(materials) == 1 } {
        puts $outFile "    # state : active"
    } else {
        puts $outFile "    # state : inactive"
    }
    puts $outFile "    if \{ $::vmdPrefs::vmdrcSwitches(materials) \} \{"

    puts $outFile "      set matsVMD \[material list\]"
    set matsTV {}
    foreach ele [.vmdPrefs.hlf.nb.materials.body.tv children {}] {
        lappend matsTV [.vmdPrefs.hlf.nb.materials.body.tv set $ele Name]
    }
    puts $outFile "      set matsVMDPREFS [list $matsTV]"
    puts $outFile "      # deletions"
    puts $outFile "      foreach ele \$matsVMD { if { \[lsearch \$matsVMDPREFS \$ele\] == -1 } { material delete \$ele } }"
    puts $outFile "      # additions"
    puts $outFile "      foreach ele \$matsVMDPREFS { if { \[lsearch \$matsVMD $ele\] == -1 } { material add \$ele } }"
    puts $outFile "      # set/change properties"
    puts $outFile "      set properties {ambient diffuse specular shininess mirror opacity outline outlinewidth transmode}"
    puts $outFile "      foreach ele {"
    foreach ele [.vmdPrefs.hlf.nb.materials.body.tv children {}] {
        puts $outFile "        [list [.vmdPrefs.hlf.nb.materials.body.tv item $ele -values]]"
    }
    puts $outFile "      } {"
    puts $outFile "        set i 1"
    puts $outFile "        foreach prop \$properties { material change \$prop \[lindex \$ele 0\] \[lindex \$ele \$i\]; incr i }"
    puts $outFile "        unset i"
    puts $outFile "      }"
    
    puts $outFile "    \}"
    puts $outFile "    # </materials>\n"
    

    # end of info exists tk_version if/then statement
    # (allow custom code to run even in text mode)
    puts $outFile "\n  \}\n"

    
    # Custom
    # ------
    puts $outFile "  # <custom>"
    if { $::vmdPrefs::vmdrcSwitches(custom) == 1 } {
        puts $outFile "  # state : active"
    } else {
        puts $outFile "  # state : inactive"
    }
    puts $outFile "  if \{ $::vmdPrefs::vmdrcSwitches(custom) \} \{"

    foreach ele [.vmdPrefs.hlf.nb.custom.tv children {}] {
        set data [.vmdPrefs.hlf.nb.custom.tv item $ele -values]
        puts $outFile "\n    # <custom entry>"
        puts $outFile "    # Description: [lindex $data 0]"
        #puts $outFile "    [lindex $data 1]"
        puts $outFile "[lindex $data 1]"
        puts $outFile "    # </custom entry>\n"
        unset data
        flush $outFile
    }

    puts $outFile "  \}"
    puts $outFile "  # </custom>"

    
    # clean up
    # close the after idle statement
    puts $outFile "\n\}\n"
    puts $outFile "# End Theme File"
    flush $outFile
    close $outFile
}
#=============================
proc ::vmdPrefs::readThemeFile { filename } {
    # reads in a vmdPrefs theme file and sets parameters accordingly

    # -----------------
    # CHECK FOR VMDPREFS HEADER TAG
    # -----------------
    # open the file for reading
    set inFile [open $filename r]
    # get the second line
    gets $inFile
    set inLine [string trim [gets $inFile]]
    # simple check that we're trying to read a valid theme file
    if { $inLine ne "\# VMDPrefs Theme File"} {
        tk_messageBox -icon warning -type ok \
            -message "Invalid file header.  This file does not appear to be a VMDPrefs theme file, or the file format has been corrupted."
        close $inFile
        return
    }

    # -----------------
    # RESET TO DEFAULTS
    # -----------------
    # reset menus tab to defaults
    # common
    foreach ele [.vmdPrefs.hlf.nb.menus.tv children common] {
        if { $ele eq "main" } {
            .vmdPrefs.hlf.nb.menus.tv set main on_off on
        } else {
            .vmdPrefs.hlf.nb.menus.tv item $ele -values [list $ele off 0 0]
        }
    }
    # other
    foreach ele [.vmdPrefs.hlf.nb.menus.tv children other] {
        .vmdPrefs.hlf.nb.menus.tv item $ele -values [list $ele off 0 0]
    }
    
    # reset colors tab to defaults
    eval ::vmdPrefs::colorsResetToDefaults
    
    # reset custom tab to defaults
    .vmdPrefs.hlf.nb.custom.tv delete [.vmdPrefs.hlf.nb.custom.tv children {}]
    set ::vmdPrefs::customDesc ""

    # -----------------
    # READ THEME FILE
    # -----------------
    set readState 0
    
    while { ![eof $inFile] } {
        set inLine [gets $inFile]

        # for debugging, uncomment below to help ID faulty lines
        #puts "processing line: $inLine"; flush stdout
        
        switch -regexp $inLine {
            {# <menus>} { set readState "menus" }
            {# </menus>} { set readState 0 }
            {# <display>} { set readState "display" }
            {# </display>} { set readState 0 }
            {# <colors>} { set readState "colors" }
            {# </colors>} { set readState 0 }
            {# <representations>} { set readState "representations" }
            {# </representations>} { set readState 0 }
            {# <materials>} { set readState "materials" }
            {# </materials>} { set readState 0 }
            {# <custom>} { set readState "custom" }
            {# </custom>} { set readState 0 }
            {# <custom entry>} {
                set readState "customEntry" 
                set customCodeDesc {}
                set customCodeLines {}
            }
            {# </custom entry>} {
                set readState "custom"
                .vmdPrefs.hlf.nb.custom.tv insert {} end -values [list $customCodeDesc [join $customCodeLines "\n"]]
            }
            default {
                # here's where the action is!
                switch -exact $readState {
                    {0} { continue }
                    {menus} {
                        set inLine [string trim $inLine]
                        switch -regexp $inLine {
                            {^\# state.*} {
                                # set the state flag
                                if { [lindex $inLine end] eq "active" } {
                                    set ::vmdPrefs::vmdrcSwitches(menus) 1
                                } else {
                                    set ::vmdPrefs::vmdrcSwitches(menus) 0
                                }
                                # burn a line to bypass the active/inactive constrol structures
                                gets $inFile
                                continue
                            }

                            {^\#.*} { continue }
                            {^$}    { continue }
                            {^\}$}  { continue }
                            default {
                                # read in menu data; catch errors, like an unrecognized menu
                                # check if x,y position is given or not.  if not given, assume 0,0 (vmd default)
                                if { [llength [string trim $inLine]] == 3 } {
                                    catch { .vmdPrefs.hlf.nb.menus.tv item [lindex $inLine 1] -values [list [lindex $inLine 1] [lindex $inLine 2] {} {}] }        
                                } else {
                                    catch { .vmdPrefs.hlf.nb.menus.tv item [lindex $inLine 1] -values [list [lindex $inLine 1] [lindex $inLine 2] [lindex $inLine 7] [lindex $inLine 8]] }        
                                }
                            }
                        }
                        
                    }

                    {display} {
                        set inLine [string trim $inLine]
                        switch -regexp $inLine {
                            {^\# state.*} {
                                # set the state flag
                                if { [lindex $inLine end] eq "active" } {
                                    set ::vmdPrefs::vmdrcSwitches(display) 1
                                } else {
                                    set ::vmdPrefs::vmdrcSwitches(display) 0
                                }
                                # burn a line to bypass the active/inactive control structure
                                gets $inFile
                                continue
                            }
                            {^\#.*} { continue }
                            {^$}    { continue }
                            {^\}$}  { continue }

                            {^display reposition.*} {
                                set ::vmdPrefs::displaySettings(reposX) [lindex $inLine 2]
                                set ::vmdPrefs::displaySettings(reposY) [lindex $inLine 3]
                            }
                            
                            {^display resize.*} {
                                set ::vmdPrefs::displaySettings(resizeW) [lindex $inLine 2]
                                set ::vmdPrefs::displaySettings(resizeH) [lindex $inLine 3]
                            }
                            
                            {^display nearclip.*} { set ::vmdPrefs::displaySettings(nearclip) [lindex $inLine 3] }
                            {^display farclip.*} { set ::vmdPrefs::displaySettings(farclip) [lindex $inLine 3] }
                            
                            {^display.*} {
                                if { [lindex $inLine 1] eq "backgroundgradient" } {
                                    if { [lindex $inLine 2] eq "off" } {
                                        set ::vmdPrefs::displaySettings(background) "color"
                                    } else {
                                        set ::vmdPrefs::displaySettings(background) "gradient"
                                    }
                                } else {
                                    set ::vmdPrefs::displaySettings([lindex $inLine 1]) [lindex $inLine 2]
                                }
                            }
                            
                            {^axes location.*} {
                                set ::vmdPrefs::displaySettings(axes) [lindex $inLine 2]
                            }
                            
                            {^light.*} {
                                set ::vmdPrefs::displaySettings(light[lindex $inLine 1]) [lindex $inLine 2]
                            }
                            
                            default { continue }
                        }
                    }
                    {colors} {
                        set inLine [string trim $inLine]
                        # look for the state flags
                        if { [lindex $inLine 1] eq "state" } {
                            switch -exact [lindex $inLine 2] {
                                "(assign)" {
                                    if { [lindex $inLine end] eq "active" } {
                                        set ::vmdPrefs::vmdrcSwitches(colorsAssign) 1
                                    } else {
                                        set ::vmdPrefs::vmdrcSwitches(colorsAssign) 0
                                    }
                                }

                                "(define)" {
                                    if { [lindex $inLine end] eq "active" } {
                                        set ::vmdPrefs::vmdrcSwitches(colorsDefine) 1
                                    } else {
                                        set ::vmdPrefs::vmdrcSwitches(colorsDefine) 0
                                    }                                    
                                }

                                "(scale)" {
                                    if { [lindex $inLine end] eq "active" } {
                                        set ::vmdPrefs::vmdrcSwitches(colorsDefine) 1
                                    } else {
                                        set ::vmdPrefs::vmdrcSwitches(colorsDefine) 0
                                    }                                                                        
                                }
                            }
                            # burn a line to bypass the active/inactive constrol structure
                            gets $inFile
                            continue
                        }

                        # skip lines that are commented out, blank lines, control structure closing bracket, or explicit addition to category db (due to brute force technique)
                        if { [regexp {^\#.*} $inLine] } { continue }
                        if { [regexp {^$} $inLine] } { continue }
                        if { $inLine eq "\}" } { continue }
                        if { [regexp {^color add item.*} $inLine] } { continue }

                        # try to match color definitions first, then category color assignments second
                        switch -glob $inLine {
                            
                            {color scale method*} {
                                set ::vmdPrefs::colorsScaleMethod [lindex $inLine end]
                            } 
                            
                            {color scale midpoint*} {
                                set ::vmdPrefs::colorsScaleMidpoint [lindex $inLine end]
                            }
                            
                            {color scale min*} {
                                set ::vmdPrefs::colorsScaleOffset [lindex $inLine end]
                            }

                            {color change rgb*} {
                                # color definitions
                                lassign [lrange $inLine 3 6] color r g b
                                .vmdPrefs.hlf.nb.colors.body.colorDefTv set $color r $r
                                .vmdPrefs.hlf.nb.colors.body.colorDefTv set $color g $g
                                .vmdPrefs.hlf.nb.colors.body.colorDefTv set $color b $b
                                unset color r g b
                            }
                            
                            default {
                                # category assignments
                                if { [llength $inLine] == 4 } {
                                    lassign [lrange $inLine 1 3] category element color
                                } else {
                                    # when elements have spaces in name (e.g. Structure, Alpha Helix)
                                    set category [lindex $inLine 1]
                                    set element [lrange $inLine 2 end-1]
                                    set color [lindex $inLine end]
                                }
                                
                                # some elements are only filled out when a structure bearing those elements is loaded (e.g. Chain, Segname)
                                if { [catch {.vmdPrefs.hlf.nb.colors.body.categoryTv set $category.$element color $color}] } {
                                    # if an error is thrown, the element has to be added first before it can be set
                                    color add item $category $element $color
                                    .vmdPrefs.hlf.nb.colors.body.categoryTv insert $category end -id "$category.$element" -values [list $element $color]
                                } 
                                unset category element color
                            }
                        }
                    }

                    {representations} {
                        set inLine [string trim $inLine]
                        switch -regexp $inLine {
                            {^\# state.*} {
                                if { [lindex $inLine end] eq "active" } {
                                    set ::vmdPrefs::vmdrcSwitches(reps) 1
                                } else {
                                    set ::vmdPrefs::vmdrcSwitches(reps) 0
                                }
                                # burn a line to bypass the active/inactive control structure
                                gets $inFile
                                continue
                            }

                            {^$}    { continue }
                            {^\#.*} { continue }
                            {^\}}   { continue }

                            default {
                                set attribute [lindex $inLine 2]
                                set value [lindex $inLine 3]
                                switch -exact $attribute {
                                    {color} {
                                        if { [llength $value] == 2 } {
                                            set ::vmdPrefs::reps(colorMethod) [lindex $value 0]
                                            set ::vmdPrefs::reps(colorID) [lindex $value 1]
                                        } else {
                                            set ::vmdPrefs::reps(colorMethod) $value
                                        }
                                    }
                                    {material} { set ::vmdPrefs::reps(material) $value }
                                    {style} { set ::vmdPrefs::reps(drawMethod) $value }
                                    {selection} { set ::vmdPrefs::reps(selection) $value }
                                    default { continue }
                                }; # end switch
                                unset attribute; unset value                                
                            }
                        }
                    }

                    {materials} {
                        set inLine [string trim $inLine]
                        # the vmdrc file contains some control structures in this section
                        # we are only looking for the state flags and the "set properties" portion
                        if { [regexp {^\# state.*} $inLine] } {
                            if { [lindex $inLine end] eq "active" } {
                                set ::vmdPrefs::vmdrcSwitches(materials) 1
                            } else {
                                set ::vmdPrefs::vmdrcSwitches(materials) 0
                            }
                        } elseif { [string match "set properties *" $inLine] } {
                            # clear out the tv
                            .vmdPrefs.hlf.nb.materials.body.tv delete [.vmdPrefs.hlf.nb.materials.body.tv children {}]
                            # burn a line
                            gets $inFile
                            # read until the end of the list
                            while { [set inLine [string trim [gets $inFile]]] ne "\} \{"} {
                                set data [string range $inLine 1 end-1]
                                if { [llength $data] == 9 } {
                                    # material definition that pre-dates the Mirror property
                                    # insert the mirror setting, defaulting to 0
                                    set data [linsert $data 5 0.000]
                                }
                                .vmdPrefs.hlf.nb.materials.body.tv insert {} end -id [lindex $data 0] -values $data
                            }
                        } else { continue }
                    }
                                        
                    {custom} {
                        set inLine [string trim $inLine]
                        # we are only looking for the custom state flag
                        # all other custom-related info should be in custom entry blocks
                        if { [regexp {^\# state.*} $inLine] } {
                            if { [lindex $inLine end] eq "active" } {
                                set ::vmdPrefs::vmdrcSwitches(custom) 1
                            } else {
                                set ::vmdPrefs::vmdrcSwitches(custom) 0
                            }
                        }
                    }

                    {customEntry} {
                        # note that we don't trim the inLine, this is to preserve any spacing
                        # that the user has entered in their code block
                        if { [regexp "\# Description:*" $inLine] } {
                            set customCodeDesc [string trim [lrange $inLine 2 end]]
                        } else {
                            # note: the vmdrc writer indents custom code by 4 spaces in the
                            #       theme file, which should be removed before loading
                            #lappend customCodeLines [string range $inLine 4 end]
                            lappend customCodeLines $inLine
                        }
                    }
                }
            }
        }; # end outer switch
    }; # end while


    # clean up
    close $inFile
}
#=============================
proc ::vmdPrefs::null { garbage } {
    # swallows any arguments passed to it
    # i.e., does nothing
    return
}
#=============================
#
# MENUS TAB RELATED PROCS
#
#=============================
proc ::vmdPrefs::menusQuery {} {
    # queries the status and location of all menus in the tv

    foreach node [.vmdPrefs.hlf.nb.menus.tv children {}] {
        foreach ele [.vmdPrefs.hlf.nb.menus.tv children $node] {
            set status [menu $ele status]
            lassign [menu $ele loc] x y
            # x,y = 0,0 is an indicator that menu position is unset
            if { $x == 0 && $y == 0 } {
                .vmdPrefs.hlf.nb.menus.tv item $ele -values [list $ele $status {} {}]
            } else {
                .vmdPrefs.hlf.nb.menus.tv item $ele -values [list $ele $status $x $y]
            }
            unset status x y
        }
    }
}
#=============================
proc ::vmdPrefs::menusPush {} {
    # pushes the menu settings to VMD

    foreach node [.vmdPrefs.hlf.nb.menus.tv children {}] {
        foreach ele [.vmdPrefs.hlf.nb.menus.tv children $node] {
            lassign [.vmdPrefs.hlf.nb.menus.tv item $ele -values] name status x y
            menu $name $status
            # only try to move valid x,y positions
            if { !($x eq "" || $y eq "") } { menu $name move $x $y }
            unset status x y
        }
    }
}
#=============================
#
# DISPLAY TAB RELATED PROCS
#
#=============================
proc ::vmdPrefs::displayPlusMinus { lbl scale } {
    # runs the -/+ buttons for display tab
    
    if { $scale < 0 } {
        # - opperation
        if { $::vmdPrefs::displaySettings($lbl) <= [expr {abs($scale)}] && $lbl ne "distance" } {
            # substraction would have made negative, set to zero (unless distance, which can be neg.)
            set ::vmdPrefs::displaySettings($lbl) 0.00
        } else {
            set ::vmdPrefs::displaySettings($lbl) [format %0.2f [expr { $::vmdPrefs::displaySettings($lbl) + $scale }]]
        }
    } elseif { $scale > 0 } {
        # + opperation
        set ::vmdPrefs::displaySettings($lbl) [format %0.2f [expr { $::vmdPrefs::displaySettings($lbl) + $scale }]]
    } else {
        return
    }
    
    # if running in control mode, tell vmd to change the value
    if { $::vmdPrefs::displayControlMode } {
        if { $lbl eq "nearclip" || $lbl eq "farclip" } {
            eval [concat display $lbl set $::vmdPrefs::displaySettings($lbl)]
        } else {
            eval [concat display $lbl $::vmdPrefs::displaySettings($lbl)]
        }
    }    
}
#=============================
proc ::vmdPrefs::displayFormatManualEntry { lbl } {
    # formats manually entered value into edit box; updates vmd if in interactive mode
    set ::vmdPrefs::displaySettings($lbl) [expr { double(round(100*$::vmdPrefs::displaySettings($lbl)))/100 }]
    if { $::vmdPrefs::displayControlMode } {
        eval [concat display $lbl $::vmdPrefs::displaySettings($lbl)]
    }
}
#=============================
proc ::vmdPrefs::displayGetCurrentVMD {} {
    # loads current VMD settings into display tab
    
    # NOTE: the current xy for openGL window cannot be queried, skip
    lassign [display get size] ::vmdPrefs::displaySettings(resizeW) ::vmdPrefs::displaySettings(resizeH)
    
    # standard "display get" settings
    foreach ele {
        {projection 0}
        {depthcue  0}
        {cuemode 0}
        {cuestart 1}
        {cueend 1}
        {cuedensity 1}
        {nearclip 1}
        {farclip 1}
        {eyesep 1}
        {focalLength 1}
        {height 1}
        {distance 1}
        {culling 0}
        {fps 0}
        {stereoswap 0}
        {cachemode 0}
        {rendermode 0}
        {shadows 0}
        {ambientocclusion 0}
        {aoambient 1}
        {aodirect 1}
    } {
        lassign $ele lbl reqRound
        set ::vmdPrefs::displaySettings($lbl) [display get $lbl]
        
        if { $reqRound } {
            set ::vmdPrefs::displaySettings($lbl) [format %0.2f $::vmdPrefs::displaySettings($lbl)]
        }
    }
    
    # specialized settings
    set ::vmdPrefs::displaySettings(axes) [axes location]
    set ::vmdPrefs::displaySettings(stereo) [lindex [display get stereo] 0]
    foreach ele {0 1 2 3} {
        set ::vmdPrefs::displaySettings(light${ele}) [lindex [light $ele status] 0]
    }
    if { [display get backgroundgradient] eq "off" } {
        set ::vmdPrefs::displaySettings(background) "color"
    } else {
        set ::vmdPrefs::displaySettings(background) "gradient"
    }
    set ::vmdPrefs::displaySettings(stage) [stage location]   
}
#=============================
proc ::vmdPrefs::displayPushSettingsToVMD {} {
    # updates VMD settings with current values
    
    # programmatically set
    foreach ele {
                projection depthcue cuemode
                cuestart cueend cuedensity
                eyesep focalLength height distance
                culling fps stereoswap
                cachemode rendermode shadows
                ambientocclusion aoambient aodirect
    } {
        eval [list display $ele $::vmdPrefs::displaySettings($ele)]
    }


    # manually set
    display reposition $::vmdPrefs::displaySettings(reposX) $::vmdPrefs::displaySettings(reposY)
    display resize $::vmdPrefs::displaySettings(resizeW) $::vmdPrefs::displaySettings(resizeH)
    display nearclip set $::vmdPrefs::displaySettings(nearclip)
    display farclip set $::vmdPrefs::displaySettings(farclip)
    display stereo [list $::vmdPrefs::displaySettings(stereo)]
    stage location $::vmdPrefs::displaySettings(stage)
    axes location $::vmdPrefs::displaySettings(axes)
    foreach ele {0 1 2 3} {
        light $ele $::vmdPrefs::displaySettings(light${ele})
    }
    if { $::vmdPrefs::displaySettings(background) eq "gradient" } {
        display backgroundgradient on
    } else {
        display backgroundgradient off
    }
}
#=============================
#
# COLORS TAB RELATED PROCS
#
#=============================
proc ::vmdPrefs::colorsCategoryColorSet { color } {
    # sets all selected entries to a given color
    foreach ele [.vmdPrefs.hlf.nb.colors.body.categoryTv selection] {
        if { [.vmdPrefs.hlf.nb.colors.body.categoryTv parent $ele] ne "" } {
            .vmdPrefs.hlf.nb.colors.body.categoryTv set $ele color $color
            if { $::vmdPrefs::colorsControlMode } {
                lassign [split $ele .] category element
                color $category $element $color
                unset category element
            }
        }
    }
}
#=============================
proc ::vmdPrefs::colorsUpdateColorBox {} {
    # updates the colored box when changing rgb values
    variable colorsR
    variable colorsG
    variable colorsB
    .vmdPrefs.hlf.nb.colors.body.colorBox configure -bg [format "#%02x%02x%02x" $colorsR $colorsG $colorsB]
}
#=============================
proc ::vmdPrefs::colorsLookupByRGB {} {
    # looks to see if there is a tcl named color with that rgb
    # returns the name or "custom"
    
    variable tclNamedColorsByRGB
    variable colorsR
    variable colorsG
    variable colorsB
    
    if { [info exists tclNamedColorsByRGB([list $colorsR $colorsG $colorsB])] } {
        return $::vmdPrefs::tclNamedColorsByRGB([list $colorsR $colorsG $colorsB])
    } else {
        return "custom"
    }    
}
#=============================
proc ::vmdPrefs::colorsUpdateScales {} {
    # updates the rgb scales based on rgb entry boxes
    
    set ::vmdPrefs::colorsRScale $::vmdPrefs::colorsR
    set ::vmdPrefs::colorsGScale $::vmdPrefs::colorsG
    set ::vmdPrefs::colorsBScale $::vmdPrefs::colorsB    
}
#=============================
proc ::vmdPrefs::colorsGetCurrentVMD {} {
    # queries VMDs current color settings
    
    # Category TV
    # clear out existing
    .vmdPrefs.hlf.nb.colors.body.categoryTv delete [.vmdPrefs.hlf.nb.colors.body.categoryTv children {}]

    # lookup data and fill
    foreach cat [colorinfo categories] {
        # build the parent
        .vmdPrefs.hlf.nb.colors.body.categoryTv insert {} end -id "$cat" -text "$cat"
        # build the children
        foreach ele [colorinfo category $cat] {
            .vmdPrefs.hlf.nb.colors.body.categoryTv insert $cat end -id "$cat.$ele" -values [list $ele [colorinfo category $cat $ele]]
        }
    }
    
    # Color Definition TV
    # clear out existing
    .vmdPrefs.hlf.nb.colors.body.colorDefTv delete [.vmdPrefs.hlf.nb.colors.body.colorDefTv children {}]

    # lookup data and fill
    foreach color [colorinfo colors] {
        set ind [colorinfo index $color]
        lassign [colorinfo rgb $color] r g b
        .vmdPrefs.hlf.nb.colors.body.colorDefTv insert {} end -id $color -values [list $ind $color [format "%.3f" $r] [format "%.3f" $g] [format "%.3f" $b]]
        unset ind r g b
    }
    
    # Color Scale
    set ::vmdPrefs::colorsScaleMethod [colorinfo scale method]
    set ::vmdPrefs::colorsScaleOffset [format %0.2f [colorinfo scale min]]
    set ::vmdPrefs::colorsScaleMidpoint [format %0.2f [colorinfo scale midpoint]]
    # update the color scale box
    ::vmdPrefs::colorsUpdateColorScaleBox
    
    # DONE
}
#=============================
proc ::vmdPrefs::colorsPushSettingsToVMD {} {
    # pushes color settings to VMD
    
    # Category TV
    foreach category [.vmdPrefs.hlf.nb.colors.body.categoryTv children {}] {
        foreach ele [.vmdPrefs.hlf.nb.colors.body.categoryTv children [list $category]] {
            lassign [.vmdPrefs.hlf.nb.colors.body.categoryTv item $ele -values] element color
            color $category $element $color
            unset element color
        }
    }
    
    # Color TV
    foreach ele [.vmdPrefs.hlf.nb.colors.body.colorDefTv children {}] {
        lassign [.vmdPrefs.hlf.nb.colors.body.colorDefTv item $ele -values] ind color r g b
        color change rgb $color [format %0.3f $r] [format %0.3f $g] [format %0.3f $b]
        unset ind color r g b    
    }
    
    # color scale
    color scale method $::vmdPrefs::colorsScaleMethod
    color scale min $::vmdPrefs::colorsScaleOffset
    color scale midpoint $::vmdPrefs::colorsScaleMidpoint    
}
#=============================
proc ::vmdPrefs::colorsResetToDefaults {} {
    # resets color settings to standard VMD defaults (hard coded)
    
    # category
    # clear the existing data
    .vmdPrefs.hlf.nb.colors.body.categoryTv delete [.vmdPrefs.hlf.nb.colors.body.categoryTv children {}]
    
    # hard coded default data
    set categoryData {
        {Display
            {Background black} {BackgroundTop black} {BackgroundBot blue2} {Foreground white} {FPS white}
        }
        {Axes
            {Y red} {X green} {Z blue} {Origin cyan} {Labels white}
        }
        {Name
            {H white} {O red} {N blue} {C cyan} {S yellow} {P tan} {Z silver} {LPA green} {LPB green}
        }
        {Type
            {H white} {O red} {N blue} {C cyan} {S yellow} {P tan} {Z silver} {LP green} {DRUD pink}
        }
        {Element
            {X cyan} {Ac ochre} {Ag ochre} {Al ochre} {Am ochre} {Ar ochre} {As ochre}
            {At ochre} {Au ochre} {B ochre} {Ba ochre} {Be ochre} {Bh ochre} {Bi ochre}
            {Bk ochre} {Br ochre} {C cyan} {Ca ochre} {Cd ochre} {Ce ochre} {Cf ochre}
            {Cl ochre} {Cm ochre} {Co ochre} {Cr ochre} {Cs ochre} {Cu ochre} {Db ochre}
            {Ds ochre} {Dy ochre} {Er ochre} {Es ochre} {Eu ochre} {F ochre} {Fe ochre}
            {Fm ochre} {Fr ochre} {Ga ochre} {Gd ochre} {Ge ochre} {H white} {He ochre}
            {Hf ochre} {Hg ochre} {Ho ochre} {Hs ochre} {I ochre} {In ochre} {Ir ochre}
            {K ochre} {Kr ochre} {La ochre} {Li ochre} {Lr ochre} {Lu ochre} {Md ochre}
            {Mg ochre} {Mn ochre} {Mo ochre} {Mt ochre} {N blue} {Na ochre} {Nb ochre}
            {Nd ochre} {Ne ochre} {Ni ochre} {No ochre} {Np ochre} {O red} {Os ochre}
            {P tan} {Pa ochre} {Pb ochre} {Pd ochre} {Pm ochre} {Po ochre} {Pr ochre}
            {Pt ochre} {Pu ochre} {Ra ochre} {Rb ochre} {Re ochre} {Rf ochre} {Rg ochre}
            {Rh ochre} {Rn ochre} {Ru ochre} {S yellow} {Sb ochre} {Sc ochre} {Se ochre}
            {Sg ochre} {Si ochre} {Sm ochre} {Sn ochre} {Sr ochre} {Ta ochre} {Tb ochre}
            {Tc ochre} {Te ochre} {Th ochre} {Ti ochre} {Tl ochre} {Tm ochre} {U ochre}
            {V ochre} {W ochre} {Xe ochre} {Y ochre} {Yb ochre} {Zn silver} {Zr ochre}
        }
        {Resname
            {ALA blue} {ARG white} {ASN tan} {ASP red} {CYS yellow} {GLY white} {GLU pink}
            {GLN orange} {HIS cyan} {ILE green} {LEU pink} {LYS cyan} {MET yellow} {PHE purple}
            {PRO ochre} {SER yellow} {THR mauve} {TRP silver} {TYR green} {VAL tan} {ADE blue}
            {CYT orange} {GUA yellow} {THY purple} {URA green} {TIP cyan} {TIP3 cyan} {WAT cyan}
            {SOL cyan} {H2O cyan} {LYR purple} {ZN silver} {NA yellow} {CL green} {HSE cyan}
            {HSD cyan} {HSP cyan} {CYX yellow}
        }
        {Restype
            {Unassigned cyan} {Solvent yellow} {Nucleic_Acid purple} {Basic blue} {Acidic red} {Polar green}
            {Nonpolar white} {Ion tan}
        }
        {Chain
        }
        {Segname
        }
        {Conformation
        }
        {Molecule
        }
        {Highlight
            {Proback green} {Nucback yellow} {Nonback blue}
        }
        {Structure
            {"Alpha Helix" blue} {3_10_Helix blue} {Pi_Helix red} {Extended_Beta yellow} {Bridge_Beta tan} {Turn cyan}
        }
        {Surface
            {Grasp gray}
        }
        {Labels
            {Atoms green} {Bonds white} {Angles yellow} {Dihedrals cyan} {Springs orange}
        }
        {Stage
            {Even gray} {Odd silver}
        }
    }
    
    # refill the category TV with the default data
    foreach catData $categoryData {
        set category [lindex $catData 0]
        .vmdPrefs.hlf.nb.colors.body.categoryTv insert {} end -id $category -text $category
        for {set i 1} {$i < [llength $catData]} {incr i} {
            .vmdPrefs.hlf.nb.colors.body.categoryTv insert $category end -id $category.[lindex $catData $i 0] -values [list [lindex $catData $i 0] [lindex $catData $i 1]]
        }
    }

    
    # colors
    # clear the tv box
    .vmdPrefs.hlf.nb.colors.body.colorDefTv delete [.vmdPrefs.hlf.nb.colors.body.colorDefTv children {}]
    
    # hard coded default data
    set colorData {
        {0 blue 0.000 0.000 1.000}
        {1 red 1.000 0.000 0.000}
        {2 gray 0.350 0.350 0.350}
        {3 orange 1.000 0.500 0.000}
        {4 yellow 1.000 1.000 0.000}
        {5 tan 0.500 0.500 0.200}
        {6 silver 0.600 0.600 0.600}
        {7 green 0.000 1.000 0.000}
        {8 white 1.000 1.000 1.000}
        {9 pink 1.000 0.600 0.600}
        {10 cyan 0.250 0.750 0.750}
        {11 purple 0.650 0.000 0.650}
        {12 lime 0.500 0.900 0.400}
        {13 mauve 0.900 0.400 0.700}
        {14 ochre 0.500 0.300 0.000}
        {15 iceblue 0.500 0.500 0.750}
        {16 black 0.000 0.000 0.000}
        {17 yellow2 0.880 0.970 0.020}
        {18 yellow3 0.550 0.900 0.040}
        {19 green2 0.000 0.900 0.040}
        {20 green3 0.000 0.900 0.500}
        {21 cyan2 0.000 0.880 1.000}
        {22 cyan3 0.000 0.760 1.000}
        {23 blue2 0.020 0.380 0.670}
        {24 blue3 0.010 0.0400 0.930}
        {25 violet 0.270 0.000 0.980}
        {26 violet2 0.450 0.000 0.900}
        {27 magenta 0.900 0.000 0.900}
        {28 magenta2 1.000 0.000 0.660}
        {29 red2 0.980 0.000 0.230}
        {30 red3 0.810 0.000 0.000}
        {31 orange2 0.890 0.350 0.000}
        {32 orange3 0.960 0.720 0.000}
    }
    
    # refill the colorDef TV with the default data
    foreach colorDat $colorData {
        .vmdPrefs.hlf.nb.colors.body.colorDefTv insert {} end -id [lindex $colorDat 1] -values $colorDat
    }
    
    # reset the color scale
    set ::vmdPrefs::colorsScaleMethod "RGB"
    set ::vmdPrefs::colorsScaleOffset 0.1
    set ::vmdPrefs::colorsSacleMidpoint 0.5
    
    # if running in control mode, push the default values to VMD
    if { $::vmdPrefs::colorsControlMode } { ::vmdPrefs::colorsPushSettingsToVMD }    
}
#=============================
proc ::vmdPrefs::initTclNamedColors {} {
    variable tclNamedColors
    variable tclNamedColorsByRGB
    
    array set tclNamedColors {
        AliceBlue {240 248 255}
        AntiqueWhite {250 235 215}
        AntiqueWhite1 {255 239 219}
        AntiqueWhite2 {238 223 204}
        AntiqueWhite3 {205 192 176}
        AntiqueWhite4 {139 131 120}
        aquamarine1 {127 255 212}
        aquamarine2 {118 238 198}
        aquamarine3 {102 205 170}
        aquamarine4 {69 139 116}
        azure1 {240 255 255}
        azure2 {224 238 238}
        azure3 {193 205 205}
        azure4 {131 139 139}
        beige {245 245 220}
        bisque1 {255 228 196}
        bisque2 {238 213 183}
        bisque3 {205 183 158}
        bisque4 {139 125 107}
        black {0 0 0}
        BlanchedAlmond {255 235 205}
        blue1 {0 0 255}
        blue2 {0 0 238}
        blue3 {0 0 205}
        blue4 {0 0 139}
        BlueViolet {138 43 226}
        brown {165 42 42}
        brown1 {255 64 64}
        brown2 {238 59 59}
        brown3 {205 51 51}
        brown4 {139 35 35}
        burlywood {222 184 135}
        burlywood1 {255 211 155}
        burlywood2 {238 197 145}
        burlywood3 {205 170 125}
        burlywood4 {139 115 85}
        CadetBlue {95 158 160}
        CadetBlue1 {152 245 255}
        CadetBlue2 {142 229 238}
        CadetBlue3 {122 197 205}
        CadetBlue4 {83 134 139}
        chartreuse1 {127 255 0}
        chartreuse2 {118 238 0}
        chartreuse3 {102 205 0}
        chartreuse4 {69 139 0}
        chocolate {210 105 30}
        chocolate1 {255 127 36}
        chocolate2 {238 118 33}
        chocolate3 {205 102 29}
        chocolate4 {139 69 19}
        coral {255 127 80}
        coral1 {255 114 86}
        coral2 {238 106 80}
        coral3 {205 91 69}
        coral4 {139 62 47}
        CornflowerBlue {100 149 237}
        cornsilk1 {255 248 220}
        cornsilk2 {238 232 205}
        cornsilk3 {205 200 177}
        cornsilk4 {139 136 120}
        cyan1 {0 255 255}
        cyan2 {0 238 238}
        cyan3 {0 205 205}
        cyan4 {0 139 139}
        DarkBlue {0 0 139}
        DarkCyan {0 139 139}
        DarkGoldenrod {184 134 11}
        DarkGoldenrod1 {255 185 15}
        DarkGoldenrod2 {238 173 14}
        DarkGoldenrod3 {205 149 12}
        DarkGoldenrod4 {139 101 8}
        DarkGray {169 169 169}
        DarkGreen {0 100 0}
        DarkGrey {169 169 169}
        DarkKhaki {189 183 107}
        DarkMagenta {139 0 139}
        DarkOliveGreen {85 107 47}
        DarkOliveGreen1 {202 255 112}
        DarkOliveGreen2 {188 238 104}
        DarkOliveGreen3 {162 205 90}
        DarkOliveGreen4 {110 139 61}
        DarkOrange {255 140 0}
        DarkOrange1 {255 127 0}
        DarkOrange2 {238 118 0}
        DarkOrange3 {205 102 0}
        DarkOrange4 {139 69 0}
        DarkOrchid {153 50 204}
        DarkOrchid1 {191 62 255}
        DarkOrchid2 {178 58 238}
        DarkOrchid3 {154 50 205}
        DarkOrchid4 {104 34 139}
        DarkRed {139 0 0}
        DarkSalmon {233 150 122}
        DarkSeaGreen {143 188 143}
        DarkSeaGreen1 {193 255 193}
        DarkSeaGreen2 {180 238 180}
        DarkSeaGreen3 {155 205 155}
        DarkSeaGreen4 {105 139 105}
        DarkSlateBlue {72 61 139}
        DarkSlateGray {47 79 79}
        DarkSlateGray1 {151 255 255}
        DarkSlateGray2 {141 238 238}
        DarkSlateGray3 {121 205 205}
        DarkSlateGray4 {82 139 139}
        DarkSlateGrey {47 79 79}
        DarkTurquoise {0 206 209}
        DarkViolet {148 0 211}
        DeepPink1 {255 20 147}
        DeepPink2 {238 18 137}
        DeepPink3 {205 16 118}
        DeepPink4 {139 10 80}
        DeepSkyBlue1 {0 191 255}
        DeepSkyBlue2 {0 178 238}
        DeepSkyBlue3 {0 154 205}
        DeepSkyBlue4 {0 104 139}
        DimGray {105 105 105}
        DimGrey {105 105 105}
        DodgerBlue1 {30 144 255}
        DodgerBlue2 {28 134 238}
        DodgerBlue3 {24 116 205}
        DodgerBlue4 {16 78 139}
        firebrick {178 34 34}
        firebrick1 {255 48 48}
        firebrick2 {238 44 44}
        firebrick3 {205 38 38}
        firebrick4 {139 26 26}
        FloralWhite {255 250 240}
        ForestGreen {34 139 34}
        gainsboro {220 220 220}
        GhostWhite {248 248 255}
        gold1 {255 215 0}
        gold2 {238 201 0}
        gold3 {205 173 0}
        gold4 {139 117 0}
        goldenrod {218 165 32}
        goldenrod1 {255 193 37}
        goldenrod2 {238 180 34}
        goldenrod3 {205 155 29}
        goldenrod4 {139 105 20}
        green1 {0 255 0}
        green2 {0 238 0}
        green3 {0 205 0}
        green4 {0 139 0}
        GreenYellow {173 255 47}
        grey {190 190 190}
        grey1 {3 3 3}
        grey2 {5 5 5}
        grey3 {8 8 8}
        grey4 {10 10 10}
        grey5 {13 13 13}
        grey6 {15 15 15}
        grey7 {18 18 18}
        grey8 {20 20 20}
        grey9 {23 23 23}
        grey10 {26 26 26}
        grey11 {28 28 28}
        grey12 {31 31 31}
        grey13 {33 33 33}
        grey14 {36 36 36}
        grey15 {38 38 38}
        grey16 {41 41 41}
        grey17 {43 43 43}
        grey18 {46 46 46}
        grey19 {48 48 48}
        grey20 {51 51 51}
        grey21 {54 54 54}
        grey22 {56 56 56}
        grey23 {59 59 59}
        grey24 {61 61 61}
        grey25 {64 64 64}
        grey26 {66 66 66}
        grey27 {69 69 69}
        grey28 {71 71 71}
        grey29 {74 74 74}
        grey30 {77 77 77}
        grey31 {79 79 79}
        grey32 {82 82 82}
        grey33 {84 84 84}
        grey34 {87 87 87}
        grey35 {89 89 89}
        grey36 {92 92 92}
        grey37 {94 94 94}
        grey38 {97 97 97}
        grey39 {99 99 99}
        grey40 {102 102 102}
        grey41 {105 105 105}
        grey42 {107 107 107}
        grey43 {110 110 110}
        grey44 {112 112 112}
        grey45 {115 115 115}
        grey46 {117 117 117}
        grey47 {120 120 120}
        grey48 {122 122 122}
        grey49 {125 125 125}
        grey50 {127 127 127}
        grey51 {130 130 130}
        grey52 {133 133 133}
        grey53 {135 135 135}
        grey54 {138 138 138}
        grey55 {140 140 140}
        grey56 {143 143 143}
        grey57 {145 145 145}
        grey58 {148 148 148}
        grey59 {150 150 150}
        grey60 {153 153 153}
        grey61 {156 156 156}
        grey62 {158 158 158}
        grey63 {161 161 161}
        grey64 {163 163 163}
        grey65 {166 166 166}
        grey66 {168 168 168}
        grey67 {171 171 171}
        grey68 {173 173 173}
        grey69 {176 176 176}
        grey70 {179 179 179}
        grey71 {181 181 181}
        grey72 {184 184 184}
        grey73 {186 186 186}
        grey74 {189 189 189}
        grey75 {191 191 191}
        grey76 {194 194 194}
        grey77 {196 196 196}
        grey78 {199 199 199}
        grey79 {201 201 201}
        grey80 {204 204 204}
        grey81 {207 207 207}
        grey82 {209 209 209}
        grey83 {212 212 212}
        grey84 {214 214 214}
        grey85 {217 217 217}
        grey86 {219 219 219}
        grey87 {222 222 222}
        grey88 {224 224 224}
        grey89 {227 227 227}
        grey90 {229 229 229}
        grey91 {232 232 232}
        grey92 {235 235 235}
        grey93 {237 237 237}
        grey94 {240 240 240}
        grey95 {242 242 242}
        grey96 {245 245 245}
        grey97 {247 247 247}
        grey98 {250 250 250}
        grey99 {252 252 252}
        grey100 {255 255 255}
        honeydew1 {240 255 240}
        honeydew2 {224 238 224}
        honeydew3 {193 205 193}
        honeydew4 {131 139 131}
        HotPink1 {255 110 180}
        HotPink2 {238 106 167}
        HotPink3 {205 96 144}
        HotPink4 {139 58 98}
        IndianRed {205 92 92}
        IndianRed1 {255 106 106}
        IndianRed2 {238 99 99}
        IndianRed3 {205 85 85}
        IndianRed4 {139 58 58}
        ivory1 {255 255 240}
        ivory2 {238 238 224}
        ivory3 {205 205 193}
        ivory4 {139 139 131}
        khaki {240 230 140}
        khaki1 {255 246 143}
        khaki2 {238 230 133}
        khaki3 {205 198 115}
        khaki4 {139 134 78}
        lavender {230 230 250}
        LavenderBlush1 {255 240 245}
        LavenderBlush2 {238 224 229}
        LavenderBlush3 {205 193 197}
        LavenderBlush4 {139 131 134}
        LawnGreen {124 252 0}
        LemonChiffon1 {255 250 205}
        LemonChiffon2 {238 233 191}
        LemonChiffon3 {205 201 165}
        LemonChiffon4 {139 137 112}
        LightBlue {173 216 230}
        LightBlue1 {191 239 255}
        LightBlue2 {178 223 238}
        LightBlue3 {154 192 205}
        LightBlue4 {104 131 139}
        LightCoral {240 128 128}
        LightCyan1 {224 255 255}
        LightCyan2 {209 238 238}
        LightCyan3 {180 205 205}
        LightCyan4 {122 139 139}
        LightGoldenrod {238 221 130}
        LightGoldenrod1 {255 236 139}
        LightGoldenrod2 {238 220 130}
        LightGoldenrod3 {205 190 112}
        LightGoldenrod4 {139 129 76}
        LightGoldenrodYellow {250 250 210}
        LightGreen {144 238 144}
        LightGrey {211 211 211}
        LightPink {255 182 193}
        LightPink1 {255 174 185}
        LightPink2 {238 162 173}
        LightPink3 {205 140 149}
        LightPink4 {139 95 101}
        LightSalmon1 {255 160 122}
        LightSalmon2 {238 149 114}
        LightSalmon3 {205 129 98}
        LightSalmon4 {139 87 66}
        LightSeaGreen {32 178 170}
        LightSkyBlue {135 206 250}
        LightSkyBlue1 {176 226 255}
        LightSkyBlue2 {164 211 238}
        LightSkyBlue3 {141 182 205}
        LightSkyBlue4 {96 123 139}
        LightSlateBlue {132 112 255}
        LightSlateGray {119 136 153}
        LightSlateGrey {119 136 153}
        LightSteelBlue {176 196 222}
        LightSteelBlue1 {202 225 255}
        LightSteelBlue2 {188 210 238}
        LightSteelBlue3 {162 181 205}
        LightSteelBlue4 {110 123 139}
        LightYellow1 {255 255 224}
        LightYellow2 {238 238 209}
        LightYellow3 {205 205 180}
        LightYellow4 {139 139 122}
        LimeGreen {50 205 50}
        linen {250 240 230}
        magenta1 {255 0 255}
        magenta2 {238 0 238}
        magenta3 {205 0 205}
        magenta4 {139 0 139}
        maroon {176 48 96}
        maroon1 {255 52 179}
        maroon2 {238 48 167}
        maroon3 {205 41 144}
        maroon4 {139 28 98}
        MediumAquamarine {102 205 170}
        MediumBlue {0 0 205}
        MediumOrchid {186 85 211}
        MediumOrchid1 {224 102 255}
        MediumOrchid2 {209 95 238}
        MediumOrchid3 {180 82 205}
        MediumOrchid4 {122 55 139}
        MediumPurple {147 112 219}
        MediumPurple1 {171 130 255}
        MediumPurple2 {159 121 238}
        MediumPurple3 {137 104 205}
        MediumPurple4 {93 71 139}
        MediumSeaGreen {60 179 113}
        MediumSlateBlue {123 104 238}
        MediumSpringGreen {0 250 154}
        MediumTurquoise {72 209 204}
        MediumVioletRed {199 21 133}
        MidnightBlue {25 25 112}
        MintCream {245 255 250}
        MistyRose1 {255 228 225}
        MistyRose2 {238 213 210}
        MistyRose3 {205 183 181}
        MistyRose4 {139 125 123}
        moccasin {255 228 181}
        NavajoWhite1 {255 222 173}
        NavajoWhite2 {238 207 161}
        NavajoWhite3 {205 179 139}
        NavajoWhite4 {139 121 94}
        navy {0 0 128}
        NavyBlue {0 0 128}
        OldLace {253 245 230}
        OliveDrab {107 142 35}
        OliveDrab1 {192 255 62}
        OliveDrab2 {179 238 58}
        OliveDrab3 {154 205 50}
        OliveDrab4 {105 139 34}
        orange1 {255 165 0}
        orange2 {238 154 0}
        orange3 {205 133 0}
        orange4 {139 90 0}
        OrangeRed1 {255 69 0}
        OrangeRed2 {238 64 0}
        OrangeRed3 {205 55 0}
        OrangeRed4 {139 37 0}
        orchid {218 112 214}
        orchid1 {255 131 250}
        orchid2 {238 122 233}
        orchid3 {205 105 201}
        orchid4 {139 71 137}
        PaleGoldenrod {238 232 170}
        PaleGreen {152 251 152}
        PaleGreen1 {154 255 154}
        PaleGreen2 {144 238 144}
        PaleGreen3 {124 205 124}
        PaleGreen4 {84 139 84}
        PaleTurquoise {175 238 238}
        PaleTurquoise1 {187 255 255}
        PaleTurquoise2 {174 238 238}
        PaleTurquoise3 {150 205 205}
        PaleTurquoise4 {102 139 139}
        PaleVioletRed {219 112 147}
        PaleVioletRed1 {255 130 171}
        PaleVioletRed2 {238 121 159}
        PaleVioletRed3 {205 104 127}
        PaleVioletRed4 {139 71 93}
        PapayaWhip {255 239 213}
        PeachPuff1 {255 218 185}
        PeachPuff2 {238 203 173}
        PeachPuff3 {205 175 149}
        PeachPuff4 {139 119 101}
        peru {205 133 63}
        pink {255 192 203}
        pink1 {255 181 197}
        pink2 {238 169 184}
        pink3 {205 145 158}
        pink4 {139 99 108}
        plum {221 160 221}
        plum1 {255 187 255}
        plum2 {238 174 238}
        plum3 {205 150 205}
        plum4 {139 102 139}
        PowderBlue {176 224 230}
        purple {160 32 240}
        purple1 {155 48 255}
        purple2 {145 44 238}
        purple3 {125 38 205}
        purple4 {85 26 139}
        red1 {255 0 0}
        red2 {238 0 0}
        red3 {205 0 0}
        red4 {139 0 0}
        RosyBrown {188 143 143}
        RosyBrown1 {255 193 193}
        RosyBrown2 {238 180 180}
        RosyBrown3 {205 155 155}
        RosyBrown4 {139 105 105}
        RoyalBlue {65 105 225}
        RoyalBlue1 {72 118 255}
        RoyalBlue2 {67 110 238}
        RoyalBlue3 {58 95 205}
        RoyalBlue4 {39 64 139}
        SaddleBrown {139 69 19}
        salmon {250 128 114}
        salmon1 {255 140 105}
        salmon2 {238 130 98}
        salmon3 {205 112 84}
        salmon4 {139 76 57}
        SandyBrown {244 164 96}
        SeaGreen {46 139 87}
        SeaGreen1 {84 255 159}
        SeaGreen2 {78 238 148}
        SeaGreen3 {67 205 128}
        SeaGreen4 {46 139 87}
        seashell1 {255 245 238}
        seashell2 {238 229 222}
        seashell3 {205 197 191}
        seashell4 {139 134 130}
        sienna {160 82 45}
        sienna1 {255 130 71}
        sienna2 {238 121 66}
        sienna3 {205 104 57}
        sienna4 {139 71 38}
        SkyBlue1 {135 206 255}
        SkyBlue2 {126 192 238}
        SkyBlue3 {108 166 205}
        SkyBlue4 {74 112 139}
        SlateBlue {106 90 205}
        SlateBlue1 {131 111 255}
        SlateBlue2 {122 103 238}
        SlateBlue3 {105 89 205}
        SlateBlue4 {71 60 139}
        SlateGray {112 128 144}
        SlateGray1 {198 226 255}
        SlateGray2 {185 211 238}
        SlateGray3 {159 182 205}
        SlateGray4 {108 123 139}
        SlateGrey {112 128 144}
        snow1 {255 250 250}
        snow2 {238 233 233}
        snow3 {205 201 201}
        snow4 {139 137 137}
        SpringGreen1 {0 255 127}
        SpringGreen2 {0 238 118}
        SpringGreen3 {0 205 102}
        SpringGreen4 {0 139 69}
        SteelBlue {70 130 180}
        SteelBlue1 {99 184 255}
        SteelBlue2 {92 172 238}
        SteelBlue3 {79 148 205}
        SteelBlue4 {54 100 139}
        tan {210 180 140}
        tan1 {255 165 79}
        tan2 {238 154 73}
        tan3 {205 133 63}
        tan4 {139 90 43}
        thistle {216 191 216}
        thistle1 {255 225 255}
        thistle2 {238 210 238}
        thistle3 {205 181 205}
        thistle4 {139 123 139}
        tomato1 {255 99 71}
        tomato2 {238 92 66}
        tomato3 {205 79 57}
        tomato4 {139 54 38}
        turquoise {64 224 208}
        turquoise1 {0 245 255}
        turquoise2 {0 229 238}
        turquoise3 {0 197 205}
        turquoise4 {0 134 139}
        violet {238 130 238}
        VioletRed {208 32 144}
        VioletRed1 {255 62 150}
        VioletRed2 {238 58 140}
        VioletRed3 {205 50 120}
        VioletRed4 {139 34 82}
        wheat {245 222 179}
        wheat1 {255 231 186}
        wheat2 {238 216 174}
        wheat3 {205 186 150}
        wheat4 {139 126 102}
        white {255 255 255}
        WhiteSmoke {245 245 245}
        yellow1 {255 255 0}
        yellow2 {238 238 0}
        yellow3 {205 205 0}
        yellow4 {139 139 0}
        YellowGreen {154 205 50}
    }
    
    foreach ele [array names tclNamedColors] {
        set tclNamedColorsByRGB($tclNamedColors($ele)) $ele
    }
}
#=============================
proc ::vmdPrefs::colorsUpdateColorScaleBox {} {
    # updates the color for each line in the canvas box used to rep the color scale
    
    # make local copies of variables required to describe the color scale
    set method $::vmdPrefs::colorsScaleMethod
    set midpoint $::vmdPrefs::colorsScaleMidpoint
    set offset $::vmdPrefs::colorsScaleOffset
    set lineIDs $::vmdPrefs::colorsLineIDs
    set lineNum [llength $lineIDs]
    
    set dY 255
    set dXLow [expr {    $midpoint  * [llength $lineIDs] }]
    set dXHi  [expr { (1-$midpoint) * [llength $lineIDs] }]
    
    # determine and set the rgb for each line in the canvas box
    for {set i 0} {$i < $lineNum} {incr i} {
    
        
        if { $method eq "BlkW" || $method eq "WBlk" } {
            # binary color scales (Blk->W, W->Blk)
            if { $i < $dXLow } {
                set rgb [expr { ((255/2.0) / $dXLow) *  $i         +     0     }]
            } elseif { $i >= $dXLow } {
                set rgb [expr { ((255/2.0) / $dXHi ) * ($i-$dXLow) + (255/2.0) }]
            } 
            
            # method-specific calculation
            if { $method eq "BlkW" } {
                set rgb [expr { 1 * $rgb +  0 }]
            } else {
                set rgb [expr {-1 * $rgb + 255}]
            }
            
            # apply the offset
            set rgb [expr { int( $rgb + $offset * 255 ) }]
            # convert rgb to r g b (to match standard proc below)
            set r $rgb; set g $rgb; set b $rgb; unset rgb

        } else {
            # ternary color scale
            # low channel
            if { $i < [expr {$midpoint*$lineNum}] } {
                set low    [expr { (-1 * $dY / $dXLow ) *  $i           + 255 }]
            } else {
                set low 0
            }
            
            # middle channel
            if { $i < [expr {$midpoint*$lineNum}] } {
                set med [expr { (     $dY / $dXLow ) *  $i           +  0  }]
            } else {
                set med [expr { (-1 * $dY / $dXHi  ) * ($i - $dXLow) + 255 }]
            }
            
            # high channel
            if { $i > [expr {$midpoint*$lineNum}] } {
                set high   [expr { (     $dY / $dXHi  ) * ($i - $dXLow) +  0  }]
            } else {
                set high 0
            }    
            
            # method specific calculation
            switch -exact $method {
                {RGB} { set r $low; set g $med; set b $high }
                {BGR} { set b $low; set g $med; set r $high }
                {RWB} {
                    set g $med
                    if { $i < $dXLow } {
                        set r [expr {$low+$med}]; set b $med
                    } elseif { $i > $dXLow } {
                        set r $med; set b [expr {$med+$high}]
                    } else {
                        set r [expr {$low+$med}]; set b [expr {$med+$high}]
                    }
                }
                {BWR} {
                    set g $med
                    if { $i < $dXLow } {
                        set b [expr {$low+$med}]; set r $med
                    } elseif { $i > $dXLow } {
                        set b $med; set r [expr {$med+$high}]
                    } else {
                        set b [expr {$low+$med}]; set r [expr {$med+$high}]
                    }
                }
                {RWG} {
                    set b $med
                    if { $i < $dXLow } {
                        set r [expr {$low+$med}]; set g $med
                    } elseif { $i > $dXLow } {
                        set r $med; set g [expr {$med+$high}]
                    } else {
                        set r [expr {$low+$med}]; set g [expr {$med+$high}]
                    }
                }
                {GWR} {
                    set b $med
                    if { $i < $dXLow } {
                        set g [expr {$low+$med}]; set r $med
                    } elseif { $i > $dXLow } {
                        set g $med; set r [expr {$med+$high}]
                    } else {
                        set g [expr {$low+$med}]; set r [expr {$med+$high}]
                    }
                }
                {GWB} {
                    set r $med
                    if { $i < $dXLow } {
                        set g [expr {$low+$med}]; set b $med
                    } elseif { $i > $dXLow } {
                        set g $med; set b [expr {$med+$high}]
                    } else {
                        set g [expr {$low+$med}]; set b [expr {$med+$high}]
                    }
                }
                {BWG} {
                    set r $med
                    if { $i < $dXLow } {
                        set b [expr {$low+$med}]; set g $med
                    } elseif { $i > $dXLow } {
                        set b $med; set g [expr {$med+$high}]
                    } else {
                        set b [expr {$low+$med}]; set g [expr {$med+$high}]
                    }
                }
                {BGryR} {
                    set med [expr {$med/2}]
                    set g $med
                    if { $i < $dXLow } {
                        set b [expr {$low+$med}]; set r $med
                    } elseif { $i > $dXLow } {
                        set b $med; set r [expr {$high+$med}]
                    } else {
                        set b $med; set r $med
                    }
                }
                {RGryB} {
                    set med [expr {$med/2}]
                    set g $med
                    if { $i < $dXLow } {
                        set r [expr {$low+$med}]; set b $med
                    } elseif { $i > $dXLow } {
                        set r $med; set b [expr {$high+$med}]
                    } else {
                        set r $med; set b $med
                    }
                }
            }; #end switch

            # apply the offset
            set r [expr { int( $r + $offset * 255 ) }]
            set g [expr { int( $g + $offset * 255 ) }]
            set b [expr { int( $b + $offset * 255 ) }]

        }; # end if-else
        
        # check offset rgb for legal color values (i.e. 0 <= x <= 255)
        if { $r > 255 } { set r 255 } elseif { $r < 0 } { set r 0 }
        if { $g > 255 } { set g 255 } elseif { $g < 0 } { set g 0 }
        if { $b > 255 } { set b 255 } elseif { $b < 0 } { set b 0 }
        
        # updated the canvas line with color
        .vmdPrefs.hlf.nb.colors.body.colorScale.scaleBox itemconfigure [lindex $lineIDs $i] -fill [format "#%02x%02x%02x" $r $g $b]
    }    
}
#=============================
#
# REPRESENTATIONS TAB RELATED PROCS
#
#=============================
proc ::vmdPrefs::repsQueryRepDefaults {} {
    # queries the default representaiton settings (top box)

    # colorMethod is a special case due to colorID
    set qColorMethod [mol default color]
    if { [llength $qColorMethod] == 2 } {
        set ::vmdPrefs::reps(colorMethod) [lindex $qColorMethod 0]
        set ::vmdPrefs::reps(colorID) [lindex [colorinfo colors] [lindex $qColorMethod 1]]
        .vmdPrefs.hlf.nb.reps.defaults.colorID configure -state normal
    } else {
        set ::vmdPrefs::reps(colorMethod) $qColorMethod
        .vmdPrefs.hlf.nb.reps.defaults.colorID configure -state disabled
    }
    unset qColorMethod

    # all others are straight-forward
    set ::vmdPrefs::reps(matrial) [mol default material]
    set ::vmdPrefs::reps(drawMethod) [mol default style]
    set ::vmdPrefs::reps(selection) [mol default selection]
}
#=============================
proc ::vmdPrefs::repsPushRepDefaults {} {
    # pushes rep settings to VMD

    # color method is special
    if { $::vmdPrefs::reps(colorMethod) eq "ColorID" } {
        mol default color [list $::vmdPrefs::reps(colorMethod) [lsearch [colorinfo colors] $::vmdPrefs::reps(colorID)]]
    } else {
        mol default color $::vmdPrefs::reps(colorMethod)
    }

    # all others are straight-forward
    mol default material $::vmdPrefs::reps(material)
    mol default style $::vmdPrefs::reps(drawMethod)
    mol default selection $::vmdPrefs::reps(selection)
}
#=============================
proc ::vmdPrefs::repsModifyDefinition {} {
    # based on tv selection, copies data over and activates proper adjustment scheme
    
    # check to make sure that we're looking at an element, not the parent
    if { [.vmdPrefs.hlf.nb.reps.definitions.tv parent [.vmdPrefs.hlf.nb.reps.definitions.tv selection]] == {} } {
        # parent selected, turn everything off
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.minus configure -state disabled
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.entry configure -state disabled
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.plus configure -state disabled
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.lbl configure -state disabled
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.scale configure -state disabled
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu configure -state disabled
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.accept configure -state disabled
        
        set ::vmdPrefs::repDefPlusMinus ""
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu.menu delete 0 end
        
        return
    } else {
        # child selected, activate the accept button (other fields will be enabled below)
        .vmdPrefs.hlf.nb.reps.definitions.modFrame.accept configure -state normal
    }
    
    # get the data
    set tvData [.vmdPrefs.hlf.nb.reps.definitions.tv item [.vmdPrefs.hlf.nb.reps.definitions.tv selection] -values]
    # parse the tv data for relevant settings
    set defaultValue [lindex $tvData 1]
    set adjMethod [lindex $tvData 2 0]
    set adjDetails [lrange [lindex $tvData 2] 1 end]
    
    # adjustment method-specific
    switch -exact $adjMethod {
        {plusMinus} {
            # activate the plusMinus, deactivate scale and menu
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.minus configure -state normal
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.entry configure -state normal
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.plus configure -state normal
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.lbl configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.scale configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu.menu delete 0 end
            set ::vmdPrefs::repDefMenu ""
            
            # copy the default value into the entry box
            set ::vmdPrefs::repDefPlusMinus $defaultValue
            
            # set the increment and sig figs
            set ::vmdPrefs::repDefPlusMinusIncr [lindex $adjDetails 0]
            set ::vmdPrefs::repDefPlusMinusSigFigs [lindex $adjDetails 1]
        }
        {scale} {
            # activate the scale, deactivate plusMinus and menu
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.minus configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.entry configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.plus configure -state disabled
            set ::vmdPrefs::repDefPlusMinus ""
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.lbl configure -state normal
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.scale configure -state normal
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu.menu delete 0 end
            set ::vmdPrefs::repDefMenu ""
                        
            # reconfigure the scale bar
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.scale configure \
                -from [lindex $adjDetails 0] \
                -to [lindex $adjDetails 1] \
                -resolution [lindex $adjDetails 2]
            
            # copy the default value into the scale
            set ::vmdPrefs::repDefScale $defaultValue
        }
        {menu} {
            # activate the menu, deactivate plusMinus and scale
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.minus configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.entry configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.plusMinus.plus configure -state disabled
            set ::vmdPrefs::repDefPlusMinus ""
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.lbl configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.scale.scale configure -state disabled
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu configure -state normal
            
            # copy the default value into the menu
            set ::vmdPrefs::repDefMenu $defaultValue
            
            # reset and rebuild the menu
            .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu.menu delete 0 end
            foreach ele $adjDetails {
                .vmdPrefs.hlf.nb.reps.definitions.modFrame.dropDown.menu.menu add command -label "$ele" -command "set ::vmdPrefs::repDefMenu [list $ele]"
            }
        }
    }; # end switch    
}
#=============================
#
# MATERIALS TAB RELATED PROCS
#
#=============================
proc ::vmdPrefs::matUpdateScaleValues { scaleEle value } {
    # updates the tv item with change to scale, in real time

    # check to make sure that something is selected    
    if { [.vmdPrefs.hlf.nb.materials.body.tv selection] eq "" } { return }
    
    # change the appropriate column based on the scale value
    .vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] $scaleEle $value
    
    # if running in control mode, change the value in the DB
    if { $::vmdPrefs::materialsControlMode } { material change [string tolower $scaleEle] [.vmdPrefs.hlf.nb.materials.body.tv set [.vmdPrefs.hlf.nb.materials.body.tv selection] Name] $value }
}
#=============================
proc ::vmdPrefs::materialsGetCurrentVMD {} {
    # queries VMD's current materials settings and rebuilds the tv
    
    # clear the tv
    .vmdPrefs.hlf.nb.materials.body.tv delete [.vmdPrefs.hlf.nb.materials.body.tv children {}]
    
    # note that Diffuse and Specular are switched in the list returned by [materials settings $ele]
    # this will likely lead to much confusion.  a simple solution is to switch the TV column order,
    # however, they are organized in a more logical manner as provided.  I'll just have to remember
    # to switch the values when reading/writing the data

    foreach ele [material list] {
        set matVals [material settings $ele]
        # switch the diffuse/specular (items 1 & 2, respectively)
        set realSpecular [lindex $matVals 1]
        set realDiffuse [lindex $matVals 2]
        lset matVals 1 $realDiffuse
        lset matVals 2 $realSpecular
        unset realSpecular realDiffuse
        
        # build a datalist with conveniently formatted numbers
        set dataList $ele
        foreach val $matVals {
            lappend dataList [format "%0.3f" $val]
        }
        
        # insert the tv entry
        .vmdPrefs.hlf.nb.materials.body.tv insert {} end -id $ele -values $dataList
        unset matVals dataList
    }    
}
#=============================
proc ::vmdPrefs::materialsPushSettingsToVMD {} {
    # pushes the current tv box settings to VMD
    
    # make a list of vmd and tv materials
    set matsVMD [material list]
    set matsTV {}
    foreach item [.vmdPrefs.hlf.nb.materials.body.tv children {}] {
        lappend matsTV [.vmdPrefs.hlf.nb.materials.body.tv set $item Name]
    }
    
    # vmd deletions -- matsVMD that are not in matsTV
    foreach ele $matsVMD {
        if { [lsearch $matsTV $ele] == -1 } { material delete $ele }
    }
    
    # vmd additions -- matsTV that are not in matsVMD
    foreach ele $matsTV {
        if { [lsearch $matsVMD $ele] == -1 } { material add $ele }
    }
    
    # set/change properties
    foreach ele [.vmdPrefs.hlf.nb.materials.body.tv children {}] {
        set dataList [.vmdPrefs.hlf.nb.materials.body.tv item $ele -values]
        
        set i 1
        foreach prop {ambient diffuse specular shininess mirror opacity outline outlinewidth transmode} {
            material change $prop [lindex $dataList 0] [lindex $dataList $i]
            incr i
        }
        unset i
    }    
}
#=============================
proc ::vmdPrefs::materialsGetDefaults {} {
    # reloads VMD defaults for material settings into the tv
    
    # clear the tv
    .vmdPrefs.hlf.nb.materials.body.tv delete [.vmdPrefs.hlf.nb.materials.body.tv children {}]
    set ::vmdPrefs::matName ""
    
    # re-insert the defaults
    # ambient, diffuse, specular, shininess, mirror, opacity, outline, outlinewidth, angle_modulated_transparency
    foreach ele {
        {Opaque         0.000 0.650 0.500 0.534 0.000 1.000 0.000 0.000 0}
        {Transparent    0.000 0.650 0.500 0.534 0.000 0.300 0.000 0.000 0}
        {BrushedMetal   0.080 0.390 0.150 1.000 0.000 0.000 0.000 0.000 0}
        {Diffuse        0.000 0.620 0.000 0.530 0.000 1.000 0.000 0.000 0}
        {Ghost          0.000 0.000 1.000 0.230 0.000 0.100 0.000 0.000 0}
        {Glass1         0.000 0.000 0.650 0.530 0.000 0.150 0.000 0.000 0}
        {Glass2         0.520 0.760 0.220 0.590 0.000 0.680 0.000 0.000 0}
        {Glass3         0.150 0.250 0.750 0.800 0.000 0.500 0.000 0.000 0}
        {Glossy         0.000 0.650 1.000 0.880 0.000 1.000 0.000 0.000 0}
        {HardPlastic    0.000 0.560 0.280 0.690 0.000 1.000 0.000 0.000 0}
        {MetallicPastel 0.000 0.260 0.550 0.190 0.000 1.000 0.000 0.000 0}
        {Steel          0.250 0.000 0.380 0.320 0.000 1.000 0.000 0.000 0}
        {Translucent    0.000 0.700 0.600 0.300 0.000 0.800 0.000 0.000 0}
        {Edgy           0.000 0.660 0.000 0.750 0.000 1.000 0.620 0.940 0}
        {EdgyShiny      0.000 0.660 0.960 0.750 0.000 1.000 0.760 0.940 0}
        {EdgyGlass      0.000 0.660 0.500 0.750 0.000 0.620 0.620 0.940 0}
        {Goodsell       0.520 1.000 0.000 0.000 0.000 1.000 4.000 0.900 0}
        {AOShiny        0.000 0.850 0.200 0.530 0.000 1.000 0.000 0.000 0}
        {AOChalky       0.000 0.850 0.000 0.530 0.000 1.000 0.000 0.000 0}
        {AOEdgy         0.000 0.900 0.200 0.530 0.000 1.000 0.620 0.930 0}
        {BlownGlass     0.040 0.340 1.000 1.000 0.000 0.100 0.000 0.000 1}
        {GlassBubble    0.250 0.340 1.000 1.000 0.000 0.040 0.000 0.000 1}
        {RTChrome       0.000 0.650 0.500 0.530 0.700 1.000 0.000 0.000 0}
    } {
        .vmdPrefs.hlf.nb.materials.body.tv insert {} end -id [lindex $ele 0] -values $ele
    }
}
#=============================
proc ::vmdPrefs::materialsUniqNameCheck { name } {
    # checks to see if name is already present in tv
    
    # build a list of existing names
    set tvNames {}
    foreach ele [.vmdPrefs.hlf.nb.materials.body.tv children {}] {
        lappend tvNames [.vmdPrefs.hlf.nb.materials.body.tv set $ele Name]
    }
    
    # returns 1 if unique, 0 of not
    if { [lsearch $tvNames $name] == -1 } { return 1 } else { return 0 }
}
#=============================
proc ::vmdPrefs::materialsGenerateNewName {} {
    # generates a unique "New" name
    
    set i 0
    set newName "New${i}"
    while { [::vmdPrefs::materialsUniqNameCheck $newName] != 1 } {
        incr i
        set newName "New${i}"
    }
    unset i
    
    return $newName
}
#=============================



