# # # # # # # # # # # # # #
#
# VMD lite main sequence  #
#
# # # # # # # # # # # # # #

namespace eval ::VMDlite::main {

    # window
    variable w
    # notebook shorthand
    variable nb
    # tabs in notebook
    variable tabs
    variable system
    variable graphics
    variable analysis

}

proc vmdlite { } {
    return [eval ::VMDlite::main::gui]
}

proc ::VMDlite::main::gui { } {

    # initialize
    ::VMDlite::main::init

    source [file join ${::VMDlite::main::VMDLITEDIR} vmdlite_guiProcs.tcl]

    variable w
    variable nb
    variable tabs
    variable system
    variable graphics
    variable analysis

    # setup the theme depending on what is available
    set themeList [ttk::style theme names]
    if { [lsearch -exact $themeList "aqua"] != -1 } {
        ttk::style theme use aqua
        set placeHolderPadX 18
    } elseif { [lsearch -exact $themeList "clam"] != -1 } {
        ttk::style theme use clam
    } elseif { [lsearch -exact $themeList "classic"] != -1 } {
        ttk::style theme use classic
    } else { ttk::style theme use default }

    if { [winfo exists .vmdlite] } {
        wm deiconify .vmdlite
        return
    }
    set ::VMDlite::main::w [toplevel ".vmdlite"]
    wm title ${w} "VMD lite"
    
    # Hotel California proc (user may never exit from VMDlite. You are
    # here forever).
    # wm protocol ${w} WM_DELETE_WINDOW {
        # if {[tk_messageBox \
                # -message "Quit?" \
                # -type yesno] eq "yes"} {
            # destroy .vmdlite
            # mol delete all
            # source [file join ${::VMDlite::main::VMDLITEDIR} vmdlite.tcl]
            # vmdlite
        # }
    # }
    
    # allow .vmdlite to expand with .
    grid columnconfigure ${w} 0 \
        -weight 1
    grid rowconfigure ${w} 0 \
        -weight 1

    # set a default initial geometry
    # note: height will resize as required by gridded components, width does NOT.
    # 800 is a graceful width for all
    wm geometry ${w} 200x150    

# # # # # # # # #
#
# INTRO WINDOW  #
#
# # # # # # # # #

    ttk::frame ${w}.hlf
    grid ${w}.hlf \
        -column 0 \
        -row 0 \
        -sticky nsew
    # allow hlf to resize with window
    grid columnconfigure ${w}.hlf 0 \
        -weight 1
    grid rowconfigure ${w}.hlf 0 \
        -weight 1

    ttk::label ${w}.hlf.load \
        -text "Welcome to VMD lite!
What would you like to do?"
    # load a lesson
    ttk::button ${w}.hlf.loadLesson \
        -text "Lesson" \
        -command ::VMDlite::main::lesson
    # load molecule
    ttk::button ${w}.hlf.sandbox \
        -text "Sandbox" \
        -command ::VMDlite::Sandbox::gui
    # place buttons
    grid ${w}.hlf.load \
        -row 0 -column 0 \
        -padx 3 -pady 3
    grid ${w}.hlf.loadLesson \
        -row 1 -column 0 \
        -padx 2 -pady 2
    grid ${w}.hlf.sandbox \
        -row 2 -column 0 \
        -padx 2 -pady 2
    
    return ${w}
}