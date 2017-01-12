proc ::VMDlite::main::init {} {

    # source the other files because reasons
    foreach file {system graphics analysis sandbox} {
        source [file join ${::VMDlite::main::VMDLITEDIR} vmdlite_${file}.tcl]
    }
    
    # Initialize namespaces
    ::VMDlite::System::init
    ::VMDlite::Graphics::init
    ::VMDlite::Analysis::init
    ::VMDlite::Sandbox::init
    
    set ::VMDlite::main::lessons { }
}

proc ::VMDlite::main::resizeTab { } {
    # change the window size to match the active notebook tab
    
    # need to force gridder to update
    update idletasks

    # resize width as well
    set dimW [winfo reqwidth [${::VMDlite::Sandbox::nb} select]]
    # line below does not resize width, as all tabs are designed with graceful extension of width
    # note +/- for offset can be +- (multimonitor setup), so the expression needs to allow for BOTH symbols;
    # hend "[+-]+"
    regexp {([0-9]+)x[0-9]+[\+\-]+[0-9]+[\+\-]+[0-9]+} [wm geometry .vmdlite] all dimW
    # manually set dimw to 750
    #set dimW 700
    set dimH [winfo reqheight [${::VMDlite::Sandbox::nb} select]]

    wm geometry .vmdlite [format "%ix%i" $dimW [expr $dimH + 65]]
    # note: 44 and 47 take care of additional padding between nb tab and window edges
    
    update idletasks
}

proc ::VMDlite::main::selEnabled { sel } {
	switch ${sel} {
        All        {set sel all }
        Nucleic   {set sel nucleic }
        Protein   {set sel protein }
        Water      {set sel water }
		Membrane	{set sel lipid }
		Other		{set sel "all and not (protein or \
					nucleic or lipid or water)"}
		default		{ }
	}
	if {${::VMDlite::System::molLoaded} == 0 } {
		return disabled
	}
	set sel [atomselect top ${sel}]
	if {[${sel} num] == 0} {
		return disabled
	} else {
		return active
	}
}

proc ::VMDlite::main::lesson { } {
    
    # get available lessons
    ::VMDlite::main::getLessons
    
    # kill original window
    destroy .vmdlite
    
    if {[llength ${::VMDlite::main::lessons}]==0} {
        tk_messageBox \
            -message "There are no lessons currently available."
        destroy .vmdlite
        vmdlite
    } else {
    
    if { [winfo exists .vmdlite] } {
        wm deiconify ::VMDlite::main::w
        return
    }
    set ::VMDlite::main::w [toplevel ".vmdlite"]
    wm title ${::VMDlite::main::w} "VMD lite"
    
    # Hotel California proc (user may never exit from VMDlite. You are
    # here forever).
    wm protocol ${::VMDlite::main::w} WM_DELETE_WINDOW {
        if {[tk_messageBox \
                -message "Quit?" \
                -type yesno] eq "yes"} {
            destroy .vmdlite
            mol delete all
            source [file join ${::VMDlite::main::VMDLITEDIR} vmdlite.tcl]
            vmdlite
        }
    }
    
    # allow .vmdlite to expand with window
    grid columnconfigure ${::VMDlite::main::w} 0 \
        -weight 1
    grid rowconfigure ${::VMDlite::main::w} 0 \
        -weight 1

    # set a default initial geometry
    # note: height will resize as required by gridded components, width does NOT.
    # 800 is a graceful width for all
    wm geometry ${VMDlite::main::w} 350x175    
    
    # build/grid a high level frame (hlf) just inside the window to contain the notebook
    ttk::frame ${::VMDlite::main::w}.hlf
    grid ${::VMDlite::main::w}.hlf \
        -column 0 \
        -row 0 \
        -sticky nsew
    # allow hlf to resize with window
    grid columnconfigure ${::VMDlite::main::w}.hlf 0 \
        -weight 1
    grid rowconfigure ${::VMDlite::main::w}.hlf 0 \
        -weight 1
    
    # variable shorthand
    set ::VMDlite::main::hlf ${::VMDlite::main::w}.hlf
    
    ttk::frame ${::VMDlite::main::hlf}.lf \
		-borderwidth 1 \
        -relief sunken
	ttk::frame ${::VMDlite::main::hlf}.rf \
		-borderwidth 1 \
        -relief sunken
	ttk::frame ${::VMDlite::main::hlf}.rf.selectLesson \
        -relief sunken

	# welcome message with selection instructions
	text ${::VMDlite::main::hlf}.lf.welcome \
		-width 18 \
		-height 8 \
		-wrap word
	${::VMDlite::main::hlf}.lf.welcome insert end \
"Please choose from the lessons listed."
	
	# middle separator bar
	ttk::separator ${::VMDlite::main::hlf}.separator \
		-orient vertical
	
	# lesson selection listbox
	listbox ${::VMDlite::main::hlf}.rf.selectLesson.list \
		-selectmode single \
		-listvariable ::VMDlite::main::lessons \
		-yscrollcommand "${::VMDlite::main::hlf}.rf.selectLesson.yscroll set"
	ttk::scrollbar ${::VMDlite::main::hlf}.rf.selectLesson.yscroll \
		-command "${::VMDlite::main::hlf}.rf.selectLesson.list yview"

	# place these bastards
	grid ${::VMDlite::main::hlf}.lf \
		-row 0 -column 0 \
        -padx 3 -pady 3
	grid ${::VMDlite::main::hlf}.rf \
		-row 0 -column 2 \
        -padx 3 -pady 3
	grid ${::VMDlite::main::hlf}.rf.selectLesson \
		-row 0 -column 2
	grid ${::VMDlite::main::hlf}.lf.welcome \
		-row 0 -column 0 \
		-padx 1 -pady 1
	grid ${::VMDlite::main::hlf}.separator \
		-row 0 -column 1 \
		-sticky ns \
		-padx 10
	pack ${::VMDlite::main::hlf}.rf.selectLesson.list ${::VMDlite::main::hlf}.rf.selectLesson.yscroll \
		-side left \
		-expand true \
		-fill y

	# when user selects lesson, activates script that gives lesson information
	# and supplies button for loading
	#${hlf}.rf.selectLesson.list selection set 0
	focus -force ${::VMDlite::main::hlf}.rf.selectLesson.list
	bind ${::VMDlite::main::hlf}.rf.selectLesson.list \
		<Button-1> { ::VMDlite::main::setLessonInfo [%W get [%W nearest %y]] }
    }
}

proc ::VMDlite::main::getLessons { } {
# Ask user to select directory in which lessons are stored.
	
    set ::VMDlite::main::lessonsDirectory [tk_chooseDirectory \
        -title "Please choose the folder where your lessons are stored." \
        -mustexist true]
    set ::VMDlite::main::lessonsFolders [glob -nocomplain -directory "${::VMDlite::main::lessonsDirectory}" *]
    foreach lesson ${::VMDlite::main::lessonsFolders} {
        set fpName [open "${lesson}/name.txt" r]

        set lessonTEMP [read ${fpName}]
        set lessonTEMP [string trim ${lessonTEMP}]
        lappend ::VMDlite::main::lessons ${lessonTEMP} ;# add this lesson to list of lessons

        unset -nocomplain ${lessonTEMP}
        close ${fpName}
    }
}

proc ::VMDlite::main::setLessonInfo { Lesson } {

    set lesson [string tolower ${Lesson}]
    set lesson [string trim ${lesson}]
    set Lesson [string trim ${Lesson}]
    set fpInformation [open "${::VMDlite::main::lessonsDirectory}/vmdlite_lesson_${lesson}/information.txt" r]
    set ::VMDlite::main::lessonInfo [read ${fpInformation}]
    close ${fpInformation}

	destroy ${::VMDlite::main::hlf}.lf.welcome \
		{*}${::VMDlite::main::hlf}.lf.lessonInfo \
		${::VMDlite::main::hlf}.lf.loadLesson
	ttk::frame ${::VMDlite::main::hlf}.lf.lessonInfo

	text ${::VMDlite::main::hlf}.lf.lessonInfo.text \
		-width 18 \
		-height 8 \
		-wrap word \
		-yscrollcommand "${::VMDlite::main::hlf}.lf.lessonInfo.yscroll set"
	set lessonInfoText ${::VMDlite::main::lessonInfo}
	${::VMDlite::main::hlf}.lf.lessonInfo.text insert end ${lessonInfoText}
	ttk::scrollbar ${::VMDlite::main::hlf}.lf.lessonInfo.yscroll \
		-command "${::VMDlite::main::hlf}.lf.lessonInfo.text yview"
	
	ttk::button ${::VMDlite::main::hlf}.lf.loadLesson \
		-text "Load ${lesson}" \
		-command [list ::VMDlite::main::loadLesson ${Lesson}]

	grid ${::VMDlite::main::hlf}.lf.lessonInfo \
		-row 0 -column 0 \
		-padx 1 -pady 1
	pack ${::VMDlite::main::hlf}.lf.lessonInfo.text \
		${::VMDlite::main::hlf}.lf.lessonInfo.yscroll \
		-side left \
		-expand true \
		-fill y
	grid ${::VMDlite::main::hlf}.lf.loadLesson \
		-row 1 -column 0 \
		-padx 1 -pady 1

}

proc ::VMDlite::main::loadLesson { Lesson } {
	
    namespace eval ::VMDlite::lesson {
    }
	set ::VMDlite::lesson:LessonName ${Lesson}
    set lesson [string tolower ${Lesson}]
    set lesson [string trim ${lesson}]
    set Lesson [string trim ${Lesson}]
    #puts ${::VMDlite::main::lessonsFolders}
	set ::VMDlite::lesson::directory ${::VMDlite::main::lessonsDirectory}/vmdlite_lesson_${lesson}
	#puts "${::VMDlite::main::lessonsDirectory}/vmdlite_lesson_${lesson}/vmdlite_lesson_${lesson}.tcl"
    source [file join ${::VMDlite::main::lessonsDirectory}/vmdlite_lesson_${lesson} vmdlite_lesson_${lesson}.tcl]
	#puts "vmdlite_lesson_${lesson}.tcl"
	source ${::VMDlite::main::lessonsDirectory}/vmdlite_lesson_${lesson}/vmdlite_lesson_${lesson}.tcl
	
    return [eval ::VMDlite::lesson::module1]

}


# ################################ #
# ################################ #
# ################################ #
# ################################ #
# ################################ #
# ################################ #
# ################################ #
# ################################ #
# ################################ #

proc ::VMDlite::lesson::init { } {
# Initializes module

    # kill original window if this is the initial module loading
    destroy .vmdlite
    if { [winfo exists .vmdlite] } {
        wm deiconify ::VMDlite::lesson::w
        return
    }
    # set titles
    set ::VMDlite::lesson::w [toplevel ".vmdlite"]
    wm title ${::VMDlite::lesson::w} "VMDlite"
    
    # Hotel California proc (user may never exit from VMDlite. You are
    # here forever).
    wm protocol ${::VMDlite::lesson::w} WM_DELETE_WINDOW {
    if {[tk_messageBox \
            -message "Quit?" \
            -type yesno] eq "yes"} {
        destroy .vmdlite
        source [file join ${::VMDlite::main::VMDLITEDIR} vmdlite.tcl]
        vmdlite
    }
}
	# initialize current frame variable (you'll see the sneakiness)
	set ::VMDlite::lesson::currentFrame 1
	mol delete all
}

proc ::VMDlite::lesson::refresh { } {
# kill existing text / buttons then create new text box.
# This proc is used at the start of each frame to refresh the
# window so we may lay new widgets.

    # destroy all widgets
    foreach i [winfo children ${::VMDlite::lesson::w}] {
	if {${i}!=${::VMDlite::lesson::w}} {
	    destroy ${i}
	}
    }

    # frame for text box (holds text and scrollbar)
    ttk::frame ${::VMDlite::lesson::w}.textFrame \
	-relief sunken
    # text box
    text ${::VMDlite::lesson::w}.textFrame.textBox \
	-wrap word \
	-font "Helvetica 20" \
	-width 50 \
	-height 10 \
        -yscrollcommand "${::VMDlite::lesson::w}.textFrame.yscroll set"
    # scrollbar
    ttk::scrollbar ${::VMDlite::lesson::w}.textFrame.yscroll \
	-command "${::VMDlite::lesson::w}.textFrame.textBox yview"

    # horizontal line for separation of text box from buttons
    ttk::separator ${::VMDlite::lesson::w}.separator \
     	-orient horizontal

    # pack everything in
    pack ${::VMDlite::lesson::w}.textFrame \
	-expand true \
	-padx 2 -pady 2 \
	-ipadx 2 -ipady 2 \
    -fill both
    pack ${::VMDlite::lesson::w}.textFrame.textBox \
 	 ${::VMDlite::lesson::w}.textFrame.yscroll \
	    -side left \
	    -expand true \
	    -fill both
    pack ${::VMDlite::lesson::w}.separator \
	-fill x \
	-pady 10
    
    # split frame refresh into two procs for better understanding / versatility
    eval ::VMDlite::lesson::progress
}

proc ::VMDlite::lesson::progress { } {
# create progress buttons (previous, exit, next)
# If we are on the first frame, no "previous" button is made.
# Likewise if we are on the final frame, no "next" button is made.

    set previousFrame [expr ${::VMDlite::lesson::currentFrame} - 1]
    set nextFrame [expr ${::VMDlite::lesson::currentFrame} + 1]

    # frame for buttons
    ttk::frame ${::VMDlite::lesson::w}.buttons \
	-relief sunken
    pack ${::VMDlite::lesson::w}.buttons \
	-side bottom \
	-fill x
    # only make "previous" button if this isn't the first frame
    if {${previousFrame}} {
	ttk::button ${::VMDlite::lesson::w}.buttons.previous \
	    -text "Previous" \
	    -command ::VMDlite::lesson::module${previousFrame}
	pack  ${::VMDlite::lesson::w}.buttons.previous \
	    -side left
    }
    # only make "next" button if this isn't the final frame
    if {${nextFrame} <= ${::VMDlite::lesson::totalFrames}} {
	ttk::button ${::VMDlite::lesson::w}.buttons.next \
	    -text "Next" \
	    -command ::VMDlite::lesson::module${nextFrame}
	pack  ${::VMDlite::lesson::w}.buttons.next \
	    -side right
    }
    # always place exit button
    ttk::button ${::VMDlite::lesson::w}.buttons.exit \
	-text "Exit" \
	-command ::VMDlite::lesson::exit
    pack ${::VMDlite::lesson::w}.buttons.exit \
	-side bottom \
	-padx 2 -pady 2 \
	-ipadx 2 -ipady 2
    if {0} {
        ttk::button ${::VMDlite::lesson::w}.buttons.existentialCrisis \
            -text "Existential crisis" ;# the button has no command evaluation upon execution.
        pack ${::VMDlite::lesson::w}.buttons.existentialCrisis
    }
}

proc ::VMDlite::lesson::showPBC { } {
# This proc will show/hide periodic boundary conditions

    switch ${::VMDlite::lesson::pbc} {
	1    {mol showperiodic [molinfo top] 0 yY;\
	       mol showperiodic [molinfo top] 1 yY;\
	       mol showperiodic [molinfo top] 2 yY}
	0   {mol showperiodic [molinfo top] 0 0;\
	       mol showperiodic [molinfo top] 1 0;\
	       mol showperiodic [molinfo top] 2 0}
    }
    # display doesn't update for some reason. This forces it to.
    scale by 1.0001
}

proc ::VMDlite::lesson::updateScale { newValue args } {
	# If this proc isn't here, the code will break. Move along.
}

proc ::VMDlite::lesson::showRep { { repnum -1} {rep $::VMDlite::lesson::representation } } {
# 
# swaps between representations based on radiobuttons

  if { $repnum >= 0 } {
    mol showrep [molinfo top] $repnum on
    switch $rep {
        Lines        {mol modstyle $repnum [molinfo top] Lines 7.0}
        Licorice    {mol modstyle $repnum [molinfo top] Licorice 0.3 30 30}
        VDW          {mol modstyle $repnum [molinfo top] VDW 0.8 30}
        NewCartoon   {mol modstyle $repnum [molinfo top] NewCartoon }
        off          {mol showrep [molinfo top] $repnum off}
    }
  } else {
    switch ${::VMDlite::lesson::representation} {
      Lines        { mol showrep [molinfo top] 0 on;\
                     mol showrep [molinfo top] 1 off;\
                     mol showrep [molinfo top] 2 off}
      Licorice    { mol showrep [molinfo top] 1 on;\
                    mol showrep [molinfo top] 0 off;\
                    mol showrep [molinfo top] 2 off}
      VDW          { mol showrep [molinfo top] 2 on;\
                     mol showrep [molinfo top] 0 off;\
                     mol showrep [molinfo top] 1 off}
    }
  }  

}

proc ::VMDlite::lesson::setTime { time } {
# sets frame to user's choice on slide. When user drags time slider, the simulation
# will move to that point.

    animate goto ${time}
}

proc ::VMDlite::lesson::DCD { input } {
# Allows TclTk to understand the concept of the DCD and have it shake hands
# with the GUI.
	
    if ${input} {
        # convenience
        set top [molinfo top]

        # sets starting state for loop
        set thisFrame [molinfo ${top} get frame]

        # sets upper bound for while loop
        set numFrames [expr [molinfo ${top} get numframes] - 1]

        # breaks the loop when pause button is pressed
        set ::VMDlite::lesson::keepTruckin 1

        # start at beginning if we're at the end
        if {${thisFrame} == ${numFrames}} {
            animate goto 0
            set thisFrame 0
        }
        
        # Ghetto loop to play through DCD. Can't just "animate forward" since
        # the GUI wouldn't update. Instead we'll go frame by frame until the
        # end. I am not a computer scientist. Deal with it.
        while {${thisFrame} < ${numFrames}} {
            
            # move forward
            animate next
            incr thisFrame
            # set time step variable depending on simulation
            set ::VMDlite::lesson::sliderPos ${thisFrame} 

            # update OpenGL
            display update ui
            
            # if the pause button is pressed, break out of this loop
            if {!${::VMDlite::lesson::keepTruckin}} { break }
            
            # stupid Tcl doesn't evaluate tasks before continuing loop otherwise
            update idletasks
            after 50
        }
    } else {
        set ::VMDlite::lesson::keepTruckin 0
        update idletasks
    }
        
}

proc ::VMDlite::lesson::step { choice } {
    # Steps forward/backward in DCD

    animate ${choice}

    update idletasks
    display update ui
    after 50 
    
    
    set thisFrame [molinfo [molinfo top] get frame]
    # Update frame for scale
    set ::VMDlite::lesson::sliderPos ${thisFrame} 
    
}

proc ::VMDlite::lesson::representationsButtons { { repnum -1}  } {

    # radio with buttons for representations
    ttk::frame ${::VMDlite::lesson::w}.radio \
	-relief sunken
    ttk::label ${::VMDlite::lesson::w}.radio.label \
	-text "Representations" 
    ttk::radiobutton ${::VMDlite::lesson::w}.radio.lines \
	-text "Lines" \
        -value "Lines" \
	-variable ::VMDlite::lesson::representation \
	-command "::VMDlite::lesson::showRep $repnum Lines" 
    ttk::radiobutton ${::VMDlite::lesson::w}.radio.licorice \
	-text "Licorice" \
        -value "Licorice" \
	-variable ::VMDlite::lesson::representation \
	-command "::VMDlite::lesson::showRep $repnum Licorice"
    ttk::radiobutton ${::VMDlite::lesson::w}.radio.vdw \
	-text "VDW" \
        -value "VDW" \
	-variable ::VMDlite::lesson::representation \
	-command "::VMDlite::lesson::showRep $repnum VDW"

    # periodic boundary conditions checkbutton
    ttk::label ${::VMDlite::lesson::w}.pbcLabel \
	-text "Periodic boundary conditions"
    ttk::checkbutton ${::VMDlite::lesson::w}.pbc \
	-variable ::VMDlite::lesson::pbc \
	-command ::VMDlite::lesson::showPBC

    # horizontal line for separation of scrollbar from buttons
    ttk::separator ${::VMDlite::lesson::w}.separator2 \
     	-orient horizontal

    pack ${::VMDlite::lesson::w}.radio \
	-side top \
	-fill x \
	-expand true \
	-padx 2 -pady 2 \
	-ipadx 2 -ipady 2

    pack ${::VMDlite::lesson::w}.radio.label \
	${::VMDlite::lesson::w}.radio.lines \
	${::VMDlite::lesson::w}.radio.licorice \
	${::VMDlite::lesson::w}.radio.vdw \
	    -side left

    pack ${::VMDlite::lesson::w}.pbcLabel \
	${::VMDlite::lesson::w}.pbc \
	    -padx 2 -pady 2 \
	    -ipadx 2 -ipady 2
    pack ${::VMDlite::lesson::w}.separator2 \
	-fill x \
	-pady 10 \
	-side bottom \
	-after ${::VMDlite::lesson::w}.radio
    
    # initialize selection buttons
    ${::VMDlite::lesson::w}.pbc invoke
    ${::VMDlite::lesson::w}.radio.lines invoke

}

proc ::VMDlite::lesson::repButtonsAndDCD { DCD { repnum -1} } {
	
    # radio with buttons for representations
    ttk::frame ${::VMDlite::lesson::w}.radio \
        -relief sunken
    ttk::label ${::VMDlite::lesson::w}.radio.label \
        -text "Representations"
    ttk::radiobutton ${::VMDlite::lesson::w}.radio.lines \
        -text "Lines" \
        -value "Lines" \
        -variable ::VMDlite::lesson::representation \
        -command "::VMDlite::lesson::showRep $repnum Lines"
    ttk::radiobutton ${::VMDlite::lesson::w}.radio.licorice \
        -text "Licorice" \
        -value "Licorice" \
        -variable ::VMDlite::lesson::representation \
        -command "::VMDlite::lesson::showRep $repnum Licorice"
    ttk::radiobutton ${::VMDlite::lesson::w}.radio.vdw \
        -text "VDW" \
        -value "VDW" \
        -variable ::VMDlite::lesson::representation \
        -command "::VMDlite::lesson::showRep $repnum VDW"

    
    # frame for time scale in DCD
    ttk::frame ${::VMDlite::lesson::w}.scale \
        -relief sunken
    # actual scale
    ttk::scale ${::VMDlite::lesson::w}.scale.bar \
        -variable ::VMDlite::lesson::${DCD} \
        -from 0 -to 499 \
        -command ::VMDlite::lesson::setTime
    # buttons for manually moving through DCD
    ttk::button ${::VMDlite::lesson::w}.scale.play \
        -text "Play" \
        -command [list ::VMDlite::lesson::DCD 1]
    ttk::button ${::VMDlite::lesson::w}.scale.stop \
        -text "Stop" \
        -command [list ::VMDlite::lesson::DCD 0]
    ttk::button ${::VMDlite::lesson::w}.scale.stepBack \
        -text "Previous frame" \
        -command [list ::VMDlite::lesson::step prev]
    ttk::button ${::VMDlite::lesson::w}.scale.stepForward \
        -text "Next frame" \
        -command [list ::VMDlite::lesson::step next]


    # horizontal line for separation of scrollbar from buttons
    ttk::separator ${::VMDlite::lesson::w}.separator2 \
     	-orient horizontal

    pack ${::VMDlite::lesson::w}.radio \
        -side top \
        -fill x \
        -expand true \
        -padx 2 -pady 2 \
        -ipadx 2 -ipady 2
    
    pack ${::VMDlite::lesson::w}.scale \
        -before ${::VMDlite::lesson::w}.buttons \
        -fill x \
        -padx 2 -pady 2 \
        -ipadx 2 -ipady 2
    pack ${::VMDlite::lesson::w}.scale.bar \
        -padx 2 -pady 2 \
        -ipadx 2 -ipady 2 \
        -fill x
    pack ${::VMDlite::lesson::w}.scale.stepBack \
        -padx 2 -pady 2 \
        -ipadx 2 -ipady 2 \
        -side left
    pack ${::VMDlite::lesson::w}.scale.play \
        -padx 2 -pady 2 \
        -ipadx 2 -ipady 2 \
        -side left
    pack ${::VMDlite::lesson::w}.scale.stop \
        -padx 2 -pady 2 \
        -ipadx 2 -ipady 2 \
        -side left
    pack ${::VMDlite::lesson::w}.scale.stepForward \
        -padx 2 -pady 2 \
        -ipadx 2 -ipady 2 \
        -side left

    pack ${::VMDlite::lesson::w}.radio.label \
	${::VMDlite::lesson::w}.radio.lines \
	${::VMDlite::lesson::w}.radio.licorice \
	${::VMDlite::lesson::w}.radio.vdw \
	    -side left

    pack ${::VMDlite::lesson::w}.separator2 \
        -fill x \
        -pady 10 \
        -side bottom \
        -after ${::VMDlite::lesson::w}.radio
    
    # initialize selection buttons
    ${::VMDlite::lesson::w}.radio.licorice invoke
    trace variable ::VMDlite::lesson::${DCD} w "::VMDlite::lesson::updateScale"

}

proc ::VMDlite::lesson::textBox { text } {
	
    ${::VMDlite::lesson::w}.textFrame.textBox insert end \
"Slide ${::VMDlite::lesson::currentFrame}. "
    ${::VMDlite::lesson::w}.textFrame.textBox insert end \
	${text}

}

proc ::VMDlite::lesson::exit { } {
# returns user to initial screen
    set jumpShip [tk_messageBox \
        -message "Are you sure you want to exit?" \
        -type yesno]
    switch ${jumpShip} {
        yes { destroy .vmdlite; mol delete all; \
            eval vmdlite}
        no  { }
    }
}
