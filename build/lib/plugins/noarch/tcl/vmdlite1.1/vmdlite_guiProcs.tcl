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

    namespace eval ::VMDlite::Lesson {
    }
    set lesson [string tolower ${Lesson}]
    set lesson [string trim ${lesson}]
    set Lesson [string trim ${Lesson}]
    puts ${::VMDlite::main::lessonsFolders}
    source [file join "${::VMDlite::main::lessonsDirectory}/vmdlite_lesson_${lesson}/vmdlite_lesson_${lesson}.tcl"]
    return [eval ::VMDlite::Lesson::${Lesson}::module1]
    
    # switch ${lesson} {
        # phospholipid    {
            # source [file join ${::VMDlite::main::VMDLITEDIR}/vmdlite_lesson_phospholipids \
                        # vmdlite_lesson_phospholipids.tcl]
            # return [eval ::VMDlite::Lesson::Phospholipids::module1]
        # }
        # "VMD Tutorial" {
            # source [file join ${::VMDlite::main::VMDLITEDIR}/vmdlite_tutorial_vmd \
                        # vmdlite_tutorial_vmd.tcl]
            # return [eval ::VMDlite::Tutorial::VMD::contents]
        # }
        # default             {
            # tk_messageBox \
                # -message "This lesson is not yet available."
        # }
    # }

}
