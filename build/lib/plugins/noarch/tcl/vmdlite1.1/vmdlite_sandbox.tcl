namespace eval ::VMDlite::Sandbox {

    variable w
    variable nb
    variable tabs
    variable system
    variable graphics
    variable analysis
}

proc ::VMDlite::Sandbox::init { } {
    
}

proc ::VMDlite::Sandbox::gui { } {
    variable w
    variable nb
    variable tabs
    variable system
    variable graphics
    variable analysis
    variable molecules
    variable molecule

    # kill original window
    destroy .vmdlite
    
    if { [winfo exists .vmdlite] } {
        wm deiconify ::VMDlite::main::w
        return
    }
    set w [toplevel ".vmdlite"]
    wm title ${::VMDlite::Sandbox::w} "VMD lite Sandbox"
    
    # # Hotel California proc (user may never exit from VMDlite. You are
    # # here forever).
    wm protocol ${w} WM_DELETE_WINDOW {
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
    grid columnconfigure ${::VMDlite::Sandbox::w} 0 \
        -weight 1
    grid rowconfigure ${::VMDlite::Sandbox::w} 0 \
        -weight 1

    # set a default initial geometry
    # note: height will resize as required by gridded components, width does NOT.
    # 800 is a graceful width for all
    wm geometry ${VMDlite::Sandbox::w} 350x175    
    
    # build/grid a high level frame (hlf) just inside the window to contain the notebook
    ttk::frame ${::VMDlite::Sandbox::w}.hlf
    grid ${::VMDlite::Sandbox::w}.hlf \
        -column 0 \
        -row 0 \
        -sticky nsew
    # allow hlf to resize with window
    grid columnconfigure ${::VMDlite::Sandbox::w}.hlf 0 \
        -weight 1
    grid rowconfigure ${::VMDlite::Sandbox::w}.hlf 0 \
        -weight 1
    
    
    
    # build/grid the notebook (nb)
    ttk::notebook ${::VMDlite::Sandbox::w}.hlf.nb
    set nb ${::VMDlite::Sandbox::w}.hlf.nb
    grid ${::VMDlite::Sandbox::nb} \
        -column 0 \
        -row 0 \
        -sticky nsew

    # create notebook tabs
    ttk::frame ${::VMDlite::Sandbox::nb}.system \
    	-width 500 -height 500
    ${::VMDlite::Sandbox::nb} add ${::VMDlite::Sandbox::nb}.system \
    	-text "System"
    grid columnconfigure ${::VMDlite::Sandbox::nb}.system 0 \
       -weight 1
    grid rowconfigure ${::VMDlite::Sandbox::nb}.system 0 \
        -weight 1
    # shorthand tab variables
    set ::VMDlite::Sandbox::system ${::VMDlite::Sandbox::nb}.system


# # # # # # # #
#
# SYSTEM TAB    
#
# # # # # # # #
    
	ttk::frame ${system}.lf \
		-borderwidth 1 \
        -relief sunken
	ttk::frame ${system}.rf \
		-borderwidth 1 \
        -relief sunken
	ttk::frame ${system}.rf.selectMolecule \
        -relief sunken

	# welcome message with selection instructions
	text ${system}.lf.welcome \
		-width 18 \
		-height 8 \
		-wrap word
	${system}.lf.welcome insert end \
"Welcome to the VMD lite sandbox! Please select a molecule\
from the list on the right to get started."
	
	# middle separator bar
	ttk::separator ${system}.separator \
		-orient vertical
	
	# molecule selection listbox
	listbox ${system}.rf.selectMolecule.list \
		-selectmode single \
		-listvariable ::VMDlite::System::molecules \
		-yscrollcommand "${system}.rf.selectMolecule.yscroll set"
	ttk::scrollbar ${system}.rf.selectMolecule.yscroll \
		-command "${system}.rf.selectMolecule.list yview"

	# place these bastards
	grid ${system}.lf \
		-row 0 -column 0 \
        -padx 3 -pady 3
	grid ${system}.rf \
		-row 0 -column 2 \
        -padx 3 -pady 3
	grid ${system}.rf.selectMolecule \
		-row 0 -column 2
	grid ${system}.lf.welcome \
		-row 0 -column 0 \
		-padx 1 -pady 1
	grid ${system}.separator \
		-row 0 -column 1 \
		-sticky ns \
		-padx 10
	pack ${system}.rf.selectMolecule.list ${system}.rf.selectMolecule.yscroll \
		-side left \
		-expand true \
		-fill y

	# when user selects molecule, activates script that gives molecule information
	# and supplies button for loading
	#${system}.rf.selectMolecule.list selection set 0
	focus -force ${system}.rf.selectMolecule.list
	bind ${system}.rf.selectMolecule.list \
		<Button-1> { ::VMDlite::System::setMolInfo [%W get [%W nearest %y]] }

    # resize window
    ::VMDlite::main::resizeTab
}

# # # # # # # # #
#
# GRAPHICS TAB 
#
# # # # # # # # #

proc ::VMDlite::Sandbox::graphicsTab { } {

    destroy {*}${::VMDlite::Sandbox::nb}.graphics \
        ${::VMDlite::Sandbox::nb}.graphics
	
    # create notebook tab
    ttk::frame ${::VMDlite::Sandbox::nb}.graphics \
    	-width 500 -height 500 \
        -relief sunken
    ${::VMDlite::Sandbox::nb} add ${::VMDlite::Sandbox::nb}.graphics \
    	-text "Graphics"
    grid columnconfigure ${::VMDlite::Sandbox::nb}.graphics 0 \
       -weight 1
    grid rowconfigure ${::VMDlite::Sandbox::nb}.graphics 0 \
        -weight 1
    # shorthand tab variables
    set ::VMDlite::Sandbox::graphics ${::VMDlite::Sandbox::nb}.graphics

    ttk::label ${::VMDlite::Sandbox::graphics}.selection \
    	-text "Selection"
	ttk::label ${::VMDlite::Sandbox::graphics}.representation \
		-text "Representations"
	ttk::separator ${::VMDlite::Sandbox::graphics}.vertSeparator1 \
		-orient vertical
	ttk::separator ${::VMDlite::Sandbox::graphics}.vertSeparator2 \
		-orient vertical
	ttk::separator ${::VMDlite::Sandbox::graphics}.horizSeparator \
		-orient horizontal
	ttk::label ${::VMDlite::Sandbox::graphics}.toggle \
		-text "on"

	set row 3
	# make names, buttons, radiobuttons for each selection and rep
	foreach {sel} { All Protein Nucleic Membrane Water Other } {
    
		set col 2
		set toggle${sel} [expr [string equal ${sel} All] ? on : off] ;# I feel so fancy.
		ttk::label ${::VMDlite::Sandbox::graphics}.label${sel} \
			-text ${sel}
		set state [eval [list ::VMDlite::main::selEnabled ${sel}]]
		checkbutton ${::VMDlite::Sandbox::graphics}.toggle${sel} \
			-variable ::VMDlite::Sandbox::toggle${sel} \
			-onvalue on \
			-offvalue off \
			-state ${state} \
			-command [list ::VMDlite::Graphics::toggleSel ${sel}]
		grid ${::VMDlite::Sandbox::graphics}.label${sel} \
			-row ${row} -column 0 \
			-sticky e \
            -padx 2 -pady 2
		grid ${::VMDlite::Sandbox::graphics}.toggle${sel} \
			-row ${row} -column 7 \
            -padx 2 -pady 2
		# initialize radiobutton selections
		set ::VMDlite::Sandbox::radio($sel) "${sel} Lines"
		
		# make radiobutton for each row/column
		foreach {rep} { Lines VDW "New Cartoon" Surface} {
			# if this is the first time
			if {${row} == 3} {
				ttk::label ${::VMDlite::Sandbox::graphics}.label${rep} \
					-text ${rep}
				grid ${::VMDlite::Sandbox::graphics}.label${rep} \
					-row 1 -column ${col} \
                    -padx 2 -pady 2
			}
			# does this selection exist and should it be enabled
			set state [::VMDlite::main::selEnabled ${sel}]

			# radiobuttons for selections / representations
			ttk::radiobutton ${::VMDlite::Sandbox::graphics}.radio${sel}${rep} \
				-variable ::VMDlite::Sandbox::radio(${sel}) \
				-value "${sel} ${rep}" \
				-state ${state} \
				-command ::VMDlite::Graphics::setRep
			grid ${::VMDlite::Sandbox::graphics}.radio${sel}${rep} \
				-row ${row} -column ${col} \
                -padx 2 -pady 2
			incr col
		}
		incr row
	}

	# have toggle box for "All" selection preset
	${::VMDlite::Sandbox::graphics}.toggleAll select
	
	# place widgets
    grid ${::VMDlite::Sandbox::graphics}.selection \
    	-row 1 -column 0 \
        -padx 2 -pady 2
	grid ${::VMDlite::Sandbox::graphics}.representation \
		-row 0 -column 1 \
		-columnspan 4 \
        -padx 2 -pady 2
	grid ${::VMDlite::Sandbox::graphics}.vertSeparator1 \
		-row 1 -column 1 \
		-sticky ns \
		-padx 3 \
		-rowspan 8
	grid ${::VMDlite::Sandbox::graphics}.vertSeparator2 \
		-row 1 -column 6 \
		-sticky ns \
		-padx 10 \
		-rowspan 9\8
	grid ${::VMDlite::Sandbox::graphics}.toggle \
		-row 1 -column 7 \
        -padx 2 -pady 2
	grid ${::VMDlite::Sandbox::graphics}.horizSeparator \
		-row 2 -column 0 \
		-columnspan 8 \
		-sticky ew \
		-pady 2
    
    set ::VMDlite::Sandbox::toggleAll on
    set ::VMDlite::Sandbox::toggleProtein off
    set ::VMDlite::Sandbox::toggleNucleic off
    set ::VMDlite::Sandbox::toggleMembrane off
    set ::VMDlite::Sandbox::toggleWater off
    set ::VMDlite::Sandbox::toggleOther off

    # resize window based on active tab
    bind ${::VMDlite::Sandbox::nb} <<NotebookTabChanged>> { ::VMDlite::main::resizeTab }    
}

# # # # # # # # #
#
# ANALYSIS TAB 
#
# # # # # # # # #

proc ::VMDlite::Sandbox::analysisTab { } {

	::VMDlite::Analysis::getResidues

    destroy {*}${::VMDlite::Sandbox::nb}.analysis \
        ${::VMDlite::Sandbox::nb}.analysis
    
    # create notebook tab
    ttk::frame ${::VMDlite::Sandbox::nb}.analysis \
    	-width 500 -height 500 \
        -relief sunken
    ${::VMDlite::Sandbox::nb} add ${::VMDlite::Sandbox::nb}.analysis \
    	-text "Analysis"
    set ::VMDlite::Sandbox::analysis ${::VMDlite::Sandbox::nb}.analysis

    ttk::frame ${::VMDlite::Sandbox::analysis}.residuesFrame
    ttk::label ${::VMDlite::Sandbox::analysis}.residuesFrame.residues \
        -text "Choose residues to highlight:"
    ttk::combobox ${::VMDlite::Sandbox::analysis}.residuesFrame.residuesCombo \
    	-values ${::VMDlite::Analysis::resList} \
        -state readonly

    bind ${::VMDlite::Sandbox::analysis}.residuesFrame.residuesCombo <<ComboboxSelected>> \
        {::VMDlite::Analysis::addResidue [%W get]}

    grid ${::VMDlite::Sandbox::analysis}.residuesFrame \
    	-row 0 -column 0 \
        -padx 3 -pady 3 \
        -sticky nsew
    grid ${::VMDlite::Sandbox::analysis}.residuesFrame.residues \
        -row 0 -column 0 \
        -columnspan 2 \
        -padx 2 -pady 2
    grid ${::VMDlite::Sandbox::analysis}.residuesFrame.residuesCombo \
    	-row 0 -column 2 \
        -columnspan 2 \
        -padx 2 -pady 2
    
}