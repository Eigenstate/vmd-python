namespace eval ::VMDlite::Graphics {
}

proc ::VMDlite::Graphics::init { } {
    set ::VMDlite::Graphics::rep "lines"
    set ::VMDlite::Graphics::repVal 0
    set ::VMDlite::Graphics::selection "all"
    set ::VMDlite::Graphics::sel 0
    set ::VMDlite::Graphics::resList { }
}

proc ::VMDlite::Graphics::toggleSel { sel } {
	switch ${sel} {
		All			{ set tempVar ${::VMDlite::Sandbox::toggleAll}; set sel 0 }
		Protein		{ set tempVar ${::VMDlite::Sandbox::toggleProtein}; set sel 1 }
		Nucleic		{ set tempVar ${::VMDlite::Sandbox::toggleNucleic}; set sel 2 }
		Membrane	{ set tempVar ${::VMDlite::Sandbox::toggleMembrane}; set sel 3 }
		Water		{ set tempVar ${::VMDlite::Sandbox::toggleWater}; set sel 4 }
		Other		{ set tempVar ${::VMDlite::Sandbox::toggleOther}; set sel 5 }
	}
	mol showrep [molinfo top] ${sel} ${tempVar}
}

proc ::VMDlite::Graphics::setRep { } {
# Setter for representation on chosen selection

    foreach sel {All Protein Nucleic Membrane Water Other} {
        #global $radio(${sel})
        set state $::VMDlite::Sandbox::radio(${sel})
        set state [split ${state}]
        set sel [lindex ${state} 0]
        set rep [lrange ${state} 1 end]
        switch ${sel} {
            All			{ set sel 0 }
            Protein		{ set sel 1 }
            Nucleic		{ set sel 2 }
            Membrane	{ set sel 3 }
            Water		{ set sel 4 }
            Other		{ set sel 5 }
        }
        switch ${rep} {
            "New Cartoon"	{set rep "newcartoon"}
            "Surface"		{set rep "surf"}
        }
        mol modstyle ${sel} [molinfo top] ${rep}
    }
}

proc ::VMDlite::Graphics::showHydrogen { } {
    # check if there is even a molecule loaded
    if {${::VMDlite::System::molLoaded} == 0} {
        tk_messageBox \
            -message "No molecule loaded!"
    } else {
        # check if we have rep 6 (hydrogens) yet
        if { [catch {info exists [mol repname 0 6]}] } {
            mol addrep [molinfo top]
        } else {
            # kill rep 6
            # mol delrep 6 [molinfo top]
        }
            # view hydrogen bonds
            mol modstyle 6 [molinfo top] Hbonds
            # make them white
            mol modcolor 6 [molinfo top] ColorID 8
    }
}

proc ::VMDlite::Graphics::resListboxMake { } {
# creates listbox for all residues in protein
# Conner Herndon June 22, 2014
	
	# kill preexisting listbox
	destroy ${::VMDlite::Sandbox::graphics}.resListbox
	# update resList variable to contain current residues
	::VMDlite::Graphics::getResidues
	# create frame for listbox list and scrolllbar
	frame ${::VMDlite::Sandbox::graphics}.resListbox
	# make the list portion of listbox
	listbox ${::VMDlite::Sandbox::graphics}.resListbox.residues \
		-selectmode single \
		-listvariable ::VMDlite::Graphics::resList \
		-yscrollcommand "${::VMDlite::Sandbox::graphics}.resListbox.yscroll set"
	# create scrollbar portion of listbox
	ttk::scrollbar ${::VMDlite::Sandbox::graphics}.resListbox.yscroll \
		-command "${::VMDlite::Sandbox::graphics}.resListbox.residues yview"
	grid ${::VMDlite::Sandbox::graphics}.resListbox \
		-row 1 \
		-column 1 \
		-padx 1 \
		-rowspan 7
	pack ${::VMDlite::Sandbox::graphics}.resListbox.residues \
		${::VMDlite::Sandbox::graphics}.resListbox.yscroll \
		-side left \
		-expand true \
		-fill y
	${::VMDlite::Sandbox::graphics}.resListbox.residues selection set 0
	focus -force ${::VMDlite::Sandbox::graphics}.resListbox.residues
	bind ${::VMDlite::Sandbox::graphics}.resListbox.residues \
		<Button-1> { ::VMDlite::Graphics::resListboxSel [%W get [%W nearest %y]] }
}
