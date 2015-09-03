namespace eval ::VMDlite::System {
 
    # shorthand
    variable system
    variable graphics
    variable analysis
    # selection styles for radiobutton display
    variable selectionChoices
    # selection styles behind the scenes
    variable selList
    # dictionary for selections/styles
    variable repArr
    # boolean: whether a molecule has been loaded
    variable molLoaded
    # radiobutton initial choice for selection
    variable sel
    # modular variable for selection style cycle
    variable repVal
    # choices for molecules to load
    variable molecules

}

proc ::VMDlite::System::init { } {

    variable system
    variable graphics
    variable analysis
    variable selectionChoices {}
    set ::VMDlite::System::selList {
        protein nucleic lipid water "all and not (protein \
or nucleic or lipid or water)"
    }
    set ::VMDlite::System::repArr {
        all lines
        protein off
        nucleic off
        membrane off
        water off
        other off
    }
    set ::VMDlite::System::molLoaded 0
    variable sel
    set ::VMDlite::System::repVal 1
    set ::VMDlite::System::molecules {
    Ubiquitin BtuB DNA RNA Membrane Aquaporin
    }
    set ::VMDlite::System::pdb 0

}

proc ::VMDlite::System::loadAndWiggle { mol } {
    variable repArr
    variable selList
    
    set ::VMDlite::Sandbox::toggleAll on
    set ::VMDlite::Sandbox::toggleProtein off
    set ::VMDlite::Sandbox::toggleNucleic off
    set ::VMDlite::Sandbox::toggleMembrane off
    set ::VMDlite::Sandbox::toggleWater off
    set ::VMDlite::Sandbox::toggleOther off
    
    # kill existing graphics tab
    destroy ${::VMDlite::Sandbox::nb}.graphics
    # kill any existing molecule that for
    # some terrible reason already exists
    mol delete all
    # put our guy in
    mol new ${::VMDlite::System::pdb}
    # acknowledge that we put 'em in
    set ::VMDlite::System::molLoaded 1
    
    # start spinning
    rock y by 0.1
    
    set ::VMDlite::System::selList {
        protein nucleic lipid water "all and not (protein \
or nucleic or lipid or water)"
    }
    
    # add representations for each selection
    for {set i 1} {${i} <= 5} {incr i} {
        mol addrep [molinfo top]
        mol modselect ${i} [molinfo top] \
			[lindex ${::VMDlite::System::selList} [expr ${i} - 1]]
        mol showrep [molinfo top] ${i} off
    }
    # add representation for hydrogen bonds
    mol addrep [molinfo top]
    mol modstyle 6 [molinfo top] Hbonds
    mol modcolor 6 [molinfo top] ColorID 8 ;# white
    # default off
    mol showrep [molinfo top] 6 off
    
    # add representation for highlighting residues
    mol addrep [molinfo top]
    mol modselect 7 [molinfo top] "not all"
    mol modstyle 7 [molinfo top] licorice 0.5 15.0 15.0 
	mol modcolor 7 [molinfo top] ColorID 8 ;# white
    # default off
    mol showrep [molinfo top] 7 off
    
    # add representation for first residue selection
    mol addrep [molinfo top]
    mol modselect 8 [molinfo top] "not all"
    mol modstyle 8 [molinfo top] licorice 0.6 15.0 15.0
    mol modcolor 8 [molinfo top] ColorID 4
    mol showrep [molinfo top] 8 off
    
    # add representation for second residue selection
    mol addrep [molinfo top]
    mol modselect 9 [molinfo top] "not all"
    mol modstyle 9 [molinfo top] licorice 0.6 15.0 15.0
    mol modcolor 9 [molinfo top] ColorID 4
    mol showrep [molinfo top] 9 off
    
    # create new tabs
    set ::VMDlite::Sandbox::firstLoad 1
    if { ${::VMDlite::Sandbox::firstLoad} } {
        ::VMDlite::Sandbox::graphicsTab
        ::VMDlite::Sandbox::analysisTab
        set ::VMDlite::Sandbox::firstLoad 0 ;# Deal with it.
    }
    
    # kill selected residues
    set ::VMDlite::Analysis::selectedResidues { }
    
    # load residues list for new molecule
    ::VMDlite::Analysis::getResidues
    # automatically select graphics tab
    ${::VMDlite::Sandbox::nb} select 1
    
}

proc ::VMDlite::System::setMolInfo { mol } {
# Displays information for molecule
# Conner Herndon June 23, 2014

	destroy ${::VMDlite::Sandbox::system}.lf.welcome \
		{*}${::VMDlite::Sandbox::system}.lf.molInfo \
		${::VMDlite::Sandbox::system}.lf.loadMolecule
	ttk::frame ${::VMDlite::Sandbox::system}.lf.molInfo

	text ${::VMDlite::Sandbox::system}.lf.molInfo.text \
		-width 18 \
		-height 8 \
		-wrap word \
		-yscrollcommand "${::VMDlite::Sandbox::system}.lf.molInfo.yscroll set"
	set molInfoText [eval [list ::VMDlite::System::getMolInfo ${mol}]]
	${::VMDlite::Sandbox::system}.lf.molInfo.text insert end ${molInfoText}
	ttk::scrollbar ${::VMDlite::Sandbox::system}.lf.molInfo.yscroll \
		-command "${::VMDlite::Sandbox::system}.lf.molInfo.text yview"
	
	ttk::button ${::VMDlite::Sandbox::system}.lf.loadMolecule \
		-text "Load ${mol}" \
		-command [list ::VMDlite::System::loadAndWiggle ${mol}]


	grid ${::VMDlite::Sandbox::system}.lf.molInfo \
		-row 0 -column 0 \
		-padx 1 -pady 1
	pack ${::VMDlite::Sandbox::system}.lf.molInfo.text \
		${::VMDlite::Sandbox::system}.lf.molInfo.yscroll \
		-side left \
		-expand true \
		-fill y
	grid ${::VMDlite::Sandbox::system}.lf.loadMolecule \
		-row 1 -column 0 \
		-padx 1 -pady 1
}

proc ::VMDlite::System::getMolInfo { mol } {
	switch ${mol} {
		Ubiquitin	{
			set ::VMDlite::System::pdb ${VMDlite::main::VMDLITEDIR}/sandboxMolecules/1ubq.pdb
			return \
"Ubiquitin is an example of a small soluble protein.  It is used for signaling in eukaryotes."
		}
		BtuB		{
			set ::VMDlite::System::pdb ${VMDlite::main::VMDLITEDIR}/sandboxMolecules/1nqg.pdb
			return \
"BtuB is an example of a beta-barrel membrane protein.  It transports vitamin B12 across the membrane in bacteria."
		}
		DNA			{
			set ::VMDlite::System::pdb ${VMDlite::main::VMDLITEDIR}/sandboxMolecules/bdna.pdb
			return \
"This is an idealized depiction of B-form DNA, the most common form"
		}
		RNA			{
			set ::VMDlite::System::pdb ${VMDlite::main::VMDLITEDIR}/sandboxMolecules/6tna.pdb
			return \
"This is a transfer RNA molecule, which delivers an amino acid to the ribosome during protein synthesis."
		}
		Membrane	{
			set ::VMDlite::System::pdb ${VMDlite::main::VMDLITEDIR}/sandboxMolecules/membrane.pdb
			return \
"Membranes surround cells for use in structure, protection, and regulation of chemicals."
		}
		Aquaporin	{
			set ::VMDlite::System::pdb ${VMDlite::main::VMDLITEDIR}/sandboxMolecules/3zoj.pdb
			return \
"Aquaporin is a typical alpha-helical membrane protein.  It serves as a water channel in a variety of organisms."
		}
		default		{
			set ::VMDlite::System::pdb ${VMDlite::main::VMDLITEDIR}/sandboxMolecules/1ubq.pdb
			return \
"You shouldn't be seeing this text."
		}
	}
}