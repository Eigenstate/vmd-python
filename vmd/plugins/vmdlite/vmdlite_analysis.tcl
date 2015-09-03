namespace eval ::VMDlite::Analysis {

}

proc ::VMDlite::Analysis::init { } {

}

proc ::VMDlite::Analysis::getResidues { } {
# gets residues in protein and puts in numbered list
# Conner Herndon June 22, 2014

	set all [atomselect top all] ;# select everything
	set resids [${all} get residue] ;# list of all resIDs
	set resnames [${all} get resname] ;# get names of residues
	set resNum [lindex ${resids} end] ;# number of residues
	set indexVal 0 ;# residue index
	set ::VMDlite::Analysis::resList { }
	set i 0 ;# counter for while loop indices
	# loop through resID list. If we reach a novel residue,
	# we add this combination to $resDict
	while {${indexVal} < ${resNum}} {
		# if the residue index value has not been recorded yet
		if {${indexVal} != [lindex ${resids} ${i}]} {
			set residue [lindex ${resnames} ${i}]
			switch ${residue} {
				ALA		{ set residue Alanine }
				ARG		{ set residue Arginine }
				ASN		{ set residue Asparagine }
				ASP		{ set residue "Aspartic Acid" }
				CYS		{ set residue Cysteine }
				GLN		{ set residue Glutamine }
				GLU		{ set residue "Glutamic Acid" }
				GLY		{ set residue Glycine }
				HIS		{ set residue Histidine }
				ILE		{ set residue Isoleucine }
				LEU		{ set residue Leucine }
				LYS		{ set residue Lysine }
				MET		{ set residue Methionine }
				PHE		{ set residue Phenylalanine }
				PRO		{ set residue Proline }
				SER		{ set residue Serine }
				THR		{ set residue Threonine }
				TRP		{ set residue Tryptophan }
				TYR		{ set residue Tyrosine }
				VAL		{ set residue Valine }
                DA          { set residue Adenine }
                DG          { set residue Guanine }
                DC          { set residue Cytosine }
                DT          { set residue Thymine }
                T           { set residue Thymine }
                A          { set residue Adenine }
                G          { set residue Guanine }
                C          { set residue Cytosine }
                U          { set residue Uracil }
                HOH         { set residue Water }
                TIP3        { set residue Water }
                default     { }
			}
			set newRes "[expr ${indexVal} + 1] ${residue}"
			lappend ::VMDlite::Analysis::resList ${newRes}
			incr indexVal ;# we got the residue at $indexVal, so move on
		}
		incr i ;# move to next index in resID list
	}
}

proc ::VMDlite::Analysis::getAtoms { resid } {
# get all atoms within the selected residue
	
    set resid [lindex ${resid} 0]
	set sel [atomselect top "residue ${resid}"]
    puts "[${sel} get index]"
	set ::VMDlite::Analysis::atomsList [${sel} get index]
    set ::VMDlite::Analysis::atom1 -1
    set ::VMDlite::Analysis::atom2 -1
	
	destroy ${::VMDlite::Sandbox::analysis}.atomsListLblFrame \
        {*}${::VMDlite::Sandbox::analysis}.atomsListLblFrame \
        ${::VMDlite::Sandbox::analysis}.reset
	
    if { [${sel} num] } {
    
        ttk::labelframe ${::VMDlite::Sandbox::analysis}.atomsListLblFrame \
            -text "Atoms"
        ttk::combobox ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.atomCombo1 \
            -values ${::VMDlite::Analysis::atomsList} \
            -state readonly
        ttk::combobox ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.atomCombo2 \
            -values ${::VMDlite::Analysis::atomsList} \
            -state readonly
        ttk::button ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.distance \
            -text "Calculate distance" \
            -command ::VMDlite::Analysis::distance
        
        grid ${::VMDlite::Sandbox::analysis}.atomsListLblFrame \
            -row 1 -column 0 \
            -rowspan 3 \
            -columnspan 2 \
            -padx 3 \
            -pady 3 \
            -sticky nsew
        grid ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.atomCombo1 \
            -row 2 -column 0 \
            -padx 2 -pady 2
        grid ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.atomCombo2 \
            -row 2 -column 1 \
            -padx 2 -pady 2
        grid ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.distance \
            -row 3 -column 0 \
            -padx 2 -pady 2

        bind ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.atomCombo1 <<ComboboxSelected>> \
            { graphics [molinfo top] delete all ;\
            ::VMDlite::Analysis::highlightAtom1 [%W get] }
        bind ${::VMDlite::Sandbox::analysis}.atomsListLblFrame.atomCombo2 <<ComboboxSelected>> \
            { graphics [molinfo top] delete all ;\
            ::VMDlite::Analysis::highlightAtom2 [%W get] }
        
        ::VMDlite::main::resizeTab
    }
}

proc ::VMDlite::Analysis::addResidue { residue } {
# adds selected residue from combobox to storage. Places
# new residue in listbox that displays storage. Updates
# specific residue selection comboboxes that allow user
# to highlight and find distances

    destroy ${::VMDlite::Sandbox::analysis}.reset \
        ${::VMDlite::Sandbox::analysis}.leftFrame \
        ${::VMDlite::Sandbox::analysis}.rightFrame \
        ${::VMDlite::Sandbox::analysis}.residuesFrame
        
    graphics [molinfo top] delete all
    mol showrep [molinfo top] 8 off
    mol showrep [molinfo top] 9 off
    
    lappend ::VMDlite::Analysis::selectedResidues ${residue} ;# add selected residue to list
    set ::VMDlite::Analysis::selectedResidues \
        [lsort -unique ${::VMDlite::Analysis::selectedResidues}] ;# only contain unique residues
        
    # create string for residue highlight selection
    set residueSelectString ""
    set count 0
    foreach item ${::VMDlite::Analysis::selectedResidues} {
        if {!${count}} {
            set residueSelectString "residue [lindex ${item} 0]"
            incr count
        } else {
            append residueSelectString " or residue [lindex ${item} 0]"
        }
    }
    puts ${residueSelectString}
    mol modselect 7 [molinfo top] "${residueSelectString}"
    mol showrep [molinfo top] 7 on
    
    ttk::frame ${::VMDlite::Sandbox::analysis}.residuesFrame
    ttk::label ${::VMDlite::Sandbox::analysis}.residuesFrame.residues \
        -text "Choose residues to highlight:"
    ttk::combobox ${::VMDlite::Sandbox::analysis}.residuesFrame.residuesCombo \
    	-values ${::VMDlite::Analysis::resList} \
        -state readonly

    grid ${::VMDlite::Sandbox::analysis}.residuesFrame \
    	-row 0 -column 0 \
        -columnspan 2 \
        -padx 2 -pady 2 \
        -sticky nsew
    grid ${::VMDlite::Sandbox::analysis}.residuesFrame.residues \
        -row 0 -column 0 \
        -columnspan 2 \
        -padx 2 -pady 2
    grid ${::VMDlite::Sandbox::analysis}.residuesFrame.residuesCombo \
    	-row 0 -column 2 \
        -columnspan 2 \
        -padx 2 -pady 2

    ttk::frame ${::VMDlite::Sandbox::analysis}.leftFrame
    ttk::frame ${::VMDlite::Sandbox::analysis}.rightFrame
    ttk::label ${::VMDlite::Sandbox::analysis}.leftFrame.words \
        -text "Find distance between:"
    ttk::combobox ${::VMDlite::Sandbox::analysis}.leftFrame.firstResidue  \
        -values ${::VMDlite::Analysis::selectedResidues} \
        -state readonly
    ttk::combobox ${::VMDlite::Sandbox::analysis}.leftFrame.secondResidue \
        -values ${::VMDlite::Analysis::selectedResidues} \
        -state readonly
    ttk::label ${::VMDlite::Sandbox::analysis}.leftFrame.words2 \
        -text ""
    
    listbox ${::VMDlite::Sandbox::analysis}.rightFrame.selected \
        -selectmode single \
        -listvariable ::VMDlite::Analysis::selectedResidues \
		-yscrollcommand "${::VMDlite::Sandbox::analysis}.rightFrame.selectedScroll set"
	ttk::scrollbar ${::VMDlite::Sandbox::analysis}.rightFrame.selectedScroll \
		-command "${::VMDlite::Sandbox::analysis}.rightFrame.selected yview"
    
    ttk::button ${::VMDlite::Sandbox::analysis}.reset \
        -text "Reset" \
        -command ::VMDlite::Analysis::reset
    
    grid ${::VMDlite::Sandbox::analysis}.leftFrame \
        -row 1 -column 0 \
        -padx 2 -pady 2
    grid ${::VMDlite::Sandbox::analysis}.rightFrame \
        -row 1 -column 1\
        -padx 2 -pady 2
        
    grid ${::VMDlite::Sandbox::analysis}.leftFrame.words \
        -row 1 -column 0 \
        -columnspan 2\
        -padx 2 -pady 2
    grid ${::VMDlite::Sandbox::analysis}.leftFrame.firstResidue \
        -row 2 -column 0\
        -padx 2 -pady 2
    grid ${::VMDlite::Sandbox::analysis}.leftFrame.secondResidue \
        -row 3 -column 0\
        -padx 2 -pady 2
    grid ${::VMDlite::Sandbox::analysis}.leftFrame.words2 \
        -row 4 -column 0 \
        -columnspan 2\
        -padx 2 -pady 2
    grid ${::VMDlite::Sandbox::analysis}.reset \
        -row 5 -column 0\
        -padx 2 -pady 2
	pack ${::VMDlite::Sandbox::analysis}.rightFrame.selected \
        ${::VMDlite::Sandbox::analysis}.rightFrame.selectedScroll \
		-side left \
		-expand true \
		-fill y

    ::VMDlite::main::resizeTab
    bind ${::VMDlite::Sandbox::analysis}.residuesFrame.residuesCombo <<ComboboxSelected>> \
        {::VMDlite::Analysis::addResidue [%W get]}
    bind ${::VMDlite::Sandbox::analysis}.leftFrame.firstResidue <<ComboboxSelected>> \
        {set tempVar [%W get]; ::VMDlite::Analysis::highlightResid [list ${tempVar} 1]}
    bind ${::VMDlite::Sandbox::analysis}.leftFrame.secondResidue <<ComboboxSelected>> \
        {set tempVar [%W get]; ::VMDlite::Analysis::highlightResid [list ${tempVar} 0]}
}

proc ::VMDlite::Analysis::highlightResid { resid } {
# change selected residue to licorice representation

    set choice [lindex ${resid} 1]
    set resid [lindex ${resid} 0]
    puts ${resid}
    if ${choice} {
        set ::VMDlite::Analysis::residue1 [lindex ${resid} 0]
        mol modselect 8 [molinfo top] "residue ${::VMDlite::Analysis::residue1}"
        mol showrep [molinfo top] 8 on
    } else {
        set ::VMDlite::Analysis::residue2 [lindex ${resid} 0]
        mol modselect 9 [molinfo top] "residue ${::VMDlite::Analysis::residue2}"
        mol showrep [molinfo top] 9 on
    }
    
    # if we have a residue selected for both boxes
    if {[info exists ::VMDlite::Analysis::residue1] &&\
        [info exists ::VMDlite::Analysis::residue2]} {
            set sel [atomselect top "residue ${::VMDlite::Analysis::residue1} \
or residue ${::VMDlite::Analysis::residue2}"]
        ::VMDlite::Analysis::distance
    } elseif {${choice}} {
        set selText "residue [lindex ${resid} 0]"
        set sel [atomselect top ${selText}]
    } elseif {!${choice}} {
        set sel [atomselect top "residue ${::VMDlite::Analysis::residue2}"]
    }

    if { [${sel} num] } {
        set center [measure center ${sel}]
        ${sel} delete
        molinfo top set center [list ${center}]
        translate to 0 0 0
        display update
        rock y by 0.1
    } else {
    tk_messageBox \
        -message "Please select a different residue. This one appears to \
not exist for some terrible reason."
    }
}

proc ::VMDlite::Analysis::highlightAtom1 { atom } {
# highlight first selected atom

    set ::VMDlite::Analysis::atom1 "index ${atom}"
    mol modselect 8 [molinfo top] ${::VMDlite::Analysis::atom1}
    mol showrep [molinfo top] 8 on

}

proc ::VMDlite::Analysis::highlightAtom2 { atom } {
# highlight second selected atom

    set ::VMDlite::Analysis::atom2 "index ${atom}"
    mol modselect 9 [molinfo top] ${::VMDlite::Analysis::atom2}
    mol showrep [molinfo top] 9 on

}

proc ::VMDlite::Analysis::distance { } {

    # kill preexisting label
    destroy ${::VMDlite::Sandbox::analysis}.leftFrame.words2
    
    # calculate the distance between the atoms
    set sel1 [atomselect top "residue ${::VMDlite::Analysis::residue1}"]
    set sel2 [atomselect top "residue ${::VMDlite::Analysis::residue2}"]
    set center1 [measure center ${sel1}]
    set center2 [measure center ${sel2}]
    set residue1x [lindex ${center1} 0]
    set residue1y [lindex ${center1} 1]
    set residue1z [lindex ${center1} 2]
    set residue2x [lindex ${center2} 0]
    set residue2y [lindex ${center2} 1]
    set residue2z [lindex ${center2} 2]
    set xSq [expr [expr ${residue2x} - ${residue1x}] ** 2] 
    set ySq [expr [expr ${residue2y} - ${residue1y}] ** 2] 
    set zSq [expr [expr ${residue2z} - ${residue1z}] ** 2]
    set dist [expr {sqrt(${xSq} + ${ySq} + ${zSq})}]

    graphics [molinfo top] delete all

    # line between residues
    graphics [molinfo top] color 29
    graphics [molinfo top] line [measure center ${sel1}] [measure center ${sel2}] width 8 style dashed

    # display distance
    ttk::label ${::VMDlite::Sandbox::analysis}.leftFrame.words2 \
        -text [format "%.3f Angstroms" ${dist}] 

    grid ${::VMDlite::Sandbox::analysis}.leftFrame.words2 \
        -row 4 -column 0 \
        -columnspan 2\
        -padx 2 -pady 2
}

proc ::VMDlite::Analysis::reset { } {

    set ::VMDlite::Analysis::selectedResidues { }
    set i 7
    while {${i} <= 9} {
        mol showrep [molinfo top] ${i} off
        incr i
    }
    set all [atomselect top all]
    set center [measure center ${all}]
    molinfo top set center [list ${center}]
    translate to 0 0 0
    display update
    rock y by 0.1
    graphics [molinfo top] delete all

    destroy ${::VMDlite::Sandbox::analysis}.reset \
              ${::VMDlite::Sandbox::analysis}.atomsListLblFrame \
            {*}${::VMDlite::Sandbox::analysis}.atomsListLblFrame
            
    destroy ${::VMDlite::Sandbox::analysis}.reset \
        ${::VMDlite::Sandbox::analysis}.leftFrame \
        ${::VMDlite::Sandbox::analysis}.rightFrame
            
    ::VMDlite::main::resizeTab
}





