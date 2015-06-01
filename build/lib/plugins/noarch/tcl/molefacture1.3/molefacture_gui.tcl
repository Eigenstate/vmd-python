##
## $Id: molefacture_gui.tcl,v 1.67 2012/04/03 15:15:03 johanstr Exp $
##

#####################################################
# Add/Edit internal coordinate.                     #
#####################################################
##
## $Id: molefacture_gui.tcl,v 1.67 2012/04/03 15:15:03 johanstr Exp $
##

proc ::Molefacture::molefacture_gui { {sel ""} } {
#puts "DEBUG A"
   variable atomlist 
   if {$sel == ""} {
     variable molidorig -1
   } elseif {[$sel num] == 0} {
      tk_messageBox -icon error -type ok -title Error \
      -message "You entered a selection containing no atoms. If you want to create a new molecule, invoke molefacture with no selection. Otherwise, please make a selection containing at least one atom."
      return
   } else {
     variable molidorig [$sel molid]
     set cursel  [atomselect $molidorig "index [$sel list]"]
     variable origsel [atomselect $molidorig "index [$sel list]"]
   }
   variable w
   variable selectcolor
   variable showvale
   variable showellp
   variable atomeditformtags
   variable atomlistformat
   variable taglist
   variable templist
   variable atomlistformat
   variable bondtaglist
   variable bondlistformat
   variable anglelistformat
   variable angletaglist
   variable FEPlist
   variable FEPlistformat
   variable FEPparentmol
   variable FEPdelseg
   variable FEPdelres
   variable FEPreplaceflag
   variable FEPoutprefix

   # values for atom editing
   variable editatom_name 
   variable editatom_type 
   variable editatom_element 
   variable editatom_charge 
   variable editatom_index 

   variable editmol_resname
   variable editmol_segname
   variable editmol_chain

   # Variable for OPLS definitions file
   variable OPLSatomdef "[file join $::env(MOLEFACTUREDIR) lib ATOMTYPE_OPLS.DEF]" 

   # Variables for testing atom typing
   variable refdir
   variable testdir
   variable at_test
   set refdir $::env(HOME)
   set testdir $::env(HOME)
   set at_test ""

   trace add variable editatom_element write ::Molefacture::update_edit_atom_elem_tr

   set taglist "Index Name Type Elem Open  FormCharge OxState Charge"
   set templist [edit_update_list $taglist]
   set taglist [lindex $templist 0]
   set atomlistformat [lindex $templist 1]

   set bondtaglist "Atom1 Atom2 Order"
   set templist [edit_update_list $bondtaglist]
   set bondtaglist [lindex $templist 0]
   set bondlistformat [lindex $templist 1]

   set angletaglist "Atom1 Atom2 Atom3"
   set anglelistformat "%5i %5i %5i"




   set FEPlist "Index Name Type Elem  FEPindex"
   set templist [edit_update_list $FEPlist]
   set FEPlist [lindex $templist 0]
   set FEPlistformat [lindex $templist 1]

   set FEPparentmol ""
   set FEPdelseg ""
   set FEPdelres ""
   set FEPreplaceflag 0
   set FEPoutprefix ""

   #Put a trace on the display options
   trace add variable showvale write ::Molefacture::draw_openvalence_tr
   trace add variable showellp write ::Molefacture::draw_openvalence_tr

   if {$sel == ""} {
     new_mol
   } else { 
     reload_selection
   }

   mol selection      "occupancy > 0.4"
   mol representation "Bonds 0.1"
   mol color          Name
   mol modrep 0 top
   mol representation "VDW 0.1"
   mol addrep top
   display resetview
#puts "DEBUG C"
   variable bondlist [bondlist]
   variable anglelist [anglelist]

   variable atomaddlist {}
   if {$sel != ""} {
   foreach index [$sel list] {
#      lappend atomaddlist $index {}
   }
   }

   assign_elements

   init_oxidation

   update_openvalence

   update_openvalence_FEP

   set_pickmode_atomedit

   # If already initialized, just turn on
   if { [winfo exists .molefac] } {
      wm deiconify .molefac
      focus .molefac
      return
   }


   set w [toplevel ".molefac"]
   wm title $w "Molefacture - Molecule Builder"
   wm resizable $w 0 0

   #Add a menubar
   frame $w.menubar -relief raised -bd 2
   menubutton $w.menubar.file -text "File" -underline 0 \
   -menu $w.menubar.file.menu
   menubutton $w.menubar.build -text "Build" -underline 0 \
   -menu $w.menubar.build.menu
   menubutton $w.menubar.set -text "Settings" -underline 0 \
   -menu $w.menubar.set.menu
   menubutton $w.menubar.help -text "Help" -underline 0 \
   -menu $w.menubar.help.menu
   menubutton $w.menubar.simulations -text "Simulations" -underline 0 \
   -menu $w.menubar.simulations.menu
   # XXX - set menubutton width to avoid truncation in OS X
   $w.menubar.file config -width 8 
   $w.menubar.build config -width 9
   $w.menubar.set config -width 12
   $w.menubar.simulations config -width 15
   $w.menubar.help config -width 9 

   ## File menu
   menu $w.menubar.file.menu -tearoff no
   $w.menubar.file.menu add command -label "New" -command ::Molefacture::new_mol_gui
   $w.menubar.file.menu add command -label "Save" -command ::Molefacture::export_molecule_gui 
   $w.menubar.file.menu add command -label "Apply changes to parent" -command ::Molefacture::apply_changes_to_parent_mol
   $w.menubar.file.menu add command -label "Write top file" -command ::Molefacture::write_topology_gui
   $w.menubar.file.menu add command -label "Write psf and pdb files" -command ::Molefacture::write_namdfiles_gui
#   $w.menubar.file.menu add command -label "Undo unsaved changes" -command ::Molefacture::undo_changes
   $w.menubar.file.menu add command -label "Quit" -command ::Molefacture::done

   ## Build menu
   menu $w.menubar.build.menu -tearoff no
   menu $w.menubar.build.abtype -title "Assign atom & bond types" -tearoff no
   $w.menubar.build.abtype add command -label "IDATM" -command ::Molefacture::run_idatm
   $w.menubar.build.abtype add command -label "Antechamber for GAFF" -command ::Molefacture::run_ante
   #$w.menubar.build.abtype add command -label "Antechamber for OPLS" -command ::Molefacture::run_ante_opls
   $w.menubar.build.abtype add command -label "Antechamber gui" -command ::Molefacture::ante_gui
   $w.menubar.build.menu add cascade -label "Assign atom & bond types" -menu $w.menubar.build.abtype
   menu $w.menubar.build.charges -title "Assign charges" -tearoff no
   $w.menubar.build.charges add command -label "Mulliken" -command ::Molefacture::run_sqm_mullik
   $w.menubar.build.charges add command -label "CM1A*1.14" -command ::Molefacture::run_sqm_cm1a
   $w.menubar.build.charges add command -label "AM1-BCC" -command ::Molefacture::runante_am1bcc
   
   $w.menubar.build.menu add cascade -label "Assign charges" -menu $w.menubar.build.charges
   $w.menubar.build.menu add command -label "Geometry optimization with SQM" -command ::Molefacture::am1_geo_opt
   $w.menubar.build.menu add command -label "Add all hydrogens" -command ::Molefacture::add_all_hydrogen
   menu $w.menubar.build.addfrag -title "Replace hydrogen with fragment" -tearoff yes
   $w.menubar.build.addfrag add command -label "Add custom..." -command "::Molefacture::add_custom_frags"
   $w.menubar.build.addfrag add command -label "Reset menu" -command "::Molefacture::reset_frags_menu"
   $w.menubar.build.addfrag add separator
   read_fragment_file [file join $::env(MOLEFACTUREDIR) lib fragments frag.mdb]
   fill_fragment_menu
   menu $w.menubar.build.basefrag -tearoff no 
   $w.menubar.build.basefrag add command -label "Add custom..." -command "::Molefacture::add_custom_basefrags"
   $w.menubar.build.basefrag add command -label "Reset menu" -command "::Molefacture::reset_basefrags_menu"
   $w.menubar.build.basefrag add separator
   read_basefrag_file [file join $::env(MOLEFACTUREDIR) lib basemol basefrag.mdb]
   fill_basefrag_menu
   $w.menubar.build.menu add cascade -label "Replace hydrogen with fragment" -menu $w.menubar.build.addfrag
   $w.menubar.build.menu add cascade -label "New molecule from fragment" -menu $w.menubar.build.basefrag
   $w.menubar.build.menu add command -label "Protein Builder" -command ::Molefacture::prot_builder_gui
#   $w.menubar.build.menu add command -label "Set Unique Atom Names" -command ::Molefacture::unique_atomnames
#   $w.menubar.build.menu add command -label "Nucleic Acid Builder" -command ::Molefacture::nuc_builder_gui

   ## Settings menu
   menu $w.menubar.set.menu -tearoff no
   $w.menubar.set.menu add checkbutton -label "Display valences" -variable [namespace current]::showvale
   $w.menubar.set.menu add checkbutton -label "Display electrons" -variable [namespace current]::showellp
   $w.menubar.set.menu add command -label "Antechamber Settings" -command ::Molefacture::antechamber_setting_gui
   $w.menubar.set.menu add command -label "SQM Settings" -command ::Molefacture::sqm_settings_gui 
   ## Help menu
   menu $w.menubar.help.menu -tearoff no
   $w.menubar.help.menu add command -label "About" \
    -command {tk_messageBox -type ok -title "About Molefacture" \
            -message "Molecule editing tool"}
   $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/molefacture"
######ADDED######
   ## Simulations menu
   menu $w.menubar.simulations.menu -tearoff no
   $w.menubar.simulations.menu add command -label "FEP" -command ::Molefacture::fep_simulations_gui
   #   $w.menubar.simulations.menu add command -label "QM/MM" -command ........
   pack $w.menubar.file $w.menubar.build $w.menubar.set $w.menubar.simulations -side left 
#   pack $w.menubar.file $w.menubar.build $w.menubar.set -side left 
######END######


   pack $w.menubar.file $w.menubar.build $w.menubar.set -side left 
   pack $w.menubar.help -side right
   grid $w.menubar -sticky ew -columnspan 2 -row 0 -column 0 -padx 0


   ############## frame for atom list and editing #################

   # Initialize atomlist formatting

#puts "Taglist: $taglist"
#puts "ALFormat: $atomlistformat"

   labelframe $w.val -bd 2 -relief ridge -text "Atoms" -padx 2 -pady 2
   label $w.val.label -wraplength 10c -justify left -text "-Pick atoms in VMD window or from the list.-"  -fg green3

   frame $w.val.list
   label $w.val.list.format -font {tkFixed 9} -textvariable ::Molefacture::taglist -relief flat -bd 2 -justify left;
#   label $w.val.list.format -font {tkFixed 9} -textvariable "test test" -relief flat -bd 2 -justify left;
   scrollbar $w.val.list.scroll -command "$w.val.list.list yview" -takefocus 0
   listbox $w.val.list.list -activestyle dotbox -yscroll "$w.val.list.scroll set" -font {tkFixed 9} \
      -width 55 -height 5 -setgrid 1 -selectmode extended -selectbackground $selectcolor -listvariable ::Molefacture::atomlist
   pack $w.val.list.format -side top -anchor w
   pack $w.val.list.list -side left -fill both -expand 1
   pack $w.val.list.scroll -side left -fill y
#   pack $w.val.list.list $w.val.list.scroll -side left -fill y -expand 1
#   pack $w.val.list.format -pady 2m 
   pack $w.val.label -pady 2m
   pack $w.val.list -padx 1m -pady 1m -fill y -expand 1

   #Editing tools
#   labelframe $w.atomedit -bd 2 -relief ridge -text "Edit Atoms" -padx 1m -pady 1m
   frame $w.val.f1
   button $w.val.f1.hyd    -text "Add hydrogen to selected atom" -command ::Molefacture::add_hydrogen_gui
   button $w.val.f1.del -text "Delete Selected Atom" -command ::Molefacture::del_atom_gui
   pack $w.val.f1.hyd $w.val.f1.del -side left -fill x -expand 1
   frame $w.val.f2
   button $w.val.f2.invert -text "Invert chirality" -command ::Molefacture::invert_gui
   button $w.val.f2.planar -text "Force planar" -command ::Molefacture::set_planar_gui
   button $w.val.f2.tetra -text "Force tetrahedral" -command ::Molefacture::set_tetra_gui
   frame $w.val.f3
   button $w.val.f3.raiseox -text "Raise oxidation state" -command ::Molefacture::raise_ox_gui
   button $w.val.f3.lowerox -text "Lower oxidation state" -command ::Molefacture::lower_ox_gui
   button $w.val.edit -text "Edit selected atom" -command ::Molefacture::edit_atom_gui
   pack $w.val.f2.invert $w.val.f2.planar $w.val.f2.tetra -side left -fill x -expand 1
   pack $w.val.f3.raiseox $w.val.f3.lowerox -side left -fill x -expand 1

   pack $w.val.f1  -side top -fill x
   pack $w.val.f2 -side top -fill x
   pack $w.val.f3 -side top -fill x
   pack $w.val.edit -side top -fill x

   labelframe $w.mol -bd 2 -relief ridge -text "Molecule" -padx 2 -pady 2
   button $w.mol.residue -text "Edit segname/resname/chain" -command ::Molefacture::edit_mol_gui
   pack $w.mol.residue -side top -fill x   
   frame $w.mol.charge
   
   label $w.mol.charge.label -text "Total charge: "
   label $w.mol.charge.total -textvariable ::Molefacture::totalcharge -width 5 
   button $w.mol.charge.edit -text "Modify charge" -command ::Molefacture::edit_charge_gui
   pack  $w.mol.charge.label  $w.mol.charge.total $w.mol.charge.edit -side left -fill x
   pack  $w.mol.charge -side top -fill x

   ############## frame for bond list and editing #################
   labelframe $w.bonds -bd 2 -relief ridge -text "Bonds" -padx 1m -pady 1m
   frame $w.bonds.list
   label $w.bonds.list.format -font {tkFixed 9} -textvariable ::Molefacture::bondtaglist -relief flat -bd 2 -justify left;
   scrollbar $w.bonds.list.scroll -command "$w.bonds.list.list yview" -takefocus 0
   listbox $w.bonds.list.list -activestyle dotbox -yscroll "$w.bonds.list.scroll set" -font {tkFixed 9} \
      -width 20 -height 5 -setgrid 1 -selectmode browse -selectbackground $selectcolor -listvariable ::Molefacture::bondlist
   pack $w.bonds.list.format -side top -anchor w
   pack $w.bonds.list.list $w.bonds.list.scroll -side left -fill y -expand 1
   #labelframe $w.bondedit -bd 2 -relief ridge -text "Edit Bonds" -padx 1m -pady 1m
   frame $w.bonds.list.f1
   button $w.bonds.list.f1.raise -text "Raise bond order" -command ::Molefacture::raise_bondorder_gui
   button $w.bonds.list.f1.lower -text "Lower bond order" -command ::Molefacture::lower_bondorder_gui
   label $w.bonds.list.distlabel -text "Adjust bond length:" 
   spinbox $w.bonds.list.distspinb -from 0 -to 10 -increment 0.05 -width 5 \
      -textvariable ::Molefacture::bondlength -command {::Molefacture::adjust_bondlength}
   pack $w.bonds.list.f1.raise $w.bonds.list.f1.lower $w.bonds.list.distlabel $w.bonds.list.distspinb -side top
   pack $w.bonds.list.f1 -side left
   pack $w.bonds.list

   frame $w.bonds.f2
   ############## frame for distance spinbox #################

   ############## frame for dihedral scale #################
   frame $w.bonds.f2.angle  -padx 2m -pady 2m
   label $w.bonds.f2.angle.label -justify left -text "Rotate bond dihedral:"
   scale $w.bonds.f2.angle.scale -orient horizontal -length 200 -from -180 -to 180.001 \
      -command {::Molefacture::rotate_bond}  -tickinterval 90 -resolution 0.05
   pack $w.bonds.f2.angle.label $w.bonds.f2.angle.scale -side left -expand yes -anchor w

   frame $w.bonds.f2.chooser
   label $w.bonds.f2.chooser.label -justify left -text "Move: "
   radiobutton $w.bonds.f2.chooser.b1 -text "Group1" -width 6 -value "Atom1" -variable [namespace current]::dihedmoveatom
   radiobutton $w.bonds.f2.chooser.b2 -text "Group2" -width 6 -value "Atom2" -variable [namespace current]::dihedmoveatom

   pack $w.bonds.f2.chooser.label $w.bonds.f2.chooser.b1 $w.bonds.f2.chooser.b2 -side left

   #$w.bonds.angle.scale set 0
   pack $w.bonds.f2.angle -side top
   pack $w.bonds.f2.chooser -side top
   pack $w.bonds.f2 -fill x

   bind $w.bonds.f2.angle.scale <ButtonRelease-1> {
      ::Molefacture::draw_openvalence      
   }



   ############## frame for angle list and editing #################
   labelframe $w.angles -bd 2 -relief ridge -text "Angles" -padx 1m -pady 1m
   frame $w.angles.list
   label $w.angles.list.format -font {tkFixed 9} -textvariable ::Molefacture::angletaglist -relief flat -bd 2 -justify left
   scrollbar $w.angles.list.scroll -command "$w.angles.list.list yview" -takefocus 0
   listbox $w.angles.list.list -activestyle dotbox -yscroll "$w.angles.list.scroll set" -font {tkFixed 9} \
      -width 20 -height 5 -setgrid 1 -selectmode browse -selectbackground $selectcolor -listvariable ::Molefacture::anglelist
   pack $w.angles.list.format -side top -anchor w
   pack $w.angles.list.list $w.angles.list.scroll -side left -fill y -expand 1
   pack $w.angles.list -side left
   ############## frame for angle scale #################
   frame $w.angles.realangle  -padx 1m -pady 1m
   label $w.angles.realangle.label -justify left -text "Adjust angle:"
   scale $w.angles.realangle.scale -orient horizontal -length 180 -from 0 -to 180 -command {::Molefacture::resize_angle}  -tickinterval 30

   frame $w.angles.realangle.chooser
   label $w.angles.realangle.chooser.label -justify left -text "Move: "
   radiobutton $w.angles.realangle.chooser.b1 -text "Group1" -width 6 -value "Atom1" -variable [namespace current]::anglemoveatom
   radiobutton $w.angles.realangle.chooser.b2 -text "Group2" -width 6 -value "Atom2" -variable [namespace current]::anglemoveatom
   radiobutton $w.angles.realangle.chooser.b3 -text "Both" -width 4 -value "Both" -variable [namespace current]::anglemoveatom
   pack $w.angles.realangle.chooser.label 
   pack $w.angles.realangle.chooser.b1 $w.angles.realangle.chooser.b2 $w.angles.realangle.chooser.b3 -side left
   pack $w.angles.realangle.chooser -side bottom -expand yes -anchor s
   pack $w.angles.realangle.label $w.angles.realangle.scale -side top -expand yes -anchor w
   #$w.bonds.angle.scale set 0
   pack $w.angles.realangle -side top

   #Frame for building options
#   labelframe $w.builder -bd 2 -relief ridge -text "Build" -padx 1m -pady 1m
#   button $w.builder.allhyd -text "Add all hydrogens to molecule" -command ::Molefacture::add_all_hydrogen
#   pack $w.builder.allhyd -side left

   #Frame for saving options
#   labelframe $w.molecule -bd 2 -relief ridge -text "Molecule" -padx 1m -pady 1m
#   button $w.molecule.save -text "Save" -command ::Molefacture::export_molecule_gui
#   button $w.molecule.quit -text "Done" -command ::Molefacture::done
#   button $w.molecule.undo -text "Undo unsaved changes" -command ::Molefacture::undo_changes
#   pack $w.molecule.save -side left -anchor w
#   pack $w.molecule.quit -side left -anchor w -fill x -expand 1
#   pack $w.molecule.undo -side left

#   button $w.edit.fix -text "Apply changes" -command ::Molefacture::fix_changes

#   pack $w.val $w.bonds $w.angles $w.charge -padx 1m -pady 1m  
   grid $w.val -padx 1 -columnspan 1 -column 0 -row 1 -rowspan 1 -sticky ew
   grid $w.mol -padx 1 -columnspan 1 -column 0 -row 2 -rowspan 1 -sticky ew
   grid $w.bonds -padx 1 -column 1 -row 1 -sticky ew
   grid $w.angles -padx 1 -column 1 -row 2 -sticky ew

   # Enable manual editing of the spinbox entry value
   bind $w.bonds.list.distspinb <Return> {
      ::Molefacture::adjust_bondlength
   }

   # This will be executed when a bond is selected:   
   bind $w.bonds.list.list <<ListboxSelect>> [namespace code {
      focus %W
      # Blank all item backgrounds
      for {set i 0} {$i<[.molefac.bonds.list.list index end]} {incr i} {
	 .molefac.bonds.list.list itemconfigure $i -background {}
      }
      # Get current selection index
      set selbond [.molefac.bonds.list.list curselection]

      # Paint the background of the selected bond
      .molefac.bonds.list.list itemconfigure $selbond -background $::Molefacture::selectcolor
      .molefac.bonds.list.list activate $selbond

      # Get the selected bond
      set selindex [lrange [lindex $::Molefacture::bondlist $selbond] 0 1]

      # Select the corresponding atoms
      ::Molefacture::select_atoms_byvmdindex $selindex

      #puts "DEBUG 3"
      # Compute the bondlength
      variable tmpmolid
      if {![llength $tmpmolid]} { return }
      variable bondcoor
      variable dihedatom
      set dihedatom(1) [lindex $selindex 0]
      set dihedatom(2) [lindex $selindex 1]
      set sel1 [atomselect $tmpmolid "index $dihedatom(1)"]
      set sel2 [atomselect $tmpmolid "index $dihedatom(2)"]
      set bondcoor(1) [join [$sel1 get {x y z}]]
      set bondcoor(2) [join [$sel2 get {x y z}]]
      set ::Molefacture::bondlength [veclength [vecsub $bondcoor(2) $bondcoor(1)]]

      # Choose a dihedral for this bond
      set bonds1 [lsearch -all -inline -not [join [$sel1 getbonds]] $dihedatom(2)]
      set bonds2 [lsearch -all -inline -not [join [$sel2 getbonds]] $dihedatom(1)]
      set dihedatom(0) [lindex $bonds1 0]
      set dihedatom(3) [lindex $bonds2 0]
      # Delete the old marks
      variable dihedmarktags
      foreach tag $dihedmarktags {
	 graphics $tmpmolid delete $tag
      }

      if {[llength $dihedatom(0)] && [llength $dihedatom(3)]} {
	 #puts "dihedatom(0)=$dihedatom(0); [join [$sel1 getbonds]]; $bonds1"
	 #puts "dihedatom(1)=$dihedatom(1)"
	 #puts "dihedatom(2)=$dihedatom(2)"
	 #puts "dihedatom(3)=$dihedatom(3); [join [$sel2 getbonds]]; $bonds2"
	 set sel0 [atomselect $tmpmolid "index $dihedatom(0)"]
	 set bondcoor(0) [join [$sel0 get {x y z}]]
	 set sel3 [atomselect $tmpmolid "index $dihedatom(3)"]
	 set bondcoor(3) [join [$sel3 get {x y z}]]
	 lappend dihedmarktags [graphics $tmpmolid color yellow]
	 lappend dihedmarktags [graphics $tmpmolid sphere $bondcoor(0) radius 0.3]
	 lappend dihedmarktags [graphics $tmpmolid sphere $bondcoor(3) radius 0.3]

	 # Generate two selections for the two molecule halves
	 variable bondsel
   puts "[llength $atomlist]"
	 set indexes1 [join [::util::bondedsel $tmpmolid $dihedatom(2) $dihedatom(1) -maxdepth [llength $atomlist]]]
	 set indexes2 [join [::util::bondedsel $tmpmolid $dihedatom(1) $dihedatom(2) -maxdepth [llength $atomlist]]]
   #puts "bondedhalves: $indexes1 | $indexes2"
   if {[havecommonelems $indexes1 $indexes2 [list $dihedatom(1) $dihedatom(2)]] > 0} {
     set indexes1 $dihedatom(1)
     set indexes2 $dihedatom(2)
   }
	 if {[array exists bondsel]} { $bondsel(1) delete; $bondsel(2) delete }
	 set bondsel(1) [atomselect $tmpmolid "index $indexes1 and not index $dihedatom(2)"]
	 set bondsel(2) [atomselect $tmpmolid "index $indexes2 and not index $dihedatom(1)"]
	 
	 # Compute the bond dihedral angle
   set Molefacture::dihedral [measure dihed [list [list $dihedatom(0) $tmpmolid] [list $dihedatom(1) $tmpmolid] [list $dihedatom(2) $tmpmolid] [list $dihedatom(3) $tmpmolid] ]]
      }

      variable w
      $w.bonds.f2.angle.scale set $::Molefacture::dihedral
   }]

   # This will be executed when an angle is selected:   
   bind $w.angles.list.list <<ListboxSelect>> [namespace code {
      focus %W
      # Blank all item backgrounds
      for {set i 0} {$i<[.molefac.angles.list.list index end]} {incr i} {
	 .molefac.angles.list.list itemconfigure $i -background {}
      }
      # Get current selection index
      set selangle [.molefac.angles.list.list curselection]

      # Paint the background of the selected bond
      .molefac.angles.list.list itemconfigure $selangle -background $::Molefacture::selectcolor
      .molefac.angles.list.list activate $selangle

      # Get the selected bond
      set selindex [lrange [lindex $::Molefacture::anglelist $selangle] 0 2]

      # Select the corresponding atoms
      ::Molefacture::select_atoms_byvmdindex $selindex

      # Get information about this angle
      variable tmpmolid
      variable angle
      variable angleatom
      variable angleaxis
      variable anglesel
      variable anglepicklist
      variable anglemoveatom
      variable anglecoor
      variable dihedmarktags

      # Delete the dihedral marks
      foreach tag $::Molefacture::dihedmarktags {
	 graphics $::Molefacture::tmpmolid delete $tag
      }

      set angleatom(1) [lindex $selindex 0]
      set angleatom(2) [lindex $selindex 1]
      set angleatom(3) [lindex $selindex 2]

      set sel1 [atomselect $tmpmolid "index $angleatom(1)"]
      set sel2 [atomselect $tmpmolid "index $angleatom(2)"]
      set sel3 [atomselect $tmpmolid "index $angleatom(3)"]

      set coor2 [join [$sel2 get {x y z}]]
      set coor1 [join [$sel1 get {x y z}]]
      set coor3 [join [$sel3 get {x y z}]]

      set anglecoor(1) $coor1
      set anglecoor(2) $coor2
      set anglecoor(3) $coor3

#      puts "Subcoors: $coor1 $coor2 $coor3"
      set vec1 [vecsub $coor2 $coor1]
      set vec2 [vecsub $coor2 $coor3]

      set axis [veccross $vec1 $vec2]
#      puts "Doing norm of $axis"
      set angleaxis [vecscale [vecnorm $axis] 1.0]
#      puts "Done"

      # Generate two selections for the two molecule halves
      set indexes1 [join [::util::bondedsel $tmpmolid $angleatom(2) $angleatom(1) -maxdepth [llength $atomlist]]]
      set indexes2 [join [::util::bondedsel $tmpmolid $angleatom(2) $angleatom(3) -maxdepth [llength $atomlist]]]
      #puts "$indexes1 | $indexes2"
      if {[havecommonelems $indexes1 $indexes2 [list $angleatom(1) $angleatom(2) $angleatom(3)]] > 0} {
        set indexes1 $angleatom(1)
        set indexes2 $angleatom(3)
      }
      if {[array exists anglesel]} { catch {$anglesel(1) delete}; catch {$anglesel(2) delete }}
      set anglesel(1) [atomselect $tmpmolid "index $indexes1 and not index $angleatom(2)"]
      set anglesel(2) [atomselect $tmpmolid "index $indexes2 and not index $angleatom(2)"]

      puts "Anglesels: [$anglesel(1) get index] | [$anglesel(2) get index]"

      #Compute the angle
      set angle [measure angle [list [list $angleatom(1) $tmpmolid] [list $angleatom(2) $tmpmolid] [list $angleatom(3) $tmpmolid] ] ]

      variable w
      $w.angles.realangle.scale set $::Molefacture::angle
   }]

   bind $w.angles.realangle.scale <ButtonRelease-1> {
      ::Molefacture::draw_openvalence      
   }

   # This will be executed when an atom is selected:   
   bind $w.val.list.list <<ListboxSelect>> {
      focus %W
      # ::Molefacture::set_pickmode_atomedit

      # Delete the dihedral marks
      foreach tag $::Molefacture::dihedmarktags {
	 graphics $::Molefacture::tmpmolid delete $tag
      }

      # Select the corresponding atoms
      ::Molefacture::select_atoms [.molefac.val.list.list curselection]
   }

   bind $w.bonds.list.list <Key-l> {
      ::Molefacture::lower_bondorder_gui
   }

   bind $w.bonds.list.list <Key-r> {
      ::Molefacture::raise_bondorder_gui
   }

   bind $w.val.list.list <Key-h> {
      ::Molefacture::add_hydrogen_gui
   }

   bind $w.val.list.list <Delete> {
      ::Molefacture::del_atom_gui
   }
#   puts "DEBUG B"
#   Check that Antechamber and SQM are available
    ::Molefacture::checkAntechamber
    ::Molefacture::checkSQM

}

proc ::Molefacture::checkAntechamber {} {
    set m .molefac.menubar.build
    ::ANTECHAMBER::acinit
    if { $::ANTECHAMBER::achere == 0 } {
        $m.abtype entryconfigure 1 -state disable
        $m.abtype entryconfigure 2 -state disable
        $m.abtype entryconfigure 3 -state disable
        $m.charges entryconfigure 2 -state disable
    } else {
        $m.abtype entryconfigure 1 -state normal
        $m.abtype entryconfigure 2 -state normal
        $m.abtype entryconfigure 3 -state normal
        $m.charges entryconfigure 2 -state normal
    }
}

proc ::Molefacture::checkSQM {} {
    ::runsqm::sqminit
    set m .molefac.menubar.build
    if { $::runsqm::sqmhere == 0 } {
        $m.charges entryconfigure 0 -state disable
        $m.charges entryconfigure 1 -state disable
        $m.menu entryconfigure 2 -state disable
    } else {
        $m.charges entryconfigure 0 -state normal
        $m.charges entryconfigure 1 -state normal
        $m.menu entryconfigure 2 -state normal
    }
}

proc ::Molefacture::select_atoms_byvmdindex {indexlist} {
  #Helper proc that translates vmd indices into indices from .molefac.val.list.list
  variable atomlist
  set outputlist [list]
  set translist [list]
  foreach molefind $atomlist {
    lappend translist [lindex $molefind 0]
  }
#  puts "Translation indices $translist"

  foreach vmdind $indexlist {
#    puts "Looking for $vmdind in $translist"
    set i [lsearch -exact -integer $translist $vmdind]
    if {$i != -1} {lappend outputlist $i}
#    puts "Found $i"
  }

#  puts "found indices $outputlist"
  select_atoms $outputlist
}


####################################################################
# This function is used to select atoms.                           #
# It paints the background of the selected list items accordingly  #
# and appends the atomindexes to variable picklist in order to be  #
# independent of the window focus.                                 #
####################################################################

proc ::Molefacture::select_atoms { indexlist } {
#WARNING: The indices passed to this need to be the indices in .molefac.val.list.list
# DO NOT just send the atom indices. Bad things will happen.
   variable picklist {}
   if {![winfo exists .molefac.val.list.list]} { return }

   .molefac.val.list.list selection clear 0 end

   # Blank all item backgrounds
   for {set i 0} {$i<[.molefac.val.list.list index end]} {incr i} {
      .molefac.val.list.list itemconfigure $i -background {}
   }

   # Select the corresponding atoms
#   puts "DEBUG: Indexlist: $indexlist"
   add_atoms_to_selection $indexlist
}
 
proc ::Molefacture::add_atoms_to_selection { indexlist } {
   variable picklist
   variable atomlist
   variable selectcolor
   if {![llength $indexlist] || ![winfo exists .molefac.val.list.list]} { return }

#   puts "DEBUG: Indexlist: $indexlist"
   foreach index $indexlist {
#      set i [lsearch $atomlist "[format "%5s" $index] *"]
#      puts "DEBUG: found $i"
      .molefac.val.list.list selection set $index
      .molefac.val.list.list itemconfigure $index -background $selectcolor      
      set indexatomind [lindex [.molefac.val.list.list get $index] 0]
#      puts $indexatomind
      lappend picklist $indexatomind
   }

   draw_selatoms
   .molefac.val.list.list see $index
   .molefac.val.list.list activate $index
}


####################################################################
# This function can be used by external programs to select atoms.  #
####################################################################

proc ::Molefacture::user_select_atoms { atomdeflist } {
   variable tmpmolid

   # Select the corresponding atoms
   set indexlist {}
   foreach atomdef $atomdeflist {
      set sel [atomselect $tmpmolid "segid [lindex $atomdef 0] and resid [lindex $atomdef 1] and name [lindex $atomdef 2] "]
      lappend indexlist [join [$sel get index]]
      $sel delete
   }

   select_atoms $indexlist
}

proc ::Molefacture::raise_bondorder_gui { } {
   variable tmpmolid

   variable picklist; #[.molefac.val.list.list curselection]
   if {[llength $picklist]!=2} {
      tk_messageBox -icon error -type ok -title Message \
	 -message "To modify the bond order, you should select exactly two atoms!"
      return
   } 

   raise_bondorder $picklist
}

proc ::Molefacture::lower_bondorder_gui { } {
   variable tmpmolid

   variable picklist; # [.molefac.val.list.list curselection]
   if {[llength $picklist]!=2} {
      tk_messageBox -icon error -type ok -title Message \
	 -message "To modify the bond order, you should select exactly two atoms!"
      return
   } 
   lower_bondorder $picklist

}

proc ::Molefacture::invert_gui {} {
  variable tmpmolid

  variable picklist
  foreach mindex $picklist {
    invert_chir $mindex
  } 
  draw_openvalence
}

proc ::Molefacture::set_planar_gui { } {
  variable tmpmolid

  variable picklist;# [.molefac.val.list.list curselection]
  foreach mindex $picklist {
    set_planar $mindex
  }
  draw_openvalence
}


proc ::Molefacture::set_tetra_gui { } {
  variable tmpmolid

  variable picklist; # [.molefac.val.list.list curselection]
  foreach mindex $picklist {
    set_tetra $mindex
  }
}

proc ::Molefacture::del_atom_gui {} {
  variable picklist
  variable tmpmolid
  variable openvalencelist
  variable picklist
  set mother 0
  set curcoor {0 0 0}

  foreach delindex $picklist {
    set retlist [del_atom $delindex 0]
  }

  update_openvalence
  variable bondlist [bondlist]
  variable anglelist [anglelist]

  set mother [lindex $retlist 0]
  set curcoor [lindex $retlist 1]

   set cursel [atomselect $tmpmolid "occupancy > 0.4"]

   variable atomlist
   set i 0
   set mother_real -1
   foreach atom $atomlist {
     if {[lindex $atom 0] == $mother} {
       set mother_real $i
       break
     }
     incr i
   }
   if {$mother_real >= 0} {
      select_atoms $mother_real
   } else {
      # Select the atom closest to the last deleted atom
      set sel [atomselect $tmpmolid "occupancy > 0.4"]
      set dist {}
      foreach coor [$sel get {x y z}] index [$sel get index] {
	 lappend dist [list $index [veclength [vecsub $coor $curcoor]]]
      }
      
      set nearbyind [lindex [lsort -real -index 1 $dist] 0 0]
      set i 0
      foreach atom $atomlist {
        if {[lindex $atom 0] == $nearbyind} {
          select_atoms $i
          break
        }
        incr i
      }
   }
   $cursel delete

}

proc ::Molefacture::add_hydrogen_gui {} {
  variable tmpmolid
  variable openvalencelist
#  variable atomaddlist
  variable picklist

  foreach mindex $picklist {
    add_hydrogen $mindex
  }
}

proc ::Molefacture::export_molecule_gui {} {
   fix_changes
   variable tmpmolid
   set types {
           {{XBGF Files} {.xbgf} }
           {{PDB Files} {.pdb} }
           {{MOL2 Files} {.mol2} }
           }
   set filename [tk_getSaveFile -parent .molefac -filetypes $types -defaultextension ".pdb"]
   set sel [atomselect $tmpmolid "occupancy>=0.8"]
   if {[regexp {xbgf$} $filename] > 0 } {
     write_xbgf "$filename" $sel
   } elseif {[regexp {mol2$} $filename] > 0} {
     $sel writemol2 "$filename"
   } else {
     $sel writepdb "$filename"
   }
   $sel delete

   variable projectsaved
   set projectsaved 1
}

proc ::Molefacture::export_molecule_gui_FEP {} {
   fix_changes
   variable tmpmolid
   variable filename_FEP
   set filename_FEP [tk_getSaveFile -parent .molefac ]
   write_topfile $filename_FEP.top "occupancy >= 0.8"
   set sel [atomselect $tmpmolid "occupancy>=0.8"]
#   set betalist {}
#   puts "betalist: $betalist"
#   foreach beta [$sel get beta] {lappend $betalist $beta}
#   puts "betalist: $betalist"
#   foreach atom $sel {$atom set beta 0}
   $sel writepdb "$filename_FEP.pdb"
#   foreach atoms $sel {$atoms set beta [lrange $betalist 0 0]
#	   set betalist [lreplace $betalist 0 0]
#   }
   $sel writepdb "$filename_FEP.fep"
   variable projectsaved
   set projectsaved 1
}

proc ::Molefacture::apply_moledits { } {

  ### CURRENTLY APPLIES TO ALL ATOMS: MODIFY TO APPLY TO PICKLIST!

  set tmpsel [atomselect $::Molefacture::tmpmolid "occupancy >= 0.5"]
  $tmpsel set resname "$::Molefacture::editmol_resname"
  $tmpsel set segname "$::Molefacture::editmol_segname"
  $tmpsel set chain "$::Molefacture::editmol_chain"
  $tmpsel set resid "$::Molefacture::editmol_resid"
  $tmpsel delete
}

proc ::Molefacture::edit_mol_gui {} {

  variable tmpmolid

  set tmpsel [atomselect $tmpmolid "occupancy >= 0.5"]
  variable editmol_resname [lindex [$tmpsel get resname] 0]
  variable editmol_segname [lindex [$tmpsel get segname] 0]
  variable editmol_chain [lindex [$tmpsel get chain] 0]
  variable editmol_resid [lindex [$tmpsel get resid] 0]

  if {[winfo exists .moleditor]} {
  wm deiconify .moleditor
    raise .moleditor
    return
  }

  set v [toplevel ".moleditor"]
  wm title $v "Molefacture - Edit Residue/Molecule"
  wm resizable $v 0 1

#  label $v.indexlabel -text "Index: "
#  label $v.index -textvariable "$editatom_index"
  frame $v.explain
  label $v.explain.text -text "This modifies the identifying\nproperties of the WHOLE molecule."
  pack $v.explain.text -side left
  frame $v.resname
  label $v.resname.namelabel -text "resname: "
  entry $v.resname.name -textvariable [namespace current]::editmol_resname
  frame $v.segname
  label $v.segname.typelabel -text "segname: "
  entry $v.segname.type -textvariable [namespace current]::editmol_segname
  pack $v.resname.namelabel $v.resname.name -side left
  pack $v.segname.typelabel $v.segname.type -side left

  frame $v.charel
  label $v.charel.chargelabel -text "chain: "
  entry $v.charel.charge -textvariable [namespace current]::editmol_chain
  pack $v.charel.chargelabel $v.charel.charge -side left

  frame $v.resid
  label $v.resid.residlabel -text "resid: "
  entry $v.resid.resid -textvariable [namespace current]::editmol_resid
  pack $v.resid.residlabel $v.resid.resid -side left

  frame $v.buttons
  button $v.buttons.apply -text "Apply" -command ::Molefacture::apply_moledits
#[namespace code {$tmpsel set resname "$editmol_resname"; $tmpsel set segname "$editmol_segname"; $tmpsel set chain "$editmol_chain"  ; after idle destroy .moleditor}]
  button $v.buttons.done -text "Done" -command [namespace code {after idle destroy .moleditor}]
#  button $v.buttons.selapply -text "Apply to selected atoms" -command [::Molefacture::apply_moledits $picklist]
  button $v.buttons.cancel -text "Cancel" -command "after idle destroy $v"
  pack $v.buttons.apply $v.buttons.done $v.buttons.cancel -side left

  pack $v.explain $v.resname $v.segname $v.charel $v.resid $v.buttons
 

  $tmpsel delete
}

proc ::Molefacture::apply_chargeedits { } {

  ### CURRENTLY APPLIES TO ALL ATOMS: MODIFY TO APPLY TO PICKLIST!
  set ::Molefacture::totalcharge $::Molefacture::curcharge
}

proc ::Molefacture::edit_charge_gui {} {

  variable curcharge

  set curcharge $::Molefacture::totalcharge
  if {[winfo exists .chargeeditor]} {
    wm deiconify .chargeeditor
    raise .chargeeditor
    return
  }

  set v [toplevel ".chargeeditor"]
  wm title $v "Molefacture - Modify Total Charge"
  wm resizable $v 0 1

#  label $v.indexlabel -text "Index: "
#  label $v.index -textvariable "$editatom_index"

  frame $v.charel
  label $v.charel.chargelabel -text "Total Charge: "
  entry $v.charel.charge -textvariable [namespace current]::curcharge
  pack $v.charel.chargelabel $v.charel.charge -side left

  frame $v.buttons
  button $v.buttons.apply -text "Apply" -command ::Molefacture::apply_chargeedits
  button $v.buttons.done -text "Done" -command [::Molefacture::apply_chargeedits; namespace code {after idle destroy .chargeeditor}]
  button $v.buttons.cancel -text "Cancel" -command "after idle destroy $v"
  pack $v.buttons.apply $v.buttons.done $v.buttons.cancel -side left

  pack $v.charel $v.buttons
 
}


proc ::Molefacture::check_at_gui { } {
  variable w
  if { [winfo exists .checkatgui] } {
    wm deiconify .checkatgui
    raise .checkatgui
    return
  }

  
  set w [toplevel ".checkatgui"]
  wm title $w "Check Atom Typing"
  wm resizable $w 0 0

  set rownum 0
  frame $w.directories
  set refdir ""
  grid [label $w.directories.refdirlabel -text "Directory containing reference structures:"] -row $rownum -column 0 -sticky e
  grid [entry $w.directories.refdir -width 20 -textvar ::Molefacture::refdir] \
    -row $rownum -column 1 
  grid [button $w.directories.refbrowse -text "Browse" -command ::Molefacture::find_refdir] -row $rownum -column 2 -columnspan 1 -sticky w
  incr rownum
  set testdir ""
  grid [label $w.directories.testdirlabel -text "Directory to run tests:"] -row $rownum -column 0 -sticky e
  grid [entry $w.directories.testdir -width 20 -textvar ::Molefacture::testdir] \
    -row $rownum -column 1 
  grid [button $w.directories.testbrowse -text "Browse" -command ::Molefacture::find_testdir] -row $rownum -column 2 -columnspan 1 -sticky w
  incr rownum
  grid [label $w.directories.at_test -text "Atom type to check:"] -row $rownum -column 0 -sticky e
  grid [entry $w.directories.at_var -width 4 -textvar ::Molefacture::at_test] \
    -row $rownum -column 1 -sticky w 
  incr rownum
  grid [button $w.directories.done -text "Run Comparisons" -command ::Molefacture::run_at_comps] -row $rownum -column 2 -columnspan 1 -sticky e
  pack $w.directories
  
}

proc ::Molefacture::run_at_comps {} {
  set report [::ANTECHAMBER::check_all_indir $::Molefacture::testdir $::Molefacture::refdir $::Molefacture::at_test]
  puts $report
}

proc ::Molefacture::find_refdir {} {
  set filename [tk_chooseDirectory -title "Choose a directory" -initialdir $::Molefacture::refdir]
  set ::Molefacture::refdir "$filename"  
}

proc ::Molefacture::find_testdir {} {
  set filename [tk_chooseDirectory -title "Choose a directory" -initialdir $::Molefacture::testdir]
  set ::Molefacture::testdir "$filename"  
}


proc ::Molefacture::edit_atom_gui {} {

  variable tmpmolid
  variable picklist
  variable periodic

  set tmpsel [atomselect $tmpmolid "index [lindex $picklist 0]"]
  variable editatom_name [$tmpsel get name]
  variable editatom_type [$tmpsel get type]
  variable editatom_element [$tmpsel get element]
  variable editatom_charge [$tmpsel get charge]
  variable editatom_index [$tmpsel get type]
  $tmpsel delete

  if {[llength $picklist] != 1} {
    tk_messageBox -icon error -type ok -title Message -message "You must select exactly one atom to edit"
    return
  }



  if {[winfo exists .atomeditor]} {
  wm deiconify .atomeditor
    raise .atomeditor
    return
  }

  set v [toplevel ".atomeditor"]
  wm title $v "Molefacture - Edit Atom"
  wm resizable $v 0 1

#  label $v.indexlabel -text "Index: "
#  label $v.index -textvariable "$editatom_index"
  frame $v.nametype
  label $v.nametype.namelabel -text "Name: "
  entry $v.nametype.name -textvariable [namespace current]::editatom_name -width 4
  label $v.nametype.typelabel -text "Type: "
  entry $v.nametype.type -textvariable [namespace current]::editatom_type -width 4
  pack $v.nametype.namelabel $v.nametype.name $v.nametype.typelabel $v.nametype.type -side left

  frame $v.charel
  label $v.charel.chargelabel -text "Charge: "
  entry $v.charel.charge -textvariable [namespace current]::editatom_charge -width 4
  label $v.charel.elementlabel -text "Element: "
  menubutton $v.charel.element -height 1 -relief raised -textvariable [namespace current]::editatom_element -menu $v.charel.element.menu
  menu $v.charel.element.menu -tearoff no
  pack $v.charel.chargelabel $v.charel.charge $v.charel.elementlabel $v.charel.element -side left

  frame $v.buttons
  button $v.buttons.finish -text "Apply" -command [namespace code {edit_atom  ; after idle destroy .atomeditor}]
  button $v.buttons.cancel -text "Cancel" -command "after idle destroy $v"
  pack $v.buttons.finish $v.buttons.cancel -side left

  #pack $v.indexlabel $v.index $v.namelabel $v.name $v.typelabel $v.type
#  pack $v.namelabel $v.name $v.typelabel $v.type
#  pack $v.chargelabel $v.charge $v.elementlabel $v.element
#  pack $v.finish $v.cancel
  pack $v.nametype $v.charel $v.buttons

  #Initialize the element menu
  foreach elementname $periodic {
    $v.charel.element.menu add radiobutton -variable [namespace current]::editatom_element -value $elementname -label $elementname
  }

}

#Procs to raise and lower oxidation state
proc ::Molefacture::raise_ox_gui {} {
  variable tmpmolid

  variable picklist;# [.molefac.val.list.list curselection]
  foreach mindex $picklist {
    raise_ox $mindex
  }
  update_openvalence
}

proc ::Molefacture::lower_ox_gui {} {
  variable tmpmolid

  variable picklist;# [.molefac.val.list.list curselection]
  foreach mindex $picklist {
    lower_ox $mindex
  }
  update_openvalence
}

proc ::Molefacture::fill_fragment_menu {} {
  # Looks through the current fragment database, and fills out the fragment menu
  # Each entry in the menu runs the replace hydrogen with fragment proc, using
  # the appropriate fragment
  # Currently clobbers all entries in the old menu, if any

  variable w
  variable addfrags

  foreach fragname [lsort [array names addfrags]] {
    set fragfile $addfrags($fragname)
    $w.menubar.build.addfrag add command -label $fragname -command "::Molefacture::replace_hydrogen_with_fragment_gui {$fragfile}"
  }

}


proc ::Molefacture::replace_hydrogen_with_fragment_gui {fragpath} {
  # GUI dummy proc to find the atoms for replacement, and then replace them
  # with the appropriate fragment

  variable picklist
  foreach mindex $picklist {
    set returncode [replace_hydrogen_with_fragment $fragpath $mindex]
    if {$returncode == 1} {
      tk_messageBox -type ok -title "Error" -icon error -message "You can only replace singly bonded hydrogen atoms! The atom you picked doesn't meet one or both of these criteria"
    }

    # Select a newly added atom (usually a hydrogen)
    global vmd_pick_atom
    variable tmpmolid
    set sel [atomselect $tmpmolid "occupancy >= 0.5"]
    set newind [lindex [$sel get index] end]
    set vmd_pick_atom $newind
    atom_picked_fctn
  }

}

proc ::Molefacture::add_custom_frags {} {
#First let them navigate to the file of interest
  variable w
  set fragfile [tk_getOpenFile]
  if {$fragfile == ""} {return}

  add_frag_file $fragfile
  $w.menubar.build.addfrag delete 3 end
  fill_fragment_menu
}

proc ::Molefacture::add_custom_basefrags {} {
#First let them navigate to the file of interest
  variable w
  set fragfile [tk_getOpenFile]
  if {$fragfile == ""} {return}

  add_basefrag_file $fragfile
  $w.menubar.build.basefrag delete 3 end
  fill_basefrag_menu
}

proc ::Molefacture::reset_basefrags_menu {} {
  variable w

  $w.menubar.build.basefrag delete 3 end
  read_basefrag_file [file join $::env(MOLEFACTUREDIR) lib basemol basefrag.mdb]
  fill_basefrag_menu
}

proc ::Molefacture::fill_basefrag_menu {} {
  # Looks through the current base fragment database, and fills out the fragment menu
  # Each entry in the menu creates a new molecule from 
  # the appropriate fragment
  # Currently clobbers all entries in the old menu, if any

  variable w
  variable basefrags

  #puts "deleting menu..."
  #menu $w.menubar.build.basefrag delete 3 end

#  puts [array names basefrags]
  foreach fragname [lsort [array names basefrags]] {
    set fragfile $basefrags($fragname)
    $w.menubar.build.basefrag add command -label $fragname -command "::Molefacture::new_mol_from_fragment {$fragfile}"
  }

}

proc ::Molefacture::new_mol_gui {} {
  # Use whenever you want to start working on a blank molecule
  # Prompts the user to make sure they don't need to save anything, and then
  # opens up a blank molecule with some preallocated hydrogens
  # Return 0 if they say no at the prompt, 1 otherwise

  set answer [tk_messageBox -message "This will abandon all editing on the current molecule. Are you sure?" -type yesno -icon question]
  switch -- $answer {
    no {return 0}
    yes {}
  }

  new_mol
  return 1
}

proc ::Molefacture::nuc_builder_gui {} {

  variable ralpha
  variable rbeta
  variable rgamma
  variable rdelta
  variable repsilon
  variable rchi
  variable rzeta
  variable nucseq
  variable nucpath

  if {[winfo exists .nucbuilder]} {
    wm deiconify .nucbuilder
    raise .nucbuilder
    return
  }

  set w [toplevel ".nucbuilder"]
  wm title $w "Molefacture Nucleic Acid Builder"
  wm resizable $w no no

  #Frame for buttons for individual nucleotides
  labelframe $w.nucs -bd 2 -relief ridge -text "Add nucleotide" -padx 1m -pady 1m
  frame $w.nucs.buttons
  grid [button $w.nucs.buttons.ade -text "A" -command {::Molefacture::add_nuc A ; ::Molefacture::update_openvalence}] -row 0 -column 0 -columnspan 1 -sticky nsew
  grid [button $w.nucs.buttons.cyt -text "C" -command {::Molefacture::add_nuc C ; ::Molefacture::update_openvalence}] -row 0 -column 1 -columnspan 1 -sticky nsew
  grid [button $w.nucs.buttons.gua -text "G" -command {::Molefacture::add_nuc G ; ::Molefacture::update_openvalence}] -row 0 -column 2 -columnspan 1 -sticky nsew
  grid [button $w.nucs.buttons.thy -text "T" -command {::Molefacture::add_nuc T ; ::Molefacture::update_openvalence}] -row 0 -column 3 -columnspan 1 -sticky nsew
  grid [button $w.nucs.buttons.ura -text "U" -command {::Molefacture::add_nuc U ; ::Molefacture::update_openvalence}] -row 0 -column 4 -columnspan 1 -sticky nsew

  pack $w.nucs.buttons -side top
  pack $w.nucs -side top

  # Alternative: build from a sequence
  frame $w.buildseq
  button $w.buildseq.set_parent_hyd -text "Set parent hydrogen" -command ::Molefacture::set_nuc_parent
  label $w.buildseq.label -text "Add a sequence: "
  entry $w.buildseq.seq -textvar [namespace current]::nucseq
  button $w.buildseq.go -text "Build" -command {::Molefacture::build_nuctextseq $::Molefacture::nucseq; set nucseq ""}

  pack $w.buildseq.set_parent_hyd -side left
  pack $w.buildseq.go -side right
  pack $w.buildseq.seq -side right 
  pack $w.buildseq.label -side right -fill x
  pack $w.buildseq -side top

  labelframe $w.type -bd 2 -relief ridge -text "Structure Type" -padx 1m -pady 1m
  frame $w.type.buttons
  radiobutton $w.type.buttons.rna -text "RNA" -width 6 -value "RNA" -variable [namespace current]::nuctype
  radiobutton $w.type.buttons.ssdna -text "ssDNA" -width 6 -value "DNA" -variable [namespace current]::nuctype
  radiobutton $w.type.buttons.adna -text "ADNA" -width 6 -value "ADNA" -variable [namespace current]::nuctype
  radiobutton $w.type.buttons.bdna -text "BDNA" -width 6 -value "BDNA" -variable [namespace current]::nuctype
  radiobutton $w.type.buttons.zdna -text "ZDNA" -width 6 -value "ZDNA" -variable [namespace current]::nuctype

  pack $w.type.buttons.rna $w.type.buttons.ssdna $w.type.buttons.adna $w.type.buttons.bdna $w.type.buttons.zdna -side left -fill x
  pack $w.type.buttons
  pack $w.type -side top

  labelframe $w.struct -bd 2 -relief ridge -text "Dihedral Angles" -padx 1m -pady 1m
  frame $w.struct.ss
  frame $w.struct.ss.row1
  frame $w.struct.ss.row2
  frame $w.struct.ss.row3
  frame $w.struct.ss.row4
  frame $w.struct.ss.row5
  frame $w.struct.ss.row6
  frame $w.struct.ss.row7
  label $w.struct.ss.row1.alphalabel -text "Alpha angle: "
  entry $w.struct.ss.row1.alpha -textvar [namespace current]::ralpha
  label $w.struct.ss.row2.betalabel -text "Beta angle: "
  entry $w.struct.ss.row2.beta -textvar [namespace current]::rbeta
  label $w.struct.ss.row3.gammalabel -text "Gamma angle: "
  entry $w.struct.ss.row3.gamma -textvar [namespace current]::rgamma
  label $w.struct.ss.row4.deltalabel -text "Delta angle: "
  entry $w.struct.ss.row4.delta -textvar [namespace current]::rdelta
  label $w.struct.ss.row5.epsilonlabel -text "Epsilon angle: "
  entry $w.struct.ss.row5.epsilon -textvar [namespace current]::repsilon
  label $w.struct.ss.row6.chilabel -text "Chi angle: "
  entry $w.struct.ss.row6.chi -textvar [namespace current]::rchi
  label $w.struct.ss.row7.zetalabel -text "Zeta angle: "
  entry $w.struct.ss.row7.zeta -textvar [namespace current]::rzeta

  frame $w.struct.im
  canvas $w.struct.im.pic -height 250 -width 152
  image create photo $w.struct.im.dihedpic -format gif -file [file join $nucpath dna_dihedrals.gif]
  $w.struct.im.pic create image 1 1 -anchor nw -image $w.struct.im.dihedpic

  pack $w.struct.ss.row1.alphalabel $w.struct.ss.row1.alpha -side left
  pack $w.struct.ss.row2.betalabel $w.struct.ss.row2.beta -side left
  pack $w.struct.ss.row3.gammalabel $w.struct.ss.row3.gamma -side left
  pack $w.struct.ss.row4.deltalabel $w.struct.ss.row4.delta -side left
  pack $w.struct.ss.row5.epsilonlabel $w.struct.ss.row5.epsilon -side left
  pack $w.struct.ss.row6.chilabel $w.struct.ss.row6.chi -side left
  pack $w.struct.ss.row7.zetalabel $w.struct.ss.row7.zeta -side left

  pack $w.struct.ss.row1 $w.struct.ss.row2 $w.struct.ss.row3 $w.struct.ss.row4 $w.struct.ss.row5 $w.struct.ss.row6 $w.struct.ss.row7 -side top

  pack $w.struct.im.pic -side left

  pack $w.struct.ss -side left
  pack $w.struct.im -side right

  pack $w.struct -side top

}

proc ::Molefacture::prot_builder_gui {} {

  variable phi
  variable psi
  variable aaseq

  if {[winfo exists .protbuilder]} {
    wm deiconify .protbuilder
    raise .protbuilder
    return
  }

  set w [toplevel ".protbuilder"]
  wm title $w "Molefacture Protein Builder"
  wm resizable $w no no

  #Frame for buttons for individual amino acids
  labelframe $w.aas -bd 2 -relief ridge -text "Add amino acids" -padx 1m -pady 1m
  frame $w.aas.buttons
  grid [button $w.aas.buttons.ala -text "ALA" -command {::Molefacture::add_aa ALA}] -row 0 -column 0 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.arg -text "ARG" -command {::Molefacture::add_aa ARG}] -row 0 -column 1 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.asn -text "ASN" -command {::Molefacture::add_aa ASN}] -row 0 -column 2 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.asp -text "ASP" -command {::Molefacture::add_aa ASP}] -row 0 -column 3 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.cys -text "CYS" -command {::Molefacture::add_aa CYS}] -row 0 -column 4 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.gln -text "GLN" -command {::Molefacture::add_aa GLN}] -row 0 -column 5 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.glu -text "GLU" -command {::Molefacture::add_aa GLU}] -row 0 -column 6 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.gly -text "GLY" -command {::Molefacture::add_aa GLY}] -row 0 -column 7 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.his -text "HIS" -command {::Molefacture::add_aa HIS}] -row 0 -column 8 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.ile -text "ILE" -command {::Molefacture::add_aa ILE}] -row 0 -column 9 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.leu -text "LEU" -command {::Molefacture::add_aa LEU}] -row 1 -column 0 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.lys -text "LYS" -command {::Molefacture::add_aa LYS}] -row 1 -column 1 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.met -text "MET" -command {::Molefacture::add_aa MET}] -row 1 -column 2 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.phe -text "PHE" -command {::Molefacture::add_aa PHE}] -row 1 -column 3 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.pro -text "PRO" -command {::Molefacture::add_proline}] -row 1 -column 4 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.ser -text "SER" -command {::Molefacture::add_aa SER}] -row 1 -column 5 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.thr -text "THR" -command {::Molefacture::add_aa THR}] -row 1 -column 6 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.trp -text "TRP" -command {::Molefacture::add_aa TRP}] -row 1 -column 7 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.tyr -text "TYR" -command {::Molefacture::add_aa TYR}] -row 1 -column 8 -columnspan 1 -sticky nsew
  grid [button $w.aas.buttons.val -text "VAL" -command {::Molefacture::add_aa VAL}] -row 1 -column 9 -columnspan 1 -sticky nsew

  pack $w.aas.buttons -side top
  pack $w.aas -side top

  # Alternative: build from a sequence
  frame $w.buildseq
  button $w.buildseq.set_parent_hyd -text "Set parent hydrogen" -command ::Molefacture::set_prot_parent
  label $w.buildseq.label -text "Add a sequence: "
  entry $w.buildseq.seq -textvar [namespace current]::aaseq
  button $w.buildseq.go -text "Build" -command {::Molefacture::build_textseq $::Molefacture::aaseq; set aaseq ""}

  pack $w.buildseq.set_parent_hyd -side left
  pack $w.buildseq.go -side right
  pack $w.buildseq.seq -side right 
  pack $w.buildseq.label -side right -fill x
  pack $w.buildseq -side top




  labelframe $w.ss -bd 2 -relief ridge -text "Phi/Psi Angles" -padx 1m -pady 1m
  frame $w.ss.row1
  frame $w.ss.row2
  label $w.ss.row1.philabel -text "Phi angle: "
  entry $w.ss.row1.phi -textvar [namespace current]::phi
  button $w.ss.row1.ahel -text "Alpha helix" -width 15 -command {::Molefacture::set_phipsi -57 -47}
  button $w.ss.row1.bs -text "Beta sheet" -width 15 -command {::Molefacture::set_phipsi -120 113}
  label $w.ss.row2.psilabel -text "Psi angle: "
  entry $w.ss.row2.psi -textvar [namespace current]::psi
  button $w.ss.row2.turn -text "Turn" -width 15 -command {::Molefacture::set_phipsi -60 30}
  button $w.ss.row2.straight -text "Straight" -width 15 -command {::Molefacture::set_phipsi -180 180}

  pack $w.ss.row1.philabel $w.ss.row1.phi $w.ss.row1.ahel $w.ss.row1.bs -side left
  pack $w.ss.row2.psilabel $w.ss.row2.psi $w.ss.row2.turn $w.ss.row2.straight -side left

  pack $w.ss.row1 $w.ss.row2 -side top
  pack $w.ss -side top

}

proc ::Molefacture::write_topology_gui {} {
   fix_changes

   variable tmpmolid
   set types {
           {{CHARMm topology file} {.top} }
           }
   set proceed [::Molefacture::unique_atomnames]
   if {$proceed == "yes"} {
        set filename [tk_getSaveFile -parent .molefac -filetypes $types -defaultextension ".top"]
        write_topfile $filename "occupancy >= 0.5"
    }
#   $sel delete
}

proc ::Molefacture::run_idatm {} {
  fix_changes
  variable tmpmolid

  set tmpsel [atomselect $tmpmolid "occupancy >= 0.8"]
  if { [$tmpsel num] > 0 } {
    ::IDATM::runtyping $tmpsel
  }
  $tmpsel delete
  update_openvalence
}

proc ::Molefacture::run_ante {} {
  fix_changes
  variable tmpmolid
  variable totalcharge

  set tmpsel [atomselect $tmpmolid "occupancy >= 0.8"]
  #::IDATM::runtyping $tmpsel
  if { [$tmpsel num] > 0 } {
    set result [::ANTECHAMBER::ac_type_in_place $tmpsel rc $totalcharge]
    if { $result != "0" } {
        tk_messageBox -message $result -type ok -title "Antechamber Error - Check Structure" -icon error
    }
  }
  $tmpsel delete
  update_openvalence
  draw_openvalence
  draw_selatoms
  clean_dihed_tags
}

proc ::Molefacture::am1_geo_opt {} {
  fix_changes
  variable tmpmolid
  variable totalcharge 
  set tmpsel [atomselect $tmpmolid "occupancy >= 0.8"]
  if { [$tmpsel num] > 1 } {
    ::runsqm::SQMopt $tmpsel $totalcharge
  }
  #set optmol [::ANTECHAMBER::run_ac_typing $tmpsel bcc $totalcharge]
  #mol delete $optmol 
  #set mopacmol [mol new mopac.pdb]
  #set geo_opt_sel [atomselect $optmol all]
  #$geo_opt_sel move [measure fit $geo_opt_sel $tmpsel]
  #$tmpsel set {x y z} [$geo_opt_sel get {x y z}]
  #mol delete $mopacmol

  update_openvalence
  draw_openvalence
  draw_selatoms
  clean_dihed_tags
}



proc ::Molefacture::run_ante_opls {} {
  fix_changes
  variable tmpmolid
  variable totalcharge

  set tmpsel [atomselect $tmpmolid "occupancy >= 0.8"]
  #::IDATM::runtyping $tmpsel
  if { [$tmpsel num] > 1 } { 
    set result [::ANTECHAMBER::ac_type_in_place $tmpsel rc $totalcharge "CUSTOM$::Molefacture::OPLSatomdef"]
    if { $result != "0" } {
        tk_messageBox -message $result -type ok -title "Antechamber Error - Check Structure" -icon error
    }
  }
  $tmpsel delete
  update_openvalence
  draw_openvalence
  draw_selatoms
  clean_dihed_tags
}

proc ::Molefacture::ante_gui {} {
  fix_changes
  ::ANTECHAMBER::antechamber_gui 1
  update_openvalence
}

proc ::Molefacture::fep_simulations_gui {} {
  variable selectcolor	
  variable fepmolid
  variable tmpmolid
  set w [toplevel ".fepsimul"]
  wm title $w "FEP Simulations options"
  wm resizable $w no no

  labelframe $w.buttons -bd 2 -relief ridge -text "Actions" -padx 2 -pady 2
  button $w.buttons.incomming -text "Define incoming atoms (g)" -command ::Molefacture::flag_incomingatoms
  button $w.buttons.outgoing -text "Define outgoing atoms (v)" -command ::Molefacture::flag_outgoingatoms
  button $w.buttons.alchemify -text "Run Alchemify" -command ::Molefacture::run_alchemify 
  button $w.buttons.common -text "Clear (x)" -command ::Molefacture::flag_commonatoms
  pack $w.buttons.incomming $w.buttons.outgoing $w.buttons.common $w.buttons.alchemify -side left  -fill x -expand 1
  pack $w.buttons  -side top -fill x

  labelframe $w.options -bd 2 -relief ridge -text "Typing and charges"
  frame $w.options.line1
  label $w.options.line1.label -text "Typing:" 
  radiobutton $w.options.line1.notypingopt -text "None" -value "None" -variable [namespace current]::feptyping
  radiobutton $w.options.line1.gafftypingopt -text "GAFF" -value "GAFF" -variable [namespace current]::feptyping
  radiobutton $w.options.line1.oplstypingopt -text "OPLS" -value "OPLS" -variable [namespace current]::feptyping
  pack $w.options.line1.label $w.options.line1.notypingopt $w.options.line1.gafftypingopt $w.options.line1.oplstypingopt -side left -fill x -expand 1 
  pack $w.options.line1 -side left -fill x -expand 1

  frame $w.options.line2
  label $w.options.line2.sclabel -text "Starting charge:" 
  entry $w.options.line2.sc -textvar [namespace current]::fepstartcharge -width 5
  label $w.options.line2.eclabel -text "Ending charge:" 
  entry $w.options.line2.ec -textvar [namespace current]::fependcharge -width 5

  pack $w.options.line2.sclabel $w.options.line2.sc $w.options.line2.eclabel $w.options.line2.ec -side left -fill x -expand 1
  pack $w.options.line2 -side left -fill x -expand 1

  pack $w.options -side top -fill x

  labelframe $w.saveoptions -bd 2 -relief ridge -text "Output options" -padx 2 -pady 2
  frame $w.saveoptions.line1
  label $w.saveoptions.line1.outpreflabel -text "Output prefix:"
  entry $w.saveoptions.line1.outpref -textvar [namespace current]::FEPoutprefix
  checkbutton $w.saveoptions.line1.mpflag -text "Merge into parent molecule" -variable [namespace current]::FEPreplaceflag
  pack $w.saveoptions.line1.outpreflabel $w.saveoptions.line1.outpref $w.saveoptions.line1.mpflag -side left -fill x -expand 1
  pack $w.saveoptions.line1 -side top -fill x
  frame $w.saveoptions.line2
  label $w.saveoptions.line2.parlabel -text "Parent prefix:"
  entry $w.saveoptions.line2.parmolid -textvar [namespace current]::FEPparentmol
  label $w.saveoptions.line2.seglabel -text "Segname:"
  entry $w.saveoptions.line2.segname -textvar [namespace current]::FEPdelseg
  label $w.saveoptions.line2.reslabel -text "Resid:"
  entry $w.saveoptions.line2.resid -textvar [namespace current]::FEPdelres
  pack $w.saveoptions.line2.parlabel $w.saveoptions.line2.parmolid $w.saveoptions.line2.seglabel $w.saveoptions.line2.segname $w.saveoptions.line2.reslabel $w.saveoptions.line2.resid -side left -fill x -expand 1
  pack $w.saveoptions.line2 -side top -fill x
  pack $w.saveoptions -side top -fill x



  labelframe $w.betalist -bd 2 -relief ridge -text "FEP selections" -padx 2 -pady 2
  
  label $w.betalist.format -font {tkFixed 9} -textvariable ::Molefacture::FEPlist -relief flat -bd 2 -justify left;
  scrollbar $w.betalist.scroll -command "$w.betalist.list yview" -takefocus 0
  listbox $w.betalist.list -activestyle dotbox -yscroll "$w.betalist.scroll set" -font {tkFixed 9} \
  -width 60 -height 5 -setgrid 1 -selectmode extended -selectbackground $selectcolor -listvariable ::Molefacture::FEPatomlist
  pack $w.betalist.format -side top -anchor w
  pack $w.betalist.list -side left -fill both -expand 1
  pack $w.betalist.scroll -side left -fill y
  pack $w.betalist -padx 1m -pady 1m -fill both -expand 1  

  set fepmolid [mol new]
  graphics $fepmolid material Transparent
  mol top $tmpmolid
  display resetview 

#  Hotkeys forme $w.val.list

  user add key g {
    ::Molefacture::flag_incomingatoms
  }
  user add key v {
    ::Molefacture::flag_outgoingatoms
  }
  user add key x {
	 ::Molefacture::flag_commonatoms
  }
}

proc ::Molefacture::run_psfgen_FEP {} {

  variable FEPreplaceflag
  variable FEPdelseg

  if {$FEPreplaceflag > 0} {
    set segn $FEPdelseg
  } else {
    set segn U
  }

   package require psfgen
   psfcontext new delete
   topology "Molefacture_fep_temp.top"
   segment $segn {pdb Molefacture_fep_temp_combined.pdb }
   coordpdb Molefacture_fep_temp_combined.pdb $segn
   writepsf Molefacture_fep_temp_combined.psf

   set tmpmol [mol new Molefacture_fep_temp_combined.pdb]
   set tmpsel [atomselect $tmpmol all]
   $tmpsel set segname $segn
   $tmpsel writepdb Molefacture_fep_temp_combined.pdb
   $tmpsel delete
   mol delete $tmpmol
}

proc ::Molefacture::run_alchemify {} {
   variable tmpmolid
   variable filename_FEP
   variable FEPreplaceflag
   variable FEPoutprefix
   variable FEPdelseg
   variable FEPdelres
   variable FEPparentmol

   mol reanalyze $tmpmolid
   do_fep_typing
   run_psfgen_FEP
   alchemify Molefacture_fep_temp_combined.psf Molefacture_fep_ready.psf Molefacture_fep_temp_combined.pdb

   if {$FEPreplaceflag != 0} {
     mol new Molefacture_fep_ready.psf
     mol addfile Molefacture_fep_temp_combined.pdb

     package require psfgen
     psfcontext new delete

     readpsf ${FEPparentmol}.psf
     coordpdb ${FEPparentmol}.pdb

     delatom ${FEPdelseg} ${FEPdelres}

     readpsf Molefacture_fep_ready.psf
     coordpdb Molefacture_fep_temp_combined.pdb

     writepsf ${FEPoutprefix}_temp.psf
     writepdb ${FEPoutprefix}.pdb

     set fepatoms [atomselect top all]
     mol new ${FEPoutprefix}.pdb
     set sel [atomselect top all]
     $sel set beta 0
     set fepnewatoms [atomselect top "segname ${FEPdelseg} and resid ${FEPdelres}"]
     $fepnewatoms set beta [$fepatoms get beta]
     $fepnewatoms set occupancy [$fepatoms get occupancy]

     $sel writepdb ${FEPoutprefix}.pdb

     file rename -force Molefacture_fep_temp.top ${FEPoutprefix}.top

     alchemify ${FEPoutprefix}_temp.psf ${FEPoutprefix}.psf ${FEPoutprefix}.pdb

     $sel delete
     $fepatoms delete
     $fepnewatoms delete
     
     mol delete top
     mol delete top

     file delete -force ${FEPoutprefix}_temp.psf

   } else {
     file rename -force Molefacture_fep_ready.psf ${FEPoutprefix}.psf
     file rename -force Molefacture_fep_temp_combined.pdb ${FEPoutprefix}.pdb
     file rename -force Molefacture_fep_temp.top ${FEPoutprefix}.top

     #set newcoormol [mol new ${FEPoutprefix}.pdb]
     #set replaceatoms [atomselect $tmpmolid "occupancy >= 0.5"]
     #set newatoms [atomselect $newcoormol all]
     #$replaceatoms set {x y z} [$newatoms get {x y z}]
     #$replaceatoms delete
     #$newatoms delete
     #mol delete $newcoormol
     atom_picked_fctn
     update_openvalence
     draw_openvalence
   }
}


proc ::Molefacture::runante_am1bcc {} {
  fix_changes
  variable tmpmolid
  variable totalcharge

  set tmpsel [atomselect $tmpmolid "occupancy >= 0.8"]
  #::IDATM::runtyping $tmpsel
  if { [$tmpsel num] > 0 } {  
    ::Molefacture::run_sqm_mullik AM1
    set result [::ANTECHAMBER::getAM1BCC $tmpsel bcc $totalcharge none ]
    if { $result != "0" } {
        tk_messageBox -message $result -type ok -title "Antechamber Error - Check Structure" -icon error
    }
  }
  $tmpsel delete
  update_openvalence
  draw_openvalence
  draw_selatoms
  clean_dihed_tags
}


proc ::Molefacture::run_sqm_mullik { {qmlvl ""} } {
  fix_changes
  variable tmpmolid
  variable totalcharge
  set curqmtheory $::runsqm::qmtheory
  if { $qmlvl !=  "" } {
    variable ::runsqm::qmtheory $qmlvl
  }
  set tmpsel [atomselect $tmpmolid "occupancy >= 0.5"]
  if { [$tmpsel num] > 0 } {
      ::runsqm::get_charges $tmpsel $totalcharge "Mulliken"
  }
  variable ::runsqm::qmtheory $curqmtheory
  $tmpsel delete
  update_openvalence
  return
}

proc ::Molefacture::run_sqm_cm1a {} {
  fix_changes
  variable tmpmolid
  variable totalcharge
  set curqmtheory $::runsqm::qmtheory
  variable ::runsqm::qmtheory "AM1"
  set tmpsel [atomselect $tmpmolid "occupancy >= 0.5"]
  if { [$tmpsel num] > 0 } {
    ::runsqm::get_charges $tmpsel $totalcharge "cm1a"
  }
  variable ::runsqm::qmtheory $curqmtheory
  $tmpsel delete
  update_openvalence
  return
}

proc ::Molefacture::antechamber_setting_gui { } {
  variable w

  if { [winfo exists .antechambersettingsgui] } {
    wm deiconify .antechambersettingsgui
    raise .antechambersettingsgui
    return
  }

  
  set w [toplevel ".antechambersettingsgui"]
  wm title $w "Antechamber Settings"
  wm resizable $w 0 0

  set rownum 0
  frame $w.settings

  grid [label $w.settings.amberhomelabel -text "AMBERHOME:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.maxcyc -width 20 -textvar env(AMBERHOME)] \
    -row $rownum -column 1 
  grid [button $w.settings.browse -text "Browse" -command ::Molefacture::find_amberhome] -row $rownum -column 2 -columnspan 1 -sticky w
  incr rownum
 
  grid [label $w.settings.oplsdefloclabel -text "OPLS Atom Typing Rules:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.oplsdefloc -width 20 -textvar ::Molefacture::OPLSatomdef] \
    -row $rownum -column 1 
  grid [button $w.settings.oplsbrowse -text "Browse" -command ::Molefacture::find_OPLSfile] -row $rownum -column 2 -columnspan 1 -sticky w
  incr rownum 

  grid [button $w.settings.done -text "Done" -command ::Molefacture::set_antechamber_settings] -row $rownum -column 2 -columnspan 1 -sticky e
  pack $w.settings
  
}

proc ::Molefacture::find_amberhome {} {
  set filename [tk_chooseDirectory -title "Choose a directory" -initialdir "$::env(HOME)"]
#  puts $filename
  set ::env(AMBERHOME) "$filename"  
}

proc ::Molefacture::find_sqm {} {
  set filename [tk_getOpenFile -title "Please locate sqm executable" -initialdir "$::env(HOME)"]
#  puts $filename
  set ::runsqm::sqmpath "$filename" 
}

proc ::Molefacture::find_OPLSfile {} {
  set filename [tk_getOpenFile -title "Select OPLS atom typing .DEF file" -initialdir [file dirname $::Molefacture::OPLSatomdef]]
#  puts $filename
  set ::Molefacture::OPLSatomdef "$filename"  
}

proc ::Molefacture::set_antechamber_settings {} {
    if { [info exist ::env(AMBERHOME)] == 1 && [string length $::env(AMBERHOME)] > 0 } { 
        set  ::ANTECHAMBER::acpath \
            [::ExecTool::find -interactive \
            -description "Antechamber" \
            -path [file join $::env(AMBERHOME) bin $::ANTECHAMBER::acbin ] $::ANTECHAMBER::acbin]
    } else {
       set  ::ANTECHAMBER::acpath \
            [::ExecTool::find -interactive \
            -description "Antechamber" $::ANTECHAMBER::acbin] 
    }
   ::ANTECHAMBER::set_oplsdef_loc $::Molefacture::OPLSatomdef
   ::Molefacture::checkAntechamber
   destroy .antechambersettingsgui
}

proc ::Molefacture::sqm_settings_gui { } {
  variable w

  if { [winfo exists .sqmsettingsgui] } {
    wm deiconify .sqmsettingsgui
    raise .sqmsettingsgui
    return
  }

  set w [toplevel ".sqmsettingsgui"]
  wm title $w "SQM Settings"
  wm resizable $w 0 0

  set rownum 0
  frame $w.settings
  grid [label $w.settings.sqmloclalbel -text "Location of sqm executable:"] -row $rownum -column 0 -sticky w
  incr rownum
  grid [entry $w.settings.sqmloc -width 25 -textvar ::runsqm::sqmpath] \
    -row $rownum -column 0 -columnspan 1
  grid [button $w.settings.browse -text "Browse" -command ::Molefacture::find_sqm] -row $rownum -column 1 -columnspan 3 -sticky w
  incr rownum

  grid [label $w.settings.qmtheorylabel -text "QM Theory:"] -row $rownum -column 0 -sticky w
  grid [menubutton $w.settings.qmtheory -menu $w.settings.qmtheory.menu -textvar ::runsqm::qmtheory -relief raised -width 10] \
    -row $rownum -column 1 -columnspan 3 -sticky w
  menu $w.settings.qmtheory.menu -tearoff no
  $w.settings.qmtheory.menu add radiobutton -label "AM1" -variable ::runsqm::qmtheory -value "AM1"
  $w.settings.qmtheory.menu add radiobutton -label "PM3" -variable ::runsqm::qmtheory -value "PM3"
  $w.settings.qmtheory.menu add radiobutton -label "MNDO" -variable ::runsqm::qmtheory -value "MNDO"
  $w.settings.qmtheory.menu add radiobutton -label "RM1" -variable ::runsqm::qmtheory -value "RM1"
  $w.settings.qmtheory.menu add radiobutton -label "PM3-PDDG" -variable ::runsqm::qmtheory -value "PM3-PDDG"
  $w.settings.qmtheory.menu add radiobutton -label "MNDO-PDDG" -variable ::runsqm::qmtheory -value "MNDO-PDDG"
  $w.settings.qmtheory.menu add radiobutton -label "PM3-CARB1" -variable ::runsqm::qmtheory -value "PM3-CARB1"
  $w.settings.qmtheory.menu add radiobutton -label "PM6" -variable ::runsqm::qmtheory -value "PM6"
  incr rownum
  grid [label $w.settings.maxcyclabel -text "Max minimisation cycles:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.maxcyc -width 4 -textvar ::runsqm::maxcyc] \
    -row $rownum -column 1 -sticky ew
  incr rownum
  grid [label $w.settings.printchargeslabel -text "Print charges during minimisation:"] -row $rownum -column 0 -sticky ew
  grid [checkbutton $w.settings.printcharges -variable ::runsqm::printcharges] \
    -row $rownum -column 1 -sticky ew
  incr rownum
  grid [label $w.settings.ntprlabel -text "Number of steps between printing\nminimisation output:" -justify left] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.ntpr -width 3 -textvar ::runsqm::ntpr] \
    -row $rownum -column 1 -sticky ew
  incr rownum
  grid [label $w.settings.peptidecorrlabel -text "Apply an MM correction to peptide\nlinkages:" -justify left] -row $rownum -column 0 -sticky w
  grid [checkbutton $w.settings.peptidecorr -variable ::runsqm::peptidecorr] \
    -row $rownum -column 1 -sticky ew
  incr rownum
  grid [label $w.settings.verbositylabel -text "Verbosity Level"] -row $rownum -column 0 -sticky w
  grid [menubutton $w.settings.verbosity -menu $w.settings.verbosity.menu -textvar ::runsqm::verbosity -relief raised -width 2] \
    -row $rownum -column 1 -columnspan 1 -sticky ew
  menu $w.settings.verbosity.menu -tearoff no
  $w.settings.verbosity.menu add radiobutton -label "0" -variable ::runsqm::verbosity -value "0"
  $w.settings.verbosity.menu add radiobutton -label "1" -variable ::runsqm::verbosity -value "1"
  $w.settings.verbosity.menu add radiobutton -label "2" -variable ::runsqm::verbosity -value "2"
  $w.settings.verbosity.menu add radiobutton -label "3" -variable ::runsqm::verbosity -value "3"
  $w.settings.verbosity.menu add radiobutton -label "4" -variable ::runsqm::verbosity -value "4"
  $w.settings.verbosity.menu add radiobutton -label "5" -variable ::runsqm::verbosity -value "5"
  incr rownum
  grid [label $w.settings.itrmaxlabel -text "Maximin number of scf iterations:" -justify left] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.itrmax -width 3 -textvar ::runsqm::itrmax] \
    -row $rownum -column 1 -sticky ew
  incr rownum
 

  grid [label $w.settings.scfconvlabel -justify left -text "SCF convergence criteria:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.scfconvcoeff -width 2 -textvar ::runsqm::scfconvcoeff] \
    -row $rownum -column 1 -sticky w 
  grid [label $w.settings.scfconvexp -justify left -text "x10^"] -row $rownum -column 1 -sticky e
  grid [scale $w.settings.scfconvscale -orient horizontal -length 100 -from -2 -to -14 -variable runsqm::scfconvexp  -tickinterval -4] -row $rownum -column 2 -columnspan 1 -sticky ew
  incr rownum
 
  grid [label $w.settings.scferrabel -justify left -text "Maximum absolute SCF error:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.scferrcoeff -width 2 -textvar ::runsqm::errconvcoeff] \
    -row $rownum -column 1 -sticky w 
  grid [label $w.settings.scfcerrexp -justify left -text "x10^"] -row $rownum -column 1 -sticky e
  grid [scale $w.settings.scferrscale -orient horizontal -length 100 -from -1 -to -16 -variable runsqm::errconvexp  -tickinterval -5] -row $rownum -column 2 -columnspan 1 -sticky ew
  incr rownum
  
  grid [button $w.settings.defaults -text "Restore Defaults" -command ::runsqm::setdefaults] -row $rownum -column 0 -columnspan 1 -sticky e
  grid [button $w.settings.apply -text "Done" -command ::Molefacture::checkSQMpars] -row $rownum -column 1 -columnspan 1
  pack $w.settings

}

proc ::Molefacture::checkAntechamber {} {

    set m .molefac.menubar.build
    ::ANTECHAMBER::acinit
    if { $::ANTECHAMBER::achere == 0 } {
        $m.abtype entryconfigure 1 -state disable
        $m.abtype entryconfigure 2 -state disable
        $m.abtype entryconfigure 3 -state disable
        $m.charges entryconfigure 2 -state disable
    } else {
        $m.abtype entryconfigure 1 -state normal
        $m.abtype entryconfigure 2 -state normal
        $m.abtype entryconfigure 3 -state normal
        $m.charges entryconfigure 2 -state normal
    }

}

proc ::Molefacture::checkSQMpars {} {

  set display_error 0 
  set msg ""
  set originalpath $::runsqm::sqmpath
  set ::runsqm::sqmhere 0 
  ::Molefacture::checkSQM
   if {$::runsqm::sqmhere == 0} {
      set display_error 1
      append msg "Warning: $originalpath not executable - sqm calculations not possible."
   } 
   if {$::runsqm::maxcyc == 0} {
      set display_error 1 
      append msg "Warning: With maxcyc = 0 only a single point calculation will be run.\n"
   }
   if {$::runsqm::itrmax > 1000} {
      set display_error 1
      append msg "Warning: If SCF convergence hasn't been reached by 1000 steps it is unlikely to converge. It is recommended that the maximum number of SCF steps is set to 1000 or less.\n"
   }
   if {$::runsqm::scfconvcoeff < 1 || $::runsqm::scfconvcoeff > 9} {
     set display_error 2
     append msg "Error: SCF convergence criteria must have a coefficient between 1 and 9.\n"
   }
   if {$::runsqm::errconvcoeff < 1 || $::runsqm::errconvcoeff > 9} {
     set display_error 2
     append msg "Error: Maximum SCF error criteria must have a coefficient between 1 and 9.\n"
   }
   if { $display_error == 2 } {
    tk_messageBox -message $msg -type ok -title "Check SQM parameters" -icon error
   } elseif { $display_error == 1} {
     tk_messageBox -message $msg -type ok -title "Check SQM parameters" -icon warning
     destroy .sqmsettingsgui
   } else {
     destroy .sqmsettingsgui
   }
   return

}

proc ::Molefacture::checkSQM {} {

    set ::runsqm::smqhere [::runsqm::sqminit]
    set m .molefac.menubar.build
    if { $::runsqm::sqmhere == 0 } {
        $m.charges entryconfigure 0 -state disable
        $m.charges entryconfigure 1 -state disable
        $m.menu entryconfigure 2 -state disable
    } else {
        $m.charges entryconfigure 0 -state normal
        $m.charges entryconfigure 1 -state normal
        $m.menu entryconfigure 2 -state normal
    }

}

proc ::Molefacture::write_namdfiles_gui { } {
  variable w

  if { [winfo exists .writenamdfilesgui] } {
    wm deiconify .writenamdfilesgui
    raise .writenamdfilesgui
    return
  }

  
  set w [toplevel ".writenamdfilesgui"]
  wm title $w "Write NAMD input files"
  wm resizable $w 0 0

  set rownum 0
  frame $w.writefiles
  variable prmloc
  variable prmprefix
  variable outloc
  if {![info exists prmloc]} {
    set prmloc ""
  }
  if {![info exists namdfilesprefix]} {
    set prmprefix "molecule"
  }
  if {![info exists prefix]} {
    set outloc "./"
  }
  grid [label $w.writefiles.info -text "This tools writes out the topology (top), structure (psf) and coordinates (pdb) files"] -row $rownum -column 0 -columnspan 4 -sticky w
  #grid [label $w.writefiles.info -text "This tools writes out the structure (psf), coordinates (pdb)\nand parameter (prm) files that you need to run NAMD\n"] -row $rownum -column 0 -columnspan 4 -sticky w
  #grid [button $w.settings.browse -text "Browse" -command ::Molefacture::find_amberhome] -row $rownum -column 2 -columnspan 1 -sticky w
  incr rownum
 
  #grid [label $w.writefiles.prmloclabel -text "Location of force-field\nparameter file:"] -row $rownum -column 0 -columnspan 2 -sticky w
  #grid [entry $w.writefiles.prmloc -width 20 -textvar ::Molefacture::prmloc ] -row $rownum -column 2 
  #grid [button $w.writefiles.prmbrowse -text "Browse" -command ::Molefacture::findprmfile] -row $rownum -column 3 -columnspan 1 -sticky w
  #incr rownum 

  grid [label $w.writefiles.prefixlabel -text "Prefix for output files:"] -row $rownum -column 0 -columnspan 2 -sticky w
  grid [entry $w.writefiles.prefix -width 20 -textvar ::Molefacture::prmprefix ] -row $rownum -column 2 
  #grid [button $w.writefiles.prmbrowse -text "Browse" -command ::Molefacture::findprmfile] -row $rownum -column 3 -columnspan 1 -sticky w
  incr rownum 

  grid [label $w.writefiles.dirlabel -text "Directory to write output:"] -row $rownum -column 0 -columnspan 2 -sticky w
  grid [entry $w.writefiles.dirloc -width 20 -textvar ::Molefacture::dirloc ] -row $rownum -column 2 
  grid [button $w.writefiles.dirrowse -text "Browse" -command ::Molefacture::findoutputdir] -row $rownum -column 3 -columnspan 1 -sticky w
  incr rownum 
  
  grid [button $w.writefiles.writefiles -text "Write output files" -command {::Molefacture::write_namdfiles [file join $::Molefacture::dirloc $::Molefacture::prmprefix] $::Molefacture::prmloc }] -row $rownum -column 2 -columnspan 1 -sticky e
  grid [button $w.writefiles.cancel -text "Cancel" -command {destroy .writenamdfilesgui}] -row $rownum -column 3 -columnspan 1 -sticky e
  pack $w.writefiles
}

proc ::Molefacture::findprmfile {} {
  set filename [tk_getOpenFile -title "Select force-field parameter file"]
#  puts $filename
  set ::Molefacture::prmloc "$filename"  
}

proc ::Molefacture::findoutputdir {} {
  set filename [tk_chooseDirectory -title "Choose an output directory" -initialdir "$::env(PWD)"]
#  puts $filename
  set ::Molefacture::dirloc "$filename"
}




