##
## $Id: molefacture_edit.tcl,v 1.15 2012/03/26 17:56:25 johanstr Exp $
##

proc ::Molefacture::raise_bondorder {picklist} {
   variable tmpmolid
#   variable bondlist
#   variable anglelist

   #puts "Picklist: $picklist"
   set all [atomselect $tmpmolid "all"]
   set bondorders [$all getbondorders]
   variable atomlist
   set sb0 [lindex [$all getbonds] [lindex $picklist 0]]
   set sb1 [lindex [$all getbonds] [lindex $picklist 1]]
   set vmdind0 [lindex $atomlist [lsearch $atomlist "[format "%5s" [lindex $picklist 0]] *"] 0]
   set vmdind1 [lindex $atomlist [lsearch $atomlist "[format "%5s" [lindex $picklist 1]] *"] 0]

   set pos0in1 [lsearch $sb1 $vmdind0]
   if {$pos0in1 == -1} {
#      lappend $sb1 $vmdind0
#      lappend $sb0 $vmdind1
      vmd_addbond $vmdind0 $vmdind1 0
      set sb0 [lindex [$all getbonds] [lindex $picklist 0]]
      set sb1 [lindex [$all getbonds] [lindex $picklist 1]]
      set bondorders [$all getbondorders]
      set pos0in1 [lsearch $sb1 $vmdind0]
   }
   set pos1in0 [lsearch $sb0 $vmdind1]
   set currorder [lindex $bondorders $vmdind0 $pos1in0]
   if {$currorder < 0} {set currorder 1}
   set currorder [expr $currorder + 1]
#puts "$vmdind0 $pos1in0 $currorder"
   lset bondorders $vmdind0 $pos1in0 $currorder
   lset bondorders $vmdind1 $pos0in1 $currorder

   $all setbondorders $bondorders
   update_openvalence

   # Update the displayed bondlist and anglelist
   update_bondlist
#   set bondlist [::Molefacture::bondlist]
#   set anglelist [::Molefacture::anglelist]
}

proc ::Molefacture::lower_bondorder {picklist } {
   variable tmpmolid
#   variable bondlist
#   variable anglelist


   set all [atomselect $tmpmolid "all"]
   set bondorders [$all getbondorders]
   variable atomlist
   set sb0 [lindex [$all getbonds] [lindex $picklist 0]]
   set sb1 [lindex [$all getbonds] [lindex $picklist 1]]
   set vmdind0 [lindex $atomlist [lsearch $atomlist "[format "%5s" [lindex $picklist 0]] *"] 0]
   set vmdind1 [lindex $atomlist [lsearch $atomlist "[format "%5s" [lindex $picklist 1]] *"] 0]
   set pos0in1 [lsearch $sb1 $vmdind0]
   if {$pos0in1 == -1} {
    return 
   }
   set pos1in0 [lsearch $sb0 $vmdind1]
   set currorder [lindex $bondorders $vmdind0 $pos1in0]
   if {$currorder < 0} {set currorder 1}
   set currorder [expr $currorder - 1]
   if {$currorder < 1} {
     vmd_delbond $vmdind0 $vmdind1 
     vmd_delbond $vmdind1 $vmdind0
     return
   }
   lset bondorders $vmdind0 $pos1in0 $currorder
   lset bondorders $vmdind1 $pos0in1 $currorder

   $all setbondorders $bondorders
   update_bondlist
   update_openvalence

   # Update the displayed bondlist
   #set bondlist [::Molefacture::bondlist]
   #set anglelist [::Molefacture::anglelist]
}

proc ::Molefacture::del_atom {delindex {updateopenv 1}} {
  variable tmpmolid
  variable openvalencelist
  variable picklist
  set mother -1
  variable picklist
#  variable cursel

#  foreach delindex $picklist {
    set currdel [atomselect $tmpmolid "index $delindex"]
    $currdel set occupancy 0.0
    foreach bondedatom [lindex [$currdel getbonds] 0] {
      vmd_delbond $bondedatom [$currdel get index] 
      vmd_delbond [$currdel get index] $bondedatom
      set mother $bondedatom
    }
     set curcoor [join [$currdel get {x y z}]]
     $currdel delete
#  }
   

  if {$updateopenv == 1} {
    update_openvalence
    variable bondlist [bondlist]
    variable anglelist [anglelist]
  }
  return [list $mother $curcoor]
}

proc ::Molefacture::update_edit_atom_elem_tr {name1 name2 op} {
# When we change the element of an atom, automatically update its name and type
  variable editatom_name
  variable editatom_type
  variable editatom_element

  #TEMPORARILY DISABLED - ADD SETTING TO RE-ENABLE BY DEFAULT
  #set editatom_name $editatom_element
  #set editatom_type $editatom_element
}

#Procs to edit an atom's properties
proc ::Molefacture::edit_atom {} {
  variable editatom_name 
  variable editatom_type 
  variable editatom_element 
  variable editatom_charge 
  variable editatom_index 
  variable tmpmolid
  variable picklist
  variable periodic

  #Make the appropriate changes to the selected atom
  set tmpsel [atomselect $tmpmolid "index [lindex $picklist 0]"]
  puts "|$editatom_name|"
  $tmpsel set name "\"$editatom_name\""
  $tmpsel set type "\"$editatom_type\""
  $tmpsel set charge $editatom_charge
  $tmpsel set element "\"$editatom_element\""
  $tmpsel delete

  #Update the menus and such
  update_openvalence
}


proc ::Molefacture::edit_incoming_atoms_FEP {} {
  variable editatom_beta
  variable tmpmolid
  variable picklist
  if {[llength $picklist] < 1} {
    return
  }
  set tmpsel [atomselect $tmpmolid "index $picklist"]
  $tmpsel set beta 1
  $tmpsel delete
  update_openvalence_FEP
}

proc ::Molefacture::edit_outgoing_atoms_FEP {} {
  variable editatom_beta
  variable tmpmolid
  variable picklist
  if {[llength $picklist] < 1} {
    return
  }
  set tmpsel [atomselect $tmpmolid "index $picklist"]
  $tmpsel set beta -1
  $tmpsel delete
  update_openvalence_FEP
}

proc ::Molefacture::edit_clear_atoms_FEP {} {
  variable editatom_beta
  variable tmpmolid
  variable picklist
  if {[llength $picklist] < 1} {
    return
  }
  set tmpsel [atomselect $tmpmolid "index $picklist"]
  $tmpsel set beta 0
  $tmpsel delete
  update_openvalence_FEP
}


proc ::Molefacture::raise_ox {mindex} {
        variable valence
        variable oxidation
        variable tmpmolid
        variable periodic
        set mother [atomselect $tmpmolid "index $mindex"]
        set element  [lindex $periodic [$mother get atomicnumber]]
        set oxlist [array get valence $element]
        set curox [lindex $oxidation $mindex]
        set nox [llength [lindex $oxlist 1]]
        incr curox 1
        #puts "oxlist: $oxlist"
        #puts "$curox $nox"
        if {$curox >= [expr $nox]} {
          tk_messageBox -icon error -type ok -title "Max Oxidation Reached" -message "Atom $mindex is already at its maximum oxidation state"
          return
        }
        set oxidation [lreplace $oxidation $mindex $mindex $curox]
}
        

proc ::Molefacture::lower_ox {mindex} {
        variable valence
        variable oxidation
        variable tmpmolid
        variable periodic
        set mother [atomselect $tmpmolid "index $mindex"]
        set element  [lindex $periodic [$mother get atomicnumber]]
        set oxlist [array get valence $element]
        set curox [lindex $oxidation $mindex]
        set nox [llength [lindex $oxlist 1]]
        incr curox -1
        #puts "oxlist: $oxlist"
        #puts "$curox $nox"
        if {$curox < 0} {
          tk_messageBox -icon error -type ok -title "Minimum Oxidation Reached" -message "Atom $mindex is already at its minimum oxidation state"
          return
        }
        set oxidation [lreplace $oxidation $mindex $mindex $curox]
}

proc ::Molefacture::unique_atomnames {} {
  variable tmpmolid

  set worksel [atomselect $tmpmolid "occupancy >= 0.5"]
  set workelems [$worksel get element]
  set worknames [$worksel get name]
  array set elemcounts [list]

  foreach elem [lsort -unique $workelems] {
    array set elemcounts [list $elem 0]
  }

  array set namecounts [list]
  foreach name [lsort -unique $worknames] {
    array set namecounts [list $name 0]
  }

  set newnames [list]
  set namechanges ""
  foreach ind [$worksel get index] elem [$worksel get element] name [$worksel get name] {
    set elemcount [array get elemcounts $elem]
    set namecount [array get namecounts $name]
    set nameval [lindex $namecount 1]
    incr nameval
    set val [lindex $elemcount 1]
    incr val
    if { $nameval > 1 } {
        set tmpname "${name}${val}"
        if { [string length $tmpname"] > 4 } {
            set tmpname "${elem}${val}"
#            lappend newnames "${elem}${val}"
        } 
        lappend newnames $tmpname
            
        append namechanges "    $ind: $name -> ${tmpname}\n"
    } else {
        lappend newnames "${name}"
    }
    array set elemcounts [list $elem $val]
    array set namecounts [list $name $nameval]
  }

  #puts $newnames
  set proceed "yes"
  if { $namechanges != "" } {
    set proceed [tk_messageBox -type yesno -title "Atom names change" -icon warning \
        -message "Not all the atom names were unique. To fix this the\nfollowing atom names have to be changed in the top file:\n${namechanges}\nDo you wish to proceed?"]
    if {$proceed == "yes"} {
        $worksel set name $newnames
    }
  }
  $worksel delete

  update_openvalence
  return $proceed
}

