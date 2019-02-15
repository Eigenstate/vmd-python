##
## $Id: runante.tcl,v 1.22 2017/11/21 21:27:10 jribeiro Exp $
##
package require exectool 
package require runsqm 
package provide runante 0.2

namespace eval ::ANTECHAMBER:: {

  variable acpath ;# path to antechamber executable
  variable electypes [list resp cm1 esp gas wc bcc cm2 mul rc]
  variable molefmode 0 ;# are we running inside of molefacture?
  variable OPLSatomtypes "[file join $::env(MOLEFACTUREDIR) lib ATOMTYPE_OPLS.DEF]"
  variable achere 0
  variable acbin ""
  if { [vmdinfo arch] == "WIN32" || [vmdinfo arch] == "WIN64" } {
     set acbin "antechamber.exe"
  } else {
     set acbin "antechamber"
  }

 
}

proc ::ANTECHAMBER::acinit { {interactive 0} } {
   set amberhomehere 0
   foreach key [array names ::env] {
       if { $key == "AMBERHOME" && $::env($key) != "" && $::env($key) != " " } {
         set amberhomehere 1
       }
   }  
   if { $amberhomehere == 1 } {
      if { $interactive == 0 } {
        set  ::ANTECHAMBER::acpath [::ExecTool::find -description "antechamber"  -path [file join $::env(AMBERHOME) bin $::ANTECHAMBER::acbin ] $::ANTECHAMBER::acbin]
      } else {
        set  ::ANTECHAMBER::acpath [::ExecTool::find $interactive -description "antechamber"  -path [file join $::env(AMBERHOME) bin $::ANTECHAMBER::acbin ] $::ANTECHAMBER::acbin]
      }
      if { ![file exist $::env(AMBERHOME)] } {
          set paths [file split $::ANTECHAMBER::acpath]
          set ::env(AMBERHOME) [lindex $paths 0]
          for {set i 1 } { $i < [expr [llength $paths] - 2] } {incr i } {
            set ::env(AMBERHOME) [file join $::env(AMBERHOME) [lindex $paths $i]]
          }        
      }
   } else {
      if { $interactive == 0 } {
        set ::ANTECHAMBER::acpath [::ExecTool::find -description "antechamber" $::ANTECHAMBER::acbin]
      } else {
        set ::ANTECHAMBER::acpath [::ExecTool::find -interactive  -description "antechamber" $::ANTECHAMBER::acbin]
      }
      set paths [file split $::ANTECHAMBER::acpath]
      if { [llength $paths] > 0 } {
        set ::env(AMBERHOME) [lindex $paths 0] 
        for {set i 1 } { $i < [expr [llength $paths] - 2] } {incr i } {
            set ::env(AMBERHOME) [file join $::env(AMBERHOME) [lindex $paths $i]]
        }
      }
   }
#   puts "$::ANTECHAMBER::acpath"
#   puts "[catch { exec $::ANTECHAMBER::acpath & } res]"
   if { [string length $::ANTECHAMBER::acpath] > 0 && [file executable $::ANTECHAMBER::acpath] } {
        set ::ANTECHAMBER::achere 1
       #if {![string is integer -strict $res]} 
       # set ::ANTECHAMBER::acpath "echo \"AMBERHOME found but $::ANTECHAMBER::acpath not executable.\" #"
       # set ::ANTECHAMBER::achere 0
   } else {
      set ::ANTECHAMBER::acpath "echo \"\$antechamber not found\" #"
      set ::ANTECHAMBER::achere 0  
      set ::env(AMBERHOME) ""
  }
  return $::ANTECHAMBER::achere
}

proc ::ANTECHAMBER::run_ac_typing {selection {chargetype bcc} {totcharge 0.0} {atomtypes gaff} {spinmult 1} {resname MOL}} {
# Run a full antechamber typing run on the selection
# To do this, write a mol2 with the atoms and initial bond orders,
# and then call the antechamber command line executable

# The fully typed molecule is loaded as a new molecule in vmd, and
# the molid of this molecule is returned

# If atomtypes begins with the string CUSTOM, the CUSTOM will be stripped and
# the remainder of that string will be taken as a path to the type definition file

  variable acpath
  global env

  ::ANTECHAMBER::acinit

  if {$::ANTECHAMBER::achere == 0} {
    set errmsg "Couldn't find antechamber executable. Please install antechamber and provide the path to your antechamber installation."
    error "Couldn't find antechamber executable. Please install antechamber and provide the path to your antechamber installation."
    return $errmsg
  }

  # Print a banner giving credit where credit is due
  ::ANTECHAMBER::AntechamberBanner
  # Sanity check input
  # Make sure we have hydrogens present
  set hydsel [atomselect [$selection molid] "[$selection text] and hydrogen"]
  
  if {[$hydsel num] == 0} {
    puts "WARNING: You are running antechamber on a structure with no hydrogens"
    puts "  You should build a structure with all hydrogens prior to running antechamber"
    puts "  Disregard this message if your molecule has no hydrogen"
  }

  $hydsel delete

  # Make sure we're using a valid charging method
  variable electypes

  if {[lsearch $electypes $chargetype] < 0} {
    set errmsg"ERROR: Invalid charge method.\nValid choices are: [join $electypes]"
    error $errmsg
    return $errmsg
  }

  # see if we're using custom types
  set customtypes 0
  set customstring "CUSTOM"
  if {[string equal -length 6 $customstring $atomtypes]} {
    set customtypes 1
    set atomtypes [string range $atomtypes 6 end]
  }

  # ##Hack to make sure we write out bonds
  # $selection setbonds [$selection getbonds]
  # ##

# Write an input mol2 file
  $selection writemol2 antechamber-temp.mol2 

  if {$customtypes == 1} {
    set typestring "-d"
  } else {
    set typestring "-at"
  }
  
  set jtype 4
  if {$atomtypes == "none"} {
    set jtype 2
    set atomtypes "gaff"
  }

  # If we're keeping charges, write a charge file
  set delchargefile 0
  set chargestring ""
  if {$chargetype != "rc"} {
    set chargestring "-c" 
# $chargetype"
  } else {
    set chargetype ""
  }

# make sure we call divcon if needed
  set divconflag 0
  if {$chargetype == "cm1"} {
     set divconflag 1
     catch { exec divcon & } res
     if {![string is integer -strict $res]} {
       set divconflag 2
    }
   }

   #JS: I removed -c $chargetype so that charges are not asked for
   # - this way antechamber runs even if sqm or divcon are not available.
   # The idea is that
   # you can run atom typing, bond typing calculations and charge calculations
   # separately if you wish, or do them all in one go with a new button or command
    
  if {$divconflag == 1} { 
    puts "$acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 $typestring $atomtypes -j $jtype -df 1 -m $spinmult $chargestring $chargetype"
    exec  $acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 $typestring "$atomtypes" -j $jtype -df 1 -m $spinmult $chargestring $chargetype 
  } elseif { $divconflag == 2 } {
    puts "$acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 $typestring \"$atomtypes\" -j $jtype -m $spinmult $chargestring $chargetype"
    exec  $acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 $typestring  "$atomtypes"  -j $jtype -m $spinmult $chargestring $chargetype
 } else {
    set command exec
    lappend command $acpath
    lappend command -fi mol2 
    lappend command -i antechamber-temp.mol2
    lappend command -fo mol2 -o antechamber-temp.out.mol2
    lappend command -nc $totcharge -s 2
    if { $typestring == "-d" } {
        lappend command $typestring \"$atomtypes\" -j $jtype -m $spinmult
    } else { 
        lappend command $typestring $atomtypes -j $jtype -m $spinmult
    }
    if { $chargestring == "-c" } { 
        lappend command $chargestring $chargetype
    }
#    puts "$acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 $typestring $atomtypes -j $jtype -m $spinmult $chargestring $chargetype"
#     exec $acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 $typestring $atomtypes -j $jtype -m $spinmult $chargestring $chargetype 
    puts "$command"
    catch { eval $command } msg
    puts $msg
  }

# Load the output mol2 file
  set newmolid [mol new antechamber-temp.out.mol2]


# clean up
  file delete [glob antechamber-temp*]

  return $newmolid
}

proc ::ANTECHAMBER::AntechamberBanner {} {
  puts "************************************************************"
  puts "* Running antechamber from AmberTools                      *"
  puts "* Please read and cite:                                    *"
  puts "*   J. Wang et al., J. Mol. Graph. Model. 25:247-260 (2006)*"
  puts "************************************************************"
}

proc ::ANTECHAMBER::getAM1BCC {selection {chargetype rc} {totalcharge 0.0} {atomtypes gaff} {spinmult 1} {resname MOL}} {

   variable acpath
   global env
# Print a banner giving credit where credit is due
  ::ANTECHAMBER::AntechamberBanner
  puts "NOTE: am1bcc charges calculated are for GAFF assigned atom-"
  puts "      and bond-types. These are assigned prior to the am1bcc"
  puts "      calculation. Original atom and bond types are retained"
  puts "      in molefacture, however."    

  ::ANTECHAMBER::acinit

   if {$::ANTECHAMBER::achere == 0} {
     set errmsg "Couldn't find antechamber executable. Please install AmberTools and provide the path to your AmberTools installation."
     error $errmsg
     return $errmsg
   }
   ##Hack to make sure we write out bonds
   $selection setbonds [$selection getbonds]
   ##
   $selection writemol2 TEMP_antechamber.mol2 

   set command exec
   lappend command $acpath
   lappend command -fi mol2 
   lappend command -i TEMP_antechamber.mol2
   lappend command -fo ac -o TEMP_antechamber-pream1bcc.ac
   lappend command -nc $totalcharge -s 2
   lappend command -j 0
   puts $command
   catch { eval $command } msg
   puts $msg

   set command exec
   lappend command am1bcc -i TEMP_antechamber-pream1bcc.ac -o TEMP_antechamber-postam1bcc.ac -s 2 -j 4 -f ac
   puts $command 
   catch { eval $command } msg
   puts $msg 

   set command exec
   lappend command $acpath
   lappend command -fi ac 
   lappend command -i TEMP_antechamber-postam1bcc.ac
   lappend command -fo mol2 -o TEMP_antechamber-am1bcc.mol2
   lappend command -nc $totalcharge -s 2
   lappend command -j 0
   puts $command
   catch { eval $command } msg
   puts $msg

  set newmolid [mol new TEMP_antechamber-am1bcc.mol2]
  puts $newmolid
  set newsel [atomselect $newmolid all]

  # Store the old names for use in tracking down bonds
  # This would be much easier if we could assume that the input is an isolated
  # fragment, but we can't/shouldn't 

  # Set the trivial properties
  #puts [$selection get charge]
  #puts [$newsel get charge]

  if { [$newsel num] != [$selection num] } {
      set errmsg "ANTECHAMBER Error: Number of atoms in antechamber-temp.out.mol2 does not match antechamber-temp.mol2."
      append errmsg "\nThis happens when there are hydrogens missing and Antechamber adds them in. Please check your structure."
      mol delete $newmolid
      return $errmsg
   }
  $selection set charge [$newsel get charge]

  mol delete $newmolid
  return "0"
}

proc ::ANTECHAMBER::ac_type_in_place {selection {chargetype rc} {totalcharge 0.0} {atomtypes gaff} {spinmult 1} {resname MOL}} {
## Wrapper around run_ac_typing that will apply the atom types, charges, 
#  and bonding pattern from antechamber to the selection in the original molecule
# In this case, the newly created molecule is then deleted
  set newmolid [run_ac_typing $selection $chargetype $totalcharge $atomtypes $spinmult $resname]
  #puts $newmolid
  set newsel [atomselect $newmolid all]
  if { [$newsel num] != [$selection num] } {
      set errmsg "ANTECHAMBER Error: Number of atoms in antechamber-temp.out.mol2 does not match antechamber-temp.mol2."
      append errmsg "\nThis happens when there are hydrogens missing and Antechamber adds them in. Please check your structure."
      mol delete $newmolid
      return $errmsg
  }
  # Store the old names for use in tracking down bonds
  # This would be much easier if we could assume that the input is an isolated
  # fragment, but we can't/shouldn't 
  set oldnames [$selection get name]
  set oldids [$selection get index]
  set oldbonds [$selection getbonds]


  # Set the trivial properties
  $selection set charge [$newsel get charge]
  if {$atomtypes != "none"} {
      $selection set type [$newsel get type]
  }
  $selection set resname [$newsel get resname]
  #$selection set name [$newsel get name]


  ### now work out the bonds
  set newnames [$newsel get name]
  set newids [$newsel get index]

  array set equivinds {};# array of oldindex->newindex pairs

  foreach oldname $oldnames oldid $oldids {
    # Find the equivalent in the new molecule
    set equivind [lindex $newids [lsearch -exact $oldname $newnames] ]
    array set equivinds {$oldid $equivind}
  }

  set fixedbonds [list]
  foreach oldbond $oldbonds newbond [$newsel getbonds] oldbo [$selection getbondorders] newbo [$newsel getbondorders] oldid $oldids {

  # If we have the same number of bonds, assume the order matches up
    if { [llength $oldbond] == [llength $newbond] } {
      lappend fixedbonds $newbo
    } else {
      # otherwise some bonds go outside of the selection
      #  note that oldbonds must then be a superset of newbonds
      set smalllist [list]
      set j 0
      for {set i 0} {$i < [llength $oldbond]} {incr i} {
        set myind [lindex $oldbond $i]
        set eqind $equivinds($myind)
        if { [lindex $newbond $j] == $eqind } {
          lappend smalllist [lindex $newbo $j]
          incr j
        } else {
          lappend smalllist [lindex $oldbo $j]
        }
      }

      lappend fixedbonds $smalllist
    }

  }

  $selection setbondorders $fixedbonds

  mol delete $newmolid
  return "0"
}

proc ::ANTECHAMBER::init_gui {} {
  variable atomsel all
  variable totcharge 0.0
  variable spinmult 1
  variable resname MOL
  variable inplace 0
  variable ante_type gaff
  variable ante_qtype bcc
  variable outfile ""
}


proc ::ANTECHAMBER::antechamber_gui { {molefacturemode 0}} {
# Just a simple gui for running antechamber in place on a selection
# This should be callable from most other plugins

# if molefacturemode is nonzero, only atoms with occupancy > 0.5 are used

  variable w
  variable molefmode
  variable inplace
  variable ante_qtype
  variable ante_type
  set molefmode $molefacturemode
  set inplace 1

  if { [winfo exists .antechambergui] } {
    wm deiconify .antechambergui
    return
  }

  init_gui
  
  set w [toplevel ".antechambergui"]
  wm title $w "Antechamber"

  set rownum 0

  frame $w.settings

  grid [label $w.settings.sellabel -text "Selection:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.selection -width 30 -textvar ::ANTECHAMBER::atomsel]  -row $rownum -column 1 -columnspan 3 -sticky ew
  incr rownum

  grid [label $w.settings.chargelabel -text "Charge:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.charge -width 5 -textvar ::ANTECHAMBER::totcharge]  -row $rownum -column 1 -sticky ew
  grid [label $w.settings.multlabel -text "Multiplicity:"] -row $rownum -column 2 -sticky ew
  grid [entry $w.settings.mult -width 5 -textvar ::ANTECHAMBER::spinmult]  -row $rownum -column 3 -sticky ew
  incr rownum

  grid [label $w.settings.rnlabel -text "Resname:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.resname -width 6 -textvar ::ANTECHAMBER::resname]  -row $rownum -column 1 -sticky ew
  grid [label $w.settings.inplacelabel -text "Operate in place:"] -row $rownum -column 2 -sticky ew
  grid [checkbutton $w.settings.inplacebutton -variable ::ANTECHAMBER::inplace]  -row $rownum -column 3 -sticky ew
  incr rownum

  grid [label $w.settings.types -text "Atom types:"] -row $rownum -column 0 -sticky w
  grid [menubutton $w.settings.typemenu -menu $w.settings.typemenu.menu -textvar ::ANTECHAMBER::ante_type -relief raised]  -row $rownum -column 1 -columnspan 3 -sticky ew
  menu $w.settings.typemenu.menu -tearoff no
  $w.settings.typemenu.menu add radiobutton -label "GAFF" -variable ::ANTECHAMBER::ante_type -value "gaff"
  $w.settings.typemenu.menu add radiobutton -label "Amber" -variable ::ANTECHAMBER::ante_type -value "amber"
  $w.settings.typemenu.menu add radiobutton -label "BCC" -variable ::ANTECHAMBER::ante_type -value "bcc"
  $w.settings.typemenu.menu add radiobutton -label "Sybyl" -variable ::ANTECHAMBER::ante_type -value "sybyl"
  incr rownum

  grid [label $w.settings.charges -text "Atom charges:"] -row $rownum -column 0 -sticky w
  grid [menubutton $w.settings.chargemenu -menu $w.settings.chargemenu.menu -textvar ::ANTECHAMBER::ante_qtype -relief raised]  -row $rownum -column 1 -columnspan 3 -sticky ew
  menu $w.settings.chargemenu.menu -tearoff no
  $w.settings.chargemenu.menu add radiobutton -label "RESP" -variable ::ANTECHAMBER::ante_qtype -value "resp"
  $w.settings.chargemenu.menu add radiobutton -label "CM1" -variable ::ANTECHAMBER::ante_qtype -value "cm1"
  $w.settings.chargemenu.menu add radiobutton -label "ESP" -variable ::ANTECHAMBER::ante_qtype -value "esp"
  $w.settings.chargemenu.menu add radiobutton -label "Gasteiger" -variable ::ANTECHAMBER::ante_qtype -value "gas"
  $w.settings.chargemenu.menu add radiobutton -label "AM1-BCC" -variable ::ANTECHAMBER::ante_qtype -value "bcc"
  $w.settings.chargemenu.menu add radiobutton -label "CM2" -variable ::ANTECHAMBER::ante_qtype -value "cm2"
  $w.settings.chargemenu.menu add radiobutton -label "Mulliken" -variable ::ANTECHAMBER::ante_qtype -value "mul"
  $w.settings.chargemenu.menu add radiobutton -label "Keep current" -variable ::ANTECHAMBER::ante_qtype -value "rc"
  incr rownum

#  grid [label $w.settings.outflabel -text "Output file:"] -row $rownum -column 0 -sticky w
#  grid [entry $w.settings.outf -width 30 -textvar ::ANTECHAMBER::outfile]  #    -row $rownum -column 1 -columnspan 3 -sticky ew
#  incr rownum

  grid [button $w.settings.rotf -text "Run ANTECHAMBER" -command [namespace current]::run_ante_gui] -row $rownum -column 0 -columnspan 4

  pack $w.settings
}

proc ::ANTECHAMBER::run_ante_gui {} {
  variable atomsel
  variable totcharge
  variable spinmult
  variable resname
  variable inplace
  variable ante_type
  variable ante_qtype
  variable outfile
  variable molefmode

  set atomselold $atomsel
  if {$molefmode == 1} {
    set atomsel "$atomsel and occupancy >= 0.5"
    mol top $::Molefacture::tmpmolid
  }
  set mysel [atomselect top "$atomsel"]
  set atomsel $atomselold
  if { [$mysel num] > 0 } {
      if {$inplace == 1} {
        set result [[namespace current]::ac_type_in_place $mysel $ante_qtype $totcharge $ante_type $spinmult $resname]
      } else {
        set result [[namespace current]::run_ac_typing $mysel $ante_qtype $totcharge $ante_type $spinmult $resname]
      }
      if { $result != "0" } {
        tk_messageBox -message $result -type ok -title "Antechamber Error - Check Structure" -icon error
      }
  }
  $mysel delete

  if {$molefmode == 1} {
    ::Molefacture::update_openvalence
  }
}

proc ::ANTECHAMBER::noblanks {mylist} {
  set newlist [list]
  foreach elem $mylist {
    if {$elem != ""} {lappend newlist $elem}
  }

  return $newlist
}


## I didn't think to do it this way...
proc ::ANTECHAMBER::convert_sqmout_to_xyz {sqmfile xyzfile} {
  set instr [open $sqmfile r]
  set ostr [open $xyzfile w]


  set atomlist [list]
  while {[gets $instr line] >= 0} {
    if {[string first "Final Structure" $line] >= 0} {
      gets $instr line
      gets $instr line
      gets $instr line
      gets $instr line
      while {[string first "QMMM" $line] >= 0} {
        lappend atomlist $line
        gets $instr line
      }

    }
  }

  close $instr

  set natom [llength $atomlist]

  puts $ostr $natom
  puts $ostr " From antechamber"
  foreach line $atomlist {
    set linearr [split $line]
    set linearr [noblanks $linearr]
    puts $ostr " [lindex $line 3] [lindex $line 4] [lindex $line 5] [lindex $line 6]"
  }

  close $ostr
}


    


### This is currently not available from the gui - maybe add as 
### an 'advanced' feature if people want to do their own atomtyping
proc ::ANTECHAMBER::check_all_indir {workdir refdir type} {
 variable acpath
    ::ANTECHAMBER::acinit

   if {$::ANTECHAMBER::achere == 0} {
    error "Couldn't find antechamber executable. Please install antechamber and provide the path to your antechamber installation."
  }
  set fnames {}
  foreach n [glob "$refdir/*.mol2"] {
      lappend fnames "[file tail $n]"
  }
  if {$type == ""} {
    return "Error: Atom Type not specified"
  }
  puts "Testing for atomtype $type"
  set curdir [pwd]
  cd $workdir 
  set num [expr [llength $fnames]]
  set reporttext ""
  puts "$fnames\n"
  for {set n 0} {$n < $num} {incr n} {
    set name [lindex $fnames $n]
    set testname [file join $workdir $name]
    set refname [file join $refdir $name] 
    puts "Testing: $refname vs $testname"
    set ref_mol [mol new "$refname"]
    if { [catch { exec $::ANTECHAMBER::acpath -fi mol2 -i "$refname" -fo mol2 -o "$testname" -s 2 -d "$ANTECHAMBER::OPLSatomtypes" -j 4 2 > "molefac_$n.log" } msg] } {
     }
   if { [file exists $testname] == 0 } {
	puts "ERROR: processing file $refname with antechamber"
   } else {

    append reporttext "Testing: $name\n"
    set test_mol [mol new "$testname"]
    set test_sel [atomselect $test_mol all]
    set ref_sel [atomselect $ref_mol all]
    set numatms1 [$test_sel num]
    set numatms2 [$ref_sel num]
    if {$numatms1 != $numatms2} {
      append reporttext "Something is wrong: Atomcounts for $testname and $refname are different.\n"
    } else {
      for {set i 0} {$i < $numatms1} {incr i} {
        set at [lindex [$test_sel get type] $i]
        set ar [lindex [$ref_sel get type] $i]
        if {$at == $type && $ar != $type} {
          append reporttext "MISMATCH: Atom $i has been assigned type $at but should be $ar.\n"
        } elseif {$ar == $type && $at != $type} {
          append reporttext "MISMATCH: Atom $i has been assigned type $at but should be $ar.\n"
        }
      }
    }
    $test_sel delete
    $ref_sel delete
    mol delete $test_mol
    mol delete $ref_mol
   }
  }
  cd $curdir
  return $reporttext
}


proc  ::ANTECHAMBER::set_oplsdef_loc { loc } {
  set ::ANTECHAMBER::OPLSatomtypes $loc
}





