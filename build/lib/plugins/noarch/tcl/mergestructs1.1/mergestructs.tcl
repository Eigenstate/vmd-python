##
## Merge Structures 1.0
##
##
## Author: Robert Brunner
##         rbrunner@uiuc.edu
##
## $Id: mergestructs.tcl,v 1.6 2011/03/09 22:16:14 johns Exp $
##

## Tell Tcl that we're a package and any dependencies we may have
package require psfgen
package provide mergestructs 1.1

namespace eval ::MergeStructs:: {
  namespace export mergestructs

  # window handles
  variable w   ;# handle to main window
  variable guiState ;# Array holding GUI state
  array set guiState {
    currentMol "none"
  }

}

#
# Create the window and initialize data structures
#
proc ::MergeStructs::mergestructs {} {
  variable guiState
  variable w
  
  puts "MergeStructs)Merging molecules"
  set ns [namespace current]

#  foreach child [winfo children $w] {
#    if { ![string equal "$child" "${w}.menubar"] } {
#      destroy $child
#    }
#  }

  # If already initialized and no new file requested, just turn on
  if { [winfo exists .mergestructs] } {
    wm deiconify $w
    return
  }

  set w .mergestructs
  catch {destroy $w} 
  toplevel $w
  wm title $w "Merge Structures"
  wm resizable $w 0 0
  wm protocol $w WM_DELETE_WINDOW "menu mergestructs off"

  guiDrawMolFileFrame $ns $w.body "Molecule 1" "psffile1" "pdbfile1"
  guiDrawMolFileFrame $ns $w.body2 "Molecule 2" "psffile2" "pdbfile2"

  frame $w.body3
  set row 0
  grid columnconfigure $w.body3 1 -weight 1
  
  grid [label $w.body3.mergedfilelabel -text "Merged file (.pdb,.psf):"] \
    -row $row -column 0 -sticky w
  grid [entry $w.body3.mergedfile -width 5 -textvariable \
    ${ns}::guiState(mergedFile)] \
    -row $row -column 1 -sticky ew
  incr row

  frame $w.buttons
  set row 0
  grid [button $w.buttons.cancel -text "Cancel" \
          -command "wm withdraw $w" ] -row $row -column 0
  grid [button $w.buttons.help -text "Help" \
          -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/mergestructs" ] -row $row -column 3
  grid [button $w.buttons.doit -text "Merge" \
          -command "${ns}::guiMergeMolecules" ] \
    -row $row -column 5
    
  pack $w.body -anchor nw -fill x
  pack $w.body2 -anchor nw -fill x
  pack $w.body3 -anchor nw -fill x
  pack $w.buttons -anchor nw -fill x
}

proc ::MergeStructs::guiDrawMolFileFrame { ns win label psfkey pdbkey } {
  frame $win
  set row 0
  grid columnconfigure $win { 0 2 } -weight 0
  grid columnconfigure $win 1 -weight 1

  grid [label $win.label -text $label] \
    -row $row -column 0 -columnspan 3 -sticky w
  incr row
    
  grid [label $win.psflabel -text "PSF: "] \
    -row $row -column 0 -sticky w
  grid [entry $win.psfpath -width 30 \
        -textvariable ${ns}::guiState($psfkey)] \
    -row $row -column 1 -sticky ew
  grid [button $win.psfbutton -text "Browse" \
         -command "set tempfile \[tk_getOpenFile -defaultextension .psf \]; \
                   if \{!\[string equal \$tempfile \"\"\]\} \{ \
                     set ${ns}::guiState($psfkey) \$tempfile; \
                   \};" \
        ] -row $row -column 2 -sticky e
  incr row
  
  grid [label $win.pdblabel -text "PDB: "] \
    -row $row -column 0 -sticky w
  grid [entry $win.pdbpath -width 30 \
          -textvariable ${ns}::guiState($pdbkey)] \
    -row $row -column 1 -sticky ew
  grid [button $win.pdbbutton -text "Browse" \
         -command "set tempfile \[tk_getOpenFile -defaultextension .pdb \]; \
                   if \{!\[string equal \$tempfile \"\"\]\} \{ \
                     set ${ns}::guiState($pdbkey) \$tempfile \
                   \};" \
        ] -row $row -column 2 -sticky e
  incr row
  
  grid [button $win.selloaded -text "Select loaded molecule" \
    -command "${ns}::guiSelectLoadedMolWin $psfkey $pdbkey" ] \
    -row $row -column 0 -columnspan 3
  incr row
}

proc ::MergeStructs::guiSelectLoadedMolWin { psffile pdbfile \
                                             { fileflag "-all" } } {
  variable guiState
  
  if { [winfo exists .ibseelctmol] } {
    destroy .ibselectmol
  }
  set aw [toplevel ".ibselectmol"]
  wm title $aw "Select molecule"
  wm resizable $aw yes yes
  grab set ".ibselectmol"
  set ns [namespace current]

  frame $aw.type
  set row 0
  grid [label $aw.type.label -text "Molecule:" ] \
    -row $row -column 0 -sticky w
  grid [menubutton $aw.type.menub \
    -menu $aw.type.menub.menu -relief raised -width 20 ] \
    -row $row -column 1 -columnspan 5 -sticky ew -ipady 2

  set guiState(molMenuName) [menu $aw.type.menub.menu -tearoff no]
  guiFillMolMenu $fileflag
  set def_label [$aw.type.menub.menu entrycget 0 -label]
  #puts "InorganicBuilder)Label is $def_label"
  #$aw.type.menub configure -text "$def_label"
  incr row 2
  
  frame $aw.buttons
  set row 0
  grid [button $aw.buttons.cancel -text Cancel -command "destroy $aw"] \
    -row $row -column 0
  grid [button $aw.buttons.add -text Select \
    -command "${ns}::guiStoreMol $psffile $pdbfile; destroy $aw"] \
    -row $row -column 1
  
  guiRepackSelectMol

}

proc ::MergeStructs::guiRepackSelectMol { } {
  set aw ".ibselectmol"
  grid $aw.type -row 0
  grid $aw.buttons -row 2
#  puts "InorganicBuilder)Repacking select mol"
}

proc ::MergeStructs::guiFillMolMenu { filetype } {
  if {[string equal $filetype "-psf"] } {
    return [guiFillMolMenuInt "molMenuName" "currentMol" -psf ]
  } else {
    return [guiFillMolMenuInt "molMenuName" "currentMol" -all ]
  }
}

proc ::MergeStructs::guiFillMolMenuInt { molMenuName currentMol \
                                             filetypes } {
  #Proc to get all the current molecules for a menu
  #For now, shamelessly ripped off from the NAMDEnergy plugin
  #which in turn ripped it off from the PME plugin
  variable guiState
  
#  puts "InorganicBuilder)Processing $molMenuName $guiState($molMenuName)"
  set name $guiState($molMenuName)
  if { ![winfo exists $name] } {
    return
  }
#  puts "InorganicBuilder)name parent is [winfo parent $name]"
  
  if { [$name index end] != 0 } {
    $name delete 0 end
  }

  set molList ""
#  puts "InorganicBuilder)Processing $molMenuName"
  foreach mm [molinfo list] {
    if { [molinfo $mm get numatoms] > 0 } {
      # if we're building the PSF molecule menu, and the molecule doesn't
      # contain a PSF file, don't include it in the list
      if { [string equal $filetypes "-psf" ] } {
#        puts "InorganicBuilder)Filling PSF menu"
        set filetypes [lindex [ molinfo $mm get filetype ] 0]
#        puts "InorganicBuilder)$mm has $filetypes"
        if { [ lsearch $filetypes "psf" ] == -1} {
          continue
        }
      }
      lappend molList $mm
      $name add command \
        -command "[winfo parent $name] configure \
                  -text \"$mm [ lindex [molinfo $mm get name] 0 ]\"; \
                  set [namespace current]::guiState($currentMol) $mm" \
        -label "$mm [molinfo $mm get name]"
    }
  }
  #set if any non-Graphics molecule is loaded
  if {[lsearch -exact $molList $guiState($currentMol)] == -1} {
    if {[lsearch -exact $molList [molinfo top]] != -1} {
      set guiState($currentMol) [molinfo top]
      set usableMolLoaded 1
    } else {
      set guiState($currentMol) "none"
      set usableMolLoaded  0
    }
  }
#  puts "InorganicBuilder)$molMenuName:molList is $molList [llength $molList]"
  if {[llength $molList] == 0} {
    $name add command \
      -command "set [namespace current]::guiState($currentMol) none; \
        [winfo parent $name] configure -text \"None loaded\";" \
      -label "None loaded"
#    puts "InorganicBuilder)Configuring [winfo parent $name]"
    [winfo parent $name] configure -text "None loaded"
  }
  
  $name invoke 0
  
#  puts "InorganicBuilder)Done processing $molMenuName"
}

proc ::MergeStructs::guiStoreMol { psffile pdbfile } {
  variable guiState
#  puts "InorganicBuilder)Storing molecule"
  set mymol $guiState(currentMol)
  set filetypes [lindex [molinfo $mymol get filetype] 0]
  set filenames [lindex [molinfo $mymol get filename] 0]
  
  set indx [lsearch $filetypes "psf"]
  if { $indx != -1 } {
    set guiState($psffile) [lindex $filenames $indx]
  } else {
    set guiState($psffile) ""
  }
  set indx [lsearch $filetypes "pdb"]
  if { $indx != -1 } {
    set guiState($pdbfile) [lindex $filenames $indx]
  } else {
    set guiState($pdbfile) ""
  }
}

proc ::MergeStructs::guiMergeMolecules {} {
  variable guiState
  set ns [namespace current]
  
  set dupsegs [detectSegmentConflicts $guiState(psffile1) $guiState(pdbfile1) \
                 $guiState(psffile2) $guiState(pdbfile2) ]
  set ndups [llength dupsegs]
  if { $ndups == 0} {
    return [mergeMolecules [list "$guiState(psffile1)" \
                                 "$guiState(pdbfile1)" ] \
                         [list "$guiState(psffile2)" \
                                "$guiState(pdbfile2)" ] \
                         $guiState(mergedFile)]
  }
  
  set guiState(mergef1)  [file tail [file rootname $guiState(psffile1)]]
  set guiState(mergef2)  [file tail [file rootname $guiState(psffile2)]]
  set guiState(mergedir) "mergetmp[clock clicks]"
  file mkdir $guiState(mergedir)
  
  set guiState(mergef1src) "$guiState(mergedir)/$guiState(mergef1).1s"
  set guiState(mergef1dst) "$guiState(mergedir)/$guiState(mergef1).1d"
  set guiState(mergef2src) "$guiState(mergedir)/$guiState(mergef2).2s"
  set guiState(mergef2dst) "$guiState(mergedir)/$guiState(mergef2).2d"
  file copy -force $guiState(psffile1) "$guiState(mergef1src).psf"
  file copy -force $guiState(pdbfile1) "$guiState(mergef1src).pdb"
  file copy -force $guiState(psffile2) "$guiState(mergef2src).psf"
  file copy -force $guiState(pdbfile2) "$guiState(mergef2src).pdb"

  after idle "${ns}::guiMergeDetectConflicts"
}

proc ::MergeStructs::guiMergeDetectConflicts {} {
  variable guiState
  set ns [namespace current]

  set dupsegs [detectSegmentConflicts \
                 "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb" \
                 "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb" ]
                 
  set ndups [llength $dupsegs]
  
  if { $ndups == 0 } {
    after idle "${ns}::guiMergeConflictsResolved"
    return
  }

  if { [winfo exists .fixconflicts] } {
    destroy .fixconflicts
  }
  set dwin [toplevel .fixconflicts]
  wm title $dwin "Fix Segment Conflicts"
  puts "dupsegs -$dupsegs-"

  frame $dwin.msgframe
  grid columnconfigure $dwin.msgframe 0 -weight 1
  if { $ndups == 1 } {
    set msgtxt "is 1 duplicated segment name"
  } else {
    set msgtxt "are $ndups duplicated segment names"
  }
  grid [message $dwin.msgframe.msg -text \
    "There $msgtxt in the two \
     files. Each file must contain a unique set of segment names. Rename the \
     segments here." -aspect 300 ] -row 0 -column 0 -sticky w
  set guiState(dupsegs) $dupsegs

  frame $dwin.options
  set row 0
  
  grid [radiobutton $dwin.options.addprefix \
          -variable ${ns}::guiState(mergeHow) -value addprefix \
          -text "Add prefix to conflicting segment names" -anchor w] \
    -row $row -column 0 -sticky w
  incr row
  
  grid [radiobutton $dwin.options.addsuffix \
          -variable ${ns}::guiState(mergeHow) -value addsuffix \
          -text "Add suffix to conflicting segment name" -anchor w] \
    -row $row -column 0 -sticky w
  incr row

  grid [radiobutton $dwin.options.replfirst \
          -variable ${ns}::guiState(mergeHow) -value replfirst \
          -text "Replace first character in conflicting segment names" \
          -anchor w] \
    -row $row -column 0 -sticky w
  incr row

  grid [radiobutton $dwin.options.specify \
          -variable ${ns}::guiState(mergeHow) -value specify \
          -text "Specify new segment names" \
          -anchor w] \
    -row $row -column 0 -sticky w
  incr row

  frame $dwin.buttons
  set row 0
  grid [button $dwin.buttons.cancel -text "Cancel" \
          -command "destroy $dwin" ] -row $row -column 0
  grid [button $dwin.buttons.submit -text "Apply segment fixes" \
      -command "destroy $dwin; ${ns}::guiMergeChooseFixMode" ] \
      -row $row -column 1
  pack $dwin.msgframe -anchor nw -fill x -expand 1
  pack $dwin.options -anchor nw -fill x -expand 1
  pack $dwin.buttons -anchor nw -fill x -expand 1
}
 
 proc ::MergeStructs::guiMergeChooseFixMode {} {
  variable guiState
  set ns [namespace current]
  
  if { [ string equal $guiState(mergeHow) "addprefix" ] } {
    guiMergeAddPrefixSuffix prefix
  } elseif { [ string equal $guiState(mergeHow) "addsuffix" ] } {
    guiMergeAddPrefixSuffix suffix
  } elseif { [ string equal $guiState(mergeHow) "replfirst" ] } {
    guiMergeReplaceChar
  } elseif { [ string equal $guiState(mergeHow) "specify" ] } {
    guiMergeFixSpecificConflicts
  } else {
    puts "ERRROROOROR!"
  }
}

proc ::MergeStructs::guiMergeAddPrefixSuffix { type } {
  variable guiState
  set ns [namespace current]

  set dupsegs [detectSegmentConflicts \
                 "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb" \
                 "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb" ]
                 
  set ndups [llength $dupsegs]
  set guiState(dupsegs) $dupsegs
  
  if { $ndups == 0 } {
    after idle "${ns}::guiMergeConflictsResolved"
    return
  }

  if { [winfo exists .fixconflicts] } {
    destroy .fixconflicts
  }
  set dwin [toplevel .fixconflicts]
  wm title $dwin "Fix Segment Conflicts"

  frame $dwin.msgframe
  grid columnconfigure $dwin.msgframe 0 -weight 1
  if { $ndups == 1 } {
    set msgtxt "is 1 duplicated segment name"
  } else {
    set msgtxt "are $ndups duplicated segment names"
  }
  grid [message $dwin.msgframe.msg -text \
    "There $msgtxt in the two \
     files. Each file must contain a unique set of segment names. Rename the \
     segments here." -aspect 300 ] -row 0 -column 0 -sticky w
  
  set toolong 0
  foreach dup $dupsegs {
    if { [ string length $dup ] > 3 } {
      set toolong 1
      break
    }
  }
  
  if { $toolong } {
    grid [message $dwin.msgframe.msg -text \
      "At least one segment name is already 4 characters, so there is no \
       space to add the $type. That segment will not be changed, but you \
       can change that specific segment name to fix that problem separately." \
      -aspect 300 ] -row $row -column 0 -sticky w
    incr row
  }
  
  frame $dwin.segchange
  grid columnconfigure $dwin.msgframe 0 -weight 1
  set row 0
  
  grid [label $dwin.segchange.title -text "Specify one-character $type, or blank for no change to that molecule"] \
    -row $row -column 0 -columnspan 2 -sticky w
  incr row

  set guiState(mergem1prefix) ""  
  grid [label $dwin.segchange.m1label -text "Molecule 1:"] \
    -row $row -column 0  -sticky w
  grid [entry $dwin.segchange.m1val -width 1 \
          -textvariable ${ns}::guiState(mergem1prefix)] \
      -row $row -column 1 -sticky ew -padx 4
  incr row
  
  set guiState(mergem2prefix) ""  
  grid [label $dwin.segchange.m2label -text "Molecule 2:"] \
    -row $row -column 0  -sticky w
  grid [entry $dwin.segchange.m2val -width 1 \
          -textvariable ${ns}::guiState(mergem2prefix)] \
      -row $row -column 1 -sticky ew -padx 4
  incr row

  frame $dwin.buttons
  set row 0
  grid [button $dwin.buttons.cancel -text "Cancel" \
          -command "destroy $dwin" ] -row $row -column 0
  grid [button $dwin.buttons.submit -text "Apply segment fixes" \
      -command "destroy $dwin; ${ns}::guiMergeApplyPrefix $type" ] \
      -row $row -column 1
  pack $dwin.msgframe -anchor nw -fill x -expand 1
  pack $dwin.segchange -anchor nw -fill x -expand 1
  pack $dwin.buttons -anchor nw -fill x -expand 1
  
  return
}

proc ::MergeStructs::guiMergeReplaceChar { } {
  variable guiState
  set ns [namespace current]

  set dupsegs [detectSegmentConflicts \
                 "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb" \
                 "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb" ]
                 
  set ndups [llength $dupsegs]
  set guiState(dupsegs) $dupsegs
  
  if { $ndups == 0 } {
    after idle "${ns}::guiMergeConflictsResolved"
    return
  }

  if { [winfo exists .fixconflicts] } {
    destroy .fixconflicts
  }
  set dwin [toplevel .fixconflicts]
  wm title $dwin "Fix Segment Conflicts"

  frame $dwin.msgframe
  grid columnconfigure $dwin.msgframe 0 -weight 1
  if { $ndups == 1 } {
    set msgtxt "is 1 duplicated segment name"
  } else {
    set msgtxt "are $ndups duplicated segment names"
  }
  grid [message $dwin.msgframe.msg -text \
    "There $msgtxt in the two \
     files. Each file must contain a unique set of segment names. Rename the \
     segments here." -aspect 300 ] -row 0 -column 0 -sticky w
  
  frame $dwin.segchange
  grid columnconfigure $dwin.msgframe 0 -weight 1
  set row 0
  
  grid [label $dwin.segchange.title -text "Specify a character to replace \
           the first character in conflicting segment name(s), \
           or blank for no change to that molecule"] \
    -row $row -column 0 -columnspan 2 -sticky w
  incr row

  set guiState(mergem1prefix) ""  
  grid [label $dwin.segchange.m1label -text "Molecule 1:"] \
    -row $row -column 0  -sticky w
  grid [entry $dwin.segchange.m1val -width 1 \
          -textvariable ${ns}::guiState(mergem1prefix)] \
      -row $row -column 1 -sticky ew -padx 4
  incr row
  
  set guiState(mergem2prefix) ""  
  grid [label $dwin.segchange.m2label -text "Molecule 2:"] \
    -row $row -column 0  -sticky w
  grid [entry $dwin.segchange.m2val -width 1 \
          -textvariable ${ns}::guiState(mergem2prefix)] \
      -row $row -column 1 -sticky ew -padx 4
  incr row

  frame $dwin.buttons
  set row 0
  grid [button $dwin.buttons.cancel -text "Cancel" \
          -command "destroy $dwin" ] -row $row -column 0
  grid [button $dwin.buttons.submit -text "Apply segment fixes" \
      -command "destroy $dwin; ${ns}::guiMergeApplyReplaceChar" ] \
      -row $row -column 1
  pack $dwin.msgframe -anchor nw -fill x -expand 1
  pack $dwin.segchange -anchor nw -fill x -expand 1
  pack $dwin.buttons -anchor nw -fill x -expand 1
  
  return
}

proc ::MergeStructs::guiMergeFixSpecificConflicts {} {
  variable guiState
  set ns [namespace current]

  set dupsegs [detectSegmentConflicts \
                 "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb" \
                 "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb" ]
                 
  set ndups [llength $dupsegs]
  
  if { $ndups == 0 } {
    after idle "${ns}::guiMergeConflictsResolved"
    return
  }

  if { [winfo exists .fixconflicts] } {
    destroy .fixconflicts
  }
  set dwin [toplevel .fixconflicts]
  wm title $dwin "Fix Segment Conflicts"
  puts "dupsegs -$dupsegs-"

  frame $dwin.msgframe
  grid columnconfigure $dwin.msgframe 0 -weight 1
  if { $ndups == 1 } {
    set msgtxt "is 1 duplicated segment name"
  } else {
    set msgtxt "are $ndups duplicated segment names"
  }
  grid [message $dwin.msgframe.msg -text \
    "There $msgtxt in the two \
     files. Each file must contain a unique set of segment names. Rename the \
     segments here." -aspect 300 ] -row 0 -column 0 -sticky w
  set guiState(dupsegs) $dupsegs
  
  frame $dwin.dups
  grid columnconfigure $dwin.dups 0 -weight 1
  set row 0
  grid [label $dwin.dups.head1 -text "Mol 1"] -row $row -column 1 -sticky w
  grid [label $dwin.dups.head2 -text "Mol 2"] -row $row -column 2 -sticky w
  incr row
  
  set i 0
  foreach dup $dupsegs {
    set guiState(l1val$i) $dup
    set guiState(l2val$i) $dup
    grid [label $dwin.dups.label$i -text "$dup"] \
      -row $row -column 0 -sticky w
    grid [entry $dwin.dups.l1val$i -width 4 \
           -textvariable ${ns}::guiState(l1val$i)] \
      -row $row -column 1 -sticky ew -padx 4
    grid [entry $dwin.dups.l2val$i -width 4 \
            -textvariable ${ns}::guiState(l2val$i)] \
      -row $row -column 2 -sticky ew -padx 4
    incr row
    incr i
  }
  frame $dwin.buttons
  set row 0
  grid [button $dwin.buttons.cancel -text "Cancel" \
          -command "destroy $dwin" ] -row $row -column 0
  grid [button $dwin.buttons.submit -text "Apply segment fixes" \
      -command "destroy $dwin; ${ns}::guiMergeApplyFixes" ] \
      -row $row -column 1
  pack $dwin.msgframe -anchor nw -fill x -expand 1
  pack $dwin.dups -anchor nw -fill x -expand 1
  pack $dwin.buttons -anchor nw -fill x -expand 1
  
  return
}

proc ::MergeStructs::guiMergeApplyFixes {} {
  variable guiState
  set ns [namespace current]

  set ndups [llength $guiState(dupsegs)]
  set mol1segs {}
  set mol2segs {}
  for { set i 0 } { $i < $ndups } { incr i } {
    set origval [lindex $guiState(dupsegs) $i]
    set l1val $guiState(l1val$i)
    set l2val $guiState(l2val$i)
    if { [string equal $l1val $l2val] } {
    }
    if { ![string equal $l1val $origval] } {
      lappend mol1segs [list $origval $l1val]
    }
    if { ![string equal $l2val $origval] } {
      lappend mol2segs [list $origval $l2val]
    }
  }
  
  if { [llength $mol1segs] != 0 } {
    set nchanged [renameSegments $mol1segs \
                   "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb" \
                   "$guiState(mergef1dst).psf" "$guiState(mergef1dst).pdb" ]
    puts "Mol 1 $nchanged segments changed"
    set tmp $guiState(mergef1src)
    set guiState(mergef1src) $guiState(mergef1dst)
    set guiState(mergef1dst) $tmp
  }
  if { [llength $mol2segs] != 0 } {
    set nchanged [renameSegments $mol2segs \
                   "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb" \
                   "$guiState(mergef2dst).psf" "$guiState(mergef2dst).pdb" ]
    puts "Mol 2 $nchanged segments changed"
    set tmp $guiState(mergef2src)
    set guiState(mergef2src) $guiState(mergef2dst)
    set guiState(mergef2dst) $tmp
 }

  after idle "${ns}::guiMergeDetectConflicts"
}  

proc ::MergeStructs::guiMergeApplyPrefix { type} {
  variable guiState
  set ns [namespace current]

  set ndups [llength $guiState(dupsegs)]
  set mol1segs {}
  set mol2segs {}
  set m1prefix $guiState(mergem1prefix)
  set m2prefix $guiState(mergem2prefix)
  
  for { set i 0 } { $i < $ndups } { incr i } {
    set origval [lindex $guiState(dupsegs) $i]
    if {[string length $origval] > 3 } {
      continue
    }
    
    if { ![string equal $m1prefix ""] } {
      if { [string equal $type "prefix" ] } {
        set newval "$m1prefix$origval"
      } else {
        set newval "$origval$m1prefix"
      }
      lappend mol1segs [list $origval $newval]
    }
    
    if { ![string equal $m2prefix ""] } {
      if { [string equal $type "prefix" ] } {
        set newval "$m2prefix$origval"
      } else {
        set newval "$origval$m2prefix"
      }
      lappend mol2segs [list $origval $newval]
    }
  }
  
  if { [llength $mol1segs] != 0 } {
    set nchanged [renameSegments $mol1segs \
                   "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb" \
                   "$guiState(mergef1dst).psf" "$guiState(mergef1dst).pdb" ]
    puts "Mol 1 $nchanged segments changed"
    set tmp $guiState(mergef1src)
    set guiState(mergef1src) $guiState(mergef1dst)
    set guiState(mergef1dst) $tmp
  }
  if { [llength $mol2segs] != 0 } {
    set nchanged [renameSegments $mol2segs \
                   "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb" \
                   "$guiState(mergef2dst).psf" "$guiState(mergef2dst).pdb" ]
    puts "Mol 2 $nchanged segments changed"
    set tmp $guiState(mergef2src)
    set guiState(mergef2src) $guiState(mergef2dst)
    set guiState(mergef2dst) $tmp
 }

  after idle "${ns}::guiMergeDetectConflicts"
}  

proc ::MergeStructs::guiMergeApplyReplaceChar { } {
  variable guiState
  set ns [namespace current]

  set ndups [llength $guiState(dupsegs)]
  set mol1segs {}
  set mol2segs {}
  set m1prefix $guiState(mergem1prefix)
  set m2prefix $guiState(mergem2prefix)
  
  for { set i 0 } { $i < $ndups } { incr i } {
    set origval [lindex $guiState(dupsegs) $i]
    
    if { ![string equal $m1prefix ""] } {
      set newval [string replace $origval 0 0 $m1prefix]
      lappend mol1segs [list $origval $newval]
    }
    
    if { ![string equal $m2prefix ""] } {
      set newval [string replace $origval 0 0 $m2prefix]
      lappend mol2segs [list $origval $newval]
    }
  }
  
  if { [llength $mol1segs] != 0 } {
    set nchanged [renameSegments $mol1segs \
                   "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb" \
                   "$guiState(mergef1dst).psf" "$guiState(mergef1dst).pdb" ]
    puts "Mol 1 $nchanged segments changed"
    set tmp $guiState(mergef1src)
    set guiState(mergef1src) $guiState(mergef1dst)
    set guiState(mergef1dst) $tmp
  }
  if { [llength $mol2segs] != 0 } {
    set nchanged [renameSegments $mol2segs \
                   "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb" \
                   "$guiState(mergef2dst).psf" "$guiState(mergef2dst).pdb" ]
    puts "Mol 2 $nchanged segments changed"
    set tmp $guiState(mergef2src)
    set guiState(mergef2src) $guiState(mergef2dst)
    set guiState(mergef2dst) $tmp
 }

  after idle "${ns}::guiMergeDetectConflicts"
} 

proc ::MergeStructs::guiMergeConflictsResolved {} {
  variable guiState

  tk_messageBox -icon info -message \
    "No more segment name conflicts were detected... Merging structures." \
    -type ok
  
  set result [mergeMolecules [list "$guiState(mergef1src).psf" \
                               "$guiState(mergef1src).pdb" ] \
                             [list "$guiState(mergef2src).psf" \
                               "$guiState(mergef2src).pdb" ] \
                             $guiState(mergedFile) ]
                             
  file delete "$guiState(mergef1src).psf" "$guiState(mergef1src).pdb"
  file delete "$guiState(mergef1dst).psf" "$guiState(mergef1dst).pdb"

  file delete "$guiState(mergef2src).psf" "$guiState(mergef2src).pdb"
  file delete "$guiState(mergef2dst).psf" "$guiState(mergef2dst).pdb"
  
  file delete $guiState(mergedir)
  
  tk_messageBox -icon info -message \
    "Structures merged." \
    -type ok
  
  return $result
}

# Open the 2 molecules, detect repeated segment names, and return a list
# of those that appear in both files

proc ::MergeStructs::detectSegmentConflicts { psf1 pdb1 psf2 pdb2 } {
  set mol1 [mol new $psf1 autobonds off]
  mol addfile $pdb1
  set sel1 [atomselect $mol1 all]
  set seglist1 [lsort -unique [$sel1 get segname]]
  $sel1 delete
  mol delete $mol1 
  
  set mol2 [mol new $psf2 autobonds off]
  mol addfile $pdb2
  set sel2 [atomselect $mol2 all]
  set seglist2 [lsort -unique [$sel2 get segname]]
  $sel2 delete
  mol delete $mol2
  
  set l1 [llength $seglist1]
  set l2 [llength $seglist2]
  # For efficiency, loop over the shorter list, search the longer list, since
  # operation count should be (presumably) na*log(nb)
  if {$l1 > $l2} {
    set sla $seglist2
    set slb $seglist1
  } else {
    set sla $seglist1
    set slb $seglist2
  }
  set dup_list {}
  foreach segname $sla {
    if { [lsearch -sorted -exact $slb $segname] != -1 } {
      lappend dup_list $segname
    }
  }
  
  return $dup_list
}

# Rename segment names in a particular PSF/PDB file. This is useful to
# fix conflicts where segment names are repeated in two PSF/PDB files
proc ::MergeStructs::renameSegments { seglist inpsf inpdb \
                                          outpsf outpdb } {
  set psfcount [renameSegmentPsf $seglist $inpsf $outpsf]
  set pdbcount [renameSegmentPdb $seglist $inpdb $outpdb]
  
#  puts "psfcount $psfcount $pdbcount $seglist $inpsf $inpdb $outpsf $outpdb"
  return [list $psfcount $pdbcount]
}

# Replace segment names in the PDB from seglist, which is composed of
# { { oldseg1 newseg1} {oldseg2 newseg2} ... }
# Code adapted from routines by Jeff Comer
proc ::MergeStructs::renameSegmentPdb {seglist inpdb outpdb} {
  # Open the pdb to extract the atom records.
  set count 0
  set out [open $outpdb w]
  set in [open $inpdb r]

  foreach line [split [read $in] \n] {
    set string0 [string range $line 0 3]

    # Just write any line that isn't an atom record.
    if {![string match $string0 "ATOM"]} {
      puts $out $line
      continue
    }

    # Get the segment name.
    set segName [string trim [string range $line 72 75]]
    
    # Does this segment match?
    # This search might be done with lsearch, but the cost of sorting the list
    # properly might be more expensive than searching linearly here
    set segNameNew $segName
    foreach seg $seglist {
      foreach { si so } $seg {}
      if {[string equal $segName $si]} {
        set segNameNew [string range [format "%s    " $so] 0 3]
        break
      }
    }

    # Just write any atom record that doesn't
    # have the segment name we're changing.
    if {[string equal $segName $segNameNew]} {
      puts $out $line
      continue
    }

    # Generate the new pdb line.
    set temp0 [string range $line 0 71]
    set temp1 [string range $line 76 end]

    # Write the new pdb line.
    puts $out ${temp0}${segNameNew}${temp1}
    incr count
  }
  close $in
  close $out
  return $count
}

# Replace segment names in the PSF from seglist, which is composed of
# { { oldseg1 newseg1} {oldseg2 newseg2} ... }
# Code adapted from routines by Jeff Comer
proc ::MergeStructs::renameSegmentPsf {seglist inpsf outpsf} {
  # Open the pdb to extract the atom records.
  set count 0
  set out [open $outpsf w]
  set in [open $inpsf r]

  set record 0
  set n 0
  set num 1

  foreach line [split [read $in] \n] {
    # If we have finished with the atom records, just write the line.
    if {$n >= $num} {
      puts $out $line
      continue
    }

    if {!$record} {
      # Check if we have started the atom records.
      if {[string match "*NATOM" $line]} {
        set record 1
        set numIndex [expr [string last "!" $line]-1]
        set num [string trim [string range $line 0 $numIndex]]
      }

      # Write the line.
      puts $out $line
    } else {
      incr n
      set segName [string trim [string range $line 9 12]]

      # Does this segment match?
      set segNameNew $segName
      foreach seg $seglist {
        foreach { si so } $seg {}
        if {[string equal $segName $si]} {
          set segNameNew [string range [format "%s    " $so] 0 3]
          break
        }
      }

      if {![string equal $segName $segNameNew]} {
        set temp0 [string range $line 0 8]
        set temp1 [string range $line 13 end]

        # Write the new line.
        puts $out ${temp0}${segNameNew}${temp1}
        incr count
      } else {
        # Just write the line.
        puts $out $line
      }

    }
  }
  close $in
  close $out
  return $count
}

proc ::MergeStructs::mergeMolecules { mol1 mol2 outfile } {
  foreach { m1psf m1pdb } $mol1 {}
  foreach { m2psf m2pdb } $mol2 {}

  set psfcon [psfcontext create]
  psfcontext eval $psfcon {
    # this is a hack, just to get the topology file
    package require readcharmmtop
    set topologyfile [format "%s/top_all27_prot_lipid_na.inp" \
      $::env(CHARMMTOPDIR)]
    topology $topologyfile
    readpsf $m1psf
    readpsf $m2psf
    coordpdb $m1pdb
    coordpdb $m2pdb

    writepdb $outfile.pdb
    writepsf $outfile.psf
  }
  psfcontext delete $psfcon

  return
}

proc mergestructs_tk {} {
  ::MergeStructs::mergestructs
  return $::MergeStructs::w
}

