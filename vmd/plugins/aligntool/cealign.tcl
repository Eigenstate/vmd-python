# align.tcl
# 
#	This file implements the Tcl code for aligning two
#	proteins, using CE.
# 
# written by Michael Patrick Januszyk for Zan Schulten
# 
# protein allignment done using CE
#   Protein Structure Alignment by Incremental COMBINATORIAL EXTENSION of
#                           the Optimal Path.
#               Authors:   Shindyalov I.N., Bourne P.E.
#                 e-mail:   {shindyal,bourne}@sdsc.edu
#                  URL:   http://ce.sdsc.edu/ce.html
# Copyright (c)  1997-2000   The Regents of the University of California
#
# SCCS: %Z% %M% %I% %E% %U%
# $Id: cealign.tcl,v 1.14 2013/04/15 15:28:02 johns Exp $
#
# Package info
# this script is executed from VMD; it takes two pdb files.and feeds them into
# CE, which aligns the second onto the 1st.
# it creates a parsable output file containing information about the files

# example run:
# 	source parse.tcl
# 	align 1BF4.pdb A 2cro.pdb -
#
#       multAlign 1gpw.pdb E 2tss.pdb A 4hhb.pdb A

# first pdb/chain entered is target
# calls pairwise CE alignment on target and each other pdb
# from these pairwise alignments, constructs a multiple
# sequence alignment

package provide cealign 1.0
#######################
#create the namespace
#######################
namespace eval ::vmd_cealign {
  namespace export cealign
}

proc ::vmd_cealign::align {PdbFileString ChainString ceDirectory ceBin ceScratch PdbDirectory} { 
  variable CEPath
  variable InfoFile
  variable PdbFileList
  variable ChainList
  variable i
  variable Chain
  variable PdbFile
  variable Cmnd
  variable error

  # sets directories
  set CEPath [file join $ceDirectory $ceBin ]
  set InfoFile [file join $ceScratch "result.txt"] 

  set PdbFileList [split $PdbFileString " "]
  set ChainList [split $ChainString " "]

  set numAlign [llength $PdbFileString]

  for {set i 0} {$i<$numAlign} {incr i} {
    set Chain($i) [lindex $ChainList $i]
    set PdbFile($i) [lindex $PdbFileList $i]
  }

  # runs CE on the two pdb files
  set Cmnd "(cd $ceDirectory;   $CEPath  - [file join $PdbDirectory $PdbFile(0)] $Chain(0) [file join $PdbDirectory $PdbFile(1)] $Chain(1) $ceScratch >! $InfoFile)"
  puts "Now to execute: $Cmnd"
  set error [catch { exec  /bin/csh -c $Cmnd }]
  #exec  /bin/csh -c $Cmnd 
  puts "ran command, error= $error"

  # call transform procedure
  transform $InfoFile $PdbDirectory $PdbFile(0) $PdbFile(1) $ceScratch 

  return

}


proc ::vmd_cealign::transform {InfoFile PdbDirectory PdbFile1 PdbFile2 scratchdir} {
  variable InfoFileH
  variable Temp2
  variable Temp1
  variable Seq1
  variable Seq2
  variable Start1
  variable Start2
  variable TM11
  variable TM12
  variable TM13
  variable TM14
  variable TM21
  variable TM22
  variable TM23
  variable TM24
  variable TM31
  variable TM32
  variable TM33
  variable TM34
  variable TM
  variable TMIndex
  variable NewFile
  variable NameLine1
  variable NameLine2
  variable tm
  variable Pdb1
  variable Pdb2
  variable SeqIndex
  variable ResID1
  variable ResID2
  variable Pos1
  variable Pos2
  variable SegNama1
  variable SegNama2
  variable Temp
  variable FormattedLine
  variable ResNumShift1
  variable ResNumShift2
  variable j
  variable Chain1
  variable Chain2

  puts "starting transform..."
  set InfoFileH [open $InfoFile r]
  set Temp2 "null"

  while {$Temp2 != "Cha"} {
    set Temp1 [gets $InfoFileH]
    set Temp2 [string range $Temp1 0 2]
  }
  set j [expr 1 + [string last ":" $Temp1]]
  set Chain1 [string range $Temp1 $j $j]
  set Temp1 [gets $InfoFileH]
  set j [expr 1 + [string last ":" $Temp1]]
  set Chain2 [string range $Temp1 $j $j]
  set Temp1 [gets $InfoFileH]
  set Temp1 [gets $InfoFileH]

  # reads in resID information
  set Seq1 [gets $InfoFileH]
  set Seq1 [gets $InfoFileH]
  set Seq2 [gets $InfoFileH]
  set Start1 [string trimleft [string range $Seq1 8 12] ]
  set Start2 [string trimleft [string range $Seq2 8 12] ]

  while {($Temp2 != "X2")  && (![eof $InfoFileH])} {
    set Temp1 [gets $InfoFileH]
    set Temp2 [string range $Temp1 5 6]
    if {[string range $Temp1 0 6] == "Chain 1"} {
      set Temp1 [string range $Temp1 14 [string len $Temp1]]
      set Seq1 $Seq1$Temp1
    }
    if {[string range $Temp1 0 6] == "Chain 2"} {
      set Temp1 [string range $Temp1 14 [string len $Temp1]]
      set Seq2 $Seq2$Temp1
    }
  }

  # reads in transformation matrix
  set TM11 [string range $Temp1 11 19]
  set TM12 [string range $Temp1 28 36]
  set TM13 [string range $Temp1 45 53]
  set TM14 [string range $Temp1 62 73]
  set Temp1 [gets $InfoFileH]
  set TM21 [string range $Temp1 11 19]
  set TM22 [string range $Temp1 28 36]
  set TM23 [string range $Temp1 45 53]
  set TM24 [string range $Temp1 62 73]
  set Temp1 [gets $InfoFileH]
  set TM31 [string range $Temp1 11 19]
  set TM32 [string range $Temp1 28 36]
  set TM33 [string range $Temp1 45 53]
  set TM34 [string range $Temp1 62 73]
  close $InfoFileH

  # formats and prints transformation matrix
  set TM [list $TM11 $TM12 $TM13 $TM14 $TM21 $TM22 $TM23 $TM24 $TM31 $TM32 $TM33 $TM34]

  for {set TMIndex 0} {$TMIndex<12} {incr TMIndex} {
    set TM [lreplace $TM $TMIndex $TMIndex [string trimleft [lindex $TM $TMIndex] ] ]
  }

  set NewFile [open [file join $scratchdir "out.align"] w]
  set NameLine1 "#Name1: "
  append NameLine1 $PdbFile1
  puts $NewFile $NameLine1
  set NameLine2 "#Name2: "
  append NameLine2 $PdbFile2
  puts $NewFile $NameLine2

  set tm "#TransMat:"
  for {set TMIndex 0} {$TMIndex<12} {incr TMIndex} {
    append tm " "
    append tm [lindex $TM $TMIndex]
  }

  puts $NewFile $tm

  # formats and prints position, resID, segName
  set Pdb1 [open [file join $PdbDirectory $PdbFile1] r]
  set Pdb2 [open [file join $PdbDirectory $PdbFile2] r]

  # make quick pass through pdb files to determine
  # where numbering starts (-1,0,1, or 2 usually)
  # and then adjust Start ResNums appropriately
  set Temp [gets $Pdb1]
  while {[string range $Temp 0 3] != "ATOM"} {
    set Temp [gets $Pdb1]
  }
  while {[string range $Temp 21 21] != $Chain1} {
    set Temp [gets $Pdb1]
  }

  set ResNumShift1 [expr [string range $Temp 23 25] -1]
  set Temp [gets $Pdb2]
  while {[string range $Temp 0 3] != "ATOM" && [string range $Temp 21 21] != $Chain2} {
    set Temp [gets $Pdb2]
  }
  set ResNumShift2 [expr [string range $Temp 23 25] -1]
  set Start1 [expr $Start1 + $ResNumShift1]
  set Start2 [expr $Start2 + $ResNumShift2]

  for {set SeqIndex 14} {$SeqIndex<[string len $Seq1]} {incr SeqIndex} {
    set ResID1 [string range $Seq1 $SeqIndex $SeqIndex]
    set ResID2 [string range $Seq2 $SeqIndex $SeqIndex]
    if {$ResID1 == "-"} {
      set Pos1 "-"
      set SegName1 "-"
    } else {
      set Pos1 $Start1
      set ResNum -1
      # determines segName
      while {$ResNum != $Start1} {
   	set Temp [gets $Pdb1]
	set ResNum [string range $Temp 23 25]
      }
      set SegName1 [string trimright [string range $Temp 72 75] ]
      if {[string len $SegName1] == 0} {
        set SegName1 "none"
      }
      incr Start1
    }
    if {$ResID2 == "-"} {
      set Pos2 "-"
      set SegName2 "-"
    } else {
      set Pos2 $Start2
      set ResNum -1
      # determines segName
      while {$ResNum != $Start2} {
        set Temp [gets $Pdb2]
        set ResNum [string range $Temp 23 25]
      }
      set SegName2 [string trimright [string range $Temp 72 75] ]
      if {[string len $SegName2] == 0} {
        set SegName2 "none"
      }
      incr Start2
    }

    set FormattedLine $Pos1
    append FormattedLine " "
    append FormattedLine $ResID1
    append FormattedLine " "
    append FormattedLine $Chain1
    append FormattedLine " "
    append FormattedLine $SegName1
    append FormattedLine " "
    append FormattedLine $Pos2
    append FormattedLine " "
    append FormattedLine $ResID2
    append FormattedLine " "
    append FormattedLine $Chain2
    append FormattedLine " "
    append FormattedLine $SegName2
    puts $InfoFileH $FormattedLine
  } 

  close $NewFile
  return 0
}

proc multAlign {pdbString chainString ceDir ceBin ceScratch pdbDir} {
  variable alignLen
  variable ceCall
  variable cePath
  variable Chain
  variable chainList
  variable count
  variable diff
  variable lstr
  variable rstr
  variable numCA1
  variable numCA2
  variable estr
  variable sstr
  variable FH_out
  variable FH_res
  variable FH_target
  variable fto
  variable fullTargSeq
  variable fullSecSeq
  variable gappedTargSeq
  variable gaps
  variable gapsAdded
  variable gapsSoFar
  variable i
  variable j
  variable line1
  variable maxTargEnd
  variable minTargStart
  variable nonGapsSoFar
  variable numAlign
  variable oldResNum
  variable PdbFile
  variable pdbList
  variable resID
  variable resNum
  variable secSeq
  variable secStart
  variable shift
  variable sstr
  variable targGap
  variable targOffset
  variable targSeq
  variable targStart
  variable tempTargGap
  variable threeToOne
  variable TM
  variable totGaps

  set cePath [file join $ceDir $ceBin]

  set pdbList [split $pdbString " "]
  set chainList [split $chainString " "]

  set numAlign [llength $pdbList]

  for {set i 0} {$i<$numAlign} {incr i} {
    set Chain($i) [lindex $chainList $i]
    set PdbFile($i) [lindex $pdbList $i]
  }

  for {set i 1} {$i<=500} {incr i} {
    set targGap($i) 0
  }

  set targOffset 0
  set minTargStart 1000
  set maxTargEnd 0

  for {set count 1} {$count<$numAlign} {incr count} {
    # call CE to find pairwise alignment of target and $numAlign pdb

    set ceCall "cd $ceDir; "
    append ceCall $cePath " - " $ceScratch "/" $PdbFile(0) " " $Chain(0) " " $ceScratch "/" $PdbFile($count) " " $Chain($count) " " $ceScratch " >! " $ceScratch "/resTemp.txt" 

    puts $ceCall
    exec  /bin/csh -c $ceCall

    set FH_res [open [file join $ceScratch "resTemp.txt"] r]

    for {set j 0} {$j<6} {incr j} {
      set line1 [gets $FH_res]
    }

    set lstr [expr [string first "=" $line1] + 1]
    set rstr [expr [string first ")" $line1] - 1]
    set numCA1 [string range $line1 $lstr $rstr]
    set line1 [gets $FH_res]
    set lstr [expr [string first "=" $line1] + 1]
    set rstr [expr [string first ")" $line1] - 1]
    set numCA2 [string range $line1 $lstr $rstr]

    if {$numCA1 <1} {
      puts "Error in file.  No C-alpha atoms found in molecule A, chain $Chain(0)."
      return -code error "No C-alpha atoms found in molecule A, chain $Chain(0)." 
    }
    if {$numCA2 <1} {
      puts "Error in file.  No C-alpha atoms found in molecule B, chain $Chain(1)." 
      return -code error "No C-alpha atoms found in molecule B, chain $Chain(1)." 
    }

    set line1 [gets $FH_res]
    set line1 [gets $FH_res]

    set sstr [expr [string first "=" $line1] + 1]
    set estr [expr [string first "R" $line1] - 1]

    set alignLen($count) [string range $line1 $sstr $estr]

    set line1 [gets $FH_res]
    set line1 [gets $FH_res]

    set gaps 0
    set totGaps 0

    set targStart($count) [string range $line1 10 12]
    set targSeq($count) [string range $line1 14 [expr [string length $line1]-1]]

    for {set i 1} {$i<=[expr $targStart($count) + $alignLen($count)]} {incr i} {
      set tempTargGap($i)  0
    }

    set alignLen($count) 0

    for {set i 14} {$i<[expr [string length $line1] - 0]} {incr i} {
      if {[string range $line1 $i $i] == "-" } {
        incr gaps
        incr totGaps
        set tempTargGap([expr $targStart($count)+$i-14-$totGaps]) $gaps
      } else {
        incr alignLen($count)
        set gaps 0
      }
    }
    set line1 [gets $FH_res]
    set secStart($count) [string range $line1 10 12]
    set secSeq($count) [string range $line1 14 [expr [string length $line1] -0]]
    if { [expr $secStart($count) - $targStart($count)] > $targOffset } {
      set targOffset [expr $secStart($count) - $targStart($count)]
    }
    if { $targStart($count) < $minTargStart } {
      set minTargStart $targStart($count)
    }

    # go through and determine the positon/number of gaps in the target chain
    set shift 0

    set line1 [gets $FH_res]
    while {[eof $FH_res] != 1} {
      if { [string range $line1 0 7] == "Chain 1:" } {
        append targSeq($count) [string range $line1 14 [expr [string length $line1] -1] ]
        set shift [expr $shift + 70]
        for {set i 14} {$i<[expr [string length $line1]]} {incr i} {
          if { [string range $line1 $i $i] == "-" } {
            incr gaps
            incr totGaps
            set tempTargGap([expr $targStart($count)+$i+$shift-14-$totGaps]) $gaps
          } else {
            incr alignLen($count)
            set gaps 0
          }

        }
      }
      if { [string range $line1 0 7] == "Chain 2:" } {
        append secSeq($count) [string range $line1 14 [expr [string length $line1] -1] ]
      }
      if { [string range $line1 5 8] == "X2 ="} {
          set TM(1,1,$count) [string range $line1 11 18]
          set TM(1,2,$count) [string range $line1 28 35]
          set TM(1,3,$count) [string range $line1 45 52]
          set TM(1,4,$count) [string range $line1 62 69]
          set line1 [gets $FH_res]
          set TM(2,1,$count) [string range $line1 11 18]
          set TM(2,2,$count) [string range $line1 28 35]
          set TM(2,3,$count) [string range $line1 45 52]
          set TM(2,4,$count) [string range $line1 62 69]
          set line1 [gets $FH_res]
          set TM(3,1,$count) [string range $line1 11 18]
          set TM(3,2,$count) [string range $line1 28 35]
          set TM(3,3,$count) [string range $line1 45 52]
          set TM(3,4,$count) [string range $line1 62 69]
      }

    set line1 [gets $FH_res]

    }

    if { $maxTargEnd < [expr $alignLen($count) + $targStart($count) - 1] } {
      set maxTargEnd [expr $targStart($count) + $alignLen($count) -1]
    }
    for {set i 1} {$i<[array size tempTargGap]} {incr i} {
      if {$tempTargGap($i) > $targGap($i)} {
        set targGap($i) $tempTargGap($i)
      }
    }

    puts ""
    for {set i 1} {$i<[array size targGap]} {incr i} {
      if {$targGap($i)>0} {
        puts -nonewline "$i $targGap($i) "
      }
    }

    close $FH_res
  }


  array set threeToOne {ALA A ARG R ASN N ASP D CYS C GLU E GLN Q GLY G HIS H ILE I LEU L LYS K MET M PHE F PRO P SER S THR T TRP W TYR Y VAL V}


  # now construct target sequence from information above

  set FH_target [open [file join $ceScratch $PdbFile(0)] r]

  set line1 "NULL"
  while { [string range $line1 0 3] != "ATOM" } {
    set line1 [gets $FH_target]
  }

  #construct full string of resIDs for target

  while { [string range $line1 21 21] != $Chain(0) } {
    set line1 [gets $FH_target]
  }
  while { [string range $line1 23 25] != $minTargStart } {
    set line1 [gets $FH_target]
  }
  set oldResNum 0
  set fullTargSeq ""
  while { [string range $line1 23 25] != $maxTargEnd } {
    set resNum [string range $line1 23 25]
    if {$resNum != $oldResNum } {
      set oldResNum $resNum
      set resID [string range $line1 17 19]
      set resID $threeToOne($resID)
      set fullTargSeq $fullTargSeq$resID
    }
    set line1 [gets $FH_target]
  }

  set resID [string range $line1 17 19]
  set resID $threeToOne($resID)
  set fullTargSeq $fullTargSeq$resID

  set gappedTargSeq $fullTargSeq
  set gapsAdded 0
  for {set i [expr $minTargStart]} {$i<=$maxTargEnd} {incr i} {
    for {set j 0} {$j<$targGap($i)} {incr j} {
      set first [string range $gappedTargSeq 0 [expr $i-$minTargStart+$gapsAdded] ]
      set last [string range $gappedTargSeq [expr 1+$i-$minTargStart+$gapsAdded] [expr [string length $gappedTargSeq] + $gapsAdded +1] ]
      set gappedTargSeq ""
      append gappedTargSeq $first "-" $last
      incr gapsAdded
    }
  }

  puts "\ngappedTargSeq = \n$gappedTargSeq"


  # Go through other sequences and create complete and gapped sequences for them

  for {set count 1} {$count<$numAlign} {incr count} {
    set gapsSoFar 0
    for {set i 1} {$i<$targStart($count)} {incr i} {
      set gapsSoFar [expr $gapsSoFar + $targGap($i)]
    }
    set fullSecSeq($count) ""
    for {set i 0} {$i<[expr $targStart($count)-$minTargStart+$gapsSoFar] } {incr i} {
      append fullSecSeq($count) "-"
    }

    set j 0

    for {set i 0} {$i<[string length $secSeq($count)]} {incr i} {
      if { [string range $gappedTargSeq [expr $j+$targStart($count)-$minTargStart+$gapsSoFar] [expr $j+$targStart($count)-$minTargStart+$gapsSoFar]] == "-" } { 
        if { [string range $targSeq($count) $i $i] == "-" } {
          append fullSecSeq($count) [string range $secSeq($count) $i $i]
        } else {
          append fullSecSeq($count) "-"
          set i [expr $i - 1 ]
        }
      } else {
        append fullSecSeq($count) [string range $secSeq($count) $i $i];
      }
      incr j
    }

    set diff [expr [string length $gappedTargSeq] - [string length $fullSecSeq($count)] ]
    for {set i 0} {$i<$diff} {incr i} {
      append fullSecSeq($count) "-"
    }

  puts "\n SECOND: \n$fullSecSeq($count)"

  }


  # write to output file

  set FH_out [open [file join $ceScratch "out.align"] w]

  # print file names
  for {set count 0} {$count<[llength $pdbList]} {incr count} {
    puts $FH_out "#Name$count: $PdbFile($count)"
  }

  # prints transformation matrix
  for {set count 1} {$count<$numAlign} {incr count} {
    puts -nonewline $FH_out "#TransMat$count:"
    for {set i 1} {$i<=3} {incr i} {
      for {set j 1} {$j<=4} {incr j} {
        set TM($i,$j,$count) [string trimleft $TM($i,$j,$count)]
        set TM($i,$j,$count) [string trimright $TM($i,$j,$count)]
        puts -nonewline $FH_out " $TM($i,$j,$count)"
      }
    }
    puts $FH_out ""
  }

  # print 4n column format for resNum chainID
  set fullSecSeq(0) $gappedTargSeq

  for {set count 0} {$count<$numAlign} {incr count} {
    set nonGapsSoFar($count) 0
  }
  for {set index 0} {$index<[expr 1*[string length $gappedTargSeq]]} {incr index} {
    for {set count 0} {$count<$numAlign} {incr count} {
      if {$count == 0} {
        if {[string range $fullSecSeq($count) $index $index] != "-"} {
          puts -nonewline $FH_out "[expr $minTargStart + $nonGapsSoFar($count)] R $Chain($count) Seg"
          incr nonGapsSoFar($count)
        } else {
          puts -nonewline $FH_out "- R $Chain($count) Seg"
        }
      } else {
        if {[string range $fullSecSeq($count) $index $index] != "-"} {
          puts -nonewline $FH_out " [expr $secStart($count) + $nonGapsSoFar($count)] R $Chain($count) Seg"
          incr nonGapsSoFar($count)
        } else {
          puts -nonewline $FH_out " - R $Chain($count) Seg"
        }
      }
    }
    puts $FH_out ""
  }
  close $FH_out
}


