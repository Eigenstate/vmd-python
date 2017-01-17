set noderank [parallel noderank]
puts "starting: noderank is $noderank"
set localList ""

proc showTime {commentString} {
  set noderank  [parallel noderank]
  set tsec [clock seconds]
  set tms [clock clicks -milliseconds]
  set ts [clock format $tsec -format %c]
  puts "$commentString Time= $ts   tsec= $tsec tms= $tms Node= $noderank";
}


proc writeCombineGathered {filename dataNameVals dataValNum usesFreeSelection numFrames glist}  {
  #normally, only node 0 should run this
  global dataValArray
  global dataValArray 
  set noderank  [parallel noderank]
  set nodecount [parallel nodecount]
  showTime "Starting writeCombineGathered\n noderank= $noderank  nodecount= $nodecount"
 
  set dataName(vals) $dataNameVals 
  set currentMol 0
  #set outfile [open $filename "w"] 
  #puts "opened file $filename"



  #for testing, write file in this proc.


  #variable dataName
  #variable dataVal
  #variable dataValNum [expr $dataNum -1 ]
  #variable currentMol
  variable numDataFrames  $numFrames
  #hard code, this version does not handle non-free selection
  #set usesFreeSelection 1
  set dataFileVersion "1.4"
  
  set writeDataFile [open $filename w]
  puts $writeDataFile "# VMD Timeline data file"
  puts $writeDataFile "# CREATOR= $::tcl_platform(user)"
  puts $writeDataFile "# MOL_NAME= [molinfo $currentMol get name]"
  puts $writeDataFile "# DATA_TITLE= $dataName(vals)"
  puts $writeDataFile "# FILE_VERSION= $dataFileVersion"
  puts $writeDataFile "# NUM_FRAMES= $numFrames "
  puts $writeDataFile "# NUM_ITEMS= $dataValNum"

  if {$usesFreeSelection} {
    puts $writeDataFile "# FREE_SELECTION= 1"
    puts $writeDataFile "#"
  } else {
    puts $writeDataFile "# FREE_SELECTION= 0"
    puts $writeDataFile "#"
    # XXX this version does not handle non-free selection 
  }

  #next lines only write free-selections

  set frame -1
  for {set selIndex 0} {$selIndex<$dataValNum} {incr selIndex} {
    showTime "In write loop, selIndex= $selIndex"
    puts $writeDataFile "freeSelLabel $dataValArray(selLabel,$selIndex)"
    puts $writeDataFile "freeSelString  $dataValArray(selText,$selIndex)"
    set listNoderank 0
    #puts "about to loop over glist"
    foreach nodeElem $glist {
    showTime  "Node loop start: listNoderank= $listNoderank, selIndex=$selIndex\n"
      set lengthNodeElem [llength $nodeElem]
      showTime "length of nodeElem is $lengthNodeElem"
      set contigPos [expr $selIndex + 1]
      foreach frameElem $nodeElem {
        #showTime "frameElem= $frameElem"
        set theFrame [lindex $frameElem 0]
        set theVal [lindex $frameElem $contigPos] 
        #showTime "gathered listNoderank= $listNoderank, frameElem= $frameElem,  theVal= $theVal"
        puts $writeDataFile "$theFrame $theVal"
      }
      incr listNoderank
    }     
  }
 
  close $writeDataFile

  showTime "Finishing writeCombineGathered Node $noderank  nodecount= $nodecount"
  return
}

proc frame_work {i userdata} {
  #diagnostic only
  set noderank  [parallel noderank]
  set nodecount [parallel nodecount]
 
  set numFrames [lindex $userdata 0] 
 
  puts "Node $noderank / $nodecount. Work is on item $i, "

  set doub [expr 2* [lindex $theList $i] ]
} 

proc test_dyn { test_list} {
 set noderank  [parallel noderank]
 set nodecount [parallel nodecount]

 set datacount [llength $test_list]
 set decomposition "dynamic"
#  set decomposition "block"

 set userdata {}
 lappend userdata $frameNum 
 #lappend userdata $test_list
 #lappend userdata 5

 if { $decomposition == "dynamic" } {
   parallel for 0 [expr $datacount - 1] test_work $userdata
 } else {
   if { $decomposition == "block" } {
     set block [blockdecompose $datacount]
     set start [lindex $block 0]
     set end   [lindex $block 1]
     set skip  1
     set len   [expr $end - $start + 1]
   } elseif { $decomposition == "roundrobin" } {
     set start $noderank
     set end   $datacount
     set skip  $nodecount
     set len   $datacount
   }

   parallel barrier
   puts "Node $noderank render loop: $start to $end ($len frames)..."

   #Question: are next lines in dyanamic mode only because start, end are unset so presumed 0?
   # won't happen if dynamic
   for {set i $start} {$i <  $end} {incr i $skip} {
     test_work $i $userdata
   }
 }
}



proc readMakeSels {currentMol partSelText keyAtomSelString usesFreeSelection trajectoryFilename selListFilename} {
  #read in or make selections and put them into dataValArray
  #also, when reading, put description data about each selection (resid, etc.) into dataValArray
  global dataValArray
  global dataValNum
  set dataValNum 0
   #how to share this?
  set noderank  [parallel noderank]
  set nodecount [parallel nodecount]
  puts "*** first readMakeSels line" 
     #a flat list with the same information as dataValArray(sel,-- )
  showTime "*** Starting readMakeSels  Node $noderank"
  showTime "readMakeSels post var print" 
   #tests
   showTime "numframes is [molinfo 0 get numframes]" 
  #start selection/calc insert
  #at this point, no frames are loaded
  # read in the single required frame, structure is not calculated so structure franme is pointless here
  mol addfile $trajectoryFilename first 0 last 0 structureFrame step 1 waitfor all
  
  if {$usesFreeSelection}  {
   # we will use term 'contig' but for read-in selections, this is really an arbitrary user selection
   showTime "starting segment checks"
   set selListFile [open $selListFilename r]
   set selFileData [read $selListFile]
   close $selListFile  
   set selFileLines [split $selFileData "\n"]
   foreach ll $selFileLines { 
      if {$ll!=""} {
        #allows two formats of selection lists. With and without names
        #selection-text;label-text  <-- semicolon separates
        set entries [split $ll ";"]
        set selPhrase [lindex $entries 0]
        if {[lindex $entries 1]==""} {
          set dataValArray(selLabel,$dataValNum) $selPhrase
        } {
          set dataValArray(selLabel,$dataValNum) [lindex $entries 1]
        }

        
        #assume is a valid selection
        puts "current line is >$ll<"
        set com "atomselect $currentMol \"$selPhrase\""
        puts "command will be >$com<   labelText is $dataValArray(selLabel,$dataValNum)"
        set dataValArray(sel,$dataValNum) [namespace eval :: $com]
        set dataValArray(selText,$dataValNum) $selPhrase
        #test that it worked
        set numcheck [$dataValArray(sel,$dataValNum) num]
        showTime "added dataValArray(sel,$dataValNum) with atoms= $numcheck" 
        incr dataValNum
      }
   }
  } else {
    #make selections and put them into dataValArray
    #also, put description data about each selection (resid, etc.) into dataValArray
       #a flat list with the same information as dataValArray(sel,-- )

    showTime "*** Starting makeSels for non-useFreeSel  Node $noderank"
    set theSel [atomselect $currentMol "$partSelText and $keyAtomSelString"]

    set alist [$theSel get index]
    set alistLength [llength $alist]
    showTime "Node $noderank alistLength= $alistLength"
    #set frameTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
   
    set i 0
    foreach a $alist {
      #puts "checking alist, a= $a"
      set aSel [atomselect $currentMol "index $a"]
      #puts "Node $noderank, frame $refFrametheSNum" 
      #set refSel [atomselect $currentMol "same residue as index $a" frame $refFrameNum]
      
      set com "atomselect $currentMol \"same residue as index $a\"" 
      #puts "about to eval >$com<"
      # make this in global namespace.  Pollution, but works.
      set dataValArray(sel,$i) [namespace eval :: $com]

      #puts "Node $noderank,  dataValArray(sel,$i) is $dataValArray(sel,$i)"
      #puts "look for thexx"
      #set thexx [$dataValArray(sel,$i) get x]
      #puts "Node $noderank,  i is $i, thexx is $thexx"

      #puts "look for theinfo"
      #set theinfo [$dataValArray(sel,$i) get {name resname x y z}]
      #puts "Node $noderank,  i is $i, theinfo is $theinfo"


      ## for the future freeSel version, set selecton text and display text here
      set dataValArray(resid,$i) [$aSel  get resid]
      set dataValArray(chain,$i) [$aSel  get chain]
      set dataValArray(segname,$i) [$aSel  get segname]

      $aSel delete
      incr i
    }
   #this returns dataValArrayNum 
    set dataValArrayNum $i
    # the seletion command name list is stored in two places, since we needed a new way to present to measure sasalilst 
    showTime "Node $noderank, Done with selections. alistLength= $alistLength dataValArrayNum= $i"
   
    #return $dataValArrayNum
    }
  #fill dataVal arrays with residue info. 

  #end selection/calc insert

  showTime "*** Continuing makeSels  Node $noderank"
  ##set theSel [atomselect $currentMol "$partSelText and $keyAtomSelString"]
  ##
  ##set alist [$theSel get index]
  ##set alistLength [llength $alist]
  ##showTime "Node $noderank alistLength= $alistLength"
  #set frameTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
 
 #this returns dataValNum 
  #create a selection list array to be passed to the API designed for this
  set selValSelList ""
  
  showTime "Node $noderank, Done with selections. dataValNum= $dataValNum, dataValNum= $dataValNum"
 
  return $dataValNum 
}


proc userDefFrame { balanceRank userdata} {
  global localList
  global dataValArray 

  global dataValNum
  global dataValArray
   
  set noderank  [parallel noderank]
  set nodecount [parallel nodecount]

  #currently, only one map will be loadd
  set currentMol 0

   showTime "starting userDefFrame, Node $noderank"

  puts "Node $noderank, before testing length of localList"
  set testLength [llength $localList]
  puts "Node $noderank, testLength= $testLength  is length of localList"

    #read userdata
  showTime "about to set userdata"
  set numFrames [lindex $userdata 0] 
   #not using numFrames yet, maybe for tiled vesions
  set dataValNum [lindex $userdata 1] 
  set trajectoryFilename [lindex $userdata 2]
  set balanceCount [lindex $userdata 3]
  set targetBalanceCount [lindex $userdata 4]
  set framesPerTask [lindex $userdata 5]
  set remainder [lindex $userdata 6] 

  showTime "done setting userdata"
 

   
  showTime "In userDefFrame, balanceRank= $balanceRank"
  showTime "Reading from userdata; numFrames= $numFrames, dataValNum= $dataValNum, trajectoryFilename= $trajectoryFilename,  balanceCount= $balanceCount, targetBalanceCount= $targetBalanceCount, framesPerTask= $framesPerTask, remainder= $remainder"

  #delete any loaded frames
  animate delete  beg 0 end -1 skip 0 0
  puts "Node $noderank,  frames deleted" 
  set start [expr $balanceRank  * $framesPerTask]
  puts "Node $noderank, balanceRank= $balanceRank, have set start"
  set end   [expr ( ($balanceRank + 1 ) * $framesPerTask ) - 1]
  if {$balanceRank >= $targetBalanceCount} {
    puts "first choice in balanceRank passed, end= $end, numFrames= $numFrames"
    if {$end > [expr $numFrames -1]} { 
      puts "second choice in balanceRank passed, end= $end"
      set end [expr $numFrames - 1]
      puts "have set end after choices"
      puts "Node $noderank, above targetBalanceCount, balanceRank= $balanceRank, targetBalanceCount= $targetBalanceCount, start=$start, end=$end"
    }
  }
  puts "Node $noderank, have chosen for balanceRank"
  puts "start= $start  end= $end, Node $noderank, balanceRank= $balanceRank"
 

  showTime "Node $noderank, about to add frames, start= $start end= $end"
  mol addfile $trajectoryFilename first $start last $end step 1 waitfor all
  puts "Node $noderank, done adding frames"
  puts "Node $noderank, balance $balanceRank, numframes is [molinfo $currentMol get numframes]" 
  #all but currentMol should be passed in.


  #set firstAnalysisFrame 0
  #set lastAnalysisFrame $numFrames 

  showTime "In userDefFrame, Node $noderank, dataValNum= $dataValNum."
  #showTime "Node $noderank, before file add"

 
  #puts "Node $noderank, already have loaded [molinfo top get numframes] frames"

  
 puts "starting loop, Node $noderank, balanceRank= $balanceRank, start=$start, end=$end"
  #make a simple list of the selections, could do this once per job, but is quick

  showTime "Node $noderank, before loop over frames, start= $start, end= $end"
  puts "Node $noderank, before dataVal print line"
 
  #puts "Node $noderank - existence test starts"
  #if [info exists dataValArray] {
  #    puts "Node $noderank, dataValArray exists"
  #} else {
  #   puts "Node $noderank, dataValArray does not exist"
  #}
  #if [info exists dataValArray(resid,0)] {
  #    puts "Node $noderank, dataValArray(resid,0) exists"
  #} else {
  #   puts "Node $noderank, dataValArray(resid,0) does not exist"
  #}
     
  #puts "before testing dataValArray resid"
  #set testres $dataValArray(resid,0)
  #puts "resid 0 is $testres"
  #puts "before testing dataValArray sel"
  #set testsel $dataValArray(sel,0)
  #puts "sel 0 is $testsel"
  #puts "Node $noderank - existence test ends"

  #puts "Node $noderank, first two in dataValArray(sel,xxx) are $dataValArray(sel,0) then $dataValArray(sel,1)" 
  #puts "Node $noderank, after dataVal print line"
  #puts "Node $noderank, the first two in dataValSelList are [lindex $dataValSelList 0] then [lindex $dataValSelList 1]"
  showTime "Node $noderank, about to do theFrame loop, dataValNum= $dataValNum"
  set proteinNucSel [atomselect $currentMol "protein or nucleic"]
  for {set theFrame $start} {$theFrame <= $end} {incr theFrame} {
    showTime "in frame loop, theFrame= $theFrame, start= $start, end= $end"
    set loadedFrame [expr $theFrame - $start ]  
    $proteinNucSel frame $loadedFrame
    showTime "just set proteinNucSel frame"
 
    for {set dataValIndex 0}  {$dataValIndex < $dataValNum} {incr dataValIndex} {
      showTime "dataVal frame loop, dataValIndex= $dataValIndex"
      $dataValArray(sel,$dataValIndex) frame $loadedFrame
      #checkRangeLimits when inputting later
    }  
    showTime "Node $noderank, before userDefTLPar call, theFrame= $theFrame loadedFrame= $loadedFrame, dataValNum= $dataValNum"

    set userDefValList ""


    for {set i 0} {$i < $dataValNum} {incr i} {
            showTime "calc loop, i= $i"
            $dataValArray(sel,$i) frame $loadedFrame
            showTime "in calc loop, about to lappend"
            showTime "dataValArray(sel,$i)= $dataValArray(sel,$i)"
            showTime "proteinNucSel=$proteinNucSel"
            showTime "dataValArray(sel,$i)= $dataValArray(sel,$i)"
            lappend userDefValList [::userDefTLPar  "NULL" $dataValArray(sel,$i) $proteinNucSel]
            showTime "userDefValList is $userDefValList"
    }
    

    showTime "Node $noderank, before lappend userDefValList to flist, theFrame= $theFrame loadedFrame= $loadedFrame"   
    set flist "$theFrame ${userDefValList}"  

    set lengthFlist [llength $flist]
    #puts "flist= $flist"
    showTime "Node $noderank, before localList lappend, lengthFlist= $lengthFlist, theFrame= $theFrame loadedFrame= $loadedFrame"   
    lappend localList $flist


    ## debug printing
    ##puts "Node $noderank;  flist: $flist"
    ##puts "Node $noderank;   locallist: $localList"

    #report length of localList
    set theLength [llength $localList]
    showTime "Length of localList is $theLength, Node $noderank. "
  }
}




proc userDefCalc {psfFilename dcdFilename partSelText usesFreeSelection selListFilename labelText userProcFileName numFrames targetRunsPerNode outFilename} {

  global dataValNum 
  global localList
  set noderank  [parallel noderank]
  set nodecount [parallel nodecount]
  set start 0
  set end [expr $numFrames - 1]
  
 
  set currentMol 0

  showTime "starting userDefCalc"
  #XX note that we are not setting start and end frames
  showTime "crosscorrCalc, Node $noderank, before file add"
  mol new $psfFilename first 0 last 0 step 1  waitfor all
  #load only 1 frame to do selections with
  #do modulo file reading here if needed

  showTime "Node $noderank, before traj load"

  set trajectoryFilename $dcdFilename
  #mol addfile $trajectoryFilename first $start last $end step 1 waitfor all

  showTime "Node $noderank, loaded [molinfo top get numframes] frames"
 
  # Make work - create apprporiately-sized pieces of work for load balanced run.
  # Here we enumerate frames (later, tiles of residues-or-selections x frames)
  # to analyze  and divide up potential tasks.  Could be work blocks.
  # 
  #set partSelText "all"
  #probably partSelText will not be used here anyway 
  set keyAtomSelString "((all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) or (name BAS) )"
  #this will be overrulled to CA-only for protein only for ss contif structure search

  # make the selections and find description data 
  #this will set the globals dataValArray, dataValArray, selValSelList, dataValNum,,,,,,,, dataValNum 

  showTime "In userDefCalc, sourcing user-defined function"
  # XX Error catching and function testing goes here.
  source $userProcFileName
  showTime "In userDefCalc, sourced user-defined function, must be named ::userDefTLPar"
    
  set selMethod 1

  #for now, force partSelText (we are assigning sels explicitly)
  #set partSelText "all"
 
 # selection method, 
 # 1 = list of user sels  0= find contig secondary structure
 # force 0, contig secondary structure method for now
 ##makeSels $currentMol $partSelText $keyAtomSelString $structureFrame $selMethod $setStructFilename $trajectoryFilename
  showTime "In userDefCalc, selListFilename=  $selListFilename"


  readMakeSels $currentMol $partSelText $keyAtomSelString $usesFreeSelection $trajectoryFilename $selListFilename
  

   

  puts "writing to userdata; trajectoryFilename is $trajectoryFilename"

  set targetBalanceCount [expr $nodecount * $targetRunsPerNode]
  puts "targetBalanceCount= $targetBalanceCount"
  set framesPerTask [expr int ($numFrames / $targetBalanceCount) ] 
  #range check here?
  set remainder [expr $numFrames % $targetBalanceCount ]
  if {$remainder ==0} {
    set balanceCount $targetBalanceCount
  } else { 
    set balanceCount [expr $targetBalanceCount + int(ceil((1.0*$remainder) / $framesPerTask))]
    puts "Node $nodecount, have remainder, set balanceCount= $balanceCount"
  }

# XX range check for giant node count?

#  if {$balanceCount > $numFrames} {
#    set balanceCount $numFrames
#  }
 
  set userdata [list $numFrames $dataValNum $trajectoryFilename $balanceCount $targetBalanceCount $framesPerTask $remainder] 
  puts "set userdata, userdata= $userdata"
  showTime "Now for parallel for, numFrames = $numFrames, dataValNum= $dataValNum trajectoryFilename= $trajectoryFilename, targetBalanceCount= $targetBalanceCount, balanceCount= $balanceCount, framesPerTask $framesPerTask, remainder= $remainder, nodecount= $nodecount"

  parallel for 0 [expr $balanceCount - 1] userDefFrame $userdata 

  showTime "After parallel for"

  parallel barrier
  showTime "After parallel barrier"
  
  set gatherList [parallel allgather $::localList]
  showTime "After parallel allgather"
  if {$noderank == 0 } {
    ##puts "Node $noderank gatherList= $gatherList"
  }
  # only print messages on node 0
  set thePid [pid]
  #set outFileName "crosscorr"
  set labelText "LABEL TEXT HERE"
  if {$noderank == 0} {
   showTime "before writeCombineGathered call: outFilename= $outFilename labelText= $labelText"
    writeCombineGathered $outFilename $labelText $dataValNum $usesFreeSelection $numFrames $gatherList
  }

}


#set n 1000000
#set alist ""; for {set i 0} {$i<$n} {incr i} {lappend alist [expr 10 * $i]}





