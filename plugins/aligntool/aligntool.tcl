#alignView.tcl  -- VMD script to compare 2 molecules, in the style of a contact map
#
# Copyright (c) 2001 The Board of Trustees of the University of Illinois
#
# Barry Isralewitz  barryi@ks.uiuc.edu    
# vmd@ks.uiuc.edu
#
# $Id: aligntool.tcl,v 1.34 2007/09/20 20:21:03 jordi Exp $
#
#
# To try out:
# 0) start vmd
# 1) mol pdbload 1tit
# 2) source zoomseq-comp.01.tcl
# 3) select "fit all", then "Calculate: Calc res-res Dists"
# 4) move over contact map with button 2 (middle button) pressed
#    to see pair selected
# 5) After changing moelcule/typed selection, re-select MoleculeA
#    popup menui, then re-select "Calculate: Calc res-res Dists".
#
#   Note: fun to try with slightly misaligned molecs, try two copies
#    slightly moved/shifted from each other.

## Tell Tcl that we're a package and any dependencies we may have
package require cealign   1.0
package provide aligntool 1.0

#######################
#create the namespace
#######################
namespace eval ::vmd_aligntool {
  namespace export aligntool
  variable fit_scalex 1 
  variable fit_scaley 1 
}

####################
#define the procs
####################
proc ::vmd_aligntool::canvasScrollY {args} { 
  variable w
  eval $w.can yview $args
#  eval $w.vertScale yview $args 
}     

proc ::vmd_aligntool::canvasScrollX {args} { 
  variable w
  eval $w.can xview $args
  eval $w.horzScale xview $args 
  return
}

proc ::vmd_aligntool::dataCanvasScrollX {args} {
  #synchronizes scollbar _and_ scale window to data movement
  variable w
  variable scalex
  variable fit_scalex
  variable xcanwindowmax
  variable xcol
  variable firstData
  variable resWidth 
  variable numHoriz
  variable xScaling
  #note -- not needed, runs fine without, isnt even called!
  puts "in dataCanvasScrollX, xScaling= $xScaling"
  if {$xScaling == 1} {return}

  #lock out access to this function
  set xScaling 1
  puts "set xScaling to $xScaling"
  set v [$w.can xview]
  set frac [expr [lindex $v 1] - [lindex $v 0] ] 
  #set frac [expr $start-$end ] 
  #asking for trouble, likely many runs till settles
  set fit_scalex [expr $frac * ( (0.0 + $xcanwindowmax - $xleftmargin ) / ($resWidth) * (1+ $alignIndex(num)) )  ]
  puts "fit_scalex= $fit_scalex, scalex= $scalex"
  if {$fit_scalex != $scalex} {
    set scalex $fit_scalex
    redraw name func op
  }
  set xScaling 0
}

proc ::vmd_aligntool::dataCanvasScrollY {args} {
  #synchronizes scollbar _and_ scale window to data movement
  variable w
  variable scaley
  variable fit_scaley
  variable ycanwindowmax
  variable ytopmargin
  variable ybottommargin
  variable ybox
  variable dataValXNum
  variable yScaling
  puts "in dataCanvasScrollY, yScaling= $yScaling"
  if {$yScaling == 1} {return}

  #lock out access to this function
  set yScaling 1

  puts "set yScaling to $yScaling"
  set v [$w.can yview]
  set frac [expr [lindex $v 1] - [lindex $v 0] ] 
  #set frac [expr $start-$end ] 
  #asking for trouble, likely many runs till settles
  if {$dataValXNum(0) > 0} {
    set fit_scaley  [expr $frac * ((0.0 + $ycanwindowmax - $ytopmargin - $ybottommargin) / ($ybox * ($dataValXNum(0) + 1) ) ) ]
    puts "fit_scaley= $fit_scaley, scaley= $scaley"
    if {$fit_scaley != $scaley} {
      set scaley $fit_scaley
      redraw name func op 
    }
  }
  set yScaling 0
}


proc ::vmd_aligntool::lookupCode {resname} {
  variable codes

  set result ""
  if {[catch { set result $codes($resname) } ]} {
    set result $resname
  } else {
    set result " $result "
  }
  return $result
}

proc ::vmd_aligntool::stopZoomSeq {} {
  menu aligntool off
}

proc ::vmd_aligntool::chooseColor {field intensity} {
  variable dataName
  set field_color_type 4 
  #hack to sefault to struct field type coloring
  if {$dataName($field) != "struct"} {
    if {$intensity < 0} {set intensity 0}
    if {$intensity > 255} {set intensity 255}
    set intensity [expr int($intensity)]
    #set field_color_type $field 
    #check color mapping
    set field_color_type 3 
  }
  #super hacky here
  switch -exact $field_color_type {            
    #temporaily diable so greyscale color  only
    3333 {   
      set red $intensity
      set green [expr 255 - $intensity]
      set blue 150 
    }
    4 {
      #the field_color_type hack sends all structs to here 
      if { 
        [catch {
          switch $intensity {
          
            B {set red 180; set green 180; set blue 0}
            C {set red 255; set green 255; set blue 255}
            E {set red 255; set green 255; set blue 100}
            T {set red 70; set green 150; set blue 150}
            G {set red 255; set green 160; set blue 255}
            H {set red 225; set green 130; set blue 225}
            I {set red 225; set green 20; set blue 20}
            default {set red 100; set green 100; set blue 100}
          }
        } ] 
      } else { #badly formatted file, intensity may be a number
        set red 0; set green 0; set blue 0 
      }
    }
    default {
      #set red [expr 200 - int (200.0 * ($intensity / 255.0) )]
      #set green $red
      #set red 140; set blue 90; set green 90;
      set red $intensity
      set green $intensity
      set blue $intensity
    }
  }
  
  #convert red blue green 0 - 255 to hex
  set hexred     [format "%02x" $red]
  set hexgreen   [format "%02x" $green]
  set hexblue    [format "%02x" $blue]
  set hexcols [list $hexred $hexgreen $hexblue]

  return $hexcols
}


proc ::vmd_aligntool::redraw {name func op} { 
  variable x1 
  variable y1 
  variable so
  variable w 
  variable monoFont
  variable xcanwindowmax 
  variable ycanwindowmax 
  variable xcanmax 
  variable ycanmax
  variable ybox 
  variable xsize 
  variable ysize 
  variable resnamelist 
  variable structlist 
  variable betalist 
  variable sel 
  variable canvasnew 
  variable scalex 
  variable scaley 
  variable dataValX 
  variable dataValXNum 
  variable dataName 
  variable dataNameLast 
  variable ytopmargin 
  variable ybottommargin 
  variable xleftmargin
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable vertTextRight
  variable vertHighLeft
  variable vertHighRight
  variable amino_code_toggle 
  variable dataWidth 
  variable dataMargin 
  variable firstData 
  variable dataMin 
  variable dataMax 
  variable xPosScaleVal
  variable usableMolLoaded
  variable rectCreated
  variable prevScalex
  variable prevScaley
  variable alignIndex
  variable resWidth
  

  if { ($usableMolLoaded) && ($alignIndex(num) >=0 ) } {
    set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValXNum(0) + 1) )]  

    set xsize [expr  $xleftmargin +  $scalex *  $resWidth * $alignIndex(num) ]

    set xcanmax(horz) $xsize

    #if {$ycanmax(data) < $ycanwindowmax} {
    #    set ycanmax(data) $ycanwindowmax
    #}


    if {$xcanmax(horz) < $xcanwindowmax} {
      set xcanmax(horz) $xcanwindowmax
    }

    #$w.vertScale configure -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)"
    $w.horzScale configure -scrollregion "0 0 $xcanmax(horz) $ycanmax(horz)"
    #drawVertScale
    drawHorzScale
    
    #puts "fieldLast is $dataNameLast"
    #temporary, until put multi-cols in
    


    #draw data on can
    #loop over all data fields

    if {! $rectCreated} {
      #this until separate data and scale highlighting
      $w.horzScale delete dataScalable
      #puts "drawing rects, scalex is $scalex"
      puts "setting min/max, firstData= $firstData" 

      #drawVertHighlight 
    }  else {


      $w.horzScale scale dataScalable $xleftmargin  $ytopmargin [expr $scalex / $prevScalex]  [expr $scaley / $prevScaley ]

    } 
    
    set rectCreated 1
    set prevScaley $scaley
    set prevScalex $scalex
  }

  return
}



proc ::vmd_aligntool::makecanvas {} {
  variable xcanmax 
  variable ycanmax
  variable w
  variable xsize
  variable ysize 
  variable xcanwindowmax 
  variable ycanwindowmax
  set xcanmax(data) $xsize 
  set ycanmax(data) $ysize
  #make main canvas
  
  canvas $w.can -width [expr $xcanwindowmax] -height $ycanwindowmax -bg #E9E9D9 -xscrollcommand "$w.xs set" -yscrollcommand "$w.ys set" -scrollregion  "0 0 $xcanmax(data) $ycanmax(data)" 
  #uncomment to enable auto-resizing
  #canvas $w.can -width [expr $xcanwindowmax] -height $ycanwindowmax -bg #E9E9D9 -xscrollcommand [namespace code dataCanvasScrollX] -yscrollcommand [namespace code dataCanvasScrollY]  -scrollregion  "0 0 $xcanmax(data) $ycanmax(data)" 
#  canvas $w.vertScale -width $xcanmax(vert) -height $ycanwindowmax -bg #C0D0C0 -yscrollcommand "$w.ys set" -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)" 

  # w.horzScale is the region on the bottom that displays the sequences
  canvas $w.horzScale -width $xcanwindowmax -height  $ycanmax(horz) -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)" -bg #A9A9A9 -xscrollcommand "$w.xs set"
# jfkdsjafksjdfksja
  pack $w.horzScale  -in $w.cfr -side top -anchor s -expand yes -fill both
  pack $w.xs -in $w.cfr -side top -anchor sw -fill x
  #now place the vertical (y) scrollbar
  place $w.ys -in $w.horzScale -relheight 1.0 -relx 0.0 -rely 0.5 -bordermode outside -anchor e 
  # may need to specify B1-presses shift/nonshift separately...
  bind $w.can <ButtonPress-1>  [namespace code {getStartedMarquee %x %y 0 data}]
  bind $w.can <Shift-ButtonPress-1>  [namespace code {getStartedMarquee %x %y 1 data}]
  bind $w.can <ButtonPress-2>  [namespace code {showPair %x %y 0 data}]
  bind $w.can <B1-Motion>  [namespace code {keepMovingMarquee %x %y data}]
  bind $w.can <B2-Motion>  [namespace code {showPair %x %y 0 data}]
  bind $w.can <ButtonRelease-1> [namespace code {letGoMarquee %x %y data}]
  bind $w.horzScale <ButtonPress-1>  [namespace code {getStartedMarquee %x %y 0 horz}]
  bind $w.horzScale <Shift-ButtonPress-1>  [namespace code {getStartedMarquee %x %y 1 horz}]
  bind $w.horzScale <ButtonPress-2>  [namespace code {showPair %x %y 0 horz}]
  bind $w.horzScale <B1-Motion>  [namespace code {keepMovingMarquee %x %y horz}]
  bind $w.horzScale <F12><B2-Motion>  [namespace code {showPair %x %y 0 horz}]
  bind $w.horzScale <ButtonRelease-1> [namespace code {letGoMarquee %x %y horz}]
  return
} 


proc ::vmd_aligntool::reconfigureCanvas {} {
  variable xcanmax
  variable ycanmax
  variable w
  variable ysize 
  variable xcanwindowmax 
  variable ycanwindowmax
  variable xcanwindowStarting
  variable xcanwindowmax 
  variable firstData
  variable xcol

  #in future, add to xcanwindowstarting if we widen window
  set xcanwindowmax  $xcanwindowStarting 


  #check if can cause trouble if no mol loaded...
  $w.can configure  -height $ycanwindowmax -width $xcanwindowmax 
  $w.horzScale configure  -height $ycanmax(horz) -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"

  #$w.vertScale configure  -width $xcanmax(vert) -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)" 
  $w.horzScale delete all
  #$w.vertScale delete all
  $w.can delete all

}

proc ::vmd_aligntool::draw_scale_highlight {} {

  variable w 
  variable dataValX 
  variable dataValXNum 
  variable xcol 
  variable ytopmargin 
  variable scalex
  variable scaley
  variable seqHeight 
  variable resWidth 
  variable rep 
  variable bond_rad 
  variable bond_res
  variable rectCreated
  variable alignIndex 
  variable xleftmargin
  variable numAlign

  # FIX - use vmd's build in color tool to match gui colors w/ vmd colors
  puts "starting draw scale hilight   alignments=$alignIndex(alignments)  alignNum= $alignIndex(num)"
  for {set i 0} {$i<$numAlign} {incr i} { 
    set red 249
    set green [expr 249-(249*$i/$numAlign)]
    set blue [expr (249*$i/$numAlign)]

    set j 0
    foreach colorY [colorinfo colors] {
      lassign [colorinfo rgb $colorY] r g b
      #puts "$j $colorY   \{$r $g $b\}"
      set redX($j) [expr int($r*255)]
      set greenX($j) [expr int($g*255)]
      set blueX($j) [expr int($b*255)]
      incr j
    }

    #convert red blue green 0 - 255 to hex
    set hexred     [format "%02x" $redX($i)]
    set hexgreen   [format "%02x" $greenX($i)]
    set hexblue    [format "%02x" $blueX($i)]
    set colorstring($i)  "\#${hexred}${hexgreen}${hexblue}" 
  }

  $w.horzScale delete trajHighlight 
  for {set i 0} {$i<$alignIndex(alignments)} {incr i} {
    set yposTop [expr -1+ $ytopmargin+   $i *1.5*$seqHeight]
    set yposBot [expr -1+ $ytopmargin+  ($i+1) *1.5*$seqHeight]
    for {set j 0} {$j < $alignIndex(num) } {incr j} { 
      #puts "alignIndex($i,$j,picked)= $alignIndex($i,$j,picked)"
      if  {$alignIndex($i,$j,picked) == 1} {
        #set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
        

        ###draw highlight only if not yet drawn -- if rectCreated is 0, we may just cleared the rects
        ###     to redraw free of accumulated scaling errors
        ###if {($dataValX(0,pickedId,$i) == "null") || ($rectCreated == 0)} 
        
        #always draw trajBox
        #after prototype, merge this with normal highlight draw method
        #puts "drawing trajBox for alignment= $i   alignPos= $j, x from  [expr $xleftmargin + $j * $resWidth] to  [expr $xleftmargin + ($j +1) * $resWidth], yposTop= $yposTop, yposBot= $yposBot"

        set trajBox [$w.horzScale create rectangle  [expr $xleftmargin + $scalex* ($j-.5) * $resWidth]  $yposTop [expr $xleftmargin + $scalex * ($j +.5) * $resWidth] $yposBot  -fill $colorstring($i) -outline "" -tags [list dataScalable trajHighlight ] ]
        $w.horzScale lower dataScalable horzScaleText
        
        
        
        
      }
    }
  }
  
  #make structure highlights (later, will loop through all alignments)
  for {set i 0} {$i<$numAlign} {incr i} {
    set text($i) [makeAlignSelText dataValX $i $dataValXNum($i) 1 $i]
  }

  for {set i 0} {$i<$numAlign} {incr i} {
    set color($i) $i
  }
  for {set a 0} {$a<$numAlign} {incr a} {
    puts "for alignment $a, text is >$text($a), color= $color($a)<"
    set mol $alignIndex($a,currentMol)
    
    if {($rep(0,$mol) != "null")} {

      if { [expr [molinfo $mol get numreps] -1] >= $rep(0,$mol) } {

        mol modselect $rep(0,$mol) $mol $text($a)
      } else {
        createHighlight  rep 0 alignIndex($a,currentMol)  $text($a) $color($a) 
      }
    } else {
      createHighlight  rep 0 alignIndex($a,currentMol)  $text($a) $color($a) 
      #mol selection $ll
      #mol modstyle $rep(0,$currentMolX(0))  $currentMolX(0) Bonds $bond_rad $bond_res
      #mol color ColorID 11 
      #get info about this
    }
  }
}

proc ::vmd_aligntool::drawVertHighlight  {} {

  variable w 
  variable dataValX 
  variable dataValXNum 
  variable xcol 
  variable ytopmargin 
  variable scaley
  variable ybox  
  variable currentMolX 
  variable rep 
  variable bond_rad 
  variable bond_res
  variable rectCreated
  variable vertHighLeft
  variable vertHighRight

  set red 255
  set green 0
  set blue 255
  #convert red blue green 0 - 255 to hex
  set hexred     [format "%02x" $red]
  set hexgreen   [format "%02x" $green]
  set hexblue    [format "%02x" $blue]
  set highlightColorString    "\#${hexred}${hexgreen}${hexblue}" 

  for {set i 0} {$i<=$dataValXNum(0)} {incr i} {
    if  {$dataValX(0,picked,$i) == 1} {
      set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
      
      
      #draw highlight only if not yet drawn -- if rectCreated is 0, we may  just cleared the rects
      #     to redraw free of accumulated scaling errors
      if {($dataValX(0,pickedId,$i) == "null") || ($rectCreated == 0)} {

        set dataValX(0,pickedId,$i)  [$w.vertScale create rectangle  $vertHighLeft $ypos $vertHighRight [expr $ypos + ($scaley * $ybox)]  -fill $highlightColorString -outline "" -tags yScalable]
        
        
        $w.vertScale lower $dataValX(0,pickedId,$i) vertScaleText 
        
      }
      
    }
  }

  set ll [makeSelText dataValX 0 $dataValXNum(0) 1]

  
  if {($rep(0,$currentMolX(0)) != "null")} {

    if { [expr [molinfo $currentMolX(0) get numreps] -1] >= $rep(0,$currentMolX(0)) } {

      mol modselect $rep(0,$currentMolX(0)) $currentMolX(0) $ll
    } else {
      createHighlight  rep 0 currentMolX(0)  $ll 11        
    }
  } else {
    createHighlight  rep 0 currentMolX(0) $ll 11        
    #mol selection $ll
    #mol modstyle $rep(0,$currentMolX(0))  $currentMolX(0) Bonds $bond_rad $bond_res
    #mol color ColorID 11 
    #get info about this
  }
  return
}


proc ::vmd_aligntool::list_pick {name element op} {
  
  global vmd_pick_atom 
  global vmd_pick_mol 
  global vmd_pick_shift_state  

  variable w 
  variable xcanmax
  variable ycanmax
  variable xcanwindowmax 
  variable ycanwindowmax
  variable ybox
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip 
  variable scaley 
  variable dataValX 
  variable dataValXNum 
  variable dataName 
  variable dataNameLast 
  variable bond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable ysize 
  # get the coordinates



  #later deal with top (and rep)  etc. for multi-mol use
  #don't implement yet


  return

  
  if {$vmd_pick_mol == $currentMolX(0)} {
    
    set sel [atomselect $currentMolX(0) "index $vmd_pick_atom"]
    
    set pickedresid [lindex [$sel get {resid}] 0] 
    set pickedchain  [lindex [$sel get {chain}] 0] 
    set pickedresname [lindex  [$sel get {resname}] 0]
    
    
    set pickedOne -1
    for {set i 0} {$i <= $dataValXNum(0)} {incr i} {
      
      if {($dataValX(0,0,$i) == $pickedresid) && ($dataValX(0,1,$i) == $pickedresname) &&  ($dataValX(0,2,$i) == $pickedchain)} {
        set pickedOne $i
        
        break
      }
    }
    
    if {$pickedOne >= 0} {
      set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
      
      #do bitwise AND to check for shift-key bit
      if {$vmd_pick_shift_state & 1} {
        set shiftPressed 1
      } else {
        set shiftPressed 0
      }
      

      
      if {$shiftPressed == 0 } {
        #delete all from canvas

        for {set i 0} {$i <= $dataValXNum(0)} {incr i} {
          set dataValX(0,picked,$i) 0
          if {$dataValX(0,pickedId,$i) != "null"} {
            $w.can delete $dataValX(0,pickedId,$i)
            set dataValX(0,pickedId,$i) "null"
          }
        }
      }
      
      
      set dataValX(0,picked,$pickedOne) 1
      
      #drawVertHighlight 
      
      #scroll to picked
      set center [expr $ytopmargin + ($ybox * $scaley * $pickedOne) ] 
      set top [expr $center - 0.5 * $ycanwindowmax]
      
      if {$top < 0} {
        set top 0
      }
      set yfrac [expr $top / $ysize]
      $w.can yview moveto $yfrac
    }
    
  }
  return
}


proc ::vmd_aligntool::findResData {cMol cMol_name dVal dataValIndex dValNum dHash selectionText ind} {            
  
  variable rep

  upvar $cMol currentMol $cMol_name currentMol_name $dVal dataVal $dValNum dataValNum $dHash dataHash 
  set dataValNum -1 
  puts "in findResData"
  puts "getting data for mol=$currentMol"
  
  set currentMol_name [molinfo $currentMol get name]
  puts "name=$currentMol_name"
  set sel [atomselect $currentMol "$selectionText and name CA"]
  #below assumes sel retrievals in same order each time, fix this 
  #by changing to one retreival and chopping up result
  set datalist  [$sel get {resid resname chain}]
  puts "Checking sequence info. for molecule $currentMol..."
  
  #clear this dataHash
  #FIX ?
#  catch {unset dataHash}
  #the upvar'd dataHash is set again below, and reappears in the namespace.        
  foreach elem $datalist {
    
    #set picked state to false -- 'picked' is only non-numerical field
    incr dataValNum
    set dataVal($dataValIndex,picked,$dataValNum) 0
    set dataVal($dataValIndex,pickedId,$dataValNum) "null"
    set theResid [ lindex [split $elem] 0]
    set dataVal($dataValIndex,0,$dataValNum) $theResid 
    
    set dataVal($dataValIndex,1,$dataValNum) [ lindex [split $elem] 1]
    set dataVal($dataValIndex,1code,$dataValNum) [lookupCode $dataVal($dataValIndex,1,$dataValNum)]
    set theChain [ lindex [split $elem] 2]
    set dataVal($dataValIndex,2,$dataValNum) $theChain 
    #for fast index searching later
    set dataHash($ind,$theResid,$theChain) $dataValNum
  }
  #if datalist is length 0 (empty), dataValNum is still -1, 
  #So we must check before each use of dataValNum          
  
  #set the molec. structure so nothing is highlighted yet
  set rep(0,$currentMol) "null"
  
  
  if {$dataValNum <= -1 } {
    #puts "Couldn't find a sequence in this molecule.\n"
    return
  }
  unset datalist
  puts "dataValNum = $dataValNum"
} 

proc ::vmd_aligntool::zoomSeqMain {} {
  #------------------------
  #------------------------
  # main code starts here
  #vars initialized a few lines down
  #puts "in zoomSeqMain.."
  variable w 
  variable monoFont
  variable eo 
  variable x1 
  variable y1 
  variable startShiftPressed 
  variable startCanvas
  variable vmd_pick_shift_state 
  variable amino_code_toggle 
  variable bond_rad 
  variable bond_res
  variable so 
  variable xcanwindowmax 
  variable ycanwindowmax 
  variable xcanmax 
  variable ycanmax
  variable ybox 
  variable ysize 
  variable resnamelist 
  variable structlist 
  variable betalist 
  variable sel 
  variable canvasnew 
  variable scaley 
  variable dataVal
  variable dataValX 
  variable dataHashX
  variable rectId
  #dataValXNum(0) is -1 if no data present, 
  variable dataValXNum 
  variable dataName 
  variable dataNameLast 
  variable ytopmargin 
  variable ybottommargin 
  variable xleftmargin
  variable xrightmargin
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable amino_code_toggle 
  variable dataWidth 
  variable resWidth
  variable dataMargin 
  variable firstData 
  variable dataMin 
  variable dataMax 
  variable xPosScaleVal
  variable fit_scalex
  variable fit_scaley
  variable usableMolLoaded 
  variable initializedVars
  variable prevScalet
  variable rectCreated
  variable windowShowing
  variable needsDataUpdate 
  variable numHoriz
  variable selTextX
  variable pairVal
  variable alignIndex
  variable numAlign

  #if there are no mols at all,
  #there certainly aren't any non-graphics mols
  if {[molinfo num] ==0} {
    set usableMolLoaded 0
  }

  #Init vars and draw interface
  if {$initializedVars == 0} {
    initVars
    draw_interface
    makecanvas
    set initializedVars 1
    #watch the slider value, tells us when to redraw
    #this sets a trace for ::vmd_aligntool::scaley
    
  } else {
    #even if no molecule is present
    reconfigureCanvas
  }   
  
  
  #-----
  #Now load info from the current molecule, must reload for every molecule change
  
  if {$usableMolLoaded} {
    #get info for new mol
    #set needsDataUpdate 0

    set dataNameLast 2
    #The number of dataNames
    
    #lets fill  a (dataNameLast +1 ) x (dataValXNum(0) +1) array
    #dataValXNum(0) we'll be the number of objects we found with VMD search
    #if doing proteins, liekly all residues, found with 'name CA'

    for {set i 0} {$i<$numAlign} {incr i} {
      set dataValXNum($i) -1
    }
    set alignIndex(num) 0
    #if no data is available, dataValXNum(0) will remain -1        

    
    # set  a new  trace below, only if dataValXNum(0) > -1        
    for {set i 0} {$i<$numAlign} {incr i} {
      findResData alignIndex($i,currentMol) currentMolX_name($i) dataValX $i dataValXNum($i) dataHashX $selTextX($i) $i
    }
    set tempTit "VMD Align "
    for {set i 0} {$i < $numAlign} {incr i} {
      append tempTit " $currentMolX_name($i) (mol $alignIndex($i,currentMol)) "
    }
      wm title $w $tempTit
    #So dataValXNum(0) is number of the last dataValX 0.  It is also #elements -1, 
    
    for {set i 0} {$i<$numAlign} {incr i} {
      puts "dataValXNum($i) = $dataValXNum($i)"
    }
    #numHoriz (and routines that use it)  will eventualy be changed
    # to reflect loaded data, and  multi-frame-data groups
    set numHoriz $dataValXNum(1) 
    if {$dataValXNum(0) <0} {
      tk_messageBox -title "File Error" -type ok -message "Error in file.  No C-alpha atoms found." -parent $w
      return
    }
    
    set fit_scalex [expr (0.0 + $xcanwindowmax - $xleftmargin ) / ( $resWidth *  (1+ $alignIndex(num)) ) ]
    set fit_scaley [expr (0.0 + $ycanwindowmax - $ytopmargin - $ybottommargin) / ($ybox * ($dataValXNum(0) + 1) ) ]
    #since we zero-count.

    set scaley 1.0
    set scalex $fit_scalex 
    puts "Restarting data, scalex = $scalex"
    #this trace only set if dataValXNum(0) != -1

    #Other variable-adding methods
    #should not change this number.  We trust $selA to always
    #give dataValXNum(0) elems, other methods might not work as well.
    #if need this data, recreate sel.        
    
    #handle if this value is 0 or -1
    
    #now lets fill in some data
    #new data, so need to redraw rects when time comes
    set rectCreated 0 
    #also set revScaley back to 1 
    set prevScaley scaley
    set prevScalex scalex 
    #fill in betalist (B-factors/temp factors called beta by VMD)
    #incr dataNameLast
    #set betalist [$selA get beta]
    #set dataName($dataNameLast) "B value"
    #jset dataMin($dataNameLast) 0.0
    #set dataMax($dataNameLast) 150.0
    #set i 0
    #foreach elem $betalist {
    #    set dataValX(0,$dataNameLast,$i) $elem
    #    incr i
    #}
    
    ## Now there are 4 dataNames,   current
    ##value of dataNameNum is 3. last is numbered (dataNameLast) = 3
    #unset  betalist ;#done with it


    #fill in traj data with res-res position 
    
    #calcDataDist
    #puts "after initial calcDataDist, dataNameLast= $dataNameLast, dataValX(0,4,0)= $dataValX(0,4,0)  dataValX(0,8,8)= $dataValX(0,8,8)"
    



    #lets add another data field, just to test...
    #incr dataNameLast

    #set test_pos_list [$selA get y]
    #set dataName($dataNameLast) "test Y"
    #set dataMin($dataNameLast) 0 
    #set dataMax($dataNameLast) 10 
    #set i 0
    #foreach elem $test_pos_list {
    #    set dataValX(0,$dataNameLast,$i)  $elem
    #    incr i
    #}
    #unset test_pos_list



    
  }
  #get min/max of some data
  #puts "time for first redraw, scales, min/max not calced"
  #redraw first time
  redraw name func ops
  
  #now draw the scales (after the data, we may need to extract min/max 

  #------
  #draw color legends, loop over all data fields
  set fieldLast  $dataNameLast
  
  #temporary, until put multi-cols in
  #for {set field $firstData} {$field <= $fieldLast} {incr field} 
  #hack so only display first 2 scales
  if {1==0} {
    puts "now to draw scales"
    for {set field $firstData} {$field <= 4} {incr field} {
      
      set xPosField [expr int ($xcol($firstData) + ($dataWidth * ($field - $firstData) ) )]
      puts "xPosField= $xPosField"
      #print the the title in center of data rectangle width
      $w.horzScale create text [expr int($xPosField + ( ($dataWidth -$dataMargin)/ 2.0) )] 1 -text "$dataName($field)" -width 200 -font $monoFont -justify center -anchor n 
      
      
      
      #make a scale across data rectange width

      set size [expr $dataWidth - $dataMargin]
      if {$dataName($field) != "struct"} {
        set minString [format "%.3g" $dataMin($field)]
        set maxString [format "%.3g" $dataMax($field)]
        $w.horzScale create text [expr $xPosField - 2  ] $xPosScaleVal -text $minString -width 50 -font $monoFont -justify center -anchor nw

        $w.horzScale create text [expr int ($xPosField - $dataMargin + $dataWidth +2 )] $xPosScaleVal -text $maxString -width 50 -font $monoFont -justify center -anchor ne
        
        set range [expr $dataMax($field) - $dataMin($field)]
        #bounds check, should really print error message
        if {$range <= 0} {
          puts "Bad range for field $dataName($field), min= $dataMin($field) max= $dataMax($field), range = $range"
          set $dataMin($field) -100
          set $dataMax($field) 100
          set $range [expr $dataMax($field) - $dataMin($field)]
          puts "Reset range for $dataName($field), new values: min= $dataMin($field) max= $dataMax($field), range = $range"
        }
        
        for {set yrect 0} {$yrect < $size} {incr yrect} {
          
          #draw linear scale
          set val [expr ( ( 0.0+ $yrect  )/ ($size -1)) * 255]
          #puts "val = $val , range = $range"
          set hexcols [chooseColor $field $val]                     
          
          set hexred [lindex $hexcols 0]
          set hexgreen [lindex $hexcols 1]
          set hexblue [lindex $hexcols 2]
          

          $w.horzScale create rectangle [expr $xPosField + $yrect] 15 [expr $xPosField + $yrect] 30 -fill  "\#${hexred}${hexgreen}${hexblue}" -outline ""
        }
      } else {
        
        set prevNameIndex -1
        for {set yrect 0} {$yrect < $size} {incr yrect} {
          set names [list T E B H G I C "other"]
          
          set nameIndex [expr int ([expr [llength $names] -1]  * ($yrect+0.0)/$size)]
          set curName [lindex $names  $nameIndex]
          
          if {$nameIndex != $prevNameIndex} {
            #set line to black
            set hexred 0
            set hexgreen 0
            set hexblue 0
            
            #draw text
            $w.horzScale create text [expr int ($xPosField + $yrect+ 3)] $xPosScaleVal -text $curName -width 20 -font $monoFont -justify left -anchor nw
          } else {
            
            
            
            set hexcols [chooseColor $field $curName]
            
            set hexred [lindex $hexcols 0]
            set hexgreen [lindex $hexcols 1]
            set hexblue [lindex $hexcols 2]
          }

          $w.horzScale create rectangle [expr $xPosField + $yrect] 15 [expr $xPosField + $yrect] 30 -fill  "\#${hexred}${hexgreen}${hexblue}" -outline ""
          set prevNameIndex $nameIndex
        }
        set hexred 0
        set hexgreen 0
        set hexblue 0
        $w.horzScale create rectangle [expr $xPosField + $yrect] 15 [expr $xPosField + $size] 30 -fill  "\#${hexred}${hexgreen}${hexblue}" -outline ""
      }
    }        
    #done with color legends
    #-------
    
  } 
  

  return
}


proc ::vmd_aligntool::molChooseMenu {name function op} {
  variable w

  variable usableMolLoaded
  variable alignIndex
  variable prevMolX
  variable nullMolString
  variable numAlign
  variable rmsdSelA 0
  variable rmsdSelB 1
  # FIX
  $w.molA.menu delete 0 end
  $w.molB.menu delete 0 end
#  $w.molC.menu delete 0 end
  $w.rmsdA.menu delete 0 end
  $w.rmsdB.menu delete 0 end



  set molList ""
  foreach mm [molinfo list] {
    if {[molinfo $mm get filetype] != "graphics"} {
      lappend molList $mm
      #add a radiobutton, but control via commands, not trace,
      #since if this used a trace, the trace's callback
      #would delete that trace var, causing app to crash.
      #variable and value only for easy button lighting
      $w.molA.menu add radiobutton -variable [namespace current]::alignIndex(0,currentMol) -value $mm -label "$mm [molinfo $mm get name]"
      $w.molB.menu add radiobutton -variable [namespace current]::alignIndex(1,currentMol) -value $mm -label "$mm [molinfo $mm get name]"
#      $w.molC.menu add radiobutton -variable [namespace current]::alignIndex(2,currentMol) -value $mm -label "$mm [molinfo $mm get name]"
      $w.rmsdA.menu add radiobutton -variable [namespace current]::rmsdSelA -value $mm -label "$mm [molinfo $mm get name]"
      $w.rmsdB.menu add radiobutton -variable [namespace current]::rmsdSelB -value $mm -label "$mm [molinfo $mm get name]"
    }
  }

  #set if any non-Graphics molecule is loaded
  if {$molList == ""} {
    set usableMolLoaded  0
    for {set index 0} {$index<$numAlign} {incr index} {
      if {$prevMolX($index) != $nullMolString} {
        set alignIndex($index,currentMol) $nullMolString
      }
    }
  } else {

    #deal with first (or from-no mol state) mol load
    # and, deal with deletion of currentMolX(0), if mols present
    # by setting the current mol to whatever top is
    for {set index 0} {$index < $numAlign} {incr index} {
      if {($usableMolLoaded == 0) || [lsearch -exact $molList $alignIndex($index,currentMol)]== -1 } {
        set usableMolLoaded 1
        set alignIndex($index,currentMol) [molinfo top]
      }
    }

  }


  
  
  return
}

proc ::vmd_aligntool::setScaling {} {
  variable w
  variable trajMin
  variable trajMax 
  #extract first part of molecule name to print here?
  puts "Enter bottom value of scale..."
  set trajMin  [gets stdin]
  puts "Enter top value of scale..."
  set trajMax [gets stdin]
  puts "trajMin = $trajMin, trajMax= $trajMax" 
  return
}

proc ::vmd_aligntool::printCanvas {} {
  variable w
  #extract first part of molecule name to print here?
  set filename "VMD_Aligntool_Window.ps"
  set filename [tk_getSaveFile -initialfile $filename -title "VMD Aligntool Print" -parent $w -filetypes [list {{Postscript Files} {.ps}} {{All files} {*} }] ]
  if {$filename != ""} {
    $w.can postscript -file $filename
  }
  
  return
}





proc ::vmd_aligntool::getStartedMarquee {x y shiftState whichCanvas} {

  variable w 
  variable x1 
  variable y1 
  variable so
  variable str 
  variable eo 
  variable g 
  variable startCanvas 
  variable startShiftPressed
  variable xcanmax
  variable ycanmax
  variable usableMolLoaded

  
  
  if {$usableMolLoaded} {

    #calculate offset for canvas scroll
    set startShiftPressed $shiftState        
    set startCanvas $whichCanvas 
    #get actual name of canvas
    switch -exact $startCanvas {
      data {set drawCan can}
      horz {set drawCan horzScale}
      default {puts "problem with finding canvas..., startCanvas= >$startCanvas<"} 
    }         
    set x [expr $x + $xcanmax($startCanvas) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax($startCanvas) * [lindex [$w.$drawCan yview] 0]] 
    
    set x1 $x
    set y1 $y
    

    puts "getStartedMarquee x= $x  y= $y, startCanvas= $startCanvas" 
    #Might have other canvas tools in future..         
    # Otherwise, start drawing rectangle for selection marquee
    
    set so [$w.$drawCan create rectangle $x $y $x $y -fill {} -outline red]
    set eo $so
  } 
  return
}


proc ::vmd_aligntool::molChooseX {index name function op} {
  variable scaley
  variable w
  variable prevMolX
  variable nullMolString
  variable rep 
  variable usableMolLoaded
  variable needsDataUpdate
  variable windowShowing
  variable alignIndex
  variable numAlign
  #this does complete restart
  #can do this more gently...
  
  #trace vdelete scaley w [namespace code redraw]
  #trace vdelete ::vmd_pick_event w  [namespace code list_pick] 
  
  #if there's a mol loaded, and there was an actual non-graphic mol last
  #time, and if there has been a selection, and thus a struct highlight
  #rep made, delete the highlight rep.
  if {($usableMolLoaded)  && ($prevMolX($index) != $nullMolString) && ($rep(0,$prevMolX($index)) != "null")} {
    #catch this since currently is exposed to user, so 
    #switching/reselecting  molecules can fix problems.
    ##puts "About to delete rep=$rep(0,$prevMolX($index)) for prevMolX($index)= $prevMolX($index)"
    #determine if this mol exists...
    if  {[lsearch -exact [molinfo list] $prevMolX($index)] != -1}  {
      #determine if this rep exists (may have been deleted by user)
      if { [expr [molinfo $prevMolX($index) get numreps] -1] >= $rep(0,$prevMolX($index)) } { 
        
        mol delrep $rep(0,$prevMolX($index)) $prevMolX($index) 
      }
    }
  }

  for {set index 0} {$index < $numAlign} {incr index} {
    set prevMolX($index) $alignIndex($index,currentMol) 
  }

  #can get here when window is not displayed if:
  #   molecule is loaded, other molecule delete via Molecule GUI form.
  # So, we'll only redraw (and possible make a length (wallclock) call
  # to STRIDE) if sequence window is showing
  
  set needsDataUpdate 1


  if {$windowShowing} {
    set needsDataUpdate 0
    #set this immediately, so other  calls can see this
    
    [namespace current]::zoomSeqMain
  }

  
  #reload/redraw stuff, settings (this may elim. need for above lines...)
  #change molecule choice and redraw if needed (visible && change) here...
  #change title of window as well
  ##wm title $w "VMD Aligntool  $currentMolX_name($index) (mol $currentMolX($index)) "
  
  #reload sutff (this may elim. need for above lines...)

  return
}

proc ::vmd_aligntool::keepMovingMarquee {x y whichCanvas} {

  variable x1 
  variable y1 
  variable so 
  variable w 
  variable xcanmax 
  variable ycanmax
  variable startCanvas
  variable usableMolLoaded
  #get actual name of canbas
  switch -exact $startCanvas {
    data {set drawCan can}
    horz {set drawCan horzScale}
    default {puts "problem with finding canvas (moving marquee)..., startCanvas= $startCanvas"}
  } 

  
  if {$usableMolLoaded} {

    #next two lines for debeugging only
    set windowx $x
    set windowy $y 
    #calculate offset for canvas scroll
    set x [expr $x + $xcanmax($startCanvas) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax($startCanvas) * [lindex [$w.$drawCan yview] 0]] 
    
    
    
    
    $w.$drawCan coords $so $x1 $y1 $x $y
  }
  return
}

proc ::vmd_aligntool::letGoMarquee {x y whichCanvas} {


  variable x1 
  variable y1 
  variable startShiftPressed 
  variable startCanvas
  variable so 
  variable eo 
  variable w 
  variable xcanmax
  variable ycanmax
  variable ySelStart 
  variable ySelFinish 
  variable ybox 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip 
  variable scalex 
  variable scaley 
  variable dataValX 
  variable dataValXNum 
  variable dataName 
  variable dataNameLast 
  variable bond_rad 
  variable bond_res 
  variable rep 
  variable xcol
  variable usableMolLoaded
  variable firstData
  variable dataWidth 
  variable ycanwindowmax  
  variable firstStructField
  variable alignIndex
  variable xleftmargin
  variable resWidth
  variable seqHeight
  variable currDist
  variable rmsdSelA
  variable rmsdSelB

  #set actual name of canvas
  switch -exact $startCanvas {
    data {set drawCan can}
    horz {set drawCan horzScale}
    default {puts "problem with finding canvas (moving marquee)..., startCanvas= $startCanvas"}
  }

  if {$usableMolLoaded} {
    #calculate offset for canvas scroll
    set x [expr $x + $xcanmax($startCanvas) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax($startCanvas) * [lindex [$w.$drawCan yview] 0]] 

    #compute the frame at xSelStart
    if {$x1 < $x} {
      set xSelStart $x1
      set xSelFinish $x
    }  else {
      set xSelStart $x
      set xSelFinish $x1
    }
    puts "xcanmax(horz)= $xcanmax(horz),  xSelStart= $xSelStart   xSelFinish= $xSelFinish" 
    
    #in initVars we hardcode firstStructField to be 4
    #later, there may be many field-groups that can be stretched 
    set xPosStructStart [expr int ($xcol($firstData) + ($dataWidth * ($firstStructField - $firstData) ) )] 

    set selStartHoriz [expr  int (($xSelStart - $xleftmargin)/ ($resWidth * $scalex))  ]
    set selFinishHoriz [expr int( ($xSelFinish - $xleftmargin)/ ($resWidth * $scalex) ) ]
    if { $selFinishHoriz> $alignIndex(num)} {
      set selFinishHoriz $alignIndex(num)
    }

    if {$y1 < $y} {
      set ySelStart $y1
      set ySelFinish $y}  else {
        
        set ySelStart $y
        set ySelFinish $y1
      }
    
    set startObject [expr 0.0 + ((0.0 + $ySelStart - $ytopmargin) / ( 1.5*$seqHeight))]
    set finishObject [expr 0.0 + ((0.0 + $ySelFinish - $ytopmargin) / (  1.5*$seqHeight))]
    
    if {$startShiftPressed == 1} {
      set singleSel 0
    } else {
      set singleSel 1
    }
    
    if {$startObject < 0} {set startObject 0}
    if {$finishObject < 0} {set finishObject 0}
    if {$startObject > $alignIndex(num)} {set startObject  $alignIndex(num)}
    if {$finishObject > $alignIndex(num)} {set finishObject $alignIndex(num)}
    set startObject [expr int($startObject)]
    set finishObject [expr int($finishObject)]
    puts "selected columns $selStartHoriz to $selFinishHoriz, alignments $startObject to $finishObject" 
   
    puts "singleSel = $singleSel" 
    
    
    #clear all if click/click-drag, don't clear if shift-click, shift-click-drag
    
    if {$singleSel == 1} {
#    puts "inside singleSel ==1 section"
#    puts "alignIndex(alignments) = $alignIndex(alignments)"
      
      for {set i 0} {$i <= $alignIndex(alignments)} {incr i} {
        for {set j 0} {$j <= $alignIndex(num)} {incr j} {
#          puts "i=$i j=$j"
          set alignIndex($i,$j,picked) 0
          #delete via tags for now
          #if {$dataValX(0,pickedId,$i) != "null"} {
          #        
          #        $w.vertScale delete $dataValX(0,pickedId,$i)
          #        set dataValX(0,pickedId,$i) "null"
          # }

        }
      } 
    } 
    
    puts "now to set flags for selection" 
    #set flags for selection
    for {set i $startObject} {$i <= $finishObject} {incr i} {
      for {set j $selStartHoriz} {$j <= $selFinishHoriz} {incr j} {
	puts "set alignIndex($i,$j,picked) 1"
        set alignIndex($i,$j,picked) 1
      }
    }

    set field 0
    #note that the column will be 0, but the data will be from picked
    
    #drawVertHighlight 
    
    
    puts "now to delete outline, eo= $eo" 
    $w.$drawCan delete $eo
    draw_scale_highlight
    
    calculateSelectedRMSD

  }
  return
}

proc ::vmd_aligntool::showall { do_redraw} {
  variable scalex 
  variable scaley 
  variable fit_scalex
  variable fit_scaley
  variable usableMolLoaded
  variable rectCreated 
  variable userScalex
  variable userScaley

  #only redraw once...
  if {$usableMolLoaded} {
    if {$do_redraw == 1} {
      set rectCreated 0
    }        
    
    set scalex $fit_scalex        
    set scaley $fit_scaley
    set userScalex 1.0
    set userScaley $fit_scaley 

    redraw name func ops
  }

  return
}


proc ::vmd_aligntool::every_res {} {
  variable usableMolLoaded
  variable rectCreated
  variable fit_scalex
  #this forces redraw, to cure any scaling floating point errors
  #that have crept in 
  set rectCreated 0
  variable scaley
  variable scalex

  if {$usableMolLoaded} {
    #redraw, set x and y  at once
    set scalex $fit_scalex 
    set userScalex 1.000 
    set scaley 1.0
    redraw name func ops
  }
  
  return
}


proc ::vmd_aligntool::resname_toggle {} {

  variable w 
  variable amino_code_toggle
  variable usableMolLoaded
  
  if {$usableMolLoaded} {


    if {$amino_code_toggle == 0} {
      set amino_code_toggle 1
      $w.resname_toggle configure -text "3-letter code"
    } else {
      set amino_code_toggle 0
      $w.resname_toggle configure -text "1-letter code"
    }
    
    redraw name function op
  }
  return
}




proc ::vmd_aligntool::initVars {} {        

  variable usableMolLoaded 0
  variable windowShowing 0
  variable needsDataUpdate 0
  variable dataNameLast 2
  variable dataValXNum
  variable eo 0
  variable x1 0 
  variable y1 0
  variable startCanvas ""
  variable startShiftPressed 0
  variable vmd_pick_shift_state 0
  variable amino_code_toggle 0
  variable bond_rad 0.5
  variable bond_res 10
  variable so ""
  variable nullMolString ""
  variable prevMolX
  variable prevMolX
  variable numAlign
  for {set i 0} {$i<$numAlign} {incr i} {
    set prevMolX($i) $nullMolString
  }
  variable  userScalex 1
  variable  userScaley 1
  variable  scalex 1
  variable  scaley 1
  variable prevScalex 1
  variable prevScaley 1
  
  variable ytopmargin 46 
  variable ybottommargin 10
  variable xrightmargin 8
  variable xleftmargin 10 
  
  
  #variable xcanwindowStarting 780 
  variable xcanwindowStarting 630 
  variable ycanwindowStarting 40 

  variable numHoriz 1
  variable xcanwindowmax  $xcanwindowStarting
  variable ycanwindowmax $ycanwindowStarting 
  variable xcanmax
  set dataValXNum(0) -1
  set xcanmax(data) 610
#  set xcanmax(vert) 95
  set xcanmax(horz) $xcanmax(data)
  #make this sensible!
  variable ycanmax
  set ycanmax(data) 400
#  set ycanmax(vert) $ycanmax(data) 
  set ycanmax(horz) 136 
  variable codes
  variable trajMin -180
  variable trajMax 180
  variable selTextX
  variable xScaling 0 
  variable yScaling 0 
  #hard coded, should change
  variable firstStructField 4
  variable alignIndex
  variable numAlign
  ##########################################
  #  All alignment data is in alignIndex
  # alignIndex has 1d, 2d, and 3d data structures.
  #We take advantage of flexible tcl hash labeling by mixing numerical index 
  #and 'enumerated' label indexes.
  #   1D data is overall alignment matrix info
  #1D: num = number of entries (columns) in alignment
  #1D: alignments = number of alignments 
  #   2D data is per-alignment info (each alignment refers to one molecule,
  #   not necessarily different ones) 
  #2D: <alignment#>, currentMol = the molid of the molec. which 
  #this alignment describes
  #2D: <alignment#>, transMat = the transformation matrix to align the described
  #molecule in space (makes sense only structrual+sequence alignment
  #programs like CE) 
  #   3D data is per-residue info
  #3D: <alignment#>, <entry#>, distance = distance between two alignments with 
  #same entry# (assumes only two alignments...should be changed later)
  #3D: <alignment#>, <entry#>, ind = index of residue info in this 
  #alignment's molecule's dataVal 
  #3D: <alignment#>, <entry#>, match = boolean, true if this res. is part 
  #of the actual #aligned ("matched" up) sequence
  #3D: <alignment#>, <entry#>, picked = boolean, true if currently picked 
  #by gui for higlighting
  ############################################
  for {set i 0} {$i<$numAlign} {incr i} {
    set selTextX($i) "all"
  }
  set alignIndex(num) 0
  #only 2 alignments hardcoded for now..
  set alignIndex(alignments) $numAlign
  for {set index 0} {$index < $numAlign} {incr index} {
    set alignIndex($index,currentMol) $nullMolString
    set alignIndex($index,transMat) "null"
  }

  array set codes {ALA A ARG R ASN N ASP D ASX B CYS C GLN Q GLU E
    GLX Z GLY G HIS H ILE I LEU L LYS K MET M PHE F PRO P SER S
    THR T TRP W TYR Y VAL V}
  
  
  

  #tests if rects for current mol have been created (should extend 
  #so memorize all rectIds in 3dim array, and track num mols-long 
  #vector of rectCreated. Would hide rects for non-disped molec,
  #and remember to delete the data when molec deleted.
  
  variable rectCreated 0
  #the height of sequence for an aligned molecule
  variable seqHeight 24
  #the box height
  variable ybox 15.0
  #text skip doesn't need to be same as ybox (e.g. if bigger numbers than boxes in 1.0 scale)
  variable vertTextSkip $ybox

  
  # For vertical scale appearance
  variable vertHighLeft 2
  variable vertHighRight 100
  variable vertTextRight 96
  #The first 3 fields, 0 to 2 are printed all together, they are text
  variable xcol
  set xcol(0) 10.0
  
  variable resWidth 10
  variable dataWidth 85
  variable dataMargin 0
  variable xPosScaleVal 32
  #so rectangge of data is drawn at width $dataWidth - $dataMargin (horizontal measures)
  #
  #residue name data is in numbered entries lower than 3
  variable firstData 3
  #puts "firstData is $firstData"
  #column that multi-col data first  appears in

  #old setting from when vertscale and data were on same canvas
  #set xcol($firstData)  96 
  set xcol($firstData)  1 
  #The 4th field (field 3) is the "first data field"
  #we use same data structure for labels and data, but now draw in separate canvases 
  
  # the names for  three fields of data 
  
  #just for self-doc
  # dataValX(0,picked,n) set if the elem is picked
  # dataValX(0,pickedId,n) contains the canvas Id of the elem's highlight rectangle
  

  variable dataName

  set dataName(picked) "picked" 
  set dataName(pickedId) "pickedId"
  #not included in count of # datanames
  
  set dataName(0) "resid"
  set dataName(1) "resname"
  set dataName(1code) "res-code"
  set dataName(2) "chain"
  ###set dataName(3) "check error.." 
  
  
}


proc ::vmd_aligntool::Show {} {
  variable windowShowing
  variable needsDataUpdate
  set windowShowing 1

  
  if {$needsDataUpdate} {
    set needsDataUpdate 0
    #set immmediately, so other binding callbacks will see
    [namespace current]::zoomSeqMain
  }
}

proc ::vmd_aligntool::Hide {} {
  variable windowShowing 
  set windowShowing 0
}



proc ::vmd_aligntool::createHighlight { theRep repInd theCurrentMol seltext color} {
  upvar $theRep rep $theCurrentMol currentMol 
  variable bond_rad
  variable bond_res
  #draw first selection, as first residue 
  set rep($repInd,$currentMol) [molinfo $currentMol get numreps]
  mol selection $seltext
  mol material Opaque
  mol color ColorID $color 
  mol addrep $currentMol
  mol modstyle $rep($repInd,$currentMol)  $currentMol  Bonds $bond_rad $bond_res
}



proc ::vmd_aligntool::draw_interface {} {
  variable w 
  variable eo 
  variable x1  
  variable y1 
  variable startCanvas
  variable startShiftPressed 
  variable vmd_pick_shift_state 
  variable amino_code_toggle 
  variable bond_rad 
  variable bond_res
  variable so 
  variable xcanwindowmax 
  variable ycanwindowmax 
  variable xcanmax 
  variable ycanmax
  variable ybox 
  variable xsize 
  variable ysize 
  variable resnamelist 
  variable structlist 
  variable betalist 
  variable sel 
  variable canvasnew 
  variable userScalex
  variable userScaley
  variable scalex 
  variable scaley 
  variable dataValX 
  variable dataValXNum 
  variable dataName 
  variable dataNameLast 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable repPairX
  variable xcol 
  variable amino_code_toggle 
  variable dataWidth 
  variable dataMargin 
  variable firstData 
  variable dataMin 
  variable dataMax 
  variable xPosScaleVal
  variable fit_scalex
  variable fit_scaley
  variable usableMolLoaded 
  variable numHoriz 
  variable selTextX
  variable pairVal
  variable alignIndex
  variable currDist
  variable numAlign
  variable alignChain
  variable rmsdSelA
  variable rmsdSelB
  frame $w.menubar -height 30 -relief raised -bd 2
  pack $w.menubar -in $w -side top -anchor nw -padx 1 -fill x
  #frame $w.fr -width 700 -height 810 -bg #FFFFFF -bd 2 ;#main frame
  #pack $w.fr

  label $w.txtlab -text "Zoom "

  frame $w.panl -width 850 -height [expr $ycanwindowmax + 80] -bg #C0C0D0 -relief raised -bd 1 
  frame $w.cfr -width 850 -height [expr $ycanwindowmax + 85] -borderwidth 1  -bg #000000 -relief raised -bd 3
  pack $w.panl -in $w -side top -padx 2  -fill y
  #pack $w.cfr -in $w.fr -side left -padx 2 -expand yes -fill both 
  pack $w.cfr -in $w -side left -padx 2 -expand yes -fill both 

  #scale $w.panl.zoomlevel -from 0.01 -to 2.01 -length 150 -sliderlength 30  -resolution 0.01 -tickinterval 0.5 -repeatinterval 30 -showvalue true -variable [namespace current]::userScaley -command [namespace code userScaleyChanged] 

  scale $w.zoomBothlevel -orient horizontal -from 0.001 -to 4.000 -length 120 -sliderlength 30  -resolution 0.001 -tickinterval 3.999 -repeatinterval 30 -showvalue true -variable [namespace current]::userScaleBoth -command [namespace code userScaleBothChanged] 
  scale $w.zoomXlevel -orient horizontal -from 0.001 -to 0.2 -length 120 -sliderlength 30  -resolution 0.001 -tickinterval 0.199 -repeatinterval 30 -showvalue true -variable [namespace current]::userScalex -command [namespace code userScalexChanged] 

  #pack $w.panl $w.cfr -in $w.fr -side left -padx 2

  button $w.showall  -text "fit all" -command [namespace code {showall 0}]
  button $w.every_res  -text "every residue" -command [namespace code every_res]
  button $w.resname_toggle  -text "1-letter code" -command [namespace code resname_toggle]
  button $w.rmsdCalc  -text "Calc RMSD" -command [namespace code {calcResResRMSD $rmsdSelA $rmsdSelB}]

  #draw canvas
  
  #trace for molecule choosing popup menu 
  trace variable ::vmd_initialize_structure w  [namespace code molChooseMenu]
 
  #FIX
  set index 0  
  menubutton $w.molA -relief raised -bd 2 -textvariable [namespace current]::alignIndex(0,currentMol) -direction flush -menu $w.molA.menu
  set index 1
  menubutton $w.molB -relief raised -bd 2 -textvariable [namespace current]::alignIndex(1,currentMol) -direction flush -menu $w.molB.menu
#  menubutton $w.molC -relief raised -bd 2 -textvariable [namespace current]::alignIndex(2,currentMol) -direction flush -menu $w.molC.menu
  menubutton $w.rmsdA -relief raised -bd 2 -textvariable [namespace current]::rmsdSelA -direction flush -menu $w.rmsdA.menu
  menubutton $w.rmsdB -relief raised -bd 2 -textvariable [namespace current]::rmsdSelB -direction flush -menu $w.rmsdB.menu
  menu $w.molA.menu
  menu $w.molB.menu
#  menu $w.molC.menu
  menu $w.rmsdA.menu
  menu $w.rmsdB.menu

  molChooseMenu name function op
  #FIX
  entry $w.molAalignChain -width 1 -text "" -textvariable [namespace current]::alignChain(0)
  entry $w.molBalignChain -width 1 -text "" -textvariable [namespace current]::alignChain(1)
#  entry $w.molCalignChain -width 1 -text "" -textvariable [namespace current]::alignChain(2)
  entry $w.molASeltext -text "all" -textvariable [namespace current]::selTextX(0)
  entry $w.molBSeltext -text "all" -textvariable [namespace current]::selTextX(1)
#  entry $w.molCSeltext -text "all" -textvariable [namespace current]::selTextX(2)
  entry $w.pairval -text "" -textvariable [namespace current]::pairVal
  label $w.molLabA -text "MoleculeA:"
  label $w.molLabChainA -text "Chain:"
  label $w.molLabB -text "MoleculeB:"
  label $w.molLabChainB -text "Chain:"
#  label $w.molLabC -text "MoleculeC:"
  label $w.resDistLab -text "RMSD:"
  entry $w.resDistVal -text "0.0" -textvariable [namespace current]::currDist

  scrollbar $w.ys -orient vertical -command [namespace code {canvasScrollY}]
  
  scrollbar $w.xs -orient horizontal -command [namespace code {canvasScrollX}]

  #fill the  top menu
  menubutton $w.menubar.file -text File -underline 0 -menu $w.menubar.file.menu
  menubutton $w.menubar.calculate -text Calculate -underline 0 -menu $w.menubar.calculate.menu
  menubutton $w.menubar.graphics -text Appearance -underline 0 -menu $w.menubar.graphics.menu

  pack $w.menubar.file  $w.menubar.calculate  $w.menubar.graphics  -side left

  menubutton $w.menubar.help -text Help -underline 0 -menu $w.menubar.help.menu
  menu $w.menubar.help.menu
  $w.menubar.help.menu add command -label "Aligntool Help" -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/aligntool"
  $w.menubar.help.menu add command -label "Structure codes..." -command  [namespace code {tk_messageBox -parent $w  -type ok -message "Secondary Structure Codes\n\nT        Turn\nE        Extended conformation\nB        Isolated bridge\nH        Alpha helix\nG        3-10 helix\nI         Pi-helix\nC        Coil (none of the above)\n" } ]

  pack $w.menubar.help -side right 
  
  #File menu
  menu $w.menubar.file.menu
  $w.menubar.file.menu add command -label "Print to file..." -command [namespace code {printCanvas} ] 
  $w.menubar.file.menu add command -label "Load data file..." -command [namespace code {loadDataFile ""}  ] 
  $w.menubar.file.menu add command -label "Write data file..." -command [namespace code {writeDataFile ""}  ] 
  $w.menubar.file.menu add command -label "Close Window" -command [namespace code stopZoomSeq] 
  
  #Calculate menu
  
  menu $w.menubar.calculate.menu 
  
  $w.menubar.calculate.menu add command -label "Clear data"  -command  [namespace code clearData] 
  $w.menubar.calculate.menu add command -label "Read pre-computed CE alignment"  -command [namespace code {loadAlignedPair} ] 
  $w.menubar.calculate.menu add command -label "Run CE align"  -command [namespace code {doCEAlign /tmp}] 
  $w.menubar.calculate.menu add command -label "Apply CE align"  -command [namespace code {applyCEAlign 1}] 
$w.menubar.calculate.menu add command -label "Calc. res-res RMSD"  -command [namespace code {calcResResRMSD $rmsdSelA $rmsdSelB}] 

  
  #Graphics menu
  
  menu $w.menubar.graphics.menu
  $w.menubar.graphics.menu add cascade -label "Highlight color/style" -menu $w.menubar.graphics.menu.highlightMenu 
  $w.menubar.graphics.menu add command -label "Set scaling..." -command  [namespace code setScaling]
  #Second level menu for highlightColor 

  set dummyHighlight 1 
  #set dummyHighlight so drawn selected first time, we use -command for actual var change
  menu $w.menubar.graphics.menu.highlightMenu
  $w.menubar.graphics.menu.highlightMenu add radiobutton -label "Yellow" -command {set highlightColor yellow} -variable dummyHighlight -value 0 
  $w.menubar.graphics.menu.highlightMenu add radiobutton -label "Purple" -command {set highlightColor purple} -variable dummyHighlight -value 1 
  
  #the w.can object made here
  set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValXNum(0) + 1))]    
  set xsize [expr  $xcol($firstData) +  ($scalex *  $dataWidth * ( $alignIndex(num)) ) ]

  place  $w.zoomXlevel -in $w.panl  -bordermode inside -rely -0.2 -y 30  -relx 0.1 -anchor n
  place  $w.zoomBothlevel -in $w.panl -bordermode inside -rely -0.2 -y 70 -relx 0.1 -anchor n 

  
  place $w.resDistLab -in $w.panl  -bordermode outside -rely .25 -relx 0.83 -anchor n
  place $w.resDistVal -in $w.panl  -width 70 -bordermode outside -rely .25 -relx 0.91 -anchor n
  place $w.molLabA -in $w.panl  -bordermode outside -rely .2 -relx 0.485 -anchor n
#  place $w.molASeltext -in $w.molLabA -width 120 -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  place $w.molLabChainA -in $w.molLabA -width 60 -bordermode outside -rely 1.7 -relx 0.0 -anchor w
  place $w.molAalignChain -in $w.molLabChainA  -bordermode outside -rely 0.0 -relx 1.2 -anchor n
  place $w.molASeltext -in $w.molLabA -width 120 -bordermode outside -rely 3.0 -relx 0.0 -anchor w
  place $w.molLabB -in $w.panl  -bordermode outside -rely .2 -relx 0.66 -anchor n
  place $w.molLabChainB -in $w.molLabB -width 60 -bordermode outside -rely 1.7 -relx 0.0 -anchor w
  place $w.molBalignChain -in $w.molLabChainB  -bordermode outside -rely 0.0 -relx 1.2 -anchor n
  place $w.molBSeltext -in $w.molLabB -width 120 -bordermode outside -rely 3.0 -relx 0.0 -anchor w
#  place $w.molLabC -in $w.panl  -bordermode outside -rely .2 -relx 0.75 -anchor n
#  place $w.molCSeltext -in $w.molLabC -width 120 -bordermode outside -rely 1.0 -relx 0.5 -anchor n
#  place $w.molCalignChain -in $w.molCSeltext -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  place $w.pairval -in $w.panl -width 70 -bordermode outside -rely .1 -relx 0.3 -anchor n
  place $w.molA -in $w.molLabA -bordermode outside -rely 0.5 -relx 1.0 -anchor w
  place $w.molB -in $w.molLabB -bordermode outside -rely 0.5 -relx 1.0 -anchor w
#  place $w.molC -in $w.molLabC -bordermode outside -rely 0.5 -relx 1.0 -anchor w
  place $w.rmsdA -in $w.rmsdCalc -bordermode outside -rely 0.5 -relx 1.04 -anchor w
  place $w.rmsdB -in $w.rmsdCalc -bordermode outside -rely 0.5 -relx 1.29 -anchor w
  #place $w.panl.zoomlevel -in $w.molBSeltext -bordermode outside -rely 1.8 -relx .5 -anchor n 
  #place $w.panl.zoomlevel -in $w.panl -rely 0.5 -relx .5 -anchor n 
  
  place $w.txtlab -in $w.pairval  -bordermode outside -rely 1.3 -relx 0.5 -anchor n
  place $w.showall -in $w.pairval  -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  place $w.every_res  -in $w.showall -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  place $w.resname_toggle -in $w.every_res -bordermode outside -rely 1.0 -y 40 -relx 0.5 -anchor n
  place $w.rmsdCalc -in $w.pairval  -bordermode outside -rely 2.1 -relx 7.10 -anchor n

  #done with interface elements     

  #ask window manager for size of window
  #turn traces  on (initialize_struct trace comes later)
  #trace variable userScalex w  [namespace code redraw]
  #trace variable userScaley w  [namespace code redraw]
  trace variable ::vmd_pick_event w [namespace code list_pick]
  for {set index 0} {$index<$numAlign} {incr index} {
    trace variable alignIndex($index,currentMol) w [namespace code {molChooseX $index}]
  }
}

proc  ::vmd_aligntool::showPair {x y shiftState whichCanvas} {
  variable xcol
  variable firstData
  variable dataWidth
  variable scalex
  variable scaley
  variable dataValX
  variable dataValXNum
  variable numHoriz
  variable xcanmax
  variable ycanmax
  variable ytopmargin
  variable ybox
  variable repPairX
  variable w
  variable pairVal
  variable numAlign
  #should use same code as marquee checks!
  #maybe store top, and restore it afterwards
  set x [expr $x + $xcanmax(data) * [lindex [$w.can xview] 0] ]
  set y [expr $y + $ycanmax(data) * [lindex [$w.can yview] 0] ]
  set cursorXnum [expr  int (($x - $xcol($firstData))/ ($dataWidth * $scalex)) - 1 ]
  set cursorYnum [expr int (0.0 + ((0.0 + $y - $ytopmargin) / ($scaley * $ybox)))]
  
  if {$cursorXnum> $numHoriz}  {
    set cursorXnum $numHoriz
  }
  
  if {$cursorXnum< 0 } { 
    set cursorXnum 0
  } 
  
  if {$cursorYnum>$dataValXNum(0)} {
    set cursorYnum $dataValXNum(0)
  }
  if {$cursorYnum < 0} {
    set cursorYnum 0 
  }

  for {set i 0} {$i<$numAlign} {incr i} {
    set residX($i) $dataValX($i,0,$cursorYnum)
    set chainX($i) $dataValX($i,2,$cursorYnum)
  }
  if {[catch {set val  $dataValX(0,[expr $firstData+$cursorXnum],$cursorYnum)}] } { 
    set pairVal "" 
  } else {
    set pairVal $val
  }
  for {set i 0} {$i<$numAlign} {incr i} {
    set seltextX($i) "resid $residX($i) and chain $chainX($i)"
  }
  #highlight resisdues in the graph
  #show the involved residues in GL window
  for {set i 0} {$i<$numAlign} {incr i} {
    if {($repPairX($i,$currentMolX($i)) != "null")} {
      if { [expr [molinfo $currentMolX($i) get numreps] -1] >= $repPairX($i,$currentMolX($i)) } {
        mol modselect $repPairX($i,$currentMolX($i)) $currentMolX($i) $seltextX($i)
      } else {
        createHighlight  repPairX $i currentMolX($i) $seltextX($i) 7 
      }
    } else {
      createHighlight repPairX $i currentMolX($i) $seltextX($i) 7 
    }
  }
  
  return
}

proc  ::vmd_aligntool::drawTimeBar {f} {                          
  variable w
  variable dataWidth
  variable scalex
  variable xcol 
  variable firstData
  variable ycanmax

  #puts "showing frame $f"
  set xTimeBarStart  [expr  ( ($f + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($firstData)]
  set xTimeBarEnd  [expr  ( ($f + 2.0 ) * ($dataWidth * $scalex)) +  $xcol($firstData)]
  #puts "xTimeBarStart= $xTimeBarStart  xTimeBarEnd = $xTimeBarEnd"
  #more efficient to re-configure x1 x2
  $w.can delete timeBarRect
  set timeBar [$w.can create rectangle  $xTimeBarStart 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "\#000000" -stipple gray50 -outline "" -tags [list dataScalable timeBarRect ] ]

  #move the time line 
} 

proc ::vmd_aligntool::writeDataFile {filename} {
  variable w
  variable dataName
  variable dataValX
  variable dataValXNum
  variable currentMolX
  variable firstStructField
  variable numHoriz

  if {$filename == ""  } {
    set filename [tk_getSaveFile -initialfile $filename -title "Save Trajectory Data file" -parent $w -filetypes [list { {.dat files} {.dat} } { {Text files} {.txt}} {{All files} {*} }] ]
    set writeDataFile [open $filename w]
    puts $writeDataFile "# VMD sequence data"
    puts $writeDataFile "# CREATOR= $::tcl_platform(user)"
    puts $writeDataFile "# DATE= [clock format [clock seconds]]"
    puts $writeDataFile "# TITLE= [molinfo $currentMolX(0) get name]"
    puts $writeDataFile "# NUM_FRAMES= $numHoriz "
    puts $writeDataFile "# FIELD= $dataName($firstStructField) "
    set endStructs [expr $firstStructField + $numHoriz]
    for {set field $firstStructField} {$field <= $endStructs} {incr field} {
      for {set i 0} {$i<=$dataValXNum(0)} {incr i} {
        set val $dataValX(0,$field,$i)
        set resid $dataValX(0,0,$i)
        set chain $dataValX(0,2,$i)
        set frame [expr $field - $firstStructField]
        puts $writeDataFile "$resid $chain CA $frame $val"
      }
    }
    close $writeDataFile
  }
  return
}




proc ::vmd_aligntool::getValsFromAlign {dv dataValIndex r ch co dataset alignPos} {
  variable alignIndex
  upvar $dv dataVal $r res $ch chain $co code 
  set index $alignIndex($dataset,$alignPos,ind)
  if {$index  == "+" }  {
    set res ""
    set chain "."
    set code "-"
  } else {
    set res $dataVal($dataValIndex,0,$index)
    set chain $dataVal($dataValIndex,2,$index)
    set code [string index $dataVal($dataValIndex,1code,$index) 1]
  }
}

proc ::vmd_aligntool::checkPairs {} {
  #just a debugging tool
  variable dataValX
  variable dataHashX
  variable dataValXNum
  variable alignIndex
  variable numAlign
  for {set i 0} {$i<$numAlign} {incr i} {
    set chainX($i) ""; set resX($i) ""; set codeX($i) ""
  }
  if {1==0} {
    for {set i 0} {$i < $alignIndex(num)} {incr i} {
      for {set j 0} {$j < $numAlign} {incr j} {
        catch {getValsFromAlign dataValX $j resX($j) chainX($j) codeX($j) $j $i } 
        puts -nonewline "$i: $resX($i) $chainX($i) $codeX($i) "
      }
      puts ""
    }
  }
  set width 40 
  set spacing 10
  set pos 0
  while {$pos < $alignIndex(num) } {
    for {set i 0} {$i<$numAlign} {incr i} {
      set alignX($i) "$i:"
      catch {getValsFromAlign dataValX $i resX($i) chainX($i) codeX($i) $i [expr  $pos] } 
    }
    puts -nonewline "align: $pos to [expr $pos + $width - 1]    "
    for {set i 0} {$i<$numAlign} {incr i} {
      puts -nonewline "$i: $resX($i) $codeX($i) $chainX($i)   "
    }
    puts ""
    for {set i 0 } {$i < $width} {incr i} {
      #for debugging, when catching getVals errors
      for {set j 0} {$j<$numAlign} {incr j} {
        set codeX($j) "*"
        catch { getValsFromAlign dataValX $j resX($j) chainX($j) codeX($j) $j [expr $i+ $pos]  }
        append alignX($j) $codeX($j)
      }
      if {[expr (($pos+$i+1.0)/$spacing) ==  int (($pos+$i+1.0)/$spacing) ]} {
        append alignX(0) " "; append align1 " "
      }
    } 
    for {set i 0} {$i<$numAlign} {incr i} {
      puts -nonewline $alignX($i) 
    }
    puts ""
    set pos [expr $pos + $width]
  }
}     

proc ::vmd_aligntool::parseLine {rnum rname c sname index theLine}  {
  upvar $rnum resnum $rname, resname $c chain $sname segname 
  set splitLine [split $theLine " "]
  puts $splitLine
  set resnum [lindex $splitLine [expr 4*$index + 0]]
  set resname [lindex $splitLine [expr 4*$index + 1]]
  set chain [lindex $splitLine [expr 4*$index + 2]]
  set segname [lindex $splitLine [expr 4*$index + 3]]
  puts "parsed line $theLine\n    $resnum $resname $chain $segname"
}

proc ::vmd_aligntool::loadAlignedPair { {fileName "" }} {
  #assumes that current A and B used to produce alignment file
  #dataValX 0,dataValXNum(0) data from molec. alignIndex(0,currentMol)
  #dataValX 1,dataValXNum(1) are data from molec.  alignIndex(1,currentMol)
  variable dataValX
  variable dataHashX
  variable dataValXNum
  variable alignIndex
  variable numAlign
  variable w 
  if {$fileName == ""  } {
    set fileName [tk_getOpenFile -initialfile $fileName -title "Open Trajectory Data file" -parent $w -filetypes [list { {.align files} {.align} } { {Text files} {.txt}} {{All files} {*} }] ]
  } 

  puts "loading pair alignment, fileName= $fileName"

  #for testing, only 2 mols in alignList, so reset num to 0
  set dataFile [open $fileName r]
  #get file lines into an array
  set commonName ""
  set fileLines ""
  set transCount 1
  while {! [eof $dataFile] } {
    gets $dataFile curLine
    puts "curLine= $curLine"
    if { (! [regexp "^#" $curLine] ) && ($curLine != "" ) } {
      lappend fileLines $curLine
      puts "just added >$curLine<   fileLines length=[llength $fileLines]"
    } else {
      if { [regexp "^#Name:" $curLine] } {
        for {set i 0} {$i<$numAlign} {incr i} {
          if { [regexp "^#Name$i:" $curLine] } {
            set nameX($i) [lindex [split $curLine " "] 1]
            puts "Loading file, nameX($i) is $nameX($i)"
	  }
        }
      } elseif  { [regexp "^#TransMat" $curLine] } {
        set transMatList [lrange [split $curLine " "] 1 end]
        puts "Loading file, transMatList is >$transMatList<"
        foreach {n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11} $transMatList { }
        set transMat [list [list $n0 $n1 $n2 $n3] [list $n4 $n5 $n6 $n7] [list $n8 $n9 $n10 $n11] [list 0 0 0 1] ]
        set alignIndex($transCount,transMat) $transMat
        puts "transMat is {$alignIndex($transCount,transMat)}" 
        incr transCount
      }
    }
  }
  #done with the file close it 
  close $dataFile
  set firstFileLine [lindex $fileLines 0]
  set lastFileLine [lindex $fileLines end] 

  for {set i 0} {$i<$numAlign} {incr i} {
    set indexXFirst($i) "+"
    set count 0
    while {$indexXFirst($i) == "+"} {
      set currFileLine [lindex $fileLines $count]
      parseLine resnumX($i) resnameX($i) chainX($i) segnameX($i) $i $currFileLine
      # set offset for initial "-"'s for each
      set indexXFirst($i) [assignIndex resnumX($i) chainX($i) dataHashX $i ]
      incr count
    }
    puts "indexXFirst($i) = $indexXFirst($i)"
  }
  #PLACE

  #the aligned info takes care of its own gapping
  #so we have to add pre-,po/>=st- gapping here.
  #  A: ==========|.................|=============
  #               | (aligned info)  
  #  B: ---=======|.................|========-----
  #   in example shiftX(0) = 3, shiftX(1) =0, startOffset=10,
  # shifts are shift at =/= condition, after filling the =/- or -/= condition
  #still MUST check for starting "-"'s and merge 'em/deal with 'em
  # just find nearest non-gap

  set startOffset 0
  for {set i 0} {$i<$numAlign} {incr i} {
    if {$indexXFirst($i) > $startOffset} {
      set startOffset $indexXFirst($i)
    }
  }
  for {set i 0} {$i<$numAlign} {incr i} {
    set shiftX($i) [expr $startOffset - $indexXFirst($i)]
  }

  # i represents position on the sequence
  # index represents the sequence number (ie mol 1, mol2)
  for {set i 0} {$i<$startOffset} {incr i} {
    puts "i = $i"
    for {set index 0} {$index < $numAlign} {incr index} {
      if {$i < $shiftX($index)} {
        set alignIndex($index,$i,ind) "+"
      } else {
        set alignIndex($index,$i,ind) [expr $i - $shiftX($index)]
      }
      set alignIndex($index,$i,picked) 0 
      set alignIndex($index,$i,match) 0 
    }
  }

  #now setting in parallel 
  set entryNum [expr $startOffset]
  foreach line $fileLines  {
    #sets the index in each mol's dataVal or "+" for each column of
    # multiple alignment
    for {set i 0} {$i<$numAlign} {incr i} {
      parseLine resnumX($i) resnameX($i) chainX($i) segnameX($i) $i $line
      set alignIndex($i,[expr $entryNum],ind) [assignIndex resnumX($i) chainX($i) dataHashX $i ]
      set alignIndex($i,[expr $entryNum],picked) 0
      set alignIndex($i,[expr $entryNum],match) 1
    }
    incr entryNum 
  }
  #if entries 0,1,2...,15 entryNum is 16
  #entries and offsets are the gapped space
  #indexes just point to individual dataVal 
  #now add endOffset just like startOffset 

  for {set i 0} {$i<$numAlign} {incr i} {
    parseLine resnumX($i) resnameX($i) chainX($i) segnameX($i) $i $lastFileLine
    # set offset for initial "-"'s for each
    set count [expr [llength $fileLines] -1]
    set indexXLast($i) "+"
    while {$indexXLast($i) == "+"} {
      set currFileLine [lindex $fileLines $count]
      parseLine resnumX($i) resnameX($i) chainX($i) segnameX($i) $i $currFileLine
      # set offset for initial "-"'s for each
      set indexXLast($i) [assignIndex resnumX($i) chainX($i) dataHashX $i ]
      set count [expr $count -1]
    }
    set indexXEnd($i) $dataValXNum($i)
    #calculate the remaining residues for each 
    set extraX($i) [expr $indexXEnd($i) - $indexXLast($i)] 
    puts "extraX($i) = $extraX($i)  indexXLast($i) = $indexXLast($i)  indexXEnd($i) = $indexXEnd($i)"
  }

  set maxExtra 0
  for {set i 0} {$i<$numAlign} {incr i} {
    if {$extraX($i) > $maxExtra} {
      set maxExtra $extraX($i)
    }
  }
  set totalNum [expr $entryNum +$maxExtra]
  for {set i 0} {$i<$numAlign} {incr i} {
    set shiftX($i) [expr $maxExtra - $extraX($i)]
  }

  # i represents position on the sequence
  # index represents the sequence number (ie mol 1, mol2)
  puts "now to enter extra loop"
  for {set i 0} {$i<$maxExtra} {incr i} {
    puts "$i +$entryNum"
    for {set index 0} {$index < $numAlign} {incr index} {
      if {$i < $extraX($index)} {
        set alignIndex($index,[expr $i+$entryNum],ind) [expr $i + $indexXLast($index)]
      } else {
        set alignIndex($index,[expr $i+$entryNum],ind) "+"
      }
      set alignIndex($index,[expr $i+$entryNum],picked) 0 
      set alignIndex($index,[expr $i+$entryNum],match) 0 
    }
  }


  set alignIndex(num) $totalNum 
  for {set i 0} {$i < $alignIndex(num)} {incr i} {
    for {set index 0} {$index<$numAlign} {incr index} {
      for {set index2 0} {$index2<$numAlign} {incr index2} {
        set alignIndex($index,$index2,$i,distance) "-"
      }
    }
  }
  puts "startOffset= $startOffset"
  puts "finished with align loading alignIndex(num)= $alignIndex(num)"
  puts "transMat is {$alignIndex(1,transMat)}" 
  puts "numAlign = $numAlign"

  unset fileLines
}

proc ::vmd_aligntool::doCEAlign {scratchdir} {
  variable alignIndex
  variable alignChain
  variable cescratch
  variable cedir
  variable cebin
  variable numAlign
  variable w

  set flagDash 0
  set flagLen 0
  for {set i 0} {$i<$numAlign} {incr i} {
    if {$alignChain($i)=="-"} {
      set flagDash 1
    }
    if {[string length $alignChain(0)] != 1} {
      set flagLen 1
    }
  }
  if {$flagDash} {
    tk_messageBox -title "Error in chain names" -type ok -message "Please enter chain names.\nNo \"-\"'s accepted." -parent $w
    return
  }
  if {$flagLen} {
    tk_messageBox -title "Error in chain names" -type ok -message "All chains must be 1 character" -parent $w
    return
  }
  set fileString ""
  set chainString ""
  for {set index 0} {$index<$numAlign} {incr index} {
    set sel [atomselect $alignIndex($index,currentMol) "all"]
    set file($index) "vmdCEtemp$index.pdb" 
    $sel writepdb [file join $scratchdir $file($index)] 
    append fileString $file($index) " "
    append chainString $alignChain($index) " "
  }
  set fileString [string range $fileString 0 [expr [string length $fileString] -2]]
  set chainString [string range $chainString 0 [expr [string length $chainString]-2]]

  setCEDirs
  ::multAlign $fileString $chainString $cedir $cebin $scratchdir $scratchdir
  loadAlignedPair [file join $scratchdir "out.align"]
  unset sel
}

proc ::vmd_aligntool::applyCEAlign {alignment} {
  variable alignIndex 
  variable numAlign
  if {$alignment == $numAlign} {
    return
  } else {
    set index $alignment
    set sel [atomselect $alignIndex($alignment,currentMol) "all"]
    $sel move $alignIndex($alignment,transMat)
    incr alignment
    applyCEAlign $alignment
  }
}


proc ::vmd_aligntool::calcResResRMSD {mol1 mol2} {
  variable w
  variable alignIndex
  variable dataValX
  variable coord1
  variable coord2
  variable rmsd
  variable rmsdList
  variable numAlign

  # always comparing only two molecules, so use A and B

  set rmsdList list
  for {set i 0} {$i < $alignIndex(num)} {incr i} {
    catch {getValsFromAlign dataValX $mol1 resX($mol1) chainX($mol1) codeX($mol1) $mol1 $i } 
    catch {getValsFromAlign dataValX $mol2 resX($mol2) chainX($mol2) codeX($mol2) $mol2 $i } 
    set indexA $alignIndex($mol1,$i,ind)
    set indexB $alignIndex($mol2,$i,ind)
    set matchA $alignIndex($mol1,$i,match)
    set matchB $alignIndex($mol2,$i,match)

    if {($matchA == 1) && ($matchB ==1)} {
      if {($indexA == "+") || ($indexB == "+") || ($chainX($mol1) == ".") || ($chainX($mol2) == ".")} { 
          lappend rmsdList "-"
	  set alignIndex($mol1,$mol2,$i,distance) "-"
	  set alignIndex($mol2,$mol1,$i,distance) "-"
      } else {
        set region1 "(name CA) and chain $chainX($mol1) and resid $resX($mol1)"
        set region2 "(name CA) and chain $chainX($mol2) and resid $resX($mol2)"
   	set index $mol1
        set sel [atomselect $alignIndex($mol1,currentMol) $region1]
        set coords1 [$sel get {x y z}]
 	set index $mol2
        set sel [atomselect $alignIndex($mol2,currentMol) $region2]
        set coords2 [$sel get {x y z}]
        set rmsd 0
        foreach coord1 $coords1 coord2 $coords2 {
            set rmsd [expr $rmsd + [veclength2 [vecsub $coord2 $coord1]]]
        }
        set rmsd [expr sqrt($rmsd)]
        lappend rmsdList $rmsd
	  set alignIndex($mol2,$mol1,$i,distance) $rmsd
	  set alignIndex($mol1,$mol2,$i,distance) $rmsd
      } 
    }
  }
  # tk_messageBox -title "RMSD values" -type ok -message $rmsdList -parent $w

  unset sel
  drawHorzScale
  calculateSelectedRMSD
}

proc ::vmd_aligntool::calculateSelectedRMSD {} {
  variable currDist
  variable alignIndex
  variable rmsdSelA
  variable rmsdSelB

  set tempDist 0
  set tempCount 0

  for {set i 0} {$i < $alignIndex(num)} {incr i} {
    if {$alignIndex($rmsdSelA,$i,picked) || $alignIndex($rmsdSelB,$i,picked)} {
      if {$alignIndex($rmsdSelA,$rmsdSelB,$i,distance) != "-"} {
        set tempDist [expr $tempDist + pow($alignIndex($rmsdSelA,$rmsdSelB,$i,distance),2)]
        incr tempCount
      }
    }
  }
  if {$tempCount != 0} {
    set currDist [expr sqrt([expr $tempDist / ($tempCount)])]
  } else {
    set currDist ""
  }
}

proc ::vmd_aligntool::setCEDirs {} {
    variable cedir
    variable cebin
    variable cescratch

    set cedir "/home/barryi/projects/vmd/sequence/ce/ce_distr"
    set cebin "CE_solaris"
    set cescratch "/tmp"

    set answer [tk_messageBox -title "Choose CE Directory" -type yesno -message "Use default directory: $cedir ?" -parent $vmd_aligntool::w]
    switch -- $answer {
      yes { }
      no {

    	set cedir [tk_chooseDirectory \
                        -title "Choose CE Directory" \
                        -mustexist true \
                        -parent $vmd_aligntool::w \
                      ]
      }
    }
   
    set answer [tk_messageBox -title "Choose CE Executable" -type yesno -message "Use default executable: $cebin ?" -parent $vmd_aligntool::w]
    switch -- $answer {
      yes { }
      no {
    set cebin [tk_getOpenFile \
                        -title "Choose CE Executable" \
                        -parent $vmd_aligntool::w \
                      ]
      }
    }
    set answer [tk_messageBox -title "Choose CE Scratch Directory" -type yesno -message "Use default scratch directory: $cescratch ?" -parent $vmd_aligntool::w]
    switch -- $answer {
      yes { }
      no {
    set cescratch [tk_chooseDirectory \
                        -title "Choose CE Scratch Directory" \
                        -mustexist true \
                        -parent $vmd_aligntool::w \
                      ]
      }
    }
   
}
    

proc ::vmd_aligntool::loadDataFile {filename} {
  variable w
  variable dataValX
  variable dataHashX
  variable dataValXNum
  variable dataName
  variable firstStructField
  variable rectCreated 
  variable dataName
  
  if {$filename == ""  } {
    set filename [tk_getOpenFile -initialfile $filename -title "Open Trajectory Data file" -parent $w -filetypes [list { {.dat files} {.dat} } { {Text files} {.txt}} {{All files} {*} }] ]

  } 
  set dataFile [open $filename r]
  #get file lines into an array
  set commonName ""
  set fileLines ""
  while {! [eof $dataFile] } {
    gets $dataFile curLine
    if { (! [regexp "^#" $curLine] ) && ($curLine != "" ) } {
      lappend fileLines $curLine
    } else {
      if { [regexp "^# FIELD=" $curLine] } { 
        set commonName [lindex [split $curLine " "] 2]
        puts "Loading file, field name is $commonName"
      } 
    }
  }
  #done with the file close it 
  close $dataFile
  #set frameList ""
  #data-containing frames
  foreach line $fileLines {
    #puts "the line is >$line<"
    foreach {resid chain atom frame val} [split $line " "] {}
    #puts "resid= $resid chain= $chain atom= $atom frame= $frame val= $val" 
    lappend frameList $frame
  } 
  #puts "framelist is $frameList"

  set frameList [lsort -unique -increasing -integer $frameList]
  set minFrame [lindex $frameList 0]
  set maxFrame [lindex $frameList end]
  puts "frameList is $frameList"
  # no lkonger find frame list, since catching errors on frame assignment
  # has same effect.  Could still 
  # assign values in a new Group
  # (temporarlily, to hard-coded fields, if still in hacky version)
  puts "now check fileLines:\n"
  foreach line $fileLines {
    #puts "assigning data, the line is >$line<"
    foreach {resid chain atom frame val} [split $line " "] {}
    #this assumes consecutive frames, should use frameList somewhere
    # if we really want proper reverse lookup
    if { [ catch {set fieldForFrame [expr $firstStructField + $frame ]} ] } {
      set fieldForFrame -2
      puts "couldn't read frame text \"$frame\""
    }

    #now do lookup via dataHashX(0) to find index in dataValX 0 
    if {[catch {set theIndex $dataHashX(0,$resid,$chain)} ]} {
      puts "failed to find data for resid=$resid, chain=$chain"
    } else {
      if { [catch {set dataValX(0,$fieldForFrame,$theIndex) $val} ]} {
        puts "didn't find data for frame $frame, field= $fieldForFrame, index= $theIndex, new_val= $val"
      } else {
        set dataName($fieldForFrame) $commonName
        #puts "succesfully assigned dataValX(0,$fieldForFrame,$theIndex) as $dataValX(0,$fieldForFrame,$theIndex)" 
      }
    }
  }  

  #now delete the list of data lines, no longer needed
  unset fileLines

  #redraw the data rects
  showall 1  

  return
}

proc ::vmd_aligntool::clearData {} {
  variable w
  variable dataValX
  variable dataValXNum
  variable firstStructField
  variable numHoriz
  variable usableMolLoaded
  variable rectCreated

  puts "clearing 2D data..."
  set endStructs [expr $firstStructField + $numHoriz]
  for {set field $firstStructField} {$field <= $endStructs} {incr field} {
    for {set i 0} {$i<=$dataValXNum(0)} {incr i} {
      set  dataValX(0,$field,$i) "null"
      # for the special struct case, the 0 shold give default color
      #puts "dataValX(0,$field,$i) is now $dataValX(0,$field,$i)"
      #set resid $dataValX(0,0,$i)
      #set chain $dataValX(0,2,$i)
      #set frame [expr $field - $firstStructField]
      #puts $writeDataFile "$resid $chain CA $frame $val"
    }
  }
  #redraw the data rects
  showall 1
  return
}

proc  ::vmd_aligntool::userScaleBothChanged {val} {
  variable userScalex
  variable userScaley
  variable userScaleBoth
  variable scaley
  variable fit_scaley
  variable scalex
  variable fit_scalex
  set scalex [expr $userScaleBoth * $fit_scalex]
  set scaley [expr $userScaleBoth * $fit_scaley]
  set userScalex  $userScaleBoth
  set userScaley $userScaleBoth
  redraw name func op
  #puts "redrawn, userScaleBoth= $userScaleBoth, scalex= $scalex, userScalex= $userScalex, scaley= $scaley, userScaley= $userScaley"
  return
}

proc  ::vmd_aligntool::userScalexChanged {val} {
  variable userScalex
  variable scalex
  variable fit_scalex
  set scalex [expr $userScalex * $fit_scalex]
  redraw name func op
  puts "redrawn, scalex= $scalex, userScalex= $userScalex"
  return
}

proc ::vmd_aligntool::userScaleyChanged {val} {
  variable userScaley
  variable scaley
  variable fit_scaley
  #until working ok, still do direct mapping
  set scaley $userScaley 
  redraw name func op
  return
}

proc ::vmd_aligntool::drawVertScale {} {
  variable w
  variable ytopmargin
  variable scaley
  variable ybox
  variable dataValXNum
  variable dataValX
  variable vertTextSkip
  variable verTextLeft
  variable vertTextRight
  variable amino_code_toggle
  variable monoFont

  $w.vertScale delete vertScaleText 

  #uncomment to enable auto-resizing
  #canvas $w.can -width [expr $xcanwindowmax] -height $ycanwindowmax -bg #E9E9D9 -xscrollcommand [namespace code dataCanvasScrollX] -yscrollcommand [namespace code dataCanvasScrollY]  -scrollregion  "0 0 $xcanmax(data) $ycanmax(data)" 
  canvas $w.vertScale -width $xcanmax(vert) -height $ycanwindowmax -bg #C0D0C0 -yscrollcommand "$w.ys set" -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)" 

  canvas $w.horzScale -width $xcanwindowmax -height  $ycanmax(horz) -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)" -bg #A9A9A9 -xscrollcommand "$w.xs set"
# jfkdsjafksjdfksja
  #pack the horizontal (x) scrollbar
  #pack $w.spacer1 -in $w.cfr -side left  -anchor e  
  #pack $w.spacer2 -in $w.cfr -side bottom -anchor s  
  #when adding new column, add to this list (maybe adjustable later)
  #The picked fields 
  
  #Add the text...
  set field 0            
  #note that the column will be 0, but the data will be from picked
  set yDataEnd [expr $ytopmargin + ($scaley * $ybox * ($dataValXNum(0) +1))]
  set y 0.0
  set yposPrev  -10000.0
  #Add the text to vertScale...
  set field 0             

  #we want text to appear in center of the dataRect we are labeling
  set vertOffset [expr $scaley * $ybox / 2.0]

  #don't do $dataValXNum(0), its done at end, to ensure always print last 
  for {set i 0} {$i <= $dataValXNum(0)} {incr i} {
    set ypos [expr $ytopmargin + ($scaley * $y) + $vertOffset]
    if { ( ($ypos - $yposPrev) >= $vertTextSkip) && ( ( $i == $dataValXNum(0)) || ( ($yDataEnd - $ypos) > $vertTextSkip) ) } {
      if {$amino_code_toggle == 0} {
        set res_string $dataValX(0,1,$i)} else {
          set res_string $dataValX(0,1code,$i)
        }

      #for speed, we use vertScaleText instead of $dataName($field)
      $w.vertScale create text $vertTextRight $ypos -text "$dataValX(0,0,$i) $res_string $dataValX(0,2,$i)" -width 200 -font $monoFont -justify right -anchor e -tags vertScaleText 

      set yposPrev  $ypos
    }        
    set y [expr $y + $vertTextSkip]
  } 
}


proc ::vmd_aligntool::drawHorzScale {} {
  variable w
  variable amino_code_toggle
  variable ytopmargin
  variable scalex
  variable dataValX
  variable dataValXNum
  variable monoFont
  variable monoBoldFont
  variable firstData
  variable seqHeight
  variable resWidth
  variable alignIndex
  variable xleftmargin
  variable numAlign
  variable rmsdSelA
  variable rmsdSelB
  $w.horzScale delete horzScaleText 

  #when adding new column, add to this list (maybe adjustable later)
  # The picked fields 
  # Add the text...
  # note that the column will be 0, but the data will be from picked
  # we want text to appear in center of the dataRect we are labeling
  # ensure minimal horizaontal spacing
  
  #currently same in both modes...
  if {$amino_code_toggle == 0} {
    set horzSpacing 15 
  } else {
    set horzSpacing 15
  } 
  
  set horzNumSpacing 100 
  set scaledHorzDataTextSkip [expr $scalex * $resWidth]
  #set scaledHorzDataOffset [expr $scalex * $resWidth/ 2.0]
  for {set i 0} {$i<$numAlign} {incr i} {
    set yposX($i) [expr $ytopmargin + $seqHeight*1.5*($i+1)]
  }
  set yNumPosTop 40 
  set yNumPosBot 115
  set xStart $xleftmargin 
  set xDataEnd  [expr int ($xStart +  $scalex * ($resWidth * $alignIndex(num) ) )] 
  set x 0 
  set xNum 0

  $w.horzScale delete barHighlight
  set maxDist 0
  for {set i 0} {$i < $alignIndex(num)} {incr i} {
    if {$alignIndex($rmsdSelA,$rmsdSelB,$i,distance) > $maxDist} {
      set maxDist $alignIndex($rmsdSelA,$rmsdSelB,$i,distance)
    }
  }
  # don't do $dataValXNum(1), its done at end, to ensure always print last 
  # numbers are scaled for 1.0 until xpos
  # this is tied to data fields, which is produced from frames upong
  # first drawing. Should really agreee with writeDataFile, which currently uses frames, not fields
  set xposPrev -1000 
  set xNumPosPrev -1000
  #there is B data in $firstData, traj data starts at firstData+1
  puts "now to loop over 0 to $alignIndex(num)"
  for {set i 0} {$i < $alignIndex(num)} {incr i} {
    #set xpos [expr int ($xStart + ($scalex * $x) + $scaledHorzDataOffset)]
    set xpos [expr int ($xStart + ($scalex * $x) ) ]


    if { ( ($xpos - $xposPrev) >= $horzSpacing) && ( ( $i == $alignIndex(num)) || ( ($xDataEnd - $xpos) > $horzSpacing) ) } {
      #puts "getting align $i" 
      for {set index 0} {$index<$numAlign} {incr index} {
        set resX($index) "##"; set codeX($index) "#"; set chainX($index) "#"
        catch {getValsFromAlign dataValX $index resX($index) chainX($index) codeX($index) $index $i }
      }
      
      if {$amino_code_toggle == 0} {
	for {set index 0} {$index<$numAlign} {incr index} {
          set res_stringX($index) $codeX($index)
        }
      } else {
	for {set index 0} {$index<$numAlign} {incr index} {
          set res_stringX($index) $codeX($index)
        }
      } 
      
      if {$alignIndex(0,$i,match) == 1} {
        set font0 $monoBoldFont
      } else { 
        set font0  $monoFont
      } 
      
      if {$alignIndex(1,$i,match) == 1} {
        set font1 $monoBoldFont
      } else { 
        set font1 $monoFont
      } 

      for {set index 0} {$index<$numAlign} {incr index} {
        $w.horzScale create text $xpos $yposX($index) -text "$res_stringX($index)\n$chainX($index)" -width 30 -font $font0 -justify center -anchor s -tags horzScaleText 
      }
      set xposPrev  $xpos
      if { ($xpos - $xNumPosPrev) >= $horzNumSpacing} {
        set xNumPosPrev  $xpos
        $w.horzScale create text $xpos [expr $yNumPosTop -25] -text "($i)" -width 50 -font $monoFont -justify center -anchor s -tags horzScaleText
        for {set index 0} {$index<$numAlign} {incr index} {
          $w.horzScale create text $xpos [expr $yposX($index)-($seqHeight*1)] -text "$resX($index)" -width 30 -font $monoFont -justify center -anchor s -tags horzScaleText
        }
      }
    }  
      
    # display bar graph
    set red 0; set green 249 ;set blue 0
    set hexred     [format "%02x" $red]
    set hexgreen   [format "%02x" $green]
    set hexblue    [format "%02x" $blue]
    for {set index 0} {$index<$numAlign} {incr index} {
      catch {getValsFromAlign dataValX $index resX($index) chainX($index) codeX($index) $index $i } 
      set indexX($index) $alignIndex(0,$i,ind)
      set matchX($index) $alignIndex(0,$i,match)
    }
    if {($matchX($rmsdSelA) == 1) && ($matchX($rmsdSelB) ==1) && ($alignIndex($rmsdSelA,$rmsdSelB,$i,distance)!="-")} {
      if {($indexX($rmsdSelA) != "+") && ($indexX($rmsdSelB) != "+") && ($chainX($rmsdSelA) != ".") && ($chainX($rmsdSelB) != ".")} {
        set barGraph [$w.horzScale create rectangle  [expr $xpos -3] [expr $yposX(0) - 30] [expr $xpos +4] [expr $yposX(0) -30 -(30/$maxDist)*$alignIndex($rmsdSelA,$rmsdSelB,$i,distance)] -fill "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags barHighlight]
      }
    }

    set x [expr $x + $resWidth]
  } 
}

proc ::vmd_aligntool::makeSelText {dval dataValIndex dataValNum pickedonly } {
  upvar $dval dataVal  

  #make selection string to display in VMD 
  set selectText  "" 
  set prevChain "Empty" 
  #Cannot be held by chain  
  for {set i 0} {$i <= $dataValNum} {incr i} {
    if { ($pickedonly == 0) || ($dataVal($dataValIndex,picked,$i) == 1  )} {
      if { [string compare $prevChain $dataVal($dataValIndex,2,$i)] != 0} {
        #chain is new or has changed
        append selectText ") or (chain $dataVal($dataValIndex,2,$i)  and resid $dataVal($dataValIndex,0,$i)"
      } else {
        append selectText " $dataVal($dataValIndex,0,$i)"
      }
      set prevChain $dataVal($dataValIndex,2,$i)
    }
  }  
  append selectText ")"
  set selectText [string trimleft $selectText ") or " ]

  #check for the state when mol first loaded
  if {$selectText ==""} {
    set selectText "none"
  } 
  return $selectText
} 

proc ::vmd_aligntool::makeAlignSelText {dval dataValIndex dataValNum pickedonly alignment} {
  variable alignIndex
  upvar $dval dataVal  

  #make selection string to display in VMD 
  set selectText  "" 
  set prevChain "Empty" 
  #Cannot be held by chain  
  for {set i 0} {$i < $alignIndex(num)} {incr i} {
    set index $alignIndex($alignment,$i,ind)
    if {($index != "+") && ( ($pickedonly == 0) || ($alignIndex($alignment,$i,picked) == 1  ) )} {
      if { [string compare $prevChain $dataVal($dataValIndex,2,$index)] != 0} {
        #chain is new or has changed
        append selectText ") or (chain $dataVal($dataValIndex,2,$index)  and resid $dataVal($dataValIndex,0,$index)"
      } else {
        append selectText " $dataVal($dataValIndex,0,$index)"
      }
      set prevChain $dataVal($dataValIndex,2,$index)
    }
  }  
  append selectText ")"
  set selectText [string trimleft $selectText ") or " ]

  #check for the state when mol first loaded
  if {$selectText ==""} {
    set selectText "none"
  } 

  return $selectText
} 

proc vmd_aligntool::assignIndex {r c dHash ind} {
  upvar $r resnum $c chain $dHash dataHash
  puts "entered assignIndex"
  #note that this does not yet care about segname, it should... 
  if {$chain=="none" } {set chain "X"}
  set index "+"

  if {$resnum != "-"} {
    #now do lookup via dataHash to find index in dataVal 
    if {[catch {set theIndex $dataHash($ind,$resnum,$chain)} ]} {
      puts $dataHash($ind,$resnum,$chain)
      puts "failed to find data for resnum=$resnum, chain=$chain"
      set index "+"
    } else {
      set index $theIndex
      puts "index= $index for dataVal($resnum,$chain)"
    }
  }
  return $index
}

proc aligntool {} {
  namespace eval ::vmd_aligntool { 
    #####################################################
    # set traces and some binidngs, then call zoomSeqMain
    #####################################################
    ####################################################
    # Create the window, in withdrawn form,
    # when script is sourced (at VMD startup)
    ####################################################
    variable numAlign 2
    set windowError 0
    set errMsg ""
    set w  .vmd_MultiSequenceWindow
    if { [catch {toplevel $w -visual truecolor} errMsg] } {
      puts "Info) Aligntool window can't find trucolor visual, will use default visual.\nInfo)   (Error reported was: $errMsg)" 
      if { [catch {toplevel $w } errMsg ]} {
        puts "Info) Default visual failed, Aligntool window cannot be created. \nInfo)   (Error reported was: $errMsg)"         
        set windowError 1
      }
    }

    if {$windowError == 0} { 
      #don't withdraw, not under vmd menu control during testing
      #wm withdraw $w
      wm title $w "VMD Aligntool"
      #wm resizable $w 0 0 
      wm resizable $w 1 1 
      variable w
      variable monoFont
      variable monoBoldFont
      variable initializedVars 0
      variable needsDataUpdate 0 
      
      #overkill for debugging, should only need to delete once....
      for {set index 0} {$index<$numAlign} {incr index} {
        trace vdelete alignIndex($index,currentMol) w [namespace code {molChooseX $index}]
      }
      set index [expr $index - 1]
      trace vdelete ::vmd_pick_event w  [namespace code list_pick] 
      trace vdelete ::vmd_initialize_structure w  [namespace code molChooseMenu]

      bind $w <Map> "+[namespace code Show]"
      bind $w <Unmap> "+[namespace code Hide]"
      #specify monospaced font, 12 pixels wide
      #font create tkFixedMulti -family Courier -size -12
      #for test run tkFixed was made by normal sequence window
      #so is in vmd's tcl interp  at startup
      set monoFont tkFixed
      
      #check errors from this (catching for testing so no re-creation errors)
      catch {font create tkFixedBold -weight bold -family Courier -size -12}
      set monoBoldFont tkFixedBold
      #slight var clear, takes vare of biggest space use
      catch {unset dataValX}
      #call to set up, after this, all is driven by trace and bind callbacks
      zoomSeqMain
    }
    return $w
  }
}



