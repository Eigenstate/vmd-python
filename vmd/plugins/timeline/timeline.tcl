# timeline.tcl  -- VMD script tdo list/select 2D trajectory info
# about a biomolecule
#
# Copyright (c) 2008 The Board of Trustees of the University of Illinois
#
# Barry Isralewitz  barryi@ks.uiuc.edu    
# vmd@ks.uiuc.edu
#
#
# $Id: timeline.tcl,v 1.100 2015/10/12 19:50:16 ryanmcgreevy Exp $

package provide timeline 2.3



#source progressBox.tcl

proc timeline {} {
  return [::timeline::startTimeline]
}


#####
#
#  Timeline programming notes
#  
#  The dataname array provides names
# of the columns of $dataVal(column,row)
# The column names are called fields,
# but (except for 0 through 2) represent
# data for animation frames. 
#  Here any column from ($dataOrigin) to ($numDataFrames + $dataOrigin -1)
# contains the same kind of data, with frame 0 in ($dataOrigin), frame 1 in ($dataOrigin+1).
# 
# daataName(vals) data values: the displayed analysis data (x-val, H-bond, etc)
# dataName(0) resid -> dataName(resid) dataVal(resid)
# dataName(1) resname -> dataName(resname) dataVal(resname)
# dataName(1code) -> dataName(rescode) dataVal(rescode)  1-letter code version of resname
# dataName(2) chain -> dataName(chain) dataVal(chain)
# dataName(segname) segname
# dataVal(referenceVal,) used as scratch space by some procs
#
#    The dataName array is a leftover from
# when timeline's vertical columns were fields
# for multiple columns of differnt sorts of data
# (as seen in the related Sequence Viewer plugin)
# 

#######################
#create the namespace
######################t#
namespace eval ::timeline  {
    variable clicked "-1"
    #XXX can we eliminate these next 4 lines by using local vars?
    variable oldmin "0.0"
    variable oldmax "2.0"
    
    variable oldFirstAnalysisFrame ""
    variable oldLastAnalysisFrame ""
    variable oldONdist "" 
    variable oldHbondDistCutoff ""
    variable oldHbondAngleCutoff ""
    variable oldRMSFstepSize ""
    variable oldRMSFwindowWidth ""
    variable oldSASArad ""
    variable oldInterDist ""
    variable oldInterSelList ""
    variable oldInterFromResSel
    variable oldInterToSel
    variable oldInterSetMethod 
    variable oldFilterMin ""
    variable oldFilterMax ""
    variable oldFilterFrameStart ""
    variable oldFilterFrameEnd ""
    variable oldFilterNumFramesRequired "" 
    variable oldCcssFile ""
    variable oldCcssDCDFile ""  
    variable oldCcssMapFile ""
    variable oldCcssMapres ""
    variable oldCcssSpacing ""
    variable oldCcssThreshold ""
    variable oldCcssUseThreshold ""
    variable oldNativeDist ""
    variable oldNativeFromSelList ""
    variable oldNativeToSel ""


    variable lastCalc "0"
       # last calculation, see ::recalc switch statement for other codes.
       #XXX is lastCalc doing any good right now?
       #XXX lastCalc _should_ be used (even in calcHbonds) after Appearance:Set Scaling.
}

####################/
#define the procs
####################

proc ::timeline::writeDataFileHeader {outDataFile molid dataTitle numFrames\
          numSelectionGroups usesFreeSelection} {
    # HEADER starts
    set dataFileVersion 1.4
    puts $outDataFile "# VMD Timeline data file"
    puts $outDataFile "# FILE_VERSION= $dataFileVersion"
    puts $outDataFile "# CREATOR= $::tcl_platform(user)"
    puts $outDataFile "# MOL_NAME= [molinfo $molid get name]"
    puts $outDataFile "# DATA_TITLE= $dataTitle"
    puts $outDataFile "# NUM_FRAMES= $numFrames "
    puts $outDataFile "# NUM_ITEMS= $numSelectionGroups"
    puts $outDataFile "# FREE_SELECTION= $usesFreeSelection"

    #HEADER ends
}


proc ::timeline::tlPutsDebug {theText} {
 #puts "*TL DEBUG* $theText"
}


proc ::timeline::recalc {} {
  
  variable lastCalc
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable currentMol
  variable dataMin
  variable dataMax
  variable w
  variable trajMin
  variable trajMax

  tlPutsDebug "starting recalc..."

  set dataMin(all) null
  set dataMax(all) null
 
  #XXX get rid of integer-associations here and in menu calls, too error prone
  switch $lastCalc {
  -2  { calcSelEmpty -2}
  -1  { }
  0   { clearData}
  1   { calcDataStruct}
  2   { calcDataX}
  3   { calcDataY}
  4   { calcDataZ}
  5   { calcDataPhi}
  6   { calcDataDeltaPhi}
  7   { calcDataPsi}
  8   { calcDataDeltaPsi}
  9   { ::rmsdtool}
  10  { calcTestFreeSel 10}
  11  { calcHbonds 11 }
  12  { calcDataUser}
  13  { calcDataAnyResFunc }
  14  { calcDisplacement }
  15  { calcDispVelocity }
  16  { calcSaltBridge 16}
  17  { calcRMSD}
  18  { calcRMSF}
  19  { calcSASA}
  20  { calcNativeContacts 20}
  21  { calcInterSelContacts 21}
  22  { calcCC}
  
  }
  postCalc;
}

proc ::timeline::canvasScrollY {args} { 
  variable w
  eval $w.can yview $args
  eval $w.vertScale yview $args 
}     
proc ::timeline::canvasScrollX {args} { 
  variable w

  eval $w.can xview $args
  eval $w.horzScale xview $args 
  eval $w.threshGraph xview $args 
  
  return
}


proc ::timeline::lookupCode {resname} {
  variable codes

  set result ""
  if {[catch { set result $codes($resname) } ]} {
    set result $resname
  } else {
    set result " $result "
  }
  return $result
}

proc ::timeline::stopZoomSeq {} {
  menu timeline off
}

proc ::timeline::chooseColor {intensity} {
  variable dataName
  variable dataOrigin
  variable colorscale
  set field_color_type s 
  #hack to default to struct field type coloring
  if {$dataName(vals) != "struct"} {
    if {$intensity < 0} {set intensity 0}
    if {$intensity > 255} {set intensity 255}
    set intensity [expr int($intensity)]
    #set field_color_type $field 
    #check color mapping
    set field_color_type default 
  }
  #super hacky here
  switch -exact $field_color_type {         
    s {
      #the field_color_type hack sends all structs to here 
      if { [catch {
        switch $intensity {
          
######
## CMM 08/28/06 mmccallum@pacific.edu
##
# modify the colors displayed in order to better match what shows up in the
# "structure" representation.  Please note that I have set 3_{10} 
# helices to be blue, to provide more contrast between the purple (alpha)
# and default mauve/pinkish for 3_{10} helices from the "structure" rep
####
#  This gives blue = 3_{10}, purple = alpha, red = pi helix
###################################################
          B {set red 180; set green 180; set blue 0}
          C {set red 255; set green 255; set blue 255}
          E {set red 255; set green 255; set blue 100}
          T {set red 70; set green 150; set blue 150}
          # G = 3_{10}
          G {set red 20; set green 20; set blue 255}
          # H = alpha;  this was fine-tuned a bit to match better.
          H {set red 235; set green 130; set blue 235}
          I {set red 225; set green 20; set blue 20}
          default {set red 100; set green 100; set blue 100}
        }
        
      } ] 
         } { #badly formatted file, intensity may be a number
        set red 0; set green 0; set blue 0 
      }
    }
    default {
      set c $colorscale(choice)
      set red $colorscale($c,$intensity,r)
      set green $colorscale($c,$intensity,g)
      set blue $colorscale($c,$intensity,b)
   } 
  }
  
  #convert red blue green 0 - 255 to hex
  set hexred     [format "%02x" $red]
  set hexgreen   [format "%02x" $green]
  set hexblue    [format "%02x" $blue]
  set hexcols [list $hexred $hexgreen $hexblue]

  return $hexcols
}

proc ::timeline::redraw {name func op} {
  
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
  variable dataVal 
  variable dataValNum 
  variable dataOrigin 
  variable dataName 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable vertTextRight
  variable vertHighLeft
  variable vertHighRight
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin
  variable dataMax 
  variable trajMin
  variable trajMax
  variable xPosScaleVal
  variable everRedrawn
  variable usableMolLoaded
  variable rectCreated
  variable prevScalex
  variable prevScaley
  variable numDataFrames
  variable progressDrawFraction
  variable progressBoxExists
  variable progressCancelPressed

  set dateStartRedraw [clock seconds]

  if { ($usableMolLoaded) && ($dataValNum == -1 ) } {
  drawColScale
  $w.selInfo configure -text " $dataName(vals) \n\n\[NO RESULTS\]\n"
  }   
  
  if { ($usableMolLoaded) && ($dataValNum >=0 ) } {
    set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValNum + 1) )]  

    set xsize [expr  $xcol($dataOrigin) +  ($scalex *  $dataWidth *  $numDataFrames)  ] 

    set ycanmax(data) $ysize
    set ycanmax(vert) $ycanmax(data)
    set xcanmax(data) $xsize
    set xcanmax(horz) $xcanmax(data)
    if {$ycanmax(data) < $ycanwindowmax} {
      set ycanmax(data) $ycanwindowmax
    }


    if {$xcanmax(data) < $xcanwindowmax} {
      set xcanmax(data) $xcanwindowmax
    }

    $w.can configure -scrollregion "0 0 $xcanmax(data) $ycanmax(data)"
    $w.vertScale configure -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)"
    $w.horzScale configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    $w.threshGraph configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    drawVertScale
    drawHorzScale
    
    
    #for example, if we have 2 frames of data, frame 0 and frame 1,
    #then numDataFrames = 2.  Since dataOrigin =3, fieldLast is 4, since data
    # is in field 3 (frame 0), field 4 (frame 1). Formula is...
    set fieldLast [expr $dataOrigin + $numDataFrames -1 ]

    #draw data on can
    #loop over all data fields

    if {! $rectCreated} {
      #this until separate data and scale highlighting
      $w.threshGraph delete xScalable
      $w.horzScale delete xScalable
      $w.vertScale delete yScalable
      $w.can delete dataScalable
      #puts "drawing rects, scalex is $scalex"
      #hack here -- for now skip B-field stuff, so minimal stuff drawn
      tlPutsDebug ": setting min/max, dataOrigin= $dataOrigin" 
      tlPutsDebug "starting redraw, date= [clock format [clock seconds] -format {%+} ]"
      set progressCalcFraction [expr 1.0 - $progressDrawFraction]
      for {set field [expr $dataOrigin ]} {$field <= $fieldLast} {incr field} {
        #tlPutsDebug "redrawing. field is $field of $fieldLast [format %.2f [expr 100.0* $field/$fieldLast]]% complete" 
        
        #find fraction completed, calcs are already done. update progressBox
        if {$progressBoxExists} {
          set completeFrac [expr $progressCalcFraction + $progressDrawFraction* double($field)/$fieldLast]
          progressBoxConfigure [expr $completeFrac] 
        }
        #set xPosFieldLeft [expr int  ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin)  ) ) ]
        #set xPosFieldRight [expr int ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin + 1 - $dataMargin)  ) ) ]
        set xPosFieldLeft [expr  ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin)  ) ) ]
        set xPosFieldRight [expr ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin + 1 - $dataMargin)  ) ) ]
        
        #now draw data rectangles
        #puts "drawing field $field at xPosField $xPosField" 
        #yipes, does this redraw all rects (even non visible) every timeXXX
        set y 0.0
        
        set intensity 0
        
        for {set i 0} {$i<=$dataValNum} {incr i} { 
          set val $dataVal($field,$i)
          if {$val != "null"} {
            #calculate color and create rectange
            
            set ypos [expr $ytopmargin + ($scaley * $y)]
            
            #should Prescan  to find range of values!   
            #this should be some per-request-method range / also allow this to be adjusted
            
            #set intensity except if field 4 (indexed struct)
            #puts "field = $field, dataName($field) = $dataName($field),i= $i" 
            if {$dataName(vals) != "struct"} {
              ##if { ( ($field != 4)  ) } open brace here 
              #set range [expr $dataMax($field) - $dataMin($field)]
              set range [expr $trajMax - $trajMin ]
              if { ($range > 0)  && ([string is double $val] )} {
                set intensity  [expr int (255. * ( (0.0 + $val - $trajMin ) / $range)) ]
                #tlPutsDebug ": $val $dataMin($field) $range $field $intensity"
              }
              
              
              
              set hexcols [chooseColor $intensity]
            } else {
              #horrifyingly, sends string for data, tcl is typeless
              set hexcols [chooseColor $val ]
            }
            foreach {hexred hexgreen hexblue} $hexcols {} 

            
            #draw data rectangle
            $w.can create rectangle  [expr $xPosFieldLeft] [expr $ypos ] [expr $xPosFieldRight]  [expr $ypos + ($scaley * $ybox)]  -fill "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags dataScalable
          }
          
          set y [expr $y + $ybox]
          if {$progressCancelPressed} {break}
        }
        if {$progressCancelPressed} {
          tlPutsDebug "progressCancelPressed" 
          #XXX clear records to function entry state
          set progressCancelPressed 0
          clearData
        } 
      }

      drawVertHighlight 
      drawColScale
    }  else {

      #$w.can scale dataRect $xcol($firstdata) $ytopmargin 1 $scaley
      #$w.can scale dataScalable $xcol($dataOrigin) [expr $ytopmargin] 1 [expr $scaley / $prevScaley ]

      $w.can scale dataScalable $xcol($dataOrigin) [expr $ytopmargin] [expr $scalex / $prevScalex]  [expr $scaley / $prevScaley ]
      #now for datarect
      $w.vertScale scale yScalable 0 [expr $ytopmargin] 1  [expr $scaley / $prevScaley ]
      $w.horzScale scale xScalable $xcol($dataOrigin) 0 [expr $scalex / $prevScalex ] 1
      $w.threshGraph scale xScalable $xcol($dataOrigin) 0 [expr $scalex / $prevScalex ] 1

    } 
    
     set rectCreated 1
    set prevScaley $scaley
    set prevScalex $scalex
    set everRedrawn 1
  
  }
  set dateEndRedraw [clock seconds]
  tlPutsDebug "done with redraw, everRedrawn= $everRedrawn"
  tlPutsDebug " start redraw= [clock format $dateEndRedraw -format {%+} ]"
  tlPutsDebug " end redraw= [clock format $dateStartRedraw -format {%+} ]"
  set secsRedraw [expr $dateEndRedraw - $dateStartRedraw]
  tlPutsDebug "$secsRedraw seconds to redraw."
  progressBoxDestroy
 
  return
}
proc ::timeline::redrawResize {name func op} {
  
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
  variable dataVal 
  variable dataValNum 
  variable dataOrigin 
  variable dataName 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable vertTextRight
  variable vertHighLeft
  variable vertHighRight
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin
  variable dataMax 
  variable trajMin
  variable trajMax
  variable xPosScaleVal
  variable everRedrawn
  variable usableMolLoaded
  variable rectCreated
  variable prevScalex
  variable prevScaley
  variable numDataFrames
  variable progressDrawFraction
  variable progressBoxExists
  variable progressCancelPressed

  set dateStartRedraw [clock seconds]

  if { ($usableMolLoaded) && ($dataValNum == -1 ) } {
  drawColScale
  $w.selInfo configure -text " $dataName(vals) \n\n\[NO RESULTS\]\n"
  }   
  
  if { ($usableMolLoaded) && ($dataValNum >=0 ) } {
    set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValNum + 1) )]  

    set xsize [expr  $xcol($dataOrigin) +  ($scalex *  $dataWidth *  $numDataFrames)  ] 

    set ycanmax(data) $ysize
    set ycanmax(vert) $ycanmax(data)
    set xcanmax(data) $xsize
    set xcanmax(horz) $xcanmax(data)
    if {$ycanmax(data) < $ycanwindowmax} {
      set ycanmax(data) $ycanwindowmax
    }


    if {$xcanmax(data) < $xcanwindowmax} {
      set xcanmax(data) $xcanwindowmax
    }

    $w.can configure -scrollregion "0 0 $xcanmax(data) $ycanmax(data)"
    $w.vertScale configure -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)"
    $w.horzScale configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    $w.threshGraph configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    drawVertScale
    drawHorzScale
    
    
    #for example, if we have 2 frames of data, frame 0 and frame 1,
    #then numDataFrames = 2.  Since dataOrigin =3, fieldLast is 4, since data
    # is in field 3 (frame 0), field 4 (frame 1). Formula is...
    set fieldLast [expr $dataOrigin + $numDataFrames -1 ]

    #draw data on can
    #loop over all data fields

    if {! $rectCreated} {
      #this until separate data and scale highlighting
      $w.threshGraph delete xScalable
      $w.horzScale delete xScalable
      $w.vertScale delete yScalable
      $w.can delete dataScalable
      #puts "drawing rects, scalex is $scalex"
      #hack here -- for now skip B-field stuff, so minimal stuff drawn
      tlPutsDebug ": setting min/max, dataOrigin= $dataOrigin" 
      tlPutsDebug "starting redraw, date= [clock format [clock seconds] -format {%+} ]"
      set progressCalcFraction [expr 1.0 - $progressDrawFraction]
      for {set field [expr $dataOrigin ]} {$field <= $fieldLast} {incr field} {
        #tlPutsDebug "redrawing. field is $field of $fieldLast [format %.2f [expr 100.0* $field/$fieldLast]]% complete" 
        
        #find fraction completed, calcs are already done. update progressBox
        if {$progressBoxExists} {
          set completeFrac [expr $progressCalcFraction + $progressDrawFraction* double($field)/$fieldLast]
          progressBoxConfigure [expr $completeFrac] 
        }
        #set xPosFieldLeft [expr int  ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin)  ) ) ]
        #set xPosFieldRight [expr int ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin + 1 - $dataMargin)  ) ) ]
        set xPosFieldLeft [expr  ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin)  ) ) ]
        set xPosFieldRight [expr ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin + 1 - $dataMargin)  ) ) ]
        
        #now draw data rectangles
        #puts "drawing field $field at xPosField $xPosField" 
        #yipes, does this redraw all rects (even non visible) every timeXXX
        set y 0.0
        
        set intensity 0
        
        for {set i 0} {$i<=$dataValNum} {incr i} { 
          set val $dataVal($field,$i)
          if {$val != "null"} {
            #calculate color and create rectange
            
            set ypos [expr $ytopmargin + ($scaley * $y)]
            
            #should Prescan  to find range of values!   
            #this should be some per-request-method range / also allow this to be adjusted
            
            #set intensity except if field 4 (indexed struct)
            #puts "field = $field, dataName($field) = $dataName($field),i= $i" 
            if {$dataName(vals) != "struct"} {
              ##if { ( ($field != 4)  ) } open brace here 
              #set range [expr $dataMax($field) - $dataMin($field)]
              set range [expr $trajMax - $trajMin ]
              if { ($range > 0)  && ([string is double $val] )} {
                set intensity  [expr int (255. * ( (0.0 + $val - $trajMin ) / $range)) ]
                #tlPutsDebug ": $val $dataMin($field) $range $field $intensity"
              }
              
              
              
              set hexcols [chooseColor $intensity]
            } else {
              #horrifyingly, sends string for data, tcl is typeless
              set hexcols [chooseColor $val ]
            }
            foreach {hexred hexgreen hexblue} $hexcols {} 

            
            #draw data rectangle
            $w.can create rectangle  [expr $xPosFieldLeft] [expr $ypos ] [expr $xPosFieldRight]  [expr $ypos + ($scaley * $ybox)]  -fill "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags dataScalable
          }
          
          set y [expr $y + $ybox]
          if {$progressCancelPressed} {break}
        }
        if {$progressCancelPressed} {
          tlPutsDebug "progreeCancelPressed" 
          #XXX clear records to function entry state
          set progressCancelPressed 0
          clearData
        } 
      }

      drawVertHighlight 
      drawColScale
    }  else {

      #$w.can scale dataRect $xcol($firstdata) $ytopmargin 1 $scaley
      #$w.can scale dataScalable $xcol($dataOrigin) [expr $ytopmargin] 1 [expr $scaley / $prevScaley ]

      $w.can scale dataScalable $xcol($dataOrigin) [expr $ytopmargin] [expr $scalex / $prevScalex]  [expr $scaley / $prevScaley ]
      #now for datarect
      $w.vertScale scale yScalable 0 [expr $ytopmargin] 1  [expr $scaley / $prevScaley ]
      $w.horzScale scale xScalable $xcol($dataOrigin) 0 [expr $scalex / $prevScalex ] 1
      $w.threshGraph scale xScalable $xcol($dataOrigin) 0 [expr $scalex / $prevScalex ] 1

    } 
    
     set rectCreated 1
    set prevScaley $scaley
    set prevScalex $scalex
    set everRedrawn 1
  
  }
  set dateEndRedraw [clock seconds]
  tlPutsDebug "done with redraw, everRedrawn= $everRedrawn"
  tlPutsDebug " start redraw= [clock format $dateEndRedraw -format {%+} ]"
  tlPutsDebug " end redraw= [clock format $dateStartRedraw -format {%+} ]"
  set secsRedraw [expr $dateEndRedraw - $dateStartRedraw]
  tlPutsDebug "$secsRedraw seconds to redraw."
  progressBoxDestroy
 
  return
}



proc ::timeline::makecanvas {} {

  variable xcanmax 
  variable ycanmax
  variable w
  variable wp
  variable xsize
  variable ysize 
  variable xcanwindowmax 
  variable ycanwindowmax
  variable threshGraphHeight
  variable horzScaleHeight
  variable vertScaleWidth 
  set xcanmax(data) $xsize 
  set ycanmax(data) $ysize
  
  
  #make main canvas




  #canvas $w.spacer1 -width [expr $vertScaleWidth+20] -height [expr $threshGraphHeight + $horzScaleHeight + 25] -bg #A00A0
  canvas $w.spacer1 -width [expr $vertScaleWidth+20] -height [expr $threshGraphHeight + $horzScaleHeight + 25] -bg #A0FFA0
  #canvas $w.spacer2 -width [expr $vertScaleWidth+20] -height [expr $threshGraphHeight + $horzScaleHeight + 25] -bg #C0C0E0
  canvas $w.spacer2 -width [expr $vertScaleWidth+20] -height [expr $threshGraphHeight + $horzScaleHeight + 25] -bg #FFC0FF
  canvas $w.can -width [expr $xcanwindowmax] -height $ycanwindowmax -bg #E9E9D9 -xscrollcommand "$w.xs set" -yscrollcommand "$w.ys set" -scrollregion  "0 0 $xcanmax(data) $ycanmax(data)" 
  canvas $w.vertScale -width $vertScaleWidth -height $ycanwindowmax -bg #C0D0C0 -yscrollcommand "$w.ys set" -scrollregion "0 0 $vertScaleWidth $ycanmax(data)" 

  canvas $w.threshGraph -width $xcanwindowmax -height  $threshGraphHeight  -scrollregion "0 0 $xcanmax(data) $threshGraphHeight" -bg #DDDDDD -xscrollcommand "$w.xs set"
  canvas $w.horzScale -width $xcanwindowmax -height  $horzScaleHeight  -scrollregion "0 0 $xcanmax(data) $horzScaleHeight" -bg #A9A9A9 -xscrollcommand "$w.xs set"
  
  #pack the horizontal (x) scrollbar
  pack $w.spacer1 -in $w.cfr -side left  -anchor e  
  pack $w.spacer2 -in $w.cfr -side bottom -anchor s  
  pack $w.can  -in $w.cfr -side left -anchor sw -padx 2 -expand yes -fill both
  #vertical scale/labels
  place $w.vertScale -in $w.can -relheight 1.0 -relx 0.0 -rely 0.5 -bordermode outside -anchor e
  #now place the vertical (y) scrollbar
  place $w.ys -in $w.vertScale -relheight 1.0 -relx 0.0 -rely 0.5 -bordermode outside -anchor e
  #now place the horizontal threshold Graph
  place $w.threshGraph -in $w.can -relwidth 1.0 -relx 0.5 -rely 1.0 -width 1 -bordermode outside -anchor n
  # horizontal scale/labels
  place $w.horzScale -in $w.threshGraph -relwidth 1.0 -relx 0.5 -rely 1.0 -bordermode outside -anchor n
  #now place the horizontal (x) scrollbar
  place $w.xs -in $w.horzScale -relwidth 1.0 -relx 0.5 -rely 1.0 -bordermode outside -anchor n

  # may need to specify B1-presses shift/nonshift separately...
  bind $w.can <ButtonPress-2>  [namespace code {getStartedMarquee %x %y 0 2 data}]
  bind $w.can <ButtonPress-3>  [namespace code {getStartedMarquee %x %y 0 3 data}]
  bind $w.can <Shift-ButtonPress-2>  [namespace code {getStartedMarquee %x %y 1 2 data}]
  bind $w.can <Control-ButtonPress-2>  [namespace code {timeBarJumpPress %x %y 0 data}]
  bind $w.can <Control-ButtonRelease-2>  [namespace code {timeBarJumpRelease %x %y 0 data}]
  bind $w.can <ButtonPress-1>  [namespace code {timeBarJumpPress %x %y 0 data}]
  bind $w.can <ButtonRelease-1>  [namespace code {timeBarJumpRelease %x %y 0 data}]
  bind $w.can <B2-Motion>  [namespace code {keepMovingMarquee %x %y 2 data}]
  bind $w.can <B3-Motion>  [namespace code {keepMovingMarquee %x %y 3 data}]
  bind $w.can <B1-Motion>  [namespace code {timeBarJump %x %y 0 data}]
  bind $w.can <Control-B1-Motion>  [namespace code {timeBarJump %x %y 0 data}]
  bind $w.can <ButtonRelease-2> [namespace code {letGoMarquee %x %y 2 data}]
  bind $w.can <ButtonRelease-3> [namespace code {letGoMarquee %x %y 3 data}]

  bind $w.vertScale <ButtonPress-2>  [namespace code {getStartedMarquee %x %y 0 2 vert}]
  bind $w.vertScale <ButtonPress-3>  [namespace code {getStartedMarquee %x %y 0 3 vert}]
  bind $w.vertScale <Shift-ButtonPress-2>  [namespace code {getStartedMarquee %x %y 1 2 vert}]
  bind $w.vertScale <ButtonPress-1>  [namespace code {timeBarJumpPress %x %y 0 vert}]
  bind $w.vertScale <ButtonRelease-1>  [namespace code {timeBarJumpRelease %x %y 0 vert}]
  bind $w.vertScale <B2-Motion>  [namespace code {keepMovingMarquee %x %y 2 vert}]
  bind $w.vertScale <B3-Motion>  [namespace code {keepMovingMarquee %x %y 3 vert}]
  bind $w.vertScale <B1-Motion>  [namespace code {timeBarJump %x %y 0 vert}]
  bind $w.vertScale <ButtonRelease-2> [namespace code {letGoMarquee %x %y 2 vert}]
  bind $w.vertScale <ButtonRelease-3> [namespace code {letGoMarquee %x %y 3 vert}]

  bind $w.horzScale <ButtonPress-2>  [namespace code {getStartedMarquee %x %y 0 2 horz}]
  bind $w.horzScale <ButtonPress-3>  [namespace code {getStartedMarquee %x %y 0 3 horz}]
  bind $w.horzScale <Shift-ButtonPress-2>  [namespace code {getStartedMarquee %x %y 1 2 horz}]
  bind $w.horzScale <ButtonPress-1>  [namespace code {timeBarJumpPress %x %y 0 horz}]
  bind $w.horzScale <ButtonRelease-1>  [namespace code {timeBarJumpRelease %x %y 0 horz}]
  bind $w.horzScale <B2-Motion>  [namespace code {keepMovingMarquee %x %y 2 horz}]
  bind $w.horzScale <B3-Motion>  [namespace code {keepMovingMarquee %x %y 3 horz}]
  bind $w.horzScale <B1-Motion>  [namespace code {timeBarJump %x %y 0 horz}]
  bind $w.horzScale <ButtonRelease-2> [namespace code {letGoMarquee %x %y 2 horz}]
  bind $w.horzScale <ButtonRelease-3> [namespace code {letGoMarquee %x %y 3 horz}]


  lower $w.spacer1 $w.cfr
  lower $w.spacer2 $w.cfr
  
  return
} 


proc ::timeline::reconfigureCanvas {} {
  variable xcanmax
  variable ycanmax
  variable w
  variable ysize 
  variable xcanwindowmax 
  variable ycanwindowmax
  variable threshGraphHeight
  variable horzScaleHeight
  variable vertScaleWidth
  variable xcanwindowStarting
  variable xcanwindowmax 
  variable dataOrigin
  variable xcol

  #in future, add to xcanwindowstarting if we widen window
  set xcanwindowmax  $xcanwindowStarting 


  #check if can cause trouble if no mol loaded...
  $w.can configure  -height $ycanwindowmax -width $xcanwindowmax 
  $w.horzScale configure  -height  $horzScaleHeight  -scrollregion "0 0 $xcanmax(data) $horzScaleHeight"
  $w.threshGraph configure -height  $threshGraphHeight  -scrollregion "0 0 $xcanmax(data) $horzScaleHeight"

  $w.vertScale configure  -width $vertScaleWidth -scrollregion "0 0 $vertScaleWidth $ycanmax(data)" 
  $w.horzScale delete all
  $w.vertScale delete all
  $w.can delete all

}

proc ::timeline::draw_traj_highlight {xStart xFinish} {

  variable w 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable xcol 
  variable ytopmargin 
  variable ytopmargin 
  variable scaley
  variable ybox  
  variable currentMol 
  variable rep 
  variable bond_rad 
  variable bond_res
  variable repColoring
  variable rectCreated

  #tlPutsDebug " now in draw_traj_highlight, xStart = $xStart, rectCreated = $rectCreated"
  $w.can delete trajHighlight 
  for {set i 0} {$i<=$dataValNum} {incr i} {
    if  {$dataVal(picked,$i) == 1} {
      set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
      
      set red 0 
      set green 0 
      set blue 255 
      #convert red blue green 0 - 255 to hex
      set hexred     [format "%02x" $red]
      set hexgreen   [format "%02x" $green]
      set hexblue    [format "%02x" $blue]
      

      ###draw highlight only if not yet drawn -- if rectCreated is 0, we may just cleared the rects
      ###     to redraw free of accumulated scaling errors
      ###if {($dataVal(pickedId,$i) == "null") || ($rectCreated == 0)} 
      
      #always draw trajBox
      #after prototype, merge this with normal highlight draw method
      #set trajBox [$w.can create rectangle  $xStart $ypos $xFinish [expr $ypos + ($scaley * $ybox)]  -fill "\#${hexred}${hexgreen}${hexblue}" -stipple gray25 -outline "" -tags [list dataScalable trajHighlight ] ]
      set trajBox [$w.can create rectangle  $xStart $ypos $xFinish [expr $ypos + ($scaley * $ybox)]  -fill "" -outline "\#${hexred}${hexgreen}${hexblue}" -tags [list dataScalable trajHighlight ] ]
      #puts "trajBox is $trajBox, xStart = $xStart, $xFinish = $xFinish"
      
      #$w.can lower $dataVal(pickedId,$i) vertScaleText 
      
      
      
    }
  }
}

proc ::timeline::drawVertHighlight  {} {

  variable w 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable xcol 
  variable ytopmargin 
  variable scaley
  variable ybox  
  variable currentMol 
  variable rep 
  variable bond_rad 
  variable bond_res
  variable repColoring
  variable rectCreated
  variable vertHighLeft
  variable vertHighRight
  variable usesFreeSelection

  set red 255
  set green 0
  set blue 255
  #convert red blue green 0 - 255 to hex
  set hexred     [format "%02x" $red]
  set hexgreen   [format "%02x" $green]
  set hexblue    [format "%02x" $blue]
  set highlightColorString    "\#${hexred}${hexgreen}${hexblue}" 

  for {set i 0} {$i<=$dataValNum} {incr i} {
    if  {$dataVal(picked,$i) == 1} {
      set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
      
      
      #draw highlight only if not yet drawn -- if rectCreated is 0, we may  just cleared the rects
      #     to redraw free of accumulated scaling errors
      if {($dataVal(pickedId,$i) == "null") || ($rectCreated == 0)} {

        set dataVal(pickedId,$i)  [$w.vertScale create rectangle  $vertHighLeft $ypos $vertHighRight [expr $ypos + ($scaley * $ybox)]  -fill $highlightColorString -outline "" -tags [list yScalable pickedHighlight] ]
        
        
        $w.vertScale lower $dataVal(pickedId,$i) vertScaleText 
        
      }
      
    }
  }

  
  #make selection string to display in VMD 
  set ll "" 
  set prevChain "Empty" 
  set prevSegname "Empty Segname String"

  #altered change for multi free selections
  #Cannot be held by chain  

  for {set i 0} {$i <= $dataValNum} {incr i} {
    if {$dataVal(picked,$i) == 1} {
      if $usesFreeSelection {
          append ll ") or ($dataVal(freeSelString,$i)"
      } else {
        if { ([string compare $prevChain $dataVal(chain,$i)] != 0) ||  ([string compare $prevSegname  $dataVal(segname,$i)] != 0)} {
          #chain or segname is new or has changed
          #tlPutsDebug "drawVertHighlight: dataVal(segname,$i)= >$dataVal(segname,$i)<"
          if {$dataVal(segname,$i) != "emptyval"} {
          append ll ") or (segname $dataVal(segname,$i) and chain $dataVal(chain,$i)  and resid $dataVal(resid,$i)"
          } else {
          append ll ") or (chain $dataVal(chain,$i)  and resid $dataVal(resid,$i)"
          }
        } else {
          append ll " $dataVal(resid,$i)"
        }
        set prevChain $dataVal(chain,$i)
        set prevSegname $dataVal(segname,$i)
      }
    }  
   }
  append ll ")"
  set ll [string trimleft $ll ") or " ]
  
  #check for the state when mol first loaded
  if {$ll ==""} {
    set ll "none"
  } 
  
  
  if {($rep($currentMol) != "null")} {
    #tlPutsDebug "About to create highlight: rep(currentMol) = $rep($currentMol)  currentMol = $currentMol"
    set theRepIndex [mol repindex $currentMol $rep($currentMol)]
    #get rep index from repname, will be -1 if has been deleted
    if { $theRepIndex != -1 } {
      mol modselect $theRepIndex $currentMol $ll
    } else {
      createHighlight  $ll      
    }
  } else {
    createHighlight  $ll        
  }
  return
}

proc ::timeline::showCursorHighlight {selText} {

  variable currentMol
  variable cursor_bond_rad
  variable cursor_bone_res
  variable cursorRepColor
  variable cursorRep

  if {($cursorRep($currentMol) != "null")} {

    set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)]
    if { $theCursorRepIndex != -1 } {
      mol modselect $theCursorRepIndex $currentMol $selText
    } else {
      createCursorHighlight  $selText      
    }
  } else {
    createCursorHighlight  $selText        
  }

}

proc ::timeline::hideCursorHighlight {} {

  variable currentMol
  variable cursor_bond_rad
  variable cursor_bone_res
  variable cursorRepColor
  variable cursorRep

  set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)]

  if {($cursorRep($currentMol) != "null")} {

    if {$theCursorRepIndex != -1} {
      mol showrep $currentMol $theCursorRepIndex 0
 
    } else {
      createCursorHighlight  $selText      
      set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)]
      mol showrep $currentMol $theCursorRepIndex 0
    }
  } else {
    createCursorHighlight  $selText        
    set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)]
    mol showrep $currentMol $theCursorRepIndex 0
  }

}

proc ::timeline::revealCursorHighlight {selText} {
#  tlPutsDebug "starting revealCursorHighlight, selText= $selText"
  #code copy from showCursorHighlight XXX 
  variable currentMol
  variable cursor_bond_rad
  variable cursor_bone_res
  variable cursorRepColor
  variable cursorRep

  if {($cursorRep($currentMol) != "null")} {

    set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)]
    if { $theCursorRepIndex != -1 } {
      mol showrep $currentMol $theCursorRepIndex 1
    } else {
      createCursorHighlight  $selText      
      set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)]
      mol showrep $currentMol $theCursorRepIndex 1
    }
  } else {
    createCursorHighlight  $selText        
    set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)]
    mol showrep $currentMol $theCursorRepIndex 1
  }

}


proc ::timeline::listPick {name element op} {
  
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
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable dataName 
  variable bond_rad 
  variable bond_res 
  variable repColoring
  variable rep 
  variable xcol 
  variable ysize 
  variable dataOrigin
  variable currentMol
  variable usesFreeSelection 
  # get the coordinates



  #later deal with top (and rep)  etc. for multi-mol use


  
  if {$vmd_pick_mol == $currentMol} {
   
    set sel [atomselect $currentMol "index $vmd_pick_atom"]
    
    set pickedresid [lindex [$sel get {resid}] 0] 
    set pickedchain  [lindex [$sel get {chain}] 0] 
    set pickedresname [lindex  [$sel get {resname}] 0]
    set pickedsegname [lindex [$sel get {segname}] 0] 
    
    set pickedOne -1
    #XXX must be changed for free Selections, deal with one atom can be on multiple selections.  So, is turned off fcor now.
    if {$usesFreeSelection==0} {
      for {set i 0} {$i <= $dataValNum} {incr i} {
        
        if {($dataVal(resid,$i) == $pickedresid) && ($dataVal(resname,$i) == $pickedresname) &&  ($dataVal(chain,$i) == $pickedchain) && ($dataVal(segname,$i)==$pickedsegname)} {
          set pickedOne $i
          
          break
        }
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

        for {set i 0} {$i <= $dataValNum} {incr i} {
          set dataVal(picked,$i) 0
          #tlPutsDebug "dataVal(pickedId,$i)= $dataVal(pickedId,$i)" 
          if {$dataVal(pickedId,$i) != "null"} {
            $w.can delete $dataVal(pickedId,$i)
            set dataVal(pickedId,$i) "null"
          }
        }
       #XXX the next line (vertscale delete) is a hack, a cleanup method, since current code loses track of pickedID. 
       # This should not really be required.  Choose this method or the pickedID method, not both.
       $w.vertScale delete pickedHighlight
      }
      
      
      set dataVal(picked,$pickedOne) 1
      
      drawVertHighlight 
      
      #scroll to picked
      set center [expr $ytopmargin + ($ybox * $scaley * $pickedOne) ] 
      set top [expr $center - 0.5 * $ycanwindowmax]
      
      if {$top < 0} {
        set top 0
      }
      set yfrac [expr $top / $ysize]
      $w.can yview moveto $yfrac
      $w.vertScale yview moveto $yfrac
    }
    
  }
  return
}



proc ::timeline::timeLineMain {} {
#------------------------
  #------------------------
  # main code starts here
  #vars initialized a few lines down
  

  #puts "in timeLineMain.."
  variable w 
  variable monoFont
  variable eo 
  variable x1 
  variable y1 
  variable startShiftPressed 
  variable startCanvas
  variable vmd_pick_shift_state 
  variable resCodeShowOneLetter 
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
  variable dataOrigin
  variable dataHash
  variable rectId
  #dataValNum is -1 if no data present, 
  variable dataValNum 
  variable dataValNumResSel
  variable dataName 
  variable ytopmargin 
  variable ybottommargin 
  variable xrightmargin
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable repColoring
  variable cursor_res
  variable cursor_bond_rad
  variable prevCursorObject
  variable prevCursorFrame
  variable bond_rad 
  variable bond_res
  variable rep 
  variable cursorRep
  variable cursorShown
  variable xcol 
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin 
  variable dataMax 
  variable xPosScaleVal
  variable currentMol
  variable fit_scalex
  variable fit_scaley
  variable everRedrawn
  variable usableMolLoaded 
  variable initializedVars
  variable prevScalet
  variable rectCreated
  variable windowShowing
  variable needsDataUpdate 
  variable numDataFrames
  variable numTrajFrames
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable partSelText 
  variable calledBySelChange
  variable highlightColor 
  variable colorscale 
  variable keyAtomSelString
  # check for usable molecule loaded
  set uml 0
  foreach mm [molinfo list] {
    if {([molinfo $mm get numatoms] > 0 )} {
      set uml 1
    }
  }
  set usableMolLoaded $uml
      

  #Init vars and draw interface
  if {$initializedVars == 0} {
    tlPutsDebug "about to initVars"
    initVars
    draw_interface
    makecanvas
    set initializedVars 1
    #watch the slider value, tells us when to redraw
    #this sets a trace for ::timeline::scaley
    
  } else {
    #even if no molecule is present
    reconfigureCanvas
  }   
  
  
  #-----
  #Now load info from the current molecule, must reload for every molecule change
  
  if {$usableMolLoaded} {
    #get info for new mol
    #set needsDataUpdate 0
    
    #The number of dataNames
    
    #Now to fill  a ((dataOrigin-1)+numDataFrames) x (dataValNumResSel +1) array
    #dataValNumResSel we'll be the number of objects we found with VMD search
    #if doing proteins and DNA, likely all residues, found with 'name CA' or 'name C3*;, etc.
    #the items 0 through dataOrigin-1 (count=dataOrigin) are 3 identifiers of residue
    #the items dataOrigin through dataOrigin+(numDataFrames-1) (count=numDataFrames) is the data for the frames.
    # The more general term (for both per-residue and free selections) will be dataValNum
    set dataValNumResSel -1
    #if no data is available, dataValNum will remain -1 
    #we are looking for dataVal when only a single res sel per line 

    # set  a new  trace below, only if dataValNum > -1  
    # following check likely is no longer necessary
    if {[molinfo $currentMol get numatoms] >= 1} {

      
      set currentMol_name [molinfo $currentMol get name]
      wm title $w "VMD Timeline  $currentMol_name (mol $currentMol) "
     
      set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]
      # gets 1 atom per protein or nucleic acid residue
      #below assumes sel retrievals in same order each time, fix this 
      #by changing to one retreival and chopping up result
      set datalist  [$sel get {resid resname chain segname}]
      puts "Info) Timeline is checking sequence info. for molecule $currentMol..."

      catch {unset dataHash}
      
      foreach elem $datalist {
        #XX optimize below, at least with foreach 
        incr dataValNumResSel
        #set picked state to false -- 'picked' is only non-numerical field
        set dataVal(picked,$dataValNumResSel) 0
        set dataVal(pickedId,$dataValNumResSel) "null"
       #XX reduce repeat splits 
        set theResid [ lindex [split $elem] 0]
        set dataVal(resid,$dataValNumResSel) $theResid 
        
        set dataVal(resname,$dataValNumResSel) [ lindex [split $elem] 1]
        set dataVal(rescode,$dataValNumResSel) [lookupCode $dataVal(resname,$dataValNumResSel)]
        set theChain [ lindex [split $elem] 2]
        set dataVal(chain,$dataValNumResSel) $theChain 
        set theSegname [lindex [split $elem] 3]
        #tlPutsDebug "segname =>$theSegname<"
        if {$theSegname =="{}"} then {
          #tlPutsDebug "segname was empty"
          set theSegname "emptyval"
        }
        set dataVal(segname,$dataValNumResSel) $theSegname 
        #tlPutsDebug "dataVal(segname,$dataValNumResSel) set to $dataVal(segname,$dataValNumResSel)"
        #for fast index searching later
        set dataHash($theResid,$theChain,$theSegname) $dataValNumResSel
      }
      #if datalist is length 0 (empty), dataValNum is still -1, 
      #So we must check before each use of dataValNum     
      
      #set the molec. structure so nothing is highlighted yet
      #set rep, cursorRep to "null" only if this molecule has never been seen
      #this way, can reuse old rep, when reselecting molecule number
      # we are using the existence of cursorRep($currentMol) to show we have seen this molecule before, so there should be some value for rep(currentMol) as well.
      if {[catch {set tester $cursorRep($currentMol)}]} {
        tlPutsDebug "in timeLineMain: set rep and cursorRep to null"
         set rep($currentMol) "null"
         set cursorRep($currentMol) "null"
      }

     
      #XX add these to above
      set prevCursorObject($currentMol) "null"
      set cursorShown($currentMol) 0
      set prevCursorFrame($currentMol) "null"
   } 
   # so dataValNum <= -1 if no sequence or atoms at all
    
    set dataValNum $dataValNumResSel   
    if {$dataValNum <= -1 } {
      puts "Info) Timeline couldn't find a sequence in this molecule.\n"
       return
    }
   
    
    
    
    #So dataValNum is number of the last dataVal.  It is also #elements -1, 
    
    #numDataFrames (and routines that use it)  will eventualy be changed
    # to reflect loaded data, and  multi-frame-data groups
    set numTrajFrames [molinfo $currentMol get numframes]
    set numDataFrames $numTrajFrames
    if {$numDataFrames >= 1} then {
           set fitNF $numDataFrames
    } else {
           set fitNF 1
    }
    set firstAnalysisFrame 0
    set  lastAnalysisFrame [expr $numDataFrames - 1]
    if {$lastAnalysisFrame < $firstAnalysisFrame} { 
      set lastAnalysisFrame $firstAnalysisFrame
    } 

    calcFitScaleXY

    set scaley 1.0
    set scalex $fit_scalex 
    tlPutsDebug "Timeline: Restarting data, scalex = $scalex, scaley= $scaley"
    #this trace only set if dataValNum != -1

    #Other variable-adding methods
    #should not change this number.  We trust $sel to always
    #give dataValNum elems, other methods might not work as well.
    
    
    #handle if this value is 0 or -1
    
    
    #don't need datalist anymore
    unset datalist 
    
    
    
    #now lets fill in some data/
    
    #new data, so need to redraw rects when time comes
    set rectCreated 0 
    #also set revScaley back to 1 
    set prevScaley scaley
    set prevScalex scalex 
    #value of dataNameNum is 2. last is numbered (dataNameLast) = 2
    
    
    #tlPutsDebug "About to act on calledbySellChange= $calledBySelChange   (partSelText= $partSelText)"
    if $calledBySelChange {
      set calledBySelChange 0
      recalc 
    } else {
      #fill in traj data with X position (very fast, but still slow for giant 10K-res molecs) 
      #tlPutsDebug "Timeline about to fill in with calcData, may not have cleared if first time" 
      set partSelText "all"
      calcSelEmpty -2
    } 
  } 
  
  #puts "time for first redraw, scales, min/max not calced"
  #redraw first time
  redraw name func ops
  
  #now draw the scales (after the data, we may need to extract min/max 
  #------
  #draw color legends, loop over all data fields
  #puts "dataName(resid) is $dataName(resid) dataName(resname) is $dataName(resname)"
  

  return
}

proc ::timeline::calcFitScaleXY {} {
  variable w
  variable numDataFrames
  variable xcanwindowmax
  variable ycanwindowmax
  variable ytopmargin
  variable ybottommargin
  variable ybox
  variable dataValNum
  variable xcol
  variable dataOrigin
  variable dataWidth
  variable fit_scalex
  variable fit_scaley 
  variable canvasWinOffsetX 
  variable canvasWinOffsetY 
  variable xcanwindowStarting 
  variable ycanwindowStarting 
  
  set canPixShownX [expr [winfo width $w] - $canvasWinOffsetX]
  if {$canPixShownX < 2} {set canPixShownX $xcanwindowStarting}
  set canPixShownY [expr [winfo height $w] -$canvasWinOffsetY]
  if {$canPixShownY < 2} {set canPixShownY $ycanwindowStarting}

  set scalingWidth $canPixShownX
  set scalingHeight $canPixShownY
 
  #if {$canPixShownX > ($xcanwindowmax - $xcol($dataOrigin))} {
  #  set scalingWidth $canPixShownX 
  #} else {
  #  set scalingWidth $xcanwindowmax
  #}
  
  #if {$canPixShownY > ($ycanwindowmax - $ytopmargin - $ybottommargin)} {
  #  set scalingHeight $canPixShownY
  #} else {
  #  set scalingHeight $ycanwindowmax
  #}
  tlPutsDebug "scalingHeight= >$scalingHeight<  scalingWidth= >$scalingWidth<  canPixShownY=>$canPixShownY< ycanwindowmax=>$ycanwindowmax<"
  
  set fit_scaley [expr (0.0 + $scalingHeight - $ytopmargin - $ybottommargin) / ($ybox * ($dataValNum + 1) ) ]

    #since we zero-count dataValNum.

  if {$numDataFrames >= 1} then {
         set fitNF $numDataFrames
  } else {
         set fitNF 1
  }
  set fit_scalex [expr (0.0 + $scalingWidth - $xcol($dataOrigin) ) / ($dataWidth * $fitNF ) ]
}


proc ::timeline::molChooseMenu {name function op} {
  variable w

  variable usableMolLoaded
  variable currentMol
  variable prevMol
  variable nullMolString
  variable dataOrigin
  variable numTrajFrames

  $w.mol.menu delete 0 end

  tlPutsDebug "In MolChooseMenu"

  set molList ""
  foreach mm [molinfo list] {
    if {([molinfo $mm get numatoms] > 0 )} {
      lappend molList $mm
      #add a radiobutton, but control via commands, not trace,
      #since if this used a trace, the trace's callback
      #would delete that trace var, causing app to crash.
      #variable and value only for easy button lighting
      ##$w.mol.menu add radiobutton -variable [namespace current]::currentMol -value $mm -label "$mm [molinfo $mm get name]" -command [namespace code "molChoose name function op"]
      $w.mol.menu add radiobutton -variable [namespace current]::currentMol -value $mm -label "$mm [molinfo $mm get name]"
    }
  }

  #set if any non-Graphics molecule is loaded
  if {$molList == ""} {
    set usableMolLoaded  0
    if {$prevMol != $nullMolString} {
      set currentMol $nullMolString
    }
  } else {

    #deal with first (or from-no mol state) mol load
    # and, deal with deletion of currentMol, if mols present
    # by setting the current mol to first usable mol in list
    if {($usableMolLoaded == 0) || [lsearch -exact $molList $currentMol]== -1 } {
      set usableMolLoaded 1
      #  
      # old line was: set currentMol [molinfo top]: works with auto-top
      # but top could be an unsable molec, instrad use first usable in list
      set currentMol [lindex $molList 0] 
      set numTrajFrames [molinfo $currentMol get numframes]
      tlPutsDebug "In MolChooseMenu, numTrajFrames= $numTrajFrames"
    }

  }


  
  
  return
}

proc ::timeline::setThresholdBounds {args} {
  variable clicked
  variable thresholdBoundMin 
  variable thresholdBoundMax

  # save old values 
  set oldmin $thresholdBoundMin 
  set oldmax $thresholdBoundMax

  set d .thresholdboundsdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set threshold bounds for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 220 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Set threshold bounds for Timeline:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Bottom value:}
    label $d.lb  -justify left -text {Top value:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::thresholdBoundMin
    entry $d.eb  -justify left -textvariable ::timeline::thresholdBoundMax
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::thresholdMakeGraph ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::thresholdBoundMin $timeline::oldmin ; set ::timeline::thresholdBoundMax $::timeline::oldmax ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::setParamsSASA {args} {
  variable clicked
  variable SASArad
  variable oldSASArad
   #the "any" refers to any function

  # save old values 
  set oldSASArad $SASArad


  set d .vmd_timeline_setparamssasadialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set SASA parameters for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {SASA parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {radius extension (A):}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::SASArad
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    set com "grab release $d; ::timeline::calcSASA; set ::timeline::clicked 1"
    button $d.ok -text {OK} -command $com

    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::SASArad $::timeline::oldSASArad ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::setParamsNativeContacts {args} {
  variable clicked
  variable nativeDist
  variable nativeFromSelList
  variable nativeToSel
  variable nativeRefFrame
  variable oldNativeDist
  variable oldNativeFromSelList
  variable oldNativeToSel
  variable oldNativeRefFrame
 #the "any" refers to any function

  # save old values 
  set oldNativeDist $nativeDist
  set oldNativeFromSelList $nativeFromSelList
  set oldNativeToSel $nativeToSel 
  set oldNativeRefFrame $nativeRefFrame

  set d .vmd_timeline_setparamsnativecontactsdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set native contact parameters for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Native contact parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Contact dist. cutoff (A):}
    label $d.lb  -justify left -text {From selection list:}
    label $d.lc  -justify left -text {To selection:}
    label $d.ld  -justify left -text {Reference frame:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb $d.lc $d.ld" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::nativeDist
    entry $d.eb  -justify left -textvariable ::timeline::nativeFromSelList
    entry $d.ec  -justify left -textvariable ::timeline::nativeToSel
    entry $d.ed  -justify left -textvariable ::timeline::nativeRefFrame
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb $d.ec $d.ed" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    set com "grab release $d; ::timeline::calcNativeContacts ${args}; set ::timeline::clicked 1"
    button $d.ok -text {OK} -command $com

    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::nativeDist $::timeline::oldNativeDist; set ::timeline::nativeFromSelList $::timeline::oldNativeFromSelList; set ::timeline::nativeToSel $::timeline::oldNativeToSel;set ::timeline::nativeRefFrame $::timeline::oldNativeRefFrame; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}
proc ::timeline::setParamsInterSelContacts {args} {
  variable clicked
  variable interDist
  variable interSelList
  variable interSelListPairs
  variable interFromResSel
  variable interToSel
  variable interSetMethod 
  variable oldInterDist
  variable oldInterSelList
  variable oldInterSelListPairs
  variable oldInterFromResSel
  variable oldInterToSel
  variable oldInterSetMethod 
   
   #the "any" refers to any function

  # save old values 
  set oldInterDist $interDist
  set oldInterSelList $interSelList
  set oldInterSelListPairs $interSelListPairs
  set oldInterFromResSel $interFromResSel
  set oldInterToSel $interToSel
  set oldInterSetMethod $interSetMethod 
 

  set d .vmd_timeline_setparamsintercontactsdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set inter-selection contact parameters for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  #wm minsize  $d 320 120  
  wm minsize  $d 400 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Inter-selection contact parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    #label $d.cba  -justify left -text {Placeholder.}
    # Have buttons set 
    radiobutton $d.cbb -text "Selection list (all combinations)" -value 0 -variable [namespace current]::interSetMethod  
    radiobutton $d.cbd -text "Selection list (pairs only)" -value 2 -variable [namespace current]::interSetMethod  
    radiobutton $d.cbf -text "Selection residues to selection" -value 1 -variable [namespace current]::interSetMethod  

    #tlPutsDebug "about to pack cbX radiobuttons /  label"

    #set i 1
    #grid columnconfigure $d.top 0 -weight 2
    #foreach l "$d.cba $d.cbb $d.cbc" {
    #    tlPutsDebug "before pack for $l"
    #    pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
    #    tlPutsDebug "before grid for $l"
    #    grid $l -in $d.top -column 0 -row $i -sticky w 
    #    tlPutsDebug "after grid for $l"
    #    incr i
    #}
    #tlPutsDebug "finished with pack cbX radiobuttons /  label"

    label $d.la  -justify left -text {Contact dist. cutoff (A):}
    label $d.lc  -justify left -text {Selection list:}
    label $d.le  -justify left -text {Selection list (even # of items):}
    label $d.lg -justify left -text {From residues in selection:}
    label $d.lh  -justify left -text {To selection:}
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.la $d.cbb $d.lc $d.cbd $d.le $d.cbf $d.lg $d.lh" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }
  
    # entries 
    entry $d.ea  -justify left -textvariable ::timeline::interDist
    entry $d.ec  -justify left -textvariable ::timeline::interSelList
    entry $d.ee  -justify left -textvariable ::timeline::interSelListPairs
    entry $d.eg  -justify left -textvariable ::timeline::interFromResSel
    entry $d.eh  -justify left -textvariable ::timeline::interToSel

    #placeholders (blank spaces)  
    label $d.phb  -justify left -text {}
    label $d.phd  -justify left -text {}
    label $d.phf  -justify left -text {}

    # e is for entry, ph is for placeholder (blank space)
    set i 1
    grid columnconfigure $d.top 2 -weight 2
    foreach l "$d.ea $d.phb $d.ec $d.phd $d.ee $d.phf $d.eg $d.eh" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
   
    #add element activate/inactivate commands for radiobuttons
    $d.cbb configure -command  "$d.lc config -state active; $d.ec config -state normal; $d.le config -state disabled; $d.ee config -state disabled ; $d.lg config -state disabled; $d.eg config -state disabled; $d.lh config -state disabled; $d.eh config -state disabled"
    $d.cbd configure -command  "$d.lc config -state disabled; $d.ec config -state disabled; $d.le config -state active; $d.ee config -state normal; $d.lg config -state disabled; $d.eg config -state disabled; $d.lh config -state disabled; $d.eh config -state disabled"
    $d.cbf configure -command  "$d.lc config -state disabled; $d.ec config -state disabled; $d.le config -state disabled; $d.ee config -state disabled ; $d.lg config -state active; $d.eg config -state normal; $d.lh config -state active; $d.eh config -state normal"
    #$d.cbf configure -command  "$d.lc config -state disabled; $d.ec config -state disabled; $d.le config -state active;  $d.ee config -state normal; $d.lf config -state active; $d.ef config -state normal" 


    #set var so a radiobutton sets just-assigned active/inactive states
    $d.cbb invoke    

    # buttons
    set com "grab release $d; ::timeline::calcInterSelContacts ${args}; set ::timeline::clicked 1"
    button $d.ok -text {OK} -command $com

    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::interDist $::timeline::oldInterDist; set ::timeline::interSelList $::timeline::oldInterSelList;set ::timeline::interSelListPairs $::timeline::oldInterSelListPairs;  set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::setParamsGlobalCcss {args} {
  variable clicked

  variable ccssMapFile 
  variable ccssMapres
  variable ccssSpacing
  variable ccssThreshold
  variable ccssUseThreshold
  variable ccssUseSpacing
  variable ccssSelMethod 
  variable ccssSelList
  variable oldCcssMapFile 
  variable oldCcssMapres
  variable oldCcssSpacing
  variable oldCcssThreshold
  variable oldCcssSelMethod 
  variable oldCcssSelList
  variable oldCcssUseThreshold
  variable oldCcssUseSpacing

 # save old values 
  set oldCcssMapFile $ccssMapFile
  set oldCcssMapres $ccssMapres
  set oldCcssSpacing $ccssSpacing
  set oldCcssThreshold $ccssThreshold
  set oldCcssUseThreshold $ccssUseThreshold
  set oldCcssUseSpacing $ccssUseSpacing
  set oldCcssSelMethod $ccssSelMethod
  set oldCcssSelList $ccssSelList

  set d .vmd_timeline_setparamsglobalccssdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set cross corr. parameters for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
    radiobutton $d.cbi -text "Generate selection list from contigous sec. structure " -value 0 -variable [namespace current]::ccssSelMethod  
    radiobutton $d.cbj -text "Custom selection list" -value 1 -variable [namespace current]::ccssSelMethod  
    #checkbutton $d.cbf -text "Use a map threshold:" -variable [namespace current]::ccssUseThreshold -onvalue 1 -offvalue 0 
    checkbutton $d.cbg -text "Use a map threshold:" -variable ::timeline::ccssUseThreshold -onvalue 1 -offvalue 0 
    checkbutton $d.cbe -text "User Defined Grid Spacing:" -variable ::timeline::ccssUseSpacing -onvalue 1 -offvalue 0 
    #-command "puts \"ccssUseThreshold= $::timeline::ccssUseThreshold\""
    label $d.head -justify center -relief raised -text {Global cross corr. sec. struct parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 3 -sticky snew 
    label $d.lc  -justify left -text {Density Map file:}
    label $d.ld  -justify left -text {Map resolution (A):}
    label $d.lf  -justify left -text {Map spacing (A):}
    label $d.lh  -justify left -text {Map threshold:}
    label $d.lk  -justify left -text {Selection list:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.lc $d.ld $d.cbe $d.lf $d.cbg $d.lh $d.cbi $d.cbj $d.lk" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    # These entries need configuration buttons.  Or make them uneditable, and set with buttons.  Or show end of the filename.
    entry $d.ec  -justify left -textvariable ::timeline::ccssMapFile

    entry $d.ed  -justify left -textvariable ::timeline::ccssMapres
    entry $d.ef  -justify left -textvariable ::timeline::ccssSpacing
    entry $d.eh  -justify left -textvariable ::timeline::ccssThreshold
    entry $d.ek  -justify left -textvariable ::timeline::ccssSelList
    #placeholder (blank spaces)
    label $d.phe  -justify left -text {}
    label $d.phg  -justify left -text {}
    label $d.phi  -justify left -text {}
    label $d.phj  -justify left -text {}
    
    set i 1
    grid columnconfigure $d.top 1 -weight 2

    # set up file name button
    #delay evaluation of tk_getOpenFile by making square brackets literal with back-slashes


    set com "set ::timeline::ccssMapFile \[tk_getOpenFile -initialfile {$ccssMapFile} -title \"Choose density map file for cross corr.\" -parent $d -filetypes { [list { {DX files} {.dx} } {{All files} {*} }]} \]"
    button $d.buttonFilenameC -text {File...} -command $com
    #XX make the other 2 here... 
    #
    #with file buttons:
    foreach {l b} "$d.ec $d.buttonFilenameC"  {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        #pack $d.filenameA  -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        grid $b  -in $d.top -column 2 -row $i -sticky w 
        incr i
    }
        
    #without file buttons:
    foreach l "$d.ed $d.phe $d.ef $d.phg $d.eh $d.phi $d.phj $d.ek" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m

        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }

    #add element activate/inactivate commands for radiobuttons
    $d.cbi configure -command  "$d.lk config -state disabled; $d.ek config -state disabled"; 
#puts "ccssSelMethod= $::timeline::ccssSelMethod" 
    $d.cbj configure -command  "$d.lk config -state active; $d.ek config -state normal"; 
#puts "ccssSelMethod= $::timeline::ccssSelMethod"

    #add elelent activate/inactivate command for threshold
    

    #set var so a radiobutton sets just-assigned active/inactive states
   
    set com1 "$d.lg config -state active; $d.eg config -state normal"
    set com2 "$d.lg config -state disabled; $d.eg config -state disabled"
    ###$d.cbf configure -command puts "ccssUseThreshold=  >$::timeline::ccssUseThreshold<"; puts "ccssSelMethod= >$::timeline::ccssSelMethod<"; if {$::timeline::ccssUseThreshold} { $d.lg config -state active; $d.eg config -state normal } else { $d.lg config -state disabled; $d.eg config -state disabled } 
    #$d.cbf configure -command {puts "ccssUseThreshold=  >$::timeline::ccssUseThreshold<"; puts "ccssSelMethod= >$::timeline::ccssSelMethod<" }




 
    # buttons
    #set com "::timeline::clearData ; ::timeline::calcGlobalCcss ${args}; set ::timeline::clicked 1"
    set com "grab release $d; ::timeline::calcCC; set ::timeline::clicked 1"
    button $d.ok -text {OK} -command $com

    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::ccssMapFile $::timeline::oldCcssMapFile;  set ::timeline::ccssSpacing $::timeline::oldCcssSpacing; set ::timeline::ccssThreshold $::timeline::oldCcssThreshold; set ::timeline::ccssUseThreshold $::timeline::oldCcssUseThreshold; set ::timeline::ccssUseSpacing $::timeline::oldCcssUseSpacing; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::swapCcssUseContents {d ccssUseThreshold} {
  global $d
  #puts "ccssUseThreshold=  >$::timeline::ccssUseThreshold<"
 # puts "ccssSelMethod= >$::timeline::ccssSelMethod<"; 
  if {$::timeline::ccssUseThreshold} {
    $d.lg config -state active; $d.eg config -state normal
  } else {
    $d.lg config -state disabled; $d.eg config -state disabled
  }
}

proc ::timeline::setParamsSaltBridge {args} {
  variable clicked
  variable ONdist
  variable oldONdist
   #the "any" refers to any function

  # save old values 
  set oldONdist $ONdist


  set d .vmd_timeline_setparamssaltbridgedialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set salt bridge parameters for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Salt bridge parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {O-N bond dist. cutoff (A):}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::ONdist
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    set com "grab release $d; ::timeline::calcSaltBridge ${args}; set ::timeline::clicked 1"
    button $d.ok -text {OK} -command $com

    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::ONdist $timeline::oldONdist ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::setParamsRMSF {args} {
  variable clicked
  variable RMSFstepSize 
  variable RMSFwindowWidth 
  variable oldRMSFstepSize 
  variable oldRMSFwindowWidth 
   #the "any" refers to any function

  # save old values 
  set oldRMSFstepSize  $RMSFstepSize
  set oldRMSFwindowWidth $RMSFwindowWidth


  set d .vmd_timeline_setparamsrmsf
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set RMSF parameters for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {RMSF parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Window width (frames):}
    label $d.lb  -justify left -text {Step size (frames):}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::RMSFwindowWidth
    entry $d.eb  -justify left -textvariable ::timeline::RMSFstepSize
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    set com "grab release $d; ::timeline::calcRMSF; set ::timeline::clicked 1"
    button $d.ok -text {OK} -command $com

    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::RMSFstepSize  $::timeline::oldRMSFstepSize; set ::timeline::RMSFwindowWidth $::timeline::oldRMSFwindowWidth ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::setParamsHbonds {args} {
  variable clicked
  variable hbondDistCutoff
  variable hbondAngleCutoff 
  variable oldHbondDistCutoff
  variable oldHbondAngleCutoff 
   #the "any" refers to any function

  # save old values 
  set oldHbondDistCutoff $hbondDistCutoff
  set oldHbondAngleCutoff $hbondAngleCutoff


  set d .vmd_timeline_setparamshbondsdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set H-bond parameters for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Hydrogen bond parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {H-bond dist. cutoff (A):}
    label $d.lb  -justify left -text {H-bond angle cutoff (deg.):}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::hbondDistCutoff
    entry $d.eb  -justify left -textvariable ::timeline::hbondAngleCutoff
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    set com "grab release $d; ::timeline::calcHbonds ${args}; set ::timeline::clicked 1"
    button $d.ok -text {OK} -command $com

    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::hbondDistCutoff $timeline::oldHbondDistCutoff; set ::timeline::hbondAngleCutoff $::timeline::oldHbondAngleCutoff; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::filterData {args} {
  variable clicked
  variable filterMin
  variable filterMax
  variable filterFrameStart
  variable filterFrameEnd
  variable filterNumFramesRequired
  variable oldFilterMin
  variable oldFilterMax
  variable oldFilterFrameStart
  variable oldFilterFrameEnd
  variable oldFilterNumFramesRequired
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  # save old values 
  set oldFilterMin $filterMin
  set oldFilterMax $filterMax
  set oldFilterFrameStart $filterFrameStart
  set oldFilterFrameEnd $filterFrameEnd
  set oldFilterNumFramesRequired $filterNumFramesRequired
   

  set d .vmd_timeline_filterdatadialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Filter data for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Filter parameters:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Min. val.:}
    label $d.lb  -justify left -text {Max. val.:}
    label $d.lc  -justify left -text {First frame:}
    label $d.ld  -justify left -text {Last frame:}
    label $d.le  -justify left -text {Min. frames req'd:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb $d.lc $d.ld $d.le" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::filterMin
    entry $d.eb  -justify left -textvariable ::timeline::filterMax
    entry $d.ec  -justify left -textvariable ::timeline::filterFrameStart
    entry $d.ed  -justify left -textvariable ::timeline::filterFrameEnd
    entry $d.ee  -justify left -textvariable ::timeline::filterNumFramesRequired
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb $d.ec $d.ed $d.ee" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::copyDataFiltered $::timeline::filterMin $::timeline::filterMax $::timeline::filterFrameStart $::timeline::filterFrameEnd $::timeline::filterNumFramesRequired ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::filterMin $::timeline::oldFilterMin; set ::timeline::filterMax $::timeline::oldFilterMax; set ::timeline::filterFrameStart $::timeline::oldFilterFrameStart; set ::timeline::filterNumFramesRequired $::timeline::oldFilterNumFramesRequired; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}


proc ::timeline::setAnalysisFrames {args} {
  variable clicked
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable oldFirstAnalysisFrame
  variable oldLastAnalysisFrame
   #the "any" refers to any function

  # save old values 
  set oldFirstAnalysisFrame $firstAnalysisFrame
  set oldLastAnalysisFrame $lastAnalysisFrame


  set d .vmd_timeline_setanalysisframesialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set analysis frames for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Analysis frame range:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {First frame:}
    label $d.lb  -justify left -text {Last frame:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::firstAnalysisFrame
    entry $d.eb  -justify left -textvariable ::timeline::lastAnalysisFrame
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::recalc ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::firstAnalysisFrame $timeline::oldFirstAnalysisFrame ; set ::timeline::lastAnalysisFramec $::timeline::oldLastAnalysisFrame ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::copyToUser {args} {
  variable clicked
  variable whichUserField 
  variable copyUserAfterCalc
   #the "any" refers to any function

  # save old values 
  set oldWhichUserField $whichUserField
  set oldCopyUserAfterCalc $copyUserAfterCalc

  set d .vmd_timeline_copyToUserDialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Copy Timeline Data to VMD User field}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Copy Timeline data to a User field:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Copy Timeline data to:}
    label $d.lb  -justify left -text {Copy after every calc.:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }
     
    #popup menu
    menubutton $d.ea -relief raised -bd 2 -width 5 -textvariable [namespace current]::whichUserField -direction flush -menu $d.ea.whichUserMenu
    menu $d.ea.whichUserMenu -tearoff no
    $d.ea.whichUserMenu add radiobutton -variable [namespace current]::whichUserField -value "user" -label "user" 
    $d.ea.whichUserMenu add radiobutton -variable [namespace current]::whichUserField -value "user2" -label "user2" 
    $d.ea.whichUserMenu add radiobutton -variable [namespace current]::whichUserField -value "user3" -label "user3" 
    $d.ea.whichUserMenu add radiobutton -variable [namespace current]::whichUserField -value "user4" -label "user4" 
    
    #check box
    checkbutton $d.eb -variable ::timeline::copyUserAfterCalc -onvalue 1 -offvalue 0 

    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::copyUser ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::whichUserField $::timeline::oldWhichUserField ; set ::timeline::copyUserAfterCalc $::timeline::oldCopyUserAfterCalc; set timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::setAnyResFunc {args} {
  variable clicked
  variable anyResFuncDesc
  variable anyResFuncName
   #the "any" refers to any function

  # save old values 
  set oldAnyResFuncDesc $anyResFuncDesc
  set oldAnyResFuncName $anyResFuncName

  set d .vmd_timeline_setanyresdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set Every-Residue Function for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Set per-residue function:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Function (TCL proc)}
    label $d.lb  -justify left -text {Label for the function:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::anyResFuncName
    entry $d.eb  -justify left -textvariable ::timeline::anyResFuncDesc
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::recalc ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::anyResFuncName $::timeline::oldAnyResFuncName ; set ::timeline::anyResFuncDesc $::timeline::oldAnyResFuncDesc ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}


proc ::timeline::setScaling {args} {
  variable clicked
  variable trajMin
  variable trajMax 

  # save old values 
  set oldmin $trajMin
  set oldmax $trajMax

  set d .scalingdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set Scaling for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 220 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Set scaling for timeline:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Bottom value:}
    label $d.lb  -justify left -text {Top value:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::trajMin
    entry $d.eb  -justify left -textvariable ::timeline::trajMax
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::showall 1; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::trajMin $::timeline::oldmin ; set ::timeline::trajMax $::timeline::oldmax ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}


proc ::timeline::printCanvas {} {
  variable w
  set filename "VMD_Timeline_Window.eps"
  set filename [tk_getSaveFile -initialfile $filename -title "VMD Timeline Print" -parent $w -filetypes [list {{Encapsulated Postscript Files} {.eps}} {{All files} {*} }] ]
  if {$filename != ""} {
    $w.can postscript -file $filename
  }
  
  return
}





proc ::timeline::getStartedMarquee {x y shiftState whichButtonPressed whichCanvas} {

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
  variable everRedrawn
  variable usableMolLoaded
  variable marqueeButton
  variable dataValNum 
  
  if {$usableMolLoaded && ($dataValNum > -1)} {

    #calculate offset for canvas scroll
    set startShiftPressed $shiftState   
    set marqueeButton $whichButtonPressed
    set startCanvas $whichCanvas 
    #get actual name of canvas
    switch -exact $startCanvas {
      data {set drawCan can}
      vert {set drawCan vertScale}
      horz {set drawCan horzScale}
      default {
          #puts "problem with finding canvas..., startCanvas= >$startCanvas<"
      } 
    }   
    set x [expr $x + $xcanmax($startCanvas) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax($startCanvas) * [lindex [$w.$drawCan yview] 0]] 
    tlPutsDebug "getStarted, y= $y, yview =  [$w.$drawCan yview]" 
    set x1 $x
    set y1 $y
    

    #puts "getStartedMarquee x= $x  y= $y, startCanvas= $startCanvas" 
    #Might have other canvas tools in future..   
    # Otherwise, start drawing rectangle for selection marquee
    
    
   if {$marqueeButton==2} {
     set outlineColor "blue"
   } else {
     set outlineColor "green"
   }   
    set so [$w.$drawCan create rectangle $x $y $x $y -fill {} -outline $outlineColor]
    set eo $so
  } 
  return
}


proc ::timeline::molChoose {name function op} {

  variable scaley
  variable w
  variable currentMol
  variable prevMol
  variable nullMolString
  variable rep 
  variable everRedrawn
  variable usableMolLoaded
  variable needsDataUpdate
  variable windowShowing
  variable dataOrigin 
  variable usesFreeSelection

  #this does complete restart
  #can do this more gently...
  
  #trace vdelete scaley w [namespace code redraw]
  #trace vdelete ::vmd_pick_atom w  [namespace code listPick] 
  
  #if there's a mol loaded, and there was an actual non-graphic mol last
  #time, and if there has been a selection, and thus a struct highlight
  #rep made, delete the highlight rep.
  if {($usableMolLoaded)  && ($everRedrawn) && ($prevMol != $nullMolString) && ($rep($prevMol) != "null")} {
    #catch this since currently is exposed to user, so 
    #switching/reselecting  molecules can fix problems.
    #determine if this mol exists...
    if  {[lsearch -exact [molinfo list] $prevMol] != -1}  {
      #determine if this rep exists (may have been deleted by user)
        
      set theRepIndex [mol repindex $prevMol $rep($prevMol)]
      if {$theRepIndex != -1} { 
        mol delrep $theRepIndex $prevMol 
      }
    }
    
  }

  set prevMol $currentMol

  #can get here when window is not displayed if:
  #   molecule is loaded, other molecule delete via Molecule GUI form.
  # So, we'll only redraw (and possible make a length (wallclock) call
  # to chosen analysis method) if timeline window is showing
  
  set needsDataUpdate 1
  set usesFreeSelection 0

  if {$windowShowing} {
    set needsDataUpdate 0
    #set this immediately, so other  calls can see this
    
    [namespace current]::timeLineMain
  }


  
  #reload/redraw stuff, settings (this may elim. need for above lines...)
  
  
  #change molecule choice and redraw if needed (visible && change) here...
  #change title of window as well
  ##wm title $w "VMD Timeline  $currentMol_name (mol $currentMol) "
  
  #reload sutff (this may elim. need for above lines...)

  return
}

proc ::timeline::keepMovingMarquee {x y whichButtonPressed whichCanvas} {

  variable x1 
  variable y1 
  variable so 
  variable w 
  variable xcanmax 
  variable ycanmax
  variable startCanvas
  variable everRedrawn
  variable usableMolLoaded
  #get actual name of canvas
  switch -exact $startCanvas {
    data {set drawCan can}
    vert {set drawCan vertScale}
    horz {set drawCan horzScale}
    default {tlPutsDebug "Info) Timeline: had problem with finding canvas (moving marquee)..., startCanvas= $startCanvas"}
  } 

  
  if {$usableMolLoaded} {

    #next two lines for debugging only
    set windowx $x
    set windowy $y 
    #calculate offset for canvas scroll
    set x [expr $x + $xcanmax($startCanvas) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax($startCanvas) * [lindex [$w.$drawCan yview] 0]] 
    
    
    
    
    $w.$drawCan coords $so $x1 $y1 $x $y
  }
  return
}

proc ::timeline::initPicked {} {
  variable dataVal
  variable dataValNum
  variable w
  for {set i 0} {$i <= $dataValNum} {incr i} {
    set dataVal(picked,$i) 0
    set dataVal(pickedId,$i) "null"
  }
}

proc ::timeline::clearAllPicked {} {
  tlPutsDebug "now to clearAllPicked"
  variable dataVal
  variable dataValNum
  variable w
  for {set i 0} {$i <= $dataValNum} {incr i} {
    set dataVal(picked,$i) 0
    if {$dataVal(pickedId,$i) != "null"} {
      $w.vertScale delete $dataVal(pickedId,$i)
      set dataVal(pickedId,$i) "null"
    }
  }
  tlPutsDebug "done with clearAllPicked"
}
proc ::timeline::letGoMarquee {x y whichButtonPressed whichCanvas} {


  variable x1 
  variable y1 
  variable startShiftPressed 
  variable marqueeButton
  variable startCanvas
  variable so 
  variable eo 
  variable w 
  variable xsize
  variable ysize
  variable xcanmax
  variable ycanmax
  variable ySelStart 
  variable ySelFinish 
  variable ybox 
  variable ytopmargin 
  variable ybottommargin 
  variable xcanwindowmax
  variable ycanwindowmax
  variable vertTextSkip 
  variable scalex 
  variable scaley 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable dataName 
  variable bond_rad 
  variable bond_res 
  variable repColoring
  variable rep 
  variable xcol
  variable currentMol
  variable everRedrawn
  variable usableMolLoaded
  variable dataOrigin
  variable dataWidth 
  variable ycanwindowmax  
  variable numDataFrames
  variable fit_scalex
  variable fit_scaley
  variable userScalex
  variable userScaley
  variable userScaleBoth
  #set actual name of canvas
  switch -exact $startCanvas {
    data {set drawCan can}
    vert {set drawCan vertScale}
    horz {set drawCan horzScale}
    default {puts "Info) Timeline: problem with finding canvas (moving marquee)..., startCanvas= $startCanvas"}
  }

  if {$usableMolLoaded && $everRedrawn && ($dataValNum > -1) } {
    #calculate offset for canvas scroll
    set x [expr $x + $xcanmax(data) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax(data) * [lindex [$w.$drawCan yview] 0]] 
    #tlPutsDebug "yview=  [$w.$drawCan yview]    y= $y"
    #compute the frame at xSelStart
    if {$x1 < $x} {
      set xSelStart $x1
      set xSelFinish $x
    }  else {
      set xSelStart $x
      set xSelFinish $x1
    }
    #puts "xSelStart is $xSelStart xSelFinish is $xSelStart" 
    
    #in initVars we hardcode dataOrigin to be 3
    #later, there may be many field-groups that can be stretched 
    #tlPutsDebug "scalex= $scalex, dataWidth= $dataWidth  xSelStart= $xSelStart"
    set selStartFrame [expr  int (($xSelStart - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]
    set selFinishFrame [expr int( ($xSelFinish - $xcol($dataOrigin))/ ($dataWidth * $scalex) ) ]
    #puts "checking limits, numDataFrames = $numDataFrames, selStartFrame= $selStartFrame   selFinishFrame= $selFinishFrame"
    if { $selStartFrame < 0} {
      set selStartFrame  0
    } 
   
    if { $selFinishFrame <  0 } {
      set selFinishFrame  0 
    }
    if { $selStartFrame >=$numDataFrames } {
      set selStartFrame  [expr $numDataFrames -1]
    } 
   
    if { $selFinishFrame >= $numDataFrames} {
      set selFinishFrame [expr $numDataFrames -1]
    }
    #puts "selected frames $selStartFrame to   $selFinishFrame"

    if {$y1 < $y} {
      set ySelStart $y1
      set ySelFinish $y}  else {
        
        set ySelStart $y
        set ySelFinish $y1
      }
    
    set startObject [expr 0.0 + ((0.0 + $ySelStart - $ytopmargin) / ($scaley * $ybox))]
    set finishObject [expr 0.0 + ((0.0 + $ySelFinish - $ytopmargin) / ($scaley * $ybox))]
    
    
    if {$startShiftPressed == 1} {
      set singleSel 0
    } else {
      set singleSel 1
    }
    
    if {$startObject < 0} {set startObject 0}
    if {$finishObject < 0} {set finishObject 0}
    if {$startObject > $dataValNum} {set startObject   $dataValNum }
    if {$finishObject > $dataValNum} {set finishObject $dataValNum }
    set startObject [expr int($startObject)]
    set finishObject [expr int($finishObject)]
    
    #optimizations obvious, much math repeated...
    set xStartFrame [expr  ( ($selStartFrame  ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    #stretch across width of ending frame
    set xFinishFrame [expr  ( ($selFinishFrame+ 1.0) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    #tlPutsDebug "  xStartFrame= $xStartFrame    xFinishFrame= $xFinishFrame xsize= $xsize"
 

    if {$marqueeButton==2}  {
      #highlight for animation
      
      #clear all if click/click-drag, don't clear if shift-click, shift-click-drag
      
      if {$singleSel == 1} {
        clearAllPicked
      } else {
        
        #just leave alone 
      }
      
      
      
      
      #set flags for selection
      for {set i $startObject} {$i <= $finishObject} {incr i} {
        set dataVal(picked,$i) 1
      }
      
      
      
      set field 0
      #note that the column will be 0, but the data will be from picked
      
      drawVertHighlight 
      
      
      #puts "now to delete outline, eo= $eo" 
      $w.$drawCan delete $eo
      $w.can delete timeBarRect 
      #now that highlight changed, can animate
      #if single selection in frame area, animate, then jump to that frame

           if {$startCanvas=="data"} { 
        if {  $selStartFrame >= 0 } {
          if {$selFinishFrame > $selStartFrame} {
            #draw a box to show selected animation

            

            #puts "now to  draw_traj_highlight $xStartFrame $xFinishFrame"
            draw_traj_highlight $xStartFrame $xFinishFrame

            set xTimeBarEnd  [expr  ( ($selStartFrame + 1.0) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
            
            #set timeBar [$w.can create rectangle  $xStartFrame 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "\#000000" -stipple gray50 -outline "" -tags [list dataScalable timeBarRect ] ]
            set timeBar [$w.can create rectangle  $xStartFrame 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "" -outline "\#000000"  -tags [list dataScalable timeBarRect ] ]
            set timeBar2 [$w.can create rectangle  $xStartFrame 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "" -outline "\#A0A0A0" -dash .  -tags [list dataScalable timeBarRect ] ]
            display update ui
            #maybe store top, and restore it afterwards
            mol top $currentMol
            #make this controllable, and animation repeatable without
            #need to reselect
            #soon, we move this loop, and anim will happen only at button push XXX
            for {set r 1} {$r <= 1} {incr r} {
              for {set f $selStartFrame} {$f <= $selFinishFrame} {incr f} {
                #puts "time for draw = [time { drawTimeBar $f}]"
                #puts "time for disp update = [time {display update ui}]" 
                animate goto $f
                drawTimeBar $f
                display update ui 
              }
            }
            $w.can delete timeBarRect 

          } 
          animate goto $selStartFrame
          drawTimeBar $selStartFrame 
          tlPutsDebug "now jumped to frame $selStartFrame for molecule $currentMol"
        }
      } 

    } else {
      $w.$drawCan delete $eo
      # zoom to requested position
      tlPutsDebug "  START scale calcs, scalex= $scalex  scaley= $scaley"
     
      if { ([expr abs($x1-$x)]<=2) && ([expr abs ($y1==$y)]<=2) } {
         #zoom out; hardcoded limit of 3 to avoid mouse wobble
        tlPutsDebug "zoom out -- x= $x  x1= $x1  ySelStart= $ySelStart startObject= $startObject ySelFinish= $ySelFinish finishObject=$finishObject startShiftPressed= $startShiftPressed"
         set scaleFacX 0.8
         set scaleFacY 0.8
         # place these in middle

         set leftborder [expr $x - 0.5 * $xcanwindowmax]
         if {$leftborder < 0}  {
           set leftborder 0
         } 

         set topborder [expr $y - 0.5 * $ycanwindowmax]
         if {$topborder < 0} {
           set topborder 0
         }
         set xf_low [expr $leftborder/$xsize]
         set yf_low [expr $topborder/$ysize]
      } else { 
        tlPutsDebug "zoom in x= $x  x1= $x1 ySelStart= $ySelStart startObject= $startObject ySelFinish= $ySelFinish finishObject=$finishObject startShiftPressed= $startShiftPressed"
        set marqueeBoxesHeight [expr $finishObject - $startObject]
        set marqueeBoxesWidth [expr $selFinishFrame- $selStartFrame]
     
       if {$marqueeBoxesWidth<= 3} then {set marqueeBoxesWidth 3}
       if {$marqueeBoxesHeight<=3} then {set marqueeBoxesHeight 3}
       tlPutsDebug "marqueeBoxesWidth= $marqueeBoxesWidth  marqueeBcxesHeight= $marqueeBoxesHeight\n    dataWidth= $dataWidth   ybox= $ybox   xsize= $xsize ysize= $ysize"
       #set scaleFacX  [expr $xcanmax(data)/( $marqueeBoxesWidth* $dataWidth)]
       #set scaleFacY  [expr $ycanmax(data)/($ytopmargin+$ybottommargin+ $marqueeBoxesHeight* $ybox)]
       set scaleFacX  [expr $xcanwindowmax/( $marqueeBoxesWidth* $dataWidth *$scalex)]
       set scaleFacY  [expr $ycanwindowmax/( $marqueeBoxesHeight* $ybox * $scaley)]
       #set xf_low [expr  ($fit_scalex * $newScaleX* ($xcol($dataOrigin) +($selStartFrame* $dataWidth)))/$xcanmax(data)]
       #set xf_high [expr  $newScaleX* ($xcol($dataOrigin) +( ($selStartFrame+ $marqueeBoxesWidth) * $dataWidth))]
       #set yf_low [expr  ($fit_scaley* $newScaleY * ($ytopmargin + ($startObject * $marqueeBoxesHeight)))/$ycanmax(data)]
       #set yf_high [expr  $newScaleY * ($ytopmargin + ($startObject * $marqueeBoxesHeight))]
       
       tlPutsDebug "scalex= $scalex  scaley= $scaley"
       set xf_low [expr $xSelStart/$xsize]
       set yf_low [expr $ySelStart/$ysize]
     }
     #ignore zoom if already zoomed in too far
     if { (2 * $scalex * $dataWidth) < $xcanwindowmax} {
        set scalex [expr $scaleFacX * $scalex]
        tlPutsDebug "setting scalex"
     }
     if {(2 * $scaley * $ybox )< $ycanwindowmax} {
       set scaley [expr $scaleFacY * $scaley]
       tlPutsDebug "setting scaley"
     }

     set userScalex [expr $scalex/ $fit_scalex ]
     set userScaley [expr $scaley/$fit_scaley ]
     set userScaleBoth $userScalex 
     tlPutsDebug "letGoMarquee: scalex= $scalex  scaley= $scaley scaleFacX= $scaleFacX  scaleFacY= $scaleFacY  userScalex= $userScalex   userScaley=$userScaley  fit_scaley= $fit_scaley **"
     #XXX find why causes odd problems
     #$w.zoomXlevel set $userScalex
     #$w.panl.zoomlevel set $userScaley
     redraw name func ops
     tlPutsDebug "xf_low= $xf_low  yf_low=$yf_low" 
     canvasScrollX moveto $xf_low 
     canvasScrollY moveto $yf_low 
    ### samp set fit_scalex [expr (0.0 + $xcanwindowmax - $xcol($dataOrigin) ) / ($dataWidth * ( $numDataFrames) ) ]
    ### sample  set fit_scaley [expr (0.0 + $ycanwindowmax - $ytopmargin - $ybottommargin) / ($ybox * ($dataValNum + 1) ) ]
    }
  }
  return
}

proc ::timeline::postCalc {} {
    #tasks after a calculation from loaded traj. frames
    variable numDataFrames 
    variable numTrajFrames  
    #do this elsewhere when we allow for skipping frames
    set numDataFrames $numTrajFrames
    postDataFill
}

proc ::timeline::postDataFill {} {
    #tasks after a data fill: file load or calculation from traj frames
    #set min/max limits
    #show results
    variable trajMin
    variable trajMax
    variable dataMin
    variable dataMax
    variable w
    variable numDataFrames 
    variable numTrajFrames  
    variable copyUserAfterCalc
    #in case errors in functions setting min/max
    if {$dataMin(all) == "null"} {
        set dataMin(all) 0
    }
    if {$dataMax(all) == "null"} {
       set dataMax(all) 1
    }

    set trajMin $dataMin(all)
    set trajMax $dataMax(all)
    tlPutsDebug "postCalc: about to set theshMaxScale; trajMin= $trajMin trajMax= $trajMax dataMin(all)= $dataMin(all)  dataMax(all)= $dataMax(all)"
    #$w.threshMinScale configure -from $dataMin(all) -to $dataMax(all) -tickinterval [expr  $dataMax(all) - $dataMin(all)-.001]
    $w.threshMinScale configure -from $dataMin(all) -to $dataMax(all) 
    $w.threshMaxScale configure -from $dataMin(all) -to $dataMax(all)
    tlPutsDebug "postCalc: done setting threshMinScale and theshMaxScale; trajMin= $trajMin trajMax= $trajMax dataMin(all)= $dataMin(all)  dataMax(all)= $dataMax(all)"
    if {$copyUserAfterCalc} {
      copyUser
      tlPutsDebug "postCalc: copied data to user field"
    }
       
    #end of postcalc operations
    showall 1
}


proc ::timeline::showall { do_redraw} {



  variable scalex 
  variable scaley 
  variable fit_scalex
  variable fit_scaley
  variable everRedrawn
  variable usableMolLoaded
  variable rectCreated 
  variable userScalex
  variable userScaley
  variable usesFreeSelection
  variable dataValNum
  variable ycanwindowmax
  variable ytopmargin
  variable ybottommargin  
  variable ybox
  variable trajMin
  variable trajMax
  variable dataMin
  variable dataMax
  
  calcFitScaleXY

    set scaley 1.0
  #only redraw once...
  if {$usableMolLoaded} {
    if {$do_redraw == 1} {
      set rectCreated 0
    }   
    
    set scalex $fit_scalex        
    set scaley $fit_scaley
    set userScalex 1.0
    set userScaley 1.0 

    redraw name func ops
  }

  return
}


proc ::timeline::every_res {} {

  variable everRedrawn
  variable usableMolLoaded
  variable rectCreated
  variable fit_scalex
  variable fit_scaley
  variable userScalex
  variable userScaley
  #this forces redraw, to cure any scaling floating point errors
  #that have crept in 
  set rectCreated 0

  variable scaley
  variable scalex

  if {$usableMolLoaded && $everRedrawn} {
    #redraw, set x and y  at once
    set scalex $fit_scalex 
    set userScalex 1.000 
    set scaley 1.0
    set userScaley [expr $scaley/$fit_scaley]
    redraw name func ops
  }
  
  return
}


proc ::timeline::residueCodeRedraw {} {

  variable w 
  variable resCodeShowOneLetter
  variable everRedrawn
  variable usableMolLoaded
  tlPutsDebug ": now in residueCodeRedraw, resiude_code_toggle is $resCodeShowOneLetter"
  
  if {$usableMolLoaded && $everRedrawn} {

    redraw name function op
  }
  return
}



proc ::timeline::initVars {} {

  variable dataFileVersion "1.4"
  variable usableMolLoaded 0
  variable everRedrawn 0
  variable windowShowing 0
  variable needsDataUpdate 0
  variable dataValNum -1
  variable dataValNumResSel -1
  variable eo 0
  variable x1 0 
  variable y1 0
  variable startCanvas ""
  variable startShiftPressed 0
  variable vmd_pick_shift_state 0
  variable resCodeShowOneLetter 0
  variable bond_rad 0.4
  variable bond_res 10
  variable repColoring "name"
  variable cursor_bond_rad 0.45 
  variable cursor_res 10
  variable cursorRepColor 1
  variable cursorRep
  variable so ""
  variable marqueeButton -1
  #better if nullMolString is 'null', alter pop-up drawing to accomodate XXX
  variable nullMolString ""
  variable currentMol $nullMolString
  variable prevMol $nullMolString

  variable  userScalex 1
  variable  userScaley 1
  variable  userScaleBoth 1
  variable  scalex 1
  variable  scaley 1
  variable prevScalex 1
  variable prevScaley 1
  
  variable ytopmargin 5
  variable ybottommargin 10
  variable xrightmargin 8

  #variable xcanwindowStarting 780 
  variable xcanwindowStarting 685 
  variable ycanwindowStarting 574 

  
  variable numDataFrames 0
  variable numTrajFrames 0
  variable xcanwindowmax  $xcanwindowStarting
  variable ycanwindowmax $ycanwindowStarting 
  variable xcanmax
  set xcanmax(data) 610
  set xcanmax(vert) 95
  # help get exposed canvas size from window manager
  variable canvasWinOffsetX 255
  variable canvasWinOffsetY 138

  set xcanmax(horz) $xcanmax(data)
  #make this sensible!
  variable ycanmax
  set ycanmax(data) 400
  set ycanmax(vert) $ycanmax(data) 
  set ycanmax(horz) 46 
  variable codes
  variable trajMin -100 
  variable trajMax 100 
  variable dataMin
  set dataMin(all) null
  variable dataMax
  set dataMax(all) null
  variable ONdist 3.2
  #
  # vars for interSelContacts start
  variable interDist 4
  # provide examples for GUI
  variable interSelList "{resid 1 to 10} {resid 11 to 20} {resid 21 to 30}"
  variable interSelListPairs "{resid 1 to 10} {resid 11 to 20} {resid 21 to 30} {resid 31 to 40}"
  variable interFromResSel "resid 20 to 40" 
  variable interToSel "segname LIG"
  # method 0 is selection list, method 1 is residues-in-selection to other-selction
  variable interSetMethod 0 
# vars for interSelContacts start
  #
  #distance cutoff in Angstroms
  variable hbondDistCutoff 3.0
  #angle cutoff in degrees
  variable  hbondAngleCutoff 20
  variable  hbondSel1 ""
  variable  hbondSel2 ""
  #RMSF params (frame counts)
  variable RMSFstepSize 1
  variable  RMSFwindowWidth 5
  #SASA radius in Angstroms
  variable SASArad 1.4
  #max number of items in threshold graph
  variable maxThresh 0
  #start variables for GlobalCcss
  variable ccssMapFile ""
  variable ccssMapres 5.0
  variable ccssSpacing 1.0
  variable ccssThreshold 0.0
  variable ccssUseThreshold 0
  variable ccssUseSpacing 0
  variable ccssSelMethod 0
  variable ccssSelList "{resid 1 to 10} {resid 11 to 20}"
  #end variables for GlobalCcss
  #start variables for native contact
  variable nativeDist 8
  variable nativeFromSelList "{resid 1 to 10} {resid 11 to 20}"
  variable nativeToSel "protein"
  variable nativeRefFrame 0 
  #end variables for native contact 
  #start variables for user copy
  variable whichUserField "user"
  variable copyUserAfterCalc 0
  #end variabcles for user copy
  #set boolean false 
  variable usesFreeSelection 0
  variable firstAnalysisFrame 0
  variable lastAnalysisFrame 0
  variable anyResFuncDesc "# contacts"
  variable anyResFuncName "::myCountContacts"
  variable thresholdBoundMin 0
  variable thresholdBoundMax 0
  variable filterMin 0
  variable filterMax 1
  variable filterFrameStart 0
  variable filterFrameEnd 100
  variable filterNumFramesRequired 1
  variable partSelText "all"
  variable calledBySelChange 0 
  array set codes {ALA A ARG R ASN N ASP D ASX B CYS C GLN Q GLU E
    GLX Z GLY G HIS H ILE I LEU L LYS K MET M PHE F PRO P SER S
    THR T TRP W TYR Y VAL V}
  variable progressDialogName .progressDialog 
  variable progressCancelPressed 0 
  variable progressLength 180 
  variable progressDrawFraction 0.7
  variable progressBoxExists 0
  #tests if rects for current mol have been created (should extend 
  #so memorize all rectIds in 3dim array, and track num mols-long 
  #vector of rectCreated. Would hide rects for non-disped molec,
  #and remember to delete the data when molec deleted.
  
  variable rectCreated 0

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
  #set xcol(0) 10.0
  variable horzScaleHeight 30
  variable threshGraphHeight 40 
  variable vertScaleWidth 100
  variable dataWidth 85
  variable dataMargin 0
  variable xPosScaleVal 32
  #so rectangle of data is drawn at width $dataWidth - $dataMargin (horizontal measures)
  #
  #residuie name data is in umbered entires numbered less than 3
  variable dataOrigin 3
  #puts "dataOrigin is $dataOrigin"
  #column that multi-col data first  appears in

  #old setting from when vertscale and data were on same canvas
  #set xcol($dataOrigin)  96 
  set xcol($dataOrigin)  1 
  #The 4th field (field 3) is the "first data field"
  #we use same data structure for labels and data, but now draw in separate canvases 
  
  # the names for  three fields of data 
  
  #just for self-doc
  # dataVal(picked,n) set if the elem is picked
  # dataVal(pickedId,n) contains the canvas Id of the elem's highlight rectangle
  

  variable dataName

  set dataName(picked) "picked" 
  set dataName(pickedId) "pickedId"
  #not included in count of # datanames
  
  set dataName(resid) "resid"
  set dataName(resname) "resname"
  set dataName(rescode) "res-code"
  set dataName(chain) "chain"
  set dataName(segname) "segname"
  ###set dataName(3) "check error.." 

  variable  keyAtomSelString "((all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) or (name BAS) )" 
  if [info exists ::timeLineSettingExtraKeyAtoms] {
    if {$::timeLineSettingExtraKeyAtoms !=""} {
      set keyAtomSelString "[string range $keyAtomSelString 0 end-1] or ( $::timeLineSettingExtraKeyAtoms ) )"  
    }
  }
 set copyBuf(dummy) 0 
 
  #make rainbow colorscale 
 variable colorscale
 for {set i 0} {$i<=255} {incr i} {
   #grayscale
    set colorscale(0,$i,r) $i
    set colorscale(0,$i,g) $i
    set colorscale(0,$i,b) $i
 
    set colorscale(1,$i,r) [expr 255-$i]
    set colorscale(1,$i,g) [expr 255-$i]
    set colorscale(1,$i,b) [expr 255-$i]


   #rainbow (blue to red) colorscale
    if {$i > 127} {
      set colorscale(2,$i,r) [expr (($i - 128) * 2 ) +1 ]
      set colorscale(2,$i,b) 0
      set colorscale(2,$i,g) [expr (255-$i) *2]
    } else {
      set colorscale(2,$i,r) 0
      set colorscale(2,$i,b) [expr (127-$i) *2]
      set colorscale(2,$i,g) [expr $i*2]
    }
  
   #rainbow (red to blue) colorscale
   if {$i > 127} {
      set colorscale(3,$i,b) [expr (($i - 128) * 2 ) +1 ]
      set colorscale(3,$i,r) 0
      set colorscale(3,$i,g) [expr (255-$i) *2]
    } else {
      set colorscale(3,$i,b) 0
      set colorscale(3,$i,r) [expr (127-$i) *2]
      set colorscale(3,$i,g) [expr $i*2]
    }


  
   #tlPutsDebug "i= $i  r= $colorscale(0,$i,r)  g= $colorscale(0,$i,g)  b= $colorscale(0,$i,b)"
   #tlPutsDebug "i= $i  r= $colorscale(1,$i,r)  g= $colorscale(1,$i,g)  b= $colorscale(1,$i,b)"
  }

set colorscale(choice) 0 
#0 is grayscale (B->W), 1 is grayscale (W->B),  2 is rainbow

variable highlightColor purple
     
#set ths var as an array, index (placeholder) is never used
}


proc ::timeline::Show {} {
  variable windowShowing
  variable needsDataUpdate
  set windowShowing 1

  
  if {$needsDataUpdate} {
    set needsDataUpdate 0
    #set immmediately, so other binding callbacks will see
    [namespace current]::timeLineMain
  }

}

proc ::timeline::Hide {} {
  variable windowShowing 
  set windowShowing 0

}

proc ::timeline::createCursorHighlight { theSel} {
  tlPutsDebug ": in create CursorHighlight"

  variable currentMol
  variable cursor_bond_rad
  variable cursor_res
  variable cursorRep
  variable cursorRepColor
  variable nullMolString
  tlPutsDebug "var block done, in create CursorHighlight"
 
  if {$currentMol == $nullMolString} {
     return
   }

  #draw first selection, as first residue 
  tlPutsDebug "in createCursorHighlight: theSel= >$theSel<"
  mol selection $theSel
  mol material Opaque
  mol addrep $currentMol
  set cursorRep($currentMol) [mol repname $currentMol [expr [molinfo $currentMol get numreps] -1]]
  set theCursorRepIndex [mol repindex $currentMol $cursorRep($currentMol)] 
  mol modstyle $theCursorRepIndex $currentMol Licorice $cursor_bond_rad $cursor_res $cursor_res
  mol modcolor $theCursorRepIndex $currentMol ColorID $cursorRepColor
 

}

proc ::timeline::createHighlight { theSel} {

  variable currentMol
  variable bond_rad
  variable bond_res
  variable rep
  variable repColoring
  variable cursorRep
  variable nullMolString
  #draw first selection, as first residue 
  
  if {$currentMol == $nullMolString} {
     return
   }
  mol selection $theSel
  mol material Opaque
  mol addrep $currentMol
  set rep($currentMol) [mol repname $currentMol [expr [molinfo $currentMol get numreps] -1]]
  set theRepIndex [mol repindex $currentMol $rep($currentMol)] 
  mol modstyle $theRepIndex  $currentMol   Licorice $bond_rad $bond_res $bond_res
  mol modcolor $theRepIndex  $currentMol ColorID $repColoring
  tlPutsDebug "rep($currentMol)= $rep($currentMol)  currentMol= $currentMol the_index= [mol repindex $currentMol $rep($currentMol)] "
}

proc ::timeline::printRedraw {} {
  variable clicked
  variable thresholdBoundMin
  variable thresholdBoundMax
  variable usesFreeSelection
  variable resCodeShowOneLetter
  variable x1 
  variable y1 
  variable so
  variable w
  variable wp 
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
  variable dataVal 
  variable dataValNum 
  variable dataOrigin 
  variable dataName 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable vertTextRight
  variable vertHighLeft
  variable vertHighRight
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin
  variable dataMax 
  variable trajMin
  variable trajMax
  variable xPosScaleVal
  variable everRedrawn
  variable usableMolLoaded
  variable rectCreated
  variable prevScalex
  variable prevScaley
  variable numDataFrames
  variable dataThreshVal
  variable currentMol 
  variable nullMolString
  variable prevCursorObject
  variable prevCursorFrame
  variable cursorRepColor
  variable cursorRep
  variable cursorShown
  tlPutsDebug "starting printRedraw"


  if { ($usableMolLoaded) && ($dataValNum >=0 )  && ($currentMol != $nullMolString)  && ($numDataFrames >= 1)} {
  #$wp.large delete all
  #local constants, eventually place in startup 
    catch {destroy $wp}
   
    if { [catch {toplevel $wp -visual truecolor} errMsg] } {
      puts "Info) Timeline printout buffer window can't find trucolor visual, will use default visual.\nInfo)   (Error reported was: $errMsg)" 
      if { [catch {toplevel $wp } errMsg ]} {
        puts "Info) Default visual failed, Timeline printout buffer window cannot be created. \nInfo)   (Error reported was: $errMsg)"    
        set printWindowError 1
        return
      }
    }
   
    wm title $wp "VMD Timeline (print preview)"
    wm protocol $wp WM_DELETE_WINDOW {set ::timeline::clicked -1}
    wm minsize  $wp 956 685 
    wm maxsize  $wp 956 685

    #canvas $wp.large -width 990 -height 700 -background #DDFFFF 
    canvas $wp.large -width 956 -height 685 -background #DDFFFF 
    pack $wp.large -in $wp -side left -padx 2 -expand yes -fill both 
   
    # print layout: constants, calculated values
    set printDataPosX 245
    set printDataPosY 5


    set selInfoWidthPrint 100 
    set vertTextLeftPrint [expr 0 + $selInfoWidthPrint + 15] 
    set vertTextRightPrint [expr $printDataPosX - 10]
    set threshGraphHeightPrint 40
    set selInfoYPos 300 
    set topData $printDataPosY
    set botData [expr $printDataPosY + $ycanwindowmax]
    set yposHorzScale [expr $botData + 21]
    set threshPlotTopPrint  [expr $yposHorzScale + 30 ]
    set horzScaleTitleYpos  [expr $yposHorzScale + 5 ]
    set leftData $printDataPosX
    set rightData [expr $printDataPosX + $xcanwindowmax]
    set vertTextWidthPrint  [expr $vertTextRightPrint - $vertTextLeftPrint]

    set colScalePosX [expr ($selInfoWidthPrint - 90) / 2 ]
    set colScalePosY [expr $botData - 28]


    #Limits in drawn canvas
    set leftCan  [expr $xcanmax(data) * [lindex [$w.can xview] 0]]
    set rightCan  [expr $xcanmax(data) * [lindex [$w.can xview] 1]]
    set topCan [expr $ycanmax(data) * [lindex [$w.can yview] 0]] 
    set botCan [expr $ycanmax(data) * [lindex [$w.can yview] 1]] 

     
    #use the text in the GUI window for printing info text
    set infoText "Highlight details:\n[$w.selInfo cget -text]\n\n Threshold:\n[$w.threshValLab cget -text]"
    if {$dataName(vals) != "struct"} {
    set infoText "$infoText\n$thresholdBoundMin to $thresholdBoundMax"
    }
    $wp.large create text [expr $selInfoWidthPrint/2] $selInfoYPos -text $infoText -justify center -width $selInfoWidthPrint -anchor n -tags printout]

    #title for horz axis
    $wp.large create text [expr (($leftData+$rightData) / 2)] $horzScaleTitleYpos -text "(frame number)" -font monofont -justify center -width [expr $xcanwindowmax] -anchor n -tags printout]
    if {$usesFreeSelection} {
      set vertScaleTitle "(selection name)"
    } else {
      set vertScaleTitle "(resid,\nresname,\nchain)" 
    }
    $wp.large create text [expr $vertTextLeftPrint - 3] 60 -text $vertScaleTitle -font monofont -justify center -width $selInfoWidthPrint -anchor e -tags printout]

    #threshold count
      

    # we print to a fixed size canvas, the dataRects will use the current scaling
    # be careful not to set any vars that are used only by screen display 
    # to capture cutoff, we will need to know where data should start and end

    ##set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValNum + 1) )]  

    ##set xsize [expr  $xcol($dataOrigin) +  ($scalex *  $dataWidth *  $numDataFrames)  ] 

    ##set ycanmax(data) $ysize
    ##set ycanmax(vert) $ycanmax(data)
    ##set xcanmax(data) $xsize
    ##set xcanmax(horz) $xcanmax(data)
    ##if {$ycanmax(data) < $ycanwindowmax} {
    ##  set ycanmax(data) $ycanwindowmax
   ## }


    ##if {$xcanmax(data) < $xcanwindowmax} {
    ##  set xcanmax(data) $xcanwindowmax
    ##}

    ##$w.can configure -scrollregion "0 0 $xcanmax(data) $ycanmax(data)"
    ##$w.vertScale configure -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)"
    ##$w.horzScale configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    ##$w.threshGraph configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"

    set fieldLast [expr $dataOrigin + $numDataFrames -1 ]
    

    if {$usableMolLoaded && $everRedrawn && ($currentMol != $nullMolString) &&  ($dataValNum >=0)} {
      thresholdData
      #### start of printThresholdGraph
      set threshPlotBottom [expr ($threshPlotTopPrint -4) + $threshGraphHeightPrint - 5] 
      set lastField [expr $dataOrigin + $numDataFrames - 1]
      set minThresh $dataThreshVal($dataOrigin)
      set minThreshField $dataOrigin
      set maxThresh $dataThreshVal($dataOrigin)
      set maxThreshField $dataOrigin
      for {set field [expr $dataOrigin+1]} {$field<=$lastField} {incr field} {
        if {$dataThreshVal($field) < $minThresh} {set minThresh $dataThreshVal($field); set minThreshField $field} 
        if {$dataThreshVal($field) > $maxThresh} {set maxThresh $dataThreshVal($field); set maxThreshField $field} 
      }
      if {$maxThresh == 0} {set depictedMaxThresh 10} else {set depictedMaxThresh $maxThresh}
      set plotFactor [expr  ($threshPlotBottom-$threshPlotTopPrint)/(0.0+$depictedMaxThresh) ]
       #count will be 0-based
       #later can do min based

      set endField [expr $dataOrigin + $numDataFrames - 1 ]

      for {set field $dataOrigin} {$field <= $endField} {incr field} {
        set frame [expr $field - $dataOrigin]
        set intermed [expr $plotFactor * $dataThreshVal($field)]
        set plotY [expr $threshPlotBottom - ($plotFactor * $dataThreshVal($field))]
        #puts "val= $intermed  dataThreshVal($field)= $dataThreshVal($field)  plotY=$plotY, field= $field threshPlotBottom=$threshPlotBottom "
        set xStart  [expr  ( ($frame + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        set xEnd  [expr  ( ($frame + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        set xStartShifted [expr $xStart - $leftCan + $printDataPosX]
        set xEndShifted [expr $xEnd - $leftCan + $printDataPosX]
        #tlPutsDebug "xEndShifted= $xEndShifted leftData= $leftData  xStartShifted= $xStartShifted rightData= $rightData"
        if {! (($xEndShifted  < $leftData) ||  ($xStartShifted > $rightData))} {
          if {$xStartShifted < $leftData} {
            set xStartShifted $leftData
          }
          if {$xEndShifted >  $rightData} {
            set xEndShifted $rightData
          }
          $wp.large create rectangle  $xStartShifted $threshPlotBottom $xEndShifted $plotY -fill "\#EE7070"  -tags printout 
        #puts "plotted   $xStart $threshPlotBottom $xEnd $plotY"
        }  
      }
      $wp.large create text $vertTextRightPrint  $threshPlotBottom -text "0" -font monofont -justify right -width 200 -anchor e -tags printout]
      $wp.large create text $vertTextRightPrint  $threshPlotTopPrint -text $maxThresh -font monofont -justify right -width 200 -anchor e -tags printout]
      $wp.large create text [expr $vertTextRightPrint - 20]  [expr ($threshPlotTopPrint + $threshPlotBottom) / 2] -text "(threshold\ncount)" -font monofont -justify center -width 150 -anchor e -tags printout]

        #####mark min of the thresh
        #set xStart  [expr  ( ([expr $minThreshField-$dataOrigin] + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
       # set xEnd  [expr  ( ([expr $minThreshField-$dataOrigin]  + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        #$wp.large create rectangle  $xStart [expr $threshPlotBottom+1] $xEnd [expr $threshPlotBottom +4] -fill "\#991010" -outline "" -tags [list xScalable threshPlotBar]

         #####mark max of the thresh
        #set xStart  [expr  ( ( $maxThreshField-$dataOrigin + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        #set xEnd  [expr  ( ( $maxThreshField-$dataOrigin + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        #$wp.large create rectangle  $xStart [expr $threshPlotTop-1] $xEnd [expr $threshPlotTop-4] -fill "\#109910" -outline "" -tags [list xScalable threshPlotBar]

       
      #### end of printThresholdGraph

        #for example, if we have 2 frames of data, frame 0 and frame 1,
      #then numDataFrames = 2.  Since dataOrigin =3, fieldLast is 4, since data
      # is in field 3 (frame 0), field 4 (frame 1). Formula is...
      
      #draw data on can
      #loop over all data fields

      if { $rectCreated} {
        #this until separate data and scale highlighting
        #puts "drawing rects, scalex is $scalex"
        #hack here -- for now skip B-field stuff, so minimal stuff drawn
        tlPutsDebug "printing: setting min/max, dataOrigin= $dataOrigin" 
        for {set field [expr $dataOrigin ]} {$field <= $fieldLast} {incr field} {
          set printCol 1;
            
          
          set xPosFieldLeft [expr  -$leftCan + $printDataPosX + ($scalex * $dataWidth * ($field - $dataOrigin)  )  ]
          set xPosFieldRight [expr  -$leftCan + $printDataPosX+  ($scalex * $dataWidth * ($field - $dataOrigin + 1 - $dataMargin)  )  ]

          if { ($xPosFieldRight  <  $leftData) || ($xPosFieldLeft > $rightData)} {
            set printCol 0;
          }
          if {$xPosFieldLeft < $leftData} {
            set xPosFieldLeft $leftData
          }
          if {$xPosFieldRight > $rightData} {
           set xPosFieldRight $rightData
          }
          
          if {$printCol} { 
            #now draw data rectangles
            #puts "drawing field $field at xPosField $xPosField" 
            #yipes, does this redraw all rects (even non visible) every timeXXX
            set y 0.0
            
            set intensity 0
            
            for {set i 0} {$i<=$dataValNum} {incr i} { 
              set val $dataVal($field,$i)
              set printBox  1;
              if {$val != "null"} {
                #calculate color and create rectangle
                
                set yposTop [expr -$topCan + $printDataPosY+ ($scaley * $y)]
                set yposBot [expr $yposTop + ($scaley * $ybox)]
                if {($yposBot < $topData) || ($yposTop > $botData)} {
                set printBox 0
              }

              if {$yposTop < $topData} {
                   set yposTop $topData
              }
              if {$yposBot > $botData} {
                   set yposBot $botData
              }

                 
              #should Prescan  to find range of values!   
              #this should be some per-request-method range / also allow this to be adjusted
              
              #set intensity except if field 4 (indexed struct)
              #puts "field = $field, dataName($field) = $dataName($field),i= $i" 
              if {$dataName(vals) != "struct"} {
                ##if { ( ($field != 4)  ) } open brace here 
                #set range [expr $dataMax($field) - $dataMin($field)]
                set range [expr $trajMax - $trajMin ]
                if { ($range > 0)  && ([string is double $val] )} {
                  set intensity  [expr int (255. * ( (0.0 + $val - $trajMin ) / $range)) ]
                  #tlPutsDebug ": $val $dataMin($field) $range $field $intensity"
                }
                
                
                
                set hexcols [chooseColor $intensity]
              } else {
                #horrifyingly, sends string for data, tcl is typeless
                set hexcols [chooseColor $val ]
              }
              foreach {hexred hexgreen hexblue} $hexcols {} 

              
              #draw data rectangle
              ## add logic to not draw if out of range 
              if {$printBox} {
                $wp.large create rectangle $xPosFieldLeft $yposTop  $xPosFieldRight $yposBot  -fill "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags printout 
              }
            }
            
            set y [expr $y + $ybox]
          }
        }
      }

     

    
       tlPutsDebug "Now start testing for print,  cursorRep($currentMol)= $cursorRep($currentMol)  cursorShown($currentMol)= $cursorShown($currentMol), prevCursorObject($currentMol)= $prevCursorObject($currentMol)  prevCursorFrame($currentMol) =$prevCursorFrame($currentMol)"  
      if {($cursorRep($currentMol)!="null") && $cursorShown($currentMol)} {
        set printCursor 1;
        set y {$ybox * $prevCursorObject($currentMol)}
        set theFrame $prevCursorFrame($currentMol)
        set yposTop [expr -$topCan + $printDataPosY+ ($scaley * $y)]
        set yposBot [expr $yposTop + ($scaley * $ybox)]
        if {($yposBot < $topData) || ($yposTop > $botData)} {
          set printCursor 0;
        }

        if {$yposTop < $topData} {
             set yposTop $topData
        }
        if {$yposBot > $botData} {
             set yposBot $botData
        }
          set xPosFieldLeft [expr  -$leftCan + $printDataPosX + ($scalex * $dataWidth * ($theFrame)  )  ]
          set xPosFieldRight [expr  -$leftCan + $printDataPosX+  ($scalex * $dataWidth * ($theFrame + 1 - $dataMargin)  )  ]

          if { ($xPosFieldRight  <  $leftData) || ($xPosFieldLeft > $rightData)} {
            set printCursor 0;
          }
          if {$xPosFieldLeft < $leftData} {
            set xPosFieldLeft $leftData
          }
          if {$xPosFieldRight > $rightData} {
           set xPosFieldRight $rightData
          }
          # tlPutsDebug "printCursor= $printCursor xPosFieldLeft= $xPosFieldLeft  xPosFieldRight= $xPosFieldRight  yposTop= $yposTop  ypoBot= $yposBot"

          if {$printCursor} { 
            tlPutsDebug "now printing cursor"  
          #$wp.large create  $vertTextLeftPrint $ypos $vertTextRightPrint [expr $ypos + ($scaley * $ybox)]    -outline "\#FF0000" -width 2 -fill "" -tags printout    
           #cursor highlight
           $wp.large create rectangle $xPosFieldLeft $yposTop  $xPosFieldRight $yposBot  -outline "\#FF0000" -width 2 -fill "" -tags printout 
           #vert scale highlght
           $wp.large create rectangle $vertTextLeftPrint $yposTop  [expr $printDataPosX - 5] $yposBot  -outline "\#FF0000" -width 2 -fill "" -tags printout 
           #horz scale highlight
           $wp.large create rectangle $xPosFieldLeft [expr $botData + 5] $xPosFieldRight $yposHorzScale -outline "\#FF0000" -width 2 -fill "" -tags printout 
         }

      }  
    }
  }  else {
     puts "Timeline: need to display a graph before printing"
  }
##end dataRect and cursor highlight printing
  
  ####drawHorzScale
    #ensure minimal horizontal spacing
    # hardcoded spacing
    set horzSpacing 27 
    set horzPad 5
    set horzSpacingPad [expr $horzSpacing + $horzPad]
    set horzDataTextSkip [expr $dataWidth]
    set scaledHorzDataTextSkip [expr $scalex * $dataWidth]
    set scaledHorzDataOffset [expr $scalex * $dataWidth / 2.0]
    set xStart [expr ($xcol($dataOrigin))]
    set xDataEnd  [expr int ($xStart +  $scalex * ($dataWidth * $numDataFrames ) ) ] 
    set x 0 



    #numbers are scaled for 1.0 until xpos
    #this is tied to data fields, which is produced from frames upon
    #first drawing. Should really agreee with writeDataFile, which currently uses frames, not fields
    set xposPrev -1000 
    set xposRightPrev -1000 
    #traj data starts at dataOrigin
    for {set frameNum 0} {$frameNum < $numDataFrames} {incr frameNum} {
      set field [expr $frameNum + $dataOrigin]
      set textWidth [font measure $monoFont -displayof $wp $frameNum] 
      set textWidthPad [expr $textWidth +$horzPad]
    ####for {set field [expr $dataOrigin]} {$field <= $fieldLast} {incr field} {}
    ####  set frameNum [expr $field - $dataOrigin -1]
      
      set xpos [expr int ($xStart + ($scalex * $x) + $scaledHorzDataOffset)]
      set xposRight [expr $xpos +int($textWidth/2)]
      if { ( ($xposRight - $xposRightPrev  ) >= $textWidthPad) && ( ( $field == $fieldLast) || ( ( $xDataEnd - $xpos) > ( 2 * $textWidth) ) ) } {
        # draw the frame number if there is room
        #for speed, we use horzScaleText instead of $dataName($field)
        set xposShifted [expr $xpos-$leftCan + $printDataPosX]
        #tlPutsDebug "xposShifted= $xposShifted   xpos= $xpos  leftCan= $leftCan leftData= $leftData  rightData= $rightData"
        if {$xposShifted >=$leftData && $xposShifted <=$rightData} {
          $wp.large create text $xposShifted $yposHorzScale -text "$frameNum" -width 30 -font $monoFont -justify center -anchor s -tags printout 
          set xposPrev $xpos
          set xposRightPrev $xposRight
        }
      }        
      set x [expr $x + $horzDataTextSkip]
    } 
    ##end drawHorzScale    


  ####drawVertScale starts
    
    #Add the text...
    set field 0           

    #note that the column will be 0, but the data will be from picked
    
    
    set yDataEnd [expr $ytopmargin + ($scaley * $ybox * ($dataValNum +1))]
    set y 0.0

    set yposPrev  -10000.0

    #Add the text to vertScale...
    set field 0            


    #we want text to appear in center of the dataRect we are labeling
    set vertOffset [expr $scaley * $ybox / 2.0]

    #don't do $dataValNum, its done at end, to ensure always print last 
    for {set i 0} {$i <= $dataValNum} {incr i} {
      #set ypos [expr $ytopmargin + ($scaley * $y) + $vertOffset]
      set ypos [expr  ($scaley * $y) + $vertOffset]
      if { ( ($ypos - $yposPrev) >= $vertTextSkip) && ( ( $i == $dataValNum) || ( ($yDataEnd - $ypos) > $vertTextSkip) ) } {
        set yposShifted [expr $ypos-$topCan+$printDataPosY]
        if {$yposShifted >= $topData && $yposShifted <=$botData} {
          if {$usesFreeSelection} {
            $wp.large create text $vertTextRightPrint $yposShifted -text $dataVal(freeSelLabel,$i)  -width $vertTextWidthPrint -font $monoFont -justify right -anchor e -tags printout 
           } else {
            if {$resCodeShowOneLetter == 0} {
              set res_string $dataVal(resname,$i)
            } else {
              set res_string $dataVal(rescode,$i)
            }
           #for speed, we use vertScaleText instead of $dataName($field)
           #how to deal with chain vs. segname?  For now, don't show segname.  Should allow toggle?
          $wp.large create text $vertTextRightPrint $yposShifted -text "$dataVal(resid,$i) $res_string $dataVal(chain,$i)" -width $vertTextWidthPrint -font $monoFont -justify right -anchor e -tags printout 
          }
       set yposPrev  $ypos
        }
       ##set yposPrev  $ypos
      }        
      set y [expr $y + $vertTextSkip]
      
    } 
    ####drawVertScale ends

    printColScale $colScalePosX $colScalePosY


        
    
    tlPutsDebug "done with print redraw, everRedrawn= $everRedrawn"

    # Now do the printing.
    set filename "VMD_Timeline_Window.eps"
    set filename [tk_getSaveFile -initialfile $filename -title "VMD Timeline Print" -parent $wp -filetypes [list {{Encapsulated Postscript Files} {.eps}} {{All files} {*} }] ]
    if {$filename != ""} {
      $wp.large postscript -pagewidth 7.0i -file $filename
    }
    # printing done
    #### now hide or destroy print window 
    catch {destroy $wp}
  } else {
    tk_messageBox -message "Timeline has nothing to print. Please calculate or load a data plot." -parent $w -icon warning -type ok -title "VMD Timeline warning"
  }
  return
}




proc ::timeline::draw_interface {} {
  variable w 

  variable eo 
  variable x1  
  variable y1 
  variable startCanvas
  variable startShiftPressed 
  variable vmd_pick_shift_state 
  variable resCodeShowOneLetter 
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
  variable dataValNum 
  variable dataVal 
  variable dataName 
  variable dataOrigin 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable bond_rad 
  variable rep 
  variable cursorRep
  variable repColoring
  variable cursor_res
  variable cursor_bond_rad
  variable xcol 
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin 
  variable dataMax 
  variable xPosScaleVal
  variable currentMol
  variable fit_scalex 1.0
  variable fit_scaley 1.0
  variable usableMolLoaded 
  variable numDataFrames 
  variable userScaleBoth
  variable partSelText 
  variable highlightColor
  variable colorscale
  frame $w.menubar -height 30 -relief raised -bd 2
  pack $w.menubar -in $w -side top -anchor nw -padx 1 -fill x

  #frame $w.fr -width 700 -height 810 -bg #FFFFFF -bd 2 ;#main frame

  #pack $w.fr

  label $w.txtlab -text "Zoom "
   tlPutsDebug " before selinfo label make"
  frame $w.panl -width 170 -height [expr $ycanwindowmax + 80] -bg #C0C0D0 -relief raised -bd 1 
  #frame $w.cfr -width 350 -height [expr $ycanwindowmax + 85] -borderwidth 1  -bg #606060 -relief raised -bd 3
  frame $w.cfr -width 350 -height [expr $ycanwindowmax + 85] -borderwidth 1  -bg #606060 -relief raised -bd 3
  tlPutsDebug " after frames"
  pack $w.panl -in $w -side left -padx 2  -fill y
  #pack $w.cfr -in $w.fr -side left -padx 2 -expand yes -fill both 
  pack $w.cfr -in $w -side left -padx 2 -expand yes -fill both 
   tlPutsDebug ": after selinfo label make"

  canvas $w.colscale -width 100 -height 60  -bg \#a0a0a0 
    pack $w.colscale -in $w.panl -side bottom -anchor sw

  scale $w.panl.zoomlevel -from 0.01 -to 8.01 -length 150 -sliderlength 30  -resolution 0.01 -tickinterval 0.5 -repeatinterval 30 -showvalue true -variable [namespace current]::userScaley -command [namespace code userScaleyChanged] 
   label $w.selInfo -text "Property\n\n ResID  Resname\n chain:  seg:\nValue\nFrame" -width 16
  scale $w.zoomBothlevel -orient horizontal -from 0.001 -to 4.000 -length 120 -sliderlength 30  -resolution 0.001 -tickinterval 3.998 -repeatinterval 30 -showvalue true -variable [namespace current]::userScaleBoth -command [namespace code userScaleBothChanged] 
  scale $w.zoomXlevel -orient horizontal -from 0.001 -to 4.000 -length 120 -sliderlength 30  -resolution 0.001 -tickinterval 3.998 -repeatinterval 30 -showvalue true -variable [namespace current]::userScalex -command [namespace code userScalexChanged] 
  #scale $w.threshMinScale  -orient horizontal -digits 4 -from -180 -to 180 -length 100 -sliderlength 20  -resolution 0.01 -tickinterval 359.99  -repeatinterval 30 -showvalue true -variable [namespace current]::thresholdBoundMin -command [namespace code threshChanged] 
  scale $w.threshMinScale  -orient horizontal -digits 4 -from -180 -to 180 -length 100 -sliderlength 20  -resolution 0.01  -repeatinterval 30 -showvalue true -variable [namespace current]::thresholdBoundMin -command [namespace code threshChanged] 
  scale $w.threshMaxScale  -orient horizontal -from -180 -to 180.000 -length 100 -sliderlength 20  -resolution 0.01  -repeatinterval 30 -showvalue true -variable [namespace current]::thresholdBoundMax  -command [namespace code threshChanged] 
 tlPutsDebug " after scales"
 label $w.threshValLab -text "-/-  - " -width 40
  #pack $w.panl $w.cfr -in $w.fr -side left -padx 2
  pack $w.panl.zoomlevel -in $w.panl -side right -ipadx 8 -padx 8 -anchor e
 tlPutsDebug " after pack panlzoom"
  button $w.showall  -text "fit all" -command [namespace code {showall 0}]
  button $w.every_res  -text "every residue" -command [namespace code every_res]

  tlPutsDebug " after buttons" 
  #trace for molecule choosing popup menu 
  trace variable ::vmd_initialize_structure w  [namespace code molChooseMenu]
  
  menubutton $w.mol -relief raised -bd 2 -textvariable [namespace current]::currentMol -direction flush -menu $w.mol.menu
  menu $w.mol.menu -tearoff no


  molChooseMenu name function op
  

  label $w.molLab -text "Molecule:"

  entry $w.partSel -textvariable ::timeline::partSelText
  bind $w.partSel <Return> {set ::timeline::calledBySelChange 1; ::timeline::timeLineMain}

  scrollbar $w.ys -command [namespace code {canvasScrollY}]
  
  scrollbar $w.xs -orient horizontal -command [namespace code {canvasScrollX}]

 tlPutsDebug ": now to fill the top menu"

  #fill the  top menu
  menubutton $w.menubar.file -text "File" -underline 0 -menu $w.menubar.file.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.file config -width 5
  menubutton $w.menubar.calculate -text "Calculate" -underline 0 -menu $w.menubar.calculate.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.calculate config -width 10
  menubutton $w.menubar.threshold -text "Threshold" -underline 0 -menu $w.menubar.threshold.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.threshold config -width 12
  menubutton $w.menubar.graphics -text "Appearance" -underline 0 -menu $w.menubar.graphics.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.graphics config -width 11
  
  menubutton $w.menubar.analysis -text "Analysis" -underline 0 -menu $w.menubar.analysis.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.analysis config -width 10

  menubutton $w.menubar.data -text "Data" -underline 0 -menu $w.menubar.data.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.data config -width 5 


  pack $w.menubar.file  $w.menubar.calculate $w.menubar.threshold  $w.menubar.analysis $w.menubar.graphics  $w.menubar.data -side left

  menubutton $w.menubar.help -text Help -underline 0 -menu $w.menubar.help.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.help config -width 5 
  menu $w.menubar.help.menu -tearoff no

  $w.menubar.help.menu add command -label "Timeline Help" -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/timeline"
  $w.menubar.help.menu add command -label "Structure codes..." -command  [namespace code {tk_messageBox -parent $w  -type ok -message "Secondary Structure Codes\n\nT        Turn\nE        Extended conformation\nB        Isolated bridge\nH        Alpha helix\nG        3-10 helix\nI         Pi-helix\nC        Coil (none of the above)\n" } ]

  pack $w.menubar.help -side right 
  
  #File menu
  menu $w.menubar.file.menu -tearoff no
  $w.menubar.file.menu add command -label "Load data file..." -command [namespace code {readDataFile ""}  ] 
  $w.menubar.file.menu add command -label "Write data file..." -command [namespace code {writeDataFile ""}  ] 
  $w.menubar.file.menu add command -label "Print to file..." -command [namespace code {printRedraw} ] 
  
  #Calculate menu
  
  menu $w.menubar.calculate.menu  -tearoff no

 tlPutsDebug ": about to register calc menus"

  $w.menubar.calculate.menu add command -label "Clear data"  -command  [namespace code clearData] 
  $w.menubar.calculate.menu add command -label "Calc. Sec. Struct"  -command [namespace code {calcDataStruct; postCalc;}] 
  #$w.menubar.calculate.menu add command -label "Calc. X position"  -command [namespace code {calcDataX; postCalc;}] 
  #$w.menubar.calculate.menu add command -label "Calc. Y position"  -command [namespace code {calcDataY; postCalc;}] 
  #$w.menubar.calculate.menu add command -label "Calc. Z position"  -command [namespace code {calcDataZ; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. Phi"  -command [namespace code {calcDataPhi; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. Delta Phi"  -command [namespace code {calcDataDeltaPhi; postCalc}] 
  $w.menubar.calculate.menu add command -label "Calc. Psi"  -command [namespace code {calcDataPsi; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. Delta Psi"  -command [namespace code {calcDataDeltaPsi; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Show User Field" -command  [namespace code {calcDataUser; postCalc;}] 
  #$w.menubar.calculate.menu add command -label "Test Free Select" -command  [namespace code {calcTestFreeSel 10; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. H-bonds..." -command  [namespace code {setParamsHbonds 11; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. Salt Bridges..." -command  [namespace code {setParamsSaltBridge 16; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. displacement" -command  [namespace code {calcDisplacement; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. RMSD" -command  [namespace code {calcRMSD; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. RMSF..." -command  [namespace code {setParamsRMSF 18; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. SASA..." -command  [namespace code {setParamsSASA 19; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. disp. velocity " -command  [namespace code {calcDispVelocity; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. inter-sel. contacts..." -command  [namespace code {setParamsInterSelContacts 21; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. native contacts..." -command  [namespace code {setParamsNativeContacts 23; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. cross-corr. ..." -command  [namespace code {setParamsGlobalCcss 22; postCalc;}] 
  $w.menubar.calculate.menu add command -label "Calc. User-defined (Per-Res.)" -command  [namespace code {calcDataAnyResFunc; postCalc;}] 
 tlPutsDebug ": done with calc menus"
  #Threshold menu
  menu $w.menubar.threshold.menu -tearoff no
  $w.menubar.threshold.menu add command -label "Set bounds..." -command  [namespace code setThresholdBounds]
  $w.menubar.threshold.menu add command -label "Make threshold graph" -command  [namespace code thresholdMakeGraph]
  $w.menubar.threshold.menu add command -label "Reset graph" -command  [namespace code thresholdClearGraph]
  tlPutsDebug ": Timeline: starting graphics menus"
  
  #Graphics menu
  menu $w.menubar.graphics.menu -tearoff no
  $w.menubar.graphics.menu add command -label "Set scaling..." -command  [namespace code setScaling]
  $w.menubar.graphics.menu add checkbutton -label "Show 1-letter codes" -variable ::timeline::resCodeShowOneLetter -onvalue 1 -offvalue 0 -command  [namespace code residueCodeRedraw]
  $w.menubar.graphics.menu add cascade -label "Highlight color/style" -menu $w.menubar.graphics.menu.highlightMenu 
  #Second level menu for highlightColor 
tlPutsDebug ": Timeline: starting Highlight menu"
  #set dummyHighlight so drawn selected first time, we use -command for actual var change
  menu $w.menubar.graphics.menu.highlightMenu -tearoff no
  $w.menubar.graphics.menu.highlightMenu add radiobutton -label "Yellow" -command {set ::timeline::highlightColor yellow} -variable TLdummyHighlight -value 0 
  $w.menubar.graphics.menu.highlightMenu add radiobutton -label "Purple" -command {set ::timeline::highlightColor purple} -variable TLdummyHighlight -value 1 
  set ::TLdummyHighlight 1 
  menu $w.menubar.graphics.menu.colorscale -tearoff no
  tlPutsDebug "about to add colorscale choices" 
  $w.menubar.graphics.menu add cascade -label "Color scale" -menu $w.menubar.graphics.menu.colorscale
  $w.menubar.graphics.menu.colorscale add radiobutton -label "Grayscale (B -> W)" -command {set ::timeline::colorscale(choice) 0; ::timeline::showall 1} -variable TLdummyColorscale -value 0 
  $w.menubar.graphics.menu.colorscale add radiobutton -label "Grayscale (W -> B)" -command {set ::timeline::colorscale(choice) 1; ::timeline::showall 1} -variable TLdummyColorscale -value 1 
  $w.menubar.graphics.menu.colorscale add radiobutton -label "Rainbow (BGR)" -command {set ::timeline::colorscale(choice) 2; ::timeline::showall 1} -variable TLdummyColorscale -value 2 
  $w.menubar.graphics.menu.colorscale add radiobutton -label "Rainbow (RGB)" -command {set ::timeline::colorscale(choice) 3; ::timeline::showall 1} -variable TLdummyColorscale -value 3 
  set ::TLdummyColorscale 0
  tlPutsDebug "added colorscale choices" 

tlPutsDebug ": Timeline: starting Analysis menu"
  #Functions menu
  menu $w.menubar.analysis.menu -tearoff no
  $w.menubar.analysis.menu add command -label "Define every-residue function..." -command  [namespace code setAnyResFunc]
  $w.menubar.analysis.menu add command -label "Set analysis frame range..." -command  [namespace code setAnalysisFrames]
  $w.menubar.analysis.menu add command -label "Filter data..." -command  [namespace code filterData] 
  $w.menubar.analysis.menu add command -label "Copy data to User field..." -command  [namespace code copyToUser]
 
  #Data menu
  menu $w.menubar.data.menu -tearoff no
  $w.menubar.data.menu add command -label "Set collection directory..." -command  [namespace code loadDataCollection]
   
tlPutsDebug ": Timeline: done with startup of Analysis menu"

  
#the w.can object made here
#XXX should decide how to deal with variable y-size (freeSels) and  even variable x-size (if ever abstract 2D plot so works with non-trajFrame values)
set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValNum + 1))]    
  set xsize [expr  $xcol($dataOrigin) +  ($scalex *  $dataWidth *  $numDataFrames ) ]




tlPutsDebug ": before some labels and similar"
  place $w.txtlab -in $w.panl  -bordermode outside -rely 0.1 -relx 0.5 -anchor n
  #place $w.showall -in $w.panl.zoomlevel  -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  place $w.showall -in $w.panl.zoomlevel  -bordermode outside -rely 1.0 -y 10.0 -relx 0.5 -anchor n
  place $w.every_res  -in $w.panl.zoomlevel -bordermode outside -rely 1.0 -y 45.0  -relx 0.5 -relwidth 1.0 -width 12  -anchor n
  #place $w.every_res  -in $w.showall -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  tlPutsDebug ": halfway"
  place $w.threshMinScale -in $w.every_res -bordermode outside -rely 1.0 -y 15 -relx 0.5 -width 90 -anchor n
  place $w.threshMaxScale -in $w.threshMinScale -bordermode outside -rely 1.0 -y 7 -relx 0.5 -width 90 -anchor n
  place $w.threshValLab -in $w.threshMaxScale -bordermode outside -rely 1.0 -y 7 -relx 0.5 -width 90 -anchor n

  place $w.partSel -in $w.panl.zoomlevel -border outside -rely 0.0 -y -5 -relx 0.5 -anchor s -width 87
 
  place $w.mol -in $w.partSel -bordermode outside -rely 0.0 -y -5  -relx 0.85 -anchor s
  place $w.molLab -in $w.mol -bordermode outside -rely 0.5 -relx 0 -anchor e
  tlPutsDebug ":placed partSel"
  place  $w.zoomBothlevel -in $w.partSel -bordermode  outside -rely 0.0 -y -37  -relx 0.5 -x 0  -width 87 -anchor s 

  tlPutsDebug ":just before zoomXlevel"
  place $w.zoomXlevel -in $w.zoomBothlevel -bordermode inside -rely 0.0 -y -15 -relx 0.5  -width 87 -anchor s 
  place $w.selInfo -in $w.cfr -bordermode inside -rely 1.0 -y -2  -relx 0.0 -x 3 -anchor sw
  #done with interface elements     
  tlPutsDebug ":done with interface"

  #ask window manager for size of window

  #turn traces  on (initialize_struct trace comes later)
  #trace variable userScalex w  [namespace code redraw]
  #trace variable userScaley w  [namespace code redraw]
  trace variable ::vmd_pick_atom w [namespace code listPick]
  trace variable currentMol w [namespace code molChoose]

}
  proc  ::timeline::timeBarJumpPress {x y shiftState whichCanvas} {
    variable xcol
    variable dataOrigin
    variable dataWidth
    variable scalex
    variable currentMol
    variable numDataFrames
    variable xcanmax
    variable ycanmax
    variable w
    #maybe store top, and restore it afterwards
    variable nullMolString
    variable ytopmargin
    variable scaley
    variable ybox
    variable dataVal
    variable dataValNum
    variable cursorShown
    variable prevCursorObject
    variable prevCursorFrame
    variable sharedCursorObject 
    variable sharedCursorFrame 
    tlPutsDebug "timeBarJumpPress starting"
    if {$currentMol == $nullMolString} {
       return
     }
    set x [expr $x + $xcanmax(data) * [lindex [$w.can xview] 0]]
    set y [expr $y + $ycanmax(data) * [lindex [$w.can yview] 0]] 

    set cursorFrame [expr  int (($x - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]

    if {$cursorFrame >= $numDataFrames}  {
      set cursorFrame [expr $numDataFrames -1]
    }
    
    if {$cursorFrame < 0 } { 
      set cursorFrame 0
    } 

    #allow scrubbing single sels
    set cursorObject [expr int (0.0 + ((0.0 + $y - $ytopmargin) / ($scaley * $ybox)) )]
    if {$cursorObject < 0} {
      set cursorObject 0
    }
    if {$cursorObject > $dataValNum} {
      set cursorObject $dataValNum 
    }
    if {$cursorObject < 0} {return}


  #  #these are only set when pressed, so we can tell if button-2 press/relase on same object+frame XXX
  #  # (a note for when we merge the silly 3 procs for timeBarJump into 1 proc) XXX
  #  # perhaps later replace this with 'object has changed' flag to dirty in button-2-moved, so press/away/back doesn't toggle
    #set prevCursorObject($currentMol)  $cursorObject ; set prevCursorFrame($currentMol) $cursorFrame
  #
    #only record the onbject, frame clicked on  for later reveal if cursorShown == 0
    if  {! $cursorShown($currentMol)} {return}
      
     
    #XX yipes, code duplication from letGoMarquee
    #Should reaclly de-sel everything, but need to keep track of separate sels, perhaps in the text vs. null later 
    if { [molinfo $currentMol get frame] != $cursorFrame } { 
      #test, and save/restore
      mol top $currentMol
      animate goto $cursorFrame

      #puts "jumped to $cursorFrame"
      drawTimeBar $cursorFrame
      #update both GL and tk timebar
      display update
      #display update ui
    }
    drawCursorObjectBar $cursorObject $cursorFrame
    set sharedCursorObject $cursorObject
    set sharedCursorFrame $cursorFrame
  }

  proc  ::timeline::timeBarJumpRelease {x y shiftState whichCanvas} {
    #XX Yikes, entire proc (nearly) is code duped from timeBarJumpPress 
    #merge these!
    variable xcol
    variable dataOrigin
    variable dataWidth
    variable scalex
    variable currentMol
    variable numDataFrames
    variable xcanmax
    variable ycanmax
    variable w
    #maybe store top, and restore it afterwards
    variable nullMolString
    variable ytopmargin
    variable scaley
    variable ybox
    variable dataVal
    variable dataValNum
    variable prevCursorObject
    variable prevCursorFrame
    variable cursorShown
    if {($currentMol == $nullMolString) || ($dataValNum < 1)} { return }

    set x [expr $x + $xcanmax(data) * [lindex [$w.can xview] 0]]
    set y [expr $y + $ycanmax(data) * [lindex [$w.can yview] 0]] 

    set cursorFrame [expr  int (($x - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]

    if {$cursorFrame >= $numDataFrames}  {
      set cursorFrame [expr $numDataFrames -1]
    }
    
    if {$cursorFrame < 0 } { 
      set cursorFrame 0
    } 

    #allow scrubbing single sels
    set cursorObject [expr int (0.0 + ((0.0 + $y - $ytopmargin) / ($scaley * $ybox)) )]
    if {$cursorObject < 0} {
      set cursorObject 0
    }
    if {$cursorObject > $dataValNum} {
      set cursorObject $dataValNum 
    }
    
    set exitFlag 0
    if { ($prevCursorObject($currentMol) == $cursorObject) &&  ($prevCursorFrame($currentMol) == $cursorFrame)} {
        tlPutsDebug "cursorShown($currentMol)=$cursorShown($currentMol)    prevCursorObject($currentMol)=$prevCursorObject($currentMol) cursorObject= $cursorObject   prevCursorFrame($currentMol)=prevCursorFrame($currentMol) cursorFrame= $cursorFrame"
        #toggle state
        if {$cursorShown($currentMol)} {
          #puts "cursor is shown, so now  hiding"
          hideCursorHighlight  
          $w.horzScale delete cursorObjectBarRect 
          $w.vertScale delete cursorObjectBarRect 
          $w.can delete cursorObjectBarRect 
          $w.can delete trajHighlight
          $w.can delete timeBarRect
          set cursorShown($currentMol) 0
          #for good measure, delete all the sequence selections too (seems like right place for this)
          clearAllPicked
          #for backup, we also delete vertScale pickedHighlight objects, but these should have been deleted in clearAllPicked
         $w.vertScale delete pickedHighlight
          set exitFlag 1 

        } else {
          #puts "cursor is hidden, so now revealing"
          set cSelText [findCursorSelText $cursorObject ]
          revealCursorHighlight $cSelText
          set cursorShown($currentMol) 1
          #and code below will redraw all rects in proper place
        }
    }
    set prevCursorObject($currentMol)  $cursorObject ; set prevCursorFrame($currentMol) $cursorFrame

    if {$exitFlag} {return}

    #yipes, code duplication from letGoMarquee
    #Should really de-sel everything, but need to keep track of separate sels, perhaps in the text vs. null later XXX
    if { [molinfo $currentMol get frame] != $cursorFrame } { 
      #test, and save/restore
      mol top $currentMol
      animate goto $cursorFrame

      #puts "jumped to $cursorFrame"
      drawTimeBar $cursorFrame
      #update both GL and tk timebar
      display update
      display update ui
      #puts "time for disp. update = [time {display update}]"
      #puts "time for disp. update ui= [time {display update ui}]"
    }
    drawCursorObjectBar $cursorObject $cursorFrame
  }



  proc  ::timeline::timeBarJump {x y shiftState whichCanvas} {
    variable xcol
    variable dataOrigin
    variable dataWidth
    variable scalex
    variable currentMol
    variable numDataFrames
    variable xcanmax
    variable ycanmax
    variable w
    #maybe store top, and restore it afterwards
    variable nullMolString
    variable ytopmargin
    variable scaley
    variable ybox
    variable dataVal
    variable dataValNum
    variable cursorShown
    if {($currentMol == $nullMolString) || ($dataValNum < 1)} { return }
   
    #tlPutsDebug "timeBarJump x= $x  y= $y  shiftState= $shiftState whichCanvas= $whichCanvas"  
    #merge these!
    set x [expr $x + $xcanmax(data) * [lindex [$w.can xview] 0]]
    set y [expr $y + $ycanmax(data) * [lindex [$w.can yview] 0]] 

    set cursorFrame [expr  int (($x - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]

    if {$cursorFrame >= $numDataFrames}  {
      set cursorFrame [expr $numDataFrames -1]
    }
    
    if {$cursorFrame < 0 } { 
      set cursorFrame 0
    } 

    #allow scrubbing single sels
    set cursorObject [expr int (0.0 + ((0.0 + $y - $ytopmargin) / ($scaley * $ybox)) )]
    if {$cursorObject < 0} {
      set cursorObject 0
    }
    if {$cursorObject > $dataValNum} {
      set cursorObject $dataValNum 
    }
    if {! $cursorShown($currentMol) } {
      set cSelText [findCursorSelText $cursorObject] 
      revealCursorHighlight $cSelText
      set cursorShown($currentMol) 1
    }
    #yipes, code duplication from letGoMarquee
    #Should really de-sel everything, but need to keep track of separate sels, perhaps in the text vs. null later XXX
    if { [molinfo $currentMol get frame] != $cursorFrame } { 
      #test, and save/restore
      #XX alter here to allow un-frame--synched data
      mol top $currentMol
      animate goto $cursorFrame

      #puts "jumped to $cursorFrame"
      drawTimeBar $cursorFrame
      #update both GL and tk timebar
      display update
      #puts "time for disp. update = [time {display update}]"
      #puts "time for disp. update ui= [time {display update ui}]"
    }
    drawCursorObjectBar $cursorObject $cursorFrame
    display update ui
  }

   

  proc  ::timeline::drawTimeBar {f} {
    variable w
    variable dataWidth
    variable scalex
    variable xcol 
    variable dataOrigin
    variable ycanmax

    set xTimeBarStart  [expr  ( ($f + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xTimeBarEnd  [expr  ( ($f + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    #more efficient to re-configure x1 x2
    $w.can delete timeBarRect
    #set timeBar [$w.can create rectangle  $xTimeBarStart 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "\#000000" -stipple gray50 -outline "" -tags [list dataScalable timeBarRect ] ]
    set timeBar [$w.can create rectangle  $xTimeBarStart 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "" -outline "\#000000"  -tags [list dataScalable timeBarRect ] ]
    set timeBar2 [$w.can create rectangle  $xTimeBarStart 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "" -outline "\#A0A0A0" -dash . -tags [list dataScalable timeBarRect ] ]
    
    #move the time line 
  } 

proc ::timeline::findCursorSelText {obj} {
  variable usesFreeSelection
  variable dataVal
   
  if $usesFreeSelection {
    set theCursorSelText $dataVal(freeSelString,$obj)
  } else {
    if {$dataVal(segname,$obj)!="emptyval"} {
      set theCursorSelText "resid $dataVal(resid,$obj) and  chain $dataVal(chain,$obj) and segname $dataVal(segname,$obj)" 
    } else {
      set theCursorSelText "resid $dataVal(resid,$obj) and  chain $dataVal(chain,$obj)" 
    }
  } 
  return $theCursorSelText
}

proc  ::timeline::drawCursorObjectBar {obj f} {
    variable w
    variable dataWidth
    variable scalex
    variable scaley
    variable ytopmargin 
    variable ybox
    variable xcol 
    variable dataOrigin
    variable ycanmax
    variable vertScaleWidth
    variable horzScaleHeight
    variable dataVal
    variable currentMol
    variable dataName
    variable usesFreeSelection
    #duplicated code from draw
    set xTimeBarStart  [expr  ( ($f + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xTimeBarEnd  [expr  ( ($f + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    #more efficient to re-configure x1 x2
   set ypos [expr $ytopmargin + ($scaley * $ybox * int ($obj))]
    $w.can delete cursorObjectBarRect
    $w.vertScale delete cursorObjectBarRect
    $w.horzScale delete cursorObjectBarRect
    #new position of cursorObject could scale width of outline... XXX


    set cursorSelText [findCursorSelText $obj]
   # tlPutsDebug "in drawCursorObjectBar: cursorSelText= $cursorSelText"
    showCursorHighlight $cursorSelText
    set cursorObjectBar [$w.can create rectangle  $xTimeBarStart $ypos $xTimeBarEnd [expr $ypos+ ($scaley * $ybox)]    -outline "\#FF0000" -width 2 -fill "" -tags [list dataScalable cursorObjectBarRect ] ] 
    set cursorObjectBarInVertScale [$w.vertScale create rectangle  1 $ypos $vertScaleWidth [expr $ypos+ ($scaley * $ybox)]    -outline "\#FF0000" -width 2 -fill "" -tags [list yScalable cursorObjectBarRect ] ] 
    set cursorObjectBarInHorzScale [$w.horzScale create rectangle  $xTimeBarStart 1 $xTimeBarEnd $horzScaleHeight    -outline "\#FF0000" -width 2 -fill "" -tags [list xScalable cursorObjectBarRect ] ] 

   configureSelInfo $obj $f
}

proc ::timeline::configureSelInfo {obj f} {
  variable w
  variable dataWidth
  variable scalex
  variable scaley
  variable ytopmargin 
  variable ybox
  variable xcol 
  variable dataOrigin
  variable ycanmax
  variable vertScaleWidth
  variable horzScaleHeight
  variable dataVal
  variable currentMol
  variable dataName
  variable usesFreeSelection

  #parameter f is frame, for cursor movement
  if {$obj=="null"} {
     $w.selInfo configure -text " $dataName(vals) \n\n\n\n"
     return
  }
     
 updateThreshLabel $f

  set v $dataVal([expr $dataOrigin+$f],$obj) 
  if {$v != "null"} {
    if $usesFreeSelection {
           
           set labelFirstPart [string range $dataVal(freeSelLabel,$obj) 0 19]
           set labelSecondPart [string range $dataVal(freeSelLabel,$obj) 20 38]
           $w.selInfo configure -text "$dataName(vals)\n\n$labelFirstPart\n$labelSecondPart\n[format "%8.4g" $v]\nFrame [format "%5g" $f]"
    } else {
         set theSegname $dataVal(segname,$obj)
         if {$theSegname == "emptyval"} {set theSegname ""}
         if {$dataName(vals)=="struct"} {
           $w.selInfo configure -text "sec. structure\n\n$dataVal(resid,$obj) $dataVal(resname,$obj)\nchain: $dataVal(chain,$obj)  seg: $theSegname\n$v\nFrame [format "%5g" $f]"
         } else {
           $w.selInfo configure -text "$dataName(vals)\n\n$dataVal(resid,$obj) $dataVal(resname,$obj)\nchain: $dataVal(chain,$obj)  seg: $theSegname\n[format "%8.4g" $v]\nFrame [format "%5g" $f]"
        }
    }
  }
} 

# fix hanging vim syntax quote""







proc  ::timeline::updateThreshLabel {theFrame} {
   variable dataThreshVal
   variable maxThresh
   variable dataOrigin
   variable prevCursorFrame
   variable dataValNum
   variable w

 # Configure threshValLab
  set theField [expr $theFrame + $dataOrigin]
  set dvnum [expr $dataValNum +1]
  if {
     [ catch {
      if { $dataThreshVal($theField) != "null"} {
        set tval $dataThreshVal([expr $theFrame + $dataOrigin]) 
        $w.threshValLab  configure -text "$tval/$maxThresh ($dvnum) "
      } else  { 
        $w.threshValLab  configure -text "-/- ($dvnum) "
      }
    }]
  } {
      $w.threshValLab  configure -text "-/- ($dvnum) "
  } 
}

proc ::timeline::copyDataFiltered {min max frameStart frameEnd framesRequired} {

    variable w
    variable dataName
    variable dataVal
    variable dataMin
    variable dataMax
    variable dataValNum
    variable currentMol
    variable numDataFrames
    variable dataOrigin
    variable usesFreeSelection
    variable dataFileVersion
    variable numTrajFrames
    set dataValNumFiltered -1 
    variable copyBuf  

    #scan through data and set filter field
    for {set i 0} {$i<=$dataValNum} {incr i} {
       set dataVal(filter,$i) 0
       set framesPassed 0
         # calc min/max on read?
        for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
          if {($dataFrame >= $frameStart) && ($dataFrame <= $frameEnd)} {
             set curField [expr $dataOrigin + $dataFrame]
             tlPutsDebug "$dataValNum $dataFrame $dataVal($curField,$i)" 
             if {($dataVal($curField,$i) <= $max ) && ($dataVal($curField,$i) >= $min) } {
              incr framesPassed
             } 
          }
        }
        if {$framesPassed >= $framesRequired} {
          set dataVal(filter,$i) 1 
          incr dataValNumFiltered
          tlPutsDebug "Added line $i, now $dataValNumFiltered lines.  framesPassed= $framesPassed"
        } 
      }
    
   # Now write data to copy buffer
   set copyBuf(molName)  [molinfo $currentMol get name]
   set copyBuf(dataName,vals) $dataName(vals)
   set copyBuf(numDataFrames) $numDataFrames
   #set copyBuf(numItems) [expr $dataValNumFiltered + 1]
   set copyBuf(dataValNum) [expr $dataValNumFiltered ]
   # equivalent to clear Data, then fill with data. Works with usesFreeSelection = 0 or 1
  
  set copyBuf(usesFreeSelection) $usesFreeSelection
  set dataValNumFilteredCounter 0
  if {$usesFreeSelection} {
    for {set i 0} {$i<=$dataValNum} {incr i} {
      if {$dataVal(filter,$i)==1} {
        set copyBuf(dataVal,freeSelLabel,$dataValNumFilteredCounter) $dataVal(freeSelLabel,$i)
        set copyBuf(dataVal,fressSelString,$dataValNumFilteredCounter) $dataVal(freeSelString,$i)
        for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
          set curField [expr $dataOrigin + $dataFrame]
          set copyBuf(dataVal,$curField,$dataValNumFilteredCounter) $dataVal($curField,$i)
        }
        incr dataValNumFilteredCounter
      }
    }
  } else {
    set endStructs [expr $dataOrigin+ ($numDataFrames - 1)]
    for {set i 0} {$i<=$dataValNum} {incr i} {
      if {$dataVal(filter,$i)==1} {
        set copyBuf(dataVal,resid,$dataValNumFilteredCounter) $dataVal(resid,$i)
        set copyBuf(dataVal,chain,$dataValNumFilteredCounter) $dataVal(chain,$i)
        set copyBuf(dataVal,segname,$dataValNumFilteredCounter) $dataVal(segname,$i)
        for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
          set curField [expr $dataOrigin + $dataFrame]
          set copyBuf(dataVal,$curField,$dataValNumFilteredCounter) $dataVal($curField,$i)
          tlPutsDebug "(read: copyBuf(dataVal):  curField= $curField  dVNFilteredCoutner= $dataValNumFilteredCounter dataVal($curField,$i)= $dataVal($curField,$i)"
        }
          #we are proceeding through frames here for timeline
          #this loop is over already-known frame info
          
         incr dataValNumFilteredCounter
 
        }
       #if dataval ends
      }
      #for ends
    }
   #else ends


 #Now read the data back in, 


  set oldNumDataFrames $numDataFrames

  clearData


  #get file lines into an array
  # really should store old values if any of this fails,
  # and set namesapce vars only after successful read
  set commonName ""
  set fileLines ""
  
  set usesFreeSelection  $copyBuf(usesFreeSelection)  
  set dataValNum $copyBuf(dataValNum) 
  set molName $copyBuf(molName)
  set numDataFrames $copyBuf(numDataFrames)
  set dataName(vals) $copyBuf(dataName,vals)
  #done with the file close it 
  if {$numDataFrames!= $numTrajFrames} {
    # XXX ask what to do here - map to frame 1, or interpolate (perhaps this should be menu setting) 
    tlPutsDebug "readDataFile: Input data frames is $numDataFrames, trajectory has $numTrajFrames  Timeline will map data frame linearly onto closest trajectory frame."
}
  if {$numDataFrames!= $oldNumDataFrames} {
    tlPutsDebug "readDataFile: oldNumDataFrames= $oldNumDataFrames   numDataFrames= $numDataFrames."
  }

  set dataMin(all) "null"
  set dataMax(all) "null"
  if {$usesFreeSelection==1} {
     #tlPutsDebug "start checking free selection"
    for {set i 0} {$i<=$dataValNum} {incr i} {
       set dataVal(freeSelLabel,$i) $copyBuf(dataVal,freeSelLabel,$i)
       set dataVal(freeSelString,$i) $copyBuf(dataVal,fressSelString,$i)
       for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
          set curField [expr $dataOrigin + $dataFrame]
          set dataVal($curField,$i) $copyBuf(dataVal,$curField,$i)
          checkRangeLimits $dataVal($curField,$i)
       }
    }
  } else {
    for {set i 0} {$i<=$dataValNum} {incr i} {
        set dataVal(resid,$i) $copyBuf(dataVal,resid,$i)
        set dataVal(chain,$i) $copyBuf(dataVal,chain,$i)
        set dataVal(segname,$i) $copyBuf(dataVal,segname,$i) 
        for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
          set curField [expr $dataOrigin + $dataFrame]
          set  dataVal($curField,$i) $copyBuf(dataVal,$curField,$i)
          checkRangeLimits $dataVal($curField,$i)
          tlPutsDebug "Copy back (copyBuf(dataVal):  curField= $curField  i= $i dataVal($curField,$i)= $dataVal($curField,$i)"
        }
    }
}
  set lastCalc -1

  if {$dataName(vals) == "struct"} {
    set dataMin(all) "null"
    set dataMax(all) "null"
  }

  #redraw the data rects
  configureSelInfo null 0
  initPicked 
  postDataFill
  return


}




proc ::timeline::writeDataFileFiltered {filename min max frameStart frameEnd framesRequired} {

    variable w
    variable dataName
    variable dataVal
    variable dataMin
    variable dataMax
    variable dataValNum
    variable currentMol
    variable numDataFrames
    variable dataOrigin
    variable usesFreeSelection
    variable dataFileVersion
    set dataValNumFiltered -1 

    if {$filename == ""  } {
      set filename [tk_getSaveFile -initialfile $filename -title "Save Filtered Trajectory Data file" -parent $w -filetypes [list { {.tml files} {.tml} } { {Text files} {.txt}} {{All files} {*} }] ]
    }
    if {$filename == ""  } {return}

    #scan through data and set filter field
    for {set i 0} {$i<=$dataValNum} {incr i} {
       set dataVal(filter,$i) 0
       set framesPassed 0
         # calc min/max on read?
        for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
          if {($dataFrame >= $frameStart) && ($dataFrame <= $frameEnd)} {
             set curField [expr $dataOrigin + $dataFrame]
             tlPutsDebug "$dataValNum $dataFrame $dataVal($curField,$i)" 
             if {($dataVal($curField,$i) <= $max ) && ($dataVal($curField,$i) >= $min) } {
              incr framesPassed
             } 
          }
        }
        if {$framesPassed >= $framesRequired} {
          set dataVal(filter,$i) 1 
          incr dataValNumFiltered
          tlPutsDebug "Added line $i, now $dataValNumFiltered lines.  framesPassed= $framesPassed"
        } 
      }
    

    set writeDataFile [open $filename w]
    puts $writeDataFile "# VMD Timeline data file"
    puts $writeDataFile "# CREATOR= $::tcl_platform(user)"
    puts $writeDataFile "# MOL_NAME= [molinfo $currentMol get name]"
    puts $writeDataFile "# DATA_TITLE= $dataName(vals)"
    puts $writeDataFile "# FILE_VERSION= $dataFileVersion"
    puts $writeDataFile "# NUM_FRAMES= $numDataFrames "
    puts $writeDataFile "# NUM_ITEMS= [expr $dataValNumFiltered + 1]"

    if {$usesFreeSelection} {
      puts $writeDataFile "# FREE_SELECTION= 1"
      puts $writeDataFile "#"
      for {set i 0} {$i<=$dataValNum} {incr i} {
        if {$dataVal(filter,$i)==1} {
          puts $writeDataFile "freeSelLabel $dataVal(freeSelLabel,$i)"
          puts $writeDataFile "freeSelString $dataVal(freeSelString,$i)"
          for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
            set curField [expr $dataOrigin + $dataFrame]
            puts $writeDataFile "$dataFrame $dataVal($curField,$i)" 
          }
        }
      }
    } else {
      puts $writeDataFile "# FREE_SELECTION= 0"
      puts $writeDataFile "#"
      set endStructs [expr $dataOrigin+ ($numDataFrames - 1)]
      for {set field $dataOrigin} {$field <= $endStructs} {incr field} {
        set frame [expr $field - $dataOrigin]
        for {set i 0} {$i<=$dataValNum} {incr i} {
          if {$dataVal(filter,$i)==1} {
            set val $dataVal($field,$i)
            set resid $dataVal(resid,$i)
            set chain $dataVal(chain,$i)
            set segname $dataVal(segname,$i)
            #we are proceeding through frames here for timeline
            #this loop is over already-known frame info
            puts $writeDataFile "$resid $chain $segname $frame $val"
          }
        }
      }
    }
    close $writeDataFile
    return
}



proc ::timeline::writeDataFile {filename} {

    variable w
    variable dataName
    variable dataVal
    variable dataMin
    variable dataMax
    variable dataValNum
    variable currentMol
    variable numDataFrames
    variable dataOrigin
    variable usesFreeSelection
    variable dataFileVersion
    if {$filename == ""  } {
      set filename [tk_getSaveFile -initialfile $filename -title "Save Trajectory Data file" -parent $w -filetypes [list { {.tml files} {.tml} } { {Text files} {.txt}} {{All files} {*} }] ]
    }
    if {$filename == ""  } {return}
    
    set writeDataFile [open $filename w]
    puts $writeDataFile "# VMD Timeline data file"
    puts $writeDataFile "# CREATOR= $::tcl_platform(user)"
    puts $writeDataFile "# MOL_NAME= [molinfo $currentMol get name]"
    puts $writeDataFile "# DATA_TITLE= $dataName(vals)"
    puts $writeDataFile "# FILE_VERSION= $dataFileVersion"
    puts $writeDataFile "# NUM_FRAMES= $numDataFrames "
    puts $writeDataFile "# NUM_ITEMS= [expr $dataValNum + 1]"

    if {$usesFreeSelection} {
      puts $writeDataFile "# FREE_SELECTION= 1"
      puts $writeDataFile "#"
      for {set i 0} {$i<=$dataValNum} {incr i} {
        puts $writeDataFile "freeSelLabel $dataVal(freeSelLabel,$i)"
        puts $writeDataFile "freeSelString $dataVal(freeSelString,$i)"
          # calc min/max on read?
        for {set dataFrame 0} {$dataFrame < $numDataFrames} {incr  dataFrame} {
          set curField [expr $dataOrigin + $dataFrame]
          puts $writeDataFile "$dataFrame $dataVal($curField,$i)" 
        }
      }
    } else {
      puts $writeDataFile "# FREE_SELECTION= 0"
      puts $writeDataFile "#"
      set endStructs [expr $dataOrigin+ ($numDataFrames - 1)]
      for {set field $dataOrigin} {$field <= $endStructs} {incr field} {
        set frame [expr $field - $dataOrigin]
        for {set i 0} {$i<=$dataValNum} {incr i} {
          set val $dataVal($field,$i)
          set resid $dataVal(resid,$i)
          set chain $dataVal(chain,$i)
          set segname $dataVal(segname,$i)
          #we are proceeding through frames here for timeline
          #this loop is over already-known frame info
          #looks backwards since inherited approach from multicolumn
          puts $writeDataFile "$resid $chain $segname $frame $val"
        }
      }
    }
    close $writeDataFile
    return
  }

  proc ::timeline::calcDataStruct {} {
    variable w
    variable dataName
    variable dataVal
    variable dataValNum
    variable dataValNumResSel
    variable dataOrigin
    variable currentMol
    variable firstTrajField
    variable numTrajFrames
    variable dataMin
    variable dataMax
    variable lastCalc
    variable nullMolString
    variable usesFreeSelection
    variable partSelText
   variable progressCancelPressed
   variable progressDrawFraction
   variable progressBoxExists
   variable firstAnalysisFrame
   variable lastAnalysisFrame
   variable keyAtomSelString

    set usesFreeSelection 0
 
    if {$currentMol == $nullMolString} {
        return 
    }
    set dataValNum $dataValNumResSel
    set lastCalc 1
    progressBox 0


    set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]
    set dataName(vals) "struct"
    #for progressBox progress
    set calcCountTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
    set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
    set calcCount 0

    for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
      animate goto $trajFrame 
      display update ui
      $sel frame $trajFrame
      #puts "set frame to $trajFrame"
      
      #puts "now update for mol $currentMol"
      mol ssrecalc $currentMol
      #puts "updated"
      set structlist [$sel get structure]
      #puts "setting dataName([$dataOrigin+$trajFrame]) to struct..."

      set i 0
      foreach elem $structlist {
        set dataVal([expr $dataOrigin+$trajFrame],$i) $elem
        incr i
        
        if {$progressCancelPressed} break 
      }
 
      unset structlist; #done with it
      if {$progressBoxExists} {
          set completeFrac [expr $progressCalcFraction * double($calcCount)/$calcCountTotal]
          progressBoxConfigure [expr $completeFrac]
       }
     
      if {$progressCancelPressed} break 

      incr calcCount
    }
    if {$progressCancelPressed} {
        tlPutsDebug "progressCancelPressed" 
        set progressCancelPressed 0
        clearData
    }
    #XX a wasted recalc, but looks odd when drawing if we delay calc frame 0 last
    animate goto $firstAnalysisFrame 
    mol ssrecalc $currentMol

    #if just setting one set of data for every frame, 
    #should clear unused dataVal()'s, etc. here.
    configureSelInfo null 0
    
    return
  }

  proc ::timeline::checkRangeLimits {elem} {
    # requires setting dataMin(all) and dataMax(all) to null with 
    # new data set
    variable dataMin
    variable dataMax
    if {$dataMin(all) == "null"} then { 
      set dataMin(all) $elem
    } else {
      # XX is it faster to do sort?
      catch {
        if {$elem < $dataMin(all)}  {
        set dataMin(all) $elem
        }
      } 
    }
    if {$dataMax(all) == "null"} then {
      set dataMax(all) $elem
    } else {
      catch {
        if {$elem > $dataMax(all)} {
        set dataMax(all) $elem
        }
       } 
    }   
  }

  proc ::timeline::calcDataProperty {propertyString unitString lastCalcVal} {
    tlPutsDebug ": in calcDataProperty"
    variable w
    variable dataName
    variable dataVal
    variable dataValNum
    variable dataValNumResSel
    variable currentMol
    variable firstTrajField
    variable numTrajFrames
    variable dataMin
    variable dataMax
    variable trajMin
    variable trajMax 
    variable lastCalc
    variable dataOrigin
    variable nullMolString
    variable usesFreeSelection
    variable anyResFuncName
    variable anyResFuncDesc
    variable firstAnalysisFrame
    variable lastAnalysisFrame
    variable partSelText 
    variable progressCancelPressed
    variable progressDrawFraction
    variable progressBoxExists
    variable keyAtomSelString
    #clearData

    #anyResFuncName is namespace-path name of a user-defined function (procedure that returns one value )(so is a function), to be is applied to all residues
    set firstFrame 0
    set lastFrame [expr $numTrajFrames -1]
    #this will be externally settable later

    set usesFreeSelection 0
    set dataValNum $dataValNumResSel

    if {$currentMol == $nullMolString} {
        #should really gray out choices unless molec is seleted XXX
        puts "Timeline: select molecule before choosing Calculate method"
        return 
    }
    set lastCalc $lastCalcVal 
      #XXX next line fragile in use, since relies on label/sel info (residue, chain, etc.) to have gotten this info in same order in different call .  Not sure this is guaranteed.
   
      progressBox 0 
      
      set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]

      set progressCalcFraction [expr 1.0 - $progressDrawFraction] 


    if {$propertyString != "anyResFunc"} {    
      tlPutsDebug ": VMD Timeline--CalcDataProperty starting for simple property (NOT per-residue anyResFunc)"
      tlPutsDebug ": Timeline: propertyString= $propertyString"
      if {$unitString==""} {
        set dataName(vals) "${propertyString}"
      } else {
        set dataName(vals) "${propertyString} (${unitString})"
      }

      #for progressBox progress
      set calcCountTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
      set calcCount 0

      for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
        set curField [expr $dataOrigin + $trajFrame]
        $sel frame $trajFrame
        #puts "set frame to $trajFrame"
        #this is quick check, not real way to do it 
        #that it doesn't use 'same selelction' 
        # currently just depends on 'sel get' order being same
        #method as other stuff, should really get data and sort it
        #
        set trajList [$sel get $propertyString]
     
 
        #tlPutsDebug "calcCountTotal= $calcCountTotal"
        #tlPutsDebug "propertyString= $propertyString; trajList= $trajList" 
        #does position detection use next line -- if not delete?
        set i 0
        foreach elem $trajList {
          set dataVal($curField,$i) $elem
          checkRangeLimits $elem
          #tlPutsDebug "checked rangeLimits for $elem, dataMin(all)= $dataMin(all) dataMax(all)= $dataMax(all)"
          incr i

          
          #tlPutsDebug "calcCount= $calcCount $calcCountTotal= $calcCountTotal   completeFrac= $completeFrac%"
          if {$progressCancelPressed} break 
        }
        incr calcCount
        if {$progressBoxExists} {
          set completeFrac [expr $progressCalcFraction * double($calcCount)/$calcCountTotal]
          progressBoxConfigure [expr $completeFrac]
        }
       if {$progressCancelPressed} break 
      } 
    } else {
      tlPutsDebug ": VMD Timeline--CalcDataProperty starting anyResFunc calc"
      #now for anyResFunc calc
      #should do syntax check, time check on at least one residue...
      set resAtomIndexList [$sel get index]
      set resAtomIndexListLength [llength $resAtomIndexList]
      # the counter i tracks residue in dataVal(curField, i)
      set i 0
      foreach resAtomIndex $resAtomIndexList {
       #tlPutsDebug ": VMD Timeline: starting res-row $i, index= $resAtomIndex"
       #tlPutsDebug "three atomselects now:"
        set resAtomSel [atomselect $currentMol "index $resAtomIndex"]
        set resCompleteSel [atomselect $currentMol "same residue as index $resAtomIndex"]
        set proteinNucSel [atomselect $currentMol "protein or nucleic"]
        # XXX how to speed up as array?
        #now the user function can choose to use either a core atom (resAtomSel) or all atoms in residue (resCompleteSel) with neither penalized for selection time
        set dataName(vals) "${anyResFuncDesc}"
        for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
        $resAtomSel frame $trajFrame
        $resCompleteSel frame $trajFrame
        $proteinNucSel frame $trajFrame
           #set frame for both sel options
           #XXX if either of above two is time consuming, use user-set switches to choose which of them gets frame set
        set curField [expr $dataOrigin + $trajFrame]
        #now run proc, in the current context 
        if {
          [catch { 
             #run the proc
             #value would be set for trajFrame and resAtomIndex
             #tlPutsDebug "about to run  user-defined proc"
             set val [$anyResFuncName $resAtomSel  $resCompleteSel  $proteinNucSel ]


             #tlPutsDebug ": user-defined proc has run"
            # XX replace with above code for built-ins?
                    
             set dataVal($curField,$i) $val
                
             checkRangeLimits $val
            
           # replace with above code for built-ins
               
             #tlPutsDebug ": frame= $trajFrame   atom index= $resAtomIndex i= $i  dataVal($curField,$i) = $dataVal($curField,$i)  dataMin(all)= $dataMin(all) dataMax(all)= $dataMax(all)"

            #XXX much wasted time here.  REPLACE with loop later.  Also, after first calc of this, should be min/max.  Second calc in a row should not happen, just use old values.
            #tlPutsDebug "assigned dataName, dataMind, dataMax for user def'd func" 
            }
          ] 
       } then {
          #complain about error
          puts "ERROR VMD::Timeline: User-defined residue procedure >${anyResFuncName}<"
          puts "ERROR VMD::Timeline: had error"
          puts "ERROR VMD::Timeline: for molecule $currentMol, atom index $resAtomIndex"
          #tlPutsDebug ": frame= $trajFrame   atom index= $resAtomIndex i= $i  dataVal($curField,$i) = $dataVal($curField,$i)  dataMin(all)= $dataMin(all) dataMax(all)= $dataMax(all)"

        }  
        if {$progressCancelPressed} break 
      }
      if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction $i  $resAtomIndexListLength] 
      } 
      if {$progressCancelPressed} break 

      incr i  
    }
  }
  if {$progressCancelPressed} {
    tlPutsDebug "progreeCancelPressed" 
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  } 
  configureSelInfo null 0
  tlPutsDebug "exiting calcDataProperty.  dataMin(all)= $dataMin(all) trajMin= $trajMin"
  return 
} 

#XXX currently, must manually match switch statemnt.  Make #these auto-register...
proc ::timeline::calcDataAnyResFunc {} {
  clearData; calcDataProperty "anyResFunc" "" 13
}

proc ::timeline::calcDataX {} {
  tlPutsDebug ": VMD Timeline: in calcDataX"
  clearData;
  calcDataProperty "x" "A" 2
  tlPutsDebug ": VMD Timeline: leaving calcDataX...."
}
proc ::timeline::calcDataY {} {
  calcDataProperty "y" "A" 3
}
proc ::timeline::calcDataZ {} {
  clearData; calcDataProperty "z" "A" 4
}

proc ::timeline::calcDataPhi {} {
  clearData; calcDataProperty "phi" "deg" 5
}

proc ::timeline::calcDataDeltaPhi {} {
  clearData; calcDataDeltaProperty "phi" "deg" 1 6
}


proc ::timeline::calcDataPsi {} {
  clearData; calcDataProperty "psi" "deg" 7
}

proc ::timeline::calcDataDeltaPsi {} {
  clearData; calcDataDeltaProperty "psi" "deg"  1 8
}

proc ::timeline::test1 {} {tlPutsDebug ": Timeline: This is test routine 1."}

proc ::timeline::calcDataUser {} {
 clearData;  calcDataProperty "user" "" 12 
}
proc ::timeline::calcDisplacement {} {
  clearData; calcDisplacementProperty  14 
}
proc ::timeline::calcDispVelocity {} {
  clearData;calcDispVelocityProperty  15 
}
proc ::timeline::calcRMSD {} {
  clearData; calcRMSDProperty 17
}
proc ::timeline::calcRMSF {} {
  clearData; calcRMSFProperty 18
}
proc ::timeline::calcSASA {} {
  clearData; calcSASAProperty 19
}

proc ::timeline::calcNative {} {
  clearData; calcNativeContacts 20 
}

proc ::timeline::calcCC {} {
  clearData; calcGlobalCcss 22 
}

proc ::timeline::calcDispVelocityProperty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable partSelText
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable keyAtomSelString

  set usesFreeSelection 0

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal

  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
  progressBox 0

  set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]


  set frameTotal [expr $lastAnalysisFrame - $firstAnalysisFrame +1]
  set calcCount 0


  set dataName(vals) "dispVel (A/frm)"
  for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {

    set curField [expr $dataOrigin + $trajFrame]
    $sel frame $trajFrame
    #puts "set frame to $trajFrame"

    #this is quick check, not real way to do it
    #that it doesn't use 'same selelction'
    # currently just depends on 'sel get' order being same
    #method as other stuff, should really get data and sort it
    #
    set trajList [$sel get {x y z}] 
    #does position detection use next line -- if not, delete?
    set i 0
    foreach elem $trajList {
      if {$trajFrame == 0} {
          tlPutsDebug ": trajFrame= $trajFrame, curField=$curField, i=$i"
          set dataVal($curField,$i) 0
      } else {
           #XX only does central atom, change to COM/COG of residue (via index)
          set dataVal($curField,$i) [veclength [vecsub $elem  $dataVal(referenceVal,$i)] ]
      }
      checkRangeLimits $dataVal($curField,$i) 
      set dataVal(referenceVal,$i) $elem 
       
      incr i
    }
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction  $calcCount $frameTotal] 
    } 
     
    if {$progressCancelPressed} break
    incr calcCount
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }
  configureSelInfo null 0
}

proc ::timeline::calcSASAProperty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable partSelText 
  variable SASArad
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable keyAtomSelString

 tlPutsDebug "starting calcSASAProperty" 
  set usesFreeSelection 0
  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
 
  #XX this should be passed in as paramater
  set rad $SASArad

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal

  progressBox 0

  set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]
  #set restrictSel [atomselect $currentMol "water or ions"]
  set alist [$sel get index]
  set alistLength [llength $alist]
  set i 0
  set dataName(vals) "SASA (A^2)"
  foreach a $alist {
    set curSel [atomselect $currentMol "same residue as index $a" ]
    for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      $curSel frame $trajFrame
      #$restrictSel frame $trajFrame
      #set dataVal($curField,$i) [measure sasa $rad $curSel -restrict $restrictSel]
      set dataVal($curField,$i) [measure sasa $rad $curSel ]
      checkRangeLimits $dataVal($curField,$i) 
      if {$progressCancelPressed} break 
    }
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction  $i $alistLength] 
    } 
    if {$progressCancelPressed} break 
    incr i
    $curSel delete
 }
 if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }
  configureSelInfo null 0
}

proc ::timeline::calcRMSFProperty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable partSelText 
  variable RMSFstepSize
  variable RMSFwindowWidth
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable keyAtomSelString
 
  tlPutsDebug "starting calcRMSFProperty" 
  set usesFreeSelection 0
  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
 
  #XX this should be passed in as paramater
  # check these are compatible
  set stepSize $RMSFstepSize
  set windowWidth $RMSFwindowWidth

  set halfWidth [expr int($windowWidth/2.0)]

  if  {$halfWidth<1} {
    set halfWidth 1
  }

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal

  progressBox 0

  set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]

  set alist [$sel get index]
  set alistLength [llength $alist]
  set i 0
  set frameTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
  set dataName(vals) "RMSF (A)"
  foreach a $alist {
    set curSel [atomselect $currentMol "same residue as index $a" ]
    for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      set firstF [expr $trajFrame-$halfWidth]
      if {$firstF < 0} then {
        set firstF 0
      }
      set lastF [expr $trajFrame+$halfWidth]
      if {$lastF > ($frameTotal -1)} then {
        set lastF [expr $frameTotal-1]
      }
      set dataVal($curField,$i) [vecmean [measure rmsf $curSel first $firstF last $lastF step $stepSize]]
      checkRangeLimits $dataVal($curField,$i) 
      if {$progressCancelPressed} break 
      }
      if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction  $i $alistLength] 
      } 
      if {$progressCancelPressed} break 
      incr i
   }
   if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }

  configureSelInfo null 0
}

proc ::timeline::calcRMSDProperty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable partSelText 
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable keyAtomSelString
 
  set usesFreeSelection 0
 
  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 

  #XX this should be passed in as paramater
  set refFrame $firstAnalysisFrame 

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal

  progressBox 0

  set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]

  set alist [$sel get index]
  set alistLength [llength $alist]
  set i 0
  set frameTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
  set dataName(vals) "RMSD (A)"
  foreach a $alist {
    set refSel [atomselect $currentMol "same residue as index $a" frame refFrame]
    set curSel [atomselect $currentMol "same residue as index $a" ]
    for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      set alistLength [llength $alist]
      $curSel frame $trajFrame
      set dataVal($curField,$i) [measure rmsd $refSel $curSel ]
      checkRangeLimits $dataVal($curField,$i) 
      if {$progressCancelPressed} break 
    }
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction  $i $alistLength] 
    } 
    $refSel delete
    $curSel delete
    if {$progressCancelPressed} break 
    incr i
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }

  configureSelInfo null 0
}


proc ::timeline::calcDisplacementProperty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable partSelText
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable keyAtomSelString


  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
 
  set usesFreeSelection 0

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal

  progressBox 0

  set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]
  set dataName(vals) "disp. (A)"

  set calcCount 0
  set frameTotal [expr $lastAnalysisFrame - $firstAnalysisFrame +1]

  for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
    set curField [expr $dataOrigin + $trajFrame]
    $sel frame $trajFrame
    #puts "set frame to $trajFrame"

    #this is quick check, not real way to do it
    #that it doesn't use 'same selelction'
    # currently just depends on 'sel get' order being same
    #method as other stuff, should really get data and sort it
    #
    set trajList [$sel get {x y z}] 
    #does position detection use next line -- if not, delete?
    set i 0
    foreach elem $trajList {
      if {$trajFrame == $firstAnalysisFrame} {
          set dataVal($curField,$i) 0
          set dataVal(referenceVal,$i) $elem 
      } else {
           #XX only does central atom, change to COM/COG of residue (via index)
          set dataVal($curField,$i) [veclength [vecsub $elem  $dataVal(referenceVal,$i)] ]
        checkRangeLimits $dataVal($curField,$i) 
      }
       
      incr i
      #XXX this per-column min/max setting not needed, should set more widely
    }
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 1.0   $progressCalcFraction  $calcCount $frameTotal] 
    } 
    if {$progressCancelPressed} break
    incr calcCount 
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }
  configureSelInfo null 0

}

proc ::timeline::calcDataDeltaProperty {propertyString unitString isAngle lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable partSelText 
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable keyAtomSelString

  set usesFreeSelection 0

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal
  progressBox 0

  set sel [atomselect $currentMol "$partSelText and $keyAtomSelString"]
  if {$unitString==""} {
    set dataName(vals) "delta-${propertyString}"
  } else {
    set dataName(vals) "delta-${propertyString} (${unitString})"
  }

  #for progressBox progress
  set calcCountTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
  set calcCount 0



   for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
    set curField [expr $dataOrigin + $trajFrame]
    $sel frame $trajFrame
    #puts "set frame to $trajFrame"

    #this is quick check, not real way to do it
    #that it doesn't use 'same selelction'
    # currently just depends on 'sel get' order being same
    #method as other stuff, should really get data and sort it
    #
    set trajList [$sel get $propertyString] 
    #does position detection use next line -- if not, delete?
    set i 0
    foreach elem $trajList {
      if {$trajFrame == $firstAnalysisFrame} {
          #tlPutsDebug ": trajFrame= $trajFrame, curField=$curField, i=$i"
          set dataVal($curField,$i) 0
          set dataVal(referenceVal,$i) $elem 
      } else {
          if $isAngle {
            set dataVal($curField,$i) [expr (fmod ((900.0 + $elem - $dataVal(referenceVal,$i)), 360 )) - 180]
          } else {
            set dataVal($curField,$i) [expr $elem - $dataVal(referenceVal,$i)]
          }
      }
       
      checkRangeLimits $dataVal($curField,$i) 
      incr i

      
      #tlPutsDebug "calcCount= $calcCount $calcCountTotal= $calcCountTotal   completeFrac= $completeFrac%"
      if {$progressCancelPressed} break 
    }
    incr calcCount
    if {$progressBoxExists} {
        set completeFrac [expr $progressCalcFraction * double($calcCount)/$calcCountTotal]
        progressBoxConfigure [expr $completeFrac]
    }
   if {$progressCancelPressed} break 
  }
  if {$progressCancelPressed} {
    tlPutsDebug "progressCancelPressed" 
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }
  configureSelInfo null 0
  return
}

proc ::timeline::calcSelEmpty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax 
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable dataThreshVal
  set usesFreeSelection 1

  set lastCalc $lastCalcVal

  
  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "Timeline: select molecule before choosing Calculate method"
      return 
  }


  for {set trajFrame 0} {$trajFrame < $numTrajFrames} {incr  trajFrame} {
    #clear out all frames, in real case set all data.
    set curField [expr $dataOrigin + $trajFrame]
    set dataThreshVal($curField) "null"
    for {set displayGroup 0} {$displayGroup<1} {incr displayGroup} {
      set dataVal($curField,$displayGroup) 0 
    }
    set dataName(vals) "--"
  }
    #use next line if really extracting data from traj geom.
    #$sel frame $trajFrame
    #manual test set
    set dataVal(freeSelLabel,0) "-" 
    set dataVal(freeSelString,0) "none"
    #since we have dataVal (---,0) to (---,0) we have 1, but
    # this number is elesewhere set so gives last number from 0 count.
    set dataValNum 0
    checkRangeLimits 0
} 


proc ::timeline::calcTestFreeSel {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax 
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  set usesFreeSelection 1

  set lastCalc $lastCalcVal

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "Timeline: select molecule before choosing Calculate method"
      return 
  }


  for {set trajFrame 0} {$trajFrame < $numTrajFrames} {incr  trajFrame} {
    #clear out traj frames of dataVal columns, in real case set all data.
    for {set displayGroup 0} {$displayGroup<7} {incr displayGroup} {
      set curField [expr $dataOrigin + $trajFrame]
      set dataVal($curField,$displayGroup) 0 
    }
  }
    #use next line if really extracting data from traj geom.
    #$sel frame $trajFrame
    set dataName(vals) "free-sel. test"
    #manual test set
    set dataVal(freeSelLabel,0) "res 23 / res 28" 
    set dataVal(freeSelString,0) "resid 23 28"
    set dataVal([expr $dataOrigin+5],0) 100
    set dataVal([expr $dataOrigin+23],0) -100
    set dataVal([expr $dataOrigin+24],0) -100
    set dataVal([expr $dataOrigin+25],0) -100
    set dataVal([expr $dataOrigin+26],0) -100
    set dataVal(freeSelLabel,1) "res 1-5 / res 70-80"
    set dataVal(freeSelString,1) "resid 1 to 5 70 to 80"
    set dataVal([expr $dataOrigin+2],1) -100
    set dataVal([expr $dataOrigin+3],1) -100
    set dataVal([expr $dataOrigin+4],1)  70 
    set dataVal([expr $dataOrigin+5],1)  90 
    set dataVal([expr $dataOrigin+6],1)  110 
    set dataVal([expr $dataOrigin+17],1) -50 
    set dataVal([expr $dataOrigin+50],1) 150 
    set dataVal(freeSelString,2) "resid 60 61 62"
    set dataVal(freeSelLabel,2) "resid 60 61 62"
    set dataVal($dataOrigin+27,2) -50 
    set dataVal($dataOrigin+40,2) 150 
    set dataVal($dataOrigin+41,2) 150 
    set dataVal($dataOrigin+42,2) 150 
    set dataVal($dataOrigin+43,2) 150 
    set dataVal(freeSelString,3)  "resid 70 to 75"
    set dataVal(freeSelLabel,3) "resid 70 to 75"
    set dataVal($dataOrigin+17,3) -10 
    set dataVal($dataOrigin+80,3) 100 
    set dataVal($dataOrigin+45,3) 70
    set dataVal($dataOrigin+46,3) 80
    set dataVal(freeSelString,4) "resid 20 to 25 28 67"
    set dataVal(freeSelLabel,4) "some in 20's and 60's"
    set dataVal($dataOrigin+27,5) -40 
    set dataVal(freeSelString,5) "resid 50 51 52 61 62"
    set dataVal(freeSelLabel,5) "favorites"
    set dataVal($dataOrigin+27,5) -40 
    set dataVal($dataOrigin+38,5) 150 
    set dataVal($dataOrigin+39,5) 150 
    set dataVal($dataOrigin+40,5) 150 
    set dataVal(freeSelLabel,6) "res 9 / res 20"
    set dataVal(freeSelString,6) "resid 9 20"
    set dataVal([expr $dataOrigin+12],6) -130
    set dataVal([expr $dataOrigin+13],6) -100
    set dataVal([expr $dataOrigin+14],6)  170 
    set dataVal([expr $dataOrigin+15],6)  90 
    set dataVal([expr $dataOrigin+16],6)  110 
    set dataVal([expr $dataOrigin+17],6) -140 
    set dataVal([expr $dataOrigin+30],6) 150 
     #since we have dataVal (---,0) to (---,6) we have 7, but
    # this number is elesewhere set so gives last number from 0 count.
    set dataValNum 6
} 

proc ::timeline::threshChanged { var} {
  thresholdMakeGraph
}

proc ::timeline::thresholdMakeGraph {} {
  variable w
  variable currentMol 
  variable nullMolString
  variable everRedrawn
  variable usableMolLoaded
  variable dataValNum

  #since trace vars head here, find way to prevent
  #arrival while still calculating values
  if {$usableMolLoaded && $everRedrawn && ($currentMol != $nullMolString) &&  ($dataValNum >=0)} {
  
    thresholdData
    $w.threshGraph delete xScalable
    drawThresholdGraph
  }
}




proc ::timeline::thresholdClearGraph {} {
  variable w
 $w.threshGraph delete xScalable
}

proc ::timeline::readDataFile {filename} {

  variable w
  variable dataOrigin
  variable dataMin
  variable dataMax
  variable dataVal
  variable dataHash
  variable dataValNum
  variable dataValNumResSel
  variable rectCreated 
  variable dataFileVersion 
  variable usesFreeSelection
  variable dataName
  variable lastCalc  
  variable numDataFrames
  variable numTrajFrames

  tlPutsDebug "in readDataFile"

  set oldNumDataFrames $numDataFrames
 

  if {$filename == ""  } {
    set filename [tk_getOpenFile -initialfile $filename -title "Open Trajectory Data file" -parent $w -filetypes [list { {.tml files} {.tml} } { {Text files} {.txt}} {{All files} {*} }] ]

  } 
  if {$filename == ""} {return}
  clearData
  set dataFile [open $filename r]
  #get file lines into an array
  # really should store old values if any of this fails,
  # and set namesapce vars only after successful read
  set commonName ""
  set fileLines ""
  while {! [eof $dataFile] } {
    gets $dataFile curLine
    if { (! [regexp "^#" $curLine] ) && ($curLine != "" ) } {
      lappend fileLines $curLine
      if {[llength $fileLines] < 100} {
      tlPutsDebug "lines [llength $fileLines] is >$curLine<"
      }
    } else {
       
      if { [regexp "^# FILE_VERSION=" $curLine] } { 
        set inputFileVersion [lindex [split $curLine " "] 2]
        tlPutsDebug "Loading file, file version is $inputFileVersion"
      }
      if { [regexp "^# DATA_TITLE=" $curLine] } { 
         regexp "^# DATA_TITLE= (.*)$" $curLine matchall commonName
        tlPutsDebug "Loading file, field name is >$commonName<"
      } 
      if { [regexp "^# FREE_SELECTION=" $curLine] } { 
        set usesFreeSelection [lindex [split $curLine " "] 2]
      } 
      #dataValNum internally counts from 0, so subtract 1 from the counting number
      if { [regexp "^# NUM_ITEMS=" $curLine] } { 
        #XXX do nothing, really should compare and flag if different.
        #set dataValNum [expr [lindex [split $curLine " "] 2] -1]
      } 
      if { [regexp "^# MOL_NAME=" $curLine] } { 
        set molName [lindex [split $curLine " "] 2]
      }
      #numDataFrames counts from 1
      if { [regexp "^# NUM_FRAMES=" $curLine] } { 
        set numDataFrames [lindex [split $curLine " "] 2]
      } 
    }
  }
  #done with the file close it 
  close $dataFile
  tlPutsDebug "readDataFile: inputFileVersion= $inputFileVersion   usesFreeSelection= $usesFreeSelection  dataValNum= $dataValNum  numItems= [expr $dataValNum+1]  "
  if {$numDataFrames!= $numTrajFrames} {
    # XXX ask what to do here - map to frame 1, or interpolate (perhaps this should be menu setting) 
    tlPutsDebug "readDataFile: Input data frames is $numDataFrames, trajectory has $numTrajFrames  Timeline will map data frame linearly onto closest trajectory frame."
}
  if {$numDataFrames!= $oldNumDataFrames} {
    tlPutsDebug "readDataFile: oldNumDataFrames= $oldNumDataFrames   numDataFrames= $numDataFrames."
  }

  set dataMin(all) "null"
  set dataMax(all) "null"
  set dataName(vals) $commonName
  if {$usesFreeSelection==1} {
     #tlPutsDebug "start checking free selection"

     set itemNum -1 
     # track when last increment so is forgiving on using label or string first for new block
    #set seenDataForItem 0
    foreach curLine $fileLines {
      tlPutsDebug "curLine= $curLine"
      if {$inputFileVersion<=1.3} then {
        if { [regexp "^freeSelLabel" $curLine] } { 
           #tlPutsDebug "found freeSelLabel..."
           regexp "^freeSelLabel (\\d+) (.*)$" $curLine matchall itemNum theLabel
           set  dataVal(freeSelLabel,$itemNum) $theLabel
           #tlPutsDebug "dataVal(freeSelLabel,$itemNum)= $dataVal(freeSelLabel,$itemNum)  theLabel= >$theLabel<"
        }
        if { [regexp "^freeSelString" $curLine] } { 
           regexp "^freeSelString (\\d+) (.*)$" $curLine matchall itemNum theSelString
           set  dataVal(freeSelString,$itemNum) $theSelString
        }
       
        if {[regexp "^(\\d+) (\\d+) (.+)$" $curLine matchall frameNum itemNum theVal]} {
          #proceed through lines
          set curField [expr $dataOrigin + $frameNum]
          set dataVal($curField,$itemNum) $theVal
          checkRangeLimits $theVal 
          tlPutsDebug "framenum= >$frameNum< itemNum= $itemNum  theVal= $theVal curField= $curField   dataVal($curField,$itemNum)= >$dataVal($curField,$itemNum)< dataMin= $dataMin(all)  dataMax=$dataMax(all)"
        }
      } else {
        tlPutsDebug "starting ver 1.4 curLine analysis"
        if { [regexp "^freeSelLabel" $curLine] } { 
           incr itemNum
           #For now, we force user to put label, then sel string.  Later cleverness may relax this requirement.
           #if {!($seenDataForItem)} {
           # set seenDataForItem 0
           #}
           tlPutsDebug "found freeSelLabel..., itemNum= $itemNum"
           regexp "^freeSelLabel (.*)$" $curLine matchall  theLabel
           set  dataVal(freeSelLabel,$itemNum) $theLabel
            tlPutsDebug "dataVal(freeSelLabel,$itemNum)= >$dataVal(freeSelLabel,$itemNum)<  theLabel= >$theLabel<"
        }
        if { [regexp "^freeSelString" $curLine] } { 
           #if {!($seenDataForItem)} {
           # set seenDataForItem 0
           #}
           regexp "^freeSelString (.*)$" $curLine matchall  theSelString
           set  dataVal(freeSelString,$itemNum) $theSelString
        }


        #if starts with a number, isn't the 'freeSelLabel' or 'freeSelString' identifier, must be data       
        if {[regexp "^(\\d+) (.+)$" $curLine matchall frameNum theVal]} {
          #proceed through lines
          #set seenDataForItem 1
          set curField [expr $dataOrigin + $frameNum]
          set dataVal($curField,$itemNum) $theVal
          checkRangeLimits $theVal 
          #tlPutsDebug "framenum= >$frameNum< itemNum= $itemNum  theVal= $theVal curField= $curField   dataVal($curField,$itemNum)= >$dataVal($curField,$itemNum)< dataMin= $dataMin(all)  dataMax=$dataMax(all)"
        } 


      }
    }
    set dataValNum $itemNum   
  } else {
      set dataValNum $dataValNumResSel
      set frameList ""
      #data-containing frames
      foreach line $fileLines {
        #puts "the line is >$line<"
        #XXX insert file version check here
        #For file version 1.3
        #foreach {resid chain atom frame val} [split $line " "] {}
        #For file version 1.4
        foreach {resid chain segname frame val} [split $line " "] {}
        # by luck, version  1.4 is downward compatible w/ 1.3, with segname in the place-held atom spot.
        lappend frameList $frame
      }  
      #puts "frameList is $frameList"
      tlPutsDebug "length of frameList is [llength $frameList]" 
      set frameList [lsort -unique -increasing -integer $frameList]
      set minFrame [lindex $frameList 0]
      set maxFrame [lindex $frameList end]
           tlPutsDebug "frameList is $frameList"
      #  no longer find frame list, since catching errors on frame assignment
      #has same effect.  Could still 
      #assign values in a new Group
      # (temporarlily, to hard-coded fields, if still in hacky version)
      tlPutsDebug "now check fileLines:\n"
      foreach line $fileLines {
        #tlPutsDebug "assigning data, the line is >$line<"
      #Need to Check for version 1.3 again. Have default segname ready. 
        foreach {resid chain segname frame val} [split $line " "] {}


        #hacky, acceptable since since this is a string > 4 chars, so won't be in PDB format files        
        if {$segname == "{}"} then {segname = "emptyval"}
  
        #this assumes consecutive frames, should use frameList somewhere
        # if we really want proper reverse lookup
        if { [ catch {set fieldForFrame [expr $dataOrigin + $frame ]} ] } {
          set fieldForFrame -2
          puts "Info) Timeline: Warning: While reading file, couldn't parse frame text \"$frame\""
        }
        #now do lookup via dataHash to find index in dataVal 
        # we depend on segname having no spaces/word breaks. 
        if {[catch {set theIndex $dataHash($resid,$chain,$segname)} ]} {
          puts "failed to find data for resid=$resid, chain=$chain, segname= $segname"
        } else {
           if { [catch {set dataVal($fieldForFrame,$theIndex) $val} ]} {
           puts "didn't find data for frame $frame, field= $fieldForFrame, index= $theIndex, new_val= $val"
         } else {
           checkRangeLimits $val 
           #tlPutsDebug "frame= >$frame<    dataVal($fieldForFrame,$theIndex) = $dataVal($fieldForFrame,$theIndex) dataMin= $dataMin(all)  dataMax=$dataMax(all)"
        }
       }
    }   
  }
  #now delete the list of data lines, no longer needed
  unset fileLines

  set lastCalc -1

  if {$dataName(vals) == "struct"} {
    set dataMin(all) "null"
    set dataMax(all) "null"
  }

  #redraw the data rects
  configureSelInfo null 0
  initPicked 
  postDataFill
  return
}

proc ::timeline::loadDataCollection {} {
  variable w
  #batch load code  here
  set ext "\[tT\]\[mM\]\[lL\]"
  set dir [tk_chooseDirectory   -title "Choose a data collection directory"]
  if { [catch { set fileList [lsort [glob -directory $dir -type f *.$ext]] } ]} {
    #no glob match or other error or cancel was pressed 
    # XX needs better response than this, and to have different responses if cancel was pressed in tk_chooseDir vs. fail to find ext files
  } else {
    $w.menubar.data.menu delete 0 end
    $w.menubar.data.menu add command -label "Set collection directory..." -command  [namespace code loadDataCollection]
    foreach f $fileList  {
      set shortf [file tail $f]
      #set cmd "\{readDataFile $f\}"
      $w.menubar.data.menu add command -label "$shortf" -command [namespace code "readDataFile $f"]
    }
  }
}


proc ::timeline::thresholdData {} {
  variable w
  variable dataVal
  variable dataValNum
  variable dataName
  variable dataOrigin
  variable numDataFrames
  variable usableMolLoaded
  variable rectCreated
  variable lastCalc
  variable dataThresh
  variable dataThreshVal
  variable thresholdBoundMin
  variable thresholdBoundMax

  #puts "in thresholdData, starting"
  set endField [expr $dataOrigin + $numDataFrames - 1 ]
  for {set field $dataOrigin} {$field <= $endField} {incr field} {
    set dataThreshVal($field) 0
    #puts "just set dataThresVal($field) to 0"
    for {set i 0} {$i<=$dataValNum} {incr i} {
      #puts "tD, started loop, i= $i"
      if {$dataName(vals) == "struct"} {
        #if { ($dataVal($field,$i) == $thresholdBoundMin) || ($dataVal($field,$i) == $thresholdBoundMax)} 
        #hack so struct at least shows some threshold graph,  do something user configurable later 
        if { ($dataVal($field,$i) == "E") || ($dataVal($field,$i) == "T")} {
 
           incr dataThreshVal($field) 
            set dataThresh($field,$i) 1
          } else {
            set dataThresh($field,$i) 0
          }
             
        } else {

          if { ($dataVal($field,$i) > $thresholdBoundMin) && ($dataVal($field,$i) <= $thresholdBoundMax)} {
              incr dataThreshVal($field) 
              #puts "incremented  dataThreshVal($field)=  $dataThreshVal($field)" 


              set dataThresh($field,$i) 1
          } else {
              set dataThresh($field,$i) 0
          }
         #tlPutsDebug  " dataThresh($field,$i)=  $dataThresh($field,$i)" 
        }
     }
     #puts "dataThreshVal($field)= $dataThreshVal($field)"   
 }
  return
}

proc ::timeline::drawThresholdGraph {} {
  variable dataThreshVal
  variable numDataFrames
  variable dataOrigin
  variable threshGraphHeight 
  variable scalex
  variable xcol
  variable w
  variable dataWidth
  variable currentMol
  variable nullMolString
  variable usableMolLoaded
  variable maxThresh
  variable prevCursorFrame
  #if {!($usableMolLoaded) || ($currentMol == $nullMolString)} {
  #   return
  # }

  #find min and max of Thresholds
  #make these variables later
  set threshPlotTop [expr  4 ]
  set threshPlotBottom [expr $threshGraphHeight - 5] 
  set lastField [expr $dataOrigin + $numDataFrames - 1]
  set minThresh $dataThreshVal($dataOrigin)
  set minThreshField $dataOrigin
  set maxThresh $dataThreshVal($dataOrigin)
  set maxThreshField $dataOrigin
  for {set field [expr $dataOrigin+1]} {$field<=$lastField} {incr field} {
    if {$dataThreshVal($field) < $minThresh} {set minThresh $dataThreshVal($field); set minThreshField $field} 
    if {$dataThreshVal($field) > $maxThresh} {set maxThresh $dataThreshVal($field); set maxThreshField $field} 
  }
  if {$maxThresh == 0} {set depictedMaxThresh 10} else {set depictedMaxThresh $maxThresh}
  set plotFactor [expr  ($threshPlotBottom-$threshPlotTop)/(0.0+$depictedMaxThresh) ]
     #puts "threshPlotTop=$threshPlotTop  thresPlotBottom=$threshPlotBottom   plotFactor= $plotFactor" 
   #count will be 0-based
   #later can do min based
   ##set plotFactor [expr 0.0+($threshPlotTop-$threshPlotBottom)/($maxThresh-$minThresh)]

  $w.threshGraph delete threshPlotBar 
  set endField [expr $dataOrigin + $numDataFrames - 1 ]

  for {set field $dataOrigin} {$field <= $endField} {incr field} {
    set frame [expr $field - $dataOrigin]
    set intermed [expr $plotFactor * $dataThreshVal($field)]
    set plotY [expr $threshPlotBottom - ($plotFactor * $dataThreshVal($field))]
    #puts "val= $intermed  dataThreshVal($field)= $dataThreshVal($field)  plotY=$plotY, field= $field threshPlotBottom=$threshPlotBottom "
    set xStart  [expr  ( ($frame + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xEnd  [expr  ( ($frame + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    $w.threshGraph create rectangle  $xStart $threshPlotBottom $xEnd $plotY -fill "\#EE7070"  -tags [list xScalable threshPlotBar]
    #puts "plotted   $xStart $threshPlotBottom $xEnd $plotY"
  }
    #mark min of the thresh
    set xStart  [expr  ( ([expr $minThreshField-$dataOrigin] + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xEnd  [expr  ( ([expr $minThreshField-$dataOrigin]  + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    $w.threshGraph create rectangle  $xStart [expr $threshPlotBottom+1] $xEnd [expr $threshPlotBottom +4] -fill "\#991010" -outline "" -tags [list xScalable threshPlotBar]

     #mark max of the thresh
    set xStart  [expr  ( ( $maxThreshField-$dataOrigin + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xEnd  [expr  ( ( $maxThreshField-$dataOrigin + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    $w.threshGraph create rectangle  $xStart [expr $threshPlotTop-1] $xEnd [expr $threshPlotTop-4] -fill "\#109910" -outline "" -tags [list xScalable threshPlotBar]

  #graph thresholds in $w.threshGraph
   
  #now show size of graph
  if {$prevCursorFrame($currentMol)!="null"} then { 
    updateThreshLabel $prevCursorFrame($currentMol)
  }  
}


proc ::timeline::calcSaltBridge {lastCalcVal} { 

  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax 
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable ONdist
  variable partSelText
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame


  clearData
  set usesFreeSelection 1

  set lastCalc $lastCalcVal

  progressBox 0

 if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      return 
  }

  #for progressBox progress
  set frameTotal [expr $lastAnalysisFrame - $firstAnalysisFrame +1]
  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
  set calcCount 0


  set listOfFrames ""
  set acsel [atomselect $currentMol "$partSelText and (protein and acidic and oxygen and not backbone)"]
  set bassel [atomselect $currentMol "$partSelText and (protein and basic and nitrogen and not backbone)"]
  for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
    set dataName(vals) "salt bridge"
    $acsel frame $trajFrame
    $bassel frame $trajFrame
    lappend listOfFrames [measure contacts $ONdist $acsel $bassel]
    incr calcCount
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 0.5  $progressCalcFraction  $calcCount $frameTotal] 
    } 
     
    if {$progressCancelPressed} break
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }


  #XXX hard set here, elsewhere should be actual value
  set groupValue 1

  set frame 0
  set frameListTotal [llength $listOfFrames]
  foreach f $listOfFrames {
    set frameList($frame) ""
    tlPutsDebug ": At top, frame= $frame   frameList($frame)=$frameList($frame)"
    #next line isn't loop, just a single assignment
    foreach {oxlist nitlist} $f {
      set selString ""
      tlPutsDebug "ox= $oxlist nitlist= $nitlist"
      foreach  o $oxlist n $nitlist  {
        set selString "index $o $n"
        if {$selString != ""} {
          lappend frameList($frame) $selString      
          #no value assoc'd with each entry here, if present, assinged val =1

          #tlPutsDebug "DEBUG: frame= $frame   frameList($frame)=$frameList($frame)"

          #now count how many in each 
          #now go through current frames groups of three
          set spaceToCodeN [string map {" " %20} $selString]
          tlPutsDebug ": selString= $selString spaceToCodeN= $spaceToCodeN  frame=$frame"
          set seenData($spaceToCodeN,$frame) $groupValue
          if {[info exists seenCount($spaceToCodeN)]} {
            incr  seenCount($spaceToCodeN)
          } else {
            set seenCount($spaceToCodeN) 1
            #just to be on the safe side...
            set seenDataValGroup($spaceToCodeN) "null"

          }

        }
      }
    }
    incr frame
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0.5 0.7  $progressCalcFraction  $frame $frameListTotal] 
    }
    if {$progressCancelPressed} break 
    incr calcCount
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  } 
  #tlPutsDebug ": all names = [array names seenCount]"

  #here the cutoff for being displayed is: 1
  #can set higher cutoff later
  set numDisplayGroups [llength [array names seenCount]  ]
  set dataValNum [expr $numDisplayGroups -1]

  #clear data and set labels
  #clear out all frames, in real case set all data.
  # following line sets number of displayed groups to number of groups that have been seen, that 
  # is, showed data that met conditions
  set displayGroupTextList [array names seenCount]  
  # there are displayGroup+1 lines of data in the display (equivalent of residues)
  set displayGroup 0
  set dGTlength [llength $displayGroupTextList]
  foreach displayGroupText $displayGroupTextList {
    tlPutsDebug "SaltBridge: displayGroup= $displayGroup  displayGroupText= $displayGroupText"
    set codeToSpaceN [string map {%20 " "} $displayGroupText]
    #regexp "^\\D+ (.*$)" $codeToSpaceN matchall regout1
    #set dataVal(freeSelLabel,$displayGroup) $regout1
    regexp "^index (\\d+) (\\d+)$" $codeToSpaceN matchall regout1 regout2
    set selox [atomselect $currentMol "index $regout1"]
    set selnit [atomselect $currentMol "index $regout2"]
    set dataVal(freeSelLabel,$displayGroup) "[$selox get resname][$selox get resid]--[$selnit get resname][$selnit get resid]"
    set dataVal(freeSelString,$displayGroup) "same residue as ($codeToSpaceN)"
    #tlPutsDebug "selox = $selox dataVal(freeSelString,$displayGroup= $dataVal(freeSelString,$displayGroup)  dataVal(freeSelString,$displayGroup)    dataVal(freeSelString,$displayGroup= $dataVal(freeSelString,$displayGroup)    codeToSpaceN= $codeToSpaceN"
  

    #set the dataVal displayGroup that corresponds to displayGroupText, will be used when writing
    set seenDataValGroup($displayGroupText) $displayGroup
    for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      set dataVal($curField,$displayGroup) 0 
      # XXX shouldn't we actually be setting min/max in next two lines?
    }
    incr displayGroup
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac .7 .9 $progressCalcFraction  $displayGroup $dGTlength] 
    }
    if {$progressCancelPressed} break 
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }

#use next line if really extracting data from traj geom.
  #$sel frame $trajFrame
  #set data (only the rare frames that have data)
  #clear out all frames, in real case set all data.
  #for {set displayGroup 0} {$displayGroup<numDisplayGroups} {incr displayGroup} 
  #first set labels
  set dataItems 0
  set displayGroupDataList [array names seenData]  
  set dGDlength [llength $displayGroupDataList]
  foreach d $displayGroupDataList {
    foreach {itemDisplayGroupText itemFrame} [split $d ","] {
    #tlPutsDebug ": d= $d   itemDisplayGroupText=$itemDisplayGroupText  itemFrame= $itemFrame"
      #turn the name back into a label and a time
      #take the number after the final comma
      set curField [expr $dataOrigin + $itemFrame]
      set displayGroup $seenDataValGroup($itemDisplayGroupText)
      set dataVal($curField,$displayGroup) $seenData($d)
       # tcl string trick: $seenData($d) should be equivalent of $seenData($itemDisplayGroupText,$itemFrame)
      #XXXX swap seenData item-frame order, for consistenecy
      incr dataItems 
    }
    
  initPicked 
  if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac .9 1.0 $progressCalcFraction  $dataItems $dGDlength ] 
  }
  if {$progressCancelPressed} break 
#tlPutsDebug " displayGroup= $displayGroup dataItems= $dataItems"
  #XXX the zero-base for a var named like "zzzzNum" is confusing.  Should set 
  #all things that refer to n objects have a value of n, not (n-1).
  } 
  if {$progressCancelPressed} {
    tlPutsDebug "progreeCancelPressed" 
    set progressCancelPressed 0
    clearData
  }
set dataMin(all) 0
set dataMax(all) 1
}

proc ::timeline::calcHbonds {lastCalcVal} { 

  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax 
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  set usesFreeSelection 1
  variable hbondDistCutoff
  variable hbondAngleCutoff 
  variable partSelText
  variable hbondSel1
  variable hbondSel2
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame


  tlPutsDebug "starting calcHbonds"
  #clearData
  set lastCalc $lastCalcVal
  progressBox 0
  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "Timeline: select molecule before choosing Calculate method"
      return 
  }


#for progressBox progress
  set frameTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
  set calcCount 0

  set listOfFrames ""
  if {$hbondSel1 != ""} {
    set sel  [atomselect $currentMol "$hbondSel1"] 
  } else { 
  set sel  [atomselect $currentMol "$partSelText and (protein or nucleic)"] 
  }

  if {$hbondSel2 != ""} {
    set sel2  [atomselect $currentMol "$hbondSel2"] 
  }
   
   for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
    set dataName(vals) "H-bond"
    $sel frame $trajFrame
    if {$hbondSel2 != ""} {
      $sel2 frame $trajFrame
      lappend listOfFrames [measure hbonds $hbondDistCutoff  $hbondAngleCutoff  $sel $sel2 ]
    } else {
      lappend listOfFrames [measure hbonds $hbondDistCutoff  $hbondAngleCutoff  $sel ]
    } 

    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0 0.5  $progressCalcFraction  $calcCount $frameTotal] 
    } 
    
    if {$progressCancelPressed} break 
    incr calcCount
  }
  
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }
  #XXX hard set here, elsewhere should be actual value
  set groupValue 1

  set frame $firstAnalysisFrame
  set frameCount [llength $listOfFrames]
  foreach f $listOfFrames {
    set frameList($frame) ""
    tlPutsDebug ": At top, frame= $frame   frameList($frame)=$frameList($frame)"
    #next line isn't loop, just a single assignment
    foreach {donors acceptors hydrogens} $f {
      set selString ""
      #tlPutsDebug ": donors= $donors  acceptors= $acceptors hydrogens=$hydrogens"
      foreach d $donors a $acceptors h $hydrogens {
        set selString "index $d $a $h"
        if {$selString != ""} {
          lappend frameList($frame) $selString      
          #no value assoc'd with each entry here, if present, assinged val =1

          #tlPutsDebug "DEBUG: frame= $frame   frameList($frame)=$frameList($frame)"

          #now count how many in each 
          #now go through current frames groups of three
          set spaceToCodeN [string map {" " %20} $selString]
          #tlPutsDebug ": selString= $selString spaceToCodeN= $spaceToCodeN  frame=$frame"
          set seenData($spaceToCodeN,$frame) $groupValue
          if {[info exists seenCount($spaceToCodeN)]} {
            incr  seenCount($spaceToCodeN)
          } else {
            set seenCount($spaceToCodeN) 1
            #just to be on the safe side...
            set seenDataValGroup($spaceToCodeN) "null"

          }

        }
      }
    }
    incr frame
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac 0.5 0.7 $progressCalcFraction [expr $frame - $firstAnalysisFrame] $frameCount ] 
    }
    if {$progressCancelPressed} break 
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }
  #tlPutsDebug ": all names = [array names seenCount]"

  #here the cutoff for being displayed is: 1
  #can set higher cutoff later
  set numDisplayGroups [llength [array names seenCount]  ]
  set dataValNum [expr $numDisplayGroups -1]
  tlPutsDebug "calcHbonds: set dataValNum to $dataValNum"

  #clear data and set labels
  #clear out all frames, in real case set all data.
  # following line sets number of displayed groups to number of groups that have been seen, that 
  # is, showed data that met conditions
  set displayGroupTextList [array names seenCount]  
  # there are displayGroup+1 lines of data in the display (equivalent of residues)
  set displayGroup 0

  set dGTlength [llength  $displayGroupTextList]
  set dGTcounter 0
  foreach displayGroupText $displayGroupTextList {
    tlPutsDebug ": displayGroup= $displayGroup  displayGroupText= $displayGroupText"
    set codeToSpaceN [string map {%20 " "} $displayGroupText]
    regexp "^\\D+ (.*$)" $codeToSpaceN matchall regout1
    set dataVal(freeSelLabel,$displayGroup) $regout1
    set dataVal(freeSelString,$displayGroup) $codeToSpaceN
    #set the dataVal displayGroup that corresponds to displayGroupText, will be used when writing
    set seenDataValGroup($displayGroupText) $displayGroup
    for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      set dataVal($curField,$displayGroup) 0 
      # XXX shouldn't we actually be setting min/max in next two lines?
    }
    incr displayGroup
    incr calcCount
 
    #XX currently no percentage update.  Divide whole scheme into rough fractions to make estimate.
    if {$progressBoxExists} {
        progressBoxConfigure  [progressFindFrac .7  0.85 $progressCalcFraction $dGTcounter $dGTlength] 
    }
    if {$progressCancelPressed} break 
    incr dGTcounter 
  }
  if {$progressCancelPressed} {
    set progressCancelPressed 0
    clearData
  }
  #use next line if really extracting data from traj geom.
  #$sel frame $trajFrame
  #set data (only the rare frames that have data)
  #clear out all frames, in real case set all data.
  #for {set displayGroup 0} {$displayGroup<numDisplayGroups} {incr displayGroup} 
  #first set labels
  set dataItems 0
  set displayGroupDataList [array names seenData]  
  set dGDLength [llength $$displayGroupDataList]
  foreach d $displayGroupDataList {
    foreach {itemDisplayGroupText itemFrame} [split $d ","] {
    #tlPutsDebug ": d= $d   itemDisplayGroupText=$itemDisplayGroupText  itemFrame= $itemFrame"
      #turn the name back into a label and a time
      #take the number after the final comma
      set curField [expr $dataOrigin + $itemFrame]
      set displayGroup $seenDataValGroup($itemDisplayGroupText)
      set dataVal($curField,$displayGroup) $seenData($d)
       # tcl string trick: $seenData($d) should be equivalent of $seenData($itemDisplayGroupText,$itemFrame)
      #XX swap seenData item-frame order, for consistenecy
      incr dataItems 
    }
    
    initPicked  
    #tlPutsDebug " displayGroup= $displayGroup dataItems= $dataItems"
    #XX the zero-base for a var named like "zzzzNum" is confusing.  Should set 
    #all things that refer to n objects have a value of n, not (n-1).

    #XX currently no percentage update.  Divide into rough fractions to make estimate
    if {$progressBoxExists} {

        progressBoxConfigure  [progressFindFrac 0.85  1.0 $progressCalcFraction $dataItems $dGDLength] 
    }
    if {$progressCancelPressed} break 
  } 
  if {$progressCancelPressed} {
    set progressCancelPressed 0
    clearData
  }
set dataMin(all) 0
set dataMax(all) 1
tlPutsDebug "end of calcHbonds: dataValNum=  $dataValNum"
}

proc ::timeline::clearData {} {
  variable w
  variable dataVal
  variable dataValNum
  variable dataOrigin
  variable numDataFrames
  variable everRedrawn
  variable usableMolLoaded
  variable rectCreated
  variable lastCalc
  variable dataMin
  variable dataMax
  variable dataThreshVal
  set dataMin(all) "null"
  set dataMax(all) "null" 
  #XX should be null, but not set to use correctly

  set lastCalc 0
  tlPutsDebug "Clearing 2D data..."
  set endStructs [expr $dataOrigin + $numDataFrames - 1 ]
  for {set field $dataOrigin} {$field <= $endStructs} {incr field} {
    set dataThreshVal($field) "null"
    for {set i 0} {$i<=$dataValNum} {incr i} {

      set  dataVal($field,$i) "null"
      # for the special struct case, the 0 shold give default color
      #puts "dataVal($field,$i) is now $dataVal($field,$i)"
      #set resid $dataVal(resid,$i)
      #set chain $dataVal(chain,$i)
      #set frame [expr $field - $dataOrigin]
      #puts $writeDataFile "$resid $chain CA $frame $val"
      
    }
  }

  #XXX clear vert amd horz scales in case redraw does not do it (lack of data)
   
  $w.vertScale delete vertScaleText 
  $w.horzScale delete horzScaleText 
  #redraw the data rects
  showall 1
  set dataValNum -1
  return
}
proc  ::timeline::userScaleBothChanged {val} {
  variable userScalex
  variable userScaley
  variable userScaleBoth
  variable scaley
  variable fit_scalex
  variable fit_scaley
  variable scalex
  variable everRedrawn
  set scalex [expr $userScaleBoth * $fit_scalex]
  set scaley [expr $userScaleBoth * $fit_scaley]
  set userScalex  $userScaleBoth
  set userScaley $userScaleBoth
  if {$everRedrawn} {
    redraw name func op
  }
  #puts "redrawn, userScaleBoth= $userScaleBoth, scalex= $scalex, userScalex= $userScalex, scaley= $scaley, userScaley= $userScaley"
  return
}



proc  ::timeline::userScalexChanged {val} {
  variable userScalex
  variable scalex
  variable fit_scalex
  variable everRedrawn
  set scalex [expr $userScalex * $fit_scalex]
  if {$everRedrawn} {
    redraw name func op
  }
  return
}


proc ::timeline::userScaleyChanged {val} {
  variable userScaley
  variable scaley
  variable fit_scaley
  variable everRedrawn
  #until working ok, still do direct mapping
  set scaley [expr $userScaley * $fit_scaley]
  #set scaley $userScaley 
  if {$everRedrawn} {
    redraw name func op
    tlPutsDebug "userScaleyChanged: redrawn, fit_scaley= $fit_scaley   scaley= $scaley   userScaley= $userScaley"
  }
  return
}

proc ::timeline::drawVertScale {} {
  variable w
  variable ytopmargin
  variable scaley
  variable ybox
  variable dataValNum
  variable dataVal
  variable vertTextSkip
  variable vertTextRight
  variable resCodeShowOneLetter
  variable monoFont
  variable usesFreeSelection
  $w.vertScale delete vertScaleText 

  if {$dataValNum >= 0} { 
    #when adding new column, add to this list (maybe adjustable later)
    #The picked fields 
    
    #Add the text...
    set field 0           

    #note that the column will be 0, but the data will be from picked
    
    
    set yDataEnd [expr $ytopmargin + ($scaley * $ybox * ($dataValNum +1))]
    set y 0.0

    set yposPrev  -10000.0

    #Add the text to vertScale...
    set field 0            



    #we want text to appear in center of the dataRect we are labeling
    set vertOffset [expr $scaley * $ybox / 2.0]

    #don't do $dataValNum, its done at end, to ensure always print last 
    for {set i 0} {$i <= $dataValNum} {incr i} {
      set ypos [expr $ytopmargin + ($scaley * $y) + $vertOffset]
      if { ( ($ypos - $yposPrev) >= $vertTextSkip) && ( ( $i == $dataValNum) || ( ($yDataEnd - $ypos) > $vertTextSkip) ) } {
        #tlPutsDebug "ypos= $ypos yposPrev= $yposPrev i= $i dataValNum= $dataValNum yDataEnd= $yDataEnd vertTextSkip= $vertTextSkip vertTextRight= $vertTextRight vertOffset= $vertOffset"
        if {$usesFreeSelection} {
          set startLabel [string range $dataVal(freeSelLabel,$i) 0 13]
          $w.vertScale create text $vertTextRight $ypos -text $startLabel -width 200 -font $monoFont -justify right -anchor e -tags vertScaleText 
         } else {
          if {$resCodeShowOneLetter == 0} {
            set res_string $dataVal(resname,$i)
          } else {
            set res_string $dataVal(rescode,$i)
          }
         #for speed, we use vertScaleText instead of $dataName($field)
         #how to deal with chain vs. segname?  For now, don't show segname.  Should allow toggle?
        $w.vertScale create text $vertTextRight $ypos -text "$dataVal(resid,$i) $res_string $dataVal(chain,$i)" -width 200 -font $monoFont -justify right -anchor e -tags vertScaleText 
        }
        set yposPrev  $ypos
      }        
      set y [expr $y + $vertTextSkip]
      
    } 
    
  } 
}


proc ::timeline::drawHorzScale {} {
  variable w
  variable ytopmargin
  variable scalex
  variable dataValNum
  variable dataVal
  variable monoFont
  variable dataOrigin
  variable xcol
  variable numDataFrames    
  variable dataWidth

  $w.horzScale delete horzScaleText 

  
  #when adding new column, add to this list (maybe adjustable later)
  #The picked fields 
  
  #Add the text...

  #note that the column will be 0, but the data will be from picked
  
  #we want text to appear in center of the dataRect we are labeling
  set fieldLast [expr $dataOrigin + $numDataFrames - 1]
  #ensure minimal horizontal spacing
  # hardcoded spacing
  set horzSpacing 27 
  set horzPad 5
  set horzSpacingPad [expr $horzSpacing + $horzPad]
  set horzDataTextSkip [expr $dataWidth]
  set scaledHorzDataTextSkip [expr $scalex * $dataWidth]
  set scaledHorzDataOffset [expr $scalex * $dataWidth / 2.0]
  set ypos 20 
  set xStart [expr ($xcol($dataOrigin))]
  set xDataEnd  [expr int ($xStart +  $scalex * ($dataWidth * $numDataFrames ) ) ] 
  set x 0 

  #numbers are scaled for 1.0 until xpos
  #this is tied to data fields, which is produced from frames upon
  #first drawing. Should really agreee with writeDataFile, which currently uses frames, not fields
  #
  # xPos is horz center of labeled frame, the frameNum is displayed centered on it
  set xposPrev -1000 
  set xposRightPrev -1000 
  #traj data starts at dataOrigin
  for {set frameNum 0} {$frameNum < $numDataFrames} {incr frameNum} {
    set field [expr $frameNum + $dataOrigin]
    set textWidth [font measure $monoFont -displayof $w $frameNum] 
    set textWidthPad [expr $textWidth +$horzPad]
  ####for {set field [expr $dataOrigin]} {$field <= $fieldLast} {incr field} {}
  ####  set frameNum [expr $field - $dataOrigin -1]
    
    set xpos [expr int ($xStart + ($scalex * $x) + $scaledHorzDataOffset)]
    set xposRight [expr $xpos +int($textWidth/2)]
    if { ( ($xposRight - $xposRightPrev  ) >= $textWidthPad) && ( ( $field == $fieldLast) || ( ( $xDataEnd - $xpos) > ( 2 * $textWidth) ) ) } {
      # draw the frame number if there is room
      #for speed, we use horzScaleText instead of $dataName($field)
        #tlPutsDebug "frameNum= $frameNum, xpos= $xpos  xposPrev= $xposPrev xposRight= $xposRight  xposRightPrev= $xposRightPrev textWidth= $textWidth  textWidthPad= $textWidthPad"
        #$w.horzScale create text $xpos $ypos -text "$frameNum" -width $horzSpacingPad -font $monoFont -justify center -anchor s -tags horzScaleText 
        if {( $field == $fieldLast)} { 
          $w.horzScale create text [expr $xpos - $textWidth/2] $ypos -text "$frameNum"  -font $monoFont -justify center -anchor s -tags horzScaleText 
        } {
          $w.horzScale create text $xpos $ypos -text "$frameNum"  -font $monoFont -justify center -anchor s -tags horzScaleText 
        }
        set xposPrev $xpos
        set xposRightPrev $xposRight
    }        
    set x [expr $x + $horzDataTextSkip]
  } 

  
}

#puts "--DEBUG--:Timeline: Completed defining drawHorzScale"

#############################################
# end of the proc definitions
############################################





####################################################
# Execution starts here. 
####################################################

#####################################################
# set traces and some binidngs, then call timeLineMain
#####################################################
proc ::timeline::startTimeline {} {
  
  ####################################################
  # Create the window, in withdrawn form,
  # when script is sourced (at VMD startup)
  ####################################################
  variable w .vmd_timeline_Window
  variable wp .vmd_timeline_printing_Window
  set windowError 0
  set printWindowError 0
  set errMsg ""

  #if timeline has already been started, just deiconify window
  if { [winfo exists $w] } {
    deiconify $w 
    return
  }

  if { [catch {toplevel $w -visual truecolor} errMsg] } {
    puts "Info) Timeline window can't find trucolor visual, will use default visual.\nInfo)   (Error reported was: $errMsg)" 
    if { [catch {toplevel $w } errMsg ]} {
      puts "Info) Default visual failed, Timeline window cannot be created. \nInfo)   (Error reported was: $errMsg)"    
      set windowError 1
    }
  }


  if {$windowError == 0} { 
    #don't withdraw, not under vmd menu control during testing
    #wm withdraw $w
    wm title $w "VMD Timeline"
    #wm resizable $w 0 0 
    wm resizable $w 1 1 

    variable w
    variable monoFont
    variable initializedVars 0
    variable needsDataUpdate 0 

    #overkill for debugging, should only need to delete once....
    trace vdelete currentMol w [namespace code molChoose]
    trace vdelete currentMol w [namespace code molChoose]
    trace vdelete ::vmd_pick_atom w  [namespace code listPick] 
    trace vdelete ::vmd_pick_atom w  [namespace code listPick] 
    trace vdelete ::vmd_initialize_structure w  [namespace code molChooseMenu]
    trace vdelete ::vmd_initialize_structure w  [namespace code molChooseMenu]


    bind $w <Map> "+[namespace code Show]"
    bind $w <Unmap> "+[namespace code Hide]"
    #specify monospaced font, 12 pixels wide
    font create tkFixedTimeline -family Courier -size -12
    #for test run tkFixedTimeline was made by normal sequence window
    #change this so plugins don't depend on eachOther:1
    set monoFont tkFixedTimeline

    #call to set up, after this, all is driven by trace and bind callbacks
    timeLineMain
  }
  return $w
}

#example analysis procedures
proc ::myCountContacts {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [llength [lindex [measure contacts 4.0 $resCompleteSel $proteinNucSel] 0]]
}

proc ::myTimelineTestSasa {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [lindex [measure contacts 4.0 $resCompleteSel $proteinNucSel] 0]]
}

proc ::myTimelineTestResX {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [$resAtomSel get x]
}
proc timeline::myResPhi {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [$resAtomSel get phi]
}
proc ::myCountContacts {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [llength [lindex [measure contacts 4.0 $resCompleteSel $proteinNucSel] 0]]
}

proc timeline::printColScale {scaleOrigX scaleOrigY} {
  variable wp
  variable dataName
  variable dataOrigin
  variable monoFont
  variable trajMin
  variable trajMax
 
  set scaleWidth 90      

  #local hard coding for current placement, later should make this visible externally
  set xPos [expr $scaleOrigX ] 
  set yPos [expr $scaleOrigY ]
  set valsYPos [expr $scaleOrigY + 36]
  set barTop [expr $scaleOrigY + 18]
  set barBottom [expr $scaleOrigY + 33]

  #abandon if data is not available
  if {[catch {set scaleTitle $dataName(vals)}]} {
   set scaleTitle "--"
   return
  } else {
    if {$scaleTitle=="struct"} {
      set scaleTitle "sec. struct."
    }
 }
      #print the the title in center of data rectangle width
      $wp.large create text [expr int($xPos+ ( $scaleWidth/ 2.0) )] $yPos -text "$scaleTitle" -width 200 -justify center -anchor n  -tags printout 
    
      #make a scale across data rectange width
      
      set size $scaleWidth 
      set frac [expr $size / 256.0] 
      if {$dataName(vals) != "struct"} {
        set minString [format "%.3g" $trajMin]
        set maxString [format "%.4g" $trajMax]
        $wp.large create text [expr $xPos - 2  ] $valsYPos -text $minString -width 50 -font $monoFont -justify center -anchor nw -tags printout 
        $wp.large create text [expr int ($xPos + $scaleWidth +2 )] $valsYPos -text $maxString -width 50 -font $monoFont -justify center -anchor ne -tags printout 
        
        set range [expr $trajMax - $trajMin]
        #bounds check, should really print error message

       
        for {set val 0} {$val< 256} {incr val} {
            
          #draw linear scale
          set hexcols [chooseColor $val]            
          
          set hexred [lindex $hexcols 0]
          set hexgreen [lindex $hexcols 1]
          set hexblue [lindex $hexcols 2]
          
          
         $wp.large create rectangle [expr 0.0 + $xPos + ($val * $frac)] $barTop [expr 0.0+ $xPos + (($val+1.0)*$frac)] $barBottom -fill  "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags printout
        }
    } else {
        set prevNameIndex -1
        for {set val 0} {$val < 256} {incr val} {
        set names [list T E B H G I C "other"]
        
        set nameIndex [expr int ([expr [llength $names] -1]  * ($val+0.0)/256)]
        set curName [lindex $names  $nameIndex]
        
        if {$nameIndex != $prevNameIndex} {
            #set line to black
            set hexred 0
            set hexgreen 0
            set hexblue 0
            
            #draw text
            $wp.large create text [expr $xPos + ($val * $frac)+ 3] $valsYPos  -text $curName -width 20 -font $monoFont -justify left -anchor nw -tags printout 

        } else {
            set hexcols [chooseColor $curName]
            
            set hexred [lindex $hexcols 0]
            set hexgreen [lindex $hexcols 1]
            set hexblue [lindex $hexcols 2]
        }

     $wp.large create rectangle [expr 0.0 + $xPos + ($val * $frac)] $barTop [expr 0.0 + $xPos + (($val+1.0) * $frac)] $barBottom -fill  "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags printout

          set prevNameIndex $nameIndex
      }
    set hexred 0
    set hexgreen 0
    set hexblue 0
     $wp.large create rectangle [expr 0.0 + $xPos + ($val * $frac)] $barTop [expr 0.0 + $xPos + (($val+1.0) * $frac)] $barBottom -fill  "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags printout
  }
}

proc timeline::drawColScale {} {
  variable w
  variable dataName
  variable dataOrigin
  variable monoFont
  variable trajMin
  variable trajMax
 
  set scaleWidth 90      

  #local hard coding for current placement, later should make this visible externally
  set xPos 7 
  set yPos 3
  set valsYPos 36
  set barTop 18
  set barBottom 33

  $w.colscale delete colorscalebar
  #abandon if data is not available
  if {[catch {set scaleTitle $dataName(vals)}]} {
   set scaleTitle "--"
   return
  } else {
    if {$scaleTitle=="struct"} {
      set scaleTitle "sec. struct."
    }
 }
      #print the the title in center of data rectangle width
      $w.colscale create text [expr int($xPos+ ( $scaleWidth/ 2.0) )] $yPos -text "$scaleTitle" -width 200 -font $monoFont -justify center -anchor n  -tags colorscalebar
    
      #make a scale across data rectange width
      
      set size $scaleWidth 
      if {$dataName(vals) != "struct"} {
        set minString [format "%.3g" $trajMin]
        set maxString [format "%.4g" $trajMax]
        
        $w.colscale create text [expr $xPos - 2  ] $valsYPos -text $minString -width 50 -font $monoFont -justify center -anchor nw -tags colorscalebar
        $w.colscale create text [expr int ($xPos + $scaleWidth +2 )] $valsYPos -text $maxString -width 50 -font $monoFont -justify center -anchor ne -tags colorscalebar
        
        set range [expr $trajMax - $trajMin]
        #bounds check, should really print error message
        
        for {set yrect 0} {$yrect < $size} {incr yrect} {
            
          #draw linear scale
          set val [expr ( ( 0.0+ $yrect  )/ ($size -1)) * 255]
          set hexcols [chooseColor $val]            
          
          set hexred [lindex $hexcols 0]
          set hexgreen [lindex $hexcols 1]
          set hexblue [lindex $hexcols 2]
          

         $w.colscale create rectangle [expr $xPos + $yrect] $barTop [expr $xPos + $yrect] $barBottom -fill  "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags colorscalebar
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
            $w.colscale create text [expr int ($xPos + $yrect+ 3)] $valsYPos  -text $curName -width 20 -font $monoFont -justify left -anchor nw -tags colorscalebar

        } else {
            set hexcols [chooseColor $curName]
            
            set hexred [lindex $hexcols 0]
            set hexgreen [lindex $hexcols 1]
            set hexblue [lindex $hexcols 2]
        }

      $w.colscale create rectangle [expr $xPos + $yrect] $barTop [expr $xPos + $yrect] $barBottom -fill  "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags colorscalebar

          set prevNameIndex $nameIndex
      }
    set hexred 0
    set hexgreen 0
    set hexblue 0
    $w.colscale create rectangle [expr $xPos + $yrect] $barBottom [expr $xPos + $size] $barTop -fill  "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags colorscalebar
  }
}


proc ::timeline::printPicRedraw {} {
  variable clicked
  variable thresholdBoundMin
  variable thresholdBoundMax
  variable usesFreeSelection
  variable resCodeShowOneLetter
  variable x1 
  variable y1 
  variable so
  variable w
  variable wp 
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
  variable dataVal 
  variable dataValNum 
  variable dataOrigin 
  variable dataName 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable vertTextRight
  variable vertHighLeft
  variable vertHighRight
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin
  variable dataMax 
  variable trajMin
  variable trajMax
  variable xPosScaleVal
  variable everRedrawn
  variable usableMolLoaded
  variable rectCreated
  variable prevScalex
  variable prevScaley
  variable numDataFrames
  variable dataThreshVal
  variable currentMol 
  variable nullMolString
  variable prevCursorObject
  variable prevCursorFrame
  variable cursorRepColor
  variable cursorRep
  variable cursorShown

  tlPutsDebug "starting printPicRedraw"


  if { ($usableMolLoaded) && ($dataValNum >=0 )  && ($currentMol != $nullMolString)  && ($numDataFrames >= 1)} {
  #$wp.large delete all
  #local constants, eventually place in startup 
    catch {destroy $wp}
   
    if { [catch {toplevel $wp -visual truecolor} errMsg] } {
      puts "Info) Timeline printout buffer window can't find trucolor visual, will use default visual.\nInfo)   (Error reported was: $errMsg)" 
      if { [catch {toplevel $wp } errMsg ]} {
        puts "Info) Default visual failed, Timeline printout buffer window cannot be created. \nInfo)   (Error reported was: $errMsg)"    
        set printWindowError 1
        return
      }
    }
   
    wm title $wp "VMD Timeline (print preview)"
    wm protocol $wp WM_DELETE_WINDOW {set ::timeline::clicked -1}
    wm minsize  $wp 956 685 
    wm maxsize  $wp 956 685

    #canvas $wp.large -width 990 -height 700 -background #DDFFFF 
    canvas $wp.large -width 956 -height 685 -background #DDFFFF 
    pack $wp.large -in $wp -side left -padx 2 -expand yes -fill both 
   
    # print layout: constants, calculated values
    set printDataPosX 245
    set printDataPosY 5


    set selInfoWidthPrint 100 
    set vertTextLeftPrint [expr 0 + $selInfoWidthPrint + 15] 
    set vertTextRightPrint [expr $printDataPosX - 10]
    set threshGraphHeightPrint 40
    set selInfoYPos 300 
    set topData $printDataPosY
    set botData [expr $printDataPosY + $ycanwindowmax]
    set yposHorzScale [expr $botData + 21]
    set threshPlotTopPrint  [expr $yposHorzScale + 30 ]
    set horzScaleTitleYpos  [expr $yposHorzScale + 5 ]
    set leftData $printDataPosX
    set rightData [expr $printDataPosX + $xcanwindowmax]
    set vertTextWidthPrint  [expr $vertTextRightPrint - $vertTextLeftPrint]

    set colScalePosX [expr ($selInfoWidthPrint - 90) / 2 ]
    set colScalePosY [expr $botData - 28]


    #Limits in drawn canvas
    set leftCan  [expr $xcanmax(data) * [lindex [$w.can xview] 0]]
    set rightCan  [expr $xcanmax(data) * [lindex [$w.can xview] 1]]
    set topCan [expr $ycanmax(data) * [lindex [$w.can yview] 0]] 
    set botCan [expr $ycanmax(data) * [lindex [$w.can yview] 1]] 

     
    #use the text in the GUI window for printing info text
    set infoText "Highlight details:\n[$w.selInfo cget -text]\n\n Threshold:\n[$w.threshValLab cget -text]"
    if {$dataName(vals) != "struct"} {
    set infoText "$infoText\n$thresholdBoundMin to $thresholdBoundMax"
    }
    $wp.large create text [expr $selInfoWidthPrint/2] $selInfoYPos -text $infoText -justify center -width $selInfoWidthPrint -anchor n -tags printout]

    #title for horz axis
    $wp.large create text [expr (($leftData+$rightData) / 2)] $horzScaleTitleYpos -text "(frame number)" -font monofont -justify center -width [expr $xcanwindowmax] -anchor n -tags printout]
    if {$usesFreeSelection} {
      set vertScaleTitle "(selection name)"
    } else {
      set vertScaleTitle "(resid,\nresname,\nchain)" 
    }
    $wp.large create text [expr $vertTextLeftPrint - 3] 60 -text $vertScaleTitle -font monofont -justify center -width $selInfoWidthPrint -anchor e -tags printout]

    #threshold count
      

    # we print to a fixed size canvas, the dataRects will use the current scaling
    # be careful not to set any vars that are used only by screen display 
    # to capture cutoff, we will need to know where data should start and end

    ##set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValNum + 1) )]  

    ##set xsize [expr  $xcol($dataOrigin) +  ($scalex *  $dataWidth *  $numDataFrames)  ] 

    ##set ycanmax(data) $ysize
    ##set ycanmax(vert) $ycanmax(data)
    ##set xcanmax(data) $xsize
    ##set xcanmax(horz) $xcanmax(data)
    ##if {$ycanmax(data) < $ycanwindowmax} {
    ##  set ycanmax(data) $ycanwindowmax
   ## }


    ##if {$xcanmax(data) < $xcanwindowmax} {
    ##  set xcanmax(data) $xcanwindowmax
    ##}

    ##$w.can configure -scrollregion "0 0 $xcanmax(data) $ycanmax(data)"
    ##$w.vertScale configure -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)"
    ##$w.horzScale configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    ##$w.threshGraph configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"

    set fieldLast [expr $dataOrigin + $numDataFrames -1 ]
    

    if {$usableMolLoaded && $everRedrawn && ($currentMol != $nullMolString) &&  ($dataValNum >=0)} {
      thresholdData
      #### start of printThresholdGraph
      set threshPlotBottom [expr ($threshPlotTopPrint -4) + $threshGraphHeightPrint - 5] 
      set lastField [expr $dataOrigin + $numDataFrames - 1]
      set minThresh $dataThreshVal($dataOrigin)
      set minThreshField $dataOrigin
      set maxThresh $dataThreshVal($dataOrigin)
      set maxThreshField $dataOrigin
      for {set field [expr $dataOrigin+1]} {$field<=$lastField} {incr field} {
        if {$dataThreshVal($field) < $minThresh} {set minThresh $dataThreshVal($field); set minThreshField $field} 
        if {$dataThreshVal($field) > $maxThresh} {set maxThresh $dataThreshVal($field); set maxThreshField $field} 
      }
      if {$maxThresh == 0} {set depictedMaxThresh 10} else {set depictedMaxThresh $maxThresh}
      set plotFactor [expr  ($threshPlotBottom-$threshPlotTopPrint)/(0.0+$depictedMaxThresh) ]
       #count will be 0-based
       #later can do min based

      set endField [expr $dataOrigin + $numDataFrames - 1 ]

      for {set field $dataOrigin} {$field <= $endField} {incr field} {
        set frame [expr $field - $dataOrigin]
        set intermed [expr $plotFactor * $dataThreshVal($field)]
        set plotY [expr $threshPlotBottom - ($plotFactor * $dataThreshVal($field))]
        #puts "val= $intermed  dataThreshVal($field)= $dataThreshVal($field)  plotY=$plotY, field= $field threshPlotBottom=$threshPlotBottom "
        set xStart  [expr  ( ($frame + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        set xEnd  [expr  ( ($frame + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        set xStartShifted [expr $xStart - $leftCan + $printDataPosX]
        set xEndShifted [expr $xEnd - $leftCan + $printDataPosX]
        #tlPutsDebug "xEndShifted= $xEndShifted leftData= $leftData  xStartShifted= $xStartShifted rightData= $rightData"
        if {! (($xEndShifted  < $leftData) ||  ($xStartShifted > $rightData))} {
          if {$xStartShifted < $leftData} {
            set xStartShifted $leftData
          }
          if {$xEndShifted >  $rightData} {
            set xEndShifted $rightData
          }
          $wp.large create rectangle  $xStartShifted $threshPlotBottom $xEndShifted $plotY -fill "\#EE7070"  -tags printout 
        #puts "plotted   $xStart $threshPlotBottom $xEnd $plotY"
        }  
      }
      $wp.large create text $vertTextRightPrint  $threshPlotBottom -text "0" -font monofont -justify right -width 200 -anchor e -tags printout]
      $wp.large create text $vertTextRightPrint  $threshPlotTopPrint -text $maxThresh -font monofont -justify right -width 200 -anchor e -tags printout]
      $wp.large create text [expr $vertTextRightPrint - 20]  [expr ($threshPlotTopPrint + $threshPlotBottom) / 2] -text "(threshold\ncount)" -font monofont -justify center -width 150 -anchor e -tags printout]

        #####mark min of the thresh
        #set xStart  [expr  ( ([expr $minThreshField-$dataOrigin] + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
       # set xEnd  [expr  ( ([expr $minThreshField-$dataOrigin]  + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        #$wp.large create rectangle  $xStart [expr $threshPlotBottom+1] $xEnd [expr $threshPlotBottom +4] -fill "\#991010" -outline "" -tags [list xScalable threshPlotBar]

         #####mark max of the thresh
        #set xStart  [expr  ( ( $maxThreshField-$dataOrigin + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        #set xEnd  [expr  ( ( $maxThreshField-$dataOrigin + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
        #$wp.large create rectangle  $xStart [expr $threshPlotTop-1] $xEnd [expr $threshPlotTop-4] -fill "\#109910" -outline "" -tags [list xScalable threshPlotBar]

       
      #### end of printThresholdGraph

        #for example, if we have 2 frames of data, frame 0 and frame 1,
      #then numDataFrames = 2.  Since dataOrigin =3, fieldLast is 4, since data
      # is in field 3 (frame 0), field 4 (frame 1). Formula is...
      
      #draw data on can
      #loop over all data fields

      if { $rectCreated} {
        #this until separate data and scale highlighting
        #puts "drawing rects, scalex is $scalex"
        #hack here -- for now skip B-field stuff, so minimal stuff drawn
        tlPutsDebug "printing: setting min/max, dataOrigin= $dataOrigin" 
        for {set field [expr $dataOrigin ]} {$field <= $fieldLast} {incr field} {
          set printCol 1;
            
          
          set xPosFieldLeft [expr  -$leftCan + $printDataPosX + ($scalex * $dataWidth * ($field - $dataOrigin)  )  ]
          set xPosFieldRight [expr  -$leftCan + $printDataPosX+  ($scalex * $dataWidth * ($field - $dataOrigin + 1 - $dataMargin)  )  ]

          if { ($xPosFieldRight  <  $leftData) || ($xPosFieldLeft > $rightData)} {
            set printCol 0;
          }
          if {$xPosFieldLeft < $leftData} {
            set xPosFieldLeft $leftData
          }
          if {$xPosFieldRight > $rightData} {
           set xPosFieldRight $rightData
          }
          
          if {$printCol} { 
            #now draw data rectangles
            #puts "drawing field $field at xPosField $xPosField" 
            #yipes, does this redraw all rects (even non visible) every timeXXX
            set y 0.0
            
            set intensity 0
            
            for {set i 0} {$i<=$dataValNum} {incr i} { 
              set val $dataVal($field,$i)
              set printBox  1;
              if {$val != "null"} {
                #calculate color and create rectangle
                
                set yposTop [expr -$topCan + $printDataPosY+ ($scaley * $y)]
                set yposBot [expr $yposTop + ($scaley * $ybox)]
                if {($yposBot < $topData) || ($yposTop > $botData)} {
                set printBox 0
              }

              if {$yposTop < $topData} {
                   set yposTop $topData
              }
              if {$yposBot > $botData} {
                   set yposBot $botData
              }

                 
              #should Prescan  to find range of values!   
              #this should be some per-request-method range / also allow this to be adjusted
              
              #set intensity except if field 4 (indexed struct)
              #puts "field = $field, dataName($field) = $dataName($field),i= $i" 
              if {$dataName(vals) != "struct"} {
                ##if { ( ($field != 4)  ) } open brace here 
                #set range [expr $dataMax($field) - $dataMin($field)]
                set range [expr $trajMax - $trajMin ]
                if { ($range > 0)  && ([string is double $val] )} {
                  set intensity  [expr int (255. * ( (0.0 + $val - $trajMin ) / $range)) ]
                  #tlPutsDebug ": $val $dataMin($field) $range $field $intensity"
                }
                
                
                
                set hexcols [chooseColor $intensity]
              } else {
                #horrifyingly, sends string for data, tcl is typeless
                set hexcols [chooseColor $val ]
              }
              foreach {hexred hexgreen hexblue} $hexcols {} 

              
              #draw data rectangle
              ## add logic to not draw if out of range 
              if {$printBox} {
                $wp.large create rectangle $xPosFieldLeft $yposTop  $xPosFieldRight $yposBot  -fill "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags printout 
              }
            }
            
            set y [expr $y + $ybox]
          }
        }
      }

     

    
       tlPutsDebug "Now start testing for print,  cursorRep($currentMol)= $cursorRep($currentMol)  cursorShown($currentMol)= $cursorShown($currentMol), prevCursorObject($currentMol)= $prevCursorObject($currentMol)  prevCursorFrame($currentMol) =$prevCursorFrame($currentMol)"  
      if {($cursorRep($currentMol)!="null") && $cursorShown($currentMol)} {
        set printCursor 1;
        set y {$ybox * $prevCursorObject($currentMol)}
        set theFrame $prevCursorFrame($currentMol)
        set yposTop [expr -$topCan + $printDataPosY+ ($scaley * $y)]
        set yposBot [expr $yposTop + ($scaley * $ybox)]
        if {($yposBot < $topData) || ($yposTop > $botData)} {
          set printCursor 0;
        }

        if {$yposTop < $topData} {
             set yposTop $topData
        }
        if {$yposBot > $botData} {
             set yposBot $botData
        }
          set xPosFieldLeft [expr  -$leftCan + $printDataPosX + ($scalex * $dataWidth * ($theFrame)  )  ]
          set xPosFieldRight [expr  -$leftCan + $printDataPosX+  ($scalex * $dataWidth * ($theFrame + 1 - $dataMargin)  )  ]

          if { ($xPosFieldRight  <  $leftData) || ($xPosFieldLeft > $rightData)} {
            set printCursor 0;
          }
          if {$xPosFieldLeft < $leftData} {
            set xPosFieldLeft $leftData
          }
          if {$xPosFieldRight > $rightData} {
           set xPosFieldRight $rightData
          }
          # tlPutsDebug "printCursor= $printCursor xPosFieldLeft= $xPosFieldLeft  xPosFieldRight= $xPosFieldRight  yposTop= $yposTop  ypoBot= $yposBot"

          if {$printCursor} { 
            tlPutsDebug "now printing cursor"  
          #$wp.large create  $vertTextLeftPrint $ypos $vertTextRightPrint [expr $ypos + ($scaley * $ybox)]    -outline "\#FF0000" -width 2 -fill "" -tags printout    
           #cursor highlight
           $wp.large create rectangle $xPosFieldLeft $yposTop  $xPosFieldRight $yposBot  -outline "\#FF0000" -width 2 -fill "" -tags printout 
           #vert scale highlght
           $wp.large create rectangle $vertTextLeftPrint $yposTop  [expr $printDataPosX - 5] $yposBot  -outline "\#FF0000" -width 2 -fill "" -tags printout 
           #horz scale highlight
           $wp.large create rectangle $xPosFieldLeft [expr $botData + 5] $xPosFieldRight $yposHorzScale -outline "\#FF0000" -width 2 -fill "" -tags printout 
         }

      }  
    }
  }  else {
     puts "Timeline: need to display a graph before printing"
  }
##end dataRect and cursor highlight printing
  
  ####drawHorzScale
    #ensure minimal horizontal spacing
    # hardcoded spacing
    set horzSpacing 27 
    set horzPad 5
    set horzSpacingPad [expr $horzSpacing + $horzPad]
    set horzDataTextSkip [expr $dataWidth]
    set scaledHorzDataTextSkip [expr $scalex * $dataWidth]
    set scaledHorzDataOffset [expr $scalex * $dataWidth / 2.0]
    set xStart [expr ($xcol($dataOrigin))]
    set xDataEnd  [expr int ($xStart +  $scalex * ($dataWidth * $numDataFrames ) ) ] 
    set x 0 



    #numbers are scaled for 1.0 until xpos
    #this is tied to data fields, which is produced from frames upon
    #first drawing. Should really agreee with writeDataFile, which currently uses frames, not fields
    set xposPrev -1000 
    set xposRightPrev -1000 
    #traj data starts at dataOrigin
    for {set frameNum 0} {$frameNum < $numDataFrames} {incr frameNum} {
      set field [expr $frameNum + $dataOrigin]
      set textWidth [font measure $monoFont -displayof $wp $frameNum] 
      set textWidthPad [expr $textWidth +$horzPad]
    ####for {set field [expr $dataOrigin]} {$field <= $fieldLast} {incr field} {}
    ####  set frameNum [expr $field - $dataOrigin -1]
      
      set xpos [expr int ($xStart + ($scalex * $x) + $scaledHorzDataOffset)]
      set xposRight [expr $xpos +int($textWidth/2)]
      if { ( ($xposRight - $xposRightPrev  ) >= $textWidthPad) && ( ( $field == $fieldLast) || ( ( $xDataEnd - $xpos) > ( 2 * $textWidth) ) ) } {
        # draw the frame number if there is room
        #for speed, we use horzScaleText instead of $dataName($field)
        set xposShifted [expr $xpos-$leftCan + $printDataPosX]
        #tlPutsDebug "xposShifted= $xposShifted   xpos= $xpos  leftCan= $leftCan leftData= $leftData  rightData= $rightData"
        if {$xposShifted >=$leftData && $xposShifted <=$rightData} {
          $wp.large create text $xposShifted $yposHorzScale -text "$frameNum" -width 30 -font $monoFont -justify center -anchor s -tags printout 
          set xposPrev $xpos
          set xposRightPrev $xposRight
        }
      }        
      set x [expr $x + $horzDataTextSkip]
    } 
    ##end drawHorzScale    


  ####drawVertScale starts
    
    #Add the text...
    set field 0           

    #note that the column will be 0, but the data will be from picked
    
    
    set yDataEnd [expr $ytopmargin + ($scaley * $ybox * ($dataValNum +1))]
    set y 0.0

    set yposPrev  -10000.0

    #Add the text to vertScale...
    set field 0            


    #we want text to appear in center of the dataRect we are labeling
    set vertOffset [expr $scaley * $ybox / 2.0]

    #don't do $dataValNum, its done at end, to ensure always print last 
    for {set i 0} {$i <= $dataValNum} {incr i} {
      #set ypos [expr $ytopmargin + ($scaley * $y) + $vertOffset]
      set ypos [expr  ($scaley * $y) + $vertOffset]
      if { ( ($ypos - $yposPrev) >= $vertTextSkip) && ( ( $i == $dataValNum) || ( ($yDataEnd - $ypos) > $vertTextSkip) ) } {
        set yposShifted [expr $ypos-$topCan+$printDataPosY]
        if {$yposShifted >= $topData && $yposShifted <=$botData} {
          if {$usesFreeSelection} {
            $wp.large create text $vertTextRightPrint $yposShifted -text $dataVal(freeSelLabel,$i)  -width $vertTextWidthPrint -font $monoFont -justify right -anchor e -tags printout 
           } else {
            if {$resCodeShowOneLetter == 0} {
              set res_string $dataVal(resname,$i)
            } else {
              set res_string $dataVal(rescode,$i)
            }
           #for speed, we use vertScaleText instead of $dataName($field)
           #how to deal with chain vs. segname?  For now, don't show segname.  Should allow toggle?
          $wp.large create text $vertTextRightPrint $yposShifted -text "$dataVal(resid,$i) $res_string $dataVal(chain,$i)" -width $vertTextWidthPrint -font $monoFont -justify right -anchor e -tags printout 
          }
       set yposPrev  $ypos
        }
       ##set yposPrev  $ypos
      }        
      set y [expr $y + $vertTextSkip]
      
    } 
    ####drawVertScale ends

    printColScale $colScalePosX $colScalePosY


        
    
    tlPutsDebug "done with print redraw, everRedrawn= $everRedrawn"

    # Now do the printing.
    set filename "VMD_Timeline_Window.png"
    display update ui
    #set filename [tk_getSaveFile -initialfile $filename -title "VMD Timeline Print" -parent $wp -filetypes [list {{PNG - Portable Network Graphicis} {.png}} {{All files} {*} }] ]
    if {$filename != ""} {
      $wp.large postscript -pagewidth 7.0i -file /tmp/tl-convert.eps
      exec /usr/local/bin/epstopdf /tmp/tl-convert.eps --outfile=/tmp/tl-convert.pdf
      exec pdftoppm -r 250 /tmp/tl-convert.pdf /tmp/tl-convert.ppm 
      exec convert /tmp/tl-convert.ppm-000001.ppm png:$filename
      #exec display $filename
      set currentMol_name [molinfo $currentMol get name]
      #::remote::distributeImageToAll $filename  "Timeline  $currentMol_name (mol $currentMol)" 
      ::remote::distributeImageToAll $filename  "Timeline"
    }
    # printing done
    #### now hide or destroy print window 
    catch {destroy $wp}
  } else {
    tk_messageBox -message "Timeline has nothing to print. Please calculate or load a data plot." -parent $w -icon warning -type ok -title "VMD Timeline warning"
  }
  return
}


### progressBox procs
## XXX  to be placed into own namespace later.
###
proc ::timeline::progressBoxDestroy {args} {
  variable progressDialogName
  variable progressBoxExists
  set d $progressDialogName
  catch {destroy $d}
  set progressBoxExists 0
  display update ui
}


proc ::timeline::progressBox {args} {
  tlPutsDebug "started Progress Box"
  variable progressDialogName
  variable progressCancelPressed 
  variable progressLength
  variable progressBoxExists
  variable clicked
  progressBoxDestroy

  set d $progressDialogName
  toplevel $d -class Dialog
  wm title $d {Timeline Calculation Progress}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 220 120  
  
  set progressBoxExists 1
  set progressCancelPressed 0 
  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Timeline calculation}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Progress:}
    scale $d.sc -orient horizontal  -length $progressLength -sliderrelief flat  -sliderlength 0   -activebackground #99FF99 -troughcolor #FFEEEE  -state active -showvalue 0  -label 0
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.sc" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    
    
    # buttons
    button $d.cancel -text {Cancel} -command {set ::timeline::progressCancelPressed 1;::timeline::progressBoxDestroy}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0

    bind $d <Destroy> {set ::timeline::clicked -1}
    #set oldFocus [focus]
    #set oldGrab [grab current $d]
    #if {[string compare $oldGrab ""]} {
    #    set grabStatus [grab status $oldGrab]
    #}
    #grab $d
    #focus $d

    # wait for user to click
    # remove this wait
    #vwait ::timeline::clicked
    #catch {focus $oldFocus}
    #catch {
    #    bind $d <Destroy> {}
    #    destroy $d
    #}
    #if {[string compare $oldGrab ""]} {
    #  if {[string compare $grabStatus "global"]} {
    #        grab $oldGrab
    #  } else {
    #      grab -global $oldGrab
    #    }
    #}
    display update ui
    return
}


proc ::timeline::progressBoxVerify {} {
puts "progressBox is present"
}

proc ::timeline::progressBoxConfigure {val} {
  #val wil be 0.0 to 100.0
  variable progressDialogName
  variable progressLength
  set d $progressDialogName
  set valp [expr 100.0 * $val]
  set valString  [format "%6.2f%%" $valp]
  set l [expr ($val) * $progressLength ]
  $d.sc configure -sliderlength $l -label $valString -state active
  display update ui
}


proc ::timeline::progressFindFrac {startFrac endFrac segmentOverallFrac count totalCount} {
  # find progress with 0.0 to 1.0 range of of $segmentOverallFrac, itself a fraction of _overall progress
  # for example, from 0.0 to 0.3 range of 0.5, so will appear as overall progress % from 0.0 to 0.15
  set a [expr $startFrac * $segmentOverallFrac  + ( $segmentOverallFrac * ($endFrac - $startFrac)  * double ($count)/$totalCount)]
  return $a
}

proc ::timeline::calcInterSelContacts {lastCalcVal} {

  variable w
  variable dataOrigin
  variable dataMin
  variable dataMax
  variable dataVal
  variable dataHash
  variable dataValNum
  variable rectCreated 
  variable dataFileVersion 
  variable usesFreeSelection
  variable dataName
  variable lastCalc  
  variable numDataFrames
  variable numTrajFrames
  variable currentMol
  variable firstAnalysisFrame 
  variable lastAnalysisFrame
  variable interSelList
  variable interSelListPairs
  variable interDist
  variable interFromResSel
  variable interToSel
  variable interSetMethod 
  variable progressDrawFraction
  variable progressBoxExists
  variable progressCancelPressed
  variable keyAtomSelString

  set dataMin(all) "null"
  set dataMax(all) "null"

  if $interSetMethod==2 {
    set selList $interSelListPairs
  } else {
    set selList $interSelList
  }
  set contactDist $interDist 
  tlPutsDebug "in calcInterSelContacts"
  set commonName "inter-sel. contacts"
  set oldNumDataFrames $numDataFrames
  set lastCalc $lastCalcVal

  clearData
  set usesFreeSelection 1
  
  progressBox 0
  tlPutsDebug "progressBox created in interSelContacts.  progressBoxExists= $progressBoxExists"
  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
   

      #dataValNum internally counts from 0, so subtract 1 from the counting number


  #validations
  # for method 0, validate selList to validSelList
  # for method 1, get key atoms and check to Sel 
  set validSelList "" 
  # method 0 is selection list, method 1 is residues-in-selection to other-selction
  switch -exact $interSetMethod {
    0 {
      if {[catch {set listLength [llength $selList]}]} {
          tk_messageBox -message "Timeline can't parse selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
      }  
      if {$listLength==0} {
        tk_messageBox -message "Timeline can't find a selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
      }

      foreach selString $selList {
        set selNum 0 
        set sel ""
        tlPutsDebug "selString is >$selString<"
        if {[catch {set sel [atomselect $currentMol "$selString"]; set selNum [$sel num] } ] } {
          tk_messageBox -message "Timeline can't make atom selection for: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
        } else {
          if {$selNum > 0} {
            lappend validSelList $selString
            tlPutsDebug "validSelList: $validSelList validSelList length:[llength $validSelList]"
          } else { 
            tlPutsDebug "num of sel is [$sel num]"
            tk_messageBox -message "Timeline can't find atoms for selection: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
          }
        }
      } 
    }
    #switch option
    1 {
      #residues-in-selection to other-selction
      # first, find the residues in interFromResSel 
      tlPutsDebug "interFromResSel= >$interFromResSel<   interToSel= >$interToSel<"
      if {[catch {set fromResSelKeyAtomsSel [atomselect $currentMol "($interFromResSel) and $keyAtomSelString"]} ]} {
        tk_messageBox -message "Timeline can't parse the From selection.  Please enter a selection like: resid 10 to 20"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
      } 

      if {[catch {set toSel [atomselect $currentMol "$interToSel"]} ]} {
        tk_messageBox -message "Timeline can't parse the To selection list.  Please enter a selection like: resid 10 to 20"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
      }  

      if { [$fromResSelKeyAtomsSel num] == 0} {
        tk_messageBox -message "Timeline can't find any residues in From selection.  Please enter a From selection which contains protein or nucleic acid residues, for example: segname LIG"   -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
      } 
    }
    2 {
      if {[catch {set listLength [llength $selList]}]} {
          tk_messageBox -message "Timeline can't parse selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
      }  
      if {$listLength==0} {
        tk_messageBox -message "Timeline can't find a selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
      }

      foreach selString $selList {
        set selNum 0 
        set sel ""
        tlPutsDebug "selString is >$selString<"
        if {[catch {set sel [atomselect $currentMol "$selString"]; set selNum [$sel num] } ] } {
          tk_messageBox -message "Timeline can't make atom selection for: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
        } else {
          if {$selNum > 0} {
            lappend validSelList $selString
            tlPutsDebug "validSelList: $validSelList validSelList length:[llength $validSelList]"
          } else { 
            tlPutsDebug "num of sel is [$sel num]"
            tk_messageBox -message "Timeline can't find atoms for selection: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
          }
        }
      } 
    }
    #make arm 2 avoid pairsswitch option
    default {
      puts "Timeline: error in calling inter-selection contacts"
      return   
    }
  }
  set dataMin(all) "null"
  set dataMax(all) "null"
  set dataName(vals) $commonName

  set lengthVsl [llength $validSelList]
  set count 0
  #
  # Now set lists, depending on method
  # n Choose 2 = n! / 2! * (n-2)! = ( (n) (n-1)) /2
  switch -exact $interSetMethod {
    0 {
      tlPutsDebug "InterSel: Now in phase B, method 0"
      set finalDataValNum [expr  ($lengthVsl * ($lengthVsl-1.0) ) /2 ]
      for {set i 0} {$i<$lengthVsl} {incr i} {
        for {set j 0} {$j<$i} {incr j} {
          tlPutsDebug "[lindex $validSelList $i] [lindex $validSelList $j]"
          set dataVal(freeSelLabel,$count) "[lindex $validSelList $i]==[lindex $validSelList $j]"
          set dataVal(freeSelString,$count) "([lindex $validSelList $i]) or ([lindex $validSelList $j])"
          
          #here loop over all frames for contacts
          set sel1 [atomselect $currentMol [lindex $validSelList $i]]
          set sel2 [atomselect $currentMol [lindex $validSelList $j]]
          tlPutsDebug "sel1 num= [$sel1 num]   sel2 num= [$sel2 num]"
          for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
            $sel1 frame $trajFrame
            $sel2 frame $trajFrame
            display update ui
            set curField [expr $dataOrigin + $trajFrame]
            tlPutsDebug "Now to measure contacts $contactDist $sel1 $sel2 sel1= >$sel1< sel2= >$sel2<"
            set dataVal($curField,$count) [llength [lindex [measure contacts $contactDist $sel1 $sel2] 0]]
            checkRangeLimits $dataVal($curField,$count)
            #tlPutsDebug "dataVal($curField,$i)= $dataVal($curField,$i)" 
           
            if {$progressCancelPressed} break 
          }
          $sel1 delete
          $sel2 delete
          tlPutsDebug "count/finalDataValNum is $count / $finalDataValNum"

          tlPutsDebug  "progressBoxExists= $progressBoxExists  progressCancelPressed= $progressCancelPressed"

          if {$progressBoxExists} {
            progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction $count  $finalDataValNum] 
          }
         
          incr count
          
         if {$progressCancelPressed} break 
        }
      }
    }
    #switch option
    1 {
      tlPutsDebug "InterSel: Now in phase B, method 1"
      set count 0
      set finalDataValNum [$fromResSelKeyAtomsSel num] 
      set fromResSelKeyAtomSelIndex [$fromResSelKeyAtomsSel get index]
      foreach a $fromResSelKeyAtomSelIndex {
        #optimize later so makes fewer selections
        set curSel [atomselect $currentMol "index $a"]
        set curResid [$curSel get resid]
        set curResname [$curSel get resname] 
        $curSel delete
        set dataVal(freeSelLabel,$count) "$curResname $curResid==$interToSel"
        set dataVal(freeSelString,$count) "(same residue as index $a) or ($interToSel)"
        #Now loop
        #here loop over all frames for contacts
        
        set sel1 [atomselect $currentMol "same residue as index $a"]
        set sel2 [atomselect $currentMol $interToSel]
          tlPutsDebug "sel1 num= [$sel1 num]   sel2 num= [$sel2 num]"
          for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
            $sel1 frame $trajFrame
            $sel2 frame $trajFrame
            display update ui
            set curField [expr $dataOrigin + $trajFrame]
            tlPutsDebug "Now to measure contacts $contactDist $sel1 $sel2 sel1= >$sel1< sel2= >$sel2<"
            set dataVal($curField,$count) [llength [lindex [measure contacts $contactDist $sel1 $sel2] 0]]
            checkRangeLimits $dataVal($curField,$count)
            #tlPutsDebug "dataVal($curField,$i)= $dataVal($curField,$i)" 
           
            if {$progressCancelPressed} break 
          }
          $sel1 delete
          $sel2 delete
          tlPutsDebug "count/finalDataValNum is $count / $finalDataValNum"

          tlPutsDebug  "progressBoxExists= $progressBoxExists  progressCancelPressed= $progressCancelPressed"

          if {$progressBoxExists} {
            progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction $count  $finalDataValNum] 
          }
         
          incr count
          
         if {$progressCancelPressed} break 
        }
        $fromResSelKeyAtomsSel delete
        $toSel delete
    }
    2 {
      tlPutsDebug "InterSel: Now in phase B, method 2"
      set finalDataValNum [expr  $lengthVsl / 2]
      for {set i 0} {$i<$lengthVsl} {set i [expr $i+2]} {
        tlPutsDebug "[lindex $validSelList $i] [lindex $validSelList [expr $i+1]]"
        set dataVal(freeSelLabel,$count) "[lindex $validSelList $i]==[lindex $validSelList [expr $i+1]]"
        set dataVal(freeSelString,$count) "([lindex $validSelList $i]) or ([lindex $validSelList [expr $i+1]])"
        
        #here loop over all frames for contacts
        set sel1 [atomselect $currentMol [lindex $validSelList $i]]
        set sel2 [atomselect $currentMol [lindex $validSelList [expr $i+1]]]
        tlPutsDebug "sel1 num= [$sel1 num]   sel2 num= [$sel2 num]"
        set calcCountTotal [expr $lastAnalysisFrame - $firstAnalysisFrame+1]
        for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
          $sel1 frame $trajFrame
          $sel2 frame $trajFrame
          display update ui
          set curField [expr $dataOrigin + $trajFrame]
          tlPutsDebug "Now to measure contacts $contactDist $sel1 $sel2 sel1= >$sel1< sel2= >$sel2<"
          set dataVal($curField,$count) [llength [lindex [measure contacts $contactDist $sel1 $sel2] 0]]
          checkRangeLimits $dataVal($curField,$count)
          #tlPutsDebug "dataVal($curField,$i)= $dataVal($curField,$i)" 
         
          if {$progressCancelPressed} break 
        }
        $sel1 delete
        $sel2 delete
        tlPutsDebug "count/finalDataValNum is $count / $finalDataValNum"

        tlPutsDebug  "progressBoxExists= $progressBoxExists  progressCancelPressed= $progressCancelPressed"

        if {$progressBoxExists} {
          progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction $count  $finalDataValNum] 
        }
       
        incr count
        
       if {$progressCancelPressed} break 
        
      }
    }
    #switch option
    default {
      puts "Timeline: error in calling inter-selection contacts"
      return   
    }
  }
  set dataValNum [expr $count-1] 

  
  if {$progressCancelPressed} {
        tlPutsDebug "progressCancelPressed" 
        set progressCancelPressed 0
        clearData
  }

  #XX hack for coloring? (good coloring needs to indicate two selections)


 
  #dataValNum-> number of intersels produced
  set lastCalc -1




  configureSelInfo null 0
  initPicked 
  postDataFill

  progressBoxDestroy
 
  return
 
} 

proc ::timeline::factorial {n} {
  set p 1
  for {set i 1} {$i<=$n} {incr i} {
   set  p [expr $p * $i]
  }
  return $p
} 



proc ::timeline::calcGlobalCcss  {lastCalcVal}  {

# timeline adaptaton of Ryan's McGreevy cross-correlation coefficient
# code, placing trajectory data into timeline data structures 
# From Ryan M.:
# per ss ccc for protein only
# output global ccc based on weighted ss ccc 
# of secondary structural segments
# With addition of user-specified selections
  
  package require mdff
  
  variable w 
  variable dataName
  variable usesFreeSelection
  variable dataVal
  variable dataValNumResSel
  variable dataValNum
  variable dataMin
  variable dataMax
  variable currentMol
  variable nullMolString
  variable progressCancelPressed
  variable progressDrawFraction
  variable progressBoxExists
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable partSelText
  variable dataOrigin
  variable lastCalc

  variable ccssMapFile 
  variable ccssMapres
  variable ccssSpacing
  variable ccssThreshold
  variable ccssUseThreshold
  variable ccssUseSpacing
  variable ccssSelMethod
  variable ccssSelList 
  
  set dataMin(all) "null"
  set dataMax(all) "null"
  
  #set usesFreeSelection 0
  #set dataValNum $dataValNumResSel

  set calcCountTotal [expr  ($lastAnalysisFrame - $firstAnalysisFrame +1)]
  set calcCount 0
  tlPutsDebug "starting globalCC"

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XX
      puts "Timeline: select molecule before choosing Calculate method"
      return 
  } 

  progressBox 0

  
  if { [catch {mol addfile $ccssMapFile $currentMol } ] } {
      tk_messageBox -message "Timeline couldn't load the density map file. Please specify a valid file."  -parent $w -icon warning -type ok -title "VMD Timeline warning"
      progressBoxDestroy
      return
   }

   set ccssMapVolId [expr [molinfo $currentMol get numvolumedata] -1]
    
   set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
   if {$ccssMapVolId == -1} {
        tk_messageBox -message "Timeline can't find any volumetric data for cross correlation. There may be a problem with loading the specified volumeteric data file."  -parent $w -icon warning -type ok -title "VMD Timeline warning"
        progressBoxDestroy
        return
   }

  set dataName(vals) "cross-corr."
  
  

  set lastCalc $lastCalcVal

  tlPutsDebug "starting crossCorr, just set lastCalc"

  if {$ccssSelMethod == 1}  {
    #evaluate string user has entered
    set usesFreeSelection 1
    set validSelList "" 
    if {[catch {set listLength [llength $ccssSelList]}]} {
      tk_messageBox -message "Timeline can't parse the From selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
      progressBoxDestroy
      return
    }  
    if {$listLength==0} {
      tk_messageBox -message "Timeline can't find a selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
      progressBoxDestroy
      return
    }
    # partNum is number of residues when doing seccondary structure, number of atoms when free selection
    #Now, loop over and check for valid SelList entries 
    set partNum 0
    foreach selString $ccssSelList {
      set selNum 0 
      set sel ""
      #tlPutsDebug "selString is >$selString<"
      if {[catch {set sel [atomselect $currentMol "$selString"]; set selNum [$sel num] } ] } {
        tk_messageBox -message "Timeline can't make atom selection for: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
      } else {
        if {$selNum > 0} {
          lappend validSelList $selString
          set partNum [expr $partNum + $selNum ]
          #tlPutsDebug "validSelList: $validSelList validSelList length:[llength $validSelList]; selNum= $selNum  partNum= $partNum"
        } else { 
          #tlPutsDebug "num of sel is [$sel num]"
          tk_messageBox -message "Timeline can't find atoms for selection: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
        }
      }
    }
    set lengthVsl [llength $validSelList]

    if {$lengthVsl <= 0} {
      tk_messageBox -message "Timeline can't find a selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
      progressBoxDestroy
      return
    }
    set dataValNum [expr $lengthVsl - 1]
    #
    # Now set names of labels and selections
    set count 0
    foreach userSelText $validSelList { 
      set dataVal(freeSelLabel,$count) $userSelText
      set dataVal(freeSelString,$count) $userSelText
      incr count
    }
  } else {
    set usesFreeSelection 0
    set atomSel [atomselect $currentMol "protein and $partSelText"]
    #XXX fix so gets proper resnum in all cases (large molecules, many chains)
    # partNum is number of residues when doing seccondary structure, number of atoms when free selection
    set partNum [llength [lsort -unique [$atomSel get resid]]]
    #tlPutsDebug "residue number = partNum= $partNum"
   # get total resid number for the atomSel
    set dataValNum $dataValNumResSel
  }
   
  
  for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
     set curField [expr $dataOrigin + $trajFrame]
           
    # initialize variables for this frame
    set CCSS 0.0
    set CCwSS 0.0
    set CCwSSList {}
    set globalCCwSS 0.0
    set nanRegister 0
    set percent 0.0
    set ccList(0) "" 

    #tlPutsDebug "starting off, just set ccList"
     
    #parse chain (or maybe done before progressBox started, to show error messages) then evaluate N parsed chain parts  with N cross-corr calls.
    if {$ccssSelMethod == 1}  {
      #assign cc values and scale 
        set listCount 0
        set vslLength [llength $validSelList] 
        foreach userSelText $validSelList { 
          set userSel [atomselect $currentMol $userSelText frame $trajFrame]
          #tlPutsDebug "userSelText= $userSelText num in userSel = [$userSel num]"
          #set CCSS [mdff ccc $userSel -i $ccssMapFile -res $ccssMapres -spacing $ccssSpacing -threshold $ccssThreshold]
          if {$ccssUseThreshold} {
           # tlPutsDebug "yes threshold use, threshold= $ccssThreshold"
            if {$ccssUseSpacing} {
              set CCSS [mdffi cc $userSel -res $ccssMapres -thresholddensity $ccssThreshold -mol $currentMol -vol $ccssMapVolId -spacing $ccssSpacing ]
            } else {
              set CCSS [mdffi cc $userSel -res $ccssMapres -thresholddensity $ccssThreshold -mol $currentMol -vol $ccssMapVolId ]
            }
          } else {
            #tlPutsDebug "no threshold use, threshold= $ccssThreshold"
            if {$ccssUseSpacing} {
              set CCSS [mdffi cc $userSel -res $ccssMapres -mol $currentMol -vol $ccssMapVolId -spacing $ccssSpacing ]
            } else {
              set CCSS [mdffi cc $userSel -res $ccssMapres -mol $currentMol -vol $ccssMapVolId ]
            }
          } 
          #tlPutsDebug "CCSS is $CCSS"
          #detect if nan
          if {![string is double $CCSS]|| $CCSS != $CCSS } {
            #tlPutsDebug nonNumericCCSS
           # number of resid = id_i - id_0 + 1
            incr nanRegister [expr $id_i - $id_0 + 1]
            #tlPutsDebug "nanRegister $nanRegister"
            #tlPutsDebug "VMD WARNING: CrossCorr NaN found.  trajFrame= $trajFame  userSelText= $userSelText num in userSel = [$userSel num] "
            set CCSS -1.0
          }
          set scaling [expr (0.0+ [$userSel num ]) / $partNum ]
          set CCwSS [expr $CCSS * $scaling]
          # X for now, do not use this
          #tlPutsDebug "scaling=$scaling;  partNum= $partNum   CCwSS= $CCwSS  userSelNum= [$userSel num]"
          #
          #set dataVal($curField,$listCount) $CCwSS
          # Here we discard the scaled value we have calculated.
          # We might use the scaled value as a user option, still prototyping.
          set dataVal($curField,$listCount) $CCSS
          checkRangeLimits $dataVal($curField,$listCount)
          lappend CCwSSList $CCwSS
          $userSel delete
          
          incr listCount
          if {$progressBoxExists} {
            progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction $listCount  $vslLength] 
         }  
         if {$progressCancelPressed} break 
        }
    } else {
      # unless recalcing secondary structure, this should be changed so only be done once
      #set all $all
      set segList [lsort -unique -dictionary [$atomSel get segid]]
      #XXX can I avoid setting these?
      # $atomSel set beta 0.0
      # $atomSel set occupancy 0.0
      #tlPutsDebug "Just set segList"
      # PDB files do not have segids
      if {[llength $segList] < 1} {
        foreach chain [lsort -unique [$atomSel get chain]] {
            set chainSel [atomselect $currentMol "chain $chain"]
            #XXX altering (admittedly empty) molec data; not good.  Avoid this with an empty-segid-atoms pass below.
            #XXX worse, current approach misses segid-less parts of mixed segid-full and segid-less molecules
            $chainSel set segid $chain
            $chainSel delete
        }
        set segList [lsort -unique -dictionary [$atomSel get segid]]
      }
      #tlPutsDebug "segList: $segList"

      foreach seg $segList {
        set segCASel [atomselect $currentMol "segid $seg and name CA" frame $trajFrame]
        #tlPutsDebug "segid $seg"
        set ssList [$segCASel get structure]
        #tlPutsDebug $ssList
        set resList [lsort -unique -integer [$segCASel get resid]]
        set resListLength [llength $resList]
        $segCASel delete

        set id_end [llength $resList]
        set id_0 0
        for {set id_i 0} {$id_i <= $id_end} {incr id_i} {
          #tlPutsDebug check2
          #do next two lines  mean the unique in resList sort could cause problems with malformed molecs?
          set ss_0 [lindex $ssList $id_i]
          set ss_1 [lindex $ssList [expr $id_i + 1]]
          #tlPutsDebug "id_i = $id_i ss_0 = $ss_0 ss_1 = $ss_1"
            # 
          if {$ss_0 != $ss_1} {
            set res_end [lindex $resList $id_i]
            set res_start [lindex $resList $id_0]
            set ssSel [atomselect $currentMol "segid $seg and (resid $res_start to $res_end)" frame $trajFrame]
            if {[llength [$ssSel list]] >0} { 
              #tlPutsDebug "segid $seg and (resid $id_0 to $id_i)"
              #tlPutsDebug "segment contains [$ssSel num] atoms."
              #tlPutsDebug "about to run: mdff ccc $ssSel -i $ccssMapFile -res $ccssMapres -spacing $ccssSpacing -threshold $ccssThreshold"
              #set CCSS [mdff ccc $ssSel -i $ccssMapFile -res $ccssMapres -spacing $ccssSpacing -threshold $ccssThreshold]
              if {$ccssUseThreshold} {
                #tlPutsDebug "yes threshold use, threshold= $ccssThreshold"
                if {$ccssUseSpacing} {
                  set CCSS [mdffi cc $ssSel -res $ccssMapres -thresholddensity $ccssThreshold -mol $currentMol -vol $ccssMapVolId -spacing $ccssSpacing ]
                } else {
                  set CCSS [mdffi cc $ssSel -res $ccssMapres -thresholddensity $ccssThreshold -mol $currentMol -vol $ccssMapVolId ]
                }
               } else {
                #tlPutsDebug "no threshold use, threshold= $ccssThreshold"
                if {$ccssUseSpacing} {
                  set CCSS [mdffi cc $ssSel -res $ccssMapres -mol $currentMol -vol $ccssMapVolId -spacing $ccssSpacing]
                } else {
                  set CCSS [mdffi cc $ssSel -res $ccssMapres -mol $currentMol -vol $ccssMapVolId ]
                }
              }
              
              #tlPutsDebug "CCSS = $CCSS"

              # in case no overlap at all
              if {![string is double $CCSS]|| $CCSS != $CCSS } {
                #tlPutsDebug check3.1
                #tlPutsDebug nonNumericCCSS
                # number of resid = id_i - id_0 + 1
                incr nanRegister [expr $res_end - $res_start + 1]
                #tlPutsDebug "nanRegister $nanRegister"
                
                tlPutsDebug "VMD WARNING: CrossCorr NaN found.  trajFrame= $trajFrame  sel. text: segid $seg and (resid $res_start to $res_end)  . num in userSel = [$ssSel num] "
                set CCSS -1.0
              }


              #tlPutsDebug check4.1
              # store raw CC

              set scaling [expr {double($res_end - $res_start + 1) / double($partNum)}]
              #tlPutsDebug "scaling=[expr {double($id_i - $id_0 + 1) / double($partNum)}]"

              #Now get scaled quantity (this is the one we want)
              set CCwSS [expr $CCSS * $scaling]
               

              # store weighted CC
              set setList [$ssSel get {name resid chain segname}]
              foreach elem $setList {
                foreach {aName aResid aChain aSegname} $elem {
                  #tlPutsDebuguts "List: name: $aName resid: $aResid chain: $aChain segname: $aSegname"
                  if {$aName=="CA"} {
                    #tlPutsDebug "a CA was found"
                    set spaceToCodeSeg [string map {" " %20} $aSegname]
                    set ccList($aResid,$aChain,$spaceToCodeSeg) $CCSS
                    #tlPutsDebug "Just set ccList($aResid,$aChain,$spaceToCodeSeg) to $ccList($aResid,$aChain,$spaceToCodeSeg)" 
                    #tlPutsDebug "setting for ccList($aResid,$aChain,$spaceToCodeSeg) CCwSS val of $CCwSS, contents=$ CCwSS val of $CCwSS, contents= $ccList($aResid,$aChain,$spaceToCodeSeg)"
                    #tlPutsDebug "set for ccList complete"
                  }
                } 
              }
                 

              lappend CCwSSList $CCwSS
              $ssSel delete

              set id_0 [expr $id_i +1]
              #tlPutsDebug "id_0 has moved to $id_0"
            } else { puts "Bad Selection: [$ssSel text]"}
          }
        }
        display update ui
        if {$progressCancelPressed} break 
      } 
   #$atomSel writepdb CCwSS.$file.$trajFrame.pdb
   #$atomSel delete
   #
   # Now loop through atoms in the selection as for a normal function lookup and look up  in the compiled ccList
      # follows standard Timeline per-res val assignment 
      set defaultSetCount 0
      # all the residues have already been found, since this is not a freeSel
      for {set i 0} {$i <= $dataValNum} {incr i} {
        set aResid $dataVal(resid,$i)
        set aChain $dataVal(chain,$i)
        set aSegname $dataVal(segname,$i)
        set spaceToCodeSeg [string map {" " %20} $aSegname]
        #Here we extract the info that we stored in ccList($resid,$chain,$spaceToCodeSeg) and we put it into dataVal
        if {[catch {set ccVal $ccList($aResid,$aChain,$spaceToCodeSeg)}] } {
          set dataVal($curField,$i) -100.0
          checkRangeLimits $dataVal($curField,$i)

          tlPutsDebug "frame $trajFrame  set default for resid= >$aResid< segname=>$aSegname< chain= >$aChain< spaceToCodeSeg= >$spaceToCodeSeg<"
          incr defaultSetCount
        } else {     
          set dataVal($curField,$i) $ccVal
          checkRangeLimits $dataVal($curField,$i)
        }
         
        if {$progressCancelPressed} break 
      }

      if {$progressCancelPressed} break 

      if {$defaultSetCount>0} {
         tlPutsDebug "Warning: VMD:Timeline - can't find cross correlation value for $defaultSetCount values in frame $trajFrame; setting these to -100.0"
      }
    } 
     
    # XXX Make sure these work for this frame (if we are reporting)
    set globalCCwSS [vecsum $CCwSSList]
    set percent [expr {1 - double($nanRegister) / double($partNum)}]
    # previously,this would return from function for a single frame, so could make list of frame numbers and scores, sorted by score
    # XX return this functionality later
    tlPutsDebug "CC Scoring Data, frame $trajFrame: $globalCCwSS $percent"
    set calcCount [expr $trajFrame - $firstAnalysisFrame+1]
    if {$progressBoxExists} {
       progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction $calcCount $calcCountTotal] 
            #tlPutsDebug "progressCalcFraction= $progressCalcFraction calcCount= $calcCount calcCountTotal= $calcCountTotal pff= [progressFindFrac 0 1.0  $progressCalcFraction $calcCount $calcCountTotal]"
    }
    if {$progressCancelPressed} break
     
  }    
  

  if {$progressCancelPressed} {
        tlPutsDebug "progressCancelPressed" 
        set progressCancelPressed 0
        clearData
  }

  #XX hack for coloring? (good coloring needs to indicate two selections)


 
  #dataValNum-> number of intersels produced
  set lastCalc $lastCalcVal




  configureSelInfo null 0
  initPicked 
  postDataFill

  progressBoxDestroy

   
 
  return
 
}





proc ::timeline::calcNativeContacts {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable currentMol
  variable keyAtomSelString
  variable nativeRefFrame
  variable nativeDist
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable nativeFromSelList
  variable nativeToSel
  variable progressDrawFraction
  variable progressBoxExists
  variable progressCancelPressed

  set dataMin(all) "null"
  set dataMax(all) "null"
 
  set refFrame $nativeRefFrame
  set cutoff $nativeDist
  set fromSelList $nativeFromSelList
  set toSel $nativeToSel
     

  set lastCalc $lastCalcVal

  clearData
  progressBox 0
  set usesFreeSelection 1


  set progressCalcFraction [expr 1.0 - $progressDrawFraction] 
    
  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "Timeline: select molecule before choosing Calculate method"
      return 
  } 

  set allSel [atomselect $currentMol "all and $keyAtomSelString"]
 
    # for each residue in fromSel
  


   # set fromSelString "protein" 
   # set toSelString "protein" 
   # set theChain T
   # set theSegname TIT
   # set theResid 30

    # insert this lower
     
        # set exToSelString [atomselect $currentMol "$toSelString and ( ( (segname $theSegname and chain $theChain) and ((resid < [expr $theResid - $exDist]) or (resid > [expr $theResid + $exDist]) ) ) or (not (segname $theSegname and chain $theChain) ) ) "]

  tlPutsDebug "starting calcNativeContacts" 

  #iterate over atoms in sel, find contacts  in refFrame, find list of contacted atoms 
  # =later, make these more than N away in sequence
  set dataName(vals) "nat. contacts"
  #set contactString  "name CA" 
  #match protein and nucleic acids and any coarse grain central atoms
  set selListMember 0
  set fromSelCount 0
  #first, set up table of native conacts for each 'from' atom

  if {[catch {set toContactSel  [atomselect $currentMol "($toSel) and $keyAtomSelString"]} ]} {
    tk_messageBox -message "Timeline can't parse the To selection list.  Please enter a selection like: resid 10 to 20"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
    progressBoxDestroy
    return
  } 

  #parse fromSelList
  set validSelList "" 
  if {[catch {set listLength [llength $fromSelList]}]} {
    tk_messageBox -message "Timeline can't parse the From selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
    progressBoxDestroy
    return
  }  
  if {$listLength==0} {
    tk_messageBox -message "Timeline can't find a from selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
    progressBoxDestroy
    return
  }

  foreach selString $fromSelList {
    set selNum 0 
    set sel ""
    tlPutsDebug "selString is >$selString<"
    if {[catch {set sel [atomselect $currentMol "$selString"]; set selNum [$sel num] } ] } {
      tk_messageBox -message "Timeline can't make atom selection for: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
    } else {
      if {$selNum > 0} {
        lappend validSelList $selString
        tlPutsDebug "validSelList: $validSelList validSelList length:[llength $validSelList]"
      } else { 
        tlPutsDebug "num of sel is [$sel num]"
        tk_messageBox -message "Timeline can't find atoms for selection: \"$selString\"" -parent $w -icon warning -type ok -title "VMD Timeline warning"
      }
    }
  }
  set lengthVsl [llength $validSelList]

  if {$lengthVsl <= 0} {
    tk_messageBox -message "Timeline can't find a from selection list.  Please enter a list like: {resid 1} {resid 2}"  -parent $w -icon warning -type ok -title "VMD Timeline warning"
    progressBoxDestroy
    return
  }
   
  foreach fromSel $validSelList {
    set fromContactSel [atomselect $currentMol "($fromSel) and $keyAtomSelString"]
    $fromContactSel frame $refFrame 
    $toContactSel frame $refFrame
    set fromList [$fromContactSel get index] 
    #tlPutsDebug "fromList is >$fromList<"
    #a is for atom index 
    #clear fromTable
    foreach a $fromList {
        set nat($a,reference) 0
        set fromTable($a) "" 
    }
    foreach {fromResult toResult} [measure contacts $cutoff $fromContactSel $toContactSel] {}
    foreach f $fromResult t $toResult {
      lappend fromTable($f) $t
      #every atom is a good contact
      incr  nat($f,reference) 
      #tlPutsDebug "appending $t to Table $a -- fromTable($f) is now $fromTable($f)"
    }

    #second, loop over frames, and in every frame check preserved contacts for each 'from' atom
    for {set theFrame $firstAnalysisFrame} {$theFrame <= $lastAnalysisFrame} {incr theFrame} {
     #set frame in selections
     $fromContactSel frame $theFrame
     $toContactSel frame $theFrame  
     #clear currFromTable
     foreach a $fromList {
       set currFromTable($a) "" 
     }
   
     foreach {fromResult toResult} [measure contacts $cutoff $fromContactSel $toContactSel]  {} 
     foreach f $fromResult t $toResult {
       lappend currFromTable($f) $t
      #puts "frame $theFrame, lappending $t to Curr. Table $f -- currFromTable($f) is now $currFromTable($f)"
    }
    
    #puts "fromList is $fromList"
    #puts $fromTable(950)
    #now find number of items fromTable(a) and currFromTable(a) have in common
      foreach a $fromList {
        set nat($a,$theFrame) 0
        set l $fromTable($a)
        #loop over contacted atoms
        foreach elem $l {
          if ([lsearch -exact $currFromTable($a) $elem]!=-1) {
            incr nat($a,$theFrame)
            #puts "** set nat($a,$theFrame) to $nat($a,$theFrame)\n from= $fromTable($a) currFrom= $currFromTable($a)\n elem= $elem a= $a"
          } 
        }
       #puts "*** Loop done.  nat($a,$theFrame) to $nat($a,$theFrame)\n from= $fromTable($a) currFrom= $currFromTable($a)\n elem= $elem a= $a theFrame= $theFrame"
      }
    }
  #here or elsewhere, make list of user selections of non-zero fromTable members, Or watch our for zero division if include zero contact. 
  
      
       
      set dataVal(freeSelLabel,$selListMember) $fromSel 
      set dataVal(freeSelString,$selListMember) $fromSel 
      set refTot 0
      foreach a $fromList {
        set refTot [expr $refTot + $nat($a,reference)]
      }
      if {$refTot <=0} {set refTot 1}
      for {set theFrame $firstAnalysisFrame} {$theFrame <=$lastAnalysisFrame} {incr theFrame} { 
        set tot 0
        foreach a $fromList {
          set tot [expr $tot + $nat($a,$theFrame)]
        }
        set curField [expr $dataOrigin + $theFrame]
        #puts "frame $theFrame dataVal($curField,$selListMember= $dataVal($curField,$selListMember)" 
        #raw numbers are double counting, but we divide, so no problem
        set val [expr $tot/(0.0+ $refTot)]
        set dataVal($curField,$selListMember) $val
        checkRangeLimits $dataVal($curField,$selListMember)
     }
    set dataValNum $selListMember
    incr selListMember
    $fromContactSel delete

    if {$progressBoxExists} {
       progressBoxConfigure  [progressFindFrac 0 1.0  $progressCalcFraction $selListMember  $lengthVsl] 
     }
    incr progCount 
    if {$progressCancelPressed} break 
  }
  if {$progressCancelPressed} {
    #XXX clear records to function entry state
    set progressCancelPressed 0
    clearData
  }
  configureSelInfo null 0
}


proc ::timeline::copyUser {} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numTrajFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable firstAnalysisFrame 
  variable lastAnalysisFrame
  variable whichUserField
  
  tlPutsDebug "starting copyUser"
  # add option to GUI to create a rep that shows user data? including e.g. mol scaleminmax 0 3 auto 
  #   or popup that says 'copied to userN, make a rep yourself"
  # add option to GUI for user, user2,user3,user4
  # add option to GUI to copy over data after each calc?

  set selAll [atomselect $currentMol "all"]
  for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
     $selAll frame $trajFrame
     $selAll set $whichUserField 0
  } 
  $selAll delete

  
  for {set i 0} {$i <= $dataValNum} {incr i} {
    if {$usesFreeSelection} {
     set sel [atomselect $currentMol $dataVal(freeSelString,$i)]
    } else { 
     set aResid $dataVal(resid,$i)
     set aChain $dataVal(chain,$i)
     set aSegname $dataVal(segname,$i)
     set sel [atomselect $currentMol "resid $aResid and chain $aChain and segname $aSegname"]
    }
    for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      $sel frame $trajFrame
      $sel set $whichUserField $dataVal($curField,$i)
    }
    $sel delete
  }
}
