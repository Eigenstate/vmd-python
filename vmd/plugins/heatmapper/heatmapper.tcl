##
## Heatmap plotting tool v1.0 (color-coded 3D plots)
##
## $Id: heatmapper.tcl,v 1.5 2013/04/15 15:52:05 johns Exp $
##
## Tcl VMD extension
##
## Authors: Anssi Nurminen, Sampo Kukkurainen, Laurie S. Kaguni, Vesa P. Hytönen
## Institute of Biomedical Technology
## University of Tampere
## Tampere, Finland
## and
## BioMediTech, Tampere, Finland
## 30-01-2012 
##
## email: anssi.nurminen_a_uta.fi
##
## Special thanks to Sampo Kukkurainen for the default color-codings
##
##
## Use Tcl command [heatmapper -loadfile filename] to launch heatmapper and read in 
## a heatmapper formatted text file
##
## Accepted command line arguments:
## title, xlabel, ylabel, xorigin, xstep, min, max, colorset, loadfile, hide, show, numbering, callback
##
## Callback example:
##    %  set hmhandle [heatmapper -loadfile heatmap.hm]
##    %  proc click {args} {puts "x=[lindex $args 0] y=[lindex $args 1] val=[lindex $args 2]"}
##    %  $hmhandle configure -callback click
## Clicking on the plot will pass 3 arguments to callback function, e.g. `click x y value`.
## `x` and `y` are indices of the data point both starting from 0.
## Following line will unset the callback function:
##    % $hmhandle configure -callback ""  

package provide heatmapper 1.1


namespace eval ::HeatMapper:: {
   proc initialize {} {      
      variable mapcount 0
      # following is used for naming images created with heatmapper
      variable imgcount 0
   }
   initialize
}

   
proc heatmapper { args } {
    
   set handle [::HeatMapper::InitPlot $args]
   return $handle 

}
   


proc ::HeatMapper::InitPlot {args} {
    
    variable mapcount
    incr ::HeatMapper::mapcount
    set new_ns "::HeatMapper::Plot${::HeatMapper::mapcount}"    
    
    if {[namespace exists $new_ns]} {
        #puts "Reinitializing namespace $new_ns."
    } else {
        #puts "Creating namespace $new_ns"
    }
    
    # DEBUG
    #foreach arg $args {
    #    puts "enter ARG: $arg"
    #}
    
    
    namespace eval $new_ns {
        
        variable fileToLoad ""
                        
        variable parent
        variable verbose 0
        variable iZoomFactor 4
        
        variable iLineCount 0
        variable iFrameCount 0
        
        variable iMinEntry 0.00
        variable iMaxEntry 100.00
        
        variable iCTH_min 0.0
        variable iCTH_max 100.0
        variable iCTH_step 10.0
        variable iColorset 0
        variable iColors [list]
        
        variable iData
        variable iShow
        
        variable iLastMouseX -1
        variable iLastMouseY -1
        
        variable iYNumbering 0
        
        # Contains keys: title, xlabel, ylabel, xorigin, yorigin, callback
        variable iMapInfo      
        
        array set iData {}
        array set iDataMappingY {}        
        array set iMapInfo {}
        
        set iMapInfo(title) "Heatmap"
        set iMapInfo(xlabel) ""
        set iMapInfo(ylabel) ""
        set iMapInfo(xorigin) 0
        set iMapInfo(numbering,0) "unknown"
        set iMapInfo(callback) ""        
        set iMapInfo(shape) ""        
        
        #set iMapInfo(yorigin) 1
        set iMapInfo(loaded) 0
        set iMapInfo(xstep) 1
        
        set iMapDefault(title) "Heatmap"
        set iMapDefault(xlabel) ""
        set iMapDefault(ylabel) ""
        set iMapDefault(xorigin) 0
        set iMapDefault(loaded) 0
        set iMapDefault(xstep) 1
        set iMapDefault(callback) ""
        set iMapDefault(shape) ""
        
        
        array set iShow {}
        
        set iShow(title) 1
        set iShow(legend) 1
        set iShow(ylabel) 1
        set iShow(xlabel) 1
        set iShow(yunits) 1
        set iShow(xunits) 1   
        
        
        
        variable iW .mapwindow${::HeatMapper::mapcount}
        

        
        proc quit { } {
            variable iW
            
            #puts "quit! $iW"
            [namespace current]::ReleaseAll
            destroy $iW
            namespace delete [namespace current]
            return
        }        

        proc plothandle { aCommand args  } {
            variable iW
            
            #puts "plothandle: $aCommand $args"
            
            switch $aCommand {
                configure { [namespace current]::ProcessCommandLineArguments yes $args; return }
                cget { return [[namespace current]::GetCommandLineArguments $args]; }
                list { return [[namespace current]::ListCommandLineArguments]; }
                quit { [namespace current]::quit; return }
            }
        }        
        
        
        catch {destroy $iW}
        
        toplevel $iW
        wm title $iW "HeatMapper ${::HeatMapper::mapcount}"
        wm iconname $iW "HM ${::HeatMapper::mapcount}"
        wm protocol $iW WM_DELETE_WINDOW "[namespace current]::quit"
        #wm withdraw $iW        
    
        

        

        
        ##
        ## Outlook
        ##  
        option add *heatmapper.*borderWidth 1
        option add *heatmapper.*Button.padY 0
        option add *heatmapper.*Menubutton.padY 0  
        
        
        ##
        ## Menubar
        ##
        frame $iW.menubar -relief raised -bd 2
        pack $iW.menubar -fill x
        
        #
        # File
        #
        menubutton $iW.menubar.file -text "File" -menu $iW.menubar.file.menu -underline 0 -pady 2
        menu $iW.menubar.file.menu -tearoff no
        $iW.menubar.file.menu add command -label "Load plot..." -command "[namespace current]::LoadFile"   
        $iW.menubar.file.menu add command -label "Save plot as Postscript..." -command "[namespace current]::PostScriptify"
        $iW.menubar.file.menu add command -label "Save heatmap data as text..." -command "[namespace current]::SaveAsText"
        $iW.menubar.file.menu add command -label "Save heatmap data as gif..." -command "[namespace current]::SaveAsImage"
        
        pack $iW.menubar.file -side left
        
        # TODO implement file menu options        
        $iW.menubar.file.menu entryconfigure 1 -state disabled
        $iW.menubar.file.menu entryconfigure 2 -state disabled
        $iW.menubar.file.menu entryconfigure 3 -state disabled
        
        #
        # Operations
        #
        menubutton $iW.menubar.operations -text "Operations" -menu $iW.menubar.operations.menu -underline 0 -pady 2
        menu $iW.menubar.operations.menu -tearoff no
        
        $iW.menubar.operations.menu add cascade -label "Load comparison heatmap..." -menu $iW.menubar.operations.menu.comparison -underline 0
        menu $iW.menubar.operations.menu.comparison -tearoff no
        
        $iW.menubar.operations.menu.comparison add command -label "Subtract" -command "[namespace current]::LoadComparison sub"
        $iW.menubar.operations.menu.comparison add command -label "Add" -command "[namespace current]::LoadComparison add"
        $iW.menubar.operations.menu.comparison add command -label "Multiply" -command "[namespace current]::LoadComparison mul"
                
        $iW.menubar.operations.menu add command -label "Apply expression..." -command "[namespace current]::ApplyExpression"

        pack $iW.menubar.operations -side left        
        
        
        #
        # Tools menu
        #
        menubutton $iW.menubar.tools -text "Tools" -menu $iW.menubar.tools.menu -underline 0 -pady 2
        menu $iW.menubar.tools.menu -tearoff no
        
        $iW.menubar.tools.menu add cascade -label "Zoom factor" -menu $iW.menubar.tools.menu.zoom -underline 0
        menu $iW.menubar.tools.menu.zoom -tearoff no
        $iW.menubar.tools.menu.zoom add radiobutton -label "x1" -command "[namespace current]::SetZoom 1" -variable [namespace current]::iZoomFactor -value 1
        $iW.menubar.tools.menu.zoom add radiobutton -label "x2" -command "[namespace current]::SetZoom 2" -variable [namespace current]::iZoomFactor -value 2
        $iW.menubar.tools.menu.zoom add radiobutton -label "x4" -command "[namespace current]::SetZoom 4" -variable [namespace current]::iZoomFactor -value 4
        $iW.menubar.tools.menu.zoom add radiobutton -label "x8" -command "[namespace current]::SetZoom 8" -variable [namespace current]::iZoomFactor -value 8
        
        $iW.menubar.tools.menu add cascade -label "Colorset" -menu $iW.menubar.tools.menu.colorset -underline 0
        menu $iW.menubar.tools.menu.colorset -tearoff no
        $iW.menubar.tools.menu.colorset add radiobutton -label "default" -command "[namespace current]::SetColorSet 0" -variable [namespace current]::iColorset -value 0
        $iW.menubar.tools.menu.colorset add radiobutton -label "Black&White" -command "[namespace current]::SetColorSet 1" -variable [namespace current]::iColorset -value 1    
        $iW.menubar.tools.menu.colorset add radiobutton -label "yellow-brown" -command "[namespace current]::SetColorSet 2" -variable [namespace current]::iColorset -value 2
        $iW.menubar.tools.menu.colorset add radiobutton -label "blue-white-red" -command "[namespace current]::SetColorSet 3" -variable [namespace current]::iColorset -value 3
            
        $iW.menubar.tools.menu add checkbutton -label "Swap axes" -variable [namespace current]::stats -underline 1
        
        $iW.menubar.tools.menu add cascade -label "Y-Axis numbering" -menu $iW.menubar.tools.menu.ynumbering -underline 0
        menu $iW.menubar.tools.menu.ynumbering -tearoff no
        $iW.menubar.tools.menu entryconfigure 4 -state disabled        

                    
        
        pack $iW.menubar.tools -side left
        
        # TODO implement tools menu options
        $iW.menubar.tools.menu entryconfigure 2 -state disabled
        
        #
        # Show menu
        #
        menubutton $iW.menubar.show -text "Show" -menu $iW.menubar.show.menu -underline 0 -pady 2
        menu $iW.menubar.show.menu -tearoff no    
        
        $iW.menubar.show.menu add checkbutton -label "Title" -variable [namespace current]::iShow(title) -command "[namespace current]::Refresh"
        $iW.menubar.show.menu add checkbutton -label "Legend" -variable [namespace current]::iShow(legend) -command "[namespace current]::Refresh"
        $iW.menubar.show.menu add checkbutton -label "Y-axis label" -variable [namespace current]::iShow(ylabel) -command "[namespace current]::Refresh"
        $iW.menubar.show.menu add checkbutton -label "Y-axis units" -variable [namespace current]::iShow(yunits) -command "[namespace current]::Refresh"
        $iW.menubar.show.menu add checkbutton -label "X-axis label" -variable [namespace current]::iShow(xlabel) -command "[namespace current]::Refresh"
        $iW.menubar.show.menu add checkbutton -label "X-axis units" -variable [namespace current]::iShow(xunits) -command "[namespace current]::Refresh"
        
        pack $iW.menubar.show -side left             
        
        # TODO implement show menu options
        $iW.menubar.show.menu entryconfigure 2 -state disabled    
        $iW.menubar.show.menu entryconfigure 3 -state disabled
        $iW.menubar.show.menu entryconfigure 4 -state disabled
        $iW.menubar.show.menu entryconfigure 5 -state disabled
        
        
        # Help
        menubutton $iW.menubar.help -text "Help" -menu $iW.menubar.help.menu -underline 0 -pady 2 
        menu $iW.menubar.help.menu -tearoff no
        set help_file [join [list "file://" $::env(HEATMAPPERDIR) "/documentation/" "index.html"] ""]
        $iW.menubar.help.menu add command -label "About" -command [namespace current]::HelpAbout
        $iW.menubar.help.menu add command -label "Help..." -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/heatmapper" 
#        $iW.menubar.help.menu add command -label "Help..." -command "vmd_open_url \"$help_file\""
        pack $iW.menubar.help -side right  
        
        
        
        # top frame
        #frame $iW.top
        #pack $iW.top -side top -fill x         
        
        
       
        #
        # Canvas
        #
        
        frame $iW.c
        
        scrollbar $iW.c.xscroll -orient horizontal -command [list $iW.c.canvas xview]
        scrollbar $iW.c.yscroll -orient vertical -command [list $iW.c.canvas yview]
        
        canvas $iW.c.canvas -xscrollcommand [list $iW.c.xscroll set] -yscrollcommand [list $iW.c.yscroll set] \
              				-highlightthickness 0 -borderwidth 0 -highlightthickness 0 -borderwidth 0 \
              				-background white -width 370 -height 370
        
        

        
        grid $iW.c.canvas $iW.c.yscroll -in $iW.c -sticky news
        grid $iW.c.xscroll -sticky ew -in $iW.c
        grid rowconfigure $iW.c 0 -weight 1
        grid columnconfigure $iW.c 0 -weight 1               

    
        set iMapImage [image create photo "hmimage$::HeatMapper::imgcount" -height 100 -width 200 -palette 256/256/256]
        incr ::HeatMapper::imgcount
        set iBgImage [image create photo "hmimage$::HeatMapper::imgcount" -height 370 -width 370 -palette 256/256/256]
        incr ::HeatMapper::imgcount
         
        set iMapImageId [$iW.c.canvas create image 10 10 -anchor nw -image $iMapImage]
        set iBgImageId [$iW.c.canvas create image 0 0 -anchor nw -image $iBgImage]
        
        #raise $scrolled_canvas
        bind $iW.c.canvas <1> "[namespace current]::MouseClick %x %y"
        bind $iW.c.canvas <B1-Motion> "[namespace current]::MouseDrag %x %y" 
        bind $iW <MouseWheel> "[namespace current]::Wheelie %x %y %D" 
        #bind $iW <MouseWheel> "[namespace current]::Wheelie %x %y -5"

        
        #pack $iW.top -side top -fill x -anchor nw
        
        
        #
        # Bottom controls
        #
    
        frame $iW.newframe -relief raised
        pack $iW.newframe -side bottom -fill x -anchor sw
        
        # Colorsets
        labelframe $iW.newframe.colorset -text "Color set" -relief ridge -bd 2 -padx 5 -pady 4
        pack $iW.newframe.colorset -side left -expand yes
        
        ttk::combobox $iW.newframe.colorset.combo -state readonly -width 13 -values [list "default" "B&W" "yellow-brown" "blue-w-red"]  
        pack $iW.newframe.colorset.combo -side left -fill x -expand yes
        $iW.newframe.colorset.combo current 0
        bind $iW.newframe.colorset.combo <<ComboboxSelected>> "[namespace current]::SetColorSet"
        
        # Zoom
        labelframe $iW.newframe.zoomset -text "Zoom" -relief ridge -bd 2 -padx 5 -pady 4
        pack $iW.newframe.zoomset -side left -expand yes
        
        ttk::combobox $iW.newframe.zoomset.combo -state readonly -width 3 -values [list x1 x2 x4 x8]  
        pack $iW.newframe.zoomset.combo -side left -fill x -expand yes   
        
        $iW.newframe.zoomset.combo current [lindex [list 0 0 1 1 2 2 2 2 3] $iZoomFactor]    
        bind $iW.newframe.zoomset.combo <<ComboboxSelected>> [namespace code {SetZoom [lindex [list 1 2 4 8] [%W current]]}]
        
        # Thresholds
        labelframe $iW.newframe.right -text "Thresholds" -relief ridge -bd 2 -padx 5 -pady 4
        pack $iW.newframe.right -side left -expand yes
        
        label $iW.newframe.right.min_label -text "Min:"
        entry $iW.newframe.right.min -width 6 -textvariable [namespace current]::iMinEntry
        bind $iW.newframe.right.min <Return> "[namespace current]::ValidateAndApplyThresholds"
        
        label $iW.newframe.right.max_label -text "Max:"
        entry $iW.newframe.right.max -width 6 -textvariable [namespace current]::iMaxEntry    
        bind $iW.newframe.right.max <Return> "[namespace current]::ValidateAndApplyThresholds"
        
        button $iW.newframe.right.apply_btn -text "Apply" -relief raised -command [namespace current]::ValidateAndApplyThresholds
        
        pack $iW.newframe.right.min_label $iW.newframe.right.min $iW.newframe.right.max_label $iW.newframe.right.max $iW.newframe.right.apply_btn -side left
        pack $iW.newframe.right.apply_btn -padx 3 -side left
       
        

        grid configure $iW.menubar -in $iW -column 0 -row 0 -sticky ew
		grid configure $iW.c -in $iW -column 0 -row 1 -sticky nsew
		grid configure $iW.newframe -in $iW -column 0 -row 2 -sticky nsew
        
        grid rowconfigure $iW 1 -weight 1
        grid columnconfigure $iW 0 -weight 1
		
		
        update
        #wm minsize $iW [winfo width $iW] [winfo height $iW]
        wm maxsize $iW [winfo width $iW] [winfo height $iW]
        wm minsize $iW [winfo width $iW] 120
        #wm maxsize $iW 300 120
        wm resizable $iW 1 1      
        
    
        # Keep printing heatmap values at points under mouse cursor
        proc Wheelie { aX aY aDelta } {
        
            variable iW
            
            #puts "Wheelie $aX $aY $aDelta"
            
            if { $aDelta > 0 } {
                $iW.c.canvas yview scroll -1 units
            } else {
                $iW.c.canvas yview scroll 1 units
            }  
        }        
        
        
        # Keep printing heatmap values at points under mouse cursor
        proc MouseDrag { aX aY } {
        
            [namespace current]::MouseClick $aX $aY 1
        }
        
        
        # Get heatmap value at cliked point
        proc MouseClick { aX aY {aDrag 0} } {
            
            variable iCanvasSizeArray
            variable iZoomFactor
            variable iData
            variable iDataMappingY
            variable iMapInfo
            
            variable iLastMouseX
            variable iLastMouseY
            
            variable iW
            
            # Convert window coordinates to scrolling canvas coordinates
            # xview & yview return two flotaing point numbers that describe the visible canvas area
            set x_list [$iW.c.canvas xview]
            set y_list [$iW.c.canvas yview]
            set scroll_reg [$iW.c.canvas cget -scrollregion]
            
            
            set first_visible_x [expr [lindex $x_list 0] * [lindex $scroll_reg 2]]            
            set first_visible_x [expr int(floor($first_visible_x))]
            
            set first_visible_y [expr [lindex $y_list 0] * [lindex $scroll_reg 3]]            
            set first_visible_y [expr int(floor($first_visible_y))]            
 
            set aX [expr $first_visible_x + $aX]
            set aY [expr $first_visible_y + $aY]
            
            # DEBUG
            #puts "Click: X: $aX -> $first_visible_x, Y: $aY -> $first_visible_y"
            #puts "Xlist: $x_list Scroll_reg: $scroll_reg"

            if { !$iMapInfo(loaded) } { return }
            
            if { $iCanvasSizeArray(heatmap,w) < 1 || $iCanvasSizeArray(heatmap,h) < 1 } {        
                return        
            }
            
            # Convert screen pixel coordinates into heatmap coordinates
            # pixel coordinates 0,0 is window nw corner
            set origo_x $iCanvasSizeArray(y_axis,w)
            set origo_y [expr $iCanvasSizeArray(title,h) + $iCanvasSizeArray(heatmap,h) - 1]
            
            # indices start from 0
            # NMWiz callback support
            set ind_fr_z [expr ($aX - $origo_x) / $iZoomFactor]
            set ind_fr [expr $ind_fr_z + $iMapInfo(xorigin)]
            set ind_rs [expr ($origo_y - $aY) / $iZoomFactor]

            # Print each datapoint only once
            if { $aDrag == 1 && $ind_fr == $iLastMouseX && $ind_rs == $iLastMouseY } {
                # Dragging and this data point has already been printed
                return
            }
            
            # NMWiz callback support
            set val ""            
            # Convert heatmap row into resid
            if { [info exists iDataMappingY(0,$ind_rs)] } {
                #set ind_rs $iDataMappingY(0,$ind_rs)
                        
                if { ![info exists iData($ind_fr,$ind_rs)] } {
                    # puts "No data at x:$aX y:$aY    ($ind_fr,$ind_rs) (fr,rs)"
                } else {    
                    
                    set extras ""
                    set e 1
                    while { [info exists iMapInfo($e,numbering)] && [info exists iDataMappingY($e,$ind_rs)] } {
                        set extras [join [concat $extras " " $iMapInfo($e,numbering) ":"  $iDataMappingY($e,$ind_rs)] ""]
                        incr e
                    }
                                        
                    set val [format %.6f $iData($ind_fr,$ind_rs)]
                    puts "$iMapInfo(0,numbering):$iDataMappingY(0,$ind_rs) f:[expr $ind_fr * $iMapInfo(xstep)]  val:$val  $extras"
                    
                    # DEBUG
                    # puts "Color: [[namespace current]::GetColor $iData($ind_fr,$ind_rs)]"
                }        
                
            } 
            
            # NMWiz callback support
            if { [string length $iMapInfo(callback)]} {
              eval $iMapInfo(callback) $ind_fr_z $ind_rs $val
            }

            set iLastMouseX $ind_fr
            set iLastMouseY $ind_rs
        }
        
        
        proc SetYNumberingMenu { } {
            
            variable iW
            variable iYNumbering
            variable iMapInfo
            
            set e 0
            
            # DEBUG
            # puts "SetYNumberingMenu"
            
            # Remove old
            $iW.menubar.tools.menu.ynumbering delete 0 last
            
            while { [info exists iMapInfo($e,numbering)] } {
                
                set title [string totitle $iMapInfo($e,numbering)];
                # puts "Add item $title"
                $iW.menubar.tools.menu.ynumbering add radiobutton -label $title -command [namespace current]::SetYLabel -variable [namespace current]::iYNumbering -value $e
                incr e
            }         
            
            if { $e == 0 } {
                $iW.menubar.tools.menu entryconfigure 4 -state disabled
            } else {
                $iW.menubar.tools.menu entryconfigure 4 -state normal
            }
        }
        
        proc SetYLabel { } {
            
            variable iW
            variable iYNumbering
            variable iMapInfo
            
            # puts "SetYLabel: iYNumbering: $iYNumbering"
            
            set iMapInfo(ylabel) [string totitle $iMapInfo($iYNumbering,numbering)]  
            [namespace current]::Refresh;
        }
        
        
        proc ValidateAndApplyThresholds { {aRefresh yes} } {
            
            variable iCTH_min
            variable iCTH_max
            
            variable iMinEntry
            variable iMaxEntry   
            
            variable iMapInfo
            
            # Make sure "," -> "." and set variables
            set iCTH_min [format "%f" [string map [list "," "."] $iMinEntry]]
            set iCTH_max [format "%f" [string map [list "," "."] $iMaxEntry]]
            
            if { $iCTH_min > $iCTH_max } {
                # swap values
                set temp $iCTH_min
                set $iCTH_min $iCTH_max
                set $iCTH_max $temp  
            }
            
            # Set decimals to 2
            set iMinEntry [format %.2f $iCTH_min]
            set iMaxEntry [format %.2f $iCTH_max]
            
            set iMapInfo(min) $iMinEntry
            set iMapInfo(max) $iMaxEntry
            
            # Update color thresholds and heatmap
            [namespace current]::SetColors
            
            if { $aRefresh } { [namespace current]::Refresh }
        }
        
        proc SetColorSet { {aSet -1}} {
            
            variable iW
                     
            if { $aSet == -1 } {
                set aSet [$iW.newframe.colorset.combo current]
            } else {
                # Set from tools menu
                $iW.newframe.colorset.combo current $aSet
            }
            
            [namespace current]::SetColors $aSet
            [namespace current]::Refresh
        }
        
        proc SetZoom { aZoom } {
            
            variable iW
            variable iZoomFactor
            
            set iZoomFactor $aZoom
            
            #Update GUI ComboBox    
            $iW.newframe.zoomset.combo current [lindex [list 0 0 1 1 2 2 2 2 3] $aZoom]
            
            [namespace current]::Refresh
            wm geometry $iW
        }
        
        proc Refresh { args } {
            
            variable iW
            variable iZoomFactor
            
            variable iFrameCount
            variable iLineCount    
            
            variable iMapImage
            variable iMapImageId
            
            variable iBgImage
            variable iBgImageId 
            
            variable iCanvasSizeArray
            
            #puts "Refresh args: $args"
            #puts "iFrameCount $iFrameCount"
            #puts "iLineCount $iLineCount"
            
            if { $iFrameCount <= 0 || $iLineCount <= 0 } { 
                # No data to plot
                return 
            }
            
            if { ![array exists iCanvasSizeArray] } {
                array set iCanvasSizeArray {}
            } else {
                # clear array
                array unset iCanvasSizeArray *
            }
            
            #|         TITLE         |
            #|------------------------
            #| Y |               | L |
            #| | |     MAP       | E |
            #| A |               | G |
            #| X |               | E |
            #| I |               | N |
            #| S |               | D |
            #|------------------------
            #|     X-AXIS Space      |    
            
            
            set map_w [expr $iZoomFactor * $iFrameCount]
            set map_h [expr $iZoomFactor * $iLineCount]    
            
            if { $map_h < 80 } { set map_h 80 }
            
            set iCanvasSizeArray(heatmap,w) $map_w
            set iCanvasSizeArray(heatmap,h) $map_h
            
            set scrolled_canvas $iW.c.canvas
            
            # Reset image and remove it from canvas for faster drawing
            $scrolled_canvas delete $iMapImageId
            image delete $iMapImage
            set iMapImage [image create photo "hmimage$::HeatMapper::imgcount" -height $map_h -width $map_w -palette 256/256/256]    
            incr ::HeatMapper::imgcount
            #
            # Plot heatmap!
            #
            [namespace current]::PlotData
            
            # Resets and redraws all canvas elements except the heatmap plot
            # DrawBg also resets and fills iCanvasSizeArray that contains sizes for all canvas
            # elements
            [namespace current]::DrawBg
            
              

            # Curb window max height a little to prevent window title going out of screen
            #set screen_w [winfo screenwidth $iW]            
            set screen_h [expr [winfo screenheight $iW] - 40]
            
            # Resize canvas
            set iCanvasSizeArray(canvas,w) [expr $iCanvasSizeArray(y_axis,w) + $iCanvasSizeArray(heatmap,w) + $iCanvasSizeArray(legend,w)]
            set iCanvasSizeArray(canvas,h) [expr $iCanvasSizeArray(title,h) + $iCanvasSizeArray(heatmap,h) + $iCanvasSizeArray(x_axis,h)]
            #set iCanvasSizeArray(canvas,h) 300
            
            #set max_w iCanvasSizeArray(canvas,w)

            set visible_canvas_h $iCanvasSizeArray(canvas,h)
            set h_maxed ""
            
            if { $visible_canvas_h > [expr $screen_h - 100] } {
	            set visible_canvas_h [expr $screen_h - 100]
	            set h_maxed "+0+0"
            }
            
            #puts "ScreenH: $screen_h"
            #puts "canvasH: $iCanvasSizeArray(canvas,h)"
            #puts "VisibleH: $visible_canvas_h"
            
            #$scrolled_canvas configure -width $iCanvasSizeArray(canvas,w) -height $iCanvasSizeArray(canvas,h)
            $scrolled_canvas configure -width $iCanvasSizeArray(canvas,w) -height $visible_canvas_h
            $scrolled_canvas configure -scrollregion "0 0 $iCanvasSizeArray(canvas,w) $iCanvasSizeArray(canvas,h)"
            
            #puts "Heatmap placing: $iCanvasSizeArray(y_axis,w),$iCanvasSizeArray(y_axis,h)"
            
            # Set image to canvas            
            set iMapImageId [$scrolled_canvas create image $iCanvasSizeArray(y_axis,w) $iCanvasSizeArray(title,h) -anchor nw -image $iMapImage]
            
            
            #$scrolled_canvas lower $iBgImageId
            #pack $scrolled_canvas -fill both -expand false
            #pack $iW.top -side top
            #pack $iW.newframe -side bottom
            
            #wm geometry $iW "$iCanvasSizeArray(canvas,w)x$iCanvasSizeArray(canvas,h)"
            
            
            #if { $max_w > $screen_w } { set max_w $screen_w }
            #set max_h iCanvasSizeArray(canvas,h)
	        #if { $max_h > $screen_h } { set max_h $screen_h }
            
            wm geometry $iW [join [list "[expr $iCanvasSizeArray(canvas,w) + 25]x[expr $visible_canvas_h + 100]" $h_maxed] ""]
            wm maxsize $iW [expr $iCanvasSizeArray(canvas,w) + 25] [expr $visible_canvas_h + 100]
            
            update
            
            #wm minsize $iW [winfo width $iW] [winfo height $iW]
            #wm maxsize $iW [winfo width $iW] [winfo height $iW]
        }
        
        proc DrawBg { } {
            
            variable iW    
            variable iData
            variable iDataMappingY
            variable iYNumbering
            
            variable iCanvasItemArray
            variable iCanvasSizeArray
            variable iZoomFactor
            
            variable iFrameCount
            variable iLineCount      
            
            variable iShow
            variable iMapInfo
            
            variable iMapImage 
            
            if { ![array exists iCanvasItemArray] } {
                array set iCanvasItemArray {}
            }
            
            
            set img_w [image width $iMapImage]
            set img_h [image height $iMapImage]
            #puts "img_w,h: $img_w,$img_h"
            
            #set x_axis_w 40
            #set y_axis_h 20
            
            #set title_h 20
            #set legend_w 60     
            
            # Margins in pixels around title text
            set title_margin 6
            
            # Margins in pixels around axes
            set axis_margin 5
                
            set max_fontsize 12
            
            # How many pixels are the axis crossing each other in origo
            set axis_extension 5
            
            set axis_tick_len 4
            
            #set y_axis_division [expr 10 $iZoomFactor]
            
            #set iCanvasSizeArray(x_axis,h) 20
            #set iCanvasSizeArray(y_axis,w) 40
            
            #set iCanvasSizeArray(title,h) 20
            
            set c $iW.c.canvas
                        
            set canvas_w [$c cget -width]
            set canvas_h [$c cget -height]
            #puts "canvas_w,h: $canvas_w,$canvas_h"
            
            # Delete old canvas items
            foreach item_id [array names iCanvasItemArray] {
                $c delete $iCanvasItemArray($item_id)
            }
            
            #
            # Draw Title
            #    
            #TODO set title max width    
            if { $iShow(title) && [llength [array get iMapInfo title]] && [string length $iMapInfo(title)] } {
            
                set title_font [font create -size 16 -family "Courier"]
                
                set title_width [expr $iCanvasSizeArray(heatmap,w)]
                
                # Enforce minimum width
                if { $title_width < 60 } {
                    set title_width 60
                }
                
                set iCanvasItemArray(title) [$c create text 0 0 -anchor center -font "Courier" -text $iMapInfo(title) -justify center -width [expr $title_width + 10]]    
                
                set bbox [$c bbox $iCanvasItemArray(title)]
                set title_h [expr [lindex $bbox 3] - [lindex $bbox 1]]    
                
                set iCanvasSizeArray(title,h) [expr $title_h + (2 * $title_margin)]
                set iCanvasSizeArray(title,w) $iCanvasSizeArray(heatmap,w)        
                

                
                font delete $title_font
                # Note: Title not yet in corect x-position, need to set y_axis width first
            } else {
                # Title hidden
                set iCanvasSizeArray(title,h) 10
                set iCanvasSizeArray(title,w) 10                
            }
           
            # Create font for axis labels
            set axis_font [font create -size 12 -family "Courier"]
            
            
            #
            # Draw Y-axis
            #       
            
            #
            # Create Y-axis fonts font
            #
            set y_ticklabel_fontsize 1    
            if { $iLineCount >= 10 } {
                # Prevent possibility to divide by zero
                set y_ticklabel_fontsize [expr ($iCanvasSizeArray(heatmap,h) / ($iLineCount / 10)) / 4 + 1]    
            } else {
                # Only one ticklabel, use static font size
                set y_ticklabel_fontsize $max_fontsize
            }
            
            # Enforce font max size
            if { $y_ticklabel_fontsize > $max_fontsize } { set y_ticklabel_fontsize 12 }
            
            # Create font for and ticklabels
            set y_tick_font [font create -size $y_ticklabel_fontsize -family "Courier"]
            
         
            set y_axis_origin [expr $iCanvasSizeArray(title,h) + $iCanvasSizeArray(heatmap,h)]
            
            # Test how much space Y-axis ticklabels need
            set max_digits "00"
            if { $iLineCount > 98 } { set max_digits "000" }
            if { $iLineCount > 998 } { set max_digits "0000" }
            
            set temp [$c create text 0 0 -anchor center -font $y_tick_font -text $max_digits]    
            set bbox [$c bbox $temp]
            set temp_w [expr [lindex $bbox 2] - [lindex $bbox 0] + 2]
            $c delete $temp
            
            # Test how much space Y-axis label takes (if any)
            set ylabel_w 0
            
            if { [llength [array get iMapInfo ylabel]] } {
                
                set temp [$c create text 0 0 -anchor e -font $axis_font -text $iMapInfo(ylabel)]    
                set bbox [$c bbox $temp]
                set ylabel_w [expr [lindex $bbox 2] - [lindex $bbox 0]]
                $c delete $temp
            }
            
            set iCanvasSizeArray(y_axis,w) [expr $ylabel_w + $temp_w + (2 * $axis_margin)]
            set iCanvasSizeArray(y_axis,h) $iCanvasSizeArray(heatmap,h)
                    
            # Draw axis line from top to bottom
            set y_axis_xpos [expr $iCanvasSizeArray(y_axis,w) - 1]
            set iCanvasItemArray(y_axis) [$c create line $y_axis_xpos $iCanvasSizeArray(title,h) $y_axis_xpos [expr $y_axis_origin + $axis_extension] -fill "#000"]
                
            
            # Draw axis label
            if { $ylabel_w } {
                set iCanvasItemArray(y_axis,label) [$c create text [expr $y_axis_xpos - $temp_w - $axis_tick_len - 2] \
                                                                   [expr $iCanvasSizeArray(title,h) + ($iCanvasSizeArray(heatmap,h) / 2)] \
                                                                   -anchor e -font $axis_font -text $iMapInfo(ylabel)]
            }    
        
            set y_label 0
            # Create axis labels and ticks
            # Loop from origo upwards
            for {set i $y_axis_origin} {$i > [expr $y_axis_origin - ($iLineCount * $iZoomFactor)]} {incr i [expr -10 * $iZoomFactor]} {
                # Draw Tick
                set iCanvasItemArray(y_axis,line,$i) [$c create line $y_axis_xpos $i [expr $y_axis_xpos - $axis_tick_len] $i -fill "#000"]
                # Draw Ticklabel
                set iCanvasItemArray(y_axis,ticklabel,$i) [$c create text [expr $y_axis_xpos - $axis_tick_len - 2] $i \
                                                           -anchor e -font $y_tick_font -text $iDataMappingY($iYNumbering,[expr ($y_axis_origin - $i) / $iZoomFactor])]        
            }    
            
            
            #
            # Draw Legend  
            #
            if { $iShow(legend) } {    
                set iCanvasSizeArray(legend,w) [[namespace current]::SetLegend [expr $iCanvasSizeArray(y_axis,w) + $iCanvasSizeArray(heatmap,w)] \
                                                                               $iCanvasSizeArray(title,h) \
                                                                               [expr $iCanvasSizeArray(heatmap,h) + 1]]     
                set iCanvasSizeArray(legend,h) [expr $iCanvasSizeArray(heatmap,h) + 1]    
            } else {
                # Legend hidden   
                set iCanvasSizeArray(legend,w) 10
                set iCanvasSizeArray(legend,h) [expr $iCanvasSizeArray(heatmap,h) + 1]    
            }
            
            #set iCanvasSizeArray(legend,w) 100
            
            #
            # Draw X-axis
            #
            
            #
            # Create X-axis fonts font
            #
            set x_ticklabel_fontsize 1    
            if { $iFrameCount >= 10 } {
                # Prevent possibility to divide by zero
                set x_ticklabel_fontsize [expr ($iCanvasSizeArray(heatmap,w) / ($iFrameCount / 10)) / 4 + 1]    
            } else {
                # Only one ticklabel, use static font size
                set x_ticklabel_fontsize $max_fontsize
            }
            
            # Enforce font max size
            if { $x_ticklabel_fontsize > $max_fontsize } { set x_ticklabel_fontsize 12 }
            
            # Create font for and ticklabels
            set x_tick_font [font create -size $x_ticklabel_fontsize -family "Courier"]
                
            
            set xa_ypos [expr $iCanvasSizeArray(title,h) + $iCanvasSizeArray(heatmap,h)]
            
            # Axis line
            set iCanvasItemArray(xaxis) [$c create line [expr $y_axis_xpos - $axis_extension] $xa_ypos [expr $y_axis_xpos + $iCanvasSizeArray(heatmap,w) + 1] $xa_ypos -fill "#000"]
            
            # Draw X-axis label
            if { [llength [array get iMapInfo xlabel]] } {
                
                set iCanvasItemArray(x_axis,label) [$c create text [expr $iCanvasSizeArray(y_axis,w) + ($iCanvasSizeArray(heatmap,w) / 2)] \
                                                                   [expr $xa_ypos + $axis_tick_len + (($max_fontsize * 2) + 8)] \
                                                                   -anchor c -font $axis_font -text $iMapInfo(xlabel)]        
                
            }
            
            #TODO X-axis labels
            set iCanvasSizeArray(x_axis,h) [expr $axis_tick_len + (($max_fontsize * 3) + 8 + 4)]
            set iCanvasSizeArray(x_axis,w) $iCanvasSizeArray(heatmap,w)
            
            set xstep 1
            if { [info exists iMapInfo(xstep)] } { set xstep $iMapInfo(xstep) }
            
            # Draw X-axis ticks and ticklabels
            for {set i $y_axis_xpos} {$i <= [expr $y_axis_xpos + $iCanvasSizeArray(heatmap,w)]} {incr i [expr 10 * $iZoomFactor]} {
                # Draw tick
                set iCanvasItemArray(x_axis,line,$i) [$c create line $i $xa_ypos $i [expr $xa_ypos + $axis_tick_len] -fill "#000"]
                # Draw tick label
                set iCanvasItemArray(x_axis,ticklabel,$i) [$c create text $i [expr $xa_ypos + $axis_tick_len + $x_ticklabel_fontsize + 2] \
                                                          -anchor c -font $x_tick_font -text [expr $iMapInfo(xorigin) + (($i - $y_axis_xpos) / $iZoomFactor) * $xstep]]    
            }     
            
            
            #set title_bbox [$c bbox $iCanvasItemArray(title)]    
            #$c move $iCanvasItemArray(title) [expr ] 0
            
            
            
            #set iCanvasItemArray(yaxis) [$c create line [expr $x_axis_w - 1] $title_h [expr $x_axis_w - 1] [expr $xa_ypos + 5] -fill "#000"]
            #set iCanvasItemArray(xaxis) [$c create line [expr $x_axis_w - 5] $xa_ypos [expr $x_axis_w + $img_w] $xa_ypos -fill "#000"]
            
            #set iCanvasItemArray(legend) [$c create rect [expr $x_axis_w + $img_w + 7] $title_h [expr $x_axis_w + $img_w + 18] [expr $img_h + 1] -outline "#000"]
            
            #puts "LegendW: $iCanvasSizeArray(legend,w)"
            
            #set title x-pos
                
            $c move $iCanvasItemArray(title) [expr ($iCanvasSizeArray(y_axis,w) + $iCanvasSizeArray(heatmap,w) + $iCanvasSizeArray(legend,w)) / 2] \
                                             [expr $iCanvasSizeArray(title,h) / 2]    
            
            font delete $axis_font
        }
        
        proc SetLegend { aXpos aYpos aHeight } {
            
            variable iW        
            variable iCanvasItemArray
            variable iCanvasSizeArray
            
            variable iCTH_min
            variable iCTH_max
            variable iCTH_step   
            variable iColors
                
            set c $iW.c.canvas
            
            # Static widths
            set map_margin 10
            set box_w 20
            set margin 3
            
            
            
            set aXpos [expr $aXpos + $map_margin]
            
            if { ![llength $iColors] } { return }
            
            set colorbox_h [expr $aHeight / [llength $iColors]]
            
            #Center vertically to map image
            set aYpos [expr $aYpos + (($aHeight % [llength $iColors])/2)]
            
            
            #puts "Number of colors used: [llength $iColors]"    
            
            
            #Legend box outline
            #set iCanvasItemArray(legend) [$c create rect $aXpos $aYpos [expr $aXpos + $legend_w] [expr $aYpos + $aHeight] -outline "#000"]
            
            set fontsize [expr ($colorbox_h / 2)]    
            if { $fontsize > 12 } { set fontsize 12 }
            #if { $fontsize < 7 } { set fontsize 7 }
            
            set fontname [font create -size $fontsize -family "Courier"]    
            
            #puts "Min: $iCTH_min, Step: $iCTH_step, Max: $iCTH_max"
            set color_num 0
            
            #Create all different colored boxes
            foreach color $iColors {
                
                set box_y1 [expr $aYpos + ($color_num * $colorbox_h)]
                set box_y2 [expr $box_y1 + $colorbox_h]
        
                #Colored boxes
                set iCanvasItemArray(legend,color,$color_num) [$c create rect $aXpos $box_y1 [expr $aXpos + $box_w] $box_y2 -outline "#000" -fill [[namespace current]::HexColor $color]]
                
                set treshold_num [expr $iCTH_min + (($color_num) * $iCTH_step)]
        
                #cut to 2 decimals
                set threshold_label [format %.2f $treshold_num]
                set iCanvasItemArray(legend,label,$color_num) [$c create text [expr $aXpos + $box_w + $margin] $box_y1 -anchor w -font $fontname -text $threshold_label]    
               
                
                
                incr color_num
            }
            
            #last label (MAX)
            set iCanvasItemArray(legend,label,$color_num) [$c create text [expr $aXpos + $box_w + 2] [expr $box_y1 + $colorbox_h] -anchor w -font $fontname -text [format %.2f $iCTH_max]]     
            
            set bbox [$c bbox $iCanvasItemArray(legend,label,$color_num)]
            set widest_label_w [expr [lindex $bbox 2] - [lindex $bbox 0]]
            #puts "widest_label_w: $widest_label_w"
            
            font delete $fontname
                
            set legend_w [expr $map_margin + $margin + $box_w + $margin + $widest_label_w + $margin]    
            
            #puts "Legend W: $legend_w"
            
            return $legend_w
            
        }
        
        
        proc SetColors { {aColorSet -1} } {
            
            variable iCTH_min
            variable iCTH_max
            variable iCTH_step  
            
            variable iColors  
            variable iColorset
            
            # Handle string format colorset argument
            if { ![string is integer -strict $aColorSet] } {
                if { [string equal $aColorSet "default"] } {       
                    set aColorSet 0
                } else {
                    set aColorSet 1
                }        
            }
            
            if { $aColorSet > -1 } {
                set iColorset $aColorSet
            }
            
            #catch {unset iColors}
            if { $iColorset == 0 } {      
                # Color format R,G,B
                # default
                set iColors [list 153,204,255 204,255,255 \
                                  255,255,255 255,255,153 \
                                  255,204,0 255,153,0 \
                                  255,102,0 255,0,0 \
                                  153,0,0 77,0,0 \
                                  0,0,0 ]
                            
            } elseif { $iColorset == 1 } {
                #Black and white
                set iColors [list 255,255,255 238,238,238 \
                                  221,221,221 204,204,204 \
                                  187,187,187 170,170,170 \
                                  153,153,153 136,136,136 \
                                  119,119,119 102,102,102 \
                                  85,85,85 68,68,68 \
                                  51,51,51 34,34,34 \
                                  17,17,17 0,0,0 ]            
            } elseif { $iColorset == 2 } {
                #yellow-brow
                set iColors [list 255,255,71 247,241,67 \
                                  230,215,58 212,186,48 \
                                  198,163,40 182,136,31 \
                                  163,105,21 144,80,13 \
                                  127,67,11 99,51,9 \
                                  83,40,7 58,14,1 ]            
            } else {
                #blue-white-red
                set iColors [list 0,0,255 43,43,255 \
                                  76,76,255 119,119,255 \
                                  145,145,255 184,184,255 \
                                  210,210,255 255,255,255 \
                                  255,210,210 255,184,184 \
                                  255,145,145 255,119,119 \
                                  255,76,76 255,43,43 \
                                  255,0,0 ]
                                 
            }
            
            
            #Color Thresholds
            set iCTH_step [expr ($iCTH_max - $iCTH_min) / double([llength $iColors])]
        
        }
        
        proc GetColor { aValue } {
            
          variable iCTH_min
          variable iCTH_max
          variable iCTH_step  
            
          variable iColors 
            
          # Color format R,G,B
          set color [lindex $iColors 0]
          set E 0.0000001
          
          if { $aValue < [expr $iCTH_min + $E] } { 
              # if below min threshold
              set color [lindex $iColors 0] 
          } elseif { $aValue > [expr $iCTH_max - $E] } {
              # if above max threshold
              set color [lindex $iColors end]   
          } else {
              set index [expr int(floor(($aValue - $iCTH_min) / $iCTH_step))]
              if { $index >= [llength $iColors] } { set index [expr $index - 1] }
              set color [lindex $iColors $index]
          }
          
          
          
          # DEBUG
          #puts "Min: $iCTH_min Max: $iCTH_max Step: $iCTH_step Value: $aValue color: $color"
          
          #puts "value $aValue:"
          #puts -nonewline "Value: [format %f $aValue] Index: [expr round(($aValue - $iCTH_min) / $iCTH_step)] Color:$color"
          
          return [[namespace current]::HexColor $color]
           
        }
        
        proc HexColor { aRGBcolor } {
            
          set hexcolor "#"
          foreach val [split $aRGBcolor ","] {
              set hexcolor [join [concat $hexcolor [format %.2X $val]] ""]
          }
          
          # Check for error
          if { [string equal $hexcolor "#"] } {
            puts "ERROR: Unable to create hexcolor out of $aRGBcolor. Using green."
            set hexcolor "#00FF00"   
          }
          
          #puts " Hexcolor: $hexcolor"
          
          return $hexcolor     
            
        }
        
        
        proc addLine {x y} {
            
            variable iW
            variable lastx
            variable lasty
            
            puts "addline $x $y"
            
            $scrolled_canvas create line 0 0 $x $y
            #set ::lastx $x 
            #set ::lasty $y
        }
        
        
        proc LoadFileDialog { } {
        
            set types {
              {{HeatMapper Data Files}     {.hm}}              
              {{All Files}        *             }
            }
            set filenameList [tk_getOpenFile -filetypes $types -multiple false]
            
            
            if { ![llength $filenameList] || ![string length [lindex $filenameList 0]]  } {       
                # Dialog cancelled
                return ""
            }
            
            set filename [lindex $filenameList 0]
            #puts "Load file: $filename"
            
            
            return $filename
        }
        
        proc SetLoadedStatus { aIsLoaded } {
            
            variable iW
            variable iMapInfo
            
            set iMapInfo(loaded) $aIsLoaded
            
            if { !$aIsLoaded } {
                $iW.menubar.file.menu entryconfigure 1 -state disabled
                $iW.menubar.file.menu entryconfigure 2 -state disabled
                $iW.menubar.file.menu entryconfigure 3 -state disabled    
                $iW.menubar.operations configure -state disabled
            } else {
                $iW.menubar.file.menu entryconfigure 1 -state normal
                $iW.menubar.file.menu entryconfigure 2 -state normal
                $iW.menubar.file.menu entryconfigure 3 -state normal                 
                $iW.menubar.operations configure -state normal
            }

        }
        
        
        proc LoadFile { {aFilename ""} } {
            
            variable iLineCount
            variable iFrameCount
            variable iMapInfo
            variable iMapDefault
            
            # If no filename given open dialog
            if { ![string length $aFilename] } {
                set aFilename [[namespace current]::LoadFileDialog]
            }
            
            # Dialog was cancelled
            if { ![string length $aFilename] } { return 0 }
            
            
            #puts "HEATMAPPER: Loading file $aFilename...."
            
            # File paths with spaces are encapsulated with \{\}
            # that need to be removed
            if { [string equal [string range $aFilename 0 0] "\{"] } {
                set aFilename [string range $aFilename 1 end-1]
            }
            
            # Checks
            if { ![string length $aFilename] } {
                puts "ERROR: No file to load"        
                return 0
            }
            if { ![file exists $aFilename] } {
                puts "ERROR: File $aFilename does not exists"
                return 0   
            }
            
            [namespace current]::SetLoadedStatus 0
            array unset iMapInfo *
            
            # Set map info to default values
            foreach def_key [array names iMapDefault] {
                set iMapInfo($def_key) $iMapDefault($def_key)
            }
             
            # Read File
            if { [file readable $aFilename] } {          
                if { [[namespace current]::ReadFile $aFilename] < 0 } {
                    puts "ERROR: Error loading file $aFilename"
                    return -1        
                }
                #success
            } else {
                puts "ERROR: Cannot find or read file: $aFilename"   
                return -1
            }
            set iMapInfo(shape) "$iFrameCount $iLineCount"
            [namespace current]::SetYNumberingMenu
            
            # Load success
            #puts "HEATMAPPER: File $aFilename loaded successfully"
            [namespace current]::SetLoadedStatus 1
            #set iMapInfo(loaded) 1
            
            # Set color and min max values
            [namespace current]::ValidateAndApplyThresholds no
            
            # Refresh by setting Zoom
            set zoom 8
            if { $iLineCount > 50 || $iFrameCount > 100 } { set zoom 4 }
            if { $iLineCount > 200 || $iFrameCount > 300 } { set zoom 2 }
            if { $iLineCount > 800 || $iFrameCount > 900 } { set zoom 1 }
            
            [namespace current]::SetZoom $zoom    
            return 0
        }
        
        # iMapInfo Contains keys: title, xlabel, ylabel, xorigin, yorigin, xstep, 
        #                         loadfile, min, max, colorset, callback
        proc GetCommandLineArguments { args } {
            variable iMapInfo    
            
     
            
            if { [llength $args] < 1 || [llength $args] > 1 } {
                puts "ERROR: Wrong number of arguments for heatmap cget."
                return ""
            } elseif { [string length [lindex $args 0]] < 2 } {
                puts "ERROR: Bad argument \"[lindex $args 0]\" for heatmap cget."
                return ""
            }
            
            # Process only one arg if multiple
            set cmd [lindex $args 0]
            
            
            # Strip leading hyphen if any
            if { [string equal "-" [string range $cmd 0 0]] } { 
                set cmd [string range $cmd 1 end]
            }       
            
            if { ![info exists iMapInfo($cmd)] } {
                puts "ERROR: Invalid argument \"$cmd\" for heatmap cget."
                return                  
            }
            
            #puts "Returning: $iMapInfo($cmd)"
            return $iMapInfo($cmd)
        }
        
        proc ListCommandLineArguments { } {
            variable iMapInfo
            return [array names iMapInfo]
        }
        
        proc ProcessCommandLineArguments { aFileLoadAllowed args } {
            
            variable iMapInfo
            variable iShow
            
            variable iMinEntry
            variable iMaxEntry
            
            variable iColorset
            
            set fileLoadCommandFound 0
            
            #puts "ProcessCommandLineArguments"
            
            # DEBUG
            #foreach arg $args {
            #    puts "proc ARG: $arg"
            #}
              
            if { [llength $args] == 0 } {
                puts "DEBUG: ProcessCommandLineArguments: No args."
                return    
            }
            
            # args can contains a list a single line with arguments separated by " "
            set list $args
            if { [llength $args] <= 1 } {
                set list [split $args " "]
            }
            
            for { set i 0 } { $i < [llength $list]  } {incr i } {
                #remove all curly brackets
                lset list $i [string map { "\{" "" "\}" ""} [lindex $list $i]]
            }
               
            
            # Process arguments
            for { set arg 0 } { $arg < [llength $list]  } {incr arg } {
             
                #puts "LISTARG: [lindex $list $arg]"
                #puts "NEXTARG: [lindex $list [expr $arg + 1]]"
                
                set cmd [lindex $list $arg]
                set next ""
                
                if { ![string equal [string index $cmd 0] "-"] } {
                    continue   
                } else {
                    # New Command, all commands start with a "-"
                    #puts "Cmd: $cmd"
                    set next [lindex $list [expr $arg + 1]]
                    set cmd [string range $cmd 1 end]
                }
                
                # If command parameter start with \" find closing \"
                # from further down the line
                if { [string equal [string index $next 0] "\""] } {
                 
                    set closer $arg
                    #Find closing "\""
                    for { set find $arg } { $find < [llength $list]  } {incr find } {
                        if { [string equal [string range [lindex $list $find] end end] "\"" ] } {
                            set closer $find                    
                            break
                        }
                    }
                    set next [join [lrange $list [expr $arg + 1] $closer] " "]
                    set arg [expr $closer + 1]
                    
                    # Strip start and end \" chars
                    set next [string range $next 1 end-1]
                    #puts "Next \"\": $next"
                       
                }        
                        
                # iMapInfo Contains keys: title, xlabel, ylabel, xorigin, xstep, 
                #                         loadfile, min, max, colorset                
                switch $cmd {
                    title { set iMapInfo(title) $next }
                    xlabel { set iMapInfo(xlabel) $next }
                    ylabel { set iMapInfo(ylabel) $next }
                    xorigin { set iMapInfo(xorigin) $next }
                    xstep { set iMapInfo(xstep) $next }
                    min { set iMapInfo(min) $next }
                    max { set iMapInfo(max) $next }   
                    colorset { set iMapInfo(colorset) $next } 
                    loadfile { 
                                if { $aFileLoadAllowed } { 
                                    set iMapInfo(loadfile) $next 
                                    set fileLoadCommandFound 1
                               }
                             }
                    
                    hide { if { [info exists iShow($next)] } { set iShow($next) 0 }  }
                    show { if { [info exists iShow($next)] } { set iShow($next) 1 }  }
                    
                    numbering { 
                                set numbering [split $next ":"]
                                for {set i 0} { $i < [llength $numbering] } {incr i} {
                                    set iMapInfo($i,numbering)   [lindex $numbering $i]
                                    # DEBUG
                                    # puts "Adding [lindex $numbering $i] to mapinfo $i,numbering"
                                }
                              }
                    
                    callback { set iMapInfo(callback) $next }
                    
                    default { puts "WARNING: Unknown command: $cmd" }
                }                
            }
            
            # Set to defaults if erroneous values given
            if { ![string is integer $iMapInfo(xstep)] } { set iMapInfo(xstep) 1 }
            if { ![string is integer $iMapInfo(xorigin)] } { set iMapInfo(xorigin) 0 }
            # Note: Min and Max could also be integers...
            if { [info exists iMapInfo(min)] && ![string is double $iMapInfo(min)] } { set iMapInfo(min) 0.00 }
            if { [info exists iMapInfo(max)] && ![string is double $iMapInfo(max)] } { set iMapInfo(max) 100.00 }
            
            if { $fileLoadCommandFound && $aFileLoadAllowed } {
                [namespace current]::LoadFile $iMapInfo(loadfile)
            } elseif { $aFileLoadAllowed || $iMapInfo(loaded) } {
                [namespace current]::Refresh                
            }            
            
            #foreach name [array names iMapInfo] {
                #puts "iMapinfo: $name  $iMapInfo($name)"
            #}
            
        }
        
        proc ReadFile { aFilename } {
            
            variable iW
            #variable iMapImage
            #variable iZoomFactor
            variable iData
            variable iDataMappingY
            
            variable iFrameCount
            variable iLineCount
            
            variable iMinEntry
            variable iMaxEntry
            
            variable iCTH_min
            variable iCTH_max
            
            variable iMapInfo
            
            
            # Unset min and max so they will be calculated
            array unset iMapInfo min
            array unset iMapInfo max
            
            
            #TODO error handling
            set fp [open $aFilename r]
            
            #Slurp it all in
            set file_data [read $fp]
            close $fp
    
            # get min/max colorid
            set mincolor [colorinfo num]
            set maxcolor [colorinfo max]
            set ncolorid [expr $maxcolor - $mincolor]

            set canvas_row_size 0
            
            # Process data file
            set data_lines [split $file_data "\n"]
            
            set x 0
            set y 0
            set frame_count 0
            set line_count 0
            
            
            set max 0.00
            set min 999999.00
            
            # Process one line at a time
            foreach line $data_lines {
                
                set line [string trim $line]
                   
                if { ![string length $line] } {
                    # Skip empty lines
                    continue
                } elseif { [string equal [string index $line 0] "#"] } {
                    # Skip comment lines
                    continue
                } elseif { [string equal [string index $line 0] "-"] } {
                    # Set setting            
                    [namespace current]::ProcessCommandLineArguments no $line
                    continue
                }
                
                # Get number used for axis label
                #set resid [string range $line 0 [expr [string first ":" $line] - 1]]
                # remove first part of numbering
                #set line [string range $line [expr [length $resid] + 1] last]
                
                # Get numbering
                set numbering [split [string range $line 0 [expr [string last ":" $line] - 1]] ":"]
                
                #puts "Resid:$resid"
                
                if { [llength $numbering] < 1 } {
                    puts "ERROR: File syntax incorrect. No resid numbering found."
                    puts "Line:$line"
                    return -1    
                }
                
                #remove numbering
                set line [string range $line [expr [string last ":" $line] + 1] end]
                
                #split into RMSD values
                set values [split $line ";"]
                
                #puts "For resid $resid frames: [llength $values]"
                
                if { !$canvas_row_size } {
                    #TODO set canvas size    
                    set canvas_row_size [expr [llength $values] - 1]
                }
                
                set fr $iMapInfo(xorigin)
                set frame_count 0
                
                
                # Start painting new heatmap row                
                foreach value $values {         
                    if { ![string length $value] } {
                        #Skip empties (last of row always empty)                
                        continue
                    }
                    
                    #Store values into iData
                    set iData($fr,$line_count) $value            
                    
                    #puts "DEBUG Read idata $fr,$resid"
                    
                    if { $value > $max } { set max $value }
                    if { $value < $min } { set min $value }
                    
                    incr fr
                    incr frame_count     
                }
                       
                for {set i 0} { $i < [llength $numbering] } {incr i} {
                    set iDataMappingY($i,$line_count) [lindex $numbering $i]
                }
        
                incr line_count
                # End of heatmap row
            }
                
            if { [info exists iMapInfo(min)] } {
                set min $iMapInfo(min)
            } 
            if { [info exists iMapInfo(max)] } {
                set max $iMapInfo(max)
            }
            
            set iMinEntry [format %.2f $min] 
            set iMaxEntry [format %.2f $max]
            
            set iCTH_min [expr double($min)]
            set iCTH_max [expr double($max)]
            
            
            set iLineCount $line_count
            set iFrameCount $frame_count
            
            #puts "iLineCount: $iLineCount iFrameCount: $iFrameCount"
            
            #[namespace current]::Refresh
            
            
            #success
            return 0
        }
        
        
        proc LoadComparison { aComparisonType { aFilename "" } } {
            
            variable iW

            variable iData
            variable iDataMappingY
            
            variable iFrameCount
            variable iLineCount

            variable iMapInfo
            

            # If no filename given open dialog
            if { ![string length $aFilename] } {
                set aFilename [[namespace current]::LoadFileDialog]
            }
            
            # Dialog was cancelled
            if { ![string length $aFilename] } { return 0 }            
            
            set operand "-"
            switch $aComparisonType {
                add { set operand "+" }
                sub { set operand "-" }
                mul { set operand "*" }    
            }
            
            
            # TODO error handling
            set fp [open $aFilename r]
            
            # Slurp it all in
            set file_data [read $fp]
            close $fp

            # Process data file
            set data_lines [split $file_data "\n"]
            
            set x 0
            set y 0
            set frame_count 0
            set line_count 0
            
            
            # Process one line at a time
            foreach line $data_lines {
                
                set line [string trim $line]
                   
                if { ![string length $line] } {
                    # Skip empty lines
                    continue
                } elseif { [string equal [string index $line 0] "#"] } {
                    # Skip comment lines
                    continue
                } elseif { [string equal [string index $line 0] "-"] } {           
                    # Skip setting lines
                    continue
                }
                
                # Get numbering
                set numbering [split [string range $line 0 [expr [string last ":" $line] - 1]] ":"]
                #set resid [string range $line 0 [expr [string first ":" $line] - 1]]
                #puts "Resid:$resid"
                
                if { [llength $numbering] < 1 } {
                    puts "ERROR: File syntax incorrect. No residue numbering found."
                    puts "Line:$line"
                    return -1    
                }
                
                #remove residue number
                set line [string range $line [expr [string last ":" $line] + 1] end]
                #split into RMSD values
                set values [split $line ";"]
                
        
                set fr $iMapInfo(xorigin)
                set frame_count 0                
                
                set had_errors 0
                
                # Note: With comparion matching residue number must match
                #       Frames are compared first frame with first frame in the maps
                # Note: Frames are stored in iDate as iMapInfo(x_origin) + 0,1,2,3... even if xstep is not 1
                #
                # Loop trough residues (lines) all frames                
                foreach value $values {         
                    # Skip if empty value (last of row always empty) 
                    if { ![string length $value] } { continue }
                    
                    # Skip if iData does not exist
                    if { ![info exists iData($fr,$line_count)] } { set had_errors 1; continue }
                    
                    # DEBUG
                    #puts "$iData($fr,$line_count) $operand $value = [expr $iData($fr,$line_count) $operand $value] at $fr,$resid"
                    
                    # Apply expression                    
                    set iData($fr,$line_count) [expr $iData($fr,$line_count) $operand $value]

                    incr fr
                    incr frame_count                         
                    
                    # If comparison has more frames, 
                    # ignore frames beyond original maps frames
                    if { $frame_count > $iFrameCount } { break }
                }
                       
                # set iDataMappingY($line_count) $resid 
                for {set i 0} { $i < [llength $numbering] } {incr i} {
                    set iDataMappingY($i,$line_count) [lindex $numbering $i]
                }
                
                incr line_count
                # End of heatmap row
            }
            
            if { $had_errors } { puts "WARNING: not all residues were matched." }
            [namespace current]::Refresh
        }        
        
        
        proc PlotData { } {
         
            variable iW
            
            variable iData
            variable iDataMappingY
            
            variable iMapImage
            variable iFrameCount    
            variable iLineCount
            
            variable iZoomFactor
            variable iMapInfo
         
            #puts "Plotting data"
            
              
            set keys [array names iData 0,*]
            set residList [list]
            
            #puts "Keys: $keys"
            
            foreach key $keys {
                set split_key [split $key ","]   
                lappend residList [lindex $split_key 1]            
            }
            
            # Do Not Sort!
            # Keep file order
            #set residList [lsort -integer $residList]
            
            #puts $residList
            
            # Image coordinates 0,0 is nw corner
            set x 0
            set y 0
            
            #puts "Framecount: $iFrameCount"
            
            # Adjust starting point if row count does not fill minimum height
            if { [image height $iMapImage] > [expr $iLineCount * $iZoomFactor] } {        
                set y [expr [image height $iMapImage] - ($iLineCount * $iZoomFactor)]
            }
            
            
            # PLOT! starting from nw corner
            for {set res [expr $iLineCount - 1]} { $res >= 0  } {incr res -1} {
                #puts "Plotting resid $resid"
                for {set fr 0} { $fr < $iFrameCount } {incr fr} {
        
                    #$iMapImage put [[namespace current]::GetColor $iData([expr $fr + $iMapInfo(xorigin)],$iDataMappingY($res))] -to $x $y [expr $x + $iZoomFactor] [expr $y + $iZoomFactor]       
                    $iMapImage put [[namespace current]::GetColor $iData([expr $fr + $iMapInfo(xorigin)],$res)] -to $x $y [expr $x + $iZoomFactor] [expr $y + $iZoomFactor]       
                    #$iMapImage put "#FFCC00" -to $x $y [expr $x + $iZoomFactor] [expr $y + $iZoomFactor]       
                    incr x $iZoomFactor
                }      
                incr y $iZoomFactor
                set x 0           
            }   
            
            # Old plot
            #foreach resid $residList {
            #    #puts "Plotting resid $resid"
            #    for {set fr 0} { $fr < $iFrameCount } {incr fr} {
            #        $iMapImage put [[namespace current]::GetColor $iData($fr,$resid)] -to $x $y [expr $x + $iZoomFactor] [expr $y + $iZoomFactor]       
            #        incr x $iZoomFactor
            #    }      
            #    incr y $iZoomFactor
            #    set x 0           
            #}    
            
            
            
            
            #puts "Finished plotting"
            
            
        }
        
        
        proc ReleaseAll { } {
            
            variable iZoomFactor
            variable iDataMappingY
            variable iData
            variable iMapInfo
            variable iShow
            
            #puts "[namespace current]::ReleaseAll"
          
            # Delete traces
            # Delete remaining selections

            # FIX: The following deletes all images whether they are created by heatmapper or not
            # for example run the following using a heatmap file
            #
            #   [heatmapper -loadfile somefile.hm] quit
            #   tk_messageBox -type okcancel
            #   [heatmapper -loadfile somefile.hm] quit
            #   tk_messageBox -type okcancel
            #
            # in the last call of tk_messageBox, creating message box will fail because all images are deleted, including ::tk::dialog::b2
            # I made a counter in ::HeatMapper:: namespace and used prefix `hmimage`, and now delete images starting with this prefix and not in use 
  
            foreach name [image names] {
                if {[string match hmimage* $name ] && ![image inuse $name]} { image delete $name; puts "image delete $name"}
            }
            
            trace vdelete iZoomFactor w [namespace current]::Refresh
        
            array unset iDataMappingY *
            array unset iData *
            
            array unset iMapInfo *
            #array unset iShow *
            
        }
        
        proc HelpAbout { } {
          
            variable iW
            set vn [package present heatmapper]
          
            tk_messageBox -title "About Heatmapper v$vn" -parent $iW -message \
            "Heatmapper v$vn extension for VMD \n\n\
            Anssi Nurminen \n\
            University of Tampere \n\
            Institute of Biomedical Technology \n\
            2011-09-27"
        
        }
        
        
        proc PostScriptify {} {
            
             variable iW    
             set ps_file "heatmap.ps"
             set scrolled_canvas $iW.c.canvas
           
             set types {
                {{Postscript files} {.ps}}
                {{All files}        *   }
             }
                       
             set newfile [tk_getSaveFile -title "Choose file name" -parent $iW \
                                         -initialdir [pwd] -filetypes $types -initialfile $ps_file]         
             if {[llength $newfile]} {
                $scrolled_canvas postscript -file $newfile
                vmdcon -info "Wrote map postscript file to $newfile."
             }
        }
        
        # Debug procedure, not used
        # Trying to find out the color of a pixel in a canvas
        proc ShowColor {w x y} {

            set color ""
            set id [$w find withtag current]
            #puts "ID: $id"
            if { $id < 0 } { puts "No Id"; return }
            
                
            set xx $x
            set yy $y
            foreach {x0 y0 x1 y1} [$w bbox $id] break
            incr xx -$x0
            incr yy -$x0
            #puts "$x $y #$id : $x0 $y0  $x1 $y1 : $xx $yy"
            #return
            
            switch -- [$w type $id] {
                image {set color [[$w itemcget $id -image] get $xx $yy]}
                line - polygon - rectangle - oval - text {
                    set color_name [$w itemcget $id -fill]
                    set color [winfo rgb $w $color_name]
                }
            }
            puts "$x $y #$id $xx $yy = $color"
            
            
            return $color
        }        
        
        proc SaveAsImage {} {
            
            variable iW
            variable iMapInfo
            variable iMapImage
                   
            if { !$iMapInfo(loaded) } { puts "ERROR: Saving map data image: Map not loaded."; return }
            
            set gif_file "heatmap.gif"
            
            set types {
                {{Graphic Interchange Format files} {.gif}}
                {{All files}        *   }
            }
                   
            set newfile [tk_getSaveFile -title "Choose file name" -parent $iW \
                                        -initialdir [pwd] -filetypes $types -initialfile $gif_file]               
            
            # If dialog cancelled
            if { ![llength $newfile]} { return }
            
            
            $iMapImage write $newfile -format GIF89 -background "#FFFFFF"
            
            vmdcon -info "Wrote map gif-file to $newfile."
        }        
        
        
        proc SaveAsText {} {
            
            variable iW
            variable iMapInfo
            variable iMapImage
            variable iData
            variable iDataMappingY
            variable iFrameCount
            variable iLineCount
                   
            if { !$iMapInfo(loaded) } { puts "ERROR: Saving map data image: Map not loaded."; return }
            
            set hm_file "heatmap.hm"
            
            set types {
                {{Heatmapper file} {.hm}}
                {{Text File} {.txt}}
                {{All files}        *   }
            }
                   
            set newfile [tk_getSaveFile -title "Choose file name" -parent $iW \
                                        -initialdir [pwd] -filetypes $types -initialfile $hm_file]               
            
            # If dialog cancelled, return
            if { ![llength $newfile]} { return }
            
            
            # TODO file name and error handling
            set fileId [open $newfile "w"]
        
        
            foreach key [array names iMapInfo] {
            
                if { [string equal $key "loaded"] } { continue }
                if { [string equal $key "loadfile"] } { continue }
                if { [string last "numbering" $key] > 0 } { continue }
                
                # Write map settings
                puts $fileId "-$key \"$iMapInfo($key)\""
            }
            
            # Write numbering information
            set numberings [list]
            foreach map_key [array names iMapInfo *,numbering] {
                lappend numberings $iMapInfo($map_key)
            }            
            puts -nonewline $fileId "-numbering \""
            puts -nonewline $fileId [join $numberings ":"]
            puts $fileId "\""
            
            for {set y 0} { $y < $iLineCount } {incr y } {
            
                set i 0
                while { [info exists iDataMappingY($i,$y)] == 1 } {
                    puts -nonewline $fileId "$iDataMappingY($i,$y):"
                    incr i
                }
                
                
                # Note: iMapInfo(xstep) not used
                for {set x $iMapInfo(xorigin)} { $x < [expr $iFrameCount + $iMapInfo(xorigin)] } {incr x} {
                    #puts "DEBUG writing $x,$iDataMappingY(0,$y)"
                    puts -nonewline $fileId "$iData($x,$y);"
                }
                
                puts $fileId " "
            }            
            
            #file written
            close $fileId
            
            vmdcon -info "Wrote heatmap data to $newfile."
        }        
                
        proc ApplyExpression {} {
            
            variable iData
            
            set expression "VALUE * 1"
            
            set expression [[namespace current]::Inputbox "Enter expression" "Enter expression to apply to all values in the heatmap\n(Tcl expr functions available):" $expression]
            
            
            if { [string equal $expression "can_zel"] } {
                # Dialog cancelled, return
                return    
            }            
            
            set val_pos [string first "VALUE" $expression]
            
            if { $val_pos == -1 } {
                puts "ERROR: Expression missing keyword \"VALUE\""
                return
            }
           
            # Replace inserted keyword(s)
            set expression [string map {"VALUE" "\$iData(\$key)"} $expression]
            
            foreach key [array names iData] {
                set iData($key) [expr $expression] 
            }
            
            [namespace current]::Refresh
        }
         
        # Simple small dialog box for getting user input
        # returns "can_zel" if user cancelled it.
        proc Inputbox { title text {default {}}} {
            
            variable iW
            
            set w .inputheatmapexpr
            
            global frmReturn
            #set frmReturn ""
            set retval ""
            
            catch {destroy $w}
            toplevel $w -class inputdlg
            wm title $w $title
            wm iconname $w $title
            wm protocol $w WM_DELETE_WINDOW { }
            wm resizable $w 0 0
            wm transient $w $iW
            
            option add *$w.*borderWidth 1
            option add *$w.*Button.padY 2
            option add *$w.*Menubutton.padY 0  
                
            
            # Create dialog
            pack [frame $w.bot -relief raised -bd 2] -side bottom -fill both
            pack [frame $w.top -relief raised -bd 2] -side top -fill both -expand 1
            option add *Inputbox.msg.wrapLength 3i widgetDefault
            
            # Set dialog text
            label $w.msg -justify left -text $text;#-font {Times 18}
            
            # Create input box
            entry $w.input -textvariable [namespace current]::retval -relief sunken -bd 1
            $w.input delete 0 end
            $w.input insert end $default
            bind $w.input <Return> "$w.b0 invoke"
            bind $w <Destroy> {set frmReturn {}}
            
            
            pack $w.msg -in $w.top -side top -expand 1 -fill both -padx 3m -pady 3m
            pack $w.input -in $w.top -side top -expand 1 -fill x -padx 3m -pady 3m
            
            # Buttons
            button $w.b0 -text "Apply" -command "set frmReturn \[$w.input get\]"
            button $w.b1 -text "Cancel" -command {set frmReturn "can_zel"}
            
            grid $w.b0 -in $w.bot -column 0 -row 0 -stick nswe -padx 10
            grid $w.b1 -in $w.bot -column 1 -row 0 -stick nswe -padx 10
            
            wm withdraw $w
            update idletasks
            
            
            set x [expr [winfo screenwidth $w]/2 - [winfo reqwidth $w]/2 - [winfo vrootx [winfo parent $w]]]
            set y [expr [winfo screenheight $w]/2 - [winfo reqheight $w]/2 - [winfo vrooty [winfo parent $w]]]
            wm geometry $w +$x+$y
            wm deiconify $w
            
            set oldfocus [focus]
            set oldgrab [grab current $w]
            if {$oldgrab != ""} {
                set grabstatus [grab status $oldgrab]
            }
            
            grab $w
            focus $w.input
            tkwait variable frmReturn
            
            set retval $frmReturn
            
            catch {focus $oldfocus}
            catch {
                bind $w Destroy {}
                destroy $w
            }
            
            if {$oldgrab != ""} {
                if {$grabstatus == "global"} {
                    grab -global $oldgrab
                } else {
                    grab $oldgrab
                }
            }
            
        
            #puts "Returning $retval"
            return $retval
        }            
           
        
        
      # End of procedures 
    };# End of new_ns

   
    eval "::HeatMapper::Plot${::HeatMapper::mapcount}::plothandle configure $args"
    
   
    return "::HeatMapper::Plot${::HeatMapper::mapcount}::plothandle"   
}


