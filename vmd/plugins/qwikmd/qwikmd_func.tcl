#
# $Id: qwikmd_func.tcl,v 1.42 2016/11/16 14:22:38 jribeiro Exp $
#
#==============================================================================
proc QWIKMD::procs {} {
    
    global tcl_platform env

     if {$::tcl_platform(os) == "Darwin"} {
        catch {exec sysctl -n hw.ncpu} proce
        return $proce
      } elseif {$::tcl_platform(os) == "Linux"} {
        catch {exec grep -c "model name" /proc/cpuinfo} proce
        return $proce
      } elseif {[string first "Windows" $::tcl_platform(os)] != -1} {
        catch {HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor } proce
        set proce [llength $proce]
        return $proce
      }
}
proc QWIKMD::loadTopologies {} {
    set QWIKMD::topolist [list]
    foreach topo $QWIKMD::TopList {
        if {[file exists $topo]} {
            lappend QWIKMD::topolist [::Toporead::read_charmm_topology $topo]
        }
    }

}

proc QWIKMD::redirectPuts {outputfile cmd } {
    set cmdOK 0
    rename ::puts ::tcl::orig::puts

    proc ::puts args "
        if {\$args == \{\}} {set args \" \"}
        if {\[file channels \[lindex \[lindex \$args 0\] 0\]  \] == \[lindex \[lindex \$args 0\] 0\] && \[lindex \[lindex \$args 0\] 0\] != \"\"} {
             uplevel \"::tcl::orig::puts \[lindex \[lindex \$args 0\] 0\] \[lrange \$args 1 end\]\"; return
        } else {
            uplevel \"::tcl::orig::puts $outputfile \$args\"; return
        }    
    "
    set outcommand ""
    set cmdOK [eval catch {$cmd} outcommand]
    rename ::puts {}
    rename ::tcl::orig::puts ::puts
    return "$cmdOK $outcommand"
}

proc QWIKMD::checkDeposit {} {
    global env
    set do 0
    set answer cancel
    set filename ".qwikmdrc"
    if {[string first "Windows" $::tcl_platform(os)] != -1} {
        set filename "qwikmd.rc"
    }
 
    set location ""
    set location ${env(HOME)}

    if {[file exists ${location}/${filename}] == 1} {
        source ${location}/${filename}
    }
    if {[info exists env(QWIKMDFOLDER)] == 0} {
        set answer [tk_messageBox -message "The folder \"qwikmd\" will be created in your home directory\
         to save user-defined MD simulation templates and residues topologies and parameters preferences. A file named $filename will also \
         be created to save this information." -title "Deposit Folder" -icon info -type ok]
         set do 1

    } elseif {[file exists $env(QWIKMDFOLDER)] == 0} {
        tk_messageBox -message "QwikMD depository folder was deleted or moved. The process to define this folder has to be repeated." -title "Deposit Folder" -icon info -type ok
        file delete -force ${location}/${filename}
        unset env(QWIKMDFOLDER)
        QWIKMD::checkDeposit
    }
    if {$do == 1} {
        set folder "${::env(HOME)}/qwikmd"
        
        if {[file exists ${folder}] == 0}  {
            file mkdir ${folder}
        }

        if {[file exists ${folder}/templates] == 0}  {
            file mkdir ${folder}/templates
            file mkdir ${folder}/templates/Explicit
            file mkdir ${folder}/templates/Implicit
            file mkdir ${folder}/templates/Vacuum
        }
            
        set templates [glob ${env(QWIKMDDIR)}/*.conf]
        foreach temp $templates {
            file copy -force ${temp} ${folder}/templates/
        }
        if {[file exists ${folder}/toppar] == 0}  {
            file mkdir ${folder}/toppar
        }
        
        set vmdrc [open ${location}/${filename} a]
        puts $vmdrc "set env(QWIKMDFOLDER) \"[file normalize ${folder}]\""
        set env(QWIKMDFOLDER) [file normalize ${folder}]
        close $vmdrc
    }
    
}
##############################################
## Orient pulling and anchor residues to the Z-axis
## in case of SMD, otherwise just move to the origin
###############################################
proc QWIKMD::orientMol {structure} {
    set selall [atomselect $structure all]
    set selmove ""
    set zaxis {0 0 -1}
    set center [measure center $selall]
    set move_dist [transoffset [vecsub {0 0 0} $center]]
    
    $selall move $move_dist
    if {$QWIKMD::membraneFrame != "" && [llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 0} {
        $selall move [measure inverse $QWIKMD::advGui(membrane,rotationMaxtrix)]    
    }
    
    if {$QWIKMD::run == "SMD"} {
        set selanchor [atomselect $structure "$QWIKMD::anchorRessel"]
        set selpulling [atomselect $structure "$QWIKMD::pullingRessel"]
        set anchor [measure center $selanchor]
        set pulling [measure center $selpulling]
        
        set axis [vecsub $pulling $anchor]
        $selpulling delete
        $selanchor delete
        set M [transvecinv $axis] 
        $selall move $M 
        set M [transaxis y -90] 
        $selall move $M 

    } 
    $selall delete

}

proc QWIKMD::boxSize {structure dist} {

    set sel [atomselect $structure "all"]
    set minmax [measure minmax $sel]
    $sel delete
    set xsp [lindex [lindex $minmax 0] 0]
    set ysp [lindex [lindex $minmax 0] 1]
    set zsp [lindex [lindex $minmax 0] 2]

    set xep [lindex [lindex $minmax 1] 0]
    set yep [lindex [lindex $minmax 1] 1]
    set zep [lindex [lindex $minmax 1] 2]
    
    set xp [expr abs($xep - $xsp)]
    set yp [expr abs($yep - $ysp)]
    set zp [expr abs($zep - $zsp)]

    set xsb  ""
    set ysb  ""
    set zsb  ""

    set xeb  ""
    set yeb  ""
    set zeb  ""

    if {$QWIKMD::run != "SMD"} {
        set dp [expr sqrt($xp*$xp+$yp*$yp+$zp*$zp)]
        set box_length [expr $dp + 2*$dist]
    
        set xsb  [expr $xsp - ($box_length-$xp)/2]
        set ysb  [expr $ysp - ($box_length-$yp)/2]
        set zsb  [expr $zsp - ($box_length-$zp)/2]

        set xeb  [expr $xep + ($box_length-$xp)/2]
        set yeb  [expr $yep + ($box_length-$yp)/2]
        set zeb  [expr $zep + ($box_length-$zp)/2]
    } else {
        set dp [expr sqrt($xp*$xp+$yp*$yp)]
        set box_length [expr $dp + 2*$dist]

        set xsb  [expr $xsp - ($box_length-$xp)/2]
        set ysb  [expr $ysp - ($box_length-$yp)/2]
        set zsb  [expr $zsp - $dist]
  
        set xeb  [expr $xep + ($box_length-$xp)/2]
        set yeb  [expr $yep + ($box_length-$yp)/2]
        set zeb  [expr $zep + $dist + $QWIKMD::basicGui(plength)]
    } 
    

    set boxmin [list $xsb $ysb $zsb]
    set boxmax [list $xeb $yeb $zeb]

    set centerX [expr [expr $xsb + $xeb] /2]
    set centerY [expr [expr $ysb + $yeb] /2]
    set centerZ [expr [expr $zsb + $zeb] /2]

    set cB1 [expr abs($xeb - $xsb)]
    set cB2 [expr abs($yeb - $ysb)]
    set cB3 [expr abs($zeb - $zsb)]

    set center [list [format %.2f $centerX] [format %.2f $centerY] [format %.2f $centerZ]]
    set length [list [format %.2f $cB1] [format %.2f $cB2] [format %.2f $cB3]]
    set QWIKMD::cellDim [list $boxmin $boxmax $center $length]

    mol delete $structure
}


proc QWIKMD::rmsdAlignCalc {sel sel_ref frame} {
    set rmsd 0
    if {$QWIKMD::advGui(analyze,basic,alicheck) == 1} {
        set seltext ""
        if {$QWIKMD::advGui(analyze,basic,alientry) != "" && $QWIKMD::advGui(analyze,basic,alientry) != "Type Selection"} {
            set seltext $QWIKMD::advGui(analyze,basic,alientry)
        } else {
            set seltext $QWIKMD::advGui(analyze,basic,alicombo)
        }
        set alisel [atomselect $QWIKMD::topMol $seltext frame 0]
        set auxsel [atomselect $QWIKMD::topMol $seltext frame $frame]
        set tmatrix [measure fit $auxsel $alisel]
        $alisel delete
        $auxsel delete
        set move_sel [atomselect $QWIKMD::topMol "all" frame $frame]
        
        $move_sel move $tmatrix
        $move_sel delete
    } 
    
    return [measure rmsd $sel $sel_ref]
}

##############################################
## Like the hydrogen bonds calculation proc,
## the general proc (QWIKMD::RmsdCalc) calls the 
## calculator proc (QWIKMD::rmsdAlignCalc) so it is
## possible to call the same calculator proc in two
## instances with slight modifications
###############################################
proc QWIKMD::RmsdCalc {} {
    
    set top $QWIKMD::topMol
    set seltext ""
    if {$QWIKMD::advGui(analyze,basic,alientry) != "" && $QWIKMD::advGui(analyze,basic,alientry) != "Type Selection"} {
        set seltext $QWIKMD::advGui(analyze,basic,alientry)
    } else {
        set seltext $QWIKMD::advGui(analyze,basic,alicombo)
    }

    set sel_ref [atomselect $QWIKMD::topMol $seltext frame 0]
    if {[$sel_ref get index] != ""} {
        set sel [atomselect $top $seltext]
        set frame [molinfo $QWIKMD::topMol get frame]
        set const [expr $QWIKMD::timestep * 1e-6]
        $sel frame $frame
        
        lappend QWIKMD::timeXrmsd [expr {$const * $QWIKMD::counterts * $QWIKMD::imdFreq} + $QWIKMD::rmsdprevx]
        lappend QWIKMD::rmsd [QWIKMD::rmsdAlignCalc $sel $sel_ref $frame]
        $QWIKMD::rmsdGui clear
        $QWIKMD::rmsdGui add $QWIKMD::timeXrmsd $QWIKMD::rmsd
        $QWIKMD::rmsdGui replot
        $sel delete
    } 
    $sel_ref delete
}

proc QWIKMD::calcSASA {globaltext restricttext probe samples} {
    
    
    array set residues ""
    set resLis [list]
    set frames [molinfo $QWIKMD::topMol get numframes]
    set residues(total) [list]
    for {set i 0} {$i < $frames} {incr i} {
        set global [atomselect $QWIKMD::topMol $globaltext frame $i]
        set restrict [atomselect $QWIKMD::topMol "\($globaltext\) and \($restricttext\)" frame $i]
        #$restrict frame $i
        set textprev "" 
        foreach resid [$restrict get resid] resname [$restrict get resname] chain [$restrict get chain] {
            set text "$resid $resname $chain"
            if {$text != $textprev} {
                if {[lsearch $resLis ${text}] == -1} {
                    lappend resLis ${text}
                }
                set selaux [atomselect $QWIKMD::topMol "resid \"$resid\" and resname $resname and chain \"$chain\" " frame $i]
                lappend residues(${text}) [measure sasa 1.4 $global -restrict $selaux -samples $samples]
            
                $selaux delete
            }
            set textprev $text
        }

        lappend residues(total) [measure sasa 1.4 $global -restrict $restrict -samples $samples]
        $restrict delete
        $global delete
    }

    # $global delete
    # $restrict delete

    return "{[array get residues]} {$resLis}"
}
#This proc is using a command that will still need to be improved (not in use)
proc QWIKMD::calcSASA2 {globaltext restricttext probe samples} {
    set global [atomselect top $globaltext]
    set restrict [atomselect top "\($globaltext\) and \($restricttext\)"]
    
    
    set textprev ""
    array set residues ""
    set resLis [list]
    set frames [molinfo $QWIKMD::topMol get numframes]
    set residues(total) [list]
    set selList [list]
    foreach resid [$restrict get resid] resname [$restrict get resname] chain [$restrict get chain] {
        set text "$resid $resname $chain"
        if {$text != $textprev} {
            
            lappend resLis ${text}
            
            lappend selList [atomselect $QWIKMD::topMol "resid \"$resid\" and resname $resname and chain \"$chain\" " frame 0]
            set textprev $text
        }
    }
    for {set i 0} {$i < $frames} {incr i} {
        $global frame $i
        $restrict frame $i
            
        foreach sel $selList {
            $sel frame $i
        }
        set val [measure sasalist 1.4 $selList -samples $samples]
        set j 0
        foreach sasa $val {
            lappend residues([lindex $resLis $j]) $sasa
            incr j
        }
        lappend residues(total) [measure sasa 1.4 $global -restrict $restrict -samples $samples]
    }

    $global delete
    $restrict delete
    return "{[array get residues]} {$resLis}"
}


proc QWIKMD::callSASA {} {

    set answer [tk_messageBox -title "SASA calculation" -message "This calculation may take a long time. Do you want to proceed?" -type yesno -icon warning]
    if {$answer == "no"} {
        return
    }
    if {$QWIKMD::advGui(analyze,advance,sasaselentry) == "" || $QWIKMD::advGui(analyze,advance,sasaselentry) == "Type Selection"} {
        return
    }

    set restrict $QWIKMD::advGui(analyze,advance,sasarestselentry)
    if {$QWIKMD::advGui(analyze,advance,sasarestselentry) == "Type Selection" || $QWIKMD::advGui(analyze,advance,sasarestselentry) == ""} {
        set restrict $QWIKMD::advGui(analyze,advance,sasaselentry)
    }
    set const 2e-6
    set samples 50
    set probe 1.4
    set sasaResList [QWIKMD::calcSASA $QWIKMD::advGui(analyze,advance,sasaselentry) $restrict $probe $samples]
    array set residues [lindex $sasaResList 0]
    set reslist [lindex $sasaResList 1]
    unset sasaResList
    $QWIKMD::advGui(analyze,advance,sasatb) delete 0 end
    set selall [atomselect $QWIKMD::topMol "all"]
    $selall set user "0.00"
    $selall delete
    for {set i 0} {$i < [llength $reslist]} {incr i} {
        if {[lindex $reslist $i] != "total"} {
            set res [lindex $reslist $i]
            if {[llength $residues($res) ] > 1} {
                set avgstdv [QWIKMD::meanSTDV $residues($res)]
            } else {
                set avgstdv "$residues($res) 0.000"
            }
            $QWIKMD::advGui(analyze,advance,sasatb) insert end "$res [format %.3f [lindex $avgstdv 0]] [format %.3f [lindex $avgstdv 1]]"
            set sel [atomselect $QWIKMD::topMol "resid \"[lindex $res 0]\" and resname [lindex $res 1] and chain [lindex $res 2] "]
            $sel set user [lindex $avgstdv 0]
            $sel delete
            unset avgstdv
        }
    }
    #set resids [$QWIKMD::advGui(analyze,advance,sasatb) getcolumn 0]
    #set minres [QWIKMD::mincalc $resids]
    set total [llength $residues(total)]
    if {[molinfo $QWIKMD::topMol get numframes] > 1} {
        set avgstdv [QWIKMD::meanSTDV $residues(total)]
    } else {
        set avgstdv "$residues(total) 0.000"
    }
    
    set xsasa [list]
    set ysasa [list]
    set j 0
    set do 1
    set const 2e-6
    set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ]
    set xtime 0
    for {set i 0} {$i < $total} {incr i} {
        
        if {$i < [lindex $QWIKMD::lastframe $j]} {
            
            if {$do == 1} {
                set logfile [open [lindex $QWIKMD::confFile $j].log r]
                while {[eof $logfile] != 1 } {
                    set line [gets $logfile]

                    if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                        set const [expr [lindex $line 2] * 1e-6]
                    }

                    if {[lindex $line 0] == "Info:" && [join [lrange $line 1 2]] == "DCD FREQUENCY" } {
                        set QWIKMD::dcdfreq [lindex $line 3]
                        break
                    }
                }
                close $logfile
                set do 0
                set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ]
            }       
        } else {
            incr j
            set do 1
        }
        if {$i > 0}  {
            set xtime [expr [lindex $xsasa end] + $increment]
        }
        lappend xsasa $xtime
        lappend ysasa [lindex $residues(total) $i] 
    }
    if {$QWIKMD::SASAGui == ""} {
        set info [QWIKMD::addplot sasa "SASA Plot" "Total SASA vs Time" "Time (ns)" "SASA (A\u00b2)"]
        set QWIKMD::SASAGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            if {$QWIKMD::sasarep != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol
                set QWIKMD::sasarep ""
            }
                $QWIKMD::SASAGui clear
                $QWIKMD::SASAGui add 0 0
                $QWIKMD::SASAGui replot
        }

        $close entryconfigure 0 -command {
            if {$QWIKMD::sasarep != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol
                set QWIKMD::sasarep ""
                foreach m [molinfo list] {
                    if {[string compare [molinfo $m get name] "{Color Scale Bar}"] == 0} {
                      mol delete $m
                    }
                }
            }
            $QWIKMD::SASAGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).sasa
            set QWIKMD::SASAGui ""
        }
        if {[file channels $QWIKMD::textLogfile] == $QWIKMD::textLogfile && $QWIKMD::textLogfile != ""} {
            puts $QWIKMD::textLogfile [QWIKMD::printSASA [lindex $xsasa end] [llength $xsasa] $QWIKMD::advGui(analyze,advance,sasaselentry) $restrict]
            flush $QWIKMD::textLogfile
        }
        
    } else {
        $QWIKMD::SASAGui clear
        $QWIKMD::SASAGui add 0 0
        $QWIKMD::SASAGui replot
    }   
    $QWIKMD::SASAGui clear
    $QWIKMD::SASAGui add $xsasa $ysasa
    $QWIKMD::SASAGui replot
    if {$QWIKMD::sasarep == ""} {
        mol addrep $QWIKMD::topMol
        set val [$QWIKMD::advGui(analyze,advance,sasatb) getcolumns 3]
        set min [QWIKMD::mincalc $val]
        set max [QWIKMD::maxcalc $val]
        set QWIKMD::sasarep [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
        mol modcolor [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol "User"
        mol modselect [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol "\($QWIKMD::advGui(analyze,advance,sasaselentry)\) and \($restrict\)"
        mol selupdate [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol on
        set rep $QWIKMD::advGui(analyze,advance,sasarep)
        if {$rep == "VDW"} {
            set rep "$rep 1.0 12.0"
        }
        mol modstyle [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol $rep
        if {$min == ""} {
            set min 0
        }
        if {$max == ""} {
            set max 0
        }
        ::ColorScaleBar::color_scale_bar 0.8 0.05 0 1 [expr round($min)] [expr round($max)] 5 white 0 -1.0 0.8 1 $QWIKMD::topMol 0 1 "SASA"
        color scale method BWR
    }
    $QWIKMD::advGui(analyze,advance,sasatb) insert end "#S Total Total [format %.3f [lindex $avgstdv 0]] [format %.3f [lindex $avgstdv 1]]"
    $QWIKMD::advGui(analyze,advance,sasatb) sortbycolumn 0
}

proc QWIKMD::callCSASA {} {
    set answer [tk_messageBox -title "Cont. Surface Area calculation" -message "This calculation may take a long time. Do you want to proceed?" -type yesno -icon warning]
    if {$answer == "no"} {
        return
    }
    if {$QWIKMD::advGui(analyze,advance,sasaselentry) == "" || $QWIKMD::advGui(analyze,advance,sasaselentry) == "Type Selection"} {
        return
    }

    if {$QWIKMD::advGui(analyze,advance,sasarestselentry) == "" || $QWIKMD::advGui(analyze,advance,sasarestselentry) == "Type Selection" || $QWIKMD::advGui(analyze,advance,sasarestselentry) == $QWIKMD::advGui(analyze,advance,sasaselentry)} {
        return
    }

    if {$QWIKMD::CSASAGui == ""} {
        if {$QWIKMD::sasarep != ""} {
            mol delrep [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol
            set QWIKMD::sasarep ""
        }
        set info [QWIKMD::addplot csasa "Cont Area Plot" "Total Contact Area vs Time" "Time (ns)" "Surface Area (A\u00b2)"]
        set QWIKMD::CSASAGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            if {$QWIKMD::sasarep != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol
                set QWIKMD::sasarep ""
            }
            $QWIKMD::CSASAGui clear
            $QWIKMD::CSASAGui add 0 0
            $QWIKMD::CSASAGui replot
        }

        $close entryconfigure 0 -command {
            if {$QWIKMD::sasarep != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol
                set QWIKMD::sasarep ""
                foreach m [molinfo list] {
                    if {[string compare [molinfo $m get name] "{Color Scale Bar}"] == 0} {
                      mol delete $m
                    }
                }
            }
            $QWIKMD::CSASAGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).csasa
            set QWIKMD::CSASAGui ""
        }

    } else {
        $QWIKMD::CSASAGui clear
        $QWIKMD::CSASAGui add 0 0
        $QWIKMD::CSASAGui replot
    } 

    set restrict $QWIKMD::advGui(analyze,advance,sasarestselentry)
    if {$QWIKMD::advGui(analyze,advance,sasarestselentry) == "Type Selection"} {
        set restrict $QWIKMD::advGui(analyze,advance,sasaselentry)
    }

    set const 2e-6
    set samples 50
    set probe 1.4
    array set residues1 ""
    array set residues2 ""
    set reslist ""
    set totABAVG0 [list]
    set totAB0 [list]
    set totABAVG1 [list]
    set totAB1 [list]
    $QWIKMD::advGui(analyze,advance,sasatb) delete 0 end
    set all [atomselect $QWIKMD::topMol "all"]
    $all set user 0.000
    $all delete
    set numframe [molinfo $QWIKMD::topMol get numframes]
    for {set j 0} {$j < 2} {incr j} {
        if {$j == 0} {

            set globalsel "\($QWIKMD::advGui(analyze,advance,sasaselentry)\) or \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)"
            set restrictsel "\(within 5 of \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)\) and \($QWIKMD::advGui(analyze,advance,sasaselentry)\)"

            set sasaResList [QWIKMD::calcSASA $globalsel $restrictsel $probe $samples]
            array set residues1 [lindex $sasaResList 0]
            set globalsel "\($QWIKMD::advGui(analyze,advance,sasaselentry)\)"

            set sasaResList [QWIKMD::calcSASA $globalsel $restrictsel $probe $samples]
            array set residues2 [lindex $sasaResList 0]
            set reslist [lindex $sasaResList 1]
        } else {
            set globalsel "\($QWIKMD::advGui(analyze,advance,sasaselentry)\) or \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)"
            set restrictsel "\(within 5 of \($QWIKMD::advGui(analyze,advance,sasaselentry)\)\) and \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)"
            
            set sasaResList [QWIKMD::calcSASA $globalsel $restrictsel $probe $samples]
            array set residues1 [lindex $sasaResList 0]
            set globalsel "\($QWIKMD::advGui(analyze,advance,sasarestselentry)\)"
            set sasaResList [QWIKMD::calcSASA $globalsel $restrictsel $probe $samples]
            array set residues2 [lindex $sasaResList 0]
            set reslist [lindex $sasaResList 1]
        }
        lappend reslist "total"     

        for {set i 0} {$i < [llength $reslist]} {incr i} {
            
                set res [lindex $reslist $i]
                set length [llength $residues1($res)]
                set diff [list]
                for {set index 0} {$index < $length} {incr index} {
                    lappend diff [expr abs([lindex $residues1($res) $index] - [lindex $residues2($res) $index]) ]
                }
                while {[llength $diff] < $numframe} {
                    lappend diff 0.00
                }
                
                if {[llength $diff] > 1} {
                    set avgstdv [QWIKMD::meanSTDV $diff]
                } else {
                    set avgstdv [list $diff 0.0]
                }
                
                if {$res != "total"} {
                    $QWIKMD::advGui(analyze,advance,sasatb) insert end "$res [format %.3f [lindex $avgstdv 0] ] [format %.3f [lindex $avgstdv 1]]"
                    set sel [atomselect $QWIKMD::topMol "resid \"[lindex $res 0]\" and resname [lindex $res 1] and chain [lindex $res 2] "]
                    $sel set user [lindex $avgstdv 0]
                    $sel delete
                } else {
                    set totABAVG$j [list [format %.3f [lindex $avgstdv 0]] [format %.3f [lindex $avgstdv 1]] ]
                    set totAB$j $diff       
                }           
                set avgstdv ""
        }       
        set sasaResList ""
        
    }
    #set resids [$QWIKMD::advGui(analyze,advance,sasatb) getcolumn 0]
    #set minres [QWIKMD::mincalc $resids]
    set total [llength $residues1(total)]
    set xsasa [list]
    set j 0
    set const 2e-6
    set do 1
    set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ]
    set xtime 0
    for {set i 0} {$i < $total} {incr i} {
        if {$i < [lindex $QWIKMD::lastframe $j]} {
                    
            if {$do == 1} {
                set logfile [open [lindex $QWIKMD::confFile $j].log r]
                while {[eof $logfile] != 1 } {
                    set line [gets $logfile]

                    if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                        set const [expr [lindex $line 2] * 1e-6]
                    }

                    if {[lindex $line 0] == "Info:" && [join [lrange $line 1 2]] == "DCD FREQUENCY" } {
                        set QWIKMD::dcdfreq [lindex $line 3]
                        break
                    }
                }
                close $logfile
                set do 0
                set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ]
            }       
        } else {
            incr j
            set do 1
        }
        if {$i > 0}  {
            set xtime [expr [lindex $xsasa end] + $increment]
        }
        lappend xsasa $xtime
    }
    $QWIKMD::CSASAGui clear
    $QWIKMD::CSASAGui add $xsasa $totAB0 -legend "Total1_2"
    $QWIKMD::CSASAGui add $xsasa $totAB1 -legend "Total2_1"
    $QWIKMD::CSASAGui replot
    if {$QWIKMD::sasarep == ""} {
        if {[file channels $QWIKMD::textLogfile] == $QWIKMD::textLogfile && $QWIKMD::textLogfile != ""} {
            puts $QWIKMD::textLogfile [QWIKMD::printContSASA [lindex $xsasa end] [llength $xsasa] $QWIKMD::advGui(analyze,advance,sasaselentry) $QWIKMD::advGui(analyze,advance,sasarestselentry)  ]    
            flush $QWIKMD::textLogfile
        }

        mol addrep $QWIKMD::topMol
        set val [$QWIKMD::advGui(analyze,advance,sasatb) getcolumns 3]
        set min [QWIKMD::mincalc $val]
        set max [QWIKMD::maxcalc $val]
        set QWIKMD::sasarep [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
        mol modcolor [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol "User"
        
        set seltext "same residue as \(\(\(within 5 of \($QWIKMD::advGui(analyze,advance,sasaselentry)\)\) and \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)\) or \(\(within 5 of \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)\) and \($QWIKMD::advGui(analyze,advance,sasaselentry)\)\)\)"
        mol modselect [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol $seltext
        mol selupdate [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol on
        set rep $QWIKMD::advGui(analyze,advance,sasarep)
        if {$rep == "VDW"} {
            set rep "$rep 1.0 12.0"
        }
        mol modstyle [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol $QWIKMD::advGui(analyze,advance,sasarep)
        set color "white"
        if {$QWIKMD::basicGui(desktop) == "white"} {
            set color "black"
        }
        if {$min == ""} {
            set min 0
        }
        if {$max == ""} {
            set max 0
        }
        ::ColorScaleBar::color_scale_bar 0.8 0.05 0 1 [expr round($min)] [expr round($max)] 5 $color 0 -1.0 0.8 1 $QWIKMD::topMol 0 1 "Cont Area"
        color scale method BWR
    }
    $QWIKMD::advGui(analyze,advance,sasatb) insert end "#S1 Total1_2 Total1_2 [format %.3f [lindex $totABAVG0 0 ]] [format %.3f [lindex  $totABAVG0  1]]"

    $QWIKMD::advGui(analyze,advance,sasatb) insert end "#S2 Total2_1 Total2_1 [format %.3f [lindex $totABAVG1 0]] [format %.3f [lindex $totABAVG1  1]]"
    $QWIKMD::advGui(analyze,advance,sasatb) sortbycolumn 0
}

proc QWIKMD::RMSFCalc {} {
    if {$QWIKMD::rmsfGui == ""} {
        set info [QWIKMD::addplot rmsf "RMSF Plot" "RMSF vs Residue Number" "Residue Number" "RMSF (A)"]
        set QWIKMD::rmsfGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            if {$QWIKMD::rmsfrep != ""} {
            mol delrep [QWIKMD::getrepnum $QWIKMD::rmsfrep] $QWIKMD::topMol
            set QWIKMD::rmsfrep ""
        }
            $QWIKMD::rmsfGui clear
            $QWIKMD::rmsfGui add 0 0
            $QWIKMD::rmsfGui replot
        }

        $close entryconfigure 0 -command {
            if {$QWIKMD::rmsfrep != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::rmsfrep] $QWIKMD::topMol
                set QWIKMD::rmsfrep ""
                foreach m [molinfo list] {
                    if {[string compare [molinfo $m get name] "{Color Scale Bar}"] == 0} {
                      mol delete $m
                    }
                }
            }
            $QWIKMD::rmsfGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).rmsf
            set QWIKMD::rmsfGui ""
        }

    } else {
        $QWIKMD::rmsfGui clear
        $QWIKMD::rmsfGui add 0 0
        $QWIKMD::rmsfGui replot
    } 

    if {$QWIKMD::load == 1 && $QWIKMD::advGui(analyze,advance,rmsfselentry) != "Type Selection"} {
        set xresid ""
        set rmsf ""
        set numframes [expr {$QWIKMD::advGui(analyze,advance,rmsfto) - $QWIKMD::advGui(analyze,advance,rmsffrom)} / $QWIKMD::advGui(analyze,advance,rmsfskip)]
        set all [atomselect $QWIKMD::topMol "all"]
        $all set user 0.000
        $all delete
        if {$numframes >= 1} {
            set sel [atomselect $QWIKMD::topMol $QWIKMD::advGui(analyze,advance,rmsfselentry)]
            if {$QWIKMD::advGui(analyze,advance,rmsfalicheck) == 1} {
                set numframesaux [molinfo $QWIKMD::topMol get numframes]
                set alignsel ""
                if {$QWIKMD::advGui(analyze,advance,rmsfalignsel) != "" && $QWIKMD::advGui(analyze,advance,rmsfalignsel) != "Type Selection"} {
                    set alignsel $QWIKMD::advGui(analyze,advance,rmsfalignsel)
                } else {
                    set alignsel $QWIKMD::advGui(analyze,advance,rmsfaligncomb)
                }
                set alisel [atomselect $QWIKMD::topMol $alignsel frame $QWIKMD::advGui(analyze,advance,rmsffrom)]
                set move_sel [atomselect $QWIKMD::topMol "all" frame 0]
                
                for {set i 0} {$i < $numframesaux} {incr i} {
                    $move_sel frame $i
                    set auxsel [atomselect $QWIKMD::topMol $alignsel frame $i]
                    set tmatrix [measure fit $auxsel $alisel]
                    $move_sel move $tmatrix
                    $auxsel delete
                }
                $move_sel delete
                $alisel delete
            }
            $sel set user 0.0
            set rmsflist [measure rmsf $sel first $QWIKMD::advGui(analyze,advance,rmsffrom) last $QWIKMD::advGui(analyze,advance,rmsfto) step $QWIKMD::advGui(analyze,advance,rmsfskip)]
            $sel set user $rmsflist

            set straux ""
            set resindex 1
            set min 100
            set max 0
            foreach resid [$sel get resid] chain [$sel get chain] {
                set str ${resid}_${chain}
                if {$str != $straux} {
                    set selcalc [atomselect $QWIKMD::topMol  "resid \"$resid\" and chain \"$chain\""]
                    lappend xresid $resindex
                    set values [$selcalc get user]
                    lappend rmsf [QWIKMD::mean $values]
                    set minaux [QWIKMD::mincalc $values]
                    if {$minaux < $min} {
                        set min $minaux
                    }
                    set maxaux [QWIKMD::maxcalc $values]
                    if {$maxaux > $max} {
                        set max $maxaux
                    }
                    $selcalc delete
                    set straux $str
                    incr resindex
                }
            }
            $sel delete
        }
        $QWIKMD::rmsfGui clear
        $QWIKMD::rmsfGui configure -nolines -raius 5 
        $QWIKMD::rmsfGui add $xresid $rmsf
        $QWIKMD::rmsfGui replot
        if {$QWIKMD::rmsfrep == ""} {
            if {[file channels $QWIKMD::textLogfile] == $QWIKMD::textLogfile && $QWIKMD::textLogfile != ""} {
                puts $QWIKMD::textLogfile [QWIKMD::printRMSF $QWIKMD::advGui(analyze,advance,rmsffrom) $QWIKMD::advGui(analyze,advance,rmsfto) $QWIKMD::advGui(analyze,advance,rmsfskip) $QWIKMD::advGui(analyze,advance,rmsfselentry) ]
                flush $QWIKMD::textLogfile
            }
            mol addrep $QWIKMD::topMol
            set QWIKMD::rmsfrep [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
            mol modcolor [QWIKMD::getrepnum $QWIKMD::rmsfrep] $QWIKMD::topMol "User"
            mol modselect [QWIKMD::getrepnum $QWIKMD::rmsfrep] $QWIKMD::topMol $QWIKMD::advGui(analyze,advance,rmsfselentry)
            set rep $QWIKMD::advGui(analyze,advance,rmsfrep)
            if {$rep == "VDW"} {
                set rep "$rep 1.0 12.0"
            }
            mol modstyle [QWIKMD::getrepnum $QWIKMD::rmsfrep] $QWIKMD::topMol $rep 
            if {$min == ""} {
                set min 0
            }
            if {$max == ""} {
                set max 0
            }
            ::ColorScaleBar::color_scale_bar 0.8 0.05 0 1 [expr round($min)] [expr round($max)] 5 white 0 -1.0 0.8 1 $QWIKMD::topMol 0 1 "RMSF"
            color scale method BWR
        }
    }
}
proc QWIKMD::EneCalc {} {
    set do 1
    set const [expr $QWIKMD::timestep * 1e-6]
    set tot 0
    set kin 0
    set pot 0
    set bond 0
    set angle 0
    set dihedral 0
    set vdw 0
    if {$QWIKMD::energyTotGui != ""} {set tot 1}
    if {$QWIKMD::energyPotGui != ""} {set pot 1}
    if {$QWIKMD::energyKineGui != ""} {set kin 1}
    if {$QWIKMD::energyBondGui != ""} {set bond 1}
    if {$QWIKMD::energyAngleGui != ""} {set angle 1}
    if {$QWIKMD::energyDehidralGui != "" } {set dihedral 1}
    if {$QWIKMD::energyVdwGui != ""} {set vdw 1}
    set xtime [expr {$const * $QWIKMD::counterts * $QWIKMD::imdFreq} + $QWIKMD::eneprevx]
    if {$tot == 1} {
        $QWIKMD::energyTotGui clear
    
        lappend QWIKMD::enetotval [molinfo $QWIKMD::topMol get energy]
        lappend QWIKMD::enetotpos $xtime
        $QWIKMD::energyTotGui add $QWIKMD::enetotpos $QWIKMD::enetotval
        if {[lindex $QWIKMD::enetotval 0] == [lindex $QWIKMD::enetotval 1] && [lindex $QWIKMD::enetotval 1] == [lindex $QWIKMD::enetotval end]} {
            $QWIKMD::energyTotGui configure -ymin [expr [lindex $QWIKMD::enetotval 1] -1] -ymax [expr [lindex $QWIKMD::enetotval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::energyTotGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::energyTotGui replot
    
    }

    if {$kin == 1} {
        $QWIKMD::energyKineGui clear
        lappend QWIKMD::enekinval [molinfo $QWIKMD::topMol get kinetic]
        lappend QWIKMD::enekinpos $xtime
        $QWIKMD::energyKineGui add $QWIKMD::enekinpos $QWIKMD::enekinval
        if {[lindex $QWIKMD::enekinval 0] == [lindex $QWIKMD::enekinval 1] && [lindex $QWIKMD::enekinval 1] == [lindex $QWIKMD::enekinval end]} {
            $QWIKMD::energyKineGui configure -ymin [expr [lindex $QWIKMD::enekinval 1] -1] -ymax [expr [lindex $QWIKMD::enekinval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::energyKineGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::energyKineGui replot
    }

    if {$pot == 1} {
        $QWIKMD::energyPotGui clear
        lappend QWIKMD::enepotval [molinfo $QWIKMD::topMol get potential]
        lappend QWIKMD::enepotpos $xtime
        $QWIKMD::energyPotGui add $QWIKMD::enepotpos $QWIKMD::enepotval
        if {[lindex $QWIKMD::enepotval 0] == [lindex $QWIKMD::enepotval 1] && [lindex $QWIKMD::enepotval 1] == [lindex $QWIKMD::enepotval end]} {
            $QWIKMD::energyPotGui configure -ymin [expr [lindex $QWIKMD::enepotval 1] -1] -ymax [expr [lindex $QWIKMD::enepotval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::energyPotGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::energyPotGui replot
    }


    if {$bond == 1} {
        $QWIKMD::energyBondGui clear
    
        lappend QWIKMD::enebondval [molinfo $QWIKMD::topMol get bond]
        lappend QWIKMD::enebondpos $xtime
        $QWIKMD::energyBondGui add $QWIKMD::enebondpos $QWIKMD::enebondval
        if {[lindex $QWIKMD::enebondval 0] == [lindex $QWIKMD::enebondval 1] && [lindex $QWIKMD::enebondval 1] == [lindex $QWIKMD::enebondval end]} {
            $QWIKMD::energyBondGui configure -ymin [expr [lindex $QWIKMD::enebondval 1] -1] -ymax [expr [lindex $QWIKMD::enebondval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::energyBondGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::energyBondGui replot
    
    }

    if {$angle == 1} {
        $QWIKMD::energyAngleGui clear
        lappend QWIKMD::eneangleval [molinfo $QWIKMD::topMol get angle]
        lappend QWIKMD::eneanglepos $xtime
        $QWIKMD::energyAngleGui add $QWIKMD::eneanglepos $QWIKMD::eneangleval
        if {[lindex $QWIKMD::eneangleval 0] == [lindex $QWIKMD::eneangleval 1] && [lindex $QWIKMD::eneangleval 1] == [lindex $QWIKMD::eneangleval end]} {
            $QWIKMD::energyAngleGui configure -ymin [expr [lindex $QWIKMD::eneangleval 1] -1] -ymax [expr [lindex $QWIKMD::eneangleval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::energyAngleGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::energyAngleGui replot
    }

    if {$dihedral == 1} {
        $QWIKMD::energyDehidralGui clear
        lappend QWIKMD::enedihedralval [molinfo $QWIKMD::topMol get dihedral]
        lappend QWIKMD::enedihedralpos $xtime
        $QWIKMD::energyDehidralGui add $QWIKMD::enedihedralpos $QWIKMD::enedihedralval 
        if {[lindex $QWIKMD::enedihedralval 0] == [lindex $QWIKMD::enedihedralval 1] && [lindex $QWIKMD::enedihedralval 1] == [lindex $QWIKMD::enedihedralval end]} {
            $QWIKMD::energyDehidralGui configure -ymin [expr [lindex $QWIKMD::enedihedralval 1] -1] -ymax [expr [lindex $QWIKMD::enedihedralval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::energyDehidralGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::energyDehidralGui replot
    }

    if {$vdw == 1} {
        $QWIKMD::energyVdwGui clear
        lappend QWIKMD::enevdwval [molinfo $QWIKMD::topMol get vdw]
        lappend QWIKMD::enevdwpos $xtime
        $QWIKMD::energyVdwGui add $QWIKMD::enevdwpos $QWIKMD::enevdwval
        if {[lindex $QWIKMD::enevdwval 0] == [lindex $QWIKMD::enevdwval 1] && [lindex $QWIKMD::enevdwval 1] == [lindex $QWIKMD::enevdwval end]} {
            $QWIKMD::energyVdwGui configure -ymin [expr [lindex $QWIKMD::enevdwval 1] -1] -ymax [expr [lindex $QWIKMD::enevdwval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::energyVdwGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::energyVdwGui replot
    }
    
}

proc QWIKMD::CondCalc {} {
    set do 1
    set const [expr $QWIKMD::timestep * 1e-6]   
    set tempaux 0
    set pressaux 0
    set volaux 0
    
    if {$QWIKMD::tempGui != ""} {set tempaux 1}
    if {$QWIKMD::pressGui != ""} {set pressaux 1}
    if {$QWIKMD::volGui != ""} {set volaux 1}
    set xtime [expr {$const * $QWIKMD::counterts * $QWIKMD::imdFreq} + $QWIKMD::condprevx]
    if {$tempaux ==1} {
        $QWIKMD::tempGui clear
            
        lappend QWIKMD::tempval [molinfo $QWIKMD::topMol get temperature]
        lappend QWIKMD::temppos $xtime

        $QWIKMD::tempGui add $QWIKMD::temppos $QWIKMD::tempval
        if {[lindex $QWIKMD::tempval 0] == [lindex $QWIKMD::tempval 1] && [lindex $QWIKMD::tempval 1] == [lindex $QWIKMD::tempval end]} {
            $QWIKMD::tempGui configure -ymin [expr [lindex $QWIKMD::tempval 1] -1] -ymax [expr [lindex $QWIKMD::tempval 1] +1] -xmin auto -xmax auto
        } else {
            $QWIKMD::tempGui configure -ymin auto -ymax auto -xmin auto -xmax auto
        }
        $QWIKMD::tempGui replot
        
    }

    if {$pressaux ==1 || $volaux == 1} {
        
        set index [expr $QWIKMD::state -1 ]
        set file "[lindex $QWIKMD::confFile $index].log"
        set prefix ""               
        set timeX "0"
        set const [expr $QWIKMD::timestep * 1e-6]
        set logfile [open $file r]
        seek $logfile $QWIKMD::condcurrentpos
        set dist ""
        set time ""
        set limit [expr $QWIKMD::calcfreq * $QWIKMD::imdFreq]
        set prevts [expr {[expr $QWIKMD::counterts -$QWIKMD::prevcounterts] * $QWIKMD::imdFreq}]
        if {$prevts < 0} {
            set prevts 0
        }
        while {[eof $logfile] != 1 } {
            set line [gets $logfile]
             
            if {[lindex $line 0] == "ENERGY:" && [lindex $line 1] != 0 && [lindex $line 1] < $prevts && $prevts != 0} {
                set line [gets $logfile]
            }
            if {[lindex $line 0] == "ENERGY:" && [lindex $line 1] != 0 && [lindex $line 1] > $QWIKMD::condprevindex} {
                if {$pressaux ==1} {
                    lappend  QWIKMD::pressval [lindex $line 19]
                }
                if {$volaux ==1} {
                    lappend  QWIKMD::volval [lindex $line 18]
                }
                
                set time [lindex $line 1]
                if {[expr $time - $QWIKMD::condprevindex] >= $limit} {
                    set xtime [expr {$const * $time} + $QWIKMD::condprevx]
                    set min 0
                    set QWIKMD::condprevindex $time
                    set QWIKMD::condcurrentpos [tell $logfile ]
                    close $logfile
                    if {$pressaux == 1} {
                        $QWIKMD::pressGui clear
                        if {[llength $QWIKMD::pressvalavg] < 2} {
                            set min [expr int([expr [llength $QWIKMD::pressval] - [expr 1.5 * $limit] -1])]  
                        }
                        set max [expr [llength $QWIKMD::pressval] -1]
                        lappend QWIKMD::pressvalavg [QWIKMD::mean [lrange $QWIKMD::pressval $min $max]]
                        lappend QWIKMD::presspos $xtime
                        $QWIKMD::pressGui add $QWIKMD::presspos $QWIKMD::pressvalavg
                        if {[llength $QWIKMD::pressvalavg] >= 2} {
                            if {[QWIKMD::format5Dec [lindex $QWIKMD::pressvalavg 0] ] == [QWIKMD::format5Dec [lindex $QWIKMD::pressvalavg 1]] && [QWIKMD::format5Dec [lindex $QWIKMD::pressvalavg 1]] == [QWIKMD::format5Dec [lindex $QWIKMD::pressvalavg end]]} {
                                $QWIKMD::pressGui configure -ymin [expr [lindex $QWIKMD::pressvalavg 1] -1] -ymax [expr [lindex $QWIKMD::pressvalavg 1] +1] -xmin auto -xmax auto
                            } else {
                                $QWIKMD::pressGui configure -ymin auto -ymax auto -xmin auto -xmax auto
                            }
                        } else {
                            $QWIKMD::pressGui configure -ymin auto -ymax auto -xmin auto -xmax auto
                        }
                        
                        $QWIKMD::pressGui replot
                    }
                    set min 0
                    if {$volaux == 1} {
                        $QWIKMD::volGui clear
                        if {[llength $QWIKMD::volvalavg] < 2} {
                            set min [expr int([expr [llength $QWIKMD::volval] - [expr 1.5 * $limit] -1])]  
                        }
                        set max [expr [llength $QWIKMD::volval] -1]
                        lappend QWIKMD::volvalavg [QWIKMD::mean [lrange $QWIKMD::volval $min $max]]
                        lappend QWIKMD::volpos $xtime
                        $QWIKMD::volGui add $QWIKMD::volpos $QWIKMD::volvalavg
                        if {[llength $QWIKMD::volvalavg] >= 2} {
                            if {[QWIKMD::format5Dec [lindex $QWIKMD::volvalavg 0]] == [QWIKMD::format5Dec [lindex $QWIKMD::volvalavg 1]] && [QWIKMD::format5Dec [lindex $QWIKMD::volvalavg 1]] == [QWIKMD::format5Dec [lindex $QWIKMD::volvalavg end]]} {
                            $QWIKMD::volGui configure -ymin [expr [lindex $QWIKMD::volvalavg 1] -1] -ymax [expr [lindex $QWIKMD::volvalavg 1] +1] -xmin auto -xmax auto
                            } else {
                                $QWIKMD::volGui configure -ymin auto -ymax auto -xmin auto -xmax auto
                            }
                        } else {
                            $QWIKMD::volGui configure -ymin auto -ymax auto -xmin auto -xmax auto
                        }
                        
                        $QWIKMD::volGui replot
                    } 
                    break
                }
            }
        }
    }           
    #### When these values pass through IMD connection, uncomment
    # if {$pressaux ==1} {
    #   $QWIKMD::pressGui clear

    #   lappend QWIKMD::pressval [molinfo $QWIKMD::topMol get pressure]
    #   lappend QWIKMD::presspos $xtime
    #   $QWIKMD::pressGui add $QWIKMD::presspos $QWIKMD::pressval
    #   if {[lindex $QWIKMD::pressval 0] == [lindex $QWIKMD::pressval 1] && [lindex $QWIKMD::pressval 1] == [lindex $QWIKMD::pressval end]} {
 #              $QWIKMD::pressGui configure -ymin [expr [lindex $QWIKMD::pressval 1] -1] -ymax [expr [lindex $QWIKMD::pressval 1] +1] -xmin auto -xmax auto
 #          } else {
 #              $QWIKMD::pressGui configure -ymin auto -ymax auto -xmin auto -xmax auto
 #          }
    #   $QWIKMD::pressGui replot
    # }

    # if {$volaux == 1} {
    #   $QWIKMD::volGui clear
        
    #   lappend QWIKMD::volval [molinfo $QWIKMD::topMol get volume]
    #   lappend QWIKMD::volpos $xtime
    #   $QWIKMD::volGui add $QWIKMD::volpos $QWIKMD::volval
    #   if {[lindex $QWIKMD::volval 0] == [lindex $QWIKMD::volval 1] && [lindex $QWIKMD::volval 1] == [lindex $QWIKMD::volval end]} {
 #              $QWIKMD::volGui configure -ymin [expr [lindex $QWIKMD::volval 1] -1] -ymax [expr [lindex $QWIKMD::volval 1] +1] -xmin auto -xmax auto
 #          } else {
 #              $QWIKMD::volGui configure -ymin auto -ymax auto -xmin auto -xmax auto
 #          }
    #   $QWIKMD::volGui replot
    # }
    
}
proc QWIKMD::SpecificHeatCalc {} {
    

    set title " Total Energy vs Time"
    set ylab "Energy\n(kcal/mol)"
    set xlab "Time (TimeSteps)"
    if {$QWIKMD::SPHGui == ""} {
        set info [QWIKMD::addplot sph "Specific Heat" $title $xlab $ylab]
        set QWIKMD::SPHGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            $QWIKMD::SPHGui clear
            
            $QWIKMD::SPHGui add 0 0
            $QWIKMD::SPHGui replot
        }

        $close entryconfigure 0 -command {
            $QWIKMD::SPHGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).sph
            set QWIKMD::tempDistGui ""
            set QWIKMD::SPHGui ""
        }
    } else {
        $QWIKMD::SPHGui clear
        $QWIKMD::SPHGui add 0 0
        $QWIKMD::SPHGui replot
    }
    set mol [mol new [lindex  ${QWIKMD::outPath}/run/$QWIKMD::inputstrct 0]]
    mol addfile  ${QWIKMD::outPath}/run/${QWIKMD::radiobtt}.dcd waitfor all
    
    if {[catch {glob *.inp} auxlist] == 0} {
        foreach file $auxlist {
            lappend parlist "-par $file"
        }
    }
    foreach par $QWIKMD::ParameterList {
        lappend parlist "-par ${QWIKMD::outPath}/run/[file tail $par]"
    }
    
    
    set sel ""
    if {[catch {atomselect $mol $QWIKMD::advGui(analyze,advance,selentry)} sel] == 0} {
        set command "namdenergy -sel $sel -all [join $parlist] -ofile ${QWIKMD::outPath}/run/$QWIKMD::radiobtt.namdenergy.dat "
        eval $command
    }
    

    set templatefile [open  ${QWIKMD::outPath}/run/$QWIKMD::radiobtt.namdenergy.dat r]
    set line [read -nonewline $templatefile]
    set line [split $line "\n"]
    close $templatefile
    set enter ""
    set lineIndex [lsearch -exact -all $line $enter]
    set avg 0.0
    set avgsqr 0.0
    set energypos ""
    set energyval ""
    for {set i 1} {$i < [llength $line]} {incr i} {
        set aux [lindex [lindex $line $i] 10]
        set avg [expr $avg + $aux]
        set avgsqr [expr $avgsqr + pow($aux,2)]
        lappend energyval [lindex [lindex $line $i] 10]
        lappend energypos [lindex [lindex $line $i] 0]
    }

    set avg [expr $avg / [expr [llength $line] -1] ]
    set avgsqr [expr $avgsqr / [expr [llength $line] -1] ]
    
    set QWIKMD::advGui(analyze,advance,kcal) [format %.3g [expr [expr $avgsqr - pow($avg,2)] / [expr $QWIKMD::advGui(analyze,advance,bkentry) * pow([expr $QWIKMD::advGui(analyze,advance,tempentry) + 273],2)]]]
    
    set m [expr [vecsum [$sel get mass]] * 1.66e-27] 
    set QWIKMD::advGui(analyze,advance,joul) [format %.3g [expr $QWIKMD::advGui(analyze,advance,kcal) / [expr 1.4386e20 * $m]]]
    $QWIKMD::SPHGui clear
    $QWIKMD::SPHGui add $energypos $energyval
    $QWIKMD::SPHGui replot
    $sel delete
    mol delete $mol
}
proc QWIKMD::TempDistCalc {} {
    set title "Temperature vs Time"
    set ylab "Temperature (K)"
    set xlab "Time (TimeSteps)"
    if {$QWIKMD::tempDistGui == ""} {
        set info [QWIKMD::addplot tmpdist "Temp Distribution" $title $xlab $ylab]
        set QWIKMD::tempDistGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            $QWIKMD::tempDistGui clear
            $QWIKMD::tempDistGui add 0 0
            $QWIKMD::tempDistGui replot
        }

        $close entryconfigure 0 -command {
            $QWIKMD::tempDistGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).tmpdist
            set QWIKMD::tempDistGui ""
        }
        set QWIKMD::actempGui ""
    } else {
        $QWIKMD::tempDistGui clear
        $QWIKMD::tempDistGui add 0 0
        $QWIKMD::tempDistGui replot
    }
    if {$QWIKMD::load == 1} {
        set tempdistpos ""
        set tempdistval ""
        set templatefile [open "${QWIKMD::radiobtt}.log" r]
        set line [read $templatefile]
        set line [split $line "\n"]
        close $templatefile
        set enter ""
        set index [lsearch -exact -all $line $enter]
        for {set i 0} {$i < [llength $index]} {incr i} {
            lset line [lindex $index $i] "{} {}"
        }

        set index [lsearch -index 0 -all $line "ENERGY:"]
        for {set i 0} {$i < [llength $index]} {incr i} {
            lappend tempdistval [lindex [lindex $line [lindex $index $i]] 15]
            lappend tempdistpos [lindex [lindex $line [lindex $index $i]] 1]
        }
        $QWIKMD::tempDistGui clear
        $QWIKMD::tempDistGui add $tempdistpos $tempdistval
        $QWIKMD::tempDistGui replot
        unset line
    }
}

proc QWIKMD::MBCalC {} {
    set title "kinetic Energy vs Atom Index"
    set ylab "Energy\n(kcal/mol)"
    set xlab "Atom Index"
    if {$QWIKMD::MBGui == ""} {
        set info [QWIKMD::addplot mb "MB Distribution" $title $xlab $ylab]
        set QWIKMD::MBGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            $QWIKMD::MBGui clear
            $QWIKMD::MBGui add 0 0
            $QWIKMD::MBGui replot
        }

        $close entryconfigure 0 -command {
            $QWIKMD::MBGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).mb
            set QWIKMD::MBGui ""
        }
        set QWIKMD::actempGui ""
    } else {
        $QWIKMD::MBGui clear
        $QWIKMD::MBGui add 0 0
        $QWIKMD::MBGui replot
    } 

    if {$QWIKMD::load == 1} {
        puts "${QWIKMD::radiobtt}.restart.vel"
        set file "${QWIKMD::radiobtt}.restart.vel"
        if {[file exists $file] !=1} {
            break
        }
        set mol [mol new [lindex $QWIKMD::inputstrct 0]]
        mol addfile $file type namdbin
        set mbpos ""
        set mbval ""
        set all [atomselect top $QWIKMD::advGui(analyze,advance,MBsel)]
        foreach m [$all get mass] v [$all get {x y z}] index [$all get index] {
            lappend mbval [expr 0.5 * $m * [vecdot $v $v] ]
            lappend mbpos $index
        }

        $QWIKMD::MBGui clear
        $QWIKMD::MBGui configure -nolines -raius 5 
        $QWIKMD::MBGui add $mbpos $mbval
        $QWIKMD::MBGui replot
        mol delete $mol
    }


}

proc QWIKMD::QTempCalc {} {
    set title "Temperature vs Time"
    set ylab "Temp (K)"
    set xlab "Time (fs)"
    if {$QWIKMD::qtempGui == ""} {
        set info [QWIKMD::addplot tquench "Temperature Quench" $title $xlab $ylab]
        set QWIKMD::qtempGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            $QWIKMD::qtempGui clear
            set QWIKMD::qtemppos ""
            set QWIKMD::qtempval ""
            $QWIKMD::qtempGui add 0 0
            $QWIKMD::qtempGui replot
        }

        $close entryconfigure 0 -command {
            $QWIKMD::qtempGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).tquench
            set QWIKMD::qtempGui ""
            set QWIKMD::qtemppos ""
            set QWIKMD::qtempval ""
        }
        set QWIKMD::actempGui ""
    } else {
        $QWIKMD::qtempGui clear
        set QWIKMD::qtemppos ""
        set QWIKMD::qtempval ""
        $QWIKMD::qtempGui add 0 0
        $QWIKMD::qtempGui replot
    } 

    set title "Temperature Autocorrelation Function"
    set ylab "C T,T"
    set xlab "Time (fs)"
    if {$QWIKMD::actempGui == ""} {
        set info [QWIKMD::addplot actquench "Temperature AC" $title $xlab $ylab]
        set QWIKMD::actempGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            $QWIKMD::actempGui clear
        set QWIKMD::acqtemppos ""
        set QWIKMD::acqtempval ""
        $QWIKMD::actempGui add 0 0
        $QWIKMD::actempGui replot
        }

        $close entryconfigure 0 -command {
            destroy $QWIKMD::advGui(analyze,advance,ntb).actquench
            set QWIKMD::actempGui ""
            set QWIKMD::acqtemppos ""
            set QWIKMD::acqtempval ""
        }

    } else {
        $QWIKMD::actempGui clear
        set QWIKMD::acqtemppos ""
        set QWIKMD::acqtempval ""
        $QWIKMD::actempGui add 0 0
        $QWIKMD::actempGui replot
    }

    if {$QWIKMD::load == 1} {
        set QWIKMD::qtemprevx 0
        set QWIKMD::qtemppos ""
        set QWIKMD::qtempval ""
        set energyfreq 1
        set const 2e-6
        set countdo 0
        for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
            set tps 0
            set tblwindw [$QWIKMD::advGui(analyze,advance,qtmeptbl) windowpath $i,0]
            set do 0
            if {[$tblwindw.r state !selected] == "selected"} {
                set do 1
                incr countdo
                $tblwindw.r state selected
            } else {
                set do 0
                $tblwindw.r state !selected
            }

            if {$do == 1} {
                set file "[lindex $QWIKMD::confFile $i].log"
                if {[file exists $file] !=1} {
                    break
                }
                set logfile [open $file r]
                set lineprev ""

                while {[eof $logfile] != 1 } {
                    set line [gets $logfile]
                    if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                        set aux [lindex $line 2]
                        set const $aux 
                    }
                    if {[lindex $line 0] == "Info:" && [join [lrange $line 1 3]] == "ENERGY OUTPUT STEPS" } {
                        set energyfreq [lindex $line 4]
                    }
                    if {[lindex $line 0] == "ENERGY:"} {
                        lappend QWIKMD::qtempval [lindex $line 15]  
                        lappend QWIKMD::qtemppos [expr {$tps * $energyfreq * $const} + $QWIKMD::qtemprevx]
                        incr tps
                    } 
                }
                close $logfile
                $QWIKMD::advGui(analyze,advance,qtmeptbl) cellconfigure $i,2 -text [expr $tps -1]
                set QWIKMD::qtemprevx [lindex $QWIKMD::qtemppos end]
            
                if {$countdo == 1} {
                    set previndex [lsearch $QWIKMD::prevconfFile [lindex $QWIKMD::confFile $i]]
                    if {$previndex < 0} {
                        set previndex 0
                    }
                    set QWIKMD::advGui(analyze,advance,tempentry) [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $previndex,4 -text]

                    set QWIKMD::acqtemppos ""
                    set QWIKMD::acqtempval ""
                    set endlag 25
                    set lc [llength $QWIKMD::qtempval]
                    set temper ""
                    set avg_temp [QWIKMD::mean $QWIKMD::qtempval]
                    set temper_adj ""
                    set data1 ""
                    set data2 ""
                    set data2sq ""
                    set dataprod ""
                    for {set k 0} {$k < $lc} {incr k} {
                        lappend temper_adj [expr [lindex $QWIKMD::qtempval $k] - $avg_temp]
                        lappend data1 0.0
                        lappend data2 0.0
                        lappend data2sq 0.0
                        lappend dataprod 0.0 
                    }

                    for {set lag 0} {$lag <= $endlag} {incr lag} {
                        
                        for {set k 0} {$k < [expr $lc-$lag]} {incr k} {
                            lset data1 $k [lindex $temper_adj $k]
                            lset data2 $k [lindex $temper_adj [expr $k+$lag]]
                            lset data2sq $k [expr [lindex $data2 $k] * [lindex $data2 $k]]
                            lset dataprod $k [expr [lindex $data1 $k] * [lindex $data2 $k]]

                        }
                        set mean1 [QWIKMD::mean $data1]
                        set mean2 [QWIKMD::mean $data2sq]
                        set meanprod [QWIKMD::mean $dataprod]
                        # Calculate the Autocorrelation Function
                        lappend QWIKMD::acqtemppos $lag
                        lappend QWIKMD::acqtempval [expr ($meanprod - $mean1*$mean1)/($mean2 - $mean1*$mean1)]
                    }
                    $QWIKMD::actempGui clear
                    $QWIKMD::actempGui add $QWIKMD::acqtemppos $QWIKMD::acqtempval
                    $QWIKMD::actempGui replot
                } 
            }
        }

        $QWIKMD::qtempGui clear
        $QWIKMD::qtempGui add $QWIKMD::qtemppos $QWIKMD::qtempval
        $QWIKMD::qtempGui replot
        set QWIKMD::qtemprevx [lindex $QWIKMD::qtemppos end]

    } 
    
}
proc QWIKMD::QFindEcho {} {
    set last 0
    for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
        set tps 0
        set tblwindw [$QWIKMD::advGui(analyze,advance,qtmeptbl) windowpath $i,0]
        set do 0
        if {[$tblwindw.r state !selected] == "selected"} {
            set do 1
            set last $i
            incr countdo
            $tblwindw.r state selected
        } else {
            set do 0
            $tblwindw.r state !selected
        }
    }
    set prevtime [$QWIKMD::advGui(analyze,advance,qtmeptbl) cellcget $last,2 -text]
    set lastquench [expr [llength $QWIKMD::qtempval] - $prevtime]
    set min [expr $lastquench +10]
    set depth [QWIKMD::mincalc [lrange $QWIKMD::qtempval $min end]]
    set avg [QWIKMD::mean [lrange $QWIKMD::qtempval $min end]]
    $QWIKMD::advGui(analyze,advance,echolb) configure -text "[format %.3f [expr $avg - $depth]] K"
    set time [lindex $QWIKMD::qtemppos [lsearch $QWIKMD::qtempval $depth]]
    $QWIKMD::advGui(analyze,advance,echotime) configure -text "[format %.3f $time] fs"

    set tau [$QWIKMD::advGui(analyze,advance,qtmeptbl) cellcget [expr $last -1],2 -text]
    set tau0 $QWIKMD::advGui(analyze,advance,decayentry)    ;# autocorrelation decay time
    set T0 [expr $QWIKMD::advGui(analyze,advance,tempentry) + 273]  ;# initial temperature
    set T1 0    ;# first temperature reassignment
    set T2 0    ;# sedcond temperature reassignment
    set lambda1 [expr sqrt($T1/$T0)]
    set lambda2 [expr sqrt($T2/$T0)]

    set y1 [expr (1 + pow($lambda1,2) + 2*pow($lambda2,2)) / 4]


    set length [llength $QWIKMD::qtemppos]
    set x [list]
    set y [list]
    for {set t $tau} {$t < [expr $tau + $prevtime]} {incr t} {
      set y2 [expr (1 + pow($lambda1,2) - 2*pow($lambda2,2)) / 4 * [expr exp(-($t-$tau)/$tau0)]]
      set y3 [expr $lambda1*$lambda2/2 * [expr exp(-abs($t-3*$tau/2)/$tau0)]]
      set y4 [expr (1 - pow($lambda1,2)) / 8 * [expr exp(-abs($t-2*$tau)/$tau0)]]
      lappend x [expr $lastquench -1 + $t - $tau] 
      lappend y [expr $T0*($y1-$y2-$y3-$y4)]
    }
    $QWIKMD::qtempGui clear
    $QWIKMD::qtempGui add $QWIKMD::qtemppos $QWIKMD::qtempval
    $QWIKMD::qtempGui add $x $y
    $QWIKMD::qtempGui replot
}

proc QWIKMD::callhbondsCalcProc {} {
    #set plot 0
     if {$QWIKMD::hbondsGui == ""} {
        #set plot 1
        set info [QWIKMD::addplot fhbonds "HBonds Plot" "HBonds vs Time" "Time (ns)" "No. HBonds"]
        set QWIKMD::hbondsGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            if {$QWIKMD::hbondsrepname != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::hbondsrepname] $QWIKMD::topMol
                set QWIKMD::hbondsrepname ""
            }
            $QWIKMD::hbondsGui clear
            set QWIKMD::timeXhbonds ""
            set QWIKMD::hbonds ""
            $QWIKMD::hbondsGui add 0 0
            $QWIKMD::hbondsGui replot
        }

        $close entryconfigure 0 -command {
            if {$QWIKMD::hbondsrepname != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::hbondsrepname] $QWIKMD::topMol
                set QWIKMD::hbondsrepname ""
            }
            $QWIKMD::hbondsGui quit
            destroy $QWIKMD::advGui(analyze,advance,ntb).fhbonds
            set QWIKMD::hbondsGui ""
        }

    } else {
        $QWIKMD::hbondsGui clear
        set QWIKMD::timeXhbonds ""
        set QWIKMD::hbonds ""
        $QWIKMD::hbondsGui add 0 0
        $QWIKMD::hbondsGui replot
    } 
    if {$QWIKMD::hbondssel == "sel" && ($QWIKMD::advGui(analyze,advance,hbondsel1entry) == "Type Selection" || $QWIKMD::advGui(analyze,advance,hbondsel1entry) == "")} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::hbondsrepname] $QWIKMD::topMol
        set QWIKMD::hbondsrepname ""
        return
    }
    if {$QWIKMD::load == 1} {
        set numframes [molinfo $QWIKMD::topMol get numframes]
        
        set j 0
        set hbonds 0
        set const 2e-6 
        set do 1
        set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ] 
        set xtime 0
        for {set i 0} {$i < $numframes} {incr i} {
            if {$i < [lindex $QWIKMD::lastframe $j]} {
                if {$do == 1} {
                    set logfile [open [lindex $QWIKMD::confFile $j].log r]
                    while {[eof $logfile] != 1 } {
                        set line [gets $logfile]

                        if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                            set const [expr [lindex $line 2] * 1e-6]
                        }

                        if {[lindex $line 0] == "Info:" && [join [lrange $line 1 2]] == "DCD FREQUENCY" } {
                            set QWIKMD::dcdfreq [lindex $line 3]
                            break
                        }
                    }
                    close $logfile
                    set do 0
                    set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ]
                }   
            } else {
                incr j
                set do 1
            }
            if {$i > 0}  {
                set xtime [expr [lindex $QWIKMD::timeXhbonds end] + $increment]
            }
            lappend QWIKMD::timeXhbonds $xtime
            lappend QWIKMD::hbonds [QWIKMD::hbondsCalcProc $i]
    
        }
        
        QWIKMD::representHbonds
        
        set QWIKMD::hbondsprevx [lindex $QWIKMD::timeXhbonds end]
        if {[file channels $QWIKMD::textLogfile] == $QWIKMD::textLogfile && $QWIKMD::textLogfile != ""} {
            puts $QWIKMD::textLogfile [QWIKMD::printHbonds $numframes]
            flush $QWIKMD::textLogfile
        } 
    } elseif {$QWIKMD::load == 0} {
        QWIKMD::HbondsCalc
    }
}

proc QWIKMD::hbondsCalcProc {frame} {
    set atmsel1 ""
    set atmsel2 ""
    set polar "(name \"N.*\" \"O.*\" \"S.*\" FA F1 F2 F3)"
    if {$QWIKMD::hbondssel == "inter" || $QWIKMD::hbondssel == "intra"} {
        set sel "all and not water and not ions and $polar"
        set atmsel1 [atomselect $QWIKMD::topMol $sel frame $frame]      
    } 
    
    if {$QWIKMD::hbondssel == "inter"} {
        set sel "water"
        set atmsel2 [atomselect $QWIKMD::topMol $sel frame $frame]
    
    } elseif {$QWIKMD::hbondssel == "sel"} {
        set atmsel1 [atomselect $QWIKMD::topMol "$QWIKMD::advGui(analyze,advance,hbondsel1entry) and $polar" frame $frame]
        if {$QWIKMD::advGui(analyze,advance,hbondsel2entry) != "Type Selection" } {
            set atmsel2 [atomselect $QWIKMD::topMol "$QWIKMD::advGui(analyze,advance,hbondsel2entry) and $polar" frame $frame]
        }
        
    }
    set hbonds -1
    if {[$atmsel1 get index] != ""} {   
        if {$atmsel2 == ""} {
            set hbonds [llength [lindex [measure hbonds 3.5 30 $atmsel1] 0]]
        } else {
            set hbonds [llength [lindex [measure hbonds 3.5 30 $atmsel1 $atmsel2] 0] ]
            set hbonds [expr $hbonds + [llength [lindex [measure hbonds 3.5 30 $atmsel2 $atmsel1] 0] ]]
            $atmsel2 delete
        }
        $atmsel1 delete
    }
    return $hbonds
}

proc QWIKMD::representHbonds {} {
    set sel1 "all and not water and not ions and (name \"N.*\" \"O.*\" \"S.*\" FA F1 F2 F3)"
    if {$QWIKMD::hbondssel == "inter"} {
        set sel2 "water"
    }
    set selrep $sel1
    if {$QWIKMD::hbondssel == "inter"} {
        append selrep " or water"
    }
    if {$QWIKMD::hbondsrepname != ""} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::hbondsrepname] $QWIKMD::topMol
        set QWIKMD::hbondsrepname ""
    }
    
    if {$QWIKMD::hbondssel == "intra"} {
        mol representation HBonds 3.5 30 10
        mol color Name
        mol selection $selrep
        mol material Opaque
        mol addrep $QWIKMD::topMol
        set QWIKMD::hbondsrepname [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
    }
    
    $QWIKMD::hbondsGui clear
    $QWIKMD::hbondsGui add $QWIKMD::timeXhbonds $QWIKMD::hbonds
    $QWIKMD::hbondsGui replot
}

proc QWIKMD::HbondsCalc {} {
    set prefix ""
        
    set hbonds [QWIKMD::hbondsCalcProc [molinfo $QWIKMD::topMol get frame]]
    if {$hbonds != -1} {    
        set const [expr $QWIKMD::timestep * 1e-6]

        lappend QWIKMD::timeXhbonds [expr {$const * $QWIKMD::counterts * $QWIKMD::imdFreq} + $QWIKMD::hbondsprevx]
        lappend QWIKMD::hbonds $hbonds
        
        QWIKMD::representHbonds
        
    } else {
        tk_messageBox -message "Atom selection is not valid." -title "Hbonds Calculation" -icon info -type ok
    }
}

proc QWIKMD::SmdCalc {} {
    set do 0
    set index [expr $QWIKMD::state -1 ]
    set file "[lindex $QWIKMD::confFile $index].log"
    if {[$QWIKMD::topGui.nbinput tab 0 -state] == "normal"} {
        if {[file exists $file] ==1 && [string match "*smd*" [lindex  $QWIKMD::confFile $index ] ] > 0} {
            set do 1
        }
        
    } elseif {[$QWIKMD::topGui.nbinput tab 1 -state] == "normal"} {
        if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) == 1 && [file exists $file] ==1} {
            set do 1
        }
    }
    if {$do ==1} {

        set prefix ""               
        set timeX "0"
        set const [expr $QWIKMD::timestep * 1e-6]
        set logfile [open $file r]
        seek $logfile $QWIKMD::smdcurrentpos
        set dist ""
        set time ""
        set limit [expr $QWIKMD::calcfreq * $QWIKMD::imdFreq]

        while {[eof $logfile] != 1 } {
            set line [gets $logfile]
            if {[lindex $line 0] == "SMD" && [lindex $line 1] != 0 && [lindex $line 1] > $QWIKMD::smdprevindex} {
                lappend  QWIKMD::smdvals [lindex $line 7]
                set time [lindex $line 1]
                if {$QWIKMD::smdxunit != "time"} {
                    set dist [lindex $line 4]
                    if {$QWIKMD::smdfirstdist == ""} {
                        set QWIKMD::smdfirstdist $dist
                    }
                }
                if {[expr $time - $QWIKMD::smdprevindex] >= $limit} {
                    set min 0
                    if {[llength $QWIKMD::smdvalsavg] < 2} {
                        set min [expr int([expr [llength $QWIKMD::smdvals] - [expr 1.5 * $limit] -1])]  
                    }
                    set max [expr [llength $QWIKMD::smdvals] -1]
                    lappend QWIKMD::smdvalsavg [QWIKMD::mean [lrange $QWIKMD::smdvals $min $max]]
                    if {$QWIKMD::smdxunit == "time"} {
                        lappend QWIKMD::timeXsmd [expr $const * $time]
                    } else {
                        lappend QWIKMD::timeXsmd [expr $dist - $QWIKMD::smdfirstdist]
                    }
                    set QWIKMD::smdprevindex $time
                    set QWIKMD::smdcurrentpos [tell $logfile ]
                    close $logfile      
                    $QWIKMD::smdGui clear
                    $QWIKMD::smdGui add $QWIKMD::timeXsmd $QWIKMD::smdvalsavg
                    $QWIKMD::smdGui replot   
                    break
                }
            }
        }           
    }
}

proc QWIKMD::callSmdCalc {} {
    
    if {$QWIKMD::smdGui == ""} {
        set title "Force vs Time"
        set xlab "Time (ns)"
        if {$QWIKMD::smdxunit == "distance"} {
            set title "Force vs Distance"
            set xlab "Distance (A)"
        }
        set info [QWIKMD::addplot smd "SMD Plot" $title $xlab "Force (pN)"]
        set QWIKMD::smdGui [lindex $info 0]

        set clear [lindex $info 1]
        set close [lindex $info 2]
        
        $clear entryconfigure 0 -command {
            $QWIKMD::smdGui clear
            set QWIKMD::timeXsmd ""
            set QWIKMD::smdvals ""
            set QWIKMD::smdvalsavg ""
            set QWIKMD::smdfirstdist ""
            $QWIKMD::smdGui add 0 0
            $QWIKMD::smdGui replot
        }

        $close entryconfigure 0 -command {

            destroy $QWIKMD::advGui(analyze,advance,ntb).smd
            $QWIKMD::smdGui quit
            set QWIKMD::smdGui ""
        }

    } else {
        $QWIKMD::smdGui clear
        set QWIKMD::timeXsmd ""
        set QWIKMD::smdvals ""
        set QWIKMD::smdvalsavg ""
        set QWIKMD::smdfirstdist ""
        $QWIKMD::smdGui add 0 0
        $QWIKMD::smdGui replot
    } 
    if {$QWIKMD::load == 1 && $QWIKMD::run == "SMD"} {
        set j 0
        set QWIKMD::smdprevx 0
        set QWIKMD::smdprevindex 0
        set findfirst 1
        set firstdistance 0
        for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
            set do 0
            set file "[lindex  $QWIKMD::confFile $i].log"
            if {[$QWIKMD::topGui.nbinput tab 0 -state] == "normal"} {
                if {[file exists $file] ==1 && [string match "*smd*" $file ] > 0} {
                    set do 1
                }
                
            } elseif {[$QWIKMD::topGui.nbinput tab 1 -state] == "normal"} {
                set confIndex [lsearch $QWIKMD::prevconfFile [lindex  $QWIKMD::confFile $i]]
                if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$confIndex,smd) == 1 && [file exists $file] ==1} {
                    set do 1
                }   
            }
            if {$do == 1} {
                set prefix ""               
                
                set const 2e-6

                set logfile [open $file r]
                set dist ""
                set time ""
                set const 0
                set limit [expr $QWIKMD::calcfreq * $QWIKMD::imdFreq]
                set smdfreq 0
                set tstepaux 0
                set index 0
                while {[eof $logfile] != 1 } {
                    set line [gets $logfile]
                    if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                        set aux [lindex $line 2]
                        set const [expr $aux * 1e-6] 
                        set tstepaux 0
                    }
            
                    if {[lindex $line 0] == "Info:" && [join [lrange $line 1 3]] == "SMD OUTPUT FREQUENCY" } {
                        set smdfreq [lindex $line 4]
                        if {$QWIKMD::basicGui(live) == 0} {
                            set limit [expr $smdfreq * 10] 
                        }
                    }
                    if {[lindex $line 0] == "SMD" && [lindex $line 1] != 0 } {
                        lappend  QWIKMD::smdvals [lindex $line 7]
                
                        if {$QWIKMD::smdxunit == "time"} {
                            set time [lindex $line 1]
                            
                        } else {
                            set dist [lindex $line 4]
                            if {$findfirst == 1} {
                                set firstdistance $dist
                                set findfirst 0
                            }
                        }
                        incr tstepaux $smdfreq
                        incr index $smdfreq
                        if {[expr $tstepaux % $limit] == 0 && $index != $QWIKMD::smdprevindex} {
                            set min 0
                            if {[llength $QWIKMD::smdvalsavg] < 2} {
                                set min [expr int([expr [llength $QWIKMD::smdvals] - [expr 1.5 * $limit] -1])]  
                            }
                            set max [expr [llength $QWIKMD::smdvals] -1]
                            lappend QWIKMD::smdvalsavg [QWIKMD::mean [lrange $QWIKMD::smdvals $min $max]]
                            if {$QWIKMD::smdxunit == "time"} {
                                lappend QWIKMD::timeXsmd [expr $const * $time]
                            } else {
                                lappend QWIKMD::timeXsmd [expr $dist - $firstdistance]
                            }
                            set QWIKMD::smdprevx [lindex $QWIKMD::timeXsmd end]
                        
                        }
                     }
                }
                if {[file channels $QWIKMD::textLogfile] == $QWIKMD::textLogfile && $QWIKMD::textLogfile != ""} {
                    puts $QWIKMD::textLogfile [QWIKMD::printSMD $QWIKMD::smdprevx $dist $limit $smdfreq $const]
                    flush $QWIKMD::textLogfile 
                }
                close $logfile      
            }
            
        }
        $QWIKMD::smdGui clear
        $QWIKMD::smdGui add $QWIKMD::timeXsmd $QWIKMD::smdvalsavg
        $QWIKMD::smdGui replot

    } 
    
}


proc QWIKMD::CallTimeLine {} {
    
    menu timeline on
}

proc QWIKMD::AddMBBox {} {
    display resetview
    set sel [atomselect $QWIKMD::topMol "all and not water and not ion"]

    set center [measure center $sel]
    set QWIKMD::advGui(membrane,center,x) [lindex $center 0]
    set QWIKMD::advGui(membrane,center,y) [lindex $center 1]
    set QWIKMD::advGui(membrane,center,z) [lindex $center 2]
    set QWIKMD::advGui(membrane,rotationMaxtrixList) [list]
    $sel delete

    set QWIKMD::advGui(membrane,rotate,x) 0
    set QWIKMD::advGui(membrane,rotate,y) 0
    set QWIKMD::advGui(membrane,rotate,z) 0

    set QWIKMD::advGui(membrane,centerxoffset) 0
    set QWIKMD::advGui(membrane,centeryoffset) 0

    set QWIKMD::advGui(membrane,trans,x) 0
    set QWIKMD::advGui(membrane,trans,y) 0
    set QWIKMD::advGui(membrane,trans,z) 0

    QWIKMD::updateMembraneBox $center
}

proc QWIKMD::updateMembraneBox {center} {
    # if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 0} {
    #   set center [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]
    #   set centermover [transoffset [vecscale $center -1]]
    #   set centerback [transoffset $center]
    #   set i 0
    #   foreach coor $QWIKMD::advGui(membrane,boxedges) {
    #       set coor [coordtrans $centermover $coor]
    #       set matrix ""
    #       if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 1} {
    #           set matrix [eval transmult [lreverse $QWIKMD::advGui(membrane,rotationMaxtrixList)]]
    #       } else {
    #           set matrix [join $QWIKMD::advGui(membrane,rotationMaxtrixList)]
    #       }
    #       set coor [vectrans [measure inverse $matrix] $coor]
    #       lset QWIKMD::advGui(membrane,boxedges) $i [coordtrans $centerback $coor]
    #       incr i
    #   }
    # }
    set boxW [expr $QWIKMD::advGui(membrane,xsize)/2]
    set boxH [expr $QWIKMD::advGui(membrane,ysize)/2]
    ###Z box dimension from the average of the phosphorus atoms in the membrane popc36_box.pdb and popec36_box.pdb
    set zsize 0.0 ; #(0,0,0)
    if {$QWIKMD::advGui(membrane,lipid) == "POPC"} {
        set zsize 39.0
    } else {
        set zsize 39.4
    }
    set boxD [expr $zsize/2]
    # set QWIKMD::advGui(membrane,xmin) [expr [lindex $center 0]-($boxW) + $QWIKMD::advGui(membrane,centerxoffset)]
    # set QWIKMD::advGui(membrane,xmax) [expr [lindex $center 0]+($boxW) + $QWIKMD::advGui(membrane,centerxoffset)]

    # set QWIKMD::advGui(membrane,ymin) [expr [lindex $center 1]-($boxH) + $QWIKMD::advGui(membrane,centeryoffset)]
    # set QWIKMD::advGui(membrane,ymax) [expr [lindex $center 1]+($boxH) + $QWIKMD::advGui(membrane,centeryoffset)]

    set QWIKMD::advGui(membrane,xmin) [expr [lindex $center 0]-($boxW)]
    set QWIKMD::advGui(membrane,xmax) [expr [lindex $center 0]+($boxW)]

    set QWIKMD::advGui(membrane,ymin) [expr [lindex $center 1]-($boxH)]
    set QWIKMD::advGui(membrane,ymax) [expr [lindex $center 1]+($boxH)]

    set QWIKMD::advGui(membrane,zmin) [expr [lindex $center 2]-($boxD)]
    set QWIKMD::advGui(membrane,zmax) [expr [lindex $center 2]+($boxD)]
    set QWIKMD::advGui(membrane,boxedges) [list]
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmin) $QWIKMD::advGui(membrane,ymin) $QWIKMD::advGui(membrane,zmin)]; #(0,0,0) 0
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmin) $QWIKMD::advGui(membrane,ymin) $QWIKMD::advGui(membrane,zmax)]; #(0,0,1) 1
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmin) $QWIKMD::advGui(membrane,ymax) $QWIKMD::advGui(membrane,zmax)]; #(0,1,1) 2
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmax) $QWIKMD::advGui(membrane,ymax) $QWIKMD::advGui(membrane,zmax)]; #(1,1,1) 3
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmin) $QWIKMD::advGui(membrane,ymax) $QWIKMD::advGui(membrane,zmin)]; #(0,1,0) 4
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmax) $QWIKMD::advGui(membrane,ymin) $QWIKMD::advGui(membrane,zmin)]; #(1,0,0) 5
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmax) $QWIKMD::advGui(membrane,ymax) $QWIKMD::advGui(membrane,zmin)]; #(1,1,0) 6
    lappend QWIKMD::advGui(membrane,boxedges) [list $QWIKMD::advGui(membrane,xmax) $QWIKMD::advGui(membrane,ymin) $QWIKMD::advGui(membrane,zmax)]; #(1,0,1) 7
    


    #update box with dimensions and membrane thickness 
    set i 0
    if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 0} {
        set center [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]
        set newcenter [list $QWIKMD::advGui(membrane,centerxoffset) $QWIKMD::advGui(membrane,centeryoffset) 0]
        set centermover [transoffset [vecsub {0 0 0} $center]]
        
        set centerback [transoffset [vecsub $center {0 0 0}]]
        set centermovenew ""
        
        foreach coor $QWIKMD::advGui(membrane,boxedges) {
            set coor [coordtrans $centermover $coor]
            set matrix ""
            if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 1} {
                set matrix [eval transmult [lreverse $QWIKMD::advGui(membrane,rotationMaxtrixList)]]
            } else {
                set matrix [join $QWIKMD::advGui(membrane,rotationMaxtrixList)]
            }

            if {$i == 0 && ($QWIKMD::advGui(membrane,centerxoffset) != 0 || $QWIKMD::advGui(membrane,centeryoffset) != 0)} {
                set newcenter [vectrans $matrix $newcenter]
                set newcenter [coordtrans $centerback $newcenter]
                set centermovenew [transoffset [vecsub $newcenter $center]]
                set newcenter [coordtrans $centermovenew $center]
            } 
            set coor [vectrans $matrix $coor]
            set coor [coordtrans $centerback $coor]
            if {$QWIKMD::advGui(membrane,centerxoffset) != 0 || $QWIKMD::advGui(membrane,centeryoffset) != 0} {
                set coor [coordtrans $centermovenew $coor]
            }
            lset QWIKMD::advGui(membrane,boxedges) $i $coor
            incr i
        }
        if {$QWIKMD::advGui(membrane,centerxoffset) != 0 || $QWIKMD::advGui(membrane,centeryoffset) != 0} {
            puts "Center: $center\nNewCenter $newcenter"
            set QWIKMD::advGui(membrane,center,x) [lindex $newcenter 0]
            set QWIKMD::advGui(membrane,center,y) [lindex $newcenter 1]
            set QWIKMD::advGui(membrane,center,z) [lindex $newcenter 2]
        }
    }
    set QWIKMD::advGui(membrane,centerxoffset) 0
    set QWIKMD::advGui(membrane,centeryoffset) 0
    QWIKMD::DrawBox
}
proc QWIKMD::DrawBox {} {
    
    foreach point $QWIKMD::membranebox {
        graphics $QWIKMD::topMol delete $point  
    }
    set QWIKMD::membranebox [list]
    set width 4

    # box vertices 
    #(0,0,0) 0
    #(0,0,1) 1
    #(0,1,1) 2
    #(1,1,1) 3
    #(0,1,0) 4
    #(1,0,0) 5
    #(1,1,0) 6
    #(1,0,1) 7

    ##### x0(0,0,0) - x1(1,0,0)
    graphics $QWIKMD::topMol color red
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 0] [lindex $QWIKMD::advGui(membrane,boxedges) 5] width $width ]

    ##### y0(0,0,0) - y1(0,1,0)
    graphics $QWIKMD::topMol color green
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 0] [lindex $QWIKMD::advGui(membrane,boxedges) 4] width $width ]
    
    ##### z0(0,0,0) - z1(0,0,1)
    graphics $QWIKMD::topMol color blue
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 0] [lindex $QWIKMD::advGui(membrane,boxedges) 1] width $width ]

    graphics $QWIKMD::topMol color yellow

    incr width -1

    ##### (1,0,0) - (1,0,1)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 5] [lindex $QWIKMD::advGui(membrane,boxedges) 7] width $width]

    ##### (1,0,1) -  (0,0,1)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 7] [lindex $QWIKMD::advGui(membrane,boxedges) 1] width $width]

    ##### (0,1,0) -  (0,1,1)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 4] [lindex $QWIKMD::advGui(membrane,boxedges) 2] width $width]

    ##### (0,1,1) - (0,1,1)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 2] [lindex $QWIKMD::advGui(membrane,boxedges) 1] width $width]


    ##### (0,1,1) - (0,0,1)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 2] [lindex $QWIKMD::advGui(membrane,boxedges) 3] width $width]

    ##### (1,0,1) - (1,1,1)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 7] [lindex $QWIKMD::advGui(membrane,boxedges) 3] width $width]

    ##### (1,1,1) - (1,1,0)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 3] [lindex $QWIKMD::advGui(membrane,boxedges) 6] width $width]

    ##### (1,1,0) - (0,1,0)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 6] [lindex $QWIKMD::advGui(membrane,boxedges) 4] width $width]

    ##### (1,1,0) - (1,0,0)
    lappend QWIKMD::membranebox [graphics $QWIKMD::topMol line [lindex $QWIKMD::advGui(membrane,boxedges) 6] [lindex $QWIKMD::advGui(membrane,boxedges) 5] width $width]
    
    if {$QWIKMD::membraneFrame == ""} {
        graphics $QWIKMD::topMol materials on
        graphics $QWIKMD::topMol material Transparent

        ### xy lower plane
        
        lappend QWIKMD::membranebox [graphics $QWIKMD::topMol triangle [lindex $QWIKMD::advGui(membrane,boxedges) 0] [lindex $QWIKMD::advGui(membrane,boxedges) 4] [lindex $QWIKMD::advGui(membrane,boxedges) 5] ]
        lappend QWIKMD::membranebox [graphics $QWIKMD::topMol triangle [lindex $QWIKMD::advGui(membrane,boxedges) 4] [lindex $QWIKMD::advGui(membrane,boxedges) 5] [lindex $QWIKMD::advGui(membrane,boxedges) 6] ]

        ### xy upper plane
        
        lappend QWIKMD::membranebox [graphics $QWIKMD::topMol triangle [lindex $QWIKMD::advGui(membrane,boxedges) 1] [lindex $QWIKMD::advGui(membrane,boxedges) 7] [lindex $QWIKMD::advGui(membrane,boxedges) 2] ]
        lappend QWIKMD::membranebox [graphics $QWIKMD::topMol triangle [lindex $QWIKMD::advGui(membrane,boxedges) 2] [lindex $QWIKMD::advGui(membrane,boxedges) 7] [lindex $QWIKMD::advGui(membrane,boxedges) 3] ]
        graphics $QWIKMD::topMol materials off
        #graphics $QWIKMD::topMol material Opaque
    }

}

proc QWIKMD::incrMembrane {sign} {
    if {$QWIKMD::advGui(membrane,efect) == "translate"} {
        set v [list]
        switch $QWIKMD::advGui(membrane,axis) {
            x {
                set v [list ${sign}$QWIKMD::advGui(membrane,multi) 0 0]
            }
            y {
                set v [list 0 ${sign}$QWIKMD::advGui(membrane,multi) 0]
            }
            z {
                set v [list 0 0 ${sign}$QWIKMD::advGui(membrane,multi)]
            }
        }
        set QWIKMD::advGui(membrane,center,$QWIKMD::advGui(membrane,axis)) [expr $QWIKMD::advGui(membrane,center,$QWIKMD::advGui(membrane,axis)) $sign $QWIKMD::advGui(membrane,multi)]
        set QWIKMD::advGui(membrane,trans,$QWIKMD::advGui(membrane,axis)) [expr $QWIKMD::advGui(membrane,trans,$QWIKMD::advGui(membrane,axis)) $sign $QWIKMD::advGui(membrane,multi)]
        set matrix [transoffset $v]
        set i 0
        foreach coor $QWIKMD::advGui(membrane,boxedges) {
            lset QWIKMD::advGui(membrane,boxedges) $i [coordtrans $matrix $coor]
            incr i
        }
    } else {
        set center [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]
        set matrix [trans origin {0.0 0.0 0.0} axis $QWIKMD::advGui(membrane,axis) "${sign}$QWIKMD::advGui(membrane,multi)"  ]
        # if {$QWIKMD::advGui(membrane,rotationMaxtrixList) == 0} {
        #   set QWIKMD::advGui(membrane,rotationMaxtrixList) [list]
        #   lappend QWIKMD::advGui(membrane,rotationMaxtrixList) $matrix
        # } else {
            lappend QWIKMD::advGui(membrane,rotationMaxtrixList) $matrix
        # }
        set i 0
        if {$sign == "+"} {
            set value [expr $QWIKMD::advGui(membrane,rotate,$QWIKMD::advGui(membrane,axis)) + $QWIKMD::advGui(membrane,multi)]
        } else {
            set value [expr $QWIKMD::advGui(membrane,rotate,$QWIKMD::advGui(membrane,axis)) - $QWIKMD::advGui(membrane,multi)]
        }
        set QWIKMD::advGui(membrane,rotate,$QWIKMD::advGui(membrane,axis)) $value
        set centermover [transoffset [vecsub {0 0 0} $center]]
        set centerback [transoffset [vecsub $center {0 0 0}]]
        foreach coor $QWIKMD::advGui(membrane,boxedges) {
            set coor [coordtrans $centermover $coor]
            set coor [vectrans $matrix $coor]
            lset QWIKMD::advGui(membrane,boxedges) $i [coordtrans $centerback $coor]
            incr i
        }
    }
    QWIKMD::updateMembraneBox [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]
}

proc QWIKMD::OptSize {} {


    set center [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]
    set membrane [atomselect $QWIKMD::membraneFrame "all"]
    set centermover [transoffset [vecsub {0 0 0} $center]]  

    $membrane move $centermover
    #set center [measure center $membrane]
    set matrix ""
    if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 0} {
        if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 1} {
            set matrix [eval transmult [lreverse $QWIKMD::advGui(membrane,rotationMaxtrixList)]]
        } else {
            set matrix [join $QWIKMD::advGui(membrane,rotationMaxtrixList)]
        }
        set inv [measure inverse $matrix]
        $membrane move $inv
    }
    # set matrixCenter [transoffset [vecsub {0 0 0} [measure center $membrane]]]
    # $membrane move $matrixCenter
    #set center [measure center $membrane]

    update idletasks
    set protein [atomselect $QWIKMD::membraneFrame "not water and not ion and not lipid" ]
    set limits [measure minmax $protein]
    $protein delete
    # set all [atomselect $QWIKMD::membraneFrame "all"]
    # set alimits [measure minmax $membrane]
    # $all delete
    $membrane delete
    set xmin [expr [lindex [lindex $limits 0] 0] - 15]
    set xmax [expr [lindex [lindex $limits 1] 0] + 15]

    set ymin [expr [lindex [lindex $limits 0] 1] - 15]
    set ymax [expr [lindex [lindex $limits 1] 1] + 15]

    set pxcenter [expr [expr $xmax + $xmin] /2]
    set pycenter [expr [expr $ymax + $ymin] /2]


    set xlength [expr round(($xmax - $xmin) + 0.5)]
    set ylength [expr round(($ymax - $ymin) + 0.5)]
    
    set QWIKMD::advGui(membrane,xsize) $xlength
    set QWIKMD::advGui(membrane,ysize) $ylength

    ## Calculate the difference (offset) between the center of box and the center of the protein. 
    ## Move the box to the center of x,y axis of the protein 
    if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 0} {
        set QWIKMD::advGui(membrane,centerxoffset) $pxcenter
        set QWIKMD::advGui(membrane,centeryoffset) $pycenter
    }

    QWIKMD::updateMembraneBox [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]
    QWIKMD::GenerateMembrane
    QWIKMD::DrawBox
}

proc QWIKMD::GenerateMembrane {} {
    global env
    if {[llength $QWIKMD::membranebox] == 0} {
        QWIKMD::AddMBBox
    }
    QWIKMD::deleteMembrane
    catch {membrane -l $QWIKMD::advGui(membrane,lipid) -x $QWIKMD::advGui(membrane,xsize) -y $QWIKMD::advGui(membrane,ysize) -top c36}
    set auxframe [molinfo top]
    set membrane [atomselect $auxframe all]

    set length [expr [llength [array names QWIKMD::chains]] /3]
    set txt ""
    for {set i 0} {$i < $length} {incr i} {
        if {$QWIKMD::chains($i,0) == 1} {
            append txt " ([lindex $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)]) or" 
        }
        
    }
    set txt [string trimleft $txt " "]
    set txt [string trimright $txt " or"]
    set topmol [atomselect $QWIKMD::topMol $txt]

    set center [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]

    if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 0 } {
        if {[llength $QWIKMD::advGui(membrane,rotationMaxtrixList)] > 1} {
            set QWIKMD::advGui(membrane,rotationMaxtrix) [eval transmult [lreverse $QWIKMD::advGui(membrane,rotationMaxtrixList) ]] 
        } else {
            set QWIKMD::advGui(membrane,rotationMaxtrix) [join $QWIKMD::advGui(membrane,rotationMaxtrixList) ]
        }
        $membrane move $QWIKMD::advGui(membrane,rotationMaxtrix)
    }
    set matrixCenter [transoffset [vecsub $center [measure center $membrane ]]]
    $membrane move $matrixCenter
    update idletasks
    set auxframe2 [::TopoTools::selections2mol "$topmol $membrane"]
    
    $membrane delete
    $topmol delete
    
    set length [expr [llength [array names QWIKMD::chains]] /3]
    set txt ""
    for {set i 0} {$i < $length} {incr i} {
        if {$QWIKMD::chains($i,0) == 1} {
            append txt " ([lindex $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)]) or" 
        }
        
    }
    set txt [string trimleft $txt " "]
    set txt [string trimright $txt " or"]
    set seltail "(same residue as ((all within 2 of ($txt)) and not name \"N.*\" \"O.*\" \"P.*\") and chain W L)"
    set selhead "(same residue as ((all within 2.5 of ($txt)) and name \"N.*\" \"O.*\" \"P.*\") and chain W L )"

    set sel [atomselect $auxframe2 "(all and not (($seltail) or ($selhead) ))" ]
    $sel writepdb ${env(TMPDIR)}/membrane.pdb
    $sel delete
    mol delete $auxframe2

    mol new ${env(TMPDIR)}/membrane.pdb waitfor all
    set QWIKMD::membraneFrame [molinfo top]
    mol modselect 0 $QWIKMD::membraneFrame  "chain W L" 
    mol modstyle 0 $QWIKMD::membraneFrame  "Lines"
    mol color Name

    mol delete $auxframe
 
    mol top $QWIKMD::topMol
    $QWIKMD::advGui(solvent,$QWIKMD::run) configure -values "Explicit"
    set QWIKMD::advGui(solvent,0) "Explicit"
    QWIKMD::ChangeSolvent
    if {[llength $QWIKMD::membranebox] == 16} {
        foreach point [lrange $QWIKMD::membranebox [expr [llength $QWIKMD::membranebox] -4]  end] {
            graphics $QWIKMD::topMol delete $point  
        }
        set QWIKMD::membranebox [lrange $QWIKMD::membranebox 0 [expr [llength $QWIKMD::membranebox] -5]]
    }
 

}
proc QWIKMD::deleteMembrane {} {
    if {$QWIKMD::membraneFrame != ""} {
        mol delete $QWIKMD::membraneFrame
        set QWIKMD::membraneFrame ""
        $QWIKMD::advGui(solvent,$QWIKMD::run) configure -values {"Vacuum" "Implicit" "Explicit"}
        if {[llength $QWIKMD::membranebox] == 12} {
            QWIKMD::DrawBox
        }   
    }
}

##############################################
## Proc to create the table with all ResidueSelect
## and previus modificaitons, such as mutations and protonation
## states. Here, the qwikMD macros are used so the 
## it is possible to change molecule classification
## of VMD by default. 
###############################################
proc QWIKMD::SelResid {} {
    if {[winfo exists $QWIKMD::selResGui] != 1} {
        QWIKMD::SelResidBuild
        wm withdraw $QWIKMD::selResGui
    }
    $QWIKMD::selresTable delete 0 end
    set QWIKMD::rename ""
            

    set tabid [expr [$QWIKMD::topGui.nbinput index current] + 1 ]
    set table $QWIKMD::selresTable
    set maintable "$QWIKMD::topGui.nbinput.f$tabid.tableframe.tb"

    set tbchains [$maintable getcolumns 0]
    set tbtypes [$maintable getcolumns 2]
    
    if {$tbchains != ""} {
        set str ""
        for {set i 0} {$i < [llength $tbchains]} {incr i} {
                set straux ""
                switch [lindex $tbtypes $i] {
                    protein {
                        set straux $QWIKMD::proteinmcr 
                    }
                    nucleic {
                        set straux $QWIKMD::nucleicmcr 
                    }
                    glycan {
                        set straux $QWIKMD::glycanmcr 
                    }
                    lipid {
                        set straux $QWIKMD::lipidmcr 
                    }
                    hetero {
                        set straux $QWIKMD::heteromcr 
                    }
                    water {
                        set straux "water"
                    }
                    default {
                        set straux [lindex $tbtypes $i]
                    }
                }
                append str "(chain \"[lindex $tbchains $i]\" and $straux) or "
        }

        set str [string trimright $str "or "]
        set sel [atomselect top "$str"]

        set str " "
        set i 0
        set macrosstr [list]
        set defVal {protein nucleic glycan lipid hetero}
        foreach macros $QWIKMD::userMacros {
            if {[lsearch $defVal [lindex $macros 0]] == -1 } {
                lappend macrosstr [lindex $macros 0] 
            }   
        }

        set retype 0
        set insertedResidues [list]
        set listMol [list]
        foreach resid [$sel get resid] resname [$sel get resname] chain [$sel get chain]\
         protein [$sel get qwikmd_protein] nucleic [$sel get qwikmd_nucleic] glycan [$sel get qwikmd_glycan] lipid [$sel get qwikmd_lipid] hetero [$sel get qwikmd_hetero] \
         water [$sel get water] macros [$sel get $macrosstr] residue [$sel get residue] {
            lappend listMol [list $resid $resname $chain $protein $nucleic $glycan $lipid $hetero $water $macros $residue]
        }
        set listMol [lsort -unique $listMol]
        set listMol [lsort -index 10 -integer -increasing $listMol]
        foreach listEle $listMol {
            set resid [lindex $listEle 0]
            set resname [lindex $listEle 1]
            set chain [lindex $listEle 2]
            set protein [lindex $listEle 3]
            set nucleic [lindex $listEle 4]
            set glycan [lindex $listEle 5]
            set lipid [lindex $listEle 6]
            set hetero [lindex $listEle 7]
            set water [lindex $listEle 8]
            set macros [lindex $listEle 9]
            set updateMainTable 0
            if {[lsearch $insertedResidues "$resid $resname $chain"] == -1 } {
                
                set type "protein"
                if {$protein == 1} {
                    set type "protein"
                } elseif {$nucleic == 1} {
                    set type "nucleic"
                } elseif {$glycan == 1} {
                    set type "glycan"
                } elseif {$lipid == 1} {
                    set type "lipid"
                } elseif {$water == 1} {
                    set type "water"
                } elseif {$macros == 1} {
                    set macroName [lindex $macrosstr [lsearch $macros 1]]
                    set type $macroName
                    set typesel $macroName
                } elseif {$hetero == 1} {
                    set type "hetero"
                }
                $table insert end "$resid $resname $chain $type"
                lappend insertedResidues "$resid $resname $chain"
                set index [lsearch -exact $QWIKMD::mutindex "${resid}_$chain"]
                set newresid ""
                if {$index != -1} {
                    set newresid "[lindex $QWIKMD::mutate(${resid}_${chain}) 0] -> [lindex $QWIKMD::mutate(${resid}_${chain}) 1]"
                    set index [lsearch -exact $QWIKMD::protindex "${resid}_$chain"]
                    if {$index != -1} {
                        append newresid " -> [lindex $QWIKMD::protonate(${resid}_${chain}) 1]"
                    }
                } elseif {[lsearch -exact $QWIKMD::protindex "${resid}_$chain"] != -1} {
                    set newresid "[lindex $QWIKMD::protonate(${resid}_${chain}) 0] -> [lindex $QWIKMD::protonate(${resid}_${chain}) 1]"
                } elseif {[lsearch -exact $QWIKMD::delete "${resid}_$chain"] != -1} {
                    $table rowconfigure $i -background white -foreground grey -selectbackground cyan -selectforeground grey
                }

                if {$newresid != ""} {
                    $table cellconfigure $i,1 -text $newresid
                    $table cellconfigure $i,1 -background #ffe1bb
                }
                if {$type == "protein"} {
                    set selaux [atomselect top "resid \"$resid\" and chain \"$chain\" "]
                    set hexcols [QWIKMD::chooseColor [lindex [$selaux get structure] 0]]
                    
                    set hexred [lindex $hexcols 0]
                    set hexgreen [lindex $hexcols 1]
                    set hexblue [lindex $hexcols 2]
                    set QWIKMD::color($i) "#${hexred}${hexgreen}${hexblue}"
                    $selaux delete
                    $table cellconfigure $i,3 -background $QWIKMD::color($i) -selectbackground $QWIKMD::color($i)
                }
                if {$QWIKMD::prepared == 0} {
                    set addToRename 0
                    set renameDone 0
                    if {$type == "nucleic" || $type == "protein"} {
                        
                        if {$type == "nucleic"} {
                            set var $QWIKMD::nucleic
                            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                                set var [concat $var [lindex [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]] 1]]
                            }
                            set index [lsearch -exact $var $resname]
                            set index2 [lsearch -exact $QWIKMD::renameindex "${resid}_$chain"]
                            if {$index == -1 && $index2 != -1} {
                                $table cellconfigure $i,1 -text $QWIKMD::dorename([lindex $QWIKMD::renameindex $ind])
                                set renameDone 1
                            } elseif {$index == -1} {
                                set addToRename 1
                            }
                        }  elseif {$type == "protein"} {
                            set var $QWIKMD::reslist
                            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                                set var [concat $var [lindex [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]] 1]]
                            }
                            set ind [lsearch -exact $QWIKMD::renameindex "${resid}_$chain"]
                            if {[lsearch -exact $var $resname] == -1 && $ind == -1 && $resname != "HIS"} {
                                set addToRename 1
                            } elseif {[lsearch -exact $var $resname] == -1 & $resname != "HIS"} {
                                $table cellconfigure $i,1 -text $QWIKMD::dorename([lindex $QWIKMD::renameindex $ind])
                                set renameDone 1
                            }
                        }
                         
                        
                    } else {
                        $table cellconfigure $i,3 -background white -selectbackground cyan
                        if {$type == "hetero"} {
                            set var $QWIKMD::hetero
                            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                                set var [concat $var [lindex [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]] 1]]
                            } else {
                                set var $QWIKMD::hetero
                            }
                            set index [lsearch -exact $var $resname]
                            set index2 [lsearch -exact $QWIKMD::renameindex "${resid}_$chain"]

                            set var $QWIKMD::heteronames
                            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                                set var [concat $var [lindex [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]] 2]]
                            } 
                            if {$index == -1 && $index2 == -1} {
                                set addToRename 1
                            } else {
                                if {$index != -1} {
                                    $table cellconfigure $i,1 -text [lindex $var $index]
                                    set renameDone 1
                                } else {
                                    set index [lsearch -exact $QWIKMD::hetero $QWIKMD::dorename(${resid}_$chain)]
                                    $table cellconfigure $i,1 -text [lindex $var $index]
                                    set renameDone 1
                                }
                                
                            }
                        } elseif {$type == "glycan"} {
                            set var $QWIKMD::carb
                            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                                set var [concat $var [lindex [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]] 1]]
                            } else {
                                set var $QWIKMD::carb
                            }
                            set index [lsearch -exact $var $resname]
                            set index2 [lsearch -exact $QWIKMD::renameindex "${resid}_$chain"]
                            set var $QWIKMD::carbnames
                            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                                set var [concat $var [lindex [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]] 2]]
                            } 
                            if {$index == -1 && $index2 == -1} {
                                set addToRename 1
                            }  else {
                                if {$index != -1} {
                                    $table cellconfigure $i,1 -text [lindex $var $index]
                                    set renameDone 1
                                } else {
                                    set index [lsearch -exact $QWIKMD::carb $QWIKMD::dorename(${resid}_$chain)]
                                    $table cellconfigure $i,1 -text [lindex $var $index]
                                    set renameDone 1
                                }
                            }
                        } elseif {$type == "lipid"} {
                            set var $QWIKMD::lipidname
                            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                                set var [concat $var [lindex [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]] 1]]
                            }
                            set ind [lsearch -exact $QWIKMD::renameindex "${resid}_$chain"]
                            if {[lsearch -exact $var $resname] == -1 && $ind == -1 } {
                                set addToRename 1
                            } elseif {[lsearch -exact $var $resname] == -1} {
                                $table cellconfigure $i,1 -text $QWIKMD::dorename([lindex $QWIKMD::renameindex $ind])
                                set renameDone 1
                            }
                        
                        }
                    }
                    
                    if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1 && $renameDone == 0 && $addToRename == 0} {
                        set macro [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]]
                        set var [lindex $macro 1]
                        set index [lsearch -exact $var $resname]
                        set index2 [lsearch -exact $QWIKMD::renameindex "${resid}_$chain"]
                        if {$index == -1 && $index2 == -1} {
                            set addToRename 1
                        }  else {
                            if {$index != -1} {
                                $table cellconfigure $i,1 -text [lindex [lindex $macro 2] $index]
                            } else {
                                set index [lsearch -exact $var $QWIKMD::dorename(${resid}_$chain)]
                                $table cellconfigure $i,1 -text [lindex [lindex $macro 2] $index]
                            }
                            
                        }
                    }
                    if {$addToRename == 1 && [lsearch $QWIKMD::delete "${resid}_$chain"] == -1 && $resname != "HIS"} {
                        set listMolecules [list $QWIKMD::reslist $QWIKMD::hetero $QWIKMD::carb $QWIKMD::nucleic $QWIKMD::lipidname]
                        set found 0
                        set indexmacro -1
                        set incrList 0
                        set macro ""
                        foreach var $listMolecules {
                            set index [lsearch -exact $var $resname]
                            if {$index != -1} {
                                set indexlist $incrList
                                set found 1
                                switch $indexlist {
                                    0 {
                                        set macro protein   
                                    }
                                    1 {
                                        set macro hetero
                                    }
                                    2 {
                                        set macro glycan
                                    }
                                    3 {
                                        set macro nucleic
                                    }
                                    4 {
                                        set macro lipid
                                    }
                                }
                                break
                            }
                            incr incrList
                        }
                        if {$found == 0} {
                            foreach mcr $QWIKMD::userMacros {
                                if {[lsearch -exact [lindex $mcr 1] $resname] != -1 && $type != [lindex $mcr 0]} {
                                    set macro [lindex $mcr 0]
                                    set found 1
                                    break
                                }
                            }
                        }
                        if {$found == 1 && $type != $macro} {
                            set txt "and not \(resid \"$resid\" and chain \"$chain\"\)"
                            set txt2 "or \(resid \"$resid\" and chain \"$chain\"\)"
                            QWIKMD::editMacros $type $txt $txt2 old
                            QWIKMD::editMacros $macro $txt $txt2 new

                            set retype 1
                        } elseif {$found == 0 && $addToRename == 1} {
                            $table rowconfigure $i -background red -selectbackground cyan
                            lappend QWIKMD::rename "${resid}_$chain"
                            set name "$chain and $type"
                            if {$QWIKMD::index_cmb($name,2) != "Throb"} {
                                set QWIKMD::index_cmb($name,2) "Throb"
                                $maintable cellconfigure $QWIKMD::index_cmb($name,3),4 -text [QWIKMD::mainTableCombosEnd $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb $QWIKMD::index_cmb($name,3) 4 "Throb"]
                            }
                        }
                        
                    } 
                }
                incr i
            }
        }
        set insertedResidues [list]
        set listMol [list]
        $sel delete
        if {$retype == 1} {
            QWIKMD::UpdateMolTypes [expr [$QWIKMD::topGui.nbinput index current] +1]
        }
    }
    if {$tabid == 2 && [llength $QWIKMD::patchestr] > 0} {
        set i 1
        foreach patch $QWIKMD::patchestr {
            $QWIKMD::selresPatcheText insert $i.0 "$patch\n"
            incr i
        }
    }
}

proc QWIKMD::editAtom {} {
    set resTable $QWIKMD::selresTable
    set row [lindex [$resTable curselection] 0]
    $QWIKMD::atmsTable delete 0 end
    set prevres ""
    
    set resid [$resTable cellcget $row,0 -text]
    set resname [$resTable cellcget $row,1 -text]
    set chain [$resTable cellcget $row,2 -text]
    set type [$resTable cellcget $row,3 -text]

    set resnameaux [split $resname "->"]
    if {[llength $resnameaux] > 1} {
        if {[lindex $resname 0] == "HIS" && ([lindex $resname 2] == "HSD" || [lindex $resname 2] == "HSE" || [lindex $resname 2] == "HSP")} {
            set resname [lindex $resname 2]
        }
    } 
    
    switch $type {
        hetero {
            set index [lsearch $QWIKMD::heteronames $resname]
            if {$index != -1} {
                set resname [lindex $QWIKMD::hetero $index ]
            }
        }
        glycan {
            set index [lsearch $QWIKMD::carbnames $resname]
            if {$index != -1} {
                set resname [lindex $QWIKMD::carb $index ]
            }
        }
        default {
            set macroindex [lsearch -index 0 $QWIKMD::userMacros $type]
            if {$macroindex > -1 && $type != "protein" && $type != "nucleic"} {
                set typemacro [lindex $QWIKMD::userMacros $macroindex]
                set index [lsearch [lindex $typemacro 2] $resname]
                if {$index != -1} {
                    set resname [lindex [lindex $typemacro 1] $index ]
                }
            }
            
        }
    }

    set sel [atomselect $QWIKMD::topMol "resid \"$resid\" and chain \"$chain\""]
    
    set index 1
    set QWIKMD::atmsOrigNames [list]
    set QWIKMD::atmsOrigResid [list]
    foreach name [$sel get name] element [$sel get element] {
        set prevAtmChangeIndex [lsearch -all -index 1 $QWIKMD::atmsRename $name]
        lappend QWIKMD::atmsOrigNames $name
        lappend QWIKMD::atmsOrigResid $resid
        if {$prevAtmChangeIndex > -1} {
            set name [lindex [lindex $QWIKMD::atmsRename $prevAtmChangeIndex] 2]
        } 
        $QWIKMD::atmsTable insert end [list $index $resname $resid $chain $name $element $type]
        
        incr index
    }
    set QWIKMD::atmsNames [list]
    set str "No Topology Found"
    set isknown [lsearch -exact $QWIKMD::rename "${resid}_$chain"]
    if {($isknown == -1 || [lsearch -exact $QWIKMD::renameindex "${resid}_$chain"] != -1) || ($type != "protein" && $resname != "HIS")} {
        set toposearch [QWIKMD::checkAtomNames $resname $QWIKMD::TopList]

        set topofile [lindex $toposearch 0]
        foreach name [lindex $toposearch 1] {
            lappend QWIKMD::atmsNames [lindex $name 0]
        }

        set do 1
        if {$topofile != -1} {
            set opentopo [open $topofile r]
        
        
            set str ""
            
            while {[eof $opentopo] != 1 && $do == 1} {
                set line1 [gets $opentopo]
                set line [split $line1]
                    
                if {[lindex $line 0] == "RESI" && [lindex $line 1] ==  $resname} {
                
                        
                    while {[lindex $line 0] != "IC" && [eof $opentopo] != 1 && [lindex $line 0] != "END"} {
                        append str "$line1\n"
                        set line1 [gets $opentopo]
                        set line [split $line1]
                        if {[lindex $line 0] == "RESI"} {
                            break
                        }
                    }
                    set do 0
                    
                }
            }
            close $opentopo
        } 
        if {$do == 0} {
            set str "Topology File: [file tail $topofile]\n${str}"
            mol off $QWIKMD::topMol
            if {[lsearch [molinfo list] $QWIKMD::atmsMol] > -1} {
                mol delete $QWIKMD::atmsMol
            }
            set QWIKMD::atmsMol [::TopoTools::selections2mol "$sel"]
            mol rename $QWIKMD::atmsMol "Edit_${resname}_Atoms"
            mol modselect 0 $QWIKMD::atmsMol all
            mol modstyle 0 $QWIKMD::atmsMol "CPK 0.500000 0.100000 12.000000 12.000000"
            mol modcolor 0 $QWIKMD::atmsMol "Element"
            display resetview
            set selAll [atomselect $QWIKMD::atmsMol all ]
            set QWIKMD::atmsLables [list]
            set indNew 1
            foreach ind [$selAll get index] {
                draw color lime
                set selInd [atomselect $QWIKMD::atmsMol "index $ind"]
                lappend QWIKMD::atmsLables [draw text [vecadd {0.2 0.0 0.0} [join  [$selInd get {x y z}]] ] $indNew size 3]
                incr indNew
                $selInd delete
            }
            $selAll delete
        }
    }
    if {$type == "protein" && $resname == "HIS" } {
        tk_messageBox -message "Please assign a protonation state to histidine residues" -type ok -icon warning
    }
    $QWIKMD::atmsText configure -state normal
    $QWIKMD::atmsText delete 1.0 end
    $QWIKMD::atmsText insert 1.0 [format %s $str]
    
    $QWIKMD::atmsText configure -height 15 -font TkFixedFont 
    $QWIKMD::atmsText configure -state disable
}

proc QWIKMD::checkAtomNames {resname topologies} {

    set topolist [list]
    set prevres ""
    set topofile "-1"
    set charmmNames [list]
    set topoindex 0
    foreach topo $topologies {
        if {[file exists $topo]} {
            set topores [::Toporead::topology_get_resid [lindex $QWIKMD::topolist $topoindex] $resname]
            if {[lindex $topores 1] == $resname} {
                set topofile $topo
                set charmmNames [::Toporead::topology_get_resid [lindex $QWIKMD::topolist $topoindex] $resname atomlist]
                break
            }
            incr topoindex
        }
        
    }
    unset topologies
    return [list ${topofile} $charmmNames]
}

proc QWIKMD::atmStartEdit {tbl row col text} {
    set w [$tbl editwinpath]
    switch [$tbl columncget $col -name] {
        AtmdNAME {
            $w configure -values $QWIKMD::atmsNames -state normal -style protocol.TCombobox -takefocus 0 -exportselection false -justify center
            bind $w <<ComboboxSelected>> {
                $QWIKMD::atmsTable finishediting    
            }
            $w set $text
        }

    }
    return $text
}


proc QWIKMD::atmEndEdit {tbl row col text} {
    if {$text == ""} {
        return [$tbl cellcget $row,$col -text]
    }
    switch [$tbl columncget $col -name] {
        AtmdNAME {
            if {[lsearch $QWIKMD::atmsNames $text ] == -1} {
                tk_messageBox -message "Please choose one atom name from the list." -icon warning -type ok
                return [$tbl cellcget $row,$col -text]
            }
            set names [$tbl getcolumns 4]
            set index [lsearch $names $text]
            if {$index != -1} {
                set residfound [$tbl cellcget $index,2 -text]
                set resid [$tbl cellcget $row,2 -text]
                if {$residfound == $resid} {
                    tk_messageBox -message "Atoms names must be unique." -icon warning -type ok
                }
            }
        }
        ResID {
            set resid [expr int([format %4.0f $text])]
        }
    }
    return $text
}
proc QWIKMD::deleteAtoms {atmindex molID} {
    # set index [$QWIKMD::atmsTable curselection]
    # if { $index == -1 } {
    #   return
    # }
    # set atmindex [expr [$QWIKMD::atmsTable cellcget $index,0 -text] -1]
    set atmsel [atomselect $molID "index $atmindex"]
    set atmbonds [join [$atmsel getbonds]]
    $atmsel setbonds [list {}]
    $atmsel set name "QWIKMDDELETE"
    $atmsel set radius 0.0
    $atmsel set resid -9999
    foreach bonds $atmbonds {
        set selaux [atomselect $molID "index $bonds"]
        set bondslist [join [$selaux getbonds]]
        set delindex [lsearch $bondslist $atmindex]
        set newbonds [lreplace $bondslist $delindex $delindex]
        if {[llength $newbonds] == 0} {
            set newbonds {}
        }
        $selaux setbonds [list $newbonds]
        $selaux delete
    }
    $atmsel delete
}

proc QWIKMD::changeAtomNames {} {
    if {[lindex [$QWIKMD::atmsTable editinfo] 1] != -1 } {
        $QWIKMD::atmsTable finishediting
    }
    set currentValName [$QWIKMD::atmsTable getcolumns 4]
    set chain [lindex [$QWIKMD::atmsTable getcolumns 3] 0]
    set currentValResid [$QWIKMD::atmsTable getcolumns 2]
    set type [$QWIKMD::atmsTable cellcget 0,6 -text]
    set atmindex [$QWIKMD::atmsTable getcolumns 0]
    set resname [$QWIKMD::atmsTable cellcget 0,1 -text]
    foreach name $currentValName {
        if {[lsearch $QWIKMD::atmsNames $name] == -1} {
            tk_messageBox -message "Please assign the correct name to atom $name." -icon warning -type ok
            return
        }
    }

    ###Delete Atoms
    set PDBresname ""
    if {[llength $QWIKMD::atmsDeleteNames] > 0} {

        set delatmindex [lindex $QWIKMD::atmsDeleteNames 0]
        set selresid  [lindex $QWIKMD::atmsOrigResid $delatmindex ]
        set delatmname [lindex $QWIKMD::atmsOrigNames $delatmindex]
        
        set selresid [atomselect $QWIKMD::topMol "resid \"$selresid\" and chain \"$chain\" and name $delatmname"]
        set PDBresname [lsort -unique [$selresid get resname]]
        $selresid delete
        
        set selresid [atomselect $QWIKMD::topMol "resname $PDBresname and name $delatmname"]
        set delresidlist [lsort -unique [$selresid get resid]]
        $selresid delete
        set numresname [llength $delresidlist]
        set answer "yes"
        if {$numresname > 1} {
            set answer [tk_messageBox -message "The atoms signed to be deleted were found in more than one residue. Do you want to delete in all?" -icon warning -title "Delete Atoms" -type yesnocancel]
            if {$answer == "cancel"} {
                return
            } 
        }

        foreach delatmindex $QWIKMD::atmsDeleteNames {
            set selresid  [lindex $QWIKMD::atmsOrigResid $delatmindex ]
            set delatmname [lindex $QWIKMD::atmsOrigNames $delatmindex]
            set sel ""
            if {$answer == "yes"} {
                set sel [atomselect $QWIKMD::topMol "resname $PDBresname and name $delatmname"]
                foreach resid [$sel get resid] chain [$sel get chain] resname [$sel get resname] index [$sel get index] {
                    QWIKMD::deleteAtoms $index $QWIKMD::topMol
                    lappend QWIKMD::atmsDeleteLog [list $resid $resname $chain $delatmname] 
                }
                set QWIKMD::atmsOrigResid [lreplace $QWIKMD::atmsOrigResid $delatmindex $delatmindex]
                set QWIKMD::atmsOrigNames [lreplace $QWIKMD::atmsOrigNames $delatmindex $delatmindex]
            } else {
                set sel [atomselect $QWIKMD::topMol "resid \"$selresid\" and chain \"$chain\" and name $delatmname"]
                QWIKMD::deleteAtoms [$sel get index] $QWIKMD::topMol
                lappend QWIKMD::atmsDeleteLog [list $selresid $PDBresname $chain $delatmname] 
            }
            $sel delete
        }
        set QWIKMD::atmsDeleteNames [list]
    }
    ###Check if there is more than one residue with the same name to apply the same changes

    set originalname [lindex $QWIKMD::atmsOrigNames 0 ]
    set originalresid [lindex $QWIKMD::atmsOrigResid 0 ]
    if {$PDBresname == ""} {
        set selresid [atomselect $QWIKMD::topMol "resid \"$originalresid\" and chain \"$chain\" and name $originalname"]
        set PDBresname [lsort -unique [$selresid get resname]]
        $selresid delete
    }
    set numresname 1
    set answer "no"
    if {$QWIKMD::resallnametype == 1} {
        set selresid [atomselect $QWIKMD::topMol "resname $PDBresname and name $originalname"]
        set residlist [lsort -unique [$selresid get resid]]
        $selresid delete
        set numresname [llength $residlist]
        
        if {$numresname > 1} {
            set answer [tk_messageBox -message "More than one residue can be changed base on this operation. Do you want to apply to all?" -icon warning -title "Atom Names" -type yesnocancel]
            if {$answer == "cancel"} {
                return
            } elseif {$answer == "no"} {
                set numresname 1
            }
        }
    }
    set listindex 0
    set resIdChange 0
    set prevResId [lindex $QWIKMD::atmsOrigResid [expr [lindex $atmindex $listindex] -1] ]
    set prevresidList ""
    foreach name $currentValName {
        set atomindex [lindex $atmindex $listindex]
        set originalname [lindex $QWIKMD::atmsOrigNames [expr $atomindex -1] ]
        set originalresid [lindex $QWIKMD::atmsOrigResid [expr $atomindex -1] ]
        ###List containing the information "{ {Residue Names} {Residue Number} {Old Atom Name} {New Atom Name}}. "
        set replaceIndex [lsearch $QWIKMD::atmsRename "$resname $originalresid $originalname *" ]
        if {$replaceIndex == -1} {
            lappend QWIKMD::atmsRename [list $resname $originalresid $originalname [lindex $currentValName $listindex] [lindex $currentValResid $listindex] ]
        } else {
            lreplace $QWIKMD::atmsRename $replaceIndex $replaceIndex [list $resname $originalresid $originalname [lindex $currentValName $listindex] [lindex $currentValResid $listindex] ]
        }
        set sel ""
        if {$answer == "yes"} {
            set sel [atomselect $QWIKMD::topMol "resname $PDBresname and name $originalname"]
            foreach resid [$sel get resid] chain [$sel get chain] {
                if {$originalname != [lindex $currentValName $listindex]} {
                    lappend QWIKMD::atmsRenameLog [list $resid $resname $chain $originalname [lindex $currentValName $listindex]] 
                }
            }
        } else {
            set sel [atomselect $QWIKMD::topMol "resid \"$originalresid\" and chain \"$chain\" and name $originalname"]
            if {$originalname != [lindex $currentValName $listindex]} {
                lappend QWIKMD::atmsRenameLog [list $originalresid $resname $chain $originalname [lindex $currentValName $listindex]] 
            }

        }
        set nameRepeatList [list]
        if {$numresname > 1} {
            for {set i 0} {$i < [llength [lsort -unique [$sel get residue]]]} {incr i} {
                lappend nameRepeatList [lindex $currentValName $listindex]
            }
        } else {
            set nameRepeatList [lindex $currentValName $listindex]
        }
        $sel set name $nameRepeatList
        set currentResid [lindex $currentValResid $listindex]
        if {$currentResid!= $originalresid} {

            $sel set resid [lindex $currentValResid $listindex]
            ###Check if the residue was marked for renaming and change the type of the new created residues
            if {[info exists QWIKMD::dorename(${originalresid}_$chain)] == 1 && $prevresidList != $currentResid && $prevResId != $currentResid } {

                lappend QWIKMD::atmsReorderLog [list $originalresid $resname $chain $originalresid $currentResid]

                lappend QWIKMD::renameindex ${currentResid}_$chain
                set QWIKMD::dorename(${currentResid}_$chain) $QWIKMD::dorename(${originalresid}_$chain)
                set toresname ""
                set txt "and not \(resid \"${currentResid}\" and chain \"$chain\"\)"
                set txt2 "or \(resid \"${currentResid}\" and chain \"$chain\"\)"

                QWIKMD::checkMacros $type $txt $txt2

                if {$type == "hetero"} {
                    append QWIKMD::heteromcr " $txt2"
                } elseif {$type == "nucleic"} {
                    append QWIKMD::nucleicmcr " $txt2"
                } elseif {$type == "lipid"} {
                    append QWIKMD::lipidmcr " $txt2"
                } elseif {$type == "glycan"} {
                    append QWIKMD::glycanmcr " $txt2"
                } elseif {$type == "protein"} {
                    append QWIKMD::proteinmcr " $txt2"
                } elseif {[lsearch -index 0 $QWIKMD::userMacros $type] > -1} {
                    atomselect macro $type "[atomselect macro $type] $txt2"
                }

                set listOfMacros [concat protein nucleic glycan lipid hetero]
                foreach macroName $listOfMacros {
                    if {$macroName != $type} {
                        QWIKMD::checkMacros $macroName $txt $txt2
                        if {$macroName == "hetero"} {
                            append QWIKMD::heteromcr " $txt"
                        } elseif {$macroName == "nucleic"} {
                            append QWIKMD::nucleicmcr " $txt"
                        } elseif {$macroName == "lipid"} {
                            append QWIKMD::lipidmcr " $txt"
                        } elseif {$macroName == "glycan"} {
                            append QWIKMD::glycanmcr " $txt"
                        } elseif {$macroName == "protein"} {
                            append QWIKMD::proteinmcr " $txt"
                        }
                    }
                }
            }

            set resIdChange 1
            
        }
        set prevresidList [lindex $currentValResid $listindex]
        $sel delete
        incr listindex
    }
    QWIKMD::deleteAtomGuiProc
    if {$resIdChange == 1} {
        atomselect macro qwikmd_protein $QWIKMD::proteinmcr
        atomselect macro qwikmd_nucleic $QWIKMD::nucleicmcr
        atomselect macro qwikmd_glycan $QWIKMD::glycanmcr
        atomselect macro qwikmd_lipid $QWIKMD::lipidmcr
        atomselect macro qwikmd_hetero $QWIKMD::heteromcr
        QWIKMD::UpdateMolTypes $QWIKMD::tabprevmodf
    }
}

proc QWIKMD::cancelAtomNames {} {
    QWIKMD::deleteAtomGuiProc
}
proc QWIKMD::checkStructur { args } {
    set topoReport [list]
    set topo 0
    set resTable $QWIKMD::selresTable
    if {[llength $QWIKMD::rename] > 0} {
        
        foreach res $QWIKMD::rename {
            set residchain [split $res "_" ]
            set resid [lindex $residchain 0]
            set chain [lindex $residchain end]
            if {[lsearch -exact $QWIKMD::renameindex $res] == -1 \
                && [lsearch -exact $QWIKMD::delete $res] == -1 && [$resTable searchcolumn 2 $chain -check [list QWIKMD::tableSearchCheck $resid] ] > -1 } {
                set str [split $res "_"]
                lappend topoReport "Unknown tolology for Residue $resid of the Chain $chain\n"
                incr topo
            }
        }
        
    } else {
        lappend topoReport "No topologies issues found\n"
    }
    set length [expr [llength [array names QWIKMD::chains]] /3]
    set txt ""
    if {$args != "init"} {
        for {set i 0} {$i < $length} {incr i} {
            if {$QWIKMD::chains($i,0) == 1 && ([regexp "protein" $QWIKMD::chains($i,1)] || [regexp "nucleic" $QWIKMD::chains($i,1)] || [regexp "glycan" $QWIKMD::chains($i,1)]) } {
                append txt " ([lindex $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)]) or" 
            }
        }
        set txt [string trimleft $txt " "]
        set txt [string trimright $txt " or"]
        if {$txt != ""} {
            set globalsel [atomselect $QWIKMD::topMol $txt]
            set residues [lsort -unique [$globalsel get residue]]
            $globalsel delete
            set txt "residue [join $residues]"
        }
    } else {
        set txt "not water and not ions and not ($QWIKMD::heteromcr)"
    }
    atomselect macro qwikmd_protein $QWIKMD::proteinmcr
    atomselect macro qwikmd_nucleic $QWIKMD::nucleicmcr
    atomselect macro qwikmd_glycan $QWIKMD::glycanmcr
    atomselect macro qwikmd_lipid $QWIKMD::lipidmcr
    atomselect macro qwikmd_hetero $QWIKMD::heteromcr

    set atomselectList [atomselect list]
    set generalchecks [strctcheck -mol $QWIKMD::topMol -selText $txt -qwikmd 1 ]
    set atomselectList2 [atomselect list]
    foreach select $atomselectList2 {
        if {[lsearch $atomselectList $select] == -1} {
            catch {$select delete}
        }
    }
    set QWIKMD::topoerror [list $topo $topoReport]
    if {$topo == 0} {
        set tabid [expr [$QWIKMD::topGui.nbinput index current] + 1 ]
        set maintable "$QWIKMD::topGui.nbinput.f$tabid.tableframe.tb"
        set colColor [$maintable getcolumn 4]
        set index [lsearch -all $colColor "Throb"]
        foreach row $index {
            set name "[$maintable cellcget $row,0 -text] and [$maintable cellcget $row,2 -text]"
            unset QWIKMD::index_cmb($name,2) 
            $maintable cellconfigure $QWIKMD::index_cmb($name,3),4 -text "aux"
            $maintable editcell $QWIKMD::index_cmb($name,3),4
            $maintable finishediting
        }
        
    }
    foreach check $generalchecks {
        switch [lindex $check 0] {
            chiralityerrors {
                set QWIKMD::chirerror [lindex $check 1]
            }
            cispeperrors {
                set QWIKMD::cisperror [lindex $check 1]
            }
            gaps {
                set QWIKMD::gaps [lindex $check 1]
            }
            torsionOutlier {
                set QWIKMD::torsionOutlier [lindex $check 1]
                set QWIKMD::torsionTotalResidue [lindex $check 2]
            }
            torsionMarginal {
                set QWIKMD::torsionMarginal [lindex $check 1]
            }

        }
    }
    QWIKMD::checkUpdate 
    
}

proc QWIKMD::checkUpdate {} {
    set do 0
    set QWIKMD::warnresid 0
    set color "green"
    set text "Topologies & Parameters"
    set colortext "black"
    if {[lindex $QWIKMD::topoerror 0] > 0} {
        set color "red"
        append text " \([lindex $QWIKMD::topoerror 0]\)"
        set colortext "blue"
        set do 1

    } 
    $QWIKMD::topolabel configure -text $text -foreground $colortext -cursor hand1
    $QWIKMD::topocolor configure -background $color

    set color "green"
    set text "Chiral Centers"
    set colortext "black"
    set number [lindex $QWIKMD::chirerror 0]
    if {$number == ""} {
        set color "yellow"
        append text " \(Failed\)"
        set colortext "blue"
        set do 1
    } elseif {$number > 0} {
        set color "yellow"
        append text " \($number\)"
        set colortext "blue"
        set do 1
        
    }

    $QWIKMD::chirlabel configure -text $text -foreground $colortext -cursor hand1
    $QWIKMD::chircolor configure -background $color  

    set color "green"
    set text "Cispetide Bond"
    set colortext "black"
    set number [lindex $QWIKMD::cisperror 0]
    if {$number == ""} {
        set color "yellow"
        append text " \(Failed\)"
        set colortext "blue"
        set do 1
    } elseif {$number > 0} {
        set color "yellow"
        append text " \($number\)"
        set colortext "blue"
        set do 1
        
    }
    $QWIKMD::cisplabel configure -text $text -foreground $colortext -cursor hand1
    $QWIKMD::cispcolor configure -background $color  

    set color "green"
    set text "Sequence Gaps"
    set colortext "black"

    if {[llength $QWIKMD::gaps] > 0} {
        set color "red"
        append text " \([llength $QWIKMD::gaps]\)"
        set colortext "blue"
        set do 1
    }
    $QWIKMD::gapslabel configure -text $text -foreground $colortext -cursor hand1
    $QWIKMD::gapscolor configure -background $color 


    ####TO-DO add label color and text for torsion outliers and marginals
    #### Tristan message :
    # This is a fairly standard scheme amongst experimental structural biologists - try running a structure of your choice past the server at http://molprobity.biochem.duke.edu/, for example.
    #  The typical rule of thumb is that a good structure should have 95% of residues in the preferred region, 5% in the allowed/marginal region, and <0.1% outliers.
    #   The actual stats the contours are based on come from a set of very high resolution crystal structures, and the cut-offs are 0.05% 2% for outliers and marginals respectively.
    #  Even if you have no outliers, going much past 5-10% marginals is a pretty good sign your structure still has some issues.
    set color "green"
    set text "Torsion Angles Outliers"
    set colortext "black"
    set numoutlier 0
    set torsionplotFail 0
    if {$QWIKMD::torsionOutlier == "Failed"} {
        set torsionplotFail 1
    }
    foreach outlier $QWIKMD::torsionOutlier {
        if {[llength [lrange $outlier 1 end]] > 0 } {
            incr numoutlier [llength [lrange $outlier 1 end]]
        }
    }
    set perc 0.0
    if {$numoutlier > 0 && $QWIKMD::torsionTotalResidue > 0} {
        set perc [format %0.2f [expr [expr $numoutlier / [expr $QWIKMD::torsionTotalResidue * 1.0]] *100]]
        set colortext "blue"
    }
    if {$torsionplotFail == 1} {
        set text "TorsionPlot Outliers check\n Failed!"
    } else {
        set text "Torsion Angles Outliers\n $perc\% \(Goal < 0.1\%\)"
    }
    #set do 1
    if {$torsionplotFail == 1} {
        set color "red"
    } elseif {$perc > 0} {
        set color "yellow"
        if {$perc >= 5} {
            set color "red"
            set do 1
        }
    }
    $QWIKMD::torsionOutliearlabel configure -text $text -foreground $colortext -cursor hand1
    $QWIKMD::torsionOutliearcolor configure -background $color 


    set color "green"
    set text "Torsion Angles Marginals"
    set colortext "black"
    set nummarginal 0
    
    foreach marginal $QWIKMD::torsionMarginal {
        if {[llength [lrange $marginal 1 end]] > 0 } {
            incr nummarginal [llength [lrange $marginal 1 end]] 
        }
    }
    set perc 0.0
    if {$nummarginal > 0 && $QWIKMD::torsionTotalResidue > 0} {
        set perc [format %0.2f [expr [expr $nummarginal / [expr $QWIKMD::torsionTotalResidue * 1.0] ] *100]]
        set colortext "blue"
    }
    if {$torsionplotFail == 1} {
        set text "TorsionPlot Marginals check\n Failed!"
    } else {
        set text "Torsion Angles Marginals\n $perc\% \(Goal < 5\%\)"
    }
    #set do 1
    if {$torsionplotFail == 1} {
        set color "red"
    } elseif {$perc > 0} {
        set color "yellow"
        if {$perc >= 10} {
            set color "red"
            set do 1
        }
    }

    $QWIKMD::torsionMarginallabel configure -text $text -foreground $colortext -cursor hand1
    $QWIKMD::torsionMarginalcolor configure -background $color
    
    if {$do == 1} {
        tk_messageBox -message "One or more warnings were generated during structure check routines.\nPlease refer to the \"Structure Manipulation/Check\" window to fix them" -title "Structure Check" -icon warning -type ok
        set QWIKMD::warnresid 1
    }
}

proc QWIKMD::changecomboType {w} {
    set table $QWIKMD::topoPARAMGUI.f1.tableframe.tb
    set combo $w
    set type [$combo get]
    set row ""

    if {$type == "other..."} {
        
        set frame [string trimright $w ".r"]
        destroy $frame
        set type "other..."
        set names [$table columncget 1 -text]

        for {set i 0} {$i < [llength $names]} {incr i} {
            set charmmres [split [lindex $names $i] "->"]
            set charmmres [string trimright [lindex $charmmres 0]]
            if {$QWIKMD::topocombolist($charmmres) == $w} {
                set row $i
                break
            }
        }
        
        if {[llength $QWIKMD::topparmTable] == 0} {
            $QWIKMD::topoPARAMGUI.f1.tableframe.tb selection set $row
            lappend QWIKMD::topparmTable $row
        }
        
        $QWIKMD::topoPARAMGUI.f1.tableframe.tb selection set $QWIKMD::topparmTable
        $QWIKMD::topoPARAMGUI.f1.tableframe.tb cellconfigure $row,2 -editable true
        $QWIKMD::topoPARAMGUI.f1.tableframe.tb editcell $row,2
    } else {
        set names [$table columncget 1 -text]
        set row [lindex [$table editinfo] 1]
        if {$row == -1} {
            for {set i 0} {$i < [llength $names]} {incr i} {
                set charmmres [split [lindex $names $i] "->"]
                set charmmres [string trimright [lindex $charmmres 0]]
                if {$QWIKMD::topocombolist($charmmres) == $w} {
                    set row $i
                    break
                }
            }
        }
            
        if {[llength $QWIKMD::topparmTable] == 0} {
            $QWIKMD::topoPARAMGUI.f1.tableframe.tb selection set $row
            lappend QWIKMD::topparmTable $row
        }
        if {[string length $type] > 10 || [llength $type] > 1 || $type == ""} {
            tk_messageBox -message "Residue type must be at maximum 10 characters long.\nPlease make sure that spaces characters are not included. Selected residues will\
            set as hetero" -title "Residue Type" -icon info -type ok
            set QWIKMD::topparmTableError 1
            set type "hetero"
        }  
        set answer "yes"
        if {$type == "protein" && [$table cellcget $row,0 -text] != [$table cellcget $row,1 -text]} {
            set answer [tk_messageBox -message "Protein residues must have the same denomination as the CHARMM residues names.\n\
            Do you want to continue and change the Residues Denomination?" -title "Residue Type" -icon info -type yesno]
            update idletasks 
        }
        for {set i 0} {$i < [llength $QWIKMD::topparmTable]} {incr i} {
            set charmmres [split [lindex $names [lindex $QWIKMD::topparmTable $i]] "->"]
            set charmmres [string trimright [lindex $charmmres 0]]
            if {[winfo exists $QWIKMD::topocombolist($charmmres)] == 0} {
                $table cellconfigure [lindex $QWIKMD::topparmTable $i],2 -window QWIKMD::editType
            }

            $QWIKMD::topocombolist($charmmres) set $type
            $QWIKMD::topocombolist($charmmres) selection clear
            if {$answer == "yes" && $type == "protein"} {
                $table cellconfigure [lindex $QWIKMD::topparmTable $i],0 -text [$table cellcget [lindex $QWIKMD::topparmTable $i],1 -text]
                
            } elseif {$answer == "no"} {
                $QWIKMD::topocombolist($charmmres) set "hetero"
                return
            }
        }
        
        set QWIKMD::topparmTable [list]
        $QWIKMD::topoPARAMGUI.f1.tableframe.tb selection clear 0 end
    }
}
proc QWIKMD::editType {tbl row col w} {
    set type [$tbl cellcget $row,$col -text]
    set values {protein nucleic glycan lipid hetero other...}
    set defVal {protein nucleic glycan lipid hetero}
    if {$QWIKMD::userMacros != ""} {
        set i 0
        foreach aux $QWIKMD::userMacros {
            if {[lsearch $defVal [lindex $aux 0]] ==-1} {
                set values "[lrange $values 0 [expr [llength $values] -2]] [lindex $aux 0] [lindex $values end]"
            }
            incr i
        } 
    }
    grid [ttk::frame $w] -sticky news
    
    ttk::style map resid.TCombobox -fieldbackground [list readonly #ffffff]
    grid [ttk::combobox $w.r -state readonly -style resid.TCombobox -values $values -width 11 -justify center -postcommand {set QWIKMD::topparmTable [$QWIKMD::topoPARAMGUI.f1.tableframe.tb curselection]}] -row 0 -column 0
    
    set txt [$tbl cellcget $row,3 -text]
    set index 0
    set typeindex [lsearch $values $type]
    if {$type > -1 && $type != "type?"} {
        set index $typeindex
    } elseif {[string match "*prot*" $txt] > 0} {
        set index 0
    } elseif {[string match "*na*" $txt] > 0} {
        set index 1
    } elseif {[string match "*carb*" $txt] > 0} {
        set index 2
    } else {
        set index 3
    } 
   
    $w.r set [lindex $values $index]
    $tbl cellconfigure $row,$col -text ""
    $tbl configure -labelcommand ""
    bind $w.r <<ComboboxSelected>> {
        QWIKMD::changecomboType %W
    }
    set QWIKMD::topocombolist([$tbl cellcget $row,1 -text]) $w.r
}

proc QWIKMD::reviewTopPar {} {
    global env

    set defVal {protein nucleic glycan lipid hetero}
    if {[llength $QWIKMD::userMacros] >0} {
        foreach macro $QWIKMD::userMacros {
            if {[lsearch $defVal [lindex $macro 0]] == -1} {
                atomselect delmacro [lindex $macro 0]
            }
        }
    }
    set QWIKMD::userMacros ""
    if {[file exists ${env(QWIKMDFOLDER)}/toppar/toppartable.txt]} {
        set toppartable [open ${env(QWIKMDFOLDER)}/toppar/toppartable.txt r]
        set temp [read -nonewline $toppartable ]
        set temp [split $temp "\n"]
        
        foreach line $temp {
            set fileentry [file join ${env(QWIKMDFOLDER)}/toppar [lindex $line 3]]
            if {[lsearch $QWIKMD::TopList ${fileentry}] == -1} {
                lappend QWIKMD::TopList ${fileentry}
            }
            if {[lsearch $QWIKMD::ParameterList ${fileentry}] == -1 && ([file extension ${fileentry}] == ".str" || [file extension ${fileentry}] == ".prm")} {
                lappend QWIKMD::ParameterList ${fileentry}
            }
            set typeindex [lsearch $defVal [lindex $line 2]]
            set macroindex [lsearch -index 0 $QWIKMD::userMacros [lindex $line 2]]
            if { $macroindex == -1} {
                set txt [list [lindex $line 2] [list [lindex $line 1]] [list [lindex $line 0]] [list [lindex $line 3]]]
                lappend QWIKMD::userMacros $txt
            } elseif {$macroindex != -1} {
                set aux [lindex $QWIKMD::userMacros $macroindex]
                set aux [list [lindex $aux 0] [concat [lindex $aux 1] [lindex $line 1]] [concat [lindex $aux 2] [lindex $line 0]] [concat [lindex $aux 3] [lindex $line 3]]]
                lset QWIKMD::userMacros $macroindex $aux
            }
        }
        
        foreach macro $QWIKMD::userMacros {
            set textaux ""
            set do 0
            switch [lindex $macro 0] {
                protein {
                    set QWIKMD::proteinmcr [string trimright [string trimleft $QWIKMD::proteinmcr "("] ")"]
                    foreach resname [lindex $macro 1] {
                        if {[lsearch $QWIKMD::proteinmcr $resname] == -1} {
                            if {$do == 0} {
                                set textaux "$QWIKMD::proteinmcr"
                                set do 1
                            }
                            append textaux " or (resname $resname)"
                        }   
                    }
                }
                glycan {
                    set QWIKMD::glycanmcr [string trimright [string trimleft $QWIKMD::glycanmcr "("] ")"]
                    foreach resname [lindex $macro 1] {
                        if {[lsearch $QWIKMD::glycanmcr $resname] ==-1} {
                            if {$do == 0} {
                                append textaux "$QWIKMD::glycanmcr"
                                set do 1
                            }
                            append textaux " or (resname $resname)"
                        }   
                    }
                }
                lipid {
                    set QWIKMD::lipidmcr [string trimright [string trimleft $QWIKMD::lipidmcr "("] ")"]
                    foreach resname [lindex $macro 1] {
                        if {[lsearch $QWIKMD::lipidmcr $resname] ==-1} {
                            if {$do == 0} {
                                append textaux "$QWIKMD::lipidmcr"
                                set do 1
                            }
                            append textaux " or (resname $resname)"
                        }   
                    }
                }
                nucleic {
                    set QWIKMD::nucleicmcr [string trimright [string trimleft $QWIKMD::nucleicmcr "("] ")"]
                    foreach resname [lindex $macro 1] {
                        if {[lsearch $QWIKMD::nucleicmcr $resname] ==-1} {
                            if {$do == 0} {
                                append textaux "$QWIKMD::nucleicmcr"
                                set do 1
                            }
                            append textaux " or (resname $resname)"
                        }   
                    }
                }
                hetero {
                    set QWIKMD::heteromcr [string trimright [string trimleft $QWIKMD::heteromcr "("] ")"]
                    foreach resname [lindex $macro 1] {
                        if {[lsearch $QWIKMD::heteromcr $resname] == -1} {
                            if {$do == 0} {
                                append textaux "$QWIKMD::heteromcr"
                                set do 1
                            }
                            append textaux " or (resname $resname)"
                        }   
                    }
                }
                default {
                    atomselect macro [lindex $macro 0] "(resname [lindex $macro 1])"
                }

            }
            switch [lindex $macro 0] {
                protein {
                    set QWIKMD::proteinmcr "\($textaux\)"
                }
                glycan {
                    set QWIKMD::glycanmcr "\($textaux\)"
                }
                lipid {
                    set QWIKMD::lipidmcr "\($textaux\)"
                }
                nucleic {
                    set QWIKMD::nucleicmcr "\($textaux\)"
                }
                hetero {
                    set QWIKMD::heteromcr "\($textaux\)"
                }
            }
        }
        close $toppartable
    }

}

proc QWIKMD::deleteTopParm {} {
    global env
    set rowlist [$QWIKMD::topoPARAMGUI.f1.tableframe.tb curselection]
    set filenamelist [list] 
    foreach row $rowlist {
        set newfile [$QWIKMD::topoPARAMGUI.f1.tableframe.tb cellcget $row,3 -text]
        if {[lsearch $filenamelist $newfile] == -1} {
            lappend filenamelist $newfile
        }
    }
    $QWIKMD::topoPARAMGUI.f1.tableframe.tb delete $rowlist

    set currentfilename [lsort -unique [$QWIKMD::topoPARAMGUI.f1.tableframe.tb getcolumns 3]]
    foreach filename $filenamelist {
        if {[lsearch $currentfilename $filename] == -1} {
            if {[file exists ${env(QWIKMDFOLDER)}/toppar/$filename] == 1} {
                file delete -force ${env(QWIKMDFOLDER)}/toppar/$filename
            }
        }
    }
    set QWIKMD::charmmprev [$QWIKMD::topoPARAMGUI.f1.tableframe.tb getcolumns 1]
}
     
proc QWIKMD::starteditType {tbl row col txt} {
    if {$col == 1} {
        set entry [split $txt "->"]     
        set txt [string trimright [lindex $entry 0]]
    }
    return $txt
}
proc QWIKMD::editResNameType {tbl row col txt} {
    if {$col == 0 || $col ==1} {
        set resname [$tbl columncget 0 -text]
        if {$col == 1} {
            set resname [list]
            foreach res [$tbl columncget 1 -text] {
                lappend resname [lindex [split $res "->"] end]      
            }
        }
        
        if {$txt == ""} {
            set txt [lindex [split [$tbl cellcget $row,1 -text] "->"] end]
            if {$col == 1} {
                set txt [$tbl cellcget $row,0 -text]
            }
        } else {
            set do 1
            if {[lsearch -all $QWIKMD::carb $txt] > 0 || [lsearch -all $QWIKMD::carbnames $txt] > 0} {
                set do 0
            } elseif {[lsearch -all $QWIKMD::hetero $txt] >= 0  || [lsearch -all $QWIKMD::heteronames $txt] > 0} {
                set do 0
            } elseif {[lsearch -all $QWIKMD::nucleic $txt] >= 0} {
                set do 0
            } elseif {[lsearch -all $QWIKMD::lipidname $txt] >= 0} {
                set do 0
            } elseif {[lsearch -all $QWIKMD::reslist $txt] >= 0} {
                set do 0
            } elseif {[lsearch -all $resname $txt] >= 0 && [lsearch -all $resname $txt] != $row} {
                set do 0
            }
            if {$do == 0} {
                set title "Residue Name"
                if {$col == 1} {
                    set title "CHARMM Name"
                }
                tk_messageBox -message "The name \"$txt\" is already in use.\n Please make sure that the chosen name is unique." -title $title  -icon info -type ok
                set QWIKMD::topparmTableError 1
                set txt [lindex [split [$tbl cellcget $row,1 -text] "->"] end]
                if {$col == 1} {
                    set txt [$tbl cellcget $row,0 -text]
                }
                $tbl selection clear 0 end
            } else {
                set max 10
                set str $txt
                if {$col == 1} {
                    set max 4
                    set str [split $txt "->"]
                    if {[llength $str] > 1} {
                        set str [string trimright [lindex $str 1]]
                    }

                }
                if {( [string length $str] > $max || [llength $str] > 1)} {
                    if {[llength $str] > 1} {
                        set txt [lindex $str 0]
                    }
                    if {[string length $str] > $max} {
                        set txt [string range $str 0 [expr $max -1]]
                    } else {
                        set txt [string range $str 0 end]
                    }
                    tk_messageBox -message "Residue denomination must be at maximum $max characters long.\n Please make sure that spaces and/or special characters are not included." -title "Residue Name" -icon info -type ok
                    set QWIKMD::topparmTableError 1
                    if {$col == 1} {
                        $tbl cellconfigure $row,1 -background red
                    }

                }
            } 

            if {$do ==1 && $col == 0} {
                set charmmres [split [$tbl cellcget $row,1 -text] "->"]
                set charmmres [string trimright [lindex $charmmres 0]]
                if {([$QWIKMD::topocombolist($charmmres) get] == "protein" || [$QWIKMD::topocombolist($charmmres) get] == "nucleic" || [$QWIKMD::topocombolist($charmmres) get] == "lipid") &&  $txt != [$tbl cellcget $row,1 -text]} {
                    tk_messageBox -message "[$QWIKMD::topocombolist($charmmres) get] residues must have the same name as the CHARMM residues names.\nPlease change residues denomination" -title "Residue Type" -icon info -type ok
                    set QWIKMD::topparmTableError 1
                    update idletasks
                    set txt [$QWIKMD::topoPARAMGUI.f1.tableframe.tb cellcget $row,1 -text]
                    update idletasks
                    return $txt
                }
            } elseif {$do ==1 && $col == 1} {
                set prev [$tbl cellcget $row,$col -text]
                set prev [split $prev "->"]
                set charmmres [split [$tbl cellcget $row,1 -text] "->"]
                set charmmres [string trimright [lindex $charmmres 0]]

                if {$txt != [string trimright [lindex $prev 0]]} {
                    
                    $tbl cellconfigure $row,1 -background white
                    set txt "[string trimright [lindex $prev 0]] -> $txt"
                    
                }
                if {([$QWIKMD::topocombolist($charmmres) get] == "protein" || [$QWIKMD::topocombolist($charmmres) get] == "nucleic" || [$QWIKMD::topocombolist($charmmres) get] == "lipid") &&  $txt != [$tbl cellcget $row,1 -text]} {
                    tk_messageBox -message "[$QWIKMD::topocombolist($charmmres) get] residues must have the same name as the CHARMM residues names.\nPlease change residues denomination" -title "Residue Type" -icon info -type ok
                    set QWIKMD::topparmTableError 1
                    $tbl cellconfigure $row,0 -text [lindex [split $txt "->"] end] 
                }
                return $txt
            }
        }
        $tbl selection clear 0 end
        return $txt
    }  elseif {$col == 2} {
        QWIKMD::changecomboType [$tbl editwinpath]
    }
    
}

proc QWIKMD::addTopParm {} {
    global env
    set types {
        {{topology + stream}       {".rtf" ".str"}        }
        {{All}       {"*"}        }
    }
    set fil [tk_getOpenFile -title "Open Topology & Parameters File" -initialdir $env(CHARMMTOPDIR) -filetypes $types]
    if {$fil != ""} {
        set table $QWIKMD::topoPARAMGUI.f1.tableframe.tb
        set infile [open ${fil} r]
        set all [regsub {"} [read $infile] ""]
        set val [split $all "\n"]
        set domessage 0
        for {set i 0} {$i < [llength $val]} {incr i} {
            
            if {[string range [lindex $val $i] 0 0] != "!"} {
                
                if { [lindex [lindex $val $i] 0] == "RESI"} {
                    set do 1
                    set res [lindex [lindex $val $i] 1]
                    if {[lsearch -all $QWIKMD::carb $res] > 0} {
                        set do 0
                    } elseif {[lsearch -all $QWIKMD::hetero $res] >= 0} {
                        set do 0
                    } elseif {[lsearch -all $QWIKMD::nucleic $res] >= 0} {
                        set do 0
                    } elseif {[lsearch -all $QWIKMD::reslist $res] >= 0} {
                        set do 0
                    } elseif {[lsearch -all [$table columncget 1 -text] $res] >=0} {
                        set do 0
                    }

                    if {$do == 1} {
                        $table insert end "$res $res type? {}"
                        $table cellconfigure end,3 -text ${fil}
                        $table cellconfigure end,2 -window QWIKMD::editType
                        if {[info exists QWIKMD::charmmchange([file tail ${fil}])] != 1} {
                            set QWIKMD::charmmchange([file tail ${fil}],0) [list]
                        }
                        lappend QWIKMD::charmmchange([file tail ${fil}],0) $res
                    }
                    if {[string length $res] > 4} {
                        $table cellconfigure end,1 -background red
                        $table cellconfigure end,1 -editable true -editwindow ttk::entry 
                        $table configure -editstartcommand QWIKMD::starteditType
                        set domessage 1
                    }
                }
            }
        }
        if {$domessage == 1} {
            tk_messageBox -message "Some residue's name contain more than 4 characters (in red).\nPlease choose an unique 4 character residue name." -title "Residue Name" -icon info -type ok
        }
        close $infile
    }
}

proc QWIKMD::applyTopParm {} {
    global env
    set table $QWIKMD::topoPARAMGUI.f1.tableframe.tb

    #check if table has any element
    if {[$table size] != 0} {
        # check if the tablelist has an editcell operation in place
        # the command pathname editinfo return {{} -1 -1} if no cell is being edit 
        if {[lindex [$table editinfo] 1] != -1 } {
            $table finishediting
        }

        if {$QWIKMD::topparmTableError == 1} {
            tk_messageBox -message "An error was generated when applying modifications. Please revise and apply again." -icon error -type ok
            set QWIKMD::topparmTableError 0
            return
        }

        set answer [tk_messageBox -message "The topologies and parameters of the molecules listed in the table will be added to \n QwikMD library. Do you want to continue?" -title "Topologies and Parameters" -icon warning -type yesno]

        if {$answer == "no"} {
            return
        }
    }

    set resname [$table columncget 0 -text]
    set charmmres [$table columncget 1 -text]
    set tbfile [$table columncget 3 -text]
    set indexes ""
    set i 0
    foreach res $charmmres {
        set prev [split $res "->"]
        if {[llength $prev] == 1 && [string length $prev] > 4} {
            lappend indexes $res
            $table cellconfigure $i,1 -background red
        }
        incr i
    }
    
    if {$indexes != ""} {
        tk_messageBox -message "Please change the CHARMM residue names of $indexes." -title "Residue Name" -icon info -type ok 
        return
    }
    
    set toppartable [open ${env(QWIKMDFOLDER)}/toppar/toppartable.txt w+]
    set i 0

    set prevfile [lindex $tbfile 0]
    set charoriginal [list]
    set charreplace [list]
    foreach res $resname chares ${charmmres} srcfile $tbfile {
        set filename ""
        if {[string first "_qwikmd" [file root [file tail ${srcfile}]]] == -1} {
            set filename  [file root [file tail ${srcfile}]]_qwikmd[file extension ${srcfile}]
        } else {
            set filename  [file root [file tail ${srcfile}]][file extension ${srcfile}]
        }
        
        set prev [split $chares "->"]
        set reoriginal $chares
        set curreplace $chares
        if {[llength $prev] > 1} {
            lappend charoriginal [string trimright [lindex $prev 0]]
            lappend charreplace [string trimleft [lindex $prev end]]
            set reoriginal [lindex $charoriginal end]
            set curreplace [lindex $charreplace end]
        }
        
        if {[file dirname ${srcfile}] != "."} {
            set file ${env(QWIKMDFOLDER)}/toppar/${filename}
            set fileincr 1
            set filenameaux $filename

            while {[file exists $file] == 1} {
                set filenameaux [file root ${filename}]_${fileincr}[file extension ${filename}]
                set file ${env(QWIKMDFOLDER)}/toppar/${filenameaux}
                incr fileincr
            }
            set filename $filenameaux
            puts $toppartable "$res\t$curreplace\t[$QWIKMD::topocombolist($reoriginal) get]\t${filename}"
            incr i
            if {$srcfile != $prevfile || $i == [$table size] } {
                set f [open ${srcfile} "r"]
                set txt [read -nonewline ${f}]
                set txt [split $txt "\n"]
                if {[llength $charoriginal] > 0} {
                    set enter ""
                    set index [lsearch -exact -all $txt $enter]
                    for {set j 0} {$j < [llength $index]} {incr j} {
                        lset txt [lindex $index $j] "{} {}"
                    }
                }           
                set out [open $file w+ ]
                foreach original $charoriginal replace $charreplace {
                    set resi "RESI"
                    set index [lsearch -regexp -all $txt (?i)^$resi]
                    
                    foreach ind $index {
                        if {[lindex [lindex $txt $ind] 1 ] == $original} {
                            set strreplace [lindex $txt $ind]
                            lset strreplace 1 $replace
                            lset txt $ind [join $strreplace]
                        }
                    }
                }
                
                if {[llength $charoriginal] > 0} {
                    set enter "{} {}"
                    set index [lsearch -exact -all $txt $enter]
                    for {set j 0} {$j < [llength $index]} {incr j} {
                        lset txt [lindex $index $j] " "
                    }
                
                }
                for {set j 0 } {$j < [llength $txt]} {incr j} {
                    puts $out [lindex $txt $j]
                }
                close $f
                close $out
                set charoriginal [list]
                set charreplace [list]
            }
        } else {
            puts $toppartable "$res\t$curreplace\t[$QWIKMD::topocombolist($reoriginal) get]\t${filename}"
            incr i
            set prevfile [lindex $tbfile $i]
        }
    }
    close $toppartable
    QWIKMD::reviewTopPar
    QWIKMD::loadTopologies
    QWIKMD::addTableTopParm
}

proc QWIKMD::addTableTopParm {} {
    global env
    if {$QWIKMD::userMacros != ""} {
        set table $QWIKMD::topoPARAMGUI.f1.tableframe.tb
        set toppartable [open ${env(QWIKMDFOLDER)}/toppar/toppartable.txt r]
        $table delete 0 end
        foreach macro $QWIKMD::userMacros {
            foreach res [lindex $macro 2] charres [lindex $macro 1] file [lindex $macro 3] {
                $table insert end [list $res $charres [lindex $macro 0] $file]
                $table cellconfigure end,0 -editable true
                $table cellconfigure end,0 -editwindow ttk::entry
                $table cellconfigure end,2 -window QWIKMD::editType
            }
            
        }
        close $toppartable
    }
}

proc QWIKMD::PrepareStructures {prefix textLogfile} {
    cd $QWIKMD::outPath/setup/
    set structure $QWIKMD::topMol

    set length [expr [llength [array names QWIKMD::chains]] /3]
    set txt ""
    for {set i 0} {$i < $length} {incr i} {
        if {$QWIKMD::chains($i,0) == 1} {
            append txt " ([lindex $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)]) or" 
        }
        
    }
    set txt [string trimleft $txt " "]
    set txt [string trimright $txt " or"]
    if {$QWIKMD::membraneFrame != ""} {
        set structure $QWIKMD::membraneFrame
        append txt " or \(chain W L\)"  
    }
    set sel [atomselect $structure $txt]
    
    if {[llength $QWIKMD::renameindex] > 0} {
        puts $textLogfile [QWIKMD::renameLog]
    }

    for {set i 0} {$i < [llength $QWIKMD::renameindex]} {incr i} {
        set val [split [lindex $QWIKMD::renameindex $i] "_"]
        
        set sel_rename [atomselect $structure "resid \"[lindex $val 0]\" and chain \"[lindex $val 1]\""]
        $sel_rename set resname $QWIKMD::dorename([lindex $QWIKMD::renameindex $i])
        $sel_rename delete 
    }

    if {$QWIKMD::membraneFrame != ""} {
        
        set lipid [atomselect $structure "chain L and lipid"]
        set resid 1
        set prev ""
        foreach residue [$lipid get residue]  {
            set txt " $residue"
            if {$prev != $txt} {
                set selaux [atomselect $structure "(residue $residue and chain L and lipid)"]
                $selaux set resid $resid
                $selaux set segname "L"
                incr resid
                $selaux delete
                set prev $residue
            }
        }
        $lipid delete
    }
    set topfiles [list]
    foreach files $QWIKMD::TopList {
        lappend topfiles [file tail $files]
    }

    if {[llength $QWIKMD::atmsRenameLog] > 0 || [llength $QWIKMD::atmsReorderLog] > 0} {
        puts $textLogfile [QWIKMD::renameReorderAtomLog]
    }

    set stfile [lindex [molinfo [molinfo top] get filename] 0]
    set sel_tem "[file tail [file root [lindex $stfile 0] ] ]_sel.pdb"
    $sel set beta 0
    $sel set occupancy 0 

    $sel writepdb $sel_tem
    $sel delete

    set structure [mol new $sel_tem waitfor all]
    display update on

    display update ui

    set sel_aux [atomselect $structure "qwikmd_protein or qwikmd_nucleic or qwikmd_glycan or qwikmd_lipid"]
    set segments [lsort -unique [$sel_aux get segname]]
    $sel_aux delete
    set mutate ""
    for {set i 0} {$i < [llength $QWIKMD::mutindex]} {incr i} {
        set chain [lindex [split [lindex $QWIKMD::mutindex $i] "_"] end]
        set index [lsearch $segments $chain\*]
        set residchain [split [lindex $QWIKMD::mutindex $i] "_" ]
        lappend mutate "[lindex $residchain 1] [lindex $residchain 0] [lindex $QWIKMD::mutate([lindex $QWIKMD::mutindex $i]) 1]"
    }

    

    if {[llength $QWIKMD::mutindex] > 0} {
        puts $textLogfile [QWIKMD::mutateLog]
    }
    for {set i 0} {$i < [llength $QWIKMD::protindex]} {incr i} {
        if {[lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1] == "HSP" || \
            [lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1] == "HSE" || [lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1] == "HSD" } {

            set chain [lindex [split [lindex $QWIKMD::protindex $i] "_"] end]
            set index [lsearch $segments $chain\*]
            set residchain [split [lindex $QWIKMD::protindex $i] "_" ]
            lappend mutate "[lindex $residchain 1] [lindex $residchain 0] [lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1]"
        }
    }
    

    if {$QWIKMD::membraneFrame != ""} {
        puts $textLogfile [QWIKMD::membraneLog]
    }

    if {[llength $QWIKMD::protindex] > 0} {
        puts $textLogfile [QWIKMD::protonateLog]
    }
    set patches [list]

    ################################################################################
    ## the protonation state is now changed through autopsf using the patch flag
    ## and not after autopsf and by QwikMD
    ################################################################################

    for {set i 0} {$i < [llength $QWIKMD::protindex]} {incr i} {
        set residchain [split [lindex $QWIKMD::protindex $i] "_" ]
        set chain [lindex $residchain end]
        set resid [lindex $residchain 0]

        if {[lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1] != "HSP" && \
            [lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1] != "HSE" && [lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1] != "HSD" } {
            lappend patches "[lindex $QWIKMD::protonate([lindex $QWIKMD::protindex $i]) 1] $chain [lindex $residchain 0]"
        }
        
    } 
    if {[llength $QWIKMD::patchestr] > 0} {
        set patches [concat $patches $QWIKMD::patchestr]
        puts $textLogfile [QWIKMD::patchLog]
    }
    # set nlines [expr [lindex [split [${QWIKMD::selresPatche}.text index end] "."] 0] -1]
    # set patchtext [split [${QWIKMD::selresPatche}.text get 1.0 $nlines.end] "\n"]
    # set patchaux [list]
    # if {[lindex $patchtext 0] != ""} {
    #   foreach patch $patchtext {
    #       if {$patch != ""} {
    #           lappend patchaux $patch
    #       }
    #   }
        
    # }

    set hetero $QWIKMD::heteromcr
    set protein $QWIKMD::proteinmcr
    set nucleic $QWIKMD::nucleicmcr
    set glycan $QWIKMD::glycanmcr
    set lipid $QWIKMD::lipidmcr
    if {[llength $QWIKMD::userMacros] >0} {
        foreach macro $QWIKMD::userMacros {
            switch $macro {
                protein {
                    append QWIKMD::proteinmcr " or [lindex $macro 0]"
                    atomselect macro qwikmd_protein $QWIKMD::proteinmcr
                }
                nucleic {
                    append QWIKMD::nucleicmcr " or [lindex $macro 0]"
                    atomselect macro qwikmd_nucleic $QWIKMD::nucleicmcr
                }
                glycan {
                    append QWIKMD::glycanmcr " or [lindex $macro 0]"
                    atomselect macro qwikmd_glycan $QWIKMD::glycanmcr
                }
                lipid {
                    append QWIKMD::lipidmcr " or [lindex $macro 0]"
                    atomselect macro qwikmd_lipid $QWIKMD::lipidmcr
                }
                default {
                    append QWIKMD::heteromcr " or [lindex $macro 0]"
                    atomselect macro qwikmd_hetero $QWIKMD::heteromcr
                }
            }
        }
    }
    
    set atpsfOk ""
    set autopsfLog ""

    #set atpsfLogFile [open "StructureFile.log" w+]
 
    #set cmd "autopsf -mol ${structure} -prefix ${prefix} -top [lreverse [list ${topfiles}]] -patch [list ${patches}] -regen -mutate [list ${mutate}] -qwikmd"
    
    #set atpsfOk [QWIKMD::redirectPuts $atpsfLogFile $cmd]
    set autopsfLog ""
    set atpsfOk [catch {autopsf -mol ${structure} -prefix ${prefix} -top [join $topfiles] -patch ${patches} -regen -mutate ${mutate} -qwikmd} autopsfLog ]
    #close $atpsfLogFile
    if {$atpsfOk >= 1} {
        return [list 1 "Autopsf error $autopsfLog"]
    }

    if {[llength $QWIKMD::userMacros] > 0} {
        foreach macro $QWIKMD::userMacros {
            switch $macro {
                protein {
                    set QWIKMD::proteinmcr $protein
                    atomselect macro qwikmd_protein $QWIKMD::proteinmcr
                }
                nucleic {
                    set QWIKMD::nucleicmcr $nucleic
                    atomselect macro qwikmd_nucleic $QWIKMD::nucleicmcr
                }
                glycan {
                    set QWIKMD::glycanmcr $glycan
                    atomselect macro qwikmd_glycan $QWIKMD::glycanmcr
                }
                lipid {
                    set QWIKMD::lipidmcr $lipid
                    atomselect macro qwikmd_lipid $QWIKMD::lipidmcr
                }
                default {
                    set QWIKMD::heteromcr $hetero
                    atomselect macro qwikmd_hetero $QWIKMD::heteromcr
                }
            }
        }
    }

    if {[file exists  ${prefix}_formatted_autopsf.psf] != 1} {
        return [list 1 "Please inspect VMD outputs for more information."]
    }

    file rename ${prefix}_formatted_autopsf.psf ${prefix}.psf
    file rename ${prefix}_formatted_autopsf.pdb ${prefix}.pdb

    mol new $prefix.psf
    mol addfile $prefix.pdb waitfor all
    set stfile [molinfo [molinfo top] get filename]
    set stctFile  [lindex $stfile 0] 
    
    set prefix  [file root [lindex $stctFile 0]]
    set pdb ""
    set psf ""

    # if MDFF protocol is selected, don't bring the structure to the origin. Assume that structure is aligned with
    # the density maps already
    if {$QWIKMD::run != "MDFF"} {
        QWIKMD::orientMol [molinfo top]
    }
    set selall [atomselect [molinfo top] "all"]
    $selall writepdb orient.pdb
    $selall delete
    ################################################################################
    ## In the equilibration MD simulaitons only the backbone is restrain. SMD simulations require
    ## the identification of the anchor residues by the beta column and the pulling residues
    ## by the occupancy collumn. 
    ################################################################################
    set name ""
    regsub -all "_formatted_autopsf" [file root [file tail [lindex $stctFile 0] ] ] "" name
    set sufix "_QwikMD"
    set solvent $QWIKMD::basicGui(solvent,0)
    set dist 7.5
    set tabid [$QWIKMD::topGui.nbinput index current]
    if {$tabid == 1} {
        set solvent $QWIKMD::advGui(solvent,0)
        if {$solvent == "Explicit"} {
            set dist $QWIKMD::advGui(solvent,boxbuffer)
        }
    }
    set constpdb [list]
    if {$solvent == "Explicit"} {
        set membrane 0
        set solv "_solvated"
        set solvateOK ""
        set dimList [list]
        if {$QWIKMD::membraneFrame != ""} {
            set membrane 1
            set lipidsel [atomselect top "chain L and not water"]
            set selall [atomselect top "all"]
            set zminmax [measure minmax $lipidsel]
            set allzminmax [measure minmax $selall]

            set zallmin [lindex [lindex $allzminmax 0] 2]
            set zallmax [lindex [lindex $allzminmax 1] 2]

            set xmin [lindex [lindex $zminmax 0] 0]
            set xmax [lindex [lindex $zminmax 1] 0]
            set ymin [lindex [lindex $zminmax 0] 1]
            set ymax [lindex [lindex $zminmax 1] 1]
            set zmin [lindex [lindex $zminmax 0] 2]
            set zmax [lindex [lindex $zminmax 1] 2]

            set zmintemp [expr $zallmin - $QWIKMD::advGui(solvent,boxbuffer)]
            lappend dimList [list [list $xmin $ymin $zmintemp] [list $xmax $ymax $zmin]]
            #set cmd "solvate [lindex $stctFile 0] orient.pdb -minmax \{ \{$xmin $ymin $zmintemp\} \{ $xmax $ymax $zmin\} \} -o \"solvateAux\" -s \"WL\""
            #set solvateOK [QWIKMD::redirectPuts $solvateLog $cmd]
            set solvateLog ""
            set solvateOK [catch {solvate [lindex $stctFile 0] orient.pdb -minmax [list [list $xmin $ymin $zmintemp] [list $xmax $ymax $zmin ] ] -o "solvateAux" -s "WL"} solvateLog]

            if {$solvateOK >= 1} {
                #close $solvateLog
                return [list 1 $solvateLog]

            }
            set zmaxtemp [expr $zallmax + $QWIKMD::advGui(solvent,boxbuffer)]

            lappend dimList [list [list $xmin $ymin $zmax] [list $xmax $ymax $zmaxtemp]]
            #set cmd "solvate solvateAux.psf solvateAux.pdb -minmax \{ \{$xmin $ymin $zmax \} \{$xmax $ymax $zmaxtemp\} \} -o $prefix$solv -s \"WU\""
            #set solvateOK [QWIKMD::redirectPuts $solvateLog $cmd]

            set solvateLog ""
            set solvateOK [catch {solvate solvateAux.psf solvateAux.pdb -minmax [list [list $xmin $ymin $zmax] [list $xmax $ymax $zmaxtemp ] ] -o $prefix$solv -s "WU"} solvateLog]

            if {$solvateOK >= 1} {
                #close $solvateLog
                return [list 1 "Solvate error $solvateLog"]
            }

            
            if {$QWIKMD::membraneFrame != ""} {
                set sel [atomselect top "all and water"]
                set minmax [measure minmax $sel]
                set xlength [expr [lindex [lindex $minmax 1] 0] - [lindex [lindex $minmax 0] 0] ]
                set ylength [expr [lindex [lindex $minmax 1] 1] - [lindex [lindex $minmax 0] 1] ]
                set zlength [expr [lindex [lindex $minmax 1] 2] - [lindex [lindex $minmax 0] 2] ]
                pbc set [list $xlength $ylength $zlength]
                $sel delete
                set sel [atomselect top "all"]
                $sel writepdb ${prefix}${solv}.pdb
                $sel delete
                
            }
            $lipidsel delete
            $selall delete
        } else {
            set cmd "solvate [lindex $stctFile 0] orient.pdb "
            if {$QWIKMD::advGui(solvent,minimalbox) == 0 || $tabid == 0} {
                QWIKMD::boxSize [molinfo top] $dist
                append cmd "-minmax \{[lrange $QWIKMD::cellDim 0 1]\} "
            } else {
                if {$QWIKMD::run == "MD" && $tabid == 1} {
                    append cmd "-rotate "
                }
                append cmd "-t $dist "
            }
            append cmd "-o $prefix$solv"
            
            # set cmd "solvate [lindex $stctFile 0] orient.pdb -minmax \{[lrange $QWIKMD::cellDim 0 1]\} -o $prefix$solv"
            #set solvateOK [QWIKMD::redirectPuts $solvateLog $cmd ]

            set solvateLog ""
            set solvateOK [eval catch {$cmd} solvateLog]

            if {$solvateOK >= 1} {
                #close $solvateLog
                return [list 1 "Solvate error $solvateLog"]
            }
            if {$QWIKMD::advGui(solvent,minimalbox) == 1} {
                set selDim [atomselect top "all"]
                set minmax [measure minmax $selDim]
                set xsp [lindex [lindex $minmax 0] 0]
                set ysp [lindex [lindex $minmax 0] 1]
                set zsp [lindex [lindex $minmax 0] 2]

                set xep [lindex [lindex $minmax 1] 0]
                set yep [lindex [lindex $minmax 1] 1]
                set zep [lindex [lindex $minmax 1] 2]

                set boxmin [list $xsp $ysp $zsp]
                set boxmax [list $xep $yep $zep]

                set centerX [expr [expr $xsp + $xep] /2]
                set centerY [expr [expr $ysp + $yep] /2]
                set centerZ [expr [expr $zsp + $zep] /2]

                set cB1 [expr abs($xep - $xsp)]
                set cB2 [expr abs($yep - $ysp)]
                set cB3 [expr abs($zep - $zsp)]

                set center [list [format %.2f $centerX] [format %.2f $centerY] [format %.2f $centerZ]]
                set length [list [format %.2f $cB1] [format %.2f $cB2] [format %.2f $cB3]]
                set QWIKMD::cellDim [list $boxmin $boxmax $center $length]
            }
            set dimList [lrange $QWIKMD::cellDim 0 1]
        }
        
        puts $textLogfile [QWIKMD::solvateLog $membrane $dimList]

        #close $solvateLog
        set cation "SOD"
        set anion "CLA"
        set ions $QWIKMD::basicGui(saltions,0)
        if {$tabid == 1} {
            set ions $QWIKMD::advGui(saltions,0)
        }
        switch $ions {
            case NaCl {
                return
            } 
            case KCl {
                set cation "POT"
            }
        }

        set atIonizeLog ""
        set atIonizeOk [catch {autoionize -psf $prefix$solv.psf -pdb $prefix$solv.pdb -sc $QWIKMD::basicGui(saltconc,0) -o ionized -cation $cation -anion $anion} atIonizeLog ]
        if {$atIonizeOk >= 1} {
            #close $solvateLog
            return [list 1 "Autoionize error $atIonizeLog"]
        }
        puts $textLogfile [QWIKMD::ionizeLog $QWIKMD::basicGui(saltconc,0) $cation $anion]
        set constpdb [list ionized.psf  ionized.pdb]
    } else {
        set constpdb [list [lindex $stctFile 0] orient.pdb]
    } 
    if {$QWIKMD::run != "MDFF" && $tabid == 0} {
        if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,equi) == 1 } {
            set mol [mol new [lindex $constpdb 0] ]
            mol addfile [lindex $constpdb 1] waitfor all
            set all [atomselect top "all"]
            set sel [atomselect top "(qwikmd_protein or qwikmd_nucleic or qwikmd_glycan or qwikmd_lipid) and backbone"]

            $all set beta 0
            $sel set beta 1

            $all writepdb [lindex $QWIKMD::confFile 0]_constraints.pdb
            mol delete $mol
            $sel delete
            $all delete
            file copy -force [lindex $QWIKMD::confFile 0]_constraints.pdb ../run/[lindex $QWIKMD::confFile 0]_constraints.pdb
        }
    }
    

    if {$tabid == 1 && $QWIKMD::run != "MDFF"} {
        set restrains [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 2]
        for {set i 0} {$i < [llength $restrains]} {incr i} {
            set do 0
            set text [lindex $restrains $i]
            if {$text != "none"} {
                set do 1
                if {$i > 0} {
                    if {[lindex $restrains $i] == [lindex $restrains [expr $i -1] ]} {
                        set do 0
                    }
                }
            }

            if {$do == 1 } {
                set all [atomselect top "all"]
                set sel [atomselect top $text]
                $all set beta 0
                $sel set beta 1
                #set indexes [$sel get index]

                $all writepdb [lindex $QWIKMD::confFile $i]_restraints.pdb
                #set file [open "[lindex $QWIKMD::confFile $i]_restraints.txt" w+]
                #set indaux ""
                #foreach ind $indexes {
                #    lappend indaux [expr $ind + 1]
                #}
                #puts $file $indaux
                #close $file
                $sel delete
                $all delete
                file copy -force [lindex $QWIKMD::confFile $i]_restraints.pdb ../run/
            }
        }
    }

    if {$QWIKMD::run == "SMD"} {
        set filename "ionized.pdb"
        if {$solvent != "Explicit"} {
            set mol [ mol new [lindex $stctFile 0]]
            mol addfile orient.pdb  waitfor all
            set filename "orient.pdb"
        } 
        set all [atomselect top "all"]
        $all set beta 0
        $all set occupancy 0
        set beta [atomselect top $QWIKMD::anchorRessel]
        set occupancy [atomselect top $QWIKMD::pullingRessel]
        # set indexes [$beta get index]
        # set file [open "SMD_anchorIndex.txt" w+]
        # puts $file $indexes
        # close $file

        # set indexes [$occupancy get index]
        # set file [open "SMD_pullingIndex.txt" w+]
        # puts $file $indexes
        # close $file
        $beta set beta 1
        $occupancy set occupancy 1
        $beta delete
        $occupancy delete

        $all writepdb $filename 
        $all delete
        # file copy -force "SMD_anchorIndex.txt" ../run/
        # file copy -force "SMD_pullingIndex.txt" ../run/
    }

    
    if {$solvent != "Explicit"} {
        file copy -force [lindex $stctFile 0] ../run/$name$sufix.psf
        file copy -force orient.pdb ../run/$name$sufix.pdb
    } else {
        file copy -force ionized.psf ../run/$name$sufix.psf
        file copy -force ionized.pdb ../run/$name$sufix.pdb
    }
    
    
    set pdb $name$sufix.pdb
    set psf $name$sufix.psf

    set QWIKMD::topMol [molinfo top]
    
    return "$psf $pdb"
}

################################################################################
## Creation of the config file for namd. In this case we have two "templates" for
## for MD and SMD simulaitons. In the next versions more templates will be required,
## so a more inteligent proc will be necessary and less hardcoded variables will be used 
################################################################################
proc QWIKMD::isSMD {filename} {
    set returnval 0
    set templatefile [open "$filename" r]
    set line [read $templatefile]
    set line [split $line "\n"]

    set enter ""
    set lineIndex [lsearch -exact -all $line $enter]
    for {set j 0} {$j < [llength $lineIndex]} {incr j} {
        lset line [lindex $lineIndex $j] "{} {}"
    }
    set smdtxt "SMD on"
    
    set lineIndex [lsearch -regexp $line (?i)$smdtxt$]
    if {$lineIndex != -1} {
        set returnval 1
    }
    close $templatefile
    return $returnval
}
proc QWIKMD::NAMDGenerator {strct step} {
    global env
    
    set tabid [$QWIKMD::topGui.nbinput index current]
    if {$tabid == 0} {      
        set conf [lindex $QWIKMD::confFile $step]   
        QWIKMD::GenerateBasicNamdFiles $strct $step
        set i 0
        while {$i < [llength  $QWIKMD::confFile]} {
            if {[string match "*_production_smd*" [lindex $QWIKMD::confFile $i] ] > 0} {
                break
            }
            incr i
        }
        # if {[string match "*_production_smd*" [lindex $QWIKMD::confFile $step] ] > 0 && $QWIKMD::run == "SMD" && $i == $step} {
        #     set input SMD_pullingIndex.txt
        #     set output SMD_Index.pdb
        #     set pdb SMD_Index.pdb
        #     set start 0
        #     set midle 53
        #     set end 60
        #     set pdb SMD_Index.pdb
            
        #     QWIKMD::addConstOccup ${QWIKMD::outPath}/run/$conf.conf $input $output $pdb $start $midle $end
    
        #     set input SMD_anchorIndex.txt
        #     set output SMD_Index.pdb
        #     set start 0
        #     set midle 59
        #         set end 66
        #     if {$step > 0} {
        #         set pdb [lindex $QWIKMD::confFile [expr $step -1]].coor
        #     } else {
        #         set pdb [lindex $strct 1]
        #     }
        #     QWIKMD::addConstOccup ${QWIKMD::outPath}/run/$conf.conf $input $output $pdb $start $midle $end
        # }

    } else {
        set QWIKMD::confFile [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
        set conf [lindex $QWIKMD::confFile $step]
        set restrains [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 2]
        set outputfile ""
        set args [$QWIKMD::advGui(protocoltb,$QWIKMD::run) rowcget $step -text]


        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,smd) 0
        set location ""
        set tempfilename ""
        set values {Minimization Annealing Equilibration MD SMD}
        set serachindex [lsearch $values [file root $conf] ]
        set location "$env(QWIKMDFOLDER)/templates/"
        if {$serachindex == -1} {
            append location $QWIKMD::advGui(solvent,0)
        }
        if {[file exists "$env(TMPDIR)/$conf.conf"] != 1 && [file exists "$env(TMPDIR)/[file root $conf].conf"] != 1} {
            if {[file exists "${QWIKMD::outPath}/run/[file root $conf].conf"] == 1 } {
                if {$QWIKMD::run == "SMD"} {
                    if {[QWIKMD::isSMD "$location/[file root $conf].conf"] == 1} {
                        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,smd) 1 
                    }
                }
                QWIKMD::GenerateNamdFiles "qwikmdTemp.psf qwikmdTemp.pdb" "$location/[file root $conf].conf" [lsearch $QWIKMD::prevconfFile [file root $conf]] [$QWIKMD::advGui(protocoltb,$QWIKMD::run) rowcget [expr $step -1] -text] "$env(TMPDIR)/[file root $conf].conf"
                set location  $env(TMPDIR)/
                set tempfilename [file root $conf].conf
            } else {
                #set location ${env(QWIKMDFOLDER)}/templates/
                set tempfilename [file root $conf].conf
            }
        } else {            
            set location  $env(TMPDIR)
            set tempfilename [file root $conf].conf
            if {[file exists "$env(TMPDIR)/$conf.conf"] != 1} {
                set tempfilename [file root $conf].conf
            } else {
                set tempfilename $conf.conf
            }
            
        }

        if {$QWIKMD::run == "SMD"} {
            if {[QWIKMD::isSMD "$location/$tempfilename"] == 1} {
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,smd) 1 
            }
        }

        if {[file exists $env(TMPDIR)/$conf.conf] != 1 } {
            set outputfile ${QWIKMD::outPath}/run/$conf.conf
            QWIKMD::GenerateNamdFiles $strct "$location/$tempfilename" $step $args ${outputfile}
        } else {
            set auxfile [open  $env(TMPDIR)/$conf.conf r]
            set QWIKMD::line [read $auxfile]
            set QWIKMD::line [split $QWIKMD::line "\n"]
            close $auxfile
            set enter ""
            set index [lsearch -exact -all $QWIKMD::line $enter]
            for {set j 0} {$j < [llength $index]} {incr j} {
                lset QWIKMD::line [lindex $index $j] "{} {}"
            }

            set auxfile [open ${QWIKMD::outPath}/run/$conf.conf w+]
            QWIKMD::replaceNAMDLine "coordinates" "coordinates [lindex $strct 1]"
            QWIKMD::replaceNAMDLine "structure" "structure [lindex $strct 0]"

            #replace the restart files in case of addition of protocols in middle of an already created protocol

            if {$step > 0} {
                set inputname [lindex $QWIKMD::confFile [expr $step -1]]
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)#binCoordinates$]
                if {$index != -1} {
                    QWIKMD::replaceNAMDLine "#binCoordinates" "binCoordinates $inputname.restart.coor"
                } else {
                     set index [lsearch -regexp -index 0 $QWIKMD::line (?i)binCoordinates$]
                     if {$index != -1} {
                        QWIKMD::replaceNAMDLine "binCoordinates" "binCoordinates $inputname.restart.coor"
                    } else {
                        puts $auxfile "binCoordinates $inputname.restart.coor"
                    }
                }
                
                QWIKMD::replaceNAMDLine "binVelocities" "binVelocities $inputname.restart.vel"
                
                 if {$QWIKMD::advGui(solvent,0) == "Explicit"} {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)#extendedSystem$]
                    if {$index != -1} {
                        QWIKMD::replaceNAMDLine "#extendedSystem" "extendedSystem $inputname.restart.xsc"
                    } else {
                        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
                        if {$index != -1} {
                            QWIKMD::replaceNAMDLine "extendedSystem" "extendedSystem $inputname.restart.xsc"
                        } else {
                            puts $auxfile "extendedSystem $inputname.restart.xsc"
                        }
                    }
                } else {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
                    QWIKMD::replaceNAMDLine "extendedSystem" "#extendedSystem $inputname.restart.xsc"
                }
                if {$QWIKMD::run == "SMD"} {
                    if {[QWIKMD::isSMD "$env(TMPDIR)/$conf.conf"] == 1} {
                        if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 1} {
                            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMDk$]
                            if {$index != -1} {
                                set QWIKMD::mdProtInfo($inputname,smdk) [string trim [join [lindex [lindex $QWIKMD::line $index] 1]]]
                            }

                            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMDVel$]
                            if {$index != -1} {
                                set QWIKMD::basicGui(pspeed) [QWIKMD::format2Dec [expr [expr [string trim [join [lindex [lindex $QWIKMD::line $index] 1]]]  / $QWIKMD::mdProtInfo($inputname,timestep) ] * 1e6 ] ]
                                set QWIKMD::mdProtInfo($inputname,pspeed) $QWIKMD::basicGui(pspeed)
                            }
                        }
                        set i 0
                        while {$i < [llength  $QWIKMD::confFile]} {
                            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$i,smd) == 1} {
                                break
                            }
                            incr i
                        }
                        set str ""
                        if {$i == $step} {
                            set str "firstTimestep 0"
                        } else {
                           set str [QWIKMD::addFirstTimeStep $step] 
                        }
                        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)firstTimestep$]
                        if {$index != -1} {
                            QWIKMD::replaceNAMDLine "firstTimestep" "$str"
                        } else {
                            puts $namdfile $str
                        }
                    }
                }
            } elseif {$step == 0} {
                if {$QWIKMD::advGui(solvent,0) == "Explicit"} {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
                    if { $index != -1} {
                        QWIKMD::replaceNAMDLine "extendedSystem" "extendedSystem $conf.xsc"
                        
                    } else {
                        puts $auxfile "extendedSystem $conf.xsc"
                    }
                } else {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
                    QWIKMD::replaceNAMDLine "extendedSystem" "#[lindex $QWIKMD::line $index]"
                }
                
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)binCoordinates$]
                QWIKMD::replaceNAMDLine "binCoordinates" "#[lindex $QWIKMD::line $index]"
                
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)binVelocities$]
                QWIKMD::replaceNAMDLine "binVelocities" "#[lindex $QWIKMD::line $index]"
            }

            ## Check if the restraints were changed after file edition

            if {[$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $step,2 -text] != "none" && $QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,smd) == 0} {
                #if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 0} {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)constraints$]
                    if { $index != -1} {
                        QWIKMD::replaceNAMDLine "constraints" "constraints on"
                        
                    } else {
                        puts $auxfile "constraints on"
                    }
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)conskcol$]
                    if { $index != -1} {
                        QWIKMD::replaceNAMDLine "conskcol" "conskcol B"
                        
                    } else {
                        puts $auxfile "conskcol B"
                    }
                #}
                set restrains [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 2]
                set index [lsearch $restrains [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $step,2 -text]]
                set reffile [lindex $QWIKMD::confFile $step]_restraints.pdb
                set constfile [lindex $QWIKMD::confFile $step]_restraints.pdb
                if {$index > -1} {
                    if {$step > 0} {
                        if {[lindex $restrains [expr $step -1]] == [lindex $restrains $step]} {
                            set stepaux $step
                            while {[lindex $restrains [expr $stepaux -1]] == [lindex $restrains $stepaux]} {
                                incr stepaux -1
                                if {$stepaux == 0} {
                                    break
                                }
                            }
                            if {$stepaux >= 1} {
                                set constfile [lindex $QWIKMD::confFile $stepaux]_restraints.pdb
                                set reffile [lindex $QWIKMD::confFile [expr $stepaux -1] ].coor
                            } else {
                                set constfile [lindex $QWIKMD::confFile 0]_restraints.pdb
                                set reffile [lindex $QWIKMD::confFile 0]_restraints.pdb
                            }
                        } else {
                            set reffile [lindex $QWIKMD::confFile [expr $step - 1] ].coor
                        }
                    } 
                } elseif {$step > 0} {
                    set reffile [lindex $QWIKMD::confFile [expr $step - 1] ].coor
                }
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)consref$]
                if { $index != -1} {
                    QWIKMD::replaceNAMDLine "consref" "consref $reffile"
                    
                } else {
                    puts $auxfile "consref $reffile"
                }
                
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)conskfile$]
                if { $index != -1} {
                    QWIKMD::replaceNAMDLine "conskfile" "conskfile $constfile"
                } else {
                    puts $auxfile "conskfile $constfile"
                }
                set QWIKMD::mdProtInfo($conf,const) 1
                set QWIKMD::mdProtInfo($conf,constsel) [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $step,2 -text]
            } elseif {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,smd) == 0} {
                set QWIKMD::mdProtInfo($conf,const) 0
                QWIKMD::replaceNAMDLine "constraints" "constraints off"
            }

            if {$QWIKMD::basicGui(live) == 1} {
                set index [lsearch -regexp $QWIKMD::line "IMDon on"]
                if {$index == -1} {
                    puts $auxfile  "# IMD Parameters"
                    puts $auxfile  "IMDon on    ;#"
                    puts $auxfile  "IMDport 3000    ;# port number (enter it in VMD)"
                    puts $auxfile  "IMDfreq 10  ;# send every 10 frame"
                    puts $auxfile  "IMDwait yes ;# wait for VMD to connect before running?\n"
                }
            }
            set enter "{} {}"
            set index [lsearch -exact -all $QWIKMD::line $enter]
            for {set i 0} {$i < [llength $index]} {incr i} {
                lset QWIKMD::line [lindex $index $i] "\n"
            }
            for {set j 0 } {$j < [llength $QWIKMD::line]} {incr j} {
                puts $auxfile [lindex $QWIKMD::line $j]
            }
            
            close $auxfile
            set outputfile ${QWIKMD::outPath}/run/$conf.conf

        }
        
    
        if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,saveAsTemplate) == 1} {
            file copy -force ${env(TMPDIR)}/$conf.conf ${env(QWIKMDFOLDER)}/templates/$QWIKMD::advGui(solvent,0)
        }
        set index [lsearch $restrains [lindex $args 2]]
        set tbline $step
        
        set i 0
        while {$i < $step} {
            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$i,smd) == 1} {
                break
            }
            incr i
        }
        if {$QWIKMD::run == "SMD" && $QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,smd) == 1 && $i == $step} {
            # if {$tbline == $index || $QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) == 1} {
            #     set start 0
            #     set midle 59
            #     set end 66
            #     set input ${conf}_restraints.txt
            #     set output ${conf}_restraints.pdb
            #     set pdb [lindex $strct 1]
            #     if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) == 1} {
            #         set input SMD_pullingIndex.txt
            #         set output SMD_Index.pdb
            #         set pdb SMD_Index.pdb
            #         set start 0
            #         set midle 53
            #         set end 60
                    
            #     }
            
            #     QWIKMD::addConstOccup ${QWIKMD::outPath}/run/$conf.conf $input $output $pdb $start $midle $end
            #     if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) == 1} {
                        
            #         set input SMD_anchorIndex.txt
            #         set output SMD_Index.pdb
            #         set start 0
            #         set midle 59
            #         set end 66
            #         if {$step > 0} {
            #             set pdb [lindex $QWIKMD::confFile [expr $step -1]].coor
            #         } else {
            #             set pdb [lindex $strct 1]
            #         }
            #         QWIKMD::addConstOccup ${QWIKMD::outPath}/run/$conf.conf $input $output $pdb $start $midle $end
            #     }
            # }
            set tbline $index
        }

        QWIKMD::addNAMDCheck $step
    
        if {$QWIKMD::advGui(solvent,0) == "Explicit" && $step == 0} {
            set file ${QWIKMD::outPath}/run/[lindex $QWIKMD::confFile 0].xsc
            pbc writexst $file
            set xst [open $file r]
            set line [read -nonewline $xst]
            close $xst
            set line [split $line "\n"]
            set values [lindex $line 2]
            set sel [atomselect top "all"]
            set center [measure center $sel]
            $sel delete
            lset values 10 [lindex $center 0]
            lset values 11 [lindex $center 1]
            lset values 12 [lindex $center 2]
            lset line 2 $values

            set xst [open $file w+]
            puts $xst [lindex $line 0]
            puts $xst [lindex $line 1]
            puts $xst [lindex $line 2]
            close $xst
        }

    }

}

proc QWIKMD::GenerateBasicNamdFiles {strct step} {
    cd ${QWIKMD::outPath}/run
    
    set prefix [lindex $QWIKMD::confFile $step]

    set QWIKMD::mdProtInfo($prefix,temp) 0
    set QWIKMD::mdProtInfo($prefix,const) 0
    set QWIKMD::mdProtInfo($prefix,constsel) "protein and backbone"
    set QWIKMD::mdProtInfo($prefix,minimize) 0
    set QWIKMD::mdProtInfo($prefix,smd) 0
    set QWIKMD::mdProtInfo($prefix,smdk) 7.0
    set QWIKMD::mdProtInfo($prefix,ramp) 0
    set QWIKMD::mdProtInfo($prefix,timestep) 2
    set QWIKMD::mdProtInfo($prefix,vdw) 1
    set QWIKMD::mdProtInfo($prefix,electro) 2
    set QWIKMD::mdProtInfo($prefix,cutoff) 12.0
    set QWIKMD::mdProtInfo($prefix,pairlist) 14.0
    set QWIKMD::mdProtInfo($prefix,switch) 10.0
    set QWIKMD::mdProtInfo($prefix,switching) 1
    set QWIKMD::mdProtInfo($prefix,gbis) 0
    set QWIKMD::mdProtInfo($prefix,alphacut) 14.0
    set QWIKMD::mdProtInfo($prefix,solvDie) 80.0
    set QWIKMD::mdProtInfo($prefix,sasa) 0
    set QWIKMD::mdProtInfo($prefix,rampList) [list]
    set QWIKMD::mdProtInfo($prefix,ensemble) "NpT"
    set QWIKMD::mdProtInfo($prefix,run) 0
    set QWIKMD::mdProtInfo($prefix,thermostat) 0
    set QWIKMD::mdProtInfo($prefix,barostat) 0

    if {[llength $QWIKMD::maxSteps] == $step} {
        lappend QWIKMD::maxSteps 0
    }
    set namdfile [open $prefix.conf w+]
    set temp [QWIKMD::format2Dec [expr $QWIKMD::basicGui(temperature,0) + 273]]
    
    puts $namdfile [string repeat "#" 20]
    puts $namdfile [string repeat "#\n" 10]
    puts $namdfile [string repeat "#" 20]

    puts $namdfile "# Initial pdb and pdf files\n"
    puts $namdfile "coordinates [lindex $strct 1]"
    puts $namdfile "structure [lindex $strct 0]\n\n"
    if {$step > 0 } {
        puts $namdfile "set inputname      [lindex $QWIKMD::confFile [expr $step -1]]"
        puts $namdfile "binCoordinates     \$inputname.restart.coor"
        puts $namdfile "binVelocities      \$inputname.restart.vel" 
        puts $namdfile "extendedSystem     \$inputname.restart.xsc\n\n"
    }
    
    puts $namdfile "# Simulation conditions"
    puts $namdfile "set temperature $temp; # Conversion of $QWIKMD::basicGui(temperature,0) degrees Celsius + 273"

    set QWIKMD::mdProtInfo($prefix,temp) $temp

    if {$step == 0} {
        if {[string match "*equilibration*" [lindex $QWIKMD::confFile $step]] > 0} {
            puts $namdfile "temperature 0\n\n"
        } else {
            puts $namdfile "temperature $temp\n\n"
        }
    } else {
        puts $namdfile "\n"
    }
    set QWIKMD::mdProtInfo($prefix,const) 0
    if {[string match "*_equilibration*" [lindex $QWIKMD::confFile $step] ] > 0 || $QWIKMD::run == "SMD" && [string match "*_production_smd*" [lindex $QWIKMD::confFile $step] ] > 0} {
        puts $namdfile "# Harmonic constraints\n"
        puts $namdfile "constraints on"
        set QWIKMD::mdProtInfo($prefix,const) 1
        if {$QWIKMD::run == "SMD" && [string match "*_production_smd*" [lindex $QWIKMD::confFile $step] ] > 0} {
            
            if {$step > 0} {
                set index [lindex [lsearch -all $QWIKMD::confFile "*_production_smd*"] 0]
                puts $namdfile "consref [lindex $QWIKMD::confFile [expr $index -1 ]].coor"
                puts $namdfile "conskfile [lindex $QWIKMD::confFile [expr $index -1 ]].coor"
            } else {
                puts $namdfile "consref [lindex $strct 1]"
                puts $namdfile "conskfile [lindex $strct 1]"
            }
            
            puts $namdfile "constraintScaling 10"
            puts $namdfile "consexp 2"
        } else {
            puts $namdfile "consref [lindex $QWIKMD::confFile 0]_constraints.pdb"
            puts $namdfile "conskfile [lindex $QWIKMD::confFile 0]_constraints.pdb"
            puts $namdfile "constraintScaling 2"
            puts $namdfile "consexp 2"
            set QWIKMD::mdProtInfo($prefix,constsel) "protein and backbone"
        }
        
        puts $namdfile "conskcol B\n\n"
        
    }
    if {$QWIKMD::run == "SMD" && [string match "*_production_smd*" [lindex $QWIKMD::confFile $step] ] > 0 } {
        puts $namdfile "# steered dynamics"
        puts $namdfile "SMD on"
        set QWIKMD::mdProtInfo($prefix,smd) 1
        if {$step > 0} {
            set index [lindex [lsearch -all $QWIKMD::confFile "*_production_smd*"] 0] 
            puts $namdfile "SMDFile [lindex $QWIKMD::confFile [expr $index -1 ]].coor"
        } else {
            puts $namdfile "SMDFile [lindex $strct 1]"
        }
        set QWIKMD::mdProtInfo($prefix,smdk) 7.0
        set QWIKMD::mdProtInfo($prefix,pspeed) $QWIKMD::basicGui(pspeed)
        puts $namdfile "SMDk 7.0"
        puts $namdfile "SMDVel [format %.3g [expr $QWIKMD::basicGui(pspeed) * 2e-6]]"
        puts $namdfile "SMDDir 0.0 0.0 1.0"
        puts $namdfile "SMDOutputFreq $QWIKMD::smdfreq"
        set i 0
        while {$i < [llength  $QWIKMD::confFile]} {
            if {[string match "*_production_smd*" [lindex $QWIKMD::confFile $i] ] > 0} {
                break
            }
            incr i
        }

        if {$i == $step} {
            puts $namdfile "firstTimestep 0"
        } else {
            puts $namdfile [QWIKMD::addFirstTimeStep $step]
        }
        
    }
    

    puts $namdfile "# Output Parameters\n"
    puts $namdfile  "binaryoutput no"
    puts $namdfile  "outputname $prefix"
    set freq $QWIKMD::smdfreq
    if {$QWIKMD::basicGui(live) == 0} {
        set freq [expr $QWIKMD::smdfreq * 10]
    }
    puts $namdfile  "outputenergies $freq"
    puts $namdfile  "outputtiming $freq"
    puts $namdfile  "outputpressure $freq"
    
    set freq $QWIKMD::dcdfreq
    puts $namdfile  "binaryrestart yes"
    puts $namdfile  "dcdfile $prefix.dcd"
    puts $namdfile  "dcdfreq $freq"
    puts $namdfile  "XSTFreq $freq"
    puts $namdfile  "restartfreq $freq"

    puts $namdfile  "restartname $prefix.restart\n\n"
    set QWIKMD::mdProtInfo($prefix,ensemble) NpT 
    if {$step == 0 && $QWIKMD::basicGui(solvent,0) == "Explicit"} {
        puts $namdfile  "# Periodic Boundary Conditions"
        set length [lindex $QWIKMD::cellDim 3]
        set center [lindex $QWIKMD::cellDim 2]
        puts $namdfile  "cellBasisVector1     [lindex $length 0]   0.0   0.0"
        puts $namdfile  "cellBasisVector2     0.0   [lindex $length 1]   0.0"
        puts $namdfile  "cellBasisVector3     0.0    0     [lindex $length 2]"
        puts $namdfile  "cellOrigin           [lindex $center 0]  [lindex $center 1]  [lindex $center 2]\n\n"

    }

    if {$QWIKMD::basicGui(solvent,0) == "Explicit"} {
        set QWIKMD::mdProtInfo($prefix,PME) 1
        puts $namdfile  "# PME Parameters\n"
        puts $namdfile  "PME on"
        puts $namdfile  "PMEGridspacing 1\n\n"
    }
    set QWIKMD::mdProtInfo($prefix,thermostat) Langevin
    puts $namdfile  "# Thermostat Parameters\n"
    puts $namdfile  "langevin on"
    if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,equi) ==1 && $step == 0} {
        puts $namdfile  "langevintemp 60"
    } else {
        puts $namdfile  "langevintemp \$temperature"
        
    }

    puts $namdfile  "langevinHydrogen    off"
    puts $namdfile  "langevindamping 1\n\n"
    
    if {$QWIKMD::basicGui(solvent,0) == "Explicit"} {
        set QWIKMD::mdProtInfo($prefix,barostat) Langevin
        set QWIKMD::mdProtInfo($prefix,press) 1
        puts $namdfile  "# Barostat Parameters\n"
        puts $namdfile  "usegrouppressure yes"
        puts $namdfile  "useflexiblecell no"
        puts $namdfile  "useConstantArea no"
        puts $namdfile  "langevinpiston on"
        puts $namdfile  "langevinpistontarget 1.01325"
        puts $namdfile  "langevinpistonperiod 200"
        puts $namdfile  "langevinpistondecay 100"
        if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,equi) ==1 && $step == 0} {
            puts $namdfile  "langevinpistontemp 60\n\n"
        } else {
            puts $namdfile  "langevinpistontemp \$temperature\n\n"
        }
        puts $namdfile  "wrapAll on"
        puts $namdfile  "wrapWater on\n\n"
    }
    

    puts $namdfile  "# Integrator Parameters\n"
    puts $namdfile  "timestep 2"
    puts $namdfile  "fullElectFrequency 2"
    puts $namdfile  "nonbondedfreq 1\n\n"

    set QWIKMD::mdProtInfo($prefix,timestep) 2
    set QWIKMD::mdProtInfo($prefix,vdw) 1
    set QWIKMD::mdProtInfo($prefix,electro) 2

    puts $namdfile  "# Force Field Parameters\n"
    puts $namdfile  "paratypecharmm on"
    set parfiles [glob *.prm]
    set parfiles [concat $parfiles [glob *.str]]
    for {set i 0} {$i < [llength $parfiles]} {incr i} {
        puts $namdfile "parameters [file tail [lindex $parfiles $i]]"
    }
    puts $namdfile  "exclude scaled1-4"
    puts $namdfile  "1-4scaling 1.0"
    puts $namdfile  "rigidbonds all"
    
    if {$QWIKMD::basicGui(solvent,0) == "Explicit"} {
        set QWIKMD::mdProtInfo($prefix,cutoff) 12.0
        set QWIKMD::mdProtInfo($prefix,pairlist) 14.0
        set QWIKMD::mdProtInfo($prefix,switch) 10.0
        set QWIKMD::mdProtInfo($prefix,switching) 1
        puts $namdfile  "cutoff 12.0"
        puts $namdfile  "pairlistdist 14.0"
        puts $namdfile  "stepspercycle 10"
        puts $namdfile  "switching on"
        puts $namdfile  "switchdist 10.0\n\n"
    } else {
        set QWIKMD::mdProtInfo($prefix,ensemble) NVE
        set QWIKMD::mdProtInfo($prefix,gbis) 1
        set QWIKMD::mdProtInfo($prefix,cutoff) 16.0
        set QWIKMD::mdProtInfo($prefix,pairlist) 18.0
        set QWIKMD::mdProtInfo($prefix,switch) 15.0
        set QWIKMD::mdProtInfo($prefix,switching) 1
        set QWIKMD::mdProtInfo($prefix,alphacut) 14.0
        set QWIKMD::mdProtInfo($prefix,solvDie) 80.0
        set QWIKMD::mdProtInfo($prefix,sasa) 1
        puts $namdfile  "#Implicit Solvent Parameters\n"
        puts $namdfile  "gbis                on"
        puts $namdfile  "alphaCutoff         14.0"
        puts $namdfile  "ionConcentration    $QWIKMD::basicGui(saltconc,0)"

        puts $namdfile  "switching  on"
        puts $namdfile  "switchdist 15"
        puts $namdfile  "cutoff     16"
        puts $namdfile  "solventDielectric   80.0"
        puts $namdfile  "sasa                on"
        puts $namdfile  "pairlistdist 18\n\n"
    }

    if {$QWIKMD::basicGui(live) == 1} {
        puts $namdfile  "# IMD Parameters"
        puts $namdfile  "IMDon  on  ;#"
        puts $namdfile  "IMDport    3000    ;# port number (enter it in VMD)"
        puts $namdfile  "IMDfreq    10  ;# send every 10 frame"
        puts $namdfile  "IMDwait    yes ;# wait for VMD to connect before running?\n\n"
    }
    # if {$QWIKMD::basicGui(live) == 1 } {
        #lappend QWIKMD::maxSteps 0
    # }
    puts $namdfile  "# Script\n"
    set auxMaxstep 0 
    set QWIKMD::mdProtInfo($prefix,minimize) 0
    set QWIKMD::mdProtInfo($prefix,smd) 0
    set QWIKMD::mdProtInfo($prefix,ramp) 0
    if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,equi) == 1 && [string match "*_equilibration*" [lindex $QWIKMD::confFile $step] ] > 0} {
            puts $namdfile  "minimize 1000"
            set QWIKMD::mdProtInfo($prefix,minimize) 1000
            incr auxMaxstep 1000
    }
    if {[string match "*_equilibration*" [lindex  $QWIKMD::confFile $step] ] > 0} {
        if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,equi) ==1 && $step == 0} {
            set QWIKMD::mdProtInfo($prefix,ramp) 1
            set QWIKMD::mdProtInfo($prefix,rampList) [list 60 $QWIKMD::mdProtInfo($prefix,temp) [QWIKMD::format2Dec [expr 500 * [expr $temp - 60] *2e-6]]]  
            puts $namdfile  "for \{set t 60\} \{\$t <= \$temperature\} \{incr t\} \{"
            if {$QWIKMD::basicGui(solvent,0) == "Explicit"} {
                puts $namdfile  "\tlangevinpistontemp \$t"
            }
            puts $namdfile  "\trun 500"
            puts $namdfile  "\tlangevintemp \$t"
            puts $namdfile  "\}"
            incr auxMaxstep [expr round(500 * [expr $temp - 60])]
        }
        set QWIKMD::mdProtInfo($prefix,run) [QWIKMD::format2Dec [expr round( 500000 *2e-6)]] 
        puts $namdfile  "run 500000"
        set val 500000
        incr auxMaxstep $val
    } elseif {$QWIKMD::run == "SMD" && [string match "*_production_smd*" [lindex $QWIKMD::confFile $step] ] > 0 } { 
        set val [QWIKMD::format0Dec [expr $QWIKMD::basicGui(mdtime,1) / 2e-6 ]]
        set QWIKMD::mdProtInfo($prefix,run) $QWIKMD::basicGui(mdtime,1)
        set QWIKMD::mdProtInfo($prefix,smd) 1
        puts $namdfile  "run $val"
        incr auxMaxstep $val
    }   elseif {[string match "*_production*" [lindex $QWIKMD::confFile $step] ] > 0 } {
        set val [QWIKMD::format0Dec [expr $QWIKMD::basicGui(mdtime,0) / 2e-6 ]]
        set QWIKMD::mdProtInfo($prefix,run) $QWIKMD::basicGui(mdtime,0)
        puts $namdfile  "run $val"
        incr auxMaxstep $val
        
    }
    # if {$QWIKMD::basicGui(live) == 1 } {
    lset QWIKMD::maxSteps $step $auxMaxstep
    # }

    ################################################################################
    ## The next if statements force the evaluation of the normal termination, and
    ## writes to a check file to if the MD simulation terminated withh success and
    ## it is possible to restart the new simulation of if any of the restart files
    ## files to write and then it is not possible to restart from this point. In this case,
    ## the QWIKMD::state is decremented one step, and the current simulation starts from 
    ## the beginning 
    ################################################################################
    # set file "[lindex $QWIKMD::confFile $step].check"

    # puts $namdfile  "set file \[open $file w+\]"

    # puts $namdfile "set done 1"
    # set str $QWIKMD::run
    
    # puts $namdfile "set run $str"
    # puts $namdfile "if \{\[file exists $prefix.restart.coor\] != 1 || \[file exists $prefix.restart.vel\] != 1 || \[file exists $prefix.restart.xsc\] != 1 \} \{"
    # puts $namdfile "\t set done 0"
    # puts $namdfile "\}"

    # puts $namdfile "if \{\$done == 1\} \{"
    # puts $namdfile "\tputs \$file \"DONE\"\n    flush \$file\n  close \$file"
    # puts $namdfile "\} else \{"
    # puts $namdfile "\tputs \$file \"One or more files filed to be written\"\n   flush \$file\n  close \$file"
    # puts $namdfile "\}"
    close $namdfile
    QWIKMD::addNAMDCheck $step
    return 
}
proc QWIKMD::replaceNAMDLine {strcompare strreplace} {
    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^$strcompare$]
    
    if { $index != -1} {
        lset QWIKMD::line $index "$strreplace"
    }
}
proc QWIKMD::GenerateNamdFiles {strct template step args outputfile} {
    global env

    set namdfile ""
    set prefix [lindex $QWIKMD::confFile $step]

    set QWIKMD::mdProtInfo($prefix,temp) 0
    set QWIKMD::mdProtInfo($prefix,const) 0
    set QWIKMD::mdProtInfo($prefix,constsel) "protein and backbone"
    set QWIKMD::mdProtInfo($prefix,minimize) 0
    set QWIKMD::mdProtInfo($prefix,smd) 0
    set QWIKMD::mdProtInfo($prefix,smdk) 7.0
    set QWIKMD::mdProtInfo($prefix,ramp) 0
    set QWIKMD::mdProtInfo($prefix,timestep) 2
    set QWIKMD::mdProtInfo($prefix,vdw) 1
    set QWIKMD::mdProtInfo($prefix,electro) 2
    set QWIKMD::mdProtInfo($prefix,cutoff) 12.0
    set QWIKMD::mdProtInfo($prefix,pairlist) 14.0
    set QWIKMD::mdProtInfo($prefix,switch) 10.0
    set QWIKMD::mdProtInfo($prefix,switching) 1
    set QWIKMD::mdProtInfo($prefix,gbis) 0
    set QWIKMD::mdProtInfo($prefix,alphacut) 14.0
    set QWIKMD::mdProtInfo($prefix,solvDie) 80.0
    set QWIKMD::mdProtInfo($prefix,sasa) 0
    set QWIKMD::mdProtInfo($prefix,rampList) [list]
    set QWIKMD::mdProtInfo($prefix,ensemble) [lindex $args 3]
    set QWIKMD::mdProtInfo($prefix,run) 0
    set QWIKMD::mdProtInfo($prefix,thermostat) 0
    set QWIKMD::mdProtInfo($prefix,barostat) 0
    set templatefile [open ${template} r]
    set QWIKMD::line [read $templatefile]
    set QWIKMD::line [split $QWIKMD::line "\n"]
    close $templatefile

    set enter ""
    set index [lsearch -exact -all $QWIKMD::line $enter]
    for {set i 0} {$i < [llength $index]} {incr i} {
        lset QWIKMD::line [lindex $index $i] "{} {}"
    }


    set namdfile [open $outputfile w+]
        
    set temp [expr [lindex $args 4] + 273]
    set  QWIKMD::mdProtInfo($prefix,temp) [QWIKMD::format0Dec $temp]
    
    QWIKMD::replaceNAMDLine "structure" "structure [lindex $strct 0]"
    QWIKMD::replaceNAMDLine "coordinates" "coordinates [lindex $strct 1]"
    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)structure$]
    
    if {$index == -1} {
        puts $namdfile "structure [lindex $strct 0]"
    }

    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)coordinates$]
    if {$index == -1} {
        puts $namdfile "coordinates [lindex $strct 1]"
    }
    set tempaux 0
    if {$prefix == "Annealing"} {
        set tempaux 60
        set QWIKMD::mdProtInfo($prefix,ramp) 1

        set str "set nSteps"
        set index [lsearch -regexp $QWIKMD::line (?i)^$str]
        set val [lindex [lindex $QWIKMD::line $index] 2]
        set totalrun [QWIKMD::format2Dec [expr $val * [expr $temp - 60] *2e-6]]
        set QWIKMD::mdProtInfo($prefix,rampList) [list 60 $QWIKMD::mdProtInfo($prefix,temp) $totalrun]
    } elseif {$prefix != "Minimization"} {
        set tempaux $temp
    }
    if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 0} {
        if {$step > 0} {
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)temperature$]
            QWIKMD::replaceNAMDLine "temperature" "#[lindex $QWIKMD::line $index]"
        } else {
            QWIKMD::replaceNAMDLine "temperature" "temperature $tempaux"
        }
    
        if {$QWIKMD::advGui(solvent,0) == "Explicit"} {

            set indexcutt [lsearch -regexp -index 0 $QWIKMD::line (?i)^cutoff$]
            if {$indexcutt == -1} {
                puts $namdfile "cutoff 12.0"
                puts $namdfile "pairlistdist 14.0"
                puts $namdfile "switching on"
                puts $namdfile "switchdist 10.0"
                set QWIKMD::mdProtInfo($prefix,cutoff) 12.0
                set QWIKMD::mdProtInfo($prefix,pairlist) 14.0
                set QWIKMD::mdProtInfo($prefix,switch) 10.0
                set QWIKMD::mdProtInfo($prefix,switching) 1
            } else {
                set QWIKMD::mdProtInfo($prefix,cutoff) [lindex [lindex $QWIKMD::line $indexcutt] 1]

                set indexcutt [lsearch -regexp -index 0 $QWIKMD::line (?i)^pairlistdist$]
                if {$indexcutt != -1} {
                    set QWIKMD::mdProtInfo($prefix,pairlist) [lindex [lindex $QWIKMD::line $indexcutt] 1]
                }

                set indexcutt [lsearch -regexp -index 0 $QWIKMD::line (?i)^switchdist$]
                if {$indexcutt != -1} {
                    set QWIKMD::mdProtInfo($prefix,switch) [lindex [lindex $QWIKMD::line $indexcutt] 1]
                }

            }
            
            QWIKMD::replaceNAMDLine "gbis"  "gbis off"
            set QWIKMD::mdProtInfo($prefix,gbis) 0

            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)PME$]
            if { $index == -1} {
                puts $namdfile  "PME on"
                set QWIKMD::mdProtInfo($prefix,PME) 1
            } else {
                if {[lrange [lindex $QWIKMD::line $index] 1 end] == "on"} {
                    set QWIKMD::mdProtInfo($prefix,PME) 1
                } else {
                    set QWIKMD::mdProtInfo($prefix,PME) 0
                }
                
            } 
            
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)PMEGridspacing$]
            if {$index == -1} {
                puts $namdfile  "PMEGridspacing 1"
            } 
            
            if {$QWIKMD::membraneFrame != "" && $step < 2} {
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)margin$]
                if { $index == -1} {
                    puts $namdfile  "margin 2.5"
                } 
            }

            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)wrapAll$]
            if { $index == -1} {
                puts $namdfile  "wrapAll on"
            } 
            
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)wrapWater$]
            if { $index == -1} {
                puts $namdfile  "wrapWater on"
            } 
            
            # QWIKMD::replaceNAMDLine "dielectric" "dielectric 1"

            if {$QWIKMD::membraneFrame != ""} {
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)useflexiblecell$]
                if { $index == -1} {
                    puts $namdfile  "useflexiblecell yes"
                } else {
                    QWIKMD::replaceNAMDLine "useflexiblecell"  "useflexiblecell yes"
                }

                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)useConstantRatio$]
                if { $index == -1} {
                    puts $namdfile  "useConstantRatio yes"
                } else {
                    QWIKMD::replaceNAMDLine "useConstantRatio"  "useConstantRatio yes"
                }
            }
        } else {
            set indexcutt [lsearch -regexp -index 0 $QWIKMD::line (?i)^cutoff$]
            
            if {$indexcutt == -1} {
                puts $namdfile "cutoff 16"
                puts $namdfile "pairlistdist 18.0"
                puts $namdfile "switching on"
                puts $namdfile "switchdist 15"
                set QWIKMD::mdProtInfo($prefix,cutoff) 16.0
                set QWIKMD::mdProtInfo($prefix,pairlist) 18.0
                set QWIKMD::mdProtInfo($prefix,switch) 15.0
                set QWIKMD::mdProtInfo($prefix,switching) 1
            } else {
                set QWIKMD::mdProtInfo($prefix,cutoff) [lindex [lindex $QWIKMD::line $indexcutt] 1]

                set indexcutt [lsearch -regexp -index 0 $QWIKMD::line (?i)^pairlistdist$]
                if {$indexcutt != -1} {
                    set QWIKMD::mdProtInfo($prefix,pairlist) [lindex [lindex $QWIKMD::line $indexcutt] 1]
                }

                set indexcutt [lsearch -regexp -index 0 $QWIKMD::line (?i)^switchdist$]
                if {$indexcutt != -1} {
                    set QWIKMD::mdProtInfo($prefix,switch) [lindex [lindex $QWIKMD::line $indexcutt] 1]
                }
            }
            
            if {$QWIKMD::advGui(solvent,0) == "Implicit"} {

                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^gbis$]
                if { $index != -1} {
                    QWIKMD::replaceNAMDLine "gbis" "gbis on"
                } else {
                    puts $namdfile  "gbis on"
                }
                set QWIKMD::mdProtInfo($prefix,gbis) 1
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^alphaCutoff$]
                if { $index != -1} {
                    QWIKMD::replaceNAMDLine "alphaCutoff" "alphaCutoff 14.0"
                } else {
                    puts $namdfile  "alphaCutoff 14.0"
                }
                set QWIKMD::mdProtInfo($prefix,alphacut) 14.0

                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^solventDielectric$]
                if { $index != -1} {
                    QWIKMD::replaceNAMDLine "solventDielectric" "solventDielectric 80.0"
                } else {
                    puts $namdfile  "solventDielectric 80.0"
                }
                set QWIKMD::mdProtInfo($prefix,solvDie) 80.0

                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^ionConcentration$]
                if { $index != -1} {
                    QWIKMD::replaceNAMDLine "ionConcentration" "ionConcentration $QWIKMD::basicGui(saltconc,0)"
                } else {
                    puts $namdfile  "ionConcentration $QWIKMD::basicGui(saltconc,0)"
                }
                set QWIKMD::mdProtInfo($prefix,sasa) 1
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^sasa$]
                if { $index != -1} {
                     QWIKMD::replaceNAMDLine "sasa" "sasa on"
                } else {
                    puts $namdfile  "sasa on"
                }
                
            } else {

                QWIKMD::replaceNAMDLine "gbis" "gbis off"
                
                QWIKMD::replaceNAMDLine "sasa" "sasa off"

                set QWIKMD::mdProtInfo($prefix,sasa) 0
                # if {$QWIKMD::advGui(solvent,0) == "Vacuum"} {
                #     set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^dielectric$]
                #     if { $index != -1} {
                #         QWIKMD::replaceNAMDLine "dielectric" "dielectric 80"
                #     } else {
                #         puts $namdfile  "dielectric 80"
                #     }
                #     set QWIKMD::mdProtInfo($prefix,diel) 80.0
                # } else {
                #     set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^Dielectric$]
                #     if { $index != -1} {
                #         QWIKMD::replaceNAMDLine "dielectric" "dielectric 1.0"
                #     }
                #     set QWIKMD::mdProtInfo($prefix,diel) 1.0
                # }

            }

            QWIKMD::replaceNAMDLine "PME" "PME off"
            QWIKMD::replaceNAMDLine "PMEGridspacing" "#PMEGridspacing 1"
            set QWIKMD::mdProtInfo($prefix,PME) 0
            
        }

        set str "set Temp"
        set index [lsearch -regexp $QWIKMD::line (?i)^$str]

        if {$index != -1} {
            lset QWIKMD::line $index "set Temp $temp"
        }
        
        set str "set nSteps"
        set index [lsearch -regexp $QWIKMD::line (?i)^$str]

        if {$index != -1} {
            set nsteps [expr [lindex $args 1] / [expr $temp - 60] ]
            lset QWIKMD::line $index "set nSteps  [expr int($nsteps + [expr fmod($nsteps,20)])]"
        }
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^minimize$]
        set QWIKMD::mdProtInfo($prefix,minimize) 0
        if {$index != -1} {
            set QWIKMD::mdProtInfo($prefix,minimize) [lindex $args 1]
            QWIKMD::replaceNAMDLine "minimize" "minimize [lindex $args 1]"
        }
        
        QWIKMD::replaceNAMDLine "run" "run [lindex $args 1]"

        set freq $QWIKMD::smdfreq
        if {$QWIKMD::basicGui(live) == 0} {
            set freq [expr $QWIKMD::smdfreq * 10]
        }
        
        if {$freq > [lindex $args 1]} {
            set freq [lindex $args 1]
        }
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^outputenergies$]
        if {$index == -1} {
            puts $namdfile  "outputenergies $freq"
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^outputtiming$]
        if {$index == -1} {
            puts $namdfile  "outputtiming $freq"
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^outputpressure$]
        if {$index == -1} {

            puts $namdfile  "outputpressure $freq"
        }

        set freq $QWIKMD::dcdfreq
        if {$freq > [lindex $args 1]} {
            set freq [lindex $args 1]
        }
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^dcdfreq$]
        if {$index == -1} {
        
            puts $namdfile  "dcdfreq $freq"
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^XSTFreq$]
        if {$index == -1} {
        
            puts $namdfile  "XSTFreq $freq"
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)^restartfreq$]
        set minimize [lsearch -regexp -index 0 $QWIKMD::line (?i)^minimize$]
        if {$index == -1} {
        
            set restart $freq
            if {$minimize != -1} {
                set restart [lindex $args 1]
            }
            puts $namdfile  "restartfreq $restart"
        }
    } else {
        ####Values for the text log file
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)temperature$]
        if {$index != -1} {
            set  QWIKMD::mdProtInfo($prefix,temp) [lindex [lindex $QWIKMD::line $index] 1]
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)cutoff$]
        if {$index != -1} {
            set  QWIKMD::mdProtInfo($prefix,cutoff) [lindex [lindex $QWIKMD::line $index] 1]
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)pairlistdist$]
        if {$index != -1} {
            set  QWIKMD::mdProtInfo($prefix,pairlist) [lindex [lindex $QWIKMD::line $index] 1]
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)switchdist$]
        if {$index != -1} {
            set  QWIKMD::mdProtInfo($prefix,switch) [lindex [lindex $QWIKMD::line $index] 1]
        } 

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)alphaCutoff$]
        if {$index != -1} {
            set  QWIKMD::mdProtInfo($prefix,alphacut) [lindex [lindex $QWIKMD::line $index] 1]
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)solventDielectric$]
        if {$index != -1} {
            set QWIKMD::mdProtInfo($prefix,solvDie) [lindex [lindex $QWIKMD::line $index] 1]
        }

        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)sasa$]
        if {$index != -1} {
            set on [string trim [join [lindex [lindex $QWIKMD::line $index] 1]]]
            if {$on == "on"} {
                set  QWIKMD::mdProtInfo($prefix,sasa) 1
            } elseif {$on == "off"} {
                set  QWIKMD::mdProtInfo($prefix,sasa) 1
            }
        }
            
        # set index [lsearch -regexp -index 0 $QWIKMD::line (?i)dielectric$]
        # if {$index != -1} {
        #     set  QWIKMD::mdProtInfo($prefix,diel) [lindex [lindex $QWIKMD::line $index] 1]
        # }

        set QWIKMD::mdProtInfo($prefix,minimize) 0
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)minimize$]
        if {$index != -1} {
            set  QWIKMD::mdProtInfo($prefix,minimize) [lindex [lindex $QWIKMD::line $index] 1]
        }

        
    
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMD$]
        if {$index != -1} {
            set on [string trim [join [lindex [lindex $QWIKMD::line $index] 1]]]
            if {$on == "on"} {
                set  QWIKMD::mdProtInfo($prefix,smd) 1
            } elseif {$on == "off"} {
                set  QWIKMD::mdProtInfo($prefix,smd) 0
            }
        }


    }
    set QWIKMD::mdProtInfo($prefix,const) 0
    if {[lindex $args 2] != "none"} {
        #if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 0} {
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)constraints$]
            if { $index != -1} {
                QWIKMD::replaceNAMDLine "constraints" "constraints on"
                
            } else {
                puts $namdfile "constraints on"
            }
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)conskcol$]
            if { $index != -1} {
                QWIKMD::replaceNAMDLine "conskcol" "conskcol B"
                
            } else {
                puts $namdfile "conskcol B"
            }
        #}
        set restrains [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 2]
        set index [lsearch $restrains [lindex $args 2]]
        set tbline $step
        set reffile [lindex $QWIKMD::confFile $step]_restraints.pdb
        set constfile [lindex $QWIKMD::confFile $step]_restraints.pdb
        if {$index > -1} {
            if {$step > 0} {
                if {[lindex $restrains [expr $step -1]] == [lindex $restrains $step]} {
                    set stepaux $step
                    while {[lindex $restrains [expr $stepaux -1]] == [lindex $restrains $stepaux]} {
                        incr stepaux -1
                        if {$stepaux == 0} {
                            break
                        }
                    }
                    if {$stepaux >= 1} {
                        set constfile [lindex $QWIKMD::confFile $stepaux]_restraints.pdb
                        set reffile [lindex $QWIKMD::confFile [expr $stepaux -1] ].coor
                    } else {
                        set constfile [lindex $QWIKMD::confFile 0]_restraints.pdb
                        set reffile [lindex $QWIKMD::confFile 0]_restraints.pdb
                    }
                } else {
                    set reffile [lindex $QWIKMD::confFile [expr $step - 1] ].coor
                }
            } 
        } elseif {$step > 0} {
            set reffile [lindex $QWIKMD::confFile [expr $step - 1] ].coor
        }
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)consref$]
        if { $index != -1} {
            QWIKMD::replaceNAMDLine "consref" "consref $reffile"
            
        } else {
            puts $namdfile "consref $reffile"
        }
        
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)conskfile$]
        if { $index != -1} {
            QWIKMD::replaceNAMDLine "conskfile" "conskfile $constfile"
        } else {
            puts $namdfile "conskfile $constfile"
        }
        set QWIKMD::mdProtInfo($prefix,const) 1
        set QWIKMD::mdProtInfo($prefix,constsel) [lindex $args 2]
    } else {
        set QWIKMD::mdProtInfo($prefix,const) 0
        QWIKMD::replaceNAMDLine "constraints" "constraints off"
    }

    set indexswichting [lsearch -regexp -index 0 $QWIKMD::line (?i)switching$] 
    set on [string trim [join [lindex [lindex $QWIKMD::line $index] 1]]]
    if {$on == "on"} {
        set  QWIKMD::mdProtInfo($prefix,switching) 1
    } elseif {$on == "off"} {
        set  QWIKMD::mdProtInfo($prefix,switching) 0
    }

    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)timestep$]
    if {$index != -1} {
        set  QWIKMD::mdProtInfo($prefix,timestep) [lindex [lindex $QWIKMD::line $index] 1]
    }
    
    if {$QWIKMD::mdProtInfo($prefix,run) == 0 && $prefix != "Minimization"} {
        set sum [lindex $args 1]
        if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 1} {
            set index [lsearch -regexp -index 0 -all $QWIKMD::line (?i)run$]
            set sum 0
            foreach ind $index {
                set sum [expr $sum + [lindex [lindex $QWIKMD::line $index] 1]]
            }
        } 
        set QWIKMD::mdProtInfo($prefix,run) [QWIKMD::format2Dec [expr $sum * $QWIKMD::mdProtInfo($prefix,timestep) * 1e-6]]
    }
    


    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)fullElectFrequency$]
    if {$index != -1} {
        set  QWIKMD::mdProtInfo($prefix,electro) [lindex [lindex $QWIKMD::line $index] 1]
    }

    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)nonbondedfreq$]
    if {$index != -1} {
        set  QWIKMD::mdProtInfo($prefix,vdw) [lindex [lindex $QWIKMD::line $index] 1]
    }
    set QWIKMD::mdProtInfo($prefix,thermostat) 0
    if {[lsearch [lindex $args 3] *T*] > -1 } {
        

        set indexLangevin [lsearch -regexp -index 0 $QWIKMD::line (?i)langevin$]
        set indexAndersen [lsearch -regexp -index 0 $QWIKMD::line (?i)loweAndersen$]
        if { $indexLangevin == -1 && $indexAndersen == -1} {
            puts $namdfile  "langevin on"
            puts $namdfile  "langevintemp $tempaux"
            set QWIKMD::mdProtInfo($prefix,thermostat) Langevin

    
        } else {
            if {[string trim [lindex [lindex $QWIKMD::line $indexLangevin] 1]] == "on" } {
                set QWIKMD::mdProtInfo($prefix,thermostat) Langevin
            } elseif {[string trim [lindex [lindex $QWIKMD::line $indexAndersen] 1]] == "on"} {
                set QWIKMD::mdProtInfo($prefix,thermostat) LoweAndersen
            }

            QWIKMD::replaceNAMDLine "langevin" "langevin on"
            QWIKMD::replaceNAMDLine "loweAndersen" "loweAndersen on"


            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)langevintemp$]
            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 0} {
                QWIKMD::replaceNAMDLine "langevintemp" "langevintemp $tempaux"
                QWIKMD::replaceNAMDLine "loweAndersenTemp" "loweAndersenTemp $tempaux"
    
                set QWIKMD::mdProtInfo($prefix,temp) $tempaux
            } else {
                if {$indexLangevin != -1} {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)langevintemp$]
                    if {$index != -1} {
                        set QWIKMD::mdProtInfo($prefix,temp) [lindex [lindex $QWIKMD::line $index] 1]
                    }
                } else {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)loweAndersenTemp$]
                    if {$index != -1} {
                        set QWIKMD::mdProtInfo($prefix,temp) [lindex [lindex $QWIKMD::line $index] 1]
                    }
                }
            }  
        }
    } else {
        QWIKMD::replaceNAMDLine "langevin" "langevin off"
        QWIKMD::replaceNAMDLine "loweAndersen" "loweAndersen off"
    }
    set QWIKMD::mdProtInfo($prefix,barostat) 0
    if {[lsearch [lindex $args 3] *p*] > -1 } {
        set indexLangevin [lsearch -regexp -index 0 $QWIKMD::line (?i)langevinpiston$]
        set indexBerendsen [lsearch -regexp -index 0 $QWIKMD::line (?i)BerendsenPressure$]
        set press "[QWIKMD::format5Dec [expr [lindex $args 5] * 1.01325]]"
        if { $indexLangevin == -1 && $indexBerendsen == -1} {
            puts $namdfile  "langevinpiston on"
            puts $namdfile  "langevinpistontarget $press"
            puts $namdfile  "langevinpistontemp $tempaux"
            set QWIKMD::mdProtInfo($prefix,barostat) Langevin
            set QWIKMD::mdProtInfo($prefix,press) [lindex $args 5]

        } else {
            QWIKMD::replaceNAMDLine "langevinpiston" "langevinpiston on"
            QWIKMD::replaceNAMDLine "BerendsenPressure" "BerendsenPressure on"
            
            
            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 0} {
                if {[string trim [lindex [lindex $QWIKMD::line $indexLangevin] 1]] == "on" } {
                    set QWIKMD::mdProtInfo($prefix,barostat) Langevin
                } elseif {[string trim [lindex [lindex $QWIKMD::line $indexBerendsen] 1]] == "on"} {
                    set QWIKMD::mdProtInfo($prefix,barostat) Berendsen
                }
                set QWIKMD::mdProtInfo($prefix,press) [lindex $args 5]
                QWIKMD::replaceNAMDLine "langevinpistontarget" "langevinpistontarget $press"
                QWIKMD::replaceNAMDLine "BerendsenPressureTarget" "BerendsenPressureTarget $press"
            } else {
                if {$indexLangevin != -1} {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)langevinpistontarget$]
                    if {$index != -1} {
                        set QWIKMD::mdProtInfo($prefix,press) [QWIKMD::format2Dec [expr [lindex [lindex $QWIKMD::line $index] 1] / 1.01325 ]]
                    }
                } else {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)BerendsenPressureTarget$]
                    if {$index != -1} {
                        set QWIKMD::mdProtInfo($prefix,press) [QWIKMD::format2Dec [expr [lindex [lindex $QWIKMD::line $index] 1] / 1.01325 ]]
                    }
                }
            }
            
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)langevinpistontemp$]
            if { $index != -1} {
                set tempaux 0
                if {$prefix == "Annealing"} {
                    set tempaux 60
                    
                } else {
                    set tempaux $temp   
                }
                QWIKMD::replaceNAMDLine "langevinpistontemp" "langevinpistontemp $tempaux"
            }
        }
    } else {
        QWIKMD::replaceNAMDLine "langevinpiston" "langevinpiston off"
        QWIKMD::replaceNAMDLine "BerendsenPressure" "BerendsenPressure off"

        set str "set barostat"
        set index [lsearch -regexp $QWIKMD::line (?i)^$str]
        if {$prefix == "Annealing"} {
            if { $index != -1} {
            lset QWIKMD::line $index "set barostat 0"
            } else {
                puts $namdfile "set barostat 0"
            }
        }

    }


    if {$step > 0} {
        set inputname [lindex $QWIKMD::confFile [expr $step -1]]
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)#binCoordinates$]
        if {$index != -1} {
            QWIKMD::replaceNAMDLine "#binCoordinates" "binCoordinates $inputname.restart.coor"
        } else {
             set index [lsearch -regexp -index 0 $QWIKMD::line (?i)binCoordinates$]
             if {$index != -1} {
                QWIKMD::replaceNAMDLine "binCoordinates" "binCoordinates $inputname.restart.coor"
            } else {
                puts $auxfile "binCoordinates $inputname.restart.coor"
            }
        }
        
        QWIKMD::replaceNAMDLine "binVelocities" "binVelocities $inputname.restart.vel"
        
         if {$QWIKMD::advGui(solvent,0) == "Explicit"} {
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)#extendedSystem$]
            if { $index != -1} {
                QWIKMD::replaceNAMDLine "#extendedSystem" "extendedSystem $inputname.restart.xsc"
            } else {
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
                if {$index != -1} {
                    QWIKMD::replaceNAMDLine "extendedSystem" "extendedSystem $inputname.restart.xsc"
                } else {
                    puts $namdfile "extendedSystem $inputname.restart.xsc"
                }
            }
        } else {
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
            QWIKMD::replaceNAMDLine "extendedSystem" "#extendedSystem $inputname.restart.xsc"
        }

    } elseif {$step == 0} {
        if {$QWIKMD::advGui(solvent,0) == "Explicit"} {
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
            if { $index != -1} {
                QWIKMD::replaceNAMDLine "extendedSystem" "extendedSystem $prefix.xsc"
                
            } else {
                puts $namdfile "extendedSystem $prefix.xsc"
            }
        } else {
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)extendedSystem$]
            QWIKMD::replaceNAMDLine "extendedSystem" "#[lindex $QWIKMD::line $index]"
        }
        
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)binCoordinates$]
        QWIKMD::replaceNAMDLine "binCoordinates" "#[lindex $QWIKMD::line $index]"
        
        set index [lsearch -regexp -index 0 $QWIKMD::line (?i)binVelocities$]
        QWIKMD::replaceNAMDLine "binVelocities" "#[lindex $QWIKMD::line $index]"

    }
    
    if {$QWIKMD::run == "SMD"} {
        if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,smd) == 1} {
            set smdtxt "SMD"
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMD$]
            if {$index != -1} {
                QWIKMD::replaceNAMDLine "SMD" "SMD on"
                set  QWIKMD::mdProtInfo($prefix,smd) 1
            } else {
                puts $namdfile "SMD on"
            }

            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$step,lock) == 0} {
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMDVel$]
                set val [format %.3g [expr $QWIKMD::basicGui(pspeed) * 2e-6]]
                if {$index != -1} {
                    QWIKMD::replaceNAMDLine "SMDVel" "SMDVel $val"
                } else {
                    puts $namdfile "SMDVel $val"
                }
                set QWIKMD::mdProtInfo($prefix,pspeed) $QWIKMD::basicGui(pspeed)
            } else {
                set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMDVel$]
                if {$index != -1} {
                    set QWIKMD::basicGui(pspeed) [QWIKMD::format2Dec [expr [expr [string trim [join [lindex [lindex $QWIKMD::line $index] 1]]]  / $QWIKMD::mdProtInfo($prefix,timestep) ] * 1e6 ] ]
                    set QWIKMD::mdProtInfo($prefix,pspeed) $QWIKMD::basicGui(pspeed)
                }
            }
            
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMDk$]
            if {$index != -1} {
                set QWIKMD::mdProtInfo($prefix,smdk) [string trim [join [lindex [lindex $QWIKMD::line $index] 1]]]
            }

            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)consref$]
            set i 0
            while {$i < [llength  $QWIKMD::confFile]} {
                if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$i,smd) == 1} {
                    break
                }
                incr i
            }

            set smdfile [lindex $strct 1]
            if {$step > 0} {
                set smdfile [lindex $QWIKMD::confFile [expr $i -1] ].coor
            }

            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)SMDFile$]
            if {$index != -1} {
                QWIKMD::replaceNAMDLine "SMDFile" "SMDFile $smdfile"
            } else {
                puts $namdfile "SMDFile $smdfile"
            } 

            
            if {$index != -1} {
                QWIKMD::replaceNAMDLine "consref" "consref $smdfile"
            } else {
                puts $namdfile "consref $smdfile"
            }
                
            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)conskfile$]
            if {$index != -1} {
                QWIKMD::replaceNAMDLine "conskfile" "conskfile $smdfile"
            } else {
                puts $namdfile "conskfile $smdfile"
            }

            set index [lsearch -regexp -index 0 $QWIKMD::line (?i)constraints$]
            if {$index != -1} {
                QWIKMD::replaceNAMDLine "constraints" "constraints on"
            } else {
                puts $namdfile "constraints on"
            }

            set i 0
            
            while {$i < [llength  $QWIKMD::confFile]} {
                if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$i,smd) == 1} {
                    break
                }
                incr i
            }
            if {[file dirname $outputfile] == "${QWIKMD::outPath}/run"} {
                if {$i == $step} {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)firstTimestep$]
                    if { $index != -1} {
                        QWIKMD::replaceNAMDLine "firstTimestep" "firstTimestep 0"
                    } else {
                        puts $namdfile "firstTimestep 0"
                    }
                } else {
                    set index [lsearch -regexp -index 0 $QWIKMD::line (?i)firstTimestep$]
                    if {$index != -1} {
                        QWIKMD::replaceNAMDLine "firstTimestep" "[QWIKMD::addFirstTimeStep $step]"
                    } else {
                        puts $namdfile [QWIKMD::addFirstTimeStep $step]
                    }
                }
            }
        }
    }

    foreach parm $QWIKMD::ParameterList {
        set file [file tail ${parm}]
        set index [lsearch -regexp $QWIKMD::line "parameters $file"]
        if {$index == -1 && ([file extension $file] == ".str" || [file extension $file] == ".prm") } {
            puts $namdfile "parameters $file"
        }
    }

    QWIKMD::replaceNAMDLine "outputname" "outputname [lindex $QWIKMD::confFile $step]"
    QWIKMD::replaceNAMDLine "dcdfile" "dcdfile [lindex $QWIKMD::confFile $step].dcd"
    QWIKMD::replaceNAMDLine "restartname" "restartname [lindex $QWIKMD::confFile $step].restart"
    if {$QWIKMD::basicGui(live) == 1} {
        set index [lsearch -regexp $QWIKMD::line "IMDon on"]
        if {$index == -1} {
            puts $namdfile  "# IMD Parameters"
            puts $namdfile  "IMDon on   ;#"
            puts $namdfile  "IMDport    3000    ;# port number (enter it in VMD)"
            puts $namdfile  "IMDfreq    10  ;# send every 10 frame"
            puts $namdfile  "IMDwait    yes ;# wait for VMD to connect before running?\n"
        }    
    }
    lset QWIKMD::maxSteps $step [lindex $args 1]
    
    
    set enter "{} {}"
    set index [lsearch -exact -all $QWIKMD::line $enter]
    for {set i 0} {$i < [llength $index]} {incr i} {
        lset QWIKMD::line [lindex $index $i] ""
    }

    for {set i 0 } {$i < [llength $QWIKMD::line]} {incr i} {
        
        puts $namdfile [lindex $QWIKMD::line $i]
    }
    
    close $namdfile
    
    return $env(TMPDIR)/$prefix.conf
    
     
}

proc QWIKMD::selectProcs {} {
    

    set procWindow ".proc"
    set QWIKMD::numProcs [QWIKMD::procs] 

    if {[winfo exists $procWindow] != 1} {
        toplevel $procWindow
        wm protocol ".proc" WM_DELETE_WINDOW {
            
            set QWIKMD::numProcs "Cancel"
            destroy ".proc"

        }
        
        wm minsize $procWindow -1 -1
        wm resizable $procWindow 0 0

        grid columnconfigure $procWindow 0 -weight 1
        grid rowconfigure $procWindow 1 -weight 1
        ## Title of the windows
        wm title $procWindow  "How many processors?" ;# titulo da pagina
        set x [expr round([winfo screenwidth .]/2.0)]
        set y [expr round([winfo screenheight .]/2.0)]
        wm geometry $procWindow -$x-$y
        grid [ttk::frame $procWindow.f0] -row 0 -column 0 -sticky ew -padx 4 -pady 4

        grid [ttk::label $procWindow.f0.lb -text "QwikMD detected $QWIKMD::numProcs processors.\nHow many do you want to use?"] -row 0 -column 0 -sticky w -padx 2
        
        set values [list]
        for {set i $QWIKMD::numProcs} {$i >= 1} {incr i -1} {
            lappend values $i
        }
        
        grid [ttk::combobox $procWindow.f0.combProcs -values $values -width 4 -state normal -justify center -textvariable QWIKMD::numProcs] -row 0 -column 1 -pady 0 -padx 4
        $procWindow.f0.combProcs set $QWIKMD::numProcs
        

        grid [ttk::frame $procWindow.f1] -row 1 -column 0 -sticky ew

        grid [ttk::button $procWindow.f1.okBut -text "Ok" -padding "2 0 2 0" -width 15 -command {
            #set QWIKMD::numProcs [.proc.f0.combProcs get] 
            destroy ".proc"
            } ] -row 0 -column 0 -sticky ns

        grid [ttk::button $procWindow.f1.cancel -text "Cancel" -padding "2 0 2 0" -width 15 -command {
            set QWIKMD::numProcs "Cancel"
             destroy ".proc"
            } ] -row 0 -column 1 -sticky ns
        #raise $procWindow
    } else {
        wm deiconify $procWindow
    }
    tkwait window $procWindow

} 


proc QWIKMD::Run {} {

    QWIKMD::selectProcs
    update idletasks
    if {$QWIKMD::numProcs == "Cancel"} {
        return
    }
    cd $QWIKMD::outPath/run

    if {[$QWIKMD::topGui.nbinput index current] == 0 && $QWIKMD::state == [llength $QWIKMD::prevconfFile]} {
        set file ""
        if {$QWIKMD::run == "SMD"} {
            set file "qwikmd_production_smd_$QWIKMD::state"
            
        } else {
            set file "qwikmd_production_$QWIKMD::state"
        }
        if {[file exists ${file}.conf] != 1} {
            lappend QWIKMD::confFile $file
            lappend QWIKMD::prevconfFile $file
        }
    }
    if {[file exists [lindex $QWIKMD::prevconfFile $QWIKMD::state].conf ] != 1} {
        
        if {[$QWIKMD::topGui.nbinput index current] == 1} {
            if {$QWIKMD::state == [$QWIKMD::advGui(protocoltb,$QWIKMD::run) size]} {
                tk_messageBox -message "Please add protocol before press Start." -title "No Protocol" -icon warning -type ok
                return
            }
            lappend QWIKMD::prevconfFile [lindex $QWIKMD::confFile end]
            #lappend QWIKMD::confFile [lindex $QWIKMD::confFile end]
        }
        
        if {$QWIKMD::basicGui(live) == 1} {
            set QWIKMD::dcdfreq 1000
            set QWIKMD::load 0
        } else {
            set QWIKMD::dcdfreq 10000
        }
        set QWIKMD::confFile $QWIKMD::prevconfFile
        QWIKMD::NAMDGenerator [lindex [molinfo $QWIKMD::topMol get filename] 0] $QWIKMD::state
        QWIKMD::SaveInputFile $QWIKMD::basicGui(workdir,0)
        $QWIKMD::runbtt configure -text "Start [QWIKMD::RunText]"
    }
    set QWIKMD::confFile $QWIKMD::prevconfFile
    if {$QWIKMD::state > 0} {
        set prevcheck [lindex $QWIKMD::prevconfFile [expr $QWIKMD::state -1] ]
        set ret 0
        if {[file exists $prevcheck.check]} {
            set fil [open  $prevcheck.check r]
            set line [read -nonewline $fil]
            close $fil
            if {$line != "DONE"} {
                set ret 1
            }
        }  else {
            set ret 1
        }
        if {$ret == 1} {
            tk_messageBox -message "Previous simulation is still running or terminated with error" -title "Running Simulation" -icon info -type ok
            file delete -force -- [lindex $QWIKMD::prevconfFile [expr $QWIKMD::state -1]].check
            incr QWIKMD::state -1
            $QWIKMD::runbtt configure -text "Start [QWIKMD::RunText]"
            $QWIKMD::runbtt configure -state normal
            return
        }
    }


    set conf [lindex $QWIKMD::prevconfFile $QWIKMD::state]
    ################################################################################
    ## New version of namd2 (NAMD_CVS-2015-10-28_Linux-x86_64-multicore-CUDA) does not 
    ## return an error and crash if there is not enough patches per GPU (so we can use)
    ## the same command for both CUDA and
    ################################################################################
    set exec_command "namd2 +idlepoll +setcpuaffinity +p${QWIKMD::numProcs} $conf.conf"
    set do 0
    set i 0
    set tabid 0
    if {[$QWIKMD::topGui.nbinput tab 0 -state] == "normal"} {
        if {[string match "*_production_smd_*" [lindex $QWIKMD::prevconfFile $QWIKMD::state ] ] > 0 && $QWIKMD::state > 0} {
            set do 1
        }
            
        while {$i < [llength  $QWIKMD::prevconfFile]} {
            if {[string match "*_production_smd*" [lindex $QWIKMD::prevconfFile $i] ] > 0} {
                break
            }
            incr i
        }

    } else {
        set tabid 1
        if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$QWIKMD::state,smd) == 1 && $QWIKMD::state > 0} {
            set do 1
        }
        while {$i < [llength $QWIKMD::prevconfFile]} {
            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$i,smd) == 1} {
                break
            }
            incr i
        }
    }
    if {$i != $QWIKMD::state} {
        set do 0
    }
    set smdfile [lindex $QWIKMD::inputstrct 0]
    if {$QWIKMD::run == "SMD" && $QWIKMD::state > 0} {
        set smdfile [lindex $QWIKMD::inputstrct 0]
        if {$QWIKMD::state > 0} {
            if {$tabid == 0} {
                set index [lindex [lsearch -all $QWIKMD::prevconfFile "*_production_smd*"] 0]
                set smdfile [lindex $QWIKMD::prevconfFile [expr $index -1 ]].coor
            } else {
                set i 0
                while {$i < [llength  $QWIKMD::prevconfFile]} {
                    if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$i,smd) == 1} {
                        break
                    }
                    incr i
                }
                set smdfile [lindex $QWIKMD::prevconfFile [expr $i -1 ]].coor 
            }
        }
    }
    if {$QWIKMD::run == "SMD" && [file exists $smdfile] != 1 && $do ==1} {
        
        QWIKMD::save_viewpoint 1
        set stfile [lindex [molinfo $QWIKMD::topMol get filename] 0]
        set name [lindex $stfile 0]
        set mol [mol new $name]
        mol addfile [lindex $QWIKMD::prevconfFile [expr $QWIKMD::state -1]].restart.coor
        set all [atomselect top "all"]
        set beta [atomselect top $QWIKMD::anchorRessel]
        set occupancy [atomselect top $QWIKMD::pullingRessel]
        $all set beta 0
        $all set occupancy 0
        $beta set beta 1
        $occupancy set occupancy 1
        $all writepdb $smdfile
        mol delete $mol
        $all delete
        $beta delete
        $occupancy delete
        mol delete $mol 
        QWIKMD::restore_viewpoint 1
        display update ui
        update
    }
    if {$QWIKMD::run == "SMD" && [string match "*_production_smd_*" [lindex $QWIKMD::prevconfFile $QWIKMD::state ] ] > 0} {
        QWIKMD::checkAnchors
        if {[lindex $QWIKMD::timeXsmd end] != ""} {
            set QWIKMD::smdprevx [lindex $QWIKMD::timeXsmd end]
            set QWIKMD::smdprevindex 0 
        }
    }

    # if {[llength $QWIKMD::volpos] == 0} {
    #   set QWIKMD::volpos 0
    # }
    # if {[llength $QWIKMD::presspos] == 0} {
    #   set QWIKMD::presspos 0
    # }

    if {[lindex $QWIKMD::volpos end] > [lindex $QWIKMD::presspos end] && [llength $QWIKMD::volpos] != 0} {
        set QWIKMD::condprevx [lindex $QWIKMD::volpos end]
    } elseif {[llength $QWIKMD::presspos] != 0} {
        set QWIKMD::condprevx [lindex $QWIKMD::presspos end]
    }
    
    set QWIKMD::condprevindex 0 

    set tabid [$QWIKMD::topGui.nbinput index current]

    if {$QWIKMD::basicGui(live) == 1} {

        set QWIKMD::load 0
        set IMDPort 3000

        set QWIKMD::smdcurrentpos 0
        $QWIKMD::runbtt configure -state disabled
        
    
        $QWIKMD::basicGui(preparebtt,$tabid)  configure -state normal
        
        set  QWIKMD::basicGui(mdPrec,0) 0
        set tabid [expr [$QWIKMD::topGui.nbinput index current] +1]
        $QWIKMD::basicGui(mdPrec,$tabid) configure -maximum [lindex $QWIKMD::maxSteps $QWIKMD::state]

        eval ::ExecTool::exec "${exec_command} >> $conf.log & "
        
        QWIKMD::connect localhost $IMDPort

        set logfile [open [lindex $QWIKMD::prevconfFile $QWIKMD::state].log r]
        while {[eof $logfile] != 1 } {
            set line [gets $logfile]

            if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                set QWIKMD::timestep [lindex $line 2]
            }

            if {[lindex $line 0] == "Info:" && [join [lrange $line 1 3]] == "INTERACTIVE MD FREQ" } {
                set QWIKMD::imdFreq [lindex $line 4]
                break
            }
        }
        close $logfile

        incr QWIKMD::state
        
        set QWIKMD::load 0
    } else {
        set QWIKMD::smdcurrentpos 0
        set answer [tk_messageBox -message "You are about to run an MD simulation in the background making the VMD windows of the current session unavailable until the simulation is finished.\nDo you want to proceed?" -title "Run Simulation" -icon question -type yesno ]
        if {$answer == "yes"} {
            $QWIKMD::runbtt configure -state disabled
    
            $QWIKMD::basicGui(preparebtt,$tabid)  configure -state normal
            eval ::ExecTool::exec $exec_command  >> $conf.log
            incr QWIKMD::state 
            set QWIKMD::load 0
            QWIKMD::updateMD
            
        }
        
    }
}

proc QWIKMD::connect {IMDHost IMDPort} {

    set attempt_delay 5000   ;# delay between attepmts to connect in ms
    set attempt_timeout 100000 ;# timeout in ms
    set solvent $QWIKMD::basicGui(solvent,0)
    set tabid [$QWIKMD::topGui.nbinput index current] 
    if {$tabid== 1} {
        set solvent $QWIKMD::advGui(solvent,0)
    }
    if {$solvent == "Explicit"} {
        set attempt_delay 10000   ;# delay between attepmts to connect in ms
        set attempt_timeout 200000 ;# timeout in ms
        pbc box -off
    }
    mol top $QWIKMD::topMol
    update idletasks
    set timecounter 0
    after $attempt_delay
    while {$timecounter <= $attempt_timeout} { 
        if ![catch { imd connect $IMDHost $IMDPort }] {
            
            imd keep 0

            trace variable ::vmd_timestep($QWIKMD::topMol) w ::QWIKMD::updateMD 
            #imd keep 0
            break
        } else {
            # else give NAMD more time
            after $attempt_delay
        }
        incr timecounter $attempt_delay
    }
    if {$timecounter > $attempt_timeout} {
        tk_messageBox -message "The simulation failed to start.\nPlease check VMD terminal window or the *.log files in the \"run\" output folder for errors." -icon error -type ok
    }
}

################################################################################
## Main proc that defines mutations, changes on protonation state, renaming and change
## residues type. If in the QWIKMD::rowSelection call UpdateRes with opt ==1 (meaning
## do the changes), global variables like QWIKMD::protonate, QWIKMD::protindex
## stores the indexes (ResID_Chain)
## QWIKMD::protres stores the different states of the Resname cells in Select Residue table
##  QWIKMD::rename stores the residues indexes that need to be rename
##  QWIKMD::renameindex stores the indexes (ResID_Chain) that will be rename during the PrepareStructures
##  QWIKMD::dorename([lindex $QWIKMD::renameindex $i]) stores the new resname for that index (ResID_Chain)
## mutations, protonation changes and type changes follows the same logit described
################################################################################

proc QWIKMD::CallUpdateRes {tbl row col text} {
    set domaintable [QWIKMD::UpdateRes $tbl $row $col $text]
    if {$domaintable != 0} {
        if {$QWIKMD::tablemode == "type"} {
            #$tbl cancelediting
            $tbl rejectinput
            QWIKMD::UpdateMolTypes [expr [$QWIKMD::topGui.nbinput index current] + 1 ]
            
        }
        if {$QWIKMD::tablemode == "rename"} {
            
            set resid [$tbl cellcget $row,0 -text]
            set chain [$tbl cellcget $row,2 -text]
            set toposearch ""
            if {[info exists QWIKMD::dorename(${resid}_$chain)] == 1 } {
                set toposearch [QWIKMD::checkAtomNames $QWIKMD::dorename(${resid}_$chain) $QWIKMD::TopList]
            } else {
                set toposearch [QWIKMD::checkAtomNames $domaintable $QWIKMD::TopList]
                $tbl cellconfigure $row,$col -text $domaintable 
            }
            
            if {[lindex $toposearch 0] != -1} {
                set sel [atomselect $QWIKMD::topMol "resid \"$resid\" and chain \"$chain\" and noh"]
                set atomnames [$sel get name]
                foreach name $atomnames {
                    if {[lsearch -index 0 [lindex $toposearch 1] $name] == -1 && [llength $atomnames] > 1} {
                        set answer [tk_messageBox -message "Atoms' name from residue $resid in the original structure don't match the CHARMM topologies. Please rename the atom's names." \
                        -title "Atom's Name" -icon info -type yesno]
                        if {$answer == "yes"} {
                            set QWIKMD::tablemode "edit"
                            QWIKMD::tableModeProc
                            $QWIKMD::selresTable selection set $row
                            QWIKMD::editAtomGuiProc
                            QWIKMD::editAtom
                            raise $QWIKMD::editATMSGui
                            return $domaintable 
                        } else {
                            return $domaintable 
                        }
                    }
                }
                $sel delete
            } else {
                tk_messageBox -message "No topologies found for the residue $text. Please add the topologies for this residue or rename it to the correct name."\
                -title "No topologies" -icon info -type okcancel]
                return $domaintable 
            }
        }
    } else {
        set domaintable [$tbl cellcget $row,$col -text] 
    }
    return $domaintable   

}
proc QWIKMD::UpdateMolTypes {tabid} {
    QWIKMD::mainTable $tabid
    QWIKMD::SelResid
    if {[llength $QWIKMD::delete] > 0} {
        set chaincol [$QWIKMD::selresTable getcolumns 2]
        set rescol [$QWIKMD::selresTable getcolumns 0]
        set tbindex [list]
        foreach delres $QWIKMD::delete {
            set residchain [split $delres "_"]
            set resindex [lsearch -all $rescol [lindex $residchain 0] ]
            foreach resind $resindex {
                if {[lindex $chaincol $resind] == [lindex $residchain end]} {
                    lappend tbindex $resind
                }
            }
        }
        set QWIKMD::delete [list]
        set auxmode $QWIKMD::tablemode
        set QWIKMD::tablemode "delete"
        $QWIKMD::selresTable selection set $tbindex
        QWIKMD::Apply
        set QWIKMD::tablemode $auxmode
    }
}

proc QWIKMD::UpdateRes {tbl row col text} {
    set resname $QWIKMD::prevRes
    set domutate 0
    set recolor 0
    set totindex ""
    set delrep 0

    set resid [$tbl cellcget $row,0 -text]
    set chain [$tbl cellcget $row,2 -text]
    set type [$tbl cellcget $row,3 -text]

    set sel [atomselect top "resid \"$resid\" and chain \"$chain\""]
    set initresid [lindex [$sel get resname] 0]
    set returntext ""

    set ind ${resid}_$chain
    
    if {$QWIKMD::tablemode == "prot"} {
        if {$text != $initresid && $QWIKMD::protres(0,3) != "" && $initresid != "HIS" && $initresid != "HSD" && $initresid != "HSE" && $initresid != "HSP"} {
            
            set returntext "$initresid -> $QWIKMD::protres(0,3) -> $text"
            set recolor 1
            set domutate 1
        } elseif {($text != $initresid && $QWIKMD::protres(0,3) == "" ) || $initresid == "HIS" || $initresid == "HSD" || $initresid == "HSE" || $initresid == "HSP"} {
            set returntext "$initresid -> $text"
            set recolor 1
            set domutate 1
        } elseif {$text == $initresid} {
            set returntext $text

            set index [lsearch -exact $QWIKMD::protindex $ind]
            if {$index != -1} {
                set QWIKMD::protindex [lreplace $QWIKMD::protindex $index $index]
                set domutate 0
            }
        }
    } elseif {$QWIKMD::tablemode == "mutate"} {
        if {$text != $initresid} {
            set returntext "$initresid -> $text"
            set recolor 1
            set domutate 1
        } else {
            set returntext $text
            set index [lsearch -exact $QWIKMD::mutindex $ind]
            #set domutate 1
            if {$index != -1} {
                set QWIKMD::mutindex [lreplace $QWIKMD::mutindex $index $index]
                set domutate 0
                set recolor 0
            }
        }
    } elseif {$QWIKMD::tablemode == "rename" || $QWIKMD::tablemode == "type"} {

        set returntext $text
        if {$text == $initresid} {

            set index [lsearch -exact $QWIKMD::renameindex $ind]
            if {$index != -1} {
                set QWIKMD::renameindex [lreplace $QWIKMD::renameindex $index $index]
                set domutate 0
                set recolor 0
            }
        } else {
            if {$QWIKMD::tablemode == "type" && $text == "protein"} {
                set answer [tk_messageBox -message "Are you sure that [$tbl cellcget $row,1 -text] is a protein residue?" -title "Residues Type" -icon info -type yesno]
                if {$answer == "no"} {
                    $tbl cancelediting
                } 
            }
            set colresname [list] 
            set colresname [concat $colresname [$tbl getcolumns 1] ]

            set resname $QWIKMD::protres(0,2)
            if {$QWIKMD::tablemode == "type"} {
                set resname [$tbl cellcget $row,1 -text]
                set totindex [lsearch -all -exact $colresname $resname]
            } else {
                set totindex [concat $totindex [lsearch -all -exact $colresname $resname]]
            }
            if {[llength $totindex] == 0} {
                set resname [$tbl cellcget $row,1 -text]
                if {$QWIKMD::tablemode == "type"} {
                    set totindex [lsearch -all -exact $colresname $resname]
                } else {
                    set totindex [concat $totindex [lsearch -all -exact $colresname $resname]]
                }

                if {[llength $totindex] == 0} {
                    set totindex $row
                }
            }
            set domutate 1
            set QWIKMD::resallnametype 1
            if {[llength $totindex] > 1} {
                if {$QWIKMD::tablemode == "type"} {
                    set msgtext "The type of one or more residues can be changed based on the chosen residue type.\nDo you want to change all?"
                    set title "Change Residues Type"
                } else {
                    set msgtext "One or more residues can be rename based on the chosen residue name.\nDo you want to rename all?"
                    set title "Rename Residues"
                }
                set answer [tk_messageBox -message $msgtext -title $title -icon question -type yesnocancel]
                
                if {$answer == "no"} {
                    set QWIKMD::resallnametype 0
                    set totindex $row
                } elseif {$answer == "cancel"} {
                    set domutate 0
                    return 0
                } 
            }
        }
    } 
    $sel delete
    $QWIKMD::selresTable selection clear 0 end
    QWIKMD::rowSelection

    set rowcolor [$tbl rowcget $row -background]
    
    if {$recolor == 1} {
        $QWIKMD::selresTable cellconfigure $row,1 -background #ffe1bb
    } else {
        $QWIKMD::selresTable cellconfigure $row,1 -background white
    }
    
    $tbl configure -labelcommand tablelist::sortByColumn

    
    set sel [atomselect $QWIKMD::topMol "resid \"$resid\" and chain \"$chain\""]
    set structure [$sel get structure]
    set hexcols [QWIKMD::chooseColor [lindex $structure 0] ]
            
    set hexred [lindex $hexcols 0]
    set hexgreen [lindex $hexcols 1]
    set hexblue [lindex $hexcols 2]

    if {$type == "protein" && $domutate ==1} {
        $tbl rowconfigure $row -background white -selectbackground cyan
        $QWIKMD::selresTable cellconfigure $row,3 -background "#${hexred}${hexgreen}${hexblue}" -selectbackground "#${hexred}${hexgreen}${hexblue}"
    
    } elseif {$type != "protein" && $domutate == 1} {
        $tbl rowconfigure $row -background $rowcolor -selectbackground cyan
        $tbl cellconfigure $row,3 -background white -selectbackground cyan
    }

    if {$QWIKMD::tablemode == "prot" && $type == "protein" && $domutate == 1} {
        if {[lsearch -exact $QWIKMD::protindex ${resid}_${chain}] == -1} {
            lappend QWIKMD::protindex ${resid}_${chain}
        }
        set QWIKMD::protonate(${resid}_${chain}) "$QWIKMD::protres(0,2) $text"


    } elseif {$QWIKMD::tablemode == "mutate" && $domutate == 1} {
    
        set index [lsearch -exact $QWIKMD::mutindex ${resid}_${chain}]
        if {$index == -1} {
            lappend QWIKMD::mutindex ${resid}_${chain}
            if {[llength $QWIKMD::mutindex] > 3} {
                tk_messageBox -message "You are mutating more than 3 residues and possibly inducing structural instability." -title "Mutations" -icon warning -type ok
            }
        }
        set index [lsearch -exact $QWIKMD::protindex ${resid}_${chain}]

        if {$index != -1 } {
            
            set QWIKMD::protindex [lreplace $QWIKMD::protindex $index $index]
        }

        set QWIKMD::mutate(${resid}_${chain}) "$initresid $text"

    } elseif {$QWIKMD::tablemode == "rename" && $domutate == 1} {

        set colresname [$tbl getcolumns 1]
        for {set i 0} {$i < [llength $totindex]} {incr i} {
        
            $tbl cellconfigure [lindex $totindex $i],3 -background white -selectbackground cyan
            set toresname ""
            set txt "or \(resid \"[$tbl cellcget [lindex $totindex $i],0 -text]\" and chain \"[$tbl cellcget [lindex $totindex $i],2 -text]\"\)"

            if {$type == "hetero"} {
                set index [lsearch $QWIKMD::heteronames $text]
                set toresname [lindex $QWIKMD::hetero $index]
                QWIKMD::checkMacros hetero $txt ""
                append QWIKMD::heteromcr " $txt"
            } elseif {$type == "nucleic"} {
                set toresname $text
                QWIKMD::checkMacros nucleic $txt ""
                append QWIKMD::nucleicmcr " $txt"
            } elseif {$type == "lipid"} {
                set toresname $text
                QWIKMD::checkMacros lipid $txt ""
                append QWIKMD::lipidmcr " $txt"
            } elseif {$type == "glycan"} {
                set index [lsearch $QWIKMD::carbnames $text]
                set toresname [lindex $QWIKMD::carb $index]
                QWIKMD::checkMacros glycan $txt ""
                append QWIKMD::glycanmcr " $txt"
            } elseif {$type == "protein"} {
                set toresname $text
            } 
            if {[lsearch -index 0 $QWIKMD::userMacros $type] > -1 && $toresname == ""} {
                set macro [lindex $QWIKMD::userMacros [lsearch -index 0 $QWIKMD::userMacros $type]]
                set toresname [lindex [lindex $macro 1] [lsearch [lindex $macro 2] $text]]
            }

            atomselect macro qwikmd_protein $QWIKMD::proteinmcr
            atomselect macro qwikmd_nucleic $QWIKMD::nucleicmcr
            atomselect macro qwikmd_glycan $QWIKMD::glycanmcr
            atomselect macro qwikmd_lipid $QWIKMD::lipidmcr
            atomselect macro qwikmd_hetero $QWIKMD::heteromcr
            set str "[$tbl cellcget [lindex $totindex $i],0 -text]_[$tbl cellcget [lindex $totindex $i],2 -text]"
            set index [lsearch -exact $QWIKMD::rename $str]
            if { $index != -1} {
                set QWIKMD::rename [lreplace $QWIKMD::rename $index $index]
            }

            set index [lsearch -exact $QWIKMD::renameindex $str]
            if { $index == -1} {
                lappend QWIKMD::renameindex $str
            }
            set QWIKMD::dorename($str) $toresname
            $tbl cellconfigure [lindex $totindex $i],1 -text $text
            $tbl rowconfigure [lindex $totindex $i] -background white -selectbackground cyan
            
            if {$type == "protein"} {
                $tbl cellconfigure [lindex $totindex $i],3 -background "#${hexred}${hexgreen}${hexblue}" -selectbackground "#${hexred}${hexgreen}${hexblue}"
    
            }
        }
    } elseif {$QWIKMD::tablemode == "type" && $domutate == 1} {
        for {set i 0} {$i < [llength $totindex]} {incr i} {
            set resindex "[$tbl cellcget [lindex $totindex $i],0 -text]_[$tbl cellcget [lindex $totindex $i],2 -text]"
            set renameind [lsearch -exact $QWIKMD::renameindex $resindex]
            if {$renameind != -1} {
                set QWIKMD::renameindex [lreplace $QWIKMD::renameindex $renameind $renameind]
                array unset QWIKMD::dorename $resindex
            }
            set toresname ""
            set txt "and not \(resid \"[$tbl cellcget [lindex $totindex $i],0 -text]\" and chain \"[$tbl cellcget [lindex $totindex $i],2 -text]\"\)"
            set txt2 "or \(resid \"[$tbl cellcget [lindex $totindex $i],0 -text]\" and chain \"[$tbl cellcget [lindex $totindex $i],2 -text]\"\)"
            QWIKMD::editMacros $QWIKMD::protres(0,2) $txt $txt2 old
            QWIKMD::editMacros $text $txt $txt2 new
            set type ""
            update idletasks    
        }
    }

    $sel delete
    array unset QWIKMD::protres
    
    set QWIKMD::selected 1
    if {$delrep == 1} {
        while {[molinfo $QWIKMD::topMol get numreps] !=  [expr $QWIKMD::repidin + $QWIKMD::aprep]} {
            mol delrep [expr [molinfo $QWIKMD::topMol get numreps] -1 ] $QWIKMD::topMol
        }
    }
    return $returntext
}

################################################################################
## Modify the macros by adding the resid the newmacro and removing from the oldmacro
## opt defines if this the operation to remove from the old or add to the new macro
################################################################################
proc QWIKMD::editMacros {macro removetxt addtxt opt} {
    QWIKMD::checkMacros $macro $removetxt $addtxt
    set seltext ""
    if {$opt == "old"} {
        set seltext $removetxt
    } else {
        set seltext $addtxt
    }
    switch -exact $macro {
        protein {
            append QWIKMD::proteinmcr " $seltext"
        }
        nucleic {
            append QWIKMD::nucleicmcr " $seltext"
        }
        glycan {
            append QWIKMD::glycanmcr " $seltext"
        }
        lipid {
            append QWIKMD::lipidmcr " $seltext"
        }
        hetero {
            append QWIKMD::heteromcr " $seltext"
        }
        default {
            atomselect macro $macro "[atomselect macro $macro] $seltext"
        }
    }
    if {$opt == "new"} {
        atomselect macro qwikmd_protein $QWIKMD::proteinmcr
        atomselect macro qwikmd_nucleic $QWIKMD::nucleicmcr
        atomselect macro qwikmd_glycan $QWIKMD::glycanmcr
        atomselect macro qwikmd_lipid $QWIKMD::lipidmcr
        atomselect macro qwikmd_hetero $QWIKMD::heteromcr
    }
    
}
################################################################################
## check for duplicated definitions on macros
################################################################################
proc QWIKMD::checkMacros {macro txt txt2} {

    proc replaceString {str1 str2} {
        return [string replace $str1 [string first $str2 $str1] [expr [string first $str2 $str1] + [string length $str2] ]] 
    }

    switch -exact $macro {
        protein {
            if {[string first $txt $QWIKMD::proteinmcr] > -1} {
                set QWIKMD::proteinmcr [[namespace current]::replaceString $QWIKMD::proteinmcr $txt]
            } 

            if {[string first $txt2 $QWIKMD::nucleicmcr] > -1} {
                set QWIKMD::proteinmcr [[namespace current]::replaceString $QWIKMD::proteinmcr $txt2]
            }
        }
        nucleic {
            if {[string first $txt $QWIKMD::nucleicmcr] > -1} {
                set QWIKMD::nucleicmcr [[namespace current]::replaceString $QWIKMD::nucleicmcr $txt]
            } 

            if {[string first $txt2 $QWIKMD::nucleicmcr] > -1} {
                set QWIKMD::nucleicmcr [[namespace current]::replaceString $QWIKMD::nucleicmcr $txt2]
            }
        }
        glycan {
            if {[string first $txt $QWIKMD::glycanmcr] > -1} {
                set QWIKMD::glycanmcr [[namespace current]::replaceString $QWIKMD::glycanmcr $txt]
            } 

            if {[string first $txt2 $QWIKMD::glycanmcr] > -1} {
                set QWIKMD::glycanmcr [[namespace current]::replaceString $QWIKMD::glycanmcr $txt2]
            }
        }
        lipid {
            if {[string first $txt $QWIKMD::lipidmcr] > -1} {
                set QWIKMD::lipidmcr [[namespace current]::replaceString $QWIKMD::lipidmcr $txt]
            } 

            if {[string first $txt2 $QWIKMD::lipidmcr] > -1} {
                set QWIKMD::lipidmcr [[namespace current]::replaceString $QWIKMD::lipidmcr $txt2]
            }
        }
        hetero {
            if {[string first $txt $QWIKMD::heteromcr] > -1} {
                set QWIKMD::heteromcr [[namespace current]::replaceString $QWIKMD::heteromcr $txt]
            } 

            if {[string first $txt2 $QWIKMD::heteromcr] > -1} {
                set QWIKMD::heteromcr [[namespace current]::replaceString $QWIKMD::heteromcr $txt2]
            }
        }
        default {
            set current [atomselect macro $macro]
            if {[string first $txt $current] > -1} {
                set current [[namespace current]::replaceString $current $txt]
            } 

            if {[string first $txt2 $current] > -1} {
                set current [[namespace current]::replaceString $current $txt2]
            }
            atomselect macro $macro $current

        }   
    }   
} 

################################################################################
## Syncr proc to match the chains selected in "Select chain/type" and the main table
################################################################################                
proc QWIKMD::reviewTable {tabid} {
    set table $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb
    $table delete 0 end
    set length [expr [llength [array names QWIKMD::chains]] /3]
    set index 0

    for {set i 0} {$i < $length} {incr i} {
        if {$QWIKMD::chains($i,0) == 0} {
            if {[info exists QWIKMD::index_cmb($QWIKMD::chains($i,1),4)] == 1} {
                mol showrep $QWIKMD::topMol [QWIKMD::getrepnum $QWIKMD::index_cmb($QWIKMD::chains($i,1),4)] off
                set previndex [$table getcolumns 0]
                set indexfind [lsearch $previndex $QWIKMD::chains($i,1)]
                $table delete $indexfind
            }
        } elseif {$QWIKMD::chains($i,0) == 1 } {
            update 
            $table insert end [list [lindex $QWIKMD::chains($i,1) 0] $QWIKMD::chains($i,2) [lindex $QWIKMD::chains($i,1) 2] {} {} ]
            if {[info exists QWIKMD::index_cmb($QWIKMD::chains($i,1),4)] != 1} {
                mol addrep $QWIKMD::topMol
                set QWIKMD::index_cmb($QWIKMD::chains($i,1),4) [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]

                $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb cellconfigure $index,3 -text [QWIKMD::mainTableCombosStart 0 $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb $index 3 "aux"]
                $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb cellconfigure $index,4 -text [QWIKMD::mainTableCombosStart 0 $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb $index 4 "aux"]
            } 
            mol showrep $QWIKMD::topMol [QWIKMD::getrepnum $QWIKMD::index_cmb($QWIKMD::chains($i,1),4)] on
            set QWIKMD::index_cmb($QWIKMD::chains($i,1),3) $index
            QWIKMD::mainTableCombosEnd $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb $index 3 $QWIKMD::index_cmb($QWIKMD::chains($i,1),1)
            QWIKMD::mainTableCombosEnd $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb $index 4 $QWIKMD::index_cmb($QWIKMD::chains($i,1),2)
            $table cellconfigure $index,3 -text $QWIKMD::index_cmb($QWIKMD::chains($i,1),1)
            $table cellconfigure $index,4 -text $QWIKMD::index_cmb($QWIKMD::chains($i,1),2)

            incr index
        }
        

    }
    
    $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb finishediting

}

################################################################################
## update analysis in live simulaiton. This proc is called everytime a frame is received
## from namd through IMD
################################################################################
proc QWIKMD::updateMD {args} {
    
    if {$QWIKMD::basicGui(live) == 1} {
        incr QWIKMD::counterts

        if {$QWIKMD::hbondsGui != ""} {
            if {[expr [expr $QWIKMD::counterts - $QWIKMD::prevcounterts] % $QWIKMD::calcfreq] == 0} {
                QWIKMD::HbondsCalc  
            }   
        }

        if {$QWIKMD::tempGui != "" || $QWIKMD::pressGui != "" || $QWIKMD::volGui != "" } {
            if {[expr [expr $QWIKMD::counterts -  $QWIKMD::prevcounterts] % $QWIKMD::calcfreq] == 0} {
                QWIKMD::CondCalc
            }
        }

        if {$QWIKMD::energyTotGui != "" || $QWIKMD::energyPotGui != "" || $QWIKMD::energyKineGui != "" \
        || $QWIKMD::energyBondGui != "" || $QWIKMD::energyAngleGui != "" || $QWIKMD::energyDehidralGui != "" || $QWIKMD::energyVdwGui != ""} {
            if {[expr [expr $QWIKMD::counterts -  $QWIKMD::prevcounterts] % $QWIKMD::calcfreq] == 0} {
                QWIKMD::EneCalc
            }   
        }
        set do 0
        if {[string match "*smd*" [lindex  $QWIKMD::confFile [expr $QWIKMD::state -1 ] ] ] > 0} {
            set do 1
        }
        if {[info exists QWIKMD::advGui(protocoltb,$QWIKMD::run,[expr $QWIKMD::state -1 ],smd)]} {
            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,[expr $QWIKMD::state -1 ],smd) == 1} {
                set do 1
            }
        }
        if {$do == 1} {
            incr QWIKMD::countertssmd
        }

        if {$QWIKMD::smdGui != ""} {
            if {[expr [expr $QWIKMD::countertssmd - $QWIKMD::prevcountertsmd] % $QWIKMD::calcfreq] == 0} {
                QWIKMD::SmdCalc 
            }
        }

        if {$QWIKMD::rmsdGui != ""} {
            if {[expr [expr $QWIKMD::counterts -  $QWIKMD::prevcounterts] % $QWIKMD::calcfreq] == 0} {
                QWIKMD::RmsdCalc
            }
        }
        incr QWIKMD::basicGui(mdPrec,0) $QWIKMD::imdFreq
        set tabid 0
        if {[$QWIKMD::topGui.nbinput tab 0 -state] == "disabled"} {
            set tabid 1
        }
        if {[expr $QWIKMD::counterts - $QWIKMD::prevcounterts] == 1} {
            if {($tabid == 0 && $QWIKMD::basicGui(solvent,0) == "Explicit") || ($tabid == 1 && $QWIKMD::advGui(solvent,0) == "Explicit") } {
                update idletasks
                #set pbcinfo ""
                #set do [catch {pbc get -first 0 -last 0 -molid $QWIKMD::topMol} pbcinfo]
                #if {$do == 0} {
                    catch {pbc set $QWIKMD::pbcInfo}
                    catch {pbc box -center bb -color yellow -width 4}
                #}
                
            }
        }   
        QWIKMD::updateTime live
    }

    
    if {[file exists "[lindex $QWIKMD::confFile [expr $QWIKMD::state -1]].check"] == 1} {
        set line ""
        
        while {$line == ""} {
            set fil [open [lindex $QWIKMD::confFile [expr $QWIKMD::state -1] ].check r]
            set line [read -nonewline $fil]
            close $fil
        } 
        if {$QWIKMD::basicGui(live) == 1} {
            trace vdelete ::vmd_timestep($QWIKMD::topMol) w ::QWIKMD::updateMD 
                 
        }
        set tabid 0
        if {[$QWIKMD::topGui.nbinput tab 0 -state] == "disabled"} {
            set tabid 1
        }
        $QWIKMD::topGui.nbinput select $tabid 
        if {$line != "DONE" } {
            incr QWIKMD::state -1
            tk_messageBox -message "One or more files failed to be written. The new simulation ready to run is [lindex $QWIKMD::prevconfFile [expr $QWIKMD::state -1]]" -title "Running Simulation" -icon warning -type ok
            $QWIKMD::runbtt configure -state normal
            $QWIKMD::runbtt configure -text "Start [QWIKMD::RunText]"
            $QWIKMD::basicGui(preparebtt,$tabid) configure -state normal
            
        } else {
            tk_messageBox -message "The molecular Dynamics simulation [lindex $QWIKMD::prevconfFile [expr $QWIKMD::state -1] ] finished. Please press Run button to continue" -title "Running Simulation" -icon info -type ok
            $QWIKMD::runbtt configure -text "Start [QWIKMD::RunText]"
            $QWIKMD::runbtt configure -state normal
            $QWIKMD::basicGui(preparebtt,$tabid)  configure -state normal
        }
        set QWIKMD::prevcounterts $QWIKMD::counterts
        if {$QWIKMD::run == "SMD"} {
            set do 0
            if {$tabid == 0} {
                if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,smd) == 1} {
                    set do 1
                }
            } else {
                if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) == 1} {
                    set do 1
                }
            }
            if {$do == 1} {
                set QWIKMD::prevcountertsmd $QWIKMD::countertssmd
            }  
        }
        
    }

}

################################################################################
## Proc called by the button Apply
## Used to validate the deletion and inclusion of residues in Select Residue window
## Also to validate the selection of the anchor and pulling residues in SMD simulations
## General residue selection using the table
################################################################################
proc QWIKMD::Apply {} {
    set table $QWIKMD::selresTable
    set lock 0
    set id [$table curselection]
    if {$id == ""} {
        return
    }
    if {$QWIKMD::anchorpulling != 1 && ($QWIKMD::tablemode == "add" || $QWIKMD::tablemode == "delete" || $QWIKMD::tablemode == "edit")} {
    
        if {$QWIKMD::tablemode == "delete"} {
            set prechain ""
            set chain ""
            set chainind 0
            set length [expr [llength [array names QWIKMD::chains]] /3]
            for {set i 0} {$i < [llength $id]} {incr i} {
                set row [lindex $id $i]
                $table rowconfigure $row -background white -foreground grey -selectbackground cyan -selectforeground grey
                $table cellconfigure $row,1 -editable false
                $table cellconfigure $row,3 -editable false
                set resid [$table cellcget $row,0 -text]
                set chain [$table cellcget $row,2 -text]
                set type [$table cellcget $row,3 -text]
                set resname [lindex [$table cellcget $row,1 -text] 0]
                set str "${resid}_${chain}"
                set index [lsearch -exact $QWIKMD::delete $str]
                set chaint "$chain and $type"
                if {$index == -1} {
                    lappend QWIKMD::delete $str
                    set index 0
                    if {$chaint == $prechain} {
                        for {set i 0} {$i < $length} {incr i} {
                            if {$QWIKMD::chains($i,1) == $chaint} {
                                set chainind $i
                                break
                            }
                        }
                        set prechain $chaint
                        if {$type == "protein" || $type == "nucleic" || $type == "hetero" || $type == "lipid" || $type == "glycan"} {
                            set QWIKMD::index_cmb($chaint,5) "chain \"$chain\" and qwikmd_${type}"
                        } else {
                            set QWIKMD::index_cmb($chaint,5) "chain \"$chain\" and $type"
                        }
                        
                    } 
                    if {$type == "protein" || $type == "nucleic" || $type == "hetero" || $type == "lipid" || $type == "glycan"} {
                        append QWIKMD::index_cmb($chaint,5) " and not (resid \"[$table cellcget $row,0 -text]\" and chain \"$chain\" and qwikmd_${type})"
                    } else {
                        append QWIKMD::index_cmb($chaint,5) " and not (resid \"[$table cellcget $row,0 -text]\" and chain \"$chain\" and $type)"
                    }
                }

            }

            for {set i 0} {$i < $length} {incr i} {
                if {$QWIKMD::chains($i,0) == 1} {
                    mol modselect $i $QWIKMD::topMol [lindex $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)]
                }
                
            }

        } elseif {$QWIKMD::tablemode == "add"} {
            
            for {set i 0} {$i < [llength $id]} {incr i} {
                set row [lindex $id $i]
                
                set resid [$table cellcget $row,0 -text]
                set chain [$table cellcget $row,2 -text]
                set type [$table cellcget $row,3 -text]
                set str "${resid}_${chain}"
                set index [lsearch -exact $QWIKMD::delete $str]
                set chaint "$chain and $type"
                if { $index != -1} {
                    set QWIKMD::delete [lreplace $QWIKMD::delete $index $index]
                    if {$type == "protein" || $type == "nucleic" || $type == "hetero" || $type == "lipid" || $type == "glycan"} {
                        set first [string first " and not (resid \"$resid\" and chain \"$chain\" and qwikmd_${type})" $QWIKMD::index_cmb($chaint,5)]
                        set length [string length " and not (resid \"$resid\" and chain \"$chain\" and qwikmd_${type})"]
                    } else {
                        set first [string first " and not (resid \"$resid\" and chain \"$chain\" and $type)" $QWIKMD::index_cmb($chaint,5)]
                        set length [string length " and not (resid \"$resid\" and chain \"$chain\" and $type)"]
                    }
                    set QWIKMD::index_cmb($chaint,5) "[string range $QWIKMD::index_cmb($chaint,5) 0 [expr $first -1]][string range $QWIKMD::index_cmb($chaint,5) [expr $first + $length] end]"
                    
                }

                set index [lsearch -exact $QWIKMD::rename $str]
                if { $index != -1} {
                    $table rowconfigure $row -background red -foreground black -selectbackground cyan -selectforeground black
                } else {
                    $table rowconfigure $row -background white -foreground black -selectbackground cyan -selectforeground black
                }
                $table cellconfigure $row,1 -editable true
                $table cellconfigure $row,3 -editable true
            }
            set text ""
            set length [expr [llength [array names QWIKMD::chains]] /3]
            for {set i 0} {$i < $length} {incr i} {
                mol modselect $i $QWIKMD::topMol $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)    
            }
            QWIKMD::SelResid
        } elseif {$QWIKMD::tablemode == "edit"} { 
            
            QWIKMD::editAtomGuiProc
            if {$id != ""} {
                QWIKMD::editAtom
            }
        }

        $table selection clear 0 end
        QWIKMD::rowSelection

    } elseif {$QWIKMD::run == "SMD" && $QWIKMD::anchorpulling == 1} {
        # if QWIKMD::buttanchor == 1 called from Anchoring Residues
        # if QWIKMD::buttanchor == 2 called from Pulling Residues
        
        set msg 0
        if {$QWIKMD::buttanchor == 1} {
            set QWIKMD::anchorRessel ""
            set QWIKMD::anchorRes ""
            for {set i 0} {$i < [llength $id]} {incr i} {
                set resid [$table cellcget [lindex  $id $i],0 -text]
                set chain [$table cellcget [lindex  $id $i],2 -text]
                if {[lsearch $QWIKMD::pullingRes ${resid}_$chain] == -1} {
                    lappend QWIKMD::anchorRes ${resid}_$chain
                    if {$i != 0} {
                        append QWIKMD::anchorRessel " or resid \"$resid\" and chain \"$chain\"" 
                    } else {
                        append QWIKMD::anchorRessel "resid \"$resid\" and chain \"$chain\"" 
                    }
                } else {
                    set msg 1
                }
            }

        } elseif {$QWIKMD::buttanchor == 2} {
            set QWIKMD::pullingRessel ""
            set QWIKMD::pullingRes ""
            for {set i 0} {$i < [llength $id]} {incr i} {
                set resid [$table cellcget [lindex  $id $i],0 -text]
                set chain [$table cellcget [lindex  $id $i],2 -text]
                if {[lsearch $QWIKMD::anchorRes ${resid}_$chain] == -1} {
                    lappend QWIKMD::pullingRes ${resid}_$chain
                    if {$i != 0} {
                        append QWIKMD::pullingRessel " or resid \"$resid\" and chain \"$chain\""
                    } else {
                        append QWIKMD::pullingRessel "resid \"$resid\" and chain \"$chain\""
                    } 
                } else {
                    set msg 1
                }
                
            }
        }
        QWIKMD::checkAnchors
        
        if {$msg == 1} {
            tk_messageBox -message "Anchor and pulling residues selections overlapped. Please review you selections" -title "Overlapping Selections" -icon info -type ok
        } else {
            set QWIKMD::anchorpulling 0
            set QWIKMD::buttanchor 0
            set lock 1
        }
        
    } else {
        set tabid [$QWIKMD::topGui.nbinput index current]
        if {$tabid == 1 && ([llength $QWIKMD::selResidSelRep] > 0 || [llength  $QWIKMD::resrepname] > 0)} {
            set cellindex [lrange [$QWIKMD::advGui(protocoltb,$QWIKMD::run) editinfo] 1 2]

            if {$QWIKMD::selResidSel == "Type Selection" || $QWIKMD::selResidSel == ""} {
                set QWIKMD::selResidSel "\(resid "
                set listChains [lsort -unique [$QWIKMD::selresTable getcolumns 2]]
                foreach chain $listChains {
                    set selRes [lsearch -index 0 -all $QWIKMD::resrepname "*_${chain}"]
                    foreach tbindex $selRes {
                        set resid [lindex [split [lindex [lindex $QWIKMD::resrepname $tbindex] 0] "_"] 0]
                        append QWIKMD::selResidSel "\"$resid\" "
                    }
                    if {[llength $selRes] > 0} {
                        append QWIKMD::selResidSel "and chain \"$chain\"\) or "
                    }
                    
                }
                if {$QWIKMD::selResidSel != ""} {
                    set QWIKMD::selResidSel [string trimright $QWIKMD::selResidSel "or "]
                }
            }
            [$QWIKMD::advGui(protocoltb,$QWIKMD::run) editwinpath] set $QWIKMD::selResidSel
            $QWIKMD::advGui(protocoltb,$QWIKMD::run) finishediting
            set QWIKMD::selResidSel ""
            set QWIKMD::selResidSelIndex [list]
            set lock 1
        }
    }
    if {$lock == 1} {
        QWIKMD::lockSelResid 1
        QWIKMD::tableModeProc
        wm withdraw $QWIKMD::selResGui 
        trace remove variable ::vmd_pick_event write QWIKMD::ResidueSelect
        mouse mode rotate
    }
    QWIKMD::SelResClearSelection
}
################################################################################
## validate simulation time and temperature
## opt 1 -- command called by changes in max length (SMD simulations)
## opt 2 -- command called by changes in pulling speed (SMD simulations)
## opt 3 -- command called by changes in simulaiton time (SMD simulations)
################################################################################
proc QWIKMD::reviewLenVelTime {opt} {
    if {$opt == 1 || $opt == 2} {
        set val [expr [expr $QWIKMD::basicGui(plength) * 1.0] / {$QWIKMD::basicGui(pspeed) * 1.0} ]
        set point [string first $val "."]
        set decimal [string length [string range $val $point end]]
        if {$decimal > 6} {
            set val [QWIKMD::format5Dec $val]
        }
        set QWIKMD::basicGui(mdtime,1) $val
    } elseif {$opt == 3} {
        set val [QWIKMD::format0Dec [expr $QWIKMD::basicGui(mdtime,1) / 2e-6]]
        set mod [expr fmod($val,20)]
        if { $mod != 0.0} { 
            set QWIKMD::basicGui(mdtime,1) [QWIKMD::format5Dec [expr [expr $val + {20 - $mod}] * 2e-6 ] ]
            
        } 
        set val [expr [expr $QWIKMD::basicGui(plength) * 1.0] / { $QWIKMD::basicGui(mdtime,1) * 1.0 }]
        set point [string first $val "."]
        set decimal [string length [string range $val $point end]]
        if {$decimal > 6} {
            set val [QWIKMD::format5Dec $val]
        }
        set QWIKMD::basicGui(pspeed) $val
    }
}

################################################################################
## check if the anchor and pulling residues are represented in the OpenGL VMD window 
################################################################################
proc QWIKMD::checkAnchors {} {
    
    if {$QWIKMD::anchorRessel != "" && $QWIKMD::showanchor == 1} {
        if {$QWIKMD::anchorrepname == ""} {
            mol representation "VDW 1.0 12.0"
            mol addrep $QWIKMD::topMol
            set QWIKMD::anchorrepname [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
            mol modcolor [QWIKMD::getrepnum $QWIKMD::anchorrepname] $QWIKMD::topMol "ColorID 2"
        }
        mol modselect [QWIKMD::getrepnum $QWIKMD::anchorrepname] $QWIKMD::topMol $QWIKMD::anchorRessel
    } elseif {$QWIKMD::anchorrepname != "" && $QWIKMD::showanchor == 0} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::anchorrepname] $QWIKMD::topMol
        set QWIKMD::anchorrepname ""
    }

    if {$QWIKMD::pullingRessel != "" && $QWIKMD::showpull == 1} {
        if {$QWIKMD::pullingrepname == ""} {
            mol representation "VDW 1.0 12.0"
            mol addrep $QWIKMD::topMol
            set QWIKMD::pullingrepname [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
            mol modcolor [QWIKMD::getrepnum $QWIKMD::pullingrepname] $QWIKMD::topMol "ColorID 10"
        }
        mol modselect [QWIKMD::getrepnum $QWIKMD::pullingrepname] $QWIKMD::topMol $QWIKMD::pullingRessel
    } elseif {$QWIKMD::pullingrepname != "" && $QWIKMD::showpull == 0} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::pullingrepname] $QWIKMD::topMol
        set QWIKMD::pullingrepname ""
    }
}

################################################################################
## Bind proc when the table inside Select Residue window is selected
################################################################################
proc QWIKMD::rowSelection {} {
    set moltop $QWIKMD::topMol
    mol selection ""
    mol representation "Licorice"
    set table $QWIKMD::selresTable
    set id [$table curselection]
    
    for {set i 0} {$i < [llength $QWIKMD::residtbprev]} {incr i} {
        if {[lsearch $id [lindex $QWIKMD::residtbprev $i]] == -1} {
            set resid [$table cellcget [lindex $QWIKMD::residtbprev $i],0 -text]
            set chain [$table cellcget  [lindex $QWIKMD::residtbprev $i],2 -text]
            set index [lsearch -index 0 $QWIKMD::resrepname "${resid}_$chain"]
            mol delrep [QWIKMD::getrepnum [lindex [lindex $QWIKMD::resrepname $index] 1] ] $QWIKMD::topMol
            set QWIKMD::resrepname [lreplace $QWIKMD::resrepname $index $index]
        }
    }
    update idletasks    

    if {$id != ""} {
        set resid [$table cellcget [lindex $id 0],0 -text]
        set resname [$table cellcget [lindex $id 0],1 -text]
        set chain [$table cellcget [lindex $id 0],2 -text]
        set type [$table cellcget [lindex $id 0],3 -text]
        # set brk 0
        if {($QWIKMD::tablemode == "type" || $QWIKMD::tablemode == "rename")} {
            
            if {$type != "water"} {
                if {$QWIKMD::tablemode == "mutate" || $QWIKMD::tablemode == "prot" || $QWIKMD::tablemode == "rename"} {
                    $table columnconfigur 3  -editable false
                    $table columnconfigure 1 -editable true
                    $table editcell [lindex $id 0],1
                } elseif {$QWIKMD::tablemode == "type"} {
                    
                    $table columnconfigur 3  -editable true
                    $table columnconfigure 1 -editable false
                    $table editcell [lindex $id 0],3
                }
            }
            # if {$brk == 1} {
            #   $QWIKMD::selresTable cancelediting
            #   $QWIKMD::selresTable columnconfigure 1 -editable false 
            #   $QWIKMD::selresTable columnconfigure 3 -editable false 
            # }
            
        } 

        for {set i 0} {$i < [llength $id]} {incr i} {
            set index [lindex $id $i]
        
            set resname [lindex [split [$table cellcget $index,1 -text] "->"] 0]
            set resid [$table cellcget $index,0 -text]
            set chain [$table cellcget $index,2 -text]
            set type [$table cellcget $index,3 -text]
            if {[lsearch -index 0 $QWIKMD::resrepname "${resid}_$chain"] == -1} {
                set st "resid \"$resid\" and chain \"$chain\""
                set repr ""
                if {$type == "protein" || $type == "nucleic" || $type == "glycan" || $type == "lipid" } {
                    set repr "Licorice"
                } elseif {$type == "hetero"} {
                    set repr "VDW 1.0 12.0"
                } elseif {$type == "water"} {
                    if {$chain == "W"} {
                        set repr "Points"
                    } else {
                        set repr "VDW 1.0 12.0"
                    }
                } else {
                    set repr "VDW 1.0 12.0"
                }
                mol representation $repr
                mol addrep $moltop
                lappend QWIKMD::resrepname [list ${resid}_$chain [mol repname $moltop [expr [molinfo $QWIKMD::topMol get numreps] -1] ]]
                mol modselect [QWIKMD::getrepnum [lindex [lindex $QWIKMD::resrepname end] 1]] $moltop $st
                mol modcolor [QWIKMD::getrepnum [lindex [lindex $QWIKMD::resrepname end] 1]] $moltop "Name"
            }
        }
    } else {
        for {set i 0} {$i < [llength $QWIKMD::resrepname]} {incr i} {
            mol delrep [QWIKMD::getrepnum [lindex [lindex $QWIKMD::resrepname $i] 1]] $QWIKMD::topMol
        }
        set QWIKMD::resrepname [list]
    }
    set QWIKMD::residtbprev $id
}
proc QWIKMD::RenderProc {} {
    set extension ".tga"
    set types [list]

    global tcl_platform env
    set types {
        {{TGA}       {".tga"}        }
    }
    set win 0
    if {[string first "Windows" $::tcl_platform(os)] != -1} {
        lappend types [list {BMP} {".bmp"}]
        set extension ".bmp"
        set win 1
    }
    
    set fil [tk_getSaveFile -title "Save Image" -filetypes $types -defaultextension $extension]

    if {$fil != ""} {
        set dimensions [display get size]
        set renderMode ""
        switch $QWIKMD::advGui(render,rendertype) {
            "Capture Display" {
                set renderMode "snapshot"
                render $renderMode $fil
            }
            "Background Render" {
                set renderMode "Tachyon"
                set scenefilename $fil.dat
                set archexe ""
                
                if {$win == 1} {
                    set archexe ".exe"
                }
                set tachyonexe [format "tachyon%s" $archexe];
                set tachyoncmd [::ExecTool::find -interactive -description "Tachyon Ray Tracer" -path [file join $env(VMDDIR) "tachyon_[vmdinfo arch]$archexe"] $tachyonexe]
                if {$tachyoncmd == {}} {
                    puts "Cannot find Tachyon"
                } else {
                    set command "$scenefilename"
                    
                    switch [file extension ${fil}] {
                        ".bmp" {
                            append command " -format BMP -o ${fil}"
                        }
                        ".tga" {
                            append command " -format Targa -o ${fil}"
                        }
                    }
                    render $renderMode $scenefilename [concat [format "\"%s\"" $tachyoncmd] $command]
                }
            }
            "Interactive Render" {
                set renderMode "TachyonLOptiXInteractive"
                render $renderMode ${fil}
            }
        }
    }

}
proc QWIKMD::save_viewpoint {view_num} {
   global viewpoints
   foreach mol [molinfo list] {
      set viewpoints($QWIKMD::topMol,0) [molinfo $mol get rotate_matrix]
      set viewpoints($QWIKMD::topMol,1) [molinfo $mol get center_matrix]
      set viewpoints($QWIKMD::topMol,2) [molinfo $mol get scale_matrix]
      set viewpoints($QWIKMD::topMol,3) [molinfo $mol get global_matrix]
   }
}

proc QWIKMD::restore_viewpoint {view_num} {
   global viewpoints
   foreach mol [molinfo list] {
      if [info exists viewpoints($QWIKMD::topMol,0)] {
        molinfo $mol set center_matrix $viewpoints($QWIKMD::topMol,1)
        molinfo $mol set rotate_matrix $viewpoints($QWIKMD::topMol,0)
        molinfo $mol set scale_matrix $viewpoints($QWIKMD::topMol,2)
        molinfo $mol set global_matrix $viewpoints($QWIKMD::topMol,3)
        
      }
   }
}

proc QWIKMD::SaveInputFile {file} {
    if {[file exists $file] == 1} {
        file delete -force -- $file
    }
    set ofile [open $file w+]

    puts $ofile [string repeat "#" 20]
    puts $ofile "#\t\t QwikMD Input File"
    puts $ofile [string repeat "#\n" 10]
    puts $ofile [string repeat "#" 20]
    
    puts $ofile "set QWIKMD::nucleicmcr \{$QWIKMD::nucleicmcr\}"
    puts $ofile "set QWIKMD::proteinmcr \{$QWIKMD::proteinmcr\}"
    puts $ofile "set QWIKMD::heteromcr \{$QWIKMD::heteromcr\}"
    puts $ofile "set QWIKMD::glycanmcr \{$QWIKMD::glycanmcr\}"
    puts $ofile "set QWIKMD::lipidmcr \{$QWIKMD::lipidmcr\}"
    puts $ofile "atomselect macro qwikmd_protein \$QWIKMD::proteinmcr"
    puts $ofile "atomselect macro qwikmd_nucleic \$QWIKMD::nucleicmcr"
    puts $ofile "atomselect macro qwikmd_glycan \$QWIKMD::glycanmcr"
    puts $ofile "atomselect macro qwikmd_lipid \$QWIKMD::lipidmcr"
    puts $ofile "atomselect macro qwikmd_hetero \$QWIKMD::heteromcr"
    set tabid [$QWIKMD::topGui.nbinput index current]
    puts $ofile "\$QWIKMD::topGui.nbinput select $tabid"
    puts $ofile "set QWIKMD::prepared $QWIKMD::prepared"
    puts $ofile "QWIKMD::changeMainTab"
    incr tabid
    puts $ofile "\$QWIKMD::topGui.nbinput.f${tabid}.nb select [$QWIKMD::topGui.nbinput.f${tabid}.nb index current]"
    puts $ofile "QWIKMD::ChangeMdSmd ${tabid}"
    
    if {$QWIKMD::prepared == 0} {
        puts $ofile "set QWIKMD::inputstrct $QWIKMD::inputstrct"
        puts $ofile "QWIKMD::LoadButt \$QWIKMD::inputstrct"
    } else {
        puts $ofile "set aux \"\[file rootname \$QWIKMD::basicGui(workdir,0)\]\""
        puts $ofile "set QWIKMD::outPath $\{aux\}"
        puts $ofile "cd $\{QWIKMD::outPath\}/run/"
        puts $ofile "set QWIKMD::inputstrct \{$QWIKMD::inputstrct\}"

        puts $ofile "QWIKMD::LoadButt {[lindex $QWIKMD::inputstrct 0] [lindex $QWIKMD::inputstrct 1]}"
    }
    set arrayaux ""
    set values [array get QWIKMD::basicGui]
    for {set j 1} {$j < [llength $values]} {incr j 2} {
        set find [regexp {.qwikmd*} [join [lindex $values $j]]]
        if {$find == 0 } {
            append arrayaux "[lrange $values [expr $j -1] $j] "
        }
    }
    puts $ofile "array set QWIKMD::basicGui \{$arrayaux\}"
    set values [array get QWIKMD::advGui]
    for {set j 1} {$j < [llength $values]} {incr j 2} {
        set find [regexp {.qwikmd*} [join [lindex $values $j]]]
        if {$find == 0 } {
            append arrayaux "[lrange $values [expr $j -1] $j] "
        }
    }
    puts $ofile "array set QWIKMD::advGui \{$arrayaux\}"
    if {$QWIKMD::prepared == 1} {
        puts $ofile "array set QWIKMD::chains \{[array get QWIKMD::chains]\}"
        puts $ofile "array set QWIKMD::index_cmb \{[array get QWIKMD::index_cmb] \}"
        puts $ofile "set QWIKMD::delete \{$QWIKMD::delete\}"
    }

    puts $ofile "array set QWIKMD::mutate \{[array get QWIKMD::mutate]\}"
    puts $ofile "array set QWIKMD::protonate \{[array get QWIKMD::protonate] \}"

    puts $ofile "set QWIKMD::mutindex \{$QWIKMD::mutindex\}"
    puts $ofile "set QWIKMD::protindex \{$QWIKMD::protindex\}"
    puts $ofile "set QWIKMD::renameindex \{$QWIKMD::renameindex\}"
    puts $ofile "array set QWIKMD::dorename \{[array get QWIKMD::dorename]\}"
    puts $ofile "set QWIKMD::patchestr \{$QWIKMD::patchestr\}"
    
    if {$QWIKMD::prepared == 0} {
        puts $ofile "array set QWIKMD::chains \{[array get QWIKMD::chains]\}"
        puts $ofile "array set QWIKMD::index_cmb \{[array get QWIKMD::index_cmb] \}"
        puts $ofile "set QWIKMD::delete \{$QWIKMD::delete\}"
    }

    #if {$QWIKMD::prepared == 1} {
        puts $ofile "set QWIKMD::state 0"

        if {$QWIKMD::prepared == 1} {
            puts $ofile "set QWIKMD::load 1"
        } else {
            set QWIKMD::prevconfFile $QWIKMD::confFile
        }
        puts $ofile "set QWIKMD::prevconfFile \{$QWIKMD::prevconfFile\}"
        puts $ofile "set QWIKMD::confFile \$QWIKMD::prevconfFile"
        
        if {$tabid == 1} {
            set solvent $QWIKMD::basicGui(solvent,0)
        } else {
            set solvent $QWIKMD::advGui(solvent,0)
            if {$QWIKMD::run != "MDFF"} {
                set lines [list]
                for {set i 0} {$i < [llength $QWIKMD::prevconfFile]} {incr i} {
                    lappend lines [$QWIKMD::advGui(protocoltb,$QWIKMD::run) rowcget $i -text]
                }
                puts $ofile "set prtclLines \{$lines\}"
                puts $ofile "for \{set i 0\} \{\$i < \[llength \$prtclLines\]\} \{incr i\} \{"
                puts $ofile "\t\$QWIKMD::advGui(protocoltb,\$QWIKMD::run) insert end \[lindex \$prtclLines \$i\]"
                if {$QWIKMD::prepared == 1} {
                    puts $ofile "\tif \{\[file exists \[lindex \[lindex \$prtclLines \$i\] 0\].dcd\] == 1\} \{"
                    puts $ofile "\t\tincr QWIKMD::state"
                    puts $ofile "\t\}"
                }
                puts $ofile "\}"
            }
        }
        if {$solvent == "Explicit" && $QWIKMD::prepared == 1} {
            puts $ofile "pbc box -center bb -color yellow -width 4"
            puts $ofile "set QWIKMD::pbcInfo \[pbc get -first 0 -last 0\]"
        } 
    #}
    
    if {$QWIKMD::run == "SMD"} {
        #puts $ofile "set QWIKMD::smd $QWIKMD::smd"
        if {$QWIKMD::anchorRes != ""} {
            puts $ofile "set QWIKMD::anchorRes \{$QWIKMD::anchorRes\}"
            puts $ofile "set QWIKMD::anchorRessel \{$QWIKMD::anchorRessel\}"
        }
        if {$QWIKMD::pullingRes != ""} {
            puts $ofile "set QWIKMD::pullingRes \{$QWIKMD::pullingRes\}"
            puts $ofile "set QWIKMD::pullingRessel \{$QWIKMD::pullingRessel\}"
        }
        puts $ofile "QWIKMD::checkAnchors"
        
    }

    if {$QWIKMD::run == "MDFF"} {
        puts $ofile "\$QWIKMD::advGui(protocoltb,MDFF) delete 0 end"
        set line [list]
        #make sure that the line is saved as a list of lists
        lappend line [$QWIKMD::advGui(protocoltb,$QWIKMD::run) rowcget 0 -text] 
        puts $ofile "set prtclLines \{$line\}"
        puts $ofile "\$QWIKMD::advGui(protocoltb,MDFF) insert end \[lindex \$prtclLines end\]"
        puts $ofile "set QWIKMD::advGui(mdff,min) $QWIKMD::advGui(mdff,min)"
        puts $ofile "set QWIKMD::advGui(mdff,mdff) $QWIKMD::advGui(mdff,mdff)"
    }
    puts $ofile "set QWIKMD::basicGui(live) $QWIKMD::basicGui(live)"
    if {$QWIKMD::basicGui(live) == 0} {
        puts $ofile "set QWIKMD::dcdfreq [expr $QWIKMD::dcdfreq * 10]"
        puts $ofile "set QWIKMD::smdfreq $QWIKMD::smdfreq"
    } 
    puts $ofile "set QWIKMD::maxSteps \{$QWIKMD::maxSteps\}"
    close $ofile
}

proc QWIKMD::ResidueSelect {args} {
    global vmd_pick_atom
      if {[winfo exists $QWIKMD::selResGui] && $QWIKMD::topMol != "" && $vmd_pick_atom != ""} {
         set table $QWIKMD::selresTable
         
        set atom [atomselect $QWIKMD::topMol "index $vmd_pick_atom"]
        set chain [lindex [$atom get chain] 0]  
        set resid [lindex [$atom get resid] 0] 
        set str "${resid}_$chain"

        set chaincol [$table getcolumns 2]
        set rescol [$table getcolumns 0]
        set i 0
        
        while { $i < [llength $rescol]} {
         set res [lindex $rescol $i]
            set index "${res}_[lindex $chaincol $i]"
            

            if {$index == $str} {
                
                set sel [$table curselection]
                
                set ind [lsearch $sel $i]
                
                if {$ind != -1} {
                    set sel [lreplace $sel $ind $ind]
                    $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butClear invoke
                } else {
                    lappend sel $i
                }
                
                if {$sel != ""} {

                    $QWIKMD::selresTable selection set $sel
                    QWIKMD::rowSelection
                    $QWIKMD::selresTable see [lindex $sel end]
                } else {
                    $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butClear invoke
                }
                
                return
            } 
            update idletasks
            incr i
        }
        $atom delete
      } else {
        return 1
      }
}

proc QWIKMD::mean {values} {
    set a 0.0
    set total [llength $values]
    for {set i 0} {$i < $total} {incr i} {
        set a [expr $a + [lindex $values $i]]
    }
    return [expr $a / $total ]
}

proc QWIKMD::meanSTDV {values} {
    set sum 0.0
    set totsum 0.0
    set total [llength $values]
    set sum 0.0
    set stdv 0.0
    set j 1
    for {set i 0} {$i < $total} {incr i} {
        if {$j == 1} {

            set sum [lindex $values $i]
            set totsum $sum
            set stdv 0.0
        } else {
            set oldm $sum 
            set totsum [expr $totsum + [lindex $values $i]]
            set sum [expr $oldm + [expr [expr [lindex $values $i] - $oldm ] /$j]]
            set stdv [expr $stdv +  [expr [expr [lindex $values $i] - $oldm] * [expr [lindex $values $i] - $sum ] ]] 
        }
        incr j
    }
    return "[expr $totsum / $total] [expr sqrt($stdv/[expr $total -1])]" 
}
# proc QWIKMD::zoom {W D} {
#   if {$D > 0 && [expr $QWIKMD::maxy - $D] > 0} {
#         set delta $D
#         set afterid [after 200 "$W configure -ymax [expr $QWIKMD::maxy - $D]"]
#     } elseif {$D < 0 && [expr $QWIKMD::maxy - $D] < $QWIKMD::maxy} {
#         set afterid [after 200 "$W configure -ymax [expr $QWIKMD::maxy - $D]"]
#     }
# }

 proc QWIKMD::mincalc {values} {
    set length [llength $values]
    set min [lindex $values 0]
    for {set i 1} {$i < $length} {incr i} {
        if {[lindex $values $i] < $min} {
            set min [lindex $values $i]
        }
    }
    return $min
 }

  proc QWIKMD::maxcalc {values} {
    set length [llength $values]
    set max [lindex $values 0]
    for {set i 1} {$i < $length} {incr i} {
        if {[lindex $values $i] > $max} {
            set max [lindex $values $i]
        }
    }
    return $max
 }
