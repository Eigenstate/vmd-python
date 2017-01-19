##
## StrctCheck 1.0
##
## A script to evaluate possible structure errors
##
## Author: Joao Ribeiro (jribeiro@ks.uiuc.edu)
##         
##         
##         
##
## $Id: structurecheck.tcl,v 1.4 2016/08/11 23:04:28 jribeiro Exp $
##


package provide structurecheck 1.0
package require chirality
package require torsionplot

namespace eval StrctCheck:: {
    set logfile ""
    set verbose 0
    set qwikmd 0
}

proc strctcheck { args }     { return [eval StrctCheck::strctcheck $args] }

proc StrctCheck::reset {} {
    set verbose 0
    set logfile ""
}


proc StrctCheck::printInfo {info} {
    variable verbose
    variable logfile
    if $verbose {
        puts "StrctCheck - $info"
    }
    if {$logfile != ""} {
        puts $StrctCheck::logfile "StrctCheck - $info"
    }
    
}

proc StrctCheck::checkChirality {molid selText} {
    variable qwikmd
    # if $qwikmd {
    #     append selText " and not nucleic"
    # } else {
    #     append selText " and not qwikmd_glycan"
    # }

    chirality reset
    set chiralerrors ""
    set listerror [list]
    catch {set chiralerrors [chirality check -mol $molid -seltext $selText]}
    if {$chiralerrors != ""} {
        for {set i 0} {$i < $chiralerrors} {incr i} { 
            lappend listerror [chirality list $i -mol $molid]
        }
    } 
    chirality reset 
    StrctCheck::printInfo "Found $chiralerrors chirality center(s) error in the molecule $molid"
    return [list $chiralerrors $listerror]
}

proc StrctCheck::checkCisPeptide {molid selText} {
    variable qwikmd


    # if $qwikmd {
    #     append selText " and not nucleic"
    # } else {
    #     append selText " and not qwikmd_glycan"
    # }

    cispeptide reset
    set cisperrors ""
    set listerror [list]
    catch {set cisperrors [cispeptide check -mol $molid -seltext $selText]}
    if {$cisperrors != ""} {
        for {set i 0} {$i < $cisperrors} {incr i} { 
            lappend listerror [cispeptide list $i -mol $molid]
        }
    }  
    cispeptide reset
    StrctCheck::printInfo "Found $cisperrors cispeptides bond(s) in the molecule $molid"
    return [list $cisperrors $listerror]
}

proc StrctCheck::checkGaps {molid selText} {
    variable qwikmd
    set oldresid ""
    set oldchain ""
    set oldsegname ""

 
    set proteinsel "protein"
    set nucleicsel "nucleic"
    set glycansel "glycan"
    if $qwikmd {
        set proteinsel "qwikmd_protein"
        set nucleicsel "qwikmd_nucleic"
        set glycansel "qwikmd_glycan"
    }
    set gpsList [list]

    set selprot [atomselect $molid "($selText) and ($proteinsel or $glycansel or $nucleicsel)"]
    set i 0
    set prevtext ""
    set prevfrag ""
    foreach chain [$selprot get chain] resid [$selprot get resid] segname [$selprot get segname] protein [$selprot get $proteinsel] glycan [$selprot get $glycansel] nucleic [$selprot get $nucleicsel] frag [$selprot get fragment] {
       set text "${chain}${resid}${segname}${protein}${nucleic}${glycan}$frag"
       if {$text != $prevtext} {

            if {($chain != $oldchain || $segname != $oldsegname)} {
                set oldchain $chain
                #set oldresid $resid
                set oldsegname $segname
                #incr i
            } else {
                if {[expr $resid - $oldresid] != 1 && ($resid != 1 && $oldresid != "-1") && $frag != $prevfrag} {
                    # check if the non-consequitive  residues are bonded or not (Protein Bond C - N) (Nucleic Bond formed by O3' - P)
                    set txt1 "resid \"$oldresid\" and chain \"$chain\""
                    set txt2 "resid \"$resid\" and chain \"$chain\""
                     set newgap 0
                    if {$protein == 1} {
                        append txt1 " and name C"
                        append txt2 " and name N"  
                    } elseif {$nucleic == 1} {
                        append txt1 " and name  \"O3.*\""
                        append txt2 " and name P"
                    } elseif {$glycan == 1} {
                        append txt1 " and name C1"
                        append txt2 " and name O1 O2 O3 O4 O6 ND2 and within 2.0 of ($txt1)"
                    } else {
                        set newgap 1
                    }
                    if {$newgap == 0} {
                        set sel1 [atomselect top [format $txt1]]
                        set sel2 [atomselect top $txt2]
                        set index1 [$sel1 get index]
                        set index2 [$sel2 get index]
                        if {[llength $index1] == 1 && [llength $index2] == 1} {
                            set dist [measure bond "$index1 $index2"]
                            if {[measure bond "$index1 $index2"] > 2.} {
                                set newgap 1
                            }
                        }
                        $sel1 delete
                        $sel2 delete
                    }
                    if {$newgap == 1} {
                        lappend gpsList [list $chain [list $oldresid $resid] ]
                    }
                                      
                } 
            }
            set oldresid $resid
            set prevtext $text
            set prevfrag $frag
        }
    }
    $selprot delete
    foreach info $gpsList {
        set residues [lindex $info 1]
        StrctCheck::printInfo "Found a gap in chain [lindex $info 0] between resid [lindex $residues 0] and [lindex $residues 1]"
    }
    return $gpsList
}

proc StrctCheck::ramacheck {molid selText} {
    set sel [atomselect $molid "((protein and name CA and not (same residue as name HT1 HN1)) or (glycan and name C1)) and ($selText)"]
    ::TorPlot::torplotReset
    set totalresidue [llength [lsort -unique [$sel get residue]]]
    set ramaoutlier [list]
    set ramamarginal [list]
    $sel delete
    set error ""
    if {$totalresidue > 0} {
       
        catch {torsionplot -cmd -molid $molid -seltext ${selText}} error

        ##Due to torsionplot/VMD limitation, only 6 types of torsion can be checked
        ## interpvoln doesn't go past n=6
        
        for {set i 0} {$i <= 6} {incr i} {
            #set aux 
            #if {[lindex $aux 0] != ""} {
                lappend ramaoutlier [::TorPlot::listOutlier $i]
            #}
            #set aux 
            #if {[lindex $aux 0] != ""} {
                lappend ramamarginal [::TorPlot::listMarginal $i]
               
            #}
        }
        ::TorPlot::torplotReset
    } 

    if {$error != ""} {
        return "Failed"
    } else {
        return [list $ramaoutlier $ramamarginal $totalresidue]
    }
    
}

proc StrctCheck::strctcheck { args } {
    variable verbose
    variable logfile
    variable qwikmd
    set result [list]
    set nargs [llength $args]
      if {$nargs < 1} {
        puts "Usage: strctcheck -mol <VMD molecule ID> -selText <selection text> -verbose <1|0> -qwikmd <1|0> -logfile <logfile name>"
        return
    }
    set molid ""
    set verbose 0
    set logfilename ""
    set selText "all"
    foreach {arg value} $args {
        switch $arg {
            "-mol" {
                set molid $value
            }
           "-verbose" {
                set verbose $value
           }
            "-logfile" {
                set logfilename ${value}
                set logfile [open $logfilename w+]
           }
           "-selText" {
                set selText $value
           }
           "-qwikmd" {
                if $value {
                    set qwikmd 1
                }
           }
        }
    }

    

    ####Check Chirality
    if {$qwikmd} {}
    set chiralityerrors [StrctCheck::checkChirality $molid "(($selText) and not water and not ion and not lipid)" ]
    lappend result [list "chiralityerrors" $chiralityerrors]
    ####Check Cispeptide residues
    set cispeperrors [StrctCheck::checkCisPeptide $molid "(($selText) and protein)"]
    lappend result [list "cispeperrors" $cispeperrors]
    ####Check Protein sequence Gaps
    set gaps [StrctCheck::checkGaps $molid $selText]
    lappend result [list "gaps" $gaps]
    ####Check Torsion angles
    set ramaerror [StrctCheck::ramacheck $molid $selText]
    # set outlier 0
    #  foreach rama [lindex $ramaerror 0] {
    #      foreach vis $rama {
    #          if {[lindex $vis 0] != ""} {
    #             incr outlier
    #          }
    #      }
    #  }

    #  set marginal 0
    #  foreach rama [lindex $ramaerror 1] {
    #      foreach vis $rama {
    #          if {[lindex $vis 0] != ""} {
    #             incr marginal
    #          }
    #      }
    #  }
    lappend result [list "torsionOutlier" [lindex $ramaerror 0] [lindex $ramaerror 2] ]
    lappend result [list "torsionMarginal" [lindex $ramaerror 1] [lindex $ramaerror 2]]
    if {$logfile != ""} {
        close $logfile
    }
    set logfile ""
    set verbose 0
    return $result
}
