############################################################################
#cr
#cr            (C) Copyright 1995-2013 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

# suboptimalPaths - import network data from networkAnalysis and create 3D graphs
#               in VMD; requires networkView
#   John Eargle - eargle@illinois.edu
#    5 Nov 2008
#   13 Feb 2009


package provide networkview 1.41

namespace eval ::NetworkView {

    variable suboptimalPathLoaded ;# whether suboptimal path data is loaded or not
    array set suboptimalPathLoaded {}

    variable suboptimalPaths ;# array with suboptimal path information
    # accessed through network index and (source,target) pair
    # netIndex,source,target,totalPathCount: number of suboptimal paths
    # netIndex,source,target,pathIndex,hopCount: number of hops in path pathIndex
    # netIndex,source,target,pathIndex,hopIndex: nodeIndex of node at hopIndex in path pathIndex
    array set suboptimalPaths {}

    variable networkType ;# type of network; used to manage the network
    # generic: not associated with structures at all
    # backbone: just C-alpha (protein) and P (nucleic) atoms
    # XXX - PROBABLY DON'T NEED community: centers of mass for the various communities
    # residue: centers of mass for each residue

}


# Read suboptimal path file
# @param suboptimalPathFilename 
proc ::NetworkView::readSuboptimalPathFile { suboptimalPathFilename } {

    puts ">readSuboptimalPathFile"
    
    variable currentNetwork
    variable networkCount
    #variable molid
    variable nodes
    #variable edges
    variable suboptimalPathLoaded
    variable suboptimalPaths
    
    set numPaths 0
    set source 0
    set target 0
    #    set numNodesTotal 0 ;# count nodes to see if it matches number of nodes loaded
    
    # Check that a network is loaded
    if {$networkCount == 0} {
	puts "Error: readSuboptimalPathFile - cannot load suboptimal path data until a network is loaded"
	return
    }
    
    # Clean any existing shortest paths
    
    
    set suboptimalPathFile [open $suboptimalPathFilename "r"]
    
    gets $suboptimalPathFile
    gets $suboptimalPathFile
    gets $suboptimalPathFile
    set line [gets $suboptimalPathFile]
    if {[regexp {The sources are: (\d+)} $line matchVar source]} {
	puts "source: $source"
    } else {
	puts "Error: readSuboptimalPathFile - could not read source index"
    }
    set line [gets $suboptimalPathFile]
    if {[regexp {The targets are: (\d+)} $line matchVar target]} {
	puts "target: $target"
    } else {
	puts "Error: readSuboptimalPathFile - could not read target index"
    }
    set line [gets $suboptimalPathFile]
    set i 0
    set line [gets $suboptimalPathFile]
    regsub -all "," $line "" line
    set stepIndex 0
    set pathNumber 0
    
    # Handle two types of suboptimal path file:
    #   1) all suboptimal paths are on the same line
    #   2) suboptimal paths are on separate lines
    while {![regexp {umber of paths} $line] && ![eof $suboptimalPathFile]} {
	#puts "line: $line"
	foreach number $line {
	    #puts "number: $number"
	    if {$number == $source && $stepIndex == 0} {
	    } elseif {$number == $target} {
		incr stepIndex
		set suboptimalPaths($currentNetwork,$source,$target,$pathNumber,hopCount) $stepIndex
		set stepIndex 0
		incr pathNumber
	    } elseif {[regexp {\(.*\)} $number]} {
		puts "path length: $number"
	    } else {
		set suboptimalPaths($currentNetwork,$source,$target,$pathNumber,$stepIndex) $number
		incr stepIndex
	    }
	}
	if {$stepIndex != 0} {
	    puts "Error: readSuboptimalPathFile - a suboptimal path did not end with the correct source ($number vs. $source)"
	    puts $line
	    return
	}
	set line [gets $suboptimalPathFile]
	regsub -all "," $line "" line
    }
    set suboptimalPaths($currentNetwork,$source,$target,totalPathCount) $pathNumber
    
    # Check total path count
    #set line [gets $suboptimalPathFile]
    close $suboptimalPathFile
    
    if {[regexp {umber of paths .+ (\d+)} $line matchVar pathCount]} {
	if {$pathCount != $pathNumber} {	    
	    puts "Error: readSuboptimalPathFile - stated number of paths different from number read in ($pathCount vs. $pathNumber)"
	}
    } else {
	puts "Error: readSuboptimalPathFile - could not read number of paths"
    }
    
    set numLoadedPaths [array get suboptimalPathLoaded "$currentNetwork"]
    if {[llength $numLoadedPaths] == 0} {
	set suboptimalPathLoaded($currentNetwork) 1
    } else {
	incr suboptimalPathLoaded($currentNetwork)
    }
    
    puts "<readSuboptimalPathFile"
    return
}


# Use suboptimal path information to activate a single path
# @param source
# @param target
# @param pathIndex
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateSuboptimalPath { args } {
    
    variable currentNetwork
    variable suboptimalPaths

    set source 0
    set target 0
    set pathIndex 0
    set setOrUnset 1

    if {[llength $args] == 3} {
	set source [lindex $args 0]
	set target [lindex $args 1]
	set pathIndex [lindex $args 2]
    } elseif {[llength $args] == 4} {
	set source [lindex $args 0]
	set target [lindex $args 1]
	set pathIndex [lindex $args 2]
	set setOrUnset [lindex $args 3]
    } else {
	puts "Error: ::NetworkView::activateSuboptimalPath - wrong number of arguments"
	puts "  activateSuboptimalPath source target pathIndex \[activateOrDeactivate\]"
	puts "    source: suboptimal path source"
	puts "    target: suboptimal path target"
	puts "    pathIndex: index of suboptimal path"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    set hopCountCheck [array get suboptimalPaths "$currentNetwork,$source,$target,$pathIndex,hopCount"]
    if {$hopCountCheck == ""} {
	puts "Error: activateSuboptimalPath - no available path from $source to $target with pathIndex $pathIndex"
	return
    }
    set hopCount $suboptimalPaths($currentNetwork,$source,$target,$pathIndex,hopCount)

    # Loop through hops along a path
    set nodeIndex1 $source
    activateNodes [list $source] $setOrUnset
    for {set i 0} {$i < [expr $hopCount - 1]} {incr i} {
	set nodeIndex2 $suboptimalPaths($currentNetwork,$source,$target,$pathIndex,$i)
	activateNodes [list $nodeIndex2] $setOrUnset
	activateInternalEdges [list $nodeIndex1 $nodeIndex2] $setOrUnset
	set nodeIndex1 $nodeIndex2
    }
    activateNodes [list $target] $setOrUnset
    activateInternalEdges [list $nodeIndex2 $target] $setOrUnset

    return
}


# Use suboptimal path information to deactivate a single path
# @param source
# @param target
# @param pathIndex
proc ::NetworkView::deactivateSuboptimalPath { source target pathIndex } {

    activateSuboptimalPath $source $target $pathIndex 0

    return
}


# Use suboptimal path information to activate multiple paths
# @param source
# @param target
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateSuboptimalPaths { args } {

    variable currentNetwork
    variable suboptimalPaths

    set source 0
    set target 0
    set setOrUnset 1

    if {[llength $args] == 2} {
	set source [lindex $args 0]
	set target [lindex $args 1]
    } elseif {[llength $args] == 3} {
	set source [lindex $args 0]
	set target [lindex $args 1]
	set setOrUnset [lindex $args 2]
    } else {
	puts "Error: ::NetworkView::activateSuboptimalPaths - wrong number of arguments"
	puts "  activateSuboptimalPaths source target \[activateOrDeactivate\]"
	puts "    source: suboptimal path source"
	puts "    target: suboptimal path target"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    set numPaths $suboptimalPaths($currentNetwork,$source,$target,totalPathCount)

    # Loop through paths
    puts -nonewline "activating "
    for {set i 0} {$i < $numPaths} {incr i} {
	activateSuboptimalPath $source $target $i $setOrUnset
	puts -nonewline "$i "
    }
    puts ""

    return
}


# Use suboptimal path information to deactivate multiple paths
# @param source
# @param target
proc ::NetworkView::deactivateSuboptimalPaths { source target } {

    activateSuboptimalPaths $source $target 0

    return
}


# Use suboptimal path information to color a single path
# @param source
# @param target
# @param pathIndex
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorSuboptimalPath { args } {

    variable currentNetwork
    variable suboptimalPaths

    set source 0
    set target 0
    set pathIndex 0
    set color 0

    if {[llength $args] == 4} {
	set source [lindex $args 0]
	set target [lindex $args 1]
	set pathIndex [lindex $args 2]
	set color [lindex $args 3]
    } else {
	puts "Error: ::NetworkView::colorSuboptimalPaths - wrong number of arguments"
	puts "  colorSuboptimalPath source target pathIndex colorId"
	puts "    source: suboptimal path source"
	puts "    target: suboptimal path target"
	puts "    pathIndex: index of suboptimal path"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    set hopCountCheck [array get suboptimalPaths "$currentNetwork,$source,$target,$pathIndex,hopCount"]
    if {$hopCountCheck == ""} {
	puts "Error: colorSuboptimalPath - no available path from $source to $target with pathIndex $pathIndex"
	return
    }
    set hopCount $suboptimalPaths($currentNetwork,$source,$target,$pathIndex,hopCount)

    # Loop through hops along a path
    set nodeIndex1 $source
    colorNodes [list $source] $color
    for {set i 0} {$i < [expr $hopCount - 1]} {incr i} {
	set nodeIndex2 $suboptimalPaths($currentNetwork,$source,$target,$pathIndex,$i)
	colorNodes [list $nodeIndex2] $color
	colorInternalEdges [list $nodeIndex1 $nodeIndex2] $color
	set nodeIndex1 $nodeIndex2
    }
    colorNodes [list $target] $color
    colorInternalEdges [list $nodeIndex2 $target] $color

    return
}


# Use suboptimal path information to color the network
# @param source
# @param target
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorSuboptimalPaths { args } {

    variable currentNetwork
    variable suboptimalPaths

    set source 0
    set target 0
    set color 0

    if {[llength $args] == 3} {
	set source [lindex $args 0]
	set target [lindex $args 1]
	set color [lindex $args 2]
    } else {
	puts "Error: ::NetworkView::colorSuboptimalPaths - wrong number of arguments"
	puts "  colorSuboptimalPaths source target colorId"
	puts "    source: suboptimal path source"
	puts "    target: suboptimal path target"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    set numPaths $suboptimalPaths($currentNetwork,$source,$target,totalPathCount)

#    colorNodes [list $source $target]

    # Loop through paths
    puts -nonewline "coloring "
    for {set i 0} {$i < $numPaths} {incr i} {
	colorSuboptimalPath $source $target $i $color
	puts -nonewline "$i "
    }
    puts ""

    return
}


# Set edge values according to the number of suboptimal paths going through them
# @param source
# @param target
proc ::NetworkView::countSuboptimalPathsPerEdge { args } {

    puts ">countSuboptimalPathsPerEdge"
    variable BIGNUM
    variable currentNetwork
    variable nodes2
    variable edges2
    variable nodeCount
    variable edgeMaxValue
    variable edgeMinValue
    variable edgeValue
    variable edgeUpperNodes
    variable suboptimalPaths

    set source 0
    set target 0
    
    if {[llength $args] == 3} {
      set source [lindex $args 0]
      set target [lindex $args 1]
      set countscale [lindex $args 2]
      puts "source: $source, target: $target, scale: $countscale"
    } else {
	puts "Error: ::NetworkView::countSuboptimalPathsPerEdge - wrong number of arguments"
	puts "  countSuboptimalPathsPerEdge source target scalingFactor"
	puts "    source: suboptimal path source"
	puts "    target: suboptimal path target"
	puts "    scalingFactor: Results scaled by this factor. Normally 1.0"
	return
    }

    set numPaths $suboptimalPaths($currentNetwork,$source,$target,totalPathCount)

    #### Array
    # Clear edge values
#    set edges($currentNetwork,maxvalue) 0
#    set edges($currentNetwork,minvalue) 1000000000
#    foreach {valueKey val} [array get edges "$currentNetwork,*,value"] {
#	puts "valueKey: $valueKey"
#	regexp {\d+,(\d+,\d+),value} $valueKey matchVar indices
#	puts "indices: $indices"
#	set edges($currentNetwork,$indices,value) 0
#    }
    
    # Calculate new values from suboptimal path counts
#    for {set i 0} {$i < $numPaths} {incr i} {
#	puts "path: $i"
#	set numHops $suboptimalPaths($currentNetwork,$source,$target,$i,hopCount)
#	set nodeIndex1 $source
#	for {set j 0} {$j < [expr $numHops - 1]} {incr j} {
#	    set nodeIndex2 $suboptimalPaths($currentNetwork,$source,$target,$i,$j)
#	    puts -nonewline "($nodeIndex1,$nodeIndex2) "
#	    if {$nodeIndex1 < $nodeIndex2} {
#		incr edges($currentNetwork,$nodeIndex1,$nodeIndex2,value)
#	    } else {
#		incr edges($currentNetwork,$nodeIndex2,$nodeIndex1,value)
#	    }
#	    set nodeIndex1 $nodeIndex2
#	}
#	if {$nodeIndex2 < $target} {
#	    incr edges($currentNetwork,$nodeIndex2,$target,value)
#	} else {
#	    incr edges($currentNetwork,$target,$nodeIndex2,value)
#	}
#	puts "($nodeIndex2,$target)"
#    }

    # Loop through edges and determine maxvalue and minvalue
#    foreach {valueKey val} [array get edges "$currentNetwork,*,value"] {
#	if {$val > $edges($currentNetwork,maxvalue)} {
#	    set edges($currentNetwork,maxvalue) $val
#	}
#	if {$val < $edges($currentNetwork,minvalue)} {
#	    set edges($currentNetwork,minvalue) $val
#	}
#    }


    ### List
    # Clear edge values
    lset edges2 $currentNetwork $edgeMaxValue 0
    lset edges2 $currentNetwork $edgeMinValue $BIGNUM

    set numNodes [lindex $nodes2 $currentNetwork $nodeCount]
    for {set index1 0} {$index1 < $numNodes} {incr index1} {
	set j 0
	foreach value [lindex $edges2 $currentNetwork $edgeValue $index1] {
	    lset edges2 $currentNetwork $edgeValue $index1 $j 0
	    incr j
	}
    }	
    
    # Calculate new values from suboptimal path counts
    for {set i 0} {$i < $numPaths} {incr i} {
	puts "path: $i"
	set numHops $suboptimalPaths($currentNetwork,$source,$target,$i,hopCount)
	set nodeIndex1 $source
	for {set j 0} {$j < [expr $numHops - 1]} {incr j} {
	    set nodeIndex2 $suboptimalPaths($currentNetwork,$source,$target,$i,$j)
	    puts -nonewline "($nodeIndex1,$nodeIndex2) "
	    if {$nodeIndex1 < $nodeIndex2} {
		set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $nodeIndex1] $nodeIndex2]
		set currentValue [lindex $edges2 $currentNetwork $edgeValue $nodeIndex1 $tempIndex2]
		lset edges2 $currentNetwork $edgeValue $nodeIndex1 $tempIndex2 [expr $currentValue + $countscale]
	    } else {
		set tempIndex1 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $nodeIndex2] $nodeIndex1]
		set currentValue [lindex $edges2 $currentNetwork $edgeValue $nodeIndex2 $tempIndex1]
		lset edges2 $currentNetwork $edgeValue $nodeIndex2 $tempIndex1 [expr $currentValue + $countscale]
	    }
	    set nodeIndex1 $nodeIndex2
	}
	if {$nodeIndex2 < $target} {
	    set tempIndex1 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $nodeIndex2] $target]
	    set currentValue [lindex $edges2 $currentNetwork $edgeValue $nodeIndex2 $tempIndex1]
	    lset edges2 $currentNetwork $edgeValue $nodeIndex2 $tempIndex1 [expr $currentValue + $countscale]
	} else {
	    set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $target] $nodeIndex2]
	    set currentValue [lindex $edges2 $currentNetwork $edgeValue $target $tempIndex2]
	    lset edges2 $currentNetwork $edgeValue $target $tempIndex2 [expr $currentValue + $countscale]
	}
	puts "($nodeIndex2,$target)"
    }

    # Loop through edges and determine maxvalue and minvalue
    foreach {valueKey val} [array get edges "$currentNetwork,*,value"] {
	if {$val > $edges($currentNetwork,maxvalue)} {
	    set edges($currentNetwork,maxvalue) $val
	}
	if {$val < $edges($currentNetwork,minvalue)} {
	    set edges($currentNetwork,minvalue) $val
	}
    }

    for {set index1 0} {$index1 < $numNodes} {incr index1} {
	foreach value [lindex $edges2 $currentNetwork $edgeValue $index1] {
	    if {$value > [lindex $edges2 $currentNetwork $edgeMaxValue]} {
		lset edges2 $currentNetwork $edgeMaxValue $value
	    }
	    if {$value < [lindex $edges2 $currentNetwork $edgeMinValue]} {
		lset edges2 $currentNetwork $edgeMinValue $value
	    }
	}
    }	
    
    puts "<countSuboptimalPathsPerEdge"
    return
}


# Print out the nodes for a given suboptimal path
# @param source
# @param target
# @param pathIndex
proc ::NetworkView::printSuboptimalPath { args } {

    variable currentNetwork
    variable suboptimalPaths

    set source 0
    set target 0
    set pathIndex 0

    if {[llength $args] == 3} {
	set source [lindex $args 0]
	set target [lindex $args 1]
	set pathIndex [lindex $args 2]
    } else {
	puts "Error: ::NetworkView::printSuboptimalPath - wrong number of arguments"
	puts "  printSuboptimalPath source target pathIndex"
	puts "    source: suboptimal path source"
	puts "    target: suboptimal path target"
	puts "    pathIndex: index of suboptimal path"
	return
    }

    set hopCountCheck [array get suboptimalPaths "$currentNetwork,$source,$target,$pathIndex,hopCount"]
    if {$hopCountCheck == ""} {
	puts "Error: colorSuboptimalPath - no available path from $source to $target with pathIndex $pathIndex"
	return
    }
    set hopCount $suboptimalPaths($currentNetwork,$source,$target,$pathIndex,hopCount)

    # Loop through hops along a path
    puts -nonewline "$target "
    for {set i 0} {$i < [expr $hopCount - 1]} {incr i} {
	puts -nonewline "$suboptimalPaths($currentNetwork,$source,$target,$pathIndex,$i) "
    }
    puts "$source"
    
    return
}
