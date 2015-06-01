############################################################################
#cr
#cr            (C) Copyright 1995-2013 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: communityNetwork.tcl,v 1.8 2014/02/20 20:16:08 kvandivo Exp $
#

# communityNetwork - import network data from networkAnalysis and create 3D graphs
#               in VMD; requires networkView
#   John Eargle - eargle@illinois.edu
#    5 Nov 2008
#   13 Feb 2009

package provide networkview 1.41

namespace eval ::NetworkView {

    variable communityLoaded ;# whether community data is loaded or not
    array set communityLoaded {}

    variable communityNodes ;# array with community node information
    # accessed through network index and community index
    # netIndex,communityCount: number of communities
    # netIndex,cIndex,coordinate: {x,y,z} location
    # netIndex,cIndex,objectIndex: OpenGL object index for deletion
    # netIndex,cIndex,nodeIndex,x: node indices for nodes in the community
    # netIndex,cIndex,value: arbitrary value assigned to communityNode
    # netIndex,cIndex,active: displayed or not
    # netIndex,cIndex,color: display color
    array set communityNodes {}

    variable communityEdges ;# array with community edge information
    # accessed through network index and pair of indices (ci1,ci2) for the communityNodes at each end
    # netIndex,ci1,ci2,weight: edge distance
    # netIndex,maxweight: maximum weight
    # netIndex,minweight: minimum weight
    # netIndex,ci1,ci2,betweenness: number of shortest paths crossing the edge
    # netIndex,ci1,ci2,objectIndex: OpenGL object index for deletion
    # netIndex,ci1,ci2,active: displayed or not
    # netIndex,ci1,ci2,color: display color
    array set communityEdges {}

    variable criticalNodes ;# array with critical node information
    # accessed through network index and community index
    # netIndex,nodeList: list of all criticalNodes
    # netIndex,cIndex,nodeCount: number of critical nodes in this community
    # netIndex,cIndex,index: node index for critical node belonging to a given community
    # netIndex,cIndex1,cIndex2,nodePairCount: number of critical node pairs connecting these two communities
    # netIndex,cIndex1,cIndex2,index: node index for critical node pair between two communities
    array set criticalNodes {}

    variable criticalEdges ;# array with critical edge information
    # accessed through network index and pair of indices (ci1,ci2) for the communityNodes at each end
    # netIndex,edgeList: list of all criticalEdges as pairs of nodes
    # netIndex,ci1,ci2,edgeIndex: edge indices for critical edges between two communities
    array set criticalEdges {}

}


# Initialize the data structures that carry community network information.  readNetworkFile has to be called first so that the node indices are set
# @param communityFilename file assigning nodes to specific communities
# @param betweennessFilename 
proc ::NetworkView::readCommunityFile { communityFilename } {

#    puts ">readCommunityFile"

    #variable molid
    variable currentNetwork
    #variable nodes
    variable nodes2
    variable nodeCoordinate
    #variable edges
    variable communityLoaded
    variable communityNodes
    variable communityEdges
    variable criticalNodes
    variable criticalEdges

    set criticalNodes($currentNetwork,nodeList) [list ]
    set criticalEdges($currentNetwork,edgeList) [list ]
    set numNodesTotal 0 ;# count nodes to see if it matches number of nodes loaded
    set community1 -1
    set community2 -1

    # Check that a network is loaded
    if {$currentNetwork == -1} {
	puts "Error: readCommunityFile - no network loaded"
	return
    }

    set communityFile [open $communityFilename "r"]

    # Read in data for communityNodes and criticalNodes
    set line [gets $communityFile]
    #[regexp {optimum number of communities is (\d*)} $line matchVar numCommunities]
    if {[regexp {optimum number of communities is (\d*)} $line matchVar numCommunities]} {
	puts "Number of communities: $numCommunities"
    } else {
	puts "Error: readCommunityFile - could not read number of communities"
	return
    }

    for {set i 0} {$i < $numCommunities} {incr i} {
	for {set j [expr $i+1]} {$j < $numCommunities} {incr j} {
	    set criticalNodes($currentNetwork,$i,$j,nodePairCount) 0
	}
    }

    set i 0
    while {![eof $communityFile]} {
	set line [gets $communityFile]
	# Set up communityNodes
	if {[regexp {residues in community (\d+) are: (.*)} $line matchVar communityIndex indexString]} {
	    set communityIndex [expr $communityIndex-1]
	    incr i
	    set centerPoint {0 0 0}
	    set j 0
	    set numNodes 0
#       puts -nonewline "Community $communityIndex: "
	    foreach index $indexString {
#          puts -nonewline " $index,"
          set communityNodes($currentNetwork,$communityIndex,nodeIndex,$j) $index
          #set centerPoint [vecadd $centerPoint $nodes($currentNetwork,$index,coordinate)]
          set centerPoint [vecadd $centerPoint [lindex $nodes2 $currentNetwork $nodeCoordinate $index]]
          incr numNodes
          incr j
	    }
	    set communityNodes($currentNetwork,$communityIndex,coordinate) [vecscale $centerPoint [expr 1.0/$numNodes]]
	    set communityNodes($currentNetwork,$communityIndex,active) 1
	    set communityNodes($currentNetwork,$communityIndex,color) blue
	    incr numNodesTotal $numNodes
       puts ""	
	} elseif {[regexp {edge connectivities between communities (\d+) and (\d+)} $line matchVar c1 c2]} {
	    # Store community indices to set up criticalNodes
	    set community1 [expr $c1-1]
	    set community2 [expr $c2-1]
	} elseif {[regexp {(\d+) (\d+) \d+\.\d+} $line matchVar index1 index2]} {
	    # Set up criticalNodePairs
	    set currentNodePairIndex $criticalNodes($currentNetwork,$community1,$community2,nodePairCount)
	    set criticalNodes($currentNetwork,$community1,$community2,$currentNodePairIndex) [list $index1 $index2]
	    lappend criticalNodes($currentNetwork,nodeList) $index1
	    lappend criticalNodes($currentNetwork,nodeList) $index2
	    incr criticalNodes($currentNetwork,$community1,$community2,nodePairCount)
	    #lappend criticalEdges($currentNetwork,edgeList) [list $index1 $index2]
	    lappend criticalEdges($currentNetwork,edgeList) "$index1,$index2"
	}
	
    }
    close $communityFile

    if {$i != $numCommunities} {
	puts "Error: readCommunityFile - stated number of communities different from number read in ($numCommunities vs. $i)"
	return
    }
    set communityNodes($currentNetwork,communityCount) $numCommunities

#    set betweennessFile [open $betweennessFilename "r"]
    set betweennessFile [open $communityFilename "r"]

    # Set up communityEdges from Total Community Flow data
    set line [gets $betweennessFile]
    while {![regexp {Total Community Flow is} $line] && ![eof $betweennessFile]} {
	set line [gets $betweennessFile]
    }
    set line [gets $betweennessFile]
    set i 0
    while {![regexp {Intercommunity Flow is} $line] && ![eof $betweennessFile]} {
	set betweennesses [concat $line]
	#puts $betweennesses
	for {set j 0} {$j < [llength $betweennesses]} {incr j} {
	    set tempVal [lindex $betweennesses [expr $j]]
	    if {$tempVal != 0} {
		#set edges($i,$j) $tempVal
		#puts "edges($i,$j) $tempVal"
		set communityEdges($currentNetwork,$i,$j,betweenness) $tempVal
		set communityEdges($currentNetwork,$i,$j,active) 1
		set communityEdges($currentNetwork,$i,$j,color) blue
	    }
	}
	
	set line [gets $betweennessFile]
	incr i
    }
    close $betweennessFile

    #drawCommunityNetwork
    
    set communityLoaded($currentNetwork) 1

#    puts "<readCommunityFile"
    return
}


# Retrieve nodeIndices for all nodes in a given community
# @param communityId ID for community whose nodes should be fetched
# @return List of nodeIndices
proc ::NetworkView::getCommunityNodes { args } {

    variable currentNetwork
    variable communityNodes

    set communityId -1
    set nodeIndices [list ]

    if {[llength $args] == 1} {
	set communityId [lindex $args 0]
    } else {
	puts "Error: ::NetworkView::getCommunityNodes - wrong number of arguments"
	puts "  getCommunityNodes communityId"
	puts "    communityId: ID for community whose nodes should be fetched"
	return
    }

#    foreach {nodeIndex} [array get communityNodes "$communityId,nodeIndex,*"] {
#	    lappend nodeIndices $nodeIndex
#    }

    foreach {key nodeIndex} [array get communityNodes "$currentNetwork,$communityId,nodeIndex,*"] {
	lappend nodeIndices $nodeIndex
    }

    return $nodeIndices
}


# Retrieve nodeIndices for all critical nodes
# @return List of nodeIndices
proc ::NetworkView::getCriticalNodes { args } {

    #variable communityNodes
    variable currentNetwork
    variable criticalNodes

#    set nodeIndices [list ]

    if {[llength $args] == 0} {
    } else {
	puts "Error: ::NetworkView::getCriticalNodes - wrong number of arguments"
	puts "  getCriticalNodes"
	return
    }

#    for {set i 0} {$i < $communityNodes($currentNetwork,communityCount)} {incr i} {
#	for {set j [expr $i+1]} {$j < $communityNodes($currentNetwork,communityCount)} {incr j} {
#	    lappend nodeIndices [lindex 0 $nodePair]
#	    lappend nodeIndices [lindex 1 $nodePair]
#	}
#    }

    return $criticalNodes($currentNetwork,nodeList)
}


# Retrieve nodeIndex pairs for all critical edges
# @return List of nodeIndex pairs
proc ::NetworkView::getCriticalEdges { args } {

    variable currentNetwork
    variable criticalEdges

    if {[llength $args] == 0} {
    } else {
	puts "Error: ::NetworkView::getCriticalEdges - wrong number of arguments"
	puts "  getCriticalEdges"
	return
    }

    return $criticalEdges($currentNetwork,edgeList)
}


# Print out a gdf (GUESS) file based on the community structure
# @param gdfFilename File name for a .gdf format file
proc ::NetworkView::writeCommunityGdf { gdfFilename } {

    variable currentNetwork
    variable molids
    variable communityNodes
    variable communityEdges

    set gdfFile [open $gdfFilename "w"]

    # Write out community node information
    puts $gdfFile "nodedef> name,x,y,community int,card int"
    foreach {communityKey coord} [array get communityNodes "$currentNetwork,*,coordinate"] {
	#puts $gdfFile "C$community,$community,[expr $size * 20]"
	regexp {\d+,(\d+),coordinate} $communityKey matchVar community
	set rotationMatrix [molinfo $molids($currentNetwork) get rotate_matrix]
	set rotationCoord [vectrans [lindex $rotationMatrix 0] $coord]
	set xCoord [expr 10 * [lindex $rotationCoord 0]]
	set yCoord [expr -10 * [lindex $rotationCoord 1]]
	set size [llength [array get communityNodes "$currentNetwork,$community,nodeIndex,*"]]
	puts $gdfFile "C$community,$xCoord,$yCoord,$community,$size"
    }

    # Write out community edge information with betweenness values
    puts $gdfFile "edgedef> node1,node2,betweenness int"
    foreach {edgeKey betweenness} [array get communityEdges "$currentNetwork,*,*,betweenness"] {
	regexp {\d+,(\d+),(\d+),betweenness} $edgeKey matchVar node1 node2
	if {$node1 < $node2} {
	    #puts $gdfFile "C$node1,C$node2,[format "%.0f" [expr $betweenness / 100.0]]"
	    puts $gdfFile "C$node1,C$node2,[format "%.0f" $betweenness]"
	}
    }
    close $gdfFile

    return
}


# Draw nodes that each represent an entire community and the edges between them
proc ::NetworkView::drawCommunityNetwork {} {
    
#    puts ">drawCommunityNetwork"

    drawCommunityNodes
    drawCommunityEdges

#    puts "<drawCommunityNetwork"
    return
}


# Draw nodes that each represent an entire community
proc ::NetworkView::drawCommunityNodes {} {

    #variable molid
    variable currentNetwork
    variable communityNodes
    variable sphereRadius
    variable sphereResolution

    set radius $sphereRadius
    set resolution $sphereResolution

    foreach colorKey [array get communityNodes "$currentNetwork,*,color"] {
	regexp {\d+,(\d+),} $colorKey matchVar index
	#puts "drawNode $index $radius $resolution"
	drawCommunityNode $index $radius $resolution
    }

    return
}


# Draw edges that connect community nodes
proc ::NetworkView::drawCommunityEdges {} {

    variable currentNetwork
    variable communityEdges
    variable cylinderRadius
    variable cylinderResolution

    set radius $cylinderRadius
    set resolution $cylinderResolution

    foreach colorKey [array get communityEdges "$currentNetwork,*,color"] {
	regexp {\d+,(\d+),(\d+),} $colorKey matchVar index1 index2
	drawCommunityEdge $index1 $index2 $radius $resolution
    }

    return
}


# Delete current communityNode OpenGL object and redraw a new one if active
# @param index1 Index of community node
# @param radius Sphere radius
# @param resolution Sphere resolution
proc ::NetworkView::drawCommunityNode { index radius resolution } {

    #puts ">drawCommunityNode"
    variable currentNetwork
    variable molids
    variable communityNodes

    # if OpenGL object already exists, delete
    set objectIndices [array get communityNodes "$currentNetwork,$index,objectIndex"]
    if {[llength $objectIndices] > 0} {
	graphics $molids($currentNetwork) delete $communityNodes([lindex $objectIndices 0])
    }

    if {$communityNodes($currentNetwork,$index,active) != 0} {
	graphics $molids($currentNetwork) color $communityNodes($currentNetwork,$index,color)
	set communityNodes($currentNetwork,$index,objectIndex) [graphics $molids($currentNetwork) sphere $communityNodes($currentNetwork,$index,coordinate) radius $radius resolution $resolution]
    }

    #puts "<drawCommunityNode"
    return
}


# Delete current communityEdge OpenGL object and redraw a new one if active
# @param index1 Index of first community node
# @param index2 Index of second community node
# @param radius Cylinder radius
# @param resolution Cylinder resolution
proc ::NetworkView::drawCommunityEdge { index1 index2 radius resolution } {

    #puts ">drawEdge"
    #puts "($index1,$index2) radius: $radius, resolution: $resolution"
    variable currentNetwork
    variable molids
    variable communityNodes
    variable communityEdges

    # if OpenGL object already exists, delete
    set objectIndices [array get communityEdges "$currentNetwork,$index1,$index2,objectIndex"]
    if {[llength $objectIndices] > 0} {
	graphics $molids($currentNetwork) delete $communityEdges([lindex $objectIndices 0])
    }

    if {$communityEdges($currentNetwork,$index1,$index2,active) != 0} {
	graphics $molids($currentNetwork) color $communityEdges($currentNetwork,$index1,$index2,color)
	set communityEdges($currentNetwork,$index1,$index2,objectIndex) [graphics $molids($currentNetwork) cylinder $communityNodes($currentNetwork,$index1,coordinate) $communityNodes($currentNetwork,$index2,coordinate) radius $radius resolution $resolution]
    }

    #puts "<drawEdge"
    return
}





################################################################################
#                                                                              #
#  All procs below apply to network nodes/edges within specified communities,  #
#  not community nodes/edges.                                                  #
#                                                                              #
################################################################################


# Activate all network nodes within a specific community
# @param communityId Community ID specifying the network nodes to be activated
# @param activateOrDeactivate Activate (1 - default) or deactivate (0)
proc ::NetworkView::activateCommunity { args } {

    set communityId -1
    set setOrUnset 1

    if {[llength $args] == 1} {
	set communityId [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set communityId [lindex $args 0]
	set setOrUnset [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateCommunity - wrong number of arguments"
	puts "  activateCommunity communityId \[activateOrDeactivate\]"
	puts "    communityId: ID for community to be activated"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    set nodeIndices [getCommunityNodes $communityId]
    activateNodes $nodeIndices $setOrUnset
    activateInternalEdges $nodeIndices $setOrUnset

    return
}


# Deactivate a specific community
# @param communityId Community ID specifying the network nodes to be deactivated
proc ::NetworkView::deactivateCommunity { communityId } {

    activateCommunity $communityId 0

    return
}


# Use community definitions to activate the network
# @param activateOrDeactivate Activate (1 - default) or deactivate (0)
# @param communityIds List of community IDs specifying the network nodes to be activated; default "all"
proc ::NetworkView::activateCommunities { args } {

    variable currentNetwork
    variable communityNodes

    set setOrUnset 1
    set communityIds [list ]
    set nodeIndices [list ]

    if {[llength $args] == 0} {
	foreach {communityKey active} [array get communityNodes "$currentNetwork,*,active"] {
	    regexp {\d+,(\d+),active} $communityKey matchVar communityId
	    lappend communityIds $communityId
	}
    } elseif {[llength $args] == 1} {
	set setOrUnset [lindex $args 0]
	foreach {communityKey active} [array get communityNodes "$currentNetwork,*,active"] {
	    regexp {\d+,(\d+),active} $communityKey matchVar communityId
	    lappend communityIds $communityId
	}
    } elseif {[llength $args] == 2} {
	set setOrUnset [lindex $args 0]
	set communityIds [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateCommunities - wrong number of arguments"
	puts "  activateCommunities \[activateOrDeactivate \[communityIds\]\]"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	puts "    communityIds: ID for communities to be activated; default all"
	return
    }

    foreach communityId $communityIds {
	set nodeIndices [concat $nodeIndices [getCommunityNodes $communityId]]
    }

    activateNodes $nodeIndices $setOrUnset
    activateInternalEdges $nodeIndices $setOrUnset
    
    return
}


# Use community definitions to deactivate the network
# @param communityIds List of community IDs specifying the network nodes to be activated
proc ::NetworkView::deactivateCommunities { communityIds } {

    activateCommunities 0 $communityIds
    
    return
}


# Color a specific community
# @param communityId Community ID specifying the network nodes to be colored
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorCommunity { args } {

    set communityId -1
    set color 0

    if {[llength $args] == 1} {
	set communityId [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set communityId [lindex $args 0]
	set color [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::colorCommunity - wrong number of arguments"
	puts "  colorCommunity communityId \[colorId\]"
	puts "    communityId: ID for community to be colored"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    set nodeIndices [getCommunityNodes $communityId]
    colorNodes $nodeIndices $color
    colorInternalEdges $nodeIndices $color

    return
}


# Use community definitions to color the network
# @param communityColorId Color for within communities (0-blue (default), 1-red, ...)
# @param interfaceColorId Color for within communities (0-blue (default), 1-red, ...)
# @param communityIds List of community IDs specifying the network nodes to be colored
proc ::NetworkView::colorCommunities { args } {

#    puts ">colorCommunities"

    variable currentNetwork
    variable communityNodes

    set communityColor 0
    set interfaceColor 1
    set communityIds [list ]
    set nodeIndices [list ]

    if {[llength $args] == 2} {
	set communityColor [lindex $args 0]
	set interfaceColor [lindex $args 1]
	foreach {communityKey active} [array get communityNodes "$currentNetwork,*,active"] {
	    regexp {\d+,(\d+),active} $communityKey matchVar communityId
	    lappend communityIds $communityId
	}
    } elseif {[llength $args] == 3} {
	set communityColor [lindex $args 0]
	set interfaceColor [lindex $args 1]
	set communityIds [lindex $args 2]
    } else {
	puts "Error: ::NetworkView::colorCommunities - wrong number of arguments"
	puts "  colorCommunities  communityColorId interfaceColorId \[communityIds\]"
	puts "    communityColorId: 0-blue (default), 1-red, ..."
	puts "    interfaceColorId: 0-blue (default), 1-red, ..."
	puts "    communityIds: ID for communities to be colored; default all"
	return
    }

    
    foreach communityId $communityIds {
	set nodeIndices [concat $nodeIndices [getCommunityNodes $communityId]]
    }

    colorNodes $nodeIndices $interfaceColor
    colorInternalEdges $nodeIndices $interfaceColor

    foreach communityId $communityIds {
	set nodeIndices [getCommunityNodes $communityId]
	colorNodes $nodeIndices $communityColor
	colorInternalEdges $nodeIndices $communityColor
    }
    
#    puts "<colorCommunities"
    return
}


# Activate critical nodes
# @param activateOrDeactivate Activate (1 - default) or deactivate (0)
proc ::NetworkView::activateCriticalNodes { args } {

    set setOrUnset 1

    if {[llength $args] == 0} {
    } elseif {[llength $args] == 1} {
	set setOrUnset [lindex $args 0]
    } else {
	puts "Error: ::NetworkView::activateCriticalNodes - wrong number of arguments"
	puts "  activateCriticalNodes \[activateOrDeactivate\]"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    set nodeIndices [getCriticalNodes]
    activateNodes $nodeIndices $setOrUnset

    set nodeIndexPairs [getCriticalEdges]
    # >New - 7 Jun 2010
    activateEdges $nodeIndexPairs $setOrUnset
    # <New

    #foreach indexPair $nodeIndexPairs {
    #activateInternalEdges $indexPair $setOrUnset
    #}

    return
}


# Deactivate critical nodes
proc ::NetworkView::deactivateCriticalNodes { } {

    activateCriticalNodes 0

    return
}


# Color critical nodes
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorCriticalNodes { args } {

    set color 0

    if {[llength $args] == 0} {
    } elseif {[llength $args] == 1} {
	set color [lindex $args 0]
    } else {
	puts "Error: ::NetworkView::colorCriticalNodes - wrong number of arguments"
	puts "  colorCriticalNodes \[colorId\]"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    set nodeIndices [getCriticalNodes]
    colorNodes $nodeIndices $color

    set nodeIndexPairs [getCriticalEdges]
    # >New 7 June 2010
    colorEdges $nodeIndexPairs $color
    # <New

    #foreach indexPair $nodeIndexPairs {
    #colorEdges $indexPair $color
    #}

    return
}
