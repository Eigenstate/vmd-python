############################################################################
#cr
#cr            (C) Copyright 1995-2013 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: networkView.tcl,v 1.10 2014/03/28 18:06:46 kvandivo Exp $
#

# networkView - import network data from networkAnalysis and create 3D graphs
#               in VMD
#   John Eargle - eargle@illinois.edu
#      Nov 2008, Feb 2009, Dec 2010, Mar-Jun 2011

package provide networkview 1.41
package require psfgen 1.5

proc networkView {args} {
    global errorInfo errorCode
    set oldcontext [psfcontext new]  ;# new context
    set errflag [catch { eval ::NetworkView::networkView $args } errMsg]
    set savedInfo $errorInfo
    set savedCode $errorCode
    psfcontext $oldcontext delete  ;# revert to old context
    if $errflag { error $errMsg $savedInfo $savedCode }
    return
}


namespace eval ::NetworkView {
    
    namespace export *

    variable BIGNUM 1000000000

    variable currentNetwork -1 ;# network index for currently active network; -1 for no active network
    variable networkCount 0 ;# number of loaded networks

    variable molids ;# array from network index to molid
    array set molids {}

    variable nodes ;# array with node information
    # accessed through network index (0 through numNetworks-1) and node index (0 through numNodes-1)
    # netIndex,index,coordinate: {x,y,z} location
    # netIndex,index,vmdIndex: VMD index
    # netIndex,index,objectIndex: OpenGL object index for deletion
    # netIndex,index,chain: VMD chain name
    # netIndex,index,resid: VMD resid
    # netIndex,index,community: community number
    # netIndex,index,value: arbitrary value assigned to node (e.g. conservation, change in CPL)
    # netIndex,index,active: displayed or not
    # netIndex,index,color: display color
    array set nodes {}

    variable nodes2 [list ]
    # accessed through network index (0 through numNetworks-1) and node index (0 through numNodes-1)
    # netIndex,coordinate,index: {x,y,z} location
    set nodeCoordinate 0
    # netIndex,vmdIndex,index: VMD index
    set nodeVmdIndex 1
    # netIndex,chain,index: VMD chain name
    set nodeChain 2
    # netIndex,resid,index: VMD resid
    set nodeResid 3
    # netIndex,value,index: arbitrary value assigned to node (e.g. conservation, change in CPL)
    set nodeValue 4
    # netIndex,active,index: displayed or not
    set nodeActive 5
    # netIndex,color,index: display color
    set nodeColor 6
    # netIndex,objectIndex,index: OpenGL object index for deletion
    set nodeObjectIndex 7
    # netIndex,community,index: community number
    set nodeCommunity 8
    # netIndex,nodeCount: number of nodes in the network
    set nodeCount 9

    variable sphereRadius 1.0   ;# OpenGL radius
    variable sphereResolution 8 ;# OpenGL resolution

    variable edges ;# array with node information
    # accessed through network index and pair of indices (i1,i2) for the nodes at each end
    # netIndex,i1,i2,weight: edge distance
    # netIndex,maxweight: maximum weight
    # netIndex,minweight: minimum weight
    # netIndex,i1,i2,betweenness: number of shortest paths crossing the edge
    # netIndex,maxbetweenness: maximum betweenness
    # netIndex,minbetweenness: minimum betweenness
    # netIndex,i1,i2,objectIndex: OpenGL object index for deletion
    # netIndex,i1,i2,value: arbitrary value assigned to edge (e.g. conservation, suboptimal path count)
    # netIndex,maxvalue: maximum value
    # netIndex,minvalue: minimum value
    # netIndex,i1,i2,active: displayed or not
    # netIndex,i1,i2,color: display color
    array set edges {}

    variable edges2 [list ]
    # accessed through network index and pair of indices (i1,i2) for the nodes at each end
    # the list index corresponding to the second index, i2, is obtained from:
    #   [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $i1] $i2]
    # netIndex,maxweight: maximum weight
    set edgeMaxWeight 0
    # netIndex,minweight: minimum weight
    set edgeMinWeight 1
    # netIndex,weight,i1,i2: edge distance
    set edgeWeight 2
    # netIndex,maxvalue: maximum value
    set edgeMaxValue 3
    # netIndex,minvalue: minimum value
    set edgeMinValue 4
    # netIndex,value,i1,i2: arbitrary value assigned to edge (e.g. conservation, suboptimal path count)
    set edgeValue 5
    # netIndex,active,i1,i2: displayed or not
    set edgeActive 6
    # netIndex,color,i1,i2: display color
    set edgeColor 7
    # netIndex,objectIndex,i1,i2: OpenGL object index for deletion
    set edgeObjectIndex 8
    # netIndex,maxbetweenness: maximum betweenness
    set edgeMaxBetweenness 9
    # netIndex,minbetweenness: minimum betweenness
    set edgeMinBetweenness 10
    # netIndex,betweenness,i1,i2: number of shortest paths crossing the edge
    set edgeBetweenness 11
    # netIndex,lowerNodes,i1: nodes < i1 that share an edge with i1
    set edgeLowerNodes 12
    # netIndex,upperNodes,i1: nodes >= i1 that share an edge with i1
    set edgeUpperNodes 13
    # netIndex,edgeCount: number of edges in the network
    set edgeCount 14

    variable cylinderRadius 0.3   ;# OpenGL radius
    variable cylinderResolution 8 ;# OpenGL resolution

    variable chainNames ;# VMD chain names for structures related to networks
    variable excludedAtoms ;# atomselection string for atoms that should be left out of the network
}


# -------------------------------------------------------------------------
# Create a new network
# @param molid
# @param networkFile
# @param chainNames
# @param excludedAtoms
# @return Whether loading the network was successful (1) or not (0)
proc ::NetworkView::networkView { args } {
    
#    puts ">networkView"
    variable networkCount
    variable chainNames
    variable excludedAtoms

    set excludedAtoms ""
    set success 0
    
    # Print usage information if no arguments are given
    if { [llength $args] < 2 || [llength $args] > 4} {
	puts "Usage: networkView molid networkFile \[chainNames \[excludedAtoms\]\]"
	puts "   molid - VMD molecule ID"
	puts "   networkFile - datafile from networkAnalysis"
	puts "   chainNames - string containing chain names"
	puts "   excludedAtoms - atomselection string for atoms that should be excluded from consideration"
	return $success
    }

    if {$networkCount == 0} {
	networkViewInit
    }

    set molid [lindex $args 0]
    set networkFilename [lindex $args 1]
    
    if {[llength $args] > 2} {
	set chainNames [lindex $args 2]
    }

    if {[llength $args] == 4} {
	set excludedAtoms [lindex $args 3]
    }

    set success [readNetworkFile $molid $networkFilename]
    if {$success == 1} {
	drawNetwork
    }
        
#    puts "<networkView: $success"
    return $success
}


# -------------------------------------------------------------------------
# Set all variables to default values
proc ::NetworkView::networkViewInit { } {

#    puts ">networkViewInit"
    variable currentNetwork
    variable molids
    variable networkCount
    variable nodes
    variable nodes2
    variable sphereRadius
    variable sphereResolution
    variable cylinderRadius
    variable cylinderResolution
    variable edges
    variable edges2
    variable networkType

    set currentNetwork -1
    array unset molids
    set networkCount 0
    
    array unset nodes
    set nodes2 [list ]
    set sphereRadius 1.0
    set sphereResolution 8

    array unset edges
    set edges2 [list ]
    set cylinderRadius 0.5
    set cylinderResolution 8

    set networkType "backbone"

#    puts "<networkViewInit"
    return
}


# -------------------------------------------------------------------------
# Initialize the data structures that carry network information
# @param molid
# @param networkFilename Name of network file
# @return Whether reading the network was successful (1) or not (0)
proc ::NetworkView::readNetworkFile { molid networkFilename } {
    
#    puts ">readNetworkFile"

    variable networkCount
    variable networkType
    variable currentNetwork
    variable molids

    set prevNetwork $currentNetwork
    set currentNetwork $networkCount
    set molids($currentNetwork) $molid
    set success 0

    set networkFile [open $networkFilename "r"]
    set line [gets $networkFile]
    if {[regexp {^\D} $line]} {
	set networkType "generic"
    }
    close $networkFile

    set success [createNodes $networkFilename]
#    puts "  createNodes: $success"
    if {$success == 1} {
	set success [createEdges $networkFilename]
#	puts "  createEdges: $success"
    }
    if {$success == 1} {
	incr networkCount
    } else {
	array unset molids $currentNetwork
	set currentNetwork $prevNetwork
    }

#    puts "<readNetworkFile: $success"
    return $success
}

# -------------------------------------------------------------------------
# Read a set of values from a matrix and load them into the current network.
# @param valueFilename Name of value file
# @return Whether reading the value file was successful (1) or not (0)
proc ::NetworkView::readValueDataFile { valueFilename } {
    
    variable currentNetwork
    variable molids
    variable nodes2
    variable nodeCount
    variable edges2
    variable edgeMaxValue
    variable edgeMinValue
    variable edgeValue
    variable edgeUpperNodes

    #   [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $i1] $i2]

    set valueFile [open $valueFilename "r"]
    set success 0

    # Skip first line if it contains an atomselection string
    # Check that the node set overlaps exactly with nodes defined in value file
    set line [gets $valueFile]
    if {[regexp {^\D} $line]} {
	set selString $line
	set sel [atomselect $molids($currentNetwork) $selString]
	if {[$sel num] != [lindex $nodes2 $currentNetwork $nodeCount]} {
	    puts "Error: number of nodes in value file ([$sel num]) does not agree with number of nodes in the network ([lindex $nodes2 $currentNetwork $nodeCount])"
	    return $success
	}
	$sel delete
	set line [gets $valueFile]
    }

    # Determine if the value file contains the correct size matrix
    if {[llength $line] != [lindex $nodes2 $currentNetwork $nodeCount]} {
	puts "Error: number of nodes ([lindex $nodes2 $currentNetwork $nodeCount]) does not agree with value matrix rank ([llength $line])"
	return $success
    }

    lset edges2 $currentNetwork $edgeMaxValue 0
    lset edges2 $currentNetwork $edgeMinValue 0

    # Load values into edge value slots
    set i 0
    while {![eof $valueFile]} {

	set j 0
	puts [lindex $edges2 $currentNetwork $edgeUpperNodes $i]
	foreach value $line {
	    #if {$value > 0} {}
	    if {$i < $j} {
		set j2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $i] $j]
		puts "($i,$j,$j2)"
		if {$j2 != -1} {
		    lset edges2 $currentNetwork $edgeValue $i $j2 $value
		    if {$value > [lindex $edges2 $currentNetwork $edgeMaxValue]} {
			lset edges2 $currentNetwork $edgeMaxValue $value
		    }
		    if {$value < [lindex $edges2 $currentNetwork $edgeMinValue]} {
			lset edges2 $currentNetwork $edgeMinValue $value
		    }
		}
	    }
	    #{}
	    incr j
	}
	incr i
	set line [gets $valueFile]
    }

    close $valueFile

    set success 1

    return $success
}

# -------------------------------------------------------------------------
# Read atomselection string from the first line of a network file
# and create the data structure that carries node information
# @param networkFilename Name of network file
# @return Whether reading in the nodes was successful (1) or not (0)
proc ::NetworkView::createNodes { networkFilename } {

    variable currentNetwork
    variable molids
    variable networkCount
    variable nodes2
    variable networkType
    variable chainNames
    variable excludedAtoms
    
    set atomNames ""
    set indices {}
    set numNodes 0
    set success 0

    set networkFile [open $networkFilename "r"]
    set line [gets $networkFile]
    if {[regexp {^\D} $line]} {
	set selString $line
	set line [gets $networkFile]
    }
    close $networkFile

    if {[string equal $networkType "generic"]} {
#	puts "generic"
	set sel [atomselect $molids($currentNetwork) $selString]
#	puts "selString: $selString"
	set indices [$sel get index]
#	puts "numIndices: [llength $indices]"
	set numNodes [$sel num]
	$sel delete
    } elseif {[string equal $networkType "backbone"]} {
#	puts "backbone"
	set atomNames "CA P PG"
	if {$excludedAtoms == ""} {
	    set sel [atomselect $molids($currentNetwork) "chain $chainNames and name $atomNames"]
	} else {
	    set sel [atomselect $molids($currentNetwork) "chain $chainNames and name $atomNames and not ($excludedAtoms)"]
	}
	set indices [$sel get index]
	$sel delete
	set numNodes [llength $indices]
    } elseif {[string equal $networkType "community"]} {	
    } elseif {[string equal $networkType "residue"]} {
    }

    # First line (second if "generic") used to determine the number of nodes
#    puts "numNodes = $numNodes"
    # Determine if the network file contains a matrix or a list of node pairs
    if {![regexp {,} $line matchVar]} {
	if {$numNodes == 0} {
	    set numNodes [llength $line]
	} elseif {[llength $line] != $numNodes} {
	    puts "Error: number of nodes ($numNodes) does not agree with adjacency matrix rank ([llength $line])"
	    return $success
	}
    }
    

    # Set up nodes2
    # accessed through network index (0 through numNetworks-1) and node index (0 through numNodes-1)
    # netIndex,coordinate,index: {x,y,z} location
    # netIndex,vmdIndex,index: VMD index
    # netIndex,chain,index: VMD chain name
    # netIndex,resid,index: VMD resid
    # netIndex,value,index: arbitrary value assigned to node (e.g. conservation, change in CPL)
    # netIndex,active,index: displayed or not
    # netIndex,color,index: display color
    # netIndex,objectIndex,index: OpenGL object index for deletion
    # netIndex,community,index: community number

    set coordinateList [list ]
    set vmdIndexList [list ]
    set chainList [list ]
    set residList [list ]
    set valueList [list ]
    set activeList [list ]
    set colorList [list ]
    set objectList [list ]
    set communityList [list ]

    for {set i 0} {$i < $numNodes} {incr i} {
	set index [lindex $indices $i]
	#set sel1 [atomselect $molid "residue $resnum and name $atomNames"]
	set sel1 [atomselect $molids($currentNetwork) "index $index"]
	#puts "residue $resnum and name $atomNames"
	set position [$sel1 get {x y z}]
	#set nodes($currentNetwork,$i,coordinate) [lindex $position 0]
	lappend coordinateList [lindex $position 0]
	#puts "  $index"
	#set nodes($currentNetwork,$i,vmdIndex) $index
	lappend vmdIndexList $index
	#set nodes($currentNetwork,$i,chain) $chain
	lappend chainList [$sel1 get chain]
	lappend residList [$sel1 get resid]
	#set nodes($currentNetwork,$i,value) 0
	lappend valueList 0
	#set nodes($currentNetwork,$i,active) 1
	lappend activeList 1
	#set nodes($currentNetwork,$i,color) blue
	lappend colorList blue
	lappend objectList -1
	lappend communityList -1
	$sel1 delete	
    }

    lappend nodes2 [list $coordinateList $vmdIndexList $chainList $residList $valueList $activeList $colorList $objectList $communityList $numNodes]
    set success 1

    return $success
}


# -------------------------------------------------------------------------
# Read a network file and create the data structure that carries edge information
# @param networkFilename Name of network file
# @return Whether reading in the edges was successful (1) or not (0)
proc ::NetworkView::createEdges { networkFilename } {

    variable networkType

    set networkFile [open $networkFilename "r"]
    set success 0

    # Skip first line if it contains an atomselection string
    if {[string equal $networkType "generic"]} {
	set line [gets $networkFile]
    }
    set line [gets $networkFile]
    close $networkFile

    # Determine if the network file contains a matrix or a list of node pairs
    if {[regexp {,} $line matchVar]} {
	set success [createEdgesFromNodeList $networkFilename]
#	puts "  createEdgesFromNodeList: $success"
    } else {
	set success [createEdgesFromMatrix $networkFilename]
#	puts "  createEdgesFromMatrix: $success"
    }
    
    return $success
}


# -------------------------------------------------------------------------
# Read a network file and create the data structure that carries edge information
# Format is a matrix of weights
# @param networkFilename Name of network file
# @return Whether reading in the edges was successful (1) or not (0)
proc ::NetworkView::createEdgesFromMatrix { networkFilename } {

    variable BIGNUM
    variable edges2
    variable networkType

    set networkFile [open $networkFilename "r"]
    set success 0

    # Skip first line if it contains an atomselection string
    if {[string equal $networkType "generic"]} {
	set line [gets $networkFile]
    }

    # Set up edges with weights
    #set edges($currentNetwork,maxweight) 0
    #set edges($currentNetwork,minweight) $BIGNUM
    #set edges($currentNetwork,maxbetweenness) 0
    #set edges($currentNetwork,minbetweenness) $BIGNUM
    #set edges($currentNetwork,maxvalue) 0
    #set edges($currentNetwork,minvalue) 0

    # netIndex,maxweight: maximum weight
    # netIndex,minweight: minimum weight
    # netIndex,weight,i1,i2: edge distance
    # netIndex,maxvalue: maximum value
    # netIndex,minvalue: minimum value
    # netIndex,value,i1,i2: arbitrary value assigned to edge (e.g. conservation, suboptimal path count)
    # netIndex,active,i1,i2: displayed or not
    # netIndex,color,i1,i2: display color
    # netIndex,objectIndex,i1,i2: OpenGL object index for deletion
    # netIndex,maxbetweenness: maximum betweenness
    # netIndex,minbetweenness: minimum betweenness
    # netIndex,betweenness,i1,i2: number of shortest paths crossing the edge

    # Set up edges2 with weights
    set maxWeight 0
    set minWeight $BIGNUM
    set maxBetweenness 0
    set minBetweenness $BIGNUM
    set maxValue 0
    set minValue 0
    set numEdges 0
    set weights [list ]
    set values [list ]
    set actives [list ]
    set colors [list ]
    set objectIndexs [list ]
    set betweennesss [list ]
    set lowerNodess [list ]
    set upperNodess [list ]

    set i 0
    while {![eof $networkFile]} {
	set line [gets $networkFile]

	set tempWeights [list ]
	set tempValues [list ]
	set tempActives [list ]
	set tempColors [list ]
	set tempObjectIndices [list ]
	set tempBetweennesses [list ]
	set tempLowerNodes [list ]
	set tempUpperNodes [list ]

	set j 0
	foreach weight $line {
	    if {$weight > 0} {
		if {$i < $j} {
		    lappend tempWeights $weight
		    if {$weight > $maxWeight} {
			set maxWeight $weight
		    }
		    if {$weight < $minWeight} {
			set minWeight $weight
		    }
		    lappend tempValues 0
		    lappend tempActives 1
		    lappend tempColors blue
		    lappend tempObjectIndices -1
		    lappend tempBetweennesses -1
		    lappend tempUpperNodes $j
		    incr numEdges
		} elseif {$i > $j} {
		    lappend tempLowerNodes $j
		}
	    }
	    incr j
	}
	lappend weights $tempWeights
	lappend values $tempValues
	lappend actives $tempActives
	lappend colors $tempColors
	lappend objectIndices $tempObjectIndices
	lappend betweennesses $tempBetweennesses
	lappend lowerNodes $tempLowerNodes
	lappend upperNodes $tempUpperNodes

	incr i
    }

    lappend edges2 [list $maxWeight $minWeight $weights $maxValue $minValue $values $actives $colors $objectIndices $maxBetweenness $minBetweenness $betweennesses $lowerNodes $upperNodes $numEdges]
    set success 1
    
    close $networkFile
    
    return $success
}


# -------------------------------------------------------------------------
# Read a network file and create the data structure that carries edge information
# Format uses node pairs and weights, e.g. "2, 4, 0.5432"
# @param networkFilename Name of network file
proc ::NetworkView::createEdgesFromNodeList { networkFilename } {

#    puts ">createEdgesFromNodeList"
    variable BIGNUM
    variable nodes2
    variable edges2
    variable networkType
    variable currentNetwork
    variable nodeCount

    set networkFile [open $networkFilename "r"]
    set success 0

    # Skip first line if it contains an atomselection string
    if {[string equal $networkType "generic"]} {
	set line [gets $networkFile]
    }

    # Read file into a list
    set nodePairs [list ]
    set firstNodes [list ]
    while {![eof $networkFile]} {
	set line [gets $networkFile]
	if {[regexp {(\d+), *(\d+), *([\d\.]+)} $line matchvar index1 index2 weight]} {
	    lappend nodePairs [list $index1 $index2 $weight]
	}
	#lappend firstNodes $index1
    }
    close $networkFile

    set nodePairs [lsort -integer -index 0 $nodePairs]
    #set firstNodes [lsort -integer -unique $firstNodes]

#    puts "nodePairs: $nodePairs"
    #puts "firstNodes: $firstNodes"

    # netIndex,maxweight: maximum weight
    # netIndex,minweight: minimum weight
    # netIndex,weight,i1,i2: edge distance
    # netIndex,maxvalue: maximum value
    # netIndex,minvalue: minimum value
    # netIndex,value,i1,i2: arbitrary value assigned to edge (e.g. conservation, suboptimal path count)
    # netIndex,active,i1,i2: displayed or not
    # netIndex,color,i1,i2: display color
    # netIndex,objectIndex,i1,i2: OpenGL object index for deletion
    # netIndex,maxbetweenness: maximum betweenness
    # netIndex,minbetweenness: minimum betweenness
    # netIndex,betweenness,i1,i2: number of shortest paths crossing the edge

    # Set up edges2 with weights
    set maxWeight 0
    set minWeight $BIGNUM
    set maxBetweenness 0
    set minBetweenness $BIGNUM
    set maxValue 0
    set minValue 0
    set numEdges 0
    set weights [list ]
    set values [list ]
    set actives [list ]
    set colors [list ]
    set objectIndexs [list ]
    set betweennesss [list ]
    set lowerNodess [list ]
    set upperNodess [list ]

    set numNodes [lindex $nodes2 $currentNetwork $nodeCount]
    #foreach node $firstNodes {}
    for {set i 0} {$i < $numNodes} {incr i} {
	set tempEdges [lsearch -index 0 -all -inline $nodePairs $i]
#	puts "tempEdges: $tempEdges"

	set tempWeights [list ]
	set tempValues [list ]
	set tempActives [list ]
	set tempColors [list ]
	set tempObjectIndices [list ]
	set tempBetweennesses [list ]
	set tempLowerNodes [list ]
	set tempUpperNodes [list ]

	foreach edge $tempEdges {
#	    puts "edge: $edge"
	    set index1 [lindex $edge 0]
	    set index2 [lindex $edge 1]
	    if {$index2 >= $numNodes} {
		puts "Error: number of nodes ($numNodes) exceeded by node ID ($index2)"
		# XXX - Clean up the nodes that have been created for this network
		set nodes2 [lreplace $nodes2 end end]
#		puts "<createEdgesFromNodeList"
		return $success
	    }
	    #set index1 [expr [lindex $edge 0] - 1]
	    #set index2 [expr [lindex $edge 1] - 1]
	    set weight [lindex $edge 2]
	    if {$index1 < $index2} {
		lappend tempWeights $weight
		if {$weight > $maxWeight} {
		    set maxWeight $weight
		}
		if {$weight < $minWeight} {
		    set minWeight $weight
		}
		lappend tempValues 0
		lappend tempActives 1
		lappend tempColors blue
		lappend tempObjectIndices -1
		lappend tempBetweennesses -1
		lappend tempUpperNodes $index2
		incr numEdges
	    }
	}

	lappend weights $tempWeights
	lappend values $tempValues
	lappend actives $tempActives
	lappend colors $tempColors
	lappend objectIndices $tempObjectIndices
	lappend betweennesses $tempBetweennesses
	lappend lowerNodes $tempLowerNodes
	lappend upperNodes $tempUpperNodes
    }

    lappend edges2 [list $maxWeight $minWeight $weights $maxValue $minValue $values $actives $colors $objectIndices $maxBetweenness $minBetweenness $betweennesses $lowerNodes $upperNodes $numEdges]
    set success 1
        
#    puts "<createEdgesFromNodeList"
    return $success
}


# -------------------------------------------------------------------------
# Set the current active network
# @param networkIndex index of network to activate
proc ::NetworkView::setCurrentNetwork { networkIndex } {

    variable currentNetwork
    variable networkCount

    if {$networkIndex < 0} {
	set currentNetwork -1
    } elseif {$networkIndex < $networkCount} {
	set currentNetwork $networkIndex
    } else {
	puts "Error: ::NetworkView::setCurrentNetwork - network index must be less than $networkCount"
    }

    return
}


# -------------------------------------------------------------------------
# Get a list of nodeIds for nodes at the interface of two subnetworks
# @param nodeIndices1 list of internal node indices
# @param nodeIndices2 list of internal node indices
# @return list of edgeIdPairs
proc ::NetworkView::getInterfaceEdges { nodeIndices1 nodeIndices2 } {

    set interfaceEdges [list ]

    set edgeIdPairs [getAllEdges [concat $nodeIndices1 $nodeIndices2]]

    foreach edgeIdPair $edgeIdPairs {
	if {[regexp {(\d+),(\d+)} $edgeIdPair matchVar node1 node2]} {
	    if {([lsearch -exact $nodeIndices1 $node1] != -1 &&
		 [lsearch -exact $nodeIndices2 $node2] != -1) ||
		([lsearch -exact $nodeIndices1 $node2] != -1 &&
		 [lsearch -exact $nodeIndices2 $node1] != -1) } {
		lappend interfaceEdges $edgeIdPair
#		puts "node1: $node1, node2: $node2"
		#puts -nonewline "$edgeIdPair "
	    }
	}
    }

    return [lsort -unique $interfaceEdges]
}


# -------------------------------------------------------------------------
# Get a list of edgeId pairs for edges connected to a given node
# @param nodeId internal node index
# @return list of edgeIdPairs
proc ::NetworkView::getEdgesByNodeId { nodeId } {

    variable currentNetwork
    variable edges
    variable edges2
    variable edgeLowerNodes
    variable edgeUpperNodes

    array set edgeIdPairs {}
    set i 0

    # List
    foreach index [lindex $edges2 $currentNetwork $edgeLowerNodes $nodeId] {
	set edgeIdPairs($i) "$index,$nodeId"
	#puts "lowerNodes: $index,$nodeId"
	incr i
    }

    foreach index [lindex $edges2 $currentNetwork $edgeUpperNodes $nodeId] {
	set edgeIdPairs($i) "$nodeId,$index"
	#puts "upperNodes: $nodeId,$index"
	incr i
    }
    
    #puts [array get edgeIdPairs]
    return [array get edgeIdPairs]
}


# -------------------------------------------------------------------------
# Retrieve all edgeIdPairs connected to any node in nodeIndices
# @param nodeIndices list of internal node indices
# @return list of edgeIdPairs
proc ::NetworkView::getAllEdges { nodeIndices } {

    set edgeIdPairs [list ]

    foreach nodeIndex $nodeIndices {
	array set idPairs [getEdgesByNodeId $nodeIndex]
	foreach {i edgeIdPair} [array get idPairs] {
	    lappend edgeIdPairs $edgeIdPair
	}
    }

    return [lsort -unique $edgeIdPairs]
}


# -------------------------------------------------------------------------
# Retrieve edgeIdPairs for edges that connect to any two nodes in nodeIndices
# @param nodeIndices list of internal node indices
# @return list of edgeIdPairs
proc ::NetworkView::getInternalEdges { nodeIndices } {

    set edgeIdPairs [list ]

    foreach nodeIndex $nodeIndices {
	array set idPairs [getEdgesByNodeId $nodeIndex]
	foreach {i edgeIdPair} [array get idPairs] {
	    # Check that both node IDs are within nodeIndices
	    regexp {(\d+),(\d+)} $edgeIdPair matchVar node1 node2
	    if { [lsearch -exact $nodeIndices $node1] != -1 && [lsearch -exact $nodeIndices $node2] != -1 } {
		lappend edgeIdPairs $edgeIdPair
	    }
	}
    }
    
    return [lsort -unique $edgeIdPairs]
}


# -------------------------------------------------------------------------
# Retrieve edgeIdPairs for edges that are below/above a given metric value
# @param metric edge associated value (weight, correlation, betweenness, value)
# @param side 0 - less than or equal to; 1 - greater than or equal to
# @param value number to be compared with
# @return list of edgeIdPairs
proc ::NetworkView::getEdgesByMetric { args } {

    variable currentNetwork
    variable nodes2
    variable nodeCount
    variable edges
    variable edges2
    variable edgeCount
    variable edgeWeight
    variable edgeValue
    variable edgeBetweenness
    variable edgeUpperNodes

    set edgeIdPairs [list ]
    set metric -1

    if {[llength $args] == 3} {
	set metricString [lindex $args 0]
	set side [lindex $args 1]
	set value [lindex $args 2]
    } else {
	puts "Error: ::NetworkView::getEdgesByMetric - wrong number of arguments"
	puts "  getEdgesByMetric metric side value"
	puts "    metric: edge associated value (weight, betweenness, value)"
	puts "    side: 0 - less than or equal to; 1 - greater than or equal to"
	puts "    value: number to be compared with"
	return
    }

    # Check for valid metric
    if {$metricString == "weight"} {
	set metric $edgeWeight
    } elseif {$metric != "value"} {
	set metric $edgeValue
    } elseif {$metric != "betweenness"} {
	set metric $edgeBetweenness
    } else {
	puts "Error: ::NetworkView::getEdgesByMetric - metric must be one of:"
	puts "    weight, betweenness, value (metric: $metric)"
	return
    }

    # List
    #set numEdges [lindex $edges2 $currentNetwork $edgeCount]
    set numNodes [lindex $nodes2 $currentNetwork $nodeCount]
    if {$side == 0} {
	for {set index1 0} {$index1 < $numNodes} {incr index1} {
	    set j 0
	    foreach x [lindex $edges2 $currentNetwork $metric $index1] {
		if {$x <= $value} {
		    set index2 [lindex $edges2 $currentNetwork $edgeUpperNodes $index1 $j]
		    lappend edgeIdPairs "$index1,$index2"
		}
		incr j
	    }
	}
    } elseif {$side == 1} {
	for {set index1 0} {$index1 < $numNodes} {incr index1} {
	    set j 0
	    foreach x [lindex $edges2 $currentNetwork $metric $index1] {
		if {$x >= $value} {
		    set index2 [lindex $edges2 $currentNetwork $edgeUpperNodes $index1 $j]
		    lappend edgeIdPairs "$index1,$index2"
		}
		incr j
	    }
	}
    } else {
	puts "Error: ::NetworkView::getEdgesByMetric - side must be 0 or 1 (side: $side)"
	return
    }

    return [lsort -unique $edgeIdPairs]
}


# -------------------------------------------------------------------------
# Retrieve nodeIndices from an atomselection
# @param selString atom selection string
# @return list of internal node indices
proc ::NetworkView::getNodesFromSelection { args } {

#    puts ">getNodesFromSelection"
    variable currentNetwork
    variable molids
    variable nodes2
    variable nodeVmdIndex

    set selString ""
    set nodeIndices [list ]

    if {[llength $args] == 1} {
       set selString [lindex $args 0]
    } else {
	puts "Error: ::NetworkView::getNodesFromSelection - wrong number of arguments"
	puts "  getNodesFromSelection selString"
	puts "    selString: atom selection string"
	return
    }

    if {[array exists molids] == 0 || [info exists molids($currentNetwork)] == 0} {
       puts "NetworkView Error) Current network ($currentNetwork) does not exist."
       puts "Have you loaded a network?"
       return
    }

    set sel [atomselect $molids($currentNetwork) $selString]
    set vmdIndices [$sel get index]

    # Get nodes from list
#    foreach index [lindex $nodes2 $currentNetwork $nodeVmdIndex] {
#	if {[lsearch -exact $vmdIndices $index] != -1} {
#	    lappend nodeIndices $index
#	}
#    }
    foreach vmdIndex $vmdIndices {
	set nodeIndex [lsearch -exact [lindex $nodes2 $currentNetwork $nodeVmdIndex] $vmdIndex]
	if {$nodeIndex != -1} {
	    lappend nodeIndices $nodeIndex
	}
    }

    #puts "nodeIndices: $nodeIndices"
    
    $sel delete

#    puts "<getNodesFromSelection"
    return $nodeIndices
}


# -------------------------------------------------------------------------
# Retrieve nodeIndices from a set of edges
# @param edgeIdPairs list of node pairs that identify specific edges
# @return list of internal node indices
proc ::NetworkView::getNodesFromEdgeList { edgeIdPairs } {

    variable currentNetwork
    variable molids
    variable nodes
    variable nodes2

    set nodeIndices [list ]

    foreach edgeIdPair $edgeIdPairs {
	set ids [split $edgeIdPair ","]
	lappend nodeIndices [lindex $ids 0] [lindex $ids 1]
    }

    return [lsort -integer -unique $nodeIndices]
}


# -------------------------------------------------------------------------
# Color edges based on their weights, betweennesses, or values
# @param lowVal low value
# @param highVal high value
proc ::NetworkView::colorBy { args } {
    
#    puts ">colorBy"
    variable currentNetwork
    variable edges
    variable edges2
    variable edgeMinWeight
    variable edgeMaxWeight
    variable edgeWeight
    variable edgeMinValue
    variable edgeMaxValue
    variable edgeValue
    variable edgeMinBetweenness
    variable edgeMaxBetweenness
    variable edgeBetweenness
    variable edgeColor

    set lowestValue 0
    set highestValue 0
    set colorData -1

    if {[llength $args] == 1} {
#	puts "  1 arg"
	set colorDataString [lindex $args 0]
	if {$colorDataString == "weight"} {
	    set colorDataString "Weight"
	} elseif {$colorDataString == "betweenness"} {
	    set colorDataString "Betweenness"
	} elseif {$colorDataString == "value"} {
	    set colorDataString "Value"
	} else {
	    puts "Error: ::NetworkView::colorBy - first argument must be weight, betweenness, or value"
	    puts "  first argument: $colorDataString"
	    return
	}
	# List
	set colorData [set edge${colorDataString}]
	set lowestValue [lindex $edges2 $currentNetwork [set edgeMin${colorDataString}]]
	set highestValue [lindex $edges2 $currentNetwork [set edgeMax${colorDataString}]]
    } elseif {[llength $args] == 3} {
#	puts "  3 args"
	set colorDataString [lindex $args 0]
	if {$colorDataString == "weight"} {
	    set colorDataString "Weight"
	} elseif {$colorDataString == "betweenness"} {
	    set colorDataString "Betweenness"
	} elseif {$colorDataString == "value"} {
	    set colorDataString "Value"
	} else {
	    puts "Error: ::NetworkView::colorBy - first argument must be weight, betweenness, or value"
	    puts "  first argument: $colorDataString"
	    return
	}
	set lowestValue [lindex $args 1]
	set highestValue [lindex $args 2]
	set colorData [set edge${colorDataString}]
    } elseif {[llength $args] != 1} {
	puts "Error: ::NetworkView::colorBy - wrong number of arguments"
	puts "  colorBy \[lowVal highVal\]"
	puts "    lowVal: lowest value in color spectrum"
	puts "    highVal: highest value in color spectrum"
	return	
    }

#    puts "lowVal,highVal: ($lowestValue,$highestValue)"
    set dataSpan [expr 1.0 * ($highestValue - $lowestValue)]
    set colorIdOffset 33.0
    set colorIdMax 1023.0

    # List
    set i 0
    foreach dataList [lindex $edges2 $currentNetwork $colorData] {
	set j 0
	foreach datum $dataList {
	    set unscaledValue $datum
	    if {$unscaledValue < $lowestValue} {
		set unscaledValue $lowestValue
	    } elseif {$unscaledValue > $highestValue} {
		set unscaledValue $highestValue
	    }
	    lset edges2 $currentNetwork $edgeColor $i $j [expr ((($unscaledValue - $lowestValue) / $dataSpan) * $colorIdMax) + $colorIdOffset]
	    incr j
	}
	incr i
    }

#    puts "<colorBy"
    
    return
}


# -------------------------------------------------------------------------
# Draw the activated network including both nodes and edges
# @param radiusParameter weight, correlation, betweenness, value, or global (default equal radii)
proc ::NetworkView::drawNetwork { args } {
    
#    puts ">drawNetwork"

    set radiusParam "equal"

    if {[llength $args] > 1} {
	puts "Error: ::NetworkView::drawNetwork - wrong number of arguments"
	puts "  drawNetwork \[radiusParameter\]"
	puts "    radiusParameter: \"weight\", \"correlation\", \"betweenness\", \"value\"; \"global\" (default equal radii)"
	return
    } elseif {[llength $args] == 1} {
	set radiusParam [lindex $args 0]
    }

    display update off
    drawNodes
    display update on
    drawEdges $radiusParam

#    puts "<drawNetwork"
    return
}


# -------------------------------------------------------------------------
# Draw activated nodes
# @param radiusParameter weight, correlation, betweenness, value, or global (default equal radii)
proc ::NetworkView::drawNodes {} {

#    puts ">drawNodes"

    variable currentNetwork
    variable nodes2
    variable nodeCount
    variable sphereRadius
    variable sphereResolution
    variable nodeObjectIndex
    variable nodeActive
    variable nodeColor
    variable molids
    variable nodeCoordinate
    #variable numNodes

    set radius $sphereRadius
    set resolution $sphereResolution

    set lastColor "UNSPECIFIED"

    # Color nodes in list
    set numNodes [lindex $nodes2 $currentNetwork $nodeCount]
    for {set index 0} {$index < $numNodes} {incr index} {
#       drawNode $i $radius $resolution
# Delete current node OpenGL object and redraw a new one if active

       # if OpenGL object already exists, delete
# objectIndex is local
       set objectIndex [lindex $nodes2 $currentNetwork $nodeObjectIndex $index]
       #puts "objectIndex: $objectIndex"
       if {$objectIndex != -1} {
          graphics $molids($currentNetwork) delete $objectIndex
# nodes2 is global
          lset nodes2 $currentNetwork $nodeObjectIndex $index -1
       }

# active is local
       set active [lindex $nodes2 $currentNetwork $nodeActive $index]
       if {$active == 1} {
          set nextColor [lindex $nodes2 $currentNetwork $nodeColor $index]

# we only want to do this if the color has actually changed
          if { $nextColor != $lastColor } {
             graphics $molids($currentNetwork) color $nextColor
             set lastColor $nextColor
          }
# coordinate is local
          set coordinate [lindex $nodes2 $currentNetwork $nodeCoordinate $index]
# objectIndex is (still) local
          set objectIndex [graphics $molids($currentNetwork) sphere $coordinate radius $radius resolution $resolution]
# nodes2 is global
          lset nodes2 $currentNetwork $nodeObjectIndex $index $objectIndex
       }

    }

#    puts "<drawNodes"
    return
}


# -------------------------------------------------------------------------
# Draw activated edges
# @param radiusParameter weight, correlation, betweenness, value, or global (default equal radii)
proc ::NetworkView::drawEdges { args } {

#    puts ">drawEdges"

    variable currentNetwork
    variable nodes2
    variable nodeCount
    variable edges
    variable edges2
    variable edgeMinWeight
    variable edgeMaxWeight
    variable edgeWeight
    variable edgeMinValue
    variable edgeMaxValue
    variable edgeValue
    variable edgeColor
    variable edgeUpperNodes
    variable cylinderRadius
    variable cylinderResolution

    set radius $cylinderRadius
    set resolution $cylinderResolution
    set radiusParam "equal"

    if {[llength $args] > 1} {
	puts "Error: ::NetworkView::drawEdges - wrong number of arguments"
	puts "  drawEdges \[radiusParameter\]"
	puts "    radiusParameter: \"weight\", \"correlation\", \"betweenness\", \"value\"; \"global\" (default equal radii)"
	return
    } elseif {[llength $args] == 1} {
	set radiusParam [lindex $args 0]
    }

#    puts "radiusParam: $radiusParam"

    ### List
    set numNodes [lindex $nodes2 $currentNetwork $nodeCount]
    if { $radiusParam == "value" ||
	 $radiusParam == "betweenness" } {
	set lowestRadius [lindex $edges2 $currentNetwork $edgeMinValue]
	set highestRadius [lindex $edges2 $currentNetwork $edgeMaxValue]
	for {set index1 0} {$index1 < $numNodes} {incr index1} {
	    set j 0
	    foreach value [lindex $edges2 $currentNetwork $edgeValue $index1] {
		set index2 [lindex $edges2 $currentNetwork $edgeUpperNodes $index1 $j]
		#set radius [expr ([lindex $edges2 $currentNetwork $edgeValue $index1 $index2] / 100.0) + 0.2]
		set radius [expr ($value / 100.0) + 0.2]
		drawEdge $index1 $index2 $radius $resolution
		incr j
	    }
	}
    } elseif {$radiusParam == "weight"} {
	set lowestRadius [lindex $edges2 $currentNetwork $edgeMinWeight]
	set highestRadius [lindex $edges2 $currentNetwork $edgeMaxWeight]
	for {set index1 0} {$index1 < $numNodes} {incr index1} {
	    set j 0
	    foreach weight [lindex $edges2 $currentNetwork $edgeWeight $index1] {
		set index2 [lindex $edges2 $currentNetwork $edgeUpperNodes $index1 $j]
		#set radius [expr 0.1 / [lindex $edges2 $currentNetwork $edgeWeight $index1 $index2]]
		set radius [expr 0.1 / $weight]
		drawEdge $index1 $index2 $radius $resolution
		incr j
	    }
	}	
    } elseif {$radiusParam == "correlation"} {
	for {set index1 0} {$index1 < $numNodes} {incr index1} {
	    set j 0
	    foreach weight [lindex $edges2 $currentNetwork $edgeWeight $index1] {
		set index2 [lindex $edges2 $currentNetwork $edgeUpperNodes $index1 $j]
		#set radius [expr exp(-[lindex $edges2 $currentNetwork $edgeWeight $index1 $index2]) * 1]
		set radius [expr exp(-$weight) * 1]
		drawEdge $index1 $index2 $radius $resolution
		incr j
	    }
	}	
    } else {
	for {set index1 0} {$index1 < $numNodes} {incr index1} {
	    #set j 0
	    foreach index2 [lindex $edges2 $currentNetwork $edgeUpperNodes $index1] {
		#set index2 [lindex $edges2 $currentNetwork $edgeUpperNodes $index1 $j]
		drawEdge $index1 $index2 $radius $resolution
		#incr j
	    }
	}	
    }
    
#    puts "<drawEdges"
    return
}


# -------------------------------------------------------------------------
# Delete current edge OpenGL object and redraw a new one if active
# @param index1 node index of first node
# @param index2 node index of second node
# @param radius number
# @param resolution number
proc ::NetworkView::drawEdge { index1 index2 radius resolution } {

    #puts ">drawEdge"
    #puts "($index1,$index2) radius: $radius, resolution: $resolution"
    variable currentNetwork
    variable molids
    variable nodes
    variable nodes2
    variable nodeCount
    variable nodeCoordinate
    variable edges
    variable edges2
    variable edgeActive
    variable edgeObjectIndex
    variable edgeColor
    variable edgeUpperNodes

    ### Edges List
    set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes $index1] $index2]
    #puts "tempIndex2: $tempIndex2"
    set objectIndex [lindex $edges2 $currentNetwork $edgeObjectIndex $index1 $tempIndex2]
    #puts "objectIndex: $objectIndex"
    if {$objectIndex >= 0} {
	#puts "  deleting"
	graphics $molids($currentNetwork) delete $objectIndex
    }
    
    ### Nodes List, Edges List
    #puts "edges2(upperNodes,$index1): [lindex $edges2 $currentNetwork $edgeUpperNodes $index1]"
    #puts "edges2(edgeActive,$index1,$tempIndex2): [lindex $edges2 $currentNetwork $edgeActive $index1 $tempIndex2]"
    if {$tempIndex2 != "" &&
	[lindex $edges2 $currentNetwork $edgeActive $index1 $tempIndex2] != 0} {
	graphics $molids($currentNetwork) color [lindex $edges2 $currentNetwork $edgeColor $index1 $tempIndex2]
	set coordinate1 [lindex $nodes2 $currentNetwork $nodeCoordinate $index1]
	set coordinate2 [lindex $nodes2 $currentNetwork $nodeCoordinate $index2]
	#puts "color: [lindex $edges2 $currentNetwork $edgeColor $index1 $tempIndex2]"
	#puts "coordinate1: $coordinate1"
	#puts "coordinate2: $coordinate2"
	#set graphicsId [graphics $molids($currentNetwork) cylinder $coordinate1 $coordinate2 radius $radius resolution $resolution]
	#puts "graphicsId: $graphicsId"
	lset edges2 $currentNetwork $edgeObjectIndex $index1 $tempIndex2 [graphics $molids($currentNetwork) cylinder $coordinate1 $coordinate2 radius $radius resolution $resolution]
    }
    
    #puts "<drawEdge"
    return
}


# -------------------------------------------------------------------------
# Globally change sphere radius
# @param radius number
proc ::NetworkView::setSphereRadius { radius } {

    variable sphereRadius
    set sphereRadius $radius

    return
}


# -------------------------------------------------------------------------
# Globally change sphere resolution
# @param resolution number
proc ::NetworkView::setSphereResolution { resolution } {

    variable sphereResolution
    set sphereResolution $resolution

    return
}


# -------------------------------------------------------------------------
# Globally change cylinder radius
# @param radius number
proc ::NetworkView::setCylinderRadius { radius } {

    variable cylinderRadius
    set cylinderRadius $radius

    return
}


# -------------------------------------------------------------------------
# globally change cylinder resolution
# @param resolution number
proc ::NetworkView::setCylinderResolution { resolution } {

    variable cylinderResolution
    set cylinderResolution $resolution

    return
}


# -------------------------------------------------------------------------
# Use nodeIndices to activate or deactivate a set of nodes
# @param nodeIndices list of internal node indices
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateNodes { args } {

    variable currentNetwork
    variable nodes
    variable nodes2
    variable nodeActive

    set nodeIndices [list ]
    set setOrUnset 1

    if {[llength $args] == 1} {
	set nodeIndices [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set nodeIndices [lindex $args 0]
	set setOrUnset [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateNodes - wrong number of arguments"
	puts "  activateNodes nodeIndices \[activateOrDeactivate\]"
	puts "    nodeIndices: list of internal node indices"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    ### List
    foreach index $nodeIndices {
	lset nodes2 $currentNetwork $nodeActive $index $setOrUnset
    }

    return
}


# -------------------------------------------------------------------------
# Use atomselection to activate or deactivate a set of nodes
# @param selString atom selection string
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateNodeSelection { args } {

    variable currentNetwork
    variable molids
    variable nodes
    variable nodes2
    variable nodeVmdIndex
    variable nodeActive
    variable nodeCount
    #variable numNodes

    set selString ""
    set setOrUnset 1

    if {[llength $args] == 1} {
	set selString [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set selString [lindex $args 0]
	set setOrUnset [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateNodeSelection - wrong number of arguments"
	puts "  activateNodeSelection selString \[activateOrDeactivate\]"
	puts "    selString: atom selection string"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    set nodeIndices [getNodesFromSelection $selString]
    activateNodes $nodeIndices $setOrUnset
    
    return
}


# -------------------------------------------------------------------------
# Use nodeIndices to deactivate a set of nodes
# @param nodeIndices list of internal node indices
proc ::NetworkView::deactivateNodes { nodeIndices } {

    activateNodes $nodeIndices 0

    return
}


# -------------------------------------------------------------------------
# Use atomselection to deactivate a set of nodes
# @param selectionString atom selection string
proc ::NetworkView::deactivateNodeSelection { selectionString } {

    activateNodeSelection $selectionString 0

    return
}


# -------------------------------------------------------------------------
# Use atomselection to color a set of nodes
# @param nodeIndices list of internal node indices
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorNodes { args } {

#    puts ">colorNodes"
    variable currentNetwork
    variable nodes
    variable nodes2
    variable nodeColor

    set nodeIndices [list ]
    set color 0

    if {[llength $args] == 1} {
	set nodeIndices [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set nodeIndices [lindex $args 0]
	set color [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::colorNodes - wrong number of arguments"
	puts "  colorNodes nodeIndices \[colorId\]"
	puts "    nodeIndices: list of internal node indices"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    ### List
    foreach nodeIndex $nodeIndices {
	lset nodes2 $currentNetwork $nodeColor $nodeIndex $color
    }

#    puts "<colorNodes"
    return
}


# -------------------------------------------------------------------------
# Use atomselection to color a set of nodes
# @param nodeIndices list of internal node indices
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorNodeSelection { args } {

#    puts ">colorNodeSelection"

    set selString ""
    set color 0

    if {[llength $args] == 1} {
	set selString [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set selString [lindex $args 0]
	set color [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::colorNodeSelection - wrong number of arguments"
	puts "  colorNodeSelection selString \[colorId\]"
	puts "    selString: selection string"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    set nodeIndices [getNodesFromSelection $selString]
    colorNodes $nodeIndices $color

#    puts "<colorNodeSelection"
    return
}


# -------------------------------------------------------------------------
# Use edgeIdPairs to activate or deactivate a set of edges
# @param edgeIdPairs list of node pairs that identify specific edges
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateEdges { args } {

    variable currentNetwork
    variable edges
    variable edges2
    variable edgeActive
    variable edgeUpperNodes

    set selString ""
    set edgeIdPairs [list ]
    set setOrUnset 1

    if {[llength $args] == 1} {
	set edgeIdPairs [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set edgeIdPairs [lindex $args 0]
	set setOrUnset [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateEdges - wrong number of arguments"
	puts "  activateEdges edgeIdPairs \[activateOrDeactivate\]"
	puts "    edgeIdPairs: list of node pairs that identify specific edges"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    foreach edgeIdPair $edgeIdPairs {	    
	#set edges($currentNetwork,$edgeIdPair,active) $setOrUnset
	# XXX - need to pass edgeIdPairs as 2-lists instead of strings
	set ids [split $edgeIdPair ","]
	set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes [lindex $ids 0]] [lindex $ids 1]]
	lset edges2 $currentNetwork $edgeActive [lindex $ids 0] $tempIndex2 $setOrUnset
    }

    return
}


# -------------------------------------------------------------------------
# Use nodeIndices to activate or deactivate a set of edges
# @param nodeIndices list of internal node indices
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateInternalEdges { args } {

    variable currentNetwork
    variable edges
    variable edges2
    variable edgeActive
    variable edgeUpperNodes

    set selString ""
    set nodeIndices [list ]
    set setOrUnset 1

    if {[llength $args] == 1} {
	set nodeIndices [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set nodeIndices [lindex $args 0]
	set setOrUnset [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateInternalEdges - wrong number of arguments"
	puts "  activateInternalEdges nodeIndices \[activateOrDeactivate\]"
	puts "    nodeIndices: list of internal node indices"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    if {$setOrUnset == 1} {
	set edgeIdPairs [getInternalEdges $nodeIndices]
    } elseif {$setOrUnset == 0} {
	set edgeIdPairs [getAllEdges $nodeIndices]
    } else {
	puts "Error ::NetworkView::activateInternalEdges: setOrUnset must be 0 or 1; setOrUnset: $setOrUnset"
	return
    }

    foreach edgeIdPair $edgeIdPairs {	    
	#set edges($currentNetwork,$edgeIdPair,active) $setOrUnset
	set ids [split $edgeIdPair ","]
	set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes [lindex $ids 0]] [lindex $ids 1]]
	lset edges2 $currentNetwork $edgeActive [lindex $ids 0] $tempIndex2 $setOrUnset
    }

    return
}


# -------------------------------------------------------------------------
# Use atomselection to activate or deactivate a set of edges
# @param selString: atom selection string
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateEdgeSelection { args } {

    set selString ""
    set setOrUnset 1

    if {[llength $args] == 1} {
	set selString [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set selString [lindex $args 0]
	set setOrUnset [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateEdgeSelection - wrong number of arguments"
	puts "  activateEdgeSelection selString internalOrExternal\[activateOrDeactivate\]"
	puts "    selString: selection string"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    set nodeIndices [getNodesFromSelection $selString]
    activateInternalEdges $nodeIndices $setOrUnset

    return
}


# -------------------------------------------------------------------------
# Use nodeIndices1 and nodeIndices2 to activate the edges between the two node sets
# @param nodeIndices1 list of internal node indices
# @param nodeIndices2 list of internal node indices
# @param activateOrDeactivate 0 to deactivate, 1 to activate (default)
proc ::NetworkView::activateInterfaceEdges { args } {

    variable currentNetwork
    variable edges
    variable edges2
    variable edgeActive
    variable edgeUpperNodes

    set setOrUnset 1

    if {[llength $args] == 2} {
	set nodeIndices1 [lindex $args 0]
	set nodeIndices2 [lindex $args 1]
    } elseif {[llength $args] == 3} {
	set selString [lindex $args 0]
	set setOrUnset [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::activateInterfaceEdges - wrong number of arguments"
	puts "  activateInterfaceEdges nodeIndices1 nodeIndices2 \[activateOrDeactivate\]"
	puts "    nodeIndices1: list of internal node indices"
	puts "    nodeIndices2: list of internal node indices"
	puts "    activateOrDeactivate: 0 to deactivate, 1 to activate (default)"
	return
    }

    set edgeIdPairs [getInterfaceEdges $nodeIndices1 $nodeIndices2]

    foreach edgeIdPair $edgeIdPairs {
	#set edges($currentNetwork,$edgeIdPair,active) $setOrUnset
	set ids [split $edgeIdPair ","]
	set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes [lindex $ids 0]] [lindex $ids 1]]
	lset edges2 $currentNetwork $edgeActive [lindex $ids 0] $tempIndex2 $setOrUnset
    }

    return
}


# -------------------------------------------------------------------------
# Use nodeIndices to deactivate a set of edges
# @param edgeIdPairs list of node pairs that identify specific edges
proc ::NetworkView::deactivateEdges { edgeIdPairs } {

    activateEdges $edgeIdPairs 0

    return
}

# -------------------------------------------------------------------------
# Use nodeIndices to deactivate a set of edges
# @param nodeIndices list of internal node indices
proc ::NetworkView::deactivateInternalEdges { nodeIndices1 nodeIndices2 } {

    activateInternalEdges $nodeIndices1 $nodeIndices2 0

    return
}


# -------------------------------------------------------------------------
# Use atomselection to deactivate a set of edges
# @param selectionString: atom selection string
proc ::NetworkView::deactivateEdgeSelection { selectionString } {

    activateEdgeSelection $selectionString 0

    return
}


# -------------------------------------------------------------------------
# Use nodeIndices1 and nodeIndices2 to deactivate the edges between the two node sets
# @param nodeIndices1 list of internal node indices
# @param nodeIndices2 list of internal node indices
proc ::NetworkView::deactivateInterfaceEdges { nodeIndices1 nodeIndices2 } {

    activateInternalEdges $nodeIndices1 $nodeIndices2 0

    return
}


# -------------------------------------------------------------------------
# Use nodeIndices to color a set of edges
# @param edgeIdPairs list of node pairs that identify specific edges
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorEdges { args } {

    variable currentNetwork
    variable edges
    variable edges2
    variable edgeColor
    variable edgeUpperNodes

    set edgeIdPairs [list ]
    set color 0

    if {[llength $args] == 1} {
	set edgeIdPairs [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set edgeIdPairs [lindex $args 0]
	set color [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::colorEdges - wrong number of arguments"
	puts "  colorEdges edgeIdPairs \[colorId\]"
	puts "    edgeIdPairs: list of node pairs that identify specific edges"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    foreach edgeIdPair $edgeIdPairs {	    
	#set edges($currentNetwork,$edgeIdPair,color) $color
	set ids [split $edgeIdPair ","]
	set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes [lindex $ids 0]] [lindex $ids 1]]
	lset edges2 $currentNetwork $edgeColor [lindex $ids 0] $tempIndex2 $color
    }

    return
}


# -------------------------------------------------------------------------
# Use nodeIndices to color a set of edges
# @param nodeIndices list of internal node indices
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorInternalEdges { args } {

#    puts ">colorInternalEdges"

    variable currentNetwork
    variable edges
    variable edges2
    variable edgeColor
    variable edgeUpperNodes

    set nodeIndices [list ]
    set color 0

    if {[llength $args] == 1} {
	set nodeIndices [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set nodeIndices [lindex $args 0]
	set color [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::colorInternalEdges - wrong number of arguments"
	puts "  colorInternalEdges nodeIndices \[colorId\]"
	puts "    nodeIndices: list of internal node indices"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    set edgeIdPairs [getInternalEdges $nodeIndices]

    foreach edgeIdPair $edgeIdPairs {	    
	#set edges($currentNetwork,$edgeIdPair,color) $color
	set ids [split $edgeIdPair ","]
	set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes [lindex $ids 0]] [lindex $ids 1]]
	#puts "edges2([lindex $ids 0],[lindex $ids 1]): $color"
	#puts "tempIndex2: $tempIndex2"
	lset edges2 $currentNetwork $edgeColor [lindex $ids 0] $tempIndex2 $color
    }

#    puts "<colorInternalEdges"
    return
}


# -------------------------------------------------------------------------
# Use atomselection to color a set of edges
# @param selString selection string
# @param colorId 0-blue (default), 1-red, ...
proc ::NetworkView::colorEdgeSelection { args } {

#    puts ">colorEdgeSelection"
    set selString ""
    set color 0

    if {[llength $args] == 1} {
	set selString [lindex $args 0]
    } elseif {[llength $args] == 2} {
	set selString [lindex $args 0]
	set color [lindex $args 1]
    } else {
	puts "Error: ::NetworkView::colorEdgeSelection - wrong number of arguments"
	puts "  colorEdgeSelection selString \[colorId\]"
	puts "    selString: selection string"
	puts "    colorId: 0-blue (default), 1-red, ..."
	return
    }

    set nodeIndices [getNodesFromSelection $selString]
    colorInternalEdges $nodeIndices $color

#    puts "<colorEdgeSelection"
    return
}


# -------------------------------------------------------------------------
# Get the VMD atom indices corresponding to a set of nodes
# @param nodeIndices list of internal node indices
proc ::NetworkView::getAtomIndices { nodeIndices } {

    variable currentNetwork
    variable nodes
    variable nodes2
    variable nodeVmdIndex

    set atomIndices [list ]

    ### List
    foreach nodeIndex $nodeIndices {
	set vmdIndex [lindex $nodes2 $currentNetwork $nodeVmdIndex $nodeIndex]
	lappend atomIndices $vmdIndex
	#puts -nonewline "$vmdIndex "
    }
    #puts ""

    return $atomIndices
}


# -------------------------------------------------------------------------
# Print VMD chain names corresponding to a set of nodes
# @param nodeIndices list of internal node indices
proc ::NetworkView::printChainNames { nodeIndices } {

    variable currentNetwork
    variable nodes
    variable nodes2
    variable nodeChain

    set chainNames [list ]

    ### List
    foreach nodeIndex $nodeIndices {
	if {[lsearch -exact $chainNames [lindex $nodes2 $currentNetwork $nodeChain $nodeIndex]] == -1} {
	    lappend chainNames [lindex $nodes2 $currentNetwork $nodeChain $nodeIndex]
	}
    }

    foreach chainName $chainNames {
	puts -nonewline "$chainName "
    }
    puts ""

    return
}

# -------------------------------------------------------------------------
# Print weights corresponding to each edge (ID pair)
# @param infoType information to retrieve; weight, betweenness, color, value
# @param edgeIdPairs list of node pairs that identify specific edges
# @return list of data with type infoType
proc ::NetworkView::getEdgeInfo { args } {

    variable currentNetwork
    variable edges
    variable edges2
    variable edgeWeight
    variable edgeBetweenness
    variable edgeColor
    variable edgeValue
    variable edgeUpperNodes

    set infoString "none"
    set infoType -1
    set edgeIdPairs [list ]
    set info [list ]

    if {[llength $args] == 2} {
      set infoString [lindex $args 0]
      set edgeIdPairs [lindex $args 1]
    } else {
      puts "Error: ::NetworkView::getEdgeInfo - wrong number of arguments"
      puts "  getEdgeInfo infoType edgeIdPairs"
      puts "    infoType: information to retrieve; weight, betweenness, color, value"
      puts "    edgeIdPairs: list of node pairs that identify specific edges"
      return
    }

    if {$infoString == "weight"} {
      set infoType $edgeWeight
    } elseif {$infoString == "betweenness"} {
      set infoType $edgeBetweenness
    } elseif {$infoString == "color"} {
      set infoType $edgeColor
    } elseif {$infoString == "value"} {
      set infoType $edgeValue
    } else {
      puts "Error ::NetworkView::getEdgeInfo: invalid infoType ($infoString)"
      return $info
    }

    foreach edgeIdPair $edgeIdPairs {
      set ids [split $edgeIdPair ","]
      set tempIndex2 [lsearch -exact [lindex $edges2 $currentNetwork $edgeUpperNodes [lindex $ids 0]] [lindex $ids 1]]
      set edgeInfo [lindex $edges2 $currentNetwork $infoType [lindex $ids 0] $tempIndex2]
      lappend info $edgeInfo
    }

    return $info
}

