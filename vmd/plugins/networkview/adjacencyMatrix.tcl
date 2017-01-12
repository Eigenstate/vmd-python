############################################################################
#cr
#cr            (C) Copyright 1995-2013 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: adjacencyMatrix.tcl,v 1.7 2014/02/20 20:16:08 kvandivo Exp $
#

# Anurag Sethi, John Eargle
# March 2008 - June 2011
package provide networkview 1.41
package require psfgen 1.5

proc getAdjacencyMatrix { args } {
    global errorInfo errorCode
    set oldcontext [psfcontext new]  ;# new context
    set errflag [catch { eval ::AdjacencyMatrix::getAdjacencyMatrix $args } errMsg]
    set savedInfo $errorInfo
    set savedCode $errorCode
    psfcontext $oldcontext delete  ;# revert to old context
    if $errflag { error $errMsg $savedInfo $savedCode }
    return
}


namespace eval ::AdjacencyMatrix {

    namespace export AdjacencyMatrix
    
    variable molId
    #variable paramFileName
    variable outfilePrefix ""
    variable carmaFilename ""
    variable hasCarma 0

    variable distanceCutoff 4.5
    variable requiredOccupancy 0.75

    # systemSelString: atomselection string for all system atoms e.g. "P N G"
    set systemSelString ""
    # nodeSelString: atomselection string for atoms where nodes are centered;
    #   nodes are found using the intersection of nodeSelString and systemSelString
    set nodeSelString ""
    array set clusters {}
    set restrictions [list]

    set covariance ""

    # nodeMap: hash from an atom to the node it belongs to
    array set nodeMap {}

    # CM: contact map
    array set CM {}
    array set A {}
    array set Corr {}
    array set AbsCorr {}

}


# Create an adjacency matrix by filtering Carma output by residue-residue contact distance
# @param molId VMD molecule ID
# @param paramFileName Parameter file name
# @param outfilePrefix Prefix for output files
# @param carmaFilename Carma output file
# @param distanceCutoff Contact atoms must be within this distance
# @param requiredOccupancy Fraction of frames in which the contact must exist
proc ::AdjacencyMatrix::getAdjacencyMatrix { args } {
    
    variable molId
    #variable paramFileName
    variable outfilePrefix
    variable carmaFilename
    variable distanceCutoff
    variable requiredOccupancy
    variable systemSelString
    variable nodeSelString
    variable clusters
    variable nodeMap
    variable CM
    variable A
    variable Corr
    variable AbsCorr

#    puts [llength $args]
    parseArgs $args
    prepareNodeMap
#    puts "right after prepareNodeMap"
    initVars

#    puts "done with initVars"
    #set selString "($systemSelString) and ($nodeSelString)"
    #set selString "name P PG or (not chain N and name CA)"
    set sel [atomselect $molId $nodeSelString]
    set names [$sel get name]
    set resnames [$sel get resname]
    set residues [$sel get residue]
    set indices [$sel get index]
    set chains [$sel get chain]
    $sel delete
    set numres [llength $residues]
    set numFrames [molinfo $molId get numframes]
    array set localNodeIndices {}

    # Set up adjacency matrix
#    puts "starting to loop through $numres res"
    for {set i 0} {$i < $numres} {incr i} {
       set name [lindex $names $i]
       set resname [lindex $resnames $i]
       set residue [lindex $residues $i]
       set clusterSelString [getClusterSelString $name $resname $residue]
       set localSelString "(within $distanceCutoff of ($clusterSelString)) and (not ($clusterSelString))"
	#puts "localSelString: $localSelString"
       set localSel [atomselect $molId $localSelString]
	# Loop through trajectory frames
#       puts "before looping through $numFrames frames"
       for {set k 0} {$k < $numFrames} {incr k} {
          $localSel frame $k
          $localSel update
          set numIndices [llength [$localSel get index]]
	    #if {$num1 > 0} {
	    # Create array of node indices
          array unset localNodeIndices
          for {set j 0} {$j < $numIndices} {incr j} {
		#set res2 [lindex [$localSel get residue] $j]
             set index [lindex [$localSel get index] $j]
             set localNodeIndices($nodeMap($index)) 1
		#if {$res2 < [expr $residue - 1] || $res2 > [expr $residue + 1]} {
		#}
          }
	    #}
          set numNodes [llength [array names localNodeIndices]]
	    # Increase count for each node index found
          for {set j 0} {$j < $numNodes} {incr j} {
             set index [lindex [array names localNodeIndices] $j]
             incr CM($i,[lsearch -exact $indices $index])
          }
       }
       $localSel delete
    }
#    puts "done setting up adjacency matrix"

    # Apply restrictions
    applyRestrictions
#    puts "Done with restrictions"

    # Check that contact matrix is symmetric
#    puts "Check contact matrix symmetry"
    set errorCount 0
    for {set i 0} {$i < $numres} {incr i} {
	for {set j [expr $i+1]} {$j < $numres} {incr j} {
	    if {$CM($i,$j) != $CM($j,$i)} {
		puts "Error: CM($i,$j) = $CM($i,$j); CM($j,$i) = $CM($j,$i)"
		incr errorCount
	    }
	}
    }
    
    if {$errorCount > 0} {
	puts "Error: contact matrix (CM) has $errorCount assymetries"
    }

#    puts "Fill adjacency matrix with contacts that occur above the requiredOccupancy"
    # Fill adjacency matrix with contacts that occur above the requiredOccupancy
    buildAdjacencyMatrix

    # Check that adjacency matrix is symmetric
#    puts "Check adjacency matrix symmetry"
    set errorCount 0
    for {set i 0} {$i < $numres} {incr i} {
	for {set j [expr $i+1]} {$j < $numres} {incr j} {
	    if {$A($i,$j) != $A($j,$i)} {
		puts "Error: A($i,$j) = $A($i,$j); A($j,$i) = $A($j,$i)"
		incr errorCount
	    }
	}
    }
    
    if {$errorCount > 0} {
	puts "Error: adjacency matrix (A) has $errorCount assymetries"
    }

    # Write adjacency matrix out to file
    writeMatrices

    puts "<getAdjacencyMatrix"
    return
}


# Initialize global variables
proc ::AdjacencyMatrix::initVars {} {

    variable molId
    variable nodeSelString
    variable carmaFilename
    variable hasCarma
    variable covariance
    variable CM
    variable A
    variable Corr
    variable AbsCorr

#    puts "initVars: begin"
    set sel [atomselect $molId $nodeSelString]
    set residues [$sel get residue]
    $sel delete
    set numres [llength $residues]

#    puts "initVars: right before hasCarma: $hasCarma, carmaFilename: $carmaFilename"
    # Reading Covariance Matrix
    #set ip [open "carma.fitted.dcd.varcov.dat" "r"]
    if {$hasCarma} {
	set ip [open $carmaFilename "r"]
#    puts "initVars: ready to set"

# we now need to read in the file.  If the file is more than
# a gig the normal file read fails, so we will put the contents
# into the elements of an array, and build a single large variable
# once done

# this makes some assumptions about the input file, so it can't
# be used for everything.  It assumes non binary, and it also
# assumes than an extra space here and there (every billion bytes
# or so) won't mess things up.
    set bytesRead 0 

    while {[gets $ip line] >= 0} {
       incr bytesRead [string length $line]

       set arrPos [expr $bytesRead / 1000000000]
       append byteArray($arrPos) $line

    }
	close $ip


    set covariance ""
    set i 0
    while {$i <= $arrPos} {
       set covariance "$covariance $byteArray($i)"
       incr i
    }

    }
    
#    puts "initVars: getting ready to init adjacency matrix"
    # Initializing Adjacency matrix
    for {set i 0 } {$i < $numres} {incr i} {
	for {set j 0} {$j < $numres} {incr j} {
	    # CM: contact matrix
	    set CM($i,$j) 0
	    # A: adjacency matrix
	    set A($i,$j) 0
	    # Corr: correlation matrix
	    set Corr($i,$j) 0
	    # AbsCorr: |correlation| matrix
	    set AbsCorr($i,$j) 0
	}
    }

    return
}


# Write adjacency matrices to files
proc ::AdjacencyMatrix::writeMatrices {} {

    variable molId
    variable nodeSelString
    variable hasCarma
    variable outfilePrefix
    variable CM
    variable A
    variable Corr
    variable AbsCorr

    set sel [atomselect $molId $nodeSelString]
    set residues [$sel get residue]
    $sel delete
    set numres [llength $residues]    

#    puts "Write adjacency matrix out to file"
    
    if {$hasCarma} {
	set outfile1 [open "${outfilePrefix}.dat" w]
	set outfile2 [open "${outfilePrefix}.corr" w]
	set outfile3 [open "${outfilePrefix}.absCorr" w]
	set outfile4 [open "${outfilePrefix}.contact" w]
	puts $outfile1 $nodeSelString
	puts $outfile2 $nodeSelString
	puts $outfile3 $nodeSelString
	puts $outfile4 $nodeSelString
	for {set i 0} {$i < $numres} {incr i} {
	    for {set j 0} {$j < $numres} {incr j} {
		puts -nonewline $outfile1 "$A($i,$j) "
		puts -nonewline $outfile2 "$Corr($i,$j) "
		puts -nonewline $outfile3 "$AbsCorr($i,$j) "
		puts -nonewline $outfile4 "$CM($i,$j) "
	    }
	    puts $outfile1 ""
	    puts $outfile2 ""
	    puts $outfile3 ""
	    puts $outfile4 ""
	}
	close $outfile1
	close $outfile2
	close $outfile3
	close $outfile4
    } else {
	set outfile1 [open "${outfilePrefix}.dat" w]
	puts $outfile1 $nodeSelString
	for {set i 0} {$i < $numres} {incr i} {
	    for {set j 0} {$j < $numres} {incr j} {
		puts -nonewline $outfile1 "$CM($i,$j) "
	    }
	    puts $outfile1 ""
	}
	close $outfile1
    }

    return
}


# Remove restricted contacts from contact matrix
proc ::AdjacencyMatrix::applyRestrictions {} {

    variable molId
    variable nodeSelString
    variable restrictions
    variable CM

    set sel [atomselect $molId $nodeSelString]
    set names [$sel get name]
    set resnames [$sel get resname]
    set residues [$sel get residue]
    set indices [$sel get index]
    set chains [$sel get chain]
    $sel delete
    set numres [llength $residues]
    set numFrames [molinfo $molId get numframes]
    array set localNodeIndices {}

    foreach restriction $restrictions {
	puts "restriction: $restriction"
	if {[string equal $restriction "notSameResidue"]} {
	    for {set i 0} {$i < $numres} {incr i} {
		set residue1 [lindex $residues $i]
		for {set j $i} {$j < $numres} {incr j} {
		    set residue2 [lindex $residues $j]
		    if {$residue1 == $residue2} {
			set CM($i,$j) 0
			set CM($j,$i) 0
		    }
		}
	    }
	} elseif {[string equal $restriction "notNeighboringResidue"]} {
	    for {set i 0} {$i < $numres} {incr i} {
		set residue1 [lindex $residues $i]
		set chain1 [lindex $chains $i]
		for {set j $i} {$j < $numres} {incr j} {
		    set residue2 [lindex $residues $j]
		    set chain2 [lindex $chains $j]
		    if {($residue1 == [expr $residue2 + 1] ||
			 $residue1 == [expr $residue2 - 1]) &&
			$chain1 == $chain2} {
			set CM($i,$j) 0
			set CM($j,$i) 0
		    }
		}
	    }
	} elseif {[string equal $restriction "notNeighboringCAlpha"]} {
	    for {set i 0} {$i < $numres} {incr i} {
		set residue1 [lindex $residues $i]
		set chain1 [lindex $chains $i]
		set name1 [lindex $names $i]
		for {set j $i} {$j < $numres} {incr j} {
		    set residue2 [lindex $residues $j]
		    set chain2 [lindex $chains $j]
		    set name2 [lindex $names $j]
		    if {($residue1 == [expr $residue2 + 1] ||
			 $residue1 == [expr $residue2 - 1]) &&
			$chain1 == $chain2 &&
			[string equal $name1 "CA"] &&
			[string equal $name2 "CA"]} {
			puts "  residue1: $residue1, residue2: $residue2"
			puts "  chain1: $chain1, chain2: $chain2"
			puts "  name1: $name1, name2: $name2"
			set CM($i,$j) 0
			set CM($j,$i) 0
		    }
		}
	    }
	} elseif {[string equal $restriction "notNeighboringPhosphate"]} {
	    for {set i 0} {$i < $numres} {incr i} {
		set residue1 [lindex $residues $i]
		set chain1 [lindex $chains $i]
		set name1 [lindex $names $i]
		if {[string equal $name1 "P"]} {
		    for {set j $i} {$j < $numres} {incr j} {
			set residue2 [lindex $residues $j]
			set chain2 [lindex $chains $j]
			set name2 [lindex $names $j]
			if {($residue1 == [expr $residue2 + 1] ||
			     $residue1 == [expr $residue2 - 1]) &&
			    $chain1 == $chain2 &&
			    [string equal $name2 "P"]} {
			    set CM($i,$j) 0
			    set CM($j,$i) 0
			}
		    }
		}
	    }
	} else {
	    puts "Error: unknown restriction ($restriction)."
	    puts "       Continuing with this restriction ignored."
	}
    }

    return
}


# Build adjacency matrix
proc ::AdjacencyMatrix::buildAdjacencyMatrix {} {

    variable molId
    variable nodeSelString
    variable requiredOccupancy
    variable covariance
    variable hasCarma
    variable CM
    variable A
    variable Corr
    variable AbsCorr    

    set sel [atomselect $molId $nodeSelString]
    set residues [$sel get residue]
    $sel delete
    set numres [llength $residues]
    set numFrames [molinfo $molId get numframes]
    set minFrames [expr $requiredOccupancy * $numFrames]

    if {$hasCarma} {
	for {set i 0} {$i < $numres} {incr i} {
	    for {set j 0} {$j < $numres} {incr j} {
		if {$CM($i,$j) >= $minFrames} {
		    set A($i,$j) [expr -log(abs([lindex $covariance [expr $i*$numres + $j]]))]
		    set Corr($i,$j) [lindex $covariance [expr $i*$numres + $j]]
		    set AbsCorr($i,$j) [expr abs([lindex $covariance [expr $i*$numres + $j]])]
		    set CM($i,$j) 1
		} else {
		    set CM($i,$j) 0
		}
	    }
	}
    } else {
	for {set i 0} {$i < $numres} {incr i} {
	    for {set j 0} {$j < $numres} {incr j} {
		if {$CM($i,$j) >= $minFrames} {
		    set A($i,$j) 1
		    set Corr($i,$j) 1
		    set AbsCorr($i,$j) 1
		    set CM($i,$j) 1
		} else {
		    set CM($i,$j) 0
		}
	    }
	}
    }

    return
}


# Loops through all node atoms, gets their atom clusters, and builds a nodeMap
proc ::AdjacencyMatrix::prepareNodeMap {} {

    puts ">prepareNodeMap"

    variable molId
    variable systemSelString
    variable nodeSelString
    variable clusters
    variable nodeMap

    array unset nodeMap

    # Loop through all node atoms, get their associated clusters, and build nodeMap
    #set selString "($systemSelString) and ($nodeSelString)"
    puts "nodeSelString: $nodeSelString"
    set sel [atomselect $molId $nodeSelString]
    set residues [$sel get residue]
    set indices [$sel get index]
    set names [$sel get name]
    set resnames [$sel get resname]
    $sel delete
    set numres [llength $residues]
    set clusterSelString ""
    
    set logFile [open "loggy.log" "w"]

    for {set i 0 } {$i < $numres} {incr i} {
	set residue [lindex $residues $i]
	set name [lindex $names $i]
	set resname [lindex $resnames $i]
	set index [lindex $indices $i]
	set clusterSelString [getClusterSelString $name $resname $residue]
	puts "clusterSelString: $clusterSelString"
	puts $logFile "clusterSelString: $clusterSelString"
	set clusterSel [atomselect $molId $clusterSelString]
	set clusterIndices [$clusterSel get index]
	set numClusterAtoms [llength $clusterIndices]
	for {set j 0} {$j < $numClusterAtoms} {incr j} {
	    set clusterIndex [lindex $clusterIndices $j]
	    if {[llength [array get nodeMap $clusterIndex]] != 0} {
		puts [array get nodeMap $clusterIndex]
		#puts "Error: atom $clusterIndex already mapped to node $index"
	    }
	    set nodeMap($clusterIndex) $index
	    #puts "nodeMap($clusterIndex) = $index"
	    puts -nonewline "$clusterIndex "
	    puts -nonewline $logFile "$clusterIndex "
	}
	puts ""
	puts $logFile ""
    }

    close $logFile

    puts "<prepareNodeMap"
    return
}


# Get the string corresponding to an atomselection for a given node atom
# @param name Atomselection atom name (e.g. CA, P)
# @param resname Atomselection residue name (e.g. ALA, GUA)
# @param residue Atomselection residue number
# @return String that determines the atomselection for a cluster
proc ::AdjacencyMatrix::getClusterSelString {name resname residue} {

    variable clusters

    set clusterSelString ""

    if {[llength [array get clusters "$name,$resname"]] == 0} {
	set clusterSelString "residue $residue"
    } else {
	set clusterSelString "residue $residue and $clusters($name,$resname)"
    }

    return $clusterSelString
}


# Get line from parameter file
# @param paramFile File containing parameters for the setup of an adjacency matrix.
# @return Next nonempty line in paramFile.
proc ::AdjacencyMatrix::getParameterLine {paramFile} {

    set line [gets $paramFile]
    puts "line: $line"
    
    if {![string equal $line ""]} {
	set psfFileName $line
	puts $line
	return $line
    } else {
	puts "Error: empty line in parameter file"
	return
    }
    
    return line
}


# Get set of lines from parameter file
# @param paramFile File containing parameters for the setup of an adjacency matrix.
# @return List of next nonempty lines in paramFile.
proc ::AdjacencyMatrix::getParameterLines {paramFile} {

    set line [gets $paramFile]
    set lines []

    #while {![eof $paramFile] && [regexp {>} $line] == 0} {}
    while {![eof $paramFile] && ![string equal $line ""]} {
	lappend lines $line
	puts $line
	set line [gets $paramFile]
    }
            
    return $lines
}


# Read in parameter file
proc ::AdjacencyMatrix::readParamFile { paramFileName } {

    variable hasCarma
    variable systemSelString
    variable nodeSelString
    variable clusters
    variable restrictions
    
    set paramFile [open $paramFileName "r"]
    
    set line [gets $paramFile]
    while {![eof $paramFile]} {
	if {[regexp {>Dcds} $line]} {
	    set hasCarma 1
	} elseif {[regexp {>SystemSelection} $line]} {
	    set systemSelString [getParameterLine $paramFile]
	} elseif {[regexp {>NodeSelection} $line]} {
	    set selString [getParameterLine $paramFile]
	} elseif {[regexp {>Clusters} $line]} {
	    while {![eof $paramFile] && [regexp {>} $line] == 0} {}
	    set clusterStrings [getParameterLines $paramFile]
	    foreach clusterString $clusterStrings {
		if {[regexp {(\S+) +(\S+) +(.+)} $clusterString temp name resname clusterSelString]} {
		    set clusters($name,$resname) $clusterSelString
		}
	    }	    
	} elseif {[regexp {>Restrictions} $line]} {
	    set restrictions [getParameterLines $paramFile]
	}
	set line [gets $paramFile]
    }
    close $paramFile
    
    # Set up nodeSelString
    if {![string equal $systemSelString ""] &&
	![string equal $selString ""]} {
	set nodeSelString "($systemSelString) and ($selString)"
    } else {
	puts "Error: not possible to build nodeSelString: ($systemSelString) and ($selString)"
	return
    }
    
    return
}


# Parse the commandline arguments
# @param molId VMD molecule ID
# @param paramFileName Parameter file name
# @param outfilePrefix Prefix for output files
# @param carmaFilename Carma output file
# @param distanceCutoff Contact atoms must be within this distance
# @param requiredOccupancy Fraction of frames in which the contact must exist
proc ::AdjacencyMatrix::parseArgs { params } {

    variable molId
    variable hasCarma
    #variable paramFileName
    variable outfilePrefix
    variable carmaFilename
    variable distanceCutoff
    variable requiredOccupancy
    variable systemSelString
    variable nodeSelString
    variable clusters
    variable restrictions
 
    if {[llength $params] < 3} {
	getAdjacencyMatrixUsage
        error "Error: two few arguments ([llength $params])"
    }
    
    # Parse the arguments.
    set molId [lindex $params 0]
    set paramFileName [lindex $params 1]
    set outfilePrefix [lindex $params 2]

    # Read in parameter file
    array unset clusters
    set restrictions [list]
    readParamFile $paramFileName
   
    if {$hasCarma} {
	if {[llength $params] > 6} {
	    getAdjacencyMatrixUsage
	    error "Error: too many arguments ([llength $params])"
	}
	set carmaFilename [lindex $params 3]	
	if {[llength $params] > 4} {
	    set distanceCutoff [lindex $params 4]
	}
	if {[llength $params] > 5} {
	    set requiredOccupancy [lindex $params 5]
	}
    } else {	
	if {[llength $params] > 5} {
	    getAdjacencyMatrixUsage
	    error "Error: too many arguments ([llength $params])"
	}
	if {[llength $params] > 3} {
	    set distanceCutoff [lindex $params 3]
	}
	if {[llength $params] > 4} {
	    set requiredOccupancy [lindex $params 4]
	}
    }

    return
}


# Prints the usage information
proc ::AdjacencyMatrix::getAdjacencyMatrixUsage {} {
    
 	puts "  Creates a network based on a contact distance cutoff and a molecular"
	puts "  dynamics trajectory."
	puts ""
        puts "  Usage: getAdjacencyMatrix molId paramFileName outfilePrefix \[carmaFilename\] \[distanceCutoff \[requiredOccupancy\]\]"
	puts "    molId: VMD molecule ID"
	puts "    paramFileName: parameter file of the form:"
	puts "      >Psf"
	puts "      >Dcds"
	puts "      >SystemSelection"
	puts "      >NodeSelection"
	puts "      >Clusters"
	puts "      >Restrictions"
	puts "    outfilePrefix: prefix used for output file names"
	puts "    carmaFilename: final correlation file generated by carma, usually"
	puts "      carma.fitted.dcd.varcov.dat; required if a dcd is being read in"
	puts "    distanceCutoff: threshold below which contacts are counted (default 4.5)"
	puts "    requiredOccupancy: fraction of the trajectory that a contact must exist"
	puts "      (default 0.75)"
   
    return
}
