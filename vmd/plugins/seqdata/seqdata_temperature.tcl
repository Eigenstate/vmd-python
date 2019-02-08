# University of Illinois Open Source License
# Copyright 2007 Luthey-Schulten Group, 
# All rights reserved.
#
# $Id: seqdata_temperature.tcl,v 1.5 2018/11/06 23:08:16 johns Exp $
# 
# Developed by: Luthey-Schulten Group
# 			     University of Illinois at Urbana-Champaign
# 			     http://faculty.scs.illinois.edu/schulten/
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the Software), to deal with 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to 
# do so, subject to the following conditions:
# 
# - Redistributions of source code must retain the above copyright notice, 
# this list of conditions and the following disclaimers.
# 
# - Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimers in the documentation 
# and/or other materials provided with the distribution.
# 
# - Neither the names of the Luthey-Schulten Group, University of Illinois at
# Urbana-Champaign, nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL 
# THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS WITH THE SOFTWARE.
#
# Author(s): Elijah Roberts

package provide seqdata 1.1
package require multiseqdialog 1.1

# Declare global variables for this package.
namespace eval ::SeqData::GrowthTemperature {

    # The data.
    variable dataMap
    
    # Gets the temperature class of the specified organism.
    # args:     taxonomyNode - The NCBI taxonomy node for the organism in question.
    # return:   The name of the organism's temperature class or an empty string if it is not known.
    proc getTemperatureClass {taxonomyNode} {
    
        variable dataMap
        
        # Load the data.
        loadDataFromMetadataRepository
    
        # See if we have an entry for this species.
        if {[info exists dataMap($taxonomyNode)]} {
            return $dataMap($taxonomyNode,class)
        }
         
        # The name could not be found, so return an empty string.
        return ""
    }
    
    proc loadDataFromMetadataRepository {} {
        
        variable dataMap
        
        # If the data has not yet been loaded, load it.
        if {![info exists dataMap]} {
            
            # Load the file from the metadata directory.
            set datadir [::MultiSeqDialog::getDirectory "metadata"]
            if {$datadir != "" && [file exists [set filename [file join $datadir "alltaxid.txt"]]]} {
                
                # Print the reference.
                printReference
                
                # Load the data.
                set records [loadFile $filename]
        
                # Output an informational message.  
                puts "MultiSeq Info) Loaded PGTdb data: $records entries."
            }
        }
    }

    #recordid				Tax_id				g_id				gname				name1				Nucleotide sequence				Protein sequence				Protein Structures				Temperature class				Growth Temperature				Optimal Growth Temperature
    #20				7				6				Azorhizobium				Azorhizobium caulinodans				52				170				0				mesophilic								30¢XC
    #21				9				32199				Buchnera				Buchnera aphidicola				419				933				0												
    #22				14				13				Dictyoglomus				Dictyoglomus thermophilum				13				23				1				thermophilic				70¢XC				
    proc loadFile {filename} {
        
        variable dataMap
        
        # Initialize the data map.
        array unset dataMap 
            
        # Open the nodes file.
        set fp [open $filename r]
        
        set records 0
        set firstRow 1
        while {![eof $fp] && [gets $fp line] >= 0} {
            
            if {$firstRow} {
                set firstRow 0
            } else {
                set columns [regexp -inline -all {\S+} $line]        
                
                # Go through the fields and save them into the map.
                if {[llength $columns] >= 5} {
                    set taxonomyId [lindex $columns 1]
                    if {![info exists dataMap($taxonomyId)]} {
                        set last3 [lindex $columns [expr [llength $columns]-3]]
                        set last2 [lindex $columns [expr [llength $columns]-2]]
                        set last1 [lindex $columns [expr [llength $columns]-1]]
                        if {$last3 == "psychrophilic" || $last3 == "mesophilic" || $last3 == "thermophilic" || $last3 == "hyperthermophilic"} {
                            incr records
                            set dataMap($taxonomyId) $taxonomyId
                            set dataMap($taxonomyId,class) $last3
                            set dataMap($taxonomyId,growth) $last2
                            set dataMap($taxonomyId,optimalGrowth) $last1
                        } elseif {$last2 == "psychrophilic" || $last2 == "mesophilic" || $last2 == "thermophilic" || $last2 == "hyperthermophilic"} {
                            incr records
                            set dataMap($taxonomyId) $taxonomyId
                            set dataMap($taxonomyId,class) $last2
                            set dataMap($taxonomyId,growth) $last1
                            set dataMap($taxonomyId,optimalGrowth) ""
                        } elseif {$last1 == "psychrophilic" || $last1 == "mesophilic" || $last1 == "thermophilic" || $last1 == "hyperthermophilic"} {
                            incr records
                            set dataMap($taxonomyId) $taxonomyId
                            set dataMap($taxonomyId,class) $last1
                            set dataMap($taxonomyId,growth) ""
                            set dataMap($taxonomyId,optimalGrowth) ""
                        }
                    }
                }
            }
        }
        
        # Close the file.
        close $fp
        
        return $records
    }
}
    
variable printedReference 0
proc printReference {} {
    
    variable printedReference
    
    # Print out the reference message.
    if {!$printedReference} {
        set printedReference 1
        puts "MultiSeq Reference) In any publication of scientific results based in part or"
        puts "MultiSeq Reference) completely on the use of PGTdb data, please cite:"
        puts "MultiSeq Reference) S.L. Huang, L.C. Wu, H.K. Liang, K.T. Pan, J.T. Horng, and M.T. Ko."
        puts "MultiSeq Reference) PGTdb: a database providing growth temperatures of prokaryotes."
        puts "MultiSeq Reference) Bioinformatics. 20:276, 2004."
    }
}

