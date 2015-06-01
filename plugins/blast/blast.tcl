############################################################################
#cr
#cr            (C) Copyright 1995-2003 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################
#
# $Id: blast.tcl,v 1.8 2015/05/21 20:17:09 kvandivo Exp $
#
# Package for using blast 

package provide blast 1.11
package require seqdata 1.1
package require exectool

namespace eval ::Blast {

    # Export the package namespace.
    namespace export Blast

    # Directory to write temp files.
    global env
    variable tempDir ""
    if {[info exists env(TMPDIR)]} {
        set tempDir $env(TMPDIR)
    }

    # The prefix for temp files.
    variable filePrefix "blast"

    # The directories where the blast executables are located.
    variable blastProgramDir ""
    variable blastMatDir ""
    variable blastDBDir ""

    # The current CPU architecture.
    variable architecture ""

    # Whether or not the reference has yet been printed.
    variable printedReference 0

    # The level of verbosity, 0=silent, 1=errors, 2=errors+info
    variable verbosity 1

    # This method sets the temp file options used by the BLAST
    # args:     newTempDir - The new temp directory to use.
    #           newFilePrefix - The prefix to use for temp files.
    proc setTempFileOptions {newTempDir newFilePrefix} {

       variable tempDir
       variable filePrefix

       # Set the temp directory and file prefix.
       set tempDir $newTempDir
       set filePrefix $newFilePrefix
    }

    # This method sets the dir options used by BLAST
    proc setBlastProgramDirs {newBlastProgramDir newBlastMatDir newBlastDBDir} {

       variable blastProgramDir
       variable blastMatDir
       variable blastDBDir

       set blastProgramDir $newBlastProgramDir
       set blastMatDir $newBlastMatDir
       set blastDBDir $newBlastDBDir
    }

    # This method sets the arch options used by BLAST
    proc setArchitecture {newArchitecture} {

       variable architecture

       set architecture $newArchitecture
    }

# ----------------------------------------------------------------------    
    # Aligns the passed in sequences.
    # arg:      sequences - The list of sequence ids that should be aligned.
    # return:   The list of aligned sequences ids.
    proc searchDatabase {sequenceIDs database {cutoff 10.0} {iterations 1} {maxResults 500}} {

       variable tempDir
       variable filePrefix

#        puts "blast.tcl.searchDatabase seqIDs: $sequenceIDs, db: $database, cutoff: $cutoff, it: $iterations, results: $maxResults"

       # Delete any old files.
       foreach file [glob -nocomplain $tempDir/$filePrefix.*] {
            file delete -force $file
       }

       # Initialize the return list.
       set blastSequenceIDs {}

       # See if this a a normal search or a profile search.
       if {[llength $sequenceIDs] == 1} {

          # Figure out the input and output filenames.
          set inputFilename "$tempDir/$filePrefix.input"
          set outputFilename "$tempDir/$filePrefix.output"

          # Get the sequence id of the query sequence.
          set querySequenceID [lindex $sequenceIDs 0]

          # Save the sequence as a fasta file.
          SeqData::Fasta::saveSequences $querySequenceID $inputFilename

          # Run blast.
#            puts "blast.tcl.searchDatabase. seqIDs length = 1 'run $tempDir blastpgp -d $database -e $cutoff -i $inputFilename -o $outputFilename -m 6 -v $maxResults -b $maxResults -j $iterations'"
          set output [run $tempDir blastpgp -d $database -e $cutoff -i $inputFilename -o $outputFilename -m 6 -v $maxResults -b $maxResults -j $iterations]

          # Parse the output.
          set blastSequenceIDs [parseBlastOutput $querySequenceID $database $outputFilename]

       # This must be a profile search.
       } else {

          # Figure out the input and output filenames.
          set targetFilename "$tempDir/$filePrefix.input.fasta"
          set profileFilename "$tempDir/$filePrefix.input.aln"
          set outputFilename "$tempDir/$filePrefix.output"

          # Get the sequence id of the query sequence.
          set querySequenceID [lindex $sequenceIDs 0]

          # Save the reference sequence as a fasta file.
          SeqData::Fasta::saveSequences $querySequenceID $targetFilename

          # Save the profile.
          SeqData::Aln::saveSequences $sequenceIDs $profileFilename

          # Run blast.
#            puts "blast.tcl.searchDatabase. seqID legnth != 1 'run $tempDir blastpgp -d $database -e $cutoff -i $targetFilename -B $profileFilename -o $outputFilename -m 6 -v $maxResults -b $maxResults -j $iterations'"
          set output [run $tempDir blastpgp -d $database -e $cutoff -i $targetFilename -B $profileFilename -o $outputFilename -m 6 -v $maxResults -b $maxResults -j $iterations]

          # Parse the output.
          set blastSequenceIDs [parseBlastOutput $querySequenceID $database $outputFilename]
       }

       return $blastSequenceIDs
    }

# ----------------------------------------------------------------------    
   # Does a search with blastn -- nucleotide query against a nucleotide 
   # database
   # arg:      sequences - The list of sequence ids that should be aligned.
   # return:   The list of aligned sequences ids.
   proc searchDatabaseNucleotide {sequenceIDs database {cutoff 10.0} {iterations 1} {maxResults 500}} {

      variable tempDir
      variable filePrefix

      # Delete any old files.
      foreach file [glob -nocomplain $tempDir/$filePrefix.*] {
         file delete -force $file
      }

      # Initialize the return list.
      set blastSequenceIDs {}

      # See if this a a normal search or a profile search.
      if {[llength $sequenceIDs] == 1} {

         # Figure out the input and output filenames.
         set inputFilename "$tempDir/$filePrefix.input"
         set outputFilename "$tempDir/$filePrefix.output"

         # Get the sequence id of the query sequence.
         set querySequenceID [lindex $sequenceIDs 0]

         # Save the sequence as a fasta file.
         SeqData::Fasta::saveSequences $querySequenceID $inputFilename

         # Run blast.
         set output [run $tempDir blastall -p blastn -d $database -e $cutoff -i $inputFilename -o $outputFilename -m 6 -v $maxResults -b $maxResults]

         # Parse the output.
         set blastSequenceIDs [parseBlastOutput $querySequenceID $database $outputFilename 1]

      # This must be a profile search.
      } else {

         # Can't to profile search with blastall
         puts "You can't do a profile search using blastall for a nucleotide search"
         set blastSequenceIDs ""
      }

      return $blastSequenceIDs
   }

# ----------------------------------------------------------------------
   proc parseBlastOutput {querySequenceID database filename {nucleotide 0}} {

      # Initialize the data structures.
      set sequenceNames {}
      array set data {}
      set querySequenceName [SeqData::getName $querySequenceID]
      set data($querySequenceName,eScore) "query"
      set data($querySequenceName,startPosition) ""
      set data($querySequenceName,endPosition) ""
      set data($querySequenceName,sequence) {}
      set data($querySequenceName,lastSectionDone) 0

      # Open the file.
      set fp [open $filename r]

      # Find the last results section.
      set lastSectionLocation 0
      while {![eof $fp]} {

            # See if this is the start of the search results section.
            if {[string first "Results from round" [gets $fp]] != -1} {
                set lastSectionLocation [tell $fp]
            }
      }

      # Parse the last results section.
      seek $fp $lastSectionLocation
      set state 0
      set nuc_state 0
      set currentSection 0
      while {![eof $fp]} {
            if {[gets $fp line] >= 0 && [string length [string trim $line]] > 0} {

                # Trim whitespace from the front and back of the line.
                set line [string trim $line]

                # See if this is the start of the search results section.
                if {$state == 0 && [string first "Sequences producing significant alignments:" $line] == 0} {
                    set state 1

                # See if this is a search result line.
                # ref|NP_560418.1| translation elongation factor aEF-1 alpha subun...    48   9e-05
                # d1c2rb_ a.3.1.1 (B:) Cytochrome c2 {Rhodobacter capsulatus}           237   8e-64
                } elseif {$state == 1 && $nucleotide == 1} {
                   if {[regexp {^\s*$} $line unneeded] == 1} {
                      if {$nuc_state == 1} {
                         set state 3
                      }
                   } elseif {[regexp {(^\S+)} $line unneeded sequenceName] == 1} {
                      # puts "seq found: $sequenceName"
                      set nuc_state 1
                      lappend sequenceNames $sequenceName
                      set eScore [lindex [regexp -inline -all -- {\S+} $line] end]
                      if {[string index $eScore 0] == "e"} {
                          set eScore "1$eScore"
                      }
                      set data($sequenceName,eScore) $eScore
                      set data($sequenceName,startPosition) ""
                      set data($sequenceName,endPosition) ""
                      set data($sequenceName,sequence) {}
                      set data($sequenceName,lastSectionDone) 0
                   }
                } elseif {$state == 1 && [string first "QUERY" $line] == -1 && $line != "" && [string first "Sequences used in model and found again:" $line] == -1 && [string first "Sequences not found previously or not previously below threshold:" $line] == -1 && [string first "CONVERGED!" $line] == -1} {

                    # Parse out the name.
                    set sequenceName ""
                    if {[regexp {[^\|]*[\|]+([^\|\.\s]+)} $line unneeded sequenceName] == 1} {
                    } elseif {[regexp {(\S+)} $line unneeded sequenceName] == 1} {
                    }

                    # Add the name, if we could parse it.
                    if {$sequenceName != ""} {
                        lappend sequenceNames $sequenceName
                        set eScore [lindex [regexp -inline -all -- {\S+} $line] end]
                        if {[string index $eScore 0] == "e"} {
                            set eScore "1$eScore"
                        }
                        set data($sequenceName,eScore) $eScore
                        set data($sequenceName,startPosition) ""
                        set data($sequenceName,endPosition) ""
                        set data($sequenceName,sequence) {}
                        set data($sequenceName,lastSectionDone) 0
                    } else {
                        error "Could not parse BLAST output line (no seq name): <$line>"
                    }

                # See if this is the start of a query section.
                #QUERY     8   GSGLFDFXXXXXXXXXXXXXXXXXXPAGKVVVEEVVNIMGKD-VI-I-GTVESGMIG--- 61
                #NP_578636 2   --GLFDFlkrkevkeeekieilSkkPAGKVVVEEVVNIMGKD-VI-I-GTVESGMIG--- 53
                #1XE1_A    8   GSGLFDFlkrkevkeeekieilSkkPAGKVVVEEVVNIxGKD-VI-I-GTVESGxIG--- 61
                #d1c2rb_   1   GDAAKGEKEF-N-K-CKTCHSIIAPDGTEIV-KGA--K--TGPNLYGVVGRTAGTYPE-F 51
                #NP_14293      ------------------------------------------------------------
                } elseif {($state == 1 && [string first "QUERY" $line] == 0) || ($state == 2 && [string first "Database:" $line] == -1)} {
                    set state 2
                    if {[regexp {(\S+)\s+(\S+)\s+(\S+)\s+(\S+)} $line unneeded sequenceName startPosition sequence endPosition] == 1} {

                        # Map the sequence name to a name from the search results.
                        if {$sequenceName == "QUERY"} {
                           set sequenceName $querySequenceName
                           incr currentSection
                        } else {
                           foreach searchResultSequenceName $sequenceNames {
                              if {[string first $searchResultSequenceName $sequenceName] != -1} {
                                 set sequenceName $searchResultSequenceName
                                 break
                              }
                           }
                        }

                        # If we do not yet have an entry for this sequence, throw an error.
                        if {![info exists data($sequenceName,lastSectionDone)]} {
                           error "Could not parse BLAST output line (unknown sequence name1: $sequenceName): <$line>"
                        }

                        # Save the line's data, if it is not a duplicate match.
                        if {$data($sequenceName,lastSectionDone) < $currentSection} {
                           set data($sequenceName,lastSectionDone) $currentSection
                           if {$data($sequenceName,startPosition) == ""} {
                              set data($sequenceName,startPosition) $startPosition
                           }
                           set data($sequenceName,endPosition) $endPosition
                           set data($sequenceName,sequence) [concat $data($sequenceName,sequence) [split $sequence {}]]
                        }

                    } elseif {[regexp {(\S+)\s+0?\s*(\-+)} $line unneeded sequenceName sequence] == 1} {

                        # Map sequence name to a name from the search results.
                        foreach searchResultSequenceName $sequenceNames {
                            if {[string first $searchResultSequenceName $sequenceName] != -1} {
                                set sequenceName $searchResultSequenceName
                                break
                            }
                        }

                        # not yet have an entry for this sequence? throw error.
                        if {![info exists data($sequenceName,lastSectionDone)]} {
                            error "Could not parse BLAST output line (unknown sequence name2: $sequenceName): <$line>"
                        }

                        # Save the line's data, if it is not a duplicate match.
                        if {$data($sequenceName,lastSectionDone) < $currentSection} {
                            set data($sequenceName,lastSectionDone) $currentSection
                            set data($sequenceName,sequence) [concat $data($sequenceName,sequence) [split $sequence {}]]
                        }

                    } else {
                        error "Could not parse BLAST output line (didn't pass regex tests): <$line>"
                    }


                # See if we are through all of the query sections.
                } elseif {$state == 2 && [string first "Database:" $line] == 0} {
                    set state 3
                }
            }
      }
      close $fp

      # Add the blast results to the query as annotations.
      ::SeqData::addAnnotation $querySequenceID blast-e-score $data($querySequenceName,eScore)
      ::SeqData::addAnnotation $querySequenceID blast-start-position $data($querySequenceName,startPosition)
      ::SeqData::addAnnotation $querySequenceID blast-end-position $data($querySequenceName,endPosition)
      ::SeqData::addAnnotation $querySequenceID blast-alignment $data($querySequenceName,sequence)

      # Extract the sequences from the database.
      set blastSequenceIDs {}
      if {[llength $sequenceNames] > 0} {
            set blastSequenceIDs [extractSequencesFromDatabase $sequenceNames $database $nucleotide]

            # Add the blast results to the sequences.
            for {set i 0} {$i < [llength $blastSequenceIDs]} {incr i} {

                # Get the name and the sequence id.
                set sequenceName [lindex $sequenceNames $i]
                set blastSequenceID [lindex $blastSequenceIDs $i]

                # Add the blast data to the sequence as annotations.
                ::SeqData::addAnnotation $blastSequenceID blast-e-score $data($sequenceName,eScore)
                ::SeqData::addAnnotation $blastSequenceID blast-start-position $data($sequenceName,startPosition)
                ::SeqData::addAnnotation $blastSequenceID blast-end-position $data($sequenceName,endPosition)
                ::SeqData::addAnnotation $blastSequenceID blast-alignment $data($sequenceName,sequence)
                ::SeqData::addAnnotation $blastSequenceID notes "Retrieved from BLAST search with e score of $data($sequenceName,eScore)."
            }
      }

      return $blastSequenceIDs
   }

# ----------------------------------------------------------------------
    proc extractSequencesFromDatabase {sequenceNames database {nucleotide 0}} {

        variable verbosity
        variable tempDir
        variable filePrefix

        # Figure out the input and output filenames.
        set inputFilename "$tempDir/$filePrefix.fastacmd.input"
        set outputFilename "$tempDir/$filePrefix.fastacmd.output"

        # Write the names into a file so that the sequences can be extracted from the database.
        set fp [open $inputFilename w]
        foreach sequenceName $sequenceNames {
            puts $fp $sequenceName
        }
        close $fp        

        # Extract the sequence from the blast database.
        if {$nucleotide == 0} {
           set output [run $tempDir fastacmd -d $database -i $inputFilename -o $outputFilename]
        } elseif {$nucleotide == 1} {
           set output [run $tempDir fastacmd -d $database -pF -i $inputFilename -o $outputFilename]
        }

        # Load the sequences.
        set blastSequenceIDs [SeqData::Fasta::loadSequences $outputFilename]

        # Make sure we got the right number of sequences.        
        if {[llength $blastSequenceIDs] != [llength $sequenceNames]} {
            if {$verbosity >= 1} {
                puts "Blast Error) Fastacmd did not return a full set of sequences: [llength $blastSequenceIDs] of [llength $sequenceNames]"
            }
            return {}
        }

        # Make sure this is the sequence ordering is correct.
        #for {set i 0} {$i < [llength $sequenceNames]} {incr i} {
        #
        #    # Get the name and the sequence id.
        #    set sequenceName [lindex $sequenceNames $i]
        #    set blastSequenceID [lindex $blastSequenceIDs $i]
        #    if {[string first $sequenceName [::SeqData::getName $blastSequenceID]] == -1} {
        #        puts "Error) Fastacmd did not return a proper sequencing order: $sequenceName"
        #        return {}
        #    }
        #}

        return $blastSequenceIDs
    }

    proc run {wd program args} {

        global env
        variable verbosity
        variable blastProgramDir
        variable blastMatDir
        variable blastDBDir
        variable architecture

        # If this is windows, append .exe to the program name.
        if {$architecture == "WIN32"} {
            append program ".exe"
        }

        # If we have a blast program dir.
        set cmd ""
        if {$blastProgramDir != ""} {

            # Try to find the binary.
            if {[file exists [file join $blastProgramDir bin $program]]} {
                set cmd "::ExecTool::exec \"[file join $blastProgramDir bin $program]\""
            } elseif {[file exists [file join $blastProgramDir $program]]} {
                set cmd "::ExecTool::exec \"[file join $blastProgramDir $program]\""
            }

            # Set the environment variables.
            if {$blastMatDir != ""} {
                if {[file isdirectory $blastMatDir]} {
                    set env(BLASTMAT) $blastMatDir
                } elseif {[file isdirectory [file join $blastProgramDir $blastMatDir]]} {
                    set env(BLASTMAT) [file join $blastProgramDir $blastMatDir]
                }
            }
            if {$blastDBDir != ""} {
                if {[file isdirectory $blastDBDir]} {
                    set env(BLASTDB) $blastDBDir
                } elseif {[file isdirectory [file join $blastProgramDir $blastDBDir]]} {
                    set env(BLASTDB) [file join $blastProgramDir $blastDBDir]
                }
            }
        }

        # If we don't have a binary, just rely on the path.
        if {$cmd == ""} {
            set cmd "::ExecTool::exec $program"
        }

        # Append the arguments.
        foreach arg $args {
            append cmd " \"$arg\""
        }

        # Print out the reference message.
        printReference        

        # Run the command and then return to the working directory.
        if {$verbosity >= 2} {
            puts "Blast Info) Running $program with command $cmd"
        }
        set rc [catch {eval $cmd} out]
        if {$verbosity >= 2} {
            puts "Blast Info) $program returned with code $rc."
        }

        # If there was an error during execution, throw it.
        if {$rc != 0} {
            if {[string first "WARNING: posProcessAlignment: Alignment recovered successfully" $out] == -1} {
                error $out
            }
        }

        return $out
    }

   proc printReference {} {

      variable verbosity
      variable printedReference

      # Print out the reference message.
      if {!$printedReference} {
         set printedReference 1
         if {$verbosity >= 2} {
            puts "Blast Reference) In any publication of scientific results based in part or"
            puts "Blast Reference) completely on the use of the program BLAST, please reference:"
            puts "Blast Reference) Altschul, Stephen F., Thomas L. Madden, Alejandro A. Sch�ffer,"
            puts "Blast Reference) Jinghui Zhang, Zheng Zhang, Webb Miller, and David J. Lipman."
            puts "Blast Reference) Gapped BLAST and PSI-BLAST: a new generation of protein database"
            puts "Blast Reference) search programs. Nucleic Acids Res. 1997;25:3389-3402."
         }
      }
   }

}
