/*****************************************************************************
*
*            Copyright (C) 2005 The Board of Trustees of the
*                        University of Illinois
*                         All Rights Reserved
*
*    This program is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License located in the COPYING file for more
*    details.
*
******************************************************************************/

/*****************************************************************************
* RCS INFORMATION:
*
*       $RCSfile: profcombine.cpp,v $
*       $Author: kvandivo $        $Locker:  $             $State: Exp $
*       $Revision: 1.2 $           $Date: 2011/01/18 22:06:00 $
*
******************************************************************************/

// $Id: profcombine.cpp,v 1.2 2011/01/18 22:06:00 kvandivo Exp $

#include <stdlib.h>
#include <stdio.h>
#include "symbol.h"
#include "alphabet.h"
#include "alphabetBuilder.h"
#include "sequence.h"
#include "alignedSequence.h"
#include "sequenceAlignment.h"
#include "fastaReader.h"
#include "fastaWriter.h"
#include "sequenceQR.h"
#include "profcombine.h"

/**
 * The name of the FASTA file that contains the first profile to combine.
 */
char *profile1Filename = NULL;

/**
 * The index of the key sequence in the first profile.
 */
int profile1KeyIndex = -1;

/**
 * The name of the FASTA file that contains the second profile to combine.
 */
char *profile2Filename = NULL;

/**
 * The index of the key sequence in the second profile.
 */
int profile2KeyIndex = -1;

/**
 * The name of the file to which the combined profile will be written.
 */
char *outputFilename = NULL;


/**
 * Entry point for the seqqr program. This program calculates an evolutionary profile given a file
 * containing aligned sequences in FASTA format.
 *
 * @param   argc    The argument count.
 * @param   argv    The argument values.
 */
int main(int argc, char** argv) {
   int i; 
    //Print the copyright notice.
    printCopyright(argc, argv);
    
    //Parse the arguments.
    if (!parseArgs(argc, argv)) {
        
        //If the arguments were not valid, print the usage and exit.
        printUsage(argc, argv);
        exit(1);
    }

    //Create an alphabet for reading the sequences.
    AlphabetBuilder* alphabetBuilder = new AlphabetBuilder();
    Alphabet* alphabet = alphabetBuilder->getProteinAlphabet();

    //Set the file to be read.    
    FASTAReader* reader1 = new FASTAReader(alphabet);
    reader1->setPath("");
    
    //Read the first profile.
    if (reader1->setFilename(profile1Filename) == 0) {
        printf("Error: file %s was not found.\n", profile1Filename);
        exit(1);
    }
    SequenceAlignment* profile1 = reader1->getSequenceAlignment();
    if (profile1 == NULL) {
        printf("Error: file %s was not a valid FASTA file.\n", profile1Filename);
        exit(1);
    }
    printf("Read profile from '%s': %d sequences of length %d\n", profile1Filename, profile1->getSequenceCount(), profile1->getLength());
    
    //Make sure all of the sequences are of the same length.
    int length = -1;
    if (profile1->getSequenceCount() > 0) length = profile1->getSequence(0)->getLength();
    for (i=1; i<profile1->getSequenceCount(); i++) {
        if (profile1->getSequence(i)->getLength() != length) {
            printf("Error: file %s did not contain a sequence alignment.\n", profile1Filename);
            exit(1);            
        }
    }
    
    //Read the second profile.
    FASTAReader* reader2 = new FASTAReader(alphabet);
    reader2->setPath("");
    if (reader2->setFilename(profile2Filename) == 0) {
        printf("Error: file %s was not found.\n", profile2Filename);
        exit(1);
    }
    SequenceAlignment* profile2 = reader2->getSequenceAlignment();
    if (profile2 == NULL) {
        printf("Error: file %s was not a valid FASTA file.\n", profile2Filename);
        exit(1);
    }
    printf("Read profile from '%s': %d sequences of length %d\n", profile2Filename, profile2->getSequenceCount(), profile2->getLength());
    
    //Make sure all of the sequences are of the same length.
    for (i=0; i<profile2->getSequenceCount(); i++) {
        if (profile2->getSequence(i)->getLength() != length) {
            printf("Error: file %s did not contain a profile that was compatible with profile 1.\n", profile2Filename);
            exit(1);            
        }
    }
    
    // Combine the two alignments contained in profile1 and profile2 into a new alignment named
    // combinedProfile using the sequences index by the variables profile1KeyIndex and
    // profile2KeyIndex respectively.
    
    SequenceAlignment* combinedProfile = new SequenceAlignment(profile1->getLength(), profile1->getSequenceCount()+profile2->getSequenceCount());
    
    //Write out the combined profile.
    FASTAWriter writer;
    writer.writeSequenceAlignment(outputFilename, combinedProfile, 60);
    printf("A combined profile has been created in '%s' with %d sequences\n", outputFilename, combinedProfile->getSequenceCount());

    //Free any used memory.
    delete alphabetBuilder;
    delete alphabet;
    delete reader1;
    delete reader2;
    delete combinedProfile;
}

/**
 * This functions parses the arguments passed in to the seqqr program. It stores the parsed values
 * in the appropriate global variables and returns whether the parsing was successful.
 *
 * @param   argc    The argument count.
 * @param   argv    The argument values.
 * @return  1 if the parsing was successful, otherwise 0.
 */
int parseArgs(int argc, char** argv) {
   int i; 
    //Parse any arguments.        
    for (i=1; i<argc; i++) {
        
        char *option = argv[i];
        
        //See if the user is trying to get help.
        if (strcmp(option, "-h") == 0 || strcmp(option, "--help") == 0) {
            return 0;
        }
        else if (strcmp(option, "-v") == 0 || strcmp(option, "--version") == 0) {
            exit(0);
        }
            
        //If this is the next to last argument and it is not an option, it must be the input filename.
        else if (i == argc-5) {
            profile1Filename = option;
        }
        else if (i == argc-4) {
            profile1KeyIndex = atoi(option);
        }
        else if (i == argc-3) {
            profile2Filename = option;
        }
        else if (i == argc-2) {
            profile2KeyIndex = atoi(option);
        }
        else if (i == argc-1) {
            outputFilename = option;
        }
            
        //This must be an invalid option.
        else {
            return 0;
        }
    }
    
    //Make sure we have a valid state for all of the variables.
    if (profile1Filename != NULL && profile1KeyIndex != -1 && profile2Filename != NULL && profile2KeyIndex != -1 && outputFilename != NULL)
        return 1;
    
    //Something must be invalid, return 0;
    return 0;
}

/**
 * This function prints the copyright notice.
 */
void printCopyright(int argc, char** argv) {
    printf("%s v%s\n", argv[0], VERSION);
    printf("Copyright (C) 2003-2011 The Board of Trustees of the University of Illinois.\n");
    printf("\n");
}


/**
 * This function prints the usage for the seqqr program.
 */
void printUsage(int argc, char** argv) {
    printf("usage: %s [OPTIONS] profile_file key_index profile_file key_index output_file\n", argv[0]);
    printf("\n");
    printf(" profile_file     A FASTA formatted file containing a profile to combine.\n");
    printf(" key_index        The index of the sequence in the preceding profile to use as\n");
    printf("                  the key sequence during the combining process.\n");
    printf(" output_file      The file to which the combined profile should be written.\n");
    printf("                  The file will be in FASTA format.\n");
    printf("\n");
    printf(" Where OPTIONS is one or more of the following:\n");
    printf("   -h --help                     Print this message.\n");
    printf("   -v --version                  Print the version and reference information.\n");
    printf("\n");
}

