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
*       $RCSfile: percid.cpp,v $
*       $Author: kvandivo $        $Locker:  $             $State: Exp $
*       $Revision: 1.2 $           $Date: 2011/01/18 22:06:00 $
*
******************************************************************************/

// $Id: percid.cpp,v 1.2 2011/01/18 22:06:00 kvandivo Exp $

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "symbol.h"
#include "alphabet.h"
#include "alphabetBuilder.h"
#include "sequence.h"
#include "alignedSequence.h"
#include "structureAlignmentReader.h"
#include "fastaReader.h"
#include "fastaWriter.h"
#include "percid.h"
#include "PIDTools.h"


/**
 * The name of the FASTA file that contains the alignment.
 */
char *fastaFilename = NULL;

/**
 * The name of the file to which the RMSD data will be written.
 */
char *outputFilename = NULL;

/**
 * If the RMSD per residue should be calculated instead of the pairwise RMSD.
 */
int perResidue = 0;

/**
 * If the molecules are RNA.
 */
int rna = 0;


/**
 * Entry point for the rmsd program. This program calculates the RMSD of the passed in structures.
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
    //AlphabetBuilder* alphabetBuilder = new AlphabetBuilder();
    Alphabet* alphabet;

    if (rna) {
      alphabet = AlphabetBuilder::createRnaAlphabet();
    } else {
      alphabet = AlphabetBuilder::createProteinAlphabet();
    }

    //Set the file to be read.    
    //FASTAReader* reader = new FASTAReader(alphabet);
    //reader->setPath("");

    /*if (reader->setFilename(fastaFilename) == 0) {
      printf("Error: file %s was not found.\n", fastaFilename);
      exit(1);
      }*/

    //Read the sequences.
    SequenceAlignment* alignment = ::FASTAReader::readSequenceAlignment(alphabet, fastaFilename);
    //SequenceAlignment* alignment = reader->getSequenceAlignment();

    if (alignment == NULL) {
        printf("Error: file %s was not a valid FASTA file.\n", fastaFilename);
        exit(1);
    }

    printf("Read alignment from '%s': %d sequences of length %d\n", fastaFilename, alignment->getNumberSequences(), alignment->getNumberPositions());
    
    //Make sure all of the sequences are of the same length.
    int length = -1;
    if (alignment->getNumberSequences() > 0) length = alignment->getSequence(0)->getSize();

    for (i=1; i<alignment->getNumberSequences(); i++) {
        if (alignment->getSequence(i)->getSize() != length) {
            printf("Error: file %s did not contain a sequence alignment.\n", fastaFilename);
            exit(1);            
        }
    }
    
    if (!perResidue) {
        FILE *out = fopen(outputFilename,"w");
        int count = alignment->getNumberSequences();
        int i,j;
        for (i=0; i<count; i++) {
	  fprintf(out,"%6.4f", (float)PIDTools::getPercentIdentity(alignment->getSequence(i), alignment->getSequence(0)));
	  //fprintf(out,"%6.4f", alignment->getSequence(i)->getPercentIdentity(alignment->getSequence(0)));
            for (j=1; j<count; j++) {
	      fprintf(out," %6.4f", (float)PIDTools::getPercentIdentity(alignment->getSequence(i), alignment->getSequence(j)));
	      //fprintf(out," %6.4f", alignment->getSequence(i)->getPercentIdentity(alignment->getSequence(j)));
            }
            fprintf(out,"\n");
        }
        fclose(out);
    }

    //Free any used memory.
    //delete alphabetBuilder;
    //delete reader;
    delete alphabet;
    return 0;
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
        else if (strcmp(option, "-p") == 0 || strcmp(option, "--per-residue") == 0) {
            perResidue = 1;
        }
        else if (strcmp(option, "-r") == 0 || strcmp(option, "--is-rna") == 0) {
            rna = 1;
        }
        //If this is the next to last argument and it is not an option, it must be the fasta filename.
        else if (i == argc-2) {
            fastaFilename = option;
        }
        //If this is the last argument and it is not an option, it must be the output filename.
        else if (i == argc-1) {
            outputFilename = option;
        }
        //This must be an invalid option.
        else {
            return 0;
        }
    }

    //Make sure we have a valid state for all of the variables.
    if (fastaFilename != NULL && outputFilename != NULL)
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
    return;
}


/**
 * This function prints the usage for the program.
 */
void printUsage(int argc, char** argv) {
    printf("usage: %s [OPTIONS] fasta_file output_file\n", argv[0]);
    printf("\n");
    printf(" fasta_file       A FASTA formatted file containing the aligned\n");
    printf("                  sequences.\n");
    printf(" output_file      The file to which the RMSD data should be written.\n");
    printf("\n");
    printf(" Where OPTIONS is one or more of the following:\n");
    printf("   -p --per-residue              Outputs the RMSD per residue instead of the\n");
    printf("                                 pairwise RMSDs.\n");
    printf("   -r --is-rna                   Specified that the molecules are RNA.\n");
    printf("   -h --help                     Print this message.\n");
    printf("   -v --version                  Print the version and reference information.\n");
    printf("\n");
    return;
}
