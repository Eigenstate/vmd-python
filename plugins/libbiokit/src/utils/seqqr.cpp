/*
 * University of Illinois Open Source License
 * Copyright 2007 Luthey-Schulten Group, 
 * All rights reserved.
 * 
 * Developed by: Luthey-Schulten Group
 * 			     University of Illinois at Urbana-Champaign
 * 			     http://www.scs.illinois.edu/~schulten
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the Software), to deal with 
 * the Software without restriction, including without limitation the rights to 
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 * of the Software, and to permit persons to whom the Software is furnished to 
 * do so, subject to the following conditions:
 * 
 * - Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimers.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimers in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * - Neither the names of the Luthey-Schulten Group, University of Illinois at
 * Urbana-Champaign, nor the names of its contributors may be used to endorse or
 * promote products derived from this Software without specific prior written
 * permission.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL 
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 * OTHER DEALINGS WITH THE SOFTWARE.
 *
 * Author(s): Elijah Roberts
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "symbol.h"
#include "alphabet.h"
#include "alphabetBuilder.h"
#include "sequence.h"
#include "alignedSequence.h"
#include "sequenceAlignment.h"
#include "fastaReader.h"
#include "fastaWriter.h"
#include "sequenceQR.h"
#include "aminoAcidQR.h"
#include "rnaQR.h"
#include "binaryQR.h"
#include "seqqr.h"

/**
 * The name of the FASTA file that contains the alignment of which to get an evolutionary proifle.
 */
char *inputFilename = NULL;

/**
 * The name of the file to which the evolutionary profile will be written.
 */
char *outputFilename = NULL;

/**
 * The percent identity at which to stop including sequences in the profile.
 */
float identityCutoff = 0.4;

/**
 * The percent of the sequences to include in the profile.
 */
int percentCutoff = -1;

/**
 * The number of sequences whose ordering should be preserved at the beginning of the matrix.
 */
int preserveCount = 0;

/**
 * If gap scaling should be performed.
 */
int performGapScaling = 1;

/**
 * The gap scale parameter used in scaling the gap values.
 */
float gapScaleParameter = 1.0;

/**
 * The norm order.
 */
float normOrder = 2;

/**
 * Use regular, nucleic, or binary version
 
 */
int type = 0;

 
/**
 * Entry point for the seqqr program. This program calculates an evolutionary profile given a file
 * containing aligned sequences in FASTA format.
 *
 * @param   argc    The argument count.
 * @param   argv    The argument values.
 */
int main(int argc, char** argv) {
    
    //Print the copyright notice.
    printCopyright(argc, argv);
    
    //Parse the arguments.
    if (!parseArgs(argc, argv)) {
        
        //If the arguments were not valid, print the usage and exit.
        printUsage(argc, argv);
        exit(1);
    }
    
    //Create an alphabet for reading the sequences.
    Alphabet* alphabet;
    if (type == 1)
        alphabet = AlphabetBuilder::createRnaAlphabet();
    else if (type == 3)
        alphabet = AlphabetBuilder::createDnaAlphabet();
    else
        alphabet = AlphabetBuilder::createProteinAlphabet();
    
    //Read the sequences.
    SequenceAlignment* alignment = FASTAReader::readSequenceAlignment(alphabet, inputFilename);
    if (alignment == NULL)
    {
        printf("Error: file %s was not found or is not a valid FASTA file.\n", inputFilename);
        exit(1);
    }
    printf("Read alignment from '%s': %d sequences of length %d.\n", inputFilename, alignment->getNumberSequences(), alignment->getNumberPositions());
    
    //Calculate the evolutionary profile.
    SequenceQR* sequenceQR = NULL;
    //Protein QR.
    if (type == 0) {
	    sequenceQR = new AminoAcidQR(alignment, preserveCount, performGapScaling, gapScaleParameter, normOrder);
        
    //RNA QR.
    } else if (type == 1) {
	    sequenceQR = new RnaQR(alignment, preserveCount, performGapScaling, gapScaleParameter, normOrder);
        
    //Binary QR.
    } else if (type == 2) {
	    sequenceQR = new BinaryQR(alignment, preserveCount, performGapScaling, gapScaleParameter, normOrder);
        
    // DNA QR.
    } else if (type == 3) {
	    sequenceQR = new RnaQR(alignment, preserveCount, performGapScaling, gapScaleParameter, normOrder);
    }
    
    //Perform the QR using the appropriate cutoff type.
    if (sequenceQR != NULL)
    {
        SequenceAlignment* profile;
        if(percentCutoff > 0) {
            profile = sequenceQR->qrWithPercentCutoff(percentCutoff);
        } else {
            profile = sequenceQR->qrWithPIDCutoff(identityCutoff);
        }
        
        //Write out the profile.
        FASTAWriter::writeSequenceAlignment(outputFilename, profile, 60);
        printf("An evolutionary profile has been created in '%s' with %d sequences.\n", outputFilename, profile->getNumberSequences());
        //delete profile;
        
    } else {
        printf("An unknown QR type was selected: %d\n", type);
    }
    
    delete sequenceQR;
    delete alignment;
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
        else if (strcmp(option, "-i") == 0 || strcmp(option, "--id-cutoff") == 0) {
            if (i < argc-1){
                float newIdentityCutoff = atof(argv[++i]);
                if (newIdentityCutoff < 1.0 || newIdentityCutoff > 100.0) {
                    printf("Error: identity cutoff value %0.2f is not valid. Value must be between 1\n", newIdentityCutoff);
                    printf("and 100.\n");
                    exit(0);
                }
                identityCutoff = newIdentityCutoff/100.0;
            } else {
                return 0;
            }
        }
        else if(strcmp(option, "-l") == 0 || strcmp(option, "--limit") == 0) {
            if (i < argc-1){
                int newPercentCutoff = atoi(argv[++i]);
                if (newPercentCutoff < 1 || newPercentCutoff > 100) {
                    printf("Error: percent cutoff value %d is not valid. Value must be between 1\n", newPercentCutoff);
                    printf("and 100.\n");
                    exit(0);
                }
                percentCutoff = newPercentCutoff;
            } else {
                return 0;
            }
        }
        else if (strcmp(option, "-p") == 0 || strcmp(option, "--preserve") == 0) {
            if (i < argc-1){
                preserveCount = atoi(argv[++i]);
                if (preserveCount < 0) return 0;
            } else {
                return 0;
            }
        }
        else if (strcmp(option, "-s") == 0 || strcmp(option, "--gap-scale") == 0) {
            if (i < argc-1){
                gapScaleParameter = atof(argv[++i]);
                if (gapScaleParameter == 0.0) return 0;
            } else {
                return 0;
            }
        }
        else if (strcmp(option, "-n") == 0 || strcmp(option, "--no-scaling") == 0) {
            performGapScaling = 0;
        }	
        else if (strcmp(option, "-o") == 0 || strcmp(option, "--order") == 0) {
            if (i < argc-1){
                normOrder = atof(argv[++i]);
                if (normOrder <= 0) {
                    printf("Error: norm order %0.2f is not valid. Value must be greater than 0.\n", normOrder);
                    printf("and 100.\n");
                    exit(0);
                }
            } else {
                return 0;
            }
        }
        else if(strcmp(option, "-a") == 0 || strcmp(option, "--protein") == 0) {	
            type = 0;
        }	
        else if(strcmp(option, "-r") == 0 || strcmp(option, "--rna") == 0) {
            type = 1;
        }
        else if(strcmp(option, "-b") == 0 || strcmp(option, "--binary") == 0) {	
            type = 2;
        }	
        else if(strcmp(option, "-d") == 0 || strcmp(option, "--dna") == 0) {
            type = 3;
        }
            
        //If this is the next to last argument and it is not an option, it must be the input filename.
        else if (i == argc-2) {
            inputFilename = option;
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
    if (inputFilename != NULL && outputFilename != NULL)
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
    printf("When publishing a work based in whole or in part on this program, please\n");
    printf("reference: Sethi, A., O'Donoghue, P. & Luthey-Schulten, Z. (2005) 'Evolutionary\n");
    printf("profiles from the QR factorization of multiple sequence alignments' Proc. Natl.\n");
    printf("Acad. Sci. USA 102, 4045-4050.\n");
    printf("\n");
}
/**
 * This function prints the usage for the seqqr program.
 */
void printUsage(int argc, char** argv) {
    printf("usage: %s [OPTIONS] input_file output_file\n", argv[0]);
    printf("\n");
    printf(" input_file       A FASTA formatted file containing the aligned\n");
    printf("                  sequences on which to perform the evolutionary\n");
    printf("                  profiling.\n");
    printf(" output_file      The file to which the evolutionary profile\n");
    printf("                  should be written. The file will be in FASTA\n");
    printf("                  format.\n");
    printf("\n");
    printf(" Where OPTIONS is one or more of the following:\n");
    printf("   -i value  --id-cutoff value   Stops the QR algorithm once there is [value]%%\n");
    printf("                                 sequence identity between any sequences in the\n");
    printf("                                 set. (default=40)\n");
    printf("   -l value  --limit value       Stops the QR algorithm once [value]%% of the\n");
    printf("                                 sequences have been ordered.\n");
    printf("   -p value  --preserve value    Preserves the ordering of the first [value]\n");
    printf("                                 sequences.\n");
    printf("   -s value  --gap-scale value   Sets the gap scaling parameter. (default=1.0)\n");
    printf("   -n        --no-scaling        Turns off gap scaling. Gaps will be value 1.\n");
    printf("   -o value  --order value       Sets the norm order to [value]. (default=2)\n");
    printf("   -a        --protein           Uses protein version of QR algorithm (default).\n");
    printf("   -b        --binary            Uses binary version of QR algorithm.\n");
    printf("   -r        --rna               Uses rna version of QR algorithm.\n");
    printf("   -d        --dna               Uses dna version of QR algorithm.\n");
    printf("   -h        --help              Print this message.\n");
    printf("   -v        --version           Print the version and reference information.\n");
    printf("\n");
}

