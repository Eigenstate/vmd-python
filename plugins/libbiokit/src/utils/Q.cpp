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
#include "Atom.h"
#include "Residue.h"
#include "Contact.h"
#include "ContactList.h"
#include "ContactTools.h"
#include "structure.h"
#include "pdbReader.h"
#include "Q.h"

/**
 * The function being performed.
 */
int function = -1;

/**
 * The number of structure arguments passed in.
 */
int numberStructureArguments = 0;

/**
 * The structure arguments.
 */
char** structureArguments = NULL;

/**
 * The maximum distance between two residues from them to be considered in contact.
 */
double contactDistance = 5.0;

/**
 * The minimum separation in the sequence for two residues to be considered native.
 */
int minSequenceDistance = 4;

/**
 * The maximum separation in the sequence for two residues to be considered native.
 */
int maxSequenceDistance = -1;

/**
 * The minimum separation in the sequence for two residues to be considered native.
 */
double maxDistanceDeviation = 1.0;

/**
 * The file to output the data to.
 */
char* outputFilename = NULL;

/**
 The first frame.
 */
int firstFrame = -1;

/**
 The last frame.
 */
int lastFrame = -1;

/**
 The frame increment.
 */
int frameIncrement = -1;


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
    
    // Get the output stream.
    FILE* outputFp = stdout;
    if (outputFilename != NULL)
    {
        outputFp = fopen(outputFilename,"w");
        if (outputFp == NULL)
        {
            fprintf(stderr, "Error: output file %s could not be created.\n", outputFilename);
            exit(1);
        }
    }
    
    if (function == 0)
    {
        writeContacts(outputFp);
    }
    else if (function == 1)
    {
        writeOrderParameters(outputFp);
    }
    else if (function == 2)
    {
        writeNativeContacts(outputFp);
    }

    // Close the output file.    
    if (outputFp != stdout)
    {
        fclose(outputFp);
        outputFp = NULL;
    }
    
    // Delete the filename array.
    if (structureArguments != NULL)
    {
        delete [] structureArguments;
        structureArguments = NULL;
    }

    return 0;
}

void writeContacts(FILE* outputFp)
{
   int i;
    // Create the alphabet and reader.
    Alphabet* alphabet = AlphabetBuilder::createProteinAlphabet();
    const char* backboneAtomName = "CA";

    // Load the structure.
    Structure* structure = PDBReader::readStructure(alphabet, structureArguments[0], backboneAtomName);
    if (structure == NULL)
    {
        fprintf(stderr, "Error: file %s was not a valid structure file.\n", structureArguments[0]);
        exit(1);
    }
    
    // Get the contacts.
    ContactList *contacts = ContactTools::getContacts(structure, contactDistance, minSequenceDistance, maxSequenceDistance);
    
    // Output the number of contacts.
    fprintf(outputFp, "Number of contacts: %d\n", contacts->getNumberContacts());
    
    // Output the contact order.
    fprintf(outputFp, "Contact order:      %0.2f\n", ContactTools::getContactOrder(contacts));
    
    // Output the contacts.
    if (contacts->getNumberContacts() > 0)
    {
        fprintf(outputFp, "RESIDUE RESID INSERTION ATOM RESIDUE RESID INSERTION ATOM DISTANCE\n");
        for (i=0; i<contacts->getNumberContacts(); i++)
        {
            Contact* contact = contacts->getContact(i);
            int residue1Index = contact->getResidue1Index();
            int residue2Index = contact->getResidue2Index();
            Residue* residue1 = contact->getResidue1();
            Residue* residue2 = contact->getResidue2();
            Atom* atom1 = contact->getAtom1();
            Atom* atom2 = contact->getAtom2();
            fprintf(outputFp, "%-7d %-5s \"%-1s\"       %-4s %-7d %-5s \"%-1s\"       %-4s %0.3f\n", residue1Index, residue1->getResID(), residue1->getInsertionName(), atom1->getName(), residue2Index, residue2->getResID(), residue2->getInsertionName(), atom2->getName(), atom1->getDistanceTo(*atom2));
        }
    }
    
    delete alphabet;
}

void writeOrderParameters(FILE* outputFp)
{
    // Create the alphabet and reader.
    Alphabet* alphabet = AlphabetBuilder::createProteinAlphabet();
    const char* backboneAtomName = "CA";
    int structureIndex;

    // Load the native structure.
    Structure* nativeStructure = PDBReader::readStructure(alphabet, structureArguments[0], backboneAtomName);
    if (nativeStructure == NULL)
    {
        fprintf(stderr, "Error: file %s was not a valid structure file.\n", structureArguments[0]);
        exit(1);
    }
        
    // Get the native contacts.
    ContactList *nativeContacts = ContactTools::getContacts(nativeStructure, contactDistance, minSequenceDistance, maxSequenceDistance);
    double contactOrder = ContactTools::getContactOrder(nativeContacts);
    
    // Process all of the comparison structures.
    fprintf(outputFp, "FRAME  Q      pCO      pCO/CO TOTAL  NATIVE NON-NATIVE RMSD     RGYR\n");
    for (structureIndex=firstFrame; structureIndex <= lastFrame; structureIndex+=frameIncrement)
    {
        char filename[4096];
        sprintf(filename, structureArguments[1], structureIndex);
        
        // Load the frame.
        Structure* comparisonStructure = PDBReader::readStructure(alphabet, filename, backboneAtomName);        
        if (comparisonStructure == NULL)
        {
            fprintf(stderr, "Error: file %s was not a valid structure file.\n", filename);
            exit(1);
        }
        
        // Get the order parameters.
        ContactList* contacts = ContactTools::getContacts(comparisonStructure, contactDistance, minSequenceDistance, maxSequenceDistance);
        ContactList* formedNativeContacts = ContactTools::getFormedNativeContacts(nativeContacts, comparisonStructure, maxDistanceDeviation);
        ContactList* nonNativeContacts = contacts->getSubsetExcluding(formedNativeContacts);
        double Q = ContactTools::getFractionNativeContacts(nativeContacts, comparisonStructure, maxDistanceDeviation);
        double pCO = ContactTools::getPartialContactOrder(nativeContacts, comparisonStructure, maxDistanceDeviation);
        double rmsd = 0.0;
        double rgyr = 0.0;
        fprintf(outputFp, "%-6d %-6.4f %-8.4f %-6.4f %-6d %-6d %-10d %-8.4f %-8.4f\n", structureIndex, Q, pCO, pCO/contactOrder, contacts->getNumberContacts(), formedNativeContacts->getNumberContacts(), nonNativeContacts->getNumberContacts(), rmsd, rgyr);
        
        delete comparisonStructure;
        comparisonStructure = NULL;
    }
    
    delete alphabet;
}

void writeNativeContacts(FILE* outputFp)
{
    // Create the alphabet and reader.
    Alphabet* alphabet = AlphabetBuilder::createProteinAlphabet();
    const char* backboneAtomName = "CA";
   int i;

    // Load the structures.
    Structure* nativeStructure = PDBReader::readStructure(alphabet, structureArguments[0], backboneAtomName);
    if (nativeStructure == NULL)
    {
        fprintf(stderr, "Error: file %s was not a valid structure file.\n", structureArguments[0]);
        exit(1);
    }
    
    Structure* comparisonStructure = PDBReader::readStructure(alphabet, structureArguments[1], backboneAtomName);
    if (comparisonStructure == NULL)
    {
        fprintf(stderr, "Error: file %s was not a valid structure file.\n", structureArguments[1]);
        exit(1);
    }
    
    // Get the contacts.
    ContactList* contacts = ContactTools::getContacts(comparisonStructure, contactDistance, minSequenceDistance, maxSequenceDistance);
    ContactList* nativeContacts = ContactTools::getContacts(nativeStructure, contactDistance, minSequenceDistance, maxSequenceDistance);
    ContactList* formedNativeContacts = ContactTools::getFormedNativeContacts(nativeContacts, comparisonStructure, maxDistanceDeviation);
    ContactList* nonNativeContacts = contacts->getSubsetExcluding(formedNativeContacts);

    // Output some information about the structure.
    fprintf(outputFp, "Number contacts:            %d\n", contacts->getNumberContacts());
    fprintf(outputFp, "Number native contacts:     %d\n", nativeContacts->getNumberContacts());
    fprintf(outputFp, "Fraction native contacts:   %0.2f\n", ContactTools::getFractionNativeContacts(nativeContacts, comparisonStructure, maxDistanceDeviation));
    fprintf(outputFp, "Number non-native contacts: %d\n", nonNativeContacts->getNumberContacts());
    fprintf(outputFp, "Native contact order:       %0.2f\n", ContactTools::getContactOrder(nativeContacts));
    fprintf(outputFp, "Partial contact order:      %0.2f\n", ContactTools::getPartialContactOrder(nativeContacts, comparisonStructure, maxDistanceDeviation));
    
    // Output the contacts.
    if (formedNativeContacts->getNumberContacts() > 0)
    {
        fprintf(outputFp, "RESIDUE RESID INSERTION ATOM RESIDUE RESID INSERTION ATOM DISTANCE\n");
        for (i=0; i<formedNativeContacts->getNumberContacts(); i++)
        {
            Contact* contact = formedNativeContacts->getContact(i);
            int residue1Index = contact->getResidue1Index();
            int residue2Index = contact->getResidue2Index();
            Residue* residue1 = contact->getResidue1();
            Residue* residue2 = contact->getResidue2();
            Atom* atom1 = contact->getAtom1();
            Atom* atom2 = contact->getAtom2();
            fprintf(outputFp, "%-7d %-5s \"%-1s\"       %-4s %-7d %-5s \"%-1s\"       %-4s %0.3f\n", residue1Index, residue1->getResID(), residue1->getInsertionName(), atom1->getName(), residue2Index, residue2->getResID(), residue2->getInsertionName(), atom2->getName(), atom1->getDistanceTo(*atom2));
        }
    }
    
    delete alphabet;
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
    if (argc > 1)
    {
        char* functionName = argv[1];
        
        //See if the user is trying to get help or the current version.
        if (strcmp(functionName, "-h") == 0 || strcmp(functionName, "--help") == 0) {
            return 0;
        }
        else if (strcmp(functionName, "-v") == 0 || strcmp(functionName, "--version") == 0) {
            exit(0);
        }
        
        // See if the user is performing the contact function.
        else if (strcmp(functionName, "contacts") == 0)
        {
            function = 0;
            
            //Parse the arguments.        
            for (i=2; i<argc; i++) {
                
                char *arg = argv[i];
                
                // If this is an option.
                if (i < argc-1 && strcmp(arg, "-c") == 0)
                {
                    arg = argv[++i];
                    contactDistance = atof(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-s") == 0)
                {
                    arg = argv[++i];
                    minSequenceDistance = atoi(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-smax") == 0)
                {
                    arg = argv[++i];
                    maxSequenceDistance = atoi(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-out") == 0)
                {
                    arg = argv[++i];
                    outputFilename = arg;
                }
                
                // If this is the structure filename.
                else if (i == argc-1)
                {
                    numberStructureArguments = 1;
                    structureArguments = new char*[1];
                    structureArguments[0] = arg;
                }
                
                //This must be an invalid option.
                else {
                    return 0;
                }
            }
            
            //Make sure we have a valid state for all of the variables.
            if (function == 0 && ((numberStructureArguments == 1 && structureArguments[0] != NULL)))
                return 1;
        }
        
        // See if the user is performing the order parameter function.
        else if (strcmp(functionName, "order_parameters") == 0)
        {
            function = 1;
            
            //Parse the arguments.        
            for (i=2; i<argc; i++) {
                
                char *arg = argv[i];
                
                // If this is an option.
                if (i < argc-1 && strcmp(arg, "-c") == 0)
                {
                    arg = argv[++i];
                    contactDistance = atof(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-s") == 0)
                {
                    arg = argv[++i];
                    minSequenceDistance = atoi(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-smax") == 0)
                {
                    arg = argv[++i];
                    maxSequenceDistance = atoi(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-d") == 0)
                {
                    arg = argv[++i];
                    maxDistanceDeviation = atof(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-out") == 0)
                {
                    arg = argv[++i];
                    outputFilename = arg;
                }
                
                // If this is the structure filename.
                else if (i == argc-5)
                {
                    numberStructureArguments = 2;
                    structureArguments = new char*[2];
                    structureArguments[0] = argv[i++];
                    structureArguments[1] = argv[i++];
                    firstFrame = atoi(argv[i++]);
                    lastFrame = atoi(argv[i++]);
                    frameIncrement = atoi(argv[i++]);
                }
                
                //This must be an invalid option.
                else {
                    return 0;
                }
            }
            
            //Make sure we have a valid state for all of the variables.
            if (function == 1 && numberStructureArguments == 2 && structureArguments[0] != NULL && structureArguments[1] != NULL && firstFrame >= 0 && lastFrame >= 0 && firstFrame <= lastFrame && frameIncrement > 0)
                return 1;
        }
        
        // See if the user is performing the contact function.
        else if (strcmp(functionName, "native_contacts") == 0)
        {
            function = 2;
            
            //Parse the arguments.        
            for (i=2; i<argc; i++) {
                
                char *arg = argv[i];
                
                // If this is an option.
                if (i < argc-1 && strcmp(arg, "-c") == 0)
                {
                    arg = argv[++i];
                    contactDistance = atof(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-s") == 0)
                {
                    arg = argv[++i];
                    minSequenceDistance = atoi(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-smax") == 0)
                {
                    arg = argv[++i];
                    maxSequenceDistance = atoi(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-d") == 0)
                {
                    arg = argv[++i];
                    maxDistanceDeviation = atof(arg);
                }
                else if (i < argc-1 && strcmp(arg, "-out") == 0)
                {
                    arg = argv[++i];
                    outputFilename = arg;
                }
                
                // If this is the structure filename.
                else if (i == argc-2)
                {
                    numberStructureArguments = 2;
                    structureArguments = new char*[2];
                    structureArguments[0] = argv[i++];
                    structureArguments[1] = argv[i++];
                }
                
                //This must be an invalid option.
                else {
                    return 0;
                }
            }
            
            //Make sure we have a valid state for all of the variables.
            if (function == 2 && ((numberStructureArguments == 2 && structureArguments[0] != NULL && structureArguments[1] != NULL)))
                return 1;
        }
    }

    //Something must be invalid, return 0;
    return 0;
}


/**
 * This function prints the copyright notice.
 */
void printCopyright(int argc, char** argv) {
    printf("%s v%s\n", argv[0], VERSION);
    printf("Copyright (C) 2007-2011 Luthey-Schulten Group, University of Illinois.\n");
    printf("\n");
}


/**
 * This function prints the usage for the seqqr program.
 */
void printUsage(int argc, char** argv) {
    printf("\n");
    printf("Computes function related to structure contacts.\n");
    printf("\n");
    printf("usage: %s (--help | --version | FUNCTION)\n", argv[0]);
    printf("\n");
    printf("\n");
    printf(" Where FUNCTION is one of the following:\n");
    printf("   Get the contacts present in a structure:\n");
    printf("     contacts [-c value] [-s value] [-smax value] [-out value] structure\n");
    printf("\n");
    printf("   Get the native contacts present in a structure:\n");
    printf("     native_contacts [-c value] [-s value] [-s value] [-d value] [-out value] native_structure comparison_structure\n");
    printf("\n");
    printf("   Calculate the order parameters (Q, pCO) for a series of structures:\n");
    printf("     order_parameters [-c value] [-s value] [-smax value] [-d value] [-out value] native_structure comparison_structure_pattern first last increment\n");
    printf("\n");
    printf(" Where the various options are defined as:\n");
    printf("   structure              A PDB file containing a structure to be analyzed.\n");
    printf("   -c                     The maximum distance between two residues from them to be considered in contact.\n");
    printf("   -s                     The minimum separation in the sequence for two residues to be considered native.\n");
    printf("   -smax                  The maximum separation in the sequence for two residues to be considered native.\n");
    printf("   -d                     The maximum distance a contact can be from its native distance to be formed.\n");
    printf("\n");
}

