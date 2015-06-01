#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "alphabet.h"
#include "alphabetBuilder.h"
//#include "qTools.h"
#include "structureAlignment.h"
#include "structureAlignmentReader.h"
#include "structureQR.h"
#include "typeConvert.h"
#include "structqr.h"




enum {QH, QPERRESIDUE, QR};
enum {DNA, RNA, PROTEIN};

int calculationType = QR;
int alphabetType = PROTEIN;
int kArgs[] = {0,0};
char* kArgStrings[] = {0,0};
int kBaseIndex = 0;

char* outputFilename = NULL;
char* fastaFilename = NULL;
char* pdbDir = NULL;

/**
 * Entry point for the qpair program.
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
//  printf("Done parsing args. calcType: %d, alph: %d, fasta: %s, Dir: %s\n", calculationType, alphabetType, fastaFilename, pdbDir);
  
  StructureAlignment* structAl1 = readStructureAlignment(fastaFilename, pdbDir);
//  printf("Done creating structAl1\n");
  if (structAl1 == NULL) {
    printf("Error: couldn't read structure alignment\n");    
    exit(1);
  }
  
  //QTools* qTools = 0;
  StructureQR* structQR = 0;
  FILE* out = 0;
  //char* kArgString = 0;
  //int check = 0;

  switch(calculationType) {
/*  case QH:
    qTools = new QTools(structAl1);
    qTools->q(0);
    out = fopen(outputFilename,"w");
    qTools->printQ(out);
    fclose(out);
    break;
  case QPERRESIDUE:
    qTools = new QTools(structAl1);
    qTools->qPerResidue();
    out = fopen(outputFilename,"w");
    qTools->printQPerResidue(out);
    fclose(out);
    break;
*/
  case QR:
    structQR = new StructureQR(structAl1);
//    printf("Preparing to do structQR->qr()\n");
    structQR->qr();
    out = fopen(outputFilename,"w");
//    printf("Preparing to print Columns\n");
    structQR->printColumns(out);
    fclose(out);
    delete structQR;
    break;
  default:
    printf("Error: invalid calculationType\n");
    exit(1);
  }
  
  delete structAl1;
/********  if (qTools != 0) {
    // XXX - SEGFAULT WHEN QTOOLS IS DELETED
    delete qTools;
  }
*/

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

  char* params = 0;
  //int i=0;
  int* paramFlags = new int[6];
  int i=0;
  for (i=0; i<6; i++) {
    paramFlags[i] = 0;
  }

  enum {P, Q, J, K, R, D};

  // Read the parameters into paramFlags
  // Given the parameter settings, set up the variables
  // Defaults
  paramFlags[Q] = 1;
  calculationType = QR;
  alphabetType = PROTEIN;
  for (i=1; i<argc; i++) {
    params = argv[i];
    /*    if (strcmp(params,"-p") == 0) {
      paramFlags[P] = 1;
      calculationType = QPERRESIDUE;
      }*/
    if (strcmp(params,"-q") == 0) {
      paramFlags[Q] = 1;
      calculationType = QR;
    }
    else if (strcmp(params,"-r") == 0) {
      paramFlags[R] = 1;
      alphabetType = RNA;
    }
    else if (strcmp(params,"-d") == 0) {
      paramFlags[D] = 1;
      alphabetType = DNA;
    }
    else if (strcmp(params,"-o") == 0 && i < (argc-1)) {
        outputFilename = argv[++i];
    }
    //If this is the next to last argument and it is not an option, it must be the input filename.
    else if (i == argc-2) {
      fastaFilename = argv[i];
    }
    
    //If this is the last argument and it is not an option, it must be the directory of the PDB files
    else if (i == argc-1) {
      pdbDir = argv[i];
    }
    else {
        return 0;
    }
  }

  //fastaFilename = argv[argc-2];
  //pdbDir = argv[argc-1];

  // Check for invalid parameter combinations
  int paramSum = paramFlags[P] + paramFlags[Q] + paramFlags[J] + paramFlags[K];
  if (paramSum > 1) {
      return 0;
  }

  paramSum = paramFlags[R] + paramFlags[D];
  if (paramSum > 1) {
      return 0;
  }

  delete paramFlags;

  if (fastaFilename != NULL && pdbDir != NULL)
    return 1;

  return 0;
}

/**
 * This function prints the copyright notice.
 *
 * @param   argc    The argument count.
 * @param   argv    The argument values.
 */
void printCopyright(int argc, char** argv) {
    printf("%s v%s\n", argv[0], VERSION);
    printf("Copyright (C) 2003-2011 The Board of Trustees of the University of Illinois.\n");
    //printf("When publishing a work based in whole or in part on this program, please\n");
    //printf("reference: Sethi, A., O'Donoghue, P. & Luthey-Schulten, Z. (2005) 'Evolutionary\n");
    //printf("profiles from the QR factorization of multiple sequence alignments' Proc. Natl.\n");
    //printf("Acad. Sci. USA 102, 4045-4050.\n");
    printf("\n");
}

/**
 * This function prints the usage for the seqqr program.
 *
 * @param   argc    The argument count.
 * @param   argv    The argument values.
*/
void printUsage(int argc, char** argv){

    printf("usage: %s [-r|-d] [-o output_file] fasta_file pdb_dir\n", argv[0]);
    printf("\n");
    printf(" fasta_file       A FASTA file containing an alignment of the sequences.\n");
    printf(" pdb_dir          A directory containing the pdb files for the sequences.\n");
    printf("\n");
    printf(" r - RNA, so use P for backbone\n");
    printf(" d - DNA, so use P for backbone\n");
    printf(" o output_file - Output the results to the specified file.\n");
    printf("\n");
    return;
}


StructureAlignment* readStructureAlignment(char* fastaFilename, char* pdbDir) 
{
//   printf("=>readStructureAlignment\n");

   char* fastaPath = 0;
   if (pdbDir[strlen(pdbDir)-1] != '/') {
      fastaPath = new char[strlen(pdbDir)+2];
      strncpy(fastaPath,pdbDir,strlen(pdbDir));
      fastaPath[strlen(pdbDir)] = '/';
      fastaPath[strlen(pdbDir)+1] = '\0';
   } else {
      fastaPath = new char[strlen(pdbDir)+1];
      strncpy(fastaPath,pdbDir,strlen(pdbDir));
      fastaPath[strlen(pdbDir)] = '\0';
      strncat(fastaPath,"\0",1);
   }
//   printf("=>readStructureAlignment. fastaPath:%s\n", fastaPath);
   //AlphabetBuilder* alphaBuild1 = new AlphabetBuilder();
   Alphabet* alpha1 = 0;
   char* backboneAtom = 0;
   switch (alphabetType) {
      case DNA:
         //alpha1 = alphaBuild1->getDnaAlphabet();
         alpha1 = AlphabetBuilder::createDnaAlphabet();
         backboneAtom = (char*)"P";
         break;
      case RNA:
         //alpha1 = alphaBuild1->getRnaAlphabet();
         alpha1 = AlphabetBuilder::createRnaAlphabet();
         backboneAtom = (char*)"P";
         break;
      case PROTEIN:
         //alpha1 = alphaBuild1->getProteinAlphabet();
         alpha1 = AlphabetBuilder::createProteinAlphabet();
         backboneAtom = (char*)"CA";
         break;
      default:
         printf("Error - getStructureAlignment; invalid alphabetType\n");
   }
//   printf("=>readStructureAlignment. done creating alphabet\n");
    
   //printf("%s\n",alpha1->toString());
   StructureAlignmentReader* strAlnRead1 = new StructureAlignmentReader(alpha1,
                                                        backboneAtom);
   if (!strAlnRead1)
   {
      printf("structqr.readStructureAlignment.  Couldn't create reader\n");
      return 0;
   }
//   printf("=>readStructureAlignment. done creating strAlnRead1\n");
   strAlnRead1->setAlignmentPath(fastaPath);
   if (!strAlnRead1->setAlignmentFilename(fastaFilename))
   {
      printf("structqr.readStructureAlignment.  Couldn't set filename.\n");
   }
   strAlnRead1->setStructurePath(fastaPath);
   StructureAlignment* structAln = strAlnRead1->getStructureAlignment();
   delete strAlnRead1;
    
//   printf("<=end of structqr.readStructureAlignment\n");
    
   return structAln;
}

