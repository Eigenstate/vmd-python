
#include "structureAlignmentReader.h"
#include <stdio.h>
#include <string.h>


// Constructor
StructureAlignmentReader::StructureAlignmentReader(Alphabet* alpha, char*
bbAtomName, int msc)
: alphabet(alpha), backboneAtomName(bbAtomName), structureCount(0),
     structureFilenamesCount(0), maxStructureCount(msc) {
   int i;
  //printf("=>StructureAlignmentReader()\n");
  //printf("alphabet: %s\n",alphabet->toString());

  alignmentFilename = 0;
  alignmentPath = 0;
  alignmentFullName = 0;
  structureFilenames = new char* [maxStructureCount];
  structurePath = 0;
  for (i=0; i<maxStructureCount; i++) {
    structureFilenames[i] = 0;
  }

  //printf("<=StructureAlignmentReader()\n");
  return;
}


// Constructor
StructureAlignmentReader::StructureAlignmentReader(Alphabet* alpha, 
                        char* bbAtomName)
: alphabet(alpha), backboneAtomName(bbAtomName), structureCount(0),
     structureFilenamesCount(0), maxStructureCount(1024) {
   int i;

  //printf("=>StructureAlignmentReader()\n");
  //printf("alphabet: %s\n",alphabet->toString());

  alignmentFilename = 0;
  alignmentPath = 0;
  alignmentFullName = 0;
  structureFilenames = new char* [maxStructureCount];
  structurePath = 0;
  for (i=0; i<maxStructureCount; i++) {
    structureFilenames[i] = 0;
  }

  //printf("<=StructureAlignmentReader()\n");
  return;
}


// Destructor
StructureAlignmentReader::~StructureAlignmentReader() {
   int i;
  if (alignmentFilename != 0) {
    delete alignmentFilename;
  }

  if (alignmentPath != 0) {
    delete alignmentPath;
  }

  if (alignmentFullName != 0) {
    alignmentFullName = 0;
  }

  if (structureFilenames != 0) {
    for (i=0; i<structureFilenamesCount; i++) {
      delete structureFilenames[i];
    }
  }

  delete structureFilenames;

  return;
}


// setAlignmentFilename
//   
int StructureAlignmentReader::setAlignmentFilename(char* fn) {

  //printf("=>StructureAlignmentReader::setAlignmentFilename\n");

  if (alignmentFilename != 0) {
    delete alignmentFilename;
    alignmentFilename = 0;
  }

  if (alignmentFullName != 0) {
    delete alignmentFullName;
    alignmentFullName = 0;
  }

  int len = strlen(fn);
  char* tempName = new char[len+5];  // len+1 is normal (+4) for suffix
  strncpy(tempName,fn,len);
  tempName[len] = '\0';

  alignmentFilename = new char[len+1];
  strncpy(alignmentFilename,fn,len);
  alignmentFilename[len] = '\0';

  // Given name works
  if (checkAlignmentFullName() == 1) {
    //printf("\n      %s works\n\n",alignmentFilename);
    //setNameFromFilename();
    //printf("<=StructureAlignmentReader::setAlignmentFilename\n");
    return 1;
  }
  //printf("   setAlignmentFilename: %s doesn't work\n",alignmentFilename);

  delete alignmentFilename;
  alignmentFilename = 0;

  //printf("   alignmentFilename doesn't work\n");
  //printf("<=StructureAlignmentReader::setAlignmentFilename\n");

  return 0;

  /*
  int len = strlen(fn);

  alignmentFilename = new char[len+1];
  strncpy(alignmentFilename,fn,len);
  alignmentFilename[len] = '\0';

  FILE * infile = fopen(alignmentFilename,"r");
  if (infile == NULL) {
    fclose(infile);
    delete alignmentFilename;
    return 0;
  }
  fclose(infile);

  return 1;
  */
}

// ---------------------------------------------------------------------

// setAlignmentPath
//
int StructureAlignmentReader::setAlignmentPath(const char* p) {
  
  int len = strlen(p);
  if (alignmentPath != 0)  delete alignmentPath;
  alignmentPath = new char[len+1];
  strncpy(alignmentPath,p,len);
  alignmentPath[len] = '\0';

  return 1;
}

// ---------------------------------------------------------------------

// setStructurePath
//
int StructureAlignmentReader::setStructurePath(char* p) {
  
  int len = strlen(p);
  if (structurePath != 0)  {
     delete structurePath;
  }
  structurePath = new char[len+1];
  strncpy(structurePath,p,len);
  structurePath[len] = '\0';

  return 1;
}


// ---------------------------------------------------------------------
// setStructureFilenames
//   
int StructureAlignmentReader::setStructureFilenames(char** fns, int nameCount) {
   int i;
  if (nameCount > maxStructureCount) {
    nameCount = maxStructureCount;
    //return 0;
  }

  if (alignmentFilename != 0) {
    delete alignmentFilename;
  }

  for (i=0; i<nameCount; i++) {
    int len = strlen(fns[i]);
    
    structureFilenames[structureFilenamesCount] = new char[len+1];
    strncpy(structureFilenames[structureFilenamesCount],fns[i],len);
    structureFilenames[structureFilenamesCount][len] = '\0';
    
    FILE * infile = fopen(structureFilenames[structureFilenamesCount],"r");
    if (infile == NULL) {
      fclose(infile);
      delete structureFilenames[structureFilenamesCount];
      nameCount--;
      //return 0;
    }
    else {
      structureFilenamesCount++;
    }
    fclose(infile);
  }
  
  return structureFilenamesCount;
}

// ---------------------------------------------------------------------
// getStructureAlignment
//   
StructureAlignment* StructureAlignmentReader::getStructureAlignment() 
{
//   printf("StructureAlignmentReader::getStructureAlignment.start\n");
   SequenceAlignment* seqAln = ::FASTAReader::readSequenceAlignment(alphabet, 
                                                  alignmentFullName);
//                                                  alignmentFilename);
   if (!seqAln)
   {
      printf("Couldn't read FASTA alignment.  (structureAlignmentReader.getStructureAlignment)\n");
      return 0;
   }
   int len = seqAln->getNumberPositions();
   int sequenceCount = seqAln->getNumberSequences();

//   printf("StructureAlignmentReader::getStructureAlignment: read %d seqs with %d positions\n", sequenceCount, len);

   int i;

   if (structureFilenamesCount == 0) {
      getStructureNamesFromAlignment(seqAln);
   }
   // else just use preloaded structureFilenames

   StructureAlignment* structAln = new StructureAlignment(len, sequenceCount);
   Structure* structure = 0;
   AlignedSequence* alnSeq = 0;
   AlignedStructure* alnStruct = 0;
   for (i=0; i<structureFilenamesCount; i++) {
//      printf("StructureAlignmentReader::getStructureAlignment: reading filename %s\n", structureFilenames[i]);
      // Get Structure
      structure = ::PDBReader::readStructure(alphabet, 
                             structureFilenames[i], backboneAtomName);
      if (structure) {
//         printf("StructureAlignmentReader::getStructureAlignment: struct name: %s\n", structure->getName());
         // Get matching AlignedSequence from seqAln
         alnSeq = getMatchingAlignedSequence(structure, seqAln);

         if (alnSeq) {

            // Build AlignedStructure (constructor in AlignedStructure)
            alnStruct = new AlignedStructure(structure, alnSeq);
            if (alnStruct) {
               // Add AlignedStructure to StructureAlignment
               structAln->addStructure(alnStruct);
               structureCount++;
            } else {
               printf("StructureAlignmentReader::getStructureAlignment: couldn't get aligned structure\n");
            }

         } else {
            printf("StructureAlignmentReader::getStructureAlignment: couldn't get matching aligned sequence for struct: %s\n",
                                           structure->getName());
         }
         delete structure;
    
      } else {
         printf("StructureAlignmentReader::getStructureAlignment: couldn't read file\n");
      }


      structure = 0;
      alnSeq = 0;
      alnStruct = 0;
      //printf("i: %d, alphabet: %s\n",i,alphabet->toString());
   }

   //delete fastaRead;
   //delete pdbRead;
   delete seqAln;
   //printf("structureCount: %d\n",structureCount);

   //printf("<=getStructureAlignment\n");
   return structAln;
}


// getSequenceAlignment
//   
SequenceAlignment* StructureAlignmentReader::getSequenceAlignment() {

  //FASTAReader* fastaRead = new FASTAReader(alphabet);

  //fastaRead->setPath(alignmentPath);
  //fastaRead->setFilename(alignmentFilename);
  //SequenceAlignment* seqAln = fastaRead->getSequenceAlignment();
  SequenceAlignment* seqAln = ::FASTAReader::readSequenceAlignment(alphabet, alignmentFilename);

  if (seqAln != 0) {
    //delete fastaRead;
    return seqAln;
  }

  //delete fastaRead;

  return 0;
}


// getMatchingAlignedSequence
//   Search a SequenceAlignment for the AlignedSequence with the
//   same non-gapped list of symbols as the given Structure
AlignedSequence* StructureAlignmentReader::getMatchingAlignedSequence(Structure* structure, SequenceAlignment* seqAln) {

    //printf("=>StructureAlignmentReader::getMatchingAlignedSequence\n");
   int i; 
    if (structure == 0 || seqAln == 0) {
        return 0;
    }
    
    AlignedSequence* alnSeq = 0;
    int sequenceCount = seqAln->getNumberSequences();
    
    const char* seqName = 0;   // Set in for loop
    const char* structName = structure->getName();
    if (structName) {
        int structNameLen = strlen(structName);

        for (i=0; i<sequenceCount; i++) {
            alnSeq = seqAln->getSequence(i);
            if (alnSeq) {
                seqName = alnSeq->getName();
                if ( seqName && !strncmp(structName,seqName,structNameLen) ) {
                    // Compare non-gapped sequences
                    if ( structure->nongapSymbolsEqual(alnSeq) ) {
                        return alnSeq;
                    }
                }
            }
        }
    }
  
    return 0;
}

// --------------------------------------------------------------------
// getStructureNamesFromAlignment
int StructureAlignmentReader::getStructureNamesFromAlignment(SequenceAlignment* seqAln) {
   int i;
  //printf("=>StructureAlignmentReader::getStructureNamesFromAlignment\n");

  const char* tempName = 0;
  int sequenceCount = seqAln->getNumberSequences();

  for (i=0; i<sequenceCount; i++) {
    tempName = seqAln->getSequence(i)->getName();
    if (tempName != 0 && structureFilenamesCount < maxStructureCount) {
//  printf("=>StructureAlignmentReader::getStructureNamesFromAlignment: %s\n", tempName);
      int len = strlen(tempName);
      structureFilenames[structureFilenamesCount] = new char[len+1];
      strncpy(structureFilenames[structureFilenamesCount],tempName,len);
      structureFilenames[structureFilenamesCount][len] = '\0';
      structureFilenamesCount++;
    }
  }

  //printf("<=StructureAlignmentReader::getStructureNamesFromAlignment\n");

  return 1;
}


// --------------------------------------------------------------------
// checkAlignmentFullName
//   Make sure that alignmentFullName corresponds to an
//   accessible file
int StructureAlignmentReader::checkAlignmentFullName() {

   if (alignmentFilename == 0) {
      return 0;
   }

   if (alignmentFullName == 0) {
      if (alignmentPath == 0) {
//         printf("   checking: %s\n",alignmentFilename);
         FILE* infile = fopen(alignmentFilename,"r");
         if (infile == NULL) {
            return 0;
         }
         fclose(infile);
         int filenameLen = strlen(alignmentFilename);
         alignmentFullName = new char[filenameLen + 1];
         strncpy(alignmentFullName,alignmentFilename,filenameLen);
         alignmentFullName[filenameLen] = '\0';
//         printf("   return1\n");
//         printf("<=PDBReader::checkFullName\n");
         return 1;
      } else {
         int pathLen = strlen(alignmentPath);
         int filenameLen = strlen(alignmentFilename);
         alignmentFullName = new char[pathLen + filenameLen + 1];
         strncpy(alignmentFullName,alignmentPath,pathLen);
         alignmentFullName[pathLen] = '\0';
         strncat(alignmentFullName,alignmentFilename,filenameLen);
         alignmentFullName[pathLen + filenameLen] = '\0';
//         printf("   checking2: %s\n",alignmentFullName);
         FILE* infile = fopen(alignmentFullName,"r");
         if (infile == NULL) {
            delete alignmentFullName;
            alignmentFullName = 0;
//            printf("<=PDBReader::checkFullName0\n");
            return 0;
         }
         fclose(infile);
//         printf("   return2\n");
//         printf("<=PDBReader::checkFullName\n");
         return 1;
      }
   } else {
//      printf("   checking: %s\n",alignmentFullName);
      FILE* infile = fopen(alignmentFullName,"r");
      if (infile == NULL) {
         delete alignmentFullName;
         alignmentFullName = 0;
//         printf("<=PDBReader::checkFullName2\n");
         return 0;
      }
      fclose(infile);
//      printf("   return3\n");
//      printf("<=PDBReader::checkFullName3\n");
      return 1;
   }
}
