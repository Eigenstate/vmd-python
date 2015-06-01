
#ifndef STRUCTUREALIGNMENTREADER_H
#define STRUCTUREALIGNMENTREADER_H

#include "alignedSequence.h"
#include "alignedStructure.h"
#include "fastaReader.h"
#include "pdbReader.h"
#include "sequenceAlignment.h"
#include "structure.h"
#include "structureAlignment.h"

//#include <stdio.h>

class StructureAlignmentReader {

 public:
 StructureAlignmentReader(Alphabet* alpha, char* bbAtomName, int msc);
 StructureAlignmentReader(Alphabet* alpha, char* bbAtomName);
  ~StructureAlignmentReader();

  int setAlignmentFilename(char* fn);
  int setAlignmentPath(const char* fn);
  int setStructureFilenames(char** fns, int nameCount);
  int setStructurePath(char* p);
  StructureAlignment* getStructureAlignment();
  SequenceAlignment* getSequenceAlignment();

 private:
  AlignedSequence* getMatchingAlignedSequence(Structure* structure, SequenceAlignment* seqAln);
  int getStructureNamesFromAlignment(SequenceAlignment* seqAln);
  int checkAlignmentFullName();

 private:
  Alphabet* alphabet;
  char* backboneAtomName;
  char* alignmentFilename;
  char* alignmentPath;      // Path to alignment file
  char* alignmentFullName;
  char** structureFilenames;
  char* structurePath;      // Path to structure files
  int structureCount;      /* Current structure count;
			      always < maxStructureCount */
  int structureFilenamesCount;
  int maxStructureCount;

};

#endif
