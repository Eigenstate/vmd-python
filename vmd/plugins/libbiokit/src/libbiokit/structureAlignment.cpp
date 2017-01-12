
#include "structureAlignment.h"



// Constructor
//
StructureAlignment::StructureAlignment(Alphabet* alpha, int mL, int mSC)
  : alphabet(alpha), length(0), maxLength(mL),
     structureCount(0), maxStructureCount(mSC) {

  alignment = new AlignedStructure* [maxStructureCount];

  return;
}


// Constructor
//
StructureAlignment::StructureAlignment(int mL, int mSC)
  : alphabet(0), length(0), maxLength(mL),
     structureCount(0), maxStructureCount(mSC) {

  alignment = new AlignedStructure* [maxStructureCount];

  return;
}


// Destructor
//
StructureAlignment::~StructureAlignment() {
   int i;
  for (i=0; i<structureCount; i++) {
    delete alignment[i];
  }

  delete alignment;

  return;
}


// addStructure
//
int StructureAlignment::addStructure(AlignedStructure* structure) {
  
  if (structureCount < maxStructureCount &&
      structure->getSize() <= maxLength) {
    if (alphabet == 0) {
      alphabet = structure->getAlphabet();
    }
    // Structure using a different alphabet
    else if (alphabet != structure->getAlphabet()) {
      return 0;
    }
    alignment[structureCount] = structure;
    structureCount++;
    if (length < structure->getSize()) {
      length = structure->getSize();
    }
    return 1;
  }

  return 0;
}


// getStructure
//
AlignedStructure* StructureAlignment::getStructure(int i) {

  //printf("=>getStructure\n");
  if (i < structureCount) {
    //printf("  alphabet: %s\n",alignment[i]->getAlphabet()->toString());
    //printf("<=getStructure: %d\n",i);
    return alignment[i];
  }

  //printf("<=getStructure: NULL\n");
  return NULL;
}


// getSymbol
//
Symbol& StructureAlignment::getSymbol(int i, int j) {

  if (i < structureCount && j < length) {
    return alignment[i]->get(j);
  }

  return alphabet->getUnknown();
}


// getCoordinate
//
Coordinate3D StructureAlignment::getCoordinate(int i, int j) {

  if (i < structureCount && j < length) {
    return alignment[i]->getCoordinate(j);
  }

  return Coordinate3D();
}


// getAlphabet
//
Alphabet* StructureAlignment::getAlphabet() {

  return alphabet;
}


// getLength
//
int StructureAlignment::getNumberPositions() {

  return length;
}


// getStructureCount
//
int StructureAlignment::getNumberStructures() {

  return structureCount;
}
