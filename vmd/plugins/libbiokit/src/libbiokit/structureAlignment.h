
#ifndef STRUCTUREALIGNMENT_H
#define STRUCTUREALIGNMENT_H

#include "alignedStructure.h"
#include "alphabet.h"
#include "coordinate3D.h"
#include "structure.h"
#include "symbol.h"

class StructureAlignment {

 public:
  StructureAlignment(Alphabet* alpha, int mL, int mSC);
  StructureAlignment(int mL, int mSC);   // maxLength, maxStructureCount
  ~StructureAlignment();

  int addStructure(AlignedStructure* structure);
  AlignedStructure* getStructure(int i);
  Symbol& getSymbol(int i, int j);
  Coordinate3D getCoordinate(int i, int j);
  Alphabet* getAlphabet();
  int getNumberPositions();
  int getNumberStructures();

 private:
  AlignedStructure** alignment;
  Alphabet* alphabet;
  int length;
  int maxLength;
  int structureCount;
  int maxStructureCount;

};

#endif
