
#ifndef ALIGNEDSTRUCTURE_H
#define ALIGNEDSTRUCTURE_H

//#include "alignedSequence.h"
//#include "symbolList.h"
#include "structure.h"

#include <stdio.h>

class AlignedSequence;
class Alphabet;
class Coordinate3D;
class Symbol;

class AlignedStructure : public Structure {

 public:
  AlignedStructure(Alphabet* alphabet, const char* name=0) : Structure(alphabet, name) {}
    //AlignedStructure(int l, Alphabet* a, char* n);
    //AlignedStructure(int l, Alphabet* a);
  AlignedStructure(Structure* structure, AlignedSequence* alnSeq);
  ~AlignedStructure();

  virtual void addResidue(Symbol& symbol, float x, float y, float z);
  virtual void addResidue(char c, float x, float y, float z); 
  virtual void addResidue(Symbol& symbol, Coordinate3D coord);

  virtual void addResidue(const char* n, float x, float y, float z) 
  {
     Structure::addResidue(n,x,y,z);
  }
  virtual void addResidue(char c, Coordinate3D coord, Residue* residue)
  {
     Structure::addResidue(c,coord,residue);
  }
  virtual void addResidue(const char* n, Coordinate3D coord, Residue* residue)
  {
     Structure::addResidue(n,coord,residue);
  }
  virtual void addResidue(Symbol& symbol, Coordinate3D coord, Residue* residue)
  {
     Structure::addResidue(symbol,coord,residue);
  }

  // XXX - REDO ALL ACCESS METHODS IN UNALIGNED VERSIONS;
  //   USE THE DEFAULT PARENT METHODS FOR THE ALIGNED DATA
  virtual void addGap();
  Symbol getUnalignedSymbol(int i);
  Coordinate3D getUnalignedCoordinate(int i);
  int getUnalignedLength();
  int alignedToUnalignedIndex(int i);
  int unalignedToAlignedIndex(int i);

 private:
  int* alignedToUnaligned;
  int* unalignedToAligned;
  int  unalignedLength;

};

#endif
