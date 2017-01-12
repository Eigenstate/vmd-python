
#include "PointerList.h"
#include "AtomList.h"
#include "alignedSequence.h"
#include "alignedStructure.h"
#include "alphabet.h"
#include "structure.h"
#include "symbol.h"
#include "symbolList.h"


// Constructor
/*AlignedStructure::AlignedStructure(int l, Alphabet* a, char* n)
  : Structure(l,a,n), unalignedLength(0) {

  alignedToUnaligned = new int[20000];
  unalignedToAligned = new int[20000];

  return;
  }*/


// Constructor
/*AlignedStructure::AlignedStructure(int l, Alphabet* a)
  : Structure(l,a), unalignedLength(0) {

  alignedToUnaligned = new int[20000];
  unalignedToAligned = new int[20000];

  return;
  }*/


// Constructor
AlignedStructure::AlignedStructure(Structure* structure,
				   AlignedSequence* alnSeq)
//  : Structure(alnSeq->getSize(), alnSeq->getAlphabet()),
  : Structure(alnSeq->getAlphabet(), alnSeq->getName()),
     unalignedLength(0) {

  // XXX: WHAT TO DO WHEN ARGS ARE BOGUS

  //printf("=>AlignedStructure::AlignedStructure\n");

  alignedToUnaligned = new int[20000];
  unalignedToAligned = new int[20000];

  int i=0;
  int j=0;
  //printf("   structure->getSize(): %d\n",structure->getSize());



  //for (; i<alnSeq->getSize(), j<structure->getSize(); i++) {
  // foobar.  Below line was previously a comma.  I switched it to
  // an &&
  for (; i<alnSeq->getSize() && j<structure->getSize(); i++) {
    if ( getAlphabet()->isGap(alnSeq->get(i)) ) {
      addGap();
      //printf("   i: %d, j: %d\n",i,j);
    }
    else {
      Coordinate3D coord = structure->getCoordinate(j);
      if (coord.isValid() && !getAlphabet()->isUnknown(alnSeq->get(i))) {
          addResidue(alnSeq->get(i), coord);
      }
      j++;
    }
  }

  if (i < alnSeq->getSize()) {
    for (; i<alnSeq->getSize(); i++) {
      if ( !getAlphabet()->isGap(alnSeq->get(i)) ) {
	printf("  AlignedStructure::no gap present!!!\n");
      }
      else {
	addGap();
      }
    }
  }

  //printf("<=AlignedStructure::AlignedStructure\n");
  return;
}


// Destructor
AlignedStructure::~AlignedStructure() {

  return;
}


// addResidue
void AlignedStructure::addResidue(Symbol& symbol, float x, float y, float z) {

  if (unalignedLength == 20000) return;

  //int success = 0;
  Structure::addResidue(symbol,x,y,z);

  //if (success == 0) return;

  alignedToUnaligned[getSize()-1] = unalignedLength;
  unalignedToAligned[unalignedLength] = getSize()-1;
  unalignedLength++;

  return;
}


void AlignedStructure::addResidue(char c, float x, float y, float z) {

  if (unalignedLength == 20000) return;

  //int success = 0;
  Structure::addResidue(c,x,y,z);

  //if (success == 0) return;

  alignedToUnaligned[getSize()-1] = unalignedLength;
  unalignedToAligned[unalignedLength] = getSize()-1;
  unalignedLength++;

  return;
}


// addResidue
void AlignedStructure::addResidue(Symbol& symbol, Coordinate3D coord) {

  //printf("=>AlignedStructure::addResidue\n");

  if (unalignedLength == 20000) {
    //printf("   Nope1\n");
    //printf("<=AlignedStructure::addResidue\n");
    return;
  }

  //int success = 0;
  Structure::addResidue(symbol,coord, 0);

  //if (success == 0) {
    //printf("   Nope2\n");
    //printf("<=AlignedStructure::addResidue\n");
    //return;
  //}

  alignedToUnaligned[getSize()-1] = unalignedLength;
  unalignedToAligned[unalignedLength] = getSize()-1;
  unalignedLength++;

  //printf("   Yep1\n");
  //printf("<=AlignedStructure::addResidue\n");

  return;
}


// addGap
void AlignedStructure::addGap() {

  //printf("=>AlignedStructure::addGap()\n");

  //int success = 0;
  Structure::addResidue(getAlphabet()->getGap(), Coordinate3D(), 0);

  //if (success == 0) return;
  if (unalignedLength <= 20000) {
    alignedToUnaligned[getSize()-1] = unalignedLength;
  }
  else {
    alignedToUnaligned[getSize()-1] = -1;
  }

  //printf("<=AlignedStructure::addGap()\n");

  return;
}


Symbol AlignedStructure::getUnalignedSymbol(int i) {

  if (i < unalignedLength &&
      unalignedToAligned[i] >= 0) {
    return get(unalignedToAligned[i]);
  }

  return 0;
}


Coordinate3D AlignedStructure::getUnalignedCoordinate(int i) {
  
  if (i < unalignedLength &&
      unalignedToAligned[i] >= 0) {
    return getCoordinate(unalignedToAligned[i]);
  }
  
  return Coordinate3D();
}


int AlignedStructure::getUnalignedLength() {

  return unalignedLength;
}


// alignedToUnalignedIndex
//   return index of non-gapped sequence corresponding to
//   the given gapped sequence index
int AlignedStructure::alignedToUnalignedIndex(int i) {

  if (i < getSize()) {
    return alignedToUnaligned[i];
  }

  return -1;
}


// unalignedToAlignedIndex
//   return index of gapped sequence corresponding to
//   the given non-gapped sequence index
int AlignedStructure::unalignedToAlignedIndex(int i) {

  if (i < unalignedLength) {
    return unalignedToAligned[i];
  }

  return -1;
}
