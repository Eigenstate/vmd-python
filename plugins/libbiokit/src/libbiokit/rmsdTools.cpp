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
 * Author(s): John Eargle, Elijah Roberts
 */

#include "Atom.h"
#include "Residue.h"
#include "rmsdTools.h"


// Constructor
//
RmsdTools::RmsdTools(StructureAlignment* sa)
  : alignment(sa) {
  
  int structCount = alignment->getNumberStructures();
  int i, j;

  rmsdScores = new float*[structCount];
  for (i=0; i<structCount; i++) {
    rmsdScores[i] = new float[structCount];
    for (j=0; j<structCount; j++) {
      rmsdScores[i][j] = 0;
    }
  }
  rmsdPerRes = new float*[structCount];
  for (i=0; i<structCount; i++) {
    rmsdPerRes[i] = new float[alignment->getNumberPositions()];
    for (j=0; j<alignment->getNumberPositions(); j++) {
      rmsdPerRes[i][j] = 0.0;
    }
  }
}


// Destructor
//
RmsdTools::~RmsdTools() {
  
  delete rmsdScores;
  delete rmsdPerRes;

  return;
}


// rmsd
//
int RmsdTools::rmsd() {

  int structCount = alignment->getNumberStructures();
  float distance = 0.0;

  float** norms = new float*[structCount];
  int i=0, j;
  for (i=0; i<structCount; i++) {
    norms[i] = new float[structCount];
    for (j=0; j<structCount; j++) {
      norms[i][j] = 0;
    }
  }
 
  int row1, row2, col;
  for (row1=0; row1<structCount; row1++) {
    for (row2=row1+1; row2<structCount; row2++) {
      for (col=0; col<alignment->getNumberPositions(); col++) {
	if ( !(alignment->getAlphabet()->isGap(alignment->getSymbol(row1,col))) &&
	     !(alignment->getAlphabet()->isGap(alignment->getSymbol(row2,col))) ) {
              Coordinate3D coord1 = alignment->getCoordinate(row1,col);
              Coordinate3D coord2 = alignment->getCoordinate(row2,col);
              distance = coord1.getDistanceTo(coord2);
              rmsdScores[row1][row2] += pow(distance,2);
              rmsdScores[row2][row1] += pow(distance,2);
              norms[row1][row2]++;
              norms[row2][row1]++;
            }
      }
    }
  }

  for (i=0; i<structCount; i++) {
    for (j=0; j<structCount; j++) {
      if (i != j) {
	if (norms[i][j] > 0) {
	  rmsdScores[i][j] /= norms[i][j];
	  rmsdScores[i][j] = sqrt(rmsdScores[i][j]);
	}
	else {
	  rmsdScores[i][j] = 0;
	  printf("Error - RmsdTools::rmsd, divide by zero\n");
	  printf("   norms[%d][%d]\n",i,j);
	}
      }
    }
  }

  delete norms;

  return 1;
}


// rmsdPerResidue
//   Calculate RMSD between aligned residues of two structures;
//   -1 if there is at least one gap in the column
int RmsdTools::rmsdPerResidue() {
   int row, col;
  for (row=1; row<alignment->getNumberStructures(); row++) {
      for (col=0; col<alignment->getNumberPositions(); col++) {
        if ( !(alignment->getAlphabet()->isGap(alignment->getSymbol(0,col))) && !(alignment->getAlphabet()->isGap(alignment->getSymbol(row,col))) ) {
          Coordinate3D coord1 = alignment->getCoordinate(0,col);
          Coordinate3D coord2 = alignment->getCoordinate(row,col);
          rmsdPerRes[row][col] = coord1.getDistanceTo(coord2);
        }
        else {
          rmsdPerRes[row][col] = -1.0;
        }
      }
  }

  return 1;
}


// printRmsd
//
int RmsdTools::printRmsd(FILE* outfile) {
   int i, j;
  if (rmsdScores == 0) {
    printf("Error: RmsdTools::printRmsd\n");
    return 0;
  }

  int structCount = alignment->getNumberStructures();
  
  for (i=0; i<structCount; i++) {
    for (j=0; j<structCount; j++) {
      fprintf(outfile," %6.4f", rmsdScores[i][j]);
    }
    fprintf(outfile,"\n");
  }

  return 1;
}


// printRmsdPerResidue
//
int RmsdTools::printRmsdPerResidue(FILE* outfile) {
   int i, j;
  if (rmsdPerRes == 0) {
    printf("Error: RmsdTools::printRmsdPerResidue\n");
    return 0;
  }

  for (i=0; i<alignment->getNumberStructures(); i++) {
      for (j=0; j<alignment->getNumberPositions(); j++) {
        fprintf(outfile,"%6.4f ", rmsdPerRes[i][j]);
      }
      fprintf(outfile,"\n");
  }

  return 1;
}

/**
 * Calculates the contact order of a structure.   
 *
 * @param structure The structure.
 * @param distanceCutoff The maximum distance between two residues from them to be considered in contact.
 * @param sequenceCutoff The minimum separation in the sequence for two residues to be considered contacts.
 * @return The number of contacts present in the structure.
 */
/*
double RmsdTools::getRMSD(Structure* referenceStructure, Structure* comparisonStructure)
{
   int i, j, row1, row2, col;
    if (referenceStructure == NULL || comparisonStructure == NULL || referenceStructure->getSize() != comparisonStructure->getSize())
        return -1.0;
    
    int numberResidues = referenceStructure->getSize();
    for (i=0; i<numberResidues; i++)
    {
        Residue* residue1 = referenceStructure->getResidue(i);
        Residue* residue2 = comparisonStructure->getResidue(i);
        if (residue1 == NULL || residue2 == NULL || residue1->getNumberAtoms() != residue2->getNumberAtoms())
            return -1.0;
        
        int numberAtoms = residue1->getNumberAtoms();
        for (j=0; j<numberAtoms; j++)
        {
            Atom* atom1 = residue1->getAtom(j);
            Atom* atom2 = residue2->getAtom(j);
            double distance = atom1->getDistanceTo(*atom2);  
        }
    }
    
    return 0.0;
}
*/        
  /*for (row1=0; row1<structCount; row1++) {
    for (row2=row1+1; row2<structCount; row2++) {
      for (col=0; col<alignment->getNumberPositions(); col++) {
	if ( !(alignment->getAlphabet()->isGap(alignment->getSymbol(row1,col))) &&
	     !(alignment->getAlphabet()->isGap(alignment->getSymbol(row2,col))) ) {
              Coordinate3D coord1 = alignment->getCoordinate(row1,col);
              Coordinate3D coord2 = alignment->getCoordinate(row2,col);
              distance = coord1.getDistanceTo(coord2);
              rmsdScores[row1][row2] += pow(distance,2);
              rmsdScores[row2][row1] += pow(distance,2);
              norms[row1][row2]++;
              norms[row2][row1]++;
            }
      }
    }
  }

  for (i=0; i<structCount; i++) {
    for (j=0; j<structCount; j++) {
      if (i != j) {
	if (norms[i][j] > 0) {
	  rmsdScores[i][j] /= norms[i][j];
	  rmsdScores[i][j] = sqrt(rmsdScores[i][j]);
      */


double RmsdTools::getRadiusOfGyration(Structure* referenceStructure, Structure* comparisonStructure)
{
    return 0.0;
}

