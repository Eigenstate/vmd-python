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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "typeConvert.h"
#include "Atom.h"
#include "alphabet.h"
#include "coordinate3D.h"
#include "Residue.h"
#include "structure.h"
#include "pdbReader.h"

// --------------------------------------------------------------------

Structure* PDBReader::readStructure(Alphabet* alphabet, char* filename, const char* backboneAtomName) {

    FILE* infile = fopen(filename, "r");
    
    if (infile == NULL) {
        return 0;
    }
    
    Structure* structure = new Structure(alphabet, filename);

    // Create the entries from the PDB file.
    char recordType[7];
    char serialNumber[6];
    char atomName[5];
    char altLoc[2];
    char residueName[4];
    char chain[2];
    char resID[5];
    char insertion[2];
    char x[9];
    char y[9];
    char z[9];
  
    // Go through the file and parse out the records.
    char line[1024];
    Coordinate3D backboneCoord;
    Residue* currentResidue = NULL;
    while (!feof(infile)) {
      
        // Get the next line.
        if (fgets(line, 1023, infile) == NULL)
            break;
      
        // Make sure this is an atom or hetatom record.
        parseField(recordType, line, 0, 6);
        if (strcmp(recordType, "ATOM") == 0 || strcmp(recordType, "HETATOM") == 0)
        {
            // Parse the fields.
            parseField(serialNumber, line, 6, 5);
            parseField(atomName, line, 12, 4);
            parseField(altLoc, line, 16, 1);
            parseField(residueName, line, 17, 3);
            parseField(chain, line, 21, 1);
            parseField(resID, line, 22, 4);
            parseField(insertion, line, 26, 1);
            parseField(x, line, 30, 8);
            parseField(y, line, 38, 8);
            parseField(z, line, 46, 8);
            
            // If this is a new residue, save the old one (if we have one) and start over.
            if (currentResidue == NULL || (strcmp(resID, currentResidue->getResID()) != 0 || strcmp(insertion, currentResidue->getInsertionName()) != 0))
            {
                if (currentResidue != NULL)
                {
                    structure->addResidue(currentResidue->getName(), backboneCoord, currentResidue);
                }
                
                // Start the new residue.
                backboneCoord.unset();
                currentResidue = new Residue(residueName, resID, insertion);
            }
            
            // If this is a backbone atom, save it.
            if (strcmp(atomName, backboneAtomName) == 0)
            {
                backboneCoord.set(charToFloat(x), charToFloat(y), charToFloat(z));
            }
            
            // Add the atom to the list.
            currentResidue->addAtom(new Atom(atomName, charToFloat(x),charToFloat(y),charToFloat(z)));
        }
        else if (strcmp(recordType, "TER") == 0)
        {
            break;
        }
    
    }
    
    // If we have one last residue, save it.
    if (currentResidue != NULL)
    {
        structure->addResidue(currentResidue->getName(), backboneCoord, currentResidue);
    }
                
    fclose(infile);

    return structure;
}

// --------------------------------------------------------------------
void PDBReader::parseField(char* destination, const char* source, int offset, int length)
{
    // Move to the first non-space character.
    while (length > 0 && source[offset] == ' ')
    {
        offset++;
        length--;
    }
    
    // Shorten to the last non-space character.
    while (length > 0 && source[offset+length-1] == ' ')
    {
        length--;
    }
    
    strncpy(destination, source+offset, length);
    destination[length] = '\0';
}

