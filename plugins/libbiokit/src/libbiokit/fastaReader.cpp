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
#include "ShortIntList.h"
#include "symbol.h"
#include "alphabet.h"
#include "alignedSequence.h"
#include "sequenceAlignment.h"
#include "fastaReader.h"


SequenceAlignment* FASTAReader::readSequenceAlignment(Alphabet* alphabet, 
                                                      const char* filename)
{
//    printf("fastaReader.readSequenceAlignment. file: %s\n", filename);
    FILE* infile = fopen(filename, "r");
    if (infile == NULL) return NULL;
    
    // Create the alignment.
    SequenceAlignment* alignment = NULL;
    
    // Go through the file and parse out the records.
    int maxLineSize = 65534;
    char *line = new char[maxLineSize+1];
    AlignedSequence* currentSequence = NULL;
    while (!feof(infile)) {
      
        // Get the next line.
        if (fgets(line, maxLineSize, infile) == NULL)
            break;
    
        // Remove any trailing newline characters.
        int last = strlen(line)-1;
        while (line[last] == '\r' || line[last] == '\n')
        {
            line[last--] = '\0';
        }
        
        // If this is a new sequence.
        if (line[0] == '>')
        {
            // If we do't have an alignment yet, create one.
            if (alignment == NULL)
            {
                alignment = new SequenceAlignment();
            }
            
            // If we already have a sequence, add it to the alignment.
            if (currentSequence != NULL)
            {
                alignment->addSequence(currentSequence);
                currentSequence = NULL;
            }
            
            // Get the sequence name.
            char* name = line+1;
            char* firstSpace = strstr(name, " ");
            if (firstSpace != NULL) *firstSpace = '\0';
            char* firstBar = strstr(name, "|");
            while (firstBar != NULL)
            {
                name = firstBar+1;
                firstBar = strstr(name, "|");
            }
            
            // Create the sequence.
            currentSequence = new AlignedSequence(alphabet, name);
        }
        
        // Otherwise if must be sequence data, if we are collecting it.
        else if (currentSequence != NULL)
        {
            // Add the sequence data.
            currentSequence->addAll(line);
        }
    }
    
    // Add the last sequence to the alignment.
    if (currentSequence != NULL)
    {
        alignment->addSequence(currentSequence);
        currentSequence = NULL;
    }
    
    // Clean up any resources used.
    fclose(infile);
    delete[] line;
    
    return alignment;
}

