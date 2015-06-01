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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sequence.h"

Sequence::Sequence(Alphabet* alphabet, const char* seqName)
: SymbolList(alphabet), name (NULL)
{
   setName(seqName);
}

Sequence::~Sequence()
{
    if (name != NULL)
    {
        delete[] name;
        name = NULL;
    }
}

const char* Sequence::getName()
{
//   printf("returning name %s\n", name);
    return name;
}

void Sequence::setName(const char *newName)
{
//   printf("Setting name from %s to %s\n", name, newName);
   // delete old name if it exists
    if (name != NULL)
    {
        delete[] name;
        name = NULL;
    }

   // if we got a new name, set object's name to it
    if (newName != NULL)
    {
        name = new char[strlen(newName)+1];
        strcpy(name, newName);
    }
}

Sequence::Sequence(const Sequence &s) : SymbolList(s)
{
   setName(s.name);
}

