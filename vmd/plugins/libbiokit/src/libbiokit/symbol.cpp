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
#include "symbol.h"

Symbol::Symbol(char o, const char* t, const char* f)
: full(NULL)
{
    set(o, t, f);
}

Symbol::Symbol(const Symbol& copyFrom)
: full(NULL)
{
    set(copyFrom.one, copyFrom.three, copyFrom.full);
}

Symbol::~Symbol() {
  
    if (full != NULL)
    {
        //printf("Deleting2 full: %x\n", this->full);
        delete[] full;
        full = NULL;
    }
}

Symbol& Symbol::operator=(const Symbol& setFrom)
{
    set(setFrom.one, setFrom.three, setFrom.full);
    return *this;
}

void Symbol::set(char o, const char* t, const char* f)
{
    //printf("set: %s (%x) from %s (%x)\n", this->full, this->full, full, full);
    // Copy the one character code.
    one = o;
    
    // Copy the three character name.
    int i=0;
    for (; t != NULL && i < (int)strlen(t) && i < 3; i++)
        three[i] = t[i];
    for (; i<4; i++)
        three[i] = '\0';
    
    // If we are resetting the name, free it first.
    if (full != NULL)
    {
        //printf("Deleting1 full: %x\n", this->full);
        delete[] full;
        full = NULL;
    }
    
    // Copy the full name.
    if (f != NULL)
    {
        int len = strlen(f);
        full = new char[len+1];
        //printf("Allocated1 full: %x\n", this->full);
        strncpy(full, f, len);
        full[len] = '\0';
    }
    else
    {
        full = new char[1];
        //printf("Allocated2 full: %x\n", this->full);
        full[0] = '\0';
    }
}

bool Symbol::operator==(const Symbol& compareTo) const
{
    return (one == compareTo.one && strcmp(three, compareTo.three) == 0 && strcmp(full, compareTo.full) == 0);
}

bool Symbol::operator!=(const Symbol& compareTo) const
{
    return !(*this == compareTo);
}

const char Symbol::getOne()
{
    return one;
}

const char* Symbol::getThree()
{
    return three;
}

const char* Symbol::getFull()
{
    return full;
}

