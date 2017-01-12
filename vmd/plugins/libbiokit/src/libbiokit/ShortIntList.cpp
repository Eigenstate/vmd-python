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
 * Author(s): Elijah Roberts
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "ShortIntList.h"

const unsigned short ShortIntList::MAX     = USHRT_MAX-1;
const unsigned short ShortIntList::INVALID = USHRT_MAX;

/**
 *
 */
// ---------------------------------------------------------------------------
ShortIntList::ShortIntList(int initialSize)
: size(0), data(NULL)
{
    maxSize = initialSize;
    data = new unsigned short[maxSize];
}

// ---------------------------------------------------------------------------
ShortIntList::ShortIntList(const ShortIntList &sil) :
                                 size(sil.size), maxSize(sil.maxSize)
{
   data = new unsigned short[maxSize];
   // Copy the existing data.
   memcpy(data, sil.data, sizeof(unsigned short)*size);
}



// ---------------------------------------------------------------------------
ShortIntList::~ShortIntList()
{
    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }
}


// ---------------------------------------------------------------------------
void ShortIntList::initialize(const int newSize, unsigned short initValue)
{
    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }
    data = new unsigned short[newSize];


    for (int i=0; i<newSize; i++) {
       data[i] = initValue;
    }
    size = maxSize = newSize;
}

// ---------------------------------------------------------------------------
void ShortIntList::add(unsigned short item)
{
    // If we don't have room, allocate a bigger array and copy the data.
    if (size >= maxSize)
    {
        // Allocate a bigger array.
        int newMaxSize = maxSize*2;
        unsigned short* newData = new unsigned short[newMaxSize];
        
        // Copy the existing data.
        memcpy(newData, data, sizeof(unsigned short)*size);
        
        // Delete the old data.
        delete[] data;
        data = newData;
        maxSize = newMaxSize;
    }
    
    // Add the item to the end of the list.
    data[size++] = item;
}

// --------------------------------------------------------------------------
void ShortIntList::optimize()
{
   if (size != maxSize) {
      // Allocate a bigger array.
      int newMaxSize = size;
      unsigned short* newData = new unsigned short[newMaxSize];
        
      // Copy the existing data.
      memcpy(newData, data, sizeof(unsigned short)*size);
        
      // Delete the old data.
      delete[] data;
      data = newData;
      maxSize = newMaxSize;
   }
}

// --------------------------------------------------------------------------
unsigned short ShortIntList::get(int index)
{
    if (index >= 0 && index < size)
    {
        return data[index];
    }
    
    return INVALID;
}

// ---------------------------------------------------------------------------
void ShortIntList::set(int index, unsigned short item)
{
    if (index >= 0 && index < size)
    {
        data[index] = item;
    }
}

// ---------------------------------------------------------------------------
void ShortIntList::printList()
{
   int i;
   for (i=0; i < size; i++)
   {
      printf("%u ", data[i]);
   }
   printf("\n");
}

// ---------------------------------------------------------------------------
int ShortIntList::getSize()
{
    return size;
}
// ---------------------------------------------------------------------------


