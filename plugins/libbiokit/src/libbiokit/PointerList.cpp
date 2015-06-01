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
#include <string.h>
#include "PointerList.h"

/**
 *
 */
PointerList::PointerList(int initialSize)
: size(0), data(NULL)
{
    maxSize = initialSize;
    data = new void *[maxSize];
}

PointerList::~PointerList()
{
    if (data != NULL)
    {
        delete [] data;
        data = NULL;
    }
}


void PointerList::add(void* item)
{
    // If we don't have room, allocate a bigger array and copy the data.
    if (size >= maxSize)
    {
        // Allocate a bigger array.
        int newMaxSize = maxSize*2;
        void **newData = new void *[newMaxSize];
        
        // Copy the existing data.
//        memcpy(newData, data, sizeof(void *)*size);

        // we have c++ objects.  Have to copy them one at a time, allowing
        // their copy constructors to do their thing
        for (int i=0; i< size; i++) {
           newData[i] = data[i];
        }

        // Delete the old data.
        delete[] data;
        data = newData;
        maxSize = newMaxSize;
    }
    
    // Add the item to the end of the list.
    data[size++] = item;
}

void *PointerList::get(int index)
{
    if (index >= 0 && index < size)
    {
        return data[index];
    }
    
    return NULL;
}

void PointerList::set(int index, void* item)
{
    if (index >= 0 && index < size)
    {
        data[index] = item;
    }
}

int PointerList::getSize()
{
    return size;
}
