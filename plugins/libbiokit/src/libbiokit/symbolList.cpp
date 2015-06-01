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
#include "alphabet.h"
#include "symbolList.h"

/* ---------------------------------------------------------------- */
SymbolList::SymbolList(Alphabet* alphabet, int initialSize)
: alphabet(alphabet), size(0), maxSize(initialSize), data(NULL)
{
    data = new unsigned char[maxSize];
}

/* ---------------------------------------------------------------- */
SymbolList::~SymbolList()
{
    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }
}

/* ---------------------------------------------------------------- */
SymbolList::SymbolList(const SymbolList &s) :
                   alphabet(s.alphabet), size(s.size), maxSize(s.maxSize)
{
   data = new unsigned char[maxSize];
   // Copy the existing data.
   memcpy(data, s.data, sizeof(unsigned char)*size);

}


/* ---------------------------------------------------------------- */
// symbolsEqual
//   Returns true if the Symbols are the same, element-by-element,
//   in both SymbolLists; returns false, otherwise
bool SymbolList::operator==(SymbolList* compareTo) {
   int i;
  if (getSize() != compareTo->getSize()) {
    return false;
  }

  for (i=0; i<getSize(); i++) {
    if ( get(i) != compareTo->get(i) ) {
      return false;
    }
  }

  return true;
}

/* ---------------------------------------------------------------- */
bool SymbolList::operator!=(SymbolList* compareTo) {
  return !(*this == compareTo);
}

/* ---------------------------------------------------------------- */
// nongapSymbolsEqual
//   Returns true if the Symbols are the same, element-by-element
//   ignoring gaps, in both SymbolLists; returns false, otherwise
bool SymbolList::nongapSymbolsEqual(SymbolList* compareTo) {
  
    int i=0;   // index for this
    int j=0;   // index for compareTo
    while (i < getSize() && j < compareTo->getSize()) {
        if ( !alphabet->isGap(get(i)) && !alphabet->isGap(compareTo->get(j)) ) {
            
           if (get(i) != compareTo->get(j) &&
                              !alphabet->isUnknown(get(i)) &&
                              !alphabet->isUnknown(compareTo->get(j))) {
              return false;
           }
           i++;
           j++;
            
        } else if ( alphabet->isGap(get(i)) ) {
           i++;
        } else if ( alphabet->isGap(compareTo->get(j)) ) {
           j++;
        }
    }
    
    // Make sure tail is all gaps
    while (i < getSize()) {
        if (alphabet->isGap(get(i)))
            i++;
        else {
            return false;
        }
    }
    
    // Make sure tail is all gaps
    while (j < compareTo->getSize()) {
        if (alphabet->isGap(compareTo->get(j)))
            j++;
        else {
            return false;
        }
    }
    
    return true;
}


/* ---------------------------------------------------------------- */
Alphabet* SymbolList::getAlphabet()
{
    return alphabet;
}

/* ---------------------------------------------------------------- */
int SymbolList::getSize()
{
    return size;
}

/* ---------------------------------------------------------------- */
void SymbolList::add(char c)
{
//   printf("%c:%d,",c,alphabet->getSymbolIndex(c));
    addSymbolIndex(alphabet->getSymbolIndex(c));
}

/* ---------------------------------------------------------------- */
void SymbolList::addAll(const char* chars)
{
   int i;
    for (i=0; i<(int)strlen(chars); i++)
        addSymbolIndex(alphabet->getSymbolIndex(chars[i]));
}

/* ---------------------------------------------------------------- */
void SymbolList::add(const char* c)
{
    addSymbolIndex(alphabet->getSymbolIndex(c));
}

/* ---------------------------------------------------------------- */
void SymbolList::add(Symbol& symbol)
{
    addSymbolIndex(alphabet->getSymbolIndex(symbol));
}

/* ---------------------------------------------------------------- */
void SymbolList::addSymbolIndex(int symbolIndex)
{
    // If we don't have room, allocate a bigger array and copy the data.
    if (size >= maxSize)
    {
        // Allocate a bigger array.
        int newMaxSize = maxSize+(maxSize/2);
        unsigned char* newData = new unsigned char[newMaxSize];
        
        // Copy the existing data.
        memcpy(newData, data, sizeof(unsigned char)*size);
        
        // Delete the old data.
        delete[] data;
        data = newData;
        maxSize = newMaxSize;
    }
    
    // Add the item to the end of the list.
    data[size++] = (unsigned char)symbolIndex;
}

/* ---------------------------------------------------------------- */
void SymbolList::optimize()
{
   if (size != maxSize) {
      // Allocate a bigger array.
      int newMaxSize = size;
      unsigned char* newData = new unsigned char[newMaxSize];

      // Copy the existing data.
      memcpy(newData, data, sizeof(unsigned char)*size);

      // Delete the old data.
      delete[] data;
      data = newData;
      maxSize = newMaxSize;
   }
}



/* ---------------------------------------------------------------- */
Symbol& SymbolList::get(int index)
{
  //printf(">get\n");
  if (index >= 0 && index < size) {
    //printf("<get\n");
    return alphabet->getSymbol((int)data[index]);
  }
  printf("<get Unknown\n");
  return alphabet->getUnknown();
}
/* ---------------------------------------------------------------- */


