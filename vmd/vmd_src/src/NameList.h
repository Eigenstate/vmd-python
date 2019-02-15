/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *        $RCSfile: NameList.h,v $
 *        $Author: johns $        $Locker:  $                $State: Exp $
 *        $Revision: 1.49 $        $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * NameList template, which stores a list of unique names indexed in the order
 * they are added.  For each name, which acts as a key, there is an associated
 * integer value.  NameList string lookups are accelerated through the use of
 * an internal hash table.
 *
 ***************************************************************************/
#ifndef NAMELIST_TEMPLATE_H
#define NAMELIST_TEMPLATE_H

#include <string.h>
#include "ResizeArray.h" 
#include "utilities.h"   // needed for stringdup()
#include "hash.h"        // needed for hash table functions

#define NLISTSIZE 64

/// Template class, stores a list of unique names indexed in the order
/// they are added.  For each name, which acts as a key, there is an associated
/// integer value.  NameList string lookups are accelerated through the use of
/// an internal hash table.
template<class T>
class NameList  {
protected:
  int Num;                   ///< number of items in the list
  ResizeArray<char *> names; ///< array of pointers to the strings; 
  ResizeArray<T> Data;       ///< integer data for the items
  hash_t hash;               ///< hash table used to accelerate finds etc.
 
public:
  ////////////////////  constructor
  // starts with no names, which are then added via the 'add_name' routine.  
  NameList(void) : names(NLISTSIZE), Data(NLISTSIZE) {
    Num = 0;
    hash_init(&hash, 127);
  }

  ////////////////////  destructor
  virtual ~NameList(void) {
    for(int i=0; i < Num; i++) {
      if(names[i])
        delete [] names[i];
    }
    hash_destroy(&hash);
  }

  int num(void) const { return Num; }    // return number of items

  // clear list. equivalent to running destructor and constructor.
  void clear(void) {
    for(int i=0; i < Num; i++) {
      if(names[i])
        delete [] names[i];
    }
    hash_destroy(&hash);
    Data.clear();
    names.clear();
    // start over.
    Num = 0;
    hash_init(&hash, 127);
  }

  // add a new name to the list, with a given associated value.
  // Return the index.  If already in the list, return the current index.
  int add_name(const char *nm, const T &val) {
    char tmpnm[128]; // temporary storage, spaces stripped from beginning + end
    memset(tmpnm, 0, sizeof(tmpnm)); // clear temp storage
 
    if (!nm)
      return (-1);      

    // strip leading and trailing spaces from the name
    char *s = tmpnm;
    while (*nm && *nm == ' ')           // skip past whitespace
      nm++;

    int len = 127;
    while (*nm && len--)                // copy the string
      *(s++) = *(nm++);

    *s = '\0';                          // terminate the copied string
    while(s != tmpnm && *(--s) == ' ')  // remove spaces at end of string
      *s = '\0';

    int myindex;  
    if ((myindex = hash_lookup(&hash, tmpnm)) != HASH_FAIL) {
      return myindex; 
    } 

    // if here, string not found; append new one, and return index = Num - 1
    names.append(stringdup(tmpnm));

    myindex = hash_insert(&hash, names[Num], Num); 

    Data.append(val);
    return Num++;
  }



  // return the name (null-terminated) for given typecode
  const char * name(int a) const {
    if (a >= 0 && a < Num) {
      return names[a];
    }
    return NULL;        // not found
  }


  // return the type index for the given name.  If the second argument is
  // given and is > 0, it is used as the max length of the names to check
  // for a match.  If is is <= 0, an exact match must be found.
  //        returns (-1) if no match is found
  int typecode(const char *nm) const {
    if (!nm)
      return -1;

    return hash_lookup(&hash, nm);  // returns -1 (HASH_FAIL) on no entry
  }


  // returns the data for the given name.  If the second argument is
  // given and is > 0, it is used as the max length of the names to check
  // for a match.  If is is <= 0, an exact match must be found.
  T data(const char *nm) const {
    if (!nm)
      return Data[0];
 
    int myindex = hash_lookup(&hash, nm);
    if (myindex != HASH_FAIL)
      return Data[myindex];
  
    return Data[0];
  }


  // returns the data for the given index
  T data(int a) const {
    if (a >= 0 && a < Num) {
      return Data[a];
    }
    return Data[0];
  }


  // set the data value for the given index
  void set_data(int a, const T &val) {
    if(a >= 0 && a < Num) {
      Data[a] = val;
    }
    // else it was an illegal index, therefore do nothing.
  }

  // change the name of an entry
  void set_name(int a, const char *nm) {
    if (a < 0 || a >= Num) 
      return;

    // delete the hash table entry...
    hash_delete(&hash, names[a]);
    delete [] names[a];
    names[a] = stringdup(nm);
    // and put the pointer in our ResizeArray into the hash table
    hash_insert(&hash, names[a], a);
  } 

};


// useful typedefs for making NameLists of NameLists
typedef NameList<int>           *NameListIntPtr;
typedef NameList<float>         *NameListFloatPtr;
typedef NameList<char>          *NameListCharPtr;
typedef NameList<char *>        *NameListStringPtr;

#endif

