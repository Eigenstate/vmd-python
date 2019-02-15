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
 *	$RCSfile: SortableArray.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.26 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   This is a variant of the Resizeable array class.  In addition to the
 * methods provided there, this template allows either quicksorting 
 * or insertion sorting of the elements in the array
 *
 ***************************************************************************/
#ifndef SORTABLEARRAY_TEMPLATE_H
#define SORTABLEARRAY_TEMPLATE_H

#include <string.h>

/// A sort-capable variant of the ResizeArray template class. 
/// In addition to the methods provided there, this template also
/// allows either quicksorting or insertion sorting of the elements 
/// in the array.
template<class T>
class SortableArray {
private:
  /// list of items, and pointer to current item.
  T *data;

  /// max number of items that can be stored in the array
  int sz;
  
  /// factor by which to make array larger, if must extend
  float resizeFactor;
  
  /// largest number of items that have been accessed so far.
  /// this is the largest index used, + 1
  int currSize;


public:
  SortableArray(int s = 10, float rsf = 2.0) {
    /// save factor by which to increase array if necessary
    resizeFactor = rsf;
    
    /// initialize values for current array size and max array size
    currSize = 0;
    sz = (s > 0 ? s : 10);
    
    /// allocate space for maximum size of array
    data = new T[sz];
  }

  ~SortableArray(void) {
    if(data)
      delete [] data; ///< free up the current array, if necessary
  }

  //
  // query routines about the state of the array
  //
    
  /// largest number of elements that have been accessed
  int num(void) { return currSize; }

  //
  // routines for accessing array elements
  //

  /// [] operator version of the item() routine
  T& operator[](int N) { return item(N); }

  /// add a new element to the end of the array.  Return index of new item.
  int append(const T& val) {
    item(currSize) = val;
    return currSize - 1;
  }

  /// return the nth item; assume N is >= 0
  T& item(int N) {
    // check and see if this is attempting to access an item larger than
    // the array.  If so, extend the max size of the array by resizeFactor,
    // and return the value at the Nth position (which will be basically
    // random).
    if(N >= sz) {			// extend size of array if necessary
      // calc new size of array
      int newsize = (int)((float)N * resizeFactor + 0.5);
  
      // allocate new space, and copy over previous copy.  We only need to
      // copy over the first currSize elements, since currSize is the max
      // index that has been accessed (read OR write).  Then delete old
      // storage.
      T *newdata = new T[newsize];
      if(data) {
        memcpy((void *)newdata, (void *)data, currSize * sizeof(T));
        delete [] data;
      }
      
      // save new values
      data = newdata;
      sz = newsize;
    }
    
    // remember what the maximum index reached so far is
    if(N >= currSize)
      currSize = N + 1;
      
    // return element at Nth position
    return data[N];
  }

  /// remove the Mth ... Nth items: move all items lower in the
  /// array up to fill the empty slots.
  /// If both arguments are < 0, removes ALL items.
  /// If the second argument is < 0, just removes item m
  void remove(int m = -1, int n = -1) {
    if(m < currSize && n < currSize) {
      if((m < 0 && n < 0) || (m >= 0 && n < 0) || (m > 0 && n > 0 && m <= n)) {
        int N = n, M = (m >= 0 ? m : 0);
        if(m < 0 && n < 0)
          N = (currSize - 1);
        else if(n < 0)
          N = M;
        else
          N = n;
        int removed = N - M + 1;
        for(int i=N+1; i < currSize; i++)
          data[i - removed] = data[i];
        currSize -= removed;
      }
    }
  }

  /// auxilliary function for qsort
  void swap (int i, int j) {
   T temp;
  
   temp = data[i];
   data[i]=data[j];
   data[j]=temp;
  }

  /// sorts elements based on a specialied field.  To change
  /// the sort, change the field which is used as a basis for
  /// comparison.  The '>' and '<' symbols are overloaded.
  void qsort (int left, int right) {
    int i, last;
  
    if (left >= right)    // fewer than two elements
       return;
    swap (left, (left + right) /2);
    last = left;
    for (i = left+1; i <=right; i++)
      if (data[i] > data[left]) 
       swap (++last, i);
    swap (left, last);
    qsort (left, last-1);
    qsort (last+1, right);
  }

  /// This is an internal sort
  void isort () {
     int i, j;
     T temp;
     for (i = currSize-2; i>=0; i--) {  // jan 16 added = to >=
       j = i+1;
       temp = data[i];
       while ((j < currSize) && (temp < data[j])) {
         data[j-1] = data[j];
         j = j+1;
       }
       data[j-1] = temp;
     }
  }
};

#endif
