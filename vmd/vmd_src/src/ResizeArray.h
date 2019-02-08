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
 *	$RCSfile: ResizeArray.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.53 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Automatically-adjusting single-dim array template.
 * 
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/
#ifndef RESIZEARRAY_TEMPLATE_H
#define RESIZEARRAY_TEMPLATE_H

#include <string.h>

/// A template class which implements a dynamically-growing, automatically
/// resizing array of data of a given type.  Elements in the array may be
/// accessed via the [] operator.  When new data is added to the end of an
/// array, the size of the array is automatically increased if necessary.
///
/// XXX Do not parametrize this class with a datatype which cannot be
///     shallow-copied!  This class uses memcpy to resize, and therefore
///     classes which contain dynamically-allocated memory blocks will
///     crash and burn if the ResizeArray ever gets resized.
template<class T>
class ResizeArray {
private:
  T *allocate(size_t n) { return new T[n]; }
  void deallocate(T *p) { delete [] p; }

  T *data;       ///< list of items, and pointer to current item.
  long sz;       ///< max number of items that can be stored in the array
  long currSize; ///< largest index used + 1


  /// resize array to accomodate up to addN new elements.
  void extend(long addN) {
    if ((currSize+addN) >= sz) {    // extend size of array if necessary
      // guarantee minimum required size increase addN, since the scaled value
      // may truncate back to the original size value when the initial number
      // of elements is very small.
      long newsize = ((long)((float)sz * 1.3f)) + addN;

      // shallow copy data to a newly allocated block since we can't
      // use realloc() due to potential use of shared memory
      T *newdata = allocate(newsize); 
      memcpy(newdata, data, currSize * sizeof(T));
      deallocate(data); 
    
      // save new values
      data = newdata;
      sz = newsize;
    }
  }

public:
  /// Constructor
  /// The first argument is the initial internal size of the array, i.e. the
  /// initial number of elements for which to allocate memory (although the
  /// initial external size of the array will be zero).  
  ResizeArray(long s = 3L) {
    currSize = 0;
    sz = (s > 0 ? s : 10L);
    data = allocate(sz); 
  }

  ~ResizeArray() {
    deallocate(data);
  }
  
  long num(void) const { return currSize; } ///< current size of array 
  T& operator[](long N) { return data[N]; } ///< unchecked accessor, for speed
  T const& operator[](long N) const { return data[N]; } ///< a const version of above


  /// add a new element to the end of the array.
  void append(const T& val) {
    if (currSize == sz) {    // extend size of array if necessary
#if 1
      long newsize = (long)((float)sz * 1.3f);

      // guarantee minimum required size increase, since the scaled value
      // may truncate back to the original size value when the initial number
      // of elements is very small.
      if (newsize == sz)
        newsize++;
#else
      extend(1);
#endif

      // shallow copy data to a newly allocated block since we can't
      // use realloc() due to potential use of shared memory
      T *newdata = allocate(newsize); 
      memcpy(newdata, data, currSize * sizeof(T));
      deallocate(data); 
    
      // save new values
      data = newdata;
      sz = newsize;
    }
    data[currSize++] = val;
  }


  /// add two new elements to the end of the array, e.g. angles.
  void append2(const T& vala, const T& valb) {
    extend(3);
    data[currSize++] = vala;
    data[currSize++] = valb;
  }


  /// add three new elements to the end of the array, e.g. angles.
  void append3(const T& vala, const T& valb, const T& valc) {
    extend(3);
    data[currSize++] = vala;
    data[currSize++] = valb;
    data[currSize++] = valc;
  }


  /// add four new elements to the end of the array, e.g. dihedrals/impropers.
  void append4(const T& vala, const T& valb, const T& valc, const T& vald) {
    extend(4);
    data[currSize++] = vala;
    data[currSize++] = valb;
    data[currSize++] = valc;
    data[currSize++] = vald;
  }


  /// add N elements to the end of the array.
  void appendN(const T& val, long addN) {
    extend(addN);
    long j;
    for (j=0; j<addN; j++)  
      data[currSize++] = val;
  }


  /// add three new elements, e.g. vertex/normal/color to the end of the array.
  void append3(const T *vals) {
    extend(3);
    data[currSize++] = vals[0];
    data[currSize++] = vals[1];
    data[currSize++] = vals[2];
  }


  /// add two groups of three new elements to the end of the array
  void append2x3(const T *valsa, const T *valsb) {
    extend(6);
    data[currSize++] = valsa[0];
    data[currSize++] = valsa[1];
    data[currSize++] = valsa[2];

    data[currSize++] = valsb[0];
    data[currSize++] = valsb[1];
    data[currSize++] = valsb[2];
  }


  /// add three groups of three new elements to the end of the array
  void append3x3(const T *valsa, const T *valsb, const T *valsc) {
    extend(9);
    data[currSize++] = valsa[0];
    data[currSize++] = valsa[1];
    data[currSize++] = valsa[2];

    data[currSize++] = valsb[0];
    data[currSize++] = valsb[1];
    data[currSize++] = valsb[2];

    data[currSize++] = valsc[0];
    data[currSize++] = valsc[1];
    data[currSize++] = valsc[2];
  }


  /// add nine new elements, e.g. v0+v1+v2, vertex+normal+color,
  /// to the end of the array.  
  void append9(const T *vals) {
    extend(9);
    data[currSize++] = vals[0];
    data[currSize++] = vals[1];
    data[currSize++] = vals[2];
    data[currSize++] = vals[3];
    data[currSize++] = vals[4];
    data[currSize++] = vals[5];
    data[currSize++] = vals[6];
    data[currSize++] = vals[7];
    data[currSize++] = vals[8];
  }


  /// add N elements to the end of the array.
  void appendlist(const T *vals, long addN) {
    extend(addN);
    long j;
    for (j=0; j<addN; j++)  
      data[currSize++] = vals[j];
  }


  /// remove an item from the array, shifting remaining items down by 1
  void remove(long n) {
    if (n < 0 || n >= currSize) return;
    for (long i=n; i<currSize-1; i++)
      data[i] = data[i+1];
    currSize--;
  }

  /// remove the last item from the array, unchecked for speed
  T& pop() {
    currSize--;
    return data[currSize]; 
  }

  /// delete entire array by defining size to be empty
  void clear() {
    currSize = 0;
  }

  /// truncate the array by defining the size to be N items less
  void truncatelastn(long N) {
    currSize -= N;
    if (currSize < 0) 
      currSize=0;
  }

  /// scan the array until the first item that matches in the array is
  /// found.  Return the index if found, (-1) otherwise.
  long find(const T& val) {
    long i;
  
    for(i=0; i < currSize; i++) {
      if(data[i] == val) 
        return i;
    }
  
    return -1;
  }
};

#endif

