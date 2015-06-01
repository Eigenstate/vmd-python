/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: Stack.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.26 $	$Date: 2010/12/16 04:08:41 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A template class which implements a simple stack of arbitrary data.
 * Items are pushed onto and popped off of the stack, or may just be
 * copied from the stack.
 *
 ***************************************************************************/
#ifndef STACK_TEMPLATE_H
#define STACK_TEMPLATE_H

/// A template class which implements a simple stack of arbitrary data.
/// Items are pushed onto and popped off of the stack, or may just be
/// copied from the stack.
template<class T>
class Stack {

private:
  T *data;     ///< list of items
  T *curr;     ///< pointer to the current item
  int sz;      ///< maximum number of items the stack can hold
  int items;   ///< current number of items on the stack
  
public:
  ////////////////////  constructor
  Stack(int s) {
    items = 0;
    data = curr = new T[sz = (s > 0 ? s : 1)];
  }


  ////////////////////  destructor
  ~Stack(void) {
    if(data)
      delete [] data;
  }

  int max_stack_size(void) { return sz; }  ///< return max stack size
  int stack_size(void) { return items; }   ///< return current stack size 
  int num(void) { return items; }          ///< return current stack size

  /// push a new item onto the stack.
  int push(const T& a) {
    if (items < sz) {
      *curr++ = a;
      items++;
    } else {
      return -1;  // stack size exceede
    }
    return 0;
  }

  
  /// duplicate the top element on the stack
  int dup(void) {
    if (items > 0) {
      return push(top());
    } else if(sz > 0) {
      curr++;
      items++;
    }
    return 0;  // success
  }


  /// pop an item off the stack, returning its reference
  T& pop(void) {
    if (items > 0) {
      items--;
      return (*--curr);
    } else {
      return *data;
    }
  }


  /// return reference to the top item, but do not pop the stack
  T& top(void) {
    if (items > 0) {
      return *(curr - 1);
    } else {
      return *data;
    }
  }

};

#endif

