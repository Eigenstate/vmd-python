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
 *	$RCSfile: P_Buttons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.25 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * A Buttons is a representation for a set of n boolean inputs.  This
 * fairly abstract class should be subclassed to make Buttons objects
 * that actually know how to get their buttons.  This is somewhat
 * parallel to the Tracker object, compare them!
 *
 ***************************************************************************/

#ifndef P_BUTTONS_H
#define P_BUTTONS_H

#define MAX_BUTTONS 100

#include "ResizeArray.h"
#include "P_SensorConfig.h"

/**
 * Common notes for VMDTracker, Feedback, and Buttons classes, collectively
 * referred to as devices in what follows.
 *
 * Constructor: The constructor must not require any input arguments.  The
 * reason for this is that an instance of every class is created and held
 * in an associative store so that it can be referenced by its device_name()
 * string.  This instantiation is done independently of any device 
 * configuration, such as what would be found in the .vmdsensors file.  
 * Constructors should thus do nothing but initialize member data to NULL or
 * default values.
 *
 * device_name(): This pure virtual function supplies a string by which the
 * device can be accessed in an associative store (since classes aren't
 * first-class objects in C++).  The name must be unique to that class,
 * among all devices of that type.
 *
 * clone(): This should do nothing more that return an instance of
 * the class.
 *
 * do_start(const SensorConfig *): Here's where the action is: This method
 * will be called from the base class start() method after general 
 * initialization is done.  This method is where the subclass should,
 * e.g., establish a connection to a remote device.  If there is no
 * class-specific initialization to do then the subclass need not override
 * this method.
 *
 * start() should be called only once in a device's life, when it is first
 * added to a Tool.  
 */


/// Buttons is a representation for a set of n boolean inputs.  This
/// fairly abstract class should be subclassed to make Buttons objects
/// that actually know how to get their buttons.  This is somewhat
/// parallel to the Tracker object, compare them!
class Buttons {
 protected: 
  ResizeArray<int> used; ///< the buttons that it uses
  int stat[MAX_BUTTONS]; ///< buttons can be changed by update

  /// Do subclass-specific startup tasks; return success.
  virtual int do_start(const SensorConfig *) { return 1; }

 public:
  Buttons() {}
  virtual ~Buttons() {}
  
  virtual const char *device_name() const = 0;
  virtual Buttons *clone() = 0;

  int start(const SensorConfig *);

  virtual void update() = 0;
  inline int state(int num) {
    if(num>used.num() || num<0) return 0;
    return stat[used[num]];
  }
};

#endif

