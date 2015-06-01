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
 *	$RCSfile: P_Feedback.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.26 $	$Date: 2010/12/16 04:08:30 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * A Feedback is a representation for force-feedback.  Subclass this for
 * specific haptic devices.
 *
 * Note that you have to call sendforce to actually do anything!
 *
 ***************************************************************************/

#ifndef P_FEEDBACK_H
#define P_FEEDBACK_H

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

/// A Feedback is a representation for force-feedback.  Subclass this for
/// specific haptic devices.
class Feedback {
public:
  /// Constructor: initialize variables, but do not establish any connection.
  Feedback() { maxforce = -1.0f; }
  virtual ~Feedback() { };

  /// Device name; must be unique to other Feedback subclasses
  virtual const char *device_name() const = 0;
  virtual Feedback *clone() = 0;

  /// Establish connection to remote device in the start() method, not
  /// in the constructor.
  int start(const SensorConfig *config) {
    set_maxforce(config->getmaxforce());
    return do_start(config);
  }

  virtual void update() = 0;

  /// corresponding functions that should just add force on to whatever
  /// is already present --- since forces and jacobians add linearly,
  /// this works!
  /// Units are all NEWTONS
  virtual void addconstraint(float k, const float *location) = 0;
  virtual void addforcefield(const float *origin, const float *force,
			     const float *jacobian) = 0;
  virtual void addplaneconstraint(float k, const float *point,
				  const float *normal) = 0;

  /// zeros out the constructed force (does not send a message to the
  /// haptic until sendforce is called!)
  virtual void zeroforce() = 0;

  /// stop forces (actually sends a message)
  virtual void forceoff() = 0;

  /// send the force that has been constructed -- we have this so the
  /// force can be built up in several parts, using the additive
  /// functions, and no incorrect forces are ever sent.
  /// The argument is the current position of the corresponding tracker
  /// - it may be used to check that the force doesn't exceed the max.
  virtual void sendforce(const float *initial_pos) = 0;

  /// set/get maxforce
  /// The max force is given by <maxforce> - if it is <0, no maximum is
  /// enforced.
  void set_maxforce(float m) { maxforce = m;    }
  float get_maxforce() const { return maxforce; }

protected:
  float maxforce;

  /// Do device-specific startup configuration.  Return success.
  virtual int do_start(const SensorConfig *) { return 1; }
};

#endif

