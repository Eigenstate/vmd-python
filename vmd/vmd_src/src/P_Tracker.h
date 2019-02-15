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
 *	$RCSfile: P_Tracker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.30 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * An object representing a connection to a machine that controls 3d
 * input devices (optionally with buttons, force-feedback, etc). One
 * connection may control one or many devices, so there needs to be a
 * global list of trackers which the Sensors peruse when first being
 * initialized.
 *
 ***************************************************************************/

#ifndef VMDTRACKER_H
#define VMDTRACKER_H

#include "Matrix4.h"
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

/// An object representing a connection to a machine that controls 3d
/// input devices (optionally with buttons, force-feedback, etc). One
/// connection may control one or many devices, so there needs to be a
/// global list of trackers which the Sensors peruse when first being
/// initialized.
/// This class is named VMDTracker to avoid name conflicts with the
/// UNC tracker library
class VMDTracker { 
 private:
  float scale;         ///< reported position will be scaled by this value
  float offset[3];
  float offset_pos[3];

  Matrix4 left_rot, right_rot;

 protected:
  int dim;             ///< set by subclass; default is three.

  float pos[3];
  Matrix4 *orient;
  Matrix4 rot_orient;

  void moveto(float x, float y, float z) {
    pos[0] = x; pos[1] = y; pos[2] = z;
  }

  /// Do device-specific configuration.  Return success.
  virtual int do_start(const SensorConfig *) { return 1; }

 public:
  /// Constructor: initialize variables, but do not establish any connection.
  VMDTracker();
  virtual ~VMDTracker();

  /// Device name; must be unique to other VMDTracker subclasses
  virtual const char *device_name() const = 0;

  virtual VMDTracker *clone() = 0;

  /// Establish connection to remote device in the start() method, not
  /// in the constructor.
  virtual int start(const SensorConfig *);
  virtual void update() = 0;
  virtual int alive() = 0; // am I alive?
  const float *position() { 
    offset_pos[0] = scale*(offset[0] + pos[0]);
    offset_pos[1] = scale*(offset[1] + pos[1]);
    offset_pos[2] = scale*(offset[2] + pos[2]);
    return offset_pos;
  }
  inline const Matrix4 &orientation() { 
    rot_orient.loadmatrix(left_rot);
    rot_orient.multmatrix(*orient);
    rot_orient.multmatrix(right_rot);
    return rot_orient;    
  }

  void set_offset(const float o[3]) {
    offset[0] = o[0];
    offset[1] = o[1];
    offset[2] = o[2];
  }
  const float *get_offset() const { return offset; }

  void set_scale(float s) { scale = s;    }
  float get_scale() const { return scale; }

  void set_right_rot(const Matrix4 *right) { right_rot = *right; }
  void set_left_rot(const Matrix4 *left)   { left_rot = *left;   }

  /// dimension of positions returned by the device; subclasses set the dim
  /// variable.
  int dimension() const { return dim; }
};

#endif

