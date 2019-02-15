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
 *	$RCSfile: P_SensorConfig.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.28 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * A SensorConfig is the object that a Sensor gets its configuration
 * from.  When it is loading a tracker's USL, it should make a
 * SensorConfig object and query it for the different sorts of offsets
 * and rotations that the particular tracker will use.
 *
 * SensorConfigs work by loading configuration files and parsing them
 * for the needed information.  The format for a file is as follows:
 * Each line is either blank, a comment, a device line, or a parameter
 * for the last device that was specified.
 *
 * A comment is any line whose first non-whitespace character is a "#"
 *
 * A device line has the form "device <name> <USL>".
 * Parameters look like TCL commands, here are examples of all types:
 *   scale .4
 *   forcescale 2
 *   offset 1 -2 0
 *   rotate right 0 0 -1 1 0 0 0 1 0
 *   rotate left 0 0 1 -1 0 0 0 -1 0
 ***************************************************************************/
#ifndef SENSOR_CONFIG_H__
#define SENSOR_CONFIG_H__

#include <stdio.h>
#include "JString.h"
#include "ResizeArray.h"
#include "Matrix4.h"

/// Provides a Sensor with configuration information by parsing
/// one or more configuration files.
class SensorConfig {
public:
  /// Constructor loads the .vmdsensors file.
  SensorConfig(const char *thedevice);
  ~SensorConfig();

  /// Seems to be used to tell if the specified device was found or not.
  const char *getUSL() const;

  /// Name of the device that will be returned
  const char *getdevice() const;

  /// find the list of supported device names --- delete the list when done!
  static ResizeArray<JString *> *getnames();

  /// Accessor routines for device config parameters
  float getmaxforce() const;
  float getscale() const;
  const float *getoffset() const;
  const Matrix4 *getright_rot() const;
  const Matrix4 *getleft_rot() const;
  const char *gettype() const;
  const char *getplace() const;
  const char *getname() const;
  const char *getnums() const;
  const ResizeArray<int> *getsensors() const;

  /// Check that the config specifies only one sensor for the device;
  /// prints a nice warning message if it doesn't.
  int have_one_sensor() const;
  
  /// Check that the place is local; print a nice warning if it isn't.
  int require_local() const;

  /// Check that the place and name are valid for a CAVE
  int require_cave_name() const;

  /// Check that the place and name are valid for FreeVR
  int require_freevr_name() const;

  /// Create a vrpn address for the current name and place; store in buf.
  void make_vrpn_address(char *buf) const;
 
private:
  /// used internally for parsing the config files
  static void ScanSensorFiles (int behavior, SensorConfig *sensor, void* params);
  void parseconfigfordevice(FILE *f, void *);
  static void parseconfigfornames(FILE *f, void *ret_void);
        
  float getfloat(const char *from, float defalt);
  int needargs(int argc,int need);
  int parseUSL();          ///< finds components, returns success
  void read_sensor_nums(); ///< break up the list into sensors
  
  int line;                ///< current line of parsing

  // the USL and its components
  JString USL;
  char device[50];
  char type[21];
  char place[101];
  char name[101];
  char nums[101];
  ResizeArray<int> sensors;

  float scale;
  float maxforce;
  float offset[3];
  Matrix4 right_rot;
  Matrix4 left_rot;
};

#endif // SENSOR_CONFIG_H__
