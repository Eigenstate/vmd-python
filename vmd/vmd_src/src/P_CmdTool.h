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
 *	$RCSfile: P_CmdTool.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.37 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 ***************************************************************************/

#include "Command.h"
#include "Matrix4.h"

/// create a new tool, attaching a sensor
class CmdToolCreate : public Command {
 public:
  CmdToolCreate(const char *thetype, int theargc, const char **theUSL);
  ~CmdToolCreate();
  char **USL;
  int argc;
  char *type;
 protected:
  virtual void create_text();
};


/// change the type of tool, keeping the sensor
class CmdToolChange : public Command {
 public:
  CmdToolChange(const char *thetype, int thenum);
  ~CmdToolChange();
  int num;
  char *type;
 protected:
  virtual void create_text();
};


/// delete a tool
class CmdToolDelete : public Command {
 public:
  CmdToolDelete(int thenum);
  int num;
 protected:
  virtual void create_text();
};


/// change the position scaling factor for a tool
class CmdToolScale : public Command {
 public:
  CmdToolScale(float thescale, int thenum);
  int num;
  float scale;
 protected:
  virtual void create_text();
};


/// change the force scaling factor for a tool
class CmdToolScaleForce : public Command {
 public:
  CmdToolScaleForce(float thescale, int thenum);
  int num;
  float scale;
 protected:
  virtual void create_text();
};


/// change the force feedback spring constant for a tool
class CmdToolScaleSpring : public Command {
 public:
  CmdToolScaleSpring(float thescale, int thenum);
  int num;
  float scale;
 protected:
  virtual void create_text();
};


/// change the position offset for a tool
class CmdToolOffset : public Command {
 public:
  CmdToolOffset(float *theoffset, int thenum);
  int num;
  float offset[3];
 protected:
  virtual void create_text();
};


/// attach a tool to a particular representation (instead of picking)
class CmdToolRep : public Command {
public:
  CmdToolRep(int thetoolnum, int themolid, int therepnum);
  int toolnum;
  int molid;
  int repnum;  
protected:
  virtual void create_text();
};


/// add a device to a tool
class CmdToolAddDevice : public Command {
 public:
  CmdToolAddDevice(const char *thename, int thenum);
  char *name;
  int num;
 protected:
  virtual void create_text();
  virtual ~CmdToolAddDevice();
};


/// delete a device from a tool
class CmdToolDeleteDevice : public Command {
 public:
  CmdToolDeleteDevice(const char *thename, int thenum);
  char *name;
  int num;
 protected:
  virtual void create_text();
  virtual ~CmdToolDeleteDevice();
};


/// register a callback with a tool 
class CmdToolCallback : public Command {
public:
  // turn it on (true) or off (false)
  // thenum is the tool number
  CmdToolCallback(int the_on);
  int on;

 protected:
  void create_text(void);
};
