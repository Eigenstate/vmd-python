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
 *	$RCSfile: P_UIVR.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.72 $	$Date: 2010/12/16 04:08:32 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's simplified Tracker code -- pgrayson@ks.uiuc.edu
 *
 * The UIVR is the Virtual Reality User Interface.  It should
 * coordinate multiple tools, taking manipulation commands from them
 * and giving them back various messages and flags.  UIVR is the thing
 * which has to know about the different kinds of tools.
 *
 ***************************************************************************/

#include "NameList.h"
#include "Matrix4.h"
#include "UIObject.h"
#include <stdio.h>

class JString;
class AtomSel;
class Tool;
class VMDTracker;
class Feedback;
class Buttons;
class SensorConfig;

/// UIObject subclass implementing a Virtual Reality user interface.
/// I coordinates multiple tools, taking manipulation commands from them
/// and giving them back various messages and flags.  UIVR is the thing
/// which has to know about the different kinds of tools.
class UIVR : public UIObject {
public:
  UIVR(VMDApp *); 
  ~UIVR();

  ///  Add a tool that finds it sensor from a USL
  int add_tool_with_USL(const char *type, int argc, const char **USL);

  /// Change the type of an existing tool
  int change_type(int toolnum, const char *type);

  /// remove the given tool; return success.
  int remove_tool(int i);

  int check_event();
  Tool *gettool(int i);
  inline int num_tools() { return tools.num(); }

  const char *tool_name(int i) {
    if(i<tool_types.num()) return tool_types.name(i);
    else return "<error>";
  }

  int tool_type(const char *nm) {
    return tool_types.typecode(nm);
  }

  int num_tool_types() { return tool_types.num(); }

  virtual int act_on_command(int, Command *);

  /// Return list of device names that have been read from the .vmdsensors file.
  static ResizeArray<JString *> *get_device_names();

  /// Return list of device names for each class of device, as obtained from
  /// the .vmdsensors file.  Only devices for which a corresponding device
  /// class exists will be returned in the list.  Delete the list and its 
  /// elements when finished.
  ResizeArray<JString *> *get_tracker_names();
  ResizeArray<JString *> *get_feedback_names();
  ResizeArray<JString *> *get_button_names();
  
  /// Set the tracker, feedback, or buttons of the given tool.  Pass NULL
  /// as the device to simply remove the device without replacing it.
  /// Return success.
  int set_tracker(int toolnum, const char *device);
  int set_feedback(int toolnum, const char *device);
  int set_buttons(int toolnum, const char *device);
  
  int set_position_scale(int toolnum, float newval);
  int set_force_scale(int toolnum, float newval);
  int set_spring_scale(int toolnum, float newval);

private:
  ResizeArray<Tool *> tools;
  NameList<int> tool_types;
  
  /// non-repeating serial number for tools.
  int tool_serialno;

  /// Create a tool of the given type and return it
  Tool *create_tool(const char *type);

  /// Add a tool of a given type (without setting up the sensor.) and
  /// return the index of the tool added.
  int add_tool(const char *type);

  /// Dynamics lists of devices
  NameList<VMDTracker *> trackerList;
  NameList<Feedback *> feedbackList;
  NameList<Buttons *> buttonList;

  /// Populate the *List lists with devices that we know about.  Eventually
  /// it should be possible to use plugins to accomplish this; for now we
  /// limit ourselves to statically-linked classes.
  void update_device_lists();

  /// Add the specified device to a tool
  int add_device_to_tool(const char *devicename, Tool *tool);
};

