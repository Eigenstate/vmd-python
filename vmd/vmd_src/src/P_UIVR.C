

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
 *	$RCSfile: P_UIVR.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.138 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 ***************************************************************************/

#ifdef VMDVRPN
#include "quat.h"
#endif

#include "TextEvent.h"
#include "P_UIVR.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "P_CmdTool.h"
#include "VMDApp.h"
#include "CommandQueue.h"
#include "Scene.h"
#include "PickList.h"
#include "P_Tool.h"
#include "P_RotateTool.h"
#include "P_JoystickTool.h"
#include "P_GrabTool.h"
#include "P_PrintTool.h"
#include "P_TugTool.h"
#include "SpringTool.h"
#include "P_PinchTool.h"
#include "MaterialList.h"
#include "P_SensorConfig.h"
#include "P_Tracker.h"
#include "P_Feedback.h"
#include "P_Buttons.h"
#include "CmdLabel.h"
#include "Inform.h"

#ifdef VMDVRPN
#include "vrpn_Button.h"
#include "vrpn_Tracker.h"
#include "vrpn_ForceDevice.h"
#endif

#include "P_Tracker.h"
#include "P_Buttons.h"
#include "P_Feedback.h"
  
#ifdef VMDCAVE
#include "P_CaveButtons.h"
#include "P_CaveTracker.h"
#endif

#ifdef VMDFREEVR
#include "P_FreeVRButtons.h"
#include "P_FreeVRTracker.h"
#endif

#ifdef VMDVRPN
#include "P_VRPNTracker.h"
#include "P_VRPNButtons.h"
#include "P_VRPNFeedback.h"
#endif

#ifdef WINGMAN
#include "wingforce.h"
#include "P_JoystickButtons.h"
#endif

#include "MobileButtons.h"
#include "MobileTracker.h"

#include "SpaceballButtons.h"
#include "SpaceballTracker.h"


UIVR::~UIVR() {
  int i;
  for (i=0;i<tools.num();i++)
    delete tools[i];
  for (i=0; i<trackerList.num(); i++) 
    delete trackerList.data(i);
  for (i=0; i<feedbackList.num(); i++) 
    delete feedbackList.data(i);
  for (i=0; i<buttonList.num(); i++) 
    delete buttonList.data(i);

}

UIVR::UIVR(VMDApp *vmdapp) : UIObject(vmdapp) {
  command_wanted(Command::TOOL_OFFSET);
  command_wanted(Command::TOOL_REP);
  command_wanted(Command::TOOL_ADD_DEVICE);
  command_wanted(Command::TOOL_DELETE_DEVICE);
  command_wanted(Command::TOOL_CALLBACK);

  tool_serialno = 0;

  tool_types.add_name("grab",     0); // grabbing/rotating molecule
  tool_types.add_name("rotate",   1); // rotating, constrained to sphere
  tool_types.add_name("joystick", 2); // relative positioning 
  tool_types.add_name("tug",      3); // apply IMD forces to molecule
  tool_types.add_name("pinch",    4); // apply IMD forces, but only tracker dir
  tool_types.add_name("spring",   5); // attach IMD springs
  tool_types.add_name("print",    6); // print tracker data

  update_device_lists();
  reset();
}

Tool *UIVR::create_tool(const char *type) {
  Displayable *parent = &(app->scene->root);
  Tool *newtool = NULL;
  if(!strupncmp(type, "grab", 10)) 
    newtool = new GrabTool(tool_serialno++, app, parent);
  else if(!strupncmp(type, "joystick", 10)) 
    newtool = new JoystickTool(tool_serialno++, app, parent);
  else if(!strupncmp(type, "tug", 10)) 
    newtool = new TugTool(tool_serialno++,app, parent);
  else if(!strupncmp(type, "pinch", 10)) 
    newtool = new PinchTool(tool_serialno++,app, parent);
  else if(!strupncmp(type, "spring", 10)) 
    newtool = new SpringTool(tool_serialno++,app, parent);
  else if(!strupncmp(type, "print", 10)) 
    newtool = new PrintTool(tool_serialno++, app, parent);
  
#ifdef VMDVRPN
  // XXX why is only this tool protected by the ifdef??
  else if(!strupncmp(type, "rotate", 10)) 
    newtool = new RotateTool(tool_serialno++,app, parent);
#endif
  else {
    msgErr << "Unrecognized tool type " << type << sendmsg;
    msgErr << "possiblities are:";
    for(int i=0;i<num_tool_types();i++) msgErr << " " << tool_types.name(i);
    msgErr << sendmsg;
    return NULL;
  }
  newtool->On();
  newtool->grabs = 0;
  return newtool;
}

int UIVR::add_tool(const char *type) {
  Tool *newtool = create_tool(type);
  if (!newtool) return -1;
  tools.append(newtool);
  return tools.num()-1;
}

int UIVR::add_tool_with_USL(const char *type, int argc, const char **USL) {
  int num = add_tool(type);
  if(num==-1) return FALSE;
  Tool *tool = tools[num];
  for (int i=0; i<argc; i++) 
    add_device_to_tool(USL[i], tool);
  return TRUE;
}

int UIVR::change_type(int toolnum, const char *type) {
  if (toolnum < 0 || toolnum >= tools.num()) return FALSE;
  Tool *newtool = create_tool(type);
  if (!newtool) return FALSE;
  Tool *oldtool = tools[toolnum];
  newtool->steal_sensor(oldtool);
  delete oldtool;
  tools[toolnum] = newtool;
  return TRUE;
}

int UIVR::remove_tool(int i) {
  if(i<0 || i >= tools.num()) return FALSE;
  Tool *deadtool = tools[i];
  delete deadtool; 
  tools.remove(i);
  return TRUE;
}

int UIVR::check_event() {
  // XXX this is not so good - tug
  // should probably be in the Tool
  // class...

  int i;
  for(i=0;i<tools.num();i++) {
    /* prune the dead tools */
    if(!tools[i]->alive()) {
      // msgErr << "UIVR: Dead tool found." << sendmsg;
      remove_tool(i);
      i--;
      continue;
    }

    if(tools[i]->orientation()==NULL) 
      continue;

    /* possibly grab on to some stuff */
    if(tools[i]->isgrabbing()) {
      tools[i]->dograb();
      tools[i]->grabs = 1;
    } else {
      if(tools[i]->grabs) tools[i]->ungrab();  //TJF Changed
      tools[i]->grabs = 0; // TJF Changed
    }
  }


  return TRUE;
}

Tool *UIVR::gettool(int i) {
  if(i<0) return NULL;
  if(i>=tools.num()) return NULL;
  return tools[i];
}

int UIVR::set_position_scale(int toolnum, float newval) {
  if (newval >= 0) {
    Tool *tool = gettool(toolnum);
    if (tool) {
      tool->setscale(newval);
      return TRUE;
    }
  }
  return FALSE;
}
int UIVR::set_force_scale(int toolnum, float newval) {
  if (newval >= 0) {
    Tool *tool = gettool(toolnum);
    if (tool) {
      tool->setforcescale(newval);
      return TRUE;
    }
  }
  return FALSE;
}
int UIVR::set_spring_scale(int toolnum, float newval) {
  if (newval >= 0) {
    Tool *tool = gettool(toolnum);
    if (tool) {
      tool->setspringscale(newval);
      return TRUE;
    }
  }
  return FALSE;
}

int UIVR::act_on_command(int type, Command *c) {
  switch(type) {
  default:  return FALSE; 
    case Command::TOOL_OFFSET:
      {
      Tool *tool = gettool(((CmdToolScaleSpring *)c)->num);
      if (!tool) return FALSE;
      tool->setoffset(((CmdToolOffset *)c)->offset);
      }
      break;
    case Command::TOOL_ADD_DEVICE:
      {
      CmdToolAddDevice *cmd = (CmdToolAddDevice *)c; 
      Tool *tool = gettool(cmd->num);
      return add_device_to_tool(cmd->name, tool);
      }
      break;
    case Command::TOOL_DELETE_DEVICE:
      {
      CmdToolAddDevice *cmd = (CmdToolAddDevice *)c; 
      Tool *tool = gettool(cmd->num);
      if (!tool) return FALSE;
      if (!cmd->name) return FALSE;
      tool->remove_device(cmd->name);
      }
      break;
    case Command::TOOL_CALLBACK:
      {
      CmdToolCallback *cmd = (CmdToolCallback *)c;
      set_callbacks(cmd->on);
      break;
      }
    case Command::TOOL_REP:
      {
      CmdToolRep *cmd = (CmdToolRep *)c;
      Tool *tool = gettool(cmd->toolnum);
      if (!tool) return FALSE;
      if (cmd->molid < 0 || cmd->repnum < 0) {
        tool->clear_rep();
      } else {
        tool->assign_rep(cmd->molid, cmd->repnum);
      }
      }
      break;
  }
  return TRUE;
}


int UIVR::add_device_to_tool(const char *device, Tool *tool) {

  if (!tool) return FALSE;
  if (!device) return FALSE;

  // Get configuration information for this device from SensorConfig.
  // Cannot add device without config information.
  SensorConfig config(device);
  if (!config.getUSL()[0]) return FALSE;

  // search for the device in the three device lists; we chould
  // deprecate this command and make separate commands for adding
  // trackers, feedback, and buttons.  Another possibility is
  // to define a superclass for all three, but then we would have
  // to downcast at some point.
  const char *devtype = config.gettype();
  if (trackerList.typecode(devtype) >= 0) {
    tool->add_tracker(trackerList.data(devtype)->clone(), &config);
  } else if (feedbackList.typecode(devtype) >= 0) {
    tool->add_feedback(feedbackList.data(devtype)->clone(), &config);
  } else if (buttonList.typecode(devtype) >= 0) {
    tool->add_buttons(buttonList.data(devtype)->clone(), &config);
  } else {
    msgErr << "Device '" << device << "' of type '" << devtype << "' not available." << sendmsg;
  }
  return TRUE;
}
  
int UIVR::set_tracker(int toolnum, const char *device) {
  Tool *tool = gettool(toolnum);
  if (!tool) return FALSE;
  if (device == NULL) {
    tool->add_tracker(NULL, NULL);
    return TRUE;
  }
  return add_device_to_tool(device, tool);
}
int UIVR::set_feedback(int toolnum, const char *device) {
  Tool *tool = gettool(toolnum);
  if (!tool) return FALSE;
  if (device == NULL) {
    tool->add_feedback(NULL, NULL);
    return TRUE;
  }
  return add_device_to_tool(device, tool);
}
int UIVR::set_buttons(int toolnum, const char *device) {
  Tool *tool = gettool(toolnum);
  if (!tool) return FALSE;
  if (device == NULL) {
    tool->add_buttons(NULL, NULL);
    return TRUE;
  }
  return add_device_to_tool(device, tool);
}

ResizeArray<JString *> *UIVR::get_device_names() {
  return SensorConfig::getnames();
}

template<class T>
ResizeArray<JString *> *generic_get_names(const T &devList) {

  ResizeArray<JString *> *names = SensorConfig::getnames();
  ResizeArray<JString *> *devnames = new ResizeArray<JString *>;
  for (int i=0; i<names->num(); i++) {
    JString *jstr = (*names)[i];
    SensorConfig config(*jstr);
    const char *type = config.gettype();
    if (devList.typecode(type) >= 0)
      devnames->append(jstr); 
    else
      delete jstr;
  } 
  delete names;
  return devnames;
}
  
ResizeArray<JString *> *UIVR::get_tracker_names() {
  return generic_get_names(trackerList);
}
ResizeArray<JString *> *UIVR::get_feedback_names() {
  return generic_get_names(feedbackList);
}
ResizeArray<JString *> *UIVR::get_button_names() {
  return generic_get_names(buttonList);
}

void UIVR::update_device_lists() {
  VMDTracker *tracker = NULL;
  Buttons *buttons = NULL;

  tracker = new SpaceballTracker(app);
  trackerList.add_name(tracker->device_name(), tracker);
  buttons = new SpaceballButtons(app);
  buttonList.add_name(buttons->device_name(), buttons);

  tracker = new MobileTracker(app);
  trackerList.add_name(tracker->device_name(), tracker);
  buttons = new MobileButtons(app);
  buttonList.add_name(buttons->device_name(), buttons);

#ifdef VMDVRPN
  Feedback *feedback = NULL;
  tracker = new VRPNTracker;
  trackerList.add_name(tracker->device_name(), tracker);
  feedback = new VRPNFeedback;
  feedbackList.add_name(feedback->device_name(), feedback);
  buttons = new VRPNButtons;
  buttonList.add_name(buttons->device_name(), buttons);
#endif

#ifdef VMDCAVE
  tracker = new CaveTracker;
  trackerList.add_name(tracker->device_name(), tracker);
  buttons = new CaveButtons;
  buttonList.add_name(buttons->device_name(), buttons);
#endif

#ifdef VMDFREEVR
  tracker = new FreeVRTracker(app);
  trackerList.add_name(tracker->device_name(), tracker);
  buttons = new FreeVRButtons(app);
  buttonList.add_name(buttons->device_name(), buttons);
#endif

#ifdef WINGMAN
  buttons = new JoystickButtons;
  buttonList.add_name(buttons->device_name(), buttons);
#endif
}
