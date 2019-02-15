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
 *	$RCSfile: P_Tool.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.67 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************/
#ifndef P_TOOL_H
#define P_TOOL_H

#include "UIObject.h"
#include "JString.h"
#include "Matrix4.h"

class Displayable;
class VMDTracker;
class Buttons;
class Feedback;
class SensorConfig;

#define TARGET_TUG  0
#define TARGET_GRAB 1

/// A Tool represents a virtual device used to manipulate objects on
/// the screen. Tools use input from VMDTracker, Feedback, and Button devices
/// and draw a pointer on the screen.  Tools can give up their
/// devices to another tool to effectively allow a tool to change its
/// type on the fly.
class Tool : public UIObject {
public:
  Tool(int id, VMDApp *, Displayable *aParentDisplayable);
  virtual ~Tool();

  /// return unqiue id
  int id() const { return my_id; }

  int steal_sensor(Tool *from); ///< steal devices from another tool

  /** 
  Add a device.  Also starts the device.  Add a NULL device to remove 
  the item.  Return success, defined as either successful removal or
  starting of the device.
  */
  int add_tracker(VMDTracker *, const SensorConfig *);
  int add_feedback(Feedback *, const SensorConfig *);
  int add_buttons(Buttons *, const SensorConfig *);

  /// Remove the device with the given name. For backwards compatibility only.
  int remove_device(const char *device);

  /// Get the names of the devices used by this tool.
  void getdevices(char **ret);
  const char *get_tracker() const { return tracker ? (const char *)trackerDev : NULL; }
  const char *get_feedback() const { return feedback ? (const char *)feedbackDev : NULL; }
  const char *get_buttons() const { return buttons ? (const char *)buttonDev : NULL; }

  /// Get/set the coordinate scaling used by this tool's Tracker.
  float getscale();
  void setscale(float scale);

  /// The spring scaling controls the stiffness of the force feedback device.
  /// Its units should be real N/m
  float getspringscale() const { return springscale; }
  void setspringscale(float s) { springscale = s;    }

  /// The force scaling scales the _exported force, i.e. the force which is 
  /// sent to UIVR and/or an external simulation.
  /// It is a spring constant, in \f$ \hbox{kcal}/\hbox{mol}/\hbox{\AA}^2 \f$.
  float getforcescale() { return forcescale; }
  void setforcescale(float f) { forcescale = f; }

  /// Get/set the position offset used by this tool's tracker.
  const float *getoffset();
  void setoffset(float *offset);

  /// return the position of this tool.
  virtual const float *position() const;

  /// return the orientation of this tool.
  virtual const Matrix4 *orientation();

  /// True iff the tool is grabbing something.
  virtual int isgrabbing();

  /// True iff there is a valid sensor for this tool.
  int alive() const { return amalive; }

  /// Invalidate sensor/tool.
  void kill() { amalive = 0; }

  /// return the name of this tool.  Must be unique to all Tool subclasses!
  virtual const char *type_name() const = 0; 

  virtual int check_event();

protected:
  float forcescale;       ///< scales force sent to the atoms
  float springscale;      ///< scales force sent to the feedback device

private:
  int amalive;            ///< tool is alive

  VMDTracker *tracker;
  Feedback *feedback;
  Buttons *buttons;
  JString trackerDev, feedbackDev, buttonDev;

  int lost_sensor;        ///< Have I given my sensor away?

  void update();
  void clear_devices();

  float pos[3];           ///< coordinates obtained from tracker
  Matrix4 orient;         ///< orientation of tracker
  const int my_id;        ///< my unique id

  // Picking state for this tool.
  int targeted_molecule;  ///< picked molecule
  int targeted_atom;      ///< picked atom
  char *targeted_rep;     ///< unique name of targeted rep

  Matrix4 last_rot;
  float last_pos[3];
  float sel_total_mass;

public:
  // Turning grabbing on and off is still done from within P_UIVR, so these
  // need to be public for now.
  int grabs;
  void dograb();
  void ungrab();

  /// Use the given rep as the targeted atoms, overriding picked atoms.
  /// Return success.
  int assign_rep(int mol, int rep);

  /// Get the molid of the currently selected rep; -1 if none.
  int get_rep_molid() const { return targeted_molecule; }
  
  /// Get the name of the currently selected rep; NULL if none.
  const char *get_rep_name() const { return targeted_rep; }

  /// Clear the selection and go back to picking atoms by hand.
  void clear_rep() { 
    delete [] targeted_rep;
    targeted_molecule = -1;
    targeted_rep = NULL;
    sel_total_mass = 0;
    let_go();
  }

protected:
  /// The visual representation of this tool.  It's not private because for
  /// some reason the TugTool needs access to some of its members.
  Displayable *dtool;
  
  /// Subclasses should override this method to perform tool-specific 
  /// actions every display loop.
  virtual void do_event() {}

  // These are all the methods for querying and manipulating the picking state.

  /// See if we're currently picking anything, and if so, update the pick
  /// variables.  Return true if anything was picked.
  int target(int target_type, float *mpos, int just_checking);

  /// Cease picking.
  void let_go() { if (!targeted_rep) targeted_molecule = -1; }

  /// Test for picking.
  int is_targeted() const { return targeted_molecule != -1; }

  /// See what was picked last without acquiring a new target.
  int get_targeted_atom(int *molret, int *atomret) const;

  /// Trigger callbacks based on current position.
  void tool_location_update();

  /// Returns the scale factor of the currently picked molecule.
  float getTargetScale();

  /// Apply the given force to the current target.
  void tug(const float *);

  /// Dimension of the tracker device in use.
  int dimension();

  /// Set force field parameters
  void setplaneconstraint(float k, const float *point, const float *normal);
  void addplaneconstraint(float k, const float *point, const float *normal);
  void setconstraint(float k, const float *point); 
  void setforcefield(const float *origin, const float *force, 
		     const float *jacobian);

  /// Deliver specified force to the feedback device.
  /// XXX Maybe this should be taken care of automatically in the update loop
  void sendforce();

  /// Stop applying force.  
  void forceoff();

  /// XXX Should be private variable; gets set to isgrabbing() by RotateTool
  int wasgrabbing;
};

#endif

