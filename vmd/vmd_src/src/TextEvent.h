/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/
/// The following are the set of events that generate Tcl
//// callbacks (or whatever interpreter we happen to be using).
//// Add more if you want to synchronize the text interpreter
//// with some other well-defined event in VMD. 

#ifndef Interp_EVENT_H 
#define Interp_EVENT_H 

#include "Command.h"
#include "JString.h"
#include "TextInterp.h"
#include "utilities.h"
#include "DisplayDevice.h"

#include <ctype.h>

class TextInterp;

/// Command subclass that acts as a base class for interpreter events
class InterpEvent : public Command {
public:
  InterpEvent()
  : Command(INTERP_EVENT) {}
  virtual ~InterpEvent() {}
  virtual void do_callback(TextInterp *) = 0;
};

/// Evaluate the given hotkey combination.
class UserKeyEvent : public InterpEvent {
private:
  char key_desc[128];

  // want to make the key command like the shift_state-less version so
  // force the shift_state change in the character, if set, and record
  // only the other terms of the shift_state by putting on either a prefix
  // such as Alt- or Ctrl- 
  void canonicalize(DisplayDevice::EventCodes ev, char c, int shiftstate) {
    key_desc[0] = 0;
    char tmp[2]; 
    tmp[1] = '\0';
  
    // Spaceball and other AUX devices ignore Keyboard/Mouse meta key state
    if (shiftstate & DisplayDevice::AUX) {
      sprintf(key_desc, "Aux-%c", c);
      return;
    }
 
    if (ev == DisplayDevice::WIN_KBD) { 
      // Guarantee case of incoming characters from display device
      if (shiftstate & DisplayDevice::SHIFT) {
        tmp[0] = toupper(c);
      } else {
        tmp[0] = tolower(c);
      }
  
      if (shiftstate & DisplayDevice::ALT) {
        strcat(key_desc, "Alt-");
      }
      if (shiftstate & DisplayDevice::CONTROL) {
        strcat(key_desc, "Control-");
        // however, if control is pressed then I have to adjust some of
        // the key information to get it in range.  Eg, Control-a == 0x01
        if (0 < tmp[0] && tmp[0] <= 26) { // a-Z
          tmp[0] += int('a') - 1;
          // and correct the case
          if (shiftstate & DisplayDevice::SHIFT) {
            tmp[0] = tolower(tmp[0]);
          }
        } else {
          // fix these by hand:
          switch (tmp[0]) {
          case 0: tmp[0] = '2'; break;
          case 27: tmp[0] = '3'; break;
          case 28: tmp[0] = '4'; break;
          case 29: tmp[0] = '5'; break;
          case 30: tmp[0] = '6'; break;
          case 31: tmp[0] = '7'; break;
          case 127: tmp[0] = '8'; break;
          default: break;
          }
        }
      }
      // and put on the character
      strcat(key_desc, tmp);
    } else {
      switch (ev) {
        case DisplayDevice::WIN_KBD_ESCAPE:   strcpy(key_desc, "Escape"); break;
        case DisplayDevice::WIN_KBD_UP:       strcpy(key_desc, "Up"); break;
        case DisplayDevice::WIN_KBD_DOWN:     strcpy(key_desc, "Down"); break;
        case DisplayDevice::WIN_KBD_LEFT:     strcpy(key_desc, "Left"); break;
        case DisplayDevice::WIN_KBD_RIGHT:    strcpy(key_desc, "Right"); break;
        case DisplayDevice::WIN_KBD_PAGE_UP:  strcpy(key_desc, "Page_Up"); break;
        case DisplayDevice::WIN_KBD_PAGE_DOWN:strcpy(key_desc, "Page_Down"); break;
        case DisplayDevice::WIN_KBD_HOME:     strcpy(key_desc, "Home"); break;
        case DisplayDevice::WIN_KBD_END:      strcpy(key_desc, "End"); break;
        case DisplayDevice::WIN_KBD_INSERT:   strcpy(key_desc, "Insert"); break;
        case DisplayDevice::WIN_KBD_DELETE:   strcpy(key_desc, "Delete"); break;
        case DisplayDevice::WIN_KBD_F1:       strcpy(key_desc, "F1"); break;
        case DisplayDevice::WIN_KBD_F2:       strcpy(key_desc, "F2"); break;
        case DisplayDevice::WIN_KBD_F3:       strcpy(key_desc, "F3"); break;
        case DisplayDevice::WIN_KBD_F4:       strcpy(key_desc, "F4"); break;
        case DisplayDevice::WIN_KBD_F5:       strcpy(key_desc, "F5"); break;
        case DisplayDevice::WIN_KBD_F6:       strcpy(key_desc, "F6"); break;
        case DisplayDevice::WIN_KBD_F7:       strcpy(key_desc, "F7"); break;
        case DisplayDevice::WIN_KBD_F8:       strcpy(key_desc, "F8"); break;
        case DisplayDevice::WIN_KBD_F9:       strcpy(key_desc, "F9"); break;
        case DisplayDevice::WIN_KBD_F10:      strcpy(key_desc, "F10"); break;
        case DisplayDevice::WIN_KBD_F11:      strcpy(key_desc, "F11"); break;
        case DisplayDevice::WIN_KBD_F12:      strcpy(key_desc, "F12"); break;
        default:
          ; // Do nothing for any other event codes
      }
    }
  }

public:
  UserKeyEvent(DisplayDevice::EventCodes ev, char thekey, int theshiftstate) { 
    canonicalize(ev, thekey, theshiftstate);
  }
  virtual void do_callback(TextInterp *interp) {
    interp->userkey_cb(key_desc);
  }
};

/// Evaluate the given string in the Tcl interpreter
class TclEvalEvent : public InterpEvent {
private:
  JString keystring;
public:
  TclEvalEvent(const char *k)
  : keystring(k) {}
  virtual void do_callback(TextInterp *interp) {
    interp->tcl_cb((const char *)keystring);
  }
};

/// This command allows us to evaluate an arbitrary string in the Python
/// interpreter.
class PythonEvalEvent : public InterpEvent {
private:
  JString keystring;
public:
  PythonEvalEvent(const char *k)
  : keystring(k) {}
  virtual void do_callback(TextInterp *interp) {
    interp->python_cb((const char *)keystring);
  }
};


/// Indicates that the frame of a certain molecule has changed
class FrameEvent : public InterpEvent {
private:
  int mol, frame;
public:
  FrameEvent(int m, int f)
  : mol(m), frame(f) {}
  virtual void do_callback(TextInterp *interp) {
    interp->frame_cb(mol, frame);
  }
};


/// Set when a molecular structure is created and destroyed
class InitializeStructureEvent : public InterpEvent {
private:
  int mol, code;
public:
  InitializeStructureEvent(int m, int c)
    : mol(m), code(c) {}
  virtual void do_callback(TextInterp *interp) {
    interp->initialize_structure_cb(mol, code);
  }
};

/// Set when a molecule is created and destroyed or modified
/// Eventually add codes A/F/D changes, etc
class MoleculeEvent : public InterpEvent {
private:
  int mol, code;
public:
  enum MolEvent { MOL_DELETE=0, MOL_NEW=1, MOL_RENAME=2, MOL_REGEN=3,
                  MOL_TOP=4 };
  MoleculeEvent(int m, MolEvent c)
    : mol(m), code((int) c) {}
  virtual void do_callback(TextInterp *interp) {
    interp->molecule_changed_cb(mol, code);
  }
};


/// Tell when the mouse mode changes.  I hope no one is using this...
class MouseModeEvent : public InterpEvent {
  private:
    JString mode;
    int submode;

  public:
    MouseModeEvent(const char *m, int sm) : mode(m), submode(sm) {}
    virtual void do_callback(TextInterp *interp) {
      interp->mousemode_cb((const char *)mode, submode);
    }
};


/// Sets the most recent mouse position and button state
class MousePositionEvent : public InterpEvent {
private:
  float x, y;
  int buttondown;

public:
  MousePositionEvent(float xv, float yv, int bdown) : x(xv), y(yv), buttondown(bdown) {}
  virtual void do_callback(TextInterp *interp) {
    interp->mouse_pos_cb(x, y, buttondown);
  }
};


/// Sets the most recent Mobile translation, rotation, and button state
class MobileEvent : public InterpEvent {
private:
  float tx;
  float ty;
  float tz;
  float rx;
  float ry; 
  float rz;
  int buttondown;

public:
  MobileEvent(float utx, float uty, float utz,
              float urx, float ury, float urz,
              int bdown) : tx(utx), ty(uty), tz(utz),
                              rx(urx), ry(ury), rz(urz), buttondown(bdown) {}
  virtual void do_callback(TextInterp *interp) {
    interp->mobile_cb(tx, ty, tz, rx, ry, rz, buttondown);
  }
};

/// Indicates that the mobile device state has changed
class MobileStateChangedEvent : public InterpEvent {
public:
  virtual void do_callback(TextInterp *interp) {
    interp->mobile_state_changed_cb();
  }
};

/// Indicates that a mobile device has a specific command to run
class MobileDeviceCommandEvent : public InterpEvent {
private:
  JString str;
public:
  MobileDeviceCommandEvent(const char *event)
    : str(event) {}
  virtual void do_callback(TextInterp *interp) {
    interp->mobile_device_command_cb((const char *)str);
  }
};


/// Sets the most recent Spaceball translation, rotation, and button state
class SpaceballEvent : public InterpEvent {
private:
  float tx;
  float ty;
  float tz;
  float rx;
  float ry; 
  float rz;
  int buttondown;

public:
  SpaceballEvent(float utx, float uty, float utz,
                 float urx, float ury, float urz,
                 int bdown) : tx(utx), ty(uty), tz(utz),
                              rx(urx), ry(ury), rz(urz), buttondown(bdown) {}
  virtual void do_callback(TextInterp *interp) {
    interp->spaceball_cb(tx, ty, tz, rx, ry, rz, buttondown);
  }
};


/// Sets the value of the last created Geometry
class PickValueEvent : public InterpEvent {
private:
  double val;
public:
  PickValueEvent(double v) : val(v) {}
  virtual void do_callback(TextInterp *interp) {
    interp->pick_value_cb((float) val);
  }
};


/// Sets the molid and atomid of the last picked atom
class PickAtomEvent : public InterpEvent {
private:
  int mol, atom, key_shift_state;
  bool is_pick;
public:
  PickAtomEvent(int m, int a , int ss, bool ispick=false)
    : mol(m), atom(a), key_shift_state(ss), is_pick(ispick) {}
  virtual void do_callback(TextInterp *interp) {
    interp->pick_atom_cb(mol, atom, key_shift_state, is_pick);
  }
};

/// Sets the molid, tag, and button of the last picked user graphics
class PickGraphicsEvent : public InterpEvent {
private:
  int mol, tag, btn, key_shift_state;
public:
  PickGraphicsEvent(int m, int t , int b, int ss)
    : mol(m), tag(t), btn(b), key_shift_state(ss) {}
  virtual void do_callback(TextInterp *interp) {
    interp->pick_graphics_cb(mol, tag, btn, key_shift_state);
  }
};

/// Sets the molid, atomid, and client of the last callback-picked atom
class PickAtomCallbackEvent : public InterpEvent {
private:
  int mol, atom;
  JString client;
public:
  PickAtomCallbackEvent(int m, int a, const char *c)
    : mol(m), atom(a), client(c) {}
  virtual void do_callback(TextInterp *interp) {
    interp->pick_atom_callback_cb(mol, atom, (const char *)client);
  }
};


/// Indicates when all the frames of a trajectory have been read
class TrajectoryReadEvent : public InterpEvent {
private:
  int id;
  char *name;
public:
  TrajectoryReadEvent(int m, const char *n) {
    id = m;
    name = stringdup(n);
  }
  ~TrajectoryReadEvent() { delete [] name; }
  virtual void do_callback(TextInterp *interp) {
    interp->trajectory_cb(id, name);
  }
};


/// Indicates that a new timestep has been received over a remote connection
class TimestepEvent : public InterpEvent {
private:
  int id, frame;
public:
  TimestepEvent(int m, int f) : id(m), frame(f) {}
  virtual void do_callback(TextInterp *interp) {
    interp->timestep_cb(id, frame);
  }
};


/// Indicates that help is desired on the given topic
class HelpEvent : public InterpEvent {
private:
  JString str;
public:
  HelpEvent(const char *topic)
    : str(topic) {}
  virtual void do_callback(TextInterp *interp) {
    interp->help_cb((const char *)str);
  }
};


/// Gives the interpreter the text of a command being written to a logfile.
class LogfileEvent : public InterpEvent {
private:
  JString str;
public:
  LogfileEvent(const char *event)
    : str(event) {}
  virtual void do_callback(TextInterp *interp) {
    interp->logfile_cb((const char *)str);
  }
};


/// XXX These are undocumented...
/// The user has requested the given labels to be graphed.
class GraphLabelEvent : public InterpEvent {
  private:
    char *type;  // Atoms, Bonds, etc.
    int *ids;
    int nlabels;
  public:
    GraphLabelEvent(const char *labeltype, const int *labels, int n) {
      type = stringdup(labeltype); 
      nlabels = n;
      if (n > 0) {
        ids = new int[n];
        memcpy(ids, labels, n*sizeof(int));
      } else {
        ids = NULL;
      }
    }
    virtual void do_callback(TextInterp *interp) {
      interp->graph_label_cb(type, ids, nlabels);
    }
    ~GraphLabelEvent() {
      delete [] type;
      delete [] ids;
    }
};


/// List the atoms involved in the last geometry object created
class PickSelectionEvent : public InterpEvent {
  private:
    int num;
    int *atoms;
  public:
    PickSelectionEvent(int n, const int *a) : num(n) {
      if (n > 0) {
        atoms = new int[n];
        memcpy(atoms, a, n*sizeof(int));
      } else {
        atoms = NULL;
      }
    }
    ~PickSelectionEvent() { delete [] atoms; }
    virtual void do_callback(TextInterp *interp) {
      interp->pick_selection_cb(num, atoms);
    }
};

#endif
