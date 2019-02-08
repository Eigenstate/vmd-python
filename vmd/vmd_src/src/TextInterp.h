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
 *      $RCSfile: TextInterp.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.33 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Text interpreter base class
 ***************************************************************************/
#ifndef TEXT_INTERP_H
#define TEXT_INTERP_H

/// Base class for all text interpreters
class TextInterp {
protected:
  TextInterp() {}

public:
  virtual ~TextInterp() {}

  /// Tell the interpreter to read its init file; presumably it knows what to do
  virtual void doInit() {}

  /// Let the interpreter have control for while
  virtual void doEvent() {}

  /// Update Tk, if possible, from within this interpreter
  /// return 1 if able to do the Tk update, 0 if unsuccessful.
  virtual int doTkUpdate() { return 0; }

  /// Pass the given string to the interpreter.  Return success.
  virtual int evalString(const char *) { return 1; }

  /// Process the given file.  Return success.
  virtual int evalFile(const char *) { return 1; }

  //
  // methods for setting global data in the text interpreter namespace
  //

  /// First argument: variable name.  Second argument: string value
  virtual void setString(const char *, const char *) {};

  //
  // callbacks for various VMD events
  //
  virtual void display_update_cb() {}
  virtual void frame_cb(int molid, int frame) {}
  virtual void help_cb(const char *topic) {}
  virtual void initialize_structure_cb(int molid, int create_or_destroy) {}
  virtual void molecule_changed_cb(int molid, int code) {}
  virtual void logfile_cb(const char *cmd) {}
  virtual void mousemode_cb(const char *mode, int submode) {}
  virtual void mouse_pos_cb(float x, float y, int buttondown) {}
  virtual void mobile_cb(float tx, float ty, float tz,
                         float rx, float ry, float rz, int buttondown) {}
  virtual void mobile_state_changed_cb() {}
  virtual void mobile_device_command_cb(const char *str) {}
  virtual void spaceball_cb(float tx, float ty, float tz,
                            float rx, float ry, float rz, int buttondown) {}
  virtual void pick_atom_cb(int molid, int atomid, int shift_state, bool is_pick) {}
  virtual void pick_atom_callback_cb(int molid, int atm, const char *client) {}
  virtual void pick_selection_cb(int n, const int *atoms) {}
  virtual void pick_graphics_cb(int molid, int tag, int btn, int shift_state) {}
  virtual void pick_value_cb(float value) {}
  virtual void python_cb(const char *cmd) {}
  virtual void tcl_cb(const char *cmd) {}
  virtual void timestep_cb(int molid, int frame) {}
  virtual void trajectory_cb(int molid, const char *fname) {}
  virtual void graph_label_cb(const char *type, const int *ids, int n) {}
  virtual void userkey_cb(const char *canonical_key_desc) {}
};

#endif


  
