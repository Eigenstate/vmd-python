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
 *      $RCSfile: PythonTextInterp.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.28 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python text interpreter
 ***************************************************************************/
#ifndef PYTHON_TEXT_INTERP_H
#define PYTHON_TEXT_INTERP_H

#if defined(__APPLE__)
// use the Apple-provided Python framework
#include "Python/Python.h"
#else
#include "Python.h"
#endif

#include "TextInterp.h"

class VMDApp;

/// TextInterp subclass implementing the Python-based text command interpreter
class PythonTextInterp : public TextInterp {
private:
  VMDApp *app;
  int needPrompt;
  int have_tkinter;
  int in_tk;

public:
  PythonTextInterp(VMDApp *);
  ~PythonTextInterp();

  virtual void doEvent();
  virtual int doTkUpdate();
  virtual int evalString(const char *);
  virtual int evalFile(const char *);

  //virtual void display_update_cb();
  virtual void frame_cb(int molid, int frame);
  //virtual void help_cb(const char *topic);
  virtual void initialize_structure_cb(int molid, int create_or_destroy);
  virtual void molecule_changed_cb(int molid, int code);
  //virtual void logfile_cb(const char *cmd);
  virtual void pick_atom_cb(int molid, int atomid, int shift_state, bool ispick);
  //virtual void pick_atom_callback_cb(int molid, int atm, const char *client);
  virtual void pick_value_cb(float value);
  virtual void python_cb(const char *cmd);
  //virtual void tcl_cb(const char *cmd) {}
  virtual void timestep_cb(int molid, int frame);
  virtual void trajectory_cb(int molid, const char *fname);
  virtual void userkey_cb(const char *canonical_key_desc);
};

#endif
