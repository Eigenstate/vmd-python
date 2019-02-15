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
 *    $RCSfile: Command.h,v $
 *    $Author: johns $    $Locker:  $        $State: Exp $
 *    $Revision: 1.172 $    $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *     This file contains the base definition of the Command object.
 *  The idea is that all actions should be instantiated as
 *  a derivitive of a Command object and placed in a command queue.
 *  At every event loop, the commands are read off the current command
 *  queue and told to "execute".  When each command
 *  is finished, all of the UIs are notified that a command of some 
 *  "cmdtype" executed, so that the UI can update its information 
 *  if so desired.
 *
 *    Collections of related commands (i.e. command which all start with
 *  the same word, but are different variations) should be placed in files
 *  with the names 'Cmd*.C' and 'Cmd*.h'.  These files should define all the
 *  specific Command subclasses, as well as provide global 'text callback'
 *  functions which are called to process text commands which start with
 *  a specific word.
 *
 ***************************************************************************/
#ifndef COMMAND_H
#define COMMAND_H

#include "Inform.h"
class CommandQueue;

/// Command base class from which all other commands are derived
/// Derived classes must provide a unique constructor, destructor, and
/// optionally provide a 'create_text' routine
/// if the command has a text equivalent which should be echoed to a log file.
class Command {
public:
  // ALL the commands types must be in this enumeration.  The last element
  // must be "TOTAL" and the sequence must start at zero.  This is because
  // we reference Command::TOTAL to figure out how many commands there are.
  enum Cmdtype { 
    ROTMAT, ROTATE, TRANSLATE, SCALE, ROCKON, ROCKOFF, STOPROT,
    ANIM_DIRECTION, ANIM_JUMP, ANIM_SKIP, ANIM_STYLE, ANIM_SPEED,
    ANIM_READ, ANIM_WRITE, ANIM_DELETE, ANIM_DUP, ANIM_NEW_FRAME, 
    ANIM_NEW_NUM_FRAMES,
    ATOMSEL_ADDMACRO, ATOMSEL_DELMACRO,
    COLOR_NAME, COLOR_CHANGE, COLOR_SCALE_METHOD, COLOR_SCALE_SETTINGS,
    COLOR_SCALE_COLORS,
    COLOR_ADD_ITEM,
    DISP_BACKGROUNDGRADIENT,
    DISP_RESETVIEW, DISP_STEREO, DISP_STEREOSWAP, 
    DISP_CACHEMODE, DISP_RENDERMODE, 
    DISP_PROJ, DISP_EYESEP, 
    DISP_FOCALLEN, DISP_LIGHT_ON, DISP_LIGHT_HL, DISP_LIGHT_ROT,
    DISP_LIGHT_MOVE, DISP_FPS,
    DISP_CLIP, DISP_DEPTHCUE, DISP_CULLING, 
    DISP_ANTIALIAS, CMD_AXES, CMD_STAGE,
    DISP_SCRHEIGHT, DISP_SCRDIST, 
    DISP_CUESTART, DISP_CUEEND, DISP_CUEDENSITY, DISP_CUEMODE,
    DISP_SHADOW,
    DISP_AO, DISP_AO_AMBIENT, DISP_AO_DIRECT, 
    DISP_DOF, DISP_DOF_FNUMBER, DISP_DOF_FOCALDIST, 
    IMD_ATTACH, IMD_SIM, IMD_RATE, IMD_COPYUNITCELL,
    INTERP_EVENT,
    LABEL_ADD, LABEL_ADDSPRING, LABEL_SHOW, LABEL_DELETE, 
    LABEL_TEXTSIZE, LABEL_TEXTTHICKNESS, LABEL_TEXTOFFSET, LABEL_TEXTFORMAT,
    MATERIAL_ADD, MATERIAL_RENAME, MATERIAL_CHANGE,
    MATERIAL_DELETE, MATERIAL_DEFAULT,
    MENU_SHOW, MENU_TK_ADD, MENU_TK_REMOVE,
    MOL_NEW, MOL_DEL, MOL_ACTIVE, MOL_FIX, MOL_ON, MOL_TOP,
    MOL_SELECT, MOL_REP, MOL_COLOR, MOL_ADDREP, MOL_DELREP,
    MOL_MODREPITEM, MOL_MODREP, MOL_MATERIAL, MOL_CANCEL, 
    MOL_REANALYZE, MOL_BONDRECALC, MOL_SSRECALC, 
    MOL_REPSELUPDATE, MOL_REPCOLORUPDATE, MOL_DRAWFRAMES, 
    MOL_SHOWPERIODIC, MOL_NUMPERIODIC, MOL_SCALEMINMAX, MOL_SMOOTHREP,
    MOL_VOLUME, MOL_RENAME,
    MOL_SHOWREP,
    MOUSE_MODE,
    PICK_EVENT, PLUGIN_UPDATE,
    RENDER, RENDER_OPTION, 
    MOBILE_MODE, SPACEBALL_MODE,
    TOOL_CREATE, TOOL_CHANGE, TOOL_DELETE, TOOL_SCALE, TOOL_SCALE_FORCE,
    TOOL_SCALE_SPRING,
    TOOL_OFFSET, TOOL_REP,
    TOOL_ADD_DEVICE, TOOL_DELETE_DEVICE, TOOL_CALLBACK,
    TOTAL
  };

private:
  Cmdtype mytype;     ///<  This is used to distinguish between "Commands"
  int hasTextCmd;     ///< flag whether the command even HAS a text equiv

protected:
/// stream-based object to format text
  Inform *cmdText;

  /// virtual function which is called when a text version of the command
  /// must be created to be printed to the console or a log file.  
  virtual void create_text() { 
    hasTextCmd = 0; 
  }

public:
  /// constructor ... command type, max length of text command, UI id
  /// constructor for derived class should print text rep of this command
  /// to the stream 'cmdText'
  Command(Cmdtype newtype)
      : mytype(newtype), hasTextCmd(1), cmdText((Inform *) 0) {}

  /// destructor; free string spaced used for text rep of this command
  virtual ~Command() {}

  /// if a Tcl text equivalent is available, write it into the given stream 
  /// return whether this command has a text equivalent
  int has_text(Inform *str) {
    cmdText = str;
    create_text();
    return hasTextCmd;
  }
    
  /// return unique ID code of this command
  Cmdtype gettype() { 
    return mytype; 
  }
};

#endif

