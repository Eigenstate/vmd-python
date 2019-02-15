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
 *      $RCSfile: CmdAnimate.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.39 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Command objects for doing animation.
 *
 ***************************************************************************/
#ifndef CMDANIMATE_H
#define CMDANIMATE_H

#include "Command.h"
#include "Animation.h"

class VMDApp;

// The following uses  the Cmdtypes:
//	ANIM_DIRECTION, ANIM_JUMP, ANIM_SKIP, ANIM_STYLE, ANIM_SPEED,
//	ANIM_READ, ANIM_WRITE, ANIM_DELETE


/// set direction of animation
class CmdAnimDir : public Command {
public:
  Animation::AnimDir newDir; ///< new direction
protected:
  virtual void create_text(void);
public:
  CmdAnimDir(Animation::AnimDir);
};


/// set style of animation
class CmdAnimStyle : public Command {
public:
  Animation::AnimStyle newStyle; ///< new direction
protected:
  virtual void create_text(void);
public:
  CmdAnimStyle(Animation::AnimStyle);
};


/// jump to a new frame
class CmdAnimJump : public Command {
public:
  int newFrame; ///< new frame, can also indicate to move to beginning or end
protected:
  virtual void create_text(void);
public:
  CmdAnimJump(int);
};


/// set frame skip value
class CmdAnimSkip : public Command {
public:
  int newSkip;                          ///< new frame skip
protected:
  virtual void create_text(void);
public:
  CmdAnimSkip(int);
};


/// set animation speed
class CmdAnimSpeed : public Command {
public:
  float newSpeed;                       ///< new animation speed
protected:
  virtual void create_text(void);
public:
  CmdAnimSpeed(float);
};


/// write frames to a file
class CmdAnimWriteFile : public Command {
public:
  int whichMol;				///< which molecule to affect
  char *fileType;			///< kind of file to write
  int begFrame, endFrame, frameSkip;	///< frames to write
  char *fileName;			///< name of file to write

protected:
  virtual void create_text();

public:
  CmdAnimWriteFile(int molid, const char *fname, const char *ftype, 
                   int bf,int ef,int fs);
  virtual ~CmdAnimWriteFile();
};


/// delete frames
class CmdAnimDelete : public Command {
public:
  int whichMol;				///< which molecule to affect
  int begFrame, endFrame, frameSkip;	///< frames to delete
protected:
  virtual void create_text(void);
public:
  CmdAnimDelete(int molid, int bf, int ef, int fs);
};


/// duplicate a given frame at the end of the traj.
class CmdAnimDup : public Command {
public:
  int whichMol;                         ///< which molecule to affect
  int whichFrame;                       ///< which frame to copy
protected:
  virtual void create_text(void);
public:
  CmdAnimDup(int frame, int molid);
};


//////////// Signal that a molecule has changed frames

/// not really a command, but that's how info gets passed to the GUI's
class CmdAnimNewFrame : public Command {
public:
  CmdAnimNewFrame() : Command(ANIM_NEW_FRAME) {}
};


/// signal that a molecule has a new number of frames
class CmdAnimNewNumFrames : public Command {
public:
  CmdAnimNewNumFrames() : Command(ANIM_NEW_NUM_FRAMES) {}
};

#endif

