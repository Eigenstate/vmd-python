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
 *      $RCSfile: CmdAnimate.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.53 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Command objects for doing animation.
 *
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "config.h"
#include "CmdAnimate.h"
#include "MoleculeList.h"
#include "CommandQueue.h"
#include "utilities.h"
#include "Inform.h"
#include "VMDApp.h"
#include "CoorPluginData.h"

// The following uses the Cmdtypes:
//	ANIM_DIRECTION, ANIM_JUMP, ANIM_SKIP, ANIM_STYLE, ANIM_SPEED,
//	ANIM_READ, ANIM_WRITE, ANIM_DELETE, ANIM_READ_DELETE_FILE


void CmdAnimDir::create_text(void) {
  *cmdText << "animate ";
  if(newDir == Animation::ANIM_REVERSE)
    *cmdText << "reverse";
  else if(newDir == Animation::ANIM_REVERSE1)
    *cmdText << "prev";
  else if(newDir == Animation::ANIM_FORWARD)
    *cmdText << "forward";
  else if(newDir == Animation::ANIM_FORWARD1)
    *cmdText << "next";
  else if(newDir == Animation::ANIM_PAUSE)
    *cmdText << "pause";
  *cmdText << ends;
}

CmdAnimDir::CmdAnimDir(Animation::AnimDir ad)
  : Command(Command::ANIM_DIRECTION) {
  newDir = ad;
}


void CmdAnimStyle::create_text(void) {
  *cmdText << "animate style " << animationStyleName[newStyle] << ends;
}

CmdAnimStyle::CmdAnimStyle(Animation::AnimStyle as)
  : Command(Command::ANIM_STYLE) {
  newStyle = as;
}

void CmdAnimJump::create_text(void) {
  *cmdText << "animate goto " << newFrame << ends;
}

CmdAnimJump::CmdAnimJump(int newval)
  : Command(Command::ANIM_JUMP) {
  newFrame = newval;
}

void CmdAnimSkip::create_text(void) {
  *cmdText << "animate skip " << newSkip << ends;
}

CmdAnimSkip::CmdAnimSkip(int newval)
  : Command(Command::ANIM_SKIP) {
  newSkip = newval;
}

void CmdAnimSpeed::create_text(void) {
  *cmdText << "animate speed " << newSpeed << ends;
}

CmdAnimSpeed::CmdAnimSpeed(float newval)
  : Command(Command::ANIM_SPEED) {
  newSpeed = newval;
}

void CmdAnimWriteFile::create_text(void) {
  *cmdText << "animate write " << fileType << " {" << fileName << "}";
  *cmdText << " beg " << begFrame;
  *cmdText << " end " << endFrame;
  *cmdText << " skip " << frameSkip;
  if(whichMol >= 0)
    *cmdText << " " << whichMol;
  *cmdText << ends;
}

CmdAnimWriteFile::CmdAnimWriteFile(int m,const char *n,const char *t,
       int bf,int ef,int fs) 
: Command(Command::ANIM_WRITE) {
  whichMol = m;
  fileType = stringdup(t);
  begFrame = bf;
  endFrame = ef;
  frameSkip = fs;
  fileName = stringdup(n);
}

CmdAnimWriteFile::~CmdAnimWriteFile(void) {
  delete [] fileName;
  delete [] fileType;
}


void CmdAnimDelete::create_text(void) {
  *cmdText << "animate delete ";
  if(begFrame < 0 && endFrame < 0 && frameSkip < 0) {
    *cmdText << "all";
  } else {
    *cmdText << " beg " << begFrame;
    *cmdText << " end " << endFrame;
    *cmdText << " skip " << frameSkip;
  }
  if(whichMol >= 0)
    *cmdText << " " << whichMol;
  *cmdText << ends;
}

CmdAnimDelete::CmdAnimDelete(int m, int bf, int ef, int fs) 
: Command(Command::ANIM_DELETE) {
  whichMol = m;
  begFrame = bf;
  endFrame = ef;
  frameSkip = fs;
}

void CmdAnimDup::create_text(void)
{
  *cmdText << "animate dup";
  if (whichFrame == -1) {
    *cmdText << " frame now";
  } else if (whichFrame < 0) {
    *cmdText << " frame null";
  } else {
    *cmdText << " frame " << whichFrame;
  }
  if (whichMol >= 0) {
    *cmdText << " " << whichMol;
  }
  *cmdText << ends;
}
CmdAnimDup::CmdAnimDup(int frame, int molid) 
: Command(Command::ANIM_DUP) {
  whichFrame = frame;
  whichMol = molid;
}

