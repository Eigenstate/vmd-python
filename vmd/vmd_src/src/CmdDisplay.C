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
 *	$RCSfile: CmdDisplay.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.84 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   These are the Commands that control the various aspects
 * of the display, like, the clipping planes, eye separation, etc.
 *
 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#include "config.h"
#include "CmdDisplay.h"
#include "utilities.h"
#include "Inform.h"

// These are the Command:: enums used by this file:
//  DISP_STEREO, DISP_STEREOSWAP,  DISP_CACHEMODE, 
//  DISP_RENDERMODE, DISP_PROJ, DISP_EYESEP, 
//  DISP_FOCALLEN, DISP_LIGHT_ON, DISP_LIGHT_HL, DISP_LIGHT_ROT, 
//  DISP_MATERIALS_CHANGE, DISP_CLIP, DISP_DEPTHCUE, DISP_ANTIALIAS, 
//  DISP_SCRHEIGHT, DISP_SCRDIST, CMD_AXES, CMD_STAGE

void CmdResetView::create_text(void) {
  *cmdText << "display resetview" << ends;
}

CmdResetView::CmdResetView()
: Command(Command::DISP_RESETVIEW) { }

void CmdDisplayStereo::create_text(void) {
  *cmdText << "display stereo " << mode << ends;
}
CmdDisplayStereo::CmdDisplayStereo(const char *newmode)
  : Command(Command::DISP_STEREO) {
  mode = stringdup(newmode);
}
CmdDisplayStereo::~CmdDisplayStereo() {
  delete [] mode;
}


void CmdDisplayStereoSwap::create_text(void) {
  *cmdText << "display stereoswap " << (onoff ? "on" : "off") << ends;
}
CmdDisplayStereoSwap::CmdDisplayStereoSwap(int turnon)
  : Command(Command::DISP_STEREOSWAP) {
  onoff = turnon;
}


void CmdDisplayCacheMode::create_text(void) {
  *cmdText << "display cachemode " << mode << ends;
}
CmdDisplayCacheMode::CmdDisplayCacheMode(const char *newmode)
  : Command(Command::DISP_CACHEMODE) {
  mode = stringdup(newmode);
}
CmdDisplayCacheMode::~CmdDisplayCacheMode() {
  delete [] mode;
}


void CmdDisplayRenderMode::create_text(void) {
  *cmdText << "display rendermode " << mode << ends;
}
CmdDisplayRenderMode::CmdDisplayRenderMode(const char *newmode)
  : Command(Command::DISP_RENDERMODE) {
  mode = stringdup(newmode);
}
CmdDisplayRenderMode::~CmdDisplayRenderMode() {
  delete [] mode;
}


void CmdDisplayProj::create_text(void) {
  *cmdText << "display projection " << projection << ends;
}

CmdDisplayProj::CmdDisplayProj(const char *proj)
: Command(Command::DISP_PROJ) {
  projection = stringdup(proj);
}
CmdDisplayProj::~CmdDisplayProj() {
  delete [] projection;
}

void CmdDisplayEyesep::create_text(void) {
  *cmdText << "display eyesep " << sep << ends;
}

CmdDisplayEyesep::CmdDisplayEyesep(float newsep)
  : Command(Command::DISP_EYESEP), sep(newsep) {}

void CmdDisplayFocallen::create_text(void) {
  *cmdText << "display focallength " << flen << ends;
}

CmdDisplayFocallen::CmdDisplayFocallen(float newlen)
  : Command(Command::DISP_FOCALLEN), flen(newlen) {}

//////////////////// set screen height value
void CmdDisplayScreenHeight::create_text(void) {
  *cmdText << "display height " << val << ends;
}

CmdDisplayScreenHeight::CmdDisplayScreenHeight(float newval)
  : Command(Command::DISP_SCRHEIGHT) {
  val = newval;
}


//////////////////// set distance to screen from origin
void CmdDisplayScreenDistance::create_text(void) {
  *cmdText << "display distance " << val << ends;
}

CmdDisplayScreenDistance::CmdDisplayScreenDistance(float newval)
  : Command(Command::DISP_SCRDIST) {
  val = newval;
}


void CmdDisplayAAOn::create_text(void) {
  *cmdText << "display antialias " << (onoff ? "on" : "off") << ends;
}

CmdDisplayAAOn::CmdDisplayAAOn(int turnon)
  : Command(Command::DISP_ANTIALIAS) {
  onoff = turnon;
}

void CmdDisplayDepthcueOn::create_text(void) {
  *cmdText << "display depthcue " << (onoff ? "on" : "off") << ends;
}

CmdDisplayDepthcueOn::CmdDisplayDepthcueOn(int turnon)
  : Command(Command::DISP_DEPTHCUE) {
  onoff = turnon;
}

void CmdDisplayCullingOn::create_text(void) {
  *cmdText << "display culling " << (onoff ? "on" : "off") << ends;
}

CmdDisplayCullingOn::CmdDisplayCullingOn(int turnon)
  : Command(Command::DISP_CULLING) {
  onoff = turnon;
}

void CmdDisplayBackgroundGradientOn::create_text(void) {
  *cmdText << "display backgroundgradient " << (onoff ? "on" : "off") << ends;
}

CmdDisplayBackgroundGradientOn::CmdDisplayBackgroundGradientOn(int turnon)
  : Command(Command::DISP_BACKGROUNDGRADIENT) {
  onoff = turnon;
}

void CmdDisplayFPSOn::create_text() {
  *cmdText << "display fps " << (onoff ? "on" : "off") << ends;
}

CmdDisplayFPSOn::CmdDisplayFPSOn(int turnon)
  : Command(Command::DISP_FPS) {
  onoff = turnon;
}

void CmdDisplayClip::create_text(void) {
  *cmdText << "display " << (changenear ? "near" : "far");
  *cmdText << "clip " << (setval ? "set " : "add ");
  *cmdText << amount << ends;
}

CmdDisplayClip::CmdDisplayClip(int ischangenear, int issetval, 
                                float newamt)
  : Command(Command::DISP_CLIP) {
  changenear = ischangenear;
  setval = issetval;
  amount = newamt;
}  

/////////////////////  depth cueing controls

void CmdDisplayCueMode::create_text(void) {
  *cmdText << "display cuemode " << mode << ends;
}

void CmdDisplayCueStart::create_text(void) {
  *cmdText << "display cuestart " << value << ends;
}

void CmdDisplayCueEnd::create_text(void) {
  *cmdText << "display cueend " << value << ends;
}

void CmdDisplayCueDensity::create_text(void) {
  *cmdText << "display cuedensity " << value << ends;
}

/// shadow controls
void CmdDisplayShadowOn::create_text(void) {
  *cmdText << "display shadows " << (onoff ? "on" : "off") << ends;
}

CmdDisplayShadowOn::CmdDisplayShadowOn(int turnon)
  : Command(Command::DISP_SHADOW) {
  onoff = turnon;
}


/// ambient occlusion controls
void CmdDisplayAOOn::create_text(void) {
  *cmdText << "display ambientocclusion " << (onoff ? "on" : "off") << ends;
}

CmdDisplayAOOn::CmdDisplayAOOn(int turnon)
  : Command(Command::DISP_AO) {
  onoff = turnon;
}

void CmdDisplayAOAmbient::create_text(void) {
  *cmdText << "display aoambient " << value << ends;
}

void CmdDisplayAODirect::create_text(void) {
  *cmdText << "display aodirect " << value << ends;
}


/// depth of field controls
void CmdDisplayDoFOn::create_text(void) {
  *cmdText << "display dof " << (onoff ? "on" : "off") << ends;
}

CmdDisplayDoFOn::CmdDisplayDoFOn(int turnon)
  : Command(Command::DISP_DOF) {
  onoff = turnon;
}

void CmdDisplayDoFFNumber::create_text(void) {
  *cmdText << "display dof_fnumber " << value << ends;
}

void CmdDisplayDoFFocalDist::create_text(void) {
  *cmdText << "display dof_focaldist " << value << ends;
}


/// axes/stage controls
void CmdDisplayAxes::create_text(void) {
  *cmdText << "axes location " << pos << ends;
}

void CmdDisplayStageLocation::create_text() {
  *cmdText << "stage location " << pos << ends;
}

void CmdDisplayStagePanels::create_text() {
  *cmdText << "stage panels " << num << ends;
}

void CmdDisplayStageSize::create_text() {
  *cmdText << "stage size " << sz << ends;
}

void CmdDisplayLightOn::create_text(void) {
  *cmdText << "light " << n << (onoff ? " on" : " off") << ends;
}

void CmdDisplayLightHL::create_text(void) {
  *cmdText << "light " << n << (hl ? " highlight" : " unhighlight") << ends;
}

void CmdDisplayLightRot::create_text(void) {
  *cmdText << "light " << n << " rot " << axis << " " << theta << ends;
}

void CmdDisplayLightMove::create_text() {
  *cmdText << "light " << n << " pos { " << pos[0] << " " << pos[1] << " "
           << pos[2] << ends;
}

