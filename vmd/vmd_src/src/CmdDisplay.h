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
 *	$RCSfile: CmdDisplay.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.68 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   These are the Commands that control the various aspects
 * of the display, like, the clipping planes, eye separation, etc.
 *
 ***************************************************************************/
#ifndef CMDDISPLAY_H
#define CMDDISPLAY_H

#include "Command.h"
#include "DisplayDevice.h"
#include "utilities.h"

class VMDApp;
class Axes;

// These are the Command:: enums used by this file:
//  DISP_RESETVIEW, DISP_STEREO, DISP_PROJ, DISP_EYESEP, 
//  DISP_FOCALLEN, DISP_LIGHT_ON, DISP_LIGHT_HL, DISP_LIGHT_ROT, 
//  DISP_MATERIALS_CHANGE, DISP_CLIP, DISP_DEPTHCUE, DISP_ANTIALIAS, 
//  DISP_CULLING, DISP_BACKGROUNDGRADIENT,
//  DISP_UPDATE, CMD_AXES, CMD_STAGE

/// reset the current view for the current scene
class CmdResetView : public Command {
protected:
  virtual void create_text(void);
public:
  CmdResetView();
};


/// set stereo mode of display
class CmdDisplayStereo : public Command {
private:
  char *mode;
protected:
  virtual void create_text(void);
public:
  CmdDisplayStereo(const char *newmode);
  ~CmdDisplayStereo();
};


/// turn on/off stereo eye swap 
class CmdDisplayStereoSwap : public Command {
private:
  int onoff;
protected:
  virtual void create_text(void);
public:
  CmdDisplayStereoSwap(int turnon);
};


/// set caching mode of display
class CmdDisplayCacheMode : public Command {
private:
  char *mode;
protected:
  virtual void create_text(void);
public:
  CmdDisplayCacheMode(const char *newmode);
  ~CmdDisplayCacheMode();
};


/// set rendering mode of display
class CmdDisplayRenderMode : public Command {
private:
  char *mode;
protected:
  virtual void create_text(void);
public:
  CmdDisplayRenderMode(const char *newmode);
  ~CmdDisplayRenderMode();
};


/// set the projection to either perspective or orthographic
class CmdDisplayProj : public Command {
public:
  CmdDisplayProj(const char *proj);
  ~CmdDisplayProj();
private:
  char *projection;
protected:
  virtual void create_text(void);
};


/// set eye separation of display
class CmdDisplayEyesep : public Command {
private:
  float sep;
protected:
  virtual void create_text(void);
public:
  CmdDisplayEyesep(float newsep);
};


/// set focal length of display
class CmdDisplayFocallen : public Command {
private:
  float flen;
protected:
  virtual void create_text(void);
public:
  CmdDisplayFocallen(float newlen);
};


/// set screen height value
class CmdDisplayScreenHeight : public Command {
private:
  float val;
protected:
  virtual void create_text(void);
public:
  CmdDisplayScreenHeight(float newval);
};


/// set distance to screen from origin
class CmdDisplayScreenDistance : public Command {
private:
  float val;
protected:
  virtual void create_text(void);
public:
  CmdDisplayScreenDistance(float newval);
};


/// turn on/off antialiasing
class CmdDisplayAAOn : public Command {
private:
  int onoff;
protected:
  virtual void create_text(void);
public:
  CmdDisplayAAOn(int turnon);
};


/// turn on/off depth cueing
class CmdDisplayDepthcueOn : public Command {
private:
  int onoff;
protected:
  virtual void create_text(void);
public:
  CmdDisplayDepthcueOn(int turnon);
};


/// turn on/off culling
class CmdDisplayCullingOn : public Command {
private:
  int onoff;
protected:
  virtual void create_text(void);
public:
  CmdDisplayCullingOn(int turnon);
};

/// turn on/off background gradient 
class CmdDisplayBackgroundGradientOn : public Command {
private:
  int onoff;
protected:
  virtual void create_text(void);
public:
  CmdDisplayBackgroundGradientOn(int turnon);
};



/// FPS indicator
class CmdDisplayFPSOn : public Command {
private:
  int onoff;
protected:
  virtual void create_text(void);
public:
  CmdDisplayFPSOn(int turnon);
};


///  clipping plane controls
/// This handles the whole range of clipping plane options
///  There are derived classes so you won't have to have the funky flags
/// to change {near,fixed} clipping plane {to a fixed,by a relative} amount
/// or not
class CmdDisplayClip : public Command {
private:
  int changenear;		// near (T) or far (F) clip plane
  int setval;			// absolute (T) or relative (F) change
  float amount;			// how much to change
protected:
  virtual void create_text(void);
public:
  CmdDisplayClip(int ischangenear, int issetval, float newamt);
};


/// Change the near clipping plane to a fixed value
/// leaving everything else up to the base class :)
class CmdDisplayClipNear : public CmdDisplayClip {
 public:
   CmdDisplayClipNear(float nearval) : 
        CmdDisplayClip(TRUE, TRUE, nearval) {
   }
};


/// Change the near clipping plane to by a relative value
class CmdDisplayClipNearRel : public CmdDisplayClip {
 public:
   CmdDisplayClipNearRel(float nearval) : 
         CmdDisplayClip(TRUE, FALSE, nearval) {
   }
};


/// Change the far clipping plane to a fixed value
class CmdDisplayClipFar : public CmdDisplayClip {
 public:
   CmdDisplayClipFar(float farclip) : 
          CmdDisplayClip(FALSE, TRUE, farclip) {
   }
};


/// Change the far clipping plane to by a relative value
class CmdDisplayClipFarRel : public CmdDisplayClip {
 public:
   CmdDisplayClipFarRel(float farclip) : 
          CmdDisplayClip(FALSE, FALSE, farclip) {
   }
};


/// Set depth cueing mode 
class CmdDisplayCueMode : public Command {
private:
  char *mode;
protected:
  void create_text();
public:
  CmdDisplayCueMode(const char *newmode)
  : Command(DISP_CUEMODE) {
    mode = stringdup(newmode);
  }
  ~CmdDisplayCueMode() { delete [] mode; }
};


/// Set the depth cueing start distance
class CmdDisplayCueStart : public Command {
private:
  float value;
protected:
  void create_text();
public:
  CmdDisplayCueStart(float newvalue)
  : Command(DISP_CUESTART), value(newvalue) {}
};


/// Set the depth cueing end distance
class CmdDisplayCueEnd : public Command {
private:
  float value;
protected:
  void create_text();
public:
  CmdDisplayCueEnd(float newvalue)
  : Command(DISP_CUEEND), value(newvalue) {}
};


/// Set the depth cueing exp/exp2 density parameter
class CmdDisplayCueDensity : public Command {
private:
  float value;
protected:
  void create_text();
public:
  CmdDisplayCueDensity(float newvalue)
  : Command(DISP_CUEDENSITY), value(newvalue) {}
};

/// Set shadow rendering mode 
class CmdDisplayShadowOn : public Command {
private:
  int onoff;
protected:
  void create_text();
public:
  CmdDisplayShadowOn(int turnon);
};


/// Set ambient occlusion rendering mode 
class CmdDisplayAOOn : public Command {
private:
  int onoff;
protected:
  void create_text();
public:
  CmdDisplayAOOn(int turnon);
};

/// Set ambient occlusion ambient factor
class CmdDisplayAOAmbient : public Command {
private:
  float value;
protected:
  void create_text();
public:
  CmdDisplayAOAmbient(float newvalue)
  : Command(DISP_AO_AMBIENT), value(newvalue) {}
};

/// Set ambient occlusion direct lighting factor
class CmdDisplayAODirect : public Command {
private:
  float value;
protected:
  void create_text();
public:
  CmdDisplayAODirect(float newvalue)
  : Command(DISP_AO_DIRECT), value(newvalue) {}
};


/// Set depth of field rendering mode
class CmdDisplayDoFOn : public Command {
private:
  int onoff;
protected:
  void create_text();
public:
  CmdDisplayDoFOn(int turnon);
};

/// Set depth of field f/stop number 
class CmdDisplayDoFFNumber : public Command {
private:
  float value;
protected:
  void create_text();
public:
  CmdDisplayDoFFNumber(float newvalue)
  : Command(DISP_DOF_FNUMBER), value(newvalue) {}
};

/// Set depth of field focal plane distance
class CmdDisplayDoFFocalDist : public Command {
private:
  float value;
protected:
  void create_text();
public:
  CmdDisplayDoFFocalDist(float newvalue)
  : Command(DISP_DOF_FOCALDIST), value(newvalue) {}
};


/// Change the axes location
class CmdDisplayAxes : public Command {
private:
  char *pos;
protected:
  virtual void create_text();
public:
  CmdDisplayAxes(const char *newpos)
  : Command(CMD_AXES) { 
    pos = stringdup(newpos);
  }
  ~CmdDisplayAxes() { delete [] pos; }
};
    

/// Change the stage location
class CmdDisplayStageLocation : public Command { 
public:
  CmdDisplayStageLocation(const char *newpos)
  : Command(CMD_STAGE) { 
    pos = stringdup(newpos);
  }
  ~CmdDisplayStageLocation() { delete [] pos; }
protected:
  void create_text();
private:
  char *pos;
};


/// Set the number of stage panels
class CmdDisplayStagePanels : public Command {
public:
  CmdDisplayStagePanels(int newnum)
  : Command(CMD_STAGE), num(newnum) {}
protected:
  void create_text();
private:
  int num;
};


/// Set the number of stage panels
class CmdDisplayStageSize : public Command {
public:
  CmdDisplayStageSize(float newsz)
  : Command(CMD_STAGE), sz(newsz) {}
protected:
  void create_text();
private:
  float sz;
};


/// Turn on/off the Nth light
class CmdDisplayLightOn : public Command {
private:
  int n, onoff;
protected:
  void create_text();
public:
  CmdDisplayLightOn(int ln, int turnon)
  : Command(DISP_LIGHT_ON), n(ln), onoff(turnon) {}
};


/// Highlight the Nth light
class CmdDisplayLightHL : public Command {
private:
  int n, hl;
protected:
  void create_text();
public:
  CmdDisplayLightHL(int ln, int highlt)
  : Command(DISP_LIGHT_HL), n(ln), hl(highlt) {}
};


/// rotate the position of the Nth light
class CmdDisplayLightRot : public Command {
private:
  float theta;
  char axis;
  int n;
protected:
  void create_text();
public:
  CmdDisplayLightRot(int ln, float th, char ax)
  : Command(DISP_LIGHT_ROT), theta(th), axis(ax), n(ln) {}
};


/// set the position of the Nth light
class CmdDisplayLightMove : public Command {
private:
  float pos[3];
  int n;
protected:
  void create_text();
public:
  CmdDisplayLightMove(int ln, const float *newpos)
  : Command(DISP_LIGHT_MOVE), n(ln) {
    for (int i=0; i<3; i++) pos[i] = newpos[i];
  }
};
 
#endif

