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
 *	$RCSfile: DisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.147 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * DisplayDevice - abstract base class for all particular objects which
 *	can process a list of drawing commands and render the drawing
 *	to some device (screen, file, preprocessing script, etc.)
 *
 ***************************************************************************/

#include <math.h>
#include "DisplayDevice.h"
#include "Inform.h"
#include "DispCmds.h"
#include "utilities.h"
#include "Mouse.h"     // for WAIT_CURSOR
#include "VMDDisplayList.h"
#include "Scene.h"

// static data for this object (yuck)
static const char *cacheNameStr[1]  = { "Off" };
static const char *renderNameStr[1] = { "Normal" };
static const char *stereoNameStr[1] = { "Off" };

const char *DisplayDevice::projNames[NUM_PROJECTIONS] = {
  "Perspective", "Orthographic"
};

const char *DisplayDevice::cueModeNames[NUM_CUE_MODES] = {
  "Linear", "Exp", "Exp2"
};

/////////////////////////  constructor and destructor  
DisplayDevice::DisplayDevice (const char *nm) : transMat(16) {
  vmdapp = NULL;              // set VMDApp ptr to NULL initially
  name = stringdup(nm);       // save the string name of this display device
  num_display_processes  = 1; // set number of rendering processes etc
  renderer_process = 1;       // we're a rendering process until told otherwise
  _needRedraw = 0;            // Start life not needing to be redrawn. 
  backgroundmode = 0;         // set default background mode to solid color

  // set drawing characteristics default values
  lineStyle = ::SOLIDLINE;
  lineWidth = 1;
  sphereRes = 3;
  cylinderRes = 6;
  sphereMode = ::SOLIDSPHERE;

  // set scalar values
  aaAvailable = cueingAvailable = TRUE;
  aaPrevious = aaEnabled = cueingEnabled = cullingEnabled = FALSE;
  xOrig = yOrig = xSize = ySize = 0;
  screenX = screenY = 0;

  // set viewing geometry ... looking from z-axis in negative direction,
  // with 90-degree field of view and assuming the origin is in the
  // center of the viewer's 'screen'
  nearClip = 0.5f;
  farClip = 10.0f;
  eyePos[0] = eyePos[1] = 0.0f;  eyePos[2] = 2.0f;
  set_screen_pos(2.0f * eyePos[2], 0.0f, 4.0f/3.0f);

  // set initial depth cueing parameters 
  // (defaults are compatible with old revs of VMD)
  cueMode = CUE_EXP2;
  cueDensity = 0.32f;
  cueStart = 0.5f;
  cueEnd = 10.0f;

  // set initial shadow mode
  shadowEnabled = 0;

  // set initial ambient occlusion lighting parameters
  aoEnabled = 0;
  aoAmbient = 0.8f;
  aoDirect = 0.3f;

  // set initial depth of field parameters
  dofEnabled = 0;      // DoF is off by default
  dofFNumber = 64;     // f/64 is a reasonable aperture for molecular scenes
  dofFocalDist = 0.7f; // front edge of mol should be in-focus by default

  // XXX stereo modes and rendering modes should be enumerated 
  // dynamically not hard-coded, to allow much greater flexibility

  // Setup stereo options ... while there is no stereo mode by default,
  // set up normal values for stereo data
  inStereo = 0;
  stereoSwap = 0;
  stereoModes = 1;
  stereoNames = stereoNameStr;

  // Setup caching mode options
  cacheMode = 0;
  cacheModes = 1;
  cacheNames = cacheNameStr;

  // Setup rendering mode options
  renderMode = 0;
  renderModes = 1;
  renderNames = renderNameStr;

  // default view/projection settings
  eyeSep = 0.065f;               // default eye seperation
  eyeDist = eyePos[2];	         // assumes viewer on pos z-axis

  float lookatorigin[3];
  vec_scale(&lookatorigin[0], -1, &eyePos[0]); // calc dir to origin
  set_eye_dir(&lookatorigin[0]);               // point camera at origin
  upDir[0] = upDir[2] = 0.0;  upDir[1] = 1.0;
  calc_eyedir();                 // determines eye separation direction
  my_projection = PERSPECTIVE;

  // load identity matrix onto top of transformation matrix stack
  Matrix4 temp_ident;
  transMat.push(temp_ident);

  mouseX = mouseY = 0;
}

// destructor
DisplayDevice::~DisplayDevice(void) {
  set_stereo_mode(0);		// return to non-stereo, if necessary
  delete [] name;
}

int DisplayDevice::set_eye_defaults() {
  float defaultDir[3];
  float defaultPos[3] = {0, 0, 2};           // camera 2 units back from origin
  float defaultUp[3] = {0, 1, 0};            // Y is up

  vec_scale(&defaultDir[0], -1, &eyePos[0]); // calc dir to origin
  set_eye_dir(&defaultDir[0]);               // point camera at origin

  set_eye_pos(&defaultPos[0]);
  set_eye_dir(&defaultDir[0]);
  set_eye_up(&defaultUp[0]);

  return TRUE;
}

/////////////////////////  protected nonvirtual routines  
// calculate the position of the near frustum plane, based on current values
// of Aspect, vSize, zDist, nearClip and eyePosition
// Assumes the viewer is looking toward the xy-plane
void DisplayDevice::calc_frustum(void) {
  float d; 
  float halfvsize = 0.5f * vSize;       
  float halfhsize = Aspect * halfvsize; // width = aspect * height

  // if the frustum parameters don't cause division by zero,
  // calculate the new view frustum
  if(eyePos[2] - zDist != 0.0f) {
    // scaling ratio for the view frustum, essentially the amount of 
    // perspective to apply.  Since we define the nearClip plane in
    // the user interface, we can control how strong the perspective
    // is by varying (eyePos[2] - zDist) or by scaling d by some other
    // user controllable factor.  In order to make this more transparent
    // to the user however, we'd need to automatically apply a scaling 
    // operation on the molecular geometry so that it looks about the same
    // despite the perspective change.  We should also be able to calculate
    // the field of view angles (vertical, horizontal, and diagonal) based
    // on all of these variables.
    d = nearClip / (eyePos[2] - zDist);

    cpRight = d * halfhsize;     // right side is at half width
     cpLeft = -cpRight;          // left side is at negative half width
       cpUp = d * halfvsize;     // top side is at half height              
     cpDown = -cpUp;             // bottom is at negative half height
  }
}


// calculate eyeSepDir, based on up vector and look vector
// eyeSepDir = 1/2 * eyeSep * (lookdir x updir) / mag(lookdir x updir)
void DisplayDevice::calc_eyedir(void) {
  float *L = eyeDir;
  float *U = upDir;
  float m, A = 0.5f * eyeSep;
  eyeSepDir[0] = L[1] * U[2] - L[2] * U[1];
  eyeSepDir[1] = L[2] * U[0] - L[0] * U[2];
  eyeSepDir[2] = L[0] * U[1] - L[1] * U[0];
  m = sqrtf(eyeSepDir[0] * eyeSepDir[0] + eyeSepDir[1] * eyeSepDir[1] +
            eyeSepDir[2] * eyeSepDir[2]);
  if(m > 0.0)
    A /= m;
  else
    A = 0.0;
  eyeSepDir[0] *= A;
  eyeSepDir[1] *= A;
  eyeSepDir[2] *= A;
}

/////////////////////////  public nonvirtual routines  

// Copy all relevant properties from one DisplayDevice to another
DisplayDevice& DisplayDevice::operator=( DisplayDevice &display) {
  int i;
  
  xOrig = display.xOrig;
  yOrig = display.yOrig;
  xSize = display.xSize;
  ySize = display.ySize;
  
  // Do something about the stack.  For the moment, only copy the top
  // item on the stack.
  if (transMat.num() > 0) {
    transMat.pop();
  }
  transMat.push( (display.transMat).top() );
  
  for (i=0; i<3; i++) {
    eyePos[i] = display.eyePos[i];
    eyeDir[i] = display.eyeDir[i];
    upDir[i] = display.upDir[i];
    eyeSepDir[i] = display.eyeSepDir[i];
  }
  
  whichEye = display.whichEye;
  nearClip = display.nearClip;
  farClip = display.farClip;
  vSize = display.vSize;
  zDist = display.zDist;
  Aspect = display.Aspect;
  cpUp = display.cpUp;
  cpDown = display.cpDown;
  cpLeft = display.cpLeft;
  cpRight = display.cpRight;
  inStereo = display.inStereo;
  eyeSep = display.eyeSep;
  eyeDist = display.eyeDist;
  lineStyle = display.lineStyle;
  lineWidth = display.lineWidth;
  my_projection = display.my_projection;
  backgroundmode = display.backgroundmode;
  cueingEnabled = display.cueingEnabled;
  cueMode = display.cueMode;
  cueDensity = display.cueDensity;
  cueStart = display.cueStart;
  cueEnd = display.cueEnd;
  shadowEnabled = display.shadowEnabled;
  aoEnabled = display.aoEnabled;
  aoAmbient = display.aoAmbient;
  aoDirect = display.aoDirect; 
  dofEnabled = display.dofEnabled;
  dofFNumber = display.dofFNumber;
  dofFocalDist = display.dofFocalDist;
  return *this;
}

/////////////////////////  public virtual routines

void DisplayDevice::do_resize_window(int w, int h) {
  xSize = w;
  ySize = h;
  set_screen_pos((float)xSize / (float)ySize);
}

//
// event handling routines
//

// queue the standard events (need only be called once ... but this is
// not done automatically by the window because it may not be necessary or
// even wanted)
void DisplayDevice::queue_events(void) { return; }

// read the next event ... returns an event type (one of the above ones),
// and a value.  Returns success, and sets arguments.
int DisplayDevice::read_event(long &, long &) { return FALSE; }

//
// get the current state of the device's pointer (i.e. cursor if it has one)
//

// abs pos of cursor from lower-left corner
int DisplayDevice::x(void) { return mouseX; }

// same, for y direction
int DisplayDevice::y(void) { return mouseY; }

// the shift state (shift key, control key, and/or alt key)
int DisplayDevice::shift_state(void) {
  return 0; // by default, nothing is down
}

// set the Nth cursor shape as the current one.  If no arg given, the
// default shape (n=0) is used.
void DisplayDevice::set_cursor(int) { }

// virtual functions to turn on/off depth cuing and antialiasing
void DisplayDevice::aa_on(void) { }
void DisplayDevice::aa_off(void) { }
void DisplayDevice::cueing_on(void) { 
  if (cueingAvailable)
    cueingEnabled = TRUE;
}
void DisplayDevice::cueing_off(void) { }
void DisplayDevice::culling_on(void) { }
void DisplayDevice::culling_off(void) { }

// return absolute 2D screen coordinates, given 2D or 3D world coordinates.
void DisplayDevice::abs_screen_loc_3D(float *wloc, float *sloc) {
  // just return world coords
  for(int i=0; i < 3; i++)
    sloc[i] = wloc[i];
}

void DisplayDevice::abs_screen_loc_2D(float *wloc, float *sloc) {
  // just return world coords
  for(int i=0; i < 2; i++)
    sloc[i] = wloc[i];
}

// change to a different stereo mode (0 means 'off')
void DisplayDevice::set_stereo_mode(int sm) {
  if(sm != 0) {
    msgErr << "DisplayDevice: Illegal stereo mode " << sm << " specified."
           << sendmsg;
  } else {
    inStereo = sm;
  }
}

// change to a different rendering mode (0 means 'normal')
void DisplayDevice::set_cache_mode(int sm) {
  if(sm != 0) {
    msgErr << "DisplayDevice: Illegal caching mode " << sm << " specified."
           << sendmsg;
  } else {
    cacheMode = sm;
  }
}

// change to a different rendering mode (0 means 'normal')
void DisplayDevice::set_render_mode(int sm) {
  if(sm != 0) {
    msgErr << "DisplayDevice: Illegal rendering mode " << sm << " specified."
           << sendmsg;
  } else {
    renderMode = sm;
  }
}

// replace the current trans matrix with the given one
void DisplayDevice::loadmatrix(const Matrix4 &m) {
  (transMat.top()).loadmatrix(m);
}

// multiply the current trans matrix with the given one
void DisplayDevice::multmatrix(const Matrix4 &m) {
  (transMat.top()).multmatrix(m);
}

//
// virtual routines for preparing to draw, drawing, and finishing drawing
//
  
int DisplayDevice::prepare3D(int) { return 1;}	// ready to draw 3D
void DisplayDevice::clear(void) { }		// erase the device
void DisplayDevice::left(void) {  		// ready to draw left eye
  whichEye = LEFTEYE; 
}
void DisplayDevice::right(void) {  		// ready to draw right eye
  whichEye = RIGHTEYE;
}
void DisplayDevice::normal(void) {  		// ready to draw non-stereo
  whichEye = NOSTEREO;
}
void DisplayDevice::update(int) { }		// finish up after drawing
void DisplayDevice::reshape(void) { }		// refresh device after change

// Grab the screen to a packed RGB unsigned char buffer
unsigned char * DisplayDevice::readpixels_rgb3u(int &x, int &y) { 
  x = 0;
  y = 0;
  return NULL;
}

// Grab the screen to a packed RGBA unsigned char buffer
unsigned char * DisplayDevice::readpixels_rgba4u(int &x, int &y) { 
  x = 0;
  y = 0;
  return NULL;
}


void DisplayDevice::find_pbc_images(const VMDDisplayList *cmdList, 
                                    ResizeArray<Matrix4> &pbcImages) {
  if (cmdList->pbc == PBC_NONE) {
    pbcImages.append(Matrix4());
    return;
  }
  ResizeArray<int> pbcCells;
  find_pbc_cells(cmdList, pbcCells);
  for (int i=0; i<pbcCells.num(); i += 3) {
    int nx=pbcCells[i  ];
    int ny=pbcCells[i+1];
    int nz=pbcCells[i+2];
    Matrix4 mat;
    for (int i1=1; i1<=nx; i1++) mat.multmatrix(cmdList->transX);
    for (int i2=-1; i2>=nx; i2--) mat.multmatrix(cmdList->transXinv);
    for (int i3=1; i3<=ny; i3++) mat.multmatrix(cmdList->transY);
    for (int i4=-1; i4>=ny; i4--) mat.multmatrix(cmdList->transYinv);
    for (int i5=1; i5<=nz; i5++) mat.multmatrix(cmdList->transZ);
    for (int i6=-1; i6>=nz; i6--) mat.multmatrix(cmdList->transZinv);
    pbcImages.append(mat);
  }
}

void DisplayDevice::find_pbc_cells(const VMDDisplayList *cmdList, 
                                    ResizeArray<int> &pbcCells) {
  int pbc = cmdList->pbc;
  if (pbc == PBC_NONE) {
    pbcCells.append3(0, 0, 0);
  } else {
    int npbc = cmdList->npbc;
    int nx = pbc & PBC_X ? npbc : 0;
    int ny = pbc & PBC_Y ? npbc : 0;
    int nz = pbc & PBC_Z ? npbc : 0;
    int nox = pbc & PBC_OPX ? -npbc : 0;
    int noy = pbc & PBC_OPY ? -npbc : 0;
    int noz = pbc & PBC_OPZ ? -npbc : 0;
    int i, j, k;
    for (i=nox; i<=nx; i++) {
      for (j=noy; j<=ny; j++) {
        for (k=noz; k<=nz; k++) {
          if (!(pbc & PBC_NOSELF && !i && !j && !k)) {
            pbcCells.append3(i, j, k);
          }
        }
      }
    }
  }
}


void DisplayDevice::find_instance_images(const VMDDisplayList *cmdList, 
                                         ResizeArray<Matrix4> &instImages) {
  int ninstances = cmdList->instances.num();

#if 0
  printf("DisplayDevice::find_instance_images(): cnt: %d flags: %0x\n", 
         ninstances, cmdList->instanceset);
#endif

  // if no instances are selected for a rep, or we're drawing something
  // that isn't a graphical representation from a molecule, then we 
  // set the instance list to the identity matrix so the geometry is drawn
  if (cmdList->instanceset == INSTANCE_NONE || ninstances == 0) {
    instImages.append(Matrix4());
    return;
  }

  // If we have a non-zero number of rep instances or the instanceset flags
  // are set to INSTANCE_ALL or INSTANCE_NOSELF, we build a list of instances
  // to be drawn and add them to the display commandlist.
  if ((cmdList->instanceset & INSTANCE_NOSELF) != INSTANCE_NOSELF) {
#if 0
    printf("appending self instance\n");
#endif
    instImages.append(Matrix4());
  }
  for (int i=0; i<ninstances; i++) {
    instImages.append(cmdList->instances[i]);
  }

  // ensure we _never_ have an empty transformation list,
  // since it shouldn't be possible except on the last displaylist link
  if (instImages.num() == 0) {
#if 1
    printf("DisplayDevice warning, no instance mats! adding one...\n");
#endif
    instImages.append(Matrix4());
  }
}

//
//*******************  the picking routine  *********************
//
// This scans the given command list until the end, finding which item is
// closest to the given pointer position.
//
// arguments are dimension of picking (2 or 3), position of pointer,
// draw command list, and returned distance from object to eye position.
// Returns ID code ('tag') for item closest to pointer, or (-1) if no pick.
// If an object is picked, the eye distance argument is set to the distance
// from the display's eye position to the object (after its position has been
// found from the transformation matrix).  If the value of the argument when
// 'pick' is called is <= 0, a pick will be generated if any item is near the
// pointer.  If the value of the argument is > 0, a pick will be generated
// only if an item is closer to the eye position than the value of the
// argument.
// For 2D picking, coordinates are relative position in window from
//	lower-left corner (both in range 0 ... 1)
// For 3D picking, coordinates are the world coords of the pointer.  They
//	are the coords of the pointer after its transformation matrix has been
//	applied, and these coordinates are compared to the coords of the objects
//	when their transformation matrices are applied.

// but first, a macro for returning the distance^2 from the eyepos to the
// given position
#define DTOEYE(x,y,z) ( (x-eyePos[0])*(x-eyePos[0]) + \
			(y-eyePos[1])*(y-eyePos[1]) + \
			(z-eyePos[2])*(z-eyePos[2]) )
#define DTOPOINT(x,y,z) ( (x-pos[0])*(x-pos[0]) + \
			(y-pos[1])*(y-pos[1]) + \
			(z-pos[2])*(z-pos[2]) )

int DisplayDevice::pick(int dim, const float *pos, const VMDDisplayList *cmdList,
			float &eyedist, int *unitcell, float window_size) {
  char *cmdptr = NULL;
  int tok;
  float newEyeDist, currEyeDist = eyedist;
  int tag = (-1), inRegion, currTag;
  float minX=0.0f, minY=0.0f, maxX=0.0f, maxY=0.0f;
  float fminX=0.0f, fminY=0.0f, fminZ=0.0f, fmaxX=0.0f, fmaxY=0.0f, fmaxZ=0.0f;
  float wpntpos[3], pntpos[3], cpos[3];

  if(!cmdList)
    return (-1);

  // initialize picking: find screen region to look for object
  if (dim == 2) {
    fminX = pos[0] - window_size;
    fmaxX = pos[0] + window_size;
    fminY = pos[1] - window_size;
    fmaxY = pos[1] + window_size;
    abs_screen_pos(fminX, fminY);
    abs_screen_pos(fmaxX, fmaxY);
#if defined(_MSC_VER)
    // Win32 lacks the C99 round() routine
    // Add +/- 0.5 then then round towards zero.
#if 1
    minX = (int) ((float) (fminX + (fminX >= 0.0f ?  0.5f : -0.5f)));
    maxX = (int) ((float) (fmaxX + (fmaxX >= 0.0f ?  0.5f : -0.5f)));
    minY = (int) ((float) (fminY + (fminY >= 0.0f ?  0.5f : -0.5f)));
    maxY = (int) ((float) (fmaxY + (fmaxY >= 0.0f ?  0.5f : -0.5f)));
#else
    minX = truncf(fminX + (fminX >= 0.0f ?  0.5f : -0.5f));
    maxX = truncf(fmaxX + (fmaxX >= 0.0f ?  0.5f : -0.5f));
    minY = truncf(fminY + (fminY >= 0.0f ?  0.5f : -0.5f));
    maxY = truncf(fmaxY + (fmaxY >= 0.0f ?  0.5f : -0.5f));
//    minX = floor(fminX + 0.5);
//    maxX = floor(fmaxX + 0.5);
//    minY = floor(fminY + 0.5);
//    maxY = floor(fmaxY + 0.5);
#endif
#else
    minX = round(fminX);
    maxX = round(fmaxX);
    minY = round(fminY);
    maxY = round(fmaxY);
#endif

  } else {
    fminX = pos[0] - window_size;
    fmaxX = pos[0] + window_size;
    fminY = pos[1] - window_size;
    fmaxY = pos[1] + window_size;
    fminZ = pos[2] - window_size;
    fmaxZ = pos[2] + window_size;
  }

  // make sure we do not disturb the regular transformation matrix
  transMat.dup();
  (transMat.top()).multmatrix(cmdList->mat);

  // Transform the current pick point for each periodic image 
  ResizeArray<Matrix4> pbcImages;
  find_pbc_images(cmdList, pbcImages);
  //int npbcimages = pbcImages.num();

  ResizeArray<int> pbcCells;
  find_pbc_cells(cmdList, pbcCells);

  // Retreive instance image transformation matrices
  ResizeArray<Matrix4> instanceImages;
  find_instance_images(cmdList, instanceImages);
  int ninstances = instanceImages.num();

  for (int pbcimage=0; pbcimage<pbcImages.num(); pbcimage++) {
    transMat.dup();
    (transMat.top()).multmatrix(pbcImages[pbcimage]);

    for (int instanceimage = 0; instanceimage < ninstances; instanceimage++) {
      transMat.dup();
      (transMat.top()).multmatrix(instanceImages[instanceimage]);

      // scan through the list, getting each command and executing it, until
      // the end of commands token is found
      VMDDisplayList::VMDLinkIter cmditer;
      cmdList->first(&cmditer);
      while((tok = cmdList->next(&cmditer, cmdptr)) != DLASTCOMMAND) {
        switch (tok) {
          case DPICKPOINT:
            // calculate the transformed position of the point
            {
              DispCmdPickPoint *cmd = (DispCmdPickPoint *)cmdptr;
              vec_copy(wpntpos, cmd->postag);
              currTag = cmd->tag;
            }
            (transMat.top()).multpoint3d(wpntpos, pntpos);

            // check if in picking region ... different for 2D and 3D
            if (dim == 2) {
              // convert the 3D world coordinate to 2D (XY) absolute screen 
              // coordinate, and a normalized Z coordinate.
              abs_screen_loc_3D(pntpos, cpos);
        
              // check whether the projected picking position falls within the 
              // view frustum, with the XY coords falling within the displayed 
              // window, and the Z coordinate falling within the view volume
              // between the front and rear clipping planes.
              inRegion = (cpos[0] >= minX && cpos[0] <= maxX &&
                          cpos[1] >= minY && cpos[1] <= maxY &&
                          cpos[2] >= 0.0  && cpos[2] <= 1.0);
            } else {
              // just check to see if the position is in a box centered on our
              // pointer.  The pointer position should already be transformed.
              inRegion = (pntpos[0] >= fminX && pntpos[0] <= fmaxX &&	
                          pntpos[1] >= fminY && pntpos[1] <= fmaxY &&
                          pntpos[2] >= fminZ && pntpos[2] <= fmaxZ);
            }

            // Clip still-viable pick points against all active clipping planes
            if (inRegion) {
              // We must perform a check against all of the active
              // user-defined clipping planes to ensure that only pick points
              // associated with visible geometry can be selected.
              int cp;
              for (cp=0; cp < VMD_MAX_CLIP_PLANE; cp++) {
                // The final result is the intersection of all of the
                // individual clipping plane tests...
                if (cmdList->clipplanes[cp].mode) {
                  float cpdist[3];
                  vec_sub(cpdist, wpntpos, cmdList->clipplanes[cp].center);
                  inRegion &= 
                    (dot_prod(cpdist, cmdList->clipplanes[cp].normal) > 0.0f);
                }
              }
            }
      
            // has a hit occurred?
            if (inRegion) {
              // yes, see if it is closer to the eye than earlier objects
              if(dim==2) 
                newEyeDist = DTOEYE(pntpos[0], pntpos[1], pntpos[2]);
              else 
                newEyeDist = DTOPOINT(pntpos[0],pntpos[1],pntpos[2]);

              if (currEyeDist < 0.0 || newEyeDist < currEyeDist) {
                currEyeDist = newEyeDist;
                tag = currTag;
                if (unitcell) {
                  unitcell[0] = pbcCells[3*pbcimage  ];
                  unitcell[1] = pbcCells[3*pbcimage+1];
                  unitcell[2] = pbcCells[3*pbcimage+2];
                }
              }
            }
            break;

          case DPICKPOINT_ARRAY:
            // loop over all of the pick points in the pick point index array
            DispCmdPickPointArray *cmd = (DispCmdPickPointArray *)cmdptr;
            float *pickpos=NULL;
            float *crds=NULL;
            int *indices=NULL;
            cmd->getpointers(crds, indices); 

            int i;
            for (i=0; i<cmd->numpicks; i++) {
              pickpos = crds + i*3L;
              if (cmd->allselected) {
                currTag = i + cmd->firstindex;
              } else {
                currTag = indices[i];
              }
              vec_copy(wpntpos, pickpos);
              (transMat.top()).multpoint3d(pickpos, pntpos);

              // check if in picking region ... different for 2D and 3D
              if (dim == 2) {
                // convert the 3D world coordinate to 2D absolute screen coord
                abs_screen_loc_3D(pntpos, cpos);
  
                // check to see if the position falls in our picking region
                // including the clipping region (cpos[2])
                inRegion = (cpos[0] >= minX && cpos[0] <= maxX &&
                            cpos[1] >= minY && cpos[1] <= maxY &&
                            cpos[2] >= 0.0  && cpos[2] <= 1.0);
              } else {
                // just check to see if the position is in a box centered on our
                // pointer.  The pointer position should already be transformed.
                inRegion = (pntpos[0] >= fminX && pntpos[0] <= fmaxX &&
                            pntpos[1] >= fminY && pntpos[1] <= fmaxY &&
                            pntpos[2] >= fminZ && pntpos[2] <= fmaxZ);
              }

              // Clip still-viable pick points against active clipping planes
              if (inRegion) {
                // We must perform a check against all of the active
                // user-defined clipping planes to ensure that only pick points
                // associated with visible geometry can be selected.
                int cp;
                for (cp=0; cp < VMD_MAX_CLIP_PLANE; cp++) {
                  // The final result is the intersection of all of the
                  // individual clipping plane tests...
                  if (cmdList->clipplanes[cp].mode) {
                    float cpdist[3];
                    vec_sub(cpdist, wpntpos, cmdList->clipplanes[cp].center);
                    inRegion &= (dot_prod(cpdist, 
                                       cmdList->clipplanes[cp].normal) > 0.0f);
                  }
                }
              }

              // has a hit occurred?
              if (inRegion) {
                // yes, see if it is closer to the eye than earlier hits
                if (dim==2)
                  newEyeDist = DTOEYE(pntpos[0], pntpos[1], pntpos[2]);
                else
                  newEyeDist = DTOPOINT(pntpos[0],pntpos[1],pntpos[2]);
  
                if (currEyeDist < 0.0 || newEyeDist < currEyeDist) {
                  currEyeDist = newEyeDist;
                  tag = currTag;
                  if (unitcell) {
                    unitcell[0] = pbcCells[3*pbcimage  ];
                    unitcell[1] = pbcCells[3*pbcimage+1];
                    unitcell[2] = pbcCells[3*pbcimage+2];
                  }
                }
              }
            }
            break;
        }
      }

      // Pop the instance image transform
      transMat.pop();
    } // end of loop over instance images

    // Pop the PBC image transform
    transMat.pop();
  } // end of loop over PBC images

  // make sure we do not disturb the regular transformation matrix
  transMat.pop();

  // return result; if tag >= 0, we found something
  eyedist = currEyeDist;
  return tag;
}



