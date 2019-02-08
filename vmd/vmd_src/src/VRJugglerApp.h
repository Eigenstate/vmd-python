/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr VRJuggler patches contributed by Martijn Kragtwijk: m.kragtwijk@rug.nl
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VRJugglerApp.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific Scene subclass for VMD
 ***************************************************************************/
#ifndef M_VRJAPP_H
#define M_VRJAPP_H

//#include <vrj/vrjConfig.h>

//#include <iostream>
//#include <iomanip>
//#include <GL/gl.h>

#include <vrj/Draw/OGL/GlApp.h>
#include "Matrix4.h"

#include <gadget/Type/PositionInterface.h>
#include <gadget/Type/DigitalInterface.h>

#include <plugins/ApplicationDataManager/UserData.h>
#include "VRJugglerSharedData.h"

namespace gadget
{
//	class PositionInterface;
//	class DigitalInterface;
}

class VRJugglerScene;

using namespace vrj;

/**
 * VRjuggler application.
 *
 * This application receives positional
 * and digital intput from the wand.
 *
 */
class M_VRJapp : public GlApp
{
public:
  M_VRJapp():
    cmdQueue()
   {;}

   virtual ~M_VRJapp (void) {;}

public: // ---- INITIALIZATION FUNCTIONS ---- //
   /**
    * Executes any initialization needed before the API is started.
    *
    * @post Device interfaces are initialized with the device names
    *       we want to use.
    * @note This is called once before OpenGL is initialized.
    */
   virtual void init();

   void setScene(VRJugglerScene* s); // pass a pointer to the VMD scene,
									// so we can call its rendering method later

   std::vector<std::string> cmdQueue;    // our local queue of command strings
   void appendCommand(const char* str);  // add the specified string to cmdQueue,
                                         // in preframe this queue is copied to the shared data

   /**
    * Executes any initialization needed after API is started but before the
    * drawManager starts the drawing loops.
    *
    * This is called once after OGL is initialized.
    */
   virtual void apiInit()
   {;}

public:
   // ----- Drawing Loop Functions ------
   //
   //  The drawing loop will look similar to this:
   //
   //  while (drawing)
   //  {
   //        preFrame();
   //       draw();
   //        intraFrame();      // Drawing is happening while here
   //       sync();
   //        postFrame();      // Drawing is now done
   //
   //       UpdateDevices();
   //  }
   //------------------------------------

   /**
    * Function that is called upon entry into a buffer of an OpenGL context
    * (window).
    *
    * @note This function is designed to be used when you want to do something
    *       only once per buffer (ie.once for left buffer, once for right
    *       buffer).
    */
   virtual void bufferPreDraw();

   /**
    * Called before start of frame.
    *
    * @note Function called after device updates but before start of drawing.
    */
   virtual void preFrame();
	
   /**
    * Called before start of frame, after preFrame().
    */
   virtual void latePreFrame();

   /**
    * Called during the Frame.
    *
    * @note Function called after drawing has been triggered but BEFORE it
    *       completes.
    */
   virtual void intraFrame();

   /**
    * Called at end of frame.
    *
    * @note Function called before updating trackers but after the frame is
    *       drawn.
    */
   virtual void postFrame();

public: // ----- OpenGL FUNCTIONS ---- //
   /**
    * Function that is called immediately after a new OGL context is created.
    * Initialize GL state here. Also used to create context specific
    * information.
    *
    * This is called once for each context.
    */
   virtual void contextInit();

   /**
    * Function to draw the scene.
    *
    * @pre OpenGL state has correct transformation and buffer selected.
    * @post The current scene has been drawn.
    *
    * @note Called 1 or more times per frame.
    */
   virtual void draw();

   // old code that draws a box
   //virtual void draw_old();

private:
   void initGLState();

   //void drawCube();

public:
   void getWandXYZ(float& x, float& y, float& z);
   void getWandRotMat(Matrix4& rot);
   bool getWandButton(unsigned nr);
   gadget::PositionInterface  mWand;    /**< Positional interface for Wand position */
   gadget::PositionInterface  mHead;    /**< Positional interface for Head position */
   gadget::DigitalInterface   mButton0; /**< Digital interface for button 0 */
   gadget::DigitalInterface   mButton1; /**< Digital interface for button 1 */

   cluster::UserData<VRJugglerSharedData> sharedData;	// commands are sent to slaves using shared data 

   VRJugglerScene* scene;
};


#endif  // M_VRJAPP_H
