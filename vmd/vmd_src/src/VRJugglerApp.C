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
 *      $RCSfile: VRJugglerApp.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * VRJuggler application initialization code for VMD
 ***************************************************************************/
#include <vrj/vrjConfig.h>

#include <M_VRJapp.h>
#include <VRJugglerScene.h>
#include <VRJugglerRoutines.h>
#include <VMDApp.h>
#include <UIText.h>
#include <Matrix4.h>
//#include <vrj/vrjConfig.h>

#include <iostream>
#include <iomanip>
//#include <GL/gl.h>

#include <vrj/Draw/OGL/GlApp.h>

#include <gadget/Type/PositionInterface.h>
#include <gadget/Type/DigitalInterface.h>

//#include <math.h>
#include <GL/gl.h>
//#include <GL/glu.h>

#include <gmtl/Matrix.h>
#include <gmtl/Generate.h>
#include <gmtl/Vec.h>

#include "Inform.h"

using namespace gmtl;
using namespace vrj;

void M_VRJapp::init()
{
	msgInfo << "---------- M_VRJapp::init() ---------------" << sendmsg;
    // Initialize devices
    mWand.init("VJWand");
    mHead.init("VJHead");
    mButton0.init("VJButton0");
    mButton1.init("VJButton1");
	
    msgInfo << " initialising shared data " << sendmsg;
	vpr::GUID new_guid("44ab594b-1dfb-40c1-8bf3-9af7b7c0ac8a");
    sharedData.init(new_guid);

    msgInfo << " initialising shared data done" << sendmsg;
}

void M_VRJapp::contextInit()
{
    initGLState();       // Initialize the GL state information. (lights, shading, etc)
}

void M_VRJapp::setScene(VRJugglerScene* s)
{
	scene = s;	
}

void M_VRJapp::appendCommand(const char* str)
{
  //msgInfo << "M_VRJapp::appendCommand(const char* str)" << sendmsg;
  cmdQueue.push_back(std::string(str));
  //msgInfo << "queue size now " << (int)cmdQueue.size() << sendmsg;	
}

/**
* Called before start of frame.
*
* @note Function called after device updates but before start of drawing.
*/
void M_VRJapp::preFrame()
{ 

  // add all commands in our queue to the userData so they can be sync'ed
	if (scene){
		if (scene->app){
			if (scene->app->jugglerMode == VRJ_MASTER){
				if(sharedData.isLocal()){
					//msgInfo << "M_VRJapp::preFrame(), I'm local" << sendmsg;
					//msgInfo << "queue size now " << (int) cmdQueue.size() << sendmsg;
					// TODO: make a mutex around cmdQueue, as the main app can write to it in parallel
					// to us reading from it
					for (unsigned i = 0; i < cmdQueue.size(); i++){
					   if(!strupcmp(cmdQueue[i].c_str(), "tool adddevice vrjugglerbuttons 0")) {
					     msgInfo << "M_VRJapp::preFrame(), skipping string " << cmdQueue[i].c_str() << sendmsg;
					     msgInfo << "The slaves should not attach the vrjugglerbuttons to the tool," << sendmsg;
					     msgInfo << "otherwise translating molecules with the tool is done twice on the slaves:" << sendmsg;
					     msgInfo << "first from their own calculations and then as a commmand from the master." << sendmsg;
					   } else {
					      msgInfo << "M_VRJapp::preFrame(), sending string " << cmdQueue[i].c_str() << sendmsg;
					     sharedData->commandStrings.push_back(cmdQueue[i]);
					   }
					} 
				}
			} else {
				//msgInfo << "M_VRJapp::preFrame(), I'm not the master" << sendmsg;
			}
		} else {
			msgErr << "M_VRJapp::draw(), scene->app NULL" << sendmsg;
		}
	} else {
		msgErr << "M_VRJapp::draw(), scene NULL" << sendmsg;
	}
  cmdQueue.clear();
  
  // std::cout << "M_VRJapp::preFrame()" << std::endl;
  if (mButton0->getData())
    {
      //     std::cout << "Button 0 pressed" << std::endl;
    }
  
  if (mButton1->getData())
    {
      //    std::cout << "Button 1 pressed" << std::endl;
    }
  //scene->app->VMD_VRJ_preframeUpdate(); // do the preframe stuff, don't draw yet
  //scene->app->VMDupdate(true);
  //	std::cout << "                      M_VRJapp::preFrame() done" << std::endl;
}

/**
* Called before start of frame, but after preframe.
*/
void M_VRJapp::latePreFrame()
{
  unsigned nrOfCommands = sharedData->commandStrings.size();
  //if (nrOfCommands>0){
    //msgInfo << "latePreFrame(), got " << (long int)nrOfCommands << " commands" << sendmsg;
  //}
  if(!sharedData.isLocal()){
    //msgInfo << "M_VRJapp::latePreFrame(), I'm a slave" << sendmsg;
    for (unsigned i = 0; i< nrOfCommands; i++){
      msgInfo << "latePreFrame(), evaluating " << sharedData->commandStrings[i].c_str() << sendmsg;
      scene->app->uiText->get_interp()->evalString(sharedData->commandStrings[i].c_str());
    }
  } 
}
// Clears the viewport.  Put the call to glClear() in this
// method so that this application will work with configurations
// using two or more viewports per display window.
void M_VRJapp::bufferPreDraw()
{
  //	std::cout << "M_VRJapp::bufferPreDraw()" << std::endl;
   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
   glClear(GL_COLOR_BUFFER_BIT);
}

void M_VRJapp::draw()
{
  //msgInfo << "M_VRJapp::draw()" << sendmsg;
  //msgInfo << "|";
  
  //msgInfo  << "scene->app->jugglerMode" << scene->app->jugglerMode <<  sendmsg;
  //scene->draw();
	if (scene){
		if (scene->app){
			if (scene->app->jugglerMode == VRJ_SLAVE){ 
				//std::cout << "jugglerMode == VRJ_SLAVE" << std::endl;
				vrjuggler_renderer();
			} else {
				// in master mode, don't do anything here
				//msgInfo << "M_VRJapp::draw() on Master, do nothing" << sendmsg;
			}
		} else {
			msgErr << "M_VRJapp::draw(), scene->app NULL" << sendmsg;
		}
	} else {
		msgErr << "M_VRJapp::draw(), scene NULL" << sendmsg;
	}
    //std::cout << "                M_VRJapp::draw() done" << std::endl;
}
/**
 * Called during the Frame.
 * 
 * @note Function called after drawing has been triggered but BEFORE it 
 *       completes.
 */
void M_VRJapp::intraFrame() 
{
  //	std::cout << "M_VRJapp::intraFrame()" << std::endl;	
}

/**
 * Called at end of frame.
 *
 * @note Function called before updating trackers but after the frame is
 *       drawn.
 */
void M_VRJapp::postFrame() 
{
	//  std::cout << "M_VRJapp::postFrame()" << std::endl;	
	//scene->app->VMDupdate(true);
	if (scene){
		if (scene->app){
			if (scene->app->jugglerMode == VRJ_SLAVE){ 
			  scene->app->VRJ_VMDupdate(true);
			}
		} else {
			msgErr << "M_VRJapp::postFrame(), scene->app NULL" << sendmsg;
		}
	} else {
		msgErr << "M_VRJapp::postFrame(), scene NULL" << sendmsg;
	}
}

void M_VRJapp::initGLState()
{
   GLfloat light0_ambient[] = { 0.1f,  0.1f,  0.1f,  1.0f};
   GLfloat light0_diffuse[] = { 0.8f,  0.8f,  0.8f,  1.0f};
   GLfloat light0_specular[] = { 1.0f,  1.0f,  1.0f,  1.0f};
   GLfloat light0_position[] = {0.0f, 0.75f, 0.75f, 0.0f};

   GLfloat mat_ambient[] = { 0.7f, 0.7f,  0.7f, 1.0f };
   GLfloat mat_diffuse[] = { 1.0f,  0.5f,  0.8f, 1.0f };
   GLfloat mat_specular[] = { 1.0,  1.0,  1.0,  1.0};
   GLfloat mat_shininess[] = { 50.0};
//   GLfloat mat_emission[] = { 1.0,  1.0,  1.0,  1.0};
   GLfloat no_mat[] = { 0.0,  0.0,  0.0,  1.0};

   glLightfv(GL_LIGHT0, GL_AMBIENT,  light0_ambient);
   glLightfv(GL_LIGHT0, GL_DIFFUSE,  light0_diffuse);
   glLightfv(GL_LIGHT0, GL_SPECULAR,  light0_specular);
   glLightfv(GL_LIGHT0, GL_POSITION,  light0_position);

   glMaterialfv( GL_FRONT, GL_AMBIENT, mat_ambient );
   glMaterialfv( GL_FRONT,  GL_DIFFUSE, mat_diffuse );
   glMaterialfv( GL_FRONT, GL_SPECULAR, mat_specular );
   glMaterialfv( GL_FRONT,  GL_SHININESS, mat_shininess );
   glMaterialfv( GL_FRONT,  GL_EMISSION, no_mat);

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_NORMALIZE);
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_COLOR_MATERIAL);
   glShadeModel(GL_SMOOTH);
}

void M_VRJapp::getWandXYZ(float& x, float& y, float& z){
 
  float units = 1.0; //getDrawScaleFactor();
  gmtl::Matrix44f wand_matrix = mWand->getData(units); 
  gmtl::Point3f wand_pos = gmtl::makeTrans<gmtl::Point3f>(wand_matrix);
  x= wand_pos[0];
  y= wand_pos[1];
  z= wand_pos[2];
}

// retrieve euler angle rotation of the Wanda
void M_VRJapp::getWandRotMat(Matrix4& rot){
 
  float units = 1.0; //getDrawScaleFactor();
  gmtl::Matrix44f wand_matrix = mWand->getData(units); 
  gmtl::Quatf wand_quat = gmtl::make<gmtl::Quatf>(wand_matrix);
  gmtl::Matrix44f wand_rot_matrix = gmtl::make<gmtl::Matrix44f>(wand_quat); 
  // msgInfo  << "M_VRJapp::getwandRotMat ";
  for (unsigned i = 0; i < 16; i++){
    //rot.mat[i] = wand_rot_matrix[i/4][i%4];
    rot.mat[i] = wand_rot_matrix[i%4][i/4];
    //msgInfo << rot.mat[i]  << " ";
  }
  //msgInfo << sendmsg;
}

// retrieve state of the specified wanda button
bool M_VRJapp::getWandButton(unsigned nr){
  if (nr==0){
    return mButton0->getData();
  } else if (nr==1){
    return mButton1->getData();
  } else {
    msgErr  << "M_VRJapp::getWandButton: unknown button index " << (int)nr << sendmsg;
  }
  return false;
}
