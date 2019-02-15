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
 *      $RCSfile: VRJugglerScene.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific Scene subclass for VMD
 ***************************************************************************/

#include <stdlib.h>
#include <vrj/Kernel/Kernel.h>
//#include <vrjuggler_ogl.h>
#include <VRJugglerApp.h>

#include "VRJugglerScene.h"
#include "VRJugglerRoutines.h"
//#include "DisplayDevice.h"
#include "VRJugglerDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "VMDApp.h" // for VMDexit();


////////////////////////////  constructor  
VRJugglerScene::VRJugglerScene(VMDApp *vmdapp) : app(vmdapp) { 
// We don't need to set up a barrier here like in CAVE mode; VRJuggler will take care of this
//
}
void VRJugglerScene::init(/*int argc, char* argv[]*/)
{
   msgInfo << "VRJugglerScene::init" << sendmsg;
   kernel = Kernel::instance();           // Get the kernel
   application = new M_VRJapp();          // Instantiate an instance of the app
   application->setScene(this);

   // IF not args passed to the program
   //    Display usage information and exit
   /*if (argc <= 1)
   {
      std::cout << "\n\n";
      std::cout << "Usage: " << argv[0] << " vjconfigfile[0] vjconfigfile[1] ... vjconfigfile[n]" << std::endl;
      exit(1);
   }

   // Load any config files specified on the command line
   for( int i = 1; i < argc; ++i )
   {
      kernel->loadConfigFile(argv[i]);
   }
*/
   // for now, just load a basic file
   //kernel->loadConfigFile("standalone.jconf");
   msgInfo << "just load a basic file" << sendmsg;
   kernel->loadConfigFile("HPCV.VMD.cave.jconf");
   // Start the kernel running
   msgInfo << "Start the kernel running" << sendmsg;
   kernel->start();
   // Give the kernel an application to execute
   msgInfo << "Give the kernel an application to execute" << sendmsg;
   kernel->setApplication(application);

}

////////////////////////////  destructor  
VRJugglerScene::~VRJugglerScene(void) {
  msgInfo << "~VRJugglerScene()" << sendmsg;
  // free things allocated from shared memory
  //free_to_VRJuggler_memory(draw_barrier);
  if(kernel) kernel->waitForKernelStop(); // stop(); werkt niet goed? 
  if(application)  delete application;
  msgInfo << "~VRJugglerScene() done" << sendmsg;
}

// Called by vrjuggler_renderer
void VRJugglerScene::draw(DisplayDevice *display) {
// Barrier synchronization for all drawing processes and the master
// should now be handled by VRJuggler
//
	if (app->jugglerMode == VRJ_SLAVE){
	  Scene::draw(display);                // draw the scene
	} else {
	  //	msgErr << "VRJugglerScene::draw(), not in VRJ_SLAVE mode!!!" << sendmsg;
	}
}

// Called by vrjuggler_renderer
void VRJugglerScene::draw_finished() {
	if (app->jugglerMode == VRJ_SLAVE){
        Scene::draw_finished();                // draw the scene
	} else {
	  	msgErr << "VRJugglerScene::draw_finished(), not in VRJ_SLAVE mode!!!" << sendmsg;
	}
}

// called in VMDupdate, this updates the values of numDisplayable[23]D
// this is called by the parent!
int VRJugglerScene::prepare() {
  return Scene::prepare(); // call regular scene prepare method
}

void VRJugglerScene::waitForKernelStop(){
	if(kernel){
		kernel->waitForKernelStop();
	} else {
		msgInfo << "VRJugglerScene::waitForKernelStop: kernel NULL" << sendmsg;
	}
}
void VRJugglerScene::appendCommand(const char* str)
{
  //msgInfo << "VRJugglerScene::appendCommand("  << str << ")" << sendmsg;	
  if (app->jugglerMode == VRJ_MASTER){
    if (application){
      application->appendCommand(str);
    } else {
      msgErr  << "VRJugglerScene::appendCommand(): application NULL!" << sendmsg;	
    }
  } else {
    msgErr << "VRJugglerScene::appendCommand(" << str << "), not in VRJ_MASTER mode!" << sendmsg;
  }
}

void VRJugglerScene::getWandXYZ(float& x, float& y, float& z)
{
  if (application){
    return application->getWandXYZ(x,y,z);
  } else {  
    msgErr  << "VRJugglerScene::getWandXYZ(): application NULL!" << sendmsg;	
  }
}


void VRJugglerScene::getWandRotMat(Matrix4& rot)
{
  if (application){
    return application->getWandRotMat(rot);
  } else {  
    msgErr  << "VRJugglerScene::getWandRotMat(): application NULL!" << sendmsg;	
  }
} 
bool VRJugglerScene::getWandButton(unsigned nr)
{
  if (application){
    return application->getWandButton(nr);
  } else {  
    msgErr  << "VRJugglerScene::getWandButton(): application NULL!" << sendmsg;	
  }
} 
