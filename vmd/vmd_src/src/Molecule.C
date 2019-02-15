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
 *	$RCSfile: Molecule.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.92 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Main Molecule objects, which contains all the capabilities necessary to
 * store, draw, and manipulate a molecule.  This adds to the functions of
 * DrawMolecule and BaseMolecule by adding routines to read and write various
 * coordinate files.
 *
 * Note: Other capabilities, such as writing structure files, mutations, etc.
 * should go into this level.
 *
 ***************************************************************************/

#include "Molecule.h"
#include "Inform.h"
#include "DisplayDevice.h"
#include "TextEvent.h"
#include "VMDApp.h"
#include "CommandQueue.h"
#include "CoorData.h"

///////////////////////////  constructor  

Molecule::Molecule(const char *src, VMDApp *vmdapp, Displayable *d) 
  : DrawMolecule(vmdapp, d), coorIOFiles(8) {

  char *strPath = NULL, *strName = NULL;
  breakup_filename(src, &strPath, &strName);
  if (!strName) strName = stringdup("unknown");

  // set the molecule name
  moleculename = stringdup(strName);
  delete [] strPath;
  delete [] strName;
}

///////////////////////////  destructor  
Molecule::~Molecule(void) {
  int i;

  // Disconnect this molecule from IMD, if it's active
  app->imd_disconnect(id());

  while(coorIOFiles.num() > 0) {
    delete coorIOFiles[0];
    coorIOFiles.remove(0);
  }

  if (moleculename)
    delete [] moleculename;

  for (i=0; i<fileList.num(); i++) {
    delete [] fileList[i];
  } 
  for (i=0; i<fileSpecList.num(); i++) {
    delete [] fileSpecList[i];
  } 
  for (i=0; i<typeList.num(); i++) {
    delete [] typeList[i];
  } 
  for (i=0; i<dbList.num(); i++) {
    delete [] dbList[i];
  } 
  for (i=0; i<accessionList.num(); i++) {
    delete [] accessionList[i];
  } 
  for (i=0; i<remarksList.num(); i++) {
    delete [] remarksList[i];
  } 
}

///////////////////////////  public routines  

int Molecule::rename(const char *newname) {
  delete [] moleculename;
  moleculename = stringdup(newname);
  return 1;
}

void Molecule::add_coor_file(CoorData *data) {
  coorIOFiles.append(data);
}

void Molecule::close_coor_file(CoorData *data) {
  // remove file from list
  msgInfo << "Finished with coordinate file " << data->name << "." << sendmsg;

  // Must be appended, not run, because callbacks could delete the molecule!
  app->commandQueue->append(
    new TrajectoryReadEvent(id(), data->name)
  );

  delete data;
}

// cancel load/save of coordinate files
// return number canceled
int Molecule::cancel() {
  int retval = 0;
  while(coorIOFiles.num() > 0) {
    msgInfo << "Canceling load/save of file: " << coorIOFiles[0]->name 
            << sendmsg;
    delete coorIOFiles[0];
    coorIOFiles.remove(0);
    retval++;
  }
  return retval;
}

int Molecule::get_new_frames() {
  int newframes = 0;

  // add a new frame if there are frames available in the I/O queue
  if (next_frame())
    newframes = 1; 

  // If an IMD simulation is in progress, store the forces in the current
  // timestep and send them to the simulation.  Otherwise, just toss them.
  if (app->imd_connected(id())) {
    // Add persistent forces to regular forces
    force_indices.appendlist(&persistent_force_indices[0], persistent_force_indices.num());
    force_vectors.appendlist(&persistent_force_vectors[0], persistent_force_vectors.num());

    // Clear old forces out of the timestep
    Timestep *ts = current();
    if (ts && ts->force) {
      memset(ts->force, 0, 3L*nAtoms*sizeof(float));
    }

    // Check for atoms forced last time that didn't show up this time.
    // XXX order N^2 in number of applied forces; could be easily improved.
    // But N now is usually 1 or 2.
    ResizeArray<int> zero_force_indices;
    ResizeArray<float> zero_forces;

    int forcecnt = force_indices.num(); // save in local var for speed

    int ii;
    for(ii=0; ii<last_force_indices.num(); ii++) {
      int j;
      int index_missing=1;
      for (j=0; j<forcecnt; j++) {
        if (force_indices[j]==last_force_indices[ii]) {
          index_missing=0;
          break;
        }
      }
      if (index_missing) {
        // this one didn't show up this time
        zero_force_indices.append(last_force_indices[ii]);
        zero_forces.append3(0, 0, 0);
      }
    }

    if (zero_force_indices.num()) {
      // There some atoms forced last time that didn't show up this time.
      app->imd_sendforces(zero_force_indices.num(), &(zero_force_indices[0]),
                          &(zero_forces[0]));
    }

    // now clear the last forces so we don't send them again
    last_force_indices.clear();

    // Set/send forces if we got any
    if (forcecnt) {
      if (ts) {
        if (!ts->force) {
          ts->force = new float[3L*nAtoms];
          memset(ts->force, 0, 3L*nAtoms*sizeof(float));
        }
        for (int i=0; i<forcecnt; i++) {
          int ind = force_indices[i];
          ts->force[3L*ind  ] += force_vectors[3L*i  ];
          ts->force[3L*ind+1] += force_vectors[3L*i+1];
          ts->force[3L*ind+2] += force_vectors[3L*i+2];
        }
      }
      // XXX If we send multiple forces for the same atom, NAMD will keep only
      // the last force.  We therefore have to sum multiple contributions to the
      // same atom before sending.  Annoying.
      ResizeArray<float> summed_forces;
      int i;
      for (i=0; i<forcecnt; i++) {
        int ind = force_indices[i];
        summed_forces.append3(&ts->force[3L*ind]);
      }
      app->imd_sendforces(forcecnt, &(force_indices[0]), &(summed_forces[0]));

      // save the force indices before clearing them
      last_force_indices.appendlist(&force_indices[0], force_indices.num());

      // now clear the force indices
      force_indices.clear();
      force_vectors.clear();
    }
  }

  // Inform the top level class that background processing
  // is going on to prevent the CPU throttling code from activating
  if (newframes > 0)
    app->background_processing_set();

  return newframes;  
}

int Molecule::next_frame() {
  CoorData::CoorDataState state = CoorData::DONE;

  while (coorIOFiles.num() > 0) {
    // R/W'ing file, do frames until DONE is returned (then close)
    state = coorIOFiles[0]->next(this);

    if (state == CoorData::DONE) {
      close_coor_file(coorIOFiles[0]);
      coorIOFiles.remove(0);
    } else {
      break; // we got a frame, return to caller;
    }
  }

  return (state == CoorData::NOTDONE);
}

// prepare for drawing ... can do one or more of the following:
//  - open a new file and start reading
//  - continue reading an already open file
//  - finish reading a file and close it
//  - always add a new frame if there are frames available
// when done, this then 'prepares' the parent class
void Molecule::prepare() {
  get_new_frames(); // add a new frame if there are frames available
  DrawMolecule::prepare(); // do prepare for parent class
}

void Molecule::addForce(int theatom, const float *f) {
  force_indices.append(theatom);
  force_vectors.append3(&f[0]);
}

void Molecule::addPersistentForce(int theatom, const float *f) {
  // reset the force on this atom if any are stored
  // remove it completely if <f> is (0,0,0)
  
  float mag2 = f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
  int ind = persistent_force_indices.find(theatom);
  if (ind < 0) {
    if (mag2 > 0) {
      // only set if non-zero
      reset_disp_list();
      persistent_force_indices.append(theatom);
      persistent_force_vectors.append3(&f[0]);
    }
  } else {
    if (mag2 > 0) {
      reset_disp_list();
      for (int i=0; i<3; i++) persistent_force_vectors[3L*ind+i] = f[i];
    } else {
      // remove the persistent force (a zero will be sent automatically!)
      persistent_force_indices.remove(ind);

      // remove same index three times since this is a ResizeArray 
      // and it will shift elements down as they are removed...
      persistent_force_vectors.remove(3L*ind);
      persistent_force_vectors.remove(3L*ind);
      persistent_force_vectors.remove(3L*ind);
    }
  }
}
 
