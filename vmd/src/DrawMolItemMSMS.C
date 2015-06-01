/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: DrawMolItemMSMS.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.48 $	$Date: 2011/06/08 18:41:42 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Use a MSMSInterface object to get a surface triangulation information
 *  There are several rendering options:
 *    1) probe radius
 *    2) surface density
 *    3) "All Atoms" -- should the surface be of the selection or the
 *        contribution of this selection to the surface of all the atoms?
 *        0 == just the selection, 1 == contribution to the all atom surface
 *
 ***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "DrawMolecule.h"
#include "DrawMolItem.h"
#include "MSMSInterface.h"
#include "Inform.h"
#include "Scene.h"

void DrawMolItem::draw_msms(float *framepos, int draw_wireframe, int allatoms, float radius, float density) {
  int i;
  int msms_ok = 1;

  // regenerate sphere coordinates if necessary
  if (needRegenerate & MOL_REGEN ||
      needRegenerate & SEL_REGEN ||
      needRegenerate & REP_REGEN) {

    msms.clear();    
    // so I need to recalculate the MSMS surface
    float *xyzr = new float[4* mol -> nAtoms];
    int   * ids = new int[mol -> nAtoms];
    int   *flgs = NULL; // note: this is NOT ALLOCATED
    const float *aradius = mol->radius();

    // Should I compute the surface of all the atoms?
    int count = 0;
    if (allatoms) {
      // no (surface of only the selected atoms)
      // get the data for the selected atoms
      float r;
      for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
	if (atomSel->on[i]) {
	  xyzr[4*count+0] = framepos[3*i+0];
	  xyzr[4*count+1] = framepos[3*i+1];
	  xyzr[4*count+2] = framepos[3*i+2];

	  r = aradius[i];
	  if (r < 0.2f) 
            r = 0.2f; // work around an MSMS bug

	  xyzr[4*count+3] = r;
	  ids[count] = i;
	  count++;
	}
      } // for
    } else {
      // yes (contribution of selected atoms to the complete surface)
      flgs = atomSel->on; // no translation is needed
      float r;
      for (i=0; i < mol->nAtoms; i++) {
	xyzr[4*count+0] = framepos[3*i+0];
	xyzr[4*count+1] = framepos[3*i+1];
	xyzr[4*count+2] = framepos[3*i+2];

	r = aradius[i];
	if (r < 0.2f) 
          r = 0.2f; // work around an MSMS bug

	xyzr[4*count+3] = r;
	ids[count] = i;
	count++;
      }
    }

    // compute the surface 
#if defined(_MSC_VER)
      msms_ok = (MSMSInterface::COMPUTED == 
                  msms.compute_from_file(radius, density, count, ids, xyzr, flgs));
#else
    if (getenv("VMDMSMSUSEFILE")) {
      msms_ok = (MSMSInterface::COMPUTED == 
                  msms.compute_from_file(radius, density, count, ids, xyzr, flgs));
    } else {
      msms_ok = (MSMSInterface::COMPUTED == 
                  msms.compute_from_socket(radius, density, count, ids, xyzr, flgs));
    }
#endif

    if (!msms_ok)
      msgInfo << "Could not compute MSMS surface" << sendmsg;
 
    // do NOT delete flgs as it points to "atomsel->on"
    delete [] ids;
    delete [] xyzr;

    msgInfo << "Done with MSMS surface." << sendmsg;
  }

  if (msms_ok && msms.faces.num() > 0) {
    append(DMATERIALON);

    float *v, *c, *n;
    int ii, ind, fnum, vnum, vsize;

    vnum = msms.coords.num();
    fnum = msms.faces.num();
    vsize = vnum * 3;

    v = new float[vsize];
    n = new float[vsize];
    c = new float[vsize];

    for (ii=0; ii<vnum; ii++) {
      ind = ii * 3;
      v[ind    ] = msms.coords[ii].x[0]; // X
      v[ind + 1] = msms.coords[ii].x[1]; // Y
      v[ind + 2] = msms.coords[ii].x[2]; // Z
    }

    for (ii=0; ii<vnum; ii++) {
      ind = ii * 3;
      n[ind    ] = msms.norms[ii].x[0]; // X
      n[ind + 1] = msms.norms[ii].x[1]; // Y
      n[ind + 2] = msms.norms[ii].x[2]; // Z
    }

    for (ii=0; ii<vnum; ii++) {
      ind = ii * 3;
      int col = atomColor->color[msms.atomids[ii]];
      const float *fp = scene->color_value(col); 
      c[ind    ] = fp[0]; // Red
      c[ind + 1] = fp[1]; // Green
      c[ind + 2] = fp[2]; // Blue
    }
 
    if (draw_wireframe) {
      int lsize = fnum * 6; 
      int * l = new int[lsize];

      int facecount = 0;
      for (ii=0; ii<fnum; ii++) {
        // draw the face
        ind = facecount * 6;
        l[ind    ] = msms.faces[ii].vertex[0]; 
        l[ind + 1] = msms.faces[ii].vertex[1]; 
        l[ind + 2] = msms.faces[ii].vertex[1]; 
        l[ind + 3] = msms.faces[ii].vertex[2]; 
        l[ind + 4] = msms.faces[ii].vertex[2]; 
        l[ind + 5] = msms.faces[ii].vertex[0]; 
        facecount++;
      }

      // Create a wire mesh
      cmdLineType.putdata(SOLIDLINE, cmdList); // set line drawing parameters
      cmdLineWidth.putdata(1, cmdList);
      cmdWireMesh.putdata(v, n, c, vnum, l, fnum*3, cmdList);
      delete [] l;
    } else {
      int fsize = fnum * 3;
      int * f = new int[fsize];

      int facecount = 0;
      for (ii=0; ii<fnum; ii++) {
        // draw the face
        ind = facecount * 3;
        f[ind    ] = msms.faces[ii].vertex[0]; 
        f[ind + 1] = msms.faces[ii].vertex[1]; 
        f[ind + 2] = msms.faces[ii].vertex[2]; 
  
        facecount++;
      }

      // Check if we're actively animating this rep in colors or in 
      // geometry, and only use ACTC if we're going to draw it more than once
      if (atomColor->method() == AtomColor::THROB) {
        // create a triangle mesh without ACTC stripification
        cmdTriMesh.putdata(v, n, c, vnum, f, fnum, 0, cmdList);
      } else {
        // create a triangle mesh, allowing ACTC to stripify it.
        cmdTriMesh.putdata(v, n, c, vnum, f, fnum, 1, cmdList);
      }

      delete [] f;
    }

    delete [] v;
    delete [] n;
    delete [] c;
  }
}



