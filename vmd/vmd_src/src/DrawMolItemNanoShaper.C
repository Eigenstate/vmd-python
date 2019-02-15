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
 *	$RCSfile: DrawMolItemNanoShaper.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.7 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Use a NanoShaperInterface object to get a surface triangulation information
 *
 ***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "DrawMolecule.h"
#include "DrawMolItem.h"
#include "NanoShaperInterface.h"
#include "Inform.h"
#include "Scene.h"

void DrawMolItem::draw_nanoshaper(float *framepos, int surftype, int draw_wireframe, float gspacing, float probe_radius, float skin_parm, float blob_parm) {
  int i;
  int ns_ok = 1;

  // regenerate sphere coordinates if necessary
  if (needRegenerate & MOL_REGEN ||
      needRegenerate & SEL_REGEN ||
      needRegenerate & REP_REGEN) {

    nanoshaper.clear();    
    // so I need to recalculate the NanoShaper surface
    float *xyzr = new float[4L*mol->nAtoms];
    int    *ids = new int[mol->nAtoms];
    int   *flgs = NULL; // note: this is NOT ALLOCATED
    const float *aradius = mol->radius();

    // Should I compute the surface of all the atoms?
    int count = 0;

    // compute surface of only the selected atoms
    // get the data for the selected atoms
    float r;
    for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
      if (atomSel->on[i]) {
        xyzr[4L*count+0] = framepos[3L*i+0];
        xyzr[4L*count+1] = framepos[3L*i+1];
        xyzr[4L*count+2] = framepos[3L*i+2];

        r = aradius[i];
        if (r < 0.2f)
          r = 0.2f; // work around an MSMS bug

        xyzr[4L*count+3] = r;
        ids[count] = i;
        count++;
      }
    } // for

    // compute the surface 
    ns_ok = (NanoShaperInterface::COMPUTED == 
                nanoshaper.compute_from_file(surftype, gspacing, probe_radius, skin_parm, blob_parm, count, ids, xyzr, flgs));

    if (!ns_ok)
      msgInfo << "Could not compute NanoShaper surface" << sendmsg;
 
    // do NOT delete flgs as it points to "atomsel->on"
    delete [] ids;
    delete [] xyzr;

    msgInfo << "Done with NanoShaper surface." << sendmsg;
  }

  if (ns_ok && nanoshaper.faces.num() > 0) {
    append(DMATERIALON);

    float *v, *c, *n;
    int ii, ind, fnum, vnum, vsize;

    vnum = nanoshaper.coords.num();
    fnum = nanoshaper.faces.num();
    vsize = vnum * 3;

    v = new float[vsize];
    n = new float[vsize];
    c = new float[vsize];

    for (ii=0; ii<vnum; ii++) {
      ind = ii * 3;
      v[ind    ] = nanoshaper.coords[ii].x[0]; // X
      v[ind + 1] = nanoshaper.coords[ii].x[1]; // Y
      v[ind + 2] = nanoshaper.coords[ii].x[2]; // Z
    }

    for (ii=0; ii<vnum; ii++) {
      ind = ii * 3;
      n[ind    ] = nanoshaper.norms[ii].x[0]; // X
      n[ind + 1] = nanoshaper.norms[ii].x[1]; // Y
      n[ind + 2] = nanoshaper.norms[ii].x[2]; // Z
    }

    for (ii=0; ii<vnum; ii++) {
      ind = ii * 3;
      int col = atomColor->color[nanoshaper.atomids[ii]];
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
        l[ind    ] = nanoshaper.faces[ii].vertex[0]; 
        l[ind + 1] = nanoshaper.faces[ii].vertex[1]; 
        l[ind + 2] = nanoshaper.faces[ii].vertex[1]; 
        l[ind + 3] = nanoshaper.faces[ii].vertex[2]; 
        l[ind + 4] = nanoshaper.faces[ii].vertex[2]; 
        l[ind + 5] = nanoshaper.faces[ii].vertex[0]; 
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
        f[ind    ] = nanoshaper.faces[ii].vertex[0]; 
        f[ind + 1] = nanoshaper.faces[ii].vertex[1]; 
        f[ind + 2] = nanoshaper.faces[ii].vertex[2]; 
  
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



