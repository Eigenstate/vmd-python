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
 *	$RCSfile: DrawMolItemRibbons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.149 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Child Displayable component of a molecule; this is responsible for doing
 * the actual drawing of a molecule.  It contains an atom color, atom
 * selection, and atom representation object to specify how this component
 * should look.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <ctype.h>              // for isdigit()

#include "DrawMolItem.h"
#include "DrawMolecule.h"
#include "DispCmds.h"
#include "Inform.h"
#include "Scene.h"
#include "TextEvent.h"
#include "BondSearch.h"
#include "DisplayDevice.h"
#include "WKFUtils.h"

#define RIBBON_ERR_NOTENOUGH            1
#define RIBBON_ERR_PROTEIN_MESS         2
#define RIBBON_ERR_MISSING_PHOSPHATE    4
#define RIBBON_ERR_MISSING_O1P_O2P      8
#define RIBBON_ERR_BADNUCLEIC          16
#define RIBBON_ERR_BADPYRIMIDINE       32
#define RIBBON_ERR_BADPURINE           64

#if 0
#define RIBBON_ERR_NOTENOUGH_STR         "Cannot draw a ribbon for a nucleic acid with only one element"
#define RIBBON_ERR_PROTEIN_MESS_STR      "Someone's been messing around with the definition of a protein!\nThings are going to look messy"
#define RIBBON_ERR_MISSING_PHOSPHATE_STR "Cannot find first phosphate of a nucleic acid, so it won't be drawn"
#define RIBBON_ERR_MISSING_O1P_O2P_STR   "Cannot find both O1P and O2P of a nucleic acid, so it won't be drawn"
#define RIBBON_ERR_BADNUCLEIC_STR        "Someone has redefined the nucleic acid atom names!"
#define RIBBON_ERR_BADPYRIMIDINE_STR     "Tried to draw a nucleic acid residue I thought was a pyrimidine, but it doesn't have the right atom names."
#define RIBBON_ERR_BADPURINE_STR         "Tried to draw a nucleic acid residue I thought was a purine, but it doesn't have the right atom names."
#endif

// draw a spline ribbon.  The math is the same as a tube. 
// However, two splines are calculated, one for coords+perp
// and the other along coords-perp.  Triangles are used to
// connect the curves.  If cylinders are used,  they are
// drawn along the spline paths.  The final result should
// look something like:
//    ooooooooooooooooooo   where the oooo are the (optional) cylinders
//    ! /| /| /| /| /| /!   the ! are a control point, in the ribbon center
//    |/ |/ |/ |/ |/ |/ |   the /| are the triangles  (size per interpolation)
//    ooooooooooooooooooo   the edges go through coords[i] +/- offset[i]
void DrawMolItem::draw_spline_ribbon(int num, float *coords,
        float *offset, int *idx, int use_cyl, float b_rad, int b_res) {
  ResizeArray<float> pickpointcoords;
  ResizeArray<int> pickpointindices;
  float q[4][3];
  // for the next 2 variables, the information for this residue
  // starts at element 2; elements 0 and 1 are copies of the last
  // two elements of the previous residue
  float pts[2][9][3];  // data for 2 edges, 7 points (+2 overlaps from
                       // the previous interpolation) and x,y,z
   
  //  contains the norms as [edge][interpolation point][x/y/z]
  float norms[2][8][3]; 
  // contains the summed norms of the sequential triangles
  // the first element (# 0) contains the info for what should have
  // been the last two triangles of the previous residue.  In other
  // words, tri_norm[0][0] contains the summed norm for the point
  // located at pts[1][0]
  float tri_norm[2][7][3]; 

  int last_loop = -10;
  int loop, i, j;
  float *tmp_ptr = (float *) malloc(2L*(num+4L) * sizeof(float) * 3L);
  if (tmp_ptr == NULL) {
    msgErr << "Cannot make a ribbon; not enough memory!" << sendmsg;
    return;
  }
  float *edge[2];

  // XXX disgusting array math here, rewrite!
  // copy the coordinates +/- the offsets into the temp ("edge") arrays
  edge[0] = tmp_ptr + 2*3L;
  memcpy(edge[0]-2*3L, coords-2*3L, (num+4L)*sizeof(float)*3L);
  edge[1] = edge[0] + (num+4)*3;
  memcpy(edge[1]-2*3L, coords-2*3L, (num+4L)*sizeof(float)*3L);
  for (j=-2*3; j<(num+2)*3L-1; j++) {
    edge[0][j] += offset[j];
    edge[1][j] -= offset[j];
  }

  // go through the data points
  for (loop=-1; loop<num; loop++) {
    int j;

    // If I'm to draw anything....
    if ((idx[loop] >= 0 && atomSel->on[idx[loop]]) ||
        (idx[loop+1] >= 0 && atomSel->on[idx[loop+1]])) {

      // construct the interpolation points (into the "pts" array)
      // remember, these are offset by two to keep some information
      // about the previous residue in the first 2 elements
      for (i=0; i<=1; i++) {
        make_spline_Q_matrix(q, spline_basis, edge[i]+(loop-1)*3);
        make_spline_interpolation(pts[i][2], 0.0f/6.0f, q);
        make_spline_interpolation(pts[i][3], 1.0f/6.0f, q);
        make_spline_interpolation(pts[i][4], 2.0f/6.0f, q);
        make_spline_interpolation(pts[i][5], 3.0f/6.0f, q);
        make_spline_interpolation(pts[i][6], 4.0f/6.0f, q);
        make_spline_interpolation(pts[i][7], 5.0f/6.0f, q);
        make_spline_interpolation(pts[i][8], 6.0f/6.0f, q);
      }

      // make the normals for each new point.
      for (i=2; i<8; i++) {
        float diff1[3], diff2[3];
        vec_sub(diff1, pts[1][i+1], pts[1][i+0]);
        vec_sub(diff2, pts[1][i+1], pts[0][i+0]);
        cross_prod(norms[1][i], diff1, diff2);
        vec_sub(diff1, pts[0][i+1], pts[0][i+0]);
        cross_prod(norms[0][i], diff1, diff2);
      }

      // if this wasn't a continuation, I need to set the
      // first 2 elements properly so the norms are smooth
      if (last_loop != loop-1) {
        vec_copy(norms[0][0], norms[0][2]);
        vec_copy(norms[1][0], norms[1][2]);
        vec_copy(norms[0][1], norms[0][2]);
        vec_copy(norms[1][1], norms[1][2]);
      }

      // sum up the values of neighboring triangles to make
      // a smooth normal
      for (j=0; j<=1; j++) {
        for (i=0; i<8-1; i++) {
          vec_add(tri_norm[j][i], norms[j][i+1],     // "this" triangle
                  norms[1-j][i+1]);  // opposite strand
          vec_add(tri_norm[j][i], tri_norm[j][i],
                  norms[j][i]);      // prev on strand
        }
      }

      // pre-normalize the normals so we don't need to have
      // OpenGL doing this for us repetitively on every frame,
      // this allows use to use the GL_RESCALE_NORMAL extension
      for (j=0; j<=1; j++) {
        for (i=0; i<8-1; i++) {
          vec_normalize(tri_norm[j][i]);
        } 
      }
	 
      // draw what I need for atom 'loop'
      if (idx[loop] >= 0 &&          // this is a real atom
          atomSel->on[idx[loop]]) {  // and it it turned on

        if (last_loop != loop - 1) {  
          // do prev. points exist? if not then I don't know if the color 
          // was properly set, so set it here
          cmdColorIndex.putdata(atomColor->color[idx[loop]], cmdList);
        }

        // draw the cylinders to finish off the last residue, if
        // need be, and draw the ones for this half of the residue
        // Cylinders are drawn on the top and bottom of the residues
        if (use_cyl) { // draw top/bot edge cylinders if need be
          // Special case the first cylinder because I want
          // it to be a smooth continuation from the previous
          // cylinder; assuming it exists
          if (last_loop != loop-1) {  // continue from previous?
            int ii;
            for (ii=0; ii<=1; ii++) {  // doesn't exist, so
              make_connection(NULL, pts[ii][2], pts[ii][3],
                              pts[ii][4], b_rad, b_res, use_cyl);
            }
          } else { // there was a previous cylinder, so be smooth
            for (i=0; i<=1; i++) {
	       make_connection(pts[i][0], pts[i][1], pts[i][2], pts[i][3],
                               b_rad, b_res, use_cyl);
               make_connection(pts[i][1], pts[i][2], pts[i][3], pts[i][4],
                               b_rad, b_res, use_cyl);
            }
          }
	       
          // and draw the rest of the cylinders for this 1/2 residue
          for (i=0; i<=1; i++) {
            make_connection(pts[i][2], pts[i][3], pts[i][4], pts[i][5],
                            b_rad, b_res, use_cyl);
            make_connection(pts[i][3], pts[i][4], pts[i][5], pts[i][6],
                            b_rad, b_res, use_cyl);
          }
        } // drew cylinders

        // Draw the triangles that connect the cylinders and make up the 
        // ribbon proper.  The funky start condition is so that it starts at
        // pts[][1] if I need to finish the previous residue, or
        // pts[][2] if I don't
        for (i= (last_loop == loop-1 ? 1: 2); i<5; i++) {
          cmdTriangle.putdata(pts[1][1+i], pts[1][0+i], pts[0][0+i],
                              tri_norm[1][0+i], tri_norm[1][-1+i],
                              tri_norm[0][-1+i], cmdList);
          cmdTriangle.putdata(pts[0][1+i], pts[1][1+i], pts[0][0+i],
                              tri_norm[0][0+i], tri_norm[1][0+i],
                              tri_norm[0][-1+i], cmdList);

          // indicate this atom can be picked
          // this spot is in the middle of the ribbon, both in length
          // and in width
          int pidx = loop * 3;
          pickpointcoords.append3(&coords[pidx]);
          pickpointindices.append(idx[loop]);
        }
      }  // atom 'loop' finished
	 
      // draw what I can for atom 'loop+1'
      if (idx[loop+1] >= 0 && atomSel->on[idx[loop+1]]) {
        // If this is on, then I may have to change the color,
        // since I'm lazy, I won't check to see if I _have_ to change it
        // but assume I do
        cmdColorIndex.putdata(atomColor->color[idx[loop+1]], cmdList);
        // I can draw the first couple of cylinders, but I need knowledge
        // of what goes on next to get a smooth fit.  Thus, I won't
        // draw them here.
        // I can't draw the last two cylinders.
        if (use_cyl) {
          for (i=0; i<=1; i++) {
            make_connection(pts[i][4], pts[i][5], pts[i][6], pts[i][7],
                            b_rad, b_res, use_cyl);
            make_connection(pts[i][5], pts[i][6], pts[i][7], pts[i][8],
                            b_rad, b_res, use_cyl);
          }
        }

        // I can draw 3 of the four triangles, but I need
        // the normals to do the last one properly
        for (i= 5; i<8-1; i++) {
/*
          cmdTriangle.putdata(pts[0][0+i], pts[1][0+i], pts[1][1+i],
                              tri_norm[0][-1+i], tri_norm[1][-1+i],
                              tri_norm[1][0+i], cmdList);
          cmdTriangle.putdata(pts[0][0+i], pts[1][1+i], pts[0][1+i],
                              tri_norm[0][-1+i], tri_norm[1][0+i],
                              tri_norm[0][0+i], cmdList);
*/
          cmdTriangle.putdata(pts[1][1+i], pts[1][0+i], pts[0][0+i],
                              tri_norm[1][0+i], tri_norm[1][-1+i],
                              tri_norm[0][-1+i], cmdList);
          cmdTriangle.putdata(pts[0][1+i], pts[1][1+i], pts[0][0+i],
                              tri_norm[0][0+i], tri_norm[1][0+i],
                              tri_norm[0][-1+i], cmdList);
        }
        last_loop = loop;
      } // atom 'loop+1' finished
	 
      // save infor for next loop, for smoothing cylinders and normals
      for (i=0; i<=1; i++) {
        vec_copy(pts[i][0], pts[i][6]);  // remember, because of the spline,
        vec_copy(pts[i][1], pts[i][7]);  // loop pts[][8] is loop+1 pts[][2]
        vec_copy(norms[i][0], norms[i][6]);
        vec_copy(norms[i][1], norms[i][7]);
      }
    } /// else nothing to draw
  } // gone down the fragment

  free(tmp_ptr);

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0],
                           &pickpointcoords[0], cmdList);
  }
}




// draw ribbons along the protein backbone
// part of this method taken (with permission) from the 'ribbon.f' in the
// Raster3D package:
//  Merritt, Ethan A. and Murphy, Michael E.P. (1994).
//   "Raster3D Version 2.0, a Program for Photorealistic Molecular Graphics"
//        Acta Cryst. D50, 869-873.

// That method was based on ideas from  Carson & Bugg, J. Molec. Graphics
//   4,121-122 (1986)
void DrawMolItem::draw_ribbons(float *framepos, float brad, int bres, float thickness) {
  sprintf (commentBuffer,"MoleculeID: %d ReprID: %d Beginning Ribbons",
           mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  // find out if I'm using lines or cylinders
  int use_cyl = TRUE; // use cylinders by default
  int rc = 0;  // clear errors
  
  if (bres <= 2 || brad < 0.01f) { 
    use_cyl = FALSE; // the representation will be as lines
  } 

  append(DMATERIALON); // enable shading 

  float ribbon_width = thickness;
  if (ribbon_width < 1.0) 
    ribbon_width = 1.0;
  ribbon_width /= 7.0;

  rc |= draw_protein_ribbons_old(framepos, bres, brad, ribbon_width, use_cyl);
  rc |= draw_nucleic_ribbons(framepos, bres, brad, ribbon_width, use_cyl, 0, 0);
  rc |= draw_base_sugar_rings(framepos, bres, brad, ribbon_width, use_cyl);

  // XXX put the more specific error messages back in here if we feel
  //     that they are helpful to the user.
  if (rc != 0) {
    if (emitstructwarning())
      msgErr << "Warning: ribbons code encountered an unusual structure, geometry may not look as expected." << sendmsg;
  }
}




int DrawMolItem::draw_protein_ribbons_old(float *framepos, int b_res, float b_rad,
                                float ribbon_width, int use_cyl) {
  float *real_coords = NULL, *coords;
  float *real_perps = NULL, *perps;
  int *real_idx = NULL, *idx;
  float *capos, *last_capos;
  float *opos,  *last_opos;
  int onum, canum, res, frag, num;
  int rc=0;

  // these are the variables used in the Raster3D package
  float a[3], b[3], c[3], d[3], e[3], g[3];  

  // go through each protein and find the CA and O
  // from that, construct the offsets to use (called "perps")
  // then do two splines and connect them
  for (frag=0; frag<mol->pfragList.num(); frag++) {
    int loop;

    num = mol->pfragList[frag]->num();  // number of residues in this fragment
    if (num < 2) {
      rc |= RIBBON_ERR_NOTENOUGH;
      continue; // can't draw a ribbon with only one element, so skip
    }

    // check that we have a valid structure before continuing
      res = (*mol->pfragList[frag])[0];
    canum = mol->find_atom_in_residue("CA", res); // get CA 
     onum = mol->find_atom_in_residue("O", res);  // get O

    if (canum < 0 || onum < 0) {
      continue; // can't find 1st CA or O of the protein, so don't draw
    }

    if (real_coords) {
      free(real_coords);
      free(real_idx);
      free(real_perps);
      real_coords = NULL;
      real_idx = NULL;
      real_perps = NULL;
    }

    // allocate memory for control points, perps, and index storage
    // The arrays have 2 extra elements at the front and end to allow
    // duplication of control points necessary for the old ribbons code.
    real_coords = (float *) malloc((num+4) * sizeof(float)*3);
       real_idx =   (int *) malloc((num+4) * sizeof(int));
     real_perps = (float *) malloc((num+4) * sizeof(float)*3);

    coords = real_coords + 2*3L;
       idx = real_idx + 2L;
     perps = real_perps + 2*3L;

    // initialize CA and O atom pointers
    capos = framepos + 3L*canum;
    last_capos = capos;
    opos = framepos + 3L*onum;
    last_opos = opos;

    // duplicate first control point data
    vec_copy(coords-6, capos);
    vec_copy(coords-3, capos);
    idx[-2] = idx[-1] = -1;

    // now go through and set the coordinates and the perps
    e[0] = e[1] = e[2] = 0.0;
    vec_copy(g, e);
    for (loop=0; loop<num; loop++) {
      res = (*mol->pfragList[frag])[loop];
      canum = mol->find_atom_in_residue("CA", res);
      if (canum >= 0) {
        capos = framepos + 3L*canum;
      }
      onum = mol->find_atom_in_residue("O", res);
      if (onum < 0) {
        onum = mol->find_atom_in_residue("OT1", res);
      }

      if (onum >= 0) {
        opos = framepos + 3L*onum;
      } else {
        rc |= RIBBON_ERR_PROTEIN_MESS;
        opos = last_opos; // try to cope if we have no oxygen index
      }

      vec_copy(coords+loop*3, capos);
      idx[loop] = canum;

      // now I need to figure out where the ribbon goes
      vec_sub(a, capos, last_capos);     // A=(pos(CA(res)) - pos(CA(res-1)))
      vec_sub(b, last_opos, last_capos); // B=(pos(O(res-1)) - pos(CA(res-1)))
      cross_prod(c, a, b);               // C=AxB, define res-1's peptide plane
      cross_prod(d, c, a);               // D=CxA, normal to plane and backbone
      // if normal is > 90 deg from  previous one, negate the new one
      if (dot_prod(d, g) < 0) {
        vec_negate(b, d);
      } else {
        vec_copy(b, d);
      }
      vec_add(e, g, b);   // average to the sum of the previous vectors
      vec_normalize(e);
      vec_scale(&perps[3L*loop], ribbon_width, e); // compute perps from normal
      vec_copy(g, e);     // make a cumulative sum; cute, IMHO
      last_capos = capos;
      last_opos = opos;
    }

    // duplicate the last control point into two extra copies
    vec_copy(coords+3L*num, capos);
    vec_copy(coords+3L*(num+1), capos);
    idx[num] = idx[num+1] = -1;

    // copy the second perp to the first since the first one didn't have
    // enough data to construct a good perpendicular.
    vec_copy(perps, perps+3);

    // duplicate the first and last perps into the extra control points
    vec_copy(perps-3, perps);
    vec_copy(perps-6, perps);
    vec_copy(perps+3L*num, perps+3L*num-3);
    vec_copy(perps+3L*num+3, perps+3L*num);

    draw_spline_ribbon(num, coords, perps, idx, use_cyl, b_rad, b_res);

    if (real_coords) {
      free(real_coords);
      free(real_idx);
      free(real_perps);
      real_coords = NULL;
      real_idx = NULL;
      real_perps = NULL;
    }
  } // drew the protein fragment ribbon

  return rc;
}


int DrawMolItem::draw_protein_ribbons_new(float *framepos, int b_res, float b_rad,
                                float ribbon_width, int use_cyl) {
  float *coords = NULL;
  float *perps = NULL;
  int *idx = NULL;
  float *capos, *last_capos, *opos, *last_opos;
  int onum, canum, frag, num, res;
  const char *modulatefield = NULL; // XXX this needs to become a parameter
  const float *modulatedata = NULL; // data field to use for width modulation
  float *modulate = NULL;           // per-control point width values
  int rc=0;

  // these are the variables used in the Raster3D package
  float a[3], b[3], c[3], d[3], e[3], g[3];  

  // Lookup atom typecodes ahead of time so we can use the most 
  // efficient variant of find_atom_in_residue() in the main loop.
  // If we can't find the atom types we need, bail out immediately
  int CAtypecode  = mol->atomNames.typecode("CA");
  int Otypecode   = mol->atomNames.typecode("O");
  int OT1typecode = mol->atomNames.typecode("OT1");

  if (CAtypecode < 0 || ((Otypecode < 0) && (OT1typecode < 0))) {
    return rc; // can't draw a ribbon without CA and O atoms for guidance
  }

  // allocate for the maximum possible per-residue control points, perps, 
  // and indices so we don't have to reallocate for every fragment
  coords = (float *) malloc((mol->nResidues) * sizeof(float)*3);
     idx =   (int *) malloc((mol->nResidues) * sizeof(int));
   perps = (float *) malloc((mol->nResidues) * sizeof(float)*3);

#if 1
  // XXX hack to let users try various stuff
  modulatefield = getenv("VMDMODULATERIBBON");
#endif
  if (modulatefield != NULL) {
    if (!strcmp(modulatefield, "user")) {
      modulatedata = mol->current()->user;
      // XXX user field can be NULL on some timesteps
    } else {
      modulatedata = mol->extraflt.data(modulatefield);
    } 

    // allocate for the maximum possible per-residue modulation values
    // so we don't have to reallocate for every fragment
    // XXX the modulate array is allocated cleared to zeros so that
    // in the case we get a NULL modulatedata pointer (user field for
    // example, we'll just end up with no modulation.
    modulate = (float *) calloc(1, mol->nResidues * sizeof(float));
  }

  // go through each protein and find the CA and O
  // from that, construct the offsets to use (called "perps")
  for (frag=0; frag<mol->pfragList.num(); frag++) {
    int cyclic=mol->pfragCyclic[frag];  // whether fragment is cyclic
    num = mol->pfragList[frag]->num();  // number of residues in this fragment
    if (num < 2) {
      rc |= RIBBON_ERR_NOTENOUGH;
      continue; // can't draw a ribbon with only one element, so skip
    }

    // check that we have a valid structure before continuing
      res = (*mol->pfragList[frag])[0];
    canum = mol->find_atom_in_residue(CAtypecode, res);
     onum = mol->find_atom_in_residue(Otypecode, res);

    if (canum < 0 || onum < 0) {
      continue; // can't find 1st CA or O of the protein, so don't draw
    }

    // initialize CA and O atom pointers
    capos = framepos + 3L*canum;
    last_capos = capos;
    opos = framepos + 3L*onum;
    last_opos = opos;

    // now go through and set the coordinates and the perps
    e[0] = e[1] = e[2] = 0.0;
    vec_copy(g, e);

    // for a cyclic structure, we use the positions of the last residue
    // to seed the initial direction vectors for the ribbon
    if (cyclic) {
      int lastres = (*mol->pfragList[frag])[num-1];
      int lastcanum = mol->find_atom_in_residue(CAtypecode, lastres);
      last_capos = framepos + 3L*lastcanum;

      int lastonum = mol->find_atom_in_residue(Otypecode, lastres);
      if (lastonum < 0 && OT1typecode >= 0) {
        lastonum = mol->find_atom_in_residue(OT1typecode, lastres);
      }
      last_opos = framepos + 3L*lastonum;

      // now I need to figure out where the ribbon goes
      vec_sub(a, capos, last_capos);     // A=(pos(CA(res)) - pos(CA(res-1)))
      vec_sub(b, last_opos, last_capos); // B=(pos(O(res-1)) - pos(CA(res-1)))
      cross_prod(c, a, b);               // C=AxB, define res-1's peptide plane
      cross_prod(d, c, a);               // D=CxA, normal to plane and backbone
      // if normal is > 90 deg from  previous one, negate the new one
      if (dot_prod(d, g) < 0) {
        vec_negate(b, d);
      } else {
        vec_copy(b, d);
      }
      vec_add(e, g, b);            // average to the sum of the previous vectors
      vec_normalize(e);
      vec_copy(g, e);              // make a cumulative sum; cute, IMHO
    }

    int loop;
    for (loop=0; loop<num; loop++) {
      res = (*mol->pfragList[frag])[loop];

      canum = mol->find_atom_in_residue(CAtypecode, res);
      if (canum >= 0) {
        capos = framepos + 3L*canum;
      }

      onum = mol->find_atom_in_residue(Otypecode, res);
      if (onum < 0 && OT1typecode >= 0) {
        onum = mol->find_atom_in_residue(OT1typecode, res);
      }
      if (onum >= 0) {
        opos = framepos + 3L*onum;
      } else {
        rc |= RIBBON_ERR_PROTEIN_MESS;
        opos = last_opos; // try to cope if we have no oxygen index
      }

      // copy the CA coordinate into the control point array
      vec_copy(coords+loop*3L, capos);

      // modulate the ribbon width by user-specified per-atom data
      if (modulatedata != NULL)
        modulate[loop] = modulatedata[canum];
 
      idx[loop] = canum;

      // now I need to figure out where the ribbon goes
      vec_sub(a, capos, last_capos);     // A=(pos(CA(res)) - pos(CA(res-1)))
      vec_sub(b, last_opos, last_capos); // B=(pos(O(res-1)) - pos(CA(res-1)))
      cross_prod(c, a, b);               // C=AxB, define res-1's peptide plane
      cross_prod(d, c, a);               // D=CxA, normal to plane and backbone
      // if normal is > 90 deg from  previous one, negate the new one
      if (dot_prod(d, g) < 0) {
        vec_negate(b, d);
      } else {
        vec_copy(b, d);
      }
      vec_add(e, g, b);            // average to the sum of the previous vectors
      vec_normalize(e);
      vec_copy(&perps[3L*loop], e); // compute perps from the normal
      vec_copy(g, e);              // make a cumulative sum; cute, IMHO
      last_capos = capos;
      last_opos = opos;
    }

    if (!cyclic) {
      // copy the second perp to the first since the first one didn't have
      // enough data to construct a good perpendicular.
      vec_copy(perps, perps+3);
    } 

    if (modulate != NULL) {
      // modulate ribbon width by user-specified per-atom field value
      float *widths = (float *) malloc(num * sizeof(float));
      float *heights = (float *) malloc(num * sizeof(float));
      float m_fac;
      int i;
      for (i=0; i<num; i++) {
#if 1
        // only allow modulation values > 0. otherwise fall back on old scheme.
        m_fac = modulate[i];
        if (m_fac <= 0.0f)
           m_fac = 1.0f;

        // modulate both width and height
        widths[i] = 7L * ribbon_width * b_rad * m_fac;
        heights[i] = b_rad*m_fac;
#else
        // modulate only width, and only additively
        widths[i] = 7L * ribbon_width * b_rad + m_fac;
        heights[i] = b_rad;
#endif
      }
      draw_spline_new(num, coords, perps, idx, widths, heights, num, b_res, cyclic);
      free(widths);
      free(heights);
    } else {
      // draw normal unmodulated ribbons
      float widths = 7L * ribbon_width * b_rad;
      float heights = b_rad;
      draw_spline_new(num, coords, perps, idx, &widths, &heights, 1, b_res, cyclic);
    }
  } // drew the protein fragment ribbon

  if (coords) {
    free(coords);
    free(idx);
    free(perps);
  }

  if (modulate) {
    free(modulate);
  }

  return rc;
}



int DrawMolItem::draw_nucleic_ribbons(float *framepos, int b_res, float b_rad,
                                float ribbon_width, int use_cyl, int use_new, 
                                int use_carb) {
  ///////////// now draw the nucleic acid ribbon
  float *real_coords = NULL, *coords;
  float *real_perps = NULL, *perps;
  int *real_idx = NULL, *idx;
  float *ppos, *last_ppos;
  float opos[3], last_opos[3];
  int cpnum;         // index of spline control point atom
  int o1num, o2num;  // indices of atoms used to construct perps
  int frag, num;
  int rc=0;

  // these are the variables used in the Raster3D package
  float a[3], b[3], c[3], d[3], e[3], g[3];  

  // go through each nucleic acid and find the phospate
  // then find the O1P/O2P, or OP1/OP2 (new nomenclature). 
  // From those construct the perps then do two splines and connect them
  for (frag=0; frag<mol->nfragList.num(); frag++) {
    int loop;

    num = mol->nfragList[frag]->num(); // number of residues in this fragment
    if (num < 2) {
      rc |= RIBBON_ERR_NOTENOUGH;
      continue;
    }
    if (real_coords) {
      free(real_coords);
      free(real_idx);
      free(real_perps);
      real_coords = NULL;
      real_idx = NULL;
      real_perps = NULL;
    }
    real_coords = (float *) malloc((num+4)*sizeof(float)*3);
    real_idx = (int *) malloc((num+4) * sizeof(int));
    real_perps = (float *) malloc((num+4) * sizeof(float)*3);
    coords = real_coords + 2*3L;
       idx = real_idx + 2L;
     perps = real_perps + 2*3L;
	 
    // okay, I've got space for the coordinates, the index translations,
    // and the perpendiculars, now initialize everything
    int res = (*mol->nfragList[frag])[0];

    // get atoms to use as the spline control points
    if (use_carb) {
      // use the carbons
      cpnum = mol->find_atom_in_residue("C3'", res);
      if (cpnum < 0)
        cpnum = mol->find_atom_in_residue("C3*", res);
      // use the phosphates if no carbon control points found
      if (cpnum < 0)
        cpnum = mol->find_atom_in_residue("P", res);
    } else { 
      // use the phosphates 
      cpnum = mol->find_atom_in_residue("P", res);
    }

    // if no P found, check the terminal atom names
    if (cpnum < 0) {
      cpnum = mol->find_atom_in_residue("H5T", res);
    }
    if (cpnum < 0) {
      cpnum = mol->find_atom_in_residue("H3T", res);
    }

    if (cpnum < 0) {
      rc |= RIBBON_ERR_MISSING_PHOSPHATE;
      continue;
    }
   
    o1num = mol->find_atom_in_residue("O1P", res);   //  and an oxygen
    if (o1num < 0)
      o1num = mol->find_atom_in_residue("OP1", res); //  and an oxygen

    o2num = mol->find_atom_in_residue("O2P", res);   //  and an oxygen
    if (o2num < 0)
      o2num = mol->find_atom_in_residue("OP2", res); //  and an oxygen

    // if we failed to find these on the terminal residue, try the next one..
    if (o1num  < 0 || o2num < 0) {
      int nextres = (*mol->nfragList[frag])[1];
      o1num = mol->find_atom_in_residue("O1P", nextres);   //  and an oxygen
      if (o1num < 0) 
        o1num = mol->find_atom_in_residue("OP1", nextres); //  and an oxygen

      o2num = mol->find_atom_in_residue("O2P", nextres);   //  and an oxygen
      if (o2num < 0)
        o2num = mol->find_atom_in_residue("OP2", nextres); //  and an oxygen

      if (o1num  < 0 || o2num < 0) {
        rc |= RIBBON_ERR_MISSING_O1P_O2P;
        continue;
      }
    }

    ppos = framepos + 3L*cpnum;
    vec_add(opos, framepos + 3L*o1num, framepos + 3L*o2num);  // along the bisector
    vec_copy(last_opos, opos);
    last_ppos = ppos;

    vec_copy(coords-6, ppos);
    vec_copy(coords-3, ppos);
    idx[-2] = idx[-1] = -1;

    // now go through and set the coordinates and the perps
    e[0] = e[1] = e[2] = 0.0;
    vec_copy(g, e);
    int abortfrag=0;
    for (loop=0; (loop<num && abortfrag==0); loop++) {
      res = (*mol->nfragList[frag])[loop];

      // get atoms to use as the spline control points
      if (use_carb) {
        // use the carbons
        cpnum = mol->find_atom_in_residue("C3'", res);
        if (cpnum < 0)
          cpnum = mol->find_atom_in_residue("C3*", res);
        // use the phosphates if no carbon control points found
        if (cpnum < 0)
          cpnum = mol->find_atom_in_residue("P", res);
      } else { 
        // use the phosphates 
        cpnum = mol->find_atom_in_residue("P", res);
      }

      // if no P found, check the terminal atom names
      if (cpnum < 0) {
        cpnum = mol->find_atom_in_residue("H5T", res);
      }
      if (cpnum < 0) {
        cpnum = mol->find_atom_in_residue("H3T", res);
      }

      // cpnum must be set to a valid atom or we'll crash
      if (cpnum >= 0) {
        ppos = framepos + 3L*cpnum;
        idx[loop] = cpnum;
      } else {
        rc |= RIBBON_ERR_MISSING_PHOSPHATE;
        abortfrag = 1;
        break;
      }
        
      o1num = mol->find_atom_in_residue("O1P", res);   //  and an oxygen
      if (o1num < 0)
        o1num = mol->find_atom_in_residue("OP1", res); //  and an oxygen

      o2num = mol->find_atom_in_residue("O2P", res);   //  and an oxygen
      if (o2num < 0)
        o2num = mol->find_atom_in_residue("OP2", res); //  and an oxygen

      if (o1num < 0 || o2num < 0) {
        rc |= RIBBON_ERR_PROTEIN_MESS;
        vec_copy(opos, last_opos);
      } else {
        float tmp[3];
        vec_sub(tmp, framepos + 3L*o1num, ppos);
        vec_sub(opos, framepos + 3L*o2num, ppos);
        vec_add(opos, tmp, opos);  // along the bisector
      }
      vec_copy(coords+loop*3, ppos);
 
      // now I need to figure out where the ribbon goes
      vec_sub(a, ppos, last_ppos);      // A=(pos(P(res)) - pos(P(res-1)))
//    vec_sub(b, last_opos, last_ppos); // B=(pos(Obisector(res-1)) - pos(P(res-1)))
//    cross_prod(c, a, b); // C=AxB defines res-1's peptide plane
      vec_copy(c, opos);   // already have the normal to the ribbon
      cross_prod(d, c, a); // D=CxA normal to plane and backbone

      // if normal is > 90 deg from  previous one, invert the new one
      if (dot_prod(d, g) < 0) { 
        vec_negate(b, d);
      } else {
        vec_copy(b, d);
      }
      vec_add(e, g, b);   // average to the sum of the previous vectors
      vec_normalize(e);
      vec_scale(&perps[3L*loop], ribbon_width, e); // compute perps from normal
      vec_copy(g, e);     // make a cumulative sum; cute, IMHO
      last_ppos = ppos;
      vec_copy(last_opos, opos);
    }

    // abort drawing the entire fragment if unfixable problems occur
    if (abortfrag)
      continue;

    // and set the final points to the last element
    vec_copy(coords+3L*num, ppos);
    vec_copy(coords+3L*(num+1), ppos);
    idx[num] = idx[num+1] = -1;

    // copy the second perp to the first since the first one didn't have
    // enough data to construct a good perpendicular.
    vec_copy(perps, perps+3);

    // now set the first and last perps correctly
    vec_copy(perps-3, perps);
    vec_copy(perps-6, perps);
    vec_copy(perps+3L*num, perps+3L*num-3);
    vec_copy(perps+3L*num+3, perps+3L*num);

    // draw the nucleic acid fragment ribbon
    if (use_new) {
      float widths = 7L * ribbon_width * b_rad;
      float heights = b_rad; 
      draw_spline_new(num, coords, perps, idx, &widths, &heights, 1, b_res, 0);
    } else {
      draw_spline_ribbon(num, coords, perps, idx, use_cyl, b_rad, b_res);
    }
  }

  if (real_coords) {
    free(real_coords);
    free(real_idx);
    free(real_perps);
    real_coords = NULL;
    real_idx = NULL;
    real_perps = NULL;
  }

  return rc;
}




int DrawMolItem::draw_base_sugar_rings(float *framepos, int b_res, float b_rad,
                              float ribbon_width, int use_cyl) {
  float *real_coords = NULL;
  int frag, num;
  int rc=0;

  ///////////// now draw the rings of the base and sugar as planes 
  // ribose 
  //   O4',C1',C2',C3'C4' or
  //   O4*,C1*,C2*,C3*C4*
  // purines (CYT,THY,URA) 
  // 	N1,C2,N3,C4,C5,C6
  // pyrimidines (ADE,GUA) 
  //      N1,C2,N3,C4,C5,C6,N7,N8,N9
  // O4',C1' and C4' are define the ribose plane and 
  // C2' and C3' then define the pucker of the ring
  // sugar -- base bonds
  //  	pyrimidine 	C1' to N9 
  // 	purine 		C1' to N1  
  float *o4ppos=NULL, *c1ppos=NULL, *c2ppos=NULL; 
  float *c3ppos=NULL, *c4ppos=NULL;
  int o4pnum, c1pnum, c2pnum, c3pnum, c4pnum;
  float *n1pos,*c2pos,*n3pos,*c4pos,*c5pos,*c6pos,*n7pos,*c8pos,*n9pos;
  int n1num,c2num,n3num,c4num,c5num,c6num,n7num,c8num,n9num;
  float rescentra[3], rescentrb[3];
  float midptc1pc4p[3];
      
  for (frag=0; frag<mol->nfragList.num(); frag++) {
    int loop;

    num = mol->nfragList[frag]->num(); // num of residues in this fragment
    if (real_coords) {
      free(real_coords);
      real_coords = NULL;
    }

    // 5atoms for the ribose but only 4 triangles
    // 9atoms max for a base
    real_coords = (float *) malloc( 14L * num * sizeof(float)*3L);
	 
    // okay, I've got space for the coordinates now go
    for (loop=0; loop<num; loop++) {
      // the furanose
      int res = (*mol->nfragList[frag])[loop];

      c1pnum = mol->find_atom_in_residue("C1'", res);
      if (c1pnum < 0) {
        c1pnum = mol->find_atom_in_residue("C1*", res);
        if (c1pnum < 0) {
          rc |= RIBBON_ERR_BADNUCLEIC;
          continue;
        }
      } 
      c1ppos = framepos + 3L*c1pnum;

      if (atomSel->on[c1pnum]) { // switch drawing by C1' atom
        o4pnum = mol->find_atom_in_residue("O4'", res);
        if (o4pnum < 0) {
          o4pnum = mol->find_atom_in_residue("O4*", res);
        }
        c2pnum = mol->find_atom_in_residue("C2'", res); 
        if (c2pnum < 0) {
          c2pnum = mol->find_atom_in_residue("C2*", res);
        }
        c3pnum = mol->find_atom_in_residue("C3'", res); 
        if (c3pnum < 0) {
          c3pnum = mol->find_atom_in_residue("C3*", res);
        }
        c4pnum = mol->find_atom_in_residue("C4'", res);
        if (c4pnum < 0) {
          c4pnum = mol->find_atom_in_residue("C4*", res);
        }

        if (o4pnum < 0 || c2pnum < 0 || c3pnum < 0 || c4pnum < 0) {
          rc |= RIBBON_ERR_BADNUCLEIC;
          continue;
        }

        o4ppos = framepos + 3L*o4pnum;
        c2ppos = framepos + 3L*c2pnum;
        c3ppos = framepos + 3L*c3pnum;
        c4ppos = framepos + 3L*c4pnum;
	 
        midpoint(midptc1pc4p, c1ppos, c4ppos);
 	 
        // now display triangles 
        cmdColorIndex.putdata(atomColor->color[c1pnum], cmdList);
        
        cmdTriangle.putdata(c4ppos,c1ppos,o4ppos,cmdList);
        cmdTriangle.putdata(c3ppos,midptc1pc4p,c4ppos,cmdList);
        cmdTriangle.putdata(c2ppos,midptc1pc4p,c3ppos,cmdList);
        cmdTriangle.putdata(c1ppos,midptc1pc4p,c2ppos,cmdList);
      }	

      // begin bases
      rescentra[0]=rescentra[1]=rescentra[2]=0.0;
      rescentrb[0]=rescentrb[1]=rescentrb[2]=0.0;
 
      // check for purine and pyrimidine specific atoms
      n9num = mol->find_atom_in_residue("N9", res);    	
      n1num = mol->find_atom_in_residue("N1", res);

      // if there is a N9, then this is a pyrimidine
      if ((n9num >= 0) && (atomSel->on[n9num])) {
        c8num = mol->find_atom_in_residue("C8", res); 
        n7num = mol->find_atom_in_residue("N7", res); 
        c6num = mol->find_atom_in_residue("C6", res); 
        c5num = mol->find_atom_in_residue("C5", res);
        c4num = mol->find_atom_in_residue("C4", res);
        n3num = mol->find_atom_in_residue("N3", res);
        c2num = mol->find_atom_in_residue("C2", res);
        n1num = mol->find_atom_in_residue("N1", res);

        if (c8num < 0 || n7num < 0 || c6num < 0 || c5num < 0 ||
            c4num < 0 || n3num < 0 || c2num < 0 || n1num < 0) {
          rc |= RIBBON_ERR_BADPYRIMIDINE;
          continue;
        }

        n9pos = framepos + 3L*n9num;    	
        vec_add(rescentra,rescentra,n9pos);
        c8pos = framepos + 3L*c8num; 
        vec_add(rescentra,rescentra,c8pos);
        n7pos = framepos + 3L*n7num; 
        vec_add(rescentra,rescentra,n7pos);

        c5pos = framepos + 3L*c5num;
        vec_add(rescentra,rescentra,c5pos);
        vec_add(rescentrb,rescentrb,c5pos);
        c4pos = framepos + 3L*c4num;
        vec_add(rescentra,rescentra,c5pos);
        vec_add(rescentrb,rescentrb,c5pos);

        c6pos = framepos + 3L*c6num; 
        vec_add(rescentrb,rescentrb,c6pos);
        n3pos = framepos + 3L*n3num;
        vec_add(rescentrb,rescentrb,n3pos);
        c2pos = framepos + 3L*c2num;
        vec_add(rescentrb,rescentrb,c2pos);
        n1pos = framepos + 3L*n1num;
        vec_add(rescentrb,rescentrb,n1pos);
		
        rescentrb[0] = rescentrb[0]/6.0f;
        rescentrb[1] = rescentrb[1]/6.0f;
        rescentrb[2] = rescentrb[2]/6.0f;

        rescentra[0] = rescentra[0]/5.0f;
        rescentra[1] = rescentra[1]/5.0f;
        rescentra[2] = rescentra[2]/5.0f;

        // draw bond from ribose to base
        cmdCylinder.putdata(c1ppos, n9pos, b_rad, b_res, 0, cmdList);
        // now display triangles
        cmdColorIndex.putdata(atomColor->color[n9num], cmdList);

        cmdTriangle.putdata(n1pos,rescentrb,c2pos,cmdList);
        cmdTriangle.putdata(c2pos,rescentrb,n3pos,cmdList);
        cmdTriangle.putdata(n3pos,rescentrb,c4pos,cmdList);
        cmdTriangle.putdata(c4pos,rescentrb,c5pos,cmdList);
        cmdTriangle.putdata(c5pos,rescentrb,c6pos,cmdList);
        cmdTriangle.putdata(c6pos,rescentrb,n1pos,cmdList);

        cmdTriangle.putdata(n9pos,rescentra,c8pos,cmdList);
        cmdTriangle.putdata(c8pos,rescentra,n7pos,cmdList);
        cmdTriangle.putdata(n7pos,rescentra,c5pos,cmdList);
        cmdTriangle.putdata(c5pos,rescentra,c4pos,cmdList);
        cmdTriangle.putdata(c5pos,rescentra,c4pos,cmdList);
        cmdTriangle.putdata(c4pos,rescentra,n9pos,cmdList);
      }	    
      else if (( n1num >= 0) && (atomSel->on[n1num])){
        // residue is purine and turned on
        c6num = mol->find_atom_in_residue("C6", res); 
        c5num = mol->find_atom_in_residue("C5", res);
        c4num = mol->find_atom_in_residue("C4", res);
        n3num = mol->find_atom_in_residue("N3", res);
        c2num = mol->find_atom_in_residue("C2", res);

        if (c6num < 0 || c5num < 0 || c4num < 0 || n3num < 0 || c2num < 0) {
          rc |= RIBBON_ERR_BADPURINE; 
          continue;
        }

        c6pos = framepos + 3L*c6num; 
        vec_add(rescentrb,rescentrb,c6pos);
        c5pos = framepos + 3L*c5num;
        vec_add(rescentrb,rescentrb,c5pos);
        c4pos = framepos + 3L*c4num;
        vec_add(rescentrb,rescentrb,c5pos);
        n3pos = framepos + 3L*n3num;
        vec_add(rescentrb,rescentrb,n3pos);
        c2pos = framepos + 3L*c2num;
        vec_add(rescentrb,rescentrb,c2pos);
        n1pos = framepos + 3L*n1num;
        vec_add(rescentrb,rescentrb,n1pos);

        rescentrb[0] = rescentrb[0]/6.0f;
        rescentrb[1] = rescentrb[1]/6.0f;
        rescentrb[2] = rescentrb[2]/6.0f;

        // draw bond from ribose to base
        cmdCylinder.putdata(c1ppos, n1pos, b_rad, b_res, 0, cmdList);
        cmdColorIndex.putdata(atomColor->color[n1num], cmdList);

        cmdTriangle.putdata(n1pos,rescentrb,c2pos,cmdList);
        cmdTriangle.putdata(c2pos,rescentrb,n3pos,cmdList);
        cmdTriangle.putdata(n3pos,rescentrb,c4pos,cmdList);
        cmdTriangle.putdata(c4pos,rescentrb,c5pos,cmdList);
        cmdTriangle.putdata(c5pos,rescentrb,c6pos,cmdList);
        cmdTriangle.putdata(c6pos,rescentrb,n1pos,cmdList);
      }
    }
  }

  return rc;
}

int DrawMolItem::draw_nucleotide_cylinders(float *framepos, int b_res, float b_rad, float ribbon_width, int use_cyl) {
  int frag, num;
  int rc=0;
  int lastcolor = -1; // only emit color changes when necessary

  b_rad *= 1.5; // hack to make it look a little better by default

  // 
  // XXX
  // Match residue names so we know which atom names to look for when we
  // generate nucleotide cyliners.  This approach requires no pre-analysis
  // of the molecule, but has the problem that performance is lost to the
  // redundant structure analysis work that goes on here.  Compared to the
  // rendering workload it is usually less significant, but it would be
  // better to build an acceleration structure that is updated on-demand,
  // e.g. when residue names have been changed by scripts or by newly 
  // loaded structures.  We should not have to grind through this process
  // again for every trajectory frame update.  This would allow us to use
  // more sophisticated approaches as well.
  //

  //
  // PDB/CHARMM residue names
  //
  int nuctypes[10];
  nuctypes[0] = mol->resNames.typecode((char *) "A");
  nuctypes[1] = mol->resNames.typecode((char *) "ADE");
  nuctypes[2] = mol->resNames.typecode((char *) "C");
  nuctypes[3] = mol->resNames.typecode((char *) "CYT");
  nuctypes[4] = mol->resNames.typecode((char *) "G");
  nuctypes[5] = mol->resNames.typecode((char *) "GUA");
  nuctypes[6] = mol->resNames.typecode((char *) "T");
  nuctypes[7] = mol->resNames.typecode((char *) "THY");
  nuctypes[8] = mol->resNames.typecode((char *) "U");
  nuctypes[9] = mol->resNames.typecode((char *) "URA");

  // 
  // AMBER uses a different nucleic residue naming convention which
  // is a combination one of 'D' or 'R' for DNA/RNA respectively,
  // followed by one of A, C, G, T, U, possibly followed by '3' or '5',
  // as described here: http://ffamber.cnsm.csulb.edu/
  // 
  int ambernuctypes[4L*2*3];
  const char *dnarnastr = "DR";
  const char *nucstr = "ACGTU";
  const char *nuctermstr = "35";

  int amberind=0;
  int foundambertypes=0;
  for (int nucind=0; nucind<5; nucind++) {
    char resstr[32];
    const char nuc = nucstr[nucind];  
    resstr[1] = nuc;

    for (int dnarna=0; dnarna<=1; dnarna++) {
      char dr = dnarnastr[dnarna];
      resstr[0] = dr;

      if (nucind == 3 && dnarna == 1)
        continue;
      if (nucind == 4 && dnarna == 0)
        continue;

      for (int termind=0; termind<3; termind++) {
        if (termind < 2) 
          resstr[2] = nuctermstr[termind];
        else 
          resstr[2] = '\0';
        resstr[3] = '\0';

        int resnameind = mol->resNames.typecode(resstr);
        ambernuctypes[amberind] = resnameind;
        if (resnameind >= 0) {
          foundambertypes++;
        }
        amberind++;
      }
    } 
  }

  ResizeArray<float> centers;
  ResizeArray<float> radii;
  ResizeArray<float> colors;
  ResizeArray<float> pickpointcoords;
  ResizeArray<int>   pickpointindices;

  // draw all fragments
  for (frag=0; frag<mol->nfragList.num(); frag++) {
    int loop;

    num = mol->nfragList[frag]->num(); // number of residues in this fragment

    // loop over all of the residues drawing nucleotide cylinders where we can
    for (loop=0; loop<num; loop++) {
      int istart = -1;  // init to invalid atom index
      int iend = -1;    // init to invalid atom index
      int res = (*mol->nfragList[frag])[loop];
      int myatom = mol->residueList[res]->atoms[0];
      int resnameindex = mol->atom(myatom)->resnameindex;

      if (istart < 0)
        istart = mol->find_atom_in_residue("C3'", res);
      if (istart < 0)
        istart = mol->find_atom_in_residue("C3*", res);
      if (istart < 0)
        istart = mol->find_atom_in_residue("C1'", res);
      if (istart < 0)
        istart = mol->find_atom_in_residue("C1*", res);

      // catch single nucleotides of interest
      if (istart < 0)
        istart = mol->find_atom_in_residue("P", res);

      // skip this nucleotide if a starting atom can't be found
      if (istart < 0)
        continue;

      // XXX now that we're using the C3 atoms as the backbone,
      // we'll only use the end atoms to select the nucleotide
      // cylinder, otherwise one can't use "backbone" and "not backbone"
      // selections to color the ribbon and nucleotide cylinders separately.  
#if 0
      // skip this nucleotide if the starting atom is turned off
      if (!(atomSel->on[istart]))
        continue;
#endif

      //
      // If nucleotide end atoms unassigned, try PDB/CHARMM residue names
      //
      if (iend < 0) {
        // assign end atoms using PDB/CHARMM residue names
        if (resnameindex == nuctypes[0] || resnameindex == nuctypes[1]) {
          // ADE 
          int n1num = mol->find_atom_in_residue("N1", res);
          if (n1num < 0 || !(atomSel->on[n1num]))
            continue;
          iend = n1num;
        } else if (resnameindex == nuctypes[2] || resnameindex == nuctypes[3]) {
          // CYT
          int n3num = mol->find_atom_in_residue("N3", res);
          if (n3num < 0 || !(atomSel->on[n3num]))
            continue;
          iend = n3num;
        } else if (resnameindex == nuctypes[4] || resnameindex == nuctypes[5]) {
          // GUA
          int n1num = mol->find_atom_in_residue("N1", res);
          if (n1num < 0 || !(atomSel->on[n1num]))
            continue;
          iend = n1num;
        } else if (resnameindex == nuctypes[6] || resnameindex == nuctypes[7] ||
                   resnameindex == nuctypes[8] || resnameindex == nuctypes[9]) {
          // THY or URA
          int o4num = mol->find_atom_in_residue("O4", res);
          if (o4num < 0 || !(atomSel->on[o4num]))
            continue;
          iend = o4num;
        } 
      }

      //
      // If nucleotide end atoms unassigned, try AMBER residue names
      //
      if (iend < 0 && foundambertypes) {
        // assign end atoms using AMBER residue names
        if (resnameindex == ambernuctypes[0] ||
            resnameindex == ambernuctypes[1] ||
            resnameindex == ambernuctypes[2] ||
            resnameindex == ambernuctypes[3] ||
            resnameindex == ambernuctypes[4] ||
            resnameindex == ambernuctypes[5]) {
          // ADE 
          int n1num = mol->find_atom_in_residue("N1", res);
          if (n1num < 0 || !(atomSel->on[n1num]))
            continue;
          iend = n1num;
        } else if (resnameindex == ambernuctypes[6] ||
                   resnameindex == ambernuctypes[7] ||
                   resnameindex == ambernuctypes[8] ||
                   resnameindex == ambernuctypes[9] ||
                   resnameindex == ambernuctypes[10] ||
                   resnameindex == ambernuctypes[11]) {
          // CYT
          int n3num = mol->find_atom_in_residue("N3", res);
          if (n3num < 0 || !(atomSel->on[n3num]))
            continue;
          iend = n3num;
        } else if (resnameindex == ambernuctypes[12] ||
                   resnameindex == ambernuctypes[13] ||
                   resnameindex == ambernuctypes[14] ||
                   resnameindex == ambernuctypes[15] ||
                   resnameindex == ambernuctypes[16] ||
                   resnameindex == ambernuctypes[17]) {
          // GUA
          int n1num = mol->find_atom_in_residue("N1", res);
          if (n1num < 0 || !(atomSel->on[n1num]))
            continue;
          iend = n1num;
        } else if (resnameindex == ambernuctypes[18] ||
                   resnameindex == ambernuctypes[19] ||
                   resnameindex == ambernuctypes[20] ||
                   resnameindex == ambernuctypes[21] ||
                   resnameindex == ambernuctypes[22] ||
                   resnameindex == ambernuctypes[23]) {
          // THY or URA
          int o4num = mol->find_atom_in_residue("O4", res);
          if (o4num < 0 || !(atomSel->on[o4num]))
            continue;
          iend = o4num;
        }
      }

      //
      // If we get to this point and no endpoint was defined, we try
      // to catch single nucleotides of interest.
      // This is a desperation move to identify modified bases that
      // use unusual residue names that can't be recognized by VMD.
      //
      if (iend < 0) {
        int o4num = mol->find_atom_in_residue("O4", res);
        if (o4num < 0 || !(atomSel->on[o4num]))
          continue;
        iend = o4num;
      }
 
      // add pick points for nucleotides 
      int pidx = 3L * istart;
      pickpointcoords.append3(&framepos[pidx]);

      pidx = 3L * iend;
      pickpointcoords.append3(&framepos[pidx]);

      pickpointindices.append(istart);
      pickpointindices.append(iend);

      // only emit color command if it has changed
      if (lastcolor != atomColor->color[istart]) {
        lastcolor = atomColor->color[istart]; 
        cmdColorIndex.putdata(lastcolor, cmdList);
      }

      // get coordinates of start and end atom
      float *cstart = framepos + 3L*istart;
      float *cend = framepos + 3L*iend;

      // draw the nucleotide cylinder
      cmdCylinder.putdata(cstart, cend, b_rad, b_res, 0, cmdList); 

      // cap ends with spheres
      const float *cp = scene->color_value(lastcolor);

#if 0
      // only draw the ribbon-end sphere the starting atom is turned off
      // (thus no ribbon is being drawn..)  If the ribbon is being drawn,
      // we'll leave that end of the cylinder totally open.
      if (!(atomSel->on[istart])) {
        centers.append3(&cstart[0]);

        radii.append(b_rad);

        colors.append3(&cp[0]);
      }
#endif

      centers.append3(&cend[0]);
      radii.append(b_rad);
      colors.append3(&cp[0]);
    } 
  } 

  // draw the spheres if we have any
  if (radii.num() > 0) {
    cmdSphereArray.putdata((float *) &centers[0],
                           (float *) &radii[0],
                           (float *) &colors[0],
                           radii.num(),
                           b_res,
                           cmdList);
  }

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0],
                           &pickpointcoords[0], cmdList);
  }

  return rc;
}


// draw ribbons along the protein backbone
void DrawMolItem::draw_ribbons_new(float *framepos, float brad, int bres, int use_bspline, float thickness) {
  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning Ribbons",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  // find out if I'm using lines or cylinders
  int use_cyl = TRUE; // use cylinders by default
  int rc = 0;  // clear errors
  
  if (bres <= 2 || brad < 0.01f) {
    use_cyl = FALSE; // the representation will be as lines
  } 

  if (use_bspline) {
    create_Bspline_basis(spline_basis);
  }

  append(DMATERIALON); // enable shading 

  float ribbon_width = thickness;
  if (ribbon_width < 1.0) 
    ribbon_width = 1.0;
  ribbon_width /= 7.0;

  rc |= draw_protein_ribbons_new(framepos, bres, brad, ribbon_width, use_cyl);
  rc |= draw_nucleic_ribbons(framepos, bres, brad, ribbon_width, use_cyl, 1, 1);
  rc |= draw_base_sugar_rings(framepos, bres, brad, ribbon_width, use_cyl);

  if (rc != 0) {
    if (emitstructwarning())
      msgErr << "Warning: ribbons code encountered an unusual structure, geometry may not look as expected." << sendmsg;
  }

  // set the spline basis back to the default CR so that if we change rep
  // styles the other reps don't get messed up.  Ahh, the joys of one big
  // monolithic class...
  if (use_bspline) {
    create_modified_CR_spline_basis(spline_basis, 1.25f);
  }
}



//
// The new ribbons code more clearly separates the calculation of
// the ribbon spline itself and the construction of the extrusion along
// the spline.  Future implementations may be able to perform the 
// extrusion work on graphics accelerator hardware rather than on the host
// CPU.
//
void DrawMolItem::draw_spline_new(int num, const float *coords,
                                  const float *offset, const int *idx, 
                                  const float *cpwidths, const float *cpheights,
                                  int cpscalefactors, int b_res, int cyclic) {
  float q[4][3]; // spline Q matrix
  float *pts = NULL;
  float *prps = NULL;
  int   *cols = NULL;
  float *interpwidths = NULL;
  float *interpheights = NULL;
  const float *widths = NULL;
  const float *heights = NULL;
  int i, j;
  int state, spanlen, numscalefactors;
  ResizeArray<float> pickpointcoords;
  ResizeArray<int> pickpointindices;

  // number of spline interpolations between control points
  int splinedivs = b_res + 2; 
  float invsplinedivs = 1.0f / ((float) splinedivs);
  int hdivs = splinedivs / 2;

  // allocations for interpolated spline points, perps, and color indices 
  // there can be a max of splinedivs * (num + 1) sections rendered, so
  // we allocate the max we'll ever need and go from there.
  pts  = (float *) malloc(sizeof(float) * splinedivs * 3L * (num + 1L));
  prps = (float *) malloc(sizeof(float) * splinedivs * 3L * (num + 1L));
  cols = (int *)   malloc(sizeof(int)   * splinedivs * (num + 1L));

  // allocate memory for per-section interpolated width/height values
  if (cpscalefactors == num) {
    numscalefactors = splinedivs * (num + 1);
    interpwidths  = (float *) malloc(sizeof(float) * numscalefactors);
    interpheights = (float *) malloc(sizeof(float) * numscalefactors);
    widths = interpwidths;
    heights = interpheights;
  } else if (cpscalefactors == 1) {
    numscalefactors = 1;
    widths  = cpwidths;
    heights = cpheights;
  } else {
    return; // XXX error, this should never happen
  }

  if ((pts != NULL) && (prps != NULL) && (cols != NULL)) { 
    // place pick points for CA atoms
    for (i=0; i<num; i++) {
      if (atomSel->on[idx[i]]) {
        pickpointcoords.append3(&coords[3L * i]);
        pickpointindices.append(idx[i]);
      }
    }

    // draw spans of connected 'on' segments 
    state=0;     // no 'on' segments yet
    spanlen = 0; // reset span length, no segs yet (# of splinedivs) 
    for (i=0; i<num; i++) {
      if (atomSel->on[idx[i]]) {
        int cindex;
        int iminus1min, iminus2min, iplus1max;
        if (!cyclic) {
          // duplicate start/end control points for non-cyclic fragments
          iminus1min = ((i - 1) >= 0) ? (i - 1) : 0;
          iminus2min = ((i - 2) >= 0) ? (i - 2) : 0;
          iplus1max  = (i + 1 < num) ? (i + 1) : (num-1);
        } else {
          // cyclic structures use modular control point indexing
          iminus1min = (num + i - 1) % num;
          iminus2min = (num + i - 2) % num;
          iplus1max  = (num + i + 1) % num;
        }

        // reset and start a new span
        if (state == 0) {
          state = 1;   // a new span is in progress
          spanlen = 0; // reset span length, no segs yet (# of splinedivs) 
        }

        // construct the interpolation points (into the "pts" array)
        // remember, these are offset by two to keep some information
        // about the previous residue in the first 2 elements
        make_spline_Q_matrix_noncontig(q, spline_basis, 
                          &coords[iminus2min * 3],
                          &coords[iminus1min * 3],
                          &coords[i          * 3],
                          &coords[iplus1max  * 3]);

        // calculated interpolated spline points
        for (j=0; j<splinedivs; j++) {    
          make_spline_interpolation(&pts[(spanlen + j) * 3], 
                                    j * invsplinedivs, q);
        }

        // range-clamp control point index since the first few are copies
        int cpind = iminus1min * 3;

        // interpolate perps
        for (j=0; j<splinedivs; j++) {  
          int ind = (spanlen + j)*3;
          float v = j * invsplinedivs;
          float vv = (1.0f - v); 
          prps[ind    ] = vv * offset[cpind    ] + v * offset[cpind + 3];
          prps[ind + 1] = vv * offset[cpind + 1] + v * offset[cpind + 4];
          prps[ind + 2] = vv * offset[cpind + 2] + v * offset[cpind + 5];
          vec_normalize(&prps[ind]);
        }

        // interpolate width/height values
        if (cpscalefactors == num) {
          float wminus1 = cpwidths[iminus1min];
          float w = cpwidths[i];
          float hminus1 = cpheights[iminus1min];
          float h = cpheights[i];

          if (w >= 0 && wminus1 >= 0) {
            // if both widths are positive, interpolate between them, 
            // otherwise take the previous width value rather than 
            // interpolating, used for drawing beta arrows.
            for (j=0; j<splinedivs; j++) {  
              int ind = spanlen + j;
              float v = j * invsplinedivs;
              float vv = (1.0f - v); 
              interpwidths[ind]  = vv * wminus1 + v * w;
              interpheights[ind] = vv * hminus1 + v * h;
            }
          } else {
            float wplus1 = cpwidths[iplus1max];
            if (wplus1 < 0) 
              wplus1 = -wplus1; // undo width negation

            if (w < 0) {
              w = -w; // undo width negation

              for (j=0; j<hdivs; j++) {  
                int ind = spanlen + j;
                float v = j * invsplinedivs;
                interpwidths[ind]  = wminus1;
                interpheights[ind] = (1.0f - v) * hminus1 + v * h;
              }
              for (j=hdivs; j<splinedivs; j++) {  
                int ind = spanlen + j;
                float v = j * invsplinedivs;
                float nv = (j-hdivs) * invsplinedivs;
                interpwidths[ind]  = (1.0f - nv) * w + nv * wplus1;
                interpheights[ind] = (1.0f - v) * hminus1 + v * h;
              }
            } else {
              wminus1 = -wminus1; // undo width negation

              for (j=0; j<hdivs; j++) {  
                int ind = spanlen + j;
                float v = j * invsplinedivs;
                float nv = (j + (splinedivs - hdivs)) * invsplinedivs;
                interpwidths[ind]  = (1.0f - nv) * wminus1 + nv * w;
                interpheights[ind] = (1.0f - v) * hminus1 + v * h;
              }
              for (j=hdivs; j<splinedivs; j++) {  
                int ind = spanlen + j;
                float v = j * invsplinedivs;
                interpwidths[ind]  = w;
                interpheights[ind] = (1.0f - v) * hminus1 + v * h;
              }
            }
          }
        }

        // lookup atom color indices and assign to the interpolated points
        cindex = atomColor->color[idx[iminus1min]]; 
        for (j=0; j<hdivs; j++) {    
          cols[spanlen + j] = cindex; // set color index for each point
        } 

        cindex = atomColor->color[idx[i]]; 
        for (j=hdivs; j<splinedivs; j++) {    
          cols[spanlen + j] = cindex; // set color index for each point
        }
  
        spanlen += splinedivs; // number of segments rendered (cp * splinediv)
      } else {
        // render a finished span
        if (state == 1) {
          state = 0; // span is finished, ready to start a new one

          // draw last section to connect with other reps...
          int iminus1min, iminus2min, iplus1max;
          if (!cyclic) {
            // duplicate start/end control points for non-cyclic fragments
            iminus1min = ((i - 1) >= 0) ? (i - 1) : 0;
            iminus2min = ((i - 2) >= 0) ? (i - 2) : 0;
            iplus1max  = (i + 1 < num) ? (i + 1) : (num-1);
          } else {
            // cyclic structures use modular control point indexing
            iminus1min = (num + i - 1) % num;
            iminus2min = (num + i - 2) % num;
            iplus1max  = (num + i + 1) % num;
          }

          int ind = spanlen * 3;
          make_spline_interpolation(&pts[ind], 1.0, q);
          int cpind = iminus2min * 3;
          prps[ind    ] = offset[cpind + 3]; 
          prps[ind + 1] = offset[cpind + 4]; 
          prps[ind + 2] = offset[cpind + 5]; 
          vec_normalize(&prps[ind]);

          if (cpscalefactors == num) {
            float w = cpwidths[i];
            float wminus1 = cpwidths[iminus1min];
            // float h = cpheights[i];
            float hminus1 = cpheights[iminus1min];

            if (w >= 0 && wminus1 >= 0) {
              interpwidths[spanlen] = wminus1;
              interpheights[spanlen] = hminus1; 
            } else {
              float wplus1 = cpwidths[iplus1max];
              if (wplus1 < 0) 
                wplus1 = -wplus1; // undo width negation

              if (w < 0) {
                w = -w; // undo width negation

                float nv = (splinedivs - hdivs) * invsplinedivs;
                interpwidths[spanlen]  = (1.0f - nv) * w + nv * wplus1;
                interpheights[spanlen] = wminus1;
              } else {
                if (w < 0) 
                  w = -w; // undo width negation
                wminus1 = -wminus1; // undo width negation
                interpwidths[spanlen]  = w;
                interpheights[spanlen] = hminus1;
              }
            }
          }

          cols[spanlen] = atomColor->color[idx[iminus1min]];
          spanlen++;

          // draw a ribbon with the spline data
          int scalenum = (cpscalefactors == num) ? spanlen : 1;
          draw_ribbon_from_points(spanlen, pts, prps, cols, 
                                  b_res, widths, heights, scalenum);
        }
      }
    } 

    if (state == 1) {
      // draw last section to connect with other reps...
      int iminus1min, iminus2min, iplus1max;
      if (!cyclic) {
        // duplicate start/end control points for non-cyclic fragments
        iminus1min = ((i - 1) >= 0) ? (i - 1) : 0;
        iminus2min = ((i - 2) >= 0) ? (i - 2) : 0;
        iplus1max  = (i + 1 < num) ? (i + 1) : (num-1);
      } else {
        // cyclic structures use modular control point indexing
        iminus1min = (num + i - 1) % num;
        iminus2min = (num + i - 2) % num;
        iplus1max  = (num + i + 1) % num;
      }

      int ind = spanlen * 3;
      make_spline_interpolation(&pts[ind], 1.0, q);
      int cpind = iminus2min * 3;
      prps[ind    ] = offset[cpind + 3]; 
      prps[ind + 1] = offset[cpind + 4]; 
      prps[ind + 2] = offset[cpind + 5]; 
      vec_normalize(&prps[ind]);

      if (cpscalefactors == num) {
        float w = cpwidths[iminus1min];
        float wminus1 = cpwidths[iminus1min];
        // float h = cpheights[iminus1min];
        float hminus1 = cpheights[iminus1min];

        if (w >= 0 && wminus1 >= 0) {
          interpwidths[spanlen] = wminus1;
          interpheights[spanlen] = hminus1; 
        } else {
          float wplus1 = cpwidths[iplus1max];
          if (wplus1 < 0) 
            wplus1 = -wplus1; // undo width negation

          if (w < 0) {
            w = -w; // undo width negation

            float nv = (splinedivs - hdivs) * invsplinedivs;
            float nvv = (1.0f - nv); 
            interpwidths[spanlen]  = nvv * w + nv * wplus1;
            interpheights[spanlen] = wminus1;
          } else {
            if (w < 0) 
              w = -w; // undo width negation
            wminus1 = -wminus1; // undo width negation
            interpwidths[spanlen]  = w;
            interpheights[spanlen] = hminus1;
          }
        }
      }

      cols[spanlen] = atomColor->color[idx[iminus1min]];
      spanlen++;

      // draw a ribbon with the spline data
      int scalenum = (cpscalefactors == num) ? spanlen : 1;
      draw_ribbon_from_points(spanlen, pts, prps, cols, 
                              b_res, widths, heights, scalenum);
    }
  }

  if (pts != NULL)
    free(pts);

  if (prps != NULL) 
    free(prps);

  if (cols != NULL)
    free(cols);

  // deallocate memory for per-section interpolated width/height values
  if (interpwidths != NULL)
    free(interpwidths);

  if (interpheights != NULL)
    free(interpheights);

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0],
                           &pickpointcoords[0], cmdList);
  }
}


// 
// Routine to render a ribbon or tube based on an array of points that
// define the backbone spline on which an extrusion is built.  The 
// cross-section of the extrusion can be any oval shape described by
// height/width parameters that are used to scale a circular cross 
// section.
//
void DrawMolItem::draw_ribbon_from_points(int numpoints, const float *points, 
                  const float *perps, const int *cols, int numpanels, 
                  const float *widths, const float *heights, 
                  int numscalefactors) {
  int numverts, numsections;
  int point, section, panel, panelindex;
  int index; // array index of the point coordinate we're working with
  float *vertexarray, *normalarray, *colorarray;
  float *panelshape, *panelverts, *panelnorms;
  float curdir[3], perpdir[3], updir[3];
  float width, height, lastwidth, lastheight;

  // we must have at least two points and two panels per section
  if (numpoints < 2 || numpanels < 2)
    return; // skip drawing if basic requirements are not met
 
  numsections = numpoints - 1;
  numverts = numpoints * numpanels;  

  // storage for final vertex array data
  vertexarray = (float *) malloc(numverts * 3L * sizeof(float));
  normalarray = (float *) malloc(numverts * 3L * sizeof(float));
   colorarray = (float *) malloc(numverts * 3L * sizeof(float));

  // storage for panel cross-section data
  panelverts = (float *) malloc(numpanels * 2L * sizeof(float));
  panelnorms = (float *) malloc(numpanels * 2L * sizeof(float));
  panelshape = (float *) malloc(numpanels * 2L * sizeof(float));

  // bail out if any of the memory allocations fail
  if (!vertexarray || !normalarray || !colorarray ||
      !panelverts  || !panelnorms  || !panelshape) {
    if (vertexarray) free(vertexarray);
    if (normalarray) free(normalarray);
    if (colorarray)  free(colorarray);
    if (panelverts)  free(panelverts);
    if (panelnorms)  free(panelnorms);
    if (panelshape)  free(panelshape);
    return;
  }

  // Pre-calculate cross-section template shape.  The current template shape 
  // is a unit circle, but it could be any shape that can be scaled efficiently
  // by the code that generates section offsets and normals.
  for (panel=0; panel<numpanels; panel++) {
    int pidx = panel * 2;
    float radangle = (float) ((VMD_TWOPI) * (panel / ((float) numpanels)));
    panelshape[pidx    ] = sinf(radangle);
    panelshape[pidx + 1] = cosf(radangle);
  }

  // initialize with invalid values so we recalculate the section 
  // vertex offsets and normals the first time through.
  lastwidth  = -999;
  lastheight = -999;

  // setup first width/height values
  width  =  widths[0];
  height = heights[0];

  // calculate each cross section from the spline points and "perp" vectors
  // that are given to us by caller, combined with the vertex offsets and
  // normals we've already calculated.
  for (point=0, index=0; point<numpoints; point++, index+=3) {

    // If we are provided with width/height values for every section, we
    // have to update the cross-sections each time the values change.
    // Otherwise we just re-use the same cross-section vertex offsets and
    // normals over and over.
    if (numscalefactors == numpoints) {
      width  = widths[point];
      height = heights[point];
    }

    // Calculate cross-section vertex offsets and normals. For Cartoon 
    // reps these have to be recalculated for each section, but for
    // a plain ribbon rep, the width/height are constant and the data can
    // be re-used over and over again for all sections.   We test to see
    // if width or height have changed, and only recalculate when necessary.
    if (width != lastwidth || height != lastheight) {
      float invwidth, invheight;

      lastwidth = width;
      lastheight = width;

      // calculate inverse scaling coefficients which are applied to normals
      invwidth = 1.0f / width;
      invheight = 1.0f / height;

#if 1
      // Standard elliptical cross-section created by scaling a unit circle
      // and applying inverse scaling to the normals.  Simple to calculate,
      // generates decent looking results.
      for (panel=0; panel<numpanels; panel++) {
        int pidx = panel * 2;
        int pidy = pidx + 1;
        float xn, yn, invlen;

        // calculate vertices for a cross-section with a given width/height
        panelverts[pidx] = width  * panelshape[pidx];
        panelverts[pidy] = height * panelshape[pidy];

        // calculate normals for a cross-section with a given width/height
        xn = invwidth  * panelshape[pidx];
        yn = invheight * panelshape[pidy];
        invlen = 1.0f / sqrtf(xn*xn + yn*yn);
        panelnorms[pidx] = xn * invlen;
        panelnorms[pidy] = yn * invlen;
      }
#else
      // Alternative cross-section shape built from a completely flat ribbon 
      // capped by two semi-circles at the edges.  Looks similar to the
      // elliptical ribbon, but has a flat face, so vertices are concentrated
      // at the edges, giving better looking shading results for the same number
      // of drawn vertices in many cases.  This might be worth trying out 
      // in place of the elliptical implementation, though it requires a 
      // few more more calculations.
  
      // calculate vertices for semi-circle cap on the right hand side
      for (panel=0; panel<(numpanels/2); panel++) {
        int pidx = panel * 2;
        int pidy = pidx + 1;

        // calculate vertices for a cross-section with a given width/height
        panelverts[pidx] = (width  / 2.0f) + panelshape[pidx];
        panelverts[pidy] = height * panelshape[pidy];
      }

      // calculate vertices for semi-circle cap on the left hand side
      for (panel=(numpanels/2); panel<numpanels; panel++) {
        int pidx = panel * 2;
        int pidy = pidx + 1;

        // calculate vertices for a cross-section with a given width/height
        panelverts[pidx] = (-width  / 2.0f) + panelshape[pidx];
        panelverts[pidy] = height * panelshape[pidy];
      }

      // calculate normals for both sides
      for (panel=0; panel<numpanels; panel++) {
        int pidx = panel * 2;
        int pidy = pidx + 1;
        float xn, yn, invlen;
    
        // calculate normals for a cross-section with a given width/height
        xn = invwidth  * panelshape[pidx];
        yn = invheight * panelshape[pidy];
        invlen = 1.0f / sqrtf(xn*xn + yn*yn);
        panelnorms[pidx] = xn * invlen;
        panelnorms[pidy] = yn * invlen;
      }
#endif
    } // end of vertex offset and normal calculations


    // calculate and normalize the direction vector, but avoid going out
    // of bounds on the last point/index we calculate.
    if (point != (numpoints - 1)) {
      vec_sub(curdir, &points[index], &points[index + 3]); 
    } else {
      // handle the endpoint case where we re-use previous direction
      // this code requires that numpoints > 1.
      vec_sub(curdir, &points[index-3], &points[index]); 
    }
    vec_normalize(curdir);                 

    // calculate "up" from direction and perp vectors
    vec_copy(perpdir, &perps[index]);    // copy perpdir
    cross_prod(updir, curdir, perpdir);  // find section "up" direction   
    vec_normalize(updir);                // normalize the up direction

    // create vertices for the cross-section
    for (panel=0; panel<numpanels; panel++) {
      panelindex = (point * numpanels + panel) * 3;
      int pidx = panel * 2;
      float xv = panelverts[pidx    ];
      float yv = panelverts[pidx + 1];

      // use the unnormalized vertex direction vectors to construct the 
      // vertex position by offsetting from the original spline point.
      vertexarray[panelindex    ] = 
        (xv * perpdir[0]) + (yv * updir[0]) + points[index    ]; 
      vertexarray[panelindex + 1] = 
        (xv * perpdir[1]) + (yv * updir[1]) + points[index + 1]; 
      vertexarray[panelindex + 2] = 
        (xv * perpdir[2]) + (yv * updir[2]) + points[index + 2]; 
    }

    // create the normals for the cross-section
    for (panel=0; panel<numpanels; panel++) {
      panelindex = (point * numpanels + panel) * 3;
      int pidx = panel * 2;
      float xn = panelnorms[pidx    ];
      float yn = panelnorms[pidx + 1];

      // basis vectors are all pre-normalized so no re-normalization of the
      // resulting calculations needs to be done here inside this loop.
      normalarray[panelindex    ] = (xn * perpdir[0]) + (yn * updir[0]); 
      normalarray[panelindex + 1] = (xn * perpdir[1]) + (yn * updir[1]); 
      normalarray[panelindex + 2] = (xn * perpdir[2]) + (yn * updir[2]); 
    }

    // assign colors to the vertices in this cross-section
    const float *fp = scene->color_value(cols[point]);
    for (panel=0; panel<numpanels; panel++) {
      panelindex = ((point * numpanels) + panel) * 3;
      colorarray[panelindex    ] = fp[0]; // Red
      colorarray[panelindex + 1] = fp[1]; // Green
      colorarray[panelindex + 2] = fp[2]; // Blue
    }
  } 

  // generate facet lists from the vertex, normal, and color arrays
  // using the triangle strip primitive in VMD.
  int numstripverts = numsections * ((numpanels * 2) + 2);
  unsigned int * faces = new unsigned int[numstripverts];
  int * vertsperstrip = new int[numsections];
  int numstrips = numsections;

  int l=0;  
  for (section=0; section<numsections; section++) {
    vertsperstrip[section] = (numpanels * 2) + 2;

    int panel;
    for (panel=0; panel<numpanels; panel++) {
      // create a 2 triangles for each panel 
      int index = ((section * numpanels) + panel);
      faces[l    ] = index + numpanels;
      faces[l + 1] = index;
      l+=2;
    }

    // create a 2 triangles for each panel 
    int index  = section * numpanels;
    faces[l    ] = index + numpanels;
    faces[l + 1] = index;

    l+=2;
  }

  // Draw the ribbon!
  append(DMATERIALON);         // enable materials and lighting

  // draw triangle strips using single-sided lighting for best speed
  DispCmdTriStrips cmdTriStrips;  
  cmdTriStrips.putdata(vertexarray, normalarray, colorarray, numverts, 
                       vertsperstrip, numstrips, faces, numstripverts, 
                       0, cmdList);

  delete [] faces;
  delete [] vertsperstrip;

  free(vertexarray);
  free(normalarray);
  free(colorarray);
  free(panelshape);
  free(panelverts);
  free(panelnorms);
}

int DrawMolItem::draw_cartoon_ribbons(float *framepos, int b_res, float b_rad,
                             float ribbon_width, int use_cyl, int use_bspline) {
  float *coords = NULL;
  float *perps = NULL;
  int *idx = NULL;
  float *capos, *last_capos, *opos, *last_opos;
  int onum, canum, frag, num, res;
  float *widths, *heights;          // per-control point widths/heights
  int rc=0;

#if defined(VMDFASTRIBBONS)
  wkf_timerhandle tm = wkf_timer_create();
  wkf_timerhandle tm2 = wkf_timer_create();
  wkf_timerhandle tm3 = wkf_timer_create();
  wkf_timerhandle tm4 = wkf_timer_create();
  wkf_timer_start(tm);
  wkf_timer_start(tm3);
  double foo=0.0;
#endif

  // these are the variables used in the Raster3D package
  float a[3], b[3], c[3], d[3], e[3], g[3];  

  if (use_bspline) {
    create_Bspline_basis(spline_basis);
  }

  // draw nucleic acid ribbons and nucleotide cylinders
  rc |= draw_nucleic_ribbons(framepos, b_res, b_rad, ribbon_width / 7.0f, use_cyl, 1, 1);
  rc |= draw_nucleotide_cylinders(framepos, b_res, b_rad, ribbon_width / 7.0f, use_cyl);

  // Lookup atom typecodes ahead of time so we can use the most 
  // efficient variant of find_atom_in_residue() in the main loop.
  // If we can't find the atom types we need, bail out immediately
  int CAtypecode  = mol->atomNames.typecode("CA");
  int Otypecode   = mol->atomNames.typecode("O");
  int OT1typecode = mol->atomNames.typecode("OT1");

  if (CAtypecode < 0 || ((Otypecode < 0) && (OT1typecode < 0))) {
    return rc; // can't draw a ribbon without CA and O atoms for guidance
  }
#if defined(VMDFASTRIBBONS)
  wkf_timer_stop(tm3);
  msgInfo << "Cartoon nucleotide time: " << wkf_timer_time(tm3) << sendmsg;
#endif

  // Indicate that I need secondary structure information
  mol->need_secondary_structure(1);  // calculate 2ndary structure if need be

#if defined(VMDFASTRIBBONS)
  wkf_timer_start(tm2);
#endif

  // allocate for the maximum possible per-residue control points, perps, 
  // and indices so we don't have to reallocate for every fragment
  coords = (float *) malloc((mol->nResidues) * sizeof(float)*3);
     idx =   (int *) malloc((mol->nResidues) * sizeof(int));
   perps = (float *) malloc((mol->nResidues) * sizeof(float)*3);

  // cross section widths and heights
  widths = (float *) malloc(mol->nResidues * sizeof(float));
 heights = (float *) malloc(mol->nResidues * sizeof(float));

  // go through each protein and find the CA and O
  // from that, construct the offsets to use (called "perps")
  for (frag=0; frag<mol->pfragList.num(); frag++) {
    int cyclic=mol->pfragCyclic[frag];  // whether fragment is cyclic
    num = mol->pfragList[frag]->num();  // number of residues in this fragment
    if (num < 2) {
      rc |= RIBBON_ERR_NOTENOUGH;
      continue; // can't draw a ribbon with only one element, so skip
    }

    // check that we have a valid structure before continuing
      res = (*mol->pfragList[frag])[0];
    canum = mol->find_atom_in_residue(CAtypecode, res);
     onum = mol->find_atom_in_residue(Otypecode, res);

    if (canum < 0 || onum < 0) {
      continue; // can't find 1st CA or O of the protein, so don't draw
    }

    // initialize last 4 CA indices for use by helix rendering code
    int ca2, ca3, ca4;
    ca2=ca3=ca4=canum;

    int starthelix=-1;
    int endhelix=-1;    
    float starthelixperp[3];

    // initialize CA and O atom pointers
    capos = framepos + 3L*canum;
    last_capos = capos;
    opos = framepos + 3L*onum;
    last_opos = opos;
      
    // now go through and set the coordinates and the perps
    e[0] = e[1] = e[2] = 0.0;
    vec_copy(g, e);

    // for a cyclic structure, we use the positions of the last residue
    // to seed the initial direction vectors for the ribbon
    if (cyclic) {
      int lastres = (*mol->pfragList[frag])[num-1];
      int lastcanum = mol->find_atom_in_residue(CAtypecode, lastres);
      last_capos = framepos + 3L*lastcanum;

      int lastonum = mol->find_atom_in_residue(Otypecode, lastres);
      if (lastonum < 0 && OT1typecode >= 0) {
        lastonum = mol->find_atom_in_residue(OT1typecode, lastres);
      }
      last_opos = framepos + 3L*lastonum;

      // now I need to figure out where the ribbon goes
      vec_sub(a, capos, last_capos);     // A=(pos(CA(res)) - pos(CA(res-1)))
      vec_sub(b, last_opos, last_capos); // B=(pos(O(res-1)) - pos(CA(res-1)))
      cross_prod(c, a, b);               // C=AxB, define res-1's peptide plane
      cross_prod(d, c, a);               // D=CxA, normal to plane and backbone
      // if normal is > 90 deg from  previous one, negate the new one
      if (dot_prod(d, g) < 0) {
        vec_negate(b, d);
      } else {
        vec_copy(b, d);
      }
      vec_add(e, g, b);            // average to the sum of the previous vectors
      vec_normalize(e);
      vec_copy(g, e);              // make a cumulative sum; cute, IMHO
    }

    int loop;
    for (loop=0; loop<num; loop++) {
      res = (*mol->pfragList[frag])[loop];
      const int ss = mol->residue(res)->sstruct;
      float helixpos[3]; // storage for modified control point position

      int newcanum = mol->find_atom_in_residue(CAtypecode, res);
      if (newcanum >= 0) {
        ca4 = ca3;
        ca3 = ca2;
        ca2 = canum;
        canum = newcanum; 
        capos = framepos + 3L*canum;
      }

      onum = mol->find_atom_in_residue(Otypecode, res);
      if (onum < 0 && OT1typecode >= 0) {
        onum = mol->find_atom_in_residue(OT1typecode, res);
      }
      if (onum >= 0) {
        opos = framepos + 3L*onum;
      } else {
        rc |= RIBBON_ERR_PROTEIN_MESS;
        opos = last_opos; // try to cope if we have no oxygen index
      }
      idx[loop] = canum;

      // calculate ribbon cross section 
      int drawhelixwithrods = 0;
      int drawbetawithribbons = 0;
#if 0
      drawhelixwithrods = (getenv("VMDHELIXRODS") != NULL);
      drawbetawithribbons = (getenv("VMDBETARIBBONS") != NULL);
#endif

      switch (ss) {
        case SS_HELIX_ALPHA:
        case SS_HELIX_3_10:
        case SS_HELIX_PI:
          if (drawhelixwithrods) { 
            // if we just entered the helix, find the end of the helix so 
            // we can generate usable perps by interpolation.
            if (starthelix == -1) {
              int helind;
              starthelix = loop;
              for (helind=loop; helind<num; helind++) {
                int hres = (*mol->pfragList[frag])[helind];
                const int hss = mol->residue(hres)->sstruct;
                if (hss == SS_HELIX_ALPHA ||
                    hss == SS_HELIX_3_10 || 
                    hss == SS_HELIX_PI) {
                  endhelix = helind;
                } else {
                  break; // stop when this helix ends
                }
              }

              // save starting perp and ending perp, for interpolation later
            }

            widths[loop] = 4 * b_rad;
            heights[loop] = 4 * b_rad;
            vec_copy(helixpos, framepos + 3L * canum); 
            vec_add(helixpos, helixpos, framepos + 3L * ca2); 
            vec_add(helixpos, helixpos, framepos + 3L * ca3); 
            vec_add(helixpos, helixpos, framepos + 3L * ca4); 
            vec_scale(helixpos, 0.25, helixpos);

            // copy the helix coordinate into the control point array
            vec_copy(coords+loop*3, helixpos);
            capos = helixpos; // perps are calculated from modified position 
          } else {
            widths[loop] = ribbon_width * b_rad;
            heights[loop] = b_rad;
            // copy the CA coordinate into the control point array
            vec_copy(coords+loop*3, capos);
          }
          break;

        case SS_TURN:
        case SS_COIL:
        case SS_BRIDGE:
        default:
          widths[loop] = b_rad;
          heights[loop] = b_rad;
          // copy the CA coordinate into the control point array
          vec_copy(coords+loop*3, capos);
          break;

        case SS_BETA:
          widths[loop] = ribbon_width * b_rad;
          heights[loop] = b_rad;

          if (drawbetawithribbons) {
            // copy the CA coordinate into the control point array
            vec_copy(coords+loop*3, capos);
          } else {
            float betapos[3]; 
            int drawarrowhead = 0;

            // filter the control point positions, giving equal weight
            // to the current CA atom and it's two neighbors, so they
            // ideally cancel out the wiggles in the beta sheet, without
            // having to switch spline basis etc.
            int caplus1 = -1; // make invalid initially

            if ((loop+1) < num) {
              int nextres = (*mol->pfragList[frag])[loop+1];
              caplus1 = mol->find_atom_in_residue(CAtypecode, nextres);

              // draw directionality arrow if we're at the end
              if (mol->residue(nextres)->sstruct != SS_BETA) 
                drawarrowhead = 1;
            } else {
              // draw directionality arrow if we're at the end
              drawarrowhead = 1;
            }

            // mark this width to avoid interpolating it, and make
            // the arrowhead wider than the beta sheet body
            // XXX non-interpolated control points are negated
            if (drawarrowhead) 
              widths[loop] = -ribbon_width * b_rad * 1.75f;

            if (caplus1 < 0)
              caplus1 = canum;

            vec_copy(betapos, framepos + 3L * canum); 
            vec_scale(betapos, 2.0f, betapos);
            vec_add(betapos, betapos, framepos + 3L * caplus1); 
            vec_add(betapos, betapos, framepos + 3L * ca2); 
            vec_scale(betapos, 0.25f, betapos);

            // copy the beta coordinate into the control point array
            vec_copy(coords+loop*3, betapos);

            // perps will be generated using the original CA position
          }
          break;
      }


      // now I need to figure out where the ribbon goes
      vec_sub(a, capos, last_capos);     // A=(pos(CA(res)) - pos(CA(res-1)))
      vec_sub(b, last_opos, last_capos); // B=(pos(O(res-1)) - pos(CA(res-1)))
      cross_prod(c, a, b);               // C=AxB, define res-1's peptide plane
      cross_prod(d, c, a);               // D=CxA, normal to plane and backbone

      // if normal is > 90 deg from  previous one, negate the new one
      if (dot_prod(d, g) < 0) {
        vec_negate(b, d);
      } else {
        vec_copy(b, d);
      }
      vec_add(e, g, b);            // average to the sum of the previous vectors
      vec_normalize(e);
      vec_copy(&perps[3L*loop], e); // compute perps from the normal
      vec_copy(g, e);              // make a cumulative sum; cute, IMHO
      last_capos = capos;
      last_opos = opos;

      if (drawhelixwithrods) {
        if (loop == starthelix) {
          // save starting helix perp
          vec_copy(starthelixperp, e); 
        } else  if (loop > starthelix && loop < endhelix) {
          // should interpolate rather than copy, but this is just for testing
          vec_copy(&perps[3L*loop], starthelixperp);
        } else if (loop == endhelix) {
          // reset helix start and end if we've reached the end
          starthelix = -1; 
          endhelix = -1; 
        }
      }
                     
    }

    if (!cyclic) {
      // copy the second perp to the first since the first one didn't have
      // enough data to construct a good perpendicular.
      vec_copy(perps, perps+3);
    }

#if defined(VMDFASTRIBBONS)
    wkf_timer_start(tm4);
#endif
    // draw the cartoon 
    draw_spline_new(num, coords, perps, idx, widths, heights, num, b_res, cyclic);
#if defined(VMDFASTRIBBONS)
    wkf_timer_stop(tm4);
    foo+=wkf_timer_time(tm4);
#endif
  } // drew the protein fragment ribbon

  if (coords) {
    free(coords);
    free(idx);
    free(perps);
    free(widths);
    free(heights);
  }

  // set the spline basis back to the default CR so that if we change rep
  // styles the other reps don't get messed up.  Ahh, the joys of one big
  // monolithic class...
  if (use_bspline) {
    create_modified_CR_spline_basis(spline_basis, 1.25f);
  }

#if defined(VMDFASTRIBBONS)
  wkf_timer_stop(tm2);
  msgInfo << "Cartoon spline time: " << wkf_timer_time(tm2) << sendmsg;
  msgInfo << "     subspline time: " << foo << sendmsg;
#endif
#if defined(VMDFASTRIBBONS)
  wkf_timer_stop(tm);
  msgInfo << "Cartoon regen time: " << wkf_timer_time(tm) << sendmsg;
  wkf_timer_destroy(tm);
  wkf_timer_destroy(tm2);
  wkf_timer_destroy(tm3);
#endif

  return rc;
}

