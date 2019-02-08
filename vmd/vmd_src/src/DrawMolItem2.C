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
 *	$RCSfile: DrawMolItem2.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.38 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A continuation of rendering types from DrawMolItem
 *
 ***************************************************************************/

#include <stdlib.h>
#include "DrawMolItem.h"
#include "DrawMolecule.h"
#include "Scene.h"

// draw the backbone trace (along the C alphas)
// if no proteins are found, connect consequtive C alpha

//// SPECIAL MACRO for the search for CA if no proteins
// it is given the atom index
// if it is invalid (<0) it puts a NULL on the queue
// if it is valid, the new coords and index are pushed on the queue
// If the 2nd and 3rd coords are valid, a line is drawn between them
//   with the right colors, etc.
#define PUSH_QUEUE(atomid) {						\
  if (atomid < 0) {							\
    memmove(CA, CA+1, 3L*sizeof(float *)); CA[3] = NULL;		\
    memmove(indicies, indicies+1, 3L*sizeof(int)); indicies[3] = -1;	\
  } else {								\
    memmove(CA, CA+1, 3L*sizeof(float *)); CA[3] = framepos+3L*atomid;	\
    memmove(indicies, indicies+1, 3L*sizeof(int)); indicies[3] = atomid;\
  }                                        			        \
  /* check if I need to draw a bond */					\
  if (CA[1] && CA[2] && atomSel->on[indicies[1]] && atomSel->on[indicies[2]]) {	\
    float midcoord[3];							\
    midcoord[0] = (CA[1][0] + CA[2][0])/2.0f;				\
    midcoord[1] = (CA[1][1] + CA[2][1])/2.0f;				\
    midcoord[2] = (CA[1][2] + CA[2][2])/2.0f;				\
    cmdColorIndex.putdata(atomColor->color[indicies[1]], cmdList);	\
    make_connection(CA[0], CA[1], midcoord, NULL, 			\
                    brad, bres, use_cyl);				\
    cmdColorIndex.putdata(atomColor->color[indicies[2]], cmdList);	\
    make_connection(NULL, midcoord, CA[2], CA[3],			\
		  brad, bres, use_cyl);      				\
  }                                        			        \
}


// clear out the queue by calling -1 (inserts NULLs)
// This actual calls the queue three more times than need be for 
// PUSH_QUEUE to work correctly, but I prefered to keep the semantics
#define EMPTY_QUEUE {				\
  int atomidcode = -1;				\
  PUSH_QUEUE(atomidcode);			\
  PUSH_QUEUE(atomidcode);			\
  PUSH_QUEUE(atomidcode);			\
  PUSH_QUEUE(atomidcode);			\
}


void DrawMolItem::draw_trace(float *framepos, float brad, int bres, int linethickness) {
  int use_cyl = FALSE;
  if (bres <= 2 || brad < 0.01) { // then going to do lines
    append(DMATERIALOFF);
    cmdLineType.putdata(SOLIDLINE, cmdList);
    cmdLineWidth.putdata(linethickness, cmdList);
  } else {
    use_cyl = TRUE;
    append(DMATERIALON);
  }

  int pnum = mol->pfragList.num();
  if (pnum > 0) {
    // go along each protein fragment
    for (int pfrag=0; pfrag<pnum; pfrag++) {

      //   Reset the queue
      // I keep track of four C alphas since I want to use
      // the "make_connection" command to draw "extended"
      // cylinders at the CA bends, and need 4 coords to do that
      // find the CA coords for each residue
      float *CA[4] = {NULL, NULL, NULL, NULL};
      int indicies[4] = {-1, -1, -1, -1};
      int res_index;

      // go down the residues in each the protein fragment
      int rnum = mol->pfragList[pfrag]->num();
      for (int res=0; res<rnum; res++) {
	res_index = (*mol->pfragList[pfrag])[res];
	PUSH_QUEUE(mol->find_atom_in_residue("CA", res_index));
      }
      // flush out the queue
      EMPTY_QUEUE
    }
  } else {
    // if there are no proteins, check for sequential records with a CA
    // given the definition of a PDB file, I can go down the list
    // and see if record(i).name == CA and record(i-1).name == CA
    //        and record(i).resid = record(i-1).resid + 1

    // As before, to connect records nicely I want to track four at a time
    float *CA[4] = {NULL, NULL, NULL, NULL};
    int indicies[4] = {-1, -1, -1, -1};
    int num = mol->nAtoms;
    int ca_num = mol->atomNames.typecode("CA");
    int last_resid = -10000;
    int resid;
    for (int i=0; i<num; i++) {
      MolAtom *atm = mol->atom(i);
      if (atm->nameindex == ca_num) {
	// found a CA, is it sequential?
	resid = atm->resid;
	if (resid == last_resid + 1) {
	  // Yippe! push it on the end of the queue
	  PUSH_QUEUE(i);
	} else {
	  EMPTY_QUEUE
	  // and add this element
	  PUSH_QUEUE(i);
	}
	last_resid = resid;
      } else {
	// didn't find a CA, so flush the queue
	EMPTY_QUEUE
	last_resid = -10000;
      }
    } // end of loop through atoms
    // and flush out any remaining data
    EMPTY_QUEUE
  } // end if check if has proteins

  // And do the same for nucleic acids
  int nnum = mol -> nfragList.num();
  if (nnum > 0) {
    // go along each nucleic fragment
    for (int nfrag=0; nfrag<nnum; nfrag++) {
      //   Reset the queue
      // I keep track of four P atoms since I want to use
      // the "make_connection" command to draw "extended"
      // cylinders at the P bends, and need 4 coords to do that
      // find the P coords for each residue
      // I keep the name CA since I want to use the same macros
      float *CA[4] = {NULL, NULL, NULL, NULL};
      int indicies[4] = {-1, -1, -1, -1};
      int res_index;

      // go down the residues in each the nucleic fragment
      int rnum = mol->nfragList[nfrag]->num();
      for (int res=0; res<rnum; res++) {
	res_index = (*mol->nfragList[nfrag])[res];
	PUSH_QUEUE(mol->find_atom_in_residue("P", res_index));
      }
      // flush out the queue
      EMPTY_QUEUE
    }
  } else {
    // if there are no proteins, check for sequential records with a P
    // given the definition of a PDB file, I can go down the list
    // and see if record(i).name == P and record(i-1).name == P
    //        and record(i).resid = record(i-1).resid + 1

    // As before, to connect records nicely I want to track four at a time
    float *CA[4] = {NULL, NULL, NULL, NULL};
    int indicies[4] = {-1, -1, -1, -1};
    int num = mol->nAtoms;
    int p_num = mol->atomNames.typecode("P");
    int last_resid = -10000;
    int resid;
    for (int i=0; i<num; i++) {
      MolAtom *atm = mol->atom(i);
      if (atm -> nameindex == p_num) {
	// found a P, is it sequential?
	resid = atm->resid;
	if (resid == last_resid + 1) {
	  // Yippe! push it on the end of the queue
	  PUSH_QUEUE(i);
	} else {
	  EMPTY_QUEUE
	  // and add this element
	  PUSH_QUEUE(i);
	}
	last_resid = resid;
      } else {
	// didn't find a P, so flush the queue
	EMPTY_QUEUE
	last_resid = -10000;
      }
    } // end of loop through atoms
    // and flush out any remaining data
    EMPTY_QUEUE

  } // end if check if has nucleic acid residues
}


// Draw dot surface
// the dot distribution is determined from Jon Leech's 'distribute'
// code.  See ftp://ftp.cs.unc.edu/pub/users/leech/points.tar.gz
// and ftp://netlib.att.com/netlib/att/math/sloane/electrons/
// All the dots are precomputed
#include "DrawMolItemSolventPoints.data"
// Note:  DrawMolItem::num_dot_surfaces is actually defined in
//    DrawMolItemSolventPoints.data
void DrawMolItem::draw_dot_surface(float *framepos, float srad, int sres, int method) {
  // XXX Hack - the value used for num_dot_surfaces should be retrieved
  // from AtomRep, I'm just not sure how to do it at the moment. 
  int num_dot_surfaces = 13;   // See the Solvent section of AtomRepInfo 

  DispCmdLineArray cmdLineArray;
  float probe_radius = srad;
  // sphereres has range 1 to n
  int surface_resolution = sres - 1; 
  if (surface_resolution >= num_dot_surfaces)   // slight paranoia
    surface_resolution = num_dot_surfaces - 1;
  if (surface_resolution < 0) 
    surface_resolution = 0;

  int num_dots = dot_surface_num_points[surface_resolution];
  float *dots = dot_surface_points[surface_resolution];
  int num_edges = dot_surface_num_lines[surface_resolution];
  int *edges = dot_surface_lines[surface_resolution];
  int *flgs = new int[num_dots];
  const float *aradius = mol->radius();

  if (method < 0) method = 0; if (method > 2) method = 2;

  append(DMATERIALOFF); // disable shading in all cases 
  cmdLineType.putdata(SOLIDLINE, cmdList); // set line drawing parameters
  cmdLineWidth.putdata(1, cmdList);

  // XXX really needs to be done only when selection or color changed
  update_lookups(atomColor, atomSel, colorlookups); 

  // temp info for drawing the little plus sign
  // I looked - none of the points are along the x axis
  float xaxis[3] = {1.0, 0.0, 0.0};
  float perp1[3], perp2[3];
  float pos1[3], pos2[3];

  ResizeArray<float> verts;
  ResizeArray<float> colors;

  for (int icolor=0; icolor<MAXCOLORS; icolor++) {
    const ColorLookup &cl = colorlookups[icolor];
    if (cl.num == 0) continue;

    const float *rgb = scene->color_value(icolor);
    if (method != 0) {
      cmdColorIndex.putdata(icolor, cmdList);
    }
    // draw points which are not within range of the bonded atoms
    for (int j=0; j<cl.num; j++) {
      const int id = cl.idlist[j];
      const MolAtom *atom = mol->atom(id);
      const float *pos = framepos + 3L*id;
      float radius = aradius[id] + probe_radius;
      for (int points=0; points < num_dots; points++) {
        const float *d = dots + 3L*points;
        flgs[points] = 1;
        float xyz[3];
        vec_scale(xyz, radius, d);
        vec_add(xyz, xyz, pos);
        // check the neighbors
        for (int nbr=0; nbr < atom->bonds; nbr++) {
          int b = atom->bondTo[nbr];
          const MolAtom *atom2 = mol->atom(b);
          float r = aradius[b] + probe_radius;
          if (distance2(xyz, framepos + 3L*b) < r*r) {
            flgs[points] = 0;
            break;
          }
          // check the neighbor's neighbors
          for (int nbr2=0; nbr2 < atom2->bonds; nbr2++) {
            int b2 = atom2->bondTo[nbr2];
            if (b2 == id) continue; // don't eliminate myself!
            float r2 = aradius[b2] + probe_radius;
            if (distance2(xyz, framepos + 3L*b2) < r2*r2) {
              flgs[points] = 0;
              break;
            }
          }
          if (!flgs[points]) break;
        }
        if (!flgs[points]) continue;

        switch (method) {
        case 0:
          // draw it as a point
          colors.append3(&rgb[0]);
          verts.append3(&xyz[0]);
          break;
        case 1:
          // draw a small cross tangent to the surface
          cross_prod(perp1, d, xaxis);  // d and xaxis are of length 1
          cross_prod(perp2, d, perp1);  // get the other tangent vector
          // scale appropriately
#define CROSS_SCALE_FACTOR 0.05f
          perp1[0] *= CROSS_SCALE_FACTOR;
          perp1[1] *= CROSS_SCALE_FACTOR;
          perp1[2] *= CROSS_SCALE_FACTOR;
          perp2[0] *= CROSS_SCALE_FACTOR;
          perp2[1] *= CROSS_SCALE_FACTOR;
          perp2[2] *= CROSS_SCALE_FACTOR;
          vec_add(pos1, xyz, perp1);
          vec_sub(pos2, xyz, perp1);
          verts.append3(&pos1[0]);
          verts.append3(&pos2[0]);
          vec_add(pos1, xyz, perp2);
          vec_sub(pos2, xyz, perp2);
          verts.append3(&pos1[0]);
          verts.append3(&pos2[0]);
          break;
        }
      }

      if (method == 1) {
        cmdLineArray.putdata(&verts[0], verts.num()/6, cmdList);
        verts.clear();
        continue;
      }

      // had to wait to accumulate all the possible vertices
      if (method == 2) {
        // draw all the connections if both points are used
        int a, b;
        int offset = 0;
        float xyz[2][3];
        // go through the dots
        for (a=0; a < num_dots; a++) {
          if (flgs[a]) {
            // if this point is turned on
            xyz[0][0] = pos[0] + radius * dots[3L*a + 0];
            xyz[0][1] = pos[1] + radius * dots[3L*a + 1];
            xyz[0][2] = pos[2] + radius * dots[3L*a + 2];
            
            // go through the matching connections
            while (offset < num_edges && edges[2L*offset] == a) {
              // is the neighbor turned on?
              b = edges[2L*offset + 1];
              if (flgs[b]) {
                xyz[1][0] = pos[0] + radius * dots[3L*b + 0];
                xyz[1][1] = pos[1] + radius * dots[3L*b + 1];
                xyz[1][2] = pos[2] + radius * dots[3L*b + 2];
                verts.append3(&xyz[0][0]);
                verts.append3(&xyz[1][0]);
              }
              offset++;
            }
          } else {
            // just go through the connection until the next number
            while (offset < num_edges && edges[2L*offset] == a) {
              offset++;
            }
          }
        } 
        cmdLineArray.putdata(&verts[0], verts.num()/6, cmdList);
        verts.clear();
      } // end of drawing as lines
    }
  }
  delete [] flgs;
  if (method == 0) {
    cmdPointArray.putdata(&verts[0], &colors[0], 1.0f, 
            verts.num()/3, cmdList);
  }
}

