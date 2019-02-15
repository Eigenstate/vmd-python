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
 *      $RCSfile: MeasurePBC.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.18 $       $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Code to measure atom distances, angles, dihedrals, etc, 
 *   accounting for periodic boundary conditions
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Measure.h"
#include "AtomSel.h"
#include "Matrix4.h"
#include "utilities.h"
#include "SpatialSearch.h"  // for find_within()
#include "MoleculeList.h"
#include "Inform.h"
#include "Timestep.h"
#include "VMDApp.h"

//
// Find an orthogonal basis R^3 with ob1=b1     
//

void orthonormal_basis(const float b[9], float e[9]) {
  float ob[3*3];
  vec_copy(ob+0, b+0);
  vec_copy(e+0, ob+0);
  vec_normalize(e+0);
  vec_triad(ob+3, b+3, -dot_prod(e+0, b+3), e+0);
  vec_copy(e+3, ob+3);
  vec_normalize(e+3);
  vec_triad(ob+6,  b+6, -dot_prod(e+0, b+6), e+0);
  vec_triad(ob+6, ob+6, -dot_prod(e+3, b+6), e+3);
  vec_copy(e+6, ob+6);
  vec_normalize(e+6);
}

//
// Returns basis vectors in coordinates of an orthonormal 
// basis obase.                                        
//

void basis_change(const float *base, const float *obase, float *newcoor, int n) {
  int i, j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      newcoor[n*i+j] = dot_prod(&base[n*j], &obase[n*i]);
    }
  }
}

// Compute matrix that transforms coordinates from an arbitrary PBC cell 
// into an orthonormal unitcell. Since the cell origin is not stored by VMD
// you have to specify it.
int measure_pbc2onc(MoleculeList *mlist, int molid, int frame, const float origin[3], Matrix4 &transform) {
  int orig_ts, max_ts;

  Molecule *mol = mlist->mol_from_id(molid);
  if( !mol )
    return MEASURE_ERR_NOMOLECULE;
 
  // get current frame number and make sure there are frames
  if((orig_ts = mol->frame()) < 0)
    return MEASURE_ERR_NOFRAMES;
  
  // get the max frame number and determine frame range
  max_ts = mol->numframes()-1;
  if (frame==-2)  frame = orig_ts;
  else if (frame>max_ts || frame==-1) frame = max_ts;

  Timestep *ts = mol->get_frame(frame);

  Matrix4 AA, BB, CC;
  ts->get_transforms(AA, BB, CC);

  // Construct the cell spanning vectors
  float cell[9];
  cell[0] = AA.mat[12];
  cell[1] = AA.mat[13];
  cell[2] = AA.mat[14];
  cell[3] = BB.mat[12];
  cell[4] = BB.mat[13];
  cell[5] = BB.mat[14];
  cell[6] = CC.mat[12];
  cell[7] = CC.mat[13];
  cell[8] = CC.mat[14];

  get_transform_to_orthonormal_cell(cell, origin, transform);

  return MEASURE_NOERR;
}

// Compute matrix that transforms coordinates from an arbitrary cell 
// into an orthonormal unitcell. Since the origin is not stored by VMD
// you have to specify it.
// This is the lowlevel backend of measure_pbc2onc().

// Here is a 2D example:
// A and B are the are the displacement vectors which are needed to create 
// the neighboring images. The parallelogram denotes the PBC cell with the origin O at its center.
// The sqare to the right indicates the orthonormal unit cell i.e. the area into which the atoms 
// will be wrapped by transformation T.
//
//                  + B                                        
//                 /                              ^ B'         
//       _________/________                       |            
//      /        /        /                   +---|---+        
//     /        /        /              T     |   |   |        
//    /        O--------/-------> A   ====>   |   O---|--> A'  
//   /                 /                      |       |        
//  /_________________/                       +-------+        
  
//  A  = displacement vector along X-axis with length a
//  B  = displacement vector in XY-plane with length b
//  A' = displacement vector along X-axis with length 1
//  B' = displacement vector along Y-axis with length 1
//  O = origin of the unit cell 

void get_transform_to_orthonormal_cell(const float *cell, const float *center, Matrix4 &transform) {
  // Orthogonalize system:
  // Find an orthonormal basis of the cell (in cartesian coords).
  // If the cell vectors from VMD/NAMD are used this should actually always
  // return the identity matrix due to the way the cell vectors A, B and C
  // are defined (i.e. A || x; B lies in the x,y-plane; A, B, C form a right
  // hand system).
  float obase[3*3];
  orthonormal_basis(cell, obase);

  // Get orthonormal base in cartesian coordinates (it is the inverse of the
  // obase->cartesian transformation):
  float identity[3*3] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float obase_cartcoor[3*3];
  basis_change(obase, identity, obase_cartcoor, 3);


  // Transform 3x3 into 4x4 matrix:
  Matrix4 obase2cartinv;
  trans_from_rotate(obase_cartcoor, &obase2cartinv);

  // This is the matrix for the obase->cartesian transformation:
  Matrix4 obase2cart = obase2cartinv;
  obase2cart.inverse();

  // Get coordinates of cell in terms of obase
  float m[3*3]; 
  basis_change(cell, obase, m, 3);
  Matrix4 rotmat;
  trans_from_rotate(m, &rotmat);
  rotmat.inverse();

  
  // Actually we have:
  // transform = translation * obase2cart * obase2cartinv * rotmat * obase2cart
  //                           `------------v------------'
  //                                       = 1
  transform = obase2cart;
  transform.multmatrix(rotmat); // pre-multiplication

  // Finally we need to apply the translation of the origin
  float origin[3];
  vec_copy(origin, center);
  vec_scaled_add(origin, -0.5, &cell[0]);
  vec_scaled_add(origin, -0.5, &cell[3]);
  vec_scaled_add(origin, -0.5, &cell[6]);
  vec_negate(origin, origin);
  //printf("origin={%g %g %g}\n", origin[0], origin[1], origin[2]);
  transform.translate(origin);
}


// Get the array of coordinates of pbc image atoms for the specified selection.
// The cutoff vector defines the region surrounding the pbc cell for which image 
// atoms shall be constructed ({6 8 0} means 6 Angstrom for the direction of A,
// 8 for B and no images in the C direction). The indexmap_array relates each
// image atom to its corresponding main atom.
// In case the molecule was aligned you can supply the alignment matrix which
// is then used to correct for the rotation and shift of the pbc cell.
// Since the pbc cell center is not stored in Timestep it must be provided.

// The algorithm transforms the unitcell so that the unitcell minus the cutoff fits
// into an orthonormal cell. Now the atoms in the rim can be easily identified and
// wrapped into the neigboring cells. This works only if the largest cutoff
// dimension is smaller than half of the smallest cell dimension. Otherwise a
// slower algorithm is used that wraps each atom into all 26 neighboring cells
// and checks if the image lies within cutoff.
//
//       ________________          
//      / ____________  /          +---------+
//     / /           / /           | +-----+ |
//    / /   core    / /   ---->    | |     |_|___orthonormal_cell
//   / /___________/ /    <----    | |     | |
//  /_______________/              | +-----+ |___rim
//                                 +---------+
//
// Alternatively one can specify a rectangular bounding box into which atoms
// shall be wrapped. It is specified in form of minmax coordinates through
// parameter *box. I.e. coordinates are produced for pbc image atom that lie
// inside the box but outside the central unit cell. This feature can be used
// for instance to retrieve coordinates of the minmax box of a selection when
// the box boundaries exceed the unit cell.
//
// If a selection is provided (sel!=NULL) we return only coordinates that are
// within the cutoff distance of that selection:

// The results are provided in form of an array of 'extended' coordinates,
// i.e. the coordinates of the requested region that don't lie in the central
// unit cell. In order to identify these coordinates with the respective atoms
// in the central cell an index map is also provided.

// TODO:
// * Allow requesting specific neighbor cells.

int measure_pbc_neighbors(MoleculeList *mlist, AtomSel *sel, int molid,
			  int frame, const Matrix4 *alignment,
			  const float *center, const float *cutoff, const float *box,
			  ResizeArray<float> *extcoord_array,
			  ResizeArray<int> *indexmap_array) {
  int orig_ts, max_ts;
  if (!box && !cutoff[0] && !cutoff[1] && !cutoff[2]) return MEASURE_NOERR;

  Molecule *mol = mlist->mol_from_id(molid);
  if( !mol )
    return MEASURE_ERR_NOMOLECULE;
 
  // get current frame number and make sure there are frames
  if((orig_ts = mol->frame()) < 0)
    return MEASURE_ERR_NOFRAMES;
  
  // get the max frame number and determine current frame
  max_ts = mol->numframes()-1;
  if (frame==-2)  frame = orig_ts;
  else if (frame>max_ts || frame==-1) frame = max_ts;

  Timestep *ts = mol->get_frame(frame);
  if (!ts) return MEASURE_ERR_NOMOLECULE;

  // Get the displacement vectors (in form of translation matrices)
  Matrix4 Tpbc[3][2];
  ts->get_transforms(Tpbc[0][1], Tpbc[1][1], Tpbc[2][1]);

  // Assign the negative cell translation vectors
  Tpbc[0][0] = Tpbc[0][1];
  Tpbc[1][0] = Tpbc[1][1];
  Tpbc[2][0] = Tpbc[2][1];
  Tpbc[0][0].inverse();
  Tpbc[1][0].inverse();
  Tpbc[2][0].inverse();

  // Construct the cell spanning vectors
  float cell[9];
  cell[0] = Tpbc[0][1].mat[12];
  cell[1] = Tpbc[0][1].mat[13];
  cell[2] = Tpbc[0][1].mat[14];
  cell[3] = Tpbc[1][1].mat[12];
  cell[4] = Tpbc[1][1].mat[13];
  cell[5] = Tpbc[1][1].mat[14];
  cell[6] = Tpbc[2][1].mat[12];
  cell[7] = Tpbc[2][1].mat[13];
  cell[8] = Tpbc[2][1].mat[14];

  float len[3];
  len[0] = sqrtf(dot_prod(&cell[0], &cell[0]));
  len[1] = sqrtf(dot_prod(&cell[3], &cell[3]));
  len[2] = sqrtf(dot_prod(&cell[6], &cell[6]));
  //printf("len={%.3f %.3f %.3f}\n", len[0], len[1], len[2]);

  int i;
  float minlen = len[0];
  if (len[1] && len[1]<minlen) minlen = len[1];
  if (len[2] && len[2]<minlen) minlen = len[2];
  minlen--;

  // The algorithm works only for atoms in adjacent neighbor cells.
  if (!box && (cutoff[0]>=len[0] || cutoff[1]>=len[1] || cutoff[2]>=len[2])) {
    return MEASURE_ERR_BADCUTOFF;
  }

  bool bigrim = 1;
  float corecell[9];
  float diag[3];
  float origin[3];
  memset(origin, 0, 3L*sizeof(float));
  Matrix4 M_norm;

  if (box) {
    // Get the matrix M_norm that transforms all atoms inside the 
    // unit cell into the normalized unitcell spanned by 
    // {1/len[0] 0 0} {0 1/len[1] 0} {0 0 1/len[2]}.
    bigrim = 1;

    float vtmp[3];
    vec_add(vtmp, &cell[0], &cell[3]);
    vec_add(diag, &cell[6], vtmp);
    //printf("diag={%.3f %.3f %.3f}\n", diag[0], diag[1], diag[2]);

    // Finally we need to apply the translation of the cell origin
    vec_copy(origin, center);
    vec_scaled_add(origin, -0.5, &cell[0]);
    vec_scaled_add(origin, -0.5, &cell[3]);
    vec_scaled_add(origin, -0.5, &cell[6]);
    vec_negate(origin, origin);
    //printf("origin={%.3f %.3f %.3f}\n", origin[0], origin[1], origin[2]);

  } else if (2.0f*cutoff[0]<minlen && 2.0f*cutoff[1]<minlen && 2.0f*cutoff[2]<minlen) {
    // The cutoff must not be larger than half of the smallest cell dimension
    // otherwise we would have to use a less efficient algorithm.

    // Get the matrix M_norm that transforms all atoms inside the 
    // corecell into the orthonormal unitcell spanned by {1 0 0} {0 1 0} {0 0 1}.
    // The corecell ist the pbc cell minus cutoffs for each dimension.
    vec_scale(&corecell[0], (len[0]-cutoff[0])/len[0], &cell[0]);
    vec_scale(&corecell[3], (len[1]-cutoff[1])/len[1], &cell[3]);
    vec_scale(&corecell[6], (len[2]-cutoff[2])/len[2], &cell[6]);
    get_transform_to_orthonormal_cell(corecell, center, M_norm);
    //printf("Using algorithm for small PBC environment.\n");

  } else {
    // Get the matrix M_norm that transforms all atoms inside the 
    // unit cell into the orthonormal unitcell spanned by {1 0 0} {0 1 0} {0 0 1}.
    get_transform_to_orthonormal_cell(cell, center, M_norm);

    bigrim = 1;
    //printf("Using algorithm for large PBC environment.\n");
  }

  // In case the molecule was aligned our pbc cell is rotated and shifted.
  // In order to transform a point P into the orthonormal cell (P') it 
  // first has to be unaligned (the inverse of the alignment):
  // P' = M_norm * (alignment^-1) * P
  Matrix4 alignmentinv(*alignment);
  alignmentinv.inverse();
  Matrix4 M_coretransform(M_norm);
  M_coretransform.multmatrix(alignmentinv);

  //printf("alignment = \n");
  //print_Matrix4(alignment);

  // Similarly if we want to transform a point P into its image P' we
  // first have to unalign it, then apply the PBC translation and 
  // finally realign:
  // P' = alignment * Tpbc * (alignment^-1) * P
  //      `-------------v--------------'
  //                transform
  int j, u;
  Matrix4 Tpbc_aligned[3][2];
  if (!box) {
    for (i=0; i<3; i++) {
      for (j=0; j<2; j++) {
        Tpbc_aligned[i][j].loadmatrix(*alignment);
        Tpbc_aligned[i][j].multmatrix(Tpbc[i][j]);
        Tpbc_aligned[i][j].multmatrix(alignmentinv);
      }
    }
  }

  Matrix4 M[3];
  float *coords = ts->pos;
  float *coor;
  float orthcoor[3], wrapcoor[3];

  //printf("cutoff={%.3f %.3f %.3f}\n", cutoff[0], cutoff[1], cutoff[2]);

  if (box) {
    float min_coord[3], max_coord[3];
    // Increase box by cutoff
    vec_sub(min_coord, box,   cutoff);
    vec_add(max_coord, box+3, cutoff);
    //printf("Wrapping atoms into rectangular bounding box.\n");
    //printf("min_coord={%.3f %.3f %.3f}\n", min_coord[0], min_coord[1], min_coord[2]);
    //printf("max_coord={%.3f %.3f %.3f}\n", max_coord[0], max_coord[1], max_coord[2]);
    vec_add(min_coord, min_coord, origin);
    vec_add(max_coord, max_coord, origin);

    float testcoor[9];
    int idx, k;
    // Loop over all atoms
    for (idx=0; idx<ts->num; idx++) { 
      coor = coords+3L*idx;

      // Apply the inverse alignment transformation
      // to the current test point.
      M_coretransform.multpoint3d(coor, orthcoor);

      // Loop over all 26 neighbor cells
      // x
      for (i=-1; i<=1; i++) {
        // Choose the direction of translation
        if      (i>0) M[0].loadmatrix(Tpbc[0][1]);
        else if (i<0) M[0].loadmatrix(Tpbc[0][0]);
        else 	      M[0].identity();
        // Translate the unaligned atom
        M[0].multpoint3d(orthcoor, testcoor);

        // y
        for (j=-1; j<=1; j++) {
          // Choose the direction of translation
          if      (j>0) M[1].loadmatrix(Tpbc[1][1]);
          else if (j<0) M[1].loadmatrix(Tpbc[1][0]);
          else 	        M[1].identity();
          // Translate the unaligned atom
          M[1].multpoint3d(testcoor, testcoor+3);

          // z
          for (k=-1; k<=1; k++) {
            if(i==0 && j==0 && k==0) continue;

            // Choose the direction of translation
            if      (k>0) M[2].loadmatrix(Tpbc[2][1]);
            else if (k<0) M[2].loadmatrix(Tpbc[2][0]);
            else    	  M[2].identity();
            // Translate the unaligned atom
            M[2].multpoint3d(testcoor+3, testcoor+6);

            // Realign atom
            alignment->multpoint3d(testcoor+6, wrapcoor);

            vec_add(testcoor+6, wrapcoor, origin);
            if (testcoor[6]<min_coord[0] || testcoor[6]>max_coord[0]) continue;
            if (testcoor[7]<min_coord[1] || testcoor[7]>max_coord[1]) continue;
            if (testcoor[8]<min_coord[2] || testcoor[8]>max_coord[2]) continue;

            // Atom is inside cutoff, add it to the list	    
            extcoord_array->append3(&wrapcoor[0]);
            indexmap_array->append(idx);
          }
        }
      }
    }

  } else if (bigrim) {
    // This is the more general but slower algorithm.
    // We loop over all atoms, move each atom to all 26 neighbor cells
    // and check if it lies inside cutoff
    float min_coord[3], max_coord[3];
    min_coord[0] = -cutoff[0]/len[0];
    min_coord[1] = -cutoff[1]/len[1];
    min_coord[2] = -cutoff[2]/len[2];
    max_coord[0] = 1.0f + cutoff[0]/len[0];
    max_coord[1] = 1.0f + cutoff[1]/len[1];
    max_coord[2] = 1.0f + cutoff[2]/len[2];

    float testcoor[3];
    int idx, k;
    // Loop over all atoms
    for (idx=0; idx<ts->num; idx++) { 
      coor = coords+3L*idx;

      // Apply the PBC --> orthonormal unitcell transformation
      // to the current test point.
      M_coretransform.multpoint3d(coor, orthcoor);

      // Loop over all 26 neighbor cells
      // x
      for (i=-1; i<=1; i++) {
        testcoor[0] = orthcoor[0]+(float)(i);
        if (testcoor[0]<min_coord[0] || testcoor[0]>max_coord[0]) continue;

        // Choose the direction of translation
        if      (i>0) M[0].loadmatrix(Tpbc_aligned[0][1]);
        else if (i<0) M[0].loadmatrix(Tpbc_aligned[0][0]);
        else          M[0].identity();

        // y
        for (j=-1; j<=1; j++) {
          testcoor[1] = orthcoor[1]+(float)(j);
          if (testcoor[1]<min_coord[1] || testcoor[1]>max_coord[1]) continue;

          // Choose the direction of translation
          if      (j>0) M[1].loadmatrix(Tpbc_aligned[1][1]);
          else if (j<0) M[1].loadmatrix(Tpbc_aligned[1][0]);
          else          M[1].identity();

          // z
          for (k=-1; k<=1; k++) {
            testcoor[2] = orthcoor[2]+(float)(k);
            if (testcoor[2]<min_coord[2] || testcoor[2]>max_coord[2]) continue;

            if(i==0 && j==0 && k==0) continue;

            // Choose the direction of translation
            if      (k>0) M[2].loadmatrix(Tpbc_aligned[2][1]);
            else if (k<0) M[2].loadmatrix(Tpbc_aligned[2][0]);
            else          M[2].identity();

            M[0].multpoint3d(coor, wrapcoor);
            M[1].multpoint3d(wrapcoor, wrapcoor);
            M[2].multpoint3d(wrapcoor, wrapcoor);

            // Atom is inside cutoff, add it to the list            
            extcoord_array->append3(&wrapcoor[0]);
            indexmap_array->append(idx);
          }
        }
      }
    }
  
  } else {
    Matrix4 Mtmp;

    for (i=0; i < ts->num; i++) { 
      // Apply the PBC --> orthonormal unitcell transformation
      // to the current test point.
      M_coretransform.multpoint3d(coords+3L*i, orthcoor);

      // Determine in which cell we are.
      int cellindex[3];    
      if      (orthcoor[0]<0) cellindex[0] = -1;
      else if (orthcoor[0]>1) cellindex[0] =  1;
      else                    cellindex[0] =  0;
      if      (orthcoor[1]<0) cellindex[1] = -1;
      else if (orthcoor[1]>1) cellindex[1] =  1;
      else                    cellindex[1] =  0;
      if      (orthcoor[2]<0) cellindex[2] = -1;
      else if (orthcoor[2]>1) cellindex[2] =  1;
      else                    cellindex[2] =  0;

      // All zero means we're inside the core --> no image.
      if (!cellindex[0] && !cellindex[1] && !cellindex[2]) continue;

      // Choose the direction of translation
      if      (orthcoor[0]<0) M[0].loadmatrix(Tpbc_aligned[0][1]);
      else if (orthcoor[0]>1) M[0].loadmatrix(Tpbc_aligned[0][0]);
      if      (orthcoor[1]<0) M[1].loadmatrix(Tpbc_aligned[1][1]);
      else if (orthcoor[1]>1) M[1].loadmatrix(Tpbc_aligned[1][0]);
      if      (orthcoor[2]<0) M[2].loadmatrix(Tpbc_aligned[2][1]);
      else if (orthcoor[2]>1) M[2].loadmatrix(Tpbc_aligned[2][0]);

      // Create wrapped copies of the atom:
      // x, y, z planes
      coor = coords+3L*i;
      for (u=0; u<3; u++) {
        if (cellindex[u] && cutoff[u]) {
          M[u].multpoint3d(coor, wrapcoor);
          extcoord_array->append3(&wrapcoor[0]);
          indexmap_array->append(i);
        }
      }

      Mtmp = M[0];

      // xy edge
      if (cellindex[0] && cellindex[1] && cutoff[0] && cutoff[1]) {
        M[0].multmatrix(M[1]);
        M[0].multpoint3d(coor, wrapcoor);
        extcoord_array->append3(&wrapcoor[0]);
        indexmap_array->append(i);
      }

      // yz edge
      if (cellindex[1] && cellindex[2] && cutoff[1] && cutoff[2]) {
        M[1].multmatrix(M[2]);
        M[1].multpoint3d(coor, wrapcoor);
        extcoord_array->append3(&wrapcoor[0]);
        indexmap_array->append(i);
      }

      // zx edge
      if (cellindex[0] && cellindex[2] && cutoff[0] && cutoff[2]) {
        M[2].multmatrix(Mtmp);
        M[2].multpoint3d(coor, wrapcoor);
        extcoord_array->append3(&wrapcoor[0]);
        indexmap_array->append(i);
      }

      // xyz corner
      if (cellindex[0] && cellindex[1] && cellindex[2]) {
        M[1].multmatrix(Mtmp);
        M[1].multpoint3d(coor, wrapcoor);
        extcoord_array->append3(&wrapcoor[0]);
        indexmap_array->append(i);
      }

    }

  } // endif

  // If a selection was provided we select extcoords
  // within cutoff of the original selection:
  if (sel) {
    int numext = sel->selected+indexmap_array->num();
    float *extcoords = new float[3L*numext];
    int   *indexmap  = new int[numext];
    int   *others    = new int[numext];
    memset(others, 0, numext);

    // Use the largest given cutoff
    float maxcutoff = cutoff[0];
    for (i=1; i<3; i++) {
      if (cutoff[i]>maxcutoff) maxcutoff = cutoff[i];
    }

    // Prepare C-array of coordinates for find_within()
    j=0;
    for (i=0; i < sel->num_atoms; i++) { 
      if (!sel->on[i]) continue; //atom is not selected
      extcoords[3L*j]   = coords[3L*i];
      extcoords[3L*j+1] = coords[3L*i+1];
      extcoords[3L*j+2] = coords[3L*i+2];
      indexmap[j] = i;
      others[j++] = 1;
    }
    for (i=0; i<indexmap_array->num(); i++) {
      extcoords[3L*j]   = (*extcoord_array)[3L*i];
      extcoords[3L*j+1] = (*extcoord_array)[3L*i+1];
      extcoords[3L*j+2] = (*extcoord_array)[3L*i+2];
      indexmap[j] = (*indexmap_array)[i];
      others[j++] = 0;
    }

    // Initialize flags array to true, find_within() results are AND'd/OR'd in.
    int *flgs   = new int[numext];
    for (i=0; i<numext; i++) {
      flgs[i] = 1;
    }

    // Find coordinates from extcoords that are within cutoff of the ones
    // with flagged in 'others' and set the flgs accordingly:
    find_within(extcoords, flgs, others, numext, maxcutoff);

    extcoord_array->clear();
    indexmap_array->clear();
    for (i=sel->selected; i<numext; i++) {
      if (!flgs[i]) continue;

      extcoord_array->append3(&extcoords[3L*i]);
      indexmap_array->append(indexmap[i]);
    }

  }

  return MEASURE_NOERR;
}  

// Computes the rectangular bounding box for the PBC cell.
// If the molecule was rotated/moved you can supply the transformation
// matrix and you'll get the bounding box of the transformed cell.
int compute_pbcminmax(MoleculeList *mlist, int molid, int frame, 
               const float *center, const Matrix4 *transform,
               float *min, float *max) {
  Molecule *mol = mlist->mol_from_id(molid);
  if( !mol )
    return MEASURE_ERR_NOMOLECULE;

  Timestep *ts = mol->get_frame(frame);
  if (!ts) return MEASURE_ERR_NOFRAMES;

  // Get the displacement vectors (in form of translation matrices)
  Matrix4 Tpbc[3];
  ts->get_transforms(Tpbc[0], Tpbc[1], Tpbc[2]);

  // Construct the cell spanning vectors
  float cell[9];
  cell[0] = Tpbc[0].mat[12];
  cell[1] = Tpbc[0].mat[13];
  cell[2] = Tpbc[0].mat[14];
  cell[3] = Tpbc[1].mat[12];
  cell[4] = Tpbc[1].mat[13];
  cell[5] = Tpbc[1].mat[14];
  cell[6] = Tpbc[2].mat[12];
  cell[7] = Tpbc[2].mat[13];
  cell[8] = Tpbc[2].mat[14];

#if 0
  float len[3];
  len[0] = sqrtf(dot_prod(&cell[0], &cell[0]));
  len[1] = sqrtf(dot_prod(&cell[3], &cell[3]));
  len[2] = sqrtf(dot_prod(&cell[6], &cell[6]));
#endif

  // Construct all 8 corners (nodes) of the bounding box
  float node[8*3];
  int n=0;
  float i, j, k;
  for (i=-0.5; i<1.f; i+=1.f) {
    for (j=-0.5; j<1.f; j+=1.f) {
      for (k=-0.5; k<1.f; k+=1.f) {
        // Apply the translation of the origin
        vec_copy(node+3L*n, center);
        vec_scaled_add(node+3L*n, i, &cell[0]);
        vec_scaled_add(node+3L*n, j, &cell[3]);
        vec_scaled_add(node+3L*n, k, &cell[6]);

        // Apply global alignment transformation
        transform->multpoint3d(node+3L*n, node+3L*n);
        n++;
      }
    }
  }

  // Find minmax coordinates of all corners
  for (n=0; n<8; n++) {
    if (!n || node[3L*n  ]<min[0])  min[0] = node[3L*n];
    if (!n || node[3L*n+1]<min[1])  min[1] = node[3L*n+1];
    if (!n || node[3L*n+2]<min[2])  min[2] = node[3L*n+2];
    if (!n || node[3L*n  ]>max[0])  max[0] = node[3L*n];
    if (!n || node[3L*n+1]>max[1])  max[1] = node[3L*n+1];
    if (!n || node[3L*n+2]>max[2])  max[2] = node[3L*n+2];
  }

  return MEASURE_NOERR;
}
