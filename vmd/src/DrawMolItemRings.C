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
 *	$RCSfile: DrawMolItemRings.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.23 $	$Date: 2010/12/16 04:08:13 $
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

#ifdef VMDWITHCARBS

#include "utilities.h"
#include "DrawMolItem.h"
#include "DrawMolecule.h"
#include "DrawRingsUtils.h"
#include <stdio.h>
#include <math.h>

void DrawMolItem::draw_rings_paperchain(float *framepos, float bipyramid_height, int maxringsize) {
  int i;
  SmallRing *ring;

  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning PaperChain Rings",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  mol->find_small_rings_and_links(mol->currentMaxPathLength, maxringsize);
    
  for (i=0; i < mol->smallringList.num(); i++) {
    ring = mol->smallringList[i];
    if (smallring_selected(*ring))
      paperchain_draw_ring(*ring,framepos,bipyramid_height);
  }
}


// Return true if all atoms in a ring are selected, false otherwise
bool DrawMolItem::smallring_selected(SmallRing &ring) {
  int N = ring.num();
  int i;
    
  for (i=0; i<N; i++)
    if (!atomSel->on[ring[i]]) return false;
  return true;
}


// Return true if all atoms in a ring linkage path are selected, false otherwise
bool DrawMolItem::linkagepath_selected(LinkagePath &path) {
  int N = path.num();
  int i;

  for (i=0; i<N; i++)
    if (!atomSel->on[path[i]]) return false;
  return true;
}


// Calculate the ring color
void DrawMolItem::paperchain_get_ring_color(SmallRing &ring, float *framepos, float *rgb) {
#if 0
  float vmin, vmax;
  atomColor->get_colorscale_minmax(&vmin, &vmax);
  if (!vmin && !vmax) {
    vmin = 0.0;
    vmax = 1.0;
  }

  hill_reilly_ring_colorscale(ring, framepos, vmin, vmax, scene, rgb);
#elif 1
  hill_reilly_ring_color(ring, framepos, rgb);
#else
  cremer_pople_ring_color(ring, framepos, rgb);
#endif
}


void DrawMolItem::get_ring_centroid_and_normal(float *centroid, float *normal, SmallRing &ring, float *framepos) {
  int N = ring.num();
  int i, nexti;
  float curvec[3], nextvec[3];

  // initialize centroid and centroid normal
  centroid[0] = centroid[1] = centroid[2] = 0.0;
  normal[0] = normal[1] = normal[2] = 0.0;

  // calculate centroid and normal
  for (i=0; i<N; i++) {
    // calculate next ring position (wrapping as necessary)
    nexti = i+1;
    if (nexti >= N) 
      nexti = 0;
    
    vec_copy(curvec, framepos + 3*ring[i]);
    vec_copy(nextvec, framepos + 3*ring[nexti]);
        
    // update centroid
    vec_add(centroid, centroid, curvec);
    
    // update normal (this is Newell's method; see Carbohydra paper)
    normal[0] += (curvec[1] - nextvec[1]) * (curvec[2] + nextvec[2]); // (Y_i - Y_next_i) * (Z_i + Z_next_i)
    normal[1] += (curvec[2] - nextvec[2]) * (curvec[0] + nextvec[0]); // (Z_i - Z_next_i) * (X_i + X_next_i)
    normal[2] += (curvec[0] - nextvec[0]) * (curvec[1] + nextvec[1]); // (X_i - X_next_i) * (Y_i + Y_next_i)      
  }

  vec_scale(centroid, 1.0f/N, centroid);
  vec_normalize(normal);
}


void DrawMolItem::paperchain_draw_ring(SmallRing &ring, float *framepos, float bipyramid_height) {
  int N = ring.num(); // the number of atoms in the current ring
  int i, nexti;
  float centroid[3], normal[3], top[3], bottom[3], curvec[3], nextvec[3], x[3];
  float rgb[3];

  paperchain_get_ring_color(ring, framepos, rgb);
  get_ring_centroid_and_normal(centroid, normal, ring, framepos);

  // calculate top and bottom points
  vec_scale(x, 0.5f*bipyramid_height, normal);
  vec_add(top, centroid, x);
  vec_sub(bottom, centroid, x);

  append(DMATERIALON); // turn on lighting

// XXX we should be looping over all of the rings
//     within this routine rather than doing them separately,
//     as we can generate one big triangle mesh from the results
//     eliminating various sources of rendering overhead if we do it right.

#if 1
  // draw triangles
  ResizeArray<float> vertices;
  ResizeArray<float> colors;
  ResizeArray<float> normals;
  ResizeArray<int>   facets;

  // add top/bottom vertices first
  vertices.append(top[0]);
  vertices.append(top[1]);
  vertices.append(top[2]);
  normals.append(normal[0]);
  normals.append(normal[1]);
  normals.append(normal[2]);
  colors.append(rgb[0]);
  colors.append(rgb[1]);
  colors.append(rgb[2]);

  vertices.append(bottom[0]);
  vertices.append(bottom[1]);
  vertices.append(bottom[2]);
  normals.append(normal[0]);
  normals.append(normal[1]);
  normals.append(normal[2]);
  colors.append(rgb[0]);
  colors.append(rgb[1]);
  colors.append(rgb[2]);

  // draw top half
  for (i=0; i<N; i++) {
    // calculate next ring position (wrapping as necessary)
    nexti = i+1;
    if (nexti >= N) nexti = 0;
    
    vec_copy(curvec, framepos + 3*ring[i]);
    vec_copy(nextvec, framepos + 3*ring[nexti]);

    vertices.append(curvec[0]);
    vertices.append(curvec[1]);
    vertices.append(curvec[2]);

    float normtop[3], tmp0[3], tmp1[3];
    vec_sub(tmp0, curvec, nextvec);
    vec_sub(tmp1, nextvec, top);
    cross_prod(normtop, tmp0, tmp1);
    vec_normalize(normtop);
    normals.append(normtop[0]);
    normals.append(normtop[1]);
    normals.append(normtop[2]);

    colors.append(rgb[0]);
    colors.append(rgb[1]);
    colors.append(rgb[2]);

    facets.append(2+i);     // curvec
    facets.append(2+nexti); // nextvec
    facets.append(0);       // top
  }

  // draw bottom half
  for (i=0; i<N; i++) {
    // calculate next ring position (wrapping as necessary)
    nexti = i+1;
    if (nexti >= N) nexti = 0;
    
    vec_copy(curvec, framepos + 3*ring[i]);
    vec_copy(nextvec, framepos + 3*ring[nexti]);

    vertices.append(curvec[0]);
    vertices.append(curvec[1]);
    vertices.append(curvec[2]);

    float normbot[3], tmp0[3], tmp1[3];
    vec_sub(tmp0, curvec, nextvec);
    vec_sub(tmp1, nextvec, bottom);
    cross_prod(normbot, tmp0, tmp1);
    vec_normalize(normbot);
    normals.append(normbot[0]);
    normals.append(normbot[1]);
    normals.append(normbot[2]);

    colors.append(rgb[0]);
    colors.append(rgb[1]);
    colors.append(rgb[2]);

    facets.append(2+N+i);     // curvec
    facets.append(2+N+nexti); // nextvec
    facets.append(1);         // bottom
  }

  // printf("TriMesh N: %d nvert: %d nface: %d  rgb[]=%0.2f,%0.2f,%0.2f\n", N, vertices.num()/3, facets.num()/3, rgb[0], rgb[1], rgb[2]);

  // draw the resulting triangle mesh
  cmdTriMesh.putdata(&vertices[0], &normals[0], &colors[0], vertices.num()/3, 
                     &facets[0], facets.num()/3, 0, cmdList);
#else    
  // draw triangles
  for (i=0; i<N; i++) {
    // calculate next ring position (wrapping as necessary)
    nexti = i+1;
    if (nexti >= N) nexti = 0;
    
    vec_copy(curvec,framepos + 3*ring[i]);
    vec_copy(nextvec,framepos + 3*ring[nexti]);

    cmdTriangle.putdata(curvec, nextvec, top, cmdList);
    cmdTriangle.putdata(curvec, nextvec, bottom, cmdList);
  }
#endif

}


void DrawMolItem::draw_rings_twister(float *framepos, int start_end_centroid, int hide_shared_links,
                                     int rib_steps, float rib_width, float rib_height,
                                     int maxringsize, int maxpathlength) {
  int i;
  LinkagePath *path;
  SmallRing *start_ring, *end_ring;

  sprintf (commentBuffer,"MoleculeID: %d ReprID: %d Beginning Twister Rings",
           mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  mol->find_small_rings_and_links(maxpathlength,maxringsize);

  append(DMATERIALOFF); // turn off lighting

  for (i=0; i < mol->smallringLinkages.paths.num(); i++) {
    path = mol->smallringLinkages.paths[i];
    start_ring = mol->smallringList[path->start_ring];
    end_ring = mol->smallringList[path->end_ring];
    
    if (linkagepath_selected(*path) && smallring_selected(*start_ring) && smallring_selected(*end_ring)) {
      if (!hide_shared_links || !mol->smallringLinkages.sharesLinkageEdges(*path))
        twister_draw_path(*path,framepos, start_end_centroid, rib_steps, rib_width, rib_height);
    }
  }
}


// XXX: Replace simple cubic spline with spline fitted through all
//      atoms in the path.
void DrawMolItem::twister_draw_path(LinkagePath &path, float *framepos, int start_end_centroid, int rib_steps, float rib_width, float rib_height) {
  SmallRing *start_ring = mol->smallringList[path.start_ring];
  SmallRing *end_ring = mol->smallringList[path.end_ring];
  float start_centroid[3], start_normal[3], end_centroid[3], end_normal[3]; // centroids and normals of start and end rings
  float start_pos[3], end_pos[3]; // postions of start and ending atoms
  float start_tangent[3], end_tangent[3]; // temporary vectors
  float start_rib[3], end_rib[3]; // ribbon start and end points
  float rib_interval, rib_inc; // ribbon interval and increment
  float splineA[3], splineB[3], splineC[3], splineD[3]; // spline co-efficients
  float tangentA[3], tangentB[3], tangentC[3], tangentD[3]; // tanget spline co-efficients
  float vectmp1[3], vectmp2[3], ftmp1, ftmp2, ftmp3;
  
  float min_axis_norm = 1e-4f; // Threshold to determine when we have a reliable rotation axis to rotate a frame about
  
  get_ring_centroid_and_normal(start_centroid, start_normal, *start_ring, framepos);
  get_ring_centroid_and_normal(end_centroid, end_normal, *end_ring, framepos);
  vec_copy(start_pos, framepos + 3*path[0]);
  vec_copy(end_pos, framepos + 3*path[path.num()-1]);

  vec_sub(start_tangent, start_pos, start_centroid);
  vec_scale(vectmp1,dot_prod(start_tangent, start_normal), start_normal);
  vec_sub(start_tangent, start_tangent, vectmp1);
  vec_normalize(start_tangent); 

  vec_sub(end_tangent, end_centroid, end_pos); // reversed direction for end tangent
  vec_scale(vectmp1,dot_prod(end_tangent, end_normal), end_normal);
  vec_sub(end_tangent, end_tangent, vectmp1);
  vec_normalize(end_tangent); 

  if (start_end_centroid == 1) {
    // move start and end points towards their centroid
    // and then onto the plane formed by the centroid and
    // the ring normal
    vec_add(start_rib, start_centroid, start_pos);
    vec_scale(start_rib, 0.5, start_rib);
    
    vec_sub(vectmp1, start_rib, start_centroid);
    vec_scale(vectmp1, dot_prod(vectmp1, start_normal), start_normal);
    vec_sub(start_rib, start_rib, vectmp1);

    vec_add(end_rib, end_centroid, end_pos);
    vec_scale(end_rib, 0.5, end_rib);

    vec_sub(vectmp1, end_rib, end_centroid);
    vec_scale(vectmp1, dot_prod(vectmp1, end_normal), end_normal);
    vec_sub(end_rib, end_rib, vectmp1);
  } else {
    vec_copy(start_rib, start_pos);
    vec_copy(end_rib, end_pos);
  }  

  // Use the smoothness-increasing interval length from "Mathematical Elements for Computer Graphics"
  rib_interval = distance(start_rib, end_rib);
  rib_inc = rib_interval / rib_steps;

  // Create ribbon spline
  ftmp1 = 1.0f / rib_interval;

  vec_zero(splineA);
  vec_sub(vectmp1, start_rib, end_rib);
  vec_scaled_add(splineA, 2.0f*ftmp1, vectmp1);
  vec_add(splineA, splineA, start_tangent);
  vec_add(splineA, splineA, end_tangent);
  vec_scale(splineA, ftmp1*ftmp1, splineA);

  vec_zero(splineB);
  vec_sub(vectmp1, end_rib, start_rib);
  vec_scaled_add(splineB, 3.0f*ftmp1, vectmp1);
  vec_scaled_add(splineB, -2.0f, start_tangent);
  vec_scaled_add(splineB, -1.0f, end_tangent);
  vec_scale(splineB, ftmp1, splineB);

  vec_copy(splineC, start_tangent);
  vec_copy(splineD, start_rib);

  // Create ribbon tangent spline
  vec_zero(tangentA);
  vec_scale(tangentB, 3.0f, splineA);
  vec_scale(tangentC, 2.0f, splineB);
  vec_copy(tangentD, splineC);

  // Construct reference frames along ribbon
  ResizeArray<RibbonFrame*> frames;
  RibbonFrame *frame, *prev_frame;
  
  // Initial frame
  frame = new RibbonFrame;
  vec_copy(frame->forward, start_tangent);
  cross_prod(frame->right, start_tangent, start_normal);
  vec_normalize(frame->right);
  cross_prod(frame->up, frame->right, start_tangent);
  vec_copy(frame->origin,start_rib);
  frame->arclength = 0.0f;
  frames.append(frame);

  // Initial estimates for frames
  float new_tangent[3] , rot_axis[3], axis_norm, rot_angle, t;
  int i;

  t = rib_inc;
  for (i=0; i < rib_steps; i++, t+= rib_inc) {
    prev_frame = frames[frames.num() - 1];

    ribbon_spline(new_tangent, tangentA, tangentB, tangentC, tangentD, t);
    vec_normalize(new_tangent);

    cross_prod(rot_axis, prev_frame->forward, new_tangent);
    axis_norm = norm(rot_axis);

    // copy previous frame
    frame = new RibbonFrame;
    vec_copy(frame->forward, prev_frame->forward);
    vec_copy(frame->right, prev_frame->right);
    vec_copy(frame->up, prev_frame->up);
    ribbon_spline(frame->origin, splineA, splineB, splineC, splineD, t);
    frame->arclength = prev_frame->arclength + distance(frame->origin, prev_frame->origin);

    // rotate frame if tangents not parallel
    if (axis_norm > min_axis_norm) {
      vec_normalize(rot_axis);
      rot_angle = acosf(dot_prod(prev_frame->forward, new_tangent));
      
      // Rotate frame angle rot_angle about rot_axis using Rodrigue's formula
      ftmp1 = cosf(rot_angle);
      ftmp2 = sinf(rot_angle);
      ftmp3 = 1.0f - ftmp1;
      
      vec_zero(vectmp1);
      vec_scaled_add(vectmp1, ftmp1, frame->forward);
      vec_scaled_add(vectmp1, ftmp3*dot_prod(rot_axis, frame->forward), rot_axis);
      cross_prod(vectmp2, rot_axis, frame->forward);
      vec_scaled_add(vectmp1, ftmp2, vectmp2);
      vec_copy(frame->forward, vectmp1);
      
      vec_zero(vectmp1);
      vec_scaled_add(vectmp1, ftmp1, frame->right);
      vec_scaled_add(vectmp1, ftmp3*dot_prod(rot_axis, frame->right), rot_axis);
      cross_prod(vectmp2, rot_axis, frame->right);
      vec_scaled_add(vectmp1, ftmp2, vectmp2);
      vec_copy(frame->right, vectmp1);
      
      vec_zero(vectmp1);
      vec_scaled_add(vectmp1, ftmp1, frame->up);
      vec_scaled_add(vectmp1, ftmp3*dot_prod(rot_axis, frame->up), rot_axis);
      cross_prod(vectmp2, rot_axis, frame->up);
      vec_scaled_add(vectmp1, ftmp2, vectmp2);
      vec_copy(frame->up, vectmp1);
    }
    
    frames.append(frame);
  }

  // calculate correct right axis for final frame and from that the correction angle
  float end_right[3], start_right[3], correction_angle, inc_angle, curr_angle;

  vec_copy(start_right, frames[0]->right);
  cross_prod(end_right, end_tangent, end_normal);
  vec_normalize(end_right);
    
  correction_angle = acosf(dot_prod(end_right, frames[frames.num()-1]->right));
  cross_prod(vectmp1, end_right, frames[frames.num()-1]->right);
  if (dot_prod(vectmp1, end_tangent) > 0) 
    correction_angle = -correction_angle;
  
  inc_angle = correction_angle / rib_steps;
  curr_angle = -inc_angle;

  // draw triangles
  ResizeArray<float> vertices;
  ResizeArray<float> colors;
  ResizeArray<float> normals;
  ResizeArray<int>   facets;
  int current_vertex_offset, next_vertex_offset;

  // XXX: Make these colours options in the GUI
  float top_color[3] = { 0.5f, 0.5f, 1.0f };
  float bottom_color[3] = { 0.9f, 0.9f, 0.9f };

  for (i=0; i <= rib_steps; i++) {
    frame = frames[i];
    curr_angle += inc_angle;
  
    // Apply correcting rotation:
    // Rotate frame angle curr_angle about frame->forward using Rodigue's formula
    ftmp1 = cosf(curr_angle);
    ftmp2 = sinf(curr_angle);
    ftmp3 = 1.0f - ftmp1;
      
    vec_zero(vectmp1);
    vec_scaled_add(vectmp1, ftmp1, frame->right);
    vec_scaled_add(vectmp1, ftmp3*dot_prod(frame->forward, frame->right), frame->forward);
    cross_prod(vectmp2, frame->forward, frame->right);
    vec_scaled_add(vectmp1, ftmp2, vectmp2);
    vec_copy(frame->right, vectmp1);

    vec_zero(vectmp1);
    vec_scaled_add(vectmp1, ftmp1, frame->up);
    vec_scaled_add(vectmp1, ftmp3*dot_prod(frame->forward, frame->up),frame->forward);
    cross_prod(vectmp2, frame->forward, frame->up);
    vec_scaled_add(vectmp1, ftmp2, vectmp2);
    vec_copy(frame->up, vectmp1);

    // vertices (of this frame's rectangle)
    vec_copy(vectmp1, frame->origin); // top right (index: +0)
    vec_scaled_add(vectmp1, rib_height, frame->up);
    vec_scaled_add(vectmp1, rib_width, frame->right);     
    vertices.append(vectmp1[0]);
    vertices.append(vectmp1[1]);
    vertices.append(vectmp1[2]);
    colors.append(top_color[0]);
    colors.append(top_color[1]);
    colors.append(top_color[2]);
    vec_add(vectmp1, frame->up, frame->right);
    vec_normalize(vectmp1);
    normals.append(vectmp1[0]);
    normals.append(vectmp1[1]);
    normals.append(vectmp1[2]);

    vec_copy(vectmp1, frame->origin); // bottom right (index: +1)
    vec_scaled_add(vectmp1, -rib_height, frame->up);
    vec_scaled_add(vectmp1, rib_width, frame->right);     
    vertices.append(vectmp1[0]);
    vertices.append(vectmp1[1]);
    vertices.append(vectmp1[2]);
    colors.append(bottom_color[0]);
    colors.append(bottom_color[1]);
    colors.append(bottom_color[2]);
    vec_sub(vectmp1, frame->right, frame->up);
    vec_normalize(vectmp1);
    normals.append(vectmp1[0]);
    normals.append(vectmp1[1]);
    normals.append(vectmp1[2]);

    vec_copy(vectmp1,frame->origin); // bottom left (index: +2)
    vec_scaled_add(vectmp1, -rib_height, frame->up);
    vec_scaled_add(vectmp1, -rib_width, frame->right);     
    vertices.append(vectmp1[0]);
    vertices.append(vectmp1[1]);
    vertices.append(vectmp1[2]);
    colors.append(bottom_color[0]);
    colors.append(bottom_color[1]);
    colors.append(bottom_color[2]);
    vec_add(vectmp1, frame->up, frame->right);
    vec_negate(vectmp1, vectmp1);
    vec_normalize(vectmp1);
    normals.append(vectmp1[0]);
    normals.append(vectmp1[1]);
    normals.append(vectmp1[2]);

    vec_copy(vectmp1,frame->origin); // top left (index: +3)
    vec_scaled_add(vectmp1, rib_height, frame->up);
    vec_scaled_add(vectmp1, -rib_width, frame->right);     
    vertices.append(vectmp1[0]);
    vertices.append(vectmp1[1]);
    vertices.append(vectmp1[2]);
    colors.append(top_color[0]);
    colors.append(top_color[1]);
    colors.append(top_color[2]);
    vec_sub(vectmp1, frame->up, frame->right);
    vec_normalize(vectmp1);
    normals.append(vectmp1[0]);
    normals.append(vectmp1[1]);
    normals.append(vectmp1[2]);

    // facets (between this frame's rectangle and the next's)

    if (i == rib_steps) 
      continue; // no facets for the last frame

    current_vertex_offset = i*4;
    next_vertex_offset = (i+1)*4;

    // top 1
    facets.append(current_vertex_offset + 0); // current, top right
    facets.append(next_vertex_offset + 0);    // next, top right
    facets.append(current_vertex_offset + 3); // current, top left

    // top 2
    facets.append(next_vertex_offset + 0);    // next, top right
    facets.append(next_vertex_offset + 3);    // next, top left
    facets.append(current_vertex_offset + 3); // current, top left

    // bottom 1 
    facets.append(current_vertex_offset + 1); // current, bottom right
    facets.append(current_vertex_offset + 2); // current, bottom left
    facets.append(next_vertex_offset + 1);    // next, bottom right

    // bottom 2
    facets.append(next_vertex_offset + 1);    // next, bottom right
    facets.append(current_vertex_offset + 2); // current, bottom left
    facets.append(next_vertex_offset + 2);    // next, bottom left

    // right 1
    facets.append(current_vertex_offset + 0); // current, top right
    facets.append(current_vertex_offset + 1); // current, bottom right
    facets.append(next_vertex_offset + 0);    // next, top right

    // right 2
    facets.append(next_vertex_offset + 0);    // next, top right
    facets.append(current_vertex_offset + 1); // current, bottom right
    facets.append(next_vertex_offset + 1);    // next, bottom right

    // left 1
    facets.append(current_vertex_offset + 3); // current, top left
    facets.append(next_vertex_offset + 3);    // next, top left
    facets.append(current_vertex_offset + 2); // current, bottom left

    // left 2
    facets.append(next_vertex_offset + 3);    // next, top left
    facets.append(next_vertex_offset + 2);    // next, bottom left
    facets.append(current_vertex_offset + 2); // current, bottom left
  }

  if (start_end_centroid == 1) {
    float first_atom[3];

    // Draw extensions of ribbon to meet hexagonal disks
    twister_draw_ribbon_extensions(vertices, colors, normals, facets,
                     start_centroid, start_normal, start_right, start_rib, rib_height, rib_width, top_color, bottom_color);
    twister_draw_ribbon_extensions(vertices, colors, normals, facets,
                     end_centroid, end_normal, end_right, end_rib, rib_height, rib_width, top_color, bottom_color);

    // Draw hexagonal disks for joining rings
    // XXX: do this only once per ring
    vec_copy(first_atom,framepos + 3*start_ring->first_atom());
    twister_draw_hexagon(vertices, colors, normals, facets,
                     start_centroid, start_normal, first_atom, rib_height, rib_width, top_color, bottom_color);

    vec_copy(first_atom,framepos + 3*end_ring->first_atom());
    twister_draw_hexagon(vertices, colors, normals, facets,
                     end_centroid, end_normal, first_atom, rib_height, rib_width, top_color, bottom_color);
  } else {
    // Draw start and end end caps
    // Start end caps
    current_vertex_offset = 0; 

    facets.append(current_vertex_offset + 3); // top left
    facets.append(current_vertex_offset + 0); // top right
    facets.append(current_vertex_offset + 1); // bottom right
    
    facets.append(current_vertex_offset + 2); // bottom left
    facets.append(current_vertex_offset + 1); // bottom right
    facets.append(current_vertex_offset + 3); // top left

    // End end caps
    current_vertex_offset = rib_steps*4;

    facets.append(current_vertex_offset + 3); // top left
    facets.append(current_vertex_offset + 0); // top right
    facets.append(current_vertex_offset + 1); // bottom right
    
    facets.append(current_vertex_offset + 2); // bottom left
    facets.append(current_vertex_offset + 1); // bottom right
    facets.append(current_vertex_offset + 3); // top left
  }

  // printf("TriMesh - frames: %d ; nvert: %d ; nface: %d\n", frames.num(), vertices.num()/3, facets.num()/3);

  // draw the resulting triangle mesh
  cmdTriMesh.putdata(&vertices[0], &normals[0], &colors[0], vertices.num()/3, 
                     &facets[0], facets.num()/3, 0, cmdList);
}


void DrawMolItem::twister_draw_ribbon_extensions(ResizeArray<float> &vertices, ResizeArray<float> &colors,
                                      ResizeArray<float> &normals, ResizeArray<int> &facets,
                                      float centroid[3], float normal[3], float right[3], float rib_point[3],
                                      float rib_height, float rib_width,
                                      float top_color[3], float bottom_color[3]) {

  float vectmp1[3], norm_tmp[3];
  float* color;
  int first_vertex = vertices.num()/3;  

  float* points[2] = { centroid, rib_point };

  float heights[2] = { rib_height, -rib_height };
  float updown[2] = { 1.0, -1.0 };
  float* side_colors[2] = { top_color, bottom_color };

  float widths[2] = { rib_width, -rib_width };

  // vertices
  for (int height=0; height<2; height++) {
    color = side_colors[height];
    vec_scale(norm_tmp, updown[height], normal);

    for (int point=0; point<2; point++) {
      for (int width=0; width<2; width++) {
        vec_copy(vectmp1, points[point]);
        vec_scaled_add(vectmp1, heights[height], normal);
        vec_scaled_add(vectmp1, width[widths], right);
        vertices.append(vectmp1[0]);
        vertices.append(vectmp1[1]);
        vertices.append(vectmp1[2]);
        colors.append(color[0]);
        colors.append(color[1]);
        colors.append(color[2]);
        normals.append(norm_tmp[0]);
        normals.append(norm_tmp[1]);
        normals.append(norm_tmp[2]);
      }
    }
  }

  // facets
  facets.append(first_vertex + 0); // right
  facets.append(first_vertex + 6);
  facets.append(first_vertex + 4);

  facets.append(first_vertex + 0);
  facets.append(first_vertex + 2);
  facets.append(first_vertex + 6);

  facets.append(first_vertex + 3); // top
  facets.append(first_vertex + 2);
  facets.append(first_vertex + 0);

  facets.append(first_vertex + 3);
  facets.append(first_vertex + 0);
  facets.append(first_vertex + 1);

  facets.append(first_vertex + 1); // left
  facets.append(first_vertex + 5);
  facets.append(first_vertex + 7);

  facets.append(first_vertex + 1);
  facets.append(first_vertex + 7);
  facets.append(first_vertex + 3);

  facets.append(first_vertex + 7); // bottom
  facets.append(first_vertex + 5);
  facets.append(first_vertex + 4);

  facets.append(first_vertex + 7);
  facets.append(first_vertex + 4);
  facets.append(first_vertex + 6);
}


void DrawMolItem::twister_draw_hexagon(ResizeArray<float> &vertices, ResizeArray<float> &colors, ResizeArray<float> &normals,
                                      ResizeArray<int> &facets, float centroid[3], float normal[3],
                                      float first_atom[3], float rib_height, float rib_width,
                                      float top_color[3], float bottom_color[3]) {

  float vectmp1[3];
  int top_centroid_offset, bottom_centroid_offset;
  int current_vertex = vertices.num()/3;

  // centroid vertices
  vec_copy(vectmp1, centroid); // top centroid vertex
  vec_scaled_add(vectmp1, rib_height, normal);
  vertices.append(vectmp1[0]);
  vertices.append(vectmp1[1]);
  vertices.append(vectmp1[2]);
  colors.append(top_color[0]);
  colors.append(top_color[1]);
  colors.append(top_color[2]);
  normals.append(normal[0]);
  normals.append(normal[1]);
  normals.append(normal[2]);
  top_centroid_offset = current_vertex++;

  vec_copy(vectmp1, centroid); // bottom centroid vertex
  vec_scaled_add(vectmp1, -rib_height, normal);
  vertices.append(vectmp1[0]);
  vertices.append(vectmp1[1]);
  vertices.append(vectmp1[2]);
  colors.append(bottom_color[0]);
  colors.append(bottom_color[1]);
  colors.append(bottom_color[2]);
  normals.append(-normal[0]);
  normals.append(-normal[1]);
  normals.append(-normal[2]);
  bottom_centroid_offset = current_vertex++;

  // vertices for hexagon edges

  const int polygon_n = 12;
  const float rot_angle = (float) VMD_TWOPI / polygon_n;
  const int first_edge_offset = current_vertex;

  float current_vec[3], polygon_point[3], vectmp2[3];
  float ftmp1, ftmp2, ftmp3;

  // set current_vec to component of (first_atom - centroid)
  // that is normal to the ring normal and then
  // scale up so that it's wider than the ribbon.
  vec_sub(current_vec, first_atom, centroid);
  vec_scale(vectmp1,dot_prod(current_vec, normal), normal);
  vec_sub(current_vec, current_vec, vectmp1);
  vec_normalize(current_vec);
  vec_scale(current_vec, (1.0f/cosf(rot_angle/2.0f))*rib_width, current_vec);

  for(int i=0;i<polygon_n;i++) {
    vec_add(polygon_point, centroid, current_vec);
    vec_copy(vectmp1, polygon_point); // top hexagon vertex
    vec_scaled_add(vectmp1, rib_height, normal);
    vertices.append(vectmp1[0]);
    vertices.append(vectmp1[1]);
    vertices.append(vectmp1[2]);
    colors.append(top_color[0]);
    colors.append(top_color[1]);
    colors.append(top_color[2]);
    normals.append(normal[0]);
    normals.append(normal[1]);
    normals.append(normal[2]);
    current_vertex++;

    vec_copy(vectmp1, polygon_point); // bottom hexagon vertex
    vec_scaled_add(vectmp1, -rib_height, normal);
    vertices.append(vectmp1[0]);
    vertices.append(vectmp1[1]);
    vertices.append(vectmp1[2]);
    colors.append(bottom_color[0]);
    colors.append(bottom_color[1]);
    colors.append(bottom_color[2]);
    normals.append(-normal[0]);
    normals.append(-normal[1]);
    normals.append(-normal[2]);
    current_vertex++;

    if (i == polygon_n-1) 
      break; // don't bother rotating the last time

    // Rotate current vec pi/3 radians about the ring normal using Rodrigue's formula
    ftmp1 = cosf(rot_angle);
    ftmp2 = sinf(rot_angle);
    ftmp3 = 1.0f - ftmp1;
      
    vec_zero(vectmp1);
    vec_scaled_add(vectmp1, ftmp1, current_vec);
    vec_scaled_add(vectmp1, ftmp3*dot_prod(normal, current_vec), normal);
    cross_prod(vectmp2, normal, current_vec);
    vec_scaled_add(vectmp1, ftmp2, vectmp2);
    vec_copy(current_vec, vectmp1);    
  }

  // facets
  int edge1_top, edge1_bottom, edge2_top, edge2_bottom;
  for(int j=0;j<polygon_n;j++) {
    edge1_top = first_edge_offset + j*2;
    edge1_bottom = edge1_top + 1;
    if (j<polygon_n-1) {
      edge2_top = edge1_top + 2;
      edge2_bottom = edge1_top + 3;
    }
    else {
      edge2_top = first_edge_offset;
      edge2_bottom = first_edge_offset + 1;
    }
  
    // top
    facets.append(top_centroid_offset);
    facets.append(edge1_top);
    facets.append(edge2_top);
   
    // bottom
    facets.append(bottom_centroid_offset);
    facets.append(edge2_bottom);
    facets.append(edge1_bottom);
    
    // outer edge
    facets.append(edge1_bottom);
    facets.append(edge2_bottom);
    facets.append(edge1_top);

    facets.append(edge1_top);
    facets.append(edge2_bottom);
    facets.append(edge2_top);
  }
}



#endif
