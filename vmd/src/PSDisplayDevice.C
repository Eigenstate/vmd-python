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
 *      $RCSfile: PSDisplayDevice.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.112 $       $Date: 2012/03/01 16:50:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *  We can render to Postscript!   Yippee!!
 *
 ***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "DepthSortObj.h"
#include "Matrix4.h"
#include "PSDisplayDevice.h"
#include "VMDDisplayList.h"
#include "Inform.h"


PSDisplayDevice::PSDisplayDevice(void)
: FileRenderer ("PostScript", "PostScript (vector graphics)", "vmdscene.ps","ghostview %s &") {
   memerror = 0;
   x_offset = 306;
   y_offset = 396;

   // initialize some variables used to cache a triangle mesh
   // approximation of a unit sphere
   sph_iter = -1;
   sph_desired_iter = 0;
   sph_nverts = 0;
   sph_verts = NULL;

   memusage = 0;
   points = 0;
   objects = 0;
}


PSDisplayDevice::~PSDisplayDevice(void) {
   // if necessary, free any memory used in caching the
   // unit sphere (see sphere_approx())
   if (sph_nverts && sph_verts) free(sph_verts);
}


void PSDisplayDevice::render(const VMDDisplayList *display_list) {
   DepthSortObject depth_obj;
   char *cmd_ptr;
   int draw;
   int tok;
   int nc;
   float a[3], b[3], c[3], d[3];
   float cent[3];
   float r;
   Matrix4 ident;
   float textsize=1.0f; // default text size

   // first we want to clear the transformation matrix stack
   while (transMat.num())
      transMat.pop();

   // push on the identity matrix
   transMat.push(ident);

   // load the display list's transformation matrix
   super_multmatrix(display_list->mat.mat);

   // Now we need to calculate the normalized position of the light
   // so we can compute angles of surfaces to that light for shading
   norm_light[0] = lightState[0].pos[0];
   norm_light[1] = lightState[0].pos[1];
   norm_light[2] = lightState[0].pos[2];
   if (norm_light[0] || norm_light[1] || norm_light[2])
      vec_normalize(norm_light);

   // Computer periodic images
   ResizeArray<Matrix4> pbcImages;
   find_pbc_images(display_list, pbcImages);
   int nimages = pbcImages.num();

   for (int pbcimage = 0; pbcimage < nimages; pbcimage++) {
     transMat.dup();
     super_multmatrix(pbcImages[pbcimage].mat);

   // Loop through the display list and add each object to our
   // depth-sort list for final rendering.
   VMDDisplayList::VMDLinkIter cmditer;
   display_list->first(&cmditer);
   while ((tok = display_list->next(&cmditer, cmd_ptr)) != DLASTCOMMAND) {
      draw = 0;
      nc = -1;

      switch (tok) {
         case DPOINT:
            // allocate memory
            depth_obj.points = (float *) malloc(sizeof(float) * 2);
            if (!depth_obj.points) {
               // memory error
               if (!memerror) {
                  memerror = 1;
                  msgErr << "PSDisplayDevice: Out of memory. Some " <<
                     "objects were not drawn." << sendmsg;
               }
               break;
            }

            // copy data
            depth_obj.npoints = 1;
            depth_obj.color = colorIndex;
            (transMat.top()).multpoint3d(((DispCmdPoint *) cmd_ptr)->pos, a);
            memcpy(depth_obj.points, a, sizeof(float) * 2);

            // compute the distance to the eye
            depth_obj.dist = compute_dist(a);

            // valid object to depth sort
            draw = 1;
            break;

         case DSPHERE:
         {
            (transMat.top()).multpoint3d(((DispCmdSphere *) cmd_ptr)->pos_r, c);
            r = scale_radius(((DispCmdSphere *) cmd_ptr)->pos_r[3]);

            sphere_approx(c, r);
            break;
         }

         case DSPHEREARRAY:
         {
            DispCmdSphereArray *sa = (DispCmdSphereArray *) cmd_ptr;
            int cIndex, rIndex; // cIndex: index of colors & centers
                                // rIndex: index of radii.
            float * centers;
            float * radii;
            float * colors;
            sa->getpointers(centers, radii, colors);

            set_sphere_res(sa->sphereres);

            for (cIndex = 0, rIndex=0; rIndex < sa->numspheres; 
                 cIndex+=3, rIndex++)
            {
                colorIndex = nearest_index(colors[cIndex],
                                           colors[cIndex+1],
                                           colors[cIndex+2]);
                (transMat.top()).multpoint3d(&centers[cIndex] , c);
                r = scale_radius(radii[rIndex]);

                sphere_approx(c, r);
            }

            break;
         }   

         case DLINE:
            // check for zero-length line (degenerate)
            if (!memcmp(((DispCmdLine *) cmd_ptr)->pos1,
                        ((DispCmdLine *) cmd_ptr)->pos2,
                        sizeof(float) * 3)) {
               // degenerate line
               break;
            }

            // allocate memory
            depth_obj.points = (float *) malloc(sizeof(float) * 4);
            if (!depth_obj.points) {
               // memory error
               if (!memerror) {
                  memerror = 1;
                  msgErr << "PSDisplayDevice: Out of memory. Some " <<
                     "objects were not drawn." << sendmsg;
               }
               break;
            }

            // copy data
            depth_obj.npoints = 2;
            depth_obj.color = colorIndex;
            (transMat.top()).multpoint3d(((DispCmdLine *) cmd_ptr)->pos1, a);
            (transMat.top()).multpoint3d(((DispCmdLine *) cmd_ptr)->pos2, b);
            memcpy(depth_obj.points, a, sizeof(float) * 2);
            memcpy(&depth_obj.points[2], b, sizeof(float) * 2);

            // compute the centerpoint of the object
            cent[0] = (a[0] + b[0]) / 2;
            cent[1] = (a[1] + b[1]) / 2;
            cent[2] = (a[2] + b[2]) / 2;

            // compute the distance to the eye
            depth_obj.dist = compute_dist(cent);

            // valid object to depth sort
            draw = 1;
            break;

         case DLINEARRAY:
           {
            // XXX much replicated code from DLINE
            float *v = (float *)cmd_ptr;
            int nlines = (int)v[0];
            v++;
            for (int i=0; i<nlines; i++) {
              // check for degenerate line
              if (!memcmp(v,v+3,3*sizeof(float)))
                break;

              // allocate memory
              depth_obj.points = (float *) malloc(sizeof(float) * 4);
              if (!depth_obj.points) {
                 // memory error
                 if (!memerror) {
                    memerror = 1;
                    msgErr << "PSDisplayDevice: Out of memory. Some " <<
                       "objects were not drawn." << sendmsg;
                 }
                 break;
              }

              // copy data
              depth_obj.npoints = 2;
              depth_obj.color = colorIndex;
              (transMat.top()).multpoint3d(v, a);
              (transMat.top()).multpoint3d(v+3, b);
              memcpy(depth_obj.points, a, sizeof(float) * 2);
              memcpy(&depth_obj.points[2], b, sizeof(float) * 2);
  
              // compute the centerpoint of the object
              cent[0] = (a[0] + b[0]) / 2;
              cent[1] = (a[1] + b[1]) / 2;
              cent[2] = (a[2] + b[2]) / 2;
  
              // compute the distance to the eye
              depth_obj.dist = compute_dist(cent);
 
              // we'll add the object here, since we have multiple objects 
              draw = 0;
              memusage += sizeof(float) * 2 * depth_obj.npoints;
              points += depth_obj.npoints;
              objects++;
              depth_list.append(depth_obj);

              v += 6;
            } 
           }
           break;           

         case DPOLYLINEARRAY:
           {
            // XXX much replicated code from DLINE / DLINEARRAY
            float *v = (float *)cmd_ptr;
            int nverts = (int)v[0];
            v++;
            for (int i=0; i<nverts-1; i++) {
              // check for degenerate line
              if (!memcmp(v,v+3,3*sizeof(float)))
                break;

              // allocate memory
              depth_obj.points = (float *) malloc(sizeof(float) * 4);
              if (!depth_obj.points) {
                 // memory error
                 if (!memerror) {
                    memerror = 1;
                    msgErr << "PSDisplayDevice: Out of memory. Some " <<
                       "objects were not drawn." << sendmsg;
                 }
                 break;
              }

              // copy data
              depth_obj.npoints = 2;
              depth_obj.color = colorIndex;
              (transMat.top()).multpoint3d(v, a);
              (transMat.top()).multpoint3d(v+3, b);
              memcpy(depth_obj.points, a, sizeof(float) * 2);
              memcpy(&depth_obj.points[2], b, sizeof(float) * 2);
  
              // compute the centerpoint of the object
              cent[0] = (a[0] + b[0]) / 2;
              cent[1] = (a[1] + b[1]) / 2;
              cent[2] = (a[2] + b[2]) / 2;
  
              // compute the distance to the eye
              depth_obj.dist = compute_dist(cent);
 
              // we'll add the object here, since we have multiple objects 
              draw = 0;
              memusage += sizeof(float) * 2 * depth_obj.npoints;
              points += depth_obj.npoints;
              objects++;
              depth_list.append(depth_obj);

              v += 3;
            } 
           }
           break;           
 
         case DCYLINDER:
         {
            int res;

            (transMat.top()).multpoint3d((float *) cmd_ptr, a);
            (transMat.top()).multpoint3d(&((float *) cmd_ptr)[3], b);
            r = scale_radius(((float *) cmd_ptr)[6]);
            res = (int) ((float *) cmd_ptr)[7];

            cylinder_approx(a, b, r, res, (int) ((float *) cmd_ptr)[8]);
            break;
         }

         case DCONE:
         {
            (transMat.top()).multpoint3d(((DispCmdCone *) cmd_ptr)->pos1, a);
            (transMat.top()).multpoint3d(((DispCmdCone *) cmd_ptr)->pos2, b);
            float r1 = scale_radius(((DispCmdCone *) cmd_ptr)->radius);
            float r2 = scale_radius(((DispCmdCone *) cmd_ptr)->radius2);

            // XXX current implementation can't draw truncated cones.
            if (r2 > 0.0f) {
              msgWarn << "PSDisplayDevice) can't draw truncated cones" 
                      << sendmsg;
            }
            cone_approx(a, b, r1);
            break;
         }

        case DTEXTSIZE:
          textsize = ((DispCmdTextSize *)cmd_ptr)->size;
          break;

        case DTEXT:
        {
            float* pos = (float *)cmd_ptr;
#if 0
            // thickness not implemented yet
            float thickness = pos[3];    // thickness is stored in 4th slot
#endif
            char* txt = (char *)(pos+4);
            int   txtlen = strlen(txt);
            // allocate memory
            depth_obj.points = (float *) malloc(sizeof(float) * 2);
            depth_obj.text   = (char  *) malloc(sizeof(char) * (txtlen+1));
            if ( !(depth_obj.points || depth_obj.text) ) {
              // memory error
              if (!memerror) {
                 memerror = 1;
                 msgErr << "PSDisplayDevice: Out of memory. Some " <<
                    "objects were not drawn." << sendmsg;
              }
              break;
           }

            // copy data
            depth_obj.npoints = 1;
            depth_obj.color = colorIndex;
            (transMat.top()).multpoint3d(((DispCmdPoint *) cmd_ptr)->pos, a);
            memcpy(depth_obj.points, a, sizeof(float) * 2);
            strcpy(depth_obj.text , txt);

            // note scale factor, stored into "light_scale", so we didn't
            // have to add a new structure member just for this.
            depth_obj.light_scale = textsize * 15;

            // compute the distance to the eye
            depth_obj.dist = compute_dist(a);

            // valid object to depth sort
            draw = 1;
            break;
         }

         case DTRIANGLE:
            // check for degenerate triangle
            if (!memcmp(((DispCmdTriangle *) cmd_ptr)->pos1,
                        ((DispCmdTriangle *) cmd_ptr)->pos2,
                        sizeof(float) * 3) ||
                !memcmp(((DispCmdTriangle *) cmd_ptr)->pos2,
                        ((DispCmdTriangle *) cmd_ptr)->pos3,
                        sizeof(float) * 3) ||
                !memcmp(((DispCmdTriangle *) cmd_ptr)->pos2,
                        ((DispCmdTriangle *) cmd_ptr)->pos3,
                        sizeof(float) * 3)) {
               // degenerate triangle
               break;
            }

            // allocate memory
            depth_obj.points = (float *) malloc(sizeof(float) * 6);
            if (!depth_obj.points) {
               // memory error
               if (!memerror) {
                  memerror = 1;
                  msgErr << "PSDisplayDevice: Out of memory. Some " <<
                     "objects were not drawn." << sendmsg;
               }
               break;
            }

            // copy data
            depth_obj.npoints = 3;
            depth_obj.color = (nc >= 0) ? nc : colorIndex;
            (transMat.top()).multpoint3d(((DispCmdTriangle *) cmd_ptr)->pos1, a);
            (transMat.top()).multpoint3d(((DispCmdTriangle *) cmd_ptr)->pos2, b);
            (transMat.top()).multpoint3d(((DispCmdTriangle *) cmd_ptr)->pos3, c);
            memcpy(depth_obj.points, a, sizeof(float) * 2);
            memcpy(&depth_obj.points[2], b, sizeof(float) * 2);
            memcpy(&depth_obj.points[4], c, sizeof(float) * 2);

            // compute the centerpoint of the object
            cent[0] = (a[0] + b[0] + c[0]) / 3;
            cent[1] = (a[1] + b[1] + c[1]) / 3;
            cent[2] = (a[2] + b[2] + c[2]) / 3;

            // compute the distance to the eye for depth sorting
            depth_obj.dist = compute_dist(cent);

            // compute a light shading factor
            depth_obj.light_scale = compute_light(a, b, c);

            // valid object to depth sort
            draw = 1;
            break;

         case DTRIMESH_C4F_N3F_V3F:
            // call a separate routine to break up the mesh into
            // its component triangles
            decompose_mesh((DispCmdTriMesh *) cmd_ptr);
            break;

         case DTRISTRIP:
            // call a separate routine to break up the strip into
            // its component triangles
            decompose_tristrip((DispCmdTriStrips *) cmd_ptr);
            break;

         case DSQUARE:
            // check for degenerate quadrilateral
            if (!memcmp(((DispCmdSquare *) cmd_ptr)->pos1,
                        ((DispCmdSquare *) cmd_ptr)->pos2,
                        sizeof(float) * 3) ||
                !memcmp(((DispCmdSquare *) cmd_ptr)->pos1,
                        ((DispCmdSquare *) cmd_ptr)->pos3,
                        sizeof(float) * 3) ||
                !memcmp(((DispCmdSquare *) cmd_ptr)->pos1,
                        ((DispCmdSquare *) cmd_ptr)->pos4,
                        sizeof(float) * 3) ||
                !memcmp(((DispCmdSquare *) cmd_ptr)->pos2,
                        ((DispCmdSquare *) cmd_ptr)->pos3,
                        sizeof(float) * 3) ||
                !memcmp(((DispCmdSquare *) cmd_ptr)->pos2,
                        ((DispCmdSquare *) cmd_ptr)->pos4,
                        sizeof(float) * 3) ||
                !memcmp(((DispCmdSquare *) cmd_ptr)->pos3,
                        ((DispCmdSquare *) cmd_ptr)->pos4,
                        sizeof(float) * 3)) {
               // degenerate quadrilateral
               break;
            }

            // allocate memory
            depth_obj.points = (float *) malloc(sizeof(float) * 8);
            if (!depth_obj.points) {
               // memory error
               if (!memerror) {
                  memerror = 1;
                  msgErr << "PSDisplayDevice: Out of memory. Some " <<
                     "objects were not drawn." << sendmsg;
               }
               break;
            }

            // copy data
            depth_obj.npoints = 4;
            depth_obj.color = colorIndex;
            (transMat.top()).multpoint3d(((DispCmdSquare *) cmd_ptr)->pos1, a);
            (transMat.top()).multpoint3d(((DispCmdSquare *) cmd_ptr)->pos2, b);
            (transMat.top()).multpoint3d(((DispCmdSquare *) cmd_ptr)->pos3, c);
            (transMat.top()).multpoint3d(((DispCmdSquare *) cmd_ptr)->pos4, d);
            memcpy(depth_obj.points, a, sizeof(float) * 2);
            memcpy(&depth_obj.points[2], b, sizeof(float) * 2);
            memcpy(&depth_obj.points[4], c, sizeof(float) * 2);
            memcpy(&depth_obj.points[6], d, sizeof(float) * 2);

            // compute the centerpoint of the object
            cent[0] = (a[0] + b[0] + c[0] + d[0]) / 4;
            cent[1] = (a[1] + b[1] + c[1] + d[1]) / 4;
            cent[2] = (a[2] + b[2] + c[2] + d[2]) / 4;

            // compute the distance to the eye for depth sorting
            depth_obj.dist = compute_dist(cent);

            // compute a light shading factor
            depth_obj.light_scale = compute_light(a, b, c);

            // valid object to depth sort
            draw = 1;
            break;

         case DCOLORINDEX:
            colorIndex = ((DispCmdColorIndex *) cmd_ptr)->color;
            break;

         case DSPHERERES:
            set_sphere_res(((int *) cmd_ptr)[0]);
            break;

         default:
            // unknown object, so just skip it
            break;
      }

      // if we have a valid object to add to the depth sort list
      if (draw && depth_obj.npoints) {
         memusage += sizeof(float) * 2 * depth_obj.npoints;
         if ( depth_obj.text )
           memusage += sizeof(char) * (1+strlen(depth_obj.text));
         points += depth_obj.npoints;
         objects++;
         depth_list.append(depth_obj);
      }

      depth_obj.npoints = 0;
      depth_obj.points = NULL;

   } // while (tok != DLASTCOMMAND)

     transMat.pop();
   } // end for() [periodic images]
}


// This is called after all molecules to be displayed have been rendered.
// We need to depth sort our list and then render each one at a time,
// we also need to first define all the PostScript functions to handle
// triangles and quadrilaterals
void PSDisplayDevice::render_done() {
   x_scale = 1.33f * 792 / Aspect / vSize;
   y_scale = x_scale;

   msgInfo << "PSDisplayDevice: peak memory totals: " << sendmsg;
   msgInfo << "    total dynamic memory used: " <<
      (long) (memusage + sizeof(DepthSortObject) * objects) << sendmsg;
   msgInfo << "    total dynamic points: " << points << sendmsg;
   msgInfo << "    total depthsorted object: " << objects << sendmsg;

   if (depth_list.num()) {
      depth_list.qsort(0, depth_list.num() - 1);
      process_depth_list();
   }
}


void PSDisplayDevice::process_depth_list(void) {
   DepthSortObject obj;
   int i, nobjs;

   nobjs = depth_list.num();
   float textsize = -20;
   for (i = 0; i < nobjs; i++) {
      obj = depth_list.item(i);

      if (obj.text) {
        // check to see if we have to output a new scaling factor
        // in the generated postscript file, only output if it has
        // changed.
	if (obj.light_scale != textsize) {
	  textsize = obj.light_scale;
	  fprintf(outfile, "%f ts\n", textsize);
	}
	fprintf(outfile, "%d 1 c (%s) %d %d text\n",
		obj.color,
		obj.text,
		(int) (obj.points[0] * x_scale + x_offset),
		(int) (obj.points[1] * y_scale + y_offset));
      } else {
	switch (obj.npoints) {
	  case 1: // point
            fprintf(outfile, "%d 1 c %d %d p\n",
		    obj.color,
		    (int) (obj.points[0] * x_scale + x_offset),
		    (int) (obj.points[1] * y_scale + y_offset));
            break;
	    
	  case 2: // line
            fprintf(outfile, "%d 1 c %d %d %d %d l\n",
		    obj.color,
		    (int) (obj.points[0] * x_scale + x_offset),
		    (int) (obj.points[1] * y_scale + y_offset),
		    (int) (obj.points[2] * x_scale + x_offset),
		    (int) (obj.points[3] * y_scale + y_offset));
            break;
	    
	  case 3: // triangle
            fprintf(outfile, "%d %.2f c %d %d %d %d %d %d t\n",
		    obj.color, obj.light_scale,
		    (int) (obj.points[0] * x_scale + x_offset),
		    (int) (obj.points[1] * y_scale + y_offset),
		    (int) (obj.points[2] * x_scale + x_offset),
		    (int) (obj.points[3] * y_scale + y_offset),
		    (int) (obj.points[4] * x_scale + x_offset),
		    (int) (obj.points[5] * y_scale + y_offset));
            break;
	    
         case 4: // quadrilateral
	   fprintf(outfile, "%d %.2f c %d %d %d %d %d %d %d %d s\n",
		   obj.color, obj.light_scale,
		   (int) (obj.points[0] * x_scale + x_offset),
		   (int) (obj.points[1] * y_scale + y_offset),
		   (int) (obj.points[2] * x_scale + x_offset),
		   (int) (obj.points[3] * y_scale + y_offset),
		   (int) (obj.points[4] * x_scale + x_offset),
		   (int) (obj.points[5] * y_scale + y_offset),
		   (int) (obj.points[6] * x_scale + x_offset),
		   (int) (obj.points[7] * y_scale + y_offset));
	   break;
	}
      }

      // free up the memory we've used
      memusage -= sizeof(float) * 2 * obj.npoints;
      if (obj.npoints) free(obj.points);
      if (obj.text) {
        memusage -= sizeof(char) * (1+strlen(obj.text));
        free(obj.text);
      }
   }

   // put the finishing touches on the Postscript output...
   fprintf(outfile, "showpage\n");
   close_file();

   // finally, clear the depth sorted list
   depth_list.remove(-1, -1);

   msgInfo << "PSDisplayDevice: end memory summary:" << sendmsg;
   msgInfo << "    total dynamic memory used: " << memusage << sendmsg;
   msgInfo << "    total dynamic points: " << points << sendmsg;
   msgInfo << "    total depthsorted object: " << objects << sendmsg;

   // reset the memory totals
   memusage = 0;
   objects = 0;
   points = 0;

   // and hooray, we're done!
}


void PSDisplayDevice::set_sphere_res(int res)
{
    // the sphere resolution has changed. if sphereRes is less than 32, we
    // will use a lookup table to achieve equal or better resolution than
    // OpenGL. otherwise we use the following equation:
    //    iterations = .9 *
    //    (sphereRes)^(1/2)

    // this is used as a lookup table to determine the proper
    // number of iterations used in the sphere approximation
    // algorithm.
    const int sph_iter_table[] = {
        0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4 };

    if (res < 0) return;
    else if (res < 32) sph_desired_iter = sph_iter_table[res];
    else sph_desired_iter = (int) (0.8f * sqrtf((float) res));
}


void PSDisplayDevice::sphere_approx(float *c, float r) {
   DepthSortObject depth_obj;
   float x[3], y[3], z[3];
   float cent[3];
   int pi, ni;
   int i;

   // first we need to determine if a recalculation of the cached
   // unit sphere is necessary. this is necessary if the number
   // of desired iterations has changed.
   if (!sph_verts || !sph_nverts || sph_iter != sph_desired_iter) {
      float a[3], b[3], c[3];
      float *newverts;
      float *oldverts;
      int nverts, ntris;
      int level;

      // remove old cached copy
      if (sph_verts && sph_nverts) free(sph_verts);

      // XXX TODO it should be possible here to use the old
      // sphere as an aid in calculating the new sphere. in
      // this manner we can save calculations during resolution
      // changes.

      newverts = (float *) malloc(sizeof(float) * 36);
      nverts = 12;
      ntris = 4;

      // start with half of a unit octahedron (front, convex half)

      // top left triangle
      newverts[0] = -1;    newverts[1] = 0;     newverts[2] = 0;
      newverts[3] = 0;     newverts[4] = 1;     newverts[5] = 0;
      newverts[6] = 0;     newverts[7] = 0;     newverts[8] = 1;

      // top right triangle
      newverts[9] = 0;     newverts[10] = 0;    newverts[11] = 1;
      newverts[12] = 0;    newverts[13] = 1;    newverts[14] = 0;
      newverts[15] = 1;    newverts[16] = 0;    newverts[17] = 0;

      // bottom right triangle
      newverts[18] = 0;    newverts[19] = 0;    newverts[20] = 1;
      newverts[21] = 1;    newverts[22] = 0;    newverts[23] = 0;
      newverts[24] = 0;    newverts[25] = -1;   newverts[26] = 0;

      // bottom left triangle
      newverts[27] = 0;    newverts[28] = 0;    newverts[29] = 1;
      newverts[30] = 0;    newverts[31] = -1;   newverts[32] = 0;
      newverts[33] = -1;   newverts[34] = 0;    newverts[35] = 0;

      for (level = 1; level < sph_desired_iter; level++) {
         oldverts = newverts;

         // allocate memory for the next iteration: we will need
         // four times the current number of vertices
         newverts = (float *) malloc(sizeof(float) * 12 * nverts);
         if (!newverts) {
            // memory error
            sph_iter = -1;
            sph_nverts = 0;
            sph_verts = NULL;
            free(oldverts);

            if (!memerror) {
               memerror = 1;
               msgErr << "PSDisplayDevice: Out of memory. Some " 
                      << "objects were not drawn." << sendmsg;
            }

            return;
         }

         pi = 0;
         ni = 0;
         for (i = 0; i < ntris; i++) {
            // compute intermediate vertices
            a[0] = (oldverts[pi    ] + oldverts[pi + 6]) / 2;
            a[1] = (oldverts[pi + 1] + oldverts[pi + 7]) / 2;
            a[2] = (oldverts[pi + 2] + oldverts[pi + 8]) / 2;
            vec_normalize(a);
            b[0] = (oldverts[pi    ] + oldverts[pi + 3]) / 2;
            b[1] = (oldverts[pi + 1] + oldverts[pi + 4]) / 2;
            b[2] = (oldverts[pi + 2] + oldverts[pi + 5]) / 2;
            vec_normalize(b);
            c[0] = (oldverts[pi + 3] + oldverts[pi + 6]) / 2;
            c[1] = (oldverts[pi + 4] + oldverts[pi + 7]) / 2;
            c[2] = (oldverts[pi + 5] + oldverts[pi + 8]) / 2;
            vec_normalize(c);

            // build triangles
            memcpy(&newverts[ni     ], &oldverts[pi], sizeof(float) * 3);
            memcpy(&newverts[ni + 3 ], b, sizeof(float) * 3);
            memcpy(&newverts[ni + 6 ], a, sizeof(float) * 3);

            memcpy(&newverts[ni + 9 ], b, sizeof(float) * 3);
            memcpy(&newverts[ni + 12], &oldverts[pi + 3], sizeof(float) * 3);
            memcpy(&newverts[ni + 15], c, sizeof(float) * 3);

            memcpy(&newverts[ni + 18], a, sizeof(float) * 3);
            memcpy(&newverts[ni + 21], b, sizeof(float) * 3);
            memcpy(&newverts[ni + 24], c, sizeof(float) * 3);

            memcpy(&newverts[ni + 27], a, sizeof(float) * 3);
            memcpy(&newverts[ni + 30], c, sizeof(float) * 3);
            memcpy(&newverts[ni + 33], &oldverts[pi + 6], sizeof(float) * 3);

            pi += 9;
            ni += 36;
         }

         free(oldverts);

         nverts *= 4;
         ntris *= 4;
      }

      sph_iter = sph_desired_iter;
      sph_nverts = nverts;
      sph_verts = newverts;
   }

   // now we're guaranteed to have a valid cached unit sphere, so
   // all we need to do is translate each coordinate based on the
   // desired position and radius, and add the triangles to the
   // depth sort list.
#if 0
   if (!points) {
      // memory error
      if (!memerror) {
         memerror = 1;
         msgErr << "PSDisplayDevice: Out of memory. Some " <<
            "objects were not drawn." << sendmsg;
      }
      return;
   }
#endif

   // perform the desired translations and scalings on each
   // vertex, then add each triangle to the depth sort list
   depth_obj.npoints = 3;
   depth_obj.color = colorIndex;

   pi = 0;
   for (i = 0; i < sph_nverts / 3; i++) {
      // allocate memory for the triangle
      depth_obj.points = (float *) malloc(sizeof(float) * 6);
      if (!depth_obj.points) {
         // memory error
         if (!memerror) {
            memerror = 1;
            msgErr << "PSDisplayDevice: Out of memory. Some " 
                   << "objects were not drawn." << sendmsg;
         }
         return;
      }

      // translations and scalings
      x[0] = r * sph_verts[pi] + c[0];
      x[1] = r * sph_verts[pi + 1] + c[1];
      x[2] = r * sph_verts[pi + 2] + c[2];
      y[0] = r * sph_verts[pi + 3] + c[0];
      y[1] = r * sph_verts[pi + 4] + c[1];
      y[2] = r * sph_verts[pi + 5] + c[2];
      z[0] = r * sph_verts[pi + 6] + c[0];
      z[1] = r * sph_verts[pi + 7] + c[1];
      z[2] = r * sph_verts[pi + 8] + c[2];

      memcpy(depth_obj.points, x, sizeof(float) * 2);
      memcpy(&depth_obj.points[2], y, sizeof(float) * 2);
      memcpy(&depth_obj.points[4], z, sizeof(float) * 2);

      // now need to compute centerpoint and distance to eye
      cent[0] = (x[0] + y[0] + z[0]) / 3;
      cent[1] = (x[1] + y[1] + z[1]) / 3;
      cent[2] = (x[2] + y[2] + z[2]) / 3;
      depth_obj.dist = compute_dist(cent);
      depth_obj.light_scale = compute_light(x, y, z);

      // and add to the depth sort list
      memusage += sizeof(float) * 2 * depth_obj.npoints;
      points += depth_obj.npoints;
      objects++;
      depth_list.append(depth_obj);

      pi += 9;
   }
}


void PSDisplayDevice::cylinder_approx(float *a, float *b, float r, int res, 
                                      int filled) {

   float axis[3];
   float perp1[3], perp2[3];
   float pt1[3], pt2[3];
   float cent[3];
   float theta, theta_inc;
   float my_sin, my_cos;
   float w[3], x[3], y[3], z[3];
   int n;

   DepthSortObject cyl_body, cyl_trailcap, cyl_leadcap;

   // check against degenerate cylinder
   if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) return;
   if (r <= 0) return;

   // first we compute the axis of the cylinder
   axis[0] = b[0] - a[0];
   axis[1] = b[1] - a[1];
   axis[2] = b[2] - a[2];
   vec_normalize(axis);

   // now we compute some arbitrary perpendicular to that axis
   if ((ABS(axis[0]) < ABS(axis[1])) &&
       (ABS(axis[0]) < ABS(axis[2]))) {
      perp1[0] = 0;
      perp1[1] = axis[2];
      perp1[2] = -axis[1];
   }
   else if ((ABS(axis[1]) < ABS(axis[2]))) {
      perp1[0] = -axis[2];
      perp1[1] = 0;
      perp1[2] = axis[0];
   }
   else {
      perp1[0] = axis[1];
      perp1[1] = -axis[0];
      perp1[2] = 0;
   }
   vec_normalize(perp1);

   // now we compute another vector perpendicular both to the
   // cylinder's axis and to the perpendicular we just found.
   cross_prod(perp2, axis, perp1);

   // initialize some stuff in the depth sort objects
   cyl_body.npoints = 4;
   cyl_body.color = colorIndex;

   if (filled & CYLINDER_TRAILINGCAP) {
      cyl_trailcap.npoints = 3;
      cyl_trailcap.color = colorIndex;
   }

   if (filled & CYLINDER_LEADINGCAP) {
      cyl_leadcap.npoints = 3;
      cyl_leadcap.color = colorIndex;
   }

   // we will start out with the point defined by perp2
   pt1[0] = r * perp2[0];
   pt1[1] = r * perp2[1];
   pt1[2] = r * perp2[2];
   theta = 0;
   theta_inc = (float) (VMD_TWOPI / res);
   for (n = 1; n <= res; n++) {
      // save the last point
      memcpy(pt2, pt1, sizeof(float) * 3);

      // increment the angle and compute new points
      theta += theta_inc;
      my_sin = sinf(theta);
      my_cos = cosf(theta);

      // compute the new points
      pt1[0] = r * (perp2[0] * my_cos + perp1[0] * my_sin);
      pt1[1] = r * (perp2[1] * my_cos + perp1[1] * my_sin);
      pt1[2] = r * (perp2[2] * my_cos + perp1[2] * my_sin);

      cyl_body.points = (float *) malloc(sizeof(float) * 8);
      cyl_trailcap.points = (float *) malloc(sizeof(float) * 6);
      cyl_leadcap.points = (float *) malloc(sizeof(float) * 6);
      if (!(cyl_body.points && cyl_trailcap.points && cyl_leadcap.points)) {
         // memory error
         if (!memerror) {
            memerror = 1;
            msgErr << "PSDisplayDevice: Out of memory. Some " <<
               "objects were not drawn." << sendmsg;
         }
         continue;
      }

      // we have to translate them back to their original point...
      w[0] = pt1[0] + a[0];
      w[1] = pt1[1] + a[1];
      w[2] = pt1[2] + a[2];
      x[0] = pt2[0] + a[0];
      x[1] = pt2[1] + a[1];
      x[2] = pt2[2] + a[2];
      y[0] = pt2[0] + b[0];
      y[1] = pt2[1] + b[1];
      y[2] = pt2[2] + b[2];
      z[0] = pt1[0] + b[0];
      z[1] = pt1[1] + b[1];
      z[2] = pt1[2] + b[2];

      memcpy(cyl_body.points, w, sizeof(float) * 2);
      memcpy(&cyl_body.points[2], x, sizeof(float) * 2);
      memcpy(&cyl_body.points[4], y, sizeof(float) * 2);
      memcpy(&cyl_body.points[6], z, sizeof(float) * 2);

      // finally, we have to compute the centerpoint of this cylinder...
      // we can make a slight optimization here since we know the
      // cylinder will be a parellelogram. we only need to average
      // 2 corner points to find the center.
      cent[0] = (w[0] + y[0]) / 2;
      cent[1] = (w[1] + y[1]) / 2;
      cent[2] = (w[2] + y[2]) / 2;
      cyl_body.dist = compute_dist(cent);

      // and finally the light scale
      cyl_body.light_scale = compute_light(w, x, y);

      // go ahead and add this to our depth-sort list
      memusage += sizeof(float) * 2 * cyl_body.npoints;
      points += cyl_body.npoints;
      objects++;
      depth_list.append(cyl_body);

      // Now do the same thing for the trailing end cap...
      if (filled & CYLINDER_TRAILINGCAP) {
        memcpy(&cyl_trailcap.points[0], x, sizeof(float) * 2);
        memcpy(&cyl_trailcap.points[2], w, sizeof(float) * 2);
        memcpy(&cyl_trailcap.points[4], a, sizeof(float) * 2);
      
        // finally, we have to compute the centerpoint of the triangle
        cent[0] = (x[0] + w[0] + a[0]) / 3;
        cent[1] = (x[1] + w[1] + a[1]) / 3;
        cent[2] = (x[2] + w[2] + a[2]) / 3;
        cyl_trailcap.dist = compute_dist(cent);

        // and finally the light scale
        cyl_trailcap.light_scale = compute_light(x, w, a);

        memusage += sizeof(float) * 2 * cyl_trailcap.npoints;
        points += cyl_trailcap.npoints;
        objects++;
        depth_list.append(cyl_trailcap);
      }

      // ...and the leading end cap.
      if (filled & CYLINDER_LEADINGCAP) {
        memcpy(cyl_leadcap.points, z, sizeof(float) * 2);
        memcpy(&cyl_leadcap.points[2], y, sizeof(float) * 2);
        memcpy(&cyl_leadcap.points[4], b, sizeof(float) * 2);

        // finally, we have to compute the centerpoint of the triangle
        cent[0] = (z[0] + y[0] + b[0]) / 3;
        cent[1] = (z[1] + y[1] + b[1]) / 3;
        cent[2] = (z[2] + y[2] + b[2]) / 3;
        cyl_leadcap.dist = compute_dist(cent);

        // and finally the light scale
        cyl_leadcap.light_scale = compute_light(z, y, b);

        memusage += sizeof(float) * 2 * cyl_leadcap.npoints;
        points += cyl_leadcap.npoints;
        objects++;
        depth_list.append(cyl_leadcap);
      }
   }
}


void PSDisplayDevice::cone_approx(float *a, float *b, float r) {
   // XXX add ability to change number of triangles
   const int tris = 20;

   float axis[3];
   float perp1[3], perp2[3];
   float pt1[3], pt2[3];
   float cent[3];
   float x[3], y[3], z[3];
   float theta, theta_inc;
   float my_sin, my_cos;
   int n;

   DepthSortObject depth_obj;

   // check against degenerate cone
   if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) return;
   if (r <= 0) return;

   // first we compute the axis of the cone
   axis[0] = b[0] - a[0];
   axis[1] = b[1] - a[1];
   axis[2] = b[2] - a[2];
   vec_normalize(axis);

   // now we compute some arbitrary perpendicular to that axis
   if ((ABS(axis[0]) < ABS(axis[1])) &&
       (ABS(axis[0]) < ABS(axis[2]))) {
      perp1[0] = 0;
      perp1[1] = axis[2];
      perp1[2] = -axis[1];
   }
   else if ((ABS(axis[1]) < ABS(axis[2]))) {
      perp1[0] = -axis[2];
      perp1[1] = 0;
      perp1[2] = axis[0];
   }
   else {
      perp1[0] = axis[1];
      perp1[1] = -axis[0];
      perp1[2] = 0;
   }
   vec_normalize(perp1);

   // now we compute another vector perpendicular both to the
   // cone's axis and to the perpendicular we just found.
   cross_prod(perp2, axis, perp1);

   // initialize some stuff in the depth sort object
   depth_obj.npoints = 3;
   depth_obj.color = colorIndex;

   // we will start out with the point defined by perp2
   pt1[0] = r * perp2[0];
   pt1[1] = r * perp2[1];
   pt1[2] = r * perp2[2];
   theta = 0;
   theta_inc = (float) (VMD_TWOPI / tris);
   for (n = 1; n <= tris; n++) {
      // save the last point
      memcpy(pt2, pt1, sizeof(float) * 3);

      // increment the angle and compute new points
      theta += theta_inc;
      my_sin = sinf(theta);
      my_cos = cosf(theta);

      // compute the new points
      pt1[0] = r * (perp2[0] * my_cos + perp1[0] * my_sin);
      pt1[1] = r * (perp2[1] * my_cos + perp1[1] * my_sin);
      pt1[2] = r * (perp2[2] * my_cos + perp1[2] * my_sin);

      depth_obj.points = (float *) malloc(sizeof(float) * 6);
      if (!depth_obj.points) {
         // memory error
         if (!memerror) {
            memerror = 1;
            msgErr << "PSDisplayDevice: Out of memory. Some " <<
               "objects were not drawn." << sendmsg;
         }
         continue;
      }

      // we have to translate them back to their original point...
      x[0] = pt1[0] + a[0];
      x[1] = pt1[1] + a[1];
      x[2] = pt1[2] + a[2];
      y[0] = pt2[0] + a[0];
      y[1] = pt2[1] + a[1];
      y[2] = pt2[2] + a[2];

      // now we use the apex of the cone as the third point
      z[0] = b[0];
      z[1] = b[1];
      z[2] = b[2];

      memcpy(depth_obj.points, x, sizeof(float) * 2);
      memcpy(&depth_obj.points[2], y, sizeof(float) * 2);
      memcpy(&depth_obj.points[4], z, sizeof(float) * 2);

      // finally, we have to compute the centerpoint of this
      // triangle...
      cent[0] = (x[0] + y[0] + z[0]) / 3;
      cent[1] = (x[1] + y[1] + z[1]) / 3;
      cent[2] = (x[2] + y[2] + z[2]) / 3;
      depth_obj.dist = compute_dist(cent);

      // and the light shading factor
      depth_obj.light_scale = compute_light(x, y, z);

      // go ahead and add this to our depth-sort list
      memusage += sizeof(float) * 2 * depth_obj.npoints;
      points += depth_obj.npoints;
      objects++;
      depth_list.append(depth_obj);
   }
}


void PSDisplayDevice::decompose_mesh(DispCmdTriMesh *mesh) {
   int i;
   int fi;
   int f1, f2, f3;
   float r, g, b;
   float x[3], y[3], z[3], cent[3];
   float *cnv;
   int *f;
   mesh->getpointers(cnv, f);
   DepthSortObject depth_obj;

   depth_obj.npoints = 3;

   fi = -3;
   for (i = 0; i < mesh->numfacets; i++) {
      fi += 3;
      f1 = f[fi    ] * 10;
      f2 = f[fi + 1] * 10;
      f3 = f[fi + 2] * 10;

      // allocate memory for the points
      depth_obj.points = (float *) malloc(6 * sizeof(float));
      if (!depth_obj.points) {
         if (!memerror) {
            memerror = 1;
            msgErr << "PSDisplayDevice: Out of memory. Some " <<
               "objects were not drawn." << sendmsg;
         }
         continue;
      }

      // average the three colors and use that average as the color for
      // this triangle
      r = (cnv[f1] + cnv[f2] + cnv[f3]) / 3;
      g = (cnv[f1 + 1] + cnv[f2 + 1] + cnv[f3 + 1]) / 3;
      b = (cnv[f1 + 2] + cnv[f2 + 2] + cnv[f3 + 2]) / 3;
      depth_obj.color = nearest_index(r, g, b);

      // transform from world coordinates to screen coordinates and copy
      // each point to the depth sort structure in one fell swoop
      (transMat.top()).multpoint3d(&cnv[f1 + 7], x);
      (transMat.top()).multpoint3d(&cnv[f2 + 7], y);
      (transMat.top()).multpoint3d(&cnv[f3 + 7], z);
      memcpy(depth_obj.points, x, sizeof(float) * 2);
      memcpy(&depth_obj.points[2], y, sizeof(float) * 2);
      memcpy(&depth_obj.points[4], z, sizeof(float) * 2);

      // compute the centerpoint of the object
      cent[0] = (x[0] + y[0] + z[0]) / 3;
      cent[1] = (x[1] + y[1] + z[1]) / 3;
      cent[2] = (x[2] + y[2] + z[2]) / 3;

      // now compute distance to eye
      depth_obj.dist = compute_dist(cent);

      // light shading factor
      depth_obj.light_scale = compute_light(x, y, z);

      // done ... add the object to the list
      memusage += sizeof(float) * 2 * depth_obj.npoints;
      points += depth_obj.npoints;
      objects++;
      depth_list.append(depth_obj);
   }

}


void PSDisplayDevice::decompose_tristrip(DispCmdTriStrips *strip)
{
    int s, t, v = 0;
    int v0, v1, v2;
    float r, g, b;
    float x[3], y[3], z[3], cent[3];
    DepthSortObject depth_obj;

    depth_obj.npoints = 3;

    // lookup table for winding order
    const int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

    float *cnv;
    int *f;
    int *vertsperstrip;
    strip->getpointers(cnv, f, vertsperstrip);

    // loop over all of the triangle strips
    for (s = 0; s < strip->numstrips; s++)
    {
        // loop over all triangles in this triangle strip
        for (t = 0; t < vertsperstrip[s] - 2; t++)
        {
            v0 = f[v + (stripaddr[t & 0x01][0])] * 10;
            v1 = f[v + (stripaddr[t & 0x01][1])] * 10;
            v2 = f[v + (stripaddr[t & 0x01][2])] * 10;

            // allocate memory for the points
            depth_obj.points = (float *) malloc(6 * sizeof(float));
            if (!depth_obj.points) {
                if (!memerror) {
                    memerror = 1;
                    msgErr << "PSDisplayDevice: Out of memory. Some "
                           << "objects were not drawn." << sendmsg;
                }
                continue;
            }

            // average the three colors and use that average as the color for
            // this triangle
            r = (cnv[v0+0] + cnv[v1+0] + cnv[v2+0]) / 3; 
            g = (cnv[v0+1] + cnv[v1+1] + cnv[v2+1]) / 3; 
            b = (cnv[v0+2] + cnv[v1+2] + cnv[v2+2]) / 3; 
            depth_obj.color = nearest_index(r, g, b);

            // transform from world coordinates to screen coordinates and copy
            // each point to the depth sort structure in one fell swoop
            (transMat.top()).multpoint3d(&cnv[v0 + 7], x);
            (transMat.top()).multpoint3d(&cnv[v1 + 7], y);
            (transMat.top()).multpoint3d(&cnv[v2 + 7], z);
            memcpy(depth_obj.points, x, sizeof(float) * 2);
            memcpy(&depth_obj.points[2], y, sizeof(float) * 2);
            memcpy(&depth_obj.points[4], z, sizeof(float) * 2);

            // compute the centerpoint of the object
            cent[0] = (x[0] + y[0] + z[0]) / 3;
            cent[1] = (x[1] + y[1] + z[1]) / 3;
            cent[2] = (x[2] + y[2] + z[2]) / 3;

            // now compute distance to eye
            depth_obj.dist = compute_dist(cent);

            // light shading factor
            depth_obj.light_scale = compute_light(x, y, z);

            // done ... add the object to the list
            memusage += sizeof(float) * 2 * depth_obj.npoints;
            points += depth_obj.npoints;
            objects++;
            depth_list.append(depth_obj);

            v++; // move on to next vertex
        } // triangles
    v+=2; // last two vertices are already used by last triangle
    } // strips  
}


void PSDisplayDevice::write_header(void) {
   int i;

   fprintf(outfile, "%%!PS-Adobe-1.0\n");
   fprintf(outfile, "%%%%DocumentFonts:Helvetica\n");
   fprintf(outfile, "%%%%Title:vmd.ps\n");
   fprintf(outfile, "%%%%Creator:VMD -- Visual Molecular Dynamics\n");
   fprintf(outfile, "%%%%CreationDate:\n");
   fprintf(outfile, "%%%%Pages:1\n");
   fprintf(outfile, "%%%%BoundingBox:0 0 612 792\n");
   fprintf(outfile, "%%%%EndComments\n");
   fprintf(outfile, "%%%%Page:1 1\n");

   fprintf(outfile, "%3.2f %3.2f %3.2f setrgbcolor    %% background color\n",
      backColor[0], backColor[1], backColor[2]);
   fprintf(outfile, "newpath\n");
   fprintf(outfile, "0 0 moveto\n");
   fprintf(outfile, "0 792 lineto\n");
   fprintf(outfile, "792 792 lineto\n");
   fprintf(outfile, "792 0 lineto\n");
   fprintf(outfile, "closepath\nfill\nstroke\n");

   // quadrilateral ( /s )
   // Format: x1 y1 x2 y2 x3 y3 x4 y4 s
   fprintf(outfile, "/s\n");
   fprintf(outfile, "{ newpath moveto lineto lineto lineto closepath fill stroke } def\n");

   // quadrilateral-w ( /sw )
   fprintf(outfile, "/sw\n");
   fprintf(outfile, "{ newpath moveto lineto lineto lineto closepath stroke } def\n");

   // triangle ( /t )
   fprintf(outfile, "/t\n");
   fprintf(outfile, "{ newpath moveto lineto lineto closepath fill stroke } def\n");

   // triangle-w ( /tw )
   fprintf(outfile, "/tw\n");
   fprintf(outfile, "{ newpath moveto lineto lineto closepath stroke } def\n");

   // point ( /p )
   // A point is drawn by making a 'cross' around the point, meaning two
   // lines from (x-1,y) to (x+1,y) and (x,y-1) to (x,y+1). The PostScript
   // here is from the old PSDisplayDevice, and it can probably be cleaned
   // up, but is not urgent.
   fprintf(outfile, "/p\n");
   fprintf(outfile, "{ dup dup dup 5 -1 roll dup dup dup 8 -1 roll exch 8 -1\n");
   fprintf(outfile, "  roll 4 1 roll 8 -1 roll 6 1 roll newpath -1 add moveto\n");
   fprintf(outfile, "  1 add lineto exch -1 add exch moveto exch 1 add exch\n");
   fprintf(outfile, "  lineto closepath stroke } def\n");

   // line ( /l )
   fprintf(outfile, "/l\n");
   fprintf(outfile, "{ newpath moveto lineto closepath stroke } def\n");

   // scalecolor ( /mc )
   // This takes an rgb triplet and scales it according to a floating point
   // value. This is useful for polygon shading and is used with the color table.
   fprintf(outfile, "/mc\n");
   fprintf(outfile, "{ dup 4 1 roll dup 3 1 roll mul 5 1 roll mul 4 1 roll\n");
   fprintf(outfile, "  mul 3 1 roll } def\n");

   // getcolor ( /gc )
   // This function retrieves a color from the color table.
   fprintf(outfile, "/gc\n");
   fprintf(outfile, "{ 2 1 roll dup 3 -1 roll get dup dup 0 get 3 1 roll\n");
   fprintf(outfile, "  1 get 3 1 roll 2 get 3 -1 roll exch } def\n");

   // /text :  draw text at given position
   fprintf(outfile, "/text\n");
   fprintf(outfile,"{ moveto show } def\n");

   // textsize ( /ts )
   fprintf(outfile, "/ts\n");
   fprintf(outfile, "{ /Helvetica findfont exch scalefont setfont } def\n");

   // load font and set defaults
   //fprintf(outfile,"15 ts\n");

   // setcolor ( /c )
   // This function retrieves a color table entry and scales it.
   fprintf(outfile, "/c\n");
   fprintf(outfile, "{ 3 1 roll gc 5 -1 roll mc setrgbcolor } def\n");

   // Now we need to write out the entire color table to the Postscript
   // file. The table is implemented as a double array. Each element in the
   // array contains another array of 3 elements -- the RGB triple. The
   // getcolor function retrieves a triplet from this array.
   fprintf(outfile, "\n");
   for (i = 0; i < MAXCOLORS; i++) {
      fprintf(outfile, "[ %.2f %.2f %.2f ]\n",
         matData[i][0],
         matData[i][1],
         matData[i][2]);
   }
   fprintf(outfile, "%d array astore\n", MAXCOLORS);
}


void PSDisplayDevice::write_trailer(void) {
}


void PSDisplayDevice::comment(const char *s) {
   fprintf(outfile, "%%%% %s\n", s);
}


inline float PSDisplayDevice::compute_dist(float *s) {
   return
      (s[0] - eyePos[0]) * (s[0] - eyePos[0]) +
      (s[1] - eyePos[1]) * (s[1] - eyePos[1]) +
      (s[2] - eyePos[2]) * (s[2] - eyePos[2]);
}


float PSDisplayDevice::compute_light(float *a, float *b, float *c) {
   float norm[3];
   float light_scale;

   // compute a normal vector to the surface of the polygon
   norm[0] =
      a[1] * (b[2] - c[2]) +
      b[1] * (c[2] - a[2]) +
      c[1] * (a[2] - b[2]);
   norm[1] =
      a[2] * (b[0] - c[0]) +
      b[2] * (c[0] - a[0]) +
      c[2] * (a[0] - b[0]);
   norm[2] =
      a[0] * (b[1] - c[1]) +
      b[0] * (c[1] - a[1]) +
      c[0] * (a[1] - b[1]);

   // if the normal vector is zero, something is wrong with the
   // object so we'll just display it with a light_scale of zero
   if (!norm[0] && !norm[1] && !norm[2]) {
      light_scale = 0;
   } else {
      // otherwise we use the dot product of the surface normal
      // and the light normal to determine a light_scale
      vec_normalize(norm);
      light_scale = dot_prod(norm, norm_light);
      if (light_scale < 0) light_scale = -light_scale;
   }

   return light_scale;
}
