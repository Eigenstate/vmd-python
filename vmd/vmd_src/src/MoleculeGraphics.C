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
 *	$RCSfile: MoleculeGraphics.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.76 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Manages a list of graphics objects.  They can be queried and modified.
 *  This is for use by the text interface (and perhaps others).
 *
 ***************************************************************************/

#include <stdio.h>
#include "MoleculeGraphics.h"
#include "DispCmds.h"
#include "VMDApp.h"
#include "Inform.h"
#include "Scene.h"

void MoleculeGraphics::create_cmdlist(void) {
  reset_disp_list();
  // set the default values
  DispCmdTriangle triangle;
  DispCmdCylinder cylinder;
  DispCmdPoint point;
  DispCmdLine line;
  DispCmdCone cone;
  DispCmdColorIndex color;
  DispCmdLineType linetype;
  DispCmdLineWidth linewidth;
  DispCmdSphere sphere;
  DispCmdSphereRes sph_res;
  //DispCmdComment comment;
  DispCmdPickPoint cmdPickPoint;

  append(DMATERIALON);
  color.putdata(0, cmdList);     // use the first color by default (blue)
  int last_res = -1;             // default sphere resolution

  int last_line = ::SOLIDLINE;       // default for lines
  linetype.putdata(last_line, cmdList); // solid and
  int last_width = 1;
  linewidth.putdata(last_width, cmdList);  //  of width 1

  // go down the list and draw things
  int num = num_elements();
  ShapeClass *shape;
  int sidx=0;
  while (sidx<num) {
    shape = &(shapes[sidx]);

    switch (shape->shape) {
      case NONE: {
        break;
      }

      case POINT: {
        append(DMATERIALOFF);
        point.putdata(shape->data+0, cmdList);
        append(DMATERIALON);
        break;
      }

      case PICKPOINT: 
        cmdPickPoint.putdata(shape->data, (int)shape->data[3], cmdList);
        break;

      case LINE: {
        append(DMATERIALOFF);
        int style = int(shape->data[6]);
        int width = int(shape->data[7]);
        if (style != last_line) {
          linetype.putdata(style, cmdList);
          last_line = style;
        }
        if (width != last_width) {
          linewidth.putdata(width, cmdList);
          last_width = width;
        }
        line.putdata(shape->data+0, shape->data+3, cmdList);
        append(DMATERIALON);
        break;
      }

      case TRIANGLE: {
        ResizeArray<float> triangle_vert_buffer;
        int tricount=0;
        while ((sidx<num) && (tricount < 50000000) && (shape->shape == TRIANGLE)) {
          triangle_vert_buffer.append9(&shape->data[0]);
          tricount++;
          sidx++; // go to the next shape/object
          shape = &(shapes[sidx]);
        }
        sidx--; // correct count prior to outer loop increment

        int cnt = triangle_vert_buffer.num() / 9;
        if (cnt > 0) {
          DispCmdTriMesh::putdata(&triangle_vert_buffer[0], (float *) NULL, 
                                  (float *) NULL, cnt, cmdList);
        }
        break;
      }

      case TRINORM: {
        triangle.putdata(shape->data+0, shape->data+3 , shape->data+6, 
                         shape->data+9, shape->data+12, shape->data+15, cmdList);
        break;
      }

      case TRICOLOR: {
        // compute rgb colors from indices
        float colors[9];
        for (int i=0; i<3; i++) {
          int c = (int)shape->data[18+i];
          c = clamp_int(c, 0, MAXCOLORS-1);
          vec_copy(colors+3L*i, scene->color_value(c));
        }
        const float *verts = shape->data+0;
        const float *norms = shape->data+9;
        int facets[3] = { 0,1,2 };
        DispCmdTriMesh::putdata(verts, norms, colors, 3, facets, 1, 0, cmdList);
      }
      break;

      case CYLINDER: {
        cylinder.putdata(shape->data+0, shape->data+3, shape->data[6], 
                         int(shape->data[7]), 
                         (int (shape->data[8])) ? 
                           CYLINDER_TRAILINGCAP | CYLINDER_LEADINGCAP : 0, 
                         cmdList);
        break;
      }

      case CONE: {
        cone.putdata(shape->data+0, shape->data+3, shape->data[6], 
                     shape->data[8], int(shape->data[7]), cmdList);
        break;
      }

      case TEXT: {
        append(DMATERIALOFF);
        DispCmdText text;
        DispCmdTextSize textSize;
        textSize.putdata(shape->data[3], cmdList);
        text.putdata(shape->data, shapetext[(int)shape->data[5]], shape->data[4], cmdList);
        append(DMATERIALON);
        break;
      }

#if 0
      // these aren't supported yet
      case DBEGINREPGEOMGROUP: {
        char *s = (char *) (shape);
        beginrepgeomgroup.putdata(s,cmdList);
        break;
      }

      case DCOMMENT: {
        char *s = (char *) (shape);
        comment.putdata(s,cmdList);
        break;
      }
#endif

      case SPHERE: {
        int res = int (shape->data[4]);
        if (res != last_res) {
          sph_res.putdata(res, cmdList);
          last_res = res;
        }
        sphere.putdata(shape->data+0, shape->data[3], cmdList);
        break;
      }

      case MATERIALS: {
        if (shape->data[0] == 0) append(DMATERIALOFF);
        else append(DMATERIALON);
        break;
      }

      case MATERIAL: {
        const float *data = shape->data;
        cmdList->ambient = data[0];
        cmdList->specular = data[1];
        cmdList->diffuse = data[2];
        cmdList->shininess = data[3];
        cmdList->mirror = data[4];
        cmdList->opacity = data[5];
        cmdList->outline = data[6];
        cmdList->outlinewidth = data[7];
        cmdList->transmode = data[8];
        cmdList->materialtag = (int)data[9];
        break;
      }

      case COLOR: {
        color.putdata(int(shape->data[0]), cmdList);
        break;
      }

      default:
        msgErr << "Sorry, can't draw that" << sendmsg;
    }

    sidx++; // go to the next shape/object
  }

  needRegenerate = 0;  // cmdlist has been udpated 
}


// resets the {next,max}_{id,index} values after something is added
// returns the value of the new element
int MoleculeGraphics::added(void) {
  needRegenerate = 1;
  next_index = shapes.num();
  int retval = next_id;
  if (next_id == max_id) { // this was a new shape
    max_id++;
  } 
  next_id = max_id;
  return retval;
}


int MoleculeGraphics::add_triangle(const float *x1, const float *x2, const float *x3) {
  // save the points
  ShapeClass s(TRIANGLE, 9, next_id);
  float *data = s.data;
  vec_copy(data+0, x1);
  vec_copy(data+3, x2);
  vec_copy(data+6, x3);
  
  // new one goes at next_id
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


int MoleculeGraphics::add_trinorm(const float *x1, const float *x2, const float *x3,
                                  const float *n1, const float *n2, const float *n3) {
  // save the points
  ShapeClass s(TRINORM, 18, next_id);
  float *data = s.data;
  vec_copy(data+ 0, x1);
  vec_copy(data+ 3, x2);
  vec_copy(data+ 6, x3);
  
  vec_copy(data+ 9, n1);
  vec_normalize(data+ 9); // normalize this normal to prevent problems later
  vec_copy(data+12, n2);
  vec_normalize(data+12); // normalize this normal to prevent problems later
  vec_copy(data+15, n3);
  vec_normalize(data+15); // normalize this normal to prevent problems later
  
  // new one goes at next_id
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);

  return added();
}


int MoleculeGraphics::add_tricolor(const float *x1, const float *x2, const float *x3,
                                   const float *n1, const float *n2, const float *n3, int c1, int c2,
          int c3) {
  ShapeClass s(TRICOLOR, 21, next_id);
  float *data = s.data;
  vec_copy(data+ 0, x1);
  vec_copy(data+ 3, x2);
  vec_copy(data+ 6, x3);

  vec_copy(data+ 9, n1);
  vec_normalize(data+ 9); // normalize this normal to prevent problems later
  vec_copy(data+12, n2);
  vec_normalize(data+12); // normalize this normal to prevent problems later
  vec_copy(data+15, n3);
  vec_normalize(data+15); // normalize this normal to prevent problems later

  data[18] = (float)c1;
  data[19] = (float)c2;
  data[20] = (float)c3;
  
  // new one goes at next_id
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);

  return added();
}


int MoleculeGraphics::add_point(const float *x) {
  ShapeClass s(POINT, 3, next_id);
  float *data = s.data;
  vec_copy(data+0, x);

  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);

  return added();
}

int MoleculeGraphics::add_pickpoint(const float *x) {
  ShapeClass s(PICKPOINT, 4, next_id);
  float *data = s.data;
  vec_copy(data+0, x);
  data[3] = (float) next_index; // this will break for anything >= 2^24

  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);

  return added();
}

int MoleculeGraphics::add_line(const float *x1, const float *x2, int style, int width) {
  ShapeClass s(LINE, 8, next_id);
  float *data = s.data;
  vec_copy(data+0, x1);
  vec_copy(data+3, x2);
  data[6] = float(style) + 0.1f;
  data[7] = float(width) + 0.1f;
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


int MoleculeGraphics::add_cylinder(const float *x1, const float *x2, float rad,
                                   int n, int filled) {
  ShapeClass s(CYLINDER, 9, next_id);
  float *data = s.data;
  vec_copy(data+0, x1);
  vec_copy(data+3, x2);
  data[6] = rad;
  data[7] = float(n) + 0.1f;
  data[8] = float(filled) + 0.1f;
  
  // new one goes at next_id
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


int MoleculeGraphics::add_cone(const float *x1, const float *x2, float rad, float radsq, int n) {
  // save the points
  ShapeClass s(CONE, 9, next_id);
  float *data = s.data;
  vec_copy(data+0, x1);
  vec_copy(data+3, x2);
  data[6] = rad;
  data[7] = float(n) + 0.1f;
  data[8] = radsq;
  
  // new one goes at next_id
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


int MoleculeGraphics::add_sphere(const float *x, float rad, int n) {
  ShapeClass s(SPHERE, 5, next_id);
  float *data = s.data;
  vec_copy(data+0, x);
  data[3] = rad;
  data[4] = float(n) + 0.1f;
  
  // new one goes at next_id
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


int MoleculeGraphics::add_text(const float *x, const char *text, 
                               float size, float thickness) {
  ShapeClass s(TEXT, 6, next_id); 
  float *data = s.data;
  vec_copy(data+0, x);
  data[3] = size;
  data[4] = thickness;
  data[5] = (float)shapetext.num(); // index where the text will be stored
  shapetext.append(stringdup(text));
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


int MoleculeGraphics::use_materials(int yes_no) {
  ShapeClass s(MATERIALS, 1, next_id);
  float *data = s.data;
  data[0] = (float) yes_no;
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


int MoleculeGraphics::use_material(const Material *mat) {
  ShapeClass s(MATERIAL, 10, next_id);
  float *data = s.data;
  data[0] = mat->ambient;
  data[1] = mat->specular;
  data[2] = mat->diffuse;
  data[3] = mat->shininess;
  data[4] = mat->mirror;
  data[5] = mat->opacity;
  data[6] = mat->outline;
  data[7] = mat->outlinewidth;
  data[8] = mat->transmode;
  data[9] = (float)mat->ind;

  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


// do this based on the index
int MoleculeGraphics::use_color(int index) {
  ShapeClass s(COLOR, 1, next_id);
  float *data = s.data;
  data[0] = float(index) + 0.1f; // just to be on the safe side for rounding
  if (next_index < num_elements())
    shapes[next_index] = s;
  else
    shapes.append(s);
  return added();
}


// return the index in the array, or -1 if it doesn't exist
int MoleculeGraphics::index_id(int find_id) {
  // the values in the array are numerically increasing, so I can do
  // a binary search.
  int max_loc = num_elements()-1;
  int min_loc = 0;
  if (max_loc < min_loc) {
    return -1;
  }
  int loc = (max_loc + min_loc) / 2;
  int id = shapes[loc].id;
  while (id != find_id && min_loc < max_loc) {
    if (id < find_id) {
      min_loc = loc+1;
    } else {
      max_loc = loc-1;
    }
    loc = (max_loc + min_loc) / 2;
    if (loc < 0) break;
    id = shapes[loc].id;
  }
  // and make sure it is for real
  if (id == find_id && shapes[loc].shape != NONE) {
    return loc;
  }
  return -1; // not found
}


// delete everything
void MoleculeGraphics::delete_all(void) {
  shapes.clear(); 
  delete_shapetext();    // since there are no references to it now
  delete_count = 0;      // and reset the internal variables
  next_index = 0;
  next_id = 0;
  max_id = 0;
  needRegenerate = 1;
}


// delete given the id
void MoleculeGraphics::delete_id(int id) {
  int index = index_id(id);
  if (index < 0) return;
  shapes[index].clear();
  delete_count++;
  if (delete_count > 1/* && 
      float(delete_count)/float(num_elements()) > 0.2*/) {
    // clear out the deleted elements
    int i, j=0, n = num_elements();
    // moving from i to j
    for (i=0; i<n; i++) {
      if (shapes[i].shape != NONE) {
        if (i != j) {
          shapes[j] = shapes[i];
        }
        j++;
      }
    }
    i=j;
    while (i<n) {
      shapes[i].clear();
      i++;
    }
    // remove in reverse order so we don't have to copy anything
    for (int k=n-1; k >= j; k--) shapes.remove(k);
    delete_count = 0;
  }
  needRegenerate = 1;
  // delete overrides a replace
  next_id = max_id;
  next_index = num_elements();
}


// have the next added shape replace the given element
// returns index
int MoleculeGraphics::replace_id(int id) {
  int index = index_id(id);
  if (index < 0) return -1;
  // if one was already assigned to be replaced, and we want to
  // replace another, increase the delete count
  if (next_id != max_id) {
    delete_count++;
  }
  // do the replacement
  shapes[index].clear();
  next_id = id;
  next_index = index;
  return index;
}
  

const char *MoleculeGraphics::info_id(int id) {
  int index = index_id(id);
  if (index < 0) return NULL;
  ShapeClass *shape;
  shape = &(shapes[index]);

  switch (shape->shape) {
  case NONE: {
    graphics_info[0] = '\0';
    return graphics_info;
  }
  case POINT: {
    sprintf(graphics_info, "point {%f %f %f}",
            shape->data[0], shape->data[1], shape->data[2]);
    return graphics_info;
  }
  case PICKPOINT: {
    sprintf(graphics_info, "pickpoint {%f %f %f} %d",
            shape->data[0], shape->data[1], shape->data[2], (int)shape->data[3]);
    return graphics_info;
  }
  case LINE: {
    sprintf(graphics_info, "line {%f %f %f} {%f %f %f} style %s width %d",
            shape->data[0], shape->data[1], shape->data[2], 
            shape->data[3], shape->data[4], shape->data[5],
            shape->data[6] < 0.5 ? "solid" : "dashed",
            int(shape->data[7]));
    return graphics_info;
  }
  case TRIANGLE: {
    sprintf(graphics_info, "triangle {%f %f %f} {%f %f %f} {%f %f %f}",
            shape->data[0], shape->data[1], shape->data[2], 
            shape->data[3], shape->data[4], shape->data[5], 
            shape->data[6], shape->data[7], shape->data[8]);
    return graphics_info;
  }
  case TRINORM: {
    sprintf(graphics_info, "trinorm {%f %f %f} {%f %f %f} {%f %f %f} "
            "{%f %f %f} {%f %f %f} {%f %f %f}",
            shape->data[0], shape->data[1], shape->data[2], 
            shape->data[3], shape->data[4], shape->data[5], 
            shape->data[6], shape->data[7], shape->data[8],
            shape->data[9], shape->data[10], shape->data[11], 
            shape->data[12], shape->data[13], shape->data[14], 
            shape->data[15], shape->data[16], shape->data[17]);
    return graphics_info;
  }
  case TRICOLOR: {
    sprintf(graphics_info, "tricolor {%f %f %f} {%f %f %f} {%f %f %f} "
            "{%f %f %f} {%f %f %f} {%f %f %f} %d %d %d",
            shape->data[0], shape->data[1], shape->data[2], 
            shape->data[3], shape->data[4], shape->data[5], 
            shape->data[6], shape->data[7], shape->data[8],
            shape->data[9], shape->data[10], shape->data[11], 
            shape->data[12], shape->data[13], shape->data[14], 
            shape->data[15], shape->data[16], shape->data[17],
      (int)shape->data[18], (int)shape->data[19], (int)shape->data[20]);
    return graphics_info;
  }
  case CYLINDER: {
    sprintf(graphics_info, "cylinder {%f %f %f} {%f %f %f} "
            "radius %f resolution %d filled %d",
            shape->data[0], shape->data[1], shape->data[2], 
            shape->data[3], shape->data[4], shape->data[5], 
            shape->data[6], int(shape->data[7]), int(shape->data[8]));
    return graphics_info;
  }
  case CONE: {
    sprintf(graphics_info, "cone {%f %f %f} {%f %f %f} "
            "radius %f radius2 %f resolution %d",
            shape->data[0], shape->data[1], shape->data[2], 
            shape->data[3], shape->data[4], shape->data[5], 
            shape->data[6], shape->data[8], int(shape->data[7]));
    return graphics_info;
  }
  case SPHERE: {
    sprintf(graphics_info, "sphere {%f %f %f} radius %f resolution %d",
            shape->data[0], shape->data[1], shape->data[2], 
            shape->data[3], int(shape->data[4]));
    return graphics_info;
  }
  case TEXT: {
    sprintf(graphics_info, "text {%f %f %f} {%s} size %f thickness %f",
            shape->data[0], shape->data[1], shape->data[2],
            shapetext[(int)shape->data[5]], shape->data[3], shape->data[4]);
    return graphics_info;
  }
  case MATERIALS: {
    sprintf(graphics_info, "materials %d", int(shape->data[0]));
    return graphics_info;
  }
  case MATERIAL: {
    sprintf(graphics_info, "material %d", int(shape->data[9]));
    return graphics_info;
  }
  case COLOR: {
    sprintf(graphics_info, "color %d", int(shape->data[0]));
    return graphics_info;
  }
  default:
    return "";
  }
}


// return the center of volume and scaling factor
#define CHECK_RANGE(v)     \
{                          \
  if (!found_one) {        \
    found_one = 1;         \
    minx = maxx = (v)[0];  \
    miny = maxy = (v)[1];  \
    minz = maxz = (v)[2];  \
  } else {                 \
    if (minx > (v)[0]) minx = (v)[0];  if (maxx < (v)[0]) maxx = (v)[0]; \
    if (miny > (v)[1]) miny = (v)[1];  if (maxy < (v)[1]) maxy = (v)[1]; \
    if (minz > (v)[2]) minz = (v)[2];  if (maxz < (v)[2]) maxz = (v)[2]; \
  }                        \
}


void MoleculeGraphics::find_sizes(void) {
  float minx=0.0f, maxx=0.0f;
  float miny=0.0f, maxy=0.0f;
  float minz=0.0f, maxz=0.0f;
  int found_one = 0;
  // go down the list and draw things
  int num = num_elements();
  ShapeClass *shape;
  for (int i=0; i<num; i++) {
    shape = &(shapes[i]);
    switch (shape->shape) {
    case NONE: {
      break;
    }
    case POINT: {
      CHECK_RANGE(shape->data+0);
      break;
    }
    case PICKPOINT: {
      CHECK_RANGE(shape->data+0);
      break;
    }
    case LINE: {
      CHECK_RANGE(shape->data+0);
      CHECK_RANGE(shape->data+3);
      break;
    }
    case TRIANGLE: {
      CHECK_RANGE(shape->data+0);
      CHECK_RANGE(shape->data+3);
      CHECK_RANGE(shape->data+6);
      break;
    }
    case TRINORM: {
      CHECK_RANGE(shape->data+0);
      CHECK_RANGE(shape->data+3);
      CHECK_RANGE(shape->data+6);
      break;
    }
    case TRICOLOR: {
      CHECK_RANGE(shape->data+0);
      CHECK_RANGE(shape->data+3);
      CHECK_RANGE(shape->data+6);
      break;
    }
    case CYLINDER: {
      CHECK_RANGE(shape->data+0);
      CHECK_RANGE(shape->data+3);
      break;
    }
    case CONE: {
      CHECK_RANGE(shape->data+0);
      CHECK_RANGE(shape->data+3);
      break;
    }
    case SPHERE: { // I suppose I should include +/- radius ...
      CHECK_RANGE(shape->data+0);
      break;
    }
    case TEXT: {   // I suppose I should include the string length size...
      CHECK_RANGE(shape->data+0);
      break;
    }
    default:
      break;
    }
  }

  // compute the values for center of volume center and scale
  if (!found_one) {
    cov_pos[0] = cov_pos[1] = cov_pos[2];
    cov_scale = 0.1f;
  } else {
    cov_pos[0] = (minx + maxx) / 2.0f;
    cov_pos[1] = (miny + maxy) / 2.0f;
    cov_pos[2] = (minz + maxz) / 2.0f;
    float dx = maxx - minx;
    float dy = maxy - miny;
    float dz = maxz - minz;
    // a bit of sanity check (eg, suppose there is only one point)
    if (dx == 0 && dy == 0 && dz == 0) dx = 10;
    if (dx > dy) {
      if (dx > dz) {
        cov_scale = 2.0f / dx;
      } else {
        cov_scale = 2.0f / dz;
      }
    } else {
      if (dy > dz) {
        cov_scale = 2.0f / dy;
      } else {
        cov_scale = 2.0f / dz;
      }
    }
  }
}


void MoleculeGraphics::delete_shapetext() {
  for (int i=0; i<shapetext.num(); i++) 
    delete [] shapetext[i];

  shapetext.clear();
}

