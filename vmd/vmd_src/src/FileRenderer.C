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
 *	$RCSfile: FileRenderer.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.181 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * The FileRenderer class implements the data and functions needed to 
 * render a scene to a file in some format (postscript, raster3d, etc.)
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"
#include "DispCmds.h"
#include "FileRenderer.h"
#include "VMDDisplayList.h"
#include "Inform.h"
#include "Scene.h"
#include "Hershey.h"

// constructor
FileRenderer::FileRenderer(const char *public_name, 
                           const char *public_pretty_name,
                           const char *default_file_name,
			   const char *default_command_line) : 
  DisplayDevice(public_name), transMat(10)
{
  // save the various names
  publicName = stringdup(public_name);
  publicPrettyName = stringdup(public_pretty_name);
  defaultFilename = stringdup(default_file_name);
  defaultCommandLine = stringdup(default_command_line);
  execCmd = stringdup(defaultCommandLine);
  has_aa = 0;
  aasamples = -1;
  aosamples = -1;
  has_imgsize = 0;
  imgwidth = imgheight = 0;
  aspectratio = 0.0f;
  curformat = -1;
  textoffset_x = 0; 
  textoffset_y = 0;
  warningflags = FILERENDERER_NOWARNINGS;

  // init some state variables
  outfile = NULL;
  isOpened = FALSE;
  my_filename = NULL;

  // initialize sphere tesselation variables
  sph_nverts = 0;
  sph_verts = NULL;
}

// close (to be on the safe side) and delete
FileRenderer::~FileRenderer(void) {
  // free sphere tessellation data
  if (sph_verts && sph_nverts) 
    free(sph_verts);

  close_file();
  delete [] my_filename;
  delete [] publicName;
  delete [] publicPrettyName;
  delete [] defaultFilename;
  delete [] defaultCommandLine;
  delete [] execCmd;
}

int FileRenderer::do_define_light(int n, float *color, float *position) {
  if (n < 0 || n >= DISP_LIGHTS)
    return FALSE;

  for (int i=0; i<3; i++) {
    lightState[n].color[i] = color[i];
    lightState[n].pos[i] = position[i];
  }
  return TRUE;
}

int FileRenderer::do_activate_light(int n, int turnon) {
  if (n < 0 || n >= DISP_LIGHTS)
    return FALSE;

  lightState[n].on = turnon;
  return TRUE;
}

int FileRenderer::do_define_adv_light(int n, float *color,
                                      float *position,
                                      float constant, float linear, float quad,
                                      float *spotdir, float fallstart, 
                                      float fallend, int spoton) {
  if(n < 0 || n >= DISP_LIGHTS)
    return FALSE;

  for (int i=0; i < 3; i++) {
    advLightState[n].color[i] = color[i];
    advLightState[n].pos[i] = position[i];
    advLightState[n].spotdir[i] = spotdir[i];
  }
  advLightState[n].constfactor = constant;
  advLightState[n].linearfactor = linear;
  advLightState[n].quadfactor = quad;
  advLightState[n].fallstart = fallstart;
  advLightState[n].fallend = fallend;
  advLightState[n].spoton = spoton;
  return TRUE;
}

int FileRenderer::do_activate_adv_light(int n, int turnon) {
  if (n < 0 || n >= DISP_LIGHTS)
    return FALSE;

  advLightState[n].on = turnon;
  return TRUE;
}



void FileRenderer::do_use_colors() {
  for (int i=0; i<MAXCOLORS; i++) {
    matData[i][0] = colorData[3L*i  ];
    matData[i][1] = colorData[3L*i+1];
    matData[i][2] = colorData[3L*i+2];
  }
}

int FileRenderer::set_imagesize(int *w, int *h) {
  if (*w < 0 || *h < 0) return FALSE;
  if (*w == imgwidth && *h == imgheight) return TRUE;
  if (!aspectratio) {
    if (*w) imgwidth = *w; 
    if (*h) imgheight = *h;
  } else {
    if (*w) {
      imgwidth = *w;
      imgheight = (int)(*w / aspectratio);
    } else if (*h) {
      imgwidth = (int)(*h * aspectratio);
      imgheight = *h;
    } else {
      if (imgwidth || imgheight) {
        int wtmp = imgwidth, htmp = imgheight;
        set_imagesize(&wtmp, &htmp);
      }
    }
  }
  update_exec_cmd();
  *w = imgwidth;
  *h = imgheight;
  return TRUE;
}

float FileRenderer::set_aspectratio(float aspect) {
  if (aspect >= 0) aspectratio = aspect;
  int w=0, h=0;
  set_imagesize(&w, &h);  // update_exec_cmd() called from set_imagesize() 
  return aspectratio;
}

int FileRenderer::nearest_index(float r, float g, float b) const {
   const float *rcol = matData[BEGREGCLRS];  // get the solid colors
   float lsq = r - rcol[0]; lsq *= lsq;
   float tmp = g - rcol[1]; lsq += tmp * tmp;
         tmp = b - rcol[2]; lsq += tmp * tmp;
   float best = lsq;
   int bestidx = BEGREGCLRS;
   for (int n=BEGREGCLRS+1; n < (BEGREGCLRS + REGCLRS + MAPCLRS); n++) {
     rcol = matData[n];
     lsq = r - rcol[0]; lsq *= lsq;
     tmp = g - rcol[1]; lsq += tmp * tmp;
     tmp = b - rcol[2]; lsq += tmp * tmp;
     if (lsq < best) {
       best = lsq;
       bestidx = n;
     }
   }
   return bestidx;
}

void FileRenderer::set_background(const float * bg) {
  backColor[0] = bg[0];
  backColor[1] = bg[1];
  backColor[2] = bg[2];
}

void FileRenderer::set_backgradient(const float *top, const float *bot) {
  vec_copy(backgradienttopcolor, top);
  vec_copy(backgradientbotcolor, bot);
}


// open file, closing the previous file if it was still open
int FileRenderer::open_file(const char *filename) {
  if (isOpened) {
    close_file();
  }
  if ((outfile = fopen(filename, "w")) == NULL) {
    msgErr << "Could not open file " << filename
           << " in current directory for writing!" << sendmsg;
    return FALSE;
  }
  my_filename = stringdup(filename);
  isOpened = TRUE;
  reset_state();
  return TRUE;
}

void FileRenderer::reset_state(void) {
  // empty out the viewing stack
  while (transMat.num()) {
    transMat.pop();
  }
  // reset everything else
  colorIndex = -1;
  materialIndex = -1;
  lineWidth = 1;
  lineStyle = ::SOLIDLINE;
  pointSize = 1;
  sphereResolution = 4;
  sphereStyle = 1;
  materials_on = 0;
}

// close the file.  This could be called by open_file if the previous
// file wasn't closed, so don't put too much here
void FileRenderer::close_file(void) {
  if (outfile) {
    fclose(outfile);
    outfile = NULL;
  }
  delete [] my_filename;
  my_filename = NULL;
  isOpened = FALSE;
}


int FileRenderer::prepare3D(int) {
  // set the eye position, based on the value of whichEye, which was
  // obtained when we copied the current visible display device to the
  // file renderer.  
  int i;
  float lookat[3];
  for (i=0; i<3; i++) 
    lookat[i] = eyePos[i] + eyeDir[i];

  switch (whichEye) {
    case LEFTEYE:
      for (i=0; i<3; i++) 
        eyePos[i] -= eyeSepDir[i];
  
      for (i=0; i<3; i++) 
        eyeDir[i] = lookat[i] - eyePos[i];
      break; 

    case RIGHTEYE:
      for (i=0; i<3; i++) 
        eyePos[i] += eyeSepDir[i]; 

      for (i=0; i<3; i++) 
        eyeDir[i] = lookat[i] - eyePos[i];
      break;

    case NOSTEREO: 
      break;
  }

  if (isOpened) {
    write_header();
  }

  return TRUE;
}

/////////////////////////////////// render the display lists

void FileRenderer::render(const VMDDisplayList *cmdList) {
  if (!cmdList) return;
  int tok, i;
  char *cmdptr; 

  // scan through the list and do the action based on the token type
  // if the command relates to the viewing state, keep track of it
  // for those renderers that don't store state
  while (transMat.num()) {    // clear the stack
    transMat.pop();
  }
  Matrix4 m;
  transMat.push(m);           // push on the identity matrix
  super_multmatrix(cmdList->mat.mat);

  colorIndex = 0;
  materialIndex = 0;
  float textsize = 1;
  pointSize = 1;
  lineWidth = 1;
  lineStyle = ::SOLIDLINE;
  sphereResolution = 4;
  sphereStyle = 1;
 
  // set the material properties
  super_set_material(cmdList->materialtag);
  mat_ambient   = cmdList->ambient;
  mat_specular  = cmdList->specular;
  mat_diffuse   = cmdList->diffuse;
  mat_shininess = cmdList->shininess;
  mat_mirror    = cmdList->mirror;
  mat_opacity   = cmdList->opacity;
  mat_outline   = cmdList->outline;
  mat_outlinewidth = cmdList->outlinewidth;
  mat_transmode = cmdList->transmode;

  for (i=0; i<VMD_MAX_CLIP_PLANE; i++) {
    clip_mode[i] = cmdList->clipplanes[i].mode;
    memcpy(&clip_center[i][0], &cmdList->clipplanes[i].center, 3L*sizeof(float));
    memcpy(&clip_normal[i][0], &cmdList->clipplanes[i].normal, 3L*sizeof(float));
    memcpy(&clip_color[i][0],  &cmdList->clipplanes[i].color,  3L*sizeof(float));
  }
  start_clipgroup();

  // initialize text offset variables.  These values should never be
  // set in one display list and applied in another, so we reset the
  // variables to zero here at the start of each display list.
  textoffset_x = 0;
  textoffset_y = 0;

  // Compute periodic image transformation matrices
  ResizeArray<Matrix4> pbcImages;
  find_pbc_images(cmdList, pbcImages);
  int npbcimages = pbcImages.num();

  // Retreive instance image transformation matrices
  ResizeArray<Matrix4> instanceImages;
  find_instance_images(cmdList, instanceImages);
  int ninstances = instanceImages.num();

for (int pbcimage = 0; pbcimage < npbcimages; pbcimage++) {
 transMat.dup();
 super_multmatrix(pbcImages[pbcimage].mat);

for (int instanceimage = 0; instanceimage < ninstances; instanceimage++) {
 transMat.dup();
 super_multmatrix(instanceImages[instanceimage].mat);

  VMDDisplayList::VMDLinkIter cmditer;
  cmdList->first(&cmditer);
  while ((tok = cmdList->next(&cmditer, cmdptr))  != DLASTCOMMAND) {
    switch (tok) {   // plot a point
    case DPOINT:
      point(((DispCmdPoint *)cmdptr)->pos);  
      break;

    case DPOINTARRAY:
      {
      DispCmdPointArray *pa = (DispCmdPointArray *)cmdptr;
      float *centers, *colors;
      pa->getpointers(centers, colors);
      point_array(pa->numpoints, pa->size, centers, colors);
      }
      break;

    case DLITPOINTARRAY:
      {
      DispCmdLitPointArray *pa = (DispCmdLitPointArray *)cmdptr;
      float *centers, *normals, *colors;
      pa->getpointers(centers, normals, colors);
      point_array_lit(pa->numpoints, pa->size, centers, normals, colors);
      }
      break;

    case DSPHERE:    // draw a sphere
      sphere((float *)cmdptr);  
      break;

    case DSPHEREARRAY:     
      {
      DispCmdSphereArray *sa = (DispCmdSphereArray *)cmdptr;
      float *centers, *radii, *colors;
      sa->getpointers(centers, radii, colors);
      sphere_array(sa->numspheres, sa->sphereres, centers, radii, colors);
      }
      break;

#ifdef VMDLATTICECUBES
    case DCUBEARRAY:     
      {
      DispCmdLatticeCubeArray *ca = (DispCmdLatticeCubeArray *)cmdptr;
      float *centers, *radii, *colors;
      ca->getpointers(centers, radii, colors);
      cube_array(ca->numcubes, centers, radii, colors);
      }
      break;
#endif

    case DLINE:    // plot a line
      // don't draw degenerate lines of zero length
      if (memcmp(cmdptr, cmdptr+3L*sizeof(float), 3L*sizeof(float))) {
	line((float *)cmdptr, ((float *)cmdptr) + 3);
      }
      break;
   
    case DLINEARRAY: // array of lines
      {
      float *v = (float *) cmdptr;
      int nlines = (int)v[0];
      v++; // vertex array begins on next floating point word
      line_array(nlines, (float) lineWidth, v);
      }
      break; 

    case DPOLYLINEARRAY: // array of lines
      {
      float *v = (float *) cmdptr;
      int nlines = (int)v[0];
      v++; // vertex array begins on next floating point word
      polyline_array(nlines, (float) lineWidth, v);
      }
      break; 

    case DCYLINDER: // plot a cylinder
      if (memcmp(cmdptr, cmdptr+3L*sizeof(float), 3L*sizeof(float))) {
	cylinder((float *)cmdptr, ((float *)cmdptr) + 3, ((float *)cmdptr)[6],
		 ((int) ((float *) cmdptr)[8]));
      }
      break;

    case DCONE:      // plot a cone  
      {
      DispCmdCone *cmd = (DispCmdCone *)cmdptr;
      if (memcmp(cmd->pos1, cmd->pos2, 3L*sizeof(float))) 
	cone_trunc(cmd->pos1, cmd->pos2, cmd->radius, cmd->radius2, cmd->res);
      }
      break;
   
    case DTRIANGLE:    // plot a triangle
      {
      DispCmdTriangle *cmd = (DispCmdTriangle *)cmdptr;
      triangle(cmd->pos1,cmd->pos2,cmd->pos3,
               cmd->norm1, cmd->norm2, cmd->norm3);
      }
      break;

    case DTRIMESH_C3F_N3F_V3F: // draw a triangle mesh
      {
      DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
      float *c=NULL, *n=NULL, *v=NULL;

      if (cmd->pervertexcolors) {
        cmd->getpointers(c, n, v);
        trimesh_c3f_n3f_v3f(c, n, v, cmd->numfacets); 
      } else if (cmd->pervertexnormals) {
        cmd->getpointers(n, v);
        trimesh_n3f_v3f(n, v, cmd->numfacets); 
      } else {
        cmd->getpointers(n, v);
        trimesh_n3fopt_v3f(n, v, cmd->numfacets); 
      }
      }
      break;

    case DTRIMESH_C4F_N3F_V3F: // draw a triangle mesh
      {
      DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
      float *cnv=NULL;
      int *f=NULL;
      cmd->getpointers(cnv, f);
      trimesh_c4n3v3(cmd->numverts, cnv, cmd->numfacets, f); 
      }
      break;

    case DTRIMESH_C4U_N3F_V3F: // draw a triangle mesh
      {
      DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
      unsigned char *c=NULL; 
      float *n=NULL, *v=NULL;

      if (cmd->pervertexcolors) {
        cmd->getpointers(c, n, v);
        trimesh_c4u_n3f_v3f(c, n, v, cmd->numfacets); 
      } else {
        cmd->getpointers(n, v);
        trimesh_n3f_v3f(n, v, cmd->numfacets); 
      }
      }
      break;

    case DTRIMESH_C4U_N3B_V3F: // draw a triangle mesh
      {
      DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
      unsigned char *c=NULL; 
      char *n=NULL;
      float *v=NULL;

      if (cmd->pervertexcolors) {
        cmd->getpointers(c, n, v);
        trimesh_c4u_n3b_v3f(c, n, v, cmd->numfacets); 
      } else {
        cmd->getpointers(n, v);
        trimesh_n3b_v3f(n, v, cmd->numfacets); 
      }
      }
      break;

    case DTRISTRIP:     // draw a triangle strip
      {
      DispCmdTriStrips *cmd = (DispCmdTriStrips *) cmdptr;
      float *cnv=NULL;
      int *f=NULL;
      int *vertsperstrip;
      cmd->getpointers(cnv, f, vertsperstrip);
      tristrip(cmd->numverts, cnv, cmd->numstrips, vertsperstrip, f); 
      }
      break;

    case DWIREMESH:     // draw a triangle mesh in wireframe
      {
      DispCmdWireMesh *cmd = (DispCmdWireMesh *) cmdptr;
      float *cnv=NULL;
      int *l=NULL;
      cmd->getpointers(cnv, l);
      wiremesh(cmd->numverts, cnv, cmd->numlines, l); 
      }
      break;

    case DSQUARE:      // plot a square (norm, 4 verticies
      {
      DispCmdSquare *cmd = (DispCmdSquare *)cmdptr;
      square(cmd->norml, cmd->pos1, cmd->pos2, cmd->pos3, cmd->pos4);
      }
      break;


    ///////////// keep track of state information as well
    case DLINEWIDTH:   //  set the line width (and in superclass)
      lineWidth = ((DispCmdLineWidth *)cmdptr)->width;
      set_line_width(lineWidth);
      break;

    case DLINESTYLE:   // set the line style (and in superclass)
      lineStyle = ((DispCmdLineType *)cmdptr)->type;
      set_line_style(lineStyle);
      break;

    case DSPHERERES:   // sphere resolution (and in superclass)
      sphereResolution = ((DispCmdSphereRes *)cmdptr)->res;
      set_sphere_res(sphereResolution);
      break;

    case DSPHERETYPE:   // sphere resolution (and in superclass)
      sphereStyle = ((DispCmdSphereType *)cmdptr)->type;
      set_sphere_style(sphereStyle);
      break;

    case DMATERIALON:
      super_materials(1); 
      break;

    case DMATERIALOFF:
      super_materials(0); 
      break;

    case DCOLORINDEX:  // change the color
      super_set_color(((DispCmdColorIndex *)cmdptr)->color);
      break; 

    case DTEXTSIZE:
      textsize = ((DispCmdTextSize *)cmdptr)->size;
      break;

    case DTEXTOFFSET:
      textoffset_x = ((DispCmdTextOffset *)cmdptr)->x;
      textoffset_y = ((DispCmdTextOffset *)cmdptr)->y;
      break;

    case DTEXT: 
      {
      float *pos = (float *)cmdptr;
      text(pos, textsize, pos[3], (char *) (pos+4));
      }
      break;

    case DBEGINREPGEOMGROUP:
      beginrepgeomgroup((char *)cmdptr);
      break;

    case DCOMMENT:
      comment((char *)cmdptr);
      break;

    // pick selections (only one implemented)
    case DPICKPOINT:
      pick_point(((DispCmdPickPoint *)cmdptr)->postag,
                 ((DispCmdPickPoint *)cmdptr)->tag); 
      break;

    case DPICKPOINT_ARRAY:
      { 
        int i;
        DispCmdPickPointArray *cmd =  ((DispCmdPickPointArray *)cmdptr);
        float *crds=NULL;
        int *indices=NULL;
        cmd->getpointers(crds, indices);
        if (cmd->allselected) {
          for (i=0; i<cmd->numpicks; i++) {
            pick_point(crds + i*3L, i);
          }
        } else {
          for (i=0; i<cmd->numpicks; i++) {
            pick_point(crds + i*3L, indices[i]); 
          }
        }
      }
      break;

    // generate warnings if any geometry token is unimplemented the renderer
#if 0
    case DSTRIPETEXON:
    case DSTRIPETEXOFF:
#endif
    case DVOLUMETEXTURE:
      {
      DispCmdVolumeTexture *cmd = (DispCmdVolumeTexture *)cmdptr;
      float xplaneeq[4];
      float yplaneeq[4];
      float zplaneeq[4];
      int i;

      // automatically generate texture coordinates by translating from
      // model coordinate space to volume coordinates.
      for (i=0; i<3; i++) {
        xplaneeq[i] = cmd->v1[i];
        yplaneeq[i] = cmd->v2[i];
        zplaneeq[i] = cmd->v3[i];
      }
      xplaneeq[3] = cmd->v0[0];
      yplaneeq[3] = cmd->v0[1];
      zplaneeq[3] = cmd->v0[2];

      // define a volumetric texture map
      define_volume_texture(cmd->ID, cmd->xsize, cmd->ysize, cmd->zsize,
                            xplaneeq, yplaneeq, zplaneeq,
                            cmd->texmap);
      volume_texture_on(1);
      }
      break;

    case DVOLSLICE:
      {
      // Since OpenGL is using texture-replace here, we emulate that
      // by disabling lighting altogether
      super_materials(0); 
      DispCmdVolSlice *cmd = (DispCmdVolSlice *)cmdptr;
      volume_texture_on(1);
      square(cmd->normal, cmd->v, cmd->v + 3, cmd->v + 6, cmd->v + 9); 
      volume_texture_off();
      super_materials(1); 
      }
      break;

    case DVOLTEXON:
      volume_texture_on(0);
      break;

    case DVOLTEXOFF:
      volume_texture_off();
      break;

#if 0
    // generate warnings if any geometry token is unimplemented the renderer
    default:
      warningflags |= FILERENDERER_NOMISCFEATURE;
      break;
#endif
    } // end of switch statement
  } // while (tok != DLASTCOMMAND)

 transMat.pop();
} // end of loop over instance images

 transMat.pop();
} // end of loop over periodic images

  end_clipgroup();
}

////////////////////////////////////////////////////////////////////


// change the active color
void FileRenderer::super_set_color(int color_index) {
  if (colorIndex != color_index) {
    colorIndex = color_index;
    set_color(color_index);
  }
}

// change the active material
void FileRenderer::super_set_material(int material_index) {
  if (materialIndex != material_index) {
    materialIndex = material_index;
    set_material(material_index);
  }
}

// turn materials on or off
void FileRenderer::super_materials(int on_or_off) {
  if (on_or_off) {
    materials_on = 1;
    activate_materials();
  } else {
    materials_on = 0;
    deactivate_materials();
  }
}


//////////////// change the viewing matrix array state ////////////////

void FileRenderer::super_load(float *cmdptr) {
  Matrix4 tmp(cmdptr);
  (transMat.top()).loadmatrix(tmp);
  load(tmp);
}
void FileRenderer::super_multmatrix(const float *cmdptr) {
  Matrix4 tmp(cmdptr);
  (transMat.top()).multmatrix(tmp);
  multmatrix(tmp);
}

void FileRenderer::super_translate(float *cmdptr) {
  (transMat.top()).translate( cmdptr[0], cmdptr[1], cmdptr[2]);
  translate( cmdptr[0], cmdptr[1], cmdptr[2]);
}

void FileRenderer::super_rot(float * cmdptr) {
  (transMat.top()).rot( cmdptr[0], 'x' + (int) (cmdptr[1]) );
  rot( cmdptr[0], 'x' + (int) (cmdptr[1]) );
}

void FileRenderer::super_scale(float *cmdptr) {
  (transMat.top()).scale( cmdptr[0], cmdptr[1], cmdptr[2] );
  scale( cmdptr[0], cmdptr[1], cmdptr[2] );
}

void FileRenderer::super_scale(float s) {
  transMat.top().scale(s,s,s);
  scale(s,s,s);
}

// return global scaling factor (used for sphere radii, similar)
float FileRenderer::scale_factor(void) {
  // of course, VMD does not have a direction-independent scaling
  // factor, so I'll fake it using an average of the scaling
  // factors in each direction.
  float scaleFactor;

  float *mat =  &transMat.top().mat[0];
  scaleFactor = (sqrtf(mat[0]*mat[0] + mat[4]*mat[4] + mat[ 8]*mat[ 8]) +
                 sqrtf(mat[1]*mat[1] + mat[5]*mat[5] + mat[ 9]*mat[ 9]) +
                 sqrtf(mat[2]*mat[2] + mat[6]*mat[6] + mat[10]*mat[10])) / 3.0f;

  return scaleFactor;
}

// scale the radius a by the global scaling factor, return as b.
float FileRenderer::scale_radius(float r) {
  float scaleFactor = scale_factor();
  if (r < 0.0) {
    msgErr << "FileRenderer: Error, Negative radius" << sendmsg;
    r = -r;
  } 
  r = r * scaleFactor;
  return r;
}


////// set the exec command 
void FileRenderer::set_exec_string(const char *extstr) {
  delete [] execCmd;
  execCmd = stringdup(extstr);
}


// default triangulated implementation of two-radius cones
void FileRenderer::cone_trunc(float *base, float *apex, float radius, float radius2, int numsides) {
  int h;
  float theta, incTheta, cosTheta, sinTheta;
  float axis[3], temp[3], perp[3], perp2[3];
  float vert0[3], vert1[3], vert2[3], edge0[3], edge1[3], face0[3], face1[3], norm0[3], norm1[3];

  axis[0] = base[0] - apex[0];
  axis[1] = base[1] - apex[1];
  axis[2] = base[2] - apex[2];
  vec_normalize(axis);

  // Find an arbitrary vector that is not the axis and has non-zero length
  temp[0] = axis[0] - 1.0f;
  temp[1] = 1.0f;
  temp[2] = 1.0f;

  // use the cross product to find orthogonal vectors
  cross_prod(perp, axis, temp);
  vec_normalize(perp);
  cross_prod(perp2, axis, perp); // shouldn't need normalization

  // Draw the triangles
  incTheta = (float) VMD_TWOPI / numsides;
  theta = 0.0;

  // if radius2 is larger than zero, we will draw quadrilateral
  // panels rather than triangular panels
  if (radius2 > 0) {
    float negaxis[3], offsetL[3], offsetT[3], vert3[3];
    int filled=1;
    vec_negate(negaxis, axis);
    memset(vert0, 0, sizeof(vert0));
    memset(vert1, 0, sizeof(vert1));
    memset(norm0, 0, sizeof(norm0));

    for (h=0; h <= numsides+3; h++) {
      // project 2-D unit circles onto perp/perp2 3-D basis vectors
      // and scale to desired radii
      cosTheta = (float) cosf(theta);
      sinTheta = (float) sinf(theta);
      offsetL[0] = radius2 * (cosTheta*perp[0] + sinTheta*perp2[0]);
      offsetL[1] = radius2 * (cosTheta*perp[1] + sinTheta*perp2[1]);
      offsetL[2] = radius2 * (cosTheta*perp[2] + sinTheta*perp2[2]);
      offsetT[0] = radius  * (cosTheta*perp[0] + sinTheta*perp2[0]);
      offsetT[1] = radius  * (cosTheta*perp[1] + sinTheta*perp2[1]);
      offsetT[2] = radius  * (cosTheta*perp[2] + sinTheta*perp2[2]);

      // copy old vertices
      vec_copy(vert2, vert0); 
      vec_copy(vert3, vert1); 
      vec_copy(norm1, norm0); 

      // calculate new vertices
      vec_add(vert0, base, offsetT);
      vec_add(vert1, apex, offsetL);

      // Use the new vertex to find new edges
      edge0[0] = vert0[0] - vert1[0];
      edge0[1] = vert0[1] - vert1[1];
      edge0[2] = vert0[2] - vert1[2];
      edge1[0] = vert0[0] - vert2[0];
      edge1[1] = vert0[1] - vert2[1];
      edge1[2] = vert0[2] - vert2[2];

      // Use the new edge to find a new facet normal
      cross_prod(norm0, edge1, edge0);
      vec_normalize(norm0);

      if (h > 2) {
	// Use the new normal to draw the previous side
	triangle(vert0, vert3, vert1, norm0, norm1, norm0);
	triangle(vert3, vert0, vert2, norm1, norm0, norm1);

	// Draw cylinder caps
	if (filled & CYLINDER_LEADINGCAP) {
	  triangle(vert1, vert3, apex, axis, axis, axis);
	}
	if (filled & CYLINDER_TRAILINGCAP) {
	  triangle(vert0, vert2, base, negaxis, negaxis, negaxis);
	}
      }

      theta += incTheta;
    }
  } else {
    // radius2 is zero, so we draw triangular panels joined at the apex
    for (h=0; h < numsides+3; h++) {
      // project 2-D unit circle onto perp/perp2 3-D basis vectors
      // and scale to desired radius
      cosTheta = (float) cosf(theta);
      sinTheta = (float) sinf(theta);
      vert0[0] = base[0] + radius * (cosTheta*perp[0] + sinTheta*perp2[0]);
      vert0[1] = base[1] + radius * (cosTheta*perp[1] + sinTheta*perp2[1]);
      vert0[2] = base[2] + radius * (cosTheta*perp[2] + sinTheta*perp2[2]);

      // Use the new vertex to find a new edge
      edge0[0] = vert0[0] - apex[0];
      edge0[1] = vert0[1] - apex[1];
      edge0[2] = vert0[2] - apex[2];

      if (h > 0) {
	// Use the new edge to find a new face
	cross_prod(face0, edge1, edge0);
	vec_normalize(face0);

	if (h > 1) {
	  // Use the new face to find the normal of the previous triangle
	  norm0[0] = (face1[0] + face0[0]) * 0.5f;
	  norm0[1] = (face1[1] + face0[1]) * 0.5f;
	  norm0[2] = (face1[2] + face0[2]) * 0.5f;
	  vec_normalize(norm0);

	  if (h > 2) {
	    // Use the new normal to draw the previous side and base of the cone
	    triangle(vert2, vert1, apex, norm1, norm0, face1);
	    triangle(vert2, vert1, base, axis, axis, axis);
	  }

	}

	// Copy the old values
	memcpy(norm1, norm0, 3L*sizeof(float));
	memcpy(vert2, vert1, 3L*sizeof(float));
	memcpy(face1, face0, 3L*sizeof(float));
      }
      memcpy(vert1, vert0, 3L*sizeof(float));
      memcpy(edge1, edge0, 3L*sizeof(float));
  
      theta += incTheta;
    }
  }
}


// default trianglulated cylinder implementation, with optional end caps
void FileRenderer::cylinder(float *base, float *apex, float radius, int filled) {
  const int numsides = 20;
  int h;
  float theta, incTheta, cosTheta, sinTheta;
  float axis[3], negaxis[3], temp[3], perp[3], perp2[3], offset[3];

  axis[0] = base[0] - apex[0];
  axis[1] = base[1] - apex[1];
  axis[2] = base[2] - apex[2];
  vec_normalize(axis);
  vec_negate(negaxis, axis);

  // Find an arbitrary vector that is not the axis and has non-zero length
  temp[0] = axis[0] - 1.0f;
  temp[1] = 1.0f;
  temp[2] = 1.0f;

  // use the cross product to find orthogonal vectors
  cross_prod(perp, axis, temp);
  vec_normalize(perp);
  cross_prod(perp2, axis, perp); // shouldn't need normalization

  // Draw the triangles
  incTheta = (float) VMD_TWOPI / numsides;
  theta = 0.0;

  const int stripsz = (2L*numsides) * 6L;
  float *stripnv = new float[stripsz];
  memset(stripnv, 0, sizeof(float) * stripsz);

  const int capsz = numsides+1;
  float *lcapnv = new float[capsz*6L];
  memset(lcapnv, 0, sizeof(float)*capsz*6L);
  vec_copy(&lcapnv[0], negaxis);
  vec_copy(&lcapnv[3], apex);

  float *tcapnv = new float[capsz*6L];
  memset(tcapnv, 0, sizeof(float)*capsz*6L);
  vec_copy(&tcapnv[0], axis);
  vec_copy(&tcapnv[3], base);

  for (h=0; h < numsides; h++) {
    // project 2-D unit circle onto perp/perp2 3-D basis vectors
    // and scale to desired radius
    cosTheta = (float) cosf(theta);
    sinTheta = (float) sinf(theta);
    offset[0] = radius * (cosTheta*perp[0] + sinTheta*perp2[0]);
    offset[1] = radius * (cosTheta*perp[1] + sinTheta*perp2[1]);
    offset[2] = radius * (cosTheta*perp[2] + sinTheta*perp2[2]);

    // calculate new vertices
    float lvert[3], tvert[3];
    vec_add(lvert, apex, offset);
    vec_add(tvert, base, offset);
    vec_copy(&stripnv[((2L*h)*6L)+9L], lvert); 
    vec_copy(&stripnv[((2L*h)*6L)+3L], tvert); 
    vec_copy(&lcapnv[((1L+h)*6L)+3L], lvert); 
    vec_copy(&tcapnv[((1L+h)*6L)+3L], tvert); 

    // new normals
    vec_normalize(offset);
    vec_copy(&stripnv[((2L*h)*6L)  ], offset);
    vec_copy(&stripnv[((2L*h)*6L)+6], offset);
    vec_copy(&lcapnv[((1+h)*6L)], negaxis);
    vec_copy(&tcapnv[((1+h)*6L)], axis);

    theta += incTheta;
  }

  const int vertsperstrip = (numsides + 1)*2L;
  int *stripfaces = new int[vertsperstrip];
  memset(stripfaces, 0, sizeof(float) * vertsperstrip);
  for (h=0; h < vertsperstrip-2; h++) {
    stripfaces[h] = h;
  }
  stripfaces[h  ] = 0; // wraparound to start
  stripfaces[h+1] = 1; // wraparound to start

  tristrip_singlecolor(2L*numsides, &stripnv[0],
                       1, &colorIndex, &vertsperstrip, &stripfaces[0]);
  delete [] stripfaces;

  // Draw cylinder caps
  if (filled & CYLINDER_LEADINGCAP) {
    const int vertsperfan = capsz+1;
    int *fanfaces = new int[vertsperfan];
    memset(fanfaces, 0, sizeof(float) * vertsperfan);
    fanfaces[0] = 0;
    for (h=1; h < capsz; h++) {
      fanfaces[h] = capsz-h;
    }
    fanfaces[h] = capsz-1; // wraparound to start
    trifan_singlecolor(capsz-1, &lcapnv[0],
                       1, &colorIndex, &vertsperfan, &fanfaces[0]);
    delete [] fanfaces;
  }
  if (filled & CYLINDER_TRAILINGCAP) {
    const int vertsperfan = capsz+1;
    int *fanfaces = new int[vertsperfan];
    memset(fanfaces, 0, sizeof(float) * vertsperfan);
    fanfaces[0] = 0;
    for (h=1; h < capsz; h++) {
      fanfaces[h] = h;
    }
    fanfaces[h] = 1; // wraparound to start
    trifan_singlecolor(capsz-1, &tcapnv[0],
                       1, &colorIndex, &vertsperfan, &fanfaces[0]);
    delete [] fanfaces;
  }

  delete [] stripnv;
  delete [] lcapnv;
  delete [] tcapnv;
}


// default cylinder-based implementation of lines used
// for ray tracing packages that can't draw real lines
void FileRenderer::line(float * a, float * b) {
  // draw a line (cylinder) from a to b
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3]; 

  if (lineStyle == ::SOLIDLINE) {
    cylinder(a, b, lineWidth * 0.1f, 0);
  } else if (lineStyle == ::DASHEDLINE) {
    vec_sub(dirvec, b, a);        // vector from a to b
    vec_copy(unitdirvec, dirvec);
    vec_normalize(unitdirvec);    // unit vector from a to b
    test = 1;
    i = 0;
    while (test == 1) {
      for (j=0; j<3; j++) {
        from[j] = (float) (a[j] + (2*i    )* 0.05 * unitdirvec[j]);
          to[j] = (float) (a[j] + (2*i + 1)* 0.05 * unitdirvec[j]);
      }
      if (fabsf(a[0] - to[0]) >= fabsf(dirvec[0])) {
        vec_copy(to, b);
        test = 0;
      }
      cylinder(from, to, lineWidth * 0.1f, 0);
      i++;
    }
  } 
}


void FileRenderer::line_array(int num, float thickness, float *points) {
  float *v = points;
  for (int i=0; i<num; i++) {
    // don't draw degenerate lines of zero length
    if (memcmp(v, v+3, 3L*sizeof(float))) {
      line(v, v+3);
    }
    v += 6;
  }
}


void FileRenderer::polyline_array(int num, float thickness, float *points) {
  float *v = points;
  for (int i=0; i<num-1; i++) {
    // don't draw degenerate lines of zero length
    if (memcmp(v, v+3, 3L*sizeof(float))) {
      line(v, v+3);
    }
    v += 3;
  }
}


// default triangulated sphere implementation
void FileRenderer::sphere(float * xyzr) {
  float c[3], r;
  int pi, ni;
  int i;
  int sph_iter = -1;
  int sph_desired_iter = 0;

  // copy params
  vec_copy(c, xyzr);
  r = xyzr[3];

  // the sphere resolution has changed. if sphereRes is less than 32, we
  // will use a lookup table to achieve equal or better resolution than
  // OpenGL. otherwise we use the following equation:
  //    iterations = .9 *
  //    (sphereRes)^(1/2)
  // This is used as a lookup table to determine the proper
  // number of iterations used in the sphere approximation
  // algorithm.
  const int sph_iter_table[] = {
      0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4 };

  if (sphereResolution < 0) 
    return;
  else if (sphereResolution < 32) 
    sph_desired_iter = sph_iter_table[sphereResolution];
  else 
    sph_desired_iter = (int) (0.8f * sqrtf((float) sphereResolution));
 
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

    newverts = (float *) malloc(sizeof(float) * 36L);
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
      newverts = (float *) malloc(sizeof(float) * 12L * nverts);
      if (!newverts) {
        // memory error
        sph_iter = -1;
        sph_nverts = 0;
        sph_verts = NULL;
        free(oldverts);
        msgErr << "FileRenderer::sphere(): Out of memory. Some "
               << "objects were not drawn." << sendmsg;
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
        memcpy(&newverts[ni     ], &oldverts[pi], sizeof(float) * 3L);
        memcpy(&newverts[ni + 3 ], b, sizeof(float) * 3L);
        memcpy(&newverts[ni + 6 ], a, sizeof(float) * 3L);

        memcpy(&newverts[ni + 9 ], b, sizeof(float) * 3L);
        memcpy(&newverts[ni + 12], &oldverts[pi + 3], sizeof(float) * 3L);
        memcpy(&newverts[ni + 15], c, sizeof(float) * 3L);

        memcpy(&newverts[ni + 18], a, sizeof(float) * 3L);
        memcpy(&newverts[ni + 21], b, sizeof(float) * 3L);
        memcpy(&newverts[ni + 24], c, sizeof(float) * 3L);

        memcpy(&newverts[ni + 27], a, sizeof(float) * 3L);
        memcpy(&newverts[ni + 30], c, sizeof(float) * 3L);
        memcpy(&newverts[ni + 33], &oldverts[pi + 6], sizeof(float) * 3L);

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
  // desired position and radius, and add the triangles
  pi = 0;
  for (i = 0; i < sph_nverts / 3; i++) {
    float v0[3], v1[3], v2[3];
    float n0[3], n1[3], n2[3];

    // calculate upper hemisphere translation and scaling
    v0[0] = r * sph_verts[pi    ] + c[0];
    v0[1] = r * sph_verts[pi + 1] + c[1];
    v0[2] = r * sph_verts[pi + 2] + c[2];
    v1[0] = r * sph_verts[pi + 3] + c[0];
    v1[1] = r * sph_verts[pi + 4] + c[1];
    v1[2] = r * sph_verts[pi + 5] + c[2];
    v2[0] = r * sph_verts[pi + 6] + c[0];
    v2[1] = r * sph_verts[pi + 7] + c[1];
    v2[2] = r * sph_verts[pi + 8] + c[2];

    // calculate upper hemisphere normals
    vec_copy(n0, &sph_verts[pi    ]);
    vec_copy(n1, &sph_verts[pi + 3]);
    vec_copy(n2, &sph_verts[pi + 6]);

    // draw upper hemisphere
    triangle(v0, v2, v1, n0, n2, n1);

    // calculate lower hemisphere translation and scaling
    v0[2] = (-r * sph_verts[pi + 2]) + c[2];
    v1[2] = (-r * sph_verts[pi + 5]) + c[2];
    v2[2] = (-r * sph_verts[pi + 8]) + c[2];

    // calculate lower hemisphere normals
    n0[2] = -n0[2];
    n1[2] = -n1[2];
    n2[2] = -n2[2];

    // draw lower hemisphere
    triangle(v0, v1, v2, n0, n1, n2);

    pi += 9;
  }
}


// render a bunch of spheres that share the same material properties,
// and resolution parameters, differing only in their individual
// positions, radii, and colors.
void FileRenderer::sphere_array(int spnum, int spres, 
                                float *centers, float *radii, float *colors) {
  int i, ind;
  set_sphere_res(spres); // set the current sphere resolution
  ind = 0;
  for (i=0; i<spnum; i++) {
    float xyzr[4];
    xyzr[0]=centers[ind    ];
    xyzr[1]=centers[ind + 1];
    xyzr[2]=centers[ind + 2];
    xyzr[3]=radii[i];

    super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2]));
    sphere(xyzr);
    ind += 3; // next sphere
  }
}


// render a bunch of cubes that share the same material properties,
// differing only in their individual positions, radii (half side length), 
// and colors.
void FileRenderer::cube_array(int cbnum, float *centers, float *radii, float *colors) {
  int i, ind;
  ind = 0;
  for (i=0; i<cbnum; i++) {
    float xyzr[4];
    xyzr[0]=centers[ind    ];
    xyzr[1]=centers[ind + 1];
    xyzr[2]=centers[ind + 2];
    xyzr[3]=radii[i];

    super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2]));
    cube(xyzr);
    ind += 3; // next sphere
  }
}


// render a bunch of points that share the same size differing only
// in their positions and colors
void FileRenderer::point_array(int num, float size, float *xyz, float *colors) {
  int i, ind;

  pointSize = (int) size;     // set the point size

  // draw all of the points
  for (ind=0,i=0; i<num; i++) {
    super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2]));
    point(&xyz[ind]);
    ind += 3;
  }

  pointSize = 1;        // reset the point size
}


// render a bunch of lighted points that share the same size differing only
// in their positions and colors
void FileRenderer::point_array_lit(int num, float size, 
                                   float *xyz, float *norms, float *colors) {
  // XXX none of the existing scene formats are able to describe
  //     shaded point sets, so we just draw unlit points
  point_array(num, size, xyz, colors);
}


// start rendering geometry for which user-defined
// clipping planes have been applied.
void FileRenderer::start_clipgroup() {
  int i;

  for (i=0; i<VMD_MAX_CLIP_PLANE; i++) {
    if (clip_mode[i]) {
      warningflags |= FILERENDERER_NOCLIP;
      break;
    }
  }
}


void FileRenderer::text(float *pos, float size, float thickness, const char *str) {
#if 1
  // each subclass has to provide its own text() method, otherwise no text
  // will be emitted, and we need to warn the user
  warningflags |= FILERENDERER_NOTEXT;
#else
  hersheyhandle hh;
  float lm, rm, x, y, ox, oy;
  int draw, odraw;

  Matrix4 m;
  transMat.push(m);           // push on the identity matrix

  while (*str != '\0') {
    hersheyDrawInitLetter(&hh, *str, &lm, &rm);
    (transMat.top()).translate(-lm, 0, 0);
    ox=0;
    oy=0;
    odraw=0;
    while (!hersheyDrawNextLine(&hh, &draw, &x, &y)) {
      if (draw && odraw) {
        float a[3], b[3];
        a[0] = ox; 
        a[1] = oy;
        a[2] = 0;
        b[0] = x;
        b[1] = y;
        b[2] = 0;

  //      printf("line: %g %g -> %g %g\n", ox, oy, x, y);
        line(a, b);        
      }

      ox=x;
      oy=y;
      odraw=draw;
    }
    (transMat.top()).translate(rm, 0, 0);

    str++;
  }

  transMat.pop();
#endif
}


