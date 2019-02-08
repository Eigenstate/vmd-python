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
 *	$RCSfile: FileRenderer.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.127 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The FileRenderer class implements the data and functions needed to 
 * render a scene to a file in some format (postscript, raster3d, etc.)
 *
 ***************************************************************************/
#ifndef FILERENDERER_H
#define FILERENDERER_H

#include <stdio.h>

#include "DisplayDevice.h"
#include "Scene.h"
#include "NameList.h"
#include "Inform.h"

#define FILERENDERER_NOWARNINGS    0
#define FILERENDERER_NOMISCFEATURE 1
#define FILERENDERER_NOCLIP        2
#define FILERENDERER_NOCUEING      4
#define FILERENDERER_NOTEXTURE     8
#define FILERENDERER_NOGEOM       16
#define FILERENDERER_NOTEXT       32

/// This is the base class for all the renderers that go to a
/// file and are on the render list.  There are five operations
/// available to the outside world
class FileRenderer : public DisplayDevice {
protected:
  char *publicName;         ///< scripting name of renderer (no spaces)
  char *publicPrettyName;   ///< name of renderer for use in GUIs
  char *defaultFilename;    ///< default output filename
  char *defaultCommandLine; ///< default rendering command

  char *execCmd;     ///< current version of the post-render command
  FILE *outfile;     ///< the current file
  int isOpened;      ///< is the file opened correctly
  char *my_filename; ///< the current filename
  int has_aa;        ///< supports antialiasing; off by default
  int aasamples;     ///< antialiasing samples, -1 if unsupported.
  int aosamples;     ///< ambient occlusion samples, -1 if unsupported.
  int has_imgsize;   ///< True if the renderer can produce an arbitrary-sized
                     ///< image; false by default.
  int warningflags;  ///< If set, emit a warning message that this
                     ///< subclass doesn't support all of the render features
                     ///< in use by the current scene
  int imgwidth, imgheight;  ///< desired size of image
  float aspectratio;        ///< Desired aspect ratio.
  NameList<int> formats;    ///< Output formats supported by this renderer
  int curformat;     ///< Currently selected format.

  float textoffset_x; ///< label text offset
  float textoffset_y; ///< label text offset

  /// Renderer-specific function to update execCmd based on the current state
  /// of aasamples, image size, etc.  Default implementation is to do nothing.
  virtual void update_exec_cmd() {}

  /// light state, passed to renderer before render commands are executed.
  struct LightState {
    float color[3];             ///< RGB color of the light
    float pos[3];               ///< Position (or direction) of the light 
    int on;                     ///< on/off state of light
  };
  LightState lightState[DISP_LIGHTS]; ///< state of all lights

  /// AdvancedLight state data
  struct AdvancedLightState {
    float color[3];             ///< RGB color of the light
    float pos[3];               ///< Position (or direction) of the light
    float constfactor;          ///< constant light factor
    float linearfactor;         ///< linear light factor
    float quadfactor;           ///< quadratic light factor
    float spotdir[3];           ///< spotlight direction
    float fallstart;            ///< spotlight falloff starting radius (radians)
    float fallend;              ///< spotlight falloff starting radius (radians)
    int spoton;                 ///< spotlighting enable flag
    int on;                     ///< on/off state of light
  };
  AdvancedLightState advLightState[DISP_LIGHTS]; ///< state of advanced lights

  /// color state, copied into here when do_use_colors is called
  float matData[MAXCOLORS][3];
  virtual void do_use_colors();

  /// background color, copied into here with set_background is called
  float backColor[3];

  float backgradientenabled;      ///< flag indicating background gradient use
  float backgradienttopcolor[3];  ///< top edge color of background gradient
  float backgradientbotcolor[3];  ///< bottom edge color of background gradient
 
public:
  /// create the renderer; set the 'visible' name for the renderer list
  FileRenderer(const char *public_name, 
               const char *public_pretty_name,
               const char *default_file_name,
	       const char *default_command_line);
  virtual ~FileRenderer(void);

  const char *visible_name(void) const { return publicName;}
  const char *pretty_name(void) const { return publicPrettyName;}
  const char *default_filename(void) const {return defaultFilename;}
  const char *default_exec_string(void) const {return defaultCommandLine;}
  const char *saved_exec_string(void) const { return execCmd; }

  void set_exec_string(const char *);

  /// Supports anti-aliasing?
  int has_antialiasing() const { return has_aa; }

  /// Get/set the AA level; return the new value.  Must be non-negative.
  int set_aasamples(int newval) {
    if (has_aa && (newval >= 0)) {
      aasamples = newval;
      update_exec_cmd();
    }
    return aasamples;
  }

  /// Get/set the AO samples; return the new value.  Must be non-negative.
  int set_aosamples(int newval) {
    if (newval >= 0) {
      aosamples = newval;
      update_exec_cmd();
    }
    return aosamples;
  }

  /// Supports arbitrary image size?
  int has_imagesize() const { return has_imgsize; }

  /// Get/set the image size.   Return success and places the current values in
  /// the passed-in pointers.  May fail if the renderer is not able to specify 
  /// the image size (e.g. snapshot).  Passing 0,0 just returns the current 
  /// values.
  int set_imagesize(int *w, int *h);

  /// Set the aspect ratio.  Negative values ignored.  Returns the new value.
  /// Also updates image size if it has been set.
  float set_aspectratio(float aspect);
  
  /// Number of output formats
  int numformats() const { return formats.num(); }
  
  /// get/set formats
  const char *format(int i) const { return formats.name(i); }
  const char *format() const { return formats.name(curformat); }
  int set_format(const char *format) {
    int ind = formats.typecode(format);
    if (ind < 0) return FALSE;
    if (curformat != ind) {
      curformat = ind;
      update_exec_cmd();
    }
    return TRUE;
  }

  /// copy in the background color
  virtual void set_background(const float *);

  /// set gradient colors
  virtual void set_backgradient(const float *top, const float *bot);

  /// open the file; don't write the header info
  /// return TRUE if opened okay
  /// if file already opened, complain, and close previous file
  /// this will also reset the state variables
  virtual int open_file(const char *filename);

  virtual int do_define_light(int n, float *color, float *position);
  virtual int do_activate_light(int n, int turnon);

  virtual int do_define_adv_light(int n, float *color, float *position,
                                  float constant, float linear, float quad,
                                  float *spotdir, float fallstart, 
                                  float fallend, int spoton); 
  virtual int do_activate_adv_light(int n, int turnon);

private:
  int sph_nverts;   ///< data for tesselating spheres with triangles
  float *sph_verts; ///< data for tesselating spheres with triangles

protected:
  /// write the header info.  This is an alias for prepare3D
  virtual void write_header(void) {};
  void reset_state(void);

public:
  virtual int prepare3D(int); 
  virtual void render(const VMDDisplayList *); // render the display list

protected:
  /// write any trailer info.  This is called by update
  virtual void write_trailer(void) {};

  /// close the file.  This is called by update, and exists
  /// due to symmetry.  Also, is called for case when open is
  /// called when a file was already open.
  virtual void close_file(void);

public:
  /// don't need to override this (unless you want to do so)
  virtual void update(int) {
    if (isOpened) {
      write_trailer();
      close_file();
      isOpened = FALSE;
  
      // Emit any pending warning messages for missing or unsupported
      // geometric primitives.
      if (warningflags & FILERENDERER_NOCLIP)
        msgWarn << "User-defined clipping planes not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOTEXT)
        msgWarn << "Text not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOTEXTURE)
        msgWarn << "Texture mapping not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOCUEING)
        msgWarn << "Depth cueing not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOGEOM)
        msgWarn << "One or more geometry types not exported for this renderer" << sendmsg;

      if (warningflags != FILERENDERER_NOWARNINGS)
        msgWarn << "Unimplemented features may negatively affect the appearance of the scene" << sendmsg;
    }
  }

protected:
  ///////////// Information about the current state //////
  // (for those that do not want to take care of it themselves)
  // the 'super_' version is called by render to set the matrix.  It
  // then calls the non-super version
  Stack<Matrix4> transMat;
  void super_load(float *cmdptr);
  virtual void load(const Matrix4& /*mat*/) {}
  void super_multmatrix(const float *cmdptr);
  virtual void multmatrix(const Matrix4& /*mat*/) {}
  void super_translate(float *cmdptr);
  virtual void translate(float /*x*/, float /*y*/, float /*z*/) {}
  void super_rot(float *cmdptr);
  virtual void rot(float /*ang*/, char /*axis*/) {}
  void super_scale(float *cmdptr);
  void super_scale(float);
  virtual void scale(float /*scalex*/, float /*scaley*/, 
		     float /*scalez*/) {}

  float scale_factor(void);         ///< return the current scaling factor
                                    ///< to use with large batches of geometry
  float scale_radius(float);        ///< apply current scaling factor to radius

  // change the color definitions
  int colorIndex;                   ///< active color index
  void super_set_color(int index);  ///< only calls set_color when index changes
  virtual void set_color(int) {}    ///< set the color index
  
  /// compute nearest index in matData using given rgb value
  /// XXX We shouldn't be doing this; a better approach would be to store the
  /// new color in the matData color table and return the new index, rather 
  /// than trying to match a 17 color palette.   
  int nearest_index(float r, float g, float b) const;

  // change the material definition
  int materialIndex;                    ///< active material index
  float mat_ambient;                    ///< active ambient value
  float mat_diffuse;                    ///< active diffuse value
  float mat_specular;                   ///< active specular value
  float mat_shininess;                  ///< active shininess value
  float mat_mirror;                     ///< active mirror value
  float mat_opacity;                    ///< active opacity value
  float mat_outline;                    ///< active outline factor
  float mat_outlinewidth;               ///< active outline width
  float mat_transmode;                  ///< active transparency mode
  void super_set_material(int index);   ///< only call set_material on idx chg
  virtual void set_material(int) {}     ///< change material index 

  float clip_center[VMD_MAX_CLIP_PLANE][3]; ///< clipping plane center
  float clip_normal[VMD_MAX_CLIP_PLANE][3]; ///< clipping plane normal
  float clip_color[VMD_MAX_CLIP_PLANE][3];  ///< clipping plane CSG color
  int clip_mode[VMD_MAX_CLIP_PLANE];        ///< clipping plane mode

  virtual void start_clipgroup();       ///< emit clipping plane group
  virtual void end_clipgroup() {}       ///< terminate clipping plane group

  // change the line definitions
  int lineWidth, lineStyle, pointSize;
  virtual void set_line_width(int new_width) {
    lineWidth = new_width;
  }
  virtual void set_line_style(int /*new_style*/) {}  ///< called by super

  // change the sphere definitions
  int sphereResolution, sphereStyle;
  virtual void set_sphere_res(int /*res*/) {}        ///< called by super
  virtual void set_sphere_style(int /*style*/) {}    ///< called by super

  int materials_on;
  void super_materials(int on_or_off);
  virtual void activate_materials(void) {}           ///< if previous is TRUE
  virtual void deactivate_materials(void) {}         ///< if super is FALSE
  

  ////////////////////// various virtual generic graphics commands

  /// draw a single-radius cone (pointy top)
  virtual void cone(float * xyz1, float * xyz2, float radius, int resolution) { 
    // if not overridden by the subclass, we just call the truncated cone
    // method with a 0.0 radius for the tip
    cone_trunc(xyz1, xyz2, radius, 0.0f, resolution);
  }

  /// draw a two-radius truncated cone
  virtual void cone_trunc(float * /*xyz1*/, float * /*xyz2*/, 
                          float /* radius*/, float /* radius2 */, 
                          int /*resolution*/);


  /// draw a cylinder, with optional caps
  virtual void cylinder(float * base, float * apex, float radius, int filled);


  /// draw a single line
  virtual void line(float * a, float * b);

  /// draw a set of lines with the same color and thickness
  virtual void line_array(int num, float thickness, float *points);

  /// draw a set of connected lines with the same color and thickness
  virtual void polyline_array(int num, float thickness, float *points);


  /// draw an unlit point
  virtual void point(float * xyz) {
    float xyzr[4];
    vec_copy(xyzr, xyz);
    xyzr[3] = lineWidth * 0.002f; // hack for renderers that don't have points
  }

  /// draw an unlighted point array
  virtual void point_array(int num, float size, float *xyz, float *colors);

  /// draw a lighted point array
  virtual void point_array_lit(int num, float size, 
                               float *xyz, float *norm, float *colors);

  /// draw a lattice cube array
  virtual void cube_array(int num, float *centers, float *radii, float *colors);

  /// draw a sphere
  virtual void sphere(float * xyzr);

  /// draw a sphere array
  virtual void sphere_array(int num, int res, float *centers, float *radii, float *colors);


  /// draw a quadrilateral
  virtual void square(float * norm, float * a, float * b, 
		      float * c, float * d) {
    // draw as two triangles, with correct winding order etc
    triangle(a, b, c, norm, norm, norm);
    triangle(a, c, d, norm, norm, norm);
  }

  /// draw an axis-aligned lattice site cube 
  virtual void cube(float * xyzr) {
    // coordinates of unit cube
    float v0[] = {-1.0, -1.0, -1.0}; 
    float v1[] = { 1.0, -1.0, -1.0}; 
    float v2[] = {-1.0,  1.0, -1.0}; 
    float v3[] = { 1.0,  1.0, -1.0}; 
    float v4[] = {-1.0, -1.0,  1.0}; 
    float v5[] = { 1.0, -1.0,  1.0}; 
    float v6[] = {-1.0,  1.0,  1.0}; 
    float v7[] = { 1.0,  1.0,  1.0}; 

    float n0[] = {0, 0,  1};
    float n1[] = {0, 0,  1};
    float n2[] = {0, -1, 0};
    float n3[] = {0, -1, 0};
    float n4[] = {1, 0, 0};
    float n5[] = {1, 0, 0};

    vec_triad(v0, xyzr, xyzr[3], v0);
    vec_triad(v1, xyzr, xyzr[3], v1);
    vec_triad(v2, xyzr, xyzr[3], v2);
    vec_triad(v3, xyzr, xyzr[3], v3);
    vec_triad(v4, xyzr, xyzr[3], v4);
    vec_triad(v5, xyzr, xyzr[3], v5);
    vec_triad(v6, xyzr, xyzr[3], v6);
    vec_triad(v7, xyzr, xyzr[3], v7);

    square(n0, v0, v1, v3, v2);
    square(n1, v4, v5, v7, v6);
    square(n2, v0, v1, v5, v4);
    square(n3, v2, v3, v7, v6);
    square(n4, v0, v2, v6, v4);
    square(n5, v1, v3, v7, v5);
  }


  /// single color triangle with interpolated surface normals
  virtual void triangle(const float * /*xyz1*/, const float * /*xyz2*/, const float * /*xyz3*/, 
                        const float * /*n1*/, const float * /*n2*/, const float * /*n3*/) {
    warningflags |= FILERENDERER_NOGEOM; // no triangles written
  }


  /// triangle with interpolated surface normals and vertex colors
  virtual void tricolor(const float * xyz1, const float * xyz2, const float * xyz3, 
                        const float * n1, const float * n2, const float * n3,
                        const float *c1, const float *c2, const float *c3) {
    int index = 1;
    float r, g, b;
    r = (c1[0] + c2[0] + c3[0]) / 3.0f; // average three vertex colors 
    g = (c1[1] + c2[1] + c3[1]) / 3.0f;
    b = (c1[2] + c2[2] + c3[2]) / 3.0f;

    index = nearest_index(r,g,b); // lookup nearest color here.
    super_set_color(index); // use the closest color

    triangle(xyz1, xyz2, xyz3, n1, n2, n3); // draw a regular triangle   
  }


  /// triangle mesh built from a vertex array
  virtual void trimesh_n3f_v3f(float *n, float *v, int numfacets) { 
    int i;
    for (i=0; i<numfacets*9; i+=9) {
      triangle(v + i    , 
               v + i + 3, 
               v + i + 6,
               n + i    , 
               n + i + 3, 
               n + i + 6); 
    }           
  }


  /// flat-shaded triangle mesh built from a vertex array,
  /// if this routine isn't overridden, it has the same behavior
  /// as trimesh_n3f_v3f(), but if it is, the FileRenderer subclass
  /// can choose to skip storing surface normals in favor of on-the-fly
  /// facet normal calculation or something similar, for greater
  /// memory efficiency.
  virtual void trimesh_n3fopt_v3f(float *n, float *v, int numfacets) { 
    trimesh_n3f_v3f(n, v, numfacets); 
  }


  virtual void trimesh_n3b_v3f(char *n, float *v, int numfacets) { 
    int i;
    const float cn2f = 1.0f / 127.5f;
    const float ci2f = 1.0f / 255.0f; 

    for (i=0; i<numfacets*9; i+=9) {
      float norm[9];

      // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = (2c+1)/(2^8-1)
      norm[0] = n[i    ] * cn2f + ci2f;
      norm[1] = n[i + 1] * cn2f + ci2f;
      norm[2] = n[i + 2] * cn2f + ci2f;
      norm[3] = n[i + 3] * cn2f + ci2f;
      norm[4] = n[i + 4] * cn2f + ci2f;
      norm[5] = n[i + 5] * cn2f + ci2f;
      norm[6] = n[i + 6] * cn2f + ci2f;
      norm[7] = n[i + 7] * cn2f + ci2f;
      norm[8] = n[i + 8] * cn2f + ci2f;

      triangle(v + i    ,
               v + i + 3, 
               v + i + 6, 
               &norm[0],
               &norm[3], 
               &norm[6]);
    }           
  }

  /// triangle mesh built from a vertex array
  virtual void trimesh_c3f_n3f_v3f(float *c, float *n, float *v, int numfacets) { 
    int i;
    for (i=0; i<numfacets*9; i+=9) {
      tricolor(v + i    ,
               v + i + 3, 
               v + i + 6, 
               n + i    , 
               n + i + 3, 
               n + i + 6, 
               c + i    , 
               c + i + 3, 
               c + i + 6);
    }           
  }

  /// triangle mesh built from a vertex array and facet vertex index arrays
  virtual void trimesh_c4n3v3(int /* numverts */, float * cnv, 
                              int numfacets, int * facets) { 
    int i;
    for (i=0; i<numfacets*3; i+=3) {
      int v0 = facets[i    ] * 10;
      int v1 = facets[i + 1] * 10;
      int v2 = facets[i + 2] * 10;
      tricolor(cnv + v0 + 7, // vertices 0, 1, 2
               cnv + v1 + 7, 
               cnv + v2 + 7,
               cnv + v0 + 4, // normals 0, 1, 2
               cnv + v1 + 4, 
               cnv + v2 + 4,
               cnv + v0,     // colors 0, 1, 2
               cnv + v1, 
               cnv + v2);
    }           
  }

  /// triangle mesh built from a vertex array
  virtual void trimesh_c4u_n3f_v3f(unsigned char *c, float *n, float *v, 
                                   int numfacets) { 
    int i, j;
    const float ci2f = 1.0f / 255.0f;
    for (i=0,j=0; i<numfacets*9; i+=9,j+=12) {
      float col[9];

      // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = c/(2^8-1)
      col[0] = c[j     ] * ci2f;
      col[1] = c[j +  1] * ci2f;
      col[2] = c[j +  2] * ci2f;
      col[3] = c[j +  4] * ci2f;
      col[4] = c[j +  5] * ci2f;
      col[5] = c[j +  6] * ci2f;
      col[6] = c[j +  8] * ci2f;
      col[7] = c[j +  9] * ci2f;
      col[8] = c[j + 10] * ci2f;

      tricolor(v + i    ,
               v + i + 3, 
               v + i + 6, 
               n + i    , 
               n + i + 3, 
               n + i + 6, 
               &col[0],
               &col[3], 
               &col[6]);
    }           
  }

  /// triangle mesh built from a vertex array
  virtual void trimesh_c4u_n3b_v3f(unsigned char *c, char *n, float *v, 
                                   int numfacets) { 
    int i, j;
    const float ci2f = 1.0f / 255.0f; // used for uchar2float and normal conv
    const float cn2f = 1.0f / 127.5f;
    for (i=0,j=0; i<numfacets*9; i+=9,j+=12) {
      float col[9], norm[9];

      // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = c/(2^8-1)
      col[0] = c[j     ] * ci2f;
      col[1] = c[j +  1] * ci2f;
      col[2] = c[j +  2] * ci2f;
      col[3] = c[j +  4] * ci2f;
      col[4] = c[j +  5] * ci2f;
      col[5] = c[j +  6] * ci2f;
      col[6] = c[j +  8] * ci2f;
      col[7] = c[j +  9] * ci2f;
      col[8] = c[j + 10] * ci2f;

      // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = (2c+1)/(2^8-1)
      norm[0] = n[i    ] * cn2f + ci2f;
      norm[1] = n[i + 1] * cn2f + ci2f;
      norm[2] = n[i + 2] * cn2f + ci2f;
      norm[3] = n[i + 3] * cn2f + ci2f;
      norm[4] = n[i + 4] * cn2f + ci2f;
      norm[5] = n[i + 5] * cn2f + ci2f;
      norm[6] = n[i + 6] * cn2f + ci2f;
      norm[7] = n[i + 7] * cn2f + ci2f;
      norm[8] = n[i + 8] * cn2f + ci2f;

      tricolor(v + i    ,
               v + i + 3, 
               v + i + 6, 
               &norm[0],
               &norm[3], 
               &norm[6],
               &col[0],
               &col[3], 
               &col[6]);
    }           
  }


  /// triangle mesh built from a vertex array and facet vertex index arrays
  virtual void trimesh_singlecolor(int cindex, int /* numverts */, float * nv, 
                                   int numfacets, int * facets) { 
    super_set_color(cindex); // set current color

    int i;
    for (i=0; i<numfacets*3; i+=3) {
      int v0 = facets[i    ] * 6;
      int v1 = facets[i + 1] * 6; 
      int v2 = facets[i + 2] * 6;
      triangle(nv + v0 + 3, // vertices 0, 1, 2
               nv + v1 + 3, 
               nv + v2 + 3,
               nv + v0,     // normals 0, 1, 2
               nv + v1, 
               nv + v2);
    }           
  }


  /// triangle strips built from a vertex array and vertex index arrays
  virtual void tristrip(int /* numverts */, const float * cnv, 
                        int numstrips, const int *vertsperstrip, 
                        const int *facets) { 
    // render triangle strips one triangle at a time
    // triangle winding order is:
    //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
    int strip, t, v = 0;
    int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };
 
    // loop over all of the triangle strips
    for (strip=0; strip < numstrips; strip++) {       
      // loop over all triangles in this triangle strip
      for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
        // render one triangle, using lookup table to fix winding order
        int v0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
        int v1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
        int v2 = facets[v + (stripaddr[t & 0x01][2])] * 10;
 
        tricolor(cnv + v0 + 7, // vertices 0, 1, 2
                 cnv + v1 + 7, 
                 cnv + v2 + 7,
                 cnv + v0 + 4, // normals 0, 1, 2
                 cnv + v1 + 4, 
                 cnv + v2 + 4,
                 cnv + v0,     // colors 0, 1, 2
                 cnv + v1, 
                 cnv + v2);
        v++; // move on to next vertex
      }
      v+=2; // last two vertices are already used by last triangle
    }
  }


  /// single-color triangle strips built from a vertex array and 
  /// vertex index arrays
  virtual void tristrip_singlecolor(int /* numverts */, const float * nv, 
                                    int numstrips, const int *stripcolindex,
                                    const int *vertsperstrip, const int *facets) { 
    // render triangle strips one triangle at a time
    // triangle winding order is:
    //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
    int strip, t, v = 0;
    int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };
 
    // loop over all of the triangle strips
    for (strip=0; strip < numstrips; strip++) {       
      super_set_color(stripcolindex[strip]); // set current color
      
      // loop over all triangles in this triangle strip
      for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
        // render one triangle, using lookup table to fix winding order
        int v0 = facets[v + (stripaddr[t & 0x01][0])] * 6;
        int v1 = facets[v + (stripaddr[t & 0x01][1])] * 6;
        int v2 = facets[v + (stripaddr[t & 0x01][2])] * 6;
 
        triangle(nv + v0 + 3, // vertices 0, 1, 2
                 nv + v1 + 3, 
                 nv + v2 + 3,
                 nv + v0, // normals 0, 1, 2
                 nv + v1, 
                 nv + v2);
        v++; // move on to next vertex
      }
      v+=2; // last two vertices are already used by last triangle
    }
  }


  /// single-color triangle fans built from a vertex array and
  /// vertex index arrays
  virtual void trifan_singlecolor(int /* numverts */, const float * nv,
                                  int numfans, const int *fancolindex,
                                  const int *vertsperfan, const int *facets) {
    // render triangle fans one triangle at a time
    // triangle winding order is:
    //   v0, v1, v2, then v0, v2, v3, then v0, v3, v4, etc.
    int fan, t, v = 0;

    // loop over all of the triangle fans
    for (fan=0; fan < numfans; fan++) {
      super_set_color(fancolindex[fan]); // set current color

      // loop over all triangles in this triangle fan
      int v0 = facets[v] * 6;
      v++;
      for (t = 1; t < (vertsperfan[fan] - 1); t++) {
        // render one triangle with correct winding order
        int v1 = facets[v    ] * 6;
        int v2 = facets[v + 1] * 6;

        triangle(nv + v0 + 3, // vertices 0, 1, 2
                 nv + v1 + 3,
                 nv + v2 + 3,
                 nv + v0, // normals 0, 1, 2
                 nv + v1,
                 nv + v2);
        v++; // move on to next vertex
      }
      v++; // last vertex is already used by last triangle
    }
  }


  /// define a volumetric texture map
  virtual void define_volume_texture(int ID, int xs, int ys, int zs,
                                     const float *xplaneeq, 
                                     const float *yplaneeq,
                                     const float *zplaneeq,
                                     unsigned char *texmap) {
    warningflags |= FILERENDERER_NOTEXTURE;
  }


  /// enable volumetric texturing, either in "replace" or "modulate" mode
  virtual void volume_texture_on(int texmode) {
    warningflags |= FILERENDERER_NOTEXTURE;
  }


  /// disable volumetric texturing
  virtual void volume_texture_off(void) {
    warningflags |= FILERENDERER_NOTEXTURE;
  }


  /// wire mesh built from a vertex array and an vertex index array
  virtual void wiremesh(int /* numverts */, float * cnv, 
                       int numlines, int * lines) { 
    int i;
    int index = 1;

    for (i=0; i<numlines; i++) {
      float r, g, b;
      int ind = i * 2;
      int v0 = lines[ind    ] * 10;
      int v1 = lines[ind + 1] * 10;

      r = cnv[v0 + 0] + cnv[v1 + 0] / 2.0f;
      g = cnv[v0 + 1] + cnv[v1 + 1] / 2.0f;
      b = cnv[v0 + 2] + cnv[v1 + 2] / 2.0f;

      index = nearest_index(r,g,b); // lookup nearest color here.
      super_set_color(index); // use the closest color

      line(cnv + v0 + 7, cnv + v1 + 7); 
    }           
  }

  /// start a new representation geometry group, used to preserve some of
  /// the original scene hierarchy when loading VMD scenes into tools like
  /// Maya, 3DS Max, etc.
  virtual void beginrepgeomgroup(const char *) {}

  /// Comment describing representation geometry
  virtual void comment(const char *) {}

  /// draw text at specified location
  virtual void text(float *pos, float size, float thickness, const char *str);

  /// here for completeness, only VRML or 'token' renderers would likely use it
  virtual void pick_point(float * /*xyz*/, int /*id*/) {}

};

#endif

