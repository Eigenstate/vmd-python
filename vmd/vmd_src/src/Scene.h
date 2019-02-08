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
 *	$RCSfile: Scene.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.64 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Scene object, which maintains a Displayable object and
 * draws them to a DisplayDevice.
 *
 ***************************************************************************/
#ifndef SCENE_H
#define SCENE_H

#include "Displayable.h"
#include "NameList.h"

class DisplayDevice;
class FileRenderer;
class DisplayColor;

// constants for this object
#define DISP_LIGHTS 4

// total number of colors defined here
#define REGCLRS         33
#define EXTRACLRS       1
#define VISCLRS         (REGCLRS - EXTRACLRS)
#define MAPCLRS         1024    
#define MAXCOLORS       (REGCLRS + MAPCLRS)

// where different type of colors start in indices
#define BEGREGCLRS      0
#define BEGMAP          REGCLRS

// regular (visible) colors
#define REGBLUE         0
#define REGRED          1
#define REGGREY         2
#define REGORANGE       3
#define REGYELLOW       4
#define REGTAN          5
#define REGSILVER       6
#define REGGREEN        7
#define REGWHITE        8
#define REGPINK         9
#define REGCYAN         10
#define REGPURPLE       11
#define REGLIME         12
#define REGMAUVRE       13
#define REGOCHRE        14
#define REGICEBLUE      15
#define REGBLACK        16

#define REGBLUE2        23

// macro to get colormap colors
#define MAPCOLOR(a)             (a + BEGMAP)


/// color gradient/ramp used by value-based coloring methods
class ColorScale {
public:
  float min[3], mid[3], max[3];
  char name[32];

  int operator==(const ColorScale c) {
    return !memcmp(&c, this, sizeof(ColorScale));
  }
};


/// Contains lists of Displayable objects and draws them to a DisplayDevice
class Scene {
private:
  /// Background drawing mode 
  int backgroundmode;           ///< background drawing mode
  int backgroundmode_changed;   ///< mode changed since last redraw
 
  /// Light state data
  struct LightState {
    float color[3];             ///< RGB color of the light
    float pos[3];               ///< Position (or direction) of the light 
    int highlighted;            ///< Whether a "highlight" line is drawn
    int on;                     ///< on/off state of light
  };
  int light_changed;            ///< lights changed since last redraw
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
    int highlighted;            ///< Whether a "highlight" line is drawn
    int on;                     ///< on/off state of light
  };
  int adv_light_changed;        ///< advanced lights changed since last redraw
  AdvancedLightState advLightState[DISP_LIGHTS]; ///< state of advanced lights

  /// color state data
  static const float defaultColor[3L*REGCLRS];
  float colorData[3L*MAXCOLORS];
  NameList<NameList<int> *> categories;
  
  NameList<int> colorNames;
  
  int scaleMethod;
  float scaleMin, scaleMid, scaleMax;
  ResizeArray<ColorScale> colorScales;
  void create_colorscale();

  // displayables to handle the foreground and background colors of the display
  DisplayColor *background;
  DisplayColor *backgradtop;
  DisplayColor *backgradbot;
  DisplayColor *foreground;

  /// background color information stored as member data in Scene so that
  /// it's accessible from other rendering processes.
  int background_color_changed;
  int background_color_id;

  /// background gradient color information stored as member data in Scene 
  /// so that it's accessible from other rendering processes.
  int backgradtop_color_changed;
  int backgradtop_color_id;
  int backgradbot_color_changed;
  int backgradbot_color_id;

  /// foreground color information stored as member data in Scene so that
  /// it's accessible from other rendering processes.
  int foreground_color_changed;
  int foreground_color_id;

public:
  Scene(void);            ///< constructor
  virtual ~Scene(void);   ///< destructor

  /// The top level Displayable, parent of all other displayables in the
  /// scene.  public for now since Scene replicates essentially the entire
  /// interface of Displayable.
  Displayable root;

  /// Change background drawing mode
  void set_background_mode(int mode);
  int  background_mode(void);

  void reset_lights(); ///< 

  //@{
  /// routines to deal with standard directional light sources
  void define_light(int n, const float *color, const float *position);
  void activate_light(int n, int turnon);
  void highlight_light(int /* n */, int /* turnon */) {}
  void rotate_light(int n, float theta, char axis);
  void move_light(int n, const float *);
  const float *light_pos(int n) const; // return light position, or NULL
  const float *light_pos_default(int n) const; // return def. light position
  const float *light_color(int n) const;
  const float *light_color_default(int n) const;
  int light_active(int n) const { return lightState[n].on; }
  int light_highlighted(int) const { return FALSE; }
  //@}

  //@{
  /// routines to deal with advanced positional light sources
  void define_adv_light(int n, const float *color, const float *position,
                        float constant, float linear, float quad, 
                        float *spotdir, float fallstart, float fallend,
                        int spoton);
  void activate_adv_light(int n, int turnon);
  void highlight_adv_light(int /* n */, int /* turnon */) {}
  void move_adv_light(int n, const float *);
  const float *adv_light_pos(int n) const; // return light position, or NULL
  const float *adv_light_pos_default(int n) const; // return def. light position
  const float *adv_light_color(int n) const;
  const float *adv_light_color_default(int n) const;
  void adv_light_attenuation(int n, float constant, float linear, float quad);
  void adv_light_get_attenuation(int n, float &constant, float &linear, float &quad) const;
  void adv_light_spotlight(int n, float *spotdir, float fallstart, 
                           float fallend, int spoton);
  const float *adv_light_get_spotlight(int n, float &fallstart,
                                       float &fallend, int &spoton) const;
  int adv_light_active(int n) const { return advLightState[n].on; }
  int adv_light_highlighted(int) const { return FALSE; }
  //@}

  //@{
  /// routines to get/set color properties
  int add_color_category(const char *catname) {
    if (categories.typecode(catname) != -1) return -1;
    return categories.add_name(catname, new NameList<int>);
  }
  int add_color_item(int cat_id, const char *name, int init_color) {
    NameList<int> *cat = categories.data(cat_id);
    return cat->add_name(name, init_color);
  }
  /// change color properties
  /// These return void because there is _no_ error checking
  void set_category_item(int cat_id, int item, int color) {
    NameList<int> *cat = categories.data(cat_id);
    cat->set_data(item, color);
    root.color_changed(cat_id);
  }
  void set_color_value(int n, const float *rgb) {
    memcpy(colorData+3L*n, rgb, 3L*sizeof(float));
    root.color_rgb_changed(n);
  }
  void set_colorscale_value(float min, float mid, float max) {
    scaleMin = min; scaleMid = mid; scaleMax = max;
    create_colorscale();
  }
  void set_colorscale_method(int method) {
    if (scaleMethod != method) {
      scaleMethod = method;
      create_colorscale();
    }
  }
  
  //Returns the color index for a color category
  int get_category_item(int cat_id, int item) {
    NameList<int> *cat = categories.data(cat_id);
    return cat->data(item);
  }

  /// Store the color scale colors in the given arrays
  int get_colorscale_colors(int whichScale, 
      float min[3], float mid[3], float max[3]);
  /// Set the color scale colors from the given arrays
  int set_colorscale_colors(int whichScale, 
      const float min[3], const float mid[3], const float max[3]);
  

  /// query color information; all const methods that assume valid inputs
  int num_categories() const { return categories.num(); }
  const char *category_name(int cat) const { return categories.name(cat); }
  int category_index(const char *catname) const { 
    return categories.typecode(catname);
  }
  int num_colors() const { return MAXCOLORS; }
  int num_regular_colors() const { return REGCLRS; }
  const char *color_name(int n) const { return colorNames.name(n); }

  /// return index of color; returns -1 if name is not a valid color name
  int color_index(const char *name) const { return colorNames.typecode(name); }
  const float *color_value(int n) const { return colorData+3L*n; }
  const float *color_default_value(int n) const { return defaultColor+3L*n; }
  int num_category_items(int cat) const { 
    return categories.data(cat)->num(); 
  }
  const char *category_item_name(int cat, int item) const {
    return categories.data(cat)->name(item);
  }
  int category_item_index(int cat, const char *item) const {
    return categories.data(cat)->typecode(item);
  }
  int category_item_value(int cat, const char *item) const {
    return categories.data(cat)->data(item);
  }
  int category_item_value(int cat, int item) const {
    return categories.data(cat)->data(item);
  }

  /// color scale methods
  void colorscale_value(float *mid, float *min, float *max) const {
    *mid = scaleMid; *min = scaleMin; *max = scaleMax; 
  }
  int num_colorscale_methods() const { return colorScales.num(); }
  int colorscale_method() const { return scaleMethod; }
  const char *colorscale_method_name(int n) const {
    return colorScales[n].name;
  }

  /// nearest_index: deprecated but needed by ImportGraphicsPlugin and
  /// MoleculeRaster3D for now.
  int nearest_index(float r, float g, float b) const;

  //@}
 
  /// prepare all registered Displayables
  /// return whether we need an update or not
  virtual int prepare();

  /// draw the scene to the given DisplayDevice, can change display states
  /// XXX note, this method should really be a 'const' method since 
  ///   it is run concurrently by several processes that share memory, but
  ///   we can't actually write it that way since the locking routines do 
  ///   indeed write to lock variables.  The code in draw() should be written
  ///   as though it were a const method however.
  virtual void draw(DisplayDevice *);
  
  /// draw the scene to a file in a given format, trying to match the
  /// view of the given DisplayDevice as closely as possible
  /// returns TRUE if successful, FALSE if not
  /// There are no stereo output formats; if there are, then things will
  /// become somewhat more difficult
  int filedraw(FileRenderer *, const char *, DisplayDevice *);

  /// perform any post-drawing cleanup, reset state caching variables, etc.
  void draw_finished();

};

#endif

