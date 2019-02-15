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
 *      $RCSfile: VMDApp.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.251 $      $Date: 2019/02/07 21:44:12 $
 *
 ***************************************************************************/

// VMDApp
// Instantiate a new VMD object!

#ifndef VMD_APP_H
#define VMD_APP_H

#include "PluginMgr.h"
class AtomSel;
class Plugin;
class CUDAAccel;
class NVENCMgr;
class DisplayDevice;
class SymbolTable;
class Scene;
class PickList;
class PickModeList;
class Axes;
class DisplayRocker;
class CommandQueue;
class UIText;
class MoleculeList;
class GeometryList;
class IMDMgr; 
class Stage;
class Mouse; 
class FPS;
class Matrix4;
class Animation;
class Mobile; 
class Spaceball; 
#ifdef WIN32
class Win32Joystick;
#endif
class UIVR; 
class FileRenderList;
class VMDTitle;
class MaterialList;
class VideoStream;
class VMDMenu;
class VMDCollab;
class QuickSurf;

#include <string.h>  // for size_t
#include "NameList.h"
#include "JString.h"
#include "vmdplugin.h"
#include "ProfileHooks.h"

#define VMD_CHECK_EVENTS  1
#define VMD_IGNORE_EVENTS 0

/// File loading parameter specification, what frames, volume sets, etc to load
struct FileSpec {
  enum WaitFor { WAIT_ALL=-1, ///< Wait for all frames to load in foreground
                 WAIT_BACK=1  ///< Proceed while frames load in background 
               };

  int autobonds;    ///< whether to allow automatic bond determination
  int filebonds;    ///< whether to honor bonds specified by the loaded file(s)
  int first;        ///< first timestep to read/write
  int last;         ///< last timestep to read/write
  int stride;       ///< stride to take in reading/writing timesteps
  int waitfor;      ///< whether to wait for all timesteps before continuing
  int nvolsets;     ///< number of volume sets in list
  int *setids;      ///< list of volumesets to load/save
  int *selection;   ///< if non-NULL, flags for selected atoms to read/write

  FileSpec() {
    autobonds = 1;       // allow automatic bond determination when necessary
    filebonds = 1;       // honor bonds read from loaded files
    first = 0;           // start with first timestep
    last = -1;           // end with last timestep
    stride = 1;          // do every frame
    waitfor = WAIT_BACK; // wait for frames to load
    nvolsets = -1;       // all volumetric sets
    setids = NULL;       // but no explicit list
    selection = NULL;    // default to all atom selected
  }

  FileSpec(const FileSpec &s) {
    autobonds=s.autobonds;
    filebonds=s.filebonds;
    first=s.first;
    last=s.last;
    stride=s.stride;
    waitfor=s.waitfor;
    nvolsets=s.nvolsets;
    if (nvolsets > 0) {
      setids = new int[nvolsets];
      memcpy(setids, s.setids, nvolsets*sizeof(int));
    } else {
      setids = NULL;
    }
    // selections aren't implemented in file loading, and this copy
    // constructor is used only by CmdMolLoad.  Therefore we're ok to
    // just ignore the selection field when making a copy of FileSpec.
    selection = NULL;
  }

  ~FileSpec() {
    if (setids)
      delete [] setids;
  }
};


/// The main VMD application instance, created by the main entry point
class VMDApp {
public:
  /// set list of command line arguments
  VMDApp(int argc, char **argv, int mpion);

  /// initialize the global variables and objects for the general library.
  /// Must be passed command-line arguments to program.  Creates graphics
  /// context and all associated objects, and then starts the UI, then adds any
  /// commands that should be done at start.  When done, flushes command queue
  /// and then returns, when the program is ready to start main event loop.
  /// Return TRUE on successful initialization, FALSE if anything failed.
  int VMDinit(int, char **, const char *, int * dispLoc, int * dispSize);
  
  ~VMDApp();

  /// Print the given error message and pause for the given number of seconds
  /// before setting a flag that willl make VMDupdate return FALSE.
  void VMDexit(const char *exitmsg, int exitcode, int pauseseconds);
 
private:
  /// Flag indicating whether or not to initialize MPI when MPI support
  /// has been compiled into the VMD binary.  This allows the same binary
  /// to be run both interactively and in batch mode parallel MPI jobs, 
  /// e.g., on the Blue Waters Cray XE6/XK7 login nodes or on the compute nodes.
  int mpienabled;

#if defined(VMDXPLOR)
  /// self-pointer used by VMD-XPLOR
  static VMDApp* obj;
#endif

  /// text message to store output in console-free applications
  static JString text_message;

  /// list of GUI menus (including both FLTK, and Tk extensions)
  NameList<VMDMenu *> *menulist; 

  /// molid counter
  int nextMolID;

  /// have we shown the Stride message yet?
  int stride_firsttime;

  /// flag; whether to exit on eof from stdin, defaults to no.
  int eofexit;  

  /// flag; whether or not background processing is going on currently
  int backgroundprocessing;

  Mouse *mouse;
  DisplayRocker *rocker;

  /// Mobile interface instance
  Mobile *mobile;

  /// spaceball instance
  Spaceball *spaceball;

#ifdef WIN32
  /// joystick instance
  Win32Joystick *win32joystick;
#endif

  VMDTitle *vmdTitle;
  FileRenderList *fileRenderList;
  PluginMgr *pluginMgr;

public:
  int argc_m;           ///< used if we want to process unknown args elsewhere
  const char **argv_m;  ///< needed by Tcl/Python initialization code

  // de facto Global variables, by virtual of being public in this singleton
  UIText *uiText;               ///< the text interface JRG: made public for save_state
  UIVR *uivr;                   ///< VR tool interface
  IMDMgr *imdMgr;               ///< IMD manager class 
  VideoStream *uivs;            ///< Video streaming UI events
  Animation *anim;              ///< generates delay-based frame change events
  DisplayDevice *display;       ///< display in which the images are rendered
  Scene *scene;                 ///< list of all Displayable objects to draw

  void *thrpool;                ///< CPU thread pool for low-latency calcs

  CUDAAccel *cuda;              ///< CUDA acceleration system handle

  NVENCMgr *nvenc;              ///< GPU hardware H.26[45] video [en|de]coder

  QuickSurf *qsurf;             ///< QuickSurf object shared by all reps, 
                                ///< to help minimize the persistent 
                                ///< GPU global memory footprint, and to make
                                ///< it easy to force-dump all persistent
                                ///< QuickSurf GPU global memory data 
                                ///< structures on-demand for GPU ray tracing,
                                ///< or other tasks that also need a lot of
                                ///< resources.  The default persistence of 
                                ///< QuickSurf GPU resources enable fast 
                                ///< trajectory playback.

  PickList *pickList;           ///< handles all picking events
  PickModeList *pickModeList;   ///< list of available picking modes
  MaterialList *materialList;   ///< list of materials
  Stage *stage;                 ///< stage object used in the scene
  Axes *axes;                   ///< axes object used in the scene
  FPS *fps;                     ///< FPS counter used in the scene
  CommandQueue *commandQueue;   ///< the command processor
  MoleculeList *moleculeList;   ///< list of all loaded molecules
  GeometryList *geometryList;   ///< list of all labels etc
  SymbolTable *atomSelParser;   ///< symbol table and atom selection parser
  VMDCollab *vmdcollab;         ///< handles collaborative VMD interaction
  NameList<char *> userKeys;    ///< lookup table for Tcl scripts
  NameList<char *> userKeyDesc; ///< describe what the hotkey does
  int UpdateDisplay;            ///< flag for whether to update the scene
  int exitFlag;                 ///< flag for whether to quit the display loop.
  int ResetViewPending;         ///< pending resetview needs attention
  char nodename[512];           ///< MPI node name
  int noderank;                 ///< MPI node rank
  int nodecount;                ///< MPI node count

  /// Highlighted molecule id and rep.  Set by GraphicsFltkMenu, used by
  /// PickModeMoveHighlightedRep
  int highlighted_molid, highlighted_rep;

  //
  // NO member variables beyond this point!
  //

  /// Background processing flag indicating whether the main event loop
  /// should throttle the CPU consumption back when there aren't any other
  /// display updates, event handling, or other activities to perform
  /// This flag is used by Molecule to inform VMD when background
  /// trajectory loading is going on, for example.
  int  background_processing() { return backgroundprocessing; }
  void background_processing_clear() { backgroundprocessing = 0; }
  void background_processing_set()   { backgroundprocessing = 1; }

  /// Turn off event checking for the text interface; if inactive it will
  /// still process text from VMD, like hotkey callbacks, but Tk menus 
  /// will not work.  This is experimental code just for the purpose of
  /// making it possible to control VMD from a thread in another program.
  void deactivate_uitext_stdin();

  //
  // methods for querying the state of the VMD menus
  //

  /// Activate Fltk menus; this should be called only once and only after
  /// VMDinit.  Return success.
  int activate_menus();

  int num_menus();               ///< Number of menus we know about.
  const char *menu_name(int);    ///< Name of nth menu; 0 <= n < num_menus()
  int menu_id(const char *name);
  int add_menu(VMDMenu *);    //< add menu.  Return success.
  int remove_menu(const char *); ///< remove menu.  Return success.

  /// Announce that the menu of the given name is a menu extension.  This
  /// lets widgets add the menu to their own pulldown menus if they wish.
  void menu_add_extension(const char *shortname, const char *menu_path);
  void menu_remove_extension(const char *shortname);
  
  /// Return 1 or 0 if the menu is on or off.  Return 0 if the menu does not
  /// exist.
  int menu_status(const char *name);

  /// Get the location of the specified menu.
  int menu_location(const char *name, int &x, int &y);
 
  /// Turn the specified menu on or off.
  int menu_show(const char *name, int on);

  /// Move the specified menu to the given location on the screen.
  int menu_move(const char *name, int x, int y);
  
  /// Tells the specified menu to select the "molno"-th molecule internally
  int menu_select_mol(const char *name, int molno);
  
  // 
  // methods for exporting to external rendering systems
  //
  
  int filerender_num();                     ///< Number of file render methods
  const char *filerender_name(int n);       ///< Name of Nth file renderer
  const char *filerender_prettyname(int n); ///< Pretty name of Nth renderer
  int filerender_valid(const char *method); ///< Return true if renderer exists

  /// Find short renderer name from the "pretty" GUI renderer name
  const char *filerender_shortname_from_prettyname(const char *pretty);

  /// Return whether given renderer supports antialiasing
  int filerender_has_antialiasing(const char *method); 

  /// Set the antialiasing sample count and return the new value
  int filerender_aasamples(const char *method, int aasamples);

  /// Set the ambient occlusion sample count and return the new value
  int filerender_aosamples(const char *method, int aosamples);

  /// Return whether the given renderer supports arbitrary image size
  int filerender_has_imagesize(const char *method);

  /// Set/get the image size.  If *width or *height are zero, then the 
  /// existing value will be used instead.  If aspect ratio is set, then
  /// the aspect ratio will be used to determine the other member of the pair.
  /// Return success.
  int filerender_imagesize(const char *method, int *imgwidth, int *imgheight);

  /// Set/get the aspect ratio for the image.  An aspect ratio of zero means
  /// the image is free to take on any size.  A positive value means the
  /// _height_ of the image will be scaled to maintain the given aspect ratio.
  /// Negative values fail.  Return success, and place the new value of the 
  /// aspect ratio in the passed-in pointer.
  int filerender_aspectratio(const char *method, float *aspect);

  /// Return the number of file formats the file renderer can produce.  Returns
  /// zero if the renderer method is invalid.
  int filerender_numformats(const char *method); 

  /// Return the ith format.  NULL if invalid.
  const char *filerender_get_format(const char *method, int i);

  /// Return name of currently selected format
  const char *filerender_cur_format(const char *method);

  /// Set the output format for the renderer.  Return success.
  int filerender_set_format(const char *method, const char *format);

  /// do the rendering; return success
  int filerender_render(const char *method, const char *filename,
                        const char *extcmd);

  /// set the command string to execute after producing the scene file
  /// Return the new value, or NULL if the method is invalid.   Specify option 
  /// as NULL to fetch the current value.
  const char *filerender_option(const char *method, const char *option);
  
  /// get the default render option for the given method.
  const char *filerender_default_option(const char *method);

  /// get the dafault filename for this render method
  const char *filerender_default_filename(const char *method);

  // 
  // methods for rotating, translating, and scaling the scene.  Return success
  //

  /// rotate the scene by or to the given angle, measured in degrees, about
  /// the given axis, either 'x', 'y', or 'z'.  For rotate_by, If incr is zero, 
  /// the rotation will be done in one redraw; otherwise the rotation will be 
  /// performed in steps of incr. 
  int scene_rotate_by(float angle, char axis, float incr = 0);
  int scene_rotate_to(float angle, char axis);

  /// Rotate the scene by the specified matrix.  The translation part will be
  /// ignored.
  int scene_rotate_by(const float *); // takes a float[16]
  int scene_rotate_to(const float *); // takes a float[16]

  /// Translate everything that isn't fixed by/to the given amount.
  int scene_translate_by(float x, float y, float z);
  int scene_translate_to(float x, float y, float z);

  /// Scale by/to the given positive scale factor.
  int scene_scale_by(float s);
  int scene_scale_to(float s);

  /// recenter the scene on the top molecule or on last settings if
  /// no-disrupt mode is enabled, unless there's only one molecule
  void scene_resetview_newmoldata();

  /// recenter the scene on the top molecule.  If there are no molecules, 
  /// just restores rotation to default value.
  void scene_resetview();
 
  /// Rock the scene by the given amount per redraw about the given axis.
  /// If nsteps is positive, rock for the specified number of steps, then
  /// reverse direction.
  int scene_rock(char axis, float step, int nsteps = 0); 
 
  /// Stop rocking the scene. 
  int scene_rockoff();

  /// Stop rocking AND persistent rotations induced by input devices (like
  /// the Mouse).
  int scene_stoprotation();

  ///
  /// Methods for affecting the animation.  Only _active_ molecules are
  /// affected.  I consider this a mistake: it would be better to be able to
  /// specify the animation settings for a particular molecule by specify a
  /// molecule ID.  However, this would make it impossible to log Tcl commands
  /// using our current syntax because the Tcl commands operate on all active
  /// molecules.  If we ever create new Tcl commands and/or deprecate the old
  /// ones, we can (and should) change thse API's.  Note that there are no
  /// get-methods here because you would need to query molecules individually.
  /// The Tcl commands just return the value for the top molecule; they can 
  /// continue to do so, but the API should be per molecule.
  ///
  /// number of animation direction choices, and their names
  int animation_num_dirs();
  const char *animation_dir_name(int);

  /// set the animation direction for all active molecules
  int animation_set_dir(int);

  /// number of animation styles, and their names
  int animation_num_styles();
  const char *animation_style_name(int);

  /// set the animation style for all active molecules
  int animation_set_style(int);

  /// set the animation frame for all active molecules.  If the specified 
  /// frame is out of range for a particular molecule, then its frame will
  /// not change.  If frame is -1, go to the first frame.  If frame is -2,
  /// go to the last frame.
  int animation_set_frame(int frame);

  /// set the stride for animation.  Must be >= 1.  
  int animation_set_stride(int);

  /// set the animation speed.  Must be a float between 0 and 1.  1 means
  /// animate as fast as possible; 0 means pause at least 0.5 seconds between
  /// frames.
  int animation_set_speed(float);


  //
  // Methods for getting plugins
  //

  /// get a plugin of the specified type and name.  If none was found, return
  /// NULL.  The returned plugin should not be deleted.  If multiple plugins
  /// are found, the one with the highest version number is returned.
  vmdplugin_t *get_plugin(const char *type, const char *name);
  
  /// Get alll plugins of the specfied type.  If no type is
  /// specified or is NULL, all loaded plugins will be returned.  
  /// Returns the number of plugins added to the list.
  int list_plugins(PluginList &, const char *type = NULL);

  /// Try to dlopen the specified shared library and access its plugin API.  
  /// Return the number of plugins found in the given library, or -1 on error. 
  int plugin_dlopen(const char *filename);

  /// Tell VMD to update its lists of plugins based on all the shared libraries
  /// it's loaded.  Methods listed below will not be updated after a call to
  /// plugin_dlopen until this method is called.  
  void plugin_update();

  //
  // display update methods
  //

  /// turn display updates on (1) or off (0)
  void display_update_on(int);

  /// return 1 or 0 if display updates are on or off, respectively.
  int display_update_status();

  /// force a screen redraw right now, without checking for UI events 
  void display_update();
  
  /// force a screen redraw right now, and also check for UI events
  void display_update_ui();

  //
  // methods for setting display properties
  // These should be considered provisional; they are here primarily to
  // allow startup options to be processed outside of VMDApp.
  //

  /// get/set the height of the screen.  Ignored unless positive.
  void display_set_screen_height(float);
  float display_get_screen_height();
   
  /// get/set the distance to the screen.  
  void display_set_screen_distance(float);
  float display_get_screen_distance();
 
  /// get/set the position of the graphics window.  
  void display_set_position(int x, int y);
  //void display_get_position(int *x, int *y); // XXX doesn't work...
 
  /// get/set the size of the graphics window.  
  void display_set_size(int w, int h);
  void display_get_size(int *w, int *h);

  /// change the stereo mode
  int display_set_stereo(const char *mode);

  /// change the stereo swapped eye mode
  int display_set_stereo_swap(int onoff);

  /// change the caching mode
  int display_set_cachemode(const char *mode);

  /// change the rendering mode
  int display_set_rendermode(const char *mode);

  /// change eye separation
  int display_set_eyesep(float sep);

  /// change focal length
  int display_set_focallen(float flen);

  /// set the projection (Perspective or Orthographic, case-insensitive)
  int display_set_projection(const char *proj);

  /// query whether the projection is a perspective projection type
  int display_projection_is_perspective(void);

  int display_set_aa(int onoff);
  int display_set_depthcue(int onoff);
  int display_set_culling(int onoff);
  int display_set_fps(int onoff);
  int display_set_background_mode(int mode);

  // set clipping plane position.  if isdelta, then treat as offset.
  int display_set_nearclip(float amt, int isdelta);
  int display_set_farclip(float amt, int isdelta);

  int stage_set_location(const char *);
  int stage_set_numpanels(int);
  int stage_set_size(float);

  int axes_set_location(const char *);

  int light_on(int lightnum, int onoff);
  int light_highlight(int lightnum, int onoff);
  int light_rotate(int lightnum, float amt, char axis);
  int light_move(int lightnum, const float *newpos);

  // XXX need to implement all of the advanced lights APIs here too...

  int depthcue_set_mode(const char *);
  int depthcue_set_start(float);
  int depthcue_set_end(float);
  int depthcue_set_density(float);

  int display_set_shadows(int onoff);

  int display_set_ao(int onoff);
  int display_set_ao_ambient(float a);
  int display_set_ao_direct(float d);
  
  int display_set_dof(int onoff);
  int display_set_dof_fnumber(float f);
  int display_set_dof_focal_dist(float d);

  // 
  /// turn on the title screen; burns CPU but will be turned off when a 
  /// molecule is loaded.
  void display_titlescreen();

  // 
  // Methods for setting color properties.  We have a set of color 
  // _categories_ (Display, Axes...), each of which contains one or more
  // _items_ (Display->Background, Axes->X, Y, Z, ...).  Each item has a
  // color _name_ mapped to it (blue, red, ...).
  //

  /// Number of color categories
  int num_color_categories();

  /// Name of the nth color category, or NULL if invalid index.
  const char *color_category(int);

  /// add a new color item, consisting of a name and a default color, to
  /// the given color category.  If the color category does not already exist,
  /// it is created.  Return success.
  int color_add_item(const char *cat, const char *item, const char *defcolor); 
  
  /// Number of color items in the given category
  int num_color_category_items(const char *category);

  /// Item for the given category and index
  const char *color_category_item(const char *category, int);

  /// Number of available colors
  int num_colors();

  /// Number of _regular_ colors, i.e., the ones that have actual names.
  int num_regular_colors();

  /// Name of nth color, where 0 <= n < num_colors().  If the index is invalid,
  /// return NULL.
  const char *color_name(int n);

  /// Index of given color.  If the color is invalid, return -1, other return
  /// a number in [0, num_colors()).  The color must be one of the colors 
  /// returned by color_name().  Hence, color_name(color_index(<string>))
  /// returns its input if <string> is a valid color, or NULL if it isn't.
  int color_index(const char *);

  /// Get RGB value of given color.  Return success.
  int color_value(const char *colorname, float *r, float *g, float *b);

  /// Get default RGB value of given color.  The colorname must be one of
  /// the regular colors, i.e. have an index in [0,num_regular_colors).
  /// Return success.
  int color_default_value(const char *colorname, float *r, float *g, float *b);
  
  /// Color mapped to given color category and item, or NULL if invalid.
  const char *color_mapping(const char *category, const char *item);
 
  /// get the restype for the given resname.  if the resname, is unknown,
  /// returns "Unassigned".
  const char *color_get_restype(const char *resname);

  /// set the residue type for the given residue name.  This will determine
  /// how the residue is colored when the coloring method is ResType.  The
  /// type must be one of the color items in the Restype color category.
  /// return success.
  int color_set_restype(const char *resname, const char *newtype);

  /// color scale info
  int colorscale_info(float *midpoint, float *min, float *max); 

  /// info about color scale methods
  int num_colorscale_methods();
  int colorscale_method_current();
  const char *colorscale_method_name(int);

  /// index for given method.  Return -1 if invalid, otherwise nonnegative.
  int colorscale_method_index(const char *);

  /// Store the color scale colors in the given arrays
  int get_colorscale_colors(int whichScale, 
      float min[3], float mid[3], float max[3]);
  /// Set the color scale colors from the given arrays
  int set_colorscale_colors(int whichScale, 
      const float min[3], const float mid[3], const float max[3]);

  /// Change the color for a particular color category and name.  
  /// Color must be one of names returned by color_name().
  int color_change_name(const char *category, const char *colorname,
                        const char *color);

  /// Change a list of colors for particular color categories and names.  
  /// Each color must be one of names returned by color_name().
  int color_change_namelist(int numcols, char **category, 
                            char **colorname, char **color);
  
  /// Returns the color string for a particular color category and name. 
  int color_get_from_name(const char *category, const char *colorname,
                          const char **color);
 
  /// Change the RGB value for the specified color.
  int color_change_rgb(const char *color, float r, float g, float b);

  /// Change the RGB values for an entire list of colors
  int color_change_rgblist(int numcols, const char **colors, float *rgb3fv);

  /// Change the settings for the color scale.
  int colorscale_setvalues(float midpoint, float min, float max);

  /// Change the color scale method.
  int colorscale_setmethod(int method);

  //
  // Command logging methods
  //
  
  /// Process the commands in the given file  
  int logfile_read(const char *path);
  
  /// save VMD state to a Tcl script.  A filename will be requested from 
  /// the user
  int save_state();
 
  /// change to a new text interpreter mode.  Currently "tcl" and "python"
  /// are supported.
  int textinterp_change(const char *interpname);

  // 
  // Methods for editing and querying molecules and molecular representations
  //
 
  /// Number of molecules currently loaded
  int num_molecules();

  /// Create a new "empty" molecule, basically a blank slate for import 
  /// low-level graphics or other data.  Return the molid of the new molecule.
  /// we also allow to set the number of atoms. this is particularly useful 
  /// for topology building scripts.
  int molecule_new(const char *name, int natoms, int docallbacks=1);

  /// Create a new molecule from a list of atom selections, copying 
  /// all existing per-atom fields from the original molecules to the new one.
  int molecule_from_selection_list(const char *name, int mergemode,
                                   int numsels, AtomSel **, int docallbacks=1);

  /// Guess a molecule file type from the given filename.  Return the filetype,
  /// or NULL if unsuccesful.
  const char *guess_filetype(const char *filename);

  /// Load data from the given file of type filetype.  If molid is -1, a new
  /// molecule will be created if the file is successfully read; otherwise
  /// molid must be a valid molecule id.  As much information will be loaded
  /// from the file as possible, and within the limits prescribed by FileSpec.
  /// Returns the molid of the molecule into which the data was read.
  /// If the file type is unknown, use guess_filetype to obtain a filetype;
  /// don't pass NULL to filetype.
  int molecule_load(int molid, const char *filename, const char *filetype, 
      const FileSpec *spec);

  /// Add volumetric data to a given molecule.  The data block will be deleted
  /// by VMD.  Return success.
  int molecule_add_volumetric(int molid, const char *dataname, 
    const float origin[3], const float xaxis[3], const float yaxis[3],
    const float zaxis[3], int xsize, int ysize, int zsize, float *datablock);

  int molecule_add_volumetric(int molid, const char *dataname, 
    const double origin[3], const double xaxis[3], const double yaxis[3],
    const double zaxis[3], int xsize, int ysize, int zsize, float *datablock);

  /// Write trajectory frames to a file.  Return number of frames written
  /// before returning, as in the addfile method.  Filetype should
  /// be one of the file types returned by savecoorfile_plugin_name().
  /// selection must be NULL, or point to an array of flags, one for each
  /// atom in the molecule, indicating which atoms' coordinates are to be
  /// written.
  int molecule_savetrajectory(int molid, const char *filename, 
                              const char *filetype, const FileSpec *spec);

  /// Delete the specified range of timesteps from the given molecule, keeping
  /// every "stride" molecule (unless stride = 0).
  int molecule_deleteframes(int molid, int first, int last, int stride);

  /// Return the array index of the molecule with the specified ID.
  /// Returns -1 if the ID does not exist, otherwise a nonnegative
  /// array index is returned.
  int molecule_index_from_id(int molid);

  /// ID of the ith molecule.  This ID is used to specify a molecule for all
  /// other methods.  Return -1 if the ith molecule is not present; otherwise
  /// the ID is a nonnegative integer unique to each molecule.
  int molecule_id(int);

  /// Return true or false if the given molid is valid
  int molecule_valid_id(int molid);

  /// number of atoms in molecule.  Return -1 on invalid molid, otherwise
  /// 0 or more.
  int molecule_numatoms(int molid);

  /// number of frames in molecule.  Return -1 on invalid molid, otherwise
  /// 0 or more.
  int molecule_numframes(int molid);

  /// Current frame in molecule.  Return -1 on invalid molid, otherwise
  /// [0, numframes())
  int molecule_frame(int molid);

  /// Duplicate the given frame.  The new fram will be appended at the end.
  /// Passing -1 for frame duplicates the current frame.  Return success.
  int molecule_dupframe(int molid, int frame);
  
  /// name of molecule
  const char *molecule_name(int molid);
  int molecule_rename(int molid, const char *newname);

  /// cancel any in-progress file I/O associated with a given molecule.
  int molecule_cancel_io(int molid);
  
  /// delete the molecule with the given id
  int molecule_delete(int molid);

  /// delete all molecules
  int molecule_delete_all(void);
  
  /// make the given molecule 'active' or 'inactive'; active molecules respond
  /// to animate requests while inactive molecules do not.
  int molecule_activate(int molid, int onoff);
  int molecule_is_active(int molid);
 
  /// make the given molecule fixed or unfixed.  Fixed molecules do not respond
  /// to scene transformation operations.
  int molecule_fix(int molid, int onoff);
  int molecule_is_fixed(int molid);
 
  /// Turn the given molecule on or off.  Turning a molecule off causes all its
  /// reps to not be rendered.
  int molecule_display(int molid, int onoff);
  int molecule_is_displayed(int molid);

  /// Make the given molecule top.  There is always exactly one top molecule,
  /// if any are loaded.
  int molecule_make_top(int molid);

  /// return the molid of the top molecule
  int molecule_top(); 

  /// number of representations for the given molecule
  int num_molreps(int molid);

  //
  // For the molrep methods, repid is not unique; it applies to different reps
  // if reps are created and then deleted.  Too bad.
  //

  /// Get/set the current representation style
  const char *molrep_get_style(int molid, int repid); 
  int molrep_set_style(int molid, int repid, const char *style);
 
  /// Get/set the current representation color
  const char *molrep_get_color(int molid, int repid); 
  int molrep_set_color(int molid, int repid, const char *color);

  /// Get/set the current representation selection 
  const char *molrep_get_selection(int molid, int repid); 
  int molrep_set_selection(int molid, int repid, const char *selection);

  /// Get the number of atoms in the rep's selection.  If invalid molid or 
  /// repid, return -1, otherwise 0 or more.
  int molrep_numselected(int molid, int repid);

  /// Get/set the current representation material 
  const char *molrep_get_material(int molid, int repid); 
  int molrep_set_material(int molid, int repid, const char *material);

  /// Number of clipping planes supported per rep.  clipid in the next few
  /// methods should be in the range [0,max)
  int num_clipplanes();

  /// Get clipping plane info for reps.  center and normal should point to
  /// space for three floats. 
  int molrep_get_clipplane(int molid, int repid, int clipid, float *center, 
                           float *normal, float *color, int *mode);
  /// set clip plane properties.
  int molrep_set_clipcenter(int molid, int repid, int clipid, 
                            const float *center);
  int molrep_set_clipnormal(int molid, int repid, int clipid, 
                            const float *normal);
  int molrep_set_clipcolor(int molid, int repid, int clipid, 
                            const float *color);
  int molrep_set_clipstatus(int molid, int repid, int clipid, int onoff);
  
  /// Set smoothing for reps.  Coordinates used for calculating graphics
  /// will be smoothed with a boxcar average 2*n+1 in size centered on the
  /// current frame.   
  int molrep_set_smoothing(int molid, int repid, int n);

  /// Get smoothing for given rep.  Returns -1 for invalid rep, otherwise
  /// 0 or higher.
  int molrep_get_smoothing(int molid, int repid);
  
  // methods for retrieving reps by name rather than index.  The name is
  // guaranteed to be unique within a given molecule and follow the rep
  // around even when its order or index changes.
  
  /// Get the name of the given rep.  Return NULL if the id is invalid.
  const char *molrep_get_name(int molid, int repid);
  
  /// Get the repid of the rep with the given name.  Return -1 if the name
  /// was not found.
  int molrep_get_by_name(int molid, const char *);
 
  /// Set periodic boundary condition display for this rep
  int molrep_set_pbc(int molid, int repid, int pbc);

  /// Get current pbc for this rep; returns -1 if invalid.
  int molrep_get_pbc(int molid, int repid);

  /// Set the number of images to display; must be 1 or higher.  Return success
  int molrep_set_pbc_images(int molid, int repid, int n);

  /// Get number of images; returns -1 on error.
  int molrep_get_pbc_images(int molid, int repid);


  /// Add an instance transform to a given molecule.
  int molecule_add_instance(int molid, Matrix4 &inst);

  /// Report number of instances in a molecule, returns -1 if invalid.
  int molecule_num_instances(int molid);

  /// Delete all instances in a molecule
  int molecule_delete_all_instances(int molid);

  /// Set molecule instance display for this rep
  int molrep_set_instances(int molid, int repid, int inst);

  /// Get current instance display for this rep; returns -1 if invalid.
  int molrep_get_instances(int molid, int repid);


  /// Show/hide individual rep; this is done in the graphics menu by double-
  /// clicking on the rep.
  int molrep_show(int molid, int repid, int onff);

  /// Return 1 if shown, 0 if hidden or does not exist.
  int molrep_is_shown(int molid, int repid);

  // The next few commands set/get a default representation parameter; these
  // parameters define the properties of the rep which would be added on the
  // next call to molecule_addrep().  They exactly parallel the methods for
  // changing an existing representation except that no molid or repid is
  // specified.
  
  const char *molecule_get_style(); 
  int molecule_set_style(const char *style);
 
  const char *molecule_get_color(); 
  int molecule_set_color(const char *color);

  const char *molecule_get_selection(); 
  int molecule_set_selection(const char *selection);

  const char *molecule_get_material(); 
  int molecule_set_material(const char *material);

  /// Add a rep to the given molecule, using parameters specified in the
  /// molecule_set methods.  molid must be a valid molecule id.
  int molecule_addrep(int molid);
  
  /// Change the specified rep, using the same settings as for addrep.
  int molecule_modrep(int molid, int repid);

  /// Delete the specified rep
  int molrep_delete(int molid, int repid);
 
  //@{
  /// Turn on/off selection auto-update for the specified rep.  When on, the 
  /// representation will recalculate its selection each time there is
  /// change in the coordinate frame of the molecule. 
  int molrep_get_selupdate(int molid, int repid);
  int molrep_set_selupdate(int molid, int repid, int onoff);
  //@}

  //@{
  /// Turn on/off automatic color update for the specified rep. 
  int molrep_get_colorupdate(int molid, int repid);
  int molrep_set_colorupdate(int molid, int repid, int onoff);
  //@}

  //@{
  /// Get/set data range of color scale.
  int molrep_get_scaleminmax(int molid, int repid, float *min, float *max);
  int molrep_set_scaleminmax(int molid, int repid, float min, float max);
  int molrep_reset_scaleminmax(int molid, int repid);
  //@}
  
  /// Set drawing of selected frames for a given rep.  Syntax is
  /// "now" or a whitespace-separated list of terms of the form
  /// n, beg:end, or beg:stride:end.
  int molrep_set_drawframes(int molid, int repid, const char *framesel);
  const char *molrep_get_drawframes(int molid, int repid);

  /// Set/unset dataset flags, indicating to VMD which fields should be
  /// written out when the molecule is saved.
  int molecule_set_dataset_flag(int molid, const char *dataflagstr, int setval);

  /// Re-analyze the molecule after atom names, bonds, and other other
  /// data have been changed.  This can be used to fix unrecognized atom
  /// names in non-standard nucleic acid residues, and fix other issues
  /// on-the-fly without having to hand-edit the files.
  int molecule_reanalyze(int molid);

  /// Force recalculation of bonds for the given molecule based on the
  /// current set of coordinates.  
  int molecule_bondsrecalc(int molid);

  /// Force the recalculation of the secondary structure for the given 
  /// molecule based on the current set of coordinates.  Return true if the
  /// secondary structure was successfully recalculated, otherwise false.
  int molecule_ssrecalc(int molid);
   
  /// Create a new wavefunction object based on existing wavefunction
  /// <waveid> with orbitals localized using the Pipek-Mezey algorithm.
  int molecule_orblocalize(int molid, int waveid);

  // 
  // IMD methods
  //
  
  /// Establish an IMD connection to the given host over the given port, using
  /// the given molecule id.  Return success.
  int imd_connect(int molid, const char *host, int port);

  /// Return true if an IMD simulation is established with the given molid.
  int imd_connected(int molid);

  /// Send forces, assuming an IMD connection is present.  Return success.
  /// Format: num, indices, forces (xyzxyzxyz).  
  int imd_sendforces(int, const int *, const float *);

  /// Disconnect IMD.  Return success.
  int imd_disconnect(int molid);



  //
  // VideoStream methods
  //

  /// Establish an connection to the given host over the given port, using
  /// the given molecule id.  Return success.
  int vs_connect(const char *host, int port);

  /// Return true if a video strem is established.
  int vs_connected();

  /// Disconnect.  Return success.
  int vs_disconnect();

    

  // 
  // MPI related parallel processing routines
  //

  /// query current node name
  const char * par_name() { return nodename; }

  /// query current node ID
  int par_rank() { return noderank; }

  /// query total number of nodes
  int par_size() { return nodecount; }

  /// perform a barrier synchronization across all nodes
  void par_barrier();



  //
  // Tool methods
  //

  int tool_create(const char *type, int argc, const char **argv);
  int tool_delete(int toolnum);
  int tool_change_type(int toolnum, const char *type);
  int tool_set_position_scale(int toolnum, float newval);
  int tool_set_force_scale(int toolnum, float newval);
  int tool_set_spring_scale(int toolnum, float newval);

  //
  // Methods for adding/querying labels
  //

  /// add a label of the given category using the given molecule id's
  /// and atom id's.  Return the index of the label object, or -1 on 
  /// error.  If toggle is true, the on/off status of the label will
  /// be toggled if the label already exists; if the label does not
  /// already exist, the newly created label will be on regardless of
  /// the value of toggle.
  int label_add(const char *category, int num_ids, const int *molids, 
      const int *atomids, const int *cells, float k, int toggle);

  /// turn on/off the nth label of the given category. Return success.
  int label_show (const char *category, int n, int onoff);

  /// delete the nth label of the given category.  If n is -1, delete all
  /// labels from that category. Return success.
  int label_delete(const char *category, int n);

  /// get/size size and thickness of text labels.  This affects all labels.
  float label_get_text_size() const;
  int label_set_text_size(float);
  float label_get_text_thickness() const;
  int label_set_text_thickness(float);

  int label_set_textoffset(const char *nm, int n, float x, float y);
  int label_set_textformat(const char *nm, int n, const char *format);

  /// Get current molid and increment counter by 1
  int next_molid() { return nextMolID++; }

  //
  // Material methods
  //

  /// add material with given name, copying settings from given material.
  /// If name is NULL a default unique name will be chosen; if copyfrom
  /// is NULL the 0th material will be copied.  The name of the new material
  /// is returned, or NULL on error.
  const char *material_add(const char *name, const char *copyfrom);

  /// delete material with given name.  Return success.
  int material_delete(const char *name);

  /// change the given material property.  property is from MaterialProperty
  /// defined in MaterialList.  Return success.
  int material_change(const char *name, int property, float val);

  /// rename the given material.  The new name must contain only 
  /// alphanumeric characters (no spaces).  Return success.
  int material_rename(const char *oldname, const char *newname);

  /// restore the default value of the material with the given index.
  /// Return success.  Fails if the material has no default.
  int material_restore_default(int);

  /// Change the mouse mode.  
  int mouse_set_mode(int mode, int setting);


  /// Change the mobile interface mode.
  int mobile_set_mode(int mode);

  /// Get the mobile interface mode.
  int mobile_get_mode();

  /// Get the list of current clients
  void mobile_get_client_list(ResizeArray <JString*>* &nick, 
                         ResizeArray <JString*>* &ip, ResizeArray <bool>* &active);

  /// Change the mobile interface network port.
  int mobile_network_port(int port);

  /// Get the mobile interface network port.
  int mobile_get_network_port();

  /// Get the version of the API that we support
  int mobile_get_APIsupported();

  /// Set the currently active client, identified by nick and ip
  int mobile_set_activeClient(const char *nick, const char *ip);

  /// Send a message to a specific client
  int mobile_sendMsg(const char *nick, const char *ip, const char *msgType, const char *msg);

  /// return the current mobile interface event data, 
  /// used by the UIVR MobileTracker interface
  void mobile_get_tracker_status(float &tx, float &ty, float &tz, 
                                 float &rx, float &ry, float &rz, 
                                 int &buttons);

  /// Change the spaceball mode.
  int spaceball_set_mode(int mode);

  /// Change the spaceball sensitivity.
  int spaceball_set_sensitivity(float s);

  /// Change the spaceball null region.
  int spaceball_set_null_region(int nr);

  /// return the current spaceball event data, 
  /// used by the UIVR SpaceballTracker interface
  void spaceball_get_tracker_status(float &tx, float &ty, float &tz, 
                                    float &rx, float &ry, float &rz, 
                                    int &buttons);

  /// show Stride message, if necessary
  void show_stride_message();

  /// Show a file dialog.  Use the first available of:
  ///   Tk, Fltk, stdin
  /// Returns a new'd filename, or NULL.
  char *vmd_choose_file(const char *title, 
		const char *extension,
		const char *extension_label,
		int do_save);

  /// Get a unique integer serial number used for identifying display lists 
  static unsigned long get_repserialnum(void);
  static unsigned long get_texserialnum(void);

  /// redraw the screen and update all things that need updatin'.  Return
  /// TRUE until exit has been requested by the user.
  int VMDupdate(int);

  /// text message access methods
  static void set_text(const char* str) { 
     text_message = str; 
  }
  static void append_text(const char* str) { 
     text_message += str; 
  }
  static void clear_text() { 
     text_message = JString(); 
  }
  static const char* get_text() { 
     return text_message; 
  }

  void set_mouse_callbacks(int on);

  void set_mouse_rocking(int on);

  // get/set eofexit status
  void set_eofexit(int onoff) { eofexit = onoff; }
  int get_eofexit() { return eofexit; }

#if defined(VMDXPLOR)
  /// return pointer to VMDApp object used by VMDXPLOR
  static VMDApp* Obj() { return obj; }
#endif
};

  /// function pointer to shared memory allocator/deallocator
  extern "C" void * (*vmd_alloc)(size_t);
  extern "C" void (*vmd_dealloc)(void *);
  extern "C" void * (*vmd_realloc)(void *, size_t);
  extern "C" void * vmd_resize_alloc(void * ptr, size_t oldsize, size_t newsize);
#endif

