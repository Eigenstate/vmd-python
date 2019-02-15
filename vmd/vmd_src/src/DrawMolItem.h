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
 *	$RCSfile: DrawMolItem.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.184 $	$Date: 2019/01/17 21:20:59 $
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
#ifndef DRAWMOLITEM_H
#define DRAWMOLITEM_H

#include "config.h"         // to force recompilation on config changes
#include "Displayable.h"
#include "DispCmds.h"
#include "AtomColor.h"
#include "AtomRep.h"
#include "AtomSel.h"
#include "PickMode.h"
#include "BaseMolecule.h"
#include "VolumeTexture.h"

class DrawMolecule;         ///< forward declaration

#ifdef VMDMSMS
#include "MSMSInterface.h"
#endif

#ifdef VMDNANOSHAPER
#include "NanoShaperInterface.h"
#endif

#ifdef VMDSURF
#include "Surf.h"
#endif

#include "QuickSurf.h"

/// Displayable subclass for creating geometric representations of molecules
class DrawMolItem : public Displayable {
public:
  AtomColor *atomColor;     ///< atom coloring method to use
  AtomRep *atomRep;         ///< atom representation to use
  AtomSel *atomSel;         ///< atom selection to use
  int repNumber;            ///< current representation ID
  char commentBuffer[1024]; ///< buffer for comments  

  char *name;               ///< name of rep, used to identify uniquely.

private:
  DrawMolecule *mol;        ///< parent molecule; kept here for convenience
  float *avg;               ///< boxcar average of the current frame
  int avgsize;              ///< size of the window to use for the average;
                            ///< total size is 2*avgsize+1.
 
  char *framesel;           ///< selection of frames to draw; if NULL, then
                            ///< draw just the current frame

  int structwarningcount;   ///< number of console warnings we've printed

  int emitstructwarning(void); ///< warning suppression function

  //@{ 
  /// useful drawing command objects, used to create display list
  DispCmdBeginRepGeomGroup cmdBeginRepGeomGroup;
  DispCmdColorIndex        cmdColorIndex;
  DispCmdComment           cmdCommentX;
  DispCmdCylinder          cmdCylinder;
  DispCmdLatticeCubeArray  cmdCubeArray;
  DispCmdLine              cmdLine;
  DispCmdLineType          cmdLineType;
  DispCmdLineWidth         cmdLineWidth;
  DispCmdPickPoint         pickPoint;
  DispCmdPointArray        cmdPointArray;
  DispCmdSphere            cmdSphere;
  DispCmdSphereRes         cmdSphres;
  DispCmdSphereType        cmdSphtype;
  DispCmdSphereArray       cmdSphereArray;
  DispCmdSquare            cmdSquare;
  DispCmdTriangle          cmdTriangle;
  DispCmdTriMesh           cmdTriMesh;
  DispCmdVolSlice          cmdVolSlice;
  DispCmdWireMesh          cmdWireMesh;
  //@}

  /// Color/atom lookup table.  Updated when color or selection is changed.
  struct ColorLookup {
    int num;
    int max;
    int *idlist;
    ColorLookup() {
      num = 0;
      max = 0;
      idlist = 0;
    }
    ~ColorLookup() {
      free(idlist);
    }
    void append(int id) {
      if (max == 0) {
        idlist = (int *)malloc(8L*sizeof(int)); 
        max = 8;
      }
      if (num == max) {
        idlist = (int *)realloc(idlist, 2L*num*sizeof(int));
        max = 2L*num;
      }
      idlist[num++] = id;
    }
  };
 
  /// color/atomid lookup table used by Line drawing routines 
  ColorLookup *colorlookups;      
  static void update_lookups(AtomColor *, AtomSel *, ColorLookup *);

  /// Store periodic boundary transformations in the display list
  void update_pbc_transformations();

  /// Store instance transformations in the display list
  void update_instance_transformations();

#ifdef VMDMSMS
  MSMSInterface msms;          ///< MSMS interface class object
#endif

#ifdef VMDNANOSHAPER
  NanoShaperInterface nanoshaper; ///< NanoShaper interface class object
#endif

#ifdef VMDSURF
  Surf surf;                   ///< Surf interface class object
#endif

  int waveftype;               ///< Wavefunction type
  int wavefspin;               ///< Wavefunction spin
  int wavefexcitation;         ///< Wavefunction excitation
  int gridorbid;               ///< Orbital ID associated with cached grid
  float orbgridspacing;        ///< Orbital grid spacing
  int orbgridisdensity;        ///< Grid contains density rather than amplitude
  VolumetricData *orbvol;      ///< Orbital grid

  // cached volume texture information to avoid unnecessary updates
  int voltexVolid;
  int voltexColorMethod;
  float voltexDataMin, voltexDataMax;

  // encapsulate data for color by volume used by this rep
  VolumeTexture volumeTexture;

  // Auxiliary data structure for rendering tubes.
  // Since the control points are based on fixed atom indices, we can
  // find all those indices once, cache them, then just look up the
  // coordinates when it's time to draw a frame.  Should speed up
  // animation, as well as perhaps make it easier to add better-looking
  // tube reps later.
 
  // TubeIndexList holds a set of control point indices.
  typedef ResizeArray<int> TubeIndexList;

  /// holds a set of tube control point indices.  
  /// We can't free memory from ResizeArrays so we make this a pointer to 
  /// allow us to save memory when not drawing tubes.
  ResizeArray<TubeIndexList *> *tubearray;

  /// regenerate the tubearray
  void generate_tubearray();

  // commands to draw different representations for a given selection 
  void draw_solid_cubes(float *, float radscale);
  void draw_solid_spheres(float *, int res, float radscale, float fixrad);
  void draw_residue_beads(float *, int res, float radscale);
  void draw_dotted_spheres(float *, float srad, int sres); ///< dotted reps
  void draw_lines(float *, int thickness, float cutoff);   ///< lines rep
  void draw_cpk_licorice(float *, int, float, int, float, int, int, float cutoff); ///< cpk and licorice reps

#ifdef VMDPOLYHEDRA
  void draw_polyhedra(float *, float); ///< polyhedra reps
#endif
  void draw_points(float *, float);                   ///< points rep
  void draw_bonds(float *, float brad, int bres, float cutoff);             ///< bonds rep

  /// routine to eliminate cylinder gaps in tube, ribbon and bonds reps
  void make_connection(float *prev, float *start, float *end, float *next,
		       float radius, int resolution, int is_cyl);

  /// routine to draw a spline curve component, used by tube and ribbon reps
  void draw_spline_curve(int num, float *coords, int *idx,
			 int use_cyl, float b_rad, int b_res);

  /// ribbon representation
  void draw_spline_ribbon(int num, float *coords, float *perps,
			  int *idx, int use_cyl, float b_rad,
			  int b_res);

  /// ribbon/tube representations
  void draw_spline_new(int num, const float *coords, 
                       const float *perps, const int *idx, 
                       const float *cpwidths, const float *cpheights, 
                       int numscalefactors, int b_res, int cyclic);

  /// new ribbon representation
  void draw_ribbon_from_points(int numpoints, const float *points, 
                  const float *perps, const int *cols, int numpanels,
                  const float *heights, const float *widths, 
                  int numscalefactors);

  void draw_tube(float *, float radius, int res);   ///< tube rep
  void draw_ribbons(float *, float brad, int bres, float thickness); ///< ribbon rep
  void draw_ribbons_new(float *, float, int, int, float);  ///< new ribbon rep
  //@{
  /// ribbon representation helper routines
  int draw_protein_ribbons_old(float *, int, float, float, int);
  int draw_protein_ribbons_new(float *, int, float, float, int);
  int draw_nucleic_ribbons(float *, int, float, float, int, int, int);
  int draw_nucleotide_cylinders(float *, int, float, float, int);
  int draw_base_sugar_rings(float *, int, float, float, int);
  int draw_cartoon_ribbons(float *, int, float, float, int, int);
  //@}

  void draw_structure(float *, float brad, int bres, int linethickness); ///< 2ndary structure, cartoon rep
  /// part of cartoon representation
  void draw_alpha_helix_cylinders(ResizeArray<float> &x,
				  ResizeArray<float> &y,
				  ResizeArray<float> &z,
				  ResizeArray<int> &atom_on, int *color,
				  float bond_rad, int bond_res,
				  float *res_start, float *res_end);

  /// part of cartoon representation
  void draw_beta_sheet(ResizeArray<float> &x, ResizeArray<float> &y,
		       ResizeArray<float> &z, ResizeArray<int> &atom_on, 
		       int *color, float ribbon_width,
		       float *res_start, float *res_end);

  void draw_trace(float *pos, float brad, int bres, int linethickness); ///< C-alpha trace
  void draw_dot_surface(float *pos, float srad, int sres, int method); ///< dot surface
  void draw_msms(float *pos, int draw_wireframe, int allatoms, float radius, float density); ///< MSMS surface from Scripps
  void draw_nanoshaper(float *pos, int surftype, int draw_wireframe, float gspacing, float probe_rad, float skin_parm, float blob_parm); ///< NanoShaper surface
  void draw_quicksurf(float *pos, int quality, float radius, float isovalue, float spacing); ///< Fast surface representation
  void draw_surface(float *pos, int draw_wireframe, float radius); ///< Surf surface from UNC
  void draw_hbonds(float *, float maxangle, int thickness, float cutoff); ///< Hydrogen bonds
  void draw_dynamic_bonds(float *, float brad, int bres, float cutoff); ///< on-the-fly bond animation

  // Drawing functions for volumetric and isosurface data 
  void draw_volslice(int volid, float slice, int axis, int texmode); ///< Volume data, textured slices 

  /// update the volumeTexture instace with current settings
  void updateVolumeTexture();

  void draw_isosurface(int, float, int, int, int, int); ///< Volume data, isosurface
  void draw_volume_field_lines(int volid, int seedusegrid, int maxseeds, float seedval, float deltacell, float minlen, float maxlen, int drawtubes, int tuberes, float thickness); ///< Volume gradient field lines

  void draw_orbital(int, int, int, int, int, float, int, int, float, int, int); ///< QM orbital isosurface

  //@{
  // helper functions for volume rendering
  void draw_volume_box_solid(VolumetricData *);
  void draw_volume_box_lines(VolumetricData *);
  void draw_volume_slice(const VolumetricData *, int, float, int);
  void draw_volume_texture(const VolumetricData *, int);
  void draw_volume_isosurface_points(const VolumetricData *, float, int, int);
  void draw_volume_isosurface_lit_points(VolumetricData *, float, int, int);
  void draw_volume_isosurface_trimesh(VolumetricData *, float, int, const float *voltex=NULL);
  void draw_volume_isosurface_lines(VolumetricData *, float, int, int);
  int  draw_volume_get_colorid(void);
  void prepare_volume_texture(const VolumetricData *v, int method);
  int calcseeds_grid(VolumetricData *v, ResizeArray<float> *seeds, int maxseedcount);
  int calcseeds_gradient_magnitude(VolumetricData *v, ResizeArray<float> *seeds, float seedmin, float seedmax, int maxseedcount);

  //@}

#ifdef VMDWITHCARBS
  //@{
  // functions for rendering small rings using the PaperChain algorithm
  void draw_rings_paperchain(float *framepos, float bipyramid_height, int maxringsize);
  void paperchain_get_ring_color(SmallRing &ring, float *framepos, float *rgb);
  void paperchain_draw_ring(SmallRing &ring, float *framepos, float bipyramid_height);
  //@}
  
  //@{
  // functions for rendering small rings using the Twister algorithm
  void draw_rings_twister(float *framepos, int start_end_centroid, int hide_shared_links, int rib_steps, float rib_width, float rib_height, int maxringsize, int maxpathlength);
  void twister_draw_path(LinkagePath &path, float *framepos, int start_end_centroid, int rib_steps, float rib_width, float rib_height);
  void twister_draw_ribbon_extensions(ResizeArray<float> &vertices, ResizeArray<float> &colors,
                                      ResizeArray<float> &normals, ResizeArray<int> &facets,
                                      float centroid[3], float normal[3], float right[3], float rib_point[3],
                                      float rib_height, float rib_width,
                                      float top_color[3], float bottom_color[3]);
  void twister_draw_hexagon(ResizeArray<float> &vertices, ResizeArray<float> &colors, ResizeArray<float> &normals,
                                      ResizeArray<int> &facets, float centroid[3], float normal[3],
                                      float first_atom[3], float rib_height, float rib_width,
                                      float top_color[3], float bottom_color[3]);
  //@}
  
  //@{
  // utility functions for Twister and Paper Chain
  void get_ring_centroid_and_normal(float *centroid, float *normal, SmallRing &ring, float *framepos);
  bool smallring_selected(SmallRing &ring);
  bool linkagepath_selected(LinkagePath &path);
  //@}
#endif

  void create_cmdlist(void);            ///< regenerate the command list
  void do_create_cmdlist();             ///< used to loop over frames

public:
  /// Keep track of whether and why we have to regenerate rendering lists
  /// In particular, MOL_REGEN means molecular data has changed, whereas
  /// if all we did was change from one timestep to another, and
  /// nothing in the timestep has changed, 
  /// then the update_ts flag is changed instead.  The idea is that when we
  /// implement cached geometry for multiple frames, or a draw all frames
  /// option, we don't want to recreate the geometry unnecessarily.  
  enum RegenChoices {NO_REGEN = 0, MOL_REGEN = 1, SEL_REGEN = 2,
                     REP_REGEN = 4, COL_REGEN = 8};

private:
  int needRegenerate;                   ///< regeneration flag
  int update_pbc;                       ///< Flag for updating PBC data
  int update_instances;                 ///< Flag for updating instance data
  int update_ss;                        ///< Flag for updating sec. structure
  int update_ts;                        ///< Flag for update timestep
  int update_traj;                      ///< Flag for modified set of timesteps
  void place_picks(float *pos);         ///< create pick points for 'on' atoms

protected:
  virtual void do_color_changed(int);
  virtual void do_color_rgb_changed(int);
  virtual void do_color_scale_changed();

public:
  // constructor: name, parent molecule, and atom drawing methods
  DrawMolItem(const char *, DrawMolecule *, AtomColor *, AtomRep *, AtomSel *);
  virtual ~DrawMolItem(void);
  
  int change_color(AtomColor *);        ///< change coloring method
  int change_rep(AtomRep *);            ///< change representation method
  int change_sel(const char *);         ///< change atom selection
  void force_recalc(int);               ///< force a recalculation of everything
  int atom_displayed(int);              ///< if Nth atom is displayed in ANY rep
  int representation_index(void);       ///< which representation this is

  int set_smoothing(int n) {           ///< Set window size; 0 for off
    if (n >= 0) {
      avgsize = n; 
      return TRUE;
    }
    return FALSE;
  }
  int get_smoothing() const {           ///< Returns size of smoothing window.
    return avgsize;
  }

  /// Controls for drawing a subset of the molecule's frames.  Pass a list of
  /// frames to be drawn, or "now" to draw just the current frame (the default).
  /// Always succeeds, invalid or out of range frames are ignored.  
  void set_drawframes(const char *frames);
  const char *get_drawframes() const { return framesel; }
   
  /// Controls for display of periodic boundary conditions.  
  /// Uses the PBC_x defines in VMDDisplayList. 
  void set_pbc(int);
  int get_pbc() const;

  /// Set the number of of periodic image replicas.  Must be 1 or higher.
  void set_pbc_images(int);
  int get_pbc_images() const;

  /// the pbc parameters in Timestep have changed; update the transformation
  /// matrices accordingly (but no need to regenerate display lists).
  void change_pbc() { update_pbc = 1; }


  /// Controls for display of molecule instances
  /// Uses the INSTANCE_x defines in VMDDisplayList. 
  void set_instances(int);
  int get_instances() const;

  /// the instance parameters have changed; update the transformation
  /// matrices accordingly (but no need to regenerate display lists).
  void change_instances() { update_instances = 1; }


  /// the secondary structure for the molecule has changed
  void change_ss() { update_ss = 1; }

  /// the currently displayed timestep has changed. 
  void change_ts() { update_ts = 1; }

  /// timesteps have been added or removed from the parent molecule
  void change_traj() { update_traj = 1; }

  //
  // public virtual routines
  //
  virtual void prepare();               ///< prepare for drawing, do updates 

  /// override pickable_on so that it returns true only when both the 
  /// rep and the molecule are on.
  virtual int pickable_on();

  //@{
  /// When picked, pass my parent molecule to the pick mode
  virtual void pick_start(PickMode *p, DisplayDevice *d,
                          int btn, int tag, const int *cell, int dim, 
                          const float *pos) {
    p->pick_molecule_start(mol, d, btn, tag, cell, dim, pos);
  }

  virtual void pick_move(PickMode *p, DisplayDevice *d,
                          int tag, int dim, const float *pos) {
    p->pick_molecule_move (mol, d, tag, dim, pos);
  }

  virtual void pick_end(PickMode *p, DisplayDevice *d) {
    p->pick_molecule_end  (mol, d);
  }
  //@}

protected:
  float spline_basis[4][4];          ///< spline basis used for ribbons/tubes
};

#endif

