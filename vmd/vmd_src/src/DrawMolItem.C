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
 *	$RCSfile: DrawMolItem.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.367 $	$Date: 2019/01/23 22:37:09 $
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

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <ctype.h>              // for isdigit()

#include "DrawMolItem.h"
#include "Molecule.h"  // for file_in_progress()
#include "DispCmds.h"
#include "Inform.h"
#include "Scene.h"
#include "TextEvent.h"
#include "BondSearch.h"
#include "DisplayDevice.h"
#ifdef VMDMSMS
#include "MSMSInterface.h" // Interface to MSMS surface generation program
#endif
#ifdef VMDNANOSHPER
#include "NanoShaperInterface.h" // Interface to NanoShaper surface program
#endif
#ifdef VMDSURF
#include "Surf.h"          // this is an interface to the SURF program
#endif
#include "VMDApp.h"        // for vmd_alloc/vmd_dealloc
#include "VolumetricData.h"

//////////////////////////  constructor  
DrawMolItem::DrawMolItem(const char *nm, DrawMolecule *dm, AtomColor *ac, 
        AtomRep *ar, AtomSel *as) 
	: Displayable(dm) {

  // save data and pointers to drawing method objects
  mol = dm;
  avg = new float[3L*mol->nAtoms];
  avgsize = 0;
  atomColor = ac;
  atomRep = ar;
  atomSel = as;
  structwarningcount = 0;

  name = stringdup(nm);
  framesel = stringdup("now");
  tubearray = NULL;

  colorlookups = new ColorLookup[MAXCOLORS]; // for color sorted line drawing

  // use non-standard Catmull-Rom spline, with slope of 1.25 instead of 2.0 
  create_modified_CR_spline_basis(spline_basis, 1.25f);

  // set orbital and volumetric data objects sentinel values or to NULL
  waveftype = -1;
  wavefspin = -1;
  wavefexcitation = -1;
  gridorbid = -1;
  orbgridspacing = -1.0f; 
  orbgridisdensity = -1;
  orbvol = NULL;

  // initialize volume texture data to invalid values
  voltexVolid = -1;
  voltexColorMethod = -1;
  voltexDataMin = 0;
  voltexDataMax = 0;

  // signal that we need a complete refresh next prepare
  needRegenerate = MOL_REGEN | COL_REGEN | SEL_REGEN;
  update_pbc = 0;
  update_ss = 0;
  update_ts = 0;
  update_traj = 0;
  update_instances = 0;
  isOn = TRUE;     // newly created reps default to on
}

//////////////////////////  destructor  
DrawMolItem::~DrawMolItem(void) {
  if (tubearray) {
    for (int i=0; i<tubearray->num(); i++) 
      delete (*tubearray)[i];
    delete tubearray;
    tubearray = NULL;
  }

  delete atomColor;
  delete atomRep;
  delete atomSel;
  delete [] colorlookups; // color/atomid lookup for color sorted line drawing
  delete [] name;
  delete [] avg;
  delete [] framesel;
  delete orbvol;
}

int DrawMolItem::emitstructwarning(void) {
  if (structwarningcount < 30) {
    structwarningcount++;
    return 1;
  } 

  if (structwarningcount == 30) {
    msgErr << "Maximum structure display warnings reached, further warnings will be suppressed" << sendmsg; 
    structwarningcount++;
    return 0;
  }

  structwarningcount++;
  return 0;
}

void DrawMolItem::update_lookups(AtomColor *ac, AtomSel *sel, 
                                    ColorLookup *lookups) {
  int i;
  for (i=0; i<MAXCOLORS; i++) {
    lookups[i].num = 0;
  }
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      int color = ac->color[i];
      lookups[color].append(i);
    }
  }
}
  
void DrawMolItem::set_pbc(int pbc) {
  cmdList->pbc = pbc;
  change_pbc(); // tell rep to update PBC transformation matrices next prepare

  // tell the molecule to notify its geometry monitors that the number of
  // periodic images has changed so that they can be turned off or on if
  // necessary.  XXX This may not be currently implemented in GeometryMol.
  mol->notify();
}

int DrawMolItem::get_pbc() const {
  return cmdList->pbc;
}

void DrawMolItem::set_pbc_images(int n) {
  if (n < 1) return;
  cmdList->npbc = n;
  need_matrix_recalc();
  mol->notify();
}
int DrawMolItem::get_pbc_images() const {
  return cmdList->npbc;
}


void DrawMolItem::set_instances(int inst) {
  cmdList->instanceset = inst;
  change_instances(); // tell rep to update instance matrices next prepare

  // tell the molecule to notify its geometry monitors that the number of
  // instance images has changed so that they can be turned off or on if
  // necessary.  XXX This may not be currently implemented in GeometryMol.
  mol->notify();
}

int DrawMolItem::get_instances() const {
  return cmdList->instanceset;
}


static void parse_frames(const char *beg, int len, int maxframe, 
    ResizeArray<int>& frames) {
  if (!len) {
    msgErr << "parse_frames got zero-length string!" << sendmsg;
    return;
  }
  // Look first for a comma separated list of frames
  int i;
  const char *firstsep = NULL, *secondsep = NULL;
  char *endptr, *token;
  for (i=0; i<len; i++) {
    if (beg[i] == ',') {
        firstsep = beg+i;
        break;
    }
  }
  long first, second, third;
  if (firstsep) {
      char *parseme = strndup(beg, len);
      const char delim = ',';
      while (true) {
        token = strsep(&parseme, &delim);
        if (!token) {
            break;
        }
        first = strtol(token, &endptr, 0);
        if (endptr == token) {
            msgErr << "frame element is invalid: " << token << sendmsg;
        }
        frames.append((int) first);
      }
      free(parseme);
      return;
  }
  
  for (i=0; i<len; i++) {
    if (beg[i] == ':') {
      firstsep = beg+i;
      break;
    }
  }
  for (++i; i < len; i++) {
    if (beg[i] == ':') {
      secondsep = beg+i;
      break;
    }
  }
  first = strtol(beg, &endptr, 0);
  if (endptr == beg) {
    msgErr << "frame element is invalid: " << beg << sendmsg;
    return;
  }
  second = third = first; // quiet compiler warnings
  if (firstsep) {
    firstsep++;
    // if no text after separator, then assume last frame.
    if (!(*firstsep)) {
      second = maxframe;
    } else {
      second = strtol(firstsep, &endptr, 0);
      if (endptr == firstsep) {
        msgErr << "frame element is invalid: " << beg << sendmsg;
        return;
      }
    }
  }
  if (secondsep) {
    secondsep++;
    if (!(*secondsep)) {
      third = maxframe;
    } else {
      third = strtol(secondsep, &endptr, 0);
      if (endptr == secondsep) {
        msgErr << "frame element is invalid: " << beg << sendmsg;
        return;
      }
    }
  }
  // append frames based on what we got:
  if (!firstsep) {
    //printf("adding frame %d\n", (int)first);
    frames.append((int)first);
  } else if (!secondsep) {
    for (long i=first; i <= second; i++) {
      //printf("adding frame %d\n", (int)i);
      frames.append((int)i);
    }
  } else {
    // prevent zero, or negative frame step sizes
    if (second < 1) {
      msgErr << "zero or negative step size is invalid: " << second << sendmsg;
      return;
    }
    // printf("first: %ld third: %ld second: %ld\n", first, third, second);

    for (long i=first; i <= third; i += second) {
      //printf("adding frame %d\n", (int)i);
      frames.append((int)i);
    } 
  }
}

void DrawMolItem::set_drawframes(const char *frames) {
  if (!frames) return;
  while (isspace(*frames)) frames++;
  if (!strncmp(frames, "now", 3) && !strncmp(framesel, "now", 3)) return;
  delete [] framesel;
  framesel = stringdup(frames);
  needRegenerate |= MOL_REGEN;
}

//////////////////////////////// private routines 

void DrawMolItem::create_cmdlist() {
  int i;

  // do we need to recreate everything?
  if (needRegenerate) {
    // Identify the representation number for output comments
    repNumber = -1;   // Nonsense default number
    for (i = 0; i < mol->components(); i++) {
      if (this == mol->component(i))
        repNumber = i;
    }

    if (needRegenerate & COL_REGEN)
      atomColor->find(mol); // execute any pending color updates

    if (needRegenerate & SEL_REGEN)
      atomSel->change(NULL, mol); // update atom selection onoff flags

    if (needRegenerate & REP_REGEN) {
      // dup cmdStr to prevent overlapping strcpy() buffer pointers
      char *newcmdstr = stringdup(atomRep->cmdStr);
      atomRep->change(newcmdstr); // update the representation text
      delete [] newcmdstr;
    }
 
    reset_disp_list(); // regenerate both data block and display commands

    //
    // Record the start of a new representation geometry group
    // This can only be called once per rep, as the generated 
    // group names need to be unique.
    char repbuf[2048];
    sprintf(repbuf, "vmd_mol%d_rep%d", mol->id(), repNumber);
    cmdBeginRepGeomGroup.putdata(repbuf, cmdList);

    // If coloring by a volume texture, setup the graphics state
    //
    // We want to generate the fewest number of volume textures possible,
    // so we actually put this code outside all per-timestep and
    // per-pbc image drawing loops.  That way we'll minimize the number of
    // OpenGL texture downloads.  I might be able to make low level renderer
    // smart enough to recognize textures by some unique serial number, but
    // that code doesn't exist yet, so the next best thing is to be extra 
    // careful in generating them in the first place.
    //
    // XXX One gotcha WRT volume textures, is that since we clamp them at
    //     the edge of the texture, there is no provision to make PBC images
    //     of volumes render nicely.  Since the PBC box isn't necessarily the
    //     same coordinate system as the volume set, not much we can do there.
    //     We'd have to implement a separate PBC transform feature for the
    //     volumetric data in order to correctly solve this.  For now, we'll
    //     just live without it.
    if (atomColor->method() == AtomColor::VOLUME ||
        atomRep->method() == AtomRep::VOLSLICE) {

      updateVolumeTexture();
      // Pass a pointer to the texture map, retained in shared memory
      float v0[3], v1[3], v2[3], v3[3];
      volumeTexture.calculateTexgenPlanes(v0, v1, v2, v3);
      DispCmdVolumeTexture cmdVolTexture;
      cmdVolTexture.putdata(
          volumeTexture.getTextureID(),
          volumeTexture.getTextureSize(),
          volumeTexture.getTextureMap(),
          v0, v1, v2, v3,
          cmdList);
    }

    if (strcmp(framesel, "now")) {
      // parse the frame selection into a list of frames to process 
      ResizeArray<int> frames;
      const char *endptr = framesel;
      int maxframe = mol->numframes() - 1;
      do {
        const char *begptr = endptr;
        while (isspace(*begptr)) begptr++;
        endptr = begptr;
        while (*endptr && !isspace(*endptr)) endptr++;
        parse_frames(begptr, endptr-begptr, maxframe, frames);
      } while (*endptr);

      int curframe = mol->frame(); // save current frame for later

      // loop over the selected frames and render them all
      for (i=0; i<frames.num(); i++) {
        int frame = frames[i];
        if (frame < 0 || frame > maxframe) 
          continue;

        mol->override_current_frame(frame);  // override current frame

        // force timestep color to update, redraw or recolor affected geometry
        // XXX for the POS[XYZ] coloring methods, the user will need to 
        // select "update color every frame" in order to get time-varying color
        switch (atomColor->method()) {
          case AtomColor::USER:
          case AtomColor::USER2:
          case AtomColor::USER3:
          case AtomColor::USER4:
          case AtomColor::PHYSICALTIME:
          case AtomColor::TIMESTEP:
          case AtomColor::VELOCITY:
            atomColor->find(mol);  // recalc atom colors and redraw
            break;
        }   
        do_create_cmdlist();     // draw selected timestep(s)
      }
      mol->override_current_frame(curframe); // restore previous frame
    } else {
      do_create_cmdlist();       // draw the current timestep only
    }

    // If we have a volume texture enabled, we must turn it back off here
    if (atomColor->method() == AtomColor::VOLUME ||
        atomRep->method() == AtomRep::VOLSLICE) {
      append(DVOLTEXOFF);
    }

    cacheskip(1);
  } else {
    cacheskip(0);
  }

  // XXX work around a bug with texturing and display list caching
  //     The OpenGL texture object is not stored in the display list 
  //     unless the rep was created while display list caching was enabled.
  //     If the rep was created beforehand, then the display list will not
  //     contain the texture data, and thus things will be drawn incorrectly.
  //     For now we'll take the easy way out and disable caching of geometry
  //     that is textured.  The right thing to do would be to add a new 
  //     flag to tell the lowest level render() routine to regen the textures
  //     at the time display lists are created.  This is a reasonable short
  //     term solution however.
  if (atomColor->method() == AtomColor::VOLUME ||
      atomRep->method() == AtomRep::VOLSLICE) {
    cacheskip(1); // don't allow this rep to be cached
  }

  needRegenerate = NO_REGEN; // done regenerating
}


// regenerate the command list
void DrawMolItem::do_create_cmdlist(void) {
  float *framepos;
  int i;

  // only render frame-oriented reps if there is a current frame
  if (mol->current()) {
    if (atomSel->do_update)
      atomSel->change(NULL, mol);     // update atom selection if necessary

    if (atomColor->do_update)
      atomColor->find(mol);           // update colors if necessary

    framepos = (mol->current())->pos; // get atom coordinates for this frame

    // average atom coordinates using a window of size ((2*avgsize) + 1)
    if (avgsize && atomSel->selected > 0) {
      const int curframe = mol->frame();
      const int begframe = curframe - avgsize;
      const int endframe = curframe + avgsize;
      const int lastframe = mol->numframes() - 1;
#if 0
      // Only smooth coordinates of selected atoms
      // XXX This would be a big performance gain in most cases, but
      //     we can't use it if we're drawing a rep that references
      //     coordinates for atoms outside the selection...
      //     In order to use this, we'll have to validate that the
      //     active representation doesn't touch non-selected atom coords
      const int atomloopfirst = atomSel->firstsel;
      const int atomlooplast  = 3L*(atomSel->lastsel+1);
#else
      // Smooth all atom coordinates
      const int atomloopfirst = 0;
      const int atomlooplast  = 3L*mol->nAtoms;
#endif
      const float rescale = 1.0f/(2.0f*avgsize+1.0f);

#define VMDPBCSMOOTH 1
#ifdef VMDPBCSMOOTH
      // for pbc aware averaging
      int isortho,usepbc;
      float a,b,c,alpha,beta,gamma;

      // set some arbitrary defaults
      isortho=usepbc=1;
      a=b=c=9999999.0f;
      alpha=beta=gamma=90.0f;

      // reference timestep for pbc wrap
      const float *ref = mol->get_frame(curframe)->pos;
#endif      

      memset(avg, 0, 3L*mol->nAtoms*sizeof(float)); // clear average array

      for (i=begframe; i<=endframe; i++) {
        int ind = i;
        if (ind < 0) 
          ind = 0;
        else if (ind > lastframe) 
          ind = lastframe;
        const float *ts = mol->get_frame(ind)->pos;

#ifdef VMDPBCSMOOTH
        // get periodic cell information for current frame
        a = mol->get_frame(ind)->a_length;
        b = mol->get_frame(ind)->b_length;
        c = mol->get_frame(ind)->c_length;
        alpha = mol->get_frame(ind)->alpha;
        beta = mol->get_frame(ind)->beta;
        gamma = mol->get_frame(ind)->gamma;

        // check validity of PBC cell side lengths
        if (fabsf(a*b*c) < 0.0001)
          usepbc=0;

        // check PBC unit cell shape to select proper low level algorithm.
        if ((alpha != 90.0) || (beta != 90.0) || (gamma != 90.0))
          isortho=0;
        
        // until we can handle non-orthogonal periodic cells turn off pbc handling
        if (!isortho)
          usepbc=0;

        if (usepbc) {
          const float ahalf=a*0.5f;
          const float bhalf=b*0.5f;
          const float chalf=c*0.5f;

          // smooth by offsetting with computed ifs using the current frame as reference.
          // (in the hope that the compiler can optimize this better than regular nested ifs).
          // only works for orthogonal cells.
          for (int j=atomloopfirst; j<atomlooplast; j += 3) {
            float adiff=ts[j  ]-ref[j  ];
            avg[j  ] += ts[j  ] + ((adiff < -ahalf) ? a : ( (adiff > ahalf) ? -a : 0));
            float bdiff=ts[j+1]-ref[j+1];
            avg[j+1] += ts[j+1] + ((bdiff < -bhalf) ? b : ( (bdiff > bhalf) ? -b : 0));
            float cdiff=ts[j+2]-ref[j+2];
            avg[j+2] += ts[j+2] + ((cdiff < -chalf) ? c : ( (cdiff > chalf) ? -c : 0));
          }
        } else {                // normal smoothing
          for (int j=atomloopfirst; j<atomlooplast; j++) {
            avg[j] += ts[j];
          }
        }
#else 
        for (int j=atomloopfirst; j<atomlooplast; j++) {
          avg[j] += ts[j];
        }
#endif
  
      }

      // scale results appropriately
      for (int j=atomloopfirst; j<atomlooplast; j++)
        avg[j] *= rescale;

      framepos = avg; // framepos points to the new average values
    }

    // now put in drawing commands, which index into the above block
    switch (atomRep->method()) {
      case AtomRep::LINES:        
        draw_lines(framepos, 
          (int)atomRep->get_data(AtomRep::LINETHICKNESS),
          atomRep->get_data(AtomRep::ISOLINETHICKNESS));
        place_picks(framepos);
        break;

      case AtomRep::BONDS:
        draw_bonds(framepos, 
          atomRep->get_data(AtomRep::BONDRAD),
          (int)atomRep->get_data(AtomRep::BONDRES),
          atomRep->get_data(AtomRep::ISOLINETHICKNESS));
        place_picks(framepos);
        break;

      case AtomRep::DYNAMICBONDS: 
        draw_dynamic_bonds(framepos, 
          atomRep->get_data(AtomRep::BONDRAD),
          (int)atomRep->get_data(AtomRep::BONDRES),
          atomRep->get_data(AtomRep::SPHERERAD)); 
        break;

      case AtomRep::HBONDS:
        draw_hbonds(framepos,
          atomRep->get_data(AtomRep::SPHERERAD),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS),
          atomRep->get_data(AtomRep::BONDRAD)); 
        break;

      case AtomRep::POINTS:       
        draw_points(framepos, atomRep->get_data(AtomRep::LINETHICKNESS)); 
        place_picks(framepos);
        break;

      case AtomRep::VDW:          
        draw_solid_spheres(framepos,
          (int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::SPHERERAD),
          0.0);
        place_picks(framepos);
        break;

      case AtomRep::CPK:          
        draw_cpk_licorice(framepos, 1, 
          atomRep->get_data(AtomRep::BONDRAD),           // bond rad
          (int)atomRep->get_data(AtomRep::BONDRES),      // bond res
          atomRep->get_data(AtomRep::SPHERERAD),         // scaled VDW rad
          (int)atomRep->get_data(AtomRep::SPHERERES),    // sphere res
          (int)atomRep->get_data(AtomRep::LINETHICKNESS),// line thickness
          atomRep->get_data(AtomRep::ISOLINETHICKNESS)); // bonds cutoff
        place_picks(framepos);
        break;

      case AtomRep::LICORICE:     
        draw_cpk_licorice(framepos, 0, 
          atomRep->get_data(AtomRep::BONDRAD),            // bond rad
          (int)atomRep->get_data(AtomRep::BONDRES),       // bond res
          atomRep->get_data(AtomRep::SPHERERAD),          // scaled VDW rad
          (int)atomRep->get_data(AtomRep::SPHERERES),     // sphere res
          (int)atomRep->get_data(AtomRep::LINETHICKNESS), // line thickness
          atomRep->get_data(AtomRep::ISOLINETHICKNESS));  // bonds cutoff
        place_picks(framepos);
        break;

#ifdef VMDPOLYHEDRA
      case AtomRep::POLYHEDRA:     
        draw_polyhedra(framepos, atomRep->get_data(AtomRep::SPHERERAD)); 
        break;
#endif

      case AtomRep::TRACE:
        draw_trace(framepos,
          atomRep->get_data(AtomRep::BONDRAD),
          (int)atomRep->get_data(AtomRep::BONDRES),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS));
        place_picks(framepos);
        break;

      case AtomRep::TUBE:
        draw_tube(framepos, 
          atomRep->get_data(AtomRep::BONDRAD),
          (int)atomRep->get_data(AtomRep::BONDRES));
        break;

      case AtomRep::RIBBONS:
        draw_ribbons(framepos, 
          atomRep->get_data(AtomRep::BONDRAD) / 3.0f,
          (int)atomRep->get_data(AtomRep::BONDRES),
          atomRep->get_data(AtomRep::LINETHICKNESS));
        break;

      case AtomRep::NEWRIBBONS:
        draw_ribbons_new(framepos,
          atomRep->get_data(AtomRep::BONDRAD) / 3.0f,
          (int)atomRep->get_data(AtomRep::BONDRES),
          (int)atomRep->get_data(AtomRep::SPHERERAD), // use bspline or not
          atomRep->get_data(AtomRep::LINETHICKNESS));
        break;

#ifdef VMDWITHCARBS
      case AtomRep::RINGS_PAPERCHAIN:
        draw_rings_paperchain(framepos,
                              atomRep->get_data(AtomRep::LINETHICKNESS), // bipyramid_height
                              (int)atomRep->get_data(AtomRep::ISOSTEPSIZE) // maximum ring size
                              ); 
        place_picks(framepos); // XXX add better pick points in the ring traversal code
        break;

      case AtomRep::RINGS_TWISTER:
        draw_rings_twister(framepos,
                           (int)atomRep->get_data(AtomRep::LINETHICKNESS), // start_end_centroid
                           (int)atomRep->get_data(AtomRep::BONDRES), // hide_shared_links
                           (int)atomRep->get_data(AtomRep::SPHERERAD), // rib_steps
                           atomRep->get_data(AtomRep::BONDRAD), // rib_width
                           atomRep->get_data(AtomRep::SPHERERES), // rib_height
                           (int)atomRep->get_data(AtomRep::ISOSTEPSIZE), // maximum ring size
                           (int)atomRep->get_data(AtomRep::ISOLINETHICKNESS) // maximum link length
                           );
        place_picks(framepos); // XXX add better pick points in the ring traversal code
        break;
#endif

      case AtomRep::NEWCARTOON:   
        draw_cartoon_ribbons(framepos, 
                             (int)atomRep->get_data(AtomRep::BONDRES),
                             atomRep->get_data(AtomRep::BONDRAD), 
                             (float) atomRep->get_data(AtomRep::LINETHICKNESS),
                             1,
                             (int)atomRep->get_data(AtomRep::SPHERERAD));
        break;

      case AtomRep::STRUCTURE:
        draw_structure(framepos,
          atomRep->get_data(AtomRep::BONDRAD),
          (int)atomRep->get_data(AtomRep::BONDRES),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS));
        break;

#if defined(VMDNANOSHAPER)
      case AtomRep::NANOSHAPER:
        draw_nanoshaper(framepos, 
          (int)atomRep->get_data(AtomRep::LINETHICKNESS), // surftype
          (int)atomRep->get_data(AtomRep::BONDRES),       // draw wireframe
          atomRep->get_data(AtomRep::GRIDSPACING),        // grid spacing
          atomRep->get_data(AtomRep::SPHERERAD),          // probe radius
          atomRep->get_data(AtomRep::SPHERERES),          // skin parm
          atomRep->get_data(AtomRep::BONDRAD));           // blob parm
        break;
#endif
#ifdef VMDMSMS
      case AtomRep::MSMS:
        draw_msms(framepos, 
          (int)atomRep->get_data(AtomRep::BONDRES),    // draw wireframe
          (atomRep->get_data(AtomRep::LINETHICKNESS) < 0.5), // all / selected
          atomRep->get_data(AtomRep::SPHERERAD),  // probe radius
          atomRep->get_data(AtomRep::SPHERERES)); // point density 
        break;
#endif
#ifdef VMDSURF   
      case AtomRep::SURF:
        draw_surface(framepos,
          (int)atomRep->get_data(AtomRep::BONDRES),    // draw wireframe
          atomRep->get_data(AtomRep::SPHERERAD)); // probe radius
        break;
#endif
#ifdef VMDQUICKSURF   
      case AtomRep::QUICKSURF:
        draw_quicksurf(framepos,
          int(atomRep->get_data(AtomRep::BONDRES)), // quality level
          atomRep->get_data(AtomRep::SPHERERAD),    // sphere radius scale
          atomRep->get_data(AtomRep::BONDRAD),      // density isovalue
          atomRep->get_data(AtomRep::GRIDSPACING)); // grid spacing
        break;
#endif

      case AtomRep::VOLSLICE:
        draw_volslice((int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::SPHERERAD),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS),
          (int)atomRep->get_data(AtomRep::BONDRES));
        break;

      case AtomRep::FIELDLINES:
        // XXX use environment variables to control new parameters until
        // we have a chance to update the GUI code...
        draw_volume_field_lines((int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::FIELDLINESEEDUSEGRID),
          (getenv("VMDFIELDLINEMAXSEEDS")) ?  atoi(getenv("VMDFIELDLINEMAXSEEDS")) : 50000,
          atomRep->get_data(AtomRep::SPHERERAD),
          atomRep->get_data(AtomRep::FIELDLINEDELTA),
          atomRep->get_data(AtomRep::BONDRAD),
          atomRep->get_data(AtomRep::BONDRES),
          atomRep->get_data(AtomRep::FIELDLINESTYLE),
          (getenv("VMDFIELDLINETUBERES")) ? atoi(getenv("VMDFIELDLINETUBERES")) : 12,
          (float) atomRep->get_data(AtomRep::LINETHICKNESS));
        break;

      case AtomRep::ISOSURFACE:   
        draw_isosurface((int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::SPHERERAD),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS),
          (int)atomRep->get_data(AtomRep::BONDRES),
          (int)atomRep->get_data(AtomRep::ISOSTEPSIZE),
          (int)atomRep->get_data(AtomRep::ISOLINETHICKNESS));
        break;

      case AtomRep::ORBITAL:   
        draw_orbital(
#if 1
                     (getenv("VMDMODENSITY") != NULL),
#else
                     0,
#endif
                     (int)atomRep->get_data(AtomRep::WAVEFNCTYPE),
                     (int)atomRep->get_data(AtomRep::WAVEFNCSPIN),
                     (int)atomRep->get_data(AtomRep::WAVEFNCEXCITATION),
                     (int)atomRep->get_data(AtomRep::SPHERERES),
                     atomRep->get_data(AtomRep::SPHERERAD),
                     (int)atomRep->get_data(AtomRep::LINETHICKNESS),
                     (int)atomRep->get_data(AtomRep::BONDRES),
                     atomRep->get_data(AtomRep::GRIDSPACING),
                     (int)atomRep->get_data(AtomRep::ISOSTEPSIZE),
                     (int)atomRep->get_data(AtomRep::ISOLINETHICKNESS));
        break;

      case AtomRep::BEADS:
        draw_residue_beads(framepos,
          (int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::SPHERERAD));
        break;

      case AtomRep::DOTTED:       
        draw_dotted_spheres(framepos, 
          atomRep->get_data(AtomRep::SPHERERAD),
          (int)atomRep->get_data(AtomRep::SPHERERES)); 
        place_picks(framepos);
        break;

      case AtomRep::SOLVENT:
        draw_dot_surface(framepos,
          atomRep->get_data(AtomRep::SPHERERAD),
          (int)atomRep->get_data(AtomRep::SPHERERES),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS) - 1); // method
        place_picks(framepos);
        break;

#ifdef VMDLATTICECUBES
      case AtomRep::LATTICECUBES:          
        draw_solid_cubes(framepos, atomRep->get_data(AtomRep::SPHERERAD));
        place_picks(framepos);
        break;
#endif

      default:
        msgErr << "Illegal atom representation in DrawMolecule." << sendmsg;
    }
  } else {
    // do reps that don't require a current frame
    switch (atomRep->method()) {
      case AtomRep::VOLSLICE:
        draw_volslice((int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::SPHERERAD),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS),
          (int)atomRep->get_data(AtomRep::BONDRES));
        break;

      case AtomRep::ISOSURFACE:   
        draw_isosurface((int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::SPHERERAD),
          (int)atomRep->get_data(AtomRep::LINETHICKNESS),
          (int)atomRep->get_data(AtomRep::BONDRES),
          (int)atomRep->get_data(AtomRep::ISOSTEPSIZE),
          (int)atomRep->get_data(AtomRep::ISOLINETHICKNESS));
        break;

      case AtomRep::FIELDLINES:
        // XXX use environment variables to control new parameters until
        // we have a chance to update the GUI code...
        draw_volume_field_lines((int)atomRep->get_data(AtomRep::SPHERERES),
          atomRep->get_data(AtomRep::FIELDLINESEEDUSEGRID),
          (getenv("VMDFIELDLINEMAXSEEDS")) ?  atoi(getenv("VMDFIELDLINEMAXSEEDS")) : 50000,
          atomRep->get_data(AtomRep::SPHERERAD),
          atomRep->get_data(AtomRep::FIELDLINEDELTA),
          atomRep->get_data(AtomRep::BONDRAD),
          atomRep->get_data(AtomRep::BONDRES),
          atomRep->get_data(AtomRep::FIELDLINESTYLE),
          (getenv("VMDFIELDLINETUBERES")) ? atoi(getenv("VMDFIELDLINETUBERES")) : 12,
          (float) atomRep->get_data(AtomRep::LINETHICKNESS));
        break;
    }
  }
}


// enable pick points for all atoms that are turned on
void DrawMolItem::place_picks(float *pos) {
  DispCmdPickPointArray cmdPickPointArray;

  // optimize pick point generation by taking advantage of
  // the firstsel/lastsel indices for the active selection
  int selseglen = atomSel->lastsel-atomSel->firstsel+1;
  cmdPickPointArray.putdata(selseglen, atomSel->selected, atomSel->firstsel,
                            &atomSel->on[atomSel->firstsel], 
                            pos + 3L*atomSel->firstsel, cmdList);
}


//////////////////////////////// protected virtual routines

void DrawMolItem::do_color_changed(int ccat) {
  // check Display and Axes categories, since volumetric and other
  // representations that uses these colors will need regeneration
  if ((ccat == scene->category_index("Display")) ||
      (ccat == scene->category_index("Axes"))) {
    needRegenerate |= COL_REGEN; // recalc colors and redraw
  }

  // check the atom category ... see if it's for the current coloring method
  if (atomColor && atomColor->current_color_use(ccat)) {
    change_color(atomColor);    // recalc atom colors and redraw
  }
}

void DrawMolItem::do_color_rgb_changed(int color) {
  int ccat;

  // check Display and Axes categories, since volumetric and other
  // representations that uses these colors will need regeneration
  ccat = scene->category_index("Display");
  if ((color == scene->category_item_value(ccat, "Background")) ||
      (color == scene->category_item_value(ccat, "Foreground")))
    needRegenerate |= COL_REGEN; // recalc colors and redraw

  ccat = scene->category_index("Axes");
  if ((color == scene->category_item_value(ccat, "X")) ||
      (color == scene->category_item_value(ccat, "Y")) ||
      (color == scene->category_item_value(ccat, "Z")) ||
      (color == scene->category_item_value(ccat, "Origin")) ||
      (color == scene->category_item_value(ccat, "Labels")))
    needRegenerate |= COL_REGEN; // recalc colors and redraw

  // check all atom colors 
  const int *colors = atomColor->color;
  for (int i=0; i<mol->nAtoms; i++) {
    if (colors[i] == color) {
      change_color(atomColor);   // recalc atom colors and redraw
      break;
    }
  }
}

void DrawMolItem::do_color_scale_changed() {
  if (atomColor->uses_colorscale()) {
    atomColor->find(mol);
    needRegenerate |= COL_REGEN; // recalc atom colors and redraw
  }
}
  
//////////////////////////////// public routines 

// change the atom coloring method.  Return success.
int DrawMolItem::change_color(AtomColor *ac) {
  if (ac) {
    *atomColor = *ac;
    needRegenerate |= COL_REGEN;
    return TRUE;
  } else 
    return FALSE;
}

// return which representation this is, i.e. an index from 0 ... # reps - 1
int DrawMolItem::representation_index(void) {
  int totalreps = mol->components();
  for (int i=0; i < totalreps; i++) 
    if (mol->component(i) == this)
      return i;

  msgErr << "Unknown molecular representation found." << sendmsg;
  return 0; // couldn't find anything that made sense.
}

// prepare for drawing ... do any updates needed right before draw.
void DrawMolItem::prepare() {
  // if the rep is off, do nothing; leave the state flags alone and
  // do all required actions when we're turned back on again.
  if (!displayed()) return;

  // force throb atom colors to update, recoloring affected geometry 
  if (atomColor->method() == AtomColor::THROB) { 
    needRegenerate |= COL_REGEN; // recalc atom colors and redraw    
  }   

  // update periodic image display
  if ((update_pbc || update_ts) && cmdList->pbc) {
    update_pbc_transformations();
    update_pbc = 0;
    need_matrix_recalc(); // sets the _needUpdate flag
  }

  // update instance image display
  if ((update_instances || update_ts) && cmdList->instanceset) {
    update_instance_transformations();
    update_instances = 0;
    need_matrix_recalc(); // sets the _needUpdate flag
  }

  // update secondary structure
  if (update_ss) {
    if (atomRep->method() == AtomRep::STRUCTURE ||
        atomRep->method() == AtomRep::NEWCARTOON)
      needRegenerate |= MOL_REGEN;

    if (atomColor->method() == AtomColor::STRUCTURE)
      needRegenerate |= COL_REGEN;

    update_ss = 0;
  }

  // update timestep
  if (update_ts) {
    // If we're drawing the current frame, treat the same as MOL_REGEN.
    // Otherwise, we're just using cached geometry and there's nothing to do.
    if (!strcmp(framesel, "now")) {
      needRegenerate |= MOL_REGEN;

      // force timestep color to update, redraw or recolor affected geometry
      // XXX for the POS[XYZ] coloring methods, the user will need to 
      // select "update color every frame" in order to get time-varying color
      switch (atomColor->method()) {
        case AtomColor::USER:
        case AtomColor::USER2:
        case AtomColor::USER3:
        case AtomColor::USER4:
        case AtomColor::PHYSICALTIME:
        case AtomColor::TIMESTEP:
        case AtomColor::VELOCITY:
          needRegenerate |= COL_REGEN; // recalc atom colors and redraw
          break;
      }   
    }

    update_ts = 0; // update complete
  }

  // new trajectory frames, update reps that draw multiple frames
  if (update_traj) {
    // If we're drawing multiple frames, wait for any molecule I/O to finish
    // before recalculating the entire set of geometry.
    if (strcmp(framesel, "now")) {  // we have a frame selection...
      // XXX ick - cast to Molecule so that we can access Molecule method
      Molecule *m = (Molecule *)mol; 
      if (!m->file_in_progress()) { // ...and file I/O is done.
        needRegenerate |= MOL_REGEN;
        update_traj = 0; // update complete
      }
    }
  }

  create_cmdlist();
}

void DrawMolItem::update_pbc_transformations() {
  const Timestep *ts = mol->current();
  if (!ts) return;

  cmdList->transX.identity();
  cmdList->transY.identity();
  cmdList->transZ.identity();
  ts->get_transforms(cmdList->transX, cmdList->transY, cmdList->transZ);
  cmdList->transXinv = cmdList->transX;
  cmdList->transYinv = cmdList->transY;
  cmdList->transZinv = cmdList->transZ;
  cmdList->transXinv.inverse();
  cmdList->transYinv.inverse();
  cmdList->transZinv.inverse();
}


void DrawMolItem::update_instance_transformations() {
  cmdList->instances.clear();
  int txcnt = mol->instances.num();
#if 0
  printf("drawmolitem::update_instance_trans(): cnt %d\n", txcnt);
#endif
  int i;
  for (i=0; i<txcnt; i++) {
    cmdList->instances.append(mol->instances[i]);
  }
}


// change the atom rep method.  Return success.
int DrawMolItem::change_rep(AtomRep *ar) {
  if (!ar) return FALSE;  // XX does this actually occur?

#ifdef VMDMSMS
  // If we're changing from MSMS to something else, free MSMS memory
  if (atomRep->method() == AtomRep::MSMS && ar->method() != AtomRep::MSMS)
    msms.clear();
#endif

#ifdef VMDNANOSHAPER
  // If we're changing from NanoShaper to something else, free NanoShaper memory
  if (atomRep->method() == AtomRep::NANOSHAPER && ar->method() != AtomRep::NANOSHAPER)
    nanoshaper.clear();
#endif

#ifdef VMDSURF
  // If we're changing from Surf to something else, free Surf memory
  if (atomRep->method() == AtomRep::SURF && ar->method() != AtomRep::SURF)
    surf.clear();
#endif

  // If we're changing from Orbital to something else, free the grid
  if (atomRep->method() == AtomRep::ORBITAL && ar->method() != AtomRep::ORBITAL) {
    waveftype = -1;
    wavefspin = -1;
    wavefexcitation = -1;
    gridorbid=-1;
    orbgridspacing = -1.0f; 
    delete orbvol;
    orbvol=NULL;
  }   

  // Free tubearray if changing from tube or cartoon to something else.
  // We need to check for cartoon because cartoon uses draw_tube to draw
  // its coils. 
  if ((atomRep->method() == AtomRep::TUBE ||
       atomRep->method() == AtomRep::STRUCTURE) && 
      (ar->method() != AtomRep::TUBE &&
       ar->method() != AtomRep::STRUCTURE)) {
    if (tubearray) {
      for (int i=0; i<tubearray->num(); i++) 
        delete (*tubearray)[i];
      delete tubearray;
      tubearray = NULL;
    }
  }
    
  *atomRep = *ar;
  needRegenerate |= REP_REGEN;
  return TRUE;
}


// change the atom selection method.  Return success.
int DrawMolItem::change_sel(const char *cmdStr) {
  if (!cmdStr) return TRUE; // no problem
  if (atomSel->change(cmdStr, mol) == AtomSel::NO_PARSE) return FALSE;
  mol->notify(); // _after_ changing atomSel, tell molecule to update
                 // geometry monitors since they need to be switched off
                 // if they point to atoms that are no longer on.
  needRegenerate |= SEL_REGEN;
  return TRUE;
}

// force a recalculation
void DrawMolItem::force_recalc(int r) {
  needRegenerate |= r;
}

// return whether the Nth atom is displayed in this representation
int DrawMolItem::atom_displayed(int n) {
  return (atomSel != NULL && mol != NULL && 
          n >= 0 && n < mol->nAtoms && atomSel->on[n]);
}

//////////////////////////////// drawing rep routines
void DrawMolItem::draw_lines(float *framepos, int thickness, float cutoff) {
  update_lookups(atomColor, atomSel, colorlookups); // update line color table
  int *nbonds = NULL;
  int *bondlists = NULL;
  if (cutoff > 0) {
    nbonds = new int[mol->nAtoms];
    memset(nbonds, 0, mol->nAtoms*sizeof(int));
    bondlists = new int[MAXATOMBONDS * mol->nAtoms];
    GridSearchPair *pairlist = vmd_gridsearch1(framepos, mol->nAtoms, atomSel->on, 
        cutoff, 0, mol->nAtoms * 27L);
    GridSearchPair *p, *tmp;
    for (p=pairlist; p != NULL; p=tmp) {
      MolAtom *atom1 = mol->atom(p->ind1);
      MolAtom *atom2 = mol->atom(p->ind2);
  
      // don't bond atoms that aren't part of the same conformation    
      // or that aren't in the all-conformations part of the structure 
      if ((atom1->altlocindex != atom2->altlocindex) &&
          ((mol->altlocNames.name(atom1->altlocindex)[0] != '\0') &&
          (mol->altlocNames.name(atom2->altlocindex)[0] != '\0'))) {
        tmp = p->next;
        free(p);
        continue;
      }
      // Prevent hydrogens from bonding with each other.
      // Use atomType info derived during initial molecule analysis for speed.
      if (atom1->atomType == ATOMHYDROGEN &&
          atom2->atomType == ATOMHYDROGEN) {
        tmp = p->next;
        free(p);
        continue;
      }
      bondlists[p->ind1 * MAXATOMBONDS + nbonds[p->ind1]] = p->ind2;
      bondlists[p->ind2 * MAXATOMBONDS + nbonds[p->ind2]] = p->ind1;
      nbonds[p->ind1]++;
      nbonds[p->ind2]++;
      tmp = p->next;
      free(p);
      continue;
    }
  }

  sprintf(commentBuffer, "MoleculeID: %d ReprID: %d Beginning Lines",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  // turn off material characteristics, set line style
  append(DMATERIALOFF);
  cmdLineType.putdata(SOLIDLINE, cmdList);
  cmdLineWidth.putdata(thickness, cmdList);

  for (int i=0; i<MAXCOLORS; i++) {
    const ColorLookup &cl = colorlookups[i];

    if (cl.num == 0) continue; // skip if no bonds of this color
    
    cmdColorIndex.putdata(i, cmdList); // set color for these half-bonds
  
    ResizeArray<float> verts;

    // maintain total vert count for max OpenGL buffer size management
    int totalverts=0;

    // loop over all half-bonds to be drawn in this color
    for (int j=0; j<cl.num; j++) { 
      const int id = cl.idlist[j];
      float *fp1 = framepos + 3L*id;
      const MolAtom *a1 = mol->atom(id);
      int bondsdrawn = 0;
      
      // draw half-bond to each bonded, displayed partner
      // NOTE: bond mid-points are calculated TWICE, once for each atom.
      // Why? So each atom is drawn entirely, including all it's half-bonds,
      // after a single set-color command, instead of having a set-color
      // command for each bond.  This reduces total graphics state changes
      // considerably, at the cost of extra calculation here.
      int n = cutoff > 0 ? nbonds[id] : a1->bonds;
      for (int k=0; k < n; k++) {
        int a2n = cutoff > 0 ? bondlists[MAXATOMBONDS*id + k] : a1->bondTo[k];
        if (atomSel->on[a2n]) {    // bonded atom displayed?
          float *fp2 = framepos + 3L*a2n;
          if (atomColor->color[a2n] == i) {
            // same color, so just draw a whole bond, but only if the atom
            // id of the other atom is higher to avoid drawing it twice.
            if (a2n > id) {
              verts.append3(&fp1[0]);
              verts.append3(&fp2[0]);
              totalverts+=2;
            }
          } else {
            float mid[3];
            mid[0] = 0.5f * (fp1[0] + fp2[0]); 
            mid[1] = 0.5f * (fp1[1] + fp2[1]); 
            mid[2] = 0.5f * (fp1[2] + fp2[2]); 
            verts.append3(&fp1[0]);
            verts.append3(&mid[0]);
            totalverts+=2;
          }
          bondsdrawn++;	               // increment counter of bonds to atom i
        }
      }
     
      // if atom is all alone (no bonds drawn) draw a point 
      if (!bondsdrawn) {  
        DispCmdPoint cmdPoint;
        cmdPoint.putdata(fp1, cmdList);
      }

      // Ensure we don't generate a vertex buffer so large that 
      // we'd crash from internal integer overflows in renderer 
      // subclasses or back-end renderers
      if (totalverts > VMDMAXVERTEXBUFSZ) {
        DispCmdLineArray cmdLineArray;
        cmdLineArray.putdata(&verts[0], verts.num()/6, cmdList);
        verts.clear(); // clear vertex buffer
        totalverts=0;  // reset counter
      }
    }

    if (verts.num()) {
      DispCmdLineArray cmdLineArray;
      cmdLineArray.putdata(&verts[0], verts.num()/6, cmdList);
    }
  }
  if (cutoff > 0) {
    delete [] nbonds;
    delete [] bondlists;
  }
}


// draw all lattice site particles as cubes
// radius is normally set to half of lattice site cube side length dimension
void DrawMolItem::draw_solid_cubes(float * framepos, float radscale) {
  int i;

  // set cube type and shading characteristics
  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning Cube",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
  append(DMATERIALON); // turn on lighting

  // maintain total vert count for max OpenGL buffer size management
  int totalverts=0;

  // draw cubes using cube array primitive
  if (radscale > 0) {           // don't draw zero scaled cubes
    long ind = 0;
    ResizeArray<float> centers;
    ResizeArray<float> radii;
    ResizeArray<float> colors;
    const float *radius = mol->radius();

    ind = atomSel->firstsel * 3;    
    for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
      // draw a cube for each selected lattice site particle
      if (atomSel->on[i]) {
        totalverts++;
        float *fp = framepos + ind;
        const float *cp; 

        centers.append3(&fp[0]);
        radii.append(radius[i]*radscale);
     
        cp = scene->color_value(atomColor->color[i]);
        colors.append3(&cp[0]);
      }
      ind += 3;

      // Ensure we don't generate a vertex buffer so large that 
      // we'd crash from internal integer overflows in renderer 
      // subclasses or back-end renderers
      if (totalverts > VMDMAXVERTEXBUFSZ) {
        cmdCubeArray.putdata((float *) &centers[0], 
                             (float *) &radii[0], 
                             (float *) &colors[0], 
                             radii.num(), 
                             cmdList);

        centers.clear(); // clear vertices
        radii.clear();   // clear vertices
        colors.clear();  // clear vertices
        totalverts=0;    // reset counter
      }
    }
   
    if (radii.num() > 0) {
      cmdCubeArray.putdata((float *) &centers[0], 
                           (float *) &radii[0], 
                           (float *) &colors[0], 
                           radii.num(), 
                           cmdList);
    }
  }
}


// draw all atoms as spheres 
// radius is set to vdw radius * radscale + fixrad
// This allows same code to be used for VDW, CPK, Licorice, etc.
void DrawMolItem::draw_solid_spheres(float * framepos, int res, 
                                     float radscale, float fixrad) {
  int i;

  // set sphere type, resolution, and shading characteristics
  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning VDW",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
  append(DMATERIALON); // turn on lighting
  cmdSphres.putdata(res, cmdList);
  cmdSphtype.putdata(SOLIDSPHERE, cmdList);

  const char *modulatefield = NULL; // XXX this needs to become a parameter
  const float *modulatedata = NULL; // data field to use for width modulation
  int  modulateoffs  = 0;    // data offset
  int  modulatemult  = 1;    // data multiples

#if 1
  // XXX hack to let users try various stuff
  modulatefield = getenv("VMDMODULATERADIUS");
#endif
  if (modulatefield != NULL) {
    if (!strcmp(modulatefield, "user")) {
      modulatedata = mol->current()->user;
      // XXX user field can be NULL on some timesteps
    } else if (!strcmp(modulatefield, "user2")) {
      modulatedata = mol->current()->user2;
      // XXX user2 field can be NULL on some timesteps
    } else if (!strcmp(modulatefield, "user3")) {
      modulatedata = mol->current()->user3;
      // XXX user3 field can be NULL on some timesteps
    } else if (!strcmp(modulatefield, "user4")) {
      modulatedata = mol->current()->user4;
      // XXX user1 field can be NULL on some timesteps
    } else if (!strcmp(modulatefield, "vx")) {
      modulatedata = mol->current()->vel;
      // XXX vx field can be NULL on some timesteps
      modulateoffs = 0; modulatemult = 3;
    } else if (!strcmp(modulatefield, "vy")) {
      modulatedata = mol->current()->vel;
      // XXX vy field can be NULL on some timesteps
      modulateoffs = 1; modulatemult = 3;
    } else if (!strcmp(modulatefield, "vz")) {
      modulatedata = mol->current()->vel;
      // XXX vz field can be NULL on some timesteps
      modulateoffs = 2; modulatemult = 3;
    } else {
      modulatedata = mol->extraflt.data(modulatefield);
    } 
  }

  // maintain total vert count for max OpenGL buffer size management
  int totalverts=0;

  // draw spheres using new sphere array primitive
  if ((radscale + fixrad) > 0) {           // don't draw zero scaled spheres
    long ind = 0;
    ResizeArray<float> centers;
    ResizeArray<float> radii;
    ResizeArray<float> colors;
    const float *radius = mol->radius();

    ind = atomSel->firstsel * 3;    
    for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
      // draw a sphere for each selected atom
      if (atomSel->on[i]) {
        totalverts++;
        centers.append3(framepos + ind);
    
        float my_rad = 1.0f;
        if (modulatedata != NULL) {
          my_rad = modulatedata[i*modulatemult + modulateoffs];
          if (my_rad <= 0.0f)
            my_rad = 1.0f;
        }
        radii.append(radius[i]*radscale*my_rad + fixrad);
        colors.append3(scene->color_value(atomColor->color[i]));
      }
      ind += 3;

      // Ensure we don't generate a vertex buffer so large that 
      // we'd crash from internal integer overflows in renderer 
      // subclasses or back-end renderers
      if (totalverts > VMDMAXVERTEXBUFSZ) {
        cmdSphereArray.putdata((float *) &centers[0], 
                               (float *) &radii[0], 
                               (float *) &colors[0], 
                               radii.num(), 
                               res,
                               cmdList);

        centers.clear(); // clear vertices
        radii.clear();   // clear vertices
        colors.clear();  // clear vertices
        totalverts=0;    // reset counter
      }
    }
   
    if (radii.num() > 0) {
      cmdSphereArray.putdata((float *) &centers[0], 
                             (float *) &radii[0], 
                             (float *) &colors[0], 
                             radii.num(), 
                             res,
                             cmdList);
    }
  }
}


// draw residues as beads
// radius is set to vdw radius * radscale + fixrad
// This allows same code to be used for VDW, CPK, Licorice, etc.
void DrawMolItem::draw_residue_beads(float * framepos, int sres, float radscale) {
  int i, resid, numres;

  // set sphere type, resolution, and shading characteristics
  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning beads",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
  append(DMATERIALON); // turn on lighting
  cmdSphres.putdata(sres, cmdList);
  cmdSphtype.putdata(SOLIDSPHERE, cmdList);

  // draw spheres using new sphere array primitive
  ResizeArray<float> centers;
  ResizeArray<float> radii;
  ResizeArray<float> colors;

  // draw a bead for each residue
  numres = mol->residueList.num();
  for (resid=0; resid<numres; resid++) {
    float com[3] = {0.0, 0.0, 0.0};
    const ResizeArray<int> &atoms = mol->residueList[resid]->atoms;
    int numatoms = atoms.num();
    int oncount = 0;
    int pickindex=-1;
    
    // find COM for residue
    for (i=0; i<numatoms; i++) {
      int idx = atoms[i];
      if (atomSel->on[idx]) {
        oncount++;
        vec_add(com, com, framepos + 3L*idx);
      }
    }

    if (oncount < 1)
      continue; // exit if there weren't any atoms

    vec_scale(com, 1.0f / (float) oncount, com);

    // find radius of bounding sphere and save last atom index for color
    int atomcolorindex=0; // initialize, to please compilers
    float boundradsq = 0.0f;
#ifdef BEADELLIPSOID
    float avdist = 0.0f;
    float majoraxis[3] = {0.0, 0.0, 0.0};
#endif
    for (i=0; i<numatoms; i++) {
      int idx = atoms[i];
      if (atomSel->on[idx]) {
        float tmpdist[3];
        atomcolorindex = idx;
        vec_sub(tmpdist, com, framepos + 3L*idx);
        float distsq = dot_prod(tmpdist, tmpdist);
        if (distsq > boundradsq) {
#ifdef BEADELLIPSOID
          // XXX if we want to draw an ellipsoid rather than a sphere,
          // we could keep track of the direction from the COM to the 
          // most distant atom so far, and use that as the major axis of
          // the ellipsoid, and use the average distance as the minor axis
          // for a rotationally symmetric ellipsoid.  
          vec_copy(majoraxis, tmpdist);
          avdist += sqrtf(distsq);
#endif
          boundradsq = distsq;
        }
      }
    }

#ifdef BEADELLIPSOID
    avdist /= (float) oncount;
    float cep1[3], cep2[3];
    vec_copy(cep1, majoraxis);
    vec_scale(cep2, -1, majoraxis);
    vec_add(cep1, cep1, com);
    vec_add(cep2, cep2, com);

    cmdColorIndex.putdata(atomColor->color[atomcolorindex], cmdList);

    // we'd draw a real ellipsoid here rather than a cylinder, but
    // drawing a cylinder is a pretty representative test of the idea.
    cmdCylinder.putdata(cep1, cep2, avdist, 8, 0, cmdList);
#endif

    centers.append3(&com[0]);
    radii.append(radscale * (sqrtf(boundradsq) + 1.0f));

    const float *cp = scene->color_value(atomColor->color[atomcolorindex]);
    colors.append3(&cp[0]);

    // Add a pick point
    // for proteins use the CA atom
    if (pickindex < 0)
      pickindex = mol->find_atom_in_residue("CA", resid); 

    // for nucleic acids, use the C3', C3*, or P atom
    if (pickindex < 0)
      pickindex = mol->find_atom_in_residue("C3'", resid); 
    if (pickindex < 0)
      pickindex = mol->find_atom_in_residue("C3*", resid); 
    if (pickindex < 0)
      pickindex = mol->find_atom_in_residue("P", resid);

    // if nothing better is found, use the first atom
    if (pickindex < 0)
      pickindex = atoms[0];

    pickPoint.putdata(com, pickindex, cmdList);
  }

#if !defined(BEADELLIPSOID)
  if (radii.num() > 0) {
    cmdSphereArray.putdata((float *) &centers[0],
                           (float *) &radii[0],
                           (float *) &colors[0],
                           radii.num(),
                           sres,
                           cmdList);
  }
#endif
}



// draw all atoms as dotted spheres of vdw radius
void DrawMolItem::draw_dotted_spheres(float * framepos, float srad, int sres) {
  float radscale;
  int i;
  const float *radius = mol->radius();

  // set sphere type, resolution, and shading characteristics
  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning Dotted VDW",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
  append(DMATERIALOFF); // turn off lighting
  cmdSphres.putdata(sres, cmdList);
  cmdSphtype.putdata(POINTSPHERE, cmdList);

  // XXX we should teach this code to sort by color and by radius, 
  //     which would allow us to avoid dynamic scaling of the spheres at
  //     display time.
  radscale = srad;
  if (radscale > 0) {                  // don't draw zero scaled spheres
    int lastcolor = -1;
    for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
      // draw a sphere for each selected atom
      if (atomSel->on[i]) {
	if (lastcolor != atomColor->color[i]) {   // set color
	  lastcolor = atomColor->color[i];
	  cmdColorIndex.putdata(lastcolor, cmdList);
	}
      
	cmdSphere.putdata(framepos+3L*i, radius[i]*radscale,cmdList);
      }
    }
  }
}


// draws each atom as a dot
void DrawMolItem::draw_points(float *framepos, float pointsize) {
  sprintf(commentBuffer, "MoleculeID: %d ReprID: %d Beginning Points",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
  append(DMATERIALOFF);

  if (atomSel->selected > 0) {
    // optimize pick point generation by taking advantage of
    // the firstsel/lastsel indices for the active selection
    int selseglen = atomSel->lastsel-atomSel->firstsel+1;

    // draw spheres and pick points
    cmdPointArray.putdata(framepos + 3L*atomSel->firstsel,
                          atomColor->color + atomSel->firstsel,
                          scene,
                          pointsize,
                          selseglen,
                          atomSel->on + atomSel->firstsel,
                          atomSel->selected,
                          cmdList);
  }
  append(DMATERIALON);

}


void DrawMolItem::draw_cpk_licorice(float *framepos, int cpk, float brad, int bres, float srad, int sres, int linethickness, float cutoff) {
  MolAtom *a1;
  int i, j, a2n;
  float radscale, fixrad; 
  int lastcolor = -1;
  int use_bonds = TRUE;

  if (cpk) { 
        brad *= 0.25f;
    radscale = srad * 0.25f; // scaled VDW rad
      fixrad = 0.0; // no fixed radius
  } else {
    radscale = 0.0;  // no VDW scaling
      fixrad = brad; // used fixed radius spheres only

    // special case -- if there is no thickness, call the wireframe
    if (brad == 0) {
      draw_lines(framepos, linethickness, cutoff);
      return;
    }
  }

  if (bres <= 2 || brad < 0.01) { 
    use_bonds = FALSE; // draw bonds using lines
  }

  int *nbonds = NULL;
  int *bondlists = NULL;
  if (cutoff > 0) {
    nbonds = new int[mol->nAtoms];
    memset(nbonds, 0, mol->nAtoms*sizeof(int));
    bondlists = new int[MAXATOMBONDS * mol->nAtoms];
    GridSearchPair *pairlist = vmd_gridsearch1(framepos, mol->nAtoms, atomSel->on, 
        cutoff, 0, mol->nAtoms * 27L);
    GridSearchPair *p, *tmp;
    for (p=pairlist; p != NULL; p=tmp) {
      MolAtom *atom1 = mol->atom(p->ind1);
      MolAtom *atom2 = mol->atom(p->ind2);
  
      // don't bond atoms that aren't part of the same conformation    
      // or that aren't in the all-conformations part of the structure 
      if ((atom1->altlocindex != atom2->altlocindex) &&
          ((mol->altlocNames.name(atom1->altlocindex)[0] != '\0') &&
          (mol->altlocNames.name(atom2->altlocindex)[0] != '\0'))) {
        tmp = p->next;
        free(p);
        continue;
      }
      // Prevent hydrogens from bonding with each other.
      // Use atomType info derived during initial molecule analysis for speed.
      if (atom1->atomType == ATOMHYDROGEN &&
          atom2->atomType == ATOMHYDROGEN) {
        tmp = p->next;
        free(p);
        continue;
      }
      bondlists[p->ind1 * MAXATOMBONDS + nbonds[p->ind1]] = p->ind2;
      bondlists[p->ind2 * MAXATOMBONDS + nbonds[p->ind2]] = p->ind1;
      nbonds[p->ind1]++;
      nbonds[p->ind2]++;
      tmp = p->next;
      free(p);
      continue;
    }
  }
  sprintf (commentBuffer,"MoleculeID: %d ReprID: %d Beginning CPK",
	 mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  append(DMATERIALON);
 
  // only draw spheres if either radscale or fixrad is nonzero 
  if ((radscale + fixrad) > 0) {   
    draw_solid_spheres(framepos, sres, radscale, fixrad);
  }

  if (use_bonds) {
    // draw bonds between atoms with cylinders
    for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
      // for each atom, draw half-bond to other displayed atoms
      if (atomSel->on[i]) {
        float mid[3], *fp1, *fp2;
        fp1 = framepos + 3L*i; // position of atom 'i'
        a1 = mol->atom(i);

        if (lastcolor != atomColor->color[i]) {
          lastcolor = atomColor->color[i];
          cmdColorIndex.putdata(lastcolor, cmdList);
        }

        // draw half-bond to each bonded, displayed partner
        int n = cutoff > 0 ? nbonds[i] : a1->bonds;
	for (j=0; j < n; j++) {
	  a2n = cutoff > 0 ? bondlists[MAXATOMBONDS*i + j] : a1->bondTo[j];
	  if (atomSel->on[a2n]) {      // bonded atom 'a2n' displayed?
	    fp2 = framepos + 3L*a2n;   // position of atom 'a2n'
            // find the bond midpoint 'mid' between atoms 'i' and 'a2n'
            mid[0] = 0.5f * (fp1[0] + fp2[0]);
            mid[1] = 0.5f * (fp1[1] + fp2[1]);
            mid[2] = 0.5f * (fp1[2] + fp2[2]);

            cmdCylinder.putdata(fp1, mid, brad, bres, 0, cmdList);
	  }
	}
      }
    }
  }
  if (cutoff > 0) {
    delete [] nbonds;
    delete [] bondlists;
  }
}



// This draws a cylindrical bond from atom 'i' to atom 'k', but
// only if both are selected.  Each end of the cylinder is colored
// appropriately.  If there are bonds connected to the ends of this
// i-k bond, then the i-k bond is extended so that the bonds have
// no gap between them
void DrawMolItem::draw_bonds(float *framepos, float brad, int bres, float cutoff) {
  MolAtom *a1, *a2;
  int g=0, h=0, i=0, j=0, k=0, l=0, m=0;
  int lastcolor = -1;
  int use_cyl;
  const float *bondorders = mol->bondorders();

  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning Bonds",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  if (bres <= 2 || brad < 0.01 ) {
    // draw bonds with lines
    use_cyl = FALSE;
    append(DMATERIALOFF);
    cmdLineType.putdata(SOLIDLINE, cmdList);
    cmdLineWidth.putdata(2, cmdList);
  } else {
    use_cyl = TRUE;
    // set up general drawing characteristics for this drawing method
    // turn on material characteristics
    append(DMATERIALON);
  }

  for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
    // Only bonds to 'on' atoms are considered.
    if (atomSel->on[i]) {
      float *p2 = framepos + 3L*i;   // set p2 to atom 'i'
      a1 = mol->atom(i);            // find a selected atom
      float idouble[3], kdouble[3]; 
      float itriple[3], ktriple[3]; 
      const float *bondorderi = bondorders + (i * MAXATOMBONDS);

      for (j=0; j<a1->bonds; j++) {
        k = a1->bondTo[j];          // find a bonded atom that is turned on
        if (k > i && atomSel->on[k]) {
          float *p1, p3[3], *p4, *p5;
          a2 = mol->atom(k);
          p4 = framepos + 3L*k;      // set p4 to atom 'k'

          // find the bond midpoint 'p3' between atoms 'i' and 'k' 
          p3[0] = 0.5f * (p2[0] + p4[0]);
          p3[1] = 0.5f * (p2[1] + p4[1]);
          p3[2] = 0.5f * (p2[2] + p4[2]);

          // We extend the ends of the bond to touch neighboring bonds
          // Look for atom 'm' bonded to 'k' that isn't 'i', to extend to
          p5 = NULL;                // only used if not NULL
          for (l=a2->bonds-1; l>=0; l--) {
            m = a2->bondTo[l];
            if (m != i && atomSel->on[m]) 
              p5 = framepos + 3L*m; // set p5 to atom 'm'
              break;
          }
  
          // Look for atom 'g' bonded to 'i' that isn't 'k', to extend to
          p1 = NULL;                // only used if not NULL
          for (h=a1->bonds-1; h>=0; h--) {
            g = a1->bondTo[h];
            if (g != k && atomSel->on[g]) 
              p1 = framepos + 3L*g; // set p1 to atom 'g'
              break;
          }

          // determine the bond order (-1 (unset), 1, 2, or 3)
          float order = 0;
          if (bondorders != NULL) {
            order = bondorderi[j];
            if (order > 1) {
              int lv;
              for (lv=0; lv<3; lv++) {
                idouble[lv] = p2[lv] + 0.333f * (p4[lv] - p2[lv]);
                kdouble[lv] = p2[lv] + 0.666f * (p4[lv] - p2[lv]);
                itriple[lv] = p2[lv] + 0.450f * (p4[lv] - p2[lv]);
                ktriple[lv] = p2[lv] + 0.550f * (p4[lv] - p2[lv]);
              } 
            }
          }

          // now I can draw the bonds, setting the color for each
          if (lastcolor != atomColor->color[i]) {
            lastcolor = atomColor->color[i];
            cmdColorIndex.putdata(lastcolor, cmdList);
          }
          make_connection(p1, p2, p3, p4, brad, bres, use_cyl);
          if (order > 1) {
            make_connection(NULL, idouble, p3, kdouble, brad * 1.5f, bres, use_cyl);
            if (order > 2)
              make_connection(NULL, itriple, p3, ktriple, brad * 2.0f, bres, use_cyl);
          }

          if (lastcolor != atomColor->color[k]) {
            lastcolor = atomColor->color[k];
            cmdColorIndex.putdata(lastcolor, cmdList);
          }
          make_connection(p2, p3, p4, p5, brad, bres, use_cyl);
          if (order > 1) {
            make_connection(idouble, p3, kdouble, NULL, brad * 1.5f, bres, use_cyl);
            if (order > 2)
              make_connection(itriple, p3, ktriple, NULL, brad * 2.0f, bres, use_cyl);
          }
        } // found k atom
      } // searching along i's bonds
    } // found i
  } // searching each atom
}


// given the current cylinder and the previous and next cylinders,
// find and draw the cylinder (or line) that best fits without making
// an overlap.  prev can be NULL, in which case it acts as if the
// previous cylinder was linear with the one to draw.  Ditto (but
// opposite) for next.
//
// The problem is that if two cylinders touch head to tail,
// but at an angle, there is a gap between them.  This routine
// eliminates the gap by extending the cylinder endpoints.
void DrawMolItem::make_connection(float *prev, float *start, float *end,
				  float *next, float rad, int res, int use_cyl)
{
  if (!start || !end) {
    msgErr << "Trying to make an extended cylinder with NULL end point(s)"
           << sendmsg;
    return;
  }

  if (!use_cyl) {
    cmdLine.putdata(start, end, cmdList);
    return;
  }

  float new_start[3], new_end[3];
  float cthis[3];
  float cnext[3];
  float cprev[3];
  float length;
  float costheta;

  vec_sub(cthis, end, start);    // cthis = end - start, cylinder (b)
  vec_normalize(cthis);

  length = 0.0f;                 // initialize to zero length by default
  if (prev != NULL) {            // previous cylinder (a)
    vec_sub(cprev, start, prev); // cprev = start - prev
    vec_normalize(cprev);

    // Compute length for aligning this cylinder with previous
    costheta = dot_prod(cprev, cthis);

    // does this exceed threshhold to start extension (might speed things up)
    if ((costheta > 0.0f) && (costheta < 0.9999f)) 
      length = rad * sqrtf((1.0f-costheta) / (1.0f+costheta));
  }
  new_start[0] = start[0] - length * cthis[0];
  new_start[1] = start[1] - length * cthis[1];  
  new_start[2] = start[2] - length * cthis[2];

  length = 0.0f;                 // initialize to zero length by default
  if (next != NULL) {            // along the next cylinder (a)
    vec_sub(cnext, next, end);   // cnext = next - end
    vec_normalize(cnext);

    // Now move the ending point to line up with the next cylinder.
    costheta = dot_prod(cthis,cnext);

    // does this exceed threshhold to start extension (might speed things up)
    if ((costheta > 0.0f) && (costheta < 0.9999f)) 
      length = rad * sqrtf((1.0f-costheta) / (1.0f+costheta));
  }
  new_end[0] = end[0] + length * cthis[0];
  new_end[1] = end[1] + length * cthis[1];
  new_end[2] = end[2] + length * cthis[2];

  cmdCylinder.putdata(new_start, new_end, rad, res, 0, cmdList);
}


// Make a spline curve from a set of coordinates.  There are 'num' coords.
// Coords goes from -2 to num+1 and has 3 floats per coord
//    idx goes from -2 to num+1 and has either the atom number or -1
void DrawMolItem::draw_spline_curve(int num, float *coords, int *idx,
                                    int use_cyl, float b_rad, int b_res) {
  float q[4][3];       // spline Q matrix
  float prev[2][3];    // the last two points for the previous spline
  float final[7][3];   // the points of the current approximation
  int last_loop = -10; // what is this value for?
  int loop;

  ResizeArray<float> pickpointcoords;
  ResizeArray<int> pickpointindices;

#if 0
  const char *modulatefield = NULL; // XXX this needs to become a parameter
  const float *modulatedata = NULL; // data field to use for width modulation
  float *modulate = NULL;           // per-control point width values

  // XXX hack to let users try various stuff
  modulatefield = getenv("VMDMODULATERIBBON");
  if (modulatefield != NULL) {
    if (!strcmp(modulatefield, "user")) {
      modulatedata = mol->current()->user;
      // XXX user field can be NULL on some timesteps
    } else {
      modulatedata = mol->extraflt.data(modulatefield);
    } 

    // allocate for the maximum possible per-residue modulation values
    // so we don't have to reallocate for every fragment
    // XXX the modulate array is allocated cleared to zeros so that
    // in the case we get a NULL modulatedata pointer (user field for
    // example, we'll just end up with no modulation.
    modulate = (float *) calloc(1, mol->nResidues * sizeof(float));
  }
#endif

  for (loop=-1; loop<num; loop++) { // go through the array
    // check if we need to do any computations, makes code faster
    // but its a bit more of a headache later on
    if ((idx[loop  ] >=0 && atomSel->on[idx[loop  ]]) ||
        (idx[loop+1] >=0 && atomSel->on[idx[loop+1]])) {
      make_spline_Q_matrix(q, spline_basis, coords+(loop-1)*3);
	 
      // evaluate the interpolation between atom "loop" and
      // "loop+1".  I'll make this in 6 pieces, 7 coords.
      make_spline_interpolation(final[0], 0.0f/6.0f, q);
      make_spline_interpolation(final[1], 1.0f/6.0f, q);
      make_spline_interpolation(final[2], 2.0f/6.0f, q);
      make_spline_interpolation(final[3], 3.0f/6.0f, q);
      make_spline_interpolation(final[4], 4.0f/6.0f, q);
      make_spline_interpolation(final[5], 5.0f/6.0f, q);
      make_spline_interpolation(final[6], 6.0f/6.0f, q);
	 
      // the possible values of 'on' are 0 (off), 1 (on),
      // 2 (draw 1st half), and 3 (draw 2nd half)
      // However, if the 1st half is drawn, the prev atom must
      // be on, and if the 2nd half is drawn, the new one must
      // be on.

      // draw what I need for atom 'loop'
      if (idx[loop] >= 0 &&          // this is a real atom
          (atomSel->on[idx[loop  ]] == 1 ||    // and it is turned on
          (atomSel->on[idx[loop  ]] == 3 && idx[loop+1] >= 0 &&
           atomSel->on[idx[loop+1]]))) {
    
        // here is the trickiness I was talking about
        // I need to see if there was a computation for
        // the first part of this atom
        if (last_loop != loop - 1) {  // no there wasn't
           cmdColorIndex.putdata(atomColor->color[idx[loop]], cmdList);
           make_connection(NULL, final[0], final[1], final[2],
                           b_rad, b_res, use_cyl);
        } else {
          // finish up drawing the previous part of this residue (I couldn't
          // do this before since I didn't know how to angle the final cylinder
          // to get a nice match up to the first of the current cylinders)
          make_connection(prev[0], prev[1], final[0], final[1],
                          b_rad, b_res, use_cyl);
          make_connection(prev[1], final[0], final[1], final[2],
                          b_rad, b_res, use_cyl);
        }
        make_connection(final[0], final[1], final[2], final[3],
                        b_rad, b_res, use_cyl);
        make_connection(final[1], final[2], final[3], final[4],
                        b_rad, b_res, use_cyl);

        // indicate this atom can be picked
        int pidx = 3L * loop;
        pickpointcoords.append3(&coords[pidx]);
        pickpointindices.append(idx[loop]);
      }
	 
      // draw what I can for atom 'loop+1'
      if (idx[loop+1] >= 0 &&
          (atomSel->on[idx[loop+1]] == 1 ||
          (atomSel->on[idx[loop+1]] == 2 && idx[loop] >= 0 &&
           atomSel->on[idx[loop]]))) {
        cmdColorIndex.putdata(atomColor->color[idx[loop+1]], cmdList);
        make_connection(final[2], final[3], final[4], final[5],
                        b_rad, b_res, use_cyl);
        make_connection(final[3], final[4], final[5], final[6],
                        b_rad, b_res, use_cyl);
        last_loop = loop;
      }
      vec_copy(prev[0], final[4]);  // save for the
      vec_copy(prev[1], final[5]);  // next interation
    } /// else nothing to draw
  } // gone down the fragment

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0], 
                           &pickpointcoords[0], cmdList);
  }
}


void DrawMolItem::generate_tubearray() {
  if (tubearray) {
    for (int i=0; i<tubearray->num(); i++) delete (*tubearray)[i];
    delete tubearray;
  }
  tubearray = new ResizeArray<TubeIndexList *>;

  int frag;
  // start with protein
  for (frag = 0; frag < mol->pfragList.num(); frag++) {
    int num = mol->pfragList[frag]->num(); // number of residues
    if (num < 2) continue;
    TubeIndexList *idx = new TubeIndexList;
    // I'll go ahead and pad the beginning and end with two -1's for now.
    // It shouldn't be necessary but that's how draw_spline_curve works.
    idx->append2(-1, -1);
    for (int loop = 0; loop < num; loop++) {
      int res = (*mol->pfragList[frag])[loop];
      // Find the CA in this residue.  If not found, use the first atom
      // in the residue, since there has to be at least one atom.
      int atomnum = mol->find_atom_in_residue("CA", res);
      if (atomnum < 0) {
        atomnum = mol->atom_residue(res)->atoms[0];
      }
      idx->append(atomnum);
    }
    idx->append2(-1, -1);
    tubearray->append(idx);
  }
  // If there were no pfrags, check for a molecule with only CA's, 
  // and draw tubes connecting CA's with consecutive resid's.

  if (mol->pfragList.num() == 0) {
    int num = mol->nAtoms;
    int ca_num = mol->atomNames.typecode("CA"); // for fast lookup
    int last_resid = -10000;
    int resid;
    MolAtom *atm = NULL;
    
    TubeIndexList *idx = NULL;
    for (int i=0; i<=num; i++) {
      if (i != num) {
        atm = mol->atom(i);
        resid = atm->resid;
      } else {
        resid = -1000; // want it be out of range, but not equal to -10000.
      }
      // find a sequential CA
      if (atm->nameindex == ca_num && resid == last_resid + 1) {
        if (idx == NULL) { 
          msgErr << "Internal error in draw_tube for CA atoms: last_resid = "
            << last_resid << " but no first atom was added." << sendmsg;
          msgErr << "Tubes may be incomplete." << sendmsg;
          return;
        }
        idx->append(i);
      } else {
        if (idx) { // were there any points?
          idx->append2(-1, -1);
          tubearray->append(idx);
          idx = NULL;
        }
        if (atm->nameindex == ca_num && i != num) {
          idx = new TubeIndexList;
          idx->append3(-1, -1, i);
        }
      }
      last_resid = resid;
    }
  }

  // Now for nucleic acid backbone
  // This is just like we did for protein, except that instead of looking
  // for CA we look for P, or C5', or C5*.
  for (frag = 0; frag < mol->nfragList.num(); frag++) {
    int num = mol->nfragList[frag]->num(); // number of residues
    if (num < 2) continue;
    TubeIndexList *idx = new TubeIndexList;
    // I'll go ahead and pad the beginning and end with two -1's for now.
    // It shouldn't be necessary.
    idx->append2(-1, -1);
    for (int loop = 0; loop < num; loop++) {
      int res = (*mol->nfragList[frag])[loop];
      // Find the P, C5', or C5* in this residue.  
      // If not found, use the first atom
      // in the residue, since there has to be at least one atom.
      int atomnum = mol->find_atom_in_residue("P", res);
      if (atomnum < 0) {
        atomnum = mol->find_atom_in_residue("C5'", res);
        if (atomnum < 0) {
          atomnum = mol->find_atom_in_residue("C5*", res);
          if (atomnum < 0) { 
            atomnum = mol->atom_residue(res)->atoms[0];
          }
        }
      }
      idx->append(atomnum);
    }
    idx->append2(-1, -1);
    tubearray->append(idx);
  }

  // If there were no nfrags, check for a molecule with only P's, 
  // and draw tubes connecting P's with consecutive resid's.
  // This is duplicated from what we did for CA's - the only line that's
  // changed is the definition of ca_num.

  if (mol->nfragList.num() == 0) {
    int num = mol->nAtoms;
    int ca_num = mol->atomNames.typecode("P"); // for fast lookup
    int last_resid = -10000;
    int resid;
    MolAtom *atm = NULL;
    
    TubeIndexList *idx = NULL;
    for (int i=0; i<=num; i++) {
      if (i != num) {
        atm = mol->atom(i);
        resid = atm->resid;
      } else {
        resid = -1000; // want it be out of range, but not equal to -10000.
      }
      // find a sequential P
      if (atm->nameindex == ca_num && resid == last_resid + 1) {
        if (idx == NULL) { 
          msgErr << "Internal error in draw_tube for P atoms: last_resid = "
            << last_resid << " but no first atom was added." << sendmsg;
          msgErr << "Tubes may be incomplete." << sendmsg;
          return;
        }
        idx->append(i);
      } else {
        if (idx) { // were there any points?
          idx->append2(-1, -1);
          tubearray->append(idx);
          idx = NULL;
        }
        if (atm->nameindex == ca_num && i != num) {
          idx = new TubeIndexList;
          idx->append3(-1, -1, i);
        }
      }
      last_resid = resid;
    }
  }
}


// draw a cubic spline (in this case, using a modified Catmull-Rom basis)
// through the C alpha of the protein and the P of the nucleic acids
void DrawMolItem::draw_tube(float *framepos, float b_rad, int b_res) {
  // If we don't have tubearray yet, we need to generate it.  We don't
  // have to worry about selection or color because that's handled by
  // draw_spline_curve().
  if (!tubearray) {
    generate_tubearray();
  }

  sprintf (commentBuffer,"MoleculeID: %d ReprID: %d Beginning Tube",
           mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  // find out if I'm using lines or cylinders
  int use_cyl = FALSE;
  if (b_res <= 2 || b_rad < 0.01) { // then going to do lines
    append(DMATERIALOFF);
    cmdLineType.putdata(SOLIDLINE, cmdList);
    cmdLineWidth.putdata(2, cmdList);
  } else {
    use_cyl = TRUE;
    append(DMATERIALON);
  }

  for (int i=0; i<tubearray->num(); i++) {
    // XXX copy the current coordinates into a temporary array.
    TubeIndexList &indxlist = *(*tubearray)[i];
    int num = indxlist.num();
    float *coords = new float[3L*num];
    // First two and last two points are always just the third and third to
    // last real control points.
    int firstind = indxlist[2];
    int lastind  = indxlist[num-3];
    memcpy(coords,   framepos+3L*firstind, 3L*sizeof(float));
    memcpy(coords+3, framepos+3L*firstind, 3L*sizeof(float));
    for (int i=2; i<num-2; i++) {
      memcpy(coords+3L*i, framepos+3L*indxlist[i], 3L*sizeof(float));
    }
    memcpy(coords+3L*(num-2), framepos+3L*lastind, 3L*sizeof(float));
    memcpy(coords+3L*(num-1), framepos+3L*lastind, 3L*sizeof(float));
    draw_spline_curve(num-4, coords+6, &(indxlist[0])+2, use_cyl, b_rad, b_res);
    delete [] coords;
  }
}


//  This draws a hydrogen bond from atom 'i' to the hydrogen of atom 'k', only
//  if both are selected and meet the hbond distance and/or angle criteria
//  distance is measured from heavy atom to heavy atom
//  the angle is from heavy atom to hydrogen to heavy atom
//  coloring is by the donor atom color specification
void DrawMolItem::draw_hbonds(float *framepos, float maxangle, int thickness, float maxdist) {
  int i, k;
  float  donortoH[3],Htoacceptor[3];
  float cosmaxangle2 = cosf(maxangle); cosmaxangle2 *= cosmaxangle2; 
  ResizeArray<float> pickpointcoords;
  ResizeArray<int> pickpointindices;

  // protect against ridiculous values which might crash us or
  // run the system out of memory.
  if (maxdist <= 0.1 || atomSel->selected == 0) {
    return;
  }
   
  int *onlist = (int *) calloc(1, mol->nAtoms * sizeof(int)); // clear to zeros
  for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
    if (atomSel->on[i] && mol->atom(i)->atomType != ATOMHYDROGEN)
      onlist[i] = 1;
  } 

  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning Hbonds",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
   
  // set up general drawing characteristics for this drawing method
  append(DMATERIALOFF);
  cmdLineType.putdata(DASHEDLINE, cmdList);
  cmdLineWidth.putdata(thickness, cmdList);

  // XXX the actual code for measuring hbonds doesn't belong here, it should
  //     be moved into Measure.[Ch] where it really belongs.  This file
  //     only implements the rendering interface, and should not be doing the
  //     hard core math, particularly since we want to expose the same
  //     feature via scripting interfaces etc.  Also, having a single
  //     implementation avoids having different bugs in the
  //     long-term.  Too late to do anything about this now, but should be
  //     addressed for the next major version when time allows.

  // XXX This code is similar, but no identical to the code in TclMeasure.C
  //     It would be better written if rather than emitting lines and 
  //     pickpoints piecemeal, if it instead got a complete list of HBonds
  //     and then built a single vertex array instead. 
 
  // loop over all SELECTED atoms, inspite of the fact that non-hbonding
  // atoms may actually be selected, searching for 
  // any pair that are closer than distance, not bonded,
  // and have a hydrogen that forms an acceptable angle.. all said 
  // and done it works pretty well only catching C-H---O hbonds 
  // only when the distance and angle are set rather unrealistic
  GridSearchPair *pairlist = vmd_gridsearch1(framepos, mol->nAtoms, onlist, maxdist, 0, mol->nAtoms * 27);
  GridSearchPair *p, *tmp;
  for (p=pairlist; p != NULL; p=tmp) {
    MolAtom *a1 = mol->atom(p->ind1);
    MolAtom *a2 = mol->atom(p->ind2);

    // ignore if bonded
    if (!a2->bonded(p->ind1)) {
      int b1 = a1->bonds;
      int b2 = a2->bonds;
      float *coor1 = framepos + 3L*p->ind1; 
      float *coor2 = framepos + 3L*p->ind2; 
  
      for (k=0; k < b2; k++) {
        if (mol->atom(a2->bondTo[k])->atomType == ATOMHYDROGEN) {
          float *hydrogen = framepos + 3L*a2->bondTo[k];
	  vec_sub(donortoH,hydrogen,coor2);
	  vec_sub(Htoacceptor,coor1,hydrogen);
          if (angle(donortoH, Htoacceptor)  < maxangle ) {
	    cmdColorIndex.putdata(atomColor->color[p->ind2], cmdList);
	    cmdLine.putdata(coor1,hydrogen, cmdList); // draw line

            // indicate the bonded atoms can be picked
            int pidx = 3L * a2->bondTo[k];
            pickpointcoords.append3(&framepos[pidx]);

            pidx = 3L * p->ind1;
            pickpointcoords.append3(&framepos[pidx]);
            pickpointindices.append2(a2->bondTo[k], p->ind1);
          }
        }
      }
      for (k=0; k < b1; k++){
        if (mol->atom(a1->bondTo[k])->atomType == ATOMHYDROGEN) {
          float *hydrogen = framepos + 3L*a1->bondTo[k];
          vec_sub(donortoH,hydrogen,coor1);
          vec_sub(Htoacceptor,coor2,hydrogen);
          if (angle(donortoH, Htoacceptor)  < maxangle ) {
            cmdColorIndex.putdata(atomColor->color[p->ind1], cmdList);
	    cmdLine.putdata(hydrogen,coor2, cmdList); // draw line

            // indicate the bonded atoms can be picked
            int pidx = 3L * a1->bondTo[k];
            pickpointcoords.append3(&framepos[pidx]);

            pidx = 3L * p->ind2;
            pickpointcoords.append3(&framepos[pidx]);
            pickpointindices.append2(a1->bondTo[k], p->ind2);
          }
        }
      }
    }
    tmp = p->next;
    free(p); 
  }
  free(onlist);

  // Revert line state to solid lines just in case other code has failed to 
  // ensure the correct line stipple state is in effect prior to drawing
  cmdLineType.putdata(SOLIDLINE, cmdList);

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0],
                           &pickpointcoords[0], cmdList);
  }
}


void DrawMolItem::draw_dynamic_bonds(float *framepos, float brad, int bres, float maxdist) {
  int lastcolor = -1;
  ResizeArray<float> pickpointcoords;
  ResizeArray<int> pickpointindices;

  // protect against ridiculous values which might crash us or
  // run the system out of memory.
  if (maxdist <= 0.1 || atomSel->selected == 0) {
    return;
  }
   
  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning dynamic bonds",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
   
  // set up general drawing characteristics for this drawing method
  append(DMATERIALON);
 
  // loop over all SELECTED atoms, in spite of the fact that non-bonded
  // atoms may actually be selected, searching for 
  // any pair that are closer than distance, not bonded,
  GridSearchPair *pairlist = vmd_gridsearch1(framepos, mol->nAtoms, atomSel->on, maxdist, 0, mol->nAtoms * 27);
  GridSearchPair *p, *tmp;
  for (p=pairlist; p != NULL; p=tmp) {
    MolAtom *atom1 = mol->atom(p->ind1);
    MolAtom *atom2 = mol->atom(p->ind2);

    // don't bond atoms that aren't part of the same conformation    
    // or that aren't in the all-conformations part of the structure 
    if ((atom1->altlocindex != atom2->altlocindex) &&
        ((mol->altlocNames.name(atom1->altlocindex)[0] != '\0') &&
         (mol->altlocNames.name(atom2->altlocindex)[0] != '\0'))) {
      tmp = p->next;
      free(p);
      continue;
    }

    // Prevent hydrogens from bonding with each other.
    // Use atomType info derived during initial molecule analysis for speed.
    if (!(atom1->atomType == ATOMHYDROGEN) ||
        !(atom2->atomType == ATOMHYDROGEN)) {
      float *coor1 = framepos + 3L*p->ind1; 
      float *coor2 = framepos + 3L*p->ind2; 
      float mid[3];
#if 0
      if (cutoff < 0) { // Do atom-specific distance check
        float d2 = distance2(coor1, coor2);
        float r1 = atom1->extra[ATOMRAD];
        float r2 = atom2->extra[ATOMRAD];
        float cut = 0.6f * (r1 + r2);
        if (d2 < cut*cut)
          // mol->add_bond(p->ind1, p->ind2);
      } else
#endif
      // mol->add_bond(p->ind1, p->ind2);

      // draw half-bond to each bonded, displayed partner
      // find the bond midpoint 'mid' between atoms 'i' and 'a2n'
      mid[0] = 0.5f * (coor1[0] + coor2[0]);
      mid[1] = 0.5f * (coor1[1] + coor2[1]);
      mid[2] = 0.5f * (coor1[2] + coor2[2]);

      if (lastcolor != atomColor->color[p->ind1]) {
        lastcolor = atomColor->color[p->ind1];
        cmdColorIndex.putdata(lastcolor, cmdList);
      }
      cmdCylinder.putdata(coor1, mid, brad, bres, 0, cmdList);

      if (lastcolor != atomColor->color[p->ind2]) {
        lastcolor = atomColor->color[p->ind2];
        cmdColorIndex.putdata(lastcolor, cmdList);
      }
      cmdCylinder.putdata(mid, coor2, brad, bres, 0, cmdList);

      // indicate the bonded atoms can be picked
      int pidx = 3L * p->ind1;
      pickpointcoords.append3(&framepos[pidx]);

      pidx = 3L * p->ind2;
      pickpointcoords.append3(&framepos[pidx]);
      pickpointindices.append2(p->ind1, p->ind2);
    }
  
    tmp = p->next;
    free(p); 
  }

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0],
                           &pickpointcoords[0], cmdList);
  }
}


#ifdef VMDPOLYHEDRA
// Based on original code by Dr. Francois-Xavier Coudert<f.coudert@ucl.ac.uk>
#define PLYMAXNB 16
void DrawMolItem::draw_polyhedra(float *framepos, float maxdist) {
  int i;
  int lastcolor = -1;

  // protect against ridiculous values which might crash us or
  // run the system out of memory.
  if (maxdist <= 0.1) {
    return;
  }

  // turn on all atoms for grid search (not using a real selection presently)
  int natoms = mol->nAtoms;
  int *onlist = new int[natoms];
  for (i=0; i<natoms; i++)
    onlist[i] = 1;

  // allocate array for selected neighbor list
  int *nblist = (int *) malloc(natoms * sizeof(int) * PLYMAXNB);
  if (nblist == NULL)
    return;
  memset(nblist, 0, natoms * sizeof(int) * PLYMAXNB);

  sprintf(commentBuffer,"MoleculeID: %d ReprID: %d Beginning polyhedra",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  // set up general drawing characteristics for this drawing method
  append(DMATERIALON);

  // loop over all SELECTED atoms, inspite of the fact that non-bonded
  // atoms may actually be selected, searching for 
  // any pair that are closer than distance, not bonded,
  GridSearchPair *pairlist = vmd_gridsearch1(framepos, natoms, onlist, maxdist, 0, natoms * (PLYMAXNB - 1));
  delete [] onlist;

  GridSearchPair *p, *tmp;

  for (p=pairlist; p != NULL; p=tmp) {
    MolAtom *atom1 = mol->atom(p->ind1);
    MolAtom *atom2 = mol->atom(p->ind2);

    // delete pairs where neither atom is selected
    if (!(atomSel->on[p->ind1] || atomSel->on[p->ind2])) {
      tmp = p->next;
      free(p);
      continue;
    }

    // don't bond atoms that aren't part of the same conformation    
    // or that aren't in the all-conformations part of the structure 
    if ((atom1->altlocindex != atom2->altlocindex) &&
        ((mol->altlocNames.name(atom1->altlocindex)[0] != '\0') &&
         (mol->altlocNames.name(atom2->altlocindex)[0] != '\0'))) {
      tmp = p->next;
      free(p);
      continue;
    }

    // record selected neighbors in the array
    // increment neighbor counts, clamping at PLYMAXNB-1 neighbors each
    int idx1 = p->ind1 * PLYMAXNB;
    int idx2 = p->ind2 * PLYMAXNB;
    if (nblist[idx1] < (PLYMAXNB-1) && nblist[idx2] < (PLYMAXNB-1)) {
      // update neighbor count 
      nblist[idx1]++; 
      nblist[idx2]++;

      // record new neighbors
      nblist[idx1 + nblist[idx1]] = p->ind2;
      nblist[idx2 + nblist[idx2]] = p->ind1;
    }

    tmp = p->next;
    free(p);
    continue;
  }

  // draw polyhedra composed of triangles for each combination
  // of three neighbors, using the neighbor atom coords as the vertices
  // XXX we should not be generating a massive list of independent 
  //     triangles as this is very wasteful of memory.  Instead, we
  //     should be building an indexed list of vertices, and generate
  //     one or more vertex array tokens for each of the colors that
  //     we'll ultimately draw.  This would save a massive amount of 
  //     display list memory that is wasted on needlessly 
  //     replicated vertex coordinates in the current implementation.
  for (i=0; i < natoms; i++) {
    int idx = i*PLYMAXNB;

    // don't draw a polyhedron if the atom isn't selected or
    // there are less than three neighbors within the user-selected 
    // cutoff distance
    if (!atomSel->on[i] || nblist[idx] < 4) 
      continue;

    // emit a color command only if necessary
    if (lastcolor != atomColor->color[i]) {
      lastcolor = atomColor->color[i];
      cmdColorIndex.putdata(lastcolor, cmdList);
    }

    int index=nblist[idx];
    for (int i1=1; i1 <= index - 2; i1++)
      for (int i2=i1; i2 <= index - 1; i2++)
        for (int i3=i2; i3 <= index; i3++)
	  cmdTriangle.putdata(framepos + 3L*nblist[idx+i1],
                              framepos + 3L*nblist[idx+i2],
                              framepos + 3L*nblist[idx+i3], cmdList);
  }

  free(nblist);
}
#endif


// find x = a*i + b where i = 0..n-1
static void least_squares(int n, const float *x, float *a, float *b) {
  float sum = 0;
  int i;
  for (i=0; i<n; i++) {    // find the sum of x
    sum += x[i];
  }
  float d = (float(n)-1.0f) / 2.0f;
  float t, sum_t2 = 0.0f;
  *a = 0.0f;
  for (i=0; i<n; i++) {
    t = (i - d);
    sum_t2 += t*t;
    *a += t*x[i];
  }
  *a /= sum_t2;
  *b = (sum/float(n) - d*(*a));
}

// given the info about the helix (length, color, etc.), draw it
// This computes a best fit line long the given points.  For each "on"
// atom, a cylinder is drawn from 1/2 a residue before to 1/2 a residue
// beyond, in the right color.
void DrawMolItem::draw_alpha_helix_cylinders(ResizeArray<float> &x,
       ResizeArray<float> &y, ResizeArray<float> &z, ResizeArray<int> &atom_on,
       int *color, float bond_rad, int bond_res,
       float *start_coord, float *end_coord)
{
  // compute the best fit line for each direction
  float a[3], b[3];
  ResizeArray<float> pickpointcoords;
  ResizeArray<int> pickpointindices;

  int num = x.num();
  least_squares(num, &(x[0]), a+0, b+0);
  least_squares(num, &(y[0]), a+1, b+1);
  least_squares(num, &(z[0]), a+2, b+2);
  
  // draw the cylinder(s)
  float start[3], end[3];
  
  start[0] = a[0] * (-0.5f) + b[0];
  start[1] = a[1] * (-0.5f) + b[1];
  start[2] = a[2] * (-0.5f) + b[2];
  vec_copy(start_coord, start);
  for (int i=0; i<x.num(); i++) {
    end[0] = a[0] * (float(i)+0.5f) + b[0];
    end[1] = a[1] * (float(i)+0.5f) + b[1];
    end[2] = a[2] * (float(i)+0.5f) + b[2];
    if (atom_on[i] >= 0) {
      // indicate this atom can be picked
      pickpointcoords.append3(x[i], y[i], z[i]);
      pickpointindices.append(atom_on[i]);
      cmdColorIndex.putdata(color[atom_on[i]], cmdList);
      int caps = 0;
      if (i == 0           || (i>0         && atom_on[i-1] < 0)) 
        caps |= CYLINDER_TRAILINGCAP;
      if (i == x.num() - 1 || (i<x.num()-1 && atom_on[i+1] < 0))
        caps |= CYLINDER_LEADINGCAP;
      if (bond_res <= 2 || bond_rad < 0.01) {
	// the representation will be as lines
	cmdLine.putdata(start, end, cmdList);
      } else {
	// draw the cylinder with closed ends
	cmdCylinder.putdata(start, end, bond_rad, bond_res, caps, cmdList);
      }
    }
    memcpy(start, end, 3L*sizeof(float));
  }
  vec_copy(end_coord, start);

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0],
                           &pickpointcoords[0], cmdList);
  }

  // reset for the next round
  x.clear();
  y.clear();
  z.clear();
  atom_on.clear();
}

#define SCALEADD(vec, offset, byscale) {   \
    vec[0] += (*(offset  ))*byscale;       \
    vec[1] += (*(offset+1))*byscale;       \
    vec[2] += (*(offset+2))*byscale;       \
}
#define SCALESUM(vec, term1, term2, byscale) {   \
    vec[0] = (*(term1  )) + (*(term2  ))*byscale;       \
    vec[1] = (*(term1+1)) + (*(term2+1))*byscale;       \
    vec[2] = (*(term1+2)) + (*(term2+2))*byscale;       \
}
#define BETASCALE (0.2f * ribbon_width)




void DrawMolItem::draw_beta_sheet(ResizeArray<float> &x,
       ResizeArray<float> &y, ResizeArray<float> &z, ResizeArray<int> &atom_on,
       int *color, float ribbon_width, float *start_coord, float *end_coord)
{
  ResizeArray<float> pickpointcoords;
  ResizeArray<int> pickpointindices;

  // compute the points between the atoms, and do a linear extrapolation
  // to get the first and last point.  (I should base it on something better
  // than a linear fit ... ).
  //   I am gauranteed at least three residues for initial data
  int num = x.num();
  int i;
  float *centers = new float[3L*(num+1)];
  for (i=1; i<num; i++) {   // compute the centers
    centers[3L*i+0] = (x[i-1]+x[i])/2;
    centers[3L*i+1] = (y[i-1]+y[i])/2;
    centers[3L*i+2] = (z[i-1]+z[i])/2;
  }
  // and the linear extrapolation
  centers[0] = 2*centers[3] - centers[6];
  centers[1] = 2*centers[4] - centers[7];
  centers[2] = 2*centers[5] - centers[8];

  centers[3L*num+0] = 2*centers[3L*num-3] - centers[3L*num-6];
  centers[3L*num+1] = 2*centers[3L*num-2] - centers[3L*num-5];
  centers[3L*num+2] = 2*centers[3L*num-1] - centers[3L*num-4];

  // now do the normals
  float *norms = new float[3L*(num+1)];  // along the width
  float *perps = new float[3L*(num+1)];  // along the height
  float d[2][3]; // deltas between successive points
  d[0][0] = x[1]-x[0];  d[0][1] = y[1]-y[0];  d[0][2] = z[1]-z[0];
  for (i=1; i<num-1; i++) {
    d[1][0] = x[i+1]-x[i];
    d[1][1] = y[i+1]-y[i];
    d[1][2] = z[i+1]-z[i];
    cross_prod(norms+3L*i, d[0], d[1]);
    vec_normalize(norms+3L*i);
    if (i%2) { // flip every other normal so they are aligned
      norms[3L*i+0] = -norms[3L*i+0];
      norms[3L*i+1] = -norms[3L*i+1];
      norms[3L*i+2] = -norms[3L*i+2];
    }
    vec_copy(d[0], d[1]);
  }
  // for the first one and last two normals, copy the end norms.
  vec_copy(norms, norms+3);
  vec_copy(norms+3L*num-3, norms+3L*num-6);
  vec_copy(norms+3L*num  , norms+3L*num-6);

  // and the perpendiculars
  for (i=0; i<num; i++) {
    vec_sub(d[0], centers+3L*i, centers+3L*i+3);
    cross_prod(perps+3L*i, d[0], norms+3L*i);
    vec_normalize(perps+3L*i);
  }
  vec_copy(perps+3L*num, perps+3L*num-3);

  // Draw rectangular blocks for each segment
  //  upper/lower, left/right corners;  this is the beginning of the block
  float ul[2][3], ur[2][3], ll[2][3], lr[2][3];
  SCALESUM(ul[0], centers, norms, ribbon_width);
  vec_copy    (ll[0], ul[0]);
  SCALEADD(ul[0], perps, BETASCALE);
  SCALEADD(ll[0], perps, -BETASCALE);

  SCALESUM(ur[0] , centers, norms, -ribbon_width);
  vec_copy    (lr[0], ur[0]);
  SCALEADD(ur[0], perps, BETASCALE);
  SCALEADD(lr[0], perps, -BETASCALE);

  // save the initial and final coordinate
  vec_copy(start_coord, centers);
  vec_copy(end_coord, centers+3L*num);

  int prev_on = 0;
  float dot=0.0f;
  for (i=0; i<num-1; i++) {  // go down the list of residues
    dot = dot_prod(perps+3L*i, perps+3L*i+3);
    // check the amount of rotation
    if (dot < 0) { // ribbons switched orientation
      // swap normals
      perps[3L*i+3] = -perps[3L*i+3];
      perps[3L*i+4] = -perps[3L*i+4];
      perps[3L*i+5] = -perps[3L*i+5];
      norms[3L*i+3] = -norms[3L*i+3];
      norms[3L*i+4] = -norms[3L*i+4];
      norms[3L*i+5] = -norms[3L*i+5];
    }

    // calculate the corners of the end of the blocks
    SCALESUM(ul[1], centers+3L*i+3, norms+3L*i+3, ribbon_width);
    vec_copy    (ll[1], ul[1]);
    SCALEADD(ul[1], perps+3L*i+3, BETASCALE);
    SCALEADD(ll[1], perps+3L*i+3, -BETASCALE);
    
    SCALESUM(ur[1] , centers+3L*i+3, norms+3L*i+3, -ribbon_width);
    vec_copy    (lr[1], ur[1]);
    SCALEADD(ur[1], perps+3L*i+3, BETASCALE);
    SCALEADD(lr[1], perps+3L*i+3, -BETASCALE);


    // draw this section, if it is on
    if (atom_on[i] >= 0) {
      // indicate this atom can be picked
      pickpointcoords.append3(x[i], y[i], z[i]);
      pickpointindices.append(atom_on[i]);

      cmdColorIndex.putdata(color[atom_on[i]], cmdList);

      if (ribbon_width != 0) {
	// draw 1 or 2 segments based on how much curvature there was
	if (i == 0 || dot > .9 || dot < -.9) {
	  // small flip, so I just need one segment
	  // draw the sides
	  cmdTriangle.putdata(ur[0], ul[0], ul[1],  // top
			      perps+3L*i, perps+3L*i, perps+3L*i+3, cmdList);
	  cmdTriangle.putdata(ur[0], ul[1], ur[1],
			      perps+3L*i, perps+3L*i+3, perps+3L*i+3, cmdList);
	  cmdTriangle.putdata(lr[0], ll[0], ll[1],  // bottom
			      perps+3L*i, perps+3L*i, perps+3L*i+3, cmdList);
	  cmdTriangle.putdata(lr[0], ll[1], lr[1],
			      perps+3L*i, perps+3L*i+3, perps+3L*i+3, cmdList);
	  
	  // and the other sides
	  cmdTriangle.putdata(ul[0], ll[0], ll[1], 
			      norms+3L*i, norms+3L*i, norms+3L*i+3, cmdList);
	  cmdTriangle.putdata(ul[0], ll[1], ul[1], 
			      norms+3L*i, norms+3L*i+3, norms+3L*i+3, cmdList);
	  cmdTriangle.putdata(ur[0], lr[0], lr[1], 
			      norms+3L*i, norms+3L*i, norms+3L*i+3, cmdList);
	  cmdTriangle.putdata(ur[0], lr[1], ur[1], 
			      norms+3L*i, norms+3L*i+3, norms+3L*i+3, cmdList);
	} else {
	  // big turn, so make more lines
	  // given the construction, I know the first and last residues
	  // don't twist (the perps are just copies) so I just do a simple
	  // spline along the centers
	  float centers_q[4][3];
	  make_spline_Q_matrix(centers_q, spline_basis, centers + 3L*i-3);
	  
	  // make the spline for the perps and norms
	  float perps_q[4][3];
	  make_spline_Q_matrix(perps_q, spline_basis, perps + 3L*i-3);
	  float norms_q[4][3];
	  make_spline_Q_matrix(norms_q, spline_basis, norms + 3L*i-3);
	  
	  // break the section into two parts, so compute the new middle
	  float new_center[3];
	  float new_perp[3];
	  float new_norm[3];
	  make_spline_interpolation(new_center, 0.5, centers_q);
	  make_spline_interpolation(new_perp  , 0.5, perps_q);
	  vec_normalize(new_perp);  // rescale correctly
	  make_spline_interpolation(new_norm  , 0.5, norms_q);
	  vec_normalize(new_norm);  // rescale correctly
	  
	  // and make the new corners
	  float ul2[3], ur2[3], ll2[3], lr2[3];
	  SCALESUM(ul2 , new_center, new_norm, ribbon_width);
	  vec_copy    (ll2, ul2);
	  SCALEADD(ul2, new_perp, BETASCALE);
	  SCALEADD(ll2, new_perp, -BETASCALE);
	  
	  SCALESUM(ur2 , new_center, new_norm, -ribbon_width);
	  vec_copy    (lr2, ur2);
	  SCALEADD(ur2, new_perp, BETASCALE);
	  SCALEADD(lr2, new_perp, -BETASCALE);
	  
	  // draw the new, intermediate blocks
	  cmdTriangle.putdata(ur[0], ul[0], ul2,  // top
			      perps+3L*i, perps+3L*i, new_perp, cmdList);
	  cmdTriangle.putdata(ur[0], ul2, ur2,
			      perps+3L*i, new_perp, new_perp, cmdList);
	  cmdTriangle.putdata(lr[0], ll[0], ll2,  // bottom
			      perps+3L*i, perps+3L*i, new_perp, cmdList);
	  cmdTriangle.putdata(lr[0], ll2, lr2,
			      perps+3L*i, new_perp, new_perp, cmdList);

	// and the other sides
	cmdTriangle.putdata(ul[0], ll[0], ll2, 
			    norms+3L*i, norms+3L*i, new_norm, cmdList);
	cmdTriangle.putdata(ul[0], ll2, ul2, 
			    norms+3L*i, new_norm, new_norm, cmdList);
	cmdTriangle.putdata(ur[0], lr[0], lr2, 
			    norms+3L*i, norms+3L*i, new_norm, cmdList);
	cmdTriangle.putdata(ur[0], lr2, ur2, 
			    norms+3L*i, new_norm, new_norm, cmdList);
	
	// get the 1/2 half
	cmdTriangle.putdata(ur2, ul2, ul[1],  // top
			    new_perp, new_perp, perps+3L*i+3, cmdList);
	cmdTriangle.putdata(ur2, ul[1], ur[1],
			    new_perp, perps+3L*i+3, perps+3L*i+3, cmdList);
	cmdTriangle.putdata(lr2, ll2, ll[1],  // bottom
			    new_perp, new_perp, perps+3L*i+3, cmdList);
	cmdTriangle.putdata(lr2, ll[1], lr[1],
			    new_perp, perps+3L*i+3, perps+3L*i+3, cmdList);
	
	// and the other sides
	cmdTriangle.putdata(ul2, ll2, ll[1], 
			    new_norm, new_norm, norms+3L*i+3, cmdList);
	cmdTriangle.putdata(ul2, ll[1], ul[1], 
			    new_norm, norms+3L*i+3, norms+3L*i+3, cmdList);
	cmdTriangle.putdata(ur2, lr2, lr[1], 
			    new_norm, new_norm, norms+3L*i+3, cmdList);
	cmdTriangle.putdata(ur2, lr[1], ur[1], 
			    new_norm, norms+3L*i+3, norms+3L*i+3, cmdList);
	}
	
	if (!prev_on) {
	  // draw the base
	  cmdTriangle.putdata(ul[0], ll[0], ur[0], cmdList);
	  cmdTriangle.putdata(ll[0], lr[0], ur[0], cmdList);
	  prev_on = 1;
	}
      } else {  // ribbon_width == 0
        cmdLine.putdata(centers+3L*i, centers+3L*i+3, cmdList);
      }
    } else { // atom_on[i] >= 0
      // this isn't on.  Was prev?  If so, draw its base
      if (prev_on && ribbon_width != 0) {
	cmdTriangle.putdata(ul[0], ll[0], ur[0], cmdList);
	cmdTriangle.putdata(ll[0], lr[0], ur[0], cmdList);
	prev_on = 0;
      }
    }

    vec_copy(ur[0], ur[1]);
    vec_copy(ul[0], ul[1]);
    vec_copy(lr[0], lr[1]);
    vec_copy(ll[0], ll[1]);
  } // end of loop though centers
  
  // but if the second to last one was on, then its end won't be visible
  if (prev_on && ribbon_width != 0) {
    cmdTriangle.putdata(ul[0], ll[0], ur[0], cmdList);
    cmdTriangle.putdata(ll[0], lr[0], ur[0], cmdList);
  }

  // swap the normals back, if they were swapped
  if (dot < 0) {
    perps[3L*i+0] = -perps[3L*i+0];
    perps[3L*i+1] = -perps[3L*i+1];
    perps[3L*i+2] = -perps[3L*i+2];
    norms[3L*i+0] = -norms[3L*i+0];
    norms[3L*i+1] = -norms[3L*i+1];
    norms[3L*i+2] = -norms[3L*i+2];
  }

  // what remains is the last one, which will be drawn as an arrow head
  if (atom_on[num-1] >= 0) {
    // indicate this atom can be picked
    pickpointcoords.append3(x[num-1], y[num-1], z[num-1]);
    pickpointindices.append(atom_on[num-1]);

    // recompute the base and tip information correctly
    // the 'norms' direction is 50% longer in each direction
    norms[3L*num-3] *= 1.5;
    norms[3L*num-2] *= 1.5;
    norms[3L*num-1] *= 1.5;

    SCALESUM(ul[0], centers+3L*num-3, norms+3L*num-3, ribbon_width);
    vec_copy    (ll[0], ul[0]);
    SCALEADD(ul[0], perps+3L*num-3, BETASCALE);
    SCALEADD(ll[0], perps+3L*num-3, -BETASCALE);

    SCALESUM(ur[0], centers+3L*num-3, norms+3L*num-3, -ribbon_width);
    vec_copy    (lr[0], ur[0]);
    SCALEADD(ur[0], perps+3L*num-3, BETASCALE);
    SCALEADD(lr[0], perps+3L*num-3, -BETASCALE);

    // and the tip has no norms component
    vec_copy    (ur[1], centers+3L*num);
    vec_copy    (lr[1], ur[1]);
    SCALEADD(ur[1], perps+3L*num, BETASCALE);
    SCALEADD(lr[1], perps+3L*num, -BETASCALE);

    // draw the arrow
    cmdColorIndex.putdata(color[atom_on[num-1]], cmdList);

    if (ribbon_width > 0) {
      // the normals are the same (from a copy) so I don't worry about them
      cmdTriangle.putdata(ul[0], ur[1], ur[0], cmdList);    // top
      // but that means here I _do_ have to switch the normal direction
      cmdTriangle.putdata(ll[0], lr[0], lr[1], cmdList);    // bottom
    
      cmdTriangle.putdata(ul[0], ll[0], lr[1], cmdList);  // sides
      cmdTriangle.putdata(ul[0], lr[1], ur[1], cmdList);
      cmdTriangle.putdata(ur[0], lr[0], lr[1], cmdList);
      cmdTriangle.putdata(ur[0], lr[1], ur[1], cmdList);

      cmdTriangle.putdata(ul[0], ll[0], ur[0], cmdList);  // and base
      cmdTriangle.putdata(ll[0], lr[0], ur[0], cmdList);
    } else {// ribbon_width == 0
      cmdLine.putdata(centers+3L*num-3, centers+3L*num, cmdList);
    } // special case for ribbon_width == 0
  }
  delete [] perps;
  delete [] norms;
  delete [] centers;

  // draw the pickpoints if we have any
  if (pickpointindices.num() > 0) {
    DispCmdPickPointArray pickPointArray;
    pickPointArray.putdata(pickpointindices.num(), &pickpointindices[0],
                           &pickpointcoords[0], cmdList);
  }

  // reset for the next round
  x.clear();
  y.clear();
  z.clear();
  atom_on.clear();
}

// draw based on secondary structure elements
void DrawMolItem::draw_structure(float *framepos, float brad, int bres, int linethickness) {
  sprintf (commentBuffer,"MoleculeID: %d ReprID: %d Beginning Cartoon",
         mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  // Indicate that I need secondary structure information
  mol->need_secondary_structure(1);  // calculate 2ndary structure if need be

  int h_start, b_start; // atom indices that start beta sheets and helices
  int atom=0;
  int frag, res;

  ResizeArray<float> resx;      // coord of start/end of structure
  ResizeArray<float> resy;
  ResizeArray<float> resz;
  ResizeArray<int> CA_num;      // CA of start/end structure
  ResizeArray<int> extra_resid; // CA of very short residues

  ResizeArray<float> x;         // coords (x,y,z) of the residues
  ResizeArray<float> y;
  ResizeArray<float> z;
  ResizeArray<int> atom_on;     // turn on/off unselected parts of the helix


  //
  // draw protein alpha helices as cylinders
  //
  if (bres <= 2 || brad < 0.01) {
    cmdLineType.putdata(SOLIDLINE, cmdList);
    cmdLineWidth.putdata(2, cmdList);
    append(DMATERIALOFF);
  } else {
    append(DMATERIALON);
  }
  
  h_start = -1; // reset the helix starting atom index
  for (frag=0; frag<mol->pfragList.num(); frag++) {
    int num = mol->pfragList[frag]->num();
    for (int resindex=0; resindex < num; resindex++) {
      res = (*mol->pfragList[frag])[resindex];
      const int ss = mol->residue(res)->sstruct;
      int residue_is_helix = 
        (ss == SS_HELIX_ALPHA || ss == SS_HELIX_3_10 ||
         ss == SS_HELIX_PI);

      if (residue_is_helix) {
        atom = mol->find_atom_in_residue("CA", res);
        if (atom >= 0) {
          if (h_start == -1) {
            h_start = atom;             // just started a helix
          }

          x.append(framepos[3L*atom+0]); // add CA atom to the coordinate arrays
          y.append(framepos[3L*atom+1]);
          z.append(framepos[3L*atom+2]);

          if (atomSel->on[atom]) {      // atom_on contains either
            atom_on.append(atom);       // the atom index (if on)
          } else {
            atom_on.append(-1);         // or -1 (if off)
          }
        } else {
          msgErr << "Missing a CA in a protein residue!!" << sendmsg;
        }
      }

      // If we got something that wasn't helix, or if we're on the last
      // residue of the fragment, draw what we have.
      if (!residue_is_helix || resindex == num-1) {
        if (x.num() <= 1) {
          // msgWarn << "Cartoon will not draw a helix of length 1" << sendmsg;
          if (x.num() == 1) {
            extra_resid.append(atom);  // keep track of these for the coil
          }
          x.clear();                   // clear the coordinate arrays
          y.clear();
          z.clear();
        } else {
          float ends[2][3];
          CA_num.append(h_start);      // save the residues which start
          CA_num.append(atom);         // and end the helix structure
          draw_alpha_helix_cylinders(x, y, z, atom_on, atomColor->color, 
                                     brad, bres, ends[0], ends[1]);

          resx.append(ends[0][0]);     // save helix start/end coordinates
          resy.append(ends[0][1]);
          resz.append(ends[0][2]);
          resx.append(ends[1][0]);
          resy.append(ends[1][1]);
          resz.append(ends[1][2]);
        }

        h_start = -1; // reset the helix starting atom index
      } // drew helix
    } // went through residues
  } // went through pfrag

  //
  // now do beta sheets via another pass through the pfrags
  //
  if (linethickness == 0) {
    cmdLineType.putdata(SOLIDLINE, cmdList);
    cmdLineWidth.putdata(2, cmdList);
    append(DMATERIALOFF);
  } else {
    append(DMATERIALON);
  }

  b_start = -1; // reset the beta sheet starting atom index
  for (frag=0; frag<mol->pfragList.num(); frag++) {
    int num = mol->pfragList[frag]->num();
    for (int resindex=0; resindex < num; resindex++) {
      int res = (*mol->pfragList[frag])[resindex];
      const int ss = mol->residue(res)->sstruct;
      int is_beta = (ss == SS_BETA || ss == SS_BRIDGE);
      if (is_beta) {
        atom = mol->find_atom_in_residue("CA", res);
        if (atom >= 0) {
          if (b_start == -1) {
            b_start = atom;             // just started a sheet
          }
          x.append(framepos[3L*atom+0]); // add CA atom to the coordinate arrays
          y.append(framepos[3L*atom+1]);
          z.append(framepos[3L*atom+2]);
          if (atomSel->on[atom]) {      // atom_on contains either
            atom_on.append(atom);       // the atom index (if on)
          } else {
            atom_on.append(-1);         // or -1 (if off)
          }
        } else {
          msgErr << "Missing a CA in a protein residue!!" << sendmsg;
        }
      }

      // Draw what we have if we got to a non-beta residue, or if this was
      // the last residue.
      if (b_start != -1 && (!is_beta || resindex == num-1)) {
        if (x.num() <= 2) {
          // msgWarn << "Cartoon will not draw a sheet with less than "
          //         << "3 residues" << sendmsg;

          extra_resid.append(b_start); // keep track of these for the coil
          if (x.num() == 2) {
            extra_resid.append(atom);  // keep track of these for the coil
          }
          x.clear();                   // clear the coordinate arrays
          y.clear();
          z.clear();
          atom_on.clear();             // clear the 'on' array
        } else {
          float ends[2][3];
          CA_num.append(b_start);      // save the residues which start
          CA_num.append(atom);         // and end the beta sheet structure
          draw_beta_sheet(x, y, z, atom_on, atomColor->color,
               linethickness / 5.0f,
               ends[0], ends[1]);
          resx.append(ends[0][0]);     // save beta sheet start/end coordinates
          resy.append(ends[0][1]);
          resz.append(ends[0][2]);
          resx.append(ends[1][0]);
          resy.append(ends[1][1]);
          resz.append(ends[1][2]);
        }

        b_start = -1; // reset the beta sheet starting atom index
      }
    }  // went through the residues
  } // went through pfrag


  // Finally, everything left I can draw as a spline.  The tricky part
  // is I need to get the start/end coordinates of the cylinders and
  // sheets correct, for otherwise  the tube doesn't meet with them nicely.
  // Luckily, I kept track of the start/end coordinates as I went along, so
  // I swap the coords of that array with the frampos data, replace atomSel'
  // 'on' with a new 'on', and call draw_tube.  Code reuse, right?  
  // Anyway, when I return I swap everything back
  {
    // finding the atoms is easy; search for 'protein not (sheet or helix)'
    // and not those atoms in the same residue as the extra_resid list.
    int nres, i;

    // allocate and clear a new 'on' list
    int *on = new int[mol->nAtoms];
    memset(on, 0, mol->nAtoms*sizeof(int));

    // find "protein and not (sheet or helix)"
    nres=mol->residueList.num();
    for (i=0; i<nres; i++) {
      const Residue *res = mol->residueList[i];
      if (res->residueType == RESPROTEIN) {
        const int ss = res->sstruct;
        if (ss == SS_TURN || ss == SS_COIL) {
          int numatoms = res->atoms.num();
          for (int j=0; j<numatoms; j++) {
            on[res->atoms[j]] = 1;
          }
        }
      }
    }
    
    // collect the leftover residues in "short" sheets or cylinders
    // and draw them as tubes as well
    nres=extra_resid.num();
    for (i=0; i<nres; i++) {
      int res = mol->atom(extra_resid[i])->uniq_resid;
      Residue *r = mol->residue(res);
      int numatoms = r->atoms.num();
      for (int j=0; j<numatoms; j++) {
        on[r->atoms[j]] = 1;
      }
    }

    // swap the real CA atom coordinates with calculated 
    // control point coordinates so that draw_tube generates
    // the right shape.
    float tmp_coords[3];
    int atom; 
    int numcaatoms = CA_num.num();
    for (i=0; i<numcaatoms; i++) {
      atom = 3L*CA_num[i];
      vec_copy(tmp_coords, framepos+atom);
      framepos[atom+0] = resx[i];
      framepos[atom+1] = resy[i];
      framepos[atom+2] = resz[i];
      resx[i] = tmp_coords[0];
      resy[i] = tmp_coords[1];
      resz[i] = tmp_coords[2];
      // turn CA of the start/end on as well
      on[CA_num[i]] = 2+i%2;    ////////  <<<<-----  this is "2" or "3"
      // the "2" is a special case for draw_spline_curve which will only
      // draw the first 1/2 if it is a 2 or second half if it is a 3
    }

    // replace the list of "on" atoms with the new list
    for (i=0; i<mol->nAtoms; i++) {
      if (!atomSel->on[i]) { // mask the coil against the selected atoms
        on[i] = 0;
      }
    }
    int *temp_on = atomSel->on;
    atomSel->on = on;

    // make the tube
    draw_tube(framepos, 
              brad / 7.0f,
              int(float(bres) * 2.0f/3.0f + 0.5f));

    // and revert to the original coordinate and 'on' arrays
    atomSel->on = temp_on;
    for (i=0; i<numcaatoms; i++) {
      atom = 3L*CA_num[i];
      framepos[atom+0] = resx[i];
      framepos[atom+1] = resy[i];
      framepos[atom+2] = resz[i];
    }

    delete [] on; // delete the temporary 'on' array
  }

  append(DMATERIALOFF);
}

int DrawMolItem::pickable_on() {
  return displayed() && mol && mol->displayed();
}

