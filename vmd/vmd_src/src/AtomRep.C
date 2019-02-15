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
 *      $RCSfile: AtomRep.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.134 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Parse and maintain the data for how a molecule should be represented.
 *
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include "AtomRep.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"

/***********************  NOTES on ADDING NEW REPRESENTATIONS ************
 So, you wanna add a new representation ... here's what you have to do:
 1. Choose a a name for the rep, and decide how many parameters it needs.
    This object has support for up to MAX_ATOMREP_DATA parameters.
 2. Choose the order in which these will be specified in the text command.
 3. Add a new identifier in the main RepMethod enumeration for this new
    rep.  The order is important; where you put the new item is where it
    will appear in lists of the different representations.
 4. Add a new element to the AtomRepInfo structure array (below).  This element
    defines the name of the new rep, and gives info about all the parameters
    needed for that rep.  You must put the item into the array in the same
    position where the new enum value was inserted into RepMethod.
 5. Once this structure is properly defined, the GUI and text interface will
    properly process and display the info for the new rep.  At this point,
    you must add the code to display the new rep in DrawMolItem.
 6. Sit back and enjoy ...

 FORMAT for AtomRepInfo STRUCTURE:
 ---------------------------------
 Easiest thing to do is to copy an item with the same number of parameters,
 and change the necessary values.  The items in each structure are:
	a. Integer enum code for the rep (same as was put in RepMethod enum)
	b. Name of rep (should be a single small word; this is the name
	   used in the GUI list, and what the user calls the rep when typing
	   in a command to change to this rep)
	c. Number of parameters required.
	d. An array of Structures which define how each parameter is used
	   for the representation.  The format for these items are:
		1. Index of the parameter in the AtomRep internal data
		   storage array.  This should be used when retrieving the
		   data; call 'get_data' with this index value to retrieve
		   the value for this parameter.  Each parameter for the
		   representation should have a different index, and they
		   should be in the range 0 ... MAX_ATOMREP_DATA-1.
		   There is an enumeration giving names for these indices,
		   which is used right now primarily for historical reasons.
		   Older parts of VMD used explicit calls to routines to get
		   values for things such as the line thickness and sphere
		   resolution, and these routines are still present.  But
		   since new reps may not necessarily have characteristics
		   which map to these items, this should not be used in the
		   future.
		2. Default value
 ***************************************************************************/

// default res of cartoon cylinders
#define DEF_CART_RES 12.0  

// default res of bond cylinders
#define DEF_BOND_RES 12.0

// default res of spheres
#define DEF_SPH_RES  12.0 

// default res of licorice reps
#define DEF_LIC_RES  12.0 

// default thickness of lines
#define DEF_LINE_RES 1.0 

// default point size
#define DEF_POINT_SIZE 1.0 

//const AtomRepDataInfo UNUSED = {0, 0.0};
#define UNUSED {0, 0.0}

// define structures which indicate what data is needed by each rep
const AtomRepParamStruct AtomRepInfo[AtomRep::TOTAL] = {
  //
  // Bond-oriented reps
  //

  // Lines
  { "Lines", 2, {
    {  AtomRep::LINETHICKNESS, DEF_LINE_RES },
    {  AtomRep::ISOLINETHICKNESS, 0.0f },
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Bonds
  { "Bonds", 2, {
    {  AtomRep::BONDRAD, 0.3f },
    {  AtomRep::BONDRES, DEF_BOND_RES },
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // DynamicBonds
  { "DynamicBonds", 3, {
    {  AtomRep::SPHERERAD, 3.0f },  // distance
    {  AtomRep::BONDRAD, 0.3f },
    {  AtomRep::BONDRES, DEF_BOND_RES },
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // HBonds
  { "HBonds", 3, {
    {  AtomRep::BONDRAD, 3.0f},     // distance
    {  AtomRep::SPHERERAD, 20.0f},  // angle
    {  AtomRep::LINETHICKNESS, 1.0f}, 
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },


  //
  // Atom-oriented reps
  //

  // Points
  { "Points", 1, { 
    {  AtomRep::LINETHICKNESS, DEF_POINT_SIZE},  // size
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, 
    UNUSED, UNUSED, UNUSED, UNUSED }
  },

  // VDW
  { "VDW", 2, {
    {  AtomRep::SPHERERAD, 1.0f},
    {  AtomRep::SPHERERES, DEF_SPH_RES },
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // CPK
  { "CPK", 5, {
    {  AtomRep::SPHERERAD, 1.0f},
    {  AtomRep::BONDRAD, 0.3f},
    {  AtomRep::SPHERERES, DEF_SPH_RES},
    {  AtomRep::BONDRES, DEF_BOND_RES},
    {  AtomRep::ISOLINETHICKNESS, 0.0f },
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED }
  },

  // Licorice
  { "Licorice", 4, {
    {  AtomRep::BONDRAD, 0.3f },
    {  AtomRep::SPHERERES, DEF_LIC_RES},
    {  AtomRep::BONDRES, DEF_LIC_RES},
    {  AtomRep::ISOLINETHICKNESS, 0.0f }, // if nonzero, cutoff distance
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },


#ifdef VMDPOLYHEDRA
  // Polyhedra
  { "Polyhedra", 1, {
    {  AtomRep::SPHERERAD, 3.0f },  // distance
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, 
    UNUSED, UNUSED, UNUSED, UNUSED }
  },
#endif

  //
  // Secondary structure reps
  //

  // C-alpha trace
  { "Trace", 2, {
    {  AtomRep::BONDRAD, 0.3f},
    {  AtomRep::BONDRES, DEF_BOND_RES},
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Tube
  { "Tube", 2, {
    {  AtomRep::BONDRAD, 0.3f },
    {  AtomRep::BONDRES, DEF_BOND_RES},
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Ribbons
  { "Ribbons", 3, {
    {  AtomRep::BONDRAD, 0.3f},
    {  AtomRep::BONDRES, DEF_BOND_RES},
    {  AtomRep::LINETHICKNESS, 2.0f},
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // NewRibbons
  { "NewRibbons", 4, {
    {  AtomRep::BONDRAD, 0.3f },
    {  AtomRep::BONDRES, 12.0f },
    {  AtomRep::LINETHICKNESS, 3.0f},
    {  AtomRep::SPHERERAD, 0},  // spline basis
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Structure
  { "Cartoon", 3, {
    {  AtomRep::BONDRAD, 2.1f},
    {  AtomRep::BONDRES, 24.0f },
    {  AtomRep::LINETHICKNESS, 5.0f},
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // NewCartoon
  { "NewCartoon", 4, {
    {  AtomRep::BONDRAD, 0.3f },
    {  AtomRep::BONDRES, 12.0f },
    {  AtomRep::LINETHICKNESS, 4.5f},
    {  AtomRep::SPHERERAD, 0},  // spline basis
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

#ifdef VMDWITHCARBS
  // PaperChain
  { "PaperChain", 2, {
    {  AtomRep::LINETHICKNESS, 1.0f }, // bipyramid_height : height of the pyramids drawn to represent rings
    {  AtomRep::ISOSTEPSIZE, 10 }, // maxringsize : maximum number of atoms in a small ring
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, 
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED }
  },
    
  // Twister
  { "Twister", 7, {
    {  AtomRep::LINETHICKNESS, 1 }, // start_end_centroid : 0 - start ribbons at the edge of the ring, 1 - start ribbons at the ring centroids
    {  AtomRep::BONDRES, 0 }, // hide_shared_links: 0 - show all linkage paths, 1 - hide linkage paths which share edges with other paths
    {  AtomRep::SPHERERAD, 10 }, // rib_steps : number of steps to use when drawing ribbons
    {  AtomRep::BONDRAD, 0.3f }, // rib_width : width of the ribbon
    {  AtomRep::SPHERERES, 0.05f }, // rib_height : height of the ribbon
    {  AtomRep::ISOSTEPSIZE, 10 }, // maxringsize : maximum number of atoms in a small ring
    {  AtomRep::ISOLINETHICKNESS, 5 }, // maxpathlength : maximum number of atoms in a path joining two small rings
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED }
  },
#endif


  //
  // Surface, Isosurface, and Solvent reps
  //

#ifdef VMDQUICKSURF
  // QuickSurf
  { "QuickSurf", 4, {
    {  AtomRep::SPHERERAD, 1.0f },  // sphere radius multiplier
    {  AtomRep::BONDRAD,   0.5f },  // density isovalue for surface
    {  AtomRep::GRIDSPACING, 1.0f}, // lattice grid spacing 
    {  AtomRep::BONDRES, 0 },       // quality
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },
#endif

#ifdef VMDMSMS
  // MSMS surface
  { "MSMS", 4, {
    {  AtomRep::SPHERERAD, 1.5f }, // probe radius
    {  AtomRep::SPHERERES, 1.5f }, // density
    {  AtomRep::LINETHICKNESS, 0}, // all atoms
    {  AtomRep::BONDRES, 0},       // wireframe
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },
#endif

#ifdef VMDNANOSHAPER
  // NanoShaper surface
  { "NanoShaper", 6, {
    {  AtomRep::LINETHICKNESS, 0},  // surface type
    {  AtomRep::BONDRES, 0},        // wireframe
    {  AtomRep::GRIDSPACING, 0.5f}, // lattice grid spacing 
    {  AtomRep::SPHERERAD, 1.40f }, // probe radius
    {  AtomRep::SPHERERES, 0.45f }, // skin parm
    {  AtomRep::BONDRAD,    0.5f }, // blob parm
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },
#endif

#ifdef VMDSURF
  // Surf
  { "Surf", 2, {
    {  AtomRep::SPHERERAD, 1.4f }, // probe radius
    {  AtomRep::BONDRES, 0},       // wireframe
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },
#endif

  // Slice of volumetric data
  { "VolumeSlice", 4, {
    { AtomRep::SPHERERAD,  0.5f }, // slice
    { AtomRep::SPHERERES,     0 }, // VolID
    { AtomRep::LINETHICKNESS, 0 }, // slice axis
    { AtomRep::BONDRES,       2 },  // quality
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Isosurface of volumetric data
  { "Isosurface", 6, {
    { AtomRep::SPHERERAD,    0.5f }, // isoval
    { AtomRep::SPHERERES,        0}, // VolID
    { AtomRep::LINETHICKNESS,    2}, // draw box and isosurface by default
    { AtomRep::BONDRES,          2}, // use points method by default
    { AtomRep::ISOSTEPSIZE,      1}, // use step size of one by default
    { AtomRep::ISOLINETHICKNESS, 1}, // lines are thickness one by default
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Volumetric gradient field lines
  { "FieldLines", 8, {
    { AtomRep::SPHERERES,      0},  // VolID
    { AtomRep::SPHERERAD,   1.8f},  // seed value
    { AtomRep::BONDRAD,    10.0f},  // minimum line length
    { AtomRep::BONDRES,    50.0f},  // maximum line length
    { AtomRep::LINETHICKNESS,  1},  // line thickness
    { AtomRep::FIELDLINESTYLE, 0},  // field lines drawn as lines
    { AtomRep::FIELDLINESEEDUSEGRID, 0}, // field line seed type
    { AtomRep::FIELDLINEDELTA, 0.25},  // field lines integrator stepsize
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED}
  },

  // Molecular orbital
  { "Orbital", 10, {
    { AtomRep::SPHERERAD,     0.05f}, // isoval
    { AtomRep::SPHERERES,         0}, // Orbital ID
    { AtomRep::LINETHICKNESS,     0}, // draw only isosurface by default
    { AtomRep::BONDRES,           0}, // use solid method by default
    { AtomRep::GRIDSPACING,  0.075f}, // default orbital grid spacing
    { AtomRep::ISOLINETHICKNESS,  1}, // lines are thickness one by default
    { AtomRep::WAVEFNCTYPE,       0}, // default wavefunction type 
    { AtomRep::WAVEFNCSPIN,       0}, // default wavefunction spin
    { AtomRep::WAVEFNCEXCITATION, 0}, // default wavefunction excitation
    { AtomRep::ISOSTEPSIZE,       1}, // use step size of one by default
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED }
  },

  // Beads 
  { "Beads", 2, {
    {  AtomRep::SPHERERAD, 1.0f},
    {  AtomRep::SPHERERES, DEF_SPH_RES },
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Dotted
  { "Dotted", 2, {
    {  AtomRep::SPHERERAD, 1.0f},
    {  AtomRep::SPHERERES, DEF_SPH_RES},
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  },

  // Dot surface
  { "Solvent", 3, {
    {  AtomRep::SPHERERAD, 0 }, // probe
    // the max value is set in the constructor for AtomRep, so don't
    // change things around too much!
    {  AtomRep::SPHERERES, 7.0f},  // detail
    {  AtomRep::LINETHICKNESS, 1.0f }, // method
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
#ifdef VMDLATTICECUBES
  },

  // LatticeCubes
  { "LatticeCubes", 1, {
    {  AtomRep::SPHERERAD, 1.0f},
    UNUSED, 
    UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
    UNUSED, UNUSED, UNUSED }
  }
#else
  }
#endif

};


//////////////////////////  constructor and destructor
// constructor; parse string and see if OK
AtomRep::AtomRep(void) {
  int i, j;
 
  // initialize variables
  repMethod  = DEFAULT_ATOMREP;
  repData = repDataStorage[repMethod];
  strcpy(cmdStr, AtomRepInfo[repMethod].name);
  for(j=0; j < TOTAL; j++) {
    memset(repDataStorage[j], 0, MAX_ATOMREP_DATA*sizeof(float));
    for(i=0; i < AtomRepInfo[j].numdata ; i++)
      repDataStorage[j][AtomRepInfo[j].repdata[i].index] =
        AtomRepInfo[j].repdata[i].defValue;
  }

}

AtomRep::~AtomRep(void) { }

//////////////////////////  private routines
// parse the given command, and store results.  Return success.
int AtomRep::parse_cmd(const char *newcmd) {
  int argc, i, j;
  char *argv[256], *cmdStrTok = NULL;

  // make sure the new command is not too long
  if(newcmd && strlen(newcmd) > MAX_ATOMREP_CMD) {
    msgErr << "Atom representation string is too long (over ";
    msgErr << MAX_ATOMREP_CMD << " characters)." << sendmsg;
    return FALSE;
  }

  // XXX Hack to make the "Off" rep work: alias to "Lines"
  if (!strupncmp(newcmd, "off", 3)) newcmd = "Lines";

  // tokenize the command
  if(!newcmd || !(cmdStrTok = str_tokenize(newcmd, &argc, argv))) {
    // no command; keep current settings
    return TRUE;
  }

  // see if the command matches a representation we know about
  for(i=0; i < AtomRep::TOTAL; i++) {
    const AtomRepParamStruct *ari = &(AtomRepInfo[i]);

    if(!strupncmp(argv[0], ari->name, CMDLEN)) {
      // the name matches; make sure we do not have too many arguments
      if((argc - 1) > ari->numdata) {
	msgErr << "Incorrect atom representation command '" << newcmd << "':";
	msgErr << "\n  '" << ari->name << "' only takes " << ari->numdata;
	msgErr << " arguments maximum." << sendmsg;

	delete [] cmdStrTok;
	return FALSE;
      }

      // indicate that we've changed to a new rep method
      repMethod = i;
      repData = repDataStorage[i];

      // copy out the data from the command and store it
      for(j=1; j < argc; j++) {
	// if a '-' is given, do not change that parameter
	if(!(argv[j][0] == '-' && argv[j][1] == '\0'))
          repData[ari->repdata[j-1].index] = (float) atof(argv[j]);
      }

      // found a match; break from for loop
      break;
    }
  }

  // if i is at the end of the list, we did not find a match
  if(i == AtomRep::TOTAL) {
    msgErr << "Unknown atom representation command '" << newcmd << "'.";
    msgErr << sendmsg;

    delete [] cmdStrTok;
    return FALSE;
  }

  // if we are here, everything went OK
  strcpy(cmdStr, newcmd);

  delete [] cmdStrTok;
  return TRUE;
}


//////////////////////////  public routines

// copy contents of another atomrep
AtomRep &AtomRep::operator=(const AtomRep &ar) {
  strcpy(cmdStr, ar.cmdStr);
  repMethod = ar.repMethod;
  repData = repDataStorage[repMethod];
  for(int j=0; j < TOTAL; j++)
    for(int i=0; i < MAX_ATOMREP_DATA; i++)
      repDataStorage[j][i] = ar.repDataStorage[j][i];

  return *this;
}

