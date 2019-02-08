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
 *      $RCSfile: AtomColor.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.105 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Parse and maintain the data for how a molecule should be colored.
 *
 ***************************************************************************/

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "AtomColor.h"
#include "DrawMolecule.h"
#include "MoleculeList.h"
#include "Scene.h"
#include "VolumetricData.h"
#include "PeriodicTable.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"

//
// XXX global string arrays with text descriptions of coloring methods
//
const char *AtomColorName[AtomColor::TOTAL] = { 
  "Name",     "Type",     "Element",
  "ResName",  "ResType",  "ResID",    
  "Chain",    "SegName",  "Conformation",
  "Molecule", "Structure", 
  "ColorID",  "Beta",     "Occupancy", 
  "Mass",     "Charge",   
  "Pos",      "PosX",     "PosY",  "PosZ", 
  "User",     "User2",    "User3", "User4", "Fragment",
  "Index",    "Backbone", "Throb",
  "PhysicalTime", "Timestep", "Velocity", "Volume"
  };

// These names are used within GUI menus, so they contain submenu names etc
const char *AtomColorMenuName[AtomColor::TOTAL] = { 
  "Name",          "Type",          "Element",
  "ResName",       "ResType",       "ResID",    
  "Chain",         "SegName",       "Conformation",
  "Molecule",      "Secondary Structure", 
  "ColorID",       "Beta",          "Occupancy", 
  "Mass",          "Charge",   
  "Position/Radial", "Position/X", "Position/Y", "Position/Z", 
  "Trajectory/User/User", "Trajectory/User/User2", "Trajectory/User/User3", "Trajectory/User/User4",
  "Fragment", "Index",  "Backbone", "Throb",
  "Trajectory/Physical Time", "Trajectory/Timestep", "Trajectory/Velocity", "Volume"
  };


//////////////////////////  constructor and destructor
// constructor; parse string and see if OK
AtomColor::AtomColor(MoleculeList *mlist, const Scene *sc) {

  // initialize variables
  molList = mlist;
  scene = sc;
  color = NULL;
  colIndex = 0;
  volIndex = 0;
  nAtoms = 0;
  mol = NULL;
  colorMethod  = DEFAULT_ATOMCOLOR;
  strcpy(cmdStr, AtomColorName[DEFAULT_ATOMCOLOR]);
  do_update = 0;
  need_recalc_minmax = TRUE;
  minRange = maxRange = 0;
}


// copy constructor
AtomColor::AtomColor(AtomColor& ar) {
  strcpy(cmdStr, ar.cmdStr);
  colorMethod = ar.colorMethod;
  colIndex = ar.colIndex;
  volIndex = ar.volIndex;
  scene = ar.scene;
  molList = ar.molList;
  mol = ar.mol;
  if(ar.color) {
    nAtoms = ar.nAtoms;
    color = new int[nAtoms];
    for(int i=0; i < nAtoms; i++)
      color[i] = ar.color[i];
  } else {
    nAtoms = 0;
    color = NULL;
  }
  do_update = ar.do_update;
  need_recalc_minmax = TRUE;
  minRange = maxRange = 0;
}


// destructor; free up space
AtomColor::~AtomColor(void) {
  if(color)
    delete [] color;
}


//////////////////////////  private routines

// parse the given command, and store results.  Return success.
int AtomColor::parse_cmd(const char *newcmd) {
  int argc, ok = TRUE;
  char *argv[128], *cmdStrTok = NULL;

  ColorMethod newMethod = colorMethod;
  int newIndex = colIndex;
  int newVol   = volIndex;
 
  // make sure the new command is not too long
  if(newcmd && strlen(newcmd) > MAX_ATOMCOLOR_CMD) {
    msgErr << "Atom coloring method string is too long (over ";
    msgErr << MAX_ATOMCOLOR_CMD << " characters)." << sendmsg;
    return FALSE;
  }

  // tokenize the command
  if(!newcmd || !(cmdStrTok = str_tokenize(newcmd, &argc, argv))) {
    // no command; keep current settings
    return TRUE;
  }

  // now parse the command
  if(!strupncmp(argv[0], "default", CMDLEN)) {
    newMethod = DEFAULT_ATOMCOLOR;
  } else if(!strupncmp(argv[0], AtomColorName[NAME], CMDLEN)) {
    newMethod = NAME;
  } else if(!strupncmp(argv[0], AtomColorName[TYPE], CMDLEN)) {
    newMethod = TYPE;
  } else if(!strupncmp(argv[0], AtomColorName[ELEMENT], CMDLEN)) {
    newMethod = ELEMENT;
  } else if(!strupncmp(argv[0], AtomColorName[RESNAME], CMDLEN)) {
    newMethod = RESNAME;
  } else if(!strupncmp(argv[0], AtomColorName[RESTYPE], CMDLEN)) {
    newMethod = RESTYPE;
  } else if(!strupncmp(argv[0], AtomColorName[RESID], CMDLEN)) {
    newMethod = RESID;
  } else if(!strupncmp(argv[0], AtomColorName[CHAIN], CMDLEN)) {
    newMethod = CHAIN;
  } else if(!strupncmp(argv[0], AtomColorName[SEGNAME], CMDLEN)) {
    newMethod = SEGNAME;
  } else if(!strupncmp(argv[0], AtomColorName[CONFORMATION], CMDLEN)) {
    newMethod = CONFORMATION;
  } else if(!strupncmp(argv[0], AtomColorName[MOLECULE], CMDLEN)) {
    newMethod = MOLECULE;
  } else if(!strupncmp(argv[0], AtomColorName[STRUCTURE], CMDLEN)) {
    newMethod = STRUCTURE;
  } else if(!strupncmp(argv[0], AtomColorName[COLORID], CMDLEN) && argc > 1) {
    newMethod = COLORID;
    newIndex = atoi(argv[1]);
  } else if(!strupncmp(argv[0], AtomColorName[BETA], CMDLEN)) {
    newMethod = BETA;
  } else if(!strupncmp(argv[0], AtomColorName[OCCUP], CMDLEN)) {
    newMethod = OCCUP;
  } else if(!strupncmp(argv[0], AtomColorName[MASS], CMDLEN)) {
    newMethod = MASS;
  } else if(!strupncmp(argv[0], AtomColorName[CHARGE], CMDLEN)) {
    newMethod = CHARGE;
  } else if(!strupncmp(argv[0], AtomColorName[USER], CMDLEN)) {
    newMethod = USER;
  } else if(!strupncmp(argv[0], AtomColorName[USER2], CMDLEN)) {
    newMethod = USER2;
  } else if(!strupncmp(argv[0], AtomColorName[USER3], CMDLEN)) {
    newMethod = USER3;
  } else if(!strupncmp(argv[0], AtomColorName[USER4], CMDLEN)) {
    newMethod = USER4;
  } else if(!strupncmp(argv[0], AtomColorName[FRAGMENT], CMDLEN)) {
    newMethod = FRAGMENT;
  } else if(!strupncmp(argv[0], AtomColorName[POS], CMDLEN)) {
    newMethod = POS;
  } else if(!strupncmp(argv[0], AtomColorName[POSX], CMDLEN)) {
    newMethod = POSX;
  } else if(!strupncmp(argv[0], AtomColorName[POSY], CMDLEN)) {
    newMethod = POSY;
  } else if(!strupncmp(argv[0], AtomColorName[POSZ], CMDLEN)) {
    newMethod = POSZ;
  } else if(!strupncmp(argv[0], AtomColorName[INDEX], CMDLEN)) {
    newMethod = INDEX;
  } else if(!strupncmp(argv[0], AtomColorName[BACKBONE], CMDLEN)) {
    newMethod = BACKBONE;
  } else if(!strupncmp(argv[0], AtomColorName[THROB], CMDLEN)) {
    newMethod = THROB;
  } else if(!strupncmp(argv[0], AtomColorName[PHYSICALTIME], CMDLEN)) {
    newMethod = PHYSICALTIME;
  } else if(!strupncmp(argv[0], AtomColorName[TIMESTEP], CMDLEN)) {
    newMethod = TIMESTEP;
  } else if(!strupncmp(argv[0], AtomColorName[VELOCITY], CMDLEN)) {
    newMethod = VELOCITY;
  } else if(!strupncmp(argv[0], AtomColorName[VOLUME], CMDLEN) && argc > 1) {
    newMethod = VOLUME;
    newVol = atoi(argv[1]);
  } else {
    // unknown representation
    ok = FALSE;
  }

  // check the command was not too long
  ok = ok && ((((newMethod == COLORID) || 
                (newMethod == VOLUME)) && argc < 4) || 
                (newMethod != COLORID && argc < 3));

  // print error message if necessary
  if (!ok) {
    msgErr << "Incorrect atom color method command '" << newcmd << "'";
    msgErr << sendmsg;
  } else {
    // command was successful, save new settings
    colorMethod = newMethod;
    colIndex = newIndex;
    volIndex = newVol;
    strcpy(cmdStr, newcmd);

    // range check colorid index    
    if (colIndex < 0)
      colIndex = 0;
    else if (colIndex > VISCLRS)
      colIndex = VISCLRS;

    // range check volume index    
    if (volIndex < 0)
      volIndex = 0;

    need_recalc_minmax = TRUE;
  }

  // delete parsing space
  delete [] cmdStrTok;
  return ok;
}


//////////////////////////  public routines

// equal operator, to change the current settings.  Does NOT change the
// current molecule, molecule list, or color list.
AtomColor& AtomColor::operator=(const AtomColor &ar) {

  // copy values
  if(cmdStr != ar.cmdStr)
    strcpy(cmdStr, ar.cmdStr);
  colorMethod = ar.colorMethod;
  colIndex = ar.colIndex;
  volIndex = ar.volIndex;
  
  // update current colors based on new settings
  need_recalc_minmax = TRUE;
  find(mol);  
    
  return *this;
}

int AtomColor::current_color_use(int ccat) {

  if((colorMethod==MOLECULE&& ccat==molList->colorCatIndex[MLCAT_MOLECULES])
    || (colorMethod==CONFORMATION&& ccat==molList->colorCatIndex[MLCAT_CONFORMATIONS])
    || (colorMethod==NAME&& ccat==molList->colorCatIndex[MLCAT_NAMES])
    || (colorMethod==TYPE&& ccat==molList->colorCatIndex[MLCAT_TYPES])
    || (colorMethod==ELEMENT&& ccat==molList->colorCatIndex[MLCAT_ELEMENTS])
    || (colorMethod==RESNAME&& ccat==molList->colorCatIndex[MLCAT_RESNAMES])
    || (colorMethod==RESTYPE&& ccat==molList->colorCatIndex[MLCAT_RESTYPES])
    || (colorMethod==CHAIN&& ccat==molList->colorCatIndex[MLCAT_CHAINS])
    || (colorMethod==SEGNAME&& ccat==molList->colorCatIndex[MLCAT_SEGNAMES])
    || (colorMethod==BACKBONE&& ccat==molList->colorCatIndex[MLCAT_SPECIAL])
    || (colorMethod==STRUCTURE&& ccat==molList->colorCatIndex[MLCAT_SSTRUCT])
    ) {
      return TRUE;
  }
  
  return FALSE;
}

// find the color index for the atoms of the given molecule.  Return success.
int AtomColor::find(DrawMolecule *m) {
  int dindex;

  // make sure things are OK
  if(!m || !molList)
    return FALSE;
    
  // save new molecule, and remove old storage if necessary  
  if(color && mol && nAtoms != m->nAtoms) {
    delete [] color;
    color = NULL;
  }

  // allocate new storage
  mol = m;
  nAtoms = m->nAtoms;
  if(!color)
    color = new int[nAtoms];

  if (need_recalc_minmax) {
    //initialize color scale for Color Methods that don't support them 
    minRange = 0.;  
    maxRange = 0.;
  }
  
  // check for special cases
  if(colorMethod == MOLECULE) {
    char buf[20];
    sprintf(buf, "%d", mol->id());
    dindex = scene->category_item_value(molList->colorCatIndex[MLCAT_MOLECULES],
        buf);
    colIndex = dindex;

    for(int i=0; i < nAtoms; color[i++] = dindex) ;

  } else if(colorMethod == COLORID) {
    dindex = colIndex;
    for(int i=0; i < nAtoms; i++) {
      color[i] = dindex;
    }
  } else if(colorMethod == THROB) {
    double timeval, t;
    t = time_of_day();
    timeval = fmod((t / 2.0), 1.0) * 254; 

    // loop every 255 time steps
    float scalefac = (float)(MAPCLRS-1) / 255.0f;
    dindex = MAPCOLOR((int)(0.5 + timeval*scalefac));

    for(int i=0; i < nAtoms; i++) {
      color[i] = dindex;
    }
  } else if(colorMethod == PHYSICALTIME) {
    dindex = MAPCOLOR(MAPCLRS/2);

    // if a timestep is current and we have a nonzero range, assign a color
    // otherwise set coloring to midrange
    Timestep *ts = mol->current();
    if (ts && maxRange > minRange) {
      float scalefac = (float)(MAPCLRS-1) / (maxRange - minRange);
      int dindex2 = (int)(scalefac * ts->physical_time - minRange);

      // dindex2 might be out of range because of user-specified min/max,
      // so clamp it.
      if (dindex2 < 0)
        dindex2 = 0;
      else if (dindex2 >= MAPCLRS) 
        dindex2 = MAPCLRS-1;

      dindex = MAPCOLOR(dindex2);
    }

    // assign the final color
    for(int i=0; i < nAtoms; i++) {
      color[i] = dindex;
    }
  } else if(colorMethod == TIMESTEP) {
    // compute index so first/last timesteps map to color scale endpoints
    float scalefac = (float)(MAPCLRS-1) / ((float) (mol->numframes()-1));
    dindex = MAPCOLOR((int)(0.5f + (mol->frame() * scalefac)));
    if (dindex >= MAPCLRS)
      dindex = MAPCLRS-1; // clamp range to end of color scale

    for(int i=0; i < nAtoms; i++) {
      color[i] = dindex;
    }
  } else if(colorMethod == VOLUME) {
    // Must use a white color so that the 3-D texture map can be 
    // 'modulated' onto the lit surface, preserving shading results
    dindex = scene->nearest_index(1.0, 1.0, 1.0); 
    for(int i=0; i < nAtoms; i++) {
      color[i] = dindex;
    }

    // Get min/max values from selected volume
    if (need_recalc_minmax && mol->num_volume_data()) {
      VolumetricData *v = mol->modify_volume_data(volIndex);
      if (v != NULL) {
        v->datarange(minRange, maxRange);
      }

      need_recalc_minmax = FALSE;
    }
  } else if(colorMethod == BACKBONE) {
    int regc = scene->category_item_value(molList->colorCatIndex[MLCAT_SPECIAL],
    		"Nonback");
    int proc = scene->category_item_value(molList->colorCatIndex[MLCAT_SPECIAL],
    		"Proback");
    int dnac = scene->category_item_value(molList->colorCatIndex[MLCAT_SPECIAL],
    		"Nucback");
    for (int i=0; i < nAtoms; i++) {
      MolAtom *a = mol->atom(i);
      int c = regc;
      if (a->atomType == ATOMPROTEINBACK ||
          a->atomType == ATOMNUCLEICBACK) {  // is this a backbone atom?
        // only color it as a backbone atom if it is actually connected 
        // to other backbone atoms
        // XXX why don't we just trust the value of atomType?
        for(int j=0; j < a->bonds; j++) {
          int bondtype = mol->atom(a->bondTo[j])->atomType;
          if (bondtype == ATOMPROTEINBACK) {
            c = proc;
            break;
          } else if (bondtype == ATOMNUCLEICBACK) {
            c = dnac;
            break;
          }
        }
      }
      color[i] = c;
    }
  } else if (colorMethod == STRUCTURE) {
    int ind = molList->colorCatIndex[MLCAT_SSTRUCT];
    int alpha_helix = scene->category_item_value(ind, "Alpha Helix");
    int helix_3_10 = scene->category_item_value(ind, "3_10_Helix");
    int pi_helix = scene->category_item_value(ind, "Pi_Helix");
    int extended_beta = scene->category_item_value(ind, "Extended_Beta");
    int bridge_beta = scene->category_item_value(ind, "Bridge_Beta");
    int turn = scene->category_item_value(ind, "Turn");
    int coil = scene->category_item_value(ind, "Coil");
    mol->need_secondary_structure(1); // make sure I have secondary structure
    for (int i=0; i<nAtoms; i++) {
      Residue *a = mol->residue(mol->atom(i)->uniq_resid);
      switch (a->sstruct) {
      case SS_HELIX_ALPHA: color[i] = alpha_helix; break;
      case SS_HELIX_3_10: color[i] = helix_3_10; break;
      case SS_HELIX_PI: color[i] = pi_helix; break;
      case SS_BETA: color[i] = extended_beta; break;
      case SS_BRIDGE: color[i] = bridge_beta; break;
      case SS_TURN: color[i] = turn; break;
      case SS_COIL: 
      default: color[i] = coil; break;
      }
    }
  } else if(colorMethod == RESID) {
    for(int i=0; i < nAtoms; i++) {
      dindex = (mol->atom(i))->resid % VISCLRS;
      while (dindex < 0) dindex += VISCLRS;
      color[i] = dindex;
    }
    
  } else if ((colorMethod >= BETA && colorMethod <= INDEX) ||
              colorMethod == VELOCITY) {
    // These all color floating point ranges.
    // The method is:
    //  1) get the array of values from either
    //    a) the symbol table
    //    b) compute the distance value
    //  2) find the min/max
    //  3) set the colors
    // If no timestep, just color by molecule
    
    // first find current timestep
    const Timestep *ts = mol->current();
    if (!ts) {
      // no timestep is current; set coloring by molecule and return
      ColorMethod oldMethod = colorMethod;
      colorMethod = MOLECULE;
      find(mol);
      colorMethod = oldMethod;
      return TRUE;
    }
    // get the data and find the min/max values
    float data_min, data_max;
    float *data = new float[nAtoms];
    if (colorMethod == POS) { 
      float *pos = mol->current()->pos;
      float cov[3];
      float tmp[3];

      mol->cov(cov[0], cov[1], cov[2]);
      vec_sub(tmp, cov, pos);
      data[0] = data_min = data_max = norm(tmp);
      for (int i=1; i<nAtoms; i++) {
        pos += 3;
        vec_sub(tmp, cov, pos);
        data[i] = norm(tmp);
        if (data_min > data[i]) data_min = data[i];
        if (data_max < data[i]) data_max = data[i];
      }
    } else if (colorMethod == POSX) { 
      float *pos = mol->current()->pos;
      data[0] = data_min = data_max = pos[0];
      for (int i=1; i<nAtoms; i++) {
        pos += 3;
        data[i] = pos[0];
        if (data_min > data[i]) data_min = data[i];
        if (data_max < data[i]) data_max = data[i];
      }
    } else if (colorMethod == POSY) { 
      float *pos = mol->current()->pos;
      data[0] = data_min = data_max = pos[1];
      for (int i=1; i<nAtoms; i++) {
        pos += 3;
        data[i] = pos[1];
        if (data_min > data[i]) data_min = data[i];
        if (data_max < data[i]) data_max = data[i];
      }
    } else if (colorMethod == POSZ) { 
      float *pos = mol->current()->pos;
      data[0] = data_min = data_max = pos[2];
      for (int i=1; i<nAtoms; i++) {
        pos += 3;
        data[i] = pos[2];
        if (data_min > data[i]) data_min = data[i];
        if (data_max < data[i]) data_max = data[i];
      }
    } else if (colorMethod == FRAGMENT) {
      data_min = 0;
      data_max = (float) (mol->nFragments-1);
      for (int i=0; i<mol->nAtoms; i++) {
        data[i] = (float) mol->residue(mol->atom(i)->uniq_resid)->fragment;
      }
    } else if (colorMethod == INDEX) {
      data_min = 0;
      data_max = (float) (nAtoms-1);
      for (int i=0; i<nAtoms; i++) 
        data[i] = (float) i;
    } else if (colorMethod == USER || colorMethod == USER2 || 
               colorMethod == USER3 || colorMethod == USER4) {
      const float *user = NULL;
      if (colorMethod == USER)
        user = mol->current()->user;
      else if (colorMethod == USER2)
        user = mol->current()->user2;
      else if (colorMethod == USER3)
        user = mol->current()->user3;
      else if (colorMethod == USER4)
        user = mol->current()->user4;

      if (!user) {
        memset(data, 0, nAtoms*sizeof(float));
        data_min = data_max = 0;
      } else {
        memcpy(data, user, nAtoms*sizeof(float));
#if 1
        minmax_1fv_aligned(data, nAtoms, &data_min, &data_max);
#else
        data_min = data_max = data[0];
        for (int i=1; i<nAtoms; i++) {
          if (data_min > data[i]) data_min = data[i];
          if (data_max < data[i]) data_max = data[i];
        }
#endif
      }
    } else if (colorMethod == VELOCITY) { 
      float *vel = mol->current()->vel;

      if (!vel) {
        memset(data, 0, nAtoms*sizeof(float));
        data_min = data_max = 0;
      } else {
        data[0] = data_min = data_max = norm(vel);
        for (int i=1; i<nAtoms; i++) {
          vel += 3;
          data[i] = norm(vel);
          if (data_min > data[i]) data_min = data[i];
          if (data_max < data[i]) data_max = data[i];
        }
      }
    } else {
      const float *atomfield;
      switch (colorMethod) {
        case BETA: atomfield = mol->beta(); break;
        case OCCUP: atomfield = mol->occupancy(); break;
        case MASS: atomfield = mol->mass(); break;
        case CHARGE: atomfield = mol->charge(); break;
        default: atomfield = mol->mass(); 
      }
      // XXX memcpy because we always free later...
      memcpy(data, atomfield, nAtoms*sizeof(float));
#if 1
      minmax_1fv_aligned(data, nAtoms, &data_min, &data_max);
#else
      data_min = data_max = data[0];
      for (int i=1; i < nAtoms; i++) {
        if (data_min > data[i]) data_min = data[i];
        if (data_max < data[i]) data_max = data[i];
      }
#endif
    }

    if (need_recalc_minmax) {
      minRange = data_min;
      maxRange = data_max;
      need_recalc_minmax = FALSE;
    } else {
      data_min = minRange;
      data_max = maxRange;
    }

    // 3) define the colors
    if (data_min == data_max) {
      for (int i=0; i<nAtoms; i++) {
        color[i] = MAPCOLOR(MAPCLRS/2);
      }
    } else {
      float scalefac = (float)(MAPCLRS-1) / (data_max - data_min);
      int dindex2;
      for (int i=0; i<nAtoms; i++) {
        dindex2 = (int)(scalefac * (data[i] - data_min));
        // dindex2 might be out of range because of user-specified min/max,
        // so clamp it.
        if (dindex2 < 0)
          dindex2 = 0;
        else if (dindex2 >= MAPCLRS) 
          dindex2 = MAPCLRS-1;
        color[i] = MAPCOLOR(dindex2);
      }
    }
    delete [] data;

  } else if(colorMethod == NAME) {
    int ind = molList->colorCatIndex[MLCAT_NAMES];
    for (int i=0; i < nAtoms; i++) {
      dindex = scene->category_item_value(ind,
          (mol->atomNames).data((mol->atom(i))->nameindex));
      color[i] = dindex;
    }
    
  } else if(colorMethod == CONFORMATION) {
    int i;
    int ind = molList->colorCatIndex[MLCAT_CONFORMATIONS];
    int allconf = scene->category_item_value(ind, "all");

    NameList<int> *molnames = &(mol->altlocNames);
    int alltypecode = molnames->typecode("");

    for(i=0; i < nAtoms; i++) {
      int atomidx = (mol->altlocNames).data((mol->atom(i))->altlocindex);
      if (atomidx == alltypecode)
        dindex = allconf;
      else
        dindex = scene->category_item_value(ind, atomidx);
      color[i] = dindex;
    }
    
  } else if(colorMethod == TYPE) {
    int ind = molList->colorCatIndex[MLCAT_TYPES];
    for(int i=0; i < nAtoms; i++) {
      dindex = scene->category_item_value(ind,
          (mol->atomTypes).data((mol->atom(i))->typeindex));
      color[i] = dindex;
    }

  } else if(colorMethod == ELEMENT) {
    int ind = molList->colorCatIndex[MLCAT_ELEMENTS];
    for(int i=0; i < nAtoms; i++) {
      dindex = scene->category_item_value(ind, 
          get_pte_label(mol->atom(i)->atomicnumber));
      color[i] = dindex;
    }
    
  } else if(colorMethod == RESNAME) {
    int ind = molList->colorCatIndex[MLCAT_RESNAMES];
    for(int i=0; i < nAtoms; i++) {
      dindex=scene->category_item_value(ind,
          (mol->resNames).data((mol->atom(i))->resnameindex));
      color[i] = dindex;
    }
    
  } else if(colorMethod == RESTYPE) {
    int ind = molList->colorCatIndex[MLCAT_RESTYPES];
    for(int i=0; i < nAtoms; i++) {
      dindex=scene->category_item_value(ind,
		(molList->resTypes).data((mol->resNames).name(mol->atom(i)->resnameindex)));
      color[i] = dindex;
    }
    
  } else if(colorMethod == CHAIN) {
    int ind = molList->colorCatIndex[MLCAT_CHAINS];
    for(int i=0; i < nAtoms; i++) {
      dindex=scene->category_item_value(ind,
          (mol->chainNames).data((mol->atom(i))->chainindex));
      color[i] = dindex;
    }

  } else if(colorMethod == SEGNAME) {
    int ind = molList->colorCatIndex[MLCAT_SEGNAMES];
    for(int i=0; i < nAtoms; i++) {
      dindex=scene->category_item_value(ind,
          (mol->segNames).data((mol->atom(i))->segnameindex));
      color[i] = dindex;
    }

  } else {
    msgErr << "Unknown coloring method " << (int)colorMethod << " in AtomColor.";
    msgErr << sendmsg;
    return FALSE;
  }
    
  return TRUE;
}

int AtomColor::uses_colorscale() const {
  return ((colorMethod >= BETA && colorMethod <= INDEX) || 
           colorMethod == VOLUME);
}

