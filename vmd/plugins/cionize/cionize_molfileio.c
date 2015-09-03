#include "cionize_grid.h"
#include "getplugins.h"
#include "cionize_structs.h"
#include "cionize_molfileio.h"
#include <string.h>
#include <math.h>

/* Do all the work to set up the api finder and such */
void setup_apis(cionize_api* molfiles) {
  init_plugins();
  molfiles->filetype[0]='\0';
  molfiles->optflags = 0;
}

/* Get the right input api for our structure and make sure it works */
int setup_inapi(const cionize_params* params, cionize_api* molfiles) {
  molfile_plugin_t* inapi;

  if (strlen(molfiles->filetype) > 0) {
    inapi = get_plugin(molfiles->filetype);
  } else {
    inapi = get_plugin_forfile(params->pdbin, molfiles->filetype);
  }

  if (!inapi || !(inapi->read_structure) || !(inapi->read_next_timestep)) {
    fprintf(stderr, "ERROR: Can't read the proper information from the input file type %s.\n", molfiles->filetype);
    return 1;
  }

  molfiles->inapi = inapi;
  return 0;
}

/* Open and read the input file into a molfile representation, 
 * and get the necessary information about it */
int read_input(cionize_params* params, cionize_api* molfiles, cionize_molecule* molecule, cionize_grid* grid) {

  int natom;
  int optflags;
  
  optflags = MOLFILE_BADOPTIONS;

  molfiles->f_in = molfiles->inapi->open_file_read(params->pdbin, molfiles->filetype, &natom);
  if (molfiles->f_in == NULL) {
    fprintf(stderr, "ERROR: Unable to open input file. Exiting...\n");
    return 1;
  }
  molecule->natom = natom;

  molfiles->atomstruct = (molfile_atom_t *)malloc(natom * sizeof(molfile_atom_t));
  molfiles->atomts.coords = (float *)malloc(3 * natom * sizeof(float));

  molfiles->inapi->read_structure(molfiles->f_in, &optflags, molfiles->atomstruct);
  if (optflags == MOLFILE_BADOPTIONS) {
    fprintf(stderr, "ERROR: Molfile plugin didn't work correctly\n");
    return 1;
  }

  molfiles->inapi->read_next_timestep(molfiles->f_in, natom, &(molfiles->atomts));

  if (!(optflags & MOLFILE_CHARGE)) {
    fprintf(stderr, "ERROR: No charges read from input file.\n");
    return 1;
  }
  molfiles->inapi->close_file_read(molfiles->f_in);

  /* Find the maximum extents of our structure */
  if (get_coord_info(params, molfiles, grid, molecule) != 0) return 1;

  return 0;
}

int output_ions(cionize_params* params, cionize_api* molfiles, cionize_grid* grid, const char* pdbout, const int* ionpos) {
  /* Output all of the newly placed ions using the plugin from outapi */
  molfile_atom_t* ionstruct;
  molfile_timestep_t ioncoords;
  float* coordarray;
  float gridspacing;

  void* f_out;

  int i;
  float x,y,z;
  int optflags;

  optflags = MOLFILE_CHARGE;

  gridspacing = grid->gridspacing;

  /* Open our output file */
  f_out = molfiles->outapi->open_file_write(pdbout, molfiles->filetype, params->nion);

  /* Allocate arrays for the molfile equivalent of our ions */
  ionstruct = malloc(params->nion * sizeof(molfile_atom_t));
  coordarray = malloc(params->nion * 3 * sizeof(float));
  if (ionstruct==NULL || coordarray==NULL) {
    fprintf(stderr, "Error: Couldn't allocate memory for temporary arrays\n");
    return 1;
  }

  ioncoords.coords = coordarray;

  /* Give some sensible cell information */
  ioncoords.A = ioncoords.B = ioncoords.C = 0.0;
  ioncoords.alpha = ioncoords.beta = ioncoords.gamma = 90.0;

  for (i=0; i<params->nion; i++) {
    /* Fill in the information in the structure and coordinate arrays */
    x = ((float) (ionpos[3*i] * gridspacing)) + grid->minx;
    y = ((float) (ionpos[3*i+1] * gridspacing)) + grid->miny;
    z = ((float) (ionpos[3*i+2] * gridspacing)) + grid->minz;

    strncpy((ionstruct[i]).name, params->ionname, 4);
    strncpy((ionstruct[i]).type, params->ionname, 4);
    strncpy((ionstruct[i]).resname, params->ionname, 4);
    ionstruct[i].resid = i + 1;
    strncpy((ionstruct[i]).segid, params->ionname, 4);
    strncpy((ionstruct[i]).chain, " ", 2);
    ionstruct[i].charge = params->ioncharge;
    ionstruct[i].bfactor = 0.0;
    ionstruct[i].occupancy = 1.0;

    coordarray[3*i] = x;
    coordarray[3*i + 1] = y;
    coordarray[3*i + 2] = z;
  }

  /* Actually write our structure and coordinates */
  molfiles->outapi->write_structure(f_out, optflags, ionstruct);
  molfiles->outapi->write_timestep(f_out, &ioncoords);

  /* Clean up */
  if (coordarray != NULL) free(coordarray); 
  if (ionstruct != NULL) free(ionstruct);
  molfiles->outapi->close_file_write(f_out);

  return 0;
}

int unpack_molfile_arrays(const cionize_params* params, cionize_api* molfiles, cionize_molecule* molecule, cionize_grid* grid) {

  int i;
  float x,y,z;
  long int xi, yi, zi;
  float gridspacing;
  float* atoms;
  long int* atomis;
  const molfile_timestep_t* ts = &(molfiles->atomts);
  const molfile_atom_t* structs = molfiles->atomstruct;

  atoms = molecule->atoms;
  atomis = molecule->atomis;
  gridspacing = grid->gridspacing;

  /*Loop through the atoms and fill in our arrays*/
  for(i=0; i<molecule->natom; i += 1) {
    x = (ts->coords)[3*i] - grid->minx;
    y = (ts->coords)[3*i + 1] - grid->miny;
    z = (ts->coords)[3*i + 2] - grid->minz;

    xi = (long int) rint(x/gridspacing);
    yi = (long int) rint(y/gridspacing);
    zi = (long int) rint(z/gridspacing);

    atoms[4*i] = x;
    atoms[4*i + 1] = y;
    atoms[4*i + 2] = z;
    atoms[4*i + 3] = structs[i].charge;

    atomis[3*i] = xi;
    atomis[3*i + 1] = yi;
    atomis[3*i + 2] = zi;
  }

  return 0;
}

int get_coord_info(const cionize_params* params, cionize_api* molfiles, cionize_grid* grid, cionize_molecule* molecule) {
  /*Pass through the input once, find the min and max values of all
   * coordinates, and give us the number of atoms and grid points
   */

  int i;
  float x, y, z;
  float maxx, maxy, maxz, minx, miny, minz;
  float* coordarray;
  float bordersize;
  float gridspacing;

  maxx = maxy = maxz = -HUGE_VAL;
  minx = miny = minz = HUGE_VAL;
  coordarray = (molfiles->atomts).coords;
  bordersize = params->bordersize;
  gridspacing = grid->gridspacing;

  if (params->expsize == 0) {


    for (i=0; i<molecule->natom; i += 1) {
      x = coordarray[3*i];
      y = coordarray[3*i + 1];
      z = coordarray[3*i + 2];

      if (x>maxx) maxx = x;
      if (y>maxy) maxy = y;
      if (z>maxz) maxz = z;
      if (x<minx) minx = x;
      if (y<miny) miny = y;
      if (z<minz) minz = z;
    }

    /* Find the number of grid points needed */
    maxx += bordersize;
    maxy += bordersize;
    maxz += bordersize;
    minx -= bordersize;
    miny -= bordersize;
    minz -= bordersize;
    x = (maxx - minx);
    y = (maxy - miny);
    z = (maxz - minz);
    grid->minx = minx;
    grid->miny = miny;
    grid->minz = minz;
    grid->maxx=maxx;
    grid->maxy = maxy;
    grid->maxz = maxz;
  } else {
    x = (grid->maxx - grid->minx);
    y = (grid->maxy - grid->miny);
    z = (grid->maxz - grid->minz);
  }


  grid->numpt = (long int) rint(x/gridspacing) + 1;
  grid->numcol = (long int) rint(y/gridspacing) + 1;
  grid->numplane = (long int) rint(z/gridspacing) + 1;


  return 0;
}

