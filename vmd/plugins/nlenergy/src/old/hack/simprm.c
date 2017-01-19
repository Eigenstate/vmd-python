/* simprm.c */

#include <string.h>
#include <ctype.h>
#include "moltypes/forceprm.h"
#include "moltypes/simprm.h"


typedef struct SimPrmParse_t {
  const char *keyword;
  int (*valparse)(void *field, const char *value);
  int byteoffset;
} SimPrmParse;

/* parse routines */
static int parse_boolean(void *field, const char *value);
static int parse_int32(void *field, const char *value);
static int parse_dreal(void *field, const char *value);
static int parse_dvec(void *field, const char *value);
static int parse_string(void *field, const char *value);
static int parse_strarray(void *field, const char *value);
static int parse_exclude(void *field, const char *value);
static int parse_rigidbonds(void *field, const char *value);
  /** declare more parse_ routines if these are insufficient **/

/* calculate relative data field offsets */
#define OFFSET(field)  (((char*)(&(((SimPrm*)NULL)->field))) - ((char*)NULL))

/* fill out one entry per SimPrm data field:
 * { keyword, parse_routine, OFFSET(spm,keyword) }
 *   + order of data fields doesn't matter (lookup uses hash table)
 *   + case doesn't matter (lookup is case insensitive)
 */
static const SimPrmParse ParseArray[] = {
  /* input files */
  { "coordinates", parse_string, OFFSET(coordinates) },
  { "structure", parse_string, OFFSET(structure) },
  { "parameters", parse_strarray, OFFSET(parameters) },
  { "paraTypeXplor", parse_boolean, OFFSET(paraTypeXplor) },
  { "paraTypeCharmm", parse_boolean, OFFSET(paraTypeCharmm) },
  { "velocities", parse_string, OFFSET(velocities) },
  { "binvelocities", parse_string, OFFSET(binvelocities) },
  { "bincoordinates", parse_string, OFFSET(bincoordinates) },
  { "cwd", parse_string, OFFSET(cwd) },

  /* output files */
  { "outputname", parse_string, OFFSET(outputname) },
  { "binaryoutput", parse_boolean, OFFSET(binaryoutput) },
  { "restartname", parse_string, OFFSET(restartname) },
  { "restartfreq", parse_int32, OFFSET(restartfreq) },
  { "restartsave", parse_boolean, OFFSET(restartsave) },
  { "binaryrestart", parse_boolean, OFFSET(binaryrestart) },
  { "dcdfile", parse_string, OFFSET(dcdfile) },
  { "dcdfreq", parse_int32, OFFSET(dcdfreq) },
  { "dcdUnitCell", parse_boolean, OFFSET(dcdUnitCell) },
  { "veldcdfile", parse_string, OFFSET(veldcdfile) },
  { "veldcdfreq", parse_int32, OFFSET(veldcdfreq) },

  /* standard output */
  { "outputEnergies", parse_int32, OFFSET(outputEnergies) },
  { "mergeCrossterms", parse_boolean, OFFSET(mergeCrossterms) },
  { "outputMomenta", parse_int32, OFFSET(outputMomenta) },
  { "outputPressure", parse_int32, OFFSET(outputPressure) },
  { "outputTiming", parse_int32, OFFSET(outputTiming) },

  /* timestep parameters */
  { "numsteps", parse_int32, OFFSET(numsteps) },
  { "timestep", parse_dreal, OFFSET(timestep) },
  { "firsttimestep", parse_int32, OFFSET(firsttimestep) },
  { "stepspercycle", parse_int32, OFFSET(stepspercycle) },

  /* simulation space partitioning */
  { "cutoff", parse_dreal, OFFSET(cutoff) },
  { "switching", parse_boolean, OFFSET(switching) },
  { "switchdist", parse_dreal, OFFSET(switchdist) },
  { "limitdist", parse_dreal, OFFSET(limitdist) },
  { "pairlistdist", parse_dreal, OFFSET(pairlistdist) },
  { "hgroupCutoff", parse_dreal, OFFSET(hgroupCutoff) },
  { "margin", parse_dreal, OFFSET(margin) },

  /* basic dynamics */
  { "exclude", parse_exclude, OFFSET(exclude) },
  { "temperature", parse_dreal, OFFSET(temperature) },
  { "comMotion", parse_boolean, OFFSET(comMotion) },
  { "zeroMomentum", parse_boolean, OFFSET(zeroMomentum) },
  { "dielectric", parse_dreal, OFFSET(dielectric) },
  { "nonbondedScaling", parse_dreal, OFFSET(nonbondedScaling) },
  { "1-4scaling", parse_dreal, OFFSET(scaling14) },
  { "vdwGeometricSigma", parse_boolean, OFFSET(vdwGeometricSigma) },
  { "seed", parse_int32, OFFSET(seed) },
  { "rigidBonds", parse_rigidbonds, OFFSET(rigidBonds) },
  { "rigidTolerance", parse_dreal, OFFSET(rigidTolerance) },
  { "rigidIterations", parse_int32, OFFSET(rigidIterations) },
  { "rigidDieOnError", parse_boolean, OFFSET(rigidDieOnError) },
  { "useSettle", parse_boolean, OFFSET(useSettle) },

  /* full direct parameters */
  { "fulldirect", parse_boolean, OFFSET(fulldirect) },

  /* periodic boundary conditions */
  { "cellBasisVector1", parse_dvec, OFFSET(cellBasisVector1) },
  { "cellBasisVector2", parse_dvec, OFFSET(cellBasisVector2) },
  { "cellBasisVector3", parse_dvec, OFFSET(cellBasisVector3) },
  { "cellOrigin", parse_dvec, OFFSET(cellOrigin) },
  { "extendedSystem", parse_string, OFFSET(extendedSystem) },
  { "xstfile", parse_string, OFFSET(xstfile) },
  { "xstfreq", parse_int32, OFFSET(xstfreq) },
  { "wrapWater", parse_boolean, OFFSET(wrapWater) },
  { "wrapAll", parse_boolean, OFFSET(wrapAll) },
  { "wrapNearest", parse_boolean, OFFSET(wrapNearest) },

  /** append additional SimPrm fields **/
};

/* parse routine definitions */
int parse_boolean(void *field, const char *value) {
  int s;
  if ((s=String_boolean((boolean *)field, value)) != OK) return ERROR(s);
  return OK;
}

int parse_int32(void *field, const char *value) {
  int s;
  if ((s=String_int32((int32 *)field, value)) != OK) return ERROR(s);
  return OK;
}

int parse_dreal(void *field, const char *value) {
  int s;
  if ((s=String_dreal((dreal *)field, value)) != OK) return ERROR(s);
  return OK;
}

int parse_dvec(void *field, const char *value) {
  int s;
  if ((s=String_dvec((dvec *)field, value)) != OK) return ERROR(s);
  return OK;
}

int parse_string(void *field, const char *value) {
  int s;
  if ((s=String_set((String *)field, value)) != OK) return ERROR(s);
  return OK;
}

int parse_strarray(void *field, const char *value) {
  int s;
  if ((s=Strarray_append_cstr((Strarray*)field, value)) != OK) return ERROR(s);
  return OK;
}

int parse_exclude(void *field, const char *value) {
  int32 *excl = (int32 *)field;
  if (strcasecmp(value, "none") == 0) *excl = EXCL_NONE;
  else if (strcasecmp(value, "1-2") == 0) *excl = EXCL_12;
  else if (strcasecmp(value, "1-3") == 0) *excl = EXCL_13;
  else if (strcasecmp(value, "1-4") == 0) *excl = EXCL_14;
  else if (strcasecmp(value, "scaled1-4") == 0) *excl = EXCL_SCALED14;
  else return ERROR(ERR_VALUE);
  return OK;
}

int parse_rigidbonds(void *field, const char *value) {
  return OK;
}

  /** define more parse_ routines if these are insufficient **/


/* set default values for SimPrm */
int set_default_values(SimPrm *p) {
  int s;
  if ((s=String_set(&(p->cwd), "./")) != OK) return ERROR(s);
  p->outputEnergies = 1;
  p->dielectric = 1.;
  p->scaling14 = 1.;
  /** add more default values **/
  return OK;
}


/**************************************************************************
 * nothing below needs to be modified
 **************************************************************************/


#define NELEMS(arr)  (sizeof(arr)/sizeof(arr[0]))

static const void *map_key(const Array *a, int32 i) {
  const SimPrmParse *p = Array_data_const(a);
  return (const void *)(p[i].keyword);
}

static int32 map_hash(const Arrmap *amap, const void *vkey) {
  const char *key = (const char *) vkey;
  int32 i=0;
  int32 hashvalue;
  while (*key != '\0') {
    int lc = tolower(*key);
    i = (i<<3) + (lc - '0');  /* hashing is case insensitive */
    key++;
  }
  hashvalue = (((i*1103515249)>>amap->downshift) & amap->mask);
  if (hashvalue < 0) {
    hashvalue = 0;
  }
  return hashvalue;
}

static int32 map_keycmp(const void *vkey1, const void *vkey2) {
  const char *key1 = (const char *) vkey1;
  const char *key2 = (const char *) vkey2;
  return (int32) strcasecmp(key1, key2);  /* lookup is case insensitive */
}


int SimPrm_init(SimPrm *p) {
  Array *parselist = &(p->parselist);
  Arrmap *parsetable = &(p->parsetable);
  int s, i;

  memset(p, 0, sizeof(SimPrm));

  if ((s=Array_init(parselist, sizeof(SimPrmParse))) != OK) {
    return ERROR(s);
  }
  if ((s=Array_setbuflen(parselist, NELEMS(ParseArray))) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(parsetable, parselist,
          map_key, map_hash, map_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  for (i = 0;  i < NELEMS(ParseArray);  i++) {
    if ((s=Array_append(parselist, &ParseArray[i])) != OK) return ERROR(s);
    if ((s=Arrmap_insert(parsetable, i)) != i) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_EXPECT));
    }
  }

  /* input file names */
  if ((s=String_init(&(p->coordinates))) != OK) return ERROR(s);
  if ((s=String_init(&(p->structure))) != OK) return ERROR(s);
  if ((s=Strarray_init(&(p->parameters))) != OK) return ERROR(s);
  if ((s=String_init(&(p->velocities))) != OK) return ERROR(s);
  if ((s=String_init(&(p->binvelocities))) != OK) return ERROR(s);
  if ((s=String_init(&(p->bincoordinates))) != OK) return ERROR(s);
  if ((s=String_init(&(p->cwd))) != OK) return ERROR(s);

  /* output file names */
  if ((s=String_init(&(p->outputname))) != OK) return ERROR(s);
  if ((s=String_init(&(p->restartname))) != OK) return ERROR(s);
  if ((s=String_init(&(p->dcdfile))) != OK) return ERROR(s);
  if ((s=String_init(&(p->veldcdfile))) != OK) return ERROR(s);

  /* additional file names */
  if ((s=String_init(&(p->consref))) != OK) return ERROR(s);
  if ((s=String_init(&(p->conskfile))) != OK) return ERROR(s);
  if ((s=String_init(&(p->fixedAtomsFile))) != OK) return ERROR(s);
  if ((s=String_init(&(p->langevinFile))) != OK) return ERROR(s);
  if ((s=String_init(&(p->tCoupleFile))) != OK) return ERROR(s);
  if ((s=String_init(&(p->extendedSystem))) != OK) return ERROR(s);
  if ((s=String_init(&(p->xstfile))) != OK) return ERROR(s);

  if ((s=set_default_values(p)) != OK) return ERROR(s);

  return OK;
}


void SimPrm_done(SimPrm *p) {
  Arrmap_done(&(p->parsetable));
  Array_done(&(p->parselist));

  /* input files */
  String_done(&(p->coordinates));
  String_done(&(p->structure));
  Strarray_done(&(p->parameters));
  String_done(&(p->velocities));
  String_done(&(p->binvelocities));
  String_done(&(p->bincoordinates));
  String_done(&(p->cwd));

  /* output files */
  String_done(&(p->outputname));
  String_done(&(p->restartname));
  String_done(&(p->dcdfile));
  String_done(&(p->veldcdfile));

  /* additional files */
  String_done(&(p->consref));
  String_done(&(p->conskfile));
  String_done(&(p->fixedAtomsFile));
  String_done(&(p->langevinFile));
  String_done(&(p->tCoupleFile));
  String_done(&(p->extendedSystem));
  String_done(&(p->xstfile));
}


int SimPrm_set(SimPrm *p, const char *key, const char *val) {
  const SimPrmParse *sp = Array_data_const(&(p->parselist));
  int i, s;
  if ((i=Arrmap_lookup(&(p->parsetable), key)) < OK) {
    return (i < FAIL ? ERROR(i) : FAIL);
  }
  if ((s=sp[i].valparse(((char *)p)+sp[i].byteoffset, val)) != OK) {
    return ERROR(s);
  }
  return OK;
}
