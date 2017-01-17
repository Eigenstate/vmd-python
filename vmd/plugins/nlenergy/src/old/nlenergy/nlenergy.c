#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/tclwrap.h"


int NLEnergy_init(NLEnergy *p) {
  int s;  /* error status */

  memset(p, 0, sizeof(NLEnergy));

  if ((s=Array_init(&(p->extatomid), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->atomid), sizeof(int32))) != OK) return ERROR(s);

  if ((s=ForcePrm_init(&(p->fprm))) != OK) return ERROR(s);
  if ((s=Topology_init(&(p->topo), &(p->fprm))) != OK) return ERROR(s);
  if ((s=Exclude_init(&(p->exclude), &(p->topo))) != OK) return ERROR(s);
  if ((s=PdbAtom_init(&(p->pdbpos))) != OK) return ERROR(s);
  if ((s=Coord_init(&(p->coord))) != OK) return ERROR(s);
  if ((s=Energy_init(&(p->ener))) != OK) return ERROR(s);
  if ((s=Fbonded_init(&(p->fbon), &(p->topo))) != OK) return ERROR(s);
  if ((s=Fnbcut_init(&(p->fnbcut), &(p->exclude))) != OK) return ERROR(s);

  if ((s=Array_init(&(p->atomsel), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->nonbsel), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->bondsel), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->anglesel), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->dihedsel), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->imprsel), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->invsel), sizeof(char))) != OK) return ERROR(s);

  if ((s=Array_init(&(p->idnonb), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->idnonb_b), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->idnbvdw), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->idnbvdw_b), sizeof(int32))) != OK) return ERROR(s);

  return OK;
}


void NLEnergy_done(NLEnergy *p) {
  NL_free(p->aselname);

  Array_done(&(p->extatomid));
  Array_done(&(p->atomid));

  ForcePrm_done(&(p->fprm));
  Topology_done(&(p->topo));
  Exclude_done(&(p->exclude));
  PdbAtom_done(&(p->pdbpos));
  Coord_done(&(p->coord));
  Energy_done(&(p->ener));
  Fbonded_done(&(p->fbon));
  Fnbcut_done(&(p->fnbcut));

  Array_done(&(p->atomsel));
  Array_done(&(p->bondsel));
  Array_done(&(p->anglesel));
  Array_done(&(p->dihedsel));
  Array_done(&(p->imprsel));
  Array_done(&(p->invsel));

  Array_done(&(p->idnonb));
  Array_done(&(p->idnonb_b));
  Array_done(&(p->idnbvdw));
  Array_done(&(p->idnbvdw_b));
}


static int setup_atomid_map(NLEnergy *p, Tcl_Interp *interp, int32 natoms) {
  char script[64];
  Tcl_Obj *obj;
  Tcl_Obj **objv;
  int32 *atomid, *extatomid;
  int32 atomidlen;
  int objc, i, s;

  INT(natoms);
  if (natoms <= 0) return ERROR(ERR_EXPECT);
  if ((s=Array_resize(&(p->extatomid),natoms)) != OK) return ERROR(s);
  extatomid = Array_data(&(p->extatomid));
  snprintf(script, sizeof(script), "%s list", p->aselname);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0) ||
      NULL==(obj = Tcl_GetObjResult(interp)) ||
      TCL_OK != Tcl_ListObjGetElements(interp, obj, &objc, &objv) ||
      objc != natoms) {
    return ERROR(ERR_EXPECT);
  }
  for (i = 0;  i < objc;  i++) {
    long n;
    if (TCL_OK != Tcl_GetLongFromObj(interp, objv[i], &n)) {
      return ERROR(ERR_EXPECT);
    }
    extatomid[i] = (int32) n;
    ASSERT(0==i || extatomid[i-1] < extatomid[i]);
  }
  ASSERT(i == natoms);
  p->firstid = extatomid[0];
  p->lastid = extatomid[natoms-1];
  INT(p->firstid);
  INT(p->lastid);
  atomidlen = p->lastid - p->firstid + 1;
  ASSERT(atomidlen >= natoms);
  if ((s=Array_resize(&(p->atomid),atomidlen)) != OK) return ERROR(s);
  atomid = Array_data(&(p->atomid));
  for (i = 0;  i < atomidlen;  i++) {  /* initialize */
    atomid[i] = FAIL;
  }
  for (i = 0;  i < natoms;  i++) {
    atomid[ extatomid[i] - p->firstid ] = i;
  }
  return OK;
}


int NLEnergy_setup(NLEnergy *p, Tcl_Interp *interp, int idnum, int molid,
    const char *aselname) {
  char script[64];
  Tcl_Obj *obj;
  int objc;
  Tcl_Obj **objv;
  int natoms, i;
  NonbPrm nonbprm;
  ForcePrm *fprm = &(p->fprm);
  Topology *topo = &(p->topo);
  Coord *coord = &(p->coord);
  int s;
  int nbonds, cnt;
  double alpha, beta, gamma, A, B, C;
  double epsalpha, epsbeta, epsgamma, cosAB, sinAB, cosAC, cosBC;
  dvec bv1, bv2, bv3, orig;  /* basis vectors and origin */
  dvec *pos;

  p->idnum = idnum;
  p->molid = molid;
  /* duplicate atom selection name before returning,
   * use atom selection to init topology and coordinates */
  if (NULL==(p->aselname = NL_strdup(aselname))) return ERROR(ERR_MEMALLOC);

  STR(p->aselname);

  /* set reasonable default nonbonded parameters */
  nonbprm.cutoff = 12;
  nonbprm.switchdist = 10;
  nonbprm.dielectric = 1;
  nonbprm.scaling14 = 1;
  nonbprm.switching = TRUE;
  nonbprm.exclude = EXCL_SCALED14;
  nonbprm.charge_model = CHARGE_FIXED;
  nonbprm.water_model = WATER_TIP3;
  if ((s=ForcePrm_set_nonbprm(fprm, &nonbprm)) != OK) return ERROR(s);
  p->fulldirect = FALSE;
  p->fulldirectvdw = FALSE;

  /* determine number of atoms for this selection */
  snprintf(script, sizeof(script), "%s num", p->aselname);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0)) return ERROR(ERR_EXPECT);
  if (NULL==(obj = Tcl_GetObjResult(interp))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_GetIntFromObj(interp, obj, &natoms)) {
    return ERROR(ERR_EXPECT);
  }

  INT(natoms);

  /* setup map for external atom numbering */
  if ((s=setup_atomid_map(p,interp,natoms)) != OK) return ERROR(ERR_EXPECT);

  /* read Atom data from our VMD atom selection */
  if ((s=Topology_setmaxnum_atom(topo, natoms)) != OK) return ERROR(s);
  snprintf(script, sizeof(script),
      "%s get { mass charge name type residue resname }", p->aselname);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0)) return ERROR(ERR_EXPECT);
  if (NULL==(obj = Tcl_GetObjResult(interp))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
    return ERROR(ERR_EXPECT);
  }
  if (objc != natoms) return ERROR(ERR_EXPECT);
  for (i = 0;  i < natoms;  i++) {
    Atom a;
    int aobjc;
    Tcl_Obj **aobjv;
    int n;
    double d;
    char *str;
    memset(&a, 0, sizeof(Atom));
    if (TCL_OK != Tcl_ListObjGetElements(interp, objv[i], &aobjc, &aobjv)) {
      return ERROR(ERR_EXPECT);
    }
    if (aobjc != 6) return ERROR(ERR_EXPECT);
    if (TCL_OK != Tcl_GetDoubleFromObj(interp, aobjv[0], &d)) {
      return ERROR(ERR_EXPECT);
    }
    a.m = d;  /* atom mass */
    if (TCL_OK != Tcl_GetDoubleFromObj(interp, aobjv[1], &d)) {
      return ERROR(ERR_EXPECT);
    }
    a.q = d;  /* atom charge */
    str = Tcl_GetStringFromObj(aobjv[2], NULL);
    snprintf(a.atomName, sizeof(AtomName), "%s", str);
    str = Tcl_GetStringFromObj(aobjv[3], NULL);
    snprintf(a.atomType, sizeof(AtomType), "%s", str);
    if (TCL_OK != Tcl_GetIntFromObj(interp, aobjv[4], &n)) {
      return ERROR(ERR_EXPECT);
    }
    a.residue = n;
    str = Tcl_GetStringFromObj(aobjv[5], NULL);
    snprintf(a.resName, sizeof(ResName), "%s", str);
    if ((s=Topology_add_atom(topo, &a)) != i) return ERROR(s);
  }

  TEXT("successfully initialized atoms");

  /*
   * read Bond data from VMD atom selection
   * this is stored in adjacency list format where each atom
   * has a list of atoms to which it is bonded
   */
  snprintf(script, sizeof(script), "%s getbonds", p->aselname);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0)) return ERROR(ERR_EXPECT);
  if (NULL==(obj = Tcl_GetObjResult(interp))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
    return ERROR(ERR_EXPECT);
  }
  if (objc != natoms) return ERROR(ERR_EXPECT);
  
  /* first determine number of bonds */
  nbonds = 0;
  for (i = 0;  i < natoms;  i++) {
    int bobjc, j, k;
    Tcl_Obj **bobjv;
    if (TCL_OK != Tcl_ListObjGetElements(interp, objv[i], &bobjc, &bobjv)) {
      return ERROR(ERR_EXPECT);
    }
    /* count only bonds to atoms within this atom selection */
    for (k = 0;  k < bobjc;  k++) {
      if ((j=atomid_from_obj(p,interp,bobjv[k])) >= 0) nbonds++;
      else if (j < FAIL) return ERROR(j);
    }
  }
  nbonds >>= 1;  /* double counted, divide nbonds by 2 */

  INT(nbonds);

  if ((s=Topology_setmaxnum_bond(topo, nbonds)) != OK) return ERROR(s);
  cnt = 0;
  for (i = 0;  i < natoms;  i++) {
    int bobjc, j, k;
    Tcl_Obj **bobjv;
    if (TCL_OK != Tcl_ListObjGetElements(interp, objv[i], &bobjc, &bobjv)) {
      return ERROR(ERR_EXPECT);
    }
    for (k = 0;  k < bobjc;  k++) {
      if ((j=atomid_from_obj(p,interp,bobjv[k])) < FAIL) return ERROR(j);
      if (i < j) {
        Bond b;
        b.atomID[0] = i;
        b.atomID[1] = j;
        if ((s=Topology_add_bond(topo, &b)) < OK) return ERROR(s);
        cnt++;
      }
    }
  }
  if (cnt != nbonds) return ERROR(ERR_EXPECT);

  TEXT("successfully initialized bonds");

  if ((s=Topology_setup_atom_cluster(topo)) != OK) return ERROR(s);
  if ((s=Topology_setup_atom_parent(topo)) != OK) return ERROR(s);
  Topology_reset_status(topo, TOPO_ATOM_ADD | TOPO_BOND_ADD);

  /* get periodic cell information */
  snprintf(script, sizeof(script),
      "molinfo %d get { alpha beta gamma a b c }", p->molid);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0)) return ERROR(ERR_EXPECT);
  if (NULL==(obj = Tcl_GetObjResult(interp))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
    return ERROR(ERR_EXPECT);
  }
  if (objc != 6) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_GetDoubleFromObj(interp, objv[0], &alpha)) {
    return ERROR(ERR_EXPECT);
  }
  if (TCL_OK != Tcl_GetDoubleFromObj(interp, objv[1], &beta)) {
    return ERROR(ERR_EXPECT);
  }
  if (TCL_OK != Tcl_GetDoubleFromObj(interp, objv[2], &gamma)) {
    return ERROR(ERR_EXPECT);
  }
  if (TCL_OK != Tcl_GetDoubleFromObj(interp, objv[3], &A)) {
    return ERROR(ERR_EXPECT);
  }
  if (TCL_OK != Tcl_GetDoubleFromObj(interp, objv[4], &B)) {
    return ERROR(ERR_EXPECT);
  }
  if (TCL_OK != Tcl_GetDoubleFromObj(interp, objv[5], &C)) {
    return ERROR(ERR_EXPECT);
  }
  /* convert from VMD (crystallographic-style cell) to NAMD basis vectors */
  epsalpha = (alpha - 90.0) * RADIANS;
  epsbeta  = (beta  - 90.0) * RADIANS;
  epsgamma = (gamma - 90.0) * RADIANS;
  cosAB = -sin(epsgamma);
  sinAB = cos(epsgamma);
  cosAC = -sin(epsbeta);
  cosBC = -sin(epsalpha);
  bv1.x = A;
  bv1.y = 0;
  bv1.z = 0;
  bv2.x = B * cosAB;
  bv2.y = B * sinAB;
  bv2.z = 0;
  if (bv2.y != 0) {
    bv3.x = C * cosAC;
    bv3.y = (B * C * cosBC - bv2.x * bv3.x) / bv2.y;
    bv3.z = sqrt(C * C - bv3.x * bv3.x - bv3.y * bv3.y);
  }
  else {
    VECZERO(bv3);
  }
  VECZERO(orig);
  if ((s=Coord_setup(coord, &orig, &bv1, &bv2, &bv3, topo)) != OK) {
    return ERROR(s);
  }
  pos = Coord_pos(coord);
  memset(Coord_vel(coord), 0, natoms*sizeof(dvec));
  memset(Coord_force(coord), 0, natoms*sizeof(dvec));

  TEXT("initialized coordinate domain");
  VEC(bv1);
  VEC(bv2);
  VEC(bv3);
  VEC(orig);
  INT(Coord_domain(coord)->periodic_x);
  INT(Coord_domain(coord)->periodic_y);
  INT(Coord_domain(coord)->periodic_z);

  /* read coordinate data from our VMD atom selection */
  snprintf(script, sizeof(script), "%s get { x y z }", p->aselname);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0)) return ERROR(ERR_EXPECT);
  if (NULL==(obj = Tcl_GetObjResult(interp))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
    return ERROR(ERR_EXPECT);
  }
  if (objc != natoms) return ERROR(ERR_EXPECT);
  for (i = 0;  i < natoms;  i++) {
    double d;
    int cobjc;
    Tcl_Obj **cobjv;
    if (TCL_OK != Tcl_ListObjGetElements(interp, objv[i], &cobjc, &cobjv)) {
      return ERROR(ERR_EXPECT);
    }
    if (cobjc != 3) return ERROR(ERR_EXPECT);
    if (TCL_OK != Tcl_GetDoubleFromObj(interp, cobjv[0], &d)) {
      return ERROR(ERR_EXPECT);
    }
    pos[i].x = d;
    if (TCL_OK != Tcl_GetDoubleFromObj(interp, cobjv[1], &d)) {
      return ERROR(ERR_EXPECT);
    }
    pos[i].y = d;
    if (TCL_OK != Tcl_GetDoubleFromObj(interp, cobjv[2], &d)) {
      return ERROR(ERR_EXPECT);
    }
    pos[i].z = d;
  }
  if ((s=Coord_update_pos(coord, UPDATE_ALL)) != OK) return ERROR(s);

  TEXT("initialized coordinate positions");

  if ((s=Fbonded_setup(&(p->fbon), Coord_domain(coord))) != OK) {
    return ERROR(s);
  }

  return OK;
}
