/* topology.c */

#include <string.h>
#include "moltypes/const.h"
#include "moltypes/topology.h"


int32 Topology_status(const Topology *t) {
  return t->status;
}


void Topology_reset_status(Topology *t, int32 flags) {
  t->status &= ~flags;
}


int Topology_setup_atom_cluster(Topology *t) {
  const Bond *bond = Array_data_const(&(t->bond));
  Atom *atom = Array_data(&(t->atom));
  const int32 natoms = Array_length(&(t->atom));
  int32 i, index, j0, j1, clusterID;
  Idseq seq;
  int s;  /* error status */

  /*
   * set clusterID to lowest atomID over all bonded atoms
   * (the clusterID propagates upwards)
   * increment clusterSize of leading atom to count number of bonded atoms
   */
  for (i = 0;  i < natoms;  i++) {
    const Idlist *bondlist = Objarr_elem_const(&(t->atom_bondlist), i);
    if (NULL == bondlist) return ERROR(ERR_EXPECT);
    if ((s=Idseq_init(&seq, bondlist)) != OK) return ERROR(s);
    atom[i].clusterSize = 0;  /* either resets clusterSize for head atom of
                       cluster before accumulation or this doesn't matter */
    clusterID = i;  /* assume we start a new cluster */
    while ((index = Idseq_getid(&seq)) != FAIL) { /* loop over atom i bonds */
      j0 = bond[index].atomID[0];
      j1 = bond[index].atomID[1];
      if (j0 < clusterID) {            /* assign smaller clusterID if found */
        clusterID = atom[j0].clusterID;
        break;
      }
      else if (j1 < clusterID) {
        clusterID = atom[j1].clusterID;
        break;
      }
    }
    Idseq_done(&seq);
    atom[i].clusterID = clusterID;
    atom[clusterID].clusterSize++;
  }

  /*
   * second pass to set clusterSize of each atom
   */
  for (i = 0;  i < natoms;  i++) {
    index = atom[i].clusterID;
    atom[i].clusterSize = atom[index].clusterSize;
  }
  return OK;
}


int Topology_setup_atom_parent(Topology *t) {
  const Bond *bond = Array_data_const(&(t->bond));
  Atom *atom = Array_data(&(t->atom));
  const int32 natoms = Array_length(&(t->atom));
  int32 i, index;

  /*
   * Set parentID to atom index if atom is heavy.
   * Otherwise, atom is light and should be bonded to exactly one
   * other atom.  If this other atom is heavy, then it is the parent.
   * If not, then both atoms must be hydrogen, and the parent is the
   * atom with the smaller index.
   */
  for (i = 0;  i < natoms;  i++) {
    if (ATOM_HEAVY==(atom[i].atomInfo & ATOM_TYPE)) {
      atom[i].parentID = i;
    }
    else {
      const Idlist *bondlist = Objarr_elem_const(&(t->atom_bondlist), i);
      int32 j;
      if (NULL == bondlist) return ERROR(ERR_EXPECT);
      if (Idlist_length(bondlist) != 1) return ERROR(ERR_EXPECT);
      if (FAIL==(index = Idlist_min(bondlist))) return ERROR(ERR_EXPECT);
      j = bond[index].atomID[0];
      if (j==i) j = bond[index].atomID[1];
      ASSERT(j != i);
      atom[i].parentID = j;
      /* validate - if j is not HEAVY, then i-j must be H2 */
      if (ATOM_HEAVY!=(atom[j].atomInfo & ATOM_TYPE)) {
        if (ATOM_HYDROGEN!=(atom[i].atomInfo & ATOM_TYPE) ||
            ATOM_HYDROGEN!=(atom[j].atomInfo & ATOM_TYPE)) {
          return ERROR(ERR_EXPECT);
        }
        bondlist = Objarr_elem_const(&(t->atom_bondlist), j);
        if (Idlist_length(bondlist) != 1) return ERROR(ERR_EXPECT);
        if (i < j) atom[i].parentID = i;  /* parent of H2 is smallest index */
      }
    }
  }

  return OK;
}


/*
 * manage atoms
 */

int Topology_setmaxnum_atom(Topology *t, int32 n) {
  int s;  /* error status */
  if (n < Array_length(&(t->atom))) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(&(t->atom), n)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(t->atom_bondlist), n)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(t->atom_anglelist), n)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(t->atom_dihedlist), n)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(t->atom_imprlist), n)) != OK) return ERROR(s);
  if ((s=Array_setbuflen(&(t->atom_indexmap), n)) != OK) return ERROR(s);
  return OK;
}


int32 Topology_add_atom(Topology *t, const Atom *p) {
  Idlist empty;
  Array *atom = &(t->atom);
  Atom *pid;
  int32 id = Array_length(atom);  /* this atom ID */
  int s;  /* error status */
  if ((s=Array_append(atom, p)) != OK) return ERROR(s);
  if ((pid=Array_elem(atom, id)) == NULL) return ERROR(ERR_EXPECT);
  pid->externID = id;  /* this fixed ID won't change after sorting atoms */
  pid->atomPrmID = ForcePrm_getid_atomprm(t->fprm, pid->atomType);
  t->status |= (FAIL==pid->atomPrmID ? TOPO_ATOM_MISSPRM : 0);
  pid->atomInfo &= ~ATOM_TYPE;  /* clear and reset the atom type */
  if      (pid->m >  MASS_HYDROGEN_MAX) pid->atomInfo |= ATOM_HEAVY;
  else if (pid->m >= MASS_HYDROGEN_MIN) pid->atomInfo |= ATOM_HYDROGEN;
  else if (pid->m > 0)  pid->atomInfo |= ATOM_DRUDE;
  else    (pid->m = 0,  pid->atomInfo |= ATOM_LONEPAIR);
  pid->inv_mass = (pid->m != 0 ? 1/pid->m : 0);  /* set inverse mass */
  if ((s=Idlist_init(&empty)) != OK) return ERROR(s);
  if ((s=Objarr_append(&(t->atom_bondlist), &empty)) != OK) return ERROR(s);
  if ((s=Objarr_append(&(t->atom_anglelist), &empty)) != OK) return ERROR(s);
  if ((s=Objarr_append(&(t->atom_dihedlist), &empty)) != OK) return ERROR(s);
  if ((s=Objarr_append(&(t->atom_imprlist), &empty)) != OK) return ERROR(s);
  if ((s=Array_append(&(t->atom_indexmap), &id)) != OK) return ERROR(s);
  Idlist_done(&empty);
  t->status |= TOPO_ATOM_ADD;
  return id;
}


int Topology_update_atom(Topology *t, const Atom *p, int32 id) {
  Array *atom = &(t->atom);
  Atom *pid;
  if ((pid=Array_elem(atom, id)) == NULL) return ERROR(ERR_RANGE);
  if (strcmp(pid->atomType, p->atomType) == 0) {
    int32 externID = pid->externID;  /* preserve fixed external ID */
    *pid = *p;  /* just copy it */
    pid->externID = externID;  /* restore fixed external ID after copy */
  }
  else {  /* all fprm dependencies must be updated */
    Idseq seq;  /* list iterator */
    Idlist *plist;  /* points to bond/angle/dihed/impr list for this atom */
    int32 j;  /* index from list */
    int32 externID = pid->externID;  /* preserve fixed external ID */
    int s;  /* error status */
    *pid = *p;  /* copy */
    pid->externID = externID;  /* restore fixed external ID after copy */
    /* update fprm ID for this atom */
    pid->atomPrmID = ForcePrm_getid_atomprm(t->fprm, pid->atomType);
    t->status |= (FAIL==pid->atomPrmID ? TOPO_ATOM_MISSPRM : 0);
    /* update fprm ID for all bonds depending on this atom */
    if ((plist=Objarr_elem(&(t->atom_bondlist), id)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if ((s=Idseq_init(&seq, plist)) != OK) return ERROR(s);
    while ((j=Idseq_getid(&seq)) != FAIL) Topology_setprm_bond(t, j);
    Idseq_done(&seq);
    /* update fprm ID for all angles depending on this atom */
    if ((plist=Objarr_elem(&(t->atom_anglelist), id)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if ((s=Idseq_init(&seq, plist)) != OK) return ERROR(s);
    while ((j=Idseq_getid(&seq)) != FAIL) Topology_setprm_angle(t, j);
    Idseq_done(&seq);
    /* update fprm ID for all diheds depending on this atom */
    if ((plist=Objarr_elem(&(t->atom_dihedlist), id)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if ((s=Idseq_init(&seq, plist)) != OK) return ERROR(s);
    while ((j=Idseq_getid(&seq)) != FAIL) Topology_setprm_dihed(t, j);
    Idseq_done(&seq);
    /* update fprm ID for all imprs depending on this atom */
    if ((plist=Objarr_elem(&(t->atom_imprlist), id)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if ((s=Idseq_init(&seq, plist)) != OK) return ERROR(s);
    while ((j=Idseq_getid(&seq)) != FAIL) Topology_setprm_impr(t, j);
    Idseq_done(&seq);
  }
  pid->atomInfo &= ~ATOM_TYPE;  /* clear and reset the atom type */
  if      (pid->m >  MASS_HYDROGEN_MAX) pid->atomInfo |= ATOM_HEAVY;
  else if (pid->m >= MASS_HYDROGEN_MIN) pid->atomInfo |= ATOM_HYDROGEN;
  else if (pid->m > 0)  pid->atomInfo |= ATOM_DRUDE;
  else    (pid->m = 0,  pid->atomInfo |= ATOM_LONEPAIR);
  pid->inv_mass = (pid->m != 0 ? 1/pid->m : 0);  /* set inverse mass */
  t->status |= TOPO_ATOM_UPDATE;
  return OK;
}


int32 Topology_setprm_atom(Topology *t, int32 id) {
  Array *atom = &(t->atom);
  Atom *pid;
  if ((pid=Array_elem(atom, id)) == NULL) return ERROR(ERR_RANGE);
  pid->atomPrmID = ForcePrm_getid_atomprm(t->fprm, pid->atomType);
  t->status |= (FAIL==pid->atomPrmID ? TOPO_ATOM_MISSPRM : 0);
  return pid->atomPrmID;
}


int Topology_setprm_atom_array(Topology *t) {
  Atom *p = Array_data(&(t->atom));
  const int32 n = Array_length(&(t->atom));
  int32 i;
  for (i = 0;  i < n;  i++) {
    p[i].atomPrmID = ForcePrm_getid_atomprm(t->fprm, p[i].atomType);
    t->status |= (FAIL==p[i].atomPrmID ? TOPO_ATOM_MISSPRM : 0);
  }
  return ((t->status & TOPO_ATOM_MISSPRM) ? FAIL : OK);
}


const Atom *Topology_atom(const Topology *t, int32 id) {
  const Array *atom = &(t->atom);
  const Atom *pid;
  if ((pid=Array_elem_const(atom, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return pid;
}


const Atom *Topology_atom_array(const Topology *t) {
  return Array_data_const(&(t->atom));
}


int32 Topology_atom_array_length(const Topology *t) {
  return Array_length(&(t->atom));
}


const Idlist *Topology_atom_bondlist(const Topology *t, int32 id) {
  const Objarr *bondlist = &(t->atom_bondlist);
  const Idlist *pid;
  if ((pid=Objarr_elem_const(bondlist, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return pid;
}


const Idlist *Topology_atom_anglelist(const Topology *t, int32 id) {
  const Objarr *anglelist = &(t->atom_anglelist);
  const Idlist *pid;
  if ((pid=Objarr_elem_const(anglelist, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return pid;
}


const Idlist *Topology_atom_dihedlist(const Topology *t, int32 id) {
  const Objarr *dihedlist = &(t->atom_dihedlist);
  const Idlist *pid;
  if ((pid=Objarr_elem_const(dihedlist, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return pid;
}


const Idlist *Topology_atom_imprlist(const Topology *t, int32 id) {
  const Objarr *imprlist = &(t->atom_imprlist);
  const Idlist *pid;
  if ((pid=Objarr_elem_const(imprlist, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return pid;
}


#if 0
/* Optimize the order of atoms for force evaluation,
 * and other special methods like simulation of Drude oscillators.
 */
int Topology_sort_atom_array(Topology *t) {
  Array newatom;   /* create new Atom array to replace old array */
  Atom *patom;     /* point to newatom buffer */
  Array sortatom;  /* create array of SortElem for atoms */
  SortElem *p;     /* point to sortatom buffer */
  const int32 natoms = Array_length(&(t->atom));
  const Atom *atom = Array_data(&(t->atom));
  int s;  /* error status */

  /* we can't sort too many atoms due to shift below */
  if (natoms > 0x1FFFFFFF) return ERROR(ERR_EXPECT);

  /* must have parents setup, since this determines our sorting */
  if ((s=Topology_setup_atom_parent(t)) != OK) return ERROR(s);

  /* create SortElem array, sort key is parentID && atom type,
   * and value is externID */
  if ((s=Array_init(&sortatom, sizeof(SortElem))) != OK) return ERROR(s);
  if ((s=Array_resize(&sortatom, natoms)) != OK) {
    Array_done(&sortatom);
    return ERROR(s);
  }
  p = Array_data(&sortatom);
  for (i = 0;  i < natoms;  i++) {
    p[i].key = ((atom[i].parentID << 2) | (atom[i].atomInfo & ATOM_TYPE));
    p[i].value = atom[i].externID;
  }
  if ((s=Sort_quick(p, natoms)) != OK) {
    Array_done(&sortatom);
    return ERROR(s);
  }

  /* create a new Atom array, copy old array into new using the
   * permutation created by sorting */
  if ((s=Array_init(&newatom, sizeof(Atom))) != OK) {
    Array_done(&sortatom);
    return ERROR(s);
  }
  if ((s=Array_resize(&newatom, natoms)) != OK) {
    Array_done(&newatom);
    Array_done(&sortatom);
    return ERROR(s);
  }
  patom = Array_data(&newatom);
  for (i = 0;  i < natoms;  i++) {
    patom[i] = atom[ p[i].value ];
  }

  /* erase old Atom array and move new Atom array to replace it */
  Array_setbuflen(&(t->atom), 0);
  t->atom = newatom;
  Array_unalias(&newatom);
  Array_done(&newatom);
  Array_done(&sortatom);

  /*
   * updates needed:
   * - atomIDs in bond, angle, dihed, impr, excl arrays
   * - keys (but not ordering) changes for bond_map, angle_map, dihed_map,
   *   impr_map, excl_map
   * - apply permutation to change order of atom_bondlist, atom_anglelist,
   *   atom_dihedlist, atom_imprlist
   * - inverse permutation in atom_indexmap
   * - (if bond, angle, dihed, impr, and excl arrays are rebuilt, then need
   *   to reset bond_open, angle_open, dihed_open, impr_open, excl_open)
   */



  return OK;
}
#endif


/* Obtain an array that maps the original atom ordering to the
 * new atom ordering:  atom_indexmap[i] gives the new index of
 * what was originally atom[i].
 *
 * Invariant under sorting:  atom[ atom_indexmap[i] ].externID == i
 */
const int32 *Toplogy_atom_indexmap(const Topology *t) {
  return Array_data_const(&(t->atom_indexmap));
}


/*
 * manage bonds
 */

int Topology_setmaxnum_bond(Topology *t, int32 n) {
  int s;  /* error status */
  if (n < Array_length(&(t->bond))) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(&(t->bond), n)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(t->bond_anglelist), n)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(t->bond_dihedlist), n)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(t->bond_imprlist), n)) != OK) return ERROR(s);
  return OK;
}


int32 Topology_add_bond(Topology *t, const Bond *p) {
  Array *bond = &(t->bond);
  Arrmap *map = &(t->bond_map);
  Array *open = &(t->bond_open);
  Bond *pid = NULL;
  Idlist *bondlist = Objarr_data(&(t->atom_bondlist));
  const Atom *atom_array = Array_data_const(&(t->atom));
  const int32 natoms = Array_length(&(t->atom));
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (p->atomID[0] < 0 || p->atomID[0] >= natoms ||
      p->atomID[1] < 0 || p->atomID[1] >= natoms) return ERROR(ERR_RANGE);
  else if (p->atomID[0] == p->atomID[1]) return ERROR(ERR_VALUE);

  if (is_append) {
    id = Array_length(bond);  /* next available array index */
    if ((s=Array_append(bond, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(bond, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(bond, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom IDs for map */
  if (pid->atomID[0] > pid->atomID[1]) {
    pid->atomID[0] = p->atomID[1];  /* reverse order */
    pid->atomID[1] = p->atomID[0];
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this bond is already defined */
    if (is_append) {
      Array_remove(bond, NULL);  /* have to undo append */
    }
    else {
      memset(pid, -1, sizeof(Bond));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }

  /* set bond parameter ID */
  pid->bondPrmID = ForcePrm_getid_bondprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType);
  t->status |= (FAIL==pid->bondPrmID ? TOPO_BOND_MISSPRM : 0);

  /* insert this bond ID into the bondlist for each atom */
  if ((s=Idlist_insert(bondlist + pid->atomID[0], id)) != OK) return ERROR(s);
  if ((s=Idlist_insert(bondlist + pid->atomID[1], id)) != OK) return ERROR(s);

  if (is_append) {
    /* append empty list to angle/dihed/impr lists for this bond */
    Idlist empty;
    if ((s=Idlist_init(&empty)) != OK) return ERROR(s);
    if ((s=Objarr_append(&(t->bond_anglelist), &empty)) != OK) return ERROR(s);
    if ((s=Objarr_append(&(t->bond_dihedlist), &empty)) != OK) return ERROR(s);
    if ((s=Objarr_append(&(t->bond_imprlist), &empty)) != OK) return ERROR(s);
    Idlist_done(&empty);
  }
  t->status |= TOPO_BOND_ADD;
  ASSERT(Objarr_length(&(t->bond_anglelist)) == Array_length(&(t->bond)));
  ASSERT(Objarr_length(&(t->bond_dihedlist)) == Array_length(&(t->bond)));
  ASSERT(Objarr_length(&(t->bond_imprlist)) == Array_length(&(t->bond)));
  return id;
}


int32 Topology_getid_bond(const Topology *t, int32 a0, int32 a1) {
  const Arrmap *map = &(t->bond_map);
  int32 id;
  int32 atomID[NUM_BOND_ATOM];

  ASSERT(NUM_BOND_ATOM == 2);

  /* enforce canonical ordering of atom IDs */
  if (a0 > a1) {
    atomID[0] = a1;
    atomID[1] = a0;
  }
  else {
    atomID[0] = a0;
    atomID[1] = a1;
  }

  if ((id=Arrmap_lookup(map, atomID)) < 0) {
    /* this bond is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id;
}


int Topology_remove_bond(Topology *t, int32 id) {
  Idseq seq;
  Idlist *plist;
  Array *bond = &(t->bond);
  Arrmap *map = &(t->bond_map);
  Array *open = &(t->bond_open);
  Idlist *bondlist = Objarr_data(&(t->atom_bondlist));
  Bond *pid;
  int32 j;
  int s;

  if ((pid=Array_elem(bond, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */

  /* must remove all angles that depend on this bond */
  if ((plist=Objarr_elem(&(t->bond_anglelist), id)) == NULL) {
    return ERROR(ERR_EXPECT);
  }
  if ((s=Idseq_init(&seq, plist)) != OK) return ERROR(s);
  while ((j=Idseq_getid(&seq)) != FAIL) Topology_remove_angle(t, j);
  Idseq_done(&seq);
  if ((s=Idlist_erase(plist)) != OK) return ERROR(s);  /* make list empty */

  /* must remove all dihedrals that depend on this bond */
  if ((plist=Objarr_elem(&(t->bond_dihedlist), id)) == NULL) {
    return ERROR(ERR_EXPECT);
  }
  if ((s=Idseq_init(&seq, plist)) != OK) return ERROR(s);
  while ((j=Idseq_getid(&seq)) != FAIL) Topology_remove_dihed(t, j);
  Idseq_done(&seq);
  if ((s=Idlist_erase(plist)) != OK) return ERROR(s);  /* make list empty */

  /* must remove all impropers that depend on this bond */
  if ((plist=Objarr_elem(&(t->bond_imprlist), id)) == NULL) {
    return ERROR(ERR_EXPECT);
  }
  if ((s=Idseq_init(&seq, plist)) != OK) return ERROR(s);
  while ((j=Idseq_getid(&seq)) != FAIL) Topology_remove_impr(t, j);
  Idseq_done(&seq);
  if ((s=Idlist_erase(plist)) != OK) return ERROR(s);  /* make list empty */

  /* remove bond ID from the bondlist for each atom */
  if ((s=Idlist_remove(bondlist + pid->atomID[0], id)) != OK) return ERROR(s);
  if ((s=Idlist_remove(bondlist + pid->atomID[1], id)) != OK) return ERROR(s);

  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, -1, sizeof(Bond));  /* clear array element */
  t->status |= TOPO_BOND_REMOVE;
  return OK;
}


int32 Topology_setprm_bond(Topology *t, int32 id) {
  Array *bond = &(t->bond);
  Bond *pid;
  const Atom *atom_array = Array_data_const(&(t->atom));
  if ((pid=Array_elem(bond, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */
  ASSERT(0 <= pid->atomID[0] && pid->atomID[0] < Array_length(&(t->atom))
      && 0 <= pid->atomID[1] && pid->atomID[1] < Array_length(&(t->atom)));
  pid->bondPrmID = ForcePrm_getid_bondprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType);
  t->status |= (FAIL==pid->bondPrmID ? TOPO_BOND_MISSPRM : 0);
  return pid->bondPrmID;
}


int Topology_setprm_bond_array(Topology *t) {
  const Atom *atom_array = Array_data_const(&(t->atom));
  Bond *bond_array = Array_data(&(t->bond));
  Bond *pid;
  const int32 nbonds = Array_length(&(t->bond));
  int32 i;
  for (i = 0;  i < nbonds;  i++) {
    pid = &bond_array[i];
    if (FAIL == pid->atomID[0]) continue;  /* skip over holes */
    pid->bondPrmID = ForcePrm_getid_bondprm(t->fprm,
        atom_array[ pid->atomID[0] ].atomType,
        atom_array[ pid->atomID[1] ].atomType);
    t->status |= (FAIL==pid->bondPrmID ? TOPO_BOND_MISSPRM : 0);
  }
  return ((t->status & TOPO_BOND_MISSPRM) ? FAIL : OK);
}


const Bond *Topology_bond(const Topology *t, int32 id) {
  const Array *bond = &(t->bond);
  const Bond *pid;
  if ((pid=Array_elem_const(bond, id)) == NULL) {  /* range check on ID */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == pid->atomID[0]) {  /* make sure this ID isn't a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const Bond *Topology_bond_array(const Topology *t) {
  return Array_data_const(&(t->bond));
}


int32 Topology_bond_array_length(const Topology *t) {
  return Array_length(&(t->bond));
}


const Idlist *Topology_bond_anglelist(const Topology *t, int32 id) {
  const Bond *bond_array = Array_data_const(&(t->bond));
  const Objarr *anglelist = &(t->bond_anglelist);
  const Idlist *pid;
  if ((pid=Objarr_elem_const(anglelist, id)) == NULL) {  /* range check */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == bond_array[id].atomID[0]) {  /* make sure not a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const Idlist *Topology_bond_dihedlist(const Topology *t, int32 id) {
  const Bond *bond_array = Array_data_const(&(t->bond));
  const Objarr *dihedlist = &(t->bond_dihedlist);
  const Idlist *pid;
  if ((pid=Objarr_elem_const(dihedlist, id)) == NULL) {  /* range check */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == bond_array[id].atomID[0]) {  /* make sure not a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const Idlist *Topology_bond_imprlist(const Topology *t, int32 id) {
  const Bond *bond_array = Array_data_const(&(t->bond));
  const Objarr *imprlist = &(t->bond_imprlist);
  const Idlist *pid;
  if ((pid=Objarr_elem_const(imprlist, id)) == NULL) {  /* range check */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == bond_array[id].atomID[0]) {  /* make sure not a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


int Topology_compact_bond_array(Topology *t) {
  Array *bond = &(t->bond);
  Arrmap *map = &(t->bond_map);
  Array *open = &(t->bond_open);
  Objarr *anglelist = &(t->bond_anglelist);
  Objarr *dihedlist = &(t->bond_dihedlist);
  Objarr *imprlist = &(t->bond_imprlist);
  Bond *bond_array = Array_data(bond);
  Idlist *anglelist_array = Objarr_data(anglelist);
  Idlist *dihedlist_array = Objarr_data(dihedlist);
  Idlist *imprlist_array = Objarr_data(imprlist);
  Idlist *atom_bondlist = Objarr_data(&(t->atom_bondlist));
  Idlist tmplist;  /* undefined */
  int32 nbonds = Array_length(bond);
  const int32 natoms = Array_length(&(t->atom));
  int32 delta = 0, i;
  int s;  /* error status */

  if (Array_length(open) == 0) return OK; /* no holes to compact */
  for (i = 0;  i < nbonds;  i++) {
    if (FAIL == bond_array[i].atomID[0]) {
      delta++;
    }
    else if (delta > 0) {  /* slide bond entries backwards by delta */
      bond_array[i-delta] = bond_array[i];
      if ((s=Arrmap_update(map, i-delta)) != i) return ERROR(s);
      /* swap lists so we don't have to free memory more than delta times */
      tmplist = anglelist_array[i-delta];
      ASSERT(Idlist_length(&tmplist) == 0);  /* hole propagates up */
      anglelist_array[i-delta] = anglelist_array[i];
      anglelist_array[i] = tmplist;
      tmplist = dihedlist_array[i-delta];
      ASSERT(Idlist_length(&tmplist) == 0);  /* hole propagates up */
      dihedlist_array[i-delta] = dihedlist_array[i];
      dihedlist_array[i] = tmplist;
      tmplist = imprlist_array[i-delta];
      ASSERT(Idlist_length(&tmplist) == 0);  /* hole propagates up */
      imprlist_array[i-delta] = imprlist_array[i];
      imprlist_array[i] = tmplist;
    }
  }
  ASSERT(Array_length(open) == delta);
  if ((s=Array_resize(bond, nbonds-delta)) != OK) return ERROR(s);
  if ((s=Objarr_resize(anglelist, nbonds-delta)) != OK) return ERROR(s);
  if ((s=Objarr_resize(dihedlist, nbonds-delta)) != OK) return ERROR(s);
  if ((s=Objarr_resize(imprlist, nbonds-delta)) != OK) return ERROR(s);
  if ((s=Array_resize(open, 0)) != OK) return ERROR(s);

  /* need to destroy and rebuild atom_bondlist */
  bond_array = Array_data(bond);
  nbonds = Array_length(bond);
  for (i = 0;  i < natoms;  i++) {
    if ((s=Idlist_erase(atom_bondlist + i)) != OK) return ERROR(s);
  }
  for (i = 0;  i < nbonds;  i++) {
    const Bond *pid = bond_array + i;
    int32 aid0 = pid->atomID[0];
    int32 aid1 = pid->atomID[1];
    ASSERT(0 <= aid0 && aid0 < natoms);
    ASSERT(0 <= aid1 && aid1 < natoms);
    if ((s=Idlist_insert(atom_bondlist + aid0, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_bondlist + aid1, i)) != OK) return ERROR(s);
  }
  return OK;
}


/*
 * manage angles
 */

int Topology_setmaxnum_angle(Topology *t, int32 n) {
  int s;  /* error status */
  if (n < Array_length(&(t->angle))) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(&(t->angle), n)) != OK) return ERROR(s);
  return OK;
}


int32 Topology_add_angle(Topology *t, const Angle *p) {
  Array *angle = &(t->angle);
  Arrmap *map = &(t->angle_map);
  Array *open = &(t->angle_open);
  Angle *pid = NULL;
  Idlist *bond_anglelist = Objarr_data(&(t->bond_anglelist));
  Idlist *atom_anglelist = Objarr_data(&(t->atom_anglelist));
  const Atom *atom_array = Array_data_const(&(t->atom));
  const int32 natoms = Array_length(&(t->atom));
  int32 id;  /* ID for this add */
  int32 bid0, bid1;  /* IDs for dependent bonds */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (p->atomID[0] < 0 || p->atomID[0] >= natoms ||
      p->atomID[1] < 0 || p->atomID[1] >= natoms ||
      p->atomID[2] < 0 || p->atomID[2] >= natoms) return ERROR(ERR_RANGE);
  else if (p->atomID[0] == p->atomID[1] ||
      p->atomID[0] == p->atomID[2] ||
      p->atomID[1] == p->atomID[2]) return ERROR(ERR_VALUE);
  else if ((bid0=Topology_getid_bond(t, p->atomID[0], p->atomID[1])) == FAIL ||
      (bid1=Topology_getid_bond(t, p->atomID[1], p->atomID[2])) == FAIL) {
    return ERROR(ERR_VALUE);
  }

  if (is_append) {
    id = Array_length(angle);  /* next available array index */
    if ((s=Array_append(angle, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(angle, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(angle, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom IDs for map */
  if (pid->atomID[0] > pid->atomID[2]) {
    pid->atomID[0] = p->atomID[2];  /* reverse order */
    pid->atomID[2] = p->atomID[0];
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this angle is already defined */
    if (is_append) {
      Array_remove(angle, NULL);  /* have to undo append */
    }
    else {
      memset(pid, -1, sizeof(Angle));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }

  /* set angle parameter ID */
  pid->anglePrmID = ForcePrm_getid_angleprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType,
      atom_array[ pid->atomID[2] ].atomType);
  t->status |= (FAIL==pid->anglePrmID ? TOPO_ANGLE_MISSPRM : 0);

  /* insert this angle ID into the anglelist for each atom */
  if ((s=Idlist_insert(atom_anglelist + pid->atomID[0], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_anglelist + pid->atomID[1], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_anglelist + pid->atomID[2], id)) != OK) {
    return ERROR(s);
  }

  /* insert this angle ID into the anglelist for each bond */
  if ((s=Idlist_insert(bond_anglelist + bid0, id)) != OK) return ERROR(s);
  if ((s=Idlist_insert(bond_anglelist + bid1, id)) != OK) return ERROR(s);

  t->status |= TOPO_ANGLE_ADD;
  return id;
}


int32 Topology_getid_angle(const Topology *t, int32 a0, int32 a1, int32 a2) {
  const Arrmap *map = &(t->angle_map);
  int32 id;
  int32 atomID[NUM_ANGLE_ATOM];

  ASSERT(NUM_ANGLE_ATOM == 3);

  /* enforce canonical ordering of atom IDs */
  if (a0 > a2) {
    atomID[0] = a2;
    atomID[1] = a1;
    atomID[2] = a0;
  }
  else {
    atomID[0] = a0;
    atomID[1] = a1;
    atomID[2] = a2;
  }

  if ((id=Arrmap_lookup(map, atomID)) < 0) {
    /* this angle is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id;
}


int Topology_remove_angle(Topology *t, int32 id) {
  Array *angle = &(t->angle);
  Arrmap *map = &(t->angle_map);
  Array *open = &(t->angle_open);
  Idlist *atom_anglelist = Objarr_data(&(t->atom_anglelist));
  Idlist *bond_anglelist = Objarr_data(&(t->bond_anglelist));
  Angle *pid;
  int32 bid0, bid1;
  int s;

  if ((pid=Array_elem(angle, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */

  /* remove angle ID from the anglelist for each atom */
  if ((s=Idlist_remove(atom_anglelist + pid->atomID[0], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_anglelist + pid->atomID[1], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_anglelist + pid->atomID[2], id)) != OK) {
    return ERROR(s);
  }

  /* remove angle ID from the anglelist for each bond */
  bid0 = Topology_getid_bond(t, pid->atomID[0], pid->atomID[1]);
  bid1 = Topology_getid_bond(t, pid->atomID[1], pid->atomID[2]);
  if (FAIL==bid0 || FAIL==bid1) return ERROR(ERR_EXPECT);
  if ((s=Idlist_remove(bond_anglelist + bid0, id)) != OK) return ERROR(s);
  if ((s=Idlist_remove(bond_anglelist + bid1, id)) != OK) return ERROR(s);

  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, -1, sizeof(Angle));  /* clear array element */
  t->status |= TOPO_ANGLE_REMOVE;
  return OK;
}


int32 Topology_setprm_angle(Topology *t, int32 id) {
  Array *angle = &(t->angle);
  Angle *pid;
  const Atom *atom_array = Array_data_const(&(t->atom));
  if ((pid=Array_elem(angle, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */
  ASSERT(0 <= pid->atomID[0] && pid->atomID[0] < Array_length(&(t->atom))
      && 0 <= pid->atomID[1] && pid->atomID[1] < Array_length(&(t->atom))
      && 0 <= pid->atomID[2] && pid->atomID[2] < Array_length(&(t->atom)));
  pid->anglePrmID = ForcePrm_getid_angleprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType,
      atom_array[ pid->atomID[2] ].atomType);
  t->status |= (FAIL==pid->anglePrmID ? TOPO_ANGLE_MISSPRM : 0);
  return pid->anglePrmID;
}


int Topology_setprm_angle_array(Topology *t) {
  const Atom *atom_array = Array_data_const(&(t->atom));
  Angle *angle_array = Array_data(&(t->angle));
  Angle *pid;
  const int32 nangles = Array_length(&(t->angle));
  int32 i;
  for (i = 0;  i < nangles;  i++) {
    pid = &angle_array[i];
    if (FAIL == pid->atomID[0]) continue;  /* skip over holes */
    pid->anglePrmID = ForcePrm_getid_angleprm(t->fprm,
        atom_array[ pid->atomID[0] ].atomType,
        atom_array[ pid->atomID[1] ].atomType,
        atom_array[ pid->atomID[2] ].atomType);
    t->status |= (FAIL==pid->anglePrmID ? TOPO_ANGLE_MISSPRM : 0);
  }
  return ((t->status & TOPO_ANGLE_MISSPRM) ? FAIL : OK);
}


const Angle *Topology_angle(const Topology *t, int32 id) {
  const Array *angle = &(t->angle);
  const Angle *pid;
  if ((pid=Array_elem_const(angle, id)) == NULL) {  /* range check on ID */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == pid->atomID[0]) {  /* make sure this ID isn't a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const Angle *Topology_angle_array(const Topology *t) {
  return Array_data_const(&(t->angle));
}


int32 Topology_angle_array_length(const Topology *t) {
  return Array_length(&(t->angle));
}


int Topology_compact_angle_array(Topology *t) {
  Array *angle = &(t->angle);
  Arrmap *map = &(t->angle_map);
  Array *open = &(t->angle_open);
  Angle *angle_array = Array_data(angle);
  Idlist *atom_anglelist = Objarr_data(&(t->atom_anglelist));
  Idlist *bond_anglelist = Objarr_data(&(t->bond_anglelist));
  int32 nangles = Array_length(angle);
  const int32 nbonds = Array_length(&(t->bond));
  const int32 natoms = Array_length(&(t->atom));
  int32 delta = 0, i;
  int s;  /* error status */

  if (Array_length(open) == 0) return OK; /* no holes to compact */
  for (i = 0;  i < nangles;  i++) {
    if (FAIL == angle_array[i].atomID[0]) {
      delta++;
    }
    else if (delta > 0) {  /* slide angle entries backwards by delta */
      angle_array[i-delta] = angle_array[i];
      if ((s=Arrmap_update(map, i-delta)) != i) return ERROR(s);
    }
  }
  ASSERT(Array_length(open) == delta);
  if ((s=Array_resize(angle, nangles-delta)) != OK) return ERROR(s);
  if ((s=Array_resize(open, 0)) != OK) return ERROR(s);

  /* need to destroy and rebuild atom_anglelist and bond_anglelist */
  angle_array = Array_data(angle);
  nangles = Array_length(angle);
  for (i = 0;  i < natoms;  i++) {
    if ((s=Idlist_erase(atom_anglelist + i)) != OK) return ERROR(s);
  }
  for (i = 0;  i < nbonds;  i++) {
    if ((s=Idlist_erase(bond_anglelist + i)) != OK) return ERROR(s);
  }
  for (i = 0;  i < nangles;  i++) {
    const Angle *pid = angle_array + i;
    int32 aid0 = pid->atomID[0];
    int32 aid1 = pid->atomID[1];
    int32 aid2 = pid->atomID[2];
    int32 bid0 = Topology_getid_bond(t, aid0, aid1);
    int32 bid1 = Topology_getid_bond(t, aid1, aid2);
    ASSERT(0 <= aid0 && aid0 < natoms);
    ASSERT(0 <= aid1 && aid1 < natoms);
    ASSERT(0 <= aid2 && aid2 < natoms);
    ASSERT(0 <= bid0 && bid0 < nbonds);
    ASSERT(0 <= bid1 && bid1 < nbonds);
    if ((s=Idlist_insert(atom_anglelist + aid0, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_anglelist + aid1, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_anglelist + aid2, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_anglelist + bid0, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_anglelist + bid1, i)) != OK) return ERROR(s);
  }
  return OK;
}


/*
 * manage dihedrals
 */

int Topology_setmaxnum_dihed(Topology *t, int32 n) {
  int s;  /* error status */
  if (n < Array_length(&(t->dihed))) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(&(t->dihed), n)) != OK) return ERROR(s);
  return OK;
}


int32 Topology_add_dihed(Topology *t, const Dihed *p) {
  Array *dihed = &(t->dihed);
  Arrmap *map = &(t->dihed_map);
  Array *open = &(t->dihed_open);
  Dihed *pid = NULL;
  Idlist *bond_dihedlist = Objarr_data(&(t->bond_dihedlist));
  Idlist *atom_dihedlist = Objarr_data(&(t->atom_dihedlist));
  const Atom *atom_array = Array_data_const(&(t->atom));
  const int32 natoms = Array_length(&(t->atom));
  int32 id;  /* ID for this add */
  int32 bid0, bid1, bid2;  /* IDs for dependent bonds */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (p->atomID[0] < 0 || p->atomID[0] >= natoms ||
      p->atomID[1] < 0 || p->atomID[1] >= natoms ||
      p->atomID[2] < 0 || p->atomID[2] >= natoms ||
      p->atomID[3] < 0 || p->atomID[3] >= natoms) return ERROR(ERR_RANGE);
  else if (p->atomID[0] == p->atomID[1] ||
      p->atomID[0] == p->atomID[2] ||
      p->atomID[0] == p->atomID[3] ||
      p->atomID[1] == p->atomID[2] ||
      p->atomID[1] == p->atomID[3] ||
      p->atomID[2] == p->atomID[3]) return ERROR(ERR_VALUE);
  else if ((bid0=Topology_getid_bond(t, p->atomID[0], p->atomID[1])) == FAIL ||
      (bid1=Topology_getid_bond(t, p->atomID[1], p->atomID[2])) == FAIL ||
      (bid2=Topology_getid_bond(t, p->atomID[2], p->atomID[3])) == FAIL) {
    return ERROR(ERR_VALUE);
  }

  if (is_append) {
    id = Array_length(dihed);  /* next available array index */
    if ((s=Array_append(dihed, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(dihed, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(dihed, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom IDs for map */
  if (pid->atomID[0] > pid->atomID[3]) {
    pid->atomID[0] = p->atomID[3];  /* reverse order */
    pid->atomID[1] = p->atomID[2];
    pid->atomID[2] = p->atomID[1];
    pid->atomID[3] = p->atomID[0];
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this dihed is already defined */
    if (is_append) {
      Array_remove(dihed, NULL);  /* have to undo append */
    }
    else {
      memset(pid, -1, sizeof(Dihed));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }

  /* set dihed parameter ID by matching against wildcards */
  pid->dihedPrmID = ForcePrm_matchid_dihedprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType,
      atom_array[ pid->atomID[2] ].atomType,
      atom_array[ pid->atomID[3] ].atomType);
  t->status |= (FAIL==pid->dihedPrmID ? TOPO_DIHED_MISSPRM : 0);

  /* insert this dihed ID into the dihedlist for each atom */
  if ((s=Idlist_insert(atom_dihedlist + pid->atomID[0], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_dihedlist + pid->atomID[1], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_dihedlist + pid->atomID[2], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_dihedlist + pid->atomID[3], id)) != OK) {
    return ERROR(s);
  }

  /* insert this dihed ID into the dihedlist for each bond */
  if ((s=Idlist_insert(bond_dihedlist + bid0, id)) != OK) return ERROR(s);
  if ((s=Idlist_insert(bond_dihedlist + bid1, id)) != OK) return ERROR(s);
  if ((s=Idlist_insert(bond_dihedlist + bid2, id)) != OK) return ERROR(s);

  t->status |= TOPO_DIHED_ADD;
  return id;
}


int32 Topology_getid_dihed(const Topology *t,
    int32 a0, int32 a1, int32 a2, int32 a3) {
  const Arrmap *map = &(t->dihed_map);
  int32 id;
  int32 atomID[NUM_DIHED_ATOM];

  ASSERT(NUM_DIHED_ATOM == 4);

  /* enforce canonical ordering of atom IDs */
  if (a0 > a3) {
    atomID[0] = a3;
    atomID[1] = a2;
    atomID[2] = a1;
    atomID[3] = a0;
  }
  else {
    atomID[0] = a0;
    atomID[1] = a1;
    atomID[2] = a2;
    atomID[3] = a3;
  }

  if ((id=Arrmap_lookup(map, atomID)) < 0) {
    /* this dihed is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id;
}


int Topology_remove_dihed(Topology *t, int32 id) {
  Array *dihed = &(t->dihed);
  Arrmap *map = &(t->dihed_map);
  Array *open = &(t->dihed_open);
  Idlist *atom_dihedlist = Objarr_data(&(t->atom_dihedlist));
  Idlist *bond_dihedlist = Objarr_data(&(t->bond_dihedlist));
  Dihed *pid;
  int32 bid0, bid1, bid2;
  int s;

  if ((pid=Array_elem(dihed, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */

  /* remove dihed ID from the dihedlist for each atom */
  if ((s=Idlist_remove(atom_dihedlist + pid->atomID[0], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_dihedlist + pid->atomID[1], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_dihedlist + pid->atomID[2], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_dihedlist + pid->atomID[3], id)) != OK) {
    return ERROR(s);
  }

  /* remove dihed ID from the dihedlist for each bond */
  bid0 = Topology_getid_bond(t, pid->atomID[0], pid->atomID[1]);
  bid1 = Topology_getid_bond(t, pid->atomID[1], pid->atomID[2]);
  bid2 = Topology_getid_bond(t, pid->atomID[2], pid->atomID[3]);
  if (FAIL==bid0 || FAIL==bid1 || FAIL==bid2) return ERROR(ERR_EXPECT);
  if ((s=Idlist_remove(bond_dihedlist + bid0, id)) != OK) return ERROR(s);
  if ((s=Idlist_remove(bond_dihedlist + bid1, id)) != OK) return ERROR(s);
  if ((s=Idlist_remove(bond_dihedlist + bid2, id)) != OK) return ERROR(s);

  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, -1, sizeof(Dihed));  /* clear array element */
  t->status |= TOPO_DIHED_REMOVE;
  return OK;
}


int32 Topology_setprm_dihed(Topology *t, int32 id) {
  Array *dihed = &(t->dihed);
  Dihed *pid;
  const Atom *atom_array = Array_data_const(&(t->atom));
  if ((pid=Array_elem(dihed, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */
  ASSERT(0 <= pid->atomID[0] && pid->atomID[0] < Array_length(&(t->atom))
      && 0 <= pid->atomID[1] && pid->atomID[1] < Array_length(&(t->atom))
      && 0 <= pid->atomID[2] && pid->atomID[2] < Array_length(&(t->atom))
      && 0 <= pid->atomID[3] && pid->atomID[3] < Array_length(&(t->atom)));
  /* set dihed parameter ID by matching against wildcards */
  pid->dihedPrmID = ForcePrm_matchid_dihedprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType,
      atom_array[ pid->atomID[2] ].atomType,
      atom_array[ pid->atomID[3] ].atomType);
  t->status |= (FAIL==pid->dihedPrmID ? TOPO_DIHED_MISSPRM : 0);
  return pid->dihedPrmID;
}


int Topology_setprm_dihed_array(Topology *t) {
  const Atom *atom_array = Array_data_const(&(t->atom));
  Dihed *dihed_array = Array_data(&(t->dihed));
  Dihed *pid;
  const int32 ndiheds = Array_length(&(t->dihed));
  int32 i;
  for (i = 0;  i < ndiheds;  i++) {
    pid = &dihed_array[i];
    if (FAIL == pid->atomID[0]) continue;  /* skip over holes */
    /* set dihed parameter ID by matching against wildcards */
    pid->dihedPrmID = ForcePrm_matchid_dihedprm(t->fprm,
        atom_array[ pid->atomID[0] ].atomType,
        atom_array[ pid->atomID[1] ].atomType,
        atom_array[ pid->atomID[2] ].atomType,
        atom_array[ pid->atomID[3] ].atomType);
    t->status |= (FAIL==pid->dihedPrmID ? TOPO_DIHED_MISSPRM : 0);
  }
  return ((t->status & TOPO_DIHED_MISSPRM) ? FAIL : OK);
}


const Dihed *Topology_dihed(const Topology *t, int32 id) {
  const Array *dihed = &(t->dihed);
  const Dihed *pid;
  if ((pid=Array_elem_const(dihed, id)) == NULL) {  /* range check on ID */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == pid->atomID[0]) {  /* make sure this ID isn't a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const Dihed *Topology_dihed_array(const Topology *t) {
  return Array_data_const(&(t->dihed));
}


int32 Topology_dihed_array_length(const Topology *t) {
  return Array_length(&(t->dihed));
}


int Topology_compact_dihed_array(Topology *t) {
  Array *dihed = &(t->dihed);
  Arrmap *map = &(t->dihed_map);
  Array *open = &(t->dihed_open);
  Dihed *dihed_array = Array_data(dihed);
  Idlist *atom_dihedlist = Objarr_data(&(t->atom_dihedlist));
  Idlist *bond_dihedlist = Objarr_data(&(t->bond_dihedlist));
  int32 ndiheds = Array_length(dihed);
  const int32 nbonds = Array_length(&(t->bond));
  const int32 natoms = Array_length(&(t->atom));
  int32 delta = 0, i;
  int s;  /* error status */

  if (Array_length(open) == 0) return OK; /* no holes to compact */
  for (i = 0;  i < ndiheds;  i++) {
    if (FAIL == dihed_array[i].atomID[0]) {
      delta++;
    }
    else if (delta > 0) {  /* slide dihed entries backwards by delta */
      dihed_array[i-delta] = dihed_array[i];
      if ((s=Arrmap_update(map, i-delta)) != i) return ERROR(s);
    }
  }
  ASSERT(Array_length(open) == delta);
  if ((s=Array_resize(dihed, ndiheds-delta)) != OK) return ERROR(s);
  if ((s=Array_resize(open, 0)) != OK) return ERROR(s);

  /* need to destroy and rebuild atom_dihedlist and bond_dihedlist */
  dihed_array = Array_data(dihed);
  ndiheds = Array_length(dihed);
  for (i = 0;  i < natoms;  i++) {
    if ((s=Idlist_erase(atom_dihedlist + i)) != OK) return ERROR(s);
  }
  for (i = 0;  i < nbonds;  i++) {
    if ((s=Idlist_erase(bond_dihedlist + i)) != OK) return ERROR(s);
  }
  for (i = 0;  i < ndiheds;  i++) {
    const Dihed *pid = dihed_array + i;
    int32 aid0 = pid->atomID[0];
    int32 aid1 = pid->atomID[1];
    int32 aid2 = pid->atomID[2];
    int32 aid3 = pid->atomID[3];
    int32 bid0 = Topology_getid_bond(t, aid0, aid1);
    int32 bid1 = Topology_getid_bond(t, aid1, aid2);
    int32 bid2 = Topology_getid_bond(t, aid2, aid3);
    ASSERT(0 <= aid0 && aid0 < natoms);
    ASSERT(0 <= aid1 && aid1 < natoms);
    ASSERT(0 <= aid2 && aid2 < natoms);
    ASSERT(0 <= aid3 && aid3 < natoms);
    ASSERT(0 <= bid0 && bid0 < nbonds);
    ASSERT(0 <= bid1 && bid1 < nbonds);
    ASSERT(0 <= bid2 && bid2 < nbonds);
    if ((s=Idlist_insert(atom_dihedlist + aid0, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_dihedlist + aid1, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_dihedlist + aid2, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_dihedlist + aid3, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_dihedlist + bid0, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_dihedlist + bid1, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_dihedlist + bid2, i)) != OK) return ERROR(s);
  }
  return OK;
}


/*
 * manage impropers
 */

int Topology_setmaxnum_impr(Topology *t, int32 n) {
  int s;  /* error status */
  if (n < Array_length(&(t->impr))) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(&(t->impr), n)) != OK) return ERROR(s);
  return OK;
}


int32 Topology_add_impr(Topology *t, const Impr *p) {
  Array *impr = &(t->impr);
  Arrmap *map = &(t->impr_map);
  Array *open = &(t->impr_open);
  Impr *pid = NULL;
  Idlist *bond_imprlist = Objarr_data(&(t->bond_imprlist));
  Idlist *atom_imprlist = Objarr_data(&(t->atom_imprlist));
  const Atom *atom_array = Array_data_const(&(t->atom));
  const int32 natoms = Array_length(&(t->atom));
  int32 id;  /* ID for this add */
  int32 bid0, bid1, bid2;  /* IDs for dependent bonds */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (p->atomID[0] < 0 || p->atomID[0] >= natoms ||
      p->atomID[1] < 0 || p->atomID[1] >= natoms ||
      p->atomID[2] < 0 || p->atomID[2] >= natoms ||
      p->atomID[3] < 0 || p->atomID[3] >= natoms) return ERROR(ERR_RANGE);
  else if (p->atomID[0] == p->atomID[1] ||
      p->atomID[0] == p->atomID[2] ||
      p->atomID[0] == p->atomID[3] ||
      p->atomID[1] == p->atomID[2] ||
      p->atomID[1] == p->atomID[3] ||
      p->atomID[2] == p->atomID[3]) return ERROR(ERR_VALUE);
  else if ((  /* either first atom is bonded to the others */
        (bid0=Topology_getid_bond(t, p->atomID[0], p->atomID[1])) == FAIL ||
        (bid1=Topology_getid_bond(t, p->atomID[0], p->atomID[2])) == FAIL ||
        (bid2=Topology_getid_bond(t, p->atomID[0], p->atomID[3])) == FAIL)
      && (    /* or last atom is bonded to the others */
        (bid0=Topology_getid_bond(t, p->atomID[0], p->atomID[3])) == FAIL ||
        (bid1=Topology_getid_bond(t, p->atomID[1], p->atomID[3])) == FAIL ||
        (bid2=Topology_getid_bond(t, p->atomID[2], p->atomID[3])) == FAIL)) {
    return ERROR(ERR_VALUE);
  }

  if (is_append) {
    id = Array_length(impr);  /* next available array index */
    if ((s=Array_append(impr, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(impr, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(impr, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom IDs for map */
  if (pid->atomID[0] > pid->atomID[3]) {
    pid->atomID[0] = p->atomID[3];  /* reverse order */
    pid->atomID[1] = p->atomID[2];
    pid->atomID[2] = p->atomID[1];
    pid->atomID[3] = p->atomID[0];
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this impr is already defined */
    if (is_append) {
      Array_remove(impr, NULL);  /* have to undo append */
    }
    else {
      memset(pid, -1, sizeof(Impr));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }

  /* set impr parameter ID by matching against wildcards */
  pid->imprPrmID = ForcePrm_matchid_imprprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType,
      atom_array[ pid->atomID[2] ].atomType,
      atom_array[ pid->atomID[3] ].atomType);
  t->status |= (FAIL==pid->imprPrmID ? TOPO_IMPR_MISSPRM : 0);

  /* insert this impr ID into the imprlist for each atom */
  if ((s=Idlist_insert(atom_imprlist + pid->atomID[0], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_imprlist + pid->atomID[1], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_imprlist + pid->atomID[2], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_insert(atom_imprlist + pid->atomID[3], id)) != OK) {
    return ERROR(s);
  }

  /* insert this impr ID into the imprlist for each bond */
  if ((s=Idlist_insert(bond_imprlist + bid0, id)) != OK) return ERROR(s);
  if ((s=Idlist_insert(bond_imprlist + bid1, id)) != OK) return ERROR(s);
  if ((s=Idlist_insert(bond_imprlist + bid2, id)) != OK) return ERROR(s);

  t->status |= TOPO_IMPR_ADD;
  return id;
}


int32 Topology_getid_impr(const Topology *t,
    int32 a0, int32 a1, int32 a2, int32 a3) {
  const Arrmap *map = &(t->impr_map);
  int32 id;
  int32 atomID[NUM_IMPR_ATOM];

  ASSERT(NUM_IMPR_ATOM == 4);

  /* enforce canonical ordering of atom IDs */
  if (a0 > a3) {
    atomID[0] = a3;
    atomID[1] = a2;
    atomID[2] = a1;
    atomID[3] = a0;
  }
  else {
    atomID[0] = a0;
    atomID[1] = a1;
    atomID[2] = a2;
    atomID[3] = a3;
  }

  if ((id=Arrmap_lookup(map, atomID)) < 0) {
    /* this impr is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id;
}


int Topology_remove_impr(Topology *t, int32 id) {
  Array *impr = &(t->impr);
  Arrmap *map = &(t->impr_map);
  Array *open = &(t->impr_open);
  Idlist *atom_imprlist = Objarr_data(&(t->atom_imprlist));
  Idlist *bond_imprlist = Objarr_data(&(t->bond_imprlist));
  Impr *pid;
  int32 b01, b02, b03, b13, b23;
  boolean fwd, bak;
  int s;

  if ((pid=Array_elem(impr, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */

  /* remove impr ID from the imprlist for each atom */
  if ((s=Idlist_remove(atom_imprlist + pid->atomID[0], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_imprlist + pid->atomID[1], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_imprlist + pid->atomID[2], id)) != OK) {
    return ERROR(s);
  }
  if ((s=Idlist_remove(atom_imprlist + pid->atomID[3], id)) != OK) {
    return ERROR(s);
  }

  /* remove impr ID from the imprlist for each bond */
  b01 = Topology_getid_bond(t, pid->atomID[0], pid->atomID[1]);
  b02 = Topology_getid_bond(t, pid->atomID[0], pid->atomID[2]);
  b03 = Topology_getid_bond(t, pid->atomID[0], pid->atomID[3]);
  b13 = Topology_getid_bond(t, pid->atomID[1], pid->atomID[3]);
  b23 = Topology_getid_bond(t, pid->atomID[2], pid->atomID[3]);

  fwd = (b01 != FAIL && b02 != FAIL && b03 != FAIL);
  bak = (b03 != FAIL && b13 != FAIL && b23 != FAIL);

  if (fwd && bak) {
    /* it's ambiguous which direction defined this improper, so we must
     * attempt to remove this improper ID from all five of bond_imprlist */
    if ((s=Idlist_remove(bond_imprlist + b01, id)) != OK && s != FAIL) {
      return ERROR(s);
    }
    if ((s=Idlist_remove(bond_imprlist + b02, id)) != OK && s != FAIL) {
      return ERROR(s);
    }
    if ((s=Idlist_remove(bond_imprlist + b03, id)) != OK && s != FAIL) {
      return ERROR(s);
    }
    if ((s=Idlist_remove(bond_imprlist + b13, id)) != OK && s != FAIL) {
      return ERROR(s);
    }
    if ((s=Idlist_remove(bond_imprlist + b23, id)) != OK && s != FAIL) {
      return ERROR(s);
    }
  }
  else if (fwd) {
    if ((s=Idlist_remove(bond_imprlist + b01, id)) != OK) return ERROR(s);
    if ((s=Idlist_remove(bond_imprlist + b02, id)) != OK) return ERROR(s);
    if ((s=Idlist_remove(bond_imprlist + b03, id)) != OK) return ERROR(s);
  }
  else if (bak) {
    if ((s=Idlist_remove(bond_imprlist + b03, id)) != OK) return ERROR(s);
    if ((s=Idlist_remove(bond_imprlist + b13, id)) != OK) return ERROR(s);
    if ((s=Idlist_remove(bond_imprlist + b23, id)) != OK) return ERROR(s);
  }
  else {
    return ERROR(ERR_EXPECT);
  }

  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, -1, sizeof(Impr));  /* clear array element */
  t->status |= TOPO_IMPR_REMOVE;
  return OK;
}


int32 Topology_setprm_impr(Topology *t, int32 id) {
  Array *impr = &(t->impr);
  Impr *pid;
  const Atom *atom_array = Array_data_const(&(t->atom));
  if ((pid=Array_elem(impr, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */
  ASSERT(0 <= pid->atomID[0] && pid->atomID[0] < Array_length(&(t->atom))
      && 0 <= pid->atomID[1] && pid->atomID[1] < Array_length(&(t->atom))
      && 0 <= pid->atomID[2] && pid->atomID[2] < Array_length(&(t->atom))
      && 0 <= pid->atomID[3] && pid->atomID[3] < Array_length(&(t->atom)));
  /* set impr parameter ID by matching against wildcards */
  pid->imprPrmID = ForcePrm_matchid_imprprm(t->fprm,
      atom_array[ pid->atomID[0] ].atomType,
      atom_array[ pid->atomID[1] ].atomType,
      atom_array[ pid->atomID[2] ].atomType,
      atom_array[ pid->atomID[3] ].atomType);
  t->status |= (FAIL==pid->imprPrmID ? TOPO_IMPR_MISSPRM : 0);
  return pid->imprPrmID;
}


int Topology_setprm_impr_array(Topology *t) {
  const Atom *atom_array = Array_data_const(&(t->atom));
  Impr *impr_array = Array_data(&(t->impr));
  Impr *pid;
  const int32 nimprs = Array_length(&(t->impr));
  int32 i;
  for (i = 0;  i < nimprs;  i++) {
    pid = &impr_array[i];
    if (FAIL == pid->atomID[0]) continue;  /* skip over holes */
    /* set impr parameter ID by matching against wildcards */
    pid->imprPrmID = ForcePrm_matchid_imprprm(t->fprm,
        atom_array[ pid->atomID[0] ].atomType,
        atom_array[ pid->atomID[1] ].atomType,
        atom_array[ pid->atomID[2] ].atomType,
        atom_array[ pid->atomID[3] ].atomType);
    t->status |= (FAIL==pid->imprPrmID ? TOPO_IMPR_MISSPRM : 0);
  }
  return ((t->status & TOPO_IMPR_MISSPRM) ? FAIL : OK);
}


const Impr *Topology_impr(const Topology *t, int32 id) {
  const Array *impr = &(t->impr);
  const Impr *pid;
  if ((pid=Array_elem_const(impr, id)) == NULL) {  /* range check on ID */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == pid->atomID[0]) {  /* make sure this ID isn't a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const Impr *Topology_impr_array(const Topology *t) {
  return Array_data_const(&(t->impr));
}


int32 Topology_impr_array_length(const Topology *t) {
  return Array_length(&(t->impr));
}


int Topology_compact_impr_array(Topology *t) {
  Array *impr = &(t->impr);
  Arrmap *map = &(t->impr_map);
  Array *open = &(t->impr_open);
  Impr *impr_array = Array_data(impr);
  Idlist *atom_imprlist = Objarr_data(&(t->atom_imprlist));
  Idlist *bond_imprlist = Objarr_data(&(t->bond_imprlist));
  int32 nimprs = Array_length(impr);
  const int32 nbonds = Array_length(&(t->bond));
  const int32 natoms = Array_length(&(t->atom));
  int32 delta = 0, i;
  int s;  /* error status */

  if (Array_length(open) == 0) return OK; /* no holes to compact */
  for (i = 0;  i < nimprs;  i++) {
    if (FAIL == impr_array[i].atomID[0]) {
      delta++;
    }
    else if (delta > 0) {  /* slide impr entries backwards by delta */
      impr_array[i-delta] = impr_array[i];
      if ((s=Arrmap_update(map, i-delta)) != i) return ERROR(s);
    }
  }
  ASSERT(Array_length(open) == delta);
  if ((s=Array_resize(impr, nimprs-delta)) != OK) return ERROR(s);
  if ((s=Array_resize(open, 0)) != OK) return ERROR(s);

  /* need to destroy and rebuild atom_imprlist and bond_imprlist */
  impr_array = Array_data(impr);
  nimprs = Array_length(impr);
  for (i = 0;  i < natoms;  i++) {
    if ((s=Idlist_erase(atom_imprlist + i)) != OK) return ERROR(s);
  }
  for (i = 0;  i < nbonds;  i++) {
    if ((s=Idlist_erase(bond_imprlist + i)) != OK) return ERROR(s);
  }
  for (i = 0;  i < nimprs;  i++) {
    const Impr *pid = impr_array + i;
    int32 aid0 = pid->atomID[0];
    int32 aid1 = pid->atomID[1];
    int32 aid2 = pid->atomID[2];
    int32 aid3 = pid->atomID[3];
    int32 bid0, bid1, bid2;
    ASSERT(0 <= aid0 && aid0 < natoms);
    ASSERT(0 <= aid1 && aid1 < natoms);
    ASSERT(0 <= aid2 && aid2 < natoms);
    ASSERT(0 <= aid3 && aid3 < natoms);
    if ((       /* either first atom is bonded to the others */
          (bid0=Topology_getid_bond(t, aid0, aid1)) == FAIL ||
          (bid1=Topology_getid_bond(t, aid0, aid2)) == FAIL ||
          (bid2=Topology_getid_bond(t, aid0, aid3)) == FAIL)
        && (    /* or last atom is bonded to the others */
          (bid0=Topology_getid_bond(t, aid0, aid3)) == FAIL ||
          (bid1=Topology_getid_bond(t, aid1, aid3)) == FAIL ||
          (bid2=Topology_getid_bond(t, aid2, aid3)) == FAIL)) {
      return ERROR(ERR_EXPECT);
    }
    ASSERT(0 <= bid0 && bid0 < nbonds);
    ASSERT(0 <= bid1 && bid1 < nbonds);
    ASSERT(0 <= bid2 && bid2 < nbonds);
    if ((s=Idlist_insert(atom_imprlist + aid0, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_imprlist + aid1, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_imprlist + aid2, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(atom_imprlist + aid3, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_imprlist + bid0, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_imprlist + bid1, i)) != OK) return ERROR(s);
    if ((s=Idlist_insert(bond_imprlist + bid2, i)) != OK) return ERROR(s);
  }
  return OK;
}


/*
 * manage explicit exclusions
 */

int Topology_setmaxnum_excl(Topology *t, int32 n) {
  int s;  /* error status */
  if (n < Array_length(&(t->excl))) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(&(t->excl), n)) != OK) return ERROR(s);
  return OK;
}


int32 Topology_add_excl(Topology *t, const Excl *p) {
  Array *excl = &(t->excl);
  Arrmap *map = &(t->excl_map);
  Array *open = &(t->excl_open);
  Excl *pid = NULL;
  const int32 natoms = Array_length(&(t->atom));
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (p->atomID[0] < 0 || p->atomID[0] >= natoms ||
      p->atomID[1] < 0 || p->atomID[1] >= natoms) return ERROR(ERR_RANGE);
  else if (p->atomID[0] == p->atomID[1]) return ERROR(ERR_VALUE);

  if (is_append) {
    id = Array_length(excl);  /* next available array index */
    if ((s=Array_append(excl, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(excl, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(excl, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom IDs for map */
  if (pid->atomID[0] > pid->atomID[1]) {
    pid->atomID[0] = p->atomID[1];  /* reverse order */
    pid->atomID[1] = p->atomID[0];
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this excl is already defined */
    if (is_append) {
      Array_remove(excl, NULL);  /* have to undo append */
    }
    else {
      memset(pid, -1, sizeof(Excl));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  t->status |= TOPO_EXCL_ADD;
  return id;
}


int32 Topology_getid_excl(const Topology *t, int32 a0, int32 a1) {
  const Arrmap *map = &(t->excl_map);
  int32 id;
  int32 atomID[NUM_EXCL_ATOM];

  ASSERT(NUM_EXCL_ATOM == 2);

  /* enforce canonical ordering of atom IDs */
  if (a0 > a1) {
    atomID[0] = a1;
    atomID[1] = a0;
  }
  else {
    atomID[0] = a0;
    atomID[1] = a1;
  }

  if ((id=Arrmap_lookup(map, atomID)) < 0) {
    /* this excl is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id;
}


int Topology_remove_excl(Topology *t, int32 id) {
  Array *excl = &(t->excl);
  Arrmap *map = &(t->excl_map);
  Array *open = &(t->excl_open);
  Excl *pid;
  int s;

  if ((pid=Array_elem(excl, id)) == NULL) return ERROR(ERR_RANGE);
  else if (FAIL == pid->atomID[0]) return ERROR(ERR_VALUE);  /* hole */

  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, -1, sizeof(Excl));  /* clear array element */
  t->status |= TOPO_EXCL_REMOVE;
  return OK;
}


const Excl *Topology_excl(const Topology *t, int32 id) {
  const Array *excl = &(t->excl);
  const Excl *pid;
  if ((pid=Array_elem_const(excl, id)) == NULL) {  /* range check on ID */
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  else if (FAIL == pid->atomID[0]) {  /* make sure this ID isn't a hole */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const Excl *Topology_excl_array(const Topology *t) {
  return Array_data_const(&(t->excl));
}


int32 Topology_excl_array_length(const Topology *t) {
  return Array_length(&(t->excl));
}


int Topology_compact_excl_array(Topology *t) {
  Array *excl = &(t->excl);
  Arrmap *map = &(t->excl_map);
  Array *open = &(t->excl_open);
  Excl *excl_array = Array_data(excl);
  const int32 nexcls = Array_length(excl);
  int32 delta = 0, i;
  int s;  /* error status */

  if (Array_length(open) == 0) return OK; /* no holes to compact */
  for (i = 0;  i < nexcls;  i++) {
    if (FAIL == excl_array[i].atomID[0]) {
      delta++;
    }
    else if (delta > 0) {  /* slide excl entries backwards by delta */
      excl_array[i-delta] = excl_array[i];
      if ((s=Arrmap_update(map, i-delta)) != i) return ERROR(s);
    }
  }
  ASSERT(Array_length(open) == delta);
  if ((s=Array_resize(excl, nexcls-delta)) != OK) return ERROR(s);
  if ((s=Array_resize(open, 0)) != OK) return ERROR(s);
  return OK;
}
