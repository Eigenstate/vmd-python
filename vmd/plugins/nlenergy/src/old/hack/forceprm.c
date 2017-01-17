/* forceprm.c */

#include <string.h>
#include "moltypes/forceprm.h"


int32 ForcePrm_status(const ForcePrm *f) {
  return f->status;
}


void ForcePrm_reset_status(ForcePrm *f, int32 flags) {
  f->status &= ~flags;
}


/*
 * manage atom parameters
 */

int32 ForcePrm_add_atomprm(ForcePrm *f, const AtomPrm *p) {
  Array *prm = &(f->atomprm);
  Arrmap *map = &(f->atomprm_map);
  Array *open = &(f->atomprm_open);
  AtomPrm *pid = NULL;
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (is_append) {
    id = Array_length(prm);  /* next available array index */
    if ((s=Array_append(prm, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  ASSERT(strlen(pid->atomType[0]) < sizeof(AtomType));

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this parameter set is already defined */
    if (is_append) {
      Array_remove(prm, NULL);  /* have to undo append */
    }
    else {
      memset(pid, 0, sizeof(AtomPrm));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  f->status |= FP_ATOMPRM_ADD;
  return id-1;
}


int32 ForcePrm_update_atomprm(ForcePrm *f, const AtomPrm *p) {
  Array *prm = &(f->atomprm);
  const Arrmap *map = &(f->atomprm_map);
  AtomPrm *pid;  /* to point to array element for update */
  int32 id;      /* ID for this update */

  ASSERT(strlen(p->atomType[0]) < sizeof(AtomType));

  if ((id=Arrmap_lookup(map, p->atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  *pid = *p;  /* copy updated parameter data */
  f->status |= FP_ATOMPRM_UPDATE;
  return id-1;
}


int32 ForcePrm_getid_atomprm(const ForcePrm *f, const char *a0) {
  const Arrmap *map = &(f->atomprm_map);
  int32 id;
  AtomType atomType[NUM_ATOMPRM_ATOMTYPE];

  ASSERT(NUM_ATOMPRM_ATOMTYPE == 1);

  ASSERT(strlen(a0) < sizeof(AtomType));
  strcpy(atomType[0], a0);

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id-1;
}


int ForcePrm_remove_atomprm(ForcePrm *f, int32 id) {
  Array *prm = &(f->atomprm);
  Arrmap *map = &(f->atomprm_map);
  Array *open = &(f->atomprm_open);
  AtomPrm *pid;
  int s;
  id++;
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_RANGE);
  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, 0, sizeof(AtomPrm));  /* clear array element */
  f->status |= FP_ATOMPRM_REMOVE;
  return OK;
}


const AtomPrm *ForcePrm_atomprm(const ForcePrm *f, int32 id) {
  const Array *prm = &(f->atomprm);
  const AtomPrm *pid;
  id++;
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    pid=Array_data_const(prm);  /* set to the "zero"-valued parameter */
  }
#if 0
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  if (0 == pid->atomType[0][0]) {  /* check this ID has not been removed */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
#endif
  return pid;
}


const AtomPrm *ForcePrm_atomprm_array(const ForcePrm *f) {
  return ((const AtomPrm *)Array_data_const(&(f->atomprm))) + 1;
}


int32 ForcePrm_atomprm_array_length(const ForcePrm *f) {
  return Array_length(&(f->atomprm)) - 1;
}


/*
 * manage bond parameters
 */

int32 ForcePrm_add_bondprm(ForcePrm *f, const BondPrm *p) {
  Array *prm = &(f->bondprm);
  Arrmap *map = &(f->bondprm_map);
  Array *open = &(f->bondprm_open);
  BondPrm *pid = NULL;
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (is_append) {
    id = Array_length(prm);  /* next available array index */
    if ((s=Array_append(prm, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(pid->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[1]) < sizeof(AtomType));
  if (strcmp(pid->atomType[0], pid->atomType[1]) > 0) {
    strcpy(pid->atomType[0], p->atomType[1]);  /* swap order */
    strcpy(pid->atomType[1], p->atomType[0]);
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this parameter set is already defined */
    if (is_append) {
      Array_remove(prm, NULL);  /* have to undo append */
    }
    else {
      memset(pid, 0, sizeof(BondPrm));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  f->status |= FP_BONDPRM_ADD;
  return id-1;
}


int32 ForcePrm_update_bondprm(ForcePrm *f, const BondPrm *p) {
  Array *prm = &(f->bondprm);
  const Arrmap *map = &(f->bondprm_map);
  BondPrm *pid;  /* to point to array element for update */
  int32 id;      /* ID for this update */
  AtomType atomType[NUM_BONDPRM_ATOMTYPE];
  boolean is_reversed = FALSE;

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(p->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[1]) < sizeof(AtomType));
  if (strcmp(p->atomType[0], p->atomType[1]) > 0) {
    strcpy(atomType[0], p->atomType[1]);  /* swap order */
    strcpy(atomType[1], p->atomType[0]);
    is_reversed = TRUE;
  }
  else {
    memcpy(atomType, p->atomType, sizeof(atomType));
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  *pid = *p;  /* copy updated parameter data */
  if (is_reversed) memcpy(pid->atomType, atomType, sizeof(pid->atomType));
  f->status |= FP_BONDPRM_UPDATE;
  return id-1;
}


int32 ForcePrm_getid_bondprm(const ForcePrm *f,
    const char *a0, const char *a1) {
  const Arrmap *map = &(f->bondprm_map);
  int32 id;
  AtomType atomType[NUM_BONDPRM_ATOMTYPE];

  ASSERT(NUM_BONDPRM_ATOMTYPE == 2);

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(a0) < sizeof(AtomType));
  ASSERT(strlen(a1) < sizeof(AtomType));
  if (strcmp(a0, a1) > 0) {
    strcpy(atomType[0], a1);  /* swap order */
    strcpy(atomType[1], a0);
  }
  else {
    strcpy(atomType[0], a0);
    strcpy(atomType[1], a1);
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id-1;
}


int ForcePrm_remove_bondprm(ForcePrm *f, int32 id) {
  Array *prm = &(f->bondprm);
  Arrmap *map = &(f->bondprm_map);
  Array *open = &(f->bondprm_open);
  BondPrm *pid;
  int s;
  id++;
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_RANGE);
  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, 0, sizeof(BondPrm));  /* clear array element */
  f->status |= FP_BONDPRM_REMOVE;
  return OK;
}


const BondPrm *ForcePrm_bondprm(const ForcePrm *f, int32 id) {
  const Array *prm = &(f->bondprm);
  const BondPrm *pid;
  id++;
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    pid=Array_data_const(prm);  /* set to the "zero"-valued parameter */
  }
#if 0
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  if (0 == pid->atomType[0][0]) {  /* check this ID has not been removed */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
#endif
  return pid;
}


const BondPrm *ForcePrm_bondprm_array(const ForcePrm *f) {
  return ((const BondPrm *)Array_data_const(&(f->bondprm))) + 1;
}


int32 ForcePrm_bondprm_array_length(const ForcePrm *f) {
  return Array_length(&(f->bondprm)) - 1;
}


/*
 * manage angle parameters
 */

int32 ForcePrm_add_angleprm(ForcePrm *f, const AnglePrm *p) {
  Array *prm = &(f->angleprm);
  Arrmap *map = &(f->angleprm_map);
  Array *open = &(f->angleprm_open);
  AnglePrm *pid = NULL;
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (is_append) {
    id = Array_length(prm);  /* next available array index */
    if ((s=Array_append(prm, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(pid->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[1]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[2]) < sizeof(AtomType));
  if (strcmp(pid->atomType[0], pid->atomType[2]) > 0) {
    strcpy(pid->atomType[0], p->atomType[2]);  /* swap order */
    strcpy(pid->atomType[2], p->atomType[0]);
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this parameter set is already defined */
    if (is_append) {
      Array_remove(prm, NULL);  /* have to undo append */
    }
    else {
      memset(pid, 0, sizeof(AnglePrm));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  f->status |= FP_ANGLEPRM_ADD;
  return id-1;
}


int32 ForcePrm_update_angleprm(ForcePrm *f, const AnglePrm *p) {
  Array *prm = &(f->angleprm);
  const Arrmap *map = &(f->angleprm_map);
  AnglePrm *pid;  /* to point to array element for update */
  int32 id;      /* ID for this update */
  AtomType atomType[NUM_ANGLEPRM_ATOMTYPE];
  boolean is_reversed = FALSE;

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(p->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[1]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[2]) < sizeof(AtomType));
  if (strcmp(p->atomType[0], p->atomType[2]) > 0) {
    strcpy(atomType[0], p->atomType[2]);  /* swap order */
    strcpy(atomType[1], p->atomType[1]);
    strcpy(atomType[2], p->atomType[0]);
    is_reversed = TRUE;
  }
  else {
    memcpy(atomType, p->atomType, sizeof(atomType));
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  *pid = *p;  /* copy updated parameter data */
  if (is_reversed) memcpy(pid->atomType, atomType, sizeof(pid->atomType));
  f->status |= FP_ANGLEPRM_UPDATE;
  return id-1;
}


int32 ForcePrm_getid_angleprm(const ForcePrm *f,
    const char *a0, const char *a1, const char *a2) {
  const Arrmap *map = &(f->angleprm_map);
  int32 id;
  AtomType atomType[NUM_ANGLEPRM_ATOMTYPE];

  ASSERT(NUM_ANGLEPRM_ATOMTYPE == 3);

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(a0) < sizeof(AtomType));
  ASSERT(strlen(a1) < sizeof(AtomType));
  ASSERT(strlen(a2) < sizeof(AtomType));
  if (strcmp(a0, a2) > 0) {
    strcpy(atomType[0], a2);  /* swap order */
    strcpy(atomType[1], a1);
    strcpy(atomType[2], a0);
  }
  else {
    strcpy(atomType[0], a0);
    strcpy(atomType[1], a1);
    strcpy(atomType[2], a2);
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id-1;
}


int ForcePrm_remove_angleprm(ForcePrm *f, int32 id) {
  Array *prm = &(f->angleprm);
  Arrmap *map = &(f->angleprm_map);
  Array *open = &(f->angleprm_open);
  AnglePrm *pid;
  int s;
  id++;
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_RANGE);
  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, 0, sizeof(AnglePrm));  /* clear array element */
  f->status |= FP_ANGLEPRM_REMOVE;
  return OK;
}


const AnglePrm *ForcePrm_angleprm(const ForcePrm *f, int32 id) {
  const Array *prm = &(f->angleprm);
  const AnglePrm *pid;
  id++;
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    pid=Array_data_const(prm);  /* set to the "zero"-valued parameter */
  }
#if 0
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  if (0 == pid->atomType[0][0]) {  /* check this ID has not been removed */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
#endif
  return pid;
}


const AnglePrm *ForcePrm_angleprm_array(const ForcePrm *f) {
  return ((const AnglePrm *)Array_data_const(&(f->angleprm))) + 1;
}


int32 ForcePrm_angleprm_array_length(const ForcePrm *f) {
  return Array_length(&(f->angleprm)) - 1;
}


/*
 * manage dihedral parameters
 */

int32 ForcePrm_add_dihedprm(ForcePrm *f, const DihedPrm *p) {
  Objarr *prm = &(f->dihedprm);
  Arrmap *map = &(f->dihedprm_map);
  Array *open = &(f->dihedprm_open);
  DihedPrm *pid = NULL;
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (is_append) {
    id = Objarr_length(prm);  /* next available array index */
    if ((s=Objarr_append(prm, p)) != OK) return ERROR(s);
    if ((pid=Objarr_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Objarr_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
    if (DihedPrm_copy(pid, p)) return ERROR(ERR_EXPECT);
  }

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(pid->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[1]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[2]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[3]) < sizeof(AtomType));
  if (strcmp(pid->atomType[0], pid->atomType[3]) > 0
      || (strcmp(pid->atomType[0], pid->atomType[3]) == 0
          && strcmp(pid->atomType[1], pid->atomType[2]) > 0)) {
    strcpy(pid->atomType[0], p->atomType[3]);  /* swap order */
    strcpy(pid->atomType[1], p->atomType[2]);
    strcpy(pid->atomType[2], p->atomType[1]);
    strcpy(pid->atomType[3], p->atomType[0]);
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this parameter set is already defined */
    if (is_append) {
      Objarr_remove(prm, NULL);  /* have to undo append */
    }
    else {
      if ((s=DihedPrm_erase(pid)) != OK) return ERROR(s); /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  f->status |= FP_DIHEDPRM_ADD;
  return id-1;
}


int32 ForcePrm_update_dihedprm(ForcePrm *f, const DihedPrm *p) {
  Objarr *prm = &(f->dihedprm);
  const Arrmap *map = &(f->dihedprm_map);
  DihedPrm *pid;  /* to point to array element for update */
  int32 id;      /* ID for this update */
  AtomType atomType[NUM_DIHEDPRM_ATOMTYPE];
  boolean is_reversed = FALSE;

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(p->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[1]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[2]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[3]) < sizeof(AtomType));
  if (strcmp(p->atomType[0], p->atomType[3]) > 0
      || (strcmp(p->atomType[0], p->atomType[3]) == 0
          && strcmp(p->atomType[1], p->atomType[2]) > 0)) {
    strcpy(atomType[0], p->atomType[3]);  /* swap order */
    strcpy(atomType[1], p->atomType[2]);
    strcpy(atomType[2], p->atomType[1]);
    strcpy(atomType[3], p->atomType[0]);
    is_reversed = TRUE;
  }
  else {
    memcpy(atomType, p->atomType, sizeof(atomType));
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  if ((pid=Objarr_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  if (DihedPrm_copy(pid, p)) return ERROR(ERR_EXPECT);  /* copy */
  if (is_reversed) memcpy(pid->atomType, atomType, sizeof(pid->atomType));
  f->status |= FP_DIHEDPRM_UPDATE;
  return id-1;
}


int32 ForcePrm_getid_dihedprm(const ForcePrm *f,
    const char *a0, const char *a1, const char *a2, const char *a3) {
  const Arrmap *map = &(f->dihedprm_map);
  int32 id;
  AtomType atomType[NUM_DIHEDPRM_ATOMTYPE];

  ASSERT(NUM_DIHEDPRM_ATOMTYPE == 4);

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(a0) < sizeof(AtomType));
  ASSERT(strlen(a1) < sizeof(AtomType));
  ASSERT(strlen(a2) < sizeof(AtomType));
  ASSERT(strlen(a3) < sizeof(AtomType));
  if (strcmp(a0, a3) > 0 || (strcmp(a0, a3) == 0 && strcmp(a1, a2) > 0)) {
    strcpy(atomType[0], a3);  /* swap order */
    strcpy(atomType[1], a2);
    strcpy(atomType[2], a1);
    strcpy(atomType[3], a0);
  }
  else {
    strcpy(atomType[0], a0);
    strcpy(atomType[1], a1);
    strcpy(atomType[2], a2);
    strcpy(atomType[3], a3);
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id-1;
}


int32 ForcePrm_matchid_dihedprm(const ForcePrm *f,
    const char *at0, const char *at1, const char *at2, const char *at3) {
  int32 id;
  /* find first wildcard match (or bail after first error) */
  if ((id=ForcePrm_getid_dihedprm(f, at0, at1, at2, at3)) == FAIL
      && (id=ForcePrm_getid_dihedprm(f, "X", at1, at2, "X")) == FAIL) {
    return FAIL;
  }
  return (id >= 0 ? id : ERROR(id));
}


int ForcePrm_remove_dihedprm(ForcePrm *f, int32 id) {
  Objarr *prm = &(f->dihedprm);
  Arrmap *map = &(f->dihedprm_map);
  Array *open = &(f->dihedprm_open);
  DihedPrm *pid;
  int s;
  id++;
  if ((pid=Objarr_elem(prm, id)) == NULL) return ERROR(ERR_RANGE);
  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  if ((s=DihedPrm_erase(pid)) != OK) return ERROR(s); /* erase the entry */
  f->status |= FP_DIHEDPRM_REMOVE;
  return OK;
}


const DihedPrm *ForcePrm_dihedprm(const ForcePrm *f, int32 id) {
  const Objarr *prm = &(f->dihedprm);
  const DihedPrm *pid;
  id++;
  if ((pid=Objarr_elem_const(prm, id)) == NULL) {
    pid=Objarr_data_const(prm);  /* set to the "zero"-valued parameter */
  }
#if 0
  if ((pid=Objarr_elem_const(prm, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  if (0 == pid->atomType[0][0]) {  /* check this ID has not been removed */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
#endif
  return pid;
}


const DihedPrm *ForcePrm_dihedprm_array(const ForcePrm *f) {
  return ((const DihedPrm *)Objarr_data_const(&(f->dihedprm))) + 1;
}


int32 ForcePrm_dihedprm_array_length(const ForcePrm *f) {
  return Objarr_length(&(f->dihedprm)) - 1;
}


/*
 * manage improper parameters
 */

int32 ForcePrm_add_imprprm(ForcePrm *f, const ImprPrm *p) {
  Array *prm = &(f->imprprm);
  Arrmap *map = &(f->imprprm_map);
  Array *open = &(f->imprprm_open);
  ImprPrm *pid = NULL;
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (is_append) {
    id = Array_length(prm);  /* next available array index */
    if ((s=Array_append(prm, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(pid->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[1]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[2]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[3]) < sizeof(AtomType));
  if (strcmp(pid->atomType[0], pid->atomType[3]) > 0
      || (strcmp(pid->atomType[0], pid->atomType[3]) == 0
          && strcmp(pid->atomType[1], pid->atomType[2]) > 0)) {
    strcpy(pid->atomType[0], p->atomType[3]);  /* swap order */
    strcpy(pid->atomType[1], p->atomType[2]);
    strcpy(pid->atomType[2], p->atomType[1]);
    strcpy(pid->atomType[3], p->atomType[0]);
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this parameter set is already defined */
    if (is_append) {
      Array_remove(prm, NULL);  /* have to undo append */
    }
    else {
      memset(pid, 0, sizeof(ImprPrm));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  f->status |= FP_IMPRPRM_ADD;
  return id-1;
}


int32 ForcePrm_update_imprprm(ForcePrm *f, const ImprPrm *p) {
  Array *prm = &(f->imprprm);
  const Arrmap *map = &(f->imprprm_map);
  ImprPrm *pid;  /* to point to array element for update */
  int32 id;      /* ID for this update */
  AtomType atomType[NUM_IMPRPRM_ATOMTYPE];
  boolean is_reversed = FALSE;

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(p->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[1]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[2]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[3]) < sizeof(AtomType));
  if (strcmp(p->atomType[0], p->atomType[3]) > 0
      || (strcmp(p->atomType[0], p->atomType[3]) == 0
          && strcmp(p->atomType[1], p->atomType[2]) > 0)) {
    strcpy(atomType[0], p->atomType[3]);  /* swap order */
    strcpy(atomType[1], p->atomType[2]);
    strcpy(atomType[2], p->atomType[1]);
    strcpy(atomType[3], p->atomType[0]);
    is_reversed = TRUE;
  }
  else {
    memcpy(atomType, p->atomType, sizeof(atomType));
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  *pid = *p;  /* copy updated parameter data */
  if (is_reversed) memcpy(pid->atomType, atomType, sizeof(pid->atomType));
  f->status |= FP_IMPRPRM_UPDATE;
  return id-1;
}


int32 ForcePrm_getid_imprprm(const ForcePrm *f,
    const char *a0, const char *a1, const char *a2, const char *a3) {
  const Arrmap *map = &(f->imprprm_map);
  int32 id;
  AtomType atomType[NUM_IMPRPRM_ATOMTYPE];

  ASSERT(NUM_IMPRPRM_ATOMTYPE == 4);

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(a0) < sizeof(AtomType));
  ASSERT(strlen(a1) < sizeof(AtomType));
  ASSERT(strlen(a2) < sizeof(AtomType));
  ASSERT(strlen(a3) < sizeof(AtomType));
  if (strcmp(a0, a3) > 0 || (strcmp(a0, a3) == 0 && strcmp(a1, a2) > 0)) {
    strcpy(atomType[0], a3);  /* swap order */
    strcpy(atomType[1], a2);
    strcpy(atomType[2], a1);
    strcpy(atomType[3], a0);
  }
  else {
    strcpy(atomType[0], a0);
    strcpy(atomType[1], a1);
    strcpy(atomType[2], a2);
    strcpy(atomType[3], a3);
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id-1;
}


int32 ForcePrm_matchid_imprprm(const ForcePrm *f,
    const char *at0, const char *at1, const char *at2, const char *at3) {
  int32 id;
  /* find first wildcard match (or bail after first error) */
  if ((id=ForcePrm_getid_imprprm(f, at0, at1, at2, at3)) == FAIL
      && (id=ForcePrm_getid_imprprm(f, at0, "X", "X", at3)) == FAIL
      && (id=ForcePrm_getid_imprprm(f, "X", at1, at2, at3)) == FAIL
      && (id=ForcePrm_getid_imprprm(f, at0, at1, at2, "X")) == FAIL
      && (id=ForcePrm_getid_imprprm(f, "X", at1, at2, "X")) == FAIL
      && (id=ForcePrm_getid_imprprm(f, "X", "X", at2, at3)) == FAIL
      && (id=ForcePrm_getid_imprprm(f, at0, at1, "X", "X")) == FAIL) {
    return FAIL;
  }
  return (id >= 0 ? id : ERROR(id));
}


int ForcePrm_remove_imprprm(ForcePrm *f, int32 id) {
  Array *prm = &(f->imprprm);
  Arrmap *map = &(f->imprprm_map);
  Array *open = &(f->imprprm_open);
  ImprPrm *pid;
  int s;
  id++;
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_RANGE);
  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, 0, sizeof(ImprPrm));  /* clear array element */
  f->status |= FP_IMPRPRM_REMOVE;
  return OK;
}


const ImprPrm *ForcePrm_imprprm(const ForcePrm *f, int32 id) {
  const Array *prm = &(f->imprprm);
  const ImprPrm *pid;
  id++;
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    pid=Array_data_const(prm);  /* set to the "zero"-valued parameter */
  }
#if 0
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  if (0 == pid->atomType[0][0]) {  /* check this ID has not been removed */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
#endif
  return pid;
}


const ImprPrm *ForcePrm_imprprm_array(const ForcePrm *f) {
  return ((const ImprPrm *)Array_data_const(&(f->imprprm))) + 1;
}


int32 ForcePrm_imprprm_array_length(const ForcePrm *f) {
  return Array_length(&(f->imprprm)) - 1;
}


/*
 * manage van der Waals pair parameters
 */

int32 ForcePrm_add_vdwpairprm(ForcePrm *f, const VdwpairPrm *p) {
  Array *prm = &(f->vdwpairprm);
  Arrmap *map = &(f->vdwpairprm_map);
  Array *open = &(f->vdwpairprm_open);
  VdwpairPrm *pid = NULL;
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (is_append) {
    id = Array_length(prm);  /* next available array index */
    if ((s=Array_append(prm, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(pid->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[1]) < sizeof(AtomType));
  if (strcmp(pid->atomType[0], pid->atomType[1]) > 0) {
    strcpy(pid->atomType[0], p->atomType[1]);  /* swap order */
    strcpy(pid->atomType[1], p->atomType[0]);
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this parameter set is already defined */
    if (is_append) {
      Array_remove(prm, NULL);  /* have to undo append */
    }
    else {
      memset(pid, 0, sizeof(VdwpairPrm));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  f->status |= FP_VDWPAIRPRM_ADD;
  return id;
}


int32 ForcePrm_update_vdwpairprm(ForcePrm *f, const VdwpairPrm *p) {
  Array *prm = &(f->vdwpairprm);
  const Arrmap *map = &(f->vdwpairprm_map);
  VdwpairPrm *pid;  /* to point to array element for update */
  int32 id;      /* ID for this update */
  AtomType atomType[NUM_VDWPAIRPRM_ATOMTYPE];
  boolean is_reversed = FALSE;

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(p->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[1]) < sizeof(AtomType));
  if (strcmp(p->atomType[0], p->atomType[1]) > 0) {
    strcpy(atomType[0], p->atomType[1]);  /* swap order */
    strcpy(atomType[1], p->atomType[0]);
    is_reversed = TRUE;
  }
  else {
    memcpy(atomType, p->atomType, sizeof(atomType));
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  *pid = *p;  /* copy updated parameter data */
  if (is_reversed) memcpy(pid->atomType, atomType, sizeof(pid->atomType));
  f->status |= FP_VDWPAIRPRM_UPDATE;
  return id;
}


int32 ForcePrm_getid_vdwpairprm(const ForcePrm *f,
    const char *a0, const char *a1) {
  const Arrmap *map = &(f->vdwpairprm_map);
  int32 id;
  AtomType atomType[NUM_VDWPAIRPRM_ATOMTYPE];

  ASSERT(NUM_VDWPAIRPRM_ATOMTYPE == 2);

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(a0) < sizeof(AtomType));
  ASSERT(strlen(a1) < sizeof(AtomType));
  if (strcmp(a0, a1) > 0) {
    strcpy(atomType[0], a1);  /* swap order */
    strcpy(atomType[1], a0);
  }
  else {
    strcpy(atomType[0], a0);
    strcpy(atomType[1], a1);
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id;
}


int ForcePrm_remove_vdwpairprm(ForcePrm *f, int32 id) {
  Array *prm = &(f->vdwpairprm);
  Arrmap *map = &(f->vdwpairprm_map);
  Array *open = &(f->vdwpairprm_open);
  VdwpairPrm *pid;
  int s;
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_RANGE);
  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, 0, sizeof(VdwpairPrm));  /* clear array element */
  f->status |= FP_VDWPAIRPRM_REMOVE;
  return OK;
}


const VdwpairPrm *ForcePrm_vdwpairprm(const ForcePrm *f, int32 id) {
  const Array *prm = &(f->vdwpairprm);
  const VdwpairPrm *pid;
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  if (0 == pid->atomType[0][0]) {  /* check this ID has not been removed */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const VdwpairPrm *ForcePrm_vdwpairprm_array(const ForcePrm *f) {
  return Array_data_const(&(f->vdwpairprm));
}


int32 ForcePrm_vdwpairprm_array_length(const ForcePrm *f) {
  return Array_length(&(f->vdwpairprm));
}


/*
 * manage Buckingham pair parameters
 */

int32 ForcePrm_add_buckpairprm(ForcePrm *f, const BuckpairPrm *p) {
  Array *prm = &(f->buckpairprm);
  Arrmap *map = &(f->buckpairprm_map);
  Array *open = &(f->buckpairprm_open);
  BuckpairPrm *pid = NULL;
  int32 id;  /* ID for this add */
  int s;     /* error status */
  boolean is_append = (Array_length(open) == 0);

  if (is_append) {
    id = Array_length(prm);  /* next available array index */
    if ((s=Array_append(prm, p)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  }
  else {  /* fill up available hole due to parameter remove */
    if ((s=Array_remove(open, &id)) != OK) return ERROR(s);
    if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
    *pid = *p;
  }

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(pid->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(pid->atomType[1]) < sizeof(AtomType));
  if (strcmp(pid->atomType[0], pid->atomType[1]) > 0) {
    strcpy(pid->atomType[0], p->atomType[1]);  /* swap order */
    strcpy(pid->atomType[1], p->atomType[0]);
  }

  if ((s=Arrmap_insert(map, id)) != id) {
    /* this parameter set is already defined */
    if (is_append) {
      Array_remove(prm, NULL);  /* have to undo append */
    }
    else {
      memset(pid, 0, sizeof(BuckpairPrm));  /* erase the entry */
      Array_append(open, &id);  /* push array index back onto open list */
    }
    return (s >= 0 ? FAIL : ERROR(s));
  }
  f->status |= FP_BUCKPAIRPRM_ADD;
  return id;
}


int32 ForcePrm_update_buckpairprm(ForcePrm *f, const BuckpairPrm *p) {
  Array *prm = &(f->buckpairprm);
  const Arrmap *map = &(f->buckpairprm_map);
  BuckpairPrm *pid;  /* to point to array element for update */
  int32 id;      /* ID for this update */
  AtomType atomType[NUM_BUCKPAIRPRM_ATOMTYPE];
  boolean is_reversed = FALSE;

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(p->atomType[0]) < sizeof(AtomType));
  ASSERT(strlen(p->atomType[1]) < sizeof(AtomType));
  if (strcmp(p->atomType[0], p->atomType[1]) > 0) {
    strcpy(atomType[0], p->atomType[1]);  /* swap order */
    strcpy(atomType[1], p->atomType[0]);
    is_reversed = TRUE;
  }
  else {
    memcpy(atomType, p->atomType, sizeof(atomType));
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_EXPECT);
  *pid = *p;  /* copy updated parameter data */
  if (is_reversed) memcpy(pid->atomType, atomType, sizeof(pid->atomType));
  f->status |= FP_BUCKPAIRPRM_UPDATE;
  return id;
}


int32 ForcePrm_getid_buckpairprm(const ForcePrm *f,
    const char *a0, const char *a1) {
  const Arrmap *map = &(f->buckpairprm_map);
  int32 id;
  AtomType atomType[NUM_BUCKPAIRPRM_ATOMTYPE];

  ASSERT(NUM_BUCKPAIRPRM_ATOMTYPE == 2);

  /* enforce canonical ordering of atom types IDing this parameter set */
  ASSERT(strlen(a0) < sizeof(AtomType));
  ASSERT(strlen(a1) < sizeof(AtomType));
  if (strcmp(a0, a1) > 0) {
    strcpy(atomType[0], a1);  /* swap order */
    strcpy(atomType[1], a0);
  }
  else {
    strcpy(atomType[0], a0);
    strcpy(atomType[1], a1);
  }

  if ((id=Arrmap_lookup(map, atomType)) < 0) {
    /* this parameter set is not defined */
    return (FAIL==id ? FAIL : ERROR(id));
  }
  return id;
}


int ForcePrm_remove_buckpairprm(ForcePrm *f, int32 id) {
  Array *prm = &(f->buckpairprm);
  Arrmap *map = &(f->buckpairprm_map);
  Array *open = &(f->buckpairprm_open);
  BuckpairPrm *pid;
  int s;
  if ((pid=Array_elem(prm, id)) == NULL) return ERROR(ERR_RANGE);
  if ((s=Arrmap_remove(map, id)) != id) {  /* remove ID from hash table */
    return (s >= 0 ? FAIL : ERROR(s));
  }
  if ((s=Array_append(open, &id)) != OK) {  /* push index onto open list */
    return ERROR(s);
  }
  memset(pid, 0, sizeof(BuckpairPrm));  /* clear array element */
  f->status |= FP_BUCKPAIRPRM_REMOVE;
  return OK;
}


const BuckpairPrm *ForcePrm_buckpairprm(const ForcePrm *f, int32 id) {
  const Array *prm = &(f->buckpairprm);
  const BuckpairPrm *pid;
  if ((pid=Array_elem_const(prm, id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  if (0 == pid->atomType[0][0]) {  /* check this ID has not been removed */
    (void) ERROR(ERR_VALUE);
    return NULL;
  }
  return pid;
}


const BuckpairPrm *ForcePrm_buckpairprm_array(const ForcePrm *f) {
  return Array_data_const(&(f->buckpairprm));
}


int32 ForcePrm_buckpairprm_array_length(const ForcePrm *f) {
  return Array_length(&(f->buckpairprm));
}
