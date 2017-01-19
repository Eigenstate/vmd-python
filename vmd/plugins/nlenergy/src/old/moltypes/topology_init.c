/* topology_init.c */

#include <string.h>
#include "moltypes/topology.h"


static int g_Idlist_init(void *v);
static void g_Idlist_done(void *v);
static int g_Idlist_copy(void *vdest, const void *cvsrc);
static int g_Idlist_erase(void *v);

static const void *g_Bond_key(const Array *, int32 i);
static int32 g_Bond_hash(const Arrmap *, const void *key);
static int32 g_Bond_keycmp(const void *key1, const void *key2);

static const void *g_Angle_key(const Array *, int32 i);
static int32 g_Angle_hash(const Arrmap *, const void *key);
static int32 g_Angle_keycmp(const void *key1, const void *key2);

static const void *g_Dihed_key(const Array *, int32 i);
static int32 g_Dihed_hash(const Arrmap *, const void *key);
static int32 g_Dihed_keycmp(const void *key1, const void *key2);

static const void *g_Impr_key(const Array *, int32 i);
static int32 g_Impr_hash(const Arrmap *, const void *key);
static int32 g_Impr_keycmp(const void *key1, const void *key2);

static const void *g_Excl_key(const Array *, int32 i);
static int32 g_Excl_hash(const Arrmap *, const void *key);
static int32 g_Excl_keycmp(const void *key1, const void *key2);


/* Topology constructor */
int Topology_init(Topology *t, const ForcePrm *f) {
  int s;  /* error status */

  if ((s=Array_init(&(t->atom), sizeof(Atom))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->bond), sizeof(Bond))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->angle), sizeof(Angle))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->dihed), sizeof(Dihed))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->impr), sizeof(Impr))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->excl), sizeof(Excl))) != OK) return ERROR(s);

  if ((s=Arrmap_init(&(t->bond_map), &(t->bond),
          g_Bond_key, g_Bond_hash, g_Bond_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(t->angle_map), &(t->angle),
          g_Angle_key, g_Angle_hash, g_Angle_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(t->dihed_map), &(t->dihed),
          g_Dihed_key, g_Dihed_hash, g_Dihed_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(t->impr_map), &(t->impr),
          g_Impr_key, g_Impr_hash, g_Impr_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(t->excl_map), &(t->excl),
          g_Excl_key, g_Excl_hash, g_Excl_keycmp, 0)) != OK) {
    return ERROR(s);
  }

  if ((s=Array_init(&(t->bond_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->angle_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->dihed_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->impr_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(t->excl_open), sizeof(int32))) != OK) return ERROR(s);

  if ((s=Objarr_init(&(t->atom_bondlist), sizeof(Idlist), g_Idlist_init,
          g_Idlist_done, g_Idlist_copy, g_Idlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(t->atom_anglelist), sizeof(Idlist), g_Idlist_init,
          g_Idlist_done, g_Idlist_copy, g_Idlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(t->atom_dihedlist), sizeof(Idlist), g_Idlist_init,
          g_Idlist_done, g_Idlist_copy, g_Idlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(t->atom_imprlist), sizeof(Idlist), g_Idlist_init,
          g_Idlist_done, g_Idlist_copy, g_Idlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Array_init(&(t->atom_indexmap), sizeof(int32))) != OK) {
    return ERROR(s);
  }

  if ((s=Objarr_init(&(t->bond_anglelist), sizeof(Idlist), g_Idlist_init,
          g_Idlist_done, g_Idlist_copy, g_Idlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(t->bond_dihedlist), sizeof(Idlist), g_Idlist_init,
          g_Idlist_done, g_Idlist_copy, g_Idlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(t->bond_imprlist), sizeof(Idlist), g_Idlist_init,
          g_Idlist_done, g_Idlist_copy, g_Idlist_erase)) != OK) {
    return ERROR(s);
  }

  t->fprm = f;
  t->status = 0;
  return OK;
}


/* Topology destructor */
void Topology_done(Topology *t) {
  Array_done(&(t->atom));
  Array_done(&(t->bond));
  Array_done(&(t->angle));
  Array_done(&(t->dihed));
  Array_done(&(t->impr));
  Array_done(&(t->excl));

  Arrmap_done(&(t->bond_map));
  Arrmap_done(&(t->angle_map));
  Arrmap_done(&(t->dihed_map));
  Arrmap_done(&(t->impr_map));
  Arrmap_done(&(t->excl_map));

  Array_done(&(t->bond_open));
  Array_done(&(t->angle_open));
  Array_done(&(t->dihed_open));
  Array_done(&(t->impr_open));
  Array_done(&(t->excl_open));

  Objarr_done(&(t->atom_bondlist));
  Objarr_done(&(t->atom_anglelist));
  Objarr_done(&(t->atom_dihedlist));
  Objarr_done(&(t->atom_imprlist));
  Array_done(&(t->atom_indexmap));

  Objarr_done(&(t->bond_anglelist));
  Objarr_done(&(t->bond_dihedlist));
  Objarr_done(&(t->bond_imprlist));
}


/* generic wrappers for Idlist constructor, destructor, copy, and erase */
static int g_Idlist_init(void *v) {
  return Idlist_init((Idlist *) v);
}

static void g_Idlist_done(void *v) {
  Idlist_done((Idlist *) v);
}

static int g_Idlist_copy(void *vdest, const void *cvsrc) {
  return Idlist_copy((Idlist *) vdest, (const Idlist *) cvsrc);
}

static int g_Idlist_erase(void *v) {
  return Idlist_erase((Idlist *) v);
}


/* needed for hashing functions below */
#define HASH_PRIME  1103515249


/*
 * low level routines for using Arrmap with Bond array
 */
const void *g_Bond_key(const Array *bond, int32 i) {
  const Bond *arr = Array_data_const(bond);
  ASSERT_P(i >= 0 && i < Array_length(bond));
  return arr[i].atomID;
}

int32 g_Bond_hash(const Arrmap *arrmap, const void *key) {
  const int32 *a = (const int32 *) key;
  int32 hashvalue, i, j;
  j = a[0];
  for (i = 1;  i < NUM_BOND_ATOM;  i++) {
    j = (j << 6) + (a[i] - a[0]);
  }
  hashvalue = (((j * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_Bond_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomID fields of two Bonds match */
  const int32 *a1 = (const int32 *) key1;
  const int32 *a2 = (const int32 *) key2;
  int32 i, diff = 0;
  for (i = 0;  i < NUM_BOND_ATOM;  i++) {
    if ((diff = a1[i] - a2[i]) != 0) return diff;
  }
  return 0;
}


/*
 * low level routines for using Arrmap with Angle array
 */
const void *g_Angle_key(const Array *angle, int32 i) {
  const Angle *arr = Array_data_const(angle);
  ASSERT_P(i >= 0 && i < Array_length(angle));
  return arr[i].atomID;
}

int32 g_Angle_hash(const Arrmap *arrmap, const void *key) {
  const int32 *a = (const int32 *) key;
  int32 hashvalue, i, j;
  j = a[0];
  for (i = 1;  i < NUM_ANGLE_ATOM;  i++) {
    j = (j << 6) + (a[i] - a[0]);
  }
  hashvalue = (((j * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_Angle_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomID fields of two Angles match */
  const int32 *a1 = (const int32 *) key1;
  const int32 *a2 = (const int32 *) key2;
  int32 i, diff = 0;
  for (i = 0;  i < NUM_ANGLE_ATOM;  i++) {
    if ((diff = a1[i] - a2[i]) != 0) return diff;
  }
  return 0;
}


/*
 * low level routines for using Arrmap with Dihed array
 */
const void *g_Dihed_key(const Array *dihed, int32 i) {
  const Dihed *arr = Array_data_const(dihed);
  ASSERT_P(i >= 0 && i < Array_length(dihed));
  return arr[i].atomID;
}

int32 g_Dihed_hash(const Arrmap *arrmap, const void *key) {
  const int32 *a = (const int32 *) key;
  int32 hashvalue, i, j;
  j = a[0];
  for (i = 1;  i < NUM_DIHED_ATOM;  i++) {
    j = (j << 6) + (a[i] - a[0]);
  }
  hashvalue = (((j * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_Dihed_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomID fields of two Diheds match */
  const int32 *a1 = (const int32 *) key1;
  const int32 *a2 = (const int32 *) key2;
  int32 i, diff = 0;
  for (i = 0;  i < NUM_DIHED_ATOM;  i++) {
    if ((diff = a1[i] - a2[i]) != 0) return diff;
  }
  return 0;
}


/*
 * low level routines for using Arrmap with Impr array
 */
const void *g_Impr_key(const Array *impr, int32 i) {
  const Impr *arr = Array_data_const(impr);
  ASSERT_P(i >= 0 && i < Array_length(impr));
  return arr[i].atomID;
}

int32 g_Impr_hash(const Arrmap *arrmap, const void *key) {
  const int32 *a = (const int32 *) key;
  int32 hashvalue, i, j;
  j = a[0];
  for (i = 1;  i < NUM_IMPR_ATOM;  i++) {
    j = (j << 6) + (a[i] - a[0]);
  }
  hashvalue = (((j * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_Impr_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomID fields of two Imprs match */
  const int32 *a1 = (const int32 *) key1;
  const int32 *a2 = (const int32 *) key2;
  int32 i, diff = 0;
  for (i = 0;  i < NUM_IMPR_ATOM;  i++) {
    if ((diff = a1[i] - a2[i]) != 0) return diff;
  }
  return 0;
}


/*
 * low level routines for using Arrmap with Excl array
 */
const void *g_Excl_key(const Array *excl, int32 i) {
  const Excl *arr = Array_data_const(excl);
  ASSERT_P(i >= 0 && i < Array_length(excl));
  return arr[i].atomID;
}

int32 g_Excl_hash(const Arrmap *arrmap, const void *key) {
  const int32 *a = (const int32 *) key;
  int32 hashvalue, i, j;
  j = a[0];
  for (i = 1;  i < NUM_EXCL_ATOM;  i++) {
    j = (j << 6) + (a[i] - a[0]);
  }
  hashvalue = (((j * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_Excl_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomID fields of two Excls match */
  const int32 *a1 = (const int32 *) key1;
  const int32 *a2 = (const int32 *) key2;
  int32 i, diff = 0;
  for (i = 0;  i < NUM_EXCL_ATOM;  i++) {
    if ((diff = a1[i] - a2[i]) != 0) return diff;
  }
  return 0;
}
