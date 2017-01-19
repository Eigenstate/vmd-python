/* forceprm_init.c */

#include <string.h>
#include "moltypes/forceprm.h"


static int g_DihedPrm_init(void *v);
static void g_DihedPrm_done(void *v);
static int g_DihedPrm_copy(void *vdest, const void *cvsrc);
static int g_DihedPrm_erase(void *v);

static const void *g_AtomPrm_key(const Array *, int32 i);
static int32 g_AtomPrm_hash(const Arrmap *, const void *key);
static int32 g_AtomPrm_keycmp(const void *key1, const void *key2);

static const void *g_BondPrm_key(const Array *, int32 i);
static int32 g_BondPrm_hash(const Arrmap *, const void *key);
static int32 g_BondPrm_keycmp(const void *key1, const void *key2);

static const void *g_AnglePrm_key(const Array *, int32 i);
static int32 g_AnglePrm_hash(const Arrmap *, const void *key);
static int32 g_AnglePrm_keycmp(const void *key1, const void *key2);

static const void *g_DihedPrm_key(const Array *, int32 i);
static int32 g_DihedPrm_hash(const Arrmap *, const void *key);
static int32 g_DihedPrm_keycmp(const void *key1, const void *key2);

static const void *g_ImprPrm_key(const Array *, int32 i);
static int32 g_ImprPrm_hash(const Arrmap *, const void *key);
static int32 g_ImprPrm_keycmp(const void *key1, const void *key2);

static const void *g_VdwpairPrm_key(const Array *, int32 i);
static int32 g_VdwpairPrm_hash(const Arrmap *, const void *key);
static int32 g_VdwpairPrm_keycmp(const void *key1, const void *key2);

static const void *g_BuckpairPrm_key(const Array *, int32 i);
static int32 g_BuckpairPrm_hash(const Arrmap *, const void *key);
static int32 g_BuckpairPrm_keycmp(const void *key1, const void *key2);


/* ForcePrm constructor */
int ForcePrm_init(ForcePrm *f) {
  int s;  /* error status */

  if ((s=Array_init(&(f->vdwtable), sizeof(VdwTableElem))) != OK) {
    return ERROR(s);
  }
  f->vdwtablelen = 0;
  if ((s=Array_init(&(f->bucktable), sizeof(BuckTableElem))) != OK) {
    return ERROR(s);
  }
  f->bucktablelen = 0;

  if ((s=Array_init(&(f->atomprm), sizeof(AtomPrm))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->bondprm), sizeof(BondPrm))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->angleprm), sizeof(AnglePrm))) != OK) return ERROR(s);
  if ((s=Objarr_init(&(f->dihedprm), sizeof(DihedPrm), g_DihedPrm_init,
          g_DihedPrm_done, g_DihedPrm_copy, g_DihedPrm_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Array_init(&(f->imprprm), sizeof(ImprPrm))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->vdwpairprm), sizeof(VdwpairPrm))) != OK) {
    return ERROR(s);
  }
  if ((s=Array_init(&(f->buckpairprm), sizeof(BuckpairPrm))) != OK) {
    return ERROR(s);
  }

  if ((s=Arrmap_init(&(f->bondprm_map), &(f->bondprm),
          g_BondPrm_key, g_BondPrm_hash, g_BondPrm_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(f->angleprm_map), &(f->angleprm),
          g_AnglePrm_key, g_AnglePrm_hash, g_AnglePrm_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(f->dihedprm_map), (const Array *) &(f->dihedprm),
          g_DihedPrm_key, g_DihedPrm_hash, g_DihedPrm_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(f->imprprm_map), &(f->imprprm),
          g_ImprPrm_key, g_ImprPrm_hash, g_ImprPrm_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(f->atomprm_map), &(f->atomprm),
          g_AtomPrm_key, g_AtomPrm_hash, g_AtomPrm_keycmp, 0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(f->vdwpairprm_map), &(f->vdwpairprm),
          g_VdwpairPrm_key, g_VdwpairPrm_hash, g_VdwpairPrm_keycmp,
          0)) != OK) {
    return ERROR(s);
  }
  if ((s=Arrmap_init(&(f->buckpairprm_map), &(f->buckpairprm),
          g_BuckpairPrm_key, g_BuckpairPrm_hash, g_BuckpairPrm_keycmp,
          0)) != OK) {
    return ERROR(s);
  }

  if ((s=Array_init(&(f->bondprm_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->angleprm_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->dihedprm_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->imprprm_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->atomprm_open), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->vdwpairprm_open), sizeof(int32))) != OK) {
    return ERROR(s);
  }
  if ((s=Array_init(&(f->buckpairprm_open), sizeof(int32))) != OK) {
    return ERROR(s);
  }

  f->status = 0;

  /* add the "zero"-valued parameters as leading array element */
  {
    AtomPrm atomp;
    BondPrm bondp;
    AnglePrm anglep;
    ImprPrm imprp;
    DihedPrm dihedp;
    DihedTerm dt;

    memset(&atomp, 0, sizeof(AtomPrm));
    memset(&bondp, 0, sizeof(BondPrm));
    memset(&anglep, 0, sizeof(AnglePrm));
    memset(&imprp, 0, sizeof(ImprPrm));
    memset(&dihedp, 0, sizeof(DihedPrm));
    memset(&dt, 0, sizeof(DihedTerm));

    if ((s=Array_append(&(f->atomprm), &atomp)) != OK) return ERROR(s);
    if ((s=Array_append(&(f->bondprm), &bondp)) != OK) return ERROR(s);
    if ((s=Array_append(&(f->angleprm), &anglep)) != OK) return ERROR(s);
    if ((s=Array_append(&(f->imprprm), &imprp)) != OK) return ERROR(s);

    if ((s=DihedPrm_init(&dihedp)) != OK) return ERROR(s);
    if ((s=DihedPrm_add_term(&dihedp, &dt)) != OK) return ERROR(s);

    if ((s=Objarr_append(&(f->dihedprm), &dihedp)) != OK) {
      DihedPrm_done(&dihedp);
      return ERROR(s);
    }
    DihedPrm_done(&dihedp);
  }

  return OK;
}


/* ForcePrm destructor */
void ForcePrm_done(ForcePrm *f) {
  Array_done(&(f->vdwtable));
  Array_done(&(f->bucktable));

  Array_done(&(f->bondprm));
  Array_done(&(f->angleprm));
  Objarr_done(&(f->dihedprm));
  Array_done(&(f->imprprm));
  Array_done(&(f->atomprm));
  Array_done(&(f->vdwpairprm));
  Array_done(&(f->buckpairprm));

  Arrmap_done(&(f->bondprm_map));
  Arrmap_done(&(f->angleprm_map));
  Arrmap_done(&(f->dihedprm_map));
  Arrmap_done(&(f->imprprm_map));
  Arrmap_done(&(f->atomprm_map));
  Arrmap_done(&(f->vdwpairprm_map));
  Arrmap_done(&(f->buckpairprm_map));

  Array_done(&(f->bondprm_open));
  Array_done(&(f->angleprm_open));
  Array_done(&(f->dihedprm_open));
  Array_done(&(f->imprprm_open));
  Array_done(&(f->atomprm_open));
  Array_done(&(f->vdwpairprm_open));
  Array_done(&(f->buckpairprm_open));
}


/* DihedPrm constructor, destructor, copy, and erase */
int DihedPrm_init(DihedPrm *p) {
  int s;
  memset(p->atomType, 0, sizeof(p->atomType));
  if ((s=Array_init(&(p->dihedterm), sizeof(DihedTerm))) != OK) {
    return ERROR(s);
  }
  return OK;
}

void DihedPrm_done(DihedPrm *p) {
  Array_done(&(p->dihedterm));
}

int DihedPrm_copy(DihedPrm *dest, const DihedPrm *src) {
  int s;
  memcpy(dest->atomType, src->atomType, sizeof(dest->atomType));
  if ((s=Array_copy(&(dest->dihedterm), &(src->dihedterm))) != OK) {
    return ERROR(s);
  }
  return OK;
}

int DihedPrm_setmaxnum_term(DihedPrm *p, int32 n) {
  int s;
  if (n < Array_length(&(p->dihedterm))) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(&(p->dihedterm), n)) != OK) return ERROR(s);
  return OK;
}

int DihedPrm_add_term(DihedPrm *p, const DihedTerm *t) {
  int s;
  if ((s=Array_append(&(p->dihedterm), t)) != OK) return ERROR(s);
  return OK;
}

const DihedTerm *DihedPrm_term_array(const DihedPrm *p) {
  return Array_data_const(&(p->dihedterm));
}

int32 DihedPrm_term_array_length(const DihedPrm *p) {
  return Array_length(&(p->dihedterm));
}

int DihedPrm_erase(DihedPrm *p) {
  int s;
  memset(p->atomType, 0, sizeof(p->atomType));
  if ((s=Array_setbuflen(&(p->dihedterm), 0)) != OK) return ERROR(s);
  return OK;
}


/* need generic wrappers of above routines for Objarr */
int g_DihedPrm_init(void *v) {
  return DihedPrm_init((DihedPrm *) v);
}

void g_DihedPrm_done(void *v) {
  DihedPrm_done((DihedPrm *) v);
}

int g_DihedPrm_copy(void *vdest, const void *cvsrc) {
  return DihedPrm_copy((DihedPrm *) vdest, (const DihedPrm *) cvsrc);
}

int g_DihedPrm_erase(void *v) {
  return DihedPrm_erase((DihedPrm *) v);
}


/* needed for hashing functions below */
#define HASH_PRIME  1103515249


/*
 * low level routines for using Arrmap with BondPrm array
 */
const void *g_BondPrm_key(const Array *bondprm, int32 i) {
  const BondPrm *arr = Array_data_const(bondprm);
  ASSERT_P(i >= 0 && i < Array_length(bondprm));
  return arr[i].atomType;
}

int32 g_BondPrm_hash(const Arrmap *arrmap, const void *key) {
  const AtomType *a = (const AtomType *) key;
  int32 i, j, k = 0;
  int32 hashvalue;
  for (i = 0;  i < NUM_BONDPRM_ATOMTYPE;  i++) {
    for (j = 0;  a[i][j] != 0;  j++) {
      k = (k << 3) + (a[i][j] - '0');
    }
    k = (k << 3) + (' ' - '0');  /* insert a space between atom types */
  }
  hashvalue = (((k * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_BondPrm_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomType fields of two BondPrms match */
  const AtomType *a1 = (const AtomType *) key1;
  const AtomType *a2 = (const AtomType *) key2;
  int32 i, j, diff = 0;
  for (i = 0;  i < NUM_BONDPRM_ATOMTYPE;  i++) {
    for (j = 0;  a1[i][j] != 0 || a2[i][j] != 0;  j++) {
      if ((diff = a1[i][j] - a2[i][j]) != 0) return diff;
    }
  }
  return 0;
}


/*
 * low level routines for using Arrmap with AnglePrm array
 */
const void *g_AnglePrm_key(const Array *angleprm, int32 i) {
  const AnglePrm *arr = Array_data_const(angleprm);
  ASSERT_P(i >= 0 && i < Array_length(angleprm));
  return arr[i].atomType;
}

int32 g_AnglePrm_hash(const Arrmap *arrmap, const void *key) {
  const AtomType *a = (const AtomType *) key;
  int32 i, j, k = 0;
  int32 hashvalue;
  for (i = 0;  i < NUM_ANGLEPRM_ATOMTYPE;  i++) {
    for (j = 0;  a[i][j] != 0;  j++) {
      k = (k << 3) + (a[i][j] - '0');
    }
    k = (k << 3) + (' ' - '0');  /* insert a space between atom types */
  }
  hashvalue = (((k * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_AnglePrm_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomType fields of two AnglePrms match */
  const AtomType *a1 = (const AtomType *) key1;
  const AtomType *a2 = (const AtomType *) key2;
  int32 i, j, diff = 0;
  for (i = 0;  i < NUM_ANGLEPRM_ATOMTYPE;  i++) {
    for (j = 0;  a1[i][j] != 0 || a2[i][j] != 0;  j++) {
      if ((diff = a1[i][j] - a2[i][j]) != 0) return diff;
    }
  }
  return 0;
}


/*
 * low level routines for using Arrmap with DihedPrm array
 */
const void *g_DihedPrm_key(const Array *a_dihedprm, int32 i) {
  const Objarr *dihedprm = (const Objarr *) a_dihedprm;
  const DihedPrm *arr = Objarr_data_const(dihedprm);
  ASSERT_P(i >= 0 && i < Objarr_length(dihedprm));
  return arr[i].atomType;
}

int32 g_DihedPrm_hash(const Arrmap *arrmap, const void *key) {
  const AtomType *a = (const AtomType *) key;
  int32 i, j, k = 0;
  int32 hashvalue;
  for (i = 0;  i < NUM_DIHEDPRM_ATOMTYPE;  i++) {
    for (j = 0;  a[i][j] != 0;  j++) {
      k = (k << 3) + (a[i][j] - '0');
    }
    k = (k << 3) + (' ' - '0');  /* insert a space between atom types */
  }
  hashvalue = (((k * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_DihedPrm_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomType fields of two DihedPrms match */
  const AtomType *a1 = (const AtomType *) key1;
  const AtomType *a2 = (const AtomType *) key2;
  int32 i, j, diff = 0;
  for (i = 0;  i < NUM_DIHEDPRM_ATOMTYPE;  i++) {
    for (j = 0;  a1[i][j] != 0 || a2[i][j] != 0;  j++) {
      if ((diff = a1[i][j] - a2[i][j]) != 0) return diff;
    }
  }
  return 0;
}


/*
 * low level routines for using Arrmap with ImprPrm array
 */
const void *g_ImprPrm_key(const Array *imprprm, int32 i) {
  const ImprPrm *arr = Array_data_const(imprprm);
  ASSERT_P(i >= 0 && i < Array_length(imprprm));
  return arr[i].atomType;
}

int32 g_ImprPrm_hash(const Arrmap *arrmap, const void *key) {
  const AtomType *a = (const AtomType *) key;
  int32 i, j, k = 0;
  int32 hashvalue;
  for (i = 0;  i < NUM_IMPRPRM_ATOMTYPE;  i++) {
    for (j = 0;  a[i][j] != 0;  j++) {
      k = (k << 3) + (a[i][j] - '0');
    }
    k = (k << 3) + (' ' - '0');  /* insert a space between atom types */
  }
  hashvalue = (((k * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_ImprPrm_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomType fields of two ImprPrms match */
  const AtomType *a1 = (const AtomType *) key1;
  const AtomType *a2 = (const AtomType *) key2;
  int32 i, j, diff = 0;
  for (i = 0;  i < NUM_IMPRPRM_ATOMTYPE;  i++) {
    for (j = 0;  a1[i][j] != 0 || a2[i][j] != 0;  j++) {
      if ((diff = a1[i][j] - a2[i][j]) != 0) return diff;
    }
  }
  return 0;
}


/*
 * low level routines for using Arrmap with AtomPrm array
 */
const void *g_AtomPrm_key(const Array *atomprm, int32 i) {
  const AtomPrm *arr = Array_data_const(atomprm);
  ASSERT_P(i >= 0 && i < Array_length(atomprm));
  return arr[i].atomType;
}

int32 g_AtomPrm_hash(const Arrmap *arrmap, const void *key) {
  const AtomType *a = (const AtomType *) key;
  int32 i, j, k = 0;
  int32 hashvalue;
  for (i = 0;  i < NUM_ATOMPRM_ATOMTYPE;  i++) {
    for (j = 0;  a[i][j] != 0;  j++) {
      k = (k << 3) + (a[i][j] - '0');
    }
    k = (k << 3) + (' ' - '0');  /* insert a space between atom types */
  }
  hashvalue = (((k * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_AtomPrm_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomType fields of two AtomPrms match */
  const AtomType *a1 = (const AtomType *) key1;
  const AtomType *a2 = (const AtomType *) key2;
  int32 i, j, diff = 0;
  for (i = 0;  i < NUM_ATOMPRM_ATOMTYPE;  i++) {
    for (j = 0;  a1[i][j] != 0 || a2[i][j] != 0;  j++) {
      if ((diff = a1[i][j] - a2[i][j]) != 0) return diff;
    }
  }
  return 0;
}


/*
 * low level routines for using Arrmap with VdwpairPrm array
 */
const void *g_VdwpairPrm_key(const Array *vdwpairprm, int32 i) {
  const VdwpairPrm *arr = Array_data_const(vdwpairprm);
  ASSERT_P(i >= 0 && i < Array_length(vdwpairprm));
  return arr[i].atomType;
}

int32 g_VdwpairPrm_hash(const Arrmap *arrmap, const void *key) {
  const AtomType *a = (const AtomType *) key;
  int32 i, j, k = 0;
  int32 hashvalue;
  for (i = 0;  i < NUM_VDWPAIRPRM_ATOMTYPE;  i++) {
    for (j = 0;  a[i][j] != 0;  j++) {
      k = (k << 3) + (a[i][j] - '0');
    }
    k = (k << 3) + (' ' - '0');  /* insert a space between atom types */
  }
  hashvalue = (((k * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_VdwpairPrm_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomType fields of two VdwpairPrms match */
  const AtomType *a1 = (const AtomType *) key1;
  const AtomType *a2 = (const AtomType *) key2;
  int32 i, j, diff = 0;
  for (i = 0;  i < NUM_VDWPAIRPRM_ATOMTYPE;  i++) {
    for (j = 0;  a1[i][j] != 0 || a2[i][j] != 0;  j++) {
      if ((diff = a1[i][j] - a2[i][j]) != 0) return diff;
    }
  }
  return 0;
}


/*
 * low level routines for using Arrmap with BuckpairPrm array
 */
const void *g_BuckpairPrm_key(const Array *buckpairprm, int32 i) {
  const BuckpairPrm *arr = Array_data_const(buckpairprm);
  ASSERT_P(i >= 0 && i < Array_length(buckpairprm));
  return arr[i].atomType;
}

int32 g_BuckpairPrm_hash(const Arrmap *arrmap, const void *key) {
  const AtomType *a = (const AtomType *) key;
  int32 i, j, k = 0;
  int32 hashvalue;
  for (i = 0;  i < NUM_BUCKPAIRPRM_ATOMTYPE;  i++) {
    for (j = 0;  a[i][j] != 0;  j++) {
      k = (k << 3) + (a[i][j] - '0');
    }
    k = (k << 3) + (' ' - '0');  /* insert a space between atom types */
  }
  hashvalue = (((k * HASH_PRIME) >> arrmap->downshift) & arrmap->mask);
  if (hashvalue < 0) hashvalue = 0;
  return hashvalue;
}

int32 g_BuckpairPrm_keycmp(const void *key1, const void *key2) {
  /* returns 0 if atomType fields of two BuckpairPrms match */
  const AtomType *a1 = (const AtomType *) key1;
  const AtomType *a2 = (const AtomType *) key2;
  int32 i, j, diff = 0;
  for (i = 0;  i < NUM_BUCKPAIRPRM_ATOMTYPE;  i++) {
    for (j = 0;  a1[i][j] != 0 || a2[i][j] != 0;  j++) {
      if ((diff = a1[i][j] - a2[i][j]) != 0) return diff;
    }
  }
  return 0;
}
