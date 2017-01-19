#include <stdlib.h>
#include <string.h>
#include "moltypes/exclude.h"


/* wrappers for building Objarr containing Idlist elements */
static int vIdlist_init(void *pv) {
  return Idlist_init((Idlist *) pv);
}

static void vIdlist_done(void *pv) {
  Idlist_done((Idlist *) pv);
}

static int vIdlist_copy(void *pvdest, const void *pvsrc) {
  return Idlist_copy((Idlist *) pvdest, (const Idlist *) pvsrc);
}

static int vIdlist_erase(void *pv) {
  return Idlist_erase((Idlist *) pv);
}


int Exclude_init(Exclude *p, const Topology *topo) {
  int s;  /* error status */
  if (NULL==topo || NULL==topo->fprm) return ERROR(ERR_EXPECT);
  memset(p, 0, sizeof(Exclude));
  p->fprm = topo->fprm;
  p->topo = topo;
  if ((s=Objarr_init(&(p->exclx), sizeof(Idlist),
          vIdlist_init, vIdlist_done, vIdlist_copy, vIdlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(p->excl12), sizeof(Idlist),
          vIdlist_init, vIdlist_done, vIdlist_copy, vIdlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(p->excl13), sizeof(Idlist),
          vIdlist_init, vIdlist_done, vIdlist_copy, vIdlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(p->excl14), sizeof(Idlist),
          vIdlist_init, vIdlist_done, vIdlist_copy, vIdlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(p->only14), sizeof(Idlist),
          vIdlist_init, vIdlist_done, vIdlist_copy, vIdlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(p->excllist), sizeof(Idlist),
          vIdlist_init, vIdlist_done, vIdlist_copy, vIdlist_erase)) != OK) {
    return ERROR(s);
  }
  if ((s=Objarr_init(&(p->scal14list), sizeof(Idlist),
          vIdlist_init, vIdlist_done, vIdlist_copy, vIdlist_erase)) != OK) {
    return ERROR(s);
  }
  return OK;
}


void Exclude_done(Exclude *p) {
  Objarr_done(&(p->exclx));
  Objarr_done(&(p->excl12));
  Objarr_done(&(p->excl13));
  Objarr_done(&(p->excl14));
  Objarr_done(&(p->only14));
  Objarr_done(&(p->excllist));
  Objarr_done(&(p->scal14list));
}


const Idlist *Exclude_excllist(const Exclude *p, int32 id) {
  const Idlist *pid;
  if ((pid=Objarr_elem_const(&(p->excllist), id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return pid;
}


const Idlist *Exclude_scal14list(const Exclude *p, int32 id) {
  const Idlist *pid;
  if ((pid=Objarr_elem_const(&(p->scal14list), id)) == NULL) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return pid;
}


int Exclude_pair(const Exclude *p, int32 i, int32 j) {
  const Idlist *pid;
  if ((pid=Objarr_elem_const(&(p->excllist), i)) == NULL) {
    return ERROR(ERR_RANGE);
  }
  return Idlist_find(pid, j) != FAIL;
}


static int finishup(Exclude *p);

/*
 * Build the nonbonded interaction exclusion lists.
 * These lists are built from Topology Excl and Bond arrays
 * and determined by the exclusion policy defined in ForcePrm.
 *
 * Generate for each atom ID i, a list of atom IDs j.
 * The semantics for the final results are
 *   excllist[i]   = { j : interaction (i,j) is to be excluded }
 *   scal14list[i] = { j : interaction (i,j) is a scaled 1-4 interaction }
 * where i \in { 0, ..., N-1 } (i.e. the set of atom IDs).
 *
 * Algorithm expressed in set notation:
 *   exclx[i]  = { j : there is an explicit exclusion (i,j) from Excl array }
 *   excl12[i] = { j : there is a bond (i,j) from Bond array }
 *   excl13[i] = excl12[i] U ( U_{ j \in excl12[i] } excl12[j] )
 *   excl14[i] = excl13[i] U ( U_{ j \in excl13[i] } excl12[j] )
 *   only14[i] = excl14[i] \ excl13[i]
 *
 *   excllist[i] = exclx[i],              if policy is EXCL_NONE
 *               = exclx[i] U excl12[i],  if policy is EXCL_12
 *               = exclx[i] U excl13[i],  if policy is EXCL_13
 *               = exclx[i] U excl14[i],  if policy is EXCL_14
 *
 *   excllist[i] = exclx[i] U excl13[i]
 *     AND
 *   scal14list[i] = only14[i] \ exclx[i],  if policy is EXCL_SCALED14
 */
int Exclude_setup(Exclude *p) {
  const Bond *bond = Topology_bond_array(p->topo);
  const Excl *excl = Topology_excl_array(p->topo);
  const int32 natoms = Topology_atom_array_length(p->topo);
  const int32 nbonds = Topology_bond_array_length(p->topo);
  const int32 nexcls = Topology_excl_array_length(p->topo);
  const int32 exclpolicy = ForcePrm_nonbprm(p->fprm)->exclude;
  Idlist *exclx, *excl12, *excl13, *excl14, *only14;
  Idlist *excllist, *scal14list;
  int32 i, j, k;
  int s;  /* error status */

  INT(exclpolicy);

  if ((s=Objarr_resize(&(p->excllist), natoms)) != OK) return ERROR(s);
  if ((s=Objarr_erase(&(p->excllist))) != OK) return ERROR(s);
  excllist = (Idlist *) Objarr_data(&(p->excllist));

  if ((s=Objarr_resize(&(p->exclx), natoms)) != OK) return ERROR(s);
  if ((s=Objarr_erase(&(p->exclx))) != OK) return ERROR(s);
  exclx = (Idlist *) Objarr_data(&(p->exclx));

  /* add self-interactions */
  for (k = 0;  k < natoms;  k++) {
    if ((s=Idlist_insert(exclx+k, k)) != OK) return ERROR(s);
  }

  /* add explicit exclusions */
  for (k = 0;  k < nexcls;  k++) {
    int32 i = excl[k].atomID[0];
    int32 j = excl[k].atomID[1];
    if (i >= 0 && j >= 0) {
      if ((s=Idlist_insert(exclx+i, j)) < FAIL) return ERROR(s);
      if ((s=Idlist_insert(exclx+j, i)) < FAIL) return ERROR(s);
    }
  }

  if (EXCL_NONE==exclpolicy) {
    for (k = 0;  k < natoms;  k++) {
      if ((s=Idlist_copy(excllist+k, exclx+k)) != OK) return ERROR(s);
    }
    return finishup(p);
  }

  if ((s=Objarr_resize(&(p->excl12), natoms)) != OK) return ERROR(s);
  if ((s=Objarr_erase(&(p->excl12))) != OK) return ERROR(s);
  excl12 = (Idlist *) Objarr_data(&(p->excl12));

  /* add 1-2 interactions */
  for (k = 0;  k < nbonds;  k++) {
    int32 i = bond[k].atomID[0];
    int32 j = bond[k].atomID[1];
    if (i >= 0 && j >= 0) {
      if ((s=Idlist_insert(excl12+i, j)) < FAIL) return ERROR(s);
      if ((s=Idlist_insert(excl12+j, i)) < FAIL) return ERROR(s);
    }
  }

  if (EXCL_12==exclpolicy) {
    for (k = 0;  k < natoms;  k++) {
      if ((s=Idlist_merge(excllist+k, exclx+k, excl12+k)) != OK) {
        return ERROR(s);
      }
    }
    return finishup(p);
  }

  if ((s=Objarr_resize(&(p->excl13), natoms)) != OK) return ERROR(s);
  if ((s=Objarr_erase(&(p->excl13))) != OK) return ERROR(s);
  excl13 = (Idlist *) Objarr_data(&(p->excl13));

  /* add 1-3 interactions, union over neighboring 1-2 interactions */
  for (i = 0;  i < natoms;  i++) {
    Idseq seq;
    if ((s=Idlist_copy(excl13+i, excl12+i)) != OK) return ERROR(s);
    if ((s=Idseq_init(&seq, excl12+i)) != OK) return ERROR(s);
    while ((j = Idseq_getid(&seq)) != FAIL) {
      if ((s=Idlist_merge_into(excl13+i, excl12+j)) != OK) return ERROR(s);
    }
    Idseq_done(&seq);
  }

  if (EXCL_13==exclpolicy) {
    for (k = 0;  k < natoms;  k++) {
      if ((s=Idlist_merge(excllist+k, exclx+k, excl13+k)) != OK) {
        return ERROR(s);
      }
    }
    return finishup(p);
  }

  if ((s=Objarr_resize(&(p->excl14), natoms)) != OK) return ERROR(s);
  if ((s=Objarr_erase(&(p->excl14))) != OK) return ERROR(s);
  excl14 = (Idlist *) Objarr_data(&(p->excl14));

  /* add 1-4 interactions, union over neighboring 1-3 interactions */
  for (i = 0;  i < natoms;  i++) {
    Idseq seq;
    if ((s=Idlist_copy(excl14+i, excl13+i)) != OK) return ERROR(s);
    if ((s=Idseq_init(&seq, excl13+i)) != OK) return ERROR(s);
    while ((j = Idseq_getid(&seq)) != FAIL) {
      if ((s=Idlist_merge_into(excl14+i, excl12+j)) != OK) return ERROR(s);
    }
    Idseq_done(&seq);
  }

  if (EXCL_14==exclpolicy) {
    for (k = 0;  k < natoms;  k++) {
      if ((s=Idlist_merge(excllist+k, exclx+k, excl14+k)) != OK) {
        return ERROR(s);
      }
    }
    return finishup(p);
  }

  ASSERT(EXCL_SCALED14==exclpolicy);

  if ((s=Objarr_resize(&(p->only14), natoms)) != OK) return ERROR(s);
  if ((s=Objarr_erase(&(p->only14))) != OK) return ERROR(s);
  only14 = (Idlist *) Objarr_data(&(p->only14));

  /* start with 1-4 interactions and remove interactions 1-3 and closer */
  for (i = 0;  i < natoms;  i++) {
    Idseq seq;
    if ((s=Idlist_copy(only14+i, excl14+i)) != OK) return ERROR(s);
    if ((s=Idseq_init(&seq, excl13+i)) != OK) return ERROR(s);
    while ((j = Idseq_getid(&seq)) != FAIL) {
      if ((s=Idlist_remove(only14+i, j)) < FAIL) return ERROR(s);
    }
    Idseq_done(&seq);
  }

  if ((s=Objarr_resize(&(p->scal14list), natoms)) != OK) return ERROR(s);
  if ((s=Objarr_erase(&(p->scal14list))) != OK) return ERROR(s);
  scal14list = (Idlist *) Objarr_data(&(p->scal14list));

  /* scaled 1-4 interactions are those in only14 less those in exclx */
  for (k = 0;  k < natoms;  k++) {
    Idseq seq;
    if ((s=Idlist_merge(excllist+k, exclx+k, excl13+k)) != OK) {
      return ERROR(s);
    }
    if ((s=Idlist_copy(scal14list+k, only14+k)) != OK) {
      return ERROR(s);
    }
    if ((s=Idseq_init(&seq, exclx+k)) != OK) return ERROR(s);
    while ((j = Idseq_getid(&seq)) != FAIL) {
      if ((s=Idlist_remove(scal14list+k, j)) < FAIL) return ERROR(s);
    }
    Idseq_done(&seq);
  }
  return finishup(p);
}


/*
 * Give back the memory needed to build the lookup arrays.
 * Sort remaining lists for faster access.
 */
int finishup(Exclude *p) {
  Idlist *excllist = Objarr_data(&(p->excllist));
  Idlist *scal14list = Objarr_data(&(p->scal14list));
  const int32 natoms = Topology_atom_array_length(p->topo);
  int32 i;
  int s;  /* error status */

  if ((s=Objarr_setbuflen(&(p->exclx), 0)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(p->excl12), 0)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(p->excl13), 0)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(p->excl14), 0)) != OK) return ERROR(s);
  if ((s=Objarr_setbuflen(&(p->only14), 0)) != OK) return ERROR(s);

  for (i = 0;  i < natoms;  i++) {
    Idlist_memsort(excllist+i);
    if (scal14list != NULL) Idlist_memsort(scal14list+i);
  }
  return OK;
}
