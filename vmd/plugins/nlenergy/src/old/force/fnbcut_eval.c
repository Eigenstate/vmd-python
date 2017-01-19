#include <math.h>
#include <string.h>
#include "force/fnbcut.h"
#include "force/nbpair.h"
#include "moltypes/vecops.h"


int Fnbcut_eval(Fnbcut *f, const dvec *pos, dvec *force, Energy *en) {
  const int32 *idmap = Array_data_const(&(f->idmap));
  const char *a = Array_data_const(&(f->alltrue));
  const int32 natoms = Topology_atom_array_length(f->topo);
  int s;  /* error status */
  if ((s=Fnbcut_eval_cellhash(f, pos, idmap, natoms)) != OK) return ERROR(s);
  if ((s=Fnbcut_eval_cellpairs(f, pos, force, en)) != OK) return ERROR(s);
  if ((s=Fnbcut_eval_scaled14(f, pos, force, en, a, a)) != OK) return ERROR(s);
  return OK;
}


int Fnbcut_eval_subset(Fnbcut *f, const dvec *pos, dvec *force, Energy *en,
    const int32 *atomID, int32 len) {
  char *isatomID = Array_data(&(f->isatomID1));
  const int32 natoms = Topology_atom_array_length(f->topo);
  int32 i;
  int s;

  if (NULL==isatomID) {  /* haven't yet allocated needed array space */
    if ((s=Array_resize(&(f->isatomID1), natoms)) != OK) return ERROR(s);
    isatomID = Array_data(&(f->isatomID1));
  }

  memset(isatomID, 0, natoms*sizeof(char));
  ASSERT(0 < len && len <= natoms);
  for (i = 0;  i < len;  i++) {
    ASSERT(0 <= atomID[i] && atomID[i] < natoms);
    isatomID[ atomID[i] ] = (char)TRUE;
  }

  if ((s=Fnbcut_eval_cellhash(f, pos, atomID, len)) != OK) return ERROR(s);
  if ((s=Fnbcut_eval_cellpairs(f, pos, force, en)) != OK) return ERROR(s);
  if ((s=Fnbcut_eval_scaled14(f, pos, force, en, isatomID, isatomID)) != OK) {
    return ERROR(s);
  }
  return OK;
}


int Fnbcut_eval_disjoint_subsets(Fnbcut *f,
    const dvec *pos, dvec *force, Energy *en,
    const int32 *atomID1, int32 len1,
    const int32 *atomID2, int32 len2) {
  Energy esubtr;
  char *isatomID1 = Array_data(&(f->isatomID1));
  char *isatomID2 = Array_data(&(f->isatomID2));
  int32 *atomID12 = Array_data(&(f->atomID12));
  dvec *fsubtr = Array_data(&(f->fsubtr));
  const int32 natoms = Topology_atom_array_length(f->topo);
  int32 i;
  int s;

  if (NULL==isatomID2) {  /* haven't yet allocated needed array space */
    if (NULL==isatomID1) {
      if ((s=Array_resize(&(f->isatomID1), natoms)) != OK) return ERROR(s);
      isatomID1 = Array_data(&(f->isatomID1));
    }
    if ((s=Array_resize(&(f->isatomID2), natoms)) != OK) return ERROR(s);
    if ((s=Array_resize(&(f->atomID12), natoms)) != OK) return ERROR(s);
    if ((s=Array_resize(&(f->fsubtr), natoms)) != OK) return ERROR(s);
    isatomID2 = Array_data(&(f->isatomID2));
    atomID12 = Array_data(&(f->atomID12));
    fsubtr = Array_data(&(f->fsubtr));
  }

  memset(&esubtr, 0, sizeof(Energy));
  memset(isatomID1, 0, natoms*sizeof(char));
  memset(isatomID2, 0, natoms*sizeof(char));
  memset(atomID12, 0, natoms*sizeof(int32));
  memset(fsubtr, 0, natoms*sizeof(dvec));
  ASSERT(0 < len1 && len1 < natoms);
  ASSERT(0 < len2 && len2 < natoms);
  ASSERT(len1 + len2 <= natoms);
  for (i = 0;  i < len1;  i++) {
    ASSERT(0 <= atomID1[i] && atomID1[i] < natoms);
    isatomID1[ atomID1[i] ] = (char)TRUE;
  }
  for (i = 0;  i < len2;  i++) {
    ASSERT(0 <= atomID2[i] && atomID2[i] < natoms);
    isatomID2[ atomID2[i] ] = (char)TRUE;
  }
  for (i = 0;  i < len1;  i++) {
    atomID12[i] = atomID1[i];
  }
  for (i = 0;  i < len2;  i++) {
    atomID12[i+len1] = atomID2[i];
  }

  /* evaluate self-interactions from set atomID1 */
  if ((s=Fnbcut_eval_cellhash(f, pos, atomID1, len1)) != OK) return ERROR(s);
  if ((s=Fnbcut_eval_cellpairs(f, pos, fsubtr, &esubtr)) != OK) return ERROR(s);

  /* evaluate self-interactions from set atomID2 */
  if ((s=Fnbcut_eval_cellhash(f, pos, atomID2, len2)) != OK) return ERROR(s);
  if ((s=Fnbcut_eval_cellpairs(f, pos, fsubtr, &esubtr)) != OK) return ERROR(s);

  /* evaluate all interactions from combined sets atomID1 and atomID2 */
  if ((s=Fnbcut_eval_cellhash(f, pos, atomID12, len1+len2)) != OK) {
    return ERROR(s);
  }
  if ((s=Fnbcut_eval_cellpairs(f, pos, force, en)) != OK) return ERROR(s);

  /* subtract out self-interactions */
  for (i = 0;  i < natoms;  i++) {
    VECSUB(force[i], force[i], fsubtr[i]);
  }
  en->pe_elec -= esubtr.pe_elec;
  en->pe_vdw -= esubtr.pe_vdw;
  en->pe_buck -= esubtr.pe_buck;

  /* fix scaled 1-4 interactions shared between the two sets */
  if ((s=Fnbcut_eval_scaled14(f, pos, force, en, isatomID1, isatomID2))!=OK) {
    return ERROR(s);
  }
  return OK;
}


int Fnbcut_eval_cellhash(Fnbcut *f, const dvec *pos,
    const int32 *atomID, int32 len) {
  const Domain *domain = f->domain;
  FnbcutCell *cell = Array_data(&(f->cell));
  int32 *next = Array_data(&(f->next));
  const dvec hashorigin = f->hashorigin;
  const dvec hashfactor = f->hashfactor;
  const ivec celldim = f->celldim;
  const int32 ncells = f->ncells;
  int32 id, n, nc, i, j, k;
#if 0
  dvec a0, a1, a2;
#endif

  ASSERT(domain != NULL);
#if 0
  a0 = domain->basis[0];
  a1 = domain->basis[1];
  a2 = domain->basis[2];
#endif
  ASSERT(cell != NULL);
  ASSERT(ncells == celldim.x * celldim.y * celldim.z);
  ASSERT(next != NULL);
  ASSERT(Array_length(&(f->next)) == Topology_atom_array_length(f->topo));
  for (nc = 0;  nc < ncells;  nc++) {
    cell[nc].head = -1;
    cell[nc].cnt = 0;
  }
  for (id = 0;  id < len;  id++) {
    dvec s;
    ivec nv;
    n = atomID[id];
    ASSERT(n >= 0 && n < Topology_atom_array_length(f->topo));
    Domain_normalize_vec(domain, &s, &nv, &pos[n]);
    /* NOTE: might not have nv=(0,0,0) due to roundoff error */
#if 0
    /* determine position wrapped into domain */
    rpos[n].x = pos[n].x + -nv.x*a0.x + -nv.y*a1.x + -nv.z*a2.x;
    rpos[n].y = pos[n].y + -nv.x*a0.y + -nv.y*a1.y + -nv.z*a2.y;
    rpos[n].z = pos[n].z + -nv.x*a0.z + -nv.y*a1.z + -nv.z*a2.z;
#endif
    /* find grid cell index */
    i = (int32) floor((s.x - hashorigin.x) * hashfactor.x);
    j = (int32) floor((s.y - hashorigin.y) * hashfactor.y);
    k = (int32) floor((s.z - hashorigin.z) * hashfactor.z);
    /*
     * adjust for nonperiodic boundaries and roundoff errors
     *
     * NOTE: correct evaluation for periodic case requires that
     *   system is wrapped prior to evaluation.
     */
    if      (i < 0)          i = 0;
    else if (i >= celldim.x) i = celldim.x - 1;
    if      (j < 0)          j = 0;
    else if (j >= celldim.y) j = celldim.y - 1;
    if      (k < 0)          k = 0;
    else if (k >= celldim.z) k = celldim.z - 1;
    nc = (k * celldim.y + j) * celldim.x + i;
    next[n] = cell[nc].head;
    cell[nc].head = n;
    cell[nc].cnt++;
  }
#if 0
#ifdef DEBUG
  nc = 0;
  for (k = 0;  k < celldim.z;  k++) {
    for (j = 0;  j < celldim.y;  j++) {
      for (i = 0;  i < celldim.x;  i++) {
        int32 index = (k*celldim.y + j)*celldim.x + i;
        printf("cell[%d,%d,%d] has %d atoms\n", i, j, k, cell[index].cnt);
        nc += cell[index].cnt;
      }
    }
  }
  printf("%d atoms total\n", nc);
#endif
#endif

  return OK;
}


int Fnbcut_eval_cellpairs(Fnbcut *fc, const dvec *pos, dvec *f, Energy *en) {
  const int32 vdwtablelen = ForcePrm_vdwtable_length(fc->fprm);
  const int32 bucktablelen = ForcePrm_bucktable_length(fc->fprm);
  const int32 ncells = fc->ncells;
  const FnbcutCell *cell = Array_data_const(&(fc->cell));
  const int32 *next = Array_data_const(&(fc->next));
  const Atom *atom = Topology_atom_array(fc->topo);
  const VdwTableElem *vdwtable = ForcePrm_vdwtable_array(fc->fprm);
  const BuckTableElem *bucktable = ForcePrm_bucktable_array(fc->fprm);
  const dvec *imageTable = fc->imageTable;
  const Exclude *exclude = fc->exclude;
  const dreal elec_const = fc->elec_const;
  const dreal cutoff2 = fc->cutoff * fc->cutoff;
  const dreal inv_cutoff2 = fc->inv_cutoff2;
  const dreal switchdist2 = fc->switchdist2;
  const dreal inv_denom_switch = fc->inv_denom_switch;
  const dreal ewald_coef = fc->ewald_coef;
  const dreal ewald_grad_coef = fc->ewald_grad_coef;
  dvec fj, pj, image;
  dvec fsum_ij, r_ij;
  dreal virial[NELEMS_VIRIAL] = { 0 };
  dreal e_elec_sum = 0;
  dreal e_vdw_sum = 0;
  dreal e_buck_sum = 0;
  dreal qj;
  dreal u, du_r, r2;
  const int32 elec_pair_potential = (fc->nbprm.nbpairType & FNBCUT_ELEC_MASK);
  const int32 vdw_pair_potential = (fc->nbprm.nbpairType & FNBCUT_VDW_MASK);
  const int32 buck_pair_potential = (fc->nbprm.nbpairType & FNBCUT_BUCK_MASK);
  const boolean is_elec = (elec_pair_potential != 0);
  const boolean is_vdw = (vdw_pair_potential != 0);
  const boolean is_buck = (buck_pair_potential != 0);
  int32 i, j, k, n, nbrcnt, ihead, jhead;
  int32 vdwtablerow;
  int32 bucktablerow;

  HEX(elec_pair_potential);
  INT(is_elec);

  /* loop over cells */
  for (n = 0;  n < ncells;  n++) {
    nbrcnt = cell[n].nbrcnt;
    jhead = cell[n].head;

    /* loop over half-shell of neighbors to this cell */
    for (k = 0;  k < nbrcnt;  k++) {
      image = imageTable[ (int)(cell[n].image[k]) ];
      ihead = cell[ cell[n].nbr[k] ].head;

      /* loop over all pairs of atoms */
      for (j = jhead;  j != -1;  j = next[j]) {

        /* subtracting wrap offset from p[j] is same as adding it to p[i] */
        VECSUB(pj, pos[j], image);

        /* accumulate into local storage for efficiency */
        VECZERO(fj);
        qj = atom[j].q;

        /* find jth row of tables */
        vdwtablerow = atom[j].atomPrmID * vdwtablelen;
        bucktablerow = atom[j].atomPrmID * bucktablelen;

        /* 0th neighbor cell is self-referential, must modify ihead */
        if (0==k) ihead = next[j];

        for (i = ihead;  i != -1;  i = next[i]) {

          VECSUB(r_ij, pj, pos[i]);
          r2 = VECLEN2(r_ij);
          if (r2 >= cutoff2) continue;
          if (Exclude_pair(exclude, i, j)) continue;

          VECZERO(fsum_ij);

          if (is_elec) {
            switch (elec_pair_potential) {
              case FNBCUT_ELEC_INFTY:
                NBPAIR_ELEC_INFTY(&u, &du_r, r2, elec_const);
                break;
              case FNBCUT_ELEC_CUTOFF:
                NBPAIR_ELEC_CUTOFF(&u, &du_r, r2, elec_const, inv_cutoff2);
                break;
              case FNBCUT_ELEC_EWALD:
                NBPAIR_ELEC_EWALD(&u, &du_r, r2, elec_const,
                    ewald_coef, ewald_grad_coef);
                break;
              default:
                return ERROR(ERR_EXPECT);
            }
            u *= atom[i].q * qj;
            du_r *= atom[i].q * qj;
            VECMUL(fsum_ij, -du_r, r_ij);
            e_elec_sum += u;
          }

          if (is_vdw) {
            const VdwTableElem *vt = vdwtable + vdwtablerow
              + atom[i].atomPrmID;
            dreal a = vt->a;
            dreal b = vt->b;
            switch (vdw_pair_potential) {
              case FNBCUT_VDW_INFTY:
                NBPAIR_VDW_INFTY(&u, &du_r, r2, a, b);
                break;
              case FNBCUT_VDW_CUTOFF:
                NBPAIR_VDW_CUTOFF(&u, &du_r, r2, a, b,
                    cutoff2, switchdist2, inv_denom_switch);
                break;
              default:
                return ERROR(ERR_EXPECT);
            }
            VECMADD(fsum_ij, -du_r, r_ij, fsum_ij);
            e_vdw_sum += u;
          }

          if (is_buck) {
            const BuckTableElem *bt = bucktable + bucktablerow
              + atom[i].atomPrmID;
            dreal a = bt->a;
            dreal b = bt->b;
            dreal c = bt->c;
            dreal as = bt->as;
            dreal bs = bt->bs;
            dreal rs2 = bt->rs2;
            switch (buck_pair_potential) {
              case FNBCUT_BUCK_INFTY:
                NBPAIR_BUCK_INFTY(&u, &du_r, r2, a, b, c, as, bs, rs2);
                break;
              case FNBCUT_BUCK_CUTOFF:
                NBPAIR_BUCK_CUTOFF(&u, &du_r, r2, a, b, c, as, bs, rs2,
                    cutoff2, switchdist2, inv_denom_switch);
                break;
              default:
                return ERROR(ERR_EXPECT);
            }
            VECMADD(fsum_ij, -du_r, r_ij, fsum_ij);
            e_buck_sum += u;
          }

          /* accumulate force on atoms i and j */
          VECADD(fj, fj, fsum_ij);
          VECSUB(f[i], f[i], fsum_ij);

          /* accumulate upper triangle of virial */
          virial[VIRIAL_XX] += fsum_ij.x * r_ij.x;
          virial[VIRIAL_XY] += fsum_ij.x * r_ij.y;
          virial[VIRIAL_XZ] += fsum_ij.x * r_ij.z;
          virial[VIRIAL_YY] += fsum_ij.y * r_ij.y;
          virial[VIRIAL_YZ] += fsum_ij.y * r_ij.z;
          virial[VIRIAL_ZZ] += fsum_ij.z * r_ij.z;

        } /* end i-loop */

        /* add accumulated force on atom j */
        VECADD(f[j], f[j], fj);

      } /* end j-loop */

    } /* end k-loop over cell neighbors */

  } /* end n-loop over all cells */

  /* accumulate energies */
  en->pe_elec += e_elec_sum;
  en->pe_vdw += e_vdw_sum;
  en->pe_buck += e_buck_sum;
  en->pe += (e_elec_sum + e_vdw_sum + e_buck_sum);

  en->f_virial[VIRIAL_XX] += virial[VIRIAL_XX];
  en->f_virial[VIRIAL_XY] += virial[VIRIAL_XY];
  en->f_virial[VIRIAL_XZ] += virial[VIRIAL_XZ];
  en->f_virial[VIRIAL_YY] += virial[VIRIAL_YY];
  en->f_virial[VIRIAL_YZ] += virial[VIRIAL_YZ];
  en->f_virial[VIRIAL_ZZ] += virial[VIRIAL_ZZ];

  return OK;
}


int Fnbcut_eval_scaled14(Fnbcut *fc, const dvec *pos, dvec *f, Energy *en,
    const char *isatomID1, const char *isatomID2) {
  const int32 vdwtablelen = ForcePrm_vdwtable_length(fc->fprm);
  const int32 natoms = Topology_atom_array_length(fc->topo);
  const Atom *atom = Topology_atom_array(fc->topo);
  const VdwTableElem *vdwtable = ForcePrm_vdwtable_array(fc->fprm);
  const Exclude *exclude = fc->exclude;
  const Domain *domain = fc->domain;
  const dreal scaling14 = fc->scaling14;
  const dreal elec_const = fc->elec_const * (scaling14 - 1.);
  const dreal cutoff2 = fc->cutoff * fc->cutoff;
  const dreal inv_cutoff2 = fc->inv_cutoff2;
  const dreal switchdist2 = fc->switchdist2;
  const dreal inv_denom_switch = fc->inv_denom_switch;
  const dreal ewald_coef = fc->ewald_coef;
  const dreal ewald_grad_coef = fc->ewald_grad_coef;
  dvec fj, pj;
  dvec fsum_ij, r_ij;
  dreal virial[NELEMS_VIRIAL] = { 0 };
  dreal e_elec_sum = 0;
  dreal e_vdw_sum = 0;
  dreal qj;
  dreal u, du_r, r2;
  const int32 elec_pair_potential = (fc->nbprm.nbpairType & FNBCUT_ELEC_MASK);
  const int32 vdw_pair_potential = (fc->nbprm.nbpairType & FNBCUT_VDW_MASK);
  const boolean is_elec = (elec_pair_potential != 0 && scaling14 != 1.);
  const boolean is_vdw = (vdw_pair_potential != 0);
  int32 i, j;
  int32 vdwtablerow;
  int s;

  if (fc->exclpolicy != EXCL_SCALED14) return OK;

#if 0
#ifdef DEBUG
  for (j = 0;  j < natoms;  j++) {
    Idseq seq;
    const Idlist *sc14list = Exclude_scal14list(exclude, j);
    NL_printf("scal14  %d: ", j);
    if ((s=Idseq_init(&seq, sc14list)) != OK) return ERROR(s);
    while ((i = Idseq_getid(&seq)) != FAIL) {
      NL_printf(" %d", i);
    }
    NL_printf("\n");
  }
#endif
#endif

  for (j = 0;  j < natoms;  j++) {
    Idseq seq;
    const Idlist *sc14list = Exclude_scal14list(exclude, j);

    /* accumulate into local storage for efficiency */
    pj = pos[j];
    VECZERO(fj);
    qj = atom[j].q;

    /* find jth row of tables */
    vdwtablerow = atom[j].atomPrmID * vdwtablelen;

    if ((s=Idseq_init(&seq, sc14list)) != OK) return ERROR(s);
    while ((i = Idseq_getid(&seq)) != FAIL) {
      if (i >= j) break;  /* end of sorted list, do (i,j)-pairs with i < j */
      if ( !((isatomID1[i]&&isatomID2[j]) || (isatomID2[i]&&isatomID1[j])) ) {
        continue;  /* make sure (i,j) are from different index subsets */
      }
      //INT(i);
      //INT(j);

      Domain_shortest_vec(domain, &r_ij, &pj, &pos[i]);
      r2 = VECLEN2(r_ij);
      if (r2 >= cutoff2) continue;

      VECZERO(fsum_ij);

      if (is_elec) {
        switch (elec_pair_potential) {
          case FNBCUT_ELEC_INFTY:
            NBPAIR_ELEC_INFTY(&u, &du_r, r2, elec_const);
            break;
          case FNBCUT_ELEC_CUTOFF:
            NBPAIR_ELEC_CUTOFF(&u, &du_r, r2, elec_const, inv_cutoff2);
            break;
          case FNBCUT_ELEC_EWALD:
            NBPAIR_ELEC_EWALD(&u, &du_r, r2, elec_const,
                ewald_coef, ewald_grad_coef);
            break;
          default:
            return ERROR(ERR_EXPECT);
        }
        u *= atom[i].q * qj;
        du_r *= atom[i].q * qj;
        VECMUL(fsum_ij, -du_r, r_ij);
        e_elec_sum += u;
      }

      if (is_vdw) {
        const VdwTableElem *vt = vdwtable + vdwtablerow + atom[i].atomPrmID;
        dreal a = vt->a14 - vt->a;  /* add modified-14, subtract original */
        dreal b = vt->b14 - vt->b;
        switch (vdw_pair_potential) {
          case FNBCUT_VDW_INFTY:
            NBPAIR_VDW_INFTY(&u, &du_r, r2, a, b);
            break;
          case FNBCUT_VDW_CUTOFF:
            NBPAIR_VDW_CUTOFF(&u, &du_r, r2, a, b,
                cutoff2, switchdist2, inv_denom_switch);
            break;
          default:
            return ERROR(ERR_EXPECT);
        }
        VECMADD(fsum_ij, -du_r, r_ij, fsum_ij);
        e_vdw_sum += u;
      }

      /* accumulate force on atoms i and j */
      VECADD(fj, fj, fsum_ij);
      VECSUB(f[i], f[i], fsum_ij);

      /* accumulate upper triangle of virial */
      virial[VIRIAL_XX] += fsum_ij.x * r_ij.x;
      virial[VIRIAL_XY] += fsum_ij.x * r_ij.y;
      virial[VIRIAL_XZ] += fsum_ij.x * r_ij.z;
      virial[VIRIAL_YY] += fsum_ij.y * r_ij.y;
      virial[VIRIAL_YZ] += fsum_ij.y * r_ij.z;
      virial[VIRIAL_ZZ] += fsum_ij.z * r_ij.z;

    } /* end while i-loop */
    Idseq_done(&seq);

    /* add accumulated force on atom j */
    VECADD(f[j], f[j], fj);

  } /* end j-loop */

  /* accumulate energies */
  en->pe_elec += e_elec_sum;
  en->pe_vdw += e_vdw_sum;
  en->pe += (e_elec_sum + e_vdw_sum);

  en->f_virial[VIRIAL_XX] += virial[VIRIAL_XX];
  en->f_virial[VIRIAL_XY] += virial[VIRIAL_XY];
  en->f_virial[VIRIAL_XZ] += virial[VIRIAL_XZ];
  en->f_virial[VIRIAL_YY] += virial[VIRIAL_YY];
  en->f_virial[VIRIAL_YZ] += virial[VIRIAL_YZ];
  en->f_virial[VIRIAL_ZZ] += virial[VIRIAL_ZZ];

  return OK;
}
