
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "topo_defs_struct.h"
#include "topo_mol_struct.h"

#if defined(_MSC_VER)
#define strcasecmp  stricmp
#define strncasecmp strnicmp
#endif

topo_mol * topo_mol_create(topo_defs *defs) {
  topo_mol *mol;
  if ( ! defs ) return 0;
  if ( (mol = (topo_mol*) malloc(sizeof(topo_mol))) ) {
    mol->newerror_handler_data = 0;
    mol->newerror_handler = 0;
    mol->defs = defs;
    mol->npatch = 0;
    mol->patches = 0;
    mol->curpatch = 0;
    mol->segment_hash = hasharray_create(
	(void**) &(mol->segment_array), sizeof(topo_mol_segment_t*));
    mol->buildseg = 0;
    mol->arena = memarena_create();
    mol->angle_arena = memarena_create();
    mol->dihedral_arena = memarena_create();
    if ( ! mol->segment_hash || ! mol->arena ) {
      topo_mol_destroy(mol);
      return 0;
    }
  }
  return mol;
}

void topo_mol_destroy(topo_mol *mol) {
  int i,n;
  topo_mol_segment_t *s;
  
  if ( ! mol ) return;

  n = hasharray_count(mol->segment_hash);
  for ( i=0; i<n; ++i ) {
    s = mol->segment_array[i];
    if ( ! s ) continue;
    hasharray_destroy(s->residue_hash);
  }
  hasharray_destroy(mol->segment_hash);
  memarena_destroy(mol->arena);
  memarena_destroy(mol->angle_arena);
  memarena_destroy(mol->dihedral_arena);
  free((void*)mol);
}

void topo_mol_error_handler(topo_mol *mol, void *v, void (*print_msg)(void *,const char *)) {
  if ( mol ) {
    mol->newerror_handler = print_msg;
    mol->newerror_handler_data = v;
  }
}

/* internal method */
static void topo_mol_log_error(topo_mol *mol, const char *msg) {
  if (mol && msg && mol->newerror_handler)
    mol->newerror_handler(mol->newerror_handler_data, msg);
}

static topo_mol_segment_t * topo_mol_get_seg(topo_mol *mol,
			const topo_mol_ident_t *target) {
  int iseg;
  char errmsg[64 + 3*NAMEMAXLEN];

  if ( ! mol ) return 0;
  iseg = hasharray_index(mol->segment_hash,target->segid);
  if ( iseg == HASHARRAY_FAIL ) {
    sprintf(errmsg,"no segment %s",target->segid);
    topo_mol_log_error(mol,errmsg);
    return 0;
  }
  return mol->segment_array[iseg];
}

static topo_mol_residue_t * topo_mol_get_res(topo_mol *mol,
			const topo_mol_ident_t *target, int irel) {
  int nres, ires;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  char errmsg[64 + 3*NAMEMAXLEN];
  seg = topo_mol_get_seg(mol,target);
  if ( ! seg ) return 0;
  nres = hasharray_count(seg->residue_hash);
  ires = hasharray_index(seg->residue_hash,target->resid);
  if ( ires == HASHARRAY_FAIL ) {
    sprintf(errmsg,"no residue %s of segment %s",
					target->resid,target->segid);
    topo_mol_log_error(mol,errmsg);
    return 0;
  }
  if ( (ires+irel) < 0 || (ires+irel) >= nres ) {
    res = seg->residue_array + ires;
    if ( irel < 0 )
      sprintf(errmsg,"no residue %d before %s:%s of segment %s",
		-1*irel,res->name,res->resid,target->segid);
    if ( irel > 0 )
      sprintf(errmsg,"no residue %d past %s:%s of segment %s",
		irel,res->name,res->resid,target->segid);
    topo_mol_log_error(mol,errmsg);
    return 0;
  }

  return (seg->residue_array + ires + irel);
}

static topo_mol_atom_t * topo_mol_get_atom(topo_mol *mol,
			const topo_mol_ident_t *target, int irel) {
  topo_mol_residue_t *res;
  topo_mol_atom_t *atom;
  char errmsg[64 + 3*NAMEMAXLEN];
  res = topo_mol_get_res(mol,target,irel);
  if ( ! res ) return 0;
  for ( atom = res->atoms; atom; atom = atom->next ) {
    if ( ! strcmp(target->aname,atom->name) ) break;
  }
  if ( ! atom ) {
    sprintf(errmsg,"no atom %s in residue %s:%s of segment %s",
		target->aname,res->name,res->resid,target->segid);
    topo_mol_log_error(mol,errmsg);
  }
  return atom;
}

static topo_mol_atom_t *topo_mol_get_atom_from_res(
    const topo_mol_residue_t *res, const char *aname) {
  topo_mol_atom_t *atom;
  for ( atom = res->atoms; atom; atom = atom->next ) {
    if ( ! strcmp(aname,atom->name) ) break;
  }
  return atom;
}

int topo_mol_segment(topo_mol *mol, const char *segid) {
  int i;
  topo_mol_segment_t *newitem;
  char errmsg[32 + NAMEMAXLEN];
  if ( ! mol ) return -1;
  mol->buildseg = 0;
  if ( NAMETOOLONG(segid) ) return -2;
  if ( ( i = hasharray_index(mol->segment_hash,segid) ) != HASHARRAY_FAIL ) {
    sprintf(errmsg,"duplicate segment key %s",segid);
    topo_mol_log_error(mol,errmsg);
    return -3;
  } else {
    i = hasharray_insert(mol->segment_hash,segid);
    if ( i == HASHARRAY_FAIL ) return -4;
    newitem = mol->segment_array[i] = (topo_mol_segment_t*)
		memarena_alloc(mol->arena,sizeof(topo_mol_segment_t));
    if ( ! newitem ) return -5;
  }
  strcpy(newitem->segid,segid);
  newitem->residue_hash = hasharray_create(
	(void**) &(newitem->residue_array), sizeof(topo_mol_residue_t));
  strcpy(newitem->pfirst,"");
  strcpy(newitem->plast,"");
  newitem->auto_angles = mol->defs->auto_angles;
  newitem->auto_dihedrals = mol->defs->auto_dihedrals;
  mol->buildseg = newitem;
  return 0;
}

int topo_mol_segment_first(topo_mol *mol, const char *rname) {
  if ( ! mol ) return -1;
  if ( ! mol->buildseg ) {
    topo_mol_log_error(mol,"no segment in progress for first patch");
    return -1;
  }
  if ( NAMETOOLONG(rname) ) return -2;
  strcpy(mol->buildseg->pfirst,rname);
  return 0;
}

int topo_mol_segment_last(topo_mol *mol, const char *rname) {
  if ( ! mol ) return -1;
  if ( ! mol->buildseg ) {
    topo_mol_log_error(mol,"no segment in progress for last patch");
    return -1;
  }
  if ( NAMETOOLONG(rname) ) return -2;
  strcpy(mol->buildseg->plast,rname);
  return 0;
}

int topo_mol_segment_auto_angles(topo_mol *mol, int autogen) {
  if ( ! mol ) return -1;
  if ( ! mol->buildseg ) {
    topo_mol_log_error(mol,"no segment in progress for auto angles");
    return -1;
  }
  mol->buildseg->auto_angles = autogen;
  return 0;
}

int topo_mol_segment_auto_dihedrals(topo_mol *mol, int autogen) {
  if ( ! mol ) return -1;
  if ( ! mol->buildseg ) {
    topo_mol_log_error(mol,"no segment in progress for auto dihedrals");
    return -1;
  }
  mol->buildseg->auto_dihedrals = autogen;
  return 0;
}

int topo_mol_residue(topo_mol *mol, const char *resid, const char *rname,
						const char *chain) {
  int i;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *newitem;
  char errmsg[32 + NAMEMAXLEN];

  if ( ! mol ) return -1;
  if ( ! mol->buildseg ) {
    topo_mol_log_error(mol,"no segment in progress for residue");
    return -1;
  }
  seg = mol->buildseg;
  if ( NAMETOOLONG(resid) ) return -2;
  if ( NAMETOOLONG(rname) ) return -3;
  if ( hasharray_index(seg->residue_hash,resid) != HASHARRAY_FAIL ) {
    sprintf(errmsg,"duplicate residue key %s",resid);
    topo_mol_log_error(mol,errmsg);
    return -3;
  }

  if ( hasharray_index(mol->defs->residue_hash,rname) == HASHARRAY_FAIL ) {
    sprintf(errmsg,"unknown residue type %s",rname);
    topo_mol_log_error(mol,errmsg);
  }

  i = hasharray_insert(seg->residue_hash,resid);
  if ( i == HASHARRAY_FAIL ) return -4;
  newitem = &(seg->residue_array[i]);
  strcpy(newitem->resid,resid);
  strcpy(newitem->name,rname);
  strcpy(newitem->chain,chain);
  newitem->atoms = 0;

  return 0;
}

int topo_mol_mutate(topo_mol *mol, const char *resid, const char *rname) {
  int ires;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  char errmsg[32 + 3*NAMEMAXLEN];

  if ( ! mol ) return -1;
  if ( ! mol->buildseg ) {
    topo_mol_log_error(mol,"no segment in progress for mutate");
    return -1;
  }
  seg = mol->buildseg;

  if ( NAMETOOLONG(resid) ) return -2;
  if ( NAMETOOLONG(rname) ) return -3;
  ires = hasharray_index(seg->residue_hash,resid);
  if ( ires == HASHARRAY_FAIL ) {
    sprintf(errmsg,"residue %s does not exist",resid);
    topo_mol_log_error(mol,errmsg);
    return -1;
  }
  res = seg->residue_array + ires;
  sprintf(errmsg,"mutating residue %s from %s to %s",resid,res->name,rname);
  topo_mol_log_error(mol,errmsg);

  if ( hasharray_index(mol->defs->residue_hash,rname) == HASHARRAY_FAIL ) {
    sprintf(errmsg,"unknown residue type %s",rname);
    topo_mol_log_error(mol,errmsg);
  }

  strcpy(res->name,rname);

  return 0;
}

static topo_mol_atom_t * topo_mol_unlink_atom(
		topo_mol_atom_t **atoms, const char *aname) {
  topo_mol_atom_t **atom;
  topo_mol_atom_t *oldatom;
  if ( ! atoms ) return 0;
  for ( atom = atoms ; *atom; atom = &((*atom)->next) ) {
    if ( ! strcmp(aname,(*atom)->name) ) break;
  }
  oldatom = *atom;
  if ( *atom ) *atom = ((*atom)->next);
  return oldatom;
}

static topo_mol_atom_t * topo_mol_find_atom(topo_mol_atom_t **newatoms,
		topo_mol_atom_t *oldatoms, const char *aname) {
  topo_mol_atom_t *atom, **newatom;
  if ( ! oldatoms ) return 0;
  for ( atom = oldatoms; atom; atom = atom->next ) {
    if ( ! strcmp(aname,atom->name) ) break;
  }
  if ( atom && *newatoms != oldatoms ) {
    for ( newatom = newatoms; *newatom != oldatoms; newatom = &(*newatom)->next );
    *newatom = atom->next;
    atom->next = *newatoms;
    *newatoms = oldatoms;
  }
  return atom;
}

static int topo_mol_add_atom(topo_mol *mol, topo_mol_atom_t **atoms,
		topo_mol_atom_t *oldatoms, topo_defs_atom_t *atomdef) {
  int idef;
  topo_mol_atom_t *atomtmp;
  topo_defs_type_t *atype;
  char errmsg[128];
  if ( ! mol || ! atoms ) return -1;
  idef = hasharray_index(mol->defs->type_hash,atomdef->type);
  if ( idef == HASHARRAY_FAIL ) {
    sprintf(errmsg,"unknown atom type %s",atomdef->type);
    topo_mol_log_error(mol,errmsg);
    return -3;
  }
  atomtmp = 0;
  if ( oldatoms ) atomtmp = topo_mol_find_atom(atoms,oldatoms,atomdef->name);
  if ( ! atomtmp ) {
    atomtmp = memarena_alloc(mol->arena,sizeof(topo_mol_atom_t));
    if ( ! atomtmp ) return -2;
    strcpy(atomtmp->name,atomdef->name);
    atomtmp->bonds = 0;
    atomtmp->angles = 0;
    atomtmp->dihedrals = 0;
    atomtmp->impropers = 0;
    atomtmp->cmaps = 0;
    atomtmp->conformations = 0;
    atomtmp->x = 0;
    atomtmp->y = 0;
    atomtmp->z = 0;
    atomtmp->xyz_state = TOPO_MOL_XYZ_VOID;
    atomtmp->partition = 0;
    atomtmp->atomid = 0;
    atomtmp->next = *atoms;
    *atoms = atomtmp;
  }
  atomtmp->copy = 0;
  atomtmp->charge = atomdef->charge;
  strcpy(atomtmp->type,atomdef->type);
  atype = &(mol->defs->type_array[idef]);
  strcpy(atomtmp->element,atype->element);
  atomtmp->mass = atype->mass;
  return 0;
}

topo_mol_bond_t * topo_mol_bond_next(
		topo_mol_bond_t *tuple, topo_mol_atom_t *atom) {
  if ( tuple->atom[0] == atom ) return tuple->next[0];
  if ( tuple->atom[1] == atom ) return tuple->next[1];
  return 0;
}

topo_mol_angle_t * topo_mol_angle_next(
		topo_mol_angle_t *tuple, topo_mol_atom_t *atom) {
  if ( tuple->atom[0] == atom ) return tuple->next[0];
  if ( tuple->atom[1] == atom ) return tuple->next[1];
  if ( tuple->atom[2] == atom ) return tuple->next[2];
  return 0;
}

topo_mol_dihedral_t * topo_mol_dihedral_next(
		topo_mol_dihedral_t *tuple, topo_mol_atom_t *atom) {
  if ( tuple->atom[0] == atom ) return tuple->next[0];
  if ( tuple->atom[1] == atom ) return tuple->next[1];
  if ( tuple->atom[2] == atom ) return tuple->next[2];
  if ( tuple->atom[3] == atom ) return tuple->next[3];
  return 0;
}

topo_mol_improper_t * topo_mol_improper_next(
		topo_mol_improper_t *tuple, topo_mol_atom_t *atom) {
  if ( tuple->atom[0] == atom ) return tuple->next[0];
  if ( tuple->atom[1] == atom ) return tuple->next[1];
  if ( tuple->atom[2] == atom ) return tuple->next[2];
  if ( tuple->atom[3] == atom ) return tuple->next[3];
  return 0;
}

topo_mol_cmap_t * topo_mol_cmap_next(
		topo_mol_cmap_t *tuple, topo_mol_atom_t *atom) {
  if ( tuple->atom[0] == atom ) return tuple->next[0];
  if ( tuple->atom[1] == atom ) return tuple->next[1];
  if ( tuple->atom[2] == atom ) return tuple->next[2];
  if ( tuple->atom[3] == atom ) return tuple->next[3];
  if ( tuple->atom[4] == atom ) return tuple->next[4];
  if ( tuple->atom[5] == atom ) return tuple->next[5];
  if ( tuple->atom[6] == atom ) return tuple->next[6];
  if ( tuple->atom[7] == atom ) return tuple->next[7];
  return 0;
}

static topo_mol_conformation_t * topo_mol_conformation_next(
		topo_mol_conformation_t *tuple, topo_mol_atom_t *atom) {
  if ( tuple->atom[0] == atom ) return tuple->next[0];
  if ( tuple->atom[1] == atom ) return tuple->next[1];
  if ( tuple->atom[2] == atom ) return tuple->next[2];
  if ( tuple->atom[3] == atom ) return tuple->next[3];
  return 0;
}

static void topo_mol_destroy_atom(topo_mol_atom_t *atom) {
  topo_mol_bond_t *bondtmp;
  topo_mol_angle_t *angletmp;
  topo_mol_dihedral_t *dihetmp;
  topo_mol_improper_t *imprtmp;
  topo_mol_cmap_t *cmaptmp;
  topo_mol_conformation_t *conftmp;
  if ( ! atom ) return;
  for ( bondtmp = atom->bonds; bondtmp;
		bondtmp = topo_mol_bond_next(bondtmp,atom) ) {
    bondtmp->del = 1;
  }
  for ( angletmp = atom->angles; angletmp;
		angletmp = topo_mol_angle_next(angletmp,atom) ) {
    angletmp->del = 1;
  }
  for ( dihetmp = atom->dihedrals; dihetmp;
		dihetmp = topo_mol_dihedral_next(dihetmp,atom) ) {
    dihetmp->del = 1;
  }
  for ( imprtmp = atom->impropers; imprtmp;
		imprtmp = topo_mol_improper_next(imprtmp,atom) ) {
    imprtmp->del = 1;
  }
  for ( cmaptmp = atom->cmaps; cmaptmp;
		cmaptmp = topo_mol_cmap_next(cmaptmp,atom) ) {
    cmaptmp->del = 1;
  }
  for ( conftmp = atom->conformations; conftmp;
		conftmp = topo_mol_conformation_next(conftmp,atom) ) {
    conftmp->del = 1;
  }
}

static void topo_mol_del_atom(topo_mol_residue_t *res, const char *aname) {
  if ( ! res ) return;
  topo_mol_destroy_atom(topo_mol_unlink_atom(&(res->atoms),aname));
}

/*
 * The add_xxx_to_residues routines exist because topo_mol_end can do
 * more intelligent error checking than what's done in the add_xxx
 * routines.  The add_xxx routines are called by topo_mol_patch, which
 * has to be more general (and more paranoid) about its input.  Returning
 * nonzero from add_xxx_to_residues is always a serious error.
 */
static int add_bond_to_residues(topo_mol *mol, 
    const topo_mol_residue_t *res1, const char *aname1,
    const topo_mol_residue_t *res2, const char *aname2) {
  topo_mol_bond_t *tuple;
  topo_mol_atom_t *a1, *a2;

  a1 = topo_mol_get_atom_from_res(res1, aname1);
  a2 = topo_mol_get_atom_from_res(res2, aname2);
  if (!a1 || !a2) return -1;
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_bond_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->bonds;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->bonds;
  tuple->atom[1] = a2;
  tuple->del = 0;
  a1->bonds = tuple;
  a2->bonds = tuple;
  return 0;
}

static int topo_mol_add_bond(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_bond_t *def) {
  topo_mol_bond_t *tuple;
  topo_mol_atom_t *a1, *a2;
  topo_mol_ident_t t1, t2;
  if (! mol) return -1;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return -2;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return -3;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return -4;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( ! a2 ) return -5;
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_bond_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->bonds;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->bonds;
  tuple->atom[1] = a2;
  tuple->del = 0;
  a1->bonds = tuple;
  a2->bonds = tuple;
  return 0;
}

static void topo_mol_del_bond(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_bond_t *def) {
  topo_mol_bond_t *tuple;
  topo_mol_atom_t *a1, *a2;
  topo_mol_ident_t t1, t2;
  if (! mol) return;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  for ( tuple = a1->bonds; tuple;
		tuple = topo_mol_bond_next(tuple,a1) ) {
    if ( tuple->atom[0] == a1 && tuple->atom[1] == a2 ) tuple->del = 1;
    if ( tuple->atom[0] == a2 && tuple->atom[1] == a1 ) tuple->del = 1;
  }
}


static int topo_mol_add_angle(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_angle_t *def) {
  topo_mol_angle_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3;
  topo_mol_ident_t t1, t2, t3;
  if (! mol) return -1;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return -2;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return -3;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return -4;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( ! a2 ) return -5;
  if ( def->res3 < 0 || def->res3 >= ntargets ) return -6;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  if ( ! a3 ) return -7;
  tuple = memarena_alloc(mol->angle_arena,sizeof(topo_mol_angle_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->angles;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->angles;
  tuple->atom[1] = a2;
  tuple->next[2] = a3->angles;
  tuple->atom[2] = a3;
  tuple->del = 0;
  a1->angles = tuple;
  a2->angles = tuple;
  a3->angles = tuple;
  return 0;
}

static void topo_mol_del_angle(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_angle_t *def) {
  topo_mol_angle_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3;
  topo_mol_ident_t t1, t2, t3;
  if (! mol) return;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( def->res3 < 0 || def->res3 >= ntargets ) return;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  for ( tuple = a1->angles; tuple;
		tuple = topo_mol_angle_next(tuple,a1) ) {
    if ( tuple->atom[0] == a1 && tuple->atom[1] == a2
	&& tuple->atom[2] == a3 ) tuple->del = 1;
    if ( tuple->atom[0] == a3 && tuple->atom[1] == a2
	&& tuple->atom[2] == a1 ) tuple->del = 1;
  }
}


static int topo_mol_add_dihedral(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_dihedral_t *def) {
  topo_mol_dihedral_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;
  topo_mol_ident_t t1, t2, t3, t4;
  if (! mol) return -1;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return -2;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return -3;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return -4;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( ! a2 ) return -5;
  if ( def->res3 < 0 || def->res3 >= ntargets ) return -6;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  if ( ! a3 ) return -7;
  if ( def->res4 < 0 || def->res4 >= ntargets ) return -8;
  t4 = targets[def->res4];
  t4.aname = def->atom4;
  a4 = topo_mol_get_atom(mol,&t4,def->rel4);
  if ( ! a4 ) return -9;
  tuple = memarena_alloc(mol->dihedral_arena,sizeof(topo_mol_dihedral_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->dihedrals;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->dihedrals;
  tuple->atom[1] = a2;
  tuple->next[2] = a3->dihedrals;
  tuple->atom[2] = a3;
  tuple->next[3] = a4->dihedrals;
  tuple->atom[3] = a4;
  tuple->del = 0;
  a1->dihedrals = tuple;
  a2->dihedrals = tuple;
  a3->dihedrals = tuple;
  a4->dihedrals = tuple;
  return 0;
}

static void topo_mol_del_dihedral(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_dihedral_t *def) {
  topo_mol_dihedral_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;
  topo_mol_ident_t t1, t2, t3, t4;
  if (! mol) return;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( def->res3 < 0 || def->res3 >= ntargets ) return;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  if ( def->res4 < 0 || def->res4 >= ntargets ) return;
  t4 = targets[def->res4];
  t4.aname = def->atom4;
  a4 = topo_mol_get_atom(mol,&t4,def->rel4);
  for ( tuple = a1->dihedrals; tuple;
		tuple = topo_mol_dihedral_next(tuple,a1) ) {
    if ( tuple->atom[0] == a1 && tuple->atom[1] == a2
	&& tuple->atom[2] == a3 && tuple->atom[3] == a4 ) tuple->del = 1;
    if ( tuple->atom[0] == a4 && tuple->atom[1] == a3
	&& tuple->atom[2] == a2 && tuple->atom[3] == a1 ) tuple->del = 1;
  }
}

static int add_improper_to_residues(topo_mol *mol, 
    const topo_mol_residue_t *res1, const char *aname1,
    const topo_mol_residue_t *res2, const char *aname2,
    const topo_mol_residue_t *res3, const char *aname3,
    const topo_mol_residue_t *res4, const char *aname4) {
  topo_mol_improper_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;

  a1 = topo_mol_get_atom_from_res(res1, aname1);
  a2 = topo_mol_get_atom_from_res(res2, aname2);
  a3 = topo_mol_get_atom_from_res(res3, aname3);
  a4 = topo_mol_get_atom_from_res(res4, aname4);
  if (!a1 || !a2 || !a3 || !a4) return -1;
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_improper_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->impropers;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->impropers;
  tuple->atom[1] = a2;
  tuple->next[2] = a3->impropers;
  tuple->atom[2] = a3;
  tuple->next[3] = a4->impropers;
  tuple->atom[3] = a4;
  tuple->del = 0;
  a1->impropers = tuple;
  a2->impropers = tuple;
  a3->impropers = tuple;
  a4->impropers = tuple;
  return 0;
}

static int topo_mol_add_improper(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_improper_t *def) {
  topo_mol_improper_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;
  topo_mol_ident_t t1, t2, t3, t4;
  if (! mol) return -1;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return -2;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return -3;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return -4;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( ! a2 ) return -5;
  if ( def->res3 < 0 || def->res3 >= ntargets ) return -6;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  if ( ! a3 ) return -7;
  if ( def->res4 < 0 || def->res4 >= ntargets ) return -8;
  t4 = targets[def->res4];
  t4.aname = def->atom4;
  a4 = topo_mol_get_atom(mol,&t4,def->rel4);
  if ( ! a4 ) return -9;
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_improper_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->impropers;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->impropers;
  tuple->atom[1] = a2;
  tuple->next[2] = a3->impropers;
  tuple->atom[2] = a3;
  tuple->next[3] = a4->impropers;
  tuple->atom[3] = a4;
  tuple->del = 0;
  a1->impropers = tuple;
  a2->impropers = tuple;
  a3->impropers = tuple;
  a4->impropers = tuple;
  return 0;
}

static void topo_mol_del_improper(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_improper_t *def) {
  topo_mol_improper_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;
  topo_mol_ident_t t1, t2, t3, t4;
  if (! mol) return;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( def->res3 < 0 || def->res3 >= ntargets ) return;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  if ( def->res4 < 0 || def->res4 >= ntargets ) return;
  t4 = targets[def->res4];
  t4.aname = def->atom4;
  a4 = topo_mol_get_atom(mol,&t4,def->rel4);
  for ( tuple = a1->impropers; tuple;
		tuple = topo_mol_improper_next(tuple,a1) ) {
    if ( tuple->atom[0] == a1 && tuple->atom[1] == a2
	&& tuple->atom[2] == a3 && tuple->atom[3] == a4 ) tuple->del = 1;
    if ( tuple->atom[0] == a4 && tuple->atom[1] == a3
	&& tuple->atom[2] == a2 && tuple->atom[3] == a1 ) tuple->del = 1;
  }
}

static int add_cmap_to_residues(topo_mol *mol, 
    const topo_mol_residue_t *resl[8], const char *anamel[8]) {
  int i;
  topo_mol_cmap_t *tuple;
  topo_mol_atom_t *al[8];

  if (! mol) return -1;
  for ( i=0; i<8; ++i ) {
    al[i] = topo_mol_get_atom_from_res(resl[i], anamel[i]);
    if (!al[i]) return -2-2*i;
  }
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_cmap_t));
  if ( ! tuple ) return -20;
  for ( i=0; i<8; ++i ) {
    tuple->next[i] = al[i]->cmaps;
    tuple->atom[i] = al[i];
  }
  for ( i=0; i<8; ++i ) {
    /* This must be in a separate loop because atoms may be repeated. */
    al[i]->cmaps = tuple;
  }
  tuple->del = 0;
  return 0;
}

static int topo_mol_add_cmap(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_cmap_t *def) {
  int i;
  topo_mol_cmap_t *tuple;
  topo_mol_atom_t *al[8];
  topo_mol_ident_t tl[8];
  if (! mol) return -1;
  for ( i=0; i<8; ++i ) {
    if ( def->resl[i] < 0 || def->resl[i] >= ntargets ) return -2-2*i;
    tl[i] = targets[def->resl[i]];
    tl[i].aname = def->atoml[i];
    al[i] = topo_mol_get_atom(mol,&tl[i],def->rell[i]);
    if ( ! al[i] ) return -3-2*i;
  }
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_cmap_t));
  if ( ! tuple ) return -20;
  for ( i=0; i<8; ++i ) {
    tuple->next[i] = al[i]->cmaps;
    tuple->atom[i] = al[i];
  }
  for ( i=0; i<8; ++i ) {
    /* This must be in a separate loop because atoms may be repeated. */
    al[i]->cmaps = tuple;
  }
  tuple->del = 0;
  return 0;
}

static void topo_mol_del_cmap(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_cmap_t *def) {
  int i;
  topo_mol_cmap_t *tuple;
  topo_mol_atom_t *al[8];
  topo_mol_ident_t tl[8];
  if (! mol) return;
  for ( i=0; i<8; ++i ) {
    if ( def->resl[i] < 0 || def->resl[i] >= ntargets ) return;
    tl[i] = targets[def->resl[i]];
    tl[i].aname = def->atoml[i];
    al[i] = topo_mol_get_atom(mol,&tl[i],def->rell[i]);
    if ( ! al[i] ) return;
  }
  for ( tuple = al[i]->cmaps; tuple;
		tuple = topo_mol_cmap_next(tuple,al[i]) ) {
    int match1, match2;
    match1 = 0;
    for ( i=0; i<4 && (tuple->atom[i] == al[i]); ++i );
    if ( i == 4 ) match1 = 1;
    for ( i=0; i<4 && (tuple->atom[i] == al[4-i]); ++i );
    if ( i == 4 ) match1 = 1;
    match2 = 0;
    for ( i=0; i<4 && (tuple->atom[4+i] == al[4+i]); ++i );
    if ( i == 4 ) match2 = 1;
    for ( i=0; i<4 && (tuple->atom[4+i] == al[8-i]); ++i );
    if ( i == 4 ) match2 = 1;
    if ( match1 && match2 ) tuple->del = 1;
  }
}


static int add_conformation_to_residues(topo_mol *mol, 
    const topo_mol_residue_t *res1, const char *aname1,
    const topo_mol_residue_t *res2, const char *aname2,
    const topo_mol_residue_t *res3, const char *aname3,
    const topo_mol_residue_t *res4, const char *aname4, 
    topo_defs_conformation_t *def) {

  topo_mol_conformation_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;
  a1 = topo_mol_get_atom_from_res(res1, aname1);
  a2 = topo_mol_get_atom_from_res(res2, aname2);
  a3 = topo_mol_get_atom_from_res(res3, aname3);
  a4 = topo_mol_get_atom_from_res(res4, aname4);
  if (!a1 || !a2 || !a3 || !a4) return -1;
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_conformation_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->conformations;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->conformations;
  tuple->atom[1] = a2;
  tuple->next[2] = a3->conformations;
  tuple->atom[2] = a3;
  tuple->next[3] = a4->conformations;
  tuple->atom[3] = a4;
  tuple->del = 0;
  tuple->improper = def->improper;
  tuple->dist12 = def->dist12;
  tuple->angle123 = def->angle123;
  tuple->dihedral = def->dihedral;
  tuple->angle234 = def->angle234;
  tuple->dist34 = def->dist34;
  a1->conformations = tuple;
  a2->conformations = tuple;
  a3->conformations = tuple;
  a4->conformations = tuple;
  return 0;
}

static int topo_mol_add_conformation(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_conformation_t *def) {
  topo_mol_conformation_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;
  topo_mol_ident_t t1, t2, t3, t4;
  if (! mol) return -1;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return -2;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return -3;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return -4;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( ! a2 ) return -5;
  if ( def->res3 < 0 || def->res3 >= ntargets ) return -6;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  if ( ! a3 ) return -7;
  if ( def->res4 < 0 || def->res4 >= ntargets ) return -8;
  t4 = targets[def->res4];
  t4.aname = def->atom4;
  a4 = topo_mol_get_atom(mol,&t4,def->rel4);
  if ( ! a4 ) return -9;
  tuple = memarena_alloc(mol->arena,sizeof(topo_mol_conformation_t));
  if ( ! tuple ) return -10;
  tuple->next[0] = a1->conformations;
  tuple->atom[0] = a1;
  tuple->next[1] = a2->conformations;
  tuple->atom[1] = a2;
  tuple->next[2] = a3->conformations;
  tuple->atom[2] = a3;
  tuple->next[3] = a4->conformations;
  tuple->atom[3] = a4;
  tuple->del = 0;
  tuple->improper = def->improper;
  tuple->dist12 = def->dist12;
  tuple->angle123 = def->angle123;
  tuple->dihedral = def->dihedral;
  tuple->angle234 = def->angle234;
  tuple->dist34 = def->dist34;
  a1->conformations = tuple;
  a2->conformations = tuple;
  a3->conformations = tuple;
  a4->conformations = tuple;
  return 0;
}

static void topo_mol_del_conformation(topo_mol *mol, const topo_mol_ident_t *targets,
				int ntargets, topo_defs_conformation_t *def) {
  topo_mol_conformation_t *tuple;
  topo_mol_atom_t *a1, *a2, *a3, *a4;
  topo_mol_ident_t t1, t2, t3, t4;
  if (! mol) return;
  if ( def->res1 < 0 || def->res1 >= ntargets ) return;
  t1 = targets[def->res1];
  t1.aname = def->atom1;
  a1 = topo_mol_get_atom(mol,&t1,def->rel1);
  if ( ! a1 ) return;
  if ( def->res2 < 0 || def->res2 >= ntargets ) return;
  t2 = targets[def->res2];
  t2.aname = def->atom2;
  a2 = topo_mol_get_atom(mol,&t2,def->rel2);
  if ( def->res3 < 0 || def->res3 >= ntargets ) return;
  t3 = targets[def->res3];
  t3.aname = def->atom3;
  a3 = topo_mol_get_atom(mol,&t3,def->rel3);
  if ( def->res4 < 0 || def->res4 >= ntargets ) return;
  t4 = targets[def->res4];
  t4.aname = def->atom4;
  a4 = topo_mol_get_atom(mol,&t4,def->rel4);
  for ( tuple = a1->conformations; tuple;
		tuple = topo_mol_conformation_next(tuple,a1) ) {
    if ( tuple->improper == def->improper
	&&  tuple->atom[0] == a1 && tuple->atom[1] == a2
	&& tuple->atom[2] == a3 && tuple->atom[3] == a4 ) tuple->del = 1;
    if ( tuple->improper == def->improper
	&& tuple->atom[0] == a4 && tuple->atom[1] == a3
	&& tuple->atom[2] == a2 && tuple->atom[3] == a1 ) tuple->del = 1;
  }
}

static int topo_mol_auto_angles(topo_mol *mol, topo_mol_segment_t *segp);
static int topo_mol_auto_dihedrals(topo_mol *mol, topo_mol_segment_t *segp);

int topo_mol_end(topo_mol *mol) {
  int i,n;
  int idef;
  topo_defs *defs;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_defs_residue_t *resdef;
  topo_defs_atom_t *atomdef;
  topo_defs_bond_t *bonddef;
  topo_defs_angle_t *angldef;
  topo_defs_dihedral_t *dihedef;
  topo_defs_improper_t *imprdef;
  topo_defs_cmap_t *cmapdef;
  topo_defs_conformation_t *confdef;
  topo_mol_ident_t target;
  char errmsg[128];
  int firstdefault=0, lastdefault=0;

  if ( ! mol ) return -1;
  if ( ! mol->buildseg ) {
    topo_mol_log_error(mol,"no segment in progress for end");
    return -1;
  }
  seg = mol->buildseg;
  mol->buildseg = 0;
  defs = mol->defs;

  /* add atoms */
  n = hasharray_count(seg->residue_hash);
  for ( i=0; i<n; ++i ) {
    res = &(seg->residue_array[i]);
    idef = hasharray_index(defs->residue_hash,res->name);
    if ( idef == HASHARRAY_FAIL ) {
      sprintf(errmsg,"unknown residue type %s",res->name);
      topo_mol_log_error(mol,errmsg);
      return -1;
    }
    resdef = &(mol->defs->residue_array[idef]);
    if ( resdef->patch ) {
      sprintf(errmsg,"unknown residue type %s",res->name);
      topo_mol_log_error(mol,errmsg);
      return -1;
    }

    /* patches */
    if ( i==0 && ! strlen(seg->pfirst) ) {
      strcpy(seg->pfirst,resdef->pfirst);
      firstdefault = 1;
    }
    if ( i==(n-1) && ! strlen(seg->plast) ) {
      strcpy(seg->plast,resdef->plast);
      lastdefault = 1;
    }

    for ( atomdef = resdef->atoms; atomdef; atomdef = atomdef->next ) {
      if ( topo_mol_add_atom(mol,&(res->atoms),0,atomdef) ) { 
        sprintf(errmsg,"add atom failed in residue %s:%s",res->name,res->resid);
        topo_mol_log_error(mol,errmsg);
        return -8;
      }
    }
  }

  for ( i=0; i<n; ++i ) {
    res = &(seg->residue_array[i]);
    idef = hasharray_index(defs->residue_hash,res->name);
    if ( idef == HASHARRAY_FAIL ) {
      sprintf(errmsg,"unknown residue type %s",res->name);
      topo_mol_log_error(mol,errmsg);
      return -1;
    }
    resdef = &(mol->defs->residue_array[idef]);
    target.segid = seg->segid;
    target.resid = res->resid;
    for ( bonddef = resdef->bonds; bonddef; bonddef = bonddef->next ) {
      int ires1, ires2;
      if (bonddef->res1 != 0 || bonddef->res2 != 0) {
        /* 
         * XXX This should be caught much earlier, like when the topology
         * file is initially read in. 
         */
        sprintf(errmsg, "ERROR: Bad bond definition %s %s-%s; skipping.",
            res->name, bonddef->atom1, bonddef->atom2);
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      ires1=bonddef->rel1+i;
      ires2=bonddef->rel2+i;
      if (ires1 < 0 || ires2 < 0 || ires1 >= n || ires2 >= n) {
        sprintf(errmsg, "Info: skipping bond %s-%s at %s of segment.", 
            bonddef->atom1, bonddef->atom2, i==0 ? "beginning" : "end");
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      if (add_bond_to_residues(mol, 
            &(seg->residue_array[ires1]), bonddef->atom1,
            &(seg->residue_array[ires2]), bonddef->atom2)) {
        sprintf(errmsg, 
            "ERROR: Missing atoms for bond %s(%d) %s(%d) in residue %s:%s",
            bonddef->atom1,bonddef->rel1,bonddef->atom2,bonddef->rel2,
            res->name,res->resid);
        topo_mol_log_error(mol, errmsg);
      }
    }
    if ( seg->auto_angles && resdef->angles ) {
      sprintf(errmsg,"Warning: explicit angles in residue %s:%s will be deleted during autogeneration",res->name,res->resid);
      topo_mol_log_error(mol,errmsg);
    }
    for ( angldef = resdef->angles; angldef; angldef = angldef->next ) {
      if ( topo_mol_add_angle(mol,&target,1,angldef) ) {
        sprintf(errmsg,"Warning: add angle failed in residue %s:%s",res->name,res->resid);
        topo_mol_log_error(mol,errmsg);
      }
    }
    if ( seg->auto_dihedrals && resdef->dihedrals) {
      sprintf(errmsg,"Warning: explicit dihedrals in residue %s:%s will be deleted during autogeneration",res->name,res->resid);
      topo_mol_log_error(mol,errmsg);
    }
    for ( dihedef = resdef->dihedrals; dihedef; dihedef = dihedef->next ) {
      if ( topo_mol_add_dihedral(mol,&target,1,dihedef) ) {
        sprintf(errmsg,"Warning: add dihedral failed in residue %s:%s",res->name,res->resid);
        topo_mol_log_error(mol,errmsg);
      }
    }
    for ( imprdef = resdef->impropers; imprdef; imprdef = imprdef->next ) {
      int ires1, ires2, ires3, ires4;
      if (imprdef->res1 != 0 || imprdef->res2 != 0 || imprdef->res3 != 0 ||
          imprdef->res4 != 0) {
        sprintf(errmsg, "ERROR: Bad improper definition %s %s-%s-%s-%s; skipping.",
            res->name, imprdef->atom1, imprdef->atom2, imprdef->atom3, 
            imprdef->atom4);
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      ires1=imprdef->rel1+i;
      ires2=imprdef->rel2+i;
      ires3=imprdef->rel3+i;
      ires4=imprdef->rel4+i;
      if (ires1 < 0 || ires2 < 0 || ires3 < 0 || ires4 < 0 ||
          ires1 >= n || ires2 >= n || ires3 >= n || ires4 >= n) {
        sprintf(errmsg,"Info: skipping improper %s-%s-%s-%s at %s of segment.", 
            imprdef->atom1, imprdef->atom2, imprdef->atom3, imprdef->atom4,
            i==0 ? "beginning" : "end");
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      if (add_improper_to_residues(mol, 
            &(seg->residue_array[ires1]), imprdef->atom1,
            &(seg->residue_array[ires2]), imprdef->atom2,
            &(seg->residue_array[ires3]), imprdef->atom3,
            &(seg->residue_array[ires4]), imprdef->atom4)) {
        sprintf(errmsg, 
            "ERROR: Missing atoms for improper %s(%d) %s(%d) %s(%d) %s(%d)\n\tin residue %s:%s",
            imprdef->atom1,imprdef->rel1,imprdef->atom2,imprdef->rel2,
            imprdef->atom3,imprdef->rel3,imprdef->atom4,imprdef->rel4,
            res->name,res->resid);
        topo_mol_log_error(mol, errmsg);
      }
    }
    for ( cmapdef = resdef->cmaps; cmapdef; cmapdef = cmapdef->next ) {
      int j, iresl[8];
      const topo_mol_residue_t *resl[8];
      const char *atoml[8];
      for ( j=0; j<8 && (cmapdef->resl[j] == 0); ++j );
      if ( j != 8 ) {
        sprintf(errmsg, "ERROR: Bad cross-term definition %s %s-%s-%s-%s-%s-%s-%s-%s; skipping.",
            res->name, cmapdef->atoml[0], cmapdef->atoml[1],
            cmapdef->atoml[2], cmapdef->atoml[3], cmapdef->atoml[4],
            cmapdef->atoml[5], cmapdef->atoml[6], cmapdef->atoml[7]);
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      for ( j=0; j<8; ++j ) {
        iresl[j] = cmapdef->rell[j]+i;
      }
      for ( j=0; j<8 && (iresl[j] >= 0) && (iresl[j] < n); ++j );
      if ( j != 8 ) {
        sprintf(errmsg,"Info: skipping cross-term %s-%s-%s-%s-%s-%s-%s-%s at %s of segment.", 
            cmapdef->atoml[0], cmapdef->atoml[1],
            cmapdef->atoml[2], cmapdef->atoml[3], cmapdef->atoml[4],
            cmapdef->atoml[5], cmapdef->atoml[6], cmapdef->atoml[7],
            i==0 ? "beginning" : "end");
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      for ( j=0; j<8; ++j ) {
        resl[j] = &seg->residue_array[iresl[j]];
        atoml[j] = cmapdef->atoml[j];
      }
      if (add_cmap_to_residues(mol, resl, atoml) ) {
        sprintf(errmsg, 
            "ERROR: Missing atoms for cross-term  %s(%d) %s(%d) %s(%d) %s(%d) %s(%d) %s(%d) %s(%d) %s(%d)\n\tin residue %s:%s",
            cmapdef->atoml[0],cmapdef->rell[0],
            cmapdef->atoml[1],cmapdef->rell[1],
            cmapdef->atoml[2],cmapdef->rell[2],
            cmapdef->atoml[3],cmapdef->rell[3],
            cmapdef->atoml[4],cmapdef->rell[4],
            cmapdef->atoml[5],cmapdef->rell[5],
            cmapdef->atoml[6],cmapdef->rell[6],
            cmapdef->atoml[7],cmapdef->rell[7],
            res->name,res->resid);
        topo_mol_log_error(mol, errmsg);
      }
    }
    for ( confdef = resdef->conformations; confdef; confdef = confdef->next ) {
      int ires1, ires2, ires3, ires4;
      if (confdef->res1 != 0 || confdef->res2 != 0 || confdef->res3 != 0 ||
          confdef->res4 != 0) {
        sprintf(errmsg, "ERROR: Bad conformation definition %s %s-%s-%s-%s; skipping.",
            res->name, confdef->atom1, confdef->atom2, confdef->atom3, 
            confdef->atom4);
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      ires1=confdef->rel1+i;
      ires2=confdef->rel2+i;
      ires3=confdef->rel3+i;
      ires4=confdef->rel4+i;
      if (ires1 < 0 || ires2 < 0 || ires3 < 0 || ires4 < 0 ||
          ires1 >= n || ires2 >= n || ires3 >= n || ires4 >= n) {
        sprintf(errmsg,"Info: skipping conformation %s-%s-%s-%s at %s of segment.", 
            confdef->atom1, confdef->atom2, confdef->atom3, confdef->atom4,
            i==0 ? "beginning" : "end");
        topo_mol_log_error(mol, errmsg);
        continue;
      }
      if (add_conformation_to_residues(mol, 
            &(seg->residue_array[ires1]), confdef->atom1,
            &(seg->residue_array[ires2]), confdef->atom2,
            &(seg->residue_array[ires3]), confdef->atom3,
            &(seg->residue_array[ires4]), confdef->atom4, confdef)) {
        sprintf(errmsg, "Warning: missing atoms for conformation %s %s-%s-%s-%s; skipping.",
            res->name, confdef->atom1, confdef->atom2, confdef->atom3, 
            confdef->atom4);
        topo_mol_log_error(mol, errmsg);
      }
    }
  }

  /* apply patches, last then first because dipeptide patch ACED depends on CT3 atom NT */

  res = &(seg->residue_array[n-1]);
  if ( ! strlen(seg->plast) ) strcpy(seg->plast,"NONE");

  target.segid = seg->segid;
  target.resid = res->resid;
  if ( topo_mol_patch(mol, &target, 1, seg->plast, 0,
	seg->auto_angles, seg->auto_dihedrals, lastdefault) ) return -10;

  res = &(seg->residue_array[0]);
  if ( ! strlen(seg->pfirst) ) strcpy(seg->pfirst,"NONE");

  target.segid = seg->segid;
  target.resid = res->resid;
  if ( topo_mol_patch(mol, &target, 1, seg->pfirst, 1,
	seg->auto_angles, seg->auto_dihedrals, firstdefault) ) return -11;

  if (seg->auto_angles && topo_mol_auto_angles(mol, seg)) return -12;
  if (seg->auto_dihedrals && topo_mol_auto_dihedrals(mol, seg)) return -13;

  return 0;
}

int topo_mol_regenerate_resids(topo_mol *mol) {
  int ires, nres, iseg, nseg, npres;
  int prevresid, resid, npatchresptrs, ipatch;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_mol_patch_t **patchptr, *patch;
  topo_mol_patchres_t *patchres, **patchresptrs;
  char newresid[NAMEMAXLEN+20], (*newpatchresids)[NAMEMAXLEN];

  if (! mol) return -1;

  nseg = hasharray_count(mol->segment_hash);
  npatchresptrs=0;

  /* clean patches so only valid items remain */
  for ( patchptr = &(mol->patches); *patchptr; ) {
    npres=0;
    for ( patchres = (*patchptr)->patchresids; patchres; patchres = patchres->next ) {
      ++npres;
      /* Test the existence of segid:resid for the patch */
      if (!topo_mol_validate_patchres(mol,patch->pname,patchres->segid, patchres->resid)) {
        break;
      }
    }
    if ( patchres ) {  /* remove patch from list */
      *patchptr = (*patchptr)->next;
      continue;
    }
    npatchresptrs += npres;
    patchptr = &((*patchptr)->next);  /* continue to next patch */
  }

  patchresptrs = malloc(npatchresptrs * sizeof(topo_mol_patchres_t*));
  if ( ! patchresptrs ) return -5;
  newpatchresids = calloc(npatchresptrs, NAMEMAXLEN);
  if ( ! newpatchresids ) return -6;

  for ( ipatch=0, patch = mol->patches; patch; patch = patch->next ) {
    for ( patchres = patch->patchresids; patchres; patchres = patchres->next ) {
      patchresptrs[ipatch++] = patchres;
    }
  }

  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if ( ! seg ) continue;
    nres = hasharray_count(seg->residue_hash);
    if ( hasharray_clear(seg->residue_hash) == HASHARRAY_FAIL ) return -2;

    prevresid = -100000;
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      resid = atoi(res->resid);
      if ( resid <= prevresid ) resid = prevresid + 1;
      sprintf(newresid, "%d", resid);
      if ( NAMETOOLONG(newresid) ) return -3;
      if ( strcmp(res->resid, newresid) ) { /* changed, need to check patches */
        for ( ipatch=0; ipatch < npatchresptrs; ++ipatch ) {
          if ( ( ! strcmp(seg->segid, patchresptrs[ipatch]->segid) ) &&
               ( ! strcmp(res->resid, patchresptrs[ipatch]->resid) ) ) {
            sprintf(newpatchresids[ipatch], "%d", resid);
          }
        }
      }
      sprintf(res->resid, "%d", resid);
      if ( hasharray_reinsert(seg->residue_hash,res->resid,ires) != ires ) return -4;
      prevresid = resid;
    }
  }

  for ( ipatch=0; ipatch < npatchresptrs; ++ipatch ) {
    if ( newpatchresids[ipatch][0] ) {
      strcpy(patchresptrs[ipatch]->resid,newpatchresids[ipatch]);
    }
  }

  free(patchresptrs);
  free(newpatchresids);
  return 0;
}

int topo_mol_regenerate_angles(topo_mol *mol) {
  int errval;
  if ( mol ) {
    memarena_destroy(mol->angle_arena);
    mol->angle_arena = memarena_create();
  }
  errval = topo_mol_auto_angles(mol,0);
  if ( errval ) {
    char errmsg[128];
    sprintf(errmsg,"Error code %d",errval);
    topo_mol_log_error(mol,errmsg);
  }
  return errval;
}

int topo_mol_regenerate_dihedrals(topo_mol *mol) {
  int errval;
  if ( mol ) {
    memarena_destroy(mol->dihedral_arena);
    mol->dihedral_arena = memarena_create();
  }
  errval = topo_mol_auto_dihedrals(mol,0);
  if ( errval ) {
    char errmsg[128];
    sprintf(errmsg,"Error code %d",errval);
    topo_mol_log_error(mol,errmsg);
  }
  return errval;
}

static int is_hydrogen(topo_mol_atom_t *atom) {
  return ( atom->mass < 3.5 && atom->name[0] == 'H' );
}

static int is_oxygen(topo_mol_atom_t *atom) {
  return ( atom->mass > 14.5 && atom->mass < 18.5 && atom->name[0] == 'O' );
}

static int topo_mol_auto_angles(topo_mol *mol, topo_mol_segment_t *segp) {
  int ires, nres, iseg, nseg;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_mol_bond_t *b1, *b2;
  topo_mol_angle_t *tuple;
  topo_mol_atom_t *atom, *a1, *a2, *a3;

  if (! mol) return -1;
  nseg = segp ? 1 : hasharray_count(mol->segment_hash);

  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = segp ? segp : mol->segment_array[iseg];
    if ( ! seg ) continue;

    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        if ( ! segp ) { atom->angles = NULL; }
        for ( tuple = atom->angles; tuple;
		tuple = topo_mol_angle_next(tuple,atom) ) {
          tuple->del = 1;
        }
      }
    }
  }

  for ( iseg=0; iseg<nseg; ++iseg ) {
  seg = segp ? segp : mol->segment_array[iseg];
  if ( ! seg ) continue;

  nres = hasharray_count(seg->residue_hash);
  for ( ires=0; ires<nres; ++ires ) {
    res = &(seg->residue_array[ires]);
    for ( atom = res->atoms; atom; atom = atom->next ) {
      a2 = atom;
      for ( b1 = atom->bonds; b1; b1 = topo_mol_bond_next(b1,atom) ) {
        if ( b1->del ) continue;
        if ( b1->atom[0] == atom ) a1 = b1->atom[1];
        else if ( b1->atom[1] == atom ) a1 = b1->atom[0];
        else return -5;
        b2 = b1;  while ( (b2 = topo_mol_bond_next(b2,atom)) ) {
          if ( b2->del ) continue;
          if ( b2->atom[0] == atom ) a3 = b2->atom[1];
          else if ( b2->atom[1] == atom ) a3 = b2->atom[0];
          else return -6;
          if ( is_hydrogen(a2) && ( ! topo_mol_bond_next(b2,atom) ) &&
               ( ( is_hydrogen(a1) && is_oxygen(a3) ) ||
                 ( is_hydrogen(a3) && is_oxygen(a1) ) ) )
            continue;  /* extra H-H bond on water */
          tuple = memarena_alloc(mol->angle_arena,sizeof(topo_mol_angle_t));
          if ( ! tuple ) return -10;
          tuple->next[0] = a1->angles;
          tuple->atom[0] = a1;
          tuple->next[1] = a2->angles;
          tuple->atom[1] = a2;
          tuple->next[2] = a3->angles;
          tuple->atom[2] = a3;
          tuple->del = 0;
          a1->angles = tuple;
          a2->angles = tuple;
          a3->angles = tuple;
        }
      }
    }
  }
  }

  return 0;
}

static int topo_mol_auto_dihedrals(topo_mol *mol, topo_mol_segment_t *segp) {
  int ires, nres, iseg, nseg, found, atomid, count1, count2;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_mol_angle_t *g1, *g2;
  topo_mol_dihedral_t *tuple;
  topo_mol_atom_t *atom, *a1=0, *a2=0, *a3=0, *a4=0;

  if (! mol) return -1;
  nseg = segp ? 1 : hasharray_count(mol->segment_hash);

  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = segp ? segp : mol->segment_array[iseg];
    if ( ! seg ) continue;

    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        if ( ! segp ) { atom->dihedrals = NULL; }
        for ( tuple = atom->dihedrals; tuple;
		tuple = topo_mol_dihedral_next(tuple,atom) ) {
          tuple->del = 1;
        }
      }
    }
  }

  /*  number atoms, needed to avoid duplicate dihedrals below  */
  /*  assumes no inter-segment bonds if segp is non-null  */
  atomid = 0;
  for ( iseg=0; iseg<nseg; ++iseg ) {
  seg = segp ? segp : mol->segment_array[iseg];
  if ( ! seg ) continue;

  nres = hasharray_count(seg->residue_hash);
  for ( ires=0; ires<nres; ++ires ) {
    res = &(seg->residue_array[ires]);
    for ( atom = res->atoms; atom; atom = atom->next ) {
      atom->atomid = ++atomid;
    }
  }
  }

  count1 = count2 = 0;

  for ( iseg=0; iseg<nseg; ++iseg ) {
  seg = segp ? segp : mol->segment_array[iseg];
  if ( ! seg ) continue;

  nres = hasharray_count(seg->residue_hash);
  for ( ires=0; ires<nres; ++ires ) {
    res = &(seg->residue_array[ires]);
    for ( atom = res->atoms; atom; atom = atom->next ) {
      for ( g1 = atom->angles; g1; g1 = topo_mol_angle_next(g1,atom) ) {
        if ( g1->del ) continue;
        if ( g1->atom[1] != atom ) continue;
        for ( g2 = atom->angles; g2; g2 = topo_mol_angle_next(g2,atom) ) {
          if ( g2->del ) continue;
          if ( g2->atom[1] == atom ) continue;
          found = 0;
          if ( g2->atom[0] == atom ) {  /*  XBX BXX  */
            if ( g2->atom[1] == g1->atom[0] ) {  /*  CBA BCD  */
              a1 = g1->atom[2];
              a2 = g1->atom[1];  /* == g2->atom[0] */
              a3 = g1->atom[0];  /* == g2->atom[1] */
              a4 = g2->atom[2];
              found = ( a1->atomid < a4->atomid );
              if ( a1 != a4 ) ++count2;
            } else if ( g2->atom[1] == g1->atom[2] ) {  /*  ABC BCD  */
              a1 = g1->atom[0];
              a2 = g1->atom[1];  /* == g2->atom[0] */
              a3 = g1->atom[2];  /* == g2->atom[1] */
              a4 = g2->atom[2];
              found = ( a1->atomid < a4->atomid );
              if ( a1 != a4 ) ++count2;
            }
          } else if ( g2->atom[2] == atom ) {  /*  XBX XXB  */
            if ( g2->atom[1] == g1->atom[0] ) {  /*  CBA DCB  */
              a1 = g1->atom[2];
              a2 = g1->atom[1];  /* == g2->atom[2] */
              a3 = g1->atom[0];  /* == g2->atom[1] */
              a4 = g2->atom[0];
              found = ( a1->atomid < a4->atomid );
              if ( a1 != a4 ) ++count2;
            } else if ( g2->atom[1] == g1->atom[2] ) {  /*  ABC DCB  */
              a1 = g1->atom[0];
              a2 = g1->atom[1];  /* == g2->atom[2] */
              a3 = g1->atom[2];  /* == g2->atom[1] */
              a4 = g2->atom[0];
              found = ( a1->atomid < a4->atomid );
              if ( a1 != a4 ) ++count2;
            }
          } else return -6;
          if ( ! found ) continue;
          ++count1;
          tuple = memarena_alloc(mol->dihedral_arena,sizeof(topo_mol_dihedral_t));
          if ( ! tuple ) return -10;
          tuple->next[0] = a1->dihedrals;
          tuple->atom[0] = a1;
          tuple->next[1] = a2->dihedrals;
          tuple->atom[1] = a2;
          tuple->next[2] = a3->dihedrals;
          tuple->atom[2] = a3;
          tuple->next[3] = a4->dihedrals;
          tuple->atom[3] = a4;
          tuple->del = 0;
          a1->dihedrals = tuple;
          a2->dihedrals = tuple;
          a3->dihedrals = tuple;
          a4->dihedrals = tuple;
        }
      }
    }
  }
  }

  if ( count2 != 2 * count1 ) return -15;  /* missing dihedrals */

  return 0;
}

int topo_mol_patch(topo_mol *mol, const topo_mol_ident_t *targets,
                        int ntargets, const char *rname, int prepend,
			int warn_angles, int warn_dihedrals, int deflt) {

  int idef;
  topo_defs_residue_t *resdef;
  topo_defs_atom_t *atomdef;
  topo_defs_bond_t *bonddef;
  topo_defs_angle_t *angldef;
  topo_defs_dihedral_t *dihedef;
  topo_defs_improper_t *imprdef;
  topo_defs_cmap_t *cmapdef;
  topo_defs_conformation_t *confdef;
  topo_mol_residue_t *res, *oldres;
  topo_mol_atom_t *oldatoms;
  char errmsg[128];

  if ( ! mol ) return -1;
  if ( mol->buildseg ) return -2;
  if ( ! mol->defs ) return -3;

  idef = hasharray_index(mol->defs->residue_hash,rname);
  if ( idef == HASHARRAY_FAIL ) {
    sprintf(errmsg,"unknown patch type %s",rname);
    topo_mol_log_error(mol,errmsg);
    return -4;
  }
  resdef = &(mol->defs->residue_array[idef]);
  if ( ! resdef->patch ) {
    sprintf(errmsg,"unknown patch type %s",rname);
    topo_mol_log_error(mol,errmsg);
    return -5;
  }

  oldres = 0;
  for ( atomdef = resdef->atoms; atomdef; atomdef = atomdef->next ) {
    if ( atomdef->res < 0 || atomdef->res >= ntargets ) return -6;
    res = topo_mol_get_res(mol,&targets[atomdef->res],atomdef->rel);
    if ( ! res ) return -7;
    if ( atomdef->del ) {
      topo_mol_del_atom(res,atomdef->name);
      oldres = 0;
      continue;
    }
    if ( res != oldres ) {
      oldres = res;
      oldatoms = res->atoms;
    }
    if ( topo_mol_add_atom(mol,&(res->atoms),oldatoms,atomdef) ) {
      sprintf(errmsg,"add atom failed in patch %s",rname);
      topo_mol_log_error(mol,errmsg);
      return -8;
    }
  }

  for ( bonddef = resdef->bonds; bonddef; bonddef = bonddef->next ) {
    if ( bonddef->del ) topo_mol_del_bond(mol,targets,ntargets,bonddef);
    else if ( topo_mol_add_bond(mol,targets,ntargets,bonddef) ) {
      sprintf(errmsg,"Warning: add bond failed in patch %s",rname);
      topo_mol_log_error(mol,errmsg);
    }
  }
  if ( warn_angles && resdef->angles ) {
    sprintf(errmsg,"Warning: explicit angles in patch %s will be deleted during autogeneration",rname);
    topo_mol_log_error(mol,errmsg);
  }
  for ( angldef = resdef->angles; angldef; angldef = angldef->next ) {
    if ( angldef->del ) topo_mol_del_angle(mol,targets,ntargets,angldef);
    else if ( topo_mol_add_angle(mol,targets,ntargets,angldef) ) {
      sprintf(errmsg,"Warning: add angle failed in patch %s",rname);
      topo_mol_log_error(mol,errmsg);
    }
  }
  if ( warn_dihedrals && resdef->dihedrals ) {
    sprintf(errmsg,"Warning: explicit dihedrals in patch %s will be deleted during autogeneration",rname);
    topo_mol_log_error(mol,errmsg);
  }
  for ( dihedef = resdef->dihedrals; dihedef; dihedef = dihedef->next ) {
    if ( dihedef->del ) topo_mol_del_dihedral(mol,targets,ntargets,dihedef);
    else if ( topo_mol_add_dihedral(mol,targets,ntargets,dihedef) ) {
      sprintf(errmsg,"Warning: add dihedral failed in patch %s",rname);
        topo_mol_log_error(mol,errmsg);
      }
  }
  for ( imprdef = resdef->impropers; imprdef; imprdef = imprdef->next ) {
    if ( imprdef->del ) topo_mol_del_improper(mol,targets,ntargets,imprdef);
    else if ( topo_mol_add_improper(mol,targets,ntargets,imprdef) ) {
      sprintf(errmsg,"Warning: add improper failed in patch %s",rname);
      topo_mol_log_error(mol,errmsg);
    }
  }
  for ( cmapdef = resdef->cmaps; cmapdef; cmapdef = cmapdef->next ) {
    if ( cmapdef->del ) topo_mol_del_cmap(mol,targets,ntargets,cmapdef);
    else if ( topo_mol_add_cmap(mol,targets,ntargets,cmapdef) ) {
      sprintf(errmsg,"Warning: add cross-term failed in patch %s",rname);
      topo_mol_log_error(mol,errmsg);
    }
  }
  for ( confdef = resdef->conformations; confdef; confdef = confdef->next ) {
    if ( confdef->del ) topo_mol_del_conformation(mol,targets,ntargets,confdef);
    else if ( topo_mol_add_conformation(mol,targets,ntargets,confdef) ) {
      sprintf(errmsg,"Warning: add conformation failed in patch %s",rname);
      topo_mol_log_error(mol,errmsg);
    }
  }

  if (strncasecmp(rname,"NONE",4)) {
    int ret;
    ret = topo_mol_add_patch(mol,rname,deflt);
    if (ret<0) {
      sprintf(errmsg,"Warning: Listing patch %s failed!",rname);
      topo_mol_log_error(mol,errmsg);
   }
    for ( idef=0; idef<ntargets; idef++ ) {
      printf("%s:%s ", targets[idef].segid,targets[idef].resid);
      topo_mol_add_patchres(mol,&targets[idef]);
    }
    printf("\n");
  }
  return 0;
}

int topo_mol_multiply_atoms(topo_mol *mol, const topo_mol_ident_t *targets,
						int ntargets, int ncopies) {
  int ipass, natoms, iatom, icopy;
  const topo_mol_ident_t *target;
  int itarget;
  topo_mol_atom_t *atom, **atoms;
  topo_mol_residue_t *res;
  topo_mol_segment_t *seg;
  int nres, ires;

  if (!mol) return -1;

  /* Quiet compiler warnings */
  natoms = 0; 
  atoms = NULL;

  /* two passes needed to find atoms */
  for (ipass=0; ipass<2; ++ipass) {
    if ( ipass ) atoms = memarena_alloc(mol->arena,
				natoms*sizeof(topo_mol_atom_t*));
    natoms = 0;
    /* walk all targets */
    for (itarget=0; itarget<ntargets; ++itarget) {
      target = targets + itarget;
      
      if (!target->resid) { /* whole segment */
        seg = topo_mol_get_seg(mol,target);
        if ( ! seg ) return -2;
        nres = hasharray_count(seg->residue_hash);
        for ( ires=0; ires<nres; ++ires ) {
          res = &(seg->residue_array[ires]);
          for ( atom = res->atoms; atom; atom = atom->next ) {
            if ( ipass ) atoms[natoms] = atom;
            ++natoms;
          }
        }
        continue;
      }

      if (!target->aname) { /* whole residue */
        res = topo_mol_get_res(mol,target,0);
        if ( ! res ) return -3;
        for ( atom = res->atoms; atom; atom = atom->next ) {
          if ( ipass ) atoms[natoms] = atom;
          ++natoms;
        }
        continue;
      }

      /* one atom */
      atom = topo_mol_get_atom(mol,target,0);
      if ( ! atom ) return -4;
      if ( ipass ) atoms[natoms] = atom;
      ++natoms;
    }
  }
  
  /* make one copy on each pass through loop */
  for (icopy=1; icopy<ncopies; ++icopy) {

  /* copy the actual atoms */
  for (iatom=0; iatom<natoms; ++iatom) {
    topo_mol_atom_t *newatom;
    atom = atoms[iatom];
    if ( atom->copy ) {
      topo_mol_log_error(mol,"an atom occurs twice in the selection");
      return -20;
    }
    newatom = memarena_alloc(mol->arena,sizeof(topo_mol_atom_t));
    if ( ! newatom ) return -5;
    memcpy(newatom,atom,sizeof(topo_mol_atom_t));
    atom->next = newatom;
    atom->copy = newatom;
    newatom->bonds = 0;
    newatom->angles = 0;
    newatom->dihedrals = 0;
    newatom->impropers = 0;
    newatom->cmaps = 0;
    newatom->conformations = 0;
  }

  /* copy associated bonds, etc. */
  for (iatom=0; iatom<natoms; ++iatom) {
    topo_mol_atom_t *a1, *a2, *a3, *a4;
    topo_mol_bond_t *bondtmp;
    topo_mol_angle_t *angletmp;
    topo_mol_dihedral_t *dihetmp;
    topo_mol_improper_t *imprtmp;
    topo_mol_cmap_t *cmaptmp;
    topo_mol_conformation_t *conftmp;
    atom = atoms[iatom];
    for ( bondtmp = atom->bonds; bondtmp;
		bondtmp = topo_mol_bond_next(bondtmp,atom) ) {
      topo_mol_bond_t *tuple;
      if ( bondtmp->del ) continue;
      if ( bondtmp->atom[0] == atom || ( ! bondtmp->atom[0]->copy ) ) ;
      else continue;
      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_bond_t));
      if ( ! tuple ) return -6;
      a1 = bondtmp->atom[0]->copy; if ( ! a1 ) a1 = bondtmp->atom[0];
      a2 = bondtmp->atom[1]->copy; if ( ! a2 ) a2 = bondtmp->atom[1];
      tuple->next[0] = a1->bonds;
      tuple->atom[0] = a1;
      tuple->next[1] = a2->bonds;
      tuple->atom[1] = a2;
      tuple->del = 0;
      a1->bonds = tuple;
      a2->bonds = tuple;
    }
    for ( angletmp = atom->angles; angletmp;
		angletmp = topo_mol_angle_next(angletmp,atom) ) {
      topo_mol_angle_t *tuple;
      if ( angletmp->del ) continue;
      if ( angletmp->atom[0] == atom || ( ! angletmp->atom[0]->copy
      && ( angletmp->atom[1] == atom || ( ! angletmp->atom[1]->copy ) ) ) ) ;
      else continue;
      tuple = memarena_alloc(mol->angle_arena,sizeof(topo_mol_angle_t));
      if ( ! tuple ) return -7;
      a1 = angletmp->atom[0]->copy; if ( ! a1 ) a1 = angletmp->atom[0];
      a2 = angletmp->atom[1]->copy; if ( ! a2 ) a2 = angletmp->atom[1];
      a3 = angletmp->atom[2]->copy; if ( ! a3 ) a3 = angletmp->atom[2];
      tuple->next[0] = a1->angles;
      tuple->atom[0] = a1;
      tuple->next[1] = a2->angles;
      tuple->atom[1] = a2;
      tuple->next[2] = a3->angles;
      tuple->atom[2] = a3;
      tuple->del = 0;
      a1->angles = tuple;
      a2->angles = tuple;
      a3->angles = tuple;
    }
    for ( dihetmp = atom->dihedrals; dihetmp;
		dihetmp = topo_mol_dihedral_next(dihetmp,atom) ) {
      topo_mol_dihedral_t *tuple;
      if ( dihetmp->del ) continue;
      if ( dihetmp->atom[0] == atom || ( ! dihetmp->atom[0]->copy
      && ( dihetmp->atom[1] == atom || ( ! dihetmp->atom[1]->copy
      && ( dihetmp->atom[2] == atom || ( ! dihetmp->atom[2]->copy ) ) ) ) ) ) ;
      else continue;
      tuple = memarena_alloc(mol->dihedral_arena,sizeof(topo_mol_dihedral_t));
      if ( ! tuple ) return -8;
      a1 = dihetmp->atom[0]->copy; if ( ! a1 ) a1 = dihetmp->atom[0];
      a2 = dihetmp->atom[1]->copy; if ( ! a2 ) a2 = dihetmp->atom[1];
      a3 = dihetmp->atom[2]->copy; if ( ! a3 ) a3 = dihetmp->atom[2];
      a4 = dihetmp->atom[3]->copy; if ( ! a4 ) a4 = dihetmp->atom[3];
      tuple->next[0] = a1->dihedrals;
      tuple->atom[0] = a1;
      tuple->next[1] = a2->dihedrals;
      tuple->atom[1] = a2;
      tuple->next[2] = a3->dihedrals;
      tuple->atom[2] = a3;
      tuple->next[3] = a4->dihedrals;
      tuple->atom[3] = a4;
      tuple->del = 0;
      a1->dihedrals = tuple;
      a2->dihedrals = tuple;
      a3->dihedrals = tuple;
      a4->dihedrals = tuple;
    }
    for ( imprtmp = atom->impropers; imprtmp;
		imprtmp = topo_mol_improper_next(imprtmp,atom) ) {
      topo_mol_improper_t *tuple;
      if ( imprtmp->del ) continue;
      if ( imprtmp->atom[0] == atom || ( ! imprtmp->atom[0]->copy
      && ( imprtmp->atom[1] == atom || ( ! imprtmp->atom[1]->copy
      && ( imprtmp->atom[2] == atom || ( ! imprtmp->atom[2]->copy ) ) ) ) ) ) ;
      else continue;
      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_improper_t));
      if ( ! tuple ) return -9;
      a1 = imprtmp->atom[0]->copy; if ( ! a1 ) a1 = imprtmp->atom[0];
      a2 = imprtmp->atom[1]->copy; if ( ! a2 ) a2 = imprtmp->atom[1];
      a3 = imprtmp->atom[2]->copy; if ( ! a3 ) a3 = imprtmp->atom[2];
      a4 = imprtmp->atom[3]->copy; if ( ! a4 ) a4 = imprtmp->atom[3];
      tuple->next[0] = a1->impropers;
      tuple->atom[0] = a1;
      tuple->next[1] = a2->impropers;
      tuple->atom[1] = a2;
      tuple->next[2] = a3->impropers;
      tuple->atom[2] = a3;
      tuple->next[3] = a4->impropers;
      tuple->atom[3] = a4;
      tuple->del = 0;
      a1->impropers = tuple;
      a2->impropers = tuple;
      a3->impropers = tuple;
      a4->impropers = tuple;
    }
    for ( cmaptmp = atom->cmaps; cmaptmp;
		cmaptmp = topo_mol_cmap_next(cmaptmp,atom) ) {
      topo_mol_atom_t *al[8];
      topo_mol_cmap_t *tuple;
      int ia, skip;
      if ( cmaptmp->del ) continue;
      skip = 0;
      for ( ia = 0; ia < 8; ++ia ) {
        if ( cmaptmp->atom[ia] == atom ) { skip = 0; break; }
        if ( cmaptmp->atom[ia]->copy ) { skip = 1; break; }
      }
      if ( skip ) continue;
      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_cmap_t));
      if ( ! tuple ) return -9;
      for ( ia = 0; ia < 8; ++ia ) {
        topo_mol_atom_t *ai;
        ai = cmaptmp->atom[ia]->copy;
        if ( ! ai ) ai = cmaptmp->atom[ia];
        al[ia] = ai;
        tuple->next[ia] = ai->cmaps;
        tuple->atom[ia] = ai;
      }
      for ( ia = 0; ia < 8; ++ia ) {
        /* This must be in a separate loop because atoms may be repeated. */
        al[ia]->cmaps = tuple;
      }
      tuple->del = 0;
    }
    for ( conftmp = atom->conformations; conftmp;
		conftmp = topo_mol_conformation_next(conftmp,atom) ) {
      topo_mol_conformation_t *tuple;
      if ( conftmp->del ) continue;
      if ( conftmp->atom[0] == atom || ( ! conftmp->atom[0]->copy
      && ( conftmp->atom[1] == atom || ( ! conftmp->atom[1]->copy
      && ( conftmp->atom[2] == atom || ( ! conftmp->atom[2]->copy ) ) ) ) ) ) ;
      else continue;
      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_conformation_t));
      if ( ! tuple ) return -10;
      a1 = conftmp->atom[0]->copy; if ( ! a1 ) a1 = conftmp->atom[0];
      a2 = conftmp->atom[1]->copy; if ( ! a2 ) a2 = conftmp->atom[1];
      a3 = conftmp->atom[2]->copy; if ( ! a3 ) a3 = conftmp->atom[2];
      a4 = conftmp->atom[3]->copy; if ( ! a4 ) a4 = conftmp->atom[3];
      tuple->next[0] = a1->conformations;
      tuple->atom[0] = a1;
      tuple->next[1] = a2->conformations;
      tuple->atom[1] = a2;
      tuple->next[2] = a3->conformations;
      tuple->atom[2] = a3;
      tuple->next[3] = a4->conformations;
      tuple->atom[3] = a4;
      tuple->del = 0;
      tuple->improper = conftmp->improper;
      tuple->dist12 = conftmp->dist12;
      tuple->angle123 = conftmp->angle123;
      tuple->dihedral = conftmp->dihedral;
      tuple->angle234 = conftmp->angle234;
      tuple->dist34 = conftmp->dist34;
      a1->conformations = tuple;
      a2->conformations = tuple;
      a3->conformations = tuple;
      a4->conformations = tuple;
    }
  }

  /* clean up copy pointers */
  for (iatom=0; iatom<natoms; ++iatom) {
    atom = atoms[iatom];
    if ( atom->partition == 0 ) atom->partition = 1;
    atom->copy->partition = atom->partition + 1;
    atoms[iatom] = atom->copy;
    atom->copy = 0;
  }

  } /* icopy */

  return 0;  /* success */
}

/* API function */
void topo_mol_delete_atom(topo_mol *mol, const topo_mol_ident_t *target) {
  
  topo_mol_residue_t *res;
  topo_mol_segment_t *seg;
  int ires, iseg;
  if (!mol) return;

  iseg = hasharray_index(mol->segment_hash,target->segid);
  if ( iseg == HASHARRAY_FAIL ) {
    char errmsg[50];
    sprintf(errmsg,"no segment %s",target->segid);
    topo_mol_log_error(mol,errmsg);
    return;
  }
  seg = mol->segment_array[iseg];
  
  if (!target->resid) {
    /* Delete this segment */
    int nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      topo_mol_atom_t *atom;
      res = &(seg->residue_array[ires]);
      atom = res->atoms;
      while (atom) {
        topo_mol_destroy_atom(atom);
        atom = atom->next;
      }
      res->atoms = 0;
    }
    hasharray_destroy(seg->residue_hash);
    mol->segment_array[iseg] = 0;
    if (hasharray_delete(mol->segment_hash, target->segid) < 0) {
      topo_mol_log_error(mol, "Unable to delete segment");
    }
    return;
  }

  ires = hasharray_index(seg->residue_hash,target->resid);
  if ( ires == HASHARRAY_FAIL ) {
    char errmsg[50];
    sprintf(errmsg,"no residue %s of segment %s",
                                        target->resid,target->segid);
    topo_mol_log_error(mol,errmsg);
    return;
  }
  res = seg->residue_array+ires;  
  
  if (!target->aname) {  
    /* Must destroy all atoms in residue, since there may be bonds between
       this residue and other atoms 
    */
    topo_mol_atom_t *atom = res->atoms;
    while (atom) {
      topo_mol_destroy_atom(atom);
      atom = atom->next;
    }
    res->atoms = 0;
    hasharray_delete(seg->residue_hash, target->resid); 
    return;
  }
  /* Just delete one atom */
  topo_mol_destroy_atom(topo_mol_unlink_atom(&(res->atoms),target->aname));
}

int topo_mol_set_element(topo_mol *mol, const topo_mol_ident_t *target,
                                        const char *element, int replace) {
  topo_mol_residue_t *res;
  topo_mol_atom_t *atom;
  if ( ! mol ) return -1;
  if ( ! target ) return -2;
  res = topo_mol_get_res(mol,target,0);
  if ( ! res ) return -3;
  for ( atom = res->atoms; atom; atom = atom->next ) {
    if ( ! strcmp(target->aname,atom->name) ) break;
  }
  if ( ! atom ) return -3;

  if ( replace || ! strlen(atom->element) ) {
    strcpy(atom->element,element);
  }
  return 0;
}

int topo_mol_set_chain(topo_mol *mol, const topo_mol_ident_t *target,
                                        const char *chain, int replace) {
  topo_mol_residue_t *res;
  if ( ! mol ) return -1;
  if ( ! target ) return -2;
  res = topo_mol_get_res(mol,target,0);
  if ( ! res ) return -3;

  if ( replace || ! strlen(res->chain) ) {
    strcpy(res->chain,chain);
  }
  return 0;
}

int topo_mol_set_xyz(topo_mol *mol, const topo_mol_ident_t *target,
                                        double x, double y, double z) {
  topo_mol_residue_t *res;
  topo_mol_atom_t *atom;
  if ( ! mol ) return -1;
  if ( ! target ) return -2;
  res = topo_mol_get_res(mol,target,0);
  if ( ! res ) return -3;
  for ( atom = res->atoms; atom; atom = atom->next ) {
    if ( ! strcmp(target->aname,atom->name) ) break;
  }
  if ( ! atom ) return -3;

  atom->x = x;
  atom->y = y;
  atom->z = z;
  atom->xyz_state = TOPO_MOL_XYZ_SET;
  return 0;
}

/* XXX Unused */
int topo_mol_clear_xyz(topo_mol *mol, const topo_mol_ident_t *target) {
  topo_mol_atom_t *atom;
  if ( ! mol ) return -1;
  if ( ! target ) return -2;

  atom = topo_mol_get_atom(mol,target,0);
  if ( ! atom ) return -3;

  atom->x = 0;
  atom->y = 0;
  atom->z = 0;
  atom->xyz_state = TOPO_MOL_XYZ_VOID;

  return 0;
}


int topo_mol_guess_xyz(topo_mol *mol) {
  char msg[128];
  int iseg,nseg,ires,nres,ucount,i,nk,nu,gcount,gwild,okwild,wcount,hcount;
  int ipass;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_mol_atom_t *atom, *a1, *a2, *a3;
  topo_mol_atom_t *ka[4];
  topo_mol_atom_t *ua[4];
  topo_mol_bond_t *bondtmp;
  topo_mol_angle_t *angletmp;
  double dihedral, angle234, dist34;
  topo_mol_atom_t **uatoms;
  topo_mol_conformation_t *conf;
  double r12x,r12y,r12z,r12,r23x,r23y,r23z,r23,ix,iy,iz,jx,jy,jz,kx,ky,kz;
  double tx,ty,tz,a,b,c;

  if ( ! mol ) return -1;

  ucount = 0;
  hcount = 0;
  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        if ( atom->xyz_state != TOPO_MOL_XYZ_SET ) {
          ++ucount;
          if ( atom->mass > 2.5 ) ++hcount;
        }
      }
    }
  }
  sprintf(msg,"Info: guessing coordinates for %d atoms (%d non-hydrogen)",
						ucount, hcount);
  topo_mol_log_error(mol,msg);

  uatoms = (topo_mol_atom_t**) malloc(ucount*sizeof(topo_mol_atom_t*));
  if ( ! uatoms ) return -2;
  ucount = 0;
  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        if ( atom->xyz_state != TOPO_MOL_XYZ_SET ) uatoms[ucount++] = atom;
      }
    }
  }

  for ( i=0; i<ucount; ++i ) uatoms[i]->xyz_state = TOPO_MOL_XYZ_VOID;

  /* everything below based on atom 4 unknown, all others known */

  /* from the CHARMM docs:

    Normal IC table entry:
                I
                 \
                  \
                   J----K
                         \
                          \
                           L
        values (Rij),(Tijk),(Pijkl),(Tjkl),(Rkl)

    Improper type of IC table entry:
                I        L
                 \     /
                  \   /
                   *K
                   |
                   |
                   J
        values (Rik),(Tikj),(Pijkl),T(jkl),(Rkl)

  */

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif

  gcount = 1;
  okwild = 0;
  wcount = 0;
  hcount = 0;
  while ( gcount || ! okwild ) {
   if ( gcount == 0 ) { if ( okwild ) break; else okwild = 1; }
   gcount = 0;
   for ( i=0; i<ucount; ++i ) { atom = uatoms[i];
    if ( atom->xyz_state != TOPO_MOL_XYZ_VOID ) continue;
    for ( conf = atom->conformations; conf;
		conf = topo_mol_conformation_next(conf,atom) ) {
      if ( conf->del ) continue;
      else if ( conf->atom[0] == atom &&
		conf->atom[1]->xyz_state != TOPO_MOL_XYZ_VOID &&
		conf->atom[2]->xyz_state != TOPO_MOL_XYZ_VOID &&
		conf->atom[3]->xyz_state != TOPO_MOL_XYZ_VOID ) {
        if ( conf->improper ) {
          a1 = conf->atom[3]; a2 = conf->atom[1]; a3 = conf->atom[2];
          dist34 = conf->dist12;
          angle234 = conf->angle123 * (M_PI/180.0);
          dihedral = -1.0 * conf->dihedral * (M_PI/180.0);
        } else {
          a1 = conf->atom[3]; a2 = conf->atom[2]; a3 = conf->atom[1];
          dist34 = conf->dist12;
          angle234 = conf->angle123 * (M_PI/180.0);
          dihedral = conf->dihedral * (M_PI/180.0);
        } 
      } 
      else if ( conf->atom[3] == atom &&
		conf->atom[2]->xyz_state != TOPO_MOL_XYZ_VOID &&
		conf->atom[1]->xyz_state != TOPO_MOL_XYZ_VOID &&
		conf->atom[0]->xyz_state != TOPO_MOL_XYZ_VOID ) {
        if ( conf->improper ) {
          a1 = conf->atom[0]; a2 = conf->atom[1]; a3 = conf->atom[2];
          dist34 = conf->dist34;
          angle234 = conf->angle234 * (M_PI/180.0);
          dihedral = conf->dihedral * (M_PI/180.0);
        } else {
          a1 = conf->atom[0]; a2 = conf->atom[1]; a3 = conf->atom[2];
          dist34 = conf->dist34;
          angle234 = conf->angle234 * (M_PI/180.0);
          dihedral = conf->dihedral * (M_PI/180.0);
        } 
      } 
      else continue;

      gwild = 0;
      if ( dist34 == 0.0 ) { dist34 = 1.0; gwild = 1; }
      if ( angle234 == 0.0 ) { angle234 = 109.0*M_PI/180.0; gwild = 1; }

      r12x = a2->x - a1->x;
      r12y = a2->y - a1->y;
      r12z = a2->z - a1->z;
      r23x = a3->x - a2->x;
      r23y = a3->y - a2->y;
      r23z = a3->z - a2->z;
      a = sqrt(r23x*r23x + r23y*r23y + r23z*r23z);
      if ( a == 0.0 ) gwild = 1; else a = 1.0 / a;
      ix = a * r23x;
      iy = a * r23y;
      iz = a * r23z;
      tx = r12y*r23z - r12z*r23y;
      ty = r12z*r23x - r12x*r23z;
      tz = r12x*r23y - r12y*r23x;
      a = sqrt(tx*tx + ty*ty + tz*tz);
      if ( a == 0.0 ) gwild = 1; else a = 1.0 / a;
      kx = a * tx;
      ky = a * ty;
      kz = a * tz;
      tx = ky*iz - kz*iy;
      ty = kz*ix - kx*iz;
      tz = kx*iy - ky*ix;
      a = sqrt(tx*tx + ty*ty + tz*tz);
      if ( a == 0.0 ) gwild = 1; else a = 1.0 / a;
      jx = a * tx;
      jy = a * ty;
      jz = a * tz;
      a = -1.0 * dist34 * cos(angle234);
      b = dist34 * sin(angle234) * cos(dihedral);
      c = dist34 * sin(angle234) * sin(dihedral);

      if ( gwild && ! okwild ) continue;
      if ( okwild ) {
        ++wcount;
        if ( atom->mass > 2.5 ) ++hcount;
      }

      atom->x = a3->x + a * ix + b * jx + c * kx;
      atom->y = a3->y + a * iy + b * jy + c * ky;
      atom->z = a3->z + a * iz + b * jz + c * kz;
      atom->xyz_state = okwild ? TOPO_MOL_XYZ_BADGUESS : TOPO_MOL_XYZ_GUESS; 
      ++gcount;
      break;  /* don't re-guess this atom */
    }
   }
  }

  /* look for bad angles due to swapped atom names */
  for ( i=0; i<ucount; ++i ) { atom = uatoms[i];
    /* only look for errors in guessed atoms */
    if ( atom->xyz_state == TOPO_MOL_XYZ_VOID ||
         atom->xyz_state == TOPO_MOL_XYZ_SET ) continue;

    for ( angletmp = atom->angles; angletmp;
		angletmp = topo_mol_angle_next(angletmp,atom) ) {
      if ( angletmp->del ) continue;
      if ( angletmp->atom[0] == atom ) {
        a1 = angletmp->atom[2];
        a2 = angletmp->atom[1];
      } else if ( angletmp->atom[2] == atom ) {
        a1 = angletmp->atom[0];
        a2 = angletmp->atom[1];
      } else continue;
      /* only use set atoms, don't hid topology file errors */
      if ( a1->xyz_state != TOPO_MOL_XYZ_SET ) continue;
      if ( a2->xyz_state != TOPO_MOL_XYZ_SET ) continue;

      r12x = a2->x - a1->x;
      r12y = a2->y - a1->y;
      r12z = a2->z - a1->z;
      r12 = sqrt(r12x*r12x + r12y*r12y + r12z*r12z);
      r23x = atom->x - a2->x;
      r23y = atom->y - a2->y;
      r23z = atom->z - a2->z;
      r23 = sqrt(r23x*r23x + r23y*r23y + r23z*r23z);
      /* assume wrong if angle is less than 45 degrees */
      if ( r12x*r23x + r12y*r23y + r12z*r23z < r12 * r23 * -0.7 ) {
        sprintf(msg, "Warning: failed to guess coordinate due to bad angle %s %s %s",
            a1->name, a2->name, atom->name);
        topo_mol_log_error(mol, msg);
        if ( atom->xyz_state == TOPO_MOL_XYZ_BADGUESS ) {
          --wcount;
          if ( atom->mass > 2.5 ) --hcount;
        }
        --gcount;
        atom->xyz_state = TOPO_MOL_XYZ_VOID; 
        break;
      }
    }
  }

  /* fallback rules for atoms without conformation records */
  for ( ipass=0; ipass<2; ++ipass ) {  /* don't do entire chain */
  for ( i=0; i<ucount; ++i ) { atom = uatoms[i];
    if ( atom->xyz_state != TOPO_MOL_XYZ_VOID ) continue;

    /* pick heaviest known atom we are bonded to (to deal with water) */
    a1 = 0;
    for ( bondtmp = atom->bonds; bondtmp;
		bondtmp = topo_mol_bond_next(bondtmp,atom) ) {
      if ( bondtmp->atom[0] == atom ) a2 = bondtmp->atom[1];
      else a2 = bondtmp->atom[0];
      if ( a2->xyz_state == TOPO_MOL_XYZ_VOID ) continue;
      if ( a1 == 0 || a2->mass > a1->mass ) a1 = a2;
    }
    if ( a1 == 0 ) continue;
    atom = a1;

    /* find all bonded atoms known and unknown coordinates */
    nk = 0;  nu = 0;
    for ( bondtmp = atom->bonds; bondtmp;
		bondtmp = topo_mol_bond_next(bondtmp,atom) ) {
      if ( bondtmp->del ) continue;
      if ( bondtmp->atom[0] == atom ) a2 = bondtmp->atom[1];
      else a2 = bondtmp->atom[0];
      if ( a2->xyz_state == TOPO_MOL_XYZ_VOID ) {
        if ( nu < 4 ) ua[nu++] = a2;
      } else {
        if ( nk < 4 ) ka[nk++] = a2;
      }
    }

    if ( ipass ) {  /* hydrogens only on second pass */
      int j;
      for ( j=0; j<nu && ua[j]->mass < 2.5; ++j );
      if ( j != nu ) continue;
    }

    if ( nu + nk > 4 ) continue;  /* no intuition beyond this case */

    if ( nk == 0 ) {  /* not bonded to any known atoms */
      a1 = ua[0];
      a1->x = atom->x + 1.0;
      a1->y = atom->y;
      a1->z = atom->z;
      a1->xyz_state = TOPO_MOL_XYZ_BADGUESS;
      ++gcount;  ++wcount;
      if ( a1->mass > 2.5 ) ++hcount;
      continue;
    }

    if ( nk == 1 ) {  /* bonded to one known atom */
      a1 = ka[0];
      ix = a1->x - atom->x;
      iy = a1->y - atom->y;
      iz = a1->z - atom->z;
      a = sqrt(ix*ix+iy*iy+iz*iz);
      if ( a ) a = 1.0 / a;  else continue;
      ix *= a; iy *= a; iz *= a;
      jx = -1.0 * iy;  jy = ix;  jz = 0;
      if ( jx*jx + jy*jy + jz*jz < 0.1 ) {
        jx = 0;  jy = -1.0 * iz;  jz = iy;
      }
      a = sqrt(jx*jx+jy*jy+jz*jz);
      if ( a ) a = 1.0 / a;  else continue;
      jx *= a; jy *= a; jz *= a;
      if ( nu == 1 ) {  /* one unknown atom */
        a = cos(109.0*M_PI/180.0);
        b = sin(109.0*M_PI/180.0);
        a2 = ua[0];
        a2->x = atom->x + a * ix + b * jx;
        a2->y = atom->y + a * iy + b * jy;
        a2->z = atom->z + a * iz + b * jz;
        a2->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a2->mass > 2.5 ) ++hcount;
      } else if ( nu == 2 ) {  /* two unknown atoms */
        a = cos(120.0*M_PI/180.0);
        b = sin(120.0*M_PI/180.0);
        a1 = ua[0];
        a2 = ua[1];
        a1->x = atom->x + a * ix + b * jx;
        a1->y = atom->y + a * iy + b * jy;
        a1->z = atom->z + a * iz + b * jz;
        a2->x = atom->x + a * ix - b * jx;
        a2->y = atom->y + a * iy - b * jy;
        a2->z = atom->z + a * iz - b * jz;
        a1->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a1->mass > 2.5 ) ++hcount;
        a2->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a2->mass > 2.5 ) ++hcount;
      } else { /* three unknown atoms */
        a1 = ua[0];
        a2 = ua[1];
        a3 = ua[2];
        /* only handle this case if at least two are hydrogens */
        if ( a1->mass > 2.5 && a2->mass > 2.5 ) continue;
        if ( a1->mass > 2.5 && a3->mass > 2.5 ) continue;
        if ( a2->mass > 2.5 && a3->mass > 2.5 ) continue;
        kx = iy*jz - iz*jy;
        ky = iz*jx - ix*jz;
        kz = ix*jy - iy*jx;
        a = sqrt(kx*kx+ky*ky+kz*kz);
        if ( a ) a = 1.0 / a;  else continue;
        kx *= a; ky *= a; kz *= a;
        a = cos(109.0*M_PI/180.0);
        b = sin(109.0*M_PI/180.0);
        a1->x = atom->x + a * ix + b * jx;
        a1->y = atom->y + a * iy + b * jy;
        a1->z = atom->z + a * iz + b * jz;
        c = b * sin(120.0*M_PI/180.0);
        b *= cos(120.0*M_PI/180.0);
        a2->x = atom->x + a * ix + b * jx + c * kx;
        a2->y = atom->y + a * iy + b * jy + c * ky;
        a2->z = atom->z + a * iz + b * jz + c * kz;
        a3->x = atom->x + a * ix + b * jx - c * kx;
        a3->y = atom->y + a * iy + b * jy - c * ky;
        a3->z = atom->z + a * iz + b * jz - c * kz;
        a1->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a1->mass > 2.5 ) ++hcount;
        a2->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a2->mass > 2.5 ) ++hcount;
        a3->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a3->mass > 2.5 ) ++hcount;
      }
      continue;
    }

    if ( nk == 2 ) {  /* bonded to two known atoms */
      a1 = ka[0];
      ix = a1->x - atom->x;
      iy = a1->y - atom->y;
      iz = a1->z - atom->z;
      a = sqrt(ix*ix+iy*iy+iz*iz);
      if ( a ) a = 1.0 / a;  else continue;
      ix *= a; iy *= a; iz *= a;
      jx = ix;  jy = iy;  jz = iz;
      a1 = ka[1];
      ix = a1->x - atom->x;
      iy = a1->y - atom->y;
      iz = a1->z - atom->z;
      a = sqrt(ix*ix+iy*iy+iz*iz);
      if ( a ) a = 1.0 / a;  else continue;
      ix *= a; iy *= a; iz *= a;
      kx = jx - ix;  ky = jy - iy;  kz = jz - iz;
      jx += ix;  jy += iy;  jz += iz;
      a = sqrt(jx*jx+jy*jy+jz*jz);
      if ( a ) a = 1.0 / a;  else continue;
      jx *= a; jy *= a; jz *= a;
      if ( nu == 1 ) {  /* one unknown atom */
        a2 = ua[0];
        a2->x = atom->x - jx;
        a2->y = atom->y - jy;
        a2->z = atom->z - jz;
        a2->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a2->mass > 2.5 ) ++hcount;
      } else {  /* two unknown atoms */
        a1 = ua[0];
        a2 = ua[1];
        /* only handle this case if both are hydrogens */
        if ( a1->mass > 2.5 || a2->mass > 2.5 ) continue;
        a = sqrt(kx*kx+ky*ky+kz*kz);
        if ( a ) a = 1.0 / a;  else continue;
        kx *= a; ky *= a; kz *= a;
        ix = jy*kz - jz*ky;
        iy = jz*kx - jx*kz;
        iz = jx*ky - jy*kx;
        a = sqrt(ix*ix+iy*iy+iz*iz);
        if ( a ) a = 1.0 / a;  else continue;
        ix *= a; iy *= a; iz *= a;
        angle234 = (180.0-0.5*109.0)*M_PI/180.0;
        a = sin(angle234);
        b = cos(angle234);
        a1->x = atom->x + a * ix + b * jx;
        a1->y = atom->y + a * iy + b * jy;
        a1->z = atom->z + a * iz + b * jz;
        a2->x = atom->x - a * ix + b * jx;
        a2->y = atom->y - a * iy + b * jy;
        a2->z = atom->z - a * iz + b * jz;
        a1->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a1->mass > 2.5 ) ++hcount;
        a2->xyz_state = TOPO_MOL_XYZ_BADGUESS;
        ++gcount;  ++wcount;
        if ( a2->mass > 2.5 ) ++hcount;
      }
      continue;
    }

    if ( nk == 3 ) {  /* bonded to three known atoms */
      a1 = ka[0];
      ix = a1->x - atom->x;
      iy = a1->y - atom->y;
      iz = a1->z - atom->z;
      a = sqrt(ix*ix+iy*iy+iz*iz);
      if ( a ) a = 1.0 / a;  else continue;
      ix *= a; iy *= a; iz *= a;
      jx = ix;  jy = iy;  jz = iz;
      a1 = ka[1];
      ix = a1->x - atom->x;
      iy = a1->y - atom->y;
      iz = a1->z - atom->z;
      a = sqrt(ix*ix+iy*iy+iz*iz);
      if ( a ) a = 1.0 / a;  else continue;
      ix *= a; iy *= a; iz *= a;
      jx += ix;  jy += iy;  jz += iz;
      a1 = ka[2];
      ix = a1->x - atom->x;
      iy = a1->y - atom->y;
      iz = a1->z - atom->z;
      a = sqrt(ix*ix+iy*iy+iz*iz);
      if ( a ) a = 1.0 / a;  else continue;
      ix *= a; iy *= a; iz *= a;
      jx += ix;  jy += iy;  jz += iz;
      a = sqrt(jx*jx+jy*jy+jz*jz);
      if ( a ) a = 1.0 / a;  else continue;
      a2 = ua[0];
      a2->x = atom->x - a * jx;
      a2->y = atom->y - a * jy;
      a2->z = atom->z - a * jz;
      a2->xyz_state = TOPO_MOL_XYZ_BADGUESS;
      ++gcount;  ++wcount;
      if ( a2->mass > 2.5 ) ++hcount;
      continue;
    }

  }
  }

  gcount = 0;
  for ( i=0; i<ucount; ++i ) {
    if ( uatoms[i]->xyz_state == TOPO_MOL_XYZ_VOID ) ++gcount;
  }
  if ( wcount ) {
    sprintf(msg,"Warning: poorly guessed coordinates for %d atoms (%d non-hydrogen):", wcount, hcount);
    topo_mol_log_error(mol,msg);
    for ( iseg=0; iseg<nseg; ++iseg ) {
      seg = mol->segment_array[iseg];
      if (! seg) continue;
      nres = hasharray_count(seg->residue_hash);
      for ( ires=0; ires<nres; ++ires ) {
        res = &(seg->residue_array[ires]);
        for ( atom = res->atoms; atom; atom = atom->next ) {
          if ( atom->xyz_state == TOPO_MOL_XYZ_BADGUESS) {
            sprintf(msg, "Warning: poorly guessed coordinate for atom %s\t %s:%s\t  %s",
                atom->name, res->name, res->resid, seg->segid);
            topo_mol_log_error(mol, msg);
          }
        }
      }
    }
  }
  if ( gcount ) {
    sprintf(msg,"Warning: failed to guess coordinates for %d atoms",gcount);
    topo_mol_log_error(mol,msg);
  }

  free((void*)uatoms);

  return 0;
}


/* Copied and modified from topo_mol_segment */
int topo_mol_add_patch(topo_mol *mol, const char *pname, int deflt) {
  topo_mol_patch_t **patches;
  topo_mol_patch_t *patchtmp;
  if ( ! mol ) return -1;
  if ( NAMETOOLONG(pname) ) return -2;
  patches = &(mol->patches);
  
  patchtmp = 0;
  patchtmp = memarena_alloc(mol->arena,sizeof(topo_mol_patch_t));
  if ( ! patchtmp ) return -3;
  
  strcpy(patchtmp->pname,pname);
  patchtmp->patchresids = 0;

  patchtmp->npres = 0;
  patchtmp->deflt = deflt;
  patchtmp->next = 0;
/*    printf("add_patch %i %s;\n", mol->npatch, patchtmp->pname);   */

  if (mol->npatch==0) {
    *patches = patchtmp;
  } else {
    mol->curpatch->next = patchtmp;
  }
  mol->curpatch = patchtmp;

  mol->npatch++;
  return 0;
}


/* Copied and modified from topo_mol_residue */
int topo_mol_add_patchres(topo_mol *mol, const topo_mol_ident_t *target) {
  topo_mol_patch_t *patch;
  topo_mol_patchres_t **patchres;
  topo_mol_patchres_t *patchrestmp;
  if ( ! mol ) return -1;
  if ( NAMETOOLONG(target->segid) ) return -2;
  if ( NAMETOOLONG(target->resid) ) return -2;

  patch = mol->curpatch; 
  patchres = &(patch->patchresids);
  patchrestmp = 0;
  patchrestmp = memarena_alloc(mol->arena,sizeof(topo_mol_patchres_t));
  if ( ! patchrestmp ) return -3;

  strcpy(patchrestmp->segid,target->segid);
  strcpy(patchrestmp->resid,target->resid);
/*   printf("add_patchres %i %s:%s;\n", patch->npres, patchrestmp->segid, patchrestmp->resid);  */
  patch->npres++;
  /* patchrestmp->next = *patchres;  old code builds list in reverse order */
  patchrestmp->next = NULL;
  while ( *patchres ) { patchres = &((*patchres)->next); }
  *patchres = patchrestmp;
  return 0;
}


/* Test the existence of segid:resid for the patch */
int topo_mol_validate_patchres(topo_mol *mol, const char *pname, const char *segid, const char *resid) {
  topo_mol_ident_t target;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  target.segid = segid;
  target.resid = resid;
  seg = topo_mol_get_seg(mol,&target);
  if ( ! seg ) {
    char errmsg[50];
    sprintf(errmsg,"Segment %s not exsisting, skipping patch %s.\n",segid,pname);
    topo_mol_log_error(mol,errmsg);
    return 0;
  }
  res = topo_mol_get_res(mol,&target,0);
  if ( ! res ) {
    char errmsg[50];
    sprintf(errmsg,"Residue %s:%s not exsisting, skipping patch %s.\n",segid,resid,pname);
    topo_mol_log_error(mol,errmsg);
    return 0;
  }
  return 1;
}
