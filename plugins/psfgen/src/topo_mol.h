
#ifndef TOPO_MOL_H
#define TOPO_MOL_H

#include "topo_defs.h"

struct topo_mol;
typedef struct topo_mol topo_mol;

topo_mol * topo_mol_create(topo_defs *defs);
void topo_mol_destroy(topo_mol *mol);

void topo_mol_error_handler(topo_mol *mol, void *, void (*print_msg)(void *,const char *));

int topo_mol_segment(topo_mol *mol, const char *segid);

int topo_mol_segment_first(topo_mol *mol, const char *rname);
int topo_mol_segment_last(topo_mol *mol, const char *rname);

int topo_mol_segment_auto_angles(topo_mol *mol, int autogen);
int topo_mol_segment_auto_dihedrals(topo_mol *mol, int autogen);

int topo_mol_residue(topo_mol *mol, const char *resid, const char *rname,
						const char *chain);
int topo_mol_mutate(topo_mol *mol, const char *resid, const char *rname);

int topo_mol_end(topo_mol *mol);

typedef struct topo_mol_ident_t {
  const char *segid;
  const char *resid;
  const char *aname;
} topo_mol_ident_t;

int topo_mol_patch(topo_mol *mol, const topo_mol_ident_t *targets,
			int ntargets, const char *rname, int prepend,
			int warn_angles, int warn_dihedrals, int deflt);

int topo_mol_regenerate_angles(topo_mol *mol);
int topo_mol_regenerate_dihedrals(topo_mol *mol);
int topo_mol_regenerate_resids(topo_mol *mol);

void topo_mol_delete_atom(topo_mol *mol, const topo_mol_ident_t *target);

int topo_mol_multiply_atoms(topo_mol *mol, const topo_mol_ident_t *targets,
					int ntargets, int ncopies);

int topo_mol_set_element(topo_mol *mol, const topo_mol_ident_t *target,
					const char *element, int replace);

int topo_mol_set_chain(topo_mol *mol, const topo_mol_ident_t *target,
					const char *chain, int replace);

int topo_mol_set_xyz(topo_mol *mol, const topo_mol_ident_t *target,
					double x, double y, double z);

int topo_mol_guess_xyz(topo_mol *mol);

int topo_mol_add_patch(topo_mol *mol, const char *pname, int deflt);

int topo_mol_add_patchres(topo_mol *mol, const topo_mol_ident_t *targets);

int topo_mol_validate_patchres(topo_mol *mol, const char *pname, const char *segid, const char *resid);

#endif

