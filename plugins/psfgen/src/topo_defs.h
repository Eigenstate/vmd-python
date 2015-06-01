
#ifndef TOPO_DEFS_H
#define TOPO_DEFS_H

struct topo_defs;
typedef struct topo_defs topo_defs;

topo_defs * topo_defs_create(void);
void topo_defs_destroy(topo_defs *defs);

void topo_defs_error_handler(topo_defs *defs, void *,
                             void (*print_msg)(void *, const char *));

void topo_defs_auto_angles(topo_defs *defs, int autogen);
void topo_defs_auto_dihedrals(topo_defs *defs, int autogen);

int topo_defs_type(topo_defs *defs, const char *atype, const char *element, double mass, int id);

int topo_defs_residue(topo_defs *defs, const char *rname, int patch);
int topo_defs_end(topo_defs *defs);

int topo_defs_atom(topo_defs *defs, const char *rname, int del,
	const char *aname, int ares, int arel,
	const char *atype, double charge);

int topo_defs_bond(topo_defs *defs, const char *rname, int del,
	const char *a1name, int a1res, int a1rel,
	const char *a2name, int a2res, int a2rel);

int topo_defs_angle(topo_defs *defs, const char *rname, int del,
	const char *a1name, int a1res, int a1rel,
	const char *a2name, int a2res, int a2rel,
	const char *a3name, int a3res, int a3rel);

int topo_defs_dihedral(topo_defs *defs, const char *rname, int del,
	const char *a1name, int a1res, int a1rel,
	const char *a2name, int a2res, int a2rel,
	const char *a3name, int a3res, int a3rel,
	const char *a4name, int a4res, int a4rel);

int topo_defs_improper(topo_defs *defs, const char *rname, int del,
	const char *a1name, int a1res, int a1rel,
	const char *a2name, int a2res, int a2rel,
	const char *a3name, int a3res, int a3rel,
	const char *a4name, int a4res, int a4rel);

int topo_defs_cmap(topo_defs *defs, const char *rname, int del,
	const char* const anamel[8], const int aresl[8], const int arell[8]);

int topo_defs_conformation(topo_defs *defs, const char *rname, int del,
	const char *a1name, int a1res, int a1rel,
	const char *a2name, int a2res, int a2rel,
	const char *a3name, int a3res, int a3rel,
	const char *a4name, int a4res, int a4rel,
	double dist12, double angle123, double dihedral, int improper,
	double angle234, double dist34);

int topo_defs_default_patching_first(topo_defs *defs, const char *pname);

int topo_defs_default_patching_last(topo_defs *defs, const char *pname);

int topo_defs_patching_first(topo_defs *defs, const char *rname,
        const char *pname);

int topo_defs_patching_last(topo_defs *defs, const char *rname,
        const char *pname);

int topo_defs_add_topofile(topo_defs *defs, const char *filename);

#endif

