#ifndef CIONIZE_INTERNALS
#define CIONIZE_INTERNALS

/* Calculate region that is too close to, or far from, the solute */
int exclude_solute(const cionize_params*, cionize_molecule*, cionize_grid*);

/* Place a set of ions using the current energy grid */
int place_n_ions(cionize_params*, cionize_grid*, int*);

/* Allocate memory for the arrays used in our calculations */
int alloc_mainarrays(cionize_grid*, cionize_molecule*);

/* Run through the second input loop, and carry out the requested ion placements */
int do_placements(cionize_params*, cionize_grid*, cionize_api*);

/* Initialize all the system parameters to sensible defaults  and open the input file*/
void init_defaults(cionize_params*, cionize_grid*);

#endif
