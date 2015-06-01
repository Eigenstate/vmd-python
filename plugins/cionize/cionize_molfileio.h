#ifndef CIONIZE_MOLFILEIO
#define CIONIZE_MOLFILEIO

/*** Set up molfile apis being used by cionize ***/

/* Do all the work to set up the api finder and such */
void setup_apis(cionize_api*);

/* Get the right input api for our structure and make sure it works */
int setup_inapi(const cionize_params*, cionize_api*);

/*** Perform coordinate/topology file io and convert to cionize internals ***/

/* Open and read the input file into a molfile representation, 
 * and get the necessary information about it */
int read_input(cionize_params*, cionize_api*, cionize_molecule*, cionize_grid*);

/* Write the current ions to a file */
int output_ions(cionize_params*, cionize_api*, cionize_grid*, const char*, const int*);

/* Get necessary information from the mofile plugin type */
int get_coord_info(const cionize_params*, cionize_api*, cionize_grid*, cionize_molecule*);

/* Convert pertinent data from molfile_plugin types to arrays for ionize */
int unpack_molfile_arrays(const cionize_params*, cionize_api*, cionize_molecule*, cionize_grid*);


#endif

