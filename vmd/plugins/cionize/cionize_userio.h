#ifndef CIONIZE_USERIO
#define CIONIZE_USERIO

/*** Functions for interacting with user input ***/

/* Parse command line arguments */
int get_opts(cionize_params*, cionize_api*, int, char**);

/* Write usage information */
void print_usage(void);

/* Open the input file for reading */
int open_input(cionize_params*);

/* Go through our initial input loop, set all the parameters, and get ready
 * for the run */
int settings_inputloop(cionize_params*, cionize_grid*);

/* Get the proper energy calculation method */
int get_ener_method_from_string(const char *str, int *emethod);

#endif

