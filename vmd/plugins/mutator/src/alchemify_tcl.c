#include "libalchemify.h"

/* This is exported as a Tcl Proc */
extern int alchemify(char *inPSF, char *outPSF, char *inFEP);

/* The Tcl procedure assumes the default PDB column (B) */
#define PDB_COL 'B'

int alchemify(char *inPSF, char *outPSF, char *inFEP) {

    FILE	*in, *out;	
    int	nFinal, nInitial, natoms;
    int	final[MAX_GROUP_SIZE], initial[MAX_GROUP_SIZE];	

    natoms = readPDB(inFEP, PDB_COL, initial, final, &nInitial, &nFinal);
    if (natoms < 0) DIE("problem reading FEP file") 

    printf("\nFEPfile : %i atoms found, %i initial, %i final.\n", natoms, nInitial, nFinal);

    if (!(nFinal || nInitial)) DIE("alchemify is not needed")

    if (!(nFinal && nInitial)) {
	printf("Either no atoms appearing, or no atoms disappearing.\n"
			"PSF file requires no modification.\n");
	exit(EXIT_SUCCESS);
    }

    in = fopen(inPSF, "r");
    if (!in) DIE("could not open input file")
    
    out = fopen(outPSF, "w");
    if (!out) CLOSE_AND_DIE(in, "could not open output file")
    
    if (process(in, out, natoms, initial, final, nInitial, nFinal)) {
	fclose(in);
	fclose(out);
	DIE("while processing PSF file")
    }

    fclose(in);
    fclose(out);

    return 0;
}

