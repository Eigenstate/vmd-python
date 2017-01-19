#include <ctype.h>
#include "libalchemify.h"

/* Private functions */
static int couples(int *p, int n, int *initial, int *final, int nInitial, int nFinal);
static int read_ints(FILE *f, int *p, int n);
static int write_ints(FILE *f, int *p, int n, int per_line);
static int skip_line(FILE *f);

int	readPDB(char *pdb, char col, int *initial, int *final, int *nInitial, int *nFinal) {

    FILE	*f;
    char	line[256];

    int		start, end;	/* range of columns in the PDB file */
    int		natoms = 0;
    float	input;
    int		end_reached = 0;
    
    *nFinal = 0;
    *nInitial = 0;

    f = fopen(pdb, "r");
    if (!f) DIE("could not open FEP file");

    switch (toupper(col)) {
	case 'X':
	    start=31; end=39;
	    break;
	case 'Y':
	    start=39; end=47;
	    break;
	case 'Z':
	    start=47; end=55;
	    break;
	case 'O':
	    start=55; end=61;
	    break;
	case 'B':
	    start=61; end=67;
	    break;
	default :
	    CLOSE_AND_DIE(f, "incorrect PDB column for alchemical flags")	
    }	

    while (!feof(f)) {
	fgets(line, 256, f);

	if (line[0]=='A'&&line[1]=='T'&&line[2]=='O'&&line[3]=='M') {
	    natoms++;
	    line[end]='\0';
	    sscanf( line + start - 1, "%f", &input);
	    if (input == 1.0) {
		if (*nFinal>=MAX_GROUP_SIZE) CLOSE_AND_DIE(f, "too many final atoms")
		final[(*nFinal)++] = natoms;
	    }
	    if (input == -1.0) {
		if (*nInitial>=MAX_GROUP_SIZE) CLOSE_AND_DIE(f, "too many initial atoms")
		initial[(*nInitial)++] = natoms;
	    }
	} else if (!strcmp(line, "END") || !strcmp(line, "END\n")) {
	    end_reached = 1;
	    break;
	}
    }

    if (!end_reached) {
	printf("WARNING: END keyword not found at the end of FEP file %s\n", pdb);
    }
    
    fclose(f);
    return natoms;
}

/************************************************************************************/
/* Does the job									    */
/************************************************************************************/
int	process(FILE *in, FILE *out, int natoms, int *initial, int *final,
		int nInitial, int nFinal) {
    int		n, i, j, removed, a, bonds;
    int		tab[4];
    char	buf[512];
    int		*p;	/* to store BONDS, ANGLES, DIHEDS etc. */
    int		wrong=0;

    if (fscanf(in, "PSF"))
	    DIE("no \"PSF\" header string found in PSF file")

    if (skip_line(in)) DIE("unexpected EOF")

    fprintf(out, "PSF\n\n");
	    
    /* read title */
    if (fscanf(in, "%i !NTITLE", &n) != 1) DIE("could not read number of title lines")
    if (skip_line(in)) DIE("unexpected EOF")
    fprintf(out, "%8i !NTITLE\n", n);

    for (i=0; i<n; i++) {
	if (!fgets(buf, 512, in)) DIE("error or EOF while reading title")
	fputs(buf, out);
    }
    if (skip_line(in)) DIE("unexpected EOF before atoms")
    fputc('\n', out);

    /* read atoms */
    if (fscanf(in, "%i !NATOM", &n) != 1) DIE("could not read number of atoms")
    if (n != natoms) DIE("incorrect number of atoms in PSF file")

    if (skip_line(in)) DIE("unexpected EOF before atoms")
    fprintf(out, "%8i !NATOM\n", n);

    for (i=0; i<n; i++) {
	if (!fgets(buf, 512, in)) DIE("error or EOF while reading atoms")
	fputs(buf, out);
    }
		
    /* number of bonds */
    if (skip_line(in)) DIE("unexpected EOF before BONDS")
    fputc('\n', out);	/* blank line */

    if (fscanf(in, "%i !NBOND", &n) != 1) DIE("could not read number of bonds")
    if (skip_line(in)) DIE("unexpected EOF before BONDS")

    if (n) {
	if (!(p=calloc(2*n, sizeof(int))))
	    DIE("memory allocation failure")
	
	/* READ */
	if (read_ints(in, p, 2 * n)) FREE_AND_DIE(p, "reading BONDS")
	if (skip_line(in)) FREE_AND_DIE(p, "unexpected EOF before ANGLES")
	
	/* WRITE */
	bonds = 0;
	for (i=0; i<n; i++)
	    if (couples(p+2*i, 2, initial, final, nInitial, nFinal)) {
		bonds++;
		wrong = 2*i;
	    }

	if (bonds) {
	    printf("WARNING : there are %i bonds coupling initial and final groups\n",  bonds);
	    printf("The BONDS section will be left unchanged.\n");
	    printf(" *** THE SETUP IS UNUSUAL, AND LIKELY TO BE WRONG, PLEASE DOUBLE-CHECK ***\n");
	    printf(" *** SEE E.G. BOND BETWEEN ATOMS %i AND %i ***\n\n", p[wrong], p[wrong+1]);
	}

	/* Special case of bonds: leave them alone - after warning the user */
	fprintf(out, "%8i !NBOND: bonds\n", n);
	for (i=0; i<n; i++)
	    write_ints(out, p+2*i, 2, 4);
	
	write_ints(NULL, NULL, 0, 0);	/* end section */
	free(p);
    } else {
	putc('\n', stdout);
	fprintf(out, "%8i !NBOND: bonds\n", 0);
    }

    fprintf(out, "\n\n");	/* blank line */


    /* angles */
    if (fscanf(in, "%i !NTHETA", &n) != 1) DIE("could not read number of angles")
    if (skip_line(in)) DIE("unexpected EOF before ANGLES")

    printf("%i angles", n);

    if (n) {
	if (!(p=calloc(3*n, sizeof(int))))
	    DIE("memory allocation failure")
	
	/* READ */
	if (read_ints(in, p, 3 * n)) FREE_AND_DIE(p, "reading ANGLES")
	if (skip_line(in)) FREE_AND_DIE(p, "unexpected EOF before ANGLES")
	
	/* WRITE */
	removed = 0;
	for (i=0; i<n; i++)
	    if (couples(p+3*i, 3, initial, final, nInitial, nFinal))
		    removed++;

	printf(" : removing %i angles coupling initial and final groups\n", removed);

	fprintf(out, "%8i !NTHETA: angles\n", n-removed);
	for (i=0; i<n; i++)
	    if (!couples(p+3*i, 3, initial, final, nInitial, nFinal)) 
		write_ints(out, p+3*i, 3, 3);

	write_ints(NULL, NULL, 0, 0);	/* end section */
	free(p);
    } else {
	putc('\n', stdout);
	fprintf(out, "%8i !NTHETA: angles\n", 0);
    }

    fprintf(out, "\n\n");	/* blank line */

    /* dihedrals */
    if (fscanf(in, "%i !NPHI", &n) != 1) DIE("could not read number of dihedrals")
    if (skip_line(in)) DIE("unexpected EOF before dih")

    printf("%i dihedrals", n);

    if (n) {
	if (!(p=calloc(4*n, sizeof(int))))
	    DIE("memory allocation failure")
	
	/* READ */
	if (read_ints(in, p, 4 * n)) FREE_AND_DIE(p, "reading DIHEDRALS")
	if (skip_line(in)) FREE_AND_DIE(p, "unexpected EOF before ANGLES")
	
	/* WRITE */
	removed = 0;
	for (i=0; i<n; i++)
	    if (couples(p+4*i, 4, initial, final, nInitial, nFinal))
		    removed++;

	printf(" : removing %i dihedrals coupling initial and final groups\n", removed);

	fprintf(out, "%8i !NPHI: dihedrals\n", n-removed);
	for (i=0; i<n; i++)
	    if (!couples(p+4*i, 4, initial, final, nInitial, nFinal)) 
		write_ints(out, p+4*i, 4, 2);

	write_ints(NULL, NULL, 0, 0);	/* end section */
	free(p);
    } else {
	putc('\n', stdout);
	fprintf(out, "%8i !NPHI: dihedrals\n", 0);
    }

    fprintf(out, "\n\n");	/* blank line */

    /* impropers */
    if (fscanf(in, "%i !NIMPHI", &n) != 1) DIE("could not read number of impropers")
    if (skip_line(in)) DIE("unexpected EOF before impropers")

    printf("%i impropers", n);

    if (n) {
	if (!(p=calloc(4*n, sizeof(int))))
	    DIE("memory allocation failure")
	
	/* READ */
	if(read_ints(in, p, 4 * n)) FREE_AND_DIE(p, "reading IMPROPERS")
	if (skip_line(in)) FREE_AND_DIE(p, "unexpected EOF before IMPR")
	
	/* WRITE */
	removed = 0;
	for (i=0; i<n; i++)
	    if (couples(p+4*i, 4, initial, final, nInitial, nFinal))
		removed++;

	printf(" : removing %i impropers coupling initial and final groups\n", removed);

	fprintf(out, "%8i !NIMPHI: impropers\n", n-removed);
	for (i=0; i<n; i++)
	    if (!couples(p+4*i, 4, initial, final, nInitial, nFinal)) 
		write_ints(out, p+4*i, 4, 2);

	write_ints(NULL, NULL, 0, 0);	/* end section */
	free(p);
    } else {
	putc('\n', stdout);
	fprintf(out, "%8i !NIMPHI: impropers\n", 0);
    }

    fprintf(out, "\n\n");	/* blank line */


    /* donors */
    if (fscanf(in, "%i !NDON", &n) != 1) DIE("could not read number of donors")
    if (skip_line(in)) DIE("unexpected EOF")

    fprintf(out, "%8i !NDON: donors\n", n);
    
    if (n) {
	for (i=0; i<n; i++) {
	    if (read_ints(in, tab, 2)) DIE("reading DONORS")
	    write_ints(out, tab, 2, 4);
	}
	write_ints(NULL, NULL, 0, 0);	/* end section */
	if (skip_line(in)) DIE("unexpected EOF")
    }
    fprintf(out, "\n\n");	/* blank line */

    /* acceptors */
    if (fscanf(in, "%i !NACC", &n) != 1) DIE("could not read number of acceptors")
    if (skip_line(in)) DIE("unexpected EOF")

    fprintf(out, "%8i !NACC: acceptors\n", n);
    
    if (n) {
	for (i=0; i<n; i++) {
	    if (read_ints(in, tab, 2)) DIE("reading ACCEPTORS")
	    write_ints(out, tab, 2, 4);
	}
	write_ints(NULL, NULL, 0, 0);	/* end section */
	if (skip_line(in)) DIE("unexpected EOF")
    }
    fprintf(out, "\n\n");	/* blank line */

    /* exclusions */
    if (fscanf(in, "%i !NNB", &n) != 1) DIE("could not read number of exclusions")
    if (skip_line(in)) DIE("unexpected EOF")

    if (n)
	printf("WARNING : %i exclusions defined in source PSF, will be ignored\n", n);
	    
    /* read and forget the lists */
    for (i=0; i<n; i++) {
	if (read_ints(in, tab, 1)) DIE("reading exclusion lists")
    }
    /* read and forget the indices */
    for (i=0; i<natoms; i++) {
	if (read_ints(in, tab, 1)) DIE("reading exclusion list indices")
    }

    printf("Writing %i exclusion pairs\n", nInitial*nFinal);
    fprintf(out, "%8i !NNB\n", nInitial*nFinal);

    /* Give every "initial" atom an exclusion list with all "final" atoms */
    for (i=0; i<nInitial; i++)
	for (j=0; j<nFinal; j++)
	    write_ints(out, final+j, 1, 8);

    write_ints(NULL, NULL, 0, 0);	/* end section */
    putc('\n', out);

    i = 0;	/* i will be the index in the exclusion lists */
    for (a=1; a<=natoms; a++) {
	/* if a is an "initial" atom, increase the index by
	 * the size of the list */
	for (j=0; j<nInitial; j++)
		if (a == initial[j])
			i += nFinal;

	write_ints(out, &i, 1, 8);
    }
	    
    write_ints(NULL, NULL, 0, 0);	/* end section */
    /* just copy the end of the file without modification */
    
    while (!feof(in)) {
	if (!fgets(buf, 512, in)) break;
	fputs(buf, out);
    }

    return 0;
}


/************************************************************************************/
/* Says whether the given parameter couples the initial and final groups	    */
/************************************************************************************/
int couples(int *p, int n, int *initial, int *final, int nInitial, int nFinal) {

    int i, j, flagI, flagF;

    flagI = flagF = 0;
    
    for (i=0; i<n; i++) {
	for (j=0; j<nInitial; j++)
	    if (p[i] == initial[j]) {
		flagI = 1;
		break;
	    }
	for (j=0; j<nFinal; j++)
	    if (p[i] == final[j]) {
		flagF = 1;
		break;
	    }
    }

    return (flagI && flagF);
}


/************************************************************************************/
/* If true, end of file has been reached					    */
/************************************************************************************/
int skip_line(FILE *f) {
    char c=' ';

    while (c!='\n' && !feof(f))	/* seek next newline */
	c=fgetc(f);

    if (feof(f)) return 1;

    while (c=='\n' && !feof(f))	/* read all the following '\n's */
	c=fgetc(f);
    
    if (feof(f)) return 1;

    ungetc(c, f);	/* the last one is not a newline */
    
    return 0;
}

/************************************************************************************/
/* Reads groups of n integers							    */
/************************************************************************************/
int read_ints(FILE *f, int *p, int n) {

    int	i;
	
    for (i=0; i<n; i++) {
	if (fscanf(f, "%i", (p+i)) != 1)
	    return 1;
    }

    return 0;
}

/************************************************************************************/
/* Writes groups of n integers (per_line groups on every line)			    */
/* call with f==NULL to reset the counter to zero				    */
/************************************************************************************/
int write_ints(FILE *f, int *p, int n, int per_line) {

    int		i;
    static int	written;	/* number of fields written in current line */

    if (!f) {		/* end of section */
	written = 0;
	return 0;
    }

    if (written >= per_line) {	/* line completed */
	fputc('\n', f);
	written = 0;
    }
    
    for (i=0; i<n; i++) {
	fprintf(f, "%8i", p[i]);
    }
    if (i) written++;	/* unless we wrote nothing */

    return 0;
}
