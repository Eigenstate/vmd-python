#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef MAC
#include <console.h>
#endif
#include "clustalw.h"

/*
*	Prototypes
*/

#ifdef MAC
extern int ccommand(char ***);
#endif

extern void *ckalloc(size_t);
extern void init_amenu(void);
extern void init_interface(void);
extern void init_matrix(void);
extern void fill_chartab(void);
extern void parse_params(Boolean);
extern void main_menu(void);

/*
*	Global variables
*/
double **tmat;

char revision_level[] = "W (1.83)";  /* JULIE  feb 2001*/

Boolean interactive=FALSE;

#ifdef MSDOS
        char *help_file_name = "clustalw.hlp";
#else
        char *help_file_name = "clustalw_help";
#endif

sint max_names; /* maximum length of names in current alignment file */

float           gap_open,      gap_extend;
float           pw_go_penalty, pw_ge_penalty;

FILE *tree;
FILE *clustal_outfile, *gcg_outfile, *nbrf_outfile, *phylip_outfile,
     *gde_outfile, *nexus_outfile;
FILE *fasta_outfile; /* Ramu */

sint  *seqlen_array;
sint max_aln_length;
short usermat[NUMRES][NUMRES], pw_usermat[NUMRES][NUMRES];
short def_aa_xref[NUMRES+1], aa_xref[NUMRES+1], pw_aa_xref[NUMRES+1];
short userdnamat[NUMRES][NUMRES], pw_userdnamat[NUMRES][NUMRES];
short def_dna_xref[NUMRES+1], dna_xref[NUMRES+1], pw_dna_xref[NUMRES+1];
sint nseqs;
sint nsets;
sint *output_index;
sint **sets;
sint *seq_weight;
sint max_aa;
sint gap_pos1;
sint gap_pos2;
sint mat_avscore;
sint profile_no;

Boolean usemenu;
Boolean dnaflag;
Boolean distance_tree;

char  **seq_array;
char **names,**titles;
char **args;
char seqname[FILENAMELEN+1];

char *gap_penalty_mask1 = NULL, *gap_penalty_mask2 = NULL;
char *sec_struct_mask1 = NULL, *sec_struct_mask2 = NULL;
sint struct_penalties;
char *ss_name1 = NULL, *ss_name2 = NULL;

Boolean user_series = FALSE;
UserMatSeries matseries;
short usermatseries[MAXMAT][NUMRES][NUMRES];
short aa_xrefseries[MAXMAT][NUMRES+1];

int main(int argc,char **argv)
{
	sint i;
	
#ifdef MAC
	argc=ccommand(&argv);
#endif

    init_amenu();
    init_interface();
    init_matrix();
	
	fill_chartab();

	if(argc>1) {
		args = (char **)ckalloc(argc * sizeof(char *));
	
		for(i=1;i<argc;++i) 
		{
			args[i-1]=(char *)ckalloc((strlen(argv[i])+1) * sizeof(char));
			strcpy(args[i-1],argv[i]);
		}
		usemenu=FALSE;
		parse_params(FALSE);

		for(i=0;i<argc-1;i++) 
			ckfree(args[i]);
		ckfree(args);
	}
	usemenu=TRUE;
	interactive=TRUE;

	main_menu();
	
	exit(0);
}

