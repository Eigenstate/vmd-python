/* Menus and command line interface for Clustal W  */
/* DES was here MARCH. 1994 */
/* DES was here SEPT.  1994 */
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdarg.h>
#include <signal.h>
#include <setjmp.h>
#include "clustalw.h"

static jmp_buf jmpbuf;
#ifndef VMS
#ifndef AIX
#ifndef BADSIG
#define BADSIG (void (*)())-1
#endif
#endif
#endif

/* static void jumpFctPtr(int); */

static void jumpFctPtr(int i)
{
   longjmp(jmpbuf,1);
}


/*
*	Prototypes
*/


static void pair_menu(void);
static void multi_menu(void);
static void gap_penalties_menu(void);
static void multiple_align_menu(void);          /* multiple alignments menu */
static void profile_align_menu(void);           /* profile       "      "   */
static void phylogenetic_tree_menu(void);       /* NJ trees/distances menu  */
static void format_options_menu(void);          /* format of alignment output */
static void tree_format_options_menu(void);     /* format of tree output */
static void ss_options_menu(void);
static sint secstroutput_options(void);
static sint read_matrix(char *title,MatMenu menu, char *matnam, sint matn, short *mat, short *xref);

/*
*	 Global variables
*/

extern float    gap_open,      gap_extend;
extern float  	dna_gap_open,  dna_gap_extend;
extern float 	prot_gap_open, prot_gap_extend;
extern float    pw_go_penalty,      pw_ge_penalty;
extern float  	dna_pw_go_penalty,  dna_pw_ge_penalty;
extern float 	prot_pw_go_penalty, prot_pw_ge_penalty;
extern float	transition_weight;
extern char 	revision_level[];
extern sint    wind_gap,ktup,window,signif;
extern sint    dna_wind_gap, dna_ktup, dna_window, dna_signif;
extern sint    prot_wind_gap,prot_ktup,prot_window,prot_signif;
extern sint	nseqs;
extern sint 	divergence_cutoff;
extern sint 	debug;
extern Boolean 	neg_matrix;
extern Boolean  quick_pairalign;
extern Boolean	reset_alignments_new;		/* DES */
extern Boolean	reset_alignments_all;		/* DES */
extern sint 	gap_dist;
extern Boolean 	no_var_penalties, no_hyd_penalties, no_pref_penalties;
extern sint 	output_order;
extern sint profile_no;
extern short 	usermat[], pw_usermat[];
extern short 	aa_xref[], pw_aa_xref[];
extern short 	userdnamat[], pw_userdnamat[];
extern short 	dna_xref[], pw_dna_xref[];

extern Boolean 	lowercase; /* Flag for GDE output - set on comm. line*/
extern Boolean 	cl_seq_numbers;
extern Boolean seqRange;  /* to append sequence range with seq names, Ranu */

extern Boolean 	output_clustal, output_nbrf, output_phylip, output_gcg, output_gde, output_nexus;
extern Boolean output_fasta; /* Ramu */

extern Boolean 	output_tree_clustal, output_tree_phylip, output_tree_distances,output_tree_nexus;
extern sint     bootstrap_format;
extern Boolean 	tossgaps, kimura;
extern Boolean  percent;
extern Boolean 	usemenu;
extern Boolean 	showaln, save_parameters;
extern Boolean	dnaflag;
extern Boolean  use_ambiguities;


extern char 	hyd_residues[];
extern char 	mtrxname[], pw_mtrxname[];
extern char 	dnamtrxname[], pw_dnamtrxname[];
extern char	seqname[];

extern sint output_struct_penalties;
extern Boolean use_ss1, use_ss2;

extern Boolean empty;
extern Boolean profile1_empty, profile2_empty;   /* whether or not profiles   */

extern char  	profile1_name[FILENAMELEN+1];
extern char  	profile2_name[FILENAMELEN+1];

extern Boolean         use_endgaps;
extern sint        matnum,pw_matnum;
extern sint        dnamatnum,pw_dnamatnum;

extern sint        helix_penalty;
extern sint        strand_penalty;
extern sint        loop_penalty;
extern sint        helix_end_minus;
extern sint        helix_end_plus;
extern sint        strand_end_minus;
extern sint        strand_end_plus;
extern sint        helix_end_penalty;
extern sint        strand_end_penalty;

extern MatMenu matrix_menu;
extern MatMenu pw_matrix_menu;
extern MatMenu dnamatrix_menu;

static char phylip_name[FILENAMELEN]="";
static char clustal_name[FILENAMELEN]="";
static char dist_name[FILENAMELEN]="";
static char nexus_name[FILENAMELEN]="";
/* static char fasta_name[FILENAMELEN]=""; */

static char p1_tree_name[FILENAMELEN]="";
static char p2_tree_name[FILENAMELEN]="";

static char *secstroutput_txt[] = {
				"Secondary Structure",
				"Gap Penalty Mask",
				"Structure and Penalty Mask",
				"None"	};
				                

static char *lin1, *lin2, *lin3;

/* static int firstres =0;	*//* range of alignment for saving as ... */
/* static int lastres = 0; */

void init_amenu(void)
{

	lin1 = (char *)ckalloc( (MAXLINE+1) * sizeof (char) );
	lin2 = (char *)ckalloc( (MAXLINE+1) * sizeof (char) );
	lin3 = (char *)ckalloc( (MAXLINE+1) * sizeof (char) );
}

void main_menu(void)
{
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }

	while(TRUE) {
		fprintf(stdout,"\n\n\n");
		fprintf(stdout," **************************************************************\n");
		fprintf(stdout," ******** CLUSTAL %s Multiple Sequence Alignments  ********\n",revision_level);
		fprintf(stdout," **************************************************************\n");
		fprintf(stdout,"\n\n");
		
		fprintf(stdout,"     1. Sequence Input From Disc\n");
		fprintf(stdout,"     2. Multiple Alignments\n");
		fprintf(stdout,"     3. Profile / Structure Alignments\n");
		fprintf(stdout,"     4. Phylogenetic trees\n");
		fprintf(stdout,"\n");
		fprintf(stdout,"     S. Execute a system command\n");
		fprintf(stdout,"     H. HELP\n");
		fprintf(stdout,"     X. EXIT (leave program)\n\n\n");
		
		getstr("Your choice",lin1, MAXLINE);

		switch(toupper(*lin1)) {
			case '1': seq_input(FALSE);
				phylip_name[0]=EOS;
				clustal_name[0]=EOS;
				dist_name[0]=EOS;
				nexus_name[0]=EOS;
				break;
			case '2': multiple_align_menu();
				break;
			case '3': profile_align_menu();
				break;
			case '4': phylogenetic_tree_menu();
				break;
			case 'S': do_system();
				break;
			case '?':
			case 'H': get_help('1');
				break;
			case 'Q':
			case 'X': exit(0);
				break;
			default: fprintf(stdout,"\n\nUnrecognised Command\n\n");
				break;
		}
	}
}









static void multiple_align_menu(void)
{
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


    while(TRUE)
    {
        fprintf(stdout,"\n\n\n");
        fprintf(stdout,"****** MULTIPLE ALIGNMENT MENU ******\n");
        fprintf(stdout,"\n\n");


        fprintf(stdout,"    1.  Do complete multiple alignment now (%s)\n",
                        (!quick_pairalign) ? "Slow/Accurate" : "Fast/Approximate");
        fprintf(stdout,"    2.  Produce guide tree file only\n");
        fprintf(stdout,"    3.  Do alignment using old guide tree file\n\n");
        fprintf(stdout,"    4.  Toggle Slow/Fast pairwise alignments = %s\n\n",
                                        (!quick_pairalign) ? "SLOW" : "FAST");
        fprintf(stdout,"    5.  Pairwise alignment parameters\n");
        fprintf(stdout,"    6.  Multiple alignment parameters\n\n");
	fprintf(stdout,"    7.  Reset gaps before alignment?");
	if(reset_alignments_new)
		fprintf(stdout," = ON\n");
	else
		fprintf(stdout," = OFF\n");
        fprintf(stdout,"    8.  Toggle screen display          = %s\n",
                                        (!showaln) ? "OFF" : "ON");
        fprintf(stdout,"    9.  Output format options\n");
        fprintf(stdout,"\n");

        fprintf(stdout,"    S.  Execute a system command\n");
        fprintf(stdout,"    H.  HELP\n");
        fprintf(stdout,"    or press [RETURN] to go back to main menu\n\n\n");

        getstr("Your choice",lin1, MAXLINE);
        if(*lin1 == EOS) return;

        switch(toupper(*lin1))
        {
        case '1': align(phylip_name);
            break;
        case '2': make_tree(phylip_name);
            break;
        case '3': get_tree(phylip_name);
            break;
        case '4': quick_pairalign ^= TRUE;
            break;
        case '5': pair_menu();
            break;
        case '6': multi_menu();
            break;
	case '7': reset_alignments_new ^= TRUE;
	    if(reset_alignments_new==TRUE)
		reset_alignments_all=FALSE;
            break;
        case '8': showaln ^= TRUE;
	    break;
        case '9': format_options_menu();
            break;
        case 'S': do_system();
            break;
        case '?':
        case 'H': get_help('2');
            break;
        case 'Q':
        case 'X': return;

        default: fprintf(stdout,"\n\nUnrecognised Command\n\n");
            break;
        }
    }
}









static void profile_align_menu(void)
{
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


    while(TRUE)
    {
	fprintf(stdout,"\n\n\n");
        fprintf(stdout,"****** PROFILE AND STRUCTURE ALIGNMENT MENU ******\n");
        fprintf(stdout,"\n\n");

        fprintf(stdout,"    1.  Input 1st. profile             ");
        if (!profile1_empty) fprintf(stdout,"(loaded)");
        fprintf(stdout,"\n");
        fprintf(stdout,"    2.  Input 2nd. profile/sequences   ");
        if (!profile2_empty) fprintf(stdout,"(loaded)");
        fprintf(stdout,"\n\n");
        fprintf(stdout,"    3.  Align 2nd. profile to 1st. profile\n");
        fprintf(stdout,"    4.  Align sequences to 1st. profile (%s)\n\n",
                        (!quick_pairalign) ? "Slow/Accurate" : "Fast/Approximate");
        fprintf(stdout,"    5.  Toggle Slow/Fast pairwise alignments = %s\n\n",
                                        (!quick_pairalign) ? "SLOW" : "FAST");
        fprintf(stdout,"    6.  Pairwise alignment parameters\n");
        fprintf(stdout,"    7.  Multiple alignment parameters\n\n");
        fprintf(stdout,"    8.  Toggle screen display                = %s\n",
                                        (!showaln) ? "OFF" : "ON");
        fprintf(stdout,"    9.  Output format options\n");
        fprintf(stdout,"    0.  Secondary structure options\n");
        fprintf(stdout,"\n");
        fprintf(stdout,"    S.  Execute a system command\n");
        fprintf(stdout,"    H.  HELP\n");
        fprintf(stdout,"    or press [RETURN] to go back to main menu\n\n\n");

        getstr("Your choice",lin1, MAXLINE);
        if(*lin1 == EOS) return;

        switch(toupper(*lin1))
        {
        case '1': profile_no = 1;      /* 1 => 1st profile */ 
          profile_input();
		  strcpy(profile1_name, seqname);
            break;
        case '2': profile_no = 2;      /* 2 => 2nd profile */
          profile_input();
		  strcpy(profile2_name, seqname);
            break;
        case '3': profile_align(p1_tree_name,p2_tree_name);       /* align the 2 alignments now */
            break;
        case '4': new_sequence_align(phylip_name);  /* align new sequences to profile 1 */
            break;
        case '5': quick_pairalign ^= TRUE;
	    break;
        case '6': pair_menu();
            break;
        case '7': multi_menu();
            break;
        case '8': showaln ^= TRUE;
	    break;
        case '9': format_options_menu();
            break;
        case '0': ss_options_menu();
            break;
        case 'S': do_system();
            break;
        case '?':
        case 'H': get_help('6');
            break;
        case 'Q':
        case 'X': return;

        default: fprintf(stdout,"\n\nUnrecognised Command\n\n");
            break;
        }
    }
}


static void ss_options_menu(void)
{
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


	while(TRUE) {
	
		fprintf(stdout,"\n\n\n");
		fprintf(stdout," ********* SECONDARY STRUCTURE OPTIONS *********\n");
		fprintf(stdout,"\n\n");

		fprintf(stdout,"     1. Use profile 1 secondary structure / penalty mask  ");
		if(use_ss1)
			fprintf(stdout,"= YES\n");
		else
			fprintf(stdout,"= NO\n");
		fprintf(stdout,"     2. Use profile 2 secondary structure / penalty mask  ");
		if(use_ss2)
			fprintf(stdout,"= YES\n");
		else
			fprintf(stdout,"= NO\n");
		fprintf(stdout,"\n");
		fprintf(stdout,"     3. Output in alignment  ");
		fprintf(stdout,"= %s\n",secstroutput_txt[output_struct_penalties]);
		fprintf(stdout,"\n");

		fprintf(stdout,"     4. Helix gap penalty                     :%d\n",(pint)helix_penalty);
		fprintf(stdout,"     5. Strand gap penalty                    :%d\n",(pint)strand_penalty);
		fprintf(stdout,"     6. Loop gap penalty                      :%d\n",(pint)loop_penalty);

		fprintf(stdout,"     7. Secondary structure terminal penalty  :%d\n",(pint)helix_end_penalty);
		fprintf(stdout,"     8. Helix terminal positions       within :%d      outside :%d\n",
		                                 (pint)helix_end_minus,(pint)helix_end_plus);
		fprintf(stdout,"     9. Strand terminal positions      within :%d      outside :%d\n",
		                                 (pint)strand_end_minus,(pint)strand_end_plus);

		fprintf(stdout,"\n\n");
		fprintf(stdout,"     H. HELP\n\n\n");
		
		getstr("Enter number (or [RETURN] to exit)",lin2, MAXLINE);
		if( *lin2 == EOS) { 
			return;
		}
		
		switch(toupper(*lin2)) {
			case '1': use_ss1 ^= TRUE;
				break;
			case '2': use_ss2 ^= TRUE;
				break;
			case '3': output_struct_penalties = secstroutput_options();
				break;
			case '4':
				fprintf(stdout,"Helix Penalty Currently: %d\n",(pint)helix_penalty);
				helix_penalty=getint("Enter number",1,9,helix_penalty);
				break;
			case '5':
				fprintf(stdout,"Strand Gap Penalty Currently: %d\n",(pint)strand_penalty);
				strand_penalty=getint("Enter number",1,9,strand_penalty);
				break;
			case '6':
				fprintf(stdout,"Loop Gap Penalty Currently: %d\n",(pint)loop_penalty);
				loop_penalty=getint("Enter number",1,9,loop_penalty);
				break;
			case '7':
				fprintf(stdout,"Secondary Structure Terminal Penalty Currently: %d\n",
				          (pint)helix_end_penalty);
				helix_end_penalty=getint("Enter number",1,9,helix_end_penalty);
				strand_end_penalty = helix_end_penalty;
				break;
			case '8':
				fprintf(stdout,"Helix Terminal Positions Currently: \n");
				fprintf(stdout,"        within helix: %d     outside helix: %d\n",
				                            (pint)helix_end_minus,(pint)helix_end_plus);
				helix_end_minus=getint("Enter number of residues within helix",0,3,helix_end_minus);
				helix_end_plus=getint("Enter number of residues outside helix",0,3,helix_end_plus);
				break;
			case '9':
				fprintf(stdout,"Strand Terminal Positions Currently: \n");
				fprintf(stdout,"        within strand: %d     outside strand: %d\n",
				                            (pint)strand_end_minus,(pint)strand_end_plus);
				strand_end_minus=getint("Enter number of residues within strand",0,3,strand_end_minus);
				strand_end_plus=getint("Enter number of residues outside strand",0,3,strand_end_plus);
				break;
			case '?':
			case 'H':
				get_help('B');
				break;
			default:
				fprintf(stdout,"\n\nUnrecognised Command\n\n");
				break;
		}
	}
}


static sint secstroutput_options(void)
{

        while(TRUE)
        {
                fprintf(stdout,"\n\n\n");
                fprintf(stdout," ********* Secondary Structure Output Menu *********\n");
                fprintf(stdout,"\n\n");


                fprintf(stdout,"     1. %s\n",secstroutput_txt[0]);
                fprintf(stdout,"     2. %s\n",secstroutput_txt[1]);
                fprintf(stdout,"     3. %s\n",secstroutput_txt[2]);
                fprintf(stdout,"     4. %s\n",secstroutput_txt[3]);
                fprintf(stdout,"     H. HELP\n\n");
                fprintf(stdout,
"     -- Current output is %s ",secstroutput_txt[output_struct_penalties]);
                fprintf(stdout,"--\n");


                getstr("\n\nEnter number (or [RETURN] to exit)",lin2, MAXLINE);
                if(*lin2 == EOS) return(output_struct_penalties);

        	switch(toupper(*lin2))
        	{
       	 		case '1': return(0);
        		case '2': return(1);
      			case '3': return(2);
        		case '4': return(3);
			case '?': 
        		case 'H': get_help('C');
            		case 'Q':
        		case 'X': return(0);

        		default: fprintf(stdout,"\n\nUnrecognised Command\n\n");
            		break;
        	}
        }
}







static void phylogenetic_tree_menu(void)
{
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


    while(TRUE)
    {
        fprintf(stdout,"\n\n\n");
        fprintf(stdout,"****** PHYLOGENETIC TREE MENU ******\n");
        fprintf(stdout,"\n\n");

        fprintf(stdout,"    1.  Input an alignment\n");
        fprintf(stdout,"    2.  Exclude positions with gaps?        ");
	if(tossgaps)
		fprintf(stdout,"= ON\n");
	else
		fprintf(stdout,"= OFF\n");
        fprintf(stdout,"    3.  Correct for multiple substitutions? ");
	if(kimura)
		fprintf(stdout,"= ON\n");
	else
		fprintf(stdout,"= OFF\n");
        fprintf(stdout,"    4.  Draw tree now\n");
        fprintf(stdout,"    5.  Bootstrap tree\n");
	fprintf(stdout,"    6.  Output format options\n");
        fprintf(stdout,"\n");
        fprintf(stdout,"    S.  Execute a system command\n");
        fprintf(stdout,"    H.  HELP\n");
        fprintf(stdout,"    or press [RETURN] to go back to main menu\n\n\n");

        getstr("Your choice",lin1, MAXLINE);
        if(*lin1 == EOS) return;

        switch(toupper(*lin1))
        {
       	 	case '1': seq_input(FALSE);
				phylip_name[0]=EOS;
				clustal_name[0]=EOS;
				dist_name[0]=EOS;
				nexus_name[0]=EOS;
         	   	break;
        	case '2': tossgaps ^= TRUE;
          	  	break;
      		case '3': kimura ^= TRUE;;
            		break;
        	case '4': phylogenetic_tree(phylip_name,clustal_name,dist_name,nexus_name,"amenu.pim");
            		break;
        	case '5': bootstrap_tree(phylip_name,clustal_name,nexus_name);
            		break;
		case '6': tree_format_options_menu();
			break;
        	case 'S': do_system();
            		break;
            	case '?':
        	case 'H': get_help('7');
            		break;
            	case 'Q':
        	case 'X': return;

        	default: fprintf(stdout,"\n\nUnrecognised Command\n\n");
            	break;
        }
    }
}






static void tree_format_options_menu(void)      /* format of tree output */
{	
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


	while(TRUE) {
	fprintf(stdout,"\n\n\n");
	fprintf(stdout," ****** Format of Phylogenetic Tree Output ******\n");
	fprintf(stdout,"\n\n");
	fprintf(stdout,"     1. Toggle CLUSTAL format tree output    =  %s\n",
					(!output_tree_clustal)  ? "OFF" : "ON");
	fprintf(stdout,"     2. Toggle Phylip format tree output     =  %s\n",
					(!output_tree_phylip)   ? "OFF" : "ON");
	fprintf(stdout,"     3. Toggle Phylip distance matrix output =  %s\n",
					(!output_tree_distances)? "OFF" : "ON");
	fprintf(stdout,"     4. Toggle Nexus format tree output      =  %s\n\n",
					(!output_tree_nexus)? "OFF" : "ON");
	fprintf(stdout,"     5. Toggle Phylip bootstrap positions    =  %s\n\n",
(bootstrap_format==BS_NODE_LABELS) ? "NODE LABELS" : "BRANCH LABELS");
	fprintf(stdout,"\n");
	fprintf(stdout,"     H. HELP\n\n\n");	
	
		getstr("Enter number (or [RETURN] to exit)",lin2, MAXLINE);
		if(*lin2 == EOS) return;
		
		switch(toupper(*lin2)) {
			case '1':
				output_tree_clustal   ^= TRUE;
				break;
			case '2':
              			output_tree_phylip    ^= TRUE;
			  	break;
			case '3':
              			output_tree_distances ^= TRUE;
			  	break;
			case '4':
              			output_tree_nexus ^= TRUE;
			  	break;
			case '5':
              			if (bootstrap_format == BS_NODE_LABELS)
					bootstrap_format = BS_BRANCH_LABELS;
				else
					bootstrap_format = BS_NODE_LABELS;
			  	break;
			case '?':
			case 'H':
				get_help('0');
				break;
			default:
				fprintf(stdout,"\n\nUnrecognised Command\n\n");
				break;
		}
	}
}


static void format_options_menu(void)      /* format of alignment output */
{	
/*	sint i; */
/*	sint length = 0; */
	char path[FILENAMELEN+1];
    int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


	while(TRUE) {
	fprintf(stdout,"\n\n\n");
	fprintf(stdout," ********* Format of Alignment Output *********\n");
	fprintf(stdout,"\n\n");
	fprintf(stdout,"     F. Toggle FASTA format output       =  %s\n\n",
					(!output_fasta) ? "OFF" : "ON");
	fprintf(stdout,"     1. Toggle CLUSTAL format output     =  %s\n",
					(!output_clustal) ? "OFF" : "ON");
	fprintf(stdout,"     2. Toggle NBRF/PIR format output    =  %s\n",
					(!output_nbrf) ? "OFF" : "ON");
	fprintf(stdout,"     3. Toggle GCG/MSF format output     =  %s\n",
					(!output_gcg) ? "OFF" : "ON");
	fprintf(stdout,"     4. Toggle PHYLIP format output      =  %s\n",
					(!output_phylip) ? "OFF" : "ON");
	fprintf(stdout,"     5. Toggle NEXUS format output       =  %s\n",
					(!output_nexus) ? "OFF" : "ON");
	fprintf(stdout,"     6. Toggle GDE format output         =  %s\n\n",
					(!output_gde) ? "OFF" : "ON");
	fprintf(stdout,"     7. Toggle GDE output case           =  %s\n",
					(!lowercase) ? "UPPER" : "LOWER");

	fprintf(stdout,"     8. Toggle CLUSTALW sequence numbers =  %s\n",
					(!cl_seq_numbers) ? "OFF" : "ON");
	fprintf(stdout,"     9. Toggle output order              =  %s\n\n",
					(output_order==0) ? "INPUT FILE" : "ALIGNED");

	fprintf(stdout,"     0. Create alignment output file(s) now?\n\n");
	fprintf(stdout,"     T. Toggle parameter output          = %s\n",
					(!save_parameters) ? "OFF" : "ON");
	fprintf(stdout,"     R. Toggle sequence range numbers =  %s\n",
					(!seqRange) ? "OFF" : "ON");
	fprintf(stdout,"\n");
	fprintf(stdout,"     H. HELP\n\n\n");	
	
		getstr("Enter number (or [RETURN] to exit)",lin2, MAXLINE);
		if(*lin2 == EOS) return;
		
		switch(toupper(*lin2)) {
			case '1':
				output_clustal ^= TRUE;
				break;
			case '2':
              			output_nbrf ^= TRUE;
			  	break;
			case '3':
              			output_gcg ^= TRUE;
			  	break;
			case '4':
              			output_phylip ^= TRUE;
			  	break;
			case '5':
              			output_nexus ^= TRUE;
			  	break;
			case '6':
              			output_gde ^= TRUE;
			  	break;
			case '7':
              			lowercase ^= TRUE;
			  	break;
			case '8':
              			cl_seq_numbers ^= TRUE;
			  	break;
			case '9':
                                if (output_order == INPUT) output_order = ALIGNED;
              			else output_order = INPUT;
			  	break;
			case 'F':
              			output_fasta ^= TRUE;
			  	break;
			case 'R':
              			seqRange ^= TRUE;
			  	break;

			case '0':		/* DES */
				if(empty) {
					error("No sequences loaded");
					break;
				}
				get_path(seqname,path);
				if(!open_alignment_output(path)) break;
				create_alignment_output(1,nseqs);
				break;
        		case 'T': save_parameters ^= TRUE;
	   			 break;
			case '?':
			case 'H':
				get_help('5');
				break;
			default:
				fprintf(stdout,"\n\nUnrecognised Command\n\n");
				break;
		}
	}
}












static void pair_menu(void)
{
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


        if(dnaflag) {
                pw_go_penalty     = dna_pw_go_penalty;
                pw_ge_penalty     = dna_pw_ge_penalty;
                ktup       = dna_ktup;
                window     = dna_window;
                signif     = dna_signif;
                wind_gap   = dna_wind_gap;

        }
        else {
                pw_go_penalty     = prot_pw_go_penalty;
                pw_ge_penalty     = prot_pw_ge_penalty;
                ktup       = prot_ktup;
                window     = prot_window;
                signif     = prot_signif;
                wind_gap   = prot_wind_gap;

        }

	while(TRUE) {
	
		fprintf(stdout,"\n\n\n");
		fprintf(stdout," ********* PAIRWISE ALIGNMENT PARAMETERS *********\n");
		fprintf(stdout,"\n\n");

		fprintf(stdout,"     Slow/Accurate alignments:\n\n");

		fprintf(stdout,"     1. Gap Open Penalty       :%4.2f\n",pw_go_penalty);
		fprintf(stdout,"     2. Gap Extension Penalty  :%4.2f\n",pw_ge_penalty);
		fprintf(stdout,"     3. Protein weight matrix  :%s\n" ,
                                        matrix_menu.opt[pw_matnum-1].title);
		fprintf(stdout,"     4. DNA weight matrix      :%s\n" ,
                                        dnamatrix_menu.opt[pw_dnamatnum-1].title);
		fprintf(stdout,"\n");

		fprintf(stdout,"     Fast/Approximate alignments:\n\n");

		fprintf(stdout,"     5. Gap penalty            :%d\n",(pint)wind_gap);
		fprintf(stdout,"     6. K-tuple (word) size    :%d\n",(pint)ktup);
		fprintf(stdout,"     7. No. of top diagonals   :%d\n",(pint)signif);
		fprintf(stdout,"     8. Window size            :%d\n\n",(pint)window);

                fprintf(stdout,"     9. Toggle Slow/Fast pairwise alignments ");
                if(quick_pairalign)
                      fprintf(stdout,"= FAST\n\n");
                else
                      fprintf(stdout,"= SLOW\n\n");


		fprintf(stdout,"     H. HELP\n\n\n");
		
		getstr("Enter number (or [RETURN] to exit)",lin2, MAXLINE);
		if( *lin2 == EOS) {
                        if(dnaflag) {
                                dna_pw_go_penalty     = pw_go_penalty;
                                dna_pw_ge_penalty     = pw_ge_penalty;
                		dna_ktup       = ktup;
                		dna_window     = window;
                		dna_signif     = signif;
                		dna_wind_gap   = wind_gap;

                        }
                        else {
                                prot_pw_go_penalty     = pw_go_penalty;
                                prot_pw_ge_penalty     = pw_ge_penalty;
                		prot_ktup       = ktup;
                		prot_window     = window;
                		prot_signif     = signif;
                		prot_wind_gap   = wind_gap;

                        }
 
			return;
		}
		
		switch(toupper(*lin2)) {
			case '1':
				fprintf(stdout,"Gap Open Penalty Currently: %4.2f\n",pw_go_penalty);
				pw_go_penalty=(float)getreal("Enter number",(double)0.0,(double)100.0,(double)pw_go_penalty);
				break;
			case '2':
				fprintf(stdout,"Gap Extension Penalty Currently: %4.2f\n",pw_ge_penalty);
				pw_ge_penalty=(float)getreal("Enter number",(double)0.0,(double)10.0,(double)pw_ge_penalty);
				break;
                        case '3':
                                pw_matnum = read_matrix("PROTEIN",pw_matrix_menu,pw_mtrxname,pw_matnum,pw_usermat,pw_aa_xref);
                                break;
                        case '4':
                                pw_dnamatnum = read_matrix("DNA",dnamatrix_menu,pw_dnamtrxname,pw_dnamatnum,pw_userdnamat,pw_dna_xref);
                                break;
			case '5':
                                fprintf(stdout,"Gap Penalty Currently: %d\n",(pint)wind_gap);
                                wind_gap=getint("Enter number",1,500,wind_gap);
				break;
			case '6':
                                fprintf(stdout,"K-tuple Currently: %d\n",(pint)ktup);
                                if(dnaflag)
                                     ktup=getint("Enter number",1,4,ktup);
                                else
                                     ktup=getint("Enter number",1,2,ktup);                                     
				break;
			case '7':
                                fprintf(stdout,"Top diagonals Currently: %d\n",(pint)signif);
                                signif=getint("Enter number",1,50,signif);
				break;
			case '8':
                                fprintf(stdout,"Window size Currently: %d\n",(pint)window);
                                window=getint("Enter number",1,50,window);
				break;
                        case '9': quick_pairalign ^= TRUE;
                                break;
			case '?':
			case 'H':
				get_help('3');
				break;
			default:
				fprintf(stdout,"\n\nUnrecognised Command\n\n");
				break;
		}
	}
}





static void multi_menu(void)
{
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


	if(dnaflag) {
		gap_open   = dna_gap_open;
		gap_extend = dna_gap_extend;
	}
	else {
		gap_open   = prot_gap_open;
		gap_extend = prot_gap_extend;
	}

	while(TRUE) {

		fprintf(stdout,"\n\n\n");
		fprintf(stdout," ********* MULTIPLE ALIGNMENT PARAMETERS *********\n");
		fprintf(stdout,"\n\n");
		
		fprintf(stdout,"     1. Gap Opening Penalty              :%4.2f\n",gap_open);
		fprintf(stdout,"     2. Gap Extension Penalty            :%4.2f\n",gap_extend);

		fprintf(stdout,"     3. Delay divergent sequences        :%d %%\n\n",(pint)divergence_cutoff);

                fprintf(stdout,"     4. DNA Transitions Weight           :%1.2f\n\n",transition_weight);
                fprintf(stdout,"     5. Protein weight matrix            :%s\n"
                                        	,matrix_menu.opt[matnum-1].title);
                fprintf(stdout,"     6. DNA weight matrix                :%s\n"
                                        	,dnamatrix_menu.opt[dnamatnum-1].title);
		fprintf(stdout,"     7. Use negative matrix              :%s\n\n",(!neg_matrix) ? "OFF" : "ON");
                fprintf(stdout,"     8. Protein Gap Parameters\n\n");
		fprintf(stdout,"     H. HELP\n\n\n");		

		getstr("Enter number (or [RETURN] to exit)",lin2, MAXLINE);

		if(*lin2 == EOS) {
			if(dnaflag) {
				dna_gap_open    = gap_open;
				dna_gap_extend  = gap_extend;
			}
			else {
				prot_gap_open   = gap_open;
				prot_gap_extend = gap_extend;
			}
			return;
		}
		
		switch(toupper(*lin2)) {
			case '1':
			fprintf(stdout,"Gap Opening Penalty Currently: %4.2f\n",gap_open);
				gap_open=(float)getreal("Enter number",(double)0.0,(double)100.0,(double)gap_open);
				break;
			case '2':
				fprintf(stdout,"Gap Extension Penalty Currently: %4.2f\n",gap_extend);
				gap_extend=(float)getreal("Enter number",(double)0.0,(double)10.0,(double)gap_extend);
				break;
			case '3':
				fprintf(stdout,"Min Identity Currently: %d\n",(pint)divergence_cutoff);
				divergence_cutoff=getint("Enter number",0,100,divergence_cutoff);
				break;
			case '4':
				fprintf(stdout,"Transition Weight Currently: %1.2f\n",transition_weight);
				transition_weight=(float)getreal("Enter number",(double)0.0,(double)1.0,(double)transition_weight);
				break;
			case '5':
                                matnum = read_matrix("PROTEIN",matrix_menu,mtrxname,matnum,usermat,aa_xref);
				break;
			case '6':
                                dnamatnum = read_matrix("DNA",dnamatrix_menu,dnamtrxname,dnamatnum,userdnamat,dna_xref);
				break;
			case '7':
				neg_matrix ^= TRUE;
				break;
			case '8':
                                gap_penalties_menu();
				break;
			case '?':
			case 'H':
				get_help('4');
				break;
			default:
				fprintf(stdout,"\n\nUnrecognised Command\n\n");
				break;
		}
	}
}






static void gap_penalties_menu(void)
{
	char c;
	sint i;
        int catchint;

        catchint = signal(SIGINT, SIG_IGN) != SIG_IGN;
        if (catchint) {
                if (setjmp(jmpbuf) != 0)
                        fprintf(stdout,"\n.. Interrupt\n");
#ifdef UNIX
                if (signal(SIGINT,jumpFctPtr) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#else
                if (signal(SIGINT,SIG_DFL) == BADSIG)
                        fprintf(stdout,"Error: signal\n");
#endif
        }


	while(TRUE) {

		fprintf(stdout,"\n\n\n");
		fprintf(stdout," ********* PROTEIN GAP PARAMETERS *********\n");
		fprintf(stdout,"\n\n\n");

		fprintf(stdout,"     1. Toggle Residue-Specific Penalties :%s\n\n",(no_pref_penalties) ? "OFF" : "ON");
		fprintf(stdout,"     2. Toggle Hydrophilic Penalties      :%s\n",(no_hyd_penalties) ? "OFF" : "ON");
		fprintf(stdout,"     3. Hydrophilic Residues              :%s\n\n"
					,hyd_residues);
		fprintf(stdout,"     4. Gap Separation Distance           :%d\n",(pint)gap_dist);
		fprintf(stdout,"     5. Toggle End Gap Separation         :%s\n\n",(!use_endgaps) ? "OFF" : "ON");
		fprintf(stdout,"     H. HELP\n\n\n");		

		getstr("Enter number (or [RETURN] to exit)",lin2, MAXLINE);

		if(*lin2 == EOS) return;
		
		switch(toupper(*lin2)) {
			case '1':
				no_pref_penalties ^= TRUE;
				break;
			case '2':
				no_hyd_penalties ^= TRUE;
				break;
			case '3':
				fprintf(stdout,"Hydrophilic Residues Currently: %s\n",hyd_residues);

				getstr("Enter residues (or [RETURN] to quit)",lin1, MAXLINE);
                                if (*lin1 != EOS) {
                                        for (i=0;i<strlen(hyd_residues) && i<26;i++) {
                                        c = lin1[i];
                                        if (isalpha(c))
                                                hyd_residues[i] = (char)toupper(c);
                                        else
                                                break;
                                        }
                                        hyd_residues[i] = EOS;
                                }
                                break;
			case '4':
				fprintf(stdout,"Gap Separation Distance Currently: %d\n",(pint)gap_dist);
				gap_dist=getint("Enter number",0,100,gap_dist);
				break;
			case '5':
				use_endgaps ^= TRUE;
				break;
			case '?':
			case 'H':
				get_help('A');
				break;
			default:
				fprintf(stdout,"\n\nUnrecognised Command\n\n");
				break;
		}
	}
}




static sint read_matrix(char *title,MatMenu menu, char *matnam, sint matn, short *mat, short *xref)
{       static char userfile[FILENAMELEN+1];
	int i;

        while(TRUE)
        {
                fprintf(stdout,"\n\n\n");
                fprintf(stdout," ********* %s WEIGHT MATRIX MENU *********\n",title);
                fprintf(stdout,"\n\n");

		for(i=0;i<menu.noptions;i++)
                	fprintf(stdout,"     %d. %s\n",i+1,menu.opt[i].title);
                fprintf(stdout,"     H. HELP\n\n");
                fprintf(stdout,
"     -- Current matrix is the %s ",menu.opt[matn-1].title);
                if(matn == menu.noptions) fprintf(stdout,"(file = %s)",userfile);
                fprintf(stdout,"--\n");


                getstr("\n\nEnter number (or [RETURN] to exit)",lin2, MAXLINE);
                if(*lin2 == EOS) return(matn);

                i=toupper(*lin2)-'0';
		if(i>0 && i<menu.noptions) {
                        strcpy(matnam,menu.opt[i-1].string);
                        matn=i;
		} else if (i==menu.noptions) {
                        if(user_mat(userfile, mat, xref)) {
                              strcpy(matnam,userfile);
                              matn=i;
                        }
		}
		else
                switch(toupper(*lin2))  {
                        case '?':
                        case 'H':
                                get_help('8');
                                break;
                        default:
                                fprintf(stdout,"\n\nUnrecognised Command\n\n");
                                break;
                }
        }
}


char prompt_for_yes_no(char *title,char *prompt)
{
	char line[80];
	char lin2[80];

	fprintf(stdout,"\n%s\n",title);
	strcpy(line,prompt);
	strcat(line, "(y/n) ? [y]");
	getstr(line,lin2, 79);
	if ((*lin2 != 'n') && (*lin2 != 'N'))
		return('y');
	else
		return('n');

}


/*
*	fatal()
*
*	Prints error msg to stdout and exits.
*	Variadic parameter list can be passed.
*
*	Return values:
*		none
*/

void fatal( char *msg,...)
{
	va_list ap;
	
	va_start(ap,msg);
	fprintf(stdout,"\n\nFATAL ERROR: ");
	vfprintf(stdout,msg,ap);
	fprintf(stdout,"\n\n");
	va_end(ap);
	exit(1);
}

/*
*	error()
*
*	Prints error msg to stdout.
*	Variadic parameter list can be passed.
*
*	Return values:
*		none
*/

void error( char *msg,...)
{
	va_list ap;
	
	va_start(ap,msg);
	fprintf(stdout,"\n\nERROR: ");
	vfprintf(stdout,msg,ap);
	fprintf(stdout,"\n\n");
	va_end(ap);
}

/*
*	warning()
*
*	Prints warning msg to stdout.
*	Variadic parameter list can be passed.
*
*	Return values:
*		none
*/

void warning( char *msg,...)
{
	va_list ap;
	
	va_start(ap,msg);
	fprintf(stdout,"\n\nWARNING: ");
	vfprintf(stdout,msg,ap);
	fprintf(stdout,"\n\n");
	va_end(ap);
}

/*
*	info()
*
*	Prints info msg to stdout.
*	Variadic parameter list can be passed.
*
*	Return values:
*		none
*/

void info( char *msg,...)
{
	va_list ap;
	
	va_start(ap,msg);
	fprintf(stdout,"\n");
	vfprintf(stdout,msg,ap);
	va_end(ap);
}
