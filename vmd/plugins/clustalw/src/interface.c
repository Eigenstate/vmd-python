/* command line interface for Clustal W  */
/* DES was here MARCH. 1994 */
/* DES was here SEPT.  1994 */
/* Fixed memory allocation bug in check_param() . Alan Bleasby Dec 2002 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <signal.h>
#include <setjmp.h>
#include "clustalw.h"
#include "param.h"

/*
*	Prototypes
*/

#ifdef UNIX
FILE    *open_path(char *);
#endif


char *nameonly(char *s) ;

static sint check_param(char **args,char *params[], char *param_arg[]);
static void set_optional_param(void);
static sint find_match(char *probe, char *list[], sint n);
static void show_aln(void);
static void create_parameter_output(void);
static void reset_align(void);
static void reset_prf1(void);
static void reset_prf2(void);
static void calc_gap_penalty_mask(int prf_length,char *struct_mask,char *gap_mask);
void print_sec_struct_mask(int prf_length,char *mask,char *struct_mask);
/*
*	 Global variables
*/

extern sint max_names;

extern Boolean interactive;

extern double  **tmat;
extern float    gap_open,      gap_extend;
extern float  	dna_gap_open,  dna_gap_extend;
extern float 	prot_gap_open, prot_gap_extend;
extern float    pw_go_penalty,      pw_ge_penalty;
extern float  	dna_pw_go_penalty,  dna_pw_ge_penalty;
extern float 	prot_pw_go_penalty, prot_pw_ge_penalty;
extern char 	revision_level[];
extern sint    wind_gap,ktup,window,signif;
extern sint    dna_wind_gap, dna_ktup, dna_window, dna_signif;
extern sint    prot_wind_gap,prot_ktup,prot_window,prot_signif;
extern sint	boot_ntrials;		/* number of bootstrap trials */
extern sint	nseqs;
extern sint	new_seq;
extern sint 	*seqlen_array;
extern sint 	divergence_cutoff;
extern sint 	debug;
extern Boolean 	no_weights;
extern Boolean 	neg_matrix;
extern Boolean  quick_pairalign;
extern Boolean	reset_alignments_new;		/* DES */
extern Boolean	reset_alignments_all;		/* DES */
extern sint 	gap_dist;
extern Boolean 	no_hyd_penalties, no_pref_penalties;
extern sint 	max_aa;
extern sint 	gap_pos1, gap_pos2;
extern sint  	max_aln_length;
extern sint 	*output_index, output_order;
extern sint profile_no;
extern short 	usermat[], pw_usermat[];
extern short 	aa_xref[], pw_aa_xref[];
extern short 	userdnamat[], pw_userdnamat[];
extern short 	dna_xref[], pw_dna_xref[];
extern sint	*seq_weight;

extern Boolean 	lowercase; /* Flag for GDE output - set on comm. line*/
extern Boolean 	cl_seq_numbers;

extern Boolean seqRange; /*Ramu */

extern Boolean 	output_clustal, output_nbrf, output_phylip, output_gcg, output_gde, output_nexus, output_fasta;
extern Boolean 	output_tree_clustal, output_tree_phylip, output_tree_distances, output_tree_nexus;
extern sint     bootstrap_format;
extern Boolean 	tossgaps, kimura;
extern Boolean  percent;
extern Boolean 	explicit_dnaflag;  /* Explicit setting of sequence type on comm.line*/
extern Boolean 	usemenu;
extern Boolean 	showaln, save_parameters;
extern Boolean	dnaflag;
extern float	transition_weight;
extern unsigned sint boot_ran_seed;


extern FILE 	*tree;
extern FILE 	*clustal_outfile, *gcg_outfile, *nbrf_outfile, *phylip_outfile, *nexus_outfile;
extern FILE     *fasta_outfile; /* Ramu */
extern FILE 	*gde_outfile;

extern char 	hyd_residues[];
extern char 	*amino_acid_codes;
extern char 	**args;
extern char	seqname[];

extern char 	**seq_array;
extern char 	**names, **titles;

extern char *gap_penalty_mask1,*gap_penalty_mask2;
extern char *sec_struct_mask1,*sec_struct_mask2;
extern sint struct_penalties,struct_penalties1,struct_penalties2;
extern sint output_struct_penalties;
extern Boolean use_ss1, use_ss2;
extern char *ss_name1,*ss_name2;


char *ss_name = NULL;
char *sec_struct_mask = NULL;
char *gap_penalty_mask = NULL;

char  	profile1_name[FILENAMELEN+1];
char  	profile2_name[FILENAMELEN+1];

Boolean empty;
Boolean profile1_empty, profile2_empty;   /* whether or not profiles   */

char 	outfile_name[FILENAMELEN+1]="";

static char 	clustal_outname[FILENAMELEN+1], gcg_outname[FILENAMELEN+1];
static char  	phylip_outname[FILENAMELEN+1],nbrf_outname[FILENAMELEN+1];
static char  	gde_outname[FILENAMELEN+1], nexus_outname[FILENAMELEN+1];
static char     fasta_outname[FILENAMELEN+1];  /* Ramu */
char     clustal_tree_name[FILENAMELEN+1]="";
char     dist_tree_name[FILENAMELEN+1]="";
char 	phylip_tree_name[FILENAMELEN+1]="";
char 	nexus_tree_name[FILENAMELEN+1]="";
char 	p1_tree_name[FILENAMELEN+1]="";
char 	p2_tree_name[FILENAMELEN+1]="";

char pim_name[FILENAMELEN+1]=""; /* Ramu */

static char *params[MAXARGS];
static char *param_arg[MAXARGS];

static char *cmd_line_type[] = {
                " ",
                "=n ",
                "=f ",
                "=string ",
                "=filename ",
                ""};

static sint numparams;
static Boolean check_tree = TRUE;

sint 	profile1_nseqs;	/* have been filled; the no. of seqs in prof 1*/
Boolean use_tree_file = FALSE,new_tree_file = FALSE;
Boolean use_tree1_file = FALSE, use_tree2_file = FALSE;
Boolean new_tree1_file = FALSE, new_tree2_file = FALSE;

static char *lin2;

MatMenu dnamatrix_menu = {3,
                {{"IUB","iub"},
                {"CLUSTALW(1.6)","clustalw"},
                {"User defined",""}}
		};

MatMenu matrix_menu = {5,
                {{"BLOSUM series","blosum"},
                {"PAM series","pam"},
                {"Gonnet series","gonnet"},
                {"Identity matrix","id"},
                {"User defined",""}}
		};
 
MatMenu pw_matrix_menu = {5,
                {{"BLOSUM 30","blosum"},
                {"PAM 350","pam"},
                {"Gonnet 250","gonnet"},
                {"Identity matrix","id"},
                {"User defined",""}}
		};


void init_interface(void)
{
  empty=TRUE;
  
  profile1_empty = TRUE;     /*  */
  profile2_empty = TRUE;     /*  */
  
  lin2 = (char *)ckalloc( (MAXLINE+1) * sizeof (char) );
	
}




static sint check_param(char **args,char *params[], char *param_arg[])
{

/*
#ifndef MAC
        char *strtok(char *s1, const char *s2);
#endif
*/
        sint     len,i,j,k,s,n,match[MAXARGS];
		Boolean 	name1 = FALSE;
		sint ajb;

	if(args[0]==NULL) return -1;



	params[0]=(char *)ckalloc((strlen(args[0])+1)*sizeof(char));
	if (args[0][0]!=COMMANDSEP)
	{
		name1 = TRUE;
		strcpy(params[0],args[0]);
	}
	else
		strcpy(params[0],&args[0][1]);

        for (i=1;i<MAXARGS;i++) {
		if(args[i]==NULL) break;
		params[i]=(char *)ckalloc((strlen(args[i])+1)*sizeof(char));
		ajb=0;
		for(j=0;j<strlen(args[i])-1;j++)
			if(isprint(args[i][j+1])) params[i][ajb++]=args[i][j+1];
		params[i][ajb]='\0';
        }

        if (i==MAXARGS) {
		fprintf(stdout,"Error: too many command line arguments\n");
 		return(-1);
	}
/*
    special case - first parameter is input filename
  */
  s = 0;
  if(name1 == TRUE) {
    strcpy(seqname, params[0]);
    /*  JULIE
	convert to lower case now
    */
#ifndef UNIX
    for(k=0;k<(sint)strlen(params[0]);++k) seqname[k]=tolower(params[0][k]);
#else
    for(k=0;k<(sint)strlen(params[0]);++k) seqname[k]=params[0][k];
#endif 
    s++;
  }
  
  n = i;
  for (i=s;i<n;i++) {
    param_arg[i] = NULL;
    len = (sint)strlen(params[i]);
    for(j=0; j<len; j++)
      if(params[i][j] == '=') {
	param_arg[i] = (char *)ckalloc((len-j) * sizeof(char));
	strncpy(param_arg[i],&params[i][j+1],len-j-1);
	params[i][j] = EOS;
	/*  JULIE
	    convert keywords to lower case now
	*/
	for(k=0;k<j;++k) params[i][k]=tolower(params[i][k]);
	param_arg[i][len-j-1] = EOS;
	break;
      }
  }
  
  /*
    for each parameter given on the command line, first search the list of recognised optional 
    parameters....
  */

  for (i=0;i<n;i++) {
    if ((i==0) && (name1 == TRUE)) continue;
    j = 0;
    match[i] = -1;
    for(;;) {
      if (cmd_line_para[j].str[0] == '\0') break;
      if (!strcmp(params[i],cmd_line_para[j].str)) {
	match[i] = j;
	*cmd_line_para[match[i]].flag = i;
	if ((cmd_line_para[match[i]].type != NOARG) &&
	    (param_arg[i] == NULL)) {
	  fprintf(stdout,
		  "Error: parameter required for /%s\n",params[i]);
	  exit(1);
	}
	/*  JULIE
	    convert parameters to lower case now, unless the parameter is a filename
	*/
#ifdef UNIX
	else if (cmd_line_para[match[i]].type != FILARG
		 && param_arg[i] != NULL)
#endif 
	  if (param_arg[i]!=0)
	    {
	      for(k=0;k<strlen(param_arg[i]);++k)
		param_arg[i][k]=tolower(param_arg[i][k]);
	    }
	break;
      }
      j++;
    }
  }
  /*
    ....then the list of recognised input files,.... 
*/
    for (i=0;i<n;i++) {
		if ((i==0) && (name1 == TRUE)) continue;
		if (match[i] != -1) continue;
		j = 0;
		for(;;) {
			if (cmd_line_file[j].str[0] == '\0') break;
			if (!strcmp(params[i],cmd_line_file[j].str)) {
				match[i] = j;
				*cmd_line_file[match[i]].flag = i;
				if ((cmd_line_file[match[i]].type != NOARG) &&
                                    (param_arg[i] == NULL)) {
					fprintf(stdout,
                       				 "Error: parameter required for /%s\n",params[i]);
					exit(1);
				}
				break;
			}
			j++;
		}
	}
/*
	....and finally the recognised verbs. 
*/
    for (i=0;i<n;i++) {
		if ((i==0) && (name1 == TRUE)) continue;
		if (match[i] != -1) continue;
		j = 0;
		for(;;) {
			if (cmd_line_verb[j].str[0] == '\0') break;
			if (!strcmp(params[i],cmd_line_verb[j].str)) {
				match[i] = j;
				*cmd_line_verb[match[i]].flag = i;
				if ((cmd_line_verb[match[i]].type != NOARG) &&
                                    (param_arg[i] == NULL)) {
					fprintf(stdout,
                       				 "Error: parameter required for /%s\n",params[i]);
					exit(1);
				}
				break;
			}
			j++;
		}
	}

/*
	check for any unrecognised parameters.
*/
    for (i=0;i<n;i++) {
		if (match[i] == -1) {
			fprintf(stdout,
                        "Error: unknown option %c%s\n",COMMANDSEP,params[i]);
			exit(1);
		}
	}
        return(n);
}

static void set_optional_param(void)
{
  int i,temp;
  int c;
  float ftemp;
  char tstr[100];
  
  /****************************************************************************/
  /* look for parameters on command line  e.g. gap penalties, k-tuple etc.    */
  /****************************************************************************/
  
  /*** ? /score=percent or /score=absolute */
  if(setscore != -1)
    if(strlen(param_arg[setscore]) > 0) {
      temp = find_match(param_arg[setscore],score_arg,2);
      if(temp == 0)
	percent = TRUE;
      else if(temp == 1)
	percent = FALSE;
      else
	fprintf(stdout,"\nUnknown SCORE type: %s\n",
		param_arg[setscore]);
    }
  
  /*** ? /seed=n */
  if(setseed != -1) {
    temp = 0;
    if(strlen(param_arg[setseed]) > 0)
      if (sscanf(param_arg[setseed],"%d",&temp)!=1) {
	fprintf(stdout,"Bad option for /seed (must be integer)\n");
	temp = 0;
      }
    if(temp > 0) boot_ran_seed = temp;
    fprintf(stdout,"\ntemp = %d; seed = %u;\n",(pint)temp,boot_ran_seed);
  }
  

/*** ? /output=PIR, GCG, GDE or PHYLIP */
		if(setoutput != -1)
		if(strlen(param_arg[setoutput]) > 0) {
			temp = find_match(param_arg[setoutput],output_arg,6);
			if (temp >= 0 && temp <= 5) {
				output_clustal = FALSE;
				output_gcg     = FALSE;
				output_phylip  = FALSE;
				output_nbrf    = FALSE;
				output_gde     = FALSE;
				output_nexus   = FALSE;
				output_fasta   = FALSE;
			}
			switch (temp) {
				case 0: /* GCG */
					output_gcg     = TRUE;
					break;
				case 1: /* GDE */
					output_gde     = TRUE;
					break;
				case 2: /* PIR */
					output_nbrf    = TRUE;
					break;
				case 3: /* PHYLIP */
					output_phylip  = TRUE;
					break;
				case 4: /* NEXUS */
					output_nexus   = TRUE;
					break;
				case 5: /* NEXUS */
					output_fasta   = TRUE;
					break;
				default:
					fprintf(stdout,"\nUnknown OUTPUT type: %s\n",
					param_arg[setoutput]);
			}
		}

/*** ? /outputtree=NJ or PHYLIP or DIST or NEXUS */
	if(setoutputtree != -1)
		if(strlen(param_arg[setoutputtree]) > 0) {
			temp = find_match(param_arg[setoutputtree],outputtree_arg,4);
			switch (temp) {
				case 0: /* NJ */
					output_tree_clustal = TRUE;
					break;
				case 1: /* PHYLIP */
					output_tree_phylip  = TRUE;
					break;
				case 2: /* DIST */
					output_tree_distances = TRUE;
					break;
				case 3: /* NEXUS */
					output_tree_nexus = TRUE;
					break;
				default:
					fprintf(stdout,"\nUnknown OUTPUT TREE type: %s\n",
					param_arg[setoutputtree]);
			}
		}

/*** ? /profile (sets type of second input file to profile) */
  if(setprofile != -1)
    profile_type = PROFILE;
  
  /*** ? /sequences (sets type of second input file to list of sequences)  */
  if(setsequences != -1)
    profile_type = SEQUENCE;
  
  
  
  /*** ? /ktuple=n */
  if(setktuple != -1) {
    temp = 0;
    if(strlen(param_arg[setktuple]) > 0)
      if (sscanf(param_arg[setktuple],"%d",&temp)!=1) {
	fprintf(stdout,"Bad option for /ktuple (must be integer)\n");
	temp = 0;
      }
    if(temp > 0) {
      if(dnaflag) {
	if(temp <= 4) {
	  ktup         = temp;
	  dna_ktup     = ktup;
	  wind_gap     = ktup + 4;
	  dna_wind_gap = wind_gap;
	}
      }
      else {
	if(temp <= 2) {
	  ktup          = temp;
	  prot_ktup     = ktup;
	  wind_gap      = ktup + 3;
	  prot_wind_gap = wind_gap;
	}
      }
    }
  }
  
  /*** ? /pairgap=n */
  if(setpairgap != -1) {
    temp = 0;
    if(strlen(param_arg[setpairgap]) > 0)
      if (sscanf(param_arg[setpairgap],"%d",&temp)!=1) {
         fprintf(stdout,"Bad option for /pairgap (must be integer)\n");
         temp = 0;
      }
    if(temp > 0)
    {
      if(dnaflag) {
         if(temp > ktup) {
            wind_gap     = temp;
            dna_wind_gap = wind_gap;
         }
      }
      else {
         if(temp > ktup) {
            wind_gap      = temp;
            prot_wind_gap = wind_gap;
         }
      }
    }
  }
  
  
/*** ? /topdiags=n   */
  if(settopdiags != -1) {
    temp = 0;
    if(strlen(param_arg[settopdiags]) > 0)
      if (sscanf(param_arg[settopdiags],"%d",&temp)!=1) {
	fprintf(stdout,"Bad option for /topdiags (must be integer)\n");
	temp = 0;
      }
    if(temp > 0)
    {
      if(dnaflag) {
         if(temp > ktup) {
           signif       = temp;
           dna_signif   = signif;
         }
      }
      else {
         if(temp > ktup) {
           signif        = temp;
           prot_signif   = signif;
         }
      }
    }
  }
	

/*** ? /window=n  */
  if(setwindow != -1) {
    temp = 0;
    if(strlen(param_arg[setwindow]) > 0)
      if (sscanf(param_arg[setwindow],"%d",&temp)!=1) {
	fprintf(stdout,"Bad option for /window (must be integer)\n");
	temp = 0;
      }
    if(temp > 0)
    {
      if(dnaflag) {
         if(temp > ktup) {
           window       = temp;
           dna_window   = window;
         }
      }
      else {
         if(temp > ktup) {
           window        = temp;
           prot_window   = window;
         }
      }
    }
  }
  
/*** ? /kimura */
  if(setkimura != -1)
    kimura = TRUE;
  
  /*** ? /tossgaps */
  if(settossgaps != -1)
    tossgaps = TRUE;
  
  
  /*** ? /negative  */
  if(setnegative != -1)
    neg_matrix = TRUE;
  
  /*** ? /noweights */
  if(setnoweights!= -1)
    no_weights = TRUE;
  
  
  /*** ? /pwmatrix=ID (user's file)  */
  if(setpwmatrix != -1)
    {
      temp=strlen(param_arg[setpwmatrix]);
      if(temp > 0) {
	for(i=0;i<temp;i++)
	  if (isupper(param_arg[setpwmatrix][i]))
	    tstr[i]=tolower(param_arg[setpwmatrix][i]);
	  else
	    tstr[i]=param_arg[setpwmatrix][i];
	tstr[i]='\0';
	if (strcmp(tstr,"blosum")==0) {
	  strcpy(pw_mtrxname, tstr);
	  pw_matnum = 1;
                        }
                        else if (strcmp(tstr,"pam")==0) {
                                strcpy(pw_mtrxname, tstr);
                                pw_matnum = 2;
                        }
                        else if (strcmp(tstr,"gonnet")==0) {
                                strcpy(pw_mtrxname, tstr);
                                pw_matnum = 3;
                        }
                        else if (strcmp(tstr,"id")==0) {
                                strcpy(pw_mtrxname, tstr);
                                pw_matnum = 4;
                        }
			else {
                                if(user_mat(param_arg[setpwmatrix], pw_usermat, pw_aa_xref))
                                  {
                                     strcpy(pw_mtrxname,param_arg[setpwmatrix]);
                                     strcpy(pw_usermtrxname,param_arg[setpwmatrix]);
                                     pw_matnum=5;
                                  }
				else exit(1);
			}

		}
	}

/*** ? /matrix=ID (user's file)  */
	if(setmatrix != -1)
	{
		temp=strlen(param_arg[setmatrix]);
		if(temp > 0) {
			for(i=0;i<temp;i++)
				if (isupper(param_arg[setmatrix][i]))
					tstr[i]=tolower(param_arg[setmatrix][i]);
				else
					tstr[i]=param_arg[setmatrix][i];
			tstr[i]='\0';
                        if (strcmp(tstr,"blosum")==0) {
                                strcpy(mtrxname, tstr);
                                matnum = 1;
                        }
                        else if (strcmp(tstr,"pam")==0) {
                                strcpy(mtrxname, tstr);
                                matnum = 2;
                        }
                        else if (strcmp(tstr,"gonnet")==0) {
                                strcpy(mtrxname, tstr);
                                matnum = 3;
                        }
                        else if (strcmp(tstr,"id")==0) {
                                strcpy(mtrxname, tstr);
                                matnum = 4;
                        }
			else {
                                if(user_mat_series(param_arg[setmatrix], usermat, aa_xref))
                                  {
                                     strcpy(mtrxname,param_arg[setmatrix]);
                                     strcpy(usermtrxname,param_arg[setmatrix]);
                                     matnum=5;
                                  }
				else exit(1);
			}

		}
	}

/*** ? /pwdnamatrix=ID (user's file)  */
	if(setpwdnamatrix != -1)
	{
		temp=strlen(param_arg[setpwdnamatrix]);
		if(temp > 0) {
			for(i=0;i<temp;i++)
				if (isupper(param_arg[setpwdnamatrix][i]))
					tstr[i]=tolower(param_arg[setpwdnamatrix][i]);
				else
					tstr[i]=param_arg[setpwdnamatrix][i];
			tstr[i]='\0';
                        if (strcmp(tstr,"iub")==0) {
                                strcpy(pw_dnamtrxname, tstr);
                                pw_dnamatnum = 1;
                        }
                        else if (strcmp(tstr,"clustalw")==0) {
                                strcpy(pw_dnamtrxname, tstr);
                                pw_dnamatnum = 2;
                        }
			else {
                                if(user_mat(param_arg[setpwdnamatrix], pw_userdnamat, pw_dna_xref))
                                  {
                                     strcpy(pw_dnamtrxname,param_arg[setpwdnamatrix]);
                                     strcpy(pw_dnausermtrxname,param_arg[setpwdnamatrix]);
                                     pw_dnamatnum=3;
                                  }
				else exit(1);
			}

		}
	}

/*** ? /matrix=ID (user's file)  */
	if(setdnamatrix != -1)
	{
		temp=strlen(param_arg[setdnamatrix]);
		if(temp > 0) {
			for(i=0;i<temp;i++)
				if (isupper(param_arg[setdnamatrix][i]))
					tstr[i]=tolower(param_arg[setdnamatrix][i]);
				else
					tstr[i]=param_arg[setdnamatrix][i];
			tstr[i]='\0';
                        if (strcmp(tstr,"iub")==0) {
                                strcpy(dnamtrxname, tstr);
                                dnamatnum = 1;
                        }
                        else if (strcmp(tstr,"clustalw")==0) {
                                strcpy(dnamtrxname, tstr);
                                dnamatnum = 2;
                        }
			else {
                                if(user_mat(param_arg[setdnamatrix], userdnamat, dna_xref))
                                  {
                                     strcpy(dnamtrxname,param_arg[setdnamatrix]);
                                     strcpy(dnausermtrxname,param_arg[setdnamatrix]);
                                     dnamatnum=3;
                                  }
				else exit(1);
			}

		}
	}
/*** ? /maxdiv= n */
	if(setmaxdiv != -1) {
		temp = 0;
		if(strlen(param_arg[setmaxdiv]) > 0)
			if (sscanf(param_arg[setmaxdiv],"%d",&temp)!=1) {
                 fprintf(stdout,"Bad option for /maxdiv (must be integer)\n");
                 temp = 0;
            }
		if (temp >= 0)
			divergence_cutoff = temp;
	}

/*** ? /gapdist= n */
	if(setgapdist != -1) {
		temp = 0;
		if(strlen(param_arg[setgapdist]) > 0)
			if (sscanf(param_arg[setgapdist],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /gapdist (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 0)
			gap_dist = temp;
	}

/*** ? /debug= n */
	if(setdebug != -1) {
		temp = 0;
		if(strlen(param_arg[setdebug]) > 0)
			if (sscanf(param_arg[setdebug],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /debug (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 0)
			debug = temp;
	}

/*** ? /outfile= (user's file)  */
	if(setoutfile != -1)
		if(strlen(param_arg[setoutfile]) > 0) {
                        strcpy(outfile_name, param_arg[setoutfile]);
		}

/*** ? /case= lower/upper  */
	if(setcase != -1) 
		if(strlen(param_arg[setcase]) > 0) {
			temp = find_match(param_arg[setcase],case_arg,2);
			if(temp == 0) {
				lowercase = TRUE;
			}
			else if(temp == 1) {
				lowercase = FALSE;
			}
			else
				fprintf(stdout,"\nUnknown case %s\n",
				param_arg[setcase]);
		}

/*** ? /seqnos=off/on  */
	if(setseqno != -1) 
		if(strlen(param_arg[setseqno]) > 0) {
			temp = find_match(param_arg[setseqno],seqno_arg,2);
			if(temp == 0) {
				cl_seq_numbers = FALSE;
			}
			else if(temp == 1) {
				cl_seq_numbers = TRUE;
			}
			else
				fprintf(stdout,"\nUnknown SEQNO option %s\n",
				param_arg[setseqno]);
		}



	if(setseqno_range != -1) 
		if(strlen(param_arg[setseqno_range]) > 0) {
			temp = find_match(param_arg[setseqno_range],seqno_range_arg,2);
			printf("\n comparing  "); 
			printf("\nparam_arg[setseqno_range]= %s", param_arg[setseqno_range]);
			/* printf("\nseqno_range_arg = %s ",seqno_range_arg); */
			printf("\n comparing \n "); 

			if(temp == 0) {
				seqRange = FALSE;
			}
			else if(temp == 1) {
				seqRange = TRUE;

			}
			else
				fprintf(stdout,"\nUnknown Sequence range  option %s\n",
				param_arg[setseqno_range]);
		}


/*** ? /range=n:m */
	if(setrange != -1) {
		temp = 0;
		if(strlen(param_arg[setrange]) > 0)
			if (sscanf(param_arg[setrange],"%d:%d",&temp,&temp)!=2) {
                 fprintf(stdout,"setrange:  Syntax Error: Cannot set range, should be from:to \n");
                 temp = 0;
            }
	}

/*** ? /range=n:m */



/*** ? /gapopen=n  */
	if(setgapopen != -1) {
		ftemp = 0.0;
		if(strlen(param_arg[setgapopen]) > 0)
			if (sscanf(param_arg[setgapopen],"%f",&ftemp)!=1) {
                         fprintf(stdout,"Bad option for /gapopen (must be real number)\n");
                         ftemp = 0.0;
                    }
		if(ftemp >= 0.0)
      {
			if(dnaflag) {
					gap_open     = ftemp;
					dna_gap_open = gap_open;
			}
			else {
					gap_open      = ftemp;
					prot_gap_open = gap_open;
			}
      }
	}


/*** ? /gapext=n   */
	if(setgapext != -1) {
		ftemp = 0.0;
		if(strlen(param_arg[setgapext]) > 0)
			if (sscanf(param_arg[setgapext],"%f",&ftemp)!=1) {
                         fprintf(stdout,"Bad option for /gapext (must be real number)\n");
                         ftemp = 0.0;
                    }
		if(ftemp >= 0)
      {
			if(dnaflag) {
					gap_extend      = ftemp;
					dna_gap_extend  = gap_extend;
			}
			else {
					gap_extend      = ftemp;
					prot_gap_extend = gap_extend;
			}
      }
	}

/*** ? /transweight=n*/
	if(settransweight != -1) {
		ftemp = 0.0;
		if(strlen(param_arg[settransweight]) > 0)
			if (sscanf(param_arg[settransweight],"%f",&ftemp)!=1) {
                         fprintf(stdout,"Bad option for /transweight (must be real number)\n");
                         ftemp = 0.0;
                    }
		transition_weight=ftemp;
	}

/*** ? /pwgapopen=n  */
	if(setpwgapopen != -1) {
		ftemp = 0.0;
		if(strlen(param_arg[setpwgapopen]) > 0)
			if (sscanf(param_arg[setpwgapopen],"%f",&ftemp)!=1) {
                         fprintf(stdout,"Bad option for /pwgapopen (must be real number)\n");
                         ftemp = 0.0;
                    }
		if(ftemp >= 0.0)
      {
			if(dnaflag) {
					pw_go_penalty  = ftemp;
                                        dna_pw_go_penalty = pw_go_penalty;
			}
			else {
					pw_go_penalty  = ftemp;
                                        prot_pw_go_penalty = pw_go_penalty;
			}
      }
	}


/*** ? /gapext=n   */
	if(setpwgapext != -1) {
		ftemp = 0.0;
		if(strlen(param_arg[setpwgapext]) > 0)
			if (sscanf(param_arg[setpwgapext],"%f",&ftemp)!=1) {
                         fprintf(stdout,"Bad option for /pwgapext (must be real number)\n");
                         ftemp = 0.0;
                    }
		if(ftemp >= 0)
      {
			if(dnaflag) {
					pw_ge_penalty  = ftemp;
                                        dna_pw_ge_penalty = pw_ge_penalty;
			}
			else {
					pw_ge_penalty  = ftemp;
                                        prot_pw_ge_penalty = pw_ge_penalty;
			}
      }
	}



/*** ? /outorder=n  */
	if(setoutorder != -1) {
		if(strlen(param_arg[setoutorder]) > 0)
			temp = find_match(param_arg[setoutorder],outorder_arg,2);
			if(temp == 0)  {	
				output_order   = INPUT;
			}
			else if(temp == 1)  {	
				output_order   = ALIGNED;
			}
			else
				fprintf(stdout,"\nUnknown OUTPUT ORDER type %s\n",
				param_arg[setoutorder]);
	}

/*** ? /bootlabels=n  */
	if(setbootlabels != -1) {
		if(strlen(param_arg[setbootlabels]) > 0)
			temp = find_match(param_arg[setbootlabels],bootlabels_arg,2);
			if(temp == 0)  {	
				bootstrap_format   = BS_NODE_LABELS;
			}
			else if(temp == 1)  {	
				bootstrap_format   = BS_BRANCH_LABELS;
			}
			else
				fprintf(stdout,"\nUnknown bootlabels type %s\n",
				param_arg[setoutorder]);
	}

/*** ? /endgaps */
	if(setuseendgaps != -1)
		use_endgaps = FALSE;

/*** ? /nopgap  */
	if(setnopgap != -1)
		no_pref_penalties = TRUE;

/*** ? /nohgap  */
	if(setnohgap != -1)
		no_hyd_penalties = TRUE;

/*** ? /novgap  */
	if(setnovgap != -1)
		no_var_penalties = FALSE;

/*** ? /hgapresidues="string"  */
	if(sethgapres != -1)
		if(strlen(param_arg[sethgapres]) > 0) {
			for (i=0;i<strlen(hyd_residues) && i<26;i++) {
				c = param_arg[sethgapres][i];
				if (isalpha(c))
					hyd_residues[i] = (char)toupper(c);
				else
					break;
			}
		}
		
		
/*** ? /nosecstr1  */
	if(setsecstr1 != -1)
		use_ss1 = FALSE;

/*** ? /nosecstr2  */
	if(setsecstr2 != -1)
		use_ss2 = FALSE;

/*** ? /secstroutput  */
	if(setsecstroutput != -1)
		if(strlen(param_arg[setsecstroutput]) > 0) {
			temp = find_match(param_arg[setsecstroutput],outputsecstr_arg,4);
			if(temp >= 0 && temp <= 3)
				output_struct_penalties = temp;
			else
				fprintf(stdout,"\nUnknown case %s\n",
				param_arg[setsecstroutput]);
		}


/*** ? /helixgap= n */
	if(sethelixgap != -1) {
		temp = 0;
		if(strlen(param_arg[sethelixgap]) > 0)
			if (sscanf(param_arg[sethelixgap],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /helixgap (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 1 && temp <= 9)
			helix_penalty = temp;
	}
	
/*** ? /strandgap= n */
	if(setstrandgap != -1) {
		temp = 0;
		if(strlen(param_arg[setstrandgap]) > 0)
			if (sscanf(param_arg[setstrandgap],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /strandgap (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 1 && temp <= 9)
			strand_penalty = temp;
	}
	
/*** ? /loopgap= n */
	if(setloopgap != -1) {
		temp = 0;
		if(strlen(param_arg[setloopgap]) > 0)
			if (sscanf(param_arg[setloopgap],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /loopgap (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 1 && temp <= 9)
			loop_penalty = temp;
	}

/*** ? /terminalgap= n */
	if(setterminalgap != -1) {
		temp = 0;
		if(strlen(param_arg[setterminalgap]) > 0)
			if (sscanf(param_arg[setterminalgap],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /terminalgap (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 1 && temp <= 9) {
			helix_end_penalty = temp;
			strand_end_penalty = temp;
		}
	}
	
/*** ? /helixendin= n */
	if(sethelixendin != -1) {
		temp = 0;
		if(strlen(param_arg[sethelixendin]) > 0)
			if (sscanf(param_arg[sethelixendin],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /helixendin (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 0 && temp <= 3)
			helix_end_minus = temp;
	}

/*** ? /helixendout= n */
	if(sethelixendout != -1) {
		temp = 0;
		if(strlen(param_arg[sethelixendout]) > 0)
			if (sscanf(param_arg[sethelixendout],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /helixendout (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 0 && temp <= 3)
			helix_end_plus = temp;
	}

/*** ? /strandendin= n */
	if(setstrandendin != -1) {
		temp = 0;
		if(strlen(param_arg[setstrandendin]) > 0)
			if (sscanf(param_arg[setstrandendin],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /strandendin (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 0 && temp <= 3)
			strand_end_minus = temp;
	}

/*** ? /strandendout= n */
	if(setstrandendout != -1) {
		temp = 0;
		if(strlen(param_arg[setstrandendout]) > 0)
			if (sscanf(param_arg[setstrandendout],"%d",&temp)!=1) {
                         fprintf(stdout,"Bad option for /strandendout (must be integer)\n");
                         temp = 0;
                    }
		if (temp >= 0 && temp <= 3)
			strand_end_plus = temp;
	}

}
 
#ifdef UNIX
FILE *open_path(char *fname)  /* to open in read-only file fname searching for 
				 it through all path directories */
{
#define Mxdir 70
        char dir[Mxdir+1], *path, *deb, *fin;
        FILE *fich;
        sint lf, ltot;
	char *path1;
 
        path=getenv("PATH"); 	/* get the list of path directories, 
					separated by :
    				*/

	/* added for File System Standards  - Francois */
	path1=(char *)ckalloc((strlen(path)+64)*sizeof(char));
	strcpy(path1,path);
	strcat(path1,"/usr/share/clustalx:/usr/local/share/clustalx"); 

        lf=(sint)strlen(fname);
        deb=path1;
        do
                {
                fin=strchr(deb,':');
                if(fin!=NULL)
                        { strncpy(dir,deb,fin-deb); ltot=fin-deb; }
                else
                        { strcpy(dir,deb); ltot=(sint)strlen(dir); }
                /* now one directory is in string dir */
                if( ltot + lf + 1 <= Mxdir)
                        {
                        dir[ltot]='/';
                        strcpy(dir+ltot+1,fname); /* now dir is appended with fi
   lename */
                        if( (fich = fopen(dir,"r") ) != NULL) break;
                        }
                else fich = NULL;
                deb=fin+1;
                }
        while (fin != NULL);
        return fich;
}
#endif


void get_help(char help_pointer)    /* Help procedure */
{	
	FILE *help_file;
	sint  i, number, nlines;
	Boolean found_help;
	char temp[MAXLINE+1];
	char token = '\0';
	char *digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	char *help_marker    = ">>HELP";

	extern char *help_file_name;

#ifdef VMS
        if((help_file=fopen(help_file_name,"r","rat=cr","rfm=var"))==NULL) {
            error("Cannot open help file [%s]",help_file_name);
            return;
        }
#else

#ifdef UNIX
        if((help_file=open_path(help_file_name))==NULL) {
             if((help_file=fopen(help_file_name,"r"))==NULL) {
                  error("Cannot open help file [%s]",help_file_name);
                  return;
             }
        }
        
#else
        if((help_file=fopen(help_file_name,"r"))==NULL) {
            error("Cannot open help file [%s]",help_file_name);
            return;
        }
#endif

#endif
/*		error("Cannot open help file [%s]",help_file_name);
		return;
	}
*/
	nlines = 0;
	number = -1;
	found_help = FALSE;

	while(TRUE) {
		if(fgets(temp,MAXLINE+1,help_file) == NULL) {
			if(!found_help)
				error("No help found in help file");
			fclose(help_file);
			return;
		}
		if(strstr(temp,help_marker)) {
                        token = ' ';
			for(i=strlen(help_marker); i<8; i++)
				if(strchr(digits, temp[i])) {
					token = temp[i];
					break;
				}
		}
		if(token == help_pointer) {
			found_help = TRUE;
			while(fgets(temp,MAXLINE+1,help_file)) {
				if(strstr(temp, help_marker)){
				  	if(usemenu) {
						fprintf(stdout,"\n");
				    		getstr("Press [RETURN] to continue",lin2, MAXLINE);
				  	}
					fclose(help_file);
					return;
				}
				if(temp[0]!='<') {
			       		fputs(temp,stdout);
			       		++nlines;
				}
			       if(usemenu) {
			          if(nlines >= PAGE_LEN) {
				     	   fprintf(stdout,"\n");
			 	  	   getstr("Press [RETURN] to continue or  X  to stop",lin2,
                  MAXLINE);
				  	   if(toupper(*lin2) == 'X') {
						   fclose(help_file);
						   return;
				  	   }
				  	   else
						   nlines = 0;
				   }
			       }
			}
			if(usemenu) {
				fprintf(stdout,"\n");
				getstr("Press [RETURN] to continue",lin2, MAXLINE);
			}
			fclose(help_file);
		}
	}
}

static void show_aln(void)         /* Alignment screen display procedure */
{
        FILE *file;
        sint  nlines;
        char temp[MAXLINE+1];
        char file_name[FILENAMELEN+1];

        if(output_clustal) strcpy(file_name,clustal_outname);
        else if(output_nbrf) strcpy(file_name,nbrf_outname);
        else if(output_gcg) strcpy(file_name,gcg_outname);
        else if(output_phylip) strcpy(file_name,phylip_outname);
        else if(output_gde) strcpy(file_name,gde_outname);
        else if(output_nexus) strcpy(file_name,nexus_outname);
        else if(output_fasta) strcpy(file_name,fasta_outname);

#ifdef VMS
        if((file=fopen(file_name,"r","rat=cr","rfm=var"))==NULL) {
#else
        if((file=fopen(file_name,"r"))==NULL) {
#endif
                error("Cannot open file [%s]",file_name);
                return;
        }

        fprintf(stdout,"\n\n");
        nlines = 0;

        while(fgets(temp,MAXLINE+1,file)) {
                fputs(temp,stdout);
                ++nlines;
                if(nlines >= PAGE_LEN) {
                        fprintf(stdout,"\n");
                        getstr("Press [RETURN] to continue or  X  to stop",lin2,
                        MAXLINE);
                        if(toupper(*lin2) == 'X') {
                                fclose(file);
                                return;
                        }
                        else
                                nlines = 0;
                }
        }
        fclose(file);
        fprintf(stdout,"\n");
        getstr("Press [RETURN] to continue",lin2, MAXLINE);
}


void parse_params(Boolean xmenus)
{
	sint i,j/*,len*/,temp;
	static sint cl_error_code=0;
        char path[FILENAMELEN];


	Boolean do_align, do_convert, do_align_only, do_tree_only, do_tree, do_boot, do_profile, do_something;

	if (!xmenus)
	{
		fprintf(stdout,"\n\n\n");
		fprintf(stdout," CLUSTAL %s Multiple Sequence Alignments\n\n\n",revision_level);
	}

	do_align = do_convert = do_align_only = do_tree_only = do_tree = do_boot = do_profile = do_something = FALSE;

	*seqname=EOS;

/* JULIE 
	len=(sint)strlen(paramstr);
   Stop converting command line to lower case - unix, mac, pc are case sensitive
	for(i=0;i<len;++i) paramstr[i]=tolower(paramstr[i]);
*/

    numparams = check_param(args, params, param_arg);
	if (numparams <0) exit(1);

	if(sethelp != -1) {
		get_help('9');
		exit(1);
	}

	if(setoptions != -1) {
		fprintf(stdout,"clustalw option list:-\n");
		for (i=0;cmd_line_verb[i].str[0] != '\0';i++) {
			fprintf(stdout,"\t\t%c%s%s",COMMANDSEP,cmd_line_verb[i].str,cmd_line_type[cmd_line_verb[i].type]);
			if (cmd_line_verb[i].type == OPTARG) {
				if (cmd_line_verb[i].arg[0][0] != '\0')
					fprintf(stdout,"=%s",cmd_line_verb[i].arg[0]);
				for (j=1;cmd_line_verb[i].arg[j][0] != '\0';j++)
					fprintf(stdout," OR %s",cmd_line_verb[i].arg[j]);
			}
			fprintf(stdout,"\n");
		}
		for (i=0;cmd_line_file[i].str[0] != '\0';i++) {
			fprintf(stdout,"\t\t%c%s%s",COMMANDSEP,cmd_line_file[i].str,cmd_line_type[cmd_line_file[i].type]);
			if (cmd_line_file[i].type == OPTARG) {
				if (cmd_line_file[i].arg[0][0] != '\0')
					fprintf(stdout,"=%s",cmd_line_file[i].arg[0]);
				for (j=1;cmd_line_file[i].arg[j][0] != '\0';j++)
					fprintf(stdout," OR %s",cmd_line_file[i].arg[j]);
			}
			fprintf(stdout,"\n");
		}
		for (i=0;cmd_line_para[i].str[0] != '\0';i++) {
			fprintf(stdout,"\t\t%c%s%s",COMMANDSEP,cmd_line_para[i].str,cmd_line_type[cmd_line_para[i].type]);
			if (cmd_line_para[i].type == OPTARG) {
				if (cmd_line_para[i].arg[0][0] != '\0')
					fprintf(stdout,"=%s",cmd_line_para[i].arg[0]);
				for (j=1;cmd_line_para[i].arg[j][0] != '\0';j++)
					fprintf(stdout," OR %s",cmd_line_para[i].arg[j]);
			}
			fprintf(stdout,"\n");
		}
		exit(1);
	}


/*****************************************************************************/
/*  Check to see if sequence type is explicitely stated..override ************/
/* the automatic checking (DNA or Protein).   /type=d or /type=p *************/
/*****************************************************************************/
	if(settype != -1)
		if(strlen(param_arg[settype])>0) {
			temp = find_match(param_arg[settype],type_arg,2);
			if(temp == 0) {
				dnaflag = FALSE;
				explicit_dnaflag = TRUE;
				info("Sequence type explicitly set to Protein");
			}
			else if(temp == 1) {
				info("Sequence type explicitly set to DNA");
				dnaflag = TRUE;
				explicit_dnaflag = TRUE;
			}
			else
				fprintf(stdout,"\nUnknown sequence type %s\n",
				param_arg[settype]);
		}


/***************************************************************************
*   check to see if 1st parameter does not start with '/' i.e. look for an *
*   input file as first parameter.   The input file can also be specified  *
*   by /infile=fname.                                                      *
****************************************************************************/
/* JULIE - moved to check_param()
	if(paramstr[0] != '/') {
		strcpy(seqname, params[0]);
	}
*/

/**************************************************/
/*  Look for /infile=file.ext on the command line */
/**************************************************/

	if(setinfile != -1) {
		if(strlen(param_arg[setinfile]) <= 0) {
			error("Bad sequence file name");
			exit(1);
		}
		strcpy(seqname, param_arg[setinfile]);
	}

	if(*seqname != EOS) {
		profile_no = 0;
		nseqs = readseqs((sint)1);
		if(nseqs < 2) {
			if(nseqs < 0) cl_error_code = 2;
			else if(nseqs == 0) cl_error_code = 3;
			else cl_error_code = 4;
                	fprintf(stdout,
			"\nNo. of seqs. read = %d. No alignment!\n",(pint)nseqs);
			exit(cl_error_code);
		}
		for(i = 1; i<=nseqs; i++) 
			info("Sequence %d: %-*s   %6.d %s",
			(pint)i,max_names,names[i],(pint)seqlen_array[i],dnaflag?"bp":"aa");
		empty = FALSE;
		do_something = TRUE;
	}

	set_optional_param();

/*********************************************************/
/* Look for /profile1=file.ext  AND  /profile2=file2.ext */
/* You must give both file names OR neither.             */
/*********************************************************/

	if(setprofile1 != -1) {
		if(strlen(param_arg[setprofile1]) <= 0) {
			error("Bad profile 1 file name");
			exit(1);
		}
		strcpy(seqname, param_arg[setprofile1]);
		profile_no = 1;
		profile_input();
		if(nseqs <= 0) {
			if(nseqs<0) cl_error_code=2;
			else if(nseqs==0) cl_error_code=3;
			exit(cl_error_code);
		}
		strcpy(profile1_name,seqname);
	}

	if(setprofile2 != -1) {
		if(strlen(param_arg[setprofile2]) <= 0) {
			error("Bad profile 2 file name");
			exit(1);
		}
		if(profile1_empty) {
			error("Only 1 profile file (profile 2) specified.");
			exit(1);
		}
		strcpy(seqname, param_arg[setprofile2]);
		profile_no = 2;
		profile_input();
		if(nseqs > profile1_nseqs) 
			do_something = do_profile = TRUE;
		else {
			if(nseqs<0) cl_error_code=2;
			else if(nseqs==0) cl_error_code=3;
			error("No sequences read from profile 2");
			exit(cl_error_code);
		}
		strcpy(profile2_name,seqname);
	}

/*************************************************************************/
/* Look for /tree or /bootstrap or /align or /usetree ******************/
/*************************************************************************/

	if (setbatch != -1)
		interactive=FALSE;

	if (setinteractive != -1)
		interactive=TRUE;

	if (interactive) {
		settree = -1;
		setbootstrap = -1;
		setalign = -1;
		setusetree = -1;
		setusetree1 = -1;
		setusetree2 = -1;
		setnewtree = -1;
		setconvert = -1;
	}

	if(settree != -1 )
   {
		if(empty) {
			error("Cannot draw tree.  No input alignment file");
			exit(1);
		}
		else 
      {
			do_tree = TRUE;
      }
   }

	if(setbootstrap != -1)
   {
		if(empty) {
			error("Cannot bootstrap tree. No input alignment file");
			exit(1);
		}
		else {
			temp = 0;
			if(param_arg[setbootstrap] != NULL)
				 if (sscanf(param_arg[setbootstrap],"%d",&temp)!=1) 
             {
                fprintf(stdout,"Bad option for /bootstrap (must be integer)\n");
                temp = 0;
             };
			if(temp > 0)          
         {
            boot_ntrials = temp;
         }
			do_boot = TRUE;
		}
   }

	if(setalign != -1)
   {
		if(empty) {
			error("Cannot align sequences.  No input file");
			exit(1);
		}
		else 
      {
			do_align = TRUE;
      }
   }

	if(setconvert != -1)
   {
		if(empty) {
			error("Cannot convert sequences.  No input file");
			exit(1);
		}
		else 
      {
			do_convert = TRUE;
      }
   }
 
	if(setusetree != -1)
   {
		if(empty) {
			error("Cannot align sequences.  No input file");
			exit(1);
		}
		else  {
		        if(strlen(param_arg[setusetree]) == 0) {
				error("Cannot align sequences.  No tree file specified");
				exit(1);
		        }
                        else {
			        strcpy(phylip_tree_name, param_arg[setusetree]);
		        }
		        use_tree_file = TRUE;
		        do_align_only = TRUE;
		}
   }

	if(setnewtree != -1)
   {
		if(empty) {
			error("Cannot align sequences.  No input file");
			exit(1);
		}
		else  {
		        if(strlen(param_arg[setnewtree]) == 0) {
				error("Cannot align sequences.  No tree file specified");
				exit(1);
		        }
                        else {
			        strcpy(phylip_tree_name, param_arg[setnewtree]);
		        }
		    new_tree_file = TRUE;
			do_tree_only = TRUE;
		}
   }
 
	if(setusetree1 != -1)
   {
		if(profile1_empty) {
			error("Cannot align profiles.  No input file");
			exit(1);
		}
		else if(profile_type == SEQUENCE) {
			error("Invalid option /usetree1.");
			exit(1);
		}
		else  {
		        if(strlen(param_arg[setusetree1]) == 0) {
				error("Cannot align profiles.  No tree file specified");
				exit(1);
		        }
                        else {
			        strcpy(p1_tree_name, param_arg[setusetree1]);
		        }
		        use_tree1_file = TRUE;
		        do_align_only = TRUE;
		}
   }

	if(setnewtree1 != -1)
   {
		if(profile1_empty) {
			error("Cannot align profiles.  No input file");
			exit(1);
		}
		else if(profile_type == SEQUENCE) {
			error("Invalid option /newtree1.");
			exit(1);
		}
		else  {
		        if(strlen(param_arg[setnewtree1]) == 0) {
				error("Cannot align profiles.  No tree file specified");
				exit(1);
		        }
                        else {
			        strcpy(p1_tree_name, param_arg[setnewtree1]);
		        }
		    new_tree1_file = TRUE;
		}
   }
 
	if(setusetree2 != -1)
   {
		if(profile2_empty) {
			error("Cannot align profiles.  No input file");
			exit(1);
		}
		else if(profile_type == SEQUENCE) {
			error("Invalid option /usetree2.");
			exit(1);
		}
		else  {
		        if(strlen(param_arg[setusetree2]) == 0) {
				error("Cannot align profiles.  No tree file specified");
				exit(1);
		        }
                        else {
			        strcpy(p2_tree_name, param_arg[setusetree2]);
		        }
		        use_tree2_file = TRUE;
		        do_align_only = TRUE;
		}
   }

	if(setnewtree2 != -1)
   {
		if(profile2_empty) {
			error("Cannot align profiles.  No input file");
			exit(1);
		}
		else if(profile_type == SEQUENCE) {
			error("Invalid option /newtree2.");
			exit(1);
		}
		else  {
		        if(strlen(param_arg[setnewtree2]) == 0) {
				error("Cannot align profiles.  No tree file specified");
				exit(1);
		        }
                        else {
			        strcpy(p2_tree_name, param_arg[setnewtree2]);
		        }
		    new_tree2_file = TRUE;
		}
   }
 

	if( (!do_tree) && (!do_boot) && (!empty) && (!do_profile) && (!do_align_only) && (!do_tree_only) && (!do_convert)) 
		do_align = TRUE;

/*** ? /quicktree  */
        if(setquicktree != -1)
		quick_pairalign = TRUE;

	if(dnaflag) {
		gap_open   = dna_gap_open;
		gap_extend = dna_gap_extend;
		pw_go_penalty  = dna_pw_go_penalty;
		pw_ge_penalty  = dna_pw_ge_penalty;
                ktup       = dna_ktup;
                window     = dna_window;
                signif     = dna_signif;
                wind_gap   = dna_wind_gap;

	}
	else {
		gap_open   = prot_gap_open;
		gap_extend = prot_gap_extend;
		pw_go_penalty  = prot_pw_go_penalty;
		pw_ge_penalty  = prot_pw_ge_penalty;
                ktup       = prot_ktup;
                window     = prot_window;
                signif     = prot_signif;
                wind_gap   = prot_wind_gap;
	}
	
	if(interactive) {
		if (!xmenus) usemenu = TRUE;
		return;
	}


	if(!do_something) {
		error("No input file(s) specified");
		exit(1);
	}




/****************************************************************************/
/* Now do whatever has been requested ***************************************/
/****************************************************************************/


	if(do_profile) {
		if (profile_type == PROFILE) profile_align(p1_tree_name,p2_tree_name);
		else new_sequence_align(phylip_tree_name);
	}

	else if(do_align)
		align(phylip_tree_name);

        else if(do_convert) {
                get_path(seqname,path);
                if(!open_alignment_output(path)) exit(1);
                create_alignment_output(1,nseqs);
        }

        else if (do_align_only)
                get_tree(phylip_tree_name);

	else if(do_tree_only)
		make_tree(phylip_tree_name);

	else if(do_tree)
		phylogenetic_tree(phylip_tree_name,clustal_tree_name,dist_tree_name,nexus_tree_name,pim_name);

	else if(do_boot)
		bootstrap_tree(phylip_tree_name,clustal_tree_name,nexus_tree_name);

	fprintf(stdout,"\n");
	exit(0);

/*******whew!***now*go*home****/
}


Boolean user_mat(char *str, short *mat, short *xref)
{
        sint maxres;

        FILE *infile;

        if(usemenu)
                getstr("Enter name of the matrix file",lin2, MAXLINE);
        else
                strcpy(lin2,str);

        if(*lin2 == EOS) return FALSE;

        if((infile=fopen(lin2,"r"))==NULL) {
                error("Cannot find matrix file [%s]",lin2);
                return FALSE;
        }

	strcpy(str, lin2);

	maxres = read_user_matrix(str, mat, xref);
        if (maxres <= 0) return FALSE;

	return TRUE;
}

Boolean user_mat_series(char *str, short *mat, short *xref)
{
        sint maxres;

        FILE *infile;

        if(usemenu)
                getstr("Enter name of the matrix file",lin2, MAXLINE);
        else
                strcpy(lin2,str);

        if(*lin2 == EOS) return FALSE;

        if((infile=fopen(lin2,"r"))==NULL) {
                error("Cannot find matrix file [%s]",lin2);
                return FALSE;
        }

	strcpy(str, lin2);

	maxres = read_matrix_series(str, mat, xref);
        if (maxres <= 0) return FALSE;

	return TRUE;
}






sint seq_input(Boolean append)
{
        sint i;
	sint local_nseqs;

	if(usemenu) {
fprintf(stdout,"\n\nSequences should all be in 1 file.\n"); 
fprintf(stdout,"\n7 formats accepted: \n");
fprintf(stdout,
"NBRF/PIR, EMBL/SwissProt, Pearson (Fasta), GDE, Clustal, GCG/MSF, RSF.\n\n\n");
/*fprintf(stdout,
"\nGCG users should use TOPIR to convert their sequence files before use.\n\n\n");*/
	}

       if (append)
          local_nseqs = readseqs(nseqs+(sint)1);
       else
          local_nseqs = readseqs((sint)1);  /*  1 is the first seq to be read */
       if(local_nseqs < 0)               /* file could not be opened */
           { 
		return local_nseqs;
           }
       else if(local_nseqs == 0)         /* no sequences */
           {
	       error("No sequences in file!  Bad format?");
               return local_nseqs;
           }
       else 
           {
	   struct_penalties1 = struct_penalties2 = NONE;
	   if (sec_struct_mask1 != NULL) sec_struct_mask1=ckfree(sec_struct_mask1);
	   if (sec_struct_mask2 != NULL) sec_struct_mask2=ckfree(sec_struct_mask2);
	   if (gap_penalty_mask1 != NULL) gap_penalty_mask1=ckfree(gap_penalty_mask1);
	   if (gap_penalty_mask2 != NULL) gap_penalty_mask2=ckfree(gap_penalty_mask2);
	   if (ss_name1 != NULL) ss_name1=ckfree(ss_name1);
	   if (ss_name2 != NULL) ss_name2=ckfree(ss_name2);
	   
		if(append) nseqs+=local_nseqs;
		else nseqs=local_nseqs;
		info("Sequences assumed to be %s",
			dnaflag?"DNA":"PROTEIN");
		if (usemenu) {
			fprintf(stdout,"\n\n");
                	for(i=1; i<=nseqs; i++) {
/* DES                         fprintf(stdout,"%s: = ",names[i]); */
                        	info("Sequence %d: %-*s   %6.d %s",
                        	(pint)i,max_names,names[i],(pint)seqlen_array[i],dnaflag?"bp":"aa");
                	}	
                }	
			if(dnaflag) {
				gap_open   = dna_gap_open;
				gap_extend = dna_gap_extend;
			}
			else {
				gap_open   = prot_gap_open;
				gap_extend = prot_gap_extend;
			}
			empty=FALSE;
	   }
	return local_nseqs;	
}







sint profile_input(void)   /* read a profile   */
{                                           /* profile_no is 1 or 2  */
        sint local_nseqs, i;
	
        if(profile_no == 2 && profile1_empty) 
           {
             error("You must read in profile number 1 first");
             return 0;
           }

    if(profile_no == 1)     /* for the 1st profile */
      {
       local_nseqs = readseqs((sint)1); /* (1) means 1st seq to be read = no. 1 */
       if(local_nseqs < 0)               /* file could not be opened */
           { 
		return local_nseqs;
           }
       else if(local_nseqs == 0)         /* no sequences  */
           {
	       error("No sequences in file!  Bad format?");
		return local_nseqs;
           }
       else if (local_nseqs > 0)
           { 				/* success; found some seqs. */
		struct_penalties1 = NONE;
		if (sec_struct_mask1 != NULL) sec_struct_mask1=ckfree(sec_struct_mask1);
		if (gap_penalty_mask1 != NULL) gap_penalty_mask1=ckfree(gap_penalty_mask1);
		if (ss_name1 != NULL) ss_name1=ckfree(ss_name1);
                if (struct_penalties != NONE) /* feature table / mask in alignment */
                	{
					struct_penalties1 = struct_penalties;
					if (struct_penalties == SECST) {
						sec_struct_mask1 = (char *)ckalloc((max_aln_length) * sizeof (char));
						for (i=0;i<max_aln_length;i++)
							sec_struct_mask1[i] = sec_struct_mask[i];
					}
					gap_penalty_mask1 = (char *)ckalloc((max_aln_length) * sizeof (char));
					for (i=0;i<max_aln_length;i++)
						gap_penalty_mask1[i] = gap_penalty_mask[i];
        				ss_name1 = (char *)ckalloc( (MAXNAMES+1) * sizeof (char));

					strcpy(ss_name1,ss_name);
if (debug>0) {
for (i=0;i<seqlen_array[1];i++)
	fprintf(stdout,"%c",gap_penalty_mask1[i]);
fprintf(stdout,"\n");
}
					}
                nseqs = profile1_nseqs = local_nseqs;
				info("No. of seqs=%d",(pint)nseqs);
				profile1_empty=FALSE;
				profile2_empty=TRUE;
	   }
      }
    else
      {			        /* first seq to be read = profile1_nseqs + 1 */
       local_nseqs = readseqs(profile1_nseqs+(sint)1); 
       if(local_nseqs < 0)               /* file could not be opened */
           { 
		return local_nseqs;
           }
       else if(local_nseqs == 0)         /* no sequences */
           {
	       error("No sequences in file!  Bad format?");
		return local_nseqs;
           }
       else if(local_nseqs > 0)
           {
		struct_penalties2 = NONE;
		if (sec_struct_mask2 != NULL) sec_struct_mask2=ckfree(sec_struct_mask2);
		if (gap_penalty_mask2 != NULL) gap_penalty_mask2=ckfree(gap_penalty_mask2);
		if (ss_name2 != NULL) ss_name2=ckfree(ss_name2);
                if (struct_penalties != NONE) /* feature table / mask in alignment */
                	{
					struct_penalties2 = struct_penalties;
					if (struct_penalties == SECST) {
						sec_struct_mask2 = (char *)ckalloc((max_aln_length) * sizeof (char));
						for (i=0;i<max_aln_length;i++)
							sec_struct_mask2[i] = sec_struct_mask[i];
					}
					gap_penalty_mask2 = (char *)ckalloc((max_aln_length) * sizeof (char));
					for (i=0;i<max_aln_length;i++)
						gap_penalty_mask2[i] = gap_penalty_mask[i];
        				ss_name2 = (char *)ckalloc( (MAXNAMES+1) * sizeof (char));
					strcpy(ss_name2,ss_name);
if (debug>0) {
for (i=0;i<seqlen_array[profile1_nseqs+1];i++)
	fprintf(stdout,"%c",gap_penalty_mask2[i]);
fprintf(stdout,"\n");
}
					}
				info("No. of seqs in profile=%d",(pint)local_nseqs);
                nseqs = profile1_nseqs + local_nseqs;
                info("Total no. of seqs     =%d",(pint)nseqs);
				profile2_empty=FALSE;
				empty = FALSE;
	   }

      }
	if (sec_struct_mask != NULL) sec_struct_mask=ckfree(sec_struct_mask);
	if (gap_penalty_mask != NULL) gap_penalty_mask=ckfree(gap_penalty_mask);
	if (ss_name != NULL) ss_name=ckfree(ss_name);

	if(local_nseqs<=0) return local_nseqs;
	
	info("Sequences assumed to be %s",
		dnaflag?"DNA":"PROTEIN");
	if (usemenu) fprintf(stdout,"\n\n");
        for(i=profile2_empty?1:profile1_nseqs+1; i<=nseqs; i++) {
                info("Sequence %d: %-*s   %6.d %s",
                   (pint)i,max_names,names[i],(pint)seqlen_array[i],dnaflag?"bp":"aa");
        }	
	if(dnaflag) {
		gap_open   = dna_gap_open;
		gap_extend = dna_gap_extend;
	}
	else {
		gap_open   = prot_gap_open;
		gap_extend = prot_gap_extend;
	}

	return nseqs;
}



static void calc_gap_penalty_mask(int prf_length, char *mask, char *gap_mask)
{
	int i,j;
	char *struct_mask;

	struct_mask = (char *)ckalloc((prf_length+1) * sizeof(char));
/*
    calculate the gap penalty mask from the secondary structures
*/
	i=0;
	while (i<prf_length) {
		if (tolower(mask[i]) == 'a' || mask[i] == '$') {
			for (j = -helix_end_plus; j<0; j++) {
				if ((i+j>=0) && (tolower(struct_mask[i+j]) != 'a')
				             && (tolower(struct_mask[i+j]) != 'b'))
					struct_mask[i+j] = 'a';
			}
			for (j = 0; j<helix_end_minus; j++) {
				if (i+j>=prf_length || (tolower(mask[i+j]) != 'a'
				                    && mask[i+j] != '$')) break;
				struct_mask[i+j] = 'a';
			}
			i += j;
			while (tolower(mask[i]) == 'a'
				                    || mask[i] == '$') {
				if (i>=prf_length) break;
				if (mask[i] == '$') {
					struct_mask[i] = 'A';
					i++;
					break;
				}
				else struct_mask[i] = mask[i];
				i++;
			}
			for (j = 0; j<helix_end_minus; j++) {
				if ((i-j-1>=0) && (tolower(mask[i-j-1]) == 'a'
				                    || mask[i-j-1] == '$'))
					struct_mask[i-j-1] = 'a';
			}
			for (j = 0; j<helix_end_plus; j++) {
				if (i+j>=prf_length) break;
				struct_mask[i+j] = 'a';
			}
		}
	 	else if (tolower(mask[i]) == 'b' || mask[i] == '%') {
			for (j = -strand_end_plus; j<0; j++) {
				if ((i+j>=0) && (tolower(struct_mask[i+j]) != 'a')
				             && (tolower(struct_mask[i+j]) != 'b'))
					struct_mask[i+j] = 'b';
			}
			for (j = 0; j<strand_end_minus; j++) {
				if (i+j>=prf_length || (tolower(mask[i+j]) != 'b'
				                    && mask[i+j] != '%')) break;
				struct_mask[i+j] = 'b';
			}
			i += j;
			while (tolower(mask[i]) == 'b'
				                    || mask[i] == '%') {
				if (i>=prf_length) break;
				if (mask[i] == '%') {
					struct_mask[i] = 'B';
					i++;
					break;
				}
				else struct_mask[i] = mask[i];
				i++;
			}
			for (j = 0; j<strand_end_minus; j++) {
				if ((i-j-1>=0) && (tolower(mask[i-j-1]) == 'b'
				                    || mask[i-j-1] == '%'))
				struct_mask[i-j-1] = 'b';
			}
			for (j = 0; j<strand_end_plus; j++) {
				if (i+j>=prf_length) break;
 				struct_mask[i+j] = 'b';
			}
		}
	else i++;
	}

	for(i=0;i<prf_length;i++) {
		switch (struct_mask[i]) {
			case 'A':
				gap_mask[i] = helix_penalty+'0';
				break;
			case 'a':
				gap_mask[i] = helix_end_penalty+'0';
				break;
			case 'B':
				gap_mask[i] = strand_penalty+'0';
				break;
			case 'b':
				gap_mask[i] = strand_end_penalty+'0';
				break;
			default:
				gap_mask[i] = loop_penalty+'0';
				break;
		}
	}

	struct_mask=ckfree(struct_mask);
	
}

void print_sec_struct_mask(int prf_length, char *mask, char *struct_mask)
{
	int i,j;

/*
    calculate the gap penalty mask from the secondary structures
*/
	i=0;
	while (i<prf_length) {
		if (tolower(mask[i]) == 'a' || mask[i] == '$') {
			for (j = 0; j<helix_end_minus; j++) {
				if (i+j>=prf_length || (tolower(mask[i+j]) != 'a'
				                    && mask[i+j] != '$')) break;
				struct_mask[i+j] = 'a';
			}
			i += j;
			while (tolower(mask[i]) == 'a'
				                    || mask[i] == '$') {
				if (i>=prf_length) break;
				if (mask[i] == '$') {
					struct_mask[i] = 'A';
					i++;
					break;
				}
				else struct_mask[i] = mask[i];
				i++;
			}
			for (j = 0; j<helix_end_minus; j++) {
				if ((i-j-1>=0) && (tolower(mask[i-j-1]) == 'a'
				                    || mask[i-j-1] == '$'))
					struct_mask[i-j-1] = 'a';
			}
		}
	 	else if (tolower(mask[i]) == 'b' || mask[i] == '%') {
			for (j = 0; j<strand_end_minus; j++) {
				if (i+j>=prf_length || (tolower(mask[i+j]) != 'b'
				                    && mask[i+j] != '%')) break;
				struct_mask[i+j] = 'b';
			}
			i += j;
			while (tolower(mask[i]) == 'b'
				                    || mask[i] == '%') {
				if (i>=prf_length) break;
				if (mask[i] == '%') {
					struct_mask[i] = 'B';
					i++;
					break;
				}
				else struct_mask[i] = mask[i];
				i++;
			}
			for (j = 0; j<strand_end_minus; j++) {
				if ((i-j-1>=0) && (tolower(mask[i-j-1]) == 'b'
				                    || mask[i-j-1] == '%'))
				struct_mask[i-j-1] = 'b';
			}
		}
	else i++;
	}
}



FILE *  open_output_file(char *prompt,      char *path, 
				char *file_name,   char *file_extension)
 
{	static char temp[FILENAMELEN+1];
	static char local_prompt[MAXLINE];
	FILE * file_handle;

/*	if (*file_name == EOS) {
*/		strcpy(file_name,path);
		strcat(file_name,file_extension);
/*	}
*/
	if(strcmp(file_name,seqname)==0) {
		warning("Output file name is the same as input file.");
		if (usemenu) {
			strcpy(local_prompt,"\n\nEnter new name to avoid overwriting ");
			strcat(local_prompt," [%s]: ");          
			fprintf(stdout,local_prompt,file_name);
			fgets(temp, FILENAMELEN, stdin);
//          gets(temp);
			if(*temp != EOS) strcpy(file_name,temp);
		}
	}
	else if (usemenu) {
		strcpy(local_prompt,prompt);
		strcat(local_prompt," [%s]: ");          
		fprintf(stdout,local_prompt,file_name);
		fgets(temp, FILENAMELEN, stdin);
//		gets(temp);
		if(*temp != EOS) strcpy(file_name,temp);
	}

#ifdef VMS
	if((file_handle=fopen(file_name,"w","rat=cr","rfm=var"))==NULL) {
#else
	if((file_handle=fopen(file_name,"w"))==NULL) {
#endif
		error("Cannot open output file [%s]",file_name);
		return NULL;
	}
	return file_handle;
}



FILE *  open_explicit_file(char *file_name)
{ 
	FILE * file_handle;

	if (*file_name == EOS) {
		error("Bad output file [%s]",file_name);
		return NULL;
	}
#ifdef VMS
	if((file_handle=fopen(file_name,"w","rat=cr","rfm=var"))==NULL) {
#else
	if((file_handle=fopen(file_name,"w"))==NULL) {
#endif
		error("Cannot open output file [%s]",file_name);
		return NULL;
	}
	return file_handle;
}


/* Ramu void */

void align(char *phylip_name)
{ 
	char path[FILENAMELEN+1];
	FILE *tree=0;
	sint count;
	
	if(empty && usemenu) {
		error("No sequences in memory. Load sequences first.");
		return;
	}

	   struct_penalties1 = struct_penalties2 = NONE;
	   if (sec_struct_mask1 != NULL) sec_struct_mask1=ckfree(sec_struct_mask1);
	   if (sec_struct_mask2 != NULL) sec_struct_mask2=ckfree(sec_struct_mask2);
	   if (gap_penalty_mask1 != NULL) gap_penalty_mask1=ckfree(gap_penalty_mask1);
	   if (gap_penalty_mask2 != NULL) gap_penalty_mask2=ckfree(gap_penalty_mask2);
	   if (ss_name1 != NULL) ss_name1=ckfree(ss_name1);
	   if (ss_name2 != NULL) ss_name2=ckfree(ss_name2);


        get_path(seqname,path);
/* DES DEBUG 
	fprintf(stdout,"\n\n Seqname = %s  \n Path = %s \n\n",seqname,path);
*/
	if(usemenu || !interactive) {
        	if(!open_alignment_output(path)) return;
	}

	if (nseqs >= 2) {

        	get_path(seqname,path);
        	if (phylip_name[0]!=EOS) {
                	if((tree = open_explicit_file(
                	phylip_name))==NULL) return;
        	}
        	else {
                 	if((tree = open_output_file(
                	"\nEnter name for new GUIDE TREE           file  ",path,
                	phylip_name,"dnd")) == NULL) return;
        	}
	}

	if (save_parameters) create_parameter_output();

	if(reset_alignments_new || reset_alignments_all) reset_align();

        info("Start of Pairwise alignments");
        info("Aligning...");
        if(dnaflag) {
                gap_open   = dna_gap_open;
                gap_extend = dna_gap_extend;
                pw_go_penalty  = dna_pw_go_penalty;
                pw_ge_penalty  = dna_pw_ge_penalty;
                ktup       = dna_ktup;
                window     = dna_window;
                signif     = dna_signif;
                wind_gap   = dna_wind_gap;

        }
        else {
                gap_open   = prot_gap_open;
                gap_extend = prot_gap_extend;
                pw_go_penalty  = prot_pw_go_penalty;
                pw_ge_penalty  = prot_pw_ge_penalty;
                ktup       = prot_ktup;
                window     = prot_window;
                signif     = prot_signif;
                wind_gap   = prot_wind_gap;

        }

        if (quick_pairalign)
           show_pair((sint)0,nseqs,(sint)0,nseqs);
        else
           pairalign((sint)0,nseqs,(sint)0,nseqs);

	if (nseqs >= 2) {

		guide_tree(tree,1,nseqs);
		info("Guide tree        file created:   [%s]",
                phylip_name);
	}

	
	count = malign((sint)0,phylip_name);
	
	if (count <= 0) return;

	if (usemenu) fprintf(stdout,"\n\n\n");
	
	create_alignment_output(1,nseqs);
        if (showaln && usemenu) show_aln();
	phylip_name[0]=EOS;
	return ;
}





void new_sequence_align(char *phylip_name)
{ 
	char path[FILENAMELEN+1];
	char tree_name[FILENAMELEN+1],temp[MAXLINE+1];
	Boolean use_tree;
	FILE *tree=0;
	sint i,j,count;
	float dscore;
	Boolean save_ss2;
	
	if(profile1_empty && usemenu) {
		error("No profile in memory. Input 1st profile first.");
		return;
	}

	if(profile2_empty && usemenu) {
		error("No sequences in memory. Input sequences first.");
		return;
	}

        get_path(profile2_name,path);

        if(usemenu || !interactive) {
        	if(!open_alignment_output(path)) return;
	}

	new_seq = profile1_nseqs+1;

/* check for secondary structure information for list of sequences */

	save_ss2 = use_ss2;
	if (struct_penalties2 != NONE && use_ss2 == TRUE && (nseqs - profile1_nseqs >
1)) {
		if (struct_penalties2 == SECST) 
			warning("Warning: ignoring secondary structure for a list of sequences");
		else if (struct_penalties2 == GMASK)
			warning("Warning: ignoring gap penalty mask for a list of sequences");
		use_ss2 = FALSE;
	}

	for (i=1;i<=new_seq;i++) {
     		for (j=i+1;j<=new_seq;j++) {
       			dscore = countid(i,j);
       			tmat[i][j] = ((double)100.0 - (double)dscore)/(double)100.0;
       			tmat[j][i] = tmat[i][j];
     		}
   	}

	tree_name[0] = EOS;
	use_tree = FALSE;
	if (nseqs >= 2) {
		if (check_tree && usemenu) {
			strcpy(tree_name,path);
			strcat(tree_name,"dnd");
#ifdef VMS
        	if((tree=fopen(tree_name,"r","rat=cr","rfm=var"))!=NULL) {
#else
        	if((tree=fopen(tree_name,"r"))!=NULL) {
#endif
		if (usemenu)
            	fprintf(stdout,"\nUse the existing GUIDE TREE file,  %s  (y/n) ? [y]: ",
                                           tree_name);
			       fgets(temp, MAXLINE, stdin);
//                gets(temp);
                if(*temp != 'n' && *temp != 'N') {
                    strcpy(phylip_name,tree_name);
                    use_tree = TRUE;
                }
                fclose(tree);
        	}
		}
		else if (!usemenu && use_tree_file) {
			use_tree = TRUE;
		}
	}
	
	if (save_parameters) create_parameter_output();

	if(reset_alignments_new || reset_alignments_all) {
/*
		reset_prf1();
*/
		reset_prf2();
	}
	else fix_gaps();

	if (struct_penalties1 == SECST)

		calc_gap_penalty_mask(seqlen_array[1],sec_struct_mask1,gap_penalty_mask1);

	if (struct_penalties2 == SECST)

calc_gap_penalty_mask(seqlen_array[profile1_nseqs+1],sec_struct_mask2,gap_penalty_mask2);


/* create the new tree file, if necessary */

	if (use_tree == FALSE) {

		if (nseqs >= 2) {
        		get_path(profile2_name,path);
        		if (phylip_name[0]!=EOS) {
                		if((tree = open_explicit_file(
                		phylip_name))==NULL) return;
        		}
        		else {
                 		if((tree = open_output_file(
                		"\nEnter name for new GUIDE TREE           file  ",path,
                		phylip_name,"dnd")) == NULL) return;
        		}
		}
        info("Start of Pairwise alignments");
        info("Aligning...");
        if(dnaflag) {
                gap_open   = dna_gap_open;
                gap_extend = dna_gap_extend;
                pw_go_penalty  = dna_pw_go_penalty;
                pw_ge_penalty  = dna_pw_ge_penalty;
                ktup       = dna_ktup;
                window     = dna_window;
                signif     = dna_signif;
                wind_gap   = dna_wind_gap;

        }
        else {
                gap_open   = prot_gap_open;
                gap_extend = prot_gap_extend;
                pw_go_penalty  = prot_pw_go_penalty;
                pw_ge_penalty  = prot_pw_ge_penalty;
                ktup       = prot_ktup;
                window     = prot_window;
                signif     = prot_signif;
                wind_gap   = prot_wind_gap;

        }

        if (quick_pairalign)
           show_pair((sint)0,nseqs,new_seq-2,nseqs);
        else
           pairalign((sint)0,nseqs,new_seq-2,nseqs);

		if (nseqs >= 2) {
			guide_tree(tree,1,nseqs);
			info("Guide tree        file created:   [%s]",
               		phylip_name);
		}
	}
	
	if (new_tree_file) return;

	count = seqalign(new_seq-2,phylip_name);
	
	use_ss2 = save_ss2;
	
	if (count <= 0) return;

	if (usemenu) fprintf(stdout,"\n\n\n");
	
	create_alignment_output(1,nseqs);
        if (showaln && usemenu) show_aln();

	phylip_name[0]=EOS;

}





void make_tree(char *phylip_name)
{
	char path[FILENAMELEN+1];
	FILE *tree;
	
	if(empty) {
		error("No sequences in memory. Load sequences first.");
		return;
	}

	   struct_penalties1 = struct_penalties2 = NONE;
	   if (sec_struct_mask1 != NULL) sec_struct_mask1=ckfree(sec_struct_mask1);
	   if (sec_struct_mask2 != NULL) sec_struct_mask2=ckfree(sec_struct_mask2);
	   if (gap_penalty_mask1 != NULL) gap_penalty_mask1=ckfree(gap_penalty_mask1);
	   if (gap_penalty_mask2 != NULL) gap_penalty_mask2=ckfree(gap_penalty_mask2);
	   if (ss_name1 != NULL) ss_name1=ckfree(ss_name1);
	   if (ss_name2 != NULL) ss_name2=ckfree(ss_name2);

	if(reset_alignments_new || reset_alignments_all) reset_align();

        get_path(seqname,path);

	if (nseqs < 2) {
		error("Less than 2 sequences in memory. Phylogenetic tree cannot be built.");
		return;
	}

	if (save_parameters) create_parameter_output();

	info("Start of Pairwise alignments");
	info("Aligning...");
        if(dnaflag) {
                gap_open   = dna_gap_open;
                gap_extend = dna_gap_extend;
                pw_go_penalty  = dna_pw_go_penalty;
                pw_ge_penalty  = dna_pw_ge_penalty;
                ktup       = dna_ktup;
                window     = dna_window;
                signif     = dna_signif;
                wind_gap   = dna_wind_gap;

        }
        else {
                gap_open   = prot_gap_open;
                gap_extend = prot_gap_extend;
                pw_go_penalty  = prot_pw_go_penalty;
                pw_ge_penalty  = prot_pw_ge_penalty;
                ktup       = prot_ktup;
                window     = prot_window;
                signif     = prot_signif;
                wind_gap   = prot_wind_gap;


        }
   
        if (quick_pairalign)
          show_pair((sint)0,nseqs,(sint)0,nseqs);
        else
          pairalign((sint)0,nseqs,(sint)0,nseqs);

	if (nseqs >= 2) {
        	get_path(seqname,path);
        	if (phylip_name[0]!=EOS) {
                	if((tree = open_explicit_file(
                	phylip_name))==NULL) return;
        	}
        	else {
                 	if((tree = open_output_file(
                	"\nEnter name for new GUIDE TREE           file  ",path,
                	phylip_name,"dnd")) == NULL) return;
        	}

		guide_tree(tree,1,nseqs);
		info("Guide tree        file created:   [%s]",
               	phylip_name);
	}
	
	if(reset_alignments_new || reset_alignments_all) reset_align();

	phylip_name[0]=EOS;
}









void get_tree(char *phylip_name)
{
	char path[FILENAMELEN+1],temp[MAXLINE+1];
	sint count;
	
	if(empty) {
		error("No sequences in memory. Load sequences first.");
		return;
	}
	   struct_penalties1 = struct_penalties2 = NONE;
	   if (sec_struct_mask1 != NULL) sec_struct_mask1=ckfree(sec_struct_mask1);
	   if (sec_struct_mask2 != NULL) sec_struct_mask2=ckfree(sec_struct_mask2);
	   if (gap_penalty_mask1 != NULL) gap_penalty_mask1=ckfree(gap_penalty_mask1);
	   if (gap_penalty_mask2 != NULL) gap_penalty_mask2=ckfree(gap_penalty_mask2);
	   if (ss_name1 != NULL) ss_name1=ckfree(ss_name1);
	   if (ss_name2 != NULL) ss_name2=ckfree(ss_name2);


        get_path(seqname,path);

        if(usemenu || !interactive) {
        	if(!open_alignment_output(path)) return;
	}

	if(reset_alignments_new || reset_alignments_all) reset_align();

        get_path(seqname,path);

        if (nseqs >= 2) {
          
        	if(usemenu) {
       			strcpy(phylip_name,path);
       			strcat(phylip_name,"dnd");

            fprintf(stdout,"\nEnter a name for the guide tree file [%s]: ",
                                           phylip_name);
			         fgets(temp, MAXLINE, stdin);
//                	gets(temp);
                	if(*temp != EOS)
                        	strcpy(phylip_name,temp);
        	}

        	if(usemenu || !interactive) {
#ifdef VMS
        		if((tree=fopen(phylip_name,"r","rat=cr","rfm=var"))==NULL) {
#else
        		if((tree=fopen(phylip_name,"r"))==NULL) {
#endif
                		error("Cannot open tree file [%s]",phylip_name);
                		return;
        		}
		}
	}
	else {
        	info("Start of Pairwise alignments");
        	info("Aligning...");
        	if(dnaflag) {
                	gap_open   = dna_gap_open;
                	gap_extend = dna_gap_extend;
                	pw_go_penalty  = dna_pw_go_penalty;
                	pw_ge_penalty  = dna_pw_ge_penalty;
                	ktup       = dna_ktup;
                	window     = dna_window;
                	signif     = dna_signif;
                	wind_gap   = dna_wind_gap;

        	}
        	else {
                	gap_open   = prot_gap_open;
                	gap_extend = prot_gap_extend;
                	pw_go_penalty  = prot_pw_go_penalty;
                	pw_ge_penalty  = prot_pw_ge_penalty;
                	ktup       = prot_ktup;
                	window     = prot_window;
                	signif     = prot_signif;
                	wind_gap   = prot_wind_gap;

        	}

            if (quick_pairalign)
                show_pair((sint)0,nseqs,(sint)0,nseqs);
            else
		   		pairalign((sint)0,nseqs,(sint)0,nseqs);
	}

	if (save_parameters) create_parameter_output();

	count = malign(0,phylip_name);
	if (count <= 0) return;

	if (usemenu) fprintf(stdout,"\n\n\n");

	create_alignment_output(1,nseqs);
        if (showaln && usemenu) show_aln();

	phylip_name[0]=EOS;
}



void profile_align(char *p1_tree_name,char *p2_tree_name)
{
	char path[FILENAMELEN+1];
	char tree_name[FILENAMELEN+1];
	char temp[MAXLINE+1];
	Boolean use_tree1,use_tree2;
	FILE *tree;
	sint count,i,j,dscore;
	
	if(profile1_empty || profile2_empty) {
		error("No sequences in memory. Load sequences first.");
		return;
	}

	get_path(profile1_name,path);
	
        if(usemenu || !interactive) {
        	if(!open_alignment_output(path)) return;
	}

	if(reset_alignments_new || reset_alignments_all) {
		reset_prf1();
		reset_prf2();
	}
	else fix_gaps();

	tree_name[0] = EOS;
	use_tree1 = FALSE;
	if (profile1_nseqs >= 2) {
		if (check_tree && usemenu) {
			strcpy(tree_name,path);
			strcat(tree_name,"dnd");
#ifdef VMS
        	if((tree=fopen(tree_name,"r","rat=cr","rfm=var"))!=NULL) {
#else
        	if((tree=fopen(tree_name,"r"))!=NULL) {
#endif
            	fprintf(stdout,"\nUse the existing GUIDE TREE file for Profile 1,  %s  (y/n) ? [y]: ",
                                           tree_name);
			       fgets(temp, MAXLINE, stdin);
//                gets(temp);
                if(*temp != 'n' && *temp != 'N') {
                    strcpy(p1_tree_name,tree_name);
                    use_tree1 = TRUE;
                }
                fclose(tree);
        	}
		}
		else if (!usemenu && use_tree1_file) {
			use_tree1 = TRUE;
		}
	}
	tree_name[0] = EOS;
	use_tree2 = FALSE;
	get_path(profile2_name,path);
	if (nseqs-profile1_nseqs >= 2) {
		if (check_tree && usemenu) {
			strcpy(tree_name,path);
			strcat(tree_name,"dnd");
#ifdef VMS
        	if((tree=fopen(tree_name,"r","rat=cr","rfm=var"))!=NULL) {
#else
        	if((tree=fopen(tree_name,"r"))!=NULL) {
#endif
            	fprintf(stdout,"\nUse the existing GUIDE TREE file for Profile 2,  %s  (y/n) ? [y]: ",
                                           tree_name);
			       fgets(temp, MAXLINE, stdin);
//                gets(temp);
                if(*temp != 'n' && *temp != 'N') {
                    strcpy(p2_tree_name,tree_name);
                    use_tree2 = TRUE;
                }
                fclose(tree);
        	}
		}
		else if (!usemenu && use_tree2_file) {
			use_tree2 = TRUE;
		}
	}
				
	if (save_parameters) create_parameter_output();

	if (struct_penalties1 == SECST)

		calc_gap_penalty_mask(seqlen_array[1],sec_struct_mask1,gap_penalty_mask1);

	if (struct_penalties2 == SECST)

		calc_gap_penalty_mask(seqlen_array[profile1_nseqs+1],sec_struct_mask2,gap_penalty_mask2);

	if (use_tree1 == FALSE)
		if (profile1_nseqs >= 2) {
                	for (i=1;i<=profile1_nseqs;i++) {
                        	for (j=i+1;j<=profile1_nseqs;j++) {
                                	dscore = countid(i,j);
                                	tmat[i][j] = (100.0 - dscore)/100.0;
                                	tmat[j][i] = tmat[i][j];
                        	}
                	}
        		get_path(profile1_name,path);
        		if (p1_tree_name[0]!=EOS) {
                		if((tree = open_explicit_file(p1_tree_name))==NULL) return;
        		}
        		else {
                 		if((tree = open_output_file(
                		"\nEnter name for new GUIDE TREE file for profile 1 ",path,
                		p1_tree_name,"dnd")) == NULL) return;
        		}

			guide_tree(tree,1,profile1_nseqs);
			info("Guide tree        file created:   [%s]",
               		p1_tree_name);
		}
	if (use_tree2 == FALSE)
		if(nseqs-profile1_nseqs >= 2) {
                	for (i=1+profile1_nseqs;i<=nseqs;i++) {
                        	for (j=i+1;j<=nseqs;j++) {
                                	dscore = countid(i,j);
                                	tmat[i][j] = (100.0 - dscore)/100.0;
                                	tmat[j][i] = tmat[i][j];
                        	}
                	}
        		if (p2_tree_name[0]!=EOS) {
                		if((tree = open_explicit_file(p2_tree_name))==NULL) return;
        		}
        		else {
        			get_path(profile2_name,path);
                 		if((tree = open_output_file(
                		"\nEnter name for new GUIDE TREE file for profile 2 ",path,
                		p2_tree_name,"dnd")) == NULL) return;
        		}
			guide_tree(tree,profile1_nseqs+1,nseqs-profile1_nseqs);
			info("Guide tree        file created:   [%s]",
               		p2_tree_name);
		}

	if (new_tree1_file || new_tree2_file) return;

/* do an initial alignment to get the pairwise identities between the two
profiles - used to set parameters for the final alignment */
	count = palign1();
	if (count == 0) return;

	reset_prf1();
	reset_prf2();

	count = palign2(p1_tree_name,p2_tree_name);

	if (count == 0) return;

	if(usemenu) fprintf(stdout,"\n\n\n");

	create_alignment_output(1,nseqs);
        if (showaln && usemenu) show_aln();

	p1_tree_name[0]=EOS;
	p2_tree_name[0]=EOS;
}






 typedef struct rangeNum {
   int start;
   int end;
 } rangeNum;
 

/**** ********************************************************************************
 *
 *
 *
 *   INPUT:  
 * 
 *   RETURNS:  the range objects with the from, to range for each seqs.
 *
 *             the best things is to couple this up with the seqnames
 *             structure (there is no struct for seqnames yet!)
 */


void fillrange(rangeNum *rnum, sint fres, sint len, sint fseq)
{  
  sint val;
  sint i,ii;
  sint j,slen;	

  char tmpName[FILENAMELEN+15];
  int istart =0;
  int iend = 0; /* to print sequence start-end with names */
  int found =0;
  int ngaps=0;
  int tmpStart=0; 
  int tmpEnd=0;
  int ntermgaps=0;
  int pregaps=0;
  int tmpk=0;
  int isRange=0;
  int formula =0;

  tmpName[0] = '\0';
  slen = 0;

  ii = fseq ;
  i = output_index[ii];
  if( (sscanf(names[i],"%[^/]/%d-%d",tmpName, &tmpStart, &tmpEnd) == 3)) {
    isRange = 1;
  }
  for(tmpk=1; tmpk<fres; tmpk++) { /* do this irrespective of above sscanf */
    val = seq_array[i][tmpk];
    if ((val < 0) || (val > max_aa)) { /*it is gap */
      pregaps++;
    }
  }
  for(j=fres; j<fres+len; j++) {
    val = seq_array[i][j];
    if((val == -3) || (val == 253))
      break;
    else if((val < 0) || (val > max_aa)) {
      /* residue = '-'; */
      ngaps++;
    }
    else {
      /* residue = amino_acid_codes[val]; */
      found = j;
    }
    if ( found && (istart == 0) ) {
      istart = found;
      ntermgaps = ngaps;
    }
    slen++;
  }
  if( seqRange) {
    printf("Name : %s ",names[i]);
    printf("\n  fres = %d ",fres);
    printf("   len = %d ",len);
    printf("\n  istart = %d ",istart);
    printf("\n  tmpStart = %d ",tmpStart);
    printf("\n  ngaps = %d ",ngaps);
    printf("\n  pregaps = %d ",pregaps);
    if (!isRange)
      formula = istart - pregaps;
    else
      formula = istart - pregaps +  ( tmpStart == 1 ? 0: tmpStart-1) ;

    printf("\n\nsuggestion  istart - pregaps + tmpStart - ntermgaps = %d - %d + %d - %d",istart,
	   pregaps,tmpStart,ntermgaps);
    printf(" formula %d ",formula);
  }
  else {
    printf("\n no range found .... strange,  istart = %d",istart);
    formula = 1;
  }
  if (pregaps == fres-1) /* all gaps -  now the conditions........ */ 
    formula = tmpStart ; /* keep the previous start... */
  formula = (formula <= 0) ? 1: formula;
  if (pregaps ==0 && tmpStart == 0) {
    formula = fres;
  }
  iend = formula + len - ngaps -1;

  rnum->start = formula;
  rnum->end = iend;
  printf("\n check... %s %d - %d",names[i],rnum->start,rnum->end);
  printf(" Done checking.........");
}


void fasta_out(FILE *fastaout, sint fres, sint len, sint fseq, sint lseq)
{

    char *seq, residue;
    sint val;
    sint i,ii;
    sint j,slen;	
    sint line_length;

    rangeNum  *rnum=0;  
/*    int tmpk; */

    seq = (char *)ckalloc((len+1) * sizeof(char)); 
    
    line_length=PAGEWIDTH-max_names;
    line_length=line_length-line_length % 10; /* round to a multiple of 10*/
    if (line_length > LINELENGTH) line_length=LINELENGTH;

    if(seqRange) {
      rnum = (struct rangeNum *) malloc(sizeof(struct rangeNum));
    }

    for(ii=fseq; ii<=lseq; ii++) {
      i = output_index[ii];
      slen = 0;
      for(j=fres; j<fres+len; j++) {
	val = seq_array[i][j];
	if((val == -3) || (val == 253))
	  break;
	else if((val < 0) || (val > max_aa)) {
	  residue = '-';
	}
	else {
	  residue = amino_acid_codes[val];
	}
	if (lowercase) 
	  seq[j-fres] = (char)tolower((int)residue);
	else
	  seq[j-fres] = residue;
	slen++;
      }
      fprintf(fastaout, ">%-s",nameonly(names[i]));
      if(seqRange) {
	fillrange(rnum,fres, len, ii);
	fprintf(fastaout,"/%d-%d",rnum->start, rnum->end);
      }
      fprintf(fastaout,"\n");
      for(j=1; j<=slen; j++) {
	fprintf(fastaout,"%c",toupper(seq[j-1]));
	if((j % line_length == 0) || (j == slen)) 
	  fprintf(fastaout,"\n");
      }
    }
    seq=ckfree((void *)seq);

    if(seqRange) 
      if (rnum) 
	free(rnum);
    /* just try and see 
    printf("\n Now....  calculating percentage identity....\n\n");
    calc_percidentity();*/

}


void clustal_out(FILE *clusout, sint fres, sint len, sint fseq, sint lseq)
{
    static char *seq1;
    static sint *seq_no;
    static sint *print_seq_no;
    char *ss_mask1=0, *ss_mask2=0;
    char  temp[MAXLINE];
    char c;
    sint val;
    sint ii,lv1,catident1[NUMRES],catident2[NUMRES],ident,chunks;
    sint i,j,k,l;
    sint pos,ptr;
    sint line_length;

    rangeNum *rnum=0;
    char tmpStr[FILENAMELEN+15];
/*    int tmpk; */

    /*
      stop doing this ...... opens duplicate files in VMS  DES
      fclose(clusout);
      if ((clusout=fopen(clustal_outname,"w")) == NULL)
      {
      fprintf(stdout,"Error opening %s\n",clustal_outfile);
      return;
      }
    */

    if(seqRange) {
      rnum = (struct rangeNum *) malloc(sizeof(struct rangeNum));
      if ( rnum ==NULL ) {
	printf("cannot alloc memory for rnum");
      }
    }

    seq_no = (sint *)ckalloc((nseqs+1) * sizeof(sint));
    print_seq_no = (sint *)ckalloc((nseqs+1) * sizeof(sint));
    for (i=fseq;i<=lseq;i++)
      {
	print_seq_no[i] = seq_no[i] = 0;
	for(j=1;j<fres;j++) {
	  val = seq_array[i][j];
	  if((val >=0) || (val <=max_aa)) seq_no[i]++;
	}
      }

    seq1 = (char *)ckalloc((max_aln_length+1) * sizeof(char));
    
    if (struct_penalties1 == SECST && use_ss1 == TRUE) {
      ss_mask1 = (char *)ckalloc((seqlen_array[1]+10) * sizeof(char));
      for (i=0;i<seqlen_array[1];i++)
	ss_mask1[i] = sec_struct_mask1[i];
      print_sec_struct_mask(seqlen_array[1],sec_struct_mask1,ss_mask1);
    }
    if (struct_penalties2 == SECST && use_ss2 == TRUE) {
      ss_mask2 = (char *)ckalloc((seqlen_array[profile1_nseqs+1]+10) * sizeof(char));
      for (i=0;i<seqlen_array[profile1_nseqs+1];i++)
	ss_mask2[i] = sec_struct_mask2[i];
      print_sec_struct_mask(seqlen_array[profile1_nseqs+1],sec_struct_mask2,ss_mask2);
    }
    
    fprintf(clusout,"CLUSTAL %s multiple sequence alignment\n\n",
	    revision_level);
    
    /* decide the line length for this alignment - maximum is LINELENGTH */
    line_length=PAGEWIDTH-max_names;
    line_length=line_length-line_length % 10; /* round to a multiple of 10*/
    if (line_length > LINELENGTH) line_length=LINELENGTH;
    
    chunks = len/line_length;
    if(len % line_length != 0)
      ++chunks;
    
    for(lv1=1;lv1<=chunks;++lv1) {
      pos = ((lv1-1)*line_length)+1;
      ptr = (len<pos+line_length-1) ? len : pos+line_length-1;
      
      fprintf(clusout,"\n");
      
      if (output_struct_penalties == 0 || output_struct_penalties == 2) {
	if (struct_penalties1 == SECST && use_ss1 == TRUE) {
	  for(i=pos;i<=ptr;++i) {
	    val=ss_mask1[i+fres-2];
	    if (val == gap_pos1 || val == gap_pos2)
	      temp[i-pos]='-';
	    else
	      temp[i-pos]=val;
	  }
	  temp[ptr-pos+1]=EOS;
	  if(seqRange) /*Ramu*/
	    fprintf(clusout,"!SS_%-*s  %s\n",max_names+15,ss_name1,temp);
	  else
	    fprintf(clusout,"!SS_%-*s  %s\n",max_names,ss_name1,temp);
	}
      }
      if (output_struct_penalties == 1 || output_struct_penalties == 2) {
	if (struct_penalties1 != NONE && use_ss1 == TRUE) {
	  for(i=pos;i<=ptr;++i) {
	    val=gap_penalty_mask1[i+fres-2];
	    if (val == gap_pos1 || val == gap_pos2)
	      temp[i-pos]='-';
	    else
	      temp[i-pos]=val;
	  }
	  temp[ptr-pos+1]=EOS;
	  fprintf(clusout,"!GM_%-*s  %s\n",max_names,ss_name1,temp);
	}
      }
      if (output_struct_penalties == 0 || output_struct_penalties == 2) {
	if (struct_penalties2 == SECST && use_ss2 == TRUE) {
	  for(i=pos;i<=ptr;++i) {
	    val=ss_mask2[i+fres-2];
	    if (val == gap_pos1 || val == gap_pos2)
	      temp[i-pos]='-';
	    else
	      temp[i-pos]=val;
	  }
	  temp[ptr-pos+1]=EOS;
	  if (seqRange )
	    fprintf(clusout,"!SS_%-*s  %s\n",max_names+15,ss_name2,temp);
	  else
	    fprintf(clusout,"!SS_%-*s  %s\n",max_names,ss_name2,temp);
	}
      }
      if (output_struct_penalties == 1 || output_struct_penalties == 2) {
	if (struct_penalties2 != NONE && use_ss2 == TRUE) {
	  for(i=pos;i<=ptr;++i) {
	    val=gap_penalty_mask2[i+fres-2];
	    if (val == gap_pos1 || val == gap_pos2)
	      temp[i-pos]='-';
	    else
	      temp[i-pos]=val;
	  }
	  temp[ptr-pos+1]=EOS;
	  fprintf(clusout,"!GM_%-*s  %s\n",max_names,ss_name2,temp);
	}
      }
      
      for(ii=fseq;ii<=lseq;++ii) {
	i=output_index[ii];
	print_seq_no[i] = 0;
	for(j=pos;j<=ptr;++j) {
	  if (j+fres-1<=seqlen_array[i])
	    val = seq_array[i][j+fres-1];
	  else val = -3;
	  if((val == -3) || (val == 253)) break;
	  else if((val < 0) || (val > max_aa)){
	    seq1[j]='-';
	  }
	  else {
	    seq1[j]=amino_acid_codes[val];
	    seq_no[i]++;
	    print_seq_no[i]=1;
	  } 
	}
	for(;j<=ptr;++j) seq1[j]='-';
	strncpy(temp,&seq1[pos],ptr-pos+1);
	temp[ptr-pos+1]=EOS;
	if (!seqRange) {
	  fprintf(clusout,"%-*s",max_names+5,names[i]); 
	}
	else {
	  fillrange(rnum,fres, len, ii);
	  sprintf(tmpStr,"%s/%d-%d", nameonly(names[i]), rnum->start, rnum->end);
	  fprintf(clusout,"%-*s",max_names+15,tmpStr);
	}
	fprintf(clusout," %s",temp);
	if (cl_seq_numbers && print_seq_no[i])
	  fprintf(clusout," %d",seq_no[i]);
	fprintf(clusout,"\n");
      }
      
      for(i=pos;i<=ptr;++i) {
	seq1[i]=' ';
	ident=0;
	for(j=1;res_cat1[j-1]!=NULL;j++) catident1[j-1] = 0;
	for(j=1;res_cat2[j-1]!=NULL;j++) catident2[j-1] = 0;
	for(j=fseq;j<=lseq;++j) {
	  if((seq_array[fseq][i+fres-1] >=0) && 
	     (seq_array[fseq][i+fres-1] <= max_aa)) {
	    if(seq_array[fseq][i+fres-1] == seq_array[j][i+fres-1])
	      ++ident;
	    for(k=1;res_cat1[k-1]!=NULL;k++) {
	      for(l=0;(c=res_cat1[k-1][l]);l++) {
		if (amino_acid_codes[(int)(seq_array[j][i+fres-1])]==c)
		  {
		    catident1[k-1]++;
		    break;
		  }
	      }
	    }
	    for(k=1;res_cat2[k-1]!=NULL;k++) {
	      for(l=0;(c=res_cat2[k-1][l]);l++) {
		if (amino_acid_codes[(int)(seq_array[j][i+fres-1])]==c)
		  {
		    catident2[k-1]++;
		    break;
		  }
	      }
	    }
	  }
	}
	if(ident==lseq-fseq+1)
	  seq1[i]='*';
	else if (!dnaflag) {
	  for(k=1;res_cat1[k-1]!=NULL;k++) {
	    if (catident1[k-1]==lseq-fseq+1) {
	      seq1[i]=':';
	      break;
	    }
	  }
	  if(seq1[i]==' ')
	    for(k=1;res_cat2[k-1]!=NULL;k++) {
	      if (catident2[k-1]==lseq-fseq+1) {
		seq1[i]='.';
		break;
	      }
	    }
	}
      }
      strncpy(temp,&seq1[pos],ptr-pos+1);
      temp[ptr-pos+1]=EOS;
      for(k=0;k<max_names+6;k++) fprintf(clusout," ");
      if(seqRange) /*<ramu>*/
	fprintf(clusout,"          "); /*</ramu>*/
      fprintf(clusout,"%s\n",temp);
    }
        
    seq1=ckfree((void *)seq1);
    if (struct_penalties1 == SECST && use_ss1 == TRUE) ckfree(ss_mask1);
    if (struct_penalties2 == SECST && use_ss2 == TRUE) ckfree(ss_mask2);
    /* DES	ckfree(output_index); */

    if(seqRange) 
      if (rnum) 
	free(rnum);
} 




void gcg_out(FILE *gcgout, sint fres, sint len, sint fseq, sint lseq)
{
  /*        static char *aacids = "XCSTPAGNDEQHRKMILVFYW";*/
  /*	static char *nbases = "XACGT";	*/
  char *seq, residue;
  sint val;
  sint *all_checks;
  sint i,ii,chunks,block;
  sint j,k,pos1,pos2;	
  long grand_checksum;
  
  /*<ramu>*/
  rangeNum *rnum=0;
  char tmpStr[FILENAMELEN+15];
/*  int tmpk; */

  if(seqRange) {
    rnum = (struct rangeNum *) malloc(sizeof(struct rangeNum));
    if ( rnum ==NULL ) {
      printf("cannot alloc memory for rnum");
    }
  }

  seq = (char *)ckalloc((max_aln_length+1) * sizeof(char));
  all_checks = (sint *)ckalloc((lseq+1) * sizeof(sint));
  
  for(i=fseq; i<=lseq; i++) {
    for(j=fres; j<=fres+len-1; j++) {
      val = seq_array[i][j];
      if((val == -3) || (val == 253)) break;
      else if((val < 0) || (val > max_aa))
	residue = '.';
      else {
	residue = amino_acid_codes[val];
      }
      seq[j-fres+1] = residue;
    }
    /* pad any short sequences with gaps, to make all sequences the same length */
    for(; j<=fres+len-1; j++) 
      seq[j-fres+1] = '.';
    all_checks[i] = SeqGCGCheckSum(seq+1, (int)len);
  }	
  
  grand_checksum = 0;
  for(i=1; i<=nseqs; i++) grand_checksum += all_checks[output_index[i]];
  grand_checksum = grand_checksum % 10000;
  fprintf(gcgout,"PileUp\n\n");
  fprintf(gcgout,"\n\n   MSF:%5d  Type: ",(pint)len);
  if(dnaflag)
    fprintf(gcgout,"N");
  else
    fprintf(gcgout,"P");
  fprintf(gcgout,"    Check:%6ld   .. \n\n", (long)grand_checksum);
  for(ii=fseq; ii<=lseq; ii++)  {
    i = output_index[ii];
    fprintf(gcgout,
	    " Name: %s oo  Len:%5d  Check:%6ld  Weight:  %.1f\n",
	    names[i],(pint)len,(long)all_checks[i],(float)seq_weight[i-1]*100.0/(float)INT_SCALE_FACTOR);
  }
  fprintf(gcgout,"\n//\n");  
  
  chunks = len/GCG_LINELENGTH;
  if(len % GCG_LINELENGTH != 0) ++chunks;
  
  for(block=1; block<=chunks; block++) {
    fprintf(gcgout,"\n\n");
    pos1 = ((block-1) * GCG_LINELENGTH) + 1;
    pos2 = (len<pos1+GCG_LINELENGTH-1)? len : pos1+GCG_LINELENGTH-1;
    for(ii=fseq; ii<=lseq; ii++) {
      i = output_index[ii];
      if (!seqRange) {
	fprintf(gcgout,"\n%-*s ",max_names+5,names[i]);
      }
      else {
	fillrange(rnum,fres, len, ii);
	sprintf(tmpStr,"%s/%d-%d",nameonly(names[i]),rnum->start,rnum->end);
	fprintf(gcgout,"\n%-*s",max_names+15,tmpStr);
      }
      for(j=pos1, k=1; j<=pos2; j++, k++) {
	/*
	  JULIE -
	  check for sint sequences - pad out with '.' characters to end of alignment
	*/
	if (j+fres-1<=seqlen_array[i])
	  val = seq_array[i][j+fres-1];
	else val = -3;
	if((val == -3) || (val == 253))
	  residue = '.';
	else if((val < 0) || (val > max_aa))
	  residue = '.';
	else {
	  residue = amino_acid_codes[val];
	}
	fprintf(gcgout,"%c",residue);
	if(j % 10 == 0) fprintf(gcgout," ");
      }
    }
  }
  /* DES	ckfree(output_index); */
  
  seq=ckfree((void *)seq);
  all_checks=ckfree((void *)all_checks);
  fprintf(gcgout,"\n\n");


  if(seqRange) if (rnum) free(rnum);
}


/* <Ramu> */
/************************************************************************
 *
 *
 *    Removes the sequence range from sequence name
 *
 *
 *    INPUT: Sequence name
 *           (e.g. finc_rat/1-200 )
 *
 *
 *    RETURNS:  pointer to string
 */

char *nameonly(char *s)
{
    static char tmp[FILENAMELEN+1];
    int i =0;

    while (*s != '/' && *s != '\0') {
	tmp[i++] = *s++;
    }
    tmp[i] = '\0';
    return &tmp[0];
}


int startFind(char *s)
{
    int i = 0;
    sint val;
    printf("\n Debug.....\n %s",s);

    while( *s ) {
	val = *s;
	if ( (val <0 ) || (val > max_aa)) {
	    i++;
	    *s++;
	    printf("%c",amino_acid_codes[val]);
	}
    }
    return i;
}

/*
void fasta_out(FILE *fastaout, sint fres, sint len, sint fseq, sint lseq)
{
	char residue;
	sint val;
	sint i,ii;
	sint j,k;	
	
	for(ii=fseq; ii<=lseq; ii++)  {
	    i = output_index[ii];
	    fprintf(fastaout,">%-s",names[i],len);
	    j = 1;
	    while(j<len) {
		if ( ! (j%80) ) {
		    fprintf(fastaout,"\n");
			}
		val = seq_array[i][j];
		if((val < 0) || (val > max_aa))
		    residue = '-';
		else {
		    residue = amino_acid_codes[val];
		}
		fprintf(fastaout,"%c",residue);
		j++;
	    }
	    fprintf(fastaout,"\n");
	}

}
*/

/* </Ramu> */

void nexus_out(FILE *nxsout, sint fres, sint len, sint fseq, sint lseq)
{
/*      static char *aacids = "XCSTPAGNDEQHRKMILVFYW";*/
/*		static char *nbases = "XACGT";	*/
  char residue;
  sint val;
  sint i,ii,chunks,block;	
  sint j,k,pos1,pos2;	
  

  /*<ramu>*/
  rangeNum *rnum=0;
  char tmpStr[FILENAMELEN+15];
/*  int tmpk; */

  if(seqRange) {
    rnum = (struct rangeNum *) malloc(sizeof(struct rangeNum));
    if ( rnum ==NULL ) {
      printf("cannot alloc memory for rnum");
    }
  }


  chunks = len/GCG_LINELENGTH;
  if(len % GCG_LINELENGTH != 0) ++chunks;
  
  fprintf(nxsout,"#NEXUS\n");
  fprintf(nxsout,"BEGIN DATA;\n");
  fprintf(nxsout,"dimensions ntax=%d nchar=%d;\n",(pint)nseqs,(pint)len);
  fprintf(nxsout,"format missing=?\n");
  fprintf(nxsout,"symbols=\"");
  for(i=0;i<=max_aa;i++)
    fprintf(nxsout,"%c",amino_acid_codes[i]);
  fprintf(nxsout,"\"\n");
  fprintf(nxsout,"interleave datatype=");
  fprintf(nxsout, dnaflag ? "DNA " : "PROTEIN ");
  fprintf(nxsout,"gap= -;\n");
  fprintf(nxsout,"\nmatrix");
  
  for(block=1; block<=chunks; block++) {
    pos1 = ((block-1) * GCG_LINELENGTH)+1;
    pos2 = (len<pos1+GCG_LINELENGTH-1)? len : pos1+GCG_LINELENGTH-1;
    for(ii=fseq; ii<=lseq; ii++)  {
      i = output_index[ii];
      if (!seqRange) {
	fprintf(nxsout,"\n%-*s ",max_names+1,names[i]);
      }
      else {
	fillrange(rnum,fres, len, ii);
	sprintf(tmpStr,"%s/%d-%d",nameonly(names[i]),rnum->start,rnum->end);
	fprintf(nxsout,"\n%-*s",max_names+15,tmpStr);
      }
      for(j=pos1, k=1; j<=pos2; j++, k++) {
	if (j+fres-1<=seqlen_array[i])
	  val = seq_array[i][j+fres-1];
	else val = -3;
	if((val == -3) || (val == 253))
	  break;
	else if((val < 0) || (val > max_aa))
	  residue = '-';
	else {
	  residue = amino_acid_codes[val];
	}
	fprintf(nxsout,"%c",residue);
      }
    }
    fprintf(nxsout,"\n");
  }
  fprintf(nxsout,";\nend;\n");
  /* DES	ckfree(output_index); */

  if(seqRange) if (rnum) free(rnum);

}




void phylip_out(FILE *phyout, sint fres, sint len, sint fseq, sint lseq)
{
/*      static char *aacids = "XCSTPAGNDEQHRKMILVFYW";*/
/*		static char *nbases = "XACGT";	*/
  char residue;
  sint val;
  sint i,ii,chunks,block;	
  sint j,k,pos1,pos2;	
  sint name_len;
  Boolean warn;
  char **snames;
  
  /*<ramu>*/
  rangeNum *rnum=0;
  char tmpStr[FILENAMELEN+15];
/*  int tmpk; */


  if(seqRange) {
    rnum = (struct rangeNum *) malloc(sizeof(struct rangeNum));
    if ( rnum ==NULL ) {
      printf("cannot alloc memory for rnum");      
    }
  }

  snames=(char **)ckalloc((lseq-fseq+2)*sizeof(char *));
  name_len=0;
  for(i=fseq; i<=lseq; i++)  {
    snames[i]=(char *)ckalloc((11)*sizeof(char));
    ii=strlen(names[i]);
    strncpy(snames[i],names[i],10);
    if(name_len<ii) name_len=ii;
  }
  if(name_len>10) {
    warn=FALSE;
    for(i=fseq; i<=lseq; i++)  {
      for(j=i+1;j<=lseq;j++) {
	if (strcmp(snames[i],snames[j]) == 0) 
	  warn=TRUE;
      }
    }
    if(warn)
      warning("Truncating sequence names to 10 characters for PHYLIP output.\n"
	      "Names in the PHYLIP format file are NOT unambiguous.");
    else
      warning("Truncating sequence names to 10 characters for PHYLIP output.");
  }
  
  
  chunks = len/GCG_LINELENGTH;
  if(len % GCG_LINELENGTH != 0) ++chunks;
  
  fprintf(phyout,"%6d %6d",(pint)nseqs,(pint)len);
  
  for(block=1; block<=chunks; block++) {
    pos1 = ((block-1) * GCG_LINELENGTH)+1;
    pos2 = (len<pos1+GCG_LINELENGTH-1)? len : pos1+GCG_LINELENGTH-1;
    for(ii=fseq; ii<=lseq; ii++)  {
      i = output_index[ii];
      if(block == 1)  {
	if(!seqRange) {
	  fprintf(phyout,"\n%-10s ",snames[i]);
	}
	else
	  {
	    fillrange(rnum,fres, len, ii);
	    sprintf(tmpStr,"%s/%d-%d",nameonly(names[i]),rnum->start,rnum->end);
	    fprintf(phyout,"\n%-*s",max_names+15,tmpStr);
	  }
      }
      else
	fprintf(phyout,"\n           ");
      for(j=pos1, k=1; j<=pos2; j++, k++) {
	if (j+fres-1<=seqlen_array[i])
	  val = seq_array[i][j+fres-1];
	else val = -3;
	if((val == -3) || (val == 253))
	  break;
	else if((val < 0) || (val > max_aa))
	  residue = '-';
	else {
	  residue = amino_acid_codes[val];
	}
	fprintf(phyout,"%c",residue);
	if(j % 10 == 0) fprintf(phyout," ");
      }
    }
    fprintf(phyout,"\n");
  }
  /* DES	ckfree(output_index); */
  
  for(i=fseq;i<=lseq;i++)
    ckfree(snames[i]);
  ckfree(snames);
  
  if(seqRange) if (rnum) free(rnum);

}





void nbrf_out(FILE *nbout, sint fres, sint len, sint fseq, sint lseq)
{
/*      static char *aacids = "XCSTPAGNDEQHRKMILVFYW";*/
/*		static char *nbases = "XACGT";	*/
	char *seq, residue;
	sint val;
	sint i,ii;
	sint j,slen;	
	sint line_length;


  /*<ramu>*/
  rangeNum *rnum=0;
  char tmpStr[FILENAMELEN+15];
/*  int tmpk; */

  if(seqRange) {
    rnum = (struct rangeNum *) malloc(sizeof(struct rangeNum));
    if ( rnum ==NULL ) {
      printf("cannot alloc memory for rnum");
    }
  }

  seq = (char *)ckalloc((max_aln_length+1) * sizeof(char));
  
  /* decide the line length for this alignment - maximum is LINELENGTH */
  line_length=PAGEWIDTH-max_names;
  line_length=line_length-line_length % 10; /* round to a multiple of 10*/
  if (line_length > LINELENGTH) line_length=LINELENGTH;
  
  for(ii=fseq; ii<=lseq; ii++) {
    i = output_index[ii];
    fprintf(nbout, dnaflag ? ">DL;" : ">P1;");
    if (!seqRange) {
      fprintf(nbout, "%s\n%s\n", names[i], titles[i]);
    }
    else {
      fillrange(rnum,fres, len, ii);
      sprintf(tmpStr,"%s/%d-%d",nameonly(names[i]),rnum->start,rnum->end);
      fprintf(nbout,"%s\n%s\n",tmpStr,titles[i]);
    }
    slen = 0;
    for(j=fres; j<fres+len; j++) {
      val = seq_array[i][j];
      if((val == -3) || (val == 253))
	break;
      else if((val < 0) || (val > max_aa))
	residue = '-';
      else {
	residue = amino_acid_codes[val];
      }
      seq[j-fres] = residue;
      slen++;
    }
    for(j=1; j<=slen; j++) {
      fprintf(nbout,"%c",seq[j-1]);
      if((j % line_length == 0) || (j == slen)) 
	fprintf(nbout,"\n");
    }
    fprintf(nbout,"*\n");
  }	
  /* DES	ckfree(output_index);  */
  
  seq=ckfree((void *)seq);

  if(seqRange) if (rnum) free(rnum);

}


void gde_out(FILE *gdeout, sint fres, sint len, sint fseq, sint lseq)
{
/*      static char *aacids = "XCSTPAGNDEQHRKMILVFYW";*/
/*		static char *nbases = "XACGT";	*/
	char *seq, residue;
	sint val;
	char *ss_mask1=0, *ss_mask2=0;
	sint i,ii;
	sint j,slen;	
	sint line_length;


  /*<ramu>*/
  rangeNum *rnum=0;
/*  char tmpStr[FILENAMELEN+15]; */
/*  int tmpk; */

  if(seqRange) {
    rnum = (struct rangeNum *) malloc(sizeof(struct rangeNum));
    if ( rnum ==NULL ) {
      printf("cannot alloc memory for rnum");
    }
  }

  seq = (char *)ckalloc((max_aln_length+1) * sizeof(char));
  
  /* decide the line length for this alignment - maximum is LINELENGTH */
  line_length=PAGEWIDTH-max_names;
  line_length=line_length-line_length % 10; /* round to a multiple of 10*/
  if (line_length > LINELENGTH) line_length=LINELENGTH;
  
  if (struct_penalties1 == SECST && use_ss1 == TRUE) {
    ss_mask1 = (char *)ckalloc((seqlen_array[1]+10) * sizeof(char));
    for (i=0;i<seqlen_array[1];i++)
      ss_mask1[i] = sec_struct_mask1[i];
    print_sec_struct_mask(seqlen_array[1],sec_struct_mask1,ss_mask1);
  }
  if (struct_penalties2 == SECST && use_ss2 == TRUE) {
    ss_mask2 = (char *)ckalloc((seqlen_array[profile1_nseqs+1]+10) *
			       sizeof(char));
    for (i=0;i<seqlen_array[profile1_nseqs+1];i++)
      ss_mask2[i] = sec_struct_mask2[i];
    print_sec_struct_mask(seqlen_array[profile1_nseqs+1],sec_struct_mask2,ss_mask2);  
  }

	
  for(ii=fseq; ii<=lseq; ii++) {
    i = output_index[ii];
    fprintf(gdeout, dnaflag ? "#" : "%%");
    if(!seqRange) {
      fprintf(gdeout, "%s\n", names[i]);
    }
    else {
      fillrange(rnum,fres, len, ii);
      fprintf(gdeout,"%s/%d-%d\n",nameonly(names[i]),rnum->start,rnum->end);
    }
    slen = 0;
    for(j=fres; j<fres+len; j++) {
      val = seq_array[i][j];
      if((val == -3) || (val == 253))
	break;
      else if((val < 0) || (val > max_aa))
	residue = '-';
      else {
	residue = amino_acid_codes[val];
      }
      if (lowercase)
	seq[j-fres] = (char)tolower((int)residue);
      else
	seq[j-fres] = residue;
      slen++;
    }
    for(j=1; j<=slen; j++) {
      fprintf(gdeout,"%c",seq[j-1]);
      if((j % line_length == 0) || (j == slen)) 
	fprintf(gdeout,"\n");
    }
  }
  /* DES	ckfree(output_index); */
  
  if (output_struct_penalties == 0 || output_struct_penalties == 2) {
    if (struct_penalties1 == SECST && use_ss1 == TRUE) {
      fprintf(gdeout,"\"SS_%-*s\n",max_names,ss_name1);
      for(i=fres; i<fres+len; i++) {
	val=ss_mask1[i-1];
	if (val == gap_pos1 || val == gap_pos2)
	  seq[i-fres]='-';
	else
	  seq[i-fres]=val;
      }
      seq[i-fres]=EOS;
      for(i=1; i<=len; i++) {
	fprintf(gdeout,"%c",seq[i-1]);
	if((i % line_length == 0) || (i == len)) 
	  fprintf(gdeout,"\n");
      }
    }
    
    if (struct_penalties2 == SECST && use_ss2 == TRUE) {
      fprintf(gdeout,"\"SS_%-*s\n",max_names,ss_name2);
      for(i=fres; i<fres+len; i++) {
	val=ss_mask2[i-1];
	if (val == gap_pos1 || val == gap_pos2)
	  seq[i-fres]='-';
	else
	  seq[i-fres]=val;
      }
      seq[i]=EOS;
      for(i=1; i<=len; i++) {
	fprintf(gdeout,"%c",seq[i-1]);
	if((i % line_length == 0) || (i == len)) 
	  fprintf(gdeout,"\n");
      }
    }
  }
  if (output_struct_penalties == 1 || output_struct_penalties == 2) {
    if (struct_penalties1 != NONE && use_ss1 == TRUE) {
      fprintf(gdeout,"\"GM_%-*s\n",max_names,ss_name1);
      for(i=fres; i<fres+len; i++) {
	val=gap_penalty_mask1[i-1];
	if (val == gap_pos1 || val == gap_pos2)
	  seq[i-fres]='-';
	else
	  seq[i-fres]=val;
      }
      seq[i]=EOS;
      for(i=1; i<=len; i++) {
	fprintf(gdeout,"%c",seq[i-1]);
	if((i % line_length == 0) || (i == len)) 
	  fprintf(gdeout,"\n");
      }
    }
    if (struct_penalties2 != NONE && use_ss2 == TRUE) {
      fprintf(gdeout,"\"GM_%-*s\n",max_names,ss_name2);
      for(i=fres; i<fres+len; i++) {
	val=gap_penalty_mask2[i-1];
	if (val == gap_pos1 || val == gap_pos2)
	  seq[i-fres]='-';
	else
	  seq[i-fres]=val;
      }
      seq[i]=EOS;
      for(i=1; i<=len; i++) {
	fprintf(gdeout,"%c",seq[i-1]);
	if((i % line_length == 0) || (i == len)) 
	  fprintf(gdeout,"\n");
      }
    }
  }
  
  if (struct_penalties1 == SECST && use_ss1 == TRUE) ckfree(ss_mask1);
  if (struct_penalties2 == SECST && use_ss2 == TRUE) ckfree(ss_mask2);
  seq=ckfree((void *)seq);
  

  if(seqRange) if (rnum) free(rnum);

}


Boolean open_alignment_output(char *path)
{

  if(!output_clustal && !output_nbrf && !output_gcg &&
     !output_phylip && !output_gde && !output_nexus && !output_fasta) {
    error("You must select an alignment output format");
    return FALSE;
  }
  
  if(output_clustal) 
  {
    if (outfile_name[0]!=EOS) {
      strcpy(clustal_outname,outfile_name);
      if((clustal_outfile = open_explicit_file(
					       clustal_outname))==NULL) return FALSE;
    }
    else {
      /* DES DEBUG 
	 fprintf(stdout,"\n\n path = %s\n clustal_outname = %s\n\n",
	 path,clustal_outname);
      */
      if((clustal_outfile = open_output_file(
					     "\nEnter a name for the CLUSTAL output file ",path,
					     clustal_outname,"aln"))==NULL) return FALSE;
      /* DES DEBUG 
	 fprintf(stdout,"\n\n path = %s\n clustal_outname = %s\n\n",
	 path,clustal_outname);
      */
    }
  }

  if(output_nbrf) 
  {
    if (outfile_name[0]!=EOS) {
      strcpy(nbrf_outname,outfile_name);
      if( (nbrf_outfile = open_explicit_file(nbrf_outname))==NULL) 
      {
         return FALSE;
      }
    }
    else
    {
      if((nbrf_outfile = open_output_file(
					  "\nEnter a name for the NBRF/PIR output file",path,
					  nbrf_outname,"pir"))==NULL) 
      {
         return FALSE;
      }
    }
  }
  if(output_gcg) 
  {
    if (outfile_name[0]!=EOS) {
      strcpy(gcg_outname,outfile_name);
      if((gcg_outfile = open_explicit_file( gcg_outname))==NULL) 
	return FALSE;
    }
    else
    {
      if((gcg_outfile = open_output_file(
					 "\nEnter a name for the GCG output file     ",path,
					 gcg_outname,"msf"))==NULL) 
      {
         return FALSE;
      }
    }
  }
  if(output_phylip) 
  {
    if (outfile_name[0]!=EOS) {
      strcpy(phylip_outname,outfile_name);
      if((phylip_outfile = open_explicit_file(
					      phylip_outname))==NULL) return FALSE;
    }
    else
    {
      if((phylip_outfile = open_output_file(
					    "\nEnter a name for the PHYLIP output file  ",path,
					    phylip_outname,"phy"))==NULL) return FALSE;
    }
  }
  if(output_gde) 
  {
    if (outfile_name[0]!=EOS) {
      strcpy(gde_outname,outfile_name);
      if((gde_outfile = open_explicit_file(
					   gde_outname))==NULL) return FALSE;
    }
    else
    {
      if((gde_outfile = open_output_file(
					 "\nEnter a name for the GDE output file     ",path,
					 gde_outname,"gde"))==NULL) return FALSE;
    }
  }
  if(output_nexus) 
  {
    if (outfile_name[0]!=EOS) {
      strcpy(nexus_outname,outfile_name);
      if((nexus_outfile = open_explicit_file(
					     nexus_outname))==NULL) return FALSE;
    }
    else
    {
      if((nexus_outfile = open_output_file(
					   "\nEnter a name for the NEXUS output file   ",path,
					   nexus_outname,"nxs"))==NULL) return FALSE;
    }
  
  }
  /* Ramu */
  if(output_fasta) 
  {
    if (outfile_name[0]!=EOS) {
      strcpy(fasta_outname,outfile_name);
      if((fasta_outfile = open_explicit_file(
					     fasta_outname))==NULL) return FALSE;
    }
    else
    {
      if((fasta_outfile = open_output_file(
					   "\nEnter a name for the Fasta output file   ",path,
					   fasta_outname,"fasta"))==NULL) return FALSE;
    }
  }
  
  return TRUE;
}




void create_alignment_output(sint fseq, sint lseq)
{
  sint i,length;
  
  sint ifres; /* starting sequence range - Ramu          */
  sint ilres; /* ending sequence range */
  char ignore; 
  Boolean rangeOK;

  length=0;

  ifres = 1;
  ilres = 0;
  rangeOK = FALSE;
  for (i=fseq;i<=lseq;i++)
    if (length < seqlen_array[i])
      length = seqlen_array[i];
  ilres=length;


  if (setrange != -1 ) {
    /* printf("\n ==================== seqRange is set \n"); */
    if ( sscanf(param_arg[setrange],"%d%[ :,-]%d",&ifres,&ignore,&ilres) !=3) {
      info("seqrange numers are not set properly, using default....");
      ifres = 1;
      ilres = length;
    }
    else
      rangeOK = TRUE;
  }
  if ( rangeOK && ilres > length ) {
    ilres = length; /* if asked for more, set the limit, Ramui */
    info("Seqrange %d is more than the %d  setting it to %d ",ilres,length,length);
  }

  /* if (usemenu) info("Consensus length = %d",(pint)length);*/

  if (usemenu) info("Consensus length = %d",(pint)ilres);  /* Ramu */

  /*
  printf("\n creating output ....... normal.... setrange = %d \n",setrange);
  printf(" ---------> %d   %d \n\n ",ifres,ilres);
  printf(" ---------> %d  \n\n ",length);
  */
  
  if(output_clustal) {
    clustal_out(clustal_outfile, ifres, ilres,  fseq, lseq);
		fclose(clustal_outfile);
		info("CLUSTAL-Alignment file created  [%s]",clustal_outname);
  }
  if(output_nbrf)  {
    nbrf_out(nbrf_outfile, ifres, ilres, /*1, length */ fseq, lseq);
    fclose(nbrf_outfile);
    info("NBRF/PIR-Alignment file created [%s]",nbrf_outname);
  }
  if(output_gcg)  {
    gcg_out(gcg_outfile, ifres, ilres, /*1, length */ fseq, lseq);
    fclose(gcg_outfile);
    info("GCG-Alignment file created      [%s]",gcg_outname);
  }
  if(output_phylip)  {
    phylip_out(phylip_outfile, ifres, ilres, /*1, length */ fseq, lseq);
    fclose(phylip_outfile);
    info("PHYLIP-Alignment file created   [%s]",phylip_outname);
  }
  if(output_gde)  {
    gde_out(gde_outfile, ifres, ilres /*1, length */, fseq, lseq);
    fclose(gde_outfile);
    info("GDE-Alignment file created      [%s]",gde_outname);
  }
  if(output_nexus)  {
    nexus_out(nexus_outfile, ifres, ilres /*1, length */, fseq, lseq);
    fclose(nexus_outfile);
    info("NEXUS-Alignment file created    [%s]",nexus_outname);
  }
  /*  Ramu */
  if(output_fasta)  {
    fasta_out(fasta_outfile, ifres, ilres /*1, length */, fseq, lseq);
    fclose(fasta_outfile);
    info("Fasta-Alignment file created    [%s]",fasta_outname);
  }
}


static void reset_align(void)   /* remove gaps from older alignments (code =
				   gap_pos1) */
{		      				/* EXCEPT for gaps that were INPUT with the seqs.*/
  register sint sl;   		     /* which have  code = gap_pos2  */
  sint i,j;
  
  for(i=1;i<=nseqs;++i) {
    sl=0;
    for(j=1;j<=seqlen_array[i];++j) {
      if(seq_array[i][j] == gap_pos1 && 
	 ( reset_alignments_new ||
	   reset_alignments_all)) continue;
      if(seq_array[i][j] == gap_pos2 && (reset_alignments_all)) continue;
      ++sl;
      seq_array[i][sl]=seq_array[i][j];
    }
    seqlen_array[i]=sl;
  }
}



static void reset_prf1(void)   /* remove gaps from older alignments (code =
				  gap_pos1) */
{		      				/* EXCEPT for gaps that were INPUT with the seqs.*/
  register sint sl;   		     /* which have  code = gap_pos2  */
  sint i,j;
  
  if (struct_penalties1 != NONE) {
    sl=0;
    for (j=0;j<seqlen_array[1];++j) {
      if (gap_penalty_mask1[j] == gap_pos1 && (reset_alignments_new ||
					       reset_alignments_all)) continue;
      if (gap_penalty_mask1[j] == gap_pos2 && (reset_alignments_all)) continue;
      gap_penalty_mask1[sl]=gap_penalty_mask1[j];
      ++sl;
    }
  }
  
  if (struct_penalties1 == SECST) {
    sl=0;
    for (j=0;j<seqlen_array[1];++j) {
      if (sec_struct_mask1[j] == gap_pos1 && (reset_alignments_new ||
					      reset_alignments_all)) continue;
      if (sec_struct_mask1[j] == gap_pos2 && (reset_alignments_all)) continue;
      sec_struct_mask1[sl]=sec_struct_mask1[j];
      ++sl;
    }
  }
  
  for(i=1;i<=profile1_nseqs;++i) {
    sl=0;
    for(j=1;j<=seqlen_array[i];++j) {
      if(seq_array[i][j] == gap_pos1 && (reset_alignments_new ||
					 reset_alignments_all)) continue;
      if(seq_array[i][j] == gap_pos2 && (reset_alignments_all)) continue;
      ++sl;
      seq_array[i][sl]=seq_array[i][j];
    }
    seqlen_array[i]=sl;
  }
  
  
}



static void reset_prf2(void)   /* remove gaps from older alignments (code =
				  gap_pos1) */
{		      				/* EXCEPT for gaps that were INPUT with the seqs.*/
  register sint sl;   		     /* which have  code = gap_pos2  */
  sint i,j;
  
  if (struct_penalties2 != NONE) {
    sl=0;
    for (j=0;j<seqlen_array[profile1_nseqs+1];++j) {
      if (gap_penalty_mask2[j] == gap_pos1 && (reset_alignments_new ||
					       reset_alignments_all)) continue;
      if (gap_penalty_mask2[j] == gap_pos2 && (reset_alignments_all)) continue;
      gap_penalty_mask2[sl]=gap_penalty_mask2[j];
      ++sl;
    }
  }
  
  if (struct_penalties2 == SECST) {
    sl=0;
    for (j=0;j<seqlen_array[profile1_nseqs+1];++j) {
      if (sec_struct_mask2[j] == gap_pos1 && (reset_alignments_new ||
					      reset_alignments_all)) continue;
      if (sec_struct_mask2[j] == gap_pos2 && (reset_alignments_all)) continue;
			sec_struct_mask2[sl]=sec_struct_mask2[j];
			++sl;
    }
  }
  
  for(i=profile1_nseqs+1;i<=nseqs;++i) {
    sl=0;
    for(j=1;j<=seqlen_array[i];++j) {
      if(seq_array[i][j] == gap_pos1 && (reset_alignments_new ||
					 reset_alignments_all)) continue;
      if(seq_array[i][j] == gap_pos2 && (reset_alignments_all)) continue;
      ++sl;
      seq_array[i][sl]=seq_array[i][j];
    }
    seqlen_array[i]=sl;
  }
  
  
}



void fix_gaps(void)   /* fix gaps introduced in older alignments (code = gap_pos1) */
{		      				
  sint i,j;
  
  if (struct_penalties1 != NONE) {
    for (j=0;j<seqlen_array[1];++j) {
      if (gap_penalty_mask1[j] == gap_pos1)
	gap_penalty_mask1[j]=gap_pos2;
    }
  }
  
  if (struct_penalties1 == SECST) {
    for (j=0;j<seqlen_array[1];++j) {
      if (sec_struct_mask1[j] == gap_pos1)
	sec_struct_mask1[j]=gap_pos2;
    }
  }
  
  for(i=1;i<=nseqs;++i) {
    for(j=1;j<=seqlen_array[i];++j) {
      if(seq_array[i][j] == gap_pos1)
	seq_array[i][j]=gap_pos2;
    }
  }
}

static sint find_match(char *probe, char *list[], sint n)
{
  sint i,j,len;
  sint count,match=0;
  
  len = (sint)strlen(probe);
  for (i=0;i<len;i++) {
    count = 0;
    for (j=0;j<n;j++) {
      if (probe[i] == list[j][i]) {
	match = j;
	count++;
      }
    }
    if (count == 0) return((sint)-1);
    if (count == 1) return(match);
  }
  return((sint)-1);
}

static void create_parameter_output(void)
{
  char parname[FILENAMELEN+1], temp[FILENAMELEN+1];
  char path[FILENAMELEN+1];
  FILE *parout;
  
  get_path(seqname,path);
  strcpy(parname,path);
  strcat(parname,"par");
  
  if(usemenu) {
    fprintf(stdout,"\nEnter a name for the parameter output file [%s]: ",
	    parname);
    fgets(temp, FILENAMELEN, stdin);
//    gets(temp);
    if(*temp != EOS)
      strcpy(parname,temp);
  }

/* create a file with execute permissions first */
  remove(parname);
  /*
    fd = creat(parname, 0777);
    close(fd);
  */
  
  if((parout = open_explicit_file(parname))==NULL) return;
  
  fprintf(parout,"clustalw \\\n");
  if (!empty && profile1_empty) fprintf(parout,"-infile=%s \\\n",seqname);
  if (!profile1_empty) fprintf(parout,"-profile1=%s\\\n",profile1_name);
  if (!profile2_empty) fprintf(parout,"-profile2=%s \\\n",profile2_name);
  if (dnaflag == TRUE) 
    fprintf(parout,"-type=dna \\\n");
  else
    fprintf(parout,"-type=protein \\\n");
  
  if (quick_pairalign) {
    fprintf(parout,"-quicktree \\\n");
    fprintf(parout,"-ktuple=%d \\\n",(pint)ktup);
    fprintf(parout,"-window=%d \\\n",(pint)window);
    fprintf(parout,"-pairgap=%d \\\n",(pint)wind_gap);
    fprintf(parout,"-topdiags=%d \\\n",(pint)signif);    
    if (percent) fprintf(parout,"-score=percent \\\n");      
    else
      fprintf(parout,"-score=absolute \\\n");      
  }
  else {
    if (!dnaflag) {
      fprintf(parout,"-pwmatrix=%s \\\n",pw_mtrxname);
      fprintf(parout,"-pwgapopen=%.2f \\\n",prot_pw_go_penalty);
      fprintf(parout,"-pwgapext=%.2f \\\n",prot_pw_ge_penalty);
    }
    else {
      fprintf(parout,"-pwgapopen=%.2f \\\n",pw_go_penalty);
      fprintf(parout,"-pwgapext=%.2f \\\n",pw_ge_penalty);
    }
  }
  
  if (!dnaflag) {
    fprintf(parout,"-matrix=%s \\\n",mtrxname);
    fprintf(parout,"-gapopen=%.2f \\\n",prot_gap_open);
    fprintf(parout,"-gapext=%.2f \\\n",prot_gap_extend);
  }
  else {
    fprintf(parout,"-gapopen=%.2f \\\n",dna_gap_open);
    fprintf(parout,"-gapext=%.2f \\\n",dna_gap_extend);
  }
  
  fprintf(parout,"-maxdiv=%d \\\n",(pint)divergence_cutoff);
  if (!use_endgaps) fprintf(parout,"-endgaps \\\n");    
  
  if (!dnaflag) {
    if (neg_matrix) fprintf(parout,"-negative \\\n");   
    if (no_pref_penalties) fprintf(parout,"-nopgap \\\n");     
    if (no_hyd_penalties) fprintf(parout,"-nohgap \\\n");     
    if (no_var_penalties) fprintf(parout,"-novgap \\\n");     
    fprintf(parout,"-hgapresidues=%s \\\n",hyd_residues);
    fprintf(parout,"-gapdist=%d \\\n",(pint)gap_dist);     
  }
  else {
    fprintf(parout,"-transweight=%.2f \\\n",transition_weight);
  }
  
  if (output_gcg) fprintf(parout,"-output=gcg \\\n");
  else if (output_gde) fprintf(parout,"-output=gde \\\n");
  else if (output_nbrf) fprintf(parout,"-output=pir \\\n");
  else if (output_phylip) fprintf(parout,"-output=phylip \\\n");
  else if (output_nexus) fprintf(parout,"-output=nexus \\\n");
  if (outfile_name[0]!=EOS) fprintf(parout,"-outfile=%s \\\n",outfile_name);
  if (output_order==ALIGNED) fprintf(parout,"-outorder=aligned \\\n");  
  else                      fprintf(parout,"-outorder=input \\\n");  
  if (output_gde)
  {
    if (lowercase) 
    {
       fprintf(parout,"-case=lower \\\n");
    }
    else           
    {
       fprintf(parout,"-case=upper \\\n");
    }
  }
  
  
  fprintf(parout,"-interactive\n");
  
  /*
    if (kimura) fprintf(parout,"-kimura \\\n");     
    if (tossgaps) fprintf(parout,"-tossgaps \\\n");   
    fprintf(parout,"-seed=%d \\\n",(pint)boot_ran_seed);
    fprintf(parout,"-bootstrap=%d \\\n",(pint)boot_ntrials);
  */
  fclose(parout);
}


#define isgap(val1) ( (val1 < 0) || (val1 > max_aa) )
#define isend(val1) ((val1 == -3)||(val1 == 253) )

void calc_percidentity(FILE *pfile)
{
  double **pmat;
/*  char residue; */
  
  float ident;
  int nmatch;
  
  sint val1, val2;
  
  sint i,j,k, length_longest;
  sint length_shortest;
  
  int rs=0, rl=0;
  /* findout sequence length, longest and shortest ; */
  length_longest=0;
  length_shortest=0;

  for (i=1;i<=nseqs;i++) {
    /*printf("\n %d :  %d ",i,seqlen_array[i]);*/
    if (length_longest < seqlen_array[i]){
      length_longest = seqlen_array[i];
      rs = i;
    }
    if (length_shortest > seqlen_array[i]) {
      length_shortest = seqlen_array[i];
      rl = i;
    }
  }
  /*
  printf("\n shortest length  %s %d ",names[rs], length_shortest);
  printf("\n longest est length  %s %d",names[rl], length_longest);
  */  

  pmat = (double **)ckalloc((nseqs+1) * sizeof(double *));
  for (i=0;i<=nseqs;i++)
    pmat[i] = (double *)ckalloc((nseqs+1) * sizeof(double));
  for (i = 0; i <= nseqs; i++)
    for (j = 0; j <= nseqs; j++)
      pmat[i][j] = 0.0;

  nmatch = 0;

  for (i=1; i <= nseqs; i++) {
    /*printf("\n %5d:  comparing %s with  ",i,names[i]); */
    for (j=i; j<=nseqs ;  j++) {
      printf("\n           %s ",names[j]);
      ident = 0;
      nmatch = 0;
      for(k=1;  k<=length_longest; k++) {
	val1 = seq_array[i][k];
	val2 = seq_array[j][k];
	if ( isend(val1) || isend(val2)) break;  /* end of sequence ????? */
	if ( isgap(val1)  || isgap(val2) ) continue; /* residue = '-'; */
	if (val1 == val2) {
	  ident++ ;
	  nmatch++;
	  /*	residue = amino_acid_codes[val1]; 
	printf("%c:",residue);
	residue = amino_acid_codes[val2]; 
	printf("%c  ",residue);*/
	}
	else {
	  nmatch++ ;
	    }
      }
      ident = ident/nmatch * 100.0 ;
      pmat[i][j] = ident;
      pmat[j][i]= ident;
      /*      printf("  %d x %d .... match %d %d \n",i,j,ident,pmat[i][j]);  */
    }

  }
  /*  printf("\n nmatch = %d\n ", nmatch);*/
  fprintf(pfile,"#\n#\n#  Percent Identity  Matrix - created by Clustal%s \n#\n#\n",revision_level);
  for(i=1;i<=nseqs;i++) {
    fprintf(pfile,"\n %5d: %-*s",i,max_names,names[i]);
    for(j=1;j<=nseqs;j++) {
      fprintf(pfile,"%8.0f",pmat[i][j]);
    }
  }
  fprintf(pfile,"\n");

  for (i=0;i<nseqs;i++) 
    pmat[i]=ckfree((void *)pmat[i]);
  pmat=ckfree((void *)pmat);

}
