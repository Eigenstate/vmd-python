/* Phyle of filogenetic tree calculating functions for CLUSTAL W */
/* DES was here  FEB. 1994 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "clustalw.h"
#include "dayhoff.h"    /* set correction for amino acid distances >= 75% */


/*
 *   Prototypes
 */
Boolean transition(sint base1, sint base2);
void tree_gap_delete(void);
void distance_matrix_output(FILE *ofile);
void nj_tree(char **tree_description, FILE *tree);
void compare_tree(char **tree1, char **tree2, sint *hits, sint n);
void print_phylip_tree(char **tree_description, FILE *tree, sint bootstrap);
void print_nexus_tree(char **tree_description, FILE *tree, sint bootstrap);
sint two_way_split(char **tree_description, FILE *tree, sint start_row, sint flag, sint bootstrap);
sint two_way_split_nexus(char **tree_description, FILE *tree, sint start_row, sint flag, sint bootstrap);
void print_tree(char **tree_description, FILE *tree, sint *totals);
static Boolean is_ambiguity(char c);
static void overspill_message(sint overspill,sint total_dists);


/*
 *   Global variables
 */

extern sint max_names;

extern double **tmat;     /* general nxn array of reals; allocated from main */
                          /* this is used as a distance matrix */
extern Boolean dnaflag;   /* TRUE for DNA seqs; FALSE for proteins */
extern Boolean tossgaps;  /* Ignore places in align. where ANY seq. has a gap*/
extern Boolean kimura;    /* Use correction for multiple substitutions */
extern Boolean output_tree_clustal;   /* clustal text output for trees */
extern Boolean output_tree_phylip;    /* phylip nested parentheses format */
extern Boolean output_tree_distances; /* phylip distance matrix */
extern Boolean output_tree_nexus;     /* nexus format tree */
extern Boolean output_pim;     /* perc identity matrix output Ramu */

extern sint    bootstrap_format;      /* bootstrap file format */
extern Boolean empty;                 /* any sequences in memory? */
extern Boolean usemenu;   /* interactive (TRUE) or command line (FALSE) */
extern sint nseqs;
extern sint max_aln_length;
extern sint *seqlen_array; /* the lengths of the sequences */
extern char **seq_array;   /* the sequences */
extern char **names;       /* the seq. names */
extern char seqname[];		/* name of input file */
extern sint gap_pos1,gap_pos2;
extern Boolean use_ambiguities;
extern char *amino_acid_codes;

static double 	*av;
static double 	*left_branch, *right_branch;
static double 	*save_left_branch, *save_right_branch;
static sint	*boot_totals;
static sint 	*tkill;
/*  
  The next line is a fossil from the days of using the cc ran()
static int 	ran_factor;
*/
static sint 	*boot_positions;
static FILE 	*phylip_phy_tree_file;
static FILE 	*clustal_phy_tree_file;
static FILE 	*distances_phy_tree_file;
static FILE 	*nexus_phy_tree_file;
static FILE     *pim_file; /* Ramu */
static Boolean 	verbose;
static char 	*tree_gaps;
static sint first_seq, last_seq;
                     /* array of weights; 1 for use this posn.; 0 don't */

extern sint boot_ntrials;		/* number of bootstrap trials */
extern unsigned sint boot_ran_seed;	/* random number generator seed */

void phylogenetic_tree(char *phylip_name,char *clustal_name,char *dist_name, char *nexus_name, char *pim_name)
/* 
   Calculate a tree using the distances in the nseqs*nseqs array tmat.
   This is the routine for getting the REAL trees after alignment.
*/
{	char path[FILENAMELEN+1];
	sint i, j;
	sint overspill = 0;
	sint total_dists;
	static char **standard_tree;
	static char **save_tree;
/*	char lin2[10]; */

	if(empty) {
		error("You must load an alignment first");
		return;
	}

	if(nseqs<2) {
		error("Alignment has only %d sequences",nseqs);
		return;
	}
	first_seq=1;
	last_seq=nseqs;

	get_path(seqname,path);
	
if(output_tree_clustal) {
        if (clustal_name[0]!=EOS) {
                if((clustal_phy_tree_file = open_explicit_file(
                clustal_name))==NULL) return;
        }
        else {
		if((clustal_phy_tree_file = open_output_file(
		"\nEnter name for CLUSTAL    tree output file  ",path,
		clustal_name,"nj")) == NULL) return;
        }
}

if(output_tree_phylip) {
        if (phylip_name[0]!=EOS) {
                if((phylip_phy_tree_file = open_explicit_file(
                phylip_name))==NULL) return;
        }
        else {
                 if((phylip_phy_tree_file = open_output_file(
		"\nEnter name for PHYLIP     tree output file  ",path,
                phylip_name,"ph")) == NULL) return;
        }
}

if(output_tree_distances)
{
        if (dist_name[0]!=EOS) {
                if((distances_phy_tree_file = open_explicit_file(
                dist_name))==NULL) return;
        }
        else {
		if((distances_phy_tree_file = open_output_file(
		"\nEnter name for distance matrix output file  ",path,
		dist_name,"dst")) == NULL) return;
        }
}

if(output_tree_nexus)
{
        if (nexus_name[0]!=EOS) {
                if((nexus_phy_tree_file = open_explicit_file(
                nexus_name))==NULL) return;
        }
        else {
		if((nexus_phy_tree_file = open_output_file(
		"\nEnter name for NEXUS tree output file  ",path,
		nexus_name,"tre")) == NULL) return;
        }
}

if(output_pim)
{
        if (pim_name[0]!=EOS) {
        	if((pim_file = open_explicit_file(
		pim_name))==NULL) return;
      }
      else {
        	if((pim_file = open_output_file(
		"\nEnter name for % Identity matrix output file  ",path,
                pim_name,"pim")) == NULL) return;
      }
}

	boot_positions = (sint *)ckalloc( (seqlen_array[first_seq]+2) * sizeof (sint) );

	for(j=1; j<=seqlen_array[first_seq]; ++j) 
		boot_positions[j] = j;		

	if(output_tree_clustal) {
		verbose = TRUE;     /* Turn on file output */
		if(dnaflag)
			overspill = dna_distance_matrix(clustal_phy_tree_file);
		else 
			overspill = prot_distance_matrix(clustal_phy_tree_file);
	}

	if(output_tree_phylip) {
		verbose = FALSE;     /* Turn off file output */
		if(dnaflag)
			overspill = dna_distance_matrix(phylip_phy_tree_file);
		else 
			overspill = prot_distance_matrix(phylip_phy_tree_file);
	}

	if(output_tree_nexus) {
		verbose = FALSE;     /* Turn off file output */
		if(dnaflag)
			overspill = dna_distance_matrix(nexus_phy_tree_file);
		else 
			overspill = prot_distance_matrix(nexus_phy_tree_file);
	}

        if(output_pim) { /* Ramu  */
          	verbose = FALSE;     /* Turn off file output */
          	if(dnaflag)
           		calc_percidentity(pim_file);
          	else
            		calc_percidentity(pim_file);
        }


	if(output_tree_distances) {
		verbose = FALSE;     /* Turn off file output */
		if(dnaflag)
			overspill = dna_distance_matrix(distances_phy_tree_file);
		else 
			overspill = prot_distance_matrix(distances_phy_tree_file);
      		distance_matrix_output(distances_phy_tree_file);
	}

/* check if any distances overflowed the distance corrections */
	if ( overspill > 0 ) {
		total_dists = (nseqs*(nseqs-1))/2;
		overspill_message(overspill,total_dists);
	}

	if(output_tree_clustal) verbose = TRUE;     /* Turn on file output */

	standard_tree   = (char **) ckalloc( (nseqs+1) * sizeof (char *) );
	for(i=0; i<nseqs+1; i++) 
		standard_tree[i]  = (char *) ckalloc( (nseqs+1) * sizeof(char) );
	save_tree   = (char **) ckalloc( (nseqs+1) * sizeof (char *) );
	for(i=0; i<nseqs+1; i++) 
		save_tree[i]  = (char *) ckalloc( (nseqs+1) * sizeof(char) );

	if(output_tree_clustal || output_tree_phylip || output_tree_nexus) 
		nj_tree(standard_tree,clustal_phy_tree_file);

	for(i=1; i<nseqs+1; i++) 
		for(j=1; j<nseqs+1; j++) 
			save_tree[i][j]  = standard_tree[i][j];

	if(output_tree_phylip) 
		print_phylip_tree(standard_tree,phylip_phy_tree_file,0);

	for(i=1; i<nseqs+1; i++) 
		for(j=1; j<nseqs+1; j++) 
			standard_tree[i][j]  = save_tree[i][j];

	if(output_tree_nexus) 
		print_nexus_tree(standard_tree,nexus_phy_tree_file,0);

/*
	print_tree(standard_tree,phy_tree_file);
*/
	tree_gaps=ckfree((void *)tree_gaps);
	boot_positions=ckfree((void *)boot_positions);
	if (left_branch != NULL) left_branch=ckfree((void *)left_branch);
	if (right_branch != NULL) right_branch=ckfree((void *)right_branch);
	if (tkill != NULL) tkill=ckfree((void *)tkill);
	if (av != NULL) av=ckfree((void *)av);
	for (i=0;i<nseqs+1;i++)
		standard_tree[i]=ckfree((void *)standard_tree[i]);
	standard_tree=ckfree((void *)standard_tree);

	for (i=0;i<nseqs+1;i++)
		save_tree[i]=ckfree((void *)save_tree[i]);
	save_tree=ckfree((void *)save_tree);

if(output_tree_clustal) {
	fclose(clustal_phy_tree_file);	
	info("Phylogenetic tree file created:   [%s]",clustal_name);
}

if(output_tree_phylip) {
	fclose(phylip_phy_tree_file);	
	info("Phylogenetic tree file created:   [%s]",phylip_name);
}

if(output_tree_distances) {
	fclose(distances_phy_tree_file);	
	info("Distance matrix  file  created:   [%s]",dist_name);
}

if(output_tree_nexus) {
	fclose(nexus_phy_tree_file);	
	info("Nexus tree file  created:   [%s]",nexus_name);
}

if(output_pim) {
	fclose(pim_file);
	info(" perc identity matrix file  created:   [%s]",pim_name);
}

}

static void overspill_message(sint overspill,sint total_dists)
{
	char err_mess[1024]="";

	sprintf(err_mess,"%d of the distances out of a total of %d",
	(pint)overspill,(pint)total_dists);
	strcat(err_mess,"\n were out of range for the distance correction.");
	strcat(err_mess,"\n");
	strcat(err_mess,"\n SUGGESTIONS: 1) remove the most distant sequences");
	strcat(err_mess,"\n           or 2) use the PHYLIP package");
	strcat(err_mess,"\n           or 3) turn off the correction.");
	strcat(err_mess,"\n Note: Use option 3 with caution! With this degree");
	strcat(err_mess,"\n of divergence you will have great difficulty");
	strcat(err_mess,"\n getting robust and reliable trees.");
	strcat(err_mess,"\n\n");
	warning(err_mess);
}



Boolean transition(sint base1, sint base2) /* TRUE if transition; else FALSE */
/* 

   assumes that the bases of DNA sequences have been translated as
   a,A = 0;   c,C = 1;   g,G = 2;   t,T,u,U = 3;  N = 4;  
   a,A = 0;   c,C = 2;   g,G = 6;   t,T,u,U =17;  

   A <--> G  and  T <--> C  are transitions;  all others are transversions.

*/
{
	if( ((base1 == 0) && (base2 == 6)) || ((base1 == 6) && (base2 == 0)) )
		return TRUE;                                     /* A <--> G */
	if( ((base1 ==17) && (base2 == 2)) || ((base1 == 2) && (base2 ==17)) )
		return TRUE;                                     /* T <--> C */
    return FALSE;
}


void tree_gap_delete(void)   /* flag all positions in alignment that have a gap */
{			  /* in ANY sequence */
	sint seqn;
	sint posn;

	tree_gaps = (char *)ckalloc( (max_aln_length+1) * sizeof (char) );
        
	for(posn=1; posn<=seqlen_array[first_seq]; ++posn) {
		tree_gaps[posn] = 0;
     	for(seqn=1; seqn<=last_seq-first_seq+1; ++seqn)  {
			if((seq_array[seqn+first_seq-1][posn] == gap_pos1) ||
			   (seq_array[seqn+first_seq-1][posn] == gap_pos2)) {
			   tree_gaps[posn] = 1;
				break;
			}
		}
	}

}

void distance_matrix_output(FILE *ofile)
{
	sint i,j;
	
	fprintf(ofile,"%6d",(pint)last_seq-first_seq+1);
	for(i=1;i<=last_seq-first_seq+1;i++) {
		fprintf(ofile,"\n%-*s ",max_names,names[i]);
		for(j=1;j<=last_seq-first_seq+1;j++) {
			fprintf(ofile,"%6.3f ",tmat[i][j]);
			if(j % 8 == 0) {
				if(j!=last_seq-first_seq+1) fprintf(ofile,"\n"); 
				if(j != last_seq-first_seq+1 ) fprintf(ofile,"          ");
			}
		}
	}
}



#ifdef ORIGINAL_NJ_TREE
void nj_tree(char **tree_description, FILE *tree)
{
	register int i;
	sint l[4],nude,k;
	sint nc,mini,minj,j,ii,jj;
	double fnseqs,fnseqs2=0,sumd;
	double diq,djq,dij,d2r,dr,dio,djo,da;
	double tmin,total,dmin;
	double bi,bj,b1,b2,b3,branch[4];
	sint typei,typej;             /* 0 = node; 1 = OTU */
	
	fnseqs = (double)last_seq-first_seq+1;

/*********************** First initialisation ***************************/
	
	if(verbose)  {
		fprintf(tree,"\n\n\t\t\tNeighbor-joining Method\n");
		fprintf(tree,"\n Saitou, N. and Nei, M. (1987)");
		fprintf(tree," The Neighbor-joining Method:");
		fprintf(tree,"\n A New Method for Reconstructing Phylogenetic Trees.");
		fprintf(tree,"\n Mol. Biol. Evol., 4(4), 406-425\n");
		fprintf(tree,"\n\n This is an UNROOTED tree\n");
		fprintf(tree,"\n Numbers in parentheses are branch lengths\n\n");
	}	

	if (fnseqs == 2) {
		if (verbose) fprintf(tree,"Cycle   1     =  SEQ:   1 (%9.5f) joins  SEQ:   2 (%9.5f)",tmat[first_seq][first_seq+1],tmat[first_seq][first_seq+1]);
		return;
	}

	mini = minj = 0;

	left_branch 	= (double *) ckalloc( (nseqs+2) * sizeof (double)   );
	right_branch    = (double *) ckalloc( (nseqs+2) * sizeof (double)   );
	tkill 		= (sint *) ckalloc( (nseqs+1) * sizeof (sint) );
	av   		= (double *) ckalloc( (nseqs+1) * sizeof (double)   );

	for(i=1;i<=last_seq-first_seq+1;++i) 
		{
		tmat[i][i] = av[i] = 0.0;
		tkill[i] = 0;
		}

/*********************** Enter The Main Cycle ***************************/

 /*	for(nc=1; nc<=(last_seq-first_seq+1-3); ++nc) {  */            	/**start main cycle**/
	for(nc=1; nc<=(last_seq-first_seq+1-3); ++nc) {
		sumd = 0.0;
		for(j=2; j<=last_seq-first_seq+1; ++j)
			for(i=1; i<j; ++i) {
				tmat[j][i] = tmat[i][j];
				sumd = sumd + tmat[i][j];
			}

		tmin = 99999.0;

/*.................compute SMATij values and find the smallest one ........*/

		for(jj=2; jj<=last_seq-first_seq+1; ++jj) 
			if(tkill[jj] != 1) 
				for(ii=1; ii<jj; ++ii)
					if(tkill[ii] != 1) {
						diq = djq = 0.0;

						for(i=1; i<=last_seq-first_seq+1; ++i) {
							diq = diq + tmat[i][ii];
							djq = djq + tmat[i][jj];
						}

						dij = tmat[ii][jj];
						d2r = diq + djq - (2.0*dij);
						dr  = sumd - dij -d2r;
						fnseqs2 = fnseqs - 2.0;
					        total= d2r+ fnseqs2*dij +dr*2.0;
						total= total / (2.0*fnseqs2);

						if(total < tmin) {
							tmin = total;
							mini = ii;
							minj = jj;
						}
					}
		

/*.................compute branch lengths and print the results ........*/


		dio = djo = 0.0;
		for(i=1; i<=last_seq-first_seq+1; ++i) {
			dio = dio + tmat[i][mini];
			djo = djo + tmat[i][minj];
		}

		dmin = tmat[mini][minj];
		dio = (dio - dmin) / fnseqs2;
		djo = (djo - dmin) / fnseqs2;
		bi = (dmin + dio - djo) * 0.5;
		bj = dmin - bi;
		bi = bi - av[mini];
		bj = bj - av[minj];

		if( av[mini] > 0.0 )
			typei = 0;
		else
			typei = 1;
		if( av[minj] > 0.0 )
			typej = 0;
		else
			typej = 1;

		if(verbose) 
	 	    fprintf(tree,"\n Cycle%4d     = ",(pint)nc);

/* 
   set negative branch lengths to zero.  Also set any tiny positive
   branch lengths to zero.
*/		if( fabs(bi) < 0.0001) bi = 0.0;
		if( fabs(bj) < 0.0001) bj = 0.0;

	    	if(verbose) {
		    if(typei == 0) 
			fprintf(tree,"Node:%4d (%9.5f) joins ",(pint)mini,bi);
		    else 
			fprintf(tree," SEQ:%4d (%9.5f) joins ",(pint)mini,bi);

		    if(typej == 0) 
			fprintf(tree,"Node:%4d (%9.5f)",(pint)minj,bj);
		    else 
			fprintf(tree," SEQ:%4d (%9.5f)",(pint)minj,bj);

		    fprintf(tree,"\n");
	    	}	


	    	left_branch[nc] = bi;
	    	right_branch[nc] = bj;

		for(i=1; i<=last_seq-first_seq+1; i++)
			tree_description[nc][i] = 0;

	     	if(typei == 0) { 
			for(i=nc-1; i>=1; i--)
				if(tree_description[i][mini] == 1) {
					for(j=1; j<=last_seq-first_seq+1; j++)  
					     if(tree_description[i][j] == 1)
						    tree_description[nc][j] = 1;
					break;
				}
		}
		else
			tree_description[nc][mini] = 1;

		if(typej == 0) {
			for(i=nc-1; i>=1; i--) 
				if(tree_description[i][minj] == 1) {
					for(j=1; j<=last_seq-first_seq+1; j++)  
					     if(tree_description[i][j] == 1)
						    tree_description[nc][j] = 1;
					break;
				}
		}
		else
			tree_description[nc][minj] = 1;
			

/* 
   Here is where the -0.00005 branch lengths come from for 3 or more
   identical seqs.
*/
/*		if(dmin <= 0.0) dmin = 0.0001; */
                if(dmin <= 0.0) dmin = 0.000001;
		av[mini] = dmin * 0.5;

/*........................Re-initialisation................................*/

		fnseqs = fnseqs - 1.0;
		tkill[minj] = 1;

		for(j=1; j<=last_seq-first_seq+1; ++j) 
			if( tkill[j] != 1 ) {
				da = ( tmat[mini][j] + tmat[minj][j] ) * 0.5;
				if( (mini - j) < 0 ) 
					tmat[mini][j] = da;
				if( (mini - j) > 0)
					tmat[j][mini] = da;
			}

		for(j=1; j<=last_seq-first_seq+1; ++j)
			tmat[minj][j] = tmat[j][minj] = 0.0;


/****/	}						/**end main cycle**/

/******************************Last Cycle (3 Seqs. left)********************/

	nude = 1;

	for(i=1; i<=last_seq-first_seq+1; ++i)
		if( tkill[i] != 1 ) {
			l[nude] = i;
			nude = nude + 1;
		}

	b1 = (tmat[l[1]][l[2]] + tmat[l[1]][l[3]] - tmat[l[2]][l[3]]) * 0.5;
	b2 =  tmat[l[1]][l[2]] - b1;
	b3 =  tmat[l[1]][l[3]] - b1;
 
	branch[1] = b1 - av[l[1]];
	branch[2] = b2 - av[l[2]];
	branch[3] = b3 - av[l[3]];

/* Reset tiny negative and positive branch lengths to zero */
	if( fabs(branch[1]) < 0.0001) branch[1] = 0.0;
	if( fabs(branch[2]) < 0.0001) branch[2] = 0.0;
	if( fabs(branch[3]) < 0.0001) branch[3] = 0.0;

	left_branch[last_seq-first_seq+1-2] = branch[1];
	left_branch[last_seq-first_seq+1-1] = branch[2];
	left_branch[last_seq-first_seq+1]   = branch[3];

	for(i=1; i<=last_seq-first_seq+1; i++)
		tree_description[last_seq-first_seq+1-2][i] = 0;

	if(verbose)
		fprintf(tree,"\n Cycle%4d (Last cycle, trichotomy):\n",(pint)nc);

	for(i=1; i<=3; ++i) {
	   if( av[l[i]] > 0.0) {
	      	if(verbose)
	      	    fprintf(tree,"\n\t\t Node:%4d (%9.5f) ",(pint)l[i],branch[i]);
		for(k=last_seq-first_seq+1-3; k>=1; k--)
			if(tree_description[k][l[i]] == 1) {
				for(j=1; j<=last_seq-first_seq+1; j++)
				 	if(tree_description[k][j] == 1)
					    tree_description[last_seq-first_seq+1-2][j] = i;
				break;
			}
	   }
	   else  {
	      	if(verbose)
	   	    fprintf(tree,"\n\t\t  SEQ:%4d (%9.5f) ",(pint)l[i],branch[i]);
		tree_description[last_seq-first_seq+1-2][l[i]] = i;
	   }
	   if(i < 3) {
	      	if(verbose)
	            fprintf(tree,"joins");
	   }
	}

	if(verbose)
		fprintf(tree,"\n");

}

#else /* ORIGINAL_NJ_TREE */

void nj_tree(char **tree_description, FILE *tree) {
	void fast_nj_tree();
 
	/*fprintf(stderr, "****** call fast_nj_tree() !!!! ******\n");*/
	fast_nj_tree(tree_description, tree);
}


/****************************************************************************
 * [ Improvement ideas in fast_nj_tree() ] by DDBJ & FUJITSU Limited.
 *						written by Tadashi Koike
 *						(takoike@genes.nig.ac.jp)
 *******************
 * <IMPROVEMENT 1> : Store the value of sum of the score to temporary array,
 *                   and use again and again.
 *
 *	In the main cycle, these are calculated again and again :
 *	    diq = sum of tmat[n][ii]   (n:1 to last_seq-first_seq+1),
 *	    djq = sum of tmat[n][jj]   (n:1 to last_seq-first_seq+1),
 *	    dio = sum of tmat[n][mini] (n:1 to last_seq-first_seq+1),
 *	    djq = sum of tmat[n][minj] (n:1 to last_seq-first_seq+1)
 *		// 'last_seq' and 'first_seq' are both constant values //
 *	and the result of above calculations is always same until 
 *	a best pair of neighbour nodes is joined.
 *
 *	So, we change the logic to calculate the sum[i] (=sum of tmat[n][i]
 *	(n:1 to last_seq-first_seq+1)) and store it to array, before
 *	beginning to find a best pair of neighbour nodes, and after that 
 *	we use them again and again.
 *
 *	    tmat[i][j]
 *	              1   2   3   4   5
 *	            +---+---+---+---+---+
 *	          1 |   |   |   |   |   |
 *	            +---+---+---+---+---+
 *	          2 |   |   |   |   |   |  1) calculate sum of tmat[n][i]
 *	            +---+---+---+---+---+        (n: 1 to last_seq-first_seq+1)
 *	          3 |   |   |   |   |   |  2) store that sum value to sum[i]
 *	            +---+---+---+---+---+
 *	          4 |   |   |   |   |   |  3) use sum[i] during finding a best
 *	            +---+---+---+---+---+     pair of neibour nodes.
 *	          5 |   |   |   |   |   |
 *	            +---+---+---+---+---+
 *	              |   |   |   |   |
 *	              V   V   V   V   V  Calculate sum , and store it to sum[i]
 *	            +---+---+---+---+---+
 *	     sum[i] |   |   |   |   |   |
 *	            +---+---+---+---+---+
 *
 *	At this time, we thought that we use upper triangle of the matrix
 *	because tmat[i][j] is equal to tmat[j][i] and tmat[i][i] is equal 
 *	to zero. Therefore, we prepared sum_rows[i] and sum_cols[i] instead 
 *	of sum[i] for storing the sum value.
 *
 *	    tmat[i][j]
 *	              1   2   3   4   5     sum_cols[i]
 *	            +---+---+---+---+---+     +---+
 *	          1     | # | # | # | # | --> |   | ... sum of tmat[1][2..5]
 *	            + - +---+---+---+---+     +---+
 *	          2         | # | # | # | --> |   | ... sum of tmat[2][3..5]
 *	            + - + - +---+---+---+     +---+
 *	          3             | # | # | --> |   | ... sum of tmat[3][4..5]
 *	            + - + - + - +---+---+     +---+
 *	          4                 | # | --> |   | ... sum of tmat[4][5]
 *	            + - + - + - + - +---+     +---+
 *	          5                     | --> |   | ... zero
 *	            + - + - + - + - + - +     +---+
 *	              |   |   |   |   |
 *	              V   V   V   V   V  Calculate sum , sotre to sum[i]
 *	            +---+---+---+---+---+
 *	sum_rows[i] |   |   |   |   |   |
 *	            +---+---+---+---+---+
 *	              |   |   |   |   |
 *	              |   |   |   |   +----- sum of tmat[1..4][5]
 *	              |   |   |   +--------- sum of tmat[1..3][4]
 *	              |   |   +------------- sum of tmat[1..2][3]
 *	              |   +----------------- sum of tmat[1][2]
 *	              +--------------------- zero
 *
 *	And we use (sum_rows[i] + sum_cols[i]) instead of sum[i].
 *
 *******************
 * <IMPROVEMENT 2> : We manage valid nodes with chain list, instead of
 *                   tkill[i] flag array.
 *
 *	In original logic, invalid(killed?) nodes after nodes-joining
 *	are managed with tkill[i] flag array (set to 1 when killed).
 *	By this method, it is conspicuous to try next node but skip it
 *	at the latter of finding a best pair of neighbor nodes.
 *
 *	So, we thought that we managed valid nodes by using a chain list 
 *	as below:
 *
 *	1) declare the list structure.
 *		struct {
 *		    sint n;		// entry number of node.
 *		    void *prev;		// pointer to previous entry.
 *		    void *next;		// pointer to next entry.
 *		}
 *	2) construct a valid node list.
 *
 *       +-----+    +-----+    +-----+    +-----+        +-----+
 * NULL<-|prev |<---|prev |<---|prev |<---|prev |<- - - -|prev |
 *       |  0  |    |  1  |    |  2  |    |  3  |        |  n  |
 *       | next|--->| next|--->| next|--->| next|- - - ->| next|->NULL
 *       +-----+    +-----+    +-----+    +-----+        +-----+
 *
 *	3) when finding a best pair of neighbor nodes, we use
 *	   this chain list as loop counter.
 *
 *	4) If an entry was killed by node-joining, this chain list is
 *	   modified to remove that entry.
 *
 *	   EX) remove the entry No 2.
 *       +-----+    +-----+               +-----+        +-----+
 * NULL<-|prev |<---|prev |<--------------|prev |<- - - -|prev |
 *       |  0  |    |  1  |               |  3  |        |  n  |
 *       | next|--->| next|-------------->| next|- - - ->| next|->NULL
 *       +-----+    +-----+               +-----+        +-----+
 *                             +-----+
 *                       NULL<-|prev |
 *                             |  2  |
 *                             | next|->NULL
 *                             +-----+
 *
 *	By this method, speed is up at the latter of finding a best pair of
 *	neighbor nodes.
 *
 *******************
 * <IMPROVEMENT 3> : Cut the frequency of division.
 *
 * At comparison between 'total' and 'tmin' in the main cycle, total is
 * divided by (2.0*fnseqs2) before comparison.  If N nodes are available, 
 * that division happen (N*(N-1))/2 order.
 *
 * We thought that the comparison relation between tmin and total/(2.0*fnseqs2)
 * is equal to the comparison relation between (tmin*2.0*fnseqs2) and total.
 * Calculation of (tmin*2.0*fnseqs2) is only one time. so we stop dividing
 * a total value and multiply tmin and (tmin*2.0*fnseqs2) instead.
 *
 *******************
 * <IMPROVEMENT 4> : some transformation of the equation (to cut operations).
 *
 * We transform an equation of calculating 'total' in the main cycle.
 *
 */


void fast_nj_tree(char **tree_description, FILE *tree)
{
	register int i;
	sint l[4],nude,k;
	sint nc,mini,minj,j,ii,jj;
	double fnseqs,fnseqs2=0,sumd;
	double diq,djq,dij/*,d2r,dr*/,dio,djo,da;
	double tmin,total,dmin;
	double bi,bj,b1,b2,b3,branch[4];
	sint typei,typej;             /* 0 = node; 1 = OTU */

	/* IMPROVEMENT 1, STEP 0 : declare  variables */
	double *sum_cols, *sum_rows, *join;

	/* IMPROVEMENT 2, STEP 0 : declare  variables */
	sint loop_limit;
	typedef struct _ValidNodeID {
	    sint n;
	    struct _ValidNodeID *prev;
	    struct _ValidNodeID *next;
	} ValidNodeID;
	ValidNodeID *tvalid, *lpi, *lpj, *lpii, *lpjj, *lp_prev, *lp_next;

	/*
	 * correspondence of the loop counter variables.
	 *   i .. lpi->n,	ii .. lpii->n
	 *   j .. lpj->n,	jj .. lpjj->n
	 */

	fnseqs = (double)last_seq-first_seq+1;

/*********************** First initialisation ***************************/
	
	if(verbose)  {
		fprintf(tree,"\n\n\t\t\tNeighbor-joining Method\n");
		fprintf(tree,"\n Saitou, N. and Nei, M. (1987)");
		fprintf(tree," The Neighbor-joining Method:");
		fprintf(tree,"\n A New Method for Reconstructing Phylogenetic Trees.");
		fprintf(tree,"\n Mol. Biol. Evol., 4(4), 406-425\n");
		fprintf(tree,"\n\n This is an UNROOTED tree\n");
		fprintf(tree,"\n Numbers in parentheses are branch lengths\n\n");
	}	

	if (fnseqs == 2) {
		if (verbose) fprintf(tree,"Cycle   1     =  SEQ:   1 (%9.5f) joins  SEQ:   2 (%9.5f)",tmat[first_seq][first_seq+1],tmat[first_seq][first_seq+1]);
		return;
	}

	mini = minj = 0;

	left_branch 	= (double *) ckalloc( (nseqs+2) * sizeof (double)   );
	right_branch    = (double *) ckalloc( (nseqs+2) * sizeof (double)   );
	tkill 		= (sint *) ckalloc( (nseqs+1) * sizeof (sint) );
	av   		= (double *) ckalloc( (nseqs+1) * sizeof (double)   );

	/* IMPROVEMENT 1, STEP 1 : Allocate memory */
	sum_cols	= (double *) ckalloc( (nseqs+1) * sizeof (double)   );
	sum_rows	= (double *) ckalloc( (nseqs+1) * sizeof (double)   );
	join		= (double *) ckalloc( (nseqs+1) * sizeof (double)   );

	/* IMPROVEMENT 2, STEP 1 : Allocate memory */
	tvalid	= (ValidNodeID *) ckalloc( (nseqs+1) * sizeof (ValidNodeID) );
	/* tvalid[0] is special entry in array. it points a header of valid entry list */
	tvalid[0].n = 0;
	tvalid[0].prev = NULL;
	tvalid[0].next = &tvalid[1];

	/* IMPROVEMENT 2, STEP 2 : Construct and initialize the entry chain list */
	for(i=1, loop_limit = last_seq-first_seq+1,
		lpi=&tvalid[1], lp_prev=&tvalid[0], lp_next=&tvalid[2] ;
		i<=loop_limit ;
		++i, ++lpi, ++lp_prev, ++lp_next)
		{
		tmat[i][i] = av[i] = 0.0;
		tkill[i] = 0;
		lpi->n = i;
		lpi->prev = lp_prev;
		lpi->next = lp_next;

		/* IMPROVEMENT 1, STEP 2 : Initialize arrays */
		sum_cols[i] = sum_rows[i] = join[i] = 0.0;
		}
	tvalid[loop_limit].next = NULL;

	/*
	 * IMPROVEMENT 1, STEP 3 : Calculate the sum of score value that 
	 * is sequence[i] to others.
	 */
	sumd = 0.0;
	for (lpj=tvalid[0].next ; lpj!=NULL ; lpj = lpj->next) {
		double tmp_sum = 0.0;
		j = lpj->n;
		/* calculate sum_rows[j] */
		for (lpi=tvalid[0].next ; lpi->n < j ; lpi = lpi->next) {
			i = lpi->n;
			tmp_sum += tmat[i][j];
			/* tmat[j][i] = tmat[i][j]; */
		}
		sum_rows[j] = tmp_sum;

		tmp_sum = 0.0;
		/* Set lpi to that lpi->n is greater than j */
		if ((lpi != NULL) && (lpi->n == j)) {
			lpi = lpi->next;
		}
		/* calculate sum_cols[j] */
		for( ; lpi!=NULL ; lpi = lpi->next) {
			i = lpi->n;
			tmp_sum += tmat[j][i];
			/* tmat[i][j] = tmat[j][i]; */
		}
		sum_cols[j] = tmp_sum;
	}

/*********************** Enter The Main Cycle ***************************/

	for(nc=1, loop_limit = (last_seq-first_seq+1-3); nc<=loop_limit; ++nc) {

		sumd = 0.0;
		/* IMPROVEMENT 1, STEP 4 : use sum value */
		for(lpj=tvalid[0].next ; lpj!=NULL ; lpj = lpj->next) {
			sumd += sum_cols[lpj->n];
		}

		/* IMPROVEMENT 3, STEP 0 : multiply tmin and 2*fnseqs2 */
		fnseqs2 = fnseqs - 2.0;		/* Set fnseqs2 at this point. */
		tmin = 99999.0 * 2.0 * fnseqs2;


/*.................compute SMATij values and find the smallest one ........*/

		mini = minj = 0;

		/* jj must starts at least 2 */
		if ((tvalid[0].next != NULL) && (tvalid[0].next->n == 1)) {
			lpjj = tvalid[0].next->next;
		} else {
			lpjj = tvalid[0].next;
		}

		for( ; lpjj != NULL; lpjj = lpjj->next) {
			jj = lpjj->n;
			for(lpii=tvalid[0].next ; lpii->n < jj ; lpii = lpii->next) {
				ii = lpii->n;
				diq = djq = 0.0;

				/* IMPROVEMENT 1, STEP 4 : use sum value */
				diq = sum_cols[ii] + sum_rows[ii];
				djq = sum_cols[jj] + sum_rows[jj];
				/*
				 * always ii < jj in this point. Use upper
				 * triangle of score matrix.
				 */
				dij = tmat[ii][jj];

				/*
				 * IMPROVEMENT 3, STEP 1 : fnseqs2 is
				 * already calculated.
				 */
				/* fnseqs2 = fnseqs - 2.0 */

				/* IMPROVEMENT 4 : transform the equation */
  /*-------------------------------------------------------------------*
   * OPTIMIZE of expression 'total = d2r + fnseqs2*dij + dr*2.0'       *
   * total = d2r + fnseq2*dij + 2.0*dr                                 *
   *       = d2r + fnseq2*dij + 2(sumd - dij - d2r)                    *
   *       = d2r + fnseq2*dij + 2*sumd - 2*dij - 2*d2r                 *
   *       =       fnseq2*dij + 2*sumd - 2*dij - 2*d2r + d2r           *
   *       = fnseq2*dij + 2*sumd - 2*dij - d2r                         *
   *       = fnseq2*dij + 2*sumd - 2*dij - (diq + djq - 2*dij)         *
   *       = fnseq2*dij + 2*sumd - 2*dij - diq - djq + 2*dij           *
   *       = fnseq2*dij + 2*sumd - 2*dij + 2*dij - diq - djq           *
   *       = fnseq2*dij + 2*sumd  - diq - djq                          *
   *-------------------------------------------------------------------*/
				total = fnseqs2*dij + 2.0*sumd  - diq - djq;

				/* 
				 * IMPROVEMENT 3, STEP 2 : abbrevlate
				 * the division on comparison between 
				 * total and tmin.
				 */
				/* total = total / (2.0*fnseqs2); */

				if(total < tmin) {
					tmin = total;
					mini = ii;
					minj = jj;
				}
			}
		}

		/* MEMO: always ii < jj in avobe loop, so mini < minj */

/*.................compute branch lengths and print the results ........*/


		dio = djo = 0.0;

		/* IMPROVEMENT 1, STEP 4 : use sum value */
		dio = sum_cols[mini] + sum_rows[mini];
		djo = sum_cols[minj] + sum_rows[minj];

		dmin = tmat[mini][minj];
		dio = (dio - dmin) / fnseqs2;
		djo = (djo - dmin) / fnseqs2;
		bi = (dmin + dio - djo) * 0.5;
		bj = dmin - bi;
		bi = bi - av[mini];
		bj = bj - av[minj];

		if( av[mini] > 0.0 )
			typei = 0;
		else
			typei = 1;
		if( av[minj] > 0.0 )
			typej = 0;
		else
			typej = 1;

		if(verbose) 
	 	    fprintf(tree,"\n Cycle%4d     = ",(pint)nc);

/* 
   set negative branch lengths to zero.  Also set any tiny positive
   branch lengths to zero.
*/		if( fabs(bi) < 0.0001) bi = 0.0;
		if( fabs(bj) < 0.0001) bj = 0.0;

	    	if(verbose) {
		    if(typei == 0) 
			fprintf(tree,"Node:%4d (%9.5f) joins ",(pint)mini,bi);
		    else 
			fprintf(tree," SEQ:%4d (%9.5f) joins ",(pint)mini,bi);

		    if(typej == 0) 
			fprintf(tree,"Node:%4d (%9.5f)",(pint)minj,bj);
		    else 
			fprintf(tree," SEQ:%4d (%9.5f)",(pint)minj,bj);

		    fprintf(tree,"\n");
	    	}	


	    	left_branch[nc] = bi;
	    	right_branch[nc] = bj;

		for(i=1; i<=last_seq-first_seq+1; i++)
			tree_description[nc][i] = 0;

	     	if(typei == 0) { 
			for(i=nc-1; i>=1; i--)
				if(tree_description[i][mini] == 1) {
					for(j=1; j<=last_seq-first_seq+1; j++)  
					     if(tree_description[i][j] == 1)
						    tree_description[nc][j] = 1;
					break;
				}
		}
		else
			tree_description[nc][mini] = 1;

		if(typej == 0) {
			for(i=nc-1; i>=1; i--) 
				if(tree_description[i][minj] == 1) {
					for(j=1; j<=last_seq-first_seq+1; j++)  
					     if(tree_description[i][j] == 1)
						    tree_description[nc][j] = 1;
					break;
				}
		}
		else
			tree_description[nc][minj] = 1;
			

/* 
   Here is where the -0.00005 branch lengths come from for 3 or more
   identical seqs.
*/
/*		if(dmin <= 0.0) dmin = 0.0001; */
                if(dmin <= 0.0) dmin = 0.000001;
		av[mini] = dmin * 0.5;

/*........................Re-initialisation................................*/

		fnseqs = fnseqs - 1.0;
		tkill[minj] = 1;

		/* IMPROVEMENT 2, STEP 3 : Remove tvalid[minj] from chain list. */
		/* [ Before ]
		 *  +---------+        +---------+        +---------+       
		 *  |prev     |<-------|prev     |<-------|prev     |<---
		 *  |    n    |        | n(=minj)|        |    n    |
		 *  |     next|------->|     next|------->|     next|----
		 *  +---------+        +---------+        +---------+ 
		 *
		 * [ After ]
		 *  +---------+                           +---------+       
		 *  |prev     |<--------------------------|prev     |<---
		 *  |    n    |                           |    n    |
		 *  |     next|-------------------------->|     next|----
		 *  +---------+                           +---------+ 
		 *                     +---------+
		 *              NULL---|prev     |
		 *                     | n(=minj)|
		 *                     |     next|---NULL
		 *                     +---------+ 
		 */
		(tvalid[minj].prev)->next = tvalid[minj].next;
		if (tvalid[minj].next != NULL) {
			(tvalid[minj].next)->prev = tvalid[minj].prev;
		}
		tvalid[minj].prev = tvalid[minj].next = NULL;

		/* IMPROVEMENT 1, STEP 5 : re-calculate sum values. */
		for(lpj=tvalid[0].next ; lpj != NULL ; lpj = lpj->next) {
			double tmp_di = 0.0;
			double tmp_dj = 0.0;
			j = lpj->n;

			/* 
			 * subtrace a score value related with 'minj' from
			 * sum arrays .
			 */
			if (j < minj) {
				tmp_dj = tmat[j][minj];
				sum_cols[j] -= tmp_dj;
			} else if (j > minj) {
				tmp_dj = tmat[minj][j];
				sum_rows[j] -= tmp_dj;
			} /* nothing to do when j is equal to minj. */
			

			/* 
			 * subtrace a score value related with 'mini' from
			 * sum arrays .
			 */
			if (j < mini) {
				tmp_di = tmat[j][mini];
				sum_cols[j] -= tmp_di;
			} else if (j > mini) {
				tmp_di = tmat[mini][j];
				sum_rows[j] -= tmp_di;
			} /* nothing to do when j is equal to mini. */

			/* 
			 * calculate a score value of the new inner node.
			 * then, store it temporary to join[] array.
			 */
			join[j] = (tmp_dj + tmp_di) * 0.5;
		}

		/* 
		 * 1)
		 * Set the score values (stored in join[]) into the matrix,
		 * row/column position is 'mini'.
		 * 2)
		 * Add a score value of the new inner node to sum arrays.
		 */
		for(lpj=tvalid[0].next ; lpj != NULL; lpj = lpj->next) {
			j = lpj->n;
			if (j < mini) {
				tmat[j][mini] = join[j];
				sum_cols[j] += join[j];
			} else if (j > mini) {
				tmat[mini][j] = join[j];
				sum_rows[j] += join[j];
			} /* nothing to do when j is equal to mini. */
		}

		/* Re-calculate sum_rows[mini],sum_cols[mini]. */
		sum_cols[mini] = sum_rows[mini] = 0.0;

		/* calculate sum_rows[mini] */
		da = 0.0;
		for(lpj=tvalid[0].next ; lpj->n < mini ; lpj = lpj->next) {
                      da += join[lpj->n];
		}
		sum_rows[mini] = da;

		/* skip if 'lpj->n' is equal to 'mini' */
		if ((lpj != NULL) && (lpj->n == mini)) {
			lpj = lpj->next;
		}

		/* calculate sum_cols[mini] */
		da = 0.0;
		for( ; lpj != NULL; lpj = lpj->next) {
                      da += join[lpj->n];
		}
		sum_cols[mini] = da;

		/*
		 * Clean up sum_rows[minj], sum_cols[minj] and score matrix
		 * related with 'minj'.
		 */
		sum_cols[minj] = sum_rows[minj] = 0.0;
		for(j=1; j<=last_seq-first_seq+1; ++j)
			tmat[minj][j] = tmat[j][minj] = join[j] = 0.0;


/****/	}						/**end main cycle**/

/******************************Last Cycle (3 Seqs. left)********************/

	nude = 1;

	for(lpi=tvalid[0].next; lpi != NULL; lpi = lpi->next) {
		l[nude] = lpi->n;
		++nude;
	}

	b1 = (tmat[l[1]][l[2]] + tmat[l[1]][l[3]] - tmat[l[2]][l[3]]) * 0.5;
	b2 =  tmat[l[1]][l[2]] - b1;
	b3 =  tmat[l[1]][l[3]] - b1;
 
	branch[1] = b1 - av[l[1]];
	branch[2] = b2 - av[l[2]];
	branch[3] = b3 - av[l[3]];

/* Reset tiny negative and positive branch lengths to zero */
	if( fabs(branch[1]) < 0.0001) branch[1] = 0.0;
	if( fabs(branch[2]) < 0.0001) branch[2] = 0.0;
	if( fabs(branch[3]) < 0.0001) branch[3] = 0.0;

	left_branch[last_seq-first_seq+1-2] = branch[1];
	left_branch[last_seq-first_seq+1-1] = branch[2];
	left_branch[last_seq-first_seq+1]   = branch[3];

	for(i=1; i<=last_seq-first_seq+1; i++)
		tree_description[last_seq-first_seq+1-2][i] = 0;

	if(verbose)
		fprintf(tree,"\n Cycle%4d (Last cycle, trichotomy):\n",(pint)nc);

	for(i=1; i<=3; ++i) {
	   if( av[l[i]] > 0.0) {
	      	if(verbose)
	      	    fprintf(tree,"\n\t\t Node:%4d (%9.5f) ",(pint)l[i],branch[i]);
		for(k=last_seq-first_seq+1-3; k>=1; k--)
			if(tree_description[k][l[i]] == 1) {
				for(j=1; j<=last_seq-first_seq+1; j++)
				 	if(tree_description[k][j] == 1)
					    tree_description[last_seq-first_seq+1-2][j] = i;
				break;
			}
	   }
	   else  {
	      	if(verbose)
	   	    fprintf(tree,"\n\t\t  SEQ:%4d (%9.5f) ",(pint)l[i],branch[i]);
		tree_description[last_seq-first_seq+1-2][l[i]] = i;
	   }
	   if(i < 3) {
	      	if(verbose)
	            fprintf(tree,"joins");
	   }
	}

	if(verbose)
		fprintf(tree,"\n");

	
	/* IMPROVEMENT 1, STEP 6 : release memory area */
	ckfree(sum_cols);
	ckfree(sum_rows);
	ckfree(join);

	/* IMPROVEMENT 2, STEP 4 : release memory area */
	ckfree(tvalid);

}
#endif /* ORIGINAL_NJ_TREE */



void bootstrap_tree(char *phylip_name,char *clustal_name, char *nexus_name)
{
	sint i,j;
	int ranno;
	char path[MAXLINE+1];
    char dummy[10];
/*	char err_mess[1024]; */
	static char **sample_tree;
	static char **standard_tree;
	static char **save_tree;
	sint total_dists, overspill = 0, total_overspill = 0;
	sint nfails = 0;

	if(empty) {
		error("You must load an alignment first");
		return;
	}

        if(nseqs<4) {
                error("Alignment has only %d sequences",nseqs);
                return;
        }

	if(!output_tree_clustal && !output_tree_phylip && !output_tree_nexus) {
		error("You must select either clustal or phylip or nexus tree output format");
		return;
	}
	get_path(seqname, path);
	
	if (output_tree_clustal) {
        if (clustal_name[0]!=EOS) {
                if((clustal_phy_tree_file = open_explicit_file(
                clustal_name))==NULL) return;
        }
        else {
		if((clustal_phy_tree_file = open_output_file(
		"\nEnter name for bootstrap output file  ",path,
		clustal_name,"njb")) == NULL) return;
        }
	}

	first_seq=1;
	last_seq=nseqs;

	if (output_tree_phylip) {
        if (phylip_name[0]!=EOS) {
                if((phylip_phy_tree_file = open_explicit_file(
                phylip_name))==NULL) return;
        }
	else {
		if((phylip_phy_tree_file = open_output_file(
		"\nEnter name for bootstrap output file  ",path,
		phylip_name,"phb")) == NULL) return;
	}
	}

	if (output_tree_nexus) {
        if (nexus_name[0]!=EOS) {
                if((nexus_phy_tree_file = open_explicit_file(
                nexus_name))==NULL) return;
        }
	else {
		if((nexus_phy_tree_file = open_output_file(
		"\nEnter name for bootstrap output file  ",path,
		nexus_name,"treb")) == NULL) return;
	}
	}

	boot_totals    = (sint *)ckalloc( (nseqs+1) * sizeof (sint) );
	for(i=0;i<nseqs+1;i++)
		boot_totals[i]=0;
		
	boot_positions = (sint *)ckalloc( (seqlen_array[first_seq]+2) * sizeof (sint) );

	for(j=1; j<=seqlen_array[first_seq]; ++j)  /* First select all positions for */
		boot_positions[j] = j;	   /* the "standard" tree */

	if(output_tree_clustal) {
		verbose = TRUE;     /* Turn on file output */
		if(dnaflag)
			overspill = dna_distance_matrix(clustal_phy_tree_file);
		else 
			overspill = prot_distance_matrix(clustal_phy_tree_file);
	}

	if(output_tree_phylip) {
		verbose = FALSE;     /* Turn off file output */
		if(dnaflag)
			overspill = dna_distance_matrix(phylip_phy_tree_file);
		else 
			overspill = prot_distance_matrix(phylip_phy_tree_file);
	}

	if(output_tree_nexus) {
		verbose = FALSE;     /* Turn off file output */
		if(dnaflag)
			overspill = dna_distance_matrix(nexus_phy_tree_file);
		else 
			overspill = prot_distance_matrix(nexus_phy_tree_file);
	}

/* check if any distances overflowed the distance corrections */
	if ( overspill > 0 ) {
		total_dists = (nseqs*(nseqs-1))/2;
		overspill_message(overspill,total_dists);
	}

	tree_gaps=ckfree((void *)tree_gaps);

	if (output_tree_clustal) verbose = TRUE;   /* Turn on screen output */

	standard_tree   = (char **) ckalloc( (nseqs+1) * sizeof (char *) );
	for(i=0; i<nseqs+1; i++) 
		standard_tree[i]   = (char *) ckalloc( (nseqs+1) * sizeof(char) );

/* compute the standard tree */

	if(output_tree_clustal || output_tree_phylip || output_tree_nexus)
		nj_tree(standard_tree,clustal_phy_tree_file);

	if (output_tree_clustal)
		fprintf(clustal_phy_tree_file,"\n\n\t\t\tBootstrap Confidence Limits\n\n");

/* save the left_branch and right_branch for phylip output */
	save_left_branch = (double *) ckalloc( (nseqs+2) * sizeof (double)   );
	save_right_branch = (double *) ckalloc( (nseqs+2) * sizeof (double)   );
	for (i=1;i<=nseqs;i++) {
		save_left_branch[i] = left_branch[i];
		save_right_branch[i] = right_branch[i];
	}
/*  
  The next line is a fossil from the days of using the cc ran()
	ran_factor = RAND_MAX / seqlen_array[first_seq]; 
*/

	if(usemenu) 
   		boot_ran_seed = 
getint("\n\nEnter seed no. for random number generator ",1,1000,boot_ran_seed);

/* do not use the native cc ran()
	srand(boot_ran_seed);
*/
       	addrandinit((unsigned long) boot_ran_seed);

	if (output_tree_clustal)
		fprintf(clustal_phy_tree_file,"\n Random number generator seed = %7u\n",
		boot_ran_seed);

	if(usemenu) 
  		boot_ntrials = 
getint("\n\nEnter number of bootstrap trials ",1,10000,boot_ntrials);

	if (output_tree_clustal) {
  		fprintf(clustal_phy_tree_file,"\n Number of bootstrap trials   = %7d\n",
	(pint)boot_ntrials);

		fprintf(clustal_phy_tree_file,
		"\n\n Diagrammatic representation of the above tree: \n");
		fprintf(clustal_phy_tree_file,"\n Each row represents 1 tree cycle;");
		fprintf(clustal_phy_tree_file," defining 2 groups.\n");
		fprintf(clustal_phy_tree_file,"\n Each column is 1 sequence; ");
		fprintf(clustal_phy_tree_file,"the stars in each line show 1 group; ");
		fprintf(clustal_phy_tree_file,"\n the dots show the other\n");
		fprintf(clustal_phy_tree_file,"\n Numbers show occurences in bootstrap samples.");
	}
/*
	print_tree(standard_tree, clustal_phy_tree_file, boot_totals);
*/
	verbose = FALSE;                   /* Turn OFF screen output */

	left_branch=ckfree((void *)left_branch);
	right_branch=ckfree((void *)right_branch);
	tkill=ckfree((void *)tkill);
	av=ckfree((void *)av);

	sample_tree   = (char **) ckalloc( (nseqs+1) * sizeof (char *) );
	for(i=0; i<nseqs+1; i++) 
		sample_tree[i]   = (char *) ckalloc( (nseqs+1) * sizeof(char) );

	if (usemenu)
	fprintf(stdout,"\n\nEach dot represents 10 trials\n\n");
        total_overspill = 0;
	nfails = 0;
	for(i=1; i<=boot_ntrials; ++i) {
		for(j=1; j<=seqlen_array[first_seq]; ++j) { /* select alignment */
							    /* positions for */
			ranno = addrand( (unsigned long) seqlen_array[1]) + 1;
			boot_positions[j] = ranno; 	    /* bootstrap sample */
		}
		if(output_tree_clustal) {
			if(dnaflag)
				overspill = dna_distance_matrix(clustal_phy_tree_file);
			else 
				overspill = prot_distance_matrix(clustal_phy_tree_file);
		}
	
		if(output_tree_phylip) {
			if(dnaflag)
				overspill = dna_distance_matrix(phylip_phy_tree_file);
			else 
				overspill = prot_distance_matrix(phylip_phy_tree_file);
		}

		if(output_tree_nexus) {
			if(dnaflag)
				overspill = dna_distance_matrix(nexus_phy_tree_file);
			else 
				overspill = prot_distance_matrix(nexus_phy_tree_file);
		}

		if( overspill > 0) {
			total_overspill = total_overspill + overspill;
			nfails++;
		}			

		tree_gaps=ckfree((void *)tree_gaps);

		if(output_tree_clustal || output_tree_phylip || output_tree_nexus) 
			nj_tree(sample_tree,clustal_phy_tree_file);

	 	left_branch=ckfree((void *)left_branch);
		right_branch=ckfree((void *)right_branch);
		tkill=ckfree((void *)tkill);
		av=ckfree((void *)av);

		compare_tree(standard_tree, sample_tree, boot_totals, last_seq-first_seq+1);
		if (usemenu) {
			if(i % 10  == 0) fprintf(stdout,".");
			if(i % 100 == 0) fprintf(stdout,"\n");
		}
	}

/* check if any distances overflowed the distance corrections */
	if ( nfails > 0 ) {
		total_dists = (nseqs*(nseqs-1))/2;
		fprintf(stdout,"\n");
		fprintf(stdout,"\n WARNING: %ld of the distances out of a total of %ld times %ld",
		(long)total_overspill,(long)total_dists,(long)boot_ntrials);
		fprintf(stdout,"\n were out of range for the distance correction.");
		fprintf(stdout,"\n This affected %d out of %d bootstrap trials.",
		(pint)nfails,(pint)boot_ntrials);
		fprintf(stdout,"\n This may not be fatal but you have been warned!");
		fprintf(stdout,"\n");
		fprintf(stdout,"\n SUGGESTIONS: 1) turn off the correction");
		fprintf(stdout,"\n           or 2) remove the most distant sequences");
		fprintf(stdout,"\n           or 3) use the PHYLIP package.");
		fprintf(stdout,"\n\n");
		if (usemenu) 
			getstr("Press [RETURN] to continue",dummy, 9);
	}


	boot_positions=ckfree((void *)boot_positions);

	for (i=1;i<nseqs+1;i++)
		sample_tree[i]=ckfree((void *)sample_tree[i]);
	sample_tree=ckfree((void *)sample_tree);
/*
	fprintf(clustal_phy_tree_file,"\n\n Bootstrap totals for each group\n");
*/
	if (output_tree_clustal)
		print_tree(standard_tree, clustal_phy_tree_file, boot_totals);

	save_tree   = (char **) ckalloc( (nseqs+1) * sizeof (char *) );
	for(i=0; i<nseqs+1; i++) 
		save_tree[i]   = (char *) ckalloc( (nseqs+1) * sizeof(char) );

	for(i=1; i<nseqs+1; i++) 
		for(j=1; j<nseqs+1; j++) 
			save_tree[i][j]  = standard_tree[i][j];

	if(output_tree_phylip) {
		left_branch 	= (double *) ckalloc( (nseqs+2) * sizeof (double)   );
		right_branch    = (double *) ckalloc( (nseqs+2) * sizeof (double)   );
		for (i=1;i<=nseqs;i++) {
			left_branch[i] = save_left_branch[i];
			right_branch[i] = save_right_branch[i];
		}
		print_phylip_tree(standard_tree,phylip_phy_tree_file,
						 bootstrap_format);
		left_branch=ckfree((void *)left_branch);
		right_branch=ckfree((void *)right_branch);
	}

	for(i=1; i<nseqs+1; i++) 
		for(j=1; j<nseqs+1; j++) 
			standard_tree[i][j]  = save_tree[i][j];

	if(output_tree_nexus) {
		left_branch 	= (double *) ckalloc( (nseqs+2) * sizeof (double)   );
		right_branch    = (double *) ckalloc( (nseqs+2) * sizeof (double)   );
		for (i=1;i<=nseqs;i++) {
			left_branch[i] = save_left_branch[i];
			right_branch[i] = save_right_branch[i];
		}
		print_nexus_tree(standard_tree,nexus_phy_tree_file,
						 bootstrap_format);
		left_branch=ckfree((void *)left_branch);
		right_branch=ckfree((void *)right_branch);
	}

	boot_totals=ckfree((void *)boot_totals);
	save_left_branch=ckfree((void *)save_left_branch);
	save_right_branch=ckfree((void *)save_right_branch);

	for (i=1;i<nseqs+1;i++)
		standard_tree[i]=ckfree((void *)standard_tree[i]);
	standard_tree=ckfree((void *)standard_tree);

	for (i=0;i<nseqs+1;i++)
		save_tree[i]=ckfree((void *)save_tree[i]);
	save_tree=ckfree((void *)save_tree);

	if (output_tree_clustal)
		fclose(clustal_phy_tree_file);

	if (output_tree_phylip)
		fclose(phylip_phy_tree_file);

	if (output_tree_nexus)
		fclose(nexus_phy_tree_file);

	if (output_tree_clustal)
		info("Bootstrap output file completed       [%s]"
		,clustal_name);
	if (output_tree_phylip)
		info("Bootstrap output file completed       [%s]"
		,phylip_name);
	if (output_tree_nexus)
		info("Bootstrap output file completed       [%s]"
		,nexus_name);
}


void compare_tree(char **tree1, char **tree2, sint *hits, sint n)
{	
	sint i,j,k;
	sint nhits1, nhits2;

	for(i=1; i<=n-3; i++)  {
		for(j=1; j<=n-3; j++)  {
			nhits1 = 0;
			nhits2 = 0;
			for(k=1; k<=n; k++) {
				if(tree1[i][k] == tree2[j][k]) nhits1++;
				if(tree1[i][k] != tree2[j][k]) nhits2++;
			}
			if((nhits1 == last_seq-first_seq+1) || (nhits2 == last_seq-first_seq+1)) hits[i]++;
		}
	}
}


void print_nexus_tree(char **tree_description, FILE *tree, sint bootstrap)
{
	sint i;
	sint old_row;
	
	fprintf(tree,"#NEXUS\n\n");

	fprintf(tree,"BEGIN TREES;\n\n");
	fprintf(tree,"\tTRANSLATE\n");
	for(i=1;i<nseqs;i++) {
		fprintf(tree,"\t\t%d	%s,\n",(pint)i,names[i]);
	}
	fprintf(tree,"\t\t%d	%s\n",(pint)nseqs,names[nseqs]);
	fprintf(tree,"\t\t;\n");

	fprintf(tree,"\tUTREE PAUP_1= ");

	if(last_seq-first_seq+1==2) {
		fprintf(tree,"(%d:%7.5f,%d:%7.5f);",first_seq,tmat[first_seq][first_seq+1],first_seq+1,tmat[first_seq][first_seq+1]);
	}
	else {

	fprintf(tree,"(");
 
	old_row=two_way_split_nexus(tree_description, tree, last_seq-first_seq+1-2,1,bootstrap);
	fprintf(tree,":%7.5f",left_branch[last_seq-first_seq+1-2]);
	if ((bootstrap==BS_BRANCH_LABELS) && (old_row>0) && (boot_totals[old_row]>0))
		fprintf(tree,"[%d]",(pint)boot_totals[old_row]);
	fprintf(tree,",");

	old_row=two_way_split_nexus(tree_description, tree, last_seq-first_seq+1-2,2,bootstrap);
	fprintf(tree,":%7.5f",left_branch[last_seq-first_seq+1-1]);
	if ((bootstrap==BS_BRANCH_LABELS) && (old_row>0) && (boot_totals[old_row]>0))
		fprintf(tree,"[%d]",(pint)boot_totals[old_row]);
	fprintf(tree,",");

	old_row=two_way_split_nexus(tree_description, tree, last_seq-first_seq+1-2,3,bootstrap);
	fprintf(tree,":%7.5f",left_branch[last_seq-first_seq+1]);
	if ((bootstrap==BS_BRANCH_LABELS) && (old_row>0) && (boot_totals[old_row]>0))
		fprintf(tree,"[%d]",(pint)boot_totals[old_row]);
	fprintf(tree,")");
        if (bootstrap==BS_NODE_LABELS) fprintf(tree,"TRICHOTOMY");
	fprintf(tree,";");
	}
	fprintf(tree,"\nENDBLOCK;\n");
}


sint two_way_split_nexus
(char **tree_description, FILE *tree, sint start_row, sint flag, sint bootstrap)
{
	sint row, new_row = 0, old_row, col, test_col = 0;
	Boolean single_seq;

	if(start_row != last_seq-first_seq+1-2) fprintf(tree,"("); 

	for(col=1; col<=last_seq-first_seq+1; col++) {
		if(tree_description[start_row][col] == flag) {
			test_col = col;
			break;
		}
	}

	single_seq = TRUE;
	for(row=start_row-1; row>=1; row--) 
		if(tree_description[row][test_col] == 1) {
			single_seq = FALSE;
			new_row = row;
			break;
		}

	if(single_seq) {
		tree_description[start_row][test_col] = 0;
		fprintf(tree,"%d",test_col+first_seq-1);
		if(start_row == last_seq-first_seq+1-2) {
			return(0);
		}

		fprintf(tree,":%7.5f,",left_branch[start_row]);
	}
	else {
		for(col=1; col<=last_seq-first_seq+1; col++) {
		    if((tree_description[start_row][col]==1)&&
		       (tree_description[new_row][col]==1))
				tree_description[start_row][col] = 0;
		}
		old_row=two_way_split_nexus(tree_description, tree, new_row, (sint)1, bootstrap);
		if(start_row == last_seq-first_seq+1-2) {
			return(new_row);
		}

		fprintf(tree,":%7.5f",left_branch[start_row]);
		if ((bootstrap==BS_BRANCH_LABELS) && (boot_totals[old_row]>0))
			fprintf(tree,"[%d]",(pint)boot_totals[old_row]);

		fprintf(tree,",");
	}


	for(col=1; col<=last_seq-first_seq+1; col++) 
		if(tree_description[start_row][col] == flag) {
			test_col = col;
			break;
		}
	
	single_seq = TRUE;
	new_row = 0;
	for(row=start_row-1; row>=1; row--) 
		if(tree_description[row][test_col] == 1) {
			single_seq = FALSE;
			new_row = row;
			break;
		}

	if(single_seq) {
		tree_description[start_row][test_col] = 0;
		fprintf(tree,"%d",test_col+first_seq-1);
		fprintf(tree,":%7.5f)",right_branch[start_row]);
	}
	else {
		for(col=1; col<=last_seq-first_seq+1; col++) {
		    if((tree_description[start_row][col]==1)&&
		       (tree_description[new_row][col]==1))
				tree_description[start_row][col] = 0;
		}
		old_row=two_way_split_nexus(tree_description, tree, new_row, (sint)1, bootstrap);
		fprintf(tree,":%7.5f",right_branch[start_row]);
		if ((bootstrap==BS_BRANCH_LABELS) && (boot_totals[old_row]>0))
			fprintf(tree,"[%d]",(pint)boot_totals[old_row]);

		fprintf(tree,")");
	}
	if ((bootstrap==BS_NODE_LABELS) && (boot_totals[start_row]>0))
			fprintf(tree,"%d",(pint)boot_totals[start_row]);
	
	return(start_row);
}


void print_phylip_tree(char **tree_description, FILE *tree, sint bootstrap)
{
	sint old_row;
	
	if(last_seq-first_seq+1==2) {
		fprintf(tree,"(%s:%7.5f,%s:%7.5f);",names[first_seq],tmat[first_seq][first_seq+1],names[first_seq+1],tmat[first_seq][first_seq+1]);
		return;
	}

	fprintf(tree,"(\n");
 
	old_row=two_way_split(tree_description, tree, last_seq-first_seq+1-2,1,bootstrap);
	fprintf(tree,":%7.5f",left_branch[last_seq-first_seq+1-2]);
	if ((bootstrap==BS_BRANCH_LABELS) && (old_row>0) && (boot_totals[old_row]>0))
		fprintf(tree,"[%d]",(pint)boot_totals[old_row]);
	fprintf(tree,",\n");

	old_row=two_way_split(tree_description, tree, last_seq-first_seq+1-2,2,bootstrap);
	fprintf(tree,":%7.5f",left_branch[last_seq-first_seq+1-1]);
	if ((bootstrap==BS_BRANCH_LABELS) && (old_row>0) && (boot_totals[old_row]>0))
		fprintf(tree,"[%d]",(pint)boot_totals[old_row]);
	fprintf(tree,",\n");

	old_row=two_way_split(tree_description, tree, last_seq-first_seq+1-2,3,bootstrap);
	fprintf(tree,":%7.5f",left_branch[last_seq-first_seq+1]);
	if ((bootstrap==BS_BRANCH_LABELS) && (old_row>0) && (boot_totals[old_row]>0))
		fprintf(tree,"[%d]",(pint)boot_totals[old_row]);
	fprintf(tree,")");
        if (bootstrap==BS_NODE_LABELS) fprintf(tree,"TRICHOTOMY");
	fprintf(tree,";\n");
}


sint two_way_split
(char **tree_description, FILE *tree, sint start_row, sint flag, sint bootstrap)
{
	sint row, new_row = 0, old_row, col, test_col = 0;
	Boolean single_seq;

	if(start_row != last_seq-first_seq+1-2) fprintf(tree,"(\n"); 

	for(col=1; col<=last_seq-first_seq+1; col++) {
		if(tree_description[start_row][col] == flag) {
			test_col = col;
			break;
		}
	}

	single_seq = TRUE;
	for(row=start_row-1; row>=1; row--) 
		if(tree_description[row][test_col] == 1) {
			single_seq = FALSE;
			new_row = row;
			break;
		}

	if(single_seq) {
		tree_description[start_row][test_col] = 0;
		fprintf(tree,"%.*s",max_names,names[test_col+first_seq-1]);
		if(start_row == last_seq-first_seq+1-2) {
			return(0);
		}

		fprintf(tree,":%7.5f,\n",left_branch[start_row]);
	}
	else {
		for(col=1; col<=last_seq-first_seq+1; col++) {
		    if((tree_description[start_row][col]==1)&&
		       (tree_description[new_row][col]==1))
				tree_description[start_row][col] = 0;
		}
		old_row=two_way_split(tree_description, tree, new_row, (sint)1, bootstrap);
		if(start_row == last_seq-first_seq+1-2) {
			return(new_row);
		}

		fprintf(tree,":%7.5f",left_branch[start_row]);
		if ((bootstrap==BS_BRANCH_LABELS) && (boot_totals[old_row]>0))
			fprintf(tree,"[%d]",(pint)boot_totals[old_row]);

		fprintf(tree,",\n");
	}


	for(col=1; col<=last_seq-first_seq+1; col++) 
		if(tree_description[start_row][col] == flag) {
			test_col = col;
			break;
		}
	
	single_seq = TRUE;
	new_row = 0;
	for(row=start_row-1; row>=1; row--) 
		if(tree_description[row][test_col] == 1) {
			single_seq = FALSE;
			new_row = row;
			break;
		}

	if(single_seq) {
		tree_description[start_row][test_col] = 0;
		fprintf(tree,"%.*s",max_names,names[test_col+first_seq-1]);
		fprintf(tree,":%7.5f)\n",right_branch[start_row]);
	}
	else {
		for(col=1; col<=last_seq-first_seq+1; col++) {
		    if((tree_description[start_row][col]==1)&&
		       (tree_description[new_row][col]==1))
				tree_description[start_row][col] = 0;
		}
		old_row=two_way_split(tree_description, tree, new_row, (sint)1, bootstrap);
		fprintf(tree,":%7.5f",right_branch[start_row]);
		if ((bootstrap==BS_BRANCH_LABELS) && (boot_totals[old_row]>0))
			fprintf(tree,"[%d]",(pint)boot_totals[old_row]);

		fprintf(tree,")\n");
	}
	if ((bootstrap==BS_NODE_LABELS) && (boot_totals[start_row]>0))
			fprintf(tree,"%d",(pint)boot_totals[start_row]);
	
	return(start_row);
}



void print_tree(char **tree_description, FILE *tree, sint *totals)
{
	sint row,col;

	fprintf(tree,"\n");

	for(row=1; row<=last_seq-first_seq+1-3; row++)  {
		fprintf(tree," \n");
		for(col=1; col<=last_seq-first_seq+1; col++) { 
			if(tree_description[row][col] == 0)
				fprintf(tree,"*");
			else
				fprintf(tree,".");
		}
		if(totals[row] > 0)
			fprintf(tree,"%7d",(pint)totals[row]);
	}
	fprintf(tree," \n");
	for(col=1; col<=last_seq-first_seq+1; col++) 
		fprintf(tree,"%1d",(pint)tree_description[last_seq-first_seq+1-2][col]);
	fprintf(tree,"\n");
}



sint dna_distance_matrix(FILE *tree)
{   
	sint m,n;
	sint j,i;
	sint res1, res2;
    sint overspill = 0;
	double p,q,e,a,b,k;	

	tree_gap_delete();  /* flag positions with gaps (tree_gaps[i] = 1 ) */
	
	if(verbose) {
		fprintf(tree,"\n");
		fprintf(tree,"\n DIST   = percentage divergence (/100)");
		fprintf(tree,"\n p      = rate of transition (A <-> G; C <-> T)");
		fprintf(tree,"\n q      = rate of transversion");
		fprintf(tree,"\n Length = number of sites used in comparison");
		fprintf(tree,"\n");
	    if(tossgaps) {
		fprintf(tree,"\n All sites with gaps (in any sequence) deleted!");
		fprintf(tree,"\n");
	    }
	    if(kimura) {
		fprintf(tree,"\n Distances corrected by Kimura's 2 parameter model:");
		fprintf(tree,"\n\n Kimura, M. (1980)");
		fprintf(tree," A simple method for estimating evolutionary ");
		fprintf(tree,"rates of base");
		fprintf(tree,"\n substitutions through comparative studies of ");
		fprintf(tree,"nucleotide sequences.");
		fprintf(tree,"\n J. Mol. Evol., 16, 111-120.");
		fprintf(tree,"\n\n");
	    }
	}

	for(m=1;   m<last_seq-first_seq+1;  ++m)     /* for every pair of sequence */
	for(n=m+1; n<=last_seq-first_seq+1; ++n) {
		p = q = e = 0.0;
		tmat[m][n] = tmat[n][m] = 0.0;
		for(i=1; i<=seqlen_array[first_seq]; ++i) {
			j = boot_positions[i];
                    	if(tossgaps && (tree_gaps[j] > 0) ) 
				goto skip;          /* gap position */
			res1 = seq_array[m+first_seq-1][j];
			res2 = seq_array[n+first_seq-1][j];
			if( (res1 == gap_pos1)     || (res1 == gap_pos2) ||
                            (res2 == gap_pos1) || (res2 == gap_pos2)) 
				goto skip;          /* gap in a seq*/
			if(!use_ambiguities)
			if( is_ambiguity((char)res1) || is_ambiguity((char)res2))
				goto skip;          /* ambiguity code in a seq*/
			e = e + 1.0;
                        if(res1 != res2) {
				if(transition(res1,res2))
					p = p + 1.0;
				else
					q = q + 1.0;
			}
		        skip:;
		}


	/* Kimura's 2 parameter correction for multiple substitutions */

		if(!kimura) {
			if (e == 0) {
				fprintf(stdout,"\n WARNING: sequences %d and %d are non-overlapping\n",m,n);
				k = 0.0;
				p = 0.0;
				q = 0.0;
			}
			else {
				k = (p+q)/e;
				if(p > 0.0)
					p = p/e;
				else
					p = 0.0;
				if(q > 0.0)
					q = q/e;
				else
					q = 0.0;
			}
			tmat[m][n] = tmat[n][m] = k;
			if(verbose)                    /* if screen output */
				fprintf(tree,        
 	     "%4d vs.%4d:  DIST = %7.4f; p = %6.4f; q = %6.4f; length = %6.0f\n"
        	                 ,(pint)m,(pint)n,k,p,q,e);
		}
		else {
			if (e == 0) {
				fprintf(stdout,"\n WARNING: sequences %d and %d are non-overlapping\n",m,n);
				p = 0.0;
				q = 0.0;
			}
			else {
				if(p > 0.0)
					p = p/e;
				else
					p = 0.0;
				if(q > 0.0)
					q = q/e;
				else
					q = 0.0;
			}

			if( ((2.0*p)+q) == 1.0 )
				a = 0.0;
			else
				a = 1.0/(1.0-(2.0*p)-q);

			if( q == 0.5 )
				b = 0.0;
			else
				b = 1.0/(1.0-(2.0*q));

/* watch for values going off the scale for the correction. */
			if( (a<=0.0) || (b<=0.0) ) {
				overspill++;
				k = 3.5;  /* arbitrary high score */ 
			}
			else 
				k = 0.5*log(a) + 0.25*log(b);
			tmat[m][n] = tmat[n][m] = k;
			if(verbose)                      /* if screen output */
	   			fprintf(tree,
             "%4d vs.%4d:  DIST = %7.4f; p = %6.4f; q = %6.4f; length = %6.0f\n"
        	                ,(pint)m,(pint)n,k,p,q,e);

		}
	}
	return overspill;	/* return the number of off-scale values */
}


sint prot_distance_matrix(FILE *tree)
{
	sint m,n;
	sint j,i;
	sint res1, res2;
    sint overspill = 0;
	double p,e,k, table_entry;	


	tree_gap_delete();  /* flag positions with gaps (tree_gaps[i] = 1 ) */
	
	if(verbose) {
		fprintf(tree,"\n");
		fprintf(tree,"\n DIST   = percentage divergence (/100)");
		fprintf(tree,"\n Length = number of sites used in comparison");
		fprintf(tree,"\n\n");
	        if(tossgaps) {
			fprintf(tree,"\n All sites with gaps (in any sequence) deleted");
			fprintf(tree,"\n");
		}
	    	if(kimura) {
			fprintf(tree,"\n Distances up tp 0.75 corrected by Kimura's empirical method:");
			fprintf(tree,"\n\n Kimura, M. (1983)");
			fprintf(tree," The Neutral Theory of Molecular Evolution.");
			fprintf(tree,"\n Page 75. Cambridge University Press, Cambridge, England.");
			fprintf(tree,"\n\n");
	    	}
	}

	for(m=1;   m<nseqs;  ++m)     /* for every pair of sequence */
	for(n=m+1; n<=nseqs; ++n) {
		p = e = 0.0;
		tmat[m][n] = tmat[n][m] = 0.0;
		for(i=1; i<=seqlen_array[1]; ++i) {
			j = boot_positions[i];
	            	if(tossgaps && (tree_gaps[j] > 0) ) goto skip; /* gap position */
			res1 = seq_array[m][j];
			res2 = seq_array[n][j];
			if( (res1 == gap_pos1)     || (res1 == gap_pos2) ||
                            (res2 == gap_pos1) || (res2 == gap_pos2)) 
                                    goto skip;   /* gap in a seq*/
			e = e + 1.0;
                        if(res1 != res2) p = p + 1.0;
		        skip:;
		}

		if(p <= 0.0) 
			k = 0.0;
		else
			k = p/e;

/* DES debug */
/* fprintf(stdout,"Seq1=%4d Seq2=%4d  k =%7.4f \n",(pint)m,(pint)n,k); */
/* DES debug */

		if(kimura) {
			if(k < 0.75) { /* use Kimura's formula */
				if(k > 0.0) k = - log(1.0 - k - (k * k/5.0) );
			}
			else {
				if(k > 0.930) {
				   overspill++;
				   k = 10.0; /* arbitrarily set to 1000% */
				}
				else {
				   table_entry = (k*1000.0) - 750.0;
                                   k = (double)dayhoff_pams[(int)table_entry];
                                   k = k/100.0;
				}
			}
		}

		tmat[m][n] = tmat[n][m] = k;
		    if(verbose)                    /* if screen output */
			fprintf(tree,        
 	                 "%4d vs.%4d  DIST = %6.4f;  length = %6.0f\n",
 	                 (pint)m,(pint)n,k,e);
	}
	return overspill;
}


void guide_tree(FILE *tree,sint firstseq,sint numseqs)
/* 
   Routine for producing unrooted NJ trees from seperately aligned
   pairwise distances.  This produces the GUIDE DENDROGRAMS in
   PHYLIP format.
*/
{
        static char **standard_tree;
        sint i;
	float dist;

	phylip_phy_tree_file=tree;
        verbose = FALSE;
	first_seq=firstseq;
	last_seq=first_seq+numseqs-1;
  
	if(numseqs==2) {
		dist=tmat[firstseq][firstseq+1]/2.0;
		fprintf(tree,"(%s:%0.5f,%s:%0.5f);\n",
			names[firstseq],dist,names[firstseq+1],dist);
	}
	else {
        standard_tree   = (char **) ckalloc( (last_seq-first_seq+2) * sizeof (char *) );
        for(i=0; i<last_seq-first_seq+2; i++)
                standard_tree[i]  = (char *) ckalloc( (last_seq-first_seq+2) * sizeof(char));

        nj_tree(standard_tree,clustal_phy_tree_file);

        print_phylip_tree(standard_tree,phylip_phy_tree_file,0);

        if(left_branch != NULL) left_branch=ckfree((void *)left_branch);
        if(right_branch != NULL) right_branch=ckfree((void *)right_branch);
        if(tkill != NULL) tkill=ckfree((void *)tkill);
        if(av != NULL) av=ckfree((void *)av);
        for (i=1;i<last_seq-first_seq+2;i++)
                standard_tree[i]=ckfree((void *)standard_tree[i]);
        standard_tree=ckfree((void *)standard_tree);
	}
        fclose(phylip_phy_tree_file);

}

static Boolean is_ambiguity(char c)
{
        int i;
	char codes[]="ACGTU";

        if(use_ambiguities==TRUE)
        {
         return FALSE;
        }

	for(i=0;i<5;i++)
        	if(amino_acid_codes[(int)c]==codes[i])
             	   return FALSE;

        return TRUE;
}

