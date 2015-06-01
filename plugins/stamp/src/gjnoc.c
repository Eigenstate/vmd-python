/******************************************************************************
 The computer software and associated documentation called STAMP hereinafter
 referred to as the WORK which is more particularly identified and described in 
 the LICENSE.  Conditions and restrictions for use of
 this package are also in the LICENSE.

 The WORK is only available to licensed institutions.

 The WORK was developed by: 
	Robert B. Russell and Geoffrey J. Barton

 Of current contact addresses:

 Robert B. Russell (RBR)             Geoffrey J. Barton (GJB)
 Bioinformatics                      EMBL-European Bioinformatics Institute
 SmithKline Beecham Pharmaceuticals  Wellcome Trust Genome Campus
 New Frontiers Science Park (North)  Hinxton, Cambridge, CB10 1SD U.K.
 Harlow, Essex, CM19 5AW, U.K.       
 Tel: +44 1279 622 884               Tel: +44 1223 494 414
 FAX: +44 1279 622 200               FAX: +44 1223 494 468
 e-mail: russelr1@mh.uk.sbphrd.com   e-mail geoff@ebi.ac.uk
                                     WWW: http://barton.ebi.ac.uk/

   The WORK is Copyright (1997,1998,1999) Robert B. Russell & Geoffrey J. Barton
	
	
	

 All use of the WORK must cite: 
 R.B. Russell and G.J. Barton, "Multiple Protein Sequence Alignment From Tertiary
  Structure Comparison: Assignment of Global and Residue Confidence Levels",
  PROTEINS: Structure, Function, and Genetics, 14:309--323 (1992).
*****************************************************************************/
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <gjutil.h>
#include <gjnoc.h>

/*#define VERY_SMALL DBL_MIN*/
#define VERY_SMALL -DBL_MAX
#define VERY_BIG DBL_MAX
#define MAX_ID_LEN 30

/* Postscript plotting constants */
#define POINT 72
#define X_OFFSET 40
#define Y_OFFSET 100

float MAXside = 11.0 * POINT; /* Max dimensions of paper */
float MINside = 8.0 * POINT;
float WTreeFraction = 0.9;    /* Fraction of max taken up by Tree */
float HTreeFraction = 0.9;    /* (Calculated after subtracting offsets) */
float TreeExtendFac = 0.05;   /* Extension of tree beyond min/max value factor */
double base_val = 1000.0;
float IdOffsetFactor =  0.01;
float NumberOffsetFactor = 0.05;
float NumberFontSize = 12;
float IdFontSize = 12;
float TitleFontSize = 12;

/****************************************************************************

   Function GJnoc:  - derived from program oc.

   General purpose cluster analysis program.  Written with flexibility
   in mind rather than all out efficiency.  

   Author: G. J. Barton (June 1993)

11/8/93:  Add single order option.  This forces the clustering to only add
a single sequence at a time to the growing cluster.  All other options
are equivalent to the full tree method.  GJB.

15/12/93:  Correct bug in definition of VERY_SMALL  ie now = -DBL_MAX.  This 
allows the complete and means linkage to work with negative numbers.

20 July 1995:  Function GJnoc.  Basically the same as oc, but can be called from 
a program.  This expects four arguments:
1. An upper diagonal array of similarities or distances 
   as defined by the GJDudarr routine.
2. A character ** array of identifiers for each entity represented by the array.
3. The number of entities.
4. A character string of parameters.  e.g. 
   "noc sim complete"  would tell the routine to expect similarities and 
   do complete linkage.  "noc dis single" = single linkage on distances.
   (You need the noc or something to give an argv[0])

2 August 1995:  The routine returns a gjnoc structure as defined in gjnoc.h.
   The structure includes the ORDER information for the tree as well as the 
   STAMP format cluster structure.

----------------------------------------------------------------------------*/

struct gjnoc *GJnoc(
		    double **arr,           /* upper diagonal array */
		    char **idents,          /* identifiers for entities */
		    int n,                  /* number of entities in array */
		    char *parms             /* string of parameters separated by spaces */
		    )
{
	extern FILE *std_in,*std_out,*std_err;
	extern double base_val;
	int nclust;
	int i,j,k,l;
	int i1loc=0,j1loc=0;
	int i2loc=0,j2loc=0;
	int i3loc=0,j3loc=0;
	double U1val,U2val,U3val,rval;
	
	struct sc *clust;
	struct sc **free_list; /* list of pointers to sc structures that are 
				  allocated by make_entity these are the ones 
				  that are not part of the main clust array, 
				  but must be 
				  freed before returning from this routine. */
	int nfree_list;

        int *unclust;      /* list of unclustered entities */
	int nunclust;      /* number of unclustered entities */

	int *notparent;    /* list of clusters that have no children */
	int nnotparent;    /* number of clusters that have no children */

	int *inx;          /* array of indexes - inx[i] is the re-ordered position */

	char *postfile=0;    /* name of PostScript output file */

	struct gjnoc *ret_val;  /* cluster structure to return for STAMP etc.*/

	/* TIME INFORMATION */
	clock_t start_time, end_time,initial_time,final_time;

	/* options */
	int sim;        /* similarity scores =1, distances = 2*/
	int type;       /* single linkage = 1, complete linkage = 2, means = 3 */
	int option=0;     /* use to feed the Grp_Grp function */
	int ps;      
	int dolog;      /* flag to take logs on output to .ps file */
	double cut=0;
	int do_cut;
	int showid;
	int amps_out;   /* output tree file and order file for amps */
	int timeclus;     /* set to 1 if timing output to std_err required */
	int single_order;
	int is_noc;     /* flag to say that this is the function version of 
			   oc */

	/* amps files */
	char *amps_order_file=0;
	char *amps_tree_file=0;
	FILE *ford;
	FILE *ftree=0;

	/* token structure - this is copied to nargs and argv */
	struct tokens *ptok;
	int argc;
	char **argv;

	FILE *fp;

	GJinitfile();

	sim = 1;
	type = 1;
	ps = 0;     /* default is no PostScript output */
	showid = 0;
	do_cut = 0;
	timeclus = 0;
	amps_out = 0;
	dolog = 0;
	single_order = 0;
	
	is_noc = 1;  /* always true when this is function noc
			means that the function will return a 
			cluster structure */

	nfree_list = 1;
	
	if(is_noc) {
	  /* since we don't want any printed output to std_err just redefine
	     this to /dev/null  (A neat advantage of not using printf... */

#if defined(_MSC_VER)
          std_err=stdout;
#else
  	  std_err = GJfopen("/dev/null","w",0);
#endif
	  /* parse the command string into argv */
	  /* first get the args into  a tokens structure */
	  ptok = GJgettokens(" \n\t",parms);
	  /* copy number of tokens and pointer to token array */
	  argc = ptok->ntok;
	  argv = ptok->tok;
	}

	start_time = clock();
	initial_time = start_time;

	if(argc > 1){
	  for(i=1;i<argc;++i){
	    if(strcmp(argv[i],"sim")==0){
	      sim = 1;
	      base_val = VERY_BIG;
	      cut = VERY_SMALL;
	    }else if(strcmp(argv[i],"dis")==0){
	      sim = 2;
	      base_val = VERY_SMALL;
	      cut = VERY_BIG;
	    }else if(strcmp(argv[i],"single")==0){
	      type = 1;
	    }else if(strcmp(argv[i],"complete")==0){
	      type = 2;
	    }else if(strcmp(argv[i],"means")==0){
	      type = 3;
	    }else if(strcmp(argv[i],"order")==0){
	      /* do a single order clustering rather than a full tree */
	      single_order = 1;
	    }else if(strcmp(argv[i],"ps")==0){
	      ++i;
	      ps = 1;
	      postfile = GJcat(2,argv[i],".ps");
	    }else if(strcmp(argv[i],"log")==0){
	      dolog = 1;
	    }else if(strcmp(argv[i],"cut")==0){
	      ++i;
	      cut = atof(argv[i]);
	      do_cut = 1;
	    }else if(strcmp(argv[i],"amps")==0){
	      amps_out = 1;
	      ++i;
	      if(single_order){
		amps_order_file = GJcat(2,argv[i],".ord");
	      }else{
		amps_tree_file = GJcat(2,argv[i],".tree");
		amps_order_file = GJcat(2,argv[i],".tord");
	      }
	    }else if(strcmp(argv[i],"id")==0){
	      showid = 1;                      /* output idents rather than index */
	    }else if(strcmp(argv[i],"timeclus")==0){
	      timeclus = 1;                      /* output idents rather than index */
	    }else{
	      fprintf(std_err,"Unrecognised Option\n");
	    }
	  }
	  }else{
	    fprintf(std_err,"Cluster analysis program\n\n");
	    fprintf(std_err,"Usage: oc <sim/dis> <single/complete/means> <ps> <cut N>\n\n");
	    fprintf(std_err,"Version 1.0 - Requires a file to be piped to standard input\n");
	    fprintf(std_err,"Format:  Line   1:  Number (N) of entities to cluster (e.g. 10)\n");
	    fprintf(std_err,"Format:  Lines 2 to 2+N-1:  Identifier codes for the entities (e.g. Entity1)\n");
	    fprintf(std_err,"Format:  N*(N-1)/2:  Distances, or similarities - ie the upper diagonal\n\n");

	    fprintf(std_err,"Options:\n");

	    fprintf(std_err,"sim = similarity /  dis = distances\n");
	    fprintf(std_err,"method = single/complete/means\n");
	    fprintf(std_err,"ps <file> = plot out dendrogram to <file.ps> \n");
	    fprintf(std_err,"log = take logs before calculation \n");
	    fprintf(std_err,"cut = only show clusters above/below the cutoff\n");
	    fprintf(std_err,"id = output identifier codes rather than indexes for entities\n");
	    fprintf(std_err,"timeclus = output times to generate each cluster\n");
	    fprintf(std_err,"amps <file> = produce amps <file>.tree and <file>.tord files\n");
	    exit(0);
	  }

	if(!is_noc) {
          /* allocate and read in the upper diagonal array - only if not noc*/
	  fprintf(std_err,"Reading Upper Diagonal\n");
	  fscanf(std_in,"%d",&n);
	  idents = read_idents(std_in,n);
          arr = read_up_diag(std_in,n);
	}

	GJfreetokens(ptok);

	if(dolog){
	  up_diag_log(arr,n);
	}


/*      write_up_diag(std_out,arr,n);*/
        fprintf(std_err,"Read: %d Entries\n",n);
	end_time = clock();
	
	fprintf(std_err,"CPU time: %f seconds\n",
    		((float) (end_time - start_time))/CLOCKS_PER_SEC); 
	start_time = end_time;
        
	/* holds index of unclustered entities */

	fprintf(std_err,"Setting up unclust\n");
	unclust = (int *) GJmalloc(sizeof(int) * n);
	for(i=0;i<n;++i){
		unclust[i] = i;
	}
	nunclust = n;

	fprintf(std_err,"Setting up notparent\n");
	notparent = (int *) GJmalloc(sizeof(int) * n);
	for(i=0;i<n;++i){
		notparent[i] = 0;
	}
	nnotparent = 0;
	
	/* set up the clust array to hold results */
	/* there are always n-1 clusters formed - we will also store the nth (all entities)*/
	fprintf(std_err,"Setting up clust\n");
	clust = (struct sc *) GJmalloc(sizeof(struct sc) * n);
	
	/* allocate space for the free list pointers - for now just length 1 */
	free_list = (struct sc **) GJmalloc(sizeof(struct sc *) * nfree_list);

	end_time = clock();
	fprintf(std_err,"CPU time: %f seconds\n",
		((float) (end_time - start_time))/CLOCKS_PER_SEC); 
	start_time = end_time;

	nclust = 0;
	U1val = VERY_SMALL;
	U2val = VERY_SMALL;
	U3val = VERY_SMALL;
	if(sim == 1){
	  if(type == 1){
	    fprintf(std_err,"Single linkage on similarity\n");
	    option = 1;
	  }else if(type == 2){
	    fprintf(std_err,"Complete linkage on similarity\n");
	    option = 0;
	  }else if(type == 3){
	    fprintf(std_err,"Means linkage on similarity\n");
	    option = 2;
	  }
	}else{
	  if(type == 1){
	    fprintf(std_err,"Single linkage on distance\n");
	    option = 0;
	  }else if(type == 2){
	    fprintf(std_err,"Complete linkage on distance\n");
	    option = 1;
	  }else if(type == 3){
	    fprintf(std_err,"Means linkage on distance\n");
	    option = 2;
	  }
	}

/* implement max similarity single linkage first */

	fprintf(std_err,"Doing Cluster Analysis...\n");
	while(nclust < (n-1)){
	        if(sim == 1){
		  U1val = VERY_SMALL;
		  U2val = VERY_SMALL;
		  U3val = VERY_SMALL;
		}else{
  		  U1val = VERY_BIG;
		  U2val = VERY_BIG;
		  U3val = VERY_BIG;
		}
		if(nunclust > 0){
/*			fprintf(std_err,"Option 1\n");*/
		        if(!single_order || (single_order && nclust == 0)){
			  /* we've not clustered all entities into groups */
			  /* find max of what is left */
			  for(i=0;i<nunclust-1;++i){
				for(j=(i+1);j<nunclust;++j){
					rval = val_A_B(arr,n,unclust[i],unclust[j]);
					if(sim == 1){
					  if(rval > U1val){
						U1val = rval;
						i1loc = unclust[i];
						j1loc = unclust[j];
					  }
					}else{
					  if(rval < U1val){
						U1val = rval;
						i1loc = unclust[i];
						j1loc = unclust[j];
					  }
					}
				}
			  } /* ... for(i=0... */ 
			} /* ...if(!single_order...*/
		}
		if(nnotparent > 0 && nunclust > 0){
/*			fprintf(std_err,"Option 2\n");*/
			/* we have some clusters */
                        /* so compare them to the unclustered entitites */
			for(i=0;i<nunclust;++i){
				for(j=0;j<nnotparent;++j){
				        k = notparent[j];
					rval = val_Grp_B(arr,n,clust[k].members,clust[k].n,unclust[i],option);
					if(sim == 1){
					  if(rval >= U2val){
					        /* need >= because want biggest previous cluster */
						U2val = rval;
						i2loc = unclust[i];
						j2loc = k;
					  }
					}else{
					  if(rval <= U2val){
					        /* need <= because want biggest previous cluster */
						U2val = rval;
						i2loc = unclust[i];
						j2loc = k;
					  }
					}
				}
			}
		}
		if(nnotparent > 1){
/*			fprintf(std_err,"Option 3\n");*/
                        /* compare clusters to clusters if there are more than 1*/
                        for(i=0;i<nnotparent-1;++i){
			    k = notparent[i];
                            for(j=(i+1);j<nnotparent;++j){
			          l = notparent[j];
				  rval = val_Grp_Grp(arr,n,clust[k].members,clust[k].n,
                                                         clust[l].members,clust[l].n,
                                                         option);
				  if(sim == 1){
				    if(rval >= U3val){
					U3val = rval;
					i3loc = k;
					j3loc = l;
				    } /* if(rval >= U3val... */
				  }else{
    				    if(rval <= U3val){
					U3val = rval;
					i3loc = k;
					j3loc = l;
				    } /* if(rval >= U3val... */
				  }
			      }  /* if j=(i+1) ... */
			} /* for(i=0... */ 
		 }
		/* find which search gave the biggest (or smallest) value */
    		if((sim ==1 && U3val >= U1val && U3val >= U2val)||
		   (sim ==2 && U3val <= U1val && U3val <= U2val)){
    		    /* joining of  two clusters */
    		    clust[nclust].score = U3val;
    		    clust[nclust].n = clust[i3loc].n + clust[j3loc].n;
    		    clust[nclust].members = (int *) GJmalloc(sizeof(int) * clust[nclust].n);
    		    for(i=0;i<clust[i3loc].n;++i){
    		        clust[nclust].members[i] = clust[i3loc].members[i];
    		    }
    		    for(i=0;i<clust[j3loc].n;++i){
    		        clust[nclust].members[i+clust[i3loc].n] = clust[j3loc].members[i];
    		    }
    		    clust[nclust].parentA = &clust[i3loc];
    		    clust[nclust].parentB = &clust[j3loc];
		    clust[nclust].lab = 1;
		    remove_notparent(notparent,&nnotparent,i3loc);
		    sub_notparent(notparent,&nnotparent,j3loc,nclust);
    		}else if((sim==1 && U2val >= U1val && U2val >= U3val)||
			 (sim==2 && U2val <= U1val && U2val <= U3val)){
    		    /* joining of  a single entity to an existing cluster */
    		    clust[nclust].score = U2val;
    		    clust[nclust].n = 1 + clust[j2loc].n;
    		    clust[nclust].members = (int *) GJmalloc(sizeof(int) * clust[nclust].n);
    		    /* copy previous cluster list into this one - not strictly neccessary */
    		    /* but done to simplify cluster comparisons above*/
    		    for(i=1;i<clust[nclust].n;++i){
    		        clust[nclust].members[i] = clust[j2loc].members[i-1];
    		    }
    		    /* stick solitary cluster member on the beginning */
    		    clust[nclust].members[0] = i2loc;
    		    /* put in parent pointers */
    		    clust[nclust].parentA = make_entity(i2loc,base_val);

		    free_list[nfree_list - 1] = clust[nclust].parentA;
		    ++nfree_list;
		    free_list = (struct sc **) GJrealloc(free_list, sizeof(struct sc *) * nfree_list);

    		    clust[nclust].parentB = &clust[j2loc];
		    clust[nclust].lab = 1;
		    sub_notparent(notparent,&nnotparent,j2loc,nclust);
    		    remove_unclust(unclust,&nunclust,i2loc);
		}else if((sim == 1 && U1val >= U2val && U1val >= U3val)||
			 (sim == 2 && U1val <= U2val && U1val <= U3val)){
		    /* unclustered pair were biggest */
		    clust[nclust].score = U1val;
    		    clust[nclust].n = 2;
		    clust[nclust].members = (int *) GJmalloc(sizeof(int) * clust[nclust].n);
		    clust[nclust].members[0] = i1loc;
		    clust[nclust].members[1] = j1loc;
		    clust[nclust].parentA = make_entity(i1loc,base_val);
		    free_list[nfree_list - 1] = clust[nclust].parentA;
		    ++nfree_list;
		    free_list = (struct sc **) GJrealloc(free_list, sizeof(struct sc *) * nfree_list);

    		    clust[nclust].parentB = make_entity(j1loc,base_val);
		    free_list[nfree_list - 1] = clust[nclust].parentB;
		    ++nfree_list;
		    free_list = (struct sc **) GJrealloc(free_list, sizeof(struct sc *) * nfree_list);

    		    clust[nclust].lab = 1;
		    add_notparent(notparent,&nnotparent,nclust);
    		    remove_unclust(unclust,&nunclust,i1loc);
		    remove_unclust(unclust,&nunclust,j1loc);
		}

/*    		fprintf(std_err,"Current cluster\n");*/
/*    		show_entity(&clust[nclust],std_err);*/

		if(timeclus){
		  fprintf(std_err,"Cluster: %d DONE!  ",nclust);
		  end_time = clock();
		  fprintf(std_err,"CPU time: %f seconds\n",
			((float) (end_time - start_time))/CLOCKS_PER_SEC); 
		  start_time = end_time;
		}

		++nclust;
	}
	/* now we should have all clusters stored in the clust array */
	/* set up the inx array */
	inx = (int *) GJmalloc(sizeof(int) * n);
	for(i=0;i<clust[nclust-1].n;++i){
	    inx[clust[nclust-1].members[i]]=i;
	}

	if(do_cut){
	  /* only output the biggest clusters that are above the cutoff */
	  if(sim == 1){
	    for(i=(nclust-1);i>=0;--i){
	      if(clust[i].lab == 1){
		/* only output clusters that aren't already output as part of something bigger */
	        if(clust[i].score >= cut){
	          fprintf(std_out,"## %d %g %d\n",i,clust[i].score,clust[i].n);
		  show_id_entity(&clust[i],idents,std_out);
		  mark_parents(&clust[i]);
	        }
	      }
	    }
	  }else{
	    for(i=(nclust-1);i>=0;--i){
	      if(clust[i].lab == 1){
		/* only output clusters that aren't already output as part of something bigger */
	        if(clust[i].score <= cut){
	          fprintf(std_out,"## %d %g %d\n",i,clust[i].score,clust[i].n);
		  show_id_entity(&clust[i],idents,std_out);
		  mark_parents(&clust[i]);
		}
	      }
	    }
	  }
	  /* now output the unclustered entities */
	  fprintf(std_out,"\nUNCLUSTERED ENTITIES\n");
	  write_unclustered(&clust[nclust-1],idents,std_out);

	}else{
	  /* output all clusters from smallest to largest */
	  if(amps_out){
	    if(!single_order){
	      fprintf(std_err,"Opening AMPS order and tree files for writing\n");
	      ford = GJfopen(amps_order_file,"w",1);
	      ftree = GJfopen(amps_tree_file,"w",1);
	    }else{
	      fprintf(std_err,"Opening AMPS order file for writing\n");
	      ford = GJfopen(amps_order_file,"w",1);
	    }
	    /* output order file */
	    if(single_order){
	      /* order file must be reversed */
	      for(i=nclust;i> -1;--i){
	        fprintf(ford,"%11d%20.2f%10d\n",clust[nclust-1].members[i]+1,0.0,0);
	      }
	    }else{
	      /* order file for use with the tree */
	      for(i=0;i<(nclust+1);++i){
		fprintf(ford,"%11d%20.2f%10d\n",clust[nclust-1].members[i]+1,0.0,0);
	      }
	    }
	    GJfclose(ford,1);
	    if(!single_order){
	      /* output the tree_file */
	      for(i=0;i<nclust;++i){
		print_amps_cluster(clust[i].parentA,inx,ftree);
		print_amps_cluster(clust[i].parentB,inx,ftree);
	      }
	      GJfclose(ftree,1);
	    }
	  }else if(is_noc) {
	    /* set up and return the cluster array for STAMP */
	    ret_val = (struct gjnoc *) GJmalloc(sizeof(struct gjnoc));
	    ret_val->clusters = (struct cluster *) GJmalloc(sizeof(struct cluster) * (nclust));
	    ret_val->order = (int *) GJmalloc(sizeof(int) * (nclust+1));

            for(i=0;i<nclust;++i){
	        ret_val->clusters[i].a.number = clust[i].parentA->n;
		ret_val->clusters[i].a.member = GJmalloc(sizeof(int) * clust[i].parentA->n);
		for(j=0;j<clust[i].parentA->n;++j) {
		  ret_val->clusters[i].a.member[j] = inx[clust[i].parentA->members[j]];
		}
	        ret_val->clusters[i].b.number = clust[i].parentB->n;
		ret_val->clusters[i].b.member = GJmalloc(sizeof(int) * clust[i].parentB->n);
		for(j=0;j<clust[i].parentB->n;++j) {
		  ret_val->clusters[i].b.member[j] = inx[clust[i].parentB->members[j]];
		}

	    }
	    /* copy over the order */
	    for(i=0;i<(nclust+1);++i) {
		  ret_val->order[i] = clust[nclust-1].members[i];
	    }

	    GJfclose(std_err,0);
	    std_err = stderr;
    	    /* free all memory  - 
	       freeing clust is not straightforward - this is why we have
	       free_list of pointers*/
	    GJfree(unclust);
	    GJfree(notparent);
	    GJfree(inx);
	    /* free up the clust array.  This is not straightforward to do
	       since some of the parentA/B's are allocated and some are addresses
	       of other parts of the clust array.  free does not complain about
	       trying to free memory that was not allocated with malloc.  However, 
	       since this is a possible bug on some machines/malloc's we keep the 
	       free_list pointers every time we use make_entity*/
	    for(i=0;i<nclust;++i) {
	      GJfree(clust[i].members);
	    }
	    for(i=0;i<(nfree_list-1);++i) {
	      GJfree(free_list[i]->members);
	      GJfree(free_list[i]);
	    }
	    GJfree(clust);
	    GJfree(free_list);
	    return ret_val;
	  }else{
	    for(i=0;i<nclust;++i){
	      fprintf(std_out,"## %d %g %d\n",i,clust[i].score,clust[i].n);
	      if(showid){
		show_id_entity(&clust[i],idents,std_out);
	      }else{
		show_entity(&clust[i],std_out);
	      }

	      /*	    show_inx_entity(&clust[i],inx,std_out);*/
	    
	      /*	    fprintf(std_out,"Parents:\n");
			    show_entity(clust[i].parentA,std_out);
			    show_entity(clust[i].parentB,std_out);
			    */
	    }
	  }
	}

	if(ps == 1){
	  fp = GJfopen(postfile,"w",1);
          draw_dendrogram(fp,clust,idents,inx,nclust,sim);
	}

	final_time = clock();
	fprintf(std_err,"Total CPU time: %f seconds\n",
    		(float) (final_time - initial_time)/CLOCKS_PER_SEC); 
	return 0;
}

/**********************************************************************
read_up_diag:
reads a file containing n, the number of entities, followed by
n*(n-1)/2 numbers that are the pair distances or similarities between
the n entities.

Returns an upper diagonal type pointer to pointer array

-----------------------------------------------------------------------*/
double **read_up_diag(FILE *infile, int n)
{
    double **ret_val;
    int i,j;
    
/*    fscanf(infile,"%d",n);*/
    ret_val = (double **) GJDudarr(n);
    for(i=0;i<(n-1);++i){
        for(j=0;j<(n-i-1);++j){
            fscanf(infile,"%lf", &ret_val[i][j]);
/*            fprintf(stderr,"%f\n",ret_val[i][j]);*/
        }
    }
    return ret_val;
}

/*********************************************************************
up_diag_log:  take logs for the upper diagonal structure
-------------------------------------------------------------------*/

void up_diag_log(double **arr,int n)
{
  int i,j;

  for(i=0;i<(n-1);++i){
      for(j=0;j<(n-i-1);++j){
	arr[i][j] = log(arr[i][j]);
      }
  }
}

/**********************************************************************
read_idents:
reads n identifiers from infile

returns a pointer to an array of identifiers
-----------------------------------------------------------------------*/
char **read_idents(FILE *infile, int n)
{
    char **ret_val;
    int i;
    char *buff/*,tbuff*/;

    buff = GJstrcreate(MAX_ID_LEN,NULL);
    ret_val = (char **) GJmalloc(sizeof(char *) * n);
    
    for(i=0;i<n;++i){
      fscanf(infile,"%s",buff);
      ret_val[i] = GJstrdup(buff);
    }
    GJfree(buff);
    return ret_val;
}

/**********************************************************************
write_up_diag:
writes out n, the number of entities, followed by
n*(n-1)/2 numbers that are the pair distances or similarities between
the n entities.

-----------------------------------------------------------------------*/
void write_up_diag(FILE *infile, double **ret_val,int n)
{
    int i,j;
    
    fprintf(infile,"Number of Entities: %d\n",n);
    for(i=0;i<(n-1);++i){
        for(j=0;j<(n-i-1);++j){
            fprintf(infile," %f",ret_val[i][j]);
        }
        fprintf(infile,"\n");
    }
}
    		    
/**********************************************************************
show_entity:

print out the score, number of members and the members for
the entity
----------------------------------------------------------------------*/
void show_entity(struct sc *entity,FILE *out)
{
    int i;

    if(entity == NULL){
        fprintf(out,"No entity\n");
        return;
    }

    fprintf(out,"Entity Score: %g  Number of members: %d\n",
            entity->score,entity->n);

    for(i=0;i < entity->n;++i){
        fprintf(out," %d",entity->members[i]);
    }
    fprintf(out,"\n");
}
/**********************************************************************
show_id_entity:

print out the score, number of members and the members for
the entity using the id codes rather than numerical positions
----------------------------------------------------------------------*/
void show_id_entity(struct sc *entity,char **id,FILE *out)
{
    int i;

    if(entity == NULL){
        fprintf(out,"No entity\n");
        return;
    }

/*    fprintf(out,"Entity Score: %f  Number of members: %d\n",
            entity->score,entity->n);
*/

    for(i=0;i < entity->n;++i){
        fprintf(out," %s",id[entity->members[i]]);
    }
    fprintf(out,"\n");
}
/**********************************************************************
show_inx_entity:

print out the score, number of members and the members for
the entity - followed by the index for the entity
----------------------------------------------------------------------*/
void show_inx_entity(struct sc *entity,int *inx,FILE *out)
{
    int i;

    if(entity == NULL){
        fprintf(out,"No entity\n");
        return;
    }

    fprintf(out,"Entity Score: %g  Number of members: %d\n",
            entity->score,entity->n);
    for(i=0;i < entity->n;++i){
        fprintf(out," %d",entity->members[i]);
    }
    fprintf(out,"\n");
    for(i=0;i < entity->n;++i){
        fprintf(out," %d",inx[entity->members[i]]);
    }
    fprintf(out,"\n");
}
/**********************************************************************
print_amps_cluster:

print out the cluster in AMPS tree file format.
ie 1x,20i5  Fortran format.

inx is necessary to so that the identifying codes are re-ordered
appropriately for AMPS

Need +1 cos AMPS uses sequence numbering from 1...N not 0...N-1

----------------------------------------------------------------------*/
void print_amps_cluster(struct sc *entity,int *inx,FILE *fp)
{
  int i;
  int j;
  
  fprintf(fp," %5d\n",entity->n);
  
  j=0;
  for(i=0;i<entity->n;++i){
    if(j == 20){
      j = 0;
    }
    if(j == 0){
      fprintf(fp," %5d",inx[entity->members[i]]+1);
    }else{
      fprintf(fp,"%5d",inx[entity->members[i]]+1);
    }
    if(j == 19){
      fprintf(fp,"\n");
    }
    ++j;
  }
  /* change from < to <= to avoid blunders at end of line */
  if(j <= 19) fprintf(fp,"\n");
}

/******************************************************************
make_entity: return pointer to malloc'd space for a struct sc
	structure for a single entity. i.
	base_val: The value form minimum distance, or maximum similarity.
-------------------------------------------------------------------*/
struct sc *make_entity(int i,double base_val)
{
	struct sc *ret_val;
	ret_val = (struct sc *) GJmalloc(sizeof(struct sc));
	ret_val->score = base_val;
	ret_val->n = 1;
	ret_val->members = (int *) GJmalloc(sizeof(int));
	ret_val->members[0] = i;
	ret_val->parentA = NULL;
	ret_val->parentB = NULL;
	ret_val->lab = 1;
	return ret_val;
}
	
/******************************************************************
val_A_B: Given the upper diagonal array, return the value stored
         in the notional arr[i][j].
         Where 0 <= i < n-1
         and   i < j < n

Note:  This could/should be turned into a macro - without the bounds
       check.
------------------------------------------------------------------

double val_A_B(
	double **arr,   
    	int n,          
    	int i,          
    	int j)          

{
    extern FILE *std_err;
    if(!(i >= 0 && i < n-1 && i < j && j < n)){
    	GJerror("Invalid request in val_A_B");
    	fprintf(std_err,"i,j,n %d %d %d \n",i,j,n);
    	exit(-1);
    }

    return arr[i][j-i-1];
}
*/         
/*******************************************************************
val_Grp_B: Given upper diagonal array,
	find min, max, or mean of comparison between the
	entities in grp and j
	choice = 0 =  find min value of grp-grp comparison
		 1 =  find max ...
		 2 =  find mean ...
		 
Note:  there is some duplication in the code to avoid testing choice
       on every step in the loop
--------------------------------------------------------------------*/
double val_Grp_B(
	double **arr,	/* upper diagonal array */
	int n,		/* side of array */
	int *grpA,	/* array containing entities of group */
	int ng,		/* number of members in grpA */
	int j,		/* entity to compare to grpA */
	int choice	/* see above */
)
{
	int k;
	double val=0;
	double rval;

	if(choice == 0){
		/* min case */
		val = VERY_BIG;
		for(k=0;k<ng;++k){
			if(grpA[k] < j){
				rval = val_A_B(arr,n,grpA[k],j);
			}else{
				rval = val_A_B(arr,n,j,grpA[k]);
			}
			if(rval < val) val = rval;
		}
	}else if (choice == 1){
		val = VERY_SMALL;
		for(k=0;k<ng;++k){
			if(grpA[k] < j){
				rval = val_A_B(arr,n,grpA[k],j);
			}else{
				rval = val_A_B(arr,n,j,grpA[k]);
			}
			if(rval > val) val = rval;
		}
	}else if (choice == 2){
		val = 0.0;
		for(k=0;k<ng;++k){
			if(grpA[k] < j){
				rval = val_A_B(arr,n,grpA[k],j);
			}else{
				rval = val_A_B(arr,n,j,grpA[k]);
			}
			val += rval;
		}
		val /= ng;
	}
	return val;
}
/*******************************************************************
val_Grp_Grp: Given upper diagonal array,
	find min, max, or mean of comparison between the
	entities in grpA and grpB
	choice = 0 =  find min value of grp-grp comparison
		 1 =  find max ...
		 2 =  find mean ...
		 
Note:  there is some duplication in the code to avoid testing choice
       on every step in the loop
--------------------------------------------------------------------*/
double val_Grp_Grp(
	double **arr,	/* upper diagonal array */
	int n,		/* side of array */
	int *grpA,	/* array containing entities of group */
	int ngA,	/* number of members in grpA */
	int *grpB,	/* array containing entities of group B*/
	int ngB,	/* number of members in grpB */
	int choice	/* see above */
)
{
	int k,l;
	double val=0;
	double rval;

	if(choice == 0){
		/* min case */
		val = VERY_BIG;
		for(k=0;k<ngA;++k){
			for(l=0;l<ngB;++l){
				if(grpA[k] < grpB[l]){
					rval = val_A_B(arr,n,grpA[k],grpB[l]);
				}else{
					rval = val_A_B(arr,n,grpB[l],grpA[k]);
				}
				if(rval < val) val = rval;
			}
		}
	}else if (choice == 1){
		val = VERY_SMALL;
		for(k=0;k<ngA;++k){
			for(l=0;l<ngB;++l){
				if(grpA[k] < grpB[l]){
					rval = val_A_B(arr,n,grpA[k],grpB[l]);
				}else{
					rval = val_A_B(arr,n,grpB[l],grpA[k]);
				}
				if(rval > val) val = rval;
			}
		}
	}else if (choice == 2){
		val = 0.0;
		for(k=0;k<ngA;++k){
			for(l=0;l<ngB;++l){
				if(grpA[k] < grpB[l]){
					rval = val_A_B(arr,n,grpA[k],grpB[l]);
				}else{
					rval = val_A_B(arr,n,grpB[l],grpA[k]);
				}
				val += rval;
			}
		}
		val /= (ngA * ngB);
	}
	return val;
}
/*****************************************************************
remove_unclust:

Take the array containing integers showing members that have
yet to be clustered and remove an element.  nunclust is returned
reduced by one.
-----------------------------------------------------------------*/
void remove_unclust(int *unclust,int *nunclust,int val)
{
    int i,j,found;
    i = 0;
    found = 0;
    
    while(i < *nunclust){
        if(unclust[i] == val){
            ++i;
            found = 1;
        }
        if(found){
            j = i - 1;
        }else{
            j = i;
        }
        unclust[j] = unclust[i];
        ++i;
    }
    --(*nunclust);
}

void iprintarr(int *arr,int n,FILE *out)
{
  int i;
  for(i=0;i<n;++i){
    fprintf(out," %d",arr[i]);
  }
  fprintf(out,"\n");
}

/******************************************************************
no_share:  return 1 if grpA and grpB have no common members
                 0 if there are common members
----------------------------------------------------------------*/

int no_share(
	    int *grpA,
	    int ngA,
	    int *grpB,
	    int ngB
)
{
  int i,j;

  for(i=0;i<ngA;++i){
    for(j=0;j<ngB;++j){
      if(grpA[i]==grpB[j])return 0;
    }
  }
  return 1;
}
/*****************************************************************
sub_notparent:  given the notparent list substitutes A by B
-----------------------------------------------------------------*/
void sub_notparent(int *np,int *n,int A,int B)
{
  int i;
  for(i=0;i<*n;++i){
    if(np[i] == A){
      np[i] = B;
      return;
    }
  }
}
/*****************************************************************
remove_notparent:  given the notparent list removes A from the list
                   
-----------------------------------------------------------------*/
void remove_notparent(int *np,int *n,int A)
{
   remove_unclust(np,n,A);
}
/*****************************************************************
add_notparent:  given the notparent list adds A to the list
                   
-----------------------------------------------------------------*/
void add_notparent(int *np,int *n,int A)
{
   np[*n] = A;
   ++(*n);
}

/**************************************************************************
Return a pointer to space for an upper diagonal square array stored rowwise
G. J. B. 16 May 1993

double precision array.
side = n;
--------------------------------------------------------------------------*/
double **GJDudarr(int n)

{
	double **ret_val;
	int i,nm1;
	
	nm1 = n - 1;
	ret_val = (double **) GJmalloc(sizeof(double *) * nm1);


	for(i=0;i<nm1;++i){
		ret_val[i] = (double *) GJmalloc(sizeof(double) * (nm1 - i));
	}
		
	return ret_val;
}
/***************************************************************************
draw_dendrogram:

Output the dendrogram to fout using PostScript commands

---------------------------------------------------------------------------*/
void draw_dendrogram(FILE *fout,
                     struct sc *clust,
		     char **idents,
                     int *inx,
                     int n,
		     int sim
) 
{
    extern double base_val;
    extern float MINside,MAXside;
    extern float WTreeFraction,HTreeFraction,TreeExtendFac;

    float Twidth,Theight;          /* x and y limits for the tree part of the plot */
    float Xoffset,Yoffset;         /* x and y offset (points) */

    int i;
    double x1,x2,x3,y1,y2;
    double Xrange,Yrange,Xmin,Xmax,Ymin,Ymax/*,TXmax*/;
/*    double Xscale,Yscale; */
    double Xid,Yid;

    char *buff;

    buff = GJstrcreate(MAX_ID_LEN,NULL);

    Xoffset = X_OFFSET;
    Yoffset = Y_OFFSET;

    Twidth = (MINside * WTreeFraction) - Xoffset;
    Theight = (MAXside * HTreeFraction) - Yoffset;

    Xmin = VERY_BIG;
    Xmax = VERY_SMALL;
    Ymin = VERY_BIG;
    Ymax = VERY_SMALL;

    /* find the min and max in X excluding the base_val */
    for(i=0;i<n;++i){
      if(sim ==1){
	if(clust[i].score < Xmin)Xmin = clust[i].score;
	if(clust[i].score != base_val && clust[i].score > Xmax)Xmax = clust[i].score;
      }else{
	if(clust[i].score != base_val && clust[i].score < Xmin)Xmin = clust[i].score;
	if(clust[i].score > Xmax)Xmax = clust[i].score;
      }
    }
    /* set the max/min value slightly bigger/smaller than the actual max/min */

    if(sim == 1)Xmax *= (1.0 + TreeExtendFac);
    if(sim == 2)Xmin *= (1.0 - TreeExtendFac);
    Xrange = Xmax - Xmin;
    Ymin = 0.0;
    Ymax = n;

    /* Mike's alterations */
    Xmin = 0.0;
    Xmax = (int) Xmax + 1;
    Xrange = Xmax - Xmin;
    Theight = 15*n;
    /* round Xmax up */
    Yrange = (double) Ymax - Ymin;

    /* write out the preamble */

    fprintf(fout,"%%!\n");
    fprintf(fout,"/Times-Roman findfont 12 scalefont setfont\n");

    /* get the X position for id codes then write them out*/
    if(sim == 1){
      Xid = Xmax + (Xrange *IdOffsetFactor);
    }else{
      Xid = Xmin - (Xrange *IdOffsetFactor);
    }
    Xid = trans_x(Xid,sim,Xmin,Xmax,Xrange,Twidth,Xoffset);
    
    for(i=0;i<(n+1);++i){
      Yid = trans_y((double) i,Ymin,Ymax,Yrange,Theight,Yoffset);
      PSPutText((float) Xid,(float) Yid,idents[clust[n-1].members[i]],fout);
    }

    /* write min and max scores below the plot */
    x1 = Xmin;
    x2 = Xmax;
    y1 = Ymin - (Yrange * NumberOffsetFactor);
    x1 = trans_x(x1,sim,Xmin,Xmax,Xrange,Twidth,Xoffset);
    x2 = trans_x(x2,sim,Xmin,Xmax,Xrange,Twidth,Xoffset);
    y1 = trans_y(y1,Ymin,Ymax,Yrange,Theight,Yoffset);

    /* draw line at bottom and mark it*/

    fprintf(fout,"%f %f moveto\n",x1,y1);
    fprintf(fout,"%f %f lineto\n",x2,y1);
    for (i=0;i<=Xmax;i++) {
      fprintf(fout,"%f %f moveto\n",x1+i*((x2-x1)/Xmax),y1);
      fprintf(fout,"%f %f lineto\n",x1+i*((x2-x1)/Xmax),y1-5);
      } 
    for (x3=0;x3<=Xmax;x3=x3+0.1) {
      fprintf(fout,"%f %f moveto\n",x1+x3*((x2-x1)/Xmax),y1);
      fprintf(fout,"%f %f lineto\n",x1+x3*((x2-x1)/Xmax),y1-2);
      } 

    /* Mike's offset */
    y1=y1-15;
    x1=x1-2;
    x2=x2-2;

    sprintf(buff,"%g",Xmin);
    PSPutText((float) x1,(float) y1,buff,fout);
    sprintf(buff,"%g",Xmax);
    PSPutText((float) x2,(float) y1,buff,fout);

    for(i=0;i<n;++i){
        x1 = clust[i].parentA->score;
        x2 = clust[i].score;
        x3 = clust[i].parentB->score;
	if(x1 == base_val && sim == 1) x1 = Xmax;
	if(x1 == base_val && sim == 2) x1 = Xmin;
	if(x3 == base_val && sim == 1) x3 = Xmax;
	if(x3 == base_val && sim == 2) x3 = Xmin;
        y1 = get_mean(clust[i].parentA[0].members,inx,clust[i].parentA[0].n);
        y2 = get_mean(clust[i].parentB[0].members,inx,clust[i].parentB[0].n);
        /* transform x and y to fit on the page, then */
	x1 = trans_x(x1,sim,Xmin,Xmax,Xrange,Twidth,Xoffset);
	x2 = trans_x(x2,sim,Xmin,Xmax,Xrange,Twidth,Xoffset);
	x3 = trans_x(x3,sim,Xmin,Xmax,Xrange,Twidth,Xoffset);
        y1 = trans_y(y1,Ymin,Ymax,Yrange,Theight,Yoffset);
        y2 = trans_y(y2,Ymin,Ymax,Yrange,Theight,Yoffset);

        draw_arch(fout,(float)x1,(float)x2,(float)x3,(float)y1,(float)y2);
    }
    fprintf(fout,"showpage\n");
}

double get_mean(int *arr,int *inx,int n)
{
    double sum;
    int i;
    sum = 0.0;
    for(i=0;i<n;++i){
        sum += (double) inx[arr[i]];
    }
    return sum/n;
}

void draw_arch(FILE *fout,
               float x1,
               float x2,
               float x3,
               float y1,
               float y2) 
{
    fprintf(fout,"%f %f moveto\n",x1,y1);
    fprintf(fout,"%f %f lineto\n",x2,y1);
    fprintf(fout,"%f %f lineto\n",x2,y2);
    fprintf(fout,"%f %f lineto\n",x3,y2);
    fprintf(fout,"stroke newpath\n");
}

double trans_x(double X,
	       int sim,
	       double Xmin,
	       double Xmax,
	       double Xrange,
	       float Twidth,
	       float Xoffset)
{
  if(sim ==1){
    return (((X - Xmin)/Xrange) * (double) Twidth) + Xoffset;
  }else{
    return (((Xmax - X)/Xrange) * (double) Twidth) + Xoffset;
  }
}
double trans_y(double X,
	       double Xmin,
	       double Xmax,
	       double Xrange,
	       float Twidth,
	       float Xoffset)
{
  return (((X - Xmin)/Xrange) * (double) Twidth) + Xoffset;
}

void PSPutText(float x,float y,char *text,FILE *outf)

{
  fprintf(outf," %f %f moveto (%s) show \n",x,y,text);
}

/**********************************************************************
mark_parents:  This works back up the tree setting the entity->lab
values to 0 for all parents.
Hopefully the recursion won't run out of space for the size of problem
we typically deal with...
----------------------------------------------------------------------*/

void mark_parents(struct sc *entity)

{
  entity->lab = 0;
  if(entity->parentA != NULL){
    mark_parents(entity->parentA);
  }
  if(entity->parentB != NULL){
    mark_parents(entity->parentB);
  }
}

/*********************************************************************
write_unclustered:

Work back down the tree and print out the identifiers of entitites that
are not clustered into any group.  If mark_parents has not been called,
then all entities will be output.  Otherwise, only those that have
not been marked will be output.
--------------------------------------------------------------------*/

void write_unclustered(struct sc *entity,char **idents,FILE *fout)

{
  if(entity->lab == 1){
    /* this could be an unclustered entity or a cluster 
       containing unclustered entities  */
    if(entity->parentA == NULL && entity->parentB == NULL){
      fprintf(fout,"%s\n",idents[entity->members[0]]);
      return;
    }
    /* we don't need both if's since any entitity with one NULL parent 
       must also have the other parent NULL
    */
    if(entity->parentA != NULL){
      write_unclustered(entity->parentA, idents,fout);
    }
    if(entity->parentB != NULL){
      write_unclustered(entity->parentB, idents,fout);
    }
  }
}


