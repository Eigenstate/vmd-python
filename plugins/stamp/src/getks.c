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
#include <stdlib.h>

#include <stamp.h>

/* This routine looks for a DSSP file, if found it reads in the DSSP summary
 *  and stores it in domain[i].sec for future use */

int getks(struct domain_loc *domain, int ndomain, struct parameters *parms) {

	FILE *DSSP;
	int i,j/*,k*/;
	int /*include,*/count;
	int total,add;
	int retval;
/*	char c,chain; */
	char *filename;
/*	char *label; */
	char *empty=0;
	
	count=0;
	retval=0;


	for(i=0; i<ndomain; ++i) {
	   fprintf(parms[0].LOG,"%s -- ",domain[i].id);
	   /* first check to see if there is a DSSP file that uses the whole ID name */
           filename=getfile(domain[i].id,parms[0].dsspfile,strlen(domain[i].id),parms[0].LOG);
	   /* if there is, use it, otherwise try to get one using the four letter code */
	   if(filename[0]=='\0') {
		free(filename);
		filename=getfile(domain[i].id,parms[0].dsspfile,4,parms[0].LOG);
	   }
	   if(filename[0]=='\0') {
	      fprintf(parms[0].LOG," no DSSP file found for %s\n",domain[i].id);
	      for(j=0; j<domain[i].ncoords; ++j) domain[i].sec[j]='?';
	      domain[i].sec[j]='\0';
	      retval=-1; /* if any of the sequences have missing secondary structures */
	   } else {
	      DSSP=openfile(filename,"r");
	      total=0;
	      fprintf(parms[0].LOG," using file %s\n",filename);
	      for(j=0; j<domain[i].nobj; ++j) {
	         if(get_dssp_sum(DSSP,domain[i].start[j],domain[i].end[j],
		    domain[i].type[j],&domain[i].sec[total],domain[i].reverse[j],
		    (parms[0].MAX_SEQ_LEN-total),&add,parms[0].LOG)==-1) 
		    retval=-1;
	         total+=add;
		 closefile(DSSP,filename);
	         DSSP=openfile(filename,"r");
	      }
	      closefile(DSSP,filename);
	      free(filename);
	      if(total!=domain[i].ncoords) {
		 fprintf(parms[0].LOG,"warning: DSSP summary found was incomplete -- the results may have errors\n");
		 for(j=total; j<domain[i].ncoords; ++j) domain[i].sec[j]='?';
		 domain[i].sec[j]='\0';
	      }
	      display_align(&domain[i].aa,1,&domain[i].sec,1,&domain[i].aa,&domain[i].sec,empty,empty,parms[0].COLUMNS,0,0,parms[0].LOG);
	   }
	}
	return retval;
	      
} 
	  


