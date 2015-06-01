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
#include <math.h>

#include <stamp.h>

/* roughfit: this writes a bloc file which consists of the sequences aligned from
 *   their N-terminal ends, and uses this to run 'ampsfit'. It then reads in the
 *   obtained transformations to proceed with STAMP.  This avoids having to
 *   run AMPS intially.  This will not always work, especially if the sequences
 *   are of significantly different lengths. 
 *
 * modification: now doesn't run ampsfit, just aligns all sequences to the
 *  first one -- avoids wonky system calls within programs */

int roughfit(struct domain_loc *domain, int ndomain, struct parameters *parms) {

	int i,j,k/*,l,m*/;
/*	int counter,ndone; */
	int ntofit;
	
	float rmsd;

/*	char sys[200];  */

	struct brookn tmps,tmpe;

	FILE *ROUGHOUT=0;
	

	printf("Running roughfit.\n");

	fprintf(parms[0].LOG,"\n\nROUGH FIT has been requested.\n");
	fprintf(parms[0].LOG,"  The sequences will be aligned from their N-terminal ends, and\n");
	fprintf(parms[0].LOG,"  the resulting equivalences will be used to generate an inital\n");
	fprintf(parms[0].LOG,"  superposition.\n");

	if(parms[0].roughout == 1 ) { /* output the transformations */
            fprintf(parms[0].LOG,"\nROUGH FIT transformations will be output to the file %s\n",parms[0].roughoutfile);
	    if((ROUGHOUT=fopen(parms[0].roughoutfile,"w"))==NULL) {
		fprintf(stderr,"Error opening file %s for writing\n",parms[0].roughoutfile);
	        exit(-1);
	    }
	    fprintf(ROUGHOUT,"%% Output from STAMP ROUGH FIT routine\n");
	    fprintf(ROUGHOUT,"%%  The sequences from the file %s have been aligned from their\n",parms[0].listfile);
	    fprintf(ROUGHOUT,"%%  N-terminal ends, andthe resulting equivalences were be used \n");
	    fprintf(ROUGHOUT,"%% to generate the superpositions given below.\n");
	}

        /*  We will fit all domains onto the first domain */
	for(i=1; i<ndomain; ++i) {
	   if(domain[i].ncoords>domain[0].ncoords) ntofit=domain[0].ncoords;
	   else ntofit=domain[i].ncoords;
           rmsd=matfit(domain[0].coords,domain[i].coords,domain[i].R,domain[i].V,ntofit,1,parms[0].PRECISION); 
	   fprintf(parms[0].LOG,"Domains %s onto %s RMS of %f on %d atoms\n",domain[0].id,domain[i].id,rmsd,ntofit); 
	   if(parms[0].roughout == 1) {
		fprintf(ROUGHOUT,"%% Domains %s onto %s RMS of %f on %d atoms\n",domain[0].id,domain[i].id,rmsd,ntofit); 
	   }
        }

	for(i=0; i<ndomain; ++i) {
	  fprintf(parms[0].LOG,"\nDomain %2d, %s, %d coordinates\n",i+1,domain[i].id,domain[i].ncoords);
	  fprintf(parms[0].LOG,"Applying the transformation...\n");
	  for(j=0; j<3; ++j) {
	      fprintf(parms[0].LOG,"| ");
	      for(k=0; k<3; ++k) fprintf(parms[0].LOG,"%8.5f ",domain[i].R[j][k]);
	      fprintf(parms[0].LOG," |    %8.5f\n",domain[i].V[j]);
	  }
	  fprintf(parms[0].LOG,"      ...to these coordinates.\n");
	  if(parms[0].roughout == 1) {
		fprintf(ROUGHOUT,"%s %s { ",domain[i].filename,domain[i].id);
		for(j=0; j<domain[i].nobj; ++j) {

		    if(domain[i].start[j].cid!=' ') tmps.cid=domain[i].start[j].cid;
		    else tmps.cid='_';
		    if(domain[i].end[j].cid!=' ') tmpe.cid=domain[i].start[j].cid;
                    else tmpe.cid='_';
		    if(domain[i].start[j].in!=' ') tmps.in=domain[i].start[j].in;
                    else tmps.in='_';
		    if(domain[i].end[j].in!=' ') tmpe.in=domain[i].start[j].in;
                    else tmpe.in='_';

		    if(domain[i].type[j]==1) fprintf(ROUGHOUT,"ALL");
		    else if(domain[i].type[j]==2) fprintf(ROUGHOUT,"CHAIN %c",domain[i].start[j].cid);
		    else fprintf(ROUGHOUT,"%c %d %c to %c %d %c",
			tmps.cid,domain[i].start[j].n,tmps.in,
			tmps.cid,domain[i].end[j].n,tmpe.in);
		    fprintf(ROUGHOUT," ");
		}
	        fprintf(ROUGHOUT,"\n");
		for(j=0; j<3; ++j) {
		   fprintf(ROUGHOUT,"%10.4f %10.4f %10.4f   %10.4f ",
			domain[i].R[j][0],domain[i].R[j][1],domain[i].R[j][2],domain[i].V[j]);
		   if(j==2) fprintf(ROUGHOUT," } ");
		   fprintf(ROUGHOUT,"\n");
		}
	  }
	  matmult(domain[i].R,domain[i].V,domain[i].coords,domain[i].ncoords,parms[0].PRECISION);
	 
	}

	/* and we are done */
	if(parms[0].roughout == 1) fclose(ROUGHOUT);
	fprintf(parms[0].LOG,"\n");

	return 0;
}
