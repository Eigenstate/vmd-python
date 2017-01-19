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

/* Scan's a database of domain descriptors using the following protocol:
 *  
 * 1) perform an inital superposition using the N-terminal region of the sequences
 * 2) pairfit(domain,ddomain,... 
 * 3) if the database domain is significantly longer than the search domain slide along 
 *    the length of the database sequence and repeat steps 1) and 2)
 * 4) report the score and transformation to an output file
 *
 *
 * Modification: 15 November 1993 SLOW_SCAN
 * Change in how initial superimpositions are performed.  See back of chapter 3 of my
 *  thesis for more details.  Briefly, rather than overlaying the query onto the database,
 *  *both* sequences are passed over each other to allow for many more superimpositions.
 *  It is hoped that this procedure will make the scans less one sided, enabling 
 *  similarities between query and database to be found regardless of which structure i
 *  query and which is database.  The structures used to refine the fit will consist 
 *  of *all* of the query and a fraction of the database (if it is significantly
 *  larger than the query structure).
 *  
 *    e.g.,
 *
 *    fit 1:
 *      query     --------------------------------------
 *      database                                   ---------------------------------
 *
 *    fit 2:
 *      query     ---------------------------------------
 *      database                     -------------------------------
 *
 *    fit 3:   
 *      query     ---------------------------------------
 *      database        ---------------------------------
 *  
 *    fit 4:
 *      query               --------------------------------------
 *      database  -----------------------------
 *
 *    fit 5:
 *      query                          -------------------------------------
 *      database   ---------------------------
 *
 *    To do this only one parameter is require.  A pointer that tells the
 *     program where to align the N-terminus of the query on the database
 *     sequence.  It will start at Ld - Lq + SCANSLIDE and end at 
 *     Ld - SCANSLIDE (Ld = database sequence length, Lq = query sequence 
 *     length, SCANSLIDE = slide parameter).  The start value will most 
 *     certainly be negative.  It is thus desireable to find the overlay of 
 *     residues.
 *
 *    To find the best similarities, it is probably best only to fit the
 *     residues around those use in the initial superimposition.
 *     Also, to keep the scans from being one-sided, the number of
 *     residues used in the fit should be, perhaps, a function of
 *     the *minimum* length of the two structures (rather than
 *     simply the length of the query structure).
 *
 *   I have worked out the problems of doing this.  
 *    The program in "overlay.c" gives the details.
 *
 * Modification 25 November 1993
 *  The scans are now completely two sided, but there is another
 *   problem.  The nature of the matrix passed to 
 *   dynamic programming means that there is a great potential for
 *   horizontal and vertical moves.
 *
 */

int slow_scan(struct domain_loc qdomain, struct parameters *parms) {

   char qendsec=0,qendaa=0;
   char dendsec=0,dendaa=0;

   int i,j,k/*,l*/;
   int n,m;
   int total,add,error;
   int end/*,len_d*/;
   int dstart,dend;
   int qstart,qend;
   int length;
   int first;
/*   int best_score_start; */
   int dcount,count,domain_count,ndomain;
   int nfit,ntrans=0,ntries=0;
   int best_nfit,best_len;
   int best_nsec;
   int nequiv;
   int best_nequiv;
/*   int a,b,c; */
   int gotsec,secskip;
   int qpera,qperb,qperc;
   int dpera,dperb,dperc;
/*   int savetype,savenobj; */
   int Q1overlay,Q2overlay,D1overlay,D2overlay;
   int Q1fit,Q2fit,D1fit,D2fit;
   int P,N,M;
   int Min_Size;
   int nsec;

   int **atoms1,**atoms2;
   int **hbcmat,**best_hbcmat;

   float score,rms,irms;
   float best_score,best_rms;
   float best_seqid,best_secid;
   float t_sec_diff,ratio;
   float test;
   float seqid,secid;
   float *vtemp;
   float **rtemp;

   double Pm;

   FILE *IN,*PDB,*TRANS;

   struct domain_loc ddomain,ddomainall,qdomainall;
/*   struct brookn savebegin,saveend; */

   ddomain.coords=(int**)malloc(parms[0].MAX_SEQ_LEN*sizeof(int*));
   ddomainall.coords=ddomain.coords;
   ddomain.aa=(char*)malloc(parms[0].MAX_SEQ_LEN*sizeof(char));
   ddomainall.aa=ddomain.aa;
   ddomain.sec=(char*)malloc(parms[0].MAX_SEQ_LEN*sizeof(char));
   ddomainall.sec=ddomain.sec;
   ddomain.numb=(struct brookn*)malloc(parms[0].MAX_SEQ_LEN*sizeof(struct brookn));
   ddomainall.numb=ddomain.numb;

   /* qdomain is defined in the above routines */
   qdomainall.coords=qdomain.coords;
   qdomainall.aa=qdomain.aa;
   qdomainall.sec=qdomain.sec;
   qdomainall.numb=qdomain.numb;
   qdomainall.ncoords=qdomain.ncoords;

   atoms1=(int**)malloc(parms[0].MAX_SEQ_LEN*sizeof(int*));
   atoms2=(int**)malloc(parms[0].MAX_SEQ_LEN*sizeof(int*));
   rtemp=(float**)malloc(3*sizeof(float*));
   for(i=0; i<3; ++i) 
      rtemp[i]=(float*)malloc(3*sizeof(float));
   vtemp=(float*)malloc(3*sizeof(float));
   secskip=0;

   hbcmat=(int**)malloc(3*sizeof(int*));
   best_hbcmat=(int**)malloc(3*sizeof(int*));
   for(i=0; i<3; ++i) {
     hbcmat[i]=(int*)malloc(3*sizeof(int));
     best_hbcmat[i]=(int*)malloc(3*sizeof(int));
   }

   /* open database file */
   if((IN=fopen(parms[0].database,"r"))==NULL) {
      fprintf(stderr,"error opening file %s\n",parms[0].database);
      return -1;
   }

   /* scanning output is to be output to a file called parms[0].scanfile */
   if((TRANS=fopen(parms[0].scanfile,"w"))==NULL) {
	fprintf(stderr,"error opening file %s\n",parms[0].scanfile);
	return -1;
   }

   /* The query domain is output so that the program superimpose can 
    *  be used to produces files for all the domains to be aligned */
   fprintf(TRANS,"%% Output from STAMP scanning routine\n%%\n");
   fprintf(TRANS,"%% Domain %s was used to scan the domain database:\n",qdomain.id);
   fprintf(TRANS,"%%  %s\n",parms[0].database);
   fprintf(TRANS,"%% %d fits were performed\n",parms[0].NPASS);
   fprintf(TRANS,"%% Fit 1 E1=%7.3f, E2=%7.3f, CUT=%7.3f\n",
	parms[0].first_E1,parms[0].first_E2,parms[0].first_CUTOFF);
   if(parms[0].NPASS==2) 
      fprintf(TRANS,"%% Fit 2 E1=%7.3f, E2=%7.3f, CUT=%7.3f\n",
          parms[0].second_E1,parms[0].second_E2,parms[0].second_CUTOFF);
   fprintf(TRANS,"%% Approximate fits (alignment from N-termini) were performed\n");
   fprintf(TRANS,"%%   at every %d residue of the database sequences \n",parms[0].SCANSLIDE);
   fprintf(TRANS,"%% Transformations were output for Sc= %6.3f\n",parms[0].SCANCUT);
   fprintf(TRANS,"%% \n");
   fprintf(TRANS,"%% Domain used to scan \n");
   fprintf(TRANS,"# Sc= 10.000 RMS=  0.01  Len= 999 nfit= 999 Seqid= 100.00 Secid= 100.00 q_len= %4d d_len= %4d n_sec= 100 n_equiv= 999 fit_pos= _  0 _ \n",
        qdomain.ncoords,ddomain.ncoords);
   printdomain(TRANS,qdomain,0);
      
   end=0;
   if(parms[0].SECSCREEN && parms[0].SECTYPE!=0) { /* calculate qpera,qperb,perc if needed */
      sec_content(qdomain.sec,qdomain.ncoords,parms[0].SECTYPE,&qpera,&qperb,&qperc);
      fprintf(parms[0].LOG,"domain %s, percent helix: %3d, sheet: %3d (coil: %3d)\n",qdomain.id,qpera,qperb,qperc);
      if(qperc==100) parms[0].SECSCREEN=0;  /* don't screen if all coil */
   }
   if(parms[0].MINFIT==-1) parms[0].MINFIT=qdomain.ncoords/2;
   first=0;
   /* find the number of domains in the file */
   ndomain=count_domain(IN); rewind(IN);
   fprintf(parms[0].LOG,"A total of %d comparisons are to be performed\n",ndomain);
   domain_count=0;
   while(!end && domain_count<ndomain) {
      best_score=0.0; best_rms=100.0; best_len=0; best_nfit=0; seqid=0.0; secid=0.0;
      best_nequiv = 0; best_nsec = 0; best_secid=0.0; best_seqid=0.0;
      for(j=0; j<3; ++j) for(k=0; k<3; ++k) hbcmat[j][k]=best_hbcmat[j][k]=0;
      secskip=0;
      error=0;
      end=domdefine(&ddomain,&i,parms[0].stampdir,parms[0].DSSP,IN,parms[0].LOG);
      if(end) break;
      if(end==-1) error=1; /* read in domain next domain descriptor from database */
      if(!error) {
         if((PDB=openfile(ddomain.filename,"r"))==NULL) {
	    fprintf(stderr,"error opening file %s \n",ddomain.filename);
	    error=1;
	 }
       }
      if(!error) {
	total=0;
	fprintf(parms[0].LOG,"\n\nComparison %3d with domain %s ",domain_count+1,ddomain.id);
	for(i=0; i<ddomain.nobj; ++i) {
	 if(!parms[0].DSSP) {
          if(igetca(PDB,&ddomain.coords[total],&ddomain.aa[total],&ddomain.numb[total],
	     &add,ddomain.start[i],ddomain.end[i],ddomain.type[i],(parms[0].MAX_SEQ_LEN-total),
	     ddomain.reverse[i],parms[0].PRECISION,parms[0].ATOMTYPE,parms[0].LOG)==-1) error=1;
	 } else {
	  if(igetcadssp(PDB,&ddomain.coords[total],&ddomain.aa[total],&ddomain.numb[total],
	     &add,ddomain.start[i],ddomain.end[i],ddomain.type[i],(parms[0].MAX_SEQ_LEN-total),
	     ddomain.reverse[i],parms[0].PRECISION,parms[0].LOG)==-1) error=1;
	  }
	  switch(ddomain.type[i]) {
	     case 1: fprintf(parms[0].LOG," all residues"); break;
	     case 2: fprintf(parms[0].LOG," chain %c",ddomain.start[i].cid); break;
	     case 3: fprintf(parms[0].LOG," from %c %4d %c to %c %4d %c",
			ddomain.start[i].cid,ddomain.start[i].n,ddomain.start[i].in,
	 	        ddomain.end[i].cid,ddomain.end[i].n,ddomain.end[i].in); break;
           }
	   fprintf(parms[0].LOG,"%4d CAs ",add); total+=add;
	   closefile(PDB,ddomain.filename); PDB=openfile(ddomain.filename,"r");
	}
	fprintf(parms[0].LOG,"\n");
	ddomain.ncoords=total;
	ddomainall.ncoords=total;
	closefile(PDB,ddomain.filename);
        /* skip if the database sequence is too short */
        if(ddomain.ncoords<(int)(parms[0].MIN_FRAC*(float)qdomain.ncoords)) {
  	   fprintf(parms[0].LOG,"MIN_FRAC * query length = %d\n",(int)(parms[0].MIN_FRAC*(float)qdomain.ncoords));
	   fprintf(parms[0].LOG,"Database structure (length = %d) is too small -- skipping this structure\n",
		ddomain.ncoords);
	   error=1;
        }
      }
      if(!error) { /* read in secondary structure if required */
	 switch(parms[0].SCANSEC) {
	    case 0: {
	      fprintf(parms[0].LOG,"No secondary structure assignment will be considered\n");
	        for(j=0; j<ddomain.ncoords; ++j) ddomain.sec[j]='?';
	        ddomain.sec[j]='\0';
		gotsec=0;
	    } break;
	    case 1: {
	      fprintf(parms[0].LOG,"Will try to find Kabsch and Sander DSSP assignments\n"); 
	      if(getks(&ddomain,1,parms)==0) gotsec=1;
	      else gotsec=0;
	      if(strlen(ddomain.sec)==0) gotsec=0;
	      }
	      break;
	    case 2: {
	      fprintf(parms[0].LOG,"Reading in secondary structure assignments from file: %s\n",parms[0].secfile);
	      if(getsec(&ddomain,1,parms)==0) gotsec=1;
	      else gotsec=0;
	      if(strlen(ddomain.sec)==0) gotsec=0;
	      }
	    default: { return -1; } 
	}

/*       if(parms[0].SCANTRUNC) {
	test=(float)ddomainall.ncoords/(float)qdomain.ncoords;
	if(test>parms[0].SCANTRUNCFACTOR) 
	     fprintf(parms[0].LOG,"Database sequence is very large relative to the query -- it will be truncated.\n");
	if(test<(1/parms[0].SCANTRUNCFACTOR))
	    fprintf(parms[0].LOG,"Query sequence is very large relative to the database -- it will be truncated.\n");
       }
*/
       fprintf(parms[0].LOG,"Sequences will be slid along each other to obtain superimpositions\n");
       fprintf(parms[0].LOG," (see manual for an explanation)\n");
       
       /* got coordinates, with no errors, now perform the inital fit */
       dcount=0; 
       ntrans=0; ntries=0;
       P=0-qdomain.ncoords+parms[0].SCANSLIDE;
       N=qdomainall.ncoords;
       M=ddomainall.ncoords;
       if(N>=M) Min_Size=(int)(parms[0].SCANTRUNCFACTOR*(float)M);
       else     Min_Size=(int)(parms[0].SCANTRUNCFACTOR*(float)N);

       /* perform initial fits and comparisons until the two sequences have been
	*  slid over each other */
       while(P<M) { 
	 /* If parms[0].SCANTRUNC is set to 1, then only a fraction of the query or database
	  *  sequence will be considered (depending on which one is longer).
	  *  The length of the fraction will be equal to the length of the shorter sequence
	  *  sequence times a factor (parms[0].SCANTRUNCFACTOR) For example, if SCANTRUNCFACTOR is 1.30, 
	  *  the the longer sequence will be no longer than 1.3 times the length of the shorter).
	  * The current database sequence is stored in ddomain (query sequence in qdomain),
	  *  whereas the total structure details are stored in ddomainall (qdomainall),
	  *  and copied back after each comparison.  
	  * This process *  speeds things up DRASTICALLY, since without it, one often compares a sequence
	  *  of, say, a hundred amino acids to things that are four hundred amino acids long, which
	  *  is just silly for most applications.  In other words, if you are scanning with a domain
	  *  you do not, generally, want some sort of global, overall comparison with something which
	  *  consists of many domains.  */

	 /* determining the atoms with which to derive the initial fit */
	 D1overlay=max(0,P);
	 D2overlay=leq(M,P+N);
	 D2overlay-=1;
	 Q1overlay=max(0,-1*P);
	 Q2overlay=leq(N,M-P); 
	 Q2overlay-=1;
	 test=(float)(D2overlay-D1overlay+1)/(float)(Q2overlay-Q1overlay+1);

/* 	 fprintf(parms[0].LOG,"query:   ");
	 for(k=0; k<psd; ++k) if(k%10==0) fprintf(parms[0].LOG," ");
	 for(k=0; k<qdomain.ncoords; ++k) if(k%10==0) fprintf(parms[0].LOG,"*");
	 fprintf(parms[0].LOG,"\n");
	 fprintf(parms[0].LOG,"database:");
	 for(k=0; k<psq; ++k) if(k%10==0) fprintf(parms[0].LOG," ");
	 for(k=0; k<ddomainall.ncoords; ++k) if(k%10==0) fprintf(parms[0].LOG,"*");
	 fprintf(parms[0].LOG,"\n");  
*/
	 if(parms[0].SCANTRUNC) { 
	   if((D2overlay-D1overlay)<Min_Size) { /* Database sequence too short */
	     add=(int)((float)(Min_Size-(D2overlay-D1overlay))/2);
	     /* bit to be added on both ends */
	     D1fit=D1overlay-add; D2fit=D2overlay;
	     if(D1fit<0) {
		D2fit+=(0-D1fit);
		D1fit=0;
	     }
	     D2fit+=add; 
	     if(D2fit>(M-1)) {
		D1fit-=(D2fit-M+1);
		D2fit=ddomainall.ncoords-1;
	     }
	     if(D1fit<0) D1fit=0;
	     ddomain.coords=&ddomainall.coords[D1fit];
	     ddomain.sec=&ddomainall.sec[D1fit];
	     ddomain.aa=&ddomainall.aa[D1fit];
	     ddomain.numb=&ddomainall.numb[D1fit];
	     ddomain.ncoords=(D2fit-D1fit+1);
	     dendsec = ddomain.sec[ddomain.ncoords];
	     dendaa  = ddomain.aa[ddomain.ncoords];
	     ddomain.sec[ddomain.ncoords]=ddomain.aa[ddomain.ncoords]='\0'; 
	     fprintf(parms[0].LOG,"For this fit, only %s residues %c %d%c to %c %d%c will be used\n",
		     ddomain.id,ddomain.numb[D1fit].cid,ddomain.numb[D1fit].n,ddomain.numb[D1fit].in,
		     ddomain.numb[D2fit-1].cid,ddomain.numb[D2fit-1].n,ddomain.numb[D2fit-1].in);
	   }
	   if((Q2overlay-Q1overlay)<Min_Size) { /* Query too short */
	     add=(int)((float)(Min_Size-(Q2overlay-Q1overlay))/2);
	     /* bit to be added to both ends */
	     Q1fit=Q1overlay-add; Q2fit=Q2overlay;
	     if(Q1fit<0) { 
	       Q2fit+=(0-Q1fit);
	       Q1fit=0;
	     }
	     Q2fit+=add;
	     if(Q2fit>(qdomainall.ncoords-1)) {
	       Q1fit-=(Q2fit-N);
	       Q2fit=qdomainall.ncoords-1;
	     }
	     if(Q1fit<0) Q1fit=0;
	     qdomain.coords=&qdomainall.coords[Q1fit];
	     qdomain.sec=&qdomainall.sec[Q1fit];
	     qdomain.aa=&qdomainall.aa[Q1fit];
	     qdomain.ncoords=(Q2fit-Q1fit+1);
	     qendsec = qdomain.sec[qdomain.ncoords];
	     qendaa  = qdomain.aa[qdomain.ncoords];
	     qdomain.aa[qdomain.ncoords]=qdomain.aa[qdomain.ncoords]='\0';
	     fprintf(parms[0].LOG,"For this fit, only %s residues %c %d%c to %c %d%c will be used\n",
		  qdomain.id,qdomain.numb[Q1fit].cid,qdomain.numb[Q1fit].n,qdomain.numb[Q1fit].in,
		  qdomain.numb[Q2fit-1].cid,qdomain.numb[Q2fit-1].n,qdomain.numb[Q2fit-1].in);
	    }
	 } else {
	     if(secskip) break; /* if we have already ruled out this string (ie.
				    *  secskip was set to one during the last value of dcount,
				    *  though we are not truncating, so we will use the same
				    *  atoms as as last time -- there is no point repeating */
	     Q1fit=D1fit=0;
	     Q2fit=N-1; Q2fit=N-1;
	}
 	 if(parms[0].SECSCREEN && gotsec) { /* if specified, screen based on secondary structure content */
	    sec_content(ddomain.sec,ddomain.ncoords,parms[0].SCANSEC,&dpera,&dperb,&dperc);
	    sec_content(qdomain.sec,qdomain.ncoords,parms[0].SCANSEC,&qpera,&qperb,&qperc);
	    fprintf(parms[0].LOG,"domain %s, percent helix: %3d, sheet: %3d (coil: %3d)\n",ddomain.id,dpera,dperb,dperc);
/*	    t_sec_diff=sqrt((float)((dpera-qpera)*(dpera-qpera)+(dperb-qperb)*(dperb-qperb)+(dperc-qperc)*(dperc-qperc))); */
	    t_sec_diff=sqrt((float)((dpera-qpera)*(dpera-qpera)+(dperb-qperb)*(dperb-qperb)));
	    fprintf(parms[0].LOG,"Secondary structure (helix & sheet) distance: %6.2f %%\n",t_sec_diff);
	    if(t_sec_diff>parms[0].SECSCREENMAX) {
	      fprintf(parms[0].LOG," Secondary structure assignments are very different -- skipping this fit.\n");
	      secskip=1;
	    } else secskip=0;
        } else {
	   t_sec_diff=100.0;
	}
        if(!secskip) {
	 fprintf(parms[0].LOG,"Aligning %s residues %c %4d %c -- %c %4d %c with ",
		qdomain.id,qdomain.numb[Q1overlay].cid,qdomain.numb[Q1overlay].n,qdomain.numb[Q1overlay].in,
		qdomain.numb[Q2overlay].cid,qdomain.numb[Q2overlay].n,qdomain.numb[Q2overlay].in);
	 fprintf(parms[0].LOG,"  %s residues %c %4d %c --  %c %4d %c for the initial fit\n",
		ddomain.id,ddomainall.numb[D1overlay].cid,ddomainall.numb[D1overlay].n,ddomainall.numb[D1overlay].in,
		ddomainall.numb[D2overlay].cid,ddomainall.numb[D2overlay].n,ddomainall.numb[D2overlay].in);
	 /* now perform an initial fit based on the atoms that are aligned */
	 count=0; 
	 for(i=0; i<(Q2overlay-Q1overlay+1); ++i) {
	   atoms1[count]=qdomainall.coords[Q1overlay+i];
	   atoms2[count]=ddomainall.coords[D1overlay+i];
	   count++;
	 }
	 /* fit the two sets of atoms */
	 irms=matfit(atoms1,atoms2,ddomain.R,ddomain.V,count,1,parms[0].PRECISION);

	 fprintf(parms[0].LOG,"Number of atoms used in the initial fit: %4d\n",count);
/*	 printmat(ddomain.R,ddomain.V,3,parms[0].LOG); */

	 /* Now apply the initial translation to the query atom seq */
	 fprintf(parms[0].LOG,"Performing initial fit\n");
	 matmult(ddomain.R,ddomain.V,ddomain.coords,ddomain.ncoords,parms[0].PRECISION);

	 for(i=0; i<3; ++i) {
	    for(j=0; j<3; ++j) 
	       if(i==j) ddomain.r[i][j]=1.0;
	       else ddomain.r[i][j]=0.0;
	    ddomain.v[i]=0.0;
	 }

	 if(parms[0].NPASS==2) {
	   parms[0].const1=-2*parms[0].first_E1*parms[0].first_E1;
	   parms[0].const2=-2*parms[0].first_E2*parms[0].first_E2;
	   parms[0].PAIRPEN=parms[0].first_PAIRPEN;
	   parms[0].CUTOFF=parms[0].first_CUTOFF;
	   parms[0].BOOLCUT=parms[0].first_BOOLCUT;
	   fprintf(parms[0].LOG,"First fit\n");
/*	   if(parms[0].BOOLEAN) 
	     fprintf(parms[0].LOG,"First fit: BOOLCUT = %5.3f\n",parms[0].first_BOOLCUT);
	   else 
	     fprintf(parms[0].LOG,"First fit: E1 = %5.2f, E2 = %5.2f, CUT = %5.2f, PEN = %5.2f\n",
		   parms[0].first_E1,parms[0].first_E2,parms[0].first_CUTOFF,parms[0].first_PAIRPEN);  */
	   if(pairfit(&qdomain,&ddomain,&score,&rms,&length,&nfit,parms,
	      0,&qstart,&qend,&dstart,&dend,&seqid,&secid,&nequiv,&nsec,hbcmat,0,-1,0)==-1) 
	      fprintf(stderr,"error in PAIRFIT\n");
	   for(i=0; i<3; ++i) {
	      for(j=0; j<3; ++j) 
		 rtemp[i][j]=ddomain.R[i][j];
	      vtemp[i]=ddomain.V[i];
	   }
	   update(ddomain.r,rtemp,ddomain.v,vtemp);
/*	   printmat(rtemp,vtemp,3,parms[0].LOG); */
	   fprintf(parms[0].LOG,"Second fit\n");
	 } /* else fprintf(parms[0].LOG,"Fitting with: ");  */
	 if((score>=parms[0].first_THRESH) || parms[0].NPASS==1) {
/*	   if(parms[0].BOOLEAN)
	     fprintf(parms[0].LOG,"BOOLCUT = %5.3f\n",parms[0].second_BOOLCUT);
	   else
	     fprintf(parms[0].LOG,"E1 = %5.2f, E2 = %5.2f, CUT = %5.2f, PEN = %5.2f\n",
	        parms[0].second_E1,parms[0].second_E2,parms[0].second_CUTOFF,parms[0].second_PAIRPEN); */
	   parms[0].const1=-2*parms[0].second_E1*parms[0].second_E1;
	   parms[0].const2=-2*parms[0].second_E2*parms[0].second_E2;
	   parms[0].PAIRPEN=parms[0].first_PAIRPEN;
	   parms[0].CUTOFF=parms[0].second_CUTOFF;
	   parms[0].BOOLCUT=parms[0].second_BOOLCUT;
	   if(pairfit(&qdomain,&ddomain,&score,&rms,&length,&nfit,parms,
	      0,&qstart,&qend,&dstart,&dend,&seqid,&secid,&nequiv,&nsec,hbcmat,parms[0].SCANALIGN,-1,1)==-1) return -1;
  	} else {
	   fprintf(parms[0].LOG," Not performed since Sc < %7.3f\n",parms[0].first_THRESH);
	   score=0.0; rms=100.0; nfit=0; length=0;
	   for(j=0; j<3; ++j) for(k=0; k<3; ++k) hbcmat[j][k]=0;
	}

	 /* reversing the transformation is done after updating the rough 
	  *  matrix below */ 
	 /* modifiy the score:
	  *  This removes the penalty that STAMP introduces when comparing a small portion
	  *   of a large structure.
	  *  During a scan we only want to penalise a score if the alignment with the
	  *   query is short.  For example, when scanning with a rossmann fold domain, one would
	  *   want to be able to find such a domain in glycogen phosphorylase (1GPB), but if the
	  *   penalty for the length of 1GPB was not removed, the score would become very
	  *   small indeed, since the rossmann fold domain is only about 1/6th of the structure.
	  *   (if you don't understand, see the paper) 
	  *  This modification has been moved to the pairfit routine */
	 
	 /* combine the rough matrix with the refined one */
	 update(ddomain.r,ddomain.R,ddomain.v,ddomain.V);
/*	 printmat(ddomain.R,ddomain.V,3,parms[0].LOG); */

	 /* output the results */
	 fprintf(parms[0].LOG,"Sum: %10s & %10s, Sc: %7.3f, RMS: %7.3f, len: %4d, max_len: %4d nfit:%4d, fit_pos: %c %3d %c\n",
		 qdomain.id,ddomain.id,score,rms,length,max(qdomain.ncoords,ddomain.ncoords), nfit,
		 ddomainall.numb[dcount].cid,ddomainall.numb[dcount].n,ddomainall.numb[dcount].in);
	 if(qdomain.ncoords>ddomain.ncoords)
	    ratio=(float)qdomain.ncoords/(float)ddomain.ncoords;
	 else 
	    ratio=(float)ddomain.ncoords/(float)qdomain.ncoords;
	 fprintf(parms[0].LOG,"Sum:      lena: %4d, lenb: %4d, ratio: %7.4f, secdist: %6.2f, seqid: %6.2f, secid: %6.2f\n",
		 qdomain.ncoords,ddomain.ncoords,ratio,t_sec_diff,seqid,secid);
	 if(score>best_score) {
	    best_score=score;
	    best_rms=rms;
	    best_nfit=nfit;
	    best_len=length;
	    best_seqid=seqid;
	    best_secid=secid;
	    best_nsec=nsec;
	    best_nequiv=nequiv;
 	    for(j=0; j<3; ++j) for(k=0; k<3; ++k) best_hbcmat[j][k]=hbcmat[j][k];
	 }
	 if(score>=parms[0].SCANCUT) { 
	    /* outputing the transformation if required */
	    fprintf(TRANS,"%% Similarity is with \n");
	    fprintf(TRANS,"%% %s %s_%d { %c %d %c to %c %d %c }\n",
	      qdomain.id,qdomain.id,ntries+1,qdomain.numb[qstart].cid,
	      qdomain.numb[qstart].n,qdomain.numb[qstart].in,
	      qdomain.numb[qend].cid,qdomain.numb[qend].n,
	      qdomain.numb[qend].in);
            fprintf(TRANS,"# Sc= %7.3f RMS= %7.3f len= %4d nfit= %4d ",
                score,rms,length,nfit);
            fprintf(TRANS,"seq_id= %5.2f sec_id= %5.2f q_len= %4d d_len= %4d n_sec= %4d n_equiv= %4d ",
                seqid,secid,qdomain.ncoords,ddomainall.ncoords,nsec,nequiv);
            fprintf(TRANS,"fit_pos= %c %3d %c \n",
                ddomainall.numb[dcount].cid,
                ddomainall.numb[dcount].n,
                ddomainall.numb[dcount].in);

	    fprintf(parms[0].LOG,"Similarty is between: %s %c %d %c to %c %d %c and ",
	      qdomain.id,qdomain.numb[qstart].cid,
	      qdomain.numb[qstart].n,qdomain.numb[qstart].in,
	      qdomain.numb[qend].cid,qdomain.numb[qend].n,
	      qdomain.numb[qend].in);
	    fprintf(parms[0].LOG,"%s %c %d %c to %c %d %c\n",
	      ddomain.id,ddomain.numb[dstart].cid,
	      ddomain.numb[dstart].n,ddomain.numb[dstart].in,
	      ddomain.numb[dend].cid,ddomain.numb[dend].n,
	      ddomain.numb[dend].in);

	    fprintf(TRANS,"%s %s_%d { ",ddomain.filename,ddomain.id,ntries+1);
	    if(parms[0].CO==1) { /* truncate domain in output */
	       fprintf(TRANS,"%c %d %c to %c %d %c \n",
		   ddomain.numb[dstart].cid,ddomain.numb[dstart].n,ddomain.numb[dstart].in,
	           ddomain.numb[dend].cid,ddomain.numb[dend].n,ddomain.numb[dend].in);
	    } else {  /* leave ddomain un-truncated */
	     for(j=0; j<ddomain.nobj; ++j) {
                if(ddomain.start[j].cid==' ') ddomain.start[j].cid='_';
                if(ddomain.start[j].in==' ') ddomain.start[j].in='_';
                if(ddomain.end[j].cid==' ') ddomain.end[j].cid='_';
                if(ddomain.end[j].in==' ') ddomain.end[j].in='_';

                switch(ddomain.type[j]) {
                   case 1: fprintf(TRANS," ALL "); break;
                   case 2: fprintf(TRANS," CHAIN %c ",ddomain.start[j].cid); break;
                   case 3: fprintf(TRANS," %c %d %c to %c %d %c ",
                     ddomain.start[j].cid,ddomain.start[j].n,
                     ddomain.start[j].in,
                     ddomain.end[j].cid,ddomain.end[j].n,
                     ddomain.end[j].in); break;
                } /* end of switch... */
                if(ddomain.start[j].cid=='_') ddomain.start[j].cid=' ';
                if(ddomain.start[j].in=='_') ddomain.start[j].in=' ';
                if(ddomain.end[j].cid=='_') ddomain.end[j].cid=' ';
                if(ddomain.end[j].in=='_') ddomain.end[j].in=' ';
             }
	     fprintf(TRANS,"\n");
	   }

	    for(j=0; j<3; ++j) {
	         for(k=0; k<3; ++k) fprintf(TRANS,"%10.5f ",ddomain.R[j][k]); 
		 fprintf(TRANS,"     %10.5f ",ddomain.V[j]);
	         if(j!=2) fprintf(TRANS,"\n");
	    }
	    fprintf(TRANS," }\n");
	    i=0; 
	    fprintf(parms[0].LOG,"Transformation # %d output.\n",ntrans+1);
	    ntrans++;
            if(parms[0].opd == 1) { fprintf(parms[0].LOG," skipping to next domain\n"); break; }
	 } else {
	    fprintf(parms[0].LOG,"No transformation output.\n");
	 }
	 if(score>parms[0].SCANCUT && parms[0].SKIPAHEAD) { /* skip over the similar region to avoid repetition */
/*	    if((dend-parms[0].SCANSLIDE-1)>psd) {
	       psd=dend-parms[0].SCANSLIDE-1;
	       ped=dend-1;
	       fprintf(parms[0].LOG,"skipping over this region...\n");
	    } */
	 }

	 revmatmult(ddomain.R,ddomain.V,ddomain.coords,ddomain.ncoords,parms[0].PRECISION);
	 /* reverses both the initial rough fit and the refined fit together */
	} 
	if(parms[0].SCANTRUNC) {
	 if((D2overlay-D1overlay)<Min_Size) {
	   ddomain.sec[ddomain.ncoords]=dendsec;
	   ddomain.aa[ddomain.ncoords]=dendaa;
	 }
	 if((Q2overlay-Q1overlay)<Min_Size) {
	   qdomain.aa[qdomain.ncoords]=qendaa;
	   qdomain.sec[qdomain.ncoords]=qendsec;
	 }
	}
	fprintf(parms[0].LOG,"\n");
	P+=parms[0].SCANSLIDE; /* slide the sequences along */
	ntries++;
	if(ntrans>0 && parms[0].opd == 1) { fprintf(parms[0].LOG," skipping to next domain\n"); break; }
       } 
       fprintf(parms[0].LOG,"Fitting %s to %s, %4d initial fits were tried, %4d transformations were output\n",qdomain.id,ddomain.id,ntries,ntrans);
       for(j=0; j<ddomainall.ncoords; ++j) free(ddomainall.coords[j]);
      } else {
	 fprintf(parms[0].LOG,"%s -- skipped.\n",ddomain.id);
      } 

      m = (int)(best_nequiv*best_seqid/(float)100);
      n = best_nequiv;
      Pm = murzin_P(n,m,0.1);

      if(strcmp(parms[0].logfile,"silent")!=0) {
        fprintf(parms[0].LOG,"Summary: %10s %10s %4d %4d ",qdomain.id,ddomain.id,ntries,ntrans);
        fprintf(parms[0].LOG,"%7.3f %7.3f %4d %4d %4d %4d %4d %4d %5.2f %5.2f %7.2e",
               best_score,best_rms,best_len,qdomain.ncoords,ddomain.ncoords,best_nfit,best_nsec,best_nequiv,best_seqid,best_secid,Pm);
/*      for(j=0; j<3; ++j) for(k=i; k<3; ++k) fprintf(parms[0].LOG,"%3d ",best_hbcmat[j][k]); */
        fprintf(parms[0].LOG,"\n");
      } else {

	printf("Scan %-15s %-15s %4d ",qdomain.id,ddomain.id,ntrans);

        printf("%7.3f %7.3f %4d %4d %4d %4d %4d %4d %6.2f %6.2f ",
               best_score,best_rms,qdomain.ncoords,ddomain.ncoords,best_len,best_nfit,
	       best_nequiv,best_nsec,best_seqid,best_secid);
        if(Pm<1e-4) { printf("%7.2e",Pm); }
        else { printf("%7.5f",Pm); }
        printf("\n");
	fflush(stdout);
      }
      for(j=0; j<3; ++j) {
        free(ddomain.R[j]);
        free(ddomain.r[j]);
      }
      free(ddomain.r);
      free(ddomain.R);
      free(ddomain.v);
      free(ddomain.V);
      free(ddomain.type);
      free(ddomain.start);
      free(ddomain.end);
      free(ddomain.reverse);
      /* reset the pointers */
      ddomain.coords=ddomainall.coords;
      ddomain.sec=ddomainall.sec;
      ddomain.aa=ddomainall.aa;
      ddomain.numb=ddomainall.numb;
      dcount++;
      domain_count++;
   } 
   fclose(IN);
   fclose(TRANS);
	 
   /* freeing to keep purify happy */
   free(ddomainall.aa);
   free(ddomainall.sec);
   free(ddomainall.numb);
   free(ddomainall.coords);
   free(qdomainall.aa);
   free(qdomainall.sec);
   free(qdomainall.numb);
   free(qdomainall.coords);

   free(atoms1);
   free(atoms2);
   for(i=0; i<3; ++i) {
      free(rtemp[i]);
      free(hbcmat[i]);
      free(best_hbcmat[i]);
   }
   free(hbcmat);
   free(best_hbcmat);
   free(rtemp);
   free(vtemp);
   return 0;
}

int disp(domain,OUTPUT)
struct domain_loc domain;
FILE *OUTPUT;
{
	int i/*,j,k*/;
	fprintf(OUTPUT,"The first ten coordinates\n");
        for(i=0; i<10; ++i) 
	   fprintf(OUTPUT,"%c %8d %8d %8d\n",domain.aa[i],domain.coords[i][0],domain.coords[i][1],domain.coords[i][2]);
	fprintf(OUTPUT,"R,V:\n");
	printmat(domain.R,domain.V,3,OUTPUT);
	fprintf(OUTPUT,"r,v:\n");
	printmat(domain.r,domain.v,3,OUTPUT);
	return 0;
}
