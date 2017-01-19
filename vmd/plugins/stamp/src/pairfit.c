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


int pairfit(struct domain_loc *domain1, struct domain_loc *domain2, float *score, float *rms,
	int *length, int *nfit, struct parameters *parms, int rev,
	int *start1, int *end1, int *start2, int *end2,
	float *seqid, float *secid, int *nequiv, int *nsec, int **hbcmat,
	int ALIGN, int count, int FINAL) { 

     int i,j/*,k*/,iter;
     int len,pcount;
     int temp_len;
     int sec_len1,sec_len2;
     int scorerise;
     int c1,c2;
     int **prob;
     int not_gap,align_len;
     int seqcount,seccount;
     int n_sec_equiv,n_pos_equiv;
     int slen,neighbors;
     int nsec1,nsec2;
     int in_sec1,in_sec2;
     int last_matched1, last_matched2;
     int xpos,ypos;
     int started1,started2;

     char ss1,ss2;
     char *touse, *puse;
     char *psec1,*psec2;

     float **r,**rt;
     float oldscore,scorediff;
     float *v,*vt;
     float rmsold;
     float *fpuse/*,*f2,*f3,*f4*/;

     struct cluster *cl;
     struct domain_loc *dcl;

/* SMJS Changed malloc(3*sizeof(float*) to malloc(3*sizeof(float) */
     v=(float*)malloc(3*sizeof(float));
/* SMJS Changed malloc(3*sizeof(float*) to malloc(3*sizeof(float) */
     vt=(float*)malloc(3*sizeof(float));
     r=(float**)malloc(3*sizeof(float*));
     rt=(float**)malloc(3*sizeof(float*));
     for(i=0; i<3; ++i) {
        r[i]=(float*)malloc(3*sizeof(float));
        rt[i]=(float*)malloc(3*sizeof(float));
     }

     cl=(struct cluster*)malloc(sizeof(struct cluster));
     cl[0].a.member=(int*)malloc(sizeof(int));
     cl[0].b.member=(int*)malloc(sizeof(int));
     dcl=(struct domain_loc*)malloc(2*sizeof(struct domain_loc));
     cl[0].a.number=1; cl[0].a.member[0]=0;
     cl[0].b.number=1; cl[0].b.member[0]=1;
     /* these will be used if pairwise output is required */

     /* allocating the probability matrix */
     prob=(int**)malloc((domain1[0].ncoords+2)*sizeof(int*));
     for(i=0; i<(domain1[0].ncoords+2); ++i)
        prob[i]=(int*)malloc((domain2[0].ncoords+2)*sizeof(int));

     touse = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
     puse = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
     fpuse = (float*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(float));
     psec1 = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
     psec2 = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
     domain1[0].align = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
     domain2[0].align = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));


     /* The matrix rt/vt must be set to the previous transformation 
      *  r/v must be set to I and the zero vector */
     if(parms[0].SW==1) 
	for(i=0; i<domain1[0].ncoords+2; ++i) 
	  for(j=0;j<domain2[0].ncoords+2; ++j) prob[i][j]=0;

     for(i=0; i<3; ++i) {
        v[i]=0.0;
	vt[i]=0.0;
        for(j=0; j<3; ++j) {
	   if(i==j) r[i][j]=rt[i][j]=1.0;
	   else r[i][j]=rt[i][j]=0.0;
	}
     } 
     rmsold=0.0;
     *score=0.0;
     scorediff=parms[0].SCORETOL+1;
     scorerise=1;
     iter=1;

     (*nfit)=100; (*seqid)=0.0; (*secid)=0.0;
     while(scorediff>parms[0].SCORETOL && iter<=parms[0].MAXPITER && (*nfit)>=3 && (scorerise || parms[0].SCORERISE!=1) ) {
	/* The probabity matrix is calculated after Rosman and Argos */
	if(parms[0].SW==1) {
	  /* corner cutting */
/*	  fprintf(parms[0].LOG,"Corner cutting, CCFACTOR=%f\n",parms[0].CCFACTOR); */
	  if(ccprobcalc(domain1[0].coords,domain2[0].coords,
	     prob, domain1[0].ncoords,domain2[0].ncoords,parms)==-1) return -1;
	} else {
	  if(probcalc(domain1[0].coords,domain2[0].coords,
	     prob,domain1[0].ncoords,domain2[0].ncoords,parms)==-1) return -1;
	}

	oldscore=*score;
	(*rms) = pairpath(domain1[0],domain2[0],prob,1,r,v,&len,score,nfit,touse,fpuse,start1,end1,start2,end2,parms);

/* float pairpath(struct domain_loc domain1, struct domain_loc domain2, int **prob,
        long int entry, float **R2, float *V2, int *len, float *score,
        int *nfit, char *ftouse, float *fpuse, int *start1, int *end1,
        int *start2,int *end2, struct parameters *parms);
*/
	if((*score)>0)
	   scorediff=(100*fabs(*score-oldscore)/(*score));
	else 
	   scorediff=0.0;
	scorerise=(((*score)-oldscore)>0);

	rmsold=(*rms);
        fprintf(parms[0].LOG,"iteration: %3d, ",iter);
	fprintf(parms[0].LOG,"RMS: %7.3f, ",(*rms));
	fprintf(parms[0].LOG," Sc diff: %6.2f %%, Sc: %7.3f, len: %4d, nfit: %4d\n",scorediff,*score,len,*nfit);
	++iter;
	/* transform the coordinates after the most recent transformation */
	matmult(r,v,domain2[0].coords,domain2[0].ncoords,parms[0].PRECISION);

	update(r,rt,v,vt); /* since we need to get the old coordinates back, we must keep track of the overall transformation */
     }  /* End of while. */ 
     (*length)=len;
     if(parms[0].PAIRPLOT) { 
	probplot(prob,domain1[0].ncoords,
		 domain2[0].ncoords,1,
		 (int)((float)parms[0].PRECISION*parms[0].CUTOFF*(!parms[0].BOOLEAN))+(!parms[0].BOOLEAN), 
		 parms[0].LOG); 
     } 
     if(scorediff<=parms[0].SCORETOL) fprintf(parms[0].LOG,"Convergence ");
     else fprintf(parms[0].LOG,"No convergence ");
     fprintf(parms[0].LOG,"after %d iterations \n",(iter-1));

     update(rt,domain2[0].r,vt,domain2[0].v);  

     /* output the alignment if required */
     if((parms[0].PAIRWISE && parms[0].PAIROUTPUT && count>=0) ||
        (ALIGN && ((*rms)>0.0) && (!parms[0].SCAN || (parms[0].SCANMODE==1 && (*score)>parms[0].SCANCUT)))) {
	/* calculate puse */
	temp_len=strlen(domain1[0].align);
	sec_len1=strlen(domain1[0].sec);
	sec_len2=strlen(domain2[0].sec);
	started1=0; started2=0;
	for(i=0; i<temp_len; ++i) {
	   if(fpuse[i]<0)
	     puse[i]=' ';
	   else if(fpuse[i]<10)
	     puse[i]=((int)fpuse[i]+48);
	   else
	     puse[i]='*';
	}
	/* make the secondary structure */
	pcount=0; 
	for(i=0; i<temp_len; ++i) {
	   if(domain1[0].align[i]!=' ') {
	      psec1[i]=domain1[0].sec[pcount];
	      if(pcount>sec_len1) psec1[i]='?';
	      pcount++;
	   } else psec1[i]=' ';
	}
	psec1[i]='\0';
	pcount=0;
	for(i=0; i<temp_len; ++i) {
	   if(domain2[0].align[i]!=' ') {
	      psec2[i]=domain2[0].sec[pcount]; 
	      if(pcount>sec_len2) psec2[i]='?';
	      pcount++; 
	   } else psec2[i]=' ';
	}
	psec2[i]='\0';
	if(parms[0].PAIRALIGN || parms[0].SCANALIGN) {
	    if(strcmp(parms[0].logfile,"silent")!=0) {
		display_align(&domain1[0].align,1,&domain2[0].align,1,
	          &psec1,&psec2,touse,puse,parms[0].COLUMNS,1,1,parms[0].LOG);
	    } else {
		display_align(&domain1[0].align,1,&domain2[0].align,1,
                  &psec1,&psec2,touse,puse,parms[0].COLUMNS,1,1,parms[0].LOG);
	    }
	}

     }
     if(parms[0].PAIROUTPUT && FINAL) { 
        dcl[0]=domain1[0];
        dcl[1]=domain2[0];
        if(makefile(dcl,0,cl[0],count,(*score),(*rms),(*length),(*nfit),fpuse,fpuse,fpuse,fpuse,1,parms)==-1) return -1;
     }
     /* calculate pairwise sequence and secondary structure identity */
     seqcount=0; seccount=0;
     align_len=0;
     c1=0; c2=0; 
     n_sec_equiv=0; 
     in_sec1=0; in_sec2=0;
     n_pos_equiv=0;
     (*nequiv) = 0;
     nsec1=nsec2=0; last_matched1=last_matched2=-1;
     slen=strlen(domain1[0].align);
     for(i=0; i<3; ++i) {
	for(j=0; j<3; ++j) hbcmat[i][j]=0;
     }
     for(i=0; i<slen; ++i) {
	not_gap=0;
	if(domain1[0].align[i]!=' ' && domain2[0].align[i]!=' ') not_gap=1;
	switch(domain1[0].sec[c1]) {
	   case 'H': case 'G': ss1='H'; break;
	   case 'E': case 'B': ss1='B'; break;
	   default: ss1='c';
	}
	switch(domain2[0].sec[c2]) {
           case 'H': ss2='H'; break;
           case 'E': case 'B': ss2='B'; break;
           default: ss2='c';
        }

	neighbors=0;
	for(j=i+1; j<slen && j<(i+5); ++j) {
	  if(fpuse[j]>=parms[0].second_CUTOFF) neighbors++;
	  else break;
	}
	for(j=i-1; j>0 && j>(i-5); --j) {
	   if(fpuse[j]>=parms[0].second_CUTOFF) neighbors++;
          else break;
        }
	if(fpuse[i]>parms[0].second_CUTOFF && neighbors>=2) {

	    /* identities are only in structural equivalences */
	    if(not_gap && domain1[0].align[i]==domain2[0].align[i]) {
			seqcount++; 
	    } 
	    if(not_gap && ss1==ss2) seccount++;

            n_pos_equiv++;
	    if(ss1=='H') xpos=0;
	    else if(ss1=='B') xpos=1;
	    else xpos=2;
	    if(ss2=='H') ypos=0;
            else if(ss2=='B') ypos=1;
            else ypos=2;
	    hbcmat[xpos][ypos]++; 
            if(xpos!=ypos) hbcmat[ypos][xpos]++;
	}

/*	printf("%c %c %c(%c) %c(%c) %7.5f %4d ",domain1[0].align[i],domain2[0].align[i],ss1,domain1[0].sec[c1],ss2,domain2[0].sec[c2],fpuse[i],touse[i]);*/
	if(ss1=='c') {
          in_sec1=0;
 	}  else {
	   if(in_sec1==0) nsec1++;
           in_sec1=1;
	}
        if(ss2=='c') {
          in_sec2=0;
        }  else {
           if(in_sec2==0) nsec2++;
           in_sec2=1;
        }

        /*printf("Conditions: not_gap = %d, in_sec1 = %d, in_sec2 = %d fpuse[%d]>=parms[0].second_CUTOFF = %d, neighbors>=2 = %d, (nsec1!=last_matched1 || nsec2!=last_matched2) = %d\n",not_gap,in_sec1,in_sec2,i,fpuse[i]>=parms[0].second_CUTOFF,neighbors>=2,(nsec1!=last_matched1 || nsec2!=last_matched2)); */
	if(not_gap && in_sec1 && in_sec2 && fpuse[i]>=parms[0].second_CUTOFF && neighbors>=2 && (nsec1!=last_matched1 || nsec2!=last_matched2)) {
	    n_sec_equiv++;
	    last_matched1=nsec1;
	    last_matched2=nsec2;
	}
	if(domain1[0].align[i]!=' ') c1++;
	if(domain2[0].align[i]!=' ') c2++;
	if(c1>0 && c1<=(strlen(domain1[0].sec)-1) && c2>0 && c2<=(strlen(domain2[0].sec)-1)) align_len++;
     }
     /* Determine the approximate number of equivalent secondary structures Pij'>=4.5 & len>=2 */

     fprintf(parms[0].LOG,"Alignment length: %4d, seqid: %4d, secid: %4d, n_sec_equiv: %4d\n",align_len,seqcount,seccount,n_sec_equiv);
     (*nsec)=n_sec_equiv;
     (*nequiv)=n_pos_equiv;
     if((i==0) || ((*score)<0.01) || ((*nequiv)<=0)) {
	(*seqid)=0.0;
	(*secid)=0.0;
     } else {
        (*seqid)=100.0*(float)seqcount/(float)(*nequiv);
        (*secid)=100.0*(float)seccount/(float)(*nequiv);
     }
     /* this means that the coordinates, at the moment are the original coordinates transformed by domain2[0].r/v */

     if(rev && (*rms)>0.0 )  {  /* if specified, reverse this and set domain.r/v = I/0 */

        revmatmult(domain2[0].r,domain2[0].v,domain2[0].coords,domain2[0].ncoords,parms[0].PRECISION); 

	for(i=0; i<3; ++i) {
	   for(j=0; j<3; ++j) 
	      if(i==j) domain2[0].r[i][j]=1.0;
	      else domain2[0].r[i][j]=0.0;
	   domain2[0].v[i]=0.0;
	}
     }

     /* new modification, change start and end to a more sensible numbers.  
      *  This is mostly important for scanning, where one is only after 
      *  the precise region of similarity (i.e. ignoring all the junk hanging
      *  off the ends) */
	temp_len=strlen(domain1[0].align);
	c1=c2=0;
	for(i=0; i<temp_len-1; ++i) {
	   if( (domain1[0].align[i]!=' ' && domain1[0].align[i+1]!=' ' && 
		domain2[0].align[i]!=' ' && domain2[0].align[i+1]!=' ')  && /* not a gap */
	       (fpuse[i]>=parms[0].SCANCUT && fpuse[i+1]>=parms[0].SCANCUT) && /* equivalent */
		c1>=(*start1) && c2>=(*start2) ) { /* further along than we were already */
		  (*start1)=c1; (*start2)=c2;
		  break;
		}
		if(domain1[0].align[i]!=' ') c1++;
		if(domain2[0].align[i]!=' ') c2++;
	 }

	 c1=domain1[0].ncoords-1;
	 c2=domain2[0].ncoords-1;
	 for(i=(temp_len-1); i>0; --i) {
	   if( (domain1[0].align[i]!=' ' && domain1[0].align[i-1]!=' ' &&
                domain2[0].align[i]!=' ' && domain2[0].align[i-1]!=' ')  && /* not a gap */
               (fpuse[i]>=parms[0].SCANCUT && fpuse[i-1]>=parms[0].SCANCUT) && /* equivalent */
                c1<=(*end1) && c2<=(*end2) ) { /* further along than we were already */
                  (*end1)=c1; (*end2)=c2;
                  break;
                }
                if(domain1[0].align[i]!=' ') c1--;
                if(domain2[0].align[i]!=' ') c2--;
         }

     for(i=0; i<(domain1[0].ncoords+2); ++i) 
	free(prob[i]);
     free(prob);
     for(i=0; i<3; ++i) {
        free(r[i]);
        free(rt[i]);
     }
     free(rt); free(r); free(vt); free(v);
     free(puse); free(fpuse); free(touse); free(psec1); free(psec2);
     free(domain1[0].align); free(domain2[0].align);
     free(cl[0].a.member); free(cl[0].b.member);
     free(cl); free(dcl);

     return 0;
}
