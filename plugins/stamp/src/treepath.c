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
/*****************************************************************************
  Also developed by: John Eargle
  Copyright (2004) John Eargle
*****************************************************************************/
#include <stdio.h>
#include <math.h>

#include <stamp.h>

/* treepath: given a cluster, a list of domains, and a probability matrix, returns
 *  the transformation superimposing the two clusters and updates their alignments
 *  Arguments: 
 *    struct domain_loc *domain -> array of domain descriptors
 *    int ndomain -> number of domain descriptors
 *    struct cluster cl -> phylogenetic tree node with 2 subclusters
 *    float **R2 -> 
 *    float *V2 -> 
 *    int **prob -> dynamic programming matrix
 *    float *score <- 
 *    float *rms <- 
 *    int *length <- 
 *    int *nfit <- number of aligned residues - 1 (starts at 0)
 *    float *Pij <- 
 *    float *Dij <- 
 *    float *distance <- 
 *    float *Pijp <- 
 *    float mean -> 
 *    float sd -> 
 *    char *fpuse <- 
 *    char *ftouse <- 
 *    struct parameters *parms -> STAMP parameters
 *  Return:
 *    int 0 for success, -1 for failure
 */

int treepath(struct domain_loc *domain, int ndomain, struct cluster cl,
	     float **R2, float *V2, int **prob, float *score, float *rms,
	     int *length, int *nfit, float *Pij, float *Dij, float *distance,
	     float *Pijp, float mean, float sd, char *fpuse, char *ftouse,
	     struct parameters *parms) {
  
  char *bestaseq, *bestbseq,*touse,*puse;
  char *taseq, *tbseq;
  char *finalaseq, *finalbseq;
  char *tmp;
  char **pseq1,**pseq2;
  unsigned char **patha;
  
  int i,j,k,l/*,m,q*/;
  int pen;
  int use,nogap;
  int match, allen,total;
  int inda,indb;
  int ia,ib;
  int pasize,pbsize;
  int natoms;
  int proba,probb;
  int finala,finalb,diffab;
  int datcount;
  
  int *acount,*bcount;
  
  int at1[3],at2[3];
  int minscore;
  int **atoms1,**atoms2;
  
  float *sumsa, *sumsb;
  
  struct olist *result;
  struct path *sortarr;
  
  
  /* Allocating memory */
  acount = (int*)malloc(ndomain*sizeof(int));
  bcount = (int*)malloc(ndomain*sizeof(int));
  sumsa = (float*)malloc(3*sizeof(float));
  sumsb = (float*)malloc(3*sizeof(float));
  taseq = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  tbseq = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  bestaseq = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  bestbseq = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  finalaseq = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  finalbseq = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  touse = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  puse = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  tmp = (char*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(char));
  pseq1=(char**)malloc(cl.a.number*sizeof(char*));
  pseq2=(char**)malloc(cl.b.number*sizeof(char*));
  
  
  pasize=strlen(domain[cl.a.member[0]].align)+1;
  pbsize=strlen(domain[cl.b.member[0]].align)+1;
  
  patha = (unsigned char**)malloc((unsigned)(pbsize+1)*sizeof(unsigned char*));
  for (i=0;i<(pbsize+1); ++i)
    patha[i]=(unsigned char *)malloc((unsigned)(pasize)*sizeof(unsigned char));
  result = (struct olist*)malloc((unsigned)sizeof(struct olist) *(pasize));
  
  
  /* We need to establish a set of equivalences for *all* proteins in each cluster.
   *  To do this we pass a sequence consisting of '1's and ' 's to aliseq in order to get
   *   a list of *aligned* positions to use for each cluster */
  /* Calling the Smith and Waterman routine */
  if (parms[0].BOOLEAN) {
    minscore=parms[0].MINFIT;
    pen=0;
  } else {
    minscore=parms[0].MINFIT*(int)((float)parms[0].PRECISION*parms[0].CUTOFF);
    pen=(int)((float)parms[0].PRECISION*parms[0].PAIRPEN);
  }
  if (parms[0].SW==1) {
    /* be warned that this method will upset the statistics described in the paper for multiple alignment */
    match = sw7ccs(pasize+1,pbsize+1,prob,pen,result,&total,patha,minscore,parms[0].CCADD,parms[0].CCFACTOR);
  } else { 
    match = swstruc(pasize+1,pbsize+1,pen,prob,result,&total,patha,minscore);
  }
  
  if (total!=0) {
    sortarr = dosort(result,&pasize,&total);

    /* JE { */
    ppath2(sortarr,&total);
    /* } */

    for (i=0; i<pasize; ++i ) { free(result[i].res); }
    free(result);
    /*	    taseq[0]=tbseq[0]=' '; */
    for (j=0; j<pasize; ++j) {
      taseq[j]='1';
    }
    taseq[j]='\0';
    for(j=0; j<pbsize; ++j)
      tbseq[j]='1';
    tbseq[j]='\0';
    aliseq(taseq,tbseq,&sortarr[0],patha,bestaseq,bestbseq,&allen,&ia,&ib); 
    
    /* check if we have gone over the MAX_SEQ_LEN limit */
    if( (sortarr[0].start.i+allen+(strlen(taseq)-sortarr[0].end.i))>parms[0].MAX_SEQ_LEN ||
	(sortarr[0].start.j+allen+(strlen(tbseq)-sortarr[0].end.j))>parms[0].MAX_SEQ_LEN) {
      fprintf(stderr,"error: MAX_SEQ_LEN limit surpassed, set value > %d in parameter file or command line\n",
		    parms[0].MAX_SEQ_LEN);
      return -1;
    }
    (*length)=allen;
    (*score)=((((float)sortarr[0].score/(float)allen) * 
	       ((float)(allen-ia)/(float)(pasize-1)) * ((float)(allen-ib)/(float)(pbsize-1)) ));
    if(!parms[0].BOOLEAN) (*score)/=(float)parms[0].PRECISION;
    /* SMJS Changed sizeof(int) to sizeof(int *) */
    atoms1=(int**)malloc((*length)*sizeof(int *));
    atoms2=(int**)malloc((*length)*sizeof(int *));
    natoms=0;
    /* get the pointers for each domain to the right place */
    for(k=0; k<cl.a.number; ++k) 
      acount[k]=0;
    for(k=0; k<cl.b.number; ++k) 
      bcount[k]=0;
    for(j=1; j<sortarr[0].start.i; ++j) 
      for(k=0; k<cl.a.number; ++k) 
	acount[k]+=(domain[cl.a.member[k]].align[j-1]!=' ');
    for(j=1; j<sortarr[0].start.j; ++j)
      for(k=0; k<cl.b.number; ++k) 
	bcount[k]+=(domain[cl.b.member[k]].align[j-1]!=' ');
    
    /* Generating a string from which the new alignment can be derived. 
     * We first  need to pad each old alignment with ' 's to make all sequences 
     *  aligned the same length (this makes this simpler for future fits) */
    finala=finalb=0;

    /* JE { */

    fprintf(parms[0].LOG,"bestaseq: %s\n",&bestaseq[0]);
    fprintf(parms[0].LOG,"bestbseq: %s\n",&bestbseq[0]);

    /* Now we add the start bits of the alignment that were not in the optimal path */ 
    if ((sortarr[0].start.i-sortarr[0].start.j) >= 0) {
      for (j=0; j<(sortarr[0].start.i-sortarr[0].start.j); ++j) {
	finalaseq[finala++]='1';
	finalbseq[finalb++]=' ';
      }
      /* These bits need to be offset from each other so that they don't produce a false alignment */
      for (j=0; j<sortarr[0].start.j-1; ++j) {
	finalaseq[finala++]='1';
	finalbseq[finalb++]=' ';
      }
      for (j=0; j<sortarr[0].start.j-1; ++j) {
	finalaseq[finala++]=' ';
	finalbseq[finalb++]='1';
      }
    }
    else {
      for (j=0; j<(sortarr[0].start.j-sortarr[0].start.i); ++j) {
	finalaseq[finala++]=' ';
	finalbseq[finalb++]='1';
      }
      /* These bits need to be offset from each other so that they don't produce a false alignment */
      for (j=0; j<sortarr[0].start.i-1; ++j) {
	finalaseq[finala++]=' ';
	finalbseq[finalb++]='1';
      }
      for (j=0; j<sortarr[0].start.i-1; ++j) {
	finalaseq[finala++]='1';
	finalbseq[finalb++]=' ';
      }
    }
    /* } */
    
    /* JE rm{
      for(j=0; j<(sortarr[0].start.i-sortarr[0].start.j); ++j) finalbseq[finalb++]=' ';
      for(j=0; j<(sortarr[0].start.j-sortarr[0].start.i); ++j) finalaseq[finala++]=' ';
      
      // Now we add the start bits of the alignment that were not in the optimal path 
      
      for(j=0; j<sortarr[0].start.i-1; ++j) finalaseq[finala++]='1';
      for(j=0; j<sortarr[0].start.j-1; ++j) finalbseq[finalb++]='1';
      }
    */
    
    for(j=0; j<finala; ++j) 
      Pij[j]=Dij[j]=Pijp[j]=distance[j]=0.0;
    datcount=finala;
    
    /* Now we copy bestaseq and bestbseq to finalaseq and finalbseq
       as appropriate */
    sprintf(&finalaseq[finala],"%s",&bestaseq[0]); 
    sprintf(&finalbseq[finalb],"%s",&bestbseq[0]); 
    finala+=strlen(&bestaseq[0]);
    finalb+=strlen(&bestbseq[0]);
    
    /* JE { */
    /* Now pad each sequence with spaces as appropriate and add the end
       bits of the alignment that were not in the optimal path */
    diffab=(finala-finalb);
    if (diffab <= 0) {
      for(j=0; j<(-1*diffab); ++j) {
	Pij[finala]=Dij[finala]=Pijp[finala]=distance[finala]=0.0;
	finalaseq[finala++]=' ';
      }
      /* These bits need to be offset from each other so that they don't produce a false alignment */
      for(j=sortarr[0].end.j; j<strlen(&tbseq[1]); ++j) {
	finalaseq[finala++]=' ';
	finalbseq[finalb++]='1';
      }
      for(j=sortarr[0].end.i; j<strlen(&taseq[1]); ++j) {
	finalaseq[finala++]='1';
	finalbseq[finalb++]=' ';
      }
    }
    else {
      for(j=0; j<diffab; ++j) {
	Pij[finalb]=Dij[finalb]=Pijp[finalb]=distance[finalb]=0.0;
	finalbseq[finalb++]=' ';
      }
      /* These bits need to be offset from each other so that they don't produce a false alignment */
      for(j=sortarr[0].end.i; j<strlen(&taseq[1]); ++j) {
	Pij[finala]=Dij[finala]=Pijp[finala]=distance[finala]=0.0;
	Pij[finalb]=Dij[finalb]=Pijp[finalb]=distance[finalb]=0.0;
	finalaseq[finala++]='1';
	finalbseq[finalb++]=' ';
      }
      for(j=sortarr[0].end.j; j<strlen(&tbseq[1]); ++j) {
	Pij[finala]=Dij[finala]=Pijp[finala]=distance[finala]=0.0;
	Pij[finalb]=Dij[finalb]=Pijp[finalb]=distance[finalb]=0.0;
	finalaseq[finala++]=' ';
	finalbseq[finalb++]='1';
      }
    }
    /* } */
    
    /* JE rm{
    // Now we add the end bits of the alignment that were not in the optimal path 
      for(j=sortarr[0].end.i-1; j<strlen(&taseq[1]); ++j) finalaseq[finala++]='1';
      for(j=sortarr[0].end.j-1; j<strlen(&tbseq[1]); ++j) finalbseq[finalb++]='1';
      
      // Now pad each sequence with spaces as appropriate
      
      diffab=(finala-finalb);
      for(j=0; j<(-1*diffab); ++j) { Pij[finala]=Dij[finala]=Pijp[finala]=distance[finala]=0.0; finalaseq[finala++]=' '; }
      for(j=0; j<diffab; ++j) { Pij[finalb]=Dij[finalb]=Pijp[finalb]=distance[finalb]=0.0; finalbseq[finalb++]=' '; }
      }
    */
    
    if(finala!=finalb) {
      fprintf(stderr,"error: something funny is going on in treepath()\n");
      return -1;
    }
    finalaseq[finala]=finalbseq[finalb]='\0'; 
    
    /* JE { */
    fprintf(parms[0].LOG,"finalaseq: %s\nfinalbseq: %s\n",finalaseq,finalbseq);
    /* } */

    /* finalaseq and finalbseq now consist of strings of '1's and ' 's that we
     *  can use in conjunction with all the sequences in cluster a and b to
     *  generate the new alignment */
    
    /* find the equivalences and calculate a set of average coordinates */
    proba=sortarr[0].start.i; probb=sortarr[0].start.j; /* probability matrix pointers */
    for(j=0; j<=strlen(&bestaseq[0]); ++j) {
      use=1;
      for(k=0; k<cl.a.number; ++k) {
	inda=cl.a.member[k];
	use*=(domain[inda].align[proba-1]!=' ');
      }
      for(k=0; k<cl.b.number; ++k) {
	indb=cl.b.member[k];
	use*=(domain[indb].align[probb-1]!=' ');
      }
      if(use) {
	for(k=0; k<cl.a.number; ++k) {
	  inda=cl.a.member[k];
	  for(l=0; l<3; ++l) {
	    if(k==0) sumsa[l]=0.0;
	    sumsa[l]+=domain[inda].coords[acount[k]][l];
	  }
	}
	for(k=0; k<cl.b.number; ++k) {
	  indb=cl.b.member[k];
	  for(l=0; l<3; ++l) {
	    if(k==0) sumsb[l]=0.0;
	    sumsb[l]+=domain[indb].coords[bcount[k]][l];
	  }
	}
	for(l=0; l<3; ++l) {
	  at1[l]=(int)(sumsa[l]/(float)cl.a.number);
	  at2[l]=(int)(sumsb[l]/(float)cl.b.number);
	}
	
	if(!parms[0].BOOLEAN) {
	  Pij[datcount+j]=(((float)prob[proba][probb]/(float)parms[0].PRECISION)*sd-mean)/(float)parms[0].PRECISION;
	  Pijp[datcount+j]=((float)prob[proba][probb]/parms[0].PRECISION);
	  Dij[datcount+j]=exp(distance[datcount+j]/parms[0].const1);
	} else {
	  Pij[datcount+j]=(float)prob[proba][probb];
	  Pijp[datcount+j]=Dij[datcount+j]=0.0;
	}
	
	distance[datcount+j]=idist(at1,at2,parms[0].PRECISION);
	distance[datcount+j]=sqrt(distance[datcount+j]);
	
	/* SMJS Changed bestaseq[j] =='1' to bestaseq[j+1] */
	if( ( (prob[proba][probb]>=(parms[0].CUTOFF*(float)parms[0].PRECISION) && !parms[0].BOOLEAN) || 
	      (prob[proba][probb]==1 && parms[0].BOOLEAN)) && bestaseq[j+1]=='1' && bestbseq[j+1]=='1') {
	  atoms1[natoms]=(int*)malloc(3*sizeof(int));
	  atoms2[natoms]=(int*)malloc(3*sizeof(int));
	  for(k=0; k<3; ++k) {
	    atoms1[natoms][k]=at1[k];
	    atoms2[natoms][k]=at2[k];
	  }
	  natoms++;
	} 
      }
      if(bestaseq[j+1]=='1') {
	for(k=0; k<cl.a.number; ++k) {
	  inda=cl.a.member[k];
	  acount[k]+=(domain[inda].align[proba-1]!=' ');
	}
	proba++;
      }
      if(bestbseq[j+1]=='1') {
	for(k=0; k<cl.b.number; ++k) {
	  indb=cl.b.member[k];
	  bcount[k]+=(domain[indb].align[probb-1]!=' ');
	}
	probb++;
      }
    }
    /* We now have an average set of coordinate with which to fit */
    if(natoms>=3) {
      (*rms)=matfit(atoms1,atoms2,R2,V2,natoms,1,parms[0].PRECISION);
    } else {
      fprintf(parms[0].LOG,"\nWARNING: matfit NOT called as there were less than three equivalences\n");
      (*rms)=100.0;
    }
    /* Now to generate the alignment 
     *
     * Cluster A first */
    for(k=0; k<cl.a.number; ++k) {
      acount[k]=0;
      inda=cl.a.member[k];
      for(j=0; j<strlen(finalaseq); ++j) {
	if(finalaseq[j]=='1') {
	  tmp[j]=domain[inda].align[acount[k]];
	  acount[k]++;
	} else {
	  tmp[j]=' ';
	}
      } 
      tmp[j]='\0';
      strcpy(domain[inda].align,tmp);
      pseq1[k]=domain[inda].align;
    }
    
    /* Cluster B */
    for(k=0; k<cl.b.number; ++k) {
      bcount[k]=0;
      indb=cl.b.member[k];
      for(j=0; j<strlen(finalbseq); ++j) {
	if(finalbseq[j]=='1') {
	  tmp[j]=domain[indb].align[bcount[k]];
	  bcount[k]++;
	} else {
	  tmp[j]=' ';
	}
      }
      tmp[j]='\0';
      strcpy(domain[indb].align,tmp);
      pseq2[k]=domain[indb].align;
    }
    
    /* now for the numbers and stuff */
    if(parms[0].TREEALLALIGN || parms[0].TREEALIGN) {
      for(k=0; k<strlen(domain[cl.a.member[0]].align); ++k) {
	nogap=1;
	for(l=0; l<cl.a.number; ++l) nogap*=(domain[cl.a.member[l]].align[k]!=' ');
	for(l=0; l<cl.b.number; ++l) nogap*=(domain[cl.b.member[l]].align[k]!=' ');
	if(nogap) {
	  touse[k]=(char)( ((Pijp[k]>=parms[0].CUTOFF && !parms[0].BOOLEAN) || 
			    (Pij[k]==1 && parms[0].BOOLEAN) ) +48);
	  if(Pijp[k]<10.00) {
	    if(!parms[0].BOOLEAN) {
	      puse[k]=(char)((int)(Pijp[k]*(Pijp[k]>0.00)) + 48);
	    } else {
	      puse[k]=(char)((int)Pij[k] + 48);
	    }
	  } else {
	    puse[k]='*';
	  }
	} else puse[k]=touse[k]=' ';
      }
    }
    puse[k]=touse[k]='\0';
    if(parms[0].TREEALLALIGN) {
      fprintf(parms[0].LOG,"\n");
      display_align(pseq1,cl.a.number,pseq2,cl.b.number,pseq1,pseq2,
		    touse,puse,parms[0].COLUMNS,0,1,parms[0].LOG);
      fprintf(parms[0].LOG,"\n");
    }
  } else {
    fprintf(stderr,"error: no alignments found\n");
    fprintf(parms[0].LOG,"  check parameters and the integrity of the structures being aligned\n");
    return -1;
  } 
  strcpy(fpuse,puse); strcpy(ftouse,touse);
  (*nfit)=natoms;
  
  /* freeing memory */
  for(i=0; i<pbsize+1; ++i) 
    free(patha[i]);
  free(patha);
  
  for(i=0; i<natoms; ++i) {
    free(atoms1[i]);
    free(atoms2[i]);
  }
  free(atoms1); free(atoms2);
  free(sortarr);
  free(acount); free(bcount);
  free(sumsa); free(sumsb);
  free(taseq); free(tbseq);
  free(bestaseq); free(bestbseq);
  free(finalaseq); free(finalbseq);
  free(touse); free(puse);
  free(tmp);
  free(pseq1); free(pseq2);
  
  return 0;
}
