#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clustalw.h"
#define ENDALN 127

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

/*
 *   Prototypes
 */
static lint 	pdiff(sint A,sint B,sint i,sint j,sint go1,sint go2);
static lint 	prfscore(sint n, sint m);
static sint 	gap_penalty1(sint i, sint j,sint k);
static sint 	open_penalty1(sint i, sint j);
static sint 	ext_penalty1(sint i, sint j);
static sint 	gap_penalty2(sint i, sint j,sint k);
static sint 	open_penalty2(sint i, sint j);
static sint 	ext_penalty2(sint i, sint j);
static void 	padd(sint k);
static void 	pdel(sint k);
static void 	palign(void);
static void 	ptracepath(sint *alen);
static void 	add_ggaps(void);
static char *     add_ggaps_mask(char *mask, int len, char *path1, char *path2);

/*
 *   Global variables
 */
extern double 	**tmat;
extern float 	gap_open, gap_extend;
extern float    transition_weight;
extern sint 	gap_pos1, gap_pos2;
extern sint 	max_aa;
extern sint 	nseqs;
extern sint 	*seqlen_array;
extern sint 	*seq_weight;
extern sint    	debug;
extern Boolean 	neg_matrix;
extern sint 	mat_avscore;
extern short  	blosum30mt[], blosum40mt[], blosum45mt[];
extern short  	blosum62mt2[], blosum80mt[];
extern short  	pam20mt[], pam60mt[];
extern short  	pam120mt[], pam160mt[], pam350mt[];
extern short  	gon40mt[], gon80mt[];
extern short    gon120mt[], gon160mt[], gon250mt[], gon350mt[];
extern short  	clustalvdnamt[],swgapdnamt[];
extern short  	idmat[];
extern short    usermat[];
extern short    userdnamat[];
extern Boolean user_series;
extern UserMatSeries matseries;

extern short	def_dna_xref[],def_aa_xref[],dna_xref[],aa_xref[];
extern sint		max_aln_length;
extern Boolean	distance_tree;
extern Boolean	dnaflag;
extern char 	mtrxname[];
extern char 	dnamtrxname[];
extern char 	**seq_array;
extern char 	*amino_acid_codes;
extern char     *gap_penalty_mask1,*gap_penalty_mask2;
extern char     *sec_struct_mask1,*sec_struct_mask2;
extern sint     struct_penalties1, struct_penalties2;
extern Boolean  use_ss1, use_ss2;
extern Boolean endgappenalties;

static sint 	print_ptr,last_print;
static sint 		*displ;

static char   	**alignment;
static sint    	*aln_len;
static sint    *aln_weight;
static char   	*aln_path1, *aln_path2;
static sint    	alignment_len;
static sint    	**profile1, **profile2;
static lint 	    *HH, *DD, *RR, *SS;
static lint 	    *gS;
static sint    matrix[NUMRES][NUMRES];
static sint    nseqs1, nseqs2;
static sint    	prf_length1, prf_length2;
static sint    *gaps;
static sint    gapcoef1,gapcoef2;
static sint    lencoef1,lencoef2;
static Boolean switch_profiles;

lint prfalign(sint *group, sint *aligned)
{

  static Boolean found;
  static Boolean negative;
  static Boolean error_given=FALSE;
  static sint    i, j, count = 0;
  static sint  NumSeq;
  static sint    len, len1, len2, is, minlen;
  static sint   se1, se2, sb1, sb2;
  static sint  maxres;
  static sint int_scale;
  static short  *matptr;
  static short	*mat_xref;
  static char   c;
  static lint    score;
  static float  scale;
  static double logmin,logdiff;
  static double pcid;


  alignment = (char **) ckalloc( nseqs * sizeof (char *) );
  aln_len = (sint *) ckalloc( nseqs * sizeof (sint) );
  aln_weight = (sint *) ckalloc( nseqs * sizeof (sint) );

  for (i=0;i<nseqs;i++)
     if (aligned[i+1] == 0) group[i+1] = 0;

  nseqs1 = nseqs2 = 0;
  for (i=0;i<nseqs;i++)
    {
        if (group[i+1] == 1) nseqs1++;
        else if (group[i+1] == 2) nseqs2++;
    }

  if ((nseqs1 == 0) || (nseqs2 == 0)) return(0.0);

  if (nseqs2 > nseqs1)
    {
     switch_profiles = TRUE;
     for (i=0;i<nseqs;i++)
       {
          if (group[i+1] == 1) group[i+1] = 2;
          else if (group[i+1] == 2) group[i+1] = 1;
       }
    }
  else
  	switch_profiles = FALSE;

  int_scale = 100;

/*
   calculate the mean of the sequence pc identities between the two groups
*/
        count = 0;
        pcid = 0.0;
	negative=neg_matrix;
        for (i=0;i<nseqs;i++)
          {
             if (group[i+1] == 1)
             for (j=0;j<nseqs;j++)
               if (group[j+1] == 2)
                    {
                       count++;
                       pcid += tmat[i+1][j+1];
                    }
          }

  pcid = pcid/(float)count;

if (debug > 0) fprintf(stdout,"mean tmat %3.1f\n", pcid);


/*
  Make the first profile.
*/
  prf_length1 = 0;
  for (i=0;i<nseqs;i++)
       if (group[i+1] == 1)
		if(seqlen_array[i+1]>prf_length1) prf_length1=seqlen_array[i+1];

  nseqs1 = 0;
if (debug>0) fprintf(stdout,"sequences profile 1:\n");
  for (i=0;i<nseqs;i++)
    {
       if (group[i+1] == 1)
          {
if (debug>0) {
extern char **names;
fprintf(stdout,"%s\n",names[i+1]);
}
             len = seqlen_array[i+1];
             alignment[nseqs1] = (char *) ckalloc( (prf_length1+2) * sizeof (char) );
             for (j=0;j<len;j++)
               alignment[nseqs1][j] = seq_array[i+1][j+1];
		for(j=len;j<prf_length1;j++)
			alignment[nseqs1][j]=gap_pos1;
             alignment[nseqs1][j] = ENDALN;
             aln_len[nseqs1] = prf_length1;
             aln_weight[nseqs1] = seq_weight[i];
             nseqs1++;
          }
    }

/*
  Make the second profile.
*/
  prf_length2 = 0;
  for (i=0;i<nseqs;i++)
       if (group[i+1] == 2)
		if(seqlen_array[i+1]>prf_length2) prf_length2=seqlen_array[i+1];

  nseqs2 = 0;
if (debug>0) fprintf(stdout,"sequences profile 2:\n");
  for (i=0;i<nseqs;i++)
    {
       if (group[i+1] == 2)
          {
if (debug>0) {
extern char **names;
fprintf(stdout,"%s\n",names[i+1]);
}
             len = seqlen_array[i+1];
             alignment[nseqs1+nseqs2] =
                   (char *) ckalloc( (prf_length2+2) * sizeof (char) );
             for (j=0;j<len;j++)
               alignment[nseqs1+nseqs2][j] = seq_array[i+1][j+1];
		for(j=len;j<prf_length2;j++)
			alignment[nseqs1+nseqs2][j]=gap_pos1;
             alignment[nseqs1+nseqs2][j] = ENDALN;
             aln_len[nseqs1+nseqs2] = prf_length2;
             aln_weight[nseqs1+nseqs2] = seq_weight[i];
             nseqs2++;
          }
    }

  max_aln_length = prf_length1 + prf_length2+2;
  
/*
   calculate real length of profiles - removing gaps!
*/
  len1=0;
  for (i=0;i<nseqs1;i++)
    {
       is=0;
       for (j=0; j<MIN(aln_len[i],prf_length1); j++)
	  {
            c = alignment[i][j];
      	    if ((c !=gap_pos1) && (c != gap_pos2)) is++;
          }
       len1+=is;
    }
  len1/=(float)nseqs1;
   
  len2=0;
  for (i=nseqs1;i<nseqs2+nseqs1;i++)
    {
       is=0;
       for (j=0; j<MIN(aln_len[i],prf_length2); j++)
	  {
            c = alignment[i][j];
      	    if ((c !=gap_pos1) && (c != gap_pos2)) is++;
          }
       len2+=is;
    }
  len2/=(float)nseqs2;

  if (dnaflag)
     {
       scale=1.0;
       if (strcmp(dnamtrxname, "iub") == 0)
	{
            matptr = swgapdnamt;
            mat_xref = def_dna_xref;
	}
       else if (strcmp(dnamtrxname, "clustalw") == 0)
	{
            matptr = clustalvdnamt;
            mat_xref = def_dna_xref;
            scale=0.66;
	}
       else 
        {
           matptr = userdnamat;
           mat_xref = dna_xref;
        }
            maxres = get_matrix(matptr, mat_xref, matrix, neg_matrix, int_scale);
            if (maxres == 0) return((sint)-1);
/*
            matrix[0][4]=transition_weight*matrix[0][0];
            matrix[4][0]=transition_weight*matrix[0][0];
            matrix[2][11]=transition_weight*matrix[0][0];
            matrix[11][2]=transition_weight*matrix[0][0];
            matrix[2][12]=transition_weight*matrix[0][0];
            matrix[12][2]=transition_weight*matrix[0][0];
*/
/* fix suggested by Chanan Rubin at Compugen */
           matrix[mat_xref[0]][mat_xref[4]]=transition_weight*matrix[0][0]; 
           matrix[mat_xref[4]][mat_xref[0]]=transition_weight*matrix[0][0]; 
           matrix[mat_xref[2]][mat_xref[11]]=transition_weight*matrix[0][0]; 
           matrix[mat_xref[11]][mat_xref[2]]=transition_weight*matrix[0][0]; 
           matrix[mat_xref[2]][mat_xref[12]]=transition_weight*matrix[0][0]; 
           matrix[mat_xref[12]][mat_xref[2]]=transition_weight*matrix[0][0]; 

          gapcoef1 = gapcoef2 = 100.0 * gap_open *scale;
          lencoef1 = lencoef2 = 100.0 * gap_extend *scale;
    }
  else
    {
  	if(len1==0 || len2==0) {
  		logmin=1.0;
  		logdiff=1.0;
  	}  
  	else {
  		minlen = MIN(len1,len2);
 		logmin = 1.0/log10((double)minlen);
 		if (len2<len1)
    	 		logdiff = 1.0+0.5*log10((double)((float)len2/(float)len1));
  		else if (len1<len2)
  	   		logdiff = 1.0+0.5*log10((double)((float)len1/(float)len2));
  		else logdiff=1.0;
		if(logdiff<0.9) logdiff=0.9;
  	}
if(debug>0) fprintf(stdout,"%d %d logmin %f   logdiff %f\n",
(pint)len1,(pint)len2, logmin,logdiff);
       scale=0.75;
       if (strcmp(mtrxname, "blosum") == 0)
        {
           scale=0.75;
           if (negative || distance_tree == FALSE) matptr = blosum40mt;
           else if (pcid > 80.0)
             {
                matptr = blosum80mt;
             }
           else if (pcid > 60.0)
             {
                matptr = blosum62mt2;
             }
           else if (pcid > 40.0)
             {
                matptr = blosum45mt;
             }
           else if (pcid > 30.0)
             {
                scale=0.5;
                matptr = blosum45mt;
             }
           else if (pcid > 20.0)
             {
                scale=0.6;
                matptr = blosum45mt;
             }
           else 
             {
                scale=0.6;
                matptr = blosum30mt;
             }
           mat_xref = def_aa_xref;

        }
       else if (strcmp(mtrxname, "pam") == 0)
        {
           scale=0.75;
           if (negative || distance_tree == FALSE) matptr = pam120mt;
           else if (pcid > 80.0) matptr = pam20mt;
           else if (pcid > 60.0) matptr = pam60mt;
           else if (pcid > 40.0) matptr = pam120mt;
           else matptr = pam350mt;
           mat_xref = def_aa_xref;
        }
       else if (strcmp(mtrxname, "gonnet") == 0)
        {
	   scale/=2.0;
           if (negative || distance_tree == FALSE) matptr = gon250mt;
           else if (pcid > 35.0)
             {
                matptr = gon80mt;
		scale/=2.0;
             }
           else if (pcid > 25.0)
             {
                if(minlen<100) matptr = gon250mt;
                else matptr = gon120mt;
             }
           else
             {
                if(minlen<100) matptr = gon350mt;
		else matptr = gon160mt;
             }
           mat_xref = def_aa_xref;
           int_scale /= 10;
        }
       else if (strcmp(mtrxname, "id") == 0)
        {
           matptr = idmat;
           mat_xref = def_aa_xref;
        }
       else if(user_series)
        {
           matptr=NULL;
	   found=FALSE;
	   for(i=0;i<matseries.nmat;i++)
		if(pcid>=matseries.mat[i].llimit && pcid<=matseries.mat[i].ulimit)
		{
			j=i;
			found=TRUE;
			break;
		}
	   if(found==FALSE)
	   {
		if(!error_given)
		warning(
"\nSeries matrix not found for sequence percent identity = %d.\n"
"(Using first matrix in series as a default.)\n"
"This alignment may not be optimal!\n"
"SUGGESTION: Check your matrix series input file and try again.",(int)pcid);
		error_given=TRUE;
		j=0;
	   }
if (debug>0) fprintf(stdout,"pcid %d  matrix %d\n",(pint)pcid,(pint)j+1);

           matptr = matseries.mat[j].matptr;
           mat_xref = matseries.mat[j].aa_xref;
/* this gives a scale of 0.5 for pcid=llimit and 1.0 for pcid=ulimit */
           scale=0.5+(pcid-matseries.mat[j].llimit)/((matseries.mat[j].ulimit-matseries.mat[j].llimit)*2.0);
        }
       else 
        {
           matptr = usermat;
           mat_xref = aa_xref;
        }
if(debug>0) fprintf(stdout,"pcid %3.1f scale %3.1f\n",pcid,scale);
      	maxres = get_matrix(matptr, mat_xref, matrix, negative, int_scale);
      if (maxres == 0)
        {
           fprintf(stdout,"Error: matrix %s not found\n", mtrxname);
           return(-1);
        }

          if (negative) {
              gapcoef1 = gapcoef2 = 100.0 * (float)(gap_open);
              lencoef1 = lencoef2 = 100.0 * gap_extend;
	  }
          else {
          if (mat_avscore <= 0)
              gapcoef1 = gapcoef2 = 100.0 * (float)(gap_open + logmin);
	  else
              gapcoef1 = gapcoef2 = scale * mat_avscore * (float)(gap_open/(logdiff*logmin));
              lencoef1 = lencoef2 = 100.0 * gap_extend;
	 }
    }
if (debug>0)
{
fprintf(stdout,"matavscore %d\n",mat_avscore);
fprintf(stdout,"Gap Open1 %d  Gap Open2 %d  Gap Extend1 %d   Gap Extend2 %d\n",
   (pint)gapcoef1,(pint)gapcoef2, (pint)lencoef1,(pint)lencoef2);
fprintf(stdout,"Matrix  %s\n", mtrxname);
}

  profile1 = (sint **) ckalloc( (prf_length1+2) * sizeof (sint *) );
  for(i=0; i<prf_length1+2; i++)
       profile1[i] = (sint *) ckalloc( (LENCOL+2) * sizeof(sint) );

  profile2 = (sint **) ckalloc( (prf_length2+2) * sizeof (sint *) );
  for(i=0; i<prf_length2+2; i++)
       profile2[i] = (sint *) ckalloc( (LENCOL+2) * sizeof(sint) );

/*
  calculate the Gap Coefficients.
*/
     gaps = (sint *) ckalloc( (max_aln_length+1) * sizeof (sint) );

     if (switch_profiles == FALSE)
        calc_gap_coeff(alignment, gaps, profile1, (char)(struct_penalties1 && use_ss1), gap_penalty_mask1,
           (sint)0, nseqs1, prf_length1, gapcoef1, lencoef1);
     else
        calc_gap_coeff(alignment, gaps, profile1, (char)(struct_penalties2 && use_ss2), gap_penalty_mask2,
           (sint)0, nseqs1, prf_length1, gapcoef1, lencoef1);
/*
  calculate the profile matrix.
*/
     calc_prf1(profile1, alignment, gaps, matrix,
          aln_weight, prf_length1, (sint)0, nseqs1);

if (debug>4)
{
extern char *amino_acid_codes;
  for (j=0;j<=max_aa;j++)
    fprintf(stdout,"%c    ", amino_acid_codes[j]);
 fprintf(stdout,"\n");
  for (i=0;i<prf_length1;i++)
   {
    for (j=0;j<=max_aa;j++)
      fprintf(stdout,"%d ", (pint)profile1[i+1][j]);
    fprintf(stdout,"%d ", (pint)profile1[i+1][gap_pos1]);
    fprintf(stdout,"%d ", (pint)profile1[i+1][gap_pos2]);
    fprintf(stdout,"%d %d\n",(pint)profile1[i+1][GAPCOL],(pint)profile1[i+1][LENCOL]);
   }
}

/*
  calculate the Gap Coefficients.
*/

     if (switch_profiles == FALSE)
        calc_gap_coeff(alignment, gaps, profile2, (char)(struct_penalties2 && use_ss2), gap_penalty_mask2,
           nseqs1, nseqs1+nseqs2, prf_length2, gapcoef2, lencoef2);
     else
        calc_gap_coeff(alignment, gaps, profile2, (char)(struct_penalties1 && use_ss1), gap_penalty_mask1,
           nseqs1, nseqs1+nseqs2, prf_length2, gapcoef2, lencoef2);
/*
  calculate the profile matrix.
*/
     calc_prf2(profile2, alignment, aln_weight,
           prf_length2, nseqs1, nseqs1+nseqs2);

     aln_weight=ckfree((void *)aln_weight);

if (debug>4)
{
extern char *amino_acid_codes;
  for (j=0;j<=max_aa;j++)
    fprintf(stdout,"%c    ", amino_acid_codes[j]);
 fprintf(stdout,"\n");
  for (i=0;i<prf_length2;i++)
   {
    for (j=0;j<=max_aa;j++)
      fprintf(stdout,"%d ", (pint)profile2[i+1][j]);
    fprintf(stdout,"%d ", (pint)profile2[i+1][gap_pos1]);
    fprintf(stdout,"%d ", (pint)profile2[i+1][gap_pos2]);
    fprintf(stdout,"%d %d\n",(pint)profile2[i+1][GAPCOL],(pint)profile2[i+1][LENCOL]);
   }
}

  aln_path1 = (char *) ckalloc( (max_aln_length+1) * sizeof(char) );
  aln_path2 = (char *) ckalloc( (max_aln_length+1) * sizeof(char) );


/*
   align the profiles
*/
/* use Myers and Miller to align two sequences */

  last_print = 0;
  print_ptr = 1;

  sb1 = sb2 = 0;
  se1 = prf_length1;
  se2 = prf_length2;

  HH = (lint *) ckalloc( (max_aln_length+1) * sizeof (lint) );
  DD = (lint *) ckalloc( (max_aln_length+1) * sizeof (lint) );
  RR = (lint *) ckalloc( (max_aln_length+1) * sizeof (lint) );
  SS = (lint *) ckalloc( (max_aln_length+1) * sizeof (lint) );
  gS = (lint *) ckalloc( (max_aln_length+1) * sizeof (lint) );
  displ = (sint *) ckalloc( (max_aln_length+1) * sizeof (sint) );

  score = pdiff(sb1, sb2, se1-sb1, se2-sb2, profile1[0][GAPCOL], profile1[prf_length1][GAPCOL]);

  HH=ckfree((void *)HH);
  DD=ckfree((void *)DD);
  RR=ckfree((void *)RR);
  SS=ckfree((void *)SS);
  gS=ckfree((void *)gS);

  ptracepath( &alignment_len);
  
  displ=ckfree((void *)displ);

  add_ggaps();

  for (i=0;i<prf_length1+2;i++)
     profile1[i]=ckfree((void *)profile1[i]);
  profile1=ckfree((void *)profile1);

  for (i=0;i<prf_length2+2;i++)
     profile2[i]=ckfree((void *)profile2[i]);
  profile2=ckfree((void *)profile2);

  prf_length1 = alignment_len;

  aln_path1=ckfree((void *)aln_path1);
  aln_path2=ckfree((void *)aln_path2);

  NumSeq = 0;
  for (j=0;j<nseqs;j++)
    {
       if (group[j+1]  == 1)
         {
            seqlen_array[j+1] = prf_length1;
	    realloc_seq(j+1,prf_length1);
            for (i=0;i<prf_length1;i++)
              seq_array[j+1][i+1] = alignment[NumSeq][i];
            NumSeq++;
         }
    }
  for (j=0;j<nseqs;j++)
    {
       if (group[j+1]  == 2)
         {
            seqlen_array[j+1] = prf_length1;
            seq_array[j+1] = (char *)realloc(seq_array[j+1], (prf_length1+2) * sizeof (char));
	    realloc_seq(j+1,prf_length1);
            for (i=0;i<prf_length1;i++)
              seq_array[j+1][i+1] = alignment[NumSeq][i];
            NumSeq++;
         }
    }

  for (i=0;i<nseqs1+nseqs2;i++)
     alignment[i]=ckfree((void *)alignment[i]);
  alignment=ckfree((void *)alignment);

  aln_len=ckfree((void *)aln_len);
  gaps=ckfree((void *)gaps);

  return(score/100);
}

static void add_ggaps(void)
{
   sint j;
   sint i,ix;
   sint len;
   char *ta;

   ta = (char *) ckalloc( (alignment_len+1) * sizeof (char) );

   for (j=0;j<nseqs1;j++)
     {
      ix = 0;
      for (i=0;i<alignment_len;i++)
        {
           if (aln_path1[i] == 2)
              {
                 if (ix < aln_len[j])
                    ta[i] = alignment[j][ix];
                 else 
                    ta[i] = ENDALN;
                 ix++;
              }
           else if (aln_path1[i] == 1)
              {
/*
   insertion in first alignment...
*/
                 ta[i] = gap_pos1;
              }
           else
              {
                 fprintf(stdout,"Error in aln_path\n");
              }
         }
       ta[i] = ENDALN;
       
       len = alignment_len;
       alignment[j] = (char *)realloc(alignment[j], (len+2) * sizeof (char));
       for (i=0;i<len;i++)
         alignment[j][i] = ta[i];
       alignment[j][len] = ENDALN;
       aln_len[j] = len;
      }

   for (j=nseqs1;j<nseqs1+nseqs2;j++)
     {
      ix = 0;
      for (i=0;i<alignment_len;i++)
        {
           if (aln_path2[i] == 2)
              {
                 if (ix < aln_len[j])
                    ta[i] = alignment[j][ix];
                 else 
                    ta[i] = ENDALN;
                 ix++;
              }
           else if (aln_path2[i] == 1)
              {
/*
   insertion in second alignment...
*/
                 ta[i] = gap_pos1;
              }
           else
              {
                 fprintf(stdout,"Error in aln_path\n");
              }
         }
       ta[i] = ENDALN;
       
       len = alignment_len;
       alignment[j] = (char *) realloc(alignment[j], (len+2) * sizeof (char) );
       for (i=0;i<len;i++)
         alignment[j][i] = ta[i];
       alignment[j][len] = ENDALN;
       aln_len[j] = len;
      }
      
   ta=ckfree((void *)ta);

   if (struct_penalties1 != NONE)
       gap_penalty_mask1 = add_ggaps_mask(gap_penalty_mask1,alignment_len,aln_path1,aln_path2);
   if (struct_penalties1 == SECST)
       sec_struct_mask1 = add_ggaps_mask(sec_struct_mask1,alignment_len,aln_path1,aln_path2);
     
   if (struct_penalties2 != NONE)
       gap_penalty_mask2 = add_ggaps_mask(gap_penalty_mask2,alignment_len,aln_path2,aln_path1);
   if (struct_penalties2 == SECST)
       sec_struct_mask2 = add_ggaps_mask(sec_struct_mask2,alignment_len,aln_path2,aln_path1);

if (debug>0)
{
  char c;
  extern char *amino_acid_codes;

   for (i=0;i<nseqs1+nseqs2;i++)
     {
      for (j=0;j<alignment_len;j++)
       {
        if (alignment[i][j] == ENDALN) break;
        else if ((alignment[i][j] == gap_pos1) || (alignment[i][j] == gap_pos2))  c = '-';
        else c = amino_acid_codes[(int)(alignment[i][j])];
        fprintf(stdout,"%c", c);
       }
      fprintf(stdout,"\n\n");
     }
}

}                  

static char * add_ggaps_mask(char *mask, int len, char *path1, char *path2)
{
   int i,ix;
   char *ta;

   ta = (char *) ckalloc( (len+1) * sizeof (char) );

       ix = 0;
       if (switch_profiles == FALSE)
        {     
         for (i=0;i<len;i++)
           {
             if (path1[i] == 2)
              {
                ta[i] = mask[ix];
                ix++;
              }
             else if (path1[i] == 1)
                ta[i] = gap_pos1;
           }
        }
       else
        {
         for (i=0;i<len;i++)
          {
            if (path2[i] == 2)
             {
               ta[i] = mask[ix];
               ix++;
             }
            else if (path2[i] == 1)
             ta[i] = gap_pos1;
           }
         }
       mask = (char *)realloc(mask,(len+2) * sizeof (char));
       for (i=0;i<len;i++)
         mask[i] = ta[i];
       mask[i] ='\0';
       
   ta=ckfree((void *)ta);

   return(mask);
}

static lint prfscore(sint n, sint m)
{
   sint    ix;
   lint  score;

   score = 0.0;
   for (ix=0; ix<=max_aa; ix++)
     {
         score += (profile1[n][ix] * profile2[m][ix]);
     }
   score += (profile1[n][gap_pos1] * profile2[m][gap_pos1]);
   score += (profile1[n][gap_pos2] * profile2[m][gap_pos2]);
   return(score/10);
   
}

static void ptracepath(sint *alen)
{
    sint i,j,k,pos,to_do;

    pos = 0;

    to_do=print_ptr-1;

    for(i=1;i<=to_do;++i) {
if (debug>1) fprintf(stdout,"%d ",(pint)displ[i]);
            if(displ[i]==0) {
                    aln_path1[pos]=2;
                    aln_path2[pos]=2;
                    ++pos;
            }
            else {
                    if((k=displ[i])>0) {
                            for(j=0;j<=k-1;++j) {
                                    aln_path2[pos+j]=2;
                                    aln_path1[pos+j]=1;
                            }
                            pos += k;
                    }
                    else {
                            k = (displ[i]<0) ? displ[i] * -1 : displ[i];
                            for(j=0;j<=k-1;++j) {
                                    aln_path1[pos+j]=2;
                                    aln_path2[pos+j]=1;
                            }
                            pos += k;
                    }
            }
    }
if (debug>1) fprintf(stdout,"\n");

   (*alen) = pos;

}

static void pdel(sint k)
{
        if(last_print<0)
                last_print = displ[print_ptr-1] -= k;
        else
                last_print = displ[print_ptr++] = -(k);
}

static void padd(sint k)
{

        if(last_print<0) {
                displ[print_ptr-1] = k;
                displ[print_ptr++] = last_print;
        }
        else
                last_print = displ[print_ptr++] = k;
}

static void palign(void)
{
        displ[print_ptr++] = last_print = 0;
}


static lint pdiff(sint A,sint B,sint M,sint N,sint go1, sint go2)
{
        sint midi,midj,type;
        lint midh;

        static lint t, tl, g, h;

{		static sint i,j;
        static lint hh, f, e, s;

/* Boundary cases: M <= 1 or N == 0 */
if (debug>2) fprintf(stdout,"A %d B %d M %d N %d midi %d go1 %d go2 %d\n", 
(pint)A,(pint)B,(pint)M,(pint)N,(pint)M/2,(pint)go1,(pint)go2);

/* if sequence B is empty....                                            */

        if(N<=0)  {

/* if sequence A is not empty....                                        */

                if(M>0) {

/* delete residues A[1] to A[M]                                          */

                        pdel(M);
                }
                return(-gap_penalty1(A,B,M));
        }

/* if sequence A is empty....                                            */

        if(M<=1) {
                if(M<=0) {

/* insert residues B[1] to B[N]                                          */

                        padd(N);
                        return(-gap_penalty2(A,B,N));
                }

/* if sequence A has just one residue....                                */

                if (go1 == 0)
                	midh =  -gap_penalty1(A+1,B+1,N);
                else
                	midh =  -gap_penalty2(A+1,B,1)-gap_penalty1(A+1,B+1,N);
                midj = 0;
                for(j=1;j<=N;j++) {
                        hh = -gap_penalty1(A,B+1,j-1) + prfscore(A+1,B+j)
                            -gap_penalty1(A+1,B+j+1,N-j);
                        if(hh>midh) {
                                midh = hh;
                                midj = j;
                        }
                }

                if(midj==0) {
                        padd(N);
                        pdel(1);
                }
                else {
                        if(midj>1) padd(midj-1);
                        palign();
                        if(midj<N) padd(N-midj);
                }
                return midh;
        }


/* Divide sequence A in half: midi */

        midi = M / 2;

/* In a forward phase, calculate all HH[j] and HH[j] */

        HH[0] = 0.0;
        t = -open_penalty1(A,B+1);
        tl = -ext_penalty1(A,B+1);
        for(j=1;j<=N;j++) {
                HH[j] = t = t+tl;
                DD[j] = t-open_penalty2(A+1,B+j);
        }

		if (go1 == 0) t = 0;
		else t = -open_penalty2(A+1,B);
        tl = -ext_penalty2(A+1,B);
        for(i=1;i<=midi;i++) {
                s = HH[0];
                HH[0] = hh = t = t+tl;
                f = t-open_penalty1(A+i,B+1);

                for(j=1;j<=N;j++) {
                	g = open_penalty1(A+i,B+j);
                	h = ext_penalty1(A+i,B+j);
                        if ((hh=hh-g-h) > (f=f-h)) f=hh;
                	g = open_penalty2(A+i,B+j);
                	h = ext_penalty2(A+i,B+j);
                        if ((hh=HH[j]-g-h) > (e=DD[j]-h)) e=hh;
                        hh = s + prfscore(A+i, B+j);
                        if (f>hh) hh = f;
                        if (e>hh) hh = e;

                        s = HH[j];
                        HH[j] = hh;
                        DD[j] = e;

                }
        }

        DD[0]=HH[0];

/* In a reverse phase, calculate all RR[j] and SS[j] */

        RR[N]=0.0;
        tl = 0.0;
        for(j=N-1;j>=0;j--) {
                g = -open_penalty1(A+M,B+j+1);
                tl -= ext_penalty1(A+M,B+j+1);
                RR[j] = g+tl;
                SS[j] = RR[j]-open_penalty2(A+M,B+j);
                gS[j] = open_penalty2(A+M,B+j);
        }

        tl = 0.0;
        for(i=M-1;i>=midi;i--) {
                s = RR[N];
                if (go2 == 0) g = 0;
                else g = -open_penalty2(A+i+1,B+N);
                tl -= ext_penalty2(A+i+1,B+N);
                RR[N] = hh = g+tl;
                t = open_penalty1(A+i,B+N);
                f = RR[N]-t;

                for(j=N-1;j>=0;j--) {
                	g = open_penalty1(A+i,B+j+1);
                	h = ext_penalty1(A+i,B+j+1);
                        if ((hh=hh-g-h) > (f=f-h-g+t)) f=hh;
                        t = g;
                	g = open_penalty2(A+i+1,B+j);
                	h = ext_penalty2(A+i+1,B+j);
                        hh=RR[j]-g-h;
                        if (i==(M-1)) {
				 e=SS[j]-h;
			}
                        else {
				e=SS[j]-h-g+open_penalty2(A+i+2,B+j);
				gS[j] = g;
			}
                        if (hh > e) e=hh;
                        hh = s + prfscore(A+i+1, B+j+1);
                        if (f>hh) hh = f;
                        if (e>hh) hh = e;

                        s = RR[j];
                        RR[j] = hh;
                        SS[j] = e;

                }
        }
        SS[N]=RR[N];
        gS[N] = open_penalty2(A+midi+1,B+N);

/* find midj, such that HH[j]+RR[j] or DD[j]+SS[j]+gap is the maximum */

        midh=HH[0]+RR[0];
        midj=0;
        type=1;
        for(j=0;j<=N;j++) {
                hh = HH[j] + RR[j];
                if(hh>=midh)
                        if(hh>midh || (HH[j]!=DD[j] && RR[j]==SS[j])) {
                                midh=hh;
                                midj=j;
                        }
        }

        for(j=N;j>=0;j--) {
                hh = DD[j] + SS[j] + gS[j];
                if(hh>midh) {
                        midh=hh;
                        midj=j;
                        type=2;
                }
        }
}

/* Conquer recursively around midpoint                                   */


        if(type==1) {             /* Type 1 gaps  */
if (debug>2) fprintf(stdout,"Type 1,1: midj %d\n",(pint)midj);
                pdiff(A,B,midi,midj,go1,1);
if (debug>2) fprintf(stdout,"Type 1,2: midj %d\n",(pint)midj);
                pdiff(A+midi,B+midj,M-midi,N-midj,1,go2);
        }
        else {
if (debug>2) fprintf(stdout,"Type 2,1: midj %d\n",(pint)midj);
                pdiff(A,B,midi-1,midj,go1, 0);
                pdel(2);
if (debug>2) fprintf(stdout,"Type 2,2: midj %d\n",(pint)midj);
                pdiff(A+midi+1,B+midj,M-midi-1,N-midj,0,go2);
        }

        return midh;       /* Return the score of the best alignment */
}

/* calculate the score for opening a gap at residues A[i] and B[j]       */

static sint open_penalty1(sint i, sint j)
{
   sint g;

   if (!endgappenalties &&(i==0 || i==prf_length1)) return(0);

   g = profile2[j][GAPCOL] + profile1[i][GAPCOL];
   return(g);
}

/* calculate the score for extending an existing gap at A[i] and B[j]    */

static sint ext_penalty1(sint i, sint j)
{
   sint h;

   if (!endgappenalties &&(i==0 || i==prf_length1)) return(0);

   h = profile2[j][LENCOL];
   return(h);
}

/* calculate the score for a gap of length k, at residues A[i] and B[j]  */

static sint gap_penalty1(sint i, sint j, sint k)
{
   sint ix;
   sint gp;
   sint g, h = 0;

   if (k <= 0) return(0);
   if (!endgappenalties &&(i==0 || i==prf_length1)) return(0);

   g = profile2[j][GAPCOL] + profile1[i][GAPCOL];
   for (ix=0;ix<k && ix+j<prf_length2;ix++)
      h += profile2[ix+j][LENCOL];

   gp = g + h;
   return(gp);
}
/* calculate the score for opening a gap at residues A[i] and B[j]       */

static sint open_penalty2(sint i, sint j)
{
   sint g;

   if (!endgappenalties &&(j==0 || j==prf_length2)) return(0);

   g = profile1[i][GAPCOL] + profile2[j][GAPCOL];
   return(g);
}

/* calculate the score for extending an existing gap at A[i] and B[j]    */

static sint ext_penalty2(sint i, sint j)
{
   sint h;

   if (!endgappenalties &&(j==0 || j==prf_length2)) return(0);

   h = profile1[i][LENCOL];
   return(h);
}

/* calculate the score for a gap of length k, at residues A[i] and B[j]  */

static sint gap_penalty2(sint i, sint j, sint k)
{
   sint ix;
   sint gp;
   sint g, h = 0;

   if (k <= 0) return(0);
   if (!endgappenalties &&(j==0 || j==prf_length2)) return(0);

   g = profile1[i][GAPCOL] + profile2[j][GAPCOL];
   for (ix=0;ix<k && ix+i<prf_length1;ix++)
      h += profile1[ix+i][LENCOL];

   gp = g + h;
   return(gp);
}
