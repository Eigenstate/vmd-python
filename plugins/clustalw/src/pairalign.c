/* Change int h to int gh everywhere  DES June 1994 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clustalw.h"

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

#define gap(k)  ((k) <= 0 ? 0 : g + gh * (k))
#define tbgap(k)  ((k) <= 0 ? 0 : tb + gh * (k))
#define tegap(k)  ((k) <= 0 ? 0 : te + gh * (k))

/*
 *	Prototypes
 */
static void add(sint v);
static sint calc_score(sint iat, sint jat, sint v1, sint v2);
static float tracepath(sint tsb1,sint tsb2);
static void forward_pass(char *ia, char *ib, sint n, sint m);
static void reverse_pass(char *ia, char *ib);
static sint diff(sint A, sint B, sint M, sint N, sint tb, sint te);
static void del(sint k);

/*
 *   Global variables
 */
#ifdef MAC
#define pwint   short
#else
#define pwint   int
#endif
static sint		int_scale;

extern double   **tmat;
extern float    pw_go_penalty;
extern float    pw_ge_penalty;
extern float	transition_weight;
extern sint 	nseqs;
extern sint 	max_aa;
extern sint 	gap_pos1,gap_pos2;
extern sint  	max_aln_length;
extern sint 	*seqlen_array;
extern sint 	debug;
extern sint  	mat_avscore;
extern short 	blosum30mt[],pam350mt[],idmat[],pw_usermat[],pw_userdnamat[];
extern short    clustalvdnamt[],swgapdnamt[];
extern short    gon250mt[];
extern short 	def_dna_xref[],def_aa_xref[],pw_dna_xref[],pw_aa_xref[];
extern Boolean  dnaflag;
extern char 	**seq_array;
extern char 	*amino_acid_codes;
extern char 	pw_mtrxname[];
extern char 	pw_dnamtrxname[];

static float 	mm_score;
static sint 	print_ptr,last_print;
static sint 	*displ;
static pwint 	*HH, *DD, *RR, *SS;
static sint 	g, gh;
static sint   	seq1, seq2;
static sint     matrix[NUMRES][NUMRES];
static pwint    maxscore;
static sint    	sb1, sb2, se1, se2;


sint pairalign(sint istart, sint iend, sint jstart, sint jend)
{
  short	 *mat_xref;
  static sint    si, sj, i;
  static sint    n,m,len1,len2;
  static sint    maxres;
  static short    *matptr;
  static char   c;
  static float gscale,ghscale;

  displ = (sint *)ckalloc((2*max_aln_length+1) * sizeof(sint));
  HH = (pwint *)ckalloc((max_aln_length) * sizeof(pwint));
  DD = (pwint *)ckalloc((max_aln_length) * sizeof(pwint));
  RR = (pwint *)ckalloc((max_aln_length) * sizeof(pwint));
  SS = (pwint *)ckalloc((max_aln_length) * sizeof(pwint));
		
#ifdef MAC
  int_scale = 10;
#else
  int_scale = 100;
#endif
  gscale=ghscale=1.0;
  if (dnaflag)
    {
      if (debug>1) fprintf(stdout,"matrix %s\n",pw_dnamtrxname);
      if (strcmp(pw_dnamtrxname, "iub") == 0)
	{ 
	  matptr = swgapdnamt;
	  mat_xref = def_dna_xref;
	}
      else if (strcmp(pw_dnamtrxname, "clustalw") == 0)
	{ 
	  matptr = clustalvdnamt;
	  mat_xref = def_dna_xref;
	  gscale=0.6667;
	  ghscale=0.751;
	}
      else
	{
	  matptr = pw_userdnamat;
	  mat_xref = pw_dna_xref;
	}
      maxres = get_matrix(matptr, mat_xref, matrix, TRUE, int_scale);
      if (maxres == 0) return((sint)-1);

      matrix[0][4]=transition_weight*matrix[0][0];
      matrix[4][0]=transition_weight*matrix[0][0];
      matrix[2][11]=transition_weight*matrix[0][0];
      matrix[11][2]=transition_weight*matrix[0][0];
      matrix[2][12]=transition_weight*matrix[0][0];
      matrix[12][2]=transition_weight*matrix[0][0];
    }
  else
    {
      if (debug>1) fprintf(stdout,"matrix %s\n",pw_mtrxname);
      if (strcmp(pw_mtrxname, "blosum") == 0)
	{
	  matptr = blosum30mt;
	  mat_xref = def_aa_xref;
	}
      else if (strcmp(pw_mtrxname, "pam") == 0)
	{
	  matptr = pam350mt;
	  mat_xref = def_aa_xref;
	}
      else if (strcmp(pw_mtrxname, "gonnet") == 0)
	{
	  matptr = gon250mt;
	  int_scale /= 10;
	  mat_xref = def_aa_xref;
	}
      else if (strcmp(pw_mtrxname, "id") == 0)
	{
	  matptr = idmat;
	  mat_xref = def_aa_xref;
	}
      else
	{
	  matptr = pw_usermat;
	  mat_xref = pw_aa_xref;
	}

      maxres = get_matrix(matptr, mat_xref, matrix, TRUE, int_scale);
      if (maxres == 0) return((sint)-1);
    }


  for (si=MAX(0,istart);si<nseqs && si<iend;si++)
    {
      n = seqlen_array[si+1];
      len1 = 0;
      for (i=1;i<=n;i++) {
	c = seq_array[si+1][i];
	if ((c!=gap_pos1) && (c != gap_pos2)) len1++;
      }

      for (sj=MAX(si+1,jstart+1);sj<nseqs && sj<jend;sj++)
	{
	  m = seqlen_array[sj+1];
	  if(n==0 || m==0) {
	    tmat[si+1][sj+1]=1.0;
	    tmat[sj+1][si+1]=1.0;
	    continue;
	  }
	  len2 = 0;
	  for (i=1;i<=m;i++) {
	    c = seq_array[sj+1][i];
	    if ((c!=gap_pos1) && (c != gap_pos2)) len2++;
	  }

	  if (dnaflag) {
	    g = 2 * (float)pw_go_penalty * int_scale*gscale;
	    gh = pw_ge_penalty * int_scale*ghscale;
	  }
	  else {
	    if (mat_avscore <= 0)
              g = 2 * (float)(pw_go_penalty + log((double)(MIN(n,m))))*int_scale;
	    else
              g = 2 * mat_avscore * (float)(pw_go_penalty +
					    log((double)(MIN(n,m))))*gscale;
	    gh = pw_ge_penalty * int_scale;
	  }

	  if (debug>1) fprintf(stdout,"go %d ge %d\n",(pint)g,(pint)gh);

	  /*
	    align the sequences
	  */
	  seq1 = si+1;
        seq2 = sj+1;

        forward_pass(&seq_array[seq1][0], &seq_array[seq2][0],
           n, m);

        reverse_pass(&seq_array[seq1][0], &seq_array[seq2][0]);

        last_print = 0;
	print_ptr = 1;
/*
        sb1 = sb2 = 1;
        se1 = n-1;
        se2 = m-1;
*/

/* use Myers and Miller to align two sequences */

        maxscore = diff(sb1-1, sb2-1, se1-sb1+1, se2-sb2+1, 
        (sint)0, (sint)0);
 
/* calculate percentage residue identity */

        mm_score = tracepath(sb1,sb2);

		if(len1==0 || len2==0) mm_score=0;
		else
			mm_score /= (float)MIN(len1,len2);

        tmat[si+1][sj+1] = ((float)100.0 - mm_score)/(float)100.0;
        tmat[sj+1][si+1] = ((float)100.0 - mm_score)/(float)100.0;

if (debug>1)
{
        fprintf(stdout,"Sequences (%d:%d) Aligned. Score: %d CompScore:  %d\n",
                           (pint)si+1,(pint)sj+1, 
                           (pint)mm_score, 
                           (pint)maxscore/(MIN(len1,len2)*100));
}
else
{
        info("Sequences (%d:%d) Aligned. Score:  %d",
                                      (pint)si+1,(pint)sj+1, 
                                      (pint)mm_score);
}

   }
  }
   displ=ckfree((void *)displ);
   HH=ckfree((void *)HH);
   DD=ckfree((void *)DD);
   RR=ckfree((void *)RR);
   SS=ckfree((void *)SS);


  return((sint)1);
}

static void add(sint v)
{

        if(last_print<0) {
                displ[print_ptr-1] = v;
                displ[print_ptr++] = last_print;
        }
        else
                last_print = displ[print_ptr++] = v;
}

static sint calc_score(sint iat,sint jat,sint v1,sint v2)
{
        sint ipos,jpos;
		sint ret;

        ipos = v1 + iat;
        jpos = v2 + jat;

        ret=matrix[(int)seq_array[seq1][ipos]][(int)seq_array[seq2][jpos]];

	return(ret);
}


static float tracepath(sint tsb1,sint tsb2)
{
	char c1,c2;
    sint  i1,i2,r;
    sint i,k,pos,to_do;
	sint count;
	float score;
	char s1[600], s2[600];

        to_do=print_ptr-1;
        i1 = tsb1;
        i2 = tsb2;

	pos = 0;
	count = 0;
        for(i=1;i<=to_do;++i) {

	  if (debug>1) fprintf(stdout,"%d ",(pint)displ[i]);
	  if(displ[i]==0) {
	    c1 = seq_array[seq1][i1];
	    c2 = seq_array[seq2][i2];
	    
	    if (debug>0)
	      {
		if (c1>max_aa) s1[pos] = '-';
		else s1[pos]=amino_acid_codes[(int)c1];
		if (c2>max_aa) s2[pos] = '-';
		else s2[pos]=amino_acid_codes[(int)c2];
	      }
	    
	    if ((c1!=gap_pos1) && (c1 != gap_pos2) &&
		(c1 == c2)) count++;
	    ++i1;
	    ++i2;
	    ++pos;
	  }
	  else {
	    if((k=displ[i])>0) {
	      
	      if (debug>0)
		for (r=0;r<k;r++)
		  {
		    s1[pos+r]='-';
		    if (seq_array[seq2][i2+r]>max_aa) s2[pos+r] = '-';
		    else s2[pos+r]=amino_acid_codes[(int)(seq_array[seq2][i2+r])];
		  }
	      
	      i2 += k;
	      pos += k;
	    }
	    else {
	      
	      if (debug>0)
		for (r=0;r<(-k);r++)
		  {
		    s2[pos+r]='-';
		    if (seq_array[seq1][i1+r]>max_aa) s1[pos+r] = '-';
		    else s1[pos+r]=amino_acid_codes[(int)(seq_array[seq1][i1+r])];
		  }
	      
	      i1 -= k;
	      pos -= k;
	    }
	  }
        }
	if (debug>0) fprintf(stdout,"\n");
	if (debug>0) 
	  {
	    for (i=0;i<pos;i++) fprintf(stdout,"%c",s1[i]);
	    fprintf(stdout,"\n");
	    for (i=0;i<pos;i++) fprintf(stdout,"%c",s2[i]);
	    fprintf(stdout,"\n");
	  }
	/*
	  if (count <= 0) count = 1;
	*/
	score = 100.0 * (float)count;
	return(score);
}


static void forward_pass(char *ia, char *ib, sint n, sint m)
{

  sint i,j;
  pwint f,hh,p,t;

  maxscore = 0;
  se1 = se2 = 0;
  for (i=0;i<=m;i++)
    {
       HH[i] = 0;
       DD[i] = -g;
    }

  for (i=1;i<=n;i++)
     {
        hh = p = 0;
		f = -g;

        for (j=1;j<=m;j++)
           {

              f -= gh; 
              t = hh - g - gh;
              if (f<t) f = t;

              DD[j] -= gh;
              t = HH[j] - g - gh;
              if (DD[j]<t) DD[j] = t;

              hh = p + matrix[(int)ia[i]][(int)ib[j]];
              if (hh<f) hh = f;
              if (hh<DD[j]) hh = DD[j];
              if (hh<0) hh = 0;

              p = HH[j];
              HH[j] = hh;

              if (hh > maxscore)
                {
                   maxscore = hh;
                   se1 = i;
                   se2 = j;
                }
           }
     }

}


static void reverse_pass(char *ia, char *ib)
{

  sint i,j;
  pwint f,hh,p,t;
  pwint cost;

  cost = 0;
  sb1 = sb2 = 1;
  for (i=se2;i>0;i--)
    {
       HH[i] = -1;
       DD[i] = -1;
    }

  for (i=se1;i>0;i--)
     {
        hh = f = -1;
        if (i == se1) p = 0;
        else p = -1;

        for (j=se2;j>0;j--)
           {

              f -= gh; 
              t = hh - g - gh;
              if (f<t) f = t;

              DD[j] -= gh;
              t = HH[j] - g - gh;
              if (DD[j]<t) DD[j] = t;

              hh = p + matrix[(int)ia[i]][(int)ib[j]];
              if (hh<f) hh = f;
              if (hh<DD[j]) hh = DD[j];

              p = HH[j];
              HH[j] = hh;

              if (hh > cost)
                {
                   cost = hh;
                   sb1 = i;
                   sb2 = j;
                   if (cost >= maxscore) break;
                }
           }
        if (cost >= maxscore) break;
     }

}

static int diff(sint A,sint B,sint M,sint N,sint tb,sint te)
{
  sint type;
  sint midi,midj,i,j;
  int midh;
  static pwint f, hh, e, s, t;
  
  if(N<=0)  {
    if(M>0) {
      del(M);
    }
    
    return(-(int)tbgap(M));
  }
  
  if(M<=1) {
    if(M<=0) {
      add(N);
      return(-(int)tbgap(N));
    }
    
    midh = -(tb+gh) - tegap(N);
    hh = -(te+gh) - tbgap(N);
    if (hh>midh) midh = hh;
    midj = 0;
    for(j=1;j<=N;j++) {
      hh = calc_score(1,j,A,B)
	- tegap(N-j) - tbgap(j-1);
      if(hh>midh) {
	midh = hh;
	midj = j;
      }
    }
    
    if(midj==0) {
      del(1);
      add(N);
    }
    else {
      if(midj>1)
	add(midj-1);
      displ[print_ptr++] = last_print = 0;
      if(midj<N)
	add(N-midj);
    }
    return midh;
  }
  
/* Divide: Find optimum midpoint (midi,midj) of cost midh */
  
  midi = M / 2;
  HH[0] = 0.0;
  t = -tb;
  for(j=1;j<=N;j++) {
    HH[j] = t = t-gh;
    DD[j] = t-g;
  }
  
  t = -tb;
  for(i=1;i<=midi;i++) {
    s=HH[0];
    HH[0] = hh = t = t-gh;
    f = t-g;
    for(j=1;j<=N;j++) {
      if ((hh=hh-g-gh) > (f=f-gh)) f=hh;
      if ((hh=HH[j]-g-gh) > (e=DD[j]-gh)) e=hh;
      hh = s + calc_score(i,j,A,B);
      if (f>hh) hh = f;
      if (e>hh) hh = e;
      
      s = HH[j];
      HH[j] = hh;
      DD[j] = e;
    }
  }
  
  DD[0]=HH[0];
  
  RR[N]=0;
  t = -te;
  for(j=N-1;j>=0;j--) {
    RR[j] = t = t-gh;
    SS[j] = t-g;
  }
  
  t = -te;
  for(i=M-1;i>=midi;i--) {
    s = RR[N];
    RR[N] = hh = t = t-gh;
    f = t-g;
    
    for(j=N-1;j>=0;j--) {
      
      if ((hh=hh-g-gh) > (f=f-gh)) f=hh;
      if ((hh=RR[j]-g-gh) > (e=SS[j]-gh)) e=hh;
      hh = s + calc_score(i+1,j+1,A,B);
      if (f>hh) hh = f;
      if (e>hh) hh = e;
      
      s = RR[j];
      RR[j] = hh;
      SS[j] = e;
      
    }
  }
  
  SS[N]=RR[N];
  
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
    hh = DD[j] + SS[j] + g;
    if(hh>midh) {
      midh=hh;
      midj=j;
      type=2;
    }
  }
  
  /* Conquer recursively around midpoint  */
  
  
  if(type==1) {             /* Type 1 gaps  */
    diff(A,B,midi,midj,tb,g);
    diff(A+midi,B+midj,M-midi,N-midj,g,te);
  }
  else {
    diff(A,B,midi-1,midj,tb,0.0);
    del(2);
    diff(A+midi+1,B+midj,M-midi-1,N-midj,0.0,te);
  }
  
  return midh;       /* Return the score of the best alignment */
}

static void del(sint k)
{
  if(last_print<0)
    last_print = displ[print_ptr-1] -= k;
  else
    last_print = displ[print_ptr++] = -(k);
}


