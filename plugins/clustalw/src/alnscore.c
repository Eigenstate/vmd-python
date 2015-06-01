#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clustalw.h"

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

/*
 *       Prototypes
 */

static sint  count_gaps(sint s1, sint s2, sint l);

/*
 *       Global Variables
 */

extern float gap_open;
extern sint   nseqs;
extern sint   *seqlen_array;
extern short   blosum45mt[];
extern short   def_aa_xref[];
extern sint   debug;
extern sint   max_aa;
extern char  **seq_array;


void aln_score(void)
{
  static short  *mat_xref, *matptr;
  static sint maxres;
  static sint  s1,s2,c1,c2;
  static sint    ngaps;
  static sint    i,l1,l2;
  static lint    score;
  static sint   matrix[NUMRES][NUMRES];

/* calculate an overall score for the alignment by summing the
scores for each pairwise alignment */

  matptr = blosum45mt;
  mat_xref = def_aa_xref;
  maxres = get_matrix(matptr, mat_xref, matrix, TRUE, 100);
  if (maxres == 0)
    {
       fprintf(stdout,"Error: matrix blosum30 not found\n");
       return;
    }

  score=0;
  for (s1=1;s1<=nseqs;s1++)
   {
    for (s2=1;s2<s1;s2++)
      {

        l1 = seqlen_array[s1];
        l2 = seqlen_array[s2];
        for (i=1;i<l1 && i<l2;i++)
          {
            c1 = seq_array[s1][i];
            c2 = seq_array[s2][i];
            if ((c1>=0) && (c1<=max_aa) && (c2>=0) && (c2<=max_aa))
                score += matrix[c1][c2];
          }

        ngaps = count_gaps(s1, s2, l1);

        score -= 100 * gap_open * ngaps;

      }
   }

  score /= 100;

  info("Alignment Score %d", (pint)score);

}

static sint count_gaps(sint s1, sint s2, sint l)
{
    sint i, g;
    sint q, r, *Q, *R;


    Q = (sint *)ckalloc((l+2) * sizeof(sint));
    R = (sint *)ckalloc((l+2) * sizeof(sint));

    Q[0] = R[0] = g = 0;

    for (i=1;i<l;i++)
      {
         if (seq_array[s1][i] > max_aa) q = 1;
         else q = 0;
         if (seq_array[s2][i] > max_aa) r = 1;
         else r = 0;

         if (((Q[i-1] <= R[i-1]) && (q != 0) && (1-r != 0)) ||
             ((Q[i-1] >= R[i-1]) && (1-q != 0) && (r != 0)))
             g += 1;
         if (q != 0) Q[i] = Q[i-1]+1;
         else Q[i] = 0;

         if (r != 0) R[i] = R[i-1]+1;
         else R[i] = 0;
     }
     
   Q=ckfree((void *)Q);
   R=ckfree((void *)R);

   return(g);
}
          

