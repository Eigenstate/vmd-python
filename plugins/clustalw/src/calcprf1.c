#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "clustalw.h"


/*
 *   Prototypes
 */

/*
 *   Global variables
 */

extern sint max_aa,gap_pos1,gap_pos2;

void calc_prf1(sint **profile, char **alignment, sint *gaps,
  sint matrix[NUMRES][NUMRES],
  sint *seq_weight, sint prf_length, sint first_seq, sint last_seq)
{

  sint **weighting, sum2, d, i, res; 
  sint numseq;
  sint r, pos;
  int f;
  float scale;

  weighting = (sint **) ckalloc( (NUMRES+2) * sizeof (sint *) );
  for (i=0;i<NUMRES+2;i++)
    weighting[i] = (sint *) ckalloc( (prf_length+2) * sizeof (sint) );

  numseq = last_seq-first_seq;

  sum2 = 0;
  for (i=first_seq; i<last_seq; i++)
       sum2 += seq_weight[i];

  for (r=0; r<prf_length; r++)
   {
      for (d=0; d<=max_aa; d++)
        {
            weighting[d][r] = 0;
            for (i=first_seq; i<last_seq; i++)
               if (d == alignment[i][r]) weighting[d][r] += seq_weight[i];
        }
      weighting[gap_pos1][r] = 0;
      for (i=first_seq; i<last_seq; i++)
         if (gap_pos1 == alignment[i][r]) weighting[gap_pos1][r] += seq_weight[i];
      weighting[gap_pos2][r] = 0;
      for (i=first_seq; i<last_seq; i++)
         if (gap_pos2 == alignment[i][r]) weighting[gap_pos2][r] += seq_weight[i];
   }

  for (pos=0; pos< prf_length; pos++)
    {
      if (gaps[pos] == numseq)
        {
           for (res=0; res<=max_aa; res++)
             {
                profile[pos+1][res] = matrix[res][gap_pos1];
             }
           profile[pos+1][gap_pos1] = matrix[gap_pos1][gap_pos1];
           profile[pos+1][gap_pos2] = matrix[gap_pos2][gap_pos1];
        }
      else
        {
           scale = (float)(numseq-gaps[pos]) / (float)numseq;
           for (res=0; res<=max_aa; res++)
             {
                f = 0;
                for (d=0; d<=max_aa; d++)
                     f += (weighting[d][pos] * matrix[d][res]);
                f += (weighting[gap_pos1][pos] * matrix[gap_pos1][res]);
                f += (weighting[gap_pos2][pos] * matrix[gap_pos2][res]);
                profile[pos+1][res] = (sint  )(((float)f / (float)sum2)*scale);
             }
           f = 0;
           for (d=0; d<=max_aa; d++)
                f += (weighting[d][pos] * matrix[d][gap_pos1]);
           f += (weighting[gap_pos1][pos] * matrix[gap_pos1][gap_pos1]);
           f += (weighting[gap_pos2][pos] * matrix[gap_pos2][gap_pos1]);
           profile[pos+1][gap_pos1] = (sint )(((float)f / (float)sum2)*scale);
           f = 0;
           for (d=0; d<=max_aa; d++)
                f += (weighting[d][pos] * matrix[d][gap_pos2]);
           f += (weighting[gap_pos1][pos] * matrix[gap_pos1][gap_pos2]);
           f += (weighting[gap_pos2][pos] * matrix[gap_pos2][gap_pos2]);
           profile[pos+1][gap_pos2] = (sint )(((float)f / (float)sum2)*scale);
        }
    }

  for (i=0;i<NUMRES+2;i++)
    weighting[i]=ckfree((void *)weighting[i]);
  weighting=ckfree((void *)weighting);

}


