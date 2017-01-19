#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "clustalw.h"


/*
 *       Prototypes
 */

/*
 *       Global Variables
 */

extern double  **tmat;
extern Boolean no_weights;
extern sint     debug;
extern sint     max_aa;
extern sint     nseqs;
extern sint     profile1_nseqs;
extern sint     nsets;
extern sint     **sets;
extern sint     divergence_cutoff;
extern sint     *seq_weight;
extern sint     output_order, *output_index;
extern Boolean distance_tree;
extern char    seqname[];
extern sint     *seqlen_array;
extern char    **seq_array;

sint malign(sint istart,char *phylip_name) /* full progressive alignment*/
{
   static 	sint *aligned;
   static 	sint *group;
   static 	sint ix;

   sint 	*maxid, max, sum;
   sint		*tree_weight;
   sint 		i,j,set,iseq=0;
   sint 		status,entries;
   lint		score = 0;


   info("Start of Multiple Alignment");

/* get the phylogenetic tree from *.ph */

   if (nseqs >= 2) 
     {
       status = read_tree(phylip_name, (sint)0, nseqs);
       if (status == 0) return((sint)0);
     }

/* calculate sequence weights according to branch lengths of the tree -
   weights in global variable seq_weight normalised to sum to 100 */

   calc_seq_weights((sint)0, nseqs, seq_weight);

/* recalculate tmat matrix as percent similarity matrix */

   status = calc_similarities(nseqs);
   if (status == 0) return((sint)0);

/* for each sequence, find the most closely related sequence */

   maxid = (sint *)ckalloc( (nseqs+1) * sizeof (sint));
   for (i=1;i<=nseqs;i++)
     {
         maxid[i] = -1;
         for (j=1;j<=nseqs;j++) 
           if (j!=i && maxid[i] < tmat[i][j]) maxid[i] = tmat[i][j];
     }

/* group the sequences according to their relative divergence */

   if (istart == 0)
     {
        sets = (sint **) ckalloc( (nseqs+1) * sizeof (sint *) );
        for(i=0;i<=nseqs;i++)
           sets[i] = (sint *)ckalloc( (nseqs+1) * sizeof (sint) );

        create_sets((sint)0,nseqs);
        info("There are %d groups",(pint)nsets);

/* clear the memory used for the phylogenetic tree */

        if (nseqs >= 2)
             clear_tree(NULL);

/* start the multiple alignments.........  */

        info("Aligning...");

/* first pass, align closely related sequences first.... */

        ix = 0;
        aligned = (sint *)ckalloc( (nseqs+1) * sizeof (sint) );
        for (i=0;i<=nseqs;i++) aligned[i] = 0;

        for(set=1;set<=nsets;++set)
         {
          entries=0;
          for (i=1;i<=nseqs;i++)
            {
               if ((sets[set][i] != 0) && (maxid[i] > divergence_cutoff))
                 {
                    entries++;
                    if  (aligned[i] == 0)
                       {
                          if (output_order==INPUT)
                            {
                              ++ix;
                              output_index[i] = i;
                            }
                          else output_index[++ix] = i;
                          aligned[i] = 1;
                       }
                 }
            }

          if(entries > 0) score = prfalign(sets[set], aligned);
          else score=0.0;


/* negative score means fatal error... exit now!  */

          if (score < 0) 
             {
                return(-1);
             }
          if ((entries > 0) && (score > 0))
             info("Group %d: Sequences:%4d      Score:%d",
             (pint)set,(pint)entries,(pint)score);
          else
             info("Group %d:                     Delayed",
             (pint)set);
        }

        for (i=0;i<=nseqs;i++)
          sets[i]=ckfree((void *)sets[i]);
        sets=ckfree(sets);
     }
   else
     {
/* clear the memory used for the phylogenetic tree */

        if (nseqs >= 2)
             clear_tree(NULL);

        aligned = (sint *)ckalloc( (nseqs+1) * sizeof (sint) );
        ix = 0;
        for (i=1;i<=istart+1;i++)
         {
           aligned[i] = 1;
           ++ix;
           output_index[i] = i;
         }
        for (i=istart+2;i<=nseqs;i++) aligned[i] = 0;
     }

/* second pass - align remaining, more divergent sequences..... */

/* if not all sequences were aligned, for each unaligned sequence,
   find it's closest pair amongst the aligned sequences.  */

    group = (sint *)ckalloc( (nseqs+1) * sizeof (sint));
    tree_weight = (sint *) ckalloc( (nseqs) * sizeof(sint) );
    for (i=0;i<nseqs;i++)
   		tree_weight[i] = seq_weight[i];

/* if we haven't aligned any sequences, in the first pass - align the
two most closely related sequences now */
   if(ix==0)
     {
        max = -1;
	iseq = 0;
        for (i=1;i<=nseqs;i++)
          {
             for (j=i+1;j<=nseqs;j++)
               {
                  if (max < tmat[i][j])
		  {
                     max = tmat[i][j];
                     iseq = i;
                  }
              }
          }
        aligned[iseq]=1;
        if (output_order == INPUT)
          {
            ++ix;
            output_index[iseq] = iseq;
          }
         else
            output_index[++ix] = iseq;
     }

    while (ix < nseqs)
      {
             for (i=1;i<=nseqs;i++) {
                if (aligned[i] == 0)
                  {
                     maxid[i] = -1;
                     for (j=1;j<=nseqs;j++) 
                        if ((maxid[i] < tmat[i][j]) && (aligned[j] != 0))
                            maxid[i] = tmat[i][j];
                  }
              }
/* find the most closely related sequence to those already aligned */

            max = -1;
	    iseq = 0;
            for (i=1;i<=nseqs;i++)
              {
                if ((aligned[i] == 0) && (maxid[i] > max))
                  {
                     max = maxid[i];
                     iseq = i;
                  }
              }


/* align this sequence to the existing alignment */
/* weight sequences with percent identity with profile*/
/* OR...., multiply sequence weights from tree by percent identity with new sequence */
   if(no_weights==FALSE) {
   for (j=0;j<nseqs;j++)
       if (aligned[j+1] != 0)
              seq_weight[j] = tree_weight[j] * tmat[j+1][iseq];
/*
  Normalise the weights, such that the sum of the weights = INT_SCALE_FACTOR
*/

         sum = 0;
         for (j=0;j<nseqs;j++)
           if (aligned[j+1] != 0)
            sum += seq_weight[j];
         if (sum == 0)
          {
           for (j=0;j<nseqs;j++)
                seq_weight[j] = 1;
                sum = j;
          }
         for (j=0;j<nseqs;j++)
           if (aligned[j+1] != 0)
             {
               seq_weight[j] = (seq_weight[j] * INT_SCALE_FACTOR) / sum;
               if (seq_weight[j] < 1) seq_weight[j] = 1;
             }
	}

         entries = 0;
         for (j=1;j<=nseqs;j++)
           if (aligned[j] != 0)
              {
                 group[j] = 1;
                 entries++;
              }
           else if (iseq==j)
              {
                 group[j] = 2;
                 entries++;
              }
         aligned[iseq] = 1;

         score = prfalign(group, aligned);
         info("Sequence:%d     Score:%d",(pint)iseq,(pint)score);
         if (output_order == INPUT)
          {
            ++ix;
            output_index[iseq] = iseq;
          }
         else
            output_index[++ix] = iseq;
      }

   group=ckfree((void *)group);
   aligned=ckfree((void *)aligned);
   maxid=ckfree((void *)maxid);
   tree_weight=ckfree((void *)tree_weight);

   aln_score();

/* make the rest (output stuff) into routine clustal_out in file amenu.c */

   return(nseqs);

}

sint seqalign(sint istart,char *phylip_name) /* sequence alignment to existing profile */
{
   static 	sint *aligned, *tree_weight;
   static 	sint *group;
   static 	sint ix;

   sint 	*maxid, max;
   sint 		i,j,status,iseq=0;
   sint 		sum,entries;
   lint		score = 0;


   info("Start of Multiple Alignment");

/* get the phylogenetic tree from *.ph */

   if (nseqs >= 2) 
     {
       status = read_tree(phylip_name, (sint)0, nseqs);
       if (status == 0) return(0);
     }

/* calculate sequence weights according to branch lengths of the tree -
   weights in global variable seq_weight normalised to sum to 100 */

   calc_seq_weights((sint)0, nseqs, seq_weight);
   
   tree_weight = (sint *) ckalloc( (nseqs) * sizeof(sint) );
   for (i=0;i<nseqs;i++)
   		tree_weight[i] = seq_weight[i];

/* recalculate tmat matrix as percent similarity matrix */

   status = calc_similarities(nseqs);
   if (status == 0) return((sint)0);

/* for each sequence, find the most closely related sequence */

   maxid = (sint *)ckalloc( (nseqs+1) * sizeof (sint));
   for (i=1;i<=nseqs;i++)
     {
         maxid[i] = -1;
         for (j=1;j<=nseqs;j++) 
           if (maxid[i] < tmat[i][j]) maxid[i] = tmat[i][j];
     }

/* clear the memory used for the phylogenetic tree */

        if (nseqs >= 2)
             clear_tree(NULL);

        aligned = (sint *)ckalloc( (nseqs+1) * sizeof (sint) );
        ix = 0;
        for (i=1;i<=istart+1;i++)
         {
           aligned[i] = 1;
           ++ix;
           output_index[i] = i;
         }
        for (i=istart+2;i<=nseqs;i++) aligned[i] = 0;

/* for each unaligned sequence, find it's closest pair amongst the
   aligned sequences.  */

    group = (sint *)ckalloc( (nseqs+1) * sizeof (sint));

    while (ix < nseqs)
      {
        if (ix > 0) 
          {
             for (i=1;i<=nseqs;i++) {
                if (aligned[i] == 0)
                  {
                     maxid[i] = -1;
                     for (j=1;j<=nseqs;j++) 
                        if ((maxid[i] < tmat[i][j]) && (aligned[j] != 0))
                            maxid[i] = tmat[i][j];
                  }
              }
          }

/* find the most closely related sequence to those already aligned */

         max = -1;
         for (i=1;i<=nseqs;i++)
           {
             if ((aligned[i] == 0) && (maxid[i] > max))
               {
                  max = maxid[i];
                  iseq = i;
               }
           }

/* align this sequence to the existing alignment */

         entries = 0;
         for (j=1;j<=nseqs;j++)
           if (aligned[j] != 0)
              {
                 group[j] = 1;
                 entries++;
              }
           else if (iseq==j)
              {
                 group[j] = 2;
                 entries++;
              }
         aligned[iseq] = 1;


/* EITHER....., set sequence weights equal to percent identity with new sequence */
/*
           for (j=0;j<nseqs;j++)
              seq_weight[j] = tmat[j+1][iseq];
*/
/* OR...., multiply sequence weights from tree by percent identity with new sequence */
           for (j=0;j<nseqs;j++)
              seq_weight[j] = tree_weight[j] * tmat[j+1][iseq];
if (debug>1)         
  for (j=0;j<nseqs;j++) if (group[j+1] == 1)fprintf (stdout,"sequence %d: %d\n", j+1,tree_weight[j]);
/*
  Normalise the weights, such that the sum of the weights = INT_SCALE_FACTOR
*/

         sum = 0;
         for (j=0;j<nseqs;j++)
           if (group[j+1] == 1) sum += seq_weight[j];
         if (sum == 0)
          {
           for (j=0;j<nseqs;j++)
                seq_weight[j] = 1;
                sum = j;
          }
         for (j=0;j<nseqs;j++)
             {
               seq_weight[j] = (seq_weight[j] * INT_SCALE_FACTOR) / sum;
               if (seq_weight[j] < 1) seq_weight[j] = 1;
             }

if (debug > 1) {
  fprintf(stdout,"new weights\n");
  for (j=0;j<nseqs;j++) if (group[j+1] == 1)fprintf( stdout,"sequence %d: %d\n", j+1,seq_weight[j]);
}

         score = prfalign(group, aligned);
         info("Sequence:%d     Score:%d",(pint)iseq,(pint)score);
         if (output_order == INPUT)
          {
            ++ix;
            output_index[iseq] = iseq;
          }
         else
            output_index[++ix] = iseq;
      }

   group=ckfree((void *)group);
   aligned=ckfree((void *)aligned);
   maxid=ckfree((void *)maxid);

   aln_score();

/* make the rest (output stuff) into routine clustal_out in file amenu.c */

   return(nseqs);

}


sint palign1(void)  /* a profile alignment */
{
   sint 		i,j,temp;
   sint 		entries;
   sint 		*aligned, *group;
   float        dscore;
   lint			score;

   info("Start of Initial Alignment");

/* calculate sequence weights according to branch lengths of the tree -
   weights in global variable seq_weight normalised to sum to INT_SCALE_FACTOR */

   temp = INT_SCALE_FACTOR/nseqs;
   for (i=0;i<nseqs;i++) seq_weight[i] = temp;

   distance_tree = FALSE;

/* do the initial alignment.........  */

   group = (sint *)ckalloc( (nseqs+1) * sizeof (sint));

   for(i=1; i<=profile1_nseqs; ++i)
         group[i] = 1;
   for(i=profile1_nseqs+1; i<=nseqs; ++i)
         group[i] = 2;
   entries = nseqs;

   aligned = (sint *)ckalloc( (nseqs+1) * sizeof (sint) );
   for (i=1;i<=nseqs;i++) aligned[i] = 1;

   score = prfalign(group, aligned);
   info("Sequences:%d      Score:%d",(pint)entries,(pint)score);
   group=ckfree((void *)group);
   aligned=ckfree((void *)aligned);

   for (i=1;i<=nseqs;i++) {
     for (j=i+1;j<=nseqs;j++) {
       dscore = countid(i,j);
       tmat[i][j] = ((double)100.0 - (double)dscore)/(double)100.0;
       tmat[j][i] = tmat[i][j];
     }
   }

   return(nseqs);
}

float countid(sint s1, sint s2)
{
   char c1,c2;
   sint i;
   sint count,total;
   float score;

   count = total = 0;
   for (i=1;i<=seqlen_array[s1] && i<=seqlen_array[s2];i++) {
     c1 = seq_array[s1][i];
     c2 = seq_array[s2][i];
     if ((c1>=0) && (c1<max_aa)) {
       total++;
       if (c1 == c2) count++;
     }

   }

   if(total==0) score=0;
   else
   score = 100.0 * (float)count / (float)total;
   return(score);

}

sint palign2(char *p1_tree_name,char *p2_tree_name)  /* a profile alignment */
{
   sint 	i,j,sum,entries,status;
   lint 		score;
   sint 	*aligned, *group;
   sint		*maxid,*p1_weight,*p2_weight;
/*   sint dscore; */

   info("Start of Multiple Alignment");

/* get the phylogenetic trees from *.ph */

   if (profile1_nseqs >= 2)
     {
        status = read_tree(p1_tree_name, (sint)0, profile1_nseqs);
        if (status == 0) return(0);
     }

/* calculate sequence weights according to branch lengths of the tree -
   weights in global variable seq_weight normalised to sum to 100 */

   p1_weight = (sint *) ckalloc( (profile1_nseqs) * sizeof(sint) );

   calc_seq_weights((sint)0, profile1_nseqs, p1_weight);

/* clear the memory for the phylogenetic tree */

   if (profile1_nseqs >= 2)
        clear_tree(NULL);

   if (nseqs-profile1_nseqs >= 2)
     {
        status = read_tree(p2_tree_name, profile1_nseqs, nseqs);
        if (status == 0) return(0);
     }

   p2_weight = (sint *) ckalloc( (nseqs) * sizeof(sint) );

   calc_seq_weights(profile1_nseqs,nseqs, p2_weight);


/* clear the memory for the phylogenetic tree */

   if (nseqs-profile1_nseqs >= 2)
        clear_tree(NULL);

/* convert tmat distances to similarities */

   for (i=1;i<nseqs;i++)
        for (j=i+1;j<=nseqs;j++) {
            tmat[i][j]=100.0-tmat[i][j]*100.0;
            tmat[j][i]=tmat[i][j];
        }
     

/* weight sequences with max percent identity with other profile*/

   maxid = (sint *)ckalloc( (nseqs+1) * sizeof (sint));
   for (i=0;i<profile1_nseqs;i++) {
         maxid[i] = 0;
         for (j=profile1_nseqs+1;j<=nseqs;j++) 
                      if(maxid[i]<tmat[i+1][j]) maxid[i] = tmat[i+1][j];
         seq_weight[i] = maxid[i]*p1_weight[i];
   }

   for (i=profile1_nseqs;i<nseqs;i++) {
         maxid[i] = -1;
         for (j=1;j<=profile1_nseqs;j++)
                      if(maxid[i]<tmat[i+1][j]) maxid[i] = tmat[i+1][j];
         seq_weight[i] = maxid[i]*p2_weight[i];
   }
/*
  Normalise the weights, such that the sum of the weights = INT_SCALE_FACTOR
*/

         sum = 0;
         for (j=0;j<nseqs;j++)
            sum += seq_weight[j];
         if (sum == 0)
          {
           for (j=0;j<nseqs;j++)
                seq_weight[j] = 1;
                sum = j;
          }
         for (j=0;j<nseqs;j++)
             {
               seq_weight[j] = (seq_weight[j] * INT_SCALE_FACTOR) / sum;
               if (seq_weight[j] < 1) seq_weight[j] = 1;
             }
if (debug > 1) {
  fprintf(stdout,"new weights\n");
  for (j=0;j<nseqs;j++) fprintf( stdout,"sequence %d: %d\n", j+1,seq_weight[j]);
}


/* do the alignment.........  */

   info("Aligning...");

   group = (sint *)ckalloc( (nseqs+1) * sizeof (sint));

   for(i=1; i<=profile1_nseqs; ++i)
         group[i] = 1;
   for(i=profile1_nseqs+1; i<=nseqs; ++i)
         group[i] = 2;
   entries = nseqs;

   aligned = (sint *)ckalloc( (nseqs+1) * sizeof (sint) );
   for (i=1;i<=nseqs;i++) aligned[i] = 1;

   score = prfalign(group, aligned);
   info("Sequences:%d      Score:%d",(pint)entries,(pint)score);
   group=ckfree((void *)group);
   p1_weight=ckfree((void *)p1_weight);
   p2_weight=ckfree((void *)p2_weight);
   aligned=ckfree((void *)aligned);
   maxid=ckfree((void *)maxid);

/* DES   output_index = (int *)ckalloc( (nseqs+1) * sizeof (int)); */
   for (i=1;i<=nseqs;i++) output_index[i] = i;

   return(nseqs);
}

