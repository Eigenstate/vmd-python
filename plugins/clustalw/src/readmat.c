#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "clustalw.h"
#include "matrices.h"


/*
 *   Prototypes
 */
static Boolean commentline(char *line);


/*
 *   Global variables
 */

extern char 	*amino_acid_codes;
extern sint 	gap_pos1, gap_pos2;
extern sint 	max_aa;
extern short 	def_dna_xref[],def_aa_xref[];
extern sint 	mat_avscore;
extern sint 	debug;
extern Boolean  dnaflag;

extern Boolean user_series;
extern UserMatSeries matseries;
extern short usermatseries[MAXMAT][NUMRES][NUMRES];
extern short aa_xrefseries[MAXMAT][NUMRES+1];


void init_matrix(void)
{

   char c1,c2;
   short i, j, maxres;

   max_aa = strlen(amino_acid_codes)-2;
   gap_pos1 = NUMRES-2;          /* code for gaps inserted by clustalw */
   gap_pos2 = NUMRES-1;           /* code for gaps already in alignment */

/*
   set up cross-reference for default matrices hard-coded in matrices.h
*/
   for (i=0;i<NUMRES;i++) def_aa_xref[i] = -1;
   for (i=0;i<NUMRES;i++) def_dna_xref[i] = -1;

   maxres = 0;
   for (i=0;(c1=amino_acid_order[i]);i++)
     {
         for (j=0;(c2=amino_acid_codes[j]);j++)
          {
           if (c1 == c2)
               {
                  def_aa_xref[i] = j;
                  maxres++;
                  break;
               }
          }
         if ((def_aa_xref[i] == -1) && (amino_acid_order[i] != '*'))
            {
                error("residue %c in matrices.h is not recognised",
                                       amino_acid_order[i]);
            }
     }

   maxres = 0;
   for (i=0;(c1=nucleic_acid_order[i]);i++)
     {
         for (j=0;(c2=amino_acid_codes[j]);j++)
          {
           if (c1 == c2)
               {
                  def_dna_xref[i] = j;
                  maxres++;
                  break;
               }
          }
         if ((def_dna_xref[i] == -1) && (nucleic_acid_order[i] != '*'))
            {
                error("nucleic acid %c in matrices.h is not recognised",
                                       nucleic_acid_order[i]);
            }
     }
}

sint get_matrix(short *matptr, short *xref, sint matrix[NUMRES][NUMRES], Boolean neg_flag, sint scale)
{
   sint gg_score = 0;
   sint gr_score = 0;
   sint i, j, k, ix = 0;
   sint ti, tj;
   sint  maxres;
   sint av1,av2,av3,min, max;
/*
   default - set all scores to 0
*/
   for (i=0;i<=max_aa;i++)
      for (j=0;j<=max_aa;j++)
          matrix[i][j] = 0;

   ix = 0;
   maxres = 0;
   for (i=0;i<=max_aa;i++)
    {
      ti = xref[i];
      for (j=0;j<=i;j++)
       {
          tj = xref[j]; 
          if ((ti != -1) && (tj != -1))
            {
               k = matptr[ix];
               if (ti==tj)
                  {
                     matrix[ti][ti] = k * scale;
                     maxres++;
                  }
               else
                  {
                     matrix[ti][tj] = k * scale;
                     matrix[tj][ti] = k * scale;
                  }
               ix++;
            }
       }
    }

   --maxres;

   av1 = av2 = av3 = 0;
   for (i=0;i<=max_aa;i++)
    {
      for (j=0;j<=i;j++)
       {
           av1 += matrix[i][j];
           if (i==j)
              {
                 av2 += matrix[i][j];
              }
           else
              {
                 av3 += matrix[i][j];
              }
       }
    }

   av1 /= (maxres*maxres)/2;
   av2 /= maxres;
   av3 /= ((float)(maxres*maxres-maxres))/2;
  mat_avscore = -av3;

  min = max = matrix[0][0];
  for (i=0;i<=max_aa;i++)
    for (j=1;j<=i;j++)
      {
        if (matrix[i][j] < min) min = matrix[i][j];
        if (matrix[i][j] > max) max = matrix[i][j];
      }
if (debug>1) fprintf(stdout,"maxres %d\n",(pint)max_aa);
if (debug>1) fprintf(stdout,"average mismatch score %d\n",(pint)av3);
if (debug>1) fprintf(stdout,"average match score %d\n",(pint)av2);
if (debug>1) fprintf(stdout,"average score %d\n",(pint)av1);

/*
   if requested, make a positive matrix - add -(lowest score) to every entry
*/
  if (neg_flag == FALSE)
   {

if (debug>1) fprintf(stdout,"min %d max %d\n",(pint)min,(pint)max);
      if (min < 0)
        {
           for (i=0;i<=max_aa;i++)
            {
              ti = xref[i];
              if (ti != -1)
                {
                 for (j=0;j<=max_aa;j++)
                   {
                    tj = xref[j];
/*
                    if (tj != -1) matrix[ti][tj] -= (2*av3);
*/
                    if (tj != -1) matrix[ti][tj] -= min;
                   }
                }
            }
        }
/*
       gr_score = av3;
       gg_score = -av3;
*/

   }



  for (i=0;i<gap_pos1;i++)
   {
      matrix[i][gap_pos1] = gr_score;
      matrix[gap_pos1][i] = gr_score;
      matrix[i][gap_pos2] = gr_score;
      matrix[gap_pos2][i] = gr_score;
   }
  matrix[gap_pos1][gap_pos1] = gg_score;
  matrix[gap_pos2][gap_pos2] = gg_score;
  matrix[gap_pos2][gap_pos1] = gg_score;
  matrix[gap_pos1][gap_pos2] = gg_score;

  maxres += 2;

  return(maxres);
}


sint read_matrix_series(char *filename, short *usermat, short *xref)
{
   FILE *fd = NULL/*, *matfd = NULL*/;
   char mat_filename[FILENAMELEN];
   char inline1[1024];
   sint  maxres = 0;
   sint nmat;
   sint n=0,llimit,ulimit;

   if (filename[0] == '\0')
     {
        error("comparison matrix not specified");
        return((sint)0);
     }
   if ((fd=fopen(filename,"r"))==NULL) 
     {
        error("cannot open %s", filename);
        return((sint)0);
     }

/* check the first line to see if it's a series or a single matrix */
   while (fgets(inline1,1024,fd) != NULL)
     {
        if (commentline(inline1)) continue;
	if(linetype(inline1,"CLUSTAL_SERIES"))
		user_series=TRUE;
	else
		user_series=FALSE;
        break;
     }

/* it's a single matrix */
  if(user_series == FALSE)
    {
	fclose(fd);
   	maxres=read_user_matrix(filename,usermat,xref);
   	return(maxres);
    }

/* it's a series of matrices, find the next MATRIX line */
   nmat=0;
   matseries.nmat=0;
   while (fgets(inline1,1024,fd) != NULL)
     {
        if (commentline(inline1)) continue;
	if(linetype(inline1,"MATRIX"))
	{
		if(sscanf(inline1+6,"%d %d %s",&llimit,&ulimit,mat_filename)!=3)
		{
			error("Bad format in file %s\n",filename);
   			fclose(fd);
			return((sint)0);
		}
		if(llimit<0 || llimit > 100 || ulimit <0 || ulimit>100)
		{
			error("Bad format in file %s\n",filename);
   			fclose(fd);
			return((sint)0);
		}
		if(ulimit<=llimit)
		{
			error("in file %s: lower limit is greater than upper (%d-%d)\n",filename,llimit,ulimit);
   			fclose(fd);
			return((sint)0);
		}
   		n=read_user_matrix(mat_filename,&usermatseries[nmat][0][0],&aa_xrefseries[nmat][0]);
		if(n<=0)
		{
			error("Bad format in matrix file %s\n",mat_filename);
   			fclose(fd);
			return((sint)0);
		}
		matseries.mat[nmat].llimit=llimit;
		matseries.mat[nmat].ulimit=ulimit;
		matseries.mat[nmat].matptr=&usermatseries[nmat][0][0];
		matseries.mat[nmat].aa_xref=&aa_xrefseries[nmat][0];
		nmat++;
	}
    }
   fclose(fd);
   matseries.nmat=nmat;

   maxres=n;
   return(maxres);

}

sint read_user_matrix(char *filename, short *usermat, short *xref)
{
   double f;
   FILE *fd;
   sint  numargs,farg;
   sint i, j, k = 0;
   char codes[NUMRES];
   char inline1[1024];
   char *args[NUMRES+4];
   char c1,c2;
   sint ix1, ix = 0;
   sint  maxres = 0;
   float scale;

   if (filename[0] == '\0')
     {
        error("comparison matrix not specified");
       	return((sint)0);
     }

   if ((fd=fopen(filename,"r"))==NULL) 
   {
       	error("cannot open %s", filename);
       	return((sint)0);
   }
   maxres = 0;
   while (fgets(inline1,1024,fd) != NULL)
     {
        if (commentline(inline1)) continue;
	if(linetype(inline1,"CLUSTAL_SERIES"))
   	{
       		error("in %s - single matrix expected.", filename);
		fclose(fd);
       		return((sint)0);
   	}
/*
   read residue characters.
*/
        k = 0;
        for (j=0;j<strlen(inline1);j++)
          {
             if (isalpha((int)inline1[j])) codes[k++] = inline1[j];
             if (k>NUMRES)
                {
                   error("too many entries in matrix %s",filename);
		   fclose(fd);
                   return((sint)0);
                }
          }
        codes[k] = '\0';
        break;
    }

   if (k == 0) 
     {
        error("wrong format in matrix %s",filename);
  	fclose(fd);
        return((sint)0);
     }

/*
   cross-reference the residues
*/
   for (i=0;i<NUMRES;i++) xref[i] = -1;

   maxres = 0;
   for (i=0;(c1=codes[i]);i++)
     {
         for (j=0;(c2=amino_acid_codes[j]);j++)
           if (c1 == c2)
               {
                  xref[i] = j;
                  maxres++;
                  break;
               }
         if ((xref[i] == -1) && (codes[i] != '*'))
            {
                warning("residue %c in matrix %s not recognised",
                                       codes[i],filename);
            }
     }


/*
   get the weights
*/

   ix = ix1 = 0;
   while (fgets(inline1,1024,fd) != NULL)
     {
        if (inline1[0] == '\n') continue;
        if (inline1[0] == '#' ||
            inline1[0] == '!') break;
        numargs = getargs(inline1, args, (int)(k+1));
        if (numargs < maxres)
          {
             error("wrong format in matrix %s",filename);
  	     fclose(fd);
             return((sint)0);
          }
        if (isalpha(args[0][0])) farg=1;
        else farg=0;

/* decide whether the matrix values are float or decimal */
	scale=1.0;
	for(i=0;i<strlen(args[farg]);i++)
		if(args[farg][i]=='.')
		{
/* we've found a float value */
			scale=10.0;
			break;
		}

        for (i=0;i<=ix;i++)
          {
             if (xref[i] != -1)
               {
                  f = atof(args[i+farg]);
                  usermat[ix1++] = (short)(f*scale);
               }
          }
        ix++;
     }
   if (ix != k+1)
     {
        error("wrong format in matrix %s",filename);
  	fclose(fd);
        return((sint)0);
     }


  maxres += 2;
  fclose(fd);

  return(maxres);
}

int getargs(char *inline1,char *args[],int max)
{

	char	*inptr;
/*
#ifndef MAC
	char	*strtok(char *s1, const char *s2);
#endif
*/
	int	i;

	inptr=inline1;
	for (i=0;i<=max;i++)
	{
		if ((args[i]=strtok(inptr," \t\n"))==NULL)
			break;
		inptr=NULL;
	}

	return(i);
}


static Boolean commentline(char *line)
{
        int i;
 
        if(line[0] == '#') return TRUE;
        for(i=0;line[i]!='\n' && line[i]!=EOS;i++) {
                if(!isspace(line[i]))
			return FALSE;
        }
        return TRUE;
}

