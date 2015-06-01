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
#include <stdlib.h>
#include <gjutil.h>
#include <gjnoc.h>

/* Given an upper diagonal matrix, return a tree in stamp format via Geoff's
 *  OC routine */

struct cluster *get_clust(double **matrix, char **ids, int ndomain, char *noc_parms) {

	int i,j/*,k*/;
	
	int nclust;

	struct gjnoc *gjclust;
	struct cluster *cl;

/*	printf("Parameters are: %s\n",noc_parms);  */

	nclust=ndomain-1; 
	gjclust=GJnoc(matrix,ids,ndomain,noc_parms);
	cl=(struct cluster*)malloc((nclust)*sizeof(struct cluster));

/*	printf("Order:\n");
	for(i=0; i<ndomain; ++i) {
		printf("%4d\n",gjclust->order[i]);
	}
	for(i=0; i<nclust; ++i) {
		printf("\nCluster %4d\nA:",i+1);
		for(j=0; j<gjclust->clusters[i].a.number; ++j) {
		printf("%4d ",gjclust->clusters[i].a.member[j]);
		}
		printf("\nB:");
		for(j=0; j<gjclust->clusters[i].b.number; ++j) {
                  printf("%4d ",gjclust->clusters[i].b.member[j]);
                }
	}
*/

	for(i=0; i<nclust; ++i) { /* take Geoff's structure and convert it to a tree */
	      cl[i].a.member=(int*)malloc(gjclust->clusters[i].a.number*sizeof(int));
	      cl[i].a.number=gjclust->clusters[i].a.number;
	      cl[i].b.member=(int*)malloc(gjclust->clusters[i].b.number*sizeof(int));
              cl[i].b.number=gjclust->clusters[i].b.number;
	      /* Now sort out the clusters */
	      for(j=0; j<gjclust->clusters[i].a.number; ++j) {
			cl[i].a.member[j]=gjclust->order[gjclust->clusters[i].a.member[j]]; 
	      }
	      for(j=0; j<gjclust->clusters[i].b.number; ++j) {
                        cl[i].b.member[j]=gjclust->order[gjclust->clusters[i].b.member[j]]; 
              }
	}
/*	printf("Returned from GJnoc...\n");
	for(i=0; i<nclust; ++i) {
	   printf("Cluster: %4d (",i+1);
              for(k=0; k<cl[i].a.number; ++k) printf("%d ",cl[i].a.member[k]);
	   printf("   and ");
	   for(k=0; k<cl[i].b.number; ++k) printf("%d ",cl[i].b.member[k]);
	   printf(") \n\n");
	}
*/
	for(i=0; i<nclust; ++i) {
	  free(gjclust->clusters[i].a.member);
	  free(gjclust->clusters[i].b.member);
	}
	free(gjclust->clusters);
	free(gjclust->order);
	free(gjclust);

	return cl;
}
