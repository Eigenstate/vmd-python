/*-------------------------------------------------------------

Copyright (C) 2000 Peter Clote. 
All Rights Reserved.

Permission to use, copy, modify, and distribute this
software and its documentation for NON-COMMERCIAL purposes
and without fee is hereby granted provided that this
copyright notice appears in all copies.


THE AUTHOR AND PUBLISHER MAKE NO REPRESENTATIONS OR
WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE AUTHORS
AND PUBLISHER SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED
BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
THIS SOFTWARE OR ITS DERIVATIVES.

-------------------------------------------------------------*/



/*********************************************
	Program: phy2.c

	Programmer: Peter Clote, 30 Dec 1996
	Phylogeny. This program REQUIRES an input file at most
	M lines corresponding to an N x N array of integer or
	floating pt numbers for the the distance matrix between N species. 
	Here N is variable, M a constant, and it is required that N <= M.
	The program checks that the input file has at MOST M lines,
	but does not check that each line consists of N numbers, where
	N is the exact number of lines.

*********************************************/

#include <math.h>
#include <string.h>			/* use strcat function */
#include <stdio.h>
#include <stdlib.h>   			/* prototypes for malloc, NULL */

//#define M 10				/* maximum of M species for phylogeny */
#define INF RAND_MAX
#define MAX(x,y)  ( (x)>(y) ? (x) : (y) )

/*-------------------------------------------
Declarations for a circularly linked list. List
will hold species names (integers in [1..N]) belonging
to a current cluster. A pointer to such lists will
be a field element of "node" to be defined later.
-------------------------------------------*/
struct listnode
{
	int val;	/* val is a species (element in [1..N])  */
	struct listnode *up;	
	struct listnode *down;
};

typedef struct listnode ListNode;
typedef ListNode *ListPtr;



/*-------------------------------------------
Declarations for binary tree of nodes, where a node
has fields for distance between the left and right clusters, for pointer to
circularly linked list of species names (integers) in the cluster,
and size of cluster.
-------------------------------------------*/
struct node
{
	int leaf;	/* leaf = 1 if true, else 0 */
	int depth;	/* depth of leaf is 0 */
	float  dist; 	/* dist is distance between left, right clusters */
	int index;	/* name of species, if node is leaf, else -1 */
	int size;	/* size of current cluster */
	struct node *left;
	struct node *right;
	struct listnode *list;	/*pointer to circ linked list of cluster elements */
}; 

typedef struct node Node;
typedef Node *Ptr;

/*-------------------------------------------
debugging procedure: used to print updates of 
distance matrix, where E[i][j] is distance between
i-th and j-th cluster.
-------------------------------------------*/
void printMatrix ( float **D, int N )
{
	int i,j;
	for (i=0;i<N;i++)
		{
		for (j=0;j<N;j++)
			printf("%4.2f\t",D[i][j]);
		printf("\n");
		}
	printf("\n");
}





int main(int argc, char *argv[])
{

	/*------------------------------------------------------
	Declarations.
	-------------------------------------------------------*/
	int N;			/* expect N x N array input, with N <= M */
	float distance( float **D, ListPtr, int, ListPtr, int);
	ListPtr join( ListPtr, ListPtr );
	void print( Ptr, int);
	void printString(Ptr);
	int num_lines(FILE *);
	FILE *in;

	int num; 	/* number of clusters */
	int i,i_index=0; 	/* i_index index of new cluster to amalgamate */
	int j,j_index=0; 	/* j_index index of new cluster to amalgamate */
	float d,dmin;	/* used to find closest clusters to amalgamate */
	//Ptr c[M];	/* c[i] is tree of i-th cluster */
	Ptr *c;
	Ptr p;
	//float D[M][M], E[M][M];
	float **D;
	float **E;
	float  distance(float **D, ListPtr p, int n, ListPtr q, int m);


	/*********  ERROR CHECKING in command line arguments *********/

	if (argc != 2)
		{
		fprintf(stderr,"Usage: %s filename.\n",argv[0]);
		exit(1);
     		}
	if ((in = fopen(argv[1],"r")) == NULL)
    		{ 
     		fprintf(stderr,"%s is not a readable file.\n", argv[1]);
     		exit(1);
     		}

	/*------------------------------------------------------
	  Compute number lines of input file. Expect N x N array of
	  float for distance matrix between N species. Do not do
	  error checking for N numbers per line, but do require
	  N to be the number of lines of input.
	  -------------------------------------------------------*/
	
   	N =  num_lines(in);
	/*
	  if ( M < N)
	  {
	  fprintf(stderr,"Set constant M to %d\n",N);
	  exit(1);
	  }
	*/
	
	/*-----------------------
	  Allocate memory for arrays
	-------------------*/
	D = (float **) malloc(N * sizeof(float*));
	E = (float **) malloc(N * sizeof(float*));
	c = (Ptr *) malloc(N * sizeof(Ptr));
	  
	for (i=0;i<N;i++) {
	  D[i] = (float *) malloc(N * sizeof(float));
	  E[i] = (float *) malloc(N * sizeof(float));
	}
	
	
	/*-------------------------------------------
	  D[i][j] distance between i-th and j-th species 
	  -------------------------------------------*/
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
			fscanf(in,"%f",&(D[i][j]));	
			


	/*-------------------------------------------
	E[i][j] distance between i-th and j-th cluster, 
	updated upon amalgamation of clusters.
	copy D into E, E[i][j] distance between i-th 
	and j-th cluster. 
	-------------------------------------------*/
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
			E[i][j] = D[i][j];


	/*-------------------------------------------
	Initialize clusters and cluster sizes.
	-------------------------------------------*/
	for (i=0;i<N;i++) 
		{
		c[i] = ( Ptr ) malloc(sizeof(Node));
		c[i]->leaf = 1;
		c[i]->depth = 0;
		c[i]->dist = 0;	
		c[i]->index = i;	
		c[i]->size = 1;
		c[i]->left = NULL;
		c[i]->right = NULL;
		c[i]->list = ( ListPtr ) malloc(sizeof(ListNode));
		c[i]->list->val = i;
		c[i]->list->up = c[i]->list->down = c[i]->list; 
		}

	printf("Initial distance matrix\n");
	printMatrix(D,N);


	/*-------------------------------------------
	Using E[i][j] find the closest 2 clusters, amalgamate,
	update array of clusters, size of clusters, and E[i][j]'s.
	Repeatedly do this, until there is only 1 cluster left.
 	-------------------------------------------*/
	num = N;
	while ( num > 1 )
		{
		dmin = INF;
		for (i=0;i<N-1;i++)
			for (j=i+1;j<N;j++)
				{
				if (( c[i] != NULL ) && ( c[j] != NULL ))
					{
					d = E[i][j];
					if ( d < dmin )
						{
						dmin = d;
						i_index = i;
						j_index = j;
					}
				}
			}

	/*-------------------------------------------
	Create new node, left child is c[i], right child is c[j].
	Value of p node is distance between c[i] and c[j] clusters.	
	c[i] is now p, the amalgamated cluster (i<j), and index = 0
	for non-leaf nodes.
 	-------------------------------------------*/
	i = i_index;
	j = j_index;
	p = (Ptr) malloc(sizeof(Node));
	p->leaf = 0;
	p->depth = 1 + MAX(c[i]->depth,c[j]->depth);
	p->dist = distance(D,c[i]->list,c[i]->size,c[j]->list,c[j]->size);
	p->index = -1; /* important */
	p->size = c[i]->size + c[j]->size;
	p->left = c[i];
	p->right = c[j];
	p->list = join(c[i]->list, c[j]->list);
	c[i] = p;
	c[j] = NULL;
	num--;		

#if 0	
	/*-------------------------------------------
	Recall that i = i_index from last call.
	Following 2 for-loops used for runtime version.
 	---------------------------------------------*/
	for (j=0;j<N;j++)
		if ( (c[j] != NULL) && (i != j))
			E[i][j] = distance(D,c[i]->list,c[i]->size,c[j]->list,c[j]->size);
		else
			E[i][j] = 0;
	for (j=0;j<N;j++)
		if ( (c[j] != NULL) && (i != j))
			E[j][i] = E[i][j];
#endif	
	
	/*-------------------------------------------
	Note: runtime version in previous 2 for-loops.
	For debugging and display, the following is better.
 	-------------------------------------------*/
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
			if (( c[i] != NULL) && (c[j] != NULL ) && i != j)
				E[i][j] = distance(D,c[i]->list,c[i]->size,c[j]->list,c[j]->size);
			else E[i][j] = 0;
	printf("Step %d\n",N-num);
	printf("Indices of clusters to amalgamate: %d, %d\n",i_index,j_index);
	printMatrix(E,N);

	}    /* end of while loop */	

	/*-------------------------------------------
	Here, there is only 1 cluster left, with index i_index,
	so print out tree.
 	-------------------------------------------*/
	printf("Phylogeny tree:----------------\n");
	print(c[i_index],1);

	printf("Tree String: ");
	printString(c[i_index]);
	printf("\n");
	
	return 0;
}    /* end of main program */	


/*-------------------------------------------
Distance is the average distance between clusters.
Sum is sum of distances between all pairs i,j where i
in left cluster, j in right cluster. Assume that p,q not NULL,
and n = size of cluster pointed to by p, and m = size of cluster
pointed to by q. The distance between clusters is then distSum / (n * m).
Need the distance matrix D[M][M].
-------------------------------------------*/
float  distance(float **D, ListPtr p, int n, ListPtr q, int m)
	{
	float sum = 0;
	int i,j;

	ListPtr r = p;
	for (i=0;i<n;i++)
		{
		ListPtr s = q;
		for (j=0;j<m;j++)
			{
			sum += D[r->val][s->val];
			s = s->down;
			}
		r = r->down;
		}
	return sum/(n*m);
	}


/*-------------------------------------------
p,q are circularly linked lists, both non-NULL.
join(p,q) returns pointer to circularly linked list obtained
by putting q list between p and p->down.
-------------------------------------------*/

ListPtr join( ListPtr p, ListPtr q )
{
if ( p->up == p )
	{
	p->up = q->up;
	q->up->down = p;
	p->down = q;
	q->up = p;	
	}
else if ( q->up == q )
	{
	q->down = p->down;
	p->down->up = q;
	p->down = q;
	q->up = p;
	}
else
	{
	q->up->down = p->down;
	p->down->up = q->up;
	p->down = q;
	q->up = p;
	}
return p;

}



/*-------------------------------------------
print phylogeny tree. When called, n should be 0 or 1.
This prints the tree either flush against the left margin,
or with a tab of 1.
-------------------------------------------*/
void print( Ptr p, int n)
{
int i;
int depth( Ptr );

if ( p->leaf == 1 )
	{
	for (i=0;i<n;i++) putchar(9);	/* 9 is tab code */
	printf("%d\n",p->index);
	}
else
	{
	print(p->right,n+1);
	for (i=0;i<n;i++) putchar(9);	/* 9 is tab code */
	printf("%2.2f\n",p->dist/(float) 2);	/* half of distance between pts*/
	print(p->left,n+1);
	}	
}

void printString( Ptr p )
{
/*  int i; */
  int depth(Ptr);

  if ( p->leaf == 1 ) {
    printf(" ( %d %d ) ",0,p->index);
  }
  else {
    printf(" ( %2.2f",p->dist/(float) 2);
    printString(p->right);
    printString(p->left);
    printf(" )");
  }
}

int num_lines(FILE *in)
{
   int n=0,c;

   while ((c = fgetc(in)) != EOF)
     if (c == '\n') ++n;
   rewind(in);
   return(n);
}

