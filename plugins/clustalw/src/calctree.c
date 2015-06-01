#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>
#include "clustalw.h"

#define MAXERRS 10

/*
 *   Prototypes
 */
static void create_tree(treeptr ptree, treeptr parent);
static void create_node(treeptr pptr, treeptr parent);
static treeptr insert_node(treeptr pptr);
static void skip_space(FILE *fd);
static treeptr avail(void);
static void set_info(treeptr p, treeptr parent, sint pleaf, char *pname, float pdist);
static treeptr reroot(treeptr ptree, sint nseqs);
static treeptr insert_root(treeptr p, float diff);
static float calc_root_mean(treeptr root, float *maxdist);
static float calc_mean(treeptr nptr, float *maxdist, sint nseqs);
static void order_nodes(void);
static sint calc_weight(sint leaf);
static void group_seqs(treeptr p, sint *next_groups, sint nseqs);
static void mark_group1(treeptr p, sint *groups, sint n);
static void mark_group2(treeptr p, sint *groups, sint n);
static void save_set(sint n, sint *groups);
static void clear_tree_nodes(treeptr p);


/*
 *   Global variables
 */
extern Boolean interactive;
extern Boolean distance_tree;
extern Boolean usemenu;
extern sint debug;
extern double **tmat;
extern sint **sets;
extern sint nsets;
extern char **names;
extern sint *seq_weight;
extern Boolean no_weights;

char ch;
FILE *fd;
treeptr *lptr;
treeptr *olptr;
treeptr *nptr;
treeptr *ptrs;
sint nnodes = 0;
sint ntotal = 0;
Boolean rooted_tree = TRUE;
static treeptr seq_tree,root;
static sint *groups, numseq;

void calc_seq_weights(sint first_seq, sint last_seq, sint *sweight)
{
  sint   i, nseqs;
  sint   temp, sum, *weight;


/*
  If there are more than three sequences....
*/
  nseqs = last_seq-first_seq;
   if ((nseqs >= 2) && (distance_tree == TRUE) && (no_weights == FALSE))
     {
/*
  Calculate sequence weights based on Phylip tree.
*/
      weight = (sint *)ckalloc((last_seq+1) * sizeof(sint));

      for (i=first_seq; i<last_seq; i++)
           weight[i] = calc_weight(i);

/*
  Normalise the weights, such that the sum of the weights = INT_SCALE_FACTOR
*/

         sum = 0;
         for (i=first_seq; i<last_seq; i++)
            sum += weight[i];

         if (sum == 0)
          {
            for (i=first_seq; i<last_seq; i++)
               weight[i] = 1;
            sum = i;
          }

         for (i=first_seq; i<last_seq; i++)
           {
              sweight[i] = (weight[i] * INT_SCALE_FACTOR) / sum;
              if (sweight[i] < 1) sweight[i] = 1;
           }

       weight=ckfree((void *)weight);

     }

   else
     {
/*
  Otherwise, use identity weights.
*/
        temp = INT_SCALE_FACTOR / nseqs;
        for (i=first_seq; i<last_seq; i++)
           sweight[i] = temp;
     }

}

void create_sets(sint first_seq, sint last_seq)
{
  sint   i, j, nseqs;

  nsets = 0;
  nseqs = last_seq-first_seq;
  if (nseqs >= 2)
     {
/*
  If there are more than three sequences....
*/
       groups = (sint *)ckalloc((nseqs+1) * sizeof(sint));
       group_seqs(root, groups, nseqs);
       groups=ckfree((void *)groups);

     }

   else
     {
       groups = (sint *)ckalloc((nseqs+1) * sizeof(sint));
       for (i=0;i<nseqs-1;i++)
         {
           for (j=0;j<nseqs;j++)
              if (j<=i) groups[j] = 1;
              else if (j==i+1) groups[j] = 2;
              else groups[j] = 0;
           save_set(nseqs, groups);
         }
       groups=ckfree((void *)groups);
     }

}

sint read_tree(char *treefile, sint first_seq, sint last_seq)
{

  char c;
  char name1[MAXNAMES+1], name2[MAXNAMES+1];
  sint i, j, k;
  Boolean found;

  numseq = 0;
  nnodes = 0;
  ntotal = 0;
  rooted_tree = TRUE;

#ifdef VMS
  if ((fd = fopen(treefile,"r","rat=cr","rfm=var")) == NULL)
#else
  if ((fd = fopen(treefile, "r")) == NULL)
#endif
    {
      error("cannot open %s", treefile);
      return((sint)0);
    }

  skip_space(fd);
  ch = (char)getc(fd);
  if (ch != '(')
    {
      error("Wrong format in tree file %s", treefile);
      return((sint)0);
    }
  rewind(fd);

  distance_tree = TRUE;

/*
  Allocate memory for tree
*/
  nptr = (treeptr *)ckalloc(3*(last_seq-first_seq+1) * sizeof(treeptr));
  ptrs = (treeptr *)ckalloc(3*(last_seq-first_seq+1) * sizeof(treeptr));
  lptr = (treeptr *)ckalloc((last_seq-first_seq+1) * sizeof(treeptr));
  olptr = (treeptr *)ckalloc((last_seq+1) * sizeof(treeptr));
  
  seq_tree = avail();
  set_info(seq_tree, NULL, 0, "", 0.0);

  create_tree(seq_tree,NULL);
  fclose(fd);


  if (numseq != last_seq-first_seq)
     {
         error("tree not compatible with alignment\n(%d sequences in alignment and %d in tree", (pint)last_seq-first_seq,(pint)numseq);
         return((sint)0);
     }

/*
  If the tree is unrooted, reroot the tree - ie. minimise the difference
  between the mean root->leaf distances for the left and right branches of
  the tree.
*/

  if (distance_tree == FALSE)
     {
  	if (rooted_tree == FALSE)
          {
       	     error("input tree is unrooted and has no distances.\nCannot align sequences");
             return((sint)0);
          }
     }

  if (rooted_tree == FALSE)
     {
        root = reroot(seq_tree, last_seq-first_seq+1);
     }
  else
     {
        root = seq_tree;
     }

/*
  calculate the 'order' of each node.
*/
  order_nodes();

  if (numseq >= 2)
     {
/*
  If there are more than three sequences....
*/
/*
  assign the sequence nodes (in the same order as in the alignment file)
*/
      for (i=first_seq; i<last_seq; i++)
       {
         if (strlen(names[i+1]) > MAXNAMES)
             warning("name %s is too long for PHYLIP tree format (max %d chars)", names[i+1],MAXNAMES);

         for (k=0; k< strlen(names[i+1]) && k<MAXNAMES ; k++)
           {
             c = names[i+1][k];
             if ((c>0x40) && (c<0x5b)) c=c | 0x20;
             if (c == ' ') c = '_';
             name2[k] = c;
           }
         name2[k]='\0';
         found = FALSE;
         for (j=0; j<numseq; j++)
           {
            for (k=0; k< strlen(lptr[j]->name) && k<MAXNAMES ; k++)
              {
                c = lptr[j]->name[k];
                if ((c>0x40) && (c<0x5b)) c=c | 0x20;
                name1[k] = c;
              }
            name1[k]='\0';
            if (strcmp(name1, name2) == 0)
              {
                olptr[i] = lptr[j];
                found = TRUE;
              }
           }
         if (found == FALSE)
           {
             error("tree not compatible with alignment:\n%s not found", name2);
             return((sint)0);
           }
       }

     }
   return((sint)1);
}

static void create_tree(treeptr ptree, treeptr parent)
{
   treeptr p;

   sint i, type;
   float dist;
   char name[MAXNAMES+1];

/*
  is this a node or a leaf ?
*/
  skip_space(fd);
  ch = (char)getc(fd);
  if (ch == '(')
    {  
/*
   this must be a node....
*/
      type = NODE;
      name[0] = '\0';
      ptrs[ntotal] = nptr[nnodes] = ptree;
      nnodes++;
      ntotal++;

      create_node(ptree, parent);

      p = ptree->left;
      create_tree(p, ptree);
           
      if ( ch == ',')
       {
          p = ptree->right;
          create_tree(p, ptree);
          if ( ch == ',')
            {
               ptree = insert_node(ptree);
               ptrs[ntotal] = nptr[nnodes] = ptree;
               nnodes++;
               ntotal++;
               p = ptree->right;
               create_tree(p, ptree);
               rooted_tree = FALSE;
            }
       }

      skip_space(fd);
      ch = (char)getc(fd);
    }
/*
   ...otherwise, this is a leaf
*/
  else
    {
      type = LEAF;
      ptrs[ntotal++] = lptr[numseq++] = ptree;
/*
   get the sequence name
*/
      name[0] = ch;
      ch = (char)getc(fd);
      i = 1;
      while ((ch != ':') && (ch != ',') && (ch != ')'))
        {
          if (i < MAXNAMES) name[i++] = ch;
          ch = (char)getc(fd);
        }
      name[i] = '\0';
      if (ch != ':')
         {
           distance_tree = FALSE;
           dist = 0.0;
         }
    }

/*
   get the distance information
*/
  dist = 0.0;
  if (ch == ':')
     {
       skip_space(fd);
       fscanf(fd,"%f",&dist);
       skip_space(fd);
       ch = (char)getc(fd);
     }
   set_info(ptree, parent, type, name, dist);


}

static void create_node(treeptr pptr, treeptr parent)
{
  treeptr t;

  pptr->parent = parent;
  t = avail();
  pptr->left = t;
  t = avail();
  pptr->right = t;
    
}

static treeptr insert_node(treeptr pptr)
{

   treeptr newnode;

   newnode = avail();
   create_node(newnode, pptr->parent);

   newnode->left = pptr;
   pptr->parent = newnode;

   set_info(newnode, pptr->parent, NODE, "", 0.0);

   return(newnode);
}

static void skip_space(FILE *fd)
{
  int   c;
  
  do
     c = getc(fd);
  while(isspace(c));

  ungetc(c, fd);
}

static treeptr avail(void)
{
   treeptr p;
   p = ckalloc(sizeof(stree));
   p->left = NULL;
   p->right = NULL;
   p->parent = NULL;
   p->dist = 0.0;
   p->leaf = 0;
   p->order = 0;
   p->name[0] = '\0';
   return(p);
}

void clear_tree(treeptr p)
{
   clear_tree_nodes(p);
      
   nptr=ckfree((void *)nptr);
   ptrs=ckfree((void *)ptrs);
   lptr=ckfree((void *)lptr);
   olptr=ckfree((void *)olptr);
}

static void clear_tree_nodes(treeptr p)
{
   if (p==NULL) p = root;
   if (p->left != NULL)
     {
       clear_tree_nodes(p->left);
     }
   if (p->right != NULL)
     {
       clear_tree_nodes(p->right);
     }
   p->left = NULL;
   p->right = NULL;
   p=ckfree((void *)p);   
}

static void set_info(treeptr p, treeptr parent, sint pleaf, char *pname, float pdist)
{
   p->parent = parent;
   p->leaf = pleaf;
   p->dist = pdist;
   p->order = 0;
   strcpy(p->name, pname);
   if (p->leaf == TRUE)
     {
        p->left = NULL;
        p->right = NULL;
     }
}

static treeptr reroot(treeptr ptree, sint nseqs)
{

   treeptr p, rootnode, rootptr;
   float   diff, mindiff = 0.0, mindepth = 1.0, maxdist;
   sint   i;
   Boolean first = TRUE;

/*
  find the difference between the means of leaf->node
  distances on the left and on the right of each node
*/
   rootptr = ptree;
   for (i=0; i<ntotal; i++)
     {
        p = ptrs[i];
        if (p->parent == NULL)
           diff = calc_root_mean(p, &maxdist);
        else
           diff = calc_mean(p, &maxdist, nseqs);

        if ((diff == 0) || ((diff > 0) && (diff < 2 * p->dist)))
          {
              if ((maxdist < mindepth) || (first == TRUE))
                 {
                    first = FALSE;
                    rootptr = p;
                    mindepth = maxdist;
                    mindiff = diff;
                 }
           }

     }

/*
  insert a new node as the ancestor of the node which produces the shallowest
  tree.
*/
   if (rootptr == ptree)
     {
        mindiff = rootptr->left->dist + rootptr->right->dist;
        rootptr = rootptr->right;
     }
   rootnode = insert_root(rootptr, mindiff);
  
   diff = calc_root_mean(rootnode, &maxdist);

   return(rootnode);
}

static treeptr insert_root(treeptr p, float diff)
{
   treeptr newp, prev, q, t;
   float dist, prevdist,td;

   newp = avail();

   t = p->parent;
   prevdist = t->dist;

   p->parent = newp;

   dist = p->dist;

   p->dist = diff / 2;
   if (p->dist < 0.0) p->dist = 0.0;
   if (p->dist > dist) p->dist = dist;

   t->dist = dist - p->dist; 

   newp->left = t;
   newp->right = p;
   newp->parent = NULL;
   newp->dist = 0.0;
   newp->leaf = NODE;

   if (t->left == p) t->left = t->parent;
   else t->right = t->parent;

   prev = t;
   q = t->parent;

   t->parent = newp;

   while (q != NULL)
     {
        if (q->left == prev)
           {
              q->left = q->parent;
              q->parent = prev;
              td = q->dist;
              q->dist = prevdist;
              prevdist = td;
              prev = q;
              q = q->left;
           }
        else
           {
              q->right = q->parent;
              q->parent = prev;
              td = q->dist;
              q->dist = prevdist;
              prevdist = td;
              prev = q;
              q = q->right;
           }
    }

/*
   remove the old root node
*/
   q = prev;
   if (q->left == NULL)
      {
         dist = q->dist;
         q = q->right;
         q->dist += dist;
         q->parent = prev->parent;
         if (prev->parent->left == prev)
            prev->parent->left = q;
         else
            prev->parent->right = q;
         prev->right = NULL;
      }
   else
      {
         dist = q->dist;
         q = q->left;
         q->dist += dist;
         q->parent = prev->parent;
         if (prev->parent->left == prev)
            prev->parent->left = q;
         else
            prev->parent->right = q;
         prev->left = NULL;
      }

   return(newp);
}

static float calc_root_mean(treeptr root, float *maxdist)
{
   float dist , lsum = 0.0, rsum = 0.0, lmean,rmean,diff;
   treeptr p;
   sint i;
   sint nl, nr;
   sint direction;
/*
   for each leaf, determine whether the leaf is left or right of the root.
*/
   dist = (*maxdist) = 0;
   nl = nr = 0;
   for (i=0; i< numseq; i++)
     {
         p = lptr[i];
         dist = 0.0;
         while (p->parent != root)
           {
               dist += p->dist;
               p = p->parent;
           }
         if (p == root->left) direction = LEFT;
         else direction = RIGHT;
         dist += p->dist;

         if (direction == LEFT)
           {
             lsum += dist;
             nl++;
           }
         else
           {
             rsum += dist;
             nr++;
           }
        if (dist > (*maxdist)) *maxdist = dist;
     }

   lmean = lsum / nl;
   rmean = rsum / nr;

   diff = lmean - rmean;
   return(diff);
}


static float calc_mean(treeptr nptr, float *maxdist, sint nseqs)
{
   float dist , lsum = 0.0, rsum = 0.0, lmean,rmean,diff;
   treeptr p, *path2root;
   float *dist2node;
   sint depth = 0, i,j , n = 0;
   sint nl , nr;
   sint direction, found;

	path2root = (treeptr *)ckalloc(nseqs * sizeof(treeptr));
	dist2node = (float *)ckalloc(nseqs * sizeof(float));
/*
   determine all nodes between the selected node and the root;
*/
   depth = (*maxdist) = dist = 0;
   nl = nr = 0;
   p = nptr;
   while (p != NULL)
     {
         path2root[depth] = p;
         dist += p->dist;
         dist2node[depth] = dist;
         p = p->parent;
         depth++;
     }
 
/*
   *nl = *nr = 0;
   for each leaf, determine whether the leaf is left or right of the node.
   (RIGHT = descendant, LEFT = not descendant)
*/
   for (i=0; i< numseq; i++)
     {
       p = lptr[i];
       if (p == nptr)
         {
            direction = RIGHT;
            dist = 0.0;
         }
       else
         {
         direction = LEFT;
         dist = 0.0;
/*
   find the common ancestor.
*/
         found = FALSE;
         n = 0;
         while ((found == FALSE) && (p->parent != NULL))
           {
               for (j=0; j< depth; j++)
                 if (p->parent == path2root[j])
                    { 
                      found = TRUE;
                      n = j;
                    }
               dist += p->dist;
               p = p->parent;
           }
         if (p == nptr) direction = RIGHT;
         }

         if (direction == LEFT)
           {
             lsum += dist;
             lsum += dist2node[n-1];
             nl++;
           }
         else
           {
             rsum += dist;
             nr++;
           }

        if (dist > (*maxdist)) *maxdist = dist;
     }

	dist2node=ckfree((void *)dist2node);
	path2root=ckfree((void *)path2root);
	
   lmean = lsum / nl;
   rmean = rsum / nr;
   
   diff = lmean - rmean;
   return(diff);
}

static void order_nodes(void)
{
   sint i;
   treeptr p;

   for (i=0; i<numseq; i++)
     {
        p = lptr[i];
        while (p != NULL)
          {
             p->order++;
             p = p->parent;
          }
     }
}


static sint calc_weight(sint leaf)
{

  treeptr p;
  float weight = 0.0;

  p = olptr[leaf];
  while (p->parent != NULL)
    {
       weight += p->dist / p->order;
       p = p->parent;
    }

  weight *= 100.0;

  return((sint)weight);

}

static void group_seqs(treeptr p, sint *next_groups, sint nseqs)
{
    sint i;
    sint *tmp_groups;

    tmp_groups = (sint *)ckalloc((nseqs+1) * sizeof(sint));
    for (i=0;i<nseqs;i++)
         tmp_groups[i] = 0;

    if (p->left != NULL)
      {
         if (p->left->leaf == NODE)
            {
               group_seqs(p->left, next_groups, nseqs);
               for (i=0;i<nseqs;i++)
                 if (next_groups[i] != 0) tmp_groups[i] = 1;
            }
         else
            {
               mark_group1(p->left, tmp_groups, nseqs);
            }
               
      }

    if (p->right != NULL)
      {
         if (p->right->leaf == NODE)
            {
               group_seqs(p->right, next_groups, nseqs);
               for (i=0;i<nseqs;i++)
                    if (next_groups[i] != 0) tmp_groups[i] = 2;
            }
         else 
            {
               mark_group2(p->right, tmp_groups, nseqs);
            }
         save_set(nseqs, tmp_groups);
      }
    for (i=0;i<nseqs;i++)
      next_groups[i] = tmp_groups[i];

    tmp_groups=ckfree((void *)tmp_groups);

}

static void mark_group1(treeptr p, sint *groups, sint n)
{
    sint i;

    for (i=0;i<n;i++)
       {
         if (olptr[i] == p)
              groups[i] = 1;
         else
              groups[i] = 0;
       }
}

static void mark_group2(treeptr p, sint *groups, sint n)
{
    sint i;

    for (i=0;i<n;i++)
       {
         if (olptr[i] == p)
              groups[i] = 2;
         else if (groups[i] != 0)
              groups[i] = 1;
       }
}

static void save_set(sint n, sint *groups)
{
    sint i;

    for (i=0;i<n;i++)
      sets[nsets+1][i+1] = groups[i];
    nsets++;
}



sint calc_similarities(sint nseqs)
{
   sint depth = 0, i,j, k, n;
   sint found;
   sint nerrs, seq1[MAXERRS],seq2[MAXERRS];
   treeptr p, *path2root;
   float dist;
   float *dist2node, bad_dist[MAXERRS];
   double **dmat;
   char err_mess[1024],err1[MAXLINE],reply[MAXLINE];

   path2root = (treeptr *)ckalloc((nseqs) * sizeof(treeptr));
   dist2node = (float *)ckalloc((nseqs) * sizeof(float));
   dmat = (double **)ckalloc((nseqs) * sizeof(double *));
   for (i=0;i<nseqs;i++)
     dmat[i] = (double *)ckalloc((nseqs) * sizeof(double));

   if (nseqs >= 2)
    {
/*
   for each leaf, determine all nodes between the leaf and the root;
*/
      for (i = 0;i<nseqs; i++)
       { 
          depth = dist = 0;
          p = olptr[i];
          while (p != NULL)
            {
                path2root[depth] = p;
                dist += p->dist;
                dist2node[depth] = dist;
                p = p->parent;
                depth++;
            }
 
/*
   for each pair....
*/
          for (j=0; j < i; j++)
            {
              p = olptr[j];
              dist = 0.0;
/*
   find the common ancestor.
*/
              found = FALSE;
              n = 0;
              while ((found == FALSE) && (p->parent != NULL))
                {
                    for (k=0; k< depth; k++)
                      if (p->parent == path2root[k])
                         { 
                           found = TRUE;
                           n = k;
                         }
                    dist += p->dist;
                    p = p->parent;
                }
   
              dmat[i][j] = dist + dist2node[n-1];
            }
        }

		nerrs = 0;
        for (i=0;i<nseqs;i++)
          {
             dmat[i][i] = 0.0;
             for (j=0;j<i;j++)
               {
                  if (dmat[i][j] < 0.01) dmat[i][j] = 0.01;
                  if (dmat[i][j] > 1.0) {
                  	if (dmat[i][j] > 1.1 && nerrs<MAXERRS) {
                  		seq1[nerrs] = i;
                  		seq2[nerrs] = j;
                  		bad_dist[nerrs] = dmat[i][j];
                  		nerrs++;
                  	}
                    dmat[i][j] = 1.0;
                  }
               }
          }
        if (nerrs>0) 
          {
             strcpy(err_mess,"The following sequences are too divergent to be aligned:\n");
             for (i=0;i<nerrs && i<5;i++)
              {
             	sprintf(err1,"           %s and %s (distance %1.3f)\n",
             	                        names[seq1[i]+1],
					names[seq2[i]+1],bad_dist[i]);
             	strcat(err_mess,err1);
              }
	     strcat(err_mess,"(All distances should be between 0.0 and 1.0)\n");
	     strcat(err_mess,"This may not be fatal but you have been warned!\n");
             strcat(err_mess,"SUGGESTION: Remove one or more problem sequences and try again");
             if(interactive) 
             	    (*reply)=prompt_for_yes_no(err_mess,"Continue ");
             else (*reply) = 'y';
             if ((*reply != 'y') && (*reply != 'Y'))
                    return((sint)0);
          }
     }
   else
     {
        for (i=0;i<nseqs;i++)
          {
             for (j=0;j<i;j++)
               {
                  dmat[i][j] = tmat[i+1][j+1];
               }
          }
     }

   path2root=ckfree((void *)path2root);
   dist2node=ckfree((void *)dist2node);
   for (i=0;i<nseqs;i++)
     {
        tmat[i+1][i+1] = 0.0;
        for (j=0;j<i;j++)
          {
             tmat[i+1][j+1] = 100.0 - (dmat[i][j]) * 100.0;
             tmat[j+1][i+1] = tmat[i+1][j+1];
          }
     }

   for (i=0;i<nseqs;i++) dmat[i]=ckfree((void *)dmat[i]);
   dmat=ckfree((void *)dmat);

   return((sint)1);
}

