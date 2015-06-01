/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: SmallRingLinkages.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.7 $	$Date: 2010/12/16 04:08:39 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * A SmallRingLinkages object contains a list of edges which lie on paths
 * connecting (orientated) SmallRings.
 *
 ***************************************************************************/
#ifndef SMALLRINGLINKAGE_H
#define SMALLRINGLINKAGE_H

#include "SmallRing.h"
#include "ResizeArray.h"
#include "inthash.h"
#include "Inform.h"

/// A Linkage Path object consists of
///  - a list of atoms in the path, contained in the SmallRing
///  - the index of the start and end rings
class LinkagePath {
public:
  SmallRing path;
  int start_ring, end_ring;
  
  LinkagePath(void) : path(), start_ring(-1), end_ring(-1) {}
  LinkagePath(SmallRing &p_path, int p_start_ring, int p_end_ring) {
    path = p_path;
    start_ring = p_start_ring;
    end_ring = p_end_ring;
  }

  int num(void) { return path.num(); }
  int operator [](int i) { return path[i]; }
  void append(int i) { path.append(i); }
  void remove_last(void) { path.remove_last(); }

  LinkagePath* copy(void) {
    LinkagePath *pathcopy;
    pathcopy = new LinkagePath(*path.copy(),start_ring,end_ring);
    return pathcopy;
  }

  friend Inform& operator << (Inform &os, LinkagePath &lp) {
    os << lp.path << " [" << lp.start_ring << "," << lp.end_ring << "]";
    return os;
  }
};

/// A Linkage Edge object consists of:
///  - left_atom (smaller atom id), right_atom (larger atom id)
///  - list of linkage paths which include the edge
class LinkageEdge {
public:
   int left_atom, right_atom;
   ResizeArray<LinkagePath *> paths;

   LinkageEdge(int p_atom_left, int p_atom_right) : paths (1) {
     left_atom = p_atom_left;
     right_atom = p_atom_right;
   }
   
   void addPath(LinkagePath &lp) {
     paths.append(&lp);
   }   
   
   friend Inform& operator << (Inform &os, LinkageEdge &le) {
     os << "(" << le.left_atom << "," << le.right_atom << ")";
     return os;
   }
};

/// A SmallRingLinkages object contains a list of edges which lie on paths
/// connecting (orientated) SmallRings.
class SmallRingLinkages {
private:
  inthash_t *edges_to_links;

public:
  ResizeArray<LinkageEdge *> links;
  ResizeArray<LinkagePath *> paths;

  SmallRingLinkages(void) : links(1) {
    edges_to_links = new inthash_t;
    // TODO: Adjust the initial number of hash buckets?
    inthash_init(edges_to_links,64);
  }

  ~SmallRingLinkages(void) {
    inthash_destroy(edges_to_links);
    delete edges_to_links;
  }

  void clear(void) {
    links.clear();
    paths.clear();
    inthash_destroy(edges_to_links);
    // TODO: Adjust the initial number of hash buckets?
    inthash_init(edges_to_links,64);
  }
  
  void addLinkagePath(LinkagePath &lp) {
    int i, atom_left, atom_right;
    LinkageEdge *link;
    atom_right = lp.path[0];

    for (i=1;i<lp.path.num();i++) {
      atom_left = atom_right;
      atom_right = lp.path[i];
      link = getLinkageEdge(atom_left,atom_right);
      link->addPath(lp);
    }
    
    paths.append(&lp);
  }

  bool sharesLinkageEdges(LinkagePath &lp) {
    int i, atom_left, atom_right;
    LinkageEdge *link;
    atom_right = lp.path[0];

    for (i=1;i<lp.path.num();i++) {
      atom_left = atom_right;
      atom_right = lp.path[i];
      link = getLinkageEdge(atom_left,atom_right);
      if (link->paths.num() > 1)
        return true;
    }
    
    return false;
  }

  LinkageEdge* getLinkageEdge(int atom_left, int atom_right) {
    int key, link_idx;
    LinkageEdge *link;
    
    order_edge_atoms(atom_left,atom_right);
    key = get_link_key(atom_left,atom_right);

    link_idx = inthash_lookup(edges_to_links, key);
    if (link_idx == HASH_FAIL) {
      link = new LinkageEdge(atom_left,atom_right);
      links.append(link);
      link_idx = links.num()-1;
      inthash_insert(edges_to_links,key,link_idx);
    }

    return links[link_idx];
  }

  void order_edge_atoms(int& atom_left, int& atom_right) {
    int t;
    if (atom_left > atom_right) {
       t = atom_left;
       atom_left = atom_right;
       atom_right = t;
    }
  }

  int get_link_key(int al, int ar) {
    // al - left atom id
    // ar - right atom id
    // triangular mapping of pairs to single numbers
    return 1 + ar*(ar+1)/2 + al*(al+1)/2 + (al+1)*ar;
  }
     
  friend Inform& operator << (Inform &os, SmallRingLinkages &srl) {
    int i, len;
    
    os << "SmallRingLinkages:\n";

    os << "Links:\n";
    len = srl.links.num();
    for (i=0; i < len; i++)
        os << "\t" << *(srl.links[i]) << "\n";

    os << "Paths:\n";
    len = srl.paths.num();
    for (i=0; i < len; i++)
        os << "\t" << *(srl.paths[i]) << "\n";
    
    return os;
  }
};

#endif
