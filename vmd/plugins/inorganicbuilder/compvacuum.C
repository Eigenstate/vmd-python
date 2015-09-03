/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2007 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

class Vector3D;
class IndexList;
class AtomData;
class MolData;
class AtomList;
class AtomGrid;
class GridList;
class AtomGridCell;

// class Vector3D
// Operations on 3-D double vectors
//
class Vector3D {
public:
  const Vector3D operator+(const Vector3D w ) const {
    Vector3D v;
    v.x = x + w.x;
    v.y = y + w.y;
    v.z = z + w.z;
    return v;
  }

  const Vector3D operator-(const Vector3D w ) const {
    Vector3D v;
    v.x = x - w.x;
    v.y = y - w.y;
    v.z = z - w.z;
    return v;
  }
  
  const Vector3D operator*(double s) const {
    Vector3D v;
    v.x = s*x;
    v.y = s*y;
    v.z = s*z;
    return v;
  }

  double length() const { return sqrt(x*x + y*y + z*z); }
  double length2() const { return x*x + y*y + z*z; }
  double x,y,z;
};

Vector3D operator*(double s, Vector3D v) {
  v.x *= s;
  v.y *= s;
  v.z *= s;
  return v;
}

Vector3D operator/(Vector3D v, double s) {
  v.x /= s;
  v.y /= s;
  v.z /= s;
  return v;
}

// class IndexList
// A grow-able list of integers, for holding indices of atoms
//
class IndexList {
public:
  IndexList() {
    num=0;
    maxnum=16;
    list = new int[16];
  }
  
  ~IndexList() {
    if (list != 0)
      delete [] list;
  }
  
  void add(const int val) {
    // If we need more space, allocate a new block that is 1.5 times larger
    // and copy everything over
    if (num == maxnum) {
      int oldmax = maxnum;
      maxnum = (int)(maxnum * 1.5 + 1);
      int* oldlist = list;
      list = new int[maxnum];
      int i;
      for(i=0;i<num;i++) {
        list[i] = oldlist[i];
      }
      delete [] oldlist;
    }
    
    // We should have enough space now, add the value
    list[num] = val;
    num++;
  }

  int size() {
    return num;
  }
  
  int get(const int i) const { return list[i]; }

private:
  int num;
  int maxnum;
  int* list;
};

// class AtomData
// Store the atom index and coordinate in one object
//
class AtomData {
public:
  int index;
  Vector3D coord;
};

// class MolData
// Class for storing and reading in the atom coordinates being processed.
// It also contains the routine for determining surface atoms, once the
// vacuum-grid has been built
//
class MolData {
public:
  MolData() {
    coords = 0;
  }
  
  ~MolData() {
    if ( coords != 0) {
      delete [] coords;
    }
  }
  
  static MolData* readFile(const char* fname,
                           double gridspacing, double radius, double threshold);

  void wrapCoords(const AtomGrid* grid);
  
  AtomList* findAtoms(const AtomGrid* grid);
  AtomList* findAtomsGrid(const AtomGrid* grid);
  double getRadius() const { return radius; }
  double getActualRadius() const { return actual_radius; }

  double gridspacing;
  double radius;
  double threshold;
  double actual_radius;
  Vector3D origin;
  Vector3D basisa;
  Vector3D basisb;
  Vector3D basisc;
  int count;
  AtomData* coords;
};

// class AtomList
// Stores the atoms that are part of the shell, and writes their indices out
// at the end
//
class AtomList {
public:
  AtomList(const int sz) {
    size = sz;
    index_list = new int[sz];
    count = 0;
  }
  
  ~AtomList() {
    if (index_list != 0) {
      delete [] index_list;
    }
  }
  
  void add(const int index) { index_list[count] = index; count++; }
  
  void storeFile(const char* fname) const 
  {
    FILE* outf = fopen(fname,"w");
    if (outf == NULL) {
      printf("Couldn't open file %s\n",fname);
      exit(-1);
    }
    
    int i;
    for(i=0; i < count; i++) {
      if (fprintf(outf,"%d\n",index_list[i]) < 0) {
        printf("Store file error %d %s\n",i,fname);
        exit(-2);
      }
    }
    
    fclose(outf);
    return;
  }
  
private:
  int* index_list;
  int count;
  int size;
};

// class GridList
// Store a list of grid points, with each grid point identified as a 3-D vector
// of ints
//
class GridList {
public:
  GridList(AtomGrid* thegrid, const int gridn)
  {
    n = gridn;
    if (n != 0)
      items = new Vector3D[n];
    else
      items = 0;
      
    indx = 0;
    grid = thegrid;
    shuffle_vec = scramble_array(n);
  }
  
  ~GridList() {
    if (items != 0)
      delete [] items;
    if (shuffle_vec != 0)
      delete [] shuffle_vec;
  }
  
  int add(const int i, const int j, const int k);
  
  Vector3D get(const int i) const {
    if (i < 0 || i >= n)
      throw -1;
      
    return items[i];
  }
  
  int getNum() const { return indx; }
  
private:
  int n;
  int indx;
  AtomGrid* grid;
  int* shuffle_vec;
  Vector3D* items;
  static int* scramble_array(const int n);
};

// class AtomGridCell
// Divide the grid into a mesh of cubic cells. Each cell contains all the grid
// points for that region, and can produce a list of "set" grid points on
// demand
//
class AtomGridCell {
public:
  AtomGridCell(AtomGrid* in_ag, int in_mina, int in_maxa, int in_minb, 
               int in_maxb, int in_minc, int in_maxc)
  {
//    std::cout << "New AtomGridCell " << mina << "," << maxa << ","
//      << minb << "," << maxb << ","
//      << minc << "," << maxc << ","
//      << std::endl;

    ag = in_ag;
    mina = in_mina;
    maxa = in_maxa;
    minb = in_minb;
    maxb = in_maxb;
    minc = in_minc;
    maxc = in_maxc;
    
    na = maxa-mina+1;
    nb = maxb-minb+1;
    nc = maxc-minc+1;
    
    noneset = true;
    getlistcalled = false;
    cellmap = 0;
    count=0;
    gridlist=0;
  }
  
  ~AtomGridCell()
  {
    if (cellmap != 0)
      delete [] cellmap;
  }

  int set(const int i, const int j, const int k);
  bool get(const int i, const int j, const int k) const;
  const GridList* get_list();
  
private:
  void build_list();
  AtomGrid* ag;
  int mina;
  int maxa;
  int minb;
  int maxb;
  int minc;
  int maxc;
  int na;
  int nb;
  int nc;
  bool noneset;
  bool getlistcalled;
  bool* cellmap;
  int count;
  GridList* gridlist;
};

// class AtomGrid
// Build and maintain a grid of boolean values for the entire space
//
class AtomGrid {
public:
  AtomGrid(const MolData* mdata);

  ~AtomGrid() {
    int i,j,k;
    for(i=0; i < cna; i++)
      for(j=0; j < cnb; j++)
        for(k=0; k < cnc; k++) {
          delete cellgrid[((i*cnb)+j)*cnc+k];
        }
    delete [] cellgrid;
    delete [] celli;
    delete [] cellj;
    delete [] cellk;
  }

  // get(const int i, const int j, const int k)
  // Returns the state (true/false) of grid point (i,j,k)
  //
  bool get(const int i, const int j, const int k) const 
  {
    // Find which cell we need, then get the value from that cell
    bool ret;
    if (i >= 0 && i < na && j >= 0 && j < nb && k >= 0 && k < nc) {
      const int ii = celli[i];
      const int jj = cellj[j];
      const int kk = cellk[k];
      const int indx = ((ii*cnb)+jj)*cnc+kk;
      ret = cellgrid[indx]->get(i,j,k);
    } else {
      ret = false;
    }
    return ret;
  }
  
  // set(const int i, const int j, const int k)
  // Set grid point (i,j,k) true. Default is false
  //
  int set(const int i, const int j, const int k) {
//    std::cout << "set called " << i << "," << j << "." << k << std::endl;
    if ( i < 0 || i >= na || j < 0 || j >= nb || k < 0 || k >= nc )
      return -1;
      
    const int ii = celli[i];
    const int jj = cellj[j];
    const int kk = cellk[k];
    return cellgrid[((ii*cnb)+jj)*cnc+kk]->set(i,j,k);
  }
  
  // get_ijk(const Vector3D vv) const
  // Convert the real coordinates vv to grid coordinate space.
  // The grid coordinates may have a decimal component, indicating
  // where it falls within the grid block.
  //
  Vector3D get_ijk(const Vector3D vv) const 
  {
    Vector3D ijk;
    const Vector3D v = vv - origin;
//    std::cout << "Pos = " << v.x << " " << v.y << " " << v.z
//               << "---" << vv.x << " " << vv.y << " " << vv.z << std::endl;
    
    ijk.x = (inv_abc[0][0] * v.x + inv_abc[0][1] * v.y + inv_abc[0][2] * v.z)
             * na;
    ijk.y = (inv_abc[1][0] * v.x + inv_abc[1][1] * v.y + inv_abc[1][2] * v.z)
             * nb;
    ijk.z = (inv_abc[2][0] * v.x + inv_abc[2][1] * v.y + inv_abc[2][2] * v.z)
             * nc;
    
    return ijk;
  }

  // get_cijk(const Vector3D vv, int& i, int& j int& k) const
  // Convert the real coordinates vv to grid coordinate space, then determine
  // the coordinates of the cell which owns that grid point, and return
  // those coordinates in i, j, k
  //
  void get_cijk(const Vector3D vv, int& i, int& j, int& k) const 
  {
    Vector3D ijk = get_ijk(vv);
    
    ijk.x = fmod(ijk.x,na);
    if (ijk.x < 0)
      ijk.x += na;
    ijk.y = fmod(ijk.y,nb);
    if (ijk.y < 0)
      ijk.y += nb;
    ijk.z = fmod(ijk.z,nc);
    if (ijk.z < 0)
      ijk.z += nc;
      
    i = celli[(int)(floor(ijk.x))];
    j = cellj[(int)(floor(ijk.y))];
    k = cellk[(int)(floor(ijk.z))];
    return;
  }

  // get_xyz(const Vector3D ijk) const
  // Convert the grid coordinates ijk to real coordinates
  // The grid coordinates may have a decimal component, indicating
  // where it falls within the grid block.
  //
  Vector3D get_xyz(const Vector3D ijk) const
  {
    Vector3D v = grida * ijk.x / na + gridb * ijk.y / nb + gridc * ijk.z / nc 
                 + origin;
    return v;
  }
  
  // get_xyz(const int i, const int j, const int k) const
  // Convert the grid coordinates ijk to real coordinates
  // The real coordinates returned are for the center of the grid block
  //
  Vector3D get_xyz(const int i, const int j, const int k) const 
  {
    const double id = (i+0.5) / na;
    const double jd = (j+0.5) / nb;
    const double kd = (k+0.5) / nc;

    Vector3D v = grida * id + gridb * jd + gridc * kd + origin;
    return v;
  }
  
  // get_cell(const int i, const int j, const int k) const
  // Convert the grid coordinates ijk to real coordinates
  // The real coordinates returned are for the center of the grid block
  //
  const GridList* get_cell(const int i, const int j, const int k) const
  {
    if (i>=0 && i<cna && j>=0 && j<cnb && k>=0 && k<cnc) {
      return cellgrid[((i*cnb)+j)*cnc+k]->get_list();
    } 
    return 0;
  }
  
  Vector3D wrap_xyz(const Vector3D xyz) const
  {
    Vector3D gridcoord = get_ijk(xyz);
    Vector3D gc2 = gridcoord;
    Vector3D i = gridcoord;
    gridcoord.x = fmod(gridcoord.x,na);
    if (gridcoord.x < 0)
      gridcoord.x += na;
    gridcoord.y = fmod(gridcoord.y,nb);
    if (gridcoord.y < 0)
      gridcoord.y += nb;
    gridcoord.z = fmod(gridcoord.z,nc);
    if (gridcoord.z < 0)
      gridcoord.z += nc;
      
    const Vector3D ret = get_xyz(gridcoord);
//    if (gc2.x != gridcoord.x || gc2.y != gridcoord.y || gc2.z != gridcoord.z) {
//      printf("gc2 %f %f %f grid %f %f %f ret %f %f %f\n",
//        gc2.x,gc2.y,gc2.z,gridcoord.x,
//        gridcoord.y,gridcoord.z,ret.x,ret.y,ret.z);
//    }
    return ret;
  }
  
  int get_na() const { return na; }
  int get_nb() const { return nb; }
  int get_nc() const { return nc; }
  
  int get_cna() const { return cna; }
  int get_cnb() const { return cnb; }
  int get_cnc() const { return cnc; }
  
  Vector3D findAtomBox(const double radius) const;
  void build(const MolData* mol_data);
  void store(char* fname);
  void print();
  static int* scramble_array(const int n);

private:
  void Inverse3(const double m[3][3], double inv[3][3]) const;
  double Cofactor3(const double mat[3][3],const int i, const int j) const;
  

  Vector3D origin;
  Vector3D grida;
  Vector3D gridb;
  Vector3D gridc;
  double inv_abc[3][3];
  int na;
  int nb;
  int nc;

  int cna;
  int cnb;
  int cnc;
  AtomGridCell** cellgrid;
  int* celli;
  int* cellj;
  int* cellk;
};

MolData* MolData::readFile(const char* fname,
                           double gridspacing, double radius, double threshold)
{
  MolData *data = new MolData;
  data->gridspacing = gridspacing;
  data->radius = radius;
  data->threshold = threshold;
  data->actual_radius = radius+threshold;

   
  FILE* inpf = fopen(fname,"r");
  if (inpf == NULL) {
    printf("Couldn't open file %s\n",fname);
    exit(-1);
  }
  
  int n_read=0;
  n_read += fscanf(inpf,"%lf %lf %lf",
    &(data->origin.x),&(data->origin.y),&(data->origin.z)); 
  n_read += fscanf(inpf,"%lf %lf %lf",
    &(data->basisa.x),&(data->basisa.y),&(data->basisa.z)); 
  n_read += fscanf(inpf,"%lf %lf %lf",
    &(data->basisb.x),&(data->basisb.y),&(data->basisb.z)); 
  n_read += fscanf(inpf,"%lf %lf %lf",
    &(data->basisc.x),&(data->basisc.y),&(data->basisc.z));
  n_read += fscanf(inpf,"%d",&(data->count));
  
  if (n_read != 13) {
    printf("Error reading header (%d)",n_read);
    exit(-2);
  }
  
  data->coords = new AtomData[data->count];
  int i;
  for(i=0; i<data->count; i++) {
    n_read = fscanf(inpf,"%d %lf %lf %lf",
                    &(data->coords[i].index),
                    &(data->coords[i].coord.x),
                    &(data->coords[i].coord.y),
                    &(data->coords[i].coord.z));
    if (n_read != 4) {
      printf("Error reading atom %d (%d)",i,n_read);
      exit(-2);
    }
  }
  fclose(inpf);
  return data;
}

void MolData::wrapCoords(const AtomGrid* grid)
{
  int i;
  for(i=0; i < count; i++) {
    coords[i].coord = grid->wrap_xyz(coords[i].coord);
  }
}

int GridList::add(const int i, const int j, const int k) 
{
  if (indx >= n)
    return -1;
  
  Vector3D ii;
  const int i2 = shuffle_vec[indx];
  items[i2].x = i;
  items[i2].y = j;
  items[i2].z = k;
  indx++;
  return indx-1;
}

int AtomGridCell::set(const int i, const int j, const int k) 
{
  if (i<mina || i>maxa || j<minb || j>maxb || k<minc || k>maxc ) {
    return -1;
  }
  if (getlistcalled) {
    return -2;
  }

  if (noneset) {
    cellmap = new bool[na*nb*nc];
    int m;
    for(m=na*nb*nc-1; m >= 0; m--)
      cellmap[m] = false;
    noneset = false;
//    std::cout << "Allocated cell " << i << "," << j << "," << k << std::endl;
  }

  const int ii = i - mina;
  const int jj = j - minb;
  const int kk = k - minc;
  const int indx = ((ii*nb)+jj)*nc+kk;
  if (!cellmap[indx]) {
    count++;
  } 
  cellmap[indx] = true;

  return 0;
}
  
  
bool AtomGridCell::get(const int i, const int j, const int k) const
{
//  std::cout << "AtomGridCell::get " << i << "," <<mina << "," << maxa 
//    << std::endl;
    
  if (i<mina || i>maxa || j<minb || j>maxb || k<minc || k>maxc ) {
    return -1;
  }
  if (noneset)
    return false;
    
  const int ii = i - mina;
  const int jj = j - minb;
  const int kk = k - minc;
  return cellmap[((ii*nb)+jj)*nc+kk];
}

const GridList* AtomGridCell::get_list()
{
  if (!getlistcalled) {
    build_list();
  }
  return gridlist;
}
  
void AtomGridCell::build_list()
{
  // Build a list containing only vacuum cells
  getlistcalled = true;
  int sz = na*nb*nc;
  bool* vac = new bool[sz];
  int i;
  for (i=0; i < sz; i++)
    vac[i] = false;
    
  int tot=0;
  int grid_idx = 0;
  for(i=mina; i <= maxa; i++) {
  //  std::cout << i << std::endl;
    int j;
    for(j=minb; j <= maxb; j++) {
//      std::cout << j << "," << minc << "-" << maxc << ":";
      int k;
      for(k=minc; k <= maxc; grid_idx++, k++) {
        if (!get(i,j,k)) {
          // Optimization:
          // If all neighbors are also vacuum, then don't add this one
          // since its an interior cell
          bool all_vac = true;
          if (i > mina && i < maxa 
              && j > minb && j < maxb
              && k > minc && k < maxc) {
            int ii;
            for(ii=-1; ii <= 1; ii++) {
              int jj;
              for(jj=-1; jj <= 1; jj++) {
                int kk;
                for(kk=-1; kk <= 1; kk++) {
                  if (get(i+ii,j+jj,k+kk)) {
//                    std::cout << "Not storing " << i << "," << j << "," << k 
//                      << std::endl;
                    all_vac = false;
                    break;
                  }
                }
                if (!all_vac)
                  break;
              }
              if (!all_vac)
                break;
            }
          } else {
            all_vac = false;
          }
          if (!all_vac) {
//            std::cout << "Storing " << i << "," << j << "," << k <<std::endl;
            vac[grid_idx] = true;
            tot++;
//            std::cout << "1";
          } // else std::cout << "2";
        } // else std::cout << "0";
      }
//      std::cout << std::endl;
    }
  }
  gridlist = new GridList(ag,tot);
  grid_idx = 0;
  int gi;
  for(gi=mina; gi <= maxa; gi++) {
    int gj;
    for(gj=minb; gj <= maxb; gj++) {
      int gk;
      for(gk=minc; gk <= maxc; grid_idx++, gk++) {
        if (vac[grid_idx]) {
//          std::cout << "Adding " << i << "," << j << "," << k <<std::endl;
          gridlist->add(gi,gj,gk);
        }
      }
    }
  }
//  std::cout << "Gridlist storing " << tot << " of " << sz 
//    << " elements noneset=" << noneset << " count=" << count << std::endl;
  delete [] vac;
}
 
AtomGrid::AtomGrid(const MolData* mdata)
{
  // We'll make a grid with rectangular cells of the specified size
  // with a corner at the origin. This means the grid may not exactly
  // fit with the periodic box
  origin = mdata->origin;
  grida = mdata->basisa;
  gridb = mdata->basisb;
  gridc = mdata->basisc;
  na = (int)ceil(mdata->basisa.length() / mdata->gridspacing);
  nb = (int)ceil(mdata->basisb.length() / mdata->gridspacing);
  nc = (int)ceil(mdata->basisc.length() / mdata->gridspacing);
  
  // We'll make a grid with rectangular cells of the specified size
  // with a corner at the origin. This means the grid may not exactly
  // fit with the periodic box
  printf("Grid %d %d %d\n",na,nb,nc);
  printf("Origin %f %f %f\n",origin.x,origin.y,origin.z);

  double a[3][3];
  
  a[0][0] = grida.x;
  a[0][1] = gridb.x;
  a[0][2] = gridc.x;
  a[1][0] = grida.y;
  a[1][1] = gridb.y;
  a[1][2] = gridc.y;
  a[2][0] = grida.z;
  a[2][1] = gridb.z;
  a[2][2] = gridc.z;
  Inverse3(a,this->inv_abc);
  
  printf("a...\n");
  printf("%f %f %f\n",a[0][0],a[0][1],a[0][2]);
  printf("%f %f %f\n",a[1][0],a[1][1],a[1][2]);
  printf("%f %f %f\n",a[2][0],a[2][1],a[2][2]);
  
  printf("inv a...\n");
  printf("%f %f %f\n",inv_abc[0][0],inv_abc[0][1],inv_abc[0][2]);
  printf("%f %f %f\n",inv_abc[1][0],inv_abc[1][1],inv_abc[1][2]);
  printf("%f %f %f\n",inv_abc[2][0],inv_abc[2][1],inv_abc[2][2]);
  
  // Find how many cells to divide this into
  Vector3D cellsz = findAtomBox(mdata->getActualRadius());
  printf("cellsz=%f, %f, %f\n",cellsz.x,cellsz.y,cellsz.z);

  cna = (int)(floor(na/cellsz.x));
  cnb = (int)(floor(nb/cellsz.y));
  cnc = (int)(floor(nc/cellsz.z));
  if (cna==0) {
    cna = 1;
  }
  if (cnb==0) {
    cnb = 1;
  }
  if (cnc==0) {
    cnc = 1;
  }
  
  printf("na=%d, %d, %d\n",na,nb,nc);
  printf("cna=%d, %d, %d\n",cna,cnb,cnc);
  
  // Build the cell grid, dividing grid points roughly-evenly among the cells
  cellgrid = new AtomGridCell*[cna*cnb*cnc];
  // Storing the cell boundaries will let us find the cells quickly.
  celli = new int[na+1];
  celli[na] = cna;
  cellj = new int[nb+1];
  cellj[nb] = cnb;
  cellk = new int[nc+1];
  cellk[nc] = cnc;

  int* gridi = new int[cna+1];
  gridi[cna] = na;
  int* gridj = new int[cnb+1];
  gridj[cnb] = nb;
  int* gridk = new int[cnc+1];
  gridk[cnc] = nc;

  // Build the mapping from grid point to cells
  int mina = 0;
  int maxa;
  int i;
  for(i=0; i < cna; i++) {
    maxa = ((i+1) * na) / cna - 1;
//    std::cout << "minmax=" << mina << "," << maxa << std::endl;
    int ii;
    for(ii=mina; ii<= maxa; ii++) {
      celli[ii] = i;
//      std::cout << "cell[" << ii << "]=" << celli[ii] << std::endl;
    }
    gridi[i] = mina;
    int minb = 0;
    int maxb;
    int j;
    for(j=0; j < cnb; j++) {
      maxb = ((j+1) * nb) / cnb - 1;
      if (i==0) {
        for(int jj=minb; jj<= maxb; jj++)
          cellj[jj] = j;
      }
      gridj[j] = minb;
      int minc = 0;
      int maxc;
      int k;
      for(k=0; k < cnc; k++) {
        maxc = ((k+1) * nc) / cnc - 1;
        if (i==0 && j==0) {
          for(int kk=minc; kk<= maxc; kk++)
            cellk[kk] = k;
        }
        gridk[k] = minc;
        minc = maxc+1;
      }
      minb = maxb+1;
    }
    mina = maxa+1;
  }

  // Now create the cells
  int ci;
  for(ci=0; ci < cna; ci++) {
    int cj;
    for(cj=0; cj < cnb; cj++) {
      int ck;
      for(ck=0; ck < cnc; ck++) {
//        std::cout << "cellgrid[" << i <<"," << j << "," << k << "," <<   ((i*cnb)+j)*cnc + k << " addr " << cellgrid << std::endl;
        cellgrid[((ci*cnb)+cj) * cnc + ck] 
          = new AtomGridCell(this,gridi[ci],gridi[ci+1]-1,
                                  gridj[cj],gridj[cj+1]-1,
                                  gridk[ck],gridk[ck+1]-1);
      }
    }
  }
  delete[] gridi;
  delete[] gridj;
  delete[] gridk;
}
  
int* GridList::scramble_array(const int n) {
  return AtomGrid::scramble_array(n); 
}

int* AtomGrid::scramble_array(const int n)
{
  if (n < 1) return 0;
    
  int* a = new int[n];
  a[0]=n-1;
  if (n == 1) return a;
  a[1] = 0;
  if (n == 2) return a;
    
  bool* b = new bool[n];
  int i;
  for(i=0; i<n; i++)
    b[i] = true;
    
  int howmany = 1;
  int incr = (n >> 1);
//  std::cout << "n=" << n << " n>>=" << (n >> 1) << std::endl;

  int nxt=2;
  while (nxt < n) {
    int i;
    for (i=0; i < howmany; i++) {
//      std::cout << "n=" << n << " nxt=" << nxt << " i=" << i << " incr=" << incr << " howmany=" << howmany;
      int nxtval = a[i+1] + incr;
//      std::cout << "  nxtval=" << nxtval;
      if (b[nxtval])  {
        a[nxt++] = nxtval;
//        std::cout << " a[" << nxt-1 << "]=" << a[nxt-1];
        b[nxtval] = false;
      } 
//      std::cout << std::endl;
      if (nxt > n-1)
        break;
    }
   howmany <<= 1;
   incr >>= 1;
   if (incr == 0)
     break;
  }
  for(i=1; i < n; i++) {
    if (nxt >= n) break;
    if (b[i])
      a[nxt++] = i;
  }
  delete [] b;
  
  return a;
}

Vector3D AtomGrid::findAtomBox(const double radius) const
{
  // Find the bounding box for the atom radius
  Vector3D cornerx[8];
  Vector3D dx, dy, dz;
  
  const Vector3D o = origin;
  dx.y = dx.z = dy.x = dy.z = dz.x = dz.y = 0;
  dx.x = dy.y = dz.z = radius;
  cornerx[0] = o;
  cornerx[1] = o + dx;
  cornerx[2] = o + dy;
  cornerx[3] = o + dz;
  cornerx[4] = cornerx[1] + dy;
  cornerx[5] = cornerx[1] + dz;
  cornerx[6] = cornerx[2] + dz;
  cornerx[7] = cornerx[4] + dz;  
  
  // Find the max i,j,k for those corners
  Vector3D ijk = get_ijk(cornerx[0]);
  double dimax = ijk.x;
  double djmax = ijk.y;
  double dkmax = ijk.z;
  
  int i;
  for(i=1; i < 8; i++) {
    ijk = get_ijk(cornerx[i]);
    if (ijk.x > dimax)
      dimax = ijk.x;
    if (ijk.y > djmax)
      djmax = ijk.y;
    if (ijk.z > dkmax)
      dkmax = ijk.z;
  }
  Vector3D vec;
  
  vec.x = dimax;
  vec.y = djmax;
  vec.z = dkmax;

  return vec;
}

void AtomGrid::build(const MolData* mdata)
{
  Vector3D corner[8];

  // Fill the grid
  // Find the bounding box for the atom radius
  Vector3D bb = findAtomBox(mdata->getRadius());
  
  const int imax = (int)ceil(bb.x);
  const int jmax = (int)ceil(bb.y);
  const int kmax = (int)ceil(bb.z);

  printf("Box=%d, %d, %d\n",imax,jmax,kmax);
  const double rsq = mdata->getRadius() * mdata->getRadius();
  int atomi;
  for(atomi=0; atomi<mdata->count; atomi++) {
    Vector3D atom = mdata->coords[atomi].coord;
//    std::cout << "Checking atom " << atomi << "(" << atom.x << "," << atom.y << "," << atom.z <<std::endl;
    Vector3D atomijk = get_ijk(atom);
    const int ai = (int)atomijk.x;
    const int aj = (int)atomijk.y;
    const int ak = (int)atomijk.z;
    int i;
    for(i=ai-imax; i <= ai+imax; i++) {
      int ii = i;
      if (i < 0)
        ii += na;
      else if (i >= na)
        ii -= na;
      int j;
      for(j=aj-jmax; j <= aj+jmax; j++) {
        int jj = j;
        if (j < 0)
          jj += nb;
        else if (j >= nb)
          jj -= nb;
        // Do the positive half of the loop. Stop once we get to
        // the first cell that is outside the boundary
        int k;
        for(k=0; k <= kmax; k++) {
          int k_no_wrap = ak+k;
          int kk = k_no_wrap;
          if (k_no_wrap < 0)
            kk += nc;
          else if (k_no_wrap >= nc)
            kk -= nc;

          // If cell is already filled, just continue
          if (get(ii,jj,kk))
            continue;

          const Vector3D v = get_xyz(i,j,k_no_wrap);
          const Vector3D dv = v - atom;
          if (dv.length2() <= rsq) {
            set(ii,jj,kk);
          } else {
            break;
          }
        }
        for(k=1; k <= kmax; k++) {
          int k_no_wrap = ak-k;
          int kk = k_no_wrap;
          if (k_no_wrap < 0)
            kk += nc;
          else if (k_no_wrap >= nc)
            kk -= nc;

          // If cell is already filled, just continue
          if (get(ii,jj,kk))
            continue;

          const Vector3D v = get_xyz(i,j,k_no_wrap);
          const Vector3D dv = v - atom;
          if (dv.length2() <= rsq) {
            set(ii,jj,kk);
          } else {
            break;
          }
        }
      }
    }
  }
  return;
}

void AtomGrid::print()
{
  int k;
  for(k=0;k < nc; k++) {
    int i;
    for(i=0; i < na; i++) {
      printf("%d:",i);
      int j;
      for(j=0; j < nb; j++) {
        if (get(i,j,k))
          printf("1");
        else printf("0");
      }
      printf("\n");
    }
    printf("%d:-------------------------------------------------------\n",k);
  }
}

void AtomGrid::store(char* fname)
{
  FILE* outf = fopen(fname,"w");
  if (outf == NULL) {
    printf("Couldn't open file %s\n",fname);
    exit(-1);
  }
  
  int k;
  for(k=0;k < nc; k++) {
    int i;
    for(i=0; i < na; i++) {
      int j;
      for(j=0; j < nb; j++) {
        if (!get(i,j,k)) {
          Vector3D v = get_xyz(i,j,k);
          if (fprintf(outf,"%f %f %f\n",v.x,v.y,v.z) < 0) {
            printf("Store grid error %s\n",fname);
            exit(-2);
          }
        }
      }
    }
  }
    
  fclose(outf);
  return;
}

void AtomGrid::Inverse3(const double m[3][3], double inv[3][3]) const 
{
  // Get adjoint matrix : transpose of cofactor matrix
  int i,j;
  for(i=0; i < 3; i++)
    for(j=0; j < 3; j++)
      inv[i][j] = Cofactor3(m,i,j);

  // Now divide by the determinant
  const double det = m[0][0] * inv[0][0] 
                   + m[0][1] * inv[1][0]
                   + m[0][2] * inv[2][0];

//    if (det == 0) {
//      error "Non-invertible"
//    }

  for(i=0; i < 3; i++)
    for(j=0; j < 3; j++)
      inv[i][j] /= det;
      
  return;
}
  
double AtomGrid::Cofactor3(const double mat[3][3],
                           const int i, const int j) const {
  int cols[3][2] = { {1,2}, {0,2}, {0,1}};

  const int row1 = cols[j][0];
  const int row2 = cols[j][1];

  const int col1 = cols[i][0];
  const int col2 = cols[i][1];
  
  const double a = mat[row1][col1];
  const double b = mat[row1][col2];
  const double c = mat[row2][col1];
  const double d = mat[row2][col2];

  const double det = a*d-b*c;
  if (((i+j) % 2) != 0)
    return -det;
  else
    return det;
}

AtomList* MolData::findAtoms(const AtomGrid* grid) 
{
  // Find the bounding box for the selection distance
  Vector3D cornerx[8];
  Vector3D dx, dy, dz;
  
  dx.y = dx.z = dy.x = dy.z = dz.x = dz.y = 0;
  dx.x = dy.y = dz.z = getActualRadius();
  cornerx[0] = origin;
  cornerx[1] = origin + dx;
  cornerx[2] = origin + dy;
  cornerx[3] = origin + dz;
  cornerx[4] = cornerx[1] + dy;
  cornerx[5] = cornerx[1] + dz;
  cornerx[6] = cornerx[2] + dz;
  cornerx[7] = cornerx[4] + dz;  
  
  // Find the max i,j,k for those corners
  Vector3D ijk = grid->get_ijk(cornerx[0]);
  double dimax = ijk.x;
  double djmax = ijk.y;
  double dkmax = ijk.z;

  int i;  
  for(i=1; i < 8; i++) {
    ijk = grid->get_ijk(cornerx[i]);
    if (ijk.x > dimax)
      dimax = ijk.x;
    if (ijk.y > djmax)
      djmax = ijk.y;
    if (ijk.z > dkmax)
      dkmax = ijk.z;
  }
  
  const int imax = (int)ceil(dimax);
  const int jmax = (int)ceil(djmax);
  const int kmax = (int)ceil(dkmax);

  // Find the bounding box for the atom radius
  Vector3D bb = grid->findAtomBox(getActualRadius());
  
  int* ai_s = AtomGrid::scramble_array(2*imax+1);
  int* aj_s = AtomGrid::scramble_array(2*jmax+1);
  int* ak_s = AtomGrid::scramble_array(2*kmax+1);
    
//  std::cout << "Max Box=" << imax << " " << jmax << " " << kmax << std::endl;
  
  // Scan atoms and check adjacent cells
  AtomList *al = new AtomList(count);
  const double r = getActualRadius();
  const double rsq = r*r;
  int atomi;
  for(atomi=0; atomi<count; atomi++) {
    bool found = false;
    Vector3D atom = coords[atomi].coord;
    Vector3D atomijk = grid->get_ijk(atom);
    const int ai = (int)atomijk.x;
    const int aj = (int)atomijk.y;
    const int ak = (int)atomijk.z;

    const int na = grid->get_na();
    const int nb = grid->get_nb();
    const int nc = grid->get_nc();

//    for(int is=-imax; !found && is <= imax; is++) {
//      int i = ai+is;
    int is;
    for(is=0; !found && is < 2*imax+1; is++) {
      int i = ai+(ai_s[is] - imax);
      int ii = i;
      if (i < 0)
        ii += na;
      else if (i >= na)
        ii -= na;
//      for(int js=-jmax; !found && js <= jmax; js++) {
//        int j = aj+js;
      int js;
      for(js=0; !found && js < 2*jmax+1; js++) {
        int j = aj + (aj_s[js] - jmax);
        int jj = j;
        if (j < 0)
          jj += nb;
        else if (j >= nb)
          jj -= nb;
        int ks;
        for(ks=0; !found && ks <= kmax; ks++) {
          int k = ak + ks;
          int kk = k;
          if (k < 0)
            kk += nc;
          else if (kk >= nc)
            kk -= nc;

          // See if its a vacuum
          if (!grid->get(ii,jj,kk)) {
            const Vector3D v = grid->get_xyz(i,j,k);
            const Vector3D dv = v - atom;
            if (dv.length2() <= rsq) {
              al->add(coords[atomi].index);
              found = true;
            } else {
              break;
            }
          }
        }
        
        for(ks=-1; !found && ks >= -kmax; ks--) {
          int k = ak + ks;
          int kk = k;
          if (k < 0)
            kk += nc;
          else if (kk >= nc)
            kk -= nc;

          // See if its a vacuum
          if (!grid->get(ii,jj,kk)) {
            const Vector3D v = grid->get_xyz(i,j,k);
            const Vector3D dv = v - atom;
            if (dv.length2() <= rsq) {
              al->add(coords[atomi].index);
              found = true;
            } else {
              break;
            }
          }
        }
      } // j loop
    } // i loop
  }
  delete [] ai_s;
  delete [] aj_s;
  delete [] ak_s;
  return al;
}

AtomList* MolData::findAtomsGrid(const AtomGrid* grid) 
{
  // Scan atoms and check adjacent cells
  AtomList *al = new AtomList(count);
  const double r = getActualRadius();
  const double rsq = r*r;
  int atomi;
  for(atomi=0; atomi<count; atomi++) {
    Vector3D atom = coords[atomi].coord;
//    std::cout << "Checking atom " << atomi << "(" << atom.x << "," << atom.y << "," << atom.z <<std::endl;
    int i0, j0, k0;
    bool found = false;
    
    grid->get_cijk(atom,i0,j0,k0);
    int di;
    for(di=-1; !found && di <= 1; di++) {
      int ci = i0 + di;
      int cna = grid->get_cna();
      int wrapi = 0;
      if (ci<0) {
        ci += cna;
        wrapi = grid->get_na();
      } else if (ci >= cna) {
        ci -= cna;
        wrapi = -grid->get_na();
      }
      int dj;
      for(dj=-1; !found && dj <= 1; dj++) {
        int cj = j0 + dj;
        int cnb = grid->get_cnb();
        int wrapj = 0;
        if (cj<0) {
          cj += cnb;
          wrapj = grid->get_nb();
        } else if (cj >= cnb) {
          cj -= cnb;
          wrapj = grid->get_nb();
        }
        int dk;
        for(dk=-1; !found && dk <= 1; dk++) {
          int ck = k0 + dk;
          int cnc = grid->get_cnc();
          int wrapk = 0;
          if (ck<0) {
            ck += cnc;
            wrapk = grid->get_nc();
          } else if (ck >= cnc) {
            ck -= cnc;
            wrapk = -grid->get_nc();
          }
          const GridList* list = grid->get_cell(ci,cj,ck);
//          std::cout << "Found " << list->getNum() << " atoms in cell " 
//            << ci << "," << cj << "," << ck << std::endl;
          int gli;
          for(gli=0; !found && gli < list->getNum(); gli++) {
            Vector3D gridpt = list->get(gli);
            gridpt.x += wrapi;
            gridpt.y += wrapj;
            gridpt.z += wrapk;
            const Vector3D v = grid->get_xyz(gridpt);
            const Vector3D dv = v - atom;
            if (dv.length2() <= rsq) {
              al->add(coords[atomi].index);
              found = true;
//              std::cout << "Found vac " << gli << std::endl;
            }
          }
        }
      }
    }
  }
  return al;
}

int main(int argc,char *argv[])
{
  if ( argc != 6 ) {
    printf(
      "Usage: %s grid-res vac-radius selection-dist coord-file index-file\n",
      argv[0]);
    return 0;
  }
  
  double res = strtod(argv[1],NULL); // Max grid resolution
  double radius = strtod(argv[2],NULL); // vacuum-sphere/atom mask radius
  double select_dist = strtod(argv[3],NULL); // selection distance
  
  printf("Reading\n");
  MolData* mol_data = MolData::readFile(argv[4],res,radius,select_dist);
  printf("Building grid\n");
  AtomGrid* atom_grid = new AtomGrid(mol_data);
  mol_data->wrapCoords(atom_grid);
  atom_grid->build(mol_data);
  
//  atom_grid->print();
//  atom_grid->store("grid.dat");
  
  printf("Searching grid\n");
  AtomList* atom_list = mol_data->findAtomsGrid(atom_grid);
  printf("Storing\n");
  atom_list->storeFile(argv[5]);

  delete atom_list;
  delete atom_grid;
  delete mol_data;
  
  return 0;
}
