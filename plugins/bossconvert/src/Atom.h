#ifndef _ATOM_H_
#define _ATOM_H_
#include <list>
#include <string>
using namespace std;

struct Bond;
struct Angle;
struct Dihedral;
struct Improper;

class Atom{
public:
	Atom(string, string, string, float, float);
	Atom();
	bool operator< (const Atom& other) const;
	bool operator== (const Atom& other) const;
	string name;
	char symbol;
	string residue;
	string typeCHARMM;
	string typeOPLSAA;
	string typeAlias;
	float mass;
	float sigma;
	float epsilon;
	float charge;
	int number;
	int boundTo;
	int addiBond;
	int b1, b2, b3;
	string tmp; //only used in additional impropers using the imprlist to store the improper type, has nothing to do with the atom itself
	~Atom();
private:
};

struct Bond{
	Atom *a1, *a2;
	float force;
	float distance;
	bool valid;
};

struct Angle{
	Atom *a1, *a2, *a3;
	float force;
	float angle;
};

struct Dihedral{
	Atom *a1, *a2, *a3, *a4;
	float v, theta;
	int n;
};

struct Improper{
	Atom *a1, *a2, *a3, *a4;
	float v, theta;
	int n;
};

#endif
