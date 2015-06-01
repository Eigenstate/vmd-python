#ifndef _PARSER_H_
#define _PARSER_H_
#include <list>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include "Atom.h"

using namespace std;

//Prototypes

void initLists();
void parseFile(string, list<vector<string> > *);
float getMass(char);
int findID(string, float, float);
float stringToFloat(string);
int stringToInt(string);
vector<string> split(const string);
int lookup();
int findBonds();
int findAngles();
int findDihedrals();
int findImpropers();
Atom * getAtom(string);
int parseOutFile(string, list<vector<string> > *);
void cleanup();

#endif //_PARSER_H_
