#include "Parser.h"
#include "AtomWeights.h"
#include "Atom.h"
#include <algorithm>

using namespace std;

/*global variables*/
list< vector <string> > parsedZFile;
list< vector <string> > parsedParFile;
list< vector <string> > parsedSBFile;
list< vector <string> > parsedOutFile;
list<vector<string> > parsedImprList;
list<Atom*> atoms;
list<string> aliases;
list<Bond*> bonds;
list<Angle*> angles;
list<Dihedral*> dihedrals;
int numResidues = 0;
//list<string> improperBonds;
//list<string> impropers;
list<Improper*> impropers;

/*prototypes*/
void initLists();
float stringToFloat(string);
int stringToInt(string);
vector<string> split(const string);
void parseFile(string, list<vector< string> > *);
float getMass(char);
int findID(string, float, float);
string printVec(vector<string>);
int lookup();
bool contains(list<string>, string);
bool contains(list<string>, string, int);
Atom * getAtom(string);
Atom * getAtom(int);
int findBonds();
int findAngles();
vector<string> getSBInfo(string);
bool parseFloat(string, float *);
bool parseInt(string, int *);
int findImpropers();
int findDihedrals();
int parseOutFile(string, list<vector<string> > *);
string printVec(vector<string>);
void cleanup();

/*definitions*/

void initLists(){
    atoms = *(new list<Atom*>());
    bonds = *(new list<Bond*>());
    angles = *(new list<Angle*>());
    dihedrals = *(new list<Dihedral*>());
    impropers = *(new list<Improper*>());
}

/*parseFile() reads a file through IO and splits each line and column by white spaces and places them into the given saveTo variable */
void parseFile(string fileName, list<vector<string> > *saveTo){
    string line;
    vector<string> parts; //parts is a vector of columns 
    ifstream file;
    file.open(fileName.c_str());

    if(!file){
        fprintf(stderr, "Error! %s could not be opened for reading.\n", fileName.c_str());
        return;
    }

    if(saveTo == &parsedSBFile){
        while(getline(file, line)){
            parts = split(line);
            if(parts.size() > 0){                
                if(parts.size() > 1){                
                    if(parts.at(1)[0] == '-' && stringToFloat(parts.at(2)) != 0.0f){
                        string part1 = parts.at(0), part2 = parts.at(1);
                        string part = part1 + " " + part2;
                        parts.erase(parts.begin(), parts.begin()+2);
                        parts.insert(parts.begin(), 1, part);
                    }
                    else if(parts.size() > 4 &&
                            parts.at(0).length() == 1 && 
                            parts.at(1).length() == 2 && 
                            parts.at(1)[0] == '-' && 
                            parts.at(2)[0] == '-' && 
                            stringToFloat(parts.at(3)) != 0.0f){
                        
                        string part1 = parts.at(0), part2 = parts.at(1), part3 = parts.at(2);
                        string part = part1 + " " + part2 + " " + part3;
                        parts.erase(parts.begin(), parts.begin()+3);
                        parts.insert(parts.begin(), 1, part);

                    }
                }
                
                    
                saveTo->push_back(parts);
            }
            }
    }
    else{
            while(getline(file, line)){
                parts = split(line);
            if(parts.size() > 0)
                saveTo->push_back(parts);
            }
    }
    file.close();

}

int parseOutFile(string fileName, list<vector<string> > *saveTo){
    ifstream file;
    file.open(fileName.c_str());

    if(!file.is_open()){
        return 0;
    }


    string line;
    vector<string> parts;

    while(getline(file, line)){
        if(line.length() == 0)
            continue;

        parts = split(line);
        if(parts.size() == 0)
            continue;

        if(parts.size() > 8){
            if(parts.at(1) == "Missing"){
                if(parts.at(2) == "Bond"){
                    if(parts.at(9).length() == 1){
                        if(parts.at(10)[0] == '-'){
                            string part1 = parts.at(9), part2 = parts.at(10);
                            string part = part1 + " " + part2;
                            parts.erase(parts.begin()+9, parts.begin()+11);
                            parts.insert(parts.begin()+9, 1, part);
                        }
                    }
                }

                if(parts.at(2) == "Angle"){
                    string splitted1 = parts.at(6).substr(0, 5), splitted2 = parts.at(6).substr(5, parts.at(6).length());
                    parts.erase(parts.begin()+6, parts.begin()+7);
                    parts.insert(parts.begin()+6, 1, splitted1);
                    parts.insert(parts.begin()+7, 1, splitted2);

                    if(parts.at(7).length() == 1){
                        if(parts.at(8)[0] == '-'){
                            string part1 = parts.at(7), part2 = parts.at(8);
                            if(part2.length() == 2){
                                if(parts.at(9)[0] == '-'){
                                    string part3 = parts.at(9);                                    
                                    parts.erase(parts.begin()+7, parts.begin()+10);
                                    string part = part1 + " " + part2 + " " + part3;
                                    parts.insert(parts.begin()+7, 1, part);
                                }
                            }
                            else{
                                parts.erase(parts.begin()+7, parts.begin()+9);
                                string part = part1 + " " + part2;
                                parts.insert(parts.begin()+7, 1, part);
                            }
                        }
                    }
                    else if(parts.at(7).length() == 4){
                        if(parts.at(8)[0] == '-'){
                            string part1 = parts.at(7), part2 = parts.at(8);
                            string part = part1 + " " + part2;
                            parts.erase(parts.begin()+7, parts.begin()+9);
                            parts.insert(parts.begin()+7, 1, part);
                        }
                    }
                }
            }
        }
        if(parts.size() > 3){
            if(parts.at(0) == "Using"){
                if(parts.at(1) == "Synonym:"){
                    if(parts.at(2).length() == 1){
                        if(parts.at(3)[0] == '-'){
                            string part1 = parts.at(2), part2 = parts.at(3);
                            if(part2.length() == 2){
                                if(parts.at(4)[0] == '-'){
                                    string part3 = parts.at(4);                                    
                                    parts.erase(parts.begin()+2, parts.begin()+5);
                                    string part = part1 + " " + part2 + " " + part3;
                                    parts.insert(parts.begin()+2, 1, part);
                                }
                            }
                            else{
                                parts.erase(parts.begin()+2, parts.begin()+4);
                                string part = part1 + " " + part2;
                                parts.insert(parts.begin()+2, 1, part);
                            }
                        }
                    }
                    else if(parts.at(2).length() == 4){
                        if(parts.at(3)[0] == '-'){
                            string part1 = parts.at(2), part2 = parts.at(3);
                            string part = part1 + " " + part2;
                            parts.erase(parts.begin()+2, parts.begin()+4);
                            parts.insert(parts.begin()+2, 1, part);
                        }
                    }
                }
            }
        }

        parsedOutFile.push_back(parts);
    }

    file.close();

    return (int)parsedOutFile.size();    
}

template<class T>
bool fromString(T& t, const string& str, ios_base& (*f)(ios_base&))
{
        stringstream ss(str);
        return !(ss >> f >> t).fail();
}

float stringToFloat(string str){
    float f;
    if(fromString<float>(f, str, std::dec)){
        return f;
    }
    else {
        //fprintf(stderr, "Error: stringToFloat() not a valid float in str: %s.\n", str.c_str());
        return 0.0f;
    }
}

int stringToInt(string str){
    int i;
    if(fromString<int>(i, str, std::dec)){
        return i;
    }
    else {
        //fprintf(stderr, "Error: stringToInt() not a valid int in str: %s.\n", str.c_str());
        return 0;
    }
}

bool parseInt(string str, int *val){
    int v;
    if(fromString<int>(v, str, std::dec)){
        *val = v;
        return true;
    }
    else
        return false;
}

bool parseFloat(string str, float *val){
    float v;
    if(fromString<float>(v, str, std::dec)){
        *val = v;
        return true;
    }
    else
        return false;
}

/*split() takes the given variable str and splits it into a vector (list) of columns by white spaces */
vector<string> split(const string str){
    string buf;
    stringstream ss(str);

    vector<string> tokens;

    while(ss >> buf){
        tokens.push_back(buf);
    }

    return tokens;
}

float getMass(char atom){
    switch(atom){
        case 'H': return H;
        case 'C': return C;
        case 'O': return O;
        case 'N': return N;
        case 'F': return F;
        case 'B': return B;
        case 'P': return P;
        case 'I': return I;
        default: return 0.0f;
    }
}

/*findID() looks into the parsed .par file for the ID field.*/
int findID(string name, float sigma, float epsilon){
    list< vector <string> >::iterator line;
    vector<string> columns;

    for(line = parsedParFile.begin(); line != parsedParFile.end(); line++){
        columns = *line;        
        if(columns.size() >= 4){
            if(columns.at(2).compare(name) == 0){
                if(stringToFloat(columns.at(4)) == sigma && stringToFloat(columns.at(5)) == epsilon){
                    return stringToInt(columns.at(0));
                }
            }
        }
    }
    return 0;
}

string printVec(vector<string> in){
    string out;    
    vector<string>::iterator filepos_it;
    for(filepos_it = in.begin(); filepos_it != in.end(); filepos_it++){
        out += *filepos_it + " ";
    }
    out += '\n';
    return out;
}

int lookup(){
    Atom *atom;
    string currRes = "";
    int al = 800; //used for atom type alias.
    int id;
    vector<string> line, line2;
    list<vector<string> >::iterator filepos_it, filepos_it2;
    for(filepos_it = parsedZFile.begin(); filepos_it != parsedZFile.end(); filepos_it++){
        line = *filepos_it;
        if(line.size() == 12 || line.size() == 11){
            if(line.at(1).compare("DUM") == 0 || line.at(1).compare("X") == 0){
                continue;
            }

            if(line.size() == 12 && line.at(10).compare(currRes) != 0){ //check if we're on a new residue and increment numResidues
                currRes = line.at(10);
                numResidues++;
            }
            atom = new Atom();
            atom->symbol = line.at(1)[0];
                        if (line.size() == 12)
                            atom->residue = line.at(10);
            for(filepos_it2 = parsedZFile.begin(); filepos_it2 != parsedZFile.end(); filepos_it2++){
                line2 = *filepos_it2;
                if(line2.size() == 6){
                    if(stringToInt(line2.at(2)) != 0){                    
                        continue;
                    }                
                    if(stringToInt(line2.at(2)) == 0 && stringToInt(line2.at(0)) != 0){
                        if(stringToInt(line.at(2)) == stringToInt(line2.at(0))){                            
                            atom->number = stringToInt(line.at(0));
                            atom->boundTo = stringToInt(line.at(4));
                            atom->b1 = stringToInt(line.at(4));
                            atom->b2 = stringToInt(line.at(6));
                            atom->b3 = stringToInt(line.at(8));                            
                            atom->typeOPLSAA = line2.at(2);                    
                            atom->mass = getMass(atom->symbol);
                            atom->sigma = stringToFloat(line2.at(4));
                            atom->epsilon = stringToFloat(line2.at(5));
                            atom->name = line.at(1);
                            atom->charge = stringToFloat(line2.at(3));
                            id = findID(atom->typeOPLSAA, atom->sigma, atom->epsilon);
                            char buf[10];
                            sprintf(buf, "%c%d", atom->symbol, id);                        
                            atom->typeCHARMM = string(buf);
                            sprintf(buf, "%c%d  %s", atom->symbol, al, atom->typeCHARMM.c_str());                            
                            aliases.push_back(string(buf));
                            sprintf(buf, "%c%d", atom->symbol, al);
                            atom->typeAlias = string(buf);
                        }
                    }
                }
            }

            atoms.push_back(atom);
            atom = NULL;
            al++;
            if(al == 900){//should there be more atoms than 100, use the upper reserved numbers.
                al = 9500;
            }
        }
    }
    int n = (int)atoms.size();
    return n;
}

Atom * getAtom(string n){
    list<Atom*>::iterator atoms_it;
    for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
        if((*atoms_it)->name == n){
            return *atoms_it;
        }
    }
    return NULL;
}

Atom * getAtom(int n){
    list<Atom*>::iterator atoms_it;
    for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
        if((*atoms_it)->number == n){
            return *atoms_it;
        }
    }
    return NULL;
}

Bond *findBond(Atom *a1, Atom *a2){ // helper function for searching the out file for bond parameters.
    Bond *b = new Bond();
    bool bondSection = false;
    list<vector<string> >::iterator filepos_it;
    for(filepos_it = parsedOutFile.begin(); filepos_it != parsedOutFile.end(); filepos_it++){
        vector<string> line = *filepos_it;        
        if(line.size() == 3 && !bondSection && line.at(0) == "Bond"){
            bondSection = true;
            continue;
        }
        if(line.size() == 3 && bondSection && line.at(0) == "Angle"){
            return NULL;
        }
        if(bondSection && line.size() >= 10){
            if((a1->number == stringToInt(line.at(0)) && a2->number == stringToInt(line.at(1))) ||
                (a1->number == stringToInt(line.at(1)) && a2->number == stringToInt(line.at(0)))){

                b->a1 = a1;
                b->a2 = a2;
                b->distance = stringToFloat(line.at(2));
                b->force = stringToFloat(line.at(3));
                b->valid = true;
                return b;
            }
        } 
        
    }
    return NULL;
}

int findBonds(){
    Atom *a1, *a2;
    Bond *bond;
    list<vector<string> >::iterator filepos_it;
    list<Atom*>::iterator atoms_it;

    for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){//loop through the atom list first to find all the listed bonds
        a1 = *atoms_it;
        if(a1->boundTo == 1 || a1->boundTo == 2){//if a1 is bonded to a dummy atom, then do not proceed.
            continue;
        }
        a2 = getAtom(a1->boundTo); //assign a2 to the atom a1 is bonded to.
        bond = findBond(a1, a2);
        if(bond != NULL){ //validate the bond
            bonds.push_back(bond); //place the bond in the bond list.
        }
    }

    for(filepos_it = parsedZFile.begin(); filepos_it != parsedZFile.end(); filepos_it++){
        vector<string> line = *filepos_it;
        if(line.size() == 2){
            a1 = getAtom(stringToInt(line.at(0)));
            a2 = getAtom(stringToInt(line.at(1)));
            a1->addiBond = stringToInt(line.at(1));
            a2->addiBond = stringToInt(line.at(0));
            if(a1 != NULL && a2 != NULL){
                bond = findBond(a1, a2);
                if(bond != NULL){
                    bonds.push_back(bond);
                }
            }
        }
    }

    return (int)bonds.size();
}

int findAngles(){
    bool angleSection = false;
    Angle *angle;
    list<vector<string> >::iterator filepos_it;
    for(filepos_it = parsedOutFile.begin(); filepos_it != parsedOutFile.end(); filepos_it++){
        vector<string> line = *filepos_it;
        if(line.size() == 3){
            if(line.at(0) == "Angle" && line.at(1) == "Bending"){
                angleSection = true;
                continue;
            }
        }
        if(line.size() == 2){
            if(line.at(0) == "Dipole" && line.at(1) == "Moment"){
                angleSection = false;
                break;
            }
        }
        if(angleSection && line.size() > 10){
            if(stringToInt(line.at(0)) != 0){
                if(stringToInt(line.at(0)) == 1 || stringToInt(line.at(0)) == 2 ||
                   stringToInt(line.at(1)) == 1 || stringToInt(line.at(1)) == 2 ||
                   stringToInt(line.at(2)) == 1 || stringToInt(line.at(2)) == 2){
                    continue;
                }
                angle = new Angle();
                angle->a1 = getAtom(stringToInt(line.at(0)));
                angle->a2 = getAtom(stringToInt(line.at(1)));
                angle->a3 = getAtom(stringToInt(line.at(2)));

                if(angle->a1 == NULL || angle->a2 == NULL || angle->a3 == NULL){
                    fprintf(stderr, "***Error! one or more atoms in an angle didn't exist, possible reason: \nthe out file used does not belong to this .z file!\nAtoms searched for: %s, %s, %s ***\n", line.at(0).c_str(), line.at(1).c_str(), line.at(2).c_str());
                    angles = *(new list<Angle*>());
                    return 0;
                }

                angle->angle = stringToFloat(line.at(3));
                angle->force = stringToFloat(line.at(4));
                angles.push_back(angle);
                
            }
        }
    }
    return (int)angles.size();
}

Bond *bondLookup(Atom *a1, Atom *a2){
    list<Bond*>::iterator bonds_it;
    for(bonds_it = bonds.begin(); bonds_it != bonds.end(); bonds_it++){
        Bond *bond = *bonds_it;
        if((bond->a1 == a1 && bond->a2 == a2) || (bond->a1 == a2 && bond->a2 == a1)){
            return bond;
        }
    }
    return NULL;
}

bool isDihedral(Atom *a1, Atom *a2, Atom *a3, Atom *a4){
    if(a1 == NULL || a2 == NULL || a3 == NULL || a4 == NULL){
        return false;
    }    
    Bond *b1 = bondLookup(a1, a2), *b2 = bondLookup(a2, a3), *b3 = bondLookup(a3, a4);
    return (b1 != NULL && b2 != NULL && b3 != NULL);
}

int findDihedrals(){
    Dihedral *dh1, *dh2, *dh3, *dh4;
    Atom *a1, *a2, *a3, *a4;
    bool dihedralSection1 = false;
    list<vector<string> >::iterator filepos_it;
    list<vector<string> >::iterator filepos_it2;
    for(filepos_it = parsedZFile.begin(); filepos_it != parsedZFile.end(); filepos_it++){
        vector<string> line = *filepos_it;
        if(line.size() >= 3){
            if(line.at(0) == "Variable" && line.at(1) == "Dihedrals"){
                dihedralSection1 = true;
                continue;
            }
            else if(line.at(0) == "Additional" && line.at(1) == "Dihedrals"){
                dihedralSection1 = false;
                break;
            }

            if(dihedralSection1 && line.size() >= 4 && stringToInt(line.at(0)) != 0 && stringToInt(line.at(1)) != 0 &&
                stringToInt(line.at(1)) != 160 && stringToInt(line.at(1)) != 161 && stringToInt(line.at(1)) != 162 && 
                stringToInt(line.at(1)) != 221 && stringToInt(line.at(1)) != 277){

                a1 = getAtom(stringToInt(line.at(0)));
                if(a1 == NULL){
                    continue;
                }

                a2 = getAtom(a1->b1);
                a3 = getAtom(a1->b2);
                a4 = getAtom(a1->b3);

                if(!isDihedral(a1, a2, a3, a4)){
                    continue;
                }

                dh1 = new Dihedral();
                dh2 = new Dihedral();
                dh3 = new Dihedral();
                dh4 = new Dihedral();

                dh1->a1 = a1;
                dh1->a2 = a2;
                dh1->a3 = a3;
                dh1->a4 = a4;

                dh2->a1 = a1;
                dh2->a2 = a2;
                dh2->a3 = a3;
                dh2->a4 = a4;

                dh3->a1 = a1;
                dh3->a2 = a2;
                dh3->a3 = a3;
                dh3->a4 = a4;

                dh4->a1 = a1;
                dh4->a2 = a2;
                dh4->a3 = a3;
                dh4->a4 = a4;

                int ln = 1;

                for( filepos_it2 = parsedParFile.begin(); filepos_it2 != parsedParFile.end(); filepos_it2++, ln++){
                    vector<string> parLine = *filepos_it2;

                    if(ln >= 3102 && parLine.size() >= 4){
                        if(stringToInt(parLine.at(0)) == stringToInt(line.at(1))){

                            dh1->v = stringToFloat(parLine.at(1));
                            dh2->v = stringToFloat(parLine.at(2));
                            dh3->v = stringToFloat(parLine.at(3));
                            dh4->v = stringToFloat(parLine.at(4));

                            dh1->n = 1;
                            dh2->n = 2;
                            dh3->n = 3;
                            dh4->n = 4;

                            dh1->theta = 0.0f;
                            dh2->theta = 180.0f;
                            dh3->theta = 0.0f;
                            dh4->theta = 180.0f;
                            break;
                        } 
                    }
                }
                dihedrals.push_back(dh1);
                dihedrals.push_back(dh2);
                dihedrals.push_back(dh3);
                dihedrals.push_back(dh4);
            } 
        }
    }
    
    bool dihedralSection2 = false;
    for(filepos_it = parsedZFile.begin(); filepos_it != parsedZFile.end(); filepos_it++){
        vector<string> zline = *filepos_it;
        if(zline.size() >= 3){
            if(zline.at(0) == "Additional" && zline.at(1) == "Dihedrals"){
                dihedralSection2 = true;
                continue;
            }
            else if(dihedralSection2 && zline.at(0) == "Final"){
                dihedralSection2 = false;
                break;
            }

            if(dihedralSection2 && zline.size() >= 6 && stringToInt(zline.at(0)) != 0 
                && stringToInt(zline.at(4)) != 160 && stringToInt(zline.at(4)) != 161 
                && stringToInt(zline.at(4)) != 162 && stringToInt(zline.at(4)) != 221 
                && stringToInt(zline.at(4)) != 277){

                dh1 = new Dihedral();
                dh2 = new Dihedral();
                dh3 = new Dihedral();
                dh4 = new Dihedral();
                dh1->a1 = getAtom(stringToInt(zline.at(0)));
                dh2->a1 = getAtom(stringToInt(zline.at(0)));
                dh3->a1 = getAtom(stringToInt(zline.at(0)));
                dh4->a1 = getAtom(stringToInt(zline.at(0)));

                dh1->a2 = getAtom(stringToInt(zline.at(1)));
                dh2->a2 = getAtom(stringToInt(zline.at(1)));
                dh3->a2 = getAtom(stringToInt(zline.at(1)));
                dh4->a2 = getAtom(stringToInt(zline.at(1)));

                dh1->a3 = getAtom(stringToInt(zline.at(2)));
                dh2->a3 = getAtom(stringToInt(zline.at(2)));
                dh3->a3 = getAtom(stringToInt(zline.at(2)));
                dh4->a3 = getAtom(stringToInt(zline.at(2)));

                dh1->a4 = getAtom(stringToInt(zline.at(3)));
                dh2->a4 = getAtom(stringToInt(zline.at(3)));
                dh3->a4 = getAtom(stringToInt(zline.at(3)));
                dh4->a4 = getAtom(stringToInt(zline.at(3)));

                if(!isDihedral(dh1->a1, dh1->a2, dh1->a3, dh1->a4)){
                    continue;
                }

                dh1->n = 1;
                dh2->n = 2;
                dh3->n = 3;
                dh4->n = 4;

                dh1->theta = 0.0f;
                dh2->theta = 180.0f;
                dh3->theta = 0.0f;
                dh4->theta = 180.0f;

                int ln = 1;
                for(filepos_it2 = parsedParFile.begin(); filepos_it2 != parsedParFile.end(); filepos_it2++, ln++){
                    vector<string> parline = *filepos_it2;

                    if(ln >= 3102 && parline.size() >= 4){
                        if(stringToInt(parline.at(0)) == stringToInt(zline.at(4))){
                            dh1->v = stringToFloat(parline.at(1));
                            dh2->v = stringToFloat(parline.at(2));
                            dh3->v = stringToFloat(parline.at(3));
                            dh4->v = stringToFloat(parline.at(4));
                            break;
                        }
                    }
                }
                dihedrals.push_back(dh1);
                dihedrals.push_back(dh2);
                dihedrals.push_back(dh3);
                dihedrals.push_back(dh4);
            }
        }
    }

    return (int)dihedrals.size();
}

bool isImproper(Atom *a1, Atom *a2, Atom *a3, Atom *a4){
    return (a1 != NULL && a2 != NULL && a3 != NULL && a4 != NULL 
        && !(a1->boundTo == a2->number && a2->boundTo == a3->number && a3->boundTo == a4->number));
}

Atom *getAtomByAlias(string alias){
    list<Atom*>::iterator atoms_it;
    for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
        Atom *a = *atoms_it;
        if(a->typeAlias == alias){
            return a;
        }
    }
    return NULL;
}

bool improperAdded(Atom *impra){
    list<Improper*>::iterator it;
    for(it = impropers.begin(); it != impropers.end(); it++){
        Improper *imp = *it;
        if(imp->a2 == impra){
            return true;
        }
    }
    return false;
}

list<Atom*> getBonded(Atom *a){
    list<Atom*> as;
    list<Bond*>::iterator it;
    for(it = bonds.begin(); it != bonds.end(); it++){
        Bond *b = *it;
        if(a == b->a1){
            as.push_back(b->a2);
        }
        else if(a == b->a2){
            as.push_back(b->a1);
        }
    }
    return as;
}

int findImpropers(){
    Improper *imp;
    Atom *a1, *a2, *a3, *a4;
    a1 = NULL;
    a2 = NULL;
    a3 = NULL;
    a4 = NULL;
    bool dihedralSection1 = false;
    list<vector<string> >::iterator filepos_it, filepos_it2;
    for(filepos_it = parsedZFile.begin(); filepos_it != parsedZFile.end(); filepos_it++){
        vector<string> line = *filepos_it;
        if(line.size() >= 3){
            if(line.at(0) == "Variable" && line.at(1) == "Dihedrals"){
                dihedralSection1 = true;
                continue;
            }
            else if(line.at(0) == "Additional" && line.at(1) == "Dihedrals"){
                dihedralSection1 = false;
                break;
            }

            if(dihedralSection1 && line.size() >= 4 && stringToInt(line.at(0)) != 0 && stringToInt(line.at(1)) != 0 &&
                (stringToInt(line.at(1)) == 160 || stringToInt(line.at(1)) == 161 || stringToInt(line.at(1)) == 162 || 
                stringToInt(line.at(1)) == 221 || stringToInt(line.at(1)) == 277)){

                a1 = getAtom(stringToInt(line.at(0)));
                if(a1 == NULL){
                    continue;
                }

                a2 = getAtom(a1->b1);
                a3 = getAtom(a1->b2);
                a4 = getAtom(a1->b3);

                if(!isImproper(a1, a2, a3, a4)){
                    continue;
                }

                imp = new Improper();

                imp->a1 = a1;
                imp->a2 = a2;
                imp->a3 = a3;
                imp->a4 = a4;

                imp->n = 2;
                imp->theta = 180.0f;

                int ln = 1;

                for(filepos_it2 = parsedParFile.begin(); filepos_it2 != parsedParFile.end(); filepos_it2++, ln++){
                    vector<string> parLine = *filepos_it2;

                    if(ln >= 3102 && parLine.size() >= 4){
                        if(stringToInt(parLine.at(0)) == stringToInt(line.at(1))){
                            imp->v = stringToFloat(parLine.at(2));
                            break;
                        }
                    }
                }
                impropers.push_back(imp);
            }
        }
    }

    bool dihedralSection2 = false;
    for(filepos_it = parsedZFile.begin(); filepos_it != parsedZFile.end(); filepos_it++){
        vector<string> zline = *filepos_it;
        if(zline.size() >= 3){
            if(zline.at(0) == "Additional" && zline.at(1) == "Dihedrals"){
                dihedralSection2 = true;
                continue;
            }
            else if(dihedralSection2 && zline.at(0) == "Final"){
                dihedralSection2 = false;
                break;
            }

            if(dihedralSection2 && zline.size() >= 6 && stringToInt(zline.at(0)) != 0){ 
                if(stringToInt(zline.at(4)) == 160 || stringToInt(zline.at(4)) == 161 
                || stringToInt(zline.at(4)) == 162 || stringToInt(zline.at(4)) == 221 
                || stringToInt(zline.at(4)) == 277){

                    a1 = getAtom(stringToInt(zline.at(0)));
                    a2 = getAtom(stringToInt(zline.at(1)));
                    a3 = getAtom(stringToInt(zline.at(2)));
                    a4 = getAtom(stringToInt(zline.at(3)));

                    if(!isImproper(a1, a2, a3, a4)){
                        continue;
                    }

                    imp = new Improper();
                    imp->a1 = a1;
                    imp->a2 = a2;
                    imp->a3 = a3;
                    imp->a4 = a4;

                    imp->n = 2;
                    imp->theta = 180.0f;

                    int ln = 1;
                    for(filepos_it2 = parsedParFile.begin(); filepos_it2 != parsedParFile.end(); filepos_it2++, ln++){
                        vector<string> parline = *filepos_it2;

                        if(ln >= 3102 && parline.size() >= 4){
                            if(stringToInt(parline.at(0)) == stringToInt(zline.at(4))){
                                imp->v = stringToFloat(parline.at(2));
                                break;
                            }
                        }
                    }
                    impropers.push_back(imp);
                }
            }
        }
    }
    list<Atom*> improperAtoms;
    list<string>::iterator al;
    for(filepos_it = parsedImprList.begin(); filepos_it != parsedImprList.end(); filepos_it++){
        vector<string> line = *filepos_it;
        for(al = aliases.begin(); al != aliases.end(); al++){
            vector<string> alias = split(*al);
            if(alias.at(1) == line.at(0)){
                a2 = getAtomByAlias(alias.at(0));
                if(a2 == NULL){
                    continue;
                }
                if(improperAdded(a2)){
                    continue;
                }
                a2->tmp = line.at(1).c_str();
                improperAtoms.push_back(a2);
            }
        }
    }    
    list<Atom*>::iterator ia, a;
    for(ia = improperAtoms.begin(); ia != improperAtoms.end(); ia++){
        a2 = *ia;                
        list<Atom*> bondedImprs = getBonded(a2);
        if(bondedImprs.size() != 3){
            //fprintf(stderr, "Error: Found more than 3 bonds associated with the listed improper atom: %s\n", a2->typeCHARMM.c_str());
            continue;
        }
        int index = 1;
        for(a = bondedImprs.begin(); a != bondedImprs.end(); a++){
            switch(index){
                case 1: a1 = *a; break;
                case 2: a3 = *a; break;
                case 3: a4 = *a; break;
                default: fprintf(stderr, "Error: something went wrong in the additional improper section!\n"); break;
            }
            index++;
        }
        imp = new Improper();
        imp->a1 = a1;
        imp->a2 = a2;
        imp->a3 = a3;
        imp->a4 = a4;

        for(filepos_it = parsedParFile.begin(); filepos_it != parsedParFile.end(); filepos_it++){
            vector<string> parLine = *filepos_it;
            if(parLine.size() >= 8 && parLine.size() <= 11){
                if(parLine.at(parLine.size()-2) == "improper" && parLine.at(parLine.size()-1) == "torsion"){
                    if(parLine.at(0) == a2->tmp){
                        imp->v = stringToFloat(parLine.at(2));
                    }
                }
            }
        }
        
        imp->n = 2;
        imp->theta = 180.0f;
        impropers.push_back(imp);
        
    }
    return (int)impropers.size();
}

vector<string> getSBInfo(string b){
    list<vector<string> >::iterator it;
    for(it = parsedSBFile.begin(); it != parsedSBFile.end(); it++){
        if((*it).at(0).compare(b) == 0){
            return *it;
        }
    }
    return *(new vector<string>());
}

bool contains(list<string> lst, string item){
    list<string>::iterator it;
    for(it = lst.begin(); it != lst.end(); it++){
        if((*it).compare(item) == 0)
            return true;
    }
    return false;
}

bool contains(list<string> lst, string item, int elem){
    list<string>::iterator it;
    for(it = lst.begin(); it != lst.end(); it++){
        string line = *it;
        vector<string> cols = split(line);        
        if(cols.at(elem).compare(item) == 0)
            return true;
    }
    return false;
}



void cleanup(){
    list<Atom*>::iterator atoms_it;
    list<Bond*>::iterator bonds_it;
    list<Angle*>::iterator angles_it;
    list<Dihedral*>::iterator dihedrals_it;
    list<Improper*>::iterator impropers_it;
    for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
        delete *atoms_it;
    }
    for(bonds_it = bonds.begin(); bonds_it != bonds.end(); bonds_it++){
        delete *bonds_it;
    }
    for(angles_it = angles.begin(); angles_it != angles.end(); angles_it++){
        delete *angles_it;
    }
    for(dihedrals_it = dihedrals.begin(); dihedrals_it != dihedrals.end(); dihedrals_it++){
        delete *dihedrals_it;
    }
    for(impropers_it = impropers.begin(); impropers_it != impropers.end(); impropers_it++){
        delete *impropers_it;
    }


    
}
