/*************************************************************************\
*    Topology 2.0                                                         *
*    ------------                                                         *
*    Authors:  Markus K. Dalghren (markusan2000@yahoo.se)                 *
*              Peter T. Dalghren (peter.t.dalghren@yahoo.com)             *
*                                                                         *
*    Converts a BOSS Z-matrix file (that includes dihedrals and OPLS atom *
*    types) and the output of a BOSS single point energy calculation to   *
*    CHARMM-formatted topology (.rtf) and parameter (.prm) files.         *  
\*************************************************************************/
#include "Parser.h"
#include "Worker.h"
#include "Atom.h"
#include <iostream>
#include <cstdio>

using namespace std;

extern list< vector <string> > parsedZFile;
extern list< vector <string> > parsedParFile;
extern list< vector <string> > parsedSBFile;
extern list< vector <string> > parsedOutFile;
extern list<vector<string> > parsedImprList;

void PrintUsage();

int main(int argc, char * argv[]){
    stringstream strStream;    
    string fileName = "";
    string bossOutFile = "out";
    string parmfilesDir = "";
    string oplssbFilename = "oplsaa.sb";
    string imprlistFilename = "imprlist";
    string oplsparFilename = "oplsaa.par";
    string curvar;
    bool validInput;


    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    if (argc == 2) {
            strStream.str(argv[1]);
            strStream >> curvar;
            strStream.clear();
            if (curvar == "-test") 
                return 0;
    }    
 
    int i = 1;
    //process arguments
    while (i < argc) {
        if (i%2 != 1 && argv[i][0] == '-' && i < argc-1) {
            cerr << "Error: Incorrect argument specification.\n";
            PrintUsage();
            return -1;
        } 
        else if (i%2 == 1 && argv[i][0] != '-' && i < argc-1) {
            cerr << "Error: Incorrect argument specification.\n";
            PrintUsage();
            return -1;
        }
        else if ( argv[i][0] == '-' && i < argc-2 ) {
            strStream.str(argv[i]);
            strStream >> curvar;
            strStream.clear();
            ++i;
            if (curvar == "-bossout") {
                strStream.str(argv[i]);
                strStream >> bossOutFile;
                strStream.clear();
                ++i;
            }
            //Removed for now to reduce confusion
            //
            //else if (curvar == "-parmdir") {
            //    strStream.str(argv[3]);
            //    strStream >> parmfilesDir;
            //    strStream.clear();
            //}
            else if (curvar == "-oplspar") {
                strStream.str(argv[i]);
                strStream >> oplsparFilename;
                strStream.clear();
                ++i;
            }
            else if (curvar == "-oplssb") {
                strStream.str(argv[i]);
                strStream >> oplssbFilename;
                strStream.clear();
                ++i;
            }
            else if (curvar == "-imprlist") {
                strStream.str(argv[i]);
                strStream >> imprlistFilename;
                strStream.clear();
                ++i;
            }
            else {
                cerr << "Error: Argument " << curvar << "not understood.\n";
                return -1;
            }
            
        }
        else if ( i == argc-1 ) {
            strStream.str(argv[i]);
            strStream >> fileName;
            strStream.clear();
            ++i;
        }
        else {
            cerr << "Error: Incorrect argument specification.\n";
            PrintUsage();
            return -1;
        }
    } 
    
   


    validInput = true;
    ifstream file;
    // Check z-matrix file
    if(fileName.length() == 0 || fileName.substr(fileName.length()-2, fileName.length()) != ".z"){
        cerr << "Error: z-matrix file must end with .z" << endl;
        validInput = false;
    }
    else {
        file.open(fileName.c_str());
        if(!file.is_open()){
            cerr << "Error: z-matrix file: " << fileName << " could not be opened!" << endl;
            validInput = false;
        }
        else{
            file.close();
        }
    }
    fileName = fileName.substr(0, fileName.length()-2);


    // Check boss output file
    file.open(bossOutFile.c_str());
    if(!file.is_open()){
        cerr << "Error: BOSS output file: " << bossOutFile << " could not be opened!" << endl;
        validInput = false;
    }
    else{
        file.close();
    }
    if (!validInput) return -1;

    
    int outlines;
    outlines = parseOutFile(bossOutFile, &parsedOutFile);
    parseFile(fileName + ".z", &parsedZFile);
    parseFile((oplsparFilename).c_str(), &parsedParFile);
    parseFile((oplssbFilename).c_str(), &parsedSBFile);
    parseFile((imprlistFilename).c_str(), &parsedImprList);

    
    int n = 0;
    n = lookup();
    cout << n << " atoms found\n";
    n = findBonds();
    cout << n << " bonds found\n";
    n = findAngles();
    cout << n << " angles found\n";
    n = findDihedrals()/4;
    cout << n << " dihedrals found\n";
    n = findImpropers();
    cout << n << " impropers found\n";

    cout << writeRTFFile((fileName + ".rtf")) << " lines written to " << (fileName + ".rtf") << endl;
    cout << writePRMFile((fileName + ".prm")) << " lines written to " << (fileName + ".prm") << endl;;
    cout << writeAliasParameters() << " lines written to alias\n";
    
    cleanup();

    return 0;
}

void PrintUsage(){
    cout << "Usage:\n  Topology [-oplspar oplsaa.par] [-oplssb oplsaa.sb] [-imprlist imprlist] [-bossout out] molecule.z\n"
         << "where:\n"
         << "\tmolecule.z  - BOSS Z-matrix containing molecule containing dihedrals and OPLS atom types.\n"
         << "\t-oplssb     - Specifies which 'oplsaa.par' file to use.\n"
         << "\t-oplspar    - Specifies which 'oplsaa.sb', file to use.\n"
         << "\t-imprlist   - Specifies which 'imprlist' file to use.\n"
         << "\t-bossout    - Specifies file with output from BOSS single point energy calculation containing OPLS parameters.\n";
        
};
