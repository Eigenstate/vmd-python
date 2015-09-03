#include "Parser.h"
#include "Atom.h"
#include "Worker.h"
#include <cstdio>
#include <math.h>

using namespace std;

//globals

extern list< vector <string> > parsedZFile;
extern list< vector <string> > parsedParFile;
extern list< vector <string> > parsedSBFile;
extern list< vector <string> > parsedOutFile;

extern list<Atom*> atoms;
extern list<Bond*> bonds;
extern list<Angle*> angles;
extern list<Dihedral*> dihedrals;
extern list<Improper*> impropers;
extern list<string> aliases;

//prototypes
int writeRTFFile(string);
int writePRMFile(string);
int writeAliasParameters();
//void clean();


const string version = "27 1";
int lines = 0;
list<string> residues;

int writeRTFFile(string outputFile){

        list<Atom*>::iterator atoms_it;
        list<Bond*>::iterator bonds_it;
        list<Improper*>::iterator impropers_it;
        list<string>::iterator rsd;

	FILE *oFile = fopen(outputFile.c_str(), "w");

	if(!oFile){
		fprintf(stderr, "%s could not be opened for writing!\n", outputFile.c_str());
		return 0;
	}

	fprintf(oFile, "%s\n", version.c_str());
	lines++;

	for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
		Atom *atom = *atoms_it;
		if(!contains(residues, atom->residue))
			residues.push_back(atom->residue);

		fprintf(oFile, "MASS	%3d %-5s  %8.5f  %c !\n", atom->number, atom->typeAlias.c_str(), getMass(atom->symbol), atom->symbol);
		lines++;
	}
        fprintf(oFile, "AUTO ANGLES DIHE\n");
        lines++;
	for(rsd = residues.begin(); rsd != residues.end(); rsd++){
		float sumCharge = 0.0f;
		for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
			Atom *atom = *atoms_it;
			if(atom->residue == *rsd)
				sumCharge += atom->charge;
		}
		fprintf(oFile, "RESI %-15s % .3f\nGroup\n", (*rsd).c_str(), sumCharge);
		lines += 2;

		for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
			Atom *atom = *atoms_it;
			if(atom->residue == *rsd)
				fprintf(oFile, "ATOM %-3s  %-5s % 8.6f !\n", atom->name.c_str(), atom->typeAlias.c_str(), atom->charge);
				lines++;
		}

		for(bonds_it = bonds.begin(); bonds_it != bonds.end(); bonds_it++){
			Bond *bond = *bonds_it;
			if(bond->a1->residue == *rsd){
				fprintf(oFile, "BOND %3s %3s\n", bond->a1->name.c_str(), bond->a2->name.c_str());
				lines++;
			}
		}

		for(impropers_it = impropers.begin(); impropers_it != impropers.end(); impropers_it++){
			Improper *imp = *impropers_it;
			if(imp->a2->residue == *rsd){
				fprintf(oFile, "IMPR %-3s %-3s %-3s %-3s\n", imp->a1->name.c_str(), imp->a2->name.c_str(), imp->a3->name.c_str(), imp->a4->name.c_str());
			}
		}
	}
	fclose(oFile);
	return lines;
}

int writePRMFile(string outputFile){
	int lines = 0;
        list<Bond*>::iterator bonds_it;
        list<Angle*>::iterator angles_it;
        list<Dihedral*>::iterator dihedrals_it;
        list<Improper*>::iterator impropers_it;
        list<Atom*>::iterator  atoms_it;
	if(angles.size() == 0){
		return 0;
	}

	FILE *oFile = fopen(outputFile.c_str(), "w");
	FILE *eFile = fopen("missingbonds", "w");
	if(!oFile){
		fprintf(stderr, "Could not open the prm file for writing!\n");
		return 0;
	}

	fprintf(oFile, "BOND\n");
	lines++;

	for(bonds_it = bonds.begin(); bonds_it != bonds.end(); bonds_it++){
		Bond *bond = *bonds_it;
		if(bond->valid){
			fprintf(oFile, "%-5s %-5s %8.2f %10.3f\n", bond->a1->typeAlias.c_str(), bond->a2->typeAlias.c_str(), bond->force, bond->distance);
			lines++;
		}
		else{
			fprintf(eFile, "%-3s-%s\n", bond->a1->name.c_str(), bond->a2->name.c_str());
		}
	}

	fprintf(oFile, "ANGLE\n");
	lines++;

	for(angles_it = angles.begin(); angles_it != angles.end(); angles_it++){
		Angle *angle = *angles_it;
		fprintf(oFile, "%-5s %-5s %-5s %8.2f %8.2f\n", angle->a1->typeAlias.c_str(), angle->a2->typeAlias.c_str(), angle->a3->typeAlias.c_str(), angle-> force, angle->angle);
		lines++;
	}

	fprintf(oFile, "DIHEDRAL\n");
	lines++;

	for(dihedrals_it = dihedrals.begin(); dihedrals_it != dihedrals.end(); dihedrals_it++){
		Dihedral *dh = *dihedrals_it;
		fprintf(oFile, "%-5s %-5s %-5s %-5s % 11.4f %5d %7.1f\n", dh->a1->typeAlias.c_str(),  dh->a2->typeAlias.c_str(), dh->a3->typeAlias.c_str(), dh->a4->typeAlias.c_str(), dh->v/2, dh->n, dh->theta);
		lines++;
	}

	fprintf(oFile, "IMPROPER\n");
	lines++;

	for(impropers_it = impropers.begin(); impropers_it != impropers.end(); impropers_it++){
		Improper *imp = *impropers_it;
		fprintf(oFile, "%-5s %-5s %-5s %-5s % 11.4f %5d %7.1f\n", imp->a1->typeAlias.c_str(), imp->a2->typeAlias.c_str(), imp->a3->typeAlias.c_str(), imp->a4->typeAlias.c_str(), imp->v/2, imp->n, imp->theta);
		lines++;
	}

	fprintf(oFile, "\n\nNONBONDED\n\n");
	lines++;

	for(atoms_it = atoms.begin(); atoms_it != atoms.end(); atoms_it++){
		Atom *a = *atoms_it;
		double v = 0.5 * pow(2, 0.16666666667) * a->sigma;
		fprintf(oFile, "%-5s %4.2f % 10.6f % 11.7lf %4.2f % 10.6f % 11.7lf\n", a->typeAlias.c_str(), 0.0f,-(a->epsilon), v, 0.0f, -((a->epsilon)/2), v);
		lines++;
	}

	fprintf(oFile, "\nEND\n");

	fclose(oFile);
	fclose(eFile);
	return lines;
}

int writeAliasParameters(){
	FILE *oFile = fopen("./alias", "w");
	int lines = 0;
        list<string>::iterator it;
	if(!oFile){
		fprintf(stderr, "Could not open the alias file for writing!\n");
		return 0;
	}

	for(it = aliases.begin(); it != aliases.end(); it++){
		fprintf(oFile, "%s\n", (*it).c_str());
		lines++;
		
	}

	fclose(oFile);
	return lines;
}

/*
void clean(){
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

    //delete &parsedZFile;
    //delete &parsedParFile;
    //delete &parsedSBFile;
    //delete &parsedOutFile;
    //delete &atoms;	
    //delete &bonds;
    //delete &angles;
    //delete &dihedrals;
    //delete &impropers;
}
*/ 
