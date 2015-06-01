#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "psfatom.h"
#include "psfres.h"
#include "psfreslist.h"
#include "psftype.h"
#include "psftypelist.h"
#include "psfbond.h"
#include "psfatomlist.h"

#if defined(WIN32) || defined(WIN64)
#define strcasecmp  stricmp
#define strncasecmp strnicmp
#endif

// Finds the last non-white char in a string starting from
// offset going backwards and inserts a terminating null.
// Returns pointer to the same string s.
char *trimright(char *s, int offset) 
{
  char *t = NULL;
  t = s+offset-1;
  while (*t && (*t == ' ' || *t == '\t')) t--;
  t[1] = '\0';
  return(s);
}

PsfRes* addTopRes(char* line, FILE* psfin, FILE* outpsf) {
	//Make a new residue
	char atomline[101];
	char temp[20];
	char resname[5];
	sscanf(line, "%s %s", temp, resname);
	PsfRes* currRes = new PsfRes(resname);
	fgets(atomline,100,psfin);
	while((strncmp(atomline, "ATOM", 4)==0) || (strncmp(atomline, "GROU", 4)==0) || (strncmp(atomline," ",1)==0) || (strncmp(atomline,"!",1)==0) || (strncmp(atomline,"\n",1)==0) ) {
//		if ((strncmp(atomline,"!",1) == 0) || (strncmp(atomline," ",1)==0) || (strncmp(atomline, "\n",1)==0) ) {
//			fprintf(stdout,"Skipped line: %s\n",atomline);
//			fgets(atomline,80,psfin);
//			continue;
//		}
		fprintf(outpsf,"%s",atomline);
		char temp[20];
		char atomname[5];
		temp[0]='\0';
		atomname[0]='\0';
		if (strncmp(atomline, "ATOM",4)==0) {
			sscanf(atomline, "%s %s", temp, atomname);
			PsfAtom* currAtom=new PsfAtom(atomname,resname);
			currRes->addatom(currAtom);
//			fprintf(stdout,"Parsing line: %s",atomline);
		}
		fgets(atomline,80,psfin);
	}
	fprintf(outpsf,"%s",atomline);
	return currRes;
}

int parsetop(char* topin, FILE* outpsf, PsfResList* knownRes) {
	char line[101];
	//char temp[20];
	//char type[5];
	//char element[3];
	PsfRes* currRes;
	FILE* psfin;
	psfin = fopen(topin, "r");

	if (psfin == NULL) {
		fputs("Warning: Couldn't open input file ", stderr);
		fputs(topin, stderr);
		fputs("\n",stderr);
//		stderr << "Warning: Couldn't open input file " << topin << endl;
		return 1;
	}

	while(fgets(line,100,psfin)) {
  //  	        // Read the list of atomtypes
// 	        if (strncmp(line, "MASS", 4) == 0) {
// 		  int  index=0;
// 		  float mass=0;
// 		  sscanf(line, "%s %i %s %f %s", temp, &index, type, &mass, element);
// 		  PsfType* currType = new PsfType(type,mass,element);
// 		  knownType->addtype(currType);
// 		}
		//Check if we've gotten to the residue part yet
		if (strncmp(line, "RESI", 4) == 0) {
			//If so, parse and add the residue
			fprintf(outpsf,"%s",line);
//			char* comment='!';
			currRes = addTopRes(line, psfin, outpsf);
			knownRes->addres(currRes);
		} else if (strncasecmp(eatwhite(line),"END",3)) {
		       	fprintf(outpsf,"%s",line);
		}
	}
	fclose(psfin);
	return 0;
}



int psfupdate(char* topin, char* inpdb, char* psfout) {
/*This function will read through a psf and pdb, and then write a new top
 * containing atoms labeled XXX for every atom that is not found in the top
 * file */ 
	//First, read in all of the known residue/atom combos
	FILE* outpsf;
	outpsf = fopen(psfout, "w");
	if (outpsf == NULL) {
		fputs("Warning: Couldn't open output file ", stderr);
		fputs(psfout, stderr);
		fputs("\n",stderr);
//		stderr << "Warning: Couldn't open input file " << topin << endl;
		return 1;
	}
	
	PsfResList*  knownRes = new PsfResList(); //Stores atoms found in psf
	//PsfTypeList* knownType = new PsfTypeList(); //stores types found in psf


	//Loops through the input tops and process each one


//	char topfiles[strlen(topin)+1]; // non-portable/illegal construct
	char topfiles[8192];
	strncpy(topfiles,topin,strlen(topin));
        topfiles[strlen(topin)]='\0';
 
	char * currtop;
	currtop=strtok(topfiles,"|");
	while (currtop != NULL) {
		if (parsetop(currtop, outpsf, knownRes) != 0) {
			fputs("Failed in parsing topology file\n",stderr);
			fputs("Bailing out from previous errors.\n",stderr);
//			stderr << "Failed in parsing topology file" << endl;
//			stderr << "Bailing out from previous errors." << endl;
//
      // Clean up and exit
      fclose(outpsf);
      delete knownRes;
			return 1;
		}
		currtop=strtok(NULL,"|");
	}

	//Now that we have all of our residues, its time to go through the pdb
	//On each atom line, we'll write a new residue and atom to the psf
	//If we don't find it already
	
	char line[101];

	PsfResList*  unknownRes  = new PsfResList();  //stores new residues
	//PsfResList*  patchRes    = new PsfResList();  //stores new residues
	PsfTypeList* unknownType = new PsfTypeList(); //stores new types
	//PsfBondList* bondlist = new PsfBondList();    //stores all bonds
	PsfAtomList* atomlist = new PsfAtomList();    //stores all atoms

	FILE* pdbin;
	pdbin = fopen(inpdb, "r");

	if (pdbin == NULL) {
		fputs("Warning: Couldn't open input pdb file ", stderr);
		fputs(inpdb, stderr);
		fputs("\n",stderr);
//		stderr << "Warning: Couldn't open input file " << topin << endl;
    // Clean up and exit
    fclose(outpsf);
    delete knownRes;
    delete unknownRes;
    delete unknownType;
    delete atomlist;
		return 1;
	}

	while(fgets(line,100,pdbin)) {
		if ((strncmp(line, "ATOM", 4)==0) || (strncmp(line, "HETATM", 6)==0)) {
			char indexstr[7];
			char resname[5];
			char segname[5];
			char resid[6];
			char atomname[6];
			char element[3];
		        char* pos = line;

			pos = pos+7;
			strncpy(indexstr,pos,6);
			pos = pos+7;
			strncpy(atomname,pos,5);
			pos = pos+6;
			strncpy(resname,pos,4);
			pos = pos+7;
			strncpy(resid,pos,5);
			pos = pos+71;
			strncpy(element,pos,2);
			element[2]='\0';
			pos = pos+4;
			strncpy(segname,pos,4);
			indexstr[6]='\0';
			atomname[5]='\0';
			resname[4]='\0';
			resid[5]='\0';
			element[2]='\0';
			segname[4]='\0';
			//fprintf(stderr, "%s, %s, %s, %s, %s\n", indexstr,atomname,resname,element,resid);
			PsfAtom* pdbatom=new PsfAtom(indexstr, atomname, resname, element, resid, segname);

			// Build a list of existing atoms
			atomlist->addatom(pdbatom);

			// Ignore unknown atoms in known RESIs.
			// This prevents from generating duplicate RESIs e.g. for ILE OXT.
			if (knownRes->lookForRes(pdbatom)) { continue; }

 			if (!((knownRes->search(pdbatom)) || (unknownRes->search(pdbatom)))) {
 			     PsfType* pdbtype=new PsfType(element);
 			     //fprintf(stdout,"Checking: %s %s %s\n",atomname,resname,element);
 			     if (!((unknownType->search(pdbtype)) )) {
 			       //fprintf(stdout,"Adding new unknown type: %s %s %s\n",atomname,resname,element);
 			       unknownType->addtype(pdbtype);
 			     } else {
 			       delete pdbtype;
 			     }

 			     //Create a new psfres and psfatom
 			     //among the unknowns
			     PsfRes* currRes = unknownRes->lookForRes(pdbatom);
			     if (currRes == NULL) {
			       //fprintf(stdout,"Didn't find RESI %s %s.\n", pdbatom->res(), pdbatom->name());
			       currRes = new PsfRes(pdbatom->res());
			       unknownRes->addres(currRes);
			     }
			     PsfAtom* pdbatom2=new PsfAtom(indexstr, atomname, resname, element, resid, segname);
			     currRes->addatom(pdbatom2);
 			} 

		}
	}

	rewind(pdbin);

 	while(fgets(line,80,pdbin)) {
	  if (strncmp(line, "CONECT", 6)==0) {
	    char currbond[7]="xxxxxx"; //Stores current bond field
	    char* bondptr;   // Pointer to current position in bond line
	    int bonds[8];    // Stores bonds of current atom
	    int numbonds;    // Stores number of bonds of current atom
	    int i=0;         // Number of the current bond
	    int numfields=0; // Number of fields in the current line
	    int pdbindex;
            numfields=(strlen(line)-1)/6;
	    bondptr=&line[0];
            numfields--;
	    bondptr += 6;
            numbonds=0;
	    //numords=0;
	    strncpy(currbond,bondptr,6);
	    //printf("8\n");
	    //fprintf(stdout, "j is %s\n", currbond);
	    pdbindex=atoi(currbond);
	    numfields--;
	    bondptr += 6;
	    while ((numfields > 0) && (numbonds < 8)) {
            strncpy(currbond,bondptr,6);
	    numfields--;
	    bondptr += 6;
	    //    fprintf(stdout,"Reading bond between %i and %s\n", j, currbond);
	      bonds[numbonds]=atoi(currbond);
	      numbonds++;
	    }
	    //fprintf(stderr, "%s", line);

	    //fprintf(stderr, "BONDS  %i, ",pdbindex);

 		  for (i=0; i<numbonds; i++) {
		    //        fprintf(stderr, "%i, ",bonds[i]);
		  }	
 		  //fprintf(stderr, "\n");
		  if (!numbonds) { continue; }

		  // Get the corresponding atom from the list
 		  PsfAtom* psfatom0;
 		  PsfAtom* psfatom1;
		  psfatom0 = atomlist->find_index(pdbindex);
		  // Find the corresponding residue
		  PsfRes* res0 = unknownRes->lookForRes(psfatom0);

      if (res0 != NULL) {
        //fprintf(stdout,"res0 %s\n",	res0->name());
        for (i=0; i<numbonds; i++) {
          //fprintf(stdout,"bonds[%i]=%i\n", i,bonds[i]);
          psfatom1 = atomlist->find_index(bonds[i]);
          PsfRes* res1 = unknownRes->lookForRes(psfatom1);
          if (!res1) {
            res1 = knownRes->lookForRes(psfatom1);
          }
          // If we didn't find the residue print a warning.
          if (!res1) { 
            printf("Warning couldn't find residue %s.\n", psfatom1->res());
            continue;
          }
          //fprintf(stdout,"res1 %s\n", res1->name());


          if (strcmp(res0->name(),res1->name())==0 && psfatom0->resid()==psfatom1->resid()) {
            // Both atoms belong to the same unknown residue, we can safely add a bond to res0
            //fprintf(stdout,"Checking bond %s-%s in resid %i\n",psfatom0->name(), psfatom1->name(),psfatom1->resid());
            PsfBond* pdbbond=new PsfBond(psfatom0->name(), psfatom1->name());

            if (!(res0->searchbond(pdbbond))) {
              //printf("Add bond %s; %s\n",psfatom0->name(), psfatom1->name());
              res0->addbond(pdbbond);
            } else {
              delete pdbbond;
            }
          } else {
            // This is a link between a known and an unknown res.
            printf("LINK %s:%s to %s:%s\n", psfatom0->res(), psfatom0->name(), psfatom1->res(), psfatom1->name());
            // 			char patchname[5], jstr[4];
            // 			strcpy(patchname,"XP");
            //			sprintf(jstr,"%i\0",j);
            // 			strcat(patchname,jstr);
            // 			PsfRes* patch = new PsfRes(patchname);
            // 			patchRes->addres(patch);
          }
        }
      }
		}
	}

	fclose(pdbin);
	//Now, we need to print new topology residues for each new atom
	unknownRes->print(outpsf);
	//printf("Input PDB: %s\n",	inpdb);
	unknownType->print(outpsf);
	fprintf(outpsf,"\n\nEND\n");
	fflush(outpsf);
	fclose(outpsf);
	delete knownRes;
	delete unknownRes;
	delete unknownType;
	delete atomlist;
	return 0;
}

// XXX This shouldn't be compiled by default, it breaks things....
	
#ifdef TEST_AUTOPSF

int main(int argc, char* argv[]) {
  printf("psfupdate test\n");
  printf("Usage:\n");
  printf("psfupdate topin pdbin topout:\n");
  char topin[1000];
  char pdbin[1000];
  char topout[1000];
  if (argc<3) { return 1; }
  strcpy(topin, argv[1]);
  strcpy(pdbin, argv[2]);
  strcpy(topout, argv[3]);
  
  printf("psfupdate %s %s %s\n", topin, pdbin, topout);
  psfupdate(topin, pdbin, topout);
  return 0;
}

#endif // TEST_AUTOPSF
