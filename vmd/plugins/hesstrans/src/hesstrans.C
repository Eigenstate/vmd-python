
//#define WANT_STREAM                  // include.h will get stream fns
//#define WANT_MATH                    // include.h will get math fns
                                     // newmatap.h will get include.h
#include "hesstrans.h"

/*
   #ifdef use_namespace
   using namespace NEWMAT;              // access NEWMAT namespace
   #endif
*/

#define PI 3.14159265358979323846
#define LIGHTSPEED 299792458


void getGeneralizedInverse(Matrix& G, Matrix& Gi) {
#ifdef DEBUG
  cout << "\n\ngetGeneralizedInverse - Singular Value\n";
#endif  

  // Singular value decomposition method
  
  // do SVD
  Matrix U, V;
  DiagonalMatrix D;
  SVD(G,D,U,V);            // X = U * D * V.t()
  
#ifdef DEBUG
  cout << "D:\n";
  cout << setw(9) << setprecision(6) << (D);
  cout << "\n\n";
#endif
  
  DiagonalMatrix Di;
  Di << D.i();
  
#ifdef DEBUG
  cout << "Di:\n";
  cout << setw(9) << setprecision(6) << (Di);
  cout << "\n\n";
#endif
  

  int i=Di.Nrows();
  for (; i>=1; i--) {
    if (Di(i) > 1000.0) {
      Di(i) = 0.0;
    }
  }
  
#ifdef DEBUG
  cout << "Di with biggies zeroed out:\n";
  cout << setw(9) << setprecision(6) << (Di);
  cout << "\n\n";
#endif
  
  //Matrix Gi;
  Gi << (U * (Di * V.t()));
  
  return;
}


/**
 * Get the Wilson's B-matrix
 */
void getBMatrix(Real** cartCoords, int numCartesians,
		bondCoord** bonds, int numBonds,
		angleCoord** angles, int numAngles,
		dihedralCoord** dihedrals, int numDihedrals,
		improperCoord** impropers, int numImpropers,
		Matrix& B) {
#ifdef DEBUG
  cout << "\n\ngetBMatrix - Constructing B Matrix\n";
#endif

  // Constructing B Matrix
  //   follows method in chapter 4 of Molecular Vibrations by Wilson, Decius, and Cross

#ifdef DEBUG  
  int numInternals = numBonds + numAngles + numDihedrals + numImpropers;
  cout << "numBonds: " << numBonds << "\n";
  cout << "numAngles: " << numAngles << "\n";
  cout << "numDihedrals: " << numDihedrals << "\n";
  cout << "numImpropers: " << numImpropers << "\n";
  cout << "numInternals: " << numInternals << "\n";
#endif

  // Load Data
  B = 0.0;
  int i = 0;
  int j = 0;
  int index1 = 0;
  int index2 = 0;
  RowVector tempCoord1(3);
  RowVector tempCoord2(3);
  Real norm = 0.0;
  
  // Bonds
  for (i=0; i<numBonds; i++) {
    index1 = bonds[i]->x1;
    index2 = bonds[i]->x2;
    //norm = bonds[i].val;   // Could calculate this, like below.
#ifdef DEBUG
    cout << "index1=" << index1 << "index2=" << index2 << "\n";
#endif
    for (j=0; j<3; j++) {
      tempCoord1(j+1) = cartCoords[index1][j];
      tempCoord2(j+1) = cartCoords[index2][j];
    }
    tempCoord1 << tempCoord1 - tempCoord2;
    norm = tempCoord1.NormFrobenius();   // XXX - don't delete
    if (norm > 0.0) {
      tempCoord1 << tempCoord1 / norm;
    }
    for (j=1; j<=3; j++) {
      B(i+1,((index1)*3)+j) =  tempCoord1(j);
      B(i+1,((index2)*3)+j) = -tempCoord1(j);
    }
  }
  
#ifdef DEBUG
  cout << "after bonds\n";
  cout << "B:\n";
  cout << setw(9) << setprecision(3) << (B);
  cout << "\n\n";
#endif

  // Angles
  int index3 = 0;
  RowVector tempCoord3(3);
  RowVector tempCoord4(3);
  RowVector tempCoord5(3);
  RowVector r21(3);   // Vector from 2nd to 1st point
  RowVector r23(3);   // Vector from 2nd to 3rd point
  RowVector e21(3);   // Unit vector from 2nd to 1st point
  RowVector e23(3);   // Unit vector from 2nd to 3rd point
  Real norm21;        // Norm of r21
  Real norm23;        // Norm of r23
  Real angle = 0.0;   // Angle in radians
  Real cosAngle123 = 0.0;
  Real sinAngle123 = 0.0;
  //Real pi = 3.14159265;
  Real scaleFactor = 0.529178;   // Scaling factor (0.529178)
  for (i=0; i<numAngles; i++) {
    index1 = angles[i]->x1;
    index2 = angles[i]->x2;
    index3 = angles[i]->x3;
    //angle = angles[i].val * (pi/180.0);   // Convert to radians.
    for (j=0; j<3; j++) {
      tempCoord1(j+1) = cartCoords[index1][j];
      tempCoord2(j+1) = cartCoords[index2][j];
      tempCoord3(j+1) = cartCoords[index3][j];
    }
    r21 << tempCoord1 - tempCoord2;
    r23 << tempCoord3 - tempCoord2;
    norm21 = r21.NormFrobenius();
    norm23 = r23.NormFrobenius();
    e21 << r21;
    if (norm21 > 0.0) {
      e21 << e21 / norm21;
    }
    e23 << r23;
    if (norm23 > 0.0) {
      e23 << e23 / norm23;
    }
    angle = acos(DotProduct(r21,r23) / (norm21 * norm23));
    cosAngle123 = DotProduct(r21,r23) / (norm21 * norm23);
    sinAngle123 = sqrt(1 - (cosAngle123 * cosAngle123));
    
#ifdef DEBUG
    cout << "r21: " << (r21) << "\n";
    cout << "r23: " << (r23) << "\n";
    cout << "norm21: " << norm21 << ", norm23: " << norm23 << "\n\n";
    cout << "e21: " << (e21) << "\n";
    cout << "e23: " << (e23) << "\n";
    cout << "cos(" << angle << "): " << cos(angle) << "\n";
    cout << "sin(" << angle << "): " << sin(angle) << "\n";
    cout << "angle: " << acos(DotProduct(r21,r23) / (norm21 * norm23)) << "\n";
    cout << "cosAngle123: " << cosAngle123 << "\n";
    cout << "sinAngle123: " << sinAngle123 << "\n";
#endif    

    // First elements of coordinate triples
    tempCoord4 << ((cosAngle123 * e21) - e23);
    tempCoord4 << (tempCoord4 * scaleFactor) / (norm21 * sinAngle123);
    for (j=1; j<=3; j++) {
      B(i+numBonds+1,((index1)*3)+j) = tempCoord4(j);
    }
    // Third elements of coordinate triples
    tempCoord5 << ((cosAngle123 * e23) - e21);
    tempCoord5 << (tempCoord5 * scaleFactor) / (norm23 * sinAngle123);
    for (j=1; j<=3; j++) {
      B(i+numBonds+1,((index3)*3)+j) = tempCoord5(j);
    }
    // Second (middle) elements of coordinate triples (depends on 1st and 3rd)
    tempCoord4 << -tempCoord4 - tempCoord5;
    for (j=1; j<=3; j++) {
      B(i+numBonds+1,((index2)*3)+j) = tempCoord4(j);
    }
  }
  
#ifdef DEBUG
  cout << "after angles\n";
  cout << "B:\n";
  cout << setw(9) << setprecision(3) << (B);
  cout << "\n\n";
#endif  

  // Dihedrals
  RowVector r12(3);   // Vector from 1st to 2nd point
  RowVector r32(3);   // Vector from 3rd to 2nd point
  RowVector r34(3);   // Vector from 3rd to 2nd point
  RowVector r43(3);   // Vector from 4th to 3rd point
  RowVector e12(3);   // Unit vector from 1st to 2nd point
  RowVector e32(3);   // Unit vector from 3rd to 2nd point
  RowVector e34(3);   // Unit vector from 3rd to 2nd point
  RowVector e43(3);   // Unit vector from 4th to 3rd point
  Real norm12;        // Norm of r12
  Real norm32;        // Norm of r32
  Real norm34;        // Norm of r34
  Real norm43;        // Norm of r43
  RowVector cross1223(3);   // Cross product of e12 and e23
  RowVector cross4332(3);   // Cross product of e43 and e32
  Real angle123 = 0.0;   // Angle in radians
  Real angle234 = 0.0;   // Angle in radians
  Real cosAngle234 = 0.0;
  Real sinAngle234 = 0.0;
  scaleFactor = 0.529178;   // Scaling factor (0.529178)
  int index4 = 0;
  RowVector tempCoord6(3);
  for (i=0; i<numDihedrals; i++) {
    index1 = dihedrals[i]->x1;
    index2 = dihedrals[i]->x2;
    index3 = dihedrals[i]->x3;
    index4 = dihedrals[i]->x4;
    for (j=0; j<3; j++) {
      tempCoord1(j+1) = cartCoords[index1][j];
      tempCoord2(j+1) = cartCoords[index2][j];
      tempCoord3(j+1) = cartCoords[index3][j];
      tempCoord4(j+1) = cartCoords[index4][j];
    }
    r12 << tempCoord2 - tempCoord1;
    r21 << tempCoord1 - tempCoord2;
    r23 << tempCoord3 - tempCoord2;
    r32 << tempCoord2 - tempCoord3;
    r34 << tempCoord4 - tempCoord3;
    r43 << tempCoord3 - tempCoord4;
    norm12 = r12.NormFrobenius();
    norm21 = r21.NormFrobenius();
    norm23 = r23.NormFrobenius();
    norm32 = r32.NormFrobenius();
    norm34 = r34.NormFrobenius();
    norm43 = r43.NormFrobenius();
#ifdef DEBUG
    cout << "norm12: " << norm12 << "\n";
    cout << "norm21: " << norm21 << "\n";
    cout << "norm23: " << norm23 << "\n";
    cout << "norm32: " << norm32 << "\n";
    cout << "norm34: " << norm34 << "\n";
    cout << "norm43: " << norm43 << "\n";
#endif
    e12 << r12 / norm12;
    e21 << r21 / norm21;
    e23 << r23 / norm23;
    e32 << r32 / norm32;
    e34 << r34 / norm34;
    e43 << r43 / norm43;
    angle123 = acos(DotProduct(r21,r23) / (norm21 * norm23));   // Wilson's angle 2
    angle234 = acos(DotProduct(r32,r34) / (norm32 * norm34));   // Wilson's angle 3
    cosAngle123 = DotProduct(r21,r23) / (norm21 * norm23);
    cosAngle234 = DotProduct(r32,r34) / (norm32 * norm34);
    sinAngle123 = sqrt(1 - (cosAngle123 * cosAngle123));
    sinAngle234 = sqrt(1 - (cosAngle234 * cosAngle234));
#ifdef DEBUG
    cout << "angle123: " << angle123 << ", cos(angle123): " << cos(angle123) << ", sin(angle123): " << sin(angle123) << "\n";
    cout << "angle234: " << angle234 << ", cos(angle234): " << cos(angle234) << ", sin(angle234): " << sin(angle234) << "\n";
    cout << "cosAngle123: " << cosAngle123 << ", sinAngle123: " << sinAngle123 << "\n";
    cout << "cosAngle234: " << cosAngle234 << ", sinAngle234: " << sinAngle234 << "\n";
#endif
    cross1223(1) = (e12(2)*e23(3)) - (e12(3)*e23(2));
    cross1223(2) = (e12(3)*e23(1)) - (e12(1)*e23(3));
    cross1223(3) = (e12(1)*e23(2)) - (e12(2)*e23(1));
    cross4332(1) = (e43(2)*e32(3)) - (e43(3)*e32(2));
    cross4332(2) = (e43(3)*e32(1)) - (e43(1)*e32(3));
    cross4332(3) = (e43(1)*e32(2)) - (e43(2)*e32(1));
#ifdef DEBUG
    cout << "cross1223 (norm " << cross1223.NormFrobenius() << "):\n";
    cout << setw(9) << setprecision(6) << (cross1223);
    cout << "\n\n";
    cout << "cross4332 (norm " << cross4332.NormFrobenius() << "):\n";
    cout << setw(9) << setprecision(6) << (cross4332);
    cout << "\n\n";
#endif
    // First elements of coordinate triples
    tempCoord5 << -((cross1223 * scaleFactor) / (norm12 * sinAngle123 * sinAngle123));
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+1,((index1)*3)+j) = tempCoord5(j);
    }
    // Second elements of coordinate triples
    tempCoord5 << ((norm23 - (norm12 * cosAngle123)) / (norm23 * norm12 * sinAngle123 * sinAngle123)) * (cross1223);
    tempCoord6 << (cosAngle234 / (norm23 * sinAngle234 * sinAngle234)) * (cross4332);
#ifdef DEBUG
    cout << "tempCoord5:\n";
    cout << setw(9) << setprecision(6) << (tempCoord5);
    cout << "tempCoord6:\n";
    cout << setw(9) << setprecision(6) << (tempCoord6);
#endif
    tempCoord5 << (tempCoord5 + tempCoord6) * scaleFactor;
#ifdef DEBUG
    cout << "tempCoord5:\n";
    cout << setw(9) << setprecision(6) << (tempCoord5);
#endif
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+1,((index2)*3)+j) = tempCoord5(j);
    }
    // Third elements of coordinate triples
    tempCoord5 << ((norm32 - (norm43 * cosAngle234)) / (norm32 * norm43 * sinAngle234 * sinAngle234)) * (cross4332);
    tempCoord6 << (cosAngle123 / (norm32 * sinAngle123 * sinAngle123)) * (cross1223);
    tempCoord5 << (tempCoord5 + tempCoord6) * scaleFactor;
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+1,((index3)*3)+j) = tempCoord5(j);
    }
    // Fourth elements of coordinate triples
    tempCoord5 << -((cross4332 * scaleFactor) / (norm43 * sinAngle234 * sinAngle234));
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+1,((index4)*3)+j) = tempCoord5(j);
    }
  }
  
#ifdef DEBUG
  cout << "B:\n";
  cout << setw(9) << setprecision(3) << (B);
  cout << "\n\n";
#endif

  // Impropers
  RowVector r41(3);   // Vector from 4th to 1st point
  RowVector r42(3);   // Vector from 4th to 2nd point
  RowVector e41(3);   // Unit vector from 4th to 1st point
  RowVector e42(3);   // Unit vector from 4th to 2nd point
  RowVector normVector(3);   // Normal to the plane
  Real norm41;        // Norm of r41
  Real norm42;        // Norm of r42
  Real angle142 = 0.0;   // Angle in radians
  Real angle143 = 0.0;   // Angle in radians
  Real angle243 = 0.0;   // Angle in radians
  Real cosAngle142 = 0.0;
  Real cosAngle143 = 0.0;
  Real cosAngle243 = 0.0;
  Real sinAngle142 = 0.0;
  Real sinAngle143 = 0.0;
  Real sinAngle243 = 0.0;
  Real apexCoeff = 0.0;   // Magnitude of central atom displacement
  scaleFactor = -0.352313;   // Scale factor (-0.352313)
  for (i=0; i<numImpropers; i++) {
    index1 = impropers[i]->x1;
    index2 = impropers[i]->x2;
    index3 = impropers[i]->x3;
    index4 = impropers[i]->x4;
    for (j=0; j<3; j++) {
      tempCoord1(j+1) = cartCoords[index1][j];
      tempCoord2(j+1) = cartCoords[index2][j];
      tempCoord3(j+1) = cartCoords[index3][j];
      tempCoord4(j+1) = cartCoords[index4][j];
    }
    r41 << tempCoord1 - tempCoord4;
    r42 << tempCoord2 - tempCoord4;
    r43 << tempCoord3 - tempCoord4;
    norm41 = r41.NormFrobenius();
    norm42 = r42.NormFrobenius();
    norm43 = r43.NormFrobenius();
    e41 << r41 / norm41;
    e42 << r42 / norm42;
    e43 << r43 / norm43;
    angle142 = acos(DotProduct(r41,r42) / (norm41 * norm42));
    angle143 = acos(DotProduct(r41,r43) / (norm41 * norm43));
    angle243 = acos(DotProduct(r42,r43) / (norm42 * norm43));
    cosAngle142 = DotProduct(r41,r42) / (norm41 * norm42);
    cosAngle143 = DotProduct(r41,r43) / (norm41 * norm43);
    cosAngle243 = DotProduct(r42,r43) / (norm42 * norm43);
    sinAngle142 = sqrt(1 - (cosAngle142 * cosAngle142));
    sinAngle143 = sqrt(1 - (cosAngle143 * cosAngle143));
    sinAngle243 = sqrt(1 - (cosAngle243 * cosAngle243));
    normVector(1) = (r41(2)*r42(3)) - (r41(3)*r42(2));
    normVector(2) = (r41(3)*r42(1)) - (r41(1)*r42(3));
    normVector(3) = (r41(1)*r42(2)) - (r41(2)*r42(1));
    normVector << normVector / normVector.NormFrobenius();
    // First elements of coordinate triples
    tempCoord5 << normVector * (scaleFactor / norm41);
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+numDihedrals+1,((index1)*3)+j) = tempCoord5(j);
    }
    // Second elements of coordinate triples
    tempCoord5 << normVector * sinAngle143 * scaleFactor;
    tempCoord5 << tempCoord5 / (norm42 * sinAngle243);
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+numDihedrals+1,((index2)*3)+j) = tempCoord5(j);
    }
    // Third elements of coordinate triples
    tempCoord5 << normVector * sinAngle142 * scaleFactor;
    tempCoord5 << tempCoord5 / (norm43 * sinAngle243);
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+numDihedrals+1,((index3)*3)+j) = tempCoord5(j);
    }
    // Fourth elements of coordinate triples
    apexCoeff = -1.0 / norm42;
    apexCoeff -= sinAngle143 / (norm42 * sinAngle243);
    apexCoeff -= sinAngle142 / (norm43 * sinAngle243);
    tempCoord5 << normVector * apexCoeff * scaleFactor;
    for (j=1; j<=3; j++) {
      B(i+numBonds+numAngles+numDihedrals+1,((index4)*3)+j) = tempCoord5(j);
    }
  }
  
  return;
}


int getInternalHessian(double** cartCoords, double* carthessian,
		       int* bondlist, int* anglelist, int* dihedlist, int* imprplist,
		       int numCartesians, int numBonds, int numAngles, int numDihedrals,
		       int numImpropers, double* hessianInternal) {
  //cout << "Running HessianTransform......";

#ifdef DEBUG
  cout << "    In getInternalHessian5\n";
  cout << "\nDemonstration of Matrix package\n";
  cout << "\nPrint a real number (may help lost memory test): " << 3.14159265 << "\n";

  cout << "numCartesians: " << numCartesians << "\n";
  cout << "numBonds: " << numBonds << "\n";
  cout << "numAngles: " << numAngles << "\n";
#endif

  // Test for any memory not deallocated after running this program
#ifdef DEBUG
  Real* s1; { ColumnVector A(8000); s1 = A.Store(); }
#endif

  {
    // The data.

#ifdef DEBUG
    cout << "  Loading Cartesians\n";
#endif
    //    Real** cartCoords = new Real*[numCartesians];
     int i=0;


#ifdef DEBUG
    cout << "  Loading Cartesian Hessian\n";
#endif
    Matrix Hc(numCartesians*3,numCartesians*3);
    Hc << carthessian;
     int j=1;
//     for (i=1; i<=numCartesians*3; i++) {
//       for (j=1; j<=numCartesians*3; j++) {
// 	Hc(i,j) = carthessian[(i-1)*numCartesians*3 + (j-1)];
//       }
//     }

#ifdef DEBUG
    cout << "  Loading Bonds\n";
#endif
    bondCoord** bonds = new bondCoord*[numBonds];
    for (i=0; i<numBonds; i++) {
      bonds[i] = new bondCoord(bondlist[i*2],bondlist[(i*2)+1]);
    }


#ifdef DEBUG
    cout << "  Loading Angles\n";
#endif
    angleCoord** angles = new angleCoord*[numAngles];
    for (i=0; i<numAngles; i++) {
      angles[i] = new angleCoord(anglelist[(i*3)],
				 anglelist[(i*3)+1],
				 anglelist[(i*3)+2]);
    }


#ifdef DEBUG
    cout << "  Loading Dihedrals\n";
#endif
    dihedralCoord** dihedrals = new dihedralCoord*[numDihedrals];
    for (i=0; i<numDihedrals; i++) {
      dihedrals[i] = new dihedralCoord(dihedlist[(i*4)],
				       dihedlist[(i*4)+1],
				       dihedlist[(i*4)+2],
				       dihedlist[(i*4)+3]);
    }


#ifdef DEBUG
    cout << "  Loading Impropers\n";
#endif
    improperCoord** impropers = new improperCoord*[numImpropers];
    for (i=0; i<numImpropers; i++) {
      impropers[i] = new improperCoord(imprplist[(i*4)],
				       imprplist[(i*4)+1],
				       imprplist[(i*4)+2],
				       imprplist[(i*4)+3]);
    }
    
    int numInternals = numBonds + numAngles + numDihedrals + numImpropers;
#ifdef DEBUG
    cout << "numInternals = " << numInternals << "\n";
#endif
    
    
    Try
      {
	Matrix B(numInternals,numCartesians*3);
	getBMatrix(cartCoords, numCartesians, bonds, numBonds, angles, numAngles,
		   dihedrals, numDihedrals, impropers, numImpropers, B);
	
#ifdef DEBUG
	cout << "B:\n";
	cout << setw(9) << setprecision(6) << (B);
	cout << "\n\n";
#endif

	Matrix G(numInternals,numInternals);
	G << B * B.t();
	
#ifdef DEBUG
	cout << "G:\n";
	cout << setw(9) << setprecision(6) << (G);
	cout << "\n\n";
#endif	

	Matrix Gi;
	getGeneralizedInverse(G,Gi);
	
#ifdef DEBUG
	cout << "Gi:\n";
	cout << setw(9) << setprecision(6) << (Gi);
	cout << "\n\n";
	
	cout << "Hc:\n";
	cout << setw(9) << setprecision(6) << (Hc);
	cout << "\n\n";
#endif	

	Matrix Hi(numInternals, numInternals);
	Hi << Gi * B * Hc * B.t() * Gi;
	
#ifdef DEBUG
	cout << "Hi:\n";
	cout << setw(9) << setprecision(3) << (Hi);
	cout << "\n\n";
#endif

	for (i=1; i<=numInternals; i++) {
	  for (j=1; j<=numInternals; j++) {
	    hessianInternal[(i-1)*numInternals + (j-1)] = Hi(i,j);
	  }
	}
	
      }
    CatchAll { cout << Exception::what(); }
    
  }
  

#ifdef DO_FREE_CHECK
  FreeCheck::Status();
#endif

#ifdef DEBUG
  Real* s2; { ColumnVector A(8000); s2 = A.Store(); }
  cout << "\n\nThe following test does not work with all compilers - see documentation\n";
  cout << "Checking for lost memory: "
       << (unsigned long)s1 << " " << (unsigned long)s2 << " ";
  if (s1 != s2) cout << " - error\n"; else cout << " - ok\n";
#endif

  //cout << "Done\n";

  return 0;
}

int getNormalModes(double** cartCoords, double* carthessian,
		   double* masslist, int numCartesians, double* frequencies,
		   double* normalmodes, int &nfreq, int decontaminate) {

  Matrix Hc(numCartesians*3,numCartesians*3);
  Hc << carthessian;

  //  mass weighted cartesian Hessian
  int i,j;
  Matrix Hmwc(numCartesians*3,numCartesians*3);
  for (i=0; i<numCartesians; i++) {
    for (j=0; j<numCartesians; j++) {
      Hmwc << Hc(i+1,j+1)/sqrt(masslist[i]*masslist[j]);
    }
  }
#ifdef DEBUG
  cout << "Hmwc:\n";
  cout << setw(9) << setprecision(3) << (Hmwc);
  cout << "\n\n";
#endif

  if (!decontaminate) {
    Matrix V;
    DiagonalMatrix eigval;
    EigenValues(Hmwc,eigval,V);    

    const float twopicinv = 1/(2*PI*LIGHTSPEED); // 2*pi*c
    for (i=0; i<numCartesians; i++) {
      frequencies[i] = twopicinv*sqrt(abs(eigval(i+1)));
      // Complex frequencies are marked with a negative sign
      if (eigval(i+1)<0) frequencies[i] *= -1.0f;
    }
  }

  // Moments of inertia tensor
  double Ixx=0, Iyy=0, Izz=0, Ixy=0, Ixz=0, Iyz=0, x, y, z, m, mtot=0.0;
  double Rcom[3] = { 0.0, 0.0, 0.0 };
  for (i=0; i<numCartesians; i++) {
    x = cartCoords[i][0];
    y = cartCoords[i][1];
    z = cartCoords[i][2];
    m = masslist[i];

    // Center of mass
    Rcom[0] += m*x;
    Rcom[1] += m*y;
    Rcom[2] += m*z;
    mtot += m;
    Ixx += m*(y*y+z*z);
    Iyy += m*(x*x+z*z);
    Izz += m*(x*x+y*y);
    Ixy -= m*x*y;
    Ixz -= m*x*z;
    Iyz -= m*y*z;
  }

  Rcom[0] /= mtot;
  Rcom[1] /= mtot;
  Rcom[2] /= mtot;

  SymmetricMatrix I(3);
  I << Ixx << Ixy << Ixz 
    << Ixy << Iyy << Iyz 
    << Ixz << Iyz << Izz;

  // Diagonalize moments of intertia
  DiagonalMatrix eigval;
  Matrix X;
  Jacobi(I, eigval, X);

#ifdef DEBUG
  cout << "I:\n";
  cout << setw(9) << setprecision(3) << (I);
  cout << "\n\n";

  cout << "X:\n";
  cout << setw(9) << setprecision(3) << (X);
  cout << "\n\n";
#endif

  // Coords wrt COM
  Matrix R(numCartesians, 3);
  for (i=0; i<numCartesians; i++) {
    R(i,0) = cartCoords[i][0]-Rcom[0];
    R(i,1) = cartCoords[i][1]-Rcom[1];
    R(i,2) = cartCoords[i][2]-Rcom[2];
  }

#ifdef DEBUG
  cout << "R:\n";
  cout << setw(9) << setprecision(3) << (R);
  cout << "\n\n";
#endif

  RowVector Px = R*I.Row(1);
  RowVector Py = R*I.Row(2);
  RowVector Pz = R*I.Row(3);

  double sqrm;
  IdentityMatrix E(3*numCartesians);
  Matrix D(3*numCartesians,3*numCartesians);
  D << E;

  for (i=0; i<numCartesians; i++) {
    sqrm = sqrt(masslist[i]);
    D(1,i*3+1) = D(2,i*3+2) = D(3,i*3+3) = sqrm;
    D(1,i*3+2) = D(1,i*3+3) = D(2,i*3+1) = D(2,i*3+3) = D(3,i*3+1) = D(3,i*3+2) = 0.0;
    for (j=1; j<=3; j++) {
      D(4,i*3+j) = (Py(i)*X(j,3)-Pz(i)*X(j,2))/sqrm;
      D(5,i*3+j) = (Pz(i)*X(j,1)-Px(i)*X(j,3))/sqrm;
      D(6,i*3+j) = (Px(i)*X(j,2)-Py(i)*X(j,1))/sqrm;
    }
  }

#ifdef DEBUG
  cout << "D:\n";
  cout << setw(9) << setprecision(3) << (D);
  cout << "\n\n";
#endif

  // Normalize D
  int ntransrot=6;
  double s[6];
  RowVector Ri, Rj; 
  for (i=0; i<6; i++) {
    //Ri = D.Row(i+1);
    //s[i] = DotProduct(Ri, Ri);
    s[i] = D.Row(i).NormFrobenius();
    if (s[i]<1e-6) ntransrot--;
    else  D.Column(i+1) *= 1.f/sqrt(s[i]);
  }


  // Remove the row with zero elements
  if (ntransrot<6) {
    int k=1;
    for (i=0; i<6; i++) {
      if (s[i]>=1e-6) {
	D.Row(k)=D.Row(i+1);
	k++;
      }
    }
  }

#ifdef DEBUG
  cout << "D normalized:\n";
  cout << setw(9) << setprecision(3) << (D);
  cout << "\n\n";
#endif

  int Nvib = 3*numCartesians-ntransrot;
  cout << "Nvib = 3N-"<<ntransrot<<" = "<<Nvib<<" nonredundant internal coordinates"<<"\n";

  // We create a 3N*3N transformation matrix D which transforms from mass weighted
  // cartesian coordinates q to internal coordinates S = Dq, where rotation and 
  // translation have been separated out.

  // The first five or six vectors are already linearly independent,
  // and we just orthogonalized them above. 
  // Now the remaining 3N-6 vectors are generated using the stabilized 
  // Gram-Schmidt algorithm. They are required to be orthogonal to the
  // translational rotational vectors.
  double dot , norm;
  for (i=1; i<=ntransrot; i++) {
    for (j=1; j<i; j++) {
      // Remove component of D_i that is parallel to D_k:
      // D_i = D_i - <D_i, D_k> * D_k
      Ri = D.Row(i);
      Rj = D.Row(j);
      dot = DotProduct(Rj, Ri);
      D.Row(i) -= dot*Rj;
    }
    // Normalize
    //norm = DotProduct(D.Row(i), D.Row(i));
    norm = D.Row(i).NormFrobenius();
    D.Row(i) *= 1/sqrt(norm);
  }
  
  int k, cik=0;
  for (i=ntransrot; i<=3*numCartesians; i++) {

    norm = 0.0;
    while (abs(norm) < 1e-12) {

      // Initialize i'th row with cik'th coord axis
      D.Row(i) = E.Row(cik);

      // Stabilized Gram-Schmidt:
      for (k=0; k<i; k++) {
	// Remove component of D_i that is parallel to D_k (k'th row):
	// D_i = D_i - <D_i, D_k> * D_k
	// <D_i, D_k> = D_k[cik], since D_i[j]=0 for j!=cik and D_i[cik]=1
	D.Row(i) -= D(k,cik)*D.Row(k);
      }

      // Length of D_i:
      norm = D.Row(i).NormFrobenius();
	
      // Check for the linear independence:
      // If norm==0 it means that all components parallel to another existing
      // vector (row of D) summed up to D_i, i.e the new D_i is linear dependent
      // on the other vectors. In this case we choose another D_cik for the
      // construction of D_i.
      if ( norm < 1e-6) {                        
	cik++; // take another vector
	norm = 0.0;
      }
    }
    D.Row(i) *= 1.f/norm;

    cik++;
  }

#ifdef DEBUG
  cout << "D orthonormalized:\n";
  cout << setw(9) << setprecision(3) << (D);
  cout << "\n\n";
#endif

  Matrix Hint;
  Hint = D.t() * Hmwc * D;

  Matrix V;
  EigenValues(Hint,eigval,V);    

  frequencies = new double[Nvib];

  const double scale = 1.f/(2.f*PI*LIGHTSPEED);
  for (i=0; i<Nvib; i++) {
    frequencies[i] = scale*sqrt(eigval(i+1));
    // Complex frequencies are marked with a negative sign
    if (eigval(i+1)<0) frequencies[i] *= -1.0f;
  }

  return Nvib;
}

// Stabilized Gram-Schmidt orthogonalization
int orthogonalize(Matrix M, Matrix &OM) {
  return 1;
}
