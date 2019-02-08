package provide plumed 2.7
namespace eval ::Plumed {
    variable templates_list_v1 {
	"Group definition"			{groupname-> [chain A] groupname<-} 
	"-" "-"
	"Absolute position"			"POSITION	LIST <xx>	DIR XYZ"
	"Distance"				"DISTANCE	LIST <xx> <yy>  DIR XYZ"
	"Minimum distance"			"MINDIST	LIST <xx> <yy>  BETA 500"
	"Angle"					"ANGLE		LIST <xx> <oo> <yy>"
	"Dihedral"				"TORSION	LIST <xx> <yy> <zz> <ww>"
	"Contacts"				"COORD		LIST <xx> <yy>  NN 6 MM 12 D_0 2.5 R_0 0.5"
	"Hydrogen bonds"			"HBONDS		LIST <xx> <yy>  TYPE nn"
	"Interfacial water"			"WATERBRIDGE	LIST <type1> <type2> <solvent> NN 8 MM 12 R_0 4.0"
	"Radius of gyration"			"RGYR		LIST <xx>"
	"Trace of the inertia tensor"		"INERTIA	LIST <xx>"
	"Electrostatic potential"		"ELSTPOT	LIST <xx> <yy>  R_0 4.0   CUT 12.0"
	"Electric dipole"			"DIPOLE		LIST <xx>"
	"-" "-"
	"RMSD from a reference"			"TARGETED TYPE MSD SQRT FRAMESET /tmp/reference.pdb  ! use absolute pathname, < 40 chars"
	"S-path variable"			"S_PATH TYPE MSD FRAMESET /tmp/frame_ NFRAMES 2 LAMBDA 9.0  ! use absolute pathnames, < 40 chars"
	"Z-path variable"			"Z_PATH TYPE MSD FRAMESET /tmp/frame_ NFRAMES 2 LAMBDA 9.0  ! use absolute pathnames, < 40 chars"
	"-" "-"
	"Dihedral correlation"			"DIHCOR NDIH 3\nA1 B1 C1 D1\nA2 B2 C2 D2\nA3 B3 C3 D3\n"
	"Alpha-beta similarity"			"ALPHABETA NDIH 3\nA1 B1 C1 D1 ref1\nA2 B2 C2 D2 ref2\nA3 B3 C3 D3 ref3\n"
	"Alpha-helix RMSD"			"ALPHARMSD	LIST <xx>  R_0 0.8  NN 8  MM 12  NOPBC"
	"Antiparallel beta-sheet RMSD"		"ANTIBETARMSD	LIST <xx>  R_0 0.8  NN 8  MM 12  STRANDS_CUTOFF 10  NOPBC"
	"Parallel beta-sheet RMSD"			"PARABETARMSD	LIST <xx>  R_0 0.8  NN 8  MM 12  STRANDS_CUTOFF 10  NOPBC"
	"Torsional RMSD"			"RMSDTOR NDIH 3\nA1 B1 C1 D1 ref1\nA2 B2 C2 D2 ref2\nA3 B3 C3 D3 ref3\n"
	"Puckering coordinates"			"PUCKERING	LIST <xx>  TYPE  [PHI|THETA|Q]"
	"-" "-"
	"Upper wall (allow lower)"		"UWALL CV <nn> LIMIT <xx>   KAPPA 100.0 EXP 4.0 EPS 1.0 OFF 0.0"
	"Lower wall (allow higher)"		"LWALL CV <nn> LIMIT <xx>   KAPPA 100.0 EXP 4.0 EPS 1.0 OFF 0.0"
    }
}

