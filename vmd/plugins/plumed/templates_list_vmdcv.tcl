package provide plumed 2.7
namespace eval ::Plumed {

    proc vmdcv_gencv1 {name content} {
	return "colvar {\n  name $name\n  $content\n}\n"
    }

    proc vmdcv_gencv2 {type blocks} {
	set o "  $type {\n"
	foreach bl $blocks {
		append o "      $bl { ... }\n"
	}
	append o "  }"
	return [vmdcv_gencv1 "my_$type" $o]
    }

    variable templates_list_vmdcv [list \
	"Empty colvar"			"colvar {\n  name myname\n  ...\n}" \
	- - \
	"Distance"			[vmdcv_gencv2 distance {group1 group2}] \
	"Distance projected on axis"	[vmdcv_gencv2 distanceZ {main ref ref2 axis}] \
	"Distance projected on plane"	[vmdcv_gencv2 distanceXY {main ref ref2 axis}] \
	"Distance vector"		[vmdcv_gencv2 distanceVec {group1 group2}] \
	"Distance unit vector"		[vmdcv_gencv2 distanceDir {group1 group2}] \
	"Generalized mean distance"	[vmdcv_gencv2 distanceInv {group1 group2 exponent}] \
	"Cartesian coordinates"		[vmdcv_gencv2 cartesian {atoms}] \
	"Angle between three groups"  	[vmdcv_gencv2 angle {group1 group2 group3}] \
	"Dihedral"		  	[vmdcv_gencv2 dihedral {group1 group2 group3 group4}] \
	"Radius of gyration"		[vmdcv_gencv2 gyration {atoms}] \
	"Inertia moment"		[vmdcv_gencv2 inertia {atoms}] \
	"Inertia around an axis"	[vmdcv_gencv2 inertiaZ {atoms axis}] \
	- - \
	"Coordination number"		[vmdcv_gencv2 coordNum {group1 group2 cutoff}] \
	"Self-coordination number"	[vmdcv_gencv2 selfCoordNum {group1 cutoff}] \
	"Hydrogen bonding"		[vmdcv_gencv2 hBond {acceptor donor cutoff}] \
	- - \
	"RMSD"				[vmdcv_gencv2 rmsd {atoms refPositionsFile refPositionsCol refPositionsColValue}] \
	"Eigenvector"  			[vmdcv_gencv2 eigenvector {atoms refPositionsFile refPositionsCol refPositionsColValue
	                                                           vectorFile vectorCol vectorColValue differenceVector}] \
    ]
}

