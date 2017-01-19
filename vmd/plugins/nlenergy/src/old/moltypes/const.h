/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/const.h
 * @brief   Important unit conversion and physical constants.
 * @author  David J. Hardy
 * @date    Mar. 2008
 */

#ifndef MOLTYPES_CONST_H
#define MOLTYPES_CONST_H


/*
 * mass (AMU) range to distinguish particular atoms
 */
#undef  MASS_HYDROGEN_MIN
#define MASS_HYDROGEN_MIN   1.0
#undef  MASS_HYDROGEN_MAX
#define MASS_HYDROGEN_MAX   1.5
#undef  MASS_OXYGEN_MIN
#define MASS_OXYGEN_MIN    15.5
#undef  MASS_OXYGEN_MAX
#define MASS_OXYGEN_MAX    16.5


/*
 * multiplicative constants to convert external velocity units (A/ps)
 * of PDB file into internal velocity units (A/fs) and back again
 */
#undef  PDBATOM_VELOCITY_INTERNAL
#define PDBATOM_VELOCITY_INTERNAL  1e-3
#undef  PDBATOM_VELOCITY_EXTERNAL
#define PDBATOM_VELOCITY_EXTERNAL  1000.0


/*
 * multiplicative constants to convert NAMD velocity units sqrt(kcal/mol/AMU)
 * of NAMD binary file into internal velocity units (A/fs) and back again
 */
#undef  NAMD_VELOCITY_INTERNAL
#define NAMD_VELOCITY_INTERNAL  0.02045482706
#undef  NAMD_VELOCITY_EXTERNAL
#define NAMD_VELOCITY_EXTERNAL  48.8882158263527
/*
 * Explanation:
 *
 * NAMD internal velocity units are sqrt(kcal/mol/AMU).
 * This comes about by scaling the fs time step dt by a
 * time factor T.  Units are chosen for T so that
 * multiplication of the force/mass = (kcal/mol/A) / AMU
 * by (dt/T)^2 gives position units A.
 *
 * http://www.ks.uiuc.edu/Research/namd/mailing_list/namd-l/4125.html
 */


/*
 * multiplicative constants to convert external energy units (kcal/mol)
 * into internal energy units (AMU*(A/fs)^2) and back again
 */
#undef  ENERGY_INTERNAL
#define ENERGY_INTERNAL    0.0004184
#undef  ENERGY_EXTERNAL
#define ENERGY_EXTERNAL    (1.0 / ENERGY_INTERNAL)


/*
 * Coulomb constant for electrostatics, units (AMU*(A/fs)^2)*(A/e^2)
 */
#undef  COULOMB
#define COULOMB            (332.0636 * ENERGY_INTERNAL)


/*
 * Boltzmann constant, units (AMU*(A/fs)^2)*(1/K)
 */
#undef  BOLTZMANN
#define BOLTZMANN          (0.001987191 * ENERGY_INTERNAL)


/*
 * convert angle from degrees to radians
 */
#undef  RADIANS
#define RADIANS            (M_PI / 180.0)


/*
 * convert angle from radians to degrees
 */
#undef  DEGREES
#define DEGREES            (180.0 / M_PI)


#endif /* MOLTYPES_CONST_H */
