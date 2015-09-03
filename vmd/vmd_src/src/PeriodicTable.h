/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: PeriodicTable.h,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $       $Date: 2008/03/27 19:38:22 $
 *
 ***************************************************************************/

/*
 * periodic table of elements and helper functions to convert
 * ordinal numbers to labels and back.
 *
 * 2002-2008 axel.kohlmeyer@theochem.ruhr-uni-bochum.de, vmd@ks.uiuc.edu
 */

const char *get_pte_label(const int idx);
float get_pte_mass(const int idx);
float get_pte_vdw_radius(const int idx);
int get_pte_idx_from_string(const char *label);

