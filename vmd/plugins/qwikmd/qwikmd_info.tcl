#
# $Id: qwikmd_info.tcl,v 1.6 2016/10/13 17:14:38 johns Exp $
#
#==============================================================================

proc QWIKMD::selectInfo {} {
set text1 "Selection Window \n\n"
set font1  "title"

set text2 "The selection section of QwikMD allows the user to select each parts of the PDB \
will be prepared for the MD simulation. For example, structures obtained with NMR spectroscopy \
usually have more than one state of the protein. It is also common that PDB structures solved \
with X-ray Crystallography have oligomers of the protein, separated in different chains. These \
oligomers are frequently an effect of the crystallization process and the protein in solution \
is presented as a monomer. QwikMD allows the user to select the desired NMR state or protein \
chains for the MD step. \n\n"
set font2 "text"

set text3 "NMR Structures \n\n"
set font3 "subtitle"

set text4 "Nuclear magnetic resonance spectroscopy of proteins, usually abbreviated protein NMR, is\
 a field of structural biology in which NMR spectroscopy is used to obtain information of the structure\
 and dynamics of proteins, nucleic acids, and their complexes. NMR structure in the PDB usually have\
 multiple steps. In order to start a MD simulation one have to select one of the steps as the initial\
 coordinates. It is usual, when running more than one simulation of the same system, to select different\
 initial steps to improve sampling of conformational structure.\n\n"
set font4 "text"

set text5 "X-ray Crystallography \n\n"
set font5 "subtitle"

set text6 "X-ray crystallography methods utilize the optical rule that electromagnetic radiation \
will interact most strongly with matter the dimensions of which are close to the wavelength of \
that radiation. X-rays are diffracted by the electron clouds in molecules, since both the \
wavelength of the X-rays and the diameter of the cloud are on the order of Angstroms. The \
diffraction patterns formed when a group of molecules is arranged in a regular, crystalline \
array, may be used to reconstruct a 3-D image of the molecule. Hydrogen atoms, however, are \
not typically detected by X-ray crystallography since their sizes are too small to interact \
with the radiation and since they contain only a single electron.  \n\n"
set font6 "text"


set text7 "Select chain/type \n\n"
set font7 "subtitle"

set text8 "With this button the user can select or deselect chains and types of \
molecules inside these chains. The selected groups will be the one that will be \
present in the MD simulation prepared with QwikMD.  \n\n"
set font8 "text"


set text9 "Structure Manipulation \n\n"
set font9 "subtitle"

set text10 "This button will open a new window where the user can do mutations, rename \
molecules that have wrong names - read more below - change protonation states, delete \
parts of the molecules and also inspect the structure with a interactive \
residue/molecule list. Select Resid is especially important in cases where one of \
the molecules/ions have wrong names, or names that are different from the name \
used in the CHARMM force field. For instance, it is common in the PDB that Ca2+ \
ions have the name CA. CHARMM recognize the name CA as alpha-Carbon of protein \
structures, and CA resname - residue name - is not recognized by CHARMM. Select Resid \
allow for the user to rename CA ions to proper Calcium parameters that will be \
compatible with the CHARMM force field.  \n\n"
set font10 "text"

set text11 "QwikMD Main Window \n\n"
set font11 "subtitle"

set text12 "In the main window of QwikMD, every chain is separated by type of molecule forming \
a group - chain/type. VMD has several types of molecules, including: protein, nucleic, \
lipids, water, among others. The user can select different Representations for each of \
the groups and also different colors.  \n\n\n\n"
set font12 "text"

set text13 "Scripting in VMD \n\n"
set font13 "title"

set text14 "Providing VMD with user made scripts can do the steps you are doing employing QwikMD. \
Advanced VMD users usually prefer to create their own scripts, as these scripts allow for total \
control and reproducibility of the analysis. Scripts are very powerful tools inside VMD as they \
allow the user to easily perform analyses that are not yet implemented. To learn more about \
scripting with VMD visit the link at the bottom of this window. \n\n"
set font14 "text"
    
    set link {http://www.ks.uiuc.edu/Training/Tutorials/vmd/bak/node4.html}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] [list $text7 $font7] [list $text8 $font8] [list $text9 $font9] [list $text10 $font10] [list $text11 $font11] [list $text12 $font12] [list $text13 $font13] [list $text14 $font14] ]
    set title "Selection section info"
    return [list ${text} $link $title]
}



proc QWIKMD::rmsdInfo {} {

    set text1 "Root-Mean-Square Deviation \n\n"
    set font1 "title"
    
    set text2 "The root-mean-square deviation (RMSD) is the measure of the average distance between the atoms (usually the backbone atoms) of superimposed proteins. In the study of globular protein conformations, one customarily measures the similarity in three-dimensional structure by the RMSD of the C\u03b1 atomic coordinates after optimal rigid body superposition. When a dynamical system fluctuates about some well-defined average position, the RMSD from the average over time can be referred to as the RMSF or root mean square fluctuation. The size of this fluctuation can be measured, for example using M\u00F6ssbauer spectroscopy or nuclear magnetic resonance, and can provide important physical information.     \n\n"
    set font2 "text"
    
    set text3 "QwikMD allows the user to perform RMSD analysis during live NAMD simulations. More advanced options for RMSD analysis can be done with VMD plugins available in VMD Main menu item Extensions - Analysis. To read more about VMD and its plugins check the link at the bottom of this window.   \n\n"
    set font3 "text"

    set link {http://www.ks.uiuc.edu/Research/vmd/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::rmsfInfo {} {

    set text1 "Root-Mean-Square Fluctuation \n\n"
    set font1 "title"
    
    set text2 "The root-mean-square fluctuation (RMSF) is the measure of the average atomic mobility (usually the backbone atoms) of superimposed proteins.    \n\n"
    set font2 "text"
    
    set text3 "QwikMD allows the user to perform RMSF analysis during live NAMD simulations. More advanced options for RMSF analysis can be done with VMD plugins available in the VMD Main menu item Extensions - Analysis. To read more about VMD and its plugins check the link at the bottom of this window.   \n\n"
    set font3 "text"

    set link {http://www.ks.uiuc.edu/Research/vmd/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::sasaPlotInfo {} {

    set text1 "Solvent Accessible Surface Area \n\n"
    set font1 "title"

    set link {http://www.ks.uiuc.edu/Research/vmd/}
    set text [list [list $text1 $font1]]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::nscaPlotInfo {} {

    set text1 "Contacting Surface Area \n\n"
    set font1 "title"

    set link {http://www.ks.uiuc.edu/Research/vmd/}
    set text [list [list $text1 $font1]]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::specificHeatPlotInfo {} {

    set text1 "Specific Heat \n\n"
    set font1 "title"

    set text2 "Specific heat is an important property of thermodynamic systems. It is the amount of heat needed to raise the temperature of an object per degree of temperature increased per unit mass, and a sensitive measure of the degree of order in a system. (Near a bistable point, the specific heat becomes large.)    \n\n"
    set font2 "text"

    set text3 "For this calculation, you need to carry out a long simulation in the NVT (canonical) ensemble that samples sufficiently the averages  <E\u00b2> and  <E> that arise in the definition of cv. This calculation is long and computationally intensive. Please follow the namd tutorial (the hyperlink present below).    \n\n"
    set font3 "text"

    global tcl_platform env
    set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-unix-html/node14.html#SECTION00051500000000000000}
     if {[string first "Windows" $::tcl_platform(os)] != -1} {
        set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-win-html/node15.html#SECTION00061500000000000000}
    }

    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3]]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::tempDistPlotInfo {} {

    set text1 "Temperature Distribution \n\n"
    set font1 "title"

    set text2 "Evaluate the temperature distribution is a good checking measurement to evaluate the thermodynamic stability of the simulation.    \n\n"
    set font2 "text"

    set text3 "For this calculation, you need to carry out a long simulation in the NVT (canonical) ensemble that sufficiently samples the fluctuations in kinetic energy, and hence in the temperature. This calculation is long and computationally intensive. Please follow the namd tutorial (the hyperlink present below).    \n\n"
    set font3 "text"

    global tcl_platform env
    set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-unix-html/node14.html#SECTION00051400000000000000}
     if {[string first "Windows" $::tcl_platform(os)] != -1} {
        set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-win-html/node15.html#SECTION00061400000000000000}
    }

    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3]]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::mbDistributionPlotInfo {} {

    set text1 "Maxwell-Boltzmann Energy Distribution \n\n"
    set font1 "title"

    global tcl_platform env
    set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-unix-html/node14.html#SECTION00051200000000000000}
     if {[string first "Windows" $::tcl_platform(os)] != -1} {
        set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-win-html/node15.html#SECTION00061200000000000000}
    }

    set text [list [list $text1 $font1]]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::tQuenchPlotInfo {} {

    set text1 "Temperature Quench echo \n\n"
    set font1 "title"

    set text2 "The motions of atoms in globular proteins (e.g., ubiquitin), referred to as internal dynamics, comprise a wide range of time scales, from high frequency vibrations about their equilibrium positions with periods of several femtoseconds to slow collective motions which require seconds or more, leading to deformations of the entire protein.\n\n"
    set font2 "text"


    set text3 "The internal dynamics of these proteins on a picosecond time scale (high frequency) can be described as a collection of weakly interacting harmonic oscillators referred to as normal modes. Since normal modes are formed by linear superposition of a large number of individual atomic oscillations, it is not surprising that the internal dynamics of proteins on this time scale has a delocalized character throughout the protein. The situation is similar to the lattice vibrations (phonons) in a crystalline solid. Experimentally, there exist ways to synchronize, through a suitable signal or perturbation, these normal modes, forcing the system in a so-called (phase) coherent state, in which normal modes oscillate in phase. The degree of coherence of the system can be probed with a second signal which through interference with the coherent normal modes may lead to resonances, referred to as echoes, which can be detected experimentally. However, the coherence of atomic motions in proteins decays through non-linear contributions to forces between atoms. This decay develops on a time scale \u03C4 which can be probed, e.g., by means of temperature echoes, and can be described by employing MD simulations.\n\n"
    set font3 "text"

    
    set text4 "In a temperature echo the coherence of the system is probed by reassigning the same atomic velocities the system had at an earlier time and then looking to an echo in the temperature at time \u03C4e, as a result of such reassignment.     \n\n"
    set font4 "text"

    set text5 "For more details and instructions please follow the namd tutorial (the hyperlink present below).    \n\n"
    set font5 "text"

    global tcl_platform env
    set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-unix-html/node15.html#SECTION00052200000000000000}
     if {[string first "Windows" $::tcl_platform(os)] != -1} {
        set link {http://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-win-html/node16.html#SECTION00062300000000000000}
    }

    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5]]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]

}

proc QWIKMD::introInfo {} {

    
set text1 "Molecular dynamics (MD)"
set font1  "subtitle"

set text2 " is a computer simulation of physical movements of atoms and molecules, widely used to study \
biological systems. The atoms and molecules are allowed to interact, giving a view of the motion of the \
atoms and consequently of protein domains in the case of protein simulations. MD has emerged as an important \
research methodology covering systems to the level of millions of atoms.\n\n"
set font2 "text"

set text3 "QwikMD"
set font3 "subtitle"

set text4 " is a VMD plugin to help to start and analyze MD simulations. The plugin helps, specially \
scientists that are starting to perform MD simulations, to prepare the necessary files to run these simulations \
in desktop machines all the way to large supercomputers. All the necessary steps, from the PDB to the \
configuration file is created with simple procedures so the user one can use the plugin to learn how to prepare MD simulations.\
 The live simulation option allows for the visualization and analysis of the simulation on the fly, helping new users to learn more about MD \
simulations and expert users to test their simulations before submitting it to run in a supercomputer. QwikMD \
integrates VMD and NAMD, two widely used software developed by the Theoretical and Computational Biophysics Group \
at University of Illinois at Urbana-Champaign.\n\n"
set font4 "text"

set text5 "NAMD (NAnoscale Molecular Dynamics)\n\n"
set font5 "title"

set text6 "Recipient of a 2002 Gordon Bell Award and a 2012 Sidney Fernbach Award, NAMD is a parallel molecular \
dynamics code designed for high-performance simulation of large biomolecular systems. Based on Charm++ parallel \
objects, NAMD scales to hundreds of cores for typical simulations and beyond 500,000 cores for the largest simulations. \
NAMD uses the popular molecular graphics program VMD for simulation setup and trajectory analysis, but is also \
file-compatible with AMBER, CHARMM, and X-PLOR. NAMD is distributed free of charge with source code. You can build \
NAMD yourself or download binaries for a wide variety of platforms. To run QwikMD you need NAMD installed in your \
machine and available in your path.\n\n"
set font6 "text"

set text6a "For LINUX/MAC users:\n"
set font6a "subtitle"

set text6b "Setting the Path: To start to use QwikMD you will need to add the namd2 directory to your path in order for the \
operational system to locate it. To perform that, add to your .bashrc (Linux) or .Profile (Mac) in your home folder \
the following line: \n"
set font6b "text"

set text6c "export PATH\
=<complete.path.for.namd\
>:<complete.path.for.namd>):$\
PATH  \n"
set font6c "text"

set text6d "Where (<complete.path.for.namd>) is the complete path to the actual folder where the namd2 executable is available. \
Example: \n"
set font6d "text"

set text6e "export PATH\
=/usr/local/NAMD_2.10:/usr/local/NAMD_2.10:$\
PATH \n"
set font6e "text"

set text6f "If you are new to Linux, visit our guide at: http://www.ks.uiuc.edu/Training/Tutorials/Reference/unixprimer.html \n\n"
set font6f "text"

set text6g "For Windows users: \n"
set font6g "subtitle"

set text6h "Setting the Path: To start to use QwikMD you will need to add the namd2 directory to your path in order for Windows \
to locate it. This can be accomplished by right-clicking Computer on the Desktop and selecting Properties - Advanced \
system settings - Advanced - Environment Variables (the precise procedure may vary depending on your version of Windows). \
Under System variables, scroll down and select Path and then Edit. At the end of the long line in Variable Value, add a \
semi-colon ; then the full path to the directory containing namd2 (but do NOT add the executable \"namd2\" at the end). \
Click OK. Now open a new command prompt. Regardless of the directory you are in, you should be able to type namd2 and run it. \n\n"
set font6h "text"

set text7 "Setting up a Molecular Dynamics Simulation:\n\n"
set font7 "subtitle"

set text8 "In order to run any MD simulation, NAMD requires at least four things:\n\n"
set font8 "text"

set text8a "1.|   a Protein Data Bank (pdb) file which stores atomic coordinates and/or velocities for the system. \
Pdb files may be generated by hand, but they are also available via the Internet for many proteins \
at http://www.pdb.org. \n\n"
set font8a "text"

set text8b "2.|   a Protein Structure File (psf) which stores structural information of the protein, such as various \
types of bonding interactions. \n\n"
set font8b "text"

set text8c "3.|   a force field parameter file. A force field is a mathematical expression of the potential which atoms \
in the system experience. CHARMM, X-PLOR, AMBER and GROMOS are four types of force fields, and NAMD is \
able to use all of them. The parameter file defines bond strengths, equilibrium lengths, etc. \n\n"
set font8c "text"

set text8d "4.|   a configuration file, in which the user specifies all the options that NAMD should adopt in running a \
simulation. The configuration file tells NAMD how the simulation is to be run. \n\n"
set font8d "text"

set text8e "QwikMD helps the user to get and/or prepare all the aforementioned files and setup MD simulations. \n\n"
set font8e "text"

set text9 "The PDB files:\n\n"
set font9 "subtitle"

set text10 "The term PDB can refer to the Protein Data Bank (http://www.rcsb.org/pdb/), to a data file \
provided there, or to any file following the PDB format. Files in the PDB include information such as the \
name of the compound, the species and tissue from which is was obtained, authorship, revision history, \
journal citation, references, amino acid sequence, stoichiometry, secondary structure locations, crystal \
lattice and symmetry group, and finally the ATOM and HETATM records containing the coordinates of the \
protein and any waters, ions, or other heterogeneous atoms in the crystal. Some PDB files include multiple \
sets of coordinates for some or all atoms. Due to the limits of x-ray crystallography and NMR structure \
analysis, the coordinates of hydrogen atoms are usually not included in the PDB. \n"
set font10 "text"

set text10a "NAMD and VMD ignore everything in a PDB file except for the ATOM and HETATM records, and when writing \
PDB files the ATOM record type is used for all atoms in the system, including solvent and ions. If you \
open a PDB file with a text viewer, the fields seen in order from left to right are the record type, atom \
ID, atom name, residue name, residue ID, x, y, and z coordinates, occupancy, temperature factor (called beta), \
segment name, and line number. If this file is loaded into VMD and then written out as a new file, most of \
the extra information will be removed, the HETATM records will become ATOM records, and the previously empty \
chain ID field (between residue name and residue ID) will be set to X (unless present in the original file), \
and the line number will be omitted. \n\n"
set font10a "text"


set text11 "The PSF files:\n\n"
set font11 "subtitle"

set text12 "A PSF file, also called a protein structure file, contains all of the molecule specific information \
needed to apply a particular force field to a molecular system. The CHARMM force field is divided into a \
topology file, which is needed to generate the PSF file, and a parameter file, which supplies specific numerical \
values for the generic CHARMM potential function. The topology file defines the atom types used in the force field; \
the atom names, types, bonds, and partial charges of each residue type; and any patches necessary to link or \
otherwise mutate these basic residues. The parameter file provides a mapping between bonded and nonbonded \
interactions involving the various combinations of atom types found in the topology file and specific spring \
constants and similar parameters for all of the bond, angle, dihedral, improper, and van der Waals terms in the \
CHARMM potential function.\n"
set font12 "text"

set text12a "The PSF file contains six main sections of interest: atoms, bonds, angles, dihedrals, impropers (dihedral \
force terms used to maintain planarity), and cross-terms. After preparing your files with QwikMD you can open \
the PSF file from the SETUP folder in a text editor and check how a PSF file looks like. Note that this SETUP \
folder is located inside the folder that you are creating for your work when pressing Prepare. \n\n"
set font12a "text"

set text13 "The Parameter files: \n\n"
set font13 "subtitle"

set text14 "A CHARMM force field parameter file contains all of the numerical constants needed to evaluate forces and \
energies, given a PSF structure file and atomic coordinates. The parameter file is closely tied to the topology file \
that was used to generate the PSF file, and the two are typically distributed together and given matching names. \
QwikMD uses CHARMM36 force field and its parameter file is available in the RUN folder located inside the folder \
that you are creating for your work when pressing Prepare. \n\n"
set font14 "text"

set text15 "The NAMD Configuration file: \n\n"
set font15 "subtitle"

set text16 "The NAMD configuration file (also called a config file, .conf file, or .namd file) is given to NAMD \
on the command line and specifies virtually everything about the simulation to be done. The only exceptions \
are details relating to the parallel execution environment, which vary between platforms. Therefore, the config \
file should be portable between machines, platforms, or numbers of processors in a run, as long as the referenced \
input files are available. QwikMD uses a standard configuration file that uses safe parameters set, which will \
most likely work for the system that can be prepared with QwikMD. The configuration file is available in the RUN \
folder located inside the folder that you are creating for your work when pressing Prepare. \n\n"
set font16 "text"

set text17 "Setting up a simulation with QwikMD: \n\n"
set font17 "title"

set text18 "To start setting up a MD simulation with NAMD first load a PDB file using the Browser button \
below or typing a PDB code in the blank space. If this is your first time working with MD simulations of \
proteins you can use, as an example, the PDB code 1UBQ. Ubiquitin (PDB code 1UBQ) is a small regulatory \
protein that is present in almost all tissues (ubiquitously) of eukaryotic organisms. \n" 
set font18 "text"

set text19 "After pressing Load you load the PDB file from your computer or you will automatically download \
from the PDB website the PDB file. Your VMD Display window will show now the structure of the molecule \
(or molecules) you just loaded. All the different chains and types of molecules will be separated in \
the QwikMD selection window. \n\n"
set font19 "text"

set text20 "You will find more information about QwikMD and how to run MD simulations in the \
information buttons of each QwikMD section.  \n\n\n\n\n\n\n\n"
set font20 "subtitle"

set text21 "When using NAMD, VMD and QwikMD please cite the following articles: \n\n"
set font21 "title"

set text22 "VMD: visual molecular dynamics - W Humphrey, A Dalke, K Schulten - Journal of Molecular Graphics - 14 (1), 33-38 - 1996\n\n"
set font22 "subtitle"

set text23 "Scalable molecular dynamics with NAMD - James C Phillips, Rosemary Braun, Wei Wang, James Gumbart, Emad \
Tajkhorshid, Elizabeth Villa, Christophe Chipot, Robert D Skeel, Laxmikant Kale, Klaus Schulten - Journal of \
Computational Chemistry -  26 (16), 1781-1802 - 2005 \n\n"
set font23 "subtitle"


    
    
    set link {http://www.ks.uiuc.edu/Research/namd/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] [list $text6a $font6a] [list $text6b $font6b] [list $text6c $font6c] [list $text6d $font6d] [list $text6e $font6e] [list $text6f $font6f] [list $text6g $font6g] [list $text6h $font6h] [list $text7 $font7] [list $text8 $font8] [list $text8a $font8a] [list $text8b $font8b] [list $text8c $font8c] [list $text8d $font8d] [list $text8e $font8e] [list $text9 $font9] [list $text10 $font10] [list $text10a $font10a] [list $text11 $font11] [list $text12 $font12] [list $text12a $font12a] [list $text13 $font13] [list $text14 $font14] [list $text15 $font15] [list $text16 $font16] [list $text17 $font17] [list $text18 $font18] [list $text19 $font19]  [list $text20 $font20]  [list $text21 $font21]  [list $text22 $font22]  [list $text23 $font23]   ]
    set title "Introduction to Molecular Dynamics and QwikMD"
    return [list ${text} $link $title]
}
















proc QWIKMD::analyInfo {} {

    set text1 "Analysis with the Computational Microscope \n\n"
    set font1 "title"
    
    set text2 "VMD is a powerful tool for analysis of structures and trajectories and should be used as a tool to think. Numerous tools for analysis are available under the VMD Main menu item Extensions - Analysis. In addition to these built-in tools, VMD users often use custom-written scripts to analyze desired properties of the simulated systems. VMD Tcl scripting capabilities are very extensive, and provide boundless opportunities for analysis. QwikMD provides the user with some of the most employed analysis tools, allowing also the analysis while performing live NAMD sections.      \n\n"
    set font2 "text"
    
    set text3 "The combination of NAMD and VMD creates what we like to call the computational microscope. Analysis of molecular dynamics trajectories is a very important step in any study that aims to understand molecular details of protein complexes. Connecting dynamics to structural data from diverse experimental sources, molecular dynamics simulations permit the exploration of biological phenomena in unparalleled detail. Advances in simulations are moving the atomic resolution descriptions of biological systems into the million-to-billion atom regime, in which numerous cell functions reside. To read more about the advances in Molecular Dynamics simulations to study large and complex system check the link at the bottom of this window.    \n\n"
    set font3 "text"


    set link {http://www.ks.uiuc.edu/Publications/Papers/paper.cgi?tbcode=PERI2015}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set title "Analysis with the Computational Microscope"
    return [list ${text} $link $title]
}











proc QWIKMD::mdSmdInfo {} {

    set text1 "Solvating the System \n\n"
    set font1 "title"
    
    set text2 "To perform MD simulations one has to mimic the environment of the \
    protein, or any other molecule of interest. The most common solvent is water and \
    there are two main ways to mimic the solvent effect. Either simulating all the \
    atoms of the solvent - explicit solvent model - or by adding dielectric constant \
    to the electrostatic calculation - implicit solvent model. Next you will find a \
    description of these models as well as a description on how to add salt to the \
    water solution in order to make a more realistic solvent model.     \n\n"
    set font2 "text"
    
    set text3 "Implicit Solvent    \n\n"
    set font3 "subtitle"
    
    set text4 "An implicit solvent model is a simulation technique that eliminates \
    the need for explicit water atoms by including many of the effects of solvent \
    in the inter-atomic force calculation. For example, polar solvent acts as a \
    dielectric and screens (lessens) electrostatic interactions. The elimination \
    of explicit water accelerates conformational explorations and increases \
    simulation speed at the cost of not modeling the solvent as accurately \
    as explicit models.    \n"
    set font4 "text"
    
    set text5 "But be careful, because implicit solvent models eliminate explicit \
    water molecules and represent water in an averaged manner, implicit solvent \
    models are considered less accurate than explicit solvent models. Always use \
    caution when employing implicit solvent for molecular dynamics research.    \n\n"
    set font5 "text"

    set text6 "Generalized Born Implicit Solvent    \n\n"
    set font6 "subtitle"
    
    set text7 "Generalized Born implicit solvent models are one particular class \
    of implicit solvent models. There are two parts to a GBIS calculation. First, \
    the Born radius of each atom is calculated. An atom's Born radius represents \
    the degree of exposure of an atom to solvent. Atoms on the surface of a \
    protein are highly exposed to solvent, their electrostatic interactions will \
    be highly screened and their Born radii will be small. Atoms buried in the \
    center of a protein will not be very exposed to solvent, their electrostatics \
    won't be screened much and their Born radii will be large. Second, \
    inter-atomic electrostatic forces are calculated based on atom separation as \
    well as the geometric mean of the two atoms' Born radii.    \n"
    set font7 "text"
    
    set text8 "QwikMD uses Generalized Born Implicit Solvent when the Implicit \
    Solvent option is selected. You can learn more about Generalized Born method \
    in the manuscript linked at the bottom of this window.     \n\n"
    set font8 "text"

    set text9 "Explicit Solvent    \n\n"
    set font9 "subtitle"
    
    set text10 "More realistic MD simulations are performed with explicit \
    representation of every atom of the solvent, usually a solution of water and \
    salt. The water box created by QwikMD is somewhat big for most studies. The \
    big water box was adopted as a safety measure. It is common to see large \
    conformational changes in proteins. These changes can make the water box too \
    small, which is hard to be observed by someone new in the field. Ideally, \
    one should work with a box, which is large enough that the protein does not \
    interact with its image in the next cell if periodic boundary conditions \
    are used. The use of periodic boundary conditions involves surrounding \
    the system under study with identical virtual unit cells. The atoms in the \
    surrounding virtual systems interact with atoms in the real system. These \
    modeling conditions are effective in eliminating surface interaction of \
    the water molecules and creating a more faithful representation of the in \
    vivo environment than a water sphere surrounded by vacuum provides.     \n\n"
    set font10 "text"
    
    set text11 "If the protein is being pulled - SMD protocol - the box should \
    be large enough that the water will still significantly immerse the protein \
    when it is fully extended.    \n\n"
    set font11 "text"
    
    set text12 "Many water molecules are available for MD simulations. NAMD \
    currently supports the 3-site TIP3P water model, the 4-site TIP4P water \
    model, and the 5-site SWM4-NDP water model (from the Drude force field). \
    As the standard water model for CHARMM, TIP3P is the model employed in \
    the simulations prepared with QwikMD.   \n\n"
    set font12 "text"
    
    set text13 "Creating a Salt Solution     \n\n"
    set font13 "subtitle"
    
    set text14 "Ions should be placed in the water to represent a more typical \
    biological environment. They are especially necessary if the protein being \
    studied carries an excess charge. In that case, the number of ions should \
    be chosen to make the system neutral. The ions present will shield the \
    regions of the protein, which carry the charge, and make the entire \
    system more stable. They should be placed in regions of potential minima, \
    since they will be forced to those regions during the simulation anyway. \
    The psf file contains the charge of each atom and may be used to determine \
    the charge of the total system or parts of it.    \n\n"
    set font14 "text"

    set text15 "One must set the desired salt concentration when preparing \
    the simulation with QwikMD. The default Salt Concentration is 0.15 mol/L. \
    Even if the Salt Concentration is set to ZERO, QwikMD will add ions to \
    neutralize the total charge of the system. Remember, in a MD simulation \
    with periodic boundary condition the total charge of the system should \
    be ZERO.     \n\n"
    set font15 "text"

        
    
    set link {http://www.ks.uiuc.edu/Publications/Papers/paper.cgi?tbcode=TANN2011A}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] [list $text7 $font7] [list $text8 $font8]  [list $text9 $font9] [list $text10 $font10] [list $text11 $font11] [list $text12 $font12] [list $text13 $font13] [list $text14 $font14] [list $text15 $font15] ]
    set title "Solvating the System"
    return [list ${text} $link $title]
}














proc QWIKMD::protocolMDInfo {} {

    set text1 "Molecular Dynamics Simulations \n\n"
    set font1 "title"
    
    set text2 "Molecular dynamics (MD) is a computer simulation of physical movements of atoms and molecules in the context of N-body simulation. The atoms and molecules are allowed to interact for a period of time, giving a view of the motion of the atoms. In the most common version, the trajectories of atoms and molecules are determined by numerically solving the Newtons equations of motion for a system of interacting particles, where forces between the particles and potential energy are defined by interatomic potentials or molecular mechanics force fields. The method was originally conceived within theoretical physics in the late 1950s but is applied today mostly in chemical physics, materials science and the modeling of biomolecules.    \n\n"
    set font2 "text"
    
    set text3 "Computer simulations of biomolecular systems have grown rapidly over the past few decades, passing from simulating very small proteins in vacuum to simulating large protein complexes in a solvated environment. All-atom MD simulations, employing classical mechanics, allowed the study of a broad range of biological systems, from small molecules such as anesthetics or small peptides, to very large protein complexes such as the ribosome or virus capsids. Hybrid classical/quantum MD simulations allowed the study of enzymatic activity or polarizable molecules in biological membranes. However, despite its success, MD simulations are still limited in two regards, inaccuracy of force fields and high computational cost. Such limitations can lead to inadequate sampling of conformational states, which in turn limits the ability to analyze and reveal functional properties of the systems being examined. All relevant states of a system must be reached in simulations in order for its dynamics and function to be meaningfully characterized. Molecular dynamics simulations have always been viewed as a general sampling method for the study of conformational changes of biomolecules. However, biological molecules are known to have rough energy landscapes, with many local minima frequently separated by high-energy barriers, making it easy to fall into a non-functional state that is hard to jump out of in most conventional simulations.     \n\n"
    set font3 "text"
    
    set text4 "As discussed, when running MD simulations it is very important to run more than one replica of the same system. Long trajectories usually also helps one to sample different conformations. Therefore a long simulation is important if big conformational changes are expected. If you want to learn more about sampling and molecular dynamics check the link at the bottom of this window.     \n\n"
    set font4 "text"
    
    set text5 "Minimization and Equilibration    \n\n"
    set font5 "subtitle"

    set text6 "QwikMD automatically runs 1000 steps of Energy Minimization and 1.0 ns of equilibration simulation, where the atoms backbone of the proteins are restrained in space.   \n\n"
    set font6 "text"
    
    set text7 "Be careful with the equilibration steps. It is common that MD minimization and equilibration simulations involve more than one minimization equilibration cycle, often fixing and releasing molecules in the system. For instance, one typically minimizes the system and then equilibrates with the atoms in the protein fixed in space, and then minimizes the system again and equilibrates again, this time with the protein free to move. Fixing the protein allows the water, which typically responds much faster to forces than the protein, to do the relaxing in the first step. This saves computational effort and prevents the introduction of artifacts from an unstable starting structure.    \n\n"
    set font7 "text"
    
    set text8 "Integration Step    \n\n"
    set font8 "subtitle"

    set text9 "The time step used in any MD simulation should be dictated by the fastest process (i.e. movement of atoms) taking place in the system. Among the various interactions, bond stretching and angle bending are the fastest, with typical bond stretching vibrations occurring on the order of once every 10-100 femtoseconds. Using a time step of 2 fs, which is close to the vibrational period (10 fs) of linear bonds involving hydrogen (fastest vibrations, since hydrogen has small mass), requires that these bonds be fixed, and only slower vibrations may be free to move, such as dihedral angle bending. For large molecules, these slower vibrations typically govern the behavior of the molecule more than the quicker ones, so bond fixing is somewhat acceptable, but should be avoided for accurate simulations. One prefers to use an MD timestep which is \u007E 1/10 of the fastest interactions in the simulation. For simulations with time step of 1 fs, one should use rigidBonds for water because water molecules have been parametrized as rigid molecules. QwikMD adopts 2 fs time step as standard.   \n\n"
    set font9 "text"
    
    set text10 "NPT Ensemble    \n\n"
    set font10 "subtitle"
    
    set text11 "In the isothermal-isobaric ensemble (NPT), number of atoms (N), pressure (P) and temperature (T) are conserved. To do control pressure and temperature a thermostat and a barostat are needed. The NPT ensemble corresponds most closely to laboratory conditions with a flask open to ambient temperature and pressure. Langevin dynamics is a means of controlling the kinetic energy of the system, and thus, controlling the system temperature and/or pressure. QwikMD uses standard NAMD protocols that employ Langevin dynamics.    \n\n"
    set font11 "text"
    
    
    set link {http://www.ks.uiuc.edu/Publications/Papers/paper.cgi?tbcode=BERN2015}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] [list $text7 $font7] [list $text8 $font8]  [list $text9 $font9] [list $text10 $font10] [list $text11 $font11]  ]
    set title "Molecular Dynamics Simulations"
    return [list ${text} $link $title]

}






proc QWIKMD::protocolSMDInfo {} {

    set text1 "Steered Molecular Dynamics Simulations \n\n"
    set font1 "title"
    
    set text2 "Steered molecular dynamics (SMD) simulations, or force probe simulations, apply forces to a protein in order to manipulate its structure by pulling it along desired degrees of freedom. These experiments can be used to reveal structural changes in a protein at the atomic level. SMD is often used to simulate events such as mechanical unfolding or stretching.   \n\n"
    set font2 "text"
    
    set text3 "There are two typical protocols of SMD: one in which pulling velocity is held constant and one in which applied force is constant. Typically, part of the studied system (e.g. an amino acid in a protein) is restrained by a harmonic potential. Forces are then applied to specific atoms at either a constant velocity or a constant force. QwikMD is set to perform constant velocity SMD.      \n\n"
    set font3 "text"
    
    set text4 "SMD has been successfully employed in a wide range of biological systems, from the investigation of protein mechanotransduction, to permeability of membrane channels and the characterization of protein-receptor interactions.     \n\n"
    set font4 "text"
    
    set text5 "Minimization and Equilibration    \n\n"
    set font5 "subtitle"

    set text6 "QwikMD automatically runs 1000 steps of Energy Minimization and 1.0 ns of equilibration simulation, where the atoms backbone of the proteins are restrained in space. A short MD simulation can be carried out before the actual SMD simulation start.   \n\n"
    set font6 "text"
    
    set text7 "Be careful with the equilibration steps. It is common that MD minimization and equilibration simulations involve more than one minimization equilibration cycle, often fixing and releasing molecules in the system. For instance, one typically minimizes the system and then equilibrates with the atoms in the protein fixed in space, and then minimizes the system again and equilibrates again, this time with the protein free to move. Fixing the protein allows the water, which typically responds much faster to forces than the protein, to do the relaxing in the first step. This saves computational effort and prevents the introduction of artifacts from an unstable starting structure.    \n\n"
    set font7 "text"
    
    set text8 "Integration Step    \n\n"
    set font8 "subtitle"

    set text9 "The time step used in any MD simulation should be dictated by the fastest process (i.e. movement of atoms) taking place in the system. Among the various interactions, bond stretching and angle bending are the fastest, with typical bond stretching vibrations occurring on the order of once every 10-100 femtoseconds. Using a time step of 2 fs, which is close to the vibrational period (10 fs) of linear bonds involving hydrogen (fastest vibrations, since hydrogen has small mass), requires that these bonds be fixed, and only slower vibrations may be free to move, such as dihedral angle bending. For large molecules, these slower vibrations typically govern the behavior of the molecule more than the quicker ones, so bond fixing is somewhat acceptable, but should be avoided for accurate simulations. One prefers to use an MD timestep which is ∼ 1/10 of the fastest interactions in the simulation. For simulations with time step of 1 fs, one should use rigidBonds for water because water molecules have been parametrized as rigid molecules. QwikMD adopts 2 fs time step as standard.   \n\n"
    set font9 "text"
    
    set text10 "NPT Ensemble    \n\n"
    set font10 "subtitle"
    
    set text11 "In the isothermal–isobaric ensemble (NPT), number of atoms (N), pressure (P) and temperature (T) are conserved. To do control pressure and temperature a thermostat and a barostat are needed. The NPT ensemble corresponds most closely to laboratory conditions with a flask open to ambient temperature and pressure. Langevin dynamics is a means of controlling the kinetic energy of the system, and thus, controlling the system temperature and/or pressure. QwikMD uses standard NAMD protocols that employ Langevin dynamics.    \n\n"
    set font11 "text"
    
    
    set link {http://www.ks.uiuc.edu/Publications/Papers/paper.cgi?tbcode=BERN2015}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] [list $text7 $font7] [list $text8 $font8]  [list $text9 $font9] [list $text10 $font10] [list $text11 $font11]  ]
    set title "Steered Molecular Dynamics Simulations"
    return [list ${text} $link $title]
}

proc QWIKMD::energiesPlotInfo {} {

    set text1 "Analyzing MD Simulation Energies      \n\n"
    set font1 "title"
    
    set text2 "QwikMD provides an easy-to-use interface to plot the energies reported in the NAMD log files during a simulation. Here the user can plot\
      the sum of all Potential energies (bonds, angles, dihedrals, impropers, electrostatics and VDW);\
      Kinetic energy (atoms energy of motion) and the Total energy (sum of the Potential and Kinetic energies).    \n\n"
    set font2 "text"
    
    set text3 "The same plotting procedure can be performed by another VMD plugin, namely \"NAMD plot\" under the menu Extensions - Analysis. For more information follow the link below.      \n\n"
    set font3 "text"
    
    
    set link {http://www.ks.uiuc.edu/Research/vmd/plugins/namdplot/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3]]
    set title "Analyze MD Simulation Energies"
    return [list ${text} $link $title]

}

proc QWIKMD::condPlotInfo {} {

    set text1 "Analyzing Temperature, Pressure and Volume      \n\n"
    set font1 "title"
    
    set text2 "To track the evaluation/equilibrium of the MD simulation is current practice to analyze the Temperature, Pressure and the Volume values. These values are\
    calculated by NAMD and reported in the log file and can be easily plot by QwikMD. As the default MD simulations performed in qwikmd are carried in the isothermal–isobaric ensemble (NPT),\
     the number of atoms (N), pressure (P) and temperature (T) are kept constant by employing thermostat and barostat, it is interesting to observe small variation of T and P values around the user defined values.     \n\n"
    set font2 "text"
    
    set text3 "For more information please follow the NAMD tutorial regarding Pressure and Temperature control sections in the link presented below.      \n\n"
    set font3 "text"
    
    
    set link {http://www.ks.uiuc.edu/Training/Tutorials/namd-index.html}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3]]
    set title "Analyzing Temperature, Pressure and Volume"
    return [list ${text} $link $title]

}




proc QWIKMD::smdPlotInfo {} {

    set text1 "Analyzing SMD simulations     \n\n"
    set font1 "title"
    
    set text2 "There are two typical protocols of SMD: one in which pulling velocity is held constant and one in which applied force is constant. Typically, part of the studied system (e.g. an atom in a protein) is restrained by a harmonic potential. Forces are then applied to specific atoms at either a constant velocity or a constant force.     \n\n"
    set font2 "text"
    
    set text3 "Here the user can plot the Force vs Distance and Force vs Time graphics. The Force profile observed here can be compared with single-molecule force spectroscopy (SMFS) experiments, which are usually performed with optical tweezers and/or atomic force microscopes (AFM).      \n\n"
    set font3 "text"

    set text4 "SMD combined with SMFS can be used to study several different biologically relevant systems. As an example, the combination of the two techniques allowed for the discovery of the strongest bimolecular interactions ever reported, when studying cellulosome proteins known as cohesin and dockerin. The findings demonstrated force activation and inter-domain stabilization of the cohesin-dockerin complex, and suggested that certain network components serve as mechanical effectors for maintaining network integrity. To read more about this work, which may help in the development of biocatalysts for production of biofuels, visit the link at the bottom of this window.       \n\n"
    set font4 "text"
    
    
    set link {http://www.ks.uiuc.edu/Highlights/?section=2015&highlight=2015-02}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] ]
    set title "Steered Molecular Dynamics - Force Profile Plots"
    return [list ${text} $link $title]

}






proc QWIKMD::vmdPluginsInfo {} {

    set text1 "VMD plugins     \n\n"
    set font1 "title"
    
    set text2 "VMD is more than a tool to show molecules. Several analysis plugins are already included in VMD and can be used for studying molecular dynamics trajectories. Here we have links to some of the most employed VMD plugins.     \n\n"
    set font2 "text"

    set text3 "In addition to Tcl and Python scripts, VMD implements - plugin - interfaces that provide a means for extending VMD at run-time without the necessity to recompile the program. The two primary types of plugins for VMD are 'molfile' plugins for reading and writing data files containing atomic, graphics, and volumetric data, and scripting extensions which implement new commands and user interfaces for performing tasks such as structure building, simulation setup, etc.     \n\n"
    set font3 "text"

    set text4 "Timeline     \n\n"
    set font4 "subtitle"

    set text5 "To analyze and identify events in molecular dynamics (MD) trajectories VMD’s Timeline plugin can be employed. Timeline creates an interactive 2D box-plot – time vs. structural component – that can show detailed structural events of an entire system over an entire MD trajectory. Events in the trajectory appear as patterns in the 2D plot. The plugin provides several built-in analysis methods, and the means to define new analysis methods. Timeline can read and write data sets, allowing external analysis and plotting with other software packages. Timeline includes features to help analysis of long trajectories and trajectories with large structures.     \n\n"
    set font5 "text"
    
    set text6 "In the main 2D box-plot graph, users identify events by looking for patterns of changing values of the analyzed parameter. The user can visually identify regions of interest – rapidly changing structure values, clusters of broken bonds, differences between stable and non-stable values, and similar. The user can explore the resulting structures by tracing the mouse cursor – scrubbing - over the identified areas. The structure is highlighted and the trajectory is moved in time to track the highlight.    \n\n"
    set font6 "text"
    
    set link {http://www.ks.uiuc.edu/Research/vmd/plugins/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] ]
    set title "VMD Plugins"
    return [list ${text} $link $title]

}











proc QWIKMD::selResiduesWindowinfo {} {

    set text1 "Residue Selection Window    \n\n"
    set font1 "title"
    
    set text2 "The residue selection window of QwikMD allows the user to make several changes in the biological system to be simulated. The changes must be made before preparing all the files necessary for MD simulations. Be very careful with residues/molecules with the entire line marked in RED, as they are unidentified molecules.     \n\n"
    set font2 "text"

    set text3 "Rename Problematic Residues/Molecules    \n\n"
    set font3 "subtitle"

    set text4 "WARNING: Residues/molecules with the entire line marked in RED must be renamed in order to be able to prepare simulation files.      \n\n"
    set font4 "text"

    set text5 "To rename a molecule/residue select the rename tool in the Table Mode and then click on the Res NAME of the molecule/residue that you want to rename. A list of possible substitutes will be available. Molecules that are not in the list are likely not available in the CHARMM force field and require a more advance parameterization step. To learn more about parameterizing a new molecule, click in the link at the bottom of this window.    \n\n"
    set font5 "text"
    
    set text6 "Mutate Amino Acid Residues and/or Nucleic Acid Bases.   \n\n"
    set font6 "subtitle"

    set text7 "QwikMD allows easy mutation of amino acid residues and nucleic acid bases. For that select the mutate tool in the Table Mode and then click on the Res NAME of the amino acid that you want to mutate.    \n\n"
    set font7 "text"

    set text8 "WARNING: Even a very small number of mutations may affect the structure of a protein drastically. Be very careful when using the mutation tool of QwikMD as you might create artifacts in your simulation.     \n\n"
    set font8 "text"

    set text9 "Change Residues Type   \n\n"
    set font9 "subtitle"

    set text10 "Sometimes, molecules categorization can be misleading (or even nonexistent), which hinders the correct identification of residues. For instance, if the user intends to mutate\
    the nucleotide Adenosine Triphosphate (ATP) by another nucleotide, it would be possible to mutate to a Guanine, Adenine, Cytosine or Thymine (or Uracil in RNA), as they\
 share the same category as nucleic residues (nucleic). To avoid such structural errors, QwikMD gives the possibility to change residues type (category),\
     so a logical choice can be made.    \n\n"
    set font10 "text"

    set text11 "Change Protonation State of Amino Acid Residues   \n\n"
    set font11 "subtitle"

    set text12 "Amino acids, depending on their environment, can present different protonation states. The user can easily change standard protonation states with QwikMD. It is recommended to check the protonation state of the amino acids before a MD simulation. Several tools can be used for that. One of the most popular tools is the PROPKA Server http://propka.org/.    \n\n"
    set font12 "text"

    set text13 "Histidine Residues. "
    set font13 "subtitle"

    set text14 "Of the 20 amino acids, histidine is the only one that ionizes within the physiological pH range ~7.4. This effect is characterized by the pKa of the amino acid side chain. For histidine, the value is 6.04. This leads to the possibility of different protonation states for histidine residues in a protein, and makes the consideration of the proper state important in MD simulations. The viable states are one in which the delta nitrogen of histidine is protonated - listed with residue name HSD in the topology file - one in which the epsilon nitrogen of histidine is protonated - HSE - and one in which both nitrogens are protonated - HSP. If not set by the user, QwikMD uses HSD as the standard histidine protonation state.    \n\n"
    set font14 "text"

    set text15 "Delete Molecules/Amino Acid Residues   \n\n"
    set font15 "subtitle"

    set text16 "It is usual to observe PDB structures presenting molecules that were used in the crystallization process or that are just not of interest of the study. It is also usual to simulate only part of a protein in a MD study. The delete option of QwikMD allows the user to easily remove molecules or a portion of a protein from the simulation step. QwikMD allows multiple selections with the delete option.   \n\n"
    set font16 "text"

    
    set link {http://www.ks.uiuc.edu/Research/vmd/plugins/fftk/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] [list $text7 $font7] [list $text8 $font8]  [list $text9 $font9] [list $text10 $font10] [list $text11 $font11] [list $text12 $font12] [list $text13 $font13] [list $text14 $font14] ]
    set title "Residue Selection Window"
    return [list ${text} $link $title]

}













proc QWIKMD::outputBrowserinfo {} {

    set text1 "QwikMD Output Files     \n\n"
    set font1 "title"
    
    set text2 "To run molecular dynamics simulations with NAMD at least four files are required: a Protein Data Bank (pdb) file a Protein Structure File (psf), a force field parameter file and a configuration file. During the preparation steps, where the system might be solvated, ionized, among other procedures, several files are created. QwikMD separates the files created in the preparation step in a SETUP folder, while files needed to run the MD simulations are in a RUN folder. These two folders are created inside the folder defined by the user in working directory window. With the same name as the folder created by the user, a file with .QwikMD extension allows the user to load simulations performed with QwikMD and also previously created preparation steps, like amino acid residues mutations or salt concentration.      \n\n"
    set font2 "text"
    
    set text3 "To run a simulation prepared with QwikMD in a computer cluster or supercomputer, one needs to copy only the RUN folder.      \n\n"
    set font3 "text"
    
    set text4 "To learn more about MD simulations check the list of reviews published by our group in the link at the bottom of this window.   \n\n"
    set font4 "text" 
    
    
    set link {http://www.ks.uiuc.edu/Publications/Papers/Review/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4]]
    set title "Load/Save Output Files"
    return [list ${text} $link $title]

}


proc QWIKMD::hbondInfo {} {
    set text1 "Hydrogen Bonds Counter \n\n"
    set font1 "title"
    
    set text2 "The Hydrogen bonds window shows the number of hydrogen bonds formed throughout a trajectory. A hydrogen bond is formed between an atom with a hydrogen bonded to it (the donor, D) and another atom (the acceptor, A) provided that the distance D-A is less than the cut-off distance (3.5 Angstroms) and the angle D-H-A is less than the cut-off angle (30 degrees).     \n\n"
    set font2 "text"
    
    set text3 "QwikMD allows the user to count the number of hydrogen bonds on the fly per each step. The hydrogen bonds can be calculated between solute (usually protein) and water molecules, or internal hydrogen bonds of the protein. To read more about VMD HBonds plugin, with many more options, check the link at the bottom of this window.   \n\n"
    set font3 "text"
    set link {http://www.ks.uiuc.edu/Research/vmd/plugins/hbonds/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set title "Hydrogen Bonds"
    return [list ${text} $link $title]

}


proc QWIKMD::MDControlsinfo {} {

    set text1 "Running NAMD    \n\n"
    set font1 "title"
    
    set text2 "NAMD is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems. NAMD scales to hundreds of processors on high-end parallel platforms, as well as tens of processors on low-cost commodity clusters, and also runs on individual desktop and laptop computers.    \n\n"
    set font2 "text"

    set text3 "QwikMD helps the user to prepare NAMD input files to run in a range of computers, from the largest supercomputers with high-end parallel platforms to the smallest laptop computers. QwikMD also allows, through the Interactive Molecular Dynamics interface of NAMD, to run live simulations where the user can look and analyze the trajectories while they are being created.     \n\n"
    set font3 "text"

    set text4 "Preparing simulations with QwikMD      \n\n"
    set font4 "subtitle"

    set text5 "When the Prepare button is pressed, scripts to perform all the steps required by the settings selected by user. Two folders will be created in the Working directory: files created in the preparation step in a SETUP folder, while files needed to run the MD simulations are in a RUN folder.     \n\n"
    set font5 "text"
    
    set text6 "NOTE: If you want to run Live Simulation, make sure you have the corresponding box checked before you click Prepare.   \n\n"
    set font6 "text"

    set text7 "Running NAMD Live    \n\n"
    set font7 "subtitle"

    set text8 "To run NAMD live just press Start after preparing the system for a live simulation.    \n\n"
    set font8 "text"

    set text9 "WARNING: Always click to FINISH your simulation before leaving VMD or closing QwikMD window. NAMD activities started from VMD will continue to run in background for all the steps requested unless you make sure the simulation was aborted before leaving QwikMD.    \n\n"
    set font9 "text"

    set text10 "Running NAMD in Background    \n\n"
    set font10 "subtitle"

    set text11 "To run NAMD live just press Start. Note that VMD window will freeze until the simulation is completed. \n\n"
    set font11 "text"

    set text12 "WARNING: Always click to FINISH your simulation before leaving VMD or closing QwikMD window. NAMD activities started from VMD will continue to run in background for all the steps requested unless you make sure the simulation was aborted before leaving QwikMD.    \n\n"
    set font12 "text"

    set text13 "Running NAMD in a Supercomputer     \n\n"
    set font13 "subtitle"

    set text14 "To run NAMD in a supercomputer copy the RUN folder to your folder in the supercomputer file system and run NAMD following the procedures for the specific computer.    \n\n"
    set font14 "text"


    
    set link {http://www.ks.uiuc.edu/Publications/Brochures/BPTL/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] [list $text6 $font6] [list $text7 $font7] [list $text8 $font8]  [list $text9 $font9] [list $text10 $font10] [list $text11 $font11] [list $text12 $font12] [list $text13 $font13] [list $text14 $font14] ]
    set title "Preparing and Running Simulations with NAMD"
    return [list ${text} $link $title]

}


proc QWIKMD::TopologiesInfo {} {
    set text1 "Topologies & Parameters    \n\n"
    set font1 "title"
    
    set text2 "A CHARMM forcefield topology file contains all of the information needed to convert a list of residue names into a complete PSF structure file. It also contains internal coordinates that allow the automatic assignment of coordinates to hydrogens and other atoms missing from a crystal PDB file.    \n\n"
    set font2 "text"

    set text3 "Structure Topologies Report.     \n\n"
    set font3 "subtitle"
    set textfont [list]
    foreach error [lindex $QWIKMD::topoerror 1] {
        lappend textfont [list $error "text"]
    }
    # set error [lindex $QWIKMD::topoerror 1]
    # set text4 "${error}"
    # set font4 "text"

    
    set link {http://mackerell.umaryland.edu/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set text [concat $text $textfont]
  
    set title "Topologies & Parameters Structure Check Report"
    return [list ${text} $link $title]
}

proc QWIKMD::ChiralityInfo {} {
    set text1 "Chiral Centers    \n\n"
    set font1 "title"
    
    set text2 "All amino acids but glycine have at least one chiral center at C\u03B1 (see Fig. 1).\
     Threonine and isoleucine have an additional chiral center at C\u03B2. According to\
      the D- / L- naming convention, naturally occurring amino acids are found in the L-configuration. Note, however, that D-amino acids do occur in biology, e. g.,\
       in cell walls of bacteria. Nucleic acids also have chiral centers. For example, in DNA the atoms C1', C3', and C4' are chiral, while RNA has an additional chiral center at C2'. Chirality is central to all molecular interactions in biological \
       systems. A simple experiment demonstrates the principle: try to shake someone’s left hand with your right.    \n\n"
    set font2 "text"

    set text3 "Chiral Centers Warnings Report.     \n\n"
    set font3 "subtitle"
    set textfont [list]
    #for {set i 0} {$i < $QWIKMD::chirerror} {incr i} {}
    foreach res [lindex $QWIKMD::chirerror 1] {
        #set res [chirality list $i -mol $QWIKMD::topMol]
        set center [lindex $res 0]
        lappend textfont [list "Chiral Center on Residue [lindex $center 2] [lindex $center 1], Chain [lindex $center 3], composed by the atoms [lindex $res 1]\n" "text"]
    } 
    if {$QWIKMD::chirerror == 0} {
        lappend textfont [list "No Chiral Centers issues found\n" "text"]
    } elseif {$QWIKMD::chirerror == ""} {
        lappend textfont [list "Error in Chiral Centers search\n" "text"]
    }
    
    # set error [lindex $QWIKMD::topoerror 1]
    # set text4 "${error}"
    # set font4 "text"

    
    set link {http://www.ks.uiuc.edu/Training/Tutorials/science/structurecheck/tutorial_structurecheck-html/node3.html}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set text [concat $text $textfont]
    
    set title "Chiral Centers Structure Check Report"
    return [list ${text} $link $title]
}


proc QWIKMD::CispeptideInfo {} {
    set text1 "Cis Peptide Bonds    \n\n"
    set font1 "title"
    
    set text2 "In naturally occurring proteins most peptide bonds are in the trans configuration. However, sometimes cis peptide bonds do occur. The vast majority of cis peptides is observed at a proline, Xaa-Pro, Xaa being any amino acid. But non-proline Xaa-non Pro cis bonds are also found in proteins, although they occur much less frequently than Xaa-Pro (see A. Jabs et al., JMB, 286, 291-304 (1999)).    \n\n"
    set font2 "text"

    set text3 "Cis Peptide Report.     \n\n"
    set font3 "subtitle"
    set textfont [list]
    #for {set i 0} {$i < [lindex $QWIKMD::cisperror 1]} {incr i} {}
    foreach res [lindex $QWIKMD::cisperror 1] {
        #set res [cispeptide list $i -mol $QWIKMD::topMol]
        set res1 [lindex $res 0]
        set res2 [lindex $res 1]
        lappend textfont [list "Cis Peptide bond found between Residue [lindex $res1 2] [lindex $res1 1], Chain [lindex $res1 3] and Residue  [lindex $res2 2] [lindex $res2 1], Chain [lindex $res2 3]\n" "text"]
    } 
    if {$QWIKMD::cisperror == 0} {
        lappend textfont [list "No Cis Peptide Bond found\n" "text"]
    }
    
    # set error [lindex $QWIKMD::topoerror 1]
    # set text4 "${error}"
    # set font4 "text"

    
    set link {http://www.ks.uiuc.edu/Training/Tutorials/science/structurecheck/tutorial_structurecheck-html/node4.html}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set text [concat $text $textfont]
   
    set title "Cis Peptide Bond Structure Check Report"
    return [list ${text} $link $title]
}


proc QWIKMD::TorsionOutlierInfo {} {
    set text1 "Torsion Angles Outliers    \n\n"
    set font1 "title"
    
    set text2 "In naturally occurring proteins most peptide bonds are in the trans configuration. However, sometimes cis peptide bonds do occur. The vast majority of cis peptides is observed at a proline, Xaa-Pro, Xaa being any amino acid. But non-proline Xaa-non Pro cis bonds are also found in proteins, although they occur much less frequently than Xaa-Pro (see A. Jabs et al., JMB, 286, 291-304 (1999)).    \n\n"
    set font2 "text"

    set text3 "Torsion Angles Outliers.     \n\n"
    set font3 "subtitle"
    set textfont [list]
    set do 0
    foreach outlier $QWIKMD::torsionOutlier {
        if {[llength [lrange $outlier 1 end]] > 0 } {
            lappend textfont [list "[lindex $outlier 0]\n" "subtitle"]
            set i 1
            foreach res [lrange $outlier 1 end] {
                set str "Chain [lindex $res 0]; "
                if {[llength $res] == 3} {
                    append str "Resid [lindex $res 1]; " 
                } else {
                    append str "Segment [lindex $res 1]; Resid [lindex $res 2]; "
                }
                append str "Score (per residue) [format %.3f [lindex $res end]]\n"
                lappend textfont [list $str "text"]
                incr i
            }
            lappend textfont [list "\n\n" "text"]
            #incr numoutlier [llength [lrange $outlier 1 end]]
            set do 1
        } elseif {$outlier == "Failed"} {
            lappend textfont "Error in the torsion angle search"
        }
        
    }
    if {$do == 0} {
        lappend textfont [list "No torsion angle outlier found\n" "text"]
    }

    
    set link {http://www.ks.uiuc.edu/Training/Tutorials/science/structurecheck/tutorial_structurecheck-html/node4.html}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set text [concat $text $textfont]
   
    set title "Torsion Angles Outliers Structure Check Report"
    return [list ${text} $link $title]
}

proc QWIKMD::TorsionMarginalInfo {} {
    set text1 "Marginal Angles Outliers    \n\n"
    set font1 "title"
    
    set text2 "In naturally occurring proteins most peptide bonds are in the trans configuration. However, sometimes cis peptide bonds do occur. The vast majority of cis peptides is observed at a proline, Xaa-Pro, Xaa being any amino acid. But non-proline Xaa-non Pro cis bonds are also found in proteins, although they occur much less frequently than Xaa-Pro (see A. Jabs et al., JMB, 286, 291-304 (1999)).    \n\n"
    set font2 "text"

    set text3 "Marginal Angles Outliers.     \n\n"
    set font3 "subtitle"
    set textfont [list]
    set do 0
    foreach marginal $QWIKMD::torsionMarginal {
        if {[llength [lrange $marginal 1 end]] > 0 } {
            lappend textfont [list "[lindex $marginal 0]\n" "subtitle"]
            set i 1
            foreach res [lrange $marginal 1 end] {
                set str "Chain [lindex $res 0]; "
                if {[llength $res] == 3} {
                    append str "Resid [lindex $res 1]; " 
                } else {
                    append str "Segment [lindex $res 1]; Resid [lindex $res 2]; "
                }
                append str "Score (per residue) [format %.3f [lindex $res end]]\n"
                lappend textfont [list $str "text"]
                incr i
            }
            lappend textfont [list "\n\n" "text"]
            #incr numoutlier [llength [lrange $outlier 1 end]]
            set do 1
        } elseif {$marginal == "Failed"} {
            lappend textfont "Error in the torsion angle search"
        }
        
    }
    if {$do == 0} {
        lappend textfont [list "No torsion angle marginal found\n" "text"]
    }

    
    set link {http://www.ks.uiuc.edu/Training/Tutorials/science/structurecheck/tutorial_structurecheck-html/node4.html}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set text [concat $text $textfont]
   
    set title "Torsion Angles Marginals Structure Check Report"
    return [list ${text} $link $title]
}

proc QWIKMD::GapsInfo {} {
    set text1 "Protein Sequence Gaps    \n\n"
    set font1 "title"
    
    set text2 "One of the greatest challenges in studying very large multi-protein complexes is \
    obtaining complete structural models of complexes as a whole, namely without\
    segments missing. Usually structures of isolated subunits are determined using,\
    for example, X-ray crystallography or NMR, and are then reassembled into\
    the multi-protein complex by using data from cryo-EM experiments. For example, \
    the molecular dynamics flexible fitting (MDFF) technique assists in the \
    re-assembly process by employing MD to perform the fitting of the initial subunit structures to the cryo-EM density, permitting flexibility, yet maintaining a \
    realistic conformation. The successful structure modeling by MDFF depends on the quality of the initial model taken from X-ray crystallography, NMR spectroscopy, homology modeling, or other sources. The main sources for initial models are X-ray structures, which leads to\
      the following issues. Electron densities of highly flexible protein regions are often very sparse or not captured at all, resulting in poor structural models or missing coordinates of these regions in deposited PDB entries. Moreover, in many cases N- and C-terminal \
       tails are truncated from protein constructs to increase their crystallization probabilities. However, these truncated and structurally unresolved regions are often essential for protein function, i.e. flexible loops acting as catalytic and binding sites,terminal\
         tails in protein-protein interaction networks and as assembly anchorsin multi-protein complexes.    \n\n"
    set font2 "text"

    set text3 "Protein Sequence Gaps Report.     \n\n"
    set font3 "subtitle"
    set textfont [list]
    
    foreach error $QWIKMD::gaps {
        lappend textfont [list "Found a gap in chain [lindex $error 0] between resid [lindex [lindex $error 1] 0] and [lindex [lindex $error 1] 1]\n" "text"]
    }
    if {[llength $QWIKMD::gaps] == 0} {
        lappend textfont [list "No Protein Sequence Gaps found\n" "text"]
    }
    # set error [lindex $QWIKMD::topoerror 1]
    # set text4 "${error}"
    # set font4 "text"

    
    set link {http://www.ks.uiuc.edu/Training/Tutorials/science/rosetta-mdff/rosetta-mdff-tutorial-html/}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] ]
    set text [concat $text $textfont]
    set title "Protein Sequence Gaps Structure Check Report"
    return [list ${text} $link $title]
}

proc QWIKMD::renderInfo {} {

    set text1 "Image Render \n\n"
    set font1 "title"
    

    set link {http://www.ks.uiuc.edu/Research/vmd/minitutorials/tachyonao/}
    set text [list [list $text1 $font1]]
    set title "Image Render"
    return [list ${text} $link $title]

}

proc QWIKMD::advAnalysisInfo {} {

    set text1 "Advanced Analysis \n\n"
    set font1 "title"
    
        

    set link {http://www.ks.uiuc.edu/Training/Tutorials/vmd/tutorial-html/node7.html}
    set text [list [list $text1 $font1]]
    set title "Advanced Analysis"
    return [list ${text} $link $title]

}

proc QWIKMD::topparInfo {} {
    global env
    set text1 "Add standard residues Topologies and Parameters \n\n"
    set font1 "title"
    
    set text2 "QwikMD only lists some of the residues present in the CHARMM 36 topology files by default. If the user \
    intends to add more standard residues to QwikMD list, such as, proteins, nucleotides, carbohydrates (glycan), lipids and other standard residues,\
     one must ONLY upload the topology file correspondent to the residue type and select the residue from the table.      \n\n"
    set font2 "text"


    set text3 "Add non-standard residues Topologies and Parameters\n\n"
    set font3 "title"
    
    set text4 "To add non-standard residues, such as residues parameterized by the user, one must upload a STREAM (.str) file containing both \
    the topology and the parameters.       \n\n"
    set font4 "text"

    set text5 "List of default parameters files\n\n"
    set font5 "title"
    
    set list ""
    foreach var [glob $env(CHARMMPARDIR)/*36*.prm] {
        append list "[file tail $var]; " 
    }
    set list [string trimright $list "; "]
    set text6 "$list       \n\n"
    set font6 "text"

    set text7 "List of default stream files\n\n"
    set font7 "title"
    
    set list ""
    foreach var [glob $env(CHARMMPARDIR)/*.str] {
        append list "[file tail $var]; " 
    }
    set list [string trimright $list "; "]
    set text8 "$list       \n\n"
    set font8 "text"
    set link {http://mackerell.umaryland.edu/charmm_ff.shtml}
    set text [list [list $text1 $font1] [list $text2 $font2] [list $text3 $font3] [list $text4 $font4] [list $text5 $font5] \
    [list $text6 $font6] [list $text7 $font7] [list $text8 $font8]]
    set title "Add Topologies and Parameters"
    return [list ${text} $link $title]

}
