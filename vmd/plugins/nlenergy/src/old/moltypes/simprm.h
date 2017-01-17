/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/simprm.h
 * @brief   Simulation parameter data container.
 * @author  David J. Hardy
 * @date    Apr. 2008
 */

#ifndef MOLTYPES_SIMPRM_H
#define MOLTYPES_SIMPRM_H

#include "nlbase/nlbase.h"

#ifdef __cplusplus
extern "C" {
#endif


  typedef struct SimPrm_t {
    Array  parselist;
    Arrmap parsetable;

    /* input files */
    String coordinates;
    String structure;
    Strarray parameters;
    boolean paraTypeXplor;
    boolean paraTypeCharmm;
    String velocities;
    String binvelocities;
    String bincoordinates;
    String cwd;

    /* output files */
    String outputname;
    boolean binaryoutput;
    String restartname;
    int32 restartfreq;
    boolean restartsave;
    boolean binaryrestart;
    String dcdfile;
    int32 dcdfreq;
    boolean dcdUnitCell;
    String veldcdfile;
    int32 veldcdfreq;

    /* standard output */
    int32 outputEnergies;
    boolean mergeCrossterms;
    int32 outputMomenta;
    int32 outputPressure;
    int32 outputTiming;

    /* timestep parameters */
    int32 numsteps;
    dreal timestep;
    int32 firsttimestep;
    int32 stepspercycle;

    /* simulation space partitioning */
    dreal cutoff;
    boolean switching;
    dreal switchdist;
    dreal limitdist;
    dreal pairlistdist;
    // enum splitPatch;
    dreal hgroupCutoff;
    dreal margin;
    // int32 pairlistMinProcs;
    // int32 pairlistsPerCycle;
    // int32 outputPairlists;
    // dreal pairlistShrink;
    // int32 pairlistGrow;
    // int32 pairlistTrigger;

    /* basic dynamics */
    int32 exclude;  /* value = "none", "1-2", "1-3", "1-4", "scaled1-4" */
    dreal temperature;
    boolean comMotion;
    boolean zeroMomentum;
    dreal dielectric;
    dreal nonbondedScaling;
    dreal scaling14;  /* keyword = "1-4scaling" */
    boolean vdwGeometricSigma;
    int32 seed;
    int32 rigidBonds;  /* value = "none", "water", "all */
    dreal rigidTolerance;
    int32 rigidIterations;
    boolean rigidDieOnError;
    boolean useSettle;

    /* DPMTA parameters */
    boolean fma;
    int32 fmaLevels;
    int32 fmaMp;
    boolean fmaFFT;
    dreal fmaTheta;
    int32 fmaFFTBlock;

    /* PME parameters */
    boolean pme;
    dreal pmeTolerance;
    int32 pmeInterpOrder;
    dreal pmeGridSpacing;
    int32 pmeGridSizeX;
    int32 pmeGridSizeY;
    int32 pmeGridSizeZ;
    // int32 pmeProcessors;
    // boolean fftwEstimate;
    // boolean fftwUseWisdom;
    // String fftwWisdomFile;
    // boolean useDPME;

    /* full direct parameters */
    boolean fulldirect;

    /* multiple timestep parameters */
    int32 fullElectFrequency;
    int32 nonbondedFreq;
    int32 mtsAlgorithm;  /* value = "impulse"/"verletI", "constant"/"naive" */
    int32 longSplitting;  /* value = "xplor", "c1" */
    boolean molly;
    dreal mollyTolerance;
    int32 mollyIterations;

    /* harmonic constraint parameters */
    boolean constraints;
    int32 consexp;
    String consref;
    String conskfile;
    int32 conskcol;  /* value = 'X', 'Y', 'Z', 'O', 'B' */
    dreal constraintScaling;
    boolean selectConstaints;
    boolean selectConstrX;
    boolean selectConstrY;
    boolean selectConstrZ;

    /* fixed atom parameters */
    boolean fixedAtoms;
    boolean fixedAtomsForces;
    String fixedAtomsFile;
    int32 fixedAtomsCol;  /* value = 'X', 'Y', 'Z', 'O', 'B' */

    /* conjugate gradient parameters */
    boolean minimization;
    dreal minTinyStep;
    dreal minBabyStep;
    dreal minLineGoal;

    /* velocity quenching parameters */
    boolean velocityQuenching;
    dreal maximumMove;

    /* Langevin dynamics parameters */
    boolean langevin;
    dreal langevinTemp;
    dreal langevinDamping;
    boolean langevinHydrogen;
    String langevinFile;
    int32 langevinCol;  /* value = 'X', 'Y', 'Z', 'O', 'B' */

    /* temperature coupling parameters */
    boolean tCouple;
    dreal tCoupleTemp;
    String tCoupleFile;
    int32 tCoupleCol;  /* value = 'X', 'Y', 'Z', 'O', 'B' */

    /* temperature rescaling parameters */
    int32 rescaleFreq;
    dreal rescaleTemp;

    /* temperature reassignment parameters */
    int32 reassignFreq;
    dreal reassignTemp;
    dreal reassignIncr;
    dreal reassignHold;

    /* spherical harmonic boundary conditions */
    boolean sphericalBC;
    dvec sphericalBCcenter;
    dreal sphericalBCr1;
    dreal sphericalBCk1;
    int32 sphericalBCexp1;
    dreal sphericalBCr2;
    dreal sphericalBCk2;
    int32 sphericalBCexp2;

    /* cylindrical harmonic boundary conditions */
    boolean cylindricalBC;
    dvec cylindricalBCcenter;
    int32 cylindricalBCaxis;  /* value = 'x', 'y', 'z' */
    dreal cylindricalBCr1;
    dreal cylindricalBCl1;
    dreal cylindricalBCk1;
    int32 cylindricalBCexp1;
    dreal cylindricalBCr2;
    dreal cylindricalBCl2;
    dreal cylindricalBCk2;
    int32 cylindricalBCexp2;

    /* periodic boundary conditions */
    dvec cellBasisVector1;
    dvec cellBasisVector2;
    dvec cellBasisVector3;
    dvec cellOrigin;
    String extendedSystem;
    String xstfile;
    int32 xstfreq;
    boolean wrapWater;
    boolean wrapAll;
    boolean wrapNearest;

  } SimPrm;


  int SimPrm_init(SimPrm *);
  void SimPrm_done(SimPrm *);

  int SimPrm_set(SimPrm *, const char *keyword, const char *value);


#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_SIMPRM_H */
