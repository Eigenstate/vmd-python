#
# $Id: qwikmd_ballon.tcl,v 1.6 2016/10/13 17:14:37 johns Exp $
#
#==============================================================================

proc QWIKMD::pdbBrowserBL {} {

    set text "Browse folders for input structure file"
    
    return $text
}

proc QWIKMD::pdbLoadBL {} {

    set text "Load input structure file"
    
    return $text
}

proc QWIKMD::pdbentryLoadBL {} {

    set text "Type PDB ID for download
or the path to the file on disk"
    
    return $text
}

proc QWIKMD::nmrBL {} {

    set text "Add or remove the NMR states to the selection
window by checking/unchecking the number"
    
    return $text
}

proc QWIKMD::addChainBL {} {

    set text "Add or remove the chain to the selection
window by checking/unchecking the
letter for chain identifier"
    
    return $text
}

proc QWIKMD::selResidBL {} {

    set text "Pops up a new window to select and
manipulate - rename, delete, mutate,
protonate - each residue individually."
    
    return $text
}

proc QWIKMD::outputBrowserBL {} {

    set text "Browse folders to specify a work 
directory for the created configuration
files for the simulation by QwikMD"
    
    return $text
}

proc QWIKMD::cbckgBlack {} {

    set text "Change VMD window background to black color"
    
    return $text
}

proc QWIKMD::cbckgWhite {} {

    set text "Change VMD window background to white color"
    
    return $text
}

proc QWIKMD::cbckgGradient {} {

    set text "Change VMD window background to a gradient"
    
    return $text
}

proc QWIKMD::prepareBL {} {

    set text "Prepares the structures for 
simulation and creates and the
configuration files for the simulation"
    
    return $text
}

proc QWIKMD::liveSimulBL {} {

    set text "Live Simulation"
    
    return $text
}

proc QWIKMD::resetBL {} {

    set text "Resets everything!
Your molecule will be deleted to
start over from the beginning"
    
    return $text
}

proc QWIKMD::runbuttBL {} {

    set text "Run MD Simulation"
    
    return $text
}

proc QWIKMD::detachBL {} {

    set text "Stops live streaming of the simulation
by VMD but keep running the simulation
by NAMD without showing"
    
    return $text
}

proc QWIKMD::pauseBL {} {

    set text "Pause live streaming by VMD and
running of the simulation by NAMD.
Click again to continue the simulation."
    
    return $text
}

proc QWIKMD::finishBL {} {

    set text "Finishes the simulation"
    
    return $text
}

proc QWIKMD::solventBL {} {

    set text "Select between explicit and implicit solvent"
    
    return $text
}

proc QWIKMD::minimalBox {} {

    set text "Minimize Solvent Box Volume and place solvent buffer surrounding the structure.
Not recommended."
    
    return $text
}

proc QWIKMD::saltConceBL {} {

    set text "Select the salt concentration"
    
    return $text
}

proc QWIKMD::saltTypeBL {} {

    set text "Select the ions of the salt"
    
    return $text
}

proc QWIKMD::EquiMDBL {} {

    set text "Runs a sterical optimization by
minimizing the energy followed by
an MD simulation with constraints
on the protein atoms to
equilibrate the solvent"
    
    return $text
}
proc QWIKMD::mdMDBL {} {

    set text "MD simulation without constraints"
    
    return $text
}

proc QWIKMD::mdTemperatureBL {} {

    set text "The target temperature of the simulation"
    
    return $text
}

proc QWIKMD::mdMaxTimeBL {} {

    set text "The simulation finishes automatically
after the maximum simulation time is
reached if the Finish button
is not pressed before"
    
    return $text
}

proc QWIKMD::smdMaxLengthBL {} {

    set text "Total SMD simulation pulling distance"
    
    return $text
}

proc QWIKMD::smdVelocityBL {} {

    set text "SMD simulation pulling velocity"
    
    return $text
}

proc QWIKMD::smdAnchorBL {} {

    set text "open the \"Select Resid\" window and
select the anchor Residues for SMD"
    
    return $text
}

proc QWIKMD::smdPullingBL {} {

    set text "open the \"Select Resid\" window and
select the pulling Residues for SMD"
    
    return $text
}

proc QWIKMD::smdEqMDBL {} {

    set text "SMD no constraints simulation"
    
    return $text
}

proc QWIKMD::smdSMDBL {} {

    set text "SMD pulling simulation"
    
    return $text
}

proc QWIKMD::selTabelChainBL {} {

    set text "Chain identifier character"
    
    return $text
}

proc QWIKMD::selTabelResidBL {} {

    set text "Chain residues ID range"
    
    return $text
}

proc QWIKMD::selTabelTypeBL {} {

    set text "Chain residues Type"
    
    return $text
}

proc QWIKMD::selTabelRepBL {} {

    set text "Chain Representation Mode"
    
    return $text
}

proc QWIKMD::selTabelColorBL {} {

    set text "Chain Representation Color"
    
    return $text
}

proc QWIKMD::selTabelProtocol {} {

    set text "Simulation Protocol"
    
    return $text
}

proc QWIKMD::selTabelNSteps {} {

    set text "Number of Simulation Steps"
    
    return $text
}

proc QWIKMD::selTabelRestraints {} {

    set text "Atom Selection Restrained (VMD syntax)"
    
    return $text
}

proc QWIKMD::selTabelEnsemble {} {

    set text "Conservation of amount of substance (N), volume (V), temperature (T) or pressure (P)"
    
    return $text
}

proc QWIKMD::selTabelPressure {} {

    set text "The target pressure of the simulation"
    
    return $text
}

proc QWIKMD::selProtocolUnlock {} {

    set text "Allow or block selected protocol for edition"
    
    return $text
}

proc QWIKMD::selProtocolEdit {} {

    set text "Edit the protocol configuration text file"
    
    return $text
}

proc QWIKMD::selProtocolAdd {} {

    set text "Add new or replicate selected protocol"
    
    return $text
}

proc QWIKMD::selProtocolDelete {} {

    set text "Delete selected protocol"
    
    return $text
}

proc QWIKMD::rmsdSelection {} {

    set text "Calculate RMSD for the selection:
Backbone - protein/nucleic backbone atoms
Alpha Carbon - atom's name CA
No Hydrogen - everything but hydrogens
All - all atoms"
    
    return $text
}


# proc QWIKMD::rmsdAlphaCBL {} {

#   set text {Selection General Selection and name CA}
    
    
#   return $text
# }

# proc QWIKMD::rmsdNoHydrogenCBL {} {
#   set text {Selection General Selection and noh}
#   return $text
# }

proc QWIKMD::rmsdGeneralSelectionBL {} {
    set text "Calculate RMSD for the atom selection (VMD syntax)"
    return $text
}

proc QWIKMD::rmsdAlignBL {} {
    set text "Align individual frames to the initial structure"
    return $text
}

proc QWIKMD::rmsdAlignSelection {} {

    set text "Align frames with the selection:
Backbone - protein/nucleic backbone atoms
Alpha Carbon - atom's name CA
No Hydrogen - everything but hydrogens
All - all atoms"
    
    return $text
}

proc QWIKMD::rmsdGeneralAlignSelectionBL {} {
    set text "Align frames using the atom selection (VMD syntax)"
    return $text
}

proc QWIKMD::rmsdCalcBL {} {
    set text "RMSD Calculate"
    return $text
}

proc QWIKMD::hbWithinSolute {} {
    set text "Hbonds Within Solute"
    return $text
}

proc QWIKMD::hbSoluteSolvent {} {
    set text "Between Solute-Solvent"
    return $text
}

proc QWIKMD::energyTotal {} {
    set text "Plot the sum of Potential and Kinetic energies"
    return $text
}
proc QWIKMD::energyKinetic {} {
    set text "Plot Kinetic energy"
    return $text
}
proc QWIKMD::energyPotential {} {
    set text "Plot Potential energy"
    return $text
}

proc QWIKMD::energyBond {} {
    set text "Plot Bonds energy"
    return $text
}

proc QWIKMD::energyAngle {} {
    set text "Plot Angles energy"
    return $text
}

proc QWIKMD::energyDihedral {} {
    set text "Plot Dihedral Angles energy"
    return $text
}

proc QWIKMD::energyVDW {} {
    set text "Plot VDW energy"
    return $text
}

proc QWIKMD::enerCalcBL {} {
    set text "Calculate Energies"
    return $text
}

proc QWIKMD::condTemp {} {
    set text "Plot MD Temperature"
    return $text
}
proc QWIKMD::condPress {} {
    set text "Plot MD Pressure"
    return $text
}
proc QWIKMD::condVolume {} {
    set text "Plot MD Volume"
    return $text
}

proc QWIKMD::condCalcBL {} {
    set text "Calculate Thermodynamics Properties"
    return $text
}

proc QWIKMD::renderTypeBL {} {
    set text "Render VMD scene as"
    return $text
}

proc QWIKMD::renderResBL {} {
    set text "Image render resolution"
    return $text
}

proc QWIKMD::renderRendBL {} {

    set text "Type of Render
Capture Display - Snapshot
Background Render - Tachyon render running in background
Interactive Render - GPU Tachyon interactive render"
    
    return $text
}

proc QWIKMD::renderBL {} {
    set text "Render VMD scene"
    return $text
}

proc QWIKMD::advcComboAnBL {} {
    set text "Select Calculation Function"
    return $text
}

proc QWIKMD::hbondsSelWithinBL {} {
    set text "Calculate the number of hydrogen bonds within the solute 
(all but water and ion molecules)"
    return $text
}

proc QWIKMD::hbondsSelintraBL {} {
    set text "Calculate the number of hydrogen bonds between 
the solute (all but water and ion molecules) and 
the solvent (water molecules)"
    return $text
}

proc QWIKMD::hbondsSelBetwSelBL {} {
    set text "Calculate the number of hydrogen bonds between the two
atom selections below"
    return $text
}

proc QWIKMD::hbondsSelEntry1BL {} {
    set text "Atom selection in VMD syntax. Both atom selections
are considered as hydrogen bonds donor and acceptor"
    return $text
}

proc QWIKMD::hbondsSelEntry2BL {} {
    set text "Atom selection in VMD syntax. Both atom selections
are considered as hydrogen bonds donor and acceptor"
    return $text
}

proc QWIKMD::smdForceTimeBL {} {
    set text "Force vs Time"
    return $text
}

proc QWIKMD::smdForceDistanceBL {} {
    set text "Force vs Distance"
    return $text
}

proc QWIKMD::rmsfGeneralSelectionBL {} {
    set text "Calculate RMSF for the atom selection (VMD syntax)"
    return $text
}

proc QWIKMD::rmsfAlignBL {} {
    set text "Align individual frames to the initial structure"
    return $text
}

proc QWIKMD::rmsfAlignSelection {} {

    set text "Align frames with the selection:
Backbone - protein/nucleic backbone atoms
Alpha Carbon - atom's name CA
No Hydrogen - everything but hydrogens
All - all atoms"
    
    return $text
}

proc QWIKMD::rmsfGeneralAlignSelectionBL {} {
    set text "Align frames using the atom selection (VMD syntax)"
    return $text
}

proc QWIKMD::rmsfInitFrameBL {} {
    set text "Frame range lower bound"
    return $text
}

proc QWIKMD::rmsfFinalFrameBL {} {
    set text "Frame range upper bound"
    return $text
}

proc QWIKMD::rmsfSkipFrameBL {} {
    set text "Skip every N frames in the selected frame range"
    return $text
}

proc QWIKMD::rmsfRepBL {} {
    set text "Structure representation mode when presenting RMSF values"
    return $text
}

proc QWIKMD::sasaSel1BL {} {
    set text "Calculate SASA for the atom selection (VMD syntax)"
    return $text
}

proc QWIKMD::sasaSel1ContactBL {} {
    set text "First contacting surface (VMD syntax)"
    return $text
}

proc QWIKMD::sasaSel2BL {} {
    set text "Present the values for the atom selection (VMD syntax)"
    return $text
}

proc QWIKMD::sasaSel2ContactBL {} {
    set text "Second contacting surface (VMD syntax)"
    return $text
}

proc QWIKMD::sasaRepBL {} {
    set text "Strucutre representation mode when presenting SASA values"
    return $text
}

proc QWIKMD::sasaTblSASABL {} {
    set text "Average Solvent Accessible Surface Area (A\u00b2)"
    return $text
}

proc QWIKMD::sasaTblSTDVBL {} {
    set text "Standard Deviation of Solvent Accessible Surface Area Values"
    return $text
}

proc QWIKMD::selectTbSelectBL {} {
    set text "Select trajectory to analyze"
    return $text
}

proc QWIKMD::selectTbNameBL {} {
    set text "Protocol name producing the trajectory"
    return $text
}

proc QWIKMD::selectTbTauBL {} {
    set text "Tau values"
    return $text
}

proc QWIKMD::spcfHeatTempBL {} {
    set text "Simulation Temperature"
    return $text
}

proc QWIKMD::spcfHeatBKBL {} {
    set text "Boltzmann constant"
    return $text
}

proc QWIKMD::spcfHeatSelBL {} {
    set text "Atoms selection for calculation (VMD Syntax)"
    return $text
}

proc QWIKMD::spcfHeatResKcalBL {} {
    set text "Calculated Specific Heat value in kcal/mol*K"
    return $text
}

proc QWIKMD::spcfHeatResJoulBL {} {
    set text "Calculated Specific Heat value in J/kg*C"
    return $text
}

proc QWIKMD::spcfTempDistEqBL {} {
    set text "Normalized values curve fitting equation"
    return $text
}

proc QWIKMD::mbDistSelBL {} {
    set text "Atoms selection for calculation (VMD Syntax)"
    return $text
}

proc QWIKMD::mbDistEqBL {} {
    set text "Normalized values curve fitting equation"
    return $text
}

proc QWIKMD::tempQAtcrrTimeBL {} {
    set text "Autocorrelation decay time"
    return $text
}

proc QWIKMD::tempQInitTempBL {} {
    set text "Initial Temperature"
    return $text
}

proc QWIKMD::tempQTempDepthBL {} {
    set text "Temperature echo depth"
    return $text
}

proc QWIKMD::tempQTempTimeBL {} {
    set text "Temperature echo time"
    return $text
}

proc QWIKMD::tempQTempEqBL {} {
    set text "Values curve fitting equation"
    return $text
}

# proc QWIKMD::smdCalcBL {} {
#   set text {SMD Calculate}
#   return $text
# }

# proc QWIKMD::timeLineBL {} {
#   set text {Call TimeLine}
#   return $text
# }

proc QWIKMD::ResidselTabelResnameBL {} {
    set text "Residues name"
    return $text
}
proc QWIKMD::ResidselTabelResidBL {} {

    set text "Residues ID number"
    
    return $text
}

proc QWIKMD::ResidselTabelChainBL {} {

    set text "Residues chain identifier character"
    
    return $text
}

proc QWIKMD::ResidselTabelTypeBL {} {

    set text "Residues Type"
    
    return $text
}

proc QWIKMD::TableMutate {} {

    set text "Mutate Selected Residue"
    
    return $text
}

proc QWIKMD::TableProtonate {} {

    set text "Change Selected Residue Protonation State"
    
    return $text
}

proc QWIKMD::TableAdd {} {

    set text "Include Selected Previous Deleted Residues"
    
    return $text
}

proc QWIKMD::TableDelete {} {

    set text "Delete Selected Residues"
    
    return $text
}

proc QWIKMD::TableRename {} {

    set text "Rename Selected Residues"
    
    return $text
}

proc QWIKMD::TableInspection {} {

    set text "Represent Selected Residues"
    
    return $text
}

proc QWIKMD::TableType {} {

    set text "Change Selected Residue type"
    
    return $text
}

proc QWIKMD::TableApply {} {

    set text "Add/Delete Selected Residues"
    
    return $text
}

proc QWIKMD::TableClear {} {

    set text "Clear Selected Residues"
    
    return $text
}

proc QWIKMD::TableSecLab {} {

    set text "T = Turn,
E = Extended Conformation,
B = Isolated Bridge,
H = Alpha-Helix,
G = 3-10 Helix,
I = Pi-Helix,
C = Coil"
    
    return $text
}
