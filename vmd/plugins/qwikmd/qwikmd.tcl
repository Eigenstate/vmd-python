#
# $Id: qwikmd.tcl,v 1.46 2016/10/24 21:16:43 jribeiro Exp $
#
#==============================================================================
# QwikMD
#
# Authors:
#   João V. Ribeiro
#   Beckman Institute for Advanced Science and Technology
#   University of Illinois, Urbana-Champaign
#   jribeiro@ks.uiuc.edu
#   http://www.ks.uiuc.edu/~jribeiro
#
#   Rafael C. Bernardi
#   Beckman Institute for Advanced Science and Technology
#   University of Illinois, Urbana-Champaign
#   rcbernardi@ks.uiuc.edu
#   http://www.ks.uiuc.edu/~rcbernardi/
#
#   Till Rudack
#   Beckman Institute for Advanced Science and Technology
#   University of Illinois, Urbana-Champaign
#   trudack@ks.uiuc.edu
#   http://www.ks.uiuc.edu/~trudack/
#
# Usage:
#   QwikMD was designed to be used exclusively through its GUI,
#   launched from the "Extensions->Simulation" menu.
#
#   Also see http://www.ks.uiuc.edu/Research/vmd/plugins/qwikmd/ for the
#   accompanying documentation.
#
#=============================================================================

package provide qwikmd 1.1

namespace eval ::QWIKMD:: {
    

    ### Read Packages
    package require Tablelist 
    package require autopsf
    package require Tk 8.5
    package require psfgen
    package require solvate
    package require topotools
    package require pbctools
    package require structurecheck
    package require membrane
    package require readcharmmtop
    package require mdff_gui
    global tcl_platform env

    ##Main GUI Variables


   #variable afterid {}
    #variable delta 0
    #variable maxy 0.0
    #variable maxx 0.0

    variable topGui ".qwikmd"
    variable bindTop 0
    variable topMol ""
    variable inputstrct ""
    variable nmrstep ""
    ################################################################################
    ## QWIKMD::chains(index "Select chain/type",indexes)
    ## contains information about the chain and keep track the initial chains and types in the pdb
    ## QWIKMD::chains(index "Select chain/type",0) -- boolean if it is select or not in "Select chain/type" menu
    ## QWIKMD::chains(index "Select chain/type",1) -- label in the "Select chain/type" dropdown menu
    ## QWIKMD::chains(index "Select chain/type",2) -- residues ids range
    ################################################################################
    array set chains ""
    ################################################################################
    ## QWIKMD::index_cmb
    ## contains information about the chain and the comboboxs in the main table
    ## QWIKMD::index_cmb($chain and $type,1) -- representation mode
    ## QWIKMD::index_cmb($chain and $type,2) -- color mode
    ## QWIKMD::index_cmb($chain and $type,3) -- index in the "Select chain/type" menu
    ## QWIKMD::index_cmb($chain and $type,4) -- main table color combobox path in the GUI
    ## QWIKMD::index_cmb($chain and $type,5) -- atomselection for that (chain and type) entry
    ################################################################################
    array set index_cmb ""
    array set cmb_type ""
    # variables equi, md and smd were depicted after the vmd 1.9.3 beta 3. Now is included in the basicGui array.
    # Kept in here for compatibility issues. 
    # Remove after all users switched to newer versions.
    variable equi 1
    variable md 1
    variable smd 1
    variable colorIdMap {{ResName} {ResType} {Name} {Structure} {Throb} {0 blue} {1 red} {2 gray} {3 orange} {4 yellow} \
    {5 tan} {6 silver} {7 green} {8 white} {9 pink} {10 cyan} {11 purple} {12 lime} {13 mauve} {14 ochre} {15 iceblue}\
     {16 black} {17 yellow2} {18 yellow3} {19 green2} {20 green3} {21 cyan2} {22 cyan3} {23 blue2} {24 blue3}\
         {25 violet} {26 violet2} {27 magenta} {28 magenta2} {29 red2} {30 red3} {31 orange2} {32 orange3} }
    variable outPath ""
    ################################################################################
    ## QWIKMD::basicGui
    ## stores widgets variables (name,0), and widgets path (name,1) if necessary
    ## QWIKMD::basicGui(mTable,0) - path to the main table in the basic protocol
    ## QWIKMD::basicGui(mTable,1) - path to the main table in the advanced protocol
    ## QWIKMD::basicGui(solvent,0) - solvent combobox
    ## QWIKMD::basicGui(solvent,boxbuffer) - solvent box buffer
    ## QWIKMD::basicGui(saltconc,0) - concentration entry
    ## QWIKMD::basicGui(saltions,"MD") - "choose salt" combobox
    ## QWIKMD::basicGui(temperature,0) -  temperature entry
    ## QWIKMD::basicGui(temperature,1) -  temperature label
    ## QWIKMD::basicGui(pspeed) - smd pulling speed
    ## QWIKMD::basicGui(plength) - smd pulling length
    ## QWIKMD::basicGui(mdtime,0) - MD simulation time
    ## QWIKMD::basicGui(mdtime,1) - SMD simulation time
    ## QWIKMD::basicGui(workdir,0) - working directory
    ## QWIKMD::basicGui(desktop) - desktop color selection
    ## QWIKMD::basicGui(scheme) - VMD scheme
    ## QWIKMD::basicGui(live) - live simulation boolean selection
    ## QWIKMD::basicGui(mdPrec,0) - live simulation boolean selection
    ## QWIKMD::basicGui(currenttime) - label for current Simulation time
    ## QWIKMD::basicGui(preparebtt,0) - lfrom basic run tab
    ## QWIKMD::basicGui(preparebtt,1) - lfrom basic run tab
    ################################################################################
    array set basicGui ""
    ################################################################################
    ## QWIKMD::advGui(addmol) - add number of molecules
    ## QWIKMD::advGui(saltconc,0)- concentration entry
    ## QWIKMD::advGui(saltions,"MD") - "choose salt" combobox
    ## QWIKMD::advGui(protocoltb,$QWIKMD::run) - table containing the protocol in Advanced Run Tab
    ## QWIKMD::advGui(protocoltb,index)- info about save as popup window
    ## QWIKMD::advGui(protocoltb,index,saveAsTemplate)- info about save as popup window
    ## QWIKMD::advGui(protocoltb,index,smd)- is smd?
    ## QWIKMD::advGui(protocoltb,index,lock)- is this protocol locked
    ## QWIKMD::advGui(analyze,level,?)- stores Guis values and path of analyze frames
    ## QWIKMD::advGui(analyze,advance,calcombo)- stores calculation combobox value in advanced analysis 
    ## QWIKMD::advGui(analyze,advance,calcbutton)- stores the path of the "Calculate" button in advanced analysis 
    ## QWIKMD::advGui(analyze,advanceframe)- stores the path of advance frame 
    ## QWIKMD::advGui(analyze,advance,qtmeptbl)- table present in temperature quench
    ## QWIKMD::advGui(analyze,advance,decayentry)- value of the autocorrelation decay time
    ################################################################################
    array set advGui ""
    variable runbtt ""
    #variable preparebtt ""
    variable notebooks ""
    ####ConfFile stores the protocols created or the protocols selected to be loaded after running the simulations
    #### prevconffile stores the list of all protocols created and saved in the qwikmd inputfile
    variable confFile ""
    variable prevconfFile ""
    variable cellDim ""
    variable logo ""
    variable state 0
    variable stop 1 
    variable load 0
    variable run MD
    variable runstep 0
    variable combovalues ""
    variable selected 1
    variable anchorpulling 0
    variable buttanchor 0
    array set color ""
    variable anchorRes ""
    variable pullingRes ""
    variable anchorRessel ""
    variable pullingRessel ""
    variable showanchor 0
    variable showpull 0
    variable ts 0
    variable restts 0
    variable lastframe ""
    variable viewpoints ""
    variable calcfreq 20
    variable smdfreq 40
    variable dcdfreq 1000
    variable timestep 2
    variable imdFreq 10
    variable hbondsprevx 0
    variable prepared 0
    variable inpFile ""
    variable showMdOpt 0
    variable bgcolor [ttk::style lookup TFrame -background]
    variable proteinmcr "(not name QWIKMDDELETE and protein)"
    variable nucleicmcr "(not name QWIKMDDELETE and nucleic)"
    variable glycanmcr "(not name QWIKMDDELETE and glycan)"
    variable lipidmcr "(not name QWIKMDDELETE and lipid)"
    variable heteromcr "(not name QWIKMDDELETE and hetero and not qwikmd_protein and not qwikmd_lipid and not qwikmd_nucleic and not qwikmd_glycan and not water)"
    atomselect macro qwikmd_protein $proteinmcr
    atomselect macro qwikmd_nucleic $nucleicmcr
    atomselect macro qwikmd_glycan $glycanmcr
    atomselect macro qwikmd_lipid $lipidmcr
    atomselect macro qwikmd_hetero $heteromcr

    variable prtclSelected -1

    array set nmrMenu ""
    array set chainMenu ""

    array set mdProtInfo ""

    variable refIndex [list]
    variable references [list]
    #variable renumber [list]
    variable textLogfile ""

    ##Select Residue GUI Variables
    variable selResidSel ""
    variable selResidSelIndex [list]
    variable selResidSelRep ""
    variable selResGui ".qwikmdResGui"
    variable selresTable ""
    variable selresPatcheFrame ""
    variable selresPatcheText ""
    variable patchestr ""
    array set protres ""
    variable tablemode "inspection"
    variable prevRes ""
    variable prevtype ""
    variable delete ""
    variable rename ""
    array set mutate ""
    variable mutindex ""
    array set protonate ""
    variable protindex ""
    array set dorename ""
    variable renameindex ""
    variable resrepname ""
    variable residtbprev ""
    variable anchorrepname ""
    variable pullingrepname ""
    variable reslist {ALA ARG ASN ASP CYS GLN GLU GLY HSD ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HSP HSE}
    variable hetero {ACET ACO ADP AMM1 AMP ATP BAR CD2 CAL CES CLA ETOH OH LIT MG NAD NADH NADP NDPH POT PYRM RUB SOD ZN2}
    variable heteronames {{Acetate} {Acetone} {ADP} {Ammonia} {AMP} {ATP} {Barium} {Cadmium II} {Calcium} {Cesium} {Chloride} {Ethanol} {Hydroxide} {Lithium} {Magnesium} \
    {NAD} {NADH} {NADP} {NDPH} {Potassium} {Pyrimidine} {Rubidium} {Sodium} {Zinc 2}}

    variable carb {AGLC BGLC AALT BALT AALL BALL AGAL BGAL AGUL BGUL AIDO BIDO AMAN BMAN ATAL BTAL AXYL BXYL AFUC BFUC ARHM BRHM}
    variable carbnames {{4C1 alpha-D-glucose} {4C1 beta-D-glucose} {4C1 alpha-D-altrose} {4C1 beta-D-altrose} {4C1 alpha-D-allose} {4C1 beta-D-allose} {4C1 alpha-D-galactose} {4C1 beta-D-galactose}\
     {4C1 alpha-D-gulose} {4C1 beta-D-gulose} {4C1 alpha-D-idose} {4C1 beta-D-idose} {4C1 alpha-D-mannos} {4C1 beta-D-mannose} {4C1 alpha-D-talose} \
    {4C1 beta-D-talose} {alpha-D-xylose} {beta-D-xylose} {alpha-L-fucose} {beta-L-fucose} {alpha-L-rhamnose} {beta-L-rhamnose}}
    variable nucleic {GUA ADE CYT THY URA}
    variable lipidname {LPPC DLPC DLPE DLPS DLPA DLPG DMPC DMPE DMPS DMPA DMPG DPPC DPPE DPPS DPPA DPPG DSPC DSPE DSPS DSPA DSPG DOPC DOPE DOPS DOPA DOPG POPC POPE POPS POPA POPG SAPC SDPC SOPC DAPC}
    variable numProcs ""
    variable gpu 1
    variable mdPrec 0
    variable maxSteps [list]
    set warnresid 0
    variable topoerror ""
    variable topolabel ""
    variable topocolor ""
    variable chirerror ""
    variable chirlabel ""
    variable chircolor ""

    variable cisperror ""
    variable cisplabel ""
    variable cispcolor ""

    variable gaps ""    
    variable gapslabel ""
    variable gapscolor ""

    variable torsionOutlier ""  
    variable torsionOutliearlabel ""
    variable torsionOutliearcolor ""
    variable torsionTotalResidue 0

    variable torsionMarginal "" 
    variable torsionMarginallabel ""
    variable torsionMarginalcolor ""

    variable tabprev -1
    variable tabprevmodf -1

    variable resallnametype 1
    ##Edit Atoms GUI Variables
    variable editATMSGui ".qwikmdeditAtm"
    variable atmsTable ""
    variable atmsText ""
    variable atmsNames ""
    variable atmsOrigNames ""
    variable atmsMol ""
    variable atmsLables ""
    variable atmsDeleteNames [list]
    variable atmsRename [list]
    variable atmsOrigResid [list]
    variable charmTopoInfo [list]
    variable atmsRenameLog [list]
    variable atmsDeleteLog [list]
    variable atmsReorderLog [list]

    #####################################
    ## List of lists defining the user specific parameters and
    ## macros atom selection
    ## each list :
    ##      index 0 - macro name/molecule type
    ##      index 1 - list of Charmm resdues names
    ##      index 2 - list of user resdues denomination
    ## 
    variable userMacros [list]
    array set topocombolist ""
    ##Select TOPO+PARAM GUI Variables
    variable topoPARAMGUI ".qwikmdTopoParam"
    variable topparmTable ""
    variable topparmTableError 0
    # motion indicators (from FFTK GUI)
    # tree element open and close indicators
    set downPoint \u25BC
    set rightPoint \u25B6
    set tempEntry "#696969"

    ##Loading Option Window
    variable loadremovewater 0
    variable loadremoveions 0
    variable loadremovehydrogen 0
    variable loadinitialstruct 0
    variable loadstride 1
    variable loadprotlist [list]
    ##RMSD Plot Variables
    
    variable rmsdGui ""
    
    variable rmsdsel "backbone"
    variable rmsdseltext "protein"
    set line ""
    variable timeXrmsd 0
    variable rmsd 0
    variable rmsdprevx 0
    variable lastrmsd -1
    variable counterts 0
    variable prevcounterts 0
    variable rmsdplotview 0
    ##Hydrogen Plot Variables
    
    variable HBondsGui ""
    variable hbondsGui ""
    variable lasthbond -1
    variable hbonds ""
    variable timeXhbonds ""
    variable hbondssel "intra"
    variable hbondsplotview 0
    variable hbondsrepname ""
    ##Energies Plot Variables
    
    variable energyTotGui ""
    variable energyKineGui ""
    variable energyPotGui ""
    variable energyBondGui ""
    variable energyAngleGui ""
    variable energyDehidralGui ""
    variable energyVdwGui ""
    
    variable lastenetot -1
    variable lastenekin -1
    variable lastenepot -1
    variable lastenebond -1
    variable lasteneangle -1
    variable lastenedihedral -1
    variable lastenevdw -1

    variable eneprevx 0
    variable enecurrentpos 0
    variable eneprevx 0
    variable enecurrentpos 0
    variable enetotval ""
    variable enetotpos ""
    variable enekinval ""
    variable enekinpos ""
    variable enepotval ""
    variable enepotpos ""
    variable enebondval ""
    variable enebondpos ""
    variable eneangleval ""
    variable eneanglepos ""
    variable enedihedralval ""
    variable enedihedralpos ""
    variable enevdwval ""
    variable enevdwpos ""
    variable enerkinetic 0
    variable enertotal 1
    variable enerpoten 0
    variable eneplotview 0
    variable enerbond 0
    variable enerangle 0
    variable enerdihedral 0 
    variable enervdw 0 
    ##Conditions Plot Variables
    variable tempGui ""
    variable pressGui ""
    variable volGui ""
    variable plotwindowCON ""
    variable CondGui ".qwikmdCONGui"
    variable lasttemp -1
    variable lastpress -1
    variable lastvol -1
    variable tempval ""
    variable temppos ""
    variable pressval ""
    variable pressvalavg ""
    variable presspos ""
    variable volval ""
    variable volvalavg ""
    variable volpos ""
    variable condcurrentpos 0
    variable tempcalc 1
    variable pressurecalc 0
    variable volumecalc 0
    variable condprevx 0
    variable condprevindex 0
    variable condplotview 0
    variable condcurrentpos 0
    ####Tmperature quench variables
    variable radiobtt ""
    variable qtempGui ""
    variable qtempval ""
    variable qtemppos ""
    variable qtemprevx
    ####Maxwell-Boltzmann Energy Distribution variables
    variable MBGui ""
    ####Specific Heat variables
    variable SPHGui ""
    ####Temperature Distribution variables
    variable tempDistGui ""
    ####SASA variables
    variable SASAGui ""
    variable sasarep ""
    variable sasarepTotal1 ""
    variable sasarepTotal2 ""
    ####CSASA variables
    variable CSASAGui ""
    
    ####RMSF variables
    variable rmsfGui ""
    variable rmsfrep ""
    ##SMD Plot Variables
    variable SMDGui ".qwikmdSMDDGui"
    variable smdGui ""
    variable plotwindowSMD ""
    variable lastsmd -1
    variable timeXsmd 0
    variable smdvals 0
    variable smdvalsavg 0
    variable smdfirstdist ""
    variable countertssmd 0
    variable smdxunit "time"
    variable smdcurrentpos 0
    variable smddistance 0
    variable smdplotview 0
    variable smdcurrentpos 0
    variable smddistance 0
    
    variable smdprevindex 0
    variable prevcountertsmd 0
    
    variable membranebox [list]
    variable membraneFrame ""

    variable pbcInfo ""
    
    global env

    set ParameterList [glob $env(CHARMMPARDIR)/*36*.prm]
    set str [glob $env(CHARMMPARDIR)/*.str]
    
    set ParameterList [concat $str $ParameterList]
    
    lappend TopList [file join $env(CHARMMTOPDIR) top_all36_prot.rtf]
    lappend TopList [file join $env(CHARMMTOPDIR) top_all36_lipid.rtf]
    lappend TopList [file join $env(CHARMMTOPDIR) top_all36_na.rtf]
    lappend TopList [file join $env(CHARMMTOPDIR) top_all36_carb.rtf]
    lappend TopList [file join $env(CHARMMTOPDIR) top_all36_cgenff.rtf]
    lappend TopList [file join $env(CHARMMTOPDIR) toppar_all36_carb_glycopeptide.str]
    lappend TopList [file join $env(CHARMMTOPDIR) toppar_water_ions_namd.str]
    for {set i 0} {$i < [llength $str]} {incr i} {
        if {[lsearch [file tail [lindex $str $i]] $TopList] == -1} {
            lappend TopList [lindex $str $i]
        }
    }
    
}

# source code base
source [file join $env(QWIKMDDIR) qwikmd_func.tcl]
source [file join $env(QWIKMDDIR) qwikmd_info.tcl]
source [file join $env(QWIKMDDIR) qwikmd_logText.tcl]
source [file join $env(QWIKMDDIR) qwikmd_ballon.tcl]

proc qwikmd {} { return [eval QWIKMD::qwikmd]}

proc QWIKMD::qwikmd {} {
    global env

    if {[winfo exists $QWIKMD::topGui] != 1} {
        QWIKMD::path
        
    } else {
        wm deiconify $QWIKMD::topGui
    }
    raise $QWIKMD::topGui

    QWIKMD::checkDeposit
    QWIKMD::resetBtt 2
    wm deiconify $QWIKMD::topGui

    return $QWIKMD::topGui
}

############################################################
## The reset command receives an option depending of what is 
## intended to clean:
##       opt = 0 - everything but simulation options and 
##                 strucuture manipulation options (e.g. mutations)
##       
##       opt = 1 - everything but strucuture manipulation options (e.g. mutations) 
##       
##       opt = 2 - restores qwikMD to the initial state
############################################################


proc QWIKMD::resetBtt {opt} {

    if {$opt > 1} {
        set continue [QWIKMD::checkIMD]
        if {$continue == 0} {
            return 1
        }
    }

    set tabid 0
    if {[$QWIKMD::topGui.nbinput tab 0 -state] == "disabled"} {
        set tabid 1
    }
    $QWIKMD::topGui.nbinput select $tabid 
    if {$opt > 0} {

        if {[winfo exists $QWIKMD::editATMSGui] == 1} {
            destroy $QWIKMD::editATMSGui
        }
        
        if {[winfo exists $QWIKMD::topoPARAMGUI] == 1} {
            destroy $QWIKMD::topoPARAMGUI
        }

        $QWIKMD::topGui.nbinput.f1.tableframe.tb delete 0 end
        $QWIKMD::topGui.nbinput.f2.tableframe.tb delete 0 end
        $QWIKMD::advGui(protocoltb,MD) delete 0 end
        $QWIKMD::advGui(protocoltb,SMD) delete 0 end
        $QWIKMD::topGui.nbinput.f1.selframe.mCHAIN.chain delete 0 end
        $QWIKMD::topGui.nbinput.f1.selframe.mNMR.nmr delete 0 end
        $QWIKMD::topGui.nbinput tab 0 -state normal
        $QWIKMD::topGui.nbinput tab 1 -state normal
        foreach note $QWIKMD::notebooks {
            set tabids [$note tabs]
            $note state "!disabled"
        }
        $QWIKMD::basicGui(workdir,1) configure -state normal
        $QWIKMD::basicGui(workdir,2) configure -state normal
        $QWIKMD::topGui.nbinput select 0
        [lindex $QWIKMD::notebooks 1] select 0
        $QWIKMD::nmrMenu(basic) configure -state normal
        $QWIKMD::nmrMenu(advanced) configure -state normal
        $QWIKMD::chainMenu(basic) configure -state normal
        $QWIKMD::chainMenu(advanced) configure -state normal
        
        ttk::style configure WorkDir.TEntry -foreground $QWIKMD::tempEntry
        ttk::style configure RmsdSel.TEntry -foreground $QWIKMD::tempEntry
        ttk::style configure RmsdAli.TEntry -foreground $QWIKMD::tempEntry
        ttk::style configure PdbEntrey.TEntry -foreground $QWIKMD::tempEntry
        set QWIKMD::basicGui(workdir,0) "Working Directory"
        
        set QWIKMD::basicGui(currenttime) "Completed 0.000 of 0.000 ns"
        set QWIKMD::basicGui(live) 0
        set QWIKMD::basicGui(mdPrec,0) 0
        set QWIKMD::run MD
        set QWIKMD::confFile ""
        set QWIKMD::prevconfFile ""
        set QWIKMD::cellDim ""
        set QWIKMD::anchorRes ""
        set QWIKMD::anchorrepname ""
        set QWIKMD::pullingrepname ""
        set QWIKMD::pullingRes ""
        set QWIKMD::viewpoints ""
        set QWIKMD::anchorpulling 0
        set QWIKMD::showanchor 0
        set QWIKMD::showpull 0
        set QWIKMD::anchorRessel ""
        set QWIKMD::pullingRessel ""
        set QWIKMD::selResidSel "Type Selection"
        set QWIKMD::selResidSelIndex [list]
        set QWIKMD::selResidSelRep ""
        set QWIKMD::inputstrct "PDB ID"
        set QWIKMD::inpFile ""
        set QWIKMD::proteinmcr "(not name QWIKMDDELETE and protein)"
        set QWIKMD::nucleicmcr "(not name QWIKMDDELETE and nucleic)"
        set QWIKMD::glycanmcr "(not name QWIKMDDELETE and glycan)"
        set QWIKMD::lipidmcr "(not name QWIKMDDELETE and lipid)"
        set QWIKMD::heteromcr "(not name QWIKMDDELETE and hetero and not qwikmd_protein and not qwikmd_lipid and not qwikmd_nucleic and not qwikmd_glycan and not water)"
        set QWIKMD::maxSteps [list]
        set QWIKMD::atmsNames ""
        set QWIKMD::atmsMol ""
        set QWIKMD::atmsLables ""
        set QWIKMD::atmsDeleteNames [list]
        set QWIKMD::atmsOrigNames ""
        set QWIKMD::atmsOrigResid ""
        set QWIKMD::atmsRename [list]
        set QWIKMD::atmsRenameLog [list]
        set QWIKMD::atmsDeleteLog [list]
        set QWIKMD::atmsReorderLog [list]


        if {[info exists QWIKMD::advGui(membrane,center,y)]} {
            unset QWIKMD::advGui(membrane,center,x)
            unset QWIKMD::advGui(membrane,center,y)
            unset QWIKMD::advGui(membrane,center,z)
            unset QWIKMD::advGui(membrane,rotate,x)
            unset QWIKMD::advGui(membrane,rotate,y)
            unset QWIKMD::advGui(membrane,rotate,z)
        }
        set QWIKMD::membranebox [list]
        global env
        set tempLib ""
        catch {glob $env(TMPDIR)/*.conf} tempLib
        if {[file isfile [lindex ${tempLib} 0]] == 1} {
            foreach file $tempLib {
                catch {file delete -force -- ${file}}
            }
        }
        catch {glob $env(TMPDIR)/Renumber_Residues.txt} tempLib
        if {[file isfile ${tempLib}] == 1} {
            catch {file delete -force -- ${tempLib}}
        }
        catch {glob $env(TMPDIR)/membrane.pdb} tempLib
        if {[file isfile ${tempLib}] == 1} {
            catch {file delete -force -- ${tempLib}}
        }
        catch {glob $env(TMPDIR)/torplot_temp.pdb} tempLib
        if {[file isfile ${tempLib}] == 1} {
            catch {file delete -force -- ${tempLib}}
        }
        }
    if {$opt >=1} {

        array unset QWIKMD::mutate *
        set QWIKMD::mutindex ""
        array unset QWIKMD::protonate *
        set QWIKMD::protindex ""
        array unset QWIKMD::dorename * 
        set QWIKMD::renameindex ""
        array unset QWIKMD::protres *
        set QWIKMD::patchestr ""
        destroy $QWIKMD::selResGui
        
        if {$opt > 1} {
            set QWIKMD::topoerror ""
            set QWIKMD::chirerror ""
            set QWIKMD::cisperror ""
            set QWIKMD::gaps ""
            set QWIKMD::torsionOutlier ""   
            set QWIKMD::torsionMarginal ""  
            set QWIKMD::torsionTotalResidue 0
            array unset QWIKMD::mdProtInfo *
            set QWIKMD::references [list]
            set QWIKMD::refIndex [list]
            set QWIKMD::textLogfile ""
            set QWIKMD::pbcInfo ""
            #set QWIKMD::renumber [list]
            #Populated the advanced protocol tables (MD and SMD)
            set prt {MD SMD}
            set QWIKMD::prtclSelected -1
            set numcols [$QWIKMD::advGui(protocoltb,MD) columncount]
            foreach run $prt {
                set QWIKMD::run $run
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) delete 0 end
                for {set i 0} {$i < 4} {incr i} {QWIKMD::addProtocol;
                    #QWIKMD::checkProc $i
                }
                for {set i 0} {$i < $numcols} {incr i} {$QWIKMD::advGui(protocoltb,$QWIKMD::run) columnconfigure $i -editable true}
            }
            
            set QWIKMD::run MD
            set QWIKMD::advGui(solvent,minimalbox) 0
            set val {MD SMD}
            foreach run $val {
                set QWIKMD::basicGui(prtcl,$run,equi) 1
                set QWIKMD::basicGui(prtcl,$run,md) 1
                $QWIKMD::basicGui(prtcl,$run,mdbtt) configure -state normal
                $QWIKMD::basicGui(prtcl,$run,equibtt) configure -state normal
                if {$run == "SMD"} {
                    set QWIKMD::basicGui(prtcl,$run,smd) 1
                    $QWIKMD::basicGui(prtcl,$run,smdbtt) configure -state normal
                }
            }
            $QWIKMD::runbtt configure -text "Start MD Simulation"
            $QWIKMD::runbtt configure -state normal
            set QWIKMD::basicGui(currenttime,0) ""
            set QWIKMD::basicGui(currenttime,1) ""

            grid conf $QWIKMD::basicGui(currenttime,pgframe) -row 5 -column 0 -pady 2 -sticky nsew
            grid conf $QWIKMD::advGui(currenttime,pgframe) -row 5 -column 0 -pady 2 -sticky nsew
            if {[info exists $QWIKMD::topoPARAMGUI.f1.tableframe.tb]} {
                $QWIKMD::topoPARAMGUI.f1.tableframe.tb delete 0 end
            }
            
            $QWIKMD::topGui.nbinput.f2.fcontrol.fcolapse.f1.imd.button_Pause configure -state normal
            $QWIKMD::topGui.nbinput.f2.fcontrol.fcolapse.f1.imd.button_Finish configure -state normal
            $QWIKMD::topGui.nbinput.f2.fcontrol.fcolapse.f1.imd.button_Detach configure -state normal
            
            set numtabs [llength [$QWIKMD::topGui.nbinput tabs]]
            for {set i 0} {$i < $numtabs} {incr i} {
                $QWIKMD::topGui.nbinput tab $i -state normal
            }

            $QWIKMD::topGui.nbinput.f[expr $tabid +1].fb.fcolapse.f1.preparereset.live configure -state normal
            $QWIKMD::topGui configure -cursor {}; update
            
            set QWIKMD::ParameterList [list] 
            set QWIKMD::TopList [list] 

            set QWIKMD::ParameterList [glob $env(CHARMMPARDIR)/*36*.prm]
            set str [glob $env(CHARMMPARDIR)/*.str]
            
            set QWIKMD::ParameterList [concat $str $QWIKMD::ParameterList]
            
            lappend QWIKMD::TopList [file join $env(CHARMMTOPDIR) top_all36_prot.rtf]
            lappend QWIKMD::TopList [file join $env(CHARMMTOPDIR) top_all36_lipid.rtf]
            lappend QWIKMD::TopList [file join $env(CHARMMTOPDIR) top_all36_na.rtf]
            lappend QWIKMD::TopList [file join $env(CHARMMTOPDIR) top_all36_carb.rtf]
            lappend QWIKMD::TopList [file join $env(CHARMMTOPDIR) top_all36_cgenff.rtf]
            lappend QWIKMD::TopList [file join $env(CHARMMTOPDIR) toppar_all36_carb_glycopeptide.str]
            lappend QWIKMD::TopList [file join $env(CHARMMTOPDIR) toppar_water_ions_namd.str]
            for {set i 0} {$i < [llength $str]} {incr i} {
                if {[lsearch [file tail [lindex $str $i]] $QWIKMD::TopList] == -1} {
                    lappend QWIKMD::TopList [lindex $str $i]
                }
            }
            QWIKMD::reviewTopPar
            QWIKMD::loadTopologies
            QWIKMD::changeBCK
            display update on
            update
            
        }

    }

    atomselect macro qwikmd_protein $QWIKMD::proteinmcr
    atomselect macro qwikmd_nucleic $QWIKMD::nucleicmcr
    atomselect macro qwikmd_glycan $QWIKMD::glycanmcr
    atomselect macro qwikmd_lipid $QWIKMD::lipidmcr
    atomselect macro qwikmd_hetero $QWIKMD::heteromcr
    destroy $QWIKMD::advGui(analyze,basic,ntb).volumecalc
    destroy $QWIKMD::advGui(analyze,basic,ntb).pressurecalc
    destroy $QWIKMD::advGui(analyze,basic,ntb).tempcalc
    destroy $QWIKMD::advGui(analyze,basic,ntb).enertotal
    destroy $QWIKMD::advGui(analyze,basic,ntb).enerkinetic
    destroy $QWIKMD::advGui(analyze,basic,ntb).enerpoten
    destroy $QWIKMD::advGui(analyze,basic,ntb).enerbond
    destroy $QWIKMD::advGui(analyze,basic,ntb).enerangle
    destroy $QWIKMD::advGui(analyze,basic,ntb).enerdihedral
    destroy $QWIKMD::advGui(analyze,basic,ntb).enervdw
    destroy $QWIKMD::advGui(analyze,basic,ntb).frmsd
    set QWIKMD::resallnametype 1
    set QWIKMD::tabprev -1
    set QWIKMD::tabprevmodf 1
    set QWIKMD::basicGui(solvent,0) "Implicit"
    set QWIKMD::advGui(solvent,0) "Explicit"
    
    set QWIKMD::advGui(solvent,boxbuffer) 15
    set QWIKMD::advGui(addmol) "10"
    set QWIKMD::basicGui(saltconc,0) "0.15"
    set QWIKMD::basicGui(saltions,0) "NaCl"
    set QWIKMD::basicGui(temperature,0) "27"
    set QWIKMD::basicGui(temperature,1) "27"
    set QWIKMD::basicGui(mdtime,0) "10.0"
    set QWIKMD::basicGui(mdtime,1) 0
    set QWIKMD::basicGui(plength) 10.0
    set QWIKMD::basicGui(pspeed) 2.5
    set QWIKMD::delete ""
    set QWIKMD::rename ""
    array unset QWIKMD::chains *
    array unset QWIKMD::index_cmb *
    array set QWIKMD::index_cmb ""
    set QWIKMD::rmsdGui ""
    set QWIKMD::smdGui ""
    set QWIKMD::hbondsGui ""
    set QWIKMD::plotwindow ""
    set QWIKMD::plotwindowSMD ""
    set QWIKMD::plotwindowHB ""
    set QWIKMD::energyTotGui ""
    set QWIKMD::energyKineGui ""
    set QWIKMD::energyPotGui ""
    
    set QWIKMD::energyBondGui ""
    set QWIKMD::energyAngleGui ""
    set QWIKMD::energyDehidralGui ""
    set QWIKMD::energyVdwGui ""
    set QWIKMD::topMol ""
    set QWIKMD::nmrstep ""
    set QWIKMD::state 0
    set QWIKMD::stop 1 
    set QWIKMD::rmsdsel "all"
    set QWIKMD::timestep 2
    set QWIKMD::imdFreq 10
    set QWIKMD::load 0
    set QWIKMD::lastframe ""
    set QWIKMD::runstep 0
    set QWIKMD::residtbprev ""
    set QWIKMD::resrepname ""
    set QWIKMD::combovalues ""
    set QWIKMD::tablemode "inspection"
    set QWIKMD::selected 1  
    set QWIKMD::buttanchor 0
    array unset QWIKMD::color *
    set QWIKMD::prevRes ""
    set QWIKMD::prevtype ""
    set QWIKMD::timeXrmsd 0
    set QWIKMD::rmsd 0
    set QWIKMD::timeXsmd ""
    set QWIKMD::smdvals ""
    set QWIKMD::smdvalsavg ""
    set QWIKMD::smdfirstdist ""
    set QWIKMD::ts 0
    set QWIKMD::counterts 0
    set QWIKMD::prevcounterts 0
    set QWIKMD::prevcountertsmd 0
    set QWIKMD::countertssmd 0
    set QWIKMD::restts 0
    set QWIKMD::smdxunit "time"
    set QWIKMD::smdcurrentpos 0
    set QWIKMD::smddistance 0
    set QWIKMD::rmsdprevx 0
    set QWIKMD::hbondsprevx 0
    set QWIKMD::timeXhbonds ""
    set QWIKMD::hbonds ""
    set QWIKMD::enertotal 1
    set QWIKMD::enerpoten 0
    set QWIKMD::enerkinetic 0
    set QWIKMD::enerbond 0
    set QWIKMD::enerangle 0
    set QWIKMD::enerdihedral 0
    set QWIKMD::enervdw 0
    set QWIKMD::calcfreq 20
    set QWIKMD::smdfreq 40
    set QWIKMD::dcdfreq 1000
    set QWIKMD::warnresid 0
    set QWIKMD::prepared 0
    set QWIKMD::hbondssel "intra"
    set QWIKMD::hbondsrepname ""
    set QWIKMD::enecurrentpos 0
    set QWIKMD::eneprevx 0
    set QWIKMD::enecurrentpos 0
    set QWIKMD::enetotval ""
    set QWIKMD::enetotpos ""

    set QWIKMD::enebondval ""
    set QWIKMD::enebondpos ""
    set QWIKMD::eneangleval ""
    set QWIKMD::eneanglepos ""
    set QWIKMD::enedihedralval ""
    set QWIKMD::enedihedralpos ""
    set QWIKMD::enevdwval ""
    set QWIKMD::enevdwpos ""

    set QWIKMD::lastrmsd -1
    set QWIKMD::lasthbond -1
    set QWIKMD::lastsmd -1
    set QWIKMD::lastenetot -1
    set QWIKMD::lastenekin -1
    set QWIKMD::lastenepot -1
    set QWIKMD::lastenebond -1
    set QWIKMD::lasteneangle -1
    set QWIKMD::lastenedihedral -1
    set QWIKMD::lastenevdw -1
    set QWIKMD::enekinval ""
    set QWIKMD::enekinpos ""
    set QWIKMD::enepotval ""
    set QWIKMD::enepotpos ""
    set QWIKMD::CondGui ".qwikmdCONGui"
    set QWIKMD::tempGui ""
    set QWIKMD::pressGui ""
    set QWIKMD::volGui ""
    set QWIKMD::plotwindowCON ""
    set QWIKMD::lasttemp -1
    set QWIKMD::lastpress -1
    set QWIKMD::lastvol -1
    set QWIKMD::tempcalc 1
    set QWIKMD::pressurecalc 0
    set QWIKMD::volumecalc 0
    set QWIKMD::condprevx 0
    set QWIKMD::condcurrentpos 0
    set QWIKMD::condprevindex 0
    set QWIKMD::condplotview 0
    set QWIKMD::pressvalavg [list]
    set QWIKMD::volvalavg [list]
    set QWIKMD::tempval ""
    set QWIKMD::temppos ""
    set QWIKMD::pressval ""
    set QWIKMD::presspos ""
    set QWIKMD::volval ""
    set QWIKMD::volpos ""
    set QWIKMD::rmsdplotview 0
    set QWIKMD::hbondsplotview 0
    set QWIKMD::eneplotview 0
    set QWIKMD::condplotview 0
    set QWIKMD::smdplotview 0
    set QWIKMD::smdprevindex 0
    set mollist [molinfo list]
    set QWIKMD::showMdOpt 0
    set QWIKMD::numProcs [QWIKMD::procs]
    set QWIKMD::gpu 1
    set QWIKMD::mdPrec 0
    set QWIKMD::topparmTable ""
    set QWIKMD::topparmTableError 0
    set QWIKMD::rmsfrep ""
    set QWIKMD::sasarep ""
    set QWIKMD::sasarepTotal1 ""
    set QWIKMD::sasarepTotal2 ""
    set QWIKMD::SASAGui ""
    set QWIKMD::CSASAGui ""
    set QWIKMD::rmsfGui ""
    set QWIKMD::SPHGui ""
    set QWIKMD::MBGui ""
    set QWIKMD::qtempGui ""
    set QWIKMD::tempDistGui ""
    set QWIKMD::membranebox [list]
    set QWIKMD::bindTop 0
    set QWIKMD::loadremovewater 0
    set QWIKMD::loadremoveions 0
    set QWIKMD::loadremovehydrogen 0
    set QWIKMD::loadinitialstruct 0
    set QWIKMD::loadstride 1
    set QWIKMD::loadprotlist [list]
    # MDFF protocol frame default values 
    set QWIKMD::advGui(mdff,min) 200
    set QWIKMD::advGui(mdff,mdff) 50000
    $QWIKMD::advGui(protocoltb,MDFF) delete 0 end
    $QWIKMD::advGui(protocoltb,MDFF) insert end {none "same fragment as protein" "same fragment as protein" "same fragment as protein"}
    for {set i 0} {$i < 4} {incr i} {$QWIKMD::advGui(protocoltb,MDFF) columnconfigure $i -editable true}
    set index [expr [llength $QWIKMD::notebooks] -2]
    for {set i $index} {$i < [llength $QWIKMD::notebooks]} {incr i} {
        set tabs [ [lindex $QWIKMD::notebooks $i] tabs ]
        foreach tab $tabs {
            destroy $tab
        }
    }
    set QWIKMD::membraneFrame ""

    for {set i 0} {$i < [llength $mollist]} {incr i} {
        mol delete [lindex $mollist $i]
    }
    set QWIKMD::basicGui(mdtime,1) [expr [expr {$QWIKMD::basicGui(plength) / $QWIKMD::basicGui(pspeed)} *100 ] /100]
    set QWIKMD::advGui(analyze,advance,calcombo) "H Bonds"
    QWIKMD::AdvancedSelected

    
    set prt {MD SMD MDFF}
    foreach run $prt {
        if {$run != "MDFF"} {
            $QWIKMD::basicGui(solvent,$run) configure -state readonly
            $QWIKMD::basicGui(saltions,$run) configure -state readonly
            $QWIKMD::basicGui(saltconc,$run) configure -state normal
        } 
        $QWIKMD::advGui(solvent,$run) configure -state readonly
        $QWIKMD::advGui(saltions,$run) configure -state readonly
        $QWIKMD::advGui(saltconc,$run) configure -state normal
        $QWIKMD::advGui(solvent,boxbuffer,$run,entry) configure -state readonly

    }
    QWIKMD::ChangeSolvent


}
#############################
## Main qwikMD GUI builder###
#############################

proc QWIKMD::path {} {
    global env
    display resetview
    set nameLayer ""
    if {[winfo exists $QWIKMD::topGui] != 1} {
        toplevel $QWIKMD::topGui

    }
    ttk::style map TCombobox -fieldbackground [list readonly #ffffff]

    grid columnconfigure $QWIKMD::topGui 0 -weight 1
    grid columnconfigure $QWIKMD::topGui 1 -weight 0
    grid rowconfigure $QWIKMD::topGui 0 -weight 0
    grid rowconfigure $QWIKMD::topGui 1 -weight 1
    ## Title of the windows
    wm title $QWIKMD::topGui "QwikMD - Easy and Fast Molecular Dynamics" ;# titulo da pagina

    wm grid $QWIKMD::topGui 50 180 1 1

    
    grid [ttk::frame $QWIKMD::topGui.f0] -row 0 -column 0 -sticky ew
    grid columnconfigure $QWIKMD::topGui.f0 0 -weight 1
    grid rowconfigure $QWIKMD::topGui.f0 0 -weight 1
    grid rowconfigure $QWIKMD::topGui.f0 1 -weight 1
    grid [ttk::frame $QWIKMD::topGui.f0.info] -row 0 -column 1 -sticky ens

    bind $QWIKMD::topGui <Button-1> {
        if {$QWIKMD::bindTop == 0} {
            wm protocol $QWIKMD::topGui WM_DELETE_WINDOW QWIKMD::closeQwikmd
            set QWIKMD::bindTop 1
        }       
    }
    ###################################################################
    ## Add info usage (QWIKMD::createInfoButton $frame $row $column
    ###################################################################

    QWIKMD::createInfoButton $QWIKMD::topGui.f0.info 0 1

    grid [ttk::button $QWIKMD::topGui.f0.info.help -text "Help..." -padding "2 0 2 0" -command {vmd_open_url [string trimright [vmdinfo www] /]/plugins/qwikmd}] -row 0 -column 0 -sticky ens -padx 2


    grid [ttk::notebook $QWIKMD::topGui.nbinput ] -row 1 -column 0 -sticky news -padx 0

    grid columnconfigure $QWIKMD::topGui.nbinput 0 -weight 1
    grid rowconfigure $QWIKMD::topGui.nbinput 1 -weight 1

    lappend QWIKMD::notebooks "$QWIKMD::topGui.nbinput"

    ttk::frame $QWIKMD::topGui.nbinput.f1
    grid columnconfigure $QWIKMD::topGui.nbinput.f1 0 -weight 1
    grid rowconfigure $QWIKMD::topGui.nbinput.f1 1 -weight 0
    grid rowconfigure $QWIKMD::topGui.nbinput.f1 2 -weight 2

    ttk::frame $QWIKMD::topGui.nbinput.f2
    grid columnconfigure $QWIKMD::topGui.nbinput.f2 0 -weight 1
    grid rowconfigure $QWIKMD::topGui.nbinput.f2 1 -weight 0
    grid rowconfigure $QWIKMD::topGui.nbinput.f2 2 -weight 2
    grid rowconfigure $QWIKMD::topGui.nbinput.f2 3 -weight 0
    grid rowconfigure $QWIKMD::topGui.nbinput.f2 4 -weight 2

    ttk::frame $QWIKMD::topGui.nbinput.f3
    grid columnconfigure $QWIKMD::topGui.nbinput.f3 0 -weight 1
    grid rowconfigure $QWIKMD::topGui.nbinput.f3 0 -weight 1


    ttk::frame $QWIKMD::topGui.nbinput.f4
    grid columnconfigure $QWIKMD::topGui.nbinput.f4 0 -weight 1
    grid rowconfigure $QWIKMD::topGui.nbinput.f4 0 -weight 1


    $QWIKMD::topGui.nbinput add $QWIKMD::topGui.nbinput.f1 -text "Easy Run" -sticky news 
    $QWIKMD::topGui.nbinput add $QWIKMD::topGui.nbinput.f2 -text "Advanced Run"  -sticky news
 
    $QWIKMD::topGui.nbinput add $QWIKMD::topGui.nbinput.f3 -text "Basic Analysis" -sticky news
    $QWIKMD::topGui.nbinput add $QWIKMD::topGui.nbinput.f4 -text "Advanced Analysis"  -sticky news

    ##################################################################################
    ## Change the content of the info button to display when Run or analysis tab is selected 
    ##################################################################################
    
    QWIKMD::BuildRun $QWIKMD::topGui.nbinput.f1 basic
    QWIKMD::BuildRun $QWIKMD::topGui.nbinput.f2 advanced

    QWIKMD::BasicAnalyzeFrame $QWIKMD::topGui.nbinput.f3
    QWIKMD::AdvancedAnalyzeFrame $QWIKMD::topGui.nbinput.f4
    bind $QWIKMD::topGui.nbinput <<NotebookTabChanged>> QWIKMD::changeMainTab
}

proc QWIKMD::checkIMD {} {
    set returnval 2
    if {$QWIKMD::basicGui(live) == 1 && $QWIKMD::prepared == 1 && [$QWIKMD::runbtt cget -state] == "disabled"} {
        set answer [tk_messageBox -message "QwikMD will terminate any active simulation. Do you want to continue?" -title "Running Simulation" -icon warning -type yesno]
        if {$answer == "yes"} {
            QWIKMD::killIMD
            set returnval 1
        } else {
            set returnval 0
        }
    }
    return $returnval
}
proc QWIKMD::closeQwikmd {} {
    if {[QWIKMD::checkIMD] == 0} {
        return
    } 
    set QWIKMD::prepared 0
    set QWIKMD::basicGui(live) 0
    QWIKMD::resetBtt 2
    set QWIKMD::bindTop 0
    wm withdraw $QWIKMD::topGui
    
}

proc QWIKMD::changeMainTab {} {
    if {[$QWIKMD::topGui.nbinput index current] == 2 || [$QWIKMD::topGui.nbinput index current] == 3} {
        bind $QWIKMD::topGui.f0.info.info <Button-1> {
            set val [QWIKMD::analyInfo]
            set QWIKMD::link [lindex $val 1]
            QWIKMD::infoWindow analyInfo [lindex $val 0] [lindex $val 2]
        }
    } else {
        bind $QWIKMD::topGui.f0.info.info <Button-1> {
            set val [QWIKMD::introInfo]
            set QWIKMD::link [lindex $val 1]
            QWIKMD::infoWindow introInfo [lindex $val 0] [lindex $val 2]                
        }

        if {([info exists QWIKMD::advGui(membrane,frame)] == 1 || [winfo exists $QWIKMD::selResGui.f1.frameOPT.atmsel] == 1)  && [winfo exists $QWIKMD::selResGui] == 1 && [wm title $QWIKMD::selResGui] == "Structure Manipulation/Check"} {       
            if {[$QWIKMD::topGui.nbinput index current] == 0 || $QWIKMD::prepared == 1 || $QWIKMD::load == 1} {
                if {[winfo exists $QWIKMD::advGui(membrane,frame)]} {
                    grid forget $QWIKMD::advGui(membrane,frame)
                    grid forget $QWIKMD::selresPatcheFrame
                }
                if {[winfo exists $QWIKMD::selResGui.f1.frameOPT.atmsel] && [$QWIKMD::topGui.nbinput index current] == 0} {
                    grid forget $QWIKMD::selResGui.f1.frameOPT.atmsel
                }
            } elseif {$QWIKMD::prepared == 0 && $QWIKMD::load == 0 && [$QWIKMD::topGui.nbinput index current] == 1} {
                grid conf $QWIKMD::advGui(membrane,frame) -row 4 -column 0 -sticky nwe -padx 2 -pady 2
                grid conf $QWIKMD::selResGui.f1.frameOPT.atmsel -row 1 -column 0 -sticky nwe -padx 4
                grid conf $QWIKMD::selresPatcheFrame -row 1 -column 0 -sticky nswe -pady 2
            }
        }
    }
    set tabid [expr [$QWIKMD::topGui.nbinput index current] +1]

    if {$QWIKMD::tabprev == -1} {
        set QWIKMD::tabprev 1
        set QWIKMD::tabprevmodf 1
    }
    
    if {$tabid <= 2 && $QWIKMD::tabprevmodf != $tabid} {
        QWIKMD::ChangeMdSmd $tabid
        $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb configure -state normal

        $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb delete 0 end
        
        set lines [$QWIKMD::topGui.nbinput.f$QWIKMD::tabprevmodf.tableframe.tb get 0 end]
        for {set i 0} {$i < [llength $lines]} {incr i} {
            $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb insert end [lindex $lines $i]
        }
        
        $QWIKMD::topGui.nbinput.f$QWIKMD::tabprevmodf.tableframe.tb delete 0 end
        $QWIKMD::topGui.nbinput.f$QWIKMD::tabprevmodf.tableframe.tb configure -state disable
        
        set QWIKMD::tabprevmodf $tabid
        QWIKMD::ChangeSolvent
    }
    if {$tabid <= 2} {
        set QWIKMD::runbtt $QWIKMD::topGui.nbinput.f$tabid.fcontrol.fcolapse.f1.run.button_Calculate
    }
    set QWIKMD::tabprev $tabid
    
}

proc QWIKMD::BuildRun {frame level} {
    set gridrow 0
    grid [ttk::frame $frame.fbtload] -row $gridrow -column 0 -sticky ewns -padx 2 -pady 4
    grid columnconfigure $frame.fbtload 1 -weight 1
    
    grid [ttk::button $frame.fbtload.btBrowser -text "Browser" -padding "2 0 2 0" -command {QWIKMD::BrowserButt}] -row 0 -column 0 -sticky w -padx 2

    QWIKMD::balloon $frame.fbtload.btBrowser [QWIKMD::pdbBrowserBL]
    ttk::style configure PdbEntrey.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::entry  $frame.fbtload.entLoad -textvariable QWIKMD::inputstrct -style PdbEntrey.TEntry -validate focus -validatecommand {
        if {[%W get] == "PDB ID"} {
            %W delete 0 end
            ttk::style configure PdbEntrey.TEntry -foreground black
        } elseif {[%W get] == ""} {
            ttk::style configure PdbEntrey.TEntry -foreground $QWIKMD::tempEntry
            set QWIKMD::inputstrct "PDB ID"
        }
        return 1
        }] -row 0 -column 1 -sticky we
    set QWIKMD::inputstrct "PDB ID"
    QWIKMD::balloon $frame.fbtload.entLoad [QWIKMD::pdbentryLoadBL]

    grid [ttk::button  $frame.fbtload.btLoad -text "Load" -padding "2 0 2 0" -command {
        global env
        if {$QWIKMD::inputstrct != "" && $QWIKMD::inputstrct != "PDB ID"} {
            set file $QWIKMD::inputstrct
            set tab [$QWIKMD::topGui.nbinput index current]
            if {[QWIKMD::resetBtt 2] == 1} {
                return
            }
            $QWIKMD::topGui.nbinput select $tab
            display update off
            $QWIKMD::topGui configure -cursor watch; update 
            
            set numtabs [llength [$QWIKMD::topGui.nbinput tabs]]
            for {set i 0} {$i < $numtabs} {incr i} {
                if {$tab != $i} {
                    $QWIKMD::topGui.nbinput tab $i -state disabled
                }
            }
    
            set QWIKMD::inputstrct $file
            ttk::style configure PdbEntrey.TEntry -foreground black
            QWIKMD::LoadButt $QWIKMD::inputstrct

            #Check for residues with multiple insertion codes and renumber them sequentially
            set selchain [atomselect $QWIKMD::topMol "all and not water and not ions"]

            set chainList [$selchain get chain]
            set chainList [lsort -unique $chainList]
            $selchain delete
            set renumber [list]
            foreach chain $chainList {
                set sel [atomselect $QWIKMD::topMol "chain \"$chain\""]
                set insertion [$sel get insertion]
                set listsort [lsort -unique $insertion]
                if {$listsort != "{ }" && [llength $listsort] > 1} {
                    # set resids [lsort -unique -integer [$sel get resid]]
                    set prevres ""
                    set txtini ""
                    set res [list]
                    set newresid [list]
                    set previnsert ""
                    set minres ""
                    set prevfrag ""
                    
                    foreach residaux [$sel get resid] insert [$sel get insertion] frag [$sel get fragment] {
                        set txt "$residaux $insert"
                        if {$minres == ""} {
                            set minres $residaux
                            set prevfrag $frag
                        }
                        if {$txt != $txtini} {
                            lappend res [atomselect $QWIKMD::topMol "chain \"$chain\" and resid \"$residaux\" and insertion \"$insert\" "]
                            if {$prevres != ""} {
                                set increment [expr $residaux - $prevres]
                                if {$increment == 0} {
                                    set increment 1
                                }
                                
                                if {$prevfrag != $frag && $increment == 1} {
                                    #set selaux [atomselect $QWIKMD::topMol "(within 2.0 of (chain \"$chain\" and resid \"$prevres\" and insertion \"$previnsert\")) and (chain \"$chain\" and resid \"$residaux\" and insertion \"$insert\")"]
                                    set increment 2
                                    #$selaux delete
                                }
                                set newresidaux [expr $increment + [lindex $newresid end]]
                                lappend renumber [list ${residaux}_${chain}_$insert ${newresidaux}_$chain]
                                lappend newresid $newresidaux
                            } else {
                                lappend newresid $minres
                            }
                            set txtini $txt
                            set prevres $residaux
                            set previnsert $insert
                            set prevfrag $frag
                        }   
                    }
                    set i 0
                    foreach selaux $res {
                        if {$i != 0} {
                            $selaux set resid [lindex $newresid $i]
                        }
                        $selaux delete
                        incr i
                    }

                }
                $sel delete 
            }
            update
            QWIKMD::UpdateMolTypes $QWIKMD::tabprevmodf
            QWIKMD::checkStructur init
            if {[llength $renumber] > 0} {
                set renumbfile [open "$env(TMPDIR)/Renumber_Residues.txt" w+]
                set w1 14
                set w2 9
                set w3 15
                set sep +-[string repeat - $w1]-+-[string repeat - $w2]-+-[string repeat - $w3]-+-[string repeat - $w1]-+-[string repeat - $w2]-+
                puts $renumbfile $sep
                puts $renumbfile [format "| %*s | %*s | %*s | %*s | %*s |" $w1 "Init Resid" $w2 "Chain" $w3 "Insert Code" $w1 "New Resid" $w2 "Chain"]
                puts $renumbfile $sep
                set chains ""
                foreach txt $renumber {
                    set initresid [split [lindex $txt 0] "_"]
                    set insert [lindex $initresid 2]
                    # if {$insert == " "} {
                    #   set insert  "{}"
                    # }
                    set finalresid [split [lindex $txt 1] "_"]
                    puts $renumbfile [format "| %*s | %*s | %*s | %*s | %*s |" $w1 "[lindex $initresid 0]" $w2 "[lindex $initresid 1]" $w3 "$insert" $w1 "[lindex $finalresid 0]" $w2 "[lindex $finalresid 1]"]
                    if {[string first [lindex $initresid 1] $chains] == -1} {
                        append chains "[lindex $initresid 1] "
                    }
                }
                puts $renumbfile $sep
                close $renumbfile
                tk_messageBox -message "One or more different insertion codes were found for the chain(s): $chains.\n\
                The renumbering table will be shown after press \"OK\" and will be saved in the working directory after preparation as \"Renumber_Residues.txt\"." -icon warning -title "Residues Renumbering" -type ok
                set instancehandle [multitext]
                $instancehandle openfile "$env(TMPDIR)/Renumber_Residues.txt"
            }
            for {set i 0} {$i < $numtabs} {incr i} {
                if {$tab != $i} {
                    $QWIKMD::topGui.nbinput tab $i -state normal
                }
            }
            $QWIKMD::topGui configure -cursor {}; update
            QWIKMD::changeBCK
            display update on
            update
        }
        
    }] -row 0 -column 2 -sticky w -padx 2

    QWIKMD::balloon $frame.fbtload.btLoad [QWIKMD::pdbLoadBL]
    #Selection frame
    incr gridrow
    grid [ttk::frame $frame.selframe] -row $gridrow -column 0 -sticky nwe
    grid columnconfigure $frame.selframe 0 -weight 1
    grid columnconfigure $frame.selframe 1 -weight 1
    grid columnconfigure $frame.selframe 2 -weight 1
    
    ttk::menubutton $frame.selframe.mNMR -text "NMR State" -menu $frame.selframe.mNMR.nmr
    ttk::menubutton $frame.selframe.mCHAIN -text "Chain/Type Selection" -menu $frame.selframe.mCHAIN.chain
    
    if {$level == "basic"} {
        menu $frame.selframe.mNMR.nmr -tearoff 0
        menu $frame.selframe.mCHAIN.chain -tearoff 0
    } else {
        $QWIKMD::topGui.nbinput.f1.selframe.mNMR.nmr clone $frame.selframe.mNMR.nmr
        $QWIKMD::topGui.nbinput.f1.selframe.mCHAIN.chain clone $frame.selframe.mCHAIN.chain
    }
    grid $frame.selframe.mNMR -row 0 -column 0 -sticky nwe -pady 4
    grid $frame.selframe.mCHAIN -row 0 -column 1 -sticky nwe -pady 4

    set QWIKMD::nmrMenu($level) $frame.selframe.mNMR
    set QWIKMD::chainMenu($level) $frame.selframe.mCHAIN

    grid [ttk::button $frame.selframe.mRESID -text "Structure Manipulation" -command {
        QWIKMD::callStrctManipulationWindow
        wm title $QWIKMD::selResGui "Structure Manipulation\/Check" 
        if {$QWIKMD::prepared != 1 && $QWIKMD::load != 1} {
            QWIKMD::lockSelResid 1
        }

    } ]  -row 0 -column 2 -sticky nwe -pady 4

    QWIKMD::balloon $frame.selframe.mNMR [QWIKMD::nmrBL]
    QWIKMD::balloon $frame.selframe.mCHAIN [QWIKMD::addChainBL]
    QWIKMD::balloon $frame.selframe.mRESID [QWIKMD::selResidBL]

    QWIKMD::createInfoButton $frame.selframe 0 4
    bind $frame.selframe.info <Button-1> {
        set val [QWIKMD::selectInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow info [lindex $val 0] [lindex $val 2]
    }
    incr gridrow
    #tablelist selection
    grid [ttk::frame $frame.tableframe] -row $gridrow -column 0 -sticky nwse -padx 4

    grid columnconfigure $frame.tableframe 0 -weight 1
    grid rowconfigure $frame.tableframe 0 -weight 1
    set fro2 $frame.tableframe
    option add *Tablelist.       frame
    option add *Tablelist.background        gray98
    option add *Tablelist.stripeBackground  #e0e8f0
    option add *Tablelist.setGrid           yes
    option add *Tablelist.movableColumns    no


        tablelist::tablelist $fro2.tb \
        -columns { 0 "Chain"     center
                0 "Residue Range"    center
                0 "Type" center
                0 "Representation" center
                0 "Color" center 
                } -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] -showseparators 0 -labelrelief groove  -labelbd 1 -selectbackground white \
                -selectforeground black -foreground black -background white -state normal -selectmode single -stretch "all" -stripebackgroun white -height 5\
                -editstartcommand {QWIKMD::mainTableCombosStart 1} -editendcommand QWIKMD::mainTableCombosEnd -forceeditendcommand true

    $fro2.tb columnconfigure 0 -width 0 -sortmode dictionary -name Chain
    $fro2.tb columnconfigure 1 -width 0 -sortmode dictionary -name Range
    $fro2.tb columnconfigure 2 -width 0 -sortmode dictionary -name type
    $fro2.tb columnconfigure 3 -width 0 -sortmode dictionary -name Representation -editable true -editwindow ttk::combobox
    $fro2.tb columnconfigure 4 -width 0 -sortmode dictionary -name Color -editable true -editwindow ttk::combobox


    grid $fro2.tb -row 0 -column 0 -sticky news
    grid columnconfigure $fro2.tb 0 -weight 1; grid rowconfigure $fro2.tb 0 -weight 1

    ##Scrool_BAr V
    scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
     grid $fro2.scr1 -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
    grid $fro2.scr2 -row 1 -column 0 -sticky swe

    bind [$fro2.tb labeltag] <Any-Enter> {
        set col [tablelist::getTablelistColumn %W]
        set help 0
        switch $col {
            0 {
                set help [QWIKMD::selTabelChainBL]
            }
            1 {
                set help [QWIKMD::selTabelResidBL]
            }
            2 {
                set help [QWIKMD::selTabelTypeBL]
            }
            3 {
                set help [QWIKMD::selTabelRepBL]
            }
            4 {
                set help [QWIKMD::selTabelColorBL]
            }
            default {
                set help $col
            }
        }
        after 1000 [list QWIKMD::balloon:show %W $help]
  
    }
    bind [$fro2.tb labeltag] <Any-Leave> "destroy %W.balloon"
    
    $fro2.tb configure -state disable
    
    incr gridrow
    grid [ttk::frame $frame.changeBack] -row $gridrow -column 0 -pady 2 -padx 2 -sticky we

    grid columnconfigure $frame.changeBack 0 -weight 1
    grid columnconfigure $frame.changeBack 1 -weight 1
    grid columnconfigure $frame.changeBack 2 -weight 1
    grid columnconfigure $frame.changeBack 3 -weight 1
    grid columnconfigure $frame.changeBack 4 -weight 1
    grid columnconfigure $frame.changeBack 5 -weight 1
    grid [ttk::label $frame.changeBack.lbcheck -text "Background"] -row 0 -column 0 -padx 1 -sticky w

    grid [ttk::radiobutton $frame.changeBack.checkBlack -text "Black" -variable QWIKMD::basicGui(desktop) -value "black" -command QWIKMD::changeBCK] -row 0 -column 1 -padx 1 -sticky w
    grid [ttk::radiobutton $frame.changeBack.checkWhite -text "White" -variable QWIKMD::basicGui(desktop) -value "white" -command QWIKMD::changeBCK] -row 0 -column 2 -padx 1 -sticky w
    grid [ttk::radiobutton $frame.changeBack.checkGradient -text "Gradient" -variable QWIKMD::basicGui(desktop) -value "gradient" -command QWIKMD::changeBCK] -row 0 -column 3 -padx 1 -sticky w

    set QWIKMD::basicGui(desktop) ""
    QWIKMD::balloon $frame.changeBack.checkBlack [QWIKMD::cbckgBlack]
    QWIKMD::balloon $frame.changeBack.checkWhite [QWIKMD::cbckgWhite]
    QWIKMD::balloon $frame.changeBack.checkGradient [QWIKMD::cbckgGradient]

    grid [ttk::label $frame.changeBack.lscheme -text "Color Scheme:"] -row 0 -column 4 -padx 2
    grid [ttk::combobox $frame.changeBack.comboscheme -values {"VMD Classic"} -textvariable QWIKMD::basicGui(scheme) -width 11 -state readonly] -row 0 -column 5 -padx 2 -sticky w
    set QWIKMD::basicGui(scheme) "VMD Classic"
    bind $frame.changeBack.comboscheme <<ComboboxSelected>> {
        %W selection clear
    }
    ## Protocol NoteBook
    incr gridrow
    grid [ttk::notebook $frame.nb -padding "1 8 1 1"] -row $gridrow -column 0 -sticky news -padx 0
    lappend QWIKMD::notebooks "$frame.nb"
    grid columnconfigure $frame.nb 0 -weight 1
    if {$level == "basic"} {
        
        ttk::frame $frame.nb.f1
        grid columnconfigure $frame.nb.f1 0 -weight 1
        grid rowconfigure $frame.nb.f1 0 -weight 1
        ttk::frame $frame.nb.f2
        grid columnconfigure $frame.nb.f2 0 -weight 1
        grid rowconfigure $frame.nb.f2 0 -weight 1

        $frame.nb add $frame.nb.f1 -text "Molecular Dynamics" -sticky new
        $frame.nb add $frame.nb.f2 -text "Steered Molecular Dynamics"  -sticky new
        
        ## Frame MD
        QWIKMD::system $frame.nb.f1 $level "MD"
        QWIKMD::protocolBasic $frame.nb.f1 "MD"

        
        #Frame SM

        #QWIKMD::notebook 
        QWIKMD::system $frame.nb.f2 $level "SMD"
        QWIKMD::protocolBasic $frame.nb.f2 "SMD"

        ##############################################
        ## hide MD options in the Run tab by default
        ##############################################
        bind $frame.nb <<NotebookTabChanged>> {QWIKMD::ChangeMdSmd [expr [$QWIKMD::topGui.nbinput index current] +1] }
        

    } else {
        set tab 1

        #Notebook tab for MD
        ttk::frame $frame.nb.f$tab
        grid columnconfigure $frame.nb.f$tab 0 -weight 1
        grid rowconfigure $frame.nb.f$tab 0 -weight 0
        grid rowconfigure $frame.nb.f$tab 1 -weight 1
        $frame.nb add $frame.nb.f$tab -text "MD" -sticky news

        QWIKMD::system $frame.nb.f$tab $level "MD"
        QWIKMD::protocolAdvanced $frame.nb.f$tab "MD"

        #Notebook tab for SMD
        incr tab
        ttk::frame $frame.nb.f$tab
        grid columnconfigure $frame.nb.f$tab 0 -weight 1
        grid rowconfigure $frame.nb.f$tab 0 -weight 0
        grid rowconfigure $frame.nb.f$tab 1 -weight 1
        $frame.nb add $frame.nb.f$tab -text "SMD"  -sticky news

        QWIKMD::system $frame.nb.f$tab $level "SMD"
        QWIKMD::protocolAdvanced $frame.nb.f$tab "SMD"
        
        #Notebook tab for MDFF
        incr tab
        ttk::frame $frame.nb.f$tab
        grid columnconfigure $frame.nb.f$tab 0 -weight 1
        grid rowconfigure $frame.nb.f$tab 0 -weight 0
        grid rowconfigure $frame.nb.f$tab 1 -weight 1
        $frame.nb add $frame.nb.f$tab -text "MDFF" -sticky news

        QWIKMD::system $frame.nb.f$tab $level "MDFF"
        QWIKMD::protocolBasic $frame.nb.f$tab "MDFF"

        ##############################################
        ## hide MD options in the Run tab by default
        ##############################################
        bind $frame.nb <<NotebookTabChanged>> {QWIKMD::ChangeMdSmd [expr [$QWIKMD::topGui.nbinput index current] +1]}
        
    }
    
    ###Simulation Setup
    incr gridrow
    grid [ttk::frame $frame.fb] -row $gridrow -column 0 -sticky news -pady 2
    grid columnconfigure $frame.fb 0 -weight 1
    grid rowconfigure $frame.fb 0 -weight 0
    grid rowconfigure $frame.fb 1 -weight 0
    grid rowconfigure $frame.fb 1 -weight 1

    grid [ttk::label $frame.fb.prt -text "$QWIKMD::downPoint Simulation Setup"] -row 0 -column 0 -sticky w -pady 1
    #QWIKMD::ChangeSolvent
    bind $frame.fb.prt <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Simulation Setup"
    }
    
    grid [ttk::frame $frame.fb.fcolapse] -row 1 -column 0 -sticky nsew -pady 5
    grid columnconfigure $frame.fb.fcolapse 0 -weight 1

    QWIKMD::createInfoButton $frame.fb 0 0
    bind $frame.fb.info <Button-1> {
        set val [QWIKMD::outputBrowserinfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow outputBrowserinfo [lindex $val 0] [lindex $val 2]
    }

    set framecolapse $frame.fb.fcolapse

    grid [ttk::frame $framecolapse.sep ] -row 1 -column 0 -sticky ew
    grid columnconfigure $framecolapse.sep 0 -weight 1
    grid [ttk::separator $framecolapse.spt -orient horizontal] -row 0 -column 0 -sticky ew -pady 0

    grid [ttk::frame $framecolapse.f1] -row 1 -column 0 -padx 2 -sticky ew
    grid columnconfigure $framecolapse.f1 0 -weight 1



    set framesetup $framecolapse.f1

    grid [ttk::frame $framesetup.fwork] -row 0 -column 0 -pady 5 -padx 2 -sticky nsew
    grid columnconfigure $framesetup.fwork 0 -weight 1
    ttk::style configure WorkDir.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::entry $framesetup.fwork.outentrey -textvariable QWIKMD::basicGui(workdir,0) -style WorkDir.TEntry -validate focus -validatecommand {
        if {[%W get] == "Working Directory"} {
            %W delete 0 end
            ttk::style configure WorkDir.TEntry -foreground black
        } elseif {[%W get] == ""} {
            ttk::style configure WorkDir.TEntry -foreground $QWIKMD::tempEntry
            set QWIKMD::basicGui(workdir,0) "Working Directory"
        }
        return 1
        }] -row 0 -column 0 -sticky ew -padx 2

    set QWIKMD::basicGui(workdir,0) "Working Directory"
    if {$level == "basic"} {
        set QWIKMD::basicGui(workdir,1) $framesetup.fwork.outentrey
    } else {
        set QWIKMD::basicGui(workdir,2) $framesetup.fwork.outentrey
    }
    

    grid [ttk::button $framesetup.fwork.outload -text "Load" -command {
            set extension ".qwikmd"
            set types {
                {{QwikMD}       {".qwikmd"}        }
                {{All}       {"*"}        }
            }
            set fil ""
            set fil [tk_getOpenFile -title "Open InputFile" -filetypes $types -defaultextension $extension]
            if {$fil != ""} {
                display update off
                QWIKMD::resetBtt 1
                set QWIKMD::basicGui(workdir,0) ${fil}

                #Make compatible with vmd1.9.3b1 versions input file
                set file [open $QWIKMD::basicGui(workdir,0) r]
                set lines [split [read $file] "\n"]
                close $file
                if {[lindex $lines [expr [llength $lines] -2 ] ]== "QWIKMD::SelResid"} {
                    file copy -force $QWIKMD::basicGui(workdir,0) $QWIKMD::basicGui(workdir,0)_bkup
                    set i 2
                    while {$i < [llength $lines]} {
                        if {[string range [lindex $lines $i] 0 25] == "array set QWIKMD::basicGui" || [string range [lindex $lines $i] 0 23] == "array set QWIKMD::advGui"} {
                            set values [lindex [lindex $lines $i] 3 ]
                            set valuesAux "\{"
                            for {set j 1} {$j < [llength $values]} {incr j 2} {
                                set find [regexp {.qwikmd*} [join [lindex $values $j]]]
                                if {$find == 0 } {
                                    append valuesAux " [lrange $values [expr $j -1] $j]"
                                }
                            }
                            
                            if {[string range [lindex $lines $i] 0 25] == "array set QWIKMD::basicGui"} {
                                lset lines $i "[string range [lindex $lines $i] 0 25] $valuesAux\}"
                            } else {
                                lset lines $i "[string range [lindex $lines $i] 0 23] $valuesAux\}"
                                
                            }

                        }
                        if {[string range [lindex $lines $i] 0 6] == "set aux"} {
                            lset lines $i "set aux \"\[file rootname $QWIKMD::basicGui(workdir,0)\]\""
                        }
                        if {[string range [lindex $lines $i] 0 18] == "QWIKMD::ChangeMdSmd"} {
                            lset lines $i "#[lindex $lines $i]"
                        }
                        if {[string range [lindex $lines $i] 0 19] == "set QWIKMD::confFile"} {
                            lappend lines "set QWIKMD::prevconfFile [string range [lindex $lines $i] 20 end]"
                        }
                        if {[string trimleft [string range [lindex $lines $i] 0 12]] == "mol addfile"} {
                            lset lines $i "#[lindex $lines $i]"
                        }
                        incr i
                    }
                    lset lines [expr [llength $lines] -3] "#[lindex $lines [expr [llength $lines] -3]]"
                    lset lines [expr [llength $lines] -4] "#[lindex $lines [expr [llength $lines] -4]]"
                    lset lines [expr [llength $lines] -5] "#[lindex $lines [expr [llength $lines] -5]]"
                    set file [open $QWIKMD::basicGui(workdir,0) w+]
                    foreach line $lines {
                        puts $file $line
                    }
                    
                    close $file
                }

                source $QWIKMD::basicGui(workdir,0)
                if {[catch {glob ${QWIKMD::outPath}/run/*.dcd} listprot] == 0 && $QWIKMD::run != "MDFF"} {
                    set tabid [$QWIKMD::topGui.nbinput index current]
                    $QWIKMD::topGui.nbinput tab 0 -state disable
                    $QWIKMD::topGui.nbinput tab 1 -state disable
                    $QWIKMD::topGui.nbinput tab 2 -state disable
                    $QWIKMD::topGui.nbinput tab 3 -state disable
                    QWIKMD::LoadOptBuild $tabid
                    $QWIKMD::topGui.nbinput tab 0 -state normal
                    $QWIKMD::topGui.nbinput tab 1 -state normal
                    $QWIKMD::topGui.nbinput tab 2 -state normal
                    $QWIKMD::topGui.nbinput tab 3 -state normal
                    $QWIKMD::topGui.nbinput select $tabid
                    if {$QWIKMD::loadprotlist == "Cancel"} {
                        QWIKMD::resetBtt 2
                        return
                    }
                    set seltext "all"
                    set sufix ""
                    set docatdcd 0
                    if {$QWIKMD::loadremovewater == 1 } {
                        append seltext " and not water "
                        set sufix "_nowater"
                        set docatdcd 1
                    }
                    if {$QWIKMD::loadremoveions == 1} {
                        append seltext " and not (ions not within 5 of protein)" 
                        append sufix "_noions"
                        set docatdcd 1

                    }
                    if {$QWIKMD::loadremovehydrogen == 1} {
                        append seltext " and noh" 
                        append sufix "_noh"
                        set docatdcd 1

                    }
                    set newloadprotlist [list]
                    set psf [file root [lindex $QWIKMD::inputstrct 0]].psf
                    set pdb [file root [lindex $QWIKMD::inputstrct 0]].pdb
                    if {$docatdcd == 1} {
                        
                        set psf [file root [lindex $QWIKMD::inputstrct 0]]$sufix.psf
                        set pdb [file root [lindex $QWIKMD::inputstrct 0]]$sufix.pdb
                        if {[file exists [file root [lindex $QWIKMD::inputstrct 0]]$sufix.psf] == 0} {
                            set sel [atomselect $QWIKMD::topMol $seltext frame 0] 
                            $sel writepsf $psf
                            $sel writepdb $pdb
                            set indexfile [open "catdcd_index.txt" w+]
                            puts $indexfile [$sel get index]
                            close $indexfile
                            $sel delete
                        }
                        set warning 0
                        for {set i 0} {$i < [llength $QWIKMD::loadprotlist]} {incr i} {
                            set indcd [lindex $QWIKMD::loadprotlist $i].dcd
                            #if {[file exists $indcd] == 1} {
                            set outcd [lindex $QWIKMD::loadprotlist $i]$sufix.dcd
                            if {[file exists $outcd] == 0} {
                                if {$warning == 0} {
                                    set answer [tk_messageBox -message "Save trajectories of a subset of atoms may take some time.\nThis process only happens once for the same subset of atoms. Do you want to continue?" -type yesno -title "Load trajectory"]
                                    if {$answer == "no"} {
                                        mol delete $QWIKMD::topMol
                                        return
                                    }
                                    set warning 1
                                }
                                catch {eval "exec catdcd -i catdcd_index.txt -o $outcd $indcd"} out
                            }
                            lappend newloadprotlist [lindex $QWIKMD::loadprotlist $i]$sufix
                            #}
                        }
                    }
                    if {$QWIKMD::loadinitialstruct == 0 || $docatdcd == 1} {
                        mol delete $QWIKMD::topMol
                        set inputstrct [list $psf]
                        if {$QWIKMD::loadinitialstruct == 1} {
                            lappend inputstrct $pdb
                        }
                        set QWIKMD::inputstrct $inputstrct
                        QWIKMD::LoadButt $QWIKMD::inputstrct
                    }
                    set dcdlist [list]
                    if {$docatdcd == 1} {
                        set dcdlist $newloadprotlist
                        if {[winfo exists $QWIKMD::advGui(analyze,advance,interradio)] == 1 && ($QWIKMD::loadremovehydrogen == 1 || $QWIKMD::loadremovewater == 1)} {
                            $QWIKMD::advGui(analyze,advance,interradio) configure -state disable
                        }
                    } else {
                        set dcdlist $QWIKMD::loadprotlist
                    }
                    set QWIKMD::lastframe [list]
                    if {[molinfo $QWIKMD::topMol get numframes] > 1} {
                        animate delete beg 1 end [molinfo $QWIKMD::topMol get numframes] skip 0 $QWIKMD::topMol
                    }
                    for {set i 0} {$i < [llength $dcdlist]} {incr i} {
                        if {[file exists [lindex $dcdlist $i].dcd] == 1} {
                            mol addfile [lindex $dcdlist $i].dcd step $QWIKMD::loadstride waitfor all
                            lappend QWIKMD::lastframe [molinfo $QWIKMD::topMol get numframes]
                        }
                    }

                    update
                    set QWIKMD::confFile $QWIKMD::loadprotlist
                    if {$tabid == 0} {
                        set solvent $QWIKMD::basicGui(solvent,0)
                    } else {
                        set solvent $QWIKMD::advGui(solvent,0)
                    }
                    if {$solvent == "Explicit"} {
                        pbc box -center bb -color yellow -width 4
                        set QWIKMD::pbcInfo [pbc get -last end -nocheck]
                    }
                    if {$tabid == 1 && $QWIKMD::run != "MDFF"} {
                        set i 0
                        foreach prtcl $QWIKMD::prevconfFile {
                            if {[lsearch $QWIKMD::confFile $prtcl] == -1} {
                                $QWIKMD::advGui(protocoltb,$QWIKMD::run) rowconfigure $i -foreground grey
                            } else {
                                $QWIKMD::advGui(protocoltb,$QWIKMD::run) rowconfigure $i -foreground black
                            }
                            incr i
                        }
                    }
                }

                if {$QWIKMD::prepared == 1} {
                    $QWIKMD::runbtt configure -state normal
                }
                $QWIKMD::runbtt configure -text "Start [QWIKMD::RunText]"
                set numframes [molinfo $QWIKMD::topMol get numframes]
                QWIKMD::updateTime load
                if {$QWIKMD::prepared == 0} {
                    array set chainsAux [array get QWIKMD::chains]
                    array set index_cmbAux [array get QWIKMD::index_cmb]
                }
                QWIKMD::mainTable [expr [$QWIKMD::topGui.nbinput index current] +1]
                if {$QWIKMD::prepared == 0} {
                    array set QWIKMD::chains [array get chainsAux]
                    array set QWIKMD::index_cmb [array get index_cmbAux]
                }
                QWIKMD::reviewTable [expr [$QWIKMD::topGui.nbinput index current] +1]
                QWIKMD::SelResid
                QWIKMD::ChangeSolvent
                if {$QWIKMD::prepared == 1} {
                    QWIKMD::lockGUI 
                    if {[file exists "$QWIKMD::outPath/[file tail $QWIKMD::outPath].infoMD"] != 1} {
                        tk_messageBox -title "Missing log File" -message "Text log file not found" -icon warning
                        if {[file exists ${QWIKMD::outPath}] == 1} {
                            set QWIKMD::textLogfile [open "$QWIKMD::outPath/[file tail $QWIKMD::outPath].infoMD" a+]
                        }
                    } else {
                        set QWIKMD::textLogfile [open "$QWIKMD::outPath/[file tail $QWIKMD::outPath].infoMD" a+]
                        puts $QWIKMD::textLogfile [QWIKMD::loadDCD]
                        flush $QWIKMD::textLogfile
                    }
                }
                display update on
                if {$QWIKMD::run == "MDFF"} {
                    QWIKMD::updateMDFF
                }
            }
            
        } -padding "2 0 2 0"] -row 0 -column 1 -pady 5 -padx 2 -sticky w

        proc saveBut {} {
            set extension ".qwikmd"
            set types {
                {{QwikMD}       {".qwikmd"}        }
            }
            
            set fil [list]
            set fil [tk_getSaveFile -title "Save InputFile" -filetypes $types -defaultextension $extension]
            if {$fil != ""} {
                if {[string first " " [file tail [file root $fil] ] ] >= 0} {
                    tk_messageBox -message "Make sure that space characters are not included in the name of the file" -icon warning -type ok
                    return
                }
                if {[string range ${fil} [expr [string length ${fil}] -7] end ] != ".qwikmd"} {
                    set fil [append fil ".qwikmd"]  
                }
                set QWIKMD::basicGui(workdir,0) $fil
                QWIKMD::SaveInputFile $QWIKMD::basicGui(workdir,0)
            }
        }
    grid [ttk::button $framesetup.fwork.outsave -text "Save" -command [namespace current]::saveBut -padding "2 0 2 0"] -row 0 -column 2 -pady 5 -padx 2 -sticky w

    grid [ttk::frame $framesetup.preparereset] -row 1 -column 0 -pady 1 -sticky ew
    grid columnconfigure $framesetup.preparereset 0 -weight 0
    grid columnconfigure $framesetup.preparereset 1 -weight 1
    grid columnconfigure $framesetup.preparereset 2 -weight 1

    grid [ttk::button $framesetup.preparereset.button_Prepare -text "Prepare" -padding "4 2 4 2"  -command {
        
        if {$QWIKMD::basicGui(workdir,0) == "Working Directory"} {
            set QWIKMD::basicGui(workdir,0) ""
        }
        
        if {[QWIKMD::PrepareBttProc $QWIKMD::basicGui(workdir,0)]  == 0} {
            QWIKMD::changeBCK
            
            QWIKMD::lockGUI
        }
    
    }] -row 0 -column 0 -pady 1 -padx 4 -sticky w

    if {$level == "basic"} {
        set QWIKMD::basicGui(preparebtt,0) $framesetup.preparereset.button_Prepare
    } else {
        set QWIKMD::basicGui(preparebtt,1) $framesetup.preparereset.button_Prepare
    }
    
    grid [ttk::checkbutton $framesetup.preparereset.live -text "Live View" -variable QWIKMD::basicGui(live)] -row 0 -column 1 -padx 2 -sticky w
    set QWIKMD::basicGui(live) 0
    grid [ttk::button $framesetup.preparereset.button_Reset -text "Reset" -padding "4 2 4 2" -command {
        display update off
        QWIKMD::resetBtt 2
        display update on
        display update ui
        update
        }
    ] -row 0 -column 2 -pady 1 -padx 4 -sticky e

    
    QWIKMD::balloon $framesetup.preparereset.button_Prepare [QWIKMD::prepareBL]
    QWIKMD::balloon $framesetup.preparereset.live [QWIKMD::liveSimulBL]
    QWIKMD::balloon $framesetup.preparereset.button_Reset [QWIKMD::resetBL]


    incr gridrow
    ###Simulation Controls
    grid [ttk::separator $frame.spt -orient horizontal] -row $gridrow -column 0 -sticky ew -pady 2
    incr gridrow
    grid [ttk::frame $frame.fcontrol] -row $gridrow -column 0 -sticky news -pady 0
    grid columnconfigure $frame.fcontrol 0 -weight 1
    grid rowconfigure $frame.fcontrol 0 -weight 1
    grid rowconfigure $frame.fcontrol 1 -weight 1
    ## buttons Exit and Calculate

    grid [ttk::label $frame.fcontrol.prt -text "$QWIKMD::downPoint Simulation Control" ] -row 0 -column 0 -sticky w -pady 0

    bind $frame.fcontrol.prt <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Simulation Control"
    }

    QWIKMD::createInfoButton $frame.fcontrol 0 0

    bind $frame.fcontrol.info <Button-1> {
        set val [QWIKMD::MDControlsinfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow mdControlsinfo [lindex $val 0] [lindex $val 2]
    }

    

    grid [ttk::frame $frame.fcontrol.fcolapse] -row 1 -column 0 -padx 2 -sticky ew
    grid columnconfigure $frame.fcontrol.fcolapse 0 -weight 1

    grid [ttk::frame $frame.fcontrol.fcolapse.f1] -row 1 -column 0 -padx 2 -sticky ew
    grid columnconfigure $frame.fcontrol.fcolapse.f1 0 -weight 1

    set framecontrol $frame.fcontrol.fcolapse.f1


    grid [ttk::frame $framecontrol.run] -row 2 -column 0 -pady 1 -padx 2 -sticky ew
    grid columnconfigure $framecontrol.run 0 -weight 1

    grid [ttk::button $framecontrol.run.button_Calculate -text "Start MD Simulation" -padding "2 2 2 2" -command {
        if {$QWIKMD::prepared == 1} {
            if {$QWIKMD::run == "MDFF"} {
                QWIKMD::updateMDFF
            } else {
                QWIKMD::Run
            }
        } else {
            tk_messageBox -message "Please select and edit your structure and then press \"Prepare\" button." -title "Running Simulation" -icon info -type ok
        }
        
    } ] -row 0 -column 0 -pady 1 -padx 2 -sticky ew
    set QWIKMD::runbtt $framecontrol.run.button_Calculate
    QWIKMD::balloon $framecontrol.run.button_Calculate  [QWIKMD::runbuttBL]
    
    grid [ttk::frame $framecontrol.imd] -row 3 -column 0 -pady 1 -padx 2 -sticky ew
    grid columnconfigure $framecontrol.imd 0 -weight 1
    grid columnconfigure $framecontrol.imd 1 -weight 1
    grid columnconfigure $framecontrol.imd 2 -weight 1
    grid [ttk::button $framecontrol.imd.button_Detach -text "Detach" -padding "4 2 4 2"  -command {QWIKMD::Detach}] -row 0 -column 1 -pady 1 -padx 2 -sticky ew

    grid [ttk::button $framecontrol.imd.button_Pause -text "Pause" -padding "4 2 4 2" -command {QWIKMD::Pause}] -row 0 -column 0 -pady 1 -padx 2 -sticky ew
    grid [ttk::button $framecontrol.imd.button_Finish -text "Finish" -padding "4 2 4 2" -command {QWIKMD::Finish}] -row 0 -column 2 -pady 1 -padx 2 -sticky ew

    QWIKMD::balloon $framecontrol.imd.button_Detach  [QWIKMD::detachBL]
    QWIKMD::balloon $framecontrol.imd.button_Pause  [QWIKMD::pauseBL]
    QWIKMD::balloon $framecontrol.imd.button_Finish  [QWIKMD::finishBL]

    
    
    grid [ttk::separator $framecontrol.spt -orient horizontal] -row 4 -column 0 -sticky ew -pady 2

    grid [ttk::frame $framecontrol.progress ] -row 5 -column 0 -pady 2 -sticky nsew
    grid columnconfigure $framecontrol.progress 1 -weight 1

    grid [ttk::label $framecontrol.progress.label -text "Progress"] -column 0 -row 0 -sticky w -padx 2 -pady 2

    grid [ttk::progressbar $framecontrol.progress.pg -mode determinate -variable QWIKMD::basicGui(mdPrec,0)] -column 1 -row 0 -sticky news -pady 0

    grid [ttk::label $framecontrol.progress.currentTimelb -textvariable QWIKMD::basicGui(currenttime,0)] -column 2 -row 0 -sticky w -padx 2 -pady 2

    set QWIKMD::basicGui(mdPrec,0) 0
    if {$level == "basic"} {
        set QWIKMD::basicGui(mdPrec,1) $framecontrol.progress.pg
        set QWIKMD::basicGui(currenttime,pgframe) $framecontrol.progress
    } else {
        $framecontrol.progress.currentTimelb configure -textvariable QWIKMD::basicGui(currenttime,1)
        set QWIKMD::basicGui(mdPrec,2) $framecontrol.progress.pg
        set QWIKMD::advGui(currenttime,pgframe) $framecontrol.progress
    }
    

    set QWIKMD::basicGui(currenttime) "Completed 0.000 of 0.000 ns"
    #########################################################
    ## Update the time displayed in the MD progress 
    ## section. When the qwikMD inputfile is load is
    ## necessary to incr -1 the MD step ($QWIKMD::state)
    ## because the success of the previous MD is only checked
    ## when the Start button is pressed 
    #########################################################
    proc updateTime {opt} {
        set tabid [$QWIKMD::topGui.nbinput index current]
        if {$QWIKMD::basicGui(live) == 0} {
            if {$tabid == 0} {
                grid forget $QWIKMD::basicGui(currenttime,pgframe)
            } else {
                grid forget $QWIKMD::advGui(currenttime,pgframe)
            }
        } else {
            set frame $QWIKMD::basicGui(currenttime,pgframe)
            if {$tabid == 1} {
                set frame $QWIKMD::advGui(currenttime,pgframe)
            }
            grid conf $frame -row 5 -column 0 -pady 2 -sticky nsew
            set index $QWIKMD::state
            if {$QWIKMD::state > 0} {
                set index [expr $QWIKMD::state -1]
            }
            set maxtime 0.0
            if {[llength $QWIKMD::maxSteps] > 0} {
                set maxtime [lindex $QWIKMD::maxSteps $index]
            } 
            set str "Simulation time: [format %.3f [expr 2e-6 * [expr $QWIKMD::counterts - $QWIKMD::prevcounterts] * 10] ]\
            of [format %.3f [expr 2e-6 * $maxtime ] ] ns"
            set QWIKMD::basicGui(currenttime,[$QWIKMD::topGui.nbinput index current]) $str
        }
    }
    
}

proc QWIKMD::callStrctManipulationWindow {} {
    if {[winfo exists $QWIKMD::selResGui] != 1} {
        QWIKMD::SelResidBuild
        QWIKMD::SelResid
    } else {
        QWIKMD::SelResidBuild
    }
    raise $QWIKMD::selResGui
    ###############################################################################################
    ## Initiate trace event when the Select Resid window is opend. This event detects 
    ## if a atom was selected in the OpenGl Window and represent it and select in the 
    ## residues table. 
    ## Note!! VMD breaks when the pick event is used and the New Cartoon representation is active.                                                          
    ###############################################################################################
    trace remove variable ::vmd_pick_event write QWIKMD::ResidueSelect
    trace variable ::vmd_pick_event w QWIKMD::ResidueSelect
    mouse mode pick
}

proc QWIKMD::system {frame level MD} { 
    grid [ttk::frame $frame.f1] -row 0 -column 0 -stick ew -pady 5
    grid columnconfigure $frame.f1 0 -weight 1
    #grid columnconfigure $frame.f1 1 -weight 1

    grid [ttk::frame $frame.f1.fsolv] -row 0 -column 0 -stick we -pady 5
    grid columnconfigure $frame.f1.fsolv 0 -weight 1
    grid columnconfigure $frame.f1.fsolv 1 -weight 1
    #grid columnconfigure $frame.f1.fsolv 2 -weight 1

    grid [ttk::frame $frame.f1.fsolv.soltype] -row 0 -column 0 -stick we -padx 3
    grid [ttk::label $frame.f1.fsolv.soltype.mSol -text "Solvent"] -row 0 -column 0 -pady 0 -sticky ns
    set values {"Implicit" "Explicit"}
    if {$level != "basic"} {
        set values {"Vacuum" "Implicit" "Explicit"}
    }
    ####Add variable QWIKMD::solvent
    grid [ttk::combobox $frame.f1.fsolv.soltype.combSolv -values $values -width 10 -state readonly -textvariable QWIKMD::basicGui(solvent,0)] -row 0 -column 1 -pady 0 -sticky ns
    QWIKMD::balloon $frame.f1.fsolv.soltype.combSolv [QWIKMD::solventBL]

    if {$level != "basic"} {
        if {0} {
            grid [ttk::frame $frame.f1.addMol] -row 0 -column 1 -stick w -pady 5 -padx 3
            grid columnconfigure $frame.f1.addMol 1 -weight 1
            grid [ttk::label $frame.f1.addMol.add -text "Add"] -row 0 -column 0 -stick news -pady 0
            grid [ttk::entry $frame.f1.addMol.addMentry -width 4 -justify right -textvariable QWIKMD::advGui(addmol)] -row 0 -column 1 -stick w -pady 0
            grid [ttk::label $frame.f1.addMol.addMLab -text "molecules of "] -row 0 -column 2 -stick news -pady 0
            grid [ttk::button $frame.f1.addMol.addMBut -text "Browser"] -row 0 -column 3 -stick news -pady 0
            set QWIKMD::advGui(addmol) "10"
        }
        set QWIKMD::advGui(solvent,boxbuffer) 15
        grid [ttk::frame $frame.f1.fsolv.boxsize] -row 0 -column 1 -stick we -padx 3
        grid columnconfigure $frame.f1.fsolv.boxsize 0 -weight 1

        grid [ttk::frame $frame.f1.fsolv.boxsize.minbox] -row 0 -column 0 -stick we -padx 2
        grid [ttk::checkbutton $frame.f1.fsolv.boxsize.minbox.chckminbox -text "Minimal Box" -variable QWIKMD::advGui(solvent,minimalbox) -command {
            if {$QWIKMD::advGui(solvent,minimalbox) == 1} {
                $QWIKMD::advGui(solvent,boxbuffer,$QWIKMD::run,entry) configure -values {12 13 14 15 16 17 18 19 20}
                if {$QWIKMD::advGui(solvent,boxbuffer) == 6} {
                    set QWIKMD::advGui(solvent,boxbuffer) 12
                }
            } else {
                $QWIKMD::advGui(solvent,boxbuffer,$QWIKMD::run,entry) configure -values {6 7 8 9 10 11 12 13 14 15}
                if {$QWIKMD::advGui(solvent,boxbuffer) > 15} {
                    set QWIKMD::advGui(solvent,boxbuffer) 15
                }
            }
        }] -row 0 -column 0 -stick ns
        set QWIKMD::advGui(solvent,minbox,$MD) $frame.f1.fsolv.boxsize.minbox.chckminbox
        set QWIKMD::advGui(solvent,minimalbox) 0

        QWIKMD::balloon $frame.f1.fsolv.boxsize.minbox.chckminbox [QWIKMD::minimalBox]
        
        grid [ttk::frame $frame.f1.fsolv.boxsize.buffer] -row 0 -column 1 -stick we -padx 2 -pady 0
        grid [ttk::label $frame.f1.fsolv.boxsize.buffer.add -text "Buffer:"] -row 0 -column 1 -stick ns -pady 0
        set values {6 7 8 9 10 11 12 13 14 15}
        
        grid [ttk::combobox $frame.f1.fsolv.boxsize.buffer.combval -values $values -state readonly -width 4 -textvariable QWIKMD::advGui(solvent,boxbuffer)] -row 0 -column 2 -sticky ns -padx 2
        grid [ttk::label $frame.f1.fsolv.boxsize.buffer.angs -text "A"] -row 0 -column 3 -stick ns -pady 0
                
        
        bind $frame.f1.fsolv.boxsize.buffer.combval <<ComboboxSelected>> {
            %W selection clear
        }
    }

    grid [ttk::frame $frame.f1.fsalt] -row 1 -column 0 -stick ew -pady 5 -padx 3
    grid columnconfigure $frame.f1.fsalt 0 -weight 1
    grid columnconfigure $frame.f1.fsalt 1 -weight 1
    grid [ttk::frame $frame.f1.fsalt.frmconc] -row 0 -column 0 -stick news -pady 0
    grid [ttk::label $frame.f1.fsalt.frmconc.salC -text "Salt Concentration"] -row 0 -column 0 -stick news -pady 0
    grid [ttk::entry $frame.f1.fsalt.frmconc.salCentry -width 7 -justify right -textvariable QWIKMD::basicGui(saltconc,0) ] -row 0 -column 1 -stick w -pady 0
    grid [ttk::label $frame.f1.fsalt.frmconc.salCLab -text "mol/L"] -row 0 -column 2 -stick news -pady 0

    QWIKMD::balloon $frame.f1.fsalt.frmconc.salC [QWIKMD::saltConceBL]
    QWIKMD::balloon $frame.f1.fsalt.frmconc.salCentry [QWIKMD::saltConceBL]

    grid [ttk::frame $frame.f1.fsalt.fcomb] -row 0 -column 1 -stick nes -pady 0

    grid [ttk::label $frame.f1.fsalt.fcomb.flsalt -text "Choose Salt"] -row 0 -column 0 -stick ns -pady 0
    set values {NaCl KCl}
    grid [ttk::combobox $frame.f1.fsalt.fcomb.combSalt -values $values -width 10 -state readonly -textvariable QWIKMD::basicGui(saltions,0)] -row 0 -column 1 -pady 0
    
    QWIKMD::createInfoButton $frame.f1 0 2

    bind $frame.f1.info <Button-1> {
        set val [QWIKMD::mdSmdInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow mdSmdInfo [lindex $val 0] [lindex $val 2]
    }

    if {$level == "basic"} {
        set QWIKMD::basicGui(solvent,0) "Implicit"
        set QWIKMD::basicGui(solvent,$MD) $frame.f1.fsolv.soltype.combSolv

        set QWIKMD::basicGui(saltions,0) "NaCl"
        set QWIKMD::basicGui(saltions,$MD) $frame.f1.fsalt.fcomb.combSalt

        set QWIKMD::basicGui(saltconc,0) "0.15"
        set QWIKMD::basicGui(saltconc,$MD) $frame.f1.fsalt.frmconc.salCentry
    } else {
        set QWIKMD::advGui(solvent,0) "Explicit"
        $frame.f1.fsolv.soltype.combSolv configure -textvariable QWIKMD::advGui(solvent,0)
        set QWIKMD::advGui(solvent,$MD) $frame.f1.fsolv.soltype.combSolv

        set QWIKMD::advGui(saltions,0) "NaCl"
        $frame.f1.fsalt.fcomb.combSalt configure -textvariable QWIKMD::advGui(saltions,0)
        set QWIKMD::advGui(saltions,$MD) $frame.f1.fsalt.fcomb.combSalt

        set QWIKMD::advGui(saltconc,0) "0.15"
        set QWIKMD::advGui(saltconc,$MD) $frame.f1.fsalt.frmconc.salCentry

        set QWIKMD::advGui(solvent,boxbuffer,$MD,entry) $frame.f1.fsolv.boxsize.buffer.combval
    }


    QWIKMD::balloon $frame.f1.fsalt.fcomb.combSalt [QWIKMD::saltTypeBL]

    bind $frame.f1.fsolv.soltype.combSolv <<ComboboxSelected>> {
        QWIKMD::ChangeSolvent
        %W selection clear
    }
    bind $frame.f1.fsalt.fcomb.combSalt <<ComboboxSelected>> {
        %W selection clear  
    }
    $frame.f1.fsalt.fcomb.combSalt configure -state disable
}

############################################################
## Add frames to the simulation notebook inside the Run tab
############################################################


proc QWIKMD::hideFrame {w frame txt} {
    set frameaux "$frame.fcolapse"
    set arrow [lindex [$w cget -text] 0]
    if {$arrow != $QWIKMD::rightPoint} {
        $w configure -text "$QWIKMD::rightPoint $txt"
        grid forget $frameaux
    } else {
        $w configure -text "$QWIKMD::downPoint $txt"
        set info [grid info $w]
        grid conf $frameaux -row [expr [lindex $info 5] +1] -column [lindex $info 3] -pady 1 -padx 2 -sticky ewns
    }
}

    
proc QWIKMD::protocolBasic {frame PRT} {

    ############################################################
    ## First the common widgets between SMD and MD are created
    ## and then, inside the if statement, the specific widgets 
    ## are created
    ############################################################

    ## Frame Protocol
    grid [ttk::frame $frame.f2] -row 1 -column 0 -sticky ew -pady 2
    grid columnconfigure $frame.f2 0 -weight 1
    
    grid [ttk::label $frame.f2.prt -text "$QWIKMD::rightPoint Protocol"] -row 0 -column 0 -sticky w -pady 2
    
    grid [ttk::frame $frame.f2.fcolapse] -row 1 -column 0 -sticky ew -pady 2
    grid columnconfigure $frame.f2.fcolapse 0 -weight 1

    grid rowconfigure $frame.f2.fcolapse 3 -weight 1
    bind $frame.f2.prt <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Protocol"
    }

    set framecolapse $frame.f2.fcolapse

    grid [ttk::frame $framecolapse.sep] -row 0 -column 0 -sticky ew -pady 2
    grid columnconfigure $framecolapse.sep 0 -weight 1

    grid [ttk::separator $framecolapse.sep.spt -orient horizontal] -row 0 -column 0 -sticky ew -pady 0

    if {$PRT != "MDFF"} {

        grid [ttk::frame $framecolapse.fcheck] -row 1 -column 0 -sticky news -padx 0 -pady 0
        grid columnconfigure $framecolapse.fcheck 3 -weight 1

        grid [ttk::checkbutton $framecolapse.fcheck.min -text "Equilibration" -variable QWIKMD::basicGui(prtcl,$PRT,equi)] -row 0 -column 0 -sticky ew -padx 2
        grid [ttk::checkbutton $framecolapse.fcheck.md -text "MD" -variable QWIKMD::basicGui(prtcl,$PRT,md)] -row 0 -column 1 -sticky ew -padx 2
        set QWIKMD::basicGui(prtcl,$PRT,equibtt) $framecolapse.fcheck.min
        set QWIKMD::basicGui(prtcl,$PRT,mdbtt) $framecolapse.fcheck.md
        grid [ttk::frame $framecolapse.sep2] -row 2 -column 0 -sticky ew -pady 2
        grid columnconfigure $framecolapse.sep2 0 -weight 1
        grid [ttk::separator $framecolapse.sep2.spt -orient horizontal] -row 0 -column 0 -sticky ew -pady 0
        
        QWIKMD::balloon $framecolapse.fcheck.min [QWIKMD::EquiMDBL]
        QWIKMD::balloon $framecolapse.fcheck.md [QWIKMD::mdMDBL]

        QWIKMD::createInfoButton $framecolapse.fcheck 0 3
        bind $framecolapse.fcheck.info <Button-1> {
            set val [QWIKMD::protocolMDInfo]
            set QWIKMD::link [lindex $val 1]
            QWIKMD::infoWindow protocolMDInfo [lindex $val 0] [lindex $val 2]
        }

        set QWIKMD::basicGui(prtcl,$PRT,equi) 1
        set QWIKMD::basicGui(prtcl,$PRT,md) 1
    } 

    grid [ttk::frame $framecolapse.fopt] -row 3 -column 0 -sticky ew -pady 5
    grid columnconfigure $framecolapse.fopt 0 -weight 1
    grid rowconfigure $framecolapse.fopt 1 -weight 1

    grid [ttk::frame $framecolapse.fopt.temp] -row 0 -column 0 -sticky ew
    grid [ttk::label $framecolapse.fopt.temp.ltemp -text "Temperature" -justify center] -row 0 -column 0 -sticky ew 

    ############################################################
    ## The format procs are outside the validatecommand beacuse inside 
    ## the validate definition section, the format command does not work 
    ############################################################

    proc format5Dec {val} {
        return [format %.5f $val]
    }
    proc format2Dec {val} {
        return [format %.2f $val]
    }
    
    proc format0Dec {val} {
        return [expr int($val)]
    }
    set QWIKMD::basicGui(temperature,0) "27"
    grid [ttk::entry $framecolapse.fopt.temp.entrytemp -width 7 -justify right -textvariable QWIKMD::basicGui(temperature,0) -validate focusout -validatecommand {
        
        if {[info exists QWIKMD::basicGui(temperature,$QWIKMD::run)]} {
            $QWIKMD::basicGui(temperature,$QWIKMD::run) configure -text [expr $QWIKMD::basicGui(temperature,0) + 273]
            $QWIKMD::basicGui(temperature,$QWIKMD::run) configure -text [expr $QWIKMD::basicGui(temperature,0) + 273]
        }

        return 1
        }] -row 0 -column 1 -sticky ew 
        
    grid [ttk::label $framecolapse.fopt.temp.lcent -text "C"] -row 0 -column 2 -sticky w 

    grid [ttk::frame $framecolapse.fopt.temp.kelvin] -row 0 -column 3 -sticky w
    grid [ttk::label $framecolapse.fopt.temp.kelvin.ltempkelvin -justify center -text [expr $QWIKMD::basicGui(temperature,0) + 273]] -row 0 -column 0 -sticky w -padx 2 
    grid [ttk::label $framecolapse.fopt.temp.kelvin.k -text "K" -justify center] -row 0 -column 1 -sticky w

    QWIKMD::balloon $framecolapse.fopt.temp.ltemp [QWIKMD::mdTemperatureBL]
    QWIKMD::balloon $framecolapse.fopt.temp.entrytemp [QWIKMD::mdTemperatureBL]

    set QWIKMD::basicGui(temperature,$PRT) $framecolapse.fopt.temp.kelvin.ltempkelvin
    if {$PRT == "MD"} {

        grid [ttk::label $framecolapse.fopt.temp.ltime -text "Simulation Time" -justify center] -row 1 -column 0 -sticky ew 
        
        grid [ttk::entry $framecolapse.fopt.temp.entrytime -width 7 -justify right -validate focusout -textvariable QWIKMD::basicGui(mdtime,0) -validatecommand {
            set val [QWIKMD::format0Dec [expr $QWIKMD::basicGui(mdtime,0) / 2e-6]]
            set mod [expr fmod($val,20)]
            if { $mod != 0.0} { 
                set QWIKMD::basicGui(mdtime,0) [QWIKMD::format5Dec [expr [expr $val + {20 - $mod}] * 2e-6 ] ]
                return 0
            } else {
                return 1
            }
            }] -row 1 -column 1 -sticky ew 
        grid [ttk::label $framecolapse.fopt.temp.lns -text "ns"] -row 1 -column 2 -sticky ew 

        set QWIKMD::basicGui(mdtime,0) "10.0"
        QWIKMD::balloon $framecolapse.fopt.temp.ltime [QWIKMD::mdMaxTimeBL]
        QWIKMD::balloon $framecolapse.fopt.temp.entrytime [QWIKMD::mdMaxTimeBL]
    } elseif {$PRT == "SMD"} {

        $framecolapse.fcheck.md configure -text "MD"

        QWIKMD::balloon $framecolapse.fcheck.md [QWIKMD::smdEqMDBL]
        grid [ttk::checkbutton $framecolapse.fcheck.smd -text "SMD" -variable QWIKMD::basicGui(prtcl,$PRT,smd)] -row 0 -column 2 -sticky ew -padx 2

        set QWIKMD::basicGui(prtcl,$PRT,smd) 1
        set QWIKMD::basicGui(prtcl,$PRT,smdbtt) $framecolapse.fcheck.smd
        QWIKMD::balloon $framecolapse.fcheck.smd [QWIKMD::smdSMDBL]

        QWIKMD::addSMDVD $framecolapse.fopt.temp 1 0

        QWIKMD::addSMDAP $framecolapse.fopt.temp 0 4

        grid [ttk::label $framecolapse.fopt.temp.mtime -text "Simulation Time" -justify center] -row 3 -column 0 -sticky ew 
        grid [ttk::entry $framecolapse.fopt.temp.entrytime -width 7 -justify right -textvariable QWIKMD::basicGui(mdtime,1) -validate focus -validatecommand {QWIKMD::reviewLenVelTime 3} ] -row 3 -column 1 -sticky ew 
        grid [ttk::label $framecolapse.fopt.temp.lmaxTime -text "ns" ] -row 3 -column 2 -sticky ew
        QWIKMD::balloon $framecolapse.fopt.temp.mtime [QWIKMD::mdMaxTimeBL]
        QWIKMD::balloon $framecolapse.fopt.temp.entrytime [QWIKMD::mdMaxTimeBL]

        set QWIKMD::basicGui(mdtime,1) 0
    } elseif {$PRT == "MDFF"} {
        grid configure $frame.f2 -sticky nsew 
        grid rowconfigure $frame.f2 1 -weight 1
        
        grid configure $framecolapse -sticky nsew 
        grid rowconfigure $framecolapse 1 -weight 0
        grid rowconfigure $framecolapse 2 -weight 0

        grid configure $framecolapse.fopt -sticky nsew 
        grid rowconfigure $framecolapse.fopt 1 -weight 2
        grid rowconfigure $framecolapse.fopt 0 -weight 0

        grid [ttk::label $framecolapse.fopt.temp.mintime -text "Minimization Steps" -justify center] -row 1 -column 0 -sticky ew
        grid [ttk::entry $framecolapse.fopt.temp.entrytime -width 7 -justify right -validate focusout -textvariable QWIKMD::advGui(mdff,min)] -row 1 -column 1 -sticky ew 

        grid [ttk::label $framecolapse.fopt.temp.mdffTime -text "MDFF Steps" -justify center] -row 2 -column 0 -sticky ew
        grid [ttk::entry $framecolapse.fopt.temp.entrymdfftime -width 7 -justify right -validate focusout -textvariable QWIKMD::advGui(mdff,mdff)] -row 2 -column 1 -sticky ew 

        set QWIKMD::advGui(mdff,min) 400
        set QWIKMD::advGui(mdff,mdff) 50000

        grid [ttk::frame $framecolapse.fopt.tableframe ] -row 1 -column 0 -sticky nwse -padx 2 -pady 2

        grid columnconfigure $framecolapse.fopt.tableframe  0 -weight 1
        grid rowconfigure $framecolapse.fopt.tableframe  0 -weight 1

        set fro2 $framecolapse.fopt.tableframe 
        option add *Tablelist.activeStyle       frame
        option add *Tablelist.background        gray98
        option add *Tablelist.stripeBackground  #e0e8f0
        option add *Tablelist.setGrid           yes
        option add *Tablelist.movableColumns    no

        tablelist::tablelist $fro2.tb \
        -columns { 0 "Fixed" center
                0 "Sec. Structure"   center
                0 "Chirality" center
                0 "Cispeptide" center
                } -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] -showseparators 0 -labelrelief groove  -labelbd 1 -selectbackground white \
                -selectforeground black -foreground black -background white -state normal -selectmode single -stretch "all" -stripebackgroun white -height 2\
                -editstartcommand QWIKMD::startEditMDFF -editendcommand QWIKMD::finishEditMDFF -forceeditendcommand true
        
        $fro2.tb columnconfigure 0 -sortmode dictionary -name Fixed
        $fro2.tb columnconfigure 1 -sortmode real -name SecStrct
        $fro2.tb columnconfigure 2 -sortmode dictionary -name Chiral
        $fro2.tb columnconfigure 3 -sortmode dictionary -name Cispep

        $fro2.tb columnconfigure 0 -width 12 -maxwidth 0 -editable true -editwindow ttk::combobox -wrap true
        $fro2.tb columnconfigure 1 -width 12 -maxwidth 0 -editable true -editwindow ttk::combobox -wrap true
        $fro2.tb columnconfigure 2 -width 12 -maxwidth 0 -editable true -editwindow ttk::combobox -wrap true
        $fro2.tb columnconfigure 3 -width 12 -maxwidth 0 -editable true -editwindow ttk::combobox -wrap true

        ##Scrool_BAr V
        scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
        grid $fro2.scr1 -row 0 -column 1  -sticky ens

        ## Scrool_Bar H
        scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
        grid $fro2.scr2 -row 1 -column 0 -sticky swe

        grid $fro2.tb -row 0 -column 0 -sticky news
        grid columnconfigure $fro2.tb 0 -weight 1; grid rowconfigure $fro2.tb 0 -weight 1

        set QWIKMD::advGui(protocoltb,$PRT) $fro2.tb

        $fro2.tb insert end {none "same fragment as protein" "same fragment as protein" "same fragment as protein"}

     } 
     grid forget $framecolapse
}

proc QWIKMD::startEditMDFF {tbl row col text} {
    set w [$tbl editwinpath]
    set values [list]
    switch [$tbl columncget $col -name] {
        Fixed {
            set values {none all "From List"}
            $w configure -values $values -state normal -style protocol.TCombobox -takefocus 0 -exportselection false -justify center
        }
        SecStrct {
            set values {none "same fragment as protein"}
            $w configure -values $values -state readonly -style protocol.TCombobox -takefocus 0 -exportselection false -justify center
        }
        Chiral {
            set values {none "same fragment as protein"}
            $w configure -values $values -state readonly -style protocol.TCombobox -takefocus 0 -exportselection false -justify center
        }
        Cispep {
            set values {none "same fragment as protein"}
            $w configure -values $values -state readonly -style protocol.TCombobox -takefocus 0 -exportselection false -justify center
            
        }
    }
    bind $w <<ComboboxSelected>> {
        $QWIKMD::advGui(protocoltb,$QWIKMD::run) finishediting  
    }
    $w set $text
    return $text
}

proc QWIKMD::finishEditMDFF {tbl row col text} {
    set w [$tbl editwinpath]
    if {[molinfo num] == 0} {
        $w selection clear
        return $text
    }
    switch [$tbl columncget $col -name] {
        Fixed {
            if {$text == "From List"} {
                set QWIKMD::anchorpulling 0
                set QWIKMD::buttanchor 0
                set QWIKMD::selResidSel "Type Selection"
                QWIKMD::selResidForSelection "MDFF Fixed Selection" [list]
                $tbl rejectinput
            } else {
                if {[lsearch {none all "From List"} $text] == -1} {
                    set checkOk [QWIKMD::checkSelection $w protocol.TEntry]
                    if !$checkOk {
                        set text "none"
                        ttk::style configure protocol.TCombobox -foreground black
                    }
                }
            }
        }
        SecStrct {
            return $text
        }
        Chiral {
            return $text
        }
        Cispep {
            return $text
        }
    }
    
    return $text
}

proc QWIKMD::protocolAdvanced {frame PRT} {

    ## Frame Protocol
    grid [ttk::frame $frame.f2] -row 1 -column 0 -sticky ewns -pady 2
    grid columnconfigure $frame.f2 0 -weight 1
    grid rowconfigure $frame.f2 0 -weight 0
    grid rowconfigure $frame.f2 1 -weight 2
    grid [ttk::label $frame.f2.prt -text "$QWIKMD::rightPoint Protocol"] -row 0 -column 0 -sticky w -pady 2

    bind $frame.f2.prt <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Protocol"
    }

    grid [ttk::frame $frame.f2.fcolapse ] -row 1 -column 0 -sticky ewns -pady 2
    grid columnconfigure $frame.f2.fcolapse 0 -weight 1
    grid rowconfigure $frame.f2.fcolapse 0 -weight 1

    grid [ttk::frame $frame.f2.fcolapse.tableframe ] -row 0 -column 0 -sticky nwse -padx 4

    grid columnconfigure $frame.f2.fcolapse.tableframe 0 -weight 1
    grid rowconfigure $frame.f2.fcolapse.tableframe 0 -weight 1

    set fro2 $frame.f2.fcolapse.tableframe
    option add *Tablelist.activeStyle       frame
    option add *Tablelist.background        gray98
    option add *Tablelist.stripeBackground  #e0e8f0
    option add *Tablelist.setGrid           yes
    option add *Tablelist.movableColumns    no

        tablelist::tablelist $fro2.tb \
        -columns { 0 "Protocol"  center
                0 "n Steps"  center
                0 "Restraints" center
                0 "Ensemble" center
                0 "Temp (C)" center 
                0 "Pressure (atm)" center 
                } -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] -showseparators 0 -labelrelief groove  -labelbd 1 -selectbackground cyan \
                -selectforeground black -foreground black -background white -state normal -selectmode single -stretch "0 1 2" -stripebackgroun white -height 5 \
                -editstartcommand QWIKMD::cellStartEditPtcl -editendcommand QWIKMD::cellEndEditPtcl -forceeditendcommand true -editselectedonly true

    $fro2.tb columnconfigure 0 -sortmode dictionary -name Protocol
    $fro2.tb columnconfigure 1 -sortmode real -name nSteps
    $fro2.tb columnconfigure 2 -sortmode dictionary -name Restraints
    $fro2.tb columnconfigure 3 -sortmode dictionary -name Ensemble
    $fro2.tb columnconfigure 4 -sortmode real -name Temp
    $fro2.tb columnconfigure 5 -sortmode real -name Pressure

    $fro2.tb columnconfigure 0 -width 12 -maxwidth 0 -editable true -editwindow ttk::combobox 
    $fro2.tb columnconfigure 1 -width 0 -maxwidth 0 -editable true -editwindow spinbox
    $fro2.tb columnconfigure 2 -width 20 -maxwidth 0 -editable true -editwindow ttk::combobox -wrap true
    $fro2.tb columnconfigure 3 -width 0 -maxwidth 0 -editable true -editwindow ttk::combobox -wrap true
    $fro2.tb columnconfigure 4 -width 0 -maxwidth 0 -editable true -editwindow spinbox
    $fro2.tb columnconfigure 5 -width 0 -maxwidth 0 -editable true -editwindow spinbox
    
    grid $fro2.tb -row 0 -column 0 -sticky news
    grid columnconfigure $fro2.tb 0 -weight 1; grid rowconfigure $fro2.tb 0 -weight 1

    ##Scrool_BAr V
    scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
     grid $fro2.scr1 -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
    grid $fro2.scr2 -row 1 -column 0 -sticky swe

    bind [$fro2.tb bodytag] <Double-Button-1>  {
        [tablelist::getTablelistPath  %W] selection clear 0 end
    }


    bind [$fro2.tb labeltag] <Any-Enter> {
        set col [tablelist::getTablelistColumn %W]
        set help 0
        switch $col {
            0 {
                set help [QWIKMD::selTabelProtocol]
            }
            1 {
                set help [QWIKMD::selTabelNSteps]
            }
            2 {
                set help [QWIKMD::selTabelRestraints]
            }
            3 {
                set help [QWIKMD::selTabelEnsemble]
            }
            4 {
                set help [QWIKMD::mdTemperatureBL]
            }
            5 {
                set help [QWIKMD::selTabelPressure]
            }
            default {
                set help $col
            }
        }
        after 1000 [list QWIKMD::balloon:show %W $help]
  
    }
    bind [$fro2.tb labeltag] <Any-Leave> "destroy %W.balloon"

    grid [ttk::frame $frame.f2.fcolapse.editProtocol] -row 1 -column 0 -sticky e    

    grid [ttk::button $frame.f2.fcolapse.editProtocol.clear -text "Clear" -padding "0 0 0 0" -command  {
        if {$QWIKMD::load == 0} {
            $QWIKMD::advGui(protocoltb,$QWIKMD::run) delete 0 end
            array unset QWIKMD::advGui protocoltb,$QWIKMD::run,*
            for {set i 0} {$i < 4} {incr i} {
                QWIKMD::addProtocol
            }
            catch {glob $env(TMPDIR)/*.conf} tempLib
            if {[file isfile [lindex ${tempLib} 0]] == 1} {
                foreach file $tempLib {
                    file delete -force -- ${file}
                }
            }
        }
    }] -row 0 -column 0 -sticky e -pady 2 -padx 0

    grid [ttk::button $frame.f2.fcolapse.editProtocol.unlock -text "Unlock" -padding "0 0 0 0" -command  {
        set index [$QWIKMD::advGui(protocoltb,$QWIKMD::run) curselection]
        if {$index != ""} {
            QWIKMD::lockUnlockProc $index
        }
    }] -row 0 -column 1 -sticky e -pady 2 -padx 0

    QWIKMD::balloon $frame.f2.fcolapse.editProtocol.unlock [QWIKMD::selProtocolUnlock]

    grid [ttk::button $frame.f2.fcolapse.editProtocol.edit -text "Edit" -padding "0 0 0 0" -command  {
        set QWIKMD::confFile [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
        QWIKMD::editProtocolProc 
    }] -row 0 -column 2 -sticky e -pady 2 -padx 0

    QWIKMD::balloon $frame.f2.fcolapse.editProtocol.edit [QWIKMD::selProtocolEdit]

    grid [ttk::button $frame.f2.fcolapse.editProtocol.add -text "+" -padding "0 0 0 0" -width 4 -command {
        QWIKMD::addProtocol
    }] -row 0 -column 3 -sticky e -pady 2 -padx 0

    QWIKMD::balloon $frame.f2.fcolapse.editProtocol.add [QWIKMD::selProtocolAdd]

    grid [ttk::button $frame.f2.fcolapse.editProtocol.delete -text "-" -padding "0 0 0 0" -width 4 -command {
        QWIKMD::deleteProtocol
    }] -row 0 -column 4 -sticky e -pady 2 -padx 0
    
    QWIKMD::balloon $frame.f2.fcolapse.editProtocol.delete [QWIKMD::selProtocolDelete]

    if {$PRT == "SMD"} {
        set QWIKMD::advGui(protocoltb,$PRT) $fro2.tb
        grid [ttk::frame $frame.f2.fcolapse.smdOPT] -row 2 -column 0 -sticky ew
        grid columnconfigure $frame.f2.fcolapse.smdOPT 0 -weight 0
        grid columnconfigure $frame.f2.fcolapse.smdOPT 1 -weight 0
        grid columnconfigure $frame.f2.fcolapse.smdOPT 2 -weight 1
        grid columnconfigure $frame.f2.fcolapse.smdOPT 3 -weight 1
        grid columnconfigure $frame.f2.fcolapse.smdOPT 4 -weight 1
        set QWIKMD::basicGui(prtcl,$PRT,smd) 1

        QWIKMD::addSMDAP $frame.f2.fcolapse.smdOPT 0 0
        QWIKMD::addSMDVD $frame.f2.fcolapse.smdOPT 0 3

    } else {
        set QWIKMD::advGui(protocoltb,$PRT) $fro2.tb
    }
    grid forget $frame.f2.fcolapse

}

proc QWIKMD::editProtocolProc {} {
    global env
    set index [$QWIKMD::advGui(protocoltb,$QWIKMD::run) curselection]
    set current [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,0 -text]
    if {$index != ""} {
        if {$QWIKMD::prepared != 1} {
            set template ""
            set tempLib ""
            set do [catch {glob $env(QWIKMDFOLDER)/templates/*.conf} tempLib]
            if { $do == 1} {
                set tempLib ""
            } else {
                set tempAux ""
                foreach temp $tempLib {
                    set aux ""
                    regsub -all ".conf" [file root [file tail $temp ] ] "" aux
                    if {$aux != [file root $current]} {
                        lappend tempAux $aux
                    }           
                }
                set tempLib [lsort -dictionary $tempAux]
            }

            set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index) $current
            if {[info exists QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,saveAsTemplate)] == -1} {
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,saveAsTemplate) 0
            }
            set QWIKMD::advGui(protocoltb,template) $QWIKMD::advGui(protocoltb,$QWIKMD::run,$index)
            if {[lindex [split $QWIKMD::advGui(protocoltb,$QWIKMD::run,$index) "."] 1] == ""} {
                set protocol ".protocol"
                if {[winfo exists $protocol] != 1} {
                    toplevel $protocol
                }
                
                grid columnconfigure $protocol 0 -weight 1
                grid rowconfigure $protocol 0 -weight 1
                ## Title of the windows
                wm title $protocol "Save Configuration file As" ;# titulo da pagina
                set x [expr round([winfo screenwidth .]/2.0)]
                set y [expr round([winfo screenheight .]/2.0)]
                wm geometry $protocol -$x-$y
                wm resizable $protocol 0 0

                grid [ttk::frame $protocol.fp] -row 0 -column 0 -sticky news -padx 10 -pady 10

                set txt "Please specify a name for your costum protocol file: "
                grid [ttk::label $protocol.fp.txt -text $txt] -row 0 -column 0 -sticky ew -padx 2
                set values [file root $current]

                set QWIKMD::prtclSelected $index
                
                grid [ttk::combobox $protocol.fp.combovalues -values $values -textvariable QWIKMD::advGui(protocoltb,template)] -row 0 -column 0 -sticky ew  
                grid [ttk::label $protocol.fp.lbnames -text "NOTE: Don't use \".\" in the protocol name."] -row 1 -column 0 -sticky ew  
                grid [ttk::checkbutton $protocol.fp.checkTemplate -variable QWIKMD::advGui(protocoltb,$QWIKMD::run,$QWIKMD::prtclSelected,saveAsTemplate) -text "Save as template for future use" -command {
                    set values {Minimization Annealing Equilibration MD SMD}
                    set tbnames [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
                    set prtname [.protocol.fp.combovalues get]
                    set index $QWIKMD::prtclSelected
                    set protname [lindex [split $prtname "."] 0 ]
                    set newname ${protname}
                    if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,saveAsTemplate) == 1 && [lsearch $values $protname] != -1} {
                        set newname "${protname}_edited"
                        set QWIKMD::advGui(protocoltb,template) $newname
                        
                    } else {
                        set newname [regsub "_edited" ${protname} ""]
                        set QWIKMD::advGui(protocoltb,template) $newname
                    }   
                }] -row 2 -column 0 -sticky ew  
                
                grid [ttk::frame $protocol.fp.foOkcancel] -row 3 -column 0 -sticky news -padx 10 -pady 10
                grid [ttk::button $protocol.fp.foOkcancel.buttonok -text "Ok" -command {
                    destroy ".protocol"
                }] -row 0 -column 0 -sticky ew  

                grid [ttk::button $protocol.fp.foOkcancel.buttoncancel -text "Cancel" -command {
                    set QWIKMD::advGui(protocoltb,$QWIKMD::run,[$QWIKMD::advGui(protocoltb,$QWIKMD::run) curselection]) "Cancel"
                    destroy ".protocol"
                }] -row 0 -column 1 -sticky ew

                tkwait window $protocol
                if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index) == "Cancel"} {
                    $QWIKMD::advGui(protocoltb,$QWIKMD::run) rejectinput
                    $QWIKMD::advGui(protocoltb,$QWIKMD::run) cancelediting
                    set prevprtcl [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,0 -text]
                    lset QWIKMD::confFile $index $prevprtcl
                    set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index) $prevprtcl
                    return
                } else {
                    lset QWIKMD::confFile $index $QWIKMD::advGui(protocoltb,template)
                }

                if {[file exists $env(QWIKMDFOLDER)/templates/$QWIKMD::advGui(solvent,0)/$QWIKMD::advGui(protocoltb,template).conf] ==1 && $QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,saveAsTemplate) == 1} {
                    set answer [tk_messageBox -message "$QWIKMD::advGui(protocoltb,template).conf protocol already exists. Do you want to replace?" -type yesnocancel -title "Protocol file" -icon info]
                    switch $answer {
                        yes {
                            continue
                        }
                        no {
                            QWIKMD::editProtocolProc
                        }
                        cancel {
                            return
                        }
                    }
                }
                
            }

            set args [$QWIKMD::advGui(protocoltb,$QWIKMD::run) rowcget $index -text]
            set outputfile $env(TMPDIR)/$QWIKMD::advGui(protocoltb,template).conf
            #set outputfile ${outputfile}.conf
            set location ""
            set template ${current}.conf
            set values {Minimization Annealing Equilibration MD SMD}
            set serachindex [lsearch $values [file root $current] ]
            set location $env(QWIKMDFOLDER)/templates/
            if {$serachindex == -1 && [catch {glob ${location}$QWIKMD::advGui(solvent,0)/[file root $current].conf}] == 0} {
                append location $QWIKMD::advGui(solvent,0)
            } elseif {[catch {glob $env(TMPDIR)/[file root $current].conf}] == 0} {
                set location $env(TMPDIR)
            }
            set tempLib ""
            set do [catch {glob $location/*.conf} tempLib]
            ### Variable to check if the duplicated restart protocol has the template already edited in the TMPDIR
            set replicTemplateNotGenerated 0

            if {[file exists "$env(TMPDIR)/${current}.conf"] == 1} {
                set template $env(TMPDIR)/${current}.conf
            } elseif {$do == 0} {
                set tempAux ""
                foreach temp $tempLib {
                    set aux ""
                    lappend tempAux [file tail $temp]   
                }
                set current [file root [file tail $current]]
                set tmpIndex [lsearch [array get QWIKMD::advGui protocoltb,$QWIKMD::run,*] $current]
                if {$tmpIndex != -1} {
                    set tmpIndex [lindex [array get QWIKMD::advGui protocoltb,$QWIKMD::run,*] [expr $tmpIndex -1] ]
                } else {
                    return
                }
                if {[lsearch $tempAux ${current}.conf] == -1 || [file exists "$env(TMPDIR)/${current}.conf"] == 1} {
                    set template $env(TMPDIR)/${current}.conf
                } elseif {[file exists $env(TMPDIR)/${current}.conf] != 1 && [lindex [split $QWIKMD::advGui(protocoltb,$QWIKMD::run,$index) "."] 1] != "" && $QWIKMD::advGui($tmpIndex,lock) == 0} {
                    set template "$location/${current}.conf"
                    if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) == 1} {
                        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) 0
                        set replicTemplateNotGenerated 1
                    }
                } else {    
                    set template $location/${current}.conf
                }
            } 
            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,saveAsTemplate) == 1} {
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) 0 
                #set replicTemplateNotGenerated 1
            }
            if {$QWIKMD::run == "SMD"} {
                if {[QWIKMD::isSMD "$template"] == 1} {
                    set conflist [list]
                    if {[llength $QWIKMD::prevconfFile] > 0} {
                        set conflist $QWIKMD::prevconfFile
                    } else {
                        set conflist $QWIKMD::confFile
                    }
                    set QWIKMD::advGui(protocoltb,$QWIKMD::run,[lsearch $conflist [file root $current]],smd) 1
                    set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) 1 
                }
            }
            QWIKMD::GenerateNamdFiles "qwikmdTemp.psf qwikmdTemp.pdb" "$template" $index $args "$outputfile"
            #if {$replicTemplateNotGenerated == 1} {
            set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) 0
            #}  
            set instancehandle [multitext -justsave]
            $instancehandle openfile "$outputfile"
            #$QWIKMD::advGui(protocoltb,$QWIKMD::run) editcell $index,0
            #$QWIKMD::advGui(protocoltb,$QWIKMD::run) finishediting
            set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index) $QWIKMD::advGui(protocoltb,template)
            $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $index,0 -text $QWIKMD::advGui(protocoltb,template)
            QWIKMD::lockUnlockProc $index
        } else {
            cd ${QWIKMD::outPath}/run
            set instancehandle [multitext -justsave]
            set file [lindex $QWIKMD::confFile $index]
            $instancehandle openfile "${file}.conf"
        }
        
    }
}
proc QWIKMD::addSMDVD {frame row col} {
    set QWIKMD::basicGui(plength) 10.0
    grid [ttk::label $frame.ltime -text "Pulling Distance" -justify center] -row $row -column $col -sticky ew 
    grid [ttk::entry $frame.entryLength -width 7 -justify right -textvariable QWIKMD::basicGui(plength) -validate focus -validatecommand {QWIKMD::reviewLenVelTime 1}] -row $row -column [expr $col +1] -sticky ew 
    grid [ttk::label $frame.lns -text "A"] -row $row -column [expr $col + 2] -sticky ew 

    QWIKMD::balloon $frame.ltime [QWIKMD::smdMaxLengthBL]
    QWIKMD::balloon $frame.entryLength [QWIKMD::smdMaxLengthBL]
    incr row
    set QWIKMD::basicGui(pspeed) 2.5
    grid [ttk::label $frame.lvel -text "Pulling Speed" -justify center] -row $row -column $col -sticky ew 
    grid [ttk::entry $frame.entryvel -width 7 -justify right -textvariable QWIKMD::basicGui(pspeed) -validate focus -validatecommand {QWIKMD::reviewLenVelTime 2}] -row $row -column [expr $col +1] -sticky ew 
    grid [ttk::label $frame.lvelUnit -text "A/ns" ] -row $row -column [expr $col + 2] -sticky ew 

    QWIKMD::balloon $frame.lvel [QWIKMD::smdVelocityBL]
    QWIKMD::balloon $frame.entryvel [QWIKMD::smdVelocityBL]
    
    set QWIKMD::basicGui(mdtime,1) [expr $QWIKMD::basicGui(plength) / $QWIKMD::basicGui(pspeed)]

    
}
proc QWIKMD::addSMDAP {frame row col} {

    grid [ttk::button $frame.pulBut -text "Pulling Residues" -padding "2 0 2 0" -width 15 -command {
        set QWIKMD::anchorpulling 1
        set QWIKMD::selResidSel $QWIKMD::pullingRessel
        QWIKMD::selResidForSelection "Select Pulling Residues" $QWIKMD::pullingRes
        set QWIKMD::buttanchor 2
        set QWIKMD::showpull 1
        QWIKMD::checkAnchors
    }] -row $row -column $col -sticky e -padx 2

    QWIKMD::balloon $frame.pulBut [QWIKMD::smdPullingBL]
    grid [ttk::checkbutton $frame.showpulling -text "Show" -variable QWIKMD::showpull -command {QWIKMD::checkAnchors}] -row $row -column [expr $col +1] -sticky e -padx 4
    incr row
    
    set QWIKMD::showpull 0
    grid [ttk::button $frame.anchorBut -text "Anchoring Residues " -padding "2 0 2 0" -width 15 -command {
        set QWIKMD::anchorpulling 1
        set QWIKMD::selResidSel $QWIKMD::anchorRessel
        QWIKMD::selResidForSelection "Select Anchoring Residues" $QWIKMD::anchorRes
        set QWIKMD::buttanchor 1
        set QWIKMD::showanchor 1
        QWIKMD::checkAnchors
    }] -row $row -column $col -sticky e -padx 2

    QWIKMD::balloon $frame.anchorBut [QWIKMD::smdAnchorBL]
    grid [ttk::checkbutton $frame.showanchor -text "Show" -variable QWIKMD::showanchor -command {QWIKMD::checkAnchors}] -row $row -column [expr $col + 1] -sticky e -padx 4
    set QWIKMD::showanchor 0
    
}

proc QWIKMD::addProtocol {} {
    global env
    set index [$QWIKMD::advGui(protocoltb,$QWIKMD::run) curselection]
    set tbnames [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
    set str ""
    set blocktemp 0
    set blockpress 0
    set ensemble "NpT"
    set lock_previous 0
    if {$QWIKMD::advGui(solvent,0) == "Implicit" || $QWIKMD::advGui(solvent,0) == "Vacuum"} {
        set ensemble "NVT"
        set blockpress 1
    } 
    if {$index == ""} {
    
        set name $QWIKMD::run
        set steps 500000
        set temperature 27
        set restraints "backbone"
        set press 1
        if {[llength $tbnames] == 0} {
            set name "Minimization"
            set restraints "backbone"
            set temperature 0
            set steps 2000
        } elseif {[llength $tbnames] == 1} {
            set name "Annealing"
            set restraints "backbone"
            set temperature 27
            set steps 144000
        } elseif {[llength $tbnames] == 2} {
            set name "Equilibration"
            set restraints "backbone"
            set temperature 27
            set steps 500000
        } else {
            set restraints "none"
        }

        set i 1
        set add 0
        set previndex -1
        while {[lsearch $tbnames $name] != -1} {
            set previndex [lsearch $tbnames $name]
            set name "[file root $name].$i"
            set add 1
            incr i
        }
        
        if {$add == 1 && $i == 2} {
            set lock_previous 1
        }
        if {$add == 1} {
            #set name [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $previndex,0 -text]
            set steps [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $previndex,1 -text]
            set restraints [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $previndex,2 -text]
            set temperature [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $previndex,4 -text]
            set press [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $previndex,5 -text]
        }

        set str [list $name $steps $restraints $ensemble $temperature $press]
        
    } else {
        set name [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,0 -text]
        # set toRemove {Minimization Annealing Equilibration}
        # if {[lsearch $toRemove $name] > -1} {
        #     tk_messageBox -message "Only one $name protocol is allowed." -title "Protocol Replication" -icon warning -type ok
        #     return
        # }

        set i 1
        set add 0
        while {[lsearch $tbnames $name] != -1} {
            set name "[file root $name].$i"
            set add 1
            incr i
        }

        if {$add == 1 && $i == 2} {
            set lock_previous 1
        }
        set nstep [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,1 -text]
        set restraints [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,2 -text]
        set ensemble [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,3 -text]
        set temp [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,4 -text]
        set press [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $index,5 -text]
        set row [$QWIKMD::advGui(protocoltb,$QWIKMD::run) rowcget $index -text]

        set str  [list $name $nstep $restraints $ensemble $temp $press]

    }
    if {$str != ""} {
        
        array set auxArr [list]
        set tblsize [$QWIKMD::advGui(protocoltb,$QWIKMD::run) size]
        set line $tblsize
        if {$index != "" && $index != [expr $tblsize -1] && $tblsize != 0} {
            set lastindex 0
            foreach name $tbnames {
                if {[file root [lindex $str 0] ] == [file root $name]} {
                    incr lastindex
                }
                
            }
            set lastindex [expr  [lsearch $tbnames "[file root [lindex $str 0] ]"] + $lastindex]
            set line $lastindex
            array set auxArr [array get QWIKMD::advGui protocoltb,*]
            set j 0
            for {set i 0} {$i < $tblsize} {incr i} {
                if {$i == $line } {
                    incr j
                }   
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,lock) $auxArr(protocoltb,$QWIKMD::run,$i,lock)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,saveAsTemplate) $auxArr(protocoltb,$QWIKMD::run,$i,saveAsTemplate)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j) $auxArr(protocoltb,$QWIKMD::run,$i)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,restrIndex) $auxArr(protocoltb,$QWIKMD::run,$i,restrIndex)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,restrsel) $auxArr(protocoltb,$QWIKMD::run,$i,restrsel)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,smd) $auxArr(protocoltb,$QWIKMD::run,$i,smd)
                incr j
            }   
        }
        
        $QWIKMD::advGui(protocoltb,$QWIKMD::run) insert $line $str
        if {$blockpress == 1} {
            $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $line,5 -editable false  
        }

        if {$blocktemp == 1} {
            $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $line,4 -editable false  
        }
        set index $line
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) 1
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,saveAsTemplate) 0
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index) [lindex $str 0]
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,restrIndex) [list]
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,restrsel) ""
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) 0
        # if {$QWIKMD::run == "SMD"} {
            
        # }
        QWIKMD::checkProc $index
        if {$lock_previous == 1} {
            $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure [expr $index - 1],0 -editable false  
        }
    }

}

proc QWIKMD::deleteProtocol {} {
    set index [$QWIKMD::advGui(protocoltb,$QWIKMD::run) curselection]
    if {$index != ""} {

        array set auxArr [list]
        set tblsize [$QWIKMD::advGui(protocoltb,$QWIKMD::run) size]
        if {$index != "" && $index != [expr [$QWIKMD::advGui(protocoltb,$QWIKMD::run) size] -1]} {
            
            array set auxArr [array get QWIKMD::advGui protocoltb,$QWIKMD::run*]
            set j 0
            for {set i 0} {$i < $tblsize} {incr i} {
                if {$i == $index} {
                    incr i
                }   
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,lock) $auxArr(protocoltb,$QWIKMD::run,$i,lock)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,saveAsTemplate) $auxArr(protocoltb,$QWIKMD::run,$i,saveAsTemplate)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j) $auxArr(protocoltb,$QWIKMD::run,$i)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,restrIndex) $auxArr(protocoltb,$QWIKMD::run,$i,restrIndex)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,restrsel) $auxArr(protocoltb,$QWIKMD::run,$i,restrsel)
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$j,smd) $auxArr(protocoltb,$QWIKMD::run,$i,smd)
                # if {$QWIKMD::run == "SMD"} {
                    
                # }
                incr j
            }   
            array unset QWIKMD::advGui protocoltb,$QWIKMD::run,[expr $tblsize -1],*
            array unset QWIKMD::advGui protocoltb,$QWIKMD::run,[expr $tblsize -1]
            array unset auxArr *
        } elseif {$tblsize == 1} {
            array unset QWIKMD::advGui protocoltb,$QWIKMD::run,*
        } else {
            array unset QWIKMD::advGui protocoltb,$QWIKMD::run,$index,*
            array unset QWIKMD::advGui protocoltb,$QWIKMD::run,$index
        }
        
        $QWIKMD::advGui(protocoltb,$QWIKMD::run) delete $index
        if {[$QWIKMD::advGui(protocoltb,$QWIKMD::run) size] > 0} {
            QWIKMD::checkProc 0
        }
        if {[expr $index -1] > -1} {
            $QWIKMD::advGui(protocoltb,$QWIKMD::run) selection set [expr $index -1]
        }
    }
}

proc QWIKMD::cellStartEditPtcl {tbl row col text} {
    global env
    set w [$tbl editwinpath]
    
    switch [$tbl columncget $col -name] {
        Protocol {
            
            set values {Minimization Annealing Equilibration MD SMD}

            set tempLib ""
            set do [catch {glob ${env(QWIKMDFOLDER)}/templates/$QWIKMD::advGui(solvent,0)/*.conf} tempLib]
            set tbvalues [$tbl getcolumns $col]
            if {$do == 0} {
                set tempAux ""
                foreach temp $tempLib {
                    set aux ""
                    regsub -all ".conf" [file tail $temp ] "" aux
                    if {[lsearch $values $aux] == -1 && [lsearch $tbvalues $aux] == -1} {
                        lappend values $aux
                    }
                    
                }
            }
            

            if {$QWIKMD::run != "SMD"} {
                set index [lsearch $values "SMD"]
                set values [lreplace $values $index $index]
            }
            
            set toRemove {Minimization Annealing Equilibration MD}

            for {set i 0} {$i < [llength $tbvalues]} {incr i} {

                set index [lsearch -all $toRemove [lindex $tbvalues $i] ]
                if {$index > -1} {
                    set valind [lsearch $values [lindex $toRemove $index]]
                    set values [lreplace $values $valind $valind]
                }
                set tbvalues [$tbl getcolumns $col]
            }
            $w configure -values $values -state readonly -style protocol.TCombobox -takefocus 0 -exportselection false -justify center
            bind $w <<ComboboxSelected>> {
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) finishediting  
            }

        }
        nSteps {
            set from 10
            set to 500000000000
            $w configure -from $from -to $to -increment 20 
        }
        Restraints {
            set values {none backbone "alpha carbon" protein "protein and not hydrogen" "From List"}
            $w configure -width 20 -values $values -state normal -style protocol.TCombobox
            bind $w <<ComboboxSelected>> {
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) finishediting  
            }
        }
        Ensemble {
            set values {NpT NVT NVE}
            if {$QWIKMD::advGui(solvent,0) == "Implicit" || $QWIKMD::advGui(solvent,0) == "Vacuum"} {
                set values {NVT NVE}    
            }
            $w configure -values $values -state readonly -style protocol.TCombobox
            bind $w <<ComboboxSelected>> {
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) finishediting  
            }
        }
        Temp {
            set from 0
            set to 1000
            $w configure -from $from -to $to -increment 0.5
        }
        Pressure {
            set from 0.0
            set to 200
            $w configure -from $from -to $to -increment 0.1
        }
    }
    return $text
}

proc QWIKMD::cellEndEditPtcl {tbl row col text} {
    
    global env
    set w [$tbl editwinpath]

    switch [$tbl columncget $col -name] {
        Protocol {

            set values {Minimization Annealing Equilibration MD SMD}
            set tempLib ""
            set index [lsearch $values $text]
            
            if {$index == -1} {
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$row,lock) 0
            } else {
                set QWIKMD::advGui(protocoltb,$QWIKMD::run,$row,lock) 1
            }
            QWIKMD::lockUnlockProc $row
            # if {$text == "Equilibration" || $text == "Minimization" || $text == "Annealing"} {
            #   $tbl cellconfigure $row,2 -text "backbone"
            #   if {$text == "Annealing"} {
            #       $tbl cellconfigure $row,1 -text 144000
            #   } elseif {$text == "Equilibration"} {
            #       $tbl cellconfigure $row,1 -text 500000
            #   } else {
            #       $tbl cellconfigure $row,1 -text 2000
            #   }
            # } else {
            #   $tbl cellconfigure $row,2 -text "none"
            # }
            if {$text == ""} {set text $QWIKMD::run}
            set QWIKMD::advGui(protocoltb,$QWIKMD::run,$row) $text
        }
        nSteps {
            set val $text
            if {[$tbl cellcget $row,0 -text] == "Annealing"} {
                set temp [expr [$tbl cellcget $row,4 -text] + 213.0]
                set annealval [expr $text / $temp ]
                set textaux $annealval
                if {[expr fmod($annealval,20)] > 0} {
                    set textaux [expr int($annealval + [expr 20 - [expr fmod($annealval,20)]])]
                    set text [expr int($textaux * $temp)]
                }
                set val $textaux 
            }
        
            if {($val <= 0 || [expr fmod($val,20)] > 0)} {
                tk_messageBox -message "Number of steps must be positive and multiple of 20." -icon warning -type ok
                $tbl rejectinput
            } else {
                lset QWIKMD::maxSteps $row $text
            }
        }
        Restraints {
            if {[molinfo num] == 0 } {
                tk_messageBox -message "No molecule loaded" -title "No Molecule" -icon warning -type ok
                return [$tbl cellcget $row,$col -text]
            }
            if {$text == ""} {set text "none"}
            if {$text != "none"} {
                if {$text == "From List"} {
                    set QWIKMD::anchorpulling 0
                    set QWIKMD::buttanchor 0
                    set QWIKMD::selResidSel "Type Selection"
                    QWIKMD::selResidForSelection "Restraints Selection" $QWIKMD::advGui(protocoltb,$QWIKMD::run,$row,restrIndex)
                    $tbl rejectinput
                } else {
                    set sel ""
                    set length [expr [llength [array names QWIKMD::chains]] /3]
                    set seltxt ""
                    for {set i 0} {$i < $length} {incr i} {
                        if {$QWIKMD::chains($i,0) == 1} {
                            append seltxt " ([lindex $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)]) or"  
                        }
                        
                    }
                    set seltextaux $text
                    if {$text == "protein and not hydrogen"} {
                        set seltextaux "protein and noh"
                    }
                    set seltxt [string trimleft $seltxt " "]
                    set seltxt [string trimright $seltxt " or"]
                    set seltxt "($seltxt) and $seltextaux"
                    set do [catch {atomselect $QWIKMD::topMol $seltxt} sel]
                    
                    if {$do == 1} {
                        set ind ""
                    } else {
                        set ind [$sel get index]
                    }
                    $sel delete
                    if {$ind == ""} {
                        tk_messageBox -message "Invalide atom selection." -icon warning -type ok
                        $tbl rejectinput
                    }
                }   
            }
        }
        Ensemble {
            if {$text == "NVE"} {
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,4 -editable false
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,5 -editable false
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,4 -foreground grey -selectforeground grey
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,5 -foreground grey -selectforeground grey
            } elseif {$text == "NVT"  && $QWIKMD::advGui(protocoltb,$QWIKMD::run,$row,lock) == 0} {
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,4 -editable true
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,5 -editable false
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,4 -foreground black -selectforeground black
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,5 -foreground grey -selectforeground grey
            } elseif {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$row,lock) == 0} {
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,4 -editable true
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,5 -editable true
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,4 -foreground black -selectforeground black
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $row,5 -foreground black -selectforeground black
            }
        }
        Temp {
            if {$text > 100} {
                tk_messageBox -message "Temperature too high. Please note that temperature is Celsius and not Kelvin." -icon warning -type ok
                
            } elseif {$text == ""} {
                set text 27
            }
        }
        Pressure {
            if {$text < 0 || $text == ""} {
                set text 1
            } 
        }
    }
    return $text
}

proc QWIKMD::BasicAnalyzeFrame {frame} {
    
    grid [ttk::frame $frame.fp ] -row 0 -column 0 -sticky nsew -pady 2 -padx 2 
    grid columnconfigure $frame.fp 0 -weight 1
    
    set row 0
    grid rowconfigure $frame.fp $row -weight 0
    grid [ttk::frame $frame.fp.rmsd -relief groove] -row $row -column 0 -sticky nsew -pady 2 -padx 2 
    grid columnconfigure $frame.fp.rmsd 0 -weight 1

    QWIKMD::RMSDFrame $frame.fp.rmsd

    incr row
    grid rowconfigure $frame.fp $row -weight 0

    grid [ttk::frame $frame.fp.energies -relief groove] -row $row -column 0 -sticky nsew -pady 4 -padx 2 
    grid columnconfigure $frame.fp.energies 0 -weight 1

    QWIKMD::EnerFrame $frame.fp.energies

    incr row
    grid rowconfigure $frame.fp $row -weight 0
    grid [ttk::frame $frame.fp.thermo -relief groove] -row $row -column 0 -sticky nsew -pady 4 -padx 2 
    grid columnconfigure $frame.fp.thermo 0 -weight 1

    QWIKMD::ThermoFrame $frame.fp.thermo

    # incr row
    # grid rowconfigure $frame.fp $row -weight 0
    # grid [ttk::frame $frame.fp.render -relief groove] -row $row -column 0 -sticky nsew -pady 4 -padx 2 
    # grid columnconfigure $frame.fp.render 0 -weight 1

    # QWIKMD::RenderFrame $frame.fp.render

    incr row
    grid rowconfigure $frame.fp $row -weight 2
    grid [ttk::frame $frame.fp.plot ] -row $row -column 0 -sticky nsew -pady 4 -padx 2 
    grid columnconfigure $frame.fp.plot 0 -weight 1

    QWIKMD::plotframe $frame.fp.plot basic

}

proc QWIKMD::RMSDFrame {frame} {

    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    grid [ttk::label $frame.header.lbtitle -text "$QWIKMD::rightPoint RMSD" -width 15] -row 0 -column 0 -sticky nw -pady 2 -padx 2

    bind $frame.header.lbtitle <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "RMSD"
    }
    grid [ttk::frame $frame.header.fcolapse ] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header.fcolapse 0 -weight 1

    grid [ttk::button $frame.header.fcolapse.rmsdRun -text "Calculate" -padding "2 2 2 2" -width 15 -command {
         #set plot 0
         if {$QWIKMD::rmsdGui == ""} {
            #set plot 1
            set info [QWIKMD::addplot frmsd "RMSD Plot" "Rmsd vs Time" "Time (ns)" "Rmsd (A)"]
            set QWIKMD::rmsdGui [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::rmsdGui clear
                set QWIKMD::timeXrmsd 0
                set QWIKMD::rmsd 0
                $QWIKMD::rmsdGui add 0 0
                $QWIKMD::rmsdGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::rmsdGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).frmsd
                set QWIKMD::rmsdGui ""
                set QWIKMD::rmsdplotview 0
            }
            set QWIKMD::rmsdplotview 1

        } else {
            $QWIKMD::rmsdGui clear
            set QWIKMD::timeXrmsd 0
            set QWIKMD::rmsd 0
            $QWIKMD::rmsdGui add 0 0
            $QWIKMD::rmsdGui replot
            set QWIKMD::rmsdplotview 1
        } 

        if {$QWIKMD::load == 1} {

            set numframes [molinfo $QWIKMD::topMol get numframes]
            set seltext ""
            if {$QWIKMD::advGui(analyze,basic,selentry) != "" && $QWIKMD::advGui(analyze,basic,selentry) != "Type Selection"} {
                set seltext $QWIKMD::advGui(analyze,basic,selentry)
            } else {
                set seltext $QWIKMD::advGui(analyze,basic,selcombo)
            }
            set sel_ref [atomselect $QWIKMD::topMol $seltext frame 0]
            set sel [atomselect $QWIKMD::topMol $seltext]
            set j 0
            set do 1
            set const 2e-6
            set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ]
            for {set i 1} {$i < $numframes} {incr i} {

                if {$i < [lindex $QWIKMD::lastframe $j]} {
                    if {$do == 1} {
                        set logfile [open [lindex $QWIKMD::confFile $j].log r]
                        while {[eof $logfile] != 1 } {
                            set line [gets $logfile]

                            if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                                set const [expr [lindex $line 2] * 1e-6]
                            }

                            if {[lindex $line 0] == "Info:" && [join [lrange $line 1 2]] == "DCD FREQUENCY" } {
                                set QWIKMD::dcdfreq [lindex $line 3]
                                break
                            }
                        }
                        close $logfile
                        set do 0
                        set increment [expr $const * [expr $QWIKMD::dcdfreq * $QWIKMD::loadstride] ]
                    }   
                } else {
                    incr j
                    set do 1
                }
                $sel frame $i
                set xtime [expr [lindex $QWIKMD::timeXrmsd end] + $increment]
                lappend QWIKMD::timeXrmsd $xtime
                lappend QWIKMD::rmsd [QWIKMD::rmsdAlignCalc $sel $sel_ref $i]
            }
            $QWIKMD::rmsdGui clear
            $QWIKMD::rmsdGui add $QWIKMD::timeXrmsd $QWIKMD::rmsd
            $QWIKMD::rmsdGui replot
            set QWIKMD::rmsdprevx [lindex $QWIKMD::timeXrmsd end]

            puts $QWIKMD::textLogfile [QWIKMD::printRMSD $numframes $seltext $const]
            flush $QWIKMD::textLogfile
        } else {
            QWIKMD::RmsdCalc
        }

    } ] -row 0 -column 0 -sticky ens -pady 2 -padx 1

    QWIKMD::balloon $frame.header.fcolapse.rmsdRun [QWIKMD::rmsdCalcBL]

    QWIKMD::createInfoButton $frame.header 0 0
    bind $frame.header.info <Button-1> {
        set val [QWIKMD::rmsdInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow rmsdInfo [lindex $val 0] [lindex $val 2]
    }

    grid [ttk::frame $frame.header.fcolapse.selection ] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header.fcolapse.selection 0 -weight 0
    grid columnconfigure $frame.header.fcolapse.selection 1 -weight 0
    grid columnconfigure $frame.header.fcolapse.selection 2 -weight 2
    

    set values {"Backbone" "Alpha Carbon" "No Hydrogen" "All"}
    grid [ttk::combobox $frame.header.fcolapse.selection.combo -values $values -width 12 -state readonly  -exportselection 0] -row 0 -column 0 -sticky nsw -padx 2
    grid [ttk::label $frame.header.fcolapse.selection.lbor -text "or"] -row 0 -column 1 -sticky w -padx 5
    
    $frame.header.fcolapse.selection.combo set "Backbone"
    set QWIKMD::advGui(analyze,basic,selcombo) "backbone"
    bind $frame.header.fcolapse.selection.combo <<ComboboxSelected>> {
        set text [%W get]
        switch  $text {
            Backbone {
                set QWIKMD::advGui(analyze,basic,selcombo) "backbone"
            }
            "Alpha Carbon" {
                set QWIKMD::advGui(analyze,basic,selcombo) "alpha carbon"
            }
            "No Hydrogen" {
                set QWIKMD::advGui(analyze,basic,selcombo) "noh"
            }
            "All" {
                set QWIKMD::advGui(analyze,basic,selcombo) "all"
            }
            
        }
        %W selection clear
    }
    ttk::style configure RmsdSel.TEntry -foreground $QWIKMD::tempEntry

    QWIKMD::balloon $frame.header.fcolapse.selection.combo [QWIKMD::rmsdSelection]

    grid [ttk::entry $frame.header.fcolapse.selection.entry -style RmsdSel.TEntry -textvariable QWIKMD::advGui(analyze,basic,selentry) -validate focus -validatecommand {
        QWIKMD::checkSelection %W RmsdSel.TEntry
        return 1
        }] -row 0 -column 2 -sticky ew -padx 2
    
    $frame.header.fcolapse.selection.entry insert end "Type Selection"

    QWIKMD::balloon $frame.header.fcolapse.selection.entry [QWIKMD::rmsdGeneralSelectionBL] 

    grid [ttk::frame $frame.header.fcolapse.align ] -row 2 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header.fcolapse.align 0 -weight 0
    grid columnconfigure $frame.header.fcolapse.align 1 -weight 0
    grid columnconfigure $frame.header.fcolapse.align 2 -weight 0
    grid columnconfigure $frame.header.fcolapse.align 3 -weight 2

    grid [ttk::checkbutton $frame.header.fcolapse.align.cAlign -text "Align Structure" -variable QWIKMD::advGui(analyze,basic,alicheck)] -row 0 -column 0 -sticky nsw -padx 2
    set QWIKMD::advGui(analyze,basic,alicheck) 0
    

    QWIKMD::balloon $frame.header.fcolapse.align.cAlign [QWIKMD::rmsdAlignBL]

    grid [ttk::combobox $frame.header.fcolapse.align.combo -values $values -width 12 -state readonly] -row 0 -column 1 -sticky nsw -padx 2
    $frame.header.fcolapse.align.combo set "Backbone"
    set QWIKMD::advGui(analyze,basic,alicombo) "backbone"
    
    bind $frame.header.fcolapse.align.combo <<ComboboxSelected>> {
        set text [%W get]
        switch  $text {
            Backbone {
                set QWIKMD::advGui(analyze,basic,alicombo) "backbone"
            }
            "Alpha Carbon" {
                set QWIKMD::advGui(analyze,basic,alicombo) "alpha carbon"
            }
            "No Hydrogen" {
                set QWIKMD::advGui(analyze,basic,alicombo) "noh"
            }
            "All" {
                set QWIKMD::advGui(analyze,basic,alicombo) "all"
            }
            
        }
        %W selection clear
    }

    QWIKMD::balloon $frame.header.fcolapse.align.combo [QWIKMD::rmsdAlignSelection]

    grid [ttk::label $frame.header.fcolapse.align.lbor -text "or"] -row 0 -column 2 -sticky ns -padx 5
    ttk::style configure RmsdAli.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::entry $frame.header.fcolapse.align.entry -style RmsdAli.TEntry -textvariable QWIKMD::advGui(analyze,basic,alientry) -validate focus -validatecommand {
        QWIKMD::checkSelection %W RmsdAli.TEntry
        return 1
        }] -row 0 -column 3 -sticky ew -padx 2
    $frame.header.fcolapse.align.entry insert end "Type Selection"

    QWIKMD::balloon $frame.header.fcolapse.align.entry [QWIKMD::rmsdGeneralAlignSelectionBL]

    set QWIKMD::rmsdsel "all"
    grid forget $frame.header.fcolapse
}

############################################################
## Temperature values during the live simulation are retreived 
## from the communication NAMD-VMD. Pressure and Volume are listed
## in the molinfo command, but NAMD never sent these values previously 
############################################################

proc QWIKMD::ThermoFrame {frame} {

    proc checkCondGui {} {
        if {[winfo exists $QWIKMD::CondGui] == 1} {
            if {[winfo ismapped $QWIKMD::CondGui] == 1} {
                $QWIKMD::topGui.nbinput.f2.fp.condit.selection.plot invoke
            }
        }
    }

    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    grid [ttk::label $frame.header.lbtitle -text "$QWIKMD::rightPoint Thermodynamics"] -row 0 -column 0 -sticky nw -pady 2 -padx 2

    QWIKMD::createInfoButton $frame.header 0 0
    
    bind $frame.header.lbtitle <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Thermodynamics"
    }
    bind $frame.header.info <Button-1> {
        set val [QWIKMD::condPlotInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow condPlotInfo [lindex $val 0] [lindex $val 2]
    }

    grid [ttk::frame $frame.header.fcolapse ] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header.fcolapse 0 -weight 1

    grid [ttk::button $frame.header.fcolapse.plot -text "Calculate" -padding "2 2 2 2" -width 15 -command {
        set ylab "Temperature (K)"
        set xlab "Time (ns)"
            
        set plot 0
        if {$QWIKMD::tempcalc == 1 && $QWIKMD::tempGui == ""}  {
            set plot 1
            set title "Temperature vs Time"

            set info [QWIKMD::addplot tempcalc "Temperature" $title $xlab $ylab]
            set QWIKMD::tempGui [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::tempGui clear
                set QWIKMD::tempval [list]
                set QWIKMD::temppos [list]
                $QWIKMD::tempGui add 0 0
                $QWIKMD::tempGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::tempGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).tempcalc
                set QWIKMD::tempGui ""
            }
        } elseif {$QWIKMD::tempcalc == 0 && $QWIKMD::tempGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).tempcalc
            set QWIKMD::tempGui ""
        }

        if {$QWIKMD::pressurecalc == 1 && $QWIKMD::pressGui == ""}  {
            set plot 1
            set title "Pressure vs Time"
            set ylab "Pressure (bar)"
            set info [QWIKMD::addplot pressurecalc "Pressure" $title $xlab $ylab]
            set QWIKMD::pressGui [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::pressGui clear
                set QWIKMD::pressval [list]
                set QWIKMD::pressvalavg [list]
                set QWIKMD::presspos [list]
                $QWIKMD::pressGui add 0 0
                $QWIKMD::pressGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::pressGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).pressurecalc
                set QWIKMD::pressGui ""
            }
        } elseif {$QWIKMD::pressurecalc == 0 && $QWIKMD::pressGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).pressurecalc
            set QWIKMD::pressGui "" 
        }

        if {$QWIKMD::volumecalc == 1 && $QWIKMD::volGui == ""}  {
            set plot 1
            set title "Volume vs Time"
            set ylab "Volume (A\u00b3)"
            set info [QWIKMD::addplot volumecalc "Volume" $title $xlab $ylab]
            set QWIKMD::volGui [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::volGui clear
                set QWIKMD::volval [list]
                set QWIKMD::volvalavg [list]
                set QWIKMD::volpos [list]
                $QWIKMD::volGui add 0 0
                $QWIKMD::volGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::volGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).volumecalc
                set QWIKMD::volGui ""
                #set QWIKMD::rmsdplotview 0
            }
        } elseif {$QWIKMD::volumecalc == 0 && $QWIKMD::volGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).volumecalc
            set QWIKMD::volGui ""
        }

            
        if {$QWIKMD::load == 1 && $plot == 1} {
            set const 2e-6  
            set time ""
            set index 0
            set limit [expr $QWIKMD::calcfreq *10]
            set limitaux $limit     
            set tempvalaux [list]
            set pressvalaux [list]
            set volvalaux [list]
            set tempaux 0
            set pressaux 0
            set volaux 0
            set QWIKMD::condprevx 0
            set QWIKMD::condprevindex 0
            set loadcondprevindex 0
            set energyfreqaux 1
            set energyfreq 1
            set print 0
            set tstepaux 0
            if {$QWIKMD::tempGui != "" && [llength $QWIKMD::temppos] == 0} {set tempaux 1}
            if {$QWIKMD::pressGui != "" && [llength $QWIKMD::presspos] == 0} {set pressaux 1}
            if {$QWIKMD::volGui != "" && [llength $QWIKMD::volpos] == 0} {set volaux 1}

            if {$tempaux ==1 || $pressaux ==1 || $volaux == 1} {
                set index 0
                set print 1
                for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
                    set file "[lindex $QWIKMD::confFile $i].log"
                    if {[file exists $file] !=1} {
                        break
                    }
                    
                    set logfile [open $file r]
                    set lineprev ""
                    set reset 0
                    while {[eof $logfile] != 1 } {
                        set line [gets $logfile]

                        if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                            set aux [lindex $line 2]
                            set const [expr $aux * 1e-6] 
                            set tstepaux 0
                        }
                        
                        if {[lindex $line 0] == "Info:" && [join [lrange $line 1 3]] == "ENERGY OUTPUT STEPS" } {
                            set energyfreq [lindex $line 4]
                            set energyfreqaux $energyfreq
                            if {$QWIKMD::basicGui(live) == 0} {
                                set limit [expr $energyfreq * 10] 
                                set limitaux $limit     
                            }
                        }

                        if {[lindex $line 0] == "TCL:" && [lindex $line 1] == "Minimizing" } {
                            set energyfreq 1
                            set limit 10
                        }
                        if {[lindex $line 0] == "TCL:" && [lindex $line 1] == "Running" && $reset == 0 } {
                            set energyfreq $energyfreqaux
                            set limit $limitaux     
                            set tstepaux 0
                            set reset 1
                        }

                        if {[lindex $line 0] == "ENERGY:" } {
                            if {$tempaux == 1} {
                                lappend  tempvalaux [lindex $line 15]
                            }

                            if {$pressaux == 1} {
                                lappend  pressvalaux [lindex $line 19]
                            }

                            if {$volaux == 1} {
                                lappend  volvalaux [lindex $line 18]
                            }
                            incr index $energyfreq
                            incr tstepaux $energyfreq
                        } 
                        if {[expr $tstepaux % $limit] == 0 && $index != $loadcondprevindex} {
                            set xtime [expr $const * $index]
                            if {$tempaux ==1} {
                                
                                set min 0
                                if {[llength $QWIKMD::tempval] < 2} {
                                    set min [expr int([expr [llength $tempvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $tempvalaux] -1]
                            
                                lappend QWIKMD::tempval [QWIKMD::mean [lrange $tempvalaux $min $max]]
                                lappend QWIKMD::temppos $xtime
                            }
                            
                            if {$pressaux ==1} {
                                set min 0
                                if {[llength $QWIKMD::pressvalavg] < 2} {
                                    set min [expr int([expr [llength $pressvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $pressvalaux] -1]
                            
                                lappend QWIKMD::pressvalavg [QWIKMD::mean [lrange $pressvalaux $min $max]]
                                lappend QWIKMD::presspos $xtime
                            }
                            if {$volaux == 1} {

                                set min 0
                                if {[llength $QWIKMD::volvalavg] < 2} {
                                    set min [expr int([expr [llength $volvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $volvalaux] -1]
                            
                                lappend QWIKMD::volvalavg [QWIKMD::mean [lrange $volvalaux $min $max]]
                                lappend QWIKMD::volpos $xtime
                            }
                            set tempvalaux [list]
                            set pressvalaux [list]
                            set volvalaux [list]
                            set loadcondprevindex $index
                        }
                        
                        set lineprev $line
                    }
                    if {[lindex $QWIKMD::temppos end] != "" && $tempaux ==1} {
                        set QWIKMD::condprevx [lindex $QWIKMD::temppos end]
                    } elseif {[lindex $QWIKMD::presspos end] != "" && $pressaux ==1} {
                        set QWIKMD::condprevx [lindex $QWIKMD::presspos end]

                    } elseif {[lindex $QWIKMD::volpos end] != "" && $volaux ==1} {
                        set QWIKMD::condprevx [lindex $QWIKMD::volpos end]
                    }
                    
                    close $logfile  
                 }
            }
            if {$QWIKMD::tempGui != "" && [llength $QWIKMD::temppos] > 1} {
                $QWIKMD::tempGui clear
                $QWIKMD::tempGui add $QWIKMD::temppos $QWIKMD::tempval
                $QWIKMD::tempGui replot
            }
            if {$QWIKMD::pressGui != "" && [llength $QWIKMD::presspos] > 1 != ""} {
                $QWIKMD::pressGui clear
                $QWIKMD::pressGui add $QWIKMD::presspos $QWIKMD::pressvalavg
                $QWIKMD::pressGui replot
            }
            if {$QWIKMD::volGui != "" && [llength $QWIKMD::volpos] > 1 != ""} {
                $QWIKMD::volGui clear
                $QWIKMD::volGui add $QWIKMD::volpos $QWIKMD::volvalavg
                $QWIKMD::volGui replot
            }
            if {$print == 1} {
                puts $QWIKMD::textLogfile [QWIKMD::printThermo $xtime $limit $energyfreq $const $tempaux $pressaux $volaux]
                flush $QWIKMD::textLogfile
            }

        } elseif {$QWIKMD::load == 0} {
            QWIKMD::CondCalc
        }

    } ] -row 0 -column 0 -sticky ens -pady 2 -padx 1

    
    QWIKMD::balloon $frame.header.fcolapse.plot [QWIKMD::condCalcBL]

    grid [ttk::frame $frame.header.fcolapse.selection ] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header.fcolapse.selection 0 -weight 1
    grid columnconfigure $frame.header.fcolapse.selection 1 -weight 1
    grid columnconfigure $frame.header.fcolapse.selection 2 -weight 1

    grid [ttk::checkbutton $frame.header.fcolapse.selection.temp -text "Temperature" -variable QWIKMD::tempcalc -command [namespace current]::checkCondGui] -row 0 -column 0 -sticky w -pady 2 -padx 4
    grid [ttk::checkbutton $frame.header.fcolapse.selection.press -text "Pressure" -variable QWIKMD::pressurecalc -command [namespace current]::checkCondGui] -row 0 -column 1 -sticky w -pady 2 -padx 4 
    grid [ttk::checkbutton $frame.header.fcolapse.selection.volume -text "Volume" -variable QWIKMD::volumecalc -command [namespace current]::checkCondGui] -row 0 -column 2 -sticky w -pady 2 -padx 4

    set QWIKMD::advGui(analyze,advance,pressbtt) $frame.header.fcolapse.selection.press
    set QWIKMD::advGui(analyze,advance,volbtt) $frame.header.fcolapse.selection.volume
    QWIKMD::balloon $frame.header.fcolapse.selection.temp [QWIKMD::condTemp]
    QWIKMD::balloon $frame.header.fcolapse.selection.press [QWIKMD::condPress]
    QWIKMD::balloon $frame.header.fcolapse.selection.volume [QWIKMD::condVolume]
    grid forget $frame.header.fcolapse
    
}

proc QWIKMD::EnerFrame {frame} {

    # proc checkEnergyGui {} {
    #   if {[winfo exists $QWIKMD::EnergyGui] == 1} {
    #       if {[winfo ismapped $QWIKMD::EnergyGui] == 1} {
    #           #$QWIKMD::topGui.nbinput.f2.fp.energies.selection.plot invoke
    #       }
    #   }
    # }
    grid [ttk::frame $frame.general ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.general 0 -weight 1

    set frame "$frame.general"
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    
    grid [ttk::label $frame.header.lbtitle -text "$QWIKMD::rightPoint Energies"] -row 0 -column 0 -sticky nw -pady 2 -padx 2
    
    bind $frame.header.lbtitle <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Energies"
    }

    QWIKMD::createInfoButton $frame.header 0 0

    bind $frame.header.info <Button-1> {
        set val [QWIKMD::energiesPlotInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow energiesPlotInfo [lindex $val 0] [lindex $val 2]
    }

    grid [ttk::frame $frame.header.fcolapse ] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header.fcolapse 0 -weight 1
    grid [ttk::button $frame.header.fcolapse.plot -text "Calculate" -padding "2 2 2 2" -width 15 -command {
        set xlab "Time (ns)"
        set ylab "Energy\n(kcal/mol)"
        set plot 0
        if {$QWIKMD::enertotal == 1 && $QWIKMD::energyTotGui == ""}  {
            set plot 1
            set title "Total Energy vs Time"

            set info [QWIKMD::addplot enertotal "Total Energy" "Total Energy vs Time" $xlab $ylab]
            set QWIKMD::energyTotGui [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::energyTotGui clear
                set QWIKMD::enetotval [list]
                set QWIKMD::enetotpos [list]
                $QWIKMD::energyTotGui add 0 0
                $QWIKMD::energyTotGui replot
            }
            $close entryconfigure 0 -command {
                $QWIKMD::energyTotGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).enertotal
                set QWIKMD::energyTotGui ""
            }
        } elseif {$QWIKMD::enertotal == 0 && $QWIKMD::energyTotGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).enertotal
            set QWIKMD::energyTotGui ""
        }

        if {$QWIKMD::enerkinetic == 1 && $QWIKMD::energyKineGui == ""} {

            set plot 1
            set title "Kinetic Energy vs Time"

            set info [QWIKMD::addplot enerkinetic "Kinetic Energy" $title $xlab $ylab]
            set QWIKMD::energyKineGui  [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::energyKineGui clear
                set QWIKMD::enekinval [list]
                set QWIKMD::enekinpos [list]
                $QWIKMD::energyKineGui add 0 0
                $QWIKMD::energyKineGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::energyKineGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).enerkinetic
                set QWIKMD::energyKineGui ""
                
            }
        } elseif {$QWIKMD::enerkinetic == 0 && $QWIKMD::energyKineGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).enerkinetic
            set QWIKMD::energyKineGui ""
        }
    
        if {$QWIKMD::enerpoten == 1 && $QWIKMD::energyPotGui == ""} {

            set plot 1
            set title "Potential Energy vs Time"

            set info [QWIKMD::addplot enerpoten "Potential Energy" $title $xlab $ylab]
            set QWIKMD::energyPotGui  [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::energyPotGui clear
                set QWIKMD::enekinval [list]
                set QWIKMD::enekinpos [list]
                $QWIKMD::energyPotGui add 0 0
                $QWIKMD::energyPotGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::energyPotGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).enerpoten
                set QWIKMD::energyPotGui ""
            }
        } elseif {$QWIKMD::enerpoten == 0 && $QWIKMD::energyPotGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).enerpoten
            set QWIKMD::energyPotGui ""
        }
        

        if {$QWIKMD::enerbond == 1 && $QWIKMD::energyBondGui == ""} {
            set plot 1
            set title "Bond Energy vs Time"

            set info [QWIKMD::addplot enerbond "Bond Energy" $title $xlab $ylab]
            set QWIKMD::energyBondGui  [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::energyBondGui clear
                set QWIKMD::enebondval [list]
                set QWIKMD::enebondpos [list]
                $QWIKMD::energyBondGui add 0 0
                $QWIKMD::energyBondGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::energyBondGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).enerbond
                set QWIKMD::energyBondGui ""
            }
        } elseif {$QWIKMD::enerbond == 0 && $QWIKMD::energyBondGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).enerbond
            set QWIKMD::energyBondGui ""
        }

        if {$QWIKMD::enerangle == 1 && $QWIKMD::energyAngleGui == ""} {
            set plot 1
            set title "Angle Energy vs Time"
            set info [QWIKMD::addplot enerangle "Angle Energy" $title $xlab $ylab]
            set QWIKMD::energyAngleGui  [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::energyAngleGui clear
                set QWIKMD::eneangleval [list]
                set QWIKMD::eneanglepos [list]
                $QWIKMD::energyAngleGui add 0 0
                $QWIKMD::energyAngleGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::energyAngleGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).enerangle
                set QWIKMD::energyAngleGui ""
            }
        } elseif {$QWIKMD::enerangle == 0 && $QWIKMD::energyAngleGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).enerangle
            set QWIKMD::energyAngleGui ""
        }

        if {$QWIKMD::enerdihedral == 1 && $QWIKMD::energyDehidralGui == ""} {
            set plot 1
            set title "Dihedral Energy vs Time"
            set info [QWIKMD::addplot enerdihedral "Dihedral Energy" $title $xlab $ylab]
            set QWIKMD::energyDehidralGui  [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::energyDehidralGui clear
                set QWIKMD::enedihedralval [list]
                set QWIKMD::enedihedralpos [list]
                $QWIKMD::energyDehidralGui add 0 0
                $QWIKMD::energyDehidralGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::energyDehidralGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).enerdihedral
                set QWIKMD::energyDehidralGui ""
            }
        } elseif {$QWIKMD::enerdihedral == 0 && $QWIKMD::energyDehidralGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).enerdihedral
            set QWIKMD::energyDehidralGui ""
        }

        if {$QWIKMD::enervdw == 1 && $QWIKMD::energyVdwGui == ""} {
            set plot 1
            set title "VDW Energy vs Time"
            set info [QWIKMD::addplot enervdw "VDW Energy" $title $xlab $ylab]
            set QWIKMD::energyVdwGui  [lindex $info 0]

            set clear [lindex $info 1]
            set close [lindex $info 2]
            
            $clear entryconfigure 0 -command {
                $QWIKMD::energyVdwGui clear
                set QWIKMD::enevdwval [list]
                set QWIKMD::enevdwpos [list]
                $QWIKMD::energyVdwGui add 0 0
                $QWIKMD::energyVdwGui replot
            }

            $close entryconfigure 0 -command {
                $QWIKMD::energyVdwGui quit
                destroy $QWIKMD::advGui(analyze,basic,ntb).enervdw
                set QWIKMD::energyVdwGui ""
            }
        } elseif {$QWIKMD::enervdw == 0 && $QWIKMD::energyVdwGui != ""} {
            destroy $QWIKMD::advGui(analyze,basic,ntb).enervdw
            set QWIKMD::energyVdwGui ""
        }

        if {$QWIKMD::load == 1 && $plot == 1} {
            
            set time ""
            set index 0
            
            
            set enetotvalaux [list]
            set enekinvalaux [list]
            set enepotvalaux [list]

            set enebondvalaux [list]
            set eneanglevalaux [list]
            set enedihedralvalaux [list]
            set enevdwvalaux [list]
            set tot 0
            set kin 0
            set pot 0
            set bond 0
            set angle 0
            set dihedral 0
            set vdw 0
            set QWIKMD::eneprevx 0
            
            if {$QWIKMD::energyTotGui != "" && [llength $QWIKMD::enetotpos] == 0} {set tot 1}
            if {$QWIKMD::energyPotGui != "" && [llength $QWIKMD::enepotpos] == 0} {set pot 1}
            if {$QWIKMD::energyKineGui != "" && [llength $QWIKMD::enekinpos] == 0} {set kin 1}
            if {$QWIKMD::energyBondGui != "" && [llength $QWIKMD::enebondpos] == 0} {set bond 1}
            if {$QWIKMD::energyAngleGui != "" && [llength $QWIKMD::eneanglepos] == 0} {set angle 1}
            if {$QWIKMD::energyDehidralGui != "" && [llength $QWIKMD::enedihedralpos] == 0} {set dihedral 1}
            if {$QWIKMD::energyVdwGui != "" && [llength $QWIKMD::enevdwpos] == 0} {set vdw 1}
            set print 0
            set xtime 0
            set limit [expr $QWIKMD::calcfreq * 10]
            set limitaux $limit 
            set print 1
            set energyfreq 1
            set const 2e-6  
            set tstep 0
            set tstepaux 0
            set eneprevindex 0
            set energyfreqaux 1
            if {$tot ==1 || $pot ==1 || $kin == 1 || $bond ==1 || $angle == 1|| $dihedral ==1 || $vdw == 1 } {
                
                for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
                    set file "[lindex $QWIKMD::confFile $i].log"
                    if {[file exists $file] != 1} {
                        break
                    }
                    
                    set logfile [open $file r]
                    set lineprev ""
                    set reset 0
                    while {[eof $logfile] != 1 } {
                        set line [gets $logfile]

                        if {[lindex $line 0] == "Info:" && [lindex $line 1] == "TIMESTEP"} {
                            set aux [lindex $line 2]
                            set const [expr $aux * 1e-6]
                            set tstepaux 0
                        }
                        
                        if {[lindex $line 0] == "Info:" && [join [lrange $line 1 3]] == "ENERGY OUTPUT STEPS" } {
                            set energyfreq [lindex $line 4]
                            set energyfreqaux $energyfreq
                            if {$QWIKMD::basicGui(live) == 0} {
                                set limit [expr $energyfreq * 10] 
                                set limitaux $limit 
                            }
                        }

                        if {[lindex $line 0] == "TCL:" && [lindex $line 1] == "Minimizing" } {
                            set energyfreq 1
                            set limit 10
                        }
                        if {[lindex $line 0] == "TCL:" && [lindex $line 1] == "Running" && $reset == 0 } {
                            set energyfreq $energyfreqaux
                            set limit $limitaux     
                            set tstepaux 0
                            set reset 1
                        }

                        if {[lindex $line 0] == "ENERGY:" } {


                            if {$bond == 1} {
                                lappend  enebondvalaux [lindex $line 2]
                            }

                            if {$angle == 1} {
                                lappend  eneanglevalaux [lindex $line 3]
                            }

                            if {$dihedral == 1} {
                                lappend  enedihedralvalaux [lindex $line 4]
                            }

                            if {$vdw == 1} {
                                lappend  enevdwvalaux [lindex $line 7]
                            }

                            if {$tot == 1} {
                                lappend  enetotvalaux [lindex $line 11]
                            }

                            if {$kin == 1} {
                                lappend  enekinvalaux [lindex $line 10]
                            }

                            if {$pot == 1} {
                                lappend  enepotvalaux [lindex $line 13]
                            }

                            incr tstep $energyfreq
                            incr tstepaux $energyfreq
                        }
                        if {[expr $tstepaux % $limit] == 0 && $tstep != $eneprevindex} {
                            set xtime [expr $const * $tstep ]
                            if {$bond ==1} {
                                
                                set min 0
                                if {[llength $QWIKMD::enebondval] < 2} {
                                    set min [expr int([expr [llength $energyfreq] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $enebondvalaux] -1]
                            
                                lappend QWIKMD::enebondval [QWIKMD::mean [lrange $enebondvalaux $min $max]]
                                lappend QWIKMD::enebondpos $xtime
                            }

                            if {$angle ==1} {
                                
                                set min 0
                                if {[llength $QWIKMD::eneangleval] < 2} {
                                    set min [expr int([expr [llength $eneanglevalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $eneanglevalaux] -1]
                            
                                lappend QWIKMD::eneangleval [QWIKMD::mean [lrange $eneanglevalaux $min $max]]
                                lappend QWIKMD::eneanglepos $xtime
                            }

                            if {$dihedral ==1} {
                                
                                set min 0
                                if {[llength $QWIKMD::enedihedralval] < 2} {
                                    set min [expr int([expr [llength $enedihedralvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $enedihedralvalaux] -1]
                            
                                lappend QWIKMD::enedihedralval [QWIKMD::mean [lrange $enedihedralvalaux $min $max]]
                                lappend QWIKMD::enedihedralpos $xtime
                            }

                            if {$vdw ==1} {
                                
                                set min 0
                                if {[llength $QWIKMD::enevdwval] < 2} {
                                    set min [expr int([expr [llength $enevdwvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $enevdwvalaux] -1]
                            
                                lappend QWIKMD::enevdwval [QWIKMD::mean [lrange $enevdwvalaux $min $max]]
                                lappend QWIKMD::enevdwpos $xtime
                            }

                            if {$tot ==1} {
                                
                                set min 0
                                if {[llength $QWIKMD::enetotval] < 2} {
                                    set min [expr int([expr [llength $enetotvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $enetotvalaux] -1]
                            
                                lappend QWIKMD::enetotval [QWIKMD::mean [lrange $enetotvalaux $min $max]]
                                lappend QWIKMD::enetotpos $xtime
                            }
                            
                            if {$kin ==1} {
                                set min 0
                                if {[llength $QWIKMD::enekinval] < 2} {
                                    set min [expr int([expr [llength $enekinvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $enekinvalaux] -1]
                            
                                lappend QWIKMD::enekinval [QWIKMD::mean [lrange $enekinvalaux $min $max]]
                                lappend QWIKMD::enekinpos $xtime
                            }
                            if {$pot == 1} {

                                set min 0
                                if {[llength $QWIKMD::enepotval] < 2} {
                                    set min [expr int([expr [llength $enepotvalaux] - [expr 1.5 * $limit] -1])]  
                                }
                                
                                set max [expr [llength $enepotvalaux] -1]
                            
                                lappend QWIKMD::enepotval [QWIKMD::mean [lrange $enepotvalaux $min $max]]
                                lappend QWIKMD::enepotpos $xtime
                            }
                            set enetotvalaux [list]
                            set enekinvalaux [list]
                            set enepotvalaux [list]
                            set enebondvalaux [list]
                            set eneanglevalaux [list]
                            set enedihedralvalaux [list]
                            set enevdwvalaux [list]
                            set eneprevindex $tstep
                        }
                        set lineprev $line
                    }
                    
                    if {[lindex $QWIKMD::enetotpos end] != "" && $tot ==1} {
                        set QWIKMD::eneprevx [lindex $QWIKMD::enetotpos end]
                    } elseif {[lindex $QWIKMD::enekinpos end] != "" && $kin ==1} {
                        set QWIKMD::eneprevx [lindex $QWIKMD::enekinpos end]

                    } elseif {[lindex $QWIKMD::enepotpos end] != "" && $pot ==1} {
                        set QWIKMD::eneprevx [lindex $QWIKMD::enepotpos end]
                    } elseif {[lindex $QWIKMD::enebondpos end] != "" && $bond ==1} {
                        set QWIKMD::eneprevx [lindex $QWIKMD::enebondpos end]
                    } elseif {[lindex $QWIKMD::eneanglepos end] != "" && $angle ==1} {
                        set QWIKMD::eneprevx [lindex $QWIKMD::eneanglepos end]
                    } elseif {[lindex $QWIKMD::enedihedralpos end] != "" && $dihedral ==1} {
                        set QWIKMD::eneprevx [lindex $QWIKMD::enedihedralpos end]
                    } elseif {[lindex $QWIKMD::enevdwpos end] != "" && $vdw ==1} {
                        set QWIKMD::eneprevx [lindex $QWIKMD::enevdwpos end]
                    }   
                        
                    
                    close $logfile  
                 }
            }
            
            if {$QWIKMD::energyTotGui != ""} {
                $QWIKMD::energyTotGui clear
                $QWIKMD::energyTotGui add $QWIKMD::enetotpos $QWIKMD::enetotval
                $QWIKMD::energyTotGui replot
            }
            if {$QWIKMD::energyKineGui != ""} {
                $QWIKMD::energyKineGui clear
                $QWIKMD::energyKineGui add $QWIKMD::enekinpos $QWIKMD::enekinval
                $QWIKMD::energyKineGui replot
            }
            if {$QWIKMD::energyPotGui != ""} {
                $QWIKMD::energyPotGui clear
                $QWIKMD::energyPotGui add $QWIKMD::enepotpos $QWIKMD::enepotval
                $QWIKMD::energyPotGui replot
            }


            if {$QWIKMD::energyBondGui != ""} {
                $QWIKMD::energyBondGui clear
                $QWIKMD::energyBondGui add $QWIKMD::enebondpos $QWIKMD::enebondval
                $QWIKMD::energyBondGui replot
            }
            if {$QWIKMD::energyAngleGui != ""} {
                $QWIKMD::energyAngleGui clear
                $QWIKMD::energyAngleGui add $QWIKMD::eneanglepos $QWIKMD::eneangleval
                $QWIKMD::energyAngleGui replot
            }
            if {$QWIKMD::energyDehidralGui != ""} {
                $QWIKMD::energyDehidralGui clear
                $QWIKMD::energyDehidralGui add $QWIKMD::enedihedralpos $QWIKMD::enedihedralval
                $QWIKMD::energyDehidralGui replot
            }

            if {$QWIKMD::energyVdwGui != ""} {
                $QWIKMD::energyVdwGui clear
                $QWIKMD::energyVdwGui add $QWIKMD::enevdwpos $QWIKMD::enevdwval
                $QWIKMD::energyVdwGui replot
            }
            
            if {$print == 1} {
                puts $QWIKMD::textLogfile [QWIKMD::printEnergies $xtime $limit $energyfreq $const $tot $kin $pot $bond $angle $dihedral $vdw]
                flush $QWIKMD::textLogfile
            }
            
        } elseif {$QWIKMD::load == 0} {
            QWIKMD::EneCalc
        }

    } ] -row 0 -column 0 -sticky ens -pady 2 -padx 1

    
    QWIKMD::balloon $frame.header.fcolapse.plot [QWIKMD::enerCalcBL]
    

    grid [ttk::frame $frame.header.fcolapse.selection ] -row 1 -column 0 -sticky we -pady 1 -padx 2 
    grid columnconfigure $frame.header.fcolapse.selection 0 -weight 1
    grid columnconfigure $frame.header.fcolapse.selection 1 -weight 1
    grid columnconfigure $frame.header.fcolapse.selection 2 -weight 1

    grid [ttk::checkbutton $frame.header.fcolapse.selection.total -text "Total" -variable QWIKMD::enertotal] -row 0 -column 0 -sticky nsw -pady 2 -padx 4
    grid [ttk::checkbutton $frame.header.fcolapse.selection.kinetic -text "Kinetic" -variable QWIKMD::enerkinetic] -row 0 -column 1 -sticky nsw -pady 2 -padx 4 
    grid [ttk::checkbutton $frame.header.fcolapse.selection.potential -text "Potential" -variable QWIKMD::enerpoten ] -row 0 -column 2 -sticky nsw -pady 2 -padx 4

    QWIKMD::balloon $frame.header.fcolapse.selection.total  [QWIKMD::energyTotal]
    QWIKMD::balloon $frame.header.fcolapse.selection.kinetic  [QWIKMD::energyKinetic]
    QWIKMD::balloon $frame.header.fcolapse.selection.potential  [QWIKMD::energyPotential]

    grid [ttk::checkbutton $frame.header.fcolapse.selection.bond -text "Bond" -variable QWIKMD::enerbond ] -row 0 -column 3 -sticky nsw -pady 2 -padx 2
    grid [ttk::checkbutton $frame.header.fcolapse.selection.angle -text "Angle" -variable QWIKMD::enerangle ] -row 1 -column 0 -sticky nsw -pady 2 -padx 2
    grid [ttk::checkbutton $frame.header.fcolapse.selection.dihedral -text "Dihedral" -variable QWIKMD::enerdihedral ] -row 1 -column 1 -sticky nsw -pady 2 -padx 2
    grid [ttk::checkbutton $frame.header.fcolapse.selection.vdw -text "VDW" -variable QWIKMD::enervdw ] -row 1 -column 2 -sticky nsw -pady 2 -padx 2

    QWIKMD::balloon $frame.header.fcolapse.selection.bond  [QWIKMD::energyBond]
    QWIKMD::balloon $frame.header.fcolapse.selection.angle  [QWIKMD::energyAngle]
    QWIKMD::balloon $frame.header.fcolapse.selection.dihedral  [QWIKMD::energyDihedral]
    QWIKMD::balloon $frame.header.fcolapse.selection.vdw  [QWIKMD::energyVDW]

    grid forget $frame.header.fcolapse
}


proc QWIKMD::RenderFrame {frame} {

    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 0
    grid columnconfigure $frame.header 1 -weight 1
    grid [ttk::label $frame.header.lbtitle -text "Render"] -row 0 -column 0 -sticky w -pady 2 

    grid [ttk::combobox $frame.header.comboType -values "Image" -width 8 -state readonly ] -row 0 -column 1 -sticky w -pady 2 -padx 2
    $frame.header.comboType set "Image"

    QWIKMD::balloon $frame.header.comboType [QWIKMD::renderTypeBL]

    grid [ttk::button $frame.header.renedr -text "Render" -padding "2 2 2 2" -width 15 -command QWIKMD::RenderProc] -row 0 -column 2 -sticky w -pady 2 -padx 2
    QWIKMD::createInfoButton $frame.header 0 3
    bind $frame.header.info <Button-1> {
        set val [QWIKMD::renderInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow renderInfo [lindex $val 0] [lindex $val 2]
    }

    QWIKMD::balloon $frame.header.renedr [QWIKMD::renderBL]

    grid [ttk::frame $frame.res ] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.res 0 -weight 0
    grid columnconfigure $frame.res 1 -weight 0
    grid columnconfigure $frame.res 2 -weight 0
    grid columnconfigure $frame.res 3 -weight 0
    grid columnconfigure $frame.res 4 -weight 0
    grid columnconfigure $frame.res 5 -weight 0
    grid columnconfigure $frame.res 6 -weight 0

    grid [ttk::label $frame.res.lbres -text "Resolution" -padding "0 0 6 0"] -row 0 -column 0 -sticky w -pady 2
    
    set values [list [display get size] "1080p" "720p" "480p"]
    grid [ttk::combobox $frame.res.combores -width 12 -justify left -values $values -state readonly] -row 0 -column 3 -sticky w -pady 2
    $frame.res.combores set [lindex $values 0]
    bind $frame.res.combores <<ComboboxSelected>> {
        set comboVal [%W get]
        if {$comboVal == "1080p"} {
            display resize 1920 1080
        } elseif {$comboVal == "720p"} {
            display resize 1280 720
        } elseif {$comboVal == "480p"} {
            display resize 640 480
        } else {
            
            display resize [lindex $comboVal 0] [lindex $comboVal 1]
        }
        %W selection clear
    }

    QWIKMD::balloon $frame.res.combores [QWIKMD::renderResBL]

    grid [ttk::frame $frame.renderMode ] -row 2 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.renderMode 1 -weight 1
    set rendelistaux [render list]
    set rendelist {{Capture Display}}
    set textlength 0
    foreach str $rendelistaux {
        if {$str == "Tachyon"} {
            lappend rendelist "Background Render"
        } elseif {$str == "TachyonLOptiXInteractive"} {
            lappend rendelist "Interactive Render"
        }   
    }
    grid [ttk::label $frame.renderMode.lbmode -text "Image Processing"] -row 0 -column 0 -sticky w -pady 2 
    grid [ttk::combobox $frame.renderMode.comboMode -values $rendelist -width 20 -state readonly -textvariable QWIKMD::advGui(render,rendertype) -justify left] -row 0 -column 1 -sticky w -pady 2 -padx 2
    set QWIKMD::advGui(render,rendertype) [lindex $rendelist 0]
    bind $frame.renderMode.comboMode <<ComboboxSelected>> {
        %W selection clear
    }

    QWIKMD::balloon $frame.renderMode.comboMode [QWIKMD::renderRendBL]

}

proc QWIKMD::AdvancedAnalyzeFrame {frame} {
    grid [ttk::frame $frame.fp ] -row 0 -column 0 -sticky nsew -pady 2 -padx 2 
    grid columnconfigure $frame.fp 0 -weight 1
    grid rowconfigure $frame.fp 0 -weight 0
    grid rowconfigure $frame.fp 2 -weight 2
    set row 0
    grid [ttk::frame $frame.fp.general -relief groove] -row $row -column 0 -sticky nsew -pady 2 -padx 2 
    grid columnconfigure $frame.fp.general 0 -weight 1

    grid [ttk::frame $frame.fp.general.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.fp.general.header 0 -weight 1
    
    grid [ttk::frame $frame.fp.general.header.cmbbutt ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.fp.general.header.cmbbutt 0 -weight 1
    grid columnconfigure $frame.fp.general.header.cmbbutt 1 -weight 1
    grid columnconfigure $frame.fp.general.header.cmbbutt 2 -weight 1
    grid columnconfigure $frame.fp.general.header.cmbbutt 3 -weight 1

    grid [ttk::label $frame.fp.general.header.cmbbutt.lbtitle -text "Analysis"] -row 0 -column 0 -sticky w -pady 2 
    set values {"H Bonds" "SMD Forces" "RMSF" "SASA" "Contact Area" "Specific Heat" "Temperature Distribution" "MB Energy Distribution" "Temperature Quench"}

    grid [ttk::combobox $frame.fp.general.header.cmbbutt.comboAn -values $values -width 22 -state readonly -textvariable QWIKMD::advGui(analyze,advance,calcombo)] -row 0 -column 1 -sticky w -pady 2 -padx 2
    bind $frame.fp.general.header.cmbbutt.comboAn <<ComboboxSelected>> {
        QWIKMD::AdvancedSelected
        %W selection clear
    }

    QWIKMD::balloon $frame.fp.general.header.cmbbutt.comboAn [QWIKMD::advcComboAnBL]

    set QWIKMD::advGui(analyze,advance,calcombo) "H Bonds"

    grid [ttk::button $frame.fp.general.header.cmbbutt.calculate -text "Calculate" -padding "2 2 2 2" -width 15 -command QWIKMD::CalcAdvcAnalyze] -row 0 -column 2 -sticky e -pady 2 -padx 2
    set QWIKMD::advGui(analyze,advance,calcbutton) $frame.fp.general.header.cmbbutt.calculate
    QWIKMD::createInfoButton $frame.fp.general.header.cmbbutt 0 3
    # bind $frame.fp.general.header.cmbbutt.info <Button-1> {
    #   set val [QWIKMD::advAnalysisInfo]
    #   set QWIKMD::link [lindex $val 1]
    #   QWIKMD::infoWindow advAnalysisInfo [lindex $val 0] [lindex $val 2]
    # }

    incr row
    grid [ttk::frame $frame.fp.general.header.fcolapse] -row 1 -column 0 -sticky news -pady 4 -padx 2 
    grid columnconfigure $frame.fp.general.header.fcolapse 0 -weight 1

    set QWIKMD::advGui(analyze,advanceframe) $frame.fp.general.header.fcolapse
    incr row
    
    grid [ttk::frame $frame.fp.plot ] -row $row -column 0 -sticky nsew -pady 4 -padx 2 
    grid columnconfigure $frame.fp.plot 0 -weight 1

    QWIKMD::plotframe $frame.fp.plot advance
    QWIKMD::AdvancedSelected    
}

proc QWIKMD::CalcAdvcAnalyze {} {
    if {$QWIKMD::basicGui(live) == 1 && $QWIKMD::load == 0 && $QWIKMD::advGui(analyze,advance,calcombo) != "H Bonds" && $QWIKMD::advGui(analyze,advance,calcombo) != "SMD Forces"} {
        tk_messageBox -message "This option is only available after loading simulation results (load QwikMD input file *.qwikmd)" -title "Calculation Not Available" -icon warning
        return
    }
    if {$QWIKMD::sasarep != ""} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol
        set QWIKMD::sasarep ""
    }
    if {$QWIKMD::sasarepTotal1 != ""} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol
        set QWIKMD::sasarepTotal1 ""
    }
    if {$QWIKMD::sasarepTotal2 != ""} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::sasarepTotal2] $QWIKMD::topMol
        set QWIKMD::sasarepTotal2 ""
    }
    foreach m [molinfo list] {
        if {[string compare [molinfo $m get name] "{Color Scale Bar}"] == 0} {
          mol delete $m
        }
    }
    if {$QWIKMD::hbondsrepname != ""} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::hbondsrepname] $QWIKMD::topMol
        set QWIKMD::hbondsrepname ""
    }
    switch  $QWIKMD::advGui(analyze,advance,calcombo) {
        
        "H Bonds" {
            QWIKMD::callhbondsCalcProc
        }
        "SMD Forces" {
            QWIKMD::callSmdCalc
        }
        "RMSF" {
            QWIKMD::RMSFCalc
        }
        "SASA" {
            QWIKMD::callSASA
        }
        "Contact Area" {
            QWIKMD::callCSASA
        }
        "Specific Heat" {
            QWIKMD::SpecificHeatCalc
        }
        "Temperature Distribution" {
            QWIKMD::TempDistCalc
        }
        "MB Energy Distribution" {
            QWIKMD::MBCalC
        }
        "Temperature Quench" {
            QWIKMD::QTempCalc
        }   
    }
}

proc QWIKMD::AdvancedSelected {} {
    if {[winfo exists $QWIKMD::advGui(analyze,advanceframe).header]} {
        destroy $QWIKMD::advGui(analyze,advanceframe).header
    }
    set infobut $QWIKMD::topGui.nbinput.f4.fp.general.header.cmbbutt.info
    switch $QWIKMD::advGui(analyze,advance,calcombo) {
        "H Bonds" {
            QWIKMD::HBFrame
            bind $infobut <Button-1> {
                set val [QWIKMD::hbondInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow hbondInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "SMD Forces" {
            QWIKMD::SMDFrame
            bind $infobut <Button-1> {
                set val [QWIKMD::smdPlotInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow smdPlotInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "RMSF" {
            QWIKMD::RMSFFrame
            bind $infobut <Button-1> {
                set val [QWIKMD::rmsfInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow rmsfInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "SASA" {
            QWIKMD::SASAFrame noncontact
            bind $infobut <Button-1> {
                set val [QWIKMD::sasaPlotInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow sasaPlotInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "Contact Area" {
            QWIKMD::SASAFrame contact
            bind $infobut <Button-1> {
                set val [QWIKMD::nscaPlotInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow nscaPlotInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "Specific Heat" {
            QWIKMD::SpecificHeatFrame
            bind $infobut <Button-1> {
                set val [QWIKMD::specificHeatPlotInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow specificHeatPlotInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "Temperature Distribution" {
            QWIKMD::TDistFrame
            bind $infobut <Button-1> {
                set val [QWIKMD::tempDistPlotInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow tempDistPlotInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "Temperature Quench" {
            QWIKMD::TQuenchFrame
            bind $infobut <Button-1> {
                set val [QWIKMD::tQuenchPlotInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow tQuenchPlotInfo [lindex $val 0] [lindex $val 2]
            }
        }
        "MB Energy Distribution" {
            QWIKMD::MBDistributionFrame
            bind $infobut <Button-1> {
                set val [QWIKMD::mbDistributionPlotInfo]
                set QWIKMD::link [lindex $val 1]
                QWIKMD::infoWindow mbDistributionPlotInfo [lindex $val 0] [lindex $val 2]
            }
        }

    }
}

proc QWIKMD::HBFrame {} {


    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    
    grid [ttk::frame $frame.header.optframe] -row 0 -column 0 -sticky nswe

    grid columnconfigure $frame.header.optframe 0 -weight 1
    grid columnconfigure $frame.header.optframe 1 -weight 1
    grid columnconfigure $frame.header.optframe 2 -weight 1

    grid [ttk::radiobutton $frame.header.optframe.intra -text "Within Solute" -variable QWIKMD::hbondssel -value "intra"] -row 0 -column 1 -sticky nsw -pady 2 -padx 4 
    grid [ttk::radiobutton $frame.header.optframe.inter -text "Between Solute\nand Solvent" -variable QWIKMD::hbondssel -value "inter"] -row 0 -column 2 -sticky nsw -pady 2 -padx 4
    grid [ttk::radiobutton $frame.header.optframe.sel -text "Between Selections" -variable QWIKMD::hbondssel -value "sel"] -row 0 -column 3 -sticky nsw -pady 2 -padx 4

    QWIKMD::balloon $frame.header.optframe.intra [QWIKMD::hbondsSelWithinBL]
    QWIKMD::balloon $frame.header.optframe.inter [QWIKMD::hbondsSelintraBL]
    QWIKMD::balloon $frame.header.optframe.sel [QWIKMD::hbondsSelBetwSelBL]

    set QWIKMD::advGui(analyze,advance,interradio) $frame.header.optframe.inter
    grid [ttk::frame $frame.header.selection] -row 1 -column 0 -sticky nswe

    grid columnconfigure $frame.header.selection 1 -weight 1

    ttk::style configure hBondSel1.TEntry -foreground $QWIKMD::tempEntry

    grid [ttk::label $frame.header.selection.sel1 -text "Selection 1"] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.selection.entrysel1 -style hBondSel1.TEntry -textvariable QWIKMD::advGui(analyze,advance,hbondsel1entry) -validate focus -validatecommand {
        QWIKMD::checkSelection %W hBondSel1.TEntry
        set QWIKMD::hbondssel "sel"
        return 1
    }] -row 0 -column 1 -sticky ew -padx 2
    set QWIKMD::advGui(analyze,advance,hbondsel1entry) "Type Selection"
    ttk::style configure hBondSel2.TEntry -foreground $QWIKMD::tempEntry

    grid [ttk::label $frame.header.selection.sel2 -text "Selection 2"] -row 1 -column 0 -sticky w
    grid [ttk::entry $frame.header.selection.entrysel2 -style hBondSel2.TEntry -textvariable QWIKMD::advGui(analyze,advance,hbondsel2entry) -validate focus -validatecommand {
        QWIKMD::checkSelection %W hBondSel2.TEntry
        set QWIKMD::hbondssel "sel"
        if {$QWIKMD::advGui(analyze,advance,hbondsel1entry) == "Type Selection"} {
            set QWIKMD::advGui(analyze,advance,hbondsel1entry) "protein"
        }
        return 1
    }] -row 1 -column 1 -sticky ew -padx 2
    set QWIKMD::advGui(analyze,advance,hbondsel2entry) "Type Selection"

    if {[$QWIKMD::topGui.nbinput tab 0 -state] == "disabled"} {
        if {$QWIKMD::advGui(solvent,0) == "Implicit" || $QWIKMD::advGui(solvent,0) == "Vacuum" } {
            $frame.header.optframe.inter configure -state disable
        } else {
            $frame.header.optframe.inter configure -state normal
        }
    } else {
        if {$QWIKMD::basicGui(solvent,0) == "Implicit"} {
            $frame.header.optframe.inter configure -state disable
        } else {
            $frame.header.optframe.inter configure -state normal
        }
    }

    QWIKMD::balloon $frame.header.selection.entrysel1 [QWIKMD::hbondsSelWithinBL]
    QWIKMD::balloon $frame.header.selection.entrysel2 [QWIKMD::hbondsSelintraBL]
}


proc QWIKMD::SMDFrame {} {

    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    
    grid [ttk::frame $frame.header.optframe] -row 0 -column 0 -sticky nswe

    grid columnconfigure $frame.header.optframe 0 -weight 1
    grid columnconfigure $frame.header.optframe 1 -weight 1
    

    grid [ttk::label $frame.header.optframe.label -text "X Axis\nUnists"] -row 0 -column 0 -sticky nsw -pady 2 -padx 4 
    
    grid [ttk::radiobutton $frame.header.optframe.ft -text "Force vs Time" -variable QWIKMD::smdxunit -value "time" -command {
        if {$QWIKMD::smdGui != ""} {
            set QWIKMD::timeXsmd ""
            set QWIKMD::smdvals ""
            set QWIKMD::smdvalsavg ""
            $QWIKMD::smdGui configure -xlabel "Time (ns)" -title "Force vs Time"
            QWIKMD::callSmdCalc
        }
        }] -row 0 -column 1 -sticky nsw -pady 2 -padx 4 

    QWIKMD::balloon $frame.header.optframe.ft [QWIKMD::smdForceTimeBL]

    grid [ttk::radiobutton $frame.header.optframe.trace -text "Force vs Distance" -variable QWIKMD::smdxunit -value "distance" -command {
        if {$QWIKMD::smdGui != ""} {
            set QWIKMD::timeXsmd ""
            set QWIKMD::smdvals ""
            set QWIKMD::smdvalsavg ""
            $QWIKMD::smdGui configure -xlabel "Distance (A)" -title "Force vs Distance"
            QWIKMD::callSmdCalc
        }
        }] -row 0 -column 2 -sticky nsw -pady 2 -padx 4

    QWIKMD::balloon $frame.header.optframe.trace [QWIKMD::smdForceDistanceBL]
}

proc QWIKMD::RMSFFrame {} {
    set row 0
    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    
    
    grid [ttk::frame $frame.header.optframe] -row $row -column 0 -sticky nswe -pady 5
    incr row
    grid columnconfigure $frame.header.optframe 0 -weight 0
    grid columnconfigure $frame.header.optframe 1 -weight 1

    ttk::style configure RmsfSel.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::label $frame.header.optframe.lbor -text "Atom Selection: "] -row 0 -column 0 -sticky w -padx 2

    grid [ttk::entry $frame.header.optframe.entry -style RmsfSel.TEntry -textvariable QWIKMD::advGui(analyze,advance,rmsfselentry) -validate focus -validatecommand {
        QWIKMD::checkSelection %W RmsfSel.TEntry
        return 1
    }] -row 0 -column 1 -sticky ew -padx 2
    set QWIKMD::advGui(analyze,advance,rmsfselentry) "protein"

    QWIKMD::balloon $frame.header.optframe.entry [QWIKMD::rmsfGeneralSelectionBL]

    grid [ttk::frame $frame.header.align] -row $row -column 0 -sticky nswe -pady 5
    incr row
    grid columnconfigure $frame.header.align 3 -weight 1


    grid [ttk::checkbutton $frame.header.align.cAlign -text "Align Structure" -variable QWIKMD::advGui(analyze,advance,rmsfalicheck)] -row 0 -column 0 -sticky nsw -padx 2
    set QWIKMD::advGui(analyze,advance,rmsfalicheck) 0

    QWIKMD::balloon $frame.header.align.cAlign [QWIKMD::rmsfAlignBL]

    set values {"Backbone" "Alpha Carbon" "No Hydrogen" "All"}
    grid [ttk::combobox $frame.header.align.combo -values $values -width 12 -state readonly  -exportselection 0] -row 0 -column 1 -sticky nsw -padx 2
    $frame.header.align.combo set "Backbone"
    set QWIKMD::advGui(analyze,advance,rmsfaligncomb) "backbone"
    bind $frame.header.align.combo <<ComboboxSelected>> {
        set text [%W get]
        switch  $text {
            Backbone {
                set QWIKMD::advGui(analyze,advance,rmsfaligncomb) "backbone"
            }
            "Alpha Carbon" {
                set QWIKMD::advGui(analyze,advance,rmsfaligncomb) "alpha carbon"
            }
            "No Hydrogen" {
                set QWIKMD::advGui(analyze,advance,rmsfaligncomb) "noh"
            }
            "All" {
                set QWIKMD::advGui(analyze,advance,rmsfaligncomb) "all"
            }
            
        }
        %W selection clear
    }

    QWIKMD::balloon $frame.header.align.combo [QWIKMD::rmsfAlignSelection]

    grid [ttk::label $frame.header.align.lbor -text "or"] -row 0 -column 2 -sticky w -padx 5

    ttk::style configure RmsfSel.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::entry $frame.header.align.entry -style RmfdSel.TEntry -textvariable QWIKMD::advGui(analyze,advance,rmsfalignsel) -validate focus -validatecommand {
        QWIKMD::checkSelection %W RmfdSel.TEntry
        return 1
    }] -row 0 -column 3 -sticky ew -padx 2
    set QWIKMD::advGui(analyze,advance,rmsfalignsel) "Type Selection"

    QWIKMD::balloon $frame.header.align.entry [QWIKMD::rmsfGeneralAlignSelectionBL] 

    grid [ttk::frame $frame.header.frames] -row $row -column 0 -sticky nswe -pady 5
    incr row

    grid columnconfigure $frame.header.frames 2 -weight 1
    grid columnconfigure $frame.header.frames 4 -weight 1

    grid [ttk::label $frame.header.frames.ltext -text "Frame Selection:"] -row 0 -column 0 -sticky w -padx 2

    set QWIKMD::advGui(analyze,advance,rmsffrom) 0
    set QWIKMD::advGui(analyze,advance,rmsfto) 1
    if {$QWIKMD::load == 1} {
        set QWIKMD::advGui(analyze,advance,rmsfto) [expr [molinfo $QWIKMD::topMol get numframes] -1]
    }
    set QWIKMD::advGui(analyze,advance,rmsfskip) 1

    grid [ttk::label $frame.header.frames.lfrom -text "From:"] -row 0 -column 1 -sticky w -padx 2
    grid [ttk::entry $frame.header.frames.entryfrom -textvariable QWIKMD::advGui(analyze,advance,rmsffrom) -width 8 -validate focus -validatecommand {
        if {[string is integer -strict $QWIKMD::advGui(analyze,advance,rmsffrom)] == 0} {
            set QWIKMD::advGui(analyze,advance,rmsffrom) 0
        }
        if {$QWIKMD::advGui(analyze,advance,rmsfto) <= $QWIKMD::advGui(analyze,advance,rmsffrom)} {
            if {$QWIKMD::advGui(analyze,advance,rmsffrom) == [expr [molinfo $QWIKMD::topMol get numframes] -1] } {
                incr QWIKMD::advGui(analyze,advance,rmsffrom) -1
            } else {
                incr QWIKMD::advGui(analyze,advance,rmsfto)
            }
        }
        return 1
    }] -row 0 -column 2 -sticky we -padx 1
    set QWIKMD::advGui(analyze,advance,rmsffrom) 0

    QWIKMD::balloon $frame.header.frames.entryfrom [QWIKMD::rmsfInitFrameBL]

    grid [ttk::label $frame.header.frames.lto -text "To:"] -row 0 -column 3 -sticky w -padx 1
    grid [ttk::entry $frame.header.frames.entryto -textvariable QWIKMD::advGui(analyze,advance,rmsfto) -width 8  -validate focus -validatecommand {
        if {[string is integer -strict $QWIKMD::advGui(analyze,advance,rmsfto)] == 0} {
            set QWIKMD::advGui(analyze,advance,rmsfto)  [expr [molinfo $QWIKMD::topMol get numframes] -1]
        }
        
        if {$QWIKMD::advGui(analyze,advance,rmsfto) <= $QWIKMD::advGui(analyze,advance,rmsffrom) } {
            if {$QWIKMD::advGui(analyze,advance,rmsffrom) == [expr [molinfo $QWIKMD::topMol get numframes] -1] } {
                incr QWIKMD::advGui(analyze,advance,rmsffrom) -1
            } else {
                incr QWIKMD::advGui(analyze,advance,rmsfto)
            }
        }
        return 1
    }] -row 0 -column 4 -sticky we -padx 1
    if {$QWIKMD::load == 1} {
        set QWIKMD::advGui(analyze,advance,rmsfto) [expr [molinfo $QWIKMD::topMol get numframes] -1]
    }

    QWIKMD::balloon $frame.header.frames.entryto [QWIKMD::rmsfFinalFrameBL]

    grid [ttk::label $frame.header.frames.lskip -text "Skip:"] -row 0 -column 5 -sticky w -padx 1
    grid [ttk::entry $frame.header.frames.entryskip -textvariable QWIKMD::advGui(analyze,advance,rmsfskip) -width 8 -validate focus -validatecommand {
        if {[string is integer -strict $QWIKMD::advGui(analyze,advance,rmsfskip)] == 0} {
            set QWIKMD::advGui(analyze,advance,rmsfskip) 1
        }
        if {$QWIKMD::advGui(analyze,advance,rmsfskip) <= 0 || $QWIKMD::advGui(analyze,advance,rmsfskip) == ""} {
            set QWIKMD::advGui(analyze,advance,rmsfskip) 1
        }
        return 1
    }] -row 0 -column 6 -sticky w -padx 1
    set QWIKMD::advGui(analyze,advance,rmsfskip) 1

    incr row
    grid [ttk::frame $frame.header.rep] -row $row -column 0 -sticky nswe -pady 5

    QWIKMD::balloon $frame.header.frames.entryskip [QWIKMD::rmsfSkipFrameBL]

    grid columnconfigure $frame.header.rep 1 -weight 1
    grid columnconfigure $frame.header.rep 2 -weight 1

    grid [ttk::label $frame.header.rep.lrep -text "Representation"] -row 0 -column 0 -sticky w -padx 2
    set rep "Off NewCartoon QuickSurf Licorice VDW Lines Beads Points"
    grid [ttk::combobox $frame.header.rep.repcmb -values $rep -textvariable QWIKMD::advGui(analyze,advance,rmsfrep) -state readonly] -row 0 -column 1 -sticky w -padx 2
    set QWIKMD::advGui(analyze,advance,rmsfrep) NewCartoon
    bind $frame.header.rep.repcmb <<ComboboxSelected>> {
        if {$QWIKMD::rmsfrep != ""} {
            set rep $QWIKMD::advGui(analyze,advance,rmsfrep)
            if {$rep == "VDW"} {
                set rep "$rep 1.0 12.0"
            }
            mol modstyle [QWIKMD::getrepnum $QWIKMD::rmsfrep] $QWIKMD::topMol $rep
        }
        %W selection clear  
    }

    QWIKMD::balloon $frame.header.rep.repcmb [QWIKMD::rmsfRepBL]
}

proc QWIKMD::SASAFrame {opt} {
    set row 0
    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    
    grid [ttk::frame $frame.header.optframe] -row $row -column 0 -sticky nswe -pady 5
    incr row
    grid columnconfigure $frame.header.optframe 0 -weight 0
    grid columnconfigure $frame.header.optframe 1 -weight 1

    set lbltext "Atom Selection: "
    if {$opt == "contact"} {
        set lbltext "Selection 1: "
    }
    grid [ttk::label $frame.header.optframe.lsel -text $lbltext] -row 0 -column 0 -sticky w -padx 2

    ttk::style configure SASASel.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::entry $frame.header.optframe.entry -style SASASel.TEntry -textvariable QWIKMD::advGui(analyze,advance,sasaselentry) -validate focus -validatecommand {
        QWIKMD::checkSelection %W SASASel.TEntry
        return 1
    }] -row 0 -column 1 -sticky ew -padx 2
    set QWIKMD::advGui(analyze,advance,sasaselentry) "protein"
    if {$opt == "noncontact"} {
        QWIKMD::balloon $frame.header.optframe.entry [QWIKMD::sasaSel1BL]
    } else {
        QWIKMD::balloon $frame.header.optframe.entry [QWIKMD::sasaSel1ContactBL]
    }
    set lbltext "Restriction Selection: "
    if {$opt == "contact"} {
        set lbltext "Selection 2: "
    }
    grid [ttk::label $frame.header.optframe.lrestsel -text $lbltext] -row 1 -column 0 -sticky w -padx 2

    ttk::style configure SASARestSel.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::entry $frame.header.optframe.restentry -style SASARestSel.TEntry -textvariable QWIKMD::advGui(analyze,advance,sasarestselentry) -validate focus -validatecommand {
        QWIKMD::checkSelection %W SASARestSel.TEntry
        return 1
    }] -row 1 -column 1 -sticky ew -padx 2
    set QWIKMD::advGui(analyze,advance,sasarestselentry) "Type Selection"


    if {$opt == "noncontact"} {
        QWIKMD::balloon $frame.header.optframe.restentry [QWIKMD::sasaSel2BL]
    } else {
        QWIKMD::balloon $frame.header.optframe.restentry [QWIKMD::sasaSel2ContactBL]
    }
    grid [ttk::frame $frame.header.rep] -row $row -column 0 -sticky news
    incr row

    grid [ttk::label $frame.header.rep.lrep -text "Representation"] -row 0 -column 0 -sticky w -padx 2
    set rep "Off NewCartoon QuickSurf Surf Licorice VDW Lines Beads Points"
    grid [ttk::combobox $frame.header.rep.repcmb -values $rep -textvariable QWIKMD::advGui(analyze,advance,sasarep) -state readonly] -row 0 -column 1 -sticky w -padx 2
    set QWIKMD::advGui(analyze,advance,sasarep) NewCartoon
    bind $frame.header.rep.repcmb <<ComboboxSelected>> {
        if {$QWIKMD::sasarep != ""} {
            set rep $QWIKMD::advGui(analyze,advance,sasarep)
            if {$rep == "VDW"} {
                set rep "$rep 1.0 12.0"
            }
            mol modstyle [QWIKMD::getrepnum $QWIKMD::sasarep] $QWIKMD::topMol $rep
        }
        %W selection clear  
    }

    QWIKMD::balloon $frame.header.rep.repcmb [QWIKMD::sasaRepBL]

    grid [ttk::frame $frame.header.tbframe] -row $row -column 0 -sticky news
    incr row

    grid columnconfigure $frame.header.tbframe 0 -weight 1

    option add *Tablelist.activeStyle       frame
    
    set fro2 $frame.header.tbframe

    option add *Tablelist.movableColumns    no
    option add *Tablelist.labelCommand      tablelist::sortByColumn


        tablelist::tablelist $fro2.tb -columns {\
            0 "Res ID" center
            0 "Res NAME" center
            0 "Chain" center
            0 "SASA Avg(A\u00b2)" center
            0 "STDV" center
        }\
        -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] \
                -showseparators 0 -labelrelief groove  -labelbd 1 -selectforeground black\
                -foreground black -background white -state normal -selectmode extended -height 10 -stretch all -stripebackgroun white -exportselection true\
                

    $fro2.tb columnconfigure 0 -selectbackground cyan -sortmode dictionary -name ResdID -maxwidth 0
    $fro2.tb columnconfigure 1 -selectbackground cyan -sortmode dictionary -name ResdName -maxwidth 0
    $fro2.tb columnconfigure 2 -selectbackground cyan -sortmode dictionary -name Chain -maxwidth 0
    $fro2.tb columnconfigure 3 -selectbackground cyan -sortmode real -name Average -maxwidth 0
    $fro2.tb columnconfigure 4 -selectbackground cyan -sortmode real -name STDV -maxwidth 0

    grid $fro2.tb -row 0 -column 0 -sticky news 

    ##Scrool_BAr V
    scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
     grid $fro2.scr1 -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
    grid $fro2.scr2 -row 1 -column 0 -sticky swe

    set QWIKMD::advGui(analyze,advance,sasatb) $fro2.tb

    bind $fro2.tb <<TablelistSelect>>  {
        set sasaind [%W curselection]
        set index [list]
        if {$sasaind != ""} {
            if {$QWIKMD::sasarepTotal1 != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol
                set QWIKMD::sasarepTotal1 ""
            }
            if {$QWIKMD::sasarepTotal2 != ""} {
                mol delrep [QWIKMD::getrepnum $QWIKMD::sasarepTotal2] $QWIKMD::topMol
                set QWIKMD::sasarepTotal2 ""
            }
            foreach tbindex $sasaind {
                set compresid [%W cellcget $tbindex,0 -text] 
                set compchain [%W cellcget $tbindex,2 -text]
                if {[string match "*Total*" $compchain ] > 0} {
                    switch $QWIKMD::advGui(analyze,advance,calcombo) {
                        "SASA" {
                            set restrict $QWIKMD::advGui(analyze,advance,sasarestselentry)
                            if {$QWIKMD::advGui(analyze,advance,sasarestselentry) == "Type Selection" || $QWIKMD::advGui(analyze,advance,sasarestselentry) == ""} {
                                set restrict $QWIKMD::advGui(analyze,advance,sasaselentry)
                            }
                            mol addrep $QWIKMD::topMol
                            set QWIKMD::sasarepTotal1 [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
                            mol modcolor [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol "User"
                            mol modselect [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol "\($QWIKMD::advGui(analyze,advance,sasaselentry)\) and \($restrict\)"
                            mol modstyle [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol "Surf"
                            mol selupdate [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol on
                        }
                        "Contact Area" {     
                            if {$compchain == "Total1_2"} {
                                mol addrep $QWIKMD::topMol
                                set globalsel "\($QWIKMD::advGui(analyze,advance,sasaselentry)\)"
                                set restrictsel "\(within 5 of \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)\) and \($QWIKMD::advGui(analyze,advance,sasaselentry)\)"
                                set QWIKMD::sasarepTotal1 [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
                                mol modcolor [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol "User"
                                mol modselect [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol "same residue as \(\($globalsel\) and \($restrictsel\)\)"
                                mol modstyle [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol "Surf"
                                mol selupdate [QWIKMD::getrepnum $QWIKMD::sasarepTotal1] $QWIKMD::topMol on
                            }
                            if {$compchain == "Total2_1"} {
                                mol addrep $QWIKMD::topMol
                                set globalsel "\($QWIKMD::advGui(analyze,advance,sasarestselentry)\)"
                                set restrictsel "\(within 5 of \($QWIKMD::advGui(analyze,advance,sasaselentry)\)\) and \($QWIKMD::advGui(analyze,advance,sasarestselentry)\)"
                                set QWIKMD::sasarepTotal2 [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
                                mol modcolor [QWIKMD::getrepnum $QWIKMD::sasarepTotal2] $QWIKMD::topMol "User"
                                mol modselect [QWIKMD::getrepnum $QWIKMD::sasarepTotal2] $QWIKMD::topMol "same residue as \(\($globalsel\) and \($restrictsel\)\)"
                                mol modstyle [QWIKMD::getrepnum $QWIKMD::sasarepTotal2] $QWIKMD::topMol "Surf"
                                mol selupdate [QWIKMD::getrepnum $QWIKMD::sasarepTotal2] $QWIKMD::topMol on
                            }
                        }

                    }
                    continue
                } 
                set residids [$QWIKMD::selresTable searchcolumn 0 $compresid -all]
                set lines [$QWIKMD::selresTable get $residids]
                if {[llength [lindex $lines 0] ] == 1} {
                    set lines [list $lines]
                }
                set residids [$QWIKMD::selresTable searchcolumn 0 $compresid -all]
                set lines [$QWIKMD::selresTable get $residids]
                if {[llength [lindex $lines 0] ] == 1} {
                    set lines [list $lines]
                }
                lappend index [lindex $residids [lsearch -index 2 $lines $compchain]]
                
            }
            if {[llength $index] > 0} {
                $QWIKMD::selresTable selection set $index
                QWIKMD::rowSelection
                for {set i 1} {$i <= [llength $index]} { incr i} {
                    set repindex [expr [llength $QWIKMD::resrepname] - $i]
                    mol modcolor [QWIKMD::getrepnum [lindex [lindex $QWIKMD::resrepname $repindex] 1] ] $QWIKMD::topMol "User"
                    mol modstyle [QWIKMD::getrepnum [lindex [lindex $QWIKMD::resrepname $repindex] 1] ] $QWIKMD::topMol "Surf"
                }
            }
        }
        if {[llength $sasaind] > 0} {
            %W selection set $sasaind
        }
    }

    bind [$fro2.tb labeltag] <Any-Enter> {
        set col [tablelist::getTablelistColumn %W]
        set help 0
        switch $col {
            0 {
                set help [QWIKMD::ResidselTabelResidBL]
            }
            1 {
                set help [QWIKMD::ResidselTabelResnameBL]
            }
            2 {
                set help [QWIKMD::ResidselTabelChainBL]
            }
            3 {
                set help [QWIKMD::sasaTblSASABL]
            }
            4 {
                set help [QWIKMD::sasaTblSTDVBL]
            }
            default {
                set help $col
            }
        }
        after 1000 [list QWIKMD::balloon:show %W $help]
  
    }
    bind [$fro2.tb labeltag] <Any-Leave> "destroy %W.balloon"
}

proc QWIKMD::SpecificHeatFrame {} {
    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1

    grid [ttk::frame $frame.header.tableframe] -row 0 -column 0 -sticky nswe -padx 4

    grid columnconfigure $frame.header.tableframe 0 -weight 1
    grid rowconfigure $frame.header.tableframe 0 -weight 1

    set table [QWIKMD::addSelectTable $frame.header.tableframe 2]
    set QWIKMD::advGui(analyze,advance,SPH) $table
    if {$QWIKMD::confFile != ""} {
        for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
            if {[file exists $QWIKMD::outPath/run/[lindex $QWIKMD::confFile $i].dcd ]} {
                $table insert end "{} {}"
                set QWIKMD::radiobtt [lindex $QWIKMD::confFile $i]
                $table cellconfigure end,1 -text [lindex $QWIKMD::confFile $i]
                $table cellconfigure end,0 -window QWIKMD::Select
            }
        }
    }

    
    grid [ttk::frame $frame.header.optframe] -row 1 -column 0 -sticky nswe -padx 4
    grid columnconfigure $frame.header.optframe 0 -weight 1

    grid [ttk::frame $frame.header.optframe.tmpconst] -row 0 -column 0 -sticky nswe -padx 4
    grid columnconfigure $frame.header.optframe.tmpconst 0 -weight 1

    grid [ttk::frame $frame.header.optframe.tmpconst.tmp] -row 0 -column 0 -sticky w -pady 2
    grid columnconfigure $frame.header.optframe.tmpconst.tmp 0 -weight 0

    grid [ttk::label $frame.header.optframe.tmpconst.tmp.lblTEMP -text "Temperature"] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.tmpconst.tmp.tempentry -textvariable QWIKMD::advGui(analyze,advance,tempentry) -width 5] -row 0 -column 1 -sticky w 
    grid [ttk::label $frame.header.optframe.tmpconst.tmp.lblTEMPunit -text "C"] -row 0 -column 2 -sticky w
    set QWIKMD::advGui(analyze,advance,tempentry) 27

    QWIKMD::balloon $frame.header.optframe.tmpconst.tmp.tempentry [QWIKMD::spcfHeatTempBL]

    grid [ttk::frame $frame.header.optframe.tmpconst.const] -row 0 -column 1 -sticky e -padx 4
    grid columnconfigure $frame.header.optframe.tmpconst.const 0 -weight 0
    grid [ttk::label $frame.header.optframe.tmpconst.const.lblBK -text "Boltzmann k"] -row 0 -column 0 -sticky e
    grid [ttk::entry $frame.header.optframe.tmpconst.const.bkentry -textvariable QWIKMD::advGui(analyze,advance,bkentry) -width 12] -row 0 -column 1 -sticky e 
    grid [ttk::label $frame.header.optframe.tmpconst.const.lblBKUnit -text "kcal/mol*K"] -row 0 -column 2 -sticky w
    set QWIKMD::advGui(analyze,advance,bkentry) 0.00198657

    QWIKMD::balloon $frame.header.optframe.tmpconst.const.bkentry [QWIKMD::spcfHeatBKBL]

    grid [ttk::frame $frame.header.optframe.sel] -row 1 -column 0 -sticky nswe -pady 4
    grid columnconfigure $frame.header.optframe.sel 1 -weight 1
    grid [ttk::label $frame.header.optframe.sel.lblsel -text "Selection"] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.sel.bkentrysel -textvariable QWIKMD::advGui(analyze,advance,selentry) -width 7] -row 0 -column 1 -sticky we -padx 2
    set QWIKMD::advGui(analyze,advance,selentry) all

    QWIKMD::balloon $frame.header.optframe.sel.bkentrysel [QWIKMD::spcfHeatSelBL]

    grid [ttk::frame $frame.header.optframe.output] -row 2 -column 0 -sticky we -pady 2
    grid columnconfigure $frame.header.optframe.output 0 -weight 0

    
    grid [ttk::label $frame.header.optframe.output.lblBK -text "Specific Heat Results:"] -row 0 -column 0 -sticky e

    grid [ttk::frame $frame.header.optframe.output.kcal] -row 1 -column 0 -sticky w -pady 4
    grid [ttk::label $frame.header.optframe.output.kcal.lbunit -text "kcal/mol*K"] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.output.kcal.entryval -textvariable QWIKMD::advGui(analyze,advance,kcal) -width 12] -row 0 -column 1 -sticky e -padx 2 
    
    QWIKMD::balloon $frame.header.optframe.output.kcal.entryval [QWIKMD::spcfHeatResKcalBL]

    grid [ttk::frame $frame.header.optframe.output.joul] -row 1 -column 1 -sticky w -pady 4
    grid [ttk::label $frame.header.optframe.output.joul.lbunit -text "J/kg*C"] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.output.joul.entryval -textvariable QWIKMD::advGui(analyze,advance,joul) -width 12] -row 0 -column 1 -sticky e -padx 2 
    
    QWIKMD::balloon $frame.header.optframe.output.joul.entryval [QWIKMD::spcfHeatResJoulBL]
}

proc QWIKMD::TDistFrame {} {
    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1

    grid [ttk::frame $frame.header.tableframe] -row 0 -column 0 -sticky nswe -padx 4

    grid columnconfigure $frame.header.tableframe 0 -weight 1
    grid rowconfigure $frame.header.tableframe 0 -weight 1

    set table [QWIKMD::addSelectTable  $frame.header.tableframe 2]
    set QWIKMD::advGui(analyze,advance,tdist) $table
    if {$QWIKMD::confFile != ""} {
        for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
            if {[file exists $QWIKMD::outPath/run/[lindex $QWIKMD::confFile $i].dcd ]} {
                $table insert end "{} {}"
                set QWIKMD::radiobtt [lindex $QWIKMD::confFile $i]
                $table cellconfigure end,1 -text [lindex $QWIKMD::confFile $i]
                $table cellconfigure end,0 -window QWIKMD::Select
            }
        }
    }
    grid [ttk::frame $frame.header.optframe] -row 1 -column 0 -sticky nswe -padx 4
    grid columnconfigure $frame.header.optframe 1 -weight 1
    
    grid [ttk::label $frame.header.optframe.lblfitting -text "Curve fitting equation = "] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.fittingentry -state normal] -row 0 -column 1 -sticky we
    $frame.header.optframe.fittingentry delete 0 end
    $frame.header.optframe.fittingentry insert end "y= a0 * exp(-(x-a1)^2/a2)"
    $frame.header.optframe.fittingentry configure -state readonly

    QWIKMD::balloon $frame.header.optframe.fittingentry [QWIKMD::spcfTempDistEqBL]
}

proc QWIKMD::MBDistributionFrame {} {
    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1

    grid [ttk::frame $frame.header.tableframe] -row 0 -column 0 -sticky nswe -padx 4

    grid columnconfigure $frame.header.tableframe 0 -weight 1
    grid rowconfigure $frame.header.tableframe 0 -weight 1

    
    set table [QWIKMD::addSelectTable  $frame.header.tableframe 2]
    if {$QWIKMD::confFile != ""} {
        for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
            if {[file exists $QWIKMD::outPath/run/[lindex $QWIKMD::confFile $i].dcd ]} {
                $table insert end "{} {}"
                set QWIKMD::radiobtt [lindex $QWIKMD::confFile $i]
                $table cellconfigure end,1 -text [lindex $QWIKMD::confFile $i]
                $table cellconfigure end,0 -window QWIKMD::Select
            }        
        }
    }
    grid [ttk::frame $frame.header.optframe] -row 1 -column 0 -sticky nswe -padx 4
    grid columnconfigure $frame.header.optframe 1 -weight 1

    grid [ttk::label $frame.header.optframe.lblatmsel -text "Atom selection : "] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.atmselentry -state normal -textvariable QWIKMD::advGui(analyze,advance,MBsel)] -row 0 -column 1 -sticky we
    set QWIKMD::advGui(analyze,advance,MBsel) all

    QWIKMD::balloon $frame.header.optframe.atmselentry [QWIKMD::mbDistSelBL]

    grid [ttk::label $frame.header.optframe.lblfitting -text "Curve fitting equation = "] -row 1 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.fittingentry -state normal] -row 1 -column 1 -sticky we
    $frame.header.optframe.fittingentry delete 0 end
    $frame.header.optframe.fittingentry insert end "y = (2/ sqrt(Pi * a0^3)) * sqrt(x) * exp (-x / a0)"
    $frame.header.optframe.fittingentry configure -state readonly

    QWIKMD::balloon $frame.header.optframe.fittingentry [QWIKMD::mbDistEqBL]

}

proc QWIKMD::TQuenchFrame {} { 
    set frame $QWIKMD::advGui(analyze,advanceframe)
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    
    grid [ttk::frame $frame.header.tableframe] -row 0 -column 0 -sticky nswe -padx 4

    grid columnconfigure $frame.header.tableframe 0 -weight 1
    grid rowconfigure $frame.header.tableframe 0 -weight 1

    set table [QWIKMD::addSelectTable $frame.header.tableframe 3]

    $table configure -editstartcommand QWIKMD::StartQTempstep -editendcommand QWIKMD::EndQTempstep -editselectedonly true
    
    set QWIKMD::advGui(analyze,advance,qtmeptbl) $table
    if {$QWIKMD::confFile != ""} {
        for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
            if {[file exists $QWIKMD::outPath/run/[lindex $QWIKMD::confFile $i].dcd ]} {
                $table insert end "{} {} {}"
                $table cellconfigure end,0 -window QWIKMD::ProcSelect
                $table cellconfigure end,1 -text [lindex $QWIKMD::confFile $i]
                $table cellconfigure end,2 -text ""
            } 
        }   
    }

    grid [ttk::frame $frame.header.optframe] -row 1 -column 0 -sticky nswe -padx 4

    grid columnconfigure $frame.header.optframe 0 -weight 0
    grid columnconfigure $frame.header.optframe 1 -weight 0
    grid columnconfigure $frame.header.optframe 2 -weight 1
    grid columnconfigure $frame.header.optframe 3 -weight 0
    grid columnconfigure $frame.header.optframe 4 -weight 0
    grid columnconfigure $frame.header.optframe 5 -weight 0
    grid rowconfigure $frame.header.optframe 0 -weight 1

    grid [ttk::label $frame.header.optframe.lblACC -text "Autocorrelation\ndecay time"] -row 0 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.entry -textvariable QWIKMD::advGui(analyze,advance,decayentry) -width 7] -row 0 -column 1 -sticky w -padx 2
    set QWIKMD::advGui(analyze,advance,decayentry) 2.4

    QWIKMD::balloon $frame.header.optframe.entry [QWIKMD::tempQAtcrrTimeBL]

    grid [ttk::label $frame.header.optframe.lblTEMP -text "Initial Temperature"] -row 0 -column 3 -sticky w
    grid [ttk::entry $frame.header.optframe.tempentry -textvariable QWIKMD::advGui(analyze,advance,tempentry) -width 7] -row 0 -column 4 -sticky w -padx 2
    grid [ttk::label $frame.header.optframe.lblTEMPunit -text "C"] -row 0 -column 5 -sticky w
    set QWIKMD::advGui(analyze,advance,tempentry) 27

    QWIKMD::balloon $frame.header.optframe.tempentry [QWIKMD::tempQInitTempBL]

    grid [ttk::label $frame.header.optframe.lblechodepth -text "Echo depth = "] -row 1 -column 0 -sticky w
    grid [ttk::label $frame.header.optframe.lblechodepthval ] -row 1 -column 1 -sticky w
    set QWIKMD::advGui(analyze,advance,echolb) $frame.header.optframe.lblechodepthval

    QWIKMD::balloon $frame.header.optframe.lblechodepthval [QWIKMD::tempQTempDepthBL]

    grid [ttk::label $frame.header.optframe.lblechoref -text "Echo time = "] -row 1 -column 3 -sticky w
    grid [ttk::label $frame.header.optframe.lblechorefval ] -row 1 -column 4 -sticky w
    set QWIKMD::advGui(analyze,advance,echotime) $frame.header.optframe.lblechorefval

    QWIKMD::balloon $frame.header.optframe.lblechorefval [QWIKMD::tempQTempTimeBL]

    grid [ttk::label $frame.header.optframe.lblfitting -text "Curve fitting equation = "] -row 2 -column 0 -sticky w
    grid [ttk::entry $frame.header.optframe.fittingentry -state normal -width 16 ] -row 2 -column 1 -sticky w
    $frame.header.optframe.fittingentry delete 0 end
    $frame.header.optframe.fittingentry insert end "y = exp(-x/a0)"
    $frame.header.optframe.fittingentry configure -state readonly

    QWIKMD::balloon $frame.header.optframe.fittingentry [QWIKMD::tempQTempEqBL]

    grid [ttk::button $frame.header.optframe.lblFEcho -text "Find Echo" -command QWIKMD::QFindEcho -padding "2 0 2 0"] -row 2 -column 4 -sticky e -padx 2 -pady 2
}

proc QWIKMD::StartQTempstep {tbl row col text} {
    set from 1
    set to 500000
    set w [$tbl editwinpath]
    $w configure -from $from -to $to -increment 1
}
proc QWIKMD::ProcSelect {tbl row col w} {
    grid [ttk::frame $w] -sticky news
    ttk::style configure selec.TCheckbutton -background white
    grid [ttk::checkbutton $w.r -style selec.TCheckbutton] -row 0 -column 0
    $w.r invoke
    $w.r state !selected
    return $w.r
}

proc QWIKMD::checkSelection {w style} {
    set returnval 1
    set text [$w get]
    set sel ""
    set table $QWIKMD::selresTable 
    if {$text == "Type Selection"} {
        $w delete 0 end
        ttk::style configure RmsdSel.TEntry -foreground black
        if {$style == "AtomSel.TEntry"} {
            #$QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state disable
            $table selection clear 0 end
            QWIKMD::SelResClearSelection
        }
        return 0
    } elseif {$text == ""} {
        set returnval 0
    } else {
        
        set aux [catch {atomselect $QWIKMD::topMol $text} sel]
        if {$aux == 1} {
            tk_messageBox -message "Atom selection invalid." -icon error -type ok
            set returnval 0
        } else {
            if {[llength [$sel get resid]] == 0} {
                tk_messageBox -message "0 atoms selected. Please choose one or more atoms" -icon warning -type ok
                set returnval 0
            }
        }
        QWIKMD::SelResClearSelection
    }
    if {$returnval == 0} {
        $w delete 0 end
        $w insert end "Type Selection"
        ttk::style configure $style -foreground $QWIKMD::tempEntry
        return $returnval
    } else {
        ttk::style configure $style -foreground black
    }

    if {$style == "AtomSel.TEntry"} {
        #$QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state normal
        if {[$w get] != "Type Selection" && $returnval == 1} {
            set QWIKMD::selResidSelIndex [list]
            foreach resid [$sel get resid] chain [$sel get chain] {
                if {[lsearch $QWIKMD::selResidSelIndex ${resid}_$chain] == -1} {
                    lappend QWIKMD::selResidSelIndex ${resid}_$chain
                }
            }
            $table selection clear 0 end
            QWIKMD::selResidForSelection [wm title $QWIKMD::selResGui] $QWIKMD::selResidSelIndex
        }
    }
    if {$sel != "" && $returnval == 0} {
        $sel delete
    }
    return $returnval
}

proc QWIKMD::addplot {frame tadbtitle title xlab ylab} {
    set tabid [$QWIKMD::topGui.nbinput index current]
    set frameaux ""
    set framesection ""
    if {$tabid == 2} {
        set frameaux $QWIKMD::advGui(analyze,basic,ntb).$frame
        set framesection $QWIKMD::advGui(analyze,basic,ntb)
    } else {
        set frameaux $QWIKMD::advGui(analyze,advance,ntb).$frame
         set framesection $QWIKMD::advGui(analyze,advance,ntb)
    }

    set plotsection [file root [file root [file root $framesection]]]
    set arrow [lindex [${plotsection}.prt cget -text] 0]
    if {$arrow == $QWIKMD::rightPoint} {
        QWIKMD::hideFrame ${plotsection}.prt $plotsection "Plots"
    }
    if {[winfo exists $frameaux] != 1} {
        ttk::frame $frameaux
        grid columnconfigure $frameaux 0 -weight 1
        grid rowconfigure $frameaux 0 -weight 1
        set tabid [$QWIKMD::topGui.nbinput index current]
        if {$tabid == 2} {
            set level basic
        } else {
            set level advance
        }
        $QWIKMD::advGui(analyze,$level,ntb) add $frameaux -text $tadbtitle -sticky news

        grid [ttk::frame $frameaux.eplot] -row 0 -column 0 -sticky news
        grid columnconfigure $frameaux.eplot 0 -weight 1
        grid rowconfigure $frameaux.eplot 0 -weight 1

    }

    set plot [multiplot embed $frameaux.eplot -xsize 600 -ysize 400 -title $title -xlabel $xlab -ylabel $ylab -lines -linewidth 2 -marker point -radius 2 -autoscale  ]
    set plotwindow [$plot getpath]

    menubutton $plotwindow.menubar.clear -text "Clear" \
    -underline 0 -menu $plotwindow.menubar.clear.menu
        
    $plotwindow.menubar.clear config -width 5

    menu $plotwindow.menubar.clear.menu -tearoff 0

    $plotwindow.menubar.clear.menu add command -label "Clear Plot"


    menubutton $plotwindow.menubar.close -text Close -underline 0 -menu $plotwindow.menubar.close.menu

    menu $plotwindow.menubar.close.menu -tearoff 0
    $plotwindow.menubar.close.menu add command -label "Close Plot"
    

    $plotwindow.menubar.close config -width 5


    pack $plotwindow.menubar.clear -side left
    pack $plotwindow.menubar.close -side left
    grid $plotwindow -row 0 -column 0 -sticky nwes
        
    return "$plot $plotwindow.menubar.clear.menu $plotwindow.menubar.close.menu"
}

proc QWIKMD::Select {tbl row col w} {
    grid [ttk::frame $w] -sticky news
    
    ttk::style configure select.TRadiobutton -background white
    grid [ttk::radiobutton $w.r -value [$tbl cellcget $row,[expr $col +1] -text] -style select.TRadiobutton -variable QWIKMD::radiobtt] -row 0 -column 0
    return $w.r
}

proc QWIKMD::StartQTempstep {tbl row col text} {
    return $text
}

proc QWIKMD::EndQTempstep {tbl row col text} {
    return $text
}
proc QWIKMD::addSelectTable {frame number} {
    set fro2 $frame
    option add *Tablelist.activeStyle       frame
    
    option add *Tablelist.movableColumns    no
    #option add *Tablelist.labelCommand      tablelist::sortByColumn


        tablelist::tablelist $fro2.tb
        $fro2.tb configure -columns {0 "Select" center 0 "Name" center}
        if {$number > 2} {
            $fro2.tb configure -columns {0 "Select" center 0 "Name" center 0 "tau" center}
        }       
        $fro2.tb configure -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] \
                -showseparators 0 -labelrelief groove  -labelbd 1 -selectforeground black\
                -foreground black -background white -state normal -selectmode extended -height 5 -stretch all -stripebackgroun white -exportselection true\
                

    $fro2.tb columnconfigure 0 -selectbackground cyan
    $fro2.tb columnconfigure 1 -selectbackground cyan

    $fro2.tb columnconfigure 0 -sortmode integer -name Select
    $fro2.tb columnconfigure 1 -sortmode dictionary -name Name

    $fro2.tb columnconfigure 0 -width 0 -maxwidth 0
    $fro2.tb columnconfigure 1 -width 0 -maxwidth 0

    grid $fro2.tb -row 0 -column 0 -sticky news


    if {$number > 2} {
        $fro2.tb columnconfigure 2 -selectbackground cyan
        $fro2.tb columnconfigure 2 -sortmode dictionary -name tau
        $fro2.tb columnconfigure 2 -width 0 -maxwidth 0
    }   

    ##Scrool_BAr V
    scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
     grid $fro2.scr1 -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
    
    grid $fro2.scr2 -row 1 -column 0 -sticky swe

    bind [$fro2.tb labeltag] <Any-Enter> {
        set col [tablelist::getTablelistColumn %W]
        set help 0
        switch $col {
            0 {
                set help [QWIKMD::selectTbSelectBL]
            }
            1 {
                set help [QWIKMD::selectTbNameBL]
            }
            2 {
                set help [QWIKMD::selectTbTauBL]
            }
            default {
                set help $col
            }
        }
        after 1000 [list QWIKMD::balloon:show %W $help]
    }
    
    bind [$fro2.tb labeltag] <Any-Leave> "destroy %W.balloon"
    return $fro2.tb

}


proc QWIKMD::plotframe {frame level} {
    grid [ttk::frame $frame.header ] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header 0 -weight 1
    grid [ttk::label $frame.header.prt -text "$QWIKMD::rightPoint Plots"] -row 0 -column 0 -sticky ew -pady 1

    bind $frame.header.prt <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Plots"
    }

    grid [ttk::frame $frame.header.fcolapse ] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frame.header.fcolapse 0 -weight 1

    grid [ttk::frame $frame.header.fcolapse.sep ] -row 0 -column 0 -sticky ew
    grid columnconfigure $frame.header.fcolapse.sep 0 -weight 1
    grid [ttk::separator $frame.header.fcolapse.spt -orient horizontal] -row 0 -column 0 -sticky ew -pady 0

    grid [ttk::frame $frame.header.fcolapse.fntb] -row 1 -column 0 -sticky news -padx 0 -pady 2
    grid columnconfigure $frame.header.fcolapse.sep 0 -weight 1

    grid [ttk::notebook $frame.header.fcolapse.fntb.ntb  -padding "0 0 0 0"] -row 0 -column 0 -sticky news -padx 0
    set QWIKMD::advGui(analyze,$level,ntb) $frame.header.fcolapse.fntb.ntb  
    grid forget $frame.header.fcolapse
    lappend QWIKMD::notebooks $frame.header.fcolapse.fntb.ntb
    
}
proc QWIKMD::killIMD {} {
    catch {imd kill}
    trace vdelete ::vmd_timestep($QWIKMD::topMol) w ::QWIKMD::updateMD 
}

proc QWIKMD::Finish {} {
    QWIKMD::killIMD
    if {$QWIKMD::state != [llength $QWIKMD::confFile]} {
        QWIKMD::updateMD
    } 
    set inputname [lindex $QWIKMD::confFile [expr $QWIKMD::state -1]]
    set fil [open $inputname.check w+]
    
    set done 1
    if {[file exists $inputname.restart.coor] != 1 || [file exists $inputname.restart.vel] != 1  || [file exists $inputname.restart.xsc] != 1  } {
        if {$QWIKMD::run == "SMD"} {
            if {[file exists $inputname.coor] != 1 } {
                set done 0
            } else {
                set done 1
            }
            
        } else {
            set done 0
        }
    } else {
        set done 1
    }

    ############################################################
    ## Save tha last x axis value of the plots for simulation 
    ## restart purpose and in case of inputfile load
    ############################################################

    if {$done == 1} {
        puts $fil "DONE"
        if {[llength $QWIKMD::rmsd] > 0} {
            set QWIKMD::lastrmsd [expr [llength $QWIKMD::rmsd] -1]
        }

        if {[llength $QWIKMD::hbonds] > 0} {
            set QWIKMD::lasthbond [expr [llength $QWIKMD::hbonds] -1]
        }

        if {[llength $QWIKMD::smdvalsavg] > 0} {
            set QWIKMD::lastsmd [expr [llength $QWIKMD::smdvalsavg] -1]
        }

        if {[llength $QWIKMD::enetotval] > 0} {
            set QWIKMD::lastenetot [expr [llength $QWIKMD::enetotval] -1]
        }

        if {[llength $QWIKMD::enekinval] > 0} {
            set QWIKMD::lastenekin [expr [llength $QWIKMD::enekinval] -1]
        }

        if {[llength $QWIKMD::enepotval] > 0} {
            set QWIKMD::lastenepot [expr [llength $QWIKMD::enepotval] -1]
        }

        if {[llength $QWIKMD::enebondval] > 0} {
            set QWIKMD::lastenebond [expr [llength $QWIKMD::enebondval] -1]
        }

        if {[llength $QWIKMD::eneangleval] > 0} {
            set QWIKMD::lasteneangle [expr [llength $QWIKMD::eneangleval] -1]
        }

        if {[llength $QWIKMD::enedihedralval] > 0} {
            set QWIKMD::lastenedihedral [expr [llength $QWIKMD::enedihedralval] -1]
        }

        if {[llength $QWIKMD::enevdwval] > 0} {
            set QWIKMD::lastenevdw [expr [llength $QWIKMD::enevdwval] -1]
        }

        if {[llength $QWIKMD::tempval] > 0} {
            set QWIKMD::lasttemp [expr [llength $QWIKMD::tempval] -1]
        }
        if {[llength $QWIKMD::pressval] > 0} {
            set QWIKMD::lastpress [expr [llength $QWIKMD::pressvalavg] -1]
        }

        if {[llength $QWIKMD::volval] > 0} {
            set QWIKMD::lastvol [expr [llength $QWIKMD::volvalavg] -1]
        }
        
    } else {
        puts $fil "One or more files filed to be written"
        
        tk_messageBox -message "One or more files failed to be written. The new simulation ready to run is [lindex $QWIKMD::confFile [expr $QWIKMD::state -1]]" -title "Running Simulation" -icon info -type ok
        
        ############################################################
        ## Delete values from the "failed" simulation
        ############################################################

        if {[llength $QWIKMD::rmsd] > 0} {
            set QWIKMD::rmsd [lrange $QWIKMD::rmsd 0 $QWIKMD::lastrmsd]
            set QWIKMD::timeXrmsd [lrange $QWIKMD::timeXrmsd 0 $QWIKMD::lastrmsd]
        }

        if {[llength $QWIKMD::hbonds] > 0} {
            set QWIKMD::hbonds [lrange $QWIKMD::hbonds 0 $QWIKMD::lasthbond]
            set QWIKMD::timeXhbonds [lrange $QWIKMD::timeXhbonds 0 $QWIKMD::lasthbond]
            
        }

        if {[llength $QWIKMD::smdvalsavg] > 0} {
            set QWIKMD::smdvalsavg [lrange $QWIKMD::smdvalsavg 0 $QWIKMD::lastsmd]
            set QWIKMD::timeXsmd [lrange $QWIKMD::timeXsmd 0 $QWIKMD::lastsmd]
        }

        if {[llength $QWIKMD::enetotval] > 0} {
            set QWIKMD::enetotval [lrange $QWIKMD::enetotval 0 $QWIKMD::lastenetot]
            set QWIKMD::enetotpos [lrange $QWIKMD::enetotpos 0 $QWIKMD::lastenetot]
        }

        if {[llength $QWIKMD::enekinval] > 0} {
            set QWIKMD::enekinval [lrange $QWIKMD::enekinval 0 $QWIKMD::lastenekin]
            set QWIKMD::enekinpos [lrange $QWIKMD::enekinpos 0 $QWIKMD::lastenekin]
        }

        if {[llength $QWIKMD::enepotval] > 0} {
            set QWIKMD::enepotval [lrange $QWIKMD::enepotval 0 $QWIKMD::lastenepot]
            set QWIKMD::enepotpos [lrange $QWIKMD::enepotpos 0 $QWIKMD::lastenepot]
        }

        if {[llength $QWIKMD::enebondval] > 0} {
            set QWIKMD::enebondval [lrange $QWIKMD::enebondval 0 $QWIKMD::lastenebond]
            set QWIKMD::enebondpos [lrange $QWIKMD::enebondpos 0 $QWIKMD::lastenebond]
        }

        if {[llength $QWIKMD::eneangleval] > 0} {
            set QWIKMD::eneangleval [lrange $QWIKMD::eneangleval 0 $QWIKMD::lasteneangle]
            set QWIKMD::eneanglepos [lrange $QWIKMD::eneanglepos 0 $QWIKMD::lasteneangle]
        }

        if {[llength $QWIKMD::enedihedralval] > 0} {
            set QWIKMD::enedihedralval [lrange $QWIKMD::enedihedralval 0 $QWIKMD::lastenedihedral]
            set QWIKMD::enedihedralpos [lrange $QWIKMD::enedihedralpos 0 $QWIKMD::lastenedihedral]
        }

        if {[llength $QWIKMD::enevdwval] > 0} {
            set QWIKMD::enevdwval [lrange $QWIKMD::enevdwval 0 $QWIKMD::lastenevdw]
            set QWIKMD::enevdwpos [lrange $QWIKMD::enevdwpos 0 $QWIKMD::lastenevdw]
        }

        if {[llength $QWIKMD::tempval] > 0} {
            set QWIKMD::tempval [lrange $QWIKMD::tempval 0 $QWIKMD::lasttemp]
            set QWIKMD::temppos [lrange $QWIKMD::temppos 0 $QWIKMD::lasttemp]
        }
        if {[llength $QWIKMD::pressval] > 0} {
            set QWIKMD::pressvalavg [lrange $QWIKMD::pressvalavg 0 $QWIKMD::lastpress]
            set QWIKMD::presspos [lrange $QWIKMD::presspos 0 $QWIKMD::lastpress]
        }

        if {[llength $QWIKMD::volval] > 0} {
            set QWIKMD::volvalavg [lrange $QWIKMD::volvalavg 0 $QWIKMD::lastvol]
            set QWIKMD::volpos [lrange $QWIKMD::volpos 0 $QWIKMD::lastvol]
        }
        file delete $inputname.check
        file delete $inputname.log
        if {$QWIKMD::state > 0} {
            incr QWIKMD::state -1
        }
        
    }
    set tabid [$QWIKMD::topGui.nbinput index current]
    set QWIKMD::prevcounterts $QWIKMD::counterts
     if {$QWIKMD::run == "SMD"} {
        set do 0
        if {$tabid == 0} {
            if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,smd) == 1} {
                set do 1
            }
        } else {
            if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,smd) == 1} {
                set do 1
            }
        }
        if {$do == 1} {
            set QWIKMD::prevcountertsmd $QWIKMD::countertssmd
        }  
    }

    $QWIKMD::runbtt configure -state normal
    $QWIKMD::runbtt configure -text "Start [QWIKMD::RunText]"


    $QWIKMD::basicGui(preparebtt,$tabid) configure -state normal

    
    close $fil
    set QWIKMD::enecurrentpos 0
    set QWIKMD::smdcurrentpos 0
    set QWIKMD::condcurrentpos 0
    
    set QWIKMD::stop 1
}

proc QWIKMD::Detach {} {
    imd detach
    trace vdelete ::vmd_timestep($QWIKMD::topMol) w ::QWIKMD::updateMD 
}

proc QWIKMD::Pause {} {
    imd pause toggle
    set tabid [expr [$QWIKMD::topGui.nbinput index current] +1]
    if {$QWIKMD::stop == 1} {
        $QWIKMD::topGui.nbinput.f$tabid.fcontrol.fcolapse.f1.imd.button_Pause configure -text "Resume"
        set QWIKMD::stop 0
    } else {
        set QWIKMD::stop 1
        $QWIKMD::topGui.nbinput.f$tabid.fcontrol.fcolapse.f1.imd.button_Pause configure -text "Pause"
    }
}

############################################################
## Residue Selection window builder. This window will become 
## a general strucutre manipulation window in the next versions
############################################################

proc QWIKMD::SelResidBuild {} {
    set tabid [$QWIKMD::topGui.nbinput index current]
    if {[winfo exists $QWIKMD::selResGui] != 1} {
        toplevel $QWIKMD::selResGui
    } else {
        wm deiconify $QWIKMD::selResGui
        raise $QWIKMD::selResGui
        return
    }   

    
    grid columnconfigure $QWIKMD::selResGui 0 -weight 2
    grid rowconfigure $QWIKMD::selResGui 0 -weight 2
    ## Title of the windows
    wm title $QWIKMD::selResGui "Structure Manipulation/Check" 

    wm protocol $QWIKMD::selResGui WM_DELETE_WINDOW {
        set QWIKMD::anchorpulling 0
        set QWIKMD::buttanchor 0
        if {$QWIKMD::topMol != ""} {
            $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butClear invoke
        }
        wm withdraw $QWIKMD::selResGui
        QWIKMD::tableModeProc
        trace remove variable ::vmd_pick_event write QWIKMD::ResidueSelect
        mouse mode rotate
      }

    grid [ttk::frame $QWIKMD::selResGui.f1] -row 0 -column 0 -sticky nsew -padx 2 -pady 4
    grid columnconfigure $QWIKMD::selResGui.f1 0 -weight 0
    #grid columnconfigure $QWIKMD::selResGui.f1 1 -weight 1
    grid rowconfigure $QWIKMD::selResGui.f1 0 -weight 2
    #grid rowconfigure $QWIKMD::selResGui.f1 1 -weight 1

    grid [ttk::frame $QWIKMD::selResGui.f1.fcol1]  -row 0 -column 0 -sticky nsew -padx 2
    grid columnconfigure $QWIKMD::selResGui.f1.fcol1 0 -weight 1
    #grid columnconfigure $QWIKMD::selResGui.f1.fcol1 1 -weight 1
    grid rowconfigure $QWIKMD::selResGui.f1.fcol1 0 -weight 3 
    grid rowconfigure $QWIKMD::selResGui.f1.fcol1 1 -weight 1
    set selframe "$QWIKMD::selResGui.f1.fcol1"

    grid [ttk::frame $selframe.tableframe] -row 0 -column 0 -sticky nswe -padx 4

    grid columnconfigure $selframe.tableframe 0 -weight 1
    grid rowconfigure $selframe.tableframe 0 -weight 1
    set fro2 $selframe.tableframe
    option add *Tablelist.activeStyle       frame
    
    option add *Tablelist.movableColumns    no
    option add *Tablelist.labelCommand      tablelist::sortByColumn


        tablelist::tablelist $fro2.tb \
        -columns { 0 "Res ID"    center
                0 "Res NAME"     center
                0 "Chain" center
                0 "Type" center
                } \
                -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] \
                -showseparators 0 -labelrelief groove  -labelbd 1 -selectforeground black\
                -foreground black -background white -state normal -selectmode extended -stretch "all" -width 45 -stripebackgroun white -exportselection true\
                -editstartcommand QWIKMD::createResCombo -editendcommand QWIKMD::CallUpdateRes 

    $fro2.tb columnconfigure 0 -selectbackground cyan
    $fro2.tb columnconfigure 1 -selectbackground cyan
    $fro2.tb columnconfigure 2 -selectbackground cyan

    $fro2.tb columnconfigure 0 -sortmode integer -name ResID
    $fro2.tb columnconfigure 1 -sortmode dictionary -name ResNAME
    $fro2.tb columnconfigure 2 -sortmode dictionary -name Chain
    $fro2.tb columnconfigure 3 -sortmode dictionary -name Type
    
    $fro2.tb columnconfigure 0 -width 0 -maxwidth 0
    $fro2.tb columnconfigure 1 -width 0 -maxwidth 0 -editable true -editwindow ttk::combobox
    $fro2.tb columnconfigure 2 -width 0 -maxwidth 0
    $fro2.tb columnconfigure 3 -width 0 -maxwidth 0 -editable true -editwindow ttk::combobox


    grid $fro2.tb -row 0 -column 0 -sticky news
    $fro2.tb configure -height 35
    set QWIKMD::selresTable $fro2.tb

    ##Scrool_BAr V
    scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
     grid $fro2.scr1 -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
    grid $fro2.scr2 -row 1 -column 0 -sticky swe

    bind [$fro2.tb labeltag] <Any-Enter> {
        set col [tablelist::getTablelistColumn %W]
        set help 0
        switch $col {
            0 {
                set help [QWIKMD::ResidselTabelResidBL]
            }
            1 {
                set help [QWIKMD::ResidselTabelResnameBL]
            }
            2 {
                set help [QWIKMD::ResidselTabelChainBL]
            }
            3 {
                set help [QWIKMD::ResidselTabelTypeBL]
            }
            default {
                set help $col
            }
        }
        after 1000 [list QWIKMD::balloon:show %W $help]
  
        
    }
    bind [$fro2.tb labeltag] <Any-Leave> "destroy %W.balloon"

    grid [ttk::frame $selframe.patchframe] -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    grid [ttk::frame $selframe.patchframe.header] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $selframe.patchframe.header 0 -weight 1

    grid [ttk::label $selframe.patchframe.header.lbtitle -text "$QWIKMD::rightPoint Modifications (Patches) List"] -row 0 -column 0 -sticky nswe -pady 2 -padx 2  
    ttk::frame $selframe.patchframe.empty
    grid [ttk::labelframe $selframe.patchframe.header.fcolapse -labelwidget $selframe.patchframe.empty] -row 1 -column 0 -sticky ews -padx 2
    grid columnconfigure $selframe.patchframe.header.fcolapse 0 -weight 1

    bind $selframe.patchframe.header.lbtitle <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Modifications (Patches) List"
    }
    
    grid [ttk::label $selframe.patchframe.header.fcolapse.format -text "NAME CHAIN1 RES1 CHAIN2 RES2\nNAME CHAIN1 RES1 CHAIN2 RES2"] -row 1 -column 0 -sticky wns -padx 2 -pady 2
    grid [tk::text $selframe.patchframe.header.fcolapse.text -font tkconfixed -wrap none -bg white -height 4 -width 45 -font TkFixedFont -relief flat -foreground black \
    -yscrollcommand [list $selframe.patchframe.header.fcolapse.scr1 set] -xscrollcommand [list $selframe.patchframe.header.fcolapse.scr2 set]] -row 2 -column 0 -sticky wens
        ##Scrool_BAr V
    scrollbar $selframe.patchframe.header.fcolapse.scr1  -orient vertical -command [list $selframe.patchframe.header.fcolapse.text yview]
    grid $selframe.patchframe.header.fcolapse.scr1  -row 2 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $selframe.patchframe.header.fcolapse.scr2  -orient horizontal -command [list $selframe.patchframe.header.fcolapse.text xview]
    grid $selframe.patchframe.header.fcolapse.scr2 -row 3 -column 0 -sticky swe

    set QWIKMD::selresPatcheFrame $selframe.patchframe
    set QWIKMD::selresPatcheText $selframe.patchframe.header.fcolapse.text

    if {$tabid == 0} {
        grid forget $QWIKMD::selresPatcheFrame
    } else {
        grid configure $QWIKMD::selresPatcheFrame -row 1 -column 0 -sticky nswe -pady 2 -padx 2 
    }

    grid forget $selframe.patchframe.header.fcolapse 

    set selframe "$QWIKMD::selResGui.f1"
    grid [ttk::frame $selframe.frameOPT] -row 0 -column 1 -sticky nwe -padx 4
    grid columnconfigure $selframe.frameOPT 0 -weight 1

    QWIKMD::createInfoButton $selframe.frameOPT 0 0
    bind $selframe.frameOPT.info <Button-1> {
        set val [QWIKMD::selResiduesWindowinfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow selResiduesWindowinfo [lindex $val 0] [lindex $val 2]
    }
    
    grid [ttk::labelframe $selframe.frameOPT.atmsel -text "Atom Selection"] -row 1 -column 0 -sticky nwe -padx 4
    grid columnconfigure $selframe.frameOPT.atmsel 0 -weight 1 

    ttk::style configure AtomSel.TEntry -foreground $QWIKMD::tempEntry
    grid [ttk::entry $selframe.frameOPT.atmsel.sel -style AtomSel.TEntry -exportselection false -textvariable QWIKMD::selResidSel -validate focus -validatecommand {
        # %V returns which event triggered the event.
        set text %V
        if {$text != "focusin" || [%W get] == "Type Selection"} {
            QWIKMD::checkSelection %W AtomSel.TEntry 
        } 
        return 1
    }] -row 0 -column 0 -sticky ew -padx 2

    bind $selframe.frameOPT.atmsel.sel <Return> {
        focus $QWIKMD::selResGui
    }

    set QWIKMD::selResidSel "Type Selection"
    if {$tabid == 0} {
        grid forget $selframe.frameOPT.atmsel   
    }
    

    grid [ttk::frame $selframe.frameOPT.manipul] -row 3 -column 0 -sticky nwe -padx 0

    ttk::frame $selframe.frameOPT.manipul.empty
    grid [ttk::labelframe $selframe.frameOPT.manipul.tableMode -labelwidget $selframe.frameOPT.manipul.empty] -row 1 -column 0 -sticky nwe -padx 4
    grid columnconfigure $selframe.frameOPT.manipul.tableMode 0 -weight 1
    grid columnconfigure $selframe.frameOPT.manipul.tableMode 1 -weight 1
    set frametbmode $selframe.frameOPT.manipul.tableMode
    grid [ttk::radiobutton $frametbmode.mutate -text "Mutate" -variable QWIKMD::tablemode -value "mutate" -command {QWIKMD::tableModeProc}] -row 0 -column 0 -sticky nswe -padx 2
    grid [ttk::radiobutton $frametbmode.protstate -text "Prot. State" -variable QWIKMD::tablemode -value "prot" -command {QWIKMD::tableModeProc}] -row 0 -column 1 -sticky snwe -padx 2
    grid [ttk::radiobutton $frametbmode.add -text "Add" -variable QWIKMD::tablemode -value "add" -command {QWIKMD::tableModeProc}] -row 1 -column 0 -sticky nswe -padx 2
    grid [ttk::radiobutton $frametbmode.delete -text "Delete" -variable QWIKMD::tablemode -value "delete" -command {QWIKMD::tableModeProc}] -row 1 -column 1 -sticky nswe -padx 2
    grid [ttk::radiobutton $frametbmode.rename -text "Rename" -variable QWIKMD::tablemode -value "rename" -command {QWIKMD::tableModeProc}] -row 2 -column 1 -sticky nswe -padx 2
    grid [ttk::radiobutton $frametbmode.inspection -text "View" -variable QWIKMD::tablemode -value "inspection" -command {QWIKMD::tableModeProc}] -row 2 -column 0 -sticky nswe -padx 2
    grid [ttk::radiobutton $frametbmode.edit -text "Edit\nAtoms" -variable QWIKMD::tablemode -value "edit" -command {QWIKMD::tableModeProc}] -row 3 -column 0 -sticky nswe -padx 2
    grid [ttk::radiobutton $frametbmode.type -text "Type" -variable QWIKMD::tablemode -value "type" -command {QWIKMD::tableModeProc}] -row 3 -column 1 -sticky nswe -padx 2

    QWIKMD::balloon $frametbmode.mutate [QWIKMD::TableMutate]
    QWIKMD::balloon $frametbmode.protstate [QWIKMD::TableProtonate]
    QWIKMD::balloon $frametbmode.add [QWIKMD::TableAdd]
    QWIKMD::balloon $frametbmode.delete [QWIKMD::TableDelete]
    QWIKMD::balloon $frametbmode.rename [QWIKMD::TableRename]
    QWIKMD::balloon $frametbmode.inspection [QWIKMD::TableInspection]
    QWIKMD::balloon $frametbmode.type [QWIKMD::TableType]

    grid [ttk::frame $selframe.frameOPT.manipul.buttFrame] -row 2 -column 0 -sticky nwe -padx 4
    grid columnconfigure $selframe.frameOPT.manipul.buttFrame 0 -weight 1

    set framebutt $selframe.frameOPT.manipul.buttFrame

    grid [ttk::button $framebutt.butApply -text "Apply" -padding "4 2 4 2" -command {QWIKMD::Apply} -state disable] -row 0 -column 0 -sticky we -pady 4
    grid [ttk::button $framebutt.butClear -text "Clear Selection" -padding "4 2 4 2" -command QWIKMD::SelResClearSelection] -row 1 -column 0 -sticky we -pady 4

    QWIKMD::balloon $framebutt.butApply [QWIKMD::TableApply]
    QWIKMD::balloon $framebutt.butClear [QWIKMD::TableClear]

    grid [ttk::button $framebutt.butAddTP -text "Add Topo+Param" -padding "4 2 4 2" -command {QWIKMD::AddTP} -state normal] -row 2 -column 0 -sticky we -pady 4



    grid [ttk::frame $selframe.frameOPT.manipul.secStrc ] -row 3 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $selframe.frameOPT.manipul.secStrc 0 -weight 1

    set frameSecLabl $selframe.frameOPT.manipul.secStrc

    grid [ttk::frame $frameSecLabl.header] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $frameSecLabl.header 0 -weight 1

    grid [ttk::label $frameSecLabl.header.lbtitle -text "$QWIKMD::downPoint Sec. Struct colors"] -row 0 -column 0 -sticky nswe -pady 2 -padx 2  
    ttk::frame $frameSecLabl.empty
    grid [ttk::labelframe $frameSecLabl.header.fcolapse -labelwidget $selframe.frameOPT.manipul.secStrc.empty] -row 1 -column 0 -sticky ews -padx 2
    grid columnconfigure $frameSecLabl.header.fcolapse 0 -weight 1


    bind $frameSecLabl.header.lbtitle <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Sec. Struct colors"
    }
    
    set w [QWIKMD::drawColScale $frameSecLabl.header.fcolapse]
        
    QWIKMD::balloon $frameSecLabl.header.fcolapse [QWIKMD::TableSecLab]

    bind $fro2.tb <<TablelistSelect>>  {
        %W columnconfigure 0 -selectbackground cyan -selectforeground black
        %W columnconfigure 1 -selectbackground cyan -selectforeground black
        %W columnconfigure 2 -selectbackground cyan -selectforeground black
        if {$QWIKMD::selResidSelRep != ""} {
            mol delrep [QWIKMD::getrepnum $QWIKMD::selResidSelRep] $QWIKMD::topMol
        }
        set QWIKMD::selResidSel "Type Selection"
        set QWIKMD::selResidSelIndex [list]
        set QWIKMD::selResidSelRep ""
        QWIKMD::rowSelection
    }
    

    grid [ttk::frame $selframe.frameOPT.manipul.membrane ] -row 4 -column 0 -sticky nwe -padx 2 -pady 2
    grid [ttk::frame $selframe.frameOPT.manipul.membrane.header] -row 0 -column 0 -sticky nswe -pady 2 -padx 2 
    grid columnconfigure $selframe.frameOPT.manipul.membrane.header 0 -weight 1

    grid [ttk::label $selframe.frameOPT.manipul.membrane.header.lbtitle -text "$QWIKMD::rightPoint Membrane"] -row 0 -column 0 -sticky nswe -pady 2 -padx 2  
    ttk::frame $selframe.frameOPT.manipul.membrane.empty
    grid [ttk::labelframe $selframe.frameOPT.manipul.membrane.header.fcolapse -labelwidget $selframe.frameOPT.manipul.membrane.empty] -row 1 -column 0 -sticky ews -padx 2
    grid columnconfigure $selframe.frameOPT.manipul.membrane.header.fcolapse 0 -weight 1

    bind $selframe.frameOPT.manipul.membrane.header.lbtitle <Button-1> {
        QWIKMD::hideFrame %W [lindex [grid info %W] 1] "Membrane"
    }
    grid forget $selframe.frameOPT.manipul.membrane.header.fcolapse

    set QWIKMD::advGui(membrane,frame) $selframe.frameOPT.manipul.membrane
    grid columnconfigure $selframe.frameOPT.manipul.membrane 0 -weight 1

    set frameMembrane $selframe.frameOPT.manipul.membrane.header.fcolapse

    grid [ttk::frame $frameMembrane.lipidopt]  -row 0 -column 0 -sticky news
    grid columnconfigure $frameMembrane.lipidopt 1 -weight 1
    grid [ttk::label $frameMembrane.lipidopt.lblipid -text "Lipid "] -row 0 -column 0 -sticky e -padx 2
    grid [ttk::combobox $frameMembrane.lipidopt.combolipid -values {POPC POPE} -state readonly -textvariable QWIKMD::advGui(membrane,lipid) ] -row 0 -column 1 -sticky ew -padx 2
    
    bind $frameMembrane.lipidopt.combolipid <<ComboboxSelected>> {
        if {[info exists QWIKMD::advGui(membrane,center,x)]} {
            QWIKMD::updateMembraneBox [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]           
        }
        %W selection clear  
    }

    grid [ttk::frame $frameMembrane.size]  -row 1 -column 0 -sticky news -pady 2
    grid columnconfigure $frameMembrane.size 0 -weight 1

    grid [ttk::label $frameMembrane.size.x -text "x" ] -row 0 -column 0 -sticky w -padx 2
    grid [ttk::entry $frameMembrane.size.xentry -width 4 -textvariable QWIKMD::advGui(membrane,xsize) -validate focusout -validatecommand {
        if {[info exists QWIKMD::advGui(membrane,center,x)]} {
            QWIKMD::updateMembraneBox [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]           
        }
        return 0
    }] -row 0 -column 1 -sticky we -padx 2
    grid [ttk::label $frameMembrane.size.xA -text "A"] -row 0 -column 2 -sticky we -padx 2

    grid [ttk::label $frameMembrane.size.y -text "y"] -row 0 -column 3 -sticky w -padx 2
    grid [ttk::entry $frameMembrane.size.yentry -width 4 -textvariable QWIKMD::advGui(membrane,ysize) -validate focusout -validatecommand {
        if {[info exists QWIKMD::advGui(membrane,center,y)]} {
            QWIKMD::updateMembraneBox [list $QWIKMD::advGui(membrane,center,x) $QWIKMD::advGui(membrane,center,y) $QWIKMD::advGui(membrane,center,z)]
        }
        return 0
    }] -row 0 -column 4 -sticky ew 
    grid [ttk::label $frameMembrane.size.yA -text "A"] -row 0 -column 5 -sticky we -padx 2

    grid [ttk::button $frameMembrane.size.box -text "Box" -padding "1 0 1 0" -command {
        QWIKMD::AddMBBox
        QWIKMD::DrawBox
    }] -row 0 -column 6 -sticky w

    grid [ttk::frame $frameMembrane.move]  -row 2 -column 0 -sticky news -pady 2
    grid [ttk::radiobutton $frameMembrane.move.translate -text "Translate" -variable QWIKMD::advGui(membrane,efect) -value "translate"] -row 0 -column 0 -sticky w -padx 2
    grid [ttk::radiobutton $frameMembrane.move.rotate -text "Rotate" -variable QWIKMD::advGui(membrane,efect) -value "rotate"] -row 0 -column 1 -sticky w -padx 2

    grid [ttk::frame $frameMembrane.axis]  -row 3 -column 0 -sticky news -pady 2    
    grid columnconfigure $frameMembrane.axis 0 -weight 1

    grid [ttk::frame $frameMembrane.axis.axisopt] -row 0 -column 0 -sticky news
    grid columnconfigure $frameMembrane.axis.axisopt 0 -weight 1
    grid columnconfigure $frameMembrane.axis.axisopt 1 -weight 1
    grid columnconfigure $frameMembrane.axis.axisopt 2 -weight 1

    grid [ttk::radiobutton $frameMembrane.axis.axisopt.x -text "x" -variable QWIKMD::advGui(membrane,axis) -value "x"] -row 0 -column 0 -sticky w -padx 2
    grid [ttk::radiobutton $frameMembrane.axis.axisopt.y -text "y" -variable QWIKMD::advGui(membrane,axis) -value "y"] -row 0 -column 1 -sticky w -padx 2
    grid [ttk::radiobutton $frameMembrane.axis.axisopt.z -text "z" -variable QWIKMD::advGui(membrane,axis) -value "z"] -row 0 -column 2 -sticky w -padx 2

    grid [ttk::frame $frameMembrane.axis.axismulti] -row 4 -column 0 -sticky news
    grid columnconfigure $frameMembrane.axis.axismulti 0 -weight 1
    grid columnconfigure $frameMembrane.axis.axismulti 1 -weight 1
    grid columnconfigure $frameMembrane.axis.axismulti 2 -weight 1
    grid columnconfigure $frameMembrane.axis.axismulti 3 -weight 1

    grid [ttk::button $frameMembrane.axis.axismulti.minus2 -text "--" -padding "1 0 1 0" -width 2 -command {
        if {$QWIKMD::advGui(membrane,efect) == "translate"} {
            set QWIKMD::advGui(membrane,multi) 5
        } else {
            set QWIKMD::advGui(membrane,multi) 15
        }
        QWIKMD::incrMembrane "-"
        }] -row 0 -column 0 -sticky ew
    grid [ttk::button $frameMembrane.axis.axismulti.minus -text "-" -padding "1 0 1 0" -width 2 -command {
        set QWIKMD::advGui(membrane,multi) 1
        QWIKMD::incrMembrane "-"
        }] -row 0 -column 1 -sticky ew

    grid [ttk::button $frameMembrane.axis.axismulti.plus -text "+"  -padding "1 0 1 0" -width 2 -command {
        set QWIKMD::advGui(membrane,multi) 1
        QWIKMD::incrMembrane "+"
        }] -row 0 -column 2 -sticky ew
    grid [ttk::button $frameMembrane.axis.axismulti.plus2 -text "++"  -padding "1 0 1 0" -width 2 -command {
        if {$QWIKMD::advGui(membrane,efect) == "translate"} {
            set QWIKMD::advGui(membrane,multi) 5
        } else {
            set QWIKMD::advGui(membrane,multi) 15
        }
        QWIKMD::incrMembrane "+"
        }] -row 0 -column 3 -sticky ew

    grid [ttk::frame $frameMembrane.buttons]  -row 4 -column 0 -sticky we -pady 2
    grid columnconfigure $frameMembrane.buttons 0 -weight 1
    grid columnconfigure $frameMembrane.buttons 1 -weight 1

    grid [ttk::button $frameMembrane.buttons.generate -text "Generate" -padding "1 0 1 0" -command QWIKMD::GenerateMembrane] -row 0 -column 0 -sticky ew
    grid [ttk::button $frameMembrane.buttons.delete -text "Delete" -padding "1 0 1 0" -command {
        QWIKMD::deleteMembrane
    }] -row 0 -column 1 -sticky ew

    grid [ttk::button $frameMembrane.optimize -text "Optimize Size" -padding "1 0 1 0" -command {
        if {$QWIKMD::membraneFrame == ""} {
            tk_messageBox -message "To optimize membrane size, please generate the membrane first." -type ok -icon warning
            return
        }
        QWIKMD::OptSize
    }] -row 5 -column 0 -sticky we

    set QWIKMD::advGui(membrane,lipid) POPC
    set QWIKMD::advGui(membrane,xsize) 30
    set QWIKMD::advGui(membrane,ysize) 30
    set QWIKMD::advGui(membrane,efect) "translate"
    set QWIKMD::advGui(membrane,axis) "x"
    set QWIKMD::advGui(membrane,multi) "1"

    if {$tabid == 0} {
        grid forget $QWIKMD::advGui(membrane,frame)

    }
    QWIKMD::tableModeProc

    grid [ttk::labelframe $selframe.frameOPT.manipul.strctChck -text "Structure Check" -padding "0 0 0 0"] -row 5 -column 0 -sticky nwe -padx 2 -pady 2
    grid columnconfigure $selframe.frameOPT.manipul.strctChck 0 -weight 1

    set frameStrcuCheck $selframe.frameOPT.manipul.strctChck

    grid [ttk::frame $frameStrcuCheck.messages ] -row 0 -column 0 -sticky nwes -padx 2 -pady 2
    grid columnconfigure $frameStrcuCheck.messages 1 -weight 1
    grid columnconfigure $frameStrcuCheck.messages 0 -weight 0
    set row 0

    set messageframe $selframe.frameOPT.manipul.strctChck.messages

    grid [label $messageframe.topoerror -background green -width 2 -relief raised -height 1 ] -row $row -column 0 -sticky e -padx 0 -pady 0
    set QWIKMD::topocolor $messageframe.topoerror

    grid [ttk::label $messageframe.topoerrortxt] -row $row -column 1 -sticky w
    set  QWIKMD::topolabel $messageframe.topoerrortxt
    bind $messageframe.topoerrortxt <Button-1> {
        set val [QWIKMD::TopologiesInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow toporeport [lindex $val 0] [lindex $val 2]
    }

    incr row
    grid [label $messageframe.chirerror -background green -width 2 -relief raised -height 1] -row $row -column 0 -sticky e -padx 0 -pady 0 -pady 2
    set QWIKMD::chircolor $messageframe.chirerror

    grid [ttk::label $messageframe.chirerrortxt -padding "0 0 0 0"] -row $row -column 1 -sticky w
    set QWIKMD::chirlabel $messageframe.chirerrortxt


    bind $messageframe.chirerrortxt <Button-1> {
        set val [QWIKMD::ChiralityInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow chirerror [lindex $val 0] [lindex $val 2]
    }

    incr row
    grid [label $messageframe.cispeperror -background green -width 2 -relief raised -height 1] -row $row -column 0 -sticky e -padx 0 -pady 0 -pady 2
    set QWIKMD::cispcolor $messageframe.cispeperror

    grid [ttk::label $messageframe.cispeperrortxt -padding "0 0 0 0"] -row $row -column 1 -sticky w
    set QWIKMD::cisplabel $messageframe.cispeperrortxt


    bind $messageframe.cispeperrortxt <Button-1> {
        set val [QWIKMD::CispeptideInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow cisperror [lindex $val 0] [lindex $val 2]
    }

    incr row
    grid [label $messageframe.gapserror -background green -width 2 -relief raised -height 1] -row $row -column 0 -sticky e -padx 0 -pady 0 -pady 2
    set QWIKMD::gapscolor $messageframe.gapserror

    grid [ttk::label $messageframe.gapserrortxt -padding "0 0 0 0"] -row $row -column 1 -sticky w
    set QWIKMD::gapslabel $messageframe.gapserrortxt


    bind $messageframe.gapserrortxt <Button-1> {
        set val [QWIKMD::GapsInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow gapsreport [lindex $val 0] [lindex $val 2]
    }

    incr row
    grid [label $messageframe.torsionOut -background green -width 2 -relief raised -height 1] -row $row -column 0 -sticky e -padx 0 -pady 0
    set QWIKMD::torsionOutliearcolor $messageframe.torsionOut

    grid [ttk::label $messageframe.torsionOuttxt -padding "0 0 0 0"] -row $row -column 1 -sticky w
    set QWIKMD::torsionOutliearlabel $messageframe.torsionOuttxt


    bind $messageframe.torsionOuttxt <Button-1> {
        set val [QWIKMD::TorsionOutlierInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow torsionOut [lindex $val 0] [lindex $val 2]
    }

    incr row
    grid [label $messageframe.torsionMarginal -background green -width 2 -relief raised -height 1] -row $row -column 0 -sticky e -padx 0 -pady 0
    set QWIKMD::torsionMarginalcolor $messageframe.torsionMarginal

    grid [ttk::label $messageframe.torsionMarginaltxt -padding "0 0 0 0"] -row $row -column 1 -sticky w
    set QWIKMD::torsionMarginallabel $messageframe.torsionMarginaltxt


    bind $messageframe.torsionMarginaltxt <Button-1> {
        set val [QWIKMD::TorsionMarginalInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow torsionMarginal [lindex $val 0] [lindex $val 2]
    }

    grid [ttk::frame $frameStrcuCheck.buttons ] -row 1 -column 0 -sticky we -padx 2 -pady 2
    grid columnconfigure $frameStrcuCheck.buttons 0 -weight 1
    grid columnconfigure $frameStrcuCheck.buttons 1 -weight 1

    grid [ttk::button $frameStrcuCheck.buttons.ignore -command {
        set color white
        set labellist [list $QWIKMD::chircolor $QWIKMD::cispcolor $QWIKMD::gapscolor $QWIKMD::torsionMarginalcolor $QWIKMD::torsionOutliearcolor]
        foreach label $labellist {
            if {[$label cget -background] != "green"} {
                $label configure -background $color
            }
        }
        
        if {[lindex $QWIKMD::topoerror 0] != 0} {
            set QWIKMD::warnresid 1
            tk_messageBox -message "Missing Topologies cannot be ignored.\nPlease refer to the \"Structure Manipulation/Check\" window to fix them" -title "Missing Topologies" -icon warning -type ok
        } else {
            if {[$QWIKMD::topocolor cget -background] != "green"} {
                $QWIKMD::topocolor configure -background $color
            }
            set QWIKMD::warnresid 0
        }
    } -padding "2 0 2 0" -text "Ignore"] -row 0 -column 0 -sticky we
    
    grid [ttk::button $frameStrcuCheck.buttons.check -command {
        QWIKMD::checkStructur
    } -padding "2 0 2 0" -text "Check"] -row 0 -column 1 -sticky we
}

proc QWIKMD::LoadOptBuild {tabid} {

    set loadoptWindow ".loadopt"

    if {[winfo exists $loadoptWindow] != 1} {
        toplevel $loadoptWindow
        wm protocol $loadoptWindow WM_DELETE_WINDOW {
            destroy ".loadopt"
        }
        
        wm minsize $loadoptWindow -1 -1
        #wm resizable $loadoptWindow 0 0

        grid columnconfigure $loadoptWindow 0 -weight 1
        grid rowconfigure $loadoptWindow 1 -weight 1
        ## Title of the windows
        wm title $loadoptWindow  "Loading Trajectories"
        set x [expr round([winfo screenwidth .]/2.0)]
        set y [expr round([winfo screenheight .]/2.0)]
        wm geometry $loadoptWindow -$x-$y
        grid [ttk::frame $loadoptWindow.f0] -row 0 -column 0 -sticky ew -padx 4 -pady 4

        if {$tabid == 0} {
            set solvent $QWIKMD::basicGui(solvent,0)
        } else {
            set solvent $QWIKMD::advGui(solvent,0)
        }
        if {$solvent == "Explicit" && [string first "Windows" $::tcl_platform(os)] == -1} {
            grid [ttk::checkbutton $loadoptWindow.f0.checkWaters -text "Don't load water molecules?" -variable QWIKMD::loadremovewater] -row 0 -column 0 -sticky w -padx 2
            grid [ttk::checkbutton $loadoptWindow.f0.checkIons -text "Don't load solvent ion molecules?" -variable QWIKMD::loadremoveions] -row 1 -column 0 -sticky w -padx 2
            grid [ttk::checkbutton $loadoptWindow.f0.checkhydrogen -text "Don't load hydrogen atoms?" -variable QWIKMD::loadremovehydrogen] -row 2 -column 0 -sticky w -padx 2
        }
        
        grid [ttk::labelframe $loadoptWindow.ftable -text "Select Trajectories"] -row 1 -column 0 -sticky nsew -padx 4 -pady 4

        grid columnconfigure $loadoptWindow.ftable 0 -weight 1
        grid rowconfigure $loadoptWindow.ftable 0 -weight 1

        set table [QWIKMD::addSelectTable $loadoptWindow.ftable 2]

        set listprot [list]
        if {[catch {glob ${QWIKMD::outPath}/run/*.dcd} listprot] == 0} {
            set j 0
            $table insert end "{} {}"
            $table cellconfigure end,0 -window QWIKMD::ProcSelect
            $table cellconfigure end,1 -text "Initial Structure"
            [$table windowpath $j,0].r state selected
            incr j
            set QWIKMD::state 0
            for {set i 0} {$i < [llength $QWIKMD::prevconfFile]} {incr i} {
                if {[lsearch $listprot "*/[lindex $QWIKMD::prevconfFile $i].dcd"] > -1} {
                    $table insert end "{} {}"
                    $table cellconfigure end,0 -window QWIKMD::ProcSelect
                    $table cellconfigure end,1 -text [lindex $QWIKMD::prevconfFile $i]
                    [$table windowpath $j,0].r state !selected
                    incr QWIKMD::state
                    incr j
                }
            }
        }

        grid [ttk::frame $loadoptWindow.fstride] -row 2 -column 0 -sticky ew -padx 4 -pady 4

        grid columnconfigure $loadoptWindow.fstride 1 -weight 1
        grid rowconfigure $loadoptWindow.fstride 0 -weight 1

        grid [ttk::label $loadoptWindow.fstride.lstride -text "Loading Trajectory Frame Step (Stride)"] -row 0 -column 0 -sticky w -padx 2
        
        grid [ttk::entry $loadoptWindow.fstride.entryStride -textvariable QWIKMD::loadstride -width 6] -row 0 -column 1 -sticky ew

        grid [ttk::frame $loadoptWindow.fbutton] -row 3 -column 0 -sticky e -padx 4 -pady 4

        grid [ttk::button $loadoptWindow.fbutton.okBut -text "Ok" -padding "1 0 1 0" -width 15 -command {
            
            set table ".loadopt.ftable.tb"
            set QWIKMD::loadprotlist [list]
            set i 0
            foreach prtcl [$table getcolumns 1] {
                set chcbt [$table windowpath $i,0].r
                set state [$chcbt state !selected]
                if { $state == "selected" && $i == 0} {
                    set QWIKMD::loadinitialstruct 1
                } elseif {$state == "selected"} {
                    lappend QWIKMD::loadprotlist $prtcl
                }
                incr i
            }
            if {[llength $QWIKMD::loadprotlist] == 0 && $QWIKMD::loadinitialstruct == 0} {
                tk_messageBox -message "Please select at least one the trajectories or the Initial Structure to be loaded in VMD." -icon warning -type ok -title "No Trajectory Selected"
            } else {
                destroy ".loadopt"
            }
            
            } ] -row 0 -column 0 -sticky ns

        grid [ttk::button $loadoptWindow.fbutton.cancel -text "Cancel" -padding "1 0 1 0" -width 15 -command {
            set QWIKMD::loadprotlist "Cancel"
            destroy ".loadopt"
            } ] -row 0 -column 1 -sticky ns
        #raise $procWindow
    } else {
        wm deiconify $loadoptWindow
    }
    tkwait window $loadoptWindow
}


############################################################
## Lock and unlock Structure Manipulation/Check Window in the case 
## of Atom selection functions (e.g. Selecting anchoring/pulling residues)
## opt 0 === lock
## opt 1 === unlock
############################################################
proc QWIKMD::lockSelResid {opt} {
    set frame "$QWIKMD::selResGui.f1.frameOPT.manipul"
    if {$opt == 0} {
        if {[winfo exists $frame.tableMode]} {
            grid forget $frame.tableMode
        }
        if {[winfo exists $frame.buttFrame.butAddTP]} {
            grid forget $frame.buttFrame.butAddTP
        }
        if {[winfo exists $QWIKMD::advGui(membrane,frame)]} {
            grid forget $QWIKMD::advGui(membrane,frame)
        }
        if {[winfo exists $QWIKMD::selresPatcheFrame]} {
            grid forget $QWIKMD::selresPatcheFrame
        }
        if {[winfo exists $frame.strctChck]} {
            grid forget $frame.strctChck
        }
        if {[winfo exists $QWIKMD::selResGui.f1.frameOPT.atmsel] == 1} {
            if {($QWIKMD::anchorpulling == 1 && $QWIKMD::run == "SMD") || [$QWIKMD::topGui.nbinput index current] == 0} {
                grid forget $QWIKMD::selResGui.f1.frameOPT.atmsel
            }
        } else {
            grid conf $QWIKMD::selResGui.f1.frameOPT.atmsel -row 1 -column 0 -sticky nwe -padx 4
        }
        
    } elseif {$opt == 1} {
        $frame.tableMode.mutate configure -state normal
        $frame.tableMode.protstate configure -state normal
        $frame.tableMode.add configure -state normal
        $frame.tableMode.delete configure -state normal
        $frame.tableMode.rename configure -state normal
        $frame.tableMode.type configure -state normal
        $frame.tableMode.edit configure -state normal
        $frame.buttFrame.butAddTP configure -state normal
        if {[$QWIKMD::topGui.nbinput index current] == 1} {
            grid conf $QWIKMD::selResGui.f1.frameOPT.atmsel -row 1 -column 0 -sticky nwe -padx 4
            if {$QWIKMD::prepared == 0 && $QWIKMD::load == 0 && [wm title $QWIKMD::selResGui] == "Structure Manipulation/Check"} {
                grid conf $QWIKMD::advGui(membrane,frame) -row 4 -column 0 -sticky nwe -padx 2 -pady 2
                grid conf $QWIKMD::selresPatcheFrame -row 1 -column 0 -sticky nswe -pady 2
            }
        } elseif {[$QWIKMD::topGui.nbinput index current] == 0} {
            grid forget $QWIKMD::selResGui.f1.frameOPT.atmsel
        }
        grid conf $frame.tableMode -row 1 -column 0 -sticky nwe -padx 4
        grid conf $frame.strctChck -row 5 -column 0 -sticky nwe -padx 2 -pady 2
        grid conf $frame.buttFrame.butAddTP -row 2 -column 0 -sticky we -pady 4

    }
}

proc QWIKMD::selResidForSelection {title tableIndexs} {
    QWIKMD::callStrctManipulationWindow
    wm title $QWIKMD::selResGui $title

    if {$title != "Structure Manipulation/Check"} {
        set QWIKMD::tablemode "inspection"
        QWIKMD::tableModeProc
        $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state normal
        QWIKMD::lockSelResid 0 
    }
    
    set table $QWIKMD::selresTable 
    $table selection clear 0 end
    
    if {[llength $tableIndexs] > 0} {
        set resid [$table getcolumns 0]
        set chains [$table getcolumns 2]
        set index ""
        for {set i 0} {$i < [llength $tableIndexs]} { incr i} {
            for {set j 0} {$j< [llength $resid]} {incr j} {
                if {[lindex $tableIndexs $i] == "[lindex $resid $j]_[lindex $chains $j]"} {
                    lappend index $j
                    break
                }
            }
            
        }
        $table columnconfigure 0 -selectbackground blue -selectforeground white
        $table columnconfigure 1 -selectbackground blue -selectforeground white
        $table columnconfigure 2 -selectbackground blue -selectforeground white
        $table selection set $index
    
        if {$QWIKMD::selResidSelRep == ""} {
            mol addrep $QWIKMD::topMol
            set QWIKMD::selResidSelRep [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]
        }
        mol modcolor [QWIKMD::getrepnum $QWIKMD::selResidSelRep] $QWIKMD::topMol "Name"
        mol modselect [QWIKMD::getrepnum $QWIKMD::selResidSelRep] $QWIKMD::topMol $QWIKMD::selResidSel
        mol modstyle [QWIKMD::getrepnum $QWIKMD::selResidSelRep] $QWIKMD::topMol "Licorice"
    }
    
}
proc QWIKMD::editAtomGuiProc {} {
    QWIKMD::save_viewpoint 1
    set tabid [$QWIKMD::topGui.nbinput index current]
    if {[winfo exists $QWIKMD::editATMSGui] != 1} {
        toplevel $QWIKMD::editATMSGui
    } else {
        wm deiconify $QWIKMD::editATMSGui
        return
    }

    grid columnconfigure $QWIKMD::editATMSGui 0 -weight 1
    grid rowconfigure $QWIKMD::editATMSGui 0 -weight 1
    ## Title of the windows
    wm title $QWIKMD::editATMSGui "Edit Atoms" ;# titulo da pagina

    wm protocol $QWIKMD::editATMSGui WM_DELETE_WINDOW {
        QWIKMD::deleteAtomGuiProc
    }

    grid [ttk::frame $QWIKMD::editATMSGui.f1] -row 0 -column 0 -sticky nsew -padx 2 -pady 4
    grid columnconfigure $QWIKMD::editATMSGui.f1 0 -weight 1
    grid columnconfigure $QWIKMD::editATMSGui.f1 1 -weight 1
    grid rowconfigure $QWIKMD::editATMSGui.f1 0 -weight 1

    set selframe "$QWIKMD::editATMSGui.f1"
    grid [ttk::frame $selframe.tableframe] -row 0 -column 0 -sticky nswe -padx 4

    grid columnconfigure $selframe.tableframe 0 -weight 1
    grid rowconfigure $selframe.tableframe 0 -weight 1

    set fro2 $selframe.tableframe
    option add *Tablelist.activeStyle       frame
    
    option add *Tablelist.movableColumns    no

        tablelist::tablelist $fro2.tb \
        -columns { 0 "Index" center
                0 "Resname"  center
                0 "Res ID"   center
                0 "Chain ID"     center
                0 "Atom Name"    center
                0 "Element" center
                0 "Type" center
                } \
                -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] \
                -showseparators 0 -labelrelief groove -labelcommand {}  -labelbd 1 -selectforeground black\
                -foreground black -background white -state normal -stretch "all" -selectmode extended -stripebackgroun white -exportselection true\
                -editstartcommand QWIKMD::atmStartEdit -editendcommand QWIKMD::atmEndEdit 

    $fro2.tb columnconfigure 0 -selectbackground cyan -width 0 -maxwidth 0 -name Index
    $fro2.tb columnconfigure 1 -selectbackground cyan -width 0 -maxwidth 0 -name ResName
    $fro2.tb columnconfigure 2 -selectbackground cyan -width 0 -maxwidth 0 -editable true -editwindow ttk::entry -name ResID
    $fro2.tb columnconfigure 3 -selectbackground cyan -width 0 -maxwidth 0 -name ChainID
    $fro2.tb columnconfigure 4 -selectbackground cyan -width 0 -maxwidth 0 -editable true -editwindow ttk::combobox -name AtmdNAME
    $fro2.tb columnconfigure 5 -selectbackground cyan -width 0 -maxwidth 0 -name Element
    $fro2.tb columnconfigure 6 -selectbackground cyan -width 0 -maxwidth 0 -name Type
    
    set QWIKMD::atmsTable $fro2.tb
    grid $fro2.tb -row 0 -column 0 -sticky news
    $fro2.tb configure -height 15 -width 0 -stretch "all"


    ##Scrool_BAr V
    scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
    grid $fro2.scr1 -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
    grid $fro2.scr2 -row 1 -column 0 -sticky swe

    grid [ttk::frame $selframe.frameInfo] -row 0 -column 1 -sticky nswe -padx 4
    grid columnconfigure $selframe.frameInfo 0 -weight 1
    grid rowconfigure $selframe.frameInfo 0 -weight 1

    grid [ttk::frame $selframe.frameInfo.txtframe] -row 0 -column 0 -sticky nswe -padx 4
    grid columnconfigure $selframe.frameInfo.txtframe 0 -weight 2
    grid rowconfigure $selframe.frameInfo.txtframe 0 -weight 2

    grid [tk::text $selframe.frameInfo.txtframe.text -font tkconfixed -wrap none -bg white -width 50 -height 1 -relief flat -foreground black -yscrollcommand [list $selframe.frameInfo.txtframe.scr1 set] -xscrollcommand [list $selframe.frameInfo.txtframe.scr2 set]] -row 0 -column 0 -sticky wens

        ##Scrool_BAr V
    scrollbar $selframe.frameInfo.txtframe.scr1  -orient vertical -command [list $selframe.frameInfo.txtframe.text yview]
    grid $selframe.frameInfo.txtframe.scr1  -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $selframe.frameInfo.txtframe.scr2  -orient horizontal -command [list $selframe.frameInfo.txtframe.text xview]
    grid $selframe.frameInfo.txtframe.scr2 -row 1 -column 0 -sticky swe

    set QWIKMD::atmsText "$selframe.frameInfo.txtframe.text"
    
    grid [ttk::frame $selframe.frameInfo.okcancelframe] -row 1 -column 0 -sticky nse -padx 4
    grid columnconfigure $selframe.frameInfo.okcancelframe 0 -weight 1
    grid rowconfigure $selframe.frameInfo.okcancelframe 0 -weight 1
    grid [ttk::button $selframe.frameInfo.okcancelframe.delete -text "Delete" -command {
        set index [$QWIKMD::atmsTable curselection]
        if { $index == -1 } {
            return
        }
        set atmindex [expr [$QWIKMD::atmsTable cellcget $index,0 -text] -1]
        QWIKMD::deleteAtoms $atmindex $QWIKMD::atmsMol
        lappend QWIKMD::atmsDeleteNames $atmindex
        $QWIKMD::atmsTable delete $index
        graphics $QWIKMD::atmsMol delete [lindex $QWIKMD::atmsLables $atmindex]
    }] -row 0 -column 0 -sticky ws -padx 2
    grid [ttk::button $selframe.frameInfo.okcancelframe.ok -text "Ok" -command QWIKMD::changeAtomNames] -row 0 -column 1 -sticky ws -padx 2
    grid [ttk::button $selframe.frameInfo.okcancelframe.cancel -text "Cancel" -command QWIKMD::cancelAtomNames] -row 0 -column 2 -sticky es -padx 2
}
proc QWIKMD::deleteAtomGuiProc {} {
    wm withdraw $QWIKMD::editATMSGui
    mol delete $QWIKMD::atmsMol
    mol top $QWIKMD::topMol
    QWIKMD::restore_viewpoint 1 
    mol on $QWIKMD::topMol
}

proc QWIKMD::SelResClearSelection {} {
    $QWIKMD::selresTable selection clear 0 end
    for {set i 0} {$i < [llength $QWIKMD::resrepname]} {incr i} {
        mol delrep [QWIKMD::getrepnum [lindex [lindex $QWIKMD::resrepname $i] 1]] $QWIKMD::topMol
    }
    if {$QWIKMD::selResidSelRep != ""} {
        mol delrep [QWIKMD::getrepnum $QWIKMD::selResidSelRep] $QWIKMD::topMol
        set QWIKMD::selResidSelRep ""
        set QWIKMD::selResidSelIndex [list]
    }
    set QWIKMD::resrepname [list]
    set QWIKMD::selected 0
}

proc QWIKMD::AddTP {} {
    global env
    if {[winfo exists $QWIKMD::topoPARAMGUI] != 1} {
        toplevel $QWIKMD::topoPARAMGUI
    } else {
        wm deiconify $QWIKMD::topoPARAMGUI
        
        focus -force $QWIKMD::topoPARAMGUI
        return
    }

    grid columnconfigure $QWIKMD::topoPARAMGUI 0 -weight 2 -minsize 120
    grid rowconfigure $QWIKMD::topoPARAMGUI 0 -weight 2

    ## Title of the windows
    wm title $QWIKMD::topoPARAMGUI "Topology & Parameters Selection" ;# titulo da pagina
    wm protocol $QWIKMD::topoPARAMGUI WM_DELETE_WINDOW {
        wm withdraw $QWIKMD::topoPARAMGUI
        if {[winfo exists $QWIKMD::topoPARAMGUI.f1.tableframe.tb] ==1} {
            $QWIKMD::topoPARAMGUI.f1.tableframe.tb selection clear 0 end
        }
     }

    grid [ttk::frame $QWIKMD::topoPARAMGUI.f1] -row 0 -column 0 -sticky nsew -padx 2 -pady 4
    grid columnconfigure $QWIKMD::topoPARAMGUI.f1 0 -weight 1
    grid rowconfigure $QWIKMD::topoPARAMGUI.f1 0 -weight 1

    set selframe "$QWIKMD::topoPARAMGUI.f1"

    grid [ttk::frame $selframe.tableframe] -row 0 -column 0 -sticky nswe -padx 4

    grid columnconfigure $selframe.tableframe 0 -weight 1 
    grid rowconfigure $selframe.tableframe 0 -weight 1

    set fro2 $selframe.tableframe
    option add *Tablelist.activeStyle       frame
    
    option add *Tablelist.movableColumns    no
    option add *Tablelist.labelCommand      tablelist::sortByColumn


        tablelist::tablelist $fro2.tb \
        -columns { 0 "Residue NAME"  center
                0 "CHARMM NAME"  center
                0 "type" center
                0 "Topo & PARM File" center
                } \
                -yscrollcommand [list $fro2.scr1 set] -xscrollcommand [list $fro2.scr2 set] \
                -showseparators 0 -labelrelief groove  -labelbd 1 -selectforeground black\
                -foreground black -background white -state normal -selectmode extended -stretch "all" -stripebackgroun white -exportselection true \
                -editendcommand QWIKMD::editResNameType -forceeditendcommand 0

    $fro2.tb columnconfigure 0 -selectbackground cyan
    $fro2.tb columnconfigure 1 -selectbackground cyan
    $fro2.tb columnconfigure 2 -selectbackground cyan

    $fro2.tb columnconfigure 0 -sortmode integer -name "Resname"
    $fro2.tb columnconfigure 1 -sortmode dictionary -name "CHARMM NAME"
    $fro2.tb columnconfigure 2 -sortmode dictionary -name "type"
    $fro2.tb columnconfigure 3 -sortmode dictionary -name "TopoPArm"
    
    $fro2.tb columnconfigure 0 -width 1 -maxwidth 0 -editable true -editwindow ttk::entry
    $fro2.tb columnconfigure 1 -width 1 -maxwidth 0
    $fro2.tb columnconfigure 2 -width 1 -maxwidth 0 
    $fro2.tb columnconfigure 3 -width 1 -maxwidth 0

    grid $fro2.tb -row 0 -column 0 -sticky news
    $fro2.tb configure -height 6 -width 70

    ##Scrool_BAr V
    scrollbar $fro2.scr1 -orient vertical -command [list $fro2.tb  yview]
     grid $fro2.scr1 -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $fro2.scr2 -orient horizontal -command [list $fro2.tb xview]
    grid $fro2.scr2 -row 1 -column 0 -sticky swe

    grid [ttk::frame $selframe.buttons] -row 2 -column 0 -sticky nse -padx 2 -pady 4

    grid [ttk::button $selframe.buttons.add -text "+" -padding "1 1 1 1"  -command QWIKMD::addTopParm -width 2] -row 0 -column 1 -sticky e -pady 2
    grid [ttk::button $selframe.buttons.delete -text "-" -padding "1 1 1 1"  -command QWIKMD::deleteTopParm -width 2] -row 0 -column 2 -sticky e -pady 2 
    grid [ttk::button $selframe.buttons.apply -text "Apply" -padding "2 0 2 0" -command QWIKMD::applyTopParm] -row 0 -column 3 -sticky ew -pady 2 -padx 2

    QWIKMD::createInfoButton $selframe.buttons 0 0
    bind $selframe.buttons.info <Button-1> {
        set val [QWIKMD::topparInfo]
        set QWIKMD::link [lindex $val 1]
        QWIKMD::infoWindow toppar [lindex $val 0] [lindex $val 2]
    }
    QWIKMD::addTableTopParm
} 

proc QWIKMD::tableModeProc {} {
    set table $QWIKMD::selresTable
    $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state normal -text "Apply"

    if {$QWIKMD::tablemode == "mutate" || $QWIKMD::tablemode == "prot" || $QWIKMD::tablemode == "rename" || $QWIKMD::tablemode == "type"} {
        $table configure -selectmode single
        $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state disable
    } elseif {$QWIKMD::tablemode == "inspection"} {
        $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state disable
        $table configure -selectmode extended
    } else {
        $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state normal
        $table configure -selectmode extended
    }
    if {$QWIKMD::tablemode == "mutate" || $QWIKMD::tablemode == "prot" || $QWIKMD::tablemode == "rename"} {
        $table columnconfigur 3  -editable false
        $table columnconfigure 1 -editable true
    } elseif {$QWIKMD::tablemode == "type"} {
        
        $table columnconfigur 3  -editable true
        $table columnconfigure 1 -editable false
    } elseif {$QWIKMD::tablemode == "edit"} {
        $QWIKMD::selResGui.f1.frameOPT.manipul.buttFrame.butApply configure -state normal -text "Edit"
        $table columnconfigur 3  -editable false
        $table columnconfigure 1 -editable false
        $table configure -selectmode single
    } else {
        $table columnconfigur 3  -editable false
        $table columnconfigure 1 -editable false
    }
    set sel [$QWIKMD::selresTable curselection]
    $table selection set $sel
    QWIKMD::rowSelection
}

############################################################
## Creates the combobox inserted in the resname column in the 
## Select Resid window. The args are automatically generated 
## by the -window configuration option of the tablelist cell 
############################################################
proc QWIKMD::createResCombo {tbl row col text} {
    set w [$tbl editwinpath]
    ttk::style map TCombobox -fieldbackground [list readonly #ffffff]
    set resname [lindex [split $text "->"] 0]
    set resid [$tbl cellcget $row,0 -text]
    set chain [$tbl cellcget $row,2 -text]
    set type [$tbl cellcget $row,3 -text]

    set sel [atomselect top "resid \"$resid\" and chain \"$chain\""]
    set ind end
    if {$QWIKMD::tablemode == "prot"} {
        set ind end
    } elseif {$QWIKMD::tablemode == "mutate"} {
        set ind 2
    }

    set list [split $text "->"]
    set initext ""
    if {$QWIKMD::tablemode == "prot" } {
        if { [llength  $list] > 1} {
            set initext [lindex $list $ind]
        } else {
            set initext [lindex $list 0] 
        }
    } elseif {$QWIKMD::tablemode == "mutate"} {
        set initext [lindex $list 0] 
    } elseif {$QWIKMD::tablemode == "rename"} {
        set initext [lindex [$sel get resname] 0]
    } elseif {$QWIKMD::tablemode == "type"} {
        set initext $text
    }
    
    set QWIKMD::protres(0,0) $initext

    $sel delete
    set QWIKMD::protres(0,1) "$row"
 
    set QWIKMD::protres(0,2) $initext
    if {[llength $list] == 3 } {
        set QWIKMD::protres(0,3) [string trim [lindex $list end] " "]
    } else {
        set QWIKMD::protres(0,3) ""
    }

    switch [$tbl columncget $col -name] {
        ResNAME {
            if {$QWIKMD::tablemode == "prot" && $type == "protein"} {
                set do 0
                set res [string trim [lindex [split $text "->"] end] " "]
                
                switch $res {
                    ASP {
                        set QWIKMD::combovalues {ASP ASPP}
                        set do 1
                    }
                    ASPP {
                        set QWIKMD::combovalues {ASP ASPP}
                        set do 1
                    }
                    GLU {
                        set QWIKMD::combovalues {GLU GLUP}
                        set do 1
                    } 
                    GLUP {
                        set QWIKMD::combovalues {GLU GLUP}
                        set do 1
                    }
                    LYS {
                        set QWIKMD::combovalues {LYS LSN}
                        set do 1
                    }
                    LSN {
                        set QWIKMD::combovalues {LYS LSN}
                        set do 1
                    }
                    CYS {
                        set QWIKMD::combovalues {CYS CYSD}
                        set do 1
                    }
                    CYSD {
                        set QWIKMD::combovalues {CYS CYSD}
                        set do 1
                    }
                    SER {
                        set QWIKMD::combovalues {SER SERD}
                        set do 1
                    }
                    SERD {
                        set QWIKMD::combovalues {SER SERD}
                        set do 1
                    }
                    HIS {
                        set QWIKMD::combovalues {HSD HSE HSP}
                        set do 1
                    }
                    HSD {
                        set QWIKMD::combovalues {HSD HSE HSP}
                        set do 1
                    }
                    HSE {
                        set QWIKMD::combovalues {HSD HSE HSP}
                        set do 1
                    }
                    
                    HSP {
                        set QWIKMD::combovalues {HSD HSE HSP}
                        set do 1
                    } 

                }
                if {$do == 0} {
                    $tbl cancelediting  
                    return
                }
            } elseif {$QWIKMD::tablemode == "mutate" && $type != "water" && $type != "hetero" && [lsearch $QWIKMD::rename ${resid}_$chain] == -1} {
                if {$type == "protein"} {
                    set QWIKMD::combovalues {ALA ARG ASN ASP CYS GLN GLU GLY HSD ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL}
                } elseif {$type == "hetero"} {
                    set QWIKMD::combovalues $QWIKMD::heteronames
                } elseif {$type == "nucleic"} {
                    set QWIKMD::combovalues $QWIKMD::nucleic
                } elseif {$type == "glycan"} {
                    set QWIKMD::combovalues $QWIKMD::carbnames
                } elseif {$type == "lipid"} {
                    set QWIKMD::combovalues $QWIKMD::lipidname
                }
                
                set macroindex [lsearch -index 0 $QWIKMD::userMacros $type]
                if {$macroindex > -1} {
                    if {$type == "protein" || $type == "nucleic" || $type == "lipid"} {
                        set QWIKMD::combovalues [concat $QWIKMD::combovalues [lindex [lindex $QWIKMD::userMacros $macroindex] 1]]
                    } else {
                        set QWIKMD::combovalues [concat $QWIKMD::combovalues [lindex [lindex $QWIKMD::userMacros $macroindex] 2]]
                    }
                } 
                
            } elseif {$QWIKMD::tablemode == "rename" && ($type != "water")} {
                if {$type == "protein" && [lsearch $QWIKMD::rename "${resid}_$chain"] == -1 && [$QWIKMD::topGui.nbinput index current] == 0} {
                    $tbl cancelediting  
                    return
                }
                if {$type == "hetero"} {
                    set QWIKMD::combovalues $QWIKMD::heteronames
                } elseif {$type == "nucleic"} {
                    set QWIKMD::combovalues {GUA ADE CYT THY URA}
                } elseif {$type == "glycan"} {
                    set QWIKMD::combovalues $QWIKMD::carbnames
                } elseif {$type == "lipid"} {
                    set QWIKMD::combovalues $QWIKMD::lipidname
                } elseif {$type == "protein"} {
                    set QWIKMD::combovalues $QWIKMD::reslist
                } else {
                    foreach macro $QWIKMD::userMacros {
                        if {[lindex $macro 0] == $type} {
                            set QWIKMD::combovalues [lindex $macro 2]
                        } 
                    }
                }
                set macroindex [lsearch -index 0 $QWIKMD::userMacros $type]
                if {$macroindex > -1} {
                    if {$type == "protein" || $type == "nucleic" || $type == "lipid" } {
                        set QWIKMD::combovalues [concat $QWIKMD::combovalues [lindex [lindex $QWIKMD::userMacros $macroindex] 1]]
                    } else {
                        set QWIKMD::combovalues [concat $QWIKMD::combovalues [lindex [lindex $QWIKMD::userMacros $macroindex] 2]]
                    } 
                } 
            }   
        }
        Type {
            if {$QWIKMD::tablemode == "type"} {
                if {$type == "protein" && [lsearch $QWIKMD::rename "${resid}_$chain"] == -1 && [$QWIKMD::topGui.nbinput index current] == 0} {
                    $tbl cancelediting  
                    return
                }
                set defVal {protein nucleic glycan lipid hetero}
                set QWIKMD::combovalues $defVal
                foreach macro $QWIKMD::userMacros {
                    if {[lsearch $defVal [lindex $macro 0]] == -1} {
                        lappend QWIKMD::combovalues [lindex $macro 0]
                    }
                }
            }
        }
    }

    set maxwidth 11
    for {set i 0} {$i < [llength $QWIKMD::combovalues]} {incr i} {
        set width [string length [lindex $QWIKMD::combovalues $i]]
        if {$width > $maxwidth} {
            set maxwidth $width
        }
    }
    $tbl columnconfigure $col -width $maxwidth
    $w configure -width $maxwidth -values $QWIKMD::combovalues -state readonly -style TCombobox
    bind $w <<ComboboxSelected>> {
        if {[winfo exists %W]} {
            $QWIKMD::selresTable finishediting
        }   
    }
    if {$QWIKMD::tablemode == "type"} {
        set QWIKMD::prevtype $QWIKMD::protres(0,2)
    }
    set QWIKMD::prevRes $text
   
    set QWIKMD::selected 0
    return $QWIKMD::protres(0,0)
}

proc QWIKMD::EndResCombo {tbl row col text} {
    $tbl finishediting

    return $text
}
############################################################
## Representation combobox in the qwikMD main window
############################################################
proc QWIKMD::mainTableCombosStart {opt tbl row col text} {
    if {$opt == 1} {
        set w [$tbl editwinpath]
    }
     
    switch [$tbl columncget $col -name] {
        Representation {
            set chain [$tbl cellcget $row,0 -text]
            set type [$tbl cellcget $row,2 -text]
            set chaint "$chain and $type"
            set rep "Off NewCartoon QuickSurf Licorice VDW Lines Beads Points"
            if {$opt == 1} {
                $w configure -values $rep -state readonly -textvariable QWIKMD::index_cmb($chaint,1)
            }
            set indrep 1
            if {$type == "protein" || $type == "nucleic" } {
                set indrep NewCartoon
            } elseif {$type == "hetero" || $type == "glycan" } {
                set indrep Licorice             
            } elseif {$type == "water"} {
                if {[$tbl cellcget $row,0 -text] == "W"} {
                    set indrep Points
                } else {
                    set indrep VDW
                }
            } elseif {$type == "lipid" } {
                set indrep Lines
            } else {
                set indrep Licorice
            }
            if {[info exists QWIKMD::index_cmb($chaint,1)] != 1} {
                set QWIKMD::index_cmb($chaint,1) $indrep
            }
            if {[info exists QWIKMD::index_cmb($chaint,3)] != 1} {
                set QWIKMD::index_cmb($chaint,3) $row
            }
            mol modselect [QWIKMD::getrepnum $QWIKMD::index_cmb($chaint,4)] $QWIKMD::topMol $QWIKMD::index_cmb($chaint,5)
            set rep $QWIKMD::index_cmb($chaint,1)
            if {$rep == "VDW"} {
                set rep "$rep 1.0 12.0"
            }
            mol modstyle [QWIKMD::getrepnum $QWIKMD::index_cmb($chaint,4)] $QWIKMD::topMol $rep
            if {$opt == 1} {
                bind $w <<ComboboxSelected>> {
                    set table $QWIKMD::topGui.nbinput.f1.tableframe.tb
                    if {[$QWIKMD::topGui.nbinput index current] == 1} {
                        set table $QWIKMD::topGui.nbinput.f2.tableframe.tb
                    }
                    $table finishediting                
                }
            }
           return $QWIKMD::index_cmb($chaint,1)
        }
        Color {
            set chain [$tbl cellcget $row,0 -text]
            set type [$tbl cellcget $row,2 -text]
            set chaint "$chain and $type"
            set sizes [list]
            foreach color $QWIKMD::colorIdMap {
                lappend sizes [string length $color]
            }
            if {$opt == 1} {
                $w configure -values $QWIKMD::colorIdMap -state readonly -textvariable QWIKMD::index_cmb($chaint,2) -width [QWIKMD::maxcalc $sizes]
            }
            if {$type == "protein" || $type == "nucleic"} {
                set chainmol [list]
                foreach str [$tbl getcolumns 0] {
                    if {[lsearch $chainmol $str] == -1} {
                        lappend chainmol $str
                    }
                }
                set index [lindex $QWIKMD::colorIdMap [expr [lsearch $chainmol [lindex $chain 0] ] + 5]]
                if {[lindex $index 1] == "" } {
                    set index "Name"
                }
                set QWIKMD::index_cmb($chaint,2) $index 
            } else {
                if {$opt == 1} {
                    $w set "Name"
                }
                set QWIKMD::index_cmb($chaint,2) "Name"
            }

            
            if { [string is integer [lindex $QWIKMD::index_cmb($chaint,2) 0]] == 0} {
                mol modcolor [QWIKMD::getrepnum $QWIKMD::index_cmb($chaint,4)] $QWIKMD::topMol "$QWIKMD::index_cmb($chaint,2)"
            } else {
                mol modcolor [QWIKMD::getrepnum $QWIKMD::index_cmb($chaint,4)] $QWIKMD::topMol "ColorID [lindex $QWIKMD::index_cmb($chaint,2) 0]"
            }
            if {$opt == 1} {
                if {$QWIKMD::index_cmb($chaint,2) == "Name" || $QWIKMD::index_cmb($chaint,2) == "Structure" || $QWIKMD::index_cmb($chaint,2) == "Throb" || $QWIKMD::index_cmb($chaint,2) == "ResName" || $QWIKMD::index_cmb($chaint,2) == "ResType"} {
                    $w set [lsearch -inline $QWIKMD::colorIdMap $QWIKMD::index_cmb($chaint,2)]
                } else {
                    $w set [lindex $QWIKMD::colorIdMap [expr [lindex $QWIKMD::index_cmb($chaint,2) 0] + 5]]
                }
                bind $w <<ComboboxSelected>> {
                    set table $QWIKMD::topGui.nbinput.f1.tableframe.tb
                    if {[$QWIKMD::topGui.nbinput index current] == 1} {
                        set table $QWIKMD::topGui.nbinput.f2.tableframe.tb
                    }
                    $table finishediting
                }
            }
            return $QWIKMD::index_cmb($chaint,2)    
        }
    }
}
############################################################
## Color combobox in the qwikMD main window
############################################################
proc QWIKMD::mainTableCombosEnd {tbl row col text} {
    set chain [$tbl cellcget $row,0 -text]
    set type [$tbl cellcget $row,2 -text]
    set chain "$chain and $type"
    if {$col == 3} {
        set QWIKMD::index_cmb($chain,1) $text
        mol modselect [QWIKMD::getrepnum $QWIKMD::index_cmb($chain,4)] $QWIKMD::topMol $QWIKMD::index_cmb($chain,5)
        if {$QWIKMD::index_cmb($chain,1) == "Off"} {
            mol showrep $QWIKMD::topMol [QWIKMD::getrepnum $QWIKMD::index_cmb($chain,4)] off
        } else {
            set rep $QWIKMD::index_cmb($chain,1)
            if {$rep == "VDW"} {
                set rep "$rep 1.0 12.0"
            }
            mol modstyle [QWIKMD::getrepnum $QWIKMD::index_cmb($chain,4)] $QWIKMD::topMol $rep
            mol showrep $QWIKMD::topMol [QWIKMD::getrepnum $QWIKMD::index_cmb($chain,4)] on
        }
    } else {
        set QWIKMD::index_cmb($chain,2) $text
        if { [string is integer [lindex $QWIKMD::index_cmb($chain,2) 0]] == 0} {
            mol modcolor [QWIKMD::getrepnum $QWIKMD::index_cmb($chain,4)] $QWIKMD::topMol "$QWIKMD::index_cmb($chain,2)"
        } else {
            mol modcolor [QWIKMD::getrepnum $QWIKMD::index_cmb($chain,4)] $QWIKMD::topMol "ColorID [lindex $QWIKMD::index_cmb($chain,2) 0]"
        }

    }
    return $text
}

##############################################
## Enable or disable smd analysis option in
## the analysis tab 
###############################################
proc QWIKMD::ChangeMdSmd {tabid} {
    if {$tabid == 1} {
        if {[$QWIKMD::topGui.nbinput.f${tabid}.nb index current] == 0} {
            set QWIKMD::run "MD"
        } elseif {[$QWIKMD::topGui.nbinput.f${tabid}.nb index current] == 1} {
            set QWIKMD::run "SMD" 
        }
    } else {
        set QWIKMD::run [$QWIKMD::topGui.nbinput.f${tabid}.nb tab [$QWIKMD::topGui.nbinput.f${tabid}.nb index current] -text ]
    }
    
    
    if {$QWIKMD::topMol != ""} {
        
        if {[winfo exists $QWIKMD::selresTable]} {
            $QWIKMD::selresTable selection clear 0 end
        
            if {$QWIKMD::state > 0} {
                return
            } else {
                
                if {$QWIKMD::run == "SMD"} {
                    QWIKMD::checkAnchors
                    
                } else {
                    if {$QWIKMD::pullingrepname != ""} {
                        mol delrep [QWIKMD::getrepnum $QWIKMD::pullingrepname] $QWIKMD::topMol
                        set QWIKMD::pullingrepname ""
                    }
                    if {$QWIKMD::anchorrepname != ""} {
                        mol delrep [QWIKMD::getrepnum $QWIKMD::anchorrepname] $QWIKMD::topMol
                        set QWIKMD::anchorrepname ""
                    }   
                }
            }
        }
        
    }
    set tabid [$QWIKMD::topGui.nbinput index current]
    if {$tabid == 1 && $QWIKMD::run != "MDFF"} {
        set QWIKMD::confFile [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
        set QWIKMD::maxSteps [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 1]
    } elseif {$tabid == 1 && $QWIKMD::run == "MDFF" && $QWIKMD::load == 0} {
        set QWIKMD::advGui(solvent,0) "Vacuum"
    }
    ##############################################
    ## Change info button text between SMD and MD
    ##############################################
    if {$QWIKMD::run == "SMD"} {
        if {$tabid == 1} {
            set QWIKMD::advGui(solvent,minimalbox) 0
            $QWIKMD::advGui(solvent,minbox,$QWIKMD::run) configure -state disable   
        }
        bind $QWIKMD::topGui.f0.info.info <Button-1> {
            set val [QWIKMD::protocolSMDInfo]
            set QWIKMD::link [lindex $val 1]
            QWIKMD::infoWindow protocolSMDInfo [lindex $val 0] [lindex $val 2]
        }
    } else {
        bind $QWIKMD::topGui.f0.info.info <Button-1> {
            set val [QWIKMD::protocolMDInfo]
            set QWIKMD::link [lindex $val 1]
            QWIKMD::infoWindow protocolMDInfo [lindex $val 0] [lindex $val 2]
        }
    }
    QWIKMD::ChangeSolvent
}

proc QWIKMD::ChangeSolvent {} {
    global env
    set tabid [$QWIKMD::topGui.nbinput index current] 
    if {[info exists QWIKMD::basicGui(solvent,0)] == 1 && $tabid == 0} {
        if {$QWIKMD::basicGui(solvent,0) == "Implicit"} {
            $QWIKMD::basicGui(saltions,$QWIKMD::run) configure -state disable

            if {[winfo exists $QWIKMD::advGui(analyze,advance,interradio)] == 1} {
                $QWIKMD::advGui(analyze,advance,interradio) configure -state disable
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,pressbtt)] == 1 && $tabid == 0} {
                $QWIKMD::advGui(analyze,advance,pressbtt) configure -state disable
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,volbtt)] == 1 && $tabid == 0} {
                $QWIKMD::advGui(analyze,advance,volbtt) configure -state disable
            }
        } else {
            $QWIKMD::basicGui(saltions,$QWIKMD::run) configure -state readonly
            
            if {[winfo exists $QWIKMD::advGui(analyze,advance,interradio)] == 1} {
                $QWIKMD::advGui(analyze,advance,interradio) configure -state normal
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,pressbtt)] == 1 && $tabid == 0} {
                $QWIKMD::advGui(analyze,advance,pressbtt) configure -state normal
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,volbtt)] == 1 && $tabid == 0} {
                $QWIKMD::advGui(analyze,advance,volbtt) configure -state normal
            }
        }
    }
    
    if {[info exists QWIKMD::advGui(solvent,0)] == 1 && $tabid == 1} {
        $QWIKMD::advGui(saltconc,$QWIKMD::run) configure -state normal
        if {$QWIKMD::advGui(solvent,0) == "Implicit" || $QWIKMD::advGui(solvent,0) == "Vacuum"} {
            $QWIKMD::advGui(saltions,$QWIKMD::run) configure -state disable 

            if {[winfo exists $QWIKMD::advGui(analyze,advance,interradio)] == 1} {
                $QWIKMD::advGui(analyze,advance,interradio) configure -state disable
            }
            $QWIKMD::advGui(solvent,boxbuffer,$QWIKMD::run,entry) configure -state disable
            set QWIKMD::advGui(solvent,minimalbox) 0
            $QWIKMD::advGui(solvent,minbox,$QWIKMD::run) configure -state disable       
            if {$QWIKMD::run != "MDFF"} {
                set ensemble [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 3]
                set indexes [lsearch -all $ensemble "NpT"]
                if {[llength $indexes] > 0} {
                    for {set i 0} {$i < [llength $indexes]} {incr i} {
                        lset ensemble [lindex $indexes $i] "NVT"
                        $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure [lindex $indexes $i],5 -editable false
                        $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure [lindex $indexes $i],5 -foreground grey -selectforeground grey
                    }
                    $QWIKMD::advGui(protocoltb,$QWIKMD::run) columnconfigure 3 -text $ensemble  
                }
            }
            if {$QWIKMD::advGui(solvent,0) == "Vacuum" || ($QWIKMD::run == "MDFF" && $QWIKMD::load == 0)} {
                $QWIKMD::advGui(saltconc,$QWIKMD::run) configure -state disable
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,pressbtt)] == 1 && $tabid == 1} {
                $QWIKMD::advGui(analyze,advance,pressbtt) configure -state disable
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,volbtt)] == 1 && $tabid == 1} {
                $QWIKMD::advGui(analyze,advance,volbtt) configure -state disable
            }
            
        } else {
            if {($QWIKMD::run != "MDFF" && $QWIKMD::advGui(solvent,0) == "Implicit") || $QWIKMD::advGui(solvent,0) == "Explicit"} {
                $QWIKMD::advGui(saltions,$QWIKMD::run) configure -state readonly 
            }
            $QWIKMD::advGui(solvent,boxbuffer,$QWIKMD::run,entry) configure -state readonly
            if {$QWIKMD::run != "SMD"} {
                $QWIKMD::advGui(solvent,minbox,$QWIKMD::run) configure -state normal
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,interradio)] == 1} {
                $QWIKMD::advGui(analyze,advance,interradio) configure -state normal
            }
            if {$QWIKMD::run != "MDFF"} {
                set ensemble [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 3]
                
                for {set i 0} {$i < [llength $ensemble]} {incr i} {
                    lset ensemble $i "NpT"
                    if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$i,lock) == 0} {
                        $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $i,4 -editable true
                        $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $i,5 -editable true
                        $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $i,5 -foreground black -selectforeground black   
                    }
                    
                }
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) columnconfigure 3 -text $ensemble
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,pressbtt)] == 1 && $tabid == 1} {
                $QWIKMD::advGui(analyze,advance,pressbtt) configure -state normal
            }
            if {[winfo exists $QWIKMD::advGui(analyze,advance,volbtt)] == 1 && $tabid == 1} {
                $QWIKMD::advGui(analyze,advance,volbtt) configure -state normal
            }
        }
        if {$QWIKMD::membraneFrame != ""} {
            set QWIKMD::advGui(solvent,minimalbox) 0
            $QWIKMD::advGui(solvent,minbox,$QWIKMD::run) configure -state disable   
        }
    }
    # Events related to the protocol rows in the protocol table
    # Not applicable to MDFF tab
    if {$tabid == 1 && $QWIKMD::run != "MDFF" && $QWIKMD::load == 0} {
        set protocolIDS [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
        set values {Minimization Annealing Equilibration MD SMD}
        set protocolIndex 0
        foreach prot $protocolIDS {
            set delete 1
            set index [lsearch $values [file root $prot]]
            if {$index == -1} {
                set tempLib ""
                set do [catch {glob ${env(QWIKMDFOLDER)}/templates/$QWIKMD::advGui(solvent,0)/[file root ${prot}].conf} tempLib]
                if {$do == 0} {
                    set delete 0
                } else {
                    set tempLib ""
                    set do [catch {glob ${env(TMPDIR)}/${prot}.conf} tempLib]
                    if {$do == 0} {
                        set delete 0
                    }
                }
            } elseif {$index != -1} {
                set delete 0
            }
            if {$delete == 1} {
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) selection set $protocolIndex
                QWIKMD::deleteProtocol
                $QWIKMD::advGui(protocoltb,$QWIKMD::run) selection clear 0 end
                QWIKMD::ChangeSolvent
            }
            incr protocolIndex
        }       
    }
}

##############################################
## Update the Start MD button with the current 
## step (QWIKMD::state)
###############################################

proc QWIKMD::RunText {} {
    set text ""
    set tabid [$QWIKMD::topGui.nbinput index current]

    if {$tabid == 0} {
        if {[string match "*equilibration*" [lindex $QWIKMD::prevconfFile $QWIKMD::state ] ] > 0} {
            set text "Equilibration Simulation $QWIKMD::state"
        } elseif {[string match "*_production_smd_*" [lindex $QWIKMD::prevconfFile $QWIKMD::state ] ] > 0 || [string match "*_production_smd_*" [lindex $QWIKMD::prevconfFile [expr $QWIKMD::state -1] ] ] > 0} {
            set text "Production SMD Simulation $QWIKMD::state"
        } else {
            set text "Production Simulation $QWIKMD::state"
        }
    } else {
        set text "[lindex $QWIKMD::prevconfFile $QWIKMD::state ] Simulation "
    }
    
    return $text
}

##############################################
## Secondary structure colors caption based on
## TimeLine
###############################################

proc QWIKMD::drawColScale {w} {
    #local hard coding for current placement, later should make this visible externally
    set xPos 7 
    set yPos 3
    set valsYPos 7
    set valText 40
    set barTop 19
    set barBottom 34
    set caption [list "Turn" "Beta Extended" "Beta Bridge" "Alpha-Helix" "3-10 Helix" "Pi-Helix" "Coil"]
    
    
    grid [ttk::frame $w.colscale] -row 0 -column 0
    grid columnconfigure $w.colscale 2 -weight 1
    set prevNameIndex -1
    set size [llength $caption]
    set names [list T E B H G I C]
     
    for {set yrect 0} {$yrect < $size} {incr yrect} {
        
        set curName [lindex $names  $yrect]
        set curcaption [lindex $caption  $yrect]
        set hexcols [QWIKMD::chooseColor $curName]
            
        set hexred [lindex $hexcols 0]
        set hexgreen [lindex $hexcols 1]
        set hexblue [lindex $hexcols 2]
        grid [ttk::label $w.colscale.${yrect}1 -text "$curName" -anchor center] -row $yrect -column 0 -sticky we
        grid [label $w.colscale.${yrect}2 -bg \#${hexred}${hexgreen}${hexblue} -width 3] -row $yrect -column 1 -sticky w -padx 4
        grid [ttk::label $w.colscale.${yrect}3 -text $curcaption] -row $yrect -column 2 -sticky ew
    }
    return $w.colscale
}
##############################################
## Secondary strucutre colors caption based on
## TimeLine
###############################################
proc QWIKMD::chooseColor {intensity} {

  set field_color_type s 
  
  switch -exact $field_color_type {         
    s {
      if { [catch {
        switch $intensity {

          B {set red 180; set green 180; set blue 0}
          C {set red 255; set green 255; set blue 255}
          E {set red 255; set green 255; set blue 100}
          T {set red 70; set green 150; set blue 150}
          G {set red 20; set green 20; set blue 255}
          H {set red 235; set green 130; set blue 235}
          I {set red 225; set green 20; set blue 20}
          default {set red 100; set green 100; set blue 100}
        }
        
      } ] 
         } { #badly formatted file, intensity may be a number
        set red 0; set green 0; set blue 0 
      }
    }
    default {
      set c $colorscale(choice)
      set red $colorscale($c,$intensity,r)
      set green $colorscale($c,$intensity,g)
      set blue $colorscale($c,$intensity,b)
   } 
  }
  
  #convert red blue green 0 - 255 to hex
  set hexred     [format "%02x" $red]
  set hexgreen   [format "%02x" $green]
  set hexblue    [format "%02x" $blue]
  set hexcols [list $hexred $hexgreen $hexblue]

  return $hexcols
}

proc QWIKMD::BrowserButt {} {
    set fil ""
    set fil [tk_getOpenFile -title "Open Molecule:" ]
    
    if {$fil != ""} {
        set QWIKMD::inputstrct $fil
    }
    
}

##############################################
## Update the table in the qwikMD main window 
## with new molecule, or when the type of molecules
## is changed in the Select Resid Window 
## Here is when the macros as set for the first time
## qwikmd_glycan qwikmd_nucleic and qwikmd_protein
###############################################
proc QWIKMD::mainTable {tabid} {
    array unset QWIKMD::chains *
    array unset QWIKMD::index_cmb *
    array set QWIKMD::index_cmb ""
    array set QWIKMD::chains ""

    while {[molinfo $QWIKMD::topMol get numreps] !=  0 } {
        mol delrep [expr [molinfo $QWIKMD::topMol get numreps] -1 ] $QWIKMD::topMol
        
    }
    set sel [atomselect $QWIKMD::topMol "all and not name QWIKMDDELETE"]

    $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb configure -state normal
    $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb delete 0 end

    
    set atomindex 0
    set macrosstr [list]
    set defVal {protein nucleic glycan lipid hetero}
    foreach macros $QWIKMD::userMacros {
        if {[lsearch $defVal [lindex $macros 0]] == -1 } {
            lappend macrosstr [lindex $macros 0] 
        }   
    }
    set numfram [molinfo $QWIKMD::topMol get numframes]
    $QWIKMD::topGui.nbinput.f$tabid.selframe.mNMR.nmr delete 0 end
    for {set i 0} {$i < $numfram} {incr i} {
        
        $QWIKMD::topGui.nbinput.f$tabid.selframe.mNMR.nmr add radiobutton -label "$i" -variable QWIKMD::nmrstep -command {
            molinfo $QWIKMD::topMol set frame $QWIKMD::nmrstep
        }
    }

    set listMol [list]
    foreach chain [$sel get chain] protein [$sel get qwikmd_protein] nucleic [$sel get qwikmd_nucleic] glycan [$sel get qwikmd_glycan] lipid [$sel get qwikmd_lipid] hetero [$sel get qwikmd_hetero]\
     water [$sel get water] macros [$sel get $macrosstr] residue [$sel get residue] {
        lappend listMol [list $chain $protein $nucleic $glycan $lipid $hetero $water $macros $residue]
    }
    set listMol [lsort -unique $listMol]
    set listMol [lsort -index 8 -integer -increasing $listMol]
    set labels [list]
    foreach listEle $listMol {
        set chain [lindex $listEle 0]
        set protein [lindex $listEle 1]
        set nucleic [lindex $listEle 2]
        set glycan [lindex $listEle 3]
        set lipid [lindex $listEle 4]
        set hetero [lindex $listEle 5]
        set water [lindex $listEle 6]
        set macros [lindex $listEle 7]

        set type "protein"
        set typesel "qwikmd_protein"
        
        if {$protein == 1} {
            set type "protein"
            set typesel "qwikmd_protein"
        } elseif {$nucleic == 1} {
            set type "nucleic"
            set typesel "qwikmd_nucleic"
        } elseif {$glycan == 1} {
            set type "glycan"
            set typesel "qwikmd_glycan"
        } elseif {$lipid == 1} {
            set type "lipid"
            set typesel "qwikmd_lipid"
        } elseif {$water == 1} {
            set type "water"
            set typesel "water"
        } elseif {$macros == 1} {
            set macroName [lindex $macrosstr [lsearch $macros 1]]
            set type $macroName
            set typesel $macroName
        } elseif {$hetero == 1} {
            set type "hetero"
            set typesel "qwikmd_hetero"
        }
        if {[lsearch -exact $labels "$chain $typesel"] != -1} {continue}
        set txt "$chain $typesel"
        lappend labels $txt 
    }
    set listMol [list]
    $sel delete

    $QWIKMD::topGui.nbinput.f$tabid.selframe.mCHAIN.chain delete 0 end
    set typeaux ""
    set lineindex 0
    for {set i 0} {$i < [llength $labels]} {incr i} {
        set type [lindex [lindex $labels $i] 1]
        set chain [lindex [lindex $labels $i] 0]
        regsub -all "qwikmd_" $type "" typeaux
        set column 0
        if {[expr $i % 20] == 0} {
            set column 1
        }
        $QWIKMD::topGui.nbinput.f$tabid.selframe.mCHAIN.chain add checkbutton -label "$chain and $typeaux"  -columnbreak $column -variable QWIKMD::chains($i,0) -command QWIKMD::selectChainType

        set QWIKMD::chains($i,1) "$chain and $typeaux"
        set selaux [atomselect $QWIKMD::topMol "chain \"$chain\" and $type"]
        set residues [lsort -unique -integer [$selaux get resid]]
        $selaux delete
        set min [lindex $residues 0]
        set max [lindex $residues end]
        
        set QWIKMD::chains($i,2) "[format %0.0f ${min}] - [format %0.0f ${max}]"
        
        if {[info exists QWIKMD::index_cmb($QWIKMD::chains($i,1),5)] != 1} {
            set auxstrng ""
            if {$type == "protein" || $type == "nucleic" } {
                set auxstrng "chain \"$chain\" and qwikmd_${type}"
            } elseif {$type == "hetero" || $type == "glycan" } {            
                set auxstrng "chain \"$chain\" and qwikmd_${type}"
            } elseif {$type == "water"} {
                set auxstrng "chain \"$chain\" and $type"
            } elseif {$type == "lipid" } {
                set auxstrng "chain \"$chain\" and qwikmd_${type}"  
            } else {
                set auxstrng "chain \"$chain\" and $type"
            }
            set QWIKMD::index_cmb($QWIKMD::chains($i,1),5) $auxstrng
        }
        if {$chain == "W" && $type == "water"} {
            set QWIKMD::chains($i,0) 0
        } elseif {$chain == "I" && $typeaux == "hetero"} {
            set selcur [atomselect $QWIKMD::topMol "chain I"]
            set res [$selcur get ion]
            if {$res != ""} {
                set QWIKMD::chains($i,0) 0
            }
            $selcur delete
        } else {
            $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb insert end [list $chain "[format %0.0f ${min}] - [format %0.0f ${max}]" $typeaux "aux" "aux"]
            mol addrep $QWIKMD::topMol
            set QWIKMD::index_cmb($QWIKMD::chains($i,1),4) [mol repname $QWIKMD::topMol [expr [molinfo $QWIKMD::topMol get numreps] -1] ]

            $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb cellconfigure $lineindex,3 -text [QWIKMD::mainTableCombosStart 0 $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb $lineindex 3 "aux"]
            $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb cellconfigure $lineindex,4 -text [QWIKMD::mainTableCombosStart 0 $QWIKMD::topGui.nbinput.f$tabid.tableframe.tb $lineindex 4 "aux"]
            incr lineindex
            set QWIKMD::chains($i,0) 1

            set sel [atomselect $QWIKMD::topMol $QWIKMD::index_cmb($QWIKMD::chains($i,1),5)]
            set res [$sel get index]
            $sel delete
            if {$res == ""} {
                set QWIKMD::chains($i,0) 0
            } 
        }
    }
    set menu $QWIKMD::topGui.nbinput.f$tabid.selframe.mCHAIN.chain.select   
    if {[winfo exists $menu] == 0} {
        menu $menu
        proc selectAllNon {opt} {
            set val 1
            if {$opt != "all"} {
                set val 0
            }
            set length [expr [llength [array names QWIKMD::chains]] /3]
            for {set i 0} {$i < $length} {incr i} {
                set QWIKMD::chains($i,0) $val
            }
            QWIKMD::selectChainType
        }
        $menu add command -label "All" -command {QWIKMD::selectAllNon "all"}
        $menu add command -label "None" -command {QWIKMD::selectAllNon "none"}
    }
    $QWIKMD::topGui.nbinput.f$tabid.selframe.mCHAIN.chain add cascade -menu $menu -label "Select"
    
    
    

    

    set QWIKMD::warnresid 0
    mouse mode rotate
    return 1
}

proc QWIKMD::selectChainType {} {
    set tabid [$QWIKMD::topGui.nbinput index current]
    set level basic
    if {$tabid == 1} {
        set level advanced
    }
    $QWIKMD::chainMenu($level) configure -state disabled
    QWIKMD::reviewTable [expr [lsearch [$QWIKMD::topGui.nbinput tabs] [$QWIKMD::topGui.nbinput select]] + 1 ]
    set length [expr [llength [array names QWIKMD::chains]] /3]
    for {set i 0} {$i < $length} {incr i} {
        set type [lindex $QWIKMD::chains($i,1) 2]
        set chain [lindex $QWIKMD::chains($i,1) 0]
        set rows [$QWIKMD::selresTable get 0 end]
        set index [lsearch -all [$QWIKMD::selresTable get 0 end] "*$chain $type"]
        foreach idx $index {
            if {$QWIKMD::chains($i,0) == 0} {
                $QWIKMD::selresTable rowconfigure $idx -hide 1
            } else {
                $QWIKMD::selresTable rowconfigure $idx -hide 0
            }
        }
    }
    $QWIKMD::chainMenu($level) configure -state normal
}

proc QWIKMD::LoadButt {fil} {
    
    if {[file isfile [lindex $fil 0] ] == 1} {
        set QWIKMD::topMol [mol new [lindex $fil 0] waitfor all]
        if {[llength $fil] > 1} {   
            mol addfile [lindex $fil 1] waitfor all
        } 
    } else {
        set QWIKMD::topMol [mol new $fil waitfor all]
    }
    set QWIKMD::nmrstep 0
    molinfo $QWIKMD::topMol set frame $QWIKMD::nmrstep
        
}   

##############################################
## Proc to prepare psf and pdb file and all the
## config files
###############################################
proc QWIKMD::tableSearchCheck {resid tbl row column text } { 
    if {[$tbl cellcget $row,0 -text] == $resid && [$tbl rowcget $row -hide] == 0} {
        return 1
    } else {
        return 0
    }
}
proc QWIKMD::PrepareBttProc {file} {
    global env
    set rtrn 0
    set tabid [$QWIKMD::topGui.nbinput index current]
    set resTable $QWIKMD::selresTable


    if {$QWIKMD::rename != ""} {
        for {set i 0} {$i < [llength $QWIKMD::rename]} {incr i} {
            set residchain [split [lindex $QWIKMD::rename $i] "_" ]
            set resid [lindex $residchain 0]
            set chain [lindex $residchain end]
            if {[lsearch -exact $QWIKMD::renameindex [lindex $QWIKMD::rename $i]] == -1 \
                && [lsearch -exact $QWIKMD::delete [lindex $QWIKMD::rename $i]] == -1 && [$resTable searchcolumn 2 $chain -check [list QWIKMD::tableSearchCheck $resid] ] > -1 } {
                set rtrn 1
                break
            }
        }
    }

    if {$QWIKMD::warnresid == 1} {
        if {[lindex $QWIKMD::topoerror 0] > 0} {
            tk_messageBox -message "One or more residues could not be identified\nPlease rename or delete them in \"Structure Manipulation/Check\" window" -title "Residues Topology" -icon warning -type ok 
        } else {
            tk_messageBox -message "One or more warnings are still active.\nPlease go to \"Structure Manipulation/Check\" window" -title "Structure Check Warnings" -type ok
        }
        return
    }

    if {$tabid == 1} {
        set nlines [expr [lindex [split [$QWIKMD::selresPatcheText index end] "."] 0] -1]
        set patchtext [split [$QWIKMD::selresPatcheText get 1.0 $nlines.end] "\n"]
        set patchaux [list]
        if {[lindex $patchtext 0] != ""} {
            foreach patch $patchtext {
                if {$patch != ""} {
                    if {[llength $patch] == 3 || [llength $patch] == 5} {
                        lappend patchaux $patch
                    } else {
                        tk_messageBox -message "The modification list is not in the correct format.\nPlease revise and prepare again." -type ok -icon warning
                        set QWIKMD::patchestr ""
                        return 
                    }
                }
            }
            set QWIKMD::patchestr $patchaux
        }
    }
    # if {[$QWIKMD::torsionOutliearcolor cget -background] == "yellow" || [$QWIKMD::torsionMarginalcolor cget -background] == "yellow" || [llength $QWIKMD::gaps] > 0 || $QWIKMD::chirerror > 0} {
    #   set answer [tk_messageBox -message "One or more warnigs are still active.\nPlease go to \"Structure Manipulation/Check\" window" -title "Structure Check Warnings" -type okcancel]
    #   if {$answer == "cancel"} {
    #       return
    #   }
    # }
    if {$QWIKMD::run == "SMD" && ($QWIKMD::anchorRessel == "" || $QWIKMD::pullingRessel == "")} {
        tk_messageBox -message "Anchor/Pulling residues were not defined. Please select them pressing \"Anchor Residues\" and \"Pulling Residues\" buttons" -title "Anchor/Pulling Residues" -icon warning -type ok 
        return
    }

    if {$file != ""} {
        $QWIKMD::topGui configure -cursor watch; update 
        set numtabs [llength [$QWIKMD::topGui.nbinput tabs]]
        for {set i 0} {$i < $numtabs} {incr i} {
            if {$tabid != $i} {
                $QWIKMD::topGui.nbinput tab $i -state disabled
            }
        }
        set outputfoldername [file rootname $file]

        set QWIKMD::outPath "${outputfoldername}"

        if {[file exists $outputfoldername]== 1} {
            cd $::env(VMDDIR)
            file delete -force -- $outputfoldername
        }
        file mkdir $outputfoldername

        set QWIKMD::textLogfile [open "${outputfoldername}/[file tail $outputfoldername].infoMD" w+] 

        puts $QWIKMD::textLogfile [QWIKMD::introText]

        $QWIKMD::runbtt configure -state disabled

        for {set i 0} {$i < [llength $QWIKMD::delete]} {incr i} {

            set index [lsearch -exact $QWIKMD::mutindex [lindex $QWIKMD::delete $i]]
            if {$index != -1} {
                set QWIKMD::mutindex [lreplace $QWIKMD::mutindex $index $index]
            }

            set index [lsearch -exact $QWIKMD::protindex [lindex $QWIKMD::delete $i]]
            if {$index != -1} {
                set QWIKMD::protindex [lreplace $QWIKMD::protindex $index $index]
            }

            set index [lsearch -exact $QWIKMD::renameindex [lindex $QWIKMD::delete $i]]
            if {$index != -1} {
                set QWIKMD::renameindex [lreplace $QWIKMD::renameindex $index $index]
            }
        }

        puts $QWIKMD::textLogfile [QWIKMD::structPrepLog]
        puts $QWIKMD::textLogfile [QWIKMD::deleteLog]
        
        cd $outputfoldername

        if {[file exists setup]!= 1} {
            file mkdir setup
        }

        if {[file exists run]!= 1} {
            file mkdir run
        }
        set pdbfile ""
        set stfile [lindex [molinfo $QWIKMD::topMol get filename] 0]
        set name "[file tail [file root [lindex $stfile 0] ] ]_original.pdb"
        if {[file isfile [lindex $QWIKMD::inputstrct 0] ] == 1 } {
            if {[llength $QWIKMD::inputstrct] == 1} {
                set pdbfile ${QWIKMD::inputstrct}
            } elseif {[llength $QWIKMD::inputstrct] == 2} {
                set pdbfile [lindex $QWIKMD::inputstrct [lsearch $QWIKMD::inputstrct "*.pdb"]]
            }
            if {$pdbfile != ""} {
                file copy -force ${QWIKMD::inputstrct} setup/$name
            }
        } else {
            if {[llength $QWIKMD::inputstrct] == 2} {
                set sel [atomselect top "all and not name QWIKMDDELETE"]
                $sel writepdb setup/$name
                $sel delete
            } else {
                #From autopsf
                set url [format "http://www.rcsb.org/pdb/downloadFile.do?fileFormat=pdb&compression=NO&structureId=%s" $QWIKMD::inputstrct]
                vmdhttpcopy $url setup/$name
                set failed 0
                if {[file exists setup/$name] ==1} {
                    if {[file size setup/$name] == 0} {
                        set failed 1
                    }
                } else {
                    set failed 0
                }
                if {$failed == 1} {
                    file delete -force setup/$name
                    tk_messageBox -message "Could not find download the original pdb file from PDB DataBank." -icon error -type ok
                    return
                }
            }
            
        }


        foreach par $QWIKMD::TopList {
            set f [open $par "r"]
            set out [open "setup/[file tail $par]" w+ ]
            set txt [read -nonewline ${f}]
            puts $out $txt
            close $f
            close $out
        }

        foreach par $QWIKMD::ParameterList {
            set f [open $par "r"]
            set out [open "run/[file tail $par]" w+ ]
            set txt [read -nonewline ${f}]
            puts $out $txt
            close $f
            close $out
        }

        if {$QWIKMD::basicGui(live) == 1} {
            set QWIKMD::dcdfreq 1000
            set QWIKMD::load 0
        } else {
            set QWIKMD::dcdfreq 10000
        }
        
        set step 0
        set prefix [file rootname [file tail $file] ]
        
        # Create NAMD input files, but not for MDFF protocol. MDFF protocol are created using MDFF Gui 
        if {$QWIKMD::run != "MDFF"} {
            if {$tabid == 0} {  
                set QWIKMD::confFile ""
                set text ""
                if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,equi) == 1} {
                    lappend QWIKMD::confFile "qwikmd_equilibration_$step"
                    incr step
                }
                
                if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,md) == 1} {
                    lappend QWIKMD::confFile "qwikmd_production_$step"
                     incr step
                }
                if {$QWIKMD::run == "SMD"} {
                    if {$QWIKMD::basicGui(prtcl,$QWIKMD::run,smd) == 1} {
                        lappend QWIKMD::confFile "qwikmd_production_smd_$step"
                        incr step
                    } 
                }
            } else {
                set QWIKMD::confFile [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
            }
            set QWIKMD::prevconfFile $QWIKMD::confFile
        }
        set strct [QWIKMD::PrepareStructures $prefix $QWIKMD::textLogfile]
        if {[string is integer [lindex $strct 0]] == 1} {
            tk_messageBox -message "Error during structure preparation: [lindex $strct 1]." -icon error
            
            flush $QWIKMD::textLogfile
            close $QWIKMD::textLogfile
            return 1
        }

        if {[file exists "$env(TMPDIR)/Renumber_Residues.txt"] == 1} {
            set renfile [open "$env(TMPDIR)/Renumber_Residues.txt" r]
            set lines [read $renfile]
            close $renfile
            set lines [split $lines "\n"]
            puts $QWIKMD::textLogfile "\nRenumbering Residues Reference Table"
            foreach str $lines {
                puts $QWIKMD::textLogfile $str
            }
            file copy -force "$env(TMPDIR)/Renumber_Residues.txt" ${QWIKMD::outPath}/setup/
        }
        puts $QWIKMD::textLogfile "[string repeat "=" 81]\n\n"

        set QWIKMD::prepared 1
        if {$QWIKMD::run != "MDFF"} {
            if {$tabid == 0} {
                set QWIKMD::maxSteps [list]
            }
            for {set i 0} {$i < [llength $QWIKMD::confFile]} {incr i} {
                QWIKMD::NAMDGenerator $strct $i
            }
            puts $QWIKMD::textLogfile [QWIKMD::printMD]
        } else {
            set QWIKMD::prevconfFile "MDFF"
        }
        
        puts $QWIKMD::textLogfile "================================== MD Analysis ====================================\n\n"
        set list [molinfo list]
        cd $QWIKMD::outPath/run/
       

        set QWIKMD::inputstrct $strct

        set QWIKMD::nmrstep 0
        set input $QWIKMD::basicGui(workdir,0)
        QWIKMD::SaveInputFile $QWIKMD::basicGui(workdir,0)
        ## Change prepared variable to 0 to avoid the notification of killing the MD simulations
        set QWIKMD::prepared 0
        set logFile $QWIKMD::textLogfile
        QWIKMD::resetBtt 1
        set QWIKMD::basicGui(workdir,0) $input
        for {set i 0} {$i < $numtabs} {incr i} {
                if {$tabid != $i} {
                    $QWIKMD::topGui.nbinput tab $i -state normal
                }
            }
        source $input
        QWIKMD::mainTable [expr [$QWIKMD::topGui.nbinput index current] +1]
        QWIKMD::reviewTable [expr [$QWIKMD::topGui.nbinput index current] +1]
        QWIKMD::SelResid
        QWIKMD::ChangeSolvent
        
        ## QWIKMD::textLogfile is cleaned during reset
        set QWIKMD::textLogfile $logFile
        
        if {$tabid == 1 && $QWIKMD::run != "SMD"} {
            set sel [atomselect $QWIKMD::topMol "all"]
            $sel set beta 0
            $sel set occupancy 0
            $sel writepdb [lindex $strct 1]
            $sel delete
        }
        
        
        if {$QWIKMD::prepared == 1 && $QWIKMD::run != "MDFF"} {
            $QWIKMD::runbtt configure -state normal
        }
        $QWIKMD::runbtt configure -text "Start [QWIKMD::RunText]"
        set numframes [molinfo $QWIKMD::topMol get numframes]
        QWIKMD::updateTime load


        # if {[$QWIKMD::topGui.nbinput index current] == 0} {
        #     $QWIKMD::topGui.nbinput tab 1 -state disable
        #     $QWIKMD::basicGui(prtcl,$QWIKMD::run,mdbtt) configure -state disable
        #     $QWIKMD::basicGui(prtcl,$QWIKMD::run,equibtt) configure -state disable
        #     if {$QWIKMD::run == "SMD"} {
        #         $QWIKMD::basicGui(prtcl,$QWIKMD::run,smdbtt) configure -state disable
        #     }
        # } else {
        #     $QWIKMD::topGui.nbinput tab 0 -state disable
        # }

        # foreach note [lrange $QWIKMD::notebooks 1 2] {
        #     $note state disabled

        # }
        ttk::style configure WorkDir.TEntry -foreground black
        $QWIKMD::basicGui(workdir,1) configure -state disable
        $QWIKMD::basicGui(workdir,2) configure -state disable

        flush $QWIKMD::textLogfile
        close $QWIKMD::textLogfile
        QWIKMD::lockGUI
        if {$QWIKMD::run == "MDFF"} {
            # set MDFF settings according to QwikMD parameters, working directory, and options chosen by the user
            QWIKMD::updateMDFF
        }
        return 0
    } else {
        QWIKMD::saveBut
        if {$QWIKMD::basicGui(workdir,0) != ""} {
            QWIKMD::PrepareBttProc $QWIKMD::basicGui(workdir,0)
        }
        
    }
}
# set MDFF settings according to QwikMD parameters, working directory, and options chosen by the user
proc QWIKMD::updateMDFF {} {
    if {$QWIKMD::basicGui(live) == 1} {
        QWIKMD::selectProcs
    }
    tk_messageBox -message "You will be redirected to MDFF GUI plug-in where you can prepare and preform MDFF simulations." -icon info -type ok
    MDFFGUI::gui::mdffgui           
    set MDFFGUI::settings::MolID $QWIKMD::topMol
    if {[info exists MDFFGUI::settings::QwikMDLogFile] == 1} {
        set MDFFGUI::settings::QwikMDLogFile "${QWIKMD::outPath}/[file tail ${QWIKMD::outPath}].infoMD"
    }
    set ::MDFFGUI::settings::FixedPDBSelText "[$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget 0,0 -text]"
    if {[$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget 0,1 -text] != "none"} {
        set ::MDFFGUI::settings::SSRestraints 1
    } else {
        set ::MDFFGUI::settings::SSRestraints 0
    }
    if {[$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget 0,2 -text] != "none"} {
        set ::MDFFGUI::settings::ChiralityRestraints 1
    } else {
        set ::MDFFGUI::settings::ChiralityRestraints 0
    }
    if {[$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget 0,3 -text] != "none"} {
        set ::MDFFGUI::settings::CispeptideRestraints 1
    } else {
        set ::MDFFGUI::settings::CispeptideRestraints 0
    }
    set MDFFGUI::settings::SimulationName $QWIKMD::confFile
    set plist [list]
    foreach par $QWIKMD::ParameterList {
        lappend plist "$QWIKMD::outPath/run/[file tail $par]"
    }
    set MDFFGUI::settings::ParameterList $plist
    set ::MDFFGUI::settings::Temperature [expr $QWIKMD::basicGui(temperature,0) + 273]
    set ::MDFFGUI::settings::FTemperature [expr $QWIKMD::basicGui(temperature,0) + 273]
    set ::MDFFGUI::settings::Minsteps $QWIKMD::advGui(mdff,min)
    set ::MDFFGUI::settings::Numsteps $QWIKMD::advGui(mdff,mdff)
    switch $QWIKMD::advGui(solvent,0) {
        Vacuum {
            set MDFFGUI::settings::PBCorGBIS ""
        }
        Implicit {
            set MDFFGUI::settings::PBCorGBIS "-gbis"
        }
        Explicit {
            set MDFFGUI::settings::PBCorGBIS "-pbc"
        }

    }
    if {$QWIKMD::basicGui(live) == 1} {
        set ::MDFFGUI::settings::IMD 1
        set ::MDFFGUI::settings::IMDWait 1
        set MDFFGUI::settings::IMDProcs $QWIKMD::numProcs
    }
    set MDFFGUI::settings::CurrentDir $QWIKMD::outPath/run/
    $QWIKMD::topGui.nbinput.f2.fcontrol.fcolapse.f1.imd.button_Pause configure -state disable
    $QWIKMD::topGui.nbinput.f2.fcontrol.fcolapse.f1.imd.button_Finish configure -state disable
    $QWIKMD::topGui.nbinput.f2.fcontrol.fcolapse.f1.imd.button_Detach configure -state disable
    wm iconify $QWIKMD::topGui
}

proc QWIKMD::lockGUI {} {
    set tabid [$QWIKMD::topGui.nbinput index current]
         
    set level basic
    if {$tabid == 1} {
        set level advanced
    }
    if {$level == "basic"} {
        $QWIKMD::basicGui(solvent,$QWIKMD::run) configure -state disabled
        $QWIKMD::basicGui(saltions,$QWIKMD::run) configure -state disabled
        $QWIKMD::basicGui(saltconc,$QWIKMD::run) configure -state disabled
        $QWIKMD::basicGui(prtcl,$QWIKMD::run,mdbtt) configure -state disable
        $QWIKMD::basicGui(prtcl,$QWIKMD::run,equibtt) configure -state disable
        if {$QWIKMD::run == "SMD"} {
            $QWIKMD::basicGui(prtcl,$QWIKMD::run,smdbtt) configure -state disable
        }
        $QWIKMD::topGui.nbinput tab 1 -state disable
    } else {
        $QWIKMD::advGui(solvent,$QWIKMD::run) configure -state disabled
        $QWIKMD::advGui(solvent,minbox,$QWIKMD::run) configure -state disable   
        $QWIKMD::advGui(saltions,$QWIKMD::run) configure -state disabled
        $QWIKMD::advGui(saltconc,$QWIKMD::run) configure -state disabled
        $QWIKMD::advGui(solvent,boxbuffer,$QWIKMD::run,entry) configure -state disabled
        $QWIKMD::topGui.nbinput tab 0 -state disable
    }
    foreach note [lrange $QWIKMD::notebooks 1 2] {
        $note state disabled
    }
    set QWIKMD::tablemode "inspection"
    
    if {[winfo exists $QWIKMD::selResGui] == 1} {
        QWIKMD::tableModeProc
        QWIKMD::lockSelResid 0
    }
    incr tabid
    $QWIKMD::topGui.nbinput.f$tabid.fb.fcolapse.f1.preparereset.live configure -state disable
    $QWIKMD::nmrMenu($level) configure -state disabled
    set numcols [$QWIKMD::advGui(protocoltb,$QWIKMD::run) columncount]
    for {set i 0} {$i < $numcols} {incr i} {$QWIKMD::advGui(protocoltb,$QWIKMD::run) columnconfigure $i -editable false}
    $QWIKMD::topGui configure -cursor {}; update    
     
}

proc QWIKMD::addNAMDCheck {step} {
    set prefix [lindex $QWIKMD::confFile $step]
    set filename [lindex $QWIKMD::confFile $step]
    set namdfile [open ${QWIKMD::outPath}/run/${filename}.conf a]
    set file "[lindex $QWIKMD::confFile $step].check"

    puts $namdfile  "set file \[open ${file} w+\]"

    puts $namdfile "set done 1"
    set str $QWIKMD::run
    
    puts $namdfile "if \{\[file exists $prefix.restart.coor\] != 1 || \[file exists $prefix.restart.vel\] != 1 || \[file exists $prefix.restart.xsc\] != 1 \} \{"
    puts $namdfile "\t set done 0"
    puts $namdfile "\}"

    puts $namdfile "if \{\$done == 1\} \{"
    puts $namdfile "\tputs \$file \"DONE\"\n    flush \$file\n  close \$file"
    puts $namdfile "\} else \{"
    puts $namdfile "\tputs \$file \"One or more files failed to be written\"\n   flush \$file\n  close \$file"
    puts $namdfile "\}"
    close $namdfile

}

# proc QWIKMD::addConstOccup {conf input output pdb start midle end} {

#     set line ""
#     if {$output == "SMD_Index.pdb" && $input == "SMD_anchorIndex.txt"} {
#         append line "set do 0\n"
#         append line "if \{\[file exists $output \] == 0\} \{\n"
#     } elseif {$output == "SMD_Index.pdb" && $input == "SMD_pullingIndex.txt"} {
#         append line "if \{\$do == 1\} \{\n"
#     }
    
#     append line "\tset pdb $pdb\n"
#     append line "\tset file \[open \$pdb r\]\n"
#     append line "\tset line \[read -nonewline \$file\]\n"
#     append line "\tset line \[split \$line \"\\n\"\]\n"
#     append line "\tclose \$file\n"
#     append line "\tset out \[open $output w+\]\n"
#     append line "\tset indexfile \[open $input r\]\n"
#     append line "\tset vmdindexes \[read -nonewline \$indexfile\]\n"
#     append line "\tclose \$indexfile\n"
#     append line "\tforeach ind \$vmdindexes \{\n"
#     append line "\t\tset index \[lsearch -index 1 \[lrange \$line 0 \[expr \[\llength \$line\] -1\] \] \$ind\]\n"
#     append line "\t\tif \{\$index > -1\} \{\n"
#     append line "\t\t\tset lineauxformat \"\"\n"
#     append line "\t\t\tset lineauxformat \[string range \[lindex \$line \$index\] $start $midle\]\n"
#     append line "\t\t\tappend lineauxformat \[format  %+*s 6 1.00\]\n"
#     append line "\t\t\tappend lineauxformat \[string range \[lindex \$line \$index\] $end end\]\n"
#     append line "\t\t\tlset line \$index \$lineauxformat\n"
#     append line "\t\t\}\n"
#     append line "\t\}\n"
#     append line "\tfor \{set i 0\} \{\$i < \[llength \$line\]\} \{incr i\} \{\n"
#     append line "\t\tputs \$out \[lindex \$line \$i\]\n"
#     append line "\t\}\n"
#     append line "\tclose \$out\n"
#     if {$output == "SMD_Index.pdb"} {
#         append line "\tset do 1"
#         append line "\}\n"
#     }
#     set file [open $conf r]
#     append line [read $file]
#     close $file
#     set file [open $conf w+]
#     puts $file $line
#     close $file
# }


proc QWIKMD::addFirstTimeStep {step} {

    set line ""

    append line "set xsc [lindex $QWIKMD::confFile [expr $step -1]].xsc\n"
    append line "if \{\[file exists \$xsc\] == 0\} \{set xsc [lindex $QWIKMD::confFile [expr $step -1]].restart.xsc\}\n"
    append line "set file \[open \$xsc r\]\n"
    append line "set line \[read -nonewline \$file\]\n"
    append line "set line \[split \$line \"\\n\"\]\n"
    append line "close \$file\n"
    append line "firstTimeStep \[lindex \[lindex \$line 2\] 0\]"
    
    return $line
}


proc QWIKMD::infoWindow {name text title} {
    
    set wname ".$name"
    if {[winfo exists $wname] != 1} {
        toplevel $wname
    } else {
        wm deiconify $wname
        return
    }
    wm geometry $wname 600x400
    grid columnconfigure $wname 0 -weight 2
    grid rowconfigure $wname 0 -weight 2
    ## Title of the windows
    wm title $wname $title ;# titulo da pagina

    grid [ttk::frame $wname.txtframe] -row 0 -column 0 -sticky nsew
    grid columnconfigure  $wname.txtframe 0 -weight 1
    grid rowconfigure $wname.txtframe 0 -weight 1

    grid [text $wname.txtframe.info -wrap word -width 420 -bg white -yscrollcommand [list $wname.txtframe.scr1 set] -xscrollcommand [list $wname.txtframe.scr2 set] -exportselection true] -row 0 -column 0 -sticky nsew -padx 2 -pady 2
    
    
    for {set i 0} {$i <= [llength $text]} {incr i} {
        set txt [lindex [lindex $text $i] 0]
        set font [lindex [lindex $text $i] 1]
        $wname.txtframe.info insert end $txt
        set ini [$wname.txtframe.info search -exact $txt 1.0 end]
        
        set line [split $ini "."]
        set fini [expr [lindex $line 1] + [string length $txt] ]
         
        $wname.txtframe.info tag add $wname$i $ini [lindex $line 0].$fini
        if {$font == "title"} {
            set fontarg "helvetica 15 bold"
        } elseif {$font == "subtitle"} {
            set fontarg "helvetica 12 bold"
        } else {
            set fontarg "helvetica 12"
        } 
        $wname.txtframe.info tag configure $wname$i -font $fontarg
    }


        ##Scrool_BAr V
    scrollbar $wname.txtframe.scr1  -orient vertical -command [list $wname.txtframe.info yview]
    grid $wname.txtframe.scr1  -row 0 -column 1  -sticky ens

    ## Scrool_Bar H
    scrollbar $wname.txtframe.scr2  -orient horizontal -command [list $wname.txtframe.info xview]
    grid $wname.txtframe.scr2 -row 1 -column 0 -sticky swe

    grid [ttk::frame $wname.linkframe] -row 1 -column 0 -sticky ew -pady 2 -padx 2
    grid columnconfigure $wname.linkframe 0 -weight 2
    grid rowconfigure $wname.linkframe 0 -weight 2

    grid [tk::text $wname.linkframe.text -bg [ttk::style lookup $wname.linkframe -background ] -width 100 -height 1 -relief flat -exportselection yes -foreground blue] -row 1 -column 0 -sticky w
    $wname.linkframe.text configure -cursor hand1
    $wname.linkframe.text see [expr [string length $QWIKMD::link] * 1.0 -1]
    $wname.linkframe.text tag add link 1.0 [expr [string length $QWIKMD::link] * 1.0 -1]
    $wname.linkframe.text insert 1.0 $QWIKMD::link link
    $wname.linkframe.text tag bind link <Button-1> {
         if {$tcl_platform(platform) eq "windows"} {
                set command [list {*}[auto_execok start] {}]
                set url [string map {& ^&} $url]
            } elseif {$tcl_platform(os) eq "Darwin"} {
                set command [list open]
            } else {
                set command [list xdg-open]
            }
            exec {*}$command $QWIKMD::link &
      
      }
      bind link <Button-1> <Enter>
      $wname.linkframe.text tag configure link -foreground blue -underline true
      $wname.linkframe.text configure -state disabled

     
     $wname.txtframe.info configure -state disable
}

proc QWIKMD::getrepnum {repname} {
    return [mol repindex $QWIKMD::topMol $repname]
}
proc QWIKMD::changeBCK {} {
    if {$QWIKMD::basicGui(desktop) == "white"} {
        color Display FPS black 
        color Axes Labels black 
    } elseif {$QWIKMD::basicGui(desktop) != ""} {
        color Display FPS white 
        color Axes Labels white 
    }
    if {$QWIKMD::basicGui(desktop) == "gradient"} {
        display backgroundgradient on
        color Display Background black
    } elseif {$QWIKMD::basicGui(desktop) != ""} {
        display backgroundgradient off
        color Display Background $QWIKMD::basicGui(desktop)
    }
}
proc QWIKMD::balloon {w help} {
    bind $w <Any-Enter> "after 5000 [list QWIKMD::balloon:show %W [list $help]]"
    bind $w <Any-Leave> "destroy %W.balloon"
}
  
proc QWIKMD::balloon:show {w arg} {
    if {[eval winfo containing  [winfo pointerxy .]]!=$w} {return}
    set top $w.balloon
    catch {destroy $top}
    toplevel $top -bd 1 -bg black
    wm overrideredirect $top 1
    if {[string equal [tk windowingsystem] aqua]}  {
        ::tk::unsupported::MacWindowStyle style $top help none
    }   
    pack [message $top.txt -aspect 10000 -bg lightyellow \
            -font fixed -text $arg]
    set wmx [winfo rootx $w]
    set wmy [expr [winfo rooty $w]+[winfo height $w]]
    wm geometry $top \
      [winfo reqwidth $top.txt]x[winfo reqheight $top.txt]+$wmx+$wmy
    raise $top
}

proc QWIKMD::tableballoon:show {tbl} {
    set w [$tbl labeltag]
    set col [tablelist::getTablelistColumn %W]
    set help 0
    
    switch $col {
        0 {
            set help [QWIKMD::selTabelChainBL]
        }
        1 {
            set help [QWIKMD::selTabelResidBL]
        }
        2 {
            set help [QWIKMD::selTabelTypeBL]
        }
        3 {
            set help [QWIKMD::selTabelRepBL]
        }
        4 {
            set help [QWIKMD::selTabelColorBL]
        }
        default {
            set help $col
        }
    }
    bind $w <Any-Enter> "after 5000 [list QWIKMD::balloon:show %W [list $help]]"
    bind $w <Any-Leave> "destroy %W.balloon"
}

proc QWIKMD::createInfoButton {frame row column} {
    image create photo QWIKMD::logo -data [QWIKMD::infoImage]
    grid [ttk::label $frame.info -image QWIKMD::logo -anchor center -background $QWIKMD::bgcolor] -row $row -column $column -sticky e -padx 0 -pady 0

    $frame.info configure -cursor hand1
}

proc QWIKMD::lockUnlockProc {index} {
    if {$QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) == 0} {
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) 1
        set state 0
        set color grey
    } else {
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$index,lock) 0
        set state 1
        set color black
    }
    set numcols {0 1 3 4 5}
    for {set i 0} {$i < [llength $numcols]} {incr i} {

        $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $index,[lindex $numcols $i] -editable $state  
        $QWIKMD::advGui(protocoltb,$QWIKMD::run) cellconfigure $index,[lindex $numcols $i] -foreground $color -selectforeground $color  
    }
}

proc QWIKMD::checkProc {line} {
    set QWIKMD::confFile [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 0]
    set QWIKMD::maxSteps [$QWIKMD::advGui(protocoltb,$QWIKMD::run) getcolumns 1]

    set values {Minimization Annealing Equilibration MD SMD}
    #set row [expr [$QWIKMD::advGui(protocoltb,$QWIKMD::run) index end] -1]
    set row $line
    set current [$QWIKMD::advGui(protocoltb,$QWIKMD::run) cellcget $row,0 -text]
    set index [lsearch $values $current]
    if {$index == -1} {
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$row,lock) 0
        
    } else {
        set QWIKMD::advGui(protocoltb,$QWIKMD::run,$row,lock) 1
        
    }
    QWIKMD::lockUnlockProc $row
}
##### Base64 string code for the info button logo. The original image is a GIF image and the decoded to base64.
##### If the orignal image is a png it will not work in linux (tested in a Centos 6.5)
proc QWIKMD::infoImage {} {
    set image {R0lGODlhFAAUAOefADVLnDdNnThOnjlPnjxSoD5UoUJXo0FYpENYo0RbpkZdp0tjq01jqlFkqlBorlRnrVdsr\
        1VtsVhtsFlws1pxtFxxslxys1pztV91tV92tl94uGF6uWV+vGZ/vWiAvGmBvmqDv3GEvW6GwG+IwnuKv3uKwHeMwnSOxn\
        mOxICPwniRyH6QxIGQw4CRxHmTyYaVxX6YzICYy4CZzIGazYKbzoObzYWbzIWczYqcy4Oe0I+cyYef0JGgzIij04qj04+i\
        z5Gjz5aizI6k0pOl0Y2p14+r2JCs2ZWr1ZGs2ZSs15iv2J+u1aCu1Z6v1pqw2Z+w1pyy2pqz3KCy2J6z2qex1Zu03Kmy1a\
        K23KS22q+52au83ra/3LbB3rbC4LjC3rnD4LvF4LzH4sHI4b/M5cDO6MHQ6MbQ58bS6s7U6MvV6c7W6s/W6c7Y69PY6tbb7\
        Njc7Njd7dje7tne7dXf79jf79rf7d3i797i8Nzj8d3k8d7k8eHk8N/m8+Lo8+Pp9OXp8+nu9+zu9e3v9u7x+O/x+O7y+PDy+\
        PHz+PL0+fL1+vT1+fT2+vX3+/b3+/b4+/f4+/j5+/j5/Pj6/Pn6/Pr6/Pn7/fr7/Pv7/fv8/fz8/fz8/vz9/v39/v3+/v7+/v\
        //////////////////////////////////////////////////////////////////////////////////////////////////\
        //////////////////////////////////////////////////////////////////////////////////////////////////\
        ////////////////////////////////////////////////////////////////////////////////////////////////\
        ///////////////////////////////////////////////////////////////////////////////////////////////y\
        H+EUNyZWF0ZWQgd2l0aCBHSU1QACH5BAEKAP8ALAAAAAAUABQAAAj+AI0IHEiwYEEiCBMmLBKlChKFCXtInNijDCBNkiZdqk\
        SGYo8cIHP4UOIJE58rSaBMOcNojpOQMGLCoOEJz4yYRwp50rLDUx+ZLoLWWORHSFAXYzh5khE0EZagJ6LqWXQjalQVdKTEiM\
        omT9QRI2wwMgMWbJpLiDiVfTIJLAgQahw1eQuCkacwmRD9eDvE01sOHAYxEgF4CacOXTxF+gAYyCbAGjRwsrQhMgoPGiZ5ih\
        NZwxdCkS9c8ORJtGkTiBzhMI2JiegIEQQ9ogAbNhhPjybA5oIpA2wHDrwY4gEceCdPaICv8LSm+IIFIf7Yeb7AQqNJLRZg8H\
        SH+gIF4LNWPGIhQcGLRofckG4DoQJ4BQfiJ9DxaEv8Ops8NSrBAI6c+AcQIKCADxziyR46pEBCEIFAIgYCBQgYwIQUGmDFG5\
        5QQokiVDRA4YQAhCjiiAMIMOKIAQEAOw==}
return $image
}



