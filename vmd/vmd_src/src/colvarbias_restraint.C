// -*- c++ -*-

#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvarbias_restraint.h"


colvarbias_restraint::colvarbias_restraint(char const *key)
  : colvarbias(key)
{
}


int colvarbias_restraint::init(std::string const &conf)
{
  colvarbias::init(conf);

  if (cvm::debug())
    cvm::log("Initializing a new restraint bias.\n");

  // TODO move these initializations to constructor and let get_keyval
  // only override existing values
  target_nstages = 0;
  target_nsteps = 0;
  force_k = 0.0;

  get_keyval(conf, "forceConstant", force_k, 1.0);

  {
    // get the initial restraint centers
    colvar_centers.resize(colvars.size());
    colvar_centers_raw.resize(colvars.size());
    size_t i;

    enable(f_cvb_apply_force);

    for (i = 0; i < colvars.size(); i++) {
      colvar_centers[i].type(colvars[i]->value());
      colvar_centers_raw[i].type(colvars[i]->value());
      if (cvm::debug()) {
        cvm::log("colvarbias_restraint: center size = "+
                 cvm::to_str(colvar_centers[i].vector1d_value.size())+"\n");
      }
    }
    if (get_keyval(conf, "centers", colvar_centers, colvar_centers)) {
      for (i = 0; i < colvars.size(); i++) {
        if (cvm::debug()) {
          cvm::log("colvarbias_restraint: parsing initial centers, i = "+cvm::to_str(i)+".\n");
        }

        colvar_centers[i].apply_constraints();
        colvar_centers_raw[i] = colvar_centers[i];
      }
    } else {
      colvar_centers.clear();
      cvm::error("Error: must define the initial centers of the restraints.\n");
    }

    if (colvar_centers.size() != colvars.size()) {
      cvm::error("Error: number of centers does not match "
                 "that of collective variables.\n");
    }
  }

  {
    if (cvm::debug()) {
      cvm::log("colvarbias_restraint: parsing target centers.\n");
    }

    size_t i;
    if (get_keyval(conf, "targetCenters", target_centers, colvar_centers)) {
      if (colvar_centers.size() != colvars.size()) {
        cvm::error("Error: number of target centers does not match "
                   "that of collective variables.\n");
      }
      b_chg_centers = true;
      for (i = 0; i < target_centers.size(); i++) {
        target_centers[i].apply_constraints();
      }
    } else {
      b_chg_centers = false;
      target_centers.clear();
    }
  }

  if (get_keyval(conf, "targetForceConstant", target_force_k, 0.0)) {
    if (b_chg_centers)
      cvm::error("Error: cannot specify both targetCenters and targetForceConstant.\n");

    starting_force_k = force_k;
    b_chg_force_k = true;

    get_keyval(conf, "targetEquilSteps", target_equil_steps, 0);

    get_keyval(conf, "lambdaSchedule", lambda_schedule, lambda_schedule);
    if (lambda_schedule.size()) {
      // There is one more lambda-point than stages
      target_nstages = lambda_schedule.size() - 1;
    }
  } else {
    b_chg_force_k = false;
  }

  if (b_chg_centers || b_chg_force_k) {
    get_keyval(conf, "targetNumSteps", target_nsteps, 0);
    if (!target_nsteps)
      cvm::error("Error: targetNumSteps must be non-zero.\n");

    if (get_keyval(conf, "targetNumStages", target_nstages, target_nstages) &&
        lambda_schedule.size()) {
      cvm::error("Error: targetNumStages and lambdaSchedule are incompatible.\n");
    }

    if (target_nstages) {
      // This means that either numStages of lambdaSchedule has been provided
      stage = 0;
      restraint_FE = 0.0;
    }

    if (get_keyval(conf, "targetForceExponent", force_k_exp, 1.0)) {
      if (! b_chg_force_k)
        cvm::log("Warning: not changing force constant: targetForceExponent will be ignored\n");
      if (force_k_exp < 1.0)
        cvm::log("Warning: for all practical purposes, targetForceExponent should be 1.0 or greater.\n");
    }
  }

  get_keyval(conf, "outputCenters", b_output_centers, false);
  if (b_chg_centers) {
    get_keyval(conf, "outputAccumulatedWork", b_output_acc_work, false);
  } else {
    b_output_acc_work = false;
  }
  acc_work = 0.0;

  if (cvm::debug())
    cvm::log("Done initializing a new restraint bias.\n");

  return COLVARS_OK;
}


colvarbias_restraint::~colvarbias_restraint()
{
  if (cvm::n_rest_biases > 0)
    cvm::n_rest_biases -= 1;
}


void colvarbias_restraint::change_configuration(std::string const &conf)
{
  get_keyval(conf, "forceConstant", force_k, force_k);
  if (get_keyval(conf, "centers", colvar_centers, colvar_centers)) {
    for (size_t i = 0; i < colvars.size(); i++) {
      colvar_centers[i].type(colvars[i]->value());
      colvar_centers[i].apply_constraints();
      colvar_centers_raw[i].type(colvars[i]->value());
      colvar_centers_raw[i] = colvar_centers[i];
    }
  }
}


cvm::real colvarbias_restraint::energy_difference(std::string const &conf)
{
  std::vector<colvarvalue> alt_colvar_centers;
  cvm::real alt_force_k;
  cvm::real alt_bias_energy = 0.0;

  get_keyval(conf, "forceConstant", alt_force_k, force_k);

  alt_colvar_centers.resize(colvars.size());
  size_t i;
  for (i = 0; i < colvars.size(); i++) {
    alt_colvar_centers[i].type(colvars[i]->value());
  }
  if (get_keyval(conf, "centers", alt_colvar_centers, colvar_centers)) {
    for (i = 0; i < colvars.size(); i++) {
      alt_colvar_centers[i].apply_constraints();
    }
  }

  for (i = 0; i < colvars.size(); i++) {
    alt_bias_energy += restraint_potential(restraint_convert_k(alt_force_k, colvars[i]->width),
					   colvars[i],
					   alt_colvar_centers[i]);
  }

  return alt_bias_energy - bias_energy;
}


int colvarbias_restraint::update()
{
  bias_energy = 0.0;

  if (cvm::debug())
    cvm::log("Updating the restraint bias \""+this->name+"\".\n");

  // Setup first stage of staged variable force constant calculation
  if (b_chg_force_k && target_nstages && cvm::step_absolute() == 0) {
    cvm::real lambda;
    if (lambda_schedule.size()) {
      lambda = lambda_schedule[0];
    } else {
      lambda = 0.0;
    }
    force_k = starting_force_k + (target_force_k - starting_force_k)
              * std::pow(lambda, force_k_exp);
    cvm::log("Restraint " + this->name + ", stage " +
        cvm::to_str(stage) + " : lambda = " + cvm::to_str(lambda));
    cvm::log("Setting force constant to " + cvm::to_str(force_k));
  }

  if (b_chg_centers) {

    if (!centers_incr.size()) {
      // if this is the first calculation, calculate the advancement
      // at each simulation step (or stage, if applicable)
      // (take current stage into account: it can be non-zero
      //  if we are restarting a staged calculation)
      centers_incr.resize(colvars.size());
      for (size_t i = 0; i < colvars.size(); i++) {
        centers_incr[i].type(colvars[i]->value());
        centers_incr[i] = (target_centers[i] - colvar_centers_raw[i]) /
          cvm::real( target_nstages ? (target_nstages - stage) :
                                      (target_nsteps - cvm::step_absolute()));
      }
      if (cvm::debug()) {
        cvm::log("Center increment for the restraint bias \""+
                  this->name+"\": "+cvm::to_str(centers_incr)+" at stage "+cvm::to_str(stage)+ ".\n");
      }
    }

    if (target_nstages) {
      if ((cvm::step_relative() > 0)
            && (cvm::step_absolute() % target_nsteps) == 0
            && stage < target_nstages) {

          for (size_t i = 0; i < colvars.size(); i++) {
            colvar_centers_raw[i] += centers_incr[i];
            colvar_centers[i] = colvar_centers_raw[i];
            colvars[i]->wrap(colvar_centers[i]);
            colvar_centers[i].apply_constraints();
          }
          stage++;
          cvm::log("Moving restraint \"" + this->name +
              "\" stage " + cvm::to_str(stage) +
              " : setting centers to " + cvm::to_str(colvar_centers) +
              " at step " +  cvm::to_str(cvm::step_absolute()));
      }
    } else if ((cvm::step_relative() > 0) && (cvm::step_absolute() <= target_nsteps)) {
      // move the restraint centers in the direction of the targets
      // (slow growth)
      for (size_t i = 0; i < colvars.size(); i++) {
        colvar_centers_raw[i] += centers_incr[i];
        colvar_centers[i] = colvar_centers_raw[i];
        colvars[i]->wrap(colvar_centers[i]);
        colvar_centers[i].apply_constraints();
      }
    }

    if (cvm::debug())
      cvm::log("Current centers for the restraint bias \""+
                this->name+"\": "+cvm::to_str(colvar_centers)+".\n");
  }

  if (b_chg_force_k) {
    // Coupling parameter, between 0 and 1
    cvm::real lambda;

    if (target_nstages) {
      // TI calculation: estimate free energy derivative
      // need current lambda
      if (lambda_schedule.size()) {
        lambda = lambda_schedule[stage];
      } else {
        lambda = cvm::real(stage) / cvm::real(target_nstages);
      }

      if (target_equil_steps == 0 || cvm::step_absolute() % target_nsteps >= target_equil_steps) {
        // Start averaging after equilibration period, if requested

        // Square distance normalized by square colvar width
        cvm::real dist_sq = 0.0;
        for (size_t i = 0; i < colvars.size(); i++) {
          dist_sq += colvars[i]->dist2(colvars[i]->value(), colvar_centers[i])
            / (colvars[i]->width * colvars[i]->width);
        }

        restraint_FE += 0.5 * force_k_exp * std::pow(lambda, force_k_exp - 1.0)
          * (target_force_k - starting_force_k) * dist_sq;
      }

      // Finish current stage...
      if (cvm::step_absolute() % target_nsteps == 0 &&
          cvm::step_absolute() > 0) {

          cvm::log("Lambda= " + cvm::to_str(lambda) + " dA/dLambda= "
              + cvm::to_str(restraint_FE / cvm::real(target_nsteps - target_equil_steps)));

        //  ...and move on to the next one
        if (stage < target_nstages) {

          restraint_FE = 0.0;
          stage++;
          if (lambda_schedule.size()) {
            lambda = lambda_schedule[stage];
          } else {
            lambda = cvm::real(stage) / cvm::real(target_nstages);
          }
          force_k = starting_force_k + (target_force_k - starting_force_k)
                    * std::pow(lambda, force_k_exp);
          cvm::log("Restraint " + this->name + ", stage " +
              cvm::to_str(stage) + " : lambda = " + cvm::to_str(lambda));
          cvm::log("Setting force constant to " + cvm::to_str(force_k));
        }
      }
    } else if (cvm::step_absolute() <= target_nsteps) {
      // update force constant (slow growth)
      lambda = cvm::real(cvm::step_absolute()) / cvm::real(target_nsteps);
      force_k = starting_force_k + (target_force_k - starting_force_k)
          * std::pow(lambda, force_k_exp);
    }
  }

  if (cvm::debug())
    cvm::log("Done updating the restraint bias \""+this->name+"\".\n");

  // Force and energy calculation
  for (size_t i = 0; i < colvars.size(); i++) {
    colvar_forces[i].type(colvars[i]->value());
    colvar_forces[i] = -1.0 * restraint_force(restraint_convert_k(force_k, colvars[i]->width),
                                              colvars[i],
                                              colvar_centers[i]);
    bias_energy += restraint_potential(restraint_convert_k(force_k, colvars[i]->width),
				       colvars[i],
				       colvar_centers[i]);
    if (cvm::debug()) {
      cvm::log("dist_grad["+cvm::to_str(i)+
                "] = "+cvm::to_str(colvars[i]->dist2_lgrad(colvars[i]->value(),
                                                             colvar_centers[i]))+"\n");
    }
  }

  if (b_output_acc_work) {
    if ((cvm::step_relative() > 0) || (cvm::step_absolute() == 0)) {
      for (size_t i = 0; i < colvars.size(); i++) {
        // project forces on the calculated increments at this step
        acc_work += colvar_forces[i] * centers_incr[i];
      }
    }
  }

  if (cvm::debug())
    cvm::log("Current forces for the restraint bias \""+
              this->name+"\": "+cvm::to_str(colvar_forces)+".\n");

  return COLVARS_OK;
}


std::istream & colvarbias_restraint::read_restart(std::istream &is)
{
  size_t const start_pos = is.tellg();

  cvm::log("Restarting restraint bias \""+
            this->name+"\".\n");

  std::string key, brace, conf;
  if ( !(is >> key)   || !(key == "restraint" || key == "harmonic") ||
       !(is >> brace) || !(brace == "{") ||
       !(is >> colvarparse::read_block("configuration", conf)) ) {

    cvm::log("Error: in reading restart configuration for restraint bias \""+
              this->name+"\" at position "+
              cvm::to_str(is.tellg())+" in stream.\n");
    is.clear();
    is.seekg(start_pos, std::ios::beg);
    is.setstate(std::ios::failbit);
    return is;
  }

//   int id = -1;
  std::string name = "";
//   if ( ( (colvarparse::get_keyval (conf, "id", id, -1, colvarparse::parse_silent)) &&
//          (id != this->id) ) ||
  if ( (colvarparse::get_keyval(conf, "name", name, std::string(""), colvarparse::parse_silent)) &&
       (name != this->name) )
    cvm::error("Error: in the restart file, the "
                      "\"restraint\" block has a wrong name\n");
//   if ( (id == -1) && (name == "") ) {
  if (name.size() == 0) {
    cvm::error("Error: \"restraint\" block in the restart file "
                      "has no identifiers.\n");
  }

  if (b_chg_centers) {
//    cvm::log ("Reading the updated restraint centers from the restart.\n");
    if (!get_keyval(conf, "centers", colvar_centers))
      cvm::error("Error: restraint centers are missing from the restart.\n");
    if (!get_keyval(conf, "centers_raw", colvar_centers_raw))
      cvm::error("Error: \"raw\" restraint centers are missing from the restart.\n");
  }

  if (b_chg_force_k) {
//    cvm::log ("Reading the updated force constant from the restart.\n");
    if (!get_keyval(conf, "forceConstant", force_k))
      cvm::error("Error: force constant is missing from the restart.\n");
  }

  if (target_nstages) {
//    cvm::log ("Reading current stage from the restart.\n");
    if (!get_keyval(conf, "stage", stage))
      cvm::error("Error: current stage is missing from the restart.\n");
  }

  if (b_output_acc_work) {
    if (!get_keyval(conf, "accumulatedWork", acc_work))
      cvm::error("Error: accumulatedWork is missing from the restart.\n");
  }

  is >> brace;
  if (brace != "}") {
    cvm::error("Error: corrupt restart information for restraint bias \""+
                      this->name+"\": no matching brace at position "+
                      cvm::to_str(is.tellg())+" in the restart file.\n");
    is.setstate(std::ios::failbit);
  }
  return is;
}


std::ostream & colvarbias_restraint::write_restart(std::ostream &os)
{
  os << "restraint {\n"
     << "  configuration {\n"
    //      << "    id " << this->id << "\n"
     << "    name " << this->name << "\n";

  if (b_chg_centers) {
    size_t i;
    os << "    centers ";
    for (i = 0; i < colvars.size(); i++) {
      os << " " << colvar_centers[i];
    }
    os << "\n";
    os << "    centers_raw ";
    for (i = 0; i < colvars.size(); i++) {
      os << " " << colvar_centers_raw[i];
    }
    os << "\n";
  }

  if (b_chg_force_k) {
    os << "    forceConstant "
       << std::setprecision(cvm::en_prec)
       << std::setw(cvm::en_width) << force_k << "\n";
  }

  if (target_nstages) {
    os << "    stage " << std::setw(cvm::it_width)
       << stage << "\n";
  }

  if (b_output_acc_work) {
    os << "    accumulatedWork " << acc_work << "\n";
  }

  os << "  }\n"
     << "}\n\n";

  return os;
}


std::ostream & colvarbias_restraint::write_traj_label(std::ostream &os)
{
  os << " ";

  if (b_output_energy)
    os << " E_"
       << cvm::wrap_string(this->name, cvm::en_width-2);

  if (b_output_centers)
    for (size_t i = 0; i < colvars.size(); i++) {
      size_t const this_cv_width = (colvars[i]->value()).output_width(cvm::cv_width);
      os << " x0_"
         << cvm::wrap_string(colvars[i]->name, this_cv_width-3);
    }

  if (b_output_acc_work)
    os << " W_"
       << cvm::wrap_string(this->name, cvm::en_width-2);

  return os;
}


std::ostream & colvarbias_restraint::write_traj(std::ostream &os)
{
  os << " ";

  if (b_output_energy)
    os << " "
       << std::setprecision(cvm::en_prec) << std::setw(cvm::en_width)
       << bias_energy;

  if (b_output_centers)
    for (size_t i = 0; i < colvars.size(); i++) {
      os << " "
         << std::setprecision(cvm::cv_prec) << std::setw(cvm::cv_width)
         << colvar_centers[i];
    }

  if (b_output_acc_work)
    os << " "
       << std::setprecision(cvm::en_prec) << std::setw(cvm::en_width)
       << acc_work;

  return os;
}


colvarbias_restraint_harmonic::colvarbias_restraint_harmonic(char const *key)
  : colvarbias_restraint(key)
{
}


int colvarbias_restraint_harmonic::init(std::string const &conf)
{
  colvarbias_restraint::init(conf);

  for (size_t i = 0; i < colvars.size(); i++) {
    if (colvars[i]->width != 1.0)
      cvm::log("The force constant for colvar \""+colvars[i]->name+
               "\" will be rescaled to "+
               cvm::to_str(restraint_convert_k(force_k, colvars[i]->width))+
               " according to the specified width.\n");
  }
  return COLVARS_OK;
}


cvm::real colvarbias_restraint_harmonic::restraint_potential(cvm::real k,
                                                             colvar const *x,
                                                             colvarvalue const &xcenter) const
{
  return 0.5 * k * x->dist2(x->value(), xcenter);
}


colvarvalue colvarbias_restraint_harmonic::restraint_force(cvm::real k,
                                                           colvar const *x,
                                                           colvarvalue const &xcenter) const
{
  return 0.5 * k * x->dist2_lgrad(x->value(), xcenter);
}


cvm::real colvarbias_restraint_harmonic::restraint_convert_k(cvm::real k,
                                                             cvm::real dist_measure) const
{
  return k / (dist_measure * dist_measure);
}



colvarbias_restraint_linear::colvarbias_restraint_linear(char const *key)
  : colvarbias_restraint(key)
{
}


int colvarbias_restraint_linear::init(std::string const &conf)
{
  colvarbias_restraint::init(conf);

  for (size_t i = 0; i < colvars.size(); i++) {
    if (colvars[i]->width != 1.0)
      cvm::log("The force constant for colvar \""+colvars[i]->name+
               "\" will be rescaled to "+
               cvm::to_str(restraint_convert_k(force_k, colvars[i]->width))+
               " according to the specified width.\n");
  }
  return COLVARS_OK;
}


cvm::real colvarbias_restraint_linear::restraint_potential(cvm::real k,
                                                           colvar const *x,
                                                           colvarvalue const &xcenter) const
{
  return k * (x->value() - xcenter);
}


colvarvalue colvarbias_restraint_linear::restraint_force(cvm::real k,
                                                         colvar const *x,
                                                         colvarvalue const &xcenter) const
{
  return k;
}


cvm::real colvarbias_restraint_linear::restraint_convert_k(cvm::real k,
                                                           cvm::real dist_measure) const
{
  return k / dist_measure;
}



colvarbias_restraint_histogram::colvarbias_restraint_histogram(char const *key)
  : colvarbias(key)
{
  lower_boundary = 0.0;
  upper_boundary = 0.0;
  width = 0.0;
  gaussian_width = 0.0;
}


int colvarbias_restraint_histogram::init(std::string const &conf)
{
  colvarbias::init(conf);

  get_keyval(conf, "lowerBoundary", lower_boundary, lower_boundary);
  get_keyval(conf, "upperBoundary", upper_boundary, upper_boundary);
  get_keyval(conf, "width", width, width);

  if (width <= 0.0) {
    cvm::error("Error: \"width\" must be positive.\n", INPUT_ERROR);
  }

  get_keyval(conf, "gaussianWidth", gaussian_width, 2.0 * width, colvarparse::parse_silent);
  get_keyval(conf, "gaussianSigma", gaussian_width, 2.0 * width);

  if (lower_boundary >= upper_boundary) {
    cvm::error("Error: the upper boundary, "+
               cvm::to_str(upper_boundary)+
               ", is not higher than the lower boundary, "+
               cvm::to_str(lower_boundary)+".\n",
               INPUT_ERROR);
  }

  cvm::real const nbins = (upper_boundary - lower_boundary) / width;
  int const nbins_round = (int)(nbins);

  if (std::fabs(nbins - cvm::real(nbins_round)) > 1.0E-10) {
    cvm::log("Warning: grid interval ("+
             cvm::to_str(lower_boundary, cvm::cv_width, cvm::cv_prec)+" - "+
             cvm::to_str(upper_boundary, cvm::cv_width, cvm::cv_prec)+
             ") is not commensurate to its bin width ("+
             cvm::to_str(width, cvm::cv_width, cvm::cv_prec)+").\n");
  }

  p.resize(nbins_round);
  ref_p.resize(nbins_round);
  p_diff.resize(nbins_round);

  bool const inline_ref_p =
    get_keyval(conf, "refHistogram", ref_p.data_array(), ref_p.data_array());
  std::string ref_p_file;
  get_keyval(conf, "refHistogramFile", ref_p_file, std::string(""));
  if (ref_p_file.size()) {
    if (inline_ref_p) {
      cvm::error("Error: cannot specify both refHistogram and refHistogramFile at the same time.\n",
                 INPUT_ERROR);
    } else {
      std::ifstream is(ref_p_file.c_str());
      std::string data_s = "";
      std::string line;
      while (getline_nocomments(is, line)) {
        data_s.append(line+"\n");
      }
      if (data_s.size() == 0) {
        cvm::error("Error: file \""+ref_p_file+"\" empty or unreadable.\n", FILE_ERROR);
      }
      is.close();
      cvm::vector1d<cvm::real> data;
      if (data.from_simple_string(data_s) != 0) {
        cvm::error("Error: could not read histogram from file \""+ref_p_file+"\".\n");
      }
      if (data.size() == 2*ref_p.size()) {
        // file contains both x and p(x)
        size_t i;
        for (i = 0; i < ref_p.size(); i++) {
          ref_p[i] = data[2*i+1];
        }
      } else if (data.size() == ref_p.size()) {
        ref_p = data;
      } else {
        cvm::error("Error: file \""+ref_p_file+"\" contains a histogram of different length.\n",
                   INPUT_ERROR);
      }
    }
  }
  cvm::real const ref_integral = ref_p.sum() * width;
  if (std::fabs(ref_integral - 1.0) > 1.0e-03) {
    cvm::log("Reference distribution not normalized, normalizing to unity.\n");
    ref_p /= ref_integral;
  }

  get_keyval(conf, "writeHistogram", b_write_histogram, false);
  get_keyval(conf, "forceConstant", force_k, 1.0);

  return COLVARS_OK;
}


colvarbias_restraint_histogram::~colvarbias_restraint_histogram()
{
  p.resize(0);
  ref_p.resize(0);
  p_diff.resize(0);
}


int colvarbias_restraint_histogram::update()
{
  if (cvm::debug())
    cvm::log("Updating the histogram restraint bias \""+this->name+"\".\n");

  size_t vector_size = 0;
  size_t icv;
  for (icv = 0; icv < colvars.size(); icv++) {
    vector_size += colvars[icv]->value().size();
  }

  cvm::real const norm = 1.0/(std::sqrt(2.0*PI)*gaussian_width*vector_size);

  // calculate the histogram
  p.reset();
  for (icv = 0; icv < colvars.size(); icv++) {
    colvarvalue const &cv = colvars[icv]->value();
    if (cv.type() == colvarvalue::type_scalar) {
      cvm::real const cv_value = cv.real_value;
      size_t igrid;
      for (igrid = 0; igrid < p.size(); igrid++) {
        cvm::real const x_grid = (lower_boundary + (igrid+0.5)*width);
        p[igrid] += norm * std::exp(-1.0 * (x_grid - cv_value) * (x_grid - cv_value) /
                                    (2.0 * gaussian_width * gaussian_width));
      }
    } else if (cv.type() == colvarvalue::type_vector) {
      size_t idim;
      for (idim = 0; idim < cv.vector1d_value.size(); idim++) {
        cvm::real const cv_value = cv.vector1d_value[idim];
        size_t igrid;
        for (igrid = 0; igrid < p.size(); igrid++) {
          cvm::real const x_grid = (lower_boundary + (igrid+0.5)*width);
          p[igrid] += norm * std::exp(-1.0 * (x_grid - cv_value) * (x_grid - cv_value) /
                                      (2.0 * gaussian_width * gaussian_width));
        }
      }
    } else {
      // TODO
    }
  }

  cvm::real const force_k_cv = force_k * vector_size;

  // calculate the difference between current and reference
  p_diff = p - ref_p;
  bias_energy = 0.5 * force_k_cv * p_diff * p_diff;

  // calculate the forces
  for (icv = 0; icv < colvars.size(); icv++) {
    colvarvalue const &cv = colvars[icv]->value();
    colvarvalue &cv_force = colvar_forces[icv];
    cv_force.type(cv);
    cv_force.reset();

    if (cv.type() == colvarvalue::type_scalar) {
      cvm::real const cv_value = cv.real_value;
      cvm::real &force = cv_force.real_value;
      size_t igrid;
      for (igrid = 0; igrid < p.size(); igrid++) {
        cvm::real const x_grid = (lower_boundary + (igrid+0.5)*width);
        force += force_k_cv * p_diff[igrid] *
          norm * std::exp(-1.0 * (x_grid - cv_value) * (x_grid - cv_value) /
                          (2.0 * gaussian_width * gaussian_width)) *
          (-1.0 * (x_grid - cv_value) / (gaussian_width * gaussian_width));
      }
    } else if (cv.type() == colvarvalue::type_vector) {
      size_t idim;
      for (idim = 0; idim < cv.vector1d_value.size(); idim++) {
        cvm::real const cv_value = cv.vector1d_value[idim];
        cvm::real &force = cv_force.vector1d_value[idim];
        size_t igrid;
        for (igrid = 0; igrid < p.size(); igrid++) {
          cvm::real const x_grid = (lower_boundary + (igrid+0.5)*width);
          force += force_k_cv * p_diff[igrid] *
            norm * std::exp(-1.0 * (x_grid - cv_value) * (x_grid - cv_value) /
                            (2.0 * gaussian_width * gaussian_width)) *
            (-1.0 * (x_grid - cv_value) / (gaussian_width * gaussian_width));
        }
      }
    } else {
      // TODO
    }
  }

  return COLVARS_OK;
}


std::ostream & colvarbias_restraint_histogram::write_restart(std::ostream &os)
{
  if (b_write_histogram) {
    std::string file_name(cvm::output_prefix+"."+this->name+".hist.dat");
    std::ofstream os(file_name.c_str());
    os << "# " << cvm::wrap_string(colvars[0]->name, cvm::cv_width)
       << "  " << "p(" << cvm::wrap_string(colvars[0]->name, cvm::cv_width-3)
       << ")\n";
    size_t igrid;
    for (igrid = 0; igrid < p.size(); igrid++) {
      cvm::real const x_grid = (lower_boundary + (igrid+1)*width);
      os << "  "
         << std::setprecision(cvm::cv_prec)
         << std::setw(cvm::cv_width)
         << x_grid
         << "  "
         << std::setprecision(cvm::cv_prec)
         << std::setw(cvm::cv_width)
         << p[igrid] << "\n";
    }
    os.close();
  }
  return os;
}


std::istream & colvarbias_restraint_histogram::read_restart(std::istream &is)
{
  return is;
}


std::ostream & colvarbias_restraint_histogram::write_traj_label(std::ostream &os)
{
  os << " ";
  if (b_output_energy) {
    os << " E_"
       << cvm::wrap_string(this->name, cvm::en_width-2);
  }
  return os;
}


std::ostream & colvarbias_restraint_histogram::write_traj(std::ostream &os)
{
  os << " ";
  if (b_output_energy) {
    os << " "
       << std::setprecision(cvm::en_prec) << std::setw(cvm::en_width)
       << bias_energy;
  }
  return os;
}
