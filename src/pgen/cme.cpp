//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cme.cpp
//! \brief Problem generator for spherical cme wave problem.  Works in Cartesian,
//!        cylindrical, and spherical coordinates.  Contains post-processing code
//!        to check whether cme is spherical for regression tests
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

// User-defined boundary conditions
// radial in cylindrical coords
void CMEInnerX1(MeshBlock *pmb, Coordinates *pco,
                 AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
// azimuth in cylindrical coords
void CMEInnerX2(MeshBlock *pmb, Coordinates *pco,
                 AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real calc_v_inner(ParameterInput *pin);
Real calc_b1_inner(ParameterInput *pin);
Real calc_b2_inner(ParameterInput *pin, Real b1, Real v_inner);
Real calc_n_inner(ParameterInput *pin, Real v_inner);
Real calc_energy_inner(ParameterInput *pin, Real n_inner, Real gamma);

int RefinementCondition(MeshBlock *pmb);

// namespaced variables, i.e. globals
// should be turned into a class with setters and getters
namespace {
Real threshold;
Real v_inner, n_inner, e_inner;
Real inner_radius, gamma_param;
} // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
    threshold = pin->GetReal("problem","thr");
  }
  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, CMEInnerX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, CMEInnerX2);
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical cme wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  gamma_param = peos->GetGamma();
  // would need to change this if not cylindrical
  inner_radius = pin->GetReal("mesh", "x1min");
  v_inner = calc_v_inner(pin);
  Real b1 = calc_b1_inner(pin);
  Real b2 = calc_b2_inner(pin, b1, v_inner);
  Real b3 = 0.0;
  n_inner = calc_n_inner(pin, v_inner);
  e_inner = calc_energy_inner(pin, n_inner, gamma_param);
  // Real rout = pin->GetReal("problem", "radius");
  // Real rin  = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
  // Real pa   = pin->GetOrAddReal("problem", "pamb", 1.0);
  // Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  // Real prat = pin->GetReal("problem", "prat");
  // Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real b0, angle;
  // if (MAGNETIC_FIELDS_ENABLED) {
  //   b0 = pin->GetReal("problem", "b0");
  //   angle = (PI/180.0)*pin->GetReal("problem", "angle");
  // }
  
  // Real gm1 = gamma_param - 1.0;

  // magnetic field
  b0 = std::sqrt( SQR(b1) + SQR(b2) + SQR(b3));
  angle = (PI/180.0)*b2; //(PI/180.0)*0.0; //pin->GetReal("problem", "angle");
  std::cout << "b0: " << b0 << " angle: " << angle << "\n";
  // get coordinates of center of cme, and convert to Cartesian if necessary
  // should all be zero for now
  Real x1_0   = pin->GetOrAddReal("problem", "x1_0", 0.0);
  Real x2_0   = pin->GetOrAddReal("problem", "x2_0", 0.0);
  Real x3_0   = pin->GetOrAddReal("problem", "x3_0", 0.0);
  Real x0, y0, z0;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    x0 = x1_0;
    y0 = x2_0;
    z0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    x0 = x1_0*std::cos(x2_0);
    y0 = x1_0*std::sin(x2_0);
    z0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
    y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
    z0 = x1_0*std::cos(x2_0);
  } else {
    // Only check legality of COORDINATE_SYSTEM once in this function
    std::stringstream msg;
    msg << "### FATAL ERROR in cme.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM=" << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  // setup uniform ambient medium with spherical over-pressured region
  // Modifies density, and energy (non-barotropic eos and relativistic dynamics) 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad;
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
          Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        }

        // ambient density
        Real den = n_inner * SQR((inner_radius / rad));
        // modify density inside sphere
        //commenting out for now
        // if (rad < rout) {
        //   if (rad < rin) {
        //     den = drat*da;
        //   } else {   // add smooth ramp in density
        //     Real f = (rad-rin) / (rout-rin);
        //     Real log_den = (1.0-f) * std::log(drat*da) + f * std::log(da);
        //     den = std::exp(log_den);
        //   }
        // }

        phydro->u(IDN,k,j,i) = den;
        // should this be a constant momentum term 
        // or a function of the inner density
        phydro->u(IM1,k,j,i) = v_inner * n_inner;
        // phydro->u(IM1,k,j,i) = v_inner * den;
        phydro->u(IM2,k,j,i) = 0.0; // no tangential velocity at beginning
        phydro->u(IM3,k,j,i) = 0.0;

        
        if (NON_BAROTROPIC_EOS) {
          // Real pres = pa;
          // modify energy, commenting out for now
          // if (rad < rout) {
          //   if (rad < rin) {
          //     pres = prat*pa;
          //   } else {  // add smooth ramp in pressure
          //     Real f = (rad-rin) / (rout-rin);
          //     Real log_pres = (1.0-f) * std::log(prat*pa) + f * std::log(pa);
          //     pres = std::exp(log_pres);
          //   }
          // }
          // phydro->u(IEN,k,j,i) = pres/gm1;
          phydro->u(IEN,k,j,i) = e_inner * std::pow((inner_radius / rad), 2.0*gamma_param);
          

          if (RELATIVISTIC_DYNAMICS) {
            std::stringstream msg;
            msg << "### FATAL ERROR in cme.cpp ProblemGenerator" << std::endl
            << "can't handle RELATIVISTIC_DYNAMICS=" << RELATIVISTIC_DYNAMICS << std::endl;
            ATHENA_ERROR(msg);
          }
          // if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
          //   phydro->u(IEN,k,j,i) += den;
        }
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          Real rad;
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            Real x = pcoord->x1v(i);
            Real y = pcoord->x2v(j);
            Real z = pcoord->x3v(k);
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
            pfield->b.x1f(k,j,i) = b0 * SQR((inner_radius/rad)) * std::cos(angle);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
            Real z = pcoord->x3v(k);
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
            Real phi = pcoord->x2v(j);
            pfield->b.x1f(k,j,i) =
                b0 * SQR((inner_radius/rad)) * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
            Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
            Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x1f(k,j,i) = b0 * SQR((inner_radius/rad)) * std::abs(std::sin(theta))
                                   * (std::cos(angle) * std::cos(phi)
                                      + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real rad;
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            Real x = pcoord->x1v(i);
            Real y = pcoord->x2v(j);
            Real z = pcoord->x3v(k);
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
            pfield->b.x2f(k,j,i) = b0 * SQR((inner_radius/rad)) * std::sin(angle);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
            Real z = pcoord->x3v(k);
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
            Real phi = pcoord->x2v(j);
            pfield->b.x2f(k,j,i) =
                b0 * SQR((inner_radius/rad)) * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
            Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
            Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x2f(k,j,i) = b0 * SQR((inner_radius/rad)) * std::cos(theta)
                                   * (std::cos(angle) * std::cos(phi)
                                      + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              pfield->b.x2f(k,j,i) *= -1.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real rad;
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            pfield->b.x3f(k,j,i) = 0.0;
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            pfield->b.x3f(k,j,i) = 0.0;
          } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
            Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
            Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
            Real phi = pcoord->x3v(k);
            pfield->b.x3f(k,j,i) =
                b0 * SQR((inner_radius/rad)) * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real rad;
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            Real x = pcoord->x1v(i);
            Real y = pcoord->x2v(j);
            Real z = pcoord->x3v(k);
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
            Real z = pcoord->x3v(k);
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
            Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
            Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          }
          phydro->u(IEN,k,j,i) += 0.5*b0*b0*std::pow((inner_radius/rad),4.0);
        }
      }
    }
  }
}

//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void CMEInnerX1(MeshBlock *pmb, Coordinates *pcoord,
                 AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0);
  // Real v_inner = calc_v_inner(pin);
  // Real n_inner = calc_n_inner(pin, v_inner);
  // Real x1_0   = pin->GetOrAddReal("problem", "x1_0", 0.0);
  // Real x2_0   = pin->GetOrAddReal("problem", "x2_0", 0.0);
  // Real x3_0   = pin->GetOrAddReal("problem", "x3_0", 0.0);
  Real x0(0.0), y0(0.0), z0(0.0);
  // if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
  //   x0 = x1_0;
  //   y0 = x2_0;
  //   z0 = x3_0;
  // } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
  //   x0 = x1_0*std::cos(x2_0);
  //   y0 = x1_0*std::sin(x2_0);
  //   z0 = x3_0;
  // } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
  //   x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
  //   y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
  //   z0 = x1_0*std::cos(x2_0);
  // } else {
  //   // Only check legality of COORDINATE_SYSTEM once in this function
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in cme.cpp ProblemGenerator" << std::endl
  //       << "Unrecognized COORDINATE_SYSTEM=" << COORDINATE_SYSTEM << std::endl;
  //   ATHENA_ERROR(msg);
  // }
  // OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          // update boundaries
          Real den = n_inner * SQR((inner_radius / rad));
          prim(IDN,k,j,il-i) = den;
          prim(IM1,k,j,il-i) = v_inner * n_inner;;
          prim(IM2,k,j,il-i) = 0.0;
          prim(IM3,k,j,il-i) = 0.0;
          if (NON_BAROTROPIC_EOS) {
            prim(IEN,k,j,il-i) = e_inner * std::pow((inner_radius / rad), 2.0*gamma_param);
          }
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          // update boundaries
          Real den = n_inner * SQR((inner_radius / rad));
          prim(IDN,k,j,il-i) = den;
          prim(IM1,k,j,il-i) = v_inner * n_inner;;
          prim(IM2,k,j,il-i) = 0.0;
          prim(IM3,k,j,il-i) = 0.0;
          if (NON_BAROTROPIC_EOS) {
            prim(IEN,k,j,il-i) = e_inner * std::pow((inner_radius / rad), 2.0*gamma_param);
          }
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          // update boundaries
          Real den = n_inner * SQR((inner_radius / rad));
          prim(IDN,k,j,il-i) = den;
          prim(IM1,k,j,il-i) = v_inner * n_inner;;
          prim(IM2,k,j,il-i) = 0.0;
          prim(IM3,k,j,il-i) = 0.0;
          if (NON_BAROTROPIC_EOS) {
            prim(IEN,k,j,il-i) = e_inner * std::pow((inner_radius / rad), 2.0*gamma_param);
          }
        }
      }
    }
  }
}

//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void CMEInnerX2(MeshBlock *pmb, Coordinates *pco,
                 AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  // OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          // GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          // prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          // vel = VelProfileCyl(rad,phi,z);
          // if (pmb->porb->orbital_advection_defined)
          //   vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          // prim(IM1,k,jl-j,i) = 0.0;
          // prim(IM2,k,jl-j,i) = vel;
          // prim(IM3,k,jl-j,i) = 0.0;
          // if (NON_BAROTROPIC_EOS)
          //   prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          // GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          // prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          // vel = VelProfileCyl(rad,phi,z);
          // if (pmb->porb->orbital_advection_defined)
          //   vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          // prim(IM1,k,jl-j,i) = 0.0;
          // prim(IM2,k,jl-j,i) = 0.0;
          // prim(IM3,k,jl-j,i) = vel;
          // if (NON_BAROTROPIC_EOS)
          //   prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Check radius of sphere to make sure it is round
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("problem","compute_error",false)) return;
  MeshBlock *pmb = my_blocks(0);

  // analysis - check shape of the spherical cme wave
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> pr;
  pr.InitWithShallowSlice(pmb->phydro->w, 4, IPR, 1);

  // get coordinate location of the center, convert to Cartesian
  Real x1_0 = pin->GetOrAddReal("problem", "x1_0", 0.0);
  Real x2_0 = pin->GetOrAddReal("problem", "x2_0", 0.0);
  Real x3_0 = pin->GetOrAddReal("problem", "x3_0", 0.0);
  Real x0, y0, z0;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    x0 = x1_0;
    y0 = x2_0;
    z0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    x0 = x1_0*std::cos(x2_0);
    y0 = x1_0*std::sin(x2_0);
    z0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
    y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
    z0 = x1_0*std::cos(x2_0);
  } else {
    // Only check legality of COORDINATE_SYSTEM once in this function
    std::stringstream msg;
    msg << "### FATAL ERROR in cme.cpp ParameterInput" << std::endl
        << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  // find indices of the center
  int ic, jc, kc;
  for (ic=is; ic<=ie; ic++)
    if (pmb->pcoord->x1f(ic) > x1_0) break;
  ic--;
  for (jc=pmb->js; jc<=pmb->je; jc++)
    if (pmb->pcoord->x2f(jc) > x2_0) break;
  jc--;
  for (kc=pmb->ks; kc<=pmb->ke; kc++)
    if (pmb->pcoord->x3f(kc) > x3_0) break;
  kc--;

  // search pressure maximum in each direction
  Real rmax = 0.0, rmin = 100.0, rave = 0.0;
  int nr = 0;
  for (int o=0; o<=6; o++) {
    int ios = 0, jos = 0, kos = 0;
    if (o == 1) ios=-10;
    else if (o == 2) ios =  10;
    else if (o == 3) jos = -10;
    else if (o == 4) jos =  10;
    else if (o == 5) kos = -10;
    else if (o == 6) kos =  10;
    for (int d=0; d<6; d++) {
      Real pmax = 0.0;
      int imax(0), jmax(0), kmax(0);
      if (d == 0) {
        if (ios != 0) continue;
        jmax = jc+jos, kmax = kc+kos;
        for (int i=ic; i>=is; i--) {
          if (pr(kmax,jmax,i)>pmax) {
            pmax = pr(kmax,jmax,i);
            imax = i;
          }
        }
      } else if (d == 1) {
        if (ios != 0) continue;
        jmax = jc+jos, kmax = kc+kos;
        for (int i=ic; i<=ie; i++) {
          if (pr(kmax,jmax,i)>pmax) {
            pmax = pr(kmax,jmax,i);
            imax = i;
          }
        }
      } else if (d == 2) {
        if (jos != 0) continue;
        imax = ic+ios, kmax = kc+kos;
        for (int j=jc; j>=js; j--) {
          if (pr(kmax,j,imax)>pmax) {
            pmax = pr(kmax,j,imax);
            jmax = j;
          }
        }
      } else if (d == 3) {
        if (jos != 0) continue;
        imax = ic+ios, kmax = kc+kos;
        for (int j=jc; j<=je; j++) {
          if (pr(kmax,j,imax)>pmax) {
            pmax = pr(kmax,j,imax);
            jmax = j;
          }
        }
      } else if (d == 4) {
        if (kos != 0) continue;
        imax = ic+ios, jmax = jc+jos;
        for (int k=kc; k>=ks; k--) {
          if (pr(k,jmax,imax)>pmax) {
            pmax = pr(k,jmax,imax);
            kmax = k;
          }
        }
      } else { // if (d == 5) {
        if (kos != 0) continue;
        imax = ic+ios, jmax = jc+jos;
        for (int k=kc; k<=ke; k++) {
          if (pr(k,jmax,imax)>pmax) {
            pmax = pr(k,jmax,imax);
            kmax = k;
          }
        }
      }

      Real xm, ym, zm;
      Real x1m = pmb->pcoord->x1v(imax);
      Real x2m = pmb->pcoord->x2v(jmax);
      Real x3m = pmb->pcoord->x3v(kmax);
      if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
        xm = x1m;
        ym = x2m;
        zm = x3m;
      } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        xm = x1m*std::cos(x2m);
        ym = x1m*std::sin(x2m);
        zm = x3m;
      } else {  // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        xm = x1m*std::sin(x2m)*std::cos(x3m);
        ym = x1m*std::sin(x2m)*std::sin(x3m);
        zm = x1m*std::cos(x2m);
      }
      Real rad = std::sqrt(SQR(xm-x0)+SQR(ym-y0)+SQR(zm-z0));
      if (rad > rmax) rmax = rad;
      if (rad < rmin) rmin = rad;
      rave += rad;
      nr++;
    }
  }
  rave /= static_cast<Real>(nr);

  // use physical grid spacing at center of cme
  Real dr_max;
  Real  x1c = pmb->pcoord->x1v(ic);
  Real dx1c = pmb->pcoord->dx1f(ic);
  Real  x2c = pmb->pcoord->x2v(jc);
  Real dx2c = pmb->pcoord->dx2f(jc);
  Real dx3c = pmb->pcoord->dx3f(kc);
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    dr_max = std::max(std::max(dx1c, dx2c), dx3c);
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    dr_max = std::max(std::max(dx1c, x1c*dx2c), dx3c);
  } else { // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    dr_max = std::max(std::max(dx1c, x1c*dx2c), x1c*std::sin(x2c)*dx3c);
  }
  Real deform=(rmax-rmin)/dr_max;

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("cmewave-shape.dat");
    std::stringstream msg;
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(),"r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(),"a",pfile)) == nullptr) {
        msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]"
            << std::endl << "Blast shape output file could not be opened" <<std::endl;
        ATHENA_ERROR(msg);
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(),"w")) == nullptr) {
        msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]"
            << std::endl << "Blast shape output file could not be opened" <<std::endl;
        ATHENA_ERROR(msg);
      }
    }
    std::fprintf(pfile,"# Offset cme wave test in %s coordinates:\n",COORDINATE_SYSTEM);
    std::fprintf(pfile,"# Rmax       Rmin       Rave        Deformation\n");
    std::fprintf(pfile,"%e  %e  %e  %e \n",rmax,rmin,rave,deform);
    std::fclose(pfile);
  }
  return;
}


// refinement condition: check the maximum pressure gradient
int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real maxeps = 0.0;
  if (pmb->pmy_mesh->f3) {
    for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
      for (int j=pmb->js-1; j<=pmb->je+1; j++) {
        for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
          Real eps = std::sqrt(SQR(0.5*(w(IPR,k,j,i+1) - w(IPR,k,j,i-1)))
                               +SQR(0.5*(w(IPR,k,j+1,i) - w(IPR,k,j-1,i)))
                               +SQR(0.5*(w(IPR,k+1,j,i) - w(IPR,k-1,j,i))))/w(IPR,k,j,i);
          maxeps = std::max(maxeps, eps);
        }
      }
    }
  } else if (pmb->pmy_mesh->f2) {
    int k = pmb->ks;
    for (int j=pmb->js-1; j<=pmb->je+1; j++) {
      for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
        Real eps = std::sqrt(SQR(0.5*(w(IPR,k,j,i+1) - w(IPR,k,j,i-1)))
                             + SQR(0.5*(w(IPR,k,j+1,i) - w(IPR,k,j-1,i))))/w(IPR,k,j,i);
        maxeps = std::max(maxeps, eps);
      }
    }
  } else {
    return 0;
  }

  if (maxeps > threshold) return 1;
  if (maxeps < 0.25*threshold) return -1;
  return 0;
}

// convert velocity to inner boundary velocity
// empirical formulation
Real calc_v_inner(ParameterInput *pin) {
  Real v_measure = pin->GetReal("problem", "v_measure");
  Real v_measure_extra = pin->GetOrAddReal("problem", "v_measure_extra", 1.0);
  Real vo = pin->GetReal("problem", "vo");
  Real v_in = (std::sqrt(v_measure / 430.7) * 0.8231
                  * v_measure_extra * v_measure * 1000.0
                 )
                  / vo;
  return v_in;
}

// convert b1 magnetic field to inner boundary 
Real calc_b1_inner(ParameterInput *pin) {
  Real min_radius = pin->GetReal("mesh", "x1min");
  Real r_meas = pin->GetReal("problem", "r_measure");
  Real v_measure = pin->GetReal("problem", "v_measure");
  Real b_measure = pin->GetReal("problem", "b_measure");
  Real omega = pin->GetReal("problem", "omega_sun");
  Real AU = pin->GetReal("problem", "AU");
  Real bo = pin->GetReal("problem", "bo");

  Real B_AU = b_measure * 1.0E-9 / bo;
  Real ratio =  (omega * AU) / (v_measure * 1000.0);
  Real B1_AU = B_AU / (std::sqrt(1.0 + SQR(ratio)));
  Real b1 = B1_AU * SQR(r_meas / min_radius);

  return b1;
}

// convert b2 magnetic field to inner boundary 
Real calc_b2_inner(ParameterInput *pin, Real b1, Real v_inner) {
  Real min_radius = pin->GetReal("mesh", "x1min");
  Real omega = pin->GetReal("problem", "omega_sun");
  Real AU = pin->GetReal("problem", "AU");
  Real vo = pin->GetReal("problem", "vo");
  Real b_extra = pin->GetOrAddReal("problem", "b_measure_extra", 1.0);

  Real ratio_inner = (omega * AU * min_radius) / ( v_inner * vo);
  Real b2 = -1.0* b1 * ratio_inner * b_extra;
  return b2;
}

// convert density to inner boundary 
Real calc_n_inner(ParameterInput *pin, Real v_inner) {
  Real min_radius = pin->GetReal("mesh", "x1min");
  Real r_meas = pin->GetReal("problem", "r_measure");
  Real v_measure = pin->GetReal("problem", "v_measure");
  Real n_measure = pin->GetReal("problem", "n_measure");
  Real vo = pin->GetReal("problem", "vo");

  Real n_in = (n_measure * SQR((r_meas / min_radius)) * v_measure * 1000.0) / (v_inner * vo);
  return n_in;
}

// convert density and temperature to energy density 
Real calc_energy_inner(ParameterInput *pin, Real n_inner, Real gamma) {
  Real min_radius = pin->GetReal("mesh", "x1min");
  Real r_meas = pin->GetReal("problem", "r_measure");
  Real T_measure = pin->GetReal("problem", "T_measure");
  // Real gamma = peos->GetGamma();

  // convert temperature to inner boundary temperature
  Real k_inner = T_measure * pow((r_meas/min_radius), (4.0/3.0));
  // convert density and temperature to energy density 
  Real es = (3.0 * n_inner * k_inner) / (gamma - 1.0);
  return es;
}