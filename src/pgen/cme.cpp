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

void SunGravity(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

Real calc_v1_inner(ParameterInput *pin);
Real calc_GM_sun(ParameterInput *pin);
Real calc_b1_inner(ParameterInput *pin);
Real calc_b2_inner(ParameterInput *pin, Real b1, Real v1_inner);
Real calc_n_inner(ParameterInput *pin, Real v1_inner);
Real calc_energy_inner(ParameterInput *pin, Real n_inner, Real gamma);

int RefinementCondition(MeshBlock *pmb);

// namespaced variables, i.e. globals
// should be turned into a class with setters and getters
namespace {
Real threshold;
Real v1_inner, v2_inner, v3_inner;
Real n_inner, e_inner;
Real inner_radius, gamma_param;
Real m_p;
Real b1, b2, b3;
Real x_0, y_0, z_0;
Real GM_norm;
bool sun_gravity;
} // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // if (adaptive) {
  //   EnrollUserRefinementCondition(RefinementCondition);
  //   threshold = pin->GetReal("problem","thr");
  // }
  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, CMEInnerX1);
  }

  // Enroll user-defined physical source terms
  // external gravitational potential
  sun_gravity = pin->GetOrAddBoolean("problem", "enable_gravity",false);
  if (sun_gravity) {
    EnrollUserExplicitSourceFunction(SunGravity);
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
  m_p = pin->GetOrAddReal("problem", "m_p", 1.67E-27);
  Real omega = pin->GetOrAddReal("problem", "omega_sun", 2.87e-6);
  v1_inner = calc_v1_inner(pin);
  v2_inner = 0.0; //omega; // create calc for v2 and v3
  v3_inner = 0.0;
  b1 = calc_b1_inner(pin);
  b2 = calc_b2_inner(pin, b1, v1_inner);
  b3 = 0.0;
  n_inner = calc_n_inner(pin, v1_inner);
  e_inner = calc_energy_inner(pin, n_inner, gamma_param);
  // Real rout = pin->GetReal("problem", "radius");
  // Real rin  = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
  // Real pa   = pin->GetOrAddReal("problem", "pamb", 1.0);
  // Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  // Real prat = pin->GetReal("problem", "prat");
  // Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real b0; //, angle;
  // if (MAGNETIC_FIELDS_ENABLED) {
  //   b0 = pin->GetReal("problem", "b0");
  //   angle = (PI/180.0)*pin->GetReal("problem", "angle");
  // }
  
  // Real gm1 = gamma_param - 1.0;

  // magnetic field
  // b0 = std::sqrt( SQR(b1) + SQR(b2) + SQR(b3));
  // angle = (PI/180.0)*b2; //(PI/180.0)*0.0; //pin->GetReal("problem", "angle");
  // get coordinates of center of sun, and convert to Cartesian if necessary
  // should all be zero for now
  Real x1_0   = pin->GetOrAddReal("problem", "x1_0", 0.0);
  Real x2_0   = pin->GetOrAddReal("problem", "x2_0", 0.0);
  Real x3_0   = pin->GetOrAddReal("problem", "x3_0", 0.0);
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    x_0 = x1_0;
    y_0 = x2_0;
    z_0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    x_0 = x1_0*std::cos(x2_0);
    y_0 = x1_0*std::sin(x2_0);
    z_0 = x3_0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    x_0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
    y_0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
    z_0 = x1_0*std::cos(x2_0);
  } else {
    // Only check legality of COORDINATE_SYSTEM once in this function
    std::stringstream msg;
    msg << "### FATAL ERROR in cme.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM=" << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  // calculate normalized gravity potential
  if (sun_gravity) {
      GM_norm = calc_GM_sun(pin);
      std::cout << GM_norm << " yo" << std::endl;
  }

  // setup uniform ambient medium
  // Modifies density, and energy (non-barotropic eos and relativistic dynamics)
  Real rad, den, energy, x, y, z;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          z = pcoord->x3v(k);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          z = pcoord->x3v(k);
        } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          z = pcoord->x1v(i)*std::cos(pcoord->x2v(j)); 
        }
        rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));

        // may be irrelevant
        if (rad < inner_radius) {
          den = n_inner;
          energy = e_inner;
        } else {
          den = n_inner * SQR((inner_radius / rad));
          energy = (e_inner * std::pow((inner_radius / rad), 2.0*gamma_param));
        }
        // ambient density and convserved variables
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i) * v1_inner;

        // currently assumes that inner velocities are given in same coordinate system
        // functions that generate values need to be functions if input
        // values need to be converted to correct coordinate system
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * v2_inner;
          phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i) * v3_inner;
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * v2_inner * rad;
          phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i) * v3_inner;
        } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * v2_inner * rad;
          phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i) * v3_inner * rad * std::sin(pcoord->x3v(k));
        }

        if (NON_BAROTROPIC_EOS) {
          // 0.5 * den *(v_r^2 + v_phi^2 + v_theta^2)
          // but move den inside and use momentum
          Real v_contrib;
          v_contrib = (0.5 / phydro->u(IDN,k,j,i)
                       * (SQR(phydro->u(IM1,k,j,i))
                          + SQR(phydro->u(IM2,k,j,i))
                          + SQR(phydro->u(IM3,k,j,i))
                          )
          );
          phydro->u(IEN,k,j,i) = energy + v_contrib;

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
  // not sure about units here or coordinate system
  if (MAGNETIC_FIELDS_ENABLED) {
    // b.x1f
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x = pcoord->x1f(i);
            y = pcoord->x2f(j);
            z = pcoord->x3f(k);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            x = pcoord->x1f(i)*std::cos(pcoord->x2f(j));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j));
            z = pcoord->x3f(k);
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            x = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::cos(pcoord->x3f(k));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::sin(pcoord->x3f(k));
            z = pcoord->x1f(i)*std::cos(pcoord->x2f(j)); 
          }

          rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
          if (rad < inner_radius) {
            pfield->b.x1f(k,j,i) = b1;
          } else {
            pfield->b.x1f(k,j,i) = b1 * SQR((inner_radius/rad));
          }
        }
      }
    }
    //b.x2f
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x = pcoord->x1f(i);
            y = pcoord->x2f(j);
            z = pcoord->x3f(k);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            x = pcoord->x1f(i)*std::cos(pcoord->x2f(j));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j));
            z = pcoord->x3f(k);
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            x = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::cos(pcoord->x3f(k));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::sin(pcoord->x3f(k));
            z = pcoord->x1f(i)*std::cos(pcoord->x2f(j)); 
          }
          rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
          if (rad < inner_radius) {
            pfield->b.x2f(k,j,i) = b2;
          } else {
            pfield->b.x2f(k,j,i) = b2 * (inner_radius/rad);
          }
        }
      }
    }
    // b.x3f
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x = pcoord->x1f(i);
            y = pcoord->x2f(j);
            z = pcoord->x3f(k);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            x = pcoord->x1f(i)*std::cos(pcoord->x2f(j));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j));
            z = pcoord->x3f(k);
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            x = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::cos(pcoord->x3f(k));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::sin(pcoord->x3f(k));
            z = pcoord->x1f(i)*std::cos(pcoord->x2f(j)); 
          }
          rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
          if (rad < inner_radius) {
            pfield->b.x3f(k,j,i) = b3;
          } else {
            pfield->b.x3f(k,j,i) = b3;
          }
        }
      }
    }

    // add magnetic field contribution to total energy
    // using face averaged values
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->u(IEN,k,j,i) += 0.5 * (SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                                           SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                                           SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)))
                                           );
          }
        }
      }
    }
  }
}

// add gravity potential term
// currently defined spherically for the sun at the center of the simulation
// converts to correct coordinates
void SunGravity(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {

  Real rad, rho, rho_0, x, y, z;
  Real grav_force, f1, f2, f3;

  // rho is the radius direction in cylindrical coordinates
  rho_0 = std::sqrt(SQR(x_0) + SQR(y_0));
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pmb->pcoord->x1v(i);
          y = pmb->pcoord->x2v(j);
          z = pmb->pcoord->x3v(k);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          x = pmb->pcoord->x1v(i)*std::cos(pmb->pcoord->x2v(j));
          y = pmb->pcoord->x1v(i)*std::sin(pmb->pcoord->x2v(j));
          z = pmb->pcoord->x3v(k);
          rho = std::sqrt(SQR(x) + SQR(y));
        } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          x = pmb->pcoord->x1v(i)*std::sin(pmb->pcoord->x2v(j))*std::cos(pmb->pcoord->x3v(k));
          y = pmb->pcoord->x1v(i)*std::sin(pmb->pcoord->x2v(j))*std::sin(pmb->pcoord->x3v(k));
          z = pmb->pcoord->x1v(i)*std::cos(pmb->pcoord->x2v(j)); 
        }

        rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
        grav_force = - GM_norm / SQR(rad) * cons(IDN,k,j,i);

        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          f1 = dt*grav_force * (x-x_0) / rad;
          f2 = dt*grav_force * (y-y_0) / rad;
          f3 = dt*grav_force * (z-z_0) / rad;
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          // not 100% sure this is correct
          f1 = dt*grav_force * (rho-rho_0) / rad;
          f2 = 0.0;
          f3 = dt*grav_force * (z-z_0) / rad;
        } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          f1 = dt*grav_force;
          f2 = 0.0;
          f3 = 0.0;
        }

        cons(IM1,k,j,i) += f1;
        cons(IM2,k,j,i) += f2;
        cons(IM3,k,j,i) += f3;

        // multiply gravitational potential by smoothing function
        // cons(IM3,k,j,i) -= dt*den*SQR(Omega_0)*x3*fsmooth;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) += dt*prim(IDN,k,j,i) * (prim(IVX,k,j,i) * f1
                                                   + prim(IVY,k,j,i) * f2
                                                   + prim(IVZ,k,j,i) * f3);
        }
      }
    }
  }
  return;
}

//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void CMEInnerX1(MeshBlock *pmb, Coordinates *pcoord,
                 AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real den, press, rad, x, y, z;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pcoord->x1v(il-i);
          y = pcoord->x2v(j);
          z = pcoord->x3v(k);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          x = pcoord->x1v(il-i)*std::cos(pcoord->x2v(j));
          y = pcoord->x1v(il-i)*std::sin(pcoord->x2v(j));
          z = pcoord->x3v(k);
        } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          x = pcoord->x1v(il-i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          y = pcoord->x1v(il-i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          z = pcoord->x1v(il-i)*std::cos(pcoord->x2v(j)); 
        }
       rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));

        // may need to be looked at again
        if (rad < inner_radius) {
          den = n_inner;
          press = e_inner * (gamma_param-1.0);
        } else {
          den = n_inner * SQR((inner_radius / rad));
          press = (e_inner * std::pow((inner_radius / rad), 2.0*gamma_param)) * (gamma_param-1.0);
        }

        prim(IDN,k,j,il-i) = den;
        prim(IVX,k,j,il-i) = v1_inner;
        // currently assumes that inner velocities are given in same coordinate system
        // functions that generate values need to be functions if input
        // values need to be converted to correct coordinate system
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          prim(IVY,k,j,il-i) = v2_inner;
          prim(IVZ,k,j,il-i) = v3_inner;
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          prim(IVY,k,j,il-i) = v2_inner * rad;
          prim(IVZ,k,j,il-i) = v3_inner;
        } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          prim(IVY,k,j,il-i) = v2_inner * rad;
          prim(IVZ,k,j,il-i) = v3_inner * rad * std::sin(pcoord->x3v(k));
        }

        if (NON_BAROTROPIC_EOS) {
          prim(IPR,k,j,il-i) = press;
        }
      }
    }
  }

  // set magnetic field in inlet ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          // copy face-centered magnetic fields into ghost zones
          // b.x1f(k,j,il-i) = b.x1f(k,j,il);
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x = pcoord->x1f(i);
            y = pcoord->x2f(j);
            z = pcoord->x3f(k);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            x = pcoord->x1f(i)*std::cos(pcoord->x2f(j));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j));
            z = pcoord->x3f(k);
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            x = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::cos(pcoord->x3f(k));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::sin(pcoord->x3f(k));
            z = pcoord->x1f(i)*std::cos(pcoord->x2f(j)); 
          }
          rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
          if (rad < inner_radius) {
            b.x1f(k,j,il-i) = b1;
          } else {
            b.x1f(k,j,il-i) = b1 * SQR((inner_radius/rad));
          }
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          // copy face-centered magnetic fields into ghost zones
          // b.x2f(k,j,il-i) = b.x2f(k,j,il);
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x = pcoord->x1f(i);
            y = pcoord->x2f(j);
            z = pcoord->x3f(k);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            x = pcoord->x1f(i)*std::cos(pcoord->x2f(j));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j));
            z = pcoord->x3f(k);
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            x = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::cos(pcoord->x3f(k));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::sin(pcoord->x3f(k));
            z = pcoord->x1f(i)*std::cos(pcoord->x2f(j)); 
          }
          rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
          if (rad < inner_radius) {
            b.x2f(k,j,il-i) = b2;
          } else {
            b.x2f(k,j,il-i) = b2 * (inner_radius/rad);
          }
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          // copy face-centered magnetic fields into ghost zones
          // b.x3f(k,j,il-i) = b.x3f(k,j,il);
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x = pcoord->x1f(i);
            y = pcoord->x2f(j);
            z = pcoord->x3f(k);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            x = pcoord->x1f(i)*std::cos(pcoord->x2f(j));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j));
            z = pcoord->x3f(k);
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            x = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::cos(pcoord->x3f(k));
            y = pcoord->x1f(i)*std::sin(pcoord->x2f(j))*std::sin(pcoord->x3f(k));
            z = pcoord->x1f(i)*std::cos(pcoord->x2f(j)); 
          }
          rad = std::sqrt(SQR(x - x_0) + SQR(y - y_0) + SQR(z - z_0));
          if (rad < inner_radius) {
            b.x3f(k,j,il-i) = b3;
          } else {
            b.x3f(k,j,il-i) = b3;
          }
        }
      }
    }
    // add magnetic field contribution to total pressure
    // using face averaged values
    if (NON_BAROTROPIC_EOS) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(IPR,k,j,il-i) += 0.5 * (SQR(0.5*(b.x1f(k,j,il-i) + b.x1f(k,j,il-i+1))) +
                                         SQR(0.5*(b.x2f(k,j,il-i) + b.x2f(k,j+1,il-i))) +
                                         SQR(0.5*(b.x3f(k,j,il-i) + b.x3f(k+1,j,il-i)))
                                        );
          }
        }
      }
    }
  }
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

// get normalized gravitational constant
Real calc_GM_sun(ParameterInput *pin) {
  Real AU = pin->GetReal("problem", "AU");
  Real t_o = pin->GetReal("problem", "t_o");
  Real GM_sun = pin->GetOrAddReal("problem", "GM_sun", 1.32712440018e20);
  Real GM_0 = GM_sun * SQR(t_o) / std::pow(AU,3);
  return GM_0;
}

// convert velocity to inner boundary velocity
// empirical formulation
Real calc_v1_inner(ParameterInput *pin) {
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
  Real ratio = (omega * AU) / (v_measure * 1000.0);
  Real B1_AU = B_AU / (std::sqrt(1.0 + SQR(ratio)));
  Real b1 = B1_AU * SQR((r_meas / min_radius));

  return b1;
}

// convert b2 magnetic field to inner boundary 
Real calc_b2_inner(ParameterInput *pin, Real b1, Real v1_inner) {
  Real min_radius = pin->GetReal("mesh", "x1min");
  Real omega = pin->GetReal("problem", "omega_sun");
  Real AU = pin->GetReal("problem", "AU");
  Real vo = pin->GetReal("problem", "vo");
  Real b_extra = pin->GetOrAddReal("problem", "b_measure_extra", 1.0);

  Real ratio_inner = omega * AU * min_radius / v1_inner / vo;
  Real b2 = -1.0* b1 * ratio_inner * b_extra;
  return b2;
}

// convert density to inner boundary 
Real calc_n_inner(ParameterInput *pin, Real v1_inner) {
  Real min_radius = pin->GetReal("mesh", "x1min");
  Real r_meas = pin->GetReal("problem", "r_measure");
  Real v_measure = pin->GetReal("problem", "v_measure");
  Real n_measure = pin->GetReal("problem", "n_measure");
  Real vo = pin->GetReal("problem", "vo");

  Real n_in = (n_measure * SQR((r_meas / min_radius)) * v_measure * 1000.0) / (v1_inner * vo);
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