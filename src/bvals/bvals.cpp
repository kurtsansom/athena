//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file bvals.cpp
//  \brief constructor/destructor and utility functions for BoundaryValues class
//======================================================================================

// C++ headers
#include <iostream>   // endl
#include <iomanip>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cstring>    // memcpy
#include <cstdlib>
#include <cmath>

// Athena++ classes headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../mesh_refinement/mesh_refinement.hpp"
#include "../mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../parameter_input.hpp"
#include "../utils/buffer_utils.hpp"

// this class header
#include "bvals.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

static NeighborIndexes ni_[56];
static int bufid_[56];

// BoundaryValues constructor - sets functions for the appropriate
// boundary conditions at each of the 6 dirs of a MeshBlock

BoundaryValues::BoundaryValues(MeshBlock *pmb, ParameterInput *pin)
{
  pmy_mblock_ = pmb;
  int cng=pmb->cnghost, cng1=0, cng2=0, cng3=0;
  if(pmb->block_size.nx2>1) cng1=cng, cng2=cng;
  if(pmb->block_size.nx3>1) cng3=cng;
  int f2d=0, f3d=0;
  if(pmb->block_size.nx2 > 1) f2d=1;
  if(pmb->block_size.nx3 > 1) f3d=1;
  for(int i=0; i<6; i++)
    BoundaryFunction_[i]=NULL;

// Set BC functions for each of the 6 boundaries in turn -------------------------------
  // Inner x1
  nface_=2; nedge_=0;
  switch(pmb->block_bcs[INNER_X1]){
    case REFLECTING_BNDRY:
      BoundaryFunction_[INNER_X1] = ReflectInnerX1;
      break;
    case OUTFLOW_BNDRY:
      BoundaryFunction_[INNER_X1] = OutflowInnerX1;
      break;
    case BLOCK_BNDRY: // block boundary
    case PERIODIC_BNDRY: // periodic boundary
      BoundaryFunction_[INNER_X1] = NULL;
      break;
    case USER_BNDRY: // user-enrolled BCs
      BoundaryFunction_[INNER_X1] = pmb->pmy_mesh->BoundaryFunction_[INNER_X1];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
          << "Flag ix1_bc=" << pmb->block_bcs[INNER_X1] << " not valid" << std::endl;
      throw std::runtime_error(msg.str().c_str());
   }

  // Outer x1
  switch(pmb->block_bcs[OUTER_X1]){
    case REFLECTING_BNDRY:
      BoundaryFunction_[OUTER_X1] = ReflectOuterX1;
      break;
    case OUTFLOW_BNDRY:
      BoundaryFunction_[OUTER_X1] = OutflowOuterX1;
      break;
    case BLOCK_BNDRY: // block boundary
    case PERIODIC_BNDRY: // periodic boundary
      BoundaryFunction_[OUTER_X1] = NULL;
      break;
    case USER_BNDRY: // user-enrolled BCs
      BoundaryFunction_[OUTER_X1] = pmb->pmy_mesh->BoundaryFunction_[OUTER_X1];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
          << "Flag ox1_bc=" << pmb->block_bcs[OUTER_X1] << " not valid" << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }

  if (pmb->block_size.nx2 > 1) {
    nface_=4; nedge_=4;
    // Inner x2
    switch(pmb->block_bcs[INNER_X2]){
      case REFLECTING_BNDRY:
        BoundaryFunction_[INNER_X2] = ReflectInnerX2;
        break;
      case OUTFLOW_BNDRY:
        BoundaryFunction_[INNER_X2] = OutflowInnerX2;
        break;
      case BLOCK_BNDRY: // block boundary
      case PERIODIC_BNDRY: // periodic boundary
      case POLAR_BNDRY: // polar boundary
        BoundaryFunction_[INNER_X2] = NULL;
        break;
      case USER_BNDRY: // user-enrolled BCs
        BoundaryFunction_[INNER_X2] = pmb->pmy_mesh->BoundaryFunction_[INNER_X2];
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ix2_bc=" << pmb->block_bcs[INNER_X2] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
     }

    // Outer x2
    switch(pmb->block_bcs[OUTER_X2]){
      case REFLECTING_BNDRY:
        BoundaryFunction_[OUTER_X2] = ReflectOuterX2;
        break;
      case OUTFLOW_BNDRY:
        BoundaryFunction_[OUTER_X2] = OutflowOuterX2;
        break;
      case BLOCK_BNDRY: // block boundary
      case PERIODIC_BNDRY: // periodic boundary
      case POLAR_BNDRY: // polar boundary
        BoundaryFunction_[OUTER_X2] = NULL;
        break;
      case USER_BNDRY: // user-enrolled BCs
        BoundaryFunction_[OUTER_X2] = pmb->pmy_mesh->BoundaryFunction_[OUTER_X2];
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ox2_bc=" << pmb->block_bcs[OUTER_X2] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
    }
  }

  if (pmb->block_size.nx3 > 1) {
    nface_=6; nedge_=12;
    // Inner x3
    switch(pmb->block_bcs[INNER_X3]){
      case REFLECTING_BNDRY:
        BoundaryFunction_[INNER_X3] = ReflectInnerX3;
        break;
      case OUTFLOW_BNDRY:
        BoundaryFunction_[INNER_X3] = OutflowInnerX3;
        break;
      case BLOCK_BNDRY: // block boundary
      case PERIODIC_BNDRY: // periodic boundary
        BoundaryFunction_[INNER_X3] = NULL;
        break;
      case USER_BNDRY: // user-enrolled BCs
        BoundaryFunction_[INNER_X3] = pmb->pmy_mesh->BoundaryFunction_[INNER_X3];
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ix3_bc=" << pmb->block_bcs[INNER_X3] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
     }

    // Outer x3
    switch(pmb->block_bcs[OUTER_X3]){
      case REFLECTING_BNDRY:
        BoundaryFunction_[OUTER_X3] = ReflectOuterX3;
        break;
      case OUTFLOW_BNDRY:
        BoundaryFunction_[OUTER_X3] = OutflowOuterX3;
        break;
      case BLOCK_BNDRY: // block boundary
      case PERIODIC_BNDRY: // periodic boundary
        BoundaryFunction_[OUTER_X3] = NULL;
        break;
      case USER_BNDRY: // user-enrolled BCs
        BoundaryFunction_[OUTER_X3] = pmb->pmy_mesh->BoundaryFunction_[OUTER_X3];
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ox3_bc=" << pmb->block_bcs[OUTER_X3] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
    }
  }

  // Count number of blocks wrapping around pole
  if (pmb->block_bcs[INNER_X2] == POLAR_BNDRY) {
    int level = pmb->loc.level - pmb->pmy_mesh->root_level;
    num_north_polar_blocks_ = pmb->pmy_mesh->nrbx3 * (1 << level);
  }
  else
    num_north_polar_blocks_ = 0;
  if (pmb->block_bcs[OUTER_X2] == POLAR_BNDRY) {
    int level = pmb->loc.level - pmb->pmy_mesh->root_level;
    num_south_polar_blocks_ = pmb->pmy_mesh->nrbx3 * (1 << level);
  }
  else
    num_south_polar_blocks_ = 0;

  // Clear flags and requests
  for(int l=0;l<NSTEP;l++) {
    for(int i=0;i<56;i++){
      hydro_flag_[l][i]=BNDRY_WAITING;
      field_flag_[l][i]=BNDRY_WAITING;
      hydro_send_[l][i]=NULL;
      hydro_recv_[l][i]=NULL;
      field_send_[l][i]=NULL;
      field_recv_[l][i]=NULL;
#ifdef MPI_PARALLEL
      req_hydro_send_[l][i]=MPI_REQUEST_NULL;
      req_hydro_recv_[l][i]=MPI_REQUEST_NULL;
#endif
    }
    for(int i=0;i<48;i++){
      emfcor_send_[l][i]=NULL;
      emfcor_recv_[l][i]=NULL;
      emfcor_flag_[l][i]=BNDRY_WAITING;
#ifdef MPI_PARALLEL
      req_emfcor_send_[l][i]=MPI_REQUEST_NULL;
      req_emfcor_recv_[l][i]=MPI_REQUEST_NULL;
#endif
    }
    for(int i=0;i<6;i++){
      flcor_send_[l][i]=NULL;
#ifdef MPI_PARALLEL
      req_flcor_send_[l][i]=MPI_REQUEST_NULL;
#endif
      for(int j=0;j<=1;j++) {
        for(int k=0;k<=1;k++) {
          flcor_recv_[l][i][j][k]=NULL;
          flcor_flag_[l][i][j][k]=BNDRY_WAITING;
#ifdef MPI_PARALLEL
          req_flcor_recv_[l][i][j][k]=MPI_REQUEST_NULL;
#endif
        }
      }
    }
    if (num_north_polar_blocks_ > 0) {
      emf_north_send_[l] = new Real *[num_north_polar_blocks_];
      emf_north_recv_[l] = new Real *[num_north_polar_blocks_];
      emf_north_flag_[l] = new enum BoundaryStatus[num_north_polar_blocks_];
#ifdef MPI_PARALLEL
      req_emf_north_send_[l] = new MPI_Request[num_north_polar_blocks_];
      req_emf_north_recv_[l] = new MPI_Request[num_north_polar_blocks_];
#endif
      for (int n = 0; n < num_north_polar_blocks_; ++n) {
        emf_north_send_[l][n] = NULL;
        emf_north_recv_[l][n] = NULL;
        emf_north_flag_[l][n] = BNDRY_WAITING;
#ifdef MPI_PARALLEL
        req_emf_north_send_[l][n] = MPI_REQUEST_NULL;
        req_emf_north_recv_[l][n] = MPI_REQUEST_NULL;
#endif
      }
    }
    if (num_south_polar_blocks_ > 0) {
      emf_south_send_[l] = new Real *[num_south_polar_blocks_];
      emf_south_recv_[l] = new Real *[num_south_polar_blocks_];
      emf_south_flag_[l] = new enum BoundaryStatus[num_south_polar_blocks_];
#ifdef MPI_PARALLEL
      req_emf_south_send_[l] = new MPI_Request[num_south_polar_blocks_];
      req_emf_south_recv_[l] = new MPI_Request[num_south_polar_blocks_];
#endif
      for (int n = 0; n < num_south_polar_blocks_; ++n) {
        emf_south_send_[l][n] = NULL;
        emf_south_recv_[l][n] = NULL;
        emf_south_flag_[l][n] = BNDRY_WAITING;
#ifdef MPI_PARALLEL
        req_emf_south_send_[l][n] = MPI_REQUEST_NULL;
        req_emf_south_recv_[l][n] = MPI_REQUEST_NULL;
#endif
      }
    }
  }

  // Allocate buffers for non-polar neighbor communication
  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->pmy_mesh->maxneighbor_;n++) {
      int size=((ni_[n].ox1==0)?pmb->block_size.nx1:NGHOST)
              *((ni_[n].ox2==0)?pmb->block_size.nx2:NGHOST)
              *((ni_[n].ox3==0)?pmb->block_size.nx3:NGHOST);
      if(pmb->pmy_mesh->multilevel==true) {
        int f2c=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2):NGHOST)
               *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2):NGHOST)
               *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2):NGHOST);
        int c2f=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2+cng1):cng)
               *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2+cng2):cng)
               *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2+cng3):cng);
        size=std::max(size,c2f);
        size=std::max(size,f2c);
      }
      size*=NHYDRO;
      hydro_send_[l][n]=new Real[size];
      hydro_recv_[l][n]=new Real[size];
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for(int l=0;l<NSTEP;l++) {
      for(int n=0;n<pmb->pmy_mesh->maxneighbor_;n++) {
        int size1=((ni_[n].ox1==0)?(pmb->block_size.nx1+1):NGHOST)
                 *((ni_[n].ox2==0)?(pmb->block_size.nx2):NGHOST)
                 *((ni_[n].ox3==0)?(pmb->block_size.nx3):NGHOST);
        int size2=((ni_[n].ox1==0)?(pmb->block_size.nx1):NGHOST)
                 *((ni_[n].ox2==0)?(pmb->block_size.nx2+f2d):NGHOST)
                 *((ni_[n].ox3==0)?(pmb->block_size.nx3):NGHOST);
        int size3=((ni_[n].ox1==0)?(pmb->block_size.nx1):NGHOST)
                 *((ni_[n].ox2==0)?(pmb->block_size.nx2):NGHOST)
                 *((ni_[n].ox3==0)?(pmb->block_size.nx3+f3d):NGHOST);
        int size=size1+size2+size3;
        if(pmb->pmy_mesh->multilevel==true) {
          if(ni_[n].type!=NEIGHBOR_FACE) {
            if(ni_[n].ox1!=0) size1=size1/NGHOST*(NGHOST+1);
            if(ni_[n].ox2!=0) size2=size2/NGHOST*(NGHOST+1);
            if(ni_[n].ox3!=0) size3=size3/NGHOST*(NGHOST+1);
          }
          size=size1+size2+size3;
          int f2c1=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2+1):cng)
                  *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2):cng)
                  *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2):cng);
          int f2c2=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2):cng)
                  *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2+f2d):cng)
                  *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2):cng);
          int f2c3=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2):cng)
                  *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2):cng)
                  *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2+f3d):cng);
          if(ni_[n].type!=NEIGHBOR_FACE) {
            if(ni_[n].ox1!=0) f2c1=f2c1/cng*(cng+1);
            if(ni_[n].ox2!=0) f2c2=f2c2/cng*(cng+1);
            if(ni_[n].ox3!=0) f2c3=f2c3/cng*(cng+1);
          }
          int fsize=f2c1+f2c2+f2c3;
          int c2f1=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2+1+cng):cng)
                  *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2+cng*f2d):cng)
                  *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2+cng*f3d):cng);
          int c2f2=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2+cng):cng)
                  *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2+f2d+cng*f2d):cng)
                  *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2+cng*f3d):cng);
          int c2f3=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2+cng):cng)
                  *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2+f2d*cng):cng)
                  *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2+f3d+cng*f3d):cng);
          int csize=c2f1+c2f2+c2f3;
          size=std::max(size,std::max(csize,fsize));
        }
        field_send_[l][n]=new Real[size];
        field_recv_[l][n]=new Real[size];

        // allocate EMF correction buffer
        if(ni_[n].type==NEIGHBOR_FACE) {
          if(pmb->block_size.nx3>1) { // 3D
            if(ni_[n].ox1!=0)
              size=(pmb->block_size.nx2+1)*(pmb->block_size.nx3)
                  +(pmb->block_size.nx2)*(pmb->block_size.nx3+1);
            else if(ni_[n].ox2!=0)
              size=(pmb->block_size.nx1+1)*(pmb->block_size.nx3)
                  +(pmb->block_size.nx1)*(pmb->block_size.nx3+1);
            else
              size=(pmb->block_size.nx1+1)*(pmb->block_size.nx2)
                  +(pmb->block_size.nx1)*(pmb->block_size.nx2+1);
          }
          else if(pmb->block_size.nx2>1) { // 2D
            if(ni_[n].ox1!=0)
              size=(pmb->block_size.nx2+1)+pmb->block_size.nx2;
            else 
              size=(pmb->block_size.nx1+1)+pmb->block_size.nx1;
          }
          else // 1D
            size=2;
        }
        else if(ni_[n].type==NEIGHBOR_EDGE) {
          if(pmb->block_size.nx3>1) { // 3D
            if(ni_[n].ox3==0) size=pmb->block_size.nx3;
            if(ni_[n].ox2==0) size=pmb->block_size.nx2;
            if(ni_[n].ox1==0) size=pmb->block_size.nx1;
          }
          else if(pmb->block_size.nx2>1)
            size=1;
        }
        else continue;
        emfcor_send_[l][n]=new Real[size];
        emfcor_recv_[l][n]=new Real[size];
      }
    }
  }

  // Allocate buffers for polar neighbor communication
  if (MAGNETIC_FIELDS_ENABLED) {
    if (num_north_polar_blocks_ > 0) {
      for (int l = 0; l < NSTEP; ++l) {
        for (int n = 0; n < num_north_polar_blocks_; ++n) {
          emf_north_send_[l][n] = new Real[pmb->block_size.nx1];
          emf_north_recv_[l][n] = new Real[pmb->block_size.nx1];
        }
      }
    }
    if (num_south_polar_blocks_ > 0) {
      for (int l = 0; l < NSTEP; ++l) {
        for (int n = 0; n < num_south_polar_blocks_; ++n) {
          emf_south_send_[l][n] = new Real[pmb->block_size.nx1];
          emf_south_recv_[l][n] = new Real[pmb->block_size.nx1];
        }
      }
    }
  }

  if(pmb->pmy_mesh->multilevel==true) { // SMR or AMR
    // allocate surface area array
    int nc1=pmb->block_size.nx1+2*NGHOST;
    sarea_[0].NewAthenaArray(nc1);
    sarea_[1].NewAthenaArray(nc1);
    int size[6], im, jm, km;
    // allocate flux correction buffer
    size[0]=size[1]=(pmb->block_size.nx2+1)/2*(pmb->block_size.nx3+1)/2*NHYDRO;
    size[2]=size[3]=(pmb->block_size.nx1+1)/2*(pmb->block_size.nx3+1)/2*NHYDRO;
    size[4]=size[5]=(pmb->block_size.nx1+1)/2*(pmb->block_size.nx2+1)/2*NHYDRO;
    if(pmb->block_size.nx3>1) // 3D
      jm=2, km=2;
    else if(pmb->block_size.nx2>1) // 2D
      jm=1, km=2;
    else // 1D
      jm=1, km=1;
    for(int l=0;l<NSTEP;l++) {
      for(int i=0;i<nface_;i++){
        flcor_send_[l][i]=new Real[size[i]];
        for(int j=0;j<jm;j++) {
          for(int k=0;k<km;k++)
            flcor_recv_[l][i][j][k]=new Real[size[i]];
        }
      }
    }
  }

 /* single CPU in the azimuthal direction with the polar boundary*/
  if(pmb->loc.level == pmb->pmy_mesh->root_level &&
     pmb->pmy_mesh->nrbx3 == 1 &&
     (pmb->block_bcs[INNER_X2]==POLAR_BNDRY||pmb->block_bcs[OUTER_X2]==POLAR_BNDRY))
       exc_.NewAthenaArray(pmb->ke+NGHOST+2);

}

// destructor

BoundaryValues::~BoundaryValues()
{
  MeshBlock *pmb=pmy_mblock_;
  for(int l=0;l<NSTEP;l++) {
    for(int i=0;i<pmb->pmy_mesh->maxneighbor_;i++) {
      delete [] hydro_send_[l][i];
      delete [] hydro_recv_[l][i];
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for(int l=0;l<NSTEP;l++) {
      for(int i=0;i<pmb->pmy_mesh->maxneighbor_;i++) { 
        delete [] field_send_[l][i];
        delete [] field_recv_[l][i];
        if(ni_[i].type==NEIGHBOR_FACE || ni_[i].type==NEIGHBOR_EDGE) {
          delete [] emfcor_send_[l][i];
          delete [] emfcor_recv_[l][i];
        }
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    if (num_north_polar_blocks_ > 0) {
      for (int l = 0; l < NSTEP; ++l) {
        for (int n = 0; n < num_north_polar_blocks_; ++n) {
          delete[] emf_north_send_[l][n];
          delete[] emf_north_recv_[l][n];
        }
      }
    }
    if (num_south_polar_blocks_ > 0) {
      for (int l = 0; l < NSTEP; ++l) {
        for (int n = 0; n < num_south_polar_blocks_; ++n) {
          delete[] emf_south_send_[l][n];
          delete[] emf_south_recv_[l][n];
        }
      }
    }
  }
  if(pmb->pmy_mesh->multilevel==true) {
    sarea_[0].DeleteAthenaArray();
    sarea_[1].DeleteAthenaArray();
    for(int l=0;l<NSTEP;l++) {
      for(int i=0;i<nface_;i++){
        delete [] flcor_send_[l][i];
        for(int j=0;j<2;j++) {
          for(int k=0;k<2;k++)
            delete [] flcor_recv_[l][i][j][k];
        }
      }
    }
  }
  if (num_north_polar_blocks_ > 0) {
    for (int l = 0; l < NSTEP; ++l) {
      delete[] emf_north_send_[l];
      delete[] emf_north_recv_[l];
      delete[] emf_north_flag_[l];
#ifdef MPI_PARALLEL
      delete[] req_emf_north_send_[l];
      delete[] req_emf_north_recv_[l];
#endif
    }
  }
  if (num_south_polar_blocks_ > 0) {
    for (int l = 0; l < NSTEP; ++l) {
      delete[] emf_south_send_[l];
      delete[] emf_south_recv_[l];
      delete[] emf_south_flag_[l];
#ifdef MPI_PARALLEL
      delete[] req_emf_south_send_[l];
      delete[] req_emf_south_recv_[l];
#endif
    }
  }
  if(pmb->loc.level == pmb->pmy_mesh->root_level &&
     pmb->pmy_mesh->nrbx3 == 1 &&
     (pmb->block_bcs[INNER_X2]==POLAR_BNDRY||pmb->block_bcs[OUTER_X2]==POLAR_BNDRY))
       exc_.DeleteAthenaArray();
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::Initialize(void)
//  \brief Initialize MPI requests
void BoundaryValues::Initialize(void)
{
  MeshBlock* pmb=pmy_mblock_;
  int myox1, myox2, myox3;
  int tag;
  int cng, cng1, cng2, cng3;
  int ssize, rsize;
  cng=cng1=pmb->cnghost;
  cng2=(pmb->block_size.nx2>1)?cng:0;
  cng3=(pmb->block_size.nx3>1)?cng:0;
  long int &lx1=pmb->loc.lx1;
  long int &lx2=pmb->loc.lx2;
  long int &lx3=pmb->loc.lx3;
  int &mylevel=pmb->loc.level;
  myox1=((int)(lx1&1L));
  myox2=((int)(lx2&1L));
  myox3=((int)(lx3&1L));
  int f2d=0, f3d=0;
  if(pmb->block_size.nx2 > 1) f2d=1;
  if(pmb->block_size.nx3 > 1) f3d=1;


  // count the number of the fine meshblocks contacting on each edge
  int eid=0;
  if(pmb->block_size.nx2 > 1) {
    for(int ox2=-1;ox2<=1;ox2+=2) {
      for(int ox1=-1;ox1<=1;ox1+=2) {
        int nis, nie, njs, nje;
        nis=std::max(ox1-1,-1), nie=std::min(ox1+1,1);
        njs=std::max(ox2-1,-1), nje=std::min(ox2+1,1);
        int nf=0, fl=mylevel;
        for(int nj=njs; nj<=nje; nj++) {
          for(int ni=nis; ni<=nie; ni++) {
            if(pmb->nblevel[1][nj+1][ni+1] > fl)
              fl++, nf=0;
            if(pmb->nblevel[1][nj+1][ni+1]==fl)
              nf++;
          }
        }
        edge_flag_[eid]=(fl==mylevel);
        nedge_fine_[eid++]=nf;
      }
    }
  }
  if(pmb->block_size.nx3 > 1) {
    for(int ox3=-1;ox3<=1;ox3+=2) {
      for(int ox1=-1;ox1<=1;ox1+=2) {
        int nis, nie, nks, nke;
        nis=std::max(ox1-1,-1), nie=std::min(ox1+1,1);
        nks=std::max(ox3-1,-1), nke=std::min(ox3+1,1);
        int nf=0, fl=mylevel;
        for(int nk=nks; nk<=nke; nk++) {
          for(int ni=nis; ni<=nie; ni++) {
            if(pmb->nblevel[nk+1][1][ni+1] > fl)
              fl++, nf=0;
            if(pmb->nblevel[nk+1][1][ni+1]==fl)
              nf++;
          }
        }
        edge_flag_[eid]=(fl==mylevel);
        nedge_fine_[eid++]=nf;
      }
    }
    for(int ox3=-1;ox3<=1;ox3+=2) {
      for(int ox2=-1;ox2<=1;ox2+=2) {
        int njs, nje, nks, nke;
        njs=std::max(ox2-1,-1), nje=std::min(ox2+1,1);
        nks=std::max(ox3-1,-1), nke=std::min(ox3+1,1);
        int nf=0, fl=mylevel;
        for(int nk=nks; nk<=nke; nk++) {
          for(int nj=njs; nj<=nje; nj++) {
            if(pmb->nblevel[nk+1][nj+1][1] > fl)
              fl++, nf=0;
            if(pmb->nblevel[nk+1][nj+1][1]==fl)
              nf++;
          }
        }
        edge_flag_[eid]=(fl==mylevel);
        nedge_fine_[eid++]=nf;
      }
    }
  }

#ifdef MPI_PARALLEL
  // Initialize non-polar neighbor communications to other ranks
  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->nneighbor;n++) {
      NeighborBlock& nb = pmb->neighbor[n];
      if(nb.rank!=Globals::my_rank) {
        if(nb.level==mylevel) { // same
          ssize=rsize=((nb.ox1==0)?pmb->block_size.nx1:NGHOST)
                     *((nb.ox2==0)?pmb->block_size.nx2:NGHOST)
                     *((nb.ox3==0)?pmb->block_size.nx3:NGHOST);
        }
        else if(nb.level<mylevel) { // coarser
          ssize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2):NGHOST)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2):NGHOST)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2):NGHOST);
          rsize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+cng1):cng1)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+cng2):cng2)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+cng3):cng3);
        }
        else { // finer
          ssize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+cng1):cng1)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+cng2):cng2)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+cng3):cng3);
          rsize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2):NGHOST)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2):NGHOST)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2):NGHOST);
        }
        ssize*=NHYDRO; rsize*=NHYDRO;
        // specify the offsets in the view point of the target block: flip ox? signs
        tag=CreateMPITag(nb.lid, l, tag_hydro, nb.targetid);
        MPI_Send_init(hydro_send_[l][nb.bufid],ssize,MPI_ATHENA_REAL,
                      nb.rank,tag,MPI_COMM_WORLD,&req_hydro_send_[l][nb.bufid]);
        tag=CreateMPITag(pmb->lid, l, tag_hydro, nb.bufid);
        MPI_Recv_init(hydro_recv_[l][nb.bufid],rsize,MPI_ATHENA_REAL,
                      nb.rank,tag,MPI_COMM_WORLD,&req_hydro_recv_[l][nb.bufid]);

        // flux correction
        if(pmb->pmy_mesh->multilevel==true && nb.type==NEIGHBOR_FACE) {
          int fi1, fi2, size;
          if(nb.fid==0 || nb.fid==1)
            fi1=myox2, fi2=myox3, size=((pmb->block_size.nx2+1)/2)*((pmb->block_size.nx3+1)/2);
          else if(nb.fid==2 || nb.fid==3)
            fi1=myox1, fi2=myox3, size=((pmb->block_size.nx1+1)/2)*((pmb->block_size.nx3+1)/2);
          else if(nb.fid==4 || nb.fid==5)
            fi1=myox1, fi2=myox2, size=((pmb->block_size.nx1+1)/2)*((pmb->block_size.nx2+1)/2);
          size*=NHYDRO;
          if(nb.level<mylevel) { // send to coarser
            tag=CreateMPITag(nb.lid, l, tag_flcor, nb.targetid);
            MPI_Send_init(flcor_send_[l][nb.fid],size,MPI_ATHENA_REAL,
                nb.rank,tag,MPI_COMM_WORLD,&req_flcor_send_[l][nb.fid]);
          }
          else if(nb.level>mylevel) { // receive from finer
            tag=CreateMPITag(pmb->lid, l, tag_flcor, nb.bufid);
            MPI_Recv_init(flcor_recv_[l][nb.fid][nb.fi2][nb.fi1],size,MPI_ATHENA_REAL,
                nb.rank,tag,MPI_COMM_WORLD,&req_flcor_recv_[l][nb.fid][nb.fi2][nb.fi1]);
          }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
          int size, csize, fsize;
          int size1=((nb.ox1==0)?(pmb->block_size.nx1+1):NGHOST)
                   *((nb.ox2==0)?(pmb->block_size.nx2):NGHOST)
                   *((nb.ox3==0)?(pmb->block_size.nx3):NGHOST);
          int size2=((nb.ox1==0)?(pmb->block_size.nx1):NGHOST)
                   *((nb.ox2==0)?(pmb->block_size.nx2+f2d):NGHOST)
                   *((nb.ox3==0)?(pmb->block_size.nx3):NGHOST);
          int size3=((nb.ox1==0)?(pmb->block_size.nx1):NGHOST)
                   *((nb.ox2==0)?(pmb->block_size.nx2):NGHOST)
                   *((nb.ox3==0)?(pmb->block_size.nx3+f3d):NGHOST);
          size=size1+size2+size3;
          if(pmb->pmy_mesh->multilevel==true) {
            if(nb.type!=NEIGHBOR_FACE) {
              if(nb.ox1!=0) size1=size1/NGHOST*(NGHOST+1);
              if(nb.ox2!=0) size2=size2/NGHOST*(NGHOST+1);
              if(nb.ox3!=0) size3=size3/NGHOST*(NGHOST+1);
            }
            size=size1+size2+size3;
            int f2c1=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+1):cng)
                    *((nb.ox2==0)?((pmb->block_size.nx2+1)/2):cng)
                    *((nb.ox3==0)?((pmb->block_size.nx3+1)/2):cng);
            int f2c2=((nb.ox1==0)?((pmb->block_size.nx1+1)/2):cng)
                    *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+f2d):cng)
                    *((nb.ox3==0)?((pmb->block_size.nx3+1)/2):cng);
            int f2c3=((nb.ox1==0)?((pmb->block_size.nx1+1)/2):cng)
                    *((nb.ox2==0)?((pmb->block_size.nx2+1)/2):cng)
                    *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+f3d):cng);
            if(nb.type!=NEIGHBOR_FACE) {
              if(nb.ox1!=0) f2c1=f2c1/cng*(cng+1);
              if(nb.ox2!=0) f2c2=f2c2/cng*(cng+1);
              if(nb.ox3!=0) f2c3=f2c3/cng*(cng+1);
            }
            fsize=f2c1+f2c2+f2c3;
            int c2f1=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+1+cng):cng)
                    *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+cng*f2d):cng)
                    *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+cng*f3d):cng);
            int c2f2=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+cng):cng)
                    *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+f2d+cng*f2d):cng)
                    *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+cng*f3d):cng);
            int c2f3=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+cng):cng)
                    *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+f2d*cng):cng)
                    *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+f3d+cng*f3d):cng);
            csize=c2f1+c2f2+c2f3;
          }
          if(nb.level==mylevel) // same
            ssize=size, rsize=size;
          else if(nb.level<mylevel) // coarser
            ssize=fsize, rsize=csize;
          else // finer
            ssize=csize, rsize=fsize;

          tag=CreateMPITag(nb.lid, l, tag_field, nb.targetid);
          MPI_Send_init(field_send_[l][nb.bufid],ssize,MPI_ATHENA_REAL,
                        nb.rank,tag,MPI_COMM_WORLD,&req_field_send_[l][nb.bufid]);
          tag=CreateMPITag(pmb->lid, l, tag_field, nb.bufid);
          MPI_Recv_init(field_recv_[l][nb.bufid],rsize,MPI_ATHENA_REAL,
                        nb.rank,tag,MPI_COMM_WORLD,&req_field_recv_[l][nb.bufid]);
          // EMF correction
          int fi1, fi2, f2csize;
          if(nb.type==NEIGHBOR_FACE) { // face
            if(pmb->block_size.nx3 > 1) { // 3D
              if(nb.fid==INNER_X1 || nb.fid==OUTER_X1) {
                size=(pmb->block_size.nx2+1)*(pmb->block_size.nx3)
                    +(pmb->block_size.nx2)*(pmb->block_size.nx3+1);
                f2csize=(pmb->block_size.nx2/2+1)*(pmb->block_size.nx3/2)
                    +(pmb->block_size.nx2/2)*(pmb->block_size.nx3/2+1);
              }
              else if(nb.fid==INNER_X2 || nb.fid==OUTER_X2) {
                size=(pmb->block_size.nx1+1)*(pmb->block_size.nx3)
                    +(pmb->block_size.nx1)*(pmb->block_size.nx3+1);
                f2csize=(pmb->block_size.nx1/2+1)*(pmb->block_size.nx3/2)
                    +(pmb->block_size.nx1/2)*(pmb->block_size.nx3/2+1);
              }
              else if(nb.fid==INNER_X3 || nb.fid==OUTER_X3) {
                size=(pmb->block_size.nx1+1)*(pmb->block_size.nx2)
                    +(pmb->block_size.nx1)*(pmb->block_size.nx2+1);
                f2csize=(pmb->block_size.nx1/2+1)*(pmb->block_size.nx2/2)
                    +(pmb->block_size.nx1/2)*(pmb->block_size.nx2/2+1);
              }
            }
            else if(pmb->block_size.nx2 > 1) { // 2D
              if(nb.fid==INNER_X1 || nb.fid==OUTER_X1) {
                size=(pmb->block_size.nx2+1)+pmb->block_size.nx2;
                f2csize=(pmb->block_size.nx2/2+1)+pmb->block_size.nx2/2;
              }
              else if(nb.fid==INNER_X2 || nb.fid==OUTER_X2) {
                size=(pmb->block_size.nx1+1)+pmb->block_size.nx1;
                f2csize=(pmb->block_size.nx1/2+1)+pmb->block_size.nx1/2;
              }
            }
            else // 1D
              size=f2csize=2;
          }
          else if(nb.type==NEIGHBOR_EDGE) { // edge
            if(pmb->block_size.nx3 > 1) { // 3D
              if(nb.eid>=0 && nb.eid<4) {
                size=pmb->block_size.nx3;
                f2csize=pmb->block_size.nx3/2;
              }
              else if(nb.eid>=4 && nb.eid<8) {
                size=pmb->block_size.nx2;
                f2csize=pmb->block_size.nx2/2;
              }
              else if(nb.eid>=8 && nb.eid<12) {
                size=pmb->block_size.nx1;
                f2csize=pmb->block_size.nx1/2;
              }
            }
            else if(pmb->block_size.nx2 > 1) // 2D
              size=f2csize=1;
          }
          else // corner
            continue;

          if(nb.level==mylevel) { // the same level
            if((nb.type==NEIGHBOR_FACE) || ((nb.type==NEIGHBOR_EDGE) && (edge_flag_[nb.eid]==true))) {
              tag=CreateMPITag(nb.lid, l, tag_emfcor, nb.targetid);
              MPI_Send_init(emfcor_send_[l][nb.bufid],size,MPI_ATHENA_REAL,
                            nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_send_[l][nb.bufid]);
              tag=CreateMPITag(pmb->lid, l, tag_emfcor, nb.bufid);
              MPI_Recv_init(emfcor_recv_[l][nb.bufid],size,MPI_ATHENA_REAL,
                            nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_recv_[l][nb.bufid]);
            }
          }
          if(nb.level>mylevel) { // finer neighbor
            tag=CreateMPITag(pmb->lid, l, tag_emfcor, nb.bufid);
            MPI_Recv_init(emfcor_recv_[l][nb.bufid],f2csize,MPI_ATHENA_REAL,
                          nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_recv_[l][nb.bufid]);
          }
          if(nb.level<mylevel) { // coarser neighbor
            tag=CreateMPITag(nb.lid, l, tag_emfcor, nb.targetid);
            MPI_Send_init(emfcor_send_[l][nb.bufid],f2csize,MPI_ATHENA_REAL,
                          nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_send_[l][nb.bufid]);
          }
        }
      }
    }
  }

  // Initialize polar neighbor communications to other ranks
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int l = 0; l < NSTEP; ++l) {
      for (int n = 0; n < num_north_polar_blocks_; ++n) {
        const PolarNeighborBlock &nb = pmb->polar_neighbor_north[n];
        if(nb.rank != Globals::my_rank) {
          tag = CreateMPITag(nb.lid, l, tag_emfpole, pmb->loc.lx3);
          MPI_Send_init(emf_north_send_[l][n], pmb->block_size.nx1, MPI_ATHENA_REAL,
              nb.rank, tag, MPI_COMM_WORLD, &req_emf_north_send_[l][n]);
          tag = CreateMPITag(pmb->lid, l, tag_emfpole, n);
          MPI_Recv_init(emf_north_recv_[l][n], pmb->block_size.nx1, MPI_ATHENA_REAL,
              nb.rank, tag, MPI_COMM_WORLD, &req_emf_north_recv_[l][n]);
        }
      }
      for (int n = 0; n < num_south_polar_blocks_; ++n) {
        const PolarNeighborBlock &nb = pmb->polar_neighbor_south[n];
        if(nb.rank != Globals::my_rank) {
          tag = CreateMPITag(nb.lid, l, tag_emfpole, pmb->loc.lx3);
          MPI_Send_init(emf_south_send_[l][n], pmb->block_size.nx1, MPI_ATHENA_REAL,
              nb.rank, tag, MPI_COMM_WORLD, &req_emf_south_send_[l][n]);
          tag = CreateMPITag(pmb->lid, l, tag_emfpole, n);
          MPI_Recv_init(emf_south_recv_[l][n], pmb->block_size.nx1, MPI_ATHENA_REAL,
              nb.rank, tag, MPI_COMM_WORLD, &req_emf_south_recv_[l][n]);
        }
      }
    }
  }
#endif
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::CheckBoundary(void)
//  \brief checks if the boundary conditions are correctly enrolled
void BoundaryValues::CheckBoundary(void)
{
  MeshBlock *pmb=pmy_mblock_;
  for(int i=0;i<nface_;i++) {
    if(pmb->block_bcs[i]==USER_BNDRY) {
      if(BoundaryFunction_[i]==NULL) {
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues::CheckBoundary" << std::endl
            << "A user-defined boundary is specified but the hydro boundary function "
            << "is not enrolled in direction " << i  << "." << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::StartReceivingForInit(void)
//  \brief initiate MPI_Irecv for initialization
void BoundaryValues::StartReceivingForInit(void)
{
#ifdef MPI_PARALLEL
  MeshBlock *pmb=pmy_mblock_;
  for(int n=0;n<pmb->nneighbor;n++) {
    NeighborBlock& nb = pmb->neighbor[n];
    if(nb.rank!=Globals::my_rank) { 
      MPI_Start(&req_hydro_recv_[0][nb.bufid]);
      if (MAGNETIC_FIELDS_ENABLED)
        MPI_Start(&req_field_recv_[0][nb.bufid]);
      // Prep sending primitives to enable cons->prim inversion before prolongation
      if (GENERAL_RELATIVITY and pmb->pmy_mesh->multilevel)
        MPI_Start(&req_hydro_recv_[1][nb.bufid]);
    }
  }
#endif
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::StartReceivingAll(void)
//  \brief initiate MPI_Irecv for all the sweeps
void BoundaryValues::StartReceivingAll(void)
{
  for(int l=0;l<NSTEP;l++)
    firsttime_[l]=true;
#ifdef MPI_PARALLEL
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->loc.level;
  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->nneighbor;n++) {
      NeighborBlock& nb = pmb->neighbor[n];
      if(nb.rank!=Globals::my_rank) { 
        MPI_Start(&req_hydro_recv_[l][nb.bufid]);
        if(nb.type==NEIGHBOR_FACE && nb.level>mylevel)
          MPI_Start(&req_flcor_recv_[l][nb.fid][nb.fi2][nb.fi1]);
        if (MAGNETIC_FIELDS_ENABLED) {
          MPI_Start(&req_field_recv_[l][nb.bufid]);
          if(nb.type==NEIGHBOR_FACE || nb.type==NEIGHBOR_EDGE) {
            if((nb.level>mylevel) || ((nb.level==mylevel) && ((nb.type==NEIGHBOR_FACE)
            || ((nb.type==NEIGHBOR_EDGE) && (edge_flag_[nb.eid]==true)))))
              MPI_Start(&req_emfcor_recv_[l][nb.bufid]);
          }
        }
      }
    }
    if (MAGNETIC_FIELDS_ENABLED) {
      for (int n = 0; n < num_north_polar_blocks_; ++n) {
        const PolarNeighborBlock &nb = pmb->polar_neighbor_north[n];
        if (nb.rank != Globals::my_rank) {
          MPI_Start(&req_emf_north_recv_[l][n]);
        }
      }
      for (int n = 0; n < num_south_polar_blocks_; ++n) {
        const PolarNeighborBlock &nb = pmb->polar_neighbor_south[n];
        if (nb.rank != Globals::my_rank) {
          MPI_Start(&req_emf_south_recv_[l][n]);
        }
      }
    }
  }
#endif
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ClearBoundaryForInit(void)
//  \brief clean up the boundary flags for initialization
void BoundaryValues::ClearBoundaryForInit(void)
{
  MeshBlock *pmb=pmy_mblock_;

  // Note step==0 corresponds to initial exchange of conserved variables, while step==1
  // corresponds to primitives sent only in the case of GR with refinement
  for(int n=0;n<pmb->nneighbor;n++) {
    NeighborBlock& nb = pmb->neighbor[n];
    hydro_flag_[0][nb.bufid] = BNDRY_WAITING;
    if (MAGNETIC_FIELDS_ENABLED)
      field_flag_[0][nb.bufid] = BNDRY_WAITING;
    if (GENERAL_RELATIVITY and pmb->pmy_mesh->multilevel)
      hydro_flag_[1][nb.bufid] = BNDRY_WAITING;
#ifdef MPI_PARALLEL
    if(nb.rank!=Globals::my_rank) {
      MPI_Wait(&req_hydro_send_[0][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
      if (MAGNETIC_FIELDS_ENABLED)
        MPI_Wait(&req_field_send_[0][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
      if (GENERAL_RELATIVITY and pmb->pmy_mesh->multilevel)
        MPI_Wait(&req_hydro_send_[1][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
    }
#endif
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ClearBoundaryAll(void)
//  \brief clean up the boundary flags after each loop
void BoundaryValues::ClearBoundaryAll(void)
{
  MeshBlock *pmb=pmy_mblock_;

  // Clear non-polar boundary communications
  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->nneighbor;n++) {
      NeighborBlock& nb = pmb->neighbor[n];
      hydro_flag_[l][nb.bufid] = BNDRY_WAITING;
      if(nb.type==NEIGHBOR_FACE)
        flcor_flag_[l][nb.fid][nb.fi2][nb.fi1] = BNDRY_WAITING;
      if (MAGNETIC_FIELDS_ENABLED) {
        field_flag_[l][nb.bufid] = BNDRY_WAITING;
        if((nb.type==NEIGHBOR_FACE) || (nb.type==NEIGHBOR_EDGE))
          emfcor_flag_[l][nb.bufid] = BNDRY_WAITING;
      }
#ifdef MPI_PARALLEL
      if(nb.rank!=Globals::my_rank) {
        MPI_Wait(&req_hydro_send_[l][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
        if(nb.type==NEIGHBOR_FACE && nb.level<pmb->loc.level)
          MPI_Wait(&req_flcor_send_[l][nb.fid],MPI_STATUS_IGNORE); // Wait for Isend
        if (MAGNETIC_FIELDS_ENABLED) {
          MPI_Wait(&req_field_send_[l][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
          if(nb.type==NEIGHBOR_FACE || nb.type==NEIGHBOR_EDGE) {
            if(nb.level < pmb->loc.level)
              MPI_Wait(&req_emfcor_send_[l][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
            else if((nb.level==pmb->loc.level) && ((nb.type==NEIGHBOR_FACE)
                || ((nb.type==NEIGHBOR_EDGE) && (edge_flag_[nb.eid]==true))))
              MPI_Wait(&req_emfcor_send_[l][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
          }
        }
      }
#endif
    }
  }

  // Clear polar boundary communications
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int l = 0; l < NSTEP; ++l) {
      for (int n = 0; n < num_north_polar_blocks_; ++n) {
        PolarNeighborBlock &nb = pmb->polar_neighbor_north[n];
        emf_north_flag_[l][n] = BNDRY_WAITING;
#ifdef MPI_PARALLEL
        if(nb.rank != Globals::my_rank)
          MPI_Wait(&req_emf_north_send_[l][n], MPI_STATUS_IGNORE);
#endif
      }
      for (int n = 0; n < num_south_polar_blocks_; ++n) {
        PolarNeighborBlock &nb = pmb->polar_neighbor_south[n];
        emf_south_flag_[l][n] = BNDRY_WAITING;
#ifdef MPI_PARALLEL
        if(nb.rank != Globals::my_rank)
          MPI_Wait(&req_emf_south_send_[l][n], MPI_STATUS_IGNORE);
#endif
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ApplyPhysicalBoundaries(AthenaArray<Real> &pdst,
//           AthenaArray<Real> &cdst, FaceField &bfdst, AthenaArray<Real> &bcdst)
//                                                   FaceField &bdst)
//  \brief Apply all the physical boundary conditions for both hydro and field
void BoundaryValues::ApplyPhysicalBoundaries(AthenaArray<Real> &pdst,
     AthenaArray<Real> &cdst, FaceField &bfdst, AthenaArray<Real> &bcdst)
{
  MeshBlock *pmb=pmy_mblock_;
  Coordinates *pco=pmb->pcoord;
  int bis=pmb->is, bie=pmb->ie, bjs=pmb->js, bje=pmb->je, bks=pmb->ks, bke=pmb->ke;
  if(pmb->pmy_mesh->face_only==false) { // extend the ghost zone
    bis=pmb->is-NGHOST;
    bie=pmb->ie+NGHOST;
    if(BoundaryFunction_[INNER_X2]==NULL && pmb->block_size.nx2>1) bjs=pmb->js-NGHOST;
    if(BoundaryFunction_[OUTER_X2]==NULL && pmb->block_size.nx2>1) bje=pmb->je+NGHOST;
    if(BoundaryFunction_[INNER_X3]==NULL && pmb->block_size.nx3>1) bks=pmb->ks-NGHOST;
    if(BoundaryFunction_[OUTER_X3]==NULL && pmb->block_size.nx3>1) bke=pmb->ke+NGHOST;
  }
  // Apply boundary function on inner-x1
  if (BoundaryFunction_[INNER_X1] != NULL) {
    BoundaryFunction_[INNER_X1](pmb, pco, pdst, bfdst, pmb->is, pmb->ie, bjs,bje,bks,bke);
    if(MAGNETIC_FIELDS_ENABLED) {
      pmb->pfield->CalculateCellCenteredField(bfdst, bcdst, pco,
        pmb->is-NGHOST, pmb->is-1, bjs, bje, bks, bke);
    }
    pmb->peos->PrimitiveToConserved(pdst, bcdst, cdst, pco,
      pmb->is-NGHOST, pmb->is-1, bjs, bje, bks, bke);
  }

  // Apply boundary function on outer-x1
  if (BoundaryFunction_[OUTER_X1] != NULL) {
    BoundaryFunction_[OUTER_X1](pmb, pco, pdst, bfdst, pmb->is, pmb->ie, bjs,bje,bks,bke);
    if(MAGNETIC_FIELDS_ENABLED) {
      pmb->pfield->CalculateCellCenteredField(bfdst, bcdst, pco,
        pmb->ie+1, pmb->ie+NGHOST, bjs, bje, bks, bke);
    }
    pmb->peos->PrimitiveToConserved(pdst, bcdst, cdst, pco,
      pmb->ie+1, pmb->ie+NGHOST, bjs, bje, bks, bke);
  }

  if(pmb->block_size.nx2>1) { // 2D or 3D

    // Apply boundary function on inner-x2
    if (BoundaryFunction_[INNER_X2] != NULL) {
      BoundaryFunction_[INNER_X2](pmb, pco, pdst, bfdst, bis,bie, pmb->js, pmb->je, bks,bke);
      if(MAGNETIC_FIELDS_ENABLED) {
        pmb->pfield->CalculateCellCenteredField(bfdst, bcdst, pco,
          bis, bie, pmb->js-NGHOST, pmb->js-1, bks, bke);
      }
      pmb->peos->PrimitiveToConserved(pdst, bcdst, cdst, pco,
        bis, bie, pmb->js-NGHOST, pmb->js-1, bks, bke);
    }

    // Apply boundary function on outer-x2
    if (BoundaryFunction_[OUTER_X2] != NULL) {
      BoundaryFunction_[OUTER_X2](pmb, pco, pdst, bfdst, bis,bie, pmb->js, pmb->je, bks,bke);
      if(MAGNETIC_FIELDS_ENABLED) {
        pmb->pfield->CalculateCellCenteredField(bfdst, bcdst, pco,
          bis, bie, pmb->je+1, pmb->je+NGHOST, bks, bke);
      }
      pmb->peos->PrimitiveToConserved(pdst, bcdst, cdst, pco,
        bis, bie, pmb->je+1, pmb->je+NGHOST, bks, bke);
    }
  }

  if(pmb->block_size.nx3>1) { // 3D
    if(pmb->pmy_mesh->face_only==false) {
      bjs=pmb->js-NGHOST;
      bje=pmb->je+NGHOST;
    }

    // Apply boundary function on inner-x3
    if (BoundaryFunction_[INNER_X3] != NULL) {
      BoundaryFunction_[INNER_X3](pmb, pco, pdst, bfdst, bis,bie,bjs,bje, pmb->ks, pmb->ke);
      if(MAGNETIC_FIELDS_ENABLED) {
        pmb->pfield->CalculateCellCenteredField(bfdst, bcdst, pco,
          bis, bie, bjs, bje, pmb->ks-NGHOST, pmb->ks-1);
      }
      pmb->peos->PrimitiveToConserved(pdst, bcdst, cdst, pco,
        bis, bie, bjs, bje, pmb->ks-NGHOST, pmb->ks-1);
    }

    // Apply boundary function on outer-x3
    if (BoundaryFunction_[OUTER_X3] != NULL) {
      BoundaryFunction_[OUTER_X3](pmb, pco, pdst, bfdst, bis,bie,bjs,bje, pmb->ks, pmb->ke);
      if(MAGNETIC_FIELDS_ENABLED) {
        pmb->pfield->CalculateCellCenteredField(bfdst, bcdst, pco,
          bis, bie, bjs, bje, pmb->ke+1, pmb->ke+NGHOST);
      }
      pmb->peos->PrimitiveToConserved(pdst, bcdst, cdst, pco,
        bis, bie, bjs, bje, pmb->ke+1, pmb->ke+NGHOST);
    }
  }

  return;
}


//--------------------------------------------------------------------------------------
//! \fn unsigned int CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2)
//  \brief calculate a buffer identifier
unsigned int CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2)
{
  unsigned int ux1=(unsigned)(ox1+1);
  unsigned int ux2=(unsigned)(ox2+1);
  unsigned int ux3=(unsigned)(ox3+1);
  return (ux1<<6) | (ux2<<4) | (ux3<<2) | (fi1<<1) | fi2;
}


//--------------------------------------------------------------------------------------
//! \fn int BufferID(int dim, bool multilevel, bool face_only)
//  \brief calculate neighbor indexes and target buffer IDs
int BufferID(int dim, bool multilevel, bool face_only)
{
  int nf1=1, nf2=1;
  if(multilevel==true) {
    if(dim>=2) nf1=2;
    if(dim>=3) nf2=2;
  }
  int b=0;
  // x1 face
  for(int n=-1; n<=1; n+=2) {
    for(int f2=0;f2<nf2;f2++) {
      for(int f1=0;f1<nf1;f1++) {
        ni_[b].ox1=n; ni_[b].ox2=0; ni_[b].ox3=0;
        ni_[b].fi1=f1; ni_[b].fi2=f2; ni_[b].type=NEIGHBOR_FACE;
        b++;
      }
    }
  }
  // x2 face
  if(dim>=2) {
    for(int n=-1; n<=1; n+=2) {
      for(int f2=0;f2<nf2;f2++) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=0; ni_[b].ox2=n; ni_[b].ox3=0;
          ni_[b].fi1=f1; ni_[b].fi2=f2; ni_[b].type=NEIGHBOR_FACE;
          b++;
        }
      }
    }
  }
  if(dim==3) {
    // x3 face
    for(int n=-1; n<=1; n+=2) {
      for(int f2=0;f2<nf2;f2++) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=0; ni_[b].ox2=0; ni_[b].ox3=n;
          ni_[b].fi1=f1; ni_[b].fi2=f2; ni_[b].type=NEIGHBOR_FACE;
          b++;
        }
      }
    }
  }
  // edges
  // x1x2
  if(dim>=2) {
    for(int m=-1; m<=1; m+=2) {
      for(int n=-1; n<=1; n+=2) {
        for(int f1=0;f1<nf2;f1++) {
          ni_[b].ox1=n; ni_[b].ox2=m; ni_[b].ox3=0;
          ni_[b].fi1=f1; ni_[b].fi2=0; ni_[b].type=NEIGHBOR_EDGE;
          b++;
        }
      }
    }
  }
  if(dim==3) {
    // x1x3
    for(int m=-1; m<=1; m+=2) {
      for(int n=-1; n<=1; n+=2) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=n; ni_[b].ox2=0; ni_[b].ox3=m;
          ni_[b].fi1=f1; ni_[b].fi2=0; ni_[b].type=NEIGHBOR_EDGE;
          b++;
        }
      }
    }
    // x2x3
    for(int m=-1; m<=1; m+=2) {
      for(int n=-1; n<=1; n+=2) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=0; ni_[b].ox2=n; ni_[b].ox3=m;
          ni_[b].fi1=f1; ni_[b].fi2=0; ni_[b].type=NEIGHBOR_EDGE;
          b++;
        }
      }
    }
    // corners
    for(int l=-1; l<=1; l+=2) {
      for(int m=-1; m<=1; m+=2) {
        for(int n=-1; n<=1; n+=2) {
          ni_[b].ox1=n; ni_[b].ox2=m; ni_[b].ox3=l;
          ni_[b].fi1=0; ni_[b].fi2=0; ni_[b].type=NEIGHBOR_CORNER;
          b++;
        }
      }
    }
  }

  for(int n=0;n<b;n++)
    bufid_[n]=CreateBufferID(ni_[n].ox1, ni_[n].ox2, ni_[n].ox3, ni_[n].fi1, ni_[n].fi2);

  return b;
}

int FindBufferID(int ox1, int ox2, int ox3, int fi1, int fi2, int bmax)
{
  int bid=CreateBufferID(ox1, ox2, ox3, fi1, fi2);

  for(int i=0;i<bmax;i++) {
    if(bid==bufid_[i]) return i;
  }
  return -1;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ProlongateBoundaries(AthenaArray<Real> &pdst,
//           AthenaArray<Real> &cdst, FaceField &bdst, AthenaArray<Real> &bcdst)
//  \brief Prolongate the level boundary using the coarse data
void BoundaryValues::ProlongateBoundaries(AthenaArray<Real> &pdst,
     AthenaArray<Real> &cdst, FaceField &bfdst, AthenaArray<Real> &bcdst)
{
  MeshBlock *pmb=pmy_mblock_;
  MeshRefinement *pmr=pmb->pmr;
  int mox1, mox2, mox3;
  long int &lx1=pmb->loc.lx1;
  long int &lx2=pmb->loc.lx2;
  long int &lx3=pmb->loc.lx3;
  int &mylevel=pmb->loc.level;
  mox1=((int)(lx1&1L)<<1)-1;
  mox2=((int)(lx2&1L)<<1)-1;
  mox3=((int)(lx3&1L)<<1)-1;

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb = pmb->neighbor[n];
    if(nb.level >= mylevel) continue;

    int mytype=std::abs(nb.ox1)+std::abs(nb.ox2)+std::abs(nb.ox3);
    // fill the required ghost-ghost zone
    int nis, nie, njs, nje, nks, nke;
    nis=std::max(nb.ox1-1,-1), nie=std::min(nb.ox1+1,1);
    if(pmb->block_size.nx2==1) njs=0, nje=0;
    else njs=std::max(nb.ox2-1,-1), nje=std::min(nb.ox2+1,1);
    if(pmb->block_size.nx3==1) nks=0, nke=0;
    else nks=std::max(nb.ox3-1,-1), nke=std::min(nb.ox3+1,1);
    for(int nk=nks; nk<=nke; nk++) {
      for(int nj=njs; nj<=nje; nj++) {
        for(int ni=nis; ni<=nie; ni++) {
          int ntype=std::abs(ni)+std::abs(nj)+std::abs(nk);
          // skip myself or coarse levels; only the same level must be restricted
          if(ntype==0 || pmb->nblevel[nk+1][nj+1][ni+1]!=mylevel) continue;

          // this neighbor block is on the same level
          // and needs to be restricted for prolongation
          int ris, rie, rjs, rje, rks, rke;
          if(ni==0) {
            ris=pmb->cis, rie=pmb->cie;
            if(nb.ox1==1) ris=pmb->cie;
            else if(nb.ox1==-1) rie=pmb->cis;
          }
          else if(ni== 1) ris=pmb->cie+1, rie=pmb->cie+1;
          else if(ni==-1) ris=pmb->cis-1, rie=pmb->cis-1;
          if(nj==0) {
            rjs=pmb->cjs, rje=pmb->cje;
            if(nb.ox2==1) rjs=pmb->cje;
            else if(nb.ox2==-1) rje=pmb->cjs;
          }
          else if(nj== 1) rjs=pmb->cje+1, rje=pmb->cje+1;
          else if(nj==-1) rjs=pmb->cjs-1, rje=pmb->cjs-1;
          if(nk==0) {
            rks=pmb->cks, rke=pmb->cke;
            if(nb.ox3==1) rks=pmb->cke;
            else if(nb.ox3==-1) rke=pmb->cks;
          }
          else if(nk== 1) rks=pmb->cke+1, rke=pmb->cke+1;
          else if(nk==-1) rks=pmb->cks-1, rke=pmb->cks-1;

          pmb->pmr->RestrictCellCenteredValues(cdst, pmr->coarse_cons_, 0, NHYDRO-1,
                                               ris, rie, rjs, rje, rks, rke);
          if (GENERAL_RELATIVITY)
            pmb->pmr->RestrictCellCenteredValues(pdst, pmr->coarse_prim_, 0, NHYDRO-1,
                                                 ris, rie, rjs, rje, rks, rke);
          if (MAGNETIC_FIELDS_ENABLED) {
            int rs=ris, re=rie+1;
            if(rs==pmb->cis   && pmb->nblevel[nk+1][nj+1][ni  ]<mylevel) rs++;
            if(re==pmb->cie+1 && pmb->nblevel[nk+1][nj+1][ni+2]<mylevel) re--;
            pmr->RestrictFieldX1(bfdst.x1f, pmr->coarse_b_.x1f, rs, re, rjs, rje, rks, rke);
            if(pmb->block_size.nx2 > 1) {
              rs=rjs, re=rje+1;
              if(rs==pmb->cjs   && pmb->nblevel[nk+1][nj  ][ni+1]<mylevel) rs++;
              if(re==pmb->cje+1 && pmb->nblevel[nk+1][nj+2][ni+1]<mylevel) re--;
              pmr->RestrictFieldX2(bfdst.x2f, pmr->coarse_b_.x2f, ris, rie, rs, re, rks, rke);
            }
            else 
              pmr->RestrictFieldX2(bfdst.x2f, pmr->coarse_b_.x2f, ris, rie, rjs, rje, rks, rke);
            if(pmb->block_size.nx3 > 1) {
              rs=rks, re=rke+1;
              if(rs==pmb->cks   && pmb->nblevel[nk  ][nj+1][ni+1]<mylevel) rs++;
              if(re==pmb->cke+1 && pmb->nblevel[nk+2][nj+1][ni+1]<mylevel) re--;
              pmr->RestrictFieldX3(bfdst.x3f, pmr->coarse_b_.x3f, ris, rie, rjs, rje, rs, re);
            }
            else
              pmr->RestrictFieldX3(bfdst.x3f, pmr->coarse_b_.x3f, ris, rie, rjs, rje, rks, rke);
          }
        }
      }
    }


    // calculate the loop limits for the ghost zones
    int cn = (NGHOST+1)/2;
    int si, ei, sj, ej, sk, ek, fsi, fei, fsj, fej, fsk, fek;
    if(nb.ox1==0) {
      si=pmb->cis, ei=pmb->cie;
      if((lx1&1L)==0L) ei++;
      else             si--;
    }
    else if(nb.ox1>0) si=pmb->cie+1,  ei=pmb->cie+cn;
    else              si=pmb->cis-cn, ei=pmb->cis-1;
    if(nb.ox2==0) {
      sj=pmb->cjs, ej=pmb->cje;
      if(pmb->block_size.nx2 > 1) {
        if((lx2&1L)==0L) ej++;
        else             sj--;
      }
    }
    else if(nb.ox2>0) sj=pmb->cje+1,  ej=pmb->cje+cn;
    else              sj=pmb->cjs-cn, ej=pmb->cjs-1;
    if(nb.ox3==0) {
      sk=pmb->cks, ek=pmb->cke;
      if(pmb->block_size.nx3 > 1) {
        if((lx3&1L)==0L) ek++;
        else             sk--;
      }
    }
    else if(nb.ox3>0) sk=pmb->cke+1,  ek=pmb->cke+cn;
    else              sk=pmb->cks-cn, ek=pmb->cks-1;

    // convert the ghost zone and ghost-ghost zones into primitive variables
    // this includes cell-centered field calculation
    int f1m=0, f1p=0, f2m=0, f2p=0, f3m=0, f3p=0;
    if(nb.ox1==0) {
      if(pmb->nblevel[1][1][0]!=-1) f1m=1;
      if(pmb->nblevel[1][1][2]!=-1) f1p=1;
    }
    else f1m=1, f1p=1;
    if(pmb->block_size.nx2>1) {
      if(nb.ox2==0) {
        if(pmb->nblevel[1][0][1]!=-1) f2m=1;
        if(pmb->nblevel[1][2][1]!=-1) f2p=1;
      }
      else f2m=1, f2p=1;
    }
    if(pmb->block_size.nx3>1) {
      if(nb.ox3==0) {
        if(pmb->nblevel[0][1][1]!=-1) f3m=1;
        if(pmb->nblevel[2][1][1]!=-1) f3p=1;
      }
      else f3m=1, f3p=1;
    }
    pmb->peos->ConservedToPrimitive(pmr->coarse_cons_, pmr->coarse_prim_,
                 pmr->coarse_b_, pmr->coarse_prim_, pmr->coarse_bcc_, pmr->pcoarsec,
                 si-f1m, ei+f1p, sj-f2m, ej+f2p, sk-f3m, ek+f3p);

    // Apply physical boundaries
    if(nb.ox1==0) {
      if(BoundaryFunction_[INNER_X1]!=NULL) {
        BoundaryFunction_[INNER_X1](pmb, pmr->pcoarsec, pmr->coarse_prim_,
                          pmr->coarse_b_, pmb->cis, pmb->cie, sj, ej, sk, ek);
      }
      if(BoundaryFunction_[OUTER_X1]!=NULL) {
        BoundaryFunction_[OUTER_X1](pmb, pmr->pcoarsec, pmr->coarse_prim_,
                          pmr->coarse_b_, pmb->cis, pmb->cie, sj, ej, sk, ek);
      }
    }
    if(nb.ox2==0 && pmb->block_size.nx2 > 1) {
      if(BoundaryFunction_[INNER_X2]!=NULL) {
        BoundaryFunction_[INNER_X2](pmb, pmr->pcoarsec, pmr->coarse_prim_,
                          pmr->coarse_b_, si, ei, pmb->cjs, pmb->cje, sk, ek);
      }
      if(BoundaryFunction_[OUTER_X2]!=NULL) {
        BoundaryFunction_[OUTER_X2](pmb, pmr->pcoarsec, pmr->coarse_prim_,
                          pmr->coarse_b_, si, ei, pmb->cjs, pmb->cje, sk, ek);
      }
    }
    if(nb.ox3==0 && pmb->block_size.nx3 > 1) {
      if(BoundaryFunction_[INNER_X3]!=NULL) {
        BoundaryFunction_[INNER_X3](pmb, pmr->pcoarsec, pmr->coarse_prim_,
                          pmr->coarse_b_, si, ei, sj, ej, pmb->cks, pmb->cke);
      }
      if(BoundaryFunction_[OUTER_X3]!=NULL) {
        BoundaryFunction_[OUTER_X3](pmb, pmr->pcoarsec, pmr->coarse_prim_,
                          pmr->coarse_b_, si, ei, sj, ej, pmb->cks, pmb->cke);
      }
    }

    // now that the ghost-ghost zones are filled
    // calculate the loop limits for the finer grid
    fsi=(si-pmb->cis)*2+pmb->is,   fei=(ei-pmb->cis)*2+pmb->is+1;
    if(pmb->block_size.nx2 > 1)
      fsj=(sj-pmb->cjs)*2+pmb->js, fej=(ej-pmb->cjs)*2+pmb->js+1;
    else fsj=pmb->js, fej=pmb->je;
    if(pmb->block_size.nx3 > 1)
      fsk=(sk-pmb->cks)*2+pmb->ks, fek=(ek-pmb->cks)*2+pmb->ks+1;
    else fsk=pmb->ks, fek=pmb->ke;

    // prolongate hydro variables using primitive
    pmr->ProlongateCellCenteredValues(pmr->coarse_prim_, pdst, 0, NHYDRO-1,
                                      si, ei, sj, ej, sk, ek);
    // prollongate magnetic fields
    if (MAGNETIC_FIELDS_ENABLED) {
      int il, iu, jl, ju, kl, ku;
      il=si, iu=ei+1;
      if((nb.ox1>=0) && (pmb->nblevel[nb.ox3+1][nb.ox2+1][nb.ox1  ]>=mylevel)) il++;
      if((nb.ox1<=0) && (pmb->nblevel[nb.ox3+1][nb.ox2+1][nb.ox1+2]>=mylevel)) iu--;
      if(pmb->block_size.nx2 > 1) {
        jl=sj, ju=ej+1;
        if((nb.ox2>=0) && (pmb->nblevel[nb.ox3+1][nb.ox2  ][nb.ox1+1]>=mylevel)) jl++;
        if((nb.ox2<=0) && (pmb->nblevel[nb.ox3+1][nb.ox2+2][nb.ox1+1]>=mylevel)) ju--;
      }
      else jl=sj, ju=ej;
      if(pmb->block_size.nx3 > 1) {
        kl=sk, ku=ek+1;
        if((nb.ox3>=0) && (pmb->nblevel[nb.ox3  ][nb.ox2+1][nb.ox1+1]>=mylevel)) kl++;
        if((nb.ox3<=0) && (pmb->nblevel[nb.ox3+2][nb.ox2+1][nb.ox1+1]>=mylevel)) ku--;
      }
      else kl=sk, ku=ek;

      // step 1. calculate x1 outer surface fields and slopes
      pmr->ProlongateSharedFieldX1(pmr->coarse_b_.x1f, bfdst.x1f, il, iu, sj, ej, sk, ek);
      // step 2. calculate x2 outer surface fields and slopes
      pmr->ProlongateSharedFieldX2(pmr->coarse_b_.x2f, bfdst.x2f, si, ei, jl, ju, sk, ek);
      // step 3. calculate x3 outer surface fields and slopes
      pmr->ProlongateSharedFieldX3(pmr->coarse_b_.x3f, bfdst.x3f, si, ei, sj, ej, kl, ku);
      // step 4. calculate the internal finer fields using the Toth & Roe method
      pmr->ProlongateInternalField(bfdst, si, ei, sj, ej, sk, ek);

      // Field prolongation completed, calculate cell centered fields
      pmb->pfield->CalculateCellCenteredField(bfdst, bcdst, pmb->pcoord,
                                              fsi, fei, fsj, fej, fsk, fek);
    }
    // calculate conservative variables
    pmb->peos->PrimitiveToConserved(pdst, bcdst, cdst, pmb->pcoord,
                                            fsi, fei, fsj, fej, fsk, fek);
  }
}
