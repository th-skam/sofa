/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_EIGENBICGSTAB_CPP
#include <sofa/component/linearsolver/iterative/EigenBiCGSTAB.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::iterative
{

using namespace sofa::linearalgebra;

void registerEigenBiCGSTAB(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Linear system solver using the stabilized bi-conjugate gradient (BiCGSTAB) of Eigen. Suited to non-symmetric systems.")
        .add< EigenBiCGSTAB<CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >());
}

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API EigenBiCGSTAB<CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;

} // namespace sofa::component::linearsolver::iterative
