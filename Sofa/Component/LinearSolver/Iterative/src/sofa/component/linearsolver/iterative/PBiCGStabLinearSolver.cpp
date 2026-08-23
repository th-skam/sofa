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
#define SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_PBICGSTABLINEARSOLVER_CPP
#include <sofa/component/linearsolver/iterative/PBiCGStabLinearSolver.inl>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/linearsystem/MatrixFreeSystem.h>
#include <sofa/component/linearsolver/iterative/PreconditionedMatrixFreeSystem.h>

#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/FullVector.h>

namespace sofa::component::linearsolver::iterative
{

using namespace sofa::linearalgebra;
using sofa::type::Mat;

/// Matrix-free path: the vectors are addressed by MultiVecDerivId, so the
/// preconditioner can be pointed straight at them and no values are copied.
template<>
void PBiCGStabLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::applyPreconditioner(Vector& out, Vector& in)
{
    auto* system = l_preconditioner->getLinearSystem();
    system->setSystemSolution(out);
    system->setRHS(in);
    l_preconditioner->solveSystem();
    system->dispatchSystemSolution(out);
}

/// The matrix-free instantiation needs a PreconditionedMatrixFreeSystem rather than
/// the plain MatrixFreeSystem the base class would create: that is the component
/// which assembles the preconditioner's matrix, on its own assemblingRate schedule.
template<>
void PBiCGStabLinearSolver<GraphScatteredMatrix, GraphScatteredVector>::checkLinearSystem()
{
    this->doCheckLinearSystem<PreconditionedMatrixFreeSystem<GraphScatteredMatrix, GraphScatteredVector> >();
}

template<>
void PBiCGStabLinearSolver<GraphScatteredMatrix, GraphScatteredVector>::ensureRequiredLinearSystemType()
{
    if (this->l_linearSystem)
    {
        if (!dynamic_cast<PreconditionedMatrixFreeSystem<GraphScatteredMatrix, GraphScatteredVector>*>(this->l_linearSystem.get()))
        {
            msg_error() << "This linear solver is designed to work with a "
                        << PreconditionedMatrixFreeSystem<GraphScatteredMatrix, GraphScatteredVector>::GetClass()->className
                        << " linear system, but a " << this->l_linearSystem->getClassName()
                        << " was found";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }
}

template<>
void PBiCGStabLinearSolver<GraphScatteredMatrix, GraphScatteredVector>::bwdInit()
{
    if (this->isComponentStateInvalid())
        return;

    if (!l_preconditioner || !this->l_linearSystem)
        return;

    auto* preconditionerLinearSystem = l_preconditioner->getLinearSystem();
    if (!preconditionerLinearSystem)
        return;

    auto* preconditionedSystem =
        dynamic_cast<PreconditionedMatrixFreeSystem<GraphScatteredMatrix, GraphScatteredVector>*>(this->l_linearSystem.get());
    if (!preconditionedSystem)
    {
        // ensureRequiredLinearSystemType() already invalidated this case at init(),
        // so getting here means the system was swapped afterwards
        msg_error() << "The linear system (" << this->l_linearSystem->getPathName()
                    << ") is not a PreconditionedMatrixFreeSystem";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "Linking the preconditioner linear system (" << preconditionerLinearSystem->getPathName()
               << ") to the BiCGSTAB linear system (" << preconditionedSystem->getPathName() << ")";
    // this link is what makes the preconditioner matrix get assembled at all
    preconditionedSystem->l_preconditionerSystem.set(preconditionerLinearSystem);
}

void registerPBiCGStabLinearSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Linear solver using the preconditioned stabilized bi-conjugate gradient iterative algorithm (BiCGSTAB), for systems that are not symmetric.")
        .add< PBiCGStabLinearSolver<GraphScatteredMatrix, GraphScatteredVector> >(true)
        .add< PBiCGStabLinearSolver<FullMatrix<SReal>, FullVector<SReal> > >()
        .add< PBiCGStabLinearSolver<CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >()
        .add< PBiCGStabLinearSolver<CompressedRowSparseMatrix<Mat<3,3,SReal> >, FullVector<SReal> > >());
}

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<GraphScatteredMatrix, GraphScatteredVector>;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<FullMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<CompressedRowSparseMatrix<Mat<3,3,SReal> >, FullVector<SReal> >;

} // namespace sofa::component::linearsolver::iterative
