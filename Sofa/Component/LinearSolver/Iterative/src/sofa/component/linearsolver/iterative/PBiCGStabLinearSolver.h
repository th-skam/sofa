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
#pragma once
#include <sofa/component/linearsolver/iterative/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/helper/map.h>

#include <cmath>

namespace sofa::component::linearsolver::iterative
{

/// Linear system solver using the preconditioned stabilized bi-conjugate gradient
/// iterative algorithm (BiCGSTAB, van der Vorst 1992).
///
/// Unlike the conjugate gradient it does not assume A = Aᵗ, so it is the solver to
/// use when the tangent operator is genuinely non-symmetric: a RigidMapping with
/// exact geometric stiffness, a follower load such as SurfacePressureForceField
/// with its tangent stiffness enabled, or a consistent corotational tangent.
///
/// The method is transpose-free by construction, which is what makes it usable on
/// SOFA's matrix-free operator: GraphScatteredMatrix can only apply A, never Aᵗ.
/// Each iteration costs two applications of A and two of the preconditioner.
///
/// Any LinearSolver that assembles a matrix can serve as the preconditioner, but
/// two of the ones in Sofa.Component.LinearSolver.Preconditioner assume the system
/// is symmetric and will simply precondition badly here: SSORPreconditioner reads
/// only the lower triangle of A for both of its sweeps, and
/// PrecomputedWarpPreconditioner factorises with a Cholesky decomposition.
/// JacobiPreconditioner and BlockJacobiPreconditioner invert diagonal blocks and
/// are indifferent to symmetry; EigenSparseLU is the natural direct preconditioner.
/// None of these choices can make the result wrong -- the convergence test is on
/// the residual of A itself -- only slow.
template<class TMatrix, class TVector>
class PBiCGStabLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{

public:

    SOFA_CLASS(
        SOFA_TEMPLATE2(PBiCGStabLinearSolver,TMatrix,TVector),
        SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, TMatrix, TVector));

    using Matrix = TMatrix;
    using Vector = TVector;
    using Real = typename Matrix::Real;
    using Inherit = sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>;

    Data<unsigned> d_maxIter; ///< Maximum number of iterations after which the iterative descent of BiCGSTAB must stop
    Data<Real> d_tolerance; ///< Desired accuracy of the solution, evaluating |r|/|b| (ratio of current residual norm over right-hand side norm)
    Data<Real> d_threshold; ///< Floor below which a scalar denominator of the recurrence is treated as a breakdown rather than divided by
    Data<bool> d_warmStart; ///< Use the previous solution as the initial estimate, instead of starting from zero
    Data<bool> d_use_precond; ///< Use a preconditioner
    SingleLink<PBiCGStabLinearSolver, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_preconditioner; ///< Link towards the linear solver used to precondition BiCGSTAB
    Data<std::map < std::string, sofa::type::vector<Real> > > d_graph; ///< Graph of residuals at each iteration

    void solve (Matrix& M, Vector& x, Vector& b) override;
    void init() override;
    void bwdInit() override;

    /// This is the whole point of the component: it tells the assembly, through
    /// MechanicalParams::setSupportOnlySymmetricMatrix, that force fields and
    /// mappings may contribute a non-symmetric tangent without symmetrising it.
    bool supportNonSymmetricSystem() const override { return true; }

protected:
    PBiCGStabLinearSolver();

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    /// Applies M⁻¹, writing M⁻¹·in into out. Separated from solve() because the
    /// linear-system interface it has to go through differs per vector type: the
    /// matrix-free vectors are addressed by MultiVecDerivId and can be handed to
    /// the preconditioner without a copy, assembled ones cannot.
    void applyPreconditioner(Vector& out, Vector& in);

    /// Only the matrix-free instantiation needs a linear system of its own kind: a
    /// PreconditionedMatrixFreeSystem, which assembles the preconditioner's matrix
    /// on its behalf and on its own schedule. The assembled instantiations keep
    /// whichever system the base class picks.
    void checkLinearSystem() override;
    void ensureRequiredLinearSystemType();

    /// Number of the current Newton iteration within the time step, used only to
    /// key the residual curves in d_graph
    int m_newtonIteration {0};
};

template<>
void PBiCGStabLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::applyPreconditioner(Vector& out, Vector& in);

template<>
void PBiCGStabLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::checkLinearSystem();

template<>
void PBiCGStabLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::ensureRequiredLinearSystemType();

template<>
void PBiCGStabLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::bwdInit();

#if !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_PBICGSTABLINEARSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<GraphScatteredMatrix, GraphScatteredVector>;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<sofa::linearalgebra::FullMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<sofa::linearalgebra::CompressedRowSparseMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PBiCGStabLinearSolver<sofa::linearalgebra::CompressedRowSparseMatrix<sofa::type::Mat<3,3,SReal> >, sofa::linearalgebra::FullVector<SReal> >;
#endif

} // namespace sofa::component::linearsolver::iterative
