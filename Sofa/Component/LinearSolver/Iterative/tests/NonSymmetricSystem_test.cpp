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
#include <sofa/testing/BaseTest.h>

#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>
#include <sofa/component/linearsolver/direct/EigenSparseLU.h>
#include <sofa/component/linearsolver/ordering/NaturalOrderingMethod.h>
#include <sofa/component/linearsolver/direct/init.h>

#include "LinearSystems.h"

/// These tests do not exercise a new solver. They establish the ground the
/// non-symmetric solver is going to be judged on: that the reference systems are
/// really stored non-symmetrically, that a symmetric method fails on them, and
/// that a non-symmetric direct solver produces the answer the iterative one will
/// be compared against.
namespace sofa::component::linearsolver::iterative::testing
{

/// CompressedRowSparseMatrix<T> resolves to CompressedRowSparseMatrixMechanical
/// with CRSMechanicalPolicy, whose IsAlwaysSymmetric flag is true. That flag makes
/// taddMulTranspose silently return A v instead of A^T v, so it is worth proving
/// it does not also fold the two triangles together on insertion -- everything
/// downstream assumes A(i,j) and A(j,i) are independent storage.
TEST(NonSymmetricSystem, StorageKeepsBothTriangles)
{
    Matrix A;
    Vector x, b;
    makeConvectionDiffusion(8, 0.9, A, x, b);

    EXPECT_NEAR(A(0, 1), -0.1, 1e-12);
    EXPECT_NEAR(A(1, 0), -1.9, 1e-12);
    EXPECT_NE(A(0, 1), A(1, 0));
}

/// The product must use the stored asymmetry, not a symmetrised view of it.
/// Checked against a hand-evaluated row of the operator.
TEST(NonSymmetricSystem, ProductUsesTheAsymmetry)
{
    Matrix A;
    Vector x, b;
    makeConvectionDiffusion(3, 0.9, A, x, b);

    // row 1 of A is (-1.9, 2.0, -0.1), x = (1, 4/3, 5/3)
    const SReal expected = -1.9 * x[0] + 2.0 * x[1] - 0.1 * x[2];
    EXPECT_NEAR(b[1], expected, 1e-12);
}

/// EigenSparseLU is one of the three components declaring
/// supportNonSymmetricSystem(); it is the oracle every iterative result is
/// compared against.
TEST(NonSymmetricSystem, DirectLuSolvesIt)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(50, 0.9, A, x, b);

    // the Eigen solver factory only knows its ordering methods once the module
    // has registered them, and nothing else in this test loads the module
    sofa::component::linearsolver::direct::init();

    using Solver = sofa::component::linearsolver::direct::EigenSparseLU<SReal>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    using NaturalOrdering = sofa::component::linearsolver::ordering::NaturalOrderingMethod;
    const NaturalOrdering::SPtr ordering = sofa::core::objectmodel::New<NaturalOrdering>();
    solver->l_orderingMethod.set(ordering.get());

    solver->init();

    Vector solution(b.size());
    solution.clear();
    solver->invert(A);
    solver->solve(A, solution, b);

    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-10);
    EXPECT_LT(relativeResidual(A, solution, b), 1e-12);
}

/// The reason this branch exists. CG assumes A = A^T when it forms its search
/// directions, so on a strongly non-symmetric operator it does not converge no
/// matter how many iterations it is given.
TEST(NonSymmetricSystem, ConjugateGradientFailsOnIt)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(50, 0.9, A, x, b);

    using Solver = CGLinearSolver<Matrix, Vector>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    solver->d_maxIter.setValue(1000);
    solver->d_tolerance.setValue(1e-12);
    solver->d_smallDenominatorThreshold.setValue(1e-30);
    solver->init();

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    const SReal residual = relativeResidual(A, solution, b);
    msg_info("NonSymmetricSystem") << "CG relative residual after 1000 iterations: " << residual;
    EXPECT_GT(residual, 1e-2) << "CG unexpectedly solved a non-symmetric system; "
                                 "raise beta in makeVariableConvectionDiffusion";
}

/// Same solver, same size, symmetric operator: CG converges. This is what makes
/// the previous test a statement about symmetry rather than about the mesh size,
/// the tolerance or the iteration budget.
TEST(NonSymmetricSystem, ConjugateGradientSucceedsWhenSymmetric)
{
    Matrix A;
    Vector x, b;
    makeSymmetricPositiveDefinite(50, A, x, b);

    using Solver = CGLinearSolver<Matrix, Vector>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    solver->d_maxIter.setValue(1000);
    solver->d_tolerance.setValue(1e-12);
    solver->d_smallDenominatorThreshold.setValue(1e-30);
    solver->init();

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    EXPECT_LT(relativeResidual(A, solution, b), 1e-9);
    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-9);
}

} // namespace sofa::component::linearsolver::iterative::testing
