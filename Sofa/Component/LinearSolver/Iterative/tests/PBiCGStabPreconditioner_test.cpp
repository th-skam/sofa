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

#include <sofa/component/linearsolver/iterative/PBiCGStabLinearSolver.h>
#include <sofa/component/linearsolver/direct/EigenSparseLU.h>
#include <sofa/component/linearsolver/direct/init.h>
#include <sofa/component/linearsolver/ordering/NaturalOrderingMethod.h>
#include <sofa/component/linearsolver/preconditioner/JacobiPreconditioner.h>
#include <sofa/linearalgebra/DiagonalMatrix.h>

#include "LinearSystems.h"

namespace sofa::component::linearsolver::iterative::testing
{

using Solver = PBiCGStabLinearSolver<Matrix, Vector>;

static Solver::SPtr makeSolver(unsigned iterations, SReal tolerance)
{
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    solver->d_maxIter.setValue(iterations);
    solver->d_tolerance.setValue(tolerance);
    return solver;
}

/// Number of iterations the solve actually took: the residual curve holds the
/// initial residual plus one entry per iteration.
static unsigned iterationCount(const Solver::SPtr& solver)
{
    return static_cast<unsigned>(solver->d_graph.getValue().at("Error 1").size()) - 1u;
}

/// A preconditioner driven outside a scene has to be given by hand what the scene
/// would normally assemble for it: a system of the right size, the matrix values,
/// and the signal that the matrix changed so that invert() runs once.
template<class Preconditioner, class FillMatrix>
static void primePreconditioner(const typename Preconditioner::SPtr& precond, sofa::Size n, FillMatrix fill)
{
    auto* system = precond->getLinearSystem();
    system->resizeSystem(n);
    fill(*system->getSystemMatrix());
    system->d_matrixChanged.setValue(true);
}

/// M = A exactly. BiCGSTAB then has alpha = 1 and s = r - alpha A M⁻¹ r = 0 at the
/// very first half-step, so it must return the exact solution after one iteration.
/// Nothing but a correct preconditioned recurrence does that.
TEST(PBiCGStabPreconditioner, ExactPreconditionerConvergesInOneIteration)
{
    sofa::component::linearsolver::direct::init();

    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(40, 0.9, A, x, b);

    using Preconditioner = sofa::component::linearsolver::direct::EigenSparseLU<SReal>;
    const Preconditioner::SPtr precond = sofa::core::objectmodel::New<Preconditioner>();
    const auto ordering = sofa::core::objectmodel::New<sofa::component::linearsolver::ordering::NaturalOrderingMethod>();
    precond->l_orderingMethod.set(ordering.get());
    precond->init();
    primePreconditioner<Preconditioner>(precond, 40, [&A](Matrix& M) { M = A; });

    const Solver::SPtr solver = makeSolver(200, 1e-10);
    solver->l_preconditioner.set(precond.get());
    solver->init();
    solver->bwdInit();
    ASSERT_FALSE(solver->isComponentStateInvalid());

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    EXPECT_EQ(iterationCount(solver), 1u);
    EXPECT_LT(relativeResidual(A, solution, b), 1e-12);
    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-10);
}

/// The real JacobiPreconditioner, on a system whose diagonal spans eight orders of
/// magnitude. It must reach the same solution as the unpreconditioned solve and
/// take strictly fewer iterations to get there.
TEST(PBiCGStabPreconditioner, JacobiReducesIterationCount)
{
    Matrix A;
    Vector x, b;
    makeBadlyScaledConvectionDiffusion(40, 0.9, 4.0, A, x, b);

    const Solver::SPtr plain = makeSolver(2000, 1e-10);
    plain->init();
    Vector plainSolution(b.size());
    plainSolution.clear();
    plain->solve(A, plainSolution, b);
    const unsigned plainIterations = iterationCount(plain);

    using Preconditioner = sofa::component::linearsolver::preconditioner::JacobiPreconditioner<
        sofa::linearalgebra::DiagonalMatrix<SReal>, Vector>;
    const Preconditioner::SPtr precond = sofa::core::objectmodel::New<Preconditioner>();
    precond->init();
    primePreconditioner<Preconditioner>(precond, 40,
        [&A](sofa::linearalgebra::DiagonalMatrix<SReal>& M)
        {
            for (sofa::Index i = 0; i < 40; ++i)
            {
                M.set(i, i, A(i, i));
            }
        });

    const Solver::SPtr preconditioned = makeSolver(2000, 1e-10);
    preconditioned->l_preconditioner.set(precond.get());
    preconditioned->init();
    preconditioned->bwdInit();
    ASSERT_FALSE(preconditioned->isComponentStateInvalid());

    Vector preconditionedSolution(b.size());
    preconditionedSolution.clear();
    preconditioned->solve(A, preconditionedSolution, b);
    const unsigned preconditionedIterations = iterationCount(preconditioned);

    msg_info("PBiCGStabPreconditioner")
        << "badly scaled system: " << plainIterations << " iterations unpreconditioned, "
        << preconditionedIterations << " with Jacobi";

    EXPECT_LT(relativeResidual(A, preconditionedSolution, b), 1e-10);
    EXPECT_LT(preconditionedIterations, plainIterations);

    // The solution error is held to a looser bound than elsewhere on purpose. The
    // rows of this system span eight orders of magnitude, so |r|/|b| is dominated
    // by the largest rows and says little about how well the smallest ones are
    // resolved. That is a property of a residual-based stopping test on a badly
    // scaled system, not of the solver: a tighter tolerance here buys accuracy in
    // the rows that already had it.
    EXPECT_LT(maxAbsoluteDifference(preconditionedSolution, x), 1e-6);
}

/// Preconditioning changes the path, not the destination.
TEST(PBiCGStabPreconditioner, PreconditionedMatchesUnpreconditionedSolution)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(40, 0.9, A, x, b);

    const Solver::SPtr plain = makeSolver(400, 1e-12);
    plain->init();
    Vector plainSolution(b.size());
    plainSolution.clear();
    plain->solve(A, plainSolution, b);

    using Preconditioner = sofa::component::linearsolver::preconditioner::JacobiPreconditioner<
        sofa::linearalgebra::DiagonalMatrix<SReal>, Vector>;
    const Preconditioner::SPtr precond = sofa::core::objectmodel::New<Preconditioner>();
    precond->init();
    primePreconditioner<Preconditioner>(precond, 40,
        [&A](sofa::linearalgebra::DiagonalMatrix<SReal>& M)
        {
            for (sofa::Index i = 0; i < 40; ++i)
            {
                M.set(i, i, A(i, i));
            }
        });

    const Solver::SPtr preconditioned = makeSolver(400, 1e-12);
    preconditioned->l_preconditioner.set(precond.get());
    preconditioned->init();
    preconditioned->bwdInit();

    Vector preconditionedSolution(b.size());
    preconditionedSolution.clear();
    preconditioned->solve(A, preconditionedSolution, b);

    EXPECT_LT(maxAbsoluteDifference(preconditionedSolution, plainSolution), 1e-9);
}

/// use_precond turns the preconditioner off without unlinking it, so the two must
/// still agree on the answer.
TEST(PBiCGStabPreconditioner, UsePrecondFalseIgnoresTheLink)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(40, 0.9, A, x, b);

    using Preconditioner = sofa::component::linearsolver::preconditioner::JacobiPreconditioner<
        sofa::linearalgebra::DiagonalMatrix<SReal>, Vector>;
    const Preconditioner::SPtr precond = sofa::core::objectmodel::New<Preconditioner>();
    precond->init();
    primePreconditioner<Preconditioner>(precond, 40,
        [&A](sofa::linearalgebra::DiagonalMatrix<SReal>& M)
        {
            for (sofa::Index i = 0; i < 40; ++i)
            {
                M.set(i, i, A(i, i));
            }
        });

    const Solver::SPtr solver = makeSolver(400, 1e-12);
    solver->l_preconditioner.set(precond.get());
    solver->d_use_precond.setValue(false);
    solver->init();
    solver->bwdInit();

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-9);
}

/// A link written in a scene but pointing nowhere must be reported, not silently
/// degraded into an unpreconditioned solve.
TEST(PBiCGStabPreconditioner, UnresolvedPreconditionerLinkIsInvalid)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());

    const Solver::SPtr solver = makeSolver(400, 1e-12);
    solver->l_preconditioner.setPath("@/thereIsNoSuchComponent");

    {
        EXPECT_MSG_EMIT(Error);
        solver->init();
    }

    EXPECT_TRUE(solver->isComponentStateInvalid());
}

} // namespace sofa::component::linearsolver::iterative::testing
