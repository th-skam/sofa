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

#include "LinearSystems.h"

#include <cmath>

/// BiCGSTAB has three scalar denominators that can reach zero on a system it cannot
/// handle: rho = rhat.r, sigma = rhat.(A M⁻¹p), and t.t. Dividing by any of them
/// once it has underflowed produces infinities and then NaNs that propagate into
/// the mechanical state, so each one is guarded. These tests check that the guards
/// fire, say which quantity broke down, and leave a finite answer behind.
namespace sofa::component::linearsolver::iterative::testing
{

using Solver = PBiCGStabLinearSolver<Matrix, Vector>;

static bool isFinite(const Vector& v)
{
    for (Vector::Index i = 0; i < v.size(); ++i)
    {
        if (!std::isfinite(v[i]))
        {
            return false;
        }
    }
    return true;
}

/// A strongly skew-dominant operator is the case BiCGSTAB cannot handle with the
/// usual shadow residual: as gamma grows, rhat.A rhat tends to zero because a skew
/// matrix has v.Av = 0 identically. The solver must stop and say so rather than
/// spin to maxIter producing garbage.
TEST(PBiCGStabBreakdown, SkewDominantSystemBreaksDownCleanly)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());

    Matrix A;
    Vector x, b;
    makeSkewDominant(30, 1000.0, A, x, b);

    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    solver->d_maxIter.setValue(2000);
    solver->d_tolerance.setValue(1e-12);
    solver->init();

    Vector solution(b.size());
    solution.clear();

    {
        EXPECT_MSG_EMIT(Warning);
        solver->solve(A, solution, b);
    }

    const unsigned iterations =
        static_cast<unsigned>(solver->d_graph.getValue().at("Error 1").size()) - 1u;

    msg_info("PBiCGStabBreakdown") << "skew-dominant system stopped after " << iterations
                                   << " iterations of a 2000 budget";

    EXPECT_LT(iterations, 2000u) << "the guard did not fire; the solver ran out its budget instead";
    EXPECT_TRUE(isFinite(solution)) << "the returned solution contains inf or NaN";
}

/// The threshold is a Data, so raising it to something absurd must turn the very
/// first iteration into a breakdown on any system at all. This tests the guard
/// itself rather than a system that happens to trigger it.
TEST(PBiCGStabBreakdown, AbsurdThresholdBreaksDownImmediately)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());

    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(30, 0.9, A, x, b);

    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    solver->d_maxIter.setValue(100);
    solver->d_tolerance.setValue(1e-12);
    solver->d_threshold.setValue(1e30);
    solver->init();

    Vector solution(b.size());
    solution.clear();

    {
        EXPECT_MSG_EMIT(Warning);
        solver->solve(A, solution, b);
    }

    EXPECT_EQ(solver->d_graph.getValue().at("Error 1").size(), 1u)
        << "the breakdown should have happened before the first residual update";
    EXPECT_TRUE(isFinite(solution));
}

/// A singular operator. Every method fails here; the requirement is only that the
/// failure is finite and reported.
TEST(PBiCGStabBreakdown, SingularSystemDoesNotProduceNaN)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());

    // two identical rows: A is singular by construction
    Matrix A;
    A.resize(4, 4);
    A.add(0, 0, 1.0); A.add(0, 1, 2.0);
    A.add(1, 0, 1.0); A.add(1, 1, 2.0);
    A.add(2, 2, 3.0); A.add(2, 3, 1.0);
    A.add(3, 2, 1.0); A.add(3, 3, 4.0);
    A.compress();

    Vector b(4);
    b[0] = 1.0; b[1] = 5.0; b[2] = 1.0; b[3] = 1.0;   // inconsistent: rows 0 and 1 disagree

    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    solver->d_maxIter.setValue(200);
    solver->d_tolerance.setValue(1e-12);
    solver->init();

    Vector solution(4);
    solution.clear();
    solver->solve(A, solution, b);

    EXPECT_TRUE(isFinite(solution)) << "a singular system produced inf or NaN";
}

} // namespace sofa::component::linearsolver::iterative::testing
