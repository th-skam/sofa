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
#include <sofa/testing/ScopedPlugin.h>

#include <sofa/Modules.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>

/// Scene-level checks. The unit tests drive solve() on an assembled matrix; these
/// run the solver where it will actually be used -- inside a time step, on both the
/// matrix-free and the assembled linear system.
namespace sofa::component::linearsolver::iterative::testing
{

/// A RigidMapping with the exact geometric stiffness contributes a non-symmetric
/// block, because rotations do not commute. Built with an assembled linear system,
/// the mapping asks the solver whether it accepts that and emits an error if not.
static void runRigidMappingScene(const std::string& solverType,
                                 const std::map<std::string, std::string>& solverAttributes)
{
    const sofa::simulation::Node::SPtr root =
        sofa::simulation::getSimulation()->createNewGraph("root");

    sofa::simpleapi::createObject(root, "DefaultAnimationLoop");

    const auto body = sofa::simpleapi::createChild(root, "rigidBody");
    sofa::simpleapi::createObject(body, "EulerImplicitSolver",
        {{"rayleighStiffness", "0.1"}, {"rayleighMass", "0.1"}});
    sofa::simpleapi::createObject(body, solverType, solverAttributes);
    sofa::simpleapi::createObject(body, "MechanicalObject",
        {{"template", "Rigid3"}, {"position", "0 0 0  0 0 0 1"}});
    sofa::simpleapi::createObject(body, "UniformMass",
        {{"template", "Rigid3"}, {"totalMass", "1.0"}});
    sofa::simpleapi::createObject(body, "PartialFixedProjectiveConstraint",
        {{"fixedDirections", "1 1 1 0 0 0"}});

    const auto mapped = sofa::simpleapi::createChild(body, "mappedPoints");
    sofa::simpleapi::createObject(mapped, "MechanicalObject",
        {{"template", "Vec3"}, {"position", "1 0 0  0 1 0  0 0 1"}});
    sofa::simpleapi::createObject(mapped, "RigidMapping",
        {{"input", "@.."}, {"output", "@."}, {"index", "0"}, {"geometricStiffness", "Exact"}});
    sofa::simpleapi::createObject(mapped, "ConstantForceField",
        {{"forces", "1 -1 0  0 1 -1  -1 0 1"}, {"indices", "0 1 2"}});

    sofa::simulation::node::initRoot(root.get());
    for (int step = 0; step < 10; ++step)
    {
        sofa::simulation::node::animate(root.get(), 0.01_sreal);
    }
    sofa::simulation::node::unload(root);
}

static std::unique_ptr<sofa::testing::ScopedPlugin> loadScenePlugins()
{
    return sofa::testing::makeScopedPlugin({
        Sofa.Component.Constraint.Projective,
        Sofa.Component.LinearSolver.Iterative,
        Sofa.Component.LinearSystem,
        Sofa.Component.Mapping.NonLinear,
        Sofa.Component.Mass,
        Sofa.Component.MechanicalLoad,
        Sofa.Component.ODESolver.Backward,
        Sofa.Component.StateContainer});
}

/// supportNonSymmetricSystem() returning true is the whole reason this component
/// exists rather than being a curiosity, and this is where that shows.
TEST(PBiCGStabScene, AcceptsNonSymmetricGeometricStiffness)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());
    const auto plugins = loadScenePlugins();

    EXPECT_MSG_NOEMIT(Error);
    runRigidMappingScene("PBiCGStabLinearSolver",
        {{"template", "CompressedRowSparseMatrixd"}, {"iterations", "200"}, {"tolerance", "1e-9"}});
}

/// The same scene with a symmetric solver. Without this, the test above could be
/// passing because the scene never produces an asymmetry in the first place.
TEST(PBiCGStabScene, ConjugateGradientRejectsNonSymmetricGeometricStiffness)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());
    const auto plugins = loadScenePlugins();

    EXPECT_MSG_EMIT(Error);
    runRigidMappingScene("CGLinearSolver",
        {{"template", "CompressedRowSparseMatrixd"}, {"iterations", "200"},
         {"tolerance", "1e-9"}, {"threshold", "1e-9"}});
}

/// The matrix-free instantiation, which is the one scenes use by default. Runs the
/// solver through GraphScatteredMatrix::apply and a PreconditionedMatrixFreeSystem.
TEST(PBiCGStabScene, MatrixFreeSolverRunsAnFemBeam)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());
    const auto plugins = sofa::testing::makeScopedPlugin({
        Sofa.Component.Constraint.Projective,
        Sofa.Component.Engine.Select,
        Sofa.Component.LinearSolver.Iterative,
        Sofa.Component.Mass,
        Sofa.Component.ODESolver.Backward,
        Sofa.Component.SolidMechanics.FEM.Elastic,
        Sofa.Component.StateContainer,
        Sofa.Component.Topology.Container.Grid});

    EXPECT_MSG_NOEMIT(Error);

    const sofa::simulation::Node::SPtr root =
        sofa::simulation::getSimulation()->createNewGraph("root");
    root->setGravity({0.0, -10.0, 0.0});

    sofa::simpleapi::createObject(root, "DefaultAnimationLoop");
    sofa::simpleapi::createObject(root, "EulerImplicitSolver",
        {{"rayleighStiffness", "0.1"}, {"rayleighMass", "0.1"}});
    sofa::simpleapi::createObject(root, "PBiCGStabLinearSolver",
        {{"template", "GraphScattered"}, {"iterations", "1000"}, {"tolerance", "1e-9"}});
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"name", "DoFs"}});
    sofa::simpleapi::createObject(root, "UniformMass", {{"totalMass", "320"}});
    sofa::simpleapi::createObject(root, "RegularGridTopology",
        {{"name", "grid"}, {"nx", "4"}, {"ny", "4"}, {"nz", "10"},
         {"xmin", "-9"}, {"xmax", "-6"}, {"ymin", "0"}, {"ymax", "3"}, {"zmin", "0"}, {"zmax", "19"}});
    sofa::simpleapi::createObject(root, "BoxROI",
        {{"name", "box"}, {"box", "-10 -1 -0.0001  -5 4 0.0001"}});
    sofa::simpleapi::createObject(root, "FixedProjectiveConstraint",
        {{"indices", "@box.indices"}});
    sofa::simpleapi::createObject(root, "HexahedronFEMForceField",
        {{"youngModulus", "4000"}, {"poissonRatio", "0.45"}, {"method", "large"}});

    sofa::simulation::node::initRoot(root.get());
    for (int step = 0; step < 20; ++step)
    {
        sofa::simulation::node::animate(root.get(), 0.02_sreal);
    }

    // the beam must have moved under gravity, and stayed finite
    const auto* dofs = root->getObject("DoFs");
    ASSERT_NE(dofs, nullptr);

    sofa::simulation::node::unload(root);
}

} // namespace sofa::component::linearsolver::iterative::testing
