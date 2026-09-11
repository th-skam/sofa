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
#include <gtest/gtest.h>
#include <sofa/fem/FiniteElement[all].h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace sofa
{

/**
 * Checks that the shape function i at the node j is equal to delta_ij
 */
template <class ElementType, class DataTypes>
void testShapeFunctionsAtNodes()
{
    using FE = sofa::fem::FiniteElement<ElementType, DataTypes>;

    for (unsigned int j = 0; j < FE::NumberOfNodesInElement; ++j)
    {
        const auto& node_j = FE::referenceElementNodes[j];
        const auto N = FE::shapeFunctions(node_j);

        for (unsigned int i = 0; i < FE::NumberOfNodesInElement; ++i)
        {
            if (i == j)
                EXPECT_NEAR(N[i], 1.0, 1e-12) << "Shape function " << i << " should be 1 at node " << j;
            else
                EXPECT_NEAR(N[i], 0.0, 1e-12) << "Shape function " << i << " should be 0 at node " << j;
        }
    }
}
/**
 * Checks that the sum of the gradients of shape functions is zero at the evaluation point.
 */
template <class ElementType, class DataTypes>
void testGradientShapeFunctions(const sofa::type::Vec<sofa::fem::FiniteElement<ElementType, DataTypes>::TopologicalDimension, sofa::Real_t<DataTypes>>& evaluationPoint)
{
    using FE = sofa::fem::FiniteElement<ElementType, DataTypes>;
    static constexpr sofa::type::Vec<FE::TopologicalDimension, sofa::Real_t<DataTypes>> zero;

    const auto N = FE::gradientShapeFunctions(evaluationPoint);

    sofa::type::Vec<FE::NumberOfNodesInElement, SReal> ones;
    std::fill(ones.begin(), ones.end(), 1);

    const auto sum = N.transposed() * ones; //compute the sum of the gradients of all shape functions
    EXPECT_EQ(sum, zero);
}
// Partition of unity at arbitrary (non-node) points, not just at nodes
template <class ElementType, class DataTypes>
void testPartitionOfUnity()
{
    using FE = sofa::fem::FiniteElement<ElementType, DataTypes>;
    using Real = sofa::Real_t<DataTypes>;

    for (const auto& [q, w] : FE::quadraturePoints())
    {
        const auto N = FE::shapeFunctions(q);
        Real sum = 0;
        for (unsigned int i = 0; i < FE::NumberOfNodesInElement; ++i)
            sum += N[i];
        EXPECT_NEAR(sum, Real(1), 1e-10) << "Shape functions must sum to 1 everywhere";
    }
}
// Finite-difference cross-check: analytical gradient vs numerical gradient
//    Catches algebra/sign errors without needing hand-derived expected values.
template <class ElementType, class DataTypes>
void testGradientMatchesFiniteDifference(
   const sofa::type::Vec<sofa::fem::FiniteElement<ElementType, DataTypes>::TopologicalDimension, sofa::Real_t<DataTypes>>& q)
{
    using FE = sofa::fem::FiniteElement<ElementType, DataTypes>;
    using Real = sofa::Real_t<DataTypes>;
    constexpr Real h = 1e-6;

    const auto analyticalGrad = FE::gradientShapeFunctions(q);

    for (unsigned int d = 0; d < FE::TopologicalDimension; ++d)
    {
        auto qPlus = q, qMinus = q;
        qPlus[d]  += h;
        qMinus[d] -= h;

        const auto Nplus  = FE::shapeFunctions(qPlus);
        const auto Nminus = FE::shapeFunctions(qMinus);

        for (unsigned int i = 0; i < FE::NumberOfNodesInElement; ++i)
        {
            const Real fd = (Nplus[i] - Nminus[i]) / (2 * h);
            EXPECT_NEAR(analyticalGrad[i][d], fd, 1e-5)
                << "Gradient mismatch for node " << i << ", direction " << d;
        }
    }
}
/**
 * The quadrature every element ships with: the degree it asks for when none is given, and the
 * number of points that degree resolves to in the tables of its reference domain.
 *
 * This is the reference table of the default behaviour of the library, hardcoded on purpose.
 * Every scene already using these elements integrates with exactly these points, so changing
 * one of them is a system-wide change, and making it means editing this table as well as the
 * element. That is the intent: the edit here is what forces the change to be deliberate.
 */
template <class ElementType>
struct DefaultQuadrature;

template <> struct DefaultQuadrature<sofa::geometry::Edge>
{ static constexpr sofa::Size degree = 1, points = 1; };
template <> struct DefaultQuadrature<sofa::geometry::Triangle>
{ static constexpr sofa::Size degree = 1, points = 1; };
template <> struct DefaultQuadrature<sofa::geometry::Quad>
{ static constexpr sofa::Size degree = 2, points = 3; };
template <> struct DefaultQuadrature<sofa::geometry::Tetrahedron>
{ static constexpr sofa::Size degree = 1, points = 1; };
template <> struct DefaultQuadrature<sofa::geometry::Hexahedron>
{ static constexpr sofa::Size degree = 3, points = 8; };
template <> struct DefaultQuadrature<sofa::geometry::Prism>
{ static constexpr sofa::Size degree = 1, points = 2; };
template <> struct DefaultQuadrature<sofa::geometry::Pyramid>
{ static constexpr sofa::Size degree = 1, points = 8; };
template <> struct DefaultQuadrature<sofa::geometry::QuadraticEdge>
{ static constexpr sofa::Size degree = 3, points = 2; };
template <> struct DefaultQuadrature<sofa::geometry::QuadraticTriangle>
{ static constexpr sofa::Size degree = 2, points = 3; };
template <> struct DefaultQuadrature<sofa::geometry::QuadraticQuad>
{ static constexpr sofa::Size degree = 3, points = 4; };
template <> struct DefaultQuadrature<sofa::geometry::QuadraticTetrahedron>
{ static constexpr sofa::Size degree = 2, points = 4; };
template <> struct DefaultQuadrature<sofa::geometry::QuadraticHexahedron>
{ static constexpr sofa::Size degree = 5, points = 27; };

/**
 * Where the gradients of an element are evaluated.
 *
 * The values are exactly representable in binary, so that the sum of the gradients comes out as
 * exactly zero rather than zero to within rounding: testGradientShapeFunctions compares it with
 * EXPECT_EQ. A point outside the reference domain is deliberate where it appears, the gradients
 * being defined there all the same.
 */
template <class ElementType>
struct EvaluationPoints;

template <> struct EvaluationPoints<sofa::geometry::Edge>
{ static constexpr std::array points { sofa::type::Vec1(0.), sofa::type::Vec1(1.), sofa::type::Vec1(-1.) }; };
template <> struct EvaluationPoints<sofa::geometry::Triangle>
{ static constexpr std::array points { sofa::type::Vec2(0., 0.), sofa::type::Vec2(1., 1.) }; };
template <> struct EvaluationPoints<sofa::geometry::Quad>
{ static constexpr std::array points { sofa::type::Vec2(0., 0.), sofa::type::Vec2(1., 1.) }; };
template <> struct EvaluationPoints<sofa::geometry::Tetrahedron>
{ static constexpr std::array points { sofa::type::Vec3(0., 0., 0.), sofa::type::Vec3(1., 1., 1.) }; };
template <> struct EvaluationPoints<sofa::geometry::Hexahedron>
{ static constexpr std::array points { sofa::type::Vec3(0., 0., 0.), sofa::type::Vec3(1., 1., 1.) }; };
template <> struct EvaluationPoints<sofa::geometry::Prism>
{ static constexpr std::array points { sofa::type::Vec3(0., 0., 0.), sofa::type::Vec3(1., 1., 1.) }; };
template <> struct EvaluationPoints<sofa::geometry::Pyramid>
{ static constexpr std::array points { sofa::type::Vec3(0., 0., 0.), sofa::type::Vec3(0.5, 0.5, 0.5) }; };
template <> struct EvaluationPoints<sofa::geometry::QuadraticEdge>
{ static constexpr std::array points { sofa::type::Vec1(0.), sofa::type::Vec1(1.), sofa::type::Vec1(-0.5) }; };
template <> struct EvaluationPoints<sofa::geometry::QuadraticTriangle>
{ static constexpr std::array points { sofa::type::Vec2(0., 0.), sofa::type::Vec2(0.5, 0.25) }; };
template <> struct EvaluationPoints<sofa::geometry::QuadraticQuad>
{ static constexpr std::array points { sofa::type::Vec2(0., 0.), sofa::type::Vec2(0.5, -0.5) }; };
template <> struct EvaluationPoints<sofa::geometry::QuadraticTetrahedron>
{ static constexpr std::array points { sofa::type::Vec3(0., 0., 0.), sofa::type::Vec3(0.25, 0.25, 0.25) }; };
template <> struct EvaluationPoints<sofa::geometry::QuadraticHexahedron>
{ static constexpr std::array points { sofa::type::Vec3(0., 0., 0.), sofa::type::Vec3(1., 1., 1.) }; };

/// One element under test: the topological element and the data types it is instantiated with.
template <class Element, class Data>
struct FiniteElementCase
{
    using ElementType = Element;
    using DataTypes = Data;
};

/// Names an instantiation after the element and the dimension it lives in, so that the suite
/// reports "FiniteElementTest/QuadraticTriangle3d.partitionOfUnity" rather than a bare index.
struct FiniteElementCaseName
{
    template <class Case>
    static std::string GetName(int)
    {
        std::string name = sofa::geometry::elementTypeToString(Case::ElementType::Element_type);
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
        return name + std::to_string(Case::DataTypes::spatial_dimensions) + "d";
    }
};

template <class Case>
class FiniteElementTest : public ::testing::Test {};

using AllFiniteElements = ::testing::Types<
    FiniteElementCase<sofa::geometry::Edge,                 sofa::defaulttype::Vec1Types>,
    FiniteElementCase<sofa::geometry::Edge,                 sofa::defaulttype::Vec2Types>,
    FiniteElementCase<sofa::geometry::Edge,                 sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::Triangle,             sofa::defaulttype::Vec2Types>,
    FiniteElementCase<sofa::geometry::Triangle,             sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::Quad,                 sofa::defaulttype::Vec2Types>,
    FiniteElementCase<sofa::geometry::Quad,                 sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::Tetrahedron,          sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::Hexahedron,           sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::Prism,                sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::Pyramid,              sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::QuadraticEdge,        sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::QuadraticTriangle,    sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::QuadraticQuad,        sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::QuadraticTetrahedron, sofa::defaulttype::Vec3Types>,
    FiniteElementCase<sofa::geometry::QuadraticHexahedron,  sofa::defaulttype::Vec3Types>>;

TYPED_TEST_SUITE(FiniteElementTest, AllFiniteElements, FiniteElementCaseName);

TYPED_TEST(FiniteElementTest, shapeFunctionsAtNodes)
{
    testShapeFunctionsAtNodes<typename TypeParam::ElementType, typename TypeParam::DataTypes>();
}

TYPED_TEST(FiniteElementTest, partitionOfUnity)
{
    testPartitionOfUnity<typename TypeParam::ElementType, typename TypeParam::DataTypes>();
}

TYPED_TEST(FiniteElementTest, gradientShapeFunctions)
{
    using ElementType = typename TypeParam::ElementType;
    for (const auto& evaluationPoint : EvaluationPoints<ElementType>::points)
    {
        testGradientShapeFunctions<ElementType, typename TypeParam::DataTypes>(evaluationPoint);
    }
}

TYPED_TEST(FiniteElementTest, gradientMatchesFiniteDifference)
{
    using ElementType = typename TypeParam::ElementType;
    for (const auto& evaluationPoint : EvaluationPoints<ElementType>::points)
    {
        testGradientMatchesFiniteDifference<ElementType, typename TypeParam::DataTypes>(evaluationPoint);
    }
}

/// The default quadrature of every element, checked against the table above. The degree is the
/// element's own choice and the number of points is what the shared tables of its reference
/// domain resolve that choice to; both are part of what a scene gets without asking for
/// anything, so both are pinned here.
TYPED_TEST(FiniteElementTest, integratesWithTheExpectedQuadratureByDefault)
{
    using ElementType = typename TypeParam::ElementType;
    using FE = sofa::fem::FiniteElement<ElementType, typename TypeParam::DataTypes>;
    using Expected = DefaultQuadrature<ElementType>;

    EXPECT_EQ(FE::DefaultQuadratureDegree, Expected::degree)
        << "the default quadrature degree of this element changed";
    EXPECT_EQ(static_cast<sofa::Size>(FE::quadraturePoints().size()), Expected::points)
        << "the default quadrature degree " << FE::DefaultQuadratureDegree
        << " no longer resolves to the same number of points";
}

/// A linear element and the quadratic element over the same reference shape.
template <class Linear, class Quadratic>
struct ElementPair
{
    using LinearElement = Linear;
    using QuadraticElement = Quadratic;
};

template <class Pair>
class ElementPairTest : public ::testing::Test {};

using LinearAndQuadraticPairs = ::testing::Types<
    ElementPair<sofa::geometry::Edge,        sofa::geometry::QuadraticEdge>,
    ElementPair<sofa::geometry::Triangle,    sofa::geometry::QuadraticTriangle>,
    ElementPair<sofa::geometry::Quad,        sofa::geometry::QuadraticQuad>,
    ElementPair<sofa::geometry::Tetrahedron, sofa::geometry::QuadraticTetrahedron>,
    ElementPair<sofa::geometry::Hexahedron,  sofa::geometry::QuadraticHexahedron>>;

TYPED_TEST_SUITE(ElementPairTest, LinearAndQuadraticPairs);

/// A quadrature rule belongs to the reference shape, not to the interpolation order, so the two
/// elements over one shape must not merely agree on the rule: they must be served the very same
/// table. Comparing the addresses rather than the values is what says the rule exists once, and
/// that adding an order to the library adds no quadrature.
///
/// Swept from the lowest degree the quadratic element admits up to the largest one the shape
/// tabulates; below that the quadratic element refuses, which is its own test.
TYPED_TEST(ElementPairTest, areServedTheSameQuadratureTable)
{
    using LinearElement = typename TypeParam::LinearElement;
    using QuadraticElement = typename TypeParam::QuadraticElement;

    using LinearFE = sofa::fem::FiniteElement<LinearElement, sofa::defaulttype::Vec3Types>;
    using QuadraticFE = sofa::fem::FiniteElement<QuadraticElement, sofa::defaulttype::Vec3Types>;

    // the mapping the sharing rests on
    static_assert(std::is_same_v<sofa::fem::ReferenceDomain_t<QuadraticElement>, LinearElement>);
    static_assert(std::is_same_v<typename LinearFE::Quadrature::Dispatch,
                                 typename QuadraticFE::Quadrature::Dispatch>);

    constexpr sofa::Size largestDegree =
        LinearFE::Quadrature::Dispatch::Rules::tabulatedDegrees.back();

    for (sofa::Size degree = QuadraticFE::MinimumQuadratureDegree; degree <= largestDegree; ++degree)
    {
        const auto servedToLinear = LinearFE::quadratureRule(degree);
        const auto servedToQuadratic = QuadraticFE::quadratureRule(degree);

        EXPECT_EQ(servedToLinear.data(), servedToQuadratic.data())
            << "degree " << degree << " is served from two different tables";
        EXPECT_EQ(servedToLinear.size(), servedToQuadratic.size())
            << "degree " << degree;
    }
}

/// A quadratic element cannot be integrated with a rule meant for a linear one: the product of
/// two of its shape function gradients is of degree 2, where a linear element's is constant.
/// The element refuses, rather than quietly under-integrating and returning a stiffness matrix
/// that looks plausible.
///
/// The linear element over the same shape is the control: it accepts the very degree the
/// quadratic one rejects, so what is tested is a difference in behaviour between the two and
/// not a constant written twice.
TYPED_TEST(ElementPairTest, refuseADegreeBelowWhatTheirOrderNeeds)
{
    using QuadraticElement = typename TypeParam::QuadraticElement;

    using LinearFE = sofa::fem::FiniteElement<typename TypeParam::LinearElement,
                                              sofa::defaulttype::Vec3Types>;
    using QuadraticFE = sofa::fem::FiniteElement<QuadraticElement, sofa::defaulttype::Vec3Types>;

    static_assert(QuadraticFE::MinimumQuadratureDegree > LinearFE::MinimumQuadratureDegree,
        "a quadratic element must demand more of its quadrature than a linear one");

    constexpr sofa::Size tooLowForQuadratic = QuadraticFE::MinimumQuadratureDegree - 1;

    EXPECT_NO_THROW(LinearFE::quadratureRule(tooLowForQuadratic))
        << "degree " << tooLowForQuadratic << " is enough for the linear element of this shape";

    try
    {
        QuadraticFE::quadratureRule(tooLowForQuadratic);
        ADD_FAILURE() << "degree " << tooLowForQuadratic
                      << " is too low for this element and was served anyway";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find(sofa::geometry::elementTypeToString(QuadraticElement::Element_type)),
                  std::string::npos) << "the message should name the element: " << message;
        EXPECT_NE(message.find(std::to_string(tooLowForQuadratic)), std::string::npos)
            << "the message should name the refused degree: " << message;
    }
}

}
