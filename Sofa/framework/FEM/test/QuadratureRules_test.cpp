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
#include <sofa/fem/QuadratureRules.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>

namespace sofa
{

/// The exponents of one monomial x^e0 * y^e1 * z^e2. Axes the domain does not have stay at 0.
using MonomialExponents = std::array<sofa::Size, 3>;

constexpr SReal factorial(sofa::Size n)
{
    SReal result = 1;
    for (sofa::Size i = 2; i <= n; ++i)
    {
        result *= static_cast<SReal>(i);
    }
    return result;
}

/// Integral of x^n over the segment [-1, 1]. Odd powers cancel by symmetry.
constexpr SReal integralOverSegment(sofa::Size n)
{
    return n % 2 == 0 ? 2. / static_cast<SReal>(n + 1) : 0.;
}

/// Integral of x^a * y^b over the unit triangle.
constexpr SReal integralOverUnitTriangle(sofa::Size a, sofa::Size b)
{
    return factorial(a) * factorial(b) / factorial(a + b + 2);
}

/// Integral of x^a * y^b * z^c over the unit tetrahedron.
constexpr SReal integralOverUnitTetrahedron(sofa::Size a, sofa::Size b, sofa::Size c)
{
    return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
}

/// Quadrature points may legitimately sit exactly on the boundary of the reference domain,
/// so containment is checked with a tolerance rather than strictly.
constexpr SReal boundaryTolerance = 1e-12;

/**
 * Everything a reference domain is expected to satisfy. Each specialization gives, independently 
 * of the tables it is checked against:
 *  - measure, the value the weights of any correct rule must sum to;
 *  - expectedRules, the inventory of the rules the domain is expected to tabulate;
 *  - isInside, whether a point lies in the domain;
 *  - monomialIntegral, the exact integral of one monomial over the domain.
 */
template <class Domain>
struct ReferenceDomainTraits;

template <> struct ReferenceDomainTraits<sofa::geometry::Edge>
{
    /// Every rule this domain tabulates, as (degree, number of points) pairs.
    static constexpr std::array<std::pair<sofa::Size, sofa::Size>, 2> expectedRules
        {{ {1, 1}, {3, 2} }};
    static constexpr SReal measure = 2.;        // length of [-1, 1]

    /// Whether a point lies in the reference domain, boundary included.
    static constexpr bool isInside(const sofa::type::Vec<1, SReal>& q)
    {
        return q[0] >= -1 - boundaryTolerance && q[0] <= 1 + boundaryTolerance;
    }

    /// Exact value of the integral of one monomial over the reference domain.
    static constexpr SReal monomialIntegral(const MonomialExponents& e)
    {
        return integralOverSegment(e[0]);
    }
};
template <> struct ReferenceDomainTraits<sofa::geometry::Triangle>
{
    static constexpr std::array<std::pair<sofa::Size, sofa::Size>, 2> expectedRules
        {{ {1, 1}, {2, 3} }};
    static constexpr SReal measure = 1. / 2.;   // area of the unit triangle

    static constexpr bool isInside(const sofa::type::Vec<2, SReal>& q)
    {
        return q[0] >= -boundaryTolerance && q[1] >= -boundaryTolerance
            && q[0] + q[1] <= 1 + boundaryTolerance;
    }

    static constexpr SReal monomialIntegral(const MonomialExponents& e)
    {
        return integralOverUnitTriangle(e[0], e[1]);
    }
};
template <> struct ReferenceDomainTraits<sofa::geometry::Quad>
{
    static constexpr std::array<std::pair<sofa::Size, sofa::Size>, 3> expectedRules
        {{ {1, 1}, {2, 3}, {3, 4} }};
    static constexpr SReal measure = 4.;        // area of [-1, 1]^2

    static constexpr bool isInside(const sofa::type::Vec<2, SReal>& q)
    {
        return std::abs(q[0]) <= 1 + boundaryTolerance
            && std::abs(q[1]) <= 1 + boundaryTolerance;
    }

    static constexpr SReal monomialIntegral(const MonomialExponents& e)
    {
        return integralOverSegment(e[0]) * integralOverSegment(e[1]);
    }
};
template <> struct ReferenceDomainTraits<sofa::geometry::Tetrahedron>
{
    static constexpr std::array<std::pair<sofa::Size, sofa::Size>, 2> expectedRules
        {{ {1, 1}, {2, 4} }};
    static constexpr SReal measure = 1. / 6.;   // volume of the unit tetrahedron

    static constexpr bool isInside(const sofa::type::Vec<3, SReal>& q)
    {
        return q[0] >= -boundaryTolerance && q[1] >= -boundaryTolerance
            && q[2] >= -boundaryTolerance
            && q[0] + q[1] + q[2] <= 1 + boundaryTolerance;
    }

    static constexpr SReal monomialIntegral(const MonomialExponents& e)
    {
        return integralOverUnitTetrahedron(e[0], e[1], e[2]);
    }
};
template <> struct ReferenceDomainTraits<sofa::geometry::Hexahedron>
{
    static constexpr std::array<std::pair<sofa::Size, sofa::Size>, 3> expectedRules
        {{ {1, 1}, {3, 8}, {5, 27} }};
    static constexpr SReal measure = 8.;        // volume of [-1, 1]^3

    static constexpr bool isInside(const sofa::type::Vec<3, SReal>& q)
    {
        return std::abs(q[0]) <= 1 + boundaryTolerance
            && std::abs(q[1]) <= 1 + boundaryTolerance
            && std::abs(q[2]) <= 1 + boundaryTolerance;
    }

    static constexpr SReal monomialIntegral(const MonomialExponents& e)
    {
        return integralOverSegment(e[0]) * integralOverSegment(e[1])
            * integralOverSegment(e[2]);
    }
};
template <> struct ReferenceDomainTraits<sofa::geometry::Prism>
{
    static constexpr std::array<std::pair<sofa::Size, sofa::Size>, 1> expectedRules
        {{ {1, 2} }};
    static constexpr SReal measure = 1. / 2.;   // unit triangle extruded over [0, 1]

    static constexpr bool isInside(const sofa::type::Vec<3, SReal>& q)
    {
        return q[0] >= -boundaryTolerance && q[1] >= -boundaryTolerance
            && q[0] + q[1] <= 1 + boundaryTolerance
            && q[2] >= -boundaryTolerance && q[2] <= 1 + boundaryTolerance;
    }

    static constexpr SReal monomialIntegral(const MonomialExponents& e)
    {
        // the unit triangle in x and y, extruded over [0, 1] in z
        return integralOverUnitTriangle(e[0], e[1]) / static_cast<SReal>(e[2] + 1);
    }
};
template <> struct ReferenceDomainTraits<sofa::geometry::Pyramid>
{
    static constexpr std::array<std::pair<sofa::Size, sofa::Size>, 1> expectedRules
        {{ {1, 8} }};
    // The pyramid is integrated in the coordinates of the cube it is collapsed from, so the
    // weights sum to the volume of that cube.
    static constexpr SReal measure = 8.;

    static constexpr bool isInside(const sofa::type::Vec<3, SReal>& q)
    {
        // the cube the pyramid is collapsed from, see measure above
        return std::abs(q[0]) <= 1 + boundaryTolerance
            && std::abs(q[1]) <= 1 + boundaryTolerance
            && std::abs(q[2]) <= 1 + boundaryTolerance;
    }

    static constexpr SReal monomialIntegral(const MonomialExponents& e)
    {
        // integrated over the cube, see measure above
        return integralOverSegment(e[0]) * integralOverSegment(e[1])
            * integralOverSegment(e[2]);
    }
};

/**
 * Calls f once per degree tabulated for Domain, in ascending order.
 *
 * Index walks tabulatedDegrees one entry at a time and the recursion stops past the last one.
 */
template <class Domain, sofa::Size Index = 0, class F>
void forEachTabulatedDegree(F&& f)
{
    using Rules = sofa::fem::QuadratureRules<Domain>;
    if constexpr (Index < Rules::tabulatedDegrees.size())
    {
        constexpr sofa::Size degree = Rules::tabulatedDegrees[Index];
        f(std::integral_constant<sofa::Size, degree>{});

        forEachTabulatedDegree<Domain, Index + 1>(f);
    }
}

/// Calls f once for every monomial of total degree at most maxDegree, over the axes the domain
/// actually has: a 2D domain never receives an exponent on z.
template <sofa::Size Dimension, class F>
void forEachMonomial(sofa::Size maxDegree, F&& f)
{
    const sofa::Size lastOnY = Dimension > 1 ? maxDegree : 0;
    const sofa::Size lastOnZ = Dimension > 2 ? maxDegree : 0;

    MonomialExponents exponents{};
    for (exponents[0] = 0; exponents[0] <= maxDegree; ++exponents[0])
    {
        for (exponents[1] = 0; exponents[1] <= lastOnY; ++exponents[1])
        {
            for (exponents[2] = 0; exponents[2] <= lastOnZ; ++exponents[2])
            {
                if (exponents[0] + exponents[1] + exponents[2] <= maxDegree)
                {
                    f(exponents);
                }
            }
        }
    }
}

/// Value of the monomial at one quadrature point.
template <sofa::Size Dimension, class Point>
SReal evaluateMonomial(const Point& q, const MonomialExponents& exponents)
{
    SReal value = 1;
    for (sofa::Size axis = 0; axis < Dimension; ++axis)
    {
        value *= std::pow(q[axis], static_cast<int>(exponents[axis]));
    }
    return value;
}

/// The monomial written out, for failure messages: "x^2 y".
inline std::string monomialExpression2String(const MonomialExponents& exponents, sofa::Size dimension)
{
    static constexpr std::array<const char*, 3> axisName{ "x", "y", "z" };

    std::ostringstream description;
    for (sofa::Size axis = 0; axis < dimension; ++axis)
    {
        if (exponents[axis] > 0)
        {
            description << axisName[axis];
            if (exponents[axis] > 1)
            {
                description << '^' << exponents[axis];
            }
            description << ' ';
        }
    }
    const std::string written = description.str();
    return written.empty() ? "1" : written;
}

/// Smallest degree in the inventory that covers the request, or 0 when the request exceeds
/// every rule the domain has. Derived from the inventory here in the tests.
template <class Domain>
constexpr sofa::Size expectedSelectedDegree(sofa::Size requestedDegree)
{
    sofa::Size selected = 0;
    for (const auto& [degree, numberOfPoints] : ReferenceDomainTraits<Domain>::expectedRules)
    {
        if (degree >= requestedDegree && (selected == 0 || degree < selected))
            selected = degree;
    }
    return selected;
}

/// Highest degree the inventory lists for this domain.
template <class Domain>
constexpr sofa::Size largestExpectedDegree()
{
    sofa::Size largest = 0;
    for (const auto& [degree, numberOfPoints] : ReferenceDomainTraits<Domain>::expectedRules)
    {
        if (degree > largest)
            largest = degree;
    }
    return largest;
}

template <class Domain>
class QuadratureRulesTest : public ::testing::Test {};

using ReferenceDomains = ::testing::Types<
    sofa::geometry::Edge,
    sofa::geometry::Triangle,
    sofa::geometry::Quad,
    sofa::geometry::Tetrahedron,
    sofa::geometry::Hexahedron,
    sofa::geometry::Prism,
    sofa::geometry::Pyramid>;

TYPED_TEST_SUITE(QuadratureRulesTest, ReferenceDomains);

/// A quadrature rule integrates the constant function 1 exactly, whatever its degree, so the
/// sum of its weights is the measure of the reference domain.
TYPED_TEST(QuadratureRulesTest, weightsSumToMeasureOfReferenceDomain)
{
    using Rules = sofa::fem::QuadratureRules<TypeParam>;
    constexpr SReal expected = ReferenceDomainTraits<TypeParam>::measure;

    forEachTabulatedDegree<TypeParam>([&expected](auto degree)
    {
        SReal weightSum = 0;
        for (const auto& [point, weight] : Rules::template points<degree.value, SReal>())
        {
            weightSum += weight;
        }
        EXPECT_NEAR(weightSum, expected, 1e-12) << "degree " << degree.value;
    });
}

/// The number of points of every tabulated rule, checked against a written-out inventory.
TYPED_TEST(QuadratureRulesTest, everyRuleHasTheExpectedNumberOfPoints)
{
    using Rules = sofa::fem::QuadratureRules<TypeParam>;
    using Traits = ReferenceDomainTraits<TypeParam>;

    // the tables and the inventory must describe the same set of rules
    EXPECT_EQ(Rules::tabulatedDegrees.size(), Traits::expectedRules.size());

    /// Number of points expected for one degree, or 0 when the inventory lists no such degree.
    const auto expectedNumberOfPoints = [](sofa::Size degree)
    {
        for (const auto& [tabulatedDegree, numberOfPoints] : Traits::expectedRules)
        {
            if (tabulatedDegree == degree)
                return numberOfPoints;
        }
        return sofa::Size{0};
    };

    forEachTabulatedDegree<TypeParam>([&expectedNumberOfPoints](auto degree)
    {
        const sofa::Size expected = expectedNumberOfPoints(degree.value);
        EXPECT_GT(expected, 0u) << "degree " << degree.value << " is tabulated but is absent "
                                   "from the expected inventory";
        EXPECT_EQ(static_cast<sofa::Size>(Rules::template points<degree.value, SReal>().size()),
                  expected) << "degree " << degree.value;
    });
}

/// Every quadrature point must lie in the reference domain it integrates over.
TYPED_TEST(QuadratureRulesTest, everyPointLiesInsideTheReferenceDomain)
{
    using Rules = sofa::fem::QuadratureRules<TypeParam>;
    using Traits = ReferenceDomainTraits<TypeParam>;

    forEachTabulatedDegree<TypeParam>([](auto degree)
    {
        sofa::Size index = 0;
        for (const auto& [point, weight] : Rules::template points<degree.value, SReal>())
        {
            EXPECT_TRUE(Traits::isInside(point))
                << "degree " << degree.value << ", point " << index << " at " << point;
            ++index;
        }
    });
}

/// A rule tabulated at degree d must integrate every polynomial of total degree at most d exactly.
/// A rule with the right weights at the wrong places passes everything above and fails here.
TYPED_TEST(QuadratureRulesTest, integratesPolynomialsUpToItsDegree)
{
    using Rules = sofa::fem::QuadratureRules<TypeParam>;
    using Traits = ReferenceDomainTraits<TypeParam>;

    forEachTabulatedDegree<TypeParam>([](auto degree)
    {
        constexpr auto rule = Rules::template points<degree.value, SReal>();

        forEachMonomial<Rules::TopologicalDimension>(degree.value,
            [&rule, &degree](const MonomialExponents& exponents)
        {
            SReal integral = 0;
            for (const auto& [point, weight] : rule)
            {
                integral += weight * evaluateMonomial<Rules::TopologicalDimension>(point, exponents);
            }

            EXPECT_NEAR(integral, Traits::monomialIntegral(exponents), 1e-12)
                << "degree " << degree.value
                << ", monomial " << monomialExpression2String(exponents, Rules::TopologicalDimension);
        });
    });
}

/// The counterpart of the test above: a rule tabulated at degree d must get at least one
/// polynomial of degree d + 1 wrong. Without it, the degree recorded in tabulatedDegrees is an
/// unverified label and could understate what the rule actually does.
///
/// It has to be "at least one" and not "all": symmetry makes many monomials of degree d + 1
/// come out exact by accident, an odd power on a symmetric domain being the obvious case.
TYPED_TEST(QuadratureRulesTest, doesNotIntegrateBeyondItsDegree)
{
    using Rules = sofa::fem::QuadratureRules<TypeParam>;
    using Traits = ReferenceDomainTraits<TypeParam>;

    forEachTabulatedDegree<TypeParam>([](auto degree)
    {
        constexpr auto rule = Rules::template points<degree.value, SReal>();
        constexpr sofa::Size degreeTooHigh = degree.value + 1;

        SReal largestError = 0;
        MonomialExponents worstMonomial{};

        forEachMonomial<Rules::TopologicalDimension>(degreeTooHigh,
            [&largestError, &worstMonomial, &rule](const MonomialExponents& exponents)
        {
            // only the monomials the rule is not supposed to reach
            if (exponents[0] + exponents[1] + exponents[2] != degreeTooHigh)
            {
                return;
            }

            SReal integral = 0;
            for (const auto& [point, weight] : rule)
            {
                integral += weight * evaluateMonomial<Rules::TopologicalDimension>(point, exponents);
            }

            const SReal error = std::abs(integral - Traits::monomialIntegral(exponents));
            if (error > largestError)
            {
                largestError = error;
                worstMonomial = exponents;
            }
        });

        // far above round-off, far below the error of a genuinely insufficient rule
        EXPECT_GT(largestError, 1e-9 * Traits::measure)
            << "the rule tabulated at degree " << degree.value
            << " integrates every monomial of degree " << degreeTooHigh
            << " exactly, so its degree is understated; closest miss was "
            << monomialExpression2String(worstMonomial, Rules::TopologicalDimension);
    });
}

/// A degree with no table of its own is served by the smallest tabulated degree above it, so a
/// caller states the accuracy it needs and never a table size. Swept over every degree the
/// domain covers; the first degree it cannot cover is then expected to select nothing.
TYPED_TEST(QuadratureRulesTest, selectsTheSmallestSufficientDegree)
{
    using Dispatch = sofa::fem::QuadratureDispatch<TypeParam>;

    constexpr sofa::Size largestDegree = largestExpectedDegree<TypeParam>();

    for (sofa::Size requested = 1; requested <= largestDegree; ++requested)
    {
        const sofa::Size selected = Dispatch::smallestSufficientDegree(requested);

        EXPECT_GT(selected, 0u) << "requested degree " << requested
                                << " is within reach but no rule was selected";
        EXPECT_EQ(selected, expectedSelectedDegree<TypeParam>(requested))
            << "requested degree " << requested;
    }

    // Past the largest tabulated degree the request cannot be served, and the dispatcher has to
    // say so rather than fall back on the closest rule it happens to have.
    EXPECT_EQ(Dispatch::smallestSufficientDegree(largestDegree + 1), 0u)
        << "requested degree " << largestDegree + 1 << " is beyond every rule of this domain";
}

/// The same selection, spelled out for the two domains whose tabulated degrees have gaps. The
/// sweep above proves it; this records what it means, and does so at compile time.
TEST(QuadratureDispatch, snapsUpOverTheGapsInTheTabulatedDegrees)
{
    using EdgeDispatch = sofa::fem::QuadratureDispatch<sofa::geometry::Edge>;
    static_assert(EdgeDispatch::smallestSufficientDegree(1) == 1);
    static_assert(EdgeDispatch::smallestSufficientDegree(2) == 3);  // no degree-2 table
    static_assert(EdgeDispatch::smallestSufficientDegree(3) == 3);
    static_assert(EdgeDispatch::smallestSufficientDegree(4) == 0);  // beyond every table

    using HexahedronDispatch = sofa::fem::QuadratureDispatch<sofa::geometry::Hexahedron>;
    static_assert(HexahedronDispatch::smallestSufficientDegree(1) == 1);
    static_assert(HexahedronDispatch::smallestSufficientDegree(2) == 3);  // no degree-2 table
    static_assert(HexahedronDispatch::smallestSufficientDegree(3) == 3);
    static_assert(HexahedronDispatch::smallestSufficientDegree(4) == 5);  // no degree-4 table
    static_assert(HexahedronDispatch::smallestSufficientDegree(5) == 5);
    static_assert(HexahedronDispatch::smallestSufficientDegree(6) == 0);  // beyond every table
}

/// A degree asked for at compile time and the same degree asked for at runtime must land on the
/// same table. The two take different paths to it, so nothing but this stops them from drifting
/// apart. Both are exercised over every degree the domain covers, not only the tabulated ones.
TYPED_TEST(QuadratureRulesTest, runtimeAndCompileTimeSelectionAgree)
{
    using Dispatch = sofa::fem::QuadratureDispatch<TypeParam>;
    constexpr sofa::Size largestDegree = largestExpectedDegree<TypeParam>();

    forEachTabulatedDegree<TypeParam>([](auto degree)
    {
        constexpr auto tabulated = Dispatch::template points<degree.value, SReal>();

        for (sofa::Size requested = 1; requested <= largestDegree; ++requested)
        {
            // every request this degree is the one to serve
            if (expectedSelectedDegree<TypeParam>(requested) != degree.value)
            {
                continue;
            }

            const auto selected = Dispatch::template rule<SReal>(requested);

            ASSERT_EQ(selected.size(), tabulated.size())
                << "requested degree " << requested << ", served by degree " << degree.value;

            for (std::size_t i = 0; i < tabulated.size(); ++i)
            {
                EXPECT_EQ(selected[i].first, tabulated[i].first)
                    << "requested degree " << requested << ", point " << i;
                EXPECT_EQ(selected[i].second, tabulated[i].second)
                    << "requested degree " << requested << ", weight " << i;
            }
        }
    });
}

/// Beyond its largest table a domain has nothing to offer, and must say so. Returning the
/// closest rule it happens to have would silently hand the caller less accuracy than asked for.
/// The boundary is checked from both sides, and the message has to name what was refused.
TYPED_TEST(QuadratureRulesTest, refusesADegreeBeyondEveryTable)
{
    using Dispatch = sofa::fem::QuadratureDispatch<TypeParam>;
    constexpr sofa::Size largestDegree = largestExpectedDegree<TypeParam>();

    EXPECT_NO_THROW(Dispatch::template rule<SReal>(largestDegree))
        << "degree " << largestDegree << " is the largest one tabulated and must be served";

    for (const sofa::Size beyond : { largestDegree + 1, largestDegree + 10 })
    {
        try
        {
            Dispatch::template rule<SReal>(beyond);
            ADD_FAILURE() << "degree " << beyond << " is beyond every table and was served anyway";
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find(sofa::geometry::elementTypeToString(TypeParam::Element_type)),
                      std::string::npos) << "the message should name the domain: " << message;
            EXPECT_NE(message.find(std::to_string(beyond)), std::string::npos)
                << "the message should name the refused degree: " << message;
        }
    }
}

/// rule() hands back a view, not a copy, so what it views has to outlive the call. Three things
/// follow if the table really has static storage: asking twice gives the same address, two
/// requests served by one degree give the same address, and a view taken early still reads
/// correctly after many later calls.
TYPED_TEST(QuadratureRulesTest, viewsOneStaticTablePerDegree)
{
    using Dispatch = sofa::fem::QuadratureDispatch<TypeParam>;
    constexpr sofa::Size largestDegree = largestExpectedDegree<TypeParam>();

    const auto takenFirst = Dispatch::template rule<SReal>(1);

    std::map<sofa::Size, const void*> addressOfServingDegree;

    for (sofa::Size requested = 1; requested <= largestDegree; ++requested)
    {
        const auto once = Dispatch::template rule<SReal>(requested);
        const auto twice = Dispatch::template rule<SReal>(requested);

        EXPECT_EQ(once.data(), twice.data())
            << "requested degree " << requested << " was served from two different addresses";

        // every request the same degree serves must be served from the one table of that degree
        const sofa::Size serving = expectedSelectedDegree<TypeParam>(requested);
        const auto [known, isNew] = addressOfServingDegree.emplace(serving, once.data());
        if (!isNew)
        {
            EXPECT_EQ(known->second, once.data())
                << "requested degree " << requested << " and an earlier request are both served "
                   "by degree " << serving << " but from different addresses";
        }
    }

    // the view taken before all of the above must still read correctly
    SReal weightSum = 0;
    for (const auto& [point, weight] : takenFirst)
    {
        weightSum += weight;
    }
    EXPECT_NEAR(weightSum, ReferenceDomainTraits<TypeParam>::measure, 1e-12)
        << "a view taken earlier no longer reads the table it was taken from";
}

/// Both selection paths stop at the first tabulated degree that covers the request and take it
/// to be the smallest one available. That is only true of an ascending list: out of order, a
/// request would silently be served by a larger rule than necessary. A repeated degree would
/// make one of its two tables unreachable, and a degree of 0 is not a rule at all.
///
/// Checked at compile time, so a violation stops the build rather than turning a test red.
TYPED_TEST(QuadratureRulesTest, tabulatedDegreesAreAscendingAndDistinct)
{
    using Rules = sofa::fem::QuadratureRules<TypeParam>;

    static_assert(!Rules::tabulatedDegrees.empty(),
        "a reference domain must tabulate at least one quadrature rule");
    static_assert(Rules::tabulatedDegrees.front() >= 1,
        "degree 0 is not a quadrature rule");

    /// Whether the degrees are listed in strictly increasing order.
    static_assert([]
    {
        const auto& degrees = Rules::tabulatedDegrees;
        for (std::size_t i = 1; i < degrees.size(); ++i)
        {
            if (degrees[i] <= degrees[i - 1])
                return false;
        }
        return true;
    }(), "tabulatedDegrees must be strictly ascending: the selection takes the first sufficient "
        "entry to be the smallest one");
}

}
