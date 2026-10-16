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

#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/HexahedronRotation.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/IdentityRotation.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/PolarDecomposition.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/RotationMethodsContainer.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/StablePolarDecomposition.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/TriangleRotation.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/CauchyStressEvaluator.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_ELEMENT_COROTATIONAL_FEM_FORCE_FIELD_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @brief A container for rotation computation methods in corotational formulations
 *
 * This class provides element-specific rotation computation strategies for common element types.
 *
 * @tparam DataTypes The data type used throughout the simulation (e.g., sofa::defaulttype::Vec3Types for 3D)
 * @tparam ElementType The element type (e.g., sofa::geometry::Triangle, sofa::geometry::Tetrahedron)
 *
 * Inherits from RotationMethodsContainer with pre-defined rotation strategies.
 *
 * The class is specialized for some elements because some rotation strategies can be used only
 * for specific elements.
 */
template <class DataTypes, class ElementType>
struct RotationMethods : RotationMethodsContainer<DataTypes, ElementType,
    StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation
>
{
    using Inherit = RotationMethodsContainer<DataTypes, ElementType, StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation>;
    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {}
};

template <class Real>
using Vec3Real = sofa::defaulttype::StdVectorTypes<sofa::type::Vec<3, Real>, sofa::type::Vec<3, Real>, Real>;

//partial specialization for linear triangle in 3D
template <class Real>
struct RotationMethods<Vec3Real<Real>, sofa::geometry::Triangle> : RotationMethodsContainer<Vec3Real<Real>, sofa::geometry::Triangle,
    StablePolarDecomposition<Vec3Real<Real>>, PolarDecomposition<Vec3Real<Real>>, IdentityRotation, TriangleRotation<Vec3Real<Real>>
>
{
    using Inherit = RotationMethodsContainer<Vec3Real<Real>, sofa::geometry::Triangle,
        StablePolarDecomposition<Vec3Real<Real>>, PolarDecomposition<Vec3Real<Real>>, IdentityRotation, TriangleRotation<Vec3Real<Real>> >;

    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {
        this->d_rotationMethod.setValue(TriangleRotation<Vec3Real<Real>>::getItem().key);
    }
};

//partial specialization for linear tetrahedron
template <class DataTypes>
struct RotationMethods<DataTypes, sofa::geometry::Tetrahedron> : RotationMethodsContainer<DataTypes, sofa::geometry::Tetrahedron,
    StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, TriangleRotation<DataTypes>
>
{
    using Inherit = RotationMethodsContainer<DataTypes, sofa::geometry::Tetrahedron,
        StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, TriangleRotation<DataTypes> >;

    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {
        this->d_rotationMethod.setValue(TriangleRotation<DataTypes>::getItem().key);
    }
};

//partial specialization for linear hexahedron
template <class DataTypes>
struct RotationMethods<DataTypes, sofa::geometry::Hexahedron> : RotationMethodsContainer<DataTypes, sofa::geometry::Hexahedron,
    StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, HexahedronRotation<DataTypes>
>
{
    using Inherit = RotationMethodsContainer<DataTypes, sofa::geometry::Hexahedron,
        StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, HexahedronRotation<DataTypes> >;

    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {
        this->d_rotationMethod.setValue(HexahedronRotation<DataTypes>::getItem().key);
    }
};

//partial specialization for quadratic tetrahedron
template <class DataTypes>
struct RotationMethods<DataTypes, sofa::geometry::QuadraticTetrahedron> : RotationMethodsContainer<DataTypes, sofa::geometry::QuadraticTetrahedron,
    PolarDecomposition<DataTypes>, IdentityRotation
>
{
    using Inherit = RotationMethodsContainer<DataTypes, sofa::geometry::QuadraticTetrahedron,
        PolarDecomposition<DataTypes>, IdentityRotation>;

    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {
        this->d_rotationMethod.setValue(PolarDecomposition<DataTypes>::getItem().key);
    }
};


template <class DataTypes, class ElementType>
class CorotationalFEMForceField :
    public BaseElementLinearFEMForceField<DataTypes, ElementType>,
    public FEMForceField<DataTypes, ElementType>,
    public CauchyStressEvaluator<DataTypes>
{
public:
    SOFA_CLASS2(
        SOFA_TEMPLATE2(CorotationalFEMForceField, DataTypes, ElementType),
        SOFA_TEMPLATE2(BaseElementLinearFEMForceField, DataTypes, ElementType),
        SOFA_TEMPLATE2(FEMForceField, DataTypes, ElementType));

private:
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    using ElementGradient = typename trait::ElementGradient;
    using ElementDisplacement = typename trait::ElementDisplacement;
    using RotationMatrix = sofa::type::Mat<trait::spatial_dimensions, trait::spatial_dimensions, sofa::Real_t<DataTypes>>;
    using DeformationGradient = typename trait::DeformationGradient;
    using StressVoigtVector = typename trait::StressVoigtVector;

public:

    CorotationalFEMForceField();

    void init() override;

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    using sofa::core::behavior::ForceField<DataTypes>::getPotentialEnergy;
    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const sofa::DataVecCoord_t<DataTypes>& x) const override;

    const sofa::type::vector<RotationMatrix>& getElementRotations() const { return m_rotations; }

    StressVoigtVector computeStress(const DeformationGradient& F, sofa::Size elementId) override;

    /// Diagnostic only, and expensive: recomputes each element's force 6 times per degree of
    /// freedom at every force evaluation. Off unless a scene turns it on.
    sofa::Data<bool> d_checkElementTangent;

protected:

    void beforeElementForce(const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& f,
        const sofa::VecCoord_t<DataTypes>& x) override;

    void computeElementsForces(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& f,
        const sofa::VecCoord_t<DataTypes>& x) override;

    void computeElementsForcesDeriv(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& elementForcesDeriv,
        const sofa::VecDeriv_t<DataTypes>& nodeDx) override;

    sofa::type::vector<RotationMatrix> m_rotations;
    sofa::type::vector<RotationMatrix> m_initialRotationsTransposed;

    /// A rotation has one component per independent plane: an angle in 2D, a vector in 3D. Q and W
    /// are the two matrices pairing it with a node's coordinates, so they are transposes in shape.
    static constexpr sofa::Size RotationComponents =
        NumberOfRotationComponents<trait::spatial_dimensions>;
    using RotationVector = sofa::type::Vec<RotationComponents, sofa::Real_t<DataTypes>>;
    using JacobianBlocks = std::array<
        sofa::type::Mat<RotationComponents, trait::spatial_dimensions, sofa::Real_t<DataTypes>>,
        trait::NumberOfNodesInElement>;
    using CouplingBlocks = std::array<
        sofa::type::Mat<trait::spatial_dimensions, RotationComponents, sofa::Real_t<DataTypes>>,
        trait::NumberOfNodesInElement>;

    /// Q = dtheta/dx per element, the derivative of the extracted rotation. Left empty where the
    /// selected rotation method cannot be differentiated, which is what keeps the tangent of those
    /// methods on the frozen-rotation path.
    sofa::type::vector<JacobianBlocks> m_rotationJacobians;

    /// W = K_l B_x - B_f per element: what dtheta multiplies in the tangent. Held alongside Q
    /// because both are formed where the rotation is, and read where the tangent is applied.
    sofa::type::vector<CouplingBlocks> m_rotationCouplings;

    /// Whether the tangent carries the dR/dx term, which is the selected method's to say.
    bool m_consistentTangent { false };

    /// W for one element, from the local displacement and force the rotation was formed with.
    void computeRotationCoupling(CouplingBlocks& coupling,
        const typename trait::ElementHessian& stiffness,
        const ElementDisplacement& localDisplacement,
        const ElementGradient& localForce,
        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& restNodes) const;

    using ElementNodes = std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>;

    /// R K_l (R^T p - q) for one element, with the rotation, local displacement and local force it
    /// was formed from. A function of the element's nodes alone, so a finite difference of it is a
    /// finite difference of the element residual -- and of the same code the solve runs.
    ElementGradient computeElementForce(const ElementNodes& nodes, const ElementNodes& restNodes,
        const typename trait::ElementHessian& stiffness,
        const RotationMatrix& initialRotationTransposed,
        RotationMatrix& rotation,
        ElementDisplacement& localDisplacement,
        ElementGradient& localForce);

    /// Diagnostic: the analytic element tangent against a central difference of the element force,
    /// and how far from symmetric it is, over every element at the current configuration. Reports the
    /// element where the error is worst.
    void checkElementTangents(const sofa::VecCoord_t<DataTypes>& x);

    /// Local displacement of the element nodes in the element's own rotated frame. Flat vector.
    ElementDisplacement computeElementLocalDisplacement(
        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes,
        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& restNodes,
        const RotationMatrix& rotation) const;

    sofa::Coord_t<DataTypes> translation(const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes) const;
    static sofa::Coord_t<DataTypes> computeCentroid(const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes);

    RotationMethods<DataTypes, ElementType> m_rotationMethods;

    void computeRotations(sofa::type::vector<RotationMatrix>& rotations,
        const sofa::VecCoord_t<DataTypes>& nodePositions,
        const sofa::VecCoord_t<DataTypes>& nodeRestPositions);
    void computeInitialRotations();
};



#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_COROTATIONAL_FEM_FORCE_FIELD_CPP)
// extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API CorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
