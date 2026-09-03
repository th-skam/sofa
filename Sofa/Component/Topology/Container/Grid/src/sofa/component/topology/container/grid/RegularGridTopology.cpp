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
#include <sofa/component/topology/container/grid/RegularGridTopology.h>

#include <sofa/core/ObjectFactory.h>

#include <cmath>

namespace sofa::component::topology::container::grid
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using type::vector;

namespace
{

/// Solve sinh(x)/x = B for x > 0, given B > 1, via Newton's method.
static SReal solveSinhOverX(SReal B)
{
    SReal x = (B < 3) ? std::sqrt(SReal(6) * (B - 1)) : std::log(2 * B) + std::log(std::log(2 * B));
    for (int it = 0; it < 50; ++it)
    {
        const SReal sh = std::sinh(x), ch = std::cosh(x);
        const SReal step = (sh / x - B) / ((x * ch - sh) / (x * x));
        x -= step;
        if (std::abs(step) < 1e-12 * std::max(SReal(1), x))
            break;
    }
    return x;
}

/// Solve sin(x)/x = B for x in (0, pi), given 0 < B < 1, via Newton's method safeguarded by
/// bisection (sin(x)/x is monotonically decreasing on that interval, so a bracket always exists).
static SReal solveSinOverX(SReal B)
{
    SReal lo = 0, hi = SReal(M_PI);
    SReal x = std::min(std::sqrt(SReal(6) * (1 - B)), hi - SReal(1e-6));
    for (int it = 0; it < 100; ++it)
    {
        const SReal s = std::sin(x), c = std::cos(x);
        const SReal f = s / x - B;
        (f > 0 ? lo : hi) = x;
        const SReal df = (x * c - s) / (x * x);
        SReal xNext = (std::abs(df) > 1e-14) ? x - f / df : SReal(0.5) * (lo + hi);
        if (xNext <= lo || xNext >= hi)
            xNext = SReal(0.5) * (lo + hi);
        if (std::abs(xNext - x) < 1e-12)
        {
            x = xNext;
            break;
        }
        x = xNext;
    }
    return x;
}

/// Vinokur (1983) stretching parameter delta for target end-spacings s0, s1 (relative to
/// uniform spacing). Solved once per axis and cached, since it requires a transcendental solve.
static SReal computeVinokurDelta(SReal s0, SReal s1)
{
    const SReal B = std::sqrt(s0 * s1);
    if (std::abs(B - 1) < 1e-10)
        return 0;
    return (B > 1) ? solveSinhOverX(B) : solveSinOverX(B);
}

/// Vinokur two-sided stretching function: maps a uniform index fraction xi in [0,1] to the
/// corresponding physical position fraction t in [0,1], for target end-spacings s0, s1 and
/// the precomputed delta = computeVinokurDelta(s0, s1).
static SReal vinokurT(SReal xi, SReal s0, SReal s1, SReal delta)
{
    const SReal B = std::sqrt(s0 * s1);
    const SReal A = std::sqrt(s0 / s1);

    SReal u;
    if (std::abs(B - 1) < 1e-10)
        u = xi * (1 + 2 * (B - 1) * (xi - SReal(0.5)) * (1 - xi));
    else if (B > 1)
        u = SReal(0.5) + std::tanh(delta * (xi - SReal(0.5))) / (2 * std::tanh(delta / 2));
    else
        u = SReal(0.5) + std::tan(delta * (xi - SReal(0.5))) / (2 * std::tan(delta / 2));

    return u / (A + (1 - A) * u);
}

/// Inverse of vinokurT: physical position fraction t in [0,1] to uniform index fraction xi in [0,1].
static SReal vinokurXi(SReal t, SReal s0, SReal s1, SReal delta)
{
    const SReal B = std::sqrt(s0 * s1);
    const SReal A = std::sqrt(s0 / s1);
    const SReal u = A * t / (1 - (1 - A) * t);

    if (std::abs(B - 1) < 1e-10)
        return u * (1 - 2 * (B - 1) * (u - SReal(0.5)) * (1 - u));
    if (B > 1)
        return SReal(0.5) + std::atanh((2 * u - 1) * std::tanh(delta / 2)) / delta;
    return SReal(0.5) + std::atan((2 * u - 1) * std::tan(delta / 2)) / delta;
}

} // anonymous namespace

/// To avoid duplicating the code in the different variants of the constructor
/// this object is using the delegating constructor feature of c++ x11.
/// The following constructor is "chained" by the other constructors to
/// defined only one the member initialization.
RegularGridTopology::RegularGridTopology(const type::Vec3i& dimXYZ)
    : GridTopology(dimXYZ)
    , d_min(initData(&d_min,type::Vec3(0_sreal, 0_sreal, 0_sreal),"min", "Min end of the diagonal"))
    , d_max(initData(&d_max,type::Vec3(1_sreal, 1_sreal, 1_sreal),"max", "Max end of the diagonal"))
    , d_p0(initData(&d_p0,type::Vec3(0_sreal, 0_sreal, 0_sreal),"p0", "Offset all the grid points"))
    , d_cellWidth(initData(&d_cellWidth, 0_sreal, "cellWidth","if > 0 : dimension of each cell in the created grid. Otherwise, the cell size is computed based on min, max, and resolution n."))
    , d_gradingRatioStart(initData(&d_gradingRatioStart, type::Vec3(1_sreal, 1_sreal, 1_sreal), "gradingRatioStart", "Per-axis target point density at the grid's start, relative to uniform spacing. 1 = uniform, > 1 = denser, < 1 = coarser."))
    , d_gradingRatioEnd(initData(&d_gradingRatioEnd, type::Vec3(1_sreal, 1_sreal, 1_sreal), "gradingRatioEnd", "Per-axis target point density at the grid's end, relative to uniform spacing. 1 = uniform, > 1 = denser, < 1 = coarser."))
{
}

RegularGridTopology::RegularGridTopology()
    : RegularGridTopology(type::Vec3i(2,2,2))
{
}

RegularGridTopology::RegularGridTopology(int nx, int ny, int nz)
    : RegularGridTopology(type::Vec3i(nx,ny,nz))
{
}

RegularGridTopology::RegularGridTopology(const type::Vec3i& n, type::BoundingBox b)
    : RegularGridTopology(n)
{
    setPos(b);
}


void RegularGridTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    float scale=1.0f;
    if (arg->getAttribute("scale")!=nullptr)
    {
        scale = arg->getAttributeAsFloat("scale", 1.0);
    }

    this->GridTopology::parse(arg);
    if (arg->getAttribute("xmin") != nullptr &&
        arg->getAttribute("ymin") != nullptr &&
        arg->getAttribute("zmin") != nullptr &&
        arg->getAttribute("xmax") != nullptr &&
        arg->getAttribute("ymax") != nullptr &&
        arg->getAttribute("zmax") != nullptr )
    {
        const float xmin = arg->getAttributeAsFloat("xmin",0);
        const float ymin = arg->getAttributeAsFloat("ymin",0);
        const float zmin = arg->getAttributeAsFloat("zmin",0);
        const float xmax = arg->getAttributeAsFloat("xmax",1);
        const float ymax = arg->getAttributeAsFloat("ymax",1);
        const float zmax = arg->getAttributeAsFloat("zmax",1);
        d_min.setValue(type::Vec3((SReal)xmin*scale, (SReal)ymin*scale, (SReal)zmin*scale));
        d_max.setValue(type::Vec3((SReal)xmax*scale, (SReal)ymax*scale, (SReal)zmax*scale));
    }
    this->setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);

}

void RegularGridTopology::init()
{
    auto s0 = sofa::helper::getWriteAccessor(d_gradingRatioStart);
    auto s1 = sofa::helper::getWriteAccessor(d_gradingRatioEnd);
    for (sofa::Size i = 0; i < 3; ++i)
    {
        if (s0[i] <= 0)
        {
            msg_error() << d_gradingRatioStart.getName() << "[" << i << "] = " << s0[i]
                << " is invalid (must be > 0). Falling back to 1 (uniform) for this axis.";
            s0[i] = 1_sreal;
        }
        if (s1[i] <= 0)
        {
            msg_error() << d_gradingRatioEnd.getName() << "[" << i << "] = " << s1[i]
                << " is invalid (must be > 0). Falling back to 1 (uniform) for this axis.";
            s1[i] = 1_sreal;
        }
        m_gradingDelta[i] = computeVinokurDelta(s0[i], s1[i]);
    }

    if (d_cellWidth.getValue())
    {
        const SReal w = d_cellWidth.getValue();

        type::Vec3i grid;
        grid[0]= (int)ceil((d_max.getValue()[0]-d_min.getValue()[0]) / w)+1;
        grid[1]= (int)ceil((d_max.getValue()[1]-d_min.getValue()[1]) / w)+1;
        grid[2]= (int)ceil((d_max.getValue()[2]-d_min.getValue()[2]) / w)+1;
        d_n.setValue(grid);
        setNbGridPoints();
    }

    Inherit1::init();
}

void RegularGridTopology::reinit()
{
    setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);

    Inherit1::reinit();
}

void RegularGridTopology::changeGridResolutionPostProcess()
{
    setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);
}

void RegularGridTopology::setPos(type::BoundingBox b)
{
    type::Vec3 m=b.minBBox(), M=b.maxBBox();
    setPos(m[0],M[0],m[1],M[1],m[2],M[2]);
}

void RegularGridTopology::setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax)
{
    SReal p0x=xmin, p0y=ymin, p0z=zmin;
    const type::Vec3i _n = d_n.getValue() - type::Vec3i(1,1,1);

    if (_n[0] > 0)
        setDx(type::Vec3((xmax - xmin) / _n[0], 0_sreal, 0_sreal));
    else
    {
        setDx(type::Vec3(xmax-xmin, 0_sreal, 0_sreal));
        p0x = (xmax+xmin)/2;
    }

    if (_n[1] > 0)
        setDy(type::Vec3(0_sreal, (ymax - ymin) / _n[1], 0_sreal));
    else
    {
        setDy(type::Vec3(0_sreal, ymax-ymin, 0_sreal));
        p0y = (ymax+ymin)/2;
    }

    if (_n[2] > 0)
        setDz(type::Vec3(0_sreal, 0_sreal, (zmax - zmin) / _n[2]));
    else
    {
        setDz(type::Vec3(0_sreal, 0_sreal, zmax-zmin));
        //p0z = (zmax+zmin)/2;
        p0z = zmin;
    }

    d_min.setValue(type::Vec3(xmin,ymin,zmin));
    d_max.setValue(type::Vec3(xmax,ymax,zmax));
    if (!d_p0.isSet())
    {
        setP0(type::Vec3(p0x,p0y,p0z));
    }
}

sofa::type::Vec3 RegularGridTopology::getPointInGrid(int i, int j, int k) const
{
    const auto& s0 = d_gradingRatioStart.getValue();
    const auto& s1 = d_gradingRatioEnd.getValue();
    const auto& n = d_n.getValue();
    // dx, dy, dz are the uniform step vectors; scale each by the graded fraction of its axis
    // instead of the uniform integer index.
    const SReal fx = n[0] > 1 ? vinokurT(SReal(i) / (n[0] - 1), s0[0], s1[0], m_gradingDelta[0]) * (n[0] - 1) : 0;
    const SReal fy = n[1] > 1 ? vinokurT(SReal(j) / (n[1] - 1), s0[1], s1[1], m_gradingDelta[1]) * (n[1] - 1) : 0;
    const SReal fz = n[2] > 1 ? vinokurT(SReal(k) / (n[2] - 1), s0[2], s1[2], m_gradingDelta[2]) * (n[2] - 1) : 0;
    return d_p0.getValue() + dx * fx + dy * fy + dz * fz;
}

RegularGridTopology::Index RegularGridTopology::findPoint(const type::Vec3& position) const
{
    const type::Vec3 p0 = d_p0.getValue();
    const type::Vec3i n = d_n.getValue();
    const type::Vec3& s0 = d_gradingRatioStart.getValue();
    const type::Vec3& s1 = d_gradingRatioEnd.getValue();
    const type::Vec3 p = position - p0;

    const int ix = n[0] > 1 ? int(std::round(vinokurXi(p[0] / (dx[0] * (n[0] - 1)), s0[0], s1[0], m_gradingDelta[0]) * (n[0] - 1))) : 0;
    const int iy = n[1] > 1 ? int(std::round(vinokurXi(p[1] / (dy[1] * (n[1] - 1)), s0[1], s1[1], m_gradingDelta[1]) * (n[1] - 1))) : 0;
    const int iz = n[2] > 1 ? int(std::round(vinokurXi(p[2] / (dz[2] * (n[2] - 1)), s0[2], s1[2], m_gradingDelta[2]) * (n[2] - 1))) : 0;

    if (ix < 0 || iy < 0 || iz < 0)
        return InvalidID;
    if (ix > n[0] - 1 || iy > n[1] - 1 || iz > n[2] - 1)
        return InvalidID;

    return getIndex(ix, iy, iz);
}

RegularGridTopology::Index RegularGridTopology::findPoint(const type::Vec3& position, const SReal epsilon) const
{
    const type::Vec3 p0 = d_p0.getValue();
    const type::Vec3i n = d_n.getValue();
    const type::Vec3& s0 = d_gradingRatioStart.getValue();
    const type::Vec3& s1 = d_gradingRatioEnd.getValue();
    const type::Vec3 p = position - p0;

    const int ix = n[0] > 1 ? int(std::round(vinokurXi(p[0] / (dx[0] * (n[0] - 1)), s0[0], s1[0], m_gradingDelta[0]) * (n[0] - 1))) : 0;
    const int iy = n[1] > 1 ? int(std::round(vinokurXi(p[1] / (dy[1] * (n[1] - 1)), s0[1], s1[1], m_gradingDelta[1]) * (n[1] - 1))) : 0;
    const int iz = n[2] > 1 ? int(std::round(vinokurXi(p[2] / (dz[2] * (n[2] - 1)), s0[2], s1[2], m_gradingDelta[2]) * (n[2] - 1))) : 0;

    if (ix < 0 || iy < 0 || iz < 0)
        return InvalidID;
    if (ix > n[0] || iy > n[1] || iz > n[2])
        return InvalidID;

    const auto node_index = getIndex(ix, iy, iz);

    // Check that the queried position lies within epsilon * (first-cell size) of the found node.
    const auto node_position = type::Vec3(
        n[0] > 1 ? vinokurT(SReal(ix) / (n[0] - 1), s0[0], s1[0], m_gradingDelta[0]) * dx[0] * (n[0] - 1) : 0,
        n[1] > 1 ? vinokurT(SReal(iy) / (n[1] - 1), s0[1], s1[1], m_gradingDelta[1]) * dy[1] * (n[1] - 1) : 0,
        n[2] > 1 ? vinokurT(SReal(iz) / (n[2] - 1), s0[2], s1[2], m_gradingDelta[2]) * dz[2] * (n[2] - 1) : 0);
    const auto m = epsilon * dx[0];
    const auto e = p - node_position;

    if (e[0]*e[0] + e[1]*e[1] + e[2]*e[2] > m*m)
        return InvalidID;

    return node_index;
}

/// Compute the fractional cell index along each axis for a point p relative to the grid origin.
/// Uses the uniform formula when both gradingRatio Datas are [1,1,1], to avoid any overhead.
void RegularGridTopology::computeCellIndex(const type::Vec3& p, SReal& x, SReal& y, SReal& z) const
{
    const auto& s0 = d_gradingRatioStart.getValue();
    const auto& s1 = d_gradingRatioEnd.getValue();
    if (s0 == type::Vec3(1_sreal, 1_sreal, 1_sreal) && s1 == type::Vec3(1_sreal, 1_sreal, 1_sreal))
    {
        x = p*dx*inv_dx2;
        y = p*dy*inv_dy2;
        z = p*dz*inv_dz2;
    }
    else
    {
        const auto& n = d_n.getValue();
        const SReal dx_len = std::sqrt(dx*dx);
        const SReal dy_len = std::sqrt(dy*dy);
        const SReal dz_len = std::sqrt(dz*dz);
        x = n[0] > 1 ? vinokurXi((p*dx/dx_len) / (dx_len*(n[0]-1)), s0[0], s1[0], m_gradingDelta[0]) * (n[0]-1) : 0;
        y = n[1] > 1 ? vinokurXi((p*dy/dy_len) / (dy_len*(n[1]-1)), s0[1], s1[1], m_gradingDelta[1]) * (n[1]-1) : 0;
        z = n[2] > 1 ? vinokurXi((p*dz/dz_len) / (dz_len*(n[2]-1)), s0[2], s1[2], m_gradingDelta[2]) * (n[2]-1) : 0;
    }
}

/// return the cube containing the given point (or -1 if not found).
RegularGridTopology::Index RegularGridTopology::findCube(const type::Vec3& pos)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2)
        return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();
    SReal x, y, z;
    computeCellIndex(p, x, y, z);
    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));
    if (   (unsigned)ix <= (unsigned)n[0]-2
            && (unsigned)iy <= (unsigned)n[1]-2
            && (unsigned)iz <= (unsigned)n[2]-2 )
    {
        return cube(ix,iy,iz);
    }
    else
    {
        return InvalidID;
    }
}

/// return the nearest cube (or -1 if not found).
RegularGridTopology::Index RegularGridTopology::findNearestCube(const type::Vec3& pos)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2) return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();
    SReal x, y, z;
    computeCellIndex(p, x, y, z);
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    int iz = int(std::floor(z));
    if (ix<0) ix=0; else if (ix>n[0]-2) ix=n[0]-2;
    if (iy<0) iy=0; else if (iy>n[1]-2) iy=n[1]-2;
    if (iz<0) iz=0; else if (iz>n[2]-2) iz=n[2]-2;
    return cube(ix,iy,iz);
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
RegularGridTopology::Index RegularGridTopology::findCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2) return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();
    SReal x, y, z;
    computeCellIndex(p, x, y, z);
    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));

    if ((unsigned)ix<=(unsigned)n[0]-2 && (unsigned)iy<=(unsigned)n[1]-2 && (unsigned)iz<=(unsigned)n[2]-2)
    {
        fx = x-ix;
        fy = y-iy;
        fz = z-iz;
        return cube(ix,iy,iz);
    }
    else
    {
        return InvalidID;
    }
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
RegularGridTopology::Index RegularGridTopology::findNearestCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2) return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();
    SReal x, y, z;
    computeCellIndex(p, x, y, z);
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    int iz = int(std::floor(z));
    if (ix<0) ix=0; else if (ix>n[0]-2) ix=n[0]-2;
    if (iy<0) iy=0; else if (iy>n[1]-2) iy=n[1]-2;
    if (iz<0) iz=0; else if (iz>n[2]-2) iz=n[2]-2;
    fx = x-ix;
    fy = y-iy;
    fz = z-iz;
    return cube(ix,iy,iz);
}


RegularGridTopology::Index RegularGridTopology::getCubeIndex( int i, int j, int k ) const
{
    const auto n = d_n.getValue();
    return (n[0]-1)* ( (n[1]-1)*k + j ) + i;
}

sofa::type::Vec3 RegularGridTopology::getCubeCoordinate(RegularGridTopology::Index i) const
{
    const auto n = d_n.getValue();
    type::Vec3 result;
    result[0] = (SReal)(i%(n[0]-1)); i/=(n[0]-1);
    result[1] = (SReal)(i%(n[1]-1)); i/=(n[1]-1);
    result[2] = (SReal)i;
    return result;
}

void RegularGridTopology::createTexCoords()
{
    const std::size_t nPts = this->getNbPoints();
    const type::Vec3i& _n = d_n.getValue();

    if ( (_n[0] == 1 && _n[1] == 1) || (_n[0] == 1 && _n[2] == 1) || (_n[1] == 1 && _n[2] == 1))
    {
        msg_warning() << "Can't create Texture coordinates as at least 2 dimensions of the grid are null." ;
        return;
    }

    auto _texCoords = sofa::helper::getWriteAccessor(this->d_seqUVs);
    _texCoords.resize(nPts);

    // check if flat grid
    type::Vec3u axes;

    if (_n[0] == 1)
        axes = type::Vec3u(1u, 2u, 0u);
    else if (_n[1] == 1)
        axes = type::Vec3u(0u, 2u, 1u);
    else
        axes = type::Vec3u(0u, 1u, 2u);

    SReal Uscale = 1/(SReal)(_n[ axes[0] ]-1);
    SReal Vscale = 1/(SReal)(_n[ axes[1] ]-1);

    for (int n1 = 0; n1 < _n[ axes[1] ]; ++n1)
    {
        for (int n0 = 0; n0 < _n[ axes[0] ]; ++n0)
        {
            const unsigned int pt1 = n0 + _n[ axes[0] ] * n1;
            const unsigned int pt2 = n0 + _n[ axes[0] ] * (n1 + _n[ axes[1] ] * (_n[ axes[2] ]-1));
            _texCoords[pt1] = Vec2(n0*Uscale, n1*Vscale);
            _texCoords[pt2] = Vec2(1- n0*Uscale, 1 - n1*Vscale);
        }
    }

    if (_n[ axes[2] ] != 0)
    {
        Uscale = 1/(SReal)(_n[ axes[0] ]-1);
        Vscale = 1/(SReal)(_n[ axes[2] ]-1);

        for (int n2 = 1; n2 < _n[ axes[2] ]-1; ++n2)
        {
            for (int n0 = 1; n0 < _n[ axes[0] ]-1; ++n0)
            {
                const unsigned int pt1 = n0 + _n[ axes[0] ] * n2;
                const unsigned int pt2 = n0 + _n[ axes[0] ] * (n2 + _n[ axes[2] ] * (_n[ axes[1] ]-1));
                _texCoords[pt1] = Vec2(n0*Uscale, n2*Vscale);
                _texCoords[pt2] = Vec2(1- n0*Uscale, 1 - n2*Vscale);
            }
        }


        Uscale = 1/(SReal)(_n[ axes[2] ]-1);
        Vscale = 1/(SReal)(_n[ axes[1] ]-1);

        for (int n1 = 1; n1 < _n[ axes[1] ]-1; ++n1)
        {
            for (int n2 = 1; n2 < _n[ axes[2] ]-1; ++n2)
            {
                const unsigned int pt1 = n2 + _n[ axes[2] ] * n1;
                const unsigned int pt2 = n2 + _n[ axes[2] ] * (n1 + _n[ axes[1] ] * (_n[ axes[0] ])-1);
                _texCoords[pt1] = Vec2(n2*Uscale, n1*Vscale);
                _texCoords[pt2] = Vec2(1- n2*Uscale, 1 - n1*Vscale);
            }
        }
    }
}

void registerRegularGridTopology(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Regular grid in 3D.")
        .add< RegularGridTopology >());
}

} //namespace sofa::component::topology::container::grid
