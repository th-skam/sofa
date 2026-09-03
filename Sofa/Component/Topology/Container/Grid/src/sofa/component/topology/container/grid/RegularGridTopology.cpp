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

namespace sofa::component::topology::container::grid
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using type::vector;

namespace
{

/// World-space position of grid point i out of n spanning total length L with geometric grading ratio r.
/// Cell sizes form a geometric series: first cell = s, next = s*r, etc.
/// r > 1: smallest cells at the start; r < 1: smallest cells at the end; r = 1: uniform.
static SReal gradedCoord(int i, int n, SReal L, SReal r)
{
    if (n <= 1)
        return 0;
    if (std::abs(r - 1) < 1e-10)
        return L * i / (n - 1);
    return L * (std::pow(r, i) - 1) / (std::pow(r, n - 1) - 1);
}

/// Inverse of gradedCoord: fractional cell index for a point at distance x from the axis origin.
static SReal gradedIndex(SReal x, int n, SReal L, SReal r)
{
    if (n <= 1 || L <= 0)
        return 0;
    if (std::abs(r - 1) < 1e-10)
        return x * (n - 1) / L;
    const SReal rn1 = std::pow(r, n - 1);
    return std::log1p(x * (rn1 - 1) / L) / std::log(r);
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
    , d_gradingRatio(initData(&d_gradingRatio, type::Vec3(1_sreal, 1_sreal, 1_sreal), "gradingRatio", "Per-axis geometric grading ratio [rx, ry, rz]. r > 1: points dense at start; r < 1: points dense at end; r = 1: uniform."))
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
    checkGradingRatio();

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
    checkGradingRatio();

    setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);

    Inherit1::reinit();
}

void RegularGridTopology::checkGradingRatio()
{
    auto r = sofa::helper::getWriteAccessor(d_gradingRatio);
    for (sofa::Size i = 0; i < 3; ++i)
    {
        if (r[i] <= 0)
        {
            msg_error() << "gradingRatio[" << i << "] = " << r[i]
                << " is invalid (must be > 0). Falling back to 1 (uniform) for this axis.";
            r[i] = 1_sreal;
        }
    }
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
    const auto& r = d_gradingRatio.getValue();
    const auto& n = d_n.getValue();
    // dx, dy, dz are the uniform step vectors; their component magnitude * (n-1) gives the full axis length.
    // For each axis, scale the step direction by the graded fraction instead of the uniform integer index.
    const SReal fx = n[0] > 1 ? gradedCoord(i, n[0], dx[0] * (n[0] - 1), r[0]) / dx[0] : 0;
    const SReal fy = n[1] > 1 ? gradedCoord(j, n[1], dy[1] * (n[1] - 1), r[1]) / dy[1] : 0;
    const SReal fz = n[2] > 1 ? gradedCoord(k, n[2], dz[2] * (n[2] - 1), r[2]) / dz[2] : 0;
    return d_p0.getValue() + dx * fx + dy * fy + dz * fz;
}

RegularGridTopology::Index RegularGridTopology::findPoint(const type::Vec3& position) const
{
    const type::Vec3 p0 = d_p0.getValue();
    const type::Vec3i n = d_n.getValue();
    const type::Vec3& r = d_gradingRatio.getValue();
    const type::Vec3 p = position - p0;

    const int ix = int(std::round(gradedIndex(p[0], n[0], dx[0] * (n[0] - 1), r[0])));
    const int iy = int(std::round(gradedIndex(p[1], n[1], dy[1] * (n[1] - 1), r[1])));
    const int iz = int(std::round(gradedIndex(p[2], n[2], dz[2] * (n[2] - 1), r[2])));

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
    const type::Vec3& r = d_gradingRatio.getValue();
    const type::Vec3 p = position - p0;

    const int ix = int(std::round(gradedIndex(p[0], n[0], dx[0] * (n[0] - 1), r[0])));
    const int iy = int(std::round(gradedIndex(p[1], n[1], dy[1] * (n[1] - 1), r[1])));
    const int iz = int(std::round(gradedIndex(p[2], n[2], dz[2] * (n[2] - 1), r[2])));

    if (ix < 0 || iy < 0 || iz < 0)
        return InvalidID;
    if (ix > n[0] || iy > n[1] || iz > n[2])
        return InvalidID;

    const auto node_index = getIndex(ix, iy, iz);

    // Check that the queried position lies within epsilon * (first-cell size) of the found node.
    const auto node_position = type::Vec3(
        gradedCoord(ix, n[0], dx[0] * (n[0] - 1), r[0]),
        gradedCoord(iy, n[1], dy[1] * (n[1] - 1), r[1]),
        gradedCoord(iz, n[2], dz[2] * (n[2] - 1), r[2]));
    const auto m = epsilon * dx[0];
    const auto e = p - node_position;

    if (e[0]*e[0] + e[1]*e[1] + e[2]*e[2] > m*m)
        return InvalidID;

    return node_index;
}

/// Compute the fractional cell index along each axis for a point p relative to the grid origin.
/// Uses the uniform formula when gradingRatio == [1,1,1] to avoid any overhead.
void RegularGridTopology::computeCellIndex(const type::Vec3& p, SReal& x, SReal& y, SReal& z) const
{
    const auto& r = d_gradingRatio.getValue();
    if (std::abs(r[0]-1) < 1e-10 && std::abs(r[1]-1) < 1e-10 && std::abs(r[2]-1) < 1e-10)
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
        x = gradedIndex(p*dx/dx_len, n[0], dx_len*(n[0]-1), r[0]);
        y = gradedIndex(p*dy/dy_len, n[1], dy_len*(n[1]-1), r[1]);
        z = gradedIndex(p*dz/dz_len, n[2], dz_len*(n[2]-1), r[2]);
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
