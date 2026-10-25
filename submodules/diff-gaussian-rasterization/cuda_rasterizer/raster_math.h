#pragma once

#include <cassert>
#include <cmath>
#include <type_traits>

#if defined(__CUDACC__)
#define RASTER_MATH_HD __host__ __device__ __forceinline__
#else
#define RASTER_MATH_HD inline
#endif

// Float-only algebra used by the rasterizer. Matrices are column-major:
// m[column][row]. Keep packed tensor layout and the original evaluation order.
namespace raster_math
{
struct vec3
{
    float x, y, z;
    RASTER_MATH_HD vec3() {}
    RASTER_MATH_HD explicit vec3(float s) : x(s), y(s), z(s) {}
    RASTER_MATH_HD vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    RASTER_MATH_HD float& operator[](int i)
    {
        assert(i >= 0 && i < 3);
        return i == 0 ? x : (i == 1 ? y : z);
    }
    RASTER_MATH_HD const float& operator[](int i) const
    {
        assert(i >= 0 && i < 3);
        return i == 0 ? x : (i == 1 ? y : z);
    }
    RASTER_MATH_HD vec3& operator+=(vec3 v)
    {
        x += v.x; y += v.y; z += v.z; return *this;
    }
    RASTER_MATH_HD vec3& operator+=(float s)
    {
        x += s; y += s; z += s; return *this;
    }
    RASTER_MATH_HD vec3& operator*=(float s)
    {
        x *= s; y *= s; z *= s; return *this;
    }
};

struct vec4
{
    float x, y, z, w;
    RASTER_MATH_HD vec4() {}
    RASTER_MATH_HD explicit vec4(float s) : x(s), y(s), z(s), w(s) {}
    RASTER_MATH_HD vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    RASTER_MATH_HD vec4& operator+=(vec4 v)
    {
        x += v.x; y += v.y; z += v.z; w += v.w; return *this;
    }
    RASTER_MATH_HD vec4& operator/=(float s)
    {
        x /= s; y /= s; z /= s; w /= s; return *this;
    }
};

RASTER_MATH_HD vec3 operator+(vec3 a, vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
RASTER_MATH_HD vec3 operator-(vec3 a, vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
RASTER_MATH_HD vec3 operator*(vec3 a, vec3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
RASTER_MATH_HD vec3 operator*(vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
RASTER_MATH_HD vec3 operator*(float s, vec3 a) { return {s*a.x, s*a.y, s*a.z}; }
RASTER_MATH_HD vec3 operator/(vec3 a, float s) { return {a.x/s, a.y/s, a.z/s}; }
RASTER_MATH_HD vec4 operator-(vec4 a, vec4 b) { return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
RASTER_MATH_HD vec4 operator*(vec4 a, float s) { return {a.x*s, a.y*s, a.z*s, a.w*s}; }
RASTER_MATH_HD vec4 operator/(vec4 a, float s) { return {a.x/s, a.y/s, a.z/s, a.w/s}; }

RASTER_MATH_HD float dot(vec3 a, vec3 b)
{
    const vec3 products(a.x*b.x, a.y*b.y, a.z*b.z);
    return (products.x + products.y) + products.z;
}
RASTER_MATH_HD float dot(vec4 a, vec4 b)
{
    // The pairwise grouping is also used by quaternion normalization backward.
    const vec4 products(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
    return (products.x + products.y) + (products.z + products.w);
}
RASTER_MATH_HD float length(vec3 v) { return ::sqrtf(dot(v, v)); }
RASTER_MATH_HD float length(vec4 v) { return ::sqrtf(dot(v, v)); }
RASTER_MATH_HD vec3 max(vec3 a, float b)
{
    // Comparison form preserves the original NaN and signed-zero behavior.
    return {a.x < b ? b : a.x, a.y < b ? b : a.y, a.z < b ? b : a.z};
}

struct mat3
{
    vec3 columns[3];
    RASTER_MATH_HD mat3() {}
    RASTER_MATH_HD explicit mat3(float diagonal)
        : columns{vec3(diagonal, 0, 0), vec3(0, diagonal, 0), vec3(0, 0, diagonal)} {}
    RASTER_MATH_HD mat3(vec3 a, vec3 b, vec3 c) : columns{a, b, c} {}
    RASTER_MATH_HD mat3(float a, float b, float c, float d, float e, float f,
                        float g, float h, float i)
        : columns{vec3(a, b, c), vec3(d, e, f), vec3(g, h, i)} {}
    RASTER_MATH_HD vec3& operator[](int i)
    {
        assert(i >= 0 && i < 3);
        return columns[i];
    }
    RASTER_MATH_HD const vec3& operator[](int i) const
    {
        assert(i >= 0 && i < 3);
        return columns[i];
    }
};

RASTER_MATH_HD mat3 transpose(const mat3& m)
{
    return {m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2]};
}
RASTER_MATH_HD mat3 operator*(float s, const mat3& m)
{
    return {m[0]*s, m[1]*s, m[2]*s};
}
RASTER_MATH_HD mat3 operator*(const mat3& a, const mat3& b)
{
    mat3 result;
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            result[column][row] = (a[0][row]*b[column][0] + a[1][row]*b[column][1])
                                  + a[2][row]*b[column][2];
    return result;
}

static_assert(sizeof(vec3) == 3*sizeof(float) && alignof(vec3) == alignof(float), "Packed vec3 required");
static_assert(sizeof(vec4) == 4*sizeof(float) && alignof(vec4) == alignof(float), "Packed vec4 required");
static_assert(sizeof(mat3) == 9*sizeof(float) && alignof(mat3) == alignof(float), "Packed mat3 required");
static_assert(std::is_standard_layout<vec3>::value && std::is_trivially_copyable<vec3>::value, "Tensor-compatible vec3 required");
static_assert(std::is_standard_layout<vec4>::value && std::is_trivially_copyable<vec4>::value, "Tensor-compatible vec4 required");
} // namespace raster_math

#undef RASTER_MATH_HD
