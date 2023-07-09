#ifndef AARECT_H
#define AARECT_H

#include "hitable.h"

class xy_rect : public hitable {
public:
    __host__ __device__ xy_rect() {}
    __host__ __device__ xy_rect(double _x0, double _x1, double _y0, double _y1, double _k, material* m) : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(m) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
public:
    material* mp;
    double x0, x1, y0, y1, k;
};

__device__ bool xy_rect::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < tmin || t > tmax)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0.0f, 0.0f, 1.0f);
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool xy_rect::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
    return true;
}

class xz_rect : public hitable {
public:
    __host__ __device__ xz_rect() {}
    __host__ __device__ xz_rect(double _x0, double _x1, double _z0, double _z1, double _k, material* m) : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(m) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
public:
    material* mp;
    double x0, x1, z0, z1, k;
};

__device__ bool xz_rect::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < tmin || t > tmax)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0.0f, 0.0f, 1.0f);
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool xz_rect::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = aabb(vec3(x0, z0, k - 0.0001), vec3(x1, z1, k + 0.0001));
    return true;
}

class yz_rect : public hitable {
public:
    __host__ __device__ yz_rect() {}
    __host__ __device__ yz_rect(double _y0, double _y1, double _z0, double _z1, double _k, material* m) : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(m) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
public:
    material* mp;
    double y0, y1, z0, z1, k;
};

__device__ bool yz_rect::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < tmin || t > tmax)
        return false;
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0.0f, 0.0f, 1.0f);
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool yz_rect::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = aabb(vec3(y0, z0, k - 0.0001), vec3(y1, z1, k + 0.0001));
    return true;
}

#endif