#ifndef AABB_H
#define AABB_H

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ray.h"

class aabb {
public:
    __device__ aabb() {}
    __device__ aabb(const vec3& a, const vec3& b) { minimum = a; maximum = b; }

    __device__ vec3 min() const { return minimum; }
    __device__ vec3 max() const { return maximum; }

    __device__ bool hit(const ray& r, double t_min, double t_max) const {
        for (int a = 0; a < 3; a++) {
            auto t0 = fmin((minimum[a] - r.origin()[a]) / r.direction()[a],
                (maximum[a] - r.origin()[a]) / r.direction()[a]);
            auto t1 = fmax((minimum[a] - r.origin()[a]) / r.direction()[a],
                (maximum[a] - r.origin()[a]) / r.direction()[a]);
            t_min = fmax(t0, t_min);
            t_max = fmin(t1, t_max);
            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    vec3 minimum;
    vec3 maximum;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(fmin(box0.min().x(), box1.min().x()),
        fmin(box0.min().y(), box1.min().y()),
        fmin(box0.min().z(), box1.min().z()));

    vec3 big(fmax(box0.max().x(), box1.max().x()),
        fmax(box0.max().y(), box1.max().y()),
        fmax(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}

#endif