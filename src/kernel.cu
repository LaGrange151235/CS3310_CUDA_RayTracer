#include <iostream>
#include <cmath>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "aarect.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "texture.h"

// stb image tools
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

// sdl gui tools
#include "qbImage.h"

#define RND (curand_uniform(&local_rand_state))
#define one_degree 0.017453293
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);

    vec3 background(0, 0, 0);
    vec3 emitted_stack[8];
    vec3 ray_color_stack[8];
    vec3 attenuation_stack[8];
    int index = 0;

    for (int i = 0; i < 6; i++) {
        hit_record rec;
        if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray_color_stack[i] = background;
            index = i;
            break;
        }

        ray scattered;
        vec3 attenuation;
        vec3 emitted = rec.mat_ptr->emitted(rec, 0, 0, vec3(0, 0, 0));
        emitted_stack[i] = emitted;
        if (!rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
            ray_color_stack[i] = emitted;
            index = i;
            break;
        }
        cur_ray = scattered;
        attenuation_stack[i] = attenuation;
    }

    for (int i = index - 1; i >= 0; --i) {
        ray_color_stack[i] = emitted_stack[i] + attenuation_stack[i] * ray_color_stack[i + 1];
    }
    return ray_color_stack[0];
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state, float* r_channel, float* g_channel, float* b_channel) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    int channel_index = (max_y - j - 1) * max_x + i;
    r_channel[channel_index] = col[0];
    g_channel[channel_index] = col[1];
    b_channel[channel_index] = col[2];
}

__global__ void change_camera(int nx, int ny, int change, camera** d_camera) {
    camera* my_camera = *d_camera;
    vec3 now_lookfrom = my_camera->origin;
    vec3 now_lookat = my_camera->target;
    vec3 now_vup = my_camera->upvec;
    float dist_to_focus = (now_lookfrom - now_lookat).length();
    float aperture = 0.1;

    vec3 dir = (now_lookat - now_lookfrom);
    vec3 left = cross(now_vup, dir);
    dir.make_unit_vector();
    left.make_unit_vector();


    if (change == SDLK_UP)
    {
        now_lookfrom += dir;
        now_lookat += dir;
    }
    if (change == SDLK_DOWN)
    {
        now_lookfrom -= dir;
        now_lookat -= dir;
    }
    if (change == SDLK_LEFT)
    {
        now_lookfrom += left;
        now_lookat += left;
    }
    if (change == SDLK_RIGHT)
    {
        now_lookfrom -= left;
        now_lookat -= left;
    }
    if (change == SDLK_w)
    {
        vec3 down = -tan(5 * one_degree) * now_vup;
        vec3 new_dir = dir + down;
        new_dir.make_unit_vector();
        now_lookat = new_dir * dist_to_focus + now_lookfrom;
        now_vup = cross(dir, left);
    }
    if (change == SDLK_s)
    {
        vec3 down = tan(5 * one_degree) * now_vup;
        vec3 new_dir = dir + down;
        new_dir.make_unit_vector();
        now_lookat = new_dir * dist_to_focus + now_lookfrom;
        now_vup = cross(dir, left);
    }
    if (change == SDLK_a)
    {
        vec3 roll = tan(5 * one_degree) * left;
        now_vup = now_vup + roll;
    }
    if (change == SDLK_d)
    {
        vec3 roll = tan(5 * one_degree) * left;
        now_vup = now_vup - roll;
    }
    if (change == SDLK_EQUALS)
    {
        dist_to_focus += 1;
        now_lookat = dist_to_focus * dir + now_lookfrom;
    }
    if (change == SDLK_MINUS)
    {
        if (dist_to_focus - 1 > 1)
            dist_to_focus -= 1;
        now_lookat = dist_to_focus * dir + now_lookfrom;
    }

    *d_camera = new camera(
        now_lookfrom,
        now_lookat,
        now_vup,
        60.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

__global__ void create_camera_0(int nx, int ny, camera** d_camera) {
    vec3 lookfrom(0, -50, -30);
    vec3 lookat(0, -50, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    *d_camera = new camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        60.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

__global__ void create_camera_1(int nx, int ny, camera** d_camera) {
    vec3 lookfrom(0, 10, 10);
    vec3 lookat(0, 5, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    *d_camera = new camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        60.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

__global__ void create_camera_2(int nx, int ny, camera** d_camera) {
    vec3 lookfrom(0, 1, 5);
    vec3 lookat(0, 1, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    *d_camera = new camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        60.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

__global__ void create_camera_3(int nx, int ny, camera** d_camera) {
    vec3 lookfrom(6, 6, 6);
    vec3 lookat(0, 1, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    *d_camera = new camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        60.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

__global__ void create_world_0(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* image = new ImageTexture(dataPtr, 4096, 4096);
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(128.0 / 255, 0, 128.0 / 255));
        Texture* color2 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        Texture* color3 = new ConstantTexture(vec3(222.0 / 255, 84.0 / 255, 36.0 / 255));
        Texture* color4 = new ConstantTexture(vec3(7.0 / 255, 115.0 / 255, 238.0 / 255));
        Texture* color5 = new ConstantTexture(vec3(61.0 / 255, 64.0 / 255, 71.0 / 255));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -50.0, 0), 100, new diffuse_light(image));
        d_list[i++] = new sphere(vec3(0, -50, 0), 1.0, new diffuse_light(color3));
        d_list[i++] = new sphere(vec3(-6, -50, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-3, -50, 0), 1.0, new lambertian(color1));
        d_list[i++] = new sphere(vec3(3, -50, 0), 1.0, new lambertian(checker));
        d_list[i++] = new sphere(vec3(6, -50, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void create_world_1(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* image = new ImageTexture(dataPtr, 4096, 4096);
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(128.0 / 255, 0, 128.0 / 255));
        Texture* color2 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        Texture* color3 = new ConstantTexture(vec3(222.0 / 255, 84.0 / 255, 36.0 / 255));
        Texture* color4 = new ConstantTexture(vec3(7.0 / 255, 115.0 / 255, 238.0 / 255));
        Texture* color5 = new ConstantTexture(vec3(61.0 / 255, 64.0 / 255, 71.0 / 255));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(checker));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new diffuse_light(color3));
        d_list[i++] = new sphere(vec3(-6, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-3, 1, 0), 1.0, new lambertian(color1));
        d_list[i++] = new sphere(vec3(3, 1, 0), 1.0, new lambertian(image));
        d_list[i++] = new sphere(vec3(6, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        d_list[i++] = new xz_rect(-8, 8, -2, 2, 5, new diffuse_light(color2));
        d_list[i++] = new xy_rect(-8, 8, 0, 5, -2, new lambertian(color4));
        d_list[i++] = new yz_rect(0, 5, -2, 2, -8, new lambertian(color4));
        d_list[i++] = new yz_rect(0, 5, -2, 2, 8, new lambertian(color4));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void create_world_2(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* image = new ImageTexture(dataPtr, 4096, 4096);
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        Texture* color2 = new ConstantTexture(vec3(128.0 / 255, 0, 128.0 / 255));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(checker));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new lambertian(color2));
        d_list[i++] = new xz_rect(-2, 2, -2, 2, 4, new diffuse_light(color1));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void create_world_3(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* image = new ImageTexture(dataPtr, 4096, 4096);
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(checker));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        d_list[i++] = new xz_rect(-2, 2, -2, 2, 4, new diffuse_light(color1));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void create_world_4(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* image = new ImageTexture(dataPtr, 4096, 4096);
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(checker));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new xz_rect(-2, 2, -2, 2, 4, new diffuse_light(color1));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void create_world_5(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* image = new ImageTexture(dataPtr, 4096, 4096);
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(checker));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new lambertian(image));
        d_list[i++] = new xz_rect(-2, 2, -2, 2, 4, new diffuse_light(color1));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void create_world_6(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(checker));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new diffuse_light(color1));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void create_world_7(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char* dataPtr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* image = new ImageTexture(dataPtr, 4096, 4096);
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)), new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        Texture* color1 = new ConstantTexture(vec3(0.9, 0.9, 0.9));
        Texture* color2 = new ConstantTexture(vec3(7.0 / 255, 115.0 / 255, 238.0 / 255));
        curandState local_rand_state = *rand_state;
        int i = 1;
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(checker));
        d_list[i++] = new xz_rect(-2, 2, -2, 2, 5, new diffuse_light(color1));
        d_list[i++] = new xz_rect(-1, 1, -1, 1, 2, new lambertian(color2));
        d_list[i++] = new xy_rect(-1, 1, 0, 2, -1, new lambertian(color2));
        d_list[i++] = new xy_rect(-1, 1, 0, 2, 1, new lambertian(color2));
        d_list[i++] = new yz_rect(0, 2, -1, 1, -1, new lambertian(color2));
        d_list[i++] = new yz_rect(0, 2, -1, 1, 1, new lambertian(color2));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void free_world_0(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 6; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

__global__ void free_world_1(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 6; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete ((xz_rect*)d_list[6])->mp;
    delete d_list[6];
    delete ((xy_rect*)d_list[7])->mp;
    delete d_list[7];
    for (int i = 8; i < 10; i++) {
        delete ((yz_rect*)d_list[i])->mp;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

__global__ void free_world_2(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 2; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete ((xz_rect*)d_list[2])->mp;
    delete d_list[2];
    delete* d_world;
    delete* d_camera;
}

__global__ void free_world_3(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 2; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

__global__ void free_world_4(hitable** d_list, hitable** d_world, camera** d_camera) {
    delete ((sphere*)d_list[0])->mat_ptr;
    delete d_list[0];
    for (int i = 1; i < 3; i++) {
        delete ((xz_rect*)d_list[i])->mp;
        delete d_list[i];
    }
    for (int i = 3; i < 5; i++) {
        delete ((xy_rect*)d_list[i])->mp;
        delete d_list[i];
    }
    for (int i = 5; i < 7; i++) {
        delete ((yz_rect*)d_list[i])->mp;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

int main(int argv, char** args) {
    // parameter initialize
    int nx = 800;
    int ny = 600;
    int ns = 4;
    int tx = 16;
    int ty = 16;
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    int scene = 0;
    int view = 0;

    std::cerr << "Which scene do you want to generate?(0-7):";
    std::cin >> scene;
    scene = scene % 8;
    std::cerr << "How many samples for 1 pixel do you want?(4-1024):";
    std::cin >> ns;
    if (scene == 0)
    {
        std::cerr << "Which view do you want to see?(0-2):";
        std::cin >> view;
        view = view % 3;
    }
    if (ns < 4)
    {
        ns = 4;
    }
    else {
        ns = ns % 1025;
        if (ns < 4)
        {
            ns = 4;
        }
    }
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";


    // check GPU info
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    std::cerr << "Device 0: " << devProp.name << "\n";


    // build GUI window
    qbImage* m_image = new qbImage();
    bool isRunning = true;
    SDL_Event event;
    SDL_Window* pWindow = NULL;
    SDL_Renderer* pRenderer = NULL;
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        return false;
    }
    pWindow = SDL_CreateWindow("CUDA_RTRT", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, nx, ny, SDL_WINDOW_SHOWN);
    if (pWindow != NULL)
    {
        pRenderer = SDL_CreateRenderer(pWindow, -1, 0);
        m_image->Initialize(nx, ny, pRenderer);
        SDL_SetRenderDrawColor(pRenderer, 255, 255, 255, 255);
        SDL_RenderClear(pRenderer);
    }


    // allocate FB
    size_t channel_size = num_pixels * sizeof(float);
    float* r_channel;
    float* g_channel;
    float* b_channel;
    checkCudaErrors(cudaMallocManaged((void**)&r_channel, channel_size));
    checkCudaErrors(cudaMallocManaged((void**)&g_channel, channel_size));
    checkCudaErrors(cudaMallocManaged((void**)&b_channel, channel_size));


    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));


    // we need that 2nd random state to be initialized for the world creation
    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // read texture image
    int iw, ih, n;
    unsigned char* idata = NULL;
    unsigned char* dataPtr = NULL;
    if (scene == 0)
    {
        if (view == 0)
        {
            idata = stbi_load("./pic/view1.jpg", &iw, &ih, &n, 0);
        }
        if (view == 1)
        {
            idata = stbi_load("./pic/view2.jpg", &iw, &ih, &n, 0);
        }
        if (view == 2)
        {
            idata = stbi_load("./pic/view3.jpg", &iw, &ih, &n, 0);
        }
        int ow = 4096;
        int oh = 4096;
        auto* odata = (unsigned char*)malloc(ow * oh * n);
        stbir_resize(idata, iw, ih, 0, odata, ow, oh, 0, STBIR_TYPE_UINT8, n, STBIR_ALPHA_CHANNEL_NONE, 0,
            STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
            STBIR_FILTER_BOX, STBIR_FILTER_BOX,
            STBIR_COLORSPACE_SRGB, nullptr
        );
        checkCudaErrors(cudaMalloc((void**)&dataPtr, ow * oh * 3 * sizeof(unsigned char)));
        checkCudaErrors(cudaMemcpy(dataPtr, odata, oh * ow * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }
    if (scene == 1 || scene == 5)
    {
        idata = stbi_load("./pic/logo.jpg", &iw, &ih, &n, 0);
        int ow = 4096;
        int oh = 4096;
        auto* odata = (unsigned char*)malloc(ow * oh * n);
        stbir_resize(idata, iw, ih, 0, odata, ow, oh, 0, STBIR_TYPE_UINT8, n, STBIR_ALPHA_CHANNEL_NONE, 0,
            STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
            STBIR_FILTER_BOX, STBIR_FILTER_BOX,
            STBIR_COLORSPACE_SRGB, nullptr
        );
        checkCudaErrors(cudaMalloc((void**)&dataPtr, ow * oh * 3 * sizeof(unsigned char)));
        checkCudaErrors(cudaMemcpy(dataPtr, odata, oh * ow * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }



    // make our world of hitables & the camera
    hitable** d_list;
    int num_hitables = 6;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    if (scene == 0)
    {
        create_world_0 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    if (scene == 1)
    {
        create_world_1 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    if (scene == 2)
    {
        create_world_2 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    if (scene == 3)
    {
        create_world_3 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    if (scene == 4)
    {
        create_world_4 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    if (scene == 5)
    {
        create_world_5 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    if (scene == 6)
    {
        create_world_6 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    if (scene == 7)
    {
        create_world_7 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, dataPtr);
    }
    //checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    if (scene == 0)
    {
        create_camera_0 << <1, 1 >> > (nx, ny, d_camera);
    }
    if (scene == 1)
    {
        create_camera_1 << <1, 1 >> > (nx, ny, d_camera);
    }
    if (scene == 2 || scene == 3 || scene == 4 || scene == 5 || scene == 6)
    {
        create_camera_2 << <1, 1 >> > (nx, ny, d_camera);
    }
    if (scene == 7)
    {
        create_camera_3 << <1, 1 >> > (nx, ny, d_camera);
    }
    //checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Render our buffer
    clock_t start, stop;
    start = clock();
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    //checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (nx, ny, ns, d_camera, d_world, d_rand_state, r_channel, g_channel, b_channel);
    //checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    m_image->m_rChannel = r_channel;
    m_image->m_gChannel = g_channel;
    m_image->m_bChannel = b_channel;
    m_image->Display();
    SDL_RenderPresent(pRenderer);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "fps: " << 1 / timer_seconds << ", time:" << timer_seconds << "\n";

    while (isRunning)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_QUIT)
            {
                isRunning = false;
            }
            else if (event.type == SDL_KEYDOWN)
            {
                // Render our buffer
                clock_t start, stop;
                start = clock();
                change_camera << <1, 1 >> > (nx, ny, event.key.keysym.sym, d_camera);
                //checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());
                dim3 blocks(nx / tx + 1, ny / ty + 1);
                dim3 threads(tx, ty);
                render << <blocks, threads >> > (nx, ny, ns, d_camera, d_world, d_rand_state, r_channel, g_channel, b_channel);
                //checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());
                m_image->m_rChannel = r_channel;
                m_image->m_gChannel = g_channel;
                m_image->m_bChannel = b_channel;
                m_image->Display();
                SDL_RenderPresent(pRenderer);
                stop = clock();
                double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
                std::cerr << "fps: " << 1 / timer_seconds << ", time:" << timer_seconds << "\n";
            }
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    if (scene == 0)
    {
        free_world_0 << <1, 1 >> > (d_list, d_world, d_camera);
        delete dataPtr;
    }
    if (scene == 1)
    {
        free_world_1 << <1, 1 >> > (d_list, d_world, d_camera);
    }
    if (scene == 2 || scene == 3 || scene == 4 || scene == 5)
    {
        free_world_2 << <1, 1 >> > (d_list, d_world, d_camera);
        if (scene == 5)
        {
            delete dataPtr;
        }
    }
    if (scene == 6)
    {
        free_world_3 << <1, 1 >> > (d_list, d_world, d_camera);
    }
    if (scene == 7)
    {
        free_world_4 << <1, 1 >> > (d_list, d_world, d_camera);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(r_channel));
    checkCudaErrors(cudaFree(g_channel));
    checkCudaErrors(cudaFree(b_channel));
    cudaDeviceReset();
}
