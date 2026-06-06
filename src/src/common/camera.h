#pragma once
#include "math.h"

struct CameraParams {
    Vec3  position;
    Vec3  look_at;
    Vec3  up;
    float vfov_deg;   // vertical field of view in degrees

    // Reserved for thin-lens depth-of-field (Phase 4 reach):
    float aperture;    // 0 = pinhole
    float focus_dist;  // used only when aperture > 0
};

// Pre-computed per-frame basis vectors derived from CameraParams.
// Passing this struct to kernels avoids re-deriving the basis on every thread.
struct CameraFrame {
    Vec3 origin;
    Vec3 lower_left;   // world-space position of the (0,0) corner of the image plane
    Vec3 horizontal;   // full-width vector across the image plane
    Vec3 vertical;     // full-height vector up the image plane
    Vec3 lens_u;       // camera right unit vector  (for thin-lens DoF sampling)
    Vec3 lens_v;       // camera up unit vector     (for thin-lens DoF sampling)
};

// Derives the camera basis from params. width/height are the output resolution.
HD CameraFrame make_camera_frame(const CameraParams& p, int width, int height) {
    float aspect   = float(width) / float(height);
    float half_h   = tanf(p.vfov_deg * PT_HALF_PI / 180.0f);  // tan(vfov/2)
    float vp_h     = 2.0f * half_h;
    float vp_w     = vp_h * aspect;

    Vec3 w = normalize(p.position - p.look_at);  // points from scene toward camera
    Vec3 u = normalize(cross(p.up, w));
    Vec3 v = cross(w, u);

    Vec3 horizontal = vp_w * u;
    Vec3 vertical   = vp_h * v;
    Vec3 lower_left = p.position - horizontal * 0.5f - vertical * 0.5f - w;

    return { p.position, lower_left, horizontal, vertical, u, v };
}

// Generates a ray through pixel (px, py) with sub-pixel offsets (ou, ov) in [0,1).
// ou/ov = 0.5 gives the pixel centre; jitter them for anti-aliasing.
// Image convention: py=0 is the TOP row (v is flipped here so the camera looks
// right-side-up without any buffer flip at output time).
//
// Thin-lens depth of field: when aperture > 0, lu/lv are uniform-disk samples
// in [-1, 1] used to offset the ray origin on the lens. focus_dist sets the
// depth at which the focal plane lies (objects at that depth are sharp).
HD Ray camera_ray(const CameraFrame& cam, int px, int py,
                  float ou, float ov, int width, int height,
                  float lu, float lv, float aperture, float focus_dist)
{
    float u = (float(px) + ou) / float(width);
    float v = 1.0f - (float(py) + ov) / float(height);  // flip y
    Vec3 dir = cam.lower_left + u * cam.horizontal + v * cam.vertical - cam.origin;

    if (aperture <= 0.f) return make_ray(cam.origin, dir);

    // Focal point: where this pixel's ray pierces the focal plane.
    Vec3 focus_pt = cam.origin + normalize(dir) * focus_dist;
    // Offset origin across the lens disk, then aim at the same focal point.
    Vec3 lens_pt  = cam.origin + (aperture * 0.5f) * (lu * cam.lens_u + lv * cam.lens_v);
    return make_ray(lens_pt, focus_pt - lens_pt);
}
