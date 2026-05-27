#include "loader.h"
#include "mesh_loader.h"
#include "bvh/bvh.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Returns the directory portion of a file path (with trailing separator).
static std::string dir_of(const std::string& path) {
    size_t last = path.find_last_of("/\\");
    return (last != std::string::npos) ? path.substr(0, last + 1) : "";
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static Vec3 parse_vec3(const json& j) {
    return { j[0].get<float>(), j[1].get<float>(), j[2].get<float>() };
}

static Vec3 parse_vec3_or(const json& obj, const std::string& key, Vec3 def) {
    return obj.contains(key) ? parse_vec3(obj[key]) : def;
}

static float parse_float_or(const json& obj, const std::string& key, float def) {
    return obj.contains(key) ? obj[key].get<float>() : def;
}

static int parse_int_or(const json& obj, const std::string& key, int def) {
    return obj.contains(key) ? obj[key].get<int>() : def;
}

// ── Sub-parsers ───────────────────────────────────────────────────────────────

static CameraParams parse_camera(const json& j) {
    CameraParams c{};
    c.position   = parse_vec3(j["position"]);
    c.look_at    = parse_vec3(j["look_at"]);
    c.up         = parse_vec3_or(j, "up", { 0.f, 1.f, 0.f });
    c.vfov_deg   = parse_float_or(j, "vfov_deg", 45.f);
    c.aperture   = parse_float_or(j, "aperture",   0.f);
    c.focus_dist = parse_float_or(j, "focus_dist", 1.f);
    return c;
}

static RenderConfig parse_render(const json& j) {
    RenderConfig r{};
    r.width         = parse_int_or(j, "width",         800);
    r.height        = parse_int_or(j, "height",        600);
    r.spp           = parse_int_or(j, "spp",           256);
    r.max_depth     = parse_int_or(j, "max_depth",     20);
    r.rr_min_depth  = parse_int_or(j, "rr_min_depth",  3);
    r.firefly_clamp = parse_float_or(j, "firefly_clamp", 0.f);
    r.output_path   = j.contains("output") ? j["output"].get<std::string>() : "render.png";

    // background: "sky" string → gradient; array → flat colour
    if (j.contains("background")) {
        const auto& bg = j["background"];
        if (bg.is_string() && bg.get<std::string>() == "sky") {
            r.bg_mode = BackgroundMode::Sky;
        } else {
            r.bg_mode    = BackgroundMode::Color;
            r.background = parse_vec3(bg);
        }
    }

    if (j.contains("tonemapping")) {
        std::string tm = j["tonemapping"].get<std::string>();
        if      (tm == "gamma")    r.tonemap = TonemapMode::Gamma;
        else if (tm == "reinhard") r.tonemap = TonemapMode::Reinhard;
        else                       r.tonemap = TonemapMode::ACES;
    }

    if (j.contains("denoise")) {
        const auto& d = j["denoise"];
        if (d.is_boolean()) {
            r.denoise.enabled = d.get<bool>();
        } else {
            r.denoise.enabled       = d.contains("enabled") ? d["enabled"].get<bool>() : true;
            r.denoise.sigma_r       = parse_float_or(d, "sigma_r", 0.12f);
            r.denoise.atrous_passes = parse_int_or(d,   "atrous_passes", 5);
        }
    }
    return r;
}

static Material parse_material(const json& j) {
    Material m{};
    m.albedo    = parse_vec3_or(j, "albedo",   { 0.8f, 0.8f, 0.8f });
    m.emission  = parse_vec3_or(j, "emission", { 0.f,  0.f,  0.f  });
    m.ior       = parse_float_or(j, "ior",       1.5f);
    m.roughness = parse_float_or(j, "roughness", 0.f);
    m.metallic  = parse_float_or(j, "metallic",  0.f);

    std::string type = j["type"].get<std::string>();
    if      (type == "diffuse")           m.type = MaterialType::Diffuse;
    else if (type == "specular_ideal")    m.type = MaterialType::SpecularIdeal;
    else if (type == "dielectric_ideal")  m.type = MaterialType::DielectricIdeal;
    else if (type == "emissive")          m.type = MaterialType::Emissive;
    else if (type == "rough_dielectric")  m.type = MaterialType::GGXDielectric;
    else throw std::runtime_error("Unknown material type: " + type);

    return m;
}

static Shape parse_shape(const json& j) {
    Shape s{};
    s.material_id = j["material"].get<int>();

    std::string type = j["type"].get<std::string>();
    if (type == "sphere") {
        s.type             = ShapeType::Sphere;
        s.sphere.center    = parse_vec3(j["center"]);
        s.sphere.radius    = j["radius"].get<float>();
    } else if (type == "cylinder") {
        s.type                  = ShapeType::Cylinder;
        s.cylinder.center       = parse_vec3(j["center"]);
        s.cylinder.axis         = normalize(parse_vec3(j["axis"]));
        s.cylinder.radius       = j["radius"].get<float>();
        s.cylinder.half_height  = j["half_height"].get<float>();
    } else if (type == "disk") {
        s.type           = ShapeType::Disk;
        s.disk.center    = parse_vec3(j["center"]);
        s.disk.normal    = normalize(parse_vec3(j["normal"]));
        s.disk.radius    = j["radius"].get<float>();
    } else if (type == "cube") {
        s.type                = ShapeType::Cube;
        s.cube.center         = parse_vec3(j["center"]);
        s.cube.half_extents   = parse_vec3(j["half_extents"]);
    } else if (type == "plane") {
        s.type           = ShapeType::Plane;
        s.plane.center   = parse_vec3(j["center"]);
        s.plane.normal   = normalize(parse_vec3(j["normal"]));
    } else {
        throw std::runtime_error("Unknown shape type: " + type);
    }
    return s;
}

static Light parse_light(const json& j) {
    Light l{};
    std::string type = j["type"].get<std::string>();

    if (type == "point") {
        l.type                = LightType::Point;
        l.point.position      = parse_vec3(j["position"]);
        l.point.color         = parse_vec3_or(j, "color", { 1.f, 1.f, 1.f });
        l.point.intensity     = parse_float_or(j, "intensity", 1.f);
    } else if (type == "directional") {
        l.type                    = LightType::Directional;
        l.directional.direction   = normalize(parse_vec3(j["direction"]));
        l.directional.color       = parse_vec3_or(j, "color", { 1.f, 1.f, 1.f });
        l.directional.intensity   = parse_float_or(j, "intensity", 1.f);
    } else if (type == "area") {
        l.type          = LightType::Area;
        l.area.shape_id = j["shape_id"].get<int>();
    } else {
        throw std::runtime_error("Unknown light type: " + type);
    }
    return l;
}

// ── Entry point ───────────────────────────────────────────────────────────────

SceneData load_scene(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open scene file: " + path);

    json j = json::parse(f);
    SceneData scene{};
    std::string scene_dir = dir_of(path);

    scene.camera = parse_camera(j["camera"]);
    scene.config = parse_render(j.contains("render") ? j["render"] : json::object());

    for (const auto& m : j["materials"])
        scene.materials.push_back(parse_material(m));

    // Parse shapes: analytic shapes pushed to scene.shapes;
    // mesh shapes load an OBJ and append triangles to scene.triangles.
    for (const auto& s : j["shapes"]) {
        std::string type = s["type"].get<std::string>();
        if (type == "mesh") {
            int mat_id = s["material"].get<int>();
            std::string file = scene_dir + s["file"].get<std::string>();

            MeshTransform xf{};
            if (s.contains("translate")) xf.translate = parse_vec3(s["translate"]);
            if (s.contains("scale"))     xf.scale     = s["scale"].get<float>();
            if (s.contains("rotate_y"))  xf.rotate_y  = s["rotate_y"].get<float>();

            int added = load_obj(file, mat_id, xf, scene.triangles);
            if (added < 0)
                throw std::runtime_error("Failed to load mesh: " + file);
            std::printf("  Loaded mesh '%s': %d triangles\n", file.c_str(), added);
        } else {
            scene.shapes.push_back(parse_shape(s));
        }
    }

    if (j.contains("lights"))
        for (const auto& l : j["lights"])
            scene.lights.push_back(parse_light(l));

    // Validate material ids for analytic shapes
    for (size_t i = 0; i < scene.shapes.size(); ++i) {
        int mid = scene.shapes[i].material_id;
        if (mid < 0 || mid >= int(scene.materials.size()))
            throw std::runtime_error("Shape " + std::to_string(i) +
                                     " references invalid material id " + std::to_string(mid));
    }

    // Build BVH over all primitives (analytic shapes + triangles)
    build_bvh(scene);

    return scene;
}
