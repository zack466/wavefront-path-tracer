#include "mesh_loader.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>

static Vec3 apply_transform(Vec3 p, const MeshTransform& xf) {
    p = p * xf.scale;
    if (xf.rotate_y != 0.0f) {
        float rad = xf.rotate_y * 3.14159265f / 180.0f;
        float c = cosf(rad), s = sinf(rad);
        p = Vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
    }
    return p + xf.translate;
}
static Vec3 apply_transform_normal(Vec3 n, const MeshTransform& xf) {
    if (xf.rotate_y != 0.0f) {
        float rad = xf.rotate_y * 3.14159265f / 180.0f;
        float c = cosf(rad), s = sinf(rad);
        n = Vec3(c*n.x + s*n.z, n.y, -s*n.x + c*n.z);
    }
    return normalize(n);
}

// Parse "v", "v/vt", "v//vn", or "v/vt/vn" → 0-based indices (-1 = absent).
// Handles both positive (1-based) and negative (relative-to-end) OBJ indices.
static void parse_face_vertex(const std::string& tok, int& vi, int& ni,
                               int n_pos, int n_nrm) {
    vi = ni = -1;
    auto resolve = [](int raw, int n) -> int {
        return raw > 0 ? raw - 1 : n + raw;  // negative: count from end
    };
    size_t s1 = tok.find('/');
    if (s1 == std::string::npos) { vi = resolve(std::stoi(tok), n_pos); return; }
    vi = resolve(std::stoi(tok.substr(0, s1)), n_pos);
    size_t s2 = tok.find('/', s1 + 1);
    if (s2 == std::string::npos) { return; }  // v/vt
    if (s2 + 1 < tok.size()) ni = resolve(std::stoi(tok.substr(s2 + 1)), n_nrm);
}

// ── Per-face data (before fan-triangulation) ──────────────────────────────────
struct PolyVert { int vi, ni; };  // position index, normal index (-1 absent)
struct PolyFace { std::vector<PolyVert> verts; };

int load_obj(const std::string& path, int material_id,
             const MeshTransform& xform, std::vector<Triangle>& out)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "mesh_loader: cannot open '%s'\n", path.c_str());
        return -1;
    }

    std::vector<Vec3>     positions;
    std::vector<Vec3>     file_normals;
    std::vector<PolyFace> poly_faces;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z; ss >> x >> y >> z;
            positions.push_back({x, y, z});
        } else if (token == "vn") {
            float x, y, z; ss >> x >> y >> z;
            Vec3 raw{x, y, z};
            float len = sqrtf(raw.x*raw.x + raw.y*raw.y + raw.z*raw.z);
            file_normals.push_back(len > 1e-8f ? raw / len : Vec3{0.f, 1.f, 0.f});
        } else if (token == "f") {
            PolyFace face;
            std::string t;
            int n_pos = (int)positions.size(), n_nrm = (int)file_normals.size();
            while (ss >> t) {
                int vi, ni;
                parse_face_vertex(t, vi, ni, n_pos, n_nrm);
                if (vi < 0 || vi >= n_pos) goto next_face;
                face.verts.push_back({vi, ni});
            }
            if (face.verts.size() >= 3) poly_faces.push_back(std::move(face));
            next_face:;
        }
    }

    if (positions.empty() || poly_faces.empty()) return 0;

    // ── Smooth normal generation ──────────────────────────────────────────────
    // If OBJ has no normals, compute area-weighted smooth normals by averaging
    // the cross-product normals of all incident faces at each vertex.
    bool has_file_normals = !file_normals.empty();
    std::vector<Vec3> smooth_normals;

    if (!has_file_normals) {
        smooth_normals.assign(positions.size(), Vec3(0.f));
        for (const auto& poly : poly_faces) {
            // Fan from vert 0
            for (size_t i = 1; i + 1 < poly.verts.size(); ++i) {
                Vec3 v0 = positions[poly.verts[0].vi];
                Vec3 v1 = positions[poly.verts[i].vi];
                Vec3 v2 = positions[poly.verts[i+1].vi];
                Vec3 fn = cross(v1 - v0, v2 - v0);  // magnitude = 2 * face area (correct weight)
                smooth_normals[poly.verts[0].vi]   += fn;
                smooth_normals[poly.verts[i].vi]   += fn;
                smooth_normals[poly.verts[i+1].vi] += fn;
            }
        }
        for (auto& n : smooth_normals) {
            float len = length(n);
            n = (len > 1e-8f) ? n / len : Vec3(0.f, 1.f, 0.f);
        }
    }

    // ── Build triangles ───────────────────────────────────────────────────────
    int added = 0;
    for (const auto& poly : poly_faces) {
        // Fan-triangulate polygon
        for (size_t i = 1; i + 1 < poly.verts.size(); ++i) {
            const PolyVert* pv[3] = { &poly.verts[0], &poly.verts[i], &poly.verts[i+1] };
            Triangle tri;
            tri.material_id    = material_id;
            tri.smooth_normals = true;  // always use vertex normals (smooth or from file)
            for (int k = 0; k < 3; ++k) {
                tri.v[k] = apply_transform(positions[pv[k]->vi], xform);
                Vec3 n;
                if (has_file_normals && pv[k]->ni >= 0 && pv[k]->ni < (int)file_normals.size())
                    n = file_normals[pv[k]->ni];
                else
                    n = smooth_normals[pv[k]->vi];
                tri.n[k] = apply_transform_normal(n, xform);
            }
            out.push_back(tri);
            ++added;
        }
    }
    return added;
}
