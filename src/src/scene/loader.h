#pragma once
#include <string>
#include "scene.h"

// Parses a JSON scene description file and returns the populated SceneData.
// Throws std::runtime_error if the file cannot be opened or is malformed.
SceneData load_scene(const std::string& path);
