#pragma once

#include "carat/gemma4_config.h"
#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include <string>

namespace carat {

struct Gemma4Model {
  Gemma4Config config;
  ShardedSafetensors weights;
  WeightPlan plan;

  static Gemma4Model open(const std::string &model_directory);
};

} // namespace carat
